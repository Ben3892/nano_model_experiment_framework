"""Run a bounded verifier-guided recursive SFT experiment on CPU."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

from ..train import load_torch_checkpoint, main as train_main, write_json
from .data import as_sft, load_splits, make_demo, select_sft, write_jsonl
from .evaluate import metrics, promotion, question_scores, sample
from .planner import MAX_REPLAY, MIN_REPLAY, dsh_plan, resolve_dsh


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="use the toy math fixture and bootstrap a seed checkpoint")
    source.add_argument("--data-dir", type=Path, help="directory with rollout/replay/monitor/dev/blind JSONL splits")
    parser.add_argument("--init-from", type=Path, help="Torch pretrain or SFT checkpoint for custom data")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generator-mode", choices=("recursive", "frozen"), default="recursive")
    parser.add_argument("--planner", choices=("fixed", "dsh"), default="fixed")
    parser.add_argument("--dsh-bin", default="dsh")
    parser.add_argument("--dsh-timeout", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--demo-bootstrap-steps", type=int, default=35)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples", type=int, default=8, help="rollouts per train question")
    parser.add_argument("--eval-samples", type=int, default=8, help="samples per evaluation question")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--min-positive-prompts", type=int, default=4)
    parser.add_argument("--min-fresh-supervised-tokens", type=int, default=8)
    parser.add_argument("--max-fresh-supervised-tokens", type=int, default=100_000)
    parser.add_argument("--replay-fraction", type=float, default=0.2)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--max-regression", type=float, default=0.02, help="maximum pass@1 loss for any dev family")
    parser.add_argument("--max-generated-tokens", type=int, default=200_000)
    parser.add_argument("--max-train-tokens", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.demo == bool(args.init_from):
        parser.error("provide --init-from with --data-dir, or use --demo on its own")
    for key in ("dsh_timeout", "rounds", "patience", "steps", "demo_bootstrap_steps", "seq_len", "batch_size",
                "samples", "eval_samples", "top_k", "max_new_tokens", "min_positive_prompts",
                "min_fresh_supervised_tokens", "max_fresh_supervised_tokens", "max_generated_tokens",
                "max_train_tokens", "seed"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 0 < args.lr < float("inf") or not 0 < args.temperature < float("inf"):
        parser.error("--lr and --temperature must be finite and positive")
    if not MIN_REPLAY <= args.replay_fraction <= MAX_REPLAY:
        parser.error(f"--replay-fraction must be in [{MIN_REPLAY}, {MAX_REPLAY}]")
    if not 0 <= args.min_gain <= 1 or not 0 <= args.max_regression <= 1:
        parser.error("--min-gain and --max-regression must be in [0, 1]")
    if args.min_fresh_supervised_tokens > args.max_fresh_supervised_tokens:
        parser.error("minimum fresh supervised tokens exceed the maximum")
    return args


def fit(args, directory, rows, monitor, steps, seed, parent=None):
    train_file, valid_file = directory / "sft_train.jsonl", directory / "sft_monitor.jsonl"
    write_jsonl(train_file, rows)
    write_jsonl(valid_file, map(as_sft, monitor))
    output = directory / "training"
    command = [
        "--stage", "sft", "--output", str(output), "--steps", str(steps),
        "--seq-len", str(args.seq_len), "--batch-size", str(args.batch_size), "--seed", str(seed),
        "--lr", str(args.lr), "--train-data", str(train_file), "--valid-data", str(valid_file),
    ]
    if parent:
        command += ["--init-from", str(parent)]
    with (directory / "training.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            status = train_main(command)
    checkpoint = output / "checkpoint.pt"
    if status or not checkpoint.is_file():
        raise RuntimeError(f"SFT failed in {directory.name}; see {directory / 'training.log'}")
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    return checkpoint, result["supervised_tokens"]


def score(model, questions, args, seed, path):
    rows = sample(model, questions, args.eval_samples, args, seed)
    scores = question_scores(questions, rows, args.eval_samples)
    write_jsonl(path, scores)
    return scores


def save_best(root, checkpoint, round_number, scores):
    result = {"checkpoint": str(checkpoint), "round": round_number, "dev": metrics(scores)}
    temp = root / ".best.tmp"
    write_json(temp, result)
    temp.replace(root / "best.json")


def run(args):
    torch.set_num_threads(1)
    root = args.output.resolve()
    executable = resolve_dsh(args.dsh_bin) if args.planner == "dsh" else None
    root.mkdir(parents=True, exist_ok=False)
    data_dir = root / "demo_data" if args.demo else args.data_dir
    if args.demo:
        make_demo(data_dir)
    splits, hashes = load_splits(data_dir, args.seq_len, args.max_new_tokens)
    if len(splits["monitor"]) < args.batch_size or (args.demo and len(splits["replay"]) < args.batch_size):
        raise ValueError("monitor and demo replay must provide at least one full training batch")

    dev_cost = len(splits["dev"]) * args.eval_samples * args.max_new_tokens
    blind_cost = len(splits["blind"]) * args.eval_samples * args.max_new_tokens
    rollout_cost = len(splits["rollout"]) * args.samples * args.max_new_tokens
    step_cost = args.seq_len * args.batch_size
    train_used = args.demo_bootstrap_steps * step_cost if args.demo else 0
    if train_used > args.max_train_tokens or dev_cost + 2 * blind_cost > args.max_generated_tokens:
        raise ValueError("budget does not cover baseline evaluation, reserved blind evaluation or demo bootstrap")
    manifest = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(root / "run.json", {"config": manifest, "question_split_sha256": hashes})
    supervised_used = 0
    if args.demo:
        directory = root / "seed"
        directory.mkdir()
        origin, supervised_used = fit(
            args, directory, list(map(as_sft, splits["replay"])),
            splits["monitor"], args.demo_bootstrap_steps, args.seed,
        )
    else:
        origin = root / "initial_checkpoint.pt"
        shutil.copyfile(args.init_from, origin)
    model = load_torch_checkpoint(origin)
    if model.config.max_seq_len != args.seq_len:
        raise ValueError("--seq-len must match the initial checkpoint")
    with origin.open("rb") as checkpoint_file:
        digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    write_json(root / "initial_checkpoint.json", {
        "path": str(origin), "sha256": digest,
    })
    # All candidates use the same per-question random seeds on dev.
    eval_seed = args.seed + 10_000
    best_scores = score(model, splits["dev"], args, eval_seed, root / "dev_initial.jsonl")
    initial = metrics(best_scores)
    best = origin
    save_best(root, best, 0, best_scores)
    generated_used = dev_cost
    replay, misses = args.replay_fraction, 0
    outcomes = []
    reason = "round_limit"

    for number in range(1, args.rounds + 1):
        if generated_used + rollout_cost + dev_cost + 2 * blind_cost > args.max_generated_tokens:
            reason = "generation_budget"
            break
        if train_used + args.steps * step_cost > args.max_train_tokens:
            reason = "train_budget"
            break
        directory = root / f"round_{number:03d}"
        directory.mkdir()
        generator = best if args.generator_mode == "recursive" else origin
        rollouts = sample(load_torch_checkpoint(generator), splits["rollout"], args.samples, args,
                          args.seed + number)
        generated_used += rollout_cost
        write_jsonl(directory / "rollouts.jsonl", rollouts)
        rows, data_report = select_sft(
            splits["rollout"], rollouts, splits["replay"], replay, args.seed + number,
            args.max_fresh_supervised_tokens,
        )
        outcome = {
            "round": number, "parent": str(best), "generator": str(generator),
            "rollout_seed": args.seed + number, "data": data_report,
        }
        if (
            data_report["positive_prompts"] < args.min_positive_prompts
            or data_report["fresh_supervised_tokens"] < args.min_fresh_supervised_tokens
            or len(rows) < args.batch_size
        ):
            reason = "insufficient_positive_coverage"
            outcome.update(promoted=False, status=reason)
            write_json(directory / "decision.json", outcome)
            outcomes.append(outcome)
            break

        candidate, supervised = fit(
            args, directory, rows, splits["monitor"], args.steps, args.seed + number, parent=best,
        )
        train_used += args.steps * step_cost
        supervised_used += supervised
        candidate_scores = score(load_torch_checkpoint(candidate), splits["dev"], args, eval_seed,
                                 directory / "dev.jsonl")
        generated_used += dev_cost
        result = promotion(best_scores, candidate_scores, args.min_gain, args.max_regression, eval_seed)
        outcome.update(result, checkpoint=str(candidate), status="evaluated", supervised_tokens=supervised)
        write_json(directory / "decision.json", outcome)
        outcomes.append(outcome)
        print(f"round {number}: pass@1 change {result['gain']:+.4f}, promoted={result['promoted']}", flush=True)
        if result["promoted"]:
            save_best(root, candidate, number, candidate_scores)
            best, best_scores, misses = candidate, candidate_scores, 0
        else:
            misses += 1
            if misses >= args.patience:
                reason = "patience_exhausted"
                break

        if (
            executable and number < args.rounds
            and generated_used + rollout_cost + dev_cost + 2 * blind_cost <= args.max_generated_tokens
            and train_used + args.steps * step_cost <= args.max_train_tokens
        ):
            report = {
                "round": number, "gain": result["gain"], "paired_ci95": result["paired_ci95"],
                "promoted": result["promoted"], "prior": result["prior"], "candidate": result["candidate"],
                "data": data_report,
            }
            replay = dsh_plan(executable, args.dsh_timeout, report, directory, replay)

    blind_initial = score(load_torch_checkpoint(origin), splits["blind"], args, eval_seed + 1,
                          root / "blind_initial.jsonl")
    generated_used += blind_cost
    blind_best = blind_initial
    if best != origin:
        blind_best = score(load_torch_checkpoint(best), splits["blind"], args, eval_seed + 1,
                           root / "blind_best.jsonl")
        generated_used += blind_cost
    result = {
        "stop_reason": reason, "best_checkpoint": str(best), "initial_dev": initial, "best_dev": metrics(best_scores),
        "blind_initial": metrics(blind_initial), "blind_best": metrics(blind_best),
        "generation_token_slots": generated_used, "train_token_slots": train_used,
        "supervised_tokens": supervised_used, "rounds": outcomes,
    }
    write_json(root / "summary.json", result)
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"RSI error: {error}", file=sys.stderr)
        return 1
    print(f"RSI stopped: {result['stop_reason']}; best checkpoint: {result['best_checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
