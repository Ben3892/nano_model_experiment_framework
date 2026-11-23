"""Run a bounded verifier-guided recursive SFT experiment."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

from ..megatron_patch_backend import check_gpus, check_patch_root
from ..train import load_torch_checkpoint, main as train_main, write_json
from .data import as_sft, load_splits, make_demo, select_sft, write_jsonl
from .evaluate import metrics, promotion, question_scores, sample
from .megatron_rollout import PatchCheckpoint, sample_patch
from .planner import MAX_REPLAY, MIN_REPLAY, dsh_plan, resolve_dsh


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="use the toy math fixture and bootstrap a seed checkpoint")
    source.add_argument("--data-dir", type=Path, help="directory with rollout/replay/monitor/dev/blind JSONL splits")
    parser.add_argument("--init-from", type=Path, help="Torch checkpoint file or Patch checkpoint directory")
    parser.add_argument("--backend", choices=("torch", "megatron-patch"), default="torch")
    parser.add_argument("--patch-root", type=Path, help="Pai-Megatron-Patch checkout")
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
    if args.backend == "megatron-patch":
        if args.demo or not args.patch_root:
            parser.error("Patch RSI requires --patch-root, --data-dir and --init-from")
        args.patch_root = args.patch_root.resolve()
    elif args.patch_root:
        parser.error("--patch-root requires --backend megatron-patch")
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
    if args.backend == "megatron-patch":
        parent.check_tracker()
        command += [
            "--backend", "megatron-patch", "--patch-root", str(args.patch_root),
            "--init-iteration", str(parent.iteration),
        ]
    if parent:
        command += ["--init-from", str(checkpoint_path(parent))]
    with (directory / "training.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            status = train_main(command)
    if status:
        raise RuntimeError(f"SFT failed in {directory.name}; see {directory / 'training.log'}")
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    checkpoint = Path(result["checkpoint"])
    if args.backend == "megatron-patch":
        return PatchCheckpoint.read(checkpoint, args.seq_len), result.get("supervised_tokens")
    if not checkpoint.is_file():
        raise RuntimeError(f"missing Torch checkpoint: {checkpoint}")
    return checkpoint, result["supervised_tokens"]


def checkpoint_path(checkpoint):
    return checkpoint.root if isinstance(checkpoint, PatchCheckpoint) else checkpoint


def sample_checkpoint(checkpoint, questions, count, args, seed, path):
    if isinstance(checkpoint, PatchCheckpoint):
        return sample_patch(checkpoint, questions, count, args, seed, path)
    return sample(load_torch_checkpoint(checkpoint), questions, count, args, seed)


def score(checkpoint, questions, args, seed, path):
    rows = sample_checkpoint(checkpoint, questions, args.eval_samples, args, seed, path)
    scores = question_scores(questions, rows, args.eval_samples)
    write_jsonl(path, scores)
    return scores


def save_best(root, checkpoint, round_number, scores):
    result = {"checkpoint": str(checkpoint_path(checkpoint)), "round": round_number, "dev": metrics(scores)}
    if isinstance(checkpoint, PatchCheckpoint):
        result["checkpoint_metadata"] = checkpoint.metadata()
    temp = root / ".best.tmp"
    write_json(temp, result)
    temp.replace(root / "best.json")


def run(args):
    torch.set_num_threads(1)
    if args.backend == "megatron-patch":
        check_patch_root(args.patch_root)
        check_gpus(1)
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
    elif args.backend == "megatron-patch":
        origin = PatchCheckpoint.read(args.init_from, args.seq_len)
    else:
        origin = root / "initial_checkpoint.pt"
        shutil.copyfile(args.init_from, origin)
    if isinstance(origin, PatchCheckpoint):
        initial_metadata = origin.metadata()
    else:
        model = load_torch_checkpoint(origin)
        if model.config.max_seq_len != args.seq_len:
            raise ValueError("--seq-len must match the initial checkpoint")
        with origin.open("rb") as checkpoint_file:
            digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        initial_metadata = {"path": str(origin), "sha256": digest}
    write_json(root / "initial_checkpoint.json", initial_metadata)
    # All candidates use the same per-question random seeds on dev.
    eval_seed = args.seed + 10_000
    best_scores = score(origin, splits["dev"], args, eval_seed, root / "dev_initial.jsonl")
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
        rollout_path = directory / "rollouts.jsonl"
        rollouts = sample_checkpoint(generator, splits["rollout"], args.samples, args,
                                     args.seed + number, rollout_path)
        generated_used += rollout_cost
        write_jsonl(rollout_path, rollouts)
        rows, data_report = select_sft(
            splits["rollout"], rollouts, splits["replay"], replay, args.seed + number,
            args.max_fresh_supervised_tokens,
        )
        outcome = {
            "round": number, "parent": str(checkpoint_path(best)),
            "generator": str(checkpoint_path(generator)),
            "rollout_seed": args.seed + number, "data": data_report,
        }
        if isinstance(best, PatchCheckpoint):
            outcome.update(parent_iteration=best.iteration, generator_iteration=generator.iteration)
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
        supervised_used = (
            supervised_used + supervised if supervised_used is not None and supervised is not None else None
        )
        candidate_scores = score(candidate, splits["dev"], args, eval_seed, directory / "dev.jsonl")
        generated_used += dev_cost
        result = promotion(best_scores, candidate_scores, args.min_gain, args.max_regression, eval_seed)
        outcome.update(result, checkpoint=str(checkpoint_path(candidate)),
                       status="evaluated", supervised_tokens=supervised)
        if isinstance(candidate, PatchCheckpoint):
            outcome["checkpoint_metadata"] = candidate.metadata()
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

    blind_initial = score(origin, splits["blind"], args, eval_seed + 1,
                          root / "blind_initial.jsonl")
    generated_used += blind_cost
    blind_best = blind_initial
    if best != origin:
        blind_best = score(best, splits["blind"], args, eval_seed + 1,
                           root / "blind_best.jsonl")
        generated_used += blind_cost
    result = {
        "stop_reason": reason, "best_checkpoint": str(checkpoint_path(best)),
        "initial_dev": initial, "best_dev": metrics(best_scores),
        "blind_initial": metrics(blind_initial), "blind_best": metrics(blind_best),
        "generation_token_slots": generated_used, "train_token_slots": train_used,
        "supervised_tokens": supervised_used, "rounds": outcomes,
    }
    if isinstance(best, PatchCheckpoint):
        result["best_checkpoint_metadata"] = best.metadata()
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
