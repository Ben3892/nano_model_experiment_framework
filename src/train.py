"""Run a small pretraining or supervised fine-tuning experiment."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from .experiment_data import EOS, IGNORE, VOCAB_SIZE, byte_tokens, load_rows, sft_example, validate_rows
from .models.dense_model import DenseGPT, DenseGPTConfig


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_torch_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["stage"] not in ("pretrain", "sft"):
        raise ValueError("expected a Torch pretrain or SFT checkpoint")
    model = DenseGPT(DenseGPTConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    return model


def torch_sampler(stage, rows, seq_len, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    if stage == "sft":
        inputs, labels = zip(*(sft_example(row, seq_len) for row in rows))
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)

        def sample():
            indices = torch.randint(len(inputs), (batch_size,), generator=generator)
            return inputs[indices], labels[indices]
    else:
        stream = torch.tensor([token for row in rows for token in byte_tokens(row) + [EOS]])
        offsets = torch.arange(seq_len + 1)

        def sample():
            starts = torch.randint(len(stream) - seq_len, (batch_size, 1), generator=generator)
            batch = stream[starts + offsets]
            return batch[:, :-1], batch[:, 1:]
    return sample


def run_torch(args, rows, record):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = DenseGPTConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=args.seq_len,
        num_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
    )
    if args.init_from:
        model = load_torch_checkpoint(args.init_from)
        if model.config != config:
            raise ValueError("SFT needs a Torch checkpoint with matching model config and seq_len")
    else:
        model = DenseGPT(config)
    # Loading builds a temporary model before restoring its weights. Keep the
    # training RNG tied to the run seed for both initialization paths.
    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train = torch_sampler(args.stage, rows["train"], args.seq_len, args.batch_size, args.seed)
    valid = torch_sampler(args.stage, rows["valid"], args.seq_len, args.batch_size, args.seed + 1)
    validation = [valid() for _ in range(args.eval_batches)]

    def loss(batch):
        inputs, labels = batch
        per_token = F.cross_entropy(model(inputs).transpose(1, 2), labels, reduction="none")
        return (per_token.sum(dim=1) / labels.ne(IGNORE).sum(dim=1)).mean()

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        value = sum(loss(batch).item() for batch in validation) / len(validation)
        record(step, "valid", value)
        model.train()
        return value

    initial = evaluate(0)
    supervised_tokens = 0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = train()
        supervised_tokens += batch[1].ne(IGNORE).sum().item()
        train_loss = loss(batch)
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        record(step, "train", train_loss.item())
    final = evaluate(args.steps)
    checkpoint = args.output / "checkpoint.pt"
    torch.save(
        {"stage": args.stage, "step": args.steps, "model_config": vars(config), "model": model.state_dict()},
        checkpoint,
    )
    return {
        "steps": args.steps,
        "parameters": model.num_parameters(),
        "train_token_slots": args.steps * args.batch_size * args.seq_len,
        "supervised_tokens": supervised_tokens,
        "initial_valid_loss": initial,
        "valid_loss": final,
        "checkpoint": str(checkpoint),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pretrain", "sft"), default="pretrain")
    parser.add_argument("--backend", choices=("torch", "megatron-patch"), default="torch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4, help="global batch size")
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--valid-data", type=Path)
    parser.add_argument("--init-from", type=Path, help="pretrain or SFT checkpoint from the same backend")
    parser.add_argument("--patch-root", type=Path, help="Pai-Megatron-Patch checkout")
    parser.add_argument("--nproc-per-node", type=int, help="single-node Patch data-parallel processes")
    parser.add_argument("--init-iteration", type=int, help="expected Patch checkpoint tracker iteration")
    parser.add_argument("--prepare-only", action="store_true", help="prepare Patch inputs without a GPU")
    args = parser.parse_args(argv)

    for key in ("steps", "seq_len", "batch_size", "eval_batches", "seed"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if bool(args.train_data) != bool(args.valid_data):
        parser.error("provide --train-data and --valid-data together")
    if args.init_from and args.stage != "sft":
        parser.error("--init-from is supported for SFT")
    if args.backend == "megatron-patch" and not args.patch_root:
        parser.error("--backend megatron-patch requires --patch-root")
    if args.backend == "torch" and (
        args.patch_root or args.prepare_only or args.nproc_per_node is not None or args.init_iteration is not None
    ):
        parser.error("Patch-specific options require --backend megatron-patch")
    if args.backend == "megatron-patch":
        if args.nproc_per_node is None:
            args.nproc_per_node = 1
        if args.nproc_per_node < 1 or args.batch_size % args.nproc_per_node:
            parser.error("--nproc-per-node must be positive and divide --batch-size")
        if args.init_iteration is not None and (args.init_iteration < 1 or not args.init_from):
            parser.error("--init-iteration must be positive and requires --init-from")

    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.train_data and args.train_data == args.valid_data:
        parser.error("train and validation must use different input files")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.backend == "megatron-patch":
            from .megatron_patch_backend import check_environment, run_patch
            check_environment(args)
        elif args.init_from and not args.init_from.is_file():
            raise ValueError(f"missing Torch checkpoint: {args.init_from}")
        rows = {
            split: load_rows(args.stage, split, path)
            for split, path in (("train", args.train_data), ("valid", args.valid_data))
        }
        for split, data in rows.items():
            validate_rows(args.stage, data, args.seq_len, args.batch_size, split)

        args.output.mkdir(parents=True, exist_ok=False)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        write_json(args.output / "config.json", config)
        for split, data in rows.items():
            suffix = ".txt" if args.stage == "pretrain" else ".jsonl"
            lines = data if args.stage == "pretrain" else [json.dumps(row, ensure_ascii=False) for row in data]
            (args.output / (split + suffix)).write_text("\n".join(lines) + "\n", encoding="utf-8")

        with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
            def record(step, split, loss):
                if not math.isfinite(loss):
                    raise RuntimeError(f"non-finite {split} loss at step {step}")
                row = dict(step=step, split=split, loss=loss)
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                if args.backend == "torch":
                    print(f"{split:5s} step {step:3d} loss {loss:.6f}", flush=True)

            run = run_torch if args.backend == "torch" else run_patch
            summary = run(args, rows, record)
        write_json(args.output / "summary.json", summary)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"Training error: {error}", file=sys.stderr)
        return 1
    print(f"Run output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
