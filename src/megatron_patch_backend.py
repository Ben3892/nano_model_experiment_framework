"""Adapter for the upstream Pai-Megatron-Patch Qwen3 entry point."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import torch

from .experiment_data import CHAT_TEMPLATE, EOS, SPECIAL_TOKENS, VOCAB_SIZE, byte_tokens

CORE = Path("backends/megatron/Megatron-LM-250624")
ENTRYPOINT = Path("examples/qwen3/pretrain_qwen.py")
TRACKER = "latest_checkpointed_iteration.txt"
TRAIN_LINE = re.compile(
    r"\biteration\s+(\d+)\s*/\s*\d+.*?"
    r"number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)"
)
TRAIN_LOSS = re.compile(r"\blm loss:\s*(\S+)")
VALID_LINE = re.compile(r"validation loss at iteration (\d+) on validation set.*?lm loss value:\s*(\S+)")


def check_environment(args):
    root = args.patch_root
    if not (root / ENTRYPOINT).is_file() or not (root / CORE / "megatron/training/training.py").is_file():
        raise ValueError(f"expected {ENTRYPOINT} and initialized {CORE} under {root}")
    if args.init_from and not (args.init_from / TRACKER).is_file():
        raise ValueError(f"missing Megatron checkpoint tracker: {args.init_from / TRACKER}")
    if not args.prepare_only and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        raise RuntimeError("Megatron-Patch MVP requires a BF16-capable NVIDIA GPU")


def write_tokenizer(directory):
    directory.mkdir()
    # Qwen2 slow BPE uses the GPT-2 byte alphabet. With zero merges every UTF-8
    # byte retains its original ID, so both backends use the same corpus tokens.
    alphabet = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = [byte for byte in range(256) if byte not in alphabet]
    vocab = {chr(byte): byte for byte in alphabet}
    vocab.update({chr(256 + index): byte for index, byte in enumerate(extra)})
    vocab.update(SPECIAL_TOKENS)
    (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "chat_template": CHAT_TEMPLATE,
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")


def write_indexed_data(args, rows):
    # Use the index writer from the same submodule as the training entry point.
    sys.path.insert(0, str(args.patch_root / CORE))
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
        import numpy as np

        for split, documents in rows.items():
            prefix = args.output / split
            builder = IndexedDatasetBuilder(str(prefix.with_suffix(".bin")), dtype=np.uint16)
            for text in documents:
                tokens = torch.tensor(byte_tokens(text) + [EOS])
                builder.add_item(tokens)
                builder.end_document()
            builder.finalize(str(prefix.with_suffix(".idx")))
    finally:
        sys.path.pop(0)


def write_sft_data(directory, rows):
    directory.mkdir()
    for split, examples in rows.items():
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for row in examples:
                example = {"query": row["prompt"], "response": row["response"]}
                file.write(json.dumps(example, ensure_ascii=False) + "\n")


def build_command(args, tokenizer, checkpoint):
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
        str(args.patch_root / ENTRYPOINT),
        "--num-layers", "2", "--hidden-size", "128", "--num-attention-heads", "4",
        "--ffn-hidden-size", "256", "--kv-channels", "32",
        "--seq-length", str(args.seq_len), "--max-position-embeddings", str(args.seq_len),
        "--micro-batch-size", str(args.batch_size), "--global-batch-size", str(args.batch_size),
        "--train-iters", str(args.steps), "--eval-iters", str(args.eval_batches),
        "--eval-interval", str(args.steps + 1), "--log-interval", "1", "--num-workers", "0",
        "--lr", str(args.lr), "--lr-decay-style", "constant", "--seed", str(args.seed),
        "--weight-decay", "0.01", "--clip-grad", "1.0",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--tokenizer-type", "NullTokenizer", "--vocab-size", str(VOCAB_SIZE),
        "--patch-tokenizer-type", "Qwen3Tokenizer", "--padded-vocab-size", "384",
        "--transformer-impl", "transformer_engine", "--attention-backend", "unfused", "--bf16",
        "--swiglu", "--normalization", "RMSNorm", "--norm-epsilon", "1e-6", "--qk-layernorm",
        "--position-embedding-type", "rope", "--rotary-base", "1000000", "--disable-bias-linear",
        "--no-rope-fusion", "--no-masked-softmax-fusion", "--no-gradient-accumulation-fusion",
        "--load", str(tokenizer), "--save", str(checkpoint),
        "--save-interval", str(args.steps), "--ckpt-format", "torch_dist",
    ]
    if args.stage == "pretrain":
        command += [
            "--train-mode", "pretrain", "--dataset", "MMAP",
            "--train-data-path", str(args.output / "train"), "--valid-data-path", str(args.output / "valid"),
            "--data-cache-path", str(args.output / "dataset-cache"),
        ]
    else:
        command += [
            "--train-mode", "finetune", "--dataset", "JSON-SFT",
            "--train-data-path", str(args.output / "patch-data" / "train.jsonl"),
            "--valid-data-path", str(args.output / "patch-data" / "valid.jsonl"),
            "--max-padding-length", str(args.seq_len), "--dataloader-type", "cyclic",
            "--eod-mask-loss", "--calculate-per-token-loss",
        ]
    if args.init_from:
        command += ["--pretrained-checkpoint", str(args.init_from), "--finetune"]
    return command


def run_patch(args, rows, record):
    tokenizer = args.output / "tokenizer"
    checkpoint = args.output / "checkpoint"
    write_tokenizer(tokenizer)
    if args.stage == "pretrain":
        write_indexed_data(args, rows)
    else:
        write_sft_data(args.output / "patch-data", rows)
    command = build_command(args, tokenizer, checkpoint)
    paths = [str(args.patch_root), str(args.patch_root / CORE)]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    environment = {
        "PYTHONPATH": os.pathsep.join(paths),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "NVTE_FLASH_ATTN": "0",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "1",
    }
    (args.output / "launch.json").write_text(
        json.dumps({"argv": command, "env": environment}, indent=2) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        print(shlex.join(command))
        return {"status": "prepared"}

    logfile = args.output / "megatron.log"
    train_steps = []
    valid_loss = None
    with logfile.open("w", encoding="utf-8") as log, subprocess.Popen(
        command, env={**os.environ, **environment}, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    ) as process:
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                match = TRAIN_LINE.search(line)
                if match:
                    step, skipped, nan = match.groups()
                    if int(skipped) or int(nan):
                        raise RuntimeError(f"Megatron reported a skipped or non-finite step {step}")
                    loss = TRAIN_LOSS.search(line)
                    if not loss:
                        raise RuntimeError(f"Megatron did not log a loss at step {step}")
                    train_steps.append(int(step))
                    record(int(step), "train", float(loss[1]))
                match = VALID_LINE.search(line)
                if match and int(match[1]) == args.steps:
                    valid_loss = float(match[2])
                    record(args.steps, "valid", valid_loss)
        except BaseException:
            process.terminate()
            raise
        if process.wait():
            raise RuntimeError(f"Megatron exited with code {process.returncode}; see {logfile}")

    if train_steps != list(range(1, args.steps + 1)) or valid_loss is None:
        raise RuntimeError(f"missing training steps or final validation loss; see {logfile}")
    tracker = checkpoint / TRACKER
    checkpoint_step = checkpoint / f"iter_{args.steps:07d}"
    if (
        not tracker.is_file()
        or tracker.read_text(encoding="utf-8").strip() != str(args.steps)
        or not (checkpoint_step / "common.pt").is_file()
        or not (checkpoint_step / ".metadata").is_file()
        or not any(checkpoint_step.glob("*.distcp"))
    ):
        raise RuntimeError(f"Megatron did not record a final checkpoint; see {logfile}")
    return {"steps": args.steps, "valid_loss": valid_loss, "checkpoint": str(checkpoint)}
