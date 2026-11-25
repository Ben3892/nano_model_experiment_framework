"""Adapter for the upstream Pai-Megatron-Patch Qwen3 entry point."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unicodedata

import torch

from .experiment_data import CHAT_TEMPLATE, EOS, SPECIAL_TOKENS, VOCAB_SIZE, byte_tokens

CORE = Path("backends/megatron/Megatron-LM-250624")
ENTRYPOINT = Path("examples/qwen3/pretrain_qwen.py")
TRACKER = "latest_checkpointed_iteration.txt"
PADDED_VOCAB_SIZE = 384
TRAIN_LINE = re.compile(
    r"\biteration\s+(\d+)\s*/\s*\d+.*?"
    r"number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)"
)
TRAIN_LOSS = re.compile(r"\blm loss:\s*(\S+)")
VALID_LINE = re.compile(r"validation loss at iteration (\d+) on validation set.*?lm loss value:\s*(\S+)")


def checkpoint_files(root, expected_iteration=None):
    tracker = root / TRACKER
    if not tracker.is_file():
        raise ValueError(f"missing Megatron checkpoint tracker: {tracker}")
    value = tracker.read_text(encoding="utf-8").strip()
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"expected a positive Megatron iteration in {tracker}, got {value!r}")
    iteration = int(value)
    if expected_iteration is not None and iteration != expected_iteration:
        raise ValueError(f"checkpoint tracker at {root} changed from {expected_iteration} to {iteration}")
    step_dir = root / f"iter_{iteration:07d}"
    shards = sorted(step_dir.glob("*.distcp"))
    files = [step_dir / "common.pt", step_dir / ".metadata", *shards]
    if not shards or any(not path.is_file() or not path.stat().st_size for path in files):
        raise ValueError(f"incomplete torch_dist checkpoint: {step_dir}")
    # Megatron uses metadata.json to distinguish torch_dist from torch_dcp.
    manifest = step_dir / "metadata.json"
    if not manifest.is_file():
        raise ValueError(f"missing torch_dist format manifest: {manifest}")
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid torch_dist format manifest: {manifest}: {error}") from error
    # Match CheckpointingConfig defaults and the strategies in Megatron-LM-250624.
    supported = {
        "sharded_backend": "torch_dist", "sharded_backend_version": 1,
        "common_backend": "torch", "common_backend_version": 1,
    }
    if not isinstance(metadata, dict) or "sharded_backend" not in metadata:
        raise ValueError(f"missing sharded_backend in {manifest}")
    for name, value in metadata.items():
        if name not in supported or type(value) is not type(supported[name]) or value != supported[name]:
            raise ValueError(f"unsupported {name}={value!r} in {manifest}")
    return iteration, [*files, manifest]


def check_patch_root(root):
    if not (root / ENTRYPOINT).is_file() or not (root / CORE / "megatron/training/training.py").is_file():
        raise ValueError(f"expected {ENTRYPOINT} and initialized {CORE} under {root}")


def check_gpus(nproc):
    if not torch.cuda.is_available() or torch.cuda.device_count() < nproc:
        raise RuntimeError(f"Megatron-Patch requires {nproc} visible NVIDIA GPU(s)")
    for device in range(nproc):
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError(f"GPU {device} does not support native BF16")


def check_environment(args):
    check_patch_root(args.patch_root)
    if args.init_from:
        args.init_iteration, _ = checkpoint_files(args.init_from, args.init_iteration)
    if not args.prepare_only:
        check_gpus(args.nproc_per_node)


def tokenizer_files():
    # Qwen2 slow BPE uses the GPT-2 byte alphabet. With zero merges every UTF-8
    # byte retains its original ID, so both backends use the same corpus tokens.
    alphabet = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = [byte for byte in range(256) if byte not in alphabet]
    vocab = {chr(byte): byte for byte in alphabet}
    vocab.update({chr(256 + index): byte for index, byte in enumerate(extra)})
    vocab.update(SPECIAL_TOKENS)
    config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "chat_template": CHAT_TEMPLATE,
    }
    return {
        "vocab.json": json.dumps(vocab),
        "merges.txt": "#version: 0.2\n",
        "tokenizer_config.json": json.dumps(config),
    }


def write_tokenizer(directory):
    directory.mkdir()
    for name, contents in tokenizer_files().items():
        (directory / name).write_text(contents, encoding="utf-8")


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
                example = {
                    "query": unicodedata.normalize("NFC", row["prompt"]),
                    "response": unicodedata.normalize("NFC", row["response"]),
                }
                file.write(json.dumps(example, ensure_ascii=False) + "\n")


def model_arguments(seq_len, tokenizer):
    return [
        "--num-layers", "2", "--hidden-size", "128", "--num-attention-heads", "4",
        "--ffn-hidden-size", "256", "--kv-channels", "32",
        "--seq-length", str(seq_len), "--max-position-embeddings", str(seq_len),
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--tokenizer-type", "NullTokenizer", "--vocab-size", str(VOCAB_SIZE),
        "--patch-tokenizer-type", "Qwen3Tokenizer", "--padded-vocab-size", str(PADDED_VOCAB_SIZE),
        "--transformer-impl", "transformer_engine", "--attention-backend", "unfused", "--bf16",
        "--swiglu", "--normalization", "RMSNorm", "--norm-epsilon", "1e-6", "--qk-layernorm",
        "--position-embedding-type", "rope", "--rotary-base", "1000000", "--disable-bias-linear",
        "--no-rope-fusion", "--no-masked-softmax-fusion", "--no-gradient-accumulation-fusion",
        "--load", str(tokenizer), "--ckpt-format", "torch_dist",
    ]


def warmstart_arguments(root, iteration):
    return ["--pretrained-checkpoint", str(root), "--ckpt-step", str(iteration), "--finetune"]


def launch_prefix(nproc):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}"]


def build_command(args, tokenizer, checkpoint):
    command = [
        *launch_prefix(args.nproc_per_node), str(args.patch_root / ENTRYPOINT),
        *model_arguments(args.seq_len, tokenizer),
        "--micro-batch-size", str(args.batch_size // args.nproc_per_node),
        "--global-batch-size", str(args.batch_size),
        "--train-iters", str(args.steps), "--eval-iters", str(args.eval_batches),
        "--eval-interval", str(args.steps + 1), "--log-interval", "1", "--num-workers", "0",
        "--lr", str(args.lr), "--lr-decay-style", "constant", "--seed", str(args.seed),
        "--weight-decay", "0.01", "--clip-grad", "1.0",
        "--save", str(checkpoint), "--save-interval", str(args.steps),
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
        command += warmstart_arguments(args.init_from, args.init_iteration)
    return command


def launch_environment(patch_root):
    paths = [str(patch_root), str(patch_root / CORE)]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return {
        "PYTHONPATH": os.pathsep.join(paths),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "NVTE_FLASH_ATTN": "0",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "1",
    }


def check_warmstart(log, root, iteration):
    loading = f"loading distributed checkpoint from {root} at iteration {iteration}"
    loaded = f"successfully loaded checkpoint from {root} ["
    with log.open(encoding="utf-8", errors="replace") as file:
        saw_loading = saw_loaded = False
        for line in file:
            saw_loading |= line.strip() == loading
            saw_loaded |= loaded in line and line.rstrip().endswith("at iteration 0")
    if not (saw_loading and saw_loaded):
        raise RuntimeError(f"could not confirm warm start from {root} iteration {iteration}; see {log}")


def run_patch(args, rows, record):
    tokenizer = args.output / "tokenizer"
    checkpoint = args.output / "checkpoint"
    write_tokenizer(tokenizer)
    if args.stage == "pretrain":
        write_indexed_data(args, rows)
    else:
        write_sft_data(args.output / "patch-data", rows)
    command = build_command(args, tokenizer, checkpoint)
    environment = launch_environment(args.patch_root)
    (args.output / "launch.json").write_text(
        json.dumps({"argv": command, "env": environment}, indent=2) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        print(shlex.join(command))
        return {"status": "prepared"}

    logfile = args.output / "megatron.log"
    train_steps = []
    valid_loss = None
    expected_topology = False
    with logfile.open("w", encoding="utf-8") as log, subprocess.Popen(
        command, env={**os.environ, **environment}, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    ) as process:
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if f"using world size: {args.nproc_per_node}, data-parallel size: {args.nproc_per_node}," in line:
                    fields = {field.strip() for field in line.split(",")}
                    expected_topology = {
                        "context-parallel size: 1",
                        "tensor-model-parallel size: 1",
                        "pipeline-model-parallel size: 1",
                    }.issubset(fields)
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

    if not expected_topology:
        raise RuntimeError(f"Megatron did not log the expected DP topology; see {logfile}")
    if train_steps != list(range(1, args.steps + 1)) or valid_loss is None:
        raise RuntimeError(f"missing training steps or final validation loss; see {logfile}")
    try:
        checkpoint_files(checkpoint, args.steps)
    except ValueError as error:
        raise RuntimeError(f"Megatron did not record a final checkpoint: {error}; see {logfile}") from error
    if args.init_from:
        check_warmstart(logfile, args.init_from, args.init_iteration)
        checkpoint_files(args.init_from, args.init_iteration)
    return {"steps": args.steps, "valid_loss": valid_loss, "checkpoint": str(checkpoint)}
