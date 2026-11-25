"""Load an experiment torch_dist checkpoint and sample in a single-rank Patch worker."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unicodedata

import torch

from ..experiment_data import EOS, sft_prompt_tokens
from ..megatron_patch_backend import (
    PADDED_VOCAB_SIZE, checkpoint_files, check_warmstart, launch_environment, launch_prefix,
    model_arguments, tokenizer_files, warmstart_arguments,
)
from ..train import write_json
from .evaluate import sample


def command_options(argv):
    options = {}
    for index, option in enumerate(argv):
        if option.startswith("--"):
            following = argv[index + 1:index + 2]
            options[option] = following[0] if following and not following[0].startswith("--") else None
    return options


@dataclass(frozen=True)
class PatchCheckpoint:
    root: Path
    iteration: int
    tokenizer: Path
    seq_len: int
    sha256: dict[str, str]

    @classmethod
    def read(cls, root, seq_len, expected_iteration=None):
        root = root.resolve()
        iteration, files = checkpoint_files(root, expected_iteration)
        run = root.parent
        config_path = run / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("backend") != "megatron-patch" or config.get("seq_len") != seq_len:
            raise ValueError(f"expected a Patch run with --seq-len {seq_len}: {config_path}")
        tokenizer = run / "tokenizer"
        for name, contents in tokenizer_files().items():
            path = tokenizer / name
            if path.read_text(encoding="utf-8") != contents:
                raise ValueError(f"incompatible RSI byte tokenizer: {path}")
            files.append(path)
        launch_path = run / "launch.json"
        launch = json.loads(launch_path.read_text(encoding="utf-8"))["argv"]
        expected = command_options(model_arguments(seq_len, tokenizer))
        del expected["--load"]
        actual = command_options(launch)
        if any(name not in actual or actual[name] != value for name, value in expected.items()):
            raise ValueError(f"incompatible Patch model arguments: {launch_path}")
        files.extend((config_path, launch_path))
        hashes = {}
        for path in files:
            with path.open("rb") as stream:
                hashes[path.relative_to(run).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
        return cls(root, iteration, tokenizer, seq_len, hashes)

    def metadata(self):
        return {
            "backend": "megatron-patch", "path": str(self.root), "iteration": self.iteration,
            "tokenizer": str(self.tokenizer), "seq_len": self.seq_len,
            "file_sha256": self.sha256,
        }

    def check_tracker(self):
        checkpoint_files(self.root, self.iteration)


def sample_patch(checkpoint, questions, count, args, seed, path):
    checkpoint.check_tracker()
    request_path = path.with_suffix(".request.json")
    response_path = path.with_suffix(".response.json")
    log_path = path.with_suffix(".megatron.log")
    write_json(request_path, {
        "questions": questions, "count": count, "seed": seed,
        "max_new_tokens": args.max_new_tokens, "top_k": args.top_k, "temperature": args.temperature,
    })
    argv = [
        *launch_prefix(1), "--module", "src.rsi.megatron_rollout",
        *model_arguments(checkpoint.seq_len, checkpoint.tokenizer),
        *warmstart_arguments(checkpoint.root, checkpoint.iteration),
        "--micro-batch-size", "1", "--global-batch-size", "1",
        "--seed", str(args.seed), "--exit-on-missing-checkpoint",
        "--rsi-request", str(request_path), "--rsi-response", str(response_path),
    ]
    env = launch_environment(args.patch_root)
    write_json(path.with_suffix(".launch.json"), {"argv": argv, "env": env})
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(argv, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode or not response_path.is_file():
        raise RuntimeError(f"Megatron rollout failed; see {log_path}")
    check_warmstart(log_path, checkpoint.root, checkpoint.iteration)
    checkpoint.check_tracker()
    rows = json.loads(response_path.read_text(encoding="utf-8"))
    if [(row["id"], row["sample"]) for row in rows] != [
        (question["id"], index) for question in questions for index in range(count)
    ]:
        raise RuntimeError(f"incomplete or unordered Megatron rollout records: {response_path}")
    return rows


class MegatronGenerator:
    def __init__(self, model, device, seq_len):
        self.model = model
        self.device = device
        self.seq_len = seq_len

    def eval(self):
        self.model.eval()

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, temperature, top_k):
        if input_ids.size(1) + max_new_tokens > self.seq_len:
            raise ValueError("rollout exceeds the Megatron context length")
        tokens = input_ids.to(self.device)
        ended = torch.zeros(tokens.size(0), 1, dtype=torch.bool, device=self.device)
        for _ in range(max_new_tokens):
            length = tokens.size(1)
            positions = torch.arange(length, device=self.device)
            mask = positions[None, :] > positions[:, None]
            logits = self.model(
                tokens, positions.expand_as(tokens), mask[None, None], runtime_gather_output=True,
            )[:, -1].float()
            if logits.size(-1) != PADDED_VOCAB_SIZE:
                raise RuntimeError(f"unexpected Patch logits vocabulary size: {logits.size(-1)}")
            # The usable alphabet is the byte IDs and EOS, in that order.
            logits = logits[:, :EOS + 1] / temperature
            values, ids = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            index = torch.multinomial(values.softmax(dim=-1), num_samples=1)
            token = ids.gather(1, index).masked_fill(ended, EOS)
            tokens = torch.cat((tokens, token), dim=1)
            ended |= token.eq(EOS)
            if ended.all():
                break
        return tokens


def main():
    from examples.qwen3.pretrain_qwen import model_provider
    from megatron.core.enums import ModelType
    from megatron.training import get_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import get_model
    from megatron_patch.arguments import get_patch_args
    from megatron_patch.tokenizer import get_tokenizer

    def add_worker_args(parser):
        get_patch_args(parser)
        parser.add_argument("--rsi-request", type=Path, required=True)
        parser.add_argument("--rsi-response", type=Path, required=True)
        return parser

    try:
        initialize_megatron(extra_args_provider=add_worker_args)
        args = get_args()
        if args.world_size != 1:
            raise ValueError("the RSI Patch rollout worker requires one rank")
        request = json.loads(args.rsi_request.read_text(encoding="utf-8"))
        questions = request["questions"]
        tokenizer = get_tokenizer()
        for question in questions:
            prompt = question["prompt"]
            patch_ids = tokenizer.apply_chat_template([
                {"role": "user", "content": unicodedata.normalize("NFC", prompt)},
            ])
            if patch_ids != sft_prompt_tokens(prompt):
                raise ValueError(f"Patch tokenizer differs from RSI prompt encoding for {question['id']}")
        models = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
        load_checkpoint(models, None, None)
        device = torch.device("cuda", torch.cuda.current_device())
        generator = MegatronGenerator(models[0], device, args.seq_length)
        options = SimpleNamespace(seq_len=args.seq_length, **{
            key: request[key] for key in ("max_new_tokens", "top_k", "temperature")
        })
        rows = sample(generator, questions, request["count"], options, request["seed"])
        write_json(args.rsi_response, rows)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
