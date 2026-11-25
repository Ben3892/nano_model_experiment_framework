"""Load an existing Patch checkpoint on one GPU and sample once."""

import argparse
from pathlib import Path

from ..megatron_patch_backend import check_gpus, check_patch_root
from ..train import write_json
from .megatron_rollout import PatchCheckpoint, sample_patch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--init-iteration", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(seed=42, max_new_tokens=4, temperature=1.0, top_k=8)
    args = parser.parse_args(argv)
    args.patch_root = args.patch_root.resolve()
    output = args.output.resolve()
    check_patch_root(args.patch_root)
    checkpoint = PatchCheckpoint.read(args.init_from, args.seq_len, args.init_iteration)
    check_gpus(1)
    output.mkdir(parents=True, exist_ok=False)
    questions = [{"id": "probe", "prompt": "1+0 = ?", "answer": 1}]
    rows = sample_patch(checkpoint, questions, 1, args, args.seed, output / "probe.jsonl")
    write_json(output / "summary.json", {
        "status": "sampled", "checkpoint_metadata": checkpoint.metadata(), "sample": rows[0],
    })
    print(f"Sampled {checkpoint.root} at iteration {checkpoint.iteration}; see {output}")


if __name__ == "__main__":
    main()
