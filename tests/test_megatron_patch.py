import contextlib
import io
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.experiment_data import EOS, PAD, UNK
from src.megatron_patch_backend import (
    CORE, ENTRYPOINT, PADDED_VOCAB_SIZE, TRACKER, build_command, checkpoint_files, write_tokenizer,
)
from src.rsi.data import make_demo
from src.rsi.megatron_rollout import MegatronGenerator, PatchCheckpoint, sample_patch
from src.rsi.rollout_smoke import main as smoke_rollout
from src.rsi.run_round import main as run_rsi
from src.train import main as train, parse_args, write_json


def fake_patch(root):
    for path in (root / ENTRYPOINT, root / CORE / "megatron/training/training.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def write_checkpoint_files(root, iteration):
    step = root / f"iter_{iteration:07d}"
    step.mkdir(parents=True)
    for name in ("common.pt", ".metadata", "__0_0.distcp"):
        (step / name).write_bytes(b"test checkpoint bytes")
    write_json(step / "metadata.json", {"sharded_backend": "torch_dist", "sharded_backend_version": 1})
    (root / TRACKER).write_text(f"{iteration}\n")


def fake_checkpoint(run, iteration):
    root = run / "checkpoint"
    write_checkpoint_files(root, iteration)
    tokenizer = run / "tokenizer"
    write_tokenizer(tokenizer)
    args = parse_args([
        "--backend", "megatron-patch", "--patch-root", str(run.parent),
        "--stage", "sft", "--steps", str(iteration), "--output", str(run),
    ])
    write_json(run / "config.json", {"backend": args.backend, "seq_len": args.seq_len})
    write_json(run / "launch.json", {"argv": build_command(args, tokenizer, root)})
    return root


def warmstart_log(parent, iteration):
    return (
        f" loading distributed checkpoint from {parent} at iteration {iteration}\n"
        f" successfully loaded checkpoint from {parent} [ t 1/1, p 1/1 ] at iteration 0\n"
    )


class PatchTests(unittest.TestCase):
    def test_checkpoint_rejects_missing_invalid_or_unsupported_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = fake_checkpoint(Path(directory) / "seed", 4)
            manifest = parent / "iter_0000004/metadata.json"
            manifest.unlink()
            with self.assertRaisesRegex(ValueError, "metadata.json"):
                checkpoint_files(parent)
            for contents in (
                "{", "null", "{}", '{"sharded_backend": "torch_dcp"}',
                '{"sharded_backend": "torch_dist", "sharded_backend_version": 2}',
                '{"sharded_backend": "torch_dist", "sharded_backend_version": true}',
                '{"sharded_backend": "torch_dist", "common_backend": "unknown"}',
                '{"sharded_backend": "torch_dist", "common_backend_version": 2}',
            ):
                with self.subTest(contents=contents), self.assertRaisesRegex(ValueError, "metadata.json"):
                    manifest.write_text(contents)
                    checkpoint_files(parent)
            write_json(manifest, {"sharded_backend": "torch_dist"})
            checkpoint = PatchCheckpoint.read(parent, 64)
            self.assertEqual(checkpoint.sha256["checkpoint/iter_0000004/metadata.json"],
                             hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_rollout_smoke_runs_one_sample_without_requiring_correct_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch_root, output = root / "patch", root / "probe"
            fake_patch(patch_root)
            parent = fake_checkpoint(root / "seed", 4)
            command = ["--patch-root", str(patch_root), "--init-from", str(parent), "--output", str(output)]
            row = {"id": "probe", "sample": 0, "response": "", "result": "no_eos"}

            def worker(argv, stdout, **kwargs):
                self.assertIn("--nproc-per-node=1", argv)
                self.assertEqual(argv[argv.index("--ckpt-step") + 1], "4")
                self.assertEqual(argv[argv.index("--pretrained-checkpoint") + 1], str(parent))
                request = json.loads((output / "probe.request.json").read_text())
                self.assertEqual((len(request["questions"]), request["count"], request["seed"]), (1, 1, 42))
                write_json(output / "probe.response.json", [row])
                stdout.write(warmstart_log(parent, 4))
                return subprocess.CompletedProcess(argv, 0)

            with patch("src.rsi.rollout_smoke.check_gpus"), patch(
                "src.rsi.megatron_rollout.subprocess.run", side_effect=worker
            ) as launch:
                with self.assertRaisesRegex(ValueError, "changed from 3 to 4"):
                    smoke_rollout([*command, "--init-iteration", "3"])
                self.assertFalse(output.exists())
                smoke_rollout([*command, "--init-iteration", "4"])
                with self.assertRaises(FileExistsError):
                    smoke_rollout([*command, "--init-iteration", "4"])
                launch.assert_called_once()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual((summary["status"], summary["sample"]), ("sampled", row))
            self.assertEqual(summary["checkpoint_metadata"]["iteration"], 4)
            for suffix in ("request.json", "launch.json", "response.json", "megatron.log"):
                self.assertTrue((output / f"probe.{suffix}").stat().st_size)

    def test_dp2_prepare_checks_parent_and_launch_paths(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            patch_root, output = root / "patch", root / "sft"
            fake_patch(patch_root)
            parent = fake_checkpoint(root / "seed", 4)
            self.assertEqual(train([
                "--backend", "megatron-patch", "--patch-root", str(patch_root), "--stage", "sft",
                "--nproc-per-node", "2", "--steps", "2", "--init-from", str(parent),
                "--init-iteration", "4", "--prepare-only", "--output", str(output),
            ]), 0)
            launch = json.loads((output / "launch.json").read_text())["argv"]
            self.assertIn("--nproc-per-node=2", launch)
            for flag, value in {
                "--micro-batch-size": "2", "--global-batch-size": "4", "--ckpt-step": "4",
                "--load": str(output / "tokenizer"), "--pretrained-checkpoint": str(parent),
                "--save": str(output / "checkpoint"),
            }.items():
                self.assertEqual(launch[launch.index(flag) + 1], value)
            self.assertEqual(json.loads((output / "summary.json").read_text()), {"status": "prepared"})
            (parent / "iter_0000004" / ".metadata").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete torch_dist checkpoint"):
                checkpoint_files(parent, 4)

    def test_patch_training_rejects_missing_warmstart_evidence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            patch_root, output = root / "patch", root / "sft"
            fake_patch(patch_root)
            parent = fake_checkpoint(root / "seed", 4)
            good_lines = (
                "using world size: 2, data-parallel size: 2, context-parallel size: 1, "
                "tensor-model-parallel size: 1, pipeline-model-parallel size: 1, encoder-pipeline-model-parallel size: 0\n"
                " iteration 1 / 1 | lm loss: 5.0 | number of skipped iterations: 0 | number of nan iterations: 0 |\n"
                " validation loss at iteration 1 on validation set | lm loss value: 4.0 |\n"
            )

            def fake_child(argv, **kwargs):
                checkpoint = Path(argv[argv.index("--save") + 1])
                write_checkpoint_files(checkpoint, 1)
                child = MagicMock()
                child.__enter__.return_value = child
                child.stdout = io.StringIO(good_lines)
                child.wait.return_value = 0
                return child

            errors = io.StringIO()
            with patch("src.megatron_patch_backend.check_gpus"), patch(
                "src.megatron_patch_backend.subprocess.Popen", side_effect=fake_child
            ), contextlib.redirect_stderr(errors):
                self.assertEqual(train([
                    "--backend", "megatron-patch", "--patch-root", str(patch_root), "--stage", "sft",
                    "--init-from", str(parent), "--nproc-per-node", "2", "--steps", "1", "--output", str(output),
                ]), 1)
            self.assertIn("could not confirm warm start", errors.getvalue())
            self.assertFalse((output / "summary.json").exists())

    def test_rollout_load_is_pinned_and_tracker_drift_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = fake_checkpoint(root / "seed", 4)
            checkpoint = PatchCheckpoint.read(parent, 64)
            questions = [{"id": "q", "prompt": "1+0", "answer": 1}]
            args = SimpleNamespace(patch_root=root, max_new_tokens=4, top_k=8, temperature=1, seed=42)
            rows = [{"id": "q", "sample": 0, "response": "1", "result": "correct"}]

            def worker(argv, stdout, **kwargs):
                self.assertIn("--nproc-per-node=1", argv)
                self.assertEqual(argv[argv.index("--ckpt-step") + 1], "4")
                self.assertEqual(argv[argv.index("--load") + 1], str(checkpoint.tokenizer))
                write_json(Path(argv[argv.index("--rsi-response") + 1]), rows)
                stdout.write(warmstart_log(parent, 4))
                return subprocess.CompletedProcess(argv, 0)

            with patch("src.rsi.megatron_rollout.subprocess.run", side_effect=worker) as launch:
                self.assertEqual(sample_patch(checkpoint, questions, 1, args, 42, root / "rollout.jsonl"), rows)
                (parent / TRACKER).write_text("5\n")
                with self.assertRaisesRegex(ValueError, "changed from 4 to 5"):
                    sample_patch(checkpoint, questions, 1, args, 42, root / "rollout-next.jsonl")
                launch.assert_called_once()

    def test_generator_filters_unusable_ids_and_keeps_eos_per_sample(self):
        def model(tokens, positions, mask, runtime_gather_output):
            self.assertTrue(runtime_gather_output)
            length = tokens.size(1)
            self.assertEqual(positions[0].tolist(), list(range(length)))
            self.assertTrue(torch.equal(mask[0, 0], torch.ones(length, length, dtype=torch.bool).triu(1)))
            logits = torch.full((*tokens.shape, PADDED_VOCAB_SIZE), -100.0)
            logits[..., [PAD, UNK, PADDED_VOCAB_SIZE - 1]] = 100.0
            logits[0, -1, EOS] = 10.0
            logits[1, -1, ord("1") if length == 2 else EOS] = 10.0
            return logits

        generator = MegatronGenerator(model, torch.device("cpu"), 8)
        tokens = generator.generate(torch.tensor([[65, 66], [65, 66]]), 4, temperature=1, top_k=1)
        self.assertEqual(tokens.tolist(), [[65, 66, EOS, EOS], [65, 66, ord("1"), EOS]])

    def test_rsi_promotes_patch_candidate_with_pinned_parent_and_missing_token_metric(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            patch_root, data, output = root / "patch", root / "data", root / "rsi"
            fake_patch(patch_root)
            make_demo(data)
            parent = fake_checkpoint(root / "seed", 4)

            def fit(command):
                args = parse_args(command)
                self.assertEqual(args.init_from, parent)
                self.assertEqual(args.init_iteration, 4)
                self.assertEqual(args.backend, "megatron-patch")
                checkpoint = fake_checkpoint(args.output, args.steps)
                write_json(args.output / "summary.json", {"checkpoint": str(checkpoint)})
                return 0

            def sample(checkpoint, questions, count, args, seed, path):
                is_correct = path.name == "rollouts.jsonl" or checkpoint.root != parent
                return [
                    {"id": q["id"], "sample": index,
                     "response": str(q["answer"] if is_correct else q["answer"] + 1),
                     "result": "correct" if is_correct else "wrong"}
                    for q in questions for index in range(count)
                ]

            with patch("src.rsi.run_round.check_gpus"), patch(
                "src.rsi.run_round.train_main", side_effect=fit
            ), patch("src.rsi.run_round.sample_patch", side_effect=sample):
                self.assertEqual(run_rsi([
                    "--backend", "megatron-patch", "--patch-root", str(patch_root), "--data-dir", str(data),
                    "--init-from", str(parent), "--output", str(output), "--rounds", "1", "--steps", "2",
                ]), 0)

            decision = json.loads((output / "round_001" / "decision.json").read_text())
            best = json.loads((output / "best.json").read_text())
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(decision["parent_iteration"], 4)
            self.assertTrue(decision["promoted"])
            self.assertEqual(best["checkpoint"], decision["checkpoint"])
            self.assertEqual(best["checkpoint_metadata"]["iteration"], 2)
            self.assertEqual(result["train_token_slots"], 2 * 4 * 64)
            self.assertIsNone(result["supervised_tokens"])


if __name__ == "__main__":
    unittest.main()
