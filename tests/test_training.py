import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.experiment_data import EOS, IGNORE, byte_tokens, sft_example
from src.megatron_patch_backend import write_sft_data
from src.train import main


class TrainingSmokeTest(unittest.TestCase):
    def test_pretrain_then_sft(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            pretrain = Path(directory) / "pretrain"
            sft = Path(directory) / "sft"
            self.assertEqual(main(["--steps", "8", "--output", str(pretrain)]), 0)
            self.assertEqual(
                main([
                    "--stage", "sft", "--steps", "8", "--output", str(sft),
                    "--init-from", str(pretrain / "checkpoint.pt"),
                ]),
                0,
            )
            for stage, output in (("pretrain", pretrain), ("sft", sft)):
                result = json.loads((output / "summary.json").read_text())
                self.assertLess(result["valid_loss"], result["initial_valid_loss"])
                metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in metrics if row["split"] == "train"], list(range(1, 9)))
                checkpoint = torch.load(result["checkpoint"], weights_only=True)
                self.assertEqual((checkpoint["stage"], checkpoint["step"]), (stage, 8))
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(["--output", str(pretrain)]), 1)
            self.assertIn("File exists", errors.getvalue())

    def test_sft_shifts_first_answer_token_and_masks_prompt_and_padding(self):
        prompt = "User: 问题\nAssistant: "
        response = "café"
        inputs, targets = sft_example({"prompt": "问题", "response": "cafe\u0301"}, 40)
        offset = len(byte_tokens(prompt)) - 1
        expected = byte_tokens(response) + [EOS]
        self.assertEqual(inputs[: offset + 1], byte_tokens(prompt))
        self.assertEqual(targets[:offset], [IGNORE] * offset)
        self.assertEqual(targets[offset : offset + len(expected)], expected)
        self.assertTrue(all(label == IGNORE for label in targets[offset + len(expected) :]))
        with self.assertRaisesRegex(ValueError, "response and EOS"):
            sft_example({"prompt": "问题", "response": response}, offset + len(expected))

    def test_sft_rejects_reserved_tokens(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            sft_example({"prompt": "question", "response": "<pad>"}, 64)

    def test_patch_sft_writes_upstream_json_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "patch-data"
            write_sft_data(output, {"train": [{"prompt": "颜色？", "response": "红"}]})
            self.assertEqual(
                json.loads((output / "train.jsonl").read_text(encoding="utf-8")),
                {"query": "颜色？", "response": "红"},
            )


if __name__ == "__main__":
    unittest.main()
