import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.rsi.data import load_splits, make_demo, select_sft
from src.rsi.evaluate import promotion, verify
from src.rsi.planner import dsh_plan
from src.rsi.run_round import main


class RsiTest(unittest.TestCase):
    def test_verifier_and_selection(self):
        self.assertEqual(verify("Check: 2+3=5\nFinal: +5", 5), "correct")
        self.assertEqual(verify("2+3=5", 5), "unparseable")
        self.assertEqual(verify("5", 5, ended=False), "no_eos")
        self.assertEqual(verify("6", 5), "wrong")
        questions = [{"id": "easy", "prompt": "easy"}, {"id": "mixed", "prompt": "mixed"}]
        rows, stats = select_sft(
            questions,
            [
                {"id": "easy", "response": "5", "result": "correct"},
                {"id": "easy", "response": " 5 ", "result": "correct"},
                {"id": "mixed", "response": "5", "result": "correct"},
                {"id": "mixed", "response": "6", "result": "wrong"},
            ],
            [{"prompt": "replay", "answer": 5}], 0.2, 42, 2,
        )
        self.assertEqual(rows, [{"prompt": "mixed", "response": "5"}])
        self.assertEqual(stats["always_correct_prompts"], 1)

    def test_split_rejects_shared_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            make_demo(root)
            blind = root / "blind.jsonl"
            blind.write_text(blind.read_text().replace('"family": "K"', '"family": "A"'))
            with self.assertRaisesRegex(ValueError, "family appears in multiple splits"):
                load_splits(root, 64, 4)

    def test_family_regression_blocks_promotion(self):
        prior = [
            {"id": str(i), "family": "target" if i < 8 else "guardrail", "correct": 4, "samples": 8}
            for i in range(10)
        ]
        good = [dict(row, correct=6) for row in prior]
        regressed = [dict(row, correct=8 if i < 8 else 2) for i, row in enumerate(prior)]
        self.assertTrue(promotion(prior, good, 0.02, 0.02, 0)["promoted"])
        rejected = promotion(prior, regressed, 0.02, 0.02, 0)
        self.assertGreater(rejected["gain"], 0)
        self.assertFalse(rejected["promoted"])

    def test_dsh_only_accepts_replay_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for document, expected in [
                ('{"replay_fraction":0.4}', 0.4),
                ('{"replay_fraction":0.4,"min_gain":0}', 0.2),
            ]:
                with patch("src.rsi.planner.subprocess.run") as run:
                    run.return_value = subprocess.CompletedProcess([], 0, document, "")
                    fraction = dsh_plan("/dsh", 10, {"round": 1}, output, 0.2)
                    self.assertEqual(fraction, expected)
                    self.assertEqual(run.call_args.args[0][1:3], ["--profile", "headless"])
                    self.assertIn("round", run.call_args.kwargs["input"])

    def test_actual_sft_recursive_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "rsi"
            self.assertEqual(
                main(["--demo", "--rounds", "5", "--min-gain", "0.18", "--output", str(output)]),
                0,
            )
            summary = json.loads((output / "summary.json").read_text())
            first, second, third, fourth = summary["rounds"]
            self.assertTrue(first["promoted"])
            self.assertEqual(second["parent"], first["checkpoint"])
            self.assertEqual(second["generator"], first["checkpoint"])
            self.assertTrue(second["promoted"])
            self.assertFalse(third["promoted"])
            self.assertFalse(fourth["promoted"])
            self.assertEqual(fourth["parent"], second["checkpoint"])
            self.assertEqual(summary["stop_reason"], "patience_exhausted")
            self.assertEqual(summary["best_checkpoint"], second["checkpoint"])
            self.assertTrue(Path(summary["best_checkpoint"]).is_file())
            self.assertGreater(summary["supervised_tokens"], 0)
            self.assertLessEqual(summary["generation_token_slots"], 200_000)
