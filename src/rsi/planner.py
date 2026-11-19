"""Optional DeepSeek Harness planner for the next round's replay ratio."""

import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from ..train import write_json

MIN_REPLAY, MAX_REPLAY = 0.1, 0.5


def dsh_plan(executable, timeout, report, directory, current):
    task = (
        "You are tuning a supervised self-training experiment. Review the aggregate report below. "
        "Return exactly one JSON object with one key replay_fraction, a number between 0.1 and 0.5. "
        "Keep the current value if the report does not justify changing it. "
        "Do not use tools.\n" + json.dumps(report)
    )
    (directory / "planner_prompt.txt").write_text(task, encoding="utf-8")
    proposal = {"status": "fallback", "replay_fraction": current}
    try:
        command = [
            executable, "--profile", "headless", "--permission-mode", "read-only",
            "--approval-policy", "never",
        ]
        with TemporaryDirectory(prefix="nano-rsi-dsh-") as workspace:
            result = subprocess.run(
                command, input=task, text=True, encoding="utf-8", errors="replace",
                capture_output=True, cwd=workspace, timeout=timeout, check=False,
            )
        (directory / "dsh.stdout").write_text(result.stdout, encoding="utf-8")
        (directory / "dsh.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"dsh exited with code {result.returncode}")
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or set(value) != {"replay_fraction"}:
            raise ValueError("expected a JSON object with only replay_fraction")
        fraction = value["replay_fraction"]
        if type(fraction) not in (int, float) or not MIN_REPLAY <= fraction <= MAX_REPLAY:
            raise ValueError("replay_fraction outside allowed range")
        proposal = {"status": "accepted", "replay_fraction": fraction}
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        proposal["reason"] = str(error)
    write_json(directory / "proposal.json", proposal)
    return proposal["replay_fraction"]


def resolve_dsh(executable):
    path = shutil.which(executable)
    if path is None:
        raise ValueError(f"DSH executable not found: {executable}")
    return str(Path(path).resolve())
