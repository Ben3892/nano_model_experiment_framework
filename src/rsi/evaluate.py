"""Deterministic verification and paired evaluation of sampled responses."""

from collections import Counter
import hashlib
from math import comb
import random
import re
import unicodedata

import torch

from ..experiment_data import EOS, SPECIAL_TOKENS, byte_tokens, sft_prompt_tokens

INTEGER = re.compile(r"[+-]?[0-9]+")
FINAL = re.compile(r"Final:\s*([+-]?[0-9]+)", re.IGNORECASE)


def verify(response, answer, ended=True):
    if not ended:
        return "no_eos"
    text = response.strip()
    match = FINAL.fullmatch(text.splitlines()[-1]) if text else None
    number = match[1] if match else text
    if not INTEGER.fullmatch(number) or len(number) > 128:
        return "unparseable"
    return "correct" if int(number) == answer else "wrong"


def sample(model, questions, count, args, seed):
    records = []
    model.eval()
    with torch.random.fork_rng(devices=[]):
        for question in questions:
            raw_seed = hashlib.sha256(f"{seed}:{question['id']}".encode()).digest()
            torch.manual_seed(int.from_bytes(raw_seed[:8], "big"))
            prefix = sft_prompt_tokens(question["prompt"])
            inputs = torch.tensor([prefix]).expand(count, -1).contiguous()
            output = model.generate(
                inputs, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
            )
            for index, tokens in enumerate(output[:, len(prefix):].tolist()):
                ended = EOS in tokens
                if ended:
                    tokens = tokens[:tokens.index(EOS)]
                try:
                    response = bytes(tokens).decode("utf-8")
                except (UnicodeError, ValueError):
                    response, result = None, "invalid_bytes"
                else:
                    normalized = unicodedata.normalize("NFC", response.strip())
                    if any(token in normalized for token in SPECIAL_TOKENS):
                        result = "reserved_token"
                    elif len(prefix) + len(byte_tokens(normalized)) + 1 > args.seq_len:
                        result = "too_long"
                    else:
                        result = verify(response, question["answer"], ended)
                records.append({
                    "id": question["id"], "sample": index, "response": response, "result": result,
                })
    return records


def question_scores(questions, records, n):
    correct = Counter(row["id"] for row in records if row["result"] == "correct")
    return [
        {"id": row["id"], "family": row["family"], "correct": correct[row["id"]], "samples": n}
        for row in questions
    ]


def metrics(scores):
    n = scores[0]["samples"]
    counts = [row["correct"] for row in scores]
    ks = sorted({1, min(n, 8), min(n, 32)})
    result = {
        f"pass@{k}": sum(1 - comb(n - c, k) / comb(n, k) for c in counts) / len(counts)
        for k in ks
    }
    result["all_correct_fraction"] = counts.count(n) / len(counts)
    result["no_correct_fraction"] = counts.count(0) / len(counts)
    return result


def promotion(prior, candidate, min_gain, max_regression, seed):
    changes, families = [], {}
    stable_losses = 0
    for before, after in zip(prior, candidate, strict=True):
        if (before["id"], before["samples"]) != (after["id"], after["samples"]):
            raise ValueError("paired evaluation requires identical question IDs and sample counts")
        n = before["samples"]
        change = (after["correct"] - before["correct"]) / n
        changes.append(change)
        families.setdefault(before["family"], []).append(change)
        stable_losses += before["correct"] == n and after["correct"] < n

    rng = random.Random(seed)
    # Resample whole questions within each fixed dev slice. Samples of the same
    # question are correlated and must stay together.
    boot = sorted(
        sum(sum(rng.choices(values, k=len(values))) for values in families.values()) / len(changes)
        for _ in range(1000)
    )
    gain = sum(changes) / len(changes)
    family_gains = {key: sum(values) / len(values) for key, values in families.items()}
    lower = boot[24]
    checks = {
        "minimum_gain": gain >= min_gain,
        "positive_ci": lower > 0,
        "family_regression": min(family_gains.values()) >= -max_regression,
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "gain": gain,
        "paired_ci95": [lower, boot[974]],
        "family_gains": family_gains,
        "stable_questions_regressed": stable_losses,
        "lost_solvable_questions": sum(a["correct"] > 0 and b["correct"] == 0 for a, b in zip(prior, candidate)),
        "newly_solvable_questions": sum(a["correct"] == 0 and b["correct"] > 0 for a, b in zip(prior, candidate)),
        "prior": metrics(prior),
        "candidate": metrics(candidate),
    }
