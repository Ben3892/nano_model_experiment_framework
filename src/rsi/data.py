"""Versioned question records and verified SFT data selection."""

from collections import Counter
import hashlib
import json
import unicodedata

from ..experiment_data import sft_example, sft_prompt_tokens

SPLITS = ("rollout", "replay", "monitor", "dev", "blind")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_splits(directory, seq_len, max_new_tokens):
    seen_ids, seen_prompts, family_splits = set(), set(), {}
    splits, hashes = {}, {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        raw = path.read_bytes()
        hashes[split] = hashlib.sha256(raw).hexdigest()
        questions = []
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            question = json.loads(line)
            location = f"{path}:{number}"
            if (
                not isinstance(question, dict)
                or any(
                    not isinstance(question.get(key), str) or not question[key].strip()
                    for key in ("id", "family", "prompt")
                )
                or type(question.get("answer")) is not int
            ):
                raise ValueError(f"{location}: expected id, family, prompt and integer answer")
            question = {key: question[key] for key in ("id", "family", "prompt", "answer")}
            prompt_key = " ".join(unicodedata.normalize("NFKC", question["prompt"]).casefold().split())
            if question["id"] in seen_ids or prompt_key in seen_prompts:
                raise ValueError(f"{location}: duplicate question ID or normalized prompt")
            family_key = " ".join(unicodedata.normalize("NFKC", question["family"]).casefold().split())
            if family_splits.setdefault(family_key, split) != split:
                raise ValueError(f"{location}: question family appears in multiple splits")
            seen_ids.add(question["id"])
            seen_prompts.add(prompt_key)
            if split in ("rollout", "dev", "blind") and (
                len(sft_prompt_tokens(question["prompt"])) + max_new_tokens > seq_len
                or len(str(question["answer"])) + 1 > max_new_tokens
            ):
                raise ValueError(f"{location}: prompt or answer does not fit decoding budget")
            sft_example(as_sft(question), seq_len)
            questions.append(question)
        if not questions:
            raise ValueError(f"{path}: split is empty")
        splits[split] = questions
    return splits, hashes


def as_sft(question):
    return {"prompt": question["prompt"], "response": str(question["answer"])}


def select_sft(questions, rollouts, replay, replay_fraction, seed, max_fresh_tokens):
    """Retain up to two unique correct responses per question."""
    correct = {question["id"]: [] for question in questions}
    attempts, successes = Counter(), Counter()
    for row in rollouts:
        attempts[row["id"]] += 1
        if row["result"] != "correct":
            continue
        successes[row["id"]] += 1
        candidates = correct[row["id"]]
        response = unicodedata.normalize("NFC", row["response"].strip())
        if response not in candidates and len(candidates) < 2:
            candidates.append(response)

    # Give each partially solved question one slot before adding variants.
    # Keep split order when it all fits to make sampling reproducible.
    ordered = sorted(
        enumerate(questions),
        key=lambda item: successes[item[1]["id"]] == attempts[item[1]["id"]],
    )
    selected, fresh_tokens = [], 0
    for variant in range(2):
        for index, question in ordered:
            responses = correct[question["id"]]
            if len(responses) <= variant:
                continue
            response = responses[variant]
            cost = len(response.encode("utf-8")) + 1
            if fresh_tokens + cost <= max_fresh_tokens:
                selected.append((index, variant, {"prompt": question["prompt"], "response": response}))
                fresh_tokens += cost
    fresh = [row for _, _, row in sorted(selected)]
    target_replay_tokens = round(fresh_tokens * replay_fraction / (1 - replay_fraction))
    replay_tokens, replay_rows = 0, []
    offset = seed % len(replay)
    for question in replay[offset:] + replay[:offset]:
        row = as_sft(question)
        cost = len(row["response"].encode("utf-8")) + 1
        if replay_tokens + cost <= target_replay_tokens:
            replay_rows.append(row)
            replay_tokens += cost
    stats = {
        "positive_prompts": len({index for index, _, _ in selected}),
        "zero_correct_prompts": sum(successes[row["id"]] == 0 for row in questions),
        "always_correct_prompts": sum(successes[row["id"]] == attempts[row["id"]] for row in questions),
        "total_prompts": len(questions),
        "fresh_examples": len(fresh),
        "fresh_supervised_tokens": fresh_tokens,
        "replay_examples": len(replay_rows),
        "replay_supervised_tokens": replay_tokens,
        "target_replay_fraction": replay_fraction,
        "actual_replay_fraction": replay_tokens / max(1, fresh_tokens + replay_tokens),
    }
    return fresh + replay_rows, stats


def make_demo(directory):
    directory.mkdir()
    tags = {
        "rollout": ("A", "B", "C"),
        "replay": ("D", "E", "F", "G"),
        "monitor": ("H",),
        "dev": ("I", "J"),
        "blind": ("K",),
    }
    for split, letters in tags.items():
        rows = [
            {"id": f"{tag}-{number}", "family": tag, "prompt": f"{tag} {number}+0=?", "answer": number}
            for tag in letters
            for number in range(10)
        ]
        write_jsonl(directory / f"{split}.jsonl", rows)
