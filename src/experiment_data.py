"""Small UTF-8 byte-token corpora shared by both experiment backends."""

import json
from pathlib import Path
import unicodedata

EOS = 256
PAD = 257
UNK = 258
VOCAB_SIZE = 259
IGNORE = -100
SPECIAL_TOKENS = {"</s>": EOS, "<pad>": PAD, "<unk>": UNK}
CHAT_TEMPLATE = (
    "User: {{ messages[0]['content'] }}\nAssistant: "
    "{% if messages|length > 1 %}{{ messages[1]['content'] }}{{ eos_token }}{% endif %}"
)


def byte_tokens(text):
    return list(text.encode("utf-8"))


def builtin_rows(stage, split):
    examples = []
    start, count = (0, 96) if split == "train" else (100, 24)
    for index in range(start, start + count):
        key, value = (("red", "crimson"), ("blue", "azure"), ("green", "jade"))[index % 3]
        if stage == "pretrain":
            examples.append(f"Record {index:03d}: the value for {key} is {value}.")
        else:
            examples.append({"prompt": f"Record {index:03d}: value for {key}?", "response": value})
    return examples


def load_rows(stage, split, path):
    if path is None:
        return builtin_rows(stage, split)

    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if stage == "pretrain":
            rows.append(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("prompt", "response")
        ):
            raise ValueError(f"{path}:{number}: expected non-empty prompt and response strings")
        rows.append({"prompt": row["prompt"], "response": row["response"]})
    return rows


def sft_example(row, seq_len):
    prompt, answer = (unicodedata.normalize("NFC", row[key]) for key in ("prompt", "response"))
    if any(token in prompt or token in answer for token in SPECIAL_TOKENS):
        raise ValueError("SFT prompt and response cannot contain reserved tokenizer tokens")
    prefix = byte_tokens(f"User: {prompt}\nAssistant: ")
    response = byte_tokens(answer) + [EOS]
    tokens = prefix + response
    if len(tokens) > seq_len:
        raise ValueError(f"SFT example needs seq_len >= {len(tokens)} for response and EOS")
    tokens += [PAD] * (seq_len + 1 - len(tokens))
    labels = [IGNORE] * (len(prefix) - 1) + response
    return tokens[:-1], labels + [IGNORE] * (seq_len - len(labels))


def validate_rows(stage, rows, seq_len, batch_size, split):
    if stage == "pretrain":
        if sum(len(byte_tokens(row)) + 1 for row in rows) <= seq_len:
            raise ValueError(f"{split} needs at least {seq_len + 1} byte tokens including EOS")
        return

    if len(rows) < batch_size:
        raise ValueError(f"{split} needs at least {batch_size} SFT rows")
    for number, row in enumerate(rows, 1):
        try:
            sft_example(row, seq_len)
        except ValueError as error:
            raise ValueError(f"{split} row {number}: {error}") from error
