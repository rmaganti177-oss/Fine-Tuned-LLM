"""STEP 4 smoke test: render real training rows through the ACTUAL Qwen3-1.7B
tokenizer (enable_thinking=False), verify (a) the tokenizer loads, (b) the
assistant JSON renders intact, (c) train_on_responses_only markers are present,
(d) examples fit max_seq_length, (e) the completion re-validates against its
schema. Catches train/serve-format bugs on CPU before any GPU spend.
"""
import json
import sys

from transformers import AutoTokenizer
from validate import validate_2020_12

MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_SEQ = 3072
INSTR_MARK = "<|im_start|>user\n"
RESP_MARK = "<|im_start|>assistant\n"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print("tokenizer loaded:", MODEL_ID, "| vocab", tok.vocab_size)

    rows = [json.loads(l) for l in open("sft_train.jsonl", encoding="utf-8")]
    print("train rows:", len(rows))

    over_len = 0
    bad_render = 0
    bad_valid = 0
    lengths = []
    shown = 0
    for i, r in enumerate(rows):
        msgs = r["messages"]
        # full rendered text (what SFT trains on)
        try:
            full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        except TypeError:
            full = tok.apply_chat_template(msgs, tokenize=False)
        n = len(tok(full)["input_ids"])
        lengths.append(n)
        if n > MAX_SEQ:
            over_len += 1
        if INSTR_MARK not in full or RESP_MARK not in full:
            bad_render += 1
        # the assistant completion must be the last JSON and re-validate
        assistant = msgs[-1]["content"]
        schema = json.loads(r["schema"])
        try:
            obj = json.loads(assistant)
            ok, _ = validate_2020_12(obj, schema)
            if not ok:
                bad_valid += 1
        except Exception:
            bad_valid += 1
        if shown < 3 and r["complexity_class"] == "nested":
            print(f"\n--- nested example {i} (tokens={n}) ---")
            print(full[-500:])
            shown += 1

    lengths.sort()
    p50 = lengths[len(lengths) // 2]
    p95 = lengths[int(len(lengths) * 0.95)]
    print(f"\ntoken length: p50={p50} p95={p95} max={lengths[-1]}")
    print(f"over max_seq({MAX_SEQ}): {over_len}")
    print(f"missing masking markers: {bad_render}")
    print(f"completions that DON'T re-validate: {bad_valid}")

    ok = (over_len < len(rows) * 0.05 and bad_render == 0 and bad_valid == 0)
    print("\nSMOKE:", "PASS - cleared for GPU" if ok else "FAIL - fix before GPU")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
