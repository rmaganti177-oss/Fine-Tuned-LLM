"""The centerpiece: in-run generation-based schema-compliance callback.
Every checkpoint it GENERATES on the gold set (native, greedy) and logs REAL
compliance % — NOT eval_loss. Selects the best checkpoint on nested-compliance.
Triple-durability logging (W&B + stdout + JSON sidecar) so headless Kaggle is
never blind and a stale log can never mislead.

Import-safe: torch/wandb/unsloth imported lazily inside the callback so this
module can be UNIT-TESTED on CPU (see _unit_test) without those deps.
"""
import json
import os
import time

from prompt import render_messages, PROMPT_HASH
from validate import (validate_2020_12, extract_json,
                      count_hallucinated_keys, looks_like_schema_echo)


def score_generations(rows, texts):
    """Pure function (no GPU/deps) -> (metrics dict, sample table). Unit-testable."""
    m = dict(n=len(rows), flat_ok=0, flat_n=0, nested_ok=0, nested_n=0,
             parse_fail=0, halluc=0, echo=0, req_missing=0)
    table = []
    for r, text in zip(rows, texts):
        schema = r["schema"] if isinstance(r["schema"], dict) else json.loads(r["schema"])
        nested = (r["complexity_class"] == "nested")
        m["nested_n" if nested else "flat_n"] += 1
        obj, ok_parse = extract_json(text)
        if not ok_parse:
            m["parse_fail"] += 1
            table.append([r.get("subset", "?"), text[:120], "PARSE_FAIL"])
            continue
        if looks_like_schema_echo(obj, schema):
            m["echo"] += 1
        m["halluc"] += count_hallucinated_keys(obj, schema)
        valid, err = validate_2020_12(obj, schema)
        if valid:
            m["nested_ok" if nested else "flat_ok"] += 1
            table.append([r.get("subset", "?"), text[:120], "OK"])
        else:
            if err and "required" in err:
                m["req_missing"] += 1
            table.append([r.get("subset", "?"), text[:120], (err or "INVALID")[:100]])
    m["compliance_flat"] = m["flat_ok"] / max(m["flat_n"], 1)
    m["compliance_nested"] = m["nested_ok"] / max(m["nested_n"], 1)
    m["compliance_all"] = (m["flat_ok"] + m["nested_ok"]) / max(m["n"], 1)
    return m, table


def make_callback(tokenizer, gold_rows, out_dir, n_fast=200, gen_batch=16,
                  max_new_tokens=640):
    """Returns a TrainerCallback. Imported here so unit tests need no transformers."""
    import torch
    import wandb
    from transformers import TrainerCallback
    from unsloth import FastLanguageModel

    fast = gold_rows[:n_fast]

    class GoldComplianceCallback(TrainerCallback):
        def __init__(self):
            self.best_nested, self.best_step = -1.0, None

        @torch.no_grad()
        def _generate(self, model):
            prompts = [tokenizer.apply_chat_template(
                           render_messages(r["schema"]), tokenize=False,
                           add_generation_prompt=True, enable_thinking=False)
                       for r in fast]
            out = []
            for i in range(0, len(prompts), gen_batch):
                chunk = prompts[i:i + gen_batch]
                enc = tokenizer(chunk, return_tensors="pt", padding=True,
                                truncation=True, max_length=3072).to(model.device)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.eos_token_id)
                new = gen[:, enc.input_ids.shape[1]:]
                out += tokenizer.batch_decode(new, skip_special_tokens=True)
            return out

        def on_evaluate(self, args, state, control, **kw):
            model, step = kw["model"], state.global_step
            try:
                FastLanguageModel.for_inference(model)
                t0 = time.time()
                texts = self._generate(model)
                m, table = score_generations(fast, texts)
                secs = time.time() - t0
                row = {"gold/compliance_all": m["compliance_all"],
                       "gold/compliance_flat": m["compliance_flat"],
                       "gold/compliance_nested": m["compliance_nested"],
                       "gold/parse_fail_rate": m["parse_fail"] / m["n"],
                       "gold/hallucinated_keys_per_sample": m["halluc"] / m["n"],
                       "gold/schema_echo_rate": m["echo"] / m["n"],
                       "gold/required_missing_rate": m["req_missing"] / m["n"],
                       "gold/gen_seconds": secs, "gold/prompt_hash_ok": PROMPT_HASH}
                wandb.log(row, step=step)
                wandb.log({"gold/samples": wandb.Table(
                    columns=["subset", "output_head", "verdict"], data=table[:25])}, step=step)
                print(f"[GOLD step={step}] all={m['compliance_all']:.3f} "
                      f"flat={m['compliance_flat']:.3f} nested={m['compliance_nested']:.3f} "
                      f"parsefail={row['gold/parse_fail_rate']:.3f} "
                      f"halluc={row['gold/hallucinated_keys_per_sample']:.2f} "
                      f"echo={row['gold/schema_echo_rate']:.3f} ({secs:.0f}s)", flush=True)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"gold_step{step}.json"), "w") as f:
                    json.dump({"step": step, **row}, f, indent=2)
                if m["compliance_nested"] > self.best_nested:
                    self.best_nested, self.best_step = m["compliance_nested"], step
                    with open(os.path.join(out_dir, "best_by_nested.json"), "w") as f:
                        json.dump({"best_step": step, "best_nested": self.best_nested}, f)
                    wandb.run.summary["best_nested_compliance"] = self.best_nested
                    wandb.run.summary["best_step"] = step
            except Exception as e:
                print(f"[GOLD step={step}] CALLBACK ERROR (training continues): {e}", flush=True)
            finally:
                FastLanguageModel.for_training(model)
                torch.cuda.empty_cache()
            return control

    return GoldComplianceCallback()


def _unit_test():
    """CPU-only: verify the metric math with canned generations."""
    rows = [
        {"schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
         "complexity_class": "flat", "subset": "t"},
        {"schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
         "complexity_class": "flat", "subset": "t"},
        {"schema": {"$defs": {"x": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}},
                    "$ref": "#/$defs/x"}, "complexity_class": "nested", "subset": "t"},
    ]
    texts = ['{"n": 5}',                         # flat OK
             'sure! {"n": "not-int"}',           # flat invalid (type)
             '```json\n{"a": "hi", "extra": 1}\n```']  # nested OK but 1 hallucinated key
    m, table = score_generations(rows, texts)
    assert m["compliance_flat"] == 0.5, m
    assert m["compliance_nested"] == 1.0, m
    assert m["halluc"] == 1, m
    assert m["parse_fail"] == 0, m
    print("callback metric unit-test PASSED:", {k: m[k] for k in
          ("compliance_all", "compliance_flat", "compliance_nested", "halluc")})


if __name__ == "__main__":
    _unit_test()
