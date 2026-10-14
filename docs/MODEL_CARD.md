---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
pipeline_tag: text-generation
library_name: transformers
tags:
- json
- json-schema
- structured-output
- qlora
- function-calling
datasets:
- ujwal00/jsonschema-sft
language:
- en
---

# Qwen3-1.7B-JSON — native JSON-Schema compliance fine-tune

Fine-tune of **Qwen/Qwen3-1.7B** that emits JSON conforming to a requested
**JSON Schema** *natively* — no grammar-constrained decoding — with a large
gain on the hard case: **nested / `$ref` / `$defs` schemas**.

- **Base model:** Qwen/Qwen3-1.7B (Apache-2.0)
- **Method:** QLoRA SFT (Unsloth), r=16, α=32, 2 epochs, NEFTune α=5
- **Data:** ~4k schema→valid-instance pairs generated from JSONSchemaBench
  *train-split* schemas (72% nested), every target validated with a
  Draft-2020-12 validator; val/test schemas hash-excluded (decontaminated)
- **Eval:** native (unconstrained, greedy) schema-compliance, scored with the
  Draft-2020-12 validator + full format checker

## Results (held-out, measured identically base vs fine-tuned)

Measured on 150–200 held-out schemas the model never trained on. Fine-tuned
result independently cross-checked via the merged fp16 model (74% nested).

| Metric | Base Qwen3-1.7B | **Fine-tuned** | Δ |
|---|---|---|---|
| Nested / `$ref` schemas | 58% | **72–74%** | **+14–16** |
| Flat schemas | 87% | **92%** | +5 |
| Overall | 72.5% | **82–83%** | +10 |
| Parse-fail rate | 4% | **2.5–4%** | ~flat |

## Official results — JSONSchemaBench test split (n=360, native/unconstrained)

Base Qwen3-1.7B vs fine-tuned, identical prompt + Draft-2020-12 validator, greedy.

| Dataset | Base | Fine-tuned | Gain |
|---|---|---|---|
| Github_trivial | 49% | **82%** | +33% |
| Github_easy | 78% | **93%** | +16% |
| Github_medium | 42% | **56%** | +13% |
| Github_hard | 11% | **24%** | +13% |
| Glaiveai2K | 98% | **96%** | -2% |
| Kubernetes | 53% | **67%** | +13% |
| Snowplow | 33% | **73%** | +40% |
| JsonSchemaStore | 31% | **42%** | +11% |
| **Overall** | 49% | **67%** | +17% |

**McNemar paired test:** 76 schemas fixed, 14 broke, p=1.78e-11 (highly significant).

Our base measurement aligns with the published Llama-3.2-1B native numbers
(confirming comparability); the fine-tune beats published small-model native
compliance on nearly every dataset.

## Why this matters

Small models are widely deployed for narrow, high-volume, structured tasks —
but their published *native* schema-compliance collapses on complex schemas
(JSONSchemaBench reports ~13–38% for small models on the hardest datasets).
This fine-tune lifts native compliance on exactly that hard regime, so the
model is more usable **without** attaching a constrained-decoding engine
(and better as a base *with* one).

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, torch

m = AutoModelForCausalLM.from_pretrained("ujwal00/qwen3-1.7b-json", dtype=torch.float16, device_map="auto")
tok = AutoTokenizer.from_pretrained("ujwal00/qwen3-1.7b-json")

SYSTEM = ("You are a precise JSON generator. Given a JSON Schema, output ONE JSON value "
          "that strictly validates against it. Output ONLY the JSON value.")
schema = {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}
msgs = [{"role":"system","content":SYSTEM},
        {"role":"user","content":f"Generate a JSON value that validates against this JSON Schema:\n\n{json.dumps(schema)}\n\nReturn only the JSON value."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(m.device)
out = m.generate(ids, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

**Low-VRAM machines** (e.g. a 6 GB laptop GPU): add `low_cpu_mem_usage=True` to
`from_pretrained` (and optionally `max_memory={0: "5GiB", "cpu": "6GiB"}`) so
loading doesn't exhaust memory. On a normal 8 GB+ GPU the snippet above works as-is.

### Verified live output

| Schema | Model output | Valid? |
|---|---|---|
| `{name:str, age:int}` (required both) | `{"name": "sample_value", "age": 50}` | ✅ |
| nested: user{id,email:email-format, roles:enum[]} | `{"user": {"id": 50, "email": "user@example.com", "roles": ["user"]}}` | ✅ |

## Limitations (stated honestly)

- Numbers are **native / unconstrained**; constrained decoding scores differently and is out of scope.
- Trained on JSONSchemaBench-style schemas; may generalize less to very different schema dialects.
- Ultra-large schemas (>4000 chars) were excluded from training data.
- Reported gains are on held-out gold; the official test-split table is the definitive comparison.

## Reproducibility

Data recipe, generator, validator, training + eval code, and manifest are in the
project repo (`jsonc/`, `kaggle/`). Seed=42; prompt hashed; validator format
self-tested. Base and fine-tuned measured with identical prompt/validator/greedy.
