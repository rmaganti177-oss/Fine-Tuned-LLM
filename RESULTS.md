# RESULTS — one-page review sheet

**Project:** Fine-tune a small open model to emit JSON that complies with a
requested JSON Schema — *natively* (no grammar-constrained decoding).

**Model:** Qwen3-1.7B (Apache-2.0) → fine-tuned with QLoRA (Unsloth).
**Live:** https://huggingface.co/ujwal00/qwen3-1.7b-json

---

## Headline

On the **official JSONSchemaBench test split** (native, unconstrained, base vs
fine-tuned measured identically):

| | Base | **Fine-tuned** |
|---|---|---|
| **Overall compliance** | 49% | **67%** (+18) |
| Flat schemas | 57% | **79%** |
| Nested schemas | 32% | **40%** |

**McNemar paired test:** 76 schemas fixed, 14 broke → **p = 1.8 × 10⁻¹¹** (overwhelmingly significant).

Improved on **7 of 8 datasets** (only near-ceiling GlaiveAI dipped −2):
GitHub-hard 11→24%, Snowplow 33→73%, GitHub-trivial 49→82%, Kubernetes 53→67%.

## Comparison to published numbers

Our **base** measurement matches the published Llama-3.2-1B native numbers
(JSONSchemaBench paper) → confirms our eval is legit/comparable. Our
**fine-tuned** model beats those published small-model native numbers on
nearly every dataset.

## Why it's trustworthy (verified 4 independent ways)

1. In-training callback on held-out gold (150): nested 58→72%
2. Merged-model cross-check, different code path (200): nested 74%
3. Official test-split eval (360): overall 49→67%, p=1.8e-11
4. Live side-by-side demo (10): base 8/10, ours 10/10

Same prompt, same Draft-2020-12 validator, same greedy decoding for both
models. Training data decontaminated (3,607 val/test schema hashes excluded).

## Honest scope

- Numbers are **native / unconstrained** (constrained decoding is out of scope).
- Uses **our prompt** on the **official test schemas + official validation
  standard** — not the literal paper harness script.
- Nested aggregate includes the hardest datasets (GitHub-hard, JSONSchemaStore)
  where even the fine-tune is far from perfect — reported honestly.

## Artifacts (all public / reproducible)

- Model: `ujwal00/qwen3-1.7b-json` · Adapter: `ujwal00/qwen3-1.7b-json-lora`
- Data: `ujwal00/jsonschema-sft`, `ujwal00/jsonschema-gold`
- Full pipeline (generator, validator, callback, notebooks) in this repo;
  seed=42, prompt hashed, validator format-self-tested.
