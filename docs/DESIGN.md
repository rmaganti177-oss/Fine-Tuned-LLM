# DESIGN — Industry-Grade JSON-Schema-Compliance Fine-Tune

*Produced by a 4-lens design + adversarial-review + synthesis workflow. Every rule traces to a documented failure from the prior function-calling project (see STATUS.md) or a reviewer-caught flaw.*

## Executive Summary

Fine-tune a small Qwen3 (primary 1.7B, fallback 4B-Instruct-2507) with Unsloth QLoRA so it emits JSON that NATIVELY (unconstrained, no grammar-constrained decoding) complies with a requested JSON Schema, measured on the public JSONSchemaBench (guidance-ai, ~9,558 schemas). The whole design is organized around one non-negotiable idea, drawn from our documented failure #1: the number we optimize and select on is GENERATION-based schema-compliance on a schema-disjoint gold held-out set, computed by an in-run callback at every checkpoint and streamed live to Weights & Biases — never eval_loss. Loss is demoted to a vanity curve. The reviewers exposed a fatal flaw in the original data plan: hypothesis-jsonschema only supports Draft-07 and does not resolve recursive $ref, so it would silently fail on exactly the nested/$defs schemas that are the entire point (documented headroom jumps from ~2-3% non-compliance on flat to ~68-69% on $defs). We resolve this decisively: we OWN a deterministic constructive generator for the nested case (walks $defs/$ref/anyOf/oneOf with a bounded-depth recursion terminator), use hypothesis-jsonschema and jsf only as best-effort coverage boosters, and gate EVERY instance through a single Draft2020-12 validator with the full format-backend that byte-for-byte matches the benchmark's checker — so our internal metric cannot diverge from the leaderboard metric for tooling reasons. Second decisive fix: the JSONSchemaBench prompt is fetched and pinned from the frozen harness BEFORE any data is built (it is an upstream hard gate, not a parenthetical), the exact prompt string + few-shot structure is hashed into the manifest, and training renders through that same format to kill train/serve skew. We report the leaderboard-comparable number the benchmark's own way (per-subset / per field-count tier, native pass@1), and we additionally report the flat-vs-nested diagnostic the GOAL asks for. Nested compliance is the headline the write-up is built on. Everything the user wants tracked is tracked: because Kaggle is headless with no live logs, W&B is our only live window, so the callback logs compliance_flat / compliance_nested / hallucinated_key_rate / schema_echo_rate / parse_fail_rate / required_missing_rate plus a sample table at every checkpoint, mirrored to stdout (survives the post-run log download) and to a durable JSON sidecar in the Kaggle output. No numbers are promised — only the method and honest measurement. Nothing is published until the user approves after seeing numbers.

## North-Star Metric

Native (unconstrained, no grammar-constrained decoding) pass@1 JSON-Schema COMPLIANCE = fraction of generated outputs that parse AND validate under a single Draft2020-12 validator with the full format-checker, measured by the FROZEN JSONSchemaBench harness on the untouched TEST split, aggregated the benchmark's way (per-subset and per field-count tier) AND sliced our diagnostic way (FLAT = no $defs/$ref and max_depth<=1 vs NESTED = has $defs/$ref or max_depth>=2). The headline is the improvement on the NESTED/$defs slice (base vs fine-tuned on identical harness), since that is where the ~68-69% headroom lives. In-run proxy that drives model selection and early-stop: gold/compliance_nested on a 200-schema schema-disjoint gold subset, greedy, computed every checkpoint. Secondary quality signals (all reported, none the headline): hallucinated-keys-per-sample, schema-echo-rate, parse-fail-rate, required-missing-rate.

## Data Plan

1. GENERATOR DECISION (resolves the #1 reviewer flaw): do NOT depend on hypothesis-jsonschema for nested schemas — it is Draft-07-only, maintenance-inactive, and does not resolve recursive $ref, so it would silently drop the $defs/recursive schemas that are the entire target. Build jsonc/construct.py: a deterministic constructive generator that resolves $ref by JSON-pointer lookup within $defs, handles allOf(merge)/anyOf|oneOf(pick first satisfiable branch)/enum|const(pick member)/prefixItems+items(tuple)/additionalProperties, and terminates recursion with a bounded-depth policy (at max_depth choose the non-recursive branch or omit the optional recursive property). Use hypothesis-jsonschema (Draft-07 subset) and jsf as SUPPLEMENTARY coverage generators only. For every schema, try constructive -> jsf -> hypothesis, keep the FIRST instance that passes the gate. Log per-complexity-class yield in the manifest; if nested yield is low the constructive path (which we own) is the backstop, so nested is never systematically filtered out.
2. VALIDATION GATE = the single source of truth, matched to the benchmark (resolves reviewer flaws #2 and #8): every candidate validates through ONE jsonschema.Draft202012Validator(schema, format_checker=draft202012_format_checker).is_valid(inst) regardless of the schema's declared $schema, because JSONSchemaBench validates everything as 2020-12 with formats ON. Pin AND verify the format backends (fqdn, idna, rfc3339-validator, rfc3987/rfc3986, isoduration, jsonpointer, uri-template, webcolors); add a startup self-test that feeds a known-bad email/date-time/uuid and ASSERTS the checker rejects them, so a missing backend can never let format-invalid data pass the gate silently. Any AssertionError -> drop + increment a typed counter; never silently keep.
3. PROMPT IS FETCHED, NOT GUESSED (resolves reviewer flaw #3; upstream hard gate BEFORE data-gen): clone the frozen JSONSchemaBench harness, read its actual generation prompt (few-shot vs zero-shot, system text, schema serialization), and pin the exact string + shot structure into jsonc/prompt.py with PROMPT_HASH = sha256 of the rendered template. Data-gen and eval import the SAME render function. Training renders through this exact format so there is zero train/serve skew. For Qwen3 set enable_thinking=False in apply_chat_template so no <think> blocks leak into output (they would tank compliance and inflate parse-fail).
4. REPORTING AXIS (resolves reviewer flaw #4): headline aggregation is the benchmark's own — per-subset (Github_trivial/easy/medium/hard/ultra, Glaiveai2K, JsonSchemaStore, Kubernetes, Snowplow, WashingtonPost) and per field-count tier. FLAT vs NESTED is reported as an ADDITIONAL diagnostic (this is what the GOAL asks to split by), never labeled 'the leaderboard number'. schema_tagger.py computes has_defs, num_refs, is_recursive, max_depth, has_oneOf/anyOf/allOf, has_enum/const, has_array/prefixItems, additionalProperties==False, required_count/property_count, formats-present.
5. TRAINING MIX inverted toward failure region: target ~70% NESTED / ~30% FLAT (deliberately NOT the benchmark's natural ratio, because we train where the model fails). Sub-quotas are TARGETS with logged shortfall, NOT blocking asserts (resolves reviewer flaw #7): aim >=3k additionalProperties:false, >=2k oneOf/anyOf, >=1k recursive/deep(depth>=3) sourced from the constructive path, >=1.5k tuple/prefixItems, >=2k enum/const, broad format coverage. Cap any single subset at <=25% to avoid domain monoculture; add ~5% simple flat JsonSchemaStore schemas as a breadth anchor (analog of the fc-project 8% general-instruct mix).
6. TRUE MINIMAL-PAIR TWINS (resolves reviewer flaw #5): do NOT draw a fresh instance for S'. Generate ONE base instance for schema S, derive S' by exactly one minimal structural edit (optional->required, flip additionalProperties, shrink an enum, swap a leaf type), then mutate ONLY the field that edit forces and keep every other value byte-identical; re-validate against S'. Both are SFT positives. Near-identical inputs with a single forced output difference is what teaches schema-reading; random value noise is banned. Budget ~1.5-2k twin pairs.
7. VALUE REALISM over property-based-testing degeneracy (resolves reviewer flaw #6): the constructive generator emits sane non-degenerate values by default; post-fill ALL string/number leaves (not just format-typed ones) with Faker/type-aware realistic values, then RE-VALIDATE against the same 2020-12+format checker. Bias toward richer variants: arrays at a sane length within [minItems,maxItems], objects with optionals populated. Optional-field variants A/B/C (required-only, required+~50% optionals, all-properties) are produced by the CONSTRUCTIVE path where we control inclusion — not by post-pruning hypothesis output (which would violate minProperties/dependentRequired/if-then).
8. STRUCTURAL DECONTAMINATION: blocklist = union over ALL val+test splits of exact_hash (sha256 of canonical key-sorted schema) AND struct_hash (same after stripping cosmetics {title,description,$comment,examples,default,$id,$schema}); drop any train candidate whose exact_hash OR struct_hash hits the blocklist. Then MinHash-LSH (datasketch, 128 perms) on the structural signature (sorted (json_pointer,type) tuples), drop train schemas with Jaccard>=0.9 vs any val/test schema. Intra-train dedup by struct_hash to prevent over-representation. TEST split is NEVER read except for the final one-shot eval. Every removal count logged to manifest.
9. SIZE/SPLITS: up to 3 distinct valid instances per surviving schema (dedup by sha256), ~20,000 SFT examples (<=25k ceiling to fit 1-2 epochs inside the 12h wall — fc-project ran 18,750 ex/2282 steps/1 epoch comfortably). Internal train/val = schema-disjoint 95/5 (~1k val, loss-curve only, explicitly demoted). Ultra-tier schemas that exceed max_seq_length are excluded from TRAINING (documented) but still evaluated by the harness as-is for honesty.
10. GOLD HELD-OUT (the in-run truth signal): gold_eval.jsonl = 512 schemas sampled from the JSONSchemaBench VAL split (never in train, never the final test), stratified ~256 nested / ~256 flat with sub-quotas for $defs, oneOf/anyOf, additionalProperties:false, recursive. Each row {schema, schema_id, complexity_class, tags}. The callback uses a fixed stratified 200-schema head every checkpoint; the full 512 runs once at end-of-run.
11. OPTIONAL DPO CORPUS for a later RL cycle (SFT is primary per the 79.5->91.7 precedent): make_negatives.py mutates a VALID instance into a plausible-but-INVALID one, but with PER-SCHEMA APPLICABILITY CHECKS (resolves reviewer flaw #9): only 'drop required' when required is non-empty, only 'hallucinate key' when additionalProperties:false, only 'violate enum' when an enum exists, etc. ASSERT each rejected FAILS the 2020-12 gate; skip inapplicable mutations and log coverage. Ship dpo_pairs.jsonl separately; never mixed into SFT.
12. ARTIFACTS: jsonc/ = prompt.py (shared render + PROMPT_HASH), schema_tagger.py, construct.py (owned generator), generate_instances.py (ensemble draw + gate + Faker + re-validate), decontaminate.py, build_dataset.py, make_negatives.py, gold_eval.py, validate.py (the ONE checker + json extractor + hallucinated-key + schema-echo helpers), smoke.py, callback.py. Outputs: sft_train.jsonl, sft_val.jsonl (rows {messages, complexity_class, tags, schema_id, instance_hash}), gold_eval.jsonl, dpo_pairs.jsonl, manifest.json. Pinned CPU data-gen env, isolated from the GPU env so it cannot perturb train/vllm pins.

## Training Plan

1. MODEL: Unsloth QLoRA (4-bit NF4) on Qwen3-1.7B (primary). At Step 0 confirm the EXACT HF id (Qwen3-1.7B and whether an Instruct/-2507 non-thinking variant exists) and always call apply_chat_template with enable_thinking=False. Fallback Qwen3-4B-Instruct-2507 only if the measured baseline shows the 1.7B lacks capacity (defined numerically in the model-selection step). Load base in a way that lets us later merge in TRUE fp16 (Lesson 2).
2. LoRA config held CONSTANT across all cycles (data is the only variable — fc-project hard rule): r=16, lora_alpha=32, lora_dropout=0, bias='none', target_modules=[q,k,v,o,gate,up,down]_proj, use_gradient_checkpointing='unsloth', use_rslora=False, random_state=42.
3. OPTIMIZER/SCHEDULE: lr=2e-4 (proven at 4B; fine for 1.7B), cosine, warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0, adamw_8bit, seed=42. Epochs: up to 2 with best-checkpoint-by-nested-compliance selection (NOT loss); the callback can early-stop. fc-project did 1 epoch (2282 steps) comfortably in the wall; 1.7B is smaller so 2 epochs fit.
4. BATCH/LENGTH: max_seq_length=3072 (schemas are long; ultra excluded from training). For 1.7B use per_device_train_batch_size=8 x grad_accum=2 x 2 GPUs -> effective ~32 (bump vs the 4B run's effective 8); re-tune down if OOM. Keep gpu_memory_utilization headroom — fc-project hit OOM at 90% util, so target <=0.85 anywhere vLLM/generation runs.
5. PACKING OFF (Lesson 3): no padding-free packing. Completions are single short JSON values ending in EOS with nothing after the closing brace. Use Unsloth train_on_responses_only to mask the prompt/schema tokens so loss is only on the JSON. smoke.py asserts the kept span equals exactly the JSON.
6. REGULARIZATION: neftune_noise_alpha=5 (helped robustness in fc-v3) — included but flagged as the one ablatable knob; if a cycle isolates its effect, that is a separate run.
7. CHECKPOINTS: evaluation_strategy='steps', eval_steps == save_steps (align callback with checkpoints), cadence ~every 150-200 steps -> ~7-8 in-run readings (vs the fc-project's single blind 4.5h shot). LoRA adapter checkpoints are tiny (tens of MB), so save_total_limit is large enough to KEEP ALL checkpoints; we merge the best-by-nested-compliance one offline. load_best_model_at_end=False (we select manually via best_by_nested.json).
8. STABILITY: T4 has NO bf16 -> train in fp16. fc-project logged an occasional NaN grad_norm at fp16; keep max_grad_norm=1.0 and watch grad_norm in W&B; if NaN recurs, lower lr to 1.5e-4. Set report_to='wandb'. The GoldComplianceCallback is wrapped so any callback exception logs and CONTINUES training (never kills a paid-quota run).
9. RESUMABILITY: 12h session cap — checkpoint every ~150-200 steps and support --resume_from_checkpoint so a killed session restarts from the last adapter; but the ~20k/2-epoch run should finish inside one session for the 1.7B.
10. SMOKE TRAIN before the real run (Lesson 5): a 50-step micro-run on 200 examples that exercises the FULL GPU pipeline — data load, template render, train_on_responses_only masking, ONE callback generation pass, checkpoint save — and must print one [GOLD step=50] line to W&B and stdout before we commit the full run.

## Observability Plan (the centerpiece)

1. W&B IS THE LIVE WINDOW: Kaggle is headless with NO mid-run log streaming, so W&B (WANDB_API_KEY as a Kaggle Secret) is the ONLY way to see a run while it is alive. report_to='wandb'; run name = {model}-{git_sha7}-{manifest_hash7}-{config_hash7} so every run is traceable to exact data+code+config. Log config: all hyperparams, PROMPT_HASH, manifest_hash, seed, library versions.
2. PER-STEP (logging_steps=10): train loss, grad_norm (watch for fp16 NaN), lr, tokens/sec, and W&B system metrics (GPU util/mem on both T4s) to catch stalls and the fc-project OOM-at-90% failure early.
3. PER-CHECKPOINT (the centerpiece — GoldComplianceCallback.on_evaluate): native/unconstrained greedy generation on the fixed 200-schema gold subset, then log gold/compliance_all, gold/compliance_flat, gold/compliance_nested (the SELECTION metric), gold/parse_fail_rate, gold/hallucinated_keys_per_sample, gold/schema_echo_rate, gold/required_missing_rate, gold/gen_seconds, and gold/prompt_hash (so format drift is visible, not silent).
4. SAMPLE TABLE: at every checkpoint log a wandb.Table of ~25 rows (schema_id, output_head, verdict/validator-error) so the user can EYEBALL what the model actually emits — this is how we catch schema-echo, fenced-code, thinking-block leakage, and truncation qualitatively, not just as a rate.
5. TRIPLE-DURABILITY of every eval number (Lesson 6 — a stale downloaded log once caused a false 'it pushed' conclusion): (1) W&B live, (2) a stdout line '[GOLD step=N] all=.. flat=.. nested=.. parsefail=.. halluc=.. echo=..' that survives the post-run Kaggle log download, (3) a durable JSON sidecar gold_step{N}.json written to the Kaggle /working output. All three are re-derived from FRESHLY generated outputs each time, never a cached artifact.
6. SELECTION STATE: the callback maintains best_nested / best_step, writes best_by_nested.json, and sets wandb.run.summary['best_nested_compliance'/'best_step'] so the winning checkpoint is unambiguous after the run without re-reading logs.
7. ALERTS: wandb.alert on (a) fp16 loss/grad NaN, (b) compliance collapse — nested drops >0.15 below running best after 40% of training (patience window to avoid early noise). Because there is no live control channel, the callback may optionally set control.should_training_stop=True on a sustained collapse to save quota (circuit breaker).
8. HEARTBEAT: a per-logging-step stdout line with step/max_steps and ETA, so even the raw Kaggle log (readable only post-run) tells us exactly how far a killed session got.
9. DASHBOARD: a saved W&B workspace panel layout with compliance_nested and compliance_flat as the top two large panels (loss deliberately small and lower) — reinforcing that the real metric, not loss, is what we watch. This becomes the public W&B report at release.

## Evaluation Plan

1. METRIC DEFINITIONS (precise, matched to benchmark): COMPLIANCE = output parses (after fenced-block strip + first-balanced-JSON extraction) AND validates under the single Draft2020-12 validator + full format backend. Also report COVERAGE (JSONSchemaBench's term for the same pass@1 idea) so numbers line up with the paper/leaderboard vocabulary. Secondary: hallucinated-keys-per-sample, schema-echo-rate, parse-fail-rate, required-missing-rate.
2. THREE-TIER EVAL to fix the fc-project 'slow eval, one blind shot' problem: (1) IN-RUN 200-schema gold subset, greedy, every checkpoint (~2-4 min each on 1.7B); (2) END-OF-RUN full 512 gold set once; (3) FINAL leaderboard-comparable = frozen JSONSchemaBench harness on the untouched TEST split, run ONCE, with the harness's EXACT prompt and sampling params.
3. NATIVE ONLY: no grammar-constrained decoding anywhere in the reported number (the GOAL's explicit requirement). Our controlled in-run/gold numbers use greedy (temperature=0) for stability and cross-checkpoint comparability; the final harness run mirrors the harness's own sampling params exactly — we report both our-prompt and harness-prompt numbers so any gap is attributable to prompt, not model.
4. BASE vs FINE-TUNED on the IDENTICAL frozen harness (fc-project pattern): report per-subset and per field-count tier, plus the flat/nested diagnostic. Headline = the nested/$defs improvement.
5. PAIRED SIGNIFICANCE: per-subset McNemar (base vs fine-tuned on the same schemas), as done for the fc-project (McNemar p reported). No single cherry-picked number.
6. FAILURE ANALYSIS: read ~20-30 residual failures on nested schemas, categorize (missing required / hallucinated key / wrong nesting / type error / enum violation / schema-echo / parse-fail / truncation), and put the table in the write-up — this is what turns a training log into a contribution.
7. vLLM eval config for T4 (Lesson 5 — avoid the 262k KV blowup and OOM): dtype=float16, max_model_len=8192 (JSON task, not the model's max), gpu_memory_utilization<=0.85, max_num_seqs tuned down, enforce_eager if memory-tight; tensor_parallel_size=1 for the 1.7B on a single T4.
8. OPTIONAL ABLATION if quota allows: retrain without the nested-inversion (natural ratio) to demonstrate WHY nested compliance moved — the single cleanest evidence for the write-up.
9. REGRESSION GUARD: confirm fine-tuning did not wreck FLAT compliance (must stay >= base flat) while lifting nested — report both so the trade-off is explicit.

## Merge & Behavioral Gate

1. MERGE IN TRUE fp16 (Lesson 2, the exact bug that silently corrupted the fc-project): load the base in fp16 (NOT bnb-4bit-dequantized) and merge the LoRA into it — Unsloth save_pretrained_merged(save_method='merged_16bit') or an explicit fp16 base + merge_and_unload. Add an assertion that the base dtype at merge time is float16 and NOT a 4-bit-quantized module.
2. ADAPTER-ON-CLEAN-BASE MICRO-TEST (the fc-project microtest.json pattern that caught the corruption): generate on a fixed ~20-schema battery with BOTH (a) base fp16 + adapter un-merged and (b) the merged fp16 model. Assert outputs match and neither shows corruption signatures: extra_brace, unclosed structure, any tokens after the final closing brace, or malformed decode. Any divergence => the merge corrupted; do not ship, investigate.
3. BEHAVIORAL GATE 2.0 (format + behavior thresholds, run via vLLM fp16 on a 128-schema battery before anything is called shippable): require parse_fail_rate within 1pt of the un-merged adapter, schema_echo_rate <= 2%, ZERO outputs with trailing tokens after the final brace, and merged nested-compliance within 1pt of the un-merged adapter's nested-compliance. Gate is a hard blocker: fail => no ship.
4. SANITY on the format contract: assert PROMPT_HASH used at merge/eval equals the one baked into the dataset manifest — the format cannot silently drift between train and serve.
5. NAMING/PROVENANCE: merged model card records base id, adapter run (W&B url), manifest_hash, PROMPT_HASH, exact eval harness version, and the best_step it was selected from — so the shipped artifact is fully traceable.
6. The gate reuses the SAME validate.py checker as data-gen and eval, so a green gate genuinely predicts the harness result rather than a parallel implementation that could disagree.

## Iteration Protocol

1. ONE VARIABLE PER CYCLE: the training recipe (LoRA/optimizer/schedule) is frozen; DATA is the only thing that changes between cycles (fc-project hard rule that made results interpretable).
2. CHEAP-BASELINE-FIRST GATE before committing a model: measure BOTH Qwen3-1.7B and Qwen3-4B-Instruct-2507 native greedy on the 200-schema gold subset (~30-45 min total). Pick 1.7B unless it 'lacks capacity', defined numerically: 1.7B chosen if its baseline nested-compliance is within ~10 absolute points of 4B AND it reliably emits parseable JSON (parse_fail < ~10%); otherwise fall back to 4B. Record both baselines as the improvement denominator.
3. FAST-THEN-FULL: iterate using the 200-schema gold subset (minutes); only promote a checkpoint to the full 512 gold set, and only promote a winner to the frozen-harness TEST run (the expensive, once-only measurement).
4. CYCLE 1 = SFT (primary, per the published 79.5->91.7 precedent). Decision gate: if end-of-run gold nested-compliance clearly beats base and the behavioral gate is green, proceed to full eval; else do a DATA cycle (adjust mix / hard-case quotas / value realism), not a recipe cycle.
5. CYCLE 2 (optional, only if SFT plateaus below expectations) = DPO on dpo_pairs.jsonl (true-negative-asserted, per-schema-applicable mutations) or schema-verifier RL — a separate run, still one-variable.
6. NEVER assume, always verify (Lesson 6): every 'it improved' / 'it pushed' claim is confirmed from a freshly generated number, not a downloaded/cached log. The stale-log false positive from the fc-project is explicitly guarded against by the triple-durability logging.
7. PROMISE THE METHOD, NOT NUMBERS (Lesson 7): between cycles we report measured deltas with paired stats, and we do not assert a target compliance figure.

## Reproducibility & Release

1. TWO ISOLATED PINNED ENVS: (1) CPU data-gen env (hypothesis-jsonschema==0.23.1, jsonschema[format]==4.23.0 + verified format backends, jsf==0.11.2, Faker==30.*, datasketch==1.6.5, datasets==2.*) — cannot touch the GPU env; (2) GPU env with unsloth, transformers<5 (vllm compat, the fc-project pin), torch/CUDA for T4, vllm fp16. Pin exact versions to avoid the fc-project env-drift class (transformers v5 vs vllm, stale torchao, numpy binary clash, tree-sitter API).
2. manifest.json is the reproducibility contract: library versions, global seed=42, PROMPT_HASH, harness commit, per-subset/per-complexity/per-tier counts, twin-pair count, generator-yield-by-class, decontam removals (exact/struct/minhash), instances-per-schema histogram, skipped-schema reasons. The generated JSONL IS the reproducible artifact (not the RNG state) — this resolves the reviewer's point that Hypothesis draws are not stable across versions.
3. SEED everything (data-gen, sampling, torch, transformers, vLLM) = 42; greedy decoding for the controlled numbers removes sampling variance entirely.
4. NOTHING PUBLISHED UNTIL THE USER APPROVES after seeing numbers (hard constraint). Artifacts stay private (HF private repo) until then.
5. RELEASE BUNDLE on approval: (a) merged fp16 model on HF with a card = eval table (both prompts, per-subset, per-tier, flat/nested, base-vs-ours, McNemar p), training recipe, limitations (native-only claim, ultra-tier excluded from training, exact harness version), how-to-run; (b) the jsonc/ data-recipe code + manifest; (c) public W&B report with the compliance-nested dashboard; (d) gold_eval.jsonl for reproducibility; (e) stretch: submission/PR to the JSONSchemaBench leaderboard.
6. LIMITATIONS stated honestly: numbers are native/unconstrained (constrained decoding would score differently and is out of scope); TEST measured once; ultra schemas excluded from training and reported as-is.

## Risk Register

1. GENERATOR CANNOT COVER NESTED (highest risk, reviewer #1): mitigated by the owned constructive generator as the nested backstop + manifest yield-by-class monitoring. TRIGGER/ACTION: if nested yield < target after Step 2, extend construct.py before proceeding — do NOT let nested get silently filtered, which would re-invert the mix to flat-heavy and defeat the project.
2. METRIC DIVERGENCE (Lesson 1): in-run gold compliance disagreeing with the harness. Mitigated by using ONE 2020-12+format checker matched to the benchmark, the SAME prompt module (PROMPT_HASH asserted equal), and reporting both prompt variants. ACTION: if gold and harness diverge >a few pts, diff the prompt/format before trusting either.
3. MERGE CORRUPTION (Lesson 2): mitigated by fp16 merge + adapter-on-clean-base micro-test + Behavioral Gate 2.0 (zero trailing-token tolerance).
4. LABEL MISALIGNMENT (Lesson 3): packing OFF, EOS-terminated single JSON, smoke-asserted kept span. Low residual risk.
5. EASY/NOISY NEGATIVES & TWINS (Lesson 4/reviewer #5): true minimal-pair twins (mutate only the forced field), per-schema-applicable DPO mutations, no random mismatches.
6. ENV DRIFT (Lesson 5): two isolated pinned envs, transformers<5 for vllm, T4 fp16 (no bf16), max_model_len=8192 (no 262k KV blowup), gpu_util<=0.85 (no OOM-at-90%), smoke train before full run.
7. BLIND MID-RUN / STALE LOG (Lesson 6): W&B live window + triple-durability (W&B/stdout/sidecar), all numbers re-derived from fresh generations, heartbeat+ETA, verify-never-assume.
8. FORMAT-BACKEND SILENT PASS (reviewer #8): startup self-test that asserts bad values are rejected; build fails if a backend is missing.
9. PROMPT SKEW (reviewer #3): prompt fetched from harness at Step 0, hashed, and used for both training and eval; enable_thinking=False so Qwen3 does not leak <think> blocks.
10. CALLBACK COST/STABILITY: greedy 200-schema gen adds ~2-4 min/checkpoint (~25-30 min total) — acceptable; callback wrapped in try/except so it can never kill a paid-quota run; torch.cuda.empty_cache() and small gen batch to avoid OOM.
11. OVER-PROMISING (Lesson 7): deliverable is the method + honest measurement; no target compliance number asserted; TEST measured once; native-only and ultra-excluded caveats stated up front.

## Step-by-Step Execution Order

1. STEP 0 (NOW, no GPU) — PROMPT + HARNESS PIN: clone the frozen JSONSchemaBench harness, read its exact generation prompt and sampling params, write jsonc/prompt.py (shared render + PROMPT_HASH), and pin the harness commit. This is the upstream gate — no data is built until the real prompt is known.
2. STEP 1 (no GPU) — VALIDATOR + FORMAT SELF-TEST: write jsonc/validate.py (single Draft2020-12 validator + format backend, JSON extractor, hallucinated-key counter, schema-echo detector). Run the format self-test that asserts a bad email/date-time/uuid are REJECTED; fail the build if any backend is missing.
3. STEP 2 (no GPU) — GENERATOR: write jsonc/construct.py (owned constructive generator with bounded-depth recursion + $ref resolution) and generate_instances.py (construct -> jsf -> hypothesis ensemble, gate every instance, Faker realism post-fill, re-validate). Verify on 10 hand-picked $defs/recursive/oneOf schemas that the constructive path actually yields validating nested instances.
4. STEP 3 (no GPU) — TAG + DECONTAMINATE + BUILD: run schema_tagger.py, decontaminate.py (exact/struct hash + MinHash vs val/test), build_dataset.py (70/30 nested/flat, variants A/B/C, true minimal-pair twins, dedup, schema-disjoint 95/5). Write sft_train.jsonl, sft_val.jsonl, manifest.json. Build gold_eval.jsonl (512 from VAL, stratified).
5. STEP 4 (no GPU) — SMOKE.PY (Lesson 5): run the full pipeline on 20 schemas (flat + $defs + oneOf + recursive), render 10 examples through the EXACT tokenizer.apply_chat_template (enable_thinking=False), assert every completion re-validates, the train_on_responses_only kept span == exactly the JSON, and no example exceeds max_seq_length. Print 5 rendered examples to READ. Unit-test callback.py OFFLINE by feeding canned generations so metric math is verified before any GPU.
6. STEP 5 (moment GPU quota resets) — MODEL-SELECTION BASELINE (~30-45 min): load Qwen3-1.7B and Qwen3-4B-Instruct-2507, run native greedy on the 200-schema gold subset via the callback's scoring path, log both to W&B. Apply the numeric pick rule (1.7B unless it lacks capacity). This is the cheapest decisive spend and it also proves the eval path works on real GPU before training.
7. STEP 6 (GPU) — SMOKE TRAIN: 50-step micro-run on 200 examples that exercises data load, masking, ONE callback pass, and a checkpoint save; require one [GOLD step=50] line in W&B + stdout + sidecar. Only proceed if green.
8. STEP 7 (GPU) — FULL SFT RUN: the ~20k-example run with GoldComplianceCallback every ~150-200 steps, report_to='wandb', all durability sinks on. Watch the W&B compliance_nested panel live. Let best-by-nested selection + circuit breaker manage it.
9. STEP 8 (GPU) — SELECT + MERGE + GATE: read best_by_nested.json, merge that checkpoint in TRUE fp16, run the adapter-on-clean-base micro-test and Behavioral Gate 2.0. Hard-block on failure.
10. STEP 9 (GPU) — EVAL: full 512 gold set, then the frozen harness on the TEST split ONCE (harness prompt + our prompt), base vs fine-tuned, per-subset/per-tier + flat/nested, McNemar. Failure analysis on ~20-30 nested residuals.
11. STEP 10 (no GPU) — PRESENT NUMBERS to the user; on approval, run the release bundle. STEP 11 (optional) — DPO/schema-RL cycle 2 if SFT plateaued.
12. QUOTA GUARDRAIL: Steps 0-4 and callback unit tests are 100% CPU/local and MUST be finished before quota resets so zero GPU-hours are spent on anything that could have been caught on CPU.

## In-Run Compliance Callback (pseudocode)

```python
# jsonc/callback.py — the "real metric during training" hook (the centerpiece).
# Runs GENERATION-based schema-compliance on a schema-disjoint gold subset,
# NATIVE (unconstrained, greedy) at every checkpoint. This — not eval_loss —
# selects the model, drives the W&B dashboard, and doubles as a merge-corruption
# tripwire. Directly answers failure #1: track the REAL downstream metric during
# training, measured on a held-out split it cannot memorize.

import json, os, time, torch, wandb
from unsloth import FastLanguageModel
from transformers import TrainerCallback
from jsonc.prompt   import render_messages, PROMPT_HASH          # the ONE shared format
from jsonc.validate import (validate_2020_12,      # SAME checker as data-gen + benchmark
                            extract_json,           # strip fences, take first balanced JSON
                            count_hallucinated_keys,
                            looks_like_schema_echo)

class GoldComplianceCallback(TrainerCallback):
    def __init__(self, tokenizer, gold_rows, out_dir,
                 n_fast=200, gen_batch=16, max_new_tokens=640):
        self.tok      = tokenizer
        self.fast     = gold_rows[:n_fast]      # fixed stratified head (~100 flat/100 nested)
        self.out_dir  = out_dir
        self.gen_batch, self.max_new = gen_batch, max_new_tokens
        self.best_nested, self.best_step = -1.0, None

    @torch.no_grad()
    def _generate(self, model, rows):
        prompts = [self.tok.apply_chat_template(
                       render_messages(r["schema"]),          # identical render to training
                       tokenize=False, add_generation_prompt=True,
                       enable_thinking=False)                 # Qwen3: NO <think> leakage
                   for r in rows]
        texts = []
        for i in range(0, len(prompts), self.gen_batch):
            chunk = prompts[i:i+self.gen_batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=3072).to(model.device)
            gen = model.generate(**enc,
                                  max_new_tokens=self.max_new,
                                  do_sample=False, temperature=None,   # GREEDY -> stable, comparable
                                  use_cache=True,
                                  pad_token_id=self.tok.eos_token_id)  # native: no logit processors/grammar
            new = gen[:, enc.input_ids.shape[1]:]
            texts += self.tok.batch_decode(new, skip_special_tokens=True)
        return texts

    def _score(self, rows, texts):
        m = dict(n=len(rows), flat_ok=0, flat_n=0, nested_ok=0, nested_n=0,
                 parse_fail=0, halluc=0, echo=0, req_missing=0)
        table = []
        for r, text in zip(rows, texts):
            nested = (r["complexity_class"] == "nested")
            m["nested_n" if nested else "flat_n"] += 1
            obj, ok_parse = extract_json(text)
            if not ok_parse:
                m["parse_fail"] += 1
                table.append([r["schema_id"], text[:180], "PARSE_FAIL"]); continue
            if looks_like_schema_echo(obj, r["schema"]): m["echo"] += 1
            m["halluc"] += count_hallucinated_keys(obj, r["schema"])
            valid, err = validate_2020_12(obj, r["schema"])   # Draft2020-12 + full formats
            if valid:
                m["nested_ok" if nested else "flat_ok"] += 1
                table.append([r["schema_id"], text[:180], "OK"])
            else:
                if err and "required" in err: m["req_missing"] += 1
                table.append([r["schema_id"], text[:180], (err or "INVALID")[:120]])
        return m, table

    def _emit(self, step, m, table, secs):
        flat   = m["flat_ok"]   / max(m["flat_n"], 1)
        nested = m["nested_ok"] / max(m["nested_n"], 1)      # <-- the SELECTION metric
        allc   = (m["flat_ok"] + m["nested_ok"]) / m["n"]
        row = {"gold/compliance_all": allc,
               "gold/compliance_flat": flat,
               "gold/compliance_nested": nested,
               "gold/parse_fail_rate": m["parse_fail"] / m["n"],
               "gold/hallucinated_keys_per_sample": m["halluc"] / m["n"],
               "gold/schema_echo_rate": m["echo"] / m["n"],
               "gold/required_missing_rate": m["req_missing"] / m["n"],
               "gold/gen_seconds": secs,
               "gold/prompt_hash": PROMPT_HASH}               # format-drift tripwire
        # (1) W&B — the only LIVE window on headless Kaggle:
        wandb.log(row, step=step)
        wandb.log({"gold/samples": wandb.Table(
            columns=["schema_id", "output_head", "verdict"], data=table[:25])}, step=step)
        # (2) stdout — survives the post-run Kaggle log download:
        print(f"[GOLD step={step}] all={allc:.3f} flat={flat:.3f} nested={nested:.3f} "
              f"parsefail={row['gold/parse_fail_rate']:.3f} "
              f"halluc={row['gold/hallucinated_keys_per_sample']:.3f} "
              f"echo={row['gold/schema_echo_rate']:.3f} ({secs:.0f}s)", flush=True)
        # (3) durable sidecar in /working — survives even if W&B drops:
        with open(os.path.join(self.out_dir, f"gold_step{step}.json"), "w") as f:
            json.dump({"step": step, **row}, f, indent=2)
        return nested

    def on_evaluate(self, args, state, control, **kw):
        model, step = kw["model"], state.global_step
        try:
            FastLanguageModel.for_inference(model)         # Unsloth fast native inference
            t0 = time.time()
            texts     = self._generate(model, self.fast)   # native / unconstrained / greedy
            m, table  = self._score(self.fast, texts)
            nested    = self._emit(step, m, table, time.time() - t0)
            if nested > self.best_nested:                  # select on nested compliance, NOT loss
                self.best_nested, self.best_step = nested, step
                with open(os.path.join(self.out_dir, "best_by_nested.json"), "w") as f:
                    json.dump({"best_step": step, "best_nested": nested}, f)
                wandb.run.summary["best_nested_compliance"] = nested
                wandb.run.summary["best_step"] = step
            # circuit breaker (patience: only after 40% of training):
            if step > 0.4 * state.max_steps and (self.best_nested - nested) > 0.15:
                wandb.alert(title="Compliance collapse",
                            text=f"nested {nested:.2f} << best {self.best_nested:.2f} @ {step}")
                # optional quota-saver: control.should_training_stop = True
        except Exception as e:                             # NEVER kill a paid-quota run
            print(f"[GOLD step={step}] CALLBACK ERROR (training continues): {e}", flush=True)
        finally:
            FastLanguageModel.for_training(model)          # ALWAYS restore training mode
            torch.cuda.empty_cache()
        return control

# Wiring: SFTTrainer(..., args=TrainingArguments(evaluation_strategy="steps",
#   eval_steps=SAVE_STEPS, save_steps=SAVE_STEPS, load_best_model_at_end=False,
#   report_to="wandb"), callbacks=[GoldComplianceCallback(tok, gold_rows, OUT)])
# eval_steps == save_steps aligns the compliance reading with each tiny LoRA checkpoint;
# all checkpoints are kept (adapters are ~tens of MB) and the best_by_nested one is
# merged in fp16 offline. eval_loss still logs, but it is the small panel, not the driver.
```
