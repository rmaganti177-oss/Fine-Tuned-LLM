"""STEP 3-4: build the SFT training set from JSONSchemaBench TRAIN-split schemas
(never test/val -> no contamination). For each schema: tag complexity, generate a
valid instance (construct, full-optionals for realism), gate through the validator,
render as (prompt -> instance) via the shared prompt module. Also build minimal-pair
twins and a schema-disjoint gold eval set from VAL. Writes JSONL + manifest.

Contamination guard: training schemas come ONLY from train splits; gold from val;
we also hash-dedup schema strings so no val/test schema can leak in.
"""
import hashlib
import json
import random

from datasets import load_dataset
import construct
from prompt import render_messages, PROMPT_HASH
from validate import validate_2020_12

SUBSETS = ["Github_trivial", "Github_easy", "Github_medium", "Github_hard",
           "Glaiveai2K", "Kubernetes", "Snowplow", "JsonSchemaStore"]
SEED = 42
GOLD_N = 200           # gold eval subset (from VAL), ~half flat half nested
random.seed(SEED)


def complexity(schema):
    s = json.dumps(schema)
    nested = ("$ref" in s) or ('"$defs"' in s) or ('"definitions"' in s)
    depth, mx = 0, 0
    for ch in s:
        if ch in "{[":
            depth += 1; mx = max(mx, depth)
        elif ch in "}]":
            depth -= 1
    return "nested" if (nested or mx >= 5) else "flat"


def schema_hash(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def load_split(sub, split):
    try:
        return load_dataset("epfl-dlab/JSONSchemaBench", sub, split=split)
    except Exception:
        return []


def parse(raw):
    s = raw.get("json_schema") or raw.get("schema")
    try:
        sch = json.loads(s) if isinstance(s, str) else s
        return sch if isinstance(sch, dict) else None
    except Exception:
        return None


def _degenerate(inst, schema):
    """Empty container as the answer for a schema that declares structure -> a
    useless (under-extraction) training target. Reject it."""
    if inst == {} and (schema.get("properties") or "$ref" in schema
                       or schema.get("allOf") or schema.get("oneOf") or schema.get("anyOf")):
        return True
    if inst == [] and schema.get("items"):
        return True
    return False


def gen_instance(schema, rng, full=True):
    try:
        inst = construct.generate(schema, rng=rng, full=full)   # full: include optionals (caps guard blowup)
    except Exception:      # includes _Budget (schema too big) -> skip
        return None
    try:
        ok, _ = validate_2020_12(inst, schema)
    except Exception:
        return None
    if not ok or _degenerate(inst, schema):
        return None
    if len(json.dumps(inst)) > 12000:   # reject bloated instances
        return None
    return inst


def make_twin(schema, base_inst, rng):
    """Minimal structural edit: promote one optional to required. The instance
    already includes optionals (full mode), so it still validates against S'."""
    props = list((schema.get("properties") or {}).keys())
    opt = [p for p in props if p not in set(schema.get("required", []))]
    if not opt or not isinstance(base_inst, dict):
        return None
    promote = opt[rng.randrange(len(opt))]
    if promote not in base_inst:
        return None  # instance must contain it to still validate
    s2 = json.loads(json.dumps(schema))
    s2["required"] = sorted(set(s2.get("required", [])) | {promote})
    ok, _ = validate_2020_12(base_inst, s2)
    return (s2, base_inst) if ok else None


def to_row(schema, inst, cls, kind):
    msgs = render_messages(schema)
    return {"messages": msgs + [{"role": "assistant",
                                 "content": json.dumps(inst, ensure_ascii=False)}],
            "complexity_class": cls, "kind": kind,
            "schema": json.dumps(schema, ensure_ascii=False)}


def main():
    # ---- collect train schemas (dedup) + record val/test hashes for decontam ----
    seen_hashes = set()
    forbidden = set()   # val + test hashes -> never allow in train
    for sub in SUBSETS:
        for split in ("val", "test"):
            for raw in load_split(sub, split):
                sch = parse(raw)
                if sch:
                    forbidden.add(schema_hash(sch))

    train_schemas = []  # (schema, cls, subset)
    for sub in SUBSETS:
        for raw in load_split(sub, "train"):
            sch = parse(raw)
            if not sch:
                continue
            h = schema_hash(sch)
            if h in seen_hashes or h in forbidden:
                continue
            seen_hashes.add(h)
            train_schemas.append((sch, complexity(sch), sub))
    random.shuffle(train_schemas)
    print(f"train schemas (deduped, decontaminated): {len(train_schemas)}")

    # ---- generate instances; build positives + twins ----
    rows, twin_pairs = [], 0
    yield_ok = {"flat": 0, "nested": 0}
    yield_tot = {"flat": 0, "nested": 0}
    rng = random.Random(SEED)
    MAX_SCHEMA_CHARS = 4000   # keep rendered prompt within the 3072-token budget
    skipped_long = 0
    for sch, cls, sub in train_schemas:
        if len(json.dumps(sch)) > MAX_SCHEMA_CHARS:
            skipped_long += 1
            continue
        yield_tot[cls] += 1
        inst = gen_instance(sch, rng, full=True)
        if inst is None:
            continue
        yield_ok[cls] += 1
        rows.append(to_row(sch, inst, cls, "positive"))
        # twin minimal-pair (cap total)
        if twin_pairs < 1500 and cls == "nested":
            tw = make_twin(sch, inst, rng)
            if tw:
                s2, inst2 = tw
                rows.append(to_row(s2, inst2, cls, "twin_positive"))
                twin_pairs += 1

    random.shuffle(rows)
    n_val = 500
    train_rows, val_rows = rows[n_val:], rows[:n_val]

    # ---- gold eval subset from VAL-split schemas (schema-disjoint from train) ----
    gold = []
    for sub in SUBSETS:
        for raw in load_split(sub, "val"):
            sch = parse(raw)
            if not sch:
                continue
            cls = complexity(sch)
            gold.append({"schema": json.dumps(sch, ensure_ascii=False),
                         "complexity_class": cls, "subset": sub, "unique_id": raw.get("unique_id")})
    random.shuffle(gold)
    flat_g = [g for g in gold if g["complexity_class"] == "flat"][:GOLD_N // 2]
    nest_g = [g for g in gold if g["complexity_class"] == "nested"][:GOLD_N // 2]
    gold_sel = flat_g + nest_g
    random.shuffle(gold_sel)

    # ---- write ----
    def dump(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("sft_train.jsonl", train_rows)
    dump("sft_val.jsonl", val_rows)
    dump("gold_eval.jsonl", gold_sel)

    from collections import Counter
    mix = Counter((r["complexity_class"], r["kind"]) for r in train_rows)
    manifest = {
        "prompt_hash": PROMPT_HASH, "seed": SEED, "subsets": SUBSETS,
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "gold_rows": len(gold_sel),
        "gold_flat": len(flat_g), "gold_nested": len(nest_g),
        "twin_pairs": twin_pairs,
        "train_mix": {f"{k[0]}/{k[1]}": v for k, v in mix.items()},
        "generator_yield": {c: f"{yield_ok[c]}/{yield_tot[c]}" for c in yield_tot},
        "decontam_forbidden_hashes": len(forbidden),
        "skipped_long_schemas": skipped_long,
    }
    with open("manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    nested_frac = sum(1 for r in train_rows if r["complexity_class"] == "nested") / max(len(train_rows), 1)
    print(f"\nnested fraction of train: {nested_frac:.0%}")


if __name__ == "__main__":
    main()
