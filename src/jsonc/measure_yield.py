"""Decisive CPU experiment (Risk #1): does the constructive generator achieve
high instance-yield on the REAL JSONSchemaBench nested schemas? If nested yield
is low, the whole project is at risk (we'd train flat-heavy). Measures yield by
complexity class across subsets, before investing in the full dataset build.
"""
import json
import random
import sys

from datasets import load_dataset
import construct
from validate import validate_2020_12

SUBSETS = ["Github_easy", "Github_medium", "Github_hard", "Glaiveai2K",
           "Kubernetes", "Snowplow", "JsonSchemaStore", "Github_trivial"]
PER_SUBSET = 40


def complexity(schema):
    s = json.dumps(schema)
    nested = ("$ref" in s) or ('"$defs"' in s) or ('"definitions"' in s)
    # depth estimate
    depth, mx = 0, 0
    for ch in s:
        if ch in "{[":
            depth += 1
            mx = max(mx, depth)
        elif ch in "}]":
            depth -= 1
    if nested or mx >= 5:
        return "nested"
    return "flat"


def try_generate(schema):
    """construct -> validate. Returns (ok, instance_or_None)."""
    try:
        inst = construct.generate(schema, rng=random.Random(7))
    except Exception:
        return False, None
    ok, _ = validate_2020_12(inst, schema)
    return ok, (inst if ok else None)


def main():
    tally = {}
    examples = {"flat": None, "nested": None}
    for sub in SUBSETS:
        try:
            ds = load_dataset("epfl-dlab/JSONSchemaBench", sub, split="test")
        except Exception as e:
            try:
                ds = load_dataset("epfl-dlab/JSONSchemaBench", sub, split="train")
            except Exception:
                print(f"[skip {sub}] {str(e)[:80]}")
                continue
        n = min(PER_SUBSET, len(ds))
        idxs = random.Random(1).sample(range(len(ds)), n)
        for i in idxs:
            raw = ds[i]
            schema_str = raw.get("json_schema") or raw.get("schema")
            try:
                schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
            except Exception:
                continue
            if not isinstance(schema, dict):
                continue
            cls = complexity(schema)
            ok, inst = try_generate(schema)
            key = (sub, cls)
            t = tally.setdefault(key, [0, 0])
            t[1] += 1
            if ok:
                t[0] += 1
                if examples[cls] is None:
                    examples[cls] = (sub, json.dumps(inst)[:150])
    # report
    print(f"\n{'subset':18s} {'class':7s}  yield")
    agg = {"flat": [0, 0], "nested": [0, 0]}
    for (sub, cls), (ok, tot) in sorted(tally.items()):
        print(f"{sub:18s} {cls:7s}  {ok}/{tot} = {ok/max(tot,1):.0%}")
        agg[cls][0] += ok
        agg[cls][1] += tot
    print("\n===== AGGREGATE YIELD =====")
    for cls in ("flat", "nested"):
        ok, tot = agg[cls]
        print(f"{cls:7s}: {ok}/{tot} = {ok/max(tot,1):.1%}")
    print("\nsample flat  :", examples["flat"])
    print("sample nested:", examples["nested"])
    nested_yield = agg["nested"][0] / max(agg["nested"][1], 1)
    print(f"\nVERDICT: nested yield = {nested_yield:.0%} ", end="")
    print("(healthy, proceed to full build)" if nested_yield >= 0.6
          else "(LOW - extend construct.py before building)")


if __name__ == "__main__":
    main()
