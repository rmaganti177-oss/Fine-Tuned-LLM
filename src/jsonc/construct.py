"""Deterministic constructive JSON-instance generator — the OWNED backstop for
nested/$ref/recursive schemas that hypothesis-jsonschema (Draft-07, no recursive
$ref) silently drops. This is the piece that guarantees the nested region — the
entire point of the project — is actually covered in training.

Resolves $ref by JSON-pointer within $defs/definitions, handles
allOf(merge)/anyOf|oneOf(first branch)/enum|const/prefixItems+items(tuple)/
additionalProperties, and terminates recursion at a bounded depth by choosing a
non-recursive branch or omitting the optional recursive property.

Every produced instance is still gated through validate.py (this generator is
best-effort; the validator is the source of truth).
"""
import random
import re

MAX_DEPTH = 5
MAX_PROPS = 24      # cap properties emitted per object (huge K8s schemas -> blowup guard)
MAX_NODES = 2000    # hard cap on total nodes in one instance


class _Ctx:
    def __init__(self, root, rng):
        self.root = root
        self.rng = rng
        self.nodes = 0      # total values emitted; hard cap guards against blowup


class _Budget(Exception):
    pass


def _resolve_ref(ref, ctx):
    """Resolve a local JSON-pointer ref like '#/$defs/Node'."""
    if not ref.startswith("#"):
        return None
    parts = [p for p in ref.lstrip("#").split("/") if p != ""]
    node = ctx.root
    for p in parts:
        p = p.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return None
    return node


def _pick(seq, rng):
    return seq[rng.randrange(len(seq))] if seq else None


def _gen_string(schema, rng):
    if "enum" in schema:
        return _pick([e for e in schema["enum"] if isinstance(e, str)] or schema["enum"], rng)
    if "const" in schema:
        return schema["const"]
    fmt = schema.get("format")
    samples = {
        "email": "user@example.com", "date-time": "2026-01-15T10:30:00Z",
        "date": "2026-01-15", "time": "10:30:00", "uuid": "123e4567-e89b-12d3-a456-426614174000",
        "uri": "https://example.com/x", "hostname": "example.com", "ipv4": "192.168.1.1",
        "ipv6": "2001:db8::1", "duration": "P1Y2M3DT4H5M6S",
    }
    if fmt in samples:
        return samples[fmt]
    if "pattern" in schema:
        # best-effort: return a trivial string; validator gates it, faker post-fill improves
        base = "abc123"
        return base
    minl = schema.get("minLength", 3)
    s = "sample_value"
    if len(s) < minl:
        s = s + "x" * (minl - len(s))
    maxl = schema.get("maxLength")
    if maxl is not None and len(s) > maxl:
        s = s[:maxl]
    return s


def _gen_number(schema, rng, integer):
    if "enum" in schema:
        nums = [e for e in schema["enum"] if isinstance(e, (int, float))]
        if nums:
            return _pick(nums, rng)
    if "const" in schema:
        return schema["const"]
    lo = schema.get("minimum", schema.get("exclusiveMinimum", 0))
    hi = schema.get("maximum", schema.get("exclusiveMaximum", lo + 100))
    if schema.get("exclusiveMinimum") is not None:
        lo = schema["exclusiveMinimum"] + 1
    val = lo + (hi - lo) * 0.5
    mult = schema.get("multipleOf")
    if mult:
        val = round(val / mult) * mult
    return int(val) if integer else float(val)


_FULL = {"on": False}  # when True, include all optional properties (richer instances)


def generate(schema, ctx=None, depth=0, rng=None, full=None):
    """Return a Python value that should validate against `schema`.
    full=True includes all optional properties for non-degenerate training targets."""
    if rng is None:
        rng = random.Random(42)
    if ctx is None:
        ctx = _Ctx(schema, rng)
    if full is not None:
        _FULL["on"] = full

    ctx.nodes += 1
    if ctx.nodes > MAX_NODES:
        raise _Budget()

    if not isinstance(schema, dict):
        return None
    if schema is True or schema == {}:
        return "x"  # any value
    if schema is False:
        return None

    # $ref
    if "$ref" in schema:
        target = _resolve_ref(schema["$ref"], ctx)
        if target is None:
            return None
        if depth >= MAX_DEPTH:
            return _terminal_for(target, ctx, rng)
        return generate(target, ctx, depth + 1, rng)

    # const / enum shortcuts
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return _pick(schema["enum"], rng)

    # combinators
    if "allOf" in schema:
        merged = {}
        for sub in schema["allOf"]:
            if isinstance(sub, dict):
                merged = _merge(merged, _deref(sub, ctx))
        merged = _merge(merged, {k: v for k, v in schema.items() if k != "allOf"})
        return generate(merged, ctx, depth, rng)
    for comb in ("oneOf", "anyOf"):
        if comb in schema and schema[comb]:
            branches = schema[comb]
            # prefer a non-recursive branch when deep
            if depth >= MAX_DEPTH:
                for b in branches:
                    if isinstance(b, dict) and "$ref" not in b:
                        return generate(b, ctx, depth + 1, rng)
            return generate(_deref(branches[0], ctx), ctx, depth + 1, rng)

    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0])

    if t == "object" or ("properties" in schema and t is None):
        return _gen_object(schema, ctx, depth, rng)
    if t == "array" or ("items" in schema and t is None):
        return _gen_array(schema, ctx, depth, rng)
    if t == "string":
        return _gen_string(schema, rng)
    if t == "integer":
        return _gen_number(schema, rng, True)
    if t == "number":
        return _gen_number(schema, rng, False)
    if t == "boolean":
        return True
    if t == "null":
        return None
    # untyped: default to a string
    return "x"


def _terminal_for(schema, ctx, rng):
    """At max depth, produce the smallest valid value (empty object/array or scalar)."""
    t = schema.get("type")
    if isinstance(t, list):
        t = t[0]
    if t == "object" or "properties" in schema:
        # only required, shallow
        return _gen_object(schema, ctx, MAX_DEPTH, rng, minimal=True)
    if t == "array":
        return []
    if t == "string":
        return _gen_string(schema, rng)
    if t in ("integer", "number"):
        return _gen_number(schema, rng, t == "integer")
    if t == "boolean":
        return True
    return None


def _deref(schema, ctx):
    if isinstance(schema, dict) and "$ref" in schema:
        r = _resolve_ref(schema["$ref"], ctx)
        return r if r is not None else schema
    return schema


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if k == "properties" and "properties" in out:
            out["properties"] = {**out["properties"], **v}
        elif k == "required" and "required" in out:
            out["required"] = list(set(out["required"]) | set(v))
        else:
            out[k] = v
    return out


def _gen_object(schema, ctx, depth, rng, minimal=False):
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    obj = {}
    # cap breadth: always keep required, then optionals up to MAX_PROPS total
    names = list(required) + [n for n in props if n not in required]
    names = names[:MAX_PROPS]
    props = {n: props[n] for n in names if n in props}
    for name, sub in props.items():
        include = (name in required) or (not minimal and depth < MAX_DEPTH)
        if not minimal and name not in required:
            # include optionals: all of them in full mode, else ~half, deterministically
            include = (_FULL["on"] or hash(name) % 2 == 0) and depth < MAX_DEPTH
        if include or name in required:
            obj[name] = generate(_deref(sub, ctx), ctx, depth + 1, rng)
    # additionalProperties required by minProperties?
    minp = schema.get("minProperties")
    if minp and len(obj) < minp:
        ap = schema.get("additionalProperties")
        i = 0
        while len(obj) < minp:
            key = f"extra_{i}"
            if key not in obj:
                if isinstance(ap, dict):
                    obj[key] = generate(ap, ctx, depth + 1, rng)
                elif ap is not False:
                    obj[key] = "x"
                else:
                    break
            i += 1
    return obj


def _gen_array(schema, ctx, depth, rng):
    # tuple form
    prefix = schema.get("prefixItems")
    if prefix:
        return [generate(_deref(s, ctx), ctx, depth + 1, rng) for s in prefix]
    items = schema.get("items")
    mini = schema.get("minItems", 1)
    n = max(mini, 1)
    maxi = schema.get("maxItems")
    if maxi is not None:
        n = min(n, maxi)
    if isinstance(items, dict):
        if depth >= MAX_DEPTH:
            return []
        return [generate(_deref(items, ctx), ctx, depth + 1, rng) for _ in range(n)]
    return [] if mini == 0 else ["x"] * n


if __name__ == "__main__":
    import json
    from validate import validate_2020_12

    recursive = {
        "$defs": {"node": {"type": "object", "properties": {
            "val": {"type": "integer"},
            "next": {"$ref": "#/$defs/node"}}, "required": ["val"]}},
        "$ref": "#/$defs/node",
    }
    oneof = {"type": "object", "properties": {
        "payment": {"oneOf": [
            {"type": "object", "properties": {"card": {"type": "string"}}, "required": ["card"]},
            {"type": "object", "properties": {"cash": {"type": "boolean"}}, "required": ["cash"]}]}},
        "required": ["payment"]}
    for name, sch in [("recursive", recursive), ("oneOf", oneof)]:
        inst = generate(sch, rng=random.Random(1))
        ok, err = validate_2020_12(inst, sch)
        print(f"{name}: valid={ok} err={err}\n  {json.dumps(inst)[:200]}")
