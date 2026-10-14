"""The single source of truth for "does this JSON comply with this schema".
Used by data-gen (gate), the in-run callback, the merge gate, and final eval —
so our internal metric CANNOT diverge from the benchmark for tooling reasons
(failure class #1). Everything is validated as Draft 2020-12 with formats ON,
matching JSONSchemaBench.

Also provides the output post-processing (fence strip + first-balanced-JSON
extraction) and the two diagnostic detectors (hallucinated keys, schema-echo).
"""
import json
import re

from jsonschema import Draft202012Validator
from jsonschema.validators import Draft202012Validator as _D
from jsonschema import FormatChecker

FORMAT_CHECKER = Draft202012Validator.FORMAT_CHECKER


def validate_2020_12(instance, schema):
    """Return (is_valid, first_error_message_or_None). Validates as 2020-12 with
    the full format checker regardless of the schema's declared $schema."""
    try:
        v = _D(schema, format_checker=FORMAT_CHECKER)
        errors = sorted(v.iter_errors(instance), key=lambda e: e.path)
        if not errors:
            return True, None
        e = errors[0]
        loc = "/".join(str(p) for p in e.path) or "<root>"
        return False, f"{e.validator} at {loc}: {e.message[:160]}"
    except Exception as ex:  # invalid schema itself, ref resolution failure, etc.
        return False, f"validator_error: {type(ex).__name__}: {str(ex)[:160]}"


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text):
    """Strip markdown fences, take the first balanced JSON value. Return (obj, ok)."""
    t = _FENCE.sub("", text).strip()
    # fast path
    try:
        return json.loads(t), True
    except Exception:
        pass
    # find first balanced { } or [ ]
    start = None
    for i, ch in enumerate(t):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None, False
    open_ch = t[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for j in range(start, len(t)):
        c = t[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:j + 1]), True
                    except Exception:
                        return None, False
    return None, False


def _schema_prop_names(schema):
    names = set()

    def walk(s):
        if not isinstance(s, dict):
            return
        for k in ("properties", "patternProperties"):
            if isinstance(s.get(k), dict):
                names.update(s[k].keys())
        for k in ("$defs", "definitions"):
            if isinstance(s.get(k), dict):
                for sub in s[k].values():
                    walk(sub)
        for k in ("items", "additionalProperties", "contains"):
            if isinstance(s.get(k), dict):
                walk(s[k])
        for comb in ("allOf", "anyOf", "oneOf"):
            for sub in s.get(comb, []) or []:
                walk(sub)

    walk(schema)
    return names


def count_hallucinated_keys(obj, schema):
    """Count object keys that appear nowhere in the schema's declared property names.
    Heuristic diagnostic (not the compliance metric), for tracking key-hallucination."""
    allowed = _schema_prop_names(schema)
    if not allowed:
        return 0
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            for k, v in o.items():
                if k not in allowed:
                    n += 1
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return n


def looks_like_schema_echo(obj, schema):
    """True if the model echoed the schema (returned something containing
    JSON-Schema meta-keywords) instead of an instance."""
    META = {"type", "properties", "required", "$defs", "definitions",
            "$schema", "additionalProperties", "items"}
    if not isinstance(obj, dict):
        return False
    hits = sum(1 for k in obj.keys() if k in META)
    # echo if it looks structurally like a schema AND schema itself has properties
    return hits >= 3 and "properties" in obj and isinstance(schema, dict)


# ---- startup self-test: a missing format backend must NOT silently pass ----
def format_self_test():
    """Assert the format checker actually REJECTS bad values. If a backend is
    missing, jsonschema silently treats the format as valid — this catches that."""
    cases = [
        ({"type": "string", "format": "email"}, "not-an-email"),
        ({"type": "string", "format": "date-time"}, "2026-13-45T99:99:99"),
        ({"type": "string", "format": "uuid"}, "not-a-uuid"),
        ({"type": "string", "format": "ipv4"}, "999.999.999.999"),
    ]
    missing = []
    for sch, bad in cases:
        ok, _ = validate_2020_12(bad, sch)
        if ok:  # should have been rejected
            missing.append(sch["format"])
    if missing:
        raise RuntimeError(
            f"FORMAT BACKENDS MISSING (bad values passed): {missing}. "
            f"pip install fqdn idna rfc3339-validator rfc3987 isoduration jsonpointer uri-template rfc3986-validator")
    return True


if __name__ == "__main__":
    format_self_test()
    print("format self-test PASSED (bad email/datetime/uuid/ipv4 all rejected)")
    # extraction tests
    for t in ['{"a":1}', '```json\n{"a":1}\n```', 'sure: {"a":1} done', '{"a":1}}']:
        print(repr(t), "->", extract_json(t))
    # compliance test
    sch = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    print("valid:", validate_2020_12({"n": 5}, sch))
    print("missing req:", validate_2020_12({}, sch))
    print("halluc keys:", count_hallucinated_keys({"n": 1, "x": 2, "y": 3}, sch))
