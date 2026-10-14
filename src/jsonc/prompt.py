"""The ONE shared prompt/render module — used identically by data-gen, training,
the in-run callback, and eval. Any drift between train and serve is the #3 failure
class from the prior project, so this is the single source of truth and it is hashed.

We define our OWN canonical prompt (we control train==eval). At final eval we ALSO
run the benchmark's own harness prompt separately and report both, so any gap is
attributable to prompt wording, not the model.
"""
import hashlib
import json

SYSTEM = (
    "You are a precise JSON generator. Given a JSON Schema, output ONE JSON value "
    "that strictly validates against it. Output ONLY the JSON value — no prose, no "
    "markdown fences, no comments. Include every required property, use only the "
    "properties and types the schema allows, and respect enums, formats, and nesting."
)

USER_TEMPLATE = (
    "Generate a JSON value that validates against this JSON Schema:\n\n{schema}\n\n"
    "Return only the JSON value."
)


def render_messages(schema):
    """schema: dict or JSON string -> chat messages list.
    Schema is serialized deterministically (sorted keys) so the prompt is stable."""
    if isinstance(schema, str):
        schema = json.loads(schema)
    schema_str = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema_str)},
    ]


def _hash():
    h = hashlib.sha256()
    h.update(SYSTEM.encode())
    h.update(USER_TEMPLATE.encode())
    return h.hexdigest()[:16]


PROMPT_HASH = _hash()

if __name__ == "__main__":
    ex = {"type": "object", "properties": {"name": {"type": "string"}},
          "required": ["name"]}
    for m in render_messages(ex):
        print(f"[{m['role']}]\n{m['content']}\n")
    print("PROMPT_HASH =", PROMPT_HASH)
