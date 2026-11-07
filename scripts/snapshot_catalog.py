#!/usr/bin/env python3
"""Maintainer-run: snapshot the public ModelRunner catalog into catalog.json.

Reads only the anonymous public model list and writes an ALLOWLIST projection
per model (identity, modalities, the Input/Output schema with every $ref
inlined). Nothing else from the API row is copied: no `info`, `paths`,
`parameters`, pricing internals or provider fields, so whatever the API row
grows later cannot leak into this public repo.

Ids are never dropped: a model that disappears from the catalog keeps its
entry with `retired: true`, so saved workflows get a clear error instead of a
missing node. Run: python3 scripts/snapshot_catalog.py
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

API = "https://modelrunner.run"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "catalog.json"
# Word-boundary match; owner names that are model publishers are fine.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from denylist import denylist  # noqa: E402

FORBIDDEN = denylist()
KEEP_SCHEMA_KEYS = {
    "type", "format", "enum", "default", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems", "items",
    "properties", "required", "anyOf", "oneOf", "allOf", "description", "title",
    "x-order", "multipleOf", "nullable", "const",
}


def fetch(path):
    req = urllib.request.Request(API + path, headers={"User-Agent": "modelrunner-comfyui-snapshot"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def inline(node, components, depth=0):
    """Resolve local $refs and keep only schema keywords from the allowlist."""
    if depth > 20:
        raise ValueError("schema nesting too deep")
    if isinstance(node, list):
        return [inline(n, components, depth + 1) for n in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        merged = {**components[name], **{k: v for k, v in node.items() if k != "$ref"}}
        return inline(merged, components, depth + 1)
    out = {}
    for k, v in node.items():
        if k == "properties":
            out[k] = {pk: inline(pv, components, depth + 1) for pk, pv in v.items()}
        elif k in KEEP_SCHEMA_KEYS:
            out[k] = inline(v, components, depth + 1)
    return out


def project(row):
    schemas = (row.get("schema") or {}).get("components", {}).get("schemas", {})
    inp = inline(schemas.get("Input", {"type": "object", "properties": {}}), schemas)
    outp = inline(schemas.get("Output", {}), schemas)
    return {
        "endpoint": f'{row["ownerName"]}/{row["alias"]}',
        "name": row.get("name") or row["alias"],
        "variantName": row.get("variantName") or "",
        "category": row.get("category") or "",
        "inputModalities": row.get("inputModalities") or [],
        "outputModalities": row.get("outputModalities") or [],
        "advancedInputsOffset": row.get("advancedInputsOffset"),
        "retiresAt": row.get("retiresAt"),
        "successorModelEndpoint": row.get("successorModelEndpoint"),
        # Average charged cost per run, USD, as the catalog reports it at snapshot time.
        "averageCost": row.get("averageCost"),
        "input": {"properties": inp.get("properties", {}), "required": inp.get("required", [])},
        "output": outp,
    }


def main():
    rows, page = [], 1
    while True:
        data = fetch(f"/models?limit=100&page={page}")
        rows += data["data"]
        if page >= data["totalPages"]:
            break
        page += 1
    models = {m["endpoint"]: m for m in map(project, rows)}

    previous = {}
    if OUT.exists():
        previous = {m["endpoint"]: m for m in json.loads(OUT.read_text())["models"]}
    for endpoint, old in previous.items():
        if endpoint not in models:
            models[endpoint] = {**old, "retired": True}

    text = json.dumps({"models": sorted(models.values(), key=lambda m: m["endpoint"])},
                      indent=1, ensure_ascii=False) + "\n"
    hits = sorted({m.group(0) for m in FORBIDDEN.finditer(text)})
    if hits:
        for line in text.splitlines():
            if FORBIDDEN.search(line):
                print("  ", line.strip()[:160], file=sys.stderr)
        sys.exit(f"refusing to write catalog.json: forbidden words {hits}")
    OUT.write_text(text)
    retired = sum(1 for m in models.values() if m.get("retired"))
    print(f"catalog.json: {len(models)} models ({retired} retired), {len(text) // 1024} KB")


if __name__ == "__main__":
    main()
