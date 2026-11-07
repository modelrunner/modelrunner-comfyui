"""Ship gate for a catalog refresh. Run before tagging a release:

    python3 scripts/check_pack.py

Maps every catalog row without ComfyUI installed and fails on a duplicate node
id, a denied word (see denylist.py) in any file git would publish, or a required
field that only extra_json can set. Prints the extra_json fallbacks so the count is reviewed.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from denylist import denylist  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def tracked_files() -> list[str]:
    """Every file git would publish: tracked plus untracked-but-not-ignored."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if (ROOT / f).is_file()]


def main() -> int:
    mapper = load("mapper", ROOT / "modelrunner_nodes" / "mapper.py")
    models = json.loads((ROOT / "catalog.json").read_text())["models"]
    typed = [m for m in models if mapper.output_kind(m)]
    failures = []

    ids = {}
    for m in typed:
        nid = mapper.node_id(m["endpoint"])
        if nid in ids:
            failures.append(f"duplicate node id {nid}: {ids[nid]} and {m['endpoint']}")
        ids[nid] = m["endpoint"]

    widgets = 0
    for m in typed:
        types = mapper.input_types(mapper.plan_fields(m))
        widgets += len(types["required"]) + len(types["optional"])

    report = mapper.fallback_report(typed)
    for endpoint, field, where in report["fields"]:
        if where == "required":
            failures.append(f"{endpoint}.{field} is required but only extra_json can set it")

    words = denylist()
    for rel in tracked_files():
        text = (ROOT / rel).read_text(errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if words.search(line):
                failures.append(f"denied word in {rel}:{n}")

    print(f"catalog rows: {len(models)}  typed nodes: {len(typed)}  "
          f"left to the generic node: {len(models) - len(typed)}  widgets/sockets: {widgets}")
    print(f"extra_json-only fields: {len(report['fields'])} across {len(report['models'])} models")
    for endpoint, field, where in report["fields"]:
        print(f"  {endpoint}  {field}  ({where})")
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
