"""Words that must never appear in anything this repo publishes.

The list itself is private, so it is not in the repo: set MODELRUNNER_DENYLIST
(comma-separated) or put one word per line in scripts/denylist.txt, which git
ignores. CI reads it from the MODELRUNNER_DENYLIST repository secret.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

LIST_FILE = Path(__file__).resolve().parent / "denylist.txt"


def denylist() -> re.Pattern:
    raw = os.environ.get("MODELRUNNER_DENYLIST", "")
    words = [w.strip() for w in re.split(r"[,\n]", raw) if w.strip()]
    if not words and LIST_FILE.is_file():
        words = [w.strip() for w in LIST_FILE.read_text().splitlines() if w.strip() and not w.startswith("#")]
    if not words:
        sys.exit("No deny-list: set MODELRUNNER_DENYLIST or create scripts/denylist.txt.")
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)
