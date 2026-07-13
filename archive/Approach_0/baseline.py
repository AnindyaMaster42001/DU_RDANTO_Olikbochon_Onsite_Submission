"""Baseline predictor: submission #1.

Two-branch rule system:
  - context present: predict faithful (1) iff the normalized response is a
    substring of the normalized context, else hallucinated (0)
  - no context: constant guess; both constants are scored on the sample set
    and the better one (by F1 on the hallucinated class) is used for test

Writes submission_baseline.csv (id,label — one row per test-set row).

Usage: python3 baseline.py
"""

import csv
import json
import re
from pathlib import Path

from evaluate import evaluate, has_context

DATA_DIR = Path(__file__).parent
OUT = DATA_DIR / "submission_baseline.csv"

_PUNCT = re.compile(r"[\s।,.;:!?\"'()\[\]{}\-–—`~*_/\\]+")


def norm(s):
    return _PUNCT.sub("", str(s))


def predict(row, noctx_label):
    if has_context(row):
        r = norm(row["response_bn"])
        return 1 if r and r in norm(row["context"]) else 0
    return noctx_label


def main():
    with open(DATA_DIR / "dataset samples.json", encoding="utf-8") as f:
        samples = json.load(f)

    best_label, best_f1 = None, -1.0
    for noctx_label in (0, 1):
        print(f"\n=== no-context rows -> constant {noctx_label} ===")
        f1 = evaluate([predict(s, noctx_label) for s in samples])
        if f1 > best_f1:
            best_label, best_f1 = noctx_label, f1
    print(
        f"\npicked no-context constant = {best_label} "
        f"(sample-set macro-F1: {best_f1:.4f})"
    )

    with open(DATA_DIR / "test set.csv", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for row in test_rows:
            w.writerow([row["id"], predict(row, best_label)])
    print(f"wrote {OUT.name}: {len(test_rows)} rows")


if __name__ == "__main__":
    main()
