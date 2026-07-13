"""Evaluation harness for the Bengali Hallucination Detection challenge.

Scores predictions against the 299-row labeled sample set. Reports both
macro-F1 (what the leaderboard appears to use: our baseline scored 0.666
on the LB vs 0.6823 local macro-F1 vs 0.7386 class-0 F1) and binary F1
on the hallucinated class (what the Rules tab describes for Phase 2).
Also reports the context-present and no-context branches separately,
since they behave like two different sub-problems.

Usage:
    python3 evaluate.py preds.csv          # preds.csv: id,label with ids 0..298
    python3 evaluate.py --self-test        # sanity-check the metric code

Or import and call `evaluate(preds)` with a list of 299 ints.
"""

import csv
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
SAMPLES = DATA_DIR / "dataset samples.json"


def load_samples():
    with open(SAMPLES, encoding="utf-8") as f:
        return json.load(f)


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


def f1_on_class(y_true, y_pred, cls=0):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def macro_f1(y_true, y_pred):
    return (f1_on_class(y_true, y_pred, 0)[0] + f1_on_class(y_true, y_pred, 1)[0]) / 2


def evaluate(preds, verbose=True):
    """preds: list of 299 ints (0/1), aligned with dataset samples.json order."""
    samples = load_samples()
    assert len(preds) == len(samples), (
        f"need {len(samples)} predictions, got {len(preds)}"
    )
    y_true = [s["label"] for s in samples]

    def report(name, idxs):
        yt = [y_true[i] for i in idxs]
        yp = [preds[i] for i in idxs]
        f1, p, r = f1_on_class(yt, yp, 0)
        acc = sum(1 for a, b in zip(yt, yp) if a == b) / len(yt)
        if verbose:
            print(
                f"{name:<18} n={len(yt):<4} "
                f"macroF1={macro_f1(yt, yp):.4f}  "
                f"F1(hallucinated)={f1:.4f}  P={p:.3f} R={r:.3f}  "
                f"acc={acc:.3f}"
            )
        return macro_f1(yt, yp)

    all_idx = range(len(samples))
    ctx_idx = [i for i, s in enumerate(samples) if has_context(s)]
    noctx_idx = [i for i, s in enumerate(samples) if not has_context(s)]

    overall = report("OVERALL", all_idx)
    report("  with context", ctx_idx)
    report("  no context", noctx_idx)
    return overall


def load_preds_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["id"]))
    return [int(r["label"]) for r in rows]


def self_test():
    samples = load_samples()
    y = [s["label"] for s in samples]
    perfect = evaluate(y, verbose=False)
    assert perfect == 1.0, perfect
    all_zero, _, _ = f1_on_class(y, [0] * len(y), 0)
    print(f"self-test OK: perfect preds -> F1 1.0; all-zero baseline F1 {all_zero:.4f}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    elif len(sys.argv) > 1:
        evaluate(load_preds_csv(sys.argv[1]))
    else:
        print(__doc__)
