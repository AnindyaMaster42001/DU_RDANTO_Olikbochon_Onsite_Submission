"""Shared helpers for Approach_1 — data loading, metric, signal I/O.

Self-contained and Kaggle-portable: it auto-locates the competition data
whether you run it from Approach_1/, from the repo root, or inside a Kaggle
notebook (/kaggle/input/...). The metric mirrors Approach_0/evaluate.py exactly.

Signal files (one per model/stage) are the interface to ensemble.py:
    {"name": <str>,
     "samples": [v0, ..., v298],     # aligned to dataset-samples order
     "test":    {"<id>": v, ...}}    # keyed by test-row id (string)
A signal value is a float in [0, 1] = P(faithful) when known, else 0.5.
"""

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ data dir
_KAGGLE = sorted(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
_CANDIDATE_DIRS = [
    Path("/kaggle/input/bengali-hallucination"),
    *_KAGGLE,
    HERE,
    HERE.parent / "Approach_0",
    HERE.parent,
    Path.cwd(),
]


def _find(dir_, *, contains, ext):
    if not dir_.is_dir():
        return None
    hits = [
        p for p in dir_.iterdir()
        if p.suffix.lower() == ext and all(c in p.name.lower() for c in contains)
    ]
    return hits[0] if hits else None


def _resolve_data_dir():
    for d in _CANDIDATE_DIRS:
        if _find(d, contains=["samples"], ext=".json") and _find(d, contains=["test"], ext=".csv"):
            return d
    raise FileNotFoundError(
        "Could not locate competition data (a *samples*.json and *test*.csv). "
        f"Looked in: {[str(d) for d in _CANDIDATE_DIRS]}"
    )


DATA_DIR = _resolve_data_dir()
OUT_DIR = HERE  # write submissions / signals next to the scripts


def load_samples():
    with open(_find(DATA_DIR, contains=["samples"], ext=".json"), encoding="utf-8") as f:
        return json.load(f)


def load_test():
    with open(_find(DATA_DIR, contains=["test"], ext=".csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ row utils
def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


_PUNCT = re.compile(r"[\s।,.;:!?\"'()\[\]{}\-–—`~*_/\\]+")


def norm(s):
    return _PUNCT.sub("", str(s))


# ------------------------------------------------------------------ metric
def f1_on_class(y_true, y_pred, cls=0):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec


def macro_f1(y_true, y_pred):
    return (f1_on_class(y_true, y_pred, 0)[0] + f1_on_class(y_true, y_pred, 1)[0]) / 2


def evaluate_samples(preds, verbose=True):
    """preds: 299 ints aligned to load_samples() order. Returns overall macro-F1."""
    samples = load_samples()
    assert len(preds) == len(samples), f"need {len(samples)} preds, got {len(preds)}"
    y = [s["label"] for s in samples]

    def report(name, idxs):
        yt, yp = [y[i] for i in idxs], [preds[i] for i in idxs]
        f1, p, r = f1_on_class(yt, yp, 0)
        acc = sum(a == b for a, b in zip(yt, yp)) / len(yt)
        if verbose:
            print(f"{name:<16} n={len(yt):<4} macroF1={macro_f1(yt, yp):.4f}  "
                  f"F1(hall)={f1:.4f}  P={p:.3f} R={r:.3f}  acc={acc:.3f}")
        return macro_f1(yt, yp)

    ctx = [i for i, s in enumerate(samples) if has_context(s)]
    noc = [i for i, s in enumerate(samples) if not has_context(s)]
    overall = report("OVERALL", range(len(samples)))
    report("  with context", ctx)
    report("  no context", noc)
    return overall


# ------------------------------------------------------------------ outputs
def write_submission(name, ids, labels):
    path = OUT_DIR / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, lab in zip(ids, labels):
            w.writerow([i, int(lab)])
    n0 = sum(1 for lab in labels if int(lab) == 0)
    print(f"wrote {name}: {len(labels)} rows ({n0} hallucinated / {len(labels) - n0} faithful)")
    return path


def save_signal(name, sample_vals, test_vals):
    """sample_vals: list[299] floats; test_vals: dict[id->float]. P(faithful)."""
    path = OUT_DIR / f"signal_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "samples": list(sample_vals), "test": dict(test_vals)}, f)
    print(f"wrote {path.name}")
    return path


def load_signal(name):
    with open(OUT_DIR / f"signal_{name}.json", encoding="utf-8") as f:
        return json.load(f)


def pick_noctx_constant(samples):
    """Best constant label for no-context rows, scored on the samples (macro-F1)."""
    noc = [i for i, s in enumerate(samples) if not has_context(s)]
    y = [samples[i]["label"] for i in noc]
    best, best_m = 0, -1.0
    for c in (0, 1):
        m = macro_f1(y, [c] * len(y))
        if m > best_m:
            best, best_m = c, m
    return best
