"""Approach_1 · Step 4 — calibrated ensemble with honest nested CV.

Fixes Approach_0's core validation flaw: it picked "best of 16 strategies" and
swept a threshold on the full 299-row sample set, which overfits it. Here every
decision (feature weights AND the threshold) is made INSIDE cross-validation
folds, so the reported number is an honest out-of-fold estimate.

  - Inputs: every signal_*.json present (nli, judge, banglabert, ...). Each is a
    per-row P(faithful). Missing signals are simply skipped.
  - Model: low-capacity logistic regression (299 rows can't feed more). Fit
    PER BRANCH (context vs no-context) since base rates differ sharply.
  - Threshold: chosen inside each fold to maximise macro-F1; the OOF preds use
    each fold's own threshold. Final threshold = median across folds.
  - Output: submission_ensemble.csv, refit on all samples.

Add a signal by dropping another signal_*.json next to this file — no code
change needed.

Usage:  python3 ensemble.py          # needs scikit-learn, numpy
"""

import glob
from pathlib import Path

import numpy as np

import common as C

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

FOLDS = 5
SEED = 42
HERE = Path(__file__).resolve().parent


def load_signals():
    sigs = []
    for p in sorted(glob.glob(str(HERE / "signal_*.json"))):
        import json
        s = json.load(open(p, encoding="utf-8"))
        sigs.append(s)
    if not sigs:
        raise SystemExit("no signal_*.json found — run nli_grounding.py / judge.py first")
    print("signals:", [s["name"] for s in sigs])
    return sigs


def matrix(sigs, rows, split):
    """Feature matrix [n_rows, n_signals] for split in {'samples','test'}."""
    n = len(rows)
    X = np.full((n, len(sigs)), 0.5)
    for j, s in enumerate(sigs):
        if split == "samples":
            X[:, j] = s["samples"]
        else:
            col = s["test"]
            X[:, j] = [col.get(str(r["id"]), 0.5) for r in rows]
    return X


def best_threshold(y, prob):
    best_t, best_m = 0.5, -1.0
    for t in np.arange(0.30, 0.71, 0.02):
        m = C.macro_f1(y, (prob >= t).astype(int).tolist())
        if m > best_m:
            best_t, best_m = float(t), m
    return best_t


def fit_branch(X, y):
    """Nested-CV OOF probs + per-fold thresholds for one branch."""
    y = np.asarray(y)
    oof = np.zeros(len(y))
    thrs = []
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        thrs.append(best_threshold(y[tr], clf.predict_proba(X[tr])[:, 1]))
    final = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
    return oof, float(np.median(thrs)), final


def main():
    if not HAVE_SK:
        print("!! install scikit-learn (see requirements.txt)")
        return
    sigs = load_signals()
    samples, test = C.load_samples(), C.load_test()
    y = np.array([s["label"] for s in samples])
    ctx_mask = np.array([C.has_context(s) for s in samples])

    Xs = matrix(sigs, samples, "samples")
    Xt = matrix(sigs, test, "test")
    t_ctx = np.array([C.has_context(r) for r in test])

    oof = np.zeros(len(y))
    test_prob = np.zeros(len(test))
    test_thr = np.zeros(len(test))
    for name, smask, tmask in [("context", ctx_mask, t_ctx),
                               ("no-context", ~ctx_mask, ~t_ctx)]:
        o, thr, clf = fit_branch(Xs[smask], y[smask])
        oof[smask] = o
        m = C.macro_f1(y[smask].tolist(), (o >= thr).astype(int).tolist())
        print(f"[{name:<10}] OOF macroF1={m:.4f}  threshold={thr:.2f}  "
              f"weights={dict(zip([s['name'] for s in sigs], np.round(clf.coef_[0], 2)))}")
        test_prob[tmask] = clf.predict_proba(Xt[tmask])[:, 1]
        test_thr[tmask] = thr

    print("\n=== honest OOF (per-branch nested CV) ===")
    C.evaluate_samples(_apply(oof, ctx_mask, y))

    labels = (test_prob >= test_thr).astype(int)
    C.write_submission("submission_ensemble.csv", [r["id"] for r in test], labels)


def _apply(oof, ctx_mask, y):
    # reconstruct per-branch thresholded OOF preds for the evaluate print
    preds = np.zeros(len(y), int)
    for mask in (ctx_mask, ~ctx_mask):
        thr = best_threshold(y[mask].tolist(), oof[mask])
        preds[mask] = (oof[mask] >= thr).astype(int)
    return preds.tolist()


if __name__ == "__main__":
    main()
