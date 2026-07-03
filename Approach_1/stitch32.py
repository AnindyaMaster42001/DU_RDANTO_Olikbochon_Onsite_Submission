"""Final stitch: 7-signal ensemble vs 7+judge32, seed-averaged, honest OOF.

Writes submission_final.csv from whichever wins (refit on all samples,
median threshold across seeds x folds). Run after downloading
signal_judge32.json from the bengali-judge32 kernel.

Usage:  python3 stitch32.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common as C

HERE = Path(__file__).resolve().parent
FOLDS = 5
SEEDS = range(5)

SIGS = {}
for p in sorted(HERE.glob("signal_*.json")):
    s = json.load(open(p, encoding="utf-8"))
    SIGS[s["name"]] = s

SAMPLES, TEST = C.load_samples(), C.load_test()
Y = np.array([s["label"] for s in SAMPLES])
CTX_S = np.array([C.has_context(s) for s in SAMPLES])
CTX_T = np.array([C.has_context(r) for r in TEST])

BASE = ["a0judge", "a0selfv", "crosslingual", "judge", "nli", "retrieval", "substring"]


def build(names, split):
    if split == "samples":
        return np.column_stack([np.asarray(SIGS[n]["samples"], float) for n in names])
    return np.column_stack(
        [[float(SIGS[n]["test"].get(str(r["id"]), 0.5)) for r in TEST] for n in names]
    ).reshape(len(TEST), len(names))


def best_threshold(y, prob):
    ts = np.arange(0.30, 0.71, 0.02)
    ms = [C.macro_f1(y, (prob >= t).astype(int).tolist()) for t in ts]
    return float(ts[int(np.argmax(ms))])


def run(names):
    per_seed, ctx_ms, noctx_ms = [], [], []
    thrs_all = {"ctx": [], "noctx": []}
    for seed in SEEDS:
        preds = np.zeros(len(Y), int)
        for key, mask in [("ctx", CTX_S), ("noctx", ~CTX_S)]:
            X, y = build(names, "samples")[mask], Y[mask]
            oof, thrs = np.zeros(len(y)), []
            skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
            for tr, va in skf.split(X, y):
                clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
                oof[va] = clf.predict_proba(X[va])[:, 1]
                thrs.append(best_threshold(y[tr], clf.predict_proba(X[tr])[:, 1]))
            thr = float(np.median(thrs))
            thrs_all[key] += thrs
            m = C.macro_f1(y.tolist(), (oof >= thr).astype(int).tolist())
            (ctx_ms if key == "ctx" else noctx_ms).append(m)
            preds[mask] = (oof >= thr).astype(int)
        per_seed.append(C.macro_f1(Y.tolist(), preds.tolist()))
    print(
        f"{'+'.join(sorted(set(names) - set(BASE))) or 'base7':<12} "
        f"OOF {np.mean(per_seed):.4f} +-{np.std(per_seed):.4f}   "
        f"ctx {np.mean(ctx_ms):.4f}  noctx {np.mean(noctx_ms):.4f}"
    )
    return float(np.mean(per_seed)), thrs_all


def main():
    assert "judge32" in SIGS, "signal_judge32.json missing — download kernel output first"
    # standalone read on the new signal
    for key, mask in [("ctx", CTX_S), ("noctx", ~CTX_S)]:
        v = np.asarray(SIGS["judge32"]["samples"], float)[mask]
        m = C.macro_f1(Y[mask].tolist(), (v >= 0.5).astype(int).tolist())
        print(f"judge32 standalone [{key}]: macroF1 {m:.4f}")

    m7, _ = run(BASE)
    m8, thrs8 = run(BASE + ["judge32"])

    names = BASE + ["judge32"] if m8 >= m7 else BASE
    thrs = thrs8 if m8 >= m7 else run(BASE)[1]
    print(f"\nselected: {len(names)} signals")

    sub = np.zeros(len(TEST), int)
    for key, smask, tmask in [("ctx", CTX_S, CTX_T), ("noctx", ~CTX_S, ~CTX_T)]:
        clf = LogisticRegression(max_iter=1000).fit(
            build(names, "samples")[smask], Y[smask]
        )
        thr = float(np.median(thrs[key]))
        sub[tmask] = (
            clf.predict_proba(build(names, "test")[tmask])[:, 1] >= thr
        ).astype(int)
        print(f"  {key}: thr={thr:.2f} weights="
              f"{dict(zip(names, np.round(clf.coef_[0], 2)))}")
    C.write_submission("submission_final.csv", [r["id"] for r in TEST], sub.tolist())


if __name__ == "__main__":
    main()
