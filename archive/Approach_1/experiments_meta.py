"""Meta-model experiments over the existing signals — no GPU needed.

Same honest protocol as ensemble.py (per-branch logistic, threshold inside
folds), but every variant is scored as the MEAN OOF macro-F1 over several CV
seeds, because on 299 rows a single split's +-0.01 is fold noise. Variants:

  features   raw signals | + abstain indicators (signal==0.5 exactly)
  pruning    all signals per branch | branch-specific subsets
  extra      +banglabert (local fine-tune probs from Approach_0)
  C          logistic regularization

Prints a ranked table; writes submission_meta.csv from the best variant
(refit on all samples, median threshold across seeds x folds).

Usage:  python3 experiments_meta.py
"""

import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common as C

HERE = Path(__file__).resolve().parent
A0 = HERE.parent / "Approach_0"
FOLDS = 5
SEEDS = range(5)

# ---------------------------------------------------------------- signals
def load_all_signals():
    sigs = {}
    for p in sorted(HERE.glob("signal_*.json")):
        s = json.load(open(p, encoding="utf-8"))
        sigs[s["name"]] = s
    # banglabert: stitch from Approach_0 local-training outputs (true OOF probs
    # on samples, full-fit probs on test)
    bb_oof = A0 / "results" / "oof_banglabert.csv"
    bb_test = A0 / "results" / "banglabert_test_probs.json"
    if bb_oof.exists() and bb_test.exists():
        rows = list(csv.DictReader(open(bb_oof)))
        rows.sort(key=lambda r: int(r["id"]))
        sigs["banglabert"] = {
            "name": "banglabert",
            "samples": [float(r["prob_faithful"]) for r in rows],
            "test": json.load(open(bb_test)),
        }
    print("signals:", sorted(sigs))
    return sigs


SIGS = load_all_signals()
SAMPLES, TEST = C.load_samples(), C.load_test()
Y = np.array([s["label"] for s in SAMPLES])
CTX_S = np.array([C.has_context(s) for s in SAMPLES])
CTX_T = np.array([C.has_context(r) for r in TEST])


def col(name, split):
    s = SIGS[name]
    if split == "samples":
        return np.asarray(s["samples"], float)
    return np.array([float(s["test"].get(str(r["id"]), 0.5)) for r in TEST])


def build(names, split, abstain):
    cols = [col(n, split) for n in names]
    X = np.column_stack(cols)
    if abstain:
        X = np.column_stack([X] + [(c == 0.5).astype(float) for c in cols])
    return X


# ---------------------------------------------------------------- harness
def best_threshold(y, prob):
    ts = np.arange(0.30, 0.71, 0.02)
    ms = [C.macro_f1(y, (prob >= t).astype(int).tolist()) for t in ts]
    return float(ts[int(np.argmax(ms))])


def run_branch(X, y, c_reg, seed):
    oof = np.zeros(len(y))
    thrs = []
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=c_reg).fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        thrs.append(best_threshold(y[tr], clf.predict_proba(X[tr])[:, 1]))
    thr = float(np.median(thrs))
    return C.macro_f1(y.tolist(), (oof >= thr).astype(int).tolist()), thrs


def run_variant(ctx_names, noctx_names, abstain, c_reg):
    per_seed, ctx_scores, noctx_scores, all_thrs = [], [], [], {"ctx": [], "noctx": []}
    for seed in SEEDS:
        total_preds = np.zeros(len(Y), int)
        for key, names, mask in [("ctx", ctx_names, CTX_S),
                                 ("noctx", noctx_names, ~CTX_S)]:
            X = build(names, "samples", abstain)[mask]
            m, thrs = run_branch(X, Y[mask], c_reg, seed)
            (ctx_scores if key == "ctx" else noctx_scores).append(m)
            all_thrs[key] += thrs
            # rebuild OOF preds for the pooled overall number
            oof = np.zeros(mask.sum())
            skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
            for tr, va in skf.split(X, Y[mask]):
                clf = LogisticRegression(max_iter=1000, C=c_reg).fit(X[tr], Y[mask][tr])
                oof[va] = clf.predict_proba(X[va])[:, 1]
            total_preds[mask] = (oof >= float(np.median(thrs))).astype(int)
        per_seed.append(C.macro_f1(Y.tolist(), total_preds.tolist()))
    return (float(np.mean(per_seed)), float(np.std(per_seed)),
            float(np.mean(ctx_scores)), float(np.mean(noctx_scores)), all_thrs)


ALL = [n for n in sorted(SIGS) if n != "banglabert"]
CTX_LEAN = ["substring", "judge", "a0judge", "nli"]
NOCTX_LEAN = ["judge", "crosslingual", "retrieval", "a0judge", "a0selfv"]

VARIANTS = {
    "v0 all/all raw C=1": (ALL, ALL, False, 1.0),
    "v1 all/all +abstain": (ALL, ALL, True, 1.0),
    "v2 lean/lean raw": (CTX_LEAN, NOCTX_LEAN, False, 1.0),
    "v3 lean/lean +abstain": (CTX_LEAN, NOCTX_LEAN, True, 1.0),
    "v4 lean/lean +abstain C=0.3": (CTX_LEAN, NOCTX_LEAN, True, 0.3),
    "v5 lean/lean +abstain C=3": (CTX_LEAN, NOCTX_LEAN, True, 3.0),
    "v6 +banglabert lean+bb": (CTX_LEAN + ["banglabert"],
                               NOCTX_LEAN + ["banglabert"], True, 1.0),
}


def main():
    results = {}
    for name, cfg in VARIANTS.items():
        mean, std, ctx_m, noctx_m, thrs = run_variant(*cfg)
        results[name] = (mean, std, ctx_m, noctx_m, cfg, thrs)
        print(f"{name:<30} OOF {mean:.4f} +-{std:.4f}   "
              f"ctx {ctx_m:.4f}  noctx {noctx_m:.4f}")

    best = max(results, key=lambda k: results[k][0])
    mean, std, ctx_m, noctx_m, (cn, nn, ab, c_reg), thrs = results[best]
    print(f"\nBEST: {best}  (mean OOF {mean:.4f})")

    # refit on all samples per branch; threshold = median across seeds x folds
    sub = np.zeros(len(TEST), int)
    for names, smask, tmask, key in [(cn, CTX_S, CTX_T, "ctx"),
                                     (nn, ~CTX_S, ~CTX_T, "noctx")]:
        Xs = build(names, "samples", ab)[smask]
        Xt = build(names, "test", ab)[tmask]
        clf = LogisticRegression(max_iter=1000, C=c_reg).fit(Xs, Y[smask])
        thr = float(np.median(thrs[key]))
        sub[tmask] = (clf.predict_proba(Xt)[:, 1] >= thr).astype(int)
        print(f"  {key}: thr={thr:.2f} "
              f"weights={dict(zip(names, np.round(clf.coef_[0][:len(names)], 2)))}")

    C.write_submission("submission_meta.csv", [r["id"] for r in TEST], sub.tolist())


if __name__ == "__main__":
    main()
