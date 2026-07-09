"""Honest OOF gate for the enwiki cross-lingual signal.

Reproduces the exact 11-signal 0.828 stack (base8 + j32sv + ret32 + sa) from
results/, then measures what adding signal_enwiki does. Reports macro-F1 (public
LB proxy) AND class-0/hallucinated F1 (the real Phase-2 metric), split
overall/ctx/noctx, plus a focused readout on the enwiki-ACTIVE subset (rows the
signal actually grounded, != 0.5) and net rescues vs the baseline stack.

Gate (prior discipline, 299 rows): +enwiki must lift OOF beyond ~0.02 noise
(esp. noctx / factual-GK bucket) OR produce net rescues on the GK wall.

Usage:  python3 validate_enwiki.py
Reads signals from results/ (canonical) and signal_enwiki.json from cwd/results.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common as C

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FOLDS, SEEDS = 5, range(5)

# The exact 0.828 stack (RESULTS.md). j32fs/j32lp/ret32wikt excluded (measured to hurt/unused).
STACK = ["a0judge", "a0selfv", "crosslingual", "judge", "nli", "retrieval",
         "substring", "judge32", "j32sv", "ret32", "sa"]

SAMPLES, TEST = C.load_samples(), C.load_test()
Y = np.array([s["label"] for s in SAMPLES])
CTX_S = np.array([C.has_context(s) for s in SAMPLES])


def load_sig(name):
    for d in (RES, HERE):
        p = d / f"signal_{name}.json"
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    return None


SIGS = {}
for n in STACK + ["enwiki"]:
    s = load_sig(n)
    if s is not None:
        SIGS[n] = s

missing = [n for n in STACK if n not in SIGS]
if missing:
    sys.exit(f"missing baseline signals {missing} — cannot reproduce the 0.828 stack")
HAVE_ENWIKI = "enwiki" in SIGS
print(f"loaded {len(SIGS)} signals; enwiki present: {HAVE_ENWIKI}\n")


def col(name):
    return np.asarray(SIGS[name]["samples"], float)


def matrix(names):
    return np.column_stack([col(n) for n in names])


def best_threshold(y, prob, metric):
    ts = np.arange(0.30, 0.71, 0.02)
    ms = [metric(y.tolist(), (prob >= t).astype(int).tolist()) for t in ts]
    return float(ts[int(np.argmax(ms))])


def class0_f1(yt, yp):
    return C.f1_on_class(yt, yp, 0)[0]


def oof_preds(names, metric):
    """Seed-averaged per-branch OOF predictions -> (mean 0/1 pred per row over seeds)."""
    acc = np.zeros(len(Y))
    for seed in SEEDS:
        preds = np.zeros(len(Y), int)
        for mask in (CTX_S, ~CTX_S):
            X, y = matrix(names)[mask], Y[mask]
            oof, thrs = np.zeros(len(y)), []
            skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
            for tr, va in skf.split(X, y):
                clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
                oof[va] = clf.predict_proba(X[va])[:, 1]
                thrs.append(best_threshold(y[tr], clf.predict_proba(X[tr])[:, 1], metric))
            preds[mask] = (oof >= float(np.median(thrs))).astype(int)
        acc += preds
    return acc / len(SEEDS)  # fraction of seeds predicting faithful


def report(tag, names, metric, metric_name):
    p = oof_preds(names, metric)
    hard = (p >= 0.5).astype(int)
    idx_all = np.arange(len(Y))
    noc = idx_all[~CTX_S]
    ctx = idx_all[CTX_S]
    def m(ix):
        return metric(Y[ix].tolist(), hard[ix].tolist())
    print(f"[{metric_name}] {tag:<14} overall {m(idx_all):.4f}   "
          f"ctx {m(ctx):.4f}   noctx {m(noc):.4f}")
    return hard, p


def main():
    print("=" * 72)
    for metric, mname in [(C.macro_f1, "macroF1"), (class0_f1, "F1-hall")]:
        base_hard, base_soft = report("base (11-sig)", STACK, metric, mname)
        if HAVE_ENWIKI:
            enw_hard, enw_soft = report("+enwiki", STACK + ["enwiki"], metric, mname)
        print()

    if not HAVE_ENWIKI:
        print("signal_enwiki.json not found yet — run after the kernel completes.")
        return

    # --- focused analysis on the subset enwiki actually grounded (!= 0.5) ---
    ev = np.asarray(SIGS["enwiki"]["samples"], float)
    active = (np.abs(ev - 0.5) > 1e-6) & (~CTX_S)
    ai = np.arange(len(Y))[active]
    print("=" * 72)
    print(f"enwiki-ACTIVE labeled subset: {len(ai)} rows "
          f"({int((Y[ai]==0).sum())} hall / {int((Y[ai]==1).sum())} faithful)")
    if len(ai):
        stand = (ev[ai] >= 0.5).astype(int)
        acc = (stand == Y[ai]).mean()
        print(f"  enwiki standalone acc on active subset: {acc:.3f}")

    # net rescues: rows the base stack got wrong that +enwiki fixes (and vice-versa)
    base_hard, _ = report("base (11-sig)", STACK, C.macro_f1, "macroF1")
    enw_hard, _ = report("+enwiki", STACK + ["enwiki"], C.macro_f1, "macroF1")
    changed = np.arange(len(Y))[base_hard != enw_hard]
    fixed = [i for i in changed if enw_hard[i] == Y[i]]
    broke = [i for i in changed if base_hard[i] == Y[i]]
    print(f"\nrows changed by enwiki: {len(changed)}   fixed {len(fixed)}   broke {len(broke)}"
          f"   NET {len(fixed)-len(broke):+d}")
    print(f"  fixed idx  {fixed}")
    print(f"  broke idx  {broke}")
    print("\nGATE: pass if noctx/F1-hall lift > ~0.02 OR net rescues >= 4 with OOF not worse.")


if __name__ == "__main__":
    main()
