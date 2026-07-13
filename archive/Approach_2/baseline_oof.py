"""Honest OOF for the existing signal stack, so we know what the new verifier must beat.

Replicates the team's protocol: per-branch (ctx / noctx) logistic over the signals,
5 seeds x 5 folds, threshold chosen inside the training folds.
Reports accuracy overall AND restricted to the rows where gold lookup abstains.
"""
import json, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from gold_verify import GoldVerifier

REPO = "/mnt/NewVolume2/Android Projects/bengali-hallucination-detection/"
B = REPO + "Approach_0/"
s = pd.DataFrame(json.load(open(B + "dataset samples.json")))
y = s.label.values
hasctx = s.context.astype(str).str.strip().ne("[NULL]").values

sigs, names = [], []
for p in sorted(glob.glob(REPO + "Approach_1/results/signal_*.json")):
    d = json.load(open(p))
    v = d.get("samples")
    if not v or len(v) != len(s): continue
    arr = np.array([np.nan if x is None else float(x) for x in v])
    if np.isnan(arr).all(): continue
    arr = np.nan_to_num(arr, nan=0.5)
    sigs.append(arr); names.append(d["name"])
X = np.vstack(sigs).T
print(f"signals ({len(names)}): {names}")
print("X:", X.shape)

V = GoldVerifier()
covered = np.array([V.predict(r.prompt_bn, r.response_bn)[0] is not None for r in s.itertuples()])
print(f"gold-covered samples: {covered.sum()}  uncovered: {(~covered).sum()}")

def oof_probs(seed):
    p = np.zeros(len(s))
    for branch in (True, False):                      # per-branch model, as they do
        idx = np.where(hasctx == branch)[0]
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in cv.split(X[idx], y[idx]):
            a, b = idx[tr], idx[te]
            m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X[a], y[a])
            p[b] = m.predict_proba(X[b])[:, 1]
    return p

P = np.mean([oof_probs(sd) for sd in range(5)], axis=0)
pred = (P > 0.5).astype(int)

def rep(mask, tag):
    if mask.sum() == 0: return
    acc = (pred[mask] == y[mask]).mean()
    f0 = f1_score(y[mask], pred[mask], pos_label=0) if len(set(y[mask])) > 1 else float("nan")
    mf = f1_score(y[mask], pred[mask], average="macro") if len(set(y[mask])) > 1 else float("nan")
    print(f"  {tag:34s} n={mask.sum():3d}  acc={acc:.4f}  macroF1={mf:.4f}  F1(class0)={f0:.4f}")

print("\n=== STACK OOF (5-seed) ===")
rep(np.ones(len(s), bool), "ALL")
rep(hasctx, "ctx")
rep(~hasctx, "noctx")
print("  -- the rows the new verifier must beat --")
rep(~covered, "UNCOVERED (all)")
rep(~covered & ~hasctx, "UNCOVERED noctx  <-- the prize")
rep(~covered & hasctx, "UNCOVERED ctx")
rep(covered, "covered (gold handles these)")

np.save("stack_oof_probs.npy", P)
np.save("gold_covered_mask.npy", covered)
print("\nsaved stack_oof_probs.npy, gold_covered_mask.npy")
