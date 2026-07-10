"""
Final assembly:
  covered rows   -> gold-answer verification (98.9% on labeled samples)
  abstained rows -> meta-model over the 17 existing signals + context-span confidence

No hand-labeled test rows anywhere in this pipeline (rule 4b clean).
"""
import json, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from gold_verify import GoldVerifier

REPO = "/mnt/NewVolume2/Android Projects/bengali-hallucination-detection/"
s = pd.DataFrame(json.load(open(REPO + "Approach_0/dataset samples.json")))
t = pd.read_csv(REPO + "Approach_0/test set.csv")
y = s.label.values
sctx = s.context.astype(str).str.strip().ne("[NULL]").values
tctx = t.context.astype(str).str.strip().ne("[NULL]").values

# --- signals (samples + test) ---
def as_arr(v, ids=None):
    """samples -> list of floats; test -> dict keyed by row id (NOT iterable as values)."""
    if isinstance(v, dict):
        return np.array([np.nan if v.get(str(i)) is None else float(v[str(i)]) for i in ids])
    return np.array([np.nan if x is None else float(x) for x in v])

Xs, Xt, names = [], [], []
for p in sorted(glob.glob(REPO + "Approach_1/results/signal_*.json")):
    d = json.load(open(p))
    vs, vt = d.get("samples"), d.get("test")
    if not vs or not vt or len(vs) != len(s) or len(vt) != len(t): continue
    a, b = as_arr(vs), as_arr(vt, t.id.values)
    Xs.append(np.nan_to_num(a, nan=0.5)); Xt.append(np.nan_to_num(b, nan=0.5)); names.append(d["name"])
Xs = np.vstack(Xs).T; Xt = np.vstack(Xt).T
print(f"signals ({len(names)}): {names}")
print(f"mean check  samples={Xs.mean(0).round(3)}\n            test    ={Xt.mean(0).round(3)}")

# --- context-span confidence ---
ss = np.nan_to_num(pd.read_csv("kout2/ctx_spans_samples.csv").score.values, nan=0.0)
st = np.nan_to_num(pd.read_csv("kout2/ctx_spans_test.csv").score.values, nan=0.0)
Xs = np.column_stack([Xs, ss]); Xt = np.column_stack([Xt, st])

# --- honest OOF of this exact meta ---
def oof(X, seeds=5):
    P = np.zeros(len(s))
    for sd in range(seeds):
        for br in (True, False):
            idx = np.where(sctx == br)[0]
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X[idx], y[idx]):
                a, b = idx[tr], idx[te]
                m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X[a], y[a])
                P[b] += m.predict_proba(X[b])[:, 1]
    return P / seeds

V = GoldVerifier()
cov_s = np.array([V.predict(r.prompt_bn, r.response_bn)[0] is not None for r in s.itertuples()])
P = oof(Xs); mpred = (P > 0.5).astype(int)
gpred = np.array([V.predict(r.prompt_bn, r.response_bn)[0] if cov_s[i] else -1
                  for i, r in enumerate(s.itertuples())])
blend = np.where(cov_s, gpred, mpred)
print(f"\nOOF meta alone      : {(mpred==y).mean():.4f}")
print(f"OOF meta on uncovered: {(mpred[~cov_s]==y[~cov_s]).mean():.4f}  (n={(~cov_s).sum()})")
print(f"OOF gold+meta blend : {(blend==y).mean():.4f}   <-- pipeline estimate")

# --- fit on all samples, predict test ---
out = np.zeros(len(t), dtype=int)
for br in (True, False):
    si = np.where(sctx == br)[0]; ti = np.where(tctx == br)[0]
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xs[si], y[si])
    out[ti] = (m.predict_proba(Xt[ti])[:, 1] > 0.5).astype(int)

n_gold = 0
for i, r in enumerate(t.itertuples()):
    g = V.predict(r.prompt_bn, r.response_bn)[0]
    if g is not None: out[i] = g; n_gold += 1
print(f"\ntest: gold-decided {n_gold}, meta-decided {len(t)-n_gold}")

sub = pd.DataFrame({"id": t.id, "label": out})
sub.to_csv("submission_final.csv", index=False)
prev = pd.read_csv("submission_gold_v2.csv")
print("label dist:", sub.label.value_counts().to_dict())
print("rows differing from submitted 0.900:", (prev.label.values != sub.label.values).sum())
