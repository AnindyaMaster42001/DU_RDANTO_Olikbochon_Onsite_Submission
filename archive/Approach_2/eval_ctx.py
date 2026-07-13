"""Gate the context-extraction lever against the stack's OOF on the same rows."""
import json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from gold_verify import GoldVerifier, equiv

B = "/mnt/NewVolume2/Android Projects/bengali-hallucination-detection/Approach_0/"
s = pd.DataFrame(json.load(open(B + "dataset samples.json")))
sp = pd.read_csv("kout2/ctx_spans_samples.csv")
stack = np.load("stack_oof_probs.npy")
covered = np.load("gold_covered_mask.npy")
hasctx = s.context.astype(str).str.strip().ne("[NULL]").values
y = s.label.values

def pred_from_span(resp, span):
    if not isinstance(span, str) or not span.strip(): return None
    ok, why = equiv(resp, span)
    return None if ok is None else int(ok)

preds = np.array([pred_from_span(r.response_bn, sp.span.iloc[i]) if hasctx[i] else None
                  for i, r in enumerate(s.itertuples())], dtype=object)
score = sp.score.values

def report(mask, tag, thr=-1e9):
    m = mask & np.array([p is not None for p in preds]) & (np.nan_to_num(score, nan=-1e9) >= thr)
    if m.sum() == 0: print(f"  {tag}: n=0"); return
    p = np.array([preds[i] for i in np.where(m)[0]], dtype=int)
    yy = y[m]
    st = (stack[m] > 0.5).astype(int)
    acc_new, acc_stack = (p == yy).mean(), (st == yy).mean()
    print(f"  {tag:26s} n={m.sum():3d}  extract_acc={acc_new:.4f}   stack_acc={acc_stack:.4f}   delta={acc_new-acc_stack:+.4f}")

print("=== context-extraction vs stack (labeled samples) ===")
for thr in (-1e9, 0, 2, 4, 6, 8):
    print(f"\n-- min span score {thr} --")
    report(hasctx, "ALL ctx", thr)
    report(hasctx & ~covered, "UNCOVERED ctx <-- target", thr)
    report(hasctx & covered, "covered ctx (sanity)", thr)

# where does it disagree with gold on covered rows? (cross-check the extractor)
print("\n=== extractor vs gold on covered ctx rows ===")
V = GoldVerifier()
agree = tot = 0
for i, r in enumerate(s.itertuples()):
    if not (hasctx[i] and covered[i]) or preds[i] is None: continue
    g = V.predict(r.prompt_bn, r.response_bn)[0]
    tot += 1; agree += (g == preds[i])
print(f"  agreement with gold verifier: {agree}/{tot} = {agree/max(tot,1):.3f}")
