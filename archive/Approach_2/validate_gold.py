import json, pandas as pd, numpy as np
from collections import Counter, defaultdict
from gold_verify import GoldVerifier

B = "/mnt/NewVolume2/Android Projects/bengali-hallucination-detection/Approach_0/"
s = pd.DataFrame(json.load(open(B + "dataset samples.json")))
t = pd.read_csv(B + "test set.csv")
V = GoldVerifier()
print(f"corpora: hallueval={len(V.qa)} idioms={len(V.idi)} mmlu={len(V.mmlu)}\n")

# ---- validate on the 299 labeled samples ----
per = defaultdict(lambda: {"n": 0, "ok": 0})
errs = []
for r in s.itertuples():
    p, src, why, gold = V.predict(r.prompt_bn, r.response_bn)
    if p is None: continue
    per[src]["n"] += 1
    if p == r.label: per[src]["ok"] += 1
    else: errs.append((src, why, r.label, str(r.prompt_bn)[:40], str(r.response_bn)[:30], str(gold)[:34]))

print("=== ACCURACY ON LABELED SAMPLES (by source) ===")
tot_n = tot_ok = 0
for src, d in per.items():
    tot_n += d["n"]; tot_ok += d["ok"]
    print(f"  {src:12s} covered={d['n']:3d}  correct={d['ok']:3d}  acc={d['ok']/d['n']:.4f}")
print(f"  {'TOTAL':12s} covered={tot_n:3d}/{len(s)} ({tot_n/len(s):.1%})  acc={tot_ok/tot_n:.4f}")

print(f"\n=== {len(errs)} ERRORS ===")
for e in errs: print("  ", e)

# ---- coverage on test ----
print("\n=== TEST COVERAGE ===")
cov = Counter(); preds = {}
for r in t.itertuples():
    p, src, why, gold = V.predict(r.prompt_bn, r.response_bn)
    if p is None: cov["uncovered"] += 1; continue
    cov[f"{src}:pred{p}"] += 1; preds[r.id] = (p, src)
for k, v in sorted(cov.items()): print(f"  {k:22s} {v}")
print(f"  covered total: {len(preds)}/{len(t)} = {len(preds)/len(t):.1%}")
json.dump({str(k): v for k, v in preds.items()}, open("gold_preds_test.json", "w"))

# ---- how much would this change our best submission? ----
sub = pd.read_csv("/mnt/NewVolume2/Android Projects/bengali-hallucination-detection/Approach_1/submission_contrastive.csv")
sub = sub.set_index("id").label.to_dict()
chg = Counter()
for i, (p, src) in preds.items():
    chg[f"{src}:{sub[i]}->{p}"] += 1
print("\n=== DELTA vs submission_contrastive (LB 0.831) ===")
for k, v in sorted(chg.items()): print(f"  {k:26s} {v}")
flips = sum(v for k, v in chg.items() if k.split(":")[1].split("->")[0] != k.split("->")[1])
print(f"  total flips: {flips} / {len(preds)} covered ({flips/len(t):.1%} of test)")
