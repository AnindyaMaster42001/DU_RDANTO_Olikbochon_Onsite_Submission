"""Override the LB-0.831 stack with gold-verified labels wherever a gold answer exists."""
import json, pandas as pd
from collections import Counter
from gold_verify import GoldVerifier

REPO = "/mnt/NewVolume2/Android Projects/bengali-hallucination-detection/"
B = REPO + "Approach_0/"
t = pd.read_csv(B + "test set.csv")
base = pd.read_csv(REPO + "Approach_1/submission_contrastive.csv")   # LB 0.831
assert list(base.columns) == ["id", "label"] and len(base) == len(t)

V = GoldVerifier()
base_map = dict(zip(base.id, base.label))
out, meta = {}, Counter()
for r in t.itertuples():
    p, src, why, gold = V.predict(r.prompt_bn, r.response_bn)
    if p is None:
        out[r.id] = base_map[r.id]; meta["kept_stack"] += 1
    else:
        out[r.id] = p
        meta["gold_override" if p != base_map[r.id] else "gold_agree"] += 1
        meta[f"src_{src}"] += 1

sub = pd.DataFrame({"id": t.id, "label": [out[i] for i in t.id]})
sub.to_csv("submission_gold.csv", index=False)

print("=== submission_gold.csv ===")
for k, v in sorted(meta.items()): print(f"  {k:16s} {v}")
print(f"\nlabel dist: base={Counter(base.label)}  gold={Counter(sub.label)}")
print(f"rows changed: {(base.label.values != sub.label.values).sum()} / {len(sub)}")
print(f"class-0 (hallucinated) count: base={(base.label==0).sum()} -> new={(sub.label==0).sum()}")
