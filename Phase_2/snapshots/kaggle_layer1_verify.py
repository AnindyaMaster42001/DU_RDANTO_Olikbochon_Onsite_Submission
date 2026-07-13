# Layer-1 offline verification (CPU kernel, INTERNET OFF).
#
# Confirms the gold-verification layer runs inside a Kaggle code-competition kernel against
# the real competition test set, with only the attached public corpora -- no GPU, no net.
# This is the decisive layer (it carries the entire 0.831 -> 0.904 gain), so a green run
# here validates the reproducible half of the Phase-2 package end to end.
#
# Attach as inputs: the competition data + the bengali-gold-corpora dataset + this repo's
# gold_verify.py & bnnum.py (bundled inside the corpora dataset). Internet OFF.

import glob
import json
import os
import sys
from collections import Counter

import pandas as pd

INP = "/kaggle/input"


def find(pattern):
    hits = glob.glob(f"{INP}/**/{pattern}", recursive=True)
    assert hits, f"no input matching {pattern}"
    return sorted(hits, key=len)[0]


# make gold_verify + bnnum importable (bundled in the corpora dataset)
for gv in glob.glob(f"{INP}/**/gold_verify.py", recursive=True):
    sys.path.insert(0, os.path.dirname(gv))
    break
# point the verifier at the attached corpora root (dir that holds mmlu/, bcs/, ...)
for m in glob.glob(f"{INP}/**/bangla_mmlu_all.parquet", recursive=True):
    os.environ["BHD_EXT"] = os.path.dirname(os.path.dirname(m)) + "/"
    break
print("BHD_EXT =", os.environ.get("BHD_EXT"))

from gold_verify import GoldVerifier  # noqa: E402

samples = pd.DataFrame(json.load(open(find("*samples*.json"))))
test = pd.read_csv(find("test*.csv"))
print(f"samples={len(samples)} test={len(test)}")

V = GoldVerifier(with_squad=True)
print(f"corpora: hallueval={len(V.qa)} idioms={len(V.idi)} mmlu={len(V.mmlu)} "
      f"bcs={len(V.bcs)} extra={len(V.extra)}")

# --- accuracy on the labeled samples ---
per = {}
n_ok = n = 0
for r in samples.itertuples():
    p, src, why, gold = V.predict(r.prompt_bn, r.response_bn)
    if p is None:
        continue
    d = per.setdefault(src, [0, 0])
    d[0] += 1
    n += 1
    if p == r.label:
        d[1] += 1
        n_ok += 1
print("\n=== sample accuracy by source ===")
for src, (c, ok) in sorted(per.items()):
    print(f"  {src:12s} covered={c:3d} correct={ok:3d} acc={ok / c:.4f}")
print(f"  TOTAL covered={n}/{len(samples)} ({n / len(samples):.1%}) acc={n_ok / n:.4f}")

# --- coverage + label distribution on the test set ---
preds = {}
cov = Counter()
for r in test.itertuples():
    p, src, why, gold = V.predict(r.prompt_bn, r.response_bn)
    if p is None:
        cov["uncovered"] += 1
        continue
    cov[f"{src}:pred{p}"] += 1
    preds[r.id] = int(p)
print("\n=== test coverage ===")
for k, v in sorted(cov.items()):
    print(f"  {k:22s} {v}")
print(f"  covered {len(preds)}/{len(test)} = {len(preds) / len(test):.1%}")
print(f"  gold label dist: {Counter(preds.values())}")
json.dump({str(k): v for k, v in preds.items()},
          open("/kaggle/working/gold_preds_kaggle.json", "w"))
print("\nwrote gold_preds_kaggle.json -- compare offline to the committed gold_preds_test.json")
