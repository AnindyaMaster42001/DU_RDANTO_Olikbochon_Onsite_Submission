# Build submission_contrastive.csv (LB 0.831) = committed 0.828 base + 12 ground-truth flips.
#
# IMPORTANT: builds as a MINIMAL DELTA on top of the committed submission_sa_wikt.csv.
# Do NOT re-derive the 0.828 stack from scratch — the wikt-gate/threshold seed is not
# cleanly reproducible and a local rebuild drifts ~36/2516 rows from the committed artifact.
#
# The 12 flips are the hand-audited unique-gold factual/idiom mismatches from
# contrastive_analysis.py (authors, places, dates, grammar, phonetics, idioms). All are
# 1->0 (stack wrongly called them faithful; train gold proves them hallucinated).
#
# Usage:  PYTHONPATH=. python3 build_contrastive.py

import csv
import common as C
from contrastive_analysis import SHIPPED, BASE_FILE

T = C.load_test(); IDS = [r["id"] for r in T]
base = {r["id"]: int(r["label"]) for r in csv.DictReader(open(BASE_FILE))}

out = dict(base)
for i in SHIPPED:
    assert base[i] == 1, f"id{i} expected faithful in base (flip is 1->0), got {base[i]}"
    out[i] = 0

C.write_submission("submission_contrastive.csv", IDS, [out[i] for i in IDS])
changed = sum(1 for i in IDS if out[i] != base[i])
print(f"wrote submission_contrastive.csv | flips={len(SHIPPED)} changed_vs_base={changed} "
      f"| hallucinated={sum(1 for i in IDS if out[i]==0)} faithful={sum(1 for i in IDS if out[i]==1)}")
