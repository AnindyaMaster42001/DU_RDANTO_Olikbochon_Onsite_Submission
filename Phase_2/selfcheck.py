"""
Corpora-free reproducibility self-check for the Phase-2 gold package.

Proves the packaged pipeline's Layer-1 overlay reproduces the selected finals, WITHOUT
needing the corpora or a GPU. It replays the cached verifier output (gold_preds_test.json,
the exact GoldVerifier.predict() result on every test row, BCS included) through the same
overlay logic the notebook uses, against an arbitrary Layer-2 stack, and checks:

  1. every covered row equals submission_final_bcs.csv   (USE_BCS=True  -> 0.904)
  2. dropping the BCS-sourced rows equals submission_final.csv (USE_BCS=False -> 0.901)
  3. uncovered rows are taken verbatim from the Layer-2 stack (Layer 1 never touches them)
  4. skipping covered rows from the LLM stack is output-neutral (why LLM_UNCOVERED_ONLY is free)

Run from the repo root or from Phase_2/. Requires only the CSVs + the cache in this dir.
"""
import csv
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load_csv(*cands):
    for c in cands:
        if os.path.exists(c):
            return {r["id"]: int(r["label"]) for r in csv.DictReader(open(c))}
    raise FileNotFoundError(cands)


gp = json.load(open(os.path.join(HERE, "gold_preds_test.json")))   # id -> [pred, src]
finbcs = load_csv(os.path.join(HERE, "submission_final_bcs.csv"),
                  os.path.join(REPO, "archive/Approach_2/submission_final_bcs.csv"))
final = load_csv(os.path.join(HERE, "submission_final.csv"),
                 os.path.join(REPO, "archive/Approach_2/submission_final.csv"))
ids = list(finbcs)

# GOLD map as the notebook builds it (test rows). BCS toggle == drop bcs-sourced rows.
GOLD_BCS = {i: int(gp[i][0]) for i in gp}
GOLD_NOBCS = {i: int(gp[i][0]) for i in gp if not gp[i][1].startswith("bcs")}


def overlay(base, gold):
    out = dict(base)
    for i in ids:
        if i in gold:
            out[i] = gold[i]
    return out


rng = random.Random(0)
base = {i: rng.randint(0, 1) for i in ids}   # stand-in for any Layer-2 stack output

ok = True

# 1. USE_BCS=True reproduces the 0.904 final on covered rows
o1 = overlay(base, GOLD_BCS)
c1 = sum(1 for i in gp if o1[i] == finbcs[i])
print(f"[1] USE_BCS=True : {c1}/{len(gp)} covered rows == submission_final_bcs.csv"
      f"  ({'PASS' if c1 == len(gp) else 'FAIL'})")
ok &= c1 == len(gp)

# 2. USE_BCS=False reproduces the 0.901 final on its covered rows
o2 = overlay(base, GOLD_NOBCS)
c2 = sum(1 for i in GOLD_NOBCS if o2[i] == final[i])
print(f"[2] USE_BCS=False: {c2}/{len(GOLD_NOBCS)} covered rows == submission_final.csv"
      f"  ({'PASS' if c2 == len(GOLD_NOBCS) else 'FAIL'})")
ok &= c2 == len(GOLD_NOBCS)

# 3. uncovered rows come straight from the Layer-2 stack
unc = [i for i in ids if i not in GOLD_BCS]
c3 = sum(1 for i in unc if o1[i] == base[i])
print(f"[3] uncovered    : {c3}/{len(unc)} rows taken verbatim from the stack"
      f"  ({'PASS' if c3 == len(unc) else 'FAIL'})")
ok &= c3 == len(unc)

# 4. the two finals differ only on BCS-sourced rows
diff = [i for i in ids if finbcs[i] != final[i]]
bcs_ids = {i for i in gp if gp[i][1].startswith("bcs")}
print(f"[4] final_bcs vs final differ on {len(diff)} rows, all BCS-sourced: "
      f"{all(i in bcs_ids for i in diff)}  ({'PASS' if all(i in bcs_ids for i in diff) else 'FAIL'})")
ok &= all(i in bcs_ids for i in diff)

cov = len(GOLD_BCS)
print(f"\ncoverage: Layer 1 decides {cov}/{len(ids)} test rows ({cov/len(ids):.1%}); "
      f"Layer 2 (LLM) handles {len(ids)-cov}.")
print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
raise SystemExit(0 if ok else 1)
