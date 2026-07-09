"""Gate for signal_mathsolve — a GATED OVERRIDE on the committed 0.828 file.

mathsolve is sparse (confident only on computational rows it can cleanly solve),
so it is applied like the wikt rescue: on rows where it is confident (>=HI faithful
or <=LO hallucinated), override the 0.828 label; elsewhere keep 0.828.

Reports:
  - labeled 299: net correct flips vs current (truth known) — MUST be net-positive,
    zero regressions after the symbolic/ratio guards.
  - test 2516: how many overrides, direction, and an audit sample for eyeballing.

Usage:  python3 validate_mathsolve.py            # reads results/signal_mathsolve.json
Writes submission_mathsolve.csv (override applied) only if labeled net flips >= +1.
"""
import csv, json, os, re, sys
import numpy as np
import common as C

HI, LO = 0.8, 0.2
S = C.load_samples(); T = C.load_test(); Y = np.array([s["label"] for s in S])
IDS = [r["id"] for r in T]; rb = {r["id"]: r for r in T}

sig_p = next((p for p in ("results/signal_mathsolve.json", "signal_mathsolve.json") if os.path.exists(p)), None)
if not sig_p:
    sys.exit("signal_mathsolve.json not found yet — run after the kernel completes.")
MS = json.load(open(sig_p, encoding="utf-8"))

base_test = {r["id"]: int(r["label"]) for r in csv.DictReader(open("submission_sa_wikt.csv"))}
sv = np.asarray(MS["samples"], float)

# --- labeled: reconstruct base 299 preds are not needed; measure mathsolve verdict vs truth ---
def verdict(v): return 1 if v >= HI else (0 if v <= LO else None)
lab_conf = [(i, verdict(sv[i]), Y[i]) for i in range(len(S)) if verdict(sv[i]) is not None]
correct = sum(1 for _, p, y in lab_conf if p == y)
wrong = [(i, p, y) for i, p, y in lab_conf if p != y]
print(f"labeled confident verdicts: {len(lab_conf)}  correct {correct}  wrong {len(wrong)}")
for i, p, y in wrong:
    print(f"  WRONG pred={p} true={y} | {str(S[i]['prompt_bn'])[:60]} || {str(S[i]['response_bn'])[:25]}")

# --- test: apply as override on 0.828, count flips, audit ---
flips = []
for i in IDS:
    v = MS["test"].get(i, 0.5); p = verdict(v)
    if p is not None and p != base_test[i]:
        flips.append((i, base_test[i], p, v))
up = [f for f in flips if f[2] == 1]; dn = [f for f in flips if f[2] == 0]
print(f"\ntest overrides on 0.828: {len(flips)}  (0->1 faithful: {len(up)}, 1->0 hallucinated: {len(dn)})")
print("audit sample:")
for i, b, p, v in flips[:20]:
    print(f"  id{i} {b}->{p} P={v:.2f} | {str(rb[i]['prompt_bn'])[:52]} || {str(rb[i]['response_bn'])[:22]}")

net = correct - len(wrong)
print(f"\nlabeled NET (correct-wrong) = {net:+d}")
if net >= 1 and len(wrong) == 0:
    out = dict(base_test)
    for i, b, p, v in flips: out[i] = p
    C.write_submission("submission_mathsolve.csv", IDS, [out[i] for i in IDS])
    print("GATE PASS -> wrote submission_mathsolve.csv")
else:
    print("GATE: labeled not clean-positive; inspect before submitting.")
