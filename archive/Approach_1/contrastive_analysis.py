# Contrastive ground-truth analysis — the lever that moved LB 0.828 -> 0.831 (#31 -> #30).
#
# IDEA (100% legal; uses TRAIN/sample labels only, never test labels):
#   Many test prompts are STANDARD Bangla exam questions that also appear, verbatim,
#   among the 299 labeled sample rows. When a FAITHFUL (label==1) sample row shares a
#   test row's prompt, that sample's response is the GOLD answer for that prompt.
#   If a test row's candidate answer DIFFERS from the unique gold, the candidate is a
#   CERTIFIED hallucination — regardless of what the 11-signal stack predicted.
#
# WHY IT TRANSFERS TO THE PRIVATE LB (unlike public-LB noise):
#   These flips are ground-truth-certain (a known-correct answer contradicts the
#   candidate). They fix genuinely wrong labels, so they help BOTH the public macro-F1
#   AND the real Phase-2 metric (binary F1 on the hallucinated class) — every flip is a
#   1->0, i.e. it ADDS a true positive to class 0.
#
# SCOPE / CEILING:
#   Only 299 labeled rows exist (no larger train set — verified). 163 of them are
#   faithful golds. Exactly 18 test rows have a candidate that differs from a UNIQUE
#   gold. After hand-audit, 12 are rock-solid; 6 are excluded (see below). This is the
#   HARD CEILING of the exact-prompt lever. The fuzzy extension (see audit_fuzzy) adds
#   ZERO safe flips — do not lower the jaccard threshold, it breaks the guarantee.
#
# Usage:  PYTHONPATH=. python3 contrastive_analysis.py
# This is READ-ONLY (prints the audit). Use build_contrastive.py to write the submission.

import csv, re
import common as C

BASE_FILE = "submission_sa_wikt.csv"   # the committed 0.828 artifact (canonical base)

BN = {"০":"0","১":"1","২":"2","৩":"3","৪":"4","৫":"5","৬":"6","৭":"7","৮":"8","৯":"9"}
def ascii_digits(s): return "".join(BN.get(c, c) for c in str(s))

def na(s):
    """Answer normalizer: Bengali digits -> ascii, strip thousands-comma, drop punct/space."""
    t = ascii_digits(s)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    return re.sub(r"[\s।,.\-–—():;{}\[\]'\"/?!]+", "", t).strip().lower()

def pn(s):
    """Prompt normalizer: digits -> ascii, drop quotes/punct/space (for exact-prompt key)."""
    t = ascii_digits(s)
    return re.sub(r"[\s।,.\-–—():;{}\[\]'\"‘’“”‌‍]+", "", t).strip().lower()

def toks(s):
    """Token set for fuzzy prompt similarity (jaccard)."""
    t = ascii_digits(s)
    t = re.sub(r"[।,.\-–—():;{}\[\]'\"‘’“”?!]+", " ", t)
    return set(w for w in t.lower().split() if len(w) > 1)


def exact_prompt_audit(S, T, base):
    """The 18 exact-prompt candidates. Returns (flip_hall, flip_faith)."""
    gold, hall = {}, {}
    for s in S:
        p, a = pn(s["prompt_bn"]), na(s["response_bn"])
        (gold if s["label"] == 1 else hall).setdefault(p, set()).add(a)

    agree = 0; flip_hall = []; flip_faith = []; multi = 0
    for r in T:
        p, a, cur = pn(r["prompt_bn"]), na(r["response_bn"]), base[r["id"]]
        if p not in gold: continue
        if a in gold[p]:                         tl = 1                     # matches a known gold
        elif p in hall and a in hall[p]:         tl = 0                     # matches a known wrong answer
        elif len(gold[p]) == 1:                  tl = 0                     # differs from the UNIQUE gold
        else:                                    multi += 1; continue       # ambiguous (>1 gold), skip
        if tl == cur: agree += 1
        elif tl == 0: flip_hall.append((r["id"], cur, r, sorted(gold[p])[0]))
        else:         flip_faith.append((r["id"], cur, r))
    print(f"[exact] gold prompts={len(gold)} agree={agree} "
          f"flip->hall={len(flip_hall)} flip->faithful={len(flip_faith)} multi_gold_skipped={multi}")
    for i, c, r, g in flip_hall:
        print(f"  id{i} {c}->0 | Q={str(r['prompt_bn'])[:42]} | cand='{str(r['response_bn'])[:22]}' gold='{g[:22]}'")
    return flip_hall, flip_faith


def audit_fuzzy(S, T, base, jmin=0.70):
    """Near-duplicate prompts (not byte-identical). DEMONSTRATES the lever is exhausted:
    every candidate is a semantic-drift trap, NOT a safe flip. Kept so the team can see why."""
    exact = set(pn(s["prompt_bn"]) for s in S)
    trn = [(toks(s["prompt_bn"]), na(s["response_bn"]), s["label"], s) for s in S]
    print(f"\n[fuzzy jaccard>={jmin}] near-dup candidates (train faithful, not exact-match):")
    for r in T:
        if pn(r["prompt_bn"]) in exact: continue
        tp, a = toks(r["prompt_bn"]), na(r["response_bn"])
        if len(tp) < 3: continue
        for tt, ta, tl, ts in trn:
            if not tt or tl != 1: continue
            j = len(tp & tt) / len(tp | tt)
            if j >= jmin:
                same = "SAME" if a == ta else "DIFF"
                print(f"  id{r['id']} cur={base[r['id']]} j={j:.2f} {same} | "
                      f"test='{str(r['prompt_bn'])[:34]}' vs train='{str(ts['prompt_bn'])[:34]}' "
                      f"| cand='{str(r['response_bn'])[:18]}' gold='{str(ts['response_bn'])[:18]}'")
                break
    print("  VERDICT: 0 safe flips. jaccard>=0.7 prompts are DIFFERENT questions "
          "(e.g. 'when born' vs 'where born') -> fuzzy contrastive is UNSAFE. Do not use.")


# The 12 shipped flips and WHY the other 6 candidates were excluded (hand-audit record).
SHIPPED = ["693","725","816","823","1490","1528","1797","1903","2103","82","113","122"]
EXCLUDED = {
    "891":  "১৮৩২ == আঠারশ বত্রিশ সালে (same year, numeral vs words) -> candidate CORRECT",
    "2373": "২রা নভেম্বর ১৮৮৬ == ২ নভেম্বর ১৮৮৬ (same date, surface-form) -> candidate CORRECT",
    "5":    "প্রখর রোদ is a valid paraphrase of খটখটে রোদ -> not a unique-gold mismatch",
    "98":   "নিয়মের ব্যতিক্রমে সিদ্ধ is the correct standard gloss -> candidate arguably MORE correct",
    "2015": "which-is-NOT MCQ ('কোনটি নয়') -> gold uniqueness unconfirmable without the choice set",
    "2028": "which-is-NOT MCQ -> gold uniqueness unconfirmable without the choice set",
}

if __name__ == "__main__":
    S, T = C.load_samples(), C.load_test()
    base = {r["id"]: int(r["label"]) for r in csv.DictReader(open(BASE_FILE))}
    exact_prompt_audit(S, T, base)
    audit_fuzzy(S, T, base)
    print(f"\nSHIPPED {len(SHIPPED)} flips: {SHIPPED}")
    print("EXCLUDED (with reasons):")
    for i, why in EXCLUDED.items():
        print(f"  id{i}: {why}")
