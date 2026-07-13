# Approach_1 — independent math-solver verifier for no-context COMPUTATIONAL rows.
#
# Rationale: the closed-book knowledge wall is unbreakable, but ARITHMETIC/ALGEBRA
# word problems (age-ratio, profit/loss, simple interest, ratio partition) are
# reproducibly SOLVABLE by computation. The stack sometimes calls a wrong numeric
# answer "faithful" (missed hallucination) — the exact high-weight Phase-2 class.
# Pipeline (no internet needed):
#   For each computational row, Qwen-32B solves the problem from scratch with CoT,
#   sampled K times (self-consistency). We extract each chain's final number,
#   majority-vote, then compare to the candidate answer's number.
#   GATE (precision > recall): emit a non-0.5 verdict ONLY when
#     (a) >= CONS of K chains agree on a clean numeric answer, AND
#     (b) the candidate answer contains a comparable number.
#   match  -> P(faithful) high;  mismatch -> P(faithful) low;  else abstain 0.5.
# The gate self-limits: on rows the model can't cleanly solve, it abstains, so it
# cannot damage rows the stack already gets right. Writes signal_mathsolve.json.

import csv, json, re, subprocess, sys
from pathlib import Path
from collections import Counter

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
import torch
from vllm import LLM, SamplingParams

KIN = Path("/kaggle/input"); OUT = Path("/kaggle/working")
def find(part, ext=None):
    for root in (KIN, Path(".")):
        if not root.exists(): continue
        h = [p for p in root.rglob("*") if p.is_file() and part in p.name.lower()
             and (ext is None or p.suffix.lower() == ext)]
        if h: return sorted(h, key=lambda p: len(str(p)))[0]
    raise FileNotFoundError(part)

samples = json.load(open(find("samples", ".json"), encoding="utf-8"))
test_rows = list(csv.DictReader(open(find("test set", ".csv"), encoding="utf-8")))
print(f"samples {len(samples)}  test {len(test_rows)}")

def has_context(r): return str(r["context"]).strip() not in ("[NULL]", "NULL", "null", "")

# --- target ONLY no-context computational rows (gate handles the rest) ---
# Procedural word-problems the 32B solves reliably (profit/loss, interest, age/ratio
# partition, algebra). EXCLUDE factual-recall rows that merely contain a number or a
# "how much" word — the solver fabricates confident-but-wrong verdicts on those.
COMP = re.compile(r"শতকরা.*লাভ|শতকরা.*ক্ষতি|লাভ.*শতকরা|ক্ষতি.*শতকরা|সরল সুদ|মুনাফা|মূলধন"
                  r"|ক্রয়মূল্য|বিক্রয়মূল্য|বয়সের অনুপাত|বণ্টন|অংশীদার"
                  r"|যোগফল|গুণফল|লসাগু|গসাগু|x এর মান|সমীকরণ|উৎপাদক|=")
FACT = re.compile(r"আয়তন|জনসংখ্যা|সাল|কবে|কোথায়|রাজধানী|আবিষ্কার|প্রতিষ্ঠা|নদী"
                  r"|সদস্য সংখ্যা|ভাবার্থ|শাব্দিক অর্থ|সমার্থ|প্রতিশব্দ|বিপরীত"
                  r"|অনুচ্ছেদ|ধারা\b|সংবিধান|বিদ্যমান|কোন গ্যাস|কোন মৌল|আবিষ্কারক"
                  r"|রাষ্ট্র|দেশ|শহর|সাহিত্য|কবি|লেখক|গ্রন্থ|যুদ্ধ")
def is_comp(r):
    p = str(r["prompt_bn"])
    return (not has_context(r)) and bool(COMP.search(p)) and not FACT.search(p)

s_idx = [i for i, r in enumerate(samples) if is_comp(r)]
t_idx = [i for i, r in enumerate(test_rows) if is_comp(r)]
print(f"computational rows: samples={len(s_idx)} test={len(t_idx)}")

# --- Bengali-digit-aware numeric extraction ---
BN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
def to_ascii(s): return str(s).translate(BN)
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")
def numbers(s):
    """All numeric values in a string, fractions -> float. Strips thousands-commas
    (Bengali '৩,৩৪৮' -> 3348) which otherwise split into [3, 348] and fabricate
    false mismatches."""
    t = re.sub(r"(?<=\d),(?=\d)", "", to_ascii(s))
    out = []
    for m in NUM.findall(t):
        m = m.replace(" ", "")
        try:
            if "/" in m:
                a, b = m.split("/"); out.append(float(a) / float(b))
            else:
                out.append(float(m))
        except (ValueError, ZeroDivisionError):
            pass
    return out
def close(a, b, rel=0.02, absol=0.5):
    return abs(a - b) <= max(absol, rel * max(abs(a), abs(b)))

# Guards: abstain when the candidate answer is non-scalar or symbolic, where a
# numeric compare is invalid and would fabricate a false hallucination.
#   - √ / cube-root / π / symbolic radicals (e.g. "√3") -> abstain
#   - ratio answers "১:৯" and ratio questions ("অনুপাত") -> abstain
SYMBOLIC = re.compile(r"√|π|পাই|রুট|বর্গমূল|ঘনমূল|মূল\b|\broot\b")
def unsafe_candidate(resp, prompt):
    r = str(resp)
    if SYMBOLIC.search(r): return True
    if ":" in to_ascii(r) or "ঃ" in r: return True            # ratio answer
    if "{" in r or "}" in r or "সেট" in str(prompt): return True  # set/list answer
    if "অনুপাত" in str(prompt): return True                    # ratio question
    return False

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
K = 8; CONS = 5  # >=5/8 chains must agree on the answer
llm = LLM(model=MODEL, dtype="half", max_model_len=4096,
          tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.92)
SP = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=700, n=K)

def solve_prompt(row):
    return ("Solve this Bengali math problem step by step. Compute carefully.\n"
            "Do NOT use the answer given anywhere; derive it yourself.\n"
            "End your response with a line in EXACTLY this format:\n"
            "ANSWER: <final numeric value only, in digits>\n\n"
            f"Problem: {row['prompt_bn']}")

def chain_answer(text):
    """Extract the number after the last ANSWER: line; fallback to last number."""
    m = re.findall(r"ANSWER:\s*([^\n]+)", text, flags=re.I)
    if m:
        n = numbers(m[-1])
        if n: return n[-1]
    n = numbers(text)
    return n[-1] if n else None

def run(rows, idx):
    vals = [0.5] * len(rows)
    if not idx: return vals
    outs = llm.chat([[{"role": "user", "content": solve_prompt(rows[i])}] for i in idx], SP)
    conf = mism = ab = 0
    for i, o in zip(idx, outs):
        ans = [chain_answer(c.text) for c in o.outputs]
        ans = [a for a in ans if a is not None]
        cand = numbers(rows[i]["response_bn"])
        if len(ans) < CONS or not cand or unsafe_candidate(rows[i]["response_bn"], rows[i]["prompt_bn"]):
            ab += 1; continue
        # majority solver answer (bucket by closeness)
        buckets = []
        for a in ans:
            for b in buckets:
                if close(a, b[0]): b.append(a); break
            else:
                buckets.append([a])
        best = max(buckets, key=len)
        if len(best) < CONS:
            ab += 1; continue
        solved = sum(best) / len(best)
        if any(close(solved, c) for c in cand):
            vals[i] = 0.9; conf += 1
        else:
            vals[i] = 0.1; mism += 1
    print(f"  verdicts: faithful={conf}  hallucinated={mism}  abstain={ab}/{len(idx)}")
    return vals

print("solving samples..."); s_vals = run(samples, s_idx)
print("solving test...");    t_vals = run(test_rows, t_idx)
json.dump({"name": "mathsolve", "samples": s_vals,
           "test": {r["id"]: v for r, v in zip(test_rows, t_vals)}},
          open(OUT / "signal_mathsolve.json", "w"))
print("wrote signal_mathsolve.json")
