# Bengali Hallucination Detection — LLM judge (step 3, submission #2)
#
# Runs on Kaggle with GPU "T4 x2" (also works on a single P100).
# Paste this whole file into ONE cell of a Kaggle notebook attached to the
# competition, or upload the .ipynb version. Runtime: roughly 1-2 hours.
#
# Pipeline (three signals per row, computed with Qwen2.5-14B-Instruct-AWQ):
#   heuristic   (context rows)  response text found verbatim in the context
#   judge       (all rows)      grounding check if context, factuality if not
#   self-verify (no-context)    model answers the question itself, then a
#                               second call checks agreement with the response
# All strategy combinations are scored on the 299 labeled samples first;
# the best one by macro-F1 is applied to the test set -> submission.csv.

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import csv
import json
import re
from itertools import product
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
MAX_CTX_CHARS = 3500

# ---------------------------------------------------------------- data
KAGGLE_INPUT = Path("/kaggle/input/bengali-hallucination")
DATA = KAGGLE_INPUT if KAGGLE_INPUT.exists() else Path(".")


def find(part):
    hits = [p for p in DATA.iterdir() if part in p.name.lower()]
    assert hits, f"no file matching {part!r} in {DATA}"
    return hits[0]


samples = json.load(open(find("samples"), encoding="utf-8"))
test_rows = list(csv.DictReader(open(find("test"), encoding="utf-8")))
print(f"samples: {len(samples)}  test rows: {len(test_rows)}")


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


_PUNCT = re.compile(r"[\s।,.;:!?\"'()\[\]{}\-–—`~*_/\\]+")


def norm(s):
    return _PUNCT.sub("", str(s))


# ---------------------------------------------------------------- metric
def f1_cls(y_true, y_pred, cls):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def macro_f1(y_true, y_pred):
    return (f1_cls(y_true, y_pred, 0) + f1_cls(y_true, y_pred, 1)) / 2


# ---------------------------------------------------------------- model
llm = LLM(
    model=MODEL,
    dtype="half",  # T4/P100 have no bf16
    max_model_len=4096,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.92,
)
VERDICT = SamplingParams(temperature=0, max_tokens=5)
ANSWER = SamplingParams(temperature=0, max_tokens=80)


def chat(prompts, params):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, params)
    return [o.outputs[0].text.strip() for o in outs]


def to_verdict(text, default=0):
    up = text.upper()
    if "YES" in up:
        return 1
    if "NO" in up:
        return 0
    return default


# ---------------------------------------------------------------- prompts
def judge_prompt(row):
    q = str(row["prompt_bn"])
    a = str(row["response_bn"])
    if has_context(row):
        ctx = str(row["context"])[:MAX_CTX_CHARS]
        return (
            "You are verifying answers to Bengali questions against a source passage.\n\n"
            f"Passage:\n{ctx}\n\nQuestion: {q}\nCandidate answer: {a}\n\n"
            "Is the candidate answer correct AND supported by the passage? "
            "Reply with exactly one word: YES or NO."
        )
    return (
        "You are a careful fact-checker for Bengali question answering. "
        "Questions may involve Bengali grammar and literature, Bangladeshi "
        "history and culture, science, or mathematics.\n\n"
        f"Question: {q}\nCandidate answer: {a}\n\n"
        "Is the candidate answer factually correct? "
        "Reply with exactly one word: YES or NO."
    )


def self_answer_prompt(row):
    return (
        "Answer the following Bengali question concisely and accurately in Bengali. "
        "Give only the answer, no explanation.\n\n"
        f"Question: {row['prompt_bn']}"
    )


def compare_prompt(row, own_answer):
    return (
        "Two answers to the same Bengali question are given.\n\n"
        f"Question: {row['prompt_bn']}\n"
        f"Answer A: {row['response_bn']}\n"
        f"Answer B: {own_answer}\n\n"
        "Do Answer A and Answer B convey the same essential answer? "
        "Reply with exactly one word: YES or NO."
    )


# ---------------------------------------------------------------- signals
def compute_signals(rows, tag):
    n = len(rows)
    sig = {"heur": [None] * n, "judge": [None] * n, "selfv": [None] * n}

    for i, r in enumerate(rows):
        if has_context(r):
            resp = norm(r["response_bn"])
            sig["heur"][i] = 1 if resp and resp in norm(r["context"]) else 0

    print(f"[{tag}] judge pass ({n} prompts)...")
    for i, out in enumerate(chat([judge_prompt(r) for r in rows], VERDICT)):
        sig["judge"][i] = to_verdict(out)

    noctx_idx = [i for i, r in enumerate(rows) if not has_context(r)]
    print(f"[{tag}] self-verify pass ({len(noctx_idx)} x2 prompts)...")
    own = chat([self_answer_prompt(rows[i]) for i in noctx_idx], ANSWER)
    comps = chat(
        [compare_prompt(rows[i], a) for i, a in zip(noctx_idx, own)], VERDICT
    )
    for i, out in zip(noctx_idx, comps):
        sig["selfv"][i] = to_verdict(out)

    with open(f"signals_{tag}.json", "w") as f:
        json.dump(sig, f)
    return sig


# ---------------------------------------------------------------- strategies
CTX_MODES = {
    "heur": lambda s, i: s["heur"][i],
    "judge": lambda s, i: s["judge"][i],
    "and": lambda s, i: s["heur"][i] & s["judge"][i],
    "or": lambda s, i: s["heur"][i] | s["judge"][i],
}
NOCTX_MODES = {
    "judge": lambda s, i: s["judge"][i],
    "selfv": lambda s, i: s["selfv"][i],
    "and": lambda s, i: s["judge"][i] & s["selfv"][i],
    "or": lambda s, i: s["judge"][i] | s["selfv"][i],
}


def predict(rows, sig, ctx_mode, noctx_mode):
    return [
        CTX_MODES[ctx_mode](sig, i)
        if has_context(r)
        else NOCTX_MODES[noctx_mode](sig, i)
        for i, r in enumerate(rows)
    ]


# ---------------------------------------------------------------- validate
sig_s = compute_signals(samples, "samples")
y_true = [s["label"] for s in samples]

best, results = None, []
for cm, nm in product(CTX_MODES, NOCTX_MODES):
    preds = predict(samples, sig_s, cm, nm)
    m, f0 = macro_f1(y_true, preds), f1_cls(y_true, preds, 0)
    results.append((m, f0, cm, nm))
    if best is None or m > best[0]:
        best = (m, f0, cm, nm)

print(f"\n{'ctx':<7}{'noctx':<7}{'macroF1':<9}F1(hall)")
for m, f0, cm, nm in sorted(results, reverse=True):
    print(f"{cm:<7}{nm:<7}{m:<9.4f}{f0:.4f}")
m, f0, cm, nm = best
print(f"\nBEST: ctx={cm} noctx={nm}  macroF1={m:.4f}  F1(hallucinated)={f0:.4f}")

ctx_i = [i for i, s in enumerate(samples) if has_context(s)]
noc_i = [i for i, s in enumerate(samples) if not has_context(s)]
bp = predict(samples, sig_s, cm, nm)
print(f"  with context  macroF1={macro_f1([y_true[i] for i in ctx_i], [bp[i] for i in ctx_i]):.4f}")
print(f"  no context    macroF1={macro_f1([y_true[i] for i in noc_i], [bp[i] for i in noc_i]):.4f}")

# ---------------------------------------------------------------- test
sig_t = compute_signals(test_rows, "test")
test_preds = predict(test_rows, sig_t, cm, nm)

with open("submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label"])
    for r, p in zip(test_rows, test_preds):
        w.writerow([r["id"], p])

n0 = test_preds.count(0)
print(f"\nwrote submission.csv: {len(test_preds)} rows "
      f"({n0} hallucinated / {len(test_preds) - n0} faithful)")
