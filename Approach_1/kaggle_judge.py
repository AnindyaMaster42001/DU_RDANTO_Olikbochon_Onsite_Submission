"""Approach_1 · Step 2 (Kaggle) — LLM judge as a continuous signal.

Self-contained: paste this whole file into ONE cell of a Kaggle notebook
attached to the competition, on a GPU accelerator (T4 x2 or P100). Runtime
~1-2h. Open-weight model => Phase-2 code-competition compliant.

Upgrades over Approach_0/kaggle_judge.py:
  - FEW-SHOT + one-line CoT before the verdict.
  - SELF-CONSISTENCY for no-context rows: sample the factuality verdict N times
    at temperature and use the YES-fraction as a calibrated P(faithful), instead
    of one brittle greedy call.
  - Emits signal_judge.json (continuous P(faithful) for samples + test) so it
    stacks with signal_nli.json in ensemble.py — not just a hard submission.

Outputs:
  submission_judge.csv   hard 0/1 (threshold tuned per-branch on the 299 samples)
  signal_judge.json      {"name","samples":[299],"test":{id:P(faithful)}}
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"   # 32B-AWQ / Gemma-2-27B also fit <9h
MAX_CTX_CHARS = 3500
N_VOTES = 5                                # self-consistency samples (no-context)

# ---------------------------------------------------------------- data
KAGGLE = Path("/kaggle/input")
DATA = next((p for p in [KAGGLE / "bengali-hallucination", *KAGGLE.glob("*")]
             if p.is_dir() and any("test" in f.name.lower() for f in p.iterdir())),
            Path("."))


def find(part):
    hits = [p for p in DATA.iterdir() if part in p.name.lower()]
    assert hits, f"no file matching {part!r} in {DATA}"
    return hits[0]


samples = json.load(open(find("samples"), encoding="utf-8"))
test_rows = list(csv.DictReader(open(find("test"), encoding="utf-8")))
print(f"samples: {len(samples)}  test rows: {len(test_rows)}")


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


# ---------------------------------------------------------------- metric
def f1_cls(y, p, c):
    tp = sum(1 for t, q in zip(y, p) if t == c and q == c)
    fp = sum(1 for t, q in zip(y, p) if t != c and q == c)
    fn = sum(1 for t, q in zip(y, p) if t == c and q != c)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def macro_f1(y, p):
    return (f1_cls(y, p, 0) + f1_cls(y, p, 1)) / 2


# ---------------------------------------------------------------- model
llm = LLM(
    model=MODEL,
    dtype="half",  # T4/P100 have no bf16
    max_model_len=4096,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.92,
)
GREEDY = SamplingParams(temperature=0, max_tokens=64)
VOTE = SamplingParams(temperature=0.7, max_tokens=64, n=N_VOTES)


def chat(prompts, params):
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
    return outs  # caller reads o.outputs[k].text


def verdict(text, default=0.5):
    lines = [l for l in text.strip().upper().splitlines() if l.strip()]
    tail = lines[-1] if lines else ""
    if "YES" in tail:
        return 1.0
    if "NO" in tail:
        return 0.0
    return default


# ---------------------------------------------------------------- prompts
FEWSHOT = (
    "Example — Q: '৫২-এর ভাষা আন্দোলন কোন সালে হয়?' A: '১৯৫২ সালে।' -> YES\n"
    "Example — Q: 'বিদ্রোহী কবিতার রচয়িতা কে?' A: 'রবীন্দ্রনাথ ঠাকুর।' -> NO\n"
)


def grounding_prompt(row):
    ctx = str(row["context"])[:MAX_CTX_CHARS]
    return (
        "Verify a Bengali answer against a source passage.\n\n"
        f"Passage:\n{ctx}\n\nQuestion: {row['prompt_bn']}\nAnswer: {row['response_bn']}\n\n"
        "Is the answer correct AND supported by the passage? "
        "Think in one short line, then reply on a new line with exactly YES or NO."
    )


def factuality_prompt(row):
    return (
        "You are a careful Bengali fact-checker (Bengali literature, Bangladeshi "
        "history/culture, science, math).\n\n" + FEWSHOT +
        f"\nQuestion: {row['prompt_bn']}\nAnswer: {row['response_bn']}\n\n"
        "Is the answer factually correct? "
        "Think in one short line, then reply on a new line with exactly YES or NO."
    )


# ---------------------------------------------------------------- signal
def judge(rows, tag):
    """Per-row P(faithful): greedy grounding (context) + self-consistency (no-ctx)."""
    vals = np.full(len(rows), 0.5)
    ctx = [i for i, r in enumerate(rows) if has_context(r)]
    noc = [i for i, r in enumerate(rows) if not has_context(r)]

    print(f"[{tag}] grounding {len(ctx)} context rows...")
    for i, o in zip(ctx, chat([grounding_prompt(rows[i]) for i in ctx], GREEDY)):
        vals[i] = verdict(o.outputs[0].text)

    print(f"[{tag}] self-consistency ({N_VOTES}x) on {len(noc)} no-context rows...")
    for i, o in zip(noc, chat([factuality_prompt(rows[i]) for i in noc], VOTE)):
        votes = [verdict(c.text) for c in o.outputs]
        vals[i] = float(np.mean([v for v in votes if v in (0.0, 1.0)] or [0.5]))
    return vals


def best_threshold(y, prob):
    best_t, best_m = 0.5, -1.0
    for t in np.arange(0.2, 0.81, 0.05):
        m = macro_f1(y, [1 if p >= t else 0 for p in prob])
        if m > best_m:
            best_t, best_m = float(t), m
    return best_t


# ---------------------------------------------------------------- run
s_vals = judge(samples, "samples")
y = [s["label"] for s in samples]
ctx_i = [i for i, s in enumerate(samples) if has_context(s)]
noc_i = [i for i, s in enumerate(samples) if not has_context(s)]

# per-branch thresholds tuned on samples (ensemble.py later does this via nested CV)
t_ctx = best_threshold([y[i] for i in ctx_i], [s_vals[i] for i in ctx_i])
t_noc = best_threshold([y[i] for i in noc_i], [s_vals[i] for i in noc_i])
s_pred = [1 if s_vals[i] >= (t_ctx if i in set(ctx_i) else t_noc) else 0
          for i in range(len(samples))]
print(f"\nsample macroF1={macro_f1(y, s_pred):.4f}  F1(hall)={f1_cls(y, s_pred, 0):.4f}  "
      f"(thr ctx={t_ctx:.2f} noctx={t_noc:.2f})")
print(f"  context   macroF1={macro_f1([y[i] for i in ctx_i], [s_pred[i] for i in ctx_i]):.4f}")
print(f"  no-context macroF1={macro_f1([y[i] for i in noc_i], [s_pred[i] for i in noc_i]):.4f}")

t_vals = judge(test_rows, "test")
t_ctx_mask = [has_context(r) for r in test_rows]
labels = [1 if t_vals[i] >= (t_ctx if t_ctx_mask[i] else t_noc) else 0
          for i in range(len(test_rows))]

with open("submission_judge.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label"])
    for r, lab in zip(test_rows, labels):
        w.writerow([r["id"], lab])

with open("signal_judge.json", "w") as f:
    json.dump({"name": "judge",
               "samples": [float(v) for v in s_vals],
               "test": {r["id"]: float(v) for r, v in zip(test_rows, t_vals)}}, f)

n0 = labels.count(0)
print(f"\nwrote submission_judge.csv ({n0} hallucinated / {len(labels) - n0} faithful) "
      f"+ signal_judge.json")
