"""Approach_1 · cross-lingual consistency signal (no-context branch, Kaggle).

The competition's headline failure is "correct in English, wrong in Bengali."
So for each no-context (closed-book) row:
  1. Ask the model to answer the question in ENGLISH (its knowledge is stronger
     there). If it genuinely doesn't know -> UNKNOWN -> abstain (0.5).
  2. Self-consistency: does the Bengali response agree with that English answer?
     YES-fraction = P(faithful).

Emits signal_crosslingual.json for ensemble.py — no-context rows only (context
rows stay 0.5, handled by substring/judge). Measured on 299 samples: no-context
macro-F1 ~0.62 alone, and it is COMPLEMENTARY to the judge — stacking both lifts
the no-context branch (0.65 -> 0.67 OOF). NOTE: ~44% of no-context questions
abstain (model is UNKNOWN in English too) -> that fraction is retrieval's job.

Self-contained: paste into ONE cell of a Kaggle GPU notebook (T4 x2 / P100)
attached to the competition. ~1-2h.
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import csv
import json
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
N_VOTES = 5

KAGGLE = Path("/kaggle/input")
DATA = next((p for p in [KAGGLE / "bengali-hallucination", *KAGGLE.glob("*")]
             if p.is_dir() and any("test" in f.name.lower() for f in p.iterdir())),
            Path("."))
llm = None


def find(part):
    hits = [p for p in DATA.iterdir() if part in p.name.lower()]
    assert hits, f"no file matching {part!r} in {DATA}"
    return hits[0]


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


def f1_cls(y, p, c):
    tp = sum(1 for t, q in zip(y, p) if t == c and q == c)
    fp = sum(1 for t, q in zip(y, p) if t != c and q == c)
    fn = sum(1 for t, q in zip(y, p) if t == c and q != c)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def macro_f1(y, p):
    return (f1_cls(y, p, 0) + f1_cls(y, p, 1)) / 2


def chat(prompts, params):
    return llm.chat([[{"role": "user", "content": p}] for p in prompts], params)


def verdict(text, default=0.5):
    lines = [l for l in text.strip().upper().splitlines() if l.strip()]
    tail = lines[-1] if lines else ""
    if "YES" in tail:
        return 1.0
    if "NO" in tail:
        return 0.0
    return default


def english_answer_prompt(row):
    return (
        "Answer this question accurately and concisely in English. "
        "If you genuinely do not know the answer, reply with exactly UNKNOWN.\n\n"
        f"Question (in Bengali): {row['prompt_bn']}\nEnglish answer:"
    )


def agreement_prompt(row, en):
    return (
        "A Bengali question has two candidate answers.\n\n"
        f"Question: {row['prompt_bn']}\n"
        f"Answer A (Bengali): {row['response_bn']}\n"
        f"Answer B (English, from a reference model): {en}\n\n"
        "Do Answer A and Answer B state the same fact (are they consistent)? "
        "Think in one short line, then reply on a new line with exactly YES or NO."
    )


ANSWER = SamplingParams(temperature=0, max_tokens=64)
VOTE = SamplingParams(temperature=0.7, max_tokens=64, n=N_VOTES)


def crosslingual(rows, tag):
    vals = np.full(len(rows), 0.5)
    noc = [i for i, r in enumerate(rows) if not has_context(r)]
    print(f"[{tag}] english answers for {len(noc)} no-context rows...")
    en = {}
    for i, o in zip(noc, chat([english_answer_prompt(rows[i]) for i in noc], ANSWER)):
        en[i] = o.outputs[0].text.strip()
    known = [i for i in noc if en[i] and "UNKNOWN" not in en[i].upper()]
    print(f"[{tag}] agreement ({N_VOTES}x) on {len(known)} rows "
          f"({len(noc) - len(known)} abstain: UNKNOWN in English)...")
    for i, o in zip(known, chat([agreement_prompt(rows[i], en[i]) for i in known], VOTE)):
        votes = [v for v in (verdict(c.text) for c in o.outputs) if v in (0.0, 1.0)]
        vals[i] = float(np.mean(votes)) if votes else 0.5
    return vals


def best_threshold(y, prob):
    best_t, best_m = 0.5, -1.0
    for t in np.arange(0.2, 0.81, 0.05):
        m = macro_f1(y, [1 if p >= t else 0 for p in prob])
        if m > best_m:
            best_t, best_m = float(t), m
    return best_t


def main():
    global llm
    samples = json.load(open(find("samples"), encoding="utf-8"))
    test_rows = list(csv.DictReader(open(find("test"), encoding="utf-8")))
    print(f"samples {len(samples)}  test {len(test_rows)}")

    llm = LLM(model=MODEL, dtype="half", max_model_len=4096,
              tensor_parallel_size=torch.cuda.device_count(),
              gpu_memory_utilization=0.92)

    s_vals = crosslingual(samples, "samples")
    y = [s["label"] for s in samples]
    noc_i = [i for i, s in enumerate(samples) if not has_context(s)]
    thr = best_threshold([y[i] for i in noc_i], [s_vals[i] for i in noc_i])
    pred = [1 if s_vals[i] >= thr else 0 for i in noc_i]
    print(f"\ncross-lingual no-context macroF1="
          f"{macro_f1([y[i] for i in noc_i], pred):.4f} (thr {thr:.2f})")

    t_vals = crosslingual(test_rows, "test")
    with open("signal_crosslingual.json", "w") as f:
        json.dump({"name": "crosslingual", "samples": [float(v) for v in s_vals],
                   "test": {r["id"]: float(v) for r, v in zip(test_rows, t_vals)}}, f)
    print("wrote signal_crosslingual.json")


if __name__ == "__main__":
    main()
