# Approach_1 — 8th signal: Qwen2.5-32B judge (single greedy pass).
#
# Runs on Kaggle T4 x2 (vLLM, GPTQ-Int4, TP=2). Deliberately minimal: one
# YES/NO judgment per row (grounding if context, factuality if not) over the
# 299 samples + 2516 test rows, written directly in ensemble.py's signal
# format as signal_judge32.json. ~30 min end to end.

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import csv
import json
import re
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
MAX_CTX_CHARS = 3500

KAGGLE_INPUT = Path("/kaggle/input")


def find(part):
    for root in (KAGGLE_INPUT, Path(".")):
        if root.exists():
            hits = [
                p for p in root.rglob("*") if p.is_file() and part in p.name.lower()
            ]
            if hits:
                return hits[0]
    raise FileNotFoundError(part)


samples = json.load(open(find("samples.json"), encoding="utf-8"))
test_rows = list(csv.DictReader(open(find("test set"), encoding="utf-8")))
print(f"samples: {len(samples)}  test rows: {len(test_rows)}")


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


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


llm = LLM(
    model=MODEL,
    dtype="half",
    max_model_len=4096,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.92,
)
VERDICT = SamplingParams(temperature=0, max_tokens=5)


def judge(rows, tag):
    msgs = [[{"role": "user", "content": judge_prompt(r)}] for r in rows]
    outs = llm.chat(msgs, VERDICT)
    vals = []
    for o in outs:
        up = o.outputs[0].text.strip().upper()
        vals.append(1.0 if "YES" in up else 0.0 if "NO" in up else 0.5)
    print(f"[{tag}] {len(vals)} judged, mean P(faithful) {sum(vals)/len(vals):.3f}")
    return vals


s_vals = judge(samples, "samples")
t_vals = judge(test_rows, "test")

with open("signal_judge32.json", "w") as f:
    json.dump(
        {
            "name": "judge32",
            "samples": s_vals,
            "test": {r["id"]: v for r, v in zip(test_rows, t_vals)},
        },
        f,
    )
print("wrote signal_judge32.json")
