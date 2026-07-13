"""Phase C (fixed) — retrieval grounding with token-safe prompt truncation.

Same as ground.py but fits every prompt under the model's token limit (Bengali
tokenizes densely, so top-6 evidence can overflow 4096 -> vLLM hard-errors).
Emits signal_retrieval.json.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams

MODEL = "/home/mb2/Qwen2.5-14B-AWQ"
N_VOTES = 5
GPU_MEM_UTIL = 0.18
MIN_FREE_GIB = 19
TOKEN_BUDGET = 3800    # leave room for generation within 4096
llm = None
tokenizer = None


def find(part, ext=None):
    for f in Path(".").iterdir():
        if part in f.name.lower() and (ext is None or f.suffix == ext):
            return f
    raise FileNotFoundError(part)


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


def prompt_text(q, a, ev):
    return (
        "Verify a Bengali answer against retrieved encyclopedia passages.\n\n"
        f"Passages:\n{ev}\n\nQuestion: {q}\nAnswer: {a}\n\n"
        "Is the answer correct AND supported by the passages? If the passages do "
        "not contain enough information, judge whether the answer is factually "
        "correct. Think in one short line, then reply on a new line with exactly YES or NO."
    )


def n_tokens(p):
    return len(tokenizer(p, add_special_tokens=False)["input_ids"])


def fit_prompt(row, evidence):
    q = str(row["prompt_bn"])
    a = str(row["response_bn"])[:1200]
    ev = "\n\n".join(evidence) if evidence else "(no relevant passage found)"
    for _ in range(8):
        p = prompt_text(q, a, ev)
        if n_tokens(p) <= TOKEN_BUDGET:
            return p
        if len(ev) > 300:
            ev = ev[:int(len(ev) * 0.7)]
        else:
            q = q[:800]  # last resort for a pathologically long question
    return prompt_text(q, a, ev[:300])


VOTE = SamplingParams(temperature=0.7, max_tokens=64, n=N_VOTES)


def ground(rows, split, evidence):
    vals = np.full(len(rows), 0.5)
    noc = [i for i, r in enumerate(rows) if not has_context(r)]
    keys = {i: f"{split}:{rows[i].get('id', i)}" for i in noc}
    print(f"[{split}] grounding {len(noc)} no-context rows vs retrieved evidence...", flush=True)
    prompts = [fit_prompt(rows[i], evidence.get(keys[i], [])) for i in noc]
    for i, o in zip(noc, chat(prompts, VOTE)):
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
    global llm, tokenizer
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    print(f"GPU free {free_gib:.1f} GiB", flush=True)
    if free_gib < MIN_FREE_GIB:
        raise SystemExit(f"ABORT: only {free_gib:.1f} GiB free.")

    evidence = json.load(open("retrieved_evidence.json", encoding="utf-8"))
    samples = json.load(open(find("samples", ".json"), encoding="utf-8"))
    test = list(csv.DictReader(open(find("test", ".csv"), encoding="utf-8")))

    llm = LLM(model=MODEL, dtype="half", max_model_len=4096, tensor_parallel_size=1,
              gpu_memory_utilization=GPU_MEM_UTIL, enforce_eager=True)
    tokenizer = llm.get_tokenizer()

    s_vals = ground(samples, "samples", evidence)
    y = [s["label"] for s in samples]
    noc_i = [i for i, s in enumerate(samples) if not has_context(s)]
    thr = best_threshold([y[i] for i in noc_i], [s_vals[i] for i in noc_i])
    pred = [1 if s_vals[i] >= thr else 0 for i in noc_i]
    print(f"\nretrieval-grounding no-context macroF1="
          f"{macro_f1([y[i] for i in noc_i], pred):.4f} (thr {thr:.2f})", flush=True)

    t_vals = ground(test, "test", evidence)
    with open("signal_retrieval.json", "w") as f:
        json.dump({"name": "retrieval", "samples": [float(v) for v in s_vals],
                   "test": {r["id"]: float(v) for r, v in zip(test, t_vals)}}, f)
    print("wrote signal_retrieval.json")


if __name__ == "__main__":
    main()
