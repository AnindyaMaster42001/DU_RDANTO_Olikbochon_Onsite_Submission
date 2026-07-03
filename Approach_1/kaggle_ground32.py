# Approach_1 — retrieval grounding v2: 32B judges answers against evidence.
#
# Consumes retrieved_evidence.json from the bengali-wiki-retrieve kernel
# (attach it via kernel_sources). Three-way verdict with logprob softening:
#   YES    evidence supports the candidate answer
#   NO     evidence contradicts it / shows a different answer
#   UNSURE evidence is irrelevant or insufficient  ->  0.5, never a wrong NO
# The sanity check showed ~40% of evidence sets are irrelevant (e.g. word-
# meaning questions vs an encyclopedia corpus), so the UNSURE lane is load-
# bearing. Writes signal_ret32.json. ~1322 calls, well under an hour.

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import csv
import json
import math
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
EV_CHARS = 700  # per passage
KAGGLE_INPUT = Path("/kaggle/input")


def find(part):
    for root in (KAGGLE_INPUT, Path(".")):
        if root.exists():
            hits = [p for p in root.rglob("*") if p.is_file() and part in p.name.lower()]
            if hits:
                return hits[0]
    raise FileNotFoundError(part)


samples = json.load(open(find("samples.json"), encoding="utf-8"))
test_rows = list(csv.DictReader(open(find("test set"), encoding="utf-8")))
evidence = json.load(open(find("retrieved_evidence"), encoding="utf-8"))
print(f"samples: {len(samples)}  test: {len(test_rows)}  evidence: {len(evidence)}")


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


def ground_prompt(row, ev):
    q, a = str(row["prompt_bn"]), str(row["response_bn"])
    blocks = "\n\n".join(
        f"[Evidence {k + 1} — {p['title']}]\n{p['text'][:EV_CHARS]}"
        for k, p in enumerate(ev[:5])
    )
    return (
        "You are verifying an answer to a Bengali question using retrieved "
        "encyclopedia passages. The passages may or may not be relevant.\n\n"
        f"{blocks}\n\n"
        f"Question: {q}\nCandidate answer: {a}\n\n"
        "Based ONLY on the evidence above: reply YES if the evidence supports "
        "the candidate answer, NO if the evidence contradicts it or shows the "
        "correct answer is different, or UNSURE if the evidence is irrelevant "
        "or insufficient to decide. Never answer NO merely because the "
        "evidence does not mention the answer. Reply with exactly one word."
    )


llm = LLM(
    model=MODEL,
    dtype="half",
    max_model_len=4096,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.92,
)
VERDICT = SamplingParams(temperature=0, max_tokens=1, logprobs=20)


def soft_verdicts(prompts):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, VERDICT)
    vals = []
    for o in outs:
        p = {"Y": 0.0, "N": 0.0, "U": 0.0}
        lps = o.outputs[0].logprobs
        for tok in (lps[0].values() if lps else []):
            t = (tok.decoded_token or "").strip().upper()
            if t and t[0] in p:
                p[t[0]] += math.exp(tok.logprob)
        tot = sum(p.values())
        vals.append(0.5 if tot < 0.05 else (p["Y"] + 0.5 * p["U"]) / tot)
    return vals


def run_split(rows, keyfn, tag):
    n = len(rows)
    vals = [0.5] * n
    idx = [i for i, r in enumerate(rows) if not has_context(r) and evidence.get(keyfn(i, r))]
    print(f"[{tag}] grounding {len(idx)} rows...")
    outs = soft_verdicts([ground_prompt(rows[i], evidence[keyfn(i, rows[i])]) for i in idx])
    for i, v in zip(idx, outs):
        vals[i] = v
    return vals


s_vals = run_split(samples, lambda i, r: f"s{i}", "samples")
t_vals = run_split(test_rows, lambda i, r: f"t{r['id']}", "test")

with open("signal_ret32.json", "w") as f:
    json.dump(
        {
            "name": "ret32",
            "samples": s_vals,
            "test": {r["id"]: v for r, v in zip(test_rows, t_vals)},
        },
        f,
    )
print("wrote signal_ret32.json")
