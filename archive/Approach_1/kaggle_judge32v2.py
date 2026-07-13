# Approach_1 — judge32 v2: anti-skepticism 32B judging (multi-signal).
#
# Error analysis on the samples showed ALL 23 unanimously-missed no-context
# rows are FAITHFUL: every judge calls true-but-obscure facts hallucinated.
# v2 counters that three ways, all with logprob-soft outputs:
#   signal_j32lp   three-way YES/NO/UNSURE verdict; UNSURE maps to 0.5 rather
#                  than a wrong NO. All rows (grounding or factuality prompt).
#   signal_j32fs   few-shot factuality (exemplars include true-but-obscure
#                  Bangladesh-specific answers). No-context rows.
#   signal_j32sv   self-verify: model answers first, then a logprob compare.
#                  No-context rows.
# Value = (P(YES) + 0.5 * P(UNSURE)) / P(parsed); 0.5 when unparseable.
#
# Kaggle T4 x2, ~9.6k calls, mostly max_tokens=1 -> roughly 1-1.5 h.

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import csv
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
MAX_CTX_CHARS = 3500

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
print(f"samples: {len(samples)}  test rows: {len(test_rows)}")


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


llm = LLM(
    model=MODEL,
    dtype="half",
    max_model_len=4096,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.92,
)
VERDICT = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
ANSWER = SamplingParams(temperature=0, max_tokens=64)


def soft_verdicts(prompts):
    """Per prompt: (P(YES)+0.5*P(UNSURE))/P(parsed) from first-token logprobs.
    YES/NO/UNSURE have distinct initials, so match on the first letter."""
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, VERDICT)
    vals = []
    for o in outs:
        import math
        p = {"Y": 0.0, "N": 0.0, "U": 0.0}
        lps = o.outputs[0].logprobs
        for tok in (lps[0].values() if lps else []):
            t = (tok.decoded_token or "").strip().upper()
            if t and t[0] in p:
                p[t[0]] += math.exp(tok.logprob)
        tot = sum(p.values())
        vals.append(0.5 if tot < 0.05 else (p["Y"] + 0.5 * p["U"]) / tot)
    return vals


def answers(prompts):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    return [o.outputs[0].text.strip() for o in llm.chat(msgs, ANSWER)]


# ---------------------------------------------------------------- prompts
THREE_WAY = (
    "Reply with exactly one word: YES if the answer is correct, NO if it is "
    "incorrect, or UNSURE if you genuinely cannot tell. Do not guess NO for "
    "facts you simply do not know — obscure but true local facts deserve YES "
    "or UNSURE, not NO."
)


def lp_prompt(row):
    q, a = str(row["prompt_bn"]), str(row["response_bn"])
    if has_context(row):
        ctx = str(row["context"])[:MAX_CTX_CHARS]
        return (
            "You are verifying answers to Bengali questions against a source passage.\n\n"
            f"Passage:\n{ctx}\n\nQuestion: {q}\nCandidate answer: {a}\n\n"
            "Is the candidate answer correct AND supported by the passage? " + THREE_WAY
        )
    return (
        "You are a careful fact-checker for Bengali question answering, "
        "including Bangladesh-specific history, culture, literature, grammar, "
        "and mathematics.\n\n"
        f"Question: {q}\nCandidate answer: {a}\n\n"
        "Is the candidate answer factually correct? " + THREE_WAY
    )


FEWSHOT = """Examples of how to judge:

Question: বাংলাদেশের জাতীয় ফুল কী?
Candidate answer: শাপলা
Verdict: YES (well-known true fact)

Question: 'অগ্নিবীণা' কাব্যগ্রন্থের রচয়িতা কে?
Candidate answer: রবীন্দ্রনাথ ঠাকুর
Verdict: NO (it was কাজী নজরুল ইসলাম — a confidently wrong attribution)

Question: ভাষা আন্দোলনের প্রথম শহিদ মিনার কোথায় নির্মিত হয়?
Candidate answer: ঢাকা মেডিকেল কলেজ প্রাঙ্গণে
Verdict: YES (obscure but true local fact — do not reject facts for being little-known)

Question: বাংলা ব্যাকরণে 'সন্ধি' কত প্রকার?
Candidate answer: পাঁচ প্রকার
Verdict: NO (স্বরসন্ধি, ব্যঞ্জনসন্ধি, বিসর্গসন্ধি — তিন প্রকার)

Now judge this one:
"""


def fs_prompt(row):
    return (
        FEWSHOT
        + f"Question: {row['prompt_bn']}\nCandidate answer: {row['response_bn']}\n\n"
        + "Is the candidate answer factually correct? " + THREE_WAY
    )


def sv_answer_prompt(row):
    return (
        "Answer the following Bengali question concisely and accurately in "
        "Bengali. Give only the answer, no explanation.\n\n"
        f"Question: {row['prompt_bn']}"
    )


def sv_compare_prompt(row, own):
    return (
        "Two answers to the same Bengali question are given.\n\n"
        f"Question: {row['prompt_bn']}\n"
        f"Answer A: {row['response_bn']}\n"
        f"Answer B: {own}\n\n"
        "Do Answer A and Answer B convey the same essential answer? "
        "Reply with exactly one word: YES, NO, or UNSURE (if Answer B does not "
        "actually answer the question)."
    )


# ---------------------------------------------------------------- run
def run_split(rows, tag):
    n = len(rows)
    noctx = [i for i, r in enumerate(rows) if not has_context(r)]
    sig = {"j32lp": [0.5] * n, "j32fs": [0.5] * n, "j32sv": [0.5] * n}

    print(f"[{tag}] three-way logprob judge ({n})...")
    for i, v in enumerate(soft_verdicts([lp_prompt(r) for r in rows])):
        sig["j32lp"][i] = v

    print(f"[{tag}] few-shot factuality ({len(noctx)})...")
    for i, v in zip(noctx, soft_verdicts([fs_prompt(rows[i]) for i in noctx])):
        sig["j32fs"][i] = v

    print(f"[{tag}] self-verify ({len(noctx)} x2)...")
    own = answers([sv_answer_prompt(rows[i]) for i in noctx])
    comps = soft_verdicts(
        [sv_compare_prompt(rows[i], a) for i, a in zip(noctx, own)]
    )
    for i, v in zip(noctx, comps):
        sig["j32sv"][i] = v
    return sig


s_sig = run_split(samples, "samples")
t_sig = run_split(test_rows, "test")

for name in ("j32lp", "j32fs", "j32sv"):
    with open(f"signal_{name}.json", "w") as f:
        json.dump(
            {
                "name": name,
                "samples": s_sig[name],
                "test": {r["id"]: v for r, v in zip(test_rows, t_sig[name])},
            },
            f,
        )
    print(f"wrote signal_{name}.json")
