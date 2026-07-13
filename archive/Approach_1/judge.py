"""Approach_1 · Step 2 — LLM judge as a continuous, robust signal.

Upgrades Approach_0's single greedy YES/NO judge in three ways:
  1. SELF-CONSISTENCY — sample the verdict N times at temperature and use the
     YES-fraction as a calibrated P(faithful), instead of one brittle greedy call.
  2. FEW-SHOT + short CoT — a couple of labelled samples in the prompt and a
     "think one line, then answer" format lift verdict quality.
  3. EVIDENCE-AWARE — if retrieval.py produced evidence for a no-context row,
     the judge grounds against it (open-book) instead of relying on memory.

Emits signal_judge.json (P(faithful) per row) for ensemble.py — no hard
thresholding here; the ensemble owns the decision boundary.

Runs on a Kaggle GPU notebook (T4x2 / P100) with vLLM. Open-weight model =>
Phase-2 compliant. Left as a SCAFFOLD: prompts, voting, and signal I/O are
real; wire `generate()` to your vLLM instance (see kaggle_judge.py in
Approach_0 for a working vLLM setup to copy).

Usage (on Kaggle):  python3 judge.py
"""

import json
from collections import Counter
from pathlib import Path

import common as C

MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"   # or 32B-AWQ / Gemma-2-27B / TigerLLM
N_VOTES = 5                                # self-consistency samples (no-context)
TEMPERATURE = 0.7
EVIDENCE = Path("retrieved_evidence.json")  # optional, from retrieval.py

FEWSHOT = (
    "Example 1 — Question: '৫২ এর ভাষা আন্দোলন কোন সালে হয়?' "
    "Answer: '১৯৫২ সালে।' -> YES\n"
    "Example 2 — Question: 'বিদ্রোহী কবিতার রচয়িতা কে?' "
    "Answer: 'রবীন্দ্রনাথ ঠাকুর।' -> NO\n"
)


# ---------------------------------------------------------------- prompts
def grounding_prompt(row, evidence=None):
    ctx = evidence if evidence else str(row["context"])[:3500]
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


def verdict(text, default=0.5):
    tail = text.strip().upper().splitlines()[-1] if text.strip() else ""
    if "YES" in tail:
        return 1
    if "NO" in tail:
        return 0
    return default


# ---------------------------------------------------------------- backend
def generate(prompts, n=1, temperature=0.0, max_tokens=64):
    """TODO: run `prompts` through vLLM, returning n samples each.

    Copy the vLLM setup from Approach_0/kaggle_judge.py:
        llm.chat([[{"role":"user","content":p}] for p in prompts],
                 SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens))
    Return: list (len==len(prompts)) of list[str] (len==n).
    """
    raise NotImplementedError("wire generate() to vLLM (see kaggle_judge.py)")


# ---------------------------------------------------------------- signal
def judge_split(rows, evidence):
    vals = [0.5] * len(rows)
    # context (or evidence-grounded) rows: one greedy grounding verdict
    g_idx = [i for i, r in enumerate(rows)
             if C.has_context(r) or evidence.get(str(r.get("id", i)))]
    if g_idx:
        outs = generate([grounding_prompt(rows[i], evidence.get(str(rows[i].get("id", i))))
                         for i in g_idx], n=1, temperature=0.0)
        for i, o in zip(g_idx, outs):
            vals[i] = float(verdict(o[0]))
    # closed-book rows: self-consistency vote -> YES-fraction as P(faithful)
    f_idx = [i for i in range(len(rows)) if i not in set(g_idx)]
    if f_idx:
        outs = generate([factuality_prompt(rows[i]) for i in f_idx],
                        n=N_VOTES, temperature=TEMPERATURE)
        for i, samples_ in zip(f_idx, outs):
            votes = [verdict(s) for s in samples_]
            yes = sum(1 for v in votes if v == 1)
            vals[i] = yes / max(1, len(votes))
    return vals


def main():
    evidence = json.load(open(EVIDENCE, encoding="utf-8")) if EVIDENCE.exists() else {}
    samples, test = C.load_samples(), C.load_test()
    s_vals = judge_split(samples, evidence)
    t_vals = judge_split(test, evidence)
    C.evaluate_samples([1 if v >= 0.5 else 0 for v in s_vals])
    C.save_signal("judge", s_vals, {r["id"]: v for r, v in zip(test, t_vals)})


if __name__ == "__main__":
    main()
