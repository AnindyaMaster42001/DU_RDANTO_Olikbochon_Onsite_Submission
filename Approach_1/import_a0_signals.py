"""Convert Approach_0 signals into Approach_1 signal files + regenerate substring.

Approach_0's Kaggle judge (kaggle_judge.py) saves raw per-row verdicts as
  signals_samples.json / signals_test.json: {"heur": [...], "judge": [...],
  "selfv": [...]} with 0/1 or null (signal not applicable to that row).
This script maps them into ensemble.py's signal format (null -> 0.5 neutral):

  signal_a0judge.json   Approach_0 judge (greedy Qwen-14B-GPTQ, different
                        prompts + quantization than Approach_1's judge)
  signal_a0selfv.json   Approach_0 self-verify (answer-then-compare, no-context)
  signal_substring.json the substring grounding rule (was never committed)

Usage:  python3 import_a0_signals.py
"""

import json
from pathlib import Path

import common as C

A0_RESULTS = Path(__file__).resolve().parent.parent / "Approach_0" / "results"


def to_prob(values):
    return [0.5 if v is None else float(v) for v in values]


def main():
    samples, test = C.load_samples(), C.load_test()
    raw_s = json.load(open(A0_RESULTS / "signals_samples.json"))
    raw_t = json.load(open(A0_RESULTS / "signals_test.json"))
    assert len(raw_s["judge"]) == len(samples)
    assert len(raw_t["judge"]) == len(test)

    for a0_name, sig_name in [("judge", "a0judge"), ("selfv", "a0selfv")]:
        s_vals = to_prob(raw_s[a0_name])
        t_vals = to_prob(raw_t[a0_name])
        C.save_signal(
            sig_name, s_vals, {r["id"]: v for r, v in zip(test, t_vals)}
        )

    def substring(rows):
        return [
            (1.0 if C.norm(r["response_bn"]) and
             C.norm(r["response_bn"]) in C.norm(r["context"]) else 0.0)
            if C.has_context(r) else 0.5
            for r in rows
        ]

    s_vals, t_vals = substring(samples), substring(test)
    C.save_signal(
        "substring", s_vals, {r["id"]: v for r, v in zip(test, t_vals)}
    )


if __name__ == "__main__":
    main()
