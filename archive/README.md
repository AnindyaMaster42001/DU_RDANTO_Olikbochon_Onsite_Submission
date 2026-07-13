# Archive — research lineage

Preserved history of team DU_RDANTO's work on the
[Bengali LLM Hallucination Detection Challenge](https://www.kaggle.com/competitions/bengali-hallucination).
Nothing here is part of the Phase-2 submission; the submission package is [`../Phase_2/`](../Phase_2/).
This directory exists so the full path from baseline to the 0.904 finals stays reproducible
and auditable.

## Contents

| path | what it is | best LB |
|---|---|---|
| `Approach_0/` | Starter work: rule-based baseline, the shared eval harness, the first Qwen-14B judge, a fine-tuned BanglaBERT signal. Also holds the raw data (`dataset samples.json`, `test set.csv`). | 0.666 → 0.748 |
| `Approach_1/` | The multi-signal ensemble line: NLI, LLM judges (14B/32B), self-verify, cross-lingual, bnwiki retrieval, wiktionary grounding, the contrastive ground-truth lever, and a per-branch nested-CV stacker. Includes the measured-dead levers (enwiki, gemma-27B, math-solver) kept as negative results. | 0.803 → 0.831 |
| `Approach_2/` | **Gold-answer verification** — the breakthrough. Retrieve the published gold answer from public corpora and check candidate/gold equivalence. This is the research behind the shipped package; its `README.md` documents coverage, the negative results (fine-tuned verifier, context-span, math-solver), and the open questions. | 0.892 → **0.904** |
| `PLAN.md` | Strategy notes from the mid-competition push (LB 0.803 → top). |
| `TEAM_FINDINGS.md` | The 2026-07-09 handoff brief (current best 0.831 at the time). Superseded by `Approach_2/` and the Phase-2 package, kept for the reasoning trail. |

## Scores in order

`0.666` baseline → `0.748` 14B judge → `0.765` 7-signal → `0.803` +ret32/self-verify →
`0.815` +wiktionary override → `0.831` +contrastive flips → `0.892`→`0.900`→`0.901`→**`0.904`**
gold verification.

The 0.901 and 0.904 finals (`submission_final.csv`, `submission_final_bcs.csv`) are the
selected Phase-1 submissions and live with the package in `../Phase_2/`.

> Some scripts under `Approach_1/` and `Approach_2/` contain absolute workstation paths
> (`/mnt/NewVolume2/...`) from the original authors' machines. They are archival and not
> meant to run as-is; the runnable, path-independent code is in `../Phase_2/`.
