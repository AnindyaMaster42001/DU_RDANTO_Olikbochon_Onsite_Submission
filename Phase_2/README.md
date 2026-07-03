# Phase 2 — Solution Package (team DU_RDANTO)

Offline, single-notebook reproduction of our Phase-1 system for the
[Bengali LLM Hallucination Detection Challenge](https://www.kaggle.com/competitions/bengali-hallucination).
Runs end-to-end on the raw test set (or the organizers' held-out fold) inside a
standard Kaggle code-competition kernel: **no internet, open-weight models
only, T4 x2, well under 9 h, ~37 GB of weights (< 50 GB).**

## System summary

Every row is split into two sub-problems. **Context present** → grounding:
substring rule + mDeBERTa NLI + two LLM judges. **No context** → closed-book
factuality: two LLM judges + 32B self-verification + **evidence-grounded
verification against a Bengali-Wikipedia index** (dense bge-m3 retrieval,
bge-reranker-v2-m3, then Qwen-32B judges the answer against the evidence with
UNSURE-safe three-way verdicts). Seven per-row signals are stacked by a
per-branch logistic regression refit on the 299 labeled samples at run time;
decision thresholds come from a 5-seed x 5-fold CV median.

Honest 5-seed out-of-fold macro-F1 on the labeled samples: **0.8311**
(context 0.9037 / no-context 0.7683). Public LB of the equivalent
configuration: **0.803**.

## Contents

| file | role |
|---|---|
| `paper/paper.pdf` | 4-page ACL-format paper report (compile: `latexmk -xelatex paper.tex`) |
| `phase2_pipeline.py` | the entire pipeline; cell 1 of the submission notebook |
| `phase2_pipeline.ipynb` | the same, as an importable notebook |
| `snapshots/snap_qwen14b.py` | pins Qwen2.5-14B-Instruct-GPTQ-Int4 (CPU kernel) |
| `snapshots/snap_qwen32b.py` | pins Qwen2.5-32B-Instruct-GPTQ-Int4 (CPU kernel) |
| `snapshots/snap_encoders.py` | pins bge-m3, bge-reranker-v2-m3, mDeBERTa-xnli (CPU kernel) |
| `snapshots/snap_vllm_wheels.py` | pins the vLLM wheel set for offline install (CPU kernel) |
| `snapshots/wiki_index_v2.py` | builds the bnwiki index: passages.jsonl + embeddings (GPU kernel) |

## Models (all open-weight)

| model | role | disk |
|---|---|---|
| Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 | judge + self-verify | ~9 GB |
| Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 | judge, self-verify, evidence grounding | ~19 GB |
| BAAI/bge-m3 | dense retrieval | ~2.3 GB |
| BAAI/bge-reranker-v2-m3 | evidence reranking | ~2.3 GB |
| MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | NLI grounding | ~1.1 GB |
| bnwiki index (HF `wikimedia/wikipedia`, `.bn` config, CC BY-SA) | evidence corpus | ~2 GB |

GPTQ (not AWQ) quantization is deliberate: it runs on both P100 (sm60) and
T4 (sm75). All Bengali-heavy prompts run with `max_model_len=8192` — Bengali
is roughly 1+ token per character in Qwen's vocabulary.

## How to assemble (once)

1. Run the five `snapshots/` kernels on Kaggle (internet ON; the four `snap_*`
   are CPU-only, `wiki_index_v2` needs a GPU for ~2.5 h). Each produces a
   pinned output. Our runs: `bengali-snap-qwen14b`, `bengali-snap-qwen32b`,
   `bengali-snap-encoders`, `bengali-snap-vllm-wheels`, `bengali-wiki-index-v2`
   under `anindyakundu42001`. Optionally convert each output to a formal
   Kaggle Dataset ("New Dataset → from notebook output") — functionally
   identical as an attached source.
2. Create the submission notebook from `phase2_pipeline.ipynb` and attach as
   inputs: the competition data + the five artifacts from step 1.
3. Notebook settings: **Accelerator = GPU T4 x2, Internet = OFF.**

## How the organizers run it

Attach the held-out fold CSV (or replace the competition test file), set the
`TEST_CSV` variable at the top of the notebook if the filename does not match
`test*.csv`, and Run All. The notebook recomputes all seven signals from
scratch for both the 299 labeled samples (stacker training) and the evaluation
rows, then writes `submission.csv` (`id,label`; 0 = hallucinated,
1 = faithful). Measured runtime on T4 x2 for 2,516 rows: see verification
below; the 9 h budget holds with ample margin.

## Verification (completed 2026-07-04)

The packaged notebook was run against the Phase-1 test set with **internet
disabled** and only the pinned artifacts attached
([bengali-phase2-repro](https://www.kaggle.com/code/anindyakundu42001/bengali-phase2-repro)):

- ran to completion fully offline in **4.96 h** on T4 x2 (9 h budget; the 32B
  stage is the long pole at ~3.9 h, the 14B stage ~1 h, everything else <5 min)
- output agrees with our submitted 10-signal predictions on **97.7%** of rows
- submitted to the leaderboard directly: **public LB 0.800** (10-signal
  reference: 0.803 — identical within split noise, as the OOF comparison
  predicted: 0.8311 vs 0.8327)

## Why these seven signals (and not our full ten)

Three Phase-1 signals (a 14B self-consistency judge, a cross-lingual check,
and the v1 retrieval branch) were computed on a personal workstation. The
portable seven reproduce inside Kaggle limits and score identically within
noise (5-seed OOF 0.8311 vs 0.8327; the difference is far below the
resolution of a 299-row validation set), so the package ships the portable
set. Full ablations and the error analysis that motivated the
evidence-grounding design are in `../Approach_1/results/RESULTS.md`.

## Submission checklist (rules §3, top-30 teams)

- [x] Runnable Kaggle notebook, offline, reproduces Phase-1 predictions
      end-to-end within limits — `bengali-phase2-repro`, verified 4.96 h / LB 0.800
- [x] 4-page paper report, ACL format, single PDF — `paper/paper.pdf`
      (**TODO: replace teammate placeholder names in `paper/paper.tex`**)
- [x] README describing environment, weights, and external models — this file
- [ ] Submit via the form linked from the Discussion tab at the start of Phase 2
