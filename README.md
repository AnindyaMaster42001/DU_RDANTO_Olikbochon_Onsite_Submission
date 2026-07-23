# অলীকবচন — Bengali LLM Hallucination Detection

Team **DU_RDANTO** · [Bengali LLM Hallucination Detection Challenge](https://www.kaggle.com/competitions/bengali-hallucination)
(Datathon 2.0 · Institute of Policy Dynamics · IUTCS).

**Task:** given a Bengali `prompt_bn`, a `response_bn`, and an optional `context`
(`[NULL]` if none), predict whether the response is **faithful** (`label = 1`) or
**hallucinated** (`label = 0`). Public metric: macro-F1; the real metric is binary F1 on the
hallucinated class, weighted toward the Bangladesh-specific (C1) band.

**Best public LB: 0.904.** The two selected finals are `submission_final_bcs.csv` (0.904)
and `submission_final.csv` (0.901).

## Onsite submission (Phase 3)

The onsite-round deliverables live at the repository root, following the required naming
convention (`TeamName_...`, team = **DU_RDANTO**):

| File | Contents |
|---|---|
| `DU_RDANTO_report.pdf` | Technical report, ≤4 pages excl. references (same as `Phase_2/paper/main.pdf`) |
| `DU_RDANTO_presentation.pdf` | Onsite presentation slides *(added by the team)* |
| `DU_RDANTO_inference_notebook.ipynb` | The final inference notebook submitted in the previous phase (same as `Phase_2/phase2_gold_pipeline.ipynb`) |

No training notebook is included: the system uses only off-the-shelf open-weight models plus
deterministic gold-answer verification, so there is no fine-tuning/training pipeline (this
deliverable is optional).

## Layout

```
Phase_2/                     ← the Phase-2 submission package (start here)
  README.md                  system, models, corpora, assembly, verification, checklist
  phase2_gold_pipeline.py/.ipynb   single offline notebook: raw test CSV → submission.csv
  gold_verify.py, bnnum.py   Layer 1: gold-answer retrieval + equivalence (CPU)
  submission_final_bcs.csv   primary final  (LB 0.904)
  submission_final.csv       insurance final (LB 0.901)
  selfcheck.py               corpora-free proof the pipeline reproduces both finals
  snapshots/                 Kaggle kernels that pin every external artifact + corpora builder
  paper/                     4-page paper report (main.tex → main.pdf)

archive/                     ← preserved research lineage (not part of the submission)
  README.md                  index of the path from baseline to 0.904
  Approach_0/                baseline + data + first judge
  Approach_1/                multi-signal ensemble, contrastive lever, dead-lever ledger
  Approach_2/                gold-answer verification (the breakthrough behind Phase_2)
  PLAN.md, TEAM_FINDINGS.md  strategy notes / handoff briefs
```

## The winning idea

Every row splits into two sub-problems: **context present** → *grounding* (is the response
supported by the passage?), **no context** → *closed-book factuality* (is it true?). The
final system adds a decisive third move: since the benchmark is built from public Bengali
corpora, the **published gold answer** to ~60% of rows already exists — retrieve it and check
candidate/gold equivalence. That deterministic, CPU-only layer carries the 0.831 → 0.904 gain
and is wrapped in the model-based stack so the package degrades gracefully on an
out-of-distribution held-out fold.

See [`Phase_2/README.md`](Phase_2/README.md) to build and run the package, and
[`archive/README.md`](archive/README.md) for the full research history.

## Quick verify (no GPU)

```bash
cd Phase_2 && python selfcheck.py     # proves the overlay reproduces both finals
```
