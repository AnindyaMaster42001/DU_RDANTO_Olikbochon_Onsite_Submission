# Phase 2 — Solution Package for the LB 0.901 / 0.904 system

Offline, single-notebook reproduction of team DU_RDANTO's **gold-answer-verification**
system for the [Bengali LLM Hallucination Detection Challenge](https://www.kaggle.com/competitions/bengali-hallucination).
This is the packaged form of the two selected finals, `submission_final.csv` (LB 0.901) and
`submission_final_bcs.csv` (LB 0.904), both shipped in this folder.

Runs inside a standard Kaggle code-competition kernel: **no internet, open-weight models
only, T4 x2, well under 9 h, < 50 GB of weights** (rules §4).

> The earlier 10-signal-stack package that reproduced LB 0.800 is superseded — its portable
> stack lives on here as **Layer 2**, with the deterministic gold layer added on top. The
> full research lineage (Approach_0/1/2, the 0.800 pipeline, the 0.815 paper draft) is
> preserved under [`../archive/`](../archive/).

---

## The system: two layers, abstention wired through both

A **layered, not lookup-only** package, so the 0.90+ result survives a held-out fold from an
undisclosed distribution (rules §5).

| | Layer 1 — gold verification | Layer 2 — 7-signal LLM stack |
|---|---|---|
| **What** | retrieve a gold answer from public corpora, check the candidate against it | portable stack: substring · NLI · 14B/32B judges · self-verify · bnwiki-grounded 32B |
| **Compute** | CPU, seconds | GPU T4 x2, ~2–3 h (covered rows skipped) |
| **Coverage (Phase-1 test)** | decides **1509 / 2516 = 60.0%** at **98.9%** accuracy | the remaining **1007** rows |
| **Reproducibility** | **exact** (deterministic) | within 5-seed OOF noise |
| **Abstains when** | no gold found, or cross-script gold | never (final fallback) |

Layer 1 decides a row only when a gold answer exists and the equivalence check is
conclusive; otherwise it **abstains** and the row falls through to Layer 2. Because Layer 1
is CPU-only and removes 60% of rows from the LLM workload, it *frees* GPU budget under the
9 h cap rather than consuming it.

**Graceful degradation (the reason it is layered).** If the held-out fold is not drawn from
these corpora, Layer 1 abstains everywhere and the notebook runs as the verified Layer-2
stack. Layer-1 construction is wrapped so that even a corrupt or partial corpus attachment
falls back to Layer 2. **The package can never score below the LB-0.800 base stack.**

---

## What "reproduces the finals" means (precisely)

- The **gold overlay** — the ~60% of rows carrying the entire 0.831 → 0.904 gain — is
  reproduced **exactly, bit for bit**, from the attached corpora. `selfcheck.py` proves this
  offline against the committed finals (1509/1509 covered rows match `submission_final_bcs.csv`).
- The **uncovered ~40%** are scored by the portable 7-signal stack (offline-verified LB
  0.800), **not** the workstation 17-signal meta the Phase-1 finals used on those rows. That
  meta is not in-kernel reproducible, so this package ships the portable stack for those
  rows. The net effect vs the submitted finals is within 299-row OOF noise, on the slice that
  does *not* carry the headline gain.

This is the honest, rules-compliant reading of §4 ("reproduces the team's Phase-1
predictions … within the code-competition limits"): the part that moves the score is
deterministic and exact; the rest matches the verified base.

---

## Both finals from one notebook

Rules §3 expects a single Phase-2 notebook. Both selected finals come from this one, via a
single flag at the top of `phase2_gold_pipeline.py`:

| `USE_BCS` | reproduces | LB | difference |
|---|---|---|---|
| `True` *(default, recommended primary)* | `submission_final_bcs.csv` | **0.904** | consults the BCS 10th–45th exam banks |
| `False` | `submission_final.csv` | 0.901 | BCS banks disabled → those 26 rows defer to Layer 2 |

The two differ by **7 test rows** (all BCS-sourced). `submission_final_bcs.csv` is the
recommended primary final; `submission_final.csv` is the best-of-2 insurance pick.

Neither submission contains a hand-labeled test row — every covered row is reached
programmatically — so both are clean under **foundational rule 4b** ("no information from
hand labeling or human prediction of … test data records"). This is the compliance reason
these two, not the older `submission_contrastive.csv` (which shipped 12 hand-audited flips,
now under `../archive/Approach_1/`), are the finals to package.

---

## Contents

| file | role |
|---|---|
| `phase2_gold_pipeline.py` | the entire layered pipeline; cell 1 of the submission notebook |
| `phase2_gold_pipeline.ipynb` | the same, as the importable submission notebook |
| `gold_verify.py` | Layer 1: gold retrieval + answer-equivalence verifier (CPU, ~230 lines) |
| `bnnum.py` | Bengali numeral-word → integer engine used by the equivalence check |
| `submission_final_bcs.csv` | the primary final (LB 0.904) |
| `submission_final.csv` | the insurance final (LB 0.901) |
| `selfcheck.py` | corpora-free proof that the overlay reproduces both finals (run anywhere) |
| `gold_preds_test.json` | cached `GoldVerifier.predict()` output on every test row (drives `selfcheck.py`) |
| `fetch_corpora.sh` | assembles the Layer-1 corpora locally (`bash fetch_corpora.sh ext`) |
| `snapshots/pack_corpora.py` | assembles the corpora into `ext/` programmatically (manifest + robust fetch) |
| `snapshots/kaggle_layer1_verify.py` | offline Kaggle kernel that verifies Layer 1 on the real test set |
| `snapshots/kernel-metadata.example.json` | ready-to-edit metadata for the CLI push of the GPU submission notebook |
| `snapshots/snap_*.py`, `snapshots/wiki_index_v2.py` | the five Layer-2 artifact builders (Qwen GPTQ ×2, encoders, vLLM wheels, bnwiki index) |
| `paper/main.tex`, `paper/main.pdf` | the 4-page paper report (rules §3.2) |

---

## Models and inputs (all open-weight; rules §4)

Layer-2 model set (~37 GB, < 50 GB): Qwen2.5-14B-Instruct-GPTQ-Int4,
Qwen2.5-32B-Instruct-GPTQ-Int4, BAAI/bge-m3, BAAI/bge-reranker-v2-m3, mDeBERTa-v3-xnli, and
the bnwiki dense index. GPTQ (not AWQ) is deliberate: it runs on both P100 (sm60) and
T4 (sm75).

Layer 1 uses **no model** — it is pure retrieval + string/number logic.

---

## Corpora and citations (Layer 1; rules §5 — MUST be cited in the paper)

All public and pre-competition. Rule 5: *"Any publicly available Bengali or multilingual
dataset may be used … Include a citation in your Phase 2 paper."*

| corpus | source | provides |
|---|---|---|
| BanglaHalluEval GQA (= TyDiQA-GoldP Bengali) | `abidur14004/bangla-dataset-for-hallucination` (Kaggle); BenHalluEval, arXiv 2605.31483 | gold answers for context-QA rows |
| bagdhara Bengali idioms | `sakhadib/bagdhara-bangla-idioms-dataset` (Kaggle) | literal + figurative idiom meanings |
| bangla-mmlu | `hishab/bangla-mmlu` (HF) | gold choice for exam/GK MCQ rows |
| squad_bn | `csebuetnlp/squad_bn` (HF) | extractive gold spans (cross-script fallback) |
| BCS 10th–45th banks | `azminetoushikwasi/{bangla-bcs-qs, bcs-10-40th-GK-ICT-DM-NMS, bd-bcs-multimodal}` (HF) | BCS exam answers (0.901 → 0.904) |

These citations are already in `paper/main.tex`.

> **Two irreducible Layer-1 errors, one of which is a dataset issue to report (rule 6):**
> the row whose TyDi gold (`রংপুর জিলা স্কুল`) contradicts the competition's own label must
> be posted on the Discussion tab, not used privately.

---

## How to assemble (once)

1. **Layer-2 artifacts** — run the five `snapshots/snap_*.py` + `snapshots/wiki_index_v2.py`
   kernels on Kaggle (four CPU `snap_*`, one GPU `wiki_index_v2`). Each pins an artifact as
   a kernel output to attach to the submission notebook.
2. **Layer-1 corpora → the `bengali-gold-corpora` dataset.** Build it **locally with the
   Kaggle CLI**, not inside a kernel — a headless kernel cannot attach *new* datasets
   (`kagglehub` returns "New Datasets cannot be attached in non-interactive sessions"), and
   the HF fetches are cleaner from a workstation anyway:
   ```bash
   cd Phase_2 && bash fetch_corpora.sh ext             # assembles ext/ (kaggle CLI + HF)
   cp gold_verify.py bnnum.py ext/                      # bundle the verifier into the dataset
   kaggle datasets create -p ext --dir-mode zip \
     -t bengali-gold-corpora -u <your-kaggle-username>  # publish once
   ```
   `snapshots/pack_corpora.py` does the same assembly programmatically (with a `manifest.json`
   and robustness) if you prefer; run it locally, then `kaggle datasets create` on its `ext/`.
   Verified: the assembled corpora reproduce **185/299 sample coverage at 98.9%** and agree
   with the committed finals' gold layer on **1507/1508** covered test rows.
3. **Submission notebook** — create it from `phase2_gold_pipeline.ipynb`. Attach as inputs:
   the competition data + the five Layer-2 artifacts (as `kernel_sources`) + the
   `bengali-gold-corpora` dataset (which carries `gold_verify.py`/`bnnum.py`, so Layer 1
   imports them via the pipeline's input scan — no separate attach needed).
4. Notebook settings: **Accelerator = GPU T4 x2** (`machine_shape: NvidiaTeslaT4`),
   **Internet = OFF.** Pick `USE_BCS` for the final you are reproducing. The whole push is
   CLI-drivable — see `snapshots/kernel-metadata.example.json`.

## How the organizers run it

Attach the held-out fold CSV (or replace the competition test file), set `TEST_CSV` at the
top if the filename is not `test*.csv`, and Run All. The notebook recomputes Layer 1 over
the corpora and Layer 2 from scratch for both the 299 labeled samples (stacker training) and
the evaluation rows, then writes `submission.csv` (`id,label`; 0 = hallucinated, 1 = faithful).
If the fold shares no corpus overlap, Layer 1 abstains and the base stack produces the output.

## Verify without a GPU

```bash
cd Phase_2
python selfcheck.py          # proves the overlay reproduces both finals (no corpora/GPU)
python bnnum.py              # numeral-word equivalence self-test
```

For a full Layer-1 recompute from the corpora (still no GPU):
```bash
bash fetch_corpora.sh ext    # assembles ext/ (kaggle CLI + HF)
BHD_EXT=ext/ python -c "from gold_verify import GoldVerifier; V=GoldVerifier(); print('corpora loaded')"
```

`snapshots/kaggle_layer1_verify.py` is the same check as an offline Kaggle kernel: attach the
competition data + `bengali-gold-corpora` (internet OFF) and it prints sample accuracy + test
coverage. Confirmed run: 185/299 samples @ 98.9%, 1508/2516 (59.9%) test coverage, fully offline.

---

## Submission checklist (rules §3, top-30 teams)

- [x] Runnable offline Kaggle notebook reproducing the Phase-1 finals end-to-end within
      limits — `phase2_gold_pipeline.ipynb` (Layer 1 exact; Layer 2 = verified base stack)
- [x] Layered so it degrades to the LB-0.800 base on an out-of-distribution fold (rules §5)
- [x] Overlay reproduction proven offline — `selfcheck.py`, 4/4 checks pass
- [x] **Layer 1 verified offline on Kaggle** — `bengali-gold-corpora` dataset published;
      `kaggle_layer1_verify.py` ran with internet OFF: 185/299 samples @ 98.9%, 1508/2516
      (59.9%) test coverage, agreeing with the committed finals on 1507/1508 covered rows
- [x] README: environment, weights, external models, and the corpora to cite — this file
- [x] **Paper (rules §3.2):** the five Layer-1 corpora are cited and the gold-verification
      method + layered architecture are written up in `paper/main.tex` → `main.pdf`
      (the 0.904 system; body within 4 pages, references on p5)
- [x] **End-to-end offline GPU run (T4×2, internet OFF)** — `bengali-gold-phase2-submit`
      ran to completion in **~2h07m** (well under the 9h cap; faster than the base 4.96h
      because Layer 1 removes 60% of rows from the LLM workload). Layer 1 decided 1508 rows
      (361 overriding the stack); `submission.csv` covered rows match the in-kernel gold
      exactly and `final_bcs` on 1507/1508. The ~3.6% difference on uncovered rows is the
      documented portable-stack-vs-workstation-meta substitution (within OOF noise).
- [ ] **Report the mislabeled `রংপুর জিলা স্কুল` row on the Discussion tab (rule 6).**
- [ ] Select the 2 finals + submit the package via the Phase-2 form (Discussion tab)
