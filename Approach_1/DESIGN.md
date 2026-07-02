# Approach_1 — retrieval-augmented Bengali fact verification

> Design doc for the team. Approach_0 hit **LB 0.666** (≈ local macro-F1 0.6823).
> This is the plan to beat it, and the scaffolding to build it incrementally.

## Diagnosis — what caps Approach_0

| Weakness | Root cause | Fix in Approach_1 |
|---|---|---|
| Substring grounding (context rows) | Lexical, not semantic — fails on paraphrase, fooled by word overlap | **NLI entailment** (`nli_grounding.py`) |
| No-context factuality leans on the judge LLM's memory | The judge hallucinates on the **C1 Bangladesh-specific** facts we're graded on | **Retrieval-augmented verification** (`retrieval.py`) |
| "Best of 16 combos / sweep threshold on 299 rows" | Selecting on the whole sample set overfits it | **Calibrated ensemble + nested CV** (`ensemble.py`) |

## Central bet

**Never ask a model to recall a fact; always ask it to check a claim against
evidence.** That is the only way to beat C1, where the model's parametric
knowledge is the problem, not the solution. It also gives the Phase-2 paper a
clean FEVER-style fact-verification framing (novelty points).

```
                       ┌──────────── row: has context? ────────────┐
                 YES ──┤                                            ├── NO
                       ▼                                            ▼
            GROUNDING branch                          RETRIEVAL branch
            premise = context (chunked)               1. retrieve evidence from
            hypothesis = response                        Bengali Wikipedia
            → NLI P(entail)                              (BM25 + dense e5, RRF)
            + judge grounding logprob                 2. run the SAME grounding
                                                         stack on the evidence
                       └───────────────┬────────────────────────────┘
                                       ▼
                   per-row P(faithful) signals  →  signal_*.json
                                       ▼
                   calibrated logistic meta-model, PER BRANCH,
                   nested-CV threshold  →  submission_ensemble.csv
```

## Files

| File | Role | Status |
|---|---|---|
| `common.py` | data loading (path autodetect), metric, signal I/O | ✅ done |
| `nli_grounding.py` | **Step 1** — NLI grounding for context rows | ✅ runnable |
| `judge.py` | **Step 2** — LLM judge: self-consistency + few-shot + logprob | scaffold (wire vLLM) |
| `retrieval.py` | **Step 3** — Bengali-Wikipedia hybrid retrieval (the C1 lever) | scaffold (attach dump) |
| `ensemble.py` | **Step 4** — calibrated per-branch meta-model, nested CV | ✅ runnable |

Every stage writes a `signal_<name>.json` = per-row `P(faithful)`. `ensemble.py`
consumes whatever signals exist, so stages are independently shippable.

## Build order

1. **NLI grounding** — replace substring on context rows. Highest confidence.
   `pip install -r requirements.txt && python3 nli_grounding.py`
2. **Judge upgrade** — logprob + self-consistency vote instead of one greedy call.
3. **Retrieval branch** — the C1 lever; converts closed-book → open-book. Most
   upside, most effort. Attach a Bengali Wikipedia dump as a Kaggle dataset.
4. **Ensemble** — stack the signals, pick threshold inside CV. Stops overfitting
   the 299. `python3 ensemble.py`

## Validation discipline (the thing Approach_0 got wrong)

- **Nested CV**: choose feature weights *and* threshold inside folds; report
  out-of-fold macro-F1 + hallucinated-class F1.
- **Per branch**: context (48/84 hall/faithful) and no-context (88/79) have very
  different base rates — fit and threshold them separately.
- **C1 probe**: hand-flag ~30–40 Bangladesh-specific sample rows and track F1
  there — the private LB weights C1 heaviest but the released data has no band
  labels, so this is our only visibility into what decides ranking.
- **Augmentation** (rules allow it): mint synthetic hallucinations by perturbing
  faithful rows (swap entities/dates/numbers) to grow the hallucinated class.

## Models (all open-weight → Phase-2 compliant)

| Stage | Model | Why |
|---|---|---|
| NLI | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | multilingual entailment, tiny, handles Bengali |
| Dense retrieval | `intfloat/multilingual-e5-large` (or LaBSE) | strong multilingual passage retrieval |
| Judge | Qwen2.5-14B/32B-Instruct-AWQ · Gemma-2-27B · TigerLLM/TituLLM | world knowledge; Bengali-native options blessed by rules |

## Compute (Phase-2: <9h on P100 / 2×T4, ≤50GB)

Comfortable. NLI + BM25 are cheap; dense-embedding 2516 queries is seconds;
build the wiki index once and cache it as a dataset. The 14–32B AWQ judge is the
only heavy part — Approach_0 already ran a 14B judge in ~1–2h.
