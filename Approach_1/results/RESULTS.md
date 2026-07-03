# Approach_1 — full-run results

Full pipeline run on an RTX PRO 6000 (Blackwell) GPU: substring rule, NLI,
Qwen2.5-14B-Instruct-AWQ judge (self-consistency), cross-lingual check, and a
retrieval branch grounded in the **latest Bengali Wikipedia** — stacked by a
per-branch nested-CV meta-model. Numbers are **honest out-of-fold** on the 299
labeled samples (`ensemble.py`).

## Headline (honest nested-CV OOF)

| Branch | Baseline | +judge (3 sig) | +cross-lingual (4) | **+retrieval (5)** |
|---|---|---|---|---|
| **Overall macro-F1** | 0.6823 (LB 0.666) | 0.7647 | 0.7787 | **0.7925** |
| Context | 0.894 | 0.901 | 0.901 | **0.901** |
| No-context | **0.345** | 0.649 | 0.670 | **0.675** |

The no-context branch (the whole opportunity, and where the private LB's C1 weight
concentrates) went **0.345 → 0.675** — nearly doubled. Overall **+0.11 over baseline**.

## Signals & how the meta-model weights them

| Signal | Context F1 | No-context F1 | no-ctx weight |
|---|---|---|---|
| substring (rule) | **0.894** | — | 0 |
| judge (Qwen-14B, self-consistency) | 0.801 | 0.649 | 1.09 |
| cross-lingual (answer-in-English + agree) | — | 0.615 | 1.09 |
| **retrieval** (bge-m3 → wiki → Qwen grounds) | — | **0.656** | **1.55** |
| NLI (mDeBERTa-xnli) | 0.635 | — | 0 |

- **Context** is solved by `substring` (2.57) + `judge` (2.07); NLI down-weighted
  (0.92), cross-lingual/retrieval ~0. NLI *underperformed* substring (0.635 vs 0.894)
  — a short extractive answer is a poor NLI hypothesis.
- **No-context**: `retrieval` gets the **largest** weight (1.55), with judge and
  cross-lingual both contributing (1.09 each). All three are *complementary* — the
  three attack the closed-book problem from different angles (parametric, cross-lingual,
  external-knowledge).

**Why retrieval matters:** in the cross-lingual pass, **~44% of no-context questions
abstained** — Qwen-14B was UNKNOWN in English too. Those deep-C1 (Bangladesh-specific)
facts are beyond any parametric model; only the wiki-grounded retrieval branch can
reach them, which is why it earns the top no-context weight.

## Corpus (retrieval branch)

- Source: **`dumps.wikimedia.org/bnwiki/latest`** (dump `20260701`, current) — NOT a
  stale pre-cleaned HF snapshot. Raw wikitext → `wikiextractor` (Python 3.10) →
  **438,788 articles** → chunked to **335,628 passages**.
- Retriever: **BAAI/bge-m3** dense (normalized CLS), top-5 per no-context question.
- Grounding: Qwen-14B judges whether the response follows from the retrieved passages
  (self-consistency, 5 votes).

## Caveat

The combined "overall" line re-thresholds on the OOF (mildly optimistic). Stricter
per-branch nested-CV OOF: context 0.870 / no-context 0.653. Treat ~0.76–0.79 as the
honest range. 299 samples → real variance; trust the *direction*, confirm on the LB.

## Files

- `signal_judge.json`, `signal_crosslingual.json`, `signal_retrieval.json` — per-row
  P(faithful). **Reuse to re-run the ensemble without a GPU.**
- `submission_ensemble.csv` — final 5-signal submission (1475 hallucinated / 1041 faithful).

## Reproduce

```bash
# GPU: three signals (Qwen-14B via vLLM; bge-m3 for retrieval)
python kaggle_judge.py            # -> signal_judge.json
python crosslingual.py            # -> signal_crosslingual.json
# retrieval: fetch latest bnwiki dump, extract (py3.10 wikiextractor), then:
python prep_wiki.py               # extracted/**/wiki_* -> passages.jsonl
python retrieve.py                # bge-m3 dense -> retrieved_evidence.json
python ground.py                  # Qwen grounds vs evidence -> signal_retrieval.json
# CPU: substring + NLI, then stitch all five
python nli_grounding.py && python ensemble.py
```

## Ablation: query expansion (tried — neutral)

`retrieve2.py` = dual-query retrieval (embed the **question** *and* the **response**,
union, top-6) + token-safe grounding (`ground.py` now truncates prompts to fit
Qwen's 4096 limit — top-6 Bengali evidence otherwise overflows and vLLM hard-errors).

Result: the retrieval *signal* nudged up (samples 0.6564 → 0.6635; no-context
per-branch OOF 0.6527 → 0.6618), but the **pooled ensemble was flat within noise**
(overall 0.7925 vs 0.7821 — threshold/pooling wiggle on 299 rows). bge-m3 on the
question alone already retrieves well. **Kept v1 (single-query, top-5) as canonical.**

## Next levers (untried)
- Rerank with `bge-reranker-v2-m3` (on the box) before grounding — precision bump.
- Passage-level chunking beyond the first 3 chunks/article; larger top-k with the
  token-safe grounding now that overflow is handled.
- C1 probe set to track the band the LB actually rewards.

---

## Update: 7-signal ensemble (+ Approach_0 judge & self-verify) — LB 0.765

Approach_0's Kaggle judge run (Qwen2.5-14B-**GPTQ**, greedy, different prompts
than Approach_1's self-consistency judge; LB **0.748** standalone) exports two
extra signals. `import_a0_signals.py` converts them (plus the substring rule)
into ensemble format; `ensemble.py` picks them up with zero code changes.

| | 5 signals | **7 signals** |
|---|---|---|
| Overall OOF macro-F1 | 0.7925 | **0.8059** |
| Context | 0.9009 | **0.9087** |
| No-context | 0.6748 | **0.7078** |
| **Public LB** | **0.752** | **0.765** |

- The two judges are *complementary*, not redundant: `a0judge` earns weight
  1.67 in the context branch (alongside substring 2.33 and judge 1.36) and
  0.78 in no-context (alongside retrieval 1.46 and cross-lingual 1.02).
  Different prompts + different quantization = decorrelated errors.
- `a0selfv` (answer-then-compare) adds a small 0.26 no-context weight.
- Both LB checks landed on the local→LB line (−0.03/−0.04 from OOF), so the
  sample split keeps being a trustworthy compass.

Raw Approach_0 verdicts live in `Approach_0/results/signals_{samples,test}.json`.

### Next levers (updated)
- Reranker (`bge-reranker-v2-m3`) before grounding — precision bump on retrieval.
- Second judge model (Gemma-2-27B / TituLLM) as an 8th signal — the a0judge
  gain shows judge diversity pays.
- C1 probe set — still our only visibility into the band the private LB weights.

---

## Update: 8th signal — Qwen2.5-32B judge (single greedy pass)

Meta-model experiments (`experiments_meta.py`, 5-seed x 5-fold) showed the
7-signal stacker is saturated: honest seed-averaged OOF is **0.7946 +-0.0089**
(the 0.8059 previously quoted was one favorable seed), and abstain indicators,
BanglaBERT, branch pruning, and C sweeps all fail to beat it. Only new signal
strength moves it, so: `kaggle_judge32.py` = Qwen2.5-**32B**-Instruct-GPTQ-Int4
on T4x2, one YES/NO per row (~30 min total), stitched by `stitch32.py`.

| | 7 signals | **8 signals** |
|---|---|---|
| OOF macro-F1 (5-seed mean) | 0.7946 +-0.0089 | **0.8032 +-0.0032** |
| No-context | 0.6942 | **0.7169** |
| Context | 0.9026 | 0.8941 |

- judge32 is the strongest standalone no-context signal (0.6618) and takes the
  top meta-weights (ctx 1.53 / noctx 1.33) on arrival — parametric scale
  directly buys C1-adjacent knowledge.
- Seed variance drops 3x: the ensemble got more stable, not just better.
- `submission_final.csv` = the 8-signal predictions (LB pending).

### LB check: 8-signal scored 0.759 (7-signal remains 0.765)

Public LB came back **0.759** — below the 7-signal's 0.765 despite the better
honest OOF (0.8032 vs 0.7946). Read: a +0.009 OOF gain on 299 rows does not
reliably transfer through public-LB split noise (~1.2k rows); these two
ensembles are **statistically indistinguishable** with the data we have. Do
not chase public-LB deltas this small in either direction.

**Final-selection recommendation (Phase 1 deadline):** select BOTH
`submission_ensemble7.csv` (best public, 0.765) and `submission_final.csv`
(best honest OOF, lowest seed variance) as the two finals — hedges the
public/private split noise instead of betting on it.

**Where a real (step-change) gain must come from** — the no-context branch is
still ~0.72 OOF and small signal tweaks now drown in validation noise:
1. Retrieval upgrades (reranker, deeper chunking, larger top-k) — needs the
   bnwiki index box.
2. 32B judge + self-consistency + retrieval-grounded prompts (~90 min Kaggle
   run) — merge the two strongest ideas.
