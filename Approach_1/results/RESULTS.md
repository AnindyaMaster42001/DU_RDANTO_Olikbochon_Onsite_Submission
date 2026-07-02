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

## Next levers
- Rerank retrieved passages (bge-reranker-v2-m3, already on the box) before grounding.
- Query expansion with response entities; larger top-k; passage-level (not lead-only).
- C1 probe set to track the band the LB actually rewards.
