# Approach_1 — full-run results

Full pipeline (NLI → judge → ensemble) run on an RTX PRO 6000 (Blackwell) GPU,
Qwen2.5-14B-Instruct-AWQ judge via vLLM (self-consistency, 5 votes on no-context
rows). Numbers below are **honest out-of-fold** on the 299 labeled samples
(per-branch nested CV in `ensemble.py`).

## Headline (honest nested-CV OOF)

| Branch | Approach_0 baseline | judge (3 signals) | **+ cross-lingual (4 signals)** |
|---|---|---|---|
| **Overall macro-F1** | 0.6823 (LB 0.666) | 0.7647 | **0.7787** |
| Context | 0.894 (substring) | 0.901 | **0.901** |
| No-context | **0.345** | 0.649 | **0.670** |

The no-context branch — the whole opportunity — went from 0.345 → 0.67. First the
Qwen-14B fact-checking judge (parametric), then a **cross-lingual consistency**
signal (answer in English, check the Bengali response agrees) added a
*complementary* lift: the no-context meta-model weights both
(`judge 1.39 + crosslingual 1.22`).

## Per-signal (samples pass)

| Signal | Context macro-F1 | No-context macro-F1 |
|---|---|---|
| substring (rule) | **0.894** | — (0.5 constant) |
| judge (Qwen2.5-14B, self-consistency) | 0.801 | 0.649 |
| cross-lingual (answer-in-English + agreement) | — | 0.615 |
| NLI (mDeBERTa-xnli) | 0.635 | — |

**NLI underperformed the substring rule on context** (0.635 vs 0.894) — feeding a
short extractive answer as an NLI hypothesis is a poor fit. The ensemble meta-model
correctly handled this: `context {substring 2.57, judge 2.08, nli 0.92, crosslingual 0}`,
`no-context {judge 1.39, crosslingual 1.22, nli 0, substring 0}`.

**Key ceiling finding:** in the cross-lingual pass, **~44% of no-context questions
abstained** — Qwen-14B replied UNKNOWN in English too. That fraction is genuinely
beyond the model's parametric knowledge (deep C1, Bangladesh-specific) and is
exactly what **retrieval** (Step 3) must cover to push no-context past ~0.67.

## Caveat

The combined "overall ~0.76" line re-picks the decision threshold on the OOF
predictions (mildly optimistic). The stricter per-branch nested-CV OOF was
**context 0.870 / no-context 0.581**. Treat ~0.72–0.76 as the honest range.

## Files

- `signal_judge.json`, `signal_crosslingual.json` — per-row P(faithful) from the
  Qwen judge and the cross-lingual check (samples + test).
  **Reuse these to re-run the ensemble without a GPU.**
- `submission_ensemble.csv` — final 4-signal submission (1425 hallucinated / 1091 faithful).
- `submission_judge.csv` — judge-only submission (1523 / 993).

## Reproduce

```bash
# GPU (Kaggle T4x2/P100, or any CUDA box): produce the two judge signals
python kaggle_judge.py            # -> signal_judge.json
python crosslingual.py            # -> signal_crosslingual.json
# CPU: substring + NLI signals, then stitch everything
python nli_grounding.py && python ensemble.py   # ensemble auto-loads all signal_*.json
```

## Next levers (not yet done)

- **Retrieval branch** (`retrieval.py`) — ground no-context C1 questions in Bengali
  Wikipedia instead of the judge's parametric memory. Expected to lift no-context
  further; needs the wiki dump attached.
- Reformulate NLI (question+answer → declarative claim) or drop it for context.
