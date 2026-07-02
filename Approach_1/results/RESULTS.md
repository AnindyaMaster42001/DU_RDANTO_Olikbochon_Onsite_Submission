# Approach_1 — full-run results

Full pipeline (NLI → judge → ensemble) run on an RTX PRO 6000 (Blackwell) GPU,
Qwen2.5-14B-Instruct-AWQ judge via vLLM (self-consistency, 5 votes on no-context
rows). Numbers below are **honest out-of-fold** on the 299 labeled samples
(per-branch nested CV in `ensemble.py`).

## Headline

| Branch | Approach_0 baseline | **Approach_1 ensemble** |
|---|---|---|
| **Overall macro-F1** | 0.6823 (LB 0.666) | **~0.76** |
| Context | 0.894 (substring) | **0.87 – 0.90** |
| No-context | **0.345** | **0.58 – 0.65** |

The no-context branch — the whole opportunity — roughly **doubled** (0.345 → ~0.65
on the judge's own samples pass; ~0.58 under strict nested CV). That is the
Qwen-14B fact-checking judge working as designed.

## Per-signal (samples pass)

| Signal | Context macro-F1 | No-context macro-F1 |
|---|---|---|
| substring (rule) | **0.894** | — (0.5 constant) |
| judge (Qwen2.5-14B, self-consistency) | 0.801 | **0.649** |
| NLI (mDeBERTa-xnli) | 0.635 | — |

**NLI underperformed the substring rule on context** (0.635 vs 0.894) — feeding a
short extractive answer as an NLI hypothesis is a poor fit. The ensemble meta-model
correctly handled this: learned weights were
`context {substring 2.57, judge 2.07, nli 0.92}`, `no-context {judge 1.66, nli 0, substring 0}`.

## Caveat

The combined "overall ~0.76" line re-picks the decision threshold on the OOF
predictions (mildly optimistic). The stricter per-branch nested-CV OOF was
**context 0.870 / no-context 0.581**. Treat ~0.72–0.76 as the honest range.

## Files

- `signal_judge.json` — per-row P(faithful) from the Qwen judge (samples + test).
  **Reuse this to re-run the ensemble without a GPU.**
- `submission_ensemble.csv` — final submission (1386 hallucinated / 1130 faithful).
- `submission_judge.csv` — judge-only submission (1523 / 993).

## Reproduce

```bash
# GPU (Kaggle T4x2/P100, or any CUDA box): produces signal_judge.json
python kaggle_judge.py
# CPU: substring + NLI signals, then stitch
python nli_grounding.py && python ensemble.py   # ensemble auto-loads all signal_*.json
```

## Next levers (not yet done)

- **Retrieval branch** (`retrieval.py`) — ground no-context C1 questions in Bengali
  Wikipedia instead of the judge's parametric memory. Expected to lift no-context
  further; needs the wiki dump attached.
- Reformulate NLI (question+answer → declarative claim) or drop it for context.
