# অলীকবচন — Bengali LLM Hallucination Detection

Team workspace for the [Bengali LLM Hallucination Detection Challenge](https://www.kaggle.com/competitions/bengali-hallucination)
(Datathon 2.0 · Institute of Policy Dynamics · IUTCS).

**Task:** given a Bengali `prompt_bn`, a `response_bn`, and an optional `context`
(`[NULL]` if none), predict whether the response is **faithful** (`label = 1`) or
**hallucinated** (`label = 0`). Metric: macro-F1 (Phase 2 scores binary F1 on the
hallucinated class).

## Layout

```
Approach_0/
  dataset samples.json     299 labeled samples (validation set — NOT a train set)
  test set.csv             2516 unlabeled rows to predict
  sample submission.csv    submission format (id,label)

  evaluate.py              shared metric harness (macro-F1 + per-branch report)
  baseline.py              #1 rule-based: substring-grounding + constant guess
  train_local.py           fine-tune BanglaBERT (5-fold CV -> test probs, ensemble signal)
  kaggle_judge.py/.ipynb   #2 LLM-as-judge (Qwen2.5-14B) — grounding + self-verify
  submission_baseline.csv  baseline predictions (LB 0.666)
```

## Core idea

Every approach splits each row into two sub-problems:
- **context present** → *grounding*: is the response supported by the passage?
- **no context** → *closed-book factuality*: is the response factually true?

## Quick start

```bash
python3 Approach_0/baseline.py                       # writes submission_baseline.csv
python3 Approach_0/evaluate.py Approach_0/submission_baseline.csv
```

`train_local.py` needs a GPU + `transformers`/`torch`; `kaggle_judge.py` is meant to
run in a Kaggle notebook with a T4×2 / P100 GPU (vLLM + Qwen2.5-14B-Instruct-AWQ).
