"""Approach_1 · Step 1 — NLI grounding for the context branch.

Replaces Approach_0's brittle substring check with a semantic entailment test:
    premise = context (chunked), hypothesis = response  ->  P(entail)
A response is faithful (1) iff the context entails it above a threshold. Long
contexts are split into sentence-packed chunks and we take the MAX entailment
over chunks, so the relevant evidence isn't truncated away.

No-context rows are left at the best constant for now (retrieval.py replaces
this branch). The per-row P(faithful) is written as a signal for ensemble.py.

Graceful: if torch/transformers aren't installed, it falls back to the
substring rule so the script still runs and produces a valid submission.
Install the real path with:  pip install -r requirements.txt

Usage:  python3 nli_grounding.py
"""

import numpy as np

import common as C

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MAX_TOKENS = 384        # per chunk (model cap is 512, leave room for hypothesis)
MAX_CHUNKS = 6          # cap chunks/row to bound compute
BATCH = 32

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAVE_NLI = True
except ImportError:
    HAVE_NLI = False


# ---------------------------------------------------------------- NLI backend
class NLI:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        # label order varies by checkpoint — read it, don't hardcode
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail = next(i for i, l in id2label.items() if "entail" in l)
        print(f"NLI on {self.device}; entail idx = {self.entail}")

    def _chunks(self, text):
        # sentence-pack into <= MAX_TOKENS chunks (Bengali danda / western enders)
        import re
        parts = [p for p in re.split(r"(?<=[।!?\n])", str(text)) if p.strip()]
        chunks, cur = [], ""
        for p in parts:
            if len(self.tok.tokenize(cur + p)) > MAX_TOKENS and cur:
                chunks.append(cur)
                cur = ""
            cur += p
        if cur:
            chunks.append(cur)
        return (chunks or [str(text)])[:MAX_CHUNKS]

    def entail_probs(self, rows):
        """For each row -> max P(entail) over context chunks vs the response."""
        pairs, owner = [], []  # (premise_chunk, hypothesis), row index
        for i, r in enumerate(rows):
            for ch in self._chunks(r["context"]):
                pairs.append((ch, str(r["response_bn"])))
                owner.append(i)
        out = np.zeros(len(rows))
        with torch.no_grad():
            for b in range(0, len(pairs), BATCH):
                prem = [p for p, _ in pairs[b:b + BATCH]]
                hyp = [h for _, h in pairs[b:b + BATCH]]
                enc = self.tok(prem, hyp, truncation=True, max_length=512,
                               padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                p = torch.softmax(logits, -1)[:, self.entail].cpu().numpy()
                for j, prob in enumerate(p):
                    k = owner[b + j]
                    out[k] = max(out[k], float(prob))
        return out


# ---------------------------------------------------------------- prediction
def context_signal(rows):
    """Return per-row P(faithful) for context rows (0.5 for no-context rows)."""
    sig = np.full(len(rows), 0.5)
    ctx_idx = [i for i, r in enumerate(rows) if C.has_context(r)]
    if not ctx_idx:
        return sig
    if HAVE_NLI:
        probs = NLI().entail_probs([rows[i] for i in ctx_idx])
    else:
        print("!! torch/transformers not installed — substring fallback for context rows")
        probs = np.array([
            1.0 if (C.norm(rows[i]["response_bn"]) and
                    C.norm(rows[i]["response_bn"]) in C.norm(rows[i]["context"]))
            else 0.0
            for i in ctx_idx
        ])
    for i, p in zip(ctx_idx, probs):
        sig[i] = p
    return sig


def sweep_threshold(samples, sig):
    """Best entailment threshold on the CONTEXT sample rows (macro-F1)."""
    ctx = [i for i, s in enumerate(samples) if C.has_context(s)]
    y = [samples[i]["label"] for i in ctx]
    best_t, best_m = 0.5, -1.0
    for t in np.arange(0.30, 0.71, 0.02):
        preds = [1 if sig[i] >= t else 0 for i in ctx]
        m = C.macro_f1(y, preds)
        if m > best_m:
            best_t, best_m = float(t), m
    return best_t, best_m


def main():
    samples, test = C.load_samples(), C.load_test()
    noctx_const = C.pick_noctx_constant(samples)

    s_sig = context_signal(samples)
    t_sig = context_signal(test)

    thr, ctx_m = sweep_threshold(samples, s_sig)
    print(f"\ncontext-branch threshold = {thr:.2f}  (context macroF1 {ctx_m:.4f})")
    print("NOTE: threshold picked on samples for a quick read; ensemble.py sets it via nested CV.\n")

    def predict(rows, sig):
        return [
            (1 if sig[i] >= thr else 0) if C.has_context(r) else noctx_const
            for i, r in enumerate(rows)
        ]

    print("=== sample-set score (NLI context branch + constant no-context) ===")
    C.evaluate_samples(predict(samples, s_sig))

    test_ids = [r["id"] for r in test]
    C.write_submission("submission_nli.csv", test_ids, predict(test, t_sig))
    C.save_signal(
        "nli",
        [float(v) for v in s_sig],
        {r["id"]: float(v) for r, v in zip(test, t_sig)},
    )


if __name__ == "__main__":
    main()
