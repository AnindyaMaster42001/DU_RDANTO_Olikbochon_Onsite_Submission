"""
Context-grounded gold extraction.

For rows that ship a context passage, the gold answer IS a span of that passage.
Rather than asking an LLM "is this answer faithful?", extract the answer span with an
extractive QA model and hand it to the same equivalence check used for gold lookup.

Targets the stack's weakest slice: uncovered context rows (OOF acc 0.733).
Emits the extracted span + confidence for every context row in test and samples.
"""
import os, json, glob
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

OUT = "/kaggle/working"
def find(p):
    h = glob.glob(f"/kaggle/input/**/{p}", recursive=True)
    assert h, p
    return h[0]

test = pd.read_csv(find("test set.csv"))
samples = pd.DataFrame(json.load(open(find("dataset samples.json"))))

dev = "cuda"
cap = torch.cuda.get_device_capability(0)
print("GPU:", torch.cuda.get_device_name(0), cap, flush=True)
assert cap >= (7, 0), f"need T4, got capability {cap}"

MODEL = "deepset/xlm-roberta-large-squad2"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL).half().to(dev).eval()

MAXLEN, STRIDE = 384, 128

@torch.no_grad()
def extract(question, context):
    """Best answer span over sliding windows; returns (text, score)."""
    enc = tok(question, context, truncation="only_second", max_length=MAXLEN,
              stride=STRIDE, return_overflowing_tokens=True,
              return_offsets_mapping=True, padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    enc = {k: v.to(dev) for k, v in enc.items()}
    out = model(**enc)
    best = (-1e9, "")
    for w in range(out.start_logits.shape[0]):
        s, e = out.start_logits[w].float().cpu().numpy(), out.end_logits[w].float().cpu().numpy()
        seq = enc["input_ids"][w].cpu().numpy()
        off = offsets[w].numpy()
        # only allow spans inside the context (offset != (0,0) and not the question part)
        valid = np.array([not (o[0] == 0 and o[1] == 0) for o in off])
        # crude context mask: everything after the first [SEP]-ish boundary
        si = np.argsort(s)[-20:][::-1]
        ei = np.argsort(e)[-20:][::-1]
        for i in si:
            for j in ei:
                if j < i or j - i > 40: continue
                if not (valid[i] and valid[j]): continue
                sc = s[i] + e[j]
                if sc > best[0]:
                    best = (sc, context[off[i][0]:off[j][1]])
    return best[1], float(best[0])

def run(df, tag):
    rows = []
    for k, r in enumerate(df.itertuples()):
        ctx = str(r.context)
        if ctx.strip() == "[NULL]" or not ctx.strip():
            rows.append(("", float("nan"))); continue
        try:
            rows.append(extract(str(r.prompt_bn), ctx))
        except Exception as ex:
            print("err", k, ex, flush=True); rows.append(("", float("nan")))
        if k % 200 == 0: print(tag, k, "/", len(df), flush=True)
    return rows

tr = run(test, "test")
pd.DataFrame({"id": test.id, "span": [a for a, _ in tr], "score": [b for _, b in tr]}).to_csv(f"{OUT}/ctx_spans_test.csv", index=False)
sr = run(samples, "samples")
pd.DataFrame({"idx": range(len(samples)), "label": samples.label,
              "span": [a for a, _ in sr], "score": [b for _, b in sr]}).to_csv(f"{OUT}/ctx_spans_samples.csv", index=False)
print("done")
