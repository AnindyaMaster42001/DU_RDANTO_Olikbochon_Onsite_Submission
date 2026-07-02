"""Query-expansion retrieval (v2): dual-query = question + response, larger pool.

Reuses the cached corpus_emb.npy (no corpus re-embed). For each no-context row,
retrieve top-8 for the QUESTION and top-8 for the RESPONSE, union them, and keep
the FINAL_K best by max similarity -> better recall on hard C1 items (the answer's
entities pull in the right article even when the question alone doesn't).
Overwrites retrieved_evidence.json.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEV = "cuda"
TOPK_EACH = 8
FINAL_K = 6
BATCH = 128
MAXLEN = 256
RESP_CHARS = 250

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).to(DEV).eval()


@torch.no_grad()
def embed(texts, tag=""):
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt").to(DEV)
        h = torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0], dim=-1)
        out.append(h.cpu().numpy().astype(np.float16))
    print(f"  embedded {tag}: {len(texts)}", flush=True)
    return np.concatenate(out)


def hasctx(r):
    return str(r["context"]).strip() not in ("[NULL]", "NULL", "null", "")


def find(part, ext):
    for f in Path(".").iterdir():
        if part in f.name.lower() and f.suffix == ext:
            return f
    raise FileNotFoundError(part)


def main():
    pas = [json.loads(l) for l in open("passages.jsonl", encoding="utf-8")]
    ptxt = [f"{p['t']}: {p['x']}" for p in pas]
    C = np.load("corpus_emb.npy").astype(np.float32)
    print(f"corpus: {C.shape}", flush=True)

    samples = json.load(open(find("samples", ".json"), encoding="utf-8"))
    test = list(csv.DictReader(open(find("test", ".csv"), encoding="utf-8")))
    rows = [(f"samples:{i}", s) for i, s in enumerate(samples) if not hasctx(s)]
    rows += [(f"test:{r['id']}", r) for r in test if not hasctx(r)]
    keys = [k for k, _ in rows]
    print(f"no-context queries: {len(rows)}", flush=True)

    Q = embed([r["prompt_bn"] for _, r in rows], "questions").astype(np.float32)
    R = embed([str(r["response_bn"])[:RESP_CHARS] for _, r in rows], "responses").astype(np.float32)

    ev = {}
    for bi in range(0, len(rows), 128):
        qs = Q[bi:bi + 128] @ C.T
        rs = R[bi:bi + 128] @ C.T
        for j in range(qs.shape[0]):
            qi = np.argpartition(-qs[j], TOPK_EACH)[:TOPK_EACH]
            ri = np.argpartition(-rs[j], TOPK_EACH)[:TOPK_EACH]
            cand = set(qi.tolist()) | set(ri.tolist())
            sc = {c: max(float(qs[j][c]), float(rs[j][c])) for c in cand}
            top = sorted(cand, key=lambda c: -sc[c])[:FINAL_K]
            ev[keys[bi + j]] = [ptxt[t] for t in top]
    json.dump(ev, open("retrieved_evidence.json", "w"), ensure_ascii=False)
    print(f"wrote retrieved_evidence.json (dual-query, top-{FINAL_K}, {len(ev)} queries)", flush=True)


if __name__ == "__main__":
    main()
