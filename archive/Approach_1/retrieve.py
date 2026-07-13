"""Phase B — bge-m3 dense retrieval: top-k Bengali-wiki passages per no-context query.

Uses the box's cached BAAI/bge-m3 via plain transformers (dense = normalized CLS).
GPU, ~3-5 GB (bge-m3 only; run alone, before the Qwen grounding phase). Corpus
embeddings are cached to corpus_emb.npy so re-runs are instant.
Emits retrieved_evidence.json  {"samples:<i>"|"test:<id>": [passage, ...]}.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEV = "cuda"
TOPK = 5
BATCH = 128
MAXLEN = 512

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).to(DEV).eval()


@torch.no_grad()
def embed(texts, tag=""):
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt").to(DEV)
        h = model(**enc).last_hidden_state[:, 0]
        h = torch.nn.functional.normalize(h, dim=-1)
        out.append(h.cpu().numpy().astype(np.float16))
        if (i // BATCH) % 100 == 0:
            print(f"  embed {tag} {i}/{len(texts)}", flush=True)
    return np.concatenate(out)


def hasctx(r):
    return str(r["context"]).strip() not in ("[NULL]", "NULL", "null", "")


def find(part, ext=None):
    for f in Path(".").iterdir():
        if part in f.name.lower() and (ext is None or f.suffix == ext):
            return f
    raise FileNotFoundError(part)


def main():
    pas = [json.loads(l) for l in open("passages.jsonl", encoding="utf-8")]
    ptxt = [f"{p['t']}: {p['x']}" for p in pas]
    print(f"passages: {len(pas)}", flush=True)

    if os.path.exists("corpus_emb.npy"):
        C = np.load("corpus_emb.npy")
    else:
        C = embed(ptxt, "corpus")
        np.save("corpus_emb.npy", C)
    print(f"corpus emb: {C.shape}", flush=True)
    Cf = C.astype(np.float32)

    samples = json.load(open(find("samples", ".json"), encoding="utf-8"))
    test = list(csv.DictReader(open(find("test", ".csv"), encoding="utf-8")))
    queries = [(f"samples:{i}", s["prompt_bn"]) for i, s in enumerate(samples) if not hasctx(s)]
    queries += [(f"test:{r['id']}", r["prompt_bn"]) for r in test if not hasctx(r)]
    print(f"no-context queries: {len(queries)}", flush=True)

    Q = embed([q for _, q in queries], "queries").astype(np.float32)
    ev = {}
    for bi in range(0, len(queries), 256):
        sims = Q[bi:bi + 256] @ Cf.T
        idx = np.argpartition(-sims, TOPK, axis=1)[:, :TOPK]
        for j, (key, _) in enumerate(queries[bi:bi + 256]):
            top = idx[j][np.argsort(-sims[j][idx[j]])]
            ev[key] = [ptxt[t] for t in top]
    json.dump(ev, open("retrieved_evidence.json", "w"), ensure_ascii=False)
    print(f"wrote retrieved_evidence.json ({len(ev)} queries)", flush=True)


if __name__ == "__main__":
    main()
