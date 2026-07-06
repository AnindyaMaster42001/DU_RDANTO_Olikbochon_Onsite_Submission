"""Wiktionary dense retrieval (bge-m3, ~2-5 GB, safe on shared box).

Corpus = wikt_passages.jsonl (headword: meaning). For each no-context query
(samples + test), retrieve top-5 dictionary entries. Emits wikt_evidence.json
{"s<i>"|"t<id>": [{"title","text"}, ...]}. Caches wikt_emb.npy.
Run from ~/claude_bengali (has 'dataset samples.json' + 'test set.csv').
"""
import csv, json, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEV = "cuda"
TOPK = 5
BATCH = 128
MAXLEN = 256   # dictionary entries + questions are short

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


def find(part, ext):
    for f in Path(".").iterdir():
        if part in f.name.lower() and f.suffix == ext:
            return f
    raise FileNotFoundError(part)


def main():
    wk = [json.loads(l) for l in open("wikt_passages.jsonl", encoding="utf-8")]
    titles = [w["t"] for w in wk]
    texts = [w["x"] for w in wk]          # "headword: meaning"
    print(f"wikt entries: {len(wk)}", flush=True)

    if os.path.exists("wikt_emb.npy"):
        C = np.load("wikt_emb.npy")
    else:
        C = embed(texts, "corpus"); np.save("wikt_emb.npy", C)
    print(f"wikt emb: {C.shape}", flush=True)
    Cf = C.astype(np.float32)

    samples = json.load(open(find("samples", ".json"), encoding="utf-8"))
    test = list(csv.DictReader(open(find("test", ".csv"), encoding="utf-8")))
    queries = [(f"s{i}", s["prompt_bn"]) for i, s in enumerate(samples) if not hasctx(s)]
    queries += [(f"t{r['id']}", r["prompt_bn"]) for r in test if not hasctx(r)]
    print(f"no-context queries: {len(queries)}", flush=True)

    Q = embed([q for _, q in queries], "queries").astype(np.float32)
    ev = {}
    for bi in range(0, len(queries), 256):
        sims = Q[bi:bi + 256] @ Cf.T
        idx = np.argpartition(-sims, TOPK, axis=1)[:, :TOPK]
        for j, (key, _) in enumerate(queries[bi:bi + 256]):
            order = idx[j][np.argsort(-sims[j][idx[j]])]
            ev[key] = [{"title": titles[t], "text": texts[t],
                        "score": float(sims[j][t])} for t in order]
    json.dump(ev, open("wikt_evidence.json", "w"), ensure_ascii=False)
    print(f"wrote wikt_evidence.json ({len(ev)} queries)", flush=True)


if __name__ == "__main__":
    main()
