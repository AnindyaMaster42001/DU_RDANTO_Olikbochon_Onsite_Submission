# Approach_1 — retrieval rebuild: high-recall bnwiki evidence for no-context rows.
#
# The 23 unanimously-missed sample rows are all FAITHFUL: nothing finds the
# evidence proving true-but-obscure answers true. This kernel maximizes
# evidence recall, upgrading the v1 retrieval (top-5, question-only, first 3
# chunks/article, no reranker):
#   - fresh bnwiki from HF wikimedia/wikipedia (latest .bn config)
#   - up to 6 chunks per article (~900 chars each)
#   - dual queries: question alone AND question+response, union of top-10 each
#   - bge-reranker-v2-m3 cross-encoder -> keep top-5 with scores
# Output: retrieved_evidence.json {"s<idx>"|"t<id>": [{title, text, score}]}
# (the 32B grounding pass runs in a separate kernel on this output).
#
# Kaggle T4 x2; embedding ~500k passages is the long pole (~2-3 h).

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "datasets", "sentence-transformers"],
    check=True,
)

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import get_dataset_config_names, load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer

CHUNK_CHARS = 900
MAX_CHUNKS = 6
TOPK_DENSE = 10
TOPK_FINAL = 5
KAGGLE_INPUT = Path("/kaggle/input")


def find(part):
    for root in (KAGGLE_INPUT, Path(".")):
        if root.exists():
            hits = [p for p in root.rglob("*") if p.is_file() and part in p.name.lower()]
            if hits:
                return hits[0]
    raise FileNotFoundError(part)


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


samples = json.load(open(find("samples.json"), encoding="utf-8"))
test_rows = list(csv.DictReader(open(find("test set"), encoding="utf-8")))

queries = []  # (key, question, response)
for i, s in enumerate(samples):
    if not has_context(s):
        queries.append((f"s{i}", str(s["prompt_bn"]), str(s["response_bn"])))
for r in test_rows:
    if not has_context(r):
        queries.append((f"t{r['id']}", str(r["prompt_bn"]), str(r["response_bn"])))
print(f"no-context queries: {len(queries)}")

# ---------------------------------------------------------------- corpus
cfgs = [c for c in get_dataset_config_names("wikimedia/wikipedia") if c.endswith(".bn")]
cfg = sorted(cfgs)[-1]
print(f"bnwiki config: {cfg}")
wiki = load_dataset("wikimedia/wikipedia", cfg, split="train")
print(f"articles: {len(wiki)}")

_SENT = re.compile(r"(?<=[।!?\n])")


def chunk(text):
    parts = [p for p in _SENT.split(text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) > CHUNK_CHARS and cur:
            chunks.append(cur)
            if len(chunks) >= MAX_CHUNKS:
                return chunks
            cur = ""
        cur += p
    if cur and len(chunks) < MAX_CHUNKS:
        chunks.append(cur)
    return chunks


passages, titles = [], []
for art in wiki:
    for ch in chunk(art["text"]):
        passages.append(ch)
        titles.append(art["title"])
print(f"passages: {len(passages)}")

# ---------------------------------------------------------------- embed
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
model.max_seq_length = 256
model.half()

emb = model.encode(
    passages,
    batch_size=384,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype(np.float16)
np.save("passage_embeddings.npy", emb)
print("embeddings:", emb.shape)

q_texts = [q for _, q, _ in queries] + [f"{q} {a}" for _, q, a in queries]
q_emb = model.encode(
    q_texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True
).astype(np.float16)

# ---------------------------------------------------------------- dense top-k
E = torch.tensor(emb, device="cuda")
Q = torch.tensor(q_emb, device="cuda")
nq = len(queries)
cand = [set() for _ in range(nq)]
BS = 256
for b in range(0, len(Q), BS):
    sims = Q[b:b + BS] @ E.T
    top = sims.topk(TOPK_DENSE, dim=1).indices.cpu().numpy()
    for j, idxs in enumerate(top):
        cand[(b + j) % nq].update(int(x) for x in idxs)
del E, Q
torch.cuda.empty_cache()

# ---------------------------------------------------------------- rerank
rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512)
evidence = {}
pairs, owner = [], []
for qi, (key, q, a) in enumerate(queries):
    for pid in sorted(cand[qi]):
        pairs.append((f"{q} উত্তর: {a}", passages[pid]))
        owner.append((qi, pid))
scores = rr.predict(pairs, batch_size=128, show_progress_bar=True)

by_q = {}
for (qi, pid), sc in zip(owner, scores):
    by_q.setdefault(qi, []).append((float(sc), pid))
for qi, (key, q, a) in enumerate(queries):
    ranked = sorted(by_q.get(qi, []), reverse=True)[:TOPK_FINAL]
    evidence[key] = [
        {"title": titles[pid], "text": passages[pid], "score": sc}
        for sc, pid in ranked
    ]

with open("retrieved_evidence.json", "w", encoding="utf-8") as f:
    json.dump(evidence, f, ensure_ascii=False)
print(f"wrote retrieved_evidence.json ({len(evidence)} queries)")
