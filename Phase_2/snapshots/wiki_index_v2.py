# Build the bnwiki retrieval index as a pinned artifact (GPU, ~2.5 h).
# Same corpus/chunking as Approach_1/kaggle_wiki_retrieve.py, but saves BOTH
# halves of the index so the Phase-2 notebook can retrieve for unseen queries:
#   passages.jsonl           one {"title", "text"} per line
#   passage_embeddings.npy   float16 [n_passages, 1024], bge-m3, normalized

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "datasets", "sentence-transformers"],
    check=True,
)

import json
import re

import numpy as np
from datasets import get_dataset_config_names, load_dataset
from sentence_transformers import SentenceTransformer

CHUNK_CHARS = 900
MAX_CHUNKS = 6

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


passages = []
with open("/kaggle/working/passages.jsonl", "w", encoding="utf-8") as f:
    for art in wiki:
        for ch in chunk(art["text"]):
            passages.append(ch)
            f.write(json.dumps({"title": art["title"], "text": ch}, ensure_ascii=False) + "\n")
print(f"passages: {len(passages)}")

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
np.save("/kaggle/working/passage_embeddings.npy", emb)
print("embeddings:", emb.shape)
