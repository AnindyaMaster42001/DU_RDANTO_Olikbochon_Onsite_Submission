"""Approach_1 · Step 3 — retrieval-augmented evidence for the no-context branch.

THE C1 LEVER. No-context rows are closed-book factuality questions; the judge
LLM hallucinates on exactly the Bangladesh-specific (C1) facts we're graded on.
So don't ask the model to recall — retrieve evidence and let the grounding
stack (nli_grounding / judge) verify against it.

Pipeline per no-context row:
  1. query = prompt_bn (+ named entities lifted from response_bn)
  2. hybrid retrieval over a Bengali Wikipedia dump:
        BM25 (lexical — great for entities, dates, spellings)
      + dense e5 (semantic — paraphrase, synonymy)
      fuse scores (reciprocal-rank fusion) -> top-k passages
  3. write retrieved_evidence.json {row_id: [passage, ...]} so the same NLI/judge
     stack used on context rows can run on the retrieved evidence.

SETUP (once, offline-legal for Phase 2):
  - Attach a Bengali Wikipedia dump as a Kaggle dataset, e.g. HF
    `wikimedia/wikipedia` config `20231101.bn`. Point WIKI_DIR at it.
  - Build & cache the BM25 + dense indices as a Kaggle dataset so the Phase-2
    kernel loads them instead of rebuilding (keeps well under the 9h budget).

This is a SCAFFOLD: the corpus loader and index build are marked TODO. The
retrieval + fusion logic is real.

Usage:  python3 retrieval.py        # needs rank_bm25, sentence-transformers
"""

from pathlib import Path

import common as C

WIKI_DIR = Path("/kaggle/input/bengali-wikipedia")   # TODO: point at the dump
EMBED_MODEL = "intfloat/multilingual-e5-large"        # dense retriever
TOP_K = 5
PASSAGE_CHARS = 600

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAVE_RETRIEVAL = True
except ImportError:
    HAVE_RETRIEVAL = False


# ---------------------------------------------------------------- corpus
def load_corpus():
    """Return list[str] of Wikipedia passages (~PASSAGE_CHARS each).

    TODO: read the attached dump, strip markup, and split articles into
    passage-sized chunks. Cache the result so this runs once.
    """
    if not WIKI_DIR.exists():
        raise FileNotFoundError(
            f"Bengali Wikipedia not found at {WIKI_DIR}. Attach the dump as a "
            "Kaggle dataset (e.g. wikimedia/wikipedia 20231101.bn) and set WIKI_DIR."
        )
    raise NotImplementedError("load_corpus: parse + chunk the wiki dump here")


def bn_tokenize(text):
    # BM25 needs tokens; whitespace + punctuation-strip is a fine Bengali start.
    return [t for t in C._PUNCT.split(str(text)) if t]


# ---------------------------------------------------------------- index
class HybridIndex:
    def __init__(self, passages):
        self.passages = passages
        self.bm25 = BM25Okapi([bn_tokenize(p) for p in passages])
        self.embedder = SentenceTransformer(EMBED_MODEL)
        # e5 wants a "passage: " prefix; queries get "query: "
        self.emb = self.embedder.encode(
            [f"passage: {p}" for p in passages],
            normalize_embeddings=True, show_progress_bar=True,
        )

    def search(self, query, k=TOP_K):
        toks = bn_tokenize(query)
        bm = self.bm25.get_scores(toks)
        qv = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)[0]
        dense = self.emb @ qv
        # reciprocal-rank fusion of the two rankings
        rank = {}
        for scores in (bm, dense):
            order = np.argsort(scores)[::-1]
            for r, idx in enumerate(order[: 50]):
                rank[idx] = rank.get(idx, 0.0) + 1.0 / (60 + r)
        top = sorted(rank, key=rank.get, reverse=True)[:k]
        return [self.passages[i][:PASSAGE_CHARS] for i in top]


# ---------------------------------------------------------------- driver
def entities(response):
    # cheap query expansion: keep the longer tokens from the response
    return " ".join(t for t in bn_tokenize(response) if len(t) >= 3)


def main():
    if not HAVE_RETRIEVAL:
        print("!! install rank_bm25 + sentence-transformers (see requirements.txt)")
        return
    import json
    index = HybridIndex(load_corpus())

    out = {}
    for split_name, rows in [("samples", C.load_samples()), ("test", C.load_test())]:
        for i, r in enumerate(rows):
            if C.has_context(r):
                continue
            key = f"{split_name}:{r.get('id', i)}"
            q = f"{r['prompt_bn']} {entities(r['response_bn'])}"
            out[key] = index.search(q)
        print(f"retrieved evidence for {split_name}")

    with open(C.OUT_DIR / "retrieved_evidence.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("wrote retrieved_evidence.json  -> feed these into nli_grounding / judge")


if __name__ == "__main__":
    main()
