"""Chunk wikiextractor JSON output -> passages.jsonl (CPU, stdlib only).

Reads extracted/**/wiki_* (one JSON per line: {id,url,title,text}) and writes
passages.jsonl with {"t": title, "x": passage}. Runs in the vllm env (no extra deps).
"""
import glob
import json

MAX_CHUNK = 1100
MAX_CHUNKS_PER_ARTICLE = 3
MIN_PASSAGE = 120

files = glob.glob("/home/mb2/bnwiki_raw/extracted/**/wiki_*", recursive=True)
print(f"wiki_* files: {len(files)}", flush=True)
n = 0
with open("passages.jsonl", "w", encoding="utf-8") as out:
    for fi, f in enumerate(files):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except Exception:
                continue
            title = a.get("title", "")
            text = a.get("text", "")
            if not text:
                continue
            paras = [p.strip() for p in text.split("\n") if p.strip()]
            # wikiextractor repeats the title as the first line -> drop it
            if paras and paras[0] == title:
                paras = paras[1:]
            chunks, cur = [], ""
            for p in paras:
                if len(cur) + len(p) > MAX_CHUNK and cur:
                    chunks.append(cur)
                    cur = ""
                    if len(chunks) >= MAX_CHUNKS_PER_ARTICLE:
                        break
                cur += (" " if cur else "") + p
            if cur and len(chunks) < MAX_CHUNKS_PER_ARTICLE:
                chunks.append(cur)
            for c in chunks:
                if len(c) >= MIN_PASSAGE:
                    out.write(json.dumps({"t": title, "x": c}, ensure_ascii=False) + "\n")
                    n += 1
        if fi % 20 == 0:
            print(f"  {fi}/{len(files)} files, {n} passages", flush=True)
print("wrote", n, "passages")
