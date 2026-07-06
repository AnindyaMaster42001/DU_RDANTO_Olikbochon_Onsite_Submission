# Approach_1 — wiktionary-grounded signal for dictionary-lookup rows.
#
# Targets the ~224 test rows asking a word/idiom meaning
# ("X এর ভাবার্থ / শাব্দিক অর্থ / অর্থ কী?"). The submitted ensemble is
# over-skeptical here (calls only 38/224 faithful; true rate ~41%), wrongly
# killing true glosses. Wikipedia is the wrong book; bn.wiktionary is the right
# one. Pipeline: bge-m3 dense-retrieve the headword's glosses from a 70k-entry
# bn.wiktionary corpus, then Qwen-32B judges the candidate answer against those
# glosses (three-way YES/NO/UNSURE, logprob-soft). Abstains (0.5) off-bucket or
# when no gloss is retrieved, so it cannot hurt other rows.
#
# Phase A (image-native torch): retrieve -> wikt_evidence.json
# Phase B (vllm torch): 32B grounding -> signal_wikt.json
# Split to avoid the vllm-wheel CUDA upgrade breaking bge-m3 (libnvrtc lesson).

import csv, json, re, subprocess, sys
from pathlib import Path

KIN = Path("/kaggle/input")

def find(part, ext=None):
    for root in (KIN, Path(".")):
        if not root.exists():
            continue
        hits = [p for p in root.rglob("*") if p.is_file() and part in p.name.lower()
                and (ext is None or p.suffix.lower() == ext)]
        if hits:
            return sorted(hits, key=lambda p: len(str(p)))[0]
    raise FileNotFoundError(part)

def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")

LOOKUP = re.compile("ভাবার্থ|শাব্দিক অর্থ|অর্থ\\s*ক[িী]")
def is_lookup(row):
    return bool(LOOKUP.search(str(row["prompt_bn"])))

QUOTE = re.compile(r'["“‘\'](.+?)["”’\']')
def headword(prompt):
    m = QUOTE.search(prompt)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\S+)\s+এর\s", prompt)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\S+)\s+শব্দের", prompt)
    return m.group(1).strip() if m else prompt.strip()

def norm(s):
    return re.sub(r"[\s।,.;:!?\"'‘’“”()\[\]{}\-–—`~*_/\\]+", "", str(s))

# ------------------------------------------------------------------ load data
samples = json.load(open(find("samples", ".json"), encoding="utf-8"))
test_rows = list(csv.DictReader(open(find("test set", ".csv"), encoding="utf-8")))
corpus = json.load(open(find("wikt_corpus", ".json"), encoding="utf-8"))["entries"]
print(f"samples={len(samples)} test={len(test_rows)} wikt_entries={len(corpus)}")

# rows we act on: no-context lookup rows
def bucket_idx(rows):
    return [i for i, r in enumerate(rows) if is_lookup(r) and not has_context(r)]
s_idx, t_idx = bucket_idx(samples), bucket_idx(test_rows)
print(f"bucket: samples={len(s_idx)} test={len(t_idx)}")

# ============================ PHASE A: retrieval ============================
EV_TOP = 4
hw_list = list(corpus.keys())
hw_norm = {norm(h): h for h in hw_list}
passages = [f"{h}: {'; '.join(corpus[h][:6])}" for h in hw_list]

from sentence_transformers import SentenceTransformer
import numpy as np
enc = SentenceTransformer(find("bge-m3").parent.as_posix() if False else "BAAI/bge-m3")
print("embedding corpus...")
P = enc.encode(passages, batch_size=256, normalize_embeddings=True,
               show_progress_bar=True, convert_to_numpy=True)

def retrieve_for(rows, idx):
    ev = {}
    queries, keys = [], []
    for i in idx:
        hw = headword(str(rows[i]["prompt_bn"]))
        # exact/normalized headword hit goes first (highest precision)
        exact = hw_norm.get(norm(hw))
        queries.append(hw)
        keys.append((i, hw, exact))
    Q = enc.encode(queries, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    sims = Q @ P.T
    for (i, hw, exact), row_sim in zip(keys, sims):
        top = np.argsort(-row_sim)[:EV_TOP + 2]
        picks, seen = [], set()
        if exact is not None:
            picks.append({"hw": exact, "gloss": corpus[exact][:6], "sim": 1.0})
            seen.add(exact)
        for j in top:
            h = hw_list[j]
            if h in seen:
                continue
            picks.append({"hw": h, "gloss": corpus[h][:6], "sim": float(row_sim[j])})
            seen.add(h)
            if len(picks) >= EV_TOP:
                break
        ev[str(i)] = {"query": hw, "picks": picks}
    return ev

wikt_ev = {"samples": retrieve_for(samples, s_idx),
           "test": retrieve_for(test_rows, t_idx)}
json.dump(wikt_ev, open("wikt_evidence.json", "w"), ensure_ascii=False)
print("wrote wikt_evidence.json")
del enc, P
import gc, torch
gc.collect(); torch.cuda.empty_cache()

# ============================ PHASE B: 32B grounding ============================
PHASE_B = r'''
import csv, json, math, re
from pathlib import Path
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
MAX_LEN = 4096

def find(part, ext=None):
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists(): continue
        hits=[p for p in root.rglob("*") if p.is_file() and part in p.name.lower()
              and (ext is None or p.suffix.lower()==ext)]
        if hits: return sorted(hits,key=lambda p:len(str(p)))[0]
    raise FileNotFoundError(part)

samples=json.load(open(find("samples",".json"),encoding="utf-8"))
test_rows=list(csv.DictReader(open(find("test set",".csv"),encoding="utf-8")))
ev=json.load(open("wikt_evidence.json",encoding="utf-8"))

def gloss_block(picks):
    out=[]
    for k,p in enumerate(picks):
        g="; ".join(p["gloss"][:4])
        out.append(f"[{k+1}] {p['hw']} — {g}")
    return "\n".join(out)

def prompt(row, picks):
    q,a=str(row["prompt_bn"]),str(row["response_bn"])
    return (
      "You are checking whether a candidate answer correctly gives the meaning "
      "of a Bengali word or idiom, using dictionary entries below. The entries "
      "may or may not be relevant.\n\n"
      f"Dictionary entries:\n{gloss_block(picks)}\n\n"
      f"Question: {q}\nCandidate answer: {a}\n\n"
      "Reply YES if the candidate answer matches (is the same meaning as, or a "
      "close paraphrase of) the correct dictionary sense. Reply NO if it gives a "
      "different or wrong meaning. Reply UNSURE only if the entries do not cover "
      "this word/idiom at all. One word only."
    )

llm=LLM(model=MODEL,dtype="half",max_model_len=MAX_LEN,
        tensor_parallel_size=torch.cuda.device_count(),gpu_memory_utilization=0.92)
VP=SamplingParams(temperature=0,max_tokens=1,logprobs=20)

def soft(prompts):
    outs=llm.chat([[{"role":"user","content":p}] for p in prompts],VP)
    vals=[]
    for o in outs:
        pr={"Y":0.0,"N":0.0,"U":0.0}
        lps=o.outputs[0].logprobs
        for tok in (lps[0].values() if lps else []):
            t=(tok.decoded_token or "").strip().upper()
            if t and t[0] in pr: pr[t[0]]+=math.exp(tok.logprob)
        tot=sum(pr.values())
        vals.append(0.5 if tot<0.05 else (pr["Y"]+0.5*pr["U"])/tot)
    return vals

def run(rows, split):
    vals=[0.5]*len(rows)
    keys=list(ev[split].keys())
    prompts=[prompt(rows[int(k)], ev[split][k]["picks"]) for k in keys]
    print(f"[{split}] grounding {len(prompts)} rows")
    for k,v in zip(keys, soft(prompts)):
        vals[int(k)]=v
    return vals

s_vals=run(samples,"samples")
t_vals=run(test_rows,"test")
json.dump({"name":"wikt","samples":s_vals,
           "test":{r["id"]:v for r,v in zip(test_rows,t_vals)}},
          open("signal_wikt.json","w"))
print("wrote signal_wikt.json")
'''
Path("phase_b.py").write_text(PHASE_B)
print("installing vllm...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
subprocess.run([sys.executable, "phase_b.py"], check=True)
print("DONE")
