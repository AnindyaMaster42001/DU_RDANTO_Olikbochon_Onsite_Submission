# Approach_1 — cross-lingual English-Wikipedia grounding for factual no-context rows.
#
# ret32/retrieval read only Bengali Wikipedia; much of the factual error mass is
# WORLD knowledge (smallest country, science, world history) that English
# Wikipedia covers far better and that the 32B reasons over more strongly in
# English. Pipeline (internet ON):
#   A) 32B turns each Bengali question into a short English Wikipedia search query.
#   B) live enwiki API: opensearch -> top titles -> intro extracts (evidence).
#   C) 32B grounds the candidate answer against the English evidence, three-way
#      UNSURE-safe, logprob-soft. Abstains (0.5) off-bucket / no evidence.
# Targets non-dictionary no-context rows only. Writes signal_enwiki.json.

import csv, json, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
import math, torch
from vllm import LLM, SamplingParams

KIN = Path("/kaggle/input")
OUT = Path("/kaggle/working")  # persisted across a kernel restart
CKPT = OUT / "enwiki_evidence_ckpt.json"  # resume retrieval if the session dies
def find(part, ext=None):
    for root in (KIN, Path(".")):
        if not root.exists(): continue
        h=[p for p in root.rglob("*") if p.is_file() and part in p.name.lower()
           and (ext is None or p.suffix.lower()==ext)]
        if h: return sorted(h,key=lambda p:len(str(p)))[0]
    raise FileNotFoundError(part)
def has_context(r): return str(r["context"]).strip() not in ("[NULL]","NULL","null","")

samples=json.load(open(find("samples",".json"),encoding="utf-8"))
test_rows=list(csv.DictReader(open(find("test set",".csv"),encoding="utf-8")))

# non-dictionary no-context rows (dictionary is handled by wikt)
DICT=re.compile("ভাবার্থ|শাব্দিক অর্থ|অর্থ\\s*ক[িী]|প্রতিশব্দ|সমার্থ|বিপরীত")
def target(r): return (not has_context(r)) and (not DICT.search(str(r["prompt_bn"])))
s_idx=[i for i,r in enumerate(samples) if target(r)]
t_idx=[i for i,r in enumerate(test_rows) if target(r)]
print(f"target rows: samples={len(s_idx)} test={len(t_idx)}")

# Preflight: fail loudly NOW if the enwiki API is unreachable, before the ~16-min
# model load. A dead/blocked network here (429, no internet) is the failure that
# silently produced an all-0.5 signal last time.
def _preflight():
    ua={"User-Agent":"BengaliHallucResearch/1.0 "
        "(https://github.com/ninadgns/bengali-hallucination-detection; research)"}
    for a in range(4):
        try:
            req=urllib.request.Request(
                "https://en.wikipedia.org/w/api.php?action=query&list=search"
                "&format=json&srlimit=1&srsearch=Padma%20River", headers=ua)
            d=json.load(urllib.request.urlopen(req,timeout=20))
            if d.get("query",{}).get("search"):
                print("preflight OK: enwiki API reachable"); return
        except Exception as e:
            print(f"preflight attempt {a}: {type(e).__name__}: {e}"); time.sleep(2**a)
    raise SystemExit("PREFLIGHT FAILED: enwiki API unreachable — aborting before model load")
_preflight()

MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
llm=LLM(model=MODEL,dtype="half",max_model_len=6144,
        tensor_parallel_size=torch.cuda.device_count(),gpu_memory_utilization=0.92)

# ---- Phase A: Bengali question -> English search query ----
def qprompt(row):
    return ("Translate the key topic of this Bengali question into a concise English "
            "Wikipedia search query (2-6 words, just the entity/topic, no answer).\n"
            f"Question: {row['prompt_bn']}\nEnglish search query:")
QP=SamplingParams(temperature=0,max_tokens=24)
def gen_queries(rows,idx):
    outs=llm.chat([[{"role":"user","content":qprompt(rows[i])}] for i in idx],QP)
    q={}
    for i,o in zip(idx,outs):
        t=o.outputs[0].text.strip().split("\n")[0][:80]
        q[i]=re.sub(r'^["\']|["\']$','',t)
    return q
sq=gen_queries(samples,s_idx); tq=gen_queries(test_rows,t_idx)
print("sample queries:", list(sq.items())[:3])

# ---- Phase B: live English Wikipedia retrieval ----
# Wikimedia 429-throttles bursts hard; a policy-compliant UA (contact URL) plus
# exponential backoff that honors Retry-After is required, and concurrency must
# stay low (this is what killed the earlier 16-worker run: 0/1133, all 429).
import random
UA={"User-Agent":"BengaliHallucResearch/1.0 "
    "(https://github.com/ninadgns/bengali-hallucination-detection; research)"}
def api(url):
    for a in range(5):
        try:
            req=urllib.request.Request(url,headers=UA)
            return json.load(urllib.request.urlopen(req,timeout=20))
        except urllib.error.HTTPError as e:
            if e.code==429:
                ra=e.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else (2**a)*0.5+random.random())
            else:
                time.sleep(2**a*0.3)
        except Exception:
            time.sleep(2**a*0.3)
    return None
def wiki_evidence(query, n_titles=2, chars=700):
    if not query: return []
    time.sleep(random.uniform(0,0.5))  # jitter to spread the burst
    s=api("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=%d&srsearch=%s"
          %(n_titles, urllib.parse.quote(query)))
    if not s: return []
    titles=[h["title"] for h in s.get("query",{}).get("search",[])[:n_titles]]
    ev=[]
    for tt in titles:
        e=api("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&format=json&titles=%s"
              %urllib.parse.quote(tt))
        if not e: continue
        for pg in e.get("query",{}).get("pages",{}).values():
            ex=(pg.get("extract") or "").strip()
            if ex: ev.append({"title":tt,"text":ex[:chars]})
    return ev

# Concurrent + checkpointed retrieval over ALL target rows at once (GPU idle here,
# so parallelize hard). Keyed by "s{idx}"/"t{idx}"; resumes from CKPT if the
# session was interrupted mid-retrieval (the failure mode that killed the last run).
def build_ev_all(jobs):
    # jobs: list of (key, query)
    done={}
    if CKPT.exists():
        try: done=json.load(open(CKPT)); print(f"resuming: {len(done)} cached")
        except Exception: done={}
    todo=[(k,q) for k,q in jobs if k not in done]
    print(f"retrieving {len(todo)} (of {len(jobs)}) with 4 workers (429-safe)")
    def work(job):
        k,q=job; return k, wiki_evidence(q)
    n=0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k,ev in ex.map(work, todo):
            done[k]=ev; n+=1
            if n%100==0:
                json.dump(done, open(CKPT,"w"))
                got=sum(1 for v in done.values() if v)
                print(f"  {n}/{len(todo)}  with-ev {got}")
    json.dump(done, open(CKPT,"w"))
    return done
jobs=[(f"s{i}",sq[i]) for i in s_idx]+[(f"t{i}",tq[i]) for i in t_idx]
EV=build_ev_all(jobs)
s_ev={i:EV.get(f"s{i}",[]) for i in s_idx}
t_ev={i:EV.get(f"t{i}",[]) for i in t_idx}
print(f"evidence: samples with-ev {sum(1 for i in s_idx if s_ev[i])}/{len(s_idx)}"
      f"  test with-ev {sum(1 for i in t_idx if t_ev[i])}/{len(t_idx)}")

# ---- Phase C: ground answer vs English evidence ----
def gprompt(row,ev):
    blocks="\n\n".join(f"[{k+1}] {p['title']}: {p['text']}" for k,p in enumerate(ev[:3]))
    return ("You verify a Bengali question's answer using English Wikipedia evidence "
            "(may be irrelevant).\n\n"
            f"{blocks}\n\nQuestion (Bengali): {row['prompt_bn']}\n"
            f"Candidate answer (Bengali): {row['response_bn']}\n\n"
            "Reply YES if the evidence supports the candidate answer, NO if it "
            "contradicts it or shows a different answer, UNSURE if the evidence is "
            "irrelevant/insufficient. Never say NO just because the evidence omits it. "
            "One word.")
GP=SamplingParams(temperature=0,max_tokens=1,logprobs=20)
def soft(prompts):
    outs=llm.chat([[{"role":"user","content":p}] for p in prompts],GP); vals=[]
    for o in outs:
        pr={"Y":0.,"N":0.,"U":0.}; lps=o.outputs[0].logprobs
        for tok in (lps[0].values() if lps else []):
            t=(tok.decoded_token or "").strip().upper()
            if t and t[0] in pr: pr[t[0]]+=math.exp(tok.logprob)
        tot=sum(pr.values()); vals.append(0.5 if tot<0.05 else (pr["Y"]+0.5*pr["U"])/tot)
    return vals
def run(rows,idx,ev):
    vals=[0.5]*len(rows)
    use=[i for i in idx if ev.get(i)]
    print(f"grounding {len(use)}/{len(idx)} (rest no-evidence abstain)")
    for i,v in zip(use, soft([gprompt(rows[i],ev[i]) for i in use])): vals[i]=v
    return vals
s_vals=run(samples,s_idx,s_ev); t_vals=run(test_rows,t_idx,t_ev)
json.dump({"name":"enwiki","samples":s_vals,
           "test":{r["id"]:v for r,v in zip(test_rows,t_vals)}},
          open(OUT/"signal_enwiki.json","w"))
print("wrote signal_enwiki.json")
