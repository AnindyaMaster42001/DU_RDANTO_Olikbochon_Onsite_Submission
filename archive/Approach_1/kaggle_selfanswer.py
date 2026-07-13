# Approach_1 — self-answer + self-consistency judge for the no-context branch.
#
# Every existing signal asks the 32B "is this answer right?" as a one-token
# verdict — no reasoning, and it anchors on the given answer. The no-context
# error mass is factual world-knowledge + reasoning/negation MCQs
# ("কোনটি নয়", analogies), where chain-of-thought + self-consistency is the
# known step change. Here the model INDEPENDENTLY derives the answer with CoT,
# sampled K times at temperature, then each chain votes whether the candidate
# matches. P(faithful) = mean YES vote. Reference-free self-answer beats
# anchored verification on exactly these hard rows. Writes signal_sa.json.

import csv, json, re, subprocess, sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
import torch
from vllm import LLM, SamplingParams

KIN = Path("/kaggle/input")
def find(part, ext=None):
    for root in (KIN, Path(".")):
        if not root.exists(): continue
        h=[p for p in root.rglob("*") if p.is_file() and part in p.name.lower()
           and (ext is None or p.suffix.lower()==ext)]
        if h: return sorted(h,key=lambda p:len(str(p)))[0]
    raise FileNotFoundError(part)

def has_context(r):
    return str(r["context"]).strip() not in ("[NULL]","NULL","null","")

samples=json.load(open(find("samples",".json"),encoding="utf-8"))
test_rows=list(csv.DictReader(open(find("test set",".csv"),encoding="utf-8")))
print(f"samples={len(samples)} test={len(test_rows)}")

MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
K=6  # self-consistency samples

def prompt(row):
    q,a=str(row["prompt_bn"]),str(row["response_bn"])
    return (
      "তুমি একজন বিশেষজ্ঞ। নিচের বাংলা প্রশ্নটি মনোযোগ দিয়ে পড়ো। প্রথমে প্রদত্ত "
      "উত্তরটি না দেখে নিজে ধাপে ধাপে চিন্তা করে প্রশ্নের সঠিক উত্তর বের করো "
      "(প্রশ্নে বিকল্প থাকলে সঠিক বিকল্পটি বাছো; 'নয় কোনটি'/'ভুল কোনটি' জাতীয় "
      "নেতিবাচক প্রশ্নে সাবধান হও)। তারপর প্রদত্ত উত্তরের সাথে তোমার উত্তর মেলাও।\n\n"
      f"প্রশ্ন: {q}\n\nপ্রদত্ত উত্তর: {a}\n\n"
      "সংক্ষেপে যুক্তি দাও, তারপর শেষ লাইনে ঠিক এই রূপে লেখো:\n"
      "VERDICT: YES  (যদি প্রদত্ত উত্তরটি সঠিক হয়)\n"
      "VERDICT: NO   (যদি প্রদত্ত উত্তরটি ভুল হয়)"
    )

llm=LLM(model=MODEL,dtype="half",max_model_len=4096,
        tensor_parallel_size=torch.cuda.device_count(),gpu_memory_utilization=0.92)
SP=SamplingParams(temperature=0.7,top_p=0.9,max_tokens=640,n=K)

VRE=re.compile(r"VERDICT:\s*(YES|NO)", re.I)
def score(rows, idx):
    vals=[0.5]*len(rows)
    msgs=[[{"role":"user","content":prompt(rows[i])}] for i in idx]
    outs=llm.chat(msgs,SP)
    for i,o in zip(idx,outs):
        yes=no=0
        for c in o.outputs:
            m=None
            for mm in VRE.finditer(c.text): m=mm  # last verdict wins
            if m:
                if m.group(1).upper()=="YES": yes+=1
                else: no+=1
        tot=yes+no
        vals[i]=0.5 if tot==0 else yes/tot
    return vals

s_idx=[i for i,r in enumerate(samples) if not has_context(r)]
t_idx=[i for i,r in enumerate(test_rows) if not has_context(r)]
print(f"no-context: samples={len(s_idx)} test={len(t_idx)}")
s_vals=score(samples,s_idx)
t_vals=score(test_rows,t_idx)
json.dump({"name":"sa","samples":s_vals,
           "test":{r["id"]:v for r,v in zip(test_rows,t_vals)}},
          open("signal_sa.json","w"))
print("wrote signal_sa.json")
