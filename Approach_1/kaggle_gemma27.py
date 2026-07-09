# Approach_1 — second-model judge: Gemma-2-27B-it (GPTQ-4bit) for stack diversity.
#
# Rationale: no bigger model fits 2xT4; the lever is error-decorrelation from a
# DIFFERENT pretraining corpus. Gemma-2-27B has the best documented Bengali
# knowledge among 2xT4-fitting models (BnMMLU ~53% vs Qwen ~50%). Soft logprob
# scoring (P(YES) vs P(NO)) — calibrated soft signals stacked better than hard
# verdicts historically (cf j32sv, sa). Judges all rows; stacker weights it.
# Output: signal_gemma27.json  (P(faithful) per row).
import os, subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","vllm"],check=True)
os.environ.setdefault("VLLM_ATTENTION_BACKEND","FLASHINFER")  # gemma-2 soft-cap; patched from load-test
import csv, json, math
from pathlib import Path
import torch
from vllm import LLM, SamplingParams

MODEL="ModelCloud/gemma-2-27b-it-gptq-4bit"
MAX_CTX_CHARS=3500
KIN=Path("/kaggle/input"); OUT=Path("/kaggle/working")

def find(part):
    for root in (KIN, Path(".")):
        if root.exists():
            h=[p for p in root.rglob("*") if p.is_file() and part in p.name.lower()]
            if h: return sorted(h,key=lambda p:len(str(p)))[0]
    raise FileNotFoundError(part)

samples=json.load(open(find("samples.json"),encoding="utf-8"))
test_rows=list(csv.DictReader(open(find("test set"),encoding="utf-8")))
print(f"samples: {len(samples)}  test rows: {len(test_rows)}")

def has_context(r): return str(r["context"]).strip() not in ("[NULL]","NULL","null","")
def judge_prompt(r):
    q=str(r["prompt_bn"]); a=str(r["response_bn"])
    if has_context(r):
        ctx=str(r["context"])[:MAX_CTX_CHARS]
        return ("You are verifying answers to Bengali questions against a source passage.\n\n"
                f"Passage:\n{ctx}\n\nQuestion: {q}\nCandidate answer: {a}\n\n"
                "Is the candidate answer correct AND supported by the passage? "
                "Reply with exactly one word: YES or NO.")
    return ("You are a careful fact-checker for Bengali question answering. "
            "Questions may involve Bengali grammar and literature, Bangladeshi "
            "history and culture, science, or mathematics.\n\n"
            f"Question: {q}\nCandidate answer: {a}\n\n"
            "Is the candidate answer factually correct? "
            "Reply with exactly one word: YES or NO.")

llm=LLM(model=MODEL, quantization="gptq", dtype="bfloat16", max_model_len=4096,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90, enforce_eager=True)
# soft: read logprobs over first token, softmax P(YES)/(P(YES)+P(NO)); abstain 0.5
SP=SamplingParams(temperature=0, max_tokens=1, logprobs=20)

def soft(rows, tag):
    outs=llm.chat([[{"role":"user","content":judge_prompt(r)}] for r in rows], SP)
    vals=[]
    for o in outs:
        lp=o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py=pn=0.0
        for tid,info in lp.items():
            tok=info.decoded_token.strip().upper() if hasattr(info,"decoded_token") else ""
            p=math.exp(info.logprob)
            if tok.startswith("YES") or tok=="Y": py+=p
            elif tok.startswith("NO") or tok=="N": pn+=p
        vals.append(py/(py+pn) if (py+pn)>0 else 0.5)
    print(f"[{tag}] {len(vals)} judged, mean P(faithful) {sum(vals)/len(vals):.3f}, "
          f"abstain {sum(1 for v in vals if v==0.5)}")
    return vals

s_vals=soft(samples,"samples")
t_vals=soft(test_rows,"test")
json.dump({"name":"gemma27","samples":s_vals,
           "test":{r["id"]:v for r,v in zip(test_rows,t_vals)}},
          open(OUT/"signal_gemma27.json","w"))
print("wrote signal_gemma27.json")
