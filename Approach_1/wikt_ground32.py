"""32B grounds answers against retrieved DICTIONARY entries -> signal_ret32wikt.json.

Same UNSURE-safe soft-verdict logic as kaggle_ground32, adapted for wiktionary
(word/idiom meaning) evidence. VPS-safe config: util 0.21 (~2.5 GB card buffer),
gptq_marlin, enforce_eager, short context. Run from ~/claude_bengali.
"""
import os, json, csv, math, re, time
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
from pathlib import Path
from vllm import LLM, SamplingParams   # no main-process CUDA (0-buffer card)

# Wiktionary is for word/idiom-meaning questions. Ground only those (+ any query
# whose top dictionary hit is strong), leave everything else neutral at 0.5.
# Keeps the run short (~190 queries) — important given the tight VRAM buffer.
WIKT_PAT = re.compile(r"(ভাবার্থ|শব্দের অর্থ|এর অর্থ|অর্থ কী|শাব্দিক অর্থ|মানে কী|"
                      r"মানে কি|প্রতিশব্দ|সমার্থক|বিপরীত|প্রবাদ|বাগধারা)")
SCORE_GATE = 0.60


def relevant(row, ev):
    if WIKT_PAT.search(str(row["prompt_bn"])):
        return True
    return bool(ev) and ev[0].get("score", 0) >= SCORE_GATE

MODEL = "/home/mb2/Qwen2.5-32B-Instruct-GPTQ-Int4"
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.21"))
MAXLEN = int(os.environ.get("MAXLEN", "1536"))
EV_TOP = 4
EV_CHARS = 120   # dictionary meanings are short; keep prompt well under MAXLEN


def find(part, ext):
    for f in Path(".").iterdir():
        if part in f.name.lower() and f.suffix == ext:
            return f
    raise FileNotFoundError(part)


def hasctx(r):
    return str(r["context"]).strip() not in ("[NULL]", "NULL", "null", "")


samples = json.load(open(find("samples", ".json"), encoding="utf-8"))
test = list(csv.DictReader(open(find("test", ".csv"), encoding="utf-8")))
evidence = json.load(open("wikt_evidence.json", encoding="utf-8"))
print(f"samples {len(samples)} test {len(test)} evidence {len(evidence)}", flush=True)


def ground_prompt(row, ev):
    q, a = str(row["prompt_bn"])[:400], str(row["response_bn"])[:200]
    blocks = "\n".join(
        f"[অভিধান {k+1}] {p['title']} — {p['text'][:EV_CHARS]}"
        for k, p in enumerate(ev[:EV_TOP])
    )
    return (
        "You are verifying an answer to a Bengali word/idiom-meaning question "
        "using retrieved DICTIONARY entries (headword — definition). The entries "
        "may or may not be relevant.\n\n"
        f"{blocks}\n\n"
        f"Question: {q}\nCandidate answer: {a}\n\n"
        "Based ONLY on the dictionary entries above: reply YES if an entry's "
        "definition supports the candidate answer's meaning, NO if an entry gives "
        "a clearly different or contradictory meaning for the SAME word/idiom, or "
        "UNSURE if no entry addresses the queried word/idiom. Never answer NO "
        "merely because the entries do not mention it. Reply with exactly one word."
    )


t0 = time.time()
llm = LLM(model=MODEL, quantization="gptq_marlin", dtype="float16",
          gpu_memory_utilization=GPU_UTIL, max_model_len=MAXLEN,
          max_num_seqs=16, enforce_eager=True, tensor_parallel_size=1,
          disable_log_stats=True)
print(f"[load] {time.time()-t0:.0f}s", flush=True)
VERDICT = SamplingParams(temperature=0, max_tokens=1, logprobs=20)


def soft_verdicts(prompts):
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], VERDICT)
    vals = []
    for o in outs:
        p = {"Y": 0.0, "N": 0.0, "U": 0.0}
        lps = o.outputs[0].logprobs
        for tok in (lps[0].values() if lps else []):
            t = (tok.decoded_token or "").strip().upper()
            if t and t[0] in p:
                p[t[0]] += math.exp(tok.logprob)
        tot = sum(p.values())
        vals.append(0.5 if tot < 0.05 else (p["Y"] + 0.5 * p["U"]) / tot)
    return vals


def run_split(rows, keyfn, tag):
    vals = [0.5] * len(rows)
    idx = [i for i, r in enumerate(rows)
           if not hasctx(r) and evidence.get(keyfn(i, r))
           and relevant(r, evidence[keyfn(i, r)])]
    print(f"[{tag}] grounding {len(idx)} rows...", flush=True)
    tg = time.time()
    outs = soft_verdicts([ground_prompt(rows[i], evidence[keyfn(i, rows[i])]) for i in idx])
    for i, v in zip(idx, outs):
        vals[i] = v
    print(f"[{tag}] done in {time.time()-tg:.0f}s", flush=True)
    return vals


s_vals = run_split(samples, lambda i, r: f"s{i}", "samples")
t_vals = run_split(test, lambda i, r: f"t{r['id']}", "test")

json.dump({"name": "ret32wikt", "samples": s_vals,
           "test": {r["id"]: v for r, v in zip(test, t_vals)}},
          open("signal_ret32wikt.json", "w"))
print("wrote signal_ret32wikt.json", flush=True)
