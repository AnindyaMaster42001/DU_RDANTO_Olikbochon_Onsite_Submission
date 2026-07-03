# Phase 2 — single-notebook offline reproduction of team DU_RDANTO's system.
#
# Given the raw competition CSVs (or the organizers' held-out fold), this
# recomputes all 7 portable signals from scratch and produces submission.csv.
# No internet is used at any point: all models, the wiki index, and the vLLM
# wheels are attached as data sources (see Phase_2/README.md).
#
# Signals (portable set, honest 5-seed OOF 0.8311; LB-equivalent config 0.803):
#   substring  response found verbatim in context           (rule, ctx rows)
#   nli        mDeBERTa-xnli max-entailment over chunks     (ctx rows)
#   a0judge    Qwen2.5-14B-GPTQ grounding/factuality judge  (all rows)
#   a0selfv    14B self-verify: answer then compare         (no-ctx rows)
#   judge32    Qwen2.5-32B-GPTQ factuality judge            (all rows)
#   j32sv      32B self-verify, three-way logprob-soft      (no-ctx rows)
#   ret32      32B grounds answer vs reranked bnwiki evidence (no-ctx rows)
# Stitch: per-branch logistic regression refit on the 299 labeled samples;
# decision threshold = median over 5-seed x 5-fold CV.
#
# The two vLLM stages run as subprocesses so GPU memory is fully released
# between the 14B and 32B models. Expected total runtime on T4 x2: ~3-4 h.

import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------- config
TEST_CSV = ""  # organizers: set to the held-out fold CSV path; "" = auto-find
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path(".")
INPUT = Path("/kaggle/input")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

NLI_MAX_TOKENS, NLI_MAX_CHUNKS, NLI_BATCH = 384, 6, 32
TOPK_DENSE, TOPK_FINAL = 10, 4
SEEDS, FOLDS = range(5), 5


# ----------------------------------------------------------------- lookup
def find_file(pattern, root=INPUT):
    rx = re.compile(pattern, re.I)
    hits = [p for p in root.rglob("*") if p.is_file() and rx.search(p.name)]
    assert hits, f"no input file matching /{pattern}/ under {root}"
    return sorted(hits)[0]


def find_model_dir(hint):
    hits = [p.parent for p in INPUT.rglob("config.json") if hint in str(p.parent).lower()]
    assert hits, f"no model dir containing {hint!r} under {INPUT}"
    return sorted(hits, key=lambda p: len(str(p)))[0]


# ----------------------------------------------------------------- offline vllm
try:
    import vllm  # noqa: F401  (preinstalled or from a previous cell)
except ImportError:
    wheels = find_file(r"^vllm-.*\.whl$").parent
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-index",
         f"--find-links={wheels}", "vllm"],
        check=True,
    )

# ----------------------------------------------------------------- data
import csv  # noqa: E402

import numpy as np  # noqa: E402

samples = json.load(open(find_file(r"samples.*\.json$"), encoding="utf-8"))
test_path = Path(TEST_CSV) if TEST_CSV else find_file(r"test.*\.csv$")
test_rows = list(csv.DictReader(open(test_path, encoding="utf-8")))
print(f"samples: {len(samples)}   test rows: {len(test_rows)} ({test_path})")


def has_context(row):
    return str(row["context"]).strip() not in ("[NULL]", "NULL", "null", "")


_PUNCT = re.compile(r"[\s।,.;:!?\"'()\[\]{}\-–—`~*_/\\]+")


def norm(s):
    return _PUNCT.sub("", str(s))


ALL_ROWS = [("s", i, r) for i, r in enumerate(samples)] + [
    ("t", r["id"], r) for r in test_rows
]
NOCTX = [(k, i, r) for k, i, r in ALL_ROWS if not has_context(r)]
SIG = {}  # name -> {(split,key): value}


def store(name, keyed_vals, default=0.5):
    d = {(k, i): default for k, i, _ in ALL_ROWS}
    d.update(keyed_vals)
    SIG[name] = d
    print(f"signal {name}: {len(keyed_vals)} computed values")


# ----------------------------------------------------------------- 1. substring
store(
    "substring",
    {
        (k, i): (1.0 if norm(r["response_bn"]) and norm(r["response_bn"]) in norm(r["context"]) else 0.0)
        for k, i, r in ALL_ROWS
        if has_context(r)
    },
)

# ----------------------------------------------------------------- 2. NLI
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

nli_dir = str(find_model_dir("mdeberta"))
tok = AutoTokenizer.from_pretrained(nli_dir)
nli = AutoModelForSequenceClassification.from_pretrained(nli_dir).cuda().eval()
entail_idx = next(
    int(i) for i, l in nli.config.id2label.items() if "entail" in l.lower()
)
_SENT = re.compile(r"(?<=[।!?\n])")


def nli_chunks(text):
    parts = [p for p in _SENT.split(str(text)) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if len(tok.tokenize(cur + p)) > NLI_MAX_TOKENS and cur:
            chunks.append(cur)
            cur = ""
        cur += p
    if cur:
        chunks.append(cur)
    return (chunks or [str(text)])[:NLI_MAX_CHUNKS]


ctx_rows = [(k, i, r) for k, i, r in ALL_ROWS if has_context(r)]
pairs, owner = [], []
for j, (k, i, r) in enumerate(ctx_rows):
    for ch in nli_chunks(r["context"]):
        pairs.append((ch, str(r["response_bn"])))
        owner.append(j)
nli_vals = np.zeros(len(ctx_rows))
with torch.no_grad():
    for b in range(0, len(pairs), NLI_BATCH):
        prem = [p for p, _ in pairs[b:b + NLI_BATCH]]
        hyp = [h for _, h in pairs[b:b + NLI_BATCH]]
        enc = tok(prem, hyp, truncation=True, max_length=512, padding=True,
                  return_tensors="pt").to("cuda")
        probs = torch.softmax(nli(**enc).logits, -1)[:, entail_idx].cpu().numpy()
        for j, pr in enumerate(probs):
            o = owner[b + j]
            nli_vals[o] = max(nli_vals[o], float(pr))
store("nli", {(k, i): float(v) for (k, i, _), v in zip(ctx_rows, nli_vals)})
del nli, tok
gc.collect()
torch.cuda.empty_cache()

# ----------------------------------------------------------------- 3. retrieval evidence
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

passages, titles = [], []
with open(find_file(r"passages\.jsonl$"), encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        passages.append(d["text"])
        titles.append(d["title"])
emb = np.load(find_file(r"passage_embeddings\.npy$"))
assert len(passages) == emb.shape[0], "passages/embeddings misaligned"
print(f"wiki index: {len(passages)} passages")

bge = SentenceTransformer(str(find_model_dir("bge-m3")), device="cuda")
bge.max_seq_length = 256
q_texts = [str(r["prompt_bn"]) for _, _, r in NOCTX] + [
    f"{r['prompt_bn']} {r['response_bn']}" for _, _, r in NOCTX
]
q_emb = bge.encode(q_texts, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True).astype(np.float16)
del bge
gc.collect()
torch.cuda.empty_cache()

E = torch.tensor(emb, device="cuda")
Q = torch.tensor(q_emb, device="cuda")
nq = len(NOCTX)
cand = [set() for _ in range(nq)]
for b in range(0, len(Q), 256):
    top = (Q[b:b + 256] @ E.T).topk(TOPK_DENSE, dim=1).indices.cpu().numpy()
    for j, idxs in enumerate(top):
        cand[(b + j) % nq].update(int(x) for x in idxs)
del E, Q
gc.collect()
torch.cuda.empty_cache()

rr = CrossEncoder(str(find_model_dir("reranker")), device="cuda", max_length=512)
rr_pairs, rr_owner = [], []
for qi, (k, i, r) in enumerate(NOCTX):
    for pid in sorted(cand[qi]):
        rr_pairs.append((f"{r['prompt_bn']} উত্তর: {r['response_bn']}", passages[pid]))
        rr_owner.append((qi, pid))
scores = rr.predict(rr_pairs, batch_size=128, show_progress_bar=True)
by_q = {}
for (qi, pid), sc in zip(rr_owner, scores):
    by_q.setdefault(qi, []).append((float(sc), pid))
evidence = {}
for qi, (k, i, r) in enumerate(NOCTX):
    ranked = sorted(by_q.get(qi, []), reverse=True)[:TOPK_FINAL]
    evidence[f"{k}{i}"] = [
        {"title": titles[pid], "text": passages[pid]} for _, pid in ranked
    ]
json.dump(evidence, open(WORK / "evidence.json", "w"), ensure_ascii=False)
del rr, emb
gc.collect()
torch.cuda.empty_cache()

# ----------------------------------------------------------------- 4+5. vLLM stages
# Rows are exchanged with the stage subprocesses via JSON; each stage loads
# one model, computes its signals, writes stage_<name>.json, and exits so the
# GPUs are clean for the next stage.
json.dump(
    {
        "rows": [
            {"split": k, "key": i, "context": str(r["context"]),
             "prompt_bn": str(r["prompt_bn"]), "response_bn": str(r["response_bn"])}
            for k, i, r in ALL_ROWS
        ]
    },
    open(WORK / "rows.json", "w"),
    ensure_ascii=False,
)

STAGE_COMMON = '''
import json, math, os, re, sys
from pathlib import Path
import torch
from vllm import LLM, SamplingParams

os.environ.setdefault("HF_HUB_OFFLINE", "1")
WORK = Path("__WORK__")
rows = json.load(open(WORK / "rows.json", encoding="utf-8"))["rows"]
def has_context(r):
    return str(r["context"]).strip() not in ("[NULL]", "NULL", "null", "")
NOCTX = [r for r in rows if not has_context(r)]
llm = LLM(model="__MODEL__", dtype="half", max_model_len=__MAX_LEN__,
          tensor_parallel_size=torch.cuda.device_count(),
          gpu_memory_utilization=0.92)
V_BIN = SamplingParams(temperature=0, max_tokens=5)
V_SOFT = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
def chat(prompts, params):
    return llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
def binary(prompts):
    out = []
    for o in chat(prompts, V_BIN):
        up = o.outputs[0].text.strip().upper()
        out.append(1.0 if "YES" in up else 0.0 if "NO" in up else 0.5)
    return out
def soft(prompts):
    out = []
    for o in chat(prompts, V_SOFT):
        p = {"Y": 0.0, "N": 0.0, "U": 0.0}
        lps = o.outputs[0].logprobs
        for tokd in (lps[0].values() if lps else []):
            t = (tokd.decoded_token or "").strip().upper()
            if t and t[0] in p:
                p[t[0]] += math.exp(tokd.logprob)
        tot = sum(p.values())
        out.append(0.5 if tot < 0.05 else (p["Y"] + 0.5 * p["U"]) / tot)
    return out
def answers(prompts, n):
    ps = SamplingParams(temperature=0, max_tokens=n)
    return [o.outputs[0].text.strip() for o in chat(prompts, ps)]
def judge_prompt(r):
    q, a = r["prompt_bn"], r["response_bn"]
    if has_context(r):
        return ("You are verifying answers to Bengali questions against a source passage.\\n\\n"
                f"Passage:\\n{r['context'][:3500]}\\n\\nQuestion: {q}\\nCandidate answer: {a}\\n\\n"
                "Is the candidate answer correct AND supported by the passage? "
                "Reply with exactly one word: YES or NO.")
    return ("You are a careful fact-checker for Bengali question answering. "
            "Questions may involve Bengali grammar and literature, Bangladeshi "
            "history and culture, science, or mathematics.\\n\\n"
            f"Question: {q}\\nCandidate answer: {a}\\n\\n"
            "Is the candidate answer factually correct? "
            "Reply with exactly one word: YES or NO.")
def sv_answer_prompt(r, style):
    return ("Answer the following Bengali question concisely and accurately in Bengali. "
            f"Give only the answer, no explanation.\\n\\nQuestion: {r['prompt_bn']}")
def key(r):
    return f"{r['split']}{r['key']}"
'''

STAGE_14B = STAGE_COMMON + '''
sig = {"a0judge": {}, "a0selfv": {}}
for r, v in zip(rows, binary([judge_prompt(r) for r in rows])):
    sig["a0judge"][key(r)] = v
own = answers([sv_answer_prompt(r, 0) for r in NOCTX], 80)
comps = binary([
    ("Two answers to the same Bengali question are given.\\n\\n"
     f"Question: {r['prompt_bn']}\\nAnswer A: {r['response_bn']}\\nAnswer B: {a}\\n\\n"
     "Do Answer A and Answer B convey the same essential answer? "
     "Reply with exactly one word: YES or NO.")
    for r, a in zip(NOCTX, own)
])
for r, v in zip(NOCTX, comps):
    sig["a0selfv"][key(r)] = v
json.dump(sig, open(WORK / "stage_14b.json", "w"))
print("stage_14b done")
'''

STAGE_32B = STAGE_COMMON + '''
evidence = json.load(open(WORK / "evidence.json", encoding="utf-8"))
sig = {"judge32": {}, "j32sv": {}, "ret32": {}}
for r, v in zip(rows, binary([judge_prompt(r) for r in rows])):
    sig["judge32"][key(r)] = v
own = answers([sv_answer_prompt(r, 0) for r in NOCTX], 64)
comps = soft([
    ("Two answers to the same Bengali question are given.\\n\\n"
     f"Question: {r['prompt_bn']}\\nAnswer A: {r['response_bn']}\\nAnswer B: {a}\\n\\n"
     "Do Answer A and Answer B convey the same essential answer? "
     "Reply with exactly one word: YES, NO, or UNSURE (if Answer B does not "
     "actually answer the question).")
    for r, a in zip(NOCTX, own)
])
for r, v in zip(NOCTX, comps):
    sig["j32sv"][key(r)] = v
g_rows = [r for r in NOCTX if evidence.get(key(r))]
def ground_prompt(r):
    ev = evidence[key(r)]
    blocks = "\\n\\n".join(
        f"[Evidence {j + 1} — {p['title']}]\\n{p['text'][:600]}"
        for j, p in enumerate(ev[:4]))
    return ("You are verifying an answer to a Bengali question using retrieved "
            "encyclopedia passages. The passages may or may not be relevant.\\n\\n"
            f"{blocks}\\n\\nQuestion: {r['prompt_bn']}\\nCandidate answer: {r['response_bn']}\\n\\n"
            "Based ONLY on the evidence above: reply YES if the evidence supports "
            "the candidate answer, NO if the evidence contradicts it or shows the "
            "correct answer is different, or UNSURE if the evidence is irrelevant "
            "or insufficient to decide. Never answer NO merely because the "
            "evidence does not mention the answer. Reply with exactly one word.")
for r, v in zip(g_rows, soft([ground_prompt(r) for r in g_rows])):
    sig["ret32"][key(r)] = v
json.dump(sig, open(WORK / "stage_32b.json", "w"))
print("stage_32b done")
'''

for name, src, model_hint, max_len in [
    ("14b", STAGE_14B, "qwen14b", 4096),
    ("32b", STAGE_32B, "qwen32b", 8192),
]:
    path = WORK / f"stage_{name}.py"
    path.write_text(
        src.replace("__WORK__", str(WORK))
        .replace("__MODEL__", str(find_model_dir(model_hint)))
        .replace("__MAX_LEN__", str(max_len))
    )
    print(f"=== running stage {name} ===")
    subprocess.run([sys.executable, str(path)], check=True)
    stage = json.load(open(WORK / f"stage_{name}.json"))
    for sig_name, d in stage.items():
        store(sig_name, {
            (k, i): d[f"{k}{i}"] for k, i, _ in ALL_ROWS if f"{k}{i}" in d
        })

# ----------------------------------------------------------------- 6. stitch
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

NAMES = ["substring", "nli", "a0judge", "a0selfv", "judge32", "j32sv", "ret32"]


def matrix(keys):
    return np.column_stack([[SIG[n][key] for key in keys] for n in NAMES])


def f1_cls(y, p, c):
    tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
    fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
    fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def macro_f1(y, p):
    return (f1_cls(y, p, 0) + f1_cls(y, p, 1)) / 2


def best_threshold(y, prob):
    ts = np.arange(0.30, 0.71, 0.02)
    return float(ts[int(np.argmax(
        [macro_f1(y, (prob >= t).astype(int).tolist()) for t in ts]))])


Y = np.array([s["label"] for s in samples])
CTX_S = np.array([has_context(s) for s in samples])
CTX_T = np.array([has_context(r) for r in test_rows])
Xs = matrix([("s", i) for i in range(len(samples))])
Xt = matrix([("t", r["id"]) for r in test_rows])

sub = np.zeros(len(test_rows), int)
for branch, smask, tmask in [("ctx", CTX_S, CTX_T), ("noctx", ~CTX_S, ~CTX_T)]:
    thrs = []
    for seed in SEEDS:
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
        for tr, _ in skf.split(Xs[smask], Y[smask]):
            clf = LogisticRegression(max_iter=1000).fit(Xs[smask][tr], Y[smask][tr])
            thrs.append(best_threshold(Y[smask][tr],
                                       clf.predict_proba(Xs[smask][tr])[:, 1]))
    clf = LogisticRegression(max_iter=1000).fit(Xs[smask], Y[smask])
    thr = float(np.median(thrs))
    sub[tmask] = (clf.predict_proba(Xt[tmask])[:, 1] >= thr).astype(int)
    print(f"[{branch}] threshold {thr:.2f} "
          f"weights {dict(zip(NAMES, np.round(clf.coef_[0], 2)))}")

with open(WORK / "submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "label"])
    for r, p in zip(test_rows, sub):
        w.writerow([r["id"], int(p)])
n0 = int((sub == 0).sum())
print(f"wrote submission.csv: {len(sub)} rows ({n0} hallucinated / {len(sub) - n0} faithful)")
