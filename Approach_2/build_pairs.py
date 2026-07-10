"""
Build (question, candidate_answer) -> is_correct pairs from the public gold banks.

Positives  = the gold answer.
Negatives  = the distractor choices (plausible wrong answers written by exam setters,
             which is exactly the hallucination distribution we must detect).

Any question that appears in the 299 labeled samples is EXCLUDED from training so the
sample set stays a clean validation signal.
"""
import json, glob, re, random
import pandas as pd
from gold_verify import norm_mmlu_q, norm_q

random.seed(0)
EXT = "ext/"
B = "/mnt/NewVolume2/Android Projects/bengali-hallucination-detection/Approach_0/"

samples = pd.DataFrame(json.load(open(B + "dataset samples.json")))
BLOCK = {norm_mmlu_q(q) for q in samples.prompt_bn}
print(f"blocked (sample) questions: {len(BLOCK)}")

rows = []
def add(q, a, y, src):
    q, a = str(q).strip(), str(a).strip()
    if not q or not a or a.lower() == "nan": return
    if norm_mmlu_q(q) in BLOCK: return
    rows.append((q, a, y, src))

# --- bangla-mmlu: 1 gold + 3 distractors ---
mm = pd.read_parquet(EXT + "mmlu/bangla_mmlu_all.parquet")
for r in mm.itertuples():
    if r.answer not in "ABCD": continue
    ch = list(r.choices)
    gi = ord(r.answer) - 65
    if gi >= len(ch): continue
    add(r.question, ch[gi], 1, "mmlu")
    for j, c in enumerate(ch):
        if j != gi: add(r.question, c, 0, "mmlu")

# --- bqad2025 MCQ ---
d = pd.read_csv(EXT + "bqad2025/bqad2025.csv")
for r in d.itertuples():
    k = str(r.Answer).strip()
    if k not in "ABCD": continue
    for col in "ABCD":
        v = getattr(r, col)
        add(r.Question, v, int(col == k), "bqad")

# --- BEnQA MCQ (Bengali side) ---
for f in glob.glob(EXT + "more/benqa/*.csv"):
    try: d = pd.read_csv(f)
    except Exception: continue
    if "Bengali Question" not in d.columns or "Correct Answer" not in d.columns: continue
    for _, r in d.iterrows():
        k = str(r["Correct Answer"]).strip()
        if k not in "ABCD": continue
        for col in "ABCD":
            v = r.get(f"{col} Bn", r.get(col))
            add(r["Bengali Question"], v, int(col == k), "benqa")

# --- bagdhara idioms: meaning-matching, with other idioms' meanings as negatives ---
ents = []
for f in glob.glob(EXT + "bagdhara-bangla-idioms-dataset/*.json"):
    try: ents.append(json.load(open(f)))
    except Exception: pass
figs = [e["figurative_meaning_bn"] for e in ents if e.get("figurative_meaning_bn")]
for e in ents:
    head = e.get("idiom", "")
    fig = e.get("figurative_meaning_bn")
    lit = e.get("literal_meaning")
    if fig:
        add(f'"{head}" এর ভাবার্থ কী?', fig, 1, "idiom")
        add(f'"{head}" এর ভাবার্থ কী?', random.choice(figs), 0, "idiom")
    if lit:
        add(f'"{head}" এর শাব্দিক অর্থ কী?', lit, 1, "idiom")
        add(f'"{head}" এর শাব্দিক অর্থ কী?', random.choice(figs), 0, "idiom")

# --- bluck_bn: positives + a random other answer as negative ---
d = pd.read_csv(EXT + "bluck-bangla/bluck_bn.csv")
ans = d.answer.astype(str).tolist()
for r in d.itertuples():
    add(r.question, r.answer, 1, "bluck")
    add(r.question, random.choice(ans), 0, "bluck")

df = pd.DataFrame(rows, columns=["question", "answer", "label", "src"])
df = df.drop_duplicates(subset=["question", "answer"])
# drop questions where a "negative" accidentally equals the gold string
g = df[df.label == 1].groupby("question").answer.apply(set).to_dict()
df = df[~df.apply(lambda r: r.label == 0 and r.answer in g.get(r.question, ()), axis=1)]

print(df.groupby(["src", "label"]).size())
print("total pairs:", len(df), " pos:", (df.label == 1).sum(), " neg:", (df.label == 0).sum())
print("unique questions:", df.question.nunique())
df.to_parquet("train_pairs.parquet", index=False)
print("-> train_pairs.parquet")
