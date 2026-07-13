"""
Fine-tune a Bengali answer-verifier cross-encoder.

Task: given (question, candidate_answer) decide whether the answer is CORRECT.
Trained on 413k pairs distilled from public Bengali gold banks (bangla-mmlu, bqad2025,
bagdhara idioms, bluck). Distractor choices are the negatives -- exam-setter distractors
are plausible-but-wrong, which is the hallucination distribution we must detect.

Applied at inference ONLY to rows where gold lookup abstains.
Emits probabilities for the 2516 test rows and the 299 labeled samples.
"""
import os, json, math, random, gc
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

MODEL   = os.environ.get("BH_MODEL", "csebuetnlp/banglabert")
MAXLEN  = 96
BS      = 64
EPOCHS  = 2
LR      = 3e-5
import glob as _glob
OUT = "/kaggle/working"

def find(pattern):
    hits = _glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} not found under /kaggle/input"
    return hits[0]

PAIRS = find("train_pairs.parquet")
TEST  = find("test set.csv")
SAMP  = find("dataset samples.json")
print("PAIRS:", PAIRS, "\nTEST:", TEST, "\nSAMP:", SAMP)

dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    cap = torch.cuda.get_device_capability(0)
    print("device:", torch.cuda.get_device_name(0), "capability", cap)
    assert cap >= (7, 0), f"GPU capability {cap} unsupported by this torch build -- need T4, not P100"
else:
    raise SystemExit("no GPU attached")

# ---------------- data ----------------
pairs = pd.read_parquet(PAIRS)
print("pairs:", pairs.shape, dict(pairs.label.value_counts()))

# hold out whole QUESTIONS (never split a question across train/val -> no leakage)
qs = pairs.question.unique()
rng = np.random.default_rng(SEED); rng.shuffle(qs)
val_q = set(qs[:4000])
tr = pairs[~pairs.question.isin(val_q)].reset_index(drop=True)
va = pairs[pairs.question.isin(val_q)].reset_index(drop=True)
print("train:", len(tr), "val:", len(va))

test = pd.read_csv(TEST)
samples = pd.DataFrame(json.load(open(SAMP)))

tok = AutoTokenizer.from_pretrained(MODEL)

class Pairs(Dataset):
    def __init__(self, q, a, y=None):
        self.q = [str(x) for x in q]; self.a = [str(x) for x in a]
        self.y = None if y is None else np.asarray(y, dtype=np.int64)
    def __len__(self): return len(self.q)
    def __getitem__(self, i):
        d = {"q": self.q[i], "a": self.a[i]}
        if self.y is not None: d["y"] = self.y[i]
        return d

def collate(batch):
    enc = tok([b["q"] for b in batch], [b["a"] for b in batch],
              truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    if "y" in batch[0]:
        enc["labels"] = torch.tensor([b["y"] for b in batch])
    return enc

tr_dl = DataLoader(Pairs(tr.question, tr.answer, tr.label), batch_size=BS, shuffle=True,
                   collate_fn=collate, num_workers=2, drop_last=True)
va_dl = DataLoader(Pairs(va.question, va.answer, va.label), batch_size=128, collate_fn=collate, num_workers=2)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = len(tr_dl) * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.cuda.amp.GradScaler()

@torch.no_grad()
def predict(q, a, bs=256):
    model.eval()
    dl = DataLoader(Pairs(q, a), batch_size=bs, collate_fn=collate, num_workers=2)
    out = []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.cuda.amp.autocast():
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)

for ep in range(EPOCHS):
    model.train()
    tot = 0.0
    for i, b in enumerate(tr_dl):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = model(**b).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        tot += loss.item()
        if i % 500 == 0: print(f"ep{ep} step {i}/{len(tr_dl)} loss {tot/(i+1):.4f}", flush=True)
    vp = predict(va.question, va.answer)
    acc = ((vp > 0.5).astype(int) == va.label.values).mean()
    print(f"== epoch {ep}: held-out pair acc = {acc:.4f}", flush=True)

# ---------------- inference ----------------
tp = predict(test.prompt_bn.astype(str), test.response_bn.astype(str))
sp = predict(samples.prompt_bn.astype(str), samples.response_bn.astype(str))
pd.DataFrame({"id": test.id, "prob_faithful": tp}).to_csv(f"{OUT}/probs_test.csv", index=False)
pd.DataFrame({"idx": range(len(samples)), "label": samples.label, "prob_faithful": sp}).to_csv(f"{OUT}/probs_samples.csv", index=False)

# quick sanity on the samples (all 299, incl. gold-covered ones)
for th in (0.3, 0.4, 0.5, 0.6, 0.7):
    pred = (sp > th).astype(int)
    print(f"samples th={th}: acc={ (pred==samples.label.values).mean():.4f}")
print("saved probs_test.csv / probs_samples.csv")
