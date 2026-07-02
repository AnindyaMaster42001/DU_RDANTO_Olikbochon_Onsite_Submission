"""Fine-tune BanglaBERT as a hallucination classifier on the 299 samples.

Input encoding: "<context> [SEP] <prompt> [SEP] <response>" -> binary label.
Protocol:
  1. 5-fold stratified CV -> honest out-of-fold macro-F1 (overall + per branch)
  2. retrain on all 299 samples -> predict the test set
Outputs:
  oof_banglabert.csv            out-of-fold sample predictions (for evaluate.py)
  banglabert_test_probs.json    P(faithful) per test row (ensemble signal)
  submission_banglabert.csv     hard 0/1 submission (thresholded at CV-best)

Usage: .venv/bin/python train_local.py
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluate import f1_on_class, has_context, macro_f1

MODEL = "csebuetnlp/banglabert"
MAX_LEN = 256
BATCH = 8
EPOCHS = 4
LR = 2e-5
SEED = 42
DATA_DIR = Path(__file__).parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_text(row):
    ctx = str(row["context"])
    ctx = "" if ctx.strip() in ("[NULL]", "NULL", "null") else ctx
    return f"{ctx} [SEP] {row['prompt_bn']} [SEP] {row['response_bn']}"


class HallDataset(Dataset):
    def __init__(self, rows, tokenizer, labels=None):
        self.enc = tokenizer(
            [build_text(r) for r in rows],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i])
        return item


def train_one(tokenizer, train_rows, train_labels, seed):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2
    ).to(DEVICE)
    loader = DataLoader(
        HallDataset(train_rows, tokenizer, train_labels),
        batch_size=BATCH,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * len(loader)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.05, total_iters=steps
    )
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            opt.zero_grad()
            with torch.amp.autocast(DEVICE, enabled=DEVICE == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        print(f"    epoch {epoch + 1}/{EPOCHS} loss {total / len(loader):.4f}")
    return model


@torch.no_grad()
def predict_probs(model, tokenizer, rows):
    model.eval()
    loader = DataLoader(HallDataset(rows, tokenizer), batch_size=32)
    probs = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.amp.autocast(DEVICE, enabled=DEVICE == "cuda"):
            logits = model(**batch).logits
        probs.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().tolist())
    return probs  # P(faithful)


def main():
    print(f"device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    samples = json.load(open(DATA_DIR / "dataset samples.json", encoding="utf-8"))
    labels = np.array([s["label"] for s in samples])

    # ---- 5-fold CV
    oof = np.zeros(len(samples))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(samples, labels)):
        print(f"fold {fold + 1}/5 (train {len(tr)}, val {len(va)})")
        model = train_one(
            tokenizer, [samples[i] for i in tr], labels[tr].tolist(), SEED + fold
        )
        oof[va] = predict_probs(model, tokenizer, [samples[i] for i in va])
        del model
        torch.cuda.empty_cache()

    # threshold sweep on OOF probabilities
    best_t, best_m = 0.5, -1.0
    for t in np.arange(0.2, 0.81, 0.05):
        m = macro_f1(labels.tolist(), (oof >= t).astype(int).tolist())
        if m > best_m:
            best_t, best_m = float(t), m
    preds = (oof >= best_t).astype(int).tolist()
    y = labels.tolist()
    ctx_i = [i for i, s in enumerate(samples) if has_context(s)]
    noc_i = [i for i, s in enumerate(samples) if not has_context(s)]
    print(f"\nCV macro-F1 {best_m:.4f} @ threshold {best_t:.2f}   "
          f"F1(hall) {f1_on_class(y, preds, 0)[0]:.4f}")
    print(f"  with context  macroF1 "
          f"{macro_f1([y[i] for i in ctx_i], [preds[i] for i in ctx_i]):.4f}")
    print(f"  no context    macroF1 "
          f"{macro_f1([y[i] for i in noc_i], [preds[i] for i in noc_i]):.4f}")

    with open(DATA_DIR / "oof_banglabert.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "prob_faithful"])
        for i, (p, pr) in enumerate(zip(preds, oof)):
            w.writerow([i, p, f"{pr:.4f}"])

    # ---- final model on all samples -> test predictions
    print("\nfinal model on all 299 samples...")
    model = train_one(tokenizer, samples, labels.tolist(), SEED)
    test_rows = list(
        csv.DictReader(open(DATA_DIR / "test set.csv", encoding="utf-8"))
    )
    probs = predict_probs(model, tokenizer, test_rows)
    with open(DATA_DIR / "banglabert_test_probs.json", "w") as f:
        json.dump({r["id"]: p for r, p in zip(test_rows, probs)}, f)
    with open(DATA_DIR / "submission_banglabert.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for r, p in zip(test_rows, probs):
            w.writerow([r["id"], int(p >= best_t)])
    n0 = sum(1 for p in probs if p < best_t)
    print(f"wrote submission_banglabert.csv: {len(probs)} rows "
          f"({n0} hallucinated / {len(probs) - n0} faithful)")


if __name__ == "__main__":
    main()
