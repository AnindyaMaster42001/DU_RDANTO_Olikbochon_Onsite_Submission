#!/usr/bin/env bash
# Fetch the public reference corpora the verifier grounds on.
# All are publicly available and permitted by competition rule 5
# ("Any publicly available Bengali or multilingual dataset may be used ...").
# Cite every one of these in the Phase-2 paper.
set -euo pipefail
EXT="${1:-ext}"
mkdir -p "$EXT"

# A. BanglaHalluEval GQA  == TyDiQA-GoldP Bengali gold answers  (arXiv 2605.31483)
kaggle datasets download -d abidur14004/bangla-dataset-for-hallucination \
  --unzip -p "$EXT/bangla-dataset-for-hallucination"

# B. bagdhara: 10,361 Bengali idioms with literal_meaning + figurative_meaning_bn
kaggle datasets download -d sakhadib/bagdhara-bangla-idioms-dataset \
  --unzip -p "$EXT/bagdhara-bangla-idioms-dataset"

# C. hishab/bangla-mmlu: 87,869 Bengali exam MCQs with answer keys (ungated)
mkdir -p "$EXT/mmlu"
python - "$EXT" <<'PY'
import sys, requests, pandas as pd
ext = sys.argv[1]
parts = []
for sp in ("test", "validation", "dev"):
    u = f"https://huggingface.co/api/datasets/hishab/bangla-mmlu/parquet/default/{sp}/0.parquet"
    p = f"{ext}/mmlu/{sp}.parquet"
    open(p, "wb").write(requests.get(u, timeout=600).content)
    d = pd.read_parquet(p); d["__split"] = sp; parts.append(d)
pd.concat(parts, ignore_index=True).to_parquet(f"{ext}/mmlu/bangla_mmlu_all.parquet")
print("bangla-mmlu rows:", sum(len(d) for d in parts))
PY

# D. csebuetnlp/squad_bn: extractive gold spans (fallback for cross-script abstains)
mkdir -p "$EXT/more"
curl -sL -o "$EXT/more/squad_bn.tar.bz2" \
  "https://huggingface.co/datasets/csebuetnlp/squad_bn/resolve/main/data/squad_bn.tar.bz2"

echo "Corpora ready under $EXT/"
