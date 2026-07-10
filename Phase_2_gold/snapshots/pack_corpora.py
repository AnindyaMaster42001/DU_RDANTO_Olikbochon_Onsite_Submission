# Package the Layer-1 gold corpora as an offline Kaggle dataset (INTERNET ON, CPU kernel).
#
# Run this once with internet enabled. It assembles every public reference corpus the
# verifier grounds on into /kaggle/working/ext/ in the exact layout gold_verify.py expects,
# then you save the kernel output as a Kaggle Dataset (e.g. "bengali-gold-corpora") and
# attach it to the offline submission notebook. All sources are public, pre-competition,
# and permitted by rule 5 — cite every one in the paper (see README "Corpora and citations").
#
# Two of the corpora (A, B) are Kaggle Datasets: attach them to THIS kernel as inputs
# (Add Input -> Datasets) and they are copied from /kaggle/input; if not attached, the
# kernel falls back to the Kaggle CLI. The rest (C, D, E) are fetched from Hugging Face.
#
# BCS toggle: E (the BCS banks) is what lifts 0.901 -> 0.904. It is always packaged here;
# the offline notebook's USE_BCS flag decides whether to consult it, so one dataset serves
# both finals.

import glob
import os
import shutil
import subprocess
import sys

import requests

EXT = "/kaggle/working/ext"
os.makedirs(EXT, exist_ok=True)


def _from_input_or_cli(slug, dest):
    """Copy an attached Kaggle dataset from /kaggle/input, else download via the CLI."""
    name = slug.split("/")[1]
    os.makedirs(dest, exist_ok=True)
    hits = glob.glob(f"/kaggle/input/{name}/**/*", recursive=True) + glob.glob(f"/kaggle/input/**/{name}/**/*", recursive=True)
    files = [h for h in set(hits) if os.path.isfile(h)]
    if files:
        for f in files:
            shutil.copy(f, os.path.join(dest, os.path.basename(f)))
        print(f"  {slug}: copied {len(files)} files from /kaggle/input")
        return
    print(f"  {slug}: not attached, downloading via kaggle CLI")
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "--unzip", "-p", dest], check=True)


# A. BanglaHalluEval GQA (= TyDiQA-GoldP Bengali gold answers) — arXiv 2605.31483
_from_input_or_cli("abidur14004/bangla-dataset-for-hallucination",
                   f"{EXT}/bangla-dataset-for-hallucination")

# B. bagdhara: Bengali idioms with literal + figurative meanings
_from_input_or_cli("sakhadib/bagdhara-bangla-idioms-dataset",
                   f"{EXT}/bagdhara-bangla-idioms-dataset")

# C. hishab/bangla-mmlu: 87,869 Bengali exam MCQs with answer keys (ungated)
os.makedirs(f"{EXT}/mmlu", exist_ok=True)
import pandas as pd
parts = []
for sp in ("test", "validation", "dev"):
    u = f"https://huggingface.co/api/datasets/hishab/bangla-mmlu/parquet/default/{sp}/0.parquet"
    p = f"{EXT}/mmlu/{sp}.parquet"
    open(p, "wb").write(requests.get(u, timeout=600).content)
    d = pd.read_parquet(p)
    d["__split"] = sp
    parts.append(d)
pd.concat(parts, ignore_index=True).to_parquet(f"{EXT}/mmlu/bangla_mmlu_all.parquet")
print("  bangla-mmlu rows:", sum(len(d) for d in parts))

# D. csebuetnlp/squad_bn: extractive gold spans (fallback for cross-script abstains)
os.makedirs(f"{EXT}/more", exist_ok=True)
open(f"{EXT}/more/squad_bn.tar.bz2", "wb").write(requests.get(
    "https://huggingface.co/datasets/csebuetnlp/squad_bn/resolve/main/data/squad_bn.tar.bz2",
    timeout=600).content)
print("  squad_bn fetched")

# E. azminetoushikwasi BCS question banks (10th-45th). `answer` indexes `options`; the
#    index BASE differs per file — load_bcs() infers it per file. (0.901 -> 0.904)
os.makedirs(f"{EXT}/bcs", exist_ok=True)
for ds in ("azminetoushikwasi/bangla-bcs-qs",
           "azminetoushikwasi/bcs-10-40th-GK-ICT-DM-NMS",
           "azminetoushikwasi/bd-bcs-multimodal"):
    tree = requests.get(f"https://huggingface.co/api/datasets/{ds}/tree/main?recursive=1",
                        timeout=60).json()
    for f in [x["path"] for x in tree if x["path"].endswith(".json")]:
        r = requests.get(f"https://huggingface.co/datasets/{ds}/resolve/main/{f}", timeout=300)
        if r.status_code == 200:
            open(f"{EXT}/bcs/{ds.split('/')[1]}__{f.replace('/', '_')}", "wb").write(r.content)
print("  bcs corpora ready")

# Sanity: build the verifier over what we just packaged (needs gold_verify.py + bnnum.py
# attached to this kernel, or run in the repo).
try:
    os.environ["BHD_EXT"] = EXT + "/"
    sys.path.insert(0, os.path.dirname(__file__) if "__file__" in globals() else ".")
    from gold_verify import GoldVerifier
    V = GoldVerifier(with_squad=False)
    print(f"\nverifier built: hallueval={len(V.qa)} idioms={len(V.idi)} "
          f"mmlu={len(V.mmlu)} bcs={len(V.bcs)} extra={len(V.extra)}")
except Exception as ex:
    print(f"\n(verifier sanity check skipped: {ex})")

total = sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(EXT) for f in fs)
print(f"\next/ assembled: {total / 1e6:.1f} MB — save this kernel's output as a Kaggle Dataset "
      f"and attach it to the submission notebook.")
