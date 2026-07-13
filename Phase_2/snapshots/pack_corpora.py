# Package the Layer-1 gold corpora as an offline Kaggle dataset (INTERNET ON, CPU kernel).
#
# Run this once with internet enabled. It assembles every public reference corpus the
# verifier grounds on into /kaggle/working/ext/ in the exact layout gold_verify.py expects,
# then you save the kernel output as a Kaggle Dataset (e.g. "bengali-gold-corpora") and
# attach it to the offline submission notebook. All sources are public, pre-competition,
# and permitted by rule 5 -- cite every one in the paper (see README "Corpora and citations").
#
# Attach the two Kaggle Datasets (A, B) to THIS kernel as inputs (Add Input -> Datasets);
# they are copied from /kaggle/input. The rest (C, D, E) are fetched from Hugging Face over
# the internet. Every corpus is wrapped so one failure cannot abort the build, and a
# manifest.json records exactly what was assembled.

import glob
import json
import os
import shutil
import traceback

import requests

EXT = "/kaggle/working/ext"
os.makedirs(EXT, exist_ok=True)
MANIFEST = {}


def _record(key, ok, detail):
    MANIFEST[key] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'OK ' if ok else 'FAIL'}] {key}: {detail}", flush=True)


def _fetch_kaggle_ds(slug, dest):
    """Download a Kaggle dataset in-kernel via kagglehub (internet on; no attach needed)
    and flatten its files into dest. Robust to the /kaggle/input mount-path variance that
    breaks attach+glob."""
    import kagglehub
    os.makedirs(dest, exist_ok=True)
    path = kagglehub.dataset_download(slug)
    n = 0
    for r, _d, fs in os.walk(path):
        for f in fs:
            try:
                shutil.copy(os.path.join(r, f), os.path.join(dest, os.path.basename(f)))
                n += 1
            except Exception:
                pass
    return n


def _get(url, path, timeout=600):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    open(path, "wb").write(r.content)
    return len(r.content)


# A. BanglaHalluEval GQA (= TyDiQA-GoldP Bengali gold answers) -- arXiv 2605.31483
try:
    n = _fetch_kaggle_ds("abidur14004/bangla-dataset-for-hallucination",
                         f"{EXT}/bangla-dataset-for-hallucination")
    _record("hallueval", n > 0, f"{n} files via kagglehub")
except Exception:
    _record("hallueval", False, traceback.format_exc().splitlines()[-1])

# B. bagdhara: Bengali idioms with literal + figurative meanings
try:
    n = _fetch_kaggle_ds("sakhadib/bagdhara-bangla-idioms-dataset",
                         f"{EXT}/bagdhara-bangla-idioms-dataset")
    _record("bagdhara", n > 0, f"{n} files via kagglehub")
except Exception:
    _record("bagdhara", False, traceback.format_exc().splitlines()[-1])

# C. hishab/bangla-mmlu: 87,869 Bengali exam MCQs with answer keys (ungated)
try:
    import pandas as pd
    os.makedirs(f"{EXT}/mmlu", exist_ok=True)
    parts = []
    for sp in ("test", "validation", "dev"):
        u = f"https://huggingface.co/api/datasets/hishab/bangla-mmlu/parquet/default/{sp}/0.parquet"
        p = f"{EXT}/mmlu/{sp}.parquet"
        _get(u, p)
        d = pd.read_parquet(p)
        d["__split"] = sp
        parts.append(d)
    pd.concat(parts, ignore_index=True).to_parquet(f"{EXT}/mmlu/bangla_mmlu_all.parquet")
    _record("mmlu", True, f"{sum(len(d) for d in parts)} rows")
except Exception:
    _record("mmlu", False, traceback.format_exc().splitlines()[-1])

# D. csebuetnlp/squad_bn: extractive gold spans (fallback for cross-script abstains)
try:
    os.makedirs(f"{EXT}/more", exist_ok=True)
    b = _get("https://huggingface.co/datasets/csebuetnlp/squad_bn/resolve/main/data/squad_bn.tar.bz2",
             f"{EXT}/more/squad_bn.tar.bz2")
    _record("squad_bn", b > 0, f"{b} bytes")
except Exception:
    _record("squad_bn", False, traceback.format_exc().splitlines()[-1])

# E. azminetoushikwasi BCS question banks (10th-45th). `answer` indexes `options`; the
#    index BASE differs per file -- load_bcs() infers it per file. (0.901 -> 0.904)
try:
    os.makedirs(f"{EXT}/bcs", exist_ok=True)
    nb = 0
    for ds in ("azminetoushikwasi/bangla-bcs-qs",
               "azminetoushikwasi/bcs-10-40th-GK-ICT-DM-NMS",
               "azminetoushikwasi/bd-bcs-multimodal"):
        tree = requests.get(f"https://huggingface.co/api/datasets/{ds}/tree/main?recursive=1",
                            timeout=60).json()
        for f in [x["path"] for x in tree if x.get("path", "").endswith(".json")]:
            r = requests.get(f"https://huggingface.co/datasets/{ds}/resolve/main/{f}", timeout=300)
            if r.status_code == 200:
                open(f"{EXT}/bcs/{ds.split('/')[1]}__{f.replace('/', '_')}", "wb").write(r.content)
                nb += 1
    _record("bcs", nb > 0, f"{nb} json files")
except Exception:
    _record("bcs", False, traceback.format_exc().splitlines()[-1])

# ---- bundle the verifier code so the dataset is self-contained for downstream kernels ----
for cf in ("gold_verify.py", "bnnum.py"):
    for src in (cf, os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", cf)):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(EXT, cf))
            break

# ---- manifest + sanity build ----
total = sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(EXT) for f in fs)
MANIFEST["_total_mb"] = round(total / 1e6, 1)
json.dump(MANIFEST, open("/kaggle/working/manifest.json", "w"), indent=2)
print("\n=== MANIFEST ===")
print(json.dumps(MANIFEST, indent=2), flush=True)

try:
    os.environ["BHD_EXT"] = EXT + "/"
    import sys
    sys.path.insert(0, os.path.dirname(__file__) if "__file__" in globals() else ".")
    from gold_verify import GoldVerifier
    V = GoldVerifier(with_squad=False)
    print(f"\nverifier built: hallueval={len(V.qa)} idioms={len(V.idi)} "
          f"mmlu={len(V.mmlu)} bcs={len(V.bcs)} extra={len(V.extra)}", flush=True)
except Exception:
    print("\n(verifier sanity check skipped:\n" + traceback.format_exc() + ")", flush=True)

print(f"\next/ assembled: {MANIFEST['_total_mb']} MB -- save this kernel's output as a Kaggle "
      f"Dataset and attach it to the submission notebook.")
