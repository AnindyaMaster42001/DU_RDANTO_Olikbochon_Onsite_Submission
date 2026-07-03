# Snapshot Qwen2.5-14B-Instruct-GPTQ-Int4 into a pinned Kaggle artifact (CPU).
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"], check=True)
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
    local_dir="/kaggle/working/qwen14b-gptq-int4",
)
print("snapshot complete")
