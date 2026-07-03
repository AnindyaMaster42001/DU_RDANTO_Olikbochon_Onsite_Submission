# Snapshot the three small encoders used by the pipeline (CPU).
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"], check=True)
from huggingface_hub import snapshot_download

SKIP = ["*.onnx", "onnx/*", "*.h5", "*.msgpack", "openvino/*"]
for repo, dirname in [
    ("BAAI/bge-m3", "bge-m3"),
    ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
    ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "mdeberta-xnli"),
]:
    snapshot_download(repo, local_dir=f"/kaggle/working/{dirname}", ignore_patterns=SKIP)
    print("snapshot complete:", dirname)
