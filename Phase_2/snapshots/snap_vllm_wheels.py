# Download the full vLLM wheel set for offline install (CPU).
# The Phase-2 notebook installs with: pip install --no-index --find-links=<this> vllm
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "download", "vllm", "-d", "/kaggle/working/wheels"],
    check=True,
)
print("wheel set complete")
