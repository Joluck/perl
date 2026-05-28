#!/usr/bin/env python
"""Download a specific subfolder from MikaStars39/PeRL on Hugging Face."""
import os
import sys
from pathlib import Path

# Set mirror if direct HF is slow/unreachable, e.g.:
# export HF_ENDPOINT=https://hf-mirror.com
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

repo_id = "MikaStars39/PeRL"
subfolder = "dapo_dora_qwen2_5_1_5b_20251126_115730/checkpoint-1024"
local_dir = Path(__file__).resolve().parent / "outputs"

# Only download files inside the checkpoint subfolder
allow_patterns = [f"{subfolder}/*"]

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

print(f"Repo:  {repo_id}")
print(f"Path:  {subfolder}")
print(f"Dest:  {local_dir}")
print(f"HF endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co (default)')}")
print()

snapshot_download(
    repo_id=repo_id,
    allow_patterns=allow_patterns,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
)

print("\nDone.")
