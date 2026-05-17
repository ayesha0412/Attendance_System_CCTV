"""
AdaFace One-Time Setup
======================
Downloads AdaFace IR-50 (MS1MV2) from HuggingFace.
Model: minchul/cvlface_adaface_ir50_ms1mv2

Run this once before train_adaface.py or live_adaface.py.

Usage:
    python setup_adaface.py
"""

import os
import sys
import subprocess

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaface_assets")
MODEL_DIR  = os.path.join(ASSETS_DIR, "cvlface_model")
HF_REPO    = "minchul/cvlface_adaface_ir50_ms1mv2"


def check_torch():
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__} found.")
    except ImportError:
        print("[..] PyTorch not found — installing (CUDA 12.6)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "torch", "torchvision",
                               "--index-url", "https://download.pytorch.org/whl/cu126"])
        print("[OK] PyTorch installed.")


def download_model():
    # Check if already downloaded (look for a config or weights file)
    marker = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(marker):
        print(f"[OK] Model already downloaded — skipping.")
        return

    print(f"[..] Downloading {HF_REPO} from HuggingFace (~170 MB)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "huggingface_hub", "transformers", "safetensors",
                               "omegaconf", "-q"])
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=MODEL_DIR,
            ignore_patterns=["*.md", "*.txt"],
        )
        print(f"[OK] Model saved to {MODEL_DIR}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print()
        print("  Check your internet connection and try again.")
        print(f"  Or manually download from: https://huggingface.co/{HF_REPO}")
        print(f"  and save all files to: {MODEL_DIR}")
        sys.exit(1)


def main():
    print("=" * 55)
    print("  AdaFace Setup")
    print(f"  Model: {HF_REPO}")
    print("=" * 55)
    print()

    os.makedirs(MODEL_DIR, exist_ok=True)

    check_torch()
    print()
    download_model()

    print()
    print("=" * 55)
    print("  Setup complete!")
    print("  Next: python train_adaface.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
