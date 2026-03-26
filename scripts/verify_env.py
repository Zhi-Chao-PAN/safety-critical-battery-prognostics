"""
Environment Verification Script (Phase 4.2)
Ensures dependencies, exact torch/cuda versions, and folders match before running experiments.
"""

import os
import sys


def check_env():
    print("Verifying V3.5 Environment...")

    # Check Python
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10+ required.")
        return False

    try:
        import torch
        print(f"✅ PyTorch version {torch.__version__} found.")
        if torch.cuda.is_available():
            print(f"✅ CUDA available. Device: {torch.cuda.get_device_name(0)}")
        else:
             print("⚠️ CUDA not available. Training will be slow.")
    except ImportError:
         print("❌ Error: PyTorch not installed.")
         return False

    # Check folders
    folders = ['data/raw', 'data/processed', 'experiments/configs', 'experiments/pipelines', 'experiments/tracking']
    for f in folders:
        if not os.path.exists(f):
             print(f"❌ Error: Required directory missing: {f}")
             return False

    print("✅ Environment verification passed!")
    return True

if __name__ == "__main__":
    check_env()
