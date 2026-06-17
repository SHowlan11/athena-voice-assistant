"""Verify GPU is ready for AI inference"""
import sys

print("=" * 50)
print("  Athena Hardware Check")
print("=" * 50)

# 1. Python version
print(f"\n[OK] Python {sys.version.split()[0]}")

# 2. PyTorch
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    print(f"     CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"     VRAM: {total_mem:.1f} GB")
        print(f"     CUDA version: {torch.version.cuda}")
    else:
        print("[WARN] CUDA not available - check driver")
except ImportError:
    print("[FAIL] PyTorch not installed")

# 3. Other key libraries
libs = ["numpy", "sounddevice", "faster_whisper", "edge_tts", "scipy"]
for lib in libs:
    try:
        __import__(lib.replace("-", "_"))
        print(f"[OK] {lib}")
    except ImportError:
        print(f"[MISS] {lib}")

print("\n" + "=" * 50)
print("  Done!")
print("=" * 50)
