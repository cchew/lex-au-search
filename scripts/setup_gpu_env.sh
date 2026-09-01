#!/usr/bin/env bash
# One-time GPU environment setup for a sharded-ingest worker box.
# Idempotent (pip is a no-op when satisfied). Colab runs it per fresh VM;
# RunPod runs it once in RunPodBackend.prepare().
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install -e ".[gpu]"

# fastembed drags in CPU-only onnxruntime; swap it for the CUDA-12 build.
# The PyPI onnxruntime-gpu wheel has targeted CUDA 12 since 1.19 (the old
# "stock wheel is CUDA-11" workaround feed is no longer needed), but it does
# NOT reliably pull cuDNN, and the RunPod stock pytorch image ships no system
# cuDNN 9 on onnxruntime's loader path - so pin the cuDNN 9 wheel explicitly.
# preload_dlls() (in _verify_gpu.py and lexausearch.indexer) then resolves it
# from site-packages regardless of shell / LD_LIBRARY_PATH / login state.
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu>=1.20,<2" "nvidia-cudnn-cu12>=9,<10"

# Real GPU check - a silent CPU fall-back is invisible to
# get_available_providers(), so force an actual CUDA InferenceSession.
python3 "$here/_verify_gpu.py"
