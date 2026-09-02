#!/usr/bin/env bash
# One-time GPU environment setup for a sharded-ingest worker box.
# Idempotent (pip is a no-op when satisfied). Colab runs it per fresh VM;
# RunPod runs it once in RunPodBackend.prepare().
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The stock RunPod image (runpod/pytorch:*-cuda12.4.1-*) ships no `zip`;
# ingest_shard.sh needs it to package shard_storage/. Colab VMs already have
# it, so `command -v` skips the apt call there.
command -v zip >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq zip; }

pip install -e ".[gpu]"

# fastembed drags in CPU-only onnxruntime; swap it for a CUDA-12 build.
# The plain PyPI onnxruntime-gpu wheel now targets CUDA 13 (needs
# libcublasLt.so.13; the RunPod stock image and Colab are CUDA 12.4) - the
# CUDA-12 build lives on Microsoft's ADO feed. That wheel links cuBLAS 12
# (present in the -devel image) but NOT cuDNN, and neither the stock RunPod
# image nor a fresh Colab VM reliably has cuDNN 9 on onnxruntime's loader
# path - so pin the cuDNN 9 wheel explicitly. preload_dlls() (in
# _verify_gpu.py and lexausearch.indexer) then resolves it from site-packages
# regardless of shell / LD_LIBRARY_PATH / login state.
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.27.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install "nvidia-cudnn-cu12>=9,<10"

# Real GPU check - a silent CPU fall-back is invisible to
# get_available_providers(), so force an actual CUDA InferenceSession.
python3 "$here/_verify_gpu.py"
