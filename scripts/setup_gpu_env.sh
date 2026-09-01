#!/usr/bin/env bash
# One-time GPU environment setup for a sharded-ingest worker box.
# Idempotent (pip is a no-op when satisfied). Colab runs it per fresh VM;
# RunPod runs it once in RunPodBackend.prepare().
set -euo pipefail

pip install -e ".[gpu]"
pip install huggingface_hub

# onnxruntime-gpu built against CUDA 12 - the stock pip wheel is CUDA-11 and
# silently falls back to CPU. Same fix as the original colab_ingest_shard.sh.
pip uninstall -y onnxruntime-gpu onnxruntime
pip install onnxruntime-gpu==1.27.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

python3 -c "
import onnxruntime
providers = onnxruntime.get_available_providers()
assert 'CUDAExecutionProvider' in providers, (
    f'CUDA not available after gpu extra install (providers: {providers}).'
)
print('setup_gpu_env: CUDA available, providers:', providers)
"
