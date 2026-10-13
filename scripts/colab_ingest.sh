#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a freshly cloned lex-au-search repo, on a GPU runtime
# (e.g. Google Colab, or any CUDA-capable box). Installs the gpu extra,
# downloads the lex-au corpus from Hugging Face, runs the ingest, and zips
# the result for download. See README.md's "GPU ingest via Colab" section
# for the notebook cells that call this script.

pip install -e ".[gpu]"

# fastembed pulls in plain CPU onnxruntime as a transitive dep, which then wins
# the "onnxruntime" import namespace over the gpu extra's onnxruntime-gpu
# (confirmed 2026-07-15: T4 attached and idle while ingest silently ran on CPU).
# The plain PyPI onnxruntime-gpu wheel now targets CUDA 13; Colab's T4 runtime
# is CUDA 12, so pull the CUDA-12 build from Microsoft's ADO feed, and pin the
# cuDNN 9 wheel so preload_dlls() (in _verify_gpu.py and lexausearch.indexer)
# can resolve it. Same setup as scripts/setup_gpu_env.sh.
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.27.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install "nvidia-cudnn-cu12>=9,<10"

# onnxruntime-gpu 1.27.0's CUDA-12 build also needs libcudart.so.12 (the CUDA
# Runtime library - distinct from cuBLAS/cuDNN above) on the loader path.
# Found missing on a fresh Python 3.13 Colab image (2026-10-13) even with
# cuDNN/cuBLAS resolving fine - see setup_gpu_env.sh's identical fix for the
# full explanation (this script duplicates that install block rather than
# calling it).
# `|| true` matters here too - see setup_gpu_env.sh's identical fix/comment
# for why a bare `find /` under `set -e pipefail` dies silently otherwise.
cudart_path="$(find / -name 'libcudart.so.12*' 2>/dev/null | head -1)" || true
if [ -n "$cudart_path" ]; then
  export LD_LIBRARY_PATH="$(dirname "$cudart_path"):${LD_LIBRARY_PATH:-}"
fi

# Real GPU check - a silent CPU fall-back is invisible to
# get_available_providers() (it lists build-time providers, not runtime loads).
python3 scripts/_verify_gpu.py

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns=['index.json', 'xml/*'])
"

# Only qdrant_storage is wiped - embed_cache.db is a persistent, content-
# addressed embedding cache (SQLite) that must survive across runs (and across
# machines) to skip re-embedding unchanged text on future delta ingests.
rm -rf qdrant_storage
lex-au-search ingest --corpus-dir corpus/ --storage-dir ./qdrant_storage --cache-path ./embed_cache.db

zip -qr qdrant_storage.zip qdrant_storage
zip -q embed_cache.zip embed_cache.db
echo "Done. qdrant_storage.zip and embed_cache.zip are ready to download."
