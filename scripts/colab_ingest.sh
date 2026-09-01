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
# Reinstall the CUDA-12 build clean, and pin the cuDNN 9 wheel so preload_dlls()
# (in _verify_gpu.py and lexausearch.indexer) can resolve it regardless of what
# the runtime ships. Same setup as scripts/setup_gpu_env.sh.
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu>=1.20,<2" "nvidia-cudnn-cu12>=9,<10"

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
