#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a freshly cloned lex-au-search repo, on a GPU runtime
# (e.g. Google Colab, or any CUDA-capable box). Installs the gpu extra,
# downloads the lex-au corpus from Hugging Face, runs the ingest, and zips
# the result for download. See README.md's "GPU ingest via Colab" section
# for the notebook cells that call this script.

pip install -e ".[gpu]"
pip install huggingface_hub

# The base install above pulls in plain CPU onnxruntime as fastembed's own
# transitive dependency, which then wins the "onnxruntime" import namespace
# over the gpu extra's onnxruntime-gpu (confirmed 2026-07-15: T4 attached and
# idle while ingest silently ran on CPU). The standard PyPI onnxruntime-gpu
# wheel also targets CUDA 11, not the CUDA 12 Colab's T4 runtime provides -
# pull the CUDA 12 build from Microsoft's ADO feed instead, uninstalling both
# packages first so neither's stale files are left behind.
pip uninstall -y onnxruntime-gpu onnxruntime
pip install onnxruntime-gpu==1.27.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

python3 -c "
import onnxruntime
providers = onnxruntime.get_available_providers()
assert 'CUDAExecutionProvider' in providers, (
    f'CUDA not available after gpu extra install (providers: {providers}). '
    'Check Runtime > Change runtime type has a GPU selected, and that '
    '!nvidia-smi shows a GPU attached.'
)
print('CUDA available, providers:', providers)
"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns=['index.json', 'xml/*'])
"

# Only qdrant_storage is wiped - embed_cache_storage is a persistent,
# content-addressed embedding cache that must survive across runs (and across
# machines) to skip re-embedding unchanged text on future delta ingests.
rm -rf qdrant_storage
lex-au-search ingest --corpus-dir corpus/ --storage-dir ./qdrant_storage --cache-dir ./embed_cache_storage

zip -qr qdrant_storage.zip qdrant_storage
zip -qr embed_cache_storage.zip embed_cache_storage
echo "Done. qdrant_storage.zip and embed_cache_storage.zip are ready to download."
