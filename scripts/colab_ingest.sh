#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a freshly cloned lex-au-search repo, on a GPU runtime
# (e.g. Google Colab, or any CUDA-capable box). Installs the gpu extra,
# downloads the lex-au corpus from Hugging Face, runs the ingest, and zips
# the result for download. See README.md's "GPU ingest via Colab" section
# for the notebook cells that call this script.

pip uninstall -y onnxruntime >/dev/null 2>&1 || true
pip install -e ".[gpu]"
pip install huggingface_hub

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns='xml/*')
"

rm -rf qdrant_storage
lex-au-search ingest --corpus-dir corpus/ --storage-dir ./qdrant_storage

zip -qr qdrant_storage.zip qdrant_storage
echo "Done. qdrant_storage.zip is ready to download."
