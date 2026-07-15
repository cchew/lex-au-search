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
# transitive dependency -- it can't be excluded from the same pip install as
# the gpu extra, since nothing tells pip the two packages are mutually
# exclusive for the same "onnxruntime" import namespace. Whichever installs
# its files last wins, so uninstall the CPU one and force-reinstall
# onnxruntime-gpu (--no-deps so this step can't pull plain onnxruntime back
# in) *after* everything else, not before.
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.20"

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
snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns='xml/*')
"

rm -rf qdrant_storage
lex-au-search ingest --corpus-dir corpus/ --storage-dir ./qdrant_storage

zip -qr qdrant_storage.zip qdrant_storage
echo "Done. qdrant_storage.zip is ready to download."
