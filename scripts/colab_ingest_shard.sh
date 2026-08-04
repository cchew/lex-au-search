#!/usr/bin/env bash
set -euo pipefail

# Per-shard variant of colab_ingest.sh - run on a FRESH Colab GPU runtime
# for ONE shard only (an Act-count slice), so each shard gets an isolated
# VM and RAM never accumulates across the whole corpus in one process. See
# docs/superpowers/specs/2026-07-27-colab-sharded-ingest-design.md.
#
# Usage: bash scripts/colab_ingest_shard.sh SHARD_INDEX SHARD_SIZE

SHARD_INDEX="$1"
SHARD_SIZE="$2"

pip install -e ".[gpu]"
pip install huggingface_hub

# Same onnxruntime-gpu CUDA-12 fix as colab_ingest.sh - see that script's
# comment for why the plain pip extra alone silently falls back to CPU.
pip uninstall -y onnxruntime-gpu onnxruntime
pip install onnxruntime-gpu==1.27.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

python3 -c "
import onnxruntime
providers = onnxruntime.get_available_providers()
assert 'CUDAExecutionProvider' in providers, (
    f'CUDA not available after gpu extra install (providers: {providers}).'
)
print('CUDA available, providers:', providers)
"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns=['index.json', 'xml/*'])
"

rm -rf shard_storage
lex-au-search ingest-shard \
    --corpus-dir corpus/ \
    --storage-dir ./shard_storage \
    --cache-path ./shard_cache.db \
    --shard-index "$SHARD_INDEX" \
    --shard-size "$SHARD_SIZE"

zip -qr shard_storage.zip shard_storage
zip -q shard_cache.zip shard_cache.db
echo "Done. shard_storage.zip and shard_cache.zip ready for colab_driver.py to download."
