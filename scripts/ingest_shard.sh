#!/usr/bin/env bash
set -euo pipefail

# Per-shard ingest work for ONE shard only (an Act-count slice), so each
# shard gets an isolated VM and RAM never accumulates across the whole
# corpus in one process. See
# docs/superpowers/specs/2026-07-27-colab-sharded-ingest-design.md.
#
# Env setup (pip, onnxruntime-gpu, CUDA check) is scripts/setup_gpu_env.sh -
# run that once per box before this. Colab runs it per fresh VM; RunPod runs
# it once in RunPodBackend.prepare().
#
# Usage: bash scripts/ingest_shard.sh SHARD_INDEX SHARD_SIZE [SEED_DB_PATH]

SHARD_INDEX="$1"
SHARD_SIZE="$2"
SEED_DB_PATH="${3:-}"

# Download only index.json first, then only THIS shard's XML files - not
# the whole ~3,078-Act corpus. Found running the real 5-Act smoke test
# (2026-08-04): downloading the full corpus took 13+ minutes and hit HF
# rate-limiting even for a 5-Act shard; at production shard-size 300 across
# 11 shards, downloading the full corpus on every shard would mean ~11x
# redundant full-corpus downloads. lexausearch is already importable here
# (setup_gpu_env.sh installed it editable), so reuse the same shard-slicing
# logic ingest-shard itself uses, rather than re-deriving it.
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns=['index.json'])
"

python3 -c "
from pathlib import Path
from huggingface_hub import snapshot_download
from lexausearch.chunker import load_corpus_act_entries, shard_bounds

corpus_dir = Path('corpus')
entries = load_corpus_act_entries(corpus_dir)
start, end = shard_bounds(len(entries), $SHARD_INDEX, $SHARD_SIZE)
xml_paths = [e['xml_path'] for e in entries[start:end]]
print(f'Shard needs {len(xml_paths)} XML files (Acts [{start}:{end}] of {len(entries)})')
if xml_paths:
    snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns=xml_paths)
"

rm -rf shard_storage

# Optional seed DB from a prior attempt at THIS shard that got killed mid-run
# (confirmed 2026-08-21: free-tier sessions get pruned at ~60min regardless
# of RAM/GPU headroom or keep-alive health) - already-embedded chunks land as
# cache hits below and skip re-embedding entirely (see
# Indexer._upsert_chunks_with_cache). The caller passes an absolute path that
# lives outside the repo checkout so the `rm -rf repo` in the backend's
# remote_cmd (which recreates this checkout fresh every attempt) can't delete
# it out from under itself. Absent file = skip, not an error.
if [ -n "$SEED_DB_PATH" ] && [ -f "$SEED_DB_PATH" ]; then
    echo "Seeding shard_cache.db from $SEED_DB_PATH ..."
    cp "$SEED_DB_PATH" ./shard_cache.db
fi

lex-au-search ingest-shard \
    --corpus-dir corpus/ \
    --storage-dir ./shard_storage \
    --cache-path ./shard_cache.db \
    --shard-index "$SHARD_INDEX" \
    --shard-size "$SHARD_SIZE"

zip -qr shard_storage.zip shard_storage
zip -q shard_cache.zip shard_cache.db
echo "Done. shard_storage.zip and shard_cache.zip ready for colab_driver.py to download."
