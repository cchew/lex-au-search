#!/usr/bin/env bash
# Manual full re-ingest of the lex-au corpus on one Colab GPU runtime, from a pinned corpus
# revision and the published embedding cache, ending in a verified index you publish yourself.
# Run from the root of a fresh lex-au-search clone. See docs/colab-reingest-runbook.md.
#
#   bash scripts/colab_reingest.sh setup     # GPU env + RAM check
#   bash scripts/colab_reingest.sh fetch     # pinned corpus + embed_cache.db from HF
#   bash scripts/colab_reingest.sh ingest    # full ingest (long; run it in the background)
#   bash scripts/colab_reingest.sh verify    # counts + golden queries gate
#   bash scripts/colab_reingest.sh publish   # push index + cache to HF (needs HF_TOKEN, write)
#
# Cache-warm route (a standard 12 GB runtime is enough; the index is built later on a machine with RAM):
#   bash scripts/colab_reingest.sh setup-gpu   # GPU env only, no RAM check
#   bash scripts/colab_reingest.sh fetch       # (as above)
#   bash scripts/colab_reingest.sh warm        # embed every cache miss into embed_cache.db (background)
#   bash scripts/colab_reingest.sh push-cache  # upload embed_cache.db to HF (needs HF_TOKEN, write)
#
# Env:
#   CORPUS_REV       required for fetch/verify: full commit sha of cchew/lex-au to ingest
#   PREVIOUS_CHUNKS  chunk count of the published index, for the 5% drop guard (default 554750)
#   MIN_RAM_GB       setup refuses to continue below this (default 24)
#   HF_TOKEN         write token scoped to the lex-au-search repos; publish only
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/.."

step="${1:-}"
CORPUS_REPO="cchew/lex-au"
CACHE_REPO="cchew/lex-au-search-embed-cache"
PREVIOUS_CHUNKS="${PREVIOUS_CHUNKS:-554750}"
MIN_RAM_GB="${MIN_RAM_GB:-24}"

# onnxruntime's CUDA-12 build needs libcudart.so.12, which pip installs under site-packages but never
# puts on the loader path. setup_gpu_env.sh exports it for its own shell only, so every step that runs
# onnxruntime has to find it again (a missing library surfaces as "libcudart.so.12: cannot open shared
# object file" on import). `|| true`: find exits non-zero on unreadable dirs under /proc.
cuda_env() {
  local lib
  lib="$(find / -name 'libcudart.so.12*' 2>/dev/null | head -1)" || true
  if [ -n "$lib" ]; then
    export LD_LIBRARY_PATH="$(dirname "$lib"):${LD_LIBRARY_PATH:-}"
    echo "CUDA runtime: $lib"
  else
    echo "WARNING: libcudart.so.12 not found; onnxruntime will fail to import its CUDA provider" >&2
  fi
}

need_rev() {
  : "${CORPUS_REV:?set CORPUS_REV to the full commit sha of $CORPUS_REPO (see the runbook)}"
  case "$CORPUS_REV" in
    [0-9a-f]*) [ "${#CORPUS_REV}" -eq 40 ] || { echo "CORPUS_REV must be a full 40-character sha" >&2; exit 2; } ;;
    *) echo "CORPUS_REV must be a commit sha, not a branch name" >&2; exit 2 ;;
  esac
}

case "$step" in
  setup)
    ram_gb="$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)"
    echo "RAM: ${ram_gb} GB"
    if [ "$ram_gb" -lt "$MIN_RAM_GB" ]; then
      echo "Only ${ram_gb} GB RAM. A full ingest holds the whole collection in memory:" >&2
      echo "switch to a High-RAM runtime (Runtime > Change runtime type) or use the sharded path." >&2
      exit 3
    fi
    bash scripts/setup_gpu_env.sh
    ;;

  setup-gpu)
    bash scripts/setup_gpu_env.sh
    ;;

  fetch)
    need_rev
    python3 - <<PY
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

head = HfApi().list_repo_commits("$CORPUS_REPO", repo_type="dataset")[0].commit_id
print("corpus head now:", head)
print("ingesting      :", "$CORPUS_REV")
if head != "$CORPUS_REV":
    print("NOTE: head has moved since the revision you pinned; check the new commits before continuing.")
snapshot_download(
    repo_id="$CORPUS_REPO", repo_type="dataset", revision="$CORPUS_REV",
    local_dir="corpus", allow_patterns=["index.json", "xml/*"],
)
hf_hub_download(repo_id="$CACHE_REPO", repo_type="dataset", filename="embed_cache.db", local_dir=".")
PY
    python3 - <<'PY'
import json
from pathlib import Path

idx = json.loads(Path("corpus/index.json").read_text())
xml = list(Path("corpus/xml").glob("*.xml"))
print(f"corpus: {len(idx['acts'])} Acts in index, {len(xml)} xml files, structure_version {idx.get('structure_version')}")
assert len(xml) >= len(idx["acts"]), "xml files missing for some indexed Acts"
assert idx.get("structure_version") == 4, "corpus is not structure version 4"
PY
    ls -lh embed_cache.db
    ;;

  ingest)
    [ -d corpus/xml ] && [ -f embed_cache.db ] || { echo "run fetch first" >&2; exit 2; }
    cuda_env
    rm -rf qdrant_storage
    date -u +"ingest start %FT%TZ"
    lex-au-search ingest --corpus-dir corpus/ --storage-dir ./qdrant_storage --cache-path ./embed_cache.db
    date -u +"ingest end   %FT%TZ"
    ;;

  verify)
    [ -d qdrant_storage ] || { echo "no qdrant_storage; run ingest first" >&2; exit 2; }
    python3 scripts/report_index_counts.py --storage-dir ./qdrant_storage
    python3 scripts/verify_index.py \
      --storage-dir ./qdrant_storage \
      --previous-chunks "$PREVIOUS_CHUNKS" \
      --corpus-index corpus/index.json
    echo "verify OK. Nothing has been published."
    ;;

  warm)
    [ -d corpus/xml ] && [ -f embed_cache.db ] || { echo "run fetch first" >&2; exit 2; }
    cuda_env
    python3 scripts/_verify_gpu.py
    date -u +"warm start %FT%TZ"
    python3 scripts/colab_embed_misses.py --corpus-dir corpus/ --cache-path ./embed_cache.db
    date -u +"warm end   %FT%TZ"
    ;;

  push-cache)
    : "${HF_TOKEN:?set HF_TOKEN (write token for the lex-au-search repos)}"
    [ -f embed_cache.db ] || { echo "no embed_cache.db" >&2; exit 2; }
    python3 - <<PY
import os
from huggingface_hub import HfApi

HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="embed_cache.db", path_in_repo="embed_cache.db",
    repo_id="$CACHE_REPO", repo_type="dataset",
    commit_message="embed cache warmed for tail windows ($CORPUS_REPO@${CORPUS_REV:-unknown})",
)
print("embed cache uploaded")
PY
    ;;

  publish)
    : "${HF_TOKEN:?set HF_TOKEN (write token for the lex-au-search repos)}"
    [ -d qdrant_storage ] || { echo "no qdrant_storage" >&2; exit 2; }
    counts="$(python3 scripts/report_index_counts.py --storage-dir ./qdrant_storage)"
    act_count="$(echo "$counts" | sed -n 's/^act_count=//p')"
    chunk_count="$(echo "$counts" | sed -n 's/^chunk_count=//p')"
    model="$(python3 -c 'from lexausearch.models import DENSE_MODEL; print(DENSE_MODEL)')"
    echo "publishing: $act_count Acts, $chunk_count chunks, model $model"
    python3 -m lexausearch.hf_cache push-index \
      --storage-dir ./qdrant_storage --model "$model" \
      --act-count "$act_count" --chunk-count "$chunk_count" --token "$HF_TOKEN"
    python3 - <<PY
import os
from huggingface_hub import HfApi

HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="embed_cache.db", path_in_repo="embed_cache.db",
    repo_id="$CACHE_REPO", repo_type="dataset",
    commit_message="embed cache after full re-ingest of $CORPUS_REPO@${CORPUS_REV:-unknown}",
)
print("embed cache uploaded")
PY
    ;;

  *)
    sed -n '2,24p' "$0"
    exit 2
    ;;
esac
