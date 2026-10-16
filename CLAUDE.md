# lex-au-search

Retrieval layer of the AU Legislative Intelligence Stack. Hybrid search (dense + BM25 sparse) over the lex-au AKN 3.0 XML corpus.

## Stack position

Corpus: lex-au (../lex-au/repo/) — AKN 3.0 XML corpus
Retrieval: lex-au-search (this repo) — hybrid search API + MCP; lex-au-graph (../lex-au-graph/repo/) — cross-reference graph + definition resolution
Applications: ClauseKit (../clause-kit/repo/) — rule extraction; Act Alike (../act-alike/repo/) — term comparison across Acts (formerly term-comparison)

Call-order note: for queries about a term's meaning or cross-Act definition chains, check lex-au-graph before or alongside search here — this project's hybrid search can match the wrong Act's use of a homonymous term.

## Setup

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

## CLI

lex-au-search ingest --corpus-dir ../lex-au/repo/corpus/
# --storage-dir (default ./qdrant_storage) is fully rebuilt every run - delete before re-running.
# --cache-path (default ./embed_cache.db, SQLite) is a persistent embedding cache - never delete it;
# re-ingesting an updated corpus skips re-embedding unchanged Acts' text via this cache.
lex-au-search ingest-delta --corpus-dir ../lex-au/repo/corpus/ --storage-dir ./qdrant_storage
# Re-indexes only Acts whose content hash changed since the last ingest; prunes removed Acts.
# --storage-dir must already hold a prior ingest - stop serve/mcp first (exclusive lock).
lex-au-search gc-cache --corpus-dir ../lex-au/repo/corpus/ --cache-path ./embed_cache.db
# Deletes embed_cache.db rows for chunk text no longer in the corpus (storage bloat cleanup,
# not required for correctness). Run opportunistically after Acts are removed/edited.
lex-au-search serve
lex-au-search mcp  # requires LEXAU_SEARCH_STORAGE env var

# Sharded-ingest embed caches live on HF (cchew/lex-au-search-embed-cache).
# One-time: python scripts/bootstrap_hf_cache.py   (needs HF_CACHE_WRITE_TOKEN in .env)
# Status:   python scripts/check_ingest_status.py

# The merged qdrant_storage/ index also round-trips through HF (cchew/lex-au-search-index),
# generation-guarded via index-catalogue.json -- same pattern as the embed-cache master.
# Manual weekly refresh only (README "Weekly refresh via HF-roundtrip") -- the GitHub
# Actions automation was built and reverted: ingest-delta OOMs opening the existing
# 534K-point store on a standard hosted runner (Qdrant local mode always loads the
# full collection into RAM, regardless of how many Acts actually changed).
python -m lexausearch.hf_cache fetch-index --dest ./hf-index --expect-model <DENSE_MODEL>
python -m lexausearch.hf_cache push-index --storage-dir ./qdrant_storage --model <DENSE_MODEL> \
  --act-count <n> --chunk-count <n>

## Tests

pytest
