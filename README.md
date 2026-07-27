# lex-au-search

Hybrid semantic search and natural language query over Australian Commonwealth legislation.

**Status: v0.4.5** - localhost only with no external APIs. Embedding model: `BAAI/bge-base-en-v1.5` (local ONNX, 768-dim, 512-token context).

## Uses / used by

- **Depends on:** [lex-au](https://github.com/cchew/lex-au) (AKN 3.0 XML corpus as the input to ingest)
- **Related:** [lex-au-graph](https://github.com/cchew/lex-au-graph) (for queries about how Acts relate to each other)
- **Used by:** [ClauseKit](https://github.com/cchew/clause-kit) (run claims against extracted legislation rules, grounded back to source clauses), term-comparison (compare how terms are defined across Acts)

Full stack map: this repo's [`STACK.md`](STACK.md) and [lex-au's `FUTURE.md`](https://github.com/cchew/lex-au/blob/main/FUTURE.md).

## Versions

- **v0.4.5** - Attempted fix for the Colab OOM on full-corpus ingest: set `on_disk=True` / `always_ram=False` on the `legislation_section` collection's vector, sparse, HNSW, and quantization config. Confirmed ineffective (2026-07-27, same crash point, identical RAM curve) - Qdrant's local/embedded Python client never implements on-disk vector storage regardless of config; `local_collection.py` always materializes the full vector set as an in-memory numpy array, and `on_disk` is accepted by the API for schema compatibility but never read by local mode's storage engine. Root cause is architectural (local mode itself), not a config issue - see Known limits. Left in place anyway since it's the correct config for a real Qdrant server, if the ingest architecture ever moves off local mode.
- **v0.4.4** - Embedding cache (`--cache-path`) rebacked by SQLite instead of a second Qdrant collection, after a Colab GPU ingest OOM'd near the end of the full corpus (confirmed 2026-07-24: Qdrant's own local-mode client warns it's unsuitable above ~20K points; measured ~25x the peak memory of SQLite for the same 100K-vector dataset, independent of HNSW settings). Cache is now a single file (`./embed_cache.db` by default, was a directory).
- **v0.4.3** - `ingest` gains a persistent, content-addressed embedding cache that survives across runs even though `--storage-dir` is still fully rebuilt each time - re-ingesting an updated corpus now skips re-embedding unchanged Acts' text. See "Delta ingest via the embedding cache".
- **v0.4.2** - `ingest` auto-detects a CUDA GPU (via `onnxruntime.get_available_providers()`) and uses it if present, falling back to CPU otherwise - same command either way. Install the `gpu` extra to enable it. Added `scripts/colab_ingest.sh` for running ingest on a free Colab GPU runtime.
- **v0.4.1** - MCP tool description AX improvements, `STACK.md` discovery doc, fixed stale version string.
- **v0.4.0** - `client.query_points()` migration, paragraph-level chunking, embedding cache, switched to `BAAI/bge-base-en-v1.5`.
- **v0.3.0** - Two-collection Qdrant, schedule clause chunking, AU cross-reference extraction, INT8 quantisation, FastMCP auto-exposure.
- **v0.1.0** - Local hybrid search (dense + BM25), FastAPI, MCP stdio.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before full ingest

The ingest command embeds sections using a local ONNX model (~270 MB, downloaded once to `~/.cache/fastembed/`). CPU-only ingest does not scale well past a few hundred Acts - budget on the order of a day or more for the full corpus (2,900+ Acts, 500,000+ chunks) on a laptop CPU, and significant RAM while running.

**GPU acceleration:** `ingest` auto-detects an available CUDA GPU and uses it transparently - no flag, no separate command. To enable it, install the `gpu` extra (`pip install -e ".[gpu]"`) on a CUDA-capable machine, which pulls in `onnxruntime-gpu` instead of the CPU-only `onnxruntime`; the two packages conflict, so uninstall the CPU one first (`pip uninstall onnxruntime && pip install -e ".[gpu]"`). Without a CUDA GPU (or without the `gpu` extra), `ingest` runs on CPU exactly as before - nothing to configure. This makes it practical to run the same `lex-au-search ingest` command on a free Colab GPU runtime, a rented GPU cloud instance, or a plain laptop, whichever you have.

**Checklist before starting:**

- [ ] Close other heavy apps (browser tabs, VS Code, etc.) to free RAM - macOS will page-out if memory is tight
- [ ] Run ingest in a Terminal window, not inside Claude Code - the sandbox has a 10-minute process limit
- [ ] `qdrant_storage/` must not already exist from a partial run - if it does, delete it and re-ingest from scratch
- [ ] The lex-au corpus must be built first (see step 1 below); confirm `corpus/xml/*.xml` files are present
- [ ] Set `LEXAU_SEARCH_STORAGE` to the absolute path of `qdrant_storage/` before wiring up the MCP server

**First-run model download:** On first ingest, FastEmbed downloads `BAAI/bge-base-en-v1.5` and `Qdrant/bm25` to `~/.cache/fastembed/` (~135 MB total). Subsequent runs skip the download.

**Resuming after interruption:** Qdrant local storage is not transactional at the ingest level - a partial run leaves a corrupt collection. Delete `qdrant_storage/` and re-run `ingest` from scratch. This does **not** apply to `--cache-path` (default `./embed_cache.db`) - leave that in place, see "Delta ingest" below.

### GPU ingest via Colab

`scripts/colab_ingest.sh` wraps the full sequence - install the `gpu` extra, download the lex-au corpus from Hugging Face, run `ingest`, zip the result - for a fresh GPU runtime. Paste these cells into a Colab notebook with a GPU runtime (Runtime > Change runtime type > T4 GPU):

```python
!git clone https://github.com/cchew/lex-au-search.git
%cd lex-au-search
!bash scripts/colab_ingest.sh
```

**Do not use `google.colab.files.download()` to fetch `qdrant_storage.zip`** - it's a browser-mediated transfer that silently truncates/corrupts files in the multi-GB range (confirmed 2026-07-15: bad CRC on the largest collection, undetected until a post-transfer integrity check). Mount Google Drive and copy it there instead, then download from Drive normally (chunked, resumable, no silent corruption):

```python
from google.colab import drive
drive.mount("/content/drive")
!cp qdrant_storage.zip "/content/drive/MyDrive/qdrant_storage.zip"
!cp embed_cache.zip "/content/drive/MyDrive/embed_cache.zip"
```

Then download both zips from Drive via its web UI or `drive.google.com`. Unzip `qdrant_storage.zip` into `lex-au-search/repo/qdrant_storage/` and `embed_cache.zip` into `lex-au-search/repo/` (producing `embed_cache.db`). Verify each transfer before trusting it: `unzip -t <file>.zip` should report no CRC errors for both.

Free-tier Colab sessions disconnect after ~12 hours of runtime or ~90 minutes idle - for a corpus this size, the GPU path should comfortably finish inside one session, but if it doesn't, re-run `scripts/colab_ingest.sh` in a fresh session (it deletes and rebuilds `qdrant_storage/` from scratch each time - there is no incremental resume for the search index itself). Transfer `embed_cache.zip` via Drive the same way, so a retried session doesn't re-pay for embeddings it already computed.

**Out-of-memory risk on the full corpus:** an early version of the embedding cache (v0.4.3) stored it as a second Qdrant collection, which OOM-killed a Colab run near the end of a full-corpus ingest (confirmed 2026-07-24) - Qdrant's local/embedded mode isn't built for datasets this size regardless of index settings. v0.4.4 replaced it with SQLite, which is ~25x lighter for this access pattern (pure id-keyed lookup, never vector search). If `ingest` still gets `Killed` with no traceback on a free-tier Colab instance, that's the kernel OOM killer - check `!free -h` beforehand and consider a higher-RAM Colab tier.

### Delta ingest via the embedding cache

`ingest` always re-chunks and re-upserts every Act in `--corpus-dir` into `--storage-dir` - there's no "only process changed Acts" flag, and `--storage-dir` should still be deleted before every run per the resuming-after-interruption note above. What's incremental is the *embedding* step: `--cache-path` (default `./embed_cache.db`) is a separate, persistent SQLite file, keyed by `UUID5(chunk_text)` - a content hash of the exact text being embedded. It is never deleted by `ingest` or `colab_ingest.sh`. Re-running `ingest` against a corpus where most Acts are unchanged produces identical chunk text for those Acts, which hits the cache and skips the (expensive) embedding call entirely - chunking and upserting still happen for every Act, but that's cheap. `ingest`'s final output line reports hit/miss counts so you can see the delta at a glance.

This means the practical pattern for re-ingesting after a small corpus update is: keep `embed_cache.db` from your last run (don't delete it, unlike `qdrant_storage/`), point `--cache-path` at it, and re-run `ingest` against the current corpus. On a persistent machine this needs no extra steps - the cache just accumulates on disk across runs. Moving the cache to a new machine (e.g. bootstrapping a laptop from a one-off Colab GPU run) just needs `embed_cache.db` copied over once, same as `qdrant_storage/`.

---

## Quickstart

**1. Get the lex-au corpus.** Download the pre-built XML from Hugging Face (no lex-au clone needed):

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns=['index.json', 'xml/*'])"
```

Or build it from source if you need Acts not yet published, or are modifying lex-au itself - see [lex-au](https://github.com/cchew/lex-au):

```bash
# Run from inside the lex-au repo
lexau build --all --corpus-dir corpus/
```

**2. Open a Terminal (not Claude Code) and index the corpus:**

```bash
cd /path/to/lex-au-search/repo
source .venv/bin/activate
lex-au-search ingest --corpus-dir corpus/   # or ../lex-au/repo/corpus/ if built from source
# CPU: a day or more for the full corpus. Faster on a CUDA GPU (auto-detected) - see "Before full ingest"
```

**3. Search:**

```bash
# HTTP API
lex-au-search serve
curl "http://localhost:8000/search?q=notification+obligations+data+breach&limit=5"

# MCP (Claude Code)
# Add to your Claude Code settings.json -- see below
```

## MCP Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "lex-au-search": {
      "command": "lex-au-search",
      "args": ["mcp"],
      "env": {
        "LEXAU_SEARCH_STORAGE": "/absolute/path/to/lex-au-search/repo/qdrant_storage"
      }
    }
  }
}
```

Then ask Claude: *"What are my notification obligations if an AI system causes harm to a consumer?"*

## API

```
GET /search?q=<query>&limit=10&act=<optional act name>

Response:
{
  "results": [
    {
      "act_name": "Privacy Act 1988",
      "frbr_uri": "/akn/au/act/1988/119/eng@2026-01-01",
      "eid": "part-III__sec-16",
      "provision_num": "16",
      "provision_type": "section",
      "heading": "Interference with privacy of an individual",
      "text": "...",
      "refs": [],
      "score": 0.847
    }
  ]
}
```

## Known limits

- No auth on HTTP API (local use only)
- `get_act_sections` and `get_act_text` return full Act content - responses exceed LLM context limits for any non-trivial Act; use `search_legislation` for NL queries
- Index covers 2,394 Acts from a 2026-07-16 Colab ingest, against lex-au's corpus at that time (2,942 Acts) - ~550-Act gap not yet diagnosed
- Full re-ingest against lex-au's current corpus (3,078 Acts as of v0.7.5, 2026-07-22) OOM-kills on Colab free tier (12.7GB) at the same point every run (~2,920 of ~3,078 Acts processed). Root cause confirmed 2026-07-27: Qdrant's local/embedded mode always keeps the full vector set resident in RAM, with no working on-disk option. Next fix is a sharded ingest (batch Acts into separate local collections, merge after) - not yet built.

## Under consideration

- **Embedding performance:** CPU ingest of the full corpus (2,900+ Acts) takes on the order of a day or more on a laptop CPU; v0.4.2's CUDA auto-detect (see "Before full ingest") is the current mitigation. OpenAI `text-embedding-3-small` API would also cut wall-clock time at the cost of an API dependency and per-token cost - not pursued while the free-GPU path covers it.
- **Query-side RAG tuning:** Santander AI's [linear-adapter-trainer](https://github.com/Santander-AI/linear-adapter-trainer) (Apache 2.0) applies a lightweight linear projection on top of a frozen embedding model to close the query-document distribution gap without retraining the embedder. Worth evaluating if retrieval precision drops as the corpus grows; the adapter can be trained on AU legislative query-passage pairs derived from existing MCP session logs.
