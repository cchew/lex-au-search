# lex-au-search

Hybrid semantic search and natural language query over Australian Commonwealth legislation.

**Status: v0.4.1** - localhost only with no external APIs. Embedding model: `BAAI/bge-base-en-v1.5` (local ONNX, 768-dim, 512-token context).

## Uses / used by

- **Depends on:** [lex-au](https://github.com/cchew/lex-au) (AKN 3.0 XML corpus as the input to ingest)
- **Related:** [lex-au-graph](https://github.com/cchew/lex-au-graph) (for queries about how Acts relate to each other)
- **Used by:** [ClauseKit](https://github.com/cchew/clause-kit) (run claims against extracted legislation rules, grounded back to source clauses), term-comparison (compare how terms are defined across Acts)

Full stack map: this repo's [`STACK.md`](STACK.md) and [lex-au's `FUTURE.md`](https://github.com/cchew/lex-au/blob/main/FUTURE.md).

## Versions

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

The ingest command embeds sections using a local ONNX model (~270 MB, downloaded once to `~/.cache/fastembed/`). With the current 11-Act corpus (~4000+ sections after v0.2.0 rebuild) it takes roughly 60-90 minutes on Apple Silicon and uses significant RAM while running.

**Checklist before starting:**

- [ ] Close other heavy apps (browser tabs, VS Code, etc.) to free RAM - macOS will page-out if memory is tight
- [ ] Run ingest in a Terminal window, not inside Claude Code - the sandbox has a 10-minute process limit
- [ ] `qdrant_storage/` must not already exist from a partial run - if it does, delete it and re-ingest from scratch
- [ ] The lex-au corpus must be built first (see step 1 below); confirm `corpus/xml/*.xml` files are present
- [ ] Set `LEXAU_SEARCH_STORAGE` to the absolute path of `qdrant_storage/` before wiring up the MCP server

**First-run model download:** On first ingest, FastEmbed downloads `BAAI/bge-base-en-v1.5` and `Qdrant/bm25` to `~/.cache/fastembed/` (~135 MB total). Subsequent runs skip the download.

**Resuming after interruption:** Qdrant local storage is not transactional at the ingest level - a partial run leaves a corrupt collection. Delete `qdrant_storage/` and re-run `ingest` from scratch.

---

## Quickstart

**1. Get the lex-au corpus.** Download the pre-built XML from Hugging Face (no lex-au clone needed):

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cchew/lex-au', repo_type='dataset', local_dir='corpus', allow_patterns='xml/*')"
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
# Takes 60-90 min on Apple Silicon; close other apps first
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
      "heading": "Interference with privacy of an individual",
      "text": "...",
      "score": 0.847
    }
  ]
}
```

## Known limits

- Index is stale against the current lex-au corpus - built against an 11/12-Act snapshot; [lex-au](https://github.com/cchew/lex-au) has since expanded to 539+ Acts. Re-ingest required before search results reflect the full corpus.
- No auth on HTTP API (local use only)
- `get_act_sections` and `get_act_text` return full Act content - responses exceed LLM context limits for any non-trivial Act; use `search_legislation` for NL queries
- Social Security (Administration) Act 1999 not indexed - notification obligation provisions fall back to model training data

## Under consideration

- **Embedding performance:** local ONNX ingest (bge-base-en-v1.5) takes 60-90 min for the current 11-Act corpus on Apple Silicon. OpenAI `text-embedding-3-small` API would likely reduce wall-clock time significantly at the cost of an API dependency and per-token cost; trade-off to evaluate for v0.5.0 when corpus expands.
- **Query-side RAG tuning:** Santander AI's [linear-adapter-trainer](https://github.com/Santander-AI/linear-adapter-trainer) (Apache 2.0) applies a lightweight linear projection on top of a frozen embedding model to close the query-document distribution gap without retraining the embedder. Worth evaluating if retrieval precision drops as the corpus grows; the adapter can be trained on AU legislative query-passage pairs derived from existing MCP session logs.
