# lex-au-search

Hybrid semantic search over Australian Commonwealth legislation -- Layer 2 of the AU Legislative Intelligence Stack.

**Status: v0.1.0** -- local-first, no Docker, no external APIs.

## Stack position

```
Layer 1: lex-au        -- AKN 3.0 XML corpus
Layer 2: lex-au-search -- hybrid search API + MCP  <- this project
Layer 3: ClauseKit     -- rule extraction JSON
Agent demo             -- NL query -> attributed answer (future)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before full ingest

The ingest command embeds ~2900 sections using a local ONNX model (~270 MB, downloaded once to `~/.cache/fastembed/`). It takes roughly 60-90 minutes on Apple Silicon and uses significant RAM while running.

**Checklist before starting:**

- [ ] Close other heavy apps (browser tabs, VS Code, etc.) to free RAM — macOS will page-out if memory is tight
- [ ] Run ingest in a Terminal window, not inside Claude Code — the sandbox has a 10-minute process limit
- [ ] `qdrant_storage/` must not already exist from a partial run — if it does, delete it and re-ingest from scratch
- [ ] The lex-au corpus must be built first (see step 1 below); confirm `corpus/xml/*.xml` files are present
- [ ] Set `LEXAU_SEARCH_STORAGE` to the absolute path of `qdrant_storage/` before wiring up the MCP server

**First-run model download:** On first ingest, FastEmbed downloads `nomic-ai/nomic-embed-text-v1.5` and `Qdrant/bm25` to `~/.cache/fastembed/` (~270 MB total). Subsequent runs skip the download.

**Resuming after interruption:** Qdrant local storage is not transactional at the ingest level — a partial run leaves a corrupt collection. Delete `qdrant_storage/` and re-run `ingest` from scratch.

---

## Quickstart

**1. Build the lex-au corpus first** (see [lex-au](https://github.com/cchew/lex-au)):

```bash
# Run from inside the lex-au repo
lexau build --all --corpus-dir corpus/
```

**2. Open a Terminal (not Claude Code) and index the corpus:**

```bash
cd /path/to/lex-au-search/repo
source .venv/bin/activate
lex-au-search ingest --corpus-dir ../lex-au/repo/corpus/
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
      "section_num": "16",
      "heading": "Interference with privacy of an individual",
      "text": "...",
      "score": 0.847
    }
  ]
}
```

## Known limits (v0.1.0)

- Schedule content (e.g. Privacy Act Schedule 1 -- APPs) not indexed (lex-au v0.1.0 gap; resolved in lex-au v0.1.1)
- No auth on HTTP API (local use only)
- Section-level citations only; subsection-level requires lex-au v0.1.1 + re-ingest
