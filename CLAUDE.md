# lex-au-search

Layer 2 of the AU Legislative Intelligence Stack. Hybrid search (dense + BM25 sparse) over the lex-au AKN 3.0 XML corpus.

## Stack position

Layer 1: lex-au (../lex-au/repo/) — AKN 3.0 XML corpus
Layer 2: lex-au-search (this repo) — hybrid search API + MCP
Layer 3: ClauseKit (../clause-kit/repo/) — rule extraction

## Setup

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

## CLI

lex-au-search ingest --corpus-dir ../lex-au/repo/corpus/
lex-au-search serve
lex-au-search mcp  # requires LEXAU_SEARCH_STORAGE env var

## Tests

pytest
