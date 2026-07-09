# lex-au-search

Retrieval layer of the AU Legislative Intelligence Stack. Hybrid search (dense + BM25 sparse) over the lex-au AKN 3.0 XML corpus.

## Stack position

Corpus: lex-au (../lex-au/repo/) — AKN 3.0 XML corpus
Retrieval: lex-au-search (this repo) — hybrid search API + MCP; lex-au-graph (../lex-au-graph/repo/) — cross-reference graph + definition resolution
Applications: ClauseKit (../clause-kit/repo/) — rule extraction; term-comparison (../term-comparison/repo/) — IM2026 definition-comparison bot

Call-order note: for queries about a term's meaning or cross-Act definition chains, check lex-au-graph before or alongside search here — this project's hybrid search can match the wrong Act's use of a homonymous term.

## Setup

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

## CLI

lex-au-search ingest --corpus-dir ../lex-au/repo/corpus/
lex-au-search serve
lex-au-search mcp  # requires LEXAU_SEARCH_STORAGE env var

## Tests

pytest
