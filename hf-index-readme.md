---
license: cc-by-4.0
language:
- en
tags:
- law
- legislation
- australia
- vector-search
- embeddings
- qdrant
- rag
- akn
pretty_name: lex-au-search-index - Vector Index for AU Legislation Search
---

# lex-au-search-index

**Pre-built Qdrant vector index for hybrid dense + BM25 search over Australian Commonwealth legislation** - download and query, no re-embedding or re-ingest required.

This is the finished output of [lex-au-search](https://github.com/cchew/lex-au-search) run against the [lex-au](https://huggingface.co/datasets/cchew/lex-au) corpus. If you just want to search AU legislation, download this and point the CLI at it - you don't need to run the ingest pipeline yourself.

**Version: v0.5.0** - 534,335 chunks (sections, subsections, schedule clauses) across 3,073 Act records, embedded with `snowflake/snowflake-arctic-embed-l` (1024-dim, local ONNX, 512-token context). Corpus pinned at [lex-au commit `f0a89da4`](https://huggingface.co/datasets/cchew/lex-au/commit/f0a89da476227b5f3dbb861be1b666d1234ac501) (2026-08-30).

See [github.com/cchew/lex-au-search](https://github.com/cchew/lex-au-search) for source code, CLI, and MCP server. See [`cchew/lex-au-search-embed-cache`](https://huggingface.co/datasets/cchew/lex-au-search-embed-cache) for the separate raw embedding cache the ingest pipeline itself uses to resume/skip re-embedding - that's a different artifact for a different audience (the pipeline, not a searcher).

## Quick example

Download the index and run the search server directly - no embedding model needed to browse:

```python
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient

path = snapshot_download(repo_id="cchew/lex-au-search-index", repo_type="dataset")
client = QdrantClient(path=f"{path}/qdrant_storage")

# Browse Act-level metadata with no query embedding required
hits, _ = client.scroll("legislation", limit=3, with_payload=True)
for h in hits:
    print(h.payload["act_name"], h.payload["frbr_uri"])
```

For real hybrid dense + BM25 search (`search_legislation`), clone [lex-au-search](https://github.com/cchew/lex-au-search), `pip install -e ".[dev]"`, then:

```bash
lex-au-search serve --storage-dir /path/to/downloaded/qdrant_storage
# or: LEXAU_SEARCH_STORAGE=/path/to/downloaded/qdrant_storage lex-au-search mcp
```

## What's in this dataset

A Qdrant local-mode storage directory (`qdrant_storage/`) with two collections:

- **`legislation`** (3,073 points) - one per Act: `act_name`, `frbr_uri`, `year`, `as_at_date`, `section_count`, `schedule_clause_count`.
- **`legislation_section`** (534,335 points) - one per chunk: 1024-dim dense vector (`snowflake-arctic-embed-l`) + BM25 sparse vector, payload `act_name`, `frbr_uri`, `eid`, `provision_num`, `provision_type`, `heading`, `text` (full chunk text), `refs` (cross-references).

On-disk size ~6.4 GB, dominated by `legislation_section` (vectors + full section text payload).

## Known limits

- 9 of the source corpus's 3,082 Acts are absent: two Constitution Alteration referendum Acts (1906, 1909) and seven old amendment/repeal-only Acts whose AKN XML doesn't fit the `<section>`-based body shape the chunker extracts from, so they chunk to zero content. All are spent/minor Acts with no operative text of real search value - see [lex-au-search's FUTURE.md](https://github.com/cchew/lex-au-search/blob/main/FUTURE.md) for the full list.
- Qdrant's local/embedded mode keeps the full vector set resident in RAM and has no ANN index, so `QdrantClient(path=...)` construction itself is slow at this scale (multiple minutes to open, 534K+ points) - expected, not a bug. Fine for local CLI/MCP use; not suited to serving concurrent requests.
- No hosted search endpoint exists for this index - it's a downloadable artifact, not a live service.

## Licence

CC BY 4.0. Source legislation is Crown copyright - Commonwealth of Australia; reproduction permitted for non-commercial and research purposes under the [PSI Framework](https://www.legislation.gov.au/Help/Copyright). The `legislation_section` payload carries substantial verbatim section text from that corpus, so this dataset inherits the same licence as [`cchew/lex-au`](https://huggingface.co/datasets/cchew/lex-au) rather than a separate one for "just the vectors."

## What's built on this

- [ClauseKit](https://github.com/cchew/clause-kit) - LLM extraction of evaluatable rules (JSON Logic) from Acts, run claims against them grounded back to source clauses
- term-comparison ("Act Alike") - compare how terms are defined across Acts

## Related

- [github.com/cchew/lex-au-search](https://github.com/cchew/lex-au-search) - source code, CLI, MCP server
- [cchew/lex-au](https://huggingface.co/datasets/cchew/lex-au) - the underlying AKN 3.0 XML corpus this index was built from
- [cchew/lex-au-search-embed-cache](https://huggingface.co/datasets/cchew/lex-au-search-embed-cache) - the ingest pipeline's own resumable embed cache (different artifact, different audience)
- [lex-au-graph](https://github.com/cchew/lex-au-graph) - cross-reference graph and definition resolution across Acts
