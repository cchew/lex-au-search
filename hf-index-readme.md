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
viewer: false
---

# lex-au-search-index

**Pre-built Qdrant vector index for hybrid dense + BM25 search over Australian Commonwealth legislation** - download and query, no re-embedding or re-ingest required.

This is an output of [lex-au-search](https://github.com/cchew/lex-au-search) run against the [lex-au](https://huggingface.co/datasets/cchew/lex-au) corpus. If you want to search AU legislation, download this and point the CLI at it.

**Version: v0.7.0, index generation 3** - 553,870 chunks (sections, subsections, schedule clauses, schedule text) across 3,086 Acts, embedded with `snowflake/snowflake-arctic-embed-l` (1024-dim, local ONNX, 512-token context). Corpus pinned at [lex-au commit `ce532dde`](https://huggingface.co/datasets/cchew/lex-au/commit/ce532ddeb514fa477d6466a892fa17e62d21513e) (lex-au v0.13.0 content, 2026-10-26). Built from [embed cache revision `c8a24dfc`](https://huggingface.co/datasets/cchew/lex-au-search-embed-cache/commit/c8a24dfcc04af3ecd703104f0223f1c36832a895). Earlier generations stay reachable by revision: generation 2 (552,305 chunks, 3,085 Acts) is `461bc8aa4ae8bbbe0f0a28fcc5485f1376451f21`.

See [github.com/cchew/lex-au-search](https://github.com/cchew/lex-au-search) for source code, CLI, and MCP server. See [`cchew/lex-au-search-embed-cache`](https://huggingface.co/datasets/cchew/lex-au-search-embed-cache) for the separate raw embedding cache the ingest pipeline itself uses to resume/skip re-embedding.

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

- **`legislation`** (3,086 points) - one per Act: `act_name`, `frbr_uri`, `year`, `as_at_date`, `section_count`, `schedule_clause_count`.
- **`legislation_section`** (553,870 points) - one per chunk: 1024-dim dense vector (`snowflake-arctic-embed-l`) + BM25 sparse vector, payload `act_name`, `frbr_uri`, `eid`, `provision_num`, `provision_type`, `schedule_num` (the schedule's printed number, e.g. `3A`, on schedule chunks), `heading`, `text` (full chunk text), `refs` (cross-references).

On-disk size ~6.4 GB, dominated by `legislation_section` (vectors + full section text payload).

## Known limits

- Dense embeddings see only a chunk's first 512 tokens, so very long chunks (about 1,200 exceed 2,000 words) rank poorly even for verbatim queries.
- Dictionary Schedules (e.g. Criminal Code Schedule 1) are cut into 400-word windows, not one chunk per definition, so a single definition can rank low and cannot be cited on its own.
- Schedule clauses that repeat the same criterion text across visa subclasses (Migration Regulations Schedule 2) are hard to tell apart unless the query names the subclass.
- Qdrant's local/embedded mode keeps the full vector set resident in RAM and has no ANN index, so `QdrantClient(path=...)` construction itself is slow at this scale (multiple minutes to open, 550K+ points) - expected, not a bug. Fine for local CLI/MCP use; not suited to serving concurrent requests.
- No hosted search endpoint exists for this index - it's a downloadable artifact, not a live service.

## Licence

CC BY 4.0. Source legislation is Crown copyright - Commonwealth of Australia; reproduction permitted for non-commercial and research purposes under the [PSI Framework](https://www.legislation.gov.au/Help/Copyright). The `legislation_section` payload carries substantial verbatim section text from that corpus, so this dataset inherits the same licence as [`cchew/lex-au`](https://huggingface.co/datasets/cchew/lex-au) rather than a separate one for "just the vectors."

## What's built on this

- [ClauseKit](https://github.com/cchew/clause-kit) - LLM extraction of evaluatable rules (JSON Logic) from Acts, run claims against them grounded back to source clauses
- [Act Alike](https://github.com/cchew/act-alike) - compare how terms are defined across Acts

## Related

- [github.com/cchew/lex-au-search](https://github.com/cchew/lex-au-search) - source code, CLI, MCP server
- [cchew/lex-au](https://huggingface.co/datasets/cchew/lex-au) - the underlying AKN 3.0 XML corpus this index was built from
- [cchew/lex-au-search-embed-cache](https://huggingface.co/datasets/cchew/lex-au-search-embed-cache) - the ingest pipeline's own resumable embed cache (different artifact, different audience)
- [lex-au-graph](https://github.com/cchew/lex-au-graph) - cross-reference graph and definition resolution across Acts
