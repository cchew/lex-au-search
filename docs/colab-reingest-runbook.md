# Manual full re-ingest on Colab

Rebuilds the published index (`cchew/lex-au-search-index`) from the current lex-au corpus on one Colab GPU runtime, using the published embedding cache so almost nothing is re-embedded. You run each step and check the output before moving on. Nothing is published until the last step.

## Why a full re-ingest

lex-au v0.16.0 changed the eIds in most Acts (schedule nesting, list scoping, TOC removal), so nearly every Act's content hash differs from index generation 4. `ingest-delta` would re-index almost everything anyway, and it has to load the whole existing index into RAM first. A fresh `ingest` from the published embedding cache is simpler and leaves no stale points.

Chunk text is mostly unchanged, so the cache should hit for nearly every chunk. Expect the run to be dominated by chunking and Qdrant upserts, not embedding.

## What you need

- A Colab runtime with a GPU **and High-RAM** (24 GB or more). The full collection lives in memory during ingest. The script refuses to continue below 24 GB.
- A Hugging Face **write** token for the lex-au-search repos (`HF_CACHE_WRITE_TOKEN` in `projects/lex-au-search/repo/.env`), added as a Colab secret named `HF_TOKEN` (key icon in the left bar, notebook access on). Only the publish step uses it. Do not paste it into a cell.
- Pinned inputs:
  - `CORPUS_REV` = `e5469362dad81f7c4a8e72f4a3909be64d398818` (cchew/lex-au, lex-au v0.16.1, structure version 4)
  - Previous index: generation 4, 554,750 chunks, 3,087 Acts

## Steps

Each block is one Colab cell.

**1. Clone and set up (about 5 minutes)**

```python
!git clone https://github.com/cchew/lex-au-search.git
%cd lex-au-search
import os
os.environ["CORPUS_REV"] = "e5469362dad81f7c4a8e72f4a3909be64d398818"
!bash scripts/colab_reingest.sh setup
```

Expect `RAM: ... GB` (24 or more) and a GPU check that prints a CUDA provider. If the GPU check fails or RAM is low, stop and change the runtime; do not continue on CPU.

**2. Fetch the corpus and the embedding cache (about 10 minutes)**

```python
!bash scripts/colab_reingest.sh fetch
```

Expect: `corpus: 3087 Acts in index, 3087 xml files, structure_version 4` and `embed_cache.db` of about 2.9 GB. If it prints `NOTE: head has moved`, someone published to cchew/lex-au after this runbook was written. Check what changed before going on.

**3. Ingest in the background, then watch the log**

```python
!nohup bash scripts/colab_reingest.sh ingest > ingest.log 2>&1 &
```

```python
!tail -n 5 ingest.log; free -g | head -2; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

Re-run the second cell to check progress. Keep the browser tab open; an idle or closed tab can end the session. The log ends with `ingest end <time>`. If the session dies, start again from step 1: Qdrant storage is not resumable, but the cache (re-downloaded in step 2) makes the retry fast.

If RAM climbs toward the limit, stop and use the sharded path (`scripts/run_sharded_ingest.py`, which I can drive from the laptop).

**4. Verify (about 5 minutes)**

```python
!bash scripts/colab_reingest.sh verify
```

This prints `act_count` and `chunk_count`, then runs the golden-query gate. It must end `verify OK. Nothing has been published.`

Check by eye before going on:

- `act_count=3087`.
- `chunk_count` close to 554,750 (the gate fails below a 5% drop). A rise is plausible; a large rise is not.
- No `FAIL` lines.

**5. Publish (only if step 4 is clean)**

```python
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!bash scripts/colab_reingest.sh publish
```

This pushes the index as generation 5 (generation-guarded; a different embedding model would be refused) and uploads the updated `embed_cache.db`.

## Rollback

Everything is versioned on HF. To go back to generation 4:

- Index: `cchew/lex-au-search-index` at commit `2c58405a073f0c8fe9618e723fe0519845de6422` (data) and `dd949158cadb2fe56ca289bd35b6b1329b348759` (catalogue).
- Embedding cache: `cchew/lex-au-search-embed-cache` at `8f68daf1926faf50a4298107a014e19228dbb9bb`.

Restore by downloading those revisions and re-uploading them as the new head.

## After publishing

Back on the laptop, with the new index in place:

- `python -m lexausearch.hf_cache fetch-index --dest ./hf-index --expect-model snowflake/snowflake-arctic-embed-l`, then `python scripts/verify_index.py --storage-dir ./hf-index/qdrant_storage --corpus-index <corpus>/index.json` as an independent check.
- Update the README index-generation line (generation 5, chunk and Act counts, corpus v0.16.0).
