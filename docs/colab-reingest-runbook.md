# Re-ingest the search index with a Colab GPU

The index is `cchew/lex-au-search-index`; the embedding cache is `cchew/lex-au-search-embed-cache`. Two routes, depending on the Colab runtime you have.

## Route A (use this): warm the cache on Colab, build the index on the laptop

A standard free runtime (about 12 GB RAM, T4 GPU) is enough, because Colab never holds the index. It only embeds texts that are missing from the cache and uploads the cache. The laptop (64 GB RAM) then builds the index with every embedding already cached.

Needed for this run: about **24,600 texts** to embed (6,129 from the corpus change plus 18,452 tail windows for long provisions). On a T4 that should take roughly 10 to 30 minutes.

### What you need

- A Colab notebook with a **GPU** runtime (Runtime > Change runtime type > T4 GPU). High-RAM is not needed.
- A Colab secret named `HF_TOKEN` (key icon in the left bar, notebook access on), whose value is `HF_CACHE_WRITE_TOKEN` from `projects/lex-au-search/repo/.env`. Only the last step uses it. Do not paste the token into a cell.
- `CORPUS_REV` = `bda7e6632ab9f9becf800e883210ba7ffc08df5c` (cchew/lex-au, corpus v0.16.1; the XML is identical to `e5469362`, the later commit only removed unused files).

### Cells, one per block

**1. Clone and set up the GPU (about 5 minutes)**

```python
!git clone https://github.com/cchew/lex-au-search.git
%cd lex-au-search
import os
os.environ["CORPUS_REV"] = "bda7e6632ab9f9becf800e883210ba7ffc08df5c"
!bash scripts/colab_reingest.sh setup-gpu
```

Expect a CUDA GPU check to pass. If it fails, stop and change the runtime; do not continue on CPU.

**2. Fetch the corpus and the cache (about 10 minutes)**

```python
!bash scripts/colab_reingest.sh fetch
```

Expect `corpus: 3087 Acts in index, 3087 xml files, structure_version 4` and `embed_cache.db` of about 2.9 GB.

**3. Embed the missing texts, in the background**

```python
!nohup bash scripts/colab_reingest.sh warm > warm.log 2>&1 &
```

```python
!tail -n 6 warm.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

Re-run the second cell to watch. First it prints `NNN cache hits, ~24600 distinct misses to embed`, then `[done/total] embedded + cached` lines. It ends with `Done. N chunk(s) embedded and written to ./embed_cache.db.` and `warm end <time>`. If the hit/miss line shows far more than about 25,000 misses, stop and tell me: the corpus or chunker is not what I measured.

If the session drops, start again from step 1. Already-embedded texts are only kept if the run finished, so a restart repeats the work, but it is short.

**4. Upload the warmed cache**

```python
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!bash scripts/colab_reingest.sh push-cache
```

Expect `embed cache uploaded`. Then tell me. The laptop does the rest: download the cache, build the index (all cache hits), run the verify gate, and publish after your go-ahead.

## Route B: whole ingest on one High-RAM Colab runtime

Needs Colab Pro (24 GB or more RAM), because the whole collection is held in memory during ingest. Steps: `setup` (checks RAM), `fetch`, `ingest` (in the background), `verify`, `publish`, using the same script and `CORPUS_REV`. The `setup` step refuses to run below 24 GB. Use Route A unless you have a High-RAM runtime and a reason.

## Rollback

Everything is versioned on HF. To go back to the generation 4 index: `cchew/lex-au-search-index` at `2c58405a073f0c8fe9618e723fe0519845de6422` (data) and `dd949158cadb2fe56ca289bd35b6b1329b348759` (catalogue). The embedding cache at `8f68daf1926faf50a4298107a014e19228dbb9bb` is the pre-run version.
