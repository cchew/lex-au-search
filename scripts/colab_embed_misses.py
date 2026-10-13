#!/usr/bin/env python3
"""One-shot catch-up: embed every cache-miss chunk across the current full
corpus and write the results into embed_cache.db. Deliberately does NOT
build the Qdrant search index -- local-mode Qdrant materialises the whole
collection into RAM regardless of on_disk config (see FUTURE.md and
README's Known limits), and Colab's free-tier RAM (~12.7GB) is less than
the ~16GB+ this project has already measured the full 534K-point
collection needing. Run this on a GPU Colab runtime to warm the cache,
push the updated embed_cache.db to HF, then run `lex-au-search ingest`
back on a machine with enough RAM (confirmed: the MBP) to build the
actual index against the now-near-fully-cached text.

Usage (after scripts/colab_ingest.sh's GPU setup steps, see README
"GPU-only cache warm-up (no index build)"):
    python scripts/colab_embed_misses.py --corpus-dir ./corpus --cache-path ./embed_cache.db
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

# print() output can be fully buffered (not line-buffered) when stdout isn't
# a real TTY -- the common case for a notebook's `!python script.py` cell.
# Without this, the progress lines below (added after a real run went silent
# for long enough that something interrupted it) could sit in the buffer and
# never actually reach the Colab output pane until the buffer filled or the
# process exited, defeating the point of adding them.
print = functools.partial(print, flush=True)

from qdrant_client import QdrantClient

from lexausearch.cache import EmbedCache
from lexausearch.chunker import chunk_corpus
from lexausearch.indexer import DENSE_MODEL, configure_client

_BATCH_SIZE = 32
_OOM_SIGNATURE = "Failed to allocate memory"


def embed_batch_with_oom_backoff(client, cache: EmbedCache, batch: list[str]) -> None:
    """Embed one batch and cache it; on an onnxruntime OOM, halve the batch
    and retry each half instead of aborting the whole run.

    Real failure 2026-10-13: a run already 78,752/94,289 through (83%)
    died on onnxruntime's BFCArena failing to allocate ~445MB for an Add
    node. unique_misses_length_bucketed sorts misses ascending by length,
    so later batches skew toward the corpus's longest texts -- the fixed
    --batch-size that comfortably fit the first 83% can still blow the T4's
    VRAM budget near the tail, and a straight resume hits the same wall
    again almost immediately since the remaining misses are exactly the
    longest ones left. Halving on OOM (down to a floor of 1 text, at which
    point a failure is genuine and left to propagate) finishes the run
    instead of losing it to one bad batch of long documents. Matched on
    the error message rather than a specific onnxruntime exception class --
    any other failure (bad input, corrupt cache, etc.) re-raises unchanged
    rather than being silently retried as if it were memory pressure."""
    if len(batch) > 1:
        try:
            _embed_and_cache(client, cache, batch)
            return
        except Exception as e:
            if _OOM_SIGNATURE not in str(e):
                raise
            mid = len(batch) // 2
            print(f"    OOM on batch of {len(batch)} texts -- splitting into "
                  f"{mid} + {len(batch) - mid} and retrying")
            embed_batch_with_oom_backoff(client, cache, batch[:mid])
            embed_batch_with_oom_backoff(client, cache, batch[mid:])
            return
    _embed_and_cache(client, cache, batch)  # floor of 1 -- let a real failure surface


def _embed_and_cache(client, cache: EmbedCache, batch: list[str]) -> None:
    fresh = dict(
        client._embed_documents(
            batch, embedding_model_name=client.embedding_model_name, embed_type="passage"
        )
    )
    cache.put_batch(fresh)


def unique_misses_length_bucketed(all_texts: list[str], hits: set[str]) -> list[str]:
    """Every distinct cache-miss text, sorted by length. Dedup matters:
    the same chunk text can repeat across chunks/Acts (shared boilerplate),
    and the cache is content-addressed, so embedding it twice is wasted
    GPU time. Length-sorting (FUTURE.md item 7, "length-bucketed embed
    batching") keeps each fixed-size batch close to homogeneous length
    instead of mixing a 20-token heading with a 4,000-token section and
    padding the whole batch to the longest member -- 1.5-3x throughput on
    this length distribution. Safe regardless of batch size or call order:
    the embed step returns a text->vector dict, order-independent and
    cache-keyed by text, so this reordering changes nothing downstream."""
    return sorted({t for t in all_texts if t not in hits}, key=len)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    args = parser.parse_args(argv)

    print(f"Chunking corpus at {args.corpus_dir} ...")
    chunks = chunk_corpus(args.corpus_dir)
    distinct_texts = sorted({c.text for c in chunks})
    print(f"{len(chunks)} total chunk instances, {len(distinct_texts)} distinct texts.")

    cache = EmbedCache(args.cache_path, model_name=DENSE_MODEL)
    print("Checking cache ...")
    # Chunked with progress output, not one cache.contains_batch(distinct_texts)
    # call -- a real run (402K distinct texts) went silent here for long
    # enough that something interrupted it; printed progress says exactly
    # how far it got if that happens again.
    #
    # contains_batch(), not get_batch(): real OOM 2026-10-13, reproduced
    # identically across 4 independent attempts, consistently dying right
    # around 80% through this exact loop. get_batch() pulls a full
    # 1024-float vector per hit; accumulating ~320K of those in `hits` is
    # hundreds of thousands of boxed-float Python lists (~24 bytes/float)
    # for data this script never uses -- it only ever checks set
    # membership to build the miss list below. contains_batch() never
    # reads the vector column at all.
    check_batch_size = 20000
    hits: set[str] = set()
    for i in range(0, len(distinct_texts), check_batch_size):
        sub = distinct_texts[i : i + check_batch_size]
        hits.update(cache.contains_batch(sub))
        done = min(i + check_batch_size, len(distinct_texts))
        print(f"  checked {done}/{len(distinct_texts)} distinct texts ({len(hits)} hits so far)")
    miss_texts = unique_misses_length_bucketed(distinct_texts, hits)
    print(f"{len(hits)} cache hits, {len(miss_texts)} distinct misses to embed.")

    if not miss_texts:
        print("Nothing to embed -- cache is already fully warm.")
        return 0

    # :memory: -- this client only exists to get a GPU-configured fastembed
    # text-embedding model loaded; it never stores a single point.
    client = configure_client(QdrantClient(":memory:"))

    total = len(miss_texts)
    for i in range(0, total, args.batch_size):
        batch = miss_texts[i : i + args.batch_size]
        embed_batch_with_oom_backoff(client, cache, batch)
        done = min(i + args.batch_size, total)
        print(f"  [{done}/{total}] embedded + cached")

    print(f"Done. {total} chunk(s) embedded and written to {args.cache_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
