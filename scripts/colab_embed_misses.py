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
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from lexausearch.cache import EmbedCache
from lexausearch.chunker import chunk_corpus
from lexausearch.indexer import DENSE_MODEL, configure_client

_BATCH_SIZE = 32


def unique_misses_length_bucketed(all_texts: list[str], hits: dict[str, list[float]]) -> list[str]:
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
    all_texts = [c.text for c in chunks]
    print(f"{len(chunks)} total chunk instances, {len(set(all_texts))} distinct texts.")

    cache = EmbedCache(args.cache_path, model_name=DENSE_MODEL)
    print("Checking cache ...")
    hits = cache.get_batch(all_texts)
    miss_texts = unique_misses_length_bucketed(all_texts, hits)
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
        fresh = dict(
            client._embed_documents(
                batch, embedding_model_name=client.embedding_model_name, embed_type="passage"
            )
        )
        cache.put_batch(fresh)
        done = min(i + args.batch_size, total)
        print(f"  [{done}/{total}] embedded + cached")

    print(f"Done. {total} chunk(s) embedded and written to {args.cache_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
