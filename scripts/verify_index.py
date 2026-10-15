"""Post-rebuild checks for a merged qdrant_storage/, run before push-index.

1. Count guard: chunk count may not fall more than 5% below the previous
   published generation unless --allow-drop is given, and every corpus Act
   must be present as an Act record.
2. Golden queries (tests/golden_queries.json): each must return the expected
   eId prefix in its top-k hits. These cover the shapes lex-au v0.10.0 broke
   for the chunker (nested clause, quoted insertion, clause-free schedule).

Exit 1 on any failure. See projects/lex-au/FUTURE.md "Downstream contract
regression suites" (component 5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_CHUNK_DROP = 0.05


def check_counts(
    act_count: int,
    chunk_count: int,
    *,
    previous_chunks: int | None,
    corpus_acts: int | None,
    allow_drop: bool = False,
) -> list[str]:
    failures: list[str] = []
    if previous_chunks and not allow_drop:
        floor = previous_chunks * (1 - MAX_CHUNK_DROP)
        if chunk_count < floor:
            failures.append(
                f"chunk_count {chunk_count} is more than {MAX_CHUNK_DROP:.0%} below the "
                f"previous generation ({previous_chunks}); pass --allow-drop if intended"
            )
    if corpus_acts is not None and act_count < corpus_acts:
        failures.append(f"act_count {act_count} < corpus Acts {corpus_acts}")
    return failures


def check_golden(searcher, queries: list[dict], top_k: int = 5) -> list[str]:
    failures: list[str] = []
    for q in queries:
        results = searcher.search(q["query"], limit=top_k, act=q.get("act"))
        hit = any(
            r.chunk.eid.startswith(q["expect_eid_prefix"])
            and q["must_contain"].lower() in r.chunk.text.lower()
            for r in results
        )
        if not hit:
            got = [r.chunk.eid for r in results]
            failures.append(f"golden query failed: {q['name']} (top-{top_k}: {got})")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--golden", default=str(Path(__file__).resolve().parent.parent / "tests" / "golden_queries.json"))
    parser.add_argument("--previous-chunks", type=int, default=None)
    parser.add_argument("--corpus-index", default=None, help="corpus index.json, to compare Act counts")
    parser.add_argument("--allow-drop", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    from lexausearch.indexer import COLLECTION_ACTS, COLLECTION_SECTIONS
    from lexausearch.searcher import Searcher

    client = QdrantClient(path=args.storage_dir)
    try:
        act_count = client.count(collection_name=COLLECTION_ACTS).count
        chunk_count = client.count(collection_name=COLLECTION_SECTIONS).count
        corpus_acts = (
            len(json.loads(Path(args.corpus_index).read_text())["acts"])
            if args.corpus_index else None
        )
        failures = check_counts(
            act_count, chunk_count, previous_chunks=args.previous_chunks,
            corpus_acts=corpus_acts, allow_drop=args.allow_drop,
        )
        failures += check_golden(
            Searcher(client), json.loads(Path(args.golden).read_text()), args.top_k
        )
    finally:
        client.close()

    print(f"act_count={act_count} chunk_count={chunk_count}")
    for f in failures:
        print(f"FAIL: {f}")
    print("OK" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
