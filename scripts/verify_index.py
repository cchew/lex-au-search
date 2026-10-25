"""Post-rebuild checks for a merged qdrant_storage/, run before push-index.

1. Count guard: chunk count may not fall more than 5% below the previous
   published generation unless --allow-drop is given, and every corpus Act
   must be present as an Act record.
2. Golden queries (tests/golden_queries.json), the shapes lex-au v0.10.0 broke
   for the chunker (nested clause, quoted insertion, clause-free schedule):
   - presence: the expected chunk must exist in the index (hard failure);
   - retrieval: it should rank within the query's max_rank (default 5). A miss
     fails, unless the entry is marked "known_gap" (a very long chunk the
     embedder only sees the first 512 tokens of), which only warns.

3. Schedule payload: every schedule_text/schedule_clause chunk must carry a
   non-empty schedule_num, so an index built from a pre-a28cf4d checkout
   cannot pass the gate.

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


def check_schedule_nums(client, collection: str | None = None) -> list[str]:
    """Fail when any schedule chunk has an empty or missing schedule_num."""
    from qdrant_client import models

    if collection is None:
        from lexausearch.indexer import COLLECTION_SECTIONS as collection

    flt = models.Filter(must=[
        models.FieldCondition(
            key="provision_type",
            match=models.MatchAny(any=["schedule_text", "schedule_clause"]),
        )
    ])
    bad = 0
    samples: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection, scroll_filter=flt, limit=5000, offset=offset,
            with_payload=["eid", "act_name", "schedule_num"],
        )
        for p in points:
            if not (p.payload or {}).get("schedule_num"):
                bad += 1
                if len(samples) < 5:
                    samples.append(f"{p.payload.get('act_name')} {p.payload.get('eid')}")
        if offset is None:
            break
    if not bad:
        return []
    return [
        f"{bad} schedule chunk(s) have empty/missing schedule_num (index built from a "
        f"pre-a28cf4d checkout?); e.g. {'; '.join(samples)}"
    ]


def _present(client, act: str | None, prefix: str, phrase: str) -> bool:
    from qdrant_client import models

    from lexausearch.indexer import COLLECTION_SECTIONS

    flt = None
    if act:
        flt = models.Filter(must=[
            models.FieldCondition(key="act_name", match=models.MatchValue(value=act))
        ])
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION_SECTIONS, scroll_filter=flt, limit=2000, offset=offset,
            with_payload=["eid", "text"],
        )
        for p in points:
            if p.payload["eid"].startswith(prefix) and phrase.lower() in p.payload["text"].lower():
                return True
        if offset is None:
            return False


def check_golden(searcher, queries: list[dict], default_rank: int = 5) -> tuple[list[str], list[str]]:
    """(failures, warnings). Presence is a hard failure; rank beyond max_rank
    fails unless the entry is a known_gap."""
    failures: list[str] = []
    warnings: list[str] = []
    for q in queries:
        if not _present(searcher._client, q.get("act"), q["expect_eid_prefix"], q["must_contain"]):
            failures.append(f"golden chunk missing from index: {q['name']}")
            continue
        max_rank = q.get("max_rank", default_rank)
        results = searcher.search(q["query"], limit=max(max_rank, 50), act=q.get("act"))
        rank = next(
            (i + 1 for i, r in enumerate(results)
             if r.chunk.eid.startswith(q["expect_eid_prefix"])
             and q["must_contain"].lower() in r.chunk.text.lower()),
            None,
        )
        if rank is not None and rank <= max_rank:
            continue
        msg = f"golden query rank {rank} > {max_rank}: {q['name']}"
        (warnings if q.get("known_gap") else failures).append(msg)
    return failures, warnings


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

    warnings: list[str] = []
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
        golden_failures, warnings = check_golden(
            Searcher(client), json.loads(Path(args.golden).read_text()), args.top_k
        )
        failures += golden_failures
        failures += check_schedule_nums(client)
    finally:
        client.close()

    print(f"act_count={act_count} chunk_count={chunk_count}")
    for w in warnings:
        print(f"WARN: {w}")
    for f in failures:
        print(f"FAIL: {f}")
    print("OK" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
