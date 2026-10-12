"""Print a storage dir's Act and chunk counts as GITHUB_OUTPUT-style
`key=value` lines, so a workflow step can capture them for the subsequent
`hf_cache push-index` call (which wants act_count/chunk_count in its
catalogue sidecar, purely informational -- not load-bearing for the
generation guard).
"""
from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient

from lexausearch.indexer import COLLECTION_ACTS, COLLECTION_SECTIONS


def report(storage_dir: str) -> tuple[int, int]:
    client = QdrantClient(path=storage_dir)
    try:
        act_count = client.count(collection_name=COLLECTION_ACTS).count
        chunk_count = client.count(collection_name=COLLECTION_SECTIONS).count
    finally:
        client.close()
    return act_count, chunk_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", required=True)
    args = parser.parse_args()

    act_count, chunk_count = report(args.storage_dir)
    print(f"act_count={act_count}")
    print(f"chunk_count={chunk_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
