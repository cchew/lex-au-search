"""One-off verification: confirm a chunk containing a previously-truncated
definition now carries the complete list content after re-ingest.
"""
from pathlib import Path
from qdrant_client import QdrantClient
from lexausearch.searcher import Searcher

STORAGE_DIR = Path("./qdrant_storage")


def main() -> None:
    client = QdrantClient(path=str(STORAGE_DIR))
    searcher = Searcher(client)

    results = searcher.search(
        "related entity means",
        limit=5,
        act="Bankruptcy Act 1966",
    )
    print(f"Results for 'related entity means' in Bankruptcy Act 1966: {len(results)}")
    for r in results:
        text = r.chunk.text
        has_lead_in_only = text.strip().endswith("any of the following:")
        has_list_content = "a relative of the person" in text
        print(f"  score={r.score:.3f} eid={r.chunk.eid}")
        print(f"    truncated (bug still present): {has_lead_in_only}")
        print(f"    complete (fix present): {has_list_content}")


if __name__ == "__main__":
    main()
