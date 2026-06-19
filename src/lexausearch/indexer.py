from __future__ import annotations

from qdrant_client import QdrantClient

from lexausearch.models import Chunk

DENSE_MODEL = "nomic-ai/nomic-embed-text-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = "legislation"


def configure_client(client: QdrantClient) -> QdrantClient:
    client.set_model(DENSE_MODEL)
    client.set_sparse_model(SPARSE_MODEL)
    return client


class Indexer:
    def __init__(self, client: QdrantClient) -> None:
        self._client = configure_client(client)

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._client.add(
            collection_name=COLLECTION,
            documents=["search_document: " + c.text for c in chunks],
            metadata=[
                {
                    "act_name": c.act_name,
                    "frbr_uri": c.frbr_uri,
                    "eid": c.eid,
                    "section_num": c.section_num,
                    "heading": c.heading,
                    "text": c.text,
                }
                for c in chunks
            ],
        )
