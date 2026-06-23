from __future__ import annotations

from qdrant_client import QdrantClient

from lexausearch.models import Chunk

DENSE_MODEL = "nomic-ai/nomic-embed-text-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = "legislation"


_configured_clients: set[int] = set()


def configure_client(client: QdrantClient) -> QdrantClient:
    cid = id(client)
    if cid not in _configured_clients:
        client.set_model(DENSE_MODEL)
        client.set_sparse_model(SPARSE_MODEL)
        _configured_clients.add(cid)
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
                    "provision_num": c.provision_num,
                    "provision_type": c.provision_type,
                    "heading": c.heading,
                    "text": c.text,
                    "refs": c.refs,
                }
                for c in chunks
            ],
            batch_size=32,
        )
