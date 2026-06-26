from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

EMBED_CACHE_COLLECTION = "embed_cache"
DENSE_VECTOR_SIZE = 768  # nomic-ai/nomic-embed-text-v1.5 output dimension
_UUID5_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace


class EmbedCache:
    """Persistent UUID5-keyed dense embedding cache backed by a Qdrant collection.

    Key: UUID5(URL_NS, text) — deterministic, content-addressed.
    Value: list[float] dense vector (768-dim for nomic-embed-text-v1.5).

    The cache is optional: Indexer falls back to client.add() when cache=None.
    """

    def __init__(self, client: QdrantClient, vector_size: int = DENSE_VECTOR_SIZE) -> None:
        self._client = client
        self._vector_size = vector_size
        self._ensure_collection()

    def _cache_id(self, text: str) -> str:
        return str(uuid.uuid5(_UUID5_NS, text))

    def _ensure_collection(self) -> None:
        try:
            self._client.create_collection(
                collection_name=EMBED_CACHE_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_size,
                    distance=qmodels.Distance.COSINE,
                    on_disk=True,
                ),
            )
        except Exception:
            pass  # already exists

    def get(self, text: str) -> list[float] | None:
        """Return cached dense vector for text, or None on miss."""
        point_id = self._cache_id(text)
        results = self._client.retrieve(
            collection_name=EMBED_CACHE_COLLECTION,
            ids=[point_id],
            with_vectors=True,
        )
        if results:
            vec = results[0].vector
            if isinstance(vec, list):
                return vec
        return None

    def put(self, text: str, vector: list[float]) -> None:
        """Store dense vector for text."""
        point_id = self._cache_id(text)
        self._client.upsert(
            collection_name=EMBED_CACHE_COLLECTION,
            points=[qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload={"text_preview": text[:100]},
            )],
        )

    def get_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """Return {text: vector} for all cache hits. Misses are absent."""
        ids = [self._cache_id(t) for t in texts]
        text_by_id = {self._cache_id(t): t for t in texts}
        results = self._client.retrieve(
            collection_name=EMBED_CACHE_COLLECTION,
            ids=ids,
            with_vectors=True,
        )
        return {
            text_by_id[str(r.id)]: r.vector
            for r in results
            if r.vector is not None and isinstance(r.vector, list)
        }
