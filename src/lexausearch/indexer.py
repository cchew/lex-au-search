from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.models import (
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PayloadSchemaType,
)

from lexausearch.models import ActRecord, Chunk

if TYPE_CHECKING:
    from lexausearch.cache import EmbedCache

DENSE_MODEL = "nomic-ai/nomic-embed-text-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_ACTS = "legislation"
COLLECTION_SECTIONS = "legislation_section"

_QUANT_CONFIG = ScalarQuantization(
    scalar=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=True,
    )
)


def configure_client(client: QdrantClient) -> QdrantClient:
    client.set_model(DENSE_MODEL)
    client.set_sparse_model(SPARSE_MODEL)
    return client


def _ensure_collection(client: QdrantClient, name: str) -> None:
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=client.get_fastembed_vector_params(on_disk=False),
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(on_disk=False),
            quantization_config=_QUANT_CONFIG,
        )
    except Exception:
        pass  # already exists


def _create_payload_indexes(
    client: QdrantClient, collection: str, fields: list[str]
) -> None:
    for field in fields:
        try:
            schema_type = (
                PayloadSchemaType.INTEGER if field == "year"
                else PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema_type,
            )
        except Exception:
            pass  # index already exists


class Indexer:
    def __init__(self, client: QdrantClient, cache: EmbedCache | None = None) -> None:
        self._client = configure_client(client)
        self._cache = cache

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        _ensure_collection(self._client, COLLECTION_SECTIONS)
        _create_payload_indexes(
            self._client, COLLECTION_SECTIONS,
            ["act_name", "frbr_uri", "provision_type"],
        )
        if self._cache is None:
            self._client.add(
                collection_name=COLLECTION_SECTIONS,
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
        else:
            self._upsert_chunks_with_cache(chunks)

    def _upsert_chunks_with_cache(self, chunks: list[Chunk]) -> None:
        """Cache-aware upsert: reuse stored dense embeddings for unchanged chunk texts."""
        prefixed = ["search_document: " + c.text for c in chunks]

        # Dense vectors: check cache, embed only misses
        cached = self._cache.get_batch(prefixed)
        miss_texts = [t for t in prefixed if t not in cached]
        if miss_texts:
            fresh = dict(
                self._client._embed_documents(
                    miss_texts,
                    embedding_model_name=self._client.embedding_model_name,
                    embed_type="passage",
                )
            )
            # Store misses in cache
            for text, vec in fresh.items():
                self._cache.put(text, vec)
            cached.update(fresh)

        # Sparse vectors: always compute (BM25 is fast)
        sparse_list = list(
            self._client._sparse_embed_documents(
                prefixed,
                embedding_model_name=self._client.sparse_embedding_model_name,
            )
        )

        dense_field = self._client.get_vector_field_name()
        sparse_field = self._client.get_sparse_vector_field_name()

        points = []
        for i, (chunk, prefixed_text, sparse_sv) in enumerate(
            zip(chunks, prefixed, sparse_list)
        ):
            dense_vec = cached[prefixed_text]
            point_vector: dict = {dense_field: dense_vec}
            if sparse_field is not None:
                point_vector[sparse_field] = qmodels.SparseVector(
                    indices=sparse_sv.indices,
                    values=sparse_sv.values,
                )
            payload = {
                "document": prefixed_text,
                "act_name": chunk.act_name,
                "frbr_uri": chunk.frbr_uri,
                "eid": chunk.eid,
                "provision_num": chunk.provision_num,
                "provision_type": chunk.provision_type,
                "heading": chunk.heading,
                "text": chunk.text,
                "refs": chunk.refs,
            }
            points.append(qmodels.PointStruct(
                id=uuid.uuid4().hex,
                vector=point_vector,
                payload=payload,
            ))

        self._client.upsert(
            collection_name=COLLECTION_SECTIONS,
            points=points,
            wait=True,
        )

    def upsert_acts(self, act_records: list[ActRecord]) -> None:
        if not act_records:
            return
        _ensure_collection(self._client, COLLECTION_ACTS)
        _create_payload_indexes(
            self._client, COLLECTION_ACTS,
            ["act_name", "frbr_uri", "year"],
        )
        self._client.add(
            collection_name=COLLECTION_ACTS,
            documents=[
                f"{r.act_name} — {r.year}"
                for r in act_records
            ],
            metadata=[
                {
                    "act_name": r.act_name,
                    "frbr_uri": r.frbr_uri,
                    "year": r.year,
                    "as_at_date": r.as_at_date,
                    "section_count": r.section_count,
                    "schedule_clause_count": r.schedule_clause_count,
                }
                for r in act_records
            ],
            batch_size=32,
        )
