from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import onnxruntime
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.models import (
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PayloadSchemaType,
)

from lexausearch.models import ActRecord, Chunk

if TYPE_CHECKING:
    from lexausearch.cache import EmbedCache

DENSE_MODEL = "snowflake/snowflake-arctic-embed-l"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_ACTS = "legislation"
COLLECTION_SECTIONS = "legislation_section"

_QUANT_CONFIG = ScalarQuantization(
    scalar=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=False,
    )
)


def _cuda_available() -> bool:
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def configure_client(client: QdrantClient) -> QdrantClient:
    cuda = _cuda_available()
    print(f"Dense embeddings: {'CUDA GPU' if cuda else 'CPU'} ({DENSE_MODEL})")
    client.set_model(DENSE_MODEL, cuda=cuda)
    client.set_sparse_model(SPARSE_MODEL)
    return client


def _ensure_collection(client: QdrantClient, name: str) -> None:
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=client.get_fastembed_vector_params(
                on_disk=True, hnsw_config=HnswConfigDiff(on_disk=True)
            ),
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(on_disk=True),
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
        self.cache_hits = 0
        self.cache_misses = 0

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        _ensure_collection(self._client, COLLECTION_SECTIONS)
        _create_payload_indexes(
            self._client, COLLECTION_SECTIONS,
            ["act_name", "frbr_uri", "provision_type"],
        )
        if self._cache is None:
            for i in range(0, len(chunks), 64):
                sub = chunks[i : i + 64]
                self._client.add(
                    collection_name=COLLECTION_SECTIONS,
                    documents=[c.text for c in sub],
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
                        for c in sub
                    ],
                    batch_size=32,
                )
        else:
            self._upsert_chunks_with_cache(chunks)

    def _upsert_chunks_with_cache(self, chunks: list[Chunk]) -> None:
        """Cache-aware upsert: reuse stored dense embeddings for unchanged chunk texts."""
        texts = [c.text for c in chunks]

        # Dense vectors: check cache, embed only misses
        cached = self._cache.get_batch(texts)
        miss_texts = [t for t in texts if t not in cached]
        self.cache_hits += len(texts) - len(miss_texts)
        self.cache_misses += len(miss_texts)
        if miss_texts:
            fresh = dict(
                self._client._embed_documents(
                    miss_texts,
                    embedding_model_name=self._client.embedding_model_name,
                    embed_type="passage",
                )
            )
            self._cache.put_batch(fresh)
            cached.update(fresh)

        # Sparse vectors: always compute (BM25 is fast)
        sparse_list = list(
            self._client._sparse_embed_documents(
                texts,
                embedding_model_name=self._client.sparse_embedding_model_name,
            )
        )

        dense_field = self._client.get_vector_field_name()
        sparse_field = self._client.get_sparse_vector_field_name()

        points = []
        for chunk, text, sparse_sv in zip(chunks, texts, sparse_list):
            dense_vec = cached[text]
            point_vector: dict = {dense_field: dense_vec}
            if sparse_field is not None:
                point_vector[sparse_field] = qmodels.SparseVector(
                    indices=sparse_sv.indices,
                    values=sparse_sv.values,
                )
            payload = {
                "document": text,
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
                    "content_hash": r.content_hash,
                }
                for r in act_records
            ],
            batch_size=32,
        )


def fetch_all_act_hashes(client: QdrantClient) -> dict[str, str]:
    """Full scroll of COLLECTION_ACTS, returning {act_name: content_hash}
    for every currently-indexed Act. Missing/legacy records (indexed before
    this field existed) contribute "" for that Act, not a KeyError."""
    if not client.collection_exists(COLLECTION_ACTS):
        return {}
    hashes: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_ACTS,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            act_name = p.payload.get("act_name")
            if act_name:
                hashes[act_name] = p.payload.get("content_hash", "")
        if offset is None:
            break
    return hashes


def delete_act(client: QdrantClient, act_name: str) -> None:
    """Delete every point belonging to act_name from both collections
    (legislation_section, legislation), by the existing act_name payload
    index (already created on both collections by _create_payload_indexes).
    Safe to call even if the Act was never indexed, or if a collection
    doesn't exist yet -- both are no-ops, not errors."""
    act_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="act_name", match=qmodels.MatchValue(value=act_name))]
    )
    for collection in (COLLECTION_SECTIONS, COLLECTION_ACTS):
        if client.collection_exists(collection):
            client.delete(collection_name=collection, points_selector=act_filter, wait=True)


def merge_shard_clients(
    shard_clients: list[QdrantClient], output_client: QdrantClient, batch_size: int = 500
) -> dict[str, int]:
    """Batch scroll+upsert every point from each shard client's collections
    into output_client's collections. Reuses _ensure_collection /
    _create_payload_indexes so quantization/HNSW/payload-index config
    matches a normal single-run ingest exactly. Bounded memory: never holds
    more than batch_size points across all shards at once."""
    configure_client(output_client)
    _ensure_collection(output_client, COLLECTION_SECTIONS)
    _create_payload_indexes(
        output_client, COLLECTION_SECTIONS, ["act_name", "frbr_uri", "provision_type"]
    )
    _ensure_collection(output_client, COLLECTION_ACTS)
    _create_payload_indexes(output_client, COLLECTION_ACTS, ["act_name", "frbr_uri", "year"])

    totals = {"sections": 0, "acts": 0}
    collection_keys = [(COLLECTION_SECTIONS, "sections"), (COLLECTION_ACTS, "acts")]
    for shard_client in shard_clients:
        for collection, key in collection_keys:
            if not shard_client.collection_exists(collection):
                continue  # shard never upserted to this collection, nothing to merge
            offset = None
            while True:
                points, offset = shard_client.scroll(
                    collection_name=collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                if not points:
                    break
                output_client.upsert(
                    collection_name=collection,
                    points=[
                        qmodels.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                        for p in points
                    ],
                    wait=True,
                )
                totals[key] += len(points)
                if offset is None:
                    break
    return totals
