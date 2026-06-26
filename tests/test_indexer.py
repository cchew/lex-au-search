import pytest
from qdrant_client import QdrantClient

from lexausearch.indexer import (
    Indexer, COLLECTION_ACTS, COLLECTION_SECTIONS, configure_client
)
from lexausearch.models import Chunk, ActRecord


def _chunk(**kwargs) -> Chunk:
    defaults = dict(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="sec-6", provision_num="6", provision_type="section",
        heading="Definitions", text="personal information means", refs=[],
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def _act_record() -> ActRecord:
    return ActRecord(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        year=1988, as_at_date="2026-01-01",
        section_count=2, schedule_clause_count=1,
    )


@pytest.fixture(scope="module")
def loaded_indexer():
    client = QdrantClient(":memory:")
    idx = Indexer(client)
    idx.upsert_chunks([_chunk()])
    idx.upsert_acts([_act_record()])
    return idx, client


def test_collection_sections_exists(loaded_indexer):
    _, client = loaded_indexer
    info = client.get_collection(COLLECTION_SECTIONS)
    assert info is not None


def test_collection_acts_exists(loaded_indexer):
    _, client = loaded_indexer
    info = client.get_collection(COLLECTION_ACTS)
    assert info is not None


def test_upsert_chunks_stores_provision_type(loaded_indexer):
    _, client = loaded_indexer
    results = client.scroll(
        collection_name=COLLECTION_SECTIONS, limit=10, with_payload=True
    )
    payloads = [p.payload for p in results[0]]
    assert any(p.get("provision_type") == "section" for p in payloads)


def test_upsert_acts_stores_year(loaded_indexer):
    _, client = loaded_indexer
    results = client.scroll(
        collection_name=COLLECTION_ACTS, limit=10, with_payload=True
    )
    payloads = [p.payload for p in results[0]]
    assert any(p.get("year") == 1988 for p in payloads)


def test_upsert_chunks_idempotent():
    client = QdrantClient(":memory:")
    idx = Indexer(client)
    chunk = _chunk()
    idx.upsert_chunks([chunk])
    idx.upsert_chunks([chunk])  # second call must not raise


def test_upsert_acts_idempotent():
    client = QdrantClient(":memory:")
    idx = Indexer(client)
    record = _act_record()
    idx.upsert_acts([record])
    idx.upsert_acts([record])  # second call must not raise


def test_configure_client_idempotent():
    client = QdrantClient(":memory:")
    configure_client(client)
    configure_client(client)  # must not raise


# --- EmbedCache tests ---

from lexausearch.cache import EmbedCache, EMBED_CACHE_COLLECTION


def test_embed_cache_creation():
    client = QdrantClient(":memory:")
    cache = EmbedCache(client)
    info = client.get_collection(EMBED_CACHE_COLLECTION)
    assert info is not None


def test_embed_cache_cold_miss():
    client = QdrantClient(":memory:")
    cache = EmbedCache(client)
    assert cache.get("some text") is None


def test_embed_cache_put_get_roundtrip():
    import math
    client = QdrantClient(":memory:")
    cache = EmbedCache(client)
    vector = [0.1] * 768
    cache.put("hello world", vector)
    result = cache.get("hello world")
    assert result is not None
    assert len(result) == 768
    # COSINE collections normalise vectors on store; check same direction via dot product ≈ 1
    norm_a = math.sqrt(sum(v ** 2 for v in vector))
    norm_b = math.sqrt(sum(v ** 2 for v in result))
    dot = sum(a * b for a, b in zip(vector, result))
    cosine_sim = dot / (norm_a * norm_b)
    assert abs(cosine_sim - 1.0) < 1e-4


def test_embed_cache_get_batch():
    client = QdrantClient(":memory:")
    cache = EmbedCache(client)
    cache.put("text one", [0.1] * 768)
    cache.put("text two", [0.2] * 768)
    hits = cache.get_batch(["text one", "text two", "text three"])
    assert "text one" in hits
    assert "text two" in hits
    assert "text three" not in hits


def test_embed_cache_uuid5_deterministic():
    client = QdrantClient(":memory:")
    cache = EmbedCache(client)
    # Same text → same UUID
    assert cache._cache_id("hello") == cache._cache_id("hello")
    # Different text → different UUID
    assert cache._cache_id("hello") != cache._cache_id("world")


def test_indexer_with_cache_smoke(privacy_chunks):
    """Cache-enabled upsert_chunks runs without error and results are searchable."""
    client = QdrantClient(":memory:")
    cache = EmbedCache(client)
    idx = Indexer(client, cache=cache)
    idx.upsert_chunks(privacy_chunks)
    results = client.scroll(
        collection_name=COLLECTION_SECTIONS, limit=10, with_payload=True
    )
    assert len(results[0]) == len(privacy_chunks)


def test_indexer_cache_second_ingest_uses_cache(privacy_chunks):
    """Second upsert_chunks with same texts skips embedding (cache hits)."""
    client = QdrantClient(":memory:")
    cache = EmbedCache(client)
    idx = Indexer(client, cache=cache)
    idx.upsert_chunks(privacy_chunks)
    # Populate cache from first run; second run should use only cached vectors
    cache_hits_before = cache.get_batch(
        [c.text for c in privacy_chunks]
    )
    assert len(cache_hits_before) == len(privacy_chunks)
    # Calling again must not raise
    idx.upsert_chunks(privacy_chunks)
