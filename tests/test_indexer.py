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


def test_configure_client_enables_cuda_when_available(monkeypatch):
    from unittest.mock import MagicMock
    from lexausearch import indexer

    monkeypatch.setattr(
        indexer.onnxruntime, "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    client = QdrantClient(":memory:")
    client.set_model = MagicMock()
    configure_client(client)
    client.set_model.assert_called_once_with(indexer.DENSE_MODEL, cuda=True)


def test_configure_client_disables_cuda_when_unavailable(monkeypatch):
    from unittest.mock import MagicMock
    from lexausearch import indexer

    monkeypatch.setattr(
        indexer.onnxruntime, "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    client = QdrantClient(":memory:")
    client.set_model = MagicMock()
    configure_client(client)
    client.set_model.assert_called_once_with(indexer.DENSE_MODEL, cuda=False)


def test_configure_client_prints_which_backend_is_used(monkeypatch, capsys):
    from unittest.mock import MagicMock
    from lexausearch import indexer

    monkeypatch.setattr(
        indexer.onnxruntime, "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    client = QdrantClient(":memory:")
    client.set_model = MagicMock()
    configure_client(client)
    assert "CUDA GPU" in capsys.readouterr().out

    monkeypatch.setattr(
        indexer.onnxruntime, "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    client2 = QdrantClient(":memory:")
    client2.set_model = MagicMock()
    configure_client(client2)
    assert "CPU" in capsys.readouterr().out


# --- EmbedCache tests ---
#
# Backed by SQLite (":memory:" for tests), not Qdrant - see cache.py's
# docstring. Cache storage is independent of the Indexer's Qdrant client.

from lexausearch.cache import EmbedCache


def test_embed_cache_cold_miss():
    cache = EmbedCache(":memory:")
    assert cache.get("some text") is None


def test_embed_cache_put_get_roundtrip():
    cache = EmbedCache(":memory:")
    vector = [0.1] * 1024
    cache.put("hello world", vector)
    result = cache.get("hello world")
    assert result is not None
    assert len(result) == 1024
    for a, b in zip(vector, result):
        assert abs(a - b) < 1e-6


def test_embed_cache_get_batch():
    cache = EmbedCache(":memory:")
    cache.put("text one", [0.1] * 1024)
    cache.put("text two", [0.2] * 1024)
    hits = cache.get_batch(["text one", "text two", "text three"])
    assert "text one" in hits
    assert "text two" in hits
    assert "text three" not in hits


def test_embed_cache_put_batch():
    cache = EmbedCache(":memory:")
    cache.put_batch({"text one": [0.1] * 1024, "text two": [0.2] * 1024})
    hits = cache.get_batch(["text one", "text two"])
    assert len(hits) == 2


def test_embed_cache_uuid5_deterministic():
    cache = EmbedCache(":memory:")
    # Same text → same UUID
    assert cache._cache_id("hello") == cache._cache_id("hello")
    # Different text → different UUID
    assert cache._cache_id("hello") != cache._cache_id("world")


def test_embed_cache_persists_to_file(tmp_path):
    """Unlike Qdrant local mode, a real file path must survive a fresh connection."""
    db_path = tmp_path / "embed_cache.db"
    EmbedCache(db_path).put("hello", [0.5] * 1024)
    reopened = EmbedCache(db_path)
    assert reopened.get("hello") is not None


def test_embed_cache_records_model_name_on_first_use():
    cache = EmbedCache(":memory:", model_name="snowflake/snowflake-arctic-embed-l")
    row = cache._conn.execute(
        "SELECT value FROM cache_meta WHERE key = 'model_name'"
    ).fetchone()
    assert row[0] == "snowflake/snowflake-arctic-embed-l"


def test_embed_cache_allows_reopening_with_same_model_name(tmp_path):
    db_path = tmp_path / "embed_cache.db"
    EmbedCache(db_path, model_name="snowflake/snowflake-arctic-embed-l").put("hello", [0.1] * 1024)
    reopened = EmbedCache(db_path, model_name="snowflake/snowflake-arctic-embed-l")
    assert reopened.get("hello") is not None


def test_embed_cache_rejects_model_name_mismatch(tmp_path):
    db_path = tmp_path / "embed_cache.db"
    EmbedCache(db_path, model_name="BAAI/bge-base-en-v1.5")
    with pytest.raises(ValueError, match="not 'snowflake"):
        EmbedCache(db_path, model_name="snowflake/snowflake-arctic-embed-l")


def test_embed_cache_no_model_name_skips_check(tmp_path):
    """Backward-compatible: omitting model_name never raises, even against a
    cache file that already has a recorded model name."""
    db_path = tmp_path / "embed_cache.db"
    EmbedCache(db_path, model_name="BAAI/bge-base-en-v1.5")
    EmbedCache(db_path)  # must not raise


def test_indexer_with_cache_smoke(privacy_chunks):
    """Cache-enabled upsert_chunks runs without error and results are searchable."""
    client = QdrantClient(":memory:")
    cache = EmbedCache(":memory:")
    idx = Indexer(client, cache=cache)
    idx.upsert_chunks(privacy_chunks)
    results = client.scroll(
        collection_name=COLLECTION_SECTIONS, limit=10, with_payload=True
    )
    assert len(results[0]) == len(privacy_chunks)


def test_indexer_cache_second_ingest_uses_cache(privacy_chunks):
    """Second upsert_chunks with same texts skips embedding (cache hits)."""
    client = QdrantClient(":memory:")
    cache = EmbedCache(":memory:")
    idx = Indexer(client, cache=cache)
    idx.upsert_chunks(privacy_chunks)
    # Populate cache from first run; second run should use only cached vectors
    cache_hits_before = cache.get_batch(
        [c.text for c in privacy_chunks]
    )
    assert len(cache_hits_before) == len(privacy_chunks)
    # Calling again must not raise
    idx.upsert_chunks(privacy_chunks)


def test_indexer_cache_counters_start_at_zero():
    client = QdrantClient(":memory:")
    idx = Indexer(client, cache=EmbedCache(":memory:"))
    assert idx.cache_hits == 0
    assert idx.cache_misses == 0


def test_indexer_cache_counters_track_misses_then_hits(privacy_chunks):
    """First ingest of new text is all misses; re-ingesting the same text is all hits."""
    client = QdrantClient(":memory:")
    idx = Indexer(client, cache=EmbedCache(":memory:"))
    idx.upsert_chunks(privacy_chunks)
    assert idx.cache_misses == len(privacy_chunks)
    assert idx.cache_hits == 0

    idx.upsert_chunks(privacy_chunks)
    assert idx.cache_misses == len(privacy_chunks)  # unchanged from first call
    assert idx.cache_hits == len(privacy_chunks)


def test_indexer_no_cache_counters_stay_zero(privacy_chunks):
    """Without a cache, hit/miss counters are meaningless and stay at zero."""
    client = QdrantClient(":memory:")
    idx = Indexer(client)
    idx.upsert_chunks(privacy_chunks)
    assert idx.cache_hits == 0
    assert idx.cache_misses == 0


def test_dense_model_is_snowflake_arctic_embed_l():
    from lexausearch import indexer
    assert indexer.DENSE_MODEL == "snowflake/snowflake-arctic-embed-l"


def test_dense_vector_size_matches_arctic_embed_l():
    from lexausearch import cache
    assert cache.DENSE_VECTOR_SIZE == 1024
