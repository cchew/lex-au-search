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


def test_fetch_all_act_hashes_empty_collection_returns_empty_dict():
    from lexausearch.indexer import fetch_all_act_hashes
    client = QdrantClient(":memory:")
    assert fetch_all_act_hashes(client) == {}


def test_fetch_all_act_hashes_returns_stored_hash():
    from lexausearch.indexer import fetch_all_act_hashes
    client = QdrantClient(":memory:")
    idx = Indexer(client)
    record = _act_record()
    record.content_hash = "abc123"
    idx.upsert_acts([record])

    hashes = fetch_all_act_hashes(client)
    assert hashes == {"Privacy Act 1988": "abc123"}


def test_fetch_all_act_hashes_defaults_empty_when_not_set():
    from lexausearch.indexer import fetch_all_act_hashes
    client = QdrantClient(":memory:")
    idx = Indexer(client)
    idx.upsert_acts([_act_record()])  # content_hash defaults to ""

    hashes = fetch_all_act_hashes(client)
    assert hashes == {"Privacy Act 1988": ""}


def test_delete_act_removes_only_target_act_points():
    from lexausearch.indexer import delete_act
    client = QdrantClient(":memory:")
    idx = Indexer(client)
    idx.upsert_chunks([_chunk(act_name="Privacy Act 1988")])
    idx.upsert_chunks([_chunk(act_name="Crimes Act 1914", eid="sec-1")])
    idx.upsert_acts([_act_record()])
    other_record = _act_record()
    other_record.act_name = "Crimes Act 1914"
    idx.upsert_acts([other_record])

    delete_act(client, "Privacy Act 1988")

    remaining_sections = client.scroll(
        collection_name=COLLECTION_SECTIONS, limit=10, with_payload=True
    )[0]
    remaining_acts = client.scroll(
        collection_name=COLLECTION_ACTS, limit=10, with_payload=True
    )[0]
    assert {p.payload["act_name"] for p in remaining_sections} == {"Crimes Act 1914"}
    assert {p.payload["act_name"] for p in remaining_acts} == {"Crimes Act 1914"}


def test_delete_act_noop_when_act_never_indexed():
    from lexausearch.indexer import delete_act
    client = QdrantClient(":memory:")
    idx = Indexer(client)
    idx.upsert_chunks([_chunk()])
    idx.upsert_acts([_act_record()])

    delete_act(client, "Some Other Act 1999")  # must not raise

    remaining = client.scroll(collection_name=COLLECTION_SECTIONS, limit=10, with_payload=True)[0]
    assert len(remaining) == 1


def test_delete_act_noop_when_collection_does_not_exist():
    from lexausearch.indexer import delete_act
    client = QdrantClient(":memory:")
    delete_act(client, "Privacy Act 1988")  # must not raise, no collections exist at all


def test_configure_client_idempotent():
    client = QdrantClient(":memory:")
    configure_client(client)
    configure_client(client)  # must not raise


def test_module_import_triggers_onnxruntime_preload(monkeypatch):
    # Without an explicit preload, onnxruntime-gpu never loads the
    # nvidia-cudnn-cu12 wheel and CUDAExecutionProvider silently downgrades to
    # CPU on the RunPod stock image.
    import importlib
    from lexausearch import indexer

    calls = []
    monkeypatch.setattr(
        indexer.onnxruntime, "preload_dlls",
        lambda *a, **k: calls.append((a, k)), raising=False,
    )
    importlib.reload(indexer)
    assert len(calls) == 1


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


def test_merge_shard_clients_combines_two_shards():
    from lexausearch.indexer import merge_shard_clients

    shard_a = QdrantClient(":memory:")
    idx_a = Indexer(shard_a)
    idx_a.upsert_chunks([_chunk(eid="sec-6", act_name="Privacy Act 1988")])
    idx_a.upsert_acts([_act_record()])

    shard_b = QdrantClient(":memory:")
    idx_b = Indexer(shard_b)
    idx_b.upsert_chunks([_chunk(eid="sec-3", act_name="Crimes Act 1914",
                                 frbr_uri="/akn/au/act/1914/6/eng@2026-01-01")])
    idx_b.upsert_acts([ActRecord(
        act_name="Crimes Act 1914", frbr_uri="/akn/au/act/1914/6/eng@2026-01-01",
        year=1914, as_at_date="2026-01-01", section_count=1, schedule_clause_count=0,
    )])

    output = QdrantClient(":memory:")
    totals = merge_shard_clients([shard_a, shard_b], output)

    assert totals == {"sections": 2, "acts": 2}
    section_points = output.scroll(collection_name=COLLECTION_SECTIONS, limit=10, with_payload=True)[0]
    act_names = {p.payload["act_name"] for p in section_points}
    assert act_names == {"Privacy Act 1988", "Crimes Act 1914"}


def test_merge_shard_clients_preserves_vectors():
    """A merged point's dense vector must round-trip, not just its payload -
    the whole point of merging is not needing to re-embed."""
    from lexausearch.indexer import merge_shard_clients

    shard_a = QdrantClient(":memory:")
    Indexer(shard_a).upsert_chunks([_chunk()])

    output = QdrantClient(":memory:")
    merge_shard_clients([shard_a], output)

    source_points = shard_a.scroll(collection_name=COLLECTION_SECTIONS, limit=10, with_vectors=True)[0]
    merged_points = output.scroll(collection_name=COLLECTION_SECTIONS, limit=10, with_vectors=True)[0]
    assert len(merged_points) == 1
    dense_field = configure_client(shard_a).get_vector_field_name()
    source_vec = source_points[0].vector[dense_field]
    merged_vec = merged_points[0].vector[dense_field]
    assert source_vec == merged_vec


def test_merge_shard_clients_batches_across_page_boundary():
    """With batch_size=1 and 3 chunks, scroll pagination must not drop or
    duplicate any point - the real risk with an off-by-one in offset handling."""
    from lexausearch.indexer import merge_shard_clients

    shard = QdrantClient(":memory:")
    idx = Indexer(shard)
    idx.upsert_chunks([
        _chunk(eid="sec-1", text="first provision text here"),
        _chunk(eid="sec-2", text="second provision text here"),
        _chunk(eid="sec-3", text="third provision text here"),
    ])

    output = QdrantClient(":memory:")
    totals = merge_shard_clients([shard], output, batch_size=1)

    assert totals["sections"] == 3
    merged = output.scroll(collection_name=COLLECTION_SECTIONS, limit=10)[0]
    assert len(merged) == 3


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


def test_merge_cache_files_combines_two_shard_caches(tmp_path):
    from lexausearch.cache import merge_cache_files

    cache_a_path = tmp_path / "shard0_cache.db"
    EmbedCache(cache_a_path).put("first text", [0.1] * 1024)

    cache_b_path = tmp_path / "shard1_cache.db"
    EmbedCache(cache_b_path).put("second text", [0.2] * 1024)

    output_path = tmp_path / "merged_cache.db"
    total = merge_cache_files([cache_a_path, cache_b_path], output_path)

    assert total == 2
    merged = EmbedCache(output_path)
    assert merged.get("first text") is not None
    assert merged.get("second text") is not None


def test_merge_cache_files_same_text_across_shards_dedupes(tmp_path):
    """UUID5 keys are content-addressed - the same text embedded in two
    shards must merge to one row, not two, since it's the same key."""
    from lexausearch.cache import merge_cache_files

    cache_a_path = tmp_path / "shard0_cache.db"
    EmbedCache(cache_a_path).put("shared text", [0.1] * 1024)

    cache_b_path = tmp_path / "shard1_cache.db"
    EmbedCache(cache_b_path).put("shared text", [0.1] * 1024)

    output_path = tmp_path / "merged_cache.db"
    merge_cache_files([cache_a_path, cache_b_path], output_path)

    import sqlite3
    conn = sqlite3.connect(str(output_path))
    row_count = conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0]
    assert row_count == 1


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


def test_indexer_cache_embeds_misses_in_bounded_batches(monkeypatch):
    """A large Act's full miss-list must not be embedded in one uncapped
    _embed_documents() call - a real Colab run hit a CUDA OOM doing exactly
    that for a large Act's chunks (2026-08-21: onnxruntime BFCArena failed
    to allocate a 428MB buffer mid-attention-computation). The non-cache
    path already batches at 32 (see upsert_chunks); the cache-aware path
    must too."""
    import lexausearch.indexer as indexer_mod
    monkeypatch.setattr(indexer_mod, "_EMBED_BATCH_SIZE", 3)

    client = QdrantClient(":memory:")
    idx = Indexer(client, cache=EmbedCache(":memory:"))
    chunks = [
        _chunk(eid=f"sec-{i}", text=f"unique text number {i} for the batching test")
        for i in range(7)
    ]

    call_sizes = []
    original = client._embed_documents

    def spy(texts, **kwargs):
        call_sizes.append(len(texts))
        return original(texts, **kwargs)

    monkeypatch.setattr(client, "_embed_documents", spy)
    idx.upsert_chunks(chunks)

    assert len(call_sizes) > 1  # split into more than one call
    assert all(size <= 3 for size in call_sizes)
    assert sum(call_sizes) == 7
    assert idx.cache_misses == 7


def test_dense_model_is_snowflake_arctic_embed_l():
    from lexausearch import indexer
    assert indexer.DENSE_MODEL == "snowflake/snowflake-arctic-embed-l"


def test_dense_vector_size_matches_arctic_embed_l():
    from lexausearch import cache
    assert cache.DENSE_VECTOR_SIZE == 1024
