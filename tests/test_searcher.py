import pytest
from qdrant_client import QdrantClient

from lexausearch.indexer import Indexer
from lexausearch.searcher import Searcher
from lexausearch.models import SearchResult, Chunk


@pytest.fixture(scope="module")
def indexed_searcher(privacy_chunks):
    """Real in-memory Qdrant + FastEmbed. Downloads model on first run (~270 MB, cached)."""
    client = QdrantClient(":memory:")
    indexer = Indexer(client)
    indexer.upsert_chunks(privacy_chunks)
    return Searcher(client)


def test_search_returns_results(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_result_has_score(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    assert all(isinstance(r.score, float) for r in results)
    assert all(isinstance(r.score, float) and r.score >= 0.0 for r in results)


def test_search_result_chunk_fields(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    r = results[0]
    assert r.chunk.act_name == "Privacy Act 1988"
    assert r.chunk.frbr_uri == "/akn/au/act/1988/119/eng@2026-01-01"
    assert r.chunk.eid in ("sec-3", "sec-6")
    assert r.chunk.text != ""


def test_search_limit_respected(indexed_searcher):
    results = indexed_searcher.search("information", limit=1)
    assert len(results) <= 1


def test_configure_client_idempotent(privacy_chunks):
    """configure_client called twice on same client must not raise."""
    from lexausearch.indexer import configure_client
    client = QdrantClient(":memory:")
    configure_client(client)
    configure_client(client)  # should not raise


def test_search_act_filter_excludes_other_acts(privacy_chunks):
    extra_chunk = Chunk(
        act_name="Fair Work Act 2009",
        frbr_uri="/akn/au/act/2009/28/eng@2026-01-01",
        eid="sec-12",
        provision_num="12",
        provision_type="section",
        heading="Definitions",
        text="12 Definitions In this Act employee means a person employed.",
        refs=[],
    )
    client = QdrantClient(":memory:")
    indexer = Indexer(client)
    indexer.upsert_chunks(privacy_chunks + [extra_chunk])
    searcher = Searcher(client)
    results = searcher.search("information", limit=5, act="Fair Work Act 2009")
    assert len(results) > 0
    assert all(r.chunk.act_name == "Fair Work Act 2009" for r in results)


def test_search_returns_provision_num(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    assert all(isinstance(r.chunk.provision_num, str) for r in results)


def test_search_provision_type_filter(privacy_chunks):
    client = QdrantClient(":memory:")
    from lexausearch.indexer import Indexer
    from lexausearch.models import Chunk
    clause_chunk = Chunk(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        eid="schedule-1__clause-1", provision_num="1",
        provision_type="schedule_clause", heading="APP 1",
        text="APP 1 entities must manage personal information openly", refs=[],
    )
    indexer = Indexer(client)
    indexer.upsert_chunks(privacy_chunks + [clause_chunk])
    searcher = Searcher(client)
    results = searcher.search("personal information", limit=10,
                              provision_type="section")
    assert all(r.chunk.provision_type == "section" for r in results)


def test_search_acts_returns_act_search_results(privacy_chunks):
    from lexausearch.indexer import Indexer
    from lexausearch.models import ActRecord
    client = QdrantClient(":memory:")
    indexer = Indexer(client)
    indexer.upsert_chunks(privacy_chunks)
    indexer.upsert_acts([ActRecord(
        act_name="Privacy Act 1988",
        frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
        year=1988, as_at_date="2026-01-01",
        section_count=2, schedule_clause_count=0,
    )])
    searcher = Searcher(client)
    results = searcher.search_acts("personal information privacy", limit=3)
    assert len(results) > 0
    assert results[0].record.act_name == "Privacy Act 1988"
    assert isinstance(results[0].score, float)
