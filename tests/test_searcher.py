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
    indexer.upsert(privacy_chunks)
    return Searcher(client)


def test_search_returns_results(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_result_has_score(indexed_searcher):
    results = indexed_searcher.search("personal information", limit=2)
    assert all(isinstance(r.score, float) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


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


def test_search_act_filter_excludes_other_acts(privacy_chunks):
    extra_chunk = Chunk(
        act_name="Fair Work Act 2009",
        frbr_uri="/akn/au/act/2009/28/eng@2026-01-01",
        eid="sec-12",
        section_num="12",
        heading="Definitions",
        text="12 Definitions In this Act employee means a person employed.",
    )
    client = QdrantClient(":memory:")
    indexer = Indexer(client)
    indexer.upsert(privacy_chunks + [extra_chunk])
    searcher = Searcher(client)
    results = searcher.search("information", limit=5, act="Fair Work Act 2009")
    assert len(results) > 0
    assert all(r.chunk.act_name == "Fair Work Act 2009" for r in results)
