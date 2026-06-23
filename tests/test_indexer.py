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
