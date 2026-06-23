import pytest
import httpx
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from lexausearch.api import create_app
from lexausearch.indexer import Indexer
from lexausearch.models import ActRecord, Chunk
from lexausearch.searcher import Searcher


@pytest.fixture(scope="module")
def api_client(privacy_chunks):
    client = QdrantClient(":memory:")
    Indexer(client).upsert_chunks(privacy_chunks)
    searcher = Searcher(client)
    app = create_app(searcher, client)
    return TestClient(app)


@pytest.fixture(scope="module")
def full_app():
    client = QdrantClient(":memory:")
    chunks = [
        Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid="sec-3",
            provision_num="3",
            provision_type="section",
            heading="Interpretation",
            text="personal information means",
            refs=[],
        ),
        Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid="schedule-1__clause-1",
            provision_num="1",
            provision_type="schedule_clause",
            heading="Open management",
            text="APP entities must manage information openly",
            refs=[],
        ),
    ]
    act_records = [
        ActRecord(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            year=1988,
            as_at_date="2026-01-01",
            section_count=1,
            schedule_clause_count=1,
        )
    ]
    idx = Indexer(client)
    idx.upsert_chunks(chunks)
    idx.upsert_acts(act_records)
    searcher = Searcher(client)
    return create_app(searcher, client)


def test_search_returns_200(api_client):
    response = api_client.get("/search?q=personal+information")
    assert response.status_code == 200


def test_search_response_shape(api_client):
    response = api_client.get("/search?q=personal+information&limit=2")
    data = response.json()
    assert "results" in data
    result = data["results"][0]
    assert "act_name" in result
    assert "frbr_uri" in result
    assert "eid" in result
    assert "provision_num" in result
    assert "provision_type" in result
    assert "heading" in result
    assert "text" in result
    assert "score" in result


def test_search_limit_parameter(api_client):
    response = api_client.get("/search?q=information&limit=1")
    data = response.json()
    assert len(data["results"]) <= 1


def test_search_act_filter(api_client):
    response = api_client.get("/search?q=information&act=Privacy+Act+1988")
    data = response.json()
    assert all(r["act_name"] == "Privacy Act 1988" for r in data["results"])


def test_search_missing_q_returns_422(api_client):
    response = api_client.get("/search")
    assert response.status_code == 422


async def test_get_legislation_act(full_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=full_app), base_url="http://test"
    ) as ac:
        r = await ac.get("/legislation/Privacy Act 1988")
    assert r.status_code == 200
    data = r.json()
    assert data["act_name"] == "Privacy Act 1988"
    assert data["year"] == 1988
    assert data["as_at_date"] == "2026-01-01"


async def test_get_legislation_act_not_found(full_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=full_app), base_url="http://test"
    ) as ac:
        r = await ac.get("/legislation/Nonexistent Act 2000")
    assert r.status_code == 404


async def test_get_legislation_sections(full_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=full_app), base_url="http://test"
    ) as ac:
        r = await ac.get("/legislation/Privacy Act 1988/sections")
    assert r.status_code == 200
    data = r.json()
    assert "chunks" in data
    assert len(data["chunks"]) == 2


async def test_get_legislation_text(full_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=full_app), base_url="http://test"
    ) as ac:
        r = await ac.get("/legislation/Privacy Act 1988/text")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "personal information" in r.text


async def test_search_provision_type_param(full_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=full_app), base_url="http://test"
    ) as ac:
        r = await ac.get("/search?q=personal+information&provision_type=section")
    assert r.status_code == 200
    data = r.json()
    if data["results"]:
        assert all(res["provision_type"] == "section" for res in data["results"])
