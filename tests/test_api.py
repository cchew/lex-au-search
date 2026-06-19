import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from lexausearch.api import create_app
from lexausearch.indexer import Indexer
from lexausearch.searcher import Searcher


@pytest.fixture(scope="module")
def api_client(privacy_chunks):
    client = QdrantClient(":memory:")
    Indexer(client).upsert(privacy_chunks)
    searcher = Searcher(client)
    app = create_app(searcher)
    return TestClient(app)


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
    assert "section_num" in result
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
