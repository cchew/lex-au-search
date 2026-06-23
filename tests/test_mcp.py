import os
import pytest
from unittest.mock import MagicMock

from lexausearch.models import Chunk, SearchResult, format_results


def _make_result(eid: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid=eid,
            provision_num=eid.split("-")[-1],
            provision_type="section",
            heading="Interpretation",
            text="In this Act personal information means...",
            refs=[],
        ),
        score=score,
    )


def _make_searcher(results=None):
    searcher = MagicMock()
    searcher.search.return_value = results or [
        SearchResult(
            chunk=Chunk(
                act_name="Privacy Act 1988",
                frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
                eid="sec-6", provision_num="6", provision_type="section",
                heading="Definitions", text="personal information", refs=[],
            ),
            score=0.9,
        )
    ]
    return searcher


# --- New tests (Task 7) ---

def test_mcp_module_importable():
    import lexausearch.mcp  # must not raise


def test_get_storage_path_raises_without_env(monkeypatch):
    monkeypatch.delenv("LEXAU_SEARCH_STORAGE", raising=False)
    from lexausearch.mcp import get_storage_path
    with pytest.raises(SystemExit):
        get_storage_path()


def test_get_storage_path_returns_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXAU_SEARCH_STORAGE", str(tmp_path))
    from lexausearch.mcp import get_storage_path
    assert get_storage_path() == tmp_path


# --- Existing tests (preserved) ---

def test_search_tool_output_contains_section_reference():
    from lexausearch.mcp import make_search_tool_handler
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [_make_result("sec-3", 0.85)]
    handler = make_search_tool_handler(mock_searcher)
    output = handler(query="personal information", limit=5, act=None)
    assert "Privacy Act 1988" in output
    assert "sec-3" in output


def test_search_tool_output_contains_score():
    from lexausearch.mcp import make_search_tool_handler
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [_make_result("sec-3", 0.85)]
    handler = make_search_tool_handler(mock_searcher)
    output = handler(query="personal information", limit=5, act=None)
    assert "0.85" in output


def test_search_tool_passes_act_filter():
    from lexausearch.mcp import make_search_tool_handler
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = []
    handler = make_search_tool_handler(mock_searcher)
    handler(query="test", limit=3, act="Privacy Act 1988")
    mock_searcher.search.assert_called_once_with(
        "test", limit=3, act="Privacy Act 1988"
    )


def test_missing_storage_env_var_raises():
    from lexausearch.mcp import get_storage_path
    original = os.environ.pop("LEXAU_SEARCH_STORAGE", None)
    try:
        with pytest.raises(SystemExit, match="LEXAU_SEARCH_STORAGE"):
            get_storage_path()
    finally:
        if original is not None:
            os.environ["LEXAU_SEARCH_STORAGE"] = original


def test_storage_env_var_returns_path():
    from lexausearch.mcp import get_storage_path
    os.environ["LEXAU_SEARCH_STORAGE"] = "/tmp/test_storage"
    try:
        path = get_storage_path()
        assert str(path) == "/tmp/test_storage"
    finally:
        del os.environ["LEXAU_SEARCH_STORAGE"]
