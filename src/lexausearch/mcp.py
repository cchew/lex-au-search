from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable

from fastmcp import FastMCP
from qdrant_client import QdrantClient

from lexausearch.api import create_app
from lexausearch.models import format_results
from lexausearch.searcher import Searcher


def get_storage_path() -> Path:
    storage = os.environ.get("LEXAU_SEARCH_STORAGE")
    if not storage:
        raise SystemExit(
            "LEXAU_SEARCH_STORAGE env var is not set. "
            "Set it to the absolute path of your qdrant_storage directory."
        )
    return Path(storage)


def make_search_tool_handler(searcher: Searcher) -> Callable:
    """Return a callable search handler for direct use in tests and legacy wiring."""
    def handler(query: str, limit: int = 5, act: str | None = None) -> str:
        results = searcher.search(query, limit=limit, act=act)
        if not results:
            return "No results found."
        return format_results(results)
    return handler


async def run_mcp_server() -> None:
    storage = get_storage_path()
    client = QdrantClient(path=str(storage))
    searcher = Searcher(client)
    app = create_app(searcher, client)
    mcp = FastMCP.from_fastapi(app, name="lex-au-search")
    await mcp.run_async(transport="stdio")
