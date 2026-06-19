from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

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
    def handler(query: str, limit: int = 5, act: str | None = None) -> str:
        results = searcher.search(query, limit=limit, act=act)
        if not results:
            return "No results found."
        return format_results(results)
    return handler


async def run_mcp_server(searcher: Searcher) -> None:
    server = Server("lex-au-search")
    search_handler = make_search_tool_handler(searcher)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_legislation",
                description=(
                    "Search Australian Commonwealth legislation by topic or question. "
                    "Returns relevant sections with FRBR citations and relevance scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language question or topic to search for",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of sections to return",
                        },
                        "act": {
                            "type": "string",
                            "description": "Optional: restrict search to a specific Act by name (e.g. 'Privacy Act 1988')",
                        },
                    },
                    "required": ["query"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "search_legislation":
            output = search_handler(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                act=arguments.get("act"),
            )
            return [types.TextContent(type="text", text=output)]
        raise ValueError(f"Unknown tool: {name}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
