from __future__ import annotations

import asyncio
from pathlib import Path

import click
from qdrant_client import QdrantClient

from lexausearch.chunker import chunk_corpus
from lexausearch.indexer import Indexer
from lexausearch.searcher import Searcher
from lexausearch.api import create_app
from lexausearch.mcp import get_storage_path, run_mcp_server


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--corpus-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to lex-au corpus directory (contains index.json and xml/)",
)
@click.option(
    "--storage-dir",
    default="./qdrant_storage",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to Qdrant local storage directory",
)
def ingest(corpus_dir: Path, storage_dir: Path) -> None:
    """Build Qdrant index from lex-au AKN corpus."""
    click.echo(f"Chunking corpus at {corpus_dir} ...")
    chunks = chunk_corpus(corpus_dir)
    click.echo(f"  {len(chunks)} sections found across all Acts.")
    click.echo(f"Indexing into {storage_dir} ...")
    client = QdrantClient(path=str(storage_dir))
    Indexer(client).upsert(chunks)
    click.echo(f"Done. {len(chunks)} chunks indexed.")


@cli.command()
@click.option(
    "--storage-dir",
    default="./qdrant_storage",
    type=click.Path(path_type=Path),
    show_default=True,
)
@click.option("--port", default=8000, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
def serve(storage_dir: Path, port: int, host: str) -> None:
    """Run FastAPI search server."""
    import uvicorn
    client = QdrantClient(path=str(storage_dir))
    searcher = Searcher(client)
    app = create_app(searcher)
    uvicorn.run(app, host=host, port=port)


@cli.command()
def mcp() -> None:
    """Run MCP stdio server for Claude Code integration."""
    storage = get_storage_path()
    client = QdrantClient(path=str(storage))
    searcher = Searcher(client)
    asyncio.run(run_mcp_server(searcher))
