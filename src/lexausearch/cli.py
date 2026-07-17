from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path

import click
from qdrant_client import QdrantClient

from lexausearch.chunker import chunk_corpus, load_corpus_act_names, missing_acts
from lexausearch.indexer import Indexer
from lexausearch.models import ActRecord
from lexausearch.searcher import Searcher
from lexausearch.api import create_app
from lexausearch.mcp import run_mcp_server


def _year_from_frbr_uri(frbr_uri: str) -> int:
    # /akn/au/act/1988/119/eng@...  →  1988
    parts = frbr_uri.split("/")
    try:
        return int(parts[4])
    except (IndexError, ValueError):
        return 0


def _as_at_from_frbr_uri(frbr_uri: str) -> str:
    # /akn/au/act/1988/119/eng@2026-01-01  →  "2026-01-01"
    if "@" in frbr_uri:
        return frbr_uri.split("@")[-1].split("/")[0]
    return ""


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
    sections = [c for c in chunks if c.provision_type == "section"]
    subsections = [c for c in chunks if c.provision_type == "subsection"]
    clauses = [c for c in chunks if c.provision_type == "schedule_clause"]
    click.echo(
        f"  {len(sections)} sections + {len(subsections)} subsections "
        f"+ {len(clauses)} schedule clauses = {len(chunks)} total chunks across all Acts."
    )

    # Build ActRecord list from chunk list
    act_chunks: dict[str, list] = defaultdict(list)
    for c in chunks:
        act_chunks[c.act_name].append(c)

    act_records: list[ActRecord] = []
    for act_name, act_chunk_list in act_chunks.items():
        frbr_uri = act_chunk_list[0].frbr_uri
        year = _year_from_frbr_uri(frbr_uri)
        as_at = _as_at_from_frbr_uri(frbr_uri)
        section_count = sum(1 for c in act_chunk_list if c.provision_type == "section")
        clause_count = sum(1 for c in act_chunk_list if c.provision_type == "schedule_clause")
        act_records.append(ActRecord(
            act_name=act_name,
            frbr_uri=frbr_uri,
            year=year,
            as_at_date=as_at,
            section_count=section_count,
            schedule_clause_count=clause_count,
        ))

    click.echo(f"Indexing {len(chunks)} chunks into {storage_dir} ...")
    client = QdrantClient(path=str(storage_dir))
    indexer = Indexer(client)
    act_names = list(act_chunks.keys())
    for i, act_name in enumerate(act_names, 1):
        act_chunk_list = act_chunks[act_name]
        click.echo(f"  [{i}/{len(act_names)}] {act_name} ({len(act_chunk_list)} chunks)")
        indexer.upsert_chunks(act_chunk_list)
    click.echo(f"  Indexing {len(act_records)} Acts into legislation collection ...")
    indexer.upsert_acts(act_records)

    corpus_act_names = load_corpus_act_names(corpus_dir)
    gap = missing_acts(corpus_act_names, set(act_chunks.keys()))
    if gap:
        click.echo(
            f"WARNING: {len(gap)} of {len(corpus_act_names)} corpus Acts produced zero "
            f"chunks and were not indexed:"
        )
        for name in gap[:10]:
            click.echo(f"  - {name}")
        if len(gap) > 10:
            click.echo(f"  ... and {len(gap) - 10} more")

    click.echo(
        f"Done. {len(chunks)} chunks + {len(act_records)} Act records indexed "
        f"({len(act_records)} of {len(corpus_act_names)} corpus Acts)."
    )


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
    app = create_app(searcher, client)
    uvicorn.run(app, host=host, port=port)


@cli.command()
def mcp() -> None:
    """Run MCP stdio server for Claude Code integration."""
    asyncio.run(run_mcp_server())
