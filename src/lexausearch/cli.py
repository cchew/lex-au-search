from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path

import click
from qdrant_client import QdrantClient

from lexausearch.cache import EmbedCache, merge_cache_files
from lexausearch.chunker import (
    chunk_corpus, chunk_corpus_for_acts, load_corpus_act_names,
    load_corpus_act_entries, shard_bounds, missing_acts, compute_act_content_hashes,
)
from lexausearch.indexer import DENSE_MODEL, Indexer, merge_shard_clients, fetch_all_act_hashes, delete_act, COLLECTION_ACTS
from lexausearch.models import ActRecord, Chunk
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


def _act_record_from_chunks(act_name: str, chunks: list[Chunk], content_hash: str) -> ActRecord:
    """Build one ActRecord from an Act's own chunk list. Shared by
    _run_ingest (whole-batch path, called once per ingest run across all
    Acts) and ingest-delta (Task 3, called once per changed Act)."""
    frbr_uri = chunks[0].frbr_uri
    return ActRecord(
        act_name=act_name,
        frbr_uri=frbr_uri,
        year=_year_from_frbr_uri(frbr_uri),
        as_at_date=_as_at_from_frbr_uri(frbr_uri),
        section_count=sum(1 for c in chunks if c.provision_type == "section"),
        schedule_clause_count=sum(1 for c in chunks if c.provision_type == "schedule_clause"),
        content_hash=content_hash,
    )


def _run_ingest(
    chunks: list[Chunk], storage_dir: Path, cache_path: Path, gap_check_names: set[str],
    corpus_dir: Path,
) -> None:
    """Shared indexing + reporting logic for `ingest` and `ingest-shard`."""
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

    act_hashes = compute_act_content_hashes(corpus_dir, set(act_chunks.keys()))
    act_records = [
        _act_record_from_chunks(act_name, act_chunk_list, act_hashes.get(act_name, ""))
        for act_name, act_chunk_list in act_chunks.items()
    ]

    click.echo(f"Indexing {len(chunks)} chunks into {storage_dir} ...")
    click.echo(f"Embedding cache: {cache_path} (persists across runs)")
    client = QdrantClient(path=str(storage_dir))
    indexer = Indexer(client, cache=EmbedCache(cache_path, model_name=DENSE_MODEL))
    act_names = list(act_chunks.keys())
    for i, act_name in enumerate(act_names, 1):
        act_chunk_list = act_chunks[act_name]
        click.echo(f"  [{i}/{len(act_names)}] {act_name} ({len(act_chunk_list)} chunks)")
        indexer.upsert_chunks(act_chunk_list)
    click.echo(f"  Indexing {len(act_records)} Acts into legislation collection ...")
    indexer.upsert_acts(act_records)

    gap = missing_acts(gap_check_names, set(act_chunks.keys()))
    if gap:
        click.echo(
            f"WARNING: {len(gap)} of {len(gap_check_names)} Acts produced zero "
            f"chunks and were not indexed:"
        )
        for name in gap[:10]:
            click.echo(f"  - {name}")
        if len(gap) > 10:
            click.echo(f"  ... and {len(gap) - 10} more")

    click.echo(
        f"Done. {len(chunks)} chunks + {len(act_records)} Act records indexed "
        f"({len(act_records)} of {len(gap_check_names)} Acts)."
    )
    click.echo(
        f"Embedding cache: {indexer.cache_hits} hits, {indexer.cache_misses} misses "
        f"({indexer.cache_hits} chunks skipped re-embedding)."
    )


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
@click.option(
    "--cache-path",
    default="./embed_cache.db",
    type=click.Path(path_type=Path),
    show_default=True,
    help=(
        "Path to a persistent SQLite embedding cache file. Unlike --storage-dir, "
        "this is never deleted between runs - it accumulates content-addressed "
        "vectors so re-ingesting unchanged text skips re-embedding."
    ),
)
def ingest(corpus_dir: Path, storage_dir: Path, cache_path: Path) -> None:
    """Build Qdrant index from lex-au AKN corpus."""
    click.echo(f"Chunking corpus at {corpus_dir} ...")
    chunks = chunk_corpus(corpus_dir)
    corpus_act_names = load_corpus_act_names(corpus_dir)
    _run_ingest(chunks, storage_dir, cache_path, corpus_act_names, corpus_dir)


@cli.command(name="ingest-shard")
@click.option(
    "--corpus-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to lex-au corpus directory (contains index.json and xml/)",
)
@click.option(
    "--storage-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Shard-local Qdrant storage directory (not shared across shards)",
)
@click.option(
    "--cache-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Shard-local embedding cache path (merged into a master cache by merge-shards)",
)
@click.option("--shard-index", required=True, type=int, help="0-based shard index")
@click.option("--shard-size", required=True, type=int, help="Acts per shard")
def ingest_shard(
    corpus_dir: Path, storage_dir: Path, cache_path: Path, shard_index: int, shard_size: int
) -> None:
    """Chunk + index only one Act-count slice of the corpus (bounded memory).

    Acts are sliced from index.json's stable order: shard N covers Acts
    [N*shard_size : (N+1)*shard_size). Run once per shard, each in its own
    process (ideally its own VM) so memory never accumulates across the
    whole corpus.
    """
    entries = load_corpus_act_entries(corpus_dir)
    start, end = shard_bounds(len(entries), shard_index, shard_size)
    shard_entries = entries[start:end]
    if not shard_entries:
        click.echo(
            f"Shard {shard_index} is empty (corpus has {len(entries)} Acts, "
            f"shard-size {shard_size}); nothing to do."
        )
        return
    shard_act_names = {e["name"] for e in shard_entries}
    click.echo(
        f"Shard {shard_index}: Acts [{start}:{end}] of {len(entries)} "
        f"({len(shard_act_names)} Acts) ..."
    )
    chunks = chunk_corpus_for_acts(corpus_dir, shard_act_names)
    _run_ingest(chunks, storage_dir, cache_path, shard_act_names, corpus_dir)


@cli.command(name="merge-shards")
@click.option(
    "--shard-storage-dirs", required=True,
    help="Comma-separated list of shard-local Qdrant storage directories to merge",
)
@click.option(
    "--shard-cache-paths", required=True,
    help="Comma-separated list of shard-local embed_cache.db files to merge",
)
@click.option(
    "--output-storage-dir", required=True, type=click.Path(path_type=Path),
    help="Path for the final merged Qdrant storage directory",
)
@click.option(
    "--output-cache-path", required=True, type=click.Path(path_type=Path),
    help="Path for the final merged embedding cache",
)
def merge_shards_cmd(
    shard_storage_dirs: str, shard_cache_paths: str, output_storage_dir: Path, output_cache_path: Path
) -> None:
    """Merge all completed shards' local storage + cache into one final
    qdrant_storage/ + embed_cache.db (same shape serve/mcp already expect).
    Runs entirely on CPU - no GPU needed, this only copies existing points
    and cached vectors, it never re-embeds."""
    storage_dirs = [Path(p.strip()) for p in shard_storage_dirs.split(",")]
    cache_paths = [Path(p.strip()) for p in shard_cache_paths.split(",")]

    click.echo(f"Merging {len(storage_dirs)} shard(s) into {output_storage_dir} ...")
    shard_clients = [QdrantClient(path=str(d)) for d in storage_dirs]
    output_client = QdrantClient(path=str(output_storage_dir))
    totals = merge_shard_clients(shard_clients, output_client)
    click.echo(f"  {totals['sections']} chunks + {totals['acts']} Act records merged.")

    click.echo(f"Merging {len(cache_paths)} shard cache(s) into {output_cache_path} ...")
    rows = merge_cache_files(cache_paths, output_cache_path)
    click.echo(f"  {rows} cache rows merged (deduplicated by content-addressed key).")

    click.echo("Done.")


@cli.command(name="ingest-delta")
@click.option(
    "--corpus-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to lex-au corpus directory (contains index.json and xml/)",
)
@click.option(
    "--storage-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to an EXISTING Qdrant local storage directory. Unlike `ingest`, "
         "this is never deleted -- it must already hold a prior `ingest`/"
         "`ingest-shard` (+ `merge-shards`) run.",
)
@click.option(
    "--cache-path",
    default="./embed_cache.db",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Path to a persistent SQLite embedding cache file (same semantics as `ingest`).",
)
def ingest_delta(corpus_dir: Path, storage_dir: Path, cache_path: Path) -> None:
    """Re-index only Acts whose content hash changed since the last ingest."""
    try:
        client = QdrantClient(path=str(storage_dir))
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            raise click.ClickException(
                f"{storage_dir} is locked by another process (e.g. `serve` or "
                f"`mcp` holding --storage-dir open). Stop that process first, "
                f"then re-run ingest-delta."
            )
        raise
    try:
        if not client.collection_exists(COLLECTION_ACTS):
            raise click.ClickException(
                f"{storage_dir} has no indexed Acts yet -- run `ingest` (or the "
                f"sharded pipeline) first before using ingest-delta."
            )

        click.echo(f"Hashing corpus at {corpus_dir} ...")
        current = compute_act_content_hashes(corpus_dir)
        click.echo(f"Reading indexed Act hashes from {storage_dir} ...")
        indexed = fetch_all_act_hashes(client)

        changed_or_new = sorted(name for name, h in current.items() if indexed.get(name) != h)
        unchanged_count = len(current) - len(changed_or_new)
        click.echo(
            f"{len(changed_or_new)} Act(s) changed or new, {unchanged_count} unchanged (skipped)."
        )
        if not changed_or_new:
            click.echo("Nothing to do.")
            return

        click.echo(f"Embedding cache: {cache_path} (persists across runs)")
        indexer = Indexer(client, cache=EmbedCache(cache_path, model_name=DENSE_MODEL))

        reindexed_count = 0
        skipped: list[str] = []
        failed: list[str] = []

        for i, act_name in enumerate(changed_or_new, 1):
            click.echo(f"  [{i}/{len(changed_or_new)}] {act_name}")
            try:
                # Chunk BEFORE deleting: a corpus regression that makes an Act
                # produce zero chunks must not delete that Act's existing index
                # entry with nothing to replace it -- stale-but-present beats
                # missing. Only delete once we know we have chunks in hand.
                act_chunk_list = chunk_corpus_for_acts(corpus_dir, {act_name})
                if not act_chunk_list:
                    click.echo(
                        f"    WARNING: {act_name} produced zero chunks, not "
                        f"re-indexed (existing index entry left in place)."
                    )
                    skipped.append(act_name)
                    continue
                delete_act(client, act_name)
                indexer.upsert_chunks(act_chunk_list)
                indexer.upsert_acts([_act_record_from_chunks(act_name, act_chunk_list, current[act_name])])
                reindexed_count += 1
            except Exception as e:
                click.echo(f"    ERROR: {act_name} failed, leaving prior index entry in place: {e}")
                failed.append(act_name)
                continue

        click.echo(
            f"Done. {reindexed_count} Act(s) re-indexed, {unchanged_count} unchanged (skipped)."
        )
        click.echo(
            f"Embedding cache: {indexer.cache_hits} hits, {indexer.cache_misses} misses "
            f"({indexer.cache_hits} chunks skipped re-embedding)."
        )
        if skipped:
            click.echo(f"Skipped (zero chunks): {', '.join(skipped)}")
        if failed:
            click.echo(f"Failed: {', '.join(failed)}")
        if skipped or failed:
            raise click.ClickException(
                f"{len(skipped)} Act(s) skipped and {len(failed)} Act(s) failed -- "
                f"index is incomplete for these Acts (prior entries left in place)."
            )
    finally:
        # Release the exclusive local-mode lock deterministically -- relying
        # on GC to drop `client` is unreliable once an exception's traceback
        # keeps this frame (and `client`) alive, e.g. via CliRunner's
        # captured exc_info in tests, or any other caller holding the error.
        client.close()


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
