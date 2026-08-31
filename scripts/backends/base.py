"""Backend-agnostic contract for sharded ingest.

A backend owns one shard's execution end to end: provision compute, run
`lex-au-search ingest-shard` for that Act-count slice, snapshot progress, and
hand back the storage + cache zips. `run_sharded_ingest.py` is the
backend-agnostic orchestrator on top.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShardResult:
    index: int
    ok: bool
    storage_zip: Path | None
    cache_zip: Path | None
    diagnosis: str


def shard_paths(shards_dir: Path, index: int) -> tuple[Path, Path]:
    return (
        shards_dir / f"shard_{index:03d}.zip",
        shards_dir / f"shard_{index:03d}_cache.zip",
    )


def checkpoint_cache_path(shards_dir: Path, index: int) -> Path:
    """Never deleted (unlike the zips, which only appear on full success) -
    accumulates across every partial attempt at this shard so a retry can
    reseed from however far the last attempt got. `check_ingest_status.py`
    reads exactly this path; do not rename or relocate it."""
    return shards_dir / f"shard_{index:03d}_checkpoint_cache.db"


class IngestBackend(ABC):
    @abstractmethod
    def prepare(self) -> None:
        """Run once before any shard. Colab: no-op."""

    @abstractmethod
    def run_shard(self, index: int, shard_size: int, seed_cache: Path | None) -> ShardResult:
        """Own the entire per-shard lifecycle for this backend.

        Best-effort accumulator contract: on a FAILED return the backend has
        merged whatever it could recover into
        `checkpoint_cache_path(shards_dir, index)` via
        `lexausearch.cache.merge_cache_files` - progress up to the last
        successful mid-run snapshot (the final interval may be lost, matching
        Colab's existing `session_lost` behaviour).

        Canonical zip-path contract: on an `ok=True` return the backend MUST
        have written both the storage zip and the cache zip to
        `shard_paths(shards_dir, index)` (the same `shards_dir` the backend
        was constructed with); `run_sharded_ingest.run_all` decides a shard is
        already done on the next run by testing exactly
        `shard_paths(shards_dir, index)[0].exists()`.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Run once after all shards, including on an exception mid-loop.
        Idempotent."""
