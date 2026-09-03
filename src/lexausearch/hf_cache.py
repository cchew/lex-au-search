"""Hugging Face dataset repo as the home of record for sharded-ingest embed
caches. Pure functions over HF_CACHE_REPO; no backend state. Runnable on a
compute VM as `python -m lexausearch.hf_cache <verb>` from any cwd.

Imports are deliberately light (no onnxruntime / qdrant_client) so the ingest
orchestrator can `import lexausearch.hf_cache` cheaply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

HF_CACHE_REPO = "cchew/lex-au-search-embed-cache"
_HUB_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"


@dataclass(frozen=True)
class ShardCacheMeta:
    model_name: str
    row_count: int
    generation: int
    updated_at: str
    sha256: str
    status: str  # "partial" | "complete"


@dataclass(frozen=True)
class Catalogue:
    dense_model: str
    shard_size: int
    total_acts: int
    master: dict


class HfCacheCorrupt(Exception):
    pass


class HfCacheModelMismatch(Exception):
    pass


class ModelGuardVerdict:
    pass


class Ok(ModelGuardVerdict):
    pass


class Cold(ModelGuardVerdict):
    pass


@dataclass
class Mismatch(ModelGuardVerdict):
    old: str
    new: str


def _shard_db_name(i: int) -> str:
    return f"shard_{i:03d}_checkpoint_cache.db"


def _shard_json_name(i: int) -> str:
    return f"shard_{i:03d}.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_token(explicit: str | None, token_file: str | None) -> str | None:
    if explicit:
        return explicit
    if token_file and Path(token_file).is_file():
        return Path(token_file).read_text().strip()
    if _HUB_TOKEN_PATH.is_file():
        return _HUB_TOKEN_PATH.read_text().strip()
    return None


def _hf_download(repo: str, filename: str, token: str | None) -> str:
    """Thin wrapper so tests can monkeypatch a single seam."""
    return hf_hub_download(
        repo_id=repo, repo_type="dataset", filename=filename,
        token=token, force_download=True,
    )


def read_catalogue(*, token: str | None) -> Catalogue | None:
    try:
        path = _hf_download(HF_CACHE_REPO, "catalogue.json", token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    data = json.loads(Path(path).read_text())
    return Catalogue(
        dense_model=data["dense_model"],
        shard_size=data["shard_size"],
        total_acts=data["total_acts"],
        master=data["master"],
    )


def _read_sidecar(shard_index: int, token: str | None) -> ShardCacheMeta | None:
    try:
        path = _hf_download(HF_CACHE_REPO, _shard_json_name(shard_index), token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    d = json.loads(Path(path).read_text())
    return ShardCacheMeta(
        model_name=d["model_name"], row_count=d["row_count"],
        generation=d["generation"], updated_at=d["updated_at"],
        sha256=d["sha256"], status=d["status"],
    )


def check_model(shard_index: int, current_model: str, *, token: str | None) -> ModelGuardVerdict:
    meta = _read_sidecar(shard_index, token)
    if meta is None:
        return Cold()
    if meta.model_name == current_model:
        return Ok()
    return Mismatch(old=meta.model_name, new=current_model)
