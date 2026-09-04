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
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError

_DEFAULT_HF_CACHE_REPO = "cchew/lex-au-search-embed-cache"
# Resolution order: `--repo` on the CLI > LEXAU_HF_CACHE_REPO in the env >
# the default. The env var is what carries a smoke run's throwaway repo to a
# compute VM, where `python -m lexausearch.hf_cache` runs as a fresh process
# and cannot see the laptop's in-process rebind of this name.
HF_CACHE_REPO = os.environ.get("LEXAU_HF_CACHE_REPO") or _DEFAULT_HF_CACHE_REPO
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


def _sha256_file(path: Path) -> str:
    # Chunked: this runs on the push path against a 700MiB+ DB, on a ~12GB
    # compute VM alongside a live embed job. Never slurp the file.
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0]
    finally:
        conn.close()


def fetch_shard_cache(shard_index: int, dest: Path, *, token: str | None,
                      expect_model: str | None = None,
                      seed_as: str | None = None) -> ShardCacheMeta | None:
    meta = _read_sidecar(shard_index, token)
    if meta is None:
        return None
    if expect_model is not None and meta.model_name != expect_model:
        raise HfCacheModelMismatch(
            f"shard {shard_index} HF cache model {meta.model_name!r} != expected {expect_model!r}"
        )
    try:
        db_src = _hf_download(HF_CACHE_REPO, _shard_db_name(shard_index), token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    db_src = Path(db_src)
    if _sha256_file(db_src) != meta.sha256:
        raise HfCacheCorrupt(f"shard {shard_index} DB sha256 mismatch vs sidecar")
    actual = _row_count(db_src)
    if actual != meta.row_count:
        raise HfCacheCorrupt(
            f"shard {shard_index} DB has {actual} rows, sidecar says {meta.row_count}"
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / (seed_as or _shard_db_name(shard_index))
    shutil.copyfile(db_src, target)
    return meta


# Imported lazily inside push to keep module import light for callers that
# never push (orchestrator pre-flight).
def _merge_cache_files(paths, out):
    from lexausearch.cache import merge_cache_files
    return merge_cache_files(paths, out)


merge_cache_files = _merge_cache_files  # test seam


def _api() -> HfApi:
    return HfApi()


def _sqlite_backup(src: Path, dst: Path) -> None:
    s = sqlite3.connect(str(src))
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


def _write_sidecar(path: Path, meta: ShardCacheMeta) -> None:
    path.write_text(json.dumps({
        "model_name": meta.model_name, "row_count": meta.row_count,
        "generation": meta.generation, "updated_at": meta.updated_at,
        "sha256": meta.sha256, "status": meta.status,
    }, indent=2))


def push_shard_cache(shard_index: int, local_db: Path, *, model_name: str,
                     status: str, token: str, overwrite: bool = False,
                     live: bool = False) -> ShardCacheMeta:
    local_db = Path(local_db)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        working = tdp / "working.db"
        if live:
            _sqlite_backup(local_db, working)
        else:
            shutil.copyfile(local_db, working)

        head = _read_sidecar(shard_index, token)
        if not overwrite and head is not None:
            if head.model_name != model_name:
                raise HfCacheModelMismatch(
                    f"shard {shard_index} HF head model {head.model_name!r} != {model_name!r}; "
                    f"refusing to merge (use the override path to replace)"
                )
            head_db = Path(_hf_download(HF_CACHE_REPO, _shard_db_name(shard_index), token))
            merged = tdp / "merged.db"
            merge_cache_files([head_db, working], merged)
            upload_db = merged
        else:
            upload_db = working

        gen = 1 if head is None else head.generation + 1
        meta = ShardCacheMeta(
            model_name=model_name, row_count=_row_count(upload_db),
            generation=gen, updated_at=_now(),
            sha256=_sha256_file(upload_db), status=status,
        )
        sidecar_path = tdp / _shard_json_name(shard_index)
        _write_sidecar(sidecar_path, meta)

        ops = [
            CommitOperationAdd(path_in_repo=_shard_db_name(shard_index), path_or_fileobj=str(upload_db)),
            CommitOperationAdd(path_in_repo=_shard_json_name(shard_index), path_or_fileobj=str(sidecar_path)),
        ]
        # NOTE: this retry branch is currently UNREACHABLE. `create_commit` is
        # called without `parent_commit`, so no optimistic-concurrency
        # precondition is sent and the Hub never answers 412. It is kept as the
        # placeholder for the FUTURE parallel-execution work (FUTURE items 5-6),
        # which is what would send a parent_commit. Under the present sequential
        # model, safety comes from pull-merge-push being commutative and
        # idempotent, not from this branch. See spec 3.2.1.
        for attempt in (1, 2):
            try:
                _api().create_commit(
                    repo_id=HF_CACHE_REPO, repo_type="dataset", operations=ops,
                    commit_message=f"shard {shard_index}: gen {gen} ({status})",
                    token=token,
                )
                break
            except HfHubHTTPError as e:
                if attempt == 1 and getattr(e.response, "status_code", None) == 412:
                    head = _read_sidecar(shard_index, token)
                    gen = 1 if head is None else head.generation + 1
                    meta = ShardCacheMeta(**{**meta.__dict__, "generation": gen})
                    _write_sidecar(sidecar_path, meta)
                    continue
                raise
        return meta


_GITATTRIBUTES = (
    "*_checkpoint_cache.db filter=lfs diff=lfs merge=lfs -text\n"
    "embed_cache.db filter=lfs diff=lfs merge=lfs -text\n"
)


def _local_marker_generation(shards_dir: Path, shard_index: int) -> int:
    p = shards_dir / _shard_json_name(shard_index)
    if not p.is_file():
        return -1
    try:
        return int(json.loads(p.read_text()).get("generation", -1))
    except (ValueError, json.JSONDecodeError):
        return -1


def mirror_to_local(shard_index: int, shards_dir: Path, *, token: str | None) -> None:
    shards_dir = Path(shards_dir)
    try:
        meta = _read_sidecar(shard_index, token)
        if meta is None:
            return
        if meta.generation <= _local_marker_generation(shards_dir, shard_index):
            return
        db_src = Path(_hf_download(HF_CACHE_REPO, _shard_db_name(shard_index), token))
        shards_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(db_src, shards_dir / _shard_db_name(shard_index))
        _write_sidecar(shards_dir / _shard_json_name(shard_index), meta)
    except Exception as e:  # noqa: BLE001 - mirror is best-effort
        print(f"[hf_cache] mirror of shard {shard_index} failed (non-fatal): {e}",
              file=sys.stderr)


def create_cache_repo(*, token: str) -> None:
    api = _api()
    api.create_repo(repo_id=HF_CACHE_REPO, repo_type="dataset", exist_ok=True, token=token)
    try:
        _hf_download(HF_CACHE_REPO, ".gitattributes", token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        with tempfile.TemporaryDirectory() as td:
            ga = Path(td) / ".gitattributes"
            ga.write_text(_GITATTRIBUTES)
            api.upload_file(
                path_or_fileobj=str(ga), path_in_repo=".gitattributes",
                repo_id=HF_CACHE_REPO, repo_type="dataset", token=token,
                commit_message="add .gitattributes (LFS patterns)",
            )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m lexausearch.hf_cache")
    sub = ap.add_subparsers(dest="verb", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("index", type=int)
    f.add_argument("--dest", required=True)
    f.add_argument("--seed-as", default=None)
    f.add_argument("--expect-model", default=None)
    f.add_argument("--token", default=None)
    f.add_argument("--token-file", default=None)
    f.add_argument("--repo", default=None,
                   help="HF dataset repo to read (overrides LEXAU_HF_CACHE_REPO)")

    p = sub.add_parser("push")
    p.add_argument("index", type=int)
    p.add_argument("--db", required=True)
    p.add_argument("--status", choices=["partial", "complete"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--token", default=None)
    p.add_argument("--token-file", default=None)
    p.add_argument("--repo", default=None,
                   help="HF dataset repo to write (overrides LEXAU_HF_CACHE_REPO)")
    return ap


def main(argv=None) -> int:
    global HF_CACHE_REPO
    args = _build_parser().parse_args(argv)
    # `--repo` wins over the env-resolved module default. Set before any HF call
    # so every read/write in this process targets the overridden repo.
    if args.repo:
        HF_CACHE_REPO = args.repo
    token = _resolve_token(args.token, args.token_file)
    if args.verb == "fetch":
        try:
            meta = fetch_shard_cache(args.index, Path(args.dest), token=token,
                                     expect_model=args.expect_model, seed_as=args.seed_as)
            print("cold start (no HF cache)" if meta is None
                  else f"fetched shard {args.index} gen {meta.generation} ({meta.row_count} rows)")
            return 0
        except HfCacheModelMismatch as e:
            print(f"MODEL MISMATCH: {e}", file=sys.stderr)
            return 4
        except HfCacheCorrupt as e:
            print(f"CORRUPT: {e}", file=sys.stderr)
            return 5
        except Exception as e:  # noqa: BLE001
            print(f"fetch failed: {e}", file=sys.stderr)
            return 1
    # push
    try:
        meta = push_shard_cache(args.index, Path(args.db), model_name=args.model,
                                status=args.status, token=token,
                                overwrite=args.overwrite, live=args.live)
        print(f"pushed shard {args.index} gen {meta.generation} ({meta.row_count} rows)")
        return 0
    except HfCacheModelMismatch as e:
        print(f"HEAD MODEL MISMATCH: {e}", file=sys.stderr)
        return 4
    except Exception as e:  # noqa: BLE001
        print(f"push failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
