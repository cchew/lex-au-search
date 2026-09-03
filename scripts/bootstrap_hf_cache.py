#!/usr/bin/env python3
"""One-time: populate the HF embed-cache repo from local shards/*.db.

Runs from the laptop over the home uplink (~1-2h for the full set; uploads
chunk and resume). See
docs/superpowers/specs/2026-09-03-hf-cache-home-of-record-design.md §8.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _envload import load_env
from lexausearch import hf_cache
from lexausearch.models import DENSE_MODEL
from huggingface_hub import CommitOperationAdd


def _build_entries(shards_dir: Path, indices: list[int], model: str) -> list[dict]:
    out = []
    for i in indices:
        db = shards_dir / hf_cache._shard_db_name(i)
        if not db.is_file():
            continue
        with tempfile.TemporaryDirectory() as td:
            flat = Path(td) / "flat.db"
            hf_cache._sqlite_backup(db, flat)
            rows = hf_cache._row_count(flat)
            sha = hf_cache._sha256_file(flat)
        status = "complete" if (shards_dir / f"shard_{i:03d}.zip").is_file() else "partial"
        out.append({"index": i, "db": db, "sidecar": {
            "model_name": model, "row_count": rows, "generation": 1,
            "updated_at": hf_cache._now(), "sha256": sha, "status": status,
        }})
    return out


def _plan(entries, catalogue, dry_run, token):
    if dry_run:
        print("DRY RUN — would commit:")
        for e in entries:
            print(f"  shard {e['index']}: {hf_cache._shard_db_name(e['index'])} "
                  f"+ {hf_cache._shard_json_name(e['index'])} "
                  f"({e['sidecar']['row_count']} rows, {e['sidecar']['status']})")
        print(f"  catalogue.json: {catalogue}")
        return
    api = hf_cache._api()
    for e in entries:
        with tempfile.TemporaryDirectory() as td:
            flat = Path(td) / "flat.db"
            hf_cache._sqlite_backup(e["db"], flat)
            sc = Path(td) / hf_cache._shard_json_name(e["index"])
            sc.write_text(json.dumps(e["sidecar"], indent=2))
            api.create_commit(
                repo_id=hf_cache.HF_CACHE_REPO, repo_type="dataset",
                operations=[
                    CommitOperationAdd(hf_cache._shard_db_name(e["index"]), str(flat)),
                    CommitOperationAdd(hf_cache._shard_json_name(e["index"]), str(sc)),
                ],
                commit_message=f"bootstrap shard {e['index']}", token=token,
            )
            print(f"  uploaded shard {e['index']}")
    with tempfile.TemporaryDirectory() as td:
        cj = Path(td) / "catalogue.json"
        cj.write_text(json.dumps(catalogue, indent=2))
        api.upload_file(path_or_fileobj=str(cj), path_in_repo="catalogue.json",
                        repo_id=hf_cache.HF_CACHE_REPO, repo_type="dataset",
                        token=token, commit_message="bootstrap catalogue")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards-dir", type=Path, default=Path("./shards"))
    ap.add_argument("--total-acts", type=int, default=3076)
    ap.add_argument("--shard-size", type=int, default=300)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    load_env(__file__)
    token = os.environ.get("HF_CACHE_WRITE_TOKEN")
    if not token:
        print("HF_CACHE_WRITE_TOKEN is not set (see .env.example)", file=sys.stderr)
        return 2

    n = ceil(args.total_acts / args.shard_size)
    indices = list(range(n))
    present = [i for i in indices if (args.shards_dir / hf_cache._shard_db_name(i)).is_file()]
    missing = [i for i in indices if i not in present]
    if missing and not args.allow_partial:
        print(f"missing local caches for shards {missing} of {n}; "
              f"re-run with --allow-partial to bootstrap only {present}", file=sys.stderr)
        print(f"missing: {missing}")
        return 1

    print(f"About to stamp every cache with DENSE_MODEL = {DENSE_MODEL}")
    if not args.yes:
        if input("Proceed? type 'yes': ").strip() != "yes":
            return 1

    hf_cache.create_cache_repo(token=token)
    entries = _build_entries(args.shards_dir, present, DENSE_MODEL)
    catalogue = {"dense_model": DENSE_MODEL, "shard_size": args.shard_size,
                 "total_acts": args.total_acts,
                 "master": {"row_count": 0, "generation": 0, "updated_at": None}}
    _plan(entries, catalogue, args.dry_run, token)
    print(f"done: {len(entries)} shard cache(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
