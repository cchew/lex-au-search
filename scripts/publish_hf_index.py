#!/usr/bin/env python3
"""Publish the finished Qdrant index (not the embed cache) to HF as a
downloadable, ready-to-query artifact.

Distinct from `hf_cache.py` / `bootstrap_hf_cache.py`, which manage
`cchew/lex-au-search-embed-cache` (content-hash -> vector, for resumable
ingest). This script publishes `cchew/lex-au-search-index`: the raw
`qdrant_storage/` directory tree as-is, so a consumer can
`snapshot_download` it straight into place and run
`lex-au-search serve --storage-dir <downloaded path>` with no re-ingest.

Usage:
    HF_HUB_DISABLE_XET=1 python scripts/publish_hf_index.py \\
        --storage-dir ./qdrant_storage --readme hf-index-readme.md

`HF_HUB_DISABLE_XET=1` matters on a residential uplink - HF's Xet transport
has wedged twice on this connection (see FUTURE.md / decisions log); classic
LFS multipart with real timeouts held ~200-400 kB/s reliably instead.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _envload import load_env

REPO_ID = "cchew/lex-au-search-index"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", type=Path, default=Path("qdrant_storage"))
    parser.add_argument("--readme", type=Path, default=Path("hf-index-readme.md"))
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env(__file__)
    token = os.environ["HF_CACHE_WRITE_TOKEN"]

    if not args.storage_dir.is_dir():
        parser.error(f"{args.storage_dir} does not exist")
    if not args.readme.is_file():
        parser.error(f"{args.readme} does not exist")

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    if args.dry_run:
        total = sum(f.stat().st_size for f in args.storage_dir.rglob("*") if f.is_file())
        print(f"DRY RUN - would create/update {args.repo_id} (dataset, public)")
        print(f"  upload_folder({args.storage_dir}) -> {total / 1e9:.2f} GB")
        print(f"  upload_file({args.readme}) -> README.md")
        return

    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)

    print(f"Uploading {args.storage_dir} -> {args.repo_id} ...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.storage_dir),
        path_in_repo="qdrant_storage",
        commit_message="Publish v0.5.0 index (534,335 chunks, 3,073 Acts)",
    )
    print("Index uploaded.")

    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(args.readme),
        path_in_repo="README.md",
        commit_message="Update dataset card",
    )
    print("README pushed. Done.")


if __name__ == "__main__":
    main()
