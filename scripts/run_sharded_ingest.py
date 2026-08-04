#!/usr/bin/env python3
"""Drives colab_driver.py through a resumable, Act-count-sharded ingest of
the lex-au corpus. Fresh Colab T4 VM per shard for full RAM isolation - see
docs/superpowers/specs/2026-07-27-colab-sharded-ingest-design.md.

Safe to re-run: any shard whose local zip is missing (never attempted, or
attempted and failed) is retried; shards with an existing zip are skipped.
Prerequisite: Task 9 of that spec's plan must have already validated
colab_driver.py against a real Colab session.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import colab_driver as cd

POLL_INTERVAL_S = 30
REPO_URL = "https://github.com/cchew/lex-au-search.git"


def _shard_paths(shards_dir: Path, index: int) -> tuple[Path, Path]:
    return (
        shards_dir / f"shard_{index:03d}.zip",
        shards_dir / f"shard_{index:03d}_cache.zip",
    )


def run_shard(index: int, shard_size: int, shards_dir: Path, gpu: str) -> bool:
    zip_path, cache_zip_path = _shard_paths(shards_dir, index)
    if zip_path.exists():
        print(f"[shard {index}] already downloaded, skipping", file=sys.stderr)
        return True

    name = f"lexau-shard-{index}"
    print(f"[shard {index}] creating session ...", file=sys.stderr)
    created = cd.create_session(name, gpu=gpu)
    if not created["ok"]:
        print(f"[shard {index}] create_session failed: {created['stderr']}", file=sys.stderr)
        return False

    try:
        if not cd.verify_session(name):
            print(f"[shard {index}] verify_session failed (no CUDA) - stopping session", file=sys.stderr)
            return False

        remote_cmd = (
            f"rm -rf repo && git clone --depth 1 {REPO_URL} repo && "
            f"cd repo && bash scripts/colab_ingest_shard.sh {index} {shard_size}"
        )
        pid = cd.run_background(name, remote_cmd)
        print(f"[shard {index}] running as PID {pid} ...", file=sys.stderr)

        while True:
            time.sleep(POLL_INTERVAL_S)
            status = cd.poll_status(name, pid)
            if status == "running":
                continue
            if status == "done":
                break
            print(f"[shard {index}] failed - last log:\n{cd.tail_log(name)}", file=sys.stderr)
            return False

        shards_dir.mkdir(parents=True, exist_ok=True)
        ok_zip = cd.download(name, "repo/shard_storage.zip", str(zip_path))
        ok_cache = cd.download(name, "repo/shard_cache.zip", str(cache_zip_path))
        if not (ok_zip and ok_cache):
            print(f"[shard {index}] download failed", file=sys.stderr)
            return False
        print(f"[shard {index}] done -> {zip_path}", file=sys.stderr)
        return True
    except Exception as e:
        # A transient colab_driver failure (e.g. an exec call timing out on
        # real backend latency, see reference-google-colab-cli-kernelclient-bug
        # memory) must fail only this shard, not crash the whole multi-shard
        # run - subsequent shards, and a later retry of this one, still work.
        print(f"[shard {index}] unexpected error: {e}", file=sys.stderr)
        return False
    finally:
        cd.stop_session(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-acts", type=int, required=True,
                     help="Total Act count in the corpus (see lex-au/repo/corpus/index.json)")
    ap.add_argument("--shard-size", type=int, default=300)
    ap.add_argument("--shards-dir", type=Path, default=Path("./shards"))
    ap.add_argument("--gpu", default="T4")
    args = ap.parse_args()

    total_shards = ceil(args.total_acts / args.shard_size)
    results = {i: run_shard(i, args.shard_size, args.shards_dir, args.gpu) for i in range(total_shards)}

    failed = [i for i, ok in results.items() if not ok]
    print(json.dumps({"total_shards": total_shards, "failed_shards": failed}, indent=2))
    if failed:
        print(f"\n{len(failed)} shard(s) failed: {failed}. Re-run this script to retry them.",
              file=sys.stderr)
        sys.exit(1)

    storage_dirs = ",".join(
        str((args.shards_dir / f"shard_{i:03d}").with_suffix("")) for i in range(total_shards)
    )
    print(
        f"\nAll {total_shards} shards downloaded to {args.shards_dir}/. Unzip each shard_NNN.zip "
        f"and shard_NNN_cache.zip, then run:\n"
        f"  lex-au-search merge-shards "
        f"--shard-storage-dirs <unzipped storage dirs, comma-separated> "
        f"--shard-cache-paths <unzipped .db files, comma-separated> "
        f"--output-storage-dir ./qdrant_storage --output-cache-path ./embed_cache.db",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
