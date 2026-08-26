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
import tempfile
import time
import zipfile
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import colab_driver as cd
from lexausearch.cache import merge_cache_files

POLL_INTERVAL_S = 30
REPO_URL = "https://github.com/cchew/lex-au-search.git"
# ~5min at POLL_INTERVAL_S=30s - bounds the worst-case lost work on a kill
# (confirmed 2026-08-21: free-tier sessions get pruned at ~60min regardless
# of resource usage or keep-alive health) to one checkpoint interval, not
# the whole run. See checkpoint_cache()'s docstring in colab_driver.py.
CHECKPOINT_INTERVAL_POLLS = 10


def _shard_paths(shards_dir: Path, index: int) -> tuple[Path, Path]:
    return (
        shards_dir / f"shard_{index:03d}.zip",
        shards_dir / f"shard_{index:03d}_cache.zip",
    )


def _checkpoint_cache_path(shards_dir: Path, index: int) -> Path:
    # Deliberately never deleted (unlike the zips above, which only appear
    # on full success) - accumulates across every partial attempt at this
    # shard so a retry can reseed from however far the last attempt got.
    return shards_dir / f"shard_{index:03d}_checkpoint_cache.db"


def _pull_checkpoint(name: str, index: int, shards_dir: Path) -> None:
    """Best-effort: snapshot the remote embed cache and fold it into this
    shard's local checkpoint accumulator. Never raises - a failed or skipped
    checkpoint cycle just means the next one tries again; it must not
    disturb the run it's monitoring."""
    status = cd.checkpoint_cache(name)
    if status != "ok":
        print(f"[shard {index}] checkpoint: {status}", file=sys.stderr)
        return
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "checkpoint.zip"
        if not cd.download(name, "repo/shard_cache_checkpoint.zip", str(zip_path)):
            print(f"[shard {index}] checkpoint: download failed", file=sys.stderr)
            return
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract("shard_cache_checkpoint.db", tmp)
        except (zipfile.BadZipFile, KeyError) as e:
            print(f"[shard {index}] checkpoint: bad zip - {e}", file=sys.stderr)
            return
        extracted = Path(tmp) / "shard_cache_checkpoint.db"
        shards_dir.mkdir(parents=True, exist_ok=True)
        rows = merge_cache_files([extracted], _checkpoint_cache_path(shards_dir, index))
    print(f"[shard {index}] checkpoint: merged ({rows} rows read)", file=sys.stderr)


def run_shard(index: int, shard_size: int, shards_dir: Path, gpu: str) -> bool:
    zip_path, cache_zip_path = _shard_paths(shards_dir, index)
    if zip_path.exists():
        print(f"[shard {index}] already downloaded, skipping", file=sys.stderr)
        return True

    name = f"lexau-shard-{index}"
    try:
        print(f"[shard {index}] creating session ...", file=sys.stderr)
        created = cd.create_session(name, gpu=gpu)
        if not created["ok"]:
            # `colab new` can register a real session both on Colab's backend
            # and in the CLI's own local registry even when this subprocess
            # call exits nonzero for an unrelated reason afterward - don't
            # skip cleanup just because we can't tell from here. The
            # `finally` below unconditionally sweeps orphaned assignments
            # (confirmed necessary 2026-08-27: this exact path leaked a
            # session three times in one session, each cascading into every
            # later shard failing with TooManyAssignmentsError).
            print(f"[shard {index}] create_session failed: {created['stderr']}", file=sys.stderr)
            return False

        if not cd.verify_session(name):
            print(f"[shard {index}] verify_session failed (no CUDA) - stopping session", file=sys.stderr)
            return False

        checkpoint_cache_path = _checkpoint_cache_path(shards_dir, index)
        if checkpoint_cache_path.exists():
            print(
                f"[shard {index}] seeding remote cache from prior partial "
                f"attempt ({checkpoint_cache_path}) ...", file=sys.stderr,
            )
            # /content/ (not repo/) - survives the remote_cmd's `rm -rf repo`
            # below, which colab_ingest_shard.sh relies on to find it.
            if not cd.upload(name, str(checkpoint_cache_path), "/content/shard_cache_seed.db"):
                print(f"[shard {index}] seed upload failed - continuing without it", file=sys.stderr)

        remote_cmd = (
            f"rm -rf repo && git clone --depth 1 {REPO_URL} repo && "
            f"cd repo && bash scripts/colab_ingest_shard.sh {index} {shard_size}"
        )
        pid = cd.run_background(name, remote_cmd)
        print(f"[shard {index}] running as PID {pid} ...", file=sys.stderr)

        poll_count = 0
        while True:
            time.sleep(POLL_INTERVAL_S)
            status = cd.poll_status(name, pid)
            if status == "running":
                print(f"[shard {index}] {cd.sample_resources(name)}", file=sys.stderr)
                poll_count += 1
                if poll_count % CHECKPOINT_INTERVAL_POLLS == 0:
                    _pull_checkpoint(name, index, shards_dir)
                continue
            if status == "done":
                break
            if status == "session_lost":
                # VM/kernel is unreachable - diagnose_failure and tail_log
                # would just re-exec against the same dead session (extra
                # timeout latency, no new information). No progress is
                # preserved for this shard; it must be retried from scratch.
                print(
                    f"[shard {index}] session lost mid-run - Colab-side "
                    f"disconnect/preemption, job outcome unknown, no evidence "
                    f"of OOM or other in-process cause. Retry from scratch.",
                    file=sys.stderr,
                )
                return False
            diagnosis = cd.diagnose_failure(name, pid)
            print(f"[shard {index}] failed - {diagnosis}", file=sys.stderr)
            print(f"[shard {index}] last log:\n{cd.tail_log(name, n=200)}", file=sys.stderr)
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
