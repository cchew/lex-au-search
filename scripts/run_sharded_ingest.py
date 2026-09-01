#!/usr/bin/env python3
"""Backend-agnostic orchestrator for a resumable, Act-count-sharded ingest
of the lex-au corpus. Each shard is embedded on its own GPU (a fresh Colab
VM, or a subprocess on a standing RunPod pod) so RAM never accumulates
across the whole corpus in one process.

Safe to re-run: any shard whose local zip is missing (never attempted, or
attempted and failed) is retried; shards with an existing zip are skipped.

See docs/superpowers/specs/2026-08-31-runpod-ingest-backend-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runpod_driver
from _envload import load_env
from backends import get_backend
from backends.base import IngestBackend, checkpoint_cache_path, shard_paths


def run_all(
    backend: IngestBackend,
    run_indices: list[int],
    shard_size: int,
    shards_dir: Path,
) -> dict[int, bool]:
    results: dict[int, bool] = {}
    try:
        backend.prepare()
        for i in run_indices:
            if shard_paths(shards_dir, i)[0].exists():
                print(f"[shard {i}] already downloaded, skipping", file=sys.stderr)
                results[i] = True
                continue
            seed = checkpoint_cache_path(shards_dir, i)
            seed = seed if seed.exists() else None
            result = backend.run_shard(i, shard_size, seed)
            results[i] = result.ok
    finally:
        backend.teardown()
    return results


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-acts", type=int, default=None,
                    help="Total Act count in the corpus (see lex-au/repo/corpus/index.json). "
                         "Required unless --reap is given.")
    ap.add_argument("--shard-size", type=int, default=300)
    ap.add_argument("--shards-dir", type=Path, default=Path("./shards"))
    ap.add_argument("--backend", choices=["colab", "runpod"], default="colab")
    ap.add_argument("--gpu", default="T4",
                    help="Colab GPU type (Colab backend only; ignored for --backend runpod)")
    ap.add_argument("--reap", action="store_true",
                    help="Terminate any leftover lexau- RunPod pods, clear the pod-id "
                         "file, and exit without ingesting. Does not take the run lock - "
                         "it will also terminate a concurrently running ingest's pod.")
    ap.add_argument("--reuse-pod", action="store_true", dest="reuse_pod",
                    help="Reuse an existing live pod recorded in .runpod_pod instead of "
                         "refusing to run.")
    ap.add_argument("--keep-pod", action="store_true", dest="keep_pod",
                    help="Leave the RunPod pod running after ingest instead of terminating it.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip interactive confirmation prompts (RunPod backend).")
    ap.add_argument("--cloud-type", choices=["COMMUNITY", "SECURE"], default="COMMUNITY",
                    dest="cloud_type", help="RunPod cloud tier to launch pods on.")
    ap.add_argument("--skip-shards", default="",
                    help="Comma-separated shard indices to leave untouched this run. "
                         "Skipped shards are not attempted, not counted as failed, and "
                         "not listed in the merge instructions - add them by hand.")
    return ap


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.reap and args.total_acts is None:
        parser.error("--total-acts is required unless --reap is given")
    load_env(__file__)

    if args.reap:
        # NOTE: --reap does not take the run lock - it will also terminate a
        # concurrently running ingest's pod.
        for pod in runpod_driver.list_pods():
            pod_id = pod.get("id")
            name = str(pod.get("name", "") or "")
            if not pod_id:
                continue
            if name.startswith("lexau-") and pod.get("desiredStatus") == "RUNNING":
                print(f"reaping {pod_id} ({name})", file=sys.stderr)
                runpod_driver.terminate_pod(pod_id)
        (Path(args.shards_dir) / ".runpod_pod").unlink(missing_ok=True)
        sys.exit(0)

    skip = {int(x) for x in args.skip_shards.split(",") if x.strip() != ""}
    total_shards = ceil(args.total_acts / args.shard_size)
    run_indices = [i for i in range(total_shards) if i not in skip]
    if skip:
        print(f"skipping shard(s) {sorted(skip)} - handle manually and merge in by hand",
              file=sys.stderr)

    backend = get_backend(args.backend, args)
    results = run_all(backend, run_indices, args.shard_size, args.shards_dir)

    failed = [i for i, ok in results.items() if not ok]
    print(json.dumps({"total_shards": total_shards, "skipped_shards": sorted(skip),
                      "failed_shards": failed}, indent=2))
    if failed:
        print(f"\n{len(failed)} shard(s) failed: {failed}. Re-run this script to retry them.",
              file=sys.stderr)
        sys.exit(1)

    scope = f"All {len(run_indices)} attempted shards" if skip else f"All {total_shards} shards"
    print(
        f"\n{scope} downloaded to {args.shards_dir}/"
        + (f" (skipped {sorted(skip)} - merge those in manually)." if skip else ".")
        + " Unzip each shard_NNN.zip "
        f"and shard_NNN_cache.zip, then run:\n"
        f"  lex-au-search merge-shards "
        f"--shard-storage-dirs <unzipped storage dirs, comma-separated> "
        f"--shard-cache-paths <unzipped .db files, comma-separated> "
        f"--output-storage-dir ./qdrant_storage --output-cache-path ./embed_cache.db",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
