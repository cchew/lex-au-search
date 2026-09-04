#!/usr/bin/env python3
"""One-shot status check for the sharded Colab ingest, replacing the ad-hoc
sequence of git/shell/sqlite/colab-cli commands otherwise run by hand each
session. Reports, per shard: COMPLETE (zip downloaded), PARTIAL (checkpoint
cache exists but no zip - a retry will reseed from it), or NOT STARTED.

Usage:
    python3 scripts/check_ingest_status.py --total-acts 3076
    python3 scripts/check_ingest_status.py --total-acts 3076 --probe-quota

--probe-quota briefly creates and immediately stops a real Colab GPU
session to check current T4 availability - costs a few seconds of quota,
skip it if that's a concern.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import colab_driver as cd


def _corpus_act_count(index_path: Path) -> int | None:
    if not index_path.exists():
        return None
    with open(index_path) as f:
        data = json.load(f)
    return len(data["acts"]) if isinstance(data, dict) else len(data)


def _checkpoint_rows(db_path: Path) -> int | None:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            return conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def shard_status(index: int, shards_dir: Path) -> dict:
    zip_path = shards_dir / f"shard_{index:03d}.zip"
    cache_zip_path = shards_dir / f"shard_{index:03d}_cache.zip"
    checkpoint_path = shards_dir / f"shard_{index:03d}_checkpoint_cache.db"

    if zip_path.exists() and cache_zip_path.exists():
        return {"index": index, "state": "COMPLETE", "detail": f"{zip_path.stat().st_size // 1_000_000}MB"}
    if checkpoint_path.exists():
        rows = _checkpoint_rows(checkpoint_path)
        detail = f"{rows} rows cached" if rows is not None else "checkpoint present, unreadable row count"
        return {"index": index, "state": "PARTIAL", "detail": detail}
    return {"index": index, "state": "NOT STARTED", "detail": ""}


def print_hf_section(shards_dir: Path) -> None:
    """Spec §6.2 columns: shard | generation | row_count | model | status |
    updated_at | local_stale?. `local_stale?` is yes when this laptop's mirror
    marker generation is behind HF's - i.e. `shards/` no longer reflects the
    home of record. Degrades to one line when HF is unreachable/offline."""
    print("\nHF cache:")
    try:
        from lexausearch import hf_cache
        cat = hf_cache.read_catalogue(token=None)
        if cat is None:
            print("  (no catalogue.json in the HF repo yet)")
            return
        n = -(-cat.total_acts // cat.shard_size)
        print(f"  {'shard':>5} {'gen':>4} {'rows':>8} {'status':>9} "
              f"{'updated_at':>21} {'local_stale?':>13}  model")
        for i in range(n):
            m = hf_cache._read_sidecar(i, None)
            if m is None:
                print(f"  {i:>5} {'-':>4} {'-':>8} {'absent':>9} "
                      f"{'-':>21} {'-':>13}")
                continue
            stale = hf_cache._local_marker_generation(Path(shards_dir), i) < m.generation
            print(f"  {i:>5} {m.generation:>4} {m.row_count:>8} {m.status:>9} "
                  f"{m.updated_at:>21} {('yes' if stale else 'no'):>13}  {m.model_name}")
    except Exception as e:  # noqa: BLE001
        print(f"  unreachable ({type(e).__name__}: {e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-acts", type=int, default=None,
                     help="Total Act count; auto-detected from lex-au/repo/corpus/index.json if omitted")
    ap.add_argument("--shard-size", type=int, default=300)
    ap.add_argument("--shards-dir", type=Path, default=Path("./shards"))
    ap.add_argument("--corpus-index", type=Path, default=Path("../lex-au/repo/corpus/index.json"))
    ap.add_argument("--backend", choices=["colab", "runpod"], default="colab",
                    help="Which ingest backend; --probe-quota only applies to colab.")
    ap.add_argument("--probe-quota", action="store_true",
                     help="Create+stop a real T4 session to check GPU availability")
    ap.add_argument("--no-hf", action="store_true",
                     help="Skip the HF cache section in the report")
    args = ap.parse_args()

    total_acts = args.total_acts or _corpus_act_count(args.corpus_index)
    if total_acts is None:
        print(f"error: --total-acts not given and {args.corpus_index} not found", file=sys.stderr)
        sys.exit(1)

    total_shards = ceil(total_acts / args.shard_size)
    shards = [shard_status(i, args.shards_dir) for i in range(total_shards)]

    result = {
        "total_acts": total_acts,
        "shard_size": args.shard_size,
        "total_shards": total_shards,
        "shards": shards,
        "complete": [s["index"] for s in shards if s["state"] == "COMPLETE"],
        "partial": [s["index"] for s in shards if s["state"] == "PARTIAL"],
        "not_started": [s["index"] for s in shards if s["state"] == "NOT STARTED"],
    }

    if args.probe_quota:
        if args.backend == "runpod":
            print("probe-quota is not applicable for the runpod backend")
            return
        name = "quota-probe-status-check"
        created = cd.create_session(name, gpu="T4")
        if created["ok"]:
            cd.stop_session(name)
            result["gpu_quota"] = "available"
        else:
            result["gpu_quota"] = f"unavailable: {created['stderr'][:200]}"

    print(json.dumps(result, indent=2))

    remaining = result["partial"] + result["not_started"]
    if remaining:
        print(
            f"\n{len(result['complete'])}/{total_shards} shards complete. "
            f"Resume with:\n  python3 scripts/run_sharded_ingest.py "
            f"--total-acts {total_acts} --shard-size {args.shard_size}",
            file=sys.stderr,
        )
    else:
        print(f"\nAll {total_shards} shards complete - ready for merge-shards.", file=sys.stderr)

    if not args.no_hf:
        print_hf_section(args.shards_dir)


if __name__ == "__main__":
    main()
