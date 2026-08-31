"""Colab backend: one fresh Colab GPU VM per shard, driven through the
`colab` CLI wrapper in scripts/colab_driver.py.

This is a behaviour-preserving lift of what used to be
`run_sharded_ingest.run_shard` - see
docs/superpowers/specs/2026-07-27-colab-sharded-ingest-design.md for why the
per-shard VM model exists (RAM isolation; free-tier ~60min prune).
"""

from __future__ import annotations

import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import colab_driver as cd
from lexausearch.cache import merge_cache_files

from backends.base import IngestBackend, ShardResult, checkpoint_cache_path, shard_paths

POLL_INTERVAL_S = 30
REPO_URL = "https://github.com/cchew/lex-au-search.git"
# ~5min at POLL_INTERVAL_S=30s - bounds the worst-case lost work on a kill
# (confirmed 2026-08-21: free-tier sessions get pruned at ~60min regardless
# of resource usage or keep-alive health) to one checkpoint interval, not
# the whole run. See checkpoint_cache()'s docstring in colab_driver.py.
CHECKPOINT_INTERVAL_POLLS = 10


class ColabBackend(IngestBackend):
    def __init__(self, shards_dir: Path, gpu: str) -> None:
        self.shards_dir = shards_dir
        self.gpu = gpu

    def prepare(self) -> None:
        return None

    def teardown(self) -> None:
        cd._release_orphaned_assignments()

    def _pull_checkpoint(self, name: str, index: int) -> None:
        """Best-effort: snapshot the remote embed cache and fold it into this
        shard's local accumulator. Never raises."""
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
            self.shards_dir.mkdir(parents=True, exist_ok=True)
            rows = merge_cache_files([extracted], checkpoint_cache_path(self.shards_dir, index))
        print(f"[shard {index}] checkpoint: merged ({rows} rows read)", file=sys.stderr)

    def run_shard(self, index: int, shard_size: int, seed_cache: Path | None) -> ShardResult:
        zip_path, cache_zip_path = shard_paths(self.shards_dir, index)
        name = f"lexau-shard-{index}"

        def fail(msg: str) -> ShardResult:
            return ShardResult(index, False, None, None, msg)

        try:
            print(f"[shard {index}] creating session ...", file=sys.stderr)
            created = cd.create_session(name, gpu=self.gpu)
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
                return fail(f"create_session failed: {created['stderr']}")

            if not cd.verify_session(name):
                print(f"[shard {index}] verify_session failed (no CUDA)", file=sys.stderr)
                return fail("verify_session failed (no CUDA)")

            if seed_cache is not None and seed_cache.exists():
                print(f"[shard {index}] seeding remote cache from {seed_cache} ...", file=sys.stderr)
                # /content/ (not repo/) - survives the remote_cmd's `rm -rf repo`
                # below, which colab_ingest_shard.sh relies on to find it.
                if not cd.upload(name, str(seed_cache), "/content/shard_cache_seed.db"):
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
                        self._pull_checkpoint(name, index)
                    continue
                if status == "done":
                    break
                if status == "session_lost":
                    # VM/kernel is unreachable - diagnose_failure and tail_log
                    # would just re-exec against the same dead session (extra
                    # timeout latency, no new information). No progress is
                    # preserved for this shard; it must be retried from scratch.
                    msg = (
                        "session lost mid-run - Colab-side disconnect/preemption, "
                        "job outcome unknown, no evidence of OOM or other in-process "
                        "cause. Retry from scratch."
                    )
                    print(f"[shard {index}] {msg}", file=sys.stderr)
                    return fail(msg)
                diagnosis = cd.diagnose_failure(name, pid)
                print(f"[shard {index}] failed - {diagnosis}", file=sys.stderr)
                print(f"[shard {index}] last log:\n{cd.tail_log(name, n=200)}", file=sys.stderr)
                return fail(f"failed - {diagnosis}")

            self.shards_dir.mkdir(parents=True, exist_ok=True)
            ok_zip = cd.download(name, "repo/shard_storage.zip", str(zip_path))
            ok_cache = cd.download(name, "repo/shard_cache.zip", str(cache_zip_path))
            if not (ok_zip and ok_cache):
                print(f"[shard {index}] download failed", file=sys.stderr)
                return fail("download failed")
            print(f"[shard {index}] done -> {zip_path}", file=sys.stderr)
            return ShardResult(index, True, zip_path, cache_zip_path, "")
        except Exception as e:  # noqa: BLE001 - one shard's failure must not crash the run
            # A transient colab_driver failure (e.g. an exec call timing out on
            # real backend latency, see reference-google-colab-cli-kernelclient-bug
            # memory) must fail only this shard, not crash the whole multi-shard
            # run - subsequent shards, and a later retry of this one, still work.
            print(f"[shard {index}] unexpected error: {e}", file=sys.stderr)
            return fail(f"unexpected error: {e}")
        finally:
            cd.stop_session(name)
