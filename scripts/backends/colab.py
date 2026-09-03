"""Colab backend: one fresh Colab GPU VM per shard, driven through the
`colab` CLI wrapper in scripts/colab_driver.py.

This is a behaviour-preserving lift of what used to be
`run_sharded_ingest.run_shard` - see
docs/superpowers/specs/2026-07-27-colab-sharded-ingest-design.md for why the
per-shard VM model exists (RAM isolation; free-tier ~60min prune).

HF-cache model (2026-09-03): the VM pulls its own seed via
`python -m lexausearch.hf_cache fetch` inside the remote command and pushes its
live embed cache back to HF via `colab_driver.exec_sync` at each checkpoint and
on completion - see
docs/superpowers/specs/2026-09-03-hf-cache-home-of-record-design.md.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import colab_driver as cd

from backends.base import IngestBackend, SeedMode, ShardResult, shard_paths
from lexausearch.models import DENSE_MODEL

POLL_INTERVAL_S = 30
REPO_URL = "https://github.com/cchew/lex-au-search.git"
# ~5min at POLL_INTERVAL_S=30s - bounds the worst-case lost work on a kill
# (confirmed 2026-08-21: free-tier sessions get pruned at ~60min regardless
# of resource usage or keep-alive health) to one checkpoint interval, not
# the whole run. Each interval the VM pushes its live embed cache to HF via
# _push_checkpoint().
CHECKPOINT_INTERVAL_POLLS = 10


class ColabBackend(IngestBackend):
    # Where ingest_shard.sh writes the live embed cache on the VM, relative to
    # the cloned repo root (the push command `cd /content/repo` first).
    _remote_cache_db_path = "shard_cache.db"

    def __init__(self, shards_dir: Path, gpu: str) -> None:
        self.shards_dir = shards_dir
        self.gpu = gpu

    def prepare(self) -> None:
        return None

    def teardown(self) -> None:
        cd._release_orphaned_assignments()

    def _push_checkpoint(
        self, name: str, index: int, overwrite: bool, status: str = "partial"
    ) -> None:
        """Best-effort: tell the VM to push its live embed cache to HF via
        `colab_driver.exec_sync`. Non-fatal on any failure; never raises."""
        cmd = (
            f"cd /content/repo && python -m lexausearch.hf_cache push {index} "
            f"--db {self._remote_cache_db_path} --status {status} --live "
            f"--model {DENSE_MODEL} --token-file /content/.hf_token"
            + (" --overwrite" if overwrite else "")
        )
        try:
            rc, _out, err = cd.exec_sync(name, cmd, timeout=180)
        except Exception as e:  # noqa: BLE001 - a checkpoint push must never break the run
            print(
                f"[colab] shard {index} checkpoint push errored (non-fatal): {e}",
                file=sys.stderr,
            )
            return
        if rc != 0:
            print(
                f"[colab] shard {index} checkpoint push failed (non-fatal): {err}",
                file=sys.stderr,
            )

    def run_shard(self, index: int, shard_size: int, seed_mode: SeedMode) -> ShardResult:
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

            # Ship the HF write token to the VM so its own `hf_cache fetch`/`push`
            # calls (in remote_cmd and via _push_checkpoint) can authenticate.
            # 0600 temp file -> /content/.hf_token (survives remote_cmd's
            # `rm -rf repo`). No token set == read-only run; skip silently.
            token = os.environ.get("HF_CACHE_WRITE_TOKEN")
            if token:
                with tempfile.NamedTemporaryFile("w", delete=False) as tf:
                    tf.write(token)
                    tok_path = tf.name
                os.chmod(tok_path, 0o600)
                try:
                    if not cd.upload(name, tok_path, "/content/.hf_token"):
                        print(f"[shard {index}] HF token upload to the VM failed", file=sys.stderr)
                        return fail("HF token upload to the VM failed")
                finally:
                    os.unlink(tok_path)

            overwrite = seed_mode is SeedMode.SEEDLESS_OVERWRITE
            fetch_line = ""
            if seed_mode is SeedMode.HF:
                fetch_line = (
                    f" && python -m lexausearch.hf_cache fetch {index} "
                    f"--dest /content --seed-as /content/shard_cache_seed.db "
                    f"--token-file /content/.hf_token "
                    f"--expect-model {DENSE_MODEL}"
                )
            remote_cmd = (
                f"rm -rf repo && git clone --depth 1 {REPO_URL} repo && cd repo "
                f"&& bash scripts/setup_gpu_env.sh"
                f"{fetch_line}"
                f" && bash scripts/ingest_shard.sh {index} {shard_size} /content/shard_cache_seed.db"
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
                        self._push_checkpoint(name, index, overwrite)
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
            self._push_checkpoint(name, index, overwrite, status="complete")
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
