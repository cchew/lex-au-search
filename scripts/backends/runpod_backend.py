"""RunPod backend: one standing, non-preemptible A6000 pod reused across
shards, driven through the stdlib-only `scripts/runpod_driver.py` REST wrapper
and a real OpenSSH client.

Task 8 delivers `__init__`, `_acquire_lock`, the `_ssh`/`_ssh_check` helpers and
full `prepare()` with the layered billing-leak guards (flock, preflight orphan
check, pod-id file, atexit + ordered signal handlers). `run_shard()` and
`teardown()` are Task 9.

See docs/superpowers/specs/2026-08-31-runpod-ingest-backend-design.md section 5.
"""

from __future__ import annotations

import atexit
import fcntl
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import runpod_driver as _runpod_driver
from lexausearch.cache import merge_cache_files

from backends.base import (
    IngestBackend,
    ShardResult,
    checkpoint_cache_path,
    shard_paths,
)

# Exceptions that mean "the SSH round-trip did not complete": OSError (ssh binary
# missing / connection refused), subprocess.SubprocessError (TimeoutExpired), and
# the RuntimeError _ssh() raises on a non-zero remote exit.
_SSH_ERRORS = (OSError, RuntimeError, subprocess.SubprocessError)

REPO_URL = "https://github.com/cchew/lex-au-search.git"
REMOTE_DIR = "~/lex-au-search"
WAIT_READY_TIMEOUT_S = 600
DASHBOARD_SSH_KEYS_URL = "https://www.runpod.io/console/user/settings"
DASHBOARD_PODS_URL = "https://www.runpod.io/console/pods"
# Ballpark A6000 on-demand rate for the cost-gate estimate only (spike 2026-09-01:
# COMMUNITY ~$0.33/hr, SECURE ~$0.53/hr). Not used for billing.
_APPROX_RATE_AUD_PER_HOUR = {"COMMUNITY": 0.55, "SECURE": 0.85}
_APPROX_SHARD0_HOURS = 6


class RunPodBackend(IngestBackend):
    def __init__(
        self,
        shards_dir: Path,
        *,
        ssh_key: str,
        cloud_type: str = "COMMUNITY",
        gpu_type_id: str = "NVIDIA RTX A6000",
        assume_yes: bool = False,
        reuse_pod: bool = False,
        keep_pod: bool = False,
        batch_size: int = 16,
        _rd=_runpod_driver,
        _run=subprocess.run,
    ) -> None:
        self.shards_dir = Path(shards_dir)
        self.ssh_key = ssh_key
        self.cloud_type = cloud_type
        self.gpu_type_id = gpu_type_id
        self.assume_yes = assume_yes
        self.reuse_pod = reuse_pod
        self.keep_pod = keep_pod
        self.batch_size = batch_size
        self._rd = _rd
        self._run = _run

        self.POD_FILE = self.shards_dir / ".runpod_pod"
        self.LOCK_FILE = self.shards_dir / ".runpod_pod.lock"

        self._pod_id: str | None = None
        self._ip: str = ""
        self._port: int = 0
        self._teardown_ran: bool = False
        self._lock_fh = None
        self._signalled: bool = False
        self._atexit_registered: bool = False
        self._signals_registered: bool = False
        # Injectable seams (also used by Task 9); keep the GPU-idle retry wait,
        # the run_shard poll loop, and process-global signal registration
        # stubbable in tests.
        self._sleep = time.sleep
        self._signal = signal.signal
        # Set lazily to `time.monotonic() + 1500` the first time run_shard's poll
        # loop hits an SSH failure; the wall-clock bound past which an
        # unreachable pod is abandoned. Tests pin it to force the bound.
        self._deadline_s: float | None = None

    # ------------------------------------------------------------------ locks
    def _acquire_lock(self) -> None:
        """flock LOCK_FILE non-blocking, held for the process lifetime. On
        contention another RunPod ingest run owns the machine -> hard exit."""
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        fh = open(self.LOCK_FILE, "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            sys.exit("another RunPod ingest run is in progress")
        self._lock_fh = fh

    # ------------------------------------------------------------------- ssh
    def _ssh(self, remote: str, *, check: bool = True):
        cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=60s",
            "-i", os.path.expanduser(self.ssh_key),
            f"root@{self._ip}",
            "-p", str(self._port),
            remote,
        ]
        res = self._run(cmd, capture_output=True, text=True)
        if check and getattr(res, "returncode", 0) != 0:
            raise RuntimeError(
                f"ssh command failed (rc={getattr(res, 'returncode', '?')}): {remote}\n"
                f"{getattr(res, 'stderr', '') or ''}"
            )
        return res

    def _ssh_check(self, remote: str) -> str:
        """Run a remote command, return its stdout (stripped). Raises on
        non-zero exit."""
        return (getattr(self._ssh(remote), "stdout", "") or "").strip()

    # -------------------------------------------------------------- prepare()
    def prepare(self) -> None:
        self._acquire_lock()
        self._preflight_orphan_check()
        self._cost_gate()

        pod_file_id = (
            self.POD_FILE.read_text().strip()
            if self.POD_FILE.exists()
            else None
        )

        reusing = False
        if (
            self.reuse_pod
            and pod_file_id
            and self._rd.get_status(pod_file_id) == "RUNNING"
        ):
            self._pod_id = pod_file_id
            reusing = True
            print(f"[runpod] reusing pod {pod_file_id} (--reuse-pod)", file=sys.stderr)
            self._register_cleanup()
        else:
            cfg = self._rd.CreatePodConfig(
                name=self._rd.pod_name(),
                gpu_type_id=self.gpu_type_id,
                cloud_type=self.cloud_type,
            )
            pod = self._rd.create_pod(cfg)
            self._pod_id = pod["id"]
            # The instant a pod id exists it is billable. Register the atexit
            # teardown hook FIRST, before the POD_FILE write, so a write failure
            # (disk full / perms) still leaves a path to terminate the pod.
            self._register_atexit()
            self.POD_FILE.write_text(f"{self._pod_id}\n")
            self._register_signal_handlers()
            print(f"[runpod] created pod {self._pod_id}", file=sys.stderr)

        self._ip, self._port = self._rd.wait_ready(
            self._pod_id, timeout_s=WAIT_READY_TIMEOUT_S
        )
        print(f"[runpod] pod ssh-ready at {self._ip}:{self._port}", file=sys.stderr)

        # 6b: one real ssh round-trip to prove key auth works.
        try:
            self._ssh("true")
        except Exception:
            print(
                "[runpod] SSH auth to the pod failed. Register your key's public "
                f"half in the RunPod dashboard ({DASHBOARD_SSH_KEYS_URL} -> SSH "
                "Keys) so RunPod injects it into the pod, then retry.",
                file=sys.stderr,
            )
            raise

        # 6c: fresh pod only - clone the repo and run the one-time env setup.
        if not reusing:
            self._ssh(
                f"git clone --depth 1 {REPO_URL} {REMOTE_DIR} && "
                f"cd {REMOTE_DIR} && bash scripts/setup_gpu_env.sh"
            )

        # 6d: no prior detached ingest job may still hold the GPU.
        self._ensure_gpu_idle()

        # 6e: CUDA execution provider must be present.
        self._ssh(
            'python3 -c "import onnxruntime as o; assert '
            "'CUDAExecutionProvider' in o.get_available_providers()\""
        )
        print("[runpod] prepare complete", file=sys.stderr)

    # --------------------------------------------------------- prepare guts
    def _preflight_orphan_check(self) -> None:
        pod_file_id = (
            self.POD_FILE.read_text().strip()
            if self.POD_FILE.exists()
            else None
        )
        # A prior unclean exit leaves POD_FILE pointing at a still-RUNNING pod.
        # Without --reuse-pod, letting prepare() continue would create a second
        # pod and overwrite the id, leaking the first one until the next --reap.
        if (
            pod_file_id
            and not self.reuse_pod
            and self._rd.get_status(pod_file_id) == "RUNNING"
        ):
            sys.exit(
                f"refusing to start: {self.POD_FILE} names pod {pod_file_id!r}, "
                "which the RunPod API still reports as RUNNING (a prior run likely "
                "exited uncleanly). Resume it with --reuse-pod, or terminate it and "
                "clear the file with:\n"
                "  python scripts/run_sharded_ingest.py --backend runpod --reap"
            )
        for pod in self._rd.list_pods() or []:
            name = pod.get("name", "") or ""
            if not name.startswith("lexau-"):
                continue
            if pod.get("desiredStatus") != "RUNNING":
                continue
            if pod.get("id") != pod_file_id:
                sys.exit(
                    "refusing to start: an unrelated RUNNING RunPod pod "
                    f"{pod.get('id')!r} (name {name!r}) already exists. If it is "
                    "a leaked ingest pod, reap it first:\n"
                    "  python scripts/run_sharded_ingest.py --backend runpod --reap"
                )

    def _cost_gate(self) -> None:
        rate = _APPROX_RATE_AUD_PER_HOUR.get(self.cloud_type, 0.85)
        est = rate * _APPROX_SHARD0_HOURS
        print(
            "RunPod ingest cost gate\n"
            "-----------------------\n"
            "A leaked pod bills until you kill it. If this run is interrupted "
            "without a clean teardown, reap it with:\n"
            "  python scripts/run_sharded_ingest.py --backend runpod --reap\n"
            f"  card:     {self.gpu_type_id}\n"
            f"  cloud:    {self.cloud_type}\n"
            f"  rate:     ~AUD {rate:.2f}/hr (approx)\n"
            f"  estimate: ~AUD {est:.2f} for a ~{_APPROX_SHARD0_HOURS}h shard run",
            file=sys.stderr,
        )
        if self.assume_yes:
            return
        if not sys.stdin.isatty():
            sys.exit(
                "cost gate: stdin is not a TTY and --yes was not given; "
                "refusing to create a billable pod non-interactively."
            )
        reply = input("Type 'yes' to create the pod and proceed: ").strip()
        if reply != "yes":
            sys.exit("cost gate: not confirmed; aborting.")

    def _register_cleanup(self) -> None:
        self._register_atexit()
        self._register_signal_handlers()

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        self._atexit_registered = True
        atexit.register(self.teardown)

    def _register_signal_handlers(self) -> None:
        if self._signals_registered:
            return
        self._signals_registered = True
        self._signal(signal.SIGINT, self._on_signal)
        self._signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        # Set a flag and exit with the conventional code so the orchestrator's
        # `finally` runs teardown() exactly once.
        self._signalled = True
        sys.exit(130 if signum == signal.SIGINT else 143)

    def _ensure_gpu_idle(self) -> None:
        query = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
        busy = (getattr(self._ssh(query, check=False), "stdout", "") or "").strip()
        if not busy:
            return
        print(
            f"[runpod] GPU busy (pids: {busy!r}); killing any stale ingest job",
            file=sys.stderr,
        )
        self._ssh("pkill -f '[i]ngest_shard.sh' || true", check=False)
        self._sleep(10)
        busy = (getattr(self._ssh(query, check=False), "stdout", "") or "").strip()
        if busy:
            sys.exit(
                "RunPod GPU still has a running compute process after pkill "
                f"(pids: {busy!r}); a prior detached ingest job is still active. "
                "Aborting rather than double-loading the GPU."
            )

    # ------------------------------------------------------ Task 9: run_shard
    POLL_INTERVAL_S = 30
    CHECKPOINT_INTERVAL_POLLS = 10
    UNREACHABLE_BOUND_S = 1500  # 25 min wall-clock cap on SSH-unreachable retry

    def run_shard(
        self, index: int, shard_size: int, seed_cache: Path | None
    ) -> ShardResult:
        work = f"~/lex-au-search/run/shard_{index:03d}"
        ckpt = checkpoint_cache_path(self.shards_dir, index)

        # --- Step 0: GPU-idle guard + clean per-shard workdir -------------
        try:
            if self._gpu_pids():
                self._ssh("pkill -f '[i]ngest_shard.sh' || true", check=False)
                self._sleep(10)
                if self._gpu_pids():
                    return ShardResult(
                        index, False, None, None, "GPU busy with a prior job"
                    )
            self._ssh(f"rm -rf {work} && mkdir -p {work}")
        except _SSH_ERRORS:
            return self._give_up_unreachable(index, work, ckpt)

        # --- Step 1: seed the remote accumulator, verify the row count ----
        if seed_cache is not None and seed_cache.exists():
            try:
                mismatch = self._upload_seed(work, seed_cache)
            except _SSH_ERRORS:
                return self._give_up_unreachable(index, work, ckpt)
            except sqlite3.Error as e:
                # Corrupt / locked local accumulator - a local fault, not an
                # unreachable pod. Fail this shard cleanly.
                return ShardResult(
                    index, False, None, None, f"seed read failed: {e}"
                )
            if mismatch is not None:
                return ShardResult(index, False, None, None, mismatch)

        # --- Step 2: detached launch ------------------------------------
        # Every token here is load-bearing and asserted verbatim by the tests:
        # `-f` + DEVNULL on both streams so ssh returns immediately; the
        # `( ... ) &` subshell so the job outlives the connection; `setsid -w`
        # so it survives the controlling terminal going away; the
        # `job.exitcode.tmp` + `mv` so a poll never reads a half-written code.
        launch = (
            f"cd {work} && LEXAU_EMBED_BATCH_SIZE={self.batch_size} "
            f"( setsid -w bash ~/lex-au-search/scripts/ingest_shard.sh "
            f"{index} {shard_size} {work}/shard_cache_seed.db "
            f">{work}/job.log 2>&1 ; echo $? >{work}/job.exitcode.tmp ; "
            f"mv {work}/job.exitcode.tmp {work}/job.exitcode ) &"
        )
        try:
            self._run(
                [
                    "ssh", "-f",
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ControlMaster=auto",
                    "-o", "ControlPersist=60s",
                    "-i", os.path.expanduser(self.ssh_key),
                    f"root@{self._ip}",
                    "-p", str(self._port),
                    launch,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except _SSH_ERRORS:
            return self._give_up_unreachable(index, work, ckpt)

        # --- Steps 3-5: poll to a terminal state ----------------------
        poll_count = 0
        while True:
            try:
                code = self._read_exitcode(work)
            except _SSH_ERRORS:
                verdict = self._handle_ssh_failure(index, work, ckpt)
                if verdict is not None:
                    return verdict
                continue

            if code == "":
                # Absent or empty exitcode file => still running.
                self._log_trend(work)
                poll_count += 1
                if poll_count % self.CHECKPOINT_INTERVAL_POLLS == 0:
                    self._safe_snapshot(index, work, ckpt)
                self._sleep(self.POLL_INTERVAL_S)
                continue

            try:
                rc = int(code)
            except ValueError:
                # Half-written despite the tmp+mv guard - poll again.
                self._sleep(self.POLL_INTERVAL_S)
                continue

            if rc == 0:
                storage_zip, cache_zip = shard_paths(self.shards_dir, index)
                self.shards_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self._scp_back(f"{work}/shard_storage.zip", storage_zip)
                    self._scp_back(f"{work}/shard_cache.zip", cache_zip)
                except _SSH_ERRORS as e:
                    # The job succeeded remotely but we could not pull the
                    # outputs. One shard's failure must not crash the run;
                    # the accumulator already holds the last snapshot.
                    return ShardResult(
                        index,
                        False,
                        None,
                        None,
                        f"failed to download shard outputs: {e}",
                    )
                return ShardResult(index, True, storage_zip, cache_zip, "")

            # Non-zero terminal: salvage whatever the cache reached, then report.
            self._safe_snapshot(index, work, ckpt)
            try:
                diagnosis = self._ssh_check(f"tail -n 200 {work}/job.log")
            except _SSH_ERRORS:
                diagnosis = f"shard exited {rc}; job.log unavailable"
            return ShardResult(index, False, None, None, diagnosis)

    # ------------------------------------------------- Task 9: run_shard guts
    def _gpu_pids(self) -> str:
        res = self._ssh(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
            check=False,
        )
        return (getattr(res, "stdout", "") or "").strip()

    def _read_exitcode(self, work: str) -> str:
        res = self._ssh(f"cat {work}/job.exitcode 2>/dev/null", check=False)
        return (getattr(res, "stdout", "") or "").strip()

    def _log_trend(self, work: str) -> None:
        """One best-effort progress line per running poll. Never raises."""
        try:
            tail = self._ssh(f"tail -n1 {work}/job.log", check=False)
            gpu = self._ssh(
                "nvidia-smi --query-gpu=memory.used,memory.total "
                "--format=csv,noheader",
                check=False,
            )
            mem = self._ssh("free -m", check=False)
            print(
                f"[runpod] {(getattr(tail, 'stdout', '') or '').strip()} "
                f"| gpu MiB {(getattr(gpu, 'stdout', '') or '').strip()} "
                f"| {' '.join((getattr(mem, 'stdout', '') or '').split()[:8])}",
                file=sys.stderr,
            )
        except Exception:  # noqa: BLE001 - trend logging must never break the loop
            pass

    def _upload_seed(self, work: str, seed_cache: Path) -> str | None:
        """Flatten the local accumulator, scp it up, and prove the remote copy
        has the same row count. Returns a diagnosis string on mismatch, else
        None."""
        conn = sqlite3.connect(str(seed_cache))
        try:
            local_rows = conn.execute(
                "SELECT COUNT(*) FROM embed_cache"
            ).fetchone()[0]
        finally:
            conn.close()
        with tempfile.TemporaryDirectory() as td:
            flat = Path(td) / "seed_flat.db"
            src = sqlite3.connect(str(seed_cache))
            dst = sqlite3.connect(str(flat))
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            self._scp_to(flat, f"{work}/shard_cache_seed.db")
        remote = self._ssh_check(
            "python3 -c \"import sqlite3; print(sqlite3.connect('"
            f"{work}/shard_cache_seed.db')"
            ".execute('SELECT COUNT(*) FROM embed_cache').fetchone()[0])\""
        )
        if int(remote) != int(local_rows):
            return (
                f"seed upload row-count mismatch: local={local_rows} "
                f"remote={remote}"
            )
        return None

    def _snapshot(self, index: int, work: str, ckpt: Path) -> None:
        """Consistent copy of the live remote cache -> fold into the local
        accumulator at `ckpt`. Raises on any SSH/scp failure."""
        self._ssh(
            "python3 -c \"import sqlite3; sqlite3.connect('"
            f"{work}/shard_cache.db')"
            f".backup(sqlite3.connect('{work}/snap.db'))\"",
            check=False,
        )
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "snap.db"
            self._scp_back(f"{work}/snap.db", snap)
            self.shards_dir.mkdir(parents=True, exist_ok=True)
            rows = merge_cache_files([snap], ckpt)
        print(
            f"[runpod] shard {index}: snapshot merged ({rows} rows read)",
            file=sys.stderr,
        )

    def _safe_snapshot(self, index: int, work: str, ckpt: Path) -> None:
        try:
            self._snapshot(index, work, ckpt)
        except Exception as e:  # noqa: BLE001 - accumulator merge is best-effort
            print(
                f"[runpod] shard {index}: snapshot failed: {e}", file=sys.stderr
            )

    def _handle_ssh_failure(self, index: int, work: str, ckpt: Path):
        """One recovery cycle after an SSH error in the poll loop. Returns a
        failed ShardResult when the run should be abandoned, or None to keep
        polling. Never terminates the pod - that is teardown()'s job."""
        if self._deadline_s is None:
            self._deadline_s = time.monotonic() + self.UNREACHABLE_BOUND_S
        try:
            status = self._rd.get_status(self._pod_id)
        except Exception:  # noqa: BLE001 - API error == cannot confirm RUNNING
            status = None
        if status != "RUNNING" or time.monotonic() >= self._deadline_s:
            return self._give_up_unreachable(index, work, ckpt)
        self._sleep(self.POLL_INTERVAL_S)
        return None

    def _give_up_unreachable(
        self, index: int, work: str, ckpt: Path
    ) -> ShardResult:
        self._safe_snapshot(index, work, ckpt)
        return ShardResult(index, False, None, None, "pod/host unreachable")

    # --------------------------------------------------------- Task 9: scp
    def _scp_base(self) -> list[str]:
        # Same connection options as _ssh(), but scp spells the port `-P`.
        return [
            "scp",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=60s",
            "-i", os.path.expanduser(self.ssh_key),
            "-P", str(self._port),
        ]

    def _scp_to(self, local: Path, remote_path: str) -> None:
        res = self._run(
            self._scp_base() + [str(local), f"root@{self._ip}:{remote_path}"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if getattr(res, "returncode", 0) != 0:
            raise RuntimeError(
                f"scp upload to {remote_path} failed (rc="
                f"{getattr(res, 'returncode', '?')}): "
                f"{getattr(res, 'stderr', '') or ''}"
            )

    def _scp_back(self, remote_path: str, local: Path) -> None:
        res = self._run(
            self._scp_base() + [f"root@{self._ip}:{remote_path}", str(local)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if getattr(res, "returncode", 0) != 0:
            raise RuntimeError(
                f"scp download of {remote_path} failed (rc="
                f"{getattr(res, 'returncode', '?')}): "
                f"{getattr(res, 'stderr', '') or ''}"
            )

    # ------------------------------------------------------ Task 9: teardown
    def teardown(self) -> None:
        """Single idempotent sequence shared by the orchestrator's `finally`,
        the atexit hook, and the signal handler."""
        if self._teardown_ran:
            return
        self._teardown_ran = True

        if self._lock_fh is not None:
            try:
                self._lock_fh.close()
            except Exception:  # noqa: BLE001 - already closed / never opened
                pass
            self._lock_fh = None

        if not self.POD_FILE.exists():
            return
        pod_id = self.POD_FILE.read_text().strip()

        if self.keep_pod:
            print(
                f"[runpod] --keep-pod: leaving pod {pod_id} RUNNING (it keeps "
                "billing).\n"
                f"  ssh -i {self.ssh_key} -p {self._port} root@{self._ip}\n"
                "  reuse it next run:  python scripts/run_sharded_ingest.py "
                "--backend runpod --reuse-pod\n"
                f"  or terminate:       python scripts/runpod_driver.py "
                f"terminate {pod_id}"
            )
            return

        # Task 4's _request only catches HTTPError, so a URLError/timeout on the
        # DELETE would propagate raw and skip the MANUAL ACTION line below.
        try:
            ok = self._rd.terminate_pod(pod_id)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[runpod] terminate raised: {e}", file=sys.stderr)

        if ok:
            self.POD_FILE.unlink(missing_ok=True)
            return

        print(
            f"MANUAL ACTION: python scripts/runpod_driver.py terminate {pod_id}"
            f"\n  {DASHBOARD_PODS_URL}"
        )
        raise RuntimeError("could not confirm RunPod pod termination")
