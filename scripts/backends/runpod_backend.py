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
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import runpod_driver as _runpod_driver
from lexausearch.cache import merge_cache_files  # noqa: F401  (Task 9 snapshot merge)

from backends.base import (
    IngestBackend,
    ShardResult,  # noqa: F401  (Task 9 return type)
    checkpoint_cache_path,  # noqa: F401  (Task 9 snapshot target)
    shard_paths,  # noqa: F401  (Task 9 zip destinations)
)

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

    # ------------------------------------------------------ Task 9 (stubs)
    def run_shard(
        self, index: int, shard_size: int, seed_cache: Path | None
    ) -> ShardResult:
        raise NotImplementedError("RunPodBackend.run_shard lands in Task 9")

    def teardown(self) -> None:
        raise NotImplementedError("RunPodBackend.teardown lands in Task 9")
