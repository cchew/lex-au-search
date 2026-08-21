#!/usr/bin/env python3
"""Wraps the `colab` CLI (googlecolab/google-colab-cli) so Claude can
create, verify, run, monitor, and tear down Colab T4 sessions without a
human at a browser tab. Mirrors gemma-challenge's deploy/provision.py:
verify before trust, teardown in a `finally`, JSON on stdout.

PREREQUISITE - a real packaging bug in google-colab-cli v0.6.0 (its
published PyPI wheel resolves `jupyter-kernel-client` to the wrong package,
see memory reference-google-colab-cli-kernelclient-bug) must be worked
around before this module's functions will work at all:

    uv tool install google-colab-cli --force --with \\
        "jupyter-kernel-client @ git+https://github.com/googlecolab/jupyter-kernel-client.git"

`colab new` also raises a fatal oauthlib exception on a granted/requested
OAuth scope mismatch unless OAUTHLIB_RELAX_TOKEN_SCOPE=1 is set - every
subprocess call below sets it unconditionally (confirmed-working fix, not
a hypothesis).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

_ENV = {**os.environ, "OAUTHLIB_RELAX_TOKEN_SCOPE": "1"}
_CUDA_CHECK = (
    # Checks GPU hardware attachment only, via torch - preinstalled on
    # stock Colab images (confirmed 2026-08-04). onnxruntime is NOT
    # preinstalled; the real onnxruntime-CUDA compatibility check happens
    # later in colab_ingest_shard.sh, after it installs onnxruntime-gpu.
    "import torch\n"
    "assert torch.cuda.is_available(), 'no CUDA device attached'\n"
    "print('CUDA_VERIFY_OK')\n"
)


def _colab(*args: str, timeout: int, input_str: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["colab", *args], env=_ENV, capture_output=True, text=True,
        timeout=timeout, input=input_str,
    )


def _parse_verify_output(returncode: int, stdout: str) -> bool:
    return returncode == 0 and "CUDA_VERIFY_OK" in stdout


def _parse_pid_output(stdout: str) -> int:
    match = re.search(r"PID (\d+)", stdout)
    if not match:
        raise ValueError(f"could not parse PID from colab exec output: {stdout!r}")
    return int(match.group(1))


def _parse_poll_output(stdout: str) -> str:
    if "DONE" in stdout:
        code = stdout.strip().split()[-1]
        return "done" if code == "0" else "failed"
    if "ALIVE" in stdout:
        return "running"
    return "failed"  # process gone but no exit-code file written - crashed hard


def create_session(name: str, gpu: str = "T4") -> dict:
    result = _colab("new", "-s", name, "--gpu", gpu, timeout=180)
    return {
        "session": name, "gpu": gpu, "ok": result.returncode == 0,
        "stdout": result.stdout.strip(), "stderr": result.stderr.strip(),
    }


def verify_session(name: str) -> bool:
    # 90s not 60s: a real free-tier T4 smoke test (2026-08-04) saw first
    # post-create exec take 21-60s+ depending on backend kernel-attach
    # latency - 60s intermittently timed out, so pad with real headroom.
    result = _colab("exec", "-s", name, timeout=90, input_str=_CUDA_CHECK)
    return _parse_verify_output(result.returncode, result.stdout)


def run_background(
    name: str, remote_command: str, log_path: str = "job.log", exit_code_path: str = "job.exitcode"
) -> int:
    """Launch remote_command detached on the session's VM; return its PID.
    Works around colab exec's 30s default timeout: the exec call itself
    only launches the process and returns immediately, a background thread
    inside the (long-lived) kernel process writes the exit code file once
    the launched process finishes, independent of this exec call returning."""
    wrapper = (
        "import subprocess, threading\n"
        f"p = subprocess.Popen(['bash', '-c', {remote_command!r}], "
        f"stdout=open({log_path!r}, 'w'), stderr=subprocess.STDOUT)\n"
        "def _wait():\n"
        "    p.wait()\n"
        f"    open({exit_code_path!r}, 'w').write(str(p.returncode))\n"
        "threading.Thread(target=_wait, daemon=False).start()\n"
        "print('PID', p.pid)\n"
    )
    result = _colab("exec", "-s", name, timeout=60, input_str=wrapper)
    if result.returncode != 0:
        raise RuntimeError(f"run_background failed to launch on {name}: {result.stderr}")
    return _parse_pid_output(result.stdout)


def poll_status(name: str, pid: int, exit_code_path: str = "job.exitcode") -> str:
    check = (
        "import os\n"
        f"alive = os.path.exists('/proc/{pid}')\n"
        f"done = os.path.exists({exit_code_path!r})\n"
        f"code = open({exit_code_path!r}).read().strip() if done else ''\n"
        "print('ALIVE' if alive else 'DEAD', 'DONE' if done else 'PENDING', code)\n"
    )
    try:
        result = _colab("exec", "-s", name, timeout=60, input_str=check)
    except subprocess.TimeoutExpired:
        # A slow status check isn't evidence the remote job died - treat as
        # still running and let the next poll retry, rather than discarding
        # real in-progress work over one transient exec-latency hiccup.
        return "running"
    if result.returncode != 0:
        # `colab exec` itself failed - our status-check probe never ran on
        # the VM at all, so nothing is known about the job process. This is
        # the VM/kernel being unreachable (Colab-side disconnect/preemption,
        # e.g. "Session 'NAME' not found"), not a job-level failure the probe
        # would otherwise have reported via _parse_poll_output. Confirmed
        # 2026-08-17: shard 0's failure was exactly this case, and the old
        # code here reported it as "failed" indistinguishable from a real
        # in-process crash.
        return "session_lost"
    return _parse_poll_output(result.stdout)


def _format_diagnosis(stdout: str) -> str:
    parts = stdout.split()
    if len(parts) >= 2 and parts[1] == "DONE":
        code = parts[2] if len(parts) > 2 else ""
        return f"exited with code {code} (job.exitcode written)"
    if not parts or parts[0] not in ("ALIVE", "DEAD"):
        # The probe itself never ran - no ALIVE/DEAD token came back, so the
        # remote VM is unreachable (typically `colab exec` printing "Session
        # 'NAME' not found" after Colab dropped the runtime and the local
        # registry desynced). Nothing whatsoever is known about the job
        # process here; reporting an OOM kill would be inventing evidence for
        # whichever theory is currently in favour. This is the signature the
        # real 2026-08-17 shard 0 failure left behind.
        return (
            f"session unreachable - no status returned from the VM "
            f"(Colab-side disconnect/preemption; job outcome unknown). "
            f"Probe output: {stdout.strip()!r}"
        )
    return (
        "crashed hard - process gone, job.exitcode never written "
        "(SIGKILL/OOM-killer signature)"
    )


def diagnose_failure(name: str, pid: int, exit_code_path: str = "job.exitcode") -> str:
    """Human-readable diagnosis for a failed shard: distinguishes a clean
    nonzero exit (real exit code available) from a hard crash (process gone,
    exit-code file never written - the signature of the whole remote process
    tree being SIGKILLed at once, e.g. by the OOM killer)."""
    check = (
        "import os\n"
        f"alive = os.path.exists('/proc/{pid}')\n"
        f"done = os.path.exists({exit_code_path!r})\n"
        f"code = open({exit_code_path!r}).read().strip() if done else ''\n"
        "print('ALIVE' if alive else 'DEAD', 'DONE' if done else 'PENDING', code)\n"
    )
    result = _colab("exec", "-s", name, timeout=60, input_str=check)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return (
            f"session unreachable - colab exec itself failed (rc={result.returncode}): "
            f"{detail!r}. Colab-side disconnect/preemption; job outcome unknown, "
            f"no evidence of OOM or any other in-process cause."
        )
    return _format_diagnosis(result.stdout)


def sample_resources(name: str) -> str:
    """One-line system RAM + GPU memory snapshot from the remote session -
    call on each poll cycle to see the memory trend leading up to a
    failure. A steady climb toward the ceiling before death points at our
    own process; a session that just vanishes with no such trend points at
    Colab-side preemption instead (see _release_orphaned_assignments)."""
    code = (
        "import subprocess\n"
        "mem = subprocess.run(['free', '-m'], capture_output=True, text=True).stdout.splitlines()\n"
        "mem_line = mem[1] if len(mem) > 1 else 'unavailable'\n"
        "gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', "
        "'--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip() or 'unavailable'\n"
        "print(f'RAM(MB): {mem_line} | GPU(MB used,total): {gpu}')\n"
    )
    try:
        # Padded like verify_session's 90s, not poll_status's 60s: this call
        # additionally shells out twice on the remote (free -m, nvidia-smi)
        # on top of the same exec-attach latency variance (confirmed
        # 2026-08-04: 21-60s+), so 30s alone is too tight and did cause a
        # real TimeoutExpired that aborted a healthy run (2026-08-21).
        result = _colab("exec", "-s", name, timeout=90, input_str=code)
    except subprocess.TimeoutExpired:
        # Best-effort diagnostic riding alongside the real status check -
        # must never abort the run it's monitoring just because one sample
        # was slow. The next poll cycle tries again.
        return "sample timed out"
    return result.stdout.strip() or "sample failed"


def tail_log(name: str, log_path: str = "job.log", n: int = 50) -> str:
    code = (
        "import subprocess\n"
        f"print(subprocess.run(['tail', '-n', '{n}', {log_path!r}], "
        "capture_output=True, text=True).stdout)\n"
    )
    result = _colab("exec", "-s", name, timeout=60, input_str=code)
    return result.stdout


def download(name: str, remote_path: str, local_path: str) -> bool:
    """colab download resolves remote_path literally, not relative to the
    exec kernel's cwd (confirmed /content on a stock Colab image, 2026-08-04)
    - a relative path like 'job.log' fails with "File or directory not
    found" even though it exists at /content/job.log. Anchor relative paths
    to /content so callers can keep passing paths relative to where
    run_background's remote commands actually execute."""
    if not remote_path.startswith("/"):
        remote_path = f"/content/{remote_path}"
    result = _colab("download", "-s", name, remote_path, local_path, timeout=600)
    return result.returncode == 0


_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB, conservative

# `colab upload`'s underlying transport (colab_cli.contents.ContentsClient.
# upload) reads the WHOLE local file, base64-encodes it (~33% larger), and
# sends it as ONE JSON PUT request - its payload schema has a "chunk" field
# but the client always hardcodes chunk=1, so nothing upstream ever actually
# splits a large file. A 174.6MB checkpoint cache upload was rejected
# outright (fast, clean nonzero exit - not a timeout), almost certainly a
# proxy/server body-size limit on that single ~232MB request (confirmed
# 2026-08-21). Since our checkpoint accumulator only grows across retries,
# this isn't a one-off - split client-side into small pieces uploaded
# separately via the same (working) single-file path, then reassemble
# remotely with a small Python script.


def upload(name: str, local_path: str, remote_path: str) -> bool:
    """Upload local_path to remote_path, anchoring a relative remote_path to
    /content (same quirk as download()). Transparently chunks files above
    _UPLOAD_CHUNK_SIZE - see that constant's comment for why."""
    if not remote_path.startswith("/"):
        remote_path = f"/content/{remote_path}"
    if os.path.getsize(local_path) <= _UPLOAD_CHUNK_SIZE:
        return _upload_whole(name, local_path, remote_path)
    return _upload_chunked(name, local_path, remote_path)


def _upload_whole(name: str, local_path: str, remote_path: str) -> bool:
    result = _colab("upload", "-s", name, local_path, remote_path, timeout=180)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        print(f"[colab_driver] upload {local_path} -> {remote_path} failed: {detail}", file=sys.stderr)
        return False
    return True


def _upload_chunked(name: str, local_path: str, remote_path: str) -> bool:
    # Every attempt gets its own token in its part filenames, rather than
    # a fixed .partNNNN name cleaned up before use - a first cut of this
    # relied on an explicit cleanup exec first, but that made correctness
    # depend on the cleanup call itself succeeding (it doesn't live outside
    # repo/, so a prior failed attempt's leftover parts survive the
    # `rm -rf repo` callers run before re-seeding). If cleanup were skipped
    # or timed out and an earlier attempt used a *different* chunk count,
    # reassembly would silently glob in those stale extra/missing parts and
    # produce a corrupt seed file instead of failing loudly. A unique token
    # makes that impossible: reassembly only ever globs this attempt's own
    # parts, so any stale files from earlier attempts are simply never
    # matched (and are harmless dead weight - the VM is torn down at
    # session end regardless).
    token = uuid.uuid4().hex[:8]

    index = 0
    with open(local_path, "rb") as f:
        while True:
            chunk = f.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(chunk)
                    tmp_path = tmp.name
                part_remote = f"{remote_path}.{token}.part{index:04d}"
                if not _upload_whole(name, tmp_path, part_remote):
                    return False
            finally:
                if tmp_path:
                    os.remove(tmp_path)
            index += 1

    reassemble = (
        "import glob, os\n"
        f"parts = sorted(glob.glob({remote_path!r} + '.{token}.part*'))\n"
        f"with open({remote_path!r}, 'wb') as out:\n"
        "    for p in parts:\n"
        "        with open(p, 'rb') as pf:\n"
        "            out.write(pf.read())\n"
        "        os.remove(p)\n"
        "print('REASSEMBLE_OK', len(parts))\n"
    )
    result = _colab("exec", "-s", name, timeout=120, input_str=reassemble)
    if "REASSEMBLE_OK" not in result.stdout:
        detail = (result.stderr or result.stdout).strip()
        print(f"[colab_driver] chunked upload reassembly failed for {remote_path}: {detail}", file=sys.stderr)
        return False
    return True


def checkpoint_cache(name: str, cache_path: str = "shard_cache.db") -> str:
    """Best-effort mid-run snapshot of the remote embedding cache, taken via
    SQLite's online backup API rather than a raw file copy - the ingest job
    on the other end keeps writing to cache_path throughout the run (see
    EmbedCache.put_batch's per-Act commit), and a raw copy could read a
    torn page mid-write. Snapshotting periodically means a session that
    gets killed mid-run (e.g. the confirmed ~60min free-tier cap, see
    2026-08-21 shard 0 history) only loses the interval since the last
    checkpoint, not the whole run - the recovered .db can reseed a retry so
    already-embedded chunks are cache hits instead of being re-embedded
    from scratch (Indexer._upsert_chunks_with_cache only calls the GPU
    embedder for cache misses).

    Returns one of "ok" (repo/shard_cache_checkpoint.zip ready to download),
    "no_db" (nothing embedded yet), "failed", or "timeout" - never raises,
    since a failed snapshot attempt must not disturb the job it's
    snapshotting alongside.
    """
    code = (
        "import sqlite3, os, zipfile\n"
        "os.chdir('/content/repo')\n"
        f"cache_path = {cache_path!r}\n"
        "if not os.path.exists(cache_path):\n"
        "    print('CHECKPOINT_NO_DB')\n"
        "else:\n"
        "    try:\n"
        "        src = sqlite3.connect(cache_path)\n"
        "        dst = sqlite3.connect('shard_cache_checkpoint.db')\n"
        "        src.backup(dst)\n"
        "        dst.close()\n"
        "        src.close()\n"
        "        with zipfile.ZipFile('shard_cache_checkpoint.zip', 'w', zipfile.ZIP_DEFLATED) as zf:\n"
        "            zf.write('shard_cache_checkpoint.db')\n"
        "        print('CHECKPOINT_OK')\n"
        "    except Exception as e:\n"
        "        print('CHECKPOINT_FAILED', repr(e))\n"
    )
    try:
        result = _colab("exec", "-s", name, timeout=90, input_str=code)
    except subprocess.TimeoutExpired:
        return "timeout"
    if "CHECKPOINT_OK" in result.stdout:
        return "ok"
    if "CHECKPOINT_NO_DB" in result.stdout:
        return "no_db"
    return "failed"


def stop_session(name: str) -> None:
    _colab("stop", "-s", name, timeout=60)
    _release_orphaned_assignments()


def _release_orphaned_assignments() -> None:
    """`colab stop -s NAME` looks the session up in a local registry
    (~/.config/colab-cli/sessions.json) and silently no-ops - printing
    "Session 'NAME' not found" - if that registry has desynced from the
    backend, which happens when Colab disconnects a runtime mid-job (a real
    risk on a multi-hour, mostly-unattended sharded run). When that
    happens the GPU assignment stays held server-side, and every later
    create_session() call fails with TooManyAssignmentsError (free tier
    allows exactly one concurrent GPU assignment) - one disconnected shard
    cascades into every remaining shard failing, not just the one that
    disconnected (confirmed 2026-08-17: shard 0 disconnected mid-run,
    shards 1-10 then all failed at create_session). Fall back to
    unassigning by endpoint directly against the backend, bypassing the
    local-name lookup that `colab stop` depends on. Best-effort: if the
    `colab` executable or its interpreter can't be resolved, this is a
    no-op rather than a hard failure.
    """
    colab_path = shutil.which("colab")
    if colab_path is None:
        return
    with open(colab_path) as f:
        shebang = f.readline().strip()
    if not shebang.startswith("#!"):
        return
    interpreter = shebang[2:]
    code = (
        "from colab_cli.common import state\n"
        "for a in state.client.list_assignments():\n"
        "    state.client.unassign(a.endpoint)\n"
    )
    try:
        subprocess.run(
            [interpreter, "-c", code], capture_output=True, text=True, timeout=60, env=_ENV
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create")
    p.add_argument("--name", required=True)
    p.add_argument("--gpu", default="T4")

    p = sub.add_parser("verify")
    p.add_argument("--name", required=True)

    p = sub.add_parser("run")
    p.add_argument("--name", required=True)
    p.add_argument("--cmd", required=True)
    p.add_argument("--log-path", default="job.log")
    p.add_argument("--exit-code-path", default="job.exitcode")

    p = sub.add_parser("status")
    p.add_argument("--name", required=True)
    p.add_argument("--pid", required=True, type=int)
    p.add_argument("--exit-code-path", default="job.exitcode")

    p = sub.add_parser("log")
    p.add_argument("--name", required=True)
    p.add_argument("--log-path", default="job.log")
    p.add_argument("-n", type=int, default=50)

    p = sub.add_parser("download")
    p.add_argument("--name", required=True)
    p.add_argument("remote_path")
    p.add_argument("local_path")

    p = sub.add_parser("stop")
    p.add_argument("--name", required=True)

    args = ap.parse_args()

    if args.command == "create":
        print(json.dumps(create_session(args.name, args.gpu)))
    elif args.command == "verify":
        print(json.dumps({"session": args.name, "cuda_ok": verify_session(args.name)}))
    elif args.command == "run":
        pid = run_background(args.name, args.cmd, args.log_path, args.exit_code_path)
        print(json.dumps({"session": args.name, "pid": pid}))
    elif args.command == "status":
        status = poll_status(args.name, args.pid, args.exit_code_path)
        print(json.dumps({"session": args.name, "pid": args.pid, "status": status}))
    elif args.command == "log":
        print(json.dumps({"session": args.name, "tail": tail_log(args.name, args.log_path, args.n)}))
    elif args.command == "download":
        ok = download(args.name, args.remote_path, args.local_path)
        print(json.dumps({"ok": ok}))
    elif args.command == "stop":
        stop_session(args.name)
        print(json.dumps({"session": args.name, "stopped": True}))


if __name__ == "__main__":
    main()
