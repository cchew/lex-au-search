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
import subprocess

_ENV = {**os.environ, "OAUTHLIB_RELAX_TOKEN_SCOPE": "1"}
_CUDA_CHECK = (
    "import onnxruntime\n"
    "providers = onnxruntime.get_available_providers()\n"
    "assert 'CUDAExecutionProvider' in providers, providers\n"
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
    result = _colab("exec", "-s", name, timeout=60, input_str=_CUDA_CHECK)
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
    result = _colab("exec", "-s", name, timeout=30, input_str=wrapper)
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
    result = _colab("exec", "-s", name, timeout=30, input_str=check)
    if result.returncode != 0:
        return "failed"
    return _parse_poll_output(result.stdout)


def tail_log(name: str, log_path: str = "job.log", n: int = 50) -> str:
    code = (
        "import subprocess\n"
        f"print(subprocess.run(['tail', '-n', '{n}', {log_path!r}], "
        "capture_output=True, text=True).stdout)\n"
    )
    result = _colab("exec", "-s", name, timeout=30, input_str=code)
    return result.stdout


def download(name: str, remote_path: str, local_path: str) -> bool:
    result = _colab("download", "-s", name, remote_path, local_path, timeout=600)
    return result.returncode == 0


def stop_session(name: str) -> None:
    _colab("stop", "-s", name, timeout=60)


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
