import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import colab_driver as cd
from colab_driver import (
    _parse_verify_output, _parse_pid_output, _parse_poll_output, _format_diagnosis,
)


def test_parse_verify_output_true_on_success_marker():
    assert _parse_verify_output(0, "CUDA available, providers: [...]\nCUDA_VERIFY_OK\n") is True


def test_parse_verify_output_false_on_nonzero_exit():
    assert _parse_verify_output(1, "CUDA_VERIFY_OK") is False


def test_parse_verify_output_false_when_marker_missing():
    assert _parse_verify_output(0, "AssertionError: CUDA not available") is False


def test_parse_pid_output_extracts_pid():
    assert _parse_pid_output("PID 12345\n") == 12345


def test_parse_pid_output_extracts_from_noisy_output():
    assert _parse_pid_output("some banner text\nPID 987\nmore text\n") == 987


def test_parse_pid_output_raises_when_no_pid_present():
    import pytest
    with pytest.raises(ValueError):
        _parse_pid_output("no pid here")


def test_parse_poll_output_running_when_alive_and_sentinel_present():
    assert _parse_poll_output("ALIVE RUNNING\n") == "running"


def test_parse_poll_output_done_on_zero_exit_code():
    assert _parse_poll_output("DEAD 0\n") == "done"


def test_parse_poll_output_failed_on_nonzero_exit_code():
    assert _parse_poll_output("DEAD 1\n") == "failed"


def test_format_diagnosis_reports_exit_code_on_clean_exit():
    assert _format_diagnosis("DEAD 1\n") == "exited with code 1 (job.exitcode written)"


def test_format_diagnosis_reports_exit_code_zero():
    assert _format_diagnosis("DEAD 0\n") == "exited with code 0 (job.exitcode written)"


def test_format_diagnosis_reports_missing_sentinel_when_launch_never_ran():
    out = _format_diagnosis("DEAD MISSING\n")
    assert out == "job.exitcode file missing - the launch wrapper never ran"
    assert "OOM" not in out


def test_format_diagnosis_reports_session_unreachable_on_empty_output():
    # No ALIVE/DEAD token means the remote never answered, so nothing is
    # known about the job process - must not be reported as an OOM kill.
    assert "OOM" not in _format_diagnosis("")
    assert "session unreachable" in _format_diagnosis("")


def test_format_diagnosis_reports_session_unreachable_when_session_not_found():
    out = _format_diagnosis("[colab] Session 'lexau-shard-0' not found.")
    assert "OOM" not in out
    assert "session unreachable" in out


def test_poll_status_reports_session_lost_when_exec_itself_fails(monkeypatch):
    # `colab exec` returning nonzero means the status-check probe never ran on
    # the VM at all - the VM/kernel is unreachable (disconnect/preemption).
    # This must not be conflated with a nonzero exit from the *job* the probe
    # would have reported on.
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(
            returncode=1, stdout="", stderr="Session 'x' not found.",
        ),
    )
    assert cd.poll_status("x", 123) == "session_lost"


def test_poll_status_reports_running_when_probe_succeeds(monkeypatch):
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="ALIVE RUNNING\n", stderr="",
        ),
    )
    assert cd.poll_status("x", 123) == "running"


def test_poll_status_reports_running_when_process_gone_but_exitcode_not_written(monkeypatch):
    # The completion race that broke Task 6's Colab smoke on 2026-09-01: the
    # launched process is reaped (DEAD) a beat before the _wait thread swaps in
    # the real code, so the sentinel still reads RUNNING. Must not be a crash.
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="DEAD RUNNING\n", stderr="",
        ),
    )
    assert cd.poll_status("x", 123) == "running"


def test_diagnose_failure_reports_session_unreachable_when_exec_itself_fails(monkeypatch):
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(
            returncode=1, stdout="", stderr="Session 'x' not found.",
        ),
    )
    out = cd.diagnose_failure("x", 123)
    assert "session unreachable" in out
    assert "OOM-killer signature" not in out  # that's the *confirmed*-crash phrasing
    assert "not found" in out


def test_sample_resources_survives_exec_timeout(monkeypatch):
    # sample_resources is a best-effort per-poll-cycle diagnostic riding
    # alongside the real status check - a slow exec call here (the same
    # backend-latency variance documented on verify_session, real T4
    # smoke test 2026-08-04: 21-60s+) must never propagate and abort an
    # otherwise-healthy monitored run.
    import subprocess

    def fake_colab(*a, **k):
        raise subprocess.TimeoutExpired(cmd="colab exec", timeout=30)

    monkeypatch.setattr(cd, "_colab", fake_colab)
    assert cd.sample_resources("x") == "sample timed out"


def test_upload_anchors_relative_remote_path_to_content(monkeypatch, tmp_path):
    captured = {}

    def fake_colab(*args, **kwargs):
        captured["args"] = args
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cd, "_colab", fake_colab)
    local = tmp_path / "seed.db"
    local.write_bytes(b"x")
    assert cd.upload("x", str(local), "shard_cache_seed.db") is True
    assert captured["args"][-1] == "/content/shard_cache_seed.db"


def test_upload_leaves_absolute_remote_path_untouched(monkeypatch, tmp_path):
    captured = {}

    def fake_colab(*args, **kwargs):
        captured["args"] = args
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cd, "_colab", fake_colab)
    local = tmp_path / "seed.db"
    local.write_bytes(b"x")
    cd.upload("x", str(local), "/content/shard_cache_seed.db")
    assert captured["args"][-1] == "/content/shard_cache_seed.db"


def test_upload_returns_false_on_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    local = tmp_path / "seed.db"
    local.write_bytes(b"x")
    assert cd.upload("x", str(local), "shard_cache_seed.db") is False


def test_upload_prints_failure_detail_to_stderr(monkeypatch, capsys, tmp_path):
    # A silent False on failure gave no way to diagnose why a real seed
    # upload failed (2026-08-21: a 174MB checkpoint cache failed to upload
    # with no captured reason) - the actual colab-upload stderr must reach
    # the log so a future failure is diagnosable without re-instrumenting.
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(
            returncode=1, stdout="", stderr="Upload failed: payload too large",
        ),
    )
    local = tmp_path / "seed.db"
    local.write_bytes(b"x")
    cd.upload("x", str(local), "shard_cache_seed.db")
    assert "payload too large" in capsys.readouterr().err


def test_upload_sends_whole_file_when_under_chunk_size(monkeypatch, tmp_path):
    calls = []

    def fake_colab(*args, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cd, "_colab", fake_colab)
    monkeypatch.setattr(cd, "_UPLOAD_CHUNK_SIZE", 1024)
    local = tmp_path / "small.db"
    local.write_bytes(b"x" * 100)

    assert cd.upload("x", str(local), "seed.db") is True
    assert len(calls) == 1
    assert calls[0][0] == "upload"


def test_upload_chunks_file_over_chunk_size_and_reassembles(monkeypatch, tmp_path):
    calls = []

    def fake_colab(*args, **kwargs):
        calls.append(args)
        if args[0] == "exec":
            return types.SimpleNamespace(returncode=0, stdout="REASSEMBLE_OK 3", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cd, "_colab", fake_colab)
    monkeypatch.setattr(cd, "_UPLOAD_CHUNK_SIZE", 10)
    local = tmp_path / "big.db"
    local.write_bytes(b"y" * 25)  # 3 chunks: 10 + 10 + 5 bytes

    assert cd.upload("x", str(local), "seed.db") is True
    upload_calls = [c for c in calls if c[0] == "upload"]
    exec_calls = [c for c in calls if c[0] == "exec"]
    assert len(upload_calls) == 3
    # Each chunk lands at a distinct, zero-padded, order-preserving remote
    # name, all sharing one per-attempt token (see test below for why).
    remote_targets = [c[-1] for c in upload_calls]
    assert remote_targets == sorted(remote_targets)
    assert remote_targets[0].startswith("/content/seed.db.")
    assert ".part0000" in remote_targets[0]
    # Just the one reassembly exec - no separate cleanup call.
    assert len(exec_calls) == 1


def test_upload_chunked_parts_carry_a_unique_per_attempt_token(monkeypatch, tmp_path):
    # A first cut cleaned up stale .partNNNN files from a previous failed
    # attempt via an explicit exec call before uploading. That made
    # correctness depend on the cleanup succeeding: if it were skipped
    # (e.g. a slow exec) and an earlier attempt used a *different* chunk
    # count, reassembly's glob would silently pull in stale leftover parts
    # and produce a corrupt seed file. A unique token per attempt makes
    # that structurally impossible - confirm two separate upload() calls
    # for the same remote_path never share a token.
    calls = []

    def fake_colab(*args, **kwargs):
        calls.append(args)
        if args[0] == "exec":
            return types.SimpleNamespace(returncode=0, stdout="REASSEMBLE_OK 3", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cd, "_colab", fake_colab)
    monkeypatch.setattr(cd, "_UPLOAD_CHUNK_SIZE", 10)
    local = tmp_path / "big.db"
    local.write_bytes(b"y" * 25)

    cd.upload("x", str(local), "seed.db")
    first_targets = [c[-1] for c in calls if c[0] == "upload"]
    calls.clear()
    cd.upload("x", str(local), "seed.db")
    second_targets = [c[-1] for c in calls if c[0] == "upload"]

    first_token = first_targets[0].split(".part")[0]
    second_token = second_targets[0].split(".part")[0]
    assert first_token != second_token


def test_upload_chunked_aborts_without_reassembling_on_chunk_failure(monkeypatch, tmp_path):
    calls = []

    def fake_colab(*args, **kwargs):
        calls.append(args)
        if args[0] == "upload" and len(
            [c for c in calls if c[0] == "upload"]
        ) == 2:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="chunk 2 rejected")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cd, "_colab", fake_colab)
    monkeypatch.setattr(cd, "_UPLOAD_CHUNK_SIZE", 10)
    local = tmp_path / "big.db"
    local.write_bytes(b"z" * 25)

    assert cd.upload("x", str(local), "seed.db") is False
    # Reassembly must never fire from an incomplete part set, which would
    # silently produce a corrupt file.
    exec_calls = [c for c in calls if c[0] == "exec"]
    assert len(exec_calls) == 0


def test_checkpoint_cache_ok_when_snapshot_succeeds(monkeypatch):
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="CHECKPOINT_OK\n", stderr=""),
    )
    assert cd.checkpoint_cache("x") == "ok"


def test_checkpoint_cache_no_db_when_nothing_embedded_yet(monkeypatch):
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="CHECKPOINT_NO_DB\n", stderr=""),
    )
    assert cd.checkpoint_cache("x") == "no_db"


def test_checkpoint_cache_failed_on_snapshot_error(monkeypatch):
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="CHECKPOINT_FAILED database is locked\n", stderr="",
        ),
    )
    assert cd.checkpoint_cache("x") == "failed"


def test_checkpoint_cache_timeout_does_not_raise(monkeypatch):
    import subprocess

    def fake_colab(*a, **k):
        raise subprocess.TimeoutExpired(cmd="colab exec", timeout=90)

    monkeypatch.setattr(cd, "_colab", fake_colab)
    assert cd.checkpoint_cache("x") == "timeout"


def test_checkpoint_cache_failed_when_exec_itself_fails(monkeypatch):
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="Session 'x' not found."),
    )
    assert cd.checkpoint_cache("x") == "failed"


def test_diagnose_failure_reports_missing_sentinel_when_vm_reachable(monkeypatch):
    # Exec succeeds (VM/kernel alive) but job.exitcode was never created - the
    # launch wrapper itself never ran. Not a session loss, and not an OOM: say
    # exactly what is known.
    monkeypatch.setattr(
        cd, "_colab",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="DEAD MISSING\n", stderr="",
        ),
    )
    out = cd.diagnose_failure("x", 123)
    assert out == "job.exitcode file missing - the launch wrapper never ran"
    assert "OOM" not in out


# --- exitcode-race fix (RED) ---------------------------------------------------
# The remote _wait() thread only writes job.exitcode after the launched process
# is reaped, so a poll can land in the window where the process is gone (DEAD)
# and the sentinel still reads RUNNING. Task 6's 2026-09-01 Colab smoke hit
# exactly this: the shard completed (both zips written) but the poll that caught
# the gap classified it "failed" with a fabricated OOM diagnosis, so the
# download never ran. New probe format is two tokens: "<ALIVE|DEAD> <content>"
# where content is RUNNING, MISSING, or the integer exit code.

def test_parse_poll_output_running_when_process_gone_but_exitcode_not_written():
    assert _parse_poll_output("DEAD RUNNING\n") == "running"


def test_parse_poll_output_failed_on_real_exit_code_even_while_process_alive():
    # A recorded integer code is authoritative regardless of the /proc check.
    assert _parse_poll_output("ALIVE 137\n") == "failed"


def test_parse_poll_output_failed_when_exitcode_file_missing():
    assert _parse_poll_output("DEAD MISSING\n") == "failed"


def test_build_run_wrapper_settles_exitcode_atomically(tmp_path):
    import time

    ec = tmp_path / "job.exitcode"
    log = tmp_path / "job.log"
    wrapper = cd._build_run_wrapper("exit 7", str(log), str(ec))
    exec(compile(wrapper, "<wrapper>", "exec"), {})

    # never observable as absent or empty: it is the sentinel or the real code
    assert ec.read_text() in ("RUNNING", "7")
    for _ in range(200):
        if ec.read_text() == "7":
            break
        time.sleep(0.05)
    assert ec.read_text() == "7"


def test_exec_sync_forwards_command_and_returns_tuple(monkeypatch):
    import colab_driver as cd
    seen = {}
    class _R:
        returncode = 0
        stdout = "ok"
        stderr = ""
    def _fake_colab(*args, timeout, input_str=None):
        seen["args"] = args
        seen["input"] = input_str
        seen["timeout"] = timeout
        return _R()
    monkeypatch.setattr(cd, "_colab", _fake_colab)
    rc, out, err = cd.exec_sync("sess-1", "echo hi", timeout=42)
    assert (rc, out, err) == (0, "ok", "")
    assert seen["args"] == ("exec", "-s", "sess-1")
    assert seen["input"] == "echo hi"
    assert seen["timeout"] == 42
