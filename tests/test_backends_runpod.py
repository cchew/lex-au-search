import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from backends.base import checkpoint_cache_path
from backends.runpod_backend import RunPodBackend


class _FakeRD:
    def __init__(self, pods=None, create=None):
        self.pods = pods or []
        self._create = create or {"id": "pod_new", "desiredStatus": "RUNNING"}
        self.terminated = []
        self.CreatePodConfig = __import__("runpod_driver").CreatePodConfig
        self.pod_name = lambda *a, **k: "lexau-ingest-TEST"
    def list_pods(self): return self.pods
    def create_pod(self, cfg, dry_run=False): return dict(self._create)
    def wait_ready(self, pid, **k): return ("1.2.3.4", 22001)
    def get_status(self, pid): return "TERMINATED" if pid in self.terminated else "RUNNING"
    def terminate_pod(self, pid, **k): self.terminated.append(pid); return True


def _be(tmp_path, rd, **kw):
    runs = []
    def fake_run(cmd, **kwargs):
        runs.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    # NOTE (Task 8 deviation): the plan's verbatim helper passed `assume_yes=True`
    # AND `**kw`; a test that overrides `assume_yes` then hits CPython's
    # "multiple values for keyword argument" TypeError. Merge so `**kw` wins.
    kwargs = dict(ssh_key="~/.ssh/k", assume_yes=True, _rd=rd, _run=fake_run)
    kwargs.update(kw)
    be = RunPodBackend(tmp_path, **kwargs)
    be._ssh_runs = runs
    # No-op the process-global signal registration seam so tests that reach
    # prepare() step 4 do not leak SIGINT/SIGTERM handlers.
    be._signal = lambda *a, **k: None
    return be


def _make_runpod_backend(tmp_path, **kw):
    return _be(tmp_path, _FakeRD(), **kw)


def test_prepare_refuses_on_unrelated_running_lexau_pod(tmp_path):
    rd = _FakeRD(pods=[{"id": "other", "name": "lexau-ingest-someoneelse", "desiredStatus": "RUNNING"}])
    be = _be(tmp_path, rd)
    with pytest.raises(SystemExit):
        be.prepare()


def test_prepare_writes_pod_file_and_registers_cleanup_before_any_raise(tmp_path, monkeypatch):
    rd = _FakeRD()
    registered = []
    monkeypatch.setattr("backends.runpod_backend.atexit.register", lambda fn: registered.append(fn))
    be = _be(tmp_path, rd)
    # make the post-create ssh auth step raise; cleanup must already be registered
    def boom(cmd, **k):
        if "true" in cmd[-1]:
            raise RuntimeError("ssh auth failed")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = boom
    with pytest.raises(RuntimeError):
        be.prepare()
    assert (tmp_path / ".runpod_pod").read_text().strip() == "pod_new"
    assert be.teardown in registered
    # The pod is ours, so the registered teardown must actually terminate it.
    assert be._owns_pod is True
    be.teardown()
    assert rd.terminated == ["pod_new"]


def test_prepare_refusal_does_not_terminate_a_pod_we_did_not_create(tmp_path, monkeypatch):
    # A cost-gate / preflight refusal unwinds into run_all's `finally` ->
    # teardown(). A pre-existing .runpod_pod (e.g. a --keep-pod instance) must
    # survive: this process never owned that pod.
    rd = _FakeRD()
    be = _be(tmp_path, rd, assume_yes=False)
    (tmp_path / ".runpod_pod").write_text("pod_kept\n")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        be.prepare()
    be.teardown()
    assert be._owns_pod is False
    assert rd.terminated == []
    assert (tmp_path / ".runpod_pod").read_text().strip() == "pod_kept"


def test_teardown_uses_in_memory_pod_id_when_pod_file_missing(tmp_path):
    # A failed POD_FILE.write_text in prepare() must not make teardown a silent
    # no-op on a live, billing pod.
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    (tmp_path / ".runpod_pod").unlink()
    be.teardown()
    assert rd.terminated == ["pod_new"]


def test_prepare_raises_when_create_pod_returns_no_id(tmp_path, monkeypatch):
    rd = _FakeRD()
    rd.create_pod = lambda *a, **k: {"desiredStatus": "RUNNING"}
    monkeypatch.setattr("backends.runpod_backend.atexit.register", lambda fn: None)
    be = _be(tmp_path, rd)
    with pytest.raises(RuntimeError, match="no id"):
        be.prepare()
    assert be._owns_pod is False
    be.teardown()
    assert rd.terminated == []


def test_cost_gate_aborts_when_not_yes_and_not_tty(tmp_path, monkeypatch):
    rd = _FakeRD()
    be = _be(tmp_path, rd, assume_yes=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        be.prepare()


def test_flock_contention_exits(tmp_path):
    rd = _FakeRD()
    be1 = _be(tmp_path, rd)
    be1._acquire_lock()          # helper that does the flock
    be2 = _be(tmp_path, rd)
    with pytest.raises(SystemExit):
        be2._acquire_lock()


def test_prepare_refuses_matching_running_pod_without_reuse_flag(tmp_path):
    # POD_FILE names a pod the API still reports RUNNING (prior unclean exit) and
    # --reuse-pod was NOT passed -> refuse rather than create a second pod and
    # overwrite the id (which would leak the first, billing).
    rd = _FakeRD()
    (tmp_path / ".runpod_pod").write_text("pod_leaked\n")
    be = _be(tmp_path, rd)  # reuse_pod defaults False
    with pytest.raises(SystemExit):
        be.prepare()
    assert rd.terminated == []  # preflight refuses; never touches the pod


def test_prepare_reuses_matching_running_pod_with_reuse_flag(tmp_path, monkeypatch):
    rd = _FakeRD()
    rd.create_pod = lambda *a, **k: pytest.fail("create_pod must not run on the --reuse-pod path")
    monkeypatch.setattr("backends.runpod_backend.atexit.register", lambda fn: None)
    (tmp_path / ".runpod_pod").write_text("pod_leaked\n")
    be = _be(tmp_path, rd, reuse_pod=True)
    be.prepare()
    assert be._pod_id == "pod_leaked"
    assert (tmp_path / ".runpod_pod").read_text().strip() == "pod_leaked"


def test_gpu_type_ids_reach_create_pod_config(tmp_path, monkeypatch):
    rd = _FakeRD()
    seen = {}
    inner = rd.create_pod

    def spy(cfg, dry_run=False):
        seen["cfg"] = cfg
        return inner(cfg, dry_run=dry_run)

    rd.create_pod = spy
    monkeypatch.setattr("backends.runpod_backend.atexit.register", lambda fn: None)
    be = _be(tmp_path, rd, gpu_type_ids=("NVIDIA RTX A5000", "NVIDIA RTX A6000"))
    be.prepare()
    assert seen["cfg"].gpu_type_ids == ("NVIDIA RTX A5000", "NVIDIA RTX A6000")


def test_cost_gate_lists_every_candidate_gpu(tmp_path, capsys):
    be = _be(tmp_path, _FakeRD(),
             gpu_type_ids=("NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"))
    be._cost_gate()
    err = capsys.readouterr().err
    assert "NVIDIA RTX A5000" in err and "NVIDIA GeForce RTX 3090" in err
    assert "first available" in err


# --------------------------------------------------------------- Task 9 tests


def _started(stdout=""):
    """Default fake-run reply, with the post-launch `test -f job.log` probe
    answered STARTED so run_shard proceeds into the poll loop."""
    return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def _is_start_probe(s) -> bool:
    return "echo STARTED" in str(s)


def _prepared(tmp_path, rd, **kw):
    be = _be(tmp_path, rd, **kw)
    be._pod_id = "pod_new"; be._ip = "1.2.3.4"; be._port = 22001
    be._lock_fh = open(tmp_path / ".lk", "w")
    # _prepared stands in for a backend that already created its own pod, so
    # teardown() is entitled to terminate it (see _owns_pod).
    be._owns_pod = True
    # Task 9 deviation: the brief's verbatim _prepared omits this, which leaves
    # test_detached_launch_command_shape (no _sleep override of its own) doing a
    # real 30s time.sleep between its two scripted polls. A no-op sleep keeps
    # that test instant; the tests that need it also set it themselves.
    be._sleep = lambda *a, **k: None
    (tmp_path / ".runpod_pod").write_text("pod_new")
    return be


def test_detached_launch_command_shape(tmp_path):
    rd = _FakeRD()
    be = _prepared(tmp_path, rd, batch_size=16)
    seq = {"n": 0}
    def fake_run(cmd, **kw):
        s = cmd[-1]
        if _is_start_probe(s):
            return _started("STARTED")
        if "job.exitcode" in s and "cat" in s:
            seq["n"] += 1
            return types.SimpleNamespace(returncode=0, stdout=("" if seq["n"] < 2 else "0"), stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    launched = {}
    orig = be._run
    def capture(cmd, **kw):
        if any("setsid -w" in str(x) for x in cmd):
            launched["cmd"] = cmd
        return orig(cmd, **kw)
    be._run = capture
    be.run_shard(0, 300, seed_cache=None)
    j = " ".join(launched["cmd"])
    assert "ssh -f" in j or ("-f" in launched["cmd"])
    assert "setsid -w" in j
    assert "-o ControlPath=~/.ssh/cm-lexau-%r@%h:%p" in j
    assert "( LEXAU_EMBED_BATCH_SIZE=16 setsid -w bash ~/lex-au-search/scripts/ingest_shard.sh 0 300 " in j
    assert "job.exitcode.tmp" in j and "mv " in j and "job.exitcode ) &" in j


def test_detached_launch_string_is_valid_bash(tmp_path):
    # `cd X && VAR=v ( ... ) &` is a bash parse error; the assignment has to sit
    # inside the subshell. Guard the real string with `bash -n`.
    import subprocess
    rd = _FakeRD()
    be = _prepared(tmp_path, rd, batch_size=16)
    captured = {}
    def cap(cmd, **kw):
        s = cmd[-1] if isinstance(cmd[-1], str) else ""
        if "setsid -w" in s:
            captured["launch"] = s
        if _is_start_probe(s):
            return _started("STARTED")
        if "nvidia-smi" in s:
            return _started()  # GPU idle
        # "0" answers `cat job.exitcode`, so the poll loop ends immediately.
        return types.SimpleNamespace(returncode=0, stdout="0", stderr="")
    be._run = cap
    be._sleep = lambda *a, **k: None
    be.run_shard(0, 300, seed_cache=None)
    r = subprocess.run(
        ["bash", "-n", "-c", captured["launch"]], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_launch_without_job_log_fails_fast(tmp_path):
    # `ssh -f` returns 0 the moment it backgrounds itself, so the only proof the
    # detached job started is job.log existing shortly after.
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    def fake_run(cmd, **kw):
        s = cmd[-1]
        if _is_start_probe(s):
            return _started("NOSTART")
        if "cat" in str(s) and "job.exitcode" in str(s):
            pytest.fail("must not enter the poll loop when the job never started")
        return _started()
    be._run = fake_run
    be._sleep = lambda *a, **k: None
    res = be.run_shard(0, 300, seed_cache=None)
    assert res.ok is False
    assert "did not start" in res.diagnosis
    assert rd.terminated == []


def test_run_loop_gives_up_at_wall_clock_cap_without_terminating(tmp_path):
    # SSH keeps working but job.exitcode never appears (wedged ingest). The
    # RUN_MAX_S bound must end the wait; the pod is teardown()'s problem.
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    def fake_run(cmd, **kw):
        s = cmd[-1]
        if _is_start_probe(s):
            return _started("STARTED")
        return _started()  # exitcode always empty -> "still running" forever
    be._run = fake_run
    be._sleep = lambda *a, **k: None
    be._run_deadline = 0.0  # monotonic() >= 0.0 -> cap hit on the first poll
    res = be.run_shard(0, 300, seed_cache=None)
    assert res.ok is False
    assert "wall-clock cap" in res.diagnosis
    assert "18h" in res.diagnosis
    assert rd.terminated == []
    assert (tmp_path / ".runpod_pod").exists()


def test_deadline_resets_between_shards(tmp_path):
    # _deadline_s is per shard: shard 0's SSH blip must not pin the unreachable
    # budget for shard 1.
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    be._deadline_s = 12345.0  # leftover from a previous shard
    def fake_run(cmd, **kw):
        s = cmd[-1]
        if _is_start_probe(s):
            return _started("STARTED")
        if "cat" in str(s) and "job.exitcode" in str(s):
            return _started("0")
        return _started()
    be._run = fake_run
    be._sleep = lambda *a, **k: None
    be.run_shard(1, 300, seed_cache=None)
    assert be._deadline_s is None  # reset on entry, never re-armed this shard


def test_poll_treats_absent_or_empty_exitcode_as_running(tmp_path):
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    polls = ["", "", "0"]
    def fake_run(cmd, **kw):
        s = cmd[-1]
        if _is_start_probe(s):
            return _started("STARTED")
        if "cat" in s and "job.exitcode" in s:
            return types.SimpleNamespace(returncode=0, stdout=polls.pop(0), stderr="")
        if "shard_storage.zip" in s or "shard_cache.zip" in s:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    be._sleep = lambda *_: None
    res = be.run_shard(0, 300, seed_cache=None)
    assert res.ok is True
    assert polls == []  # all three polls consumed; empties were "running", not "failed"


def test_nonzero_exitcode_pulls_final_snapshot_and_sets_diagnosis(tmp_path, monkeypatch):
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    merged = []
    monkeypatch.setattr("backends.runpod_backend.merge_cache_files", lambda srcs, target: merged.append(target))
    def fake_run(cmd, **kw):
        s = cmd[-1]
        if _is_start_probe(s):
            return _started("STARTED")
        if "cat" in s and "job.exitcode" in s:
            return types.SimpleNamespace(returncode=0, stdout="1", stderr="")
        if "tail -n 200" in s:
            return types.SimpleNamespace(returncode=0, stdout="...boom...", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    be._sleep = lambda *_: None
    res = be.run_shard(0, 300, seed_cache=None)
    assert res.ok is False
    assert "boom" in res.diagnosis
    assert merged and merged[-1] == checkpoint_cache_path(tmp_path, 0)


def test_ssh_unreachable_checks_status_and_does_not_terminate_within_bound(tmp_path):
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    def fake_run(cmd, **kw):
        raise RuntimeError("network down")
    be._run = fake_run
    be._sleep = lambda *_: None
    # run_shard resets _deadline_s per shard, so pin it through the override seam.
    be._deadline_override = 0.0  # force the 25-min bound immediately for the test
    res = be.run_shard(0, 300, seed_cache=None)
    assert res.ok is False
    assert "unreachable" in res.diagnosis.lower()
    assert rd.terminated == []          # run_shard never terminates
    assert (tmp_path / ".runpod_pod").exists()


def test_scp_back_failure_on_success_path_returns_failed_result(tmp_path):
    # Job finished 0 remotely, but pulling shard_storage.zip fails -> the run
    # must degrade to a failed ShardResult, never raise out of run_shard.
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    def fake_run(cmd, **kw):
        s = cmd[-1]
        if _is_start_probe(s):
            return _started("STARTED")
        if "cat" in s and "job.exitcode" in s:
            return types.SimpleNamespace(returncode=0, stdout="0", stderr="")
        if any("shard_storage.zip" in str(x) for x in cmd):
            return types.SimpleNamespace(returncode=1, stdout="", stderr="scp: connection closed")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    be._sleep = lambda *_: None
    res = be.run_shard(0, 300, seed_cache=None)
    assert res.ok is False
    assert "download" in res.diagnosis.lower()
    assert rd.terminated == []


def test_seed_read_failure_returns_failed_result(tmp_path):
    # A corrupt / non-sqlite local accumulator is a local fault, not an
    # unreachable pod: fail this shard cleanly with a "seed read failed" message.
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    seed = tmp_path / "seed.db"
    seed.write_text("this is not a sqlite database")
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    be._sleep = lambda *_: None
    res = be.run_shard(0, 300, seed_cache=seed)
    assert res.ok is False
    assert "seed read failed" in res.diagnosis.lower()
    assert rd.terminated == []


def test_poll_loop_ssh_drop_gives_up_at_deadline_without_terminating(tmp_path):
    # Launch succeeds, then every poll-loop SSH call raises. get_status stays
    # "RUNNING", so only the wall-clock bound (_deadline_s, pinned to 0.0) ends
    # the wait. run_shard must not terminate the pod or clear .runpod_pod.
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    state = {"launched": False}
    def fake_run(cmd, **kw):
        if any("setsid -w" in str(x) for x in cmd):
            state["launched"] = True
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if state["launched"]:
            raise RuntimeError("connection dropped mid-run")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    be._sleep = lambda *_: None
    be._deadline_override = 0.0  # monotonic() >= 0.0 always true -> give up first cycle
    res = be.run_shard(0, 300, seed_cache=None)
    assert res.ok is False
    assert "unreachable" in res.diagnosis.lower()
    assert rd.terminated == []
    assert (tmp_path / ".runpod_pod").exists()


def test_teardown_is_idempotent_and_terminates_once(tmp_path):
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    be.teardown()
    be.teardown()
    assert rd.terminated == ["pod_new"]
    assert not (tmp_path / ".runpod_pod").exists()


def test_teardown_raises_when_termination_unconfirmed(tmp_path):
    rd = _FakeRD()
    rd.terminate_pod = lambda pid, **k: False
    be = _prepared(tmp_path, rd)
    with pytest.raises(RuntimeError):
        be.teardown()
    assert (tmp_path / ".runpod_pod").exists()   # retained for the next --reap


def test_keep_pod_leaves_pod_and_prints_reuse_hint(tmp_path, capsys):
    rd = _FakeRD()
    be = _prepared(tmp_path, rd, keep_pod=True)
    be.teardown()
    assert rd.terminated == []
    assert "--reuse-pod" in capsys.readouterr().out
    assert (tmp_path / ".runpod_pod").exists()


def test_snapshot_backup_cds_into_workdir_not_tilde_literal(tmp_path):
    # `work` is "~/lex-au-search/run/shard_000"; Python's sqlite3.connect() does
    # not expand `~`. The backup command must `cd` into the workdir (shell
    # expands it) and use relative db names, else snap.db lands at a literal
    # ./~/... path and the scp-back fails "No such file".
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    seen = []
    def fake_run(cmd, **kw):
        s = cmd[-1]
        seen.append(s)
        if _is_start_probe(s):
            return _started("STARTED")
        if "cat" in s and "job.exitcode" in s:
            return types.SimpleNamespace(returncode=0, stdout="1", stderr="")
        if "tail -n 200" in s:
            return types.SimpleNamespace(returncode=0, stdout="boom", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    be._sleep = lambda *_: None
    be.run_shard(0, 300, seed_cache=None)
    backup = next(s for s in seen if "sqlite3" in s and "backup" in s)
    assert backup.startswith("cd ~/lex-au-search/run/shard_000 &&")
    assert "connect('shard_cache.db')" in backup
    assert "connect('snap.db')" in backup
    # no tilde-prefixed path inside the python -c literal
    assert "'~/" not in backup


def test_scp_base_uses_a_dedicated_connection_not_the_ssh_controlmaster(tmp_path):
    # Bulk transfers must not multiplex over the _ssh ControlMaster socket
    # (ControlPersist=60s, meant for quick commands): a large copy over it can
    # drop mid-stream and scp still exits 0. Each transfer gets its own
    # connection + keepalives.
    be = _be(tmp_path, _FakeRD())
    base = " ".join(be._scp_base())
    assert "ControlMaster=no" in base
    assert "ControlPath=none" in base
    assert "cm-lexau" not in base
    assert "ServerAliveInterval=15" in base


def _scp_probe_run(remote_size_by_attempt, *, scp_rc=0):
    """fake _run that answers `stat -c %s` with the next size in the list and
    every scp invocation with scp_rc."""
    sizes = list(remote_size_by_attempt)
    def fake_run(cmd, **kw):
        if cmd and cmd[0] == "scp":
            return types.SimpleNamespace(returncode=scp_rc, stdout="", stderr="boom")
        if cmd and cmd[0] == "ssh" and str(cmd[-1]).startswith("stat -c %s"):
            nxt = sizes.pop(0) if sizes else sizes_last[0]
            return types.SimpleNamespace(returncode=0, stdout=f"{nxt}\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    sizes_last = [remote_size_by_attempt[-1]] if remote_size_by_attempt else [-1]
    return fake_run


def test_scp_to_retries_and_raises_on_persistent_truncation(tmp_path):
    be = _prepared(tmp_path, _FakeRD())
    be._sleep = lambda *_: None
    local = tmp_path / "seed_flat.db"
    local.write_bytes(b"x" * 800)                       # want == 800
    be._run = _scp_probe_run([48, 48, 48])             # remote always short
    with pytest.raises(RuntimeError) as e:
        be._scp_to(local, "~/w/shard_cache_seed.db")
    msg = str(e.value)
    assert "after 3 attempts" in msg
    assert "local=800" in msg and "remote=48" in msg


def test_scp_to_returns_once_remote_size_matches(tmp_path):
    be = _prepared(tmp_path, _FakeRD())
    be._sleep = lambda *_: None
    local = tmp_path / "seed_flat.db"
    local.write_bytes(b"x" * 800)
    calls = {"scp": 0}
    def fake_run(cmd, **kw):
        if cmd and cmd[0] == "scp":
            calls["scp"] += 1
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd and cmd[0] == "ssh" and str(cmd[-1]).startswith("stat -c %s"):
            size = 48 if calls["scp"] < 2 else 800    # short, then full
            return types.SimpleNamespace(returncode=0, stdout=f"{size}\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    be._scp_to(local, "~/w/shard_cache_seed.db")        # must not raise
    assert calls["scp"] == 2


def test_seed_upload_ssh_failure_is_labelled_seed_upload_not_unreachable(tmp_path, monkeypatch):
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    be._sleep = lambda *_: None
    seed = tmp_path / "seed.db"
    import sqlite3 as _sq
    c = _sq.connect(seed); c.execute("create table embed_cache(k text)"); c.commit(); c.close()
    monkeypatch.setattr(be, "_upload_seed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scp upload to X failed after 3 attempts (rc=0 local=800 remote=48)")))
    res = be.run_shard(0, 300, seed_cache=seed)
    assert res.ok is False
    assert "seed upload failed" in res.diagnosis.lower()
    assert "unreachable" not in res.diagnosis.lower()
    assert rd.terminated == []


def test_rsync_up_command_shape(tmp_path):
    be = _prepared(tmp_path, _FakeRD())
    be._sleep = lambda *_: None
    local = tmp_path / "seed_flat.db"
    local.write_bytes(b"x" * 500)
    seen = []
    def fake_run(cmd, **kw):
        seen.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    be._remote_size = lambda *_: 500          # matches immediately
    be._rsync_up(local, "~/w/shard_cache_seed.db")
    rs = next(c for c in seen if c and c[0] == "rsync")
    j = " ".join(rs)
    assert "--partial" in rs and "--append" in rs and "--inplace" in rs
    assert "--no-whole-file" in rs
    assert any(x.startswith("--timeout=") for x in rs)
    e_idx = rs.index("-e")
    assert "ControlPath=none" in rs[e_idx + 1]
    assert rs[-1].endswith(":~/w/shard_cache_seed.db")


def test_rsync_up_loops_until_remote_size_reaches_local(tmp_path):
    be = _prepared(tmp_path, _FakeRD())
    be._sleep = lambda *_: None
    local = tmp_path / "seed_flat.db"
    local.write_bytes(b"x" * 900)
    be._run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr="")
    sizes = iter([300, 600, 900])
    be._remote_size = lambda *_: next(sizes)
    be._rsync_up(local, "~/w/seed.db")        # must return without raising


def test_rsync_up_aborts_when_no_forward_progress(tmp_path):
    be = _prepared(tmp_path, _FakeRD())
    be._sleep = lambda *_: None
    local = tmp_path / "seed_flat.db"
    local.write_bytes(b"x" * 900)
    be._run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._remote_size = lambda *_: 128          # stuck, never grows
    with pytest.raises(RuntimeError) as e:
        be._rsync_up(local, "~/w/seed.db")
    assert "no forward progress" in str(e.value)
    assert "128/900" in str(e.value)


def test_rsync_up_treats_a_timed_out_round_as_a_retry(tmp_path):
    import subprocess as _sp
    be = _prepared(tmp_path, _FakeRD())
    be._sleep = lambda *_: None
    local = tmp_path / "seed_flat.db"
    local.write_bytes(b"x" * 400)
    calls = {"n": 0}
    def fake_run(cmd, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sp.TimeoutExpired(cmd, 1)   # first round dies mid-transfer
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    be._run = fake_run
    sizes = iter([150, 400])
    be._remote_size = lambda *_: next(sizes)
    be._rsync_up(local, "~/w/seed.db")         # timeout is not fatal
    assert calls["n"] == 2


def test_upload_seed_uses_rsync_not_scp(tmp_path, monkeypatch):
    import sqlite3 as _sq
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    seed = tmp_path / "seed.db"
    c = _sq.connect(seed)
    c.execute("create table embed_cache(k text)")
    c.executemany("insert into embed_cache values (?)", [("a",), ("b",), ("c",)])
    c.commit(); c.close()
    used = {}
    monkeypatch.setattr(be, "_rsync_up", lambda l, r: used.setdefault("rsync", (l, r)))
    monkeypatch.setattr(be, "_scp_to", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scp must not be used for the seed")))
    monkeypatch.setattr(be, "_ssh_check", lambda *_: "3")   # remote row count
    assert be._upload_seed("~/lex-au-search/run/shard_000", seed) is None
    assert "rsync" in used


def test_ssh_base_has_no_remote_command(tmp_path):
    be = _make_runpod_backend(tmp_path)  # existing helper in this file
    be._ip, be._port = "1.2.3.4", "22"
    base = be._ssh_base()
    assert base[0] == "ssh"
    assert base[-1] == "22" and base[-2] == "-p"
    assert "root@1.2.3.4" in base
    # no shell command trailing
    assert not any(tok.startswith("python -m") or "cat >" in tok for tok in base)


def test_write_hf_token_pipes_via_stdin_not_argv(tmp_path, monkeypatch):
    be = _make_runpod_backend(tmp_path)
    be._ip, be._port = "1.2.3.4", "22"
    captured = {}
    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input")
        class _R: returncode = 0
        return _R()
    monkeypatch.setattr(be, "_run", _fake_run)
    be._write_hf_token("hf_secretvalue")
    assert "hf_secretvalue" not in " ".join(captured["cmd"])
    assert captured["input"] == "hf_secretvalue"
    assert "cat > ~/.cache/huggingface/token" in " ".join(captured["cmd"])
