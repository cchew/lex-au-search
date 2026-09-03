import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from backends.base import SeedMode, checkpoint_cache_path
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
    be.run_shard(0, 300, SeedMode.HF)
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
    be.run_shard(0, 300, SeedMode.HF)
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
    res = be.run_shard(0, 300, SeedMode.HF)
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
    res = be.run_shard(0, 300, SeedMode.HF)
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
    be.run_shard(1, 300, SeedMode.HF)
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
    res = be.run_shard(0, 300, SeedMode.HF)
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
    res = be.run_shard(0, 300, SeedMode.HF)
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
    res = be.run_shard(0, 300, SeedMode.HF)
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
    res = be.run_shard(0, 300, SeedMode.HF)
    assert res.ok is False
    assert "download" in res.diagnosis.lower()
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
    res = be.run_shard(0, 300, SeedMode.HF)
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
    be.run_shard(0, 300, SeedMode.HF)
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


# ----------------------------------------------- Task 14: SeedMode fetch swap


def test_seedmode_hf_issues_hf_fetch_over_ssh(tmp_path, monkeypatch):
    be = _make_runpod_backend(tmp_path)
    be._ip, be._port = "1.2.3.4", "22"
    calls = []
    class _R:
        returncode = 0
        stdout = "fetched shard 0 gen 3"
        stderr = ""
    monkeypatch.setattr(be, "_gpu_pids", lambda: "")  # GPU idle; Step 0 proceeds
    monkeypatch.setattr(be, "_ssh", lambda remote, check=True: calls.append(remote) or _R())
    monkeypatch.setattr(be, "_launch_detached", lambda *a, **k: None)  # stub the rest
    monkeypatch.setattr(
        be, "_poll_to_terminal",
        lambda *a, **k: __import__("backends.base", fromlist=["ShardResult"]).ShardResult(
            0, True, tmp_path / "s.zip", tmp_path / "c.zip", ""
        ),
    )
    be.run_shard(0, 300, SeedMode.HF)
    assert any("python -m lexausearch.hf_cache fetch 0" in c for c in calls)
    assert any("--expect-model snowflake/snowflake-arctic-embed-l" in c for c in calls)


def test_seedmode_hf_fetch_nonzero_fails_shard_without_launch(tmp_path, monkeypatch):
    be = _make_runpod_backend(tmp_path)
    be._ip, be._port = "1.2.3.4", "22"
    class _R:
        returncode = 4
        stdout = ""
        stderr = "MODEL MISMATCH: shard 0 ..."
    monkeypatch.setattr(be, "_ssh", lambda remote, check=True: _R())
    launched = []
    monkeypatch.setattr(be, "_launch_detached", lambda *a, **k: launched.append(1))
    res = be.run_shard(0, 300, SeedMode.HF)
    assert res.ok is False and "MODEL MISMATCH" in res.diagnosis
    assert not launched


def test_seedmode_overwrite_skips_fetch(tmp_path, monkeypatch):
    be = _make_runpod_backend(tmp_path)
    be._ip, be._port = "1.2.3.4", "22"
    calls = []
    class _R:
        returncode = 0; stdout = ""; stderr = ""
    monkeypatch.setattr(be, "_ssh", lambda remote, check=True: calls.append(remote) or _R())
    monkeypatch.setattr(be, "_launch_detached", lambda *a, **k: None)
    monkeypatch.setattr(
        be, "_poll_to_terminal",
        lambda *a, **k: __import__("backends.base", fromlist=["ShardResult"]).ShardResult(
            0, True, tmp_path / "s.zip", tmp_path / "c.zip", ""
        ),
    )
    be.run_shard(0, 300, SeedMode.SEEDLESS_OVERWRITE)
    assert not any("hf_cache fetch" in c for c in calls)
    assert be._overwrite is True
