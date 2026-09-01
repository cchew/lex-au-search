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


# --------------------------------------------------------------- Task 9 tests


def _prepared(tmp_path, rd, **kw):
    be = _be(tmp_path, rd, **kw)
    be._pod_id = "pod_new"; be._ip = "1.2.3.4"; be._port = 22001
    be._lock_fh = open(tmp_path / ".lk", "w")
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
    assert "LEXAU_EMBED_BATCH_SIZE=16" in j
    assert "( setsid -w bash ~/lex-au-search/scripts/ingest_shard.sh 0 300 " in j
    assert "job.exitcode.tmp" in j and "mv " in j and "job.exitcode ) &" in j


def test_poll_treats_absent_or_empty_exitcode_as_running(tmp_path):
    rd = _FakeRD()
    be = _prepared(tmp_path, rd)
    polls = ["", "", "0"]
    def fake_run(cmd, **kw):
        s = cmd[-1]
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
    be._deadline_s = 0.0  # force the 25-min bound immediately for the test
    res = be.run_shard(0, 300, seed_cache=None)
    assert res.ok is False
    assert "unreachable" in res.diagnosis.lower()
    assert rd.terminated == []          # run_shard never terminates
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
