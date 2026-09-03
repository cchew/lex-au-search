# tests/test_backends_colab.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backends.base import SeedMode
from backends.colab import ColabBackend
import backends.colab as colab_mod


class _Recorder:
    """Stand-in for the colab_driver module. Records calls; scripted returns."""

    def __init__(self, poll_sequence):
        self.calls = []
        self._poll_sequence = list(poll_sequence)
        self.checkpoint_calls = 0

    def create_session(self, name, gpu="T4"):
        self.calls.append(("create_session", name, gpu))
        return {"ok": True, "stderr": ""}

    def verify_session(self, name):
        self.calls.append(("verify_session", name))
        return True

    def upload(self, name, local, remote):
        self.calls.append(("upload", name, remote))
        return True

    def run_background(self, name, cmd):
        self.last_remote_cmd = cmd
        self.calls.append(("run_background", name, cmd))
        return 4321

    def poll_status(self, name, pid):
        v = self._poll_sequence.pop(0)
        self.calls.append(("poll_status", v))
        return v

    def sample_resources(self, name):
        return "RAM(MB): ... | GPU(MB used,total): 3, 15360"

    def exec_sync(self, name, cmd, timeout=180):
        self.calls.append(("exec_sync", name, cmd))
        self.last_exec_cmd = cmd
        return (0, "", "")

    def checkpoint_cache(self, name, cache_path="shard_cache.db"):
        self.checkpoint_calls += 1
        return "no_db"

    def download(self, name, remote, local):
        self.calls.append(("download", remote))
        Path(local).write_bytes(b"zip")
        return True

    def diagnose_failure(self, name, pid):
        return "exited with code 1"

    def tail_log(self, name, n=50):
        return "...log tail..."

    def stop_session(self, name):
        self.calls.append(("stop_session", name))

    def _release_orphaned_assignments(self):
        self.calls.append(("_release_orphaned_assignments",))


def _install_recorder(monkeypatch, rec):
    for attr in dir(rec):
        if not attr.startswith("__") and callable(getattr(rec, attr)):
            monkeypatch.setattr(colab_mod.cd, attr, getattr(rec, attr), raising=False)


def test_success_path_call_order_and_zip_written(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["running", "done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.delenv("HF_CACHE_WRITE_TOKEN", raising=False)

    be = ColabBackend(shards_dir=tmp_path, gpu="T4")
    res = be.run_shard(0, 300, SeedMode.HF)

    assert res.ok is True
    assert res.storage_zip == tmp_path / "shard_000.zip"
    assert (tmp_path / "shard_000.zip").exists()
    names = [c[0] for c in rec.calls]
    # No HF_CACHE_WRITE_TOKEN -> no token upload. The poll loop no longer calls
    # checkpoint_cache; the terminal "done" path pushes once via exec_sync
    # (--status complete).
    assert names == [
        "create_session", "verify_session", "run_background",
        "poll_status", "poll_status", "download", "download",
        "exec_sync", "stop_session",
    ]


def test_remote_cmd_contains_hf_fetch(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.delenv("HF_CACHE_WRITE_TOKEN", raising=False)

    ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.HF)

    cmd = rec.last_remote_cmd
    assert "python -m lexausearch.hf_cache fetch 0" in cmd
    assert "--expect-model snowflake/snowflake-arctic-embed-l" in cmd
    # No `|| true` swallowing the fetch: the fetch segment (up to the next &&)
    # must be a hard link in the chain.
    assert "|| true" not in cmd.split("hf_cache fetch")[1].split("&&")[0]
    # Ordering: fetch sits between setup and ingest.
    assert cmd.index("setup_gpu_env.sh") < cmd.index("hf_cache fetch") < cmd.index("ingest_shard.sh")


def test_seedless_overwrite_omits_fetch(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.delenv("HF_CACHE_WRITE_TOKEN", raising=False)

    ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.SEEDLESS_OVERWRITE)

    assert "hf_cache fetch" not in rec.last_remote_cmd


def test_token_uploaded_when_env_set(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.setenv("HF_CACHE_WRITE_TOKEN", "hf_x")

    ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.HF)

    assert ("upload", "lexau-shard-0", "/content/.hf_token") in rec.calls
    # Token upload happens before the job is launched.
    upload_i = rec.calls.index(("upload", "lexau-shard-0", "/content/.hf_token"))
    run_i = next(i for i, c in enumerate(rec.calls) if c[0] == "run_background")
    assert upload_i < run_i


def test_token_upload_failure_fails_shard(monkeypatch, tmp_path):
    class _FailUpload(_Recorder):
        def upload(self, name, local, remote):
            self.calls.append(("upload", name, remote))
            return False

    rec = _FailUpload(poll_sequence=["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.setenv("HF_CACHE_WRITE_TOKEN", "hf_x")

    res = ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.HF)

    assert res.ok is False
    assert "token upload" in res.diagnosis.lower()
    assert not any(c[0] == "run_background" for c in rec.calls)
    assert ("stop_session", "lexau-shard-0") in rec.calls


def test_session_lost_returns_failure_without_checkpoint_push(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["running", "session_lost"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.delenv("HF_CACHE_WRITE_TOKEN", raising=False)

    res = ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.HF)

    assert res.ok is False
    assert "session lost" in res.diagnosis.lower()
    assert not any(c[0] == "exec_sync" for c in rec.calls)
    assert ("stop_session", "lexau-shard-0") in rec.calls


def test_push_checkpoint_uses_exec_sync(monkeypatch, tmp_path):
    # 10 "running" polls then "done" -> exactly one interval push at poll 10,
    # plus the terminal --status complete push.
    rec = _Recorder(poll_sequence=["running"] * 10 + ["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.delenv("HF_CACHE_WRITE_TOKEN", raising=False)

    ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.HF)

    pushes = [c for c in rec.calls if c[0] == "exec_sync"]
    assert any(
        "python -m lexausearch.hf_cache push 0" in c[2] and "--status partial" in c[2]
        for c in pushes
    )
    assert any(
        "python -m lexausearch.hf_cache push 0" in c[2] and "--status complete" in c[2]
        for c in pushes
    )
    # Interval push targets the VM's live cache under the cloned repo.
    partial = next(c for c in pushes if "--status partial" in c[2])
    assert partial[2].startswith("cd /content/repo && ")
    assert "--db shard_cache.db" in partial[2]
    assert "--live" in partial[2]
    assert "--token-file /content/.hf_token" in partial[2]
    assert "--overwrite" not in partial[2]


def test_push_checkpoint_overwrite_on_seedless(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["running"] * 10 + ["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.delenv("HF_CACHE_WRITE_TOKEN", raising=False)

    ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.SEEDLESS_OVERWRITE)

    pushes = [c for c in rec.calls if c[0] == "exec_sync"]
    assert pushes and all("--overwrite" in c[2] for c in pushes)


def test_push_checkpoint_non_fatal_on_nonzero_rc(monkeypatch, tmp_path):
    class _BadPush(_Recorder):
        def exec_sync(self, name, cmd, timeout=180):
            self.calls.append(("exec_sync", name, cmd))
            return (1, "", "boom")

    rec = _BadPush(poll_sequence=["running", "done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.delenv("HF_CACHE_WRITE_TOKEN", raising=False)

    res = ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.HF)

    # A failed terminal push must not sink an otherwise-successful shard.
    assert res.ok is True


def test_remote_cmd_uses_split_setup_and_ingest_scripts(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)
    monkeypatch.delenv("HF_CACHE_WRITE_TOKEN", raising=False)
    ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, SeedMode.HF)
    cmd = rec.last_remote_cmd
    assert "bash scripts/setup_gpu_env.sh" in cmd
    assert "bash scripts/ingest_shard.sh 0 300 /content/shard_cache_seed.db" in cmd
    assert "colab_ingest_shard.sh" not in cmd


def test_teardown_sweeps_orphans(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=[])
    _install_recorder(monkeypatch, rec)
    ColabBackend(shards_dir=tmp_path, gpu="T4").teardown()
    assert ("_release_orphaned_assignments",) in rec.calls
