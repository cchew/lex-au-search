# tests/test_backends_colab.py
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backends.base import checkpoint_cache_path
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
        self.calls.append(("run_background", name))
        return 4321

    def poll_status(self, name, pid):
        v = self._poll_sequence.pop(0)
        self.calls.append(("poll_status", v))
        return v

    def sample_resources(self, name):
        return "RAM(MB): ... | GPU(MB used,total): 3, 15360"

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
        if not attr.startswith("__"):
            monkeypatch.setattr(colab_mod.cd, attr, getattr(rec, attr), raising=False)


def test_success_path_call_order_and_zip_written(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["running", "done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)

    be = ColabBackend(shards_dir=tmp_path, gpu="T4")
    res = be.run_shard(0, 300, seed_cache=None)

    assert res.ok is True
    assert res.storage_zip == tmp_path / "shard_000.zip"
    assert (tmp_path / "shard_000.zip").exists()
    names = [c[0] for c in rec.calls]
    assert names == [
        "create_session", "verify_session", "run_background",
        "poll_status", "poll_status", "download", "download", "stop_session",
    ]


def test_seed_uploaded_when_accumulator_exists(monkeypatch, tmp_path):
    checkpoint_cache_path(tmp_path, 0).write_bytes(b"db")
    rec = _Recorder(poll_sequence=["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)

    ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, seed_cache=checkpoint_cache_path(tmp_path, 0))

    assert ("upload", "lexau-shard-0", "/content/shard_cache_seed.db") in rec.calls


def test_session_lost_returns_failure_without_checkpoint_pull(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=["running", "session_lost"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)

    res = ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, seed_cache=None)

    assert res.ok is False
    assert "session lost" in res.diagnosis.lower()
    assert rec.checkpoint_calls == 0
    assert ("stop_session", "lexau-shard-0") in rec.calls


def test_checkpoint_pulled_on_interval(monkeypatch, tmp_path):
    # 10 "running" polls then "done" -> exactly one checkpoint pull at poll 10
    rec = _Recorder(poll_sequence=["running"] * 10 + ["done"])
    _install_recorder(monkeypatch, rec)
    monkeypatch.setattr(colab_mod.time, "sleep", lambda *_: None)

    ColabBackend(shards_dir=tmp_path, gpu="T4").run_shard(0, 300, seed_cache=None)

    assert rec.checkpoint_calls == 1


def test_teardown_sweeps_orphans(monkeypatch, tmp_path):
    rec = _Recorder(poll_sequence=[])
    _install_recorder(monkeypatch, rec)
    ColabBackend(shards_dir=tmp_path, gpu="T4").teardown()
    assert ("_release_orphaned_assignments",) in rec.calls
