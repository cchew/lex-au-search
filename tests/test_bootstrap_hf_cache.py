import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bootstrap_hf_cache as boot

# Set env var for tests
os.environ.setdefault("HF_CACHE_WRITE_TOKEN", "test-token")


def _mkdb(p, rows):
    c = sqlite3.connect(str(p))
    c.execute("CREATE TABLE embed_cache (id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
    c.executemany("INSERT INTO embed_cache VALUES (?, ?)", [(f"i{n}", b"\x00") for n in range(rows)])
    c.commit(); c.close()


def test_completeness_check_aborts_on_missing_shard(tmp_path, monkeypatch, capsys):
    shards = tmp_path / "shards"; shards.mkdir()
    _mkdb(shards / "shard_000_checkpoint_cache.db", 3)  # only shard 0 of 11
    monkeypatch.setattr(boot.hf_cache, "create_cache_repo", lambda **k: None)
    rc = boot.main(["--shards-dir", str(shards), "--total-acts", "3076",
                    "--shard-size", "300", "--yes", "--dry-run"])
    assert rc != 0
    assert "missing" in capsys.readouterr().out.lower()


def test_allow_partial_proceeds(tmp_path, monkeypatch):
    shards = tmp_path / "shards"; shards.mkdir()
    _mkdb(shards / "shard_000_checkpoint_cache.db", 3)
    planned = {}
    monkeypatch.setattr(boot.hf_cache, "create_cache_repo", lambda **k: None)
    monkeypatch.setattr(boot, "_plan", lambda *a, **k: planned.setdefault("called", True))
    rc = boot.main(["--shards-dir", str(shards), "--total-acts", "3076",
                    "--shard-size", "300", "--yes", "--allow-partial", "--dry-run"])
    assert rc == 0 and planned["called"]


def test_sidecar_status_from_zip_presence(tmp_path):
    shards = tmp_path / "shards"; shards.mkdir()
    _mkdb(shards / "shard_000_checkpoint_cache.db", 5)
    (shards / "shard_000.zip").write_bytes(b"zip")  # -> complete
    _mkdb(shards / "shard_001_checkpoint_cache.db", 2)  # no zip -> partial
    work = tmp_path / "work"; work.mkdir()
    entries = boot._build_entries(shards, [0, 1], "test-model", work)
    by_i = {e["index"]: e for e in entries}
    assert by_i[0]["sidecar"]["status"] == "complete"
    assert by_i[0]["sidecar"]["generation"] == 1
    assert by_i[1]["sidecar"]["status"] == "partial"
    assert by_i[0]["sidecar"]["model_name"] == "test-model"


def test_build_entries_hashes_the_file_it_hands_on(tmp_path):
    """One .backup per shard: the bytes hashed into the sidecar must be the
    exact bytes the upload sends, or fetch fails the sha256 check forever."""
    import hashlib
    shards = tmp_path / "shards"; shards.mkdir()
    _mkdb(shards / "shard_000_checkpoint_cache.db", 7)
    work = tmp_path / "work"; work.mkdir()
    entries = boot._build_entries(shards, [0], "test-model", work)
    flat = Path(entries[0]["upload_db"])
    assert flat.is_file(), "the hashed backup must persist for the upload"
    assert hashlib.sha256(flat.read_bytes()).hexdigest() == entries[0]["sidecar"]["sha256"]


def test_uploaded_db_bytes_match_the_sidecar_sha(tmp_path, monkeypatch):
    import hashlib
    shards = tmp_path / "shards"; shards.mkdir()
    _mkdb(shards / "shard_000_checkpoint_cache.db", 4)
    monkeypatch.setattr(boot.hf_cache, "create_cache_repo", lambda **k: None)
    seen = {}

    class _Api:
        def create_commit(self, repo_id, repo_type, operations, commit_message, token):
            db_op = next(o for o in operations if str(o.path_in_repo).endswith(".db"))
            sc_op = next(o for o in operations if str(o.path_in_repo).endswith(".json"))
            seen["db_sha"] = hashlib.sha256(Path(db_op.path_or_fileobj).read_bytes()).hexdigest()
            seen["sidecar"] = json.loads(Path(sc_op.path_or_fileobj).read_text())

        def upload_file(self, **k):
            pass

    monkeypatch.setattr(boot.hf_cache, "_api", lambda: _Api())
    rc = boot.main(["--shards-dir", str(shards), "--total-acts", "300",
                    "--shard-size", "300", "--yes"])
    assert rc == 0
    assert seen["sidecar"]["sha256"] == seen["db_sha"]


def test_dry_run_makes_no_hf_api_writes(tmp_path, monkeypatch):
    shards = tmp_path / "shards"; shards.mkdir()
    _mkdb(shards / "shard_000_checkpoint_cache.db", 3)
    writes = []
    monkeypatch.setattr(boot.hf_cache, "create_cache_repo",
                        lambda **k: writes.append("create_cache_repo"))
    monkeypatch.setattr(boot.hf_cache, "_api",
                        lambda: writes.append("_api") or (_ for _ in ()).throw(
                            AssertionError("dry-run must not touch the HF API")))
    rc = boot.main(["--shards-dir", str(shards), "--total-acts", "300",
                    "--shard-size", "300", "--yes", "--dry-run"])
    assert rc == 0
    assert writes == []


def test_repo_flag_targets_the_named_repo(tmp_path, monkeypatch):
    shards = tmp_path / "shards"; shards.mkdir()
    _mkdb(shards / "shard_000_checkpoint_cache.db", 2)
    monkeypatch.setattr(boot.hf_cache, "HF_CACHE_REPO", boot.hf_cache._DEFAULT_HF_CACHE_REPO)
    created = {}
    monkeypatch.setattr(boot.hf_cache, "create_cache_repo",
                        lambda **k: created.setdefault("repo", boot.hf_cache.HF_CACHE_REPO))
    repos = []

    class _Api:
        def create_commit(self, repo_id, **k):
            repos.append(repo_id)

        def upload_file(self, repo_id, **k):
            repos.append(repo_id)

    monkeypatch.setattr(boot.hf_cache, "_api", lambda: _Api())
    rc = boot.main(["--shards-dir", str(shards), "--total-acts", "300",
                    "--shard-size", "300", "--yes", "--repo", "cchew/throwaway"])
    assert rc == 0
    assert created["repo"] == "cchew/throwaway"
    assert repos and set(repos) == {"cchew/throwaway"}
