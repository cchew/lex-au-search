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
    entries = boot._build_entries(shards, [0, 1], "test-model")
    by_i = {e["index"]: e for e in entries}
    assert by_i[0]["sidecar"]["status"] == "complete"
    assert by_i[0]["sidecar"]["generation"] == 1
    assert by_i[1]["sidecar"]["status"] == "partial"
    assert by_i[0]["sidecar"]["model_name"] == "test-model"
