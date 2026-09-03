import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from lexausearch import hf_cache


def test_repo_constant():
    assert hf_cache.HF_CACHE_REPO == "cchew/lex-au-search-embed-cache"


def test_shard_names():
    assert hf_cache._shard_db_name(0) == "shard_000_checkpoint_cache.db"
    assert hf_cache._shard_json_name(10) == "shard_010.json"


def test_resolve_token_precedence(tmp_path, monkeypatch):
    tf = tmp_path / "tok"
    tf.write_text("from-file\n")
    hub_tok = tmp_path / "hub_token"
    hub_tok.write_text("from-hub\n")
    monkeypatch.setattr(hf_cache, "_HUB_TOKEN_PATH", hub_tok)

    assert hf_cache._resolve_token("explicit", str(tf)) == "explicit"
    assert hf_cache._resolve_token(None, str(tf)) == "from-file"
    assert hf_cache._resolve_token(None, None) == "from-hub"
    monkeypatch.setattr(hf_cache, "_HUB_TOKEN_PATH", tmp_path / "missing")
    assert hf_cache._resolve_token(None, None) is None


def test_read_catalogue_parses(monkeypatch, tmp_path):
    cat = tmp_path / "catalogue.json"
    cat.write_text('{"dense_model": "m", "shard_size": 300, "total_acts": 3076, '
                   '"master": {"row_count": 0, "generation": 0, "updated_at": null}}')
    monkeypatch.setattr(hf_cache, "_hf_download", lambda repo, filename, token: str(cat))
    c = hf_cache.read_catalogue(token=None)
    assert c.dense_model == "m"
    assert c.total_acts == 3076
    assert c.master["generation"] == 0


def test_read_catalogue_absent_returns_none(monkeypatch):
    def _raise(*a, **k):
        from huggingface_hub.utils import EntryNotFoundError
        raise EntryNotFoundError("no catalogue")
    monkeypatch.setattr(hf_cache, "_hf_download", _raise)
    assert hf_cache.read_catalogue(token=None) is None


def test_check_model_cold_when_no_sidecar(monkeypatch):
    def _raise(*a, **k):
        from huggingface_hub.utils import EntryNotFoundError
        raise EntryNotFoundError("no sidecar")
    monkeypatch.setattr(hf_cache, "_hf_download", _raise)
    assert isinstance(hf_cache.check_model(0, "m", token=None), hf_cache.Cold)


def test_check_model_ok_and_mismatch(monkeypatch, tmp_path):
    sidecar = tmp_path / "shard_000.json"
    sidecar.write_text('{"model_name": "old-model", "row_count": 5, "generation": 2, '
                       '"updated_at": "t", "sha256": "x", "status": "partial"}')
    monkeypatch.setattr(hf_cache, "_hf_download", lambda repo, filename, token: str(sidecar))
    assert isinstance(hf_cache.check_model(0, "old-model", token=None), hf_cache.Ok)
    v = hf_cache.check_model(0, "new-model", token=None)
    assert isinstance(v, hf_cache.Mismatch)
    assert v.old == "old-model" and v.new == "new-model"


import hashlib
import sqlite3
from huggingface_hub.utils import EntryNotFoundError


def _make_db(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE embed_cache (id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
    conn.executemany("INSERT INTO embed_cache VALUES (?, ?)",
                     [(f"id{i}", b"\x00\x00\x00\x00") for i in range(rows)])
    conn.commit()
    conn.close()


def _sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _wire_hf(monkeypatch, db_path, sidecar_dict):
    def _dl(repo, filename, token):
        if filename.endswith(".json"):
            p = Path(db_path).parent / "sc.json"
            p.write_text(__import__("json").dumps(sidecar_dict))
            return str(p)
        return str(db_path)
    monkeypatch.setattr(hf_cache, "_hf_download", _dl)


def test_fetch_happy_path(tmp_path, monkeypatch):
    src = tmp_path / "src.db"
    _make_db(src, 5)
    sidecar = {"model_name": "m", "row_count": 5, "generation": 3,
               "updated_at": "t", "sha256": _sha256(src), "status": "partial"}
    _wire_hf(monkeypatch, src, sidecar)
    dest = tmp_path / "work"
    dest.mkdir()
    meta = hf_cache.fetch_shard_cache(0, dest, token=None, expect_model="m", seed_as="shard_cache_seed.db")
    assert meta.row_count == 5 and meta.generation == 3
    assert (dest / "shard_cache_seed.db").exists()


def test_fetch_cold_returns_none(tmp_path, monkeypatch):
    def _dl(repo, filename, token):
        raise EntryNotFoundError("nope")
    monkeypatch.setattr(hf_cache, "_hf_download", _dl)
    assert hf_cache.fetch_shard_cache(0, tmp_path, token=None) is None


def test_fetch_sha256_mismatch_raises(tmp_path, monkeypatch):
    src = tmp_path / "src.db"
    _make_db(src, 5)
    sidecar = {"model_name": "m", "row_count": 5, "generation": 1,
               "updated_at": "t", "sha256": "deadbeef", "status": "partial"}
    _wire_hf(monkeypatch, src, sidecar)
    with pytest.raises(hf_cache.HfCacheCorrupt):
        hf_cache.fetch_shard_cache(0, tmp_path, token=None)


def test_fetch_row_count_mismatch_raises(tmp_path, monkeypatch):
    src = tmp_path / "src.db"
    _make_db(src, 3)
    sidecar = {"model_name": "m", "row_count": 99, "generation": 1,
               "updated_at": "t", "sha256": _sha256(src), "status": "partial"}
    _wire_hf(monkeypatch, src, sidecar)
    with pytest.raises(hf_cache.HfCacheCorrupt):
        hf_cache.fetch_shard_cache(0, tmp_path, token=None)


def test_fetch_model_mismatch_raises(tmp_path, monkeypatch):
    src = tmp_path / "src.db"
    _make_db(src, 5)
    sidecar = {"model_name": "old", "row_count": 5, "generation": 1,
               "updated_at": "t", "sha256": _sha256(src), "status": "partial"}
    _wire_hf(monkeypatch, src, sidecar)
    with pytest.raises(hf_cache.HfCacheModelMismatch):
        hf_cache.fetch_shard_cache(0, tmp_path, token=None, expect_model="new")
