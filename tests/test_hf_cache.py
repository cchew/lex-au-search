import sys
from pathlib import Path
from unittest.mock import Mock

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


class _FakeApi:
    def __init__(self, fail_once_with_412=False):
        self.commits = []
        self.fail_once_with_412 = fail_once_with_412
        self._create_commit_called = 0

    def create_commit(self, repo_id, repo_type, operations, commit_message, parent_commit=None, token=None):
        self._create_commit_called += 1
        if self.fail_once_with_412 and self._create_commit_called == 1:
            from huggingface_hub.utils import HfHubHTTPError
            mock_response = Mock()
            mock_response.status_code = 412
            raise HfHubHTTPError("conflict", response=mock_response)
        self.commits.append({"ops": operations, "parent": parent_commit})


def test_push_non_overwrite_merges_and_bumps_generation(tmp_path, monkeypatch):
    head = tmp_path / "head.db"
    _make_db(head, 4)
    local = tmp_path / "local.db"
    _make_db(local, 6)  # 6 fresh rows, ids id0..id5; head has id0..id3
    head_sidecar = {"model_name": "m", "row_count": 4, "generation": 7,
                    "updated_at": "t", "sha256": _sha256(head), "status": "partial"}
    _wire_hf(monkeypatch, head, head_sidecar)
    fake = _FakeApi()
    monkeypatch.setattr(hf_cache, "_api", lambda: fake)
    captured = {}
    real_merge = hf_cache.merge_cache_files
    def _spy(paths, out):
        captured["paths"] = [str(p) for p in paths]
        return real_merge(paths, out)
    monkeypatch.setattr(hf_cache, "merge_cache_files", _spy)

    meta = hf_cache.push_shard_cache(0, local, model_name="m", status="partial", token="t")

    assert meta.generation == 8
    assert len(captured["paths"]) == 2  # [head, local]
    # Verify paths are distinct: head_db and working.db
    assert captured["paths"][0] != captured["paths"][1]
    # Verify first path is the head DB (created by _hf_download stub)
    assert Path(captured["paths"][0]).name == Path(head).name
    # Verify second path is working.db
    assert "working.db" in captured["paths"][1]
    assert len(fake.commits) == 1
    assert len(fake.commits[0]["ops"]) == 2  # DB + sidecar


def test_push_overwrite_skips_merge_and_head_check(tmp_path, monkeypatch):
    local = tmp_path / "local.db"
    _make_db(local, 3)
    # head sidecar has a DIFFERENT model; overwrite must ignore it
    head_sidecar = {"model_name": "old", "row_count": 4, "generation": 2,
                    "updated_at": "t", "sha256": "x", "status": "partial"}
    _wire_hf(monkeypatch, local, head_sidecar)
    fake = _FakeApi()
    monkeypatch.setattr(hf_cache, "_api", lambda: fake)
    merge_called = []
    monkeypatch.setattr(hf_cache, "merge_cache_files", lambda *a: merge_called.append(1))

    meta = hf_cache.push_shard_cache(0, local, model_name="new", status="complete",
                                     token="t", overwrite=True)

    assert not merge_called
    assert meta.model_name == "new" and meta.generation == 3


def test_push_non_overwrite_head_model_mismatch_raises(tmp_path, monkeypatch):
    local = tmp_path / "local.db"
    _make_db(local, 3)
    head_sidecar = {"model_name": "old", "row_count": 4, "generation": 2,
                    "updated_at": "t", "sha256": "x", "status": "partial"}
    _wire_hf(monkeypatch, local, head_sidecar)
    fake = _FakeApi()
    monkeypatch.setattr(hf_cache, "_api", lambda: fake)
    with pytest.raises(hf_cache.HfCacheModelMismatch):
        hf_cache.push_shard_cache(0, local, model_name="new", status="partial", token="t")
    # Verify NO commit was made (raises before commit)
    assert fake.commits == []


def test_push_live_backs_up_before_reading(tmp_path, monkeypatch):
    live_db = tmp_path / "live.db"
    _make_db(live_db, 2)
    _wire_hf(monkeypatch, live_db, None)  # cold head
    monkeypatch.setattr(hf_cache, "_read_sidecar", lambda i, t: None)
    fake = _FakeApi()
    monkeypatch.setattr(hf_cache, "_api", lambda: fake)
    call_order = []
    real_backup = hf_cache._sqlite_backup
    def _spy_backup(src, dst):
        call_order.append("backup")
        return real_backup(src, dst)
    real_row_count = hf_cache._row_count
    def _spy_row_count(db_path):
        call_order.append("row_count")
        return real_row_count(db_path)
    monkeypatch.setattr(hf_cache, "_sqlite_backup", _spy_backup)
    monkeypatch.setattr(hf_cache, "_row_count", _spy_row_count)

    meta = hf_cache.push_shard_cache(0, live_db, model_name="m", status="partial",
                                    token="t", live=True)
    # Verify backup precedes row_count (reading the DB)
    assert call_order[0] == "backup"
    assert "row_count" in call_order
    assert meta.generation == 1  # no head


def test_push_412_retry_re_reads_head_and_bumps_generation(tmp_path, monkeypatch):
    local = tmp_path / "local.db"
    _make_db(local, 5)
    head = tmp_path / "head.db"
    _make_db(head, 3)

    # First read returns initial head (gen 2), second read (after 412) returns updated head (gen 5)
    read_sidecar_calls = [0]
    initial_head = {"model_name": "m", "row_count": 3, "generation": 2,
                    "updated_at": "t", "sha256": _sha256(head), "status": "partial"}
    retried_head = {"model_name": "m", "row_count": 3, "generation": 5,
                    "updated_at": "t2", "sha256": _sha256(head), "status": "partial"}

    def _read_sidecar_spy(shard_index, token):
        read_sidecar_calls[0] += 1
        if read_sidecar_calls[0] == 1:
            return hf_cache.ShardCacheMeta(**initial_head)
        else:  # Second call (during retry)
            return hf_cache.ShardCacheMeta(**retried_head)

    monkeypatch.setattr(hf_cache, "_read_sidecar", _read_sidecar_spy)

    def _hf_download_spy(repo, filename, token):
        if filename.endswith(".json"):
            # Return initial head sidecar for first fetch
            p = tmp_path / "sc.json"
            p.write_text(__import__("json").dumps(initial_head))
            return str(p)
        return str(head)

    monkeypatch.setattr(hf_cache, "_hf_download", _hf_download_spy)

    # FakeApi that fails on first call with 412, succeeds on retry
    fake = _FakeApi(fail_once_with_412=True)
    monkeypatch.setattr(hf_cache, "_api", lambda: fake)

    meta = hf_cache.push_shard_cache(0, local, model_name="m", status="partial", token="t")

    # After 412 retry, generation should be retried_head.generation + 1 = 5 + 1 = 6
    assert meta.generation == 6
    # Verify create_commit was called twice (first failed, second succeeded)
    assert fake._create_commit_called == 2
    # Verify final commit succeeded
    assert len(fake.commits) == 1
