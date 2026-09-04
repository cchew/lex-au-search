import sqlite3
import pytest
from lexausearch.cache import merge_cache_files, EmbedCache


def test_merge_cache_files_closes_output_connection(tmp_path):
    """Test that merge_cache_files closes the output connection.

    If the connection is leaked, this WAL-mode file would still be locked on
    some platforms; more robustly, we assert no lingering sqlite connections
    by opening exclusively.
    """
    src = tmp_path / "src.db"
    EmbedCache(src)  # create an empty valid cache db
    out = tmp_path / "out.db"

    merge_cache_files([src], out)

    # Try to open the output DB in exclusive mode.
    # If the connection leaked, this would fail with a timeout/lock error.
    conn = sqlite3.connect(f"file:{out}?mode=rw", uri=True, timeout=1)
    try:
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("COMMIT")
    finally:
        conn.close()


def test_merge_cache_files_closes_every_connection_when_a_shard_raises(tmp_path, monkeypatch):
    """A raise inside the per-shard loop must not leak the per-shard connection
    (previously closed only on the happy path) or the output connection."""
    import lexausearch.cache as cache_mod

    good = tmp_path / "good.db"
    EmbedCache(good)
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"this is definitely not a sqlite database" * 32)
    out = tmp_path / "out.db"

    opened = []
    real_connect = sqlite3.connect

    def _tracking(*a, **k):
        conn = real_connect(*a, **k)
        opened.append(conn)
        return conn

    monkeypatch.setattr(cache_mod.sqlite3, "connect", _tracking)

    with pytest.raises(sqlite3.DatabaseError):
        merge_cache_files([good, bad], out)

    assert opened, "expected merge_cache_files to open connections"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")  # ProgrammingError == already closed
