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


def test_get_batch_never_issues_a_query_past_the_safe_placeholder_count(tmp_path):
    """Real failure 2026-10-13: a corpus-wide cache check against ~402K
    distinct chunk texts crashed on a Colab Python 3.13 build with
    sqlite3.OperationalError: too many SQL variables -- get_batch built
    one query with a placeholder per text. SQLITE_MAX_VARIABLE_NUMBER is
    compile-time and varies by build (this repo's own .venv tolerates
    400K+ fine, so the fix can't be tested by just picking a big N and
    expecting a crash locally) -- assert the implementation issues
    multiple smaller SELECTs rather than one giant one, same batching
    already applied to gc()'s delete path. set_trace_callback is the
    real per-instance sqlite3 hook for this (Connection itself is an
    immutable C type -- its methods can't be monkeypatched)."""
    cache = EmbedCache(tmp_path / "cache.db")
    n = 1500
    cache.put_batch({f"text {i}": [float(i)] for i in range(n)})

    select_statements = []
    cache._conn.set_trace_callback(
        lambda sql: select_statements.append(sql) if "WHERE id IN" in sql and "SELECT" in sql else None
    )

    hits = cache.get_batch([f"text {i}" for i in range(n)])
    cache._conn.set_trace_callback(None)

    assert len(hits) == n
    assert hits["text 0"] == pytest.approx([0.0])
    assert hits["text 1499"] == pytest.approx([1499.0])
    # 1500 inputs batched at <=500/query must take at least 3 SELECTs --
    # the pre-fix code issues exactly 1, with 1500 placeholders in it.
    assert len(select_statements) >= 3, select_statements


def test_get_batch_mixed_hits_and_misses_across_batches(tmp_path):
    cache = EmbedCache(tmp_path / "cache.db")
    n = 1500
    cache.put_batch({f"text {i}": [float(i)] for i in range(n)})

    # Half of these exist, half don't -- misses must be silently absent,
    # not raise, same contract as the small-input case, across multiple
    # internal batches.
    query = [f"text {i}" for i in range(n)] + [f"missing {i}" for i in range(n)]
    hits = cache.get_batch(query)

    assert len(hits) == n
    assert all(f"missing {i}" not in hits for i in range(n))


def test_gc_deletes_rows_whose_text_is_no_longer_live(tmp_path):
    cache = EmbedCache(tmp_path / "cache.db")
    cache.put_batch({
        "live text": [0.1, 0.2],
        "orphaned text": [0.3, 0.4],
    })
    assert cache.row_count() == 2

    removed = cache.gc(live_texts={"live text"})

    assert removed == 1
    assert cache.row_count() == 1
    assert cache.get("live text") == pytest.approx([0.1, 0.2])
    assert cache.get("orphaned text") is None


def test_gc_is_a_noop_when_every_row_is_still_live(tmp_path):
    cache = EmbedCache(tmp_path / "cache.db")
    cache.put_batch({"a": [1.0], "b": [2.0]})

    removed = cache.gc(live_texts={"a", "b", "c (never cached)"})

    assert removed == 0
    assert cache.row_count() == 2


def test_gc_batches_deletes_past_the_sqlite_bound_variable_limit(tmp_path):
    """Orphan count can exceed SQLite's per-statement bound-parameter limit on
    a real production cache (hundreds of thousands of rows) -- a naive single
    `DELETE ... WHERE id IN (...)` would raise sqlite3.OperationalError."""
    cache = EmbedCache(tmp_path / "cache.db")
    n = 1500
    cache.put_batch({f"orphan {i}": [float(i)] for i in range(n)})
    assert cache.row_count() == n

    removed = cache.gc(live_texts=set())

    assert removed == n
    assert cache.row_count() == 0


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
