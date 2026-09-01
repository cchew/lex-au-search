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
