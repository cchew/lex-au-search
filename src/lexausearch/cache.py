from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np

DENSE_VECTOR_SIZE = 768  # BAAI/bge-base-en-v1.5 output dimension
_UUID5_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace


class EmbedCache:
    """Persistent UUID5-keyed dense embedding cache backed by SQLite.

    Key: UUID5(URL_NS, text) — deterministic, content-addressed.
    Value: float32 dense vector (768-dim for BAAI/bge-base-en-v1.5), stored as a BLOB.

    Backed by SQLite rather than a Qdrant collection: this cache is only ever
    read by exact-id lookup, never vector similarity search, and Qdrant's
    local/embedded mode is documented as unsuitable above ~20K points -
    measured ~25x the peak memory of SQLite for the same 100K-vector dataset
    (2026-07-24, after an OOM kill on Colab traced to this).

    The cache is optional: Indexer falls back to client.add() when cache=None.
    """

    def __init__(self, db_path: str | Path, vector_size: int = DENSE_VECTOR_SIZE) -> None:
        self._vector_size = vector_size
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embed_cache (id TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        self._conn.commit()

    def _cache_id(self, text: str) -> str:
        return str(uuid.uuid5(_UUID5_NS, text))

    def get(self, text: str) -> list[float] | None:
        """Return cached dense vector for text, or None on miss."""
        row = self._conn.execute(
            "SELECT vector FROM embed_cache WHERE id = ?", (self._cache_id(text),)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).tolist()

    def put(self, text: str, vector: list[float]) -> None:
        """Store dense vector for text."""
        self.put_batch({text: vector})

    def put_batch(self, vectors: dict[str, list[float]]) -> None:
        """Store dense vectors for multiple texts in a single transaction."""
        if not vectors:
            return
        rows = [
            (self._cache_id(text), np.asarray(vec, dtype=np.float32).tobytes())
            for text, vec in vectors.items()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embed_cache VALUES (?, ?)", rows
        )
        self._conn.commit()

    def get_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """Return {text: vector} for all cache hits. Misses are absent."""
        if not texts:
            return {}
        id_to_text = {self._cache_id(t): t for t in texts}
        placeholders = ",".join("?" * len(id_to_text))
        rows = self._conn.execute(
            f"SELECT id, vector FROM embed_cache WHERE id IN ({placeholders})",
            list(id_to_text.keys()),
        ).fetchall()
        return {
            id_to_text[row_id]: np.frombuffer(vec, dtype=np.float32).tolist()
            for row_id, vec in rows
        }
