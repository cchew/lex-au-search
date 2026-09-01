import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _reload_indexer():
    import lexausearch.indexer as m
    return importlib.reload(m)


def test_default_batch_size_is_16(monkeypatch):
    monkeypatch.delenv("LEXAU_EMBED_BATCH_SIZE", raising=False)
    assert _reload_indexer()._EMBED_BATCH_SIZE == 16


def test_env_overrides_batch_size(monkeypatch):
    monkeypatch.setenv("LEXAU_EMBED_BATCH_SIZE", "32")
    assert _reload_indexer()._EMBED_BATCH_SIZE == 32


def test_reset_after(monkeypatch):
    monkeypatch.delenv("LEXAU_EMBED_BATCH_SIZE", raising=False)
    _reload_indexer()  # leave the module at its default for other tests
