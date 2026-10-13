import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from colab_embed_misses import embed_batch_with_oom_backoff, unique_misses_length_bucketed
from lexausearch.cache import EmbedCache


class _FakeClient:
    """Stub standing in for the real QdrantClient's _embed_documents -- lets
    tests drive the exact failure sequence (which calls OOM, which succeed)
    without a real onnxruntime session."""

    embedding_model_name = "fake-model"

    def __init__(self, oom_on_sizes: set[int] = frozenset()):
        self.oom_on_sizes = oom_on_sizes
        self.calls: list[list[str]] = []

    def _embed_documents(self, batch, embedding_model_name, embed_type):
        self.calls.append(list(batch))
        if len(batch) in self.oom_on_sizes:
            raise RuntimeError(
                "Fail : 1 : FAIL : ... Failed to allocate memory for requested buffer of size 444735488"
            )
        return [(text, [float(len(text))]) for text in batch]


def test_dedupes_repeated_texts():
    all_texts = ["short", "long text here", "short", "short"]
    result = unique_misses_length_bucketed(all_texts, hits=set())
    assert sorted(result) == sorted({"short", "long text here"})


def test_excludes_cache_hits():
    all_texts = ["a hit", "a miss"]
    hits = {"a hit"}
    result = unique_misses_length_bucketed(all_texts, hits)
    assert result == ["a miss"]


def test_sorts_by_length_ascending():
    all_texts = ["a longer chunk of text", "short", "medium length text"]
    result = unique_misses_length_bucketed(all_texts, hits=set())
    assert result == sorted(all_texts, key=len)
    assert [len(t) for t in result] == sorted(len(t) for t in all_texts)


def test_empty_input_returns_empty():
    assert unique_misses_length_bucketed([], hits=set()) == []


def test_everything_a_hit_returns_empty():
    all_texts = ["a", "b"]
    hits = {"a", "b"}
    assert unique_misses_length_bucketed(all_texts, hits) == []


def test_oom_backoff_halves_the_batch_and_caches_every_text(tmp_path):
    """Real failure 2026-10-13: a batch of 32 long texts OOM'd onnxruntime's
    BFCArena 83% through a run. Splitting in half (and recursing on each
    half) must eventually succeed and cache every text, not just some."""
    client = _FakeClient(oom_on_sizes={4})
    cache = EmbedCache(tmp_path / "cache.db")
    batch = ["t1", "t2", "t3", "t4"]

    embed_batch_with_oom_backoff(client, cache, batch)

    assert cache.contains_batch(batch) == set(batch)
    # one failed attempt at the full batch of 4, then two successful halves of 2
    sizes = sorted(len(c) for c in client.calls)
    assert sizes == [2, 2, 4]


def test_oom_backoff_floor_of_one_still_raises_a_genuine_failure(tmp_path):
    """A single text that OOMs has nothing left to halve - the failure must
    propagate, not be swallowed, so a truly pathological text is surfaced
    rather than silently dropped from the cache."""
    client = _FakeClient(oom_on_sizes={1})
    cache = EmbedCache(tmp_path / "cache.db")

    with pytest.raises(RuntimeError, match="Failed to allocate memory"):
        embed_batch_with_oom_backoff(client, cache, ["one bad text"])


def test_oom_backoff_reraises_non_oom_errors_without_retrying(tmp_path):
    """A failure unrelated to memory pressure (bad input, corrupt state, etc.)
    must not be silently retried as if it were an OOM - that would mask a
    real bug behind a misleading 'splitting and retrying' log line."""

    class _BrokenClient(_FakeClient):
        def _embed_documents(self, batch, embedding_model_name, embed_type):
            raise ValueError("completely unrelated failure")

    client = _BrokenClient()
    cache = EmbedCache(tmp_path / "cache.db")

    with pytest.raises(ValueError, match="completely unrelated failure"):
        embed_batch_with_oom_backoff(client, cache, ["a", "b"])
