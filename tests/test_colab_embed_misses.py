import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from colab_embed_misses import unique_misses_length_bucketed


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
