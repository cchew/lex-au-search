import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import verify_index  # noqa: E402
from lexausearch.chunker import chunk_xml  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden_queries.json"
CORPUS = Path(__file__).resolve().parent.parent / "hf-corpus"


def test_small_chunk_drop_passes():
    assert verify_index.check_counts(3085, 530_000, previous_chunks=534_335, corpus_acts=3085) == []


def test_large_chunk_drop_fails_unless_allowed():
    fails = verify_index.check_counts(3085, 400_000, previous_chunks=534_335, corpus_acts=3085)
    assert fails and "below the previous generation" in fails[0]
    assert verify_index.check_counts(
        3085, 400_000, previous_chunks=534_335, corpus_acts=3085, allow_drop=True
    ) == []


def test_missing_acts_fail():
    assert verify_index.check_counts(3000, 540_000, previous_chunks=None, corpus_acts=3085)


class _FakeSearcher:
    """Stands in for Searcher: `present` is the set of chunk texts indexed,
    `ranked` the eIds returned for any query."""

    def __init__(self, ranked, present=True):
        self._ranked = ranked
        self._present = present
        self._client = self

    def search(self, query, limit=5, act=None, provision_type=None):
        return [
            SimpleNamespace(chunk=SimpleNamespace(eid=e, text="a Thing"))
            for e in self._ranked
        ]


def _q(**kw):
    base = {"name": "n", "query": "q", "act": "A", "expect_eid_prefix": "schedule-2", "must_contain": "thing"}
    return {**base, **kw}


@pytest.fixture(autouse=True)
def _stub_presence(monkeypatch):
    monkeypatch.setattr(verify_index, "_present", lambda client, act, prefix, phrase: client._present)


def test_golden_hit():
    fails, warns = verify_index.check_golden(_FakeSearcher(["schedule-2__x"]), [_q()])
    assert fails == [] and warns == []


def test_golden_missing_chunk_is_a_hard_failure():
    fails, _ = verify_index.check_golden(_FakeSearcher(["schedule-2__x"], present=False), [_q(known_gap=True)])
    assert fails and "missing from index" in fails[0]


def test_golden_rank_miss_fails():
    fails, warns = verify_index.check_golden(_FakeSearcher(["sec-1"] * 6 + ["schedule-2__x"]), [_q()])
    assert fails and not warns


def test_golden_known_gap_rank_miss_only_warns():
    fails, warns = verify_index.check_golden(
        _FakeSearcher(["sec-1"] * 6 + ["schedule-2__x"]), [_q(known_gap=True)]
    )
    assert not fails and warns


def test_golden_per_query_max_rank_is_honoured():
    fails, _ = verify_index.check_golden(
        _FakeSearcher(["sec-1"] * 9 + ["schedule-2__x"]), [_q(max_rank=15)]
    )
    assert fails == []


@pytest.mark.skipif(not (CORPUS / "index.json").exists(), reason="hf-corpus not restored")
def test_every_golden_expectation_exists_in_the_chunked_corpus():
    acts = json.loads((CORPUS / "index.json").read_text())["acts"]
    for q in json.loads(GOLDEN.read_text()):
        entry = next(a for a in acts.values() if a["name"] == q["act"])
        chunks = chunk_xml(CORPUS / entry["xml_path"], entry["name"])
        assert any(
            c.eid.startswith(q["expect_eid_prefix"])
            and q["must_contain"].lower() in c.text.lower()
            for c in chunks
        ), q["name"]
