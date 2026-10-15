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
    def __init__(self, eids_by_query):
        self._eids = eids_by_query

    def search(self, query, limit=5, act=None, provision_type=None):
        eid, text = self._eids[query]
        return [SimpleNamespace(chunk=SimpleNamespace(eid=eid, text=text))]


def test_golden_hit_and_miss():
    q = [{"name": "n", "query": "q", "act": "A", "expect_eid_prefix": "schedule-2", "must_contain": "thing"}]
    assert verify_index.check_golden(_FakeSearcher({"q": ("schedule-2__x", "a Thing")}), q) == []
    assert verify_index.check_golden(_FakeSearcher({"q": ("sec-5", "a Thing")}), q)
    assert verify_index.check_golden(_FakeSearcher({"q": ("schedule-2__x", "other")}), q)


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
