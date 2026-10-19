"""Contract tests: the chunker must handle every structural shape in lex-au's golden set.

Fixtures come from the pinned `lexau-contract-fixtures` package (see pyproject dev extras).
A lex-au shape change fails here before a rebuild, not a month after (the 2026-09 incident).
"""
import logging
import os

import pytest

try:
    import lexau_contract_fixtures as fx
except ImportError:  # pragma: no cover
    if os.environ.get("CI"):
        raise
    pytest.skip("lexau-contract-fixtures not installed", allow_module_level=True)

from lexausearch.chunker import (
    SUPPORTED_STRUCTURE_VERSIONS,
    chunk_corpus,
    chunk_xml,
)
from tests.test_chunker_schedule_coverage import MAX_CHUNK_WORDS, _own_text_uncovered

SLUGS = sorted(fx.manifest()["fixtures"])


def _path(slug):
    return fx.corpus_dir() / "xml" / f"{slug}.xml"


def test_chunker_supports_the_fixture_structure_version():
    assert fx.structure_version() in SUPPORTED_STRUCTURE_VERSIONS


@pytest.mark.parametrize("slug", SLUGS)
def test_all_attachment_own_text_is_covered(slug):
    assert _own_text_uncovered(_path(slug)) == []


def test_corpus_chunks_without_gap_warnings(caplog):
    with caplog.at_level(logging.WARNING, logger="lexausearch.chunker"):
        chunks = chunk_corpus(fx.corpus_dir())
    assert chunks
    assert not [r.message for r in caplog.records if "coverage gap" in r.message]


def test_no_chunk_has_an_empty_eid_or_empty_text():
    for c in chunk_corpus(fx.corpus_dir()):
        assert c.eid, c.act_name
        assert c.text.strip(), c.eid


def test_schedule_chunks_respect_the_size_cap():
    for c in chunk_corpus(fx.corpus_dir()):
        if c.provision_type in ("schedule_clause", "schedule_text"):
            assert len(c.text.split()) <= MAX_CHUNK_WORDS * 1.5, c.eid


@pytest.mark.parametrize("slug", SLUGS)
def test_every_fixture_with_schedule_content_yields_schedule_chunks(slug):
    facts = fx.manifest()["fixtures"][slug]["facts"]
    chunks = chunk_xml(_path(slug), slug)
    if facts["attachment_words"] > 0:
        assert any(c.provision_type.startswith("schedule") for c in chunks)


@pytest.mark.parametrize("shape", ["nested_clause", "quoted_amendment", "oversized_clause"])
def test_nested_shapes_are_chunked_as_schedule_content(shape):
    for slug in fx.fixtures_with(shape):
        chunks = chunk_xml(_path(slug), slug)
        assert any(c.provision_type.startswith("schedule") for c in chunks), slug
