import json
import logging

import pytest

from lexausearch.chunker import (
    UnsupportedCorpusStructure,
    check_structure_version,
    chunk_corpus,
    chunk_corpus_for_acts,
)


def _corpus(tmp_path, **extra):
    (tmp_path / "xml").mkdir()
    index = {"acts": {}, "updated_at": None, **extra}
    (tmp_path / "index.json").write_text(json.dumps(index))
    return tmp_path


@pytest.mark.parametrize("version", [1, 2])
def test_supported_versions_pass(version):
    check_structure_version({"structure_version": version})


def test_unknown_version_is_refused():
    with pytest.raises(UnsupportedCorpusStructure, match="structure_version 3"):
        check_structure_version({"structure_version": 3})


def test_missing_version_warns_but_proceeds(caplog):
    with caplog.at_level(logging.WARNING, logger="lexausearch.chunker"):
        check_structure_version({"acts": {}})
    assert any("no structure_version" in r.message for r in caplog.records)


def test_chunk_corpus_refuses_an_unknown_structure(tmp_path):
    with pytest.raises(UnsupportedCorpusStructure):
        chunk_corpus(_corpus(tmp_path, structure_version=99))


def test_chunk_corpus_for_acts_refuses_an_unknown_structure(tmp_path):
    with pytest.raises(UnsupportedCorpusStructure):
        chunk_corpus_for_acts(_corpus(tmp_path, structure_version=99), set())


def test_chunk_corpus_accepts_the_current_structure(tmp_path):
    assert chunk_corpus(_corpus(tmp_path, structure_version=2)) == []
