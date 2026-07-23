import json

from click.testing import CliRunner

from lexausearch.cli import cli
from tests.conftest import PRIVACY_ACT_XML


def _write_corpus(corpus_dir):
    xml_dir = corpus_dir / "xml"
    xml_dir.mkdir()
    (xml_dir / "privacy-act-1988.xml").write_text(PRIVACY_ACT_XML)
    index = {
        "acts": {
            "privacy-act-1988": {
                "name": "Privacy Act 1988",
                "xml_path": "xml/privacy-act-1988.xml",
            }
        }
    }
    (corpus_dir / "index.json").write_text(json.dumps(index))


def test_ingest_second_run_reports_cache_hits(tmp_path):
    """Re-ingesting an unchanged corpus should skip re-embedding via the persistent cache."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir)
    storage_dir = tmp_path / "qdrant_storage"
    cache_dir = tmp_path / "embed_cache_storage"

    runner = CliRunner()
    args = [
        "ingest",
        "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir),
        "--cache-dir", str(cache_dir),
    ]

    first = runner.invoke(cli, args)
    assert first.exit_code == 0, first.output
    assert "0 hits" in first.output

    # Cache dir must survive independently of storage-dir being wiped, as
    # colab_ingest.sh does before every run.
    import shutil
    shutil.rmtree(storage_dir)

    second = runner.invoke(cli, args)
    assert second.exit_code == 0, second.output
    assert "0 misses" in second.output
