import json
import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lexausearch.cli import cli
from report_index_counts import report
from tests.conftest import PRIVACY_ACT_XML


def test_report_counts_match_a_real_ingested_storage_dir(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    xml_dir = corpus_dir / "xml"
    xml_dir.mkdir()
    (xml_dir / "privacy-act-1988.xml").write_text(PRIVACY_ACT_XML)
    (corpus_dir / "index.json").write_text(json.dumps({
        "acts": {"privacy-act-1988": {"name": "Privacy Act 1988", "xml_path": "xml/privacy-act-1988.xml"}}
    }))
    storage_dir = tmp_path / "storage"

    result = CliRunner().invoke(cli, [
        "ingest", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(tmp_path / "cache.db"),
    ])
    assert result.exit_code == 0, result.output

    act_count, chunk_count = report(str(storage_dir))

    assert act_count == 1
    assert chunk_count > 0
