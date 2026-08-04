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
    cache_path = tmp_path / "embed_cache.db"

    runner = CliRunner()
    args = [
        "ingest",
        "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir),
        "--cache-path", str(cache_path),
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


def _write_two_act_corpus(corpus_dir):
    xml_dir = corpus_dir / "xml"
    xml_dir.mkdir()
    (xml_dir / "privacy-act-1988.xml").write_text(PRIVACY_ACT_XML)
    (xml_dir / "crimes-act-1914.xml").write_text(PRIVACY_ACT_XML)  # body reused, name differs
    index = {
        "acts": {
            "privacy-act-1988": {"name": "Privacy Act 1988", "xml_path": "xml/privacy-act-1988.xml"},
            "crimes-act-1914": {"name": "Crimes Act 1914", "xml_path": "xml/crimes-act-1914.xml"},
        }
    }
    (corpus_dir / "index.json").write_text(json.dumps(index))


def test_ingest_shard_only_indexes_its_own_slice(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_two_act_corpus(corpus_dir)
    storage_dir = tmp_path / "shard0_storage"
    cache_path = tmp_path / "shard0_cache.db"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest-shard",
        "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir),
        "--cache-path", str(cache_path),
        "--shard-index", "0",
        "--shard-size", "1",
    ])
    assert result.exit_code == 0, result.output
    # index.json's dict order is Privacy Act 1988 then Crimes Act 1914 (see
    # _write_two_act_corpus), so shard 0 of size 1 is Privacy Act 1988 only.
    assert "Privacy Act 1988" in result.output
    assert "Crimes Act 1914" not in result.output

    from qdrant_client import QdrantClient
    from lexausearch.indexer import COLLECTION_ACTS
    client = QdrantClient(path=str(storage_dir))
    points = client.scroll(collection_name=COLLECTION_ACTS, limit=10, with_payload=True)[0]
    act_names = {p.payload["act_name"] for p in points}
    assert act_names == {"Privacy Act 1988"}  # shard 0 of size 1 = first Act in index.json order


def test_ingest_shard_second_slice_gets_second_act(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_two_act_corpus(corpus_dir)
    storage_dir = tmp_path / "shard1_storage"
    cache_path = tmp_path / "shard1_cache.db"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest-shard",
        "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir),
        "--cache-path", str(cache_path),
        "--shard-index", "1",
        "--shard-size", "1",
    ])
    assert result.exit_code == 0, result.output

    from qdrant_client import QdrantClient
    from lexausearch.indexer import COLLECTION_ACTS
    client = QdrantClient(path=str(storage_dir))
    points = client.scroll(collection_name=COLLECTION_ACTS, limit=10, with_payload=True)[0]
    act_names = {p.payload["act_name"] for p in points}
    assert act_names == {"Crimes Act 1914"}  # shard 1 of size 1 = second Act in index.json order


def test_ingest_shard_out_of_range_index_is_a_no_op(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_two_act_corpus(corpus_dir)
    storage_dir = tmp_path / "shard99_storage"
    cache_path = tmp_path / "shard99_cache.db"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest-shard",
        "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir),
        "--cache-path", str(cache_path),
        "--shard-index", "99",
        "--shard-size", "1",
    ])
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower()
