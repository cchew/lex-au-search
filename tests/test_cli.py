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


def test_ingest_writes_content_hash_to_act_record(tmp_path):
    import hashlib
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir)
    storage_dir = tmp_path / "qdrant_storage"
    cache_path = tmp_path / "embed_cache.db"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest",
        "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir),
        "--cache-path", str(cache_path),
    ])
    assert result.exit_code == 0, result.output

    from qdrant_client import QdrantClient
    from lexausearch.indexer import COLLECTION_ACTS
    client = QdrantClient(path=str(storage_dir))
    points = client.scroll(collection_name=COLLECTION_ACTS, limit=10, with_payload=True)[0]
    expected_hash = hashlib.sha256(PRIVACY_ACT_XML.encode()).hexdigest()
    assert any(p.payload.get("content_hash") == expected_hash for p in points)


def test_ingest_delta_reindexes_only_changed_act(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_two_act_corpus(corpus_dir)
    storage_dir = tmp_path / "storage"
    cache_path = tmp_path / "cache.db"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])
    assert result.exit_code == 0, result.output

    from qdrant_client import QdrantClient, models
    from lexausearch.indexer import COLLECTION_SECTIONS

    def _section_ids_for(act_name):
        client = QdrantClient(path=str(storage_dir))
        points = client.scroll(
            collection_name=COLLECTION_SECTIONS, limit=100, with_payload=True,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="act_name", match=models.MatchValue(value=act_name))
            ]),
        )[0]
        return {p.id for p in points}

    crimes_ids_before = _section_ids_for("Crimes Act 1914")
    privacy_ids_before = _section_ids_for("Privacy Act 1988")
    assert crimes_ids_before and privacy_ids_before  # sanity: both actually indexed

    # Mutate only Privacy Act's XML -- changes its hash. Crimes Act is untouched.
    xml_path = corpus_dir / "xml" / "privacy-act-1988.xml"
    xml_path.write_text(xml_path.read_text() + "\n<!-- amended -->")

    result = runner.invoke(cli, [
        "ingest-delta", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])
    assert result.exit_code == 0, result.output
    assert "1 Act(s) changed or new, 1 unchanged" in result.output

    crimes_ids_after = _section_ids_for("Crimes Act 1914")
    privacy_ids_after = _section_ids_for("Privacy Act 1988")

    # Untouched Act: identical point IDs -- genuinely skipped, not re-upserted.
    assert crimes_ids_after == crimes_ids_before
    # Changed Act: old points are gone, replaced by new ones (delete_act ran).
    assert privacy_ids_before.isdisjoint(privacy_ids_after)
    assert privacy_ids_after  # new points do exist


def test_ingest_delta_nothing_changed_is_a_noop(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir)
    storage_dir = tmp_path / "storage"
    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    runner.invoke(cli, [
        "ingest", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])

    result = runner.invoke(cli, [
        "ingest-delta", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])
    assert result.exit_code == 0, result.output
    assert "Nothing to do" in result.output


def test_ingest_delta_picks_up_a_brand_new_act(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir)
    storage_dir = tmp_path / "storage"
    cache_path = tmp_path / "cache.db"
    runner = CliRunner()
    runner.invoke(cli, [
        "ingest", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])

    # Add a second Act to the corpus that was never ingested.
    (corpus_dir / "xml" / "crimes-act-1914.xml").write_text(PRIVACY_ACT_XML)
    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text())
    index["acts"]["crimes-act-1914"] = {"name": "Crimes Act 1914", "xml_path": "xml/crimes-act-1914.xml"}
    index_path.write_text(json.dumps(index))

    result = runner.invoke(cli, [
        "ingest-delta", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])
    assert result.exit_code == 0, result.output
    assert "1 Act(s) changed or new, 1 unchanged" in result.output

    from qdrant_client import QdrantClient
    from lexausearch.indexer import COLLECTION_ACTS
    client = QdrantClient(path=str(storage_dir))
    points = client.scroll(collection_name=COLLECTION_ACTS, limit=10, with_payload=True)[0]
    assert {p.payload["act_name"] for p in points} == {"Privacy Act 1988", "Crimes Act 1914"}


def test_ingest_delta_errors_when_storage_dir_not_yet_ingested(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir)
    storage_dir = tmp_path / "never_ingested"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest-delta", "--corpus-dir", str(corpus_dir), "--storage-dir", str(storage_dir),
    ])
    assert result.exit_code != 0
    assert "ingest" in result.output.lower()


_ZERO_CHUNK_AKN_XML = """\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRExpression>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01"/>
        </FRBRExpression>
      </identification>
    </meta>
    <body>
    </body>
  </act>
</akomaNtoso>"""


def _ids_for_act(storage_dir, collection, act_name):
    from qdrant_client import QdrantClient, models
    client = QdrantClient(path=str(storage_dir))
    points = client.scroll(
        collection_name=collection, limit=100, with_payload=True,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="act_name", match=models.MatchValue(value=act_name))
        ]),
    )[0]
    return {p.id for p in points}


def test_ingest_delta_skips_zero_chunk_act_without_deleting_index_entry(tmp_path):
    """A corpus regression that makes an Act stop producing chunks must not
    delete that Act's existing index entry -- stale-but-present beats missing."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_two_act_corpus(corpus_dir)
    storage_dir = tmp_path / "storage"
    cache_path = tmp_path / "cache.db"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])
    assert result.exit_code == 0, result.output

    from lexausearch.indexer import COLLECTION_SECTIONS, COLLECTION_ACTS

    section_ids_before = _ids_for_act(storage_dir, COLLECTION_SECTIONS, "Privacy Act 1988")
    act_ids_before = _ids_for_act(storage_dir, COLLECTION_ACTS, "Privacy Act 1988")
    assert section_ids_before and act_ids_before  # sanity: actually indexed

    # Replace Privacy Act's XML with a valid-but-empty AKN doc: different
    # content hash (so it's picked up as "changed"), but zero extractable
    # chunks -- simulates a corpus/parser regression.
    xml_path = corpus_dir / "xml" / "privacy-act-1988.xml"
    xml_path.write_text(_ZERO_CHUNK_AKN_XML)

    result = runner.invoke(cli, [
        "ingest-delta", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])
    assert result.exit_code != 0, result.output
    assert "zero chunks" in result.output.lower()
    assert "privacy act 1988" in result.output.lower()

    # Old points for the zero-chunk Act must still be present -- not deleted.
    assert _ids_for_act(storage_dir, COLLECTION_SECTIONS, "Privacy Act 1988") == section_ids_before
    assert _ids_for_act(storage_dir, COLLECTION_ACTS, "Privacy Act 1988") == act_ids_before


def test_ingest_delta_isolates_per_act_failure(tmp_path):
    """A malformed Act's XML must not abort the run for the rest of the batch."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_two_act_corpus(corpus_dir)
    storage_dir = tmp_path / "storage"
    cache_path = tmp_path / "cache.db"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])
    assert result.exit_code == 0, result.output

    # Crimes Act's XML becomes malformed -- chunk_xml/etree.parse will raise.
    (corpus_dir / "xml" / "crimes-act-1914.xml").write_text("<not><valid xml")
    # Privacy Act gets a genuine (parseable) amendment -- should still succeed.
    privacy_xml_path = corpus_dir / "xml" / "privacy-act-1988.xml"
    privacy_xml_path.write_text(privacy_xml_path.read_text() + "\n<!-- amended -->")

    result = runner.invoke(cli, [
        "ingest-delta", "--corpus-dir", str(corpus_dir),
        "--storage-dir", str(storage_dir), "--cache-path", str(cache_path),
    ])
    assert result.exit_code != 0, result.output
    assert "crimes act 1914" in result.output.lower()

    from qdrant_client import QdrantClient
    from lexausearch.indexer import COLLECTION_ACTS
    client = QdrantClient(path=str(storage_dir))
    points = client.scroll(collection_name=COLLECTION_ACTS, limit=10, with_payload=True)[0]
    act_hashes = {p.payload["act_name"]: p.payload["content_hash"] for p in points}

    # Both Acts remain indexed -- the failure isolated to Crimes Act, and
    # Privacy Act's genuinely-changed content still made it through.
    assert set(act_hashes) == {"Privacy Act 1988", "Crimes Act 1914"}
    import hashlib
    expected_privacy_hash = hashlib.sha256(privacy_xml_path.read_bytes()).hexdigest()
    assert act_hashes["Privacy Act 1988"] == expected_privacy_hash


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


def test_merge_shards_combines_two_local_shard_storages(tmp_path):
    # Build two tiny shard storages the way ingest-shard would.
    shard0_corpus = tmp_path / "corpus0"
    shard0_corpus.mkdir()
    xml_dir0 = shard0_corpus / "xml"
    xml_dir0.mkdir()
    (xml_dir0 / "privacy-act-1988.xml").write_text(PRIVACY_ACT_XML)
    (shard0_corpus / "index.json").write_text(json.dumps({
        "acts": {"privacy-act-1988": {"name": "Privacy Act 1988", "xml_path": "xml/privacy-act-1988.xml"}}
    }))
    shard0_storage = tmp_path / "shard0_storage"
    shard0_cache = tmp_path / "shard0_cache.db"

    shard1_corpus = tmp_path / "corpus1"
    shard1_corpus.mkdir()
    xml_dir1 = shard1_corpus / "xml"
    xml_dir1.mkdir()
    (xml_dir1 / "crimes-act-1914.xml").write_text(PRIVACY_ACT_XML)
    (shard1_corpus / "index.json").write_text(json.dumps({
        "acts": {"crimes-act-1914": {"name": "Crimes Act 1914", "xml_path": "xml/crimes-act-1914.xml"}}
    }))
    shard1_storage = tmp_path / "shard1_storage"
    shard1_cache = tmp_path / "shard1_cache.db"

    runner = CliRunner()
    for corpus_dir, storage_dir, cache_path in [
        (shard0_corpus, shard0_storage, shard0_cache),
        (shard1_corpus, shard1_storage, shard1_cache),
    ]:
        result = runner.invoke(cli, [
            "ingest-shard",
            "--corpus-dir", str(corpus_dir),
            "--storage-dir", str(storage_dir),
            "--cache-path", str(cache_path),
            "--shard-index", "0",
            "--shard-size", "10",
        ])
        assert result.exit_code == 0, result.output

    output_storage = tmp_path / "merged_storage"
    output_cache = tmp_path / "merged_cache.db"
    result = runner.invoke(cli, [
        "merge-shards",
        "--shard-storage-dirs", f"{shard0_storage},{shard1_storage}",
        "--shard-cache-paths", f"{shard0_cache},{shard1_cache}",
        "--output-storage-dir", str(output_storage),
        "--output-cache-path", str(output_cache),
    ])
    assert result.exit_code == 0, result.output

    from qdrant_client import QdrantClient
    from lexausearch.indexer import COLLECTION_ACTS
    merged_client = QdrantClient(path=str(output_storage))
    points = merged_client.scroll(collection_name=COLLECTION_ACTS, limit=10, with_payload=True)[0]
    act_names = {p.payload["act_name"] for p in points}
    assert act_names == {"Privacy Act 1988", "Crimes Act 1914"}


def test_merge_shards_requires_exactly_one_cache_source(tmp_path):
    from click.testing import CliRunner
    from lexausearch.cli import cli
    r = CliRunner().invoke(cli, ["merge-shards",
        "--shard-storage-dirs", str(tmp_path / "s0"),
        "--output-storage-dir", str(tmp_path / "out"),
        "--output-cache-path", str(tmp_path / "out.db")])
    assert r.exit_code != 0
    assert "exactly one of --from-hf" in r.output.lower()


def test_merge_shards_mixed_model_guard_refuses(tmp_path, monkeypatch):
    # two local caches, each with a sidecar naming a different model
    import json, sqlite3
    from click.testing import CliRunner
    from lexausearch.cli import cli
    for i, model in [(0, "model-a"), (1, "model-b")]:
        db = tmp_path / f"shard_{i:03d}_checkpoint_cache.db"
        c = sqlite3.connect(str(db)); c.execute("CREATE TABLE embed_cache (id TEXT PRIMARY KEY, vector BLOB NOT NULL)"); c.commit(); c.close()
        (tmp_path / f"shard_{i:03d}.json").write_text(json.dumps({"model_name": model,
            "row_count": 0, "generation": 1, "updated_at": "t", "sha256": "x", "status": "partial"}))
    (tmp_path / "s0").mkdir(); (tmp_path / "s1").mkdir()
    r = CliRunner().invoke(cli, ["merge-shards",
        "--shard-storage-dirs", f"{tmp_path/'s0'},{tmp_path/'s1'}",
        "--shard-cache-paths", f"{tmp_path/'shard_000_checkpoint_cache.db'},{tmp_path/'shard_001_checkpoint_cache.db'}",
        "--output-storage-dir", str(tmp_path / "out"),
        "--output-cache-path", str(tmp_path / "out.db")])
    assert r.exit_code != 0
    assert "refusing to merge" in r.output.lower()


def test_merge_shards_from_hf_cleans_up_its_temp_dir(tmp_path, monkeypatch):
    """--from-hf downloads every shard cache into a mkdtemp() dir; for the
    11-shard production merge that is ~9GB that must not be abandoned in /tmp."""
    import sqlite3
    import tempfile as _tempfile
    from pathlib import Path as _Path
    from click.testing import CliRunner
    from lexausearch import cli as cli_mod, hf_cache
    from lexausearch.cli import cli

    monkeypatch.setattr(cli_mod, "QdrantClient", lambda path: object())
    monkeypatch.setattr(cli_mod, "merge_shard_clients",
                        lambda shards, out: {"sections": 0, "acts": 0})
    monkeypatch.setattr(hf_cache, "read_catalogue",
                        lambda token: hf_cache.Catalogue("m", 300, 300, {}))

    made: list[_Path] = []
    real_mkdtemp = _tempfile.mkdtemp

    def _spy_mkdtemp(*a, **k):
        d = real_mkdtemp(*a, **k)
        made.append(_Path(d))
        return d

    monkeypatch.setattr(_tempfile, "mkdtemp", _spy_mkdtemp)

    def _fetch(i, dest, token=None, seed_as=None, expect_model=None):
        db = _Path(dest) / seed_as
        c = sqlite3.connect(str(db))
        c.execute("CREATE TABLE embed_cache (id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        c.commit()
        c.close()
        return hf_cache.ShardCacheMeta("m", 0, 1, "t", "x", "complete")

    monkeypatch.setattr(hf_cache, "fetch_shard_cache", _fetch)

    r = CliRunner().invoke(cli, [
        "merge-shards", "--from-hf",
        "--shard-storage-dirs", str(tmp_path / "s0"),
        "--output-storage-dir", str(tmp_path / "out"),
        "--output-cache-path", str(tmp_path / "out.db"),
    ])
    assert r.exit_code == 0, r.output
    assert made, "expected --from-hf to allocate a temp dir"
    assert not made[0].exists(), f"temp dir leaked: {made[0]}"


def test_merge_shards_from_hf_cleans_up_temp_dir_on_failure(tmp_path, monkeypatch):
    import tempfile as _tempfile
    from pathlib import Path as _Path
    from click.testing import CliRunner
    from lexausearch import cli as cli_mod, hf_cache
    from lexausearch.cli import cli

    monkeypatch.setattr(hf_cache, "read_catalogue",
                        lambda token: hf_cache.Catalogue("m", 300, 300, {}))
    made: list[_Path] = []
    real_mkdtemp = _tempfile.mkdtemp
    monkeypatch.setattr(_tempfile, "mkdtemp",
                        lambda *a, **k: made.append(_Path(real_mkdtemp(*a, **k))) or str(made[-1]))
    monkeypatch.setattr(hf_cache, "fetch_shard_cache", lambda *a, **k: None)  # shard absent

    r = CliRunner().invoke(cli, [
        "merge-shards", "--from-hf",
        "--shard-storage-dirs", str(tmp_path / "s0"),
        "--output-storage-dir", str(tmp_path / "out"),
        "--output-cache-path", str(tmp_path / "out.db"),
    ])
    assert r.exit_code != 0
    assert made and not made[0].exists(), "temp dir leaked on the error path"
