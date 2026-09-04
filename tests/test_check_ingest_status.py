import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_ingest_status.py"


def test_probe_quota_is_noop_for_runpod(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--probe-quota", "--backend", "runpod",
         "--total-acts", "3076", "--shards-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    assert "not applicable for the runpod backend" in (r.stdout + r.stderr).lower()
    assert "colab" not in r.stderr.lower()  # never touched colab_driver


def test_hf_section_degrades_when_unreachable(tmp_path, monkeypatch):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--total-acts", "3076", "--shards-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert r.returncode == 0
    assert "hf cache" in (r.stdout + r.stderr).lower()
    assert "unreachable" in (r.stdout + r.stderr).lower()


def test_no_hf_flag_skips_section(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--total-acts", "3076", "--shards-dir", str(tmp_path), "--no-hf"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    assert "hf cache" not in r.stdout.lower()


def test_hf_section_reports_updated_at_and_local_stale(tmp_path, monkeypatch, capsys):
    """Spec 6.2 columns: shard | generation | row_count | model | status |
    updated_at | local_stale?"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import check_ingest_status as cis
    from lexausearch import hf_cache

    monkeypatch.setattr(hf_cache, "read_catalogue",
                        lambda token: hf_cache.Catalogue("m", 300, 600, {}))
    metas = {
        0: hf_cache.ShardCacheMeta("m", 12, 5, "2026-09-03T01:02:03Z", "x", "partial"),
        1: None,
    }
    monkeypatch.setattr(hf_cache, "_read_sidecar", lambda i, t: metas[i])
    # shard 0's local marker is behind HF -> stale
    monkeypatch.setattr(hf_cache, "_local_marker_generation", lambda d, i: 3)

    cis.print_hf_section(tmp_path)
    out = capsys.readouterr().out
    assert "updated_at" in out and "local_stale?" in out
    assert "2026-09-03T01:02:03Z" in out
    # columns: shard gen rows status updated_at local_stale? model
    row0 = next(l for l in out.splitlines() if l.split()[:1] == ["0"])
    assert row0.split() == ["0", "5", "12", "partial", "2026-09-03T01:02:03Z", "yes", "m"]


def test_hf_section_marks_current_local_marker_not_stale(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import check_ingest_status as cis
    from lexausearch import hf_cache

    monkeypatch.setattr(hf_cache, "read_catalogue",
                        lambda token: hf_cache.Catalogue("m", 300, 300, {}))
    monkeypatch.setattr(hf_cache, "_read_sidecar",
                        lambda i, t: hf_cache.ShardCacheMeta("m", 12, 5, "t", "x", "complete"))
    monkeypatch.setattr(hf_cache, "_local_marker_generation", lambda d, i: 5)
    cis.print_hf_section(tmp_path)
    out = capsys.readouterr().out
    row0 = next(l for l in out.splitlines() if l.split()[:1] == ["0"])
    assert row0.split()[5] == "no"
