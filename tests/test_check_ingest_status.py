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
