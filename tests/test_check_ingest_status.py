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
