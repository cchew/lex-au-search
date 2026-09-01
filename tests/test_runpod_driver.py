import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import runpod_driver as rd


def test_build_create_payload_has_spike_defaults():
    cfg = rd.CreatePodConfig(name="lexau-ingest-x")
    p = rd.build_create_payload(cfg)
    assert p["gpuTypeIds"] == ["NVIDIA RTX A6000"]
    assert p["cloudType"] == "COMMUNITY"
    assert p["computeType"] == "GPU"
    assert p["gpuCount"] == 1
    assert p["interruptible"] is False
    assert p["containerDiskInGb"] == 80
    assert p["volumeInGb"] == 0
    assert p["ports"] == ["22/tcp"]
    assert p["supportPublicIp"] is True
    assert p["allowedCudaVersions"] == ["12.4", "12.5", "12.6", "12.7", "12.8"]
    assert p["imageName"].startswith("runpod/pytorch:2.4.0-py3.11-cuda12.4.1")
    assert p["name"] == "lexau-ingest-x"


def test_build_create_payload_honours_cloud_and_gpu_override():
    cfg = rd.CreatePodConfig(name="n", cloud_type="SECURE", gpu_type_id="NVIDIA RTX A5000")
    p = rd.build_create_payload(cfg)
    assert p["cloudType"] == "SECURE"
    assert p["gpuTypeIds"] == ["NVIDIA RTX A5000"]


def test_pod_name_format():
    from datetime import datetime, timezone
    n = rd.pod_name(datetime(2026, 9, 1, 3, 4, 5, tzinfo=timezone.utc))
    assert n == "lexau-ingest-20260901T030405Z"


def test_api_key_exits_when_absent(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(rd, "load_env", lambda *_: None)  # no .env fallback
    with pytest.raises(SystemExit) as e:
        rd.api_key()
    assert e.value.code == 2
    assert "RUNPOD_API_KEY" in capsys.readouterr().err


def test_api_key_returns_from_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_test")
    assert rd.api_key() == "rpa_test"
