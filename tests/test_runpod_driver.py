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


# --- Task 4: REST functions + CLI ------------------------------------------
import io
import json
import types as _types


class _Resp:
    """Minimal urlopen-response stand-in (BytesIO has no __dict__)."""

    def __init__(self, status, body):
        self.status = status
        self._b = json.dumps(body).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeHTTP:
    """Scriptable stand-in for urllib.request.urlopen."""
    def __init__(self, responses):
        self._responses = list(responses)  # each: (status, body_dict) or Exception
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append((req.get_method(), req.full_url))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return _Resp(status, body)


def _patch_http(monkeypatch, fake):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_test")
    monkeypatch.setattr(rd.urllib.request, "urlopen", fake)


def test_dry_run_prints_payload_and_makes_no_call(monkeypatch, capsys):
    called = _FakeHTTP([])
    _patch_http(monkeypatch, called)
    out = rd.create_pod(rd.CreatePodConfig(name="n"), dry_run=True)
    assert out == {}
    assert called.calls == []
    printed = json.loads(capsys.readouterr().out)
    assert printed["gpuTypeIds"] == ["NVIDIA RTX A6000"]


def test_create_pod_falls_back_to_secure_on_no_instances(monkeypatch, capsys):
    import urllib.error
    err = urllib.error.HTTPError(
        rd.API_BASE + "/pods", 500, "err", {},
        io.BytesIO(json.dumps({"error": "create pod: There are no instances currently available"}).encode()),
    )
    ok = (201, {"id": "pod_secure", "desiredStatus": "RUNNING"})
    fake = _FakeHTTP([err, ok])
    _patch_http(monkeypatch, fake)
    pod = rd.create_pod(rd.CreatePodConfig(name="n", cloud_type="COMMUNITY"))
    assert pod["id"] == "pod_secure"
    assert "SECURE" in capsys.readouterr().err
    assert len(fake.calls) == 2


def test_ssh_coords_none_until_ip_and_port_present(monkeypatch):
    fake = _FakeHTTP([
        (200, {"publicIp": "", "portMappings": {}}),
        (200, {"publicIp": "1.2.3.4", "portMappings": {}}),
        (200, {"publicIp": "1.2.3.4", "portMappings": {"22": 20273}}),
    ])
    _patch_http(monkeypatch, fake)
    assert rd.ssh_coords("p") is None
    assert rd.ssh_coords("p") is None
    assert rd.ssh_coords("p") == ("1.2.3.4", 20273)


def test_wait_ready_times_out_with_injected_clock(monkeypatch):
    import pytest
    fake = _FakeHTTP([(200, {"publicIp": "", "portMappings": {}})] * 100)
    _patch_http(monkeypatch, fake)
    t = [0.0]
    with pytest.raises(TimeoutError):
        rd.wait_ready("p", timeout_s=30, sleep=lambda s: t.__setitem__(0, t[0] + s), now=lambda: t[0])


def test_get_status_reports_terminated_on_404(monkeypatch):
    import urllib.error
    err = urllib.error.HTTPError(rd.API_BASE + "/pods/p", 404, "gone", {}, io.BytesIO(b"{}"))
    _patch_http(monkeypatch, _FakeHTTP([err]))
    assert rd.get_status("p") == "TERMINATED"


def test_terminate_pod_confirms_deletion(monkeypatch):
    import urllib.error
    del_ok = (204, {})
    gone = urllib.error.HTTPError(rd.API_BASE + "/pods/p", 404, "gone", {}, io.BytesIO(b"{}"))
    _patch_http(monkeypatch, _FakeHTTP([del_ok, gone]))
    assert rd.terminate_pod("p", sleep=lambda _: None) is True


def test_terminate_pod_prints_manual_command_when_unconfirmed(monkeypatch, capsys):
    still = (200, {"id": "p", "desiredStatus": "RUNNING"})
    _patch_http(monkeypatch, _FakeHTTP([(204, {}), still, still, still]))
    assert rd.terminate_pod("p", sleep=lambda _: None) is False
    assert "MANUAL ACTION" in capsys.readouterr().out
