# tests/test_run_sharded_ingest.py
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backends.base import shard_paths
from backends.fake import FakeBackend
import run_sharded_ingest as rsi


@pytest.fixture(autouse=True)
def _offline_hf_cache(monkeypatch):
    """run_all now runs a laptop-side HF model pre-flight and mirrors the shard
    cache local after every shard. Keep the orchestration unit tests offline;
    individual tests override check_model to exercise the mismatch paths."""
    monkeypatch.setattr(rsi.hf_cache, "check_model", lambda *a, **k: rsi.hf_cache.Ok())
    monkeypatch.setattr(rsi.hf_cache, "mirror_to_local", lambda *a, **k: None)


def test_prepare_once_teardown_once_on_happy_path(tmp_path):
    be = FakeBackend()
    rsi.run_all(be, [0, 1, 2], 300, tmp_path)
    assert be.prepare_calls == 1
    assert be.teardown_calls == 1
    assert [c[0] for c in be.run_shard_calls] == [0, 1, 2]


def test_teardown_runs_when_a_shard_raises(tmp_path):
    be = FakeBackend(raise_on_index=1)
    with pytest.raises(RuntimeError):
        rsi.run_all(be, [0, 1, 2], 300, tmp_path)
    assert be.teardown_calls == 1
    assert [c[0] for c in be.run_shard_calls] == [0, 1]  # stopped at the raise


def test_existing_zip_is_skipped(tmp_path):
    shard_paths(tmp_path, 1)[0].write_bytes(b"zip")  # shard_001.zip present
    be = FakeBackend()
    results = rsi.run_all(be, [0, 1, 2], 300, tmp_path)
    assert [c[0] for c in be.run_shard_calls] == [0, 2]
    assert results == {0: True, 1: True, 2: True}


def test_one_shard_failing_does_not_stop_the_rest(tmp_path):
    be = FakeBackend(ok_by_index={1: False})
    results = rsi.run_all(be, [0, 1, 2], 300, tmp_path)
    assert results == {0: True, 1: False, 2: True}
    assert [c[0] for c in be.run_shard_calls] == [0, 1, 2]


def test_model_ok_yields_seedmode_hf(tmp_path, monkeypatch):
    import run_sharded_ingest as rsi
    from backends.base import SeedMode
    monkeypatch.setattr(rsi.hf_cache, "check_model", lambda *a, **k: rsi.hf_cache.Ok())
    monkeypatch.setattr(rsi.hf_cache, "mirror_to_local", lambda *a, **k: None)
    be = FakeBackend()
    rsi.run_all(be, [0, 1], 300, tmp_path)
    assert be.run_shard_calls == [(0, 300, SeedMode.HF), (1, 300, SeedMode.HF)]


def test_model_mismatch_non_tty_no_flag_exits(tmp_path, monkeypatch):
    import run_sharded_ingest as rsi
    monkeypatch.setattr(rsi.hf_cache, "check_model",
                        lambda *a, **k: rsi.hf_cache.Mismatch("old", "new"))
    monkeypatch.setattr(rsi.hf_cache, "mirror_to_local", lambda *a, **k: None)
    monkeypatch.setattr(rsi.sys.stdin, "isatty", lambda: False)
    be = FakeBackend()
    with __import__("pytest").raises(SystemExit):
        rsi.run_all(be, [0], 300, tmp_path)
    assert be.run_shard_calls == []


def test_model_mismatch_with_flag_yields_overwrite(tmp_path, monkeypatch):
    import run_sharded_ingest as rsi
    from backends.base import SeedMode
    monkeypatch.setattr(rsi.hf_cache, "check_model",
                        lambda *a, **k: rsi.hf_cache.Mismatch("old", "new"))
    monkeypatch.setattr(rsi.hf_cache, "mirror_to_local", lambda *a, **k: None)
    be = FakeBackend()
    rsi.run_all(be, [0], 300, tmp_path, reseed_on_model_mismatch=True)
    assert be.run_shard_calls == [(0, 300, SeedMode.SEEDLESS_OVERWRITE)]


def test_mirror_called_after_every_shard(tmp_path, monkeypatch):
    import run_sharded_ingest as rsi
    monkeypatch.setattr(rsi.hf_cache, "check_model", lambda *a, **k: rsi.hf_cache.Ok())
    calls = []
    monkeypatch.setattr(rsi.hf_cache, "mirror_to_local",
                        lambda i, d, **k: calls.append(i))
    be = FakeBackend(ok_by_index={1: False})
    rsi.run_all(be, [0, 1], 300, tmp_path)
    assert calls == [0, 1]  # mirror even when shard 1 failed


def test_backend_arg_defaults_to_colab():
    p = rsi._build_parser()
    ns = p.parse_args(["--total-acts", "10"])
    assert ns.backend == "colab"


def test_new_runpod_flags_parse():
    p = rsi._build_parser()
    ns = p.parse_args(["--total-acts", "10", "--backend", "runpod",
                       "--yes", "--keep-pod", "--reuse-pod", "--cloud-type", "SECURE"])
    assert ns.backend == "runpod" and ns.yes and ns.keep_pod and ns.reuse_pod
    assert ns.cloud_type == "SECURE"


def test_hf_cache_repo_default_tracks_the_module_constant():
    # A plain run must not trip the override/env-export branch, which keys off
    # `args.hf_cache_repo != hf_cache._DEFAULT_HF_CACHE_REPO`. Defaulting the
    # argparse value to the same constant (not a re-typed literal) is what keeps
    # the two in lockstep if the constant ever changes.
    p = rsi._build_parser()
    ns = p.parse_args(["--total-acts", "10"])
    assert ns.hf_cache_repo == rsi.hf_cache._DEFAULT_HF_CACHE_REPO


def test_reap_terminates_lexau_pods_and_clears_file(tmp_path, monkeypatch, capsys):
    calls = {"terminated": []}
    fake_rd = types.SimpleNamespace(
        list_pods=lambda: [
            {"id": "a", "name": "lexau-ingest-1", "desiredStatus": "RUNNING"},
            {"id": "b", "name": "other", "desiredStatus": "RUNNING"},
        ],
        terminate_pod=lambda pid, **k: calls["terminated"].append(pid) or True,
    )
    monkeypatch.setattr(rsi, "runpod_driver", fake_rd, raising=False)
    (tmp_path / ".runpod_pod").write_text("a")
    with pytest.raises(SystemExit) as e:
        rsi.main(["--total-acts", "10", "--backend", "runpod", "--reap",
                  "--shards-dir", str(tmp_path)])
    assert e.value.code == 0
    assert calls["terminated"] == ["a"]
    assert not (tmp_path / ".runpod_pod").exists()


def test_reap_does_not_require_total_acts(tmp_path, monkeypatch):
    calls = {"terminated": []}
    fake_rd = types.SimpleNamespace(
        list_pods=lambda: [
            {"id": "a", "name": "lexau-ingest-1", "desiredStatus": "RUNNING"},
            {"name": "lexau-ingest-broken", "desiredStatus": "RUNNING"},  # no id
        ],
        terminate_pod=lambda pid, **k: calls["terminated"].append(pid) or True,
    )
    monkeypatch.setattr(rsi, "runpod_driver", fake_rd, raising=False)
    with pytest.raises(SystemExit) as e:
        rsi.main(["--backend", "runpod", "--reap", "--shards-dir", str(tmp_path)])
    assert e.value.code == 0
    assert calls["terminated"] == ["a"]  # the id-less pod is skipped, not crashed on


def test_total_acts_still_required_without_reap(tmp_path):
    with pytest.raises(SystemExit) as e:
        rsi.main(["--backend", "runpod", "--shards-dir", str(tmp_path)])
    assert e.value.code == 2


def _runpod_args(tmp_path):
    return types.SimpleNamespace(
        shards_dir=tmp_path, cloud_type="COMMUNITY", yes=True,
        reuse_pod=False, keep_pod=False,
    )


def test_get_backend_runpod_parses_gpu_type_ids_env(tmp_path, monkeypatch):
    from backends import get_backend
    monkeypatch.setenv("RUNPOD_GPU_TYPE_IDS",
                       " NVIDIA RTX A5000 , NVIDIA GeForce RTX 3090 ,, NVIDIA RTX A6000 ")
    be = get_backend("runpod", _runpod_args(tmp_path))
    assert be.gpu_type_ids == (
        "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000",
    )


def test_get_backend_runpod_gpu_type_ids_absent_is_empty(tmp_path, monkeypatch):
    from backends import get_backend
    monkeypatch.delenv("RUNPOD_GPU_TYPE_IDS", raising=False)
    be = get_backend("runpod", _runpod_args(tmp_path))
    assert be.gpu_type_ids == ()


# --------------------------- I4: a transient check_model error must not abort

def test_check_model_error_fails_one_shard_without_aborting_run(tmp_path, monkeypatch, capsys):
    def _flaky(i, *a, **k):
        if i == 0:
            raise RuntimeError("HF 503 Service Unavailable")
        return rsi.hf_cache.Ok()

    monkeypatch.setattr(rsi.hf_cache, "check_model", _flaky)
    be = FakeBackend()
    results = rsi.run_all(be, [0, 1], 300, tmp_path)
    assert results == {0: False, 1: True}
    assert [c[0] for c in be.run_shard_calls] == [1]  # shard 0 never launched
    assert "shard 0" in capsys.readouterr().err


# --------------------------- C1: --hf-cache-repo must reach the compute VM

def _fake_backend_main(monkeypatch, tmp_path, extra):
    be = FakeBackend()
    monkeypatch.setattr(rsi, "get_backend", lambda name, args: be)
    rsi.main(["--total-acts", "300", "--shard-size", "300",
              "--shards-dir", str(tmp_path), *extra])
    return be


def test_hf_cache_repo_override_is_exported_to_the_environment(tmp_path, monkeypatch):
    import os
    monkeypatch.delenv("LEXAU_HF_CACHE_REPO", raising=False)
    monkeypatch.setattr(rsi.hf_cache, "HF_CACHE_REPO", rsi.hf_cache._DEFAULT_HF_CACHE_REPO)
    _fake_backend_main(monkeypatch, tmp_path, ["--hf-cache-repo", "cchew/throwaway"])
    assert os.environ["LEXAU_HF_CACHE_REPO"] == "cchew/throwaway"
    assert rsi.hf_cache.HF_CACHE_REPO == "cchew/throwaway"


def test_default_hf_cache_repo_sets_no_env_override(tmp_path, monkeypatch):
    import os
    monkeypatch.delenv("LEXAU_HF_CACHE_REPO", raising=False)
    _fake_backend_main(monkeypatch, tmp_path, [])
    assert "LEXAU_HF_CACHE_REPO" not in os.environ


def test_completion_hint_uses_the_from_hf_recipe(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LEXAU_HF_CACHE_REPO", raising=False)
    _fake_backend_main(monkeypatch, tmp_path, [])
    err = capsys.readouterr().err
    assert "--from-hf --push-hf" in err
    assert "--shard-cache-paths" not in err
