import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backends.base import IngestBackend, ShardResult, shard_paths, checkpoint_cache_path
from backends import get_backend
from backends.colab import ColabBackend


def test_shard_result_fields():
    r = ShardResult(index=3, ok=True, storage_zip=Path("a.zip"), cache_zip=Path("b.zip"), diagnosis="")
    assert (r.index, r.ok, r.diagnosis) == (3, True, "")
    assert r.storage_zip == Path("a.zip")


def test_ingest_backend_cannot_be_instantiated():
    with pytest.raises(TypeError):
        IngestBackend()


def test_subclass_missing_a_method_cannot_be_instantiated():
    class Partial(IngestBackend):
        def prepare(self): ...
        def teardown(self): ...
        # run_shard intentionally missing

    with pytest.raises(TypeError):
        Partial()


def test_full_subclass_instantiates():
    class Full(IngestBackend):
        def prepare(self): ...
        def run_shard(self, index, shard_size, seed_mode):
            return ShardResult(index, True, None, None, "")
        def teardown(self): ...

    assert isinstance(Full(), IngestBackend)


def test_shard_paths_zero_pads_to_three_digits():
    a, b = shard_paths(Path("/x/shards"), 7)
    assert a == Path("/x/shards/shard_007.zip")
    assert b == Path("/x/shards/shard_007_cache.zip")


def test_checkpoint_cache_path_matches_contract():
    assert checkpoint_cache_path(Path("/x/shards"), 0) == Path("/x/shards/shard_000_checkpoint_cache.db")


def _args(tmp_path):
    return types.SimpleNamespace(shards_dir=tmp_path, gpu="T4")


def test_get_backend_colab(tmp_path):
    be = get_backend("colab", _args(tmp_path))
    assert isinstance(be, ColabBackend)
    assert be.shards_dir == tmp_path and be.gpu == "T4"


def test_get_backend_runpod_constructs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_SSH_KEY", "~/.ssh/k")
    args = types.SimpleNamespace(shards_dir=tmp_path, cloud_type="COMMUNITY",
                                 yes=True, reuse_pod=False, keep_pod=False, gpu="T4")
    be = get_backend("runpod", args)
    from backends.runpod_backend import RunPodBackend
    assert isinstance(be, RunPodBackend)


def test_get_backend_unknown(tmp_path):
    with pytest.raises(ValueError):
        get_backend("bogus", _args(tmp_path))
