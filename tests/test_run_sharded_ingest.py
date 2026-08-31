# tests/test_run_sharded_ingest.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backends.base import checkpoint_cache_path, shard_paths
from backends.fake import FakeBackend
import run_sharded_ingest as rsi


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


def test_seed_passed_only_when_accumulator_exists(tmp_path):
    checkpoint_cache_path(tmp_path, 2).write_bytes(b"db")
    be = FakeBackend()
    rsi.run_all(be, [1, 2], 300, tmp_path)
    seeds = {idx: seed for idx, _, seed in be.run_shard_calls}
    assert seeds[1] is None
    assert seeds[2] == checkpoint_cache_path(tmp_path, 2)


def test_backend_arg_defaults_to_colab():
    p = rsi._build_parser()
    ns = p.parse_args(["--total-acts", "10"])
    assert ns.backend == "colab"
