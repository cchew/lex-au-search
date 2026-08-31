import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backends.fake import FakeBackend


def test_records_calls_and_returns_default_ok():
    be = FakeBackend()
    be.prepare()
    r = be.run_shard(2, 300, None)
    be.teardown()
    assert be.prepare_calls == 1
    assert be.teardown_calls == 1
    assert be.run_shard_calls == [(2, 300, None)]
    assert r.ok is True and r.index == 2


def test_ok_by_index_controls_result():
    be = FakeBackend(ok_by_index={1: False})
    assert be.run_shard(1, 300, None).ok is False
    assert be.run_shard(2, 300, None).ok is True


def test_raise_on_index():
    be = FakeBackend(raise_on_index=5)
    with pytest.raises(RuntimeError):
        be.run_shard(5, 300, None)
