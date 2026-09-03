"""In-memory backend for exercising run_sharded_ingest's orchestration
without any cloud calls."""

from __future__ import annotations

from backends.base import IngestBackend, ShardResult, SeedMode


class FakeBackend(IngestBackend):
    def __init__(
        self,
        ok_by_index: dict[int, bool] | None = None,
        raise_on_index: int | None = None,
    ) -> None:
        self._ok_by_index = ok_by_index or {}
        self._raise_on_index = raise_on_index
        self.prepare_calls = 0
        self.teardown_calls = 0
        self.run_shard_calls: list[tuple[int, int, SeedMode]] = []

    def prepare(self) -> None:
        self.prepare_calls += 1

    def run_shard(self, index: int, shard_size: int, seed_mode: SeedMode) -> ShardResult:
        self.run_shard_calls.append((index, shard_size, seed_mode))
        if self._raise_on_index == index:
            raise RuntimeError(f"FakeBackend forced failure on shard {index}")
        return ShardResult(index, self._ok_by_index.get(index, True), None, None, "")

    def teardown(self) -> None:
        self.teardown_calls += 1
