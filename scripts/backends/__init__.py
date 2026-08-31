"""Backend registry for sharded ingest."""

from __future__ import annotations

from backends.base import IngestBackend


def get_backend(name: str, args) -> IngestBackend:
    if name == "colab":
        from backends.colab import ColabBackend

        return ColabBackend(args.shards_dir, args.gpu)
    if name == "runpod":
        raise NotImplementedError("RunPod backend lands in Phase B")
    raise ValueError(f"unknown backend: {name!r}")
