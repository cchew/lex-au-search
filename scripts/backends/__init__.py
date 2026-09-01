"""Backend registry for sharded ingest."""

from __future__ import annotations

from backends.base import IngestBackend


def get_backend(name: str, args) -> IngestBackend:
    if name == "colab":
        from backends.colab import ColabBackend

        return ColabBackend(args.shards_dir, args.gpu)
    if name == "runpod":
        import os
        from backends.runpod_backend import RunPodBackend

        ssh_key = os.environ.get("RUNPOD_SSH_KEY") or os.path.expanduser("~/.ssh/id_ed25519")
        gpu_type_ids = tuple(
            s.strip() for s in os.environ.get("RUNPOD_GPU_TYPE_IDS", "").split(",") if s.strip()
        )
        return RunPodBackend(
            args.shards_dir,
            ssh_key=ssh_key,
            cloud_type=args.cloud_type,
            gpu_type_ids=gpu_type_ids,
            assume_yes=args.yes,
            reuse_pod=args.reuse_pod,
            keep_pod=args.keep_pod,
        )
    raise ValueError(f"unknown backend: {name!r}")
