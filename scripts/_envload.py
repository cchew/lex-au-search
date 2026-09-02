"""Hand-rolled .env loader (no python-dotenv dependency). Resolves the file
relative to the calling script's directory parent (repo root for scripts/*.py),
not cwd, so `python scripts/runpod_driver.py` works from anywhere."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(script_file: str | Path) -> None:
    env_path = Path(script_file).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one layer of matching surrounding quotes so a value with spaces
        # (e.g. RUNPOD_GPU_TYPE_IDS="NVIDIA RTX A5000,...") is also valid for a
        # shell `set -a && source .env`.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = os.path.expanduser(value)
