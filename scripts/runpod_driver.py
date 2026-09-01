#!/usr/bin/env python3
"""Stdlib-only RunPod REST v1 wrapper + CLI for the sharded-ingest RunPod
backend. Only the surface backends/runpod_backend.py uses:
create / list / status / wait-ready / terminate.

RUNPOD_API_KEY from the environment, or from repo `.env` (see _envload).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from _envload import load_env

API_BASE = "https://rest.runpod.io/v1"
STOCK_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


@dataclass
class CreatePodConfig:
    name: str
    image_name: str = STOCK_IMAGE
    gpu_type_id: str = "NVIDIA RTX A6000"
    cloud_type: str = "COMMUNITY"
    container_disk_gb: int = 80
    allowed_cuda_versions: tuple[str, ...] = ("12.4", "12.5", "12.6", "12.7", "12.8")


def build_create_payload(cfg: CreatePodConfig) -> dict:
    return {
        "name": cfg.name,
        "imageName": cfg.image_name,
        "cloudType": cfg.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": [cfg.gpu_type_id],
        "gpuCount": 1,
        "interruptible": False,
        "containerDiskInGb": cfg.container_disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "allowedCudaVersions": list(cfg.allowed_cuda_versions),
    }


def pod_name(now: datetime | None = None) -> str:
    return f"lexau-ingest-{(now or datetime.now(timezone.utc)):%Y%m%dT%H%M%SZ}"


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        load_env(__file__)
        key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print(
            "RUNPOD_API_KEY not set. `export RUNPOD_API_KEY=...`, or "
            "`set -a && source .env && set +a`.",
            file=sys.stderr,
        )
        sys.exit(2)
    return key
