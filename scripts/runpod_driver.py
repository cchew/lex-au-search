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
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _envload import load_env

API_BASE = "https://rest.runpod.io/v1"
STOCK_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


@dataclass
class CreatePodConfig:
    name: str
    image_name: str = STOCK_IMAGE
    gpu_type_id: str = "NVIDIA RTX A6000"
    # Ordered preference list. When non-empty it is sent verbatim as
    # `gpuTypeIds` and RunPod picks the first type with capacity; when empty
    # the single `gpu_type_id` is used. A6000 (48GB) is wild overkill for the
    # ~6GB arctic-embed-l job, so a run can prefer cheaper/abundant cards.
    gpu_type_ids: tuple[str, ...] = ()
    cloud_type: str = "COMMUNITY"
    container_disk_gb: int = 80
    allowed_cuda_versions: tuple[str, ...] = ("12.4", "12.5", "12.6", "12.7", "12.8")


def build_create_payload(cfg: CreatePodConfig) -> dict:
    return {
        "name": cfg.name,
        "imageName": cfg.image_name,
        "cloudType": cfg.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": list(cfg.gpu_type_ids) if cfg.gpu_type_ids else [cfg.gpu_type_id],
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


def _request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        API_BASE + path,
        method=method,
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return getattr(r, "status", 200), (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        return e.code, (json.loads(raw) if raw.strip() else {})
    except (urllib.error.URLError, socket.timeout, TimeoutError):
        # A DNS blip or read timeout during the 10-minute wait_ready poll must
        # not propagate out and abandon a pod that was just created and is
        # already billing. Report "no answer" and let the caller retry.
        return 0, {}


def create_pod(cfg: CreatePodConfig, *, dry_run: bool = False) -> dict:
    payload = build_create_payload(cfg)
    if dry_run:
        print(json.dumps(payload, indent=2))
        return {}
    status, resp = _request("POST", "/pods", payload)
    if (
        status == 500
        and cfg.cloud_type == "COMMUNITY"
        and "no instances currently available" in json.dumps(resp).lower()
    ):
        print("COMMUNITY has no instances; retrying on SECURE ...", file=sys.stderr)
        status, resp = _request("POST", "/pods", {**payload, "cloudType": "SECURE"})
    if status not in (200, 201):
        raise RuntimeError(f"create pod failed ({status}): {resp}")
    return resp


def list_pods() -> list[dict]:
    _, resp = _request("GET", "/pods")
    return resp if isinstance(resp, list) else resp.get("pods", []) or []


def get_status(pod_id: str) -> str:
    status, resp = _request("GET", f"/pods/{pod_id}")
    if status == 404:
        return "TERMINATED"
    return resp.get("desiredStatus", "UNKNOWN")


def ssh_coords(pod_id: str) -> tuple[str, int] | None:
    _, resp = _request("GET", f"/pods/{pod_id}")
    ip = resp.get("publicIp") or ""
    pm = resp.get("portMappings") or {}
    port = pm.get("22") if isinstance(pm, dict) else None
    if ip and port:
        return ip, int(port)
    return None


def wait_ready(pod_id: str, *, timeout_s: int = 600, sleep=time.sleep, now=time.monotonic) -> tuple[str, int]:
    start = now()
    while now() - start < timeout_s:
        coords = ssh_coords(pod_id)
        if coords:
            return coords
        sleep(10)
    raise TimeoutError(f"pod {pod_id} not SSH-ready after {timeout_s}s")


def terminate_pod(pod_id: str, *, sleep=time.sleep) -> bool:
    _request("DELETE", f"/pods/{pod_id}")
    for i in range(3):
        if get_status(pod_id) == "TERMINATED":
            return True
        if i < 2:
            sleep(5)
    print(
        f"MANUAL ACTION: python scripts/runpod_driver.py terminate {pod_id}\n"
        f"  https://www.runpod.io/console/pods",
    )
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--name", default=None)
    c.add_argument("--gpu", default="NVIDIA RTX A6000")
    c.add_argument("--gpu-type-ids", default="",
                   help="Comma-separated ordered preference list; overrides --gpu. "
                        "RunPod picks the first type with capacity.")
    c.add_argument("--cloud-type", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    sub.add_parser("list")
    for name in ("status", "wait-ready", "terminate"):
        sp = sub.add_parser(name)
        sp.add_argument("pod_id")
    args = ap.parse_args()

    if args.cmd == "create":
        gpu_type_ids = tuple(s.strip() for s in args.gpu_type_ids.split(",") if s.strip())
        cfg = CreatePodConfig(
            name=args.name or pod_name(),
            gpu_type_id=args.gpu,
            gpu_type_ids=gpu_type_ids,
            cloud_type=args.cloud_type,
        )
        out = create_pod(cfg, dry_run=args.dry_run)
        if out:
            print(json.dumps(out, indent=2))
    elif args.cmd == "list":
        print(json.dumps(list_pods(), indent=2))
    elif args.cmd == "status":
        print(get_status(args.pod_id))
    elif args.cmd == "wait-ready":
        ip, port = wait_ready(args.pod_id)
        print(f"{ip} {port}")
    elif args.cmd == "terminate":
        sys.exit(0 if terminate_pod(args.pod_id) else 1)


if __name__ == "__main__":
    main()
