#!/usr/bin/env python3
"""Fail loudly if onnxruntime cannot actually execute on CUDA.

`onnxruntime.get_available_providers()` lists the providers the wheel was
BUILT with, not the ones that can be loaded at runtime - so a silent
fall-back to CPU (missing cuDNN 9, wrong CUDA, no GPU) sails straight past
it. This forces a real CUDA `InferenceSession` on a 72-byte Identity model
and checks it did not downgrade.

Used by setup_gpu_env.sh (post-install), ingest_shard.sh (before a
multi-hour embed), and RunPodBackend.prepare() step 6e (which also runs on
the --reuse-pod path, where setup_gpu_env.sh is skipped).
"""
from __future__ import annotations

import base64
import sys

import onnxruntime as ort

# x[1,4] @ w[4,4] (w = identity initializer) -> y[1,4]. A real MatMul/GEMM, so
# constructing the session forces onnxruntime to bind a GPU BLAS kernel - a box
# where CUDA EP inits but cuDNN/cuBLAS is a wrong minor version would still fail
# here, unlike an Identity-only probe. opset 13, ir_version 8; regenerated with
#   onnx.helper.make_model(make_graph([make_node("MatMul", ["x","w"], ["y"])],
#       "gpuprobe", [in], [out], initializer=[eye4]), ...)
_PROBE_ONNX = base64.b64decode(
    "CAg6lAEKEQoBeAoBdxIBeSIGTWF0TXVsEghncHVwcm9iZSpLCAQIBBABIkAAAIA/AAAAAAAAAAA"
    "AAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/QgF3Wh"
    "MKAXgSDgoMCAESCAoCCAEKAggEYhMKAXkSDgoMCAESCAoCCAEKAggEQgQKABAN"
)


def main() -> None:
    # onnxruntime-gpu resolves cuDNN 9 from the nvidia-cudnn-cu12 wheel only
    # when preload is triggered and no `import torch` preceded it.
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload()
        except Exception as e:  # noqa: BLE001 - informational only
            print(f"_verify_gpu: preload_dlls() warning: {e}", file=sys.stderr)

    try:
        sess = ort.InferenceSession(_PROBE_ONNX, providers=["CUDAExecutionProvider"])
    except Exception as e:  # noqa: BLE001 - any construction failure is fatal here
        sys.exit(f"_verify_gpu: CUDA InferenceSession failed to construct: {e}")

    providers = sess.get_providers()
    if "CUDAExecutionProvider" not in providers:
        sys.exit(
            "_verify_gpu: onnxruntime fell back off CUDA (active providers: "
            f"{providers}). cuDNN 9 / CUDA 12 runtime missing on the loader path?"
        )
    print(f"_verify_gpu: CUDAExecutionProvider active ({providers})")


if __name__ == "__main__":
    main()
