# tests/test_shell_scripts.py
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"


def test_setup_gpu_env_has_the_env_level_steps():
    s = (SCRIPTS / "setup_gpu_env.sh").read_text()
    assert 'pip install -e ".[gpu]"' in s
    # CUDA-12 onnxruntime-gpu from the ADO feed (plain PyPI now ships CUDA-13),
    # PLUS an explicit cuDNN 9 wheel (neither RunPod stock nor Colab reliably
    # has it on onnxruntime's loader path).
    assert "aiinfra.pkgs.visualstudio.com" in s
    assert "onnxruntime-gpu==1.27.0" in s
    assert "nvidia-cudnn-cu12" in s
    # Validation is delegated to the shared real-CUDA probe, not an inline
    # python3 -c assert (get_available_providers() is a false green).
    assert "_verify_gpu.py" in s
    assert 'python3 -c "import onnxruntime' not in s


def test_setup_gpu_env_installs_zip_binary():
    # Stock RunPod images have no `zip`; ingest_shard.sh packages shard_storage/
    # with it. Guarded by `command -v` so Colab (already has zip) skips the apt.
    s = (SCRIPTS / "setup_gpu_env.sh").read_text()
    assert "apt-get install" in s
    assert "zip" in s
    assert "command -v zip" in s


def test_setup_gpu_env_does_no_per_shard_work():
    s = (SCRIPTS / "setup_gpu_env.sh").read_text()
    assert "ingest-shard" not in s
    assert "snapshot_download" not in s and "hf download" not in s


def test_verify_gpu_forces_a_real_cuda_session():
    s = (SCRIPTS / "_verify_gpu.py").read_text()
    assert "InferenceSession" in s
    assert 'providers=["CUDAExecutionProvider"]' in s
    assert "preload_dlls" in s
    # must abort (non-zero exit) on fallback, not just print
    assert "sys.exit(" in s


def test_ingest_shard_verifies_gpu_before_the_long_embed():
    s = (SCRIPTS / "ingest_shard.sh").read_text()
    assert "_verify_gpu.py" in s
    # the check sits before the ingest-shard call
    assert s.index("_verify_gpu.py") < s.index("lex-au-search ingest-shard")


def test_ingest_shard_renamed_and_stripped():
    assert not (SCRIPTS / "colab_ingest_shard.sh").exists()
    s = (SCRIPTS / "ingest_shard.sh").read_text()
    assert 'pip install -e ".[gpu]"' not in s
    assert "onnxruntime-gpu==1.27.0" not in s
    assert "LEXAU_EMBED_BATCH_SIZE" not in s
    assert "lex-au-search ingest-shard" in s


def test_ingest_shard_takes_optional_seed_path_arg():
    s = (SCRIPTS / "ingest_shard.sh").read_text()
    assert "/content/shard_cache_seed.db" not in s
    assert "$3" in s or "${3" in s


def test_gpu_extra_bundles_cudnn_and_huggingface_hub():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    gpu = data["project"]["optional-dependencies"]["gpu"]
    joined = " ".join(gpu)
    assert "huggingface_hub" in joined  # imported by ingest_shard.sh
    assert "nvidia-cudnn-cu12" in joined
