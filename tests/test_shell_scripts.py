# tests/test_shell_scripts.py
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent / "scripts"


def test_setup_gpu_env_has_the_env_level_steps():
    s = (SCRIPTS / "setup_gpu_env.sh").read_text()
    assert 'pip install -e ".[gpu]"' in s
    assert "pip install huggingface_hub" in s or "pip install -q huggingface_hub" in s
    assert "onnxruntime-gpu==1.27.0" in s
    assert "aiinfra.pkgs.visualstudio.com" in s
    assert "CUDAExecutionProvider" in s


def test_setup_gpu_env_does_no_per_shard_work():
    s = (SCRIPTS / "setup_gpu_env.sh").read_text()
    assert "ingest-shard" not in s
    assert "snapshot_download" not in s and "hf download" not in s


def test_ingest_shard_renamed_and_stripped():
    assert not (SCRIPTS / "colab_ingest_shard.sh").exists()
    s = (SCRIPTS / "ingest_shard.sh").read_text()
    assert 'pip install -e ".[gpu]"' not in s          # moved to setup_gpu_env.sh
    assert "onnxruntime-gpu==1.27.0" not in s
    assert "LEXAU_EMBED_BATCH_SIZE" not in s           # caller's env only
    assert "lex-au-search ingest-shard" in s


def test_ingest_shard_takes_optional_seed_path_arg():
    s = (SCRIPTS / "ingest_shard.sh").read_text()
    assert "/content/shard_cache_seed.db" not in s     # no hardcoded Colab path
    assert "$3" in s or "${3" in s                     # third positional = seed db
