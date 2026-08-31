import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from _envload import load_env


def test_loads_keys_relative_to_scriptdir_parent(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / ".env").write_text(
        "# a comment\n"
        "\n"
        "RUNPOD_API_KEY=rpa_secret\n"
        "RUNPOD_SSH_KEY=~/.ssh/id_ed25519\n"
        "MALFORMED LINE NO EQUALS\n"
    )
    fake_script = repo / "scripts" / "runpod_driver.py"
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_SSH_KEY", raising=False)

    load_env(fake_script)

    assert os.environ["RUNPOD_API_KEY"] == "rpa_secret"
    assert os.environ["RUNPOD_SSH_KEY"] == os.path.expanduser("~/.ssh/id_ed25519")


def test_does_not_override_existing_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / ".env").write_text("RUNPOD_API_KEY=from_file\n")
    monkeypatch.setenv("RUNPOD_API_KEY", "from_shell")

    load_env(repo / "scripts" / "x.py")

    assert os.environ["RUNPOD_API_KEY"] == "from_shell"


def test_missing_env_file_is_silent(tmp_path):
    load_env(tmp_path / "scripts" / "x.py")  # no .env anywhere; must not raise
