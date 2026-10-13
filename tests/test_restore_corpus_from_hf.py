import importlib.util
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "restore_corpus_from_hf",
        Path(__file__).parent.parent / "scripts" / "restore_corpus_from_hf.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restore_calls_snapshot_download_with_expected_args(monkeypatch, tmp_path):
    module = _load_module()
    calls = {}

    def fake_snapshot_download(*, repo_id, repo_type, local_dir, allow_patterns, max_workers):
        calls["repo_id"] = repo_id
        calls["repo_type"] = repo_type
        calls["local_dir"] = local_dir
        calls["allow_patterns"] = allow_patterns
        calls["max_workers"] = max_workers
        return local_dir

    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)

    local_dir = str(tmp_path / "corpus")
    result = module.restore("cchew/lex-au", local_dir)

    assert calls == {
        "repo_id": "cchew/lex-au",
        "repo_type": "dataset",
        "local_dir": local_dir,
        "allow_patterns": ["index.json", "xml/*.xml"],
        "max_workers": 16,
    }
    assert result == local_dir


def test_restore_max_workers_is_overridable(monkeypatch, tmp_path):
    module = _load_module()
    seen = {}
    monkeypatch.setattr(
        module, "snapshot_download",
        lambda **kwargs: seen.setdefault("max_workers", kwargs["max_workers"]) or kwargs["local_dir"],
    )

    module.restore("cchew/lex-au", str(tmp_path / "corpus"), max_workers=32)

    assert seen["max_workers"] == 32
