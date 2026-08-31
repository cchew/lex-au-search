"""The RunPod stock image (runpod/pytorch:*-py3.11-cuda12.4.1-*) ships
Python 3.11; the spec's §12 spike proved the package must install there.
This guards against the floor drifting back to 3.12 and against 3.12-only
syntax creeping in."""
import ast
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).parent.parent


def test_requires_python_allows_3_11():
    meta = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert meta["project"]["requires-python"] == ">=3.11"


def test_no_pep695_or_312_only_syntax_in_shipped_code():
    # Parse every src/ and scripts/ .py under a 3.11 grammar proxy: reject
    # PEP 695 type-parameter syntax, which is the most likely 3.12-ism.
    offenders = []
    for base in ("src", "scripts"):
        for p in (REPO / base).rglob("*.py"):
            tree = ast.parse(p.read_text(), filename=str(p))
            for node in ast.walk(tree):
                if type(node).__name__ in {"TypeAlias", "TypeVar", "ParamSpec", "TypeVarTuple"} \
                        and getattr(node, "lineno", None):
                    offenders.append(f"{p}:{node.lineno}")
                # PEP 695: FunctionDef/ClassDef.type_params non-empty
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                        and getattr(node, "type_params", []):
                    offenders.append(f"{p}:{node.lineno} (PEP 695 type params)")
    assert not offenders, offenders
