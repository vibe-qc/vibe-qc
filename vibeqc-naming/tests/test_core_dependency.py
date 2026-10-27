"""Core dependency and offline packaging contract; no native build required."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (ROOT / "scripts/_venv_helpers.sh").is_file(),
    reason="core integration checks require the source checkout",
)


def runtime():
    spec = importlib.util.spec_from_file_location(
        "packaging_runtime", ROOT / "packaging/macos/runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_dependency_and_uv_source():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "vibeqc-naming>=0.1.0,<0.2" in project["project"]["dependencies"]
    assert (
        "python/vibeqc_naming"
        not in project["tool"]["scikit-build"]["wheel"]["packages"]
    )
    assert project["tool"]["uv"]["sources"]["vibeqc-naming"] == {
        "path": "vibeqc-naming"
    }


def wheel(
    tmp_path, *, version="0.1.0", pure="true", payload=b"qualified source", extra=False
):
    path = tmp_path / "vibeqc_naming-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("vibeqc_naming/__init__.py", payload)
        z.writestr(
            "vibeqc_naming-0.1.0.dist-info/METADATA",
            f"Name: vibeqc-naming\nVersion: {version}\n",
        )
        z.writestr(
            "vibeqc_naming-0.1.0.dist-info/WHEEL",
            f"Root-Is-Purelib: {pure}\nTag: py3-none-any\n",
        )
        if extra:
            z.writestr("vibeqc/__init__.py", b"foreign core")
    return path


def source_hash():
    data = b"qualified source"
    return {
        "vibeqc_naming/__init__.py": hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
    }


def test_wheel_is_bound_to_selected_source(tmp_path):
    path = wheel(tmp_path)
    result = runtime().check_naming_wheel(path, "0.1.0", source_hash())
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"version": "0.2.0"},
        {"pure": "false"},
        {"payload": b"changed same version"},
        {"extra": True},
    ],
)
def test_bad_naming_wheel_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        runtime().check_naming_wheel(wheel(tmp_path, **kwargs), "0.1.0", source_hash())


def test_installer_installs_standalone_before_core(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    log = tmp_path / "calls.jsonl"
    python = venv / "bin/python"
    python.write_text(
        "#!"
        + sys.executable
        + "\nimport json,sys\nwith open("
        + repr(str(log))
        + ',"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'
    )
    python.chmod(0o755)
    subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1"; vibeqc_pip_install_editable "$2" "[test]" "$3"',
            "test",
            str(ROOT / "scripts/_venv_helpers.sh"),
            str(venv),
            str(ROOT),
        ],
        check=True,
    )
    calls = [json.loads(x) for x in log.read_text().splitlines()]
    naming = next(i for i, c in enumerate(calls) if str(ROOT / "vibeqc-naming") in c)
    core = next(i for i, c in enumerate(calls) if str(ROOT) + "[test]" in c)
    assert naming < core
    assert "-e" not in calls[naming]


def test_installer_refuses_missing_standalone(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; vibeqc_pip_install_editable "$2" "" "$3"',
            "test",
            str(ROOT / "scripts/_venv_helpers.sh"),
            str(venv),
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "co-located vibeqc-naming" in result.stderr
