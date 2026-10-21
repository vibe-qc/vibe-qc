"""The standalone ``vibeqc-naming`` distribution (#536).

Metadata, wheel and sdist contents, the sdist round trip and the editable
layout. hatchling builds into temporary directories; nothing is installed
and no network is used.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

DIST_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = DIST_ROOT / "pyproject.toml"
CHECKOUT_SOURCE = DIST_ROOT.parent / "python" / "vibeqc_naming"
SDIST_SOURCE = DIST_ROOT / "src" / "vibeqc_naming"
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?$")


def _hook_module():
    spec = importlib.util.spec_from_file_location(
        "vibeqc_naming_hatch_build", DIST_ROOT / "hatch_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build(target: str, cwd: Path, out: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", target, "-d", str(out)],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    suffix = ".whl" if target == "wheel" else ".tar.gz"
    [artifact] = [p for p in out.iterdir() if p.name.endswith(suffix)]
    return artifact


def _wheel_files(wheel: Path) -> list[str]:
    """The packaged files, without the ``.dist-info`` metadata."""
    with zipfile.ZipFile(wheel) as archive:
        return sorted(n for n in archive.namelist() if ".dist-info/" not in n)


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_dir() -> Path:
    for candidate in (CHECKOUT_SOURCE, SDIST_SOURCE):
        if candidate.is_dir():
            return candidate
    pytest.skip("no package source beside the distribution")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, Path]:
    pytest.importorskip("hatchling")
    wheel_dir = tmp_path_factory.mktemp("wheel")
    sdist_dir = tmp_path_factory.mktemp("sdist")
    return _build("wheel", DIST_ROOT, wheel_dir), _build("sdist", DIST_ROOT, sdist_dir)


def test_pyproject_and_dunder_version_agree(pyproject):
    import vibeqc_naming

    declared = pyproject["project"]["version"]
    assert declared == vibeqc_naming.__version__, (
        f"version drift: pyproject.toml says {declared!r} but "
        f"vibeqc_naming.__version__ is {vibeqc_naming.__version__!r}"
    )
    assert _SEMVER.match(declared), declared


def test_the_distribution_is_pure_python(pyproject, source_dir):
    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    names = {re.split(r"[<>=!~ ;\[]", dep, 1)[0] for dep in pyproject["project"]["dependencies"]}
    assert names == {"numpy"}
    compiled = [p.name for p in source_dir.rglob("*") if p.suffix in {".so", ".pyd", ".dylib", ".c", ".cpp"}]
    assert compiled == []


def test_requires_python_matches_vibe_qc(pyproject):
    root = DIST_ROOT.parent / "pyproject.toml"
    if not root.is_file():
        pytest.skip("not inside a vibe-qc checkout")
    expected = tomllib.loads(root.read_text(encoding="utf-8"))["project"]["requires-python"]
    assert pyproject["project"]["requires-python"] == expected


def test_wheel_ships_every_engine_module_and_no_test_script(built, source_dir):
    wheel, _ = built
    files = _wheel_files(wheel)
    hook = _hook_module()
    for module in sorted(source_dir.glob("*.py")):
        shipped = f"vibeqc_naming/{module.name}" in files
        assert shipped != hook.is_test_script(module.name), module.name
    assert "vibeqc_naming/__init__.py" in files
    assert "vibeqc_naming/iupac_name.py" in files
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in files)
    assert files == sorted(f"vibeqc_naming/{rel.as_posix()}" for rel in hook.shipped_files(source_dir))


def test_wheel_metadata(built, pyproject):
    wheel, _ = built
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith("/METADATA"))).decode()
        wheel_file = archive.read(next(n for n in archive.namelist() if n.endswith("/WHEEL"))).decode()
    assert "Name: vibeqc-naming" in metadata
    assert f"Version: {pyproject['project']['version']}" in metadata
    assert "Requires-Dist: numpy>=1.26" in metadata
    assert "Requires-Dist: spglib>=2.0; extra == 'solids'" in metadata
    assert "Root-Is-Purelib: true" in wheel_file
    assert "py3-none-any" in wheel.name


def test_sdist_carries_the_source_and_rebuilds_the_same_wheel(built, tmp_path):
    wheel, sdist = built
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        archive.extractall(tmp_path, filter="data")
    assert any(name.endswith("/src/vibeqc_naming/__init__.py") for name in names)
    assert any(name.endswith("/hatch_build.py") for name in names)
    assert not any("_smoke_test" in name or "/_test_" in name for name in names)
    [extracted] = [p for p in tmp_path.iterdir() if p.is_dir()]
    rebuilt = _build("wheel", extracted, tmp_path / "rebuilt")
    assert _wheel_files(rebuilt) == _wheel_files(wheel)


def test_editable_wheel_points_at_the_checkout_source(tmp_path, monkeypatch):
    hatchling_build = pytest.importorskip("hatchling.build")
    if not CHECKOUT_SOURCE.is_dir():
        pytest.skip("not inside a vibe-qc checkout")
    monkeypatch.chdir(DIST_ROOT)
    name = hatchling_build.build_editable(str(tmp_path))
    with zipfile.ZipFile(tmp_path / name) as archive:
        files = archive.namelist()
        [pth] = [n for n in files if n.endswith(".pth")]
        entries = archive.read(pth).decode().split()
    assert [Path(e).resolve() for e in entries] == [CHECKOUT_SOURCE.parent.resolve()]
    assert not any(n.startswith("vibeqc_naming/") for n in files)
