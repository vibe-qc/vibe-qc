"""Packaging metadata smoke tests for optional extras.

These tests inspect local project metadata only. They do not install
packages and never need network access.
"""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _load_pyproject(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _requirement_name(requirement: str) -> str:
    head = requirement.split(";", 1)[0].strip()
    for sep in ("[", "<", ">", "=", "!", "~"):
        head = head.split(sep, 1)[0].strip()
    return head.lower()


def test_base_runtime_declares_ase_for_semiempirical_optimization() -> None:
    pyproject = _load_pyproject(ROOT / "pyproject.toml")
    dependencies = {
        _requirement_name(req) for req in pyproject["project"]["dependencies"]
    }

    assert "ase" in dependencies, (
        "Base release installs must include ASE because "
        "run_job(optimize=True) routes semiempirical/MLIP methods through "
        "ASE, while fleet release deployments install with --extras none."
    )


def test_vibe_view_project_metadata_matches_gpu_viewer_contract() -> None:
    pyproject = _load_pyproject(ROOT / "vibe-view" / "pyproject.toml")
    project = pyproject["project"]

    assert project["name"] == "vibeview"
    assert project["requires-python"] == ">=3.11"
    assert project["scripts"]["vibe-view"] == "vibeview.cli:main"

    dependencies = {dep.split(">=", 1)[0].lower() for dep in project["dependencies"]}
    assert "pyvista" in dependencies
    assert not {"trame", "trame-vtk", "trame-vuetify"} & dependencies

    viewer = {
        dep.split(">=", 1)[0].lower()
        for dep in project["optional-dependencies"]["viewer"]
    }
    assert {"trame", "trame-vtk", "trame-vuetify", "uvicorn"}.issubset(viewer)


def test_viewer_extra_is_terminal_viewer_not_vibe_view_gpu() -> None:
    pyproject = _load_pyproject(ROOT / "pyproject.toml")
    extras = pyproject["project"]["optional-dependencies"]

    viewer = extras["viewer"]
    assert any(req.lower().startswith("moltui") for req in viewer)
    assert not any("vibeview" in req.lower() for req in viewer)


def test_test_and_dev_extras_cover_bundled_ml_predictor() -> None:
    pyproject = _load_pyproject(ROOT / "pyproject.toml")
    extras = pyproject["project"]["optional-dependencies"]
    for extra in ("test", "dev"):
        assert any(
            req.lower().startswith("scikit-learn") for req in extras[extra]
        ), f".[{extra}] cannot run tests/test_ml_kpredictor.py"


def test_sdist_excludes_repository_scratch_tree() -> None:
    pyproject = _load_pyproject(ROOT / "pyproject.toml")
    sdist = pyproject["tool"]["scikit-build"]["sdist"]

    assert "tmp/**" in sdist["exclude"], (
        "scikit-build-core includes unignored, untracked files by default; "
        "the repository-root tmp/ scratch tree must be excluded explicitly"
    )
    assert not any(
        pattern.lstrip("/").startswith("tmp/")
        for pattern in sdist.get("include", ())
    ), "sdist.include takes precedence over sdist.exclude"


def test_viewer_gpu_extra_declares_co_located_vibe_view_when_restored() -> None:
    pyproject = _load_pyproject(ROOT / "pyproject.toml")
    extras = pyproject["project"]["optional-dependencies"]
    vibe_view_project = _load_pyproject(ROOT / "vibe-view" / "pyproject.toml")
    vibe_view_name = vibe_view_project["project"]["name"]  # "vibeview"

    assert "viewer-gpu" in extras

    viewer_gpu = extras["viewer-gpu"]
    # The extra declares an ordinary version pin for the vibe-view
    # distribution ("vibeview"); the co-located source path is wired
    # separately in [tool.uv.sources] so uv resolves it from the
    # checkout pre-PyPI.
    assert any(req.lower().startswith(vibe_view_name) for req in viewer_gpu)

    # Regression guard (the bug this structure fixes): the pin must NOT
    # be a relative ``file:`` direct-URL. uv refuses to parse a relative
    # path out of the built-wheel metadata ("relative path without a
    # working directory"), and because uv reads *every* extra's
    # Requires-Dist that one bad URL broke even the base
    # ``uv pip install`` / ``uv pip install -e .`` (no extras requested).
    assert not any("file:" in req.lower() for req in viewer_gpu), (
        "viewer-gpu must not use a relative file: direct-URL — it breaks "
        "`uv pip install`. Use a version pin + [tool.uv.sources] path."
    )

    # The co-located checkout path is wired for uv (vibe-qc's stated
    # installer) via [tool.uv.sources]. Drop this entry once vibeview is
    # published to PyPI and the pin resolves from PyPI unchanged.
    uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    assert vibe_view_name in uv_sources, (
        "[tool.uv.sources] must redirect the vibeview pin to the "
        "co-located vibe-view/ checkout path."
    )
    assert uv_sources[vibe_view_name].get("path") == "vibe-view"
