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


# GitLab #212: ``test_vibe_view_project_metadata_matches_gpu_viewer_contract``
# used to open ``vibe-view/pyproject.toml`` here and assert the viewer's own
# name, Python floor, console script and dependency split. The 2026-09-08
# split moved that package to its own repository, so the load raised
# FileNotFoundError before any assertion ran. Those assertions are not lost:
# vibe-view#23 ("Own the viewer packaging metadata checks after the repository
# split") moved them to project 35's own tests/test_packaging_metadata.py,
# which is the only place a change to the viewer's metadata can actually be
# caught. What belongs here is what vibe-qc itself controls: the shape of its
# own ``viewer-gpu`` extra, pinned below.


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


def test_viewer_gpu_extra_pins_the_independent_vibeview_distribution() -> None:
    """The viewer is an independent distribution, not a co-located subproject.

    GitLab #212: this used to require ``[tool.uv.sources].vibeview`` to
    redirect the pin at a ``vibe-view/`` directory inside this checkout. The
    2026-09-08 split removed that directory and ``pyproject.toml`` dropped the
    redirect deliberately ("vibe-view now lives in its own repository, so
    there is no co-located directory to redirect to"), so the test demanded a
    layout the repository is no longer supposed to have. It now pins the
    post-split contract, which also catches the redirect being reintroduced.
    """
    pyproject = _load_pyproject(ROOT / "pyproject.toml")
    extras = pyproject["project"]["optional-dependencies"]

    assert "viewer-gpu" in extras
    viewer_gpu = extras["viewer-gpu"]

    # An ordinary version pin on the published distribution name.
    assert any(req.lower().startswith("vibeview") for req in viewer_gpu), (
        "viewer-gpu must pin the vibeview distribution. Got: "
        + repr(viewer_gpu)
    )

    # Regression guard, and the reason this structure exists: the pin must
    # NOT be a relative ``file:`` direct-URL. uv refuses to parse a relative
    # path out of the built-wheel metadata ("relative path without a working
    # directory"), and because uv reads *every* extra's Requires-Dist that
    # one bad URL broke even the base ``uv pip install`` with no extras
    # requested.
    assert not any("file:" in req.lower() for req in viewer_gpu), (
        "viewer-gpu must not use a relative file: direct-URL -- it breaks "
        "`uv pip install`. Use a plain version pin. Got: " + repr(viewer_gpu)
    )

    # No source redirect may point at a co-located viewer checkout: there is
    # no vibe-view/ directory in this repository any more, so such an entry
    # would make uv resolve against a path that does not exist.
    uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    assert "vibeview" not in uv_sources, (
        "vibe-view lives in its own repository since the 2026-09-08 split, "
        "so [tool.uv.sources] must not redirect the pin to a co-located "
        "path. Got: " + repr(uv_sources)
    )
    assert not (ROOT / "vibe-view").exists(), (
        "a vibe-view/ directory reappeared in the core checkout; the viewer "
        "is an independent repository (project 35)"
    )
