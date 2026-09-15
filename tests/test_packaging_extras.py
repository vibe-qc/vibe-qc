"""Packaging-extras coherence (audit finding 5).

The audit found that optional extras such as ``viewer-gpu`` are not
smoke-tested and that what the docs claim (``pip install -e
'.[viewer-gpu]'``) is not necessarily what ``pyproject.toml`` ships.

These tests read the two relevant ``pyproject.toml`` files directly —
no network, no install — and check the metadata is internally
consistent.  In particular:

* The ``viewer-gpu`` extra is declared in the root ``pyproject.toml``,
  per ``docs/design_qvf_format.md`` § 2.2 and
  ``vibe-view/HANDOVER.md``.
* It points at the co-located ``vibeview`` distribution at
  ``vibe-view/pyproject.toml`` (or another resolvable source).

Where the declared metadata is currently out of step with what the
docs / handover claim, the test is marked ``xfail(strict=True)`` so
the gap is explicit and a fix flips it green automatically.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_pyproject(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def root_pyproject() -> dict:
    return _load_pyproject(REPO_ROOT / "pyproject.toml")


@pytest.fixture(scope="module")
def vibe_view_pyproject() -> dict:
    p = REPO_ROOT / "vibe-view" / "pyproject.toml"
    if not p.exists():
        pytest.skip(
            "vibe-view/pyproject.toml not present — viewer-gpu extra "
            "coherence cannot be checked in this checkout."
        )
    return _load_pyproject(p)


class TestRootExtrasMetadata:
    """Static checks on the root pyproject extras table."""

    def test_extras_table_present(self, root_pyproject):
        proj = root_pyproject.get("project", {})
        extras = proj.get("optional-dependencies", {})
        # We rely on the historical extras to keep the docs honest.
        for required in ("test", "dev", "docs", "viewer"):
            assert required in extras, (
                f"root pyproject.toml is missing the documented "
                f"`{required}` extra (see docs/installation.md)."
            )

    def test_viewer_extra_points_at_moltui(self, root_pyproject):
        """``viewer`` extra is the terminal-side viewer (moltui).

        Distinct from the 3D ``viewer-gpu`` extra below — both are
        documented as alternative installs.
        """
        extras = root_pyproject["project"]["optional-dependencies"]
        viewer = extras["viewer"]
        assert any("moltui" in dep.lower() for dep in viewer), (
            f"`viewer` extra should pull in moltui (terminal viewer); "
            f"got {viewer!r}"
        )


class TestViewerGpuExtraIsDeclared:
    """The ``viewer-gpu`` extra is documented in
    ``docs/design_qvf_format.md`` § 2.2 and ``vibe-view/HANDOVER.md``.

    History: briefly removed in ``acd681c7`` (a plain ``"vibeview"``
    string can't resolve the co-located sibling with no source);
    restored in ``f0593ab6`` as a PEP 508 direct URL (``vibeview @
    file:./vibe-view/``). That relative URL then broke ``uv pip
    install`` entirely — uv won't parse a relative path out of the
    built-wheel metadata, and it reads every extra's Requires-Dist, so
    even the base no-extras install failed. The current form is a
    ``vibeview[viewer]>=0.1`` pin redirected to the co-located checkout via
    ``[tool.uv.sources]`` — uv resolves it locally; pip uses the
    two-step ``pip install -e . && pip install -e 'vibe-view[viewer]'``. See
    CHANGELOG ``[Unreleased]`` "uv installability".
    """

    def test_viewer_gpu_extra_present(self, root_pyproject):
        extras = root_pyproject["project"]["optional-dependencies"]
        assert "viewer-gpu" in extras, (
            "expected `viewer-gpu` extra in root pyproject.toml; "
            "found extras: " + ", ".join(sorted(extras))
        )

    def test_viewer_gpu_extra_resolves_to_vibeview(self, root_pyproject):
        extras = root_pyproject["project"]["optional-dependencies"]
        viewer_gpu = extras["viewer-gpu"]
        joined = " ".join(viewer_gpu).lower()
        assert "vibeview" in joined or "vibe-view" in joined, (
            f"viewer-gpu extra should pull in `vibeview` "
            f"(co-located at vibe-view/); got {viewer_gpu!r}"
        )

    def test_viewer_gpu_extra_uses_colocated_subproject(
        self, root_pyproject
    ):
        """The extra must resolve to the co-located sibling, not a
        (non-existent) PyPI release of ``vibeview``.

        Until both vibe-qc and vibe-view publish to PyPI, the co-located
        path is wired through ``[tool.uv.sources]`` (uv — vibe-qc's
        stated installer — resolves ``vibeview`` from ``vibe-view/``).
        The extra string itself must NOT carry a relative ``file:``
        direct-URL: uv refuses to parse a relative path out of the
        built-wheel metadata, and because it reads *every* extra's
        Requires-Dist that one URL broke even the base ``uv pip
        install``. See CHANGELOG ``[Unreleased]`` "uv installability".
        """
        extras = root_pyproject["project"]["optional-dependencies"]
        viewer_gpu = extras["viewer-gpu"]
        joined = " ".join(viewer_gpu).lower()

        # No relative file: direct-URL (the bug this structure fixes).
        assert "file:" not in joined, (
            "viewer-gpu must not use a relative file: direct-URL — it "
            "breaks `uv pip install`. Use a pin + [tool.uv.sources]. "
            "Got: " + repr(viewer_gpu)
        )

        # The co-located checkout path is wired via [tool.uv.sources].
        uv_sources = (
            root_pyproject.get("tool", {}).get("uv", {}).get("sources", {})
        )
        assert "vibeview" in uv_sources, (
            "viewer-gpu pin must be redirected to the co-located "
            "vibe-view/ sibling via [tool.uv.sources] until vibeview "
            "ships to PyPI. Got sources: " + repr(uv_sources)
        )
        assert uv_sources["vibeview"].get("path") == "vibe-view", (
            "[tool.uv.sources].vibeview must point at the co-located "
            "vibe-view/ checkout path. Got: "
            + repr(uv_sources.get("vibeview"))
        )


class TestViewerGpuCoherenceWithSubproject:
    """Cross-check root extra vs co-located vibe-view package metadata.

    Even before the extra is wired in the root, we can assert that the
    co-located ``vibe-view/pyproject.toml`` is shaped so the extra can
    plausibly point at it: same project name, same Python floor, MPL
    licence to match vibe-qc's redistribution model (§ 1 of
    CLAUDE.md).
    """

    def test_subproject_name(self, vibe_view_pyproject):
        assert vibe_view_pyproject["project"]["name"] == "vibeview"

    def test_subproject_python_floor_matches_root(
        self, root_pyproject, vibe_view_pyproject
    ):
        root_floor = root_pyproject["project"].get("requires-python", "")
        sub_floor = vibe_view_pyproject["project"].get("requires-python", "")
        # Both must require ≥ 3.11 (vibe-qc's published floor).
        assert "3.11" in root_floor
        assert "3.11" in sub_floor, (
            f"vibe-view requires-python ({sub_floor!r}) drifted from "
            f"vibe-qc root ({root_floor!r}); viewer-gpu install will "
            "fail on the older interpreters vibe-qc still supports."
        )

    def test_subproject_license_is_mpl(self, vibe_view_pyproject):
        license_field = vibe_view_pyproject["project"].get("license", "")
        # `license` can be a string or a {text=...} table in PEP 621.
        if isinstance(license_field, dict):
            license_field = license_field.get("text", "")
        assert "MPL" in str(license_field).upper(), (
            f"vibe-view license drifted from MPL-2.0 "
            f"(vibe-qc's bundled-data discipline; CLAUDE.md § 1); "
            f"got {license_field!r}"
        )


@pytest.fixture(scope="module")
def vibe_basis_pyproject() -> dict:
    p = REPO_ROOT / "vibe-basis" / "pyproject.toml"
    if not p.exists():
        pytest.skip(
            "vibe-basis/pyproject.toml not present — basisopt extra "
            "coherence cannot be checked in this checkout."
        )
    return _load_pyproject(p)


# The six modules under ``vibeqc.basis_optimization`` that import
# ``vibe_basis`` at module top level. They ship inside the vibe-qc wheel
# (`[tool.scikit-build] wheel.packages`), so without the ``basisopt``
# extra they are unimportable wherever vibe-qc is installed — which was
# true of all 40 local venvs when this was audited on 2026-07-26.
VIBE_BASIS_DEPENDENT_MODULES = (
    "vibeqc.basis_optimization.calculators",
    "vibeqc.basis_optimization.recipes.crystal_objective",
    "vibeqc.basis_optimization.recipes.crystal_stage1",
    "vibeqc.basis_optimization.recipes.crystal_stage2",
    "vibeqc.basis_optimization.recipes.crystal_stage3",
    "vibeqc.basis_optimization.recipes.production",
)


class TestBasisoptExtraIsDeclared:
    """The ``basisopt`` extra makes ``vibe_basis`` installable.

    Decision recorded in
    ``handovers/HANDOVER_VIBE_BASIS_DEPLOYMENT.md`` (2026-07-26):
    vibe-basis is **not** a managed vq program and gets no fleet rollout
    lane — it is a client that submits *to* the fleet (it shells out to
    the ``vq`` CLI and never imports it) rather than a payload that runs
    on it. What it does get is this extra, so its version is verifiable
    wherever vibe-qc is installed instead of merely present as source.

    Same structural trap as ``viewer-gpu`` above: the extra must be an
    ordinary pin redirected through ``[tool.uv.sources]``, never a
    relative ``file:`` direct URL.
    """

    def test_basisopt_extra_present(self, root_pyproject):
        extras = root_pyproject["project"]["optional-dependencies"]
        assert "basisopt" in extras, (
            "expected `basisopt` extra in root pyproject.toml; "
            "found extras: " + ", ".join(sorted(extras))
        )

    def test_basisopt_extra_resolves_to_vibe_basis(self, root_pyproject):
        basisopt = root_pyproject["project"]["optional-dependencies"]["basisopt"]
        joined = " ".join(basisopt).lower()
        assert "vibe-basis" in joined or "vibe_basis" in joined, (
            f"basisopt extra should pull in `vibe-basis` "
            f"(co-located at vibe-basis/); got {basisopt!r}"
        )

    def test_basisopt_extra_uses_colocated_subproject(self, root_pyproject):
        basisopt = root_pyproject["project"]["optional-dependencies"]["basisopt"]
        joined = " ".join(basisopt).lower()

        assert "file:" not in joined, (
            "basisopt must not use a relative file: direct-URL — uv reads "
            "every extra's Requires-Dist, so one unparseable relative URL "
            "breaks even the base install (the viewer-gpu regression). "
            "Use a pin + [tool.uv.sources]. Got: " + repr(basisopt)
        )

        uv_sources = root_pyproject.get("tool", {}).get("uv", {}).get("sources", {})
        assert "vibe-basis" in uv_sources, (
            "basisopt pin must be redirected to the co-located "
            "vibe-basis/ sibling via [tool.uv.sources] until vibe-basis "
            "ships to PyPI. Got sources: " + repr(uv_sources)
        )
        assert uv_sources["vibe-basis"].get("path") == "vibe-basis", (
            "[tool.uv.sources].vibe-basis must point at the co-located "
            "vibe-basis/ checkout path. Got: "
            + repr(uv_sources.get("vibe-basis"))
        )

    def test_basisopt_extra_stays_light(self, root_pyproject):
        """CLAUDE.md § 9 — the heavy optimizers stay behind vibe-basis's
        own extras (``nlopt`` / ``iminuit`` / ``scipy`` / ``vq``).

        The bare ``vibe-basis`` pin adds only click + numpy, both already
        vibe-qc dependencies. Requesting one of its heavy extras from
        here would silently widen vibe-qc's install footprint.
        """
        basisopt = root_pyproject["project"]["optional-dependencies"]["basisopt"]
        joined = " ".join(basisopt).lower()
        for heavy in ("nlopt", "iminuit", "[all]"):
            assert heavy not in joined, (
                f"basisopt must not pull vibe-basis's `{heavy}` extra — "
                "keep the footprint minimal per CLAUDE.md § 9. "
                f"Got {basisopt!r}"
            )


class TestBasisoptCoherenceWithSubproject:
    """Cross-check the root extra against vibe-basis's own metadata."""

    def test_subproject_name(self, vibe_basis_pyproject):
        assert vibe_basis_pyproject["project"]["name"] == "vibe-basis"

    def test_subproject_python_floor_matches_root(
        self, root_pyproject, vibe_basis_pyproject
    ):
        root_floor = root_pyproject["project"].get("requires-python", "")
        sub_floor = vibe_basis_pyproject["project"].get("requires-python", "")
        assert "3.11" in root_floor
        assert "3.11" in sub_floor, (
            f"vibe-basis requires-python ({sub_floor!r}) is stricter than "
            f"vibe-qc root ({root_floor!r}); `.[basisopt]` becomes "
            "unresolvable on interpreters vibe-qc still supports."
        )

    def test_subproject_license_is_mpl(self, vibe_basis_pyproject):
        license_field = vibe_basis_pyproject["project"].get("license", "")
        if isinstance(license_field, dict):
            license_field = license_field.get("text", "")
        assert "MPL" in str(license_field).upper(), (
            f"vibe-basis license drifted from MPL-2.0 "
            f"(CLAUDE.md § 1); got {license_field!r}"
        )

    def test_subproject_declares_its_own_version_line(
        self, root_pyproject, vibe_basis_pyproject
    ):
        """vibe-basis versions independently of vibe-qc.

        See ``vibe-basis/VERSIONING.md``. The number reaches the release
        record through ``make_release_report.py``'s
        ``sibling_versions_at_release.vibe_basis``, which reads this
        field at the peeled release commit — so it must exist and must
        not simply mirror vibe-qc's.
        """
        sub_version = vibe_basis_pyproject["project"]["version"]
        root_version = root_pyproject["project"].get("version")
        assert sub_version, "vibe-basis must declare a version"
        if root_version:
            assert sub_version != root_version, (
                "vibe-basis must carry its own version line, not vibe-qc's "
                f"({sub_version!r}); see vibe-basis/VERSIONING.md."
            )

    def test_versioning_doc_present(self):
        doc = REPO_ROOT / "vibe-basis" / "VERSIONING.md"
        assert doc.exists(), (
            "vibe-basis/VERSIONING.md is the scheme's source of truth for "
            "the independent sibling version line."
        )


class TestBasisOptimizationModulesImport:
    """The point of the extra: these six become importable.

    Skipped where ``vibe_basis`` is absent — a checkout installed
    without ``[basisopt]`` is a supported configuration, and this suite
    must not demand the extra. Where the extra *is* installed, a
    regression in the wiring fails here rather than at the first real
    optimization run.
    """

    def test_vibe_basis_importable_or_skip(self):
        pytest.importorskip(
            "vibe_basis",
            reason="install with `uv pip install -e '.[basisopt]'`",
        )

    @pytest.mark.parametrize("module", VIBE_BASIS_DEPENDENT_MODULES)
    def test_module_imports(self, module):
        pytest.importorskip(
            "vibe_basis",
            reason="install with `uv pip install -e '.[basisopt]'`",
        )
        import importlib

        importlib.import_module(module)
