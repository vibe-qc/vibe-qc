"""Banner wiring for the optional dispersion backends (dftd3, dftd4).

Both packages ship as PyPI wheels with bundled binary libraries
(libdftd3 / libdftd4) that vibe-qc dynamically links at runtime via
ctypes/cffi inside ``vibeqc.compute_d3`` / ``vibeqc.compute_d4``.
CLAUDE.md § 6 mandates banner coverage for every linked native dep
— this test pins the contract that both versions surface in
``library_versions()`` and reach the rendered banner when the
optional ``[dispersion]`` extra is installed.

Mirrors the libecpint pattern in :mod:`tests.test_libecpint_smoke`
and the BLAS pattern in :mod:`tests.test_blas_linkage`.
"""

from __future__ import annotations

import importlib

import pytest


def _dftd3_installed() -> bool:
    try:
        importlib.import_module("dftd3")
        return True
    except ImportError:
        return False


def _dftd4_installed() -> bool:
    try:
        importlib.import_module("dftd4")
        return True
    except ImportError:
        return False


def test_library_versions_always_has_dispersion_keys():
    """``library_versions()`` must always include ``dftd3`` and
    ``dftd4`` keys, regardless of whether the optional ``[dispersion]``
    extra is installed. Values are either a version string or the
    sentinel ``"none"``.
    """
    from vibeqc.banner import library_versions

    versions = library_versions()
    assert "dftd3" in versions, (
        "library_versions() missing 'dftd3' key — banner won't "
        "surface the optional D3-BJ backend linkage."
    )
    assert "dftd4" in versions, (
        "library_versions() missing 'dftd4' key — banner won't "
        "surface the optional D4 dispersion backend linkage."
    )
    # When absent, key must be "none" (not "unknown" — that's reserved
    # for probe failures on installed packages).
    if not _dftd3_installed():
        assert versions["dftd3"] == "none", (
            f"dftd3 not installed but library_versions()['dftd3'] = "
            f"{versions['dftd3']!r}; expected 'none'."
        )
    if not _dftd4_installed():
        assert versions["dftd4"] == "none", (
            f"dftd4 not installed but library_versions()['dftd4'] = "
            f"{versions['dftd4']!r}; expected 'none'."
        )


@pytest.mark.skipif(not _dftd4_installed(),
                    reason="dftd4 not installed in this env")
def test_library_versions_reports_dftd4_version_when_installed():
    """When the optional ``dftd4`` package is importable, its version
    must reach ``library_versions()`` (matching the package's
    ``__version__`` attribute, not the ``"none"`` / ``"unknown"``
    fallbacks)."""
    import dftd4
    from vibeqc.banner import library_versions

    versions = library_versions()
    assert versions["dftd4"] == dftd4.__version__, (
        f"library_versions()['dftd4'] = {versions['dftd4']!r} but "
        f"dftd4.__version__ = {dftd4.__version__!r}"
    )


@pytest.mark.skipif(not _dftd3_installed(),
                    reason="dftd3 not installed in this env")
def test_library_versions_reports_dftd3_version_when_installed():
    """Same contract as ``dftd4`` above but for ``dftd3``."""
    import dftd3
    from vibeqc.banner import library_versions

    versions = library_versions()
    assert versions["dftd3"] == dftd3.__version__, (
        f"library_versions()['dftd3'] = {versions['dftd3']!r} but "
        f"dftd3.__version__ = {dftd3.__version__!r}"
    )


@pytest.mark.skipif(not (_dftd3_installed() or _dftd4_installed()),
                    reason="neither dftd3 nor dftd4 installed")
def test_banner_renders_dispersion_line_when_extra_installed():
    """When at least one of dftd3 / dftd4 is installed, the rendered
    banner must include the ``dispersion:`` line so a persisted SCF
    log captures which dispersion-backend versions were available
    at the run.
    """
    from vibeqc.banner import banner

    rendered = banner()
    assert "dispersion:" in rendered, (
        "Banner output missing 'dispersion:' line despite at least "
        f"one of dftd3/dftd4 being installed. Got:\n{rendered}"
    )
    if _dftd3_installed():
        assert "dftd3 " in rendered, (
            "Banner output missing the dftd3 version label. "
            f"Got:\n{rendered}"
        )
    if _dftd4_installed():
        assert "dftd4 " in rendered, (
            "Banner output missing the dftd4 version label. "
            f"Got:\n{rendered}"
        )
