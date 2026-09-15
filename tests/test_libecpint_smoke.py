"""Phase 14a smoke test: libecpint links into vibe-qc.

Confirms the vendored libecpint (built by ``scripts/build_libecpint.sh``
into ``third_party/libecpint/install/``) is reachable from the Python
extension and that constructing an ``ECPIntegrator`` doesn't crash.

The actual ECP integral computation API (matrix elements
``⟨χ_μ | V_ECP | χ_ν⟩``) lands in Phase 14b.

The :func:`vibeqc.banner.library_versions` integration (v0.8.0
banner / libecpint coupled fix, prep doc § 488-501) is verified
below to ensure the C++ accessor reaches the banner's "linked: ..."
line — guarding against a regression where the binding is dropped
but the defensive ``getattr`` fallback silently substitutes
``"unknown"``.
"""

from __future__ import annotations

import vibeqc._vibeqc_core as core
import math
import pytest


@pytest.mark.parametrize("exponent", [0.5, 10., 100., 10000.])
@pytest.mark.parametrize("center", [0., 0.3, 0.72])
def test_radial_quadrature_resolves_displaced_gaussians(exponent, center):
    value, converged = core._libecpint_gaussian_quadrature(4096, exponent, center)
    root = math.sqrt(exponent)
    exact = math.sqrt(math.pi / exponent) * 0.5 * (
        math.erf(root * (1 - center)) + math.erf(root * (1 + center))
    )
    assert converged
    assert abs(value - exact) < 1e-12


def test_radial_quadrature_refuses_unresolved_gaussian():
    _, converged = core._libecpint_gaussian_quadrature(32, 10000., 0.3)
    assert not converged


def test_radial_quadrature_never_certifies_nan():
    value, converged = core._libecpint_gaussian_quadrature(32, float('nan'), 0.3)
    assert math.isnan(value)
    assert not converged


def test_libecpint_version_string_nonempty():
    v = core.libecpint_version()
    assert isinstance(v, str)
    assert v
    # Pinned version reported by our vendor wrapper.
    assert "1.0.7" in v
    # Library compiled with the L_max we expect (matches the upstream
    # source build defaults).
    assert "MAX_L=5" in v


def test_library_versions_surfaces_libecpint():
    """``library_versions()`` must include a ``libecpint`` key whose
    value matches the C++ accessor — not the ``"unknown"`` fallback.

    The defensive ``getattr(_core, "libecpint_version", ...)`` in
    :func:`vibeqc.banner.library_versions` falls back to ``"unknown"``
    when the C++ binding is missing (older `_core` extension on a
    stale build). On a freshly-built extension the binding is
    present, so this test pins the "real version reaches the banner"
    contract — matches the libint / libxc / spglib test pattern
    in :mod:`tests.test_blas_linkage`.
    """
    from vibeqc.banner import library_versions

    versions = library_versions()
    assert "libecpint" in versions, (
        "library_versions() missing 'libecpint' key — banner won't "
        "surface libecpint linkage on the 'linked: ...' line."
    )
    assert versions["libecpint"] != "unknown", (
        "library_versions()['libecpint'] resolved to 'unknown' — the "
        "C++ binding _vibeqc_core.libecpint_version() is missing or "
        "raising. Rebuild the extension against the current cpp/src "
        "tree (the binding was added in Phase 14a / commit fe49e4a)."
    )
    assert "1.0.7" in versions["libecpint"], (
        f"Unexpected libecpint version string: {versions['libecpint']!r}. "
        "Expected the vendored 1.0.7 build."
    )


def test_banner_renders_libecpint_line():
    """The banner's 'linked:' line must include the libecpint label
    so a persisted SCF log records the ECP-integral-library version.

    Mirrors :func:`tests.test_blas_linkage.test_banner_renders_blas_line`.
    """
    from vibeqc.banner import banner

    rendered = banner()
    assert "libecpint " in rendered, (
        "Banner output doesn't surface the libecpint version. "
        f"Got:\n{rendered}"
    )
