"""FFTW3 banner-probe smoke test (M1 — GAPW track).

FFTW3 has been a required build dep since the FFT-Poisson long-range
Hartree solver landed, but its version was not surfaced in the banner
until the v0.10.x GAPW work picked up the linked-deps inventory per
CLAUDE.md § 6.

Mirrors :mod:`tests.test_libecpint_smoke` exactly — the
fftw3_version() C++ accessor must reach library_versions() and the
banner's "linked: ..." line without the defensive ``getattr``
fallback silently substituting ``"unknown"``.
"""

from __future__ import annotations

import re

import vibeqc._vibeqc_core as core


# FFTW reports versions like "3.3.10-sse2-avx" — major.minor.patch with
# an optional dash-separated suffix listing compile-time SIMD flags.
# This regex captures the leading semver triple.
_FFTW_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def test_fftw3_version_string_nonempty():
    v = core.fftw3_version()
    assert isinstance(v, str)
    assert v
    # Must start with a semver-shaped string — the C++ wrapper strips
    # the upstream "fftw-" prefix so the banner reads consistent with
    # the other libraries.
    m = _FFTW_SEMVER_RE.match(v)
    assert m, f"Unexpected fftw3 version string: {v!r}"
    # vibe-qc vendors FFTW 3.3.10 via scripts/build_fftw.sh; system
    # FFTW3 on most distros is also 3.3.x. Major version must be 3.
    assert int(m.group(1)) == 3, f"FFTW major != 3: {v!r}"


def test_library_versions_surfaces_fftw3():
    """``library_versions()`` must include a ``fftw3`` key whose value
    matches the C++ accessor — not the ``"unknown"`` fallback.

    Pins the contract that the C++ binding reaches the banner; the
    defensive ``getattr(_core, "fftw3_version", ...)`` would silently
    paper over a missing binding on a stale build.
    """
    from vibeqc.banner import library_versions

    versions = library_versions()
    assert "fftw3" in versions, (
        "library_versions() missing 'fftw3' key — banner won't surface "
        "FFTW3 linkage on the 'linked: ...' line."
    )
    assert versions["fftw3"] != "unknown", (
        "library_versions()['fftw3'] resolved to 'unknown' — the "
        "C++ binding _vibeqc_core.fftw3_version() is missing or "
        "raising. Rebuild the extension against the current cpp/src "
        "tree."
    )
    assert _FFTW_SEMVER_RE.match(versions["fftw3"]), (
        f"Unexpected fftw3 version string in library_versions: "
        f"{versions['fftw3']!r}."
    )


def test_banner_renders_fftw3_line():
    """The banner's 'linked:' line must include the fftw3 label so a
    persisted SCF log records the FFT-library version used by the
    Hartree solver.

    Mirrors :func:`tests.test_libecpint_smoke.test_banner_renders_libecpint_line`.
    """
    from vibeqc.banner import banner

    rendered = banner()
    assert "fftw3 " in rendered, (
        "Banner output doesn't surface the fftw3 version. "
        f"Got:\n{rendered}"
    )
