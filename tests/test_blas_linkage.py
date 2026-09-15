"""Verify the BLAS+LAPACK linkage and Eigen-delegation flags.

Without a linked BLAS, every Eigen dgemm/dsyev/dpotrf runs Eigen's
generic-C++ kernels — a several × slowdown at SCF size. This test
pins the contract that:

  1. ``vibeqc._vibeqc_core.blas_info()`` exists and returns the
     three-key dict ``{libraries, blas_enabled, lapacke_enabled}``.
  2. The default build (``VIBEQC_USE_BLAS=ON``, no override) actually
     succeeds in finding *some* BLAS on every supported platform.
     macOS gets Accelerate, Linux gets either OpenBLAS or netlib —
     either way ``blas_enabled`` must be True. CI failures here
     surface a misconfigured FindBLAS path before they show up as a
     silent 5× slowdown.
  3. The Python banner picks up the linkage info and surfaces a
     short ``blas <backend>`` label on its "linked:" line — so the
     SCF log of any persisted calculation records whether it ran on
     a real BLAS or the Eigen-generic path.
  4. The default-thread setdefault behaviour in ``vibeqc/__init__.py``
     is robust: importing vibeqc never *overrides* a user-set
     ``OPENBLAS_NUM_THREADS``, and on a clean shell it pins the BLAS
     to single-threaded so it doesn't fight vibe-qc's OpenMP.

These checks are deliberately tolerant of which specific BLAS is
linked — the contract is "some BLAS is linked", not "Accelerate"
or "OpenBLAS" specifically.
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# blas_info() binding
# ---------------------------------------------------------------------------

def test_blas_info_binding_exists():
    """The C++ binding for build-time BLAS linkage info must be present."""
    from vibeqc import _vibeqc_core

    assert hasattr(_vibeqc_core, "blas_info"), (
        "vibeqc._vibeqc_core.blas_info() missing — the BLAS+LAPACK "
        "linkage commit added a pybind11 binding that reads "
        "build_config.hpp; missing here means the C extension was "
        "built against a stale tree (rebuild needed)."
    )

    info = _vibeqc_core.blas_info()
    assert set(info.keys()) >= {
        "libraries", "blas_enabled", "lapacke_enabled"
    }, f"blas_info() must return libraries/blas_enabled/lapacke_enabled; got {info}"
    assert isinstance(info["libraries"], str)
    assert isinstance(info["blas_enabled"], bool)
    assert isinstance(info["lapacke_enabled"], bool)


def test_blas_is_actually_linked_by_default():
    """The default build (VIBEQC_USE_BLAS=ON, no vendor pin) must succeed
    in linking some BLAS on every supported platform.

    macOS provides Apple Accelerate via the OS; Linux provides at least
    reference netlib BLAS on every distro we test against. If this fails
    in CI, the FindBLAS path is broken and we'd silently ship a build
    with Eigen-generic kernels — exactly the regression this commit
    set out to prevent.
    """
    from vibeqc import _vibeqc_core

    info = _vibeqc_core.blas_info()
    if not info["blas_enabled"]:
        pytest.fail(
            "EIGEN_USE_BLAS was NOT set at build time — the C extension "
            "was built without a linked BLAS, so all dense matrix "
            "operations run Eigen's generic kernels (much slower at "
            "SCF size). Check the CMake configure log for the "
            f"'vibe-qc: BLAS_LIBRARIES = ...' line. Raw info: {info}"
        )
    assert info["libraries"], (
        "blas_enabled=True but libraries string is empty — inconsistent "
        "build_config.hpp; rebuild the C extension."
    )


# ---------------------------------------------------------------------------
# banner integration
# ---------------------------------------------------------------------------

def test_library_versions_includes_blas_key():
    """library_versions() should always include a 'blas' key. The value
    is one of: a backend label (Accelerate / OpenBLAS / MKL / netlib
    BLAS / …, optionally suffixed " +LAPACKE"), the literal string
    'none' when VIBEQC_USE_BLAS=OFF, or 'unknown' when the probe fails.
    """
    from vibeqc.banner import library_versions

    versions = library_versions()
    assert "blas" in versions, (
        "library_versions() missing 'blas' key — banner won't surface "
        "BLAS linkage in the SCF log."
    )
    assert versions["blas"], "blas value must be non-empty"


def test_banner_renders_blas_line():
    """The banner's 'linked:' line must include the blas label so a
    persisted SCF log records whether the run used a real BLAS."""
    from vibeqc.banner import banner

    rendered = banner()
    assert "blas " in rendered, (
        "Banner output doesn't surface the BLAS backend label. "
        f"Got:\n{rendered}"
    )


def test_banner_blas_label_matches_linkage():
    """If the C extension reports blas_enabled=True, the banner label
    must not say 'none'. The label may say 'unknown' if the C extension
    failed to load, but for a build that successfully linked BLAS the
    banner must reflect that."""
    from vibeqc import _vibeqc_core
    from vibeqc.banner import library_versions

    info = _vibeqc_core.blas_info()
    label = library_versions()["blas"]

    if info["blas_enabled"]:
        assert label != "none", (
            f"blas_info reports blas_enabled=True but banner says 'none' — "
            f"label-mapping logic in banner._blas_backend_label is wrong "
            f"for libraries={info['libraries']!r}"
        )
        if info["lapacke_enabled"]:
            assert "+LAPACKE" in label, (
                f"lapacke_enabled=True but banner label missing "
                f"'+LAPACKE' marker: {label!r}"
            )


# ---------------------------------------------------------------------------
# Threading default
# ---------------------------------------------------------------------------

def test_init_pins_openblas_threads_to_one_by_default(monkeypatch):
    """A fresh shell with no OPENBLAS_NUM_THREADS export should see
    vibe-qc pin it to 1 to avoid nested oversubscription with its own
    OpenMP layer.

    We re-invoke the pinning helper after clearing the env vars so the
    behaviour is observable from an in-process test.
    """
    from vibeqc import _pin_blas_threads

    for key in (
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    _pin_blas_threads()

    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.environ.get("MKL_NUM_THREADS") == "1"
    assert os.environ.get("VECLIB_MAXIMUM_THREADS") == "1"
    assert os.environ.get("BLIS_NUM_THREADS") == "1"


def test_init_respects_user_set_openblas_threads(monkeypatch):
    """A user who explicitly exports OPENBLAS_NUM_THREADS=4 in their
    shell wants that — not a silent override. setdefault must respect
    the pre-existing value."""
    from vibeqc import _pin_blas_threads

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    _pin_blas_threads()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4", (
        "vibe-qc overrode a user-set OPENBLAS_NUM_THREADS — setdefault "
        "behaviour is broken."
    )
