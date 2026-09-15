"""basis_library/ + LIBINT_DATA_PATH plumbing.

Since the wheel-bundled-basis-library fix, the basis library lives
*inside* the Python package at ``python/vibeqc/basis_library/``, not
at the repo root — and since the populate decoupling (c392f39d), a
build-time overlay at ``build/basis_library/`` is preferred over the
bundled copy when present. ``vibeqc/__init__.py``'s
``_install_basis_library()`` documents the full resolution order;
tests here verify that documented contract, not one branch of it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import vibeqc as _vq
from vibeqc import Atom, BasisSet, Molecule

# Candidate dirs per the resolution order in ``_install_basis_library``
# (vibeqc/__init__.py), derived from the *imported* package location —
# not this test file's checkout — so expectations stay consistent with
# whatever tree the editable-install finder actually resolved to.
PACKAGE_DIR = Path(_vq.__file__).resolve().parent
PACKAGE_ROOT = PACKAGE_DIR.parent.parent
# Build-time overlay written by scripts/setup_basis_library.sh
# (gitignored; present on dev boxes, absent on fresh clones + wheels).
BUILD_OVERLAY_DIR = PACKAGE_ROOT / "build" / "basis_library"
# Path the package bundles in the wheel + uses when no overlay exists.
BUNDLED_BASIS_DIR = PACKAGE_DIR / "basis_library"
# Legacy repo-root layout, kept for backwards compatibility.
LEGACY_BASIS_DIR = PACKAGE_ROOT / "basis_library"


def test_libint_data_path_points_at_our_library():
    """LIBINT_DATA_PATH follows the documented resolution order.

    ``_install_basis_library`` prefers the gitignored build-time
    overlay over the bundled copy when the overlay exists, so the
    expected path is environment-dependent: overlay on dev boxes that
    ran ``scripts/setup_basis_library.sh``, bundled everywhere else.
    Asserting the bundled path unconditionally (this test pre-fix)
    phantom-reds on any checkout where the overlay is populated.
    """
    actual = Path(os.environ["LIBINT_DATA_PATH"]).resolve()
    candidates = {
        BUILD_OVERLAY_DIR.resolve(),
        BUNDLED_BASIS_DIR.resolve(),
        LEGACY_BASIS_DIR.resolve(),
    }
    if actual not in candidates:
        # Branch 1 of the resolution order: a pre-set LIBINT_DATA_PATH
        # is the user's override and wins, so the in-repo expectation
        # below doesn't apply. (The resolver itself only ever picks one
        # of the three candidates, so an out-of-tree value cannot be a
        # resolver bug.)
        pytest.skip(f"LIBINT_DATA_PATH user-pinned to {actual} (branch 1: user wins)")
    expected = (
        BUILD_OVERLAY_DIR
        if _vq._basis_library_has_standard_payload(BUILD_OVERLAY_DIR)
        else BUNDLED_BASIS_DIR
    )
    assert actual == expected.resolve(), (
        f"LIBINT_DATA_PATH = {actual}, expected {expected.resolve()} "
        "([...]/build/basis_library complete: "
        f"{_vq._basis_library_has_standard_payload(BUILD_OVERLAY_DIR)})"
    )


def test_basis_library_resolution_order(tmp_path, monkeypatch):
    """Pin every branch of the documented resolution order deterministically.

    Branches 2–4 (build overlay > bundled > legacy > nothing) via the
    extracted ``_resolve_basis_library`` against a synthetic repo
    layout; branch 1 (pre-set env var wins) via
    ``_install_basis_library`` itself.
    """
    package_dir = tmp_path / "python" / "vibeqc"
    overlay = tmp_path / "build" / "basis_library"
    bundled = package_dir / "basis_library"
    legacy = tmp_path / "basis_library"
    for candidate in (overlay, bundled, legacy):
        (candidate / "basis").mkdir(parents=True)
        for name in ("def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94"):
            (candidate / "basis" / name).write_text("canary\n")

    # 2. Build overlay beats the bundled + legacy copies ...
    assert _vq._resolve_basis_library(package_dir) == overlay
    # ... but a stale overlay that predates a newly bundled custom basis
    # must not shadow the committed package payload.
    (overlay / "basis" / "6-311+g3df2p.g94").unlink()
    assert _vq._resolve_basis_library(package_dir) == bundled
    (overlay / "basis" / "6-311+g3df2p.g94").write_text("canary\n")
    (overlay / "basis" / "pob-tzvp-rev2.g94").unlink()
    assert _vq._resolve_basis_library(package_dir) == bundled
    # It counts only with the standard payload, not as a bare dir or
    # stock-only libint directory.
    (overlay / "basis" / "def2-svp.g94").unlink()
    # 3. Bundled copy beats legacy.
    assert _vq._resolve_basis_library(package_dir) == bundled
    # 4. Legacy repo-root layout as last resort.
    for name in ("def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94"):
        (bundled / "basis" / name).unlink()
    (bundled / "basis").rmdir()
    assert _vq._resolve_basis_library(package_dir) == legacy
    # No candidate at all -> None (the installer then leaves env alone).
    for name in ("def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94"):
        (legacy / "basis" / name).unlink()
    (legacy / "basis").rmdir()
    assert _vq._resolve_basis_library(package_dir) is None

    # 1. A pre-set complete LIBINT_DATA_PATH wins: the installer must not
    # touch a user-provided full basis library.
    sentinel_root = tmp_path / "user-pinned-elsewhere"
    (sentinel_root / "basis").mkdir(parents=True)
    for name in ("def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94"):
        (sentinel_root / "basis" / name).write_text("sentinel\n")
    sentinel = str(sentinel_root)
    monkeypatch.setenv("LIBINT_DATA_PATH", sentinel)
    _vq._install_basis_library()
    assert os.environ["LIBINT_DATA_PATH"] == sentinel


def test_incomplete_libint_data_path_falls_back_to_bundled(tmp_path, monkeypatch):
    """A stock/incomplete libint data dir must not shadow bundled def2 sets."""
    incomplete = tmp_path / "libint-stock"
    (incomplete / "basis").mkdir(parents=True)
    (incomplete / "basis" / "sto-3g.g94").write_text("stock-only\n")
    monkeypatch.setenv("LIBINT_DATA_PATH", str(incomplete))

    _vq._install_basis_library()

    assert Path(os.environ["LIBINT_DATA_PATH"]).resolve() != incomplete.resolve()
    assert _vq._basis_library_has_standard_payload(os.environ["LIBINT_DATA_PATH"])


def test_stale_libint_data_path_without_pob_falls_back_to_bundled(
    tmp_path,
    monkeypatch,
):
    """A release overlay without POB must not shadow the bundled POB basis."""
    stale = tmp_path / "stale-release-overlay"
    (stale / "basis").mkdir(parents=True)
    (stale / "basis" / "def2-svp.g94").write_text("old release canary\n")
    (stale / "basis" / "6-311+g3df2p.g94").write_text("old release canary\n")
    monkeypatch.setenv("LIBINT_DATA_PATH", str(stale))

    _vq._install_basis_library()

    assert Path(os.environ["LIBINT_DATA_PATH"]).resolve() != stale.resolve()
    assert (
        Path(os.environ["LIBINT_DATA_PATH"]) / "basis" / "pob-tzvp-rev2.g94"
    ).is_file()


def test_basis_library_has_standard_sets():
    basis_dir = BUNDLED_BASIS_DIR / "basis"
    assert basis_dir.is_dir(), (
        f"{basis_dir} missing — run ./scripts/setup_basis_library.sh"
    )
    for name in ("sto-3g", "6-31g", "cc-pvdz", "def2-tzvp", "pob-tzvp-rev2"):
        assert (basis_dir / f"{name}.g94").is_file(), (
            f"missing standard basis {name}.g94 in {basis_dir}"
        )


@pytest.mark.parametrize(
    "basis_name,expected_nbf_h2o",
    [
        ("sto-3g", 7),
        ("6-31g*", 18),  # O has d; H has no polarization in 6-31G*
        ("cc-pvdz", 24),
        ("def2-tzvp", 43),
    ],
)
def test_standard_basis_loads_from_library(basis_name, expected_nbf_h2o):
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.5, 0.3]),
            Atom(1, [0.0, -0.5, 0.3]),
        ]
    )
    basis = BasisSet(mol, basis_name)
    assert basis.nbasis == expected_nbf_h2o


def test_custom_basis_override_is_picked_up(tmp_path, monkeypatch):
    """A custom .g94 dropped into a basis-library dir loads via libint.

    Mirrors the bundled library into a tmpdir, drops the fake basis
    into the tmpdir's ``basis/`` subdir, and overrides
    ``LIBINT_DATA_PATH`` so libint resolves there. **NEVER mutates the
    in-tree ``python/vibeqc/basis_library/``** — running this test
    against the live tree (i.e. invoking ``setup_basis_library.sh``)
    causes the committed ``basis/`` files to drift whenever ``custom/``
    diverges from ``basis/``, which is what bit b2e827e's "unintended
    Ne entry" earlier. The setup script is a build-time helper; the
    test exercises the libint-load half of the contract only.
    """
    import shutil

    # Mirror the bundled basis library into a private tmpdir tree.
    tmp_library = tmp_path / "basis_library"
    shutil.copytree(BUNDLED_BASIS_DIR, tmp_library)

    # Drop a one-atom H basis (single s function) directly into the
    # tmpdir's basis/ — libint looks there via LIBINT_DATA_PATH/basis/.
    fake_basis_name = "test-vibe-qc-fake"
    fake_file = tmp_library / "basis" / f"{fake_basis_name}.g94"
    fake_file.write_text(
        "!  test-vibe-qc-fake\n"
        "! H\n"
        "****\n"
        "H     0\n"
        "S   1   1.00\n"
        "      1.0000000              1.0000000\n"
        "****\n"
    )

    # Point libint at the tmpdir for this test only (monkeypatch undoes
    # the env-var change at teardown).
    monkeypatch.setenv("LIBINT_DATA_PATH", str(tmp_library))

    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = BasisSet(mol, fake_basis_name)
    assert basis.nbasis == 1
    assert basis.nshells == 1


def test_unknown_basis_name_raises_cleanly():
    """libint2 silently returns an empty BasisSet when no matching .g94 is
    found. vibe-qc must reject that up front, otherwise downstream integral
    evaluators segfault (observed with e.g. '6-311++g**' which ships under
    a different name). Regression guard for that crash."""
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ]
    )
    # '6-311++g**' now ships in our library (BSE fetch); use a genuinely missing name.
    with pytest.raises(RuntimeError, match="no shells loaded"):
        BasisSet(mol, "not-a-real-basis-name")
    # Completely fabricated name — same failure mode, same clean error.
    with pytest.raises(RuntimeError, match="no shells loaded"):
        BasisSet(mol, "completely-fabricated-basis")
