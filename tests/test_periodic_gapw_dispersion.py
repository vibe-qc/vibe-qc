"""GAPW + D3-BJ dispersion wiring (v0.12-prep).

Covers the ``dispersion=`` / ``dispersion_functional=`` parameters on
both :func:`run_periodic_rhf_gpw` (single-k) and
:func:`run_periodic_rks_gpw_multi_k` (multi-k). The dispersion
correction is *system-only* (positions + species, not the SCF method),
so the test surface is light: backwards-compat (dispersion=None →
e_dispersion=0), correct sign + non-zero magnitude on H2, total
energy == SCF + dispersion, and the same on the multi-k path.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import PlaneWaveGrid


# Set to True if the dftd3 python bindings are not importable. The
# tests below then degrade to a single "NotImplementedError" expectation.
try:
    import dftd3  # noqa: F401
    _DFTD3_OK = True
except Exception:  # pragma: no cover
    _DFTD3_OK = False


# ---------- helpers (mirror test_periodic_gapw_j.py) ------------------


def _h2_periodic_system(L: float = 12.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    return sys


def _h2_basis(L: float = 12.0):
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    return vq.BasisSet(mol, "sto-3g")


# ---------- single-k path ---------------------------------------------


def test_rhf_gpw_no_dispersion_is_backwards_compatible():
    """``dispersion=None`` (the default) → e_dispersion=0.0 and
    the total energy equals the SCF energy with no add-on."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 12.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)

    r = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
    )
    assert r.converged
    assert r.e_dispersion == 0.0
    assert r.breakdown.e_dispersion == 0.0
    # No dispersion: total == breakdown.e_total == SCF.
    assert r.energy == pytest.approx(r.breakdown.e_total, abs=1e-12)


@pytest.mark.skipif(not _DFTD3_OK, reason="dftd3 bindings unavailable")
def test_rhf_gpw_with_d3bj_pbe_is_nonzero_negative():
    """``dispersion='d3-bj'`` + ``dispersion_functional='pbe'`` on H2
    yields a non-zero negative dispersion correction; the SCF total
    equals (SCF energy without dispersion) + e_dispersion."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 12.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)

    r0 = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30,
        functional="pbe", quiet=True,
    )
    r1 = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30,
        functional="pbe",
        dispersion="d3-bj", dispersion_functional="pbe",
        quiet=True,
    )
    assert r0.converged and r1.converged
    assert r1.e_dispersion < 0.0
    assert abs(r1.e_dispersion) > 1e-7  # non-trivial magnitude
    # SCF energy is identical (dispersion does not affect the Fock).
    # The total energy with dispersion equals (SCF total) + e_disp.
    assert r1.energy == pytest.approx(
        r0.energy + r1.e_dispersion, abs=1e-10
    )
    # breakdown.e_total reflects the added dispersion too.
    assert r1.breakdown.e_total == pytest.approx(
        r0.breakdown.e_total + r1.e_dispersion, abs=1e-10
    )


# ---------- multi-k path ----------------------------------------------


@pytest.mark.skipif(not _DFTD3_OK, reason="dftd3 bindings unavailable")
def test_multi_k_gpw_with_d3bj_pbe_is_nonzero_negative():
    """Same expectation on the multi-k RKS path."""
    from vibeqc.periodic_gapw_j import run_periodic_rks_gpw_multi_k

    L = 12.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)
    kmesh = core.monkhorst_pack(system, [1, 1, 1])

    r0 = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh,
        functional="pbe", grid=grid, quiet=True,
    )
    r1 = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh,
        functional="pbe", grid=grid,
        dispersion="d3-bj", dispersion_functional="pbe",
        quiet=True,
    )
    assert r0.converged and r1.converged
    assert r1.e_dispersion < 0.0
    assert abs(r1.e_dispersion) > 1e-7
    assert r1.energy == pytest.approx(
        r0.energy + r1.e_dispersion, abs=1e-10
    )
    assert r1.breakdown.e_total == pytest.approx(
        r0.breakdown.e_total + r1.e_dispersion, abs=1e-10
    )


def test_multi_k_gpw_no_dispersion_is_backwards_compatible():
    """``dispersion=None`` (default) on the multi-k path leaves the
    total + breakdown untouched."""
    from vibeqc.periodic_gapw_j import run_periodic_rks_gpw_multi_k

    L = 12.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48, cutoff_ha=300.0)
    kmesh = core.monkhorst_pack(system, [1, 1, 1])

    r = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh,
        functional="lda", grid=grid, quiet=True,
    )
    assert r.converged
    assert r.e_dispersion == 0.0
    assert r.breakdown.e_dispersion == 0.0


# ---------- dependency-missing path (informational only) -------------


@pytest.mark.skipif(_DFTD3_OK, reason="dftd3 bindings ARE installed")
def test_rhf_gpw_dispersion_raises_when_dftd3_missing():  # pragma: no cover
    """If dftd3 isn't installed, requesting dispersion must raise
    NotImplementedError with a remediation hint."""
    from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

    L = 12.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 32, 32, 32, cutoff_ha=200.0)
    with pytest.raises(NotImplementedError, match="dftd3"):
        run_periodic_rhf_gpw(
            system, basis, grid=grid, max_iter=10,
            functional="pbe",
            dispersion="d3-bj", dispersion_functional="pbe",
            quiet=True,
        )
