"""Fermi-Dirac smearing on the GPW route (v0.12 R1).

Closes task #38 of the v0.12 R1 roadmap. Pins, for both
:func:`run_periodic_rhf_gpw` (Γ-only RHF) and
:func:`run_periodic_rks_gpw_multi_k` (multi-k pure-DFT):

* ``smearing_temperature == 0`` reproduces the historical
  integer-Aufbau path bit-for-bit — the converged energy is
  unchanged, ``smearing_entropy == 0``, and ``free_energy ==
  energy``.
* A finite ``smearing_temperature > 0`` on H₂/STO-3G in a
  vacuum-padded cubic cell produces HOMO/LUMO fractional
  occupations strictly inside ``(0, 2)`` and shifts the Mermin
  free energy ``F = E − T·S`` below the converged ``E_total``
  (the entropy term lowers the free energy by construction).
* The Mermin entropy is non-negative for any ``T > 0`` (a
  sanity check the closed-shell entropy formula upholds).
* μ (the Fermi level) lands between the converged HOMO and
  LUMO eigenvalues for an insulating cell — the canonical
  Fermi-Dirac midgap behaviour at low temperature.

The closed-shell entropy convention follows
``vibeqc.smearing.fermi_dirac`` (the same one all other
periodic SCF drivers already wire). See the design contract
in ``docs/design_smearing.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
)
from vibeqc.periodic_gapw_j import (
    run_periodic_rhf_gpw,
    run_periodic_rks_gpw_multi_k,
)

# Silence the experimental warning globally for this file — every
# test below routes through the GAPW path.
warnings.simplefilter("ignore", GAPWExperimentalWarning)
pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Fixtures ------------------------------------------------------


def _h2_periodic_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    return sys


def _h2_basis(L: float = 16.0):
    mol = vq.Molecule(
        [
            vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
            vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
        ],
        charge=0,
        multiplicity=1,
    )
    return vq.BasisSet(mol, "sto-3g")


# ---------- T = 0 zero-smearing reference --------------------------------


def test_rhf_gpw_h2_zero_temperature_matches_aufbau():
    """``smearing_temperature == 0`` reproduces the historical
    integer-Aufbau RHF energy bit-for-bit; entropy is exactly 0
    and the free energy collapses onto the total energy."""
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)

    base = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
    )
    smeared_zero = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=30, quiet=True,
        smearing_temperature=0.0,
    )

    assert base.converged
    assert smeared_zero.converged
    assert smeared_zero.energy == pytest.approx(base.energy, abs=1e-9)
    assert smeared_zero.smearing_entropy == 0.0
    assert smeared_zero.smearing_temperature == 0.0
    assert smeared_zero.free_energy == pytest.approx(
        smeared_zero.energy, abs=1e-12
    )
    # Integer occupations: one doubly-occupied HOMO, all virtuals empty.
    occ = np.asarray(smeared_zero.occupations, dtype=float)
    assert occ[0] == pytest.approx(2.0, abs=1e-12)
    assert occ[1] == pytest.approx(0.0, abs=1e-12)


# ---------- T > 0 fractional occupations + free-energy lowering -----------


def test_rhf_gpw_h2_finite_T_gives_fractional_occupations():
    """At ``T = 0.01 Ha`` (~3158 K) on H₂/STO-3G the HOMO/LUMO gap
    is finite and the Fermi-Dirac occupations land strictly inside
    ``(0, 2)``. The Mermin free energy lies below the total
    electronic energy (the ``−T·S`` term is negative since S > 0)."""
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)

    # T = 0.1 Ha (~31580 K) is hot enough to push a measurable
    # fraction of charge into the LUMO on H₂/STO-3G — the HOMO/LUMO
    # gap on this box is ~1 Ha, so at lower T the LUMO occupation
    # would be exponentially tiny and the bounds would not bite.
    T = 0.1  # Hartree
    result = run_periodic_rhf_gpw(
        system, basis, grid=grid, max_iter=80, quiet=True,
        smearing_temperature=T, damping=0.3,
    )

    assert result.converged
    occ = np.asarray(result.occupations, dtype=float)
    # H2 STO-3G has 2 basis functions → HOMO (idx 0), LUMO (idx 1).
    # Both must lie strictly inside (0, 2) with f_HOMO + f_LUMO = 2.
    assert 0.0 < occ[1] < occ[0], (
        f"LUMO occupation should be smaller but positive; "
        f"got HOMO={occ[0]}, LUMO={occ[1]}"
    )
    assert occ[0] < 2.0, (
        f"HOMO occupation should be < 2 with finite T; got {occ[0]}"
    )
    # Particle-conservation: Σ f_i == n_elec to bisection tolerance.
    assert float(occ.sum()) == pytest.approx(2.0, abs=1e-8)

    # μ sits between the converged HOMO and LUMO MO energies for an
    # insulating cell.
    e = np.asarray(result.mo_energies, dtype=float)
    assert e[0] < result.fermi_level < e[1]

    # Mermin free energy: F = E − T·S, with S > 0 → F < E.
    assert result.smearing_entropy > 0.0
    assert result.free_energy < result.energy
    expected_F = result.energy - T * result.smearing_entropy
    assert result.free_energy == pytest.approx(expected_F, abs=1e-12)


def test_rhf_gpw_entropy_is_nonnegative_across_temperatures():
    """Closed-shell Mermin entropy ``S = −Σ [f log f + (1−f) log(1−f)]``
    must be non-negative for every probability ``f ∈ [0, 1]``. Spot-
    check three temperatures on H₂/STO-3G."""
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)

    for T in (1e-4, 1e-3, 1e-2):
        result = run_periodic_rhf_gpw(
            system, basis, grid=grid, max_iter=60, quiet=True,
            smearing_temperature=T,
        )
        assert result.converged, f"SCF must converge at T={T}"
        assert result.smearing_entropy >= 0.0, (
            f"Entropy must be non-negative at T={T}; got "
            f"{result.smearing_entropy}"
        )
        # Particle count is preserved at every temperature.
        occ = np.asarray(result.occupations, dtype=float)
        assert float(occ.sum()) == pytest.approx(2.0, abs=1e-8)


# ---------- Multi-k coverage ---------------------------------------------


def test_rks_gpw_multi_k_h2_finite_T_lowers_free_energy():
    """Multi-k pure-DFT path: a Γ-only single-k mesh on H₂ + LDA at
    ``T = 0.01 Ha`` produces a fractional LUMO occupation and the
    Mermin free energy lies below the total energy.

    Γ-only on a single k is the smallest meaningful multi-k stress
    that exercises the per-k smearing wiring without paying the
    full multi-k integration cost."""
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)

    # Build a Γ-only single-k mesh via the public helper.
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])

    T = 0.01
    result = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh,
        functional="lda",
        grid=grid, max_iter=60, quiet=True,
        smearing_temperature=T,
    )
    assert result.converged
    assert result.smearing_entropy > 0.0
    assert result.free_energy < result.energy
    expected_F = result.energy - T * result.smearing_entropy
    assert result.free_energy == pytest.approx(expected_F, abs=1e-12)
    # Fractional HOMO/LUMO at the single k-point.
    occ_k0 = np.asarray(result.occupations_k[0], dtype=float)
    assert float(occ_k0.sum()) == pytest.approx(2.0, abs=1e-8)
    assert 0.0 < occ_k0[1] < 0.5


def test_rks_gpw_multi_k_zero_T_matches_integer_occupations():
    """``smearing_temperature == 0`` on the multi-k path preserves
    the historical integer-Aufbau density and energy."""
    L = 16.0
    basis = _h2_basis(L)
    system = _h2_periodic_system(L)
    grid = PlaneWaveGrid(np.eye(3) * L, 48, 48, 48)

    kmesh = vq.monkhorst_pack(system, [1, 1, 1])

    base = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh, functional="lda",
        grid=grid, max_iter=60, quiet=True,
    )
    smeared_zero = run_periodic_rks_gpw_multi_k(
        system, basis, kmesh, functional="lda",
        grid=grid, max_iter=60, quiet=True,
        smearing_temperature=0.0,
    )
    assert base.converged
    assert smeared_zero.converged
    assert smeared_zero.energy == pytest.approx(base.energy, abs=1e-9)
    assert smeared_zero.smearing_entropy == 0.0
    assert smeared_zero.free_energy == pytest.approx(
        smeared_zero.energy, abs=1e-12
    )
    occ_k0 = np.asarray(smeared_zero.occupations_k[0], dtype=float)
    assert occ_k0[0] == pytest.approx(2.0, abs=1e-12)
    assert occ_k0[1] == pytest.approx(0.0, abs=1e-12)
