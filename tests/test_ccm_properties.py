"""Derivable properties on the cyclic cluster (BvK torus) — AICCM Task C.

HOMO–LUMO / band gap, Mulliken / Löwdin populations (with ``S^CCM``), the
finite-cluster dipole (with the Resta caveat), and a finite-difference nuclear
gradient. Each property is grounded against an independent check: the gap vs the
raw spectrum, the charges' sum vs the cluster charge + translational symmetry,
the dipole's vanishing for a centrosymmetric cluster, and the gradient vs
vibe-qc's molecular analytic gradient in the isolated limit.

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014); Resta,
Phys. Rev. Lett. 80, 1800 (1998). Derivation: docs/aiccm2026dev_a_followon.md § C.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    PeriodicSystem,
    RHFOptions,
    compute_gradient,
    run_rhf,
)
from vibeqc.periodic.ccm import (
    CCMSystem,
    ccm_dipole,
    ccm_homo_lumo_gap,
    ccm_lowdin_charges,
    ccm_mulliken_charges,
    ccm_numerical_gradient,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _h2_chain(cell=6.0, vac=15.0, d=1.3):
    return PeriodicSystem(
        3, np.diag([cell, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [d, 0, 0])], charge=0, multiplicity=1,
    )


@pytest.fixture(scope="module")
def h2_311():
    ccm = CCMSystem(_h2_chain(), (3, 1, 1), "sto-3g")
    scf = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    return ccm, scf


# --------------------------------------------------------------------------- #
# HOMO–LUMO / band gap
# --------------------------------------------------------------------------- #
def test_gap_matches_spectrum(h2_311):
    ccm, scf = h2_311
    g = ccm_homo_lumo_gap(scf, ccm)
    eps = np.sort(np.asarray(scf.mo_energies, dtype=float))
    n_occ = ccm.supercell.n_electrons() // 2
    assert g.gap == pytest.approx(eps[n_occ] - eps[n_occ - 1], abs=1e-12)
    assert g.homo == pytest.approx(eps[n_occ - 1])
    assert g.lumo == pytest.approx(eps[n_occ])
    assert g.gap > 0.0  # insulating H₂ chain
    assert g.spin == "restricted"


def test_spin_resolved_gap_open_shell():
    """UHF result → spin-resolved gap (HOMO/LUMO from the α/β spectra)."""
    from vibeqc.periodic.ccm.uhf import run_ccm_uhf

    unit = PeriodicSystem(
        1, np.array([[1.8, 0, 0], [0, 15.0, 0], [0, 0, 15.0]]),
        [Atom(1, [0, 0, 0])], charge=0, multiplicity=2)
    ccm = CCMSystem(unit, (3, 1, 1), "sto-3g")
    uhf = run_ccm_uhf(ccm, method="aiccm2026dev-a")
    g = ccm_homo_lumo_gap(uhf, ccm)
    assert g.spin == "unrestricted"
    ea = np.sort(np.asarray(uhf.mo_energies_alpha))
    eb = np.sort(np.asarray(uhf.mo_energies_beta))
    homo = max(ea[uhf.n_alpha - 1], eb[uhf.n_beta - 1])
    lumo = min(ea[uhf.n_alpha], eb[uhf.n_beta])
    assert g.homo == pytest.approx(homo)
    assert g.lumo == pytest.approx(lumo)
    assert g.gap == pytest.approx(lumo - homo)


# --------------------------------------------------------------------------- #
# Mulliken / Löwdin populations
# --------------------------------------------------------------------------- #
def test_charges_sum_to_cluster_charge(h2_311):
    ccm, scf = h2_311
    for fn in (ccm_mulliken_charges, ccm_lowdin_charges):
        pop = fn(scf, ccm)
        assert pop.charges.sum() == pytest.approx(float(ccm.supercell.charge), abs=1e-9)
        assert pop.charges.shape == (ccm.n_atoms,)
        assert pop.charges_per_cell.shape == (ccm.n_basis_atoms,)


def test_homonuclear_charges_vanish(h2_311):
    ccm, scf = h2_311
    pop = ccm_mulliken_charges(scf, ccm)
    # homonuclear chain → ~zero per-atom charge, small translational spread
    assert np.allclose(pop.charges_per_cell, 0.0, atol=1e-3)
    assert pop.translational_spread < 1e-2


def test_polar_chain_charge_transfer():
    """LiH chain: Li carries positive, H negative Mulliken charge."""
    unit = PeriodicSystem(
        3, np.diag([3.2 * BOHR, 20.0, 20.0]),
        [Atom(3, [0, 0, 0]), Atom(1, [1.6 * BOHR, 0, 0])], charge=0, multiplicity=1)
    ccm = CCMSystem(unit, (3, 1, 1), "sto-3g")
    scf = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    pop = ccm_mulliken_charges(scf, ccm)
    q_li, q_h = pop.charges_per_cell
    assert q_li > 0.0 > q_h


# --------------------------------------------------------------------------- #
# Dipole (finite-cluster; Resta caveat)
# --------------------------------------------------------------------------- #
def test_centrosymmetric_dipole_vanishes(h2_311):
    ccm, scf = h2_311
    mu = ccm_dipole(scf, ccm)
    assert np.linalg.norm(mu) < 1e-8


# --------------------------------------------------------------------------- #
# Numerical nuclear gradient
# --------------------------------------------------------------------------- #
def test_gradient_translational_invariance(h2_311):
    ccm, scf = h2_311
    g = ccm_numerical_gradient(ccm, method="aiccm2026dev-a", h=1e-3)
    assert g.shape == (ccm.n_basis_atoms, 3)
    # Σ forces over the cell vanishes (translational invariance)
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-6)


def test_gradient_isolated_limit_matches_molecular():
    """CCM (1,1,1) in a huge box reproduces the molecular analytic RHF gradient."""
    geom = [(1, [0, 0, 0]), (1, [1.3, 0, 0])]
    iso = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]), [Atom(z, p) for z, p in geom])
    ccm = CCMSystem(iso, (1, 1, 1), "sto-3g")
    g_ccm = ccm_numerical_gradient(ccm, method="aiccm2026dev-a", h=5e-4)

    mol = Molecule([Atom(z, p) for z, p in geom], 0, 1)
    b = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    g_mol = np.asarray(compute_gradient(mol, b, run_rhf(mol, b, opts)))
    assert np.max(np.abs(g_ccm - g_mol)) < 1e-6
