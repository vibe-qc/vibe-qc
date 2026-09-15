"""KDIIS — Kollmar 1997 SCF accelerator tests.

KDIIS uses the orbital-rotation gradient g_ai = F^MO_{ai} (occ-vir
block of F in the canonical MO basis) as the DIIS error vector,
instead of Pulay's AO-basis commutator e = F D S − S D F. The
extrapolation machinery is identical (Pulay least-squares with
constraint Σ c_i = 1); only the error metric differs.

Coverage:

  1. Default-off back-compat: ``scf_accelerator = DIIS`` (the default)
     is unchanged — KDIIS is opt-in via ``SCFAccelerator.KDIIS``.
  2. KDIIS converges to the same SCF fixed point as DIIS on standard
     closed-shell + open-shell test cases (energy parity to ~1e-9 Ha).
  3. KDIIS works for all four flavours (RHF / UHF / RKS / UKS) — the
     accelerator surface is uniform.
  4. Bounded iteration count on easy cases (no regression).

Reference:
  C. Kollmar, "Convergence optimization of restricted open-shell
  self-consistent field calculations", Int. J. Quantum Chem. 62,
  617-637 (1997),
  doi:10.1002/(SICI)1097-461X(1997)62:6<617::AID-QUA5>3.0.CO;2-Z
  Exposed by ORCA as the opt-in ``!KDIIS`` keyword. vibe-qc reuses
  Kollmar's error vector but still diagonalizes the extrapolated Fock;
  Kollmar's own scheme is diagonalization-free.
"""

from __future__ import annotations

import pytest

from vibeqc import (
    BasisSet,
    RHFOptions,
    UHFOptions,
    RKSOptions,
    UKSOptions,
    SCFAccelerator,
    run_rhf,
    run_uhf,
    run_rks,
    run_uks,
)

from .conftest import ANGSTROM_TO_BOHR, make_molecule


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def _h2o_atoms():
    return [
        (8, [0.0, 0.0,  0.117 * ANGSTROM_TO_BOHR]),
        (1, [0.0,  0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
        (1, [0.0, -0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
    ]


def _oh_radical_atoms():
    """Hydroxyl radical (doublet, 9 electrons)."""
    return [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
    ]


@pytest.fixture
def h2o_basis():
    mol = make_molecule(_h2o_atoms())
    return mol, BasisSet(mol, "sto-3g")


@pytest.fixture
def oh_radical():
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    return mol, BasisSet(mol, "sto-3g")


# ---------------------------------------------------------------------------
# Enum surface — KDIIS is registered alongside DIIS / EDIIS / EDIIS_DIIS.
# ---------------------------------------------------------------------------

def test_scf_accelerator_enum_includes_kdiis():
    """The SCFAccelerator enum exposes KDIIS as a selectable option."""
    assert hasattr(SCFAccelerator, "KDIIS")
    # Sanity: distinct from the other family members.
    assert SCFAccelerator.KDIIS != SCFAccelerator.DIIS
    assert SCFAccelerator.KDIIS != SCFAccelerator.EDIIS
    assert SCFAccelerator.KDIIS != SCFAccelerator.EDIIS_DIIS


def test_default_scf_accelerator_is_not_kdiis():
    """Default for all four flavours is EDIIS_DIIS (Track F, v0.8.0) —
    KDIIS is opt-in, not the new default."""
    assert RHFOptions().scf_accelerator != SCFAccelerator.KDIIS
    assert UHFOptions().scf_accelerator != SCFAccelerator.KDIIS
    assert RKSOptions().scf_accelerator != SCFAccelerator.KDIIS
    assert UKSOptions().scf_accelerator != SCFAccelerator.KDIIS
    assert RHFOptions().scf_accelerator == SCFAccelerator.EDIIS_DIIS


# ---------------------------------------------------------------------------
# RHF / RKS — closed-shell parity vs DIIS.
# ---------------------------------------------------------------------------

def test_rhf_kdiis_matches_diis_h2o(h2o_basis):
    """RHF / sto-3g: KDIIS reaches the same energy as DIIS to 1e-9 Ha."""
    mol, basis = h2o_basis

    o_diis = RHFOptions()
    o_diis.scf_accelerator = SCFAccelerator.DIIS
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-7
    r_diis = run_rhf(mol, basis, o_diis)
    assert r_diis.converged

    o_kdiis = RHFOptions()
    o_kdiis.scf_accelerator = SCFAccelerator.KDIIS
    o_kdiis.conv_tol_energy = 1e-10
    o_kdiis.conv_tol_grad = 1e-7
    r_kdiis = run_rhf(mol, basis, o_kdiis)
    assert r_kdiis.converged
    assert r_kdiis.energy == pytest.approx(r_diis.energy, abs=1e-9)


def test_rks_kdiis_matches_diis_h2o(h2o_basis):
    """RKS / LDA / sto-3g: KDIIS reaches the same energy as DIIS."""
    mol, basis = h2o_basis

    o_diis = RKSOptions()
    o_diis.functional = "LDA"
    o_diis.scf_accelerator = SCFAccelerator.DIIS
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-7
    r_diis = run_rks(mol, basis, o_diis)
    assert r_diis.converged

    o_kdiis = RKSOptions()
    o_kdiis.functional = "LDA"
    o_kdiis.scf_accelerator = SCFAccelerator.KDIIS
    o_kdiis.conv_tol_energy = 1e-10
    o_kdiis.conv_tol_grad = 1e-7
    r_kdiis = run_rks(mol, basis, o_kdiis)
    assert r_kdiis.converged
    assert r_kdiis.energy == pytest.approx(r_diis.energy, abs=1e-8)


# ---------------------------------------------------------------------------
# UHF / UKS — open-shell parity vs DIIS (per-spin κ stacked into a single
# error vector with shared coefficient set, matching the EDIIS open-shell
# convention).
# ---------------------------------------------------------------------------

def test_uhf_kdiis_matches_diis_oh_radical(oh_radical):
    """UHF / sto-3g on OH·: KDIIS reaches the same energy as DIIS, and
    `<S^2>` is consistent (parity of converged density)."""
    mol, basis = oh_radical

    o_diis = UHFOptions()
    o_diis.scf_accelerator = SCFAccelerator.DIIS
    o_diis.max_iter = 250
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-8
    r_diis = run_uhf(mol, basis, o_diis)
    assert r_diis.converged

    o_kdiis = UHFOptions()
    o_kdiis.scf_accelerator = SCFAccelerator.KDIIS
    o_kdiis.max_iter = 250
    o_kdiis.conv_tol_energy = 1e-10
    o_kdiis.conv_tol_grad = 1e-8
    r_kdiis = run_uhf(mol, basis, o_kdiis)
    assert r_kdiis.converged
    assert r_kdiis.energy == pytest.approx(r_diis.energy, abs=1e-9)
    assert r_kdiis.s_squared == pytest.approx(r_diis.s_squared, abs=1e-7)


def test_uks_kdiis_options_field_present():
    """UKSOptions exposes ``scf_accelerator``, including the new
    KDIIS choice."""
    o = UKSOptions()
    o.scf_accelerator = SCFAccelerator.KDIIS
    assert o.scf_accelerator == SCFAccelerator.KDIIS


@pytest.fixture
def o_triplet():
    """Triplet O atom / sto-3g — canonical simplest open-shell UKS case,
    shared with test_molecular_level_shift.py. OH·/LDA/sto-3g is too
    flat in spin space and the alpha/beta densities don't break symmetry
    reliably from a SAD guess."""
    from vibeqc import Atom, Molecule
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0])], charge=0, multiplicity=3)
    return mol, BasisSet(mol, "sto-3g")


def test_uks_kdiis_matches_diis_o_triplet(o_triplet):
    """UKS / LDA / sto-3g on triplet O: KDIIS reaches the same energy as
    DIIS, and ``<S^2>`` is consistent. Exercises the per-spin κ-stacked
    error vector on the DFT path (RKS covers the closed-shell DFT case;
    UHF covers the open-shell HF case; this closes the matrix corner)."""
    mol, basis = o_triplet

    o_diis = UKSOptions()
    o_diis.functional = "LDA"
    o_diis.scf_accelerator = SCFAccelerator.DIIS
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-8
    r_diis = run_uks(mol, basis, o_diis)
    assert r_diis.converged

    o_kdiis = UKSOptions()
    o_kdiis.functional = "LDA"
    o_kdiis.scf_accelerator = SCFAccelerator.KDIIS
    o_kdiis.conv_tol_energy = 1e-10
    o_kdiis.conv_tol_grad = 1e-8
    r_kdiis = run_uks(mol, basis, o_kdiis)
    assert r_kdiis.converged
    assert r_kdiis.energy == pytest.approx(r_diis.energy, abs=1e-8)
    assert r_kdiis.s_squared == pytest.approx(r_diis.s_squared, abs=1e-7)


# ---------------------------------------------------------------------------
# Iteration-count sanity — KDIIS shouldn't blow up easy cases.
# ---------------------------------------------------------------------------

def test_rhf_kdiis_runs_to_convergence_within_iter_budget(h2o_basis):
    """KDIIS on H2O / sto-3g converges in a comparable number of iters
    to plain DIIS — KDIIS's selling point on hard cases is robustness,
    not iteration count on easy ones; here we just verify the iter
    count stays bounded."""
    mol, basis = h2o_basis
    opts = RHFOptions()
    opts.scf_accelerator = SCFAccelerator.KDIIS
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-6
    r = run_rhf(mol, basis, opts)
    assert r.converged
    # H2O / sto-3g with SAD guess — DIIS converges in ~10 iters.
    # KDIIS should be in the same ballpark; allow 2x headroom.
    assert r.n_iter <= 20
