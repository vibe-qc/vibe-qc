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


# ---------------------------------------------------------------------------
# M5: the population SIDECAR, not just the library functions above.
#
# Before this wiring a real-Gamma job wrote a population file whose every
# section was a TypeError row. Two distinct defects produced that, and both
# are worth a named guard: the supercell-Gamma results' orbitals span the
# SUPERCELL while the sidecar's molecule is the unit cell, and their
# density_alpha / density_beta are declared FIELDS whose value is None on a
# closed-shell run -- so every hasattr-based open-shell test downstream took
# the wrong branch and evaluated None + None. That trap already cost #679.
# ---------------------------------------------------------------------------


def _lih_cell(box: float = 8.0, d: float = 3.0):
    return PeriodicSystem(
        3, np.diag([box, box, box]),
        [Atom(3, [box / 2, box / 2, box / 2 - d / 2]),
         Atom(1, [box / 2, box / 2, box / 2 + d / 2])], 0, 1)


def test_total_density_treats_a_none_valued_spin_field_as_closed_shell():
    """The #679 trap, at its root rather than at one of its symptoms.

    ``hasattr`` is True on the runner adapter results because the field
    EXISTS; it is None-valued on a closed-shell run. Anything testing
    presence rather than value adds None to None.
    """

    from vibeqc.periodic.ccm.properties import _total_density

    class _ClosedShellAdapterShaped:
        density = np.eye(3)
        density_alpha = None      # declared, unset -- the trap
        density_beta = None

    got = _total_density(_ClosedShellAdapterShaped())
    assert np.allclose(got, np.eye(3))


@pytest.mark.parametrize("variant", ["real-gamma", "four-center"])
@pytest.mark.parametrize("nrep", [(1, 1, 1), (2, 1, 1)])
def test_population_sidecar_has_no_error_rows(variant, nrep, tmp_path):
    """Every section either carries numbers or a deliberate ``unsupported:``.

    A section reading ``N/A -- TypeError`` is the regression this guards:
    it looks like a considered gate but is a crash.
    """

    from vibeqc import run_periodic_job

    cell = _lih_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    stem = str(tmp_path / "pop")
    run_periodic_job(cell, basis, method="aiccm", variant=variant,
                     aiccm_lattice_extension=nrep, write_population_file=True,
                     initial_guess="HCORE", output=stem)
    text = (tmp_path / "pop.population.txt").read_text(encoding="utf-8")
    for bad in ("TypeError", "ValueError", "AttributeError", "Traceback"):
        assert bad not in text, f"{bad} in the population sidecar:\n{text}"
    # And the charges really are there, not merely un-crashed.
    rows = [ln for ln in text.splitlines() if ln.strip()
            and not ln.startswith("#")]
    assert rows, f"no data rows at all:\n{text}"


@pytest.mark.parametrize("variant", ["real-gamma", "four-center"])
def test_population_sidecar_charges_are_per_cell_and_neutral(variant, tmp_path):
    """Per-cell charges, one row per unit-cell atom, summing to the cell charge.

    ``charges`` runs over SUPERCELL atoms and would give N_c times too many
    rows at nrep>1; ``charges_per_cell`` is the image average the sidecar
    owes. At nrep=(2,1,1) the two differ in length, so this pins the choice.
    """

    from vibeqc import run_periodic_job

    cell = _lih_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    stem = str(tmp_path / "pop")
    run_periodic_job(cell, basis, method="aiccm", variant=variant,
                     aiccm_lattice_extension=(2, 1, 1),
                     write_population_file=True, initial_guess="HCORE",
                     output=stem)
    text = (tmp_path / "pop.population.txt").read_text(encoding="utf-8")
    # The section header line itself ends in "===", so split on the whole
    # marker and stop at the NEXT section marker rather than the first "===".
    after = text.split("# === Mulliken atomic charges ===", 1)[1]
    section = after.split("# === ", 1)[0]
    charges = [float(ln.split("\t")[3]) for ln in section.splitlines()
               if ln.strip() and not ln.startswith("#")]
    assert len(charges) == 2, f"expected one row per unit-cell atom: {charges}"
    assert sum(charges) == pytest.approx(0.0, abs=1e-8)


def test_ccm_per_cell_charges_reduce_to_the_molecular_ones_in_vacuum(tmp_path):
    """The physics check the sidecar rests on.

    At nrep=(1,1,1) in a large box the cyclic cluster IS the isolated
    molecule, so its per-cell Mulliken charges must reproduce vibe-qc's
    molecular ones. This also fixes the SIGN convention against an
    independent implementation: on LiH/STO-3G both put a small NEGATIVE
    charge on Li, a known minimal-basis artefact rather than a bug, and a
    sign flip anywhere in the folding would show up here immediately.
    """

    from vibeqc import run_job
    from vibeqc.properties import mulliken_charges
    from vibeqc.periodic.ccm.real_gamma_runner import run_real_gamma_scf
    from vibeqc.periodic.ccm.properties import ccm_mulliken_charges

    d = 3.0
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, d])], 0, 1)
    mbasis = BasisSet(mol, "sto-3g")
    molecular = np.asarray(mulliken_charges(
        run_job(mol, basis="sto-3g", method="RHF",
                output=str(tmp_path / "iso")),
        mbasis, mol))

    rg = run_real_gamma_scf(_lih_cell(box=24.0, d=d), "sto-3g", "RHF",
                            (1, 1, 1))
    per_cell = np.asarray(
        ccm_mulliken_charges(rg.ccm_result, rg.ccm_system).charges_per_cell)

    assert np.max(np.abs(per_cell - molecular)) < 5.0e-3
    # Same sign on both atoms, checked explicitly: agreeing in magnitude
    # while disagreeing in sign would pass a loose tolerance on a near-zero
    # charge but mean the folding inverted the convention.
    assert np.all(np.sign(per_cell) == np.sign(molecular))
