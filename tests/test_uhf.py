"""UHF end-to-end cross-checks against PySCF."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    UHFOptions,
    run_rhf,
    run_uhf,
)

from .conftest import GEOMETRIES


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _pyscf_uhf(atoms_bohr, basis_name, *, charge: int, spin: int):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.charge = charge
    mol.spin = spin  # PySCF: spin = 2S = n_alpha - n_beta
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    s2, _ = mf.spin_square()
    return mf.e_tot, s2, mf.mo_energy  # mo_energy is (2, n_bf) for UHF


def _tight_uhf_opts() -> UHFOptions:
    # UHF on some open-shell doublets (e.g. OH / 6-31G*) plateaus with a
    # residual orbital-gradient norm ~1e-8 while the energy is already at
    # machine precision — matches PySCF behavior. Require 1e-8 for grad
    # rather than 1e-10 so tests are satisfiable without level shifting.
    opts = UHFOptions()
    # 500 iters is comfortable; OH / 6-31G* needs ~320 from the symmetry-
    # broken Hcore guess before the orbital gradient dips below 1e-8.
    opts.max_iter = 500
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-8
    opts.damping = 0.0
    return opts


# Open-shell test geometries: (label, atoms_bohr, basis, charge, multiplicity).
# The SAD initial guess (default) avoids the false local minimum that trapped
# pure-Hcore UHF on OH / 6-31G*.
OPEN_SHELL_CASES = [
    ("H-doublet",       [(1, [0.0, 0.0, 0.0])],                                     "sto-3g", 0, 2),
    ("H-doublet-6-31gs",[(1, [0.0, 0.0, 0.0])],                                     "6-31g*", 0, 2),
    ("OH-doublet",      [(8, [0.0, 0.0, 0.0]),
                         (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],                 "sto-3g", 0, 2),
    ("OH-doublet-6-31g*",[(8, [0.0, 0.0, 0.0]),
                          (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],                "6-31g*", 0, 2),
    ("O2-triplet",      [(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])],              "sto-3g", 0, 3),
    ("O2-triplet-6-31g*",[(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])],             "6-31g*", 0, 3),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult",
    OPEN_SHELL_CASES,
    ids=[c[0] for c in OPEN_SHELL_CASES],
)
def test_uhf_energy_matches_pyscf(label, atoms, basis_name, charge, mult):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms],
                   charge=charge, multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    result = run_uhf(mol, basis, _tight_uhf_opts())
    assert result.converged, f"{label}: UHF did not converge in {result.n_iter} iters"

    spin = mult - 1   # PySCF uses 2S
    ref_E, ref_s2, _ = _pyscf_uhf(atoms, basis_name, charge=charge, spin=spin)
    diff = result.energy - ref_E
    assert abs(diff) < 1e-9, (
        f"{label}: E_vibeqc = {result.energy:.12f}, E_pyscf = {ref_E:.12f}, "
        f"diff = {diff:+.2e}"
    )
    # <S^2> should also match (it's derivable from the converged orbitals).
    assert result.s_squared == pytest.approx(ref_s2, abs=1e-8)
    assert result.s_squared_deviation == pytest.approx(
        result.s_squared - result.s_squared_ideal,
        abs=1e-14,
    )
    if label.startswith("OH-doublet"):
        # The deliberately contamination-sensitive radical remains above the
        # ideal doublet value in both compact and split-valence bases.
        assert result.s_squared_deviation > 1.0e-4
    if label.startswith("O2-triplet"):
        assert result.s_squared_ideal == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize(
    "mol_key,basis_name",
    [("H2", "sto-3g"), ("H2", "6-31g*"), ("H2O", "sto-3g"), ("H2O", "6-31g*")],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_uhf_on_closed_shell_matches_rhf(mol_key, basis_name):
    """For closed-shell singlets, UHF must recover RHF energy exactly
    (alpha and beta orbitals are identical; exchange contributions match)."""
    atoms = GEOMETRIES[mol_key]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, basis_name)
    r_uhf = run_uhf(mol, basis, _tight_uhf_opts())
    # RHF with matching tolerance
    from vibeqc import RHFOptions
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf_opts.damping = 0.0
    r_rhf = run_rhf(mol, basis, rhf_opts)
    assert r_uhf.converged and r_rhf.converged
    assert abs(r_uhf.energy - r_rhf.energy) < 1e-10
    # Closed-shell: <S^2> = 0 exactly.
    assert abs(r_uhf.s_squared) < 1e-10


def test_uhf_rejects_mult_electron_inconsistency():
    # Constructor catches inconsistency before UHF even starts.
    with pytest.raises(ValueError, match="inconsistent"):
        Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], multiplicity=2)


def test_uhf_s2_ideal_value_for_doublet():
    """Single H atom: UHF must give <S^2> = 0.75 exactly (one-electron
    system, no possible spin contamination)."""
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    r = run_uhf(mol, basis, _tight_uhf_opts())
    assert r.s_squared == pytest.approx(0.75, abs=1e-10)
    assert r.s_squared_ideal == pytest.approx(0.75, abs=1e-12)


def test_uhf_mo_energies_match_pyscf_doublet_oh():
    atoms = [(8, [0.0, 0.0, 0.0]),
             (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms], multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    r = run_uhf(mol, basis, _tight_uhf_opts())

    _, _, pyscf_mo = _pyscf_uhf(atoms, "sto-3g", charge=0, spin=1)
    # pyscf_mo is shape (2, nbf): row 0 = alpha, row 1 = beta
    np.testing.assert_allclose(
        np.sort(np.array(r.mo_energies_alpha)),
        np.sort(pyscf_mo[0]),
        atol=1e-9,
        err_msg="alpha MO energies disagree with PySCF",
    )
    np.testing.assert_allclose(
        np.sort(np.array(r.mo_energies_beta)),
        np.sort(pyscf_mo[1]),
        atol=1e-9,
        err_msg="beta MO energies disagree with PySCF",
    )
