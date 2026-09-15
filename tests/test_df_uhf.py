"""DF-UHF: density-fitted unrestricted Hartree-Fock energies.

Mirrors tests/test_df_rhf.py for the open-shell UHF path. Pins:

  1. Internal — DF-UHF agrees with direct UHF up to JKfit fit error.
  2. PySCF parity — DF-UHF matches ``mf.density_fit(auxbasis=...)`` to
     numerical precision when both libraries use the same aux.
  3. Spin contamination — <S^2> matches the direct path; pure-spin
     systems (single-electron H atom doublet) give exactly 0.75.
  4. Closed-shell consistency — DF-UHF on a closed-shell singlet
     (n_α = n_β) recovers DF-RHF energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    UHFOptions,
    run_rhf,
    run_uhf,
)

from .conftest import GEOMETRIES


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _vibeqc_uhf(atoms_bohr, basis_name, charge, mult, *,
                density_fit, aux_basis_name=""):
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
        charge=charge, multiplicity=mult,
    )
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    # OH / 6-31G* and similar plateau at orbital-grad ~1e-8 with energy at
    # machine precision; loose grad tol matches tests/test_uhf.py.
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 500
    opts.density_fit = density_fit
    opts.aux_basis = aux_basis_name
    return run_uhf(mol, basis, opts)


def _pyscf_uhf_df(atoms_bohr, basis_name, aux_basis_name, *, charge, spin):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol).density_fit(auxbasis=aux_basis_name)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    assert mf.converged, f"PySCF DF-UHF did not converge {basis_name}/{aux_basis_name}"
    s2, _ = mf.spin_square()
    return mf.e_tot, s2, mf.mo_energy


# Open-shell test geometries: (label, atoms, basis, charge, multiplicity, aux).
# JK auxiliary basis is the universal Weigend (BSE-fetched) so PySCF can
# load the same name.
OPEN_SHELL_DF_CASES = [
    ("H-doublet",
     [(1, [0.0, 0.0, 0.0])],
     "def2-svp", 0, 2, "def2-universal-jkfit"),
    ("OH-doublet",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
     "def2-svp", 0, 2, "def2-universal-jkfit"),
    ("OH-doublet-cc",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
     "cc-pvdz", 0, 2, "cc-pvdz-jkfit"),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult,aux_basis",
    OPEN_SHELL_DF_CASES,
    ids=[c[0] for c in OPEN_SHELL_DF_CASES],
)
def test_df_uhf_close_to_direct_uhf(
    label, atoms, basis_name, charge, mult, aux_basis,
):
    """DF-UHF total energy lies within Eichkorn / Weigend fit error of
    direct UHF on the same orbital basis."""
    direct = _vibeqc_uhf(atoms, basis_name, charge, mult, density_fit=False)
    df = _vibeqc_uhf(atoms, basis_name, charge, mult,
                     density_fit=True, aux_basis_name=aux_basis)

    assert direct.converged, f"{label}: direct UHF did not converge"
    assert df.converged, f"{label}: DF-UHF did not converge"

    delta = df.energy - direct.energy
    assert abs(delta) < 5e-4, (
        f"{label}/{basis_name}/{aux_basis}: DF-direct gap = {delta:+.3e} Ha "
        f"(direct = {direct.energy:.10f}, DF = {df.energy:.10f})"
    )


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult,aux_basis",
    OPEN_SHELL_DF_CASES,
    ids=[c[0] for c in OPEN_SHELL_DF_CASES],
)
def test_df_uhf_matches_pyscf_df(
    label, atoms, basis_name, charge, mult, aux_basis,
):
    """vibe-qc DF-UHF matches PySCF DF-UHF on the same aux basis."""
    df = _vibeqc_uhf(atoms, basis_name, charge, mult,
                     density_fit=True, aux_basis_name=aux_basis)
    spin = mult - 1
    e_pyscf, s2_pyscf, _ = _pyscf_uhf_df(
        atoms, basis_name, aux_basis, charge=charge, spin=spin,
    )
    delta = df.energy - e_pyscf
    assert abs(delta) < 1e-9, (
        f"{label}/{basis_name}/{aux_basis}: vibeqc-PySCF DF-UHF gap = "
        f"{delta:+.3e} Ha (vibeqc = {df.energy:.12f}, "
        f"PySCF = {e_pyscf:.12f})"
    )
    # Spin contamination should match too — it's derivable from the
    # converged orbitals and S, so DF and direct agree on it within
    # whatever fit-error perturbation moves the orbitals.
    assert abs(df.s_squared - s2_pyscf) < 1e-7, (
        f"{label}: <S^2> disagrees with PySCF DF-UHF "
        f"(vibeqc = {df.s_squared}, pyscf = {s2_pyscf})"
    )


def test_df_uhf_h_doublet_pure_spin():
    """Single H atom: the only one-electron spin state. UHF must give
    <S^2> = 0.75 exactly regardless of DF or direct path — there are no
    occupied beta orbitals to spin-contaminate against."""
    df = _vibeqc_uhf(
        [(1, [0.0, 0.0, 0.0])], "def2-svp", 0, 2,
        density_fit=True, aux_basis_name="def2-universal-jkfit",
    )
    assert df.s_squared == pytest.approx(0.75, abs=1e-10)
    assert df.s_squared_ideal == pytest.approx(0.75, abs=1e-12)


def test_df_uhf_closed_shell_recovers_df_rhf():
    """For a closed-shell singlet (H2O), DF-UHF must reproduce DF-RHF
    to numerical precision: alpha and beta densities match, exchange
    contributions are identical."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")

    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf_opts.density_fit = True
    rhf_opts.aux_basis = "def2-universal-jkfit"
    r_rhf = run_rhf(mol, basis, rhf_opts)

    uhf_opts = UHFOptions()
    uhf_opts.conv_tol_energy = 1e-12
    uhf_opts.conv_tol_grad = 1e-8
    uhf_opts.density_fit = True
    uhf_opts.aux_basis = "def2-universal-jkfit"
    r_uhf = run_uhf(mol, basis, uhf_opts)

    assert r_uhf.converged and r_rhf.converged
    assert abs(r_uhf.energy - r_rhf.energy) < 1e-9, (
        f"DF-UHF on H2O singlet: E_uhf = {r_uhf.energy:.12f}, "
        f"E_rhf = {r_rhf.energy:.12f}, "
        f"gap = {r_uhf.energy - r_rhf.energy:+.3e}"
    )
    assert abs(r_uhf.s_squared) < 1e-8


def test_df_uhf_requires_aux_basis():
    """density_fit=True with empty aux_basis raises — same contract
    as run_rhf."""
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
                   multiplicity=2)
    basis = BasisSet(mol, "def2-svp")
    opts = UHFOptions()
    opts.density_fit = True
    opts.aux_basis = ""
    with pytest.raises(ValueError, match="aux_basis"):
        run_uhf(mol, basis, opts)
