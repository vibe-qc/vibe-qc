"""SOS-MP2 (Jung-Lochan-Dutoi-Head-Gordon, J. Chem. Phys. 121, 9793
(2004)) — opposite-spin-only second-order Møller-Plesset.

Pins:

  1. SOS-MP2 = c_os · e_os with c_ss = 0 — the same-spin pair
     contribution is dropped, only the opposite-spin (αβ singlet pair)
     remains and is scaled by 1.3.
  2. ``run_sos_mp2`` matches PySCF DFMP2 with the same coefficients
     applied to PySCF's ``e_corr_os`` / ``e_corr_ss`` decomposition.
  3. Internal: e_correlation == 1.3 * e_os exactly (c_ss = 0 drops
     the same-spin term).
  4. UMP2 analogue ``run_sos_ump2`` does the same with the αβ channel
     only.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    MP2Options,
    Molecule,
    RHFOptions,
    UHFOptions,
    UMP2Options,
    run_mp2,
    run_rhf,
    run_sos_mp2,
    run_sos_ump2,
    run_uhf,
    run_ump2,
)

from .conftest import ANGSTROM_TO_BOHR, GEOMETRIES


SOS_C_OS = 1.3
SOS_C_SS = 0.0


# ---------------------------------------------------------------------
# Closed-shell SOS-MP2.
# ---------------------------------------------------------------------

def _vibeqc_rhf(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    # See test_scs_mp2.py — H2O/cc-pvtz needs >100 iters at this
    # tolerance with the current SCF stack.
    opts.max_iter = 300
    hf = run_rhf(mol, basis, opts)
    assert hf.converged
    return mol, basis, hf


def _pyscf_dfmp2_components(atoms_bohr, basis_name, aux_basis_name):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf, mp

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged
    m = mp.MP2(mf, frozen=0).density_fit(auxbasis=aux_basis_name)
    m.kernel()
    return float(m.e_corr_os), float(m.e_corr_ss), float(m.e_corr)


@pytest.mark.parametrize(
    "mol_key,orb,aux",
    [
        ("H2O", "cc-pvdz", "cc-pvdz-ri"),
        ("H2O", "cc-pvtz", "cc-pvtz-ri"),
        ("CH4", "cc-pvdz", "cc-pvdz-ri"),
    ],
    ids=lambda v: v,
)
def test_sos_mp2_matches_pyscf_dfmp2(mol_key, orb, aux):
    """SOS-MP2 (1.3, 0) on RI-MP2 matches PySCF DFMP2 with the same
    coefficients."""
    atoms = GEOMETRIES[mol_key]
    mol, basis, hf = _vibeqc_rhf(atoms, orb)

    sos = run_sos_mp2(mol, basis, hf, aux_basis=aux, frozen_core=0)
    e_corr_os_ps, e_corr_ss_ps, _ = _pyscf_dfmp2_components(atoms, orb, aux)
    e_sos_ps = SOS_C_OS * e_corr_os_ps + SOS_C_SS * e_corr_ss_ps  # c_ss = 0

    delta = sos.e_correlation - e_sos_ps
    assert abs(delta) < 1e-9, (
        f"{mol_key}/{orb}/{aux}: vibeqc SOS-MP2 vs PySCF gap = "
        f"{delta:+.3e} Ha (vibeqc = {sos.e_correlation:.12f}, "
        f"PySCF = {e_sos_ps:.12f})"
    )


def test_sos_mp2_drops_same_spin():
    """c_ss = 0 means the same-spin contribution is completely dropped
    — e_correlation depends only on e_os and c_os."""
    mol, basis, hf = _vibeqc_rhf(GEOMETRIES["H2O"], "cc-pvdz")
    sos = run_sos_mp2(mol, basis, hf, aux_basis="cc-pvdz-ri")
    assert sos.e_correlation == pytest.approx(SOS_C_OS * sos.e_os, rel=1e-14)


def test_sos_mp2_is_less_negative_than_canonical():
    """SOS drops the same-spin term (which is negative for bound states)
    and scales opposite-spin by 1.3 vs canonical's 1. Net effect depends
    on the system, but for H2O / cc-pvdz the canonical correlation is
    more negative than SOS."""
    mol, basis, hf = _vibeqc_rhf(GEOMETRIES["H2O"], "cc-pvdz")
    canonical = run_mp2(mol, basis, hf)
    sos = run_sos_mp2(mol, basis, hf, density_fit=False)
    assert canonical.e_correlation < sos.e_correlation < 0


def test_sos_mp2_options_path_matches_wrapper():
    """The wrapper is a thin shim over MP2Options + run_mp2."""
    mol, basis, hf = _vibeqc_rhf(GEOMETRIES["H2O"], "cc-pvdz")
    opts = MP2Options()
    opts.n_frozen_core = 0
    opts.density_fit = True
    opts.aux_basis = "cc-pvdz-ri"
    opts.c_os = SOS_C_OS
    opts.c_ss = SOS_C_SS
    manual = run_mp2(mol, basis, hf, opts)
    wrapper = run_sos_mp2(
        mol, basis, hf, aux_basis="cc-pvdz-ri", frozen_core=0
    )
    assert manual.e_correlation == pytest.approx(
        wrapper.e_correlation, rel=1e-14)


# ---------------------------------------------------------------------
# Open-shell SOS-UMP2.
# ---------------------------------------------------------------------

def _vibeqc_uhf(atoms_bohr, basis_name, *, charge, mult):
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
        charge=charge, multiplicity=mult,
    )
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 500
    hf = run_uhf(mol, basis, opts)
    assert hf.converged
    return mol, basis, hf


def _pyscf_dfump2_components(atoms_bohr, basis_name, aux_basis_name,
                              *, charge, spin):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf, mp

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    m = mp.MP2(mf, frozen=0).density_fit(auxbasis=aux_basis_name)
    m.kernel()
    return float(m.e_corr_os), float(m.e_corr_ss), float(m.e_corr)


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult,aux",
    [
        ("OH-doublet",
         [(8, [0.0, 0.0, 0.0]),
          (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
         "cc-pvdz", 0, 2, "cc-pvdz-ri"),
        ("O2-triplet",
         [(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])],
         "cc-pvdz", 0, 3, "cc-pvdz-ri"),
    ],
    ids=["OH-doublet", "O2-triplet"],
)
def test_sos_ump2_matches_pyscf(label, atoms, basis_name, charge, mult, aux):
    """SOS-UMP2 keeps only the αβ channel scaled by 1.3."""
    mol, basis, hf = _vibeqc_uhf(atoms, basis_name, charge=charge, mult=mult)
    sos = run_sos_ump2(mol, basis, hf, aux_basis=aux, frozen_core=0)
    spin = mult - 1
    e_corr_os_ps, _, _ = _pyscf_dfump2_components(
        atoms, basis_name, aux, charge=charge, spin=spin)
    e_sos_ps = SOS_C_OS * e_corr_os_ps
    delta = sos.e_correlation - e_sos_ps
    assert abs(delta) < 1e-9, (
        f"{label}/{basis_name}/{aux}: vibeqc SOS-UMP2 vs PySCF gap = "
        f"{delta:+.3e} Ha (vibeqc = {sos.e_correlation:.12f}, "
        f"PySCF = {e_sos_ps:.12f})"
    )


def test_sos_ump2_drops_same_spin():
    """e_correlation == 1.3 * e_ab (e_aa, e_bb contribute via c_ss = 0)."""
    atoms = [(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])]
    mol, basis, hf = _vibeqc_uhf(atoms, "cc-pvdz", charge=0, mult=3)
    sos = run_sos_ump2(mol, basis, hf, aux_basis="cc-pvdz-ri")
    assert sos.e_correlation == pytest.approx(SOS_C_OS * sos.e_ab, rel=1e-14)
