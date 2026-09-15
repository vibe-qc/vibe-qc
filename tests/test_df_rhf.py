"""DF-RHF: density-fitted closed-shell Hartree-Fock energies.

These tests pin three things:

  1. Internal consistency — DF-RHF agrees with direct RHF up to the
     auxiliary-basis fit error. For the bundled def2 / cc-pV*Z JKfit
     families on neutral organics this is well below the chemical-
     accuracy band (1e-4 Ha is comfortable; published Eichkorn 1995 /
     Weigend 2008 fits are often sub-µHa per atom).

  2. PySCF parity — DF-RHF matches PySCF's ``mf.density_fit(auxbasis=...)``
     to machine precision when the same orbital + auxiliary basis are
     used. PySCF uses the same Cholesky-of-V factorisation; energies
     and MO eigenvalues should agree to ≤ 1 µHa modulo orthogonalisation
     conventions.

  3. Error path — ``density_fit=True`` with empty ``aux_basis`` raises
     a clear ValueError (not a silent fallback).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    default_aux_basis_for,
    run_rhf,
)

from .conftest import GEOMETRIES


# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------

def _vibeqc_rhf(atoms_bohr, basis_name, *, density_fit, aux_basis_name=""):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.density_fit = density_fit
    opts.aux_basis = aux_basis_name
    return run_rhf(mol, basis, opts)


def _pyscf_rhf_df(atoms_bohr, basis_name, aux_basis_name):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol).density_fit(auxbasis=aux_basis_name)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged, f"PySCF DF-RHF did not converge on {basis_name}/{aux_basis_name}"
    return mf.e_tot, mf.mo_energy


# (orbital, aux) pairings for the vibe-qc-internal direct-vs-DF check.
# Covers the full bundled JKfit landscape: def2 per-zeta (Weigend 2008,
# JCC 29, 167) and Dunning cc-pV*Z (libint convention "*-jkfit").
DIRECT_VS_DF_PAIRS = [
    ("H2O", "def2-svp",   "def2-svp-jk"),
    ("H2O", "def2-tzvp",  "def2-tzvp-jk"),
    ("H2O", "cc-pvdz",    "cc-pvdz-jkfit"),
    ("CH4", "def2-svp",   "def2-svp-jk"),
    ("CH4", "def2-tzvp",  "def2-tzvp-jk"),
]

# Subset usable for PySCF cross-validation. PySCF and vibe-qc/libint share
# **the cc-pV*Z-JKfit files only**; for def2 orbital bases the two libraries
# ship *different* auxiliary bases:
#   - vibe-qc/libint follows the ORCA per-zeta convention: ``def2-svp-jk``,
#     ``def2-tzvp-jk``, ``def2-tzvpp-jk``, etc. (Weigend 2008, sized to
#     the orbital basis).
#   - PySCF ships Weigend's universal JKfit (``def2-universal-jkfit``)
#     applicable across the whole def2 family.
# Both are valid Coulomb-metric fits but they have different residuals,
# so a vibe-qc(def2-svp-jk) ↔ PySCF(def2-universal-jkfit) comparison would
# tangle the SCF energy difference with an aux-basis difference. Pinning
# parity tests to cc-pV*Z-JKfit gives an apples-to-apples cross-check;
# direct-vs-DF on def2 (above) covers the def2 path through vibe-qc
# alone.
PYSCF_PARITY_PAIRS = [
    # Dunning JK-fit pairings (libint and PySCF ship identical files).
    ("H2O", "cc-pvdz", "cc-pvdz-jkfit"),
    ("H2O", "cc-pvtz", "cc-pvtz-jkfit"),
    ("CH4", "cc-pvdz", "cc-pvdz-jkfit"),
    # def2 universal JKfit — BSE-fetched into vibe-qc, also PySCF's
    # default for the def2 family. Apples-to-apples cross-validation.
    ("H2O", "def2-svp",  "def2-universal-jkfit"),
    ("H2O", "def2-tzvp", "def2-universal-jkfit"),
    ("CH4", "def2-svp",  "def2-universal-jkfit"),
]


# -----------------------------------------------------------------------------
# DF-RHF total energy ↔ direct RHF total energy (fit-error-bounded).
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mol_key,orb,aux",
    DIRECT_VS_DF_PAIRS,
    ids=[f"{m}-{o}-{a}" for m, o, a in DIRECT_VS_DF_PAIRS],
)
def test_df_rhf_close_to_direct_rhf(mol_key, orb, aux):
    """DF-RHF total energy must lie within auxiliary-basis fit error of
    direct (four-index ERI) RHF. The published JKfit fit error on
    neutral organics is well below 1e-4 Ha total; pin at 5e-4 to leave
    slack for SCF-convergence-tail interactions."""
    atoms = GEOMETRIES[mol_key]
    direct = _vibeqc_rhf(atoms, orb, density_fit=False)
    df = _vibeqc_rhf(atoms, orb, density_fit=True, aux_basis_name=aux)

    assert direct.converged, f"direct RHF did not converge ({mol_key}/{orb})"
    assert df.converged, f"DF-RHF did not converge ({mol_key}/{orb}/{aux})"

    delta = df.energy - direct.energy
    assert abs(delta) < 5e-4, (
        f"{mol_key}/{orb}/{aux}: DF-direct gap = {delta:+.3e} Ha "
        f"(direct = {direct.energy:.10f}, DF = {df.energy:.10f})"
    )


# -----------------------------------------------------------------------------
# DF-RHF total energy ↔ PySCF DF-RHF total energy (machine precision).
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mol_key,orb,aux",
    PYSCF_PARITY_PAIRS,
    ids=[f"{m}-{o}-{a}" for m, o, a in PYSCF_PARITY_PAIRS],
)
def test_df_rhf_matches_pyscf_df(mol_key, orb, aux):
    """vibe-qc and PySCF both implement the same Cholesky-of-V RI
    factorisation with the same auxiliary basis, so the SCF energies
    must agree to numerical precision (1e-9 Ha is comfortable; PySCF
    uses scipy LAPACK Cholesky too)."""
    atoms = GEOMETRIES[mol_key]
    df = _vibeqc_rhf(atoms, orb, density_fit=True, aux_basis_name=aux)
    e_pyscf, _ = _pyscf_rhf_df(atoms, orb, aux)

    delta = df.energy - e_pyscf
    assert abs(delta) < 1e-9, (
        f"{mol_key}/{orb}/{aux}: vibeqc-PySCF DF gap = {delta:+.3e} Ha "
        f"(vibeqc = {df.energy:.12f}, PySCF = {e_pyscf:.12f})"
    )


@pytest.mark.parametrize(
    "mol_key,orb,aux",
    [("H2O", "cc-pvdz", "cc-pvdz-jkfit"),
     ("CH4", "cc-pvdz", "cc-pvdz-jkfit")],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_df_rhf_mo_energies_match_pyscf(mol_key, orb, aux):
    atoms = GEOMETRIES[mol_key]
    df = _vibeqc_rhf(atoms, orb, density_fit=True, aux_basis_name=aux)
    _, mo_pyscf = _pyscf_rhf_df(atoms, orb, aux)

    mo_vq = np.array(df.mo_energies)
    np.testing.assert_allclose(
        np.sort(mo_vq), np.sort(mo_pyscf), atol=1e-8,
        err_msg=f"DF-RHF MO energies disagree with PySCF "
                f"on {mol_key}/{orb}/{aux}",
    )


# -----------------------------------------------------------------------------
# Error paths and edge cases.
# -----------------------------------------------------------------------------

def test_df_rhf_requires_aux_basis():
    """density_fit=True with empty aux_basis must raise — no silent
    fallback to default-aux selection on the C++ side. Autodetection
    is a Python-side concern (vibeqc.default_aux_basis_for)."""
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES["H2O"]])
    basis = BasisSet(mol, "def2-svp")
    opts = RHFOptions()
    opts.density_fit = True
    opts.aux_basis = ""
    with pytest.raises(ValueError, match="aux_basis"):
        run_rhf(mol, basis, opts)


def test_df_rhf_autodetect_aux_basis_via_python_helper():
    """The supported workflow for autodetection: resolve aux name in
    Python via default_aux_basis_for, pass through to the SCF driver."""
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES["H2O"]])
    basis = BasisSet(mol, "def2-tzvp")
    aux_name = default_aux_basis_for("def2-tzvp", kind="jk")
    assert aux_name == "def2-tzvp-jk"

    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.density_fit = True
    opts.aux_basis = aux_name
    result = run_rhf(mol, basis, opts)
    assert result.converged


def test_df_rhf_iteration_count_close_to_direct():
    """DF should not materially change the SCF convergence path. With
    SAD guess + DIIS, DF and direct RHF on H2O / def2-svp should
    converge in roughly the same number of iterations (within ±2)."""
    atoms = GEOMETRIES["H2O"]
    direct = _vibeqc_rhf(atoms, "def2-svp", density_fit=False)
    df = _vibeqc_rhf(atoms, "def2-svp", density_fit=True,
                     aux_basis_name="def2-svp-jk")
    assert abs(direct.n_iter - df.n_iter) <= 2, (
        f"DF iteration count {df.n_iter} differs materially from direct "
        f"{direct.n_iter} (>2 iters apart)."
    )


def test_df_rhf_density_is_idempotent():
    """A converged density satisfies D · S · D = 2D for closed-shell
    (idempotent up to the factor of 2 from D = 2 C_occ C_occ^T). This
    holds for any correct RHF implementation, regardless of DF or
    direct path; checking it here guards against a subtle bug where
    the converged D doesn't actually correspond to the converged C."""
    atoms = GEOMETRIES["H2O"]
    df = _vibeqc_rhf(atoms, "def2-svp", density_fit=True,
                     aux_basis_name="def2-svp-jk")
    from vibeqc import compute_overlap
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")
    S = compute_overlap(basis)
    D = np.array(df.density)
    DSD = D @ S @ D
    np.testing.assert_allclose(DSD, 2.0 * D, atol=1e-8)
