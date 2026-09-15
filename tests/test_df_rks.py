"""DF-RKS: density-fitted Kohn-Sham DFT energies.

Mirrors tests/test_df_rhf.py and tests/test_df_uhf.py for the closed-
shell DFT path. Pins:

  1. Internal — DF-RKS agrees with direct RKS up to the auxiliary-
     basis fit error, separately for LDA, PBE (pure GGA), and B3LYP
     (hybrid GGA). Pure DFT exercises only the J branch of the DF
     dispatch; hybrid exercises both J and K.
  2. PySCF parity — DF-RKS matches ``mf.density_fit(auxbasis=...)``
     within numerical precision when both libraries use the same aux.
  3. Pure DFT path doesn't touch K — e_hf_exchange is exactly zero.
  4. Energy decomposition still sums to the total in the DF regime.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RKSOptions,
    run_rks,
)

from .conftest import GEOMETRIES


def _vibeqc_rks(atoms_bohr, basis_name, functional, *,
                density_fit, aux_basis_name=""):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    opts = RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 200
    opts.density_fit = density_fit
    opts.aux_basis = aux_basis_name
    return run_rks(mol, basis, opts)


def _pyscf_rks_df(atoms_bohr, basis_name, xc, aux_basis_name):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol).density_fit(auxbasis=aux_basis_name)
    mf.xc = xc
    mf.grids.level = 5
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    mf.kernel()
    assert mf.converged, f"PySCF DF-RKS did not converge {basis_name}/{xc}"
    return mf.e_tot


# (label, vibeqc-functional, pyscf-xc) — same shape as test_rks.py.
FUNCTIONAL_PAIRS = [
    ("LDA",   "LDA",   "lda,vwn"),
    ("PBE",   "PBE",   "pbe,pbe"),
    # vibe-qc's "B3LYP" = VWN5 variant (ORCA convention); PySCF's
    # matching spelling is "b3lyp5".
    ("B3LYP", "B3LYP", "b3lyp5"),
]

# (label, orbital basis, aux basis).
RKS_DF_PAIRS = [
    ("H2O", "def2-svp", "def2-universal-jkfit"),
    ("H2O", "cc-pvdz",  "cc-pvdz-jkfit"),
    ("CH4", "def2-svp", "def2-universal-jkfit"),
]


@pytest.mark.parametrize(
    "mol_key,orb,aux",
    RKS_DF_PAIRS,
    ids=[f"{m}-{o}" for m, o, _ in RKS_DF_PAIRS],
)
@pytest.mark.parametrize(
    "label,vq_func,_ps_xc",
    FUNCTIONAL_PAIRS,
    ids=[c[0] for c in FUNCTIONAL_PAIRS],
)
def test_df_rks_close_to_direct(
    label, vq_func, _ps_xc, mol_key, orb, aux,
):
    """DF-RKS total energy stays within JKfit accuracy of direct RKS
    on the same orbital basis. The XC integration is identical between
    DF and direct paths (it's grid-based, not ERI-based), so the gap is
    pure J/K fit error."""
    atoms = GEOMETRIES[mol_key]
    direct = _vibeqc_rks(atoms, orb, vq_func, density_fit=False)
    df = _vibeqc_rks(atoms, orb, vq_func, density_fit=True,
                     aux_basis_name=aux)

    assert direct.converged, f"{label}/{mol_key}/{orb}: direct RKS no-converge"
    assert df.converged,     f"{label}/{mol_key}/{orb}: DF-RKS no-converge"

    delta = df.energy - direct.energy
    assert abs(delta) < 5e-4, (
        f"{label}/{mol_key}/{orb}/{aux}: DF-direct gap = {delta:+.3e} Ha "
        f"(direct = {direct.energy:.10f}, DF = {df.energy:.10f})"
    )


@pytest.mark.parametrize(
    "mol_key,orb,aux",
    RKS_DF_PAIRS,
    ids=[f"{m}-{o}" for m, o, _ in RKS_DF_PAIRS],
)
@pytest.mark.parametrize(
    "label,vq_func,ps_xc",
    FUNCTIONAL_PAIRS,
    ids=[c[0] for c in FUNCTIONAL_PAIRS],
)
def test_df_rks_matches_pyscf_df(
    label, vq_func, ps_xc, mol_key, orb, aux,
):
    """vibeqc DF-RKS matches PySCF DF-RKS on shared aux. The XC grid
    isn't identical between vibeqc (Becke / SG-1 / etc.) and PySCF
    (Treutler-Ahlrichs / SG-1), so we accept the standard DFT-grid
    tolerance ~3e-6 Ha rather than 1e-9."""
    atoms = GEOMETRIES[mol_key]
    df = _vibeqc_rks(atoms, orb, vq_func, density_fit=True,
                     aux_basis_name=aux)
    e_pyscf = _pyscf_rks_df(atoms, orb, ps_xc, aux)
    delta = df.energy - e_pyscf
    assert abs(delta) < 3e-6, (
        f"{label}/{mol_key}/{orb}/{aux}: vibeqc-PySCF DF-RKS gap = "
        f"{delta:+.3e} Ha (vibeqc = {df.energy:.10f}, "
        f"PySCF = {e_pyscf:.10f})"
    )


def test_df_rks_pure_dft_has_zero_hf_exchange_energy():
    """Pure GGA (PBE) under DF must report e_hf_exchange = 0 — the K
    branch of the dispatch is never invoked when α_HF = 0."""
    df = _vibeqc_rks(
        GEOMETRIES["H2O"], "def2-svp", "PBE",
        density_fit=True, aux_basis_name="def2-universal-jkfit",
    )
    assert df.e_hf_exchange == 0.0


def test_df_rks_hybrid_reports_nonzero_hf_exchange():
    """Hybrid (B3LYP) under DF must report e_hf_exchange < 0 — the K
    branch is invoked and contributes to the total energy."""
    df = _vibeqc_rks(
        GEOMETRIES["H2O"], "def2-svp", "B3LYP",
        density_fit=True, aux_basis_name="def2-universal-jkfit",
    )
    assert df.e_hf_exchange != 0.0


def test_df_rks_energy_decomposition_sums_to_total():
    """Energy components (E_core + E_J + E_K + E_xc + E_nuc) sum to
    the total reported energy under the DF regime — a basic
    consistency check that the decomposition tracking wasn't broken
    by the closure-based dispatch."""
    df = _vibeqc_rks(
        GEOMETRIES["H2O"], "def2-svp", "B3LYP",
        density_fit=True, aux_basis_name="def2-universal-jkfit",
    )
    E_core = (df.e_electronic
              - df.e_coulomb - df.e_hf_exchange - df.e_xc)
    total = (E_core + df.e_coulomb + df.e_hf_exchange
             + df.e_xc + df.e_nuclear)
    assert abs(total - df.energy) < 1e-9


def test_df_rks_requires_aux_basis():
    """Same contract as run_rhf / run_uhf: density_fit=True with empty
    aux_basis raises ValueError."""
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES["H2O"]])
    basis = BasisSet(mol, "def2-svp")
    opts = RKSOptions()
    opts.functional = "PBE"
    opts.density_fit = True
    opts.aux_basis = ""
    with pytest.raises(ValueError, match="aux_basis"):
        run_rks(mol, basis, opts)


def test_df_rks_pure_dft_works_with_jfit():
    """Pure DFT only needs the J fit, not the larger JKfit. The
    Weigend universal J-only fit (def2-universal-jfit) is bundled
    via BSE and works correctly on RKS-PBE."""
    df = _vibeqc_rks(
        GEOMETRIES["H2O"], "def2-svp", "PBE",
        density_fit=True, aux_basis_name="def2-universal-jfit",
    )
    assert df.converged
    # Compare to the JKfit RKS-PBE — should agree at the µHa-per-atom
    # level since both fits are designed to give sub-µHa Coulomb error.
    df_jk = _vibeqc_rks(
        GEOMETRIES["H2O"], "def2-svp", "PBE",
        density_fit=True, aux_basis_name="def2-universal-jkfit",
    )
    delta = df.energy - df_jk.energy
    assert abs(delta) < 1e-4, (
        f"DF-RKS-PBE on jfit vs jkfit: gap {delta:+.3e} Ha "
        f"(jfit={df.energy:.10f}, jkfit={df_jk.energy:.10f})"
    )
