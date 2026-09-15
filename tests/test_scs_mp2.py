"""SCS-MP2 (Grimme, J. Chem. Phys. 118, 9095 (2003)).

Pins:

  1. Spin-component coefficients ``c_os, c_ss`` compose with the existing
     MP2 (canonical) and DF-MP2 (RI) paths without code duplication —
     both paths share the same ``(e_os, e_ss)`` accumulation, scaling
     is applied at the final sum.
  2. With c_os = c_ss = 1 (the default of MP2Options), the result
     reproduces canonical/RI MP2 to machine precision.
  3. Grimme's SCS-MP2 (c_os = 6/5, c_ss = 1/3) matches PySCF's
     spin-component split applied with the same coefficients: vibe-qc
     and PySCF use the same Grimme decomposition
       E_os = Σ (ia|jb)² / Δ
       E_ss = Σ [(ia|jb)² − (ia|jb)(ib|ja)] / Δ
     so the scaled sum agrees at the MP2-energy level.
  4. The ``run_scs_mp2`` convenience wrapper defaults to RI-MP2 with the
     orbital basis's per-zeta RIfit aux auto-resolved.
  5. The UMP2 analogue ``run_scs_ump2`` applies the same coefficients
     per Grimme channel definition (αβ = opposite-spin, αα+ββ =
     same-spin) and matches PySCF UMP2 with the same scales.
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
    default_aux_basis_for,
    run_mp2,
    run_rhf,
    run_scs_mp2,
    run_scs_ump2,
    run_uhf,
    run_ump2,
)

from .conftest import ANGSTROM_TO_BOHR, GEOMETRIES


SCS_C_OS = 6.0 / 5.0
SCS_C_SS = 1.0 / 3.0


# ---------------------------------------------------------------------
# Closed-shell SCS-MP2.
# ---------------------------------------------------------------------

def _vibeqc_rhf(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    # cc-pvtz on H2O at tight tolerances takes ~150-200 iters with the
    # current EDIIS+DIIS accelerator; the framework default max_iter=100
    # cuts it off short of convergence.
    opts.max_iter = 300
    hf = run_rhf(mol, basis, opts)
    assert hf.converged
    return mol, basis, hf


def _pyscf_dfmp2_components(atoms_bohr, basis_name, aux_basis_name):
    """Reference DF-MP2 with PySCF, returning (e_corr_os, e_corr_ss,
    e_corr_total). PySCF's e_corr_os / e_corr_ss use the Grimme spin-
    component split, identical to vibe-qc's convention."""
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
    e_corr_os = float(m.e_corr_os)
    e_corr_ss = float(m.e_corr_ss)
    return e_corr_os, e_corr_ss, float(m.e_corr)


@pytest.mark.parametrize(
    "mol_key,orb,aux",
    [
        ("H2O", "cc-pvdz", "cc-pvdz-ri"),
        ("H2O", "cc-pvtz", "cc-pvtz-ri"),
        ("CH4", "cc-pvdz", "cc-pvdz-ri"),
    ],
    ids=lambda v: v,
)
def test_scs_mp2_matches_pyscf_dfmp2(mol_key, orb, aux):
    """SCS-MP2 (Grimme 6/5, 1/3) on RI-MP2 matches PySCF's DFMP2 with
    the same coefficients. Both codes use the Grimme decomposition,
    so the scaled sum agrees at machine precision when the unscaled
    correlation matches (the existing DF-MP2 PySCF parity test
    already pins ~1e-9 Ha on these cases)."""
    atoms = GEOMETRIES[mol_key]
    mol, basis, hf = _vibeqc_rhf(atoms, orb)

    # vibe-qc SCS-MP2 via the convenience wrapper.
    scs = run_scs_mp2(mol, basis, hf, aux_basis=aux, frozen_core=0)

    e_corr_os_ps, e_corr_ss_ps, _ = _pyscf_dfmp2_components(atoms, orb, aux)
    e_scs_ps = SCS_C_OS * e_corr_os_ps + SCS_C_SS * e_corr_ss_ps

    delta = scs.e_correlation - e_scs_ps
    assert abs(delta) < 1e-9, (
        f"{mol_key}/{orb}/{aux}: vibeqc SCS-MP2 vs PySCF gap = "
        f"{delta:+.3e} Ha (vibeqc = {scs.e_correlation:.12f}, "
        f"PySCF = {e_scs_ps:.12f})"
    )


def test_scs_mp2_canonical_recovery():
    """c_os = c_ss = 1 recovers canonical RMP2 (same kernel, scale
    factors composed at the final sum)."""
    mol, basis, hf = _vibeqc_rhf(GEOMETRIES["H2O"], "cc-pvdz")
    canonical = run_mp2(mol, basis, hf)
    scaled = run_scs_mp2(mol, basis, hf, density_fit=False,
                         c_os=1.0, c_ss=1.0)
    assert scaled.e_correlation == pytest.approx(
        canonical.e_correlation, rel=1e-12)


def test_scs_mp2_scaling_invariant():
    """Internal decomposition: e_correlation = c_os * e_os + c_ss * e_ss."""
    mol, basis, hf = _vibeqc_rhf(GEOMETRIES["H2O"], "cc-pvdz")
    scs = run_scs_mp2(mol, basis, hf, aux_basis="cc-pvdz-ri")
    assert scs.e_correlation == pytest.approx(
        SCS_C_OS * scs.e_os + SCS_C_SS * scs.e_ss, rel=1e-14)


def test_scs_mp2_default_is_ri_with_autoresolved_aux():
    """Calling run_scs_mp2 with the default density_fit=True and an
    empty aux_basis auto-resolves to the per-zeta RIfit aux for the
    orbital basis."""
    mol, basis, hf = _vibeqc_rhf(GEOMETRIES["H2O"], "def2-tzvp")
    # Should pick def2-tzvp-rifit.
    scs_auto = run_scs_mp2(mol, basis, hf)
    scs_explicit = run_scs_mp2(mol, basis, hf,
                                aux_basis=default_aux_basis_for(
                                    "def2-tzvp", kind="ri"))
    assert scs_auto.e_correlation == pytest.approx(
        scs_explicit.e_correlation, rel=1e-14)


def test_scs_mp2_options_path_matches_wrapper():
    """The convenience wrapper is a thin shim over MP2Options + run_mp2 —
    setting c_os/c_ss/density_fit/aux_basis by hand reproduces it."""
    mol, basis, hf = _vibeqc_rhf(GEOMETRIES["H2O"], "cc-pvdz")
    opts = MP2Options()
    opts.n_frozen_core = 0
    opts.density_fit = True
    opts.aux_basis = "cc-pvdz-ri"
    opts.c_os = SCS_C_OS
    opts.c_ss = SCS_C_SS
    manual = run_mp2(mol, basis, hf, opts)
    wrapper = run_scs_mp2(
        mol, basis, hf, aux_basis="cc-pvdz-ri", frozen_core=0
    )
    assert manual.e_correlation == pytest.approx(
        wrapper.e_correlation, rel=1e-14)


def test_scs_mp2_wrapper_threads_report_ri_residual():
    """``report_ri_residual`` reaches MP2Options through the wrapper:
    off by default, on when requested, and the residual it surfaces
    equals the one a hand-built MP2Options run produces."""
    mol, basis, hf = _vibeqc_rhf(GEOMETRIES["H2O"], "cc-pvdz")

    # Default: diagnostic off.
    off = run_scs_mp2(
        mol, basis, hf, aux_basis="cc-pvdz-ri", frozen_core=0
    )
    assert off.ri_residual_reported is False
    assert off.e_os_ri_residual == 0.0

    # Opt-in via the wrapper kwarg.
    on = run_scs_mp2(
        mol,
        basis,
        hf,
        aux_basis="cc-pvdz-ri",
        frozen_core=0,
        report_ri_residual=True,
    )
    assert on.ri_residual_reported is True

    # Identical to the hand-built MP2Options path.
    opts = MP2Options()
    opts.n_frozen_core = 0
    opts.density_fit = True
    opts.aux_basis = "cc-pvdz-ri"
    opts.c_os = SCS_C_OS
    opts.c_ss = SCS_C_SS
    opts.report_ri_residual = True
    manual = run_mp2(mol, basis, hf, opts)
    assert on.e_os_ri_residual == pytest.approx(
        manual.e_os_ri_residual, abs=1e-12)
    assert on.e_ss_ri_residual == pytest.approx(
        manual.e_ss_ri_residual, abs=1e-12)


# ---------------------------------------------------------------------
# Open-shell SCS-UMP2.
# ---------------------------------------------------------------------

UMP2_CASES = [
    ("OH-doublet",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
     "cc-pvdz", 0, 2, "cc-pvdz-ri"),
    ("O2-triplet",
     [(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])],
     "cc-pvdz", 0, 3, "cc-pvdz-ri"),
]


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
    # PySCF UMP2 reports e_corr_ss (αα + ββ) and e_corr_os (αβ).
    return float(m.e_corr_os), float(m.e_corr_ss), float(m.e_corr)


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult,aux",
    UMP2_CASES,
    ids=[c[0] for c in UMP2_CASES],
)
def test_scs_ump2_matches_pyscf(label, atoms, basis_name, charge, mult, aux):
    """SCS-UMP2 = c_os * e_ab + c_ss * (e_aa + e_bb) — matches PySCF
    DFUMP2 with the same coefficients to machine precision (modulo the
    underlying DF-UMP2 / PySCF parity, which is already pinned)."""
    mol, basis, hf = _vibeqc_uhf(atoms, basis_name, charge=charge, mult=mult)
    scs = run_scs_ump2(mol, basis, hf, aux_basis=aux, frozen_core=0)
    spin = mult - 1
    e_corr_os_ps, e_corr_ss_ps, _ = _pyscf_dfump2_components(
        atoms, basis_name, aux, charge=charge, spin=spin)
    e_scs_ps = SCS_C_OS * e_corr_os_ps + SCS_C_SS * e_corr_ss_ps
    delta = scs.e_correlation - e_scs_ps
    assert abs(delta) < 1e-9, (
        f"{label}/{basis_name}/{aux}: vibeqc SCS-UMP2 vs PySCF gap = "
        f"{delta:+.3e} Ha (vibeqc = {scs.e_correlation:.12f}, "
        f"PySCF = {e_scs_ps:.12f})"
    )


def test_scs_ump2_canonical_recovery():
    """c_os = c_ss = 1 recovers canonical UMP2."""
    atoms = [(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])]
    mol, basis, hf = _vibeqc_uhf(atoms, "cc-pvdz", charge=0, mult=3)
    canonical = run_ump2(mol, basis, hf)
    scaled = run_scs_ump2(mol, basis, hf, density_fit=False,
                           c_os=1.0, c_ss=1.0)
    assert scaled.e_correlation == pytest.approx(
        canonical.e_correlation, rel=1e-12)


def test_scs_ump2_scaling_invariant():
    """e_correlation = c_os * e_ab + c_ss * (e_aa + e_bb)."""
    atoms = [(8, [0.0, 0.0, 0.0]), (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])]
    mol, basis, hf = _vibeqc_uhf(atoms, "cc-pvdz", charge=0, mult=2)
    scs = run_scs_ump2(mol, basis, hf, aux_basis="cc-pvdz-ri")
    assert scs.e_correlation == pytest.approx(
        SCS_C_OS * scs.e_ab + SCS_C_SS * (scs.e_aa + scs.e_bb), rel=1e-14)


def test_scs_ump2_wrapper_threads_report_ri_residual():
    """``report_ri_residual`` reaches UMP2Options through the wrapper:
    off by default, on when requested, all three channel residuals
    populated."""
    atoms = [(8, [0.0, 0.0, 0.0]), (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])]
    mol, basis, hf = _vibeqc_uhf(atoms, "cc-pvdz", charge=0, mult=2)

    off = run_scs_ump2(
        mol, basis, hf, aux_basis="cc-pvdz-ri", frozen_core=0
    )
    assert off.ri_residual_reported is False
    assert off.e_ab_ri_residual == 0.0

    on = run_scs_ump2(
        mol,
        basis,
        hf,
        aux_basis="cc-pvdz-ri",
        frozen_core=0,
        report_ri_residual=True,
    )
    assert on.ri_residual_reported is True

    # Equals the hand-built UMP2Options path.
    opts = UMP2Options()
    opts.n_frozen_core = 0
    opts.density_fit = True
    opts.aux_basis = "cc-pvdz-ri"
    opts.c_os = SCS_C_OS
    opts.c_ss = SCS_C_SS
    opts.report_ri_residual = True
    manual = run_ump2(mol, basis, hf, opts)
    assert on.e_aa_ri_residual == pytest.approx(
        manual.e_aa_ri_residual, abs=1e-12)
    assert on.e_bb_ri_residual == pytest.approx(
        manual.e_bb_ri_residual, abs=1e-12)
    assert on.e_ab_ri_residual == pytest.approx(
        manual.e_ab_ri_residual, abs=1e-12)
