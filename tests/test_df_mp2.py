"""DF-MP2: density-fitted closed-shell second-order Møller-Plesset.

Pins:

  1. Internal — DF-MP2 correlation energy agrees with direct MP2 up to
     the auxiliary-basis fit error of the chosen RI aux. The Weigend
     RIfit family is designed to give sub-µHa/atom correlation-energy
     fit error on neutral organics.
  2. PySCF parity — DF-MP2 matches
     ``pyscf.mp.MP2(mf, frozen=0).density_fit(auxbasis=...)`` to machine
     precision when both libraries use the same direct-RHF reference,
     all-electron protocol, orbital basis, and explicit RI auxiliary basis.
  3. Spin decomposition (e_ss + e_os = e_correlation) holds under DF
     identically to the direct path.
  4. The DF path requires aux_basis to be set explicitly — empty
     aux_basis + density_fit=True raises.
  5. RI-MP2 with the JKfit aux (over-fit but technically valid) also
     converges to a sensible answer — guards against the case where a
     user accidentally passes a JKfit name to MP2.
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
    default_aux_basis_for,
    run_mp2,
    run_rhf,
)

from .conftest import GEOMETRIES


def _all_electron_mp2_options():
    """Preserve the pre-#140 all-electron DF/reference evidence."""
    options = MP2Options()
    options.n_frozen_core = 0
    return options


def _vibeqc_mp2(atoms_bohr, basis_name, *,
                density_fit, aux_basis_name="",
                rhf_density_fit=False, rhf_aux_basis=""):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    # cc-pvtz on H2O at tight tolerances takes ~150-200 iters with the
    # current EDIIS+DIIS accelerator; the framework default max_iter=100
    # cuts it off short of convergence. Matches the workaround already
    # in tests/test_scs_mp2.py:63-66 and tests/test_sos_mp2.py.
    rhf_opts.max_iter = 300
    rhf_opts.density_fit = rhf_density_fit
    rhf_opts.aux_basis = rhf_aux_basis
    hf = run_rhf(mol, basis, rhf_opts)
    assert hf.converged

    mp2_opts = _all_electron_mp2_options()
    mp2_opts.density_fit = density_fit
    mp2_opts.aux_basis = aux_basis_name
    return hf, run_mp2(mol, basis, hf, mp2_opts)


def _pyscf_mp2_df(atoms_bohr, basis_name, aux_basis_name):
    """RHF (direct) → DF-MP2 with the requested aux. PySCF DF-MP2 reads
    the SCF reference; we run vanilla SCF here so the test isolates the
    MP2 DF correctness from any SCF-DF differences."""
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
    mf.kernel()
    assert mf.converged
    m = mp.MP2(mf, frozen=0).density_fit(auxbasis=aux_basis_name)
    m.kernel()
    # m.e_tot is already mf.e_tot + m.e_corr — don't double-add.
    return mf.e_tot, m.e_corr, m.e_tot


# ---------------------------------------------------------------------
# Internal direct↔DF parity (every test case uses the SAME orbital basis
# and direct RHF reference — only the MP2 step differs).
# ---------------------------------------------------------------------

DIRECT_VS_DF_CASES = [
    ("H2O", "def2-svp",  "def2-svp-rifit"),
    ("H2O", "def2-tzvp", "def2-tzvp-rifit"),
    ("H2O", "cc-pvdz",   "cc-pvdz-ri"),
    ("CH4", "def2-svp",  "def2-svp-rifit"),
    ("CH4", "cc-pvdz",   "cc-pvdz-ri"),
]


@pytest.mark.parametrize(
    "mol_key,orb,aux",
    DIRECT_VS_DF_CASES,
    ids=[f"{m}-{o}" for m, o, _ in DIRECT_VS_DF_CASES],
)
def test_df_mp2_close_to_direct(mol_key, orb, aux):
    """RI-MP2 correlation energy agrees with direct MP2 up to RIfit
    accuracy. For H2O / def2-svp / def2-svp-rifit the published
    Weigend-Häser-Patzelt-Ahlrichs fit error is sub-µHa per atom; pin
    at 1e-4 Ha total to leave slack."""
    atoms = GEOMETRIES[mol_key]
    _, direct = _vibeqc_mp2(atoms, orb, density_fit=False)
    _, df = _vibeqc_mp2(atoms, orb, density_fit=True,
                         aux_basis_name=aux)
    delta = df.e_correlation - direct.e_correlation
    assert abs(delta) < 1e-4, (
        f"{mol_key}/{orb}/{aux}: DF-MP2 vs direct correlation gap = "
        f"{delta:+.3e} Ha (direct = {direct.e_correlation:.10f}, "
        f"DF = {df.e_correlation:.10f})"
    )


# ---------------------------------------------------------------------
# PySCF parity. Use the cc-pV*Z-RI family (shared between vibe-qc and
# PySCF as identical .dat / .g94 files).
# ---------------------------------------------------------------------

PYSCF_PARITY_CASES = [
    ("H2O", "cc-pvdz", "cc-pvdz-ri"),
    ("H2O", "cc-pvtz", "cc-pvtz-ri"),
    ("CH4", "cc-pvdz", "cc-pvdz-ri"),
]


@pytest.mark.parametrize(
    "mol_key,orb,aux",
    PYSCF_PARITY_CASES,
    ids=[f"{m}-{o}" for m, o, _ in PYSCF_PARITY_CASES],
)
def test_df_mp2_matches_pyscf_df(mol_key, orb, aux):
    """vibe-qc and PySCF both use the same Cholesky-of-V RI factorisation
    + the same MP2 contraction structure. With identical SCF references
    (vanilla RHF on both sides at conv_tol=1e-12) the DF-MP2 correlation
    should agree to machine precision."""
    atoms = GEOMETRIES[mol_key]
    _, df = _vibeqc_mp2(atoms, orb, density_fit=True,
                         aux_basis_name=aux)
    e_hf_ps, e_corr_ps, e_total_ps = _pyscf_mp2_df(atoms, orb, aux)
    delta_corr = df.e_correlation - e_corr_ps
    delta_total = df.e_total - e_total_ps
    assert abs(delta_corr) < 1e-9, (
        f"{mol_key}/{orb}/{aux}: vibeqc-PySCF DF-MP2 corr gap = "
        f"{delta_corr:+.3e} Ha (vibeqc = {df.e_correlation:.12f}, "
        f"PySCF = {e_corr_ps:.12f})"
    )
    assert abs(delta_total) < 1e-9


# ---------------------------------------------------------------------
# Energy decomposition consistency.
# ---------------------------------------------------------------------

def test_df_mp2_spin_decomposition_sums_to_total():
    """Same-spin + opposite-spin = total correlation under DF."""
    _, df = _vibeqc_mp2(
        GEOMETRIES["H2O"], "def2-svp",
        density_fit=True, aux_basis_name="def2-svp-rifit",
    )
    assert df.e_ss + df.e_os == pytest.approx(df.e_correlation, rel=1e-12)
    assert df.e_hf + df.e_correlation == pytest.approx(df.e_total, rel=1e-14)


def test_df_mp2_correlation_is_negative():
    """For a bound state, MP2 correlation is strictly negative (energy
    denominators ε_i + ε_j − ε_a − ε_b < 0)."""
    _, df = _vibeqc_mp2(
        GEOMETRIES["H2O"], "def2-svp",
        density_fit=True, aux_basis_name="def2-svp-rifit",
    )
    assert df.e_correlation < 0


# ---------------------------------------------------------------------
# Workflow tests.
# ---------------------------------------------------------------------

def test_df_mp2_requires_aux_basis():
    """density_fit=True with empty aux_basis raises ValueError — same
    contract as run_rhf / run_uhf / run_rks / run_uks."""
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES["H2O"]])
    basis = BasisSet(mol, "def2-svp")
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    hf = run_rhf(mol, basis, rhf_opts)
    mp2_opts = _all_electron_mp2_options()
    mp2_opts.density_fit = True
    mp2_opts.aux_basis = ""
    with pytest.raises(ValueError, match="aux_basis"):
        run_mp2(mol, basis, hf, mp2_opts)


def test_df_mp2_autodetect_via_python_helper():
    """The supported workflow for autodetecting an MP2 RI aux."""
    aux_name = default_aux_basis_for("def2-tzvp", kind="ri")
    assert aux_name == "def2-tzvp-rifit"
    _, df = _vibeqc_mp2(
        GEOMETRIES["H2O"], "def2-tzvp",
        density_fit=True, aux_basis_name=aux_name,
    )
    assert df.e_correlation < 0


def test_df_mp2_with_jkfit_aux_works():
    """Using a JKfit aux for MP2 is technically valid (over-fit, but
    correct). Guards against a silent failure if a user copies the
    aux from an HF DF setup."""
    _, df = _vibeqc_mp2(
        GEOMETRIES["H2O"], "def2-svp",
        density_fit=True, aux_basis_name="def2-universal-jkfit",
    )
    # Compare to RIfit MP2 — should agree at the JK-vs-RI fit-difference
    # level. JKfit is over-fit for correlation, but the difference is
    # well below 1 mHa for H2O.
    _, df_ri = _vibeqc_mp2(
        GEOMETRIES["H2O"], "def2-svp",
        density_fit=True, aux_basis_name="def2-svp-rifit",
    )
    assert abs(df.e_correlation - df_ri.e_correlation) < 1e-3


def test_df_mp2_with_df_rhf_reference():
    """End-to-end DF: DF-RHF reference + DF-MP2 correlation. Both can
    be DF-ed independently; this is the cheapest path for production
    workflows."""
    atoms = GEOMETRIES["H2O"]
    _, df = _vibeqc_mp2(
        atoms, "def2-svp",
        density_fit=True, aux_basis_name="def2-svp-rifit",
        rhf_density_fit=True, rhf_aux_basis="def2-svp-jk",
    )
    assert df.e_correlation < 0
    # Compare to all-direct MP2: total should be within JKfit + RIfit
    # combined fit error (~µHa).
    _, direct = _vibeqc_mp2(atoms, "def2-svp", density_fit=False)
    delta = df.e_total - direct.e_total
    assert abs(delta) < 1e-3


def test_df_mp2_ri_residual_diagnostic():
    """``MP2Options.report_ri_residual`` surfaces the per-bucket Dunlap
    fit residual on ``MP2Result.{e_os_ri_residual, e_ss_ri_residual}``.

    The Coulomb-metric Dunlap fit on a finite RIfit aux gives
    structurally biased buckets (``E_os`` over-estimated relative to
    canonical, ``E_ss`` under-estimated) that partially cancel in the
    unscaled sum but survive SCS/SOS scaling. This test pins the
    diagnostic plumbing: off by default, only fires when both flags
    are set, matches an independently computed canonical reference.
    """
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf_opts.max_iter = 300
    hf = run_rhf(mol, basis, rhf_opts)
    assert hf.converged

    # Diagnostic off → flag false, residuals untouched (zero).
    opts = _all_electron_mp2_options()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-rifit"
    mp_off = run_mp2(mol, basis, hf, opts)
    assert mp_off.ri_residual_reported is False
    assert mp_off.e_os_ri_residual == 0.0
    assert mp_off.e_ss_ri_residual == 0.0

    # Diagnostic on with density_fit=True → residuals populated.
    opts.report_ri_residual = True
    mp_on = run_mp2(mol, basis, hf, opts)
    assert mp_on.ri_residual_reported is True

    # Cross-check: the residual must equal (RI − canonical) computed
    # independently on the same RHF.
    opts_can = _all_electron_mp2_options()
    mp_can = run_mp2(mol, basis, hf, opts_can)
    assert mp_on.e_os_ri_residual == pytest.approx(
        mp_on.e_os - mp_can.e_os, abs=1e-12)
    assert mp_on.e_ss_ri_residual == pytest.approx(
        mp_on.e_ss - mp_can.e_ss, abs=1e-12)

    # Diagnostic flag with density_fit=False → does NOT fire (canonical
    # path has no residual to report).
    opts.density_fit = False
    mp_can_with_flag = run_mp2(mol, basis, hf, opts)
    assert mp_can_with_flag.ri_residual_reported is False
    assert mp_can_with_flag.e_os_ri_residual == 0.0
