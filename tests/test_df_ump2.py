"""DF-UMP2: density-fitted open-shell second-order Møller-Plesset.

Mirrors tests/test_df_mp2.py for the open-shell UMP2 path. The three
spin channels (αα, ββ, αβ) all route through the DF factorisation;
α and β MO B-tensors are built once each and the OVOV blocks form via
single GEMMs (αα = B_α^T·B_α, ββ = B_β^T·B_β, αβ = B_α^T·B_β).

Pins:

  1. Internal — DF-UMP2 correlation agrees with direct UMP2 to RIfit
     accuracy across H, OH, and O₂ doublet/triplet references.
  2. PySCF parity on shared cc-pV*Z-RI aux.
  3. Closed-shell consistency — DF-UMP2 on a singlet reference (D_α =
     D_β) reproduces DF-RMP2 (n_alpha = n_beta channels collapse).
  4. Channel decomposition (αα + ββ + αβ = e_correlation) holds under DF.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    InitialGuess,
    MP2Options,
    Molecule,
    RHFOptions,
    UHFOptions,
    UMP2Options,
    run_mp2,
    run_rhf,
    run_uhf,
    run_ump2,
)

from .conftest import GEOMETRIES


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _all_electron_mp2_options():
    options = MP2Options()
    options.n_frozen_core = 0
    return options


def _all_electron_ump2_options():
    """Preserve the pre-#140 all-electron DF/reference evidence."""
    options = UMP2Options()
    options.n_frozen_core = 0
    return options


def _vibeqc_ump2(atoms_bohr, basis_name, charge, mult, *,
                 density_fit, aux_basis_name=""):
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
        charge=charge, multiplicity=mult,
    )
    basis = BasisSet(mol, basis_name)
    uhf_opts = UHFOptions()
    uhf_opts.conv_tol_energy = 1e-12
    uhf_opts.conv_tol_grad = 1e-8
    uhf_opts.max_iter = 500
    hf = run_uhf(mol, basis, uhf_opts)
    assert hf.converged

    ump2_opts = _all_electron_ump2_options()
    ump2_opts.density_fit = density_fit
    ump2_opts.aux_basis = aux_basis_name
    return hf, run_ump2(mol, basis, hf, ump2_opts)


def _pyscf_ump2_df(atoms_bohr, basis_name, aux_basis_name, *,
                   charge, spin):
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
    return mf.e_tot, m.e_corr, m.e_tot


# ---------------------------------------------------------------------
# Internal direct↔DF parity for open-shell systems.
# ---------------------------------------------------------------------

DIRECT_VS_DF_OPEN_CASES = [
    ("OH-doublet",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
     "def2-svp", 0, 2, "def2-svp-rifit"),
    ("OH-doublet-cc",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
     "cc-pvdz", 0, 2, "cc-pvdz-ri"),
    ("O2-triplet",
     [(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])],
     "cc-pvdz", 0, 3, "cc-pvdz-ri"),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult,aux",
    DIRECT_VS_DF_OPEN_CASES,
    ids=[c[0] for c in DIRECT_VS_DF_OPEN_CASES],
)
def test_df_ump2_close_to_direct(label, atoms, basis_name, charge, mult, aux):
    """RI-UMP2 correlation matches direct UMP2 to RIfit accuracy. The
    Weigend RIfit family is sized for sub-µHa-per-atom error on the
    correlation energy of typical organics; for first-row open-shell
    systems with concentrated correlation (O2 triplet at cc-pvdz: 250
    mHa correlation, ~0.4 mHa RI fit error from the two oxygens) the
    gap can reach the 1e-3 Ha range. Tolerance is sized to the worst
    case in the parametrisation rather than the best."""
    _, direct = _vibeqc_ump2(atoms, basis_name, charge, mult,
                             density_fit=False)
    _, df = _vibeqc_ump2(atoms, basis_name, charge, mult,
                         density_fit=True, aux_basis_name=aux)
    delta = df.e_correlation - direct.e_correlation
    assert abs(delta) < 1e-3, (
        f"{label}/{basis_name}/{aux}: DF-UMP2 vs direct corr gap = "
        f"{delta:+.3e} Ha (direct = {direct.e_correlation:.10f}, "
        f"DF = {df.e_correlation:.10f})"
    )


# ---------------------------------------------------------------------
# PySCF parity on shared cc-pV*Z-RI aux.
# ---------------------------------------------------------------------

PYSCF_PARITY_OPEN_CASES = [
    ("OH-doublet",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
     "cc-pvdz", 0, 2, "cc-pvdz-ri"),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult,aux",
    PYSCF_PARITY_OPEN_CASES,
    ids=[c[0] for c in PYSCF_PARITY_OPEN_CASES],
)
def test_df_ump2_matches_pyscf_df(
    label, atoms, basis_name, charge, mult, aux,
):
    """vibe-qc DF-UMP2 matches PySCF DF-UMP2 on a shared aux basis to
    machine precision (both libraries use the same Cholesky-of-V RI
    factorisation). Same caveat as DF-RMP2 PySCF parity: requires both
    libraries' UHF references to converge cleanly to identical
    energies, which holds for OH/cc-pvdz."""
    _, df = _vibeqc_ump2(atoms, basis_name, charge, mult,
                         density_fit=True, aux_basis_name=aux)
    spin = mult - 1
    e_hf_ps, e_corr_ps, e_total_ps = _pyscf_ump2_df(
        atoms, basis_name, aux, charge=charge, spin=spin,
    )
    delta_corr = df.e_correlation - e_corr_ps
    assert abs(delta_corr) < 1e-9, (
        f"{label}/{basis_name}/{aux}: vibeqc-PySCF DF-UMP2 corr gap = "
        f"{delta_corr:+.3e} Ha (vibeqc = {df.e_correlation:.12f}, "
        f"PySCF = {e_corr_ps:.12f})"
    )


# ---------------------------------------------------------------------
# Spin-channel decomposition.
# ---------------------------------------------------------------------

def test_df_ump2_channel_sum_equals_correlation():
    _, df = _vibeqc_ump2(
        [(8, [0.0, 0.0, 0.0]),
         (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
        "def2-svp", 0, 2,
        density_fit=True, aux_basis_name="def2-svp-rifit",
    )
    assert df.e_aa + df.e_bb + df.e_ab == pytest.approx(
        df.e_correlation, rel=1e-12,
    )
    assert df.e_hf + df.e_correlation == pytest.approx(
        df.e_total, rel=1e-14,
    )


def test_df_ump2_correlation_is_negative():
    _, df = _vibeqc_ump2(
        [(8, [0.0, 0.0, 0.0]),
         (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
        "def2-svp", 0, 2,
        density_fit=True, aux_basis_name="def2-svp-rifit",
    )
    assert df.e_correlation < 0


# ---------------------------------------------------------------------
# Closed-shell consistency (UMP2 on singlet ↔ RMP2).
# ---------------------------------------------------------------------

def test_df_ump2_on_closed_shell_matches_df_rmp2():
    """For a closed-shell singlet (n_α = n_β), DF-UMP2 must reproduce
    DF-RMP2 correlation to numerical precision: αα and ββ channels
    are degenerate, αβ is identical, and antisymmetrised same-spin
    integrals reduce to the standard RMP2 spin sum."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")

    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, rhf_opts)
    rmp2_opts = _all_electron_mp2_options()
    rmp2_opts.density_fit = True
    rmp2_opts.aux_basis = "def2-svp-rifit"
    rmp2 = run_mp2(mol, basis, rhf, rmp2_opts)

    uhf_opts = UHFOptions()
    uhf_opts.conv_tol_energy = 1e-12
    uhf_opts.conv_tol_grad = 1e-8
    uhf = run_uhf(mol, basis, uhf_opts)
    ump2_opts = _all_electron_ump2_options()
    ump2_opts.density_fit = True
    ump2_opts.aux_basis = "def2-svp-rifit"
    ump2 = run_ump2(mol, basis, uhf, ump2_opts)

    delta = ump2.e_correlation - rmp2.e_correlation
    assert abs(delta) < 1e-9, (
        f"DF-UMP2 on H2O singlet: e_corr_uhf = {ump2.e_correlation}, "
        f"e_corr_rhf = {rmp2.e_correlation}, gap = {delta:+.3e}"
    )


# ---------------------------------------------------------------------
# Workflow tests.
# ---------------------------------------------------------------------

def test_df_ump2_requires_aux_basis():
    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = BasisSet(mol, "def2-svp")
    uhf = run_uhf(mol, basis, UHFOptions())
    opts = _all_electron_ump2_options()
    opts.density_fit = True
    opts.aux_basis = ""
    with pytest.raises(ValueError, match="aux_basis"):
        run_ump2(mol, basis, uhf, opts)


def test_df_ump2_ri_residual_diagnostic():
    """``UMP2Options.report_ri_residual`` surfaces the per-channel RI
    fit residual on ``UMP2Result.{e_aa,e_bb,e_ab}_ri_residual``.

    Open-shell counterpart of test_df_mp2.py::
    test_df_mp2_ri_residual_diagnostic. Uses the OH doublet so all
    three channels (αα, ββ, αβ) are populated. Pins the plumbing:
    off by default, fires only when both flags are set, and each
    residual equals (RI − canonical) computed independently on the
    same UHF reference.
    """
    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = BasisSet(mol, "def2-svp")
    uhf_opts = UHFOptions()
    uhf_opts.conv_tol_energy = 1e-12
    uhf_opts.conv_tol_grad = 1e-8
    uhf_opts.max_iter = 500
    uhf = run_uhf(mol, basis, uhf_opts)
    assert uhf.converged

    # Diagnostic off → flag false, residuals untouched (zero).
    opts = _all_electron_ump2_options()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-rifit"
    ump2_off = run_ump2(mol, basis, uhf, opts)
    assert ump2_off.ri_residual_reported is False
    assert ump2_off.e_aa_ri_residual == 0.0
    assert ump2_off.e_bb_ri_residual == 0.0
    assert ump2_off.e_ab_ri_residual == 0.0

    # Diagnostic on with density_fit=True → per-channel residuals set.
    opts.report_ri_residual = True
    ump2_on = run_ump2(mol, basis, uhf, opts)
    assert ump2_on.ri_residual_reported is True

    # Cross-check: each residual equals (RI − canonical) channel energy.
    opts_can = _all_electron_ump2_options()
    ump2_can = run_ump2(mol, basis, uhf, opts_can)
    assert ump2_on.e_aa_ri_residual == pytest.approx(
        ump2_on.e_aa - ump2_can.e_aa, abs=1e-12)
    assert ump2_on.e_bb_ri_residual == pytest.approx(
        ump2_on.e_bb - ump2_can.e_bb, abs=1e-12)
    assert ump2_on.e_ab_ri_residual == pytest.approx(
        ump2_on.e_ab - ump2_can.e_ab, abs=1e-12)

    # Diagnostic flag with density_fit=False → does NOT fire.
    opts.density_fit = False
    ump2_can_with_flag = run_ump2(mol, basis, uhf, opts)
    assert ump2_can_with_flag.ri_residual_reported is False
    assert ump2_can_with_flag.e_ab_ri_residual == 0.0
