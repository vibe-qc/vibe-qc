"""End-to-end DF analytic gradient — RHF and RKS via the user-facing
``compute_gradient`` / ``compute_gradient_rks`` API with
``GradientOptions.density_fit = True``.

Pins:

  1. Direct vs DF analytic gradient — agreement up to JKfit fit error
     (typically sub-mHa/bohr per atom on neutral organics).
  2. DF analytic vs DF finite difference (each FD step re-runs DF-RHF
     / DF-RKS at the displaced geometry). Truncation floor at
     h = 1e-4 is ~1e-7 Ha/bohr; tolerance 1e-5 leaves slack for SCF
     convergence variability.
  3. Empty aux_basis raises ValueError (same contract as the SCF
     drivers).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    GradientOptions,
    GridOptions,
    Molecule,
    RHFOptions,
    RKSOptions,
    SCFAccelerator,
    compute_gradient,
    compute_gradient_rks,
    run_rhf,
    run_rks,
)

from .conftest import GEOMETRIES


def _rhf_at(positions, atom_Zs, basis_name, *, density_fit=False, aux=""):
    mol = Molecule([Atom(int(Z), list(p)) for Z, p in zip(atom_Zs, positions)])
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    # EDIIS+DIIS hybrid: plain DIIS plateaus on the larger f-shell
    # cases (H2CO/def2-tzvp) at this tight tolerance.
    opts.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    opts.max_iter = 200
    opts.density_fit = density_fit
    opts.aux_basis = aux
    return mol, basis, run_rhf(mol, basis, opts)


def _rks_at(positions, atom_Zs, basis_name, functional, *,
            density_fit=False, aux=""):
    mol = Molecule([Atom(int(Z), list(p)) for Z, p in zip(atom_Zs, positions)])
    basis = BasisSet(mol, basis_name)
    opts = RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 200
    opts.density_fit = density_fit
    opts.aux_basis = aux
    return mol, basis, run_rks(mol, basis, opts)


def _fd_total_energy(energy_at, positions, h=1e-4):
    n_atoms = len(positions)
    grad = np.zeros((n_atoms, 3))
    for A in range(n_atoms):
        for c in range(3):
            pp = [list(p) for p in positions]; pp[A][c] += h
            pm = [list(p) for p in positions]; pm[A][c] -= h
            grad[A, c] = (energy_at(pp) - energy_at(pm)) / (2 * h)
    return grad


# ---------------------------------------------------------------------
# RHF DF gradient.
# ---------------------------------------------------------------------

DF_RHF_GRADIENT_CASES = [
    ("H2",  "def2-svp", "def2-svp-jk"),
    ("H2O", "def2-svp", "def2-svp-jk"),
    ("CH4", "def2-svp", "def2-svp-jk"),
]


@pytest.mark.parametrize(
    "mol_key,orb,aux", DF_RHF_GRADIENT_CASES,
    ids=[f"{m}-{o}" for m, o, _ in DF_RHF_GRADIENT_CASES],
)
def test_df_rhf_gradient_close_to_direct(mol_key, orb, aux):
    """DF-RHF analytic gradient agrees with direct RHF analytic gradient
    on the same converged direct-RHF reference (only the 2e-gradient
    assembly differs)."""
    atoms = GEOMETRIES[mol_key]
    Zs = [Z for Z, _ in atoms]
    pos = [list(xyz) for _, xyz in atoms]
    mol, basis, rhf = _rhf_at(pos, Zs, orb)
    assert rhf.converged

    g_direct = compute_gradient(mol, basis, rhf)
    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = aux
    g_df = compute_gradient(mol, basis, rhf, opts)

    delta = np.abs(g_df - g_direct).max()
    # 1e-4 tolerance: the DF fitting error on these def2-svp cases is
    # measured at <=3.6e-5 Ha/bohr (worst: H2/def2-svp). 1e-4 leaves
    # ~3x headroom over the fitting floor while still catching any
    # mHa-scale kernel regression (the 3c-ERI engine-state-leak bug
    # fixed in dc02c69 was ~2 mHa).
    assert delta < 1e-4, (
        f"{mol_key}/{orb}/{aux}: DF vs direct RHF gradient max abs "
        f"diff = {delta:.3e} Ha/bohr"
    )


@pytest.mark.parametrize(
    "mol_key,orb,aux", DF_RHF_GRADIENT_CASES,
    ids=[f"{m}-{o}" for m, o, _ in DF_RHF_GRADIENT_CASES],
)
def test_df_rhf_gradient_matches_fd(mol_key, orb, aux):
    """DF-RHF analytic gradient agrees with FD of DF-RHF total energy.
    Each FD step re-runs DF-RHF at the displaced geometry."""
    atoms = GEOMETRIES[mol_key]
    Zs = [Z for Z, _ in atoms]
    pos = [list(xyz) for _, xyz in atoms]
    mol, basis, rhf = _rhf_at(pos, Zs, orb, density_fit=True, aux=aux)
    assert rhf.converged

    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = aux
    g_an = compute_gradient(mol, basis, rhf, opts)

    def df_energy_at(positions):
        _, _, hf = _rhf_at(positions, Zs, orb, density_fit=True, aux=aux)
        assert hf.converged
        return hf.energy

    g_fd = _fd_total_energy(df_energy_at, pos)
    delta = np.abs(g_an - g_fd).max()
    assert delta < 1e-5, (
        f"{mol_key}/{orb}/{aux}: DF-RHF analytic-FD max diff = "
        f"{delta:.3e} Ha/bohr"
    )


def test_df_rhf_gradient_requires_aux_basis():
    """density_fit=True with empty aux_basis raises."""
    atoms = GEOMETRIES["H2"]
    Zs = [Z for Z, _ in atoms]
    pos = [list(xyz) for _, xyz in atoms]
    mol, basis, rhf = _rhf_at(pos, Zs, "def2-svp")
    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = ""
    with pytest.raises(ValueError, match="aux_basis"):
        compute_gradient(mol, basis, rhf, opts)


# ---------------------------------------------------------------------
# RKS DF gradient.
# ---------------------------------------------------------------------

DF_RKS_GRADIENT_CASES = [
    ("LDA", "H2O", "def2-svp", "def2-svp-jk"),
    ("PBE", "H2O", "def2-svp", "def2-svp-jk"),
    ("B3LYP", "H2O", "def2-svp", "def2-svp-jk"),
    ("LDA", "CH4", "def2-svp", "def2-svp-jk"),
]


@pytest.mark.parametrize(
    "func,mol_key,orb,aux", DF_RKS_GRADIENT_CASES,
    ids=[f"{f}-{m}-{o}" for f, m, o, _ in DF_RKS_GRADIENT_CASES],
)
def test_df_rks_gradient_close_to_direct(func, mol_key, orb, aux):
    """DF-RKS analytic gradient agrees with direct RKS gradient. The
    XC-Pulay piece is shared (grid-based, independent of DF); only the
    J + α_HF·K assembly differs. Pure DFT (LDA, PBE) exercises only
    the J path; hybrid (B3LYP α=0.2) exercises both."""
    atoms = GEOMETRIES[mol_key]
    Zs = [Z for Z, _ in atoms]
    pos = [list(xyz) for _, xyz in atoms]
    mol, basis, rks = _rks_at(pos, Zs, orb, func)
    assert rks.converged

    g_direct = compute_gradient_rks(mol, basis, rks, GridOptions())
    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = aux
    g_df = compute_gradient_rks(mol, basis, rks, GridOptions(), opts)

    delta = np.abs(g_df - g_direct).max()
    # 1e-4 tolerance — see test_df_rhf_gradient_close_to_direct. The
    # XC-Pulay piece is shared between DF and direct, so the residual
    # here is again just the J + alpha_HF*K DF fitting error
    # (measured <=1.3e-5 Ha/bohr on these cases).
    assert delta < 1e-4, (
        f"{func}/{mol_key}/{orb}/{aux}: DF vs direct RKS gradient max "
        f"abs diff = {delta:.3e} Ha/bohr"
    )


def test_df_rks_gradient_requires_aux_basis():
    atoms = GEOMETRIES["H2O"]
    Zs = [Z for Z, _ in atoms]
    pos = [list(xyz) for _, xyz in atoms]
    mol, basis, rks = _rks_at(pos, Zs, "def2-svp", "PBE")
    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = ""
    with pytest.raises(ValueError, match="aux_basis"):
        compute_gradient_rks(mol, basis, rks, GridOptions(), opts)


# ---------------------------------------------------------------------
# F-shell DF gradient regression.
# ---------------------------------------------------------------------
#
# H2CO + def2-tzvp + def2-tzvp-jk caught the first iteration of a libint
# engine state leak (~2.8 mHa on heavy-atom-z) and was fixed by copying
# a fresh engine per ``(sP, sM)`` group (commit dc02c69).
#
# HCOOH (formic acid) + def2-tzvp + def2-tzvp-jk catches the *second*
# iteration: two adjacent same-l heavy atoms (the carboxyl O=C-O-H
# oxygens) trigger an additional ~5–15 mHa per-atom-component
# misattribution from the same engine-state leak, surviving the
# per-(sP, sM) freshness fix because the inner sN loop still reuses
# one engine across different orbital ket-2 shells. The proper fix
# (this commit) copies a fresh engine per ``(sP, sM, sN)`` call;
# residual drops to ~1e-7 Ha/bohr (FD truncation floor).

def test_df_rhf_gradient_h2co_def2_tzvp_matches_direct():
    """DF-RHF analytic gradient on H2CO/def2-tzvp/def2-tzvp-jk agrees
    with the direct analytic gradient to ~1e-5 Ha/bohr.

    Pre-fix this disagreed by ~2 mHa on heavy-atom-z components due to
    a libint engine state leak across compute() calls for BraKet::xs_xx
    derivatives. The 3c gradient kernel now uses a fresh engine copy
    per (sP, sM) group to avoid the leak.
    """
    ANG = 1.8897261339213
    coords_ang = [
        (0.0, 0.0, 0.000),
        (0.0, 0.0, 1.205),
        (0.0, 0.943, -0.587),
        (0.0, -0.943, -0.587),
    ]
    Zs = [6, 8, 1, 1]
    pos = [list(np.array(c) * ANG) for c in coords_ang]
    mol, basis, rhf = _rhf_at(pos, Zs, "def2-tzvp")
    assert rhf.converged

    g_direct = compute_gradient(mol, basis, rhf)
    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-tzvp-jk"
    g_df = compute_gradient(mol, basis, rhf, opts)

    delta = np.abs(g_df - g_direct).max()
    # Tolerance set to 5e-5 — the DF fitting error on def2-tzvp/def2-tzvp-jk
    # is typically ~1e-5 Ha/bohr (intrinsic to the DF approximation, not
    # the kernel). Pre-fix this disagreed by ~2 mHa.
    assert delta < 5e-5, (
        f"H2CO/def2-tzvp/def2-tzvp-jk: DF vs direct RHF gradient max "
        f"abs diff = {delta:.3e} Ha/bohr (pre-fix was ~2 mHa due to "
        f"libint xs_xx engine state leak)"
    )


def test_df_rhf_gradient_hcooh_def2_tzvp_matches_direct():
    """DF-RHF analytic gradient on HCOOH/def2-tzvp/def2-tzvp-jk agrees
    with the direct analytic gradient to ~1e-4 Ha/bohr.

    Formic acid has two adjacent same-l heavy atoms — the carboxyl
    O=C-O-H oxygens. Pre-fix (per-(sP, sM) engine freshness only) the
    3c-ERI derivative kernel mis-attributed ~5–13 mHa per component
    between the two oxygens: the inner sN loop reused one libint
    engine across orbital ket-2 shells, leaking derivative-buffer
    state. The factor-2 unique-pair doubling masked the leak as a
    correct contracted scalar but a wrong per-atom split. The fix
    copies a fresh engine per (sP, sM, sN) call
    (cpp/src/df.cpp::compute_3c_eri_gradient_weighted). Post-fix the
    3c-kernel residual vs finite-difference is ~1e-7 Ha/bohr.

    This is the smallest reproducer of the same-l-different-atom
    variant; the larger glycine/def2-tzvp case (~115 mHa on the DF
    total gradient) is the field report this regression guards.
    """
    ANG = 1.8897261339213
    coords_ang = [
        (-1.193, 0.183, 0.000),   # H (bonded to C)
        (-0.040, 0.519, 0.000),   # C
        (0.913, -0.179, 0.000),   # O (carbonyl)
        (0.219, 1.785, 0.000),    # O (hydroxyl)
        (1.146, 1.967, 0.000),    # H (hydroxyl)
    ]
    Zs = [1, 6, 8, 8, 1]
    pos = [list(np.array(c) * ANG) for c in coords_ang]
    mol, basis, rhf = _rhf_at(pos, Zs, "def2-tzvp")
    assert rhf.converged

    g_direct = compute_gradient(mol, basis, rhf)
    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-tzvp-jk"
    g_df = compute_gradient(mol, basis, rhf, opts)

    delta = np.abs(g_df - g_direct).max()
    # Tolerance 2e-4 — the DF fitting error on def2-tzvp/def2-tzvp-jk for
    # a polar carboxylic-acid group runs a bit higher than the ~1e-5 of
    # the small non-polar cases. Pre-fix the carboxyl oxygens disagreed
    # by ~8–13 mHa; this tolerance still catches any mHa-scale regression
    # of the engine-state-leak bug.
    assert delta < 2e-4, (
        f"HCOOH/def2-tzvp/def2-tzvp-jk: DF vs direct RHF gradient max "
        f"abs diff = {delta:.3e} Ha/bohr (pre-fix was ~8-13 mHa on the "
        f"carboxyl oxygens due to a libint xs_xx engine state leak in "
        f"the inner sN loop of compute_3c_eri_gradient_weighted)"
    )
