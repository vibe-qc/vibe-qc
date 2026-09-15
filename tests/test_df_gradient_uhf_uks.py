"""DF analytic gradient — UHF and UKS via the user-facing
``compute_gradient_uhf`` / ``compute_gradient_uks`` API with
``GradientOptions.density_fit = True``.

Sibling of ``tests/test_df_gradient.py`` — same parity contract,
extended to open-shell (UHF) and open-shell DFT (UKS).

Pins:

  1. DF analytic gradient agrees with direct analytic gradient on the
     same converged UHF / UKS reference (only the 2e-gradient assembly
     differs).
  2. Empty ``aux_basis`` + ``density_fit=True`` raises ValueError.

Note: a previous version of this file (pre-Fix-C, v0.7.3 branch) had
two tests that exercised a Python-level "auto-route through DF on
f-shell bases" workaround (Fix D). Fix C lands the canonical 1/8 +
l-canonical reorder in the direct kernel itself
(``cpp/src/gradient.cpp::two_electron_gradient_contribution``), so
the auto-route is no longer needed and has been removed. The
auto-route tests were removed with it.
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
    UHFOptions,
    UKSOptions,
    compute_gradient_uhf,
    compute_gradient_uks,
    run_uhf,
    run_uks,
)
from .conftest import ANGSTROM_TO_BOHR


# ---------------------------------------------------------------------------
# Small open-shell reference: OH radical (doublet, 9 electrons).
# ---------------------------------------------------------------------------

_OH_RADICAL_ATOMS_BOHR = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
]


def _uhf_at(atoms_bohr, basis_name, *, density_fit=False, aux=""):
    mol = Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms_bohr],
                   multiplicity=2)
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-9
    opts.max_iter = 200
    opts.density_fit = density_fit
    opts.aux_basis = aux
    return mol, basis, run_uhf(mol, basis, opts)


def _uks_at(atoms_bohr, basis_name, functional, *, density_fit=False, aux=""):
    mol = Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms_bohr],
                   multiplicity=2)
    basis = BasisSet(mol, basis_name)
    opts = UKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-9
    # The OH· doublet is a ²Π state: the default-grid XC quadrature breaks
    # the π_x/π_y degeneracy anisotropically at the ~1.7e-7 commutator
    # scale, so ‖[F,DS]‖ plateaus there (the energy is converged to 1e-10
    # long before). The old per-spin DIIS wandered under a 1e-7 gate after
    # ~170 iterations by noise; the spin-coupled DIIS (2026-07) sits
    # stably on the floor. 5e-7 is far below anything that could move the
    # gradient-parity assert this test exists for.
    opts.conv_tol_grad = 5e-7
    opts.max_iter = 500
    opts.damping = 0.7
    opts.density_fit = density_fit
    opts.aux_basis = aux
    return mol, basis, run_uks(mol, basis, opts)


# ---------------------------------------------------------------------------
# DF UHF gradient vs direct UHF gradient.
# ---------------------------------------------------------------------------

def test_df_uhf_gradient_matches_direct_oh_radical():
    """DF-UHF analytic gradient agrees with direct UHF analytic gradient
    on the same converged direct-UHF reference (only the 2e-gradient
    assembly differs). The DF normalisation in
    ``uhf_df_two_electron_gradient`` (per-spin α_HF/2) is verified by
    this round-trip — if the factor were wrong the comparison would
    fail at the 1 mHa scale or worse."""
    mol, basis, uhf = _uhf_at(_OH_RADICAL_ATOMS_BOHR, "def2-svp")
    assert uhf.converged

    # Direct path.
    g_direct = compute_gradient_uhf(mol, basis, uhf)

    # DF path on the same converged reference.
    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"
    g_df = compute_gradient_uhf(mol, basis, uhf, opts)

    delta = np.abs(g_df - g_direct).max()
    # 1e-4 tolerance: the DF fitting error on OH/def2-svp is measured
    # at ~1.2e-5 Ha/bohr. Tightened from 5e-4 after the 3c-ERI
    # engine-state-leak fix (dc02c69) — 1e-4 still leaves headroom
    # over the fitting floor but now catches mHa-scale regressions.
    assert delta < 1e-4, (
        f"DF vs direct UHF gradient max abs diff = {delta:.3e} Ha/bohr; "
        f"per-spin K factor may be wrong (closed-shell limit gives "
        f"compute_k_gradient(C, alpha_hf) = 2 * compute_k_gradient(C, "
        f"alpha_hf/2), so the UHF formula compute_j_gradient(D_total) "
        f"+ compute_k_gradient(C_α, alpha_hf/2) "
        f"+ compute_k_gradient(C_β, alpha_hf/2) is correct)"
    )


# ---------------------------------------------------------------------------
# Pure-functional UKS DF (α_HF = 0) — K piece short-circuits.
# ---------------------------------------------------------------------------
#
# Hybrid UKS-DF (α_HF ∈ (0, 1)) parity isn't a separate test on this
# branch — the UHF test above covers α_HF = 1 (full per-spin K), and
# the UKS-PBE test below covers α_HF = 0 (J only). Hybrid DFT is a
# linear interpolation between those two: same code path, same
# scaling. ``uhf_df_two_electron_gradient`` short-circuits the K
# call entirely when ``alpha_hf == 0``, exercising both branches
# across these two tests. Adding a UKS-B3LYP-on-OH·/def2-svp parity
# test was tried but blocked on the underlying UKS SCF not converging
# without the EDIIS+DIIS / SOSCF accelerator stack on this branch
# (the OH· hybrid open-shell case is one of the documented "hard"
# convergence regimes; the certification matrix v0.13.5 picks it up
# explicitly).

def test_df_uks_pbe_gradient_matches_direct_oh_radical():
    """Pure GGA (α_HF = 0) — only the J piece of the DF gradient
    contributes, and the per-spin K short-circuit in
    ``uhf_df_two_electron_gradient`` is exercised."""
    mol, basis, uks = _uks_at(_OH_RADICAL_ATOMS_BOHR, "def2-svp", "PBE")
    assert uks.converged

    g_direct = compute_gradient_uks(mol, basis, uks)

    opts = GradientOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"
    g_df = compute_gradient_uks(mol, basis, uks, options=opts)

    delta = np.abs(g_df - g_direct).max()
    # 1e-4 tolerance — see test_df_uhf_gradient_matches_direct_oh_radical.
    # DF fitting error on UKS-PBE/OH/def2-svp measured ~4.7e-6 Ha/bohr.
    assert delta < 1e-4, (
        f"DF vs direct UKS-PBE gradient max abs diff = {delta:.3e} "
        f"Ha/bohr"
    )


# ---------------------------------------------------------------------------
# Empty aux_basis + density_fit=True raises.
# ---------------------------------------------------------------------------

def test_df_uhf_gradient_requires_aux_basis():
    mol, basis, uhf = _uhf_at(_OH_RADICAL_ATOMS_BOHR, "def2-svp")
    opts = GradientOptions()
    opts.density_fit = True
    # opts.aux_basis intentionally left empty.
    with pytest.raises((ValueError, RuntimeError),
                        match=r"aux_basis"):
        compute_gradient_uhf(mol, basis, uhf, opts)


def test_df_uks_gradient_requires_aux_basis():
    mol, basis, uks = _uks_at(_OH_RADICAL_ATOMS_BOHR, "def2-svp", "LDA")
    opts = GradientOptions()
    opts.density_fit = True
    with pytest.raises((ValueError, RuntimeError),
                        match=r"aux_basis"):
        compute_gradient_uks(mol, basis, uks, options=opts)
