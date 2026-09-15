"""Phase D2c-KS-UHF / D2d-KS-UHF / D2e-KS-UHF — second-order UKS parity.

Counterpart of ``tests/test_rks_second_order.py`` for the open-shell
side. UKS Newton / TRAH use the polarised XC kernel
(``make_polarised_xc_kernel_builder``, which dispatches LDA → polarised-
LDA builder and — since Phase 17e — GGA / hybrid-GGA → polarised-GGA
builder). SOSCF doesn't need a kernel (per-spin F already carries
V_xc,σ). Meta-GGA Newton / TRAH still raise: the τ-dependent polarised
fxc is a later phase.

Reference system: the OH· radical, a doublet.

Coverage:
  1. UKS Newton + TRAH — LDA, PBE (pure GGA), B3LYP (hybrid GGA): both
     second-order schemes converge to the same minimum, and it is the
     physical broken-symmetry doublet (⟨S²⟩ ≈ 0.75). The polarised GGA
     f_xc kernel (Phase 17e) is what makes the GGA / hybrid-GGA
     orbital-Hessian matvec correct.
  2. Where plain DIIS also reaches that minimum (LDA, PBE), Newton
     agrees on the energy. B3LYP DIIS collapses to the ⟨S²⟩ = 0
     closed-shell solution on this system — second-order SCF recovers
     the doublet, which is exactly why it matters.
  3. UKS-LDA SOSCF: matches DIIS at the asymptotic accuracy (1e-4 Ha).
  4. UKS meta-GGA Newton: raises with a clear roadmap pointer.
  5. Defaults: all three thresholds default to 0 (back-compat).

Note on tolerances: the test grid is deliberately coarse for speed.
Coarse-grid GGA XC quadrature floors the SCF gradient near ~1e-6 on
the OH· radical, so ``conv_tol_grad`` is 1e-6 here (finer grids reach
1e-7 but cost far too much for a unit test). Different converged
methods (DIIS / Newton / TRAH) stop at slightly different depths
within that gradient tolerance — Newton overshoots quadratically,
TRAH stops at its trust-region criterion — so along the radical's
flat direction the converged energies can disagree by ~1e-7. All
energy-parity asserts therefore use ``abs=1e-6`` ("the energy agrees
to the SCF gradient convergence tolerance"); a genuinely wrong
minimum (e.g. the ⟨S²⟩ = 0 closed-shell collapse) would differ by
tens of mHa, not 1e-7.
"""

from __future__ import annotations

import pytest

import vibeqc as vq
from vibeqc import Atom, BasisSet, Molecule, UKSOptions, run_uks


# ---------------------------------------------------------------------------
# Fixtures — OH· radical, doublet.
# ---------------------------------------------------------------------------

def _oh_radical():
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.0, 0.97 * 1.8897259886])],
        multiplicity=2,
    )


@pytest.fixture
def oh_radical():
    mol = _oh_radical()
    return mol, BasisSet(mol, "sto-3g")


def _make_opts(functional):
    o = UKSOptions()
    o.functional = functional
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-6     # coarse-grid GGA grad-noise floor — see
                                # the module docstring.
    o.max_iter = 250
    o.grid.n_radial = 40
    o.grid.n_theta = 14
    o.grid.n_phi = 28
    return o


# ---------------------------------------------------------------------------
# Defaults — no surprise activation.
# ---------------------------------------------------------------------------

def test_uks_options_second_order_thresholds_default_zero():
    o = UKSOptions()
    assert o.newton_threshold == pytest.approx(0.0)
    assert o.soscf_threshold == pytest.approx(0.0)
    assert o.trah_threshold == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Newton + TRAH converge to the same physical minimum, across the
# LDA / GGA / hybrid-GGA functional classes.
#
#   LDA   — polarised-LDA kernel (v2rho2 only).
#   PBE   — polarised-GGA kernel (Phase 17e: v2rhosigma / v2sigma2 + the
#           σ_αβ cross term).
#   B3LYP — hybrid GGA, α_HF = 0.20; HF exchange rides the JKBuilder
#           ``alpha_hf`` scale, the GGA piece rides the polarised-GGA
#           fxc kernel.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("functional", ["LDA", "PBE", "B3LYP"])
def test_uks_newton_and_trah_converge_to_same_minimum(oh_radical, functional):
    mol, basis = oh_radical

    o_n = _make_opts(functional)
    o_n.newton_threshold = 1.0
    r_n = run_uks(mol, basis, o_n)

    o_t = _make_opts(functional)
    o_t.trah_threshold = 1.0
    r_t = run_uks(mol, basis, o_t)

    assert r_n.converged, f"{functional}: UKS Newton did not converge"
    assert r_t.converged, f"{functional}: UKS TRAH did not converge"
    # Two independent second-order schemes — same minimum (to the SCF
    # gradient convergence tolerance; see the module docstring).
    assert r_n.energy == pytest.approx(r_t.energy, abs=1e-6)
    assert r_n.s_squared == pytest.approx(r_t.s_squared, abs=1e-4)
    # ...and it is the physical broken-symmetry doublet, not the
    # ⟨S²⟩ = 0 closed-shell collapse.
    assert 0.74 < r_n.s_squared < 0.80


@pytest.mark.parametrize("functional", ["LDA", "PBE"])
def test_uks_newton_matches_diis(oh_radical, functional):
    """From a broken-symmetry initial guess, first-order DIIS reaches the
    open-shell minimum (LDA, PBE) and UKS Newton must agree on the energy
    and ⟨S²⟩. The ``atomic_spins`` seed matters: from the symmetric AUTO
    guess, LDA DIIS stalls on the ⟨S²⟩ = 0 closed-shell saddle (no
    first-order step leaves it, so it never converges), whereas Newton's
    second-order step descends off the saddle unaided. Seeding a
    broken-symmetry guess is the standard way to converge first-order SCF
    to such a state. B3LYP is covered by the Newton/TRAH-only test above
    instead."""
    mol, basis = oh_radical

    o_diis = _make_opts(functional)
    o_diis.atomic_spins = [1, 1]   # break spin symmetry so DIIS leaves the
                                    # closed-shell saddle (see docstring).
    if functional == "PBE":
        # The coupled-spin DIIS (2026-07) converges smoothly to the
        # coarse-grid GGA quadrature-noise floor, which on this system
        # sits at ~1.4e-6 — just above the 1e-6 default of _make_opts.
        # (The old per-spin histories wandered enough to dip below 1e-6
        # by luck.) 2e-6 is still an order of magnitude below anything
        # that could hide a wrong minimum; the energy-parity assert
        # against Newton below is the real check.
        o_diis.conv_tol_grad = 2e-6
    r_diis = run_uks(mol, basis, o_diis)
    assert r_diis.converged
    assert 0.74 < r_diis.s_squared < 0.80   # DIIS found the doublet

    o_newton = _make_opts(functional)
    o_newton.newton_threshold = 1.0
    r_newton = run_uks(mol, basis, o_newton)
    assert r_newton.converged
    assert r_newton.energy == pytest.approx(r_diis.energy, abs=1e-6)
    assert r_newton.s_squared == pytest.approx(r_diis.s_squared, abs=1e-4)


def test_uks_lda_soscf_matches_diis(oh_radical):
    """SOSCF on UKS-LDA — same documented Neese 2000 asymptotic
    plateau as the closed-shell RKS-LDA case. Assert energy parity
    to 1e-4 Ha at conv_tol_grad = 1e-4."""
    mol, basis = oh_radical

    o_diis = _make_opts("LDA")
    o_diis.atomic_spins = [1, 1]   # break spin symmetry so DIIS reaches
                                    # the doublet (see the
                                    # test_uks_newton_matches_diis docstring).
    r_diis = run_uks(mol, basis, o_diis)
    assert r_diis.converged

    o_soscf = _make_opts("LDA")
    o_soscf.soscf_threshold = 0.01     # tighter than the HF 1.0 — see
                                        # rks_second_order test comment
    o_soscf.conv_tol_grad = 1e-4
    r_soscf = run_uks(mol, basis, o_soscf)
    assert r_soscf.converged
    assert r_soscf.energy == pytest.approx(r_diis.energy, abs=1e-4)


# ---------------------------------------------------------------------------
# Meta-GGA Newton — raises with a clear roadmap pointer. The polarised
# τ-dependent fxc kernel is not yet plumbed; the factory must raise
# rather than silently returning a GGA kernel (which would omit the τ
# second-derivative block and corrupt the Newton step).
# ---------------------------------------------------------------------------

def test_uks_mgga_newton_raises_pointer(oh_radical):
    mol, basis = oh_radical
    o = _make_opts("tpssh")
    o.newton_threshold = 1.0
    with pytest.raises(RuntimeError, match="meta-GGA"):
        run_uks(mol, basis, o)
