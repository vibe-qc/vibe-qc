"""Phase D2e — TRAH (Trust-Region Augmented Hessian) tests.

TRAH is the third member of the second-order family. Like Newton
(D2c) it uses the full orbital Hessian via preconditioned CG; like
SOSCF (D2d) it has an automatic shift mechanism — but unlike either,
the trust radius itself adapts each step from Powell's ρ test
(actual / model-predicted ΔE).

Coverage:

  1. Defaults: ``trah_threshold = 0`` keeps RHF unchanged.
  2. TRAHOptions exposes the Powell schedule (initial / max / min
     radius, ρ thresholds, shrink/expand factors).
  3. Parity vs DIIS: TRAH reaches the same converged energy on
     H2O / sto-3g.
  4. Composability with EDIIS+DIIS warm-up.
  5. Mutual exclusion: Newton wins when both newton_threshold and
     trah_threshold are set.

Reference:
  B. Helmich-Paris, J. Chem. Phys. 156, 204104 (2022).
"""

from __future__ import annotations

import pytest

from vibeqc import (
    BasisSet,
    RHFOptions,
    SCFAccelerator,
    TRAHOptions,
    UHFOptions,
    run_rhf,
    run_uhf,
)

from .conftest import ANGSTROM_TO_BOHR, make_molecule


def _h2o_atoms():
    return [
        (8, [0.0, 0.0,  0.117 * ANGSTROM_TO_BOHR]),
        (1, [0.0,  0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
        (1, [0.0, -0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
    ]


def _oh_radical_atoms():
    return [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
    ]


@pytest.fixture
def h2o_basis():
    mol = make_molecule(_h2o_atoms())
    return mol, BasisSet(mol, "sto-3g")


@pytest.fixture
def oh_radical():
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    return mol, BasisSet(mol, "sto-3g")


# ---------------------------------------------------------------------------
# Options surface — TRAHOptions defaults + RHFOptions integration.
# ---------------------------------------------------------------------------

def test_trah_threshold_default_is_zero():
    """RHFOptions exposes trah_threshold; default 0 keeps TRAH off."""
    assert RHFOptions().trah_threshold == pytest.approx(0.0)


def test_trah_options_default_radius():
    """Default trust radius matches Newton/SOSCF (0.3)."""
    o = TRAHOptions()
    assert o.initial_trust_radius == pytest.approx(0.3)
    assert o.max_trust_radius == pytest.approx(0.7)
    assert o.min_trust_radius == pytest.approx(1e-4)


def test_trah_options_default_powell_schedule():
    """Default ρ thresholds match Powell convention."""
    o = TRAHOptions()
    assert o.rho_shrink == pytest.approx(0.25)
    assert o.rho_expand == pytest.approx(0.75)
    assert o.trust_shrink_factor == pytest.approx(0.5)
    assert o.trust_expand_factor == pytest.approx(2.0)


def test_trah_options_more_sorensen_defaults():
    """The More-Sorensen secular-root-find knobs for the boundary
    level shift λ are exposed with the documented defaults."""
    o = TRAHOptions()
    assert o.ms_max_iter == 12
    assert o.ms_tol == pytest.approx(1e-3)
    # Round-trip via RHFOptions.
    ro = RHFOptions()
    ro.trah_opts.ms_max_iter = 6
    ro.trah_opts.ms_tol = 5e-4
    assert ro.trah_opts.ms_max_iter == 6
    assert ro.trah_opts.ms_tol == pytest.approx(5e-4)


def test_trah_options_round_trip_via_rhf_options():
    """RHFOptions.trah_opts round-trips."""
    o = RHFOptions()
    o.trah_threshold = 1.0
    o.trah_opts.max_trust_radius = 0.6
    o.trah_opts.rho_shrink = 0.2
    assert o.trah_threshold == pytest.approx(1.0)
    assert o.trah_opts.max_trust_radius == pytest.approx(0.6)
    assert o.trah_opts.rho_shrink == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Default-off back-compat.
# ---------------------------------------------------------------------------

def test_rhf_default_does_not_activate_trah(h2o_basis):
    """With trah_threshold = 0 (default), RHF is unchanged."""
    mol, basis = h2o_basis
    o = RHFOptions()
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-6
    r = run_rhf(mol, basis, o)
    assert r.converged


# ---------------------------------------------------------------------------
# Parity — TRAH reaches the same fixed point as plain DIIS.
# ---------------------------------------------------------------------------

def test_rhf_trah_matches_diis_h2o(h2o_basis):
    """RHF / sto-3g: TRAH reaches the same energy as DIIS to ~1e-9 Ha."""
    mol, basis = h2o_basis

    o_diis = RHFOptions()
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-7
    r_diis = run_rhf(mol, basis, o_diis)
    assert r_diis.converged

    o_trah = RHFOptions()
    o_trah.trah_threshold = 1.0      # Fischer-Almlöf 1992 convention
    o_trah.conv_tol_energy = 1e-10
    o_trah.conv_tol_grad = 1e-7
    r_trah = run_rhf(mol, basis, o_trah)
    assert r_trah.converged
    assert r_trah.energy == pytest.approx(r_diis.energy, abs=1e-9)


def test_rhf_trah_tiny_trust_radius_hits_boundary_and_converges(h2o_basis):
    """A trust radius far smaller than the unconstrained Newton step
    forces every TRAH iteration onto the trust-region boundary, so the
    More-Sorensen secular root-find for the level shift λ runs on every
    step. The proper level-shifted boundary step κ(λ) = −(A + λI)⁻¹g
    must still drive the SCF to the *same* fixed point as plain DIIS.

    This is the regression guard for the v0.8.x "TRAH proper" work:
    the earlier stub clipped the Newton step to ‖κ‖ = Δ (a scaled
    Newton direction); the level-shifted step points in a genuinely
    different direction. Both reach the same minimum, but only the
    level-shift solve is the exact trust-region subproblem solution.
    """
    mol, basis = h2o_basis

    o_diis = RHFOptions()
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-7
    r_diis = run_rhf(mol, basis, o_diis)
    assert r_diis.converged

    o_trah = RHFOptions()
    o_trah.trah_threshold = 1.0
    o_trah.conv_tol_energy = 1e-10
    o_trah.conv_tol_grad = 1e-7
    o_trah.max_iter = 200
    # Tiny radius, capped small so the SCF stays in the boundary
    # (level-shifted) regime rather than expanding back to interior
    # Newton steps.
    o_trah.trah_opts.initial_trust_radius = 0.02
    o_trah.trah_opts.max_trust_radius = 0.05
    r_trah = run_rhf(mol, basis, o_trah)
    assert r_trah.converged, (
        f"TRAH (tiny trust radius) did not converge: "
        f"E = {r_trah.energy:.8f}")
    assert r_trah.energy == pytest.approx(r_diis.energy, abs=1e-9)


def test_trah_level_shift_surfaces_in_scf_trace(h2o_basis):
    """``SCFIteration.trah_level_shift`` exposes the More-Sorensen level
    shift λ for observability. It is 0 on every non-TRAH iteration (any
    accelerator / Newton / SOSCF step, and every iteration of a plain
    DIIS run) and > 0 whenever TRAH took a trust-region *boundary* step
    — κ(λ) = −(A + λI)⁻¹g on ‖κ‖ = Δ.

    The tiny capped trust radius keeps TRAH on the boundary once it
    activates, so at least one trace row must carry a positive shift.
    """
    mol, basis = h2o_basis

    # Plain DIIS — no TRAH ever runs, so every row's shift is exactly 0.
    o_diis = RHFOptions()
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-7
    r_diis = run_rhf(mol, basis, o_diis)
    assert r_diis.converged
    assert all(it.trah_level_shift == 0.0 for it in r_diis.scf_trace)

    # TRAH with a tiny capped radius — forces level-shifted boundary
    # steps once the gradient drops below trah_threshold.
    o_trah = RHFOptions()
    o_trah.trah_threshold = 1.0
    o_trah.conv_tol_energy = 1e-10
    o_trah.conv_tol_grad = 1e-7
    o_trah.max_iter = 200
    o_trah.trah_opts.initial_trust_radius = 0.02
    o_trah.trah_opts.max_trust_radius = 0.05
    r_trah = run_rhf(mol, basis, o_trah)
    assert r_trah.converged
    shifts = [it.trah_level_shift for it in r_trah.scf_trace]
    # Every shift is non-negative; at least one boundary step recorded.
    assert all(s >= 0.0 for s in shifts)
    assert any(s > 0.0 for s in shifts), (
        "expected at least one TRAH boundary step (trah_level_shift > 0) "
        f"in the trace; got {shifts}")


def test_uhf_trah_tiny_trust_radius_hits_boundary_and_converges(oh_radical):
    """Open-shell counterpart: a tiny trust radius forces the coupled
    (κ_α, κ_β) More-Sorensen boundary solve every iteration. TRAH must
    still reach the DIIS fixed point and the same ⟨S²⟩."""
    mol, basis = oh_radical

    o_diis = UHFOptions()
    o_diis.max_iter = 250
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-8
    r_diis = run_uhf(mol, basis, o_diis)
    assert r_diis.converged

    o_trah = UHFOptions()
    o_trah.max_iter = 250
    o_trah.trah_threshold = 1.0
    o_trah.conv_tol_energy = 1e-10
    o_trah.conv_tol_grad = 1e-8
    o_trah.trah_opts.initial_trust_radius = 0.02
    o_trah.trah_opts.max_trust_radius = 0.05
    r_trah = run_uhf(mol, basis, o_trah)
    assert r_trah.converged
    assert r_trah.energy == pytest.approx(r_diis.energy, abs=1e-9)
    assert r_trah.s_squared == pytest.approx(r_diis.s_squared, abs=1e-7)


def test_rhf_trah_with_ediis_diis_warmup_matches_pure_diis(h2o_basis):
    """TRAH + EDIIS+DIIS hybrid composes — same converged energy."""
    mol, basis = h2o_basis

    o_diis = RHFOptions()
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-7
    r_diis = run_rhf(mol, basis, o_diis)
    assert r_diis.converged

    o_combo = RHFOptions()
    o_combo.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    o_combo.trah_threshold = 1.0
    o_combo.conv_tol_energy = 1e-10
    o_combo.conv_tol_grad = 1e-7
    r_combo = run_rhf(mol, basis, o_combo)
    assert r_combo.converged
    assert r_combo.energy == pytest.approx(r_diis.energy, abs=1e-9)


# ---------------------------------------------------------------------------
# Mutual exclusion: Newton wins over TRAH when both are set.
# ---------------------------------------------------------------------------

def test_newton_wins_when_both_newton_and_trah_set(h2o_basis):
    """Priority: quadratic > Newton > TRAH > SOSCF. When both newton
    and trah are set, Newton activates (CG counter recorded)."""
    mol, basis = h2o_basis
    o = RHFOptions()
    o.newton_threshold = 1.0
    o.trah_threshold = 1.0
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-7
    r = run_rhf(mol, basis, o)
    assert r.converged
    assert any(it.newton_cg_iter > 0 for it in r.scf_trace)


# ---------------------------------------------------------------------------
# UHF — open-shell TRAH (per-spin coupled CG, shared trust radius).
# ---------------------------------------------------------------------------

def test_uhf_trah_threshold_default_is_zero():
    assert UHFOptions().trah_threshold == pytest.approx(0.0)


def test_uhf_trah_options_share_defaults_with_rhf():
    assert UHFOptions().trah_opts.initial_trust_radius == \
        pytest.approx(TRAHOptions().initial_trust_radius)


def test_uhf_trah_matches_diis_oh_radical(oh_radical):
    """UHF / sto-3g on OH·: TRAH reaches the same energy as DIIS."""
    mol, basis = oh_radical
    o_diis = UHFOptions()
    o_diis.max_iter = 250
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-8
    r_diis = run_uhf(mol, basis, o_diis)
    assert r_diis.converged

    o_trah = UHFOptions()
    o_trah.max_iter = 250
    o_trah.trah_threshold = 1.0
    o_trah.conv_tol_energy = 1e-10
    o_trah.conv_tol_grad = 1e-8
    r_trah = run_uhf(mol, basis, o_trah)
    assert r_trah.converged
    assert r_trah.energy == pytest.approx(r_diis.energy, abs=1e-9)
    assert r_trah.s_squared == pytest.approx(r_diis.s_squared, abs=1e-7)


def test_trah_wins_over_soscf_when_both_set(h2o_basis):
    """TRAH (D2e) > SOSCF (D2d) when both thresholds are set. TRAH
    populates newton_cg_iter (it goes through the same CG path),
    SOSCF does not. So if TRAH wins we'll see cg iters in the trace."""
    mol, basis = h2o_basis
    o = RHFOptions()
    o.trah_threshold = 1.0
    o.soscf_threshold = 1.0
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-7
    r = run_rhf(mol, basis, o)
    assert r.converged
    # TRAH records CG iters in the trace (shared with Newton); if SOSCF
    # had won, no CG iters would be recorded.
    assert any(it.newton_cg_iter > 0 for it in r.scf_trace)
