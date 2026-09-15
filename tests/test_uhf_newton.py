"""Phase D2c open-shell — UHF Newton (coupled α+β orbital-Hessian
Newton-CG) tests.

Pins the contract:

  1. With ``newton_threshold = 0`` (default), UHF behaves bit-for-bit
     as before — no Newton activation, no SCF trace change.

  2. With ``newton_threshold = 1.0`` (Fischer-Almlöf 1992 production
     setting), UHF converges to the *same* energy *and* the same
     <S^2> as plain DIIS on standard reference systems. The
     accelerator changes the path through the SCF manifold, not the
     converged fixed point.

  3. UHF Newton + EDIIS+DIIS hybrid composes: warm-up handles the
     iterations far from convergence, Newton closes out the
     asymptotic regime with quadratic convergence.

  4. UHFOptions has the new ``newton_threshold`` / ``newton_opts``
     fields with documented defaults.

The open-shell Newton step couples α and β rotations through the
shared J in the orbital Hessian (single CG on the stacked
(κ_α, κ_β) trial vector). See ``cpp/src/newton.cpp::uhf_newton_step``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import (
    BasisSet,
    SCFAccelerator,
    NewtonOptions,
    UHFOptions,
    run_uhf,
)
from .conftest import ANGSTROM_TO_BOHR, make_molecule


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def _oh_radical_atoms():
    """Hydroxyl radical (doublet, 9 electrons)."""
    return [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
    ]


def _h_atom_atoms():
    """Hydrogen atom (doublet, 1 electron) — trivial sanity case."""
    return [(1, [0.0, 0.0, 0.0])]


@pytest.fixture
def oh_radical():
    mol = make_molecule(_oh_radical_atoms(), multiplicity=2)
    return mol, BasisSet(mol, "sto-3g")


# ---------------------------------------------------------------------------
# Defaults — back-compat: Newton off, threshold 0.
# ---------------------------------------------------------------------------

def test_uhf_newton_threshold_default_is_zero():
    opts = UHFOptions()
    assert opts.newton_threshold == pytest.approx(0.0)


def test_uhf_newton_options_share_defaults_with_rhf():
    so = UHFOptions().newton_opts
    # Same NewtonOptions type used by RHF; defaults are shared.
    assert so.cg_max_iter == 50
    assert so.cg_tol == pytest.approx(1e-4)
    assert so.trust_radius == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Newton activation — back-compat with default off.
# ---------------------------------------------------------------------------

def test_uhf_default_does_not_activate_newton(oh_radical):
    """With ``newton_threshold = 0`` (default), no iteration should
    record any Newton CG work in the trace."""
    mol, basis = oh_radical
    opts = UHFOptions()
    opts.max_iter = 250
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    r = run_uhf(mol, basis, opts)
    assert r.converged
    assert all(it.newton_cg_iter == 0 for it in r.scf_trace)


# ---------------------------------------------------------------------------
# Newton parity — converges to the same fixed point as DIIS.
# ---------------------------------------------------------------------------

def test_uhf_newton_matches_diis_oh_radical(oh_radical):
    """UHF Newton after a DIIS warm-up reaches the same converged
    energy and the same <S^2> as plain DIIS. The accelerator changes
    the path through the SCF manifold, not the converged fixed
    point."""
    mol, basis = oh_radical
    base = UHFOptions()
    base.max_iter = 250
    base.conv_tol_energy = 1e-10
    base.conv_tol_grad = 1e-8
    r_diis = run_uhf(mol, basis, base)

    so = UHFOptions()
    so.max_iter = 250
    so.conv_tol_energy = 1e-10
    so.conv_tol_grad = 1e-8
    so.newton_threshold = 1.0
    r_newton = run_uhf(mol, basis, so)

    assert r_diis.converged
    assert r_newton.converged
    assert r_newton.energy == pytest.approx(r_diis.energy, abs=1e-9)
    assert r_newton.s_squared == pytest.approx(r_diis.s_squared, abs=1e-7)


def test_uhf_newton_activates_in_asymptotic_regime(oh_radical):
    """With ``newton_threshold = 1.0`` enabled on OH·, at least one
    iteration in the trace records ``newton_cg_iter > 0`` — the Newton
    Newton step ran on some iteration."""
    mol, basis = oh_radical
    opts = UHFOptions()
    opts.max_iter = 250
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.newton_threshold = 1.0
    r = run_uhf(mol, basis, opts)
    assert r.converged
    activated = [it for it in r.scf_trace if it.newton_cg_iter > 0]
    assert len(activated) > 0
    # CG should converge in a sensible number of iters (well below
    # the cg_max_iter = 50 default) once the SCF gradient is below
    # the activation threshold.
    assert max(it.newton_cg_iter for it in activated) <= 30


def test_uhf_newton_plus_ediis_diis_hybrid_matches_diis(oh_radical):
    """Composing Newton (Newton finisher) with the EDIIS+DIIS hybrid
    warm-up should still reach the same fixed point. This is the
    "production" stack: EDIIS+DIIS far from convergence → Newton
    near convergence."""
    mol, basis = oh_radical
    base = UHFOptions()
    base.max_iter = 250
    base.conv_tol_energy = 1e-10
    base.conv_tol_grad = 1e-8
    r_diis = run_uhf(mol, basis, base)

    full = UHFOptions()
    full.max_iter = 250
    full.conv_tol_energy = 1e-10
    full.conv_tol_grad = 1e-8
    full.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    full.newton_threshold = 1.0
    r_full = run_uhf(mol, basis, full)

    assert r_diis.converged
    assert r_full.converged
    assert r_full.energy == pytest.approx(r_diis.energy, abs=1e-9)
    assert r_full.s_squared == pytest.approx(r_diis.s_squared, abs=1e-7)


# ---------------------------------------------------------------------------
# Edge case — one spin has no virtual subspace.
# ---------------------------------------------------------------------------

def test_uhf_newton_handles_single_electron_atom():
    """Hydrogen atom (1α electron, 0β electrons, 1 sto-3g basis
    function): the β spin has n_occ_β = 0 — the β κ block is
    degenerate and the Newton step should rotate α only. This pins
    that the empty-spin guard in ``uhf_newton_step`` doesn't crash."""
    mol = make_molecule(_h_atom_atoms(), multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    opts = UHFOptions()
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.newton_threshold = 1.0
    r = run_uhf(mol, basis, opts)
    assert r.converged
    # H atom in STO-3G: 1 AO, n_α = 1, n_β = 0, no virtuals on either
    # spin — Newton is essentially a no-op rotation but mustn't crash.
    # <S^2> = 0.75 exactly (single unpaired electron, no contamination).
    assert r.s_squared == pytest.approx(0.75, abs=1e-9)
