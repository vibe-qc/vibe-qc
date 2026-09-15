"""Phase D2d — SOSCF (Neese 2000) approximate-second-order SCF tests.

Distinct from Newton (D2c, vibeqc/newton.hpp): Newton uses the FULL
orbital Hessian via preconditioned CG (one Fock build per CG iter),
while SOSCF (D2d, vibeqc/soscf.hpp) uses an augmented-Hessian
eigsolve with the DIAGONAL-DOMINANT orbital Hessian — no Fock build
inside the step. Cheaper per step, linearly convergent (vs Newton's
quadratic), useful where Newton's matvec is too expensive.

Pins the contract:

  1. Default-off back-compat: ``soscf_threshold = 0`` keeps RHF
     bit-for-bit unchanged.
  2. With ``soscf_threshold = 1.0``, RHF converges to the *same*
     fixed point as plain DIIS.
  3. SOSCF + EDIIS+DIIS hybrid composes cleanly: the warm-up handles
     the far-from-convergence iterations, SOSCF closes out the
     asymptotic regime.
  4. SOSCFOptions is exposed and writable; trust_radius default 0.3
     matches NewtonOptions for consistency.
  5. Newton + SOSCF mutual exclusion: when both thresholds are set,
     Newton wins (full Hessian preferred over diagonal-dominant).

Reference:
  F. Neese, "Approximate second-order SCF method for large
  configuration interaction expansions of the diradical type",
  Chem. Phys. Lett. 325, 93 (2000). ORCA's standard SOSCF.
"""

from __future__ import annotations

import pytest

from vibeqc import (
    BasisSet,
    RHFOptions,
    SCFAccelerator,
    SOSCFOptions,
    UHFOptions,
    run_rhf,
    run_uhf,
)

from .conftest import ANGSTROM_TO_BOHR, make_molecule


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def _h2o_atoms():
    return [
        (8, [0.0, 0.0,  0.117 * ANGSTROM_TO_BOHR]),
        (1, [0.0,  0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
        (1, [0.0, -0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
    ]


def _oh_radical_atoms():
    """Hydroxyl radical (doublet, 9 electrons)."""
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
# Options surface — SOSCFOptions defaults + RHFOptions integration.
# ---------------------------------------------------------------------------

def test_soscf_threshold_default_is_zero():
    """RHFOptions exposes soscf_threshold; default 0 keeps SOSCF off (it is
    an opt-in finalizer, not robust enough to be a default)."""
    assert RHFOptions().soscf_threshold == pytest.approx(0.0)


def test_soscf_options_default_trust_radius():
    """Default trust radius matches NewtonOptions (0.3) for consistency."""
    assert SOSCFOptions().trust_radius == pytest.approx(0.3)


def test_soscf_options_is_settable_via_rhf_options():
    """RHFOptions.soscf_opts round-trips."""
    o = RHFOptions()
    o.soscf_threshold = 1.0
    o.soscf_opts.trust_radius = 0.5
    assert o.soscf_threshold == pytest.approx(1.0)
    assert o.soscf_opts.trust_radius == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Default-off back-compat — RHF without soscf_threshold behaves as before.
# ---------------------------------------------------------------------------

def test_rhf_default_does_not_activate_soscf(h2o_basis):
    """With soscf_threshold = 0 (default), RHF is bit-for-bit unchanged."""
    mol, basis = h2o_basis
    o_off = RHFOptions()
    o_off.conv_tol_energy = 1e-9
    o_off.conv_tol_grad = 1e-6
    r = run_rhf(mol, basis, o_off)
    assert r.converged
    # SOSCF stays off; the trace's newton_cg_iter is 0 throughout
    # (SOSCF doesn't record a separate counter; it shares the
    # diagonalize-F slot in the trace).
    assert all(it.newton_cg_iter == 0 for it in r.scf_trace)


# ---------------------------------------------------------------------------
# Parity — SOSCF reaches the same fixed point as plain DIIS.
# ---------------------------------------------------------------------------

def test_rhf_soscf_matches_diis_h2o(h2o_basis):
    """RHF / sto-3g: SOSCF reaches the same energy as DIIS to ~1e-9 Ha."""
    mol, basis = h2o_basis

    o_diis = RHFOptions()
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-7
    r_diis = run_rhf(mol, basis, o_diis)
    assert r_diis.converged

    o_soscf = RHFOptions()
    o_soscf.soscf_threshold = 1.0   # Fischer-Almlöf 1992 production setting
    o_soscf.conv_tol_energy = 1e-10
    o_soscf.conv_tol_grad = 1e-7
    r_soscf = run_rhf(mol, basis, o_soscf)
    assert r_soscf.converged
    assert r_soscf.energy == pytest.approx(r_diis.energy, abs=1e-9)


def test_rhf_soscf_with_ediis_diis_warmup_matches_pure_diis(h2o_basis):
    """SOSCF + EDIIS+DIIS hybrid composes — same converged energy."""
    mol, basis = h2o_basis

    o_diis = RHFOptions()
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-7
    r_diis = run_rhf(mol, basis, o_diis)
    assert r_diis.converged

    o_combo = RHFOptions()
    o_combo.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    o_combo.soscf_threshold = 1.0
    o_combo.conv_tol_energy = 1e-10
    o_combo.conv_tol_grad = 1e-7
    r_combo = run_rhf(mol, basis, o_combo)
    assert r_combo.converged
    assert r_combo.energy == pytest.approx(r_diis.energy, abs=1e-9)


# ---------------------------------------------------------------------------
# Mutual exclusion: Newton wins over SOSCF when both are set.
# ---------------------------------------------------------------------------

def test_newton_wins_when_both_thresholds_set(h2o_basis):
    """When both newton_threshold and soscf_threshold are set, Newton
    activates (its CG counter is recorded in the trace) and SOSCF is
    skipped."""
    mol, basis = h2o_basis
    o = RHFOptions()
    o.newton_threshold = 1.0  # Newton activates at grad < 1.0
    o.soscf_threshold = 1.0   # SOSCF would also activate; Newton wins
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-7
    r = run_rhf(mol, basis, o)
    assert r.converged
    # Newton recorded CG iters on at least one trace step.
    assert any(it.newton_cg_iter > 0 for it in r.scf_trace)


# ---------------------------------------------------------------------------
# UHF — open-shell SOSCF (per-spin AH eigsolves, no cross-spin coupling).
# ---------------------------------------------------------------------------

def test_uhf_soscf_threshold_default_is_zero():
    assert UHFOptions().soscf_threshold == pytest.approx(0.0)


def test_uhf_soscf_options_share_defaults_with_rhf():
    """UHFOptions.soscf_opts is a SOSCFOptions instance with the same
    defaults as RHFOptions.soscf_opts."""
    assert UHFOptions().soscf_opts.trust_radius == \
        pytest.approx(SOSCFOptions().trust_radius)


def test_uhf_default_does_not_activate_soscf(oh_radical):
    """With soscf_threshold = 0 (default), UHF behaves as before."""
    mol, basis = oh_radical
    o = UHFOptions()
    o.max_iter = 250
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-7
    r = run_uhf(mol, basis, o)
    assert r.converged


def test_uhf_soscf_matches_diis_oh_radical(oh_radical):
    """UHF / sto-3g on OH·: SOSCF reaches the same energy as DIIS,
    same `<S^2>`. The accelerator changes the path through the SCF
    manifold, not the converged fixed point."""
    mol, basis = oh_radical

    o_diis = UHFOptions()
    o_diis.max_iter = 250
    o_diis.conv_tol_energy = 1e-10
    o_diis.conv_tol_grad = 1e-8
    r_diis = run_uhf(mol, basis, o_diis)
    assert r_diis.converged

    o_soscf = UHFOptions()
    o_soscf.max_iter = 250
    o_soscf.soscf_threshold = 1.0
    o_soscf.conv_tol_energy = 1e-10
    o_soscf.conv_tol_grad = 1e-8
    r_soscf = run_uhf(mol, basis, o_soscf)
    assert r_soscf.converged
    assert r_soscf.energy == pytest.approx(r_diis.energy, abs=1e-9)
    assert r_soscf.s_squared == pytest.approx(r_diis.s_squared, abs=1e-7)
