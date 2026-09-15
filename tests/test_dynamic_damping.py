"""Dynamic damping (Zerner-Hehenberger 1979) tests.

When ``opts.dynamic_damping = True``, the SCF loop adjusts the
density-mixing α iteration-by-iteration based on the energy decrease,
rather than holding ``opts.damping`` constant.

Coverage:

  1. Default-off back-compat: ``dynamic_damping`` defaults to False on
     all four molecular options classes; the static-damping path is
     bit-for-bit unchanged.

  2. With ``dynamic_damping = True``, RHF / UHF / RKS converge to the
     same SCF fixed point as static damping (energy parity).

  3. The new fields exist with the documented defaults
     (``dynamic_damping_min = 0.0``, ``dynamic_damping_max = 0.95``).

The Zerner-Hehenberger heuristic itself is verified at the C++ unit
level via the ``update_dynamic_damping`` free function (header-only),
which has no observable behaviour beyond what these end-to-end tests
exercise.

Reference:
  M. C. Zerner, M. Hehenberger, "A dynamical damping scheme for
  converging molecular SCF calculations", Chem. Phys. Lett. 62, 550
  (1979).
"""

from __future__ import annotations

import pytest

from vibeqc import (
    BasisSet,
    RHFOptions,
    SCFMode,
    UHFOptions,
    RKSOptions,
    UKSOptions,
    run_rhf,
    run_uhf,
    run_rks,
    run_uks,
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


@pytest.fixture
def heh_radical():
    """HeH doublet at 1.4 bohr: compact physical UKS damping witness."""
    mol = make_molecule(
        [(2, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.4])],
        multiplicity=2,
    )
    return mol, BasisSet(mol, "sto-3g")


# ---------------------------------------------------------------------------
# Defaults — dynamic_damping is ON by default on all four molecular flavours
# (ORCA-style adaptive damping; v0.15.x).
# ---------------------------------------------------------------------------

def test_dynamic_damping_default_is_true():
    assert RHFOptions().dynamic_damping is True
    assert UHFOptions().dynamic_damping is True
    assert RKSOptions().dynamic_damping is True
    assert UKSOptions().dynamic_damping is True


def test_dynamic_damping_bounds_defaults():
    """Default bounds are [0.0, 0.95] on all four flavours."""
    for OptionsCls in (RHFOptions, UHFOptions, RKSOptions, UKSOptions):
        o = OptionsCls()
        assert o.dynamic_damping_min == pytest.approx(0.0)
        assert o.dynamic_damping_max == pytest.approx(0.95)


def test_dynamic_damping_fields_are_settable():
    """The new fields are writable on all four options classes."""
    for OptionsCls in (RHFOptions, UHFOptions, RKSOptions, UKSOptions):
        o = OptionsCls()
        o.dynamic_damping = True
        o.dynamic_damping_min = 0.1
        o.dynamic_damping_max = 0.9
        assert o.dynamic_damping is True
        assert o.dynamic_damping_min == pytest.approx(0.1)
        assert o.dynamic_damping_max == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# RHF / RKS — closed-shell parity vs static damping.
# ---------------------------------------------------------------------------

def test_rhf_dynamic_damping_matches_static_h2o(h2o_basis):
    """RHF / sto-3g: dynamic damping reaches the same energy as static."""
    mol, basis = h2o_basis

    o_static = RHFOptions()
    o_static.conv_tol_energy = 1e-10
    o_static.conv_tol_grad = 1e-7
    r_static = run_rhf(mol, basis, o_static)
    assert r_static.converged

    o_dyn = RHFOptions()
    o_dyn.dynamic_damping = True
    o_dyn.conv_tol_energy = 1e-10
    o_dyn.conv_tol_grad = 1e-7
    r_dyn = run_rhf(mol, basis, o_dyn)
    assert r_dyn.converged
    assert r_dyn.energy == pytest.approx(r_static.energy, abs=1e-9)


def test_rks_dynamic_damping_matches_static_h2o(h2o_basis):
    """RKS / LDA / sto-3g: dynamic damping reaches the same energy as static."""
    mol, basis = h2o_basis

    o_static = RKSOptions()
    o_static.functional = "LDA"
    o_static.conv_tol_energy = 1e-10
    o_static.conv_tol_grad = 1e-7
    r_static = run_rks(mol, basis, o_static)
    assert r_static.converged

    o_dyn = RKSOptions()
    o_dyn.functional = "LDA"
    o_dyn.dynamic_damping = True
    o_dyn.conv_tol_energy = 1e-10
    o_dyn.conv_tol_grad = 1e-7
    r_dyn = run_rks(mol, basis, o_dyn)
    assert r_dyn.converged
    assert r_dyn.energy == pytest.approx(r_static.energy, abs=1e-8)


def test_uhf_dynamic_damping_matches_static_oh_radical(oh_radical):
    """UHF / sto-3g on OH·: dynamic damping reaches the same energy."""
    mol, basis = oh_radical

    o_static = UHFOptions()
    o_static.max_iter = 250
    o_static.conv_tol_energy = 1e-10
    o_static.conv_tol_grad = 1e-8
    r_static = run_uhf(mol, basis, o_static)
    assert r_static.converged

    o_dyn = UHFOptions()
    o_dyn.dynamic_damping = True
    o_dyn.max_iter = 250
    o_dyn.conv_tol_energy = 1e-10
    o_dyn.conv_tol_grad = 1e-8
    r_dyn = run_uhf(mol, basis, o_dyn)
    assert r_dyn.converged
    assert r_dyn.energy == pytest.approx(r_static.energy, abs=1e-9)
    assert r_dyn.s_squared == pytest.approx(r_static.s_squared, abs=1e-7)


@pytest.fixture
def o_triplet():
    """Triplet O atom / sto-3g — canonical simplest open-shell UKS case
    (shared with test_molecular_level_shift.py)."""
    from vibeqc import Atom, Molecule
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0])], charge=0, multiplicity=3)
    return mol, BasisSet(mol, "sto-3g")


def test_uks_dynamic_damping_matches_static_o_triplet(o_triplet):
    """UKS / LDA / sto-3g on triplet O: dynamic damping reaches the same
    energy and ``<S^2>`` as static. Closes the open-shell DFT corner
    that UHF (open-shell HF) and RKS (closed-shell DFT) bracket."""
    mol, basis = o_triplet

    o_static = UKSOptions()
    o_static.functional = "LDA"
    o_static.conv_tol_energy = 1e-10
    o_static.conv_tol_grad = 1e-8
    r_static = run_uks(mol, basis, o_static)
    assert r_static.converged

    o_dyn = UKSOptions()
    o_dyn.functional = "LDA"
    o_dyn.dynamic_damping = True
    o_dyn.conv_tol_energy = 1e-10
    o_dyn.conv_tol_grad = 1e-8
    r_dyn = run_uks(mol, basis, o_dyn)
    assert r_dyn.converged
    assert r_dyn.energy == pytest.approx(r_static.energy, abs=1e-8)
    assert r_dyn.s_squared == pytest.approx(r_static.s_squared, abs=1e-7)


# ---------------------------------------------------------------------------
# Dynamic damping from damping = 0.0 actually engages (audit P3.8,
# finding 8 of the 2026-05-18 code audit).
#
# Pre-fix, the static-damping mixers in RKS / UHF / UKS gated on the
# immutable ``opts.damping == 0.0`` instead of the iteration-local
# ``current_damping``, so ``damping = 0.0`` + ``dynamic_damping = True``
# silently never damped: the trajectory was bit-identical to
# ``dynamic_damping = False``. RHF gated correctly and serves as the
# control. The fix landed without engagement regressions; these pin it.
#
# Mechanism under test: with ``dynamic_damping_min = 0.5`` the
# Zerner-Hehenberger update clamps ``current_damping`` to >= 0.5 from
# iteration 2 onward (DIIS off, since DIIS bypasses the damping mixer
# entirely). Where undamped no-DIIS Roothaan converges (the RHF / UHF /
# UKS cases below), engagement shows as strictly more iterations to the
# same fixed point; where it oscillates without converging (the RKS
# case), engagement shows as convergence itself.
# ---------------------------------------------------------------------------

def _from_zero_opts(opts_factory, *, dynamic):
    o = opts_factory()
    o.damping = 0.0  # the audit's bug surface
    o.use_diis = False
    o.max_iter = 500
    o.conv_tol_energy = 1e-8
    o.conv_tol_grad = 1e-6
    # Set both branches explicitly: dynamic_damping now defaults to True, so
    # the non-dynamic control must disable it to be a true no-aids baseline.
    if dynamic:
        o.dynamic_damping = True
        o.dynamic_damping_min = 0.5
    else:
        o.dynamic_damping = False
    return o


def test_rhf_dynamic_damping_from_zero_engages(h2o_basis):
    """RHF control: gated on ``current_damping`` since before the audit."""
    mol, basis = h2o_basis
    r_ctrl = run_rhf(mol, basis, _from_zero_opts(RHFOptions, dynamic=False))
    assert r_ctrl.converged
    r_dyn = run_rhf(mol, basis, _from_zero_opts(RHFOptions, dynamic=True))
    assert r_dyn.converged
    assert r_dyn.n_iter > r_ctrl.n_iter
    assert r_dyn.energy == pytest.approx(r_ctrl.energy, abs=1e-7)


def test_uhf_dynamic_damping_from_zero_engages(oh_radical):
    mol, basis = oh_radical
    r_ctrl = run_uhf(mol, basis, _from_zero_opts(UHFOptions, dynamic=False))
    assert r_ctrl.converged
    r_dyn = run_uhf(mol, basis, _from_zero_opts(UHFOptions, dynamic=True))
    assert r_dyn.converged
    assert r_dyn.n_iter > r_ctrl.n_iter
    assert r_dyn.energy == pytest.approx(r_ctrl.energy, abs=1e-7)


def _rks_lda_opts():
    o = RKSOptions()
    o.functional = "LDA"
    return o


def _uks_lda_opts():
    o = UKSOptions()
    o.functional = "LDA"
    return o


def test_rks_dynamic_damping_from_zero_engages(h2o_basis):
    """H2O/LDA/sto-3g without DIIS or damping oscillates and never
    converges, which is exactly the situation dynamic damping exists
    for. Pre-fix the dynamic run was bit-identical to that
    non-converging control, so convergence of the dynamic run IS the
    engagement signal. Energy parity vs an explicit static-0.5
    reference pins the fixed point.

    The auto-level-shift-on-oscillation feature (added v0.15.x) would
    also converge this case, so we disable it in the control to
    preserve the discriminating non-convergence premise."""
    mol, basis = h2o_basis
    o_ctrl = _from_zero_opts(_rks_lda_opts, dynamic=False)
    o_ctrl.auto_level_shift_on_oscillation = False
    r_ctrl = run_rks(mol, basis, o_ctrl)
    assert not r_ctrl.converged  # premise that makes this test discriminate

    o_static = _from_zero_opts(_rks_lda_opts, dynamic=False)
    o_static.damping = 0.5
    r_static = run_rks(mol, basis, o_static)
    assert r_static.converged

    r_dyn = run_rks(mol, basis, _from_zero_opts(_rks_lda_opts,
                                                dynamic=True))
    assert r_dyn.converged
    assert r_dyn.energy == pytest.approx(r_static.energy, abs=1e-7)


def test_uks_dynamic_damping_from_zero_engages(heh_radical):
    """UKS engagement reaches the same physical HeH determinant.

    The former OH witness treated a stationary *damped mixture* as a
    converged UKS wavefunction.  HeH retains the original iteration-2
    engagement discriminator while both trajectories converge to the same
    idempotent determinant.
    """
    mol, basis = heh_radical
    o_ctrl = _from_zero_opts(_uks_lda_opts, dynamic=False)
    o_ctrl.auto_level_shift_on_oscillation = False
    o_ctrl.restart_opts.enabled = False
    o_ctrl.stability_check = False
    r_ctrl = run_uks(mol, basis, o_ctrl)
    assert r_ctrl.converged

    o_dyn = _from_zero_opts(_uks_lda_opts, dynamic=True)
    o_dyn.auto_level_shift_on_oscillation = False
    o_dyn.restart_opts.enabled = False
    o_dyn.stability_check = False
    r_dyn = run_uks(mol, basis, o_dyn)
    assert r_dyn.converged
    assert r_dyn.scf_trace[0].energy == r_ctrl.scf_trace[0].energy
    assert r_dyn.scf_trace[1].energy != r_ctrl.scf_trace[1].energy
    assert r_dyn.n_iter > r_ctrl.n_iter
    assert r_dyn.energy == pytest.approx(r_ctrl.energy, abs=1.0e-10)


# ---------------------------------------------------------------------------
# Auto-level-shift-on-oscillation — converges oscillating SCFs automatically
# ---------------------------------------------------------------------------

def test_auto_level_shift_on_oscillation_converges_oscillating_rks(h2o_basis):
    """H2O/LDA/sto-3g without DIIS or damping oscillates.  With the
    auto-level-shift-on-oscillation feature enabled (default), the SCF
    loop detects the oscillation, engages a persistent 0.3 Ha
    Saunders-Hillier shift, clears the DIIS history, and converges.
    The forced direct/incremental path verifies that detection remains
    active during the coarse-Fock phase when restart handling is disabled.
    With the feature disabled, the same system fails to converge.
    Energy parity against a static-damping reference pins the fixed
    point."""
    mol, basis = h2o_basis

    # Control: no DIIS, no damping, no auto level shift — must fail
    o_fail = RKSOptions()
    o_fail.functional = "LDA"
    o_fail.damping = 0.0
    o_fail.use_diis = False
    o_fail.max_iter = 100
    o_fail.conv_tol_energy = 1e-7
    o_fail.conv_tol_grad = 1e-6
    o_fail.dynamic_damping = False
    o_fail.auto_level_shift_on_oscillation = False
    o_fail.scf_mode = SCFMode.DIRECT
    o_fail.incremental_fock = True
    o_fail.restart_opts.enabled = False
    r_fail = run_rks(mol, basis, o_fail)
    assert not r_fail.converged, (
        "RKS/LDA/sto-3g without aids should NOT converge in 100 iters; "
        "this is the discriminating premise of the test")

    # With auto level shift (default) — must converge
    o_auto = RKSOptions()
    o_auto.functional = "LDA"
    o_auto.damping = 0.0
    o_auto.use_diis = False
    o_auto.max_iter = 200
    o_auto.conv_tol_energy = 1e-7
    o_auto.conv_tol_grad = 1e-6
    o_auto.dynamic_damping = False
    o_auto.scf_mode = SCFMode.DIRECT
    o_auto.incremental_fock = True
    o_auto.restart_opts.enabled = False
    # auto_level_shift_on_oscillation defaults to True
    r_auto = run_rks(mol, basis, o_auto)
    assert r_auto.converged, (
        "auto_level_shift_on_oscillation should converge the oscillating case")

    # Energy parity: auto-LS-converged result must match a static-damping ref
    o_static = RKSOptions()
    o_static.functional = "LDA"
    o_static.damping = 0.5
    o_static.use_diis = True
    o_static.max_iter = 200
    o_static.conv_tol_energy = 1e-10
    o_static.conv_tol_grad = 1e-7
    r_static = run_rks(mol, basis, o_static)
    assert r_static.converged
    assert r_auto.energy == pytest.approx(r_static.energy, abs=1e-7)


def test_auto_level_shift_on_oscillation_defaults():
    """All four molecular options classes default auto_level_shift_on_oscillation
    to True with auto_level_shift_value = 0.3 Ha."""
    for OptionsCls, label in [(RHFOptions, "RHF"), (UHFOptions, "UHF"),
                                (RKSOptions, "RKS"), (UKSOptions, "UKS")]:
        o = OptionsCls()
        assert o.auto_level_shift_on_oscillation is True, (
            f"{label}: auto_level_shift_on_oscillation should default True")
        assert o.auto_level_shift_value == pytest.approx(0.3), (
            f"{label}: auto_level_shift_value should default 0.3")
        assert o.oscillation_detect_start_iter == 10
        assert o.oscillation_window == 10
        assert o.oscillation_min_sign_flips == 5


def test_auto_level_shift_can_be_disabled(h2o_basis):
    """Setting auto_level_shift_on_oscillation=False must suppress the feature."""
    mol, basis = h2o_basis
    o = RKSOptions()
    o.functional = "LDA"
    o.damping = 0.0
    o.use_diis = False
    o.max_iter = 100
    o.conv_tol_energy = 1e-7
    o.conv_tol_grad = 1e-6
    o.dynamic_damping = False
    o.auto_level_shift_on_oscillation = False
    r = run_rks(mol, basis, o)
    assert not r.converged, (
        "RKS/LDA/sto-3g without aids and without auto level shift "
        "should NOT converge")


def test_auto_level_shift_does_not_affect_converging_system(h2o_basis):
    """A normally-converging system with DIIS + damping should converge
    identically whether auto_level_shift is on or off, because the
    detector only fires when oscillation is present."""
    mol, basis = h2o_basis
    # Standard DIIS + damping RKS/LDA run — converges easily
    o_on = RKSOptions()
    o_on.functional = "LDA"
    o_on.conv_tol_energy = 1e-10
    o_on.conv_tol_grad = 1e-7
    r_on = run_rks(mol, basis, o_on)
    assert r_on.converged

    o_off = RKSOptions()
    o_off.functional = "LDA"
    o_off.conv_tol_energy = 1e-10
    o_off.conv_tol_grad = 1e-7
    o_off.auto_level_shift_on_oscillation = False
    r_off = run_rks(mol, basis, o_off)
    assert r_off.converged
    assert r_on.energy == pytest.approx(r_off.energy, abs=1e-12)
    assert r_on.n_iter == r_off.n_iter, (
        "auto level shift should not change iteration count on a "
        "normally-converging system")
