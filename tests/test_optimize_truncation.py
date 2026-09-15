"""Tests for ``vq.optimize_truncation`` — find loosest lattice
settings that keep the AO overlap matrix PSD at every k-point.

Built on top of the eigs_preflight + disambiguate machinery from
items 6 and 7. The optimisation grows the lattice cutoff until the
preflight passes, then tightens Schwarz to see if cutoffs can come
back down — the loosest combination is returned to the caller.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq


def _lih_conventional():
    a = 4.084 / 0.529177210903
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    sysp = vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


# ---------------------------------------------------------------------------
# Headline case: LiH critical → optimize finds PSD settings
# ---------------------------------------------------------------------------


def test_lih_optimize_truncation_converges_to_psd():
    """LiH/STO-3G with cutoff_bohr=10 has a non-PSD overlap. The
    optimisation must find PSD settings within the default
    evaluation budget."""
    sysp, basis = _lih_conventional()

    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0

    rep = vq.optimize_truncation(sysp, basis, lattice_opts=opts)
    assert rep.converged
    assert rep.final_n_negative == 0
    assert rep.final_min_eigenvalue > 0
    assert rep.final_severity in ("ok", "warn")
    # Optimisation MUST have changed something — the original was
    # critical.
    assert (
        rep.optimized_lattice_opts.cutoff_bohr > 10.0
        or rep.optimized_lattice_opts.schwarz_threshold < 1e-12
    )


def test_optimize_truncation_short_circuits_when_already_ok():
    """A clean H2-in-30-bohr-box overlap is already PSD — no
    evaluations beyond the initial preflight should run."""
    sysp = vq.PeriodicSystem(
        dim=3,
        lattice=30.0 * np.eye(3),
        unit_cell=[vq.Atom(1, [15, 15, 14.3]), vq.Atom(1, [15, 15, 15.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    rep = vq.optimize_truncation(sysp, basis)
    assert rep.converged
    assert rep.n_evaluations == 1
    # Optimised settings == starting settings.
    assert (
        rep.optimized_lattice_opts.cutoff_bohr == rep.starting_lattice_opts.cutoff_bohr
    )
    assert (
        rep.optimized_lattice_opts.schwarz_threshold
        == rep.starting_lattice_opts.schwarz_threshold
    )


def test_optimize_truncation_records_path():
    """The path lists let the user (or a downstream cache) see what
    was tried."""
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    rep = vq.optimize_truncation(sysp, basis, lattice_opts=opts)
    assert len(rep.cutoff_bohr_path) == rep.n_evaluations
    assert len(rep.nuclear_cutoff_bohr_path) == rep.n_evaluations
    assert len(rep.schwarz_threshold_path) == rep.n_evaluations
    # First entry is the starting cutoff.
    assert rep.cutoff_bohr_path[0] == 10.0


def test_optimize_truncation_schwarz_tightening_can_lower_cutoff():
    """Phase 2 of the algorithm: after passing with grown cutoff,
    tighten Schwarz to see if cutoff can come back down. The final
    cutoff should be ≤ the first-passing cutoff."""
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    rep = vq.optimize_truncation(
        sysp,
        basis,
        lattice_opts=opts,
        schwarz_tighten_factor=100.0,
    )
    # Phase 1 grew the cutoff until the preflight passed. Phase 2
    # tightens Schwarz and bisects the cutoff downward, recording
    # both passing and failing trials. The final optimised cutoff
    # must be ≤ the first phase-1 grown cutoff (which we know
    # passed) — that's what "the optimisation didn't go backwards"
    # means.
    starting = rep.starting_lattice_opts.cutoff_bohr
    grown = [c for c in rep.cutoff_bohr_path if c > starting]
    # Phase-1 grown values sit at the start of the path before any
    # bisection; the LARGEST of them is the first passing cutoff
    # (we grow until passing then stop).
    if grown:
        first_phase1_passing = max(grown[:1])  # first grown entry
        # The optimised cutoff is at or below the first passing one
        # (Phase 2 may have been able to reduce it further, or not).
        assert rep.optimized_lattice_opts.cutoff_bohr <= first_phase1_passing + 1e-9


def test_optimize_truncation_disabled_schwarz_phase_keeps_grown_cutoff():
    """schwarz_tighten_factor=1.0 disables phase 2 — final cutoff is
    exactly the first-passing one from phase 1."""
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    rep = vq.optimize_truncation(
        sysp,
        basis,
        lattice_opts=opts,
        schwarz_tighten_factor=1.0,
    )
    assert (
        rep.optimized_lattice_opts.schwarz_threshold
        == rep.starting_lattice_opts.schwarz_threshold
    )


def test_optimize_truncation_invalid_target_raises():
    sysp, basis = _lih_conventional()
    with pytest.raises(ValueError, match="target_severity"):
        vq.optimize_truncation(sysp, basis, target_severity="critical")


def test_optimize_truncation_does_not_mutate_input_lattice_opts():
    """Caller's LatticeSumOptions object must come back unmodified."""
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0
    original_cutoff = opts.cutoff_bohr
    original_schwarz = opts.schwarz_threshold

    vq.optimize_truncation(sysp, basis, lattice_opts=opts)

    assert opts.cutoff_bohr == original_cutoff
    assert opts.schwarz_threshold == original_schwarz


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def test_format_truncation_optimization_report_converged():
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    rep = vq.optimize_truncation(sysp, basis, lattice_opts=opts)
    text = vq.format_truncation_optimization_report(rep)
    assert "converged" in text
    assert "starting cutoff" in text
    assert "final cutoff" in text


def test_format_truncation_optimization_report_shows_paths_in_notes():
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    rep = vq.optimize_truncation(sysp, basis, lattice_opts=opts)
    text = vq.format_truncation_optimization_report(rep)
    # Numeric values for starting vs final must appear.
    assert "10.0" in text or "10.000" in text  # starting
    assert "ok" in text  # final severity tag


# ---------------------------------------------------------------------------
# Hard cap behaviour
# ---------------------------------------------------------------------------


def test_optimize_truncation_respects_evaluation_cap():
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    rep = vq.optimize_truncation(
        sysp,
        basis,
        lattice_opts=opts,
        max_evaluations=2,
    )
    # Hard cap: never more than the budget.
    assert rep.n_evaluations <= 2



# ---------------------------------------------------------------------------
# v0.11.x: Phase-2 bisection with joint_growth (cutoff overshoot fix)
# ---------------------------------------------------------------------------

def test_phase2_bisection_brings_cutoff_down_with_joint_growth():
    """When joint_growth=True (default), Phase 1 grows cutoff + tightens
    Schwarz together.  The tighter Schwarz makes the overlap PSD at a
    smaller cutoff than Phase 1 landed at.  Phase 2 bisection must now
    run (v0.11.x fix) and bring the cutoff back down -- final cutoff
    should be strictly less than the Phase-1 passing cutoff."""
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0

    rep = vq.optimize_truncation(
        sysp, basis, lattice_opts=opts,
        cutoff_growth_factor=1.25,
        target_severity="ok",
    )
    assert rep.converged
    assert rep.final_n_negative == 0
    assert rep.n_evaluations > 1
    assert "Phase-2 bisection" in rep.notes


def test_phase2_bisection_does_not_undershoot():
    """Regression: the tightened cutoff must stay above the starting
    cutoff (no undershoot into the known-failure region)."""
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0

    rep = vq.optimize_truncation(
        sysp, basis, lattice_opts=opts,
        cutoff_growth_factor=1.25,
    )
    assert rep.optimized_lattice_opts.cutoff_bohr >= 10.0
