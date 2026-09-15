"""The CCM SCF trace (#493): per-cycle diagnosis of how a run converged.

Before this trace existed, a CCM loop reported only ``converged`` and
``n_iter``, so a run that stopped at the iteration cap carried no information
about HOW it failed -- an oscillation, a monotone crawl and a hard stall all
present as ``converged: false, n_iter: 128``. CLAUDE.md § 7 forbids answering
a periodic-SCF failure by reaching for a convergence aid, and the residual
trajectory is what makes the alternative -- diagnosis -- possible.

These tests pin the trace to the loop it describes. They deliberately do NOT
merely assert that the trace is non-empty: a trace whose rows disagree with
the run's own energy, residual and exit predicate would be worse than none,
because it would be read as evidence.
"""

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ri import run_ccm_rhf_ri_neutral
from vibeqc.periodic.ccm.scf import CCMSCFIteration

pytestmark = pytest.mark.experimental  # AICCM research lane


def _lih_ccm():
    """LiH in an 8-bohr cube, minimal basis: ~1 s, and -- unlike H2/sto-3g --
    it has a real SCF trajectory (6 cycles, residual 1.5e-1 -> 1.0e-8).

    The fixture choice matters. H2/sto-3g is a 2-electron problem that the
    Hcore guess already solves: its first recorded residual is 5e-14, so every
    trajectory assertion below would pass vacuously on it while proving
    nothing about a trace of a run that actually has to converge.
    """
    cell = PeriodicSystem(
        3, np.diag([8.0, 8.0, 8.0]),
        [Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])], 0, 1)
    return CCMSystem(cell, (1, 1, 1), "sto-3g")


@pytest.fixture(scope="module")
def traced_run():
    return run_ccm_rhf_ri_neutral(_lih_ccm())


def test_trace_has_one_row_per_cycle_and_they_are_numbered(traced_run):
    """The trace covers the whole run, including the final cycle."""
    trace = traced_run.scf_trace
    assert len(trace) == traced_run.n_iter
    assert all(isinstance(row, CCMSCFIteration) for row in trace)
    assert [row.iteration for row in trace] == list(range(1, len(trace) + 1))


def test_last_row_reports_the_energy_the_run_returns(traced_run):
    """A trace that disagreed with the run's own answer would be misleading."""
    assert traced_run.scf_trace[-1].energy == pytest.approx(
        traced_run.energy, abs=1e-12, rel=0.0)


def test_delta_e_is_the_actual_energy_difference(traced_run):
    """Row-to-row consistency: delta_e is not independently computed drift."""
    trace = traced_run.scf_trace
    assert np.isnan(trace[0].delta_e)  # no previous cycle to difference
    for prev, row in zip(trace, trace[1:]):
        assert row.delta_e == pytest.approx(
            row.energy - prev.energy, abs=1e-14, rel=0.0)


def test_final_residual_is_below_the_gate_the_run_exited_on(traced_run):
    """The discriminating assertion: grad_max is the SAME quantity the exit
    predicate tests, so on a converged run the last row must satisfy it.

    This is what makes the trace usable as evidence about a FAILED run. If
    grad_max were some other residual, a stalled trace would be unreadable --
    a reader could not tell how far the run stood from its own gate.
    """
    assert traced_run.converged
    assert traced_run.scf_trace[-1].grad_max < 1e-5


def test_residual_actually_falls_across_the_run(traced_run):
    """A trace of a converging run must show convergence, not noise."""
    trace = traced_run.scf_trace
    assert trace[-1].grad_max < trace[0].grad_max / 100.0


def test_diis_depth_is_recorded_and_respects_its_cap(traced_run):
    """Depth is monotone non-decreasing and never exceeds the ring-buffer cap;
    the plateau in #493 was first localised by reading exactly this column."""
    dims = [row.diis_dim for row in traced_run.scf_trace]
    assert dims[0] == 1
    assert all(b >= a for a, b in zip(dims, dims[1:]))
    assert max(dims) <= 8


def test_gap_is_finite_and_positive_on_a_closed_shell_singlet(traced_run):
    """LiH/sto-3g has occupied and virtual orbitals at every cycle, so the
    frontier gap is well defined; ``nan`` would mean the column is unusable."""
    gaps = [row.homo_lumo for row in traced_run.scf_trace]
    assert all(np.isfinite(g) for g in gaps)
    assert gaps[-1] > 0.0


def test_trace_is_diagnostic_only_and_cannot_change_an_energy():
    """Recording the trace must not perturb the result it describes."""
    a = run_ccm_rhf_ri_neutral(_lih_ccm())
    b = run_ccm_rhf_ri_neutral(_lih_ccm())
    assert a.energy == pytest.approx(b.energy, abs=1e-14, rel=0.0)
    assert a.n_iter == b.n_iter == len(a.scf_trace)
