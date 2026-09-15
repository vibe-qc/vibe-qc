"""The BIPOLE post-SCF final-density phase must be observable (#115).

Every BIPOLE driver evaluates the committed density with a **non-incremental**
Fock build, reusing the in-loop confirmation when available. This build screens
the full density, so it can cost much more than a late incremental update:
measured at ~22,000 s on a six-basis-function LiH/STO-3G (2,2,2) cell whose
final iterations cost 819 s.

Before this phase announced itself the failure mode was total. A 24 h canary
(validation-job) converged at wall 67,593 s and then spent **>= 5.22 h
emitting nothing** before the wall killed it -- ``batch-results.json`` with
``n_cases_done 1``, ``.system`` still ``status='running'``, no rc, and a
converged energy that survived only because someone read it out of a truncated
``.out``. A phase that emits nothing is also a phase nobody can bound, and
from outside it is indistinguishable from a hang.

These tests pin the two properties that make it attributable:

* the phase announces itself in the ordinary progress stream **before** it
  starts, and reports its own wall time when it ends;
* the loop's energy reaches the structured log **before** the expensive work
  begins, so a wall-kill inside finalisation can no longer destroy a
  converged result.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    PeriodicRHFOptions,
    monkhorst_pack,
)
from vibeqc.progress import ProgressLogger
from vibeqc.structured_log import structured_log

BEGIN = "scf_final_density_begin"
END = "scf_final_density_end"


CONV_TOL_ENERGY = 1e-8


def _run(progress, max_iter: int = 40, use_diis: bool = True, driver: str = "rhf"):
    import importlib
    from vibeqc._vibeqc_core import PeriodicKSOptions

    lattice = np.eye(3) * 7.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions() if driver in ("rks", "uks") else PeriodicRHFOptions()
    if driver in ("rks", "uks"):
        opts.functional = "pbe0"
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 5.0
    opts.max_iter = max_iter
    opts.conv_tol_energy = CONV_TOL_ENERGY
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = use_diis
    module = importlib.import_module("vibeqc.pbc_bipole" + ("" if driver == "rhf" else "_" + driver))
    return getattr(module, "run_pbc_bipole_" + driver)(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=progress,
    )


def test_final_density_phase_announces_itself_in_the_progress_stream():
    """Behavioural: the phase is named before it runs and timed when it ends.

    Asserted on the emitted progress text of a real run, so at the parent
    commit this fails because a real run says nothing between the SCF table
    and the converged line -- not because a symbol is missing (LEARNINGS
    L124).
    """
    buf = io.StringIO()
    _run(ProgressLogger(stream=buf, verbose=4))
    text = buf.getvalue()
    assert "final density evaluation" in text, (
        "the post-SCF final-density phase emits nothing; a >= 5.22 h silence "
        "here is indistinguishable from a hang (#115).\n" + text
    )
    start = text.index("final density evaluation")
    assert "done (" in text[start:], (
        "the phase announces itself but never reports completion or its "
        "wall time, so its cost stays unbounded from outside.\n" + text
    )
    # The announcement must precede the SCF verdict: a reader who sees the
    # loop finish and then silence must already know what is running.
    assert text.index("SCF loop finished at iteration") < text.index(
        "final density evaluation"
    )


def test_converged_energy_is_durable_before_finalisation_begins(tmp_path: Path):
    """The loop's energy reaches the structured log BEFORE the rebuild.

    This is the half that turns a wall-kill inside finalisation from a lost
    result into a recorded one.
    """
    log_path = tmp_path / "run.jsonl"
    with structured_log(log_path, enabled=True):
        result = _run(False)
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    names = [e.get("event") for e in events]
    assert BEGIN in names, (
        "no structured event marks the start of the post-SCF phase, so a "
        f"job killed inside it leaves no record of the SCF result. Saw: "
        f"{sorted(set(names))}"
    )
    assert END in names
    assert names.index(BEGIN) < names.index(END)
    begin = events[names.index(BEGIN)]
    # The energy recorded before the rebuild is the loop's own converged
    # value; the rebuild re-evaluates the SAME fixed point, so on a converged
    # run the two must agree to the SCF's own energy threshold -- not merely
    # be present. (On a non-converged run they legitimately differ, which is
    # exactly why the rebuild is mandatory; that branch is covered below.)
    assert result.converged
    assert begin["energy"] == pytest.approx(
        result.energy, abs=10.0 * CONV_TOL_ENERGY
    )
    assert begin["n_iter"] == result.n_iter
    assert begin["converged"] is bool(result.converged)
    # Cost attribution: the phase reports its own wall time, so the additive
    # term is measurable from the log alone.
    assert events[names.index(END)]["wall_s"] >= 0.0


def test_phase_is_recorded_even_when_the_scf_did_not_converge():
    """Max-iteration exhaustion still runs the rebuild, so it still reports.

    The rebuild is mandatory on both branches -- the loop commits a new
    density after evaluating the previous one either way -- so a
    non-converged run must not be the silent case.
    """
    buf = io.StringIO()
    result = _run(ProgressLogger(stream=buf, verbose=4), max_iter=1,
                  use_diis=False)
    assert not result.converged
    text = buf.getvalue()
    assert "NOT converged" in text
    assert "final density evaluation" in text


# ---------------------------------------------------------------------------
# #115 verifier findings (2026-09-02): on a converged run the expensive cold
# rebuild is the in-loop exact confirmation (#514, the #116 terminal check),
# which ran BEFORE the announced phase; and a raising rebuild was reported as
# a successful end from a bare ``finally``. These tests are ordering-aware.
# ---------------------------------------------------------------------------

CONFIRM_BEGIN = "scf_exact_confirmation_begin"
CONFIRM_END = "scf_exact_confirmation_end"


def _spy_cold_builds(monkeypatch, buf, on_cold=None, driver="rhf"):
    """Record, at every non-incremental build, whether the announcement was
    already in the progress stream; optionally raise from the first one."""
    import importlib

    module = importlib.import_module("vibeqc.pbc_bipole" + ("" if driver == "rhf" else "_" + driver))
    builder = "build_bipole_" + ("unrestricted" if driver in ("uhf", "uks") else "restricted") + "_fock"
    original = getattr(module, builder)
    seen = []

    def spy(ctx, *densities, **kwargs):
        if not kwargs.get("use_incremental", True):
            seen.append(
                "exact-operator confirmation of the provisional convergence"
                in buf.getvalue()
            )
            if on_cold is not None:
                on_cold()
        return original(ctx, *densities, **kwargs)

    monkeypatch.setattr(module, builder, spy)
    return seen


def test_cold_rebuild_of_a_converged_run_is_announced_before_it_runs(
    monkeypatch, tmp_path: Path
):
    """Every non-incremental build of a converged run is preceded by its
    announcement, and the durable begin record precedes the final phase."""
    buf = io.StringIO()
    seen = _spy_cold_builds(monkeypatch, buf)
    log_path = tmp_path / "order.jsonl"
    with structured_log(log_path, enabled=True):
        result = _run(ProgressLogger(stream=buf, verbose=4))
    assert result.converged
    assert seen and all(seen), (
        "a cold exact rebuild ran before the phase announced it, so a "
        f"wall-kill there is unattributable (announced-before flags: {seen})"
    )
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    names = [e.get("event") for e in events]
    assert CONFIRM_BEGIN in names and CONFIRM_END in names
    assert names.index(CONFIRM_BEGIN) < names.index(CONFIRM_END) < names.index(BEGIN)
    confirm = events[names.index(CONFIRM_BEGIN)]
    # The provisional state is durable before the expensive work: the
    # loop's own energy at the iteration that met the gate.
    assert confirm["converged"] is True
    assert confirm["reused"] is False
    assert confirm["energy"] == pytest.approx(result.energy, abs=10.0 * CONV_TOL_ENERGY)
    assert events[names.index(CONFIRM_END)]["status"] == "done"
    assert events[names.index(CONFIRM_END)]["wall_s"] >= 0.0
    # The post-loop phase reused that build and says so: no second cold build.
    final_begin = events[names.index(BEGIN)]
    final_end = events[names.index(END)]
    assert final_begin["reused"] is True
    assert final_end["status"] == "reused"
    assert seen.count(True) == len(seen) == 1
    text = buf.getvalue()
    assert "reusing the exact confirmation rebuild" in text
    assert text.index("exact-operator confirmation") < text.index(
        "final density evaluation"
    )


@pytest.mark.parametrize("driver", ["rhf", "uhf", "rks", "uks"])
def test_provisional_energy_reaches_out_before_cold_build(
    monkeypatch, tmp_path: Path, driver: str
):
    """The ordinary output survives interruption without an optional log."""
    import re
    from vibeqc.output.channel import OutputChannel

    out_path = tmp_path / "run.out"
    snapshots = []
    _spy_cold_builds(
        monkeypatch, io.StringIO(),
        on_cold=lambda: snapshots.append(out_path.read_text()), driver=driver,
    )
    with OutputChannel.to_file(out_path):
        result = _run(False, driver=driver)
    assert result.converged
    assert len(snapshots) == 1
    before = snapshots[0]
    assert "Exact density evaluation pending (Ha)" in before
    match = re.search(r"Provisional SCF energy\s+(-?\d+\.\d+)", before)
    assert match is not None, before
    assert float(match[1]) == pytest.approx(result.energy, abs=10 * CONV_TOL_ENERGY)
    assert before.endswith("\n"), "The durable phase record must end its output line"
    assert "Exact density evaluation done" not in before
    after = out_path.read_text()
    assert "Exact density evaluation done (s)" in after
    assert after.endswith("\n")
    assert re.search(r"Build[^\n]*cold full density\n", after)
    assert not re.search(r"Wall time[^\n]*iter ", after)


def test_a_raising_rebuild_is_reported_as_raised_not_done(monkeypatch, tmp_path: Path):
    """A rebuild that throws ends its phase with ``status="raised"`` and the
    exception propagates; nothing says "done"."""
    from vibeqc.output.channel import OutputChannel

    buf = io.StringIO()
    out_path = tmp_path / "raised.out"

    def boom():
        raise RuntimeError("simulated integral failure inside the cold rebuild")

    _spy_cold_builds(monkeypatch, buf, on_cold=boom)
    log_path = tmp_path / "raised.jsonl"
    with (
        structured_log(log_path, enabled=True),
        OutputChannel.to_file(out_path),
        pytest.raises(RuntimeError, match="simulated integral failure"),
    ):
        _run(ProgressLogger(stream=buf, verbose=4))
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    names = [e.get("event") for e in events]
    assert CONFIRM_BEGIN in names
    end = events[names.index(CONFIRM_END)]
    assert end["status"] == "raised"
    assert "simulated integral failure" in end["error"]
    text = buf.getvalue()
    assert "FAILED after" in text
    assert "confirmation of the provisional convergence at iteration" in text
    assert "done (" not in text[text.index("exact-operator confirmation"):]
    # The loop's provisional energy survived in the begin record.
    assert events[names.index(CONFIRM_BEGIN)]["energy"] < 0.0
    output = out_path.read_text()
    assert "Provisional SCF energy" in output
    assert "Exact density evaluation raised" in output
    assert "Exact density evaluation done" not in output


def test_max_iteration_exit_pays_exactly_one_cold_build_in_the_final_phase(
    monkeypatch, tmp_path: Path
):
    buf = io.StringIO()
    seen = _spy_cold_builds(monkeypatch, buf)
    log_path = tmp_path / "capped.jsonl"
    with structured_log(log_path, enabled=True):
        result = _run(ProgressLogger(stream=buf, verbose=4), max_iter=1, use_diis=False)
    assert not result.converged
    assert len(seen) == 1  # the mandatory post-loop rebuild, nothing else
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    names = [e.get("event") for e in events]
    assert CONFIRM_BEGIN not in names
    assert events[names.index(BEGIN)]["reused"] is False
    assert events[names.index(END)]["status"] == "done"


@pytest.mark.parametrize("driver", ["rhf", "rks", "uhf", "uks"])
def test_last_iteration_refusal_reuses_the_exact_operator(monkeypatch, tmp_path, driver):
    """A refused confirmation at the cap is still the exact final-density build."""
    import importlib

    baseline = _run(False, driver=driver)
    assert baseline.converged
    module = importlib.import_module("vibeqc.pbc_bipole" + ("" if driver == "rhf" else "_" + driver))
    original = module.bipole_terminal_check
    exact_checks = []

    def refuse_in_loop(*args, **kwargs):
        exact_checks.append((kwargs["phase"], kwargs["exact_objective"]))
        verdict, details = original(*args, **kwargs)
        return (False if kwargs["phase"] == "in_loop" else verdict), details

    monkeypatch.setattr(module, "bipole_terminal_check", refuse_in_loop)
    buf = io.StringIO()
    seen = _spy_cold_builds(monkeypatch, buf, driver=driver)
    log_path = tmp_path / "refused.jsonl"
    with structured_log(log_path, enabled=True):
        result = _run(ProgressLogger(stream=buf, verbose=4), max_iter=baseline.n_iter, driver=driver)
    assert not result.converged
    # Independent Ewald reductions can differ by a final rounding bit.
    assert result.energy == pytest.approx(baseline.energy, rel=0., abs=1e-12)
    # Within this run the exact operator and its energy must be reused exactly.
    assert exact_checks == [("in_loop", result.energy), ("post_loop", result.energy)]
    assert len(seen) == 1
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    final_begin = next(e for e in events if e.get("event") == BEGIN)
    assert final_begin["reused"] is True
    assert final_begin["converged"] is False
