"""Post-mortem perf log — phase tracking, env var, run_job kwarg.

The companion to v0.5.1's live SCF logging. Live logging shows
progress *during* a run; the perf log shows where the time went
*afterwards*. Pin the contract:

  1. ``perf_log()`` context manager activates a fresh tracker; on
     exit, writes the report to the resolved path (kwarg or
     ``$VIBEQC_PERFLOG`` env var).
  2. ``PerfScope("name")`` accumulates wall + CPU into the active
     tracker; no-op when no tracker is active (so call sites can
     sprinkle scopes unconditionally).
  3. ``run_job(perf_log=True)`` writes ``{output}.perf`` sibling to
     ``.out`` / ``.molden`` / ``.system``; ``perf_log=path`` writes
     to a custom path; ``perf_log=False`` (default) is silent.
  4. ``$VIBEQC_PERFLOG`` env-var wins when ``perf_log`` is left at
     ``None`` — useful for batch scripts.
  5. Every off-by-default code path produces no perf file.
  6. Output is parseable: every documented section appears, and the
     phase summary lists every PerfScope opened during the block.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import vibeqc.output.formats.perf as perf_format
from vibeqc.output.document import DEFAULT_POLICY

from vibeqc import (
    Atom,
    Molecule,
    PerfScope,
    PerfTracker,
    active_tracker,
    format_perf_report,
    perf_log,
    run_job,
)


def _h2o() -> Molecule:
    return Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])


def test_timing_summary_sizes_rule_without_ragged_annotation() -> None:
    text = perf_format._format_timing_summary(
        [("SCF total", 1.5, None), ("SCF avg per iter", 0.5, "(3 iters)")],
        label_width=28,
        body_indent=2,
        policy=DEFAULT_POLICY,
    )
    lines = text.splitlines()
    assert lines[0] == "  Timings (wall clock, seconds)"
    assert len(lines[1]) == len(lines[2])
    assert len(lines[3]) > len(lines[1])


def test_timing_summary_unit_title_and_values_move_together() -> None:
    milliseconds = DEFAULT_POLICY.with_unit("time", "ms")
    text = perf_format._format_timing_summary(
        [("Job total", 1.5, None)],
        label_width=12,
        policy=milliseconds,
    )
    assert "Timings (wall clock, milliseconds)" in text
    assert "1500.000" in text


# ---------------------------------------------------------------------------
# 1. PerfScope basics
# ---------------------------------------------------------------------------

def test_perf_scope_no_op_when_no_tracker_active() -> None:
    """A bare PerfScope outside any perf_log() block must be a
    no-op — call sites should be able to sprinkle scopes
    unconditionally without worrying about whether a tracker is
    active."""
    assert active_tracker() is None
    with PerfScope("nothing"):
        time.sleep(0.001)
    assert active_tracker() is None  # still no leaked state


def test_perf_scope_accumulates_wall_and_cpu(tmp_path: Path) -> None:
    """Inside a perf_log() block, opening a PerfScope adds a row to
    the tracker with positive wall + CPU times."""
    with perf_log() as tracker:
        with PerfScope("compute_thing"):
            # busy-loop so CPU time accumulates
            t0 = time.perf_counter()
            x = 0
            while time.perf_counter() - t0 < 0.005:
                x += 1
        with PerfScope("compute_thing"):
            time.sleep(0.001)

    assert "compute_thing" in tracker.phases
    stat = tracker.phases["compute_thing"]
    assert stat.n_calls == 2
    assert stat.wall_s > 0.0
    assert stat.cpu_s >= 0.0


def test_active_tracker_returns_current_inside_block() -> None:
    """``active_tracker()`` returns the tracker yielded by
    ``perf_log()`` while inside the block, ``None`` outside."""
    assert active_tracker() is None
    with perf_log() as tracker:
        assert active_tracker() is tracker
    assert active_tracker() is None


# ---------------------------------------------------------------------------
# 2. perf_log() output file
# ---------------------------------------------------------------------------

def test_perf_log_writes_report_to_path_kwarg(tmp_path: Path) -> None:
    target = tmp_path / "out.perf"
    with perf_log(target):
        with PerfScope("phase_a"):
            time.sleep(0.001)
        with PerfScope("phase_b"):
            time.sleep(0.001)
    assert target.is_file()
    text = target.read_text()
    assert "vibe-qc performance / debug log" in text
    assert "Phase summary" in text
    assert "phase_a" in text
    assert "phase_b" in text


def test_perf_log_no_path_no_file_written(tmp_path: Path) -> None:
    """``perf_log()`` with no path and no env var still accumulates
    into the yielded tracker (callers can read it programmatically)
    but writes no file."""
    with perf_log() as tracker:
        with PerfScope("x"):
            pass
    assert "x" in tracker.phases
    assert not list(tmp_path.iterdir())  # nothing written


def test_perf_log_env_var_resolves_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``$VIBEQC_PERFLOG`` is honored when ``perf_log()`` is called
    with no explicit path."""
    target = tmp_path / "from_env.perf"
    monkeypatch.setenv("VIBEQC_PERFLOG", str(target))
    with perf_log():
        with PerfScope("from_env_phase"):
            pass
    assert target.is_file()
    assert "from_env_phase" in target.read_text()


def test_perf_log_explicit_path_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_target = tmp_path / "env.perf"
    arg_target = tmp_path / "arg.perf"
    monkeypatch.setenv("VIBEQC_PERFLOG", str(env_target))
    with perf_log(arg_target):
        with PerfScope("p"):
            pass
    assert arg_target.is_file()
    assert not env_target.exists()


def test_perf_log_write_failure_uses_output_warning(tmp_path: Path) -> None:
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("block", encoding="utf-8")
    target = blocking_file / "out.perf"

    with pytest.warns(
        UserWarning,
        match=r"performance_log.*FileExistsError",
    ):
        with perf_log(target):
            pass

    assert not target.exists()


# ---------------------------------------------------------------------------
# 3. run_job integration
# ---------------------------------------------------------------------------

def test_run_job_perf_log_true_writes_sibling(tmp_path: Path) -> None:
    """``run_job(perf_log=True)`` writes ``{output}.perf`` next to
    ``{output}.out`` — same UX as the ``.system`` manifest."""
    stem = tmp_path / "h2o_perf_true"
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem,
            perf_log=True, progress=False)
    perf = stem.with_suffix(".perf")
    assert perf.is_file()
    text = perf.read_text()
    assert "run_job.total" in text
    assert "scf.rhf" in text
    assert "basis_set_construction" in text


def test_run_job_perf_log_explicit_path(tmp_path: Path) -> None:
    stem = tmp_path / "h2o_perf_path"
    custom = tmp_path / "custom_perf.txt"
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem,
            perf_log=custom, progress=False)
    assert custom.is_file()
    # And no sibling .perf next to .out — the explicit path wins.
    assert not stem.with_suffix(".perf").exists()


def test_run_job_perf_log_default_is_off(tmp_path: Path) -> None:
    """Default ``perf_log=None`` and no ``$VIBEQC_PERFLOG`` env var
    produce no perf file — preserving the historical
    pre-v0.5.2 behavior for callers that don't opt in."""
    stem = tmp_path / "h2o_no_perf"
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem,
            progress=False)
    assert not stem.with_suffix(".perf").exists()


def test_run_job_perf_log_env_var_enables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "from_env.perf"
    monkeypatch.setenv("VIBEQC_PERFLOG", str(target))
    stem = tmp_path / "h2o_env_perf"
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem,
            progress=False)
    assert target.is_file()


def test_run_job_perf_log_records_scf_iterations(tmp_path: Path) -> None:
    """The perf report carries a per-iteration SCF table — the
    post-mortem analogue of the live progress trace."""
    stem = tmp_path / "h2o_iter_table"
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem,
            perf_log=True, progress=False)
    text = stem.with_suffix(".perf").read_text()
    assert "SCF iterations" in text
    # The H2O / STO-3G / RHF SCF converges in ~8 iters; we should
    # see at least 5 numbered rows in the table.
    iter_lines = [line for line in text.splitlines()
                  if line.strip().startswith(tuple(str(i) for i in range(1, 16)))]
    assert len(iter_lines) >= 5


def test_scf_trace_iteration_walls_are_measured() -> None:
    """BUG87-A regression: the molecular C++ SCF trace carries MEASURED
    per-iteration walls. The shipped defect: every iteration of a
    2258 s glycine RIJCOSX SCF printed 'wall (s)' as 0.000 because
    SCFIteration had no wall field and the renderer defaulted the
    missing key to 0.0."""
    from vibeqc import BasisSet, run_rhf

    mol = _h2o()
    result = run_rhf(mol, BasisSet(mol, "sto-3g"))
    trace = list(result.scf_trace)
    assert len(trace) >= 3
    # Every iteration carries a strictly positive measured wall, and
    # the sum is a plausible measurement (H2O/STO-3G is far below 60 s).
    assert all(step.wall_s > 0.0 for step in trace)
    assert 0.0 < sum(step.wall_s for step in trace) < 60.0


def test_perf_table_renders_dashes_for_unmeasured_wall() -> None:
    """A trace row WITHOUT a measured wall renders '--' in the wall
    column — the renderer must never invent a 0.000 measurement
    (BUG87-A output-truthfulness fix)."""
    tracker = PerfTracker(threads=1)
    tracker.add_scf_iter(iter=1, energy=-1.0, dE=None, grad=1e-3, diis=1,
                         wall_s=None)
    tracker.add_scf_iter(iter=2, energy=-1.1, dE=-0.1, grad=1e-4, diis=2,
                         wall_s=0.123)
    text = format_perf_report(tracker)
    row1 = next(ln for ln in text.splitlines()
                if ln.strip().startswith("1 ") and "-1.0" in ln)
    row2 = next(ln for ln in text.splitlines()
                if ln.strip().startswith("2 ") and "-1.1" in ln)
    assert row1.rstrip().endswith("--")
    assert row2.rstrip().endswith("0.123")


def test_run_job_perf_log_records_memory_snapshots(
    tmp_path: Path,
) -> None:
    """A ``Memory snapshots`` block shows ``start_of_scf`` and
    ``end_of_scf`` rows so users can see RSS growth caused by the
    Fock build."""
    stem = tmp_path / "h2o_mem"
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem,
            perf_log=True, progress=False)
    text = stem.with_suffix(".perf").read_text()
    assert "Memory snapshots" in text
    assert "start_of_scf" in text
    assert "end_of_scf" in text


# ---------------------------------------------------------------------------
# 4. format_perf_report — pure function on PerfTracker
# ---------------------------------------------------------------------------

def test_format_perf_report_handles_empty_tracker() -> None:
    """Reporting on a freshly-constructed tracker (no scopes opened)
    must produce a well-formed report — no crashes on division by
    zero, no missing sections."""
    tracker = PerfTracker(threads=4)
    text = format_perf_report(tracker)
    assert "Total wall time" in text
    assert "OMP threads:     4" in text
    assert "Phases tracked:  0" in text


def test_format_perf_report_flags_under_parallelised_phases() -> None:
    """A phase consuming > 5% of wall but with par < 0.7 × threads
    gets flagged in the 'Under-parallelised hot paths' section.
    Engineered by feeding a tracker known timing values."""
    tracker = PerfTracker(threads=8)
    # 1.0 s wall, 1.0 s cpu → par = 1.0 / (1.0 × 8) = 0.125 (under)
    tracker.add_scope("slow_serial_phase", wall_s=1.0, cpu_s=1.0)
    # Force the total wall to look like 1.0 s so % wall is 100%.
    tracker.t_start = time.perf_counter() - 1.0
    text = format_perf_report(tracker)
    assert "Under-parallelised hot paths" in text
    assert "slow_serial_phase" in text


def test_detect_threads_caps_host_runtime_by_one_cpu_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-CPU allocation on a 96-core host is a one-thread job.

    Regression for RP209 BUG 41: the old fallback imported the native module
    from the wrong package level, then used ``os.cpu_count()`` and fabricated
    96 threads plus 0.01x parallelism for a genuinely serial allocation.
    """
    monkeypatch.setattr(perf_format, "_native_thread_count", lambda: 96)
    monkeypatch.setattr(perf_format, "_affinity_thread_count", lambda: 96)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("VQ_CPUS", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setattr(os, "cpu_count", lambda: 96)

    threads = perf_format._detect_threads()
    assert threads == 1

    tracker = PerfTracker(threads=threads)
    tracker.add_scope("run_job.total", wall_s=1.0, cpu_s=1.0)
    tracker.t_start = time.perf_counter() - 1.0
    text = format_perf_report(tracker)
    assert "OMP threads:     1" in text
    assert "Under-parallelised hot paths" not in text


def test_detect_threads_never_guesses_from_host_core_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No allocation/runtime signal means conservative one, not host cores."""
    monkeypatch.setattr(perf_format, "_native_thread_count", lambda: None)
    monkeypatch.setattr(perf_format, "_affinity_thread_count", lambda: None)
    for name in perf_format._ALLOCATION_THREAD_ENVS:
        monkeypatch.delenv(name, raising=False)

    def _host_count_must_not_be_read() -> int:
        raise AssertionError("host core count is not a job allocation")

    monkeypatch.setattr(os, "cpu_count", _host_count_must_not_be_read)
    assert perf_format._detect_threads() == 1


def test_detect_threads_does_not_treat_unrestricted_affinity_as_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host-wide affinity mask is availability, not an allocation."""
    monkeypatch.setattr(perf_format, "_native_thread_count", lambda: None)
    monkeypatch.setattr(perf_format, "_affinity_thread_count", lambda: 96)
    for name in perf_format._ALLOCATION_THREAD_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert perf_format._detect_threads() == 1


# ---------------------------------------------------------------------------
# 7. JSON perf artefact — a ``.json`` target must hold JSON, not prose.
#
# Reproducer for PERF-JSON-PLAIN-TEXT: campaign payloads pass
# ``perf_log="<stem>.perf.json"``; every one of 545 converged cases
# wrote the plain-text report under that name, so ``json.loads()``
# raised on 545/545 and no automated performance harvest was possible.
# The writer now dispatches on the target suffix.
# ---------------------------------------------------------------------------

def test_perf_log_json_suffix_writes_parseable_json(tmp_path: Path) -> None:
    target = tmp_path / "out.perf.json"
    with perf_log(target):
        with PerfScope("phase_a"):
            time.sleep(0.001)
    assert target.is_file()
    payload = json.loads(target.read_text())
    assert payload["schema"] == perf_format.PERF_JSON_SCHEMA
    assert payload["threads"] >= 1
    assert payload["total_wall_s"] > 0.0
    names = [phase["name"] for phase in payload["phases"]]
    assert "phase_a" in names


def test_perf_log_text_suffix_still_writes_the_text_report(
    tmp_path: Path,
) -> None:
    """The default ``.perf`` artefact is unchanged — no silent format
    swap for callers that never asked for JSON."""
    target = tmp_path / "out.perf"
    with perf_log(target):
        with PerfScope("phase_a"):
            time.sleep(0.001)
    text = target.read_text()
    assert text.startswith("=")
    assert "vibe-qc performance / debug log" in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_run_job_perf_log_json_path_parses_as_json(tmp_path: Path) -> None:
    """End-to-end: the exact campaign call shape that produced the
    545 unparseable artefacts."""
    stem = tmp_path / "h2o_perf_json"
    target = Path(str(stem) + ".perf.json")
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem,
            perf_log=str(target), progress=False)
    assert target.is_file()
    payload = json.loads(target.read_text())
    names = [phase["name"] for phase in payload["phases"]]
    assert "run_job.total" in names
    assert "scf.rhf" in names
    assert payload["scf_iters"], "SCF iteration rows must survive into JSON"


def test_large_molden_export_runs_after_compact_artefacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout in bulk Molden output must not discard settled results.

    Issue #28 captured a converged 2,152-function UKS calculation whose
    carrier timeout fired while the 278 MB Molden file was being written.
    The old ordering entered that writer before population, citations, XYZ,
    QVF, or the performance context had settled, so all of those cheaper
    artefacts vanished with it.  Raise a ``BaseException`` at the Molden
    boundary to model the carrier's uncatchable process kill and pin that
    every compact result has already reached disk at that point.
    """
    from vibeqc.output.formats import molden as molden_format

    class SimulatedCarrierTimeout(BaseException):
        pass

    stem = tmp_path / "h2_timeout"
    perf_path = Path(str(stem) + ".perf.json")
    expected = (
        stem.with_suffix(".out"),
        perf_path,
        stem.parent / f"{stem.name}.population.txt",
        stem.parent / f"{stem.name}.population.json",
        stem.with_suffix(".xyz"),
        stem.with_suffix(".bibtex"),
        stem.with_suffix(".references"),
        stem.with_suffix(".qvf"),
    )

    def interrupt_molden(*args: object, **kwargs: object) -> None:
        missing = [path.name for path in expected if not path.is_file()]
        if missing:
            pytest.fail(
                "Molden export started before compact artefacts settled: "
                + ", ".join(missing)
            )
        import tomllib

        manifest = tomllib.loads(stem.with_suffix(".system").read_text())
        outcomes = {
            Path(row["path"]).name: bool(row["written"])
            for row in manifest["outputs"]["files"]
        }
        uncommitted = [path.name for path in expected if not outcomes[path.name]]
        if uncommitted:
            pytest.fail(
                "Molden export started before compact artefacts were recorded: "
                + ", ".join(uncommitted)
            )
        raise SimulatedCarrierTimeout

    monkeypatch.setattr(molden_format, "write_molden", interrupt_molden)
    molecule = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    )

    with pytest.raises(SimulatedCarrierTimeout):
        run_job(
            molecule,
            basis="sto-3g",
            method="rhf",
            output=stem,
            perf_log=perf_path,
            output_qvf=True,
            citations=True,
            write_molden_file=True,
            write_population_file=True,
            progress=False,
            verbose=0,
        )


def test_run_job_perf_json_manifest_declares_json_format(
    tmp_path: Path,
) -> None:
    """The ``.system`` manifest must not describe the JSON artefact as
    ``format = "text"`` — the manifest is the machine-readable index a
    harvest reads before opening the file."""
    stem = tmp_path / "h2o_perf_manifest"
    target = Path(str(stem) + ".perf.json")
    run_job(_h2o(), basis="sto-3g", method="rhf", output=stem,
            perf_log=str(target), progress=False)
    manifest = stem.with_suffix(".system").read_text()
    perf_block = [
        block for block in manifest.split("[[plan.files]]")
        if 'role          = "perf"' in block
    ]
    assert perf_block, "manifest carries no perf plan entry"
    assert 'format        = "json"' in perf_block[0]


def test_perf_json_report_is_serialisable_for_empty_tracker() -> None:
    payload = json.loads(perf_format.format_perf_json(PerfTracker()))
    assert payload["phases"] == []
    assert payload["scf_iters"] == []
    assert payload["memory_snapshots"] == []
