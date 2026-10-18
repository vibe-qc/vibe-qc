"""Tests for :mod:`vibeqc.progress` — the live-progress logger that
defeats output buffering during long SCF runs.

Coverage:

1. Stand-alone formatter: ``ProgressLogger`` writes flushed plain-ASCII
   lines for stage/iteration/info, even when the stream is not a TTY.
2. ``progress=False`` and ``progress=None`` are silent.
3. Native molecular RHF/UHF/RKS/UKS iterations reach the terminal,
   structured log, and manifest while the compiled call is still active.
4. Callback replay, attempt selection, nested contexts, failures, helper
   SCFs, and staged RIJCOSX renumbering preserve exact-once delivery.
5. ``progress=True`` against a Γ-only periodic Ewald RHF run produces a
   per-iteration trace whose iteration count matches ``result.n_iter``.
6. The log-file tee writes lines as they happen — verified by reopening
   the file mid-run.

The Γ-only test uses H₂ / STO-3G in a 30-bohr box, the same minimal
cell the wider periodic-Ewald suite (``test_periodic_rhf_ewald.py``)
uses; that keeps the SCF cheap (~0.5 s) while exercising the actual
Python SCF loop where progress is wired in.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.runner as runner_module
from vibeqc.progress import ProgressLogger, resolve_progress


# ---------------------------------------------------------------------------
# 1. Stand-alone formatter
# ---------------------------------------------------------------------------


def test_progress_logger_no_tty_writes_flushed_plain_text() -> None:
    """A non-TTY stream must still receive plain-text output, flushed
    after each call. We check both the line content and that the
    stream was actually flushed by reading immediately after each
    write (StringIO is always synchronously visible — the flush is
    cheap insurance against future swap-ins of a real file)."""
    buf = io.StringIO()
    plog = ProgressLogger(stream=buf, verbose=True)

    plog.info("hello world")
    after_info = buf.getvalue()
    assert "hello world" in after_info
    assert after_info.endswith("\n")

    with plog.stage("integrals", detail="cutoff 12.0 bohr"):
        time.sleep(0.001)
    after_stage = buf.getvalue()
    assert "[integrals]" in after_stage
    assert "cutoff 12.0 bohr" in after_stage
    assert "[integrals] done" in after_stage

    plog.iteration(1, energy=-1.0, dE=0.0, grad=1e-3, diis=0)
    plog.iteration(2, energy=-1.001, dE=-1e-3, grad=1e-4, diis=2)
    out = buf.getvalue()
    assert "iter    1" in out
    assert "iter    2" in out
    assert re.search(r"E\s*=\s*-1\.\d+\s*Ha", out)
    # The dE column reads "--" on iter 1 and a signed scientific number on iter 2.
    assert re.search(r"iter\s+1.*dE\s*=\s*--", out)
    assert re.search(r"iter\s+2.*dE\s*=\s*-1\.\d+e-\d+", out)

    plog.converged(n_iter=2, energy=-1.001, converged=True)
    final = buf.getvalue()
    assert "SCF converged in 2 iterations" in final


# ---------------------------------------------------------------------------
# 2. progress=False / None is silent
# ---------------------------------------------------------------------------


def test_progress_off_when_disabled_produces_zero_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``resolve_progress(False)`` and ``resolve_progress(None)`` must
    both return a no-op logger that prints nothing on any method."""
    for arg in (False, None):
        plog = resolve_progress(arg)
        plog.info("not visible")
        plog.warn("not visible")
        plog.banner("not visible")
        with plog.stage("not visible"):
            pass
        plog.iteration(1, energy=0.0, dE=0.0, grad=0.0, diis=0)
        plog.converged(n_iter=1, energy=0.0, converged=True)
    assert capsys.readouterr().out == ""


def test_resolve_progress_falsy_values_are_silently_off() -> None:
    # 0 / "" / [] read as "off" and must not warn -- only an unexpected
    # *truthy* object is a probable mistake.
    import warnings

    for arg in (0, "", [], 0.0):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            plog = resolve_progress(arg)
        assert not plog.enabled


def test_resolve_progress_warns_on_an_unexpected_truthy_object() -> None:
    # An object with .write but no .iteration -- e.g. an OutputChannel or a
    # stray stream -- is not a progress sink. Silently nulling it dropped
    # all progress with no signal; it must warn now.
    class _StreamLike:
        def write(self, s):  # has write, lacks iteration
            return len(s)

    with pytest.warns(RuntimeWarning, match="not a bool, None, ProgressLogger"):
        plog = resolve_progress(_StreamLike())
    assert not plog.enabled


def test_resolve_progress_does_not_warn_on_a_duck_typed_iteration_sink() -> None:
    # The checkpoint-proxy shape: has a callable iteration -> pass through,
    # no warning.
    import warnings

    class _Proxy:
        def iteration(self, n, **kw):
            pass

    proxy = _Proxy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_progress(proxy) is proxy


# ---------------------------------------------------------------------------
# 3. End-to-end: molecular C++ SCF progress is live and exact-once
# ---------------------------------------------------------------------------


def _h2_molecule() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, -0.7]),
            vq.Atom(1, [0.0, 0.0, 0.7]),
        ]
    )


def _molecular_method_options(method: str) -> dict[str, object]:
    """Keep unrestricted progress tests out of the Davidson phase.

    Stability analysis has its own timing/observability contract (issue
    #205).  The four-method callback test isolates the SCF loop; a separate
    test below models the corrective SCF that a stability follow can launch.
    """
    if method == "uhf":
        options = vq.UHFOptions()
        options.stability_check = False
        return {"uhf_options": options}
    if method == "uks":
        options = vq.UKSOptions()
        options.stability_check = False
        return {"uks_options": options}
    return {}


def _molecular_job_kwargs(method: str) -> dict[str, object]:
    return {
        "citations": False,
        "crash_dump": False,
        "output_qvf": False,
        "record_hostname": False,
        "write_molden_file": False,
        "write_population_file": False,
        "write_xyz_file": False,
        **_molecular_method_options(method),
    }


def _structured_records(output: Path) -> list[dict[str, object]]:
    path = output.with_suffix(".scf.jsonl")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _scf_rows(output: Path) -> list[dict[str, object]]:
    return [
        row
        for row in _structured_records(output)
        if row.get("event") == "scf_iter"
    ]


@pytest.mark.parametrize(
    ("method", "functional", "native_entrypoint", "sink_kind"),
    [
        pytest.param("rhf", None, "run_rhf", "visible", id="rhf"),
        pytest.param("uhf", None, "run_uhf", "visible", id="uhf"),
        pytest.param("rks", "PBE", "run_rks", "visible", id="rks"),
        pytest.param("uks", "PBE", "run_uks", "visible", id="uks"),
        pytest.param("rhf", None, "run_rhf", "silent", id="rhf-silent"),
        pytest.param("rhf", None, "run_rhf", "duck", id="rhf-duck"),
    ],
)
def test_molecular_scf_iterations_are_durable_inside_native_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    method: str,
    functional: str | None,
    native_entrypoint: str,
    sink_kind: str,
) -> None:
    """Every native row is readable immediately and equals the final trace."""
    molecule = _h2_molecule()
    output = tmp_path / f"{method}-{sink_kind}"
    structured_path = output.with_suffix(".scf.jsonl")
    manifest_path = output.with_suffix(".system")
    original_native = getattr(runner_module, native_entrypoint)
    original_emit = runner_module.StructuredLog.emit
    original_update = runner_module.OutputWriter.update_progress
    in_native_call = False
    structured_observations: list[dict[str, object]] = []
    manifest_observations: list[dict[str, object]] = []
    duck_iterations: list[tuple[int, dict[str, object], bool]] = []

    class _LiveTextBuffer(io.StringIO):
        """Record the buffer only when a newly flushed iteration is visible."""

        def __init__(self) -> None:
            super().__init__()
            self.iteration_snapshots: list[tuple[bool, str]] = []
            self._n_iterations = 0

        def flush(self) -> None:
            super().flush()
            text = self.getvalue()
            n_iterations = len(re.findall(r"iter\s+\d+\s+E\s*=", text))
            if n_iterations > self._n_iterations:
                self.iteration_snapshots.append((in_native_call, text))
                self._n_iterations = n_iterations

    progress_buffer = _LiveTextBuffer()

    class _ConsumingProgress:
        """A valid duck sink may consume rows instead of rendering them."""

        def __init__(self, inner: ProgressLogger) -> None:
            self.inner = inner

        def iteration(self, n: int, **fields: object) -> None:
            duck_iterations.append((n, dict(fields), in_native_call))

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

    if sink_kind == "visible":
        progress_sink: object = ProgressLogger(
            stream=progress_buffer,
            verbose=4,
        )
    elif sink_kind == "silent":
        progress_sink = False
    else:
        progress_sink = _ConsumingProgress(
            ProgressLogger(stream=progress_buffer, verbose=4)
        )

    def observed_emit(self, event: str, **payload: object) -> None:
        enabled = self.enabled
        original_emit(self, event, **payload)
        if event != "scf_iter" or not enabled:
            return
        assert in_native_call, "scf_iter was deferred until native return"
        records = [
            json.loads(line)
            for line in structured_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        assert records[-1]["event"] == "scf_iter"
        structured_observations.append(records[-1])

    def observed_update(self, **fields: object) -> None:
        original_update(self, **fields)
        if (
            not in_native_call
            or "attempt" not in fields
            or "iteration" not in fields
        ):
            return
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        progress_fields = dict(manifest["progress"])
        assert int(progress_fields["iteration"]) == int(fields["iteration"])
        manifest_observations.append(progress_fields)

    def observed_native(*args, **kwargs):
        nonlocal in_native_call
        in_native_call = True
        try:
            return original_native(*args, **kwargs)
        finally:
            in_native_call = False

    monkeypatch.setattr(runner_module.StructuredLog, "emit", observed_emit)
    monkeypatch.setattr(
        runner_module.OutputWriter,
        "update_progress",
        observed_update,
    )
    monkeypatch.setattr(runner_module, native_entrypoint, observed_native)

    result = runner_module.run_job(
        molecule,
        basis="sto-3g",
        method=method,
        functional=functional,
        grid_level="coarse",
        output=output,
        structured_log=True,
        progress=progress_sink,
        **_molecular_job_kwargs(method),
    )

    canonical_trace = list(result.scf_trace)
    expected_iterations = [int(step.iter) for step in canonical_trace]
    assert len(structured_observations) == len(canonical_trace)
    assert len(manifest_observations) == len(canonical_trace)
    assert [int(row["iter"]) for row in structured_observations] == (
        expected_iterations
    )

    if sink_kind == "visible":
        assert len(progress_buffer.iteration_snapshots) == len(canonical_trace)
        assert all(live for live, _ in progress_buffer.iteration_snapshots)
        text_iterations = re.findall(
            r"iter\s+(\d+)\s+E\s*=",
            progress_buffer.getvalue(),
        )
        assert [int(value) for value in text_iterations] == expected_iterations
    elif sink_kind == "silent":
        assert progress_buffer.getvalue() == ""
        assert not re.search(r"iter\s+\d+\s+E\s*=", capsys.readouterr().out)
    else:
        assert [n for n, _, _ in duck_iterations] == expected_iterations
        assert all(live for _, _, live in duck_iterations)
        assert not re.search(
            r"iter\s+\d+\s+E\s*=",
            progress_buffer.getvalue(),
        )

    final_rows = _scf_rows(output)
    assert len(final_rows) == len(canonical_trace)
    for row, manifest_row, step in zip(
        final_rows,
        manifest_observations,
        canonical_trace,
    ):
        assert int(row["iter"]) == int(step.iter)
        assert float(row["energy"]) == float(step.energy)
        if int(step.iter) == 1:
            assert row["dE"] is None
        else:
            assert float(row["dE"]) == float(step.delta_e)
        assert float(row["grad_norm"]) == float(step.grad_norm)
        assert int(row["diis_subspace"]) == int(step.diis_subspace)
        assert row["phase"] == "scf"
        assert int(row["attempt"]) == 1

        assert manifest_row["phase"] == "scf"
        assert int(manifest_row["iteration"]) == int(step.iter)
        assert int(manifest_row["attempt"]) == 1
        assert float(manifest_row["energy_eh"]) == float(step.energy)
        assert float(manifest_row["gradient_norm"]) == float(step.grad_norm)
        assert int(manifest_row["diis_subspace"]) == int(
            step.diis_subspace
        )

    # The scoped dispatcher must be inert once run_job returns.
    from vibeqc.output._cpp_diagnostics import _forward_progress

    structured_before = structured_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    terminal_before = progress_buffer.getvalue()
    _forward_progress("iter=999 energy=-999 dE=0 grad=0 diis=0")
    assert structured_path.read_bytes() == structured_before
    assert manifest_path.read_bytes() == manifest_before
    assert progress_buffer.getvalue() == terminal_before


@pytest.mark.parametrize("failed_sink", ["structured", "manifest", "text"])
def test_molecular_progress_sinks_fail_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_sink: str,
) -> None:
    """One optional progress destination cannot suppress either sibling."""
    molecule = _h2_molecule()
    output = tmp_path / f"failure-{failed_sink}"
    original_emit = runner_module.StructuredLog.emit
    original_update = runner_module.OutputWriter.update_progress
    manifest_calls: list[dict[str, object]] = []
    terminal_calls: list[int] = []
    progress_buffer = io.StringIO()

    def isolated_emit(self, event: str, **payload: object) -> None:
        if failed_sink == "structured" and event == "scf_iter":
            raise OSError("synthetic structured-log failure")
        original_emit(self, event, **payload)

    def isolated_update(self, **fields: object) -> None:
        if "attempt" in fields and "selected_attempt" not in fields:
            manifest_calls.append(dict(fields))
            if failed_sink == "manifest":
                raise OSError("synthetic manifest rewrite failure")
        original_update(self, **fields)

    class _ObservedProgress:
        def __init__(self) -> None:
            self.inner = ProgressLogger(stream=progress_buffer, verbose=4)

        def iteration(self, n: int, **fields: object) -> None:
            terminal_calls.append(n)
            if failed_sink == "text":
                raise OSError("synthetic terminal failure")
            self.inner.iteration(n, **fields)

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

    monkeypatch.setattr(runner_module.StructuredLog, "emit", isolated_emit)
    monkeypatch.setattr(
        runner_module.OutputWriter,
        "update_progress",
        isolated_update,
    )
    result = runner_module.run_job(
        molecule,
        basis="sto-3g",
        method="rhf",
        output=output,
        structured_log=True,
        progress=_ObservedProgress(),
        **_molecular_job_kwargs("rhf"),
    )

    expected = list(range(1, int(result.n_iter) + 1))
    assert terminal_calls == expected
    assert [int(row["iteration"]) for row in manifest_calls] == expected
    rows = _scf_rows(output)
    if failed_sink == "structured":
        assert rows == []
    else:
        assert [int(row["iter"]) for row in rows] == expected
    text_iterations = re.findall(
        r"iter\s+(\d+)\s+E\s*=",
        progress_buffer.getvalue(),
    )
    if failed_sink == "text":
        assert text_iterations == []
    else:
        assert [int(value) for value in text_iterations] == expected

    # The failed live manifest path is retried by the terminal headline write.
    manifest = tomllib.loads(
        output.with_suffix(".system").read_text(encoding="utf-8")
    )
    assert manifest["progress"]["phase"] == "scf"
    assert int(manifest["progress"]["iteration"]) == int(result.n_iter)


def test_molecular_progress_replays_driver_without_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A producer with only a terminal trace still gets one NDJSON replay."""
    from vibeqc.output._cpp_diagnostics import install_progress_handler

    molecule = _h2_molecule()
    output = tmp_path / "callback-free"
    original = runner_module.run_rhf

    def callback_free_native(*args, **kwargs):
        install_progress_handler(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "run_rhf", callback_free_native)
    result = runner_module.run_job(
        molecule,
        basis="sto-3g",
        method="rhf",
        output=output,
        structured_log=True,
        progress=False,
        **_molecular_job_kwargs("rhf"),
    )

    rows = _scf_rows(output)
    assert len(rows) == int(result.n_iter)
    assert [int(row["iter"]) for row in rows] == list(
        range(1, int(result.n_iter) + 1)
    )
    assert all(row["phase"] == "scf" for row in rows)
    assert all("attempt" not in row for row in rows)
    for row, step in zip(rows, result.scf_trace):
        assert float(row["energy"]) == float(step.energy)
        assert float(row["grad_norm"]) == float(step.grad_norm)


def test_partial_native_callback_replays_only_missing_canonical_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost callback cannot make an already-durable prefix appear twice."""
    from vibeqc.output._cpp_diagnostics import (
        _forward_progress,
        install_progress_handler,
    )

    molecule = _h2_molecule()
    basis = vq.BasisSet(molecule, "sto-3g")
    install_progress_handler(None)
    canonical = runner_module.run_rhf(molecule, basis, vq.RHFOptions())
    canonical_trace = list(canonical.scf_trace)
    assert len(canonical_trace) >= 2

    def partially_observable_native(*args, **kwargs):
        first = canonical_trace[0]
        _forward_progress(
            f"iter={int(first.iter)} "
            f"energy={float(first.energy):.17g} "
            f"dE={float(first.delta_e):.17g} "
            f"grad={float(first.grad_norm):.17g} "
            f"diis={int(first.diis_subspace)}"
        )
        return canonical

    monkeypatch.setattr(
        runner_module,
        "run_rhf",
        partially_observable_native,
    )
    output = tmp_path / "partial-callback"
    result = runner_module.run_job(
        molecule,
        basis="sto-3g",
        method="rhf",
        output=output,
        structured_log=True,
        progress=False,
        **_molecular_job_kwargs("rhf"),
    )

    rows = _scf_rows(output)
    assert len(rows) == len(canonical_trace) == int(result.n_iter)
    assert [int(row["iter"]) for row in rows] == [
        int(step.iter) for step in canonical_trace
    ]
    assert [float(row["energy"]) for row in rows] == [
        float(step.energy) for step in canonical_trace
    ]
    assert int(rows[0]["attempt"]) == 1


@pytest.mark.parametrize("structured_enabled", [True, False])
def test_molecular_progress_scope_cleans_up_after_native_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    structured_enabled: bool,
) -> None:
    """A failed native call retains completed rows but leaks no handler."""
    from vibeqc.crash_dump import load_dump
    from vibeqc.output._cpp_diagnostics import _forward_progress

    # Exercise the explicit flag independently of queue/env telemetry defaults.
    monkeypatch.delenv("VQ_WORKDIR", raising=False)
    monkeypatch.delenv("VIBEQC_STRUCTURED_LOG", raising=False)
    molecule = _h2_molecule()
    output = tmp_path / f"native-failure-{structured_enabled}"
    structured_path = output.with_suffix(".scf.jsonl")
    manifest_path = output.with_suffix(".system")
    progress_buffer = io.StringIO()

    def failing_native(*args, **kwargs):
        _forward_progress("iter=1 energy=-1 dE=0 grad=0.1 diis=0")
        _forward_progress("iter=2 energy=-1.1 dE=-0.1 grad=0.01 diis=2")
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(runner_module, "run_rhf", failing_native)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        runner_module.run_job(
            molecule,
            basis="sto-3g",
            method="rhf",
            output=output,
            structured_log=structured_enabled,
            progress=ProgressLogger(stream=progress_buffer, verbose=4),
            citations=False,
            crash_dump=True,
            output_qvf=False,
            record_hostname=False,
            write_molden_file=False,
            write_population_file=False,
            write_xyz_file=False,
        )

    if structured_enabled:
        assert [int(row["iter"]) for row in _scf_rows(output)] == [1, 2]
    else:
        assert not structured_path.exists()
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assert int(manifest["progress"]["iteration"]) == 2
    dump = load_dump(output.with_suffix(".dump"))
    assert int(dump["crash"]["n_iters_completed"]) == 2

    structured_before = (
        structured_path.read_bytes() if structured_enabled else None
    )
    manifest_before = manifest_path.read_bytes()
    terminal_before = progress_buffer.getvalue()
    _forward_progress("iter=999 energy=-999 dE=0 grad=0 diis=0")
    if structured_enabled:
        assert structured_path.read_bytes() == structured_before
    else:
        assert not structured_path.exists()
    assert manifest_path.read_bytes() == manifest_before
    assert progress_buffer.getvalue() == terminal_before


@pytest.mark.parametrize("raise_inside", [False, True])
def test_molecular_progress_scope_restores_nested_handler(
    raise_inside: bool,
) -> None:
    """Nested ContextVar registration restores on success and exception."""
    from vibeqc.output._cpp_diagnostics import (
        _forward_progress,
        install_progress_handler,
    )

    outer_rows: list[int] = []
    inner_rows: list[int] = []
    phase_before = runner_module._ACTIVE_MOLECULAR_PROGRESS_PHASE.get()
    install_progress_handler(
        lambda fields: outer_rows.append(int(fields["iter"]))
    )
    try:
        if raise_inside:
            with pytest.raises(RuntimeError, match="scope failure"):
                with runner_module._molecular_progress_scope(
                    lambda fields: inner_rows.append(int(fields["iter"])),
                    phase="nested.scf",
                ):
                    assert (
                        runner_module._ACTIVE_MOLECULAR_PROGRESS_PHASE.get()
                        == "nested.scf"
                    )
                    _forward_progress("iter=1 energy=-1 dE=0 grad=0 diis=0")
                    raise RuntimeError("scope failure")
        else:
            with runner_module._molecular_progress_scope(
                lambda fields: inner_rows.append(int(fields["iter"])),
                phase="nested.scf",
            ):
                assert (
                    runner_module._ACTIVE_MOLECULAR_PROGRESS_PHASE.get()
                    == "nested.scf"
                )
                _forward_progress("iter=1 energy=-1 dE=0 grad=0 diis=0")
        assert runner_module._ACTIVE_MOLECULAR_PROGRESS_PHASE.get() == phase_before
        _forward_progress("iter=2 energy=-1.1 dE=-0.1 grad=0 diis=2")
    finally:
        install_progress_handler(None)

    assert inner_rows == [1]
    assert outer_rows == [2]


def test_molecular_progress_scope_isolates_concurrent_threads() -> None:
    """Two concurrent native contexts cannot cross-route iteration rows."""
    from vibeqc.output._cpp_diagnostics import _forward_progress

    barrier = threading.Barrier(2)
    rows: dict[int, list[tuple[int, str]]] = {1: [], 2: []}
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            with runner_module._molecular_progress_scope(
                lambda fields: rows[worker_id].append(
                    (
                        int(fields["iter"]),
                        runner_module._ACTIVE_MOLECULAR_PROGRESS_PHASE.get(),
                    )
                ),
                phase=f"thread-{worker_id}.scf",
            ):
                barrier.wait(timeout=5.0)
                _forward_progress(
                    f"iter={worker_id} energy=-{worker_id} "
                    "dE=0 grad=0 diis=0"
                )
                barrier.wait(timeout=5.0)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(worker_id,))
        for worker_id in (1, 2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert rows == {
        1: [(1, "thread-1.scf")],
        2: [(2, "thread-2.scf")],
    }


def test_concurrent_native_rhf_jobs_do_not_cross_route_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real concurrent pybind calls keep each row with its own writer."""
    molecules = {
        "short": _h2_molecule(),
        "long": vq.Molecule(
            [
                vq.Atom(1, [0.0, 0.0, -1.0]),
                vq.Atom(1, [0.0, 0.0, 1.0]),
            ]
        ),
    }
    outputs = {name: tmp_path / name for name in molecules}
    buffers = {name: io.StringIO() for name in molecules}
    results: dict[str, object] = {}
    errors: list[BaseException] = []
    native_barrier = threading.Barrier(len(molecules))
    original_native = runner_module.run_rhf
    previous_threads = int(vq.get_num_threads())

    def synchronized_native(*args, **kwargs):
        native_barrier.wait(timeout=10.0)
        return original_native(*args, **kwargs)

    monkeypatch.setattr(runner_module, "run_rhf", synchronized_native)

    def worker(name: str) -> None:
        try:
            results[name] = runner_module.run_job(
                molecules[name],
                basis="sto-3g",
                method="rhf",
                output=outputs[name],
                structured_log=True,
                progress=ProgressLogger(stream=buffers[name], verbose=4),
                **_molecular_job_kwargs("rhf"),
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(name,))
        for name in molecules
    ]
    vq.set_num_threads(1)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20.0)
    finally:
        vq.set_num_threads(previous_threads)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert set(results) == set(molecules)
    assert float(results["short"].energy) != float(results["long"].energy)

    for name, result in results.items():
        canonical_trace = list(result.scf_trace)
        rows = _scf_rows(outputs[name])
        assert len(rows) == len(canonical_trace)
        assert [int(row["iter"]) for row in rows] == [
            int(step.iter) for step in canonical_trace
        ]
        assert [float(row["energy"]) for row in rows] == [
            float(step.energy) for step in canonical_trace
        ]
        assert all(row["phase"] == "scf" for row in rows)
        assert all(int(row["attempt"]) == 1 for row in rows)
        assert re.findall(
            r"iter\s+(\d+)\s+E\s*=",
            buffers[name].getvalue(),
        ) == [str(int(step.iter)) for step in canonical_trace]

        manifest = tomllib.loads(
            outputs[name].with_suffix(".system").read_text(encoding="utf-8")
        )
        assert int(manifest["progress"]["iteration"]) == int(
            canonical_trace[-1].iter
        )
        assert float(manifest["progress"]["energy_eh"]) == float(
            canonical_trace[-1].energy
        )
        records = _structured_records(outputs[name])
        assert records[-1]["event"] == "job_end"


@pytest.mark.parametrize("raise_inside", [False, True])
def test_periodic_then_molecular_progress_does_not_cross_route(
    raise_inside: bool,
) -> None:
    """Periodic lifecycle cleanup cannot capture a later molecular row."""
    import vibeqc.periodic_runner as periodic_runner_module
    from vibeqc.output._cpp_diagnostics import (
        _forward_progress,
        install_progress_handler,
    )

    periodic_rows: list[int] = []
    molecular_rows: list[int] = []

    @periodic_runner_module._periodic_output_lifecycle
    def completed_periodic_job() -> None:
        install_progress_handler(
            lambda fields: periodic_rows.append(int(fields["iter"]))
        )
        _forward_progress("iter=1 energy=-1 dE=0 grad=0 diis=0")
        if raise_inside:
            raise RuntimeError("periodic failure")

    install_progress_handler(None)
    try:
        if raise_inside:
            with pytest.raises(RuntimeError, match="periodic failure"):
                completed_periodic_job()
        else:
            completed_periodic_job()
        with runner_module._molecular_progress_scope(
            lambda fields: molecular_rows.append(int(fields["iter"]))
        ):
            _forward_progress("iter=2 energy=-1.1 dE=-0.1 grad=0 diis=2")
        _forward_progress("iter=3 energy=-1.2 dE=-0.1 grad=0 diis=3")
    finally:
        install_progress_handler(None)

    assert periodic_rows == [1]
    assert molecular_rows == [2]


@pytest.mark.parametrize("energy", [float("nan"), float("inf"), float("-inf")])
def test_molecular_progress_identity_handles_nonfinite_energy(
    energy: float,
) -> None:
    """Failure trajectories retain a stable exact replay identity."""
    live = [runner_module._scf_progress_identity(4, energy)]
    live.remove(runner_module._scf_progress_identity(4, energy))
    assert live == []


def test_stability_follow_reconvergence_labels_only_scf_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stability-triggered follow is a second SCF, not a Davidson row.

    The synthetic wrapper models the corrective reconvergence launched after
    a stability eigenvector is followed.  It deliberately does not claim
    live observability for the Davidson stability solve itself (issue #205).
    """
    molecule = _h2_molecule()
    output = tmp_path / "stability-follow"
    original = runner_module.run_uhf

    def stability_follow(*args, **kwargs):
        original(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "run_uhf", stability_follow)
    result = runner_module.run_job(
        molecule,
        basis="sto-3g",
        method="uhf",
        output=output,
        structured_log=True,
        progress=False,
        **_molecular_job_kwargs("uhf"),
    )

    records = _structured_records(output)
    rows = [row for row in records if row.get("event") == "scf_iter"]
    expected_one_attempt = list(range(1, int(result.n_iter) + 1))
    assert [int(row["attempt"]) for row in rows] == (
        [1] * int(result.n_iter) + [2] * int(result.n_iter)
    )
    assert [int(row["iter"]) for row in rows[: int(result.n_iter)]] == (
        expected_one_attempt
    )
    assert [int(row["iter"]) for row in rows[int(result.n_iter) :]] == (
        expected_one_attempt
    )
    assert all(row["phase"] == "scf" for row in rows)
    assert not any(
        str(row.get("phase", "")).startswith("stability")
        for row in records
    )
    converged = [
        row for row in records if row.get("event") == "scf_converged"
    ]
    assert len(converged) == 1
    assert int(converged[0]["selected_attempt"]) == 2

    manifest = tomllib.loads(
        output.with_suffix(".system").read_text(encoding="utf-8")
    )
    assert int(manifest["progress"]["selected_attempt"]) == 2


def test_post_scf_helper_rows_precede_job_end_and_restore_headline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helper SCFs remain visible without stealing the terminal headline."""
    import vibeqc.atomization as atomization_module
    from vibeqc.output._cpp_diagnostics import _forward_progress

    molecule = _h2_molecule()
    output = tmp_path / "post-scf-helper"

    def synthetic_atomization(*args, **kwargs):
        _forward_progress("iter=1 energy=-0.4 dE=0 grad=0.2 diis=0")
        _forward_progress("iter=2 energy=-0.5 dE=-0.1 grad=0.01 diis=2")
        return SimpleNamespace(
            e_molecule=-1.1,
            e_atoms_sum=-1.0,
            atomization=0.1,
            atomization_per_atom=0.05,
            atomization_kcal=62.7509474063,
            atomic_energies={1: -0.5},
            n_atoms=2,
        )

    monkeypatch.setattr(
        atomization_module,
        "atomization_energy",
        synthetic_atomization,
    )
    result = runner_module.run_job(
        molecule,
        basis="sto-3g",
        method="rhf",
        atomization=True,
        output=output,
        structured_log=True,
        progress=False,
        **_molecular_job_kwargs("rhf"),
    )

    records = _structured_records(output)
    headline = [
        row
        for row in records
        if row.get("event") == "scf_iter" and row.get("phase") == "scf"
    ]
    helper = [
        row
        for row in records
        if row.get("event") == "scf_iter"
        and row.get("phase") == "atomization.scf"
    ]
    assert len(headline) == int(result.n_iter)
    assert [int(row["iter"]) for row in helper] == [1, 2]
    helper_indices = [records.index(row) for row in helper]
    job_end_index = next(
        index
        for index, row in enumerate(records)
        if row.get("event") == "job_end"
    )
    assert max(helper_indices) < job_end_index
    assert records[-1]["event"] == "job_end"

    manifest = tomllib.loads(
        output.with_suffix(".system").read_text(encoding="utf-8")
    )
    assert manifest["progress"]["phase"] == "scf"
    assert int(manifest["progress"]["iteration"]) == int(result.n_iter)
    assert int(manifest["progress"]["selected_attempt"]) == 1
    assert float(manifest["progress"]["energy_eh"]) == float(result.energy)


def test_staged_rijcosx_global_trace_does_not_replay_live_local_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RIJCOSX-style 1..N segments deduplicate against global numbering.

    Each native grid segment restarts its local cycle at one, while the
    staging wrapper returns one canonical trace renumbered across segments.
    Exact ordered energies identify the already-durable rows; replay must not
    append a second copy merely because the iteration labels differ.
    """
    from vibeqc.output._cpp_diagnostics import (
        _forward_progress,
        install_progress_handler,
    )

    molecule = _h2_molecule()
    basis = vq.BasisSet(molecule, "sto-3g")
    install_progress_handler(None)
    inner_result = runner_module.run_rhf(molecule, basis, vq.RHFOptions())
    canonical_trace = [
        SimpleNamespace(
            iter=1,
            energy=-1.0,
            delta_e=0.0,
            grad_norm=4.0e-1,
            diis_subspace=1,
            wall_s=0.0,
        ),
        SimpleNamespace(
            iter=2,
            energy=-1.08,
            delta_e=-8.0e-2,
            grad_norm=1.0e-1,
            diis_subspace=2,
            wall_s=0.0,
        ),
        SimpleNamespace(
            iter=3,
            energy=-1.10,
            delta_e=-2.0e-2,
            grad_norm=1.0e-2,
            diis_subspace=3,
            wall_s=0.0,
        ),
        SimpleNamespace(
            iter=4,
            energy=-1.11,
            delta_e=-1.0e-2,
            grad_norm=1.0e-3,
            diis_subspace=4,
            wall_s=0.0,
        ),
    ]
    split = max(1, len(canonical_trace) // 2)
    segments = [canonical_trace[:split], canonical_trace[split:]]
    assert all(segments)

    class _AggregatedResult:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.scf_trace = canonical_trace
            self.n_iter = len(canonical_trace)
            self.energy = float(canonical_trace[-1].energy)
            self.converged = bool(inner.converged)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    def staged_native(*args, **kwargs):
        for segment in segments:
            for local_iteration, step in enumerate(segment, start=1):
                local_delta_e = (
                    0.0
                    if local_iteration == 1
                    else float(step.delta_e)
                )
                _forward_progress(
                    f"iter={local_iteration} "
                    f"energy={float(step.energy):.17g} "
                    f"dE={local_delta_e:.17g} "
                    f"grad={float(step.grad_norm):.17g} "
                    f"diis={int(step.diis_subspace)}"
                )
        return _AggregatedResult(inner_result)

    monkeypatch.setattr(runner_module, "run_rhf", staged_native)
    output = tmp_path / "staged-rijcosx"
    result = runner_module.run_job(
        molecule,
        basis="sto-3g",
        method="rhf",
        output=output,
        structured_log=True,
        progress=False,
        **_molecular_job_kwargs("rhf"),
    )

    rows = _scf_rows(output)
    assert len(rows) == len(canonical_trace) == int(result.n_iter)
    expected_local_iterations = [
        local_iteration
        for segment in segments
        for local_iteration in range(1, len(segment) + 1)
    ]
    assert [int(row["iter"]) for row in rows] == expected_local_iterations
    assert [float(row["energy"]) for row in rows] == [
        float(step.energy) for step in canonical_trace
    ]
    assert [int(row["attempt"]) for row in rows] == [
        attempt
        for attempt, segment in enumerate(segments, start=1)
        for _ in segment
    ]
    assert [int(step.iter) for step in result.scf_trace] == list(
        range(1, len(canonical_trace) + 1)
    )
    converged = [
        row
        for row in _structured_records(output)
        if row.get("event") == "scf_converged"
    ]
    assert converged[0]["selected_attempts"] == [1, 2]


def test_real_staged_rijcosx_rows_are_live_and_exact_once(
    tmp_path: Path,
) -> None:
    """The production three-grid RHF path preserves all local segments."""
    options = vq.RHFOptions()
    options.density_fit = True
    options.aux_basis = "def2-universal-jkfit"
    options.cosx = True
    output = tmp_path / "real-staged-rijcosx"
    result = runner_module.run_job(
        _h2_molecule(),
        basis="def2-svp",
        method="rhf",
        rhf_options=options,
        output=output,
        structured_log=True,
        progress=False,
        **_molecular_job_kwargs("rhf"),
    )

    canonical_trace = list(result.scf_trace)
    rows = _scf_rows(output)
    # Current H2/def2-SVP reaches 4 + 4 + 1 rows. Keep enough latitude for a
    # mathematically equivalent convergence-floor change while still proving
    # all three staged grids actually ran.
    assert 7 <= len(canonical_trace) <= 12
    assert len(rows) == len(canonical_trace) == int(result.n_iter)
    assert [float(row["energy"]) for row in rows] == [
        float(step.energy) for step in canonical_trace
    ]
    assert [int(step.iter) for step in canonical_trace] == list(
        range(1, len(canonical_trace) + 1)
    )

    local_iterations = [int(row["iter"]) for row in rows]
    reset_indices = [
        index
        for index in range(1, len(local_iterations))
        if local_iterations[index] <= local_iterations[index - 1]
    ]
    assert len(reset_indices) == 2
    attempt_ids = list(dict.fromkeys(int(row["attempt"]) for row in rows))
    assert attempt_ids == [1, 2, 3]
    assert sum(int(row["attempt"]) == 3 for row in rows) == 1

    converged = [
        row
        for row in _structured_records(output)
        if row.get("event") == "scf_converged"
    ]
    assert len(converged) == 1
    assert converged[0]["selected_attempts"] == attempt_ids


def test_real_cpcm_attempts_stream_exact_once(
    tmp_path: Path,
) -> None:
    """Gas and solvated CPCM SCFs retain their local attempt counters."""
    output = tmp_path / "real-cpcm-attempts"
    result = runner_module.run_job(
        _h2_molecule(),
        basis="sto-3g",
        method="rhf",
        solvent="water",
        output=output,
        structured_log=True,
        progress=False,
        **_molecular_job_kwargs("rhf"),
    )

    canonical_trace = list(result.scf_trace)
    rows = _scf_rows(output)
    assert len(rows) == len(canonical_trace)
    assert [int(row["iter"]) for row in rows] == [
        int(step.iter) for step in canonical_trace
    ]
    assert [float(row["energy"]) for row in rows] == [
        float(step.energy) for step in canonical_trace
    ]
    assert all(row["phase"] == "scf" for row in rows)

    attempt_ids = list(dict.fromkeys(int(row["attempt"]) for row in rows))
    assert len(attempt_ids) >= 2
    for attempt in attempt_ids:
        local_iterations = [
            int(row["iter"])
            for row in rows
            if int(row["attempt"]) == attempt
        ]
        assert local_iterations == list(range(1, len(local_iterations) + 1))

    records = _structured_records(output)
    converged = [
        row for row in records if row.get("event") == "scf_converged"
    ]
    assert len(converged) == 1
    assert converged[0]["selected_attempts"] == attempt_ids
    assert records[-1]["event"] == "job_end"


def test_quiet_cpp_diagnostics_still_stream_live_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QUIET C++ progress survives the quietest diagnostic threshold."""
    import vibeqc.output as output_module
    from vibeqc._vibeqc_core import (
        _get_diagnostics_level,
        _set_diagnostics_level,
    )
    from vibeqc.output._cpp_diagnostics import (
        _forward_progress,
        _progress_handler,
    )

    output_module.install_diagnostics_bridge()
    saved_level = output_module.output_level()
    saved_cpp_level = int(_get_diagnostics_level())
    saved_handler = _progress_handler.get()
    restored_handler = saved_handler
    output = tmp_path / "quiet-native-progress"
    original_native = runner_module.run_rhf
    original_emit = runner_module.StructuredLog.emit
    in_native_call = False
    live_states: list[bool] = []

    def observed_native(*args, **kwargs):
        nonlocal in_native_call
        in_native_call = True
        try:
            return original_native(*args, **kwargs)
        finally:
            in_native_call = False

    def observed_emit(self, event: str, **payload: object) -> None:
        original_emit(self, event, **payload)
        if event == "scf_iter" and self.enabled:
            live_states.append(in_native_call)

    monkeypatch.setattr(runner_module, "run_rhf", observed_native)
    monkeypatch.setattr(runner_module.StructuredLog, "emit", observed_emit)
    try:
        output_module.output_level(output_module.Level.QUIET)
        assert output_module.output_level() is output_module.Level.QUIET
        assert int(_get_diagnostics_level()) == int(output_module.Level.QUIET)
        result = runner_module.run_job(
            _h2_molecule(),
            basis="sto-3g",
            method="rhf",
            output=output,
            structured_log=True,
            progress=False,
            **_molecular_job_kwargs("rhf"),
        )
    finally:
        restored_handler = _progress_handler.get()
        if restored_handler is not saved_handler:
            _progress_handler.set(saved_handler)
        output_module.output_level(saved_level)
        # Preserve the exact pre-test native threshold even if another test
        # had intentionally placed it out of sync with the Python default.
        _set_diagnostics_level(saved_cpp_level)

    assert output_module.output_level() is saved_level
    assert int(_get_diagnostics_level()) == saved_cpp_level
    assert restored_handler is saved_handler
    assert _progress_handler.get() is saved_handler
    assert live_states == [True] * len(result.scf_trace)
    rows = _scf_rows(output)
    assert len(rows) == len(result.scf_trace)
    assert [float(row["energy"]) for row in rows] == [
        float(step.energy) for step in result.scf_trace
    ]

    structured_path = output.with_suffix(".scf.jsonl")
    structured_before = structured_path.read_bytes()
    if saved_handler is None:
        _forward_progress("iter=999 energy=-999 dE=0 grad=0 diis=0")
    assert structured_path.read_bytes() == structured_before


# ---------------------------------------------------------------------------
# 5. End-to-end: periodic SCF with progress=True emits iteration lines
# ---------------------------------------------------------------------------


def _h2_in_box(box: float = 30.0):
    """Tiny H₂ / STO-3G cell shared with test_periodic_rhf_ewald.py."""
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _default_periodic_opts() -> vq.PeriodicSCFOptions:
    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-7
    return opts


def test_progress_during_periodic_scf_emits_one_line_per_iteration(
) -> None:
    """Run a Γ-only periodic SCF with ``progress=`` directed at an
    in-memory buffer; the captured output must include exactly
    ``result.n_iter`` ``iter <n>`` lines and a ``converged``
    summary."""
    sysp, basis = _h2_in_box()
    opts = _default_periodic_opts()
    buf = io.StringIO()
    plog = ProgressLogger(stream=buf, verbose=True)

    result = vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        progress=plog,
    )

    out = buf.getvalue()
    iter_lines = re.findall(r"iter\s+(\d+)\s+E\s*=", out)
    assert int(result.n_iter) >= 2
    assert len(iter_lines) == int(result.n_iter), (
        f"iter-line count {len(iter_lines)} != result.n_iter "
        f"{result.n_iter}\n--- captured ---\n{out}"
    )
    assert "[integrals_lattice]" in out
    status = "converged" if result.converged else "NOT converged"
    assert f"SCF {status} in {result.n_iter} iterations" in out


# ---------------------------------------------------------------------------
# 6. Tee to a log file produces incremental writes (tail -f works)
# ---------------------------------------------------------------------------


def test_incremental_log_file_grows_during_iteration(tmp_path: Path) -> None:
    """A logger configured with ``log_path`` must append + flush after
    every call. We can't easily verify ``tail -f`` semantics from
    pytest, but we can verify the file size grows monotonically as
    successive ``iteration()`` calls land."""
    log_path = tmp_path / "progress.log"
    plog = ProgressLogger(stream=io.StringIO(), log_path=log_path,
                          verbose=True)
    plog.banner("start")
    sizes = [log_path.stat().st_size]

    for n in range(1, 5):
        plog.iteration(n, energy=-1.0 - 0.001 * n, dE=-1e-3, grad=1e-3,
                       diis=0)
        # Re-stat from disk after each emit; size must be strictly
        # increasing (the open-append-flush dance writes at least one
        # full line).
        sizes.append(log_path.stat().st_size)

    plog.converged(n_iter=4, energy=-1.004, converged=True)
    sizes.append(log_path.stat().st_size)

    for prev, cur in zip(sizes, sizes[1:]):
        assert cur > prev, f"log file did not grow: sizes={sizes}"

    final = log_path.read_text(encoding="utf-8")
    assert "iter    1" in final
    assert "iter    4" in final
    assert "SCF converged in 4 iterations" in final
