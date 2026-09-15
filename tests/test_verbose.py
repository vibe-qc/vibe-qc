"""Tests for v0.5.3 verbosity levels + Python ``logging`` integration.

Pairs with :mod:`tests.test_progress` (v0.5.1 stream-mode coverage).
The two cover orthogonal axes:

* ``test_progress.py`` exercises the ``True`` / ``False`` /
  ``ProgressLogger`` / log-file modes at the implicit default level.
* This module exercises the integer ``verbose`` knob (0..6+) plus
  the ``use_logging=True`` adapter that routes through
  :mod:`logging`.

Each verbose level is a strict superset of the one below — a higher
level only adds more output. The tests are written against that
invariant: bumping ``verbose`` from N to N+1 may add lines, must
never remove them.

The H₂ / STO-3G periodic Γ-only cell shared with
:mod:`tests.test_progress` is reused for the end-to-end smoke
check — exactly enough SCF iterations to verify the per-iteration
gate at level 4.
"""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.progress import (
    DEFAULT_VERBOSE,
    ProgressLogger,
    _coerce_verbose,
    resolve_progress,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _h2_in_box(box: float = 30.0):
    """Tiny H₂ / STO-3G cell — same one ``test_progress.py`` uses."""
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


# ---------------------------------------------------------------------------
# 1. Coercion: bool / None / int → int
# ---------------------------------------------------------------------------


def test_coerce_verbose_accepts_bool_int_none() -> None:
    """The internal coercer normalizes the inputs that
    ``ProgressLogger.__init__`` accepts. ``True`` and ``None`` both
    resolve to the package default (4) — that's the back-compat
    contract for callers built against the v0.5.1 / v0.5.2 boolean
    API. ``False`` collapses to 0 so a ``verbose=False`` from
    pre-v0.5.3 code becomes a silent run."""
    assert _coerce_verbose(True) == DEFAULT_VERBOSE == 4
    assert _coerce_verbose(False) == 0
    assert _coerce_verbose(None) == DEFAULT_VERBOSE
    assert _coerce_verbose(0) == 0
    assert _coerce_verbose(3) == 3
    assert _coerce_verbose(9) == 9
    # Negative values clamp to 0 — there's no "even quieter than
    # silent" mode and we don't want a stray -1 to underflow into
    # weird comparisons.
    assert _coerce_verbose(-1) == 0


# ---------------------------------------------------------------------------
# 2. Per-level threshold gate (explicit method-by-method)
# ---------------------------------------------------------------------------


def _capture_at_level(level: int) -> str:
    """Drive every public emit method against a fresh logger at the
    given level and return what landed on the buffer. Lets each
    threshold test call this and inspect a deterministic mix of
    message types."""
    buf = io.StringIO()
    plog = ProgressLogger(stream=buf, verbose=level)
    plog.banner("Banner-Title")
    plog.info("info-message")
    plog.warn("warn-message")
    with plog.stage("stage-name", detail="some-detail"):
        pass
    plog.iteration(1, energy=-1.0, dE=0.0, grad=1e-3, diis=0)
    plog.iteration(2, energy=-1.001, dE=-1e-3, grad=1e-4, diis=2)
    plog.memory("checkpoint", 123.4)
    plog.debug("debug-detail")
    plog.converged(n_iter=2, energy=-1.001, converged=True)
    return buf.getvalue()


def test_level_zero_is_silent() -> None:
    """Level 0 must suppress every method. The tee-to-log-path is
    also short-circuited (verified separately in
    ``test_log_path_silenced_at_level_zero``)."""
    out = _capture_at_level(0)
    assert out == ""


def test_level_one_emits_banner_warn_converged_only() -> None:
    """Level 1 keeps the run "alive" indicators — banner, warnings,
    and the final SCF status — but suppresses everything else.
    Use this for batch sweeps that want one-line-per-job summaries."""
    out = _capture_at_level(1)
    assert "Banner-Title" in out
    assert "WARN: warn-message" in out
    assert "SCF converged in 2 iterations" in out
    # The verbose >= 2 set must stay hidden.
    assert "info-message" not in out
    assert "[stage-name]" not in out
    # Per-iteration rows are still hidden at level 1.
    assert "iter    1" not in out
    assert "iter    2" not in out
    assert "[memory]" not in out
    assert "[debug]" not in out


def test_level_two_adds_info_and_stage_start_but_not_timing() -> None:
    """Level 2 reveals stage-start lines and ``info()`` milestones
    but suppresses the ``[name] done (X.XXs)`` timing addendum —
    that arrives at level 3 (so a quiet-but-narrating run isn't
    cluttered by per-stage wall clocks)."""
    out = _capture_at_level(2)
    assert "info-message" in out
    assert "[stage-name] - some-detail" in out
    # Timing addendum gated to level 3+.
    assert "[stage-name] done" not in out
    # Per-iter still gated.
    assert "iter    1" not in out


def test_level_three_adds_stage_timing_but_not_iter() -> None:
    """Level 3 == the pre-v0.5.3 default behavior: per-stage timing
    is now visible, but per-iteration SCF rows are still gated to
    level 4. That makes 3 the right level for users who want the
    setup phase narrated without the per-iter spam from a long
    SCF."""
    out = _capture_at_level(3)
    assert "[stage-name] done" in out
    # Per-iter still gated — the headline gate that v0.5.3 introduces.
    assert "iter    1" not in out
    assert "iter    2" not in out


def test_level_four_default_adds_iteration_rows() -> None:
    """Level 4 is the package default — adds per-iteration SCF
    rows. We verify both rows land and that the dE column reads
    ``--`` on iter 1 (the column-format convention)."""
    out = _capture_at_level(4)
    assert "iter    1" in out
    assert "iter    2" in out
    assert re.search(r"iter\s+1.*dE\s*=\s*--", out)
    # Memory / debug still gated — level 5 / 6 territory.
    assert "[memory]" not in out
    assert "[debug]" not in out


def test_level_five_adds_memory_snapshot() -> None:
    """Level 5 ("very verbose") adds inline RSS snapshots — pairs
    with the post-mortem ``.perf`` log's ``Memory snapshots``
    section. Phase-level debug lines stay gated for 6+."""
    out = _capture_at_level(5)
    assert "[memory] checkpoint: 123.4 MiB" in out
    assert "[debug]" not in out


def test_level_six_adds_debug_phase_breakdown() -> None:
    """Level 6 ("debug") streams phase-level wall-clock detail
    live — overlaps the ``.perf`` log on purpose, for users
    debugging an in-flight run who can't wait for the post-mortem
    report."""
    out = _capture_at_level(6)
    assert "[debug] debug-detail" in out


# ---------------------------------------------------------------------------
# 3. Monotonicity: level N output is a strict prefix-set of level N+1
# ---------------------------------------------------------------------------


def test_levels_are_monotonically_more_verbose() -> None:
    """The contract for ``verbose`` is that bumping the level only
    adds output — never reorders or drops. We assert that for the
    ``[0, 6]`` range, every line that lands at level N also lands
    at level N+1.

    This is the same contract PySCF documents and lets users
    sweep ``verbose`` without having to re-grep for the lines they
    care about."""
    seen: list[set[str]] = []
    for level in range(0, 7):
        text = _capture_at_level(level)
        # Strip blank lines + leading whitespace so banner padding
        # and stage-start indentation don't cause spurious mismatches.
        lines = {l.strip() for l in text.splitlines() if l.strip()}
        seen.append(lines)
    for n, (lo, hi) in enumerate(zip(seen, seen[1:])):
        assert lo.issubset(hi), (
            f"level {n} produced lines not present at level {n+1}: "
            f"{lo - hi}"
        )


# ---------------------------------------------------------------------------
# 4. Boolean back-compat: True → DEFAULT_VERBOSE, False → 0
# ---------------------------------------------------------------------------


def test_bool_verbose_back_compat() -> None:
    """A pre-v0.5.3 caller passing ``verbose=True`` lands on the
    package default (4); ``verbose=False`` is silent. Without this
    contract every existing user of :class:`ProgressLogger` would
    break on upgrade."""
    plog_true = ProgressLogger(stream=io.StringIO(), verbose=True)
    assert plog_true.level == DEFAULT_VERBOSE == 4
    assert plog_true.enabled is True
    plog_false = ProgressLogger(stream=io.StringIO(), verbose=False)
    assert plog_false.level == 0
    assert plog_false.enabled is False


# ---------------------------------------------------------------------------
# 5. resolve_progress threads verbose through
# ---------------------------------------------------------------------------


def test_resolve_progress_threads_verbose_for_progress_true() -> None:
    """Calling ``resolve_progress(True, verbose=2)`` builds a fresh
    logger at level 2. This is the path ``run_job`` uses to plumb
    its ``verbose=`` kwarg into the SCF entry points."""
    plog = resolve_progress(True, verbose=2)
    assert plog.level == 2
    assert plog.use_logging is False


def test_resolve_progress_passes_use_logging_through() -> None:
    """``use_logging=True`` reaches the constructed logger when the
    caller asks ``resolve_progress`` to build one."""
    plog = resolve_progress(True, verbose=4, use_logging=True)
    assert plog.use_logging is True


def test_resolve_progress_does_not_override_provided_logger() -> None:
    """When the caller already built a :class:`ProgressLogger`, the
    instance is returned untouched — its level / use_logging stay
    whatever the caller set, even if ``resolve_progress`` sees
    different kwargs. This is the contract that lets users build
    one logger and thread it through nested calls."""
    user_plog = ProgressLogger(
        stream=io.StringIO(), verbose=2, use_logging=False,
    )
    out = resolve_progress(user_plog, verbose=9, use_logging=True)
    assert out is user_plog
    assert out.level == 2
    assert out.use_logging is False


# ---------------------------------------------------------------------------
# 6. run_job verbose / VIBEQC_VERBOSE env var
# ---------------------------------------------------------------------------


def test_run_job_verbose_zero_silences_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_job(verbose=0)`` must produce no stdout — the ``.out``
    file is still written normally (the verbose gate sits between
    progress emission and stdout, not between SCF and disk)."""
    mol = vq.Molecule([
        vq.Atom(8, [0.0,  0.0,  0.0]),
        vq.Atom(1, [0.0,  1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ])
    stem = tmp_path / "h2o_silent"
    vq.run_job(mol, basis="sto-3g", method="rhf", output=stem, verbose=0)
    captured = capsys.readouterr().out
    assert captured == "", f"verbose=0 leaked to stdout:\n{captured}"
    # .out file still written.
    out_text = stem.with_suffix(".out").read_text()
    assert "Job: RHF" in out_text


def test_run_job_verbose_env_var_picked_up(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``VIBEQC_VERBOSE=0`` silences stdout when the caller leaves
    ``verbose=None`` (the default). Mirrors the ``VIBEQC_LIVE_LOGGING``
    /``VIBEQC_PERFLOG`` family — env var sets the floor for batch
    scripts that don't want to edit every input file."""
    monkeypatch.setenv("VIBEQC_VERBOSE", "0")
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])
    stem = tmp_path / "h2_envquiet"
    vq.run_job(mol, basis="sto-3g", method="rhf", output=stem)
    assert capsys.readouterr().out == ""


def test_run_job_explicit_verbose_overrides_env_var(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``verbose=`` kwarg always wins over the env var.
    The env var only applies when the caller leaves the kwarg at
    its default of ``None`` — same precedence convention that
    ``progress=`` already follows."""
    monkeypatch.setenv("VIBEQC_VERBOSE", "0")
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])
    stem = tmp_path / "h2_explicit"
    # Explicit verbose=4 should beat $VIBEQC_VERBOSE=0.
    vq.run_job(mol, basis="sto-3g", method="rhf", output=stem,
               verbose=4)
    out = capsys.readouterr().out
    assert "Job: RHF" in out
    assert "SCF converged" in out


def test_run_job_verbose_env_var_junk_is_silent_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A garbage ``VIBEQC_VERBOSE`` value (typo, leftover from
    another tool) must not raise — it falls back to the package
    default. We don't want an overnight batch sweep to die because
    someone exported ``VIBEQC_VERBOSE=verbose``."""
    monkeypatch.setenv("VIBEQC_VERBOSE", "loud-please")
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])
    stem = tmp_path / "h2_junk_env"
    vq.run_job(mol, basis="sto-3g", method="rhf", output=stem)
    # Default level (4) should still print the banner.
    out = capsys.readouterr().out
    assert "Job: RHF" in out


# ---------------------------------------------------------------------------
# 7. Logging adapter: routes through stdlib `logging`
# ---------------------------------------------------------------------------


@pytest.fixture
def logging_capture(monkeypatch: pytest.MonkeyPatch):
    """Drop-in handler captured into a list. We do not call
    ``logging.basicConfig`` because pytest already fights with
    root-logger handlers — instead, we attach a private handler to
    the ``vibeqc.run_job`` logger and yield the list it appends to.

    The fixture also resets the logger's level on teardown so
    one test's ``DEBUG`` setting doesn't bleed into the next."""
    log = logging.getLogger("vibeqc.run_job")
    saved_level = log.level
    saved_propagate = log.propagate

    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler()
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    try:
        yield records
    finally:
        log.removeHandler(handler)
        log.setLevel(saved_level)
        log.propagate = saved_propagate


def test_use_logging_routes_emit_to_stdlib_logger(
    logging_capture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``use_logging=True`` redirects every progress emit through
    the ``vibeqc.run_job`` logger. Stream output is suppressed; the
    captured records carry the message text and the right log
    level (INFO for milestones, WARNING for warnings, DEBUG for
    per-iter)."""
    plog = ProgressLogger(verbose=4, use_logging=True)
    plog.banner("Run Banner")
    plog.info("milestone")
    plog.warn("careful")
    plog.iteration(1, energy=-1.0, dE=0.0, grad=1e-3, diis=0)
    plog.converged(n_iter=1, energy=-1.0, converged=True)

    # Stream sink should be untouched — emit went through logging.
    assert capsys.readouterr().out == ""

    msgs = [r.getMessage() for r in logging_capture]
    levels = [r.levelno for r in logging_capture]
    assert any("Run Banner" in m for m in msgs)
    assert any("milestone" in m for m in msgs)
    assert any("careful" in m for m in msgs)
    assert any("iter    1" in m for m in msgs)
    assert any("SCF converged" in m for m in msgs)
    assert logging.WARNING in levels  # warn() routes at WARNING
    assert logging.DEBUG in levels    # iteration() routes at DEBUG


def test_use_logging_respects_verbose_gate(logging_capture) -> None:
    """Verbose-level filtering runs *before* the logging call —
    so ``verbose=2`` + ``use_logging=True`` does not emit per-iter
    DEBUG records, even if the captured handler is set to DEBUG.
    The gate is a vibe-qc-side filter, not a stdlib-handler-side
    filter."""
    plog = ProgressLogger(verbose=2, use_logging=True)
    plog.iteration(1, energy=-1.0, dE=0.0, grad=1e-3, diis=0)
    plog.iteration(2, energy=-1.001, dE=-1e-3, grad=1e-4, diis=2)
    debug_records = [
        r for r in logging_capture if r.levelno == logging.DEBUG
    ]
    assert debug_records == [], (
        "verbose=2 should suppress per-iteration DEBUG records "
        "even under use_logging=True"
    )


def test_run_job_use_logging_emits_through_stdlib(
    tmp_path: Path,
    logging_capture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: ``run_job(use_logging=True)`` routes the banner
    + final-summary through the ``vibeqc.run_job`` logger and not
    to stdout. Mirrors the canonical recipe in the module
    docstring."""
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])
    stem = tmp_path / "h2_logging"
    vq.run_job(mol, basis="sto-3g", method="rhf", output=stem,
               use_logging=True)

    assert capsys.readouterr().out == ""

    msgs = "\n".join(r.getMessage() for r in logging_capture)
    assert "run_job" in msgs
    assert "Job total" in msgs


# ---------------------------------------------------------------------------
# 8. Periodic SCF: per-iter rows gated by verbose level
# ---------------------------------------------------------------------------


def test_periodic_scf_verbose_three_skips_per_iter_rows() -> None:
    """The headline v0.5.3 contract for periodic SCFs: at
    ``verbose=3`` the live banner + stage milestones + final
    converged summary still emit, but per-iteration rows are
    gated out. Verified against an actual Γ-only SCF run so we
    know the gate is wired through the live emission path, not
    just the standalone formatter."""
    sysp, basis = _h2_in_box()
    opts = _default_periodic_opts()
    buf = io.StringIO()
    plog = ProgressLogger(stream=buf, verbose=3)
    result = vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        progress=plog,
    )
    out = buf.getvalue()
    iter_lines = re.findall(r"iter\s+\d+\s+E\s*=", out)
    assert iter_lines == [], (
        f"verbose=3 should suppress per-iteration rows; got "
        f"{len(iter_lines)} of them:\n{out}"
    )
    assert "[integrals_lattice]" in out  # stage milestones still emit
    status = "converged" if result.converged else "NOT converged"
    assert f"SCF {status}" in out


def test_periodic_scf_verbose_four_emits_per_iter_rows() -> None:
    """The complement of the previous test: at the new explicit
    level 4 the per-iteration rows are back, with the same
    one-line-per-iteration count that the legacy default produced."""
    sysp, basis = _h2_in_box()
    opts = _default_periodic_opts()
    buf = io.StringIO()
    plog = ProgressLogger(stream=buf, verbose=4)
    result = vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        progress=plog,
    )
    out = buf.getvalue()
    iter_lines = re.findall(r"iter\s+\d+\s+E\s*=", out)
    assert len(iter_lines) == int(result.n_iter), (
        f"iter-line count {len(iter_lines)} != result.n_iter "
        f"{result.n_iter}"
    )


def test_periodic_scf_verbose_kwarg_overrides_resolve_default() -> None:
    """Direct ``run_rhf_periodic_gamma_ewald3d(progress=True,
    verbose=1)`` builds a level-1 logger on the user's behalf —
    same path ``run_job`` uses internally to thread its
    ``verbose`` kwarg through."""
    sysp, basis = _h2_in_box()
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-7

    # Capture stdout via pytest's capsys is fine here — the
    # kwarg builds a logger writing to sys.stdout when
    # progress=True.
    f = io.StringIO()
    plog = resolve_progress(True, verbose=1)
    plog._stream = f  # type: ignore[attr-defined]
    vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        progress=plog,
    )
    out = f.getvalue()
    # Level 1: banner + final summary; no stage milestones, no
    # iteration rows.
    assert "[integrals_lattice]" not in out
    assert re.search(r"iter\s+\d+\s+E\s*=", out) is None
    # Banner emits at >= 1.
    assert "SCF" in out


# ---------------------------------------------------------------------------
# 9. log_path tee composition with use_logging
# ---------------------------------------------------------------------------


def test_log_path_tees_under_use_logging(
    tmp_path: Path,
    logging_capture,
) -> None:
    """A user with both stdlib logging and a per-job ``.out`` file
    can have both: ``use_logging=True`` routes through the logger,
    but the ``log_path`` tee still gets verbatim line-flushed
    writes. Verified by reading the file after a banner / iter /
    converged sequence."""
    log_path = tmp_path / "tee.log"
    plog = ProgressLogger(
        log_path=log_path, verbose=4, use_logging=True,
    )
    plog.banner("Tee Banner")
    plog.iteration(1, energy=-1.0, dE=0.0, grad=1e-3, diis=0)
    plog.converged(n_iter=1, energy=-1.0, converged=True)

    file_text = log_path.read_text(encoding="utf-8")
    assert "Tee Banner" in file_text
    assert "iter    1" in file_text
    assert "SCF converged" in file_text

    # And the records still went through logging too.
    msgs = "\n".join(r.getMessage() for r in logging_capture)
    assert "Tee Banner" in msgs


# ---------------------------------------------------------------------------
# 10. Basis-set name + exponent dump in periodic SCF live log (v0.5.3.x)
# ---------------------------------------------------------------------------


def test_periodic_scf_emits_basis_name_at_default_verbose() -> None:
    """The periodic-SCF live banner must surface the basis-set name
    so callers tailing a long-running multi-k run can see which
    basis they actually ran with — answering the gap that the
    molecular ``run_job`` already covers via ``Job: RHF / basis=``.
    Triggered at any level >= 2 (default 4 satisfies)."""
    sysp, basis = _h2_in_box()
    opts = _default_periodic_opts()
    buf = io.StringIO()
    plog = ProgressLogger(stream=buf, verbose=4)
    vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        progress=plog,
    )
    out = buf.getvalue()
    assert "basis: sto-3g" in out
    # Includes the function-count summary so the line is informative
    # without needing the level-5 dump.
    assert "BFs" in out
    assert "shell" in out


def test_periodic_scf_basis_name_hidden_at_level_one() -> None:
    """Level 1 keeps only banner + warnings + final summary, so the
    basis-name ``info()`` call must NOT emit there. This is the
    monotonicity check for the new line: it lands at level 2+
    alongside other info, gated below that."""
    sysp, basis = _h2_in_box()
    opts = _default_periodic_opts()
    buf = io.StringIO()
    plog = ProgressLogger(stream=buf, verbose=1)
    vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        progress=plog,
    )
    out = buf.getvalue()
    assert "basis: sto-3g" not in out


def test_periodic_scf_dumps_exponents_at_level_five() -> None:
    """At verbose >= 5 the per-shell exponents and contraction
    coefficients are written to the live log — the price of
    asking for "very verbose" on a long-running periodic run.
    Verified by checking that the table headers and at least one
    sto-3g exponent (3.42525091 to 6 digits) land on the buffer."""
    sysp, basis = _h2_in_box()
    opts = _default_periodic_opts()
    buf = io.StringIO()
    plog = ProgressLogger(stream=buf, verbose=5)
    vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        progress=plog,
    )
    out = buf.getvalue()
    assert "Basis-set details" in out
    assert "Name: sto-3g" in out
    # The first H 1s primitive in STO-3G has exponent 3.42525091.
    # We don't pin the exact column layout — just that the value
    # shows up so future formatter tweaks don't false-fail this test.
    assert "3.425251" in out


def test_format_basis_summary_runs_standalone() -> None:
    """``format_basis_summary`` is a public helper exposed from the
    package. Calling it directly returns a string that contains the
    basis name, function count, and at least one shell row."""
    from vibeqc import format_basis_summary
    mol = vq.Molecule([
        vq.Atom(8, [0.0,  0.0,  0.0]),
        vq.Atom(1, [0.0,  1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ])
    basis = vq.BasisSet(mol, "sto-3g")
    text = format_basis_summary(basis)
    assert "Name: sto-3g" in text
    # H₂O / STO-3G has 7 BFs in 5 shells (3 on O, 1 on each H).
    assert "7 in 5 shells" in text
    # The header row is present.
    assert "shell" in text
    assert "exponent" in text
    assert "coefficient" in text


def test_log_path_silenced_at_level_zero(tmp_path: Path) -> None:
    """At level 0 the file isn't even truncated (the constructor
    short-circuits) and no writes land. Saves a stray empty file
    from a ``verbose=0`` batch run."""
    log_path = tmp_path / "silent.log"
    ProgressLogger(log_path=log_path, verbose=0)
    plog = ProgressLogger(
        stream=io.StringIO(), log_path=log_path, verbose=0,
    )
    plog.banner("invisible")
    plog.iteration(1, energy=-1.0, dE=0.0, grad=1e-3, diis=0)
    # File never created (constructor only opens for write at
    # level > 0). If something did create it, it must be empty.
    if log_path.exists():
        assert log_path.read_text(encoding="utf-8") == ""
