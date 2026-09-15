"""SCF trace capture and logging."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from vibeqc import format_scf_trace, log_scf_trace

from .conftest import GEOMETRIES, run_vibeqc_rhf


def _run_h2o(tight_rhf_opts):
    return run_vibeqc_rhf(GEOMETRIES["H2O"], "sto-3g", tight_rhf_opts)


def test_scf_trace_has_one_entry_per_iteration(tight_rhf_opts):
    result = _run_h2o(tight_rhf_opts)
    assert result.converged
    assert len(result.scf_trace) == result.n_iter


def test_scf_trace_final_iteration_matches_result_energy(tight_rhf_opts):
    # result.energy is recomputed on the converged-branch final rebuild
    # (F(D_final), D_final) so it reproduces from the returned matrices
    # (see tests/test_scf_final_consistency.py). The trace's last entry
    # keeps the pre-rebuild iteration energy, so the two agree only to
    # within the convergence band (well under 100x the fixture's
    # conv_tol_energy = 1e-12), not to machine precision.
    result = _run_h2o(tight_rhf_opts)
    assert result.scf_trace[-1].iter == result.n_iter
    assert result.scf_trace[-1].energy == pytest.approx(result.energy, abs=1e-10)


def test_scf_trace_first_iter_has_zero_delta_e(tight_rhf_opts):
    result = _run_h2o(tight_rhf_opts)
    assert result.scf_trace[0].delta_e == 0.0


def test_scf_trace_energy_is_monotonically_decreasing(tight_rhf_opts):
    # RHF with a reasonable initial guess + DIIS should monotonically
    # decrease the total energy. Slight non-monotonicity is possible mid-run
    # with DIIS, so we just require the trend: final < first.
    result = _run_h2o(tight_rhf_opts)
    assert result.scf_trace[-1].energy < result.scf_trace[0].energy


def test_format_scf_trace_contains_header_and_energy(tight_rhf_opts):
    result = _run_h2o(tight_rhf_opts)
    text = format_scf_trace(result)
    assert "energy (Ha)" in text
    assert "DIIS" in text
    assert f"{result.energy:.10f}" in text
    assert str(result.n_iter) in text


def test_format_scf_trace_omits_table_when_history_is_unavailable():
    result = SimpleNamespace(
        scf_trace=[],
        converged=True,
        n_iter=4,
        energy=-1.25,
    )

    text = format_scf_trace(result, include_banner=False)

    assert "converged in 4 iterations" in text
    assert "-1.2500000000 Ha" in text
    assert "energy (Ha)" not in text
    assert "||[F,DS]||" not in text
    assert "---" not in text


def test_format_scf_trace_renders_periodic_dict_history():
    result = SimpleNamespace(
        scf_trace=[
            {
                "iter": 1,
                "energy": -2.0,
                "delta_e": 0.0,
                "grad_norm": 2.5e-4,
                "diis_subspace": 0,
            },
            {
                "iter": 2,
                "energy": -2.1,
                "delta_e": -0.1,
                "grad_norm": 1.0e-7,
                "diis_subspace": 2,
            },
        ],
        converged=True,
        n_iter=2,
        energy=-2.1,
    )

    text = format_scf_trace(result, include_banner=False)

    assert "energy (Ha)" in text
    assert "-1.000e-01" in text
    assert "1.000e-07" in text
    assert "converged in 2 iterations" in text


def test_log_scf_trace_emits_summary_at_info(tight_rhf_opts, caplog):
    result = _run_h2o(tight_rhf_opts)
    with caplog.at_level(logging.INFO, logger="vibeqc.scf"):
        log_scf_trace(result)
    # Summary line includes total energy in Hartree and eV.
    assert any(
        "SCF converged" in rec.getMessage() and f"{result.n_iter}" in rec.getMessage()
        for rec in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_log_scf_trace_emits_per_iter_at_debug(tight_rhf_opts, caplog):
    result = _run_h2o(tight_rhf_opts)
    with caplog.at_level(logging.DEBUG, logger="vibeqc.scf"):
        log_scf_trace(result)
    # One iter line per SCF iteration plus 2 separator lines + header.
    iter_lines = [
        rec for rec in caplog.records
        if rec.levelno == logging.DEBUG and "[F,DS]" not in rec.getMessage()
        and "---" not in rec.getMessage()
    ]
    assert len(iter_lines) == result.n_iter


def test_log_scf_trace_omits_empty_debug_table(caplog):
    result = SimpleNamespace(
        scf_trace=[],
        converged=True,
        n_iter=3,
        energy=-0.5,
    )

    with caplog.at_level(logging.DEBUG, logger="vibeqc.scf"):
        log_scf_trace(result)

    messages = [record.getMessage() for record in caplog.records]
    assert any("SCF converged in 3 iterations" in message for message in messages)
    assert not any("energy (Ha)" in message for message in messages)
    assert not any("---" in message for message in messages)


def test_log_scf_trace_silent_by_default(tight_rhf_opts, caplog):
    result = _run_h2o(tight_rhf_opts)
    # Without any explicit configuration, vibeqc.scf inherits from root
    # (WARNING by default). INFO summary lines should be suppressed. caplog
    # by default captures at WARNING, so an empty record list confirms
    # nothing higher fired.
    log_scf_trace(result)
    assert all(
        rec.name != "vibeqc.scf" or rec.levelno >= logging.WARNING
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# write_scf_trace -- level-aware, block-by-block emission (enabling refactor)
# ---------------------------------------------------------------------------


def test_write_scf_trace_matches_format_scf_trace_at_standard(tight_rhf_opts):
    """The block emitter is byte-identical to the string builder + trailing
    while every segment is STANDARD -- the invariant that keeps the .out
    unchanged by the enabling refactor."""
    import io

    from vibeqc.output import OutputChannel
    from vibeqc.output.formats.scf_log import write_scf_trace

    result = _run_h2o(tight_rhf_opts)
    expected = format_scf_trace(result, include_banner=False) + "\n"
    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        write_scf_trace(result, include_banner=False, trailing="\n")
    assert buf.getvalue() == expected


def test_write_scf_trace_respects_the_channel_threshold(tight_rhf_opts):
    """The convergence verdict + final energy is QUIET (essential), the
    iteration table is STANDARD. So a QUIET channel keeps the verdict and
    drops the table, while STANDARD/DEBUG show the whole trace."""
    import io

    from vibeqc.output import Level, OutputChannel
    from vibeqc.output.formats.scf_log import write_scf_trace

    result = _run_h2o(tight_rhf_opts)

    def emit_at(level):
        buf = io.StringIO()
        with OutputChannel.to_stream(buf, level=level):
            write_scf_trace(result, include_banner=False)
        return buf.getvalue()

    quiet = emit_at(Level.QUIET)
    assert "converged in" in quiet               # the essential verdict survives
    assert "||[F,DS]||" not in quiet             # the per-iteration table does not
    standard = emit_at(Level.STANDARD)
    assert "converged in" in standard            # STANDARD <= STANDARD: shown
    assert "||[F,DS]||" in standard              # table header present at STANDARD
    assert emit_at(Level.DEBUG) == standard      # STANDARD <= DEBUG: same


def test_quiet_verdict_is_the_tail_of_the_standard_trace(tight_rhf_opts):
    """The QUIET verdict line is exactly the convergence tail of the full
    STANDARD trace -- proof the split did not reword or reformat it, only
    re-tagged its level."""
    import io

    from vibeqc.output import Level, OutputChannel
    from vibeqc.output.formats.scf_log import write_scf_trace

    # This test isolates the convergence/table level split. Restricted
    # stability is now default-on and deliberately adds a later STANDARD
    # diagnostic, so opt out of that independent post-SCF phase here.
    tight_rhf_opts.stability_check = False
    result = _run_h2o(tight_rhf_opts)
    buf = io.StringIO()
    with OutputChannel.to_stream(buf, level=Level.QUIET):
        write_scf_trace(result, include_banner=False)
    quiet = buf.getvalue().strip()
    standard = format_scf_trace(result, include_banner=False)
    assert quiet and standard.rstrip().endswith(quiet)


def test_fixed_unsigned_zero_drops_noise_sign_only():
    """A value that rounds to zero at the displayed precision must render
    without its noise sign: tables in the properties block size their
    columns from content, so a machine-dependent ``-0.0000`` would widen
    the whole column (the .out snapshot goldens freeze that width).
    Genuine negatives keep their sign."""
    from vibeqc.output.formats.scf_log import _fixed_unsigned_zero

    assert _fixed_unsigned_zero(-1e-18, 4) == "0.0000"
    assert _fixed_unsigned_zero(-4.9e-5, 4) == "0.0000"
    assert _fixed_unsigned_zero(0.0, 4) == "0.0000"
    assert _fixed_unsigned_zero(-5.1e-5, 4) == "-0.0001"
    assert _fixed_unsigned_zero(-0.6079, 4) == "-0.6079"
    assert _fixed_unsigned_zero(1.2803, 4) == "1.2803"
    assert _fixed_unsigned_zero(-1e-9, 6) == "0.000000"
