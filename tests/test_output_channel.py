"""Tests for the ambient output channel (``vibeqc.output.channel``).

The channel replaces a file handle threaded through every call chain. The
properties pinned here are the ones the runners depend on: a bare import
opens nothing, ``write()`` outside a channel is silent, files are
line-buffered so ``tail -f`` works, and channels nest.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from vibeqc.output import (
    Level,
    OutputChannel,
    active_channel,
    flush,
    output_level,
    strict_output,
    write,
)


@pytest.fixture
def _restore_strict():
    """Restore strict mode after a test flips it, so it cannot leak."""
    saved = strict_output()
    try:
        yield
    finally:
        strict_output(saved)


@pytest.fixture
def _restore_level():
    """Restore the default channel level after a test changes it."""
    saved = output_level()
    try:
        yield
    finally:
        output_level(saved)


# --------------------------------------------------------------------
# The null / no-channel default
# --------------------------------------------------------------------


def test_write_without_a_channel_is_a_silent_no_op(capsys):
    write("this must vanish\n")
    flush()
    assert capsys.readouterr().out == ""


def test_importing_the_module_opens_no_file_and_raises_nothing():
    # Run in a fresh interpreter: a stale ContextVar from another test
    # could mask an import side effect.
    code = textwrap.dedent(
        """
        from vibeqc.output import write, flush, active_channel
        assert active_channel() is None
        write("nothing")
        flush()
        print("clean")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip() == "clean"


def test_active_channel_is_none_outside_a_with_block(tmp_path):
    assert active_channel() is None
    with OutputChannel.to_file(tmp_path / "a.out") as ch:
        assert active_channel() is ch
    assert active_channel() is None


def test_null_channel_discards_but_is_installed(tmp_path, capsys):
    with OutputChannel.null() as ch:
        assert active_channel() is ch
        write("gone\n")
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------


def test_to_file_writes_the_text(tmp_path):
    path = tmp_path / "job.out"
    with OutputChannel.to_file(path):
        write("hello\n")
        write("world\n")
    assert path.read_text() == "hello\nworld\n"


def test_to_file_append_mode_preserves_existing_content(tmp_path):
    path = tmp_path / "job.out"
    path.write_text("first\n")
    with OutputChannel.to_file(path, mode="a"):
        write("second\n")
    assert path.read_text() == "first\nsecond\n"


def test_bipole_optimizer_summary_uses_the_ambient_channel(tmp_path, capsys):
    from vibeqc.bipole_optimize import _emit_opt_line

    _emit_opt_line("outside")
    assert capsys.readouterr().out == ""

    path = tmp_path / "job.out"
    with OutputChannel.to_file(path):
        _emit_opt_line("Atomic relaxation")
    assert path.read_text() == "Atomic relaxation\n"


def test_to_file_rejects_an_unsupported_mode(tmp_path):
    with pytest.raises(ValueError, match="mode must be"):
        OutputChannel.to_file(tmp_path / "x.out", mode="rb")


def test_to_stdout_writes_to_the_terminal(capsys):
    with OutputChannel.to_stdout():
        write("live\n")
    assert capsys.readouterr().out == "live\n"


def test_tee_writes_to_both_file_and_stdout(tmp_path, capsys):
    path = tmp_path / "job.out"
    with OutputChannel.tee(path):
        write("both\n")
    assert path.read_text() == "both\n"
    assert capsys.readouterr().out == "both\n"


def test_to_stream_writes_to_a_caller_owned_stream():
    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        write("captured\n")
    # The channel must not close a stream it did not open.
    assert not buf.closed
    assert buf.getvalue() == "captured\n"


def test_stdout_is_resolved_per_write_not_captured_at_enter(capsys):
    # capsys swaps sys.stdout; a channel that captured it at __enter__
    # would write to the pre-swap object and the assertion would fail.
    with OutputChannel.to_stdout():
        write("after swap\n")
    assert "after swap" in capsys.readouterr().out


# --------------------------------------------------------------------
# Streaming guarantee
# --------------------------------------------------------------------


def test_file_is_line_buffered_so_tail_f_sees_lines_immediately(tmp_path):
    path = tmp_path / "job.out"
    with OutputChannel.to_file(path):
        write("first line\n")
        # No flush() call: line buffering alone must have pushed it.
        assert path.read_text() == "first line\n"
        write("partial, no newline")
        flush()
        assert path.read_text() == "first line\npartial, no newline"


def test_explicit_flush_pushes_a_partial_line(tmp_path):
    path = tmp_path / "job.out"
    with OutputChannel.to_file(path):
        write("no newline yet")
        flush()
        assert path.read_text() == "no newline yet"


# --------------------------------------------------------------------
# Context-manager semantics
# --------------------------------------------------------------------


def test_channels_nest_and_restore_the_outer_channel(tmp_path):
    outer, inner = tmp_path / "outer.out", tmp_path / "inner.out"
    with OutputChannel.to_file(outer):
        write("o1\n")
        with OutputChannel.to_file(inner):
            write("i1\n")
        write("o2\n")
    assert outer.read_text() == "o1\no2\n"
    assert inner.read_text() == "i1\n"


def test_exception_inside_the_block_propagates_and_closes_the_file(tmp_path):
    path = tmp_path / "job.out"
    with pytest.raises(RuntimeError, match="boom"):
        with OutputChannel.to_file(path):
            write("before\n")
            raise RuntimeError("boom")
    assert active_channel() is None
    # Text written before the failure survives -- a crashed job's .out is
    # the primary diagnostic.
    assert path.read_text() == "before\n"


def test_channel_is_not_reentrant(tmp_path):
    ch = OutputChannel.to_file(tmp_path / "a.out")
    with ch:
        with pytest.raises(RuntimeError, match="not reentrant"):
            with ch:
                pass


def test_reusing_a_closed_channel_raises(tmp_path):
    ch = OutputChannel.to_file(tmp_path / "a.out")
    with ch:
        pass
    with pytest.raises(RuntimeError, match="already been closed"):
        with ch:
            pass


def test_write_on_a_closed_channel_raises(tmp_path):
    ch = OutputChannel.to_file(tmp_path / "a.out")
    with ch:
        pass
    with pytest.raises(RuntimeError, match="closed OutputChannel"):
        ch.write("x")


def test_close_is_idempotent(tmp_path):
    ch = OutputChannel.to_file(tmp_path / "a.out")
    with ch:
        write("x\n")
    ch.close()
    ch.close()


def test_write_returns_the_character_count():
    buf = io.StringIO()
    with OutputChannel.to_stream(buf) as ch:
        assert ch.write("abc") == 3


# --------------------------------------------------------------------
# The single-threaded assumption this design rests on
# --------------------------------------------------------------------


def test_contextvar_does_not_reach_a_worker_thread(tmp_path):
    """Pins the documented limitation, so it fails loudly if relied upon.

    A ContextVar is per-context. A thread started with threading.Thread
    begins with a fresh context, so it sees the default (None) and its
    writes are dropped. vibe-qc spawns no Python threads today; if it
    ever does, this test breaks and channel.write() needs a
    threading.local fallback.
    """
    path = tmp_path / "job.out"
    seen: list[object] = []

    def worker() -> None:
        seen.append(active_channel())
        write("from thread\n")

    with OutputChannel.to_file(path):
        write("from main\n")
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert seen == [None], "a worker thread now inherits the channel"
    assert path.read_text() == "from main\n"


_THREAD_NEEDLES = ("threading.Thread(", "ThreadPoolExecutor(")

# Sites that deliberately spawn Python threads, keyed by (path relative to
# python/vibeqc, needle) -> (occurrences, why the ContextVar gap is harmless
# there). An entry is only admissible when *nothing on the threaded call
# path writes to vibeqc.output* -- that has to be established by reading the
# call chain, not assumed. A thread that emits needs the threading.local
# fallback channel.py's docstring describes, not an entry here.
_THREAD_ALLOWLIST: dict[tuple[str, str], tuple[int, str]] = {
    ("periodic_k_gdf.py", "ThreadPoolExecutor("): (
        1,
        "_k_from_signed_factors maps only its _k_for_bra closure. The worker "
        "reads cached arrays and signed density factors, asks the pure integer "
        "_exchange_auxiliary_panel_rank helper for a bounded panel, and performs "
        "NumPy contractions. Neither closure nor helper emits output or calls "
        "native OpenMP code. MPI partitioning/gathering happens on the caller.",
    ),
    ("periodic_corrected_exchange.py", "ThreadPoolExecutor("): (
        1,
        "CorrectedEwaldExchange.k_space_terms_all_k farms the O(n_k^2) "
        "corrected-"
        "exchange k loop over threads (f179f55f2, 7.1x on 16 threads). The "
        "worker chain -- k_space_terms_at_k -> bipole_fock_ewald."
        "compute_K_long_range_at_k -> _aopair_ft / bipole_ext_el_pole / the "
        "C++ core -- is pure numerical work: none of those modules imports "
        "vibeqc.output or vibeqc.progress, and the C++ core emits nothing by "
        "invariant (tests/test_cpp_emits_no_user_output.py). No write() can "
        "reach the pool, so no channel needs to.",
    ),
    ("periodic/ccm/neutral.py", "ThreadPoolExecutor("): (
        1,
        "ccm_neutral_cderi_fold farms the per-(k_a, k_b) unit-cell fits of "
        "one q-channel over threads (d7b82612d, measured 3.95x on 4). The "
        "pool maps exactly one closure -- _build_one -> np.asarray + "
        "aux_basis.build_lpq_bloch_native_fft. Traced transitively that is "
        "25 functions to depth 6 across two modules, vibeqc.aux_basis and "
        "vibeqc._aopair_ft; NEITHER FILE contains a print() or imports "
        "vibeqc.output / vibeqc.progress anywhere, which is a whole-file "
        "property, so no branch of either can emit. The builder's optional "
        "progress= logger is never passed by the worker and _progress_info "
        "returns immediately on None. Everything else the chain calls is "
        "numpy/scipy or the C++ core (direct_lattice_cells, "
        "ao_pair_fourier_transform_bloch_ss/_cxx), which emits nothing by "
        "invariant (tests/test_cpp_emits_no_user_output.py). Confirmed at "
        "runtime: profiling every worker thread of a threaded fold -- s-only "
        "C++ kernel, forced pure-Python MD path, and the symmetry star path "
        "-- ran 163-168 distinct functions per variant and entered "
        "vibeqc/output or progress.py zero times.\n"
        "CAUTION for the next editor: unlike the entry above, THIS file does "
        "import write() (neutral.py line ~120), and ccm_neutral_cderi_fold "
        "itself calls it -- deliberately hoisted above the pool, in the "
        "symmetry-setup block, on the calling thread. Keep it that way. A "
        "write() moved into _build_one, or into anything it calls, is "
        "dropped silently rather than failing here.",
    ),
}


def test_python_threads_only_where_the_output_gap_is_harmless():
    """The premise of the ContextVar choice, checked against the tree.

    A ContextVar does not cross into a worker thread, so a ``write()``
    there is dropped silently. Any thread in ``python/vibeqc`` must
    therefore be on a call path that emits nothing -- reviewed once and
    recorded in ``_THREAD_ALLOWLIST`` with its reasoning. A new thread
    site, or a new thread in an already-allowlisted file, fails here.
    """
    root = Path(__file__).resolve().parents[1] / "python" / "vibeqc"
    found: dict[tuple[str, str], int] = {}
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for needle in _THREAD_NEEDLES:
            count = text.count(needle)
            if count:
                found[(str(py.relative_to(root)), needle)] = count

    offenders = [
        f"{path}: {needle} x{count} "
        f"(allowlisted: x{_THREAD_ALLOWLIST.get((path, needle), (0, ''))[0]})"
        for (path, needle), count in sorted(found.items())
        if _THREAD_ALLOWLIST.get((path, needle), (0, ""))[0] != count
    ]
    assert not offenders, (
        "vibe-qc spawns a Python thread that is not covered by "
        "_THREAD_ALLOWLIST; vibeqc.output.channel's ContextVar will not "
        "reach it, so any write() below it is dropped silently. Either "
        "confirm the threaded call path emits nothing and add an entry "
        "with that reasoning, or give channel.write() a threading.local "
        "fallback.\n" + "\n".join(offenders)
    )

    # An allowlist entry that no longer matches a real site is stale: drop
    # it, so the exemption cannot outlive the code it was written for.
    stale = sorted(key for key in _THREAD_ALLOWLIST if key not in found)
    assert not stale, (
        "_THREAD_ALLOWLIST has entries for threads that no longer exist; "
        "remove them.\n" + "\n".join(f"{path}: {needle}" for path, needle in stale)
    )


# --------------------------------------------------------------------
# Strict mode -- silent loss becomes a loud failure
# --------------------------------------------------------------------


def test_strict_mode_is_off_by_default(capsys):
    # Production stays lenient: a channel-less write vanishes, never raises.
    assert strict_output() is False
    write("vanishes\n")  # must not raise
    assert capsys.readouterr().out == ""


def test_strict_mode_makes_channelless_write_raise(_restore_strict):
    strict_output(True)
    with pytest.raises(RuntimeError, match="no active OutputChannel"):
        write("x\n")


def test_strict_mode_makes_channelless_flush_raise(_restore_strict):
    strict_output(True)
    with pytest.raises(RuntimeError, match="no active OutputChannel"):
        flush()


def test_strict_mode_does_not_affect_writes_inside_a_channel(_restore_strict, tmp_path):
    strict_output(True)
    path = tmp_path / "job.out"
    with OutputChannel.to_file(path):
        write("kept\n")  # a channel is active -> strict is irrelevant
        flush()
    assert path.read_text() == "kept\n"


def test_strict_output_returns_the_resulting_value(_restore_strict):
    assert strict_output(True) is True
    assert strict_output() is True
    assert strict_output(False) is False


def test_strict_mode_reads_the_env_var_at_import():
    # The env gate must actually wire through, checked in a fresh
    # interpreter so this test's own process state can't mask it.
    code = textwrap.dedent(
        """
        from vibeqc.output import write, strict_output
        assert strict_output() is True, "VIBEQC_STRICT_OUTPUT=1 not honored"
        try:
            write("x")
        except RuntimeError:
            print("raised")
        else:
            raise SystemExit("strict write did not raise")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "VIBEQC_STRICT_OUTPUT": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "raised"


# --------------------------------------------------------------------
# Log levels -- methods tag writes, the channel filters by threshold
# --------------------------------------------------------------------


def _levels_through(channel_level):
    buf = io.StringIO()
    with OutputChannel.to_stream(buf, level=channel_level):
        write("q\n", Level.QUIET)
        write("s\n")  # default STANDARD
        write("v\n", Level.VERBOSE)
        write("d\n", Level.DEBUG)
    return buf.getvalue().split()


def test_default_channel_level_is_standard():
    assert output_level() == Level.STANDARD
    # An untagged write is STANDARD and appears on a default channel.
    assert _levels_through(Level.STANDARD) == ["q", "s"]


def test_quiet_channel_shows_only_quiet():
    assert _levels_through(Level.QUIET) == ["q"]


def test_verbose_channel_adds_verbose():
    assert _levels_through(Level.VERBOSE) == ["q", "s", "v"]


def test_debug_channel_shows_everything():
    assert _levels_through(Level.DEBUG) == ["q", "s", "v", "d"]


def test_untagged_write_is_standard_so_out_is_unchanged(tmp_path):
    # The whole point: existing write(text) calls keep emitting on a
    # default channel, so the .out byte content does not change.
    path = tmp_path / "job.out"
    with OutputChannel.to_file(path):  # default STANDARD
        write("line one\n")
        write("line two\n")
    assert path.read_text() == "line one\nline two\n"


def test_suppressed_write_still_returns_len_for_the_stream_protocol():
    # print(..., file=channel) checks the returned count; a suppressed
    # message must still report len so print does not error.
    buf = io.StringIO()
    with OutputChannel.to_stream(buf, level=Level.QUIET) as ch:
        assert ch.write("verbose text", Level.VERBOSE) == len("verbose text")
    assert buf.getvalue() == ""


def test_print_to_channel_defaults_to_standard(tmp_path):
    path = tmp_path / "job.out"
    with OutputChannel.to_file(path) as ch:
        print("via print", file=ch)  # print calls write(s) with no level
    assert path.read_text() == "via print\n"


def test_set_level_raises_the_threshold_at_runtime():
    buf = io.StringIO()
    with OutputChannel.to_stream(buf, level=Level.STANDARD) as ch:
        write("hidden\n", Level.DEBUG)  # suppressed at STANDARD
        ch.set_level(Level.DEBUG)
        write("shown\n", Level.DEBUG)  # now visible
    assert buf.getvalue() == "shown\n"


def test_output_level_accepts_name_int_and_enum(_restore_level):
    assert output_level("verbose") == Level.VERBOSE
    assert output_level(3) == Level.DEBUG
    assert output_level(Level.QUIET) == Level.QUIET


def test_output_level_rejects_a_bad_name(_restore_level):
    with pytest.raises(ValueError, match="unknown output level"):
        output_level("chatty")


def test_output_level_sets_the_default_for_new_channels(_restore_level):
    output_level("verbose")
    # A channel built with no explicit level adopts the new default.
    assert _levels_through_default() == ["q", "s", "v"]


def _levels_through_default():
    buf = io.StringIO()
    with OutputChannel.to_stream(buf):  # no explicit level -> the default
        write("q\n", Level.QUIET)
        write("s\n")
        write("v\n", Level.VERBOSE)
        write("d\n", Level.DEBUG)
    return buf.getvalue().split()


def test_env_var_sets_the_default_level_at_import():
    # Fresh interpreter so this process's default can't mask it.
    code = textwrap.dedent(
        """
        from vibeqc.output import output_level, Level
        assert output_level() is Level.VERBOSE, output_level()
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={**os.environ, "VIBEQC_OUTPUT_LEVEL": "verbose"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# --------------------------------------------------------------------
# section() / section_header() -- consistent titled blocks
# --------------------------------------------------------------------


def test_section_header_sizes_the_rule_to_the_title_by_default():
    from vibeqc.output import section_header

    assert section_header("Energy") == "  Energy\n  ------\n"


def test_section_header_fixed_width():
    from vibeqc.output import section_header

    assert section_header("X", width=4) == "  X\n  ----\n"


def test_section_writes_header_then_yields_for_the_body(tmp_path):
    from vibeqc.output import section

    path = tmp_path / "job.out"
    with OutputChannel.to_file(path):
        with section("Convergence", width=11):
            write("    ok\n")
    assert path.read_text() == "  Convergence\n  -----------\n    ok\n"


def test_section_is_byte_identical_to_the_hand_drawn_pattern(tmp_path):
    # The migration invariant: `with section(title, width=56):` reproduces
    # the runners' old `write(title); write("  " + "-"*56)` exactly.
    from vibeqc.output import section

    a, b = tmp_path / "a.out", tmp_path / "b.out"
    with OutputChannel.to_file(a):
        write("  Energy summary\n")
        write("  " + "-" * 56 + "\n")
        write("    E = 1\n")
    with OutputChannel.to_file(b):
        with section("Energy summary", width=56):
            write("    E = 1\n")
    assert a.read_text() == b.read_text()


def test_section_carries_its_level():
    from vibeqc.output import Level, section

    buf = io.StringIO()
    with OutputChannel.to_stream(buf, level=Level.QUIET):
        with section("Diagnostics", level=Level.DEBUG):
            pass
    assert buf.getvalue() == ""  # DEBUG header suppressed on a QUIET channel


# --------------------------------------------------------------------
# record() -- one call, human .out line + structured event
# --------------------------------------------------------------------


def test_record_writes_human_line_and_structured_event(tmp_path):
    import json

    from vibeqc.output import record
    from vibeqc.output.formats.structured_log import structured_log

    human = tmp_path / "job.out"
    jsonl = tmp_path / "job.scf.jsonl"
    with OutputChannel.to_file(human):
        with structured_log(jsonl):
            record("total_energy", "  Total energy: -76.0 Ha\n", value=-76.0)
    # human surface
    assert human.read_text() == "  Total energy: -76.0 Ha\n"
    # machine surface
    events = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    rec = [e for e in events if e.get("event") == "total_energy"]
    assert len(rec) == 1 and rec[0]["value"] == -76.0


def test_record_human_none_emits_only_the_structured_event(tmp_path):
    import json

    from vibeqc.output import record
    from vibeqc.output.formats.structured_log import structured_log

    human = tmp_path / "job.out"
    jsonl = tmp_path / "job.scf.jsonl"
    with OutputChannel.to_file(human):
        with structured_log(jsonl):
            record("machine_only", None, k=1)
    assert human.read_text() == ""
    assert '"machine_only"' in jsonl.read_text()


def test_record_no_op_surfaces_are_independent(capsys):
    # No channel and no structured log: record() must not raise.
    from vibeqc.output import record

    record("nothing", "vanishes\n", x=1)
    assert capsys.readouterr().out == ""


def test_record_respects_the_channel_level():
    from vibeqc.output import Level, record

    buf = io.StringIO()
    with OutputChannel.to_stream(buf, level=Level.QUIET):
        record("x", "verbose text\n", level=Level.VERBOSE)
    assert buf.getvalue() == ""  # human suppressed; structured still fired


def test_record_once_key_coalesces_human_and_structured_surfaces(tmp_path):
    from vibeqc.output import record
    from vibeqc.output.formats.structured_log import structured_log

    log_path = tmp_path / "events.jsonl"
    buf = io.StringIO()
    with structured_log(log_path):
        with OutputChannel.to_stream(buf):
            record("guard", "first\n", once_key="guard.fold", value=1)
            record("guard", "second\n", once_key="guard.fold", value=2)
            record("guard", "other\n", once_key="guard.other", value=3)

    assert buf.getvalue() == "first\nother\n"
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    guards = [event for event in events if event["event"] == "guard"]
    assert [event["value"] for event in guards] == [1, 3]


def test_record_once_key_works_for_a_machine_only_scope(tmp_path):
    from vibeqc.output import record
    from vibeqc.output.formats.structured_log import structured_log

    log_path = tmp_path / "events.jsonl"
    with structured_log(log_path):
        record("guard", None, once_key="guard.fold", value=1)
        record("guard", None, once_key="guard.fold", value=2)

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    guards = [event for event in events if event["event"] == "guard"]
    assert [event["value"] for event in guards] == [1]


def test_record_once_key_scope_resets_with_a_fresh_channel():
    from vibeqc.output import record

    first = io.StringIO()
    second = io.StringIO()
    with OutputChannel.to_stream(first):
        record("guard", "first\n", once_key="guard.fold")
        record("guard", "duplicate\n", once_key="guard.fold")
    with OutputChannel.to_stream(second):
        record("guard", "fresh\n", once_key="guard.fold")

    assert first.getvalue() == "first\n"
    assert second.getvalue() == "fresh\n"


def test_record_once_key_rejects_non_string_keys():
    from vibeqc.output import record

    with pytest.raises(TypeError, match="once_key must be a string or None"):
        record("guard", None, once_key=object())


def test_record_once_key_is_not_claimed_when_strict_write_raises(
    tmp_path,
    _restore_strict,
):
    from vibeqc.output import record
    from vibeqc.output.formats.structured_log import structured_log

    log_path = tmp_path / "events.jsonl"
    buf = io.StringIO()
    strict_output(True)
    with structured_log(log_path):
        with pytest.raises(RuntimeError, match="no active OutputChannel"):
            record("guard", "first\n", once_key="guard.fold", value=1)
        with OutputChannel.to_stream(buf):
            record("guard", "retry\n", once_key="guard.fold", value=2)

    assert buf.getvalue() == "retry\n"
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    guards = [event for event in events if event["event"] == "guard"]
    assert [event["value"] for event in guards] == [2]


def test_warn_matches_the_hand_written_warning_line():
    from vibeqc.output import warn

    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        warn("CIF write failed")
    # Byte-identical to the write("  WARNING: ...\n") the runners scatter,
    # so migrating those to warn() does not move the .out at STANDARD.
    assert buf.getvalue() == "  WARNING: CIF write failed\n"


def test_warn_and_note_forward_once_key():
    from vibeqc.output import note, warn

    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        warn("fold warning", once_key="fold.warning")
        warn("fold warning repeated", once_key="fold.warning")
        note("fold note", once_key="fold.note")
        note("fold note repeated", once_key="fold.note")

    assert buf.getvalue() == (
        "  WARNING: fold warning\n"
        "  Note: fold note\n"
    )


def test_warn_survives_quiet():
    from vibeqc.output import Level, warn

    buf = io.StringIO()
    with OutputChannel.to_stream(buf, level=Level.QUIET):
        warn("basis fell back to STO-3G")
    assert buf.getvalue() == "  WARNING: basis fell back to STO-3G\n"


def test_warn_fires_a_structured_warning_event(tmp_path):
    from vibeqc.output import warn
    from vibeqc.output.formats.structured_log import structured_log

    log_path = tmp_path / "events.jsonl"
    buf = io.StringIO()
    with structured_log(log_path):
        with OutputChannel.to_stream(buf):
            warn("coupled LMP2 did not converge", cycles=50)
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    warning = next(e for e in events if e["event"] == "warning")
    assert warning["message"] == "coupled LMP2 did not converge"
    assert warning["cycles"] == 50


def test_note_is_standard_level():
    from vibeqc.output import Level, note

    buf = io.StringIO()
    with OutputChannel.to_stream(buf, level=Level.QUIET):
        note("using default grid")
    assert buf.getvalue() == ""  # a note is not essential; quiet drops it

    buf = io.StringIO()
    with OutputChannel.to_stream(buf):
        note("using default grid")
    assert buf.getvalue() == "  Note: using default grid\n"
