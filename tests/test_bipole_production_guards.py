"""Production-readiness guards for the BIPOLE route.

Three previously silent hazards now warn (2026-07-12 bug batch from the
route audit; see handovers/HANDOVER_BIPOLE_PRODUCTION.md):

1. The 3-D legacy (non-exchange-split) gauge at multi-k reports a total
   energy that is NOT stationary — the spheropole enters the energy but
   never the Fock operator (documented Ha-scale wrong-sign k-mesh
   dependence). It stays reachable on purpose (ad-hoc k-lists / band
   paths auto-fall back to it), so it must announce itself.
2. Odd-electron cells at (default) multiplicity=1 are silently promoted
   to the minimum open shell (the deliberate 7991ae58 behavior for
   Al/Cu primitive cells); a user who *meant* a singlet must be told.
3. Net-charged cells run against an implicit jellium background whose
   absolute energy is convention- and volume-dependent.
"""

from __future__ import annotations

import io
import json
import warnings

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import PeriodicKSOptions, PeriodicRHFOptions
from vibeqc.output import Level, OutputChannel, structured_log
from vibeqc.pbc_bipole_common import (
    _spin_occupations,
    warn_bipole_charged_cell,
    warn_bipole_legacy_multik_gauge,
)
from vibeqc.progress import ProgressLogger, resolve_progress


def _h2_cell(charge: int = 0, multiplicity: int = 1):
    box = 8.0
    c = box / 2
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
        charge=charge,
        multiplicity=multiplicity,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lih_rocksalt_gamma():
    """Exact primitive-cell geometry from the archived cutoff-cost case."""
    a = 4.0840 / 0.529177210903
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [a / 2.0, a / 2.0, a / 2.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    return system, basis, kmesh


def _semantic_capture(tmp_path, callback, *, level, event, raises=None):
    """Return human output, selected structured events, and callback result."""
    human = io.StringIO()
    event_index = len(list(tmp_path.iterdir()))
    events_path = tmp_path / f"bipole-event-{event_index}.jsonl"
    with structured_log(events_path):
        with OutputChannel.to_stream(human, level=level):
            if raises is None:
                result = callback()
            else:
                with pytest.raises(raises) as caught:
                    callback()
                result = caught
    events = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if line.strip()
    ]
    return (
        human.getvalue(),
        [e for e in events if e.get("event") == event],
        result,
    )


def _semantic_warning_capture(tmp_path, callback, *, raises=None):
    """Capture quiet-level human output plus structured warnings."""
    return _semantic_capture(
        tmp_path,
        callback,
        level=Level.QUIET,
        event="warning",
        raises=raises,
    )


def _semantic_note_capture(tmp_path, callback, *, raises=None):
    """Capture standard-level human output plus structured notes."""
    return _semantic_capture(
        tmp_path,
        callback,
        level=Level.STANDARD,
        event="note",
        raises=raises,
    )


def test_legacy_multik_gauge_warns_through_driver(tmp_path):
    """RHF + explicit legacy gauge + multi-k emits the non-stationarity warning."""
    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = False
    with pytest.warns(RuntimeWarning, match="not a stationary value"):
        human, events, _ = _semantic_warning_capture(
            tmp_path,
            lambda: vq.run_pbc_bipole_rhf(
                system,
                basis,
                kmesh,
                opts,
                use_exchange_ewald_split=False,
                progress=False,
            ),
        )
    message = (
        "legacy-gauge multi-k total energies are non-stationary "
        "(spheropole in E, not in F) -- diagnostics only"
    )
    assert human == f"  WARNING: {message}\n"
    assert len(events) == 1
    assert events[0]["message"] == message
    assert events[0]["warning_kind"] == "legacy_multik_gauge"
    assert events[0]["n_k"] == 2


def test_corrected_gauge_multik_does_not_warn():
    """The default corrected gauge at multi-k stays warning-free."""
    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        vq.run_pbc_bipole_rhf(system, basis, kmesh, opts, progress=False)


def test_archived_lih_default_fold_stops_before_fock(monkeypatch):
    """The default LiH deck must not enter SCF with unreliable AO support.

    The archived production case measured an S(Gamma) fold drift of
    7.0985e-2 at the default 15-bohr cutoff and then spent hours in BIPOLE.
    Current main reproduces that drift in the driver's real preflight.  The
    two-electron sentinel makes this a seconds-scale cutoff/cost regression.
    """
    import vibeqc.pbc_bipole as bipole

    system, basis, kmesh = _lih_rocksalt_gamma()
    opts = PeriodicRHFOptions()
    opts.max_iter = 1
    opts.use_diis = False

    def entered_expensive_fock(*_args, **_kwargs):
        raise AssertionError("entered the expensive BIPOLE Fock build")

    monkeypatch.setattr(
        bipole,
        "build_bipole_restricted_fock",
        entered_expensive_fock,
    )
    with pytest.raises(
        RuntimeError,
        match=r"S\(Γ\) fold truncation 7\.1e-02.*refusing to enter SCF",
    ):
        vq.run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            use_exchange_ewald_split=True,
            progress=False,
        )


@pytest.mark.parametrize("drift", [1e-3, 1e-1])
def test_fold_truncation_uses_semantic_output(tmp_path, monkeypatch, drift):
    """Moderate/high RHF fold drift becomes a structured note/warning."""
    import vibeqc.pbc_bipole_common as common

    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = False
    monkeypatch.setattr(
        common,
        "s_fold_truncation_drift",
        lambda *_args, **_kwargs: drift,
    )
    live = io.StringIO()
    plog = ProgressLogger(stream=live, verbose=2)

    capture = (
        _semantic_warning_capture if drift > 1e-2 else _semantic_note_capture
    )
    human, events, result = capture(
        tmp_path,
        lambda: vq.run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            use_exchange_ewald_split=True,
            progress=plog,
        ),
        raises=RuntimeError if drift > 1e-2 else None,
    )
    if drift > 1e-2:
        message = (
            "S(Γ) fold truncation 1.0e-01 at cutoff 6.0 bohr -- the "
            "lattice sums are badly under-converged for this basis's AO "
            "tails; absolute energies are UNRELIABLE (spurious SCF states "
            "possible). Increase lattice_opts.cutoff_bohr until the drift "
            "falls below 1e-4."
        )
        human_prefix = "WARNING"
        kind_key = "warning_kind"
        live_prefix = "WARNING"
        assert "refusing to enter SCF" in str(result.value)
    else:
        message = (
            "S(Γ) fold truncation 1.0e-03 at cutoff 6.0 bohr -- expect "
            "~1e-03-scale absolute-energy truncation; increase "
            "cutoff_bohr for tighter work"
        )
        human_prefix = "Note"
        kind_key = "note_kind"
        live_prefix = "note"
    fold_note = f"  {human_prefix}: {message}\n"
    if drift > 1e-2:
        assert human == fold_note
    else:
        assert human.startswith(fold_note)
        phase_output = human[len(fold_note):]
        assert "Exact density evaluation pending (Ha)" in phase_output
        assert "Exact density evaluation done (s)" in phase_output
    assert len(events) == 1
    assert events[0]["message"] == message
    assert events[0][kind_key] == "fold_truncation"
    assert events[0]["method"] == "rhf"
    assert events[0]["drift"] == drift
    assert events[0]["cutoff_bohr"] == 6.0
    assert events[0]["n_k"] == 1
    assert f"    {live_prefix}: {message}\n" in live.getvalue()


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
def test_critical_fold_fails_after_warning_all_four_drivers(
    tmp_path, monkeypatch, method
):
    """Every direct BIPOLE driver records the warning, then fails closed."""
    import vibeqc.pbc_bipole_common as common

    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions() if method in {"rks", "uks"} else PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = False
    monkeypatch.setattr(
        common,
        "s_fold_truncation_drift",
        lambda *_args, **_kwargs: 0.1,
    )
    runner = getattr(vq, f"run_pbc_bipole_{method}")
    kwargs = {"functional": "SVWN"} if method in {"rks", "uks"} else {}

    human, events, error = _semantic_warning_capture(
        tmp_path,
        lambda: runner(
            system,
            basis,
            kmesh,
            opts,
            use_exchange_ewald_split=True,
            progress=False,
            **kwargs,
        ),
        raises=RuntimeError,
    )

    assert human.startswith("  WARNING: S(Γ) fold truncation 1.0e-01")
    assert len(events) == 1
    assert events[0]["warning_kind"] == "fold_truncation"
    assert events[0]["method"] == method
    assert events[0]["drift"] == 0.1
    assert events[0]["cutoff_bohr"] == 6.0
    assert events[0]["n_k"] == 1
    assert "refusing to enter SCF" in str(error.value)


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
@pytest.mark.parametrize(
    "drift",
    [1e-6, 1e-3],
    ids=["converged-support", "moderate-support"],
)
def test_successful_corrected_gauge_result_preserves_overlap_fold_drift(
    monkeypatch, method, drift
):
    """Every direct driver returns the corrected-gauge fold diagnostic."""
    import vibeqc.pbc_bipole_common as common

    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions() if method in {"rks", "uks"} else PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = False
    monkeypatch.setattr(
        common,
        "s_fold_truncation_drift",
        lambda *_args, **_kwargs: drift,
    )
    runner = getattr(vq, f"run_pbc_bipole_{method}")
    kwargs = {"functional": "SVWN"} if method in {"rks", "uks"} else {}

    result = runner(
        system,
        basis,
        kmesh,
        opts,
        use_exchange_ewald_split=True,
        progress=False,
        **kwargs,
    )

    assert result.exchange_ewald_split is True
    assert result.overlap_fold_drift == pytest.approx(drift)


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
@pytest.mark.parametrize(
    "drift",
    [False, float("nan"), float("inf"), -1.0],
    ids=["boolean", "nan", "infinite", "negative"],
)
def test_corrected_gauge_rejects_invalid_overlap_fold_drift(
    monkeypatch, method, drift
):
    """Invalid preflight telemetry cannot be reinterpreted as support."""
    import vibeqc.pbc_bipole_common as common

    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions() if method in {"rks", "uks"} else PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    monkeypatch.setattr(
        common,
        "s_fold_truncation_drift",
        lambda *_args, **_kwargs: drift,
    )
    runner = getattr(vq, f"run_pbc_bipole_{method}")
    kwargs = {"functional": "SVWN"} if method in {"rks", "uks"} else {}

    with pytest.raises(RuntimeError, match="finite nonnegative drift"):
        runner(
            system,
            basis,
            kmesh,
            opts,
            use_exchange_ewald_split=True,
            progress=False,
            **kwargs,
        )


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
def test_legacy_gauge_result_records_overlap_fold_drift(monkeypatch, method):
    """The fold guard is gauge-independent (hoisted 2026-08-06).

    Historically the guard sat inside ``if exchange_split_active:`` in
    all four drivers, so legacy-gauge runs skipped the reliability check
    entirely (the fold-guard hole, HANDOVER_OPEN_BUGS_V015.md) — and
    this test pinned that hole as intended behavior. The k-fold
    eigenproblem consumes S(k) in BOTH gauges, so the guard now runs
    unconditionally and legacy results record the measured drift."""
    import vibeqc.pbc_bipole_common as common

    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions() if method in {"rks", "uks"} else PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = False

    monkeypatch.setattr(
        common,
        "s_fold_truncation_drift",
        lambda *_args, **_kwargs: 3.3e-7,
    )
    runner = getattr(vq, f"run_pbc_bipole_{method}")
    kwargs = {"functional": "SVWN"} if method in {"rks", "uks"} else {}

    result = runner(
        system,
        basis,
        kmesh,
        opts,
        use_exchange_ewald_split=False,
        progress=False,
        **kwargs,
    )

    assert result.exchange_ewald_split is False
    assert result.overlap_fold_drift == pytest.approx(3.3e-7)


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
def test_legacy_gauge_fold_guard_fails_closed(monkeypatch, method):
    """Unreliable fold support refuses SCF on the LEGACY gauge too."""
    import vibeqc.pbc_bipole_common as common

    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions() if method in {"rks", "uks"} else PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = False

    monkeypatch.setattr(
        common,
        "s_fold_truncation_drift",
        lambda *_args, **_kwargs: 7.7e-2,
    )
    runner = getattr(vq, f"run_pbc_bipole_{method}")
    kwargs = {"functional": "SVWN"} if method in {"rks", "uks"} else {}

    with pytest.raises(
        common.BipoleFoldUnreliableError, match="refusing to enter SCF"
    ):
        runner(
            system,
            basis,
            kmesh,
            opts,
            use_exchange_ewald_split=False,
            progress=False,
            **kwargs,
        )


def test_legacy_gauge_gamma_does_not_warn(tmp_path):
    """Γ-only legacy gauge is the documented CRYSTAL convention — no warning."""
    system, _ = _h2_cell()
    plog = resolve_progress(False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        human, events, _ = _semantic_warning_capture(
            tmp_path,
            lambda: warn_bipole_legacy_multik_gauge(system, False, 1, plog),
        )
    assert human == ""
    assert events == []


def test_odd_electron_mult1_promotion_warns():
    system, _ = _h2_cell()  # geometry irrelevant; use an H-atom cell below
    h_cell = vq.PeriodicSystem(
        3, np.eye(3) * 8.0, [vq.Atom(1, [4.0, 4.0, 4.0])]
    )  # 1 electron, default multiplicity=1
    with pytest.warns(RuntimeWarning, match="minimum open shell"):
        n_alpha, n_beta = _spin_occupations(h_cell)
    assert (n_alpha, n_beta) == (1, 0)  # the 7991ae58 behavior is unchanged


def test_explicit_doublet_does_not_warn():
    h_cell = vq.PeriodicSystem(
        3, np.eye(3) * 8.0, [vq.Atom(1, [4.0, 4.0, 4.0])], multiplicity=2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _spin_occupations(h_cell) == (1, 0)


def test_charged_cell_warns(tmp_path):
    system, _ = _h2_cell(charge=1, multiplicity=2)
    live = io.StringIO()
    plog = ProgressLogger(stream=live, verbose=2)
    with pytest.warns(RuntimeWarning, match="jellium"):
        human, events, _ = _semantic_warning_capture(
            tmp_path,
            lambda: warn_bipole_charged_cell(system, plog),
        )
    message = (
        "net cell charge +1 -> jellium background; "
        "absolute E is convention-dependent"
    )
    assert human == f"  WARNING: {message}\n"
    assert len(events) == 1
    assert events[0]["message"] == message
    assert events[0]["warning_kind"] == "charged_cell"
    assert events[0]["charge"] == 1
    assert live.getvalue() == f"    WARNING: {message}\n"


def test_semantic_diagnostic_emits_for_each_invocation(tmp_path):
    """BIPOLE leaves emission policy to the shared output logger."""
    system, _ = _h2_cell(charge=1, multiplicity=2)
    live = io.StringIO()
    plog = ProgressLogger(stream=live, verbose=2)

    def emit_twice():
        warn_bipole_charged_cell(system, plog)
        warn_bipole_charged_cell(system, plog)

    with pytest.warns(RuntimeWarning, match="jellium") as caught:
        human, events, _ = _semantic_warning_capture(tmp_path, emit_twice)
    message = (
        "net cell charge +1 -> jellium background; "
        "absolute E is convention-dependent"
    )
    assert len(caught) == 2
    assert human == 2 * f"  WARNING: {message}\n"
    assert len(events) == 2
    assert live.getvalue() == 2 * f"    WARNING: {message}\n"


def test_neutral_cell_does_not_warn(tmp_path):
    system, _ = _h2_cell()
    plog = resolve_progress(False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        human, events, _ = _semantic_warning_capture(
            tmp_path,
            lambda: warn_bipole_charged_cell(system, plog),
        )
    assert human == ""
    assert events == []


def test_exact_j_core_tail_warns_on_dense_core_basis(tmp_path):
    """MgO/STO-3G at ke=200 warns; the needed-ke estimate matches the
    triage's ~11000 Ha figure (2*ln(1e6)*gamma_max, Mg 1s ~ 300)."""
    from vibeqc.pbc_bipole_common import warn_bipole_exact_j_core_tail

    ANG2BOHR = 1.0 / 0.529177210903
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plog = resolve_progress(False, verbose=False)
    with pytest.warns(RuntimeWarning, match="core reciprocal"):
        human, events, ke_needed = _semantic_warning_capture(
            tmp_path,
            lambda: warn_bipole_exact_j_core_tail(basis, 200.0, plog),
        )
    assert ke_needed is not None and 8000.0 < ke_needed < 16000.0
    message = (
        f"exact-FT J ke_cutoff 200 Ha < ~{ke_needed:.0f} Ha "
        "core-tail coverage -- absolute totals undercount; "
        "differences unaffected"
    )
    assert human == f"  WARNING: {message}\n"
    assert len(events) == 1
    assert events[0]["message"] == message
    assert events[0]["warning_kind"] == "exact_j_core_tail"
    assert events[0]["ke_cutoff_ha"] == 200.0
    assert events[0]["ke_needed_ha"] == pytest.approx(ke_needed)
    assert events[0]["precision"] == 1e-6
    # A cutoff covering the tails stays silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        silent_human, silent_events, result = _semantic_warning_capture(
            tmp_path,
            lambda: warn_bipole_exact_j_core_tail(
                basis,
                ke_needed + 1.0,
                plog,
            ),
        )
    assert result is None
    assert silent_human == ""
    assert silent_events == []


def test_exact_j_core_tail_silent_on_light_basis(tmp_path):
    """H2/STO-3G core tails are covered at the production ke=200."""
    from vibeqc.pbc_bipole_common import warn_bipole_exact_j_core_tail

    system, basis = _h2_cell()
    plog = resolve_progress(False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        human, events, result = _semantic_warning_capture(
            tmp_path,
            lambda: warn_bipole_exact_j_core_tail(basis, 200.0, plog),
        )
    assert result is None
    assert human == ""
    assert events == []


def _h2_vacuum_box_18():
    """H2 centred in an 18-bohr box: reliable fold support at the input
    geometry (the periodic .out snapshot fixture)."""
    box = 18.0
    c = box / 2
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    kmesh = vq.monkhorst_pack(system, (1, 1, 1))
    return system, kmesh


def test_fold_refusal_on_trial_geometry_is_a_barrier(monkeypatch):
    """A fold-gate refusal after the first good evaluation is recoverable.

    L-BFGS-B's first unit trial step in fractional coordinates can throw
    atoms ~one lattice vector out of the cell; the fold preflight rightly
    refuses that geometry. Inside an optimization that refusal must act as
    the same penalty barrier as ``SCFNonConvergence`` -- the line search
    backs off -- not kill the whole job (which is exactly what the H2
    periodic-geomopt snapshot job did before this fix)."""
    import vibeqc.pbc_bipole_common as common
    from vibeqc.bipole_optimize import OptimizeResult, relax_atoms

    system, kmesh = _h2_vacuum_box_18()
    orig = common.s_fold_truncation_drift
    calls = {"n": 0}

    def drift_with_one_bad_probe(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            return 5e-2  # the >1e-2 unreliable branch -> gate refusal
        return orig(*args, **kwargs)

    monkeypatch.setattr(
        common, "s_fold_truncation_drift", drift_with_one_bad_probe
    )
    result = relax_atoms(
        system,
        "sto-3g",
        kmesh,
        method="RHF",
        max_iter=1,
        conv_tol_grad=1e-3,
        cutoff_bohr=15.0,
        use_exchange_ewald_split=True,
    )
    assert calls["n"] >= 2  # the refusal really fired
    assert isinstance(result, OptimizeResult)
    # The barrier energy (1e6 Ha) is never used as a real objective value.
    assert result.energy < 0.0


def test_fold_refusal_at_initial_geometry_still_aborts(monkeypatch):
    """With no good evaluation to back off to, the refusal stays fatal."""
    import vibeqc.pbc_bipole_common as common
    from vibeqc.bipole_optimize import relax_atoms

    system, kmesh = _h2_vacuum_box_18()
    monkeypatch.setattr(
        common, "s_fold_truncation_drift", lambda *a, **k: 5e-2
    )
    assert issubclass(common.BipoleFoldUnreliableError, RuntimeError)
    with pytest.raises(
        common.BipoleFoldUnreliableError, match="refusing to enter SCF"
    ):
        relax_atoms(
            system,
            "sto-3g",
            kmesh,
            method="RHF",
            max_iter=1,
            conv_tol_grad=1e-3,
            cutoff_bohr=15.0,
            use_exchange_ewald_split=True,
        )
