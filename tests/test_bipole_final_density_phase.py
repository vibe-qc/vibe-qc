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


def _check_refused_confirmation_xc(
    monkeypatch, driver, mesh, functional, damped, *, capped=False,
):
    """A refused exact check must not repeat quadrature at the same density.

    Refuse one check without changing its numerical operands. Damping must
    invalidate reuse, and the later exact confirmation must still run.
    """
    import hashlib
    import importlib
    from vibeqc._vibeqc_core import PeriodicKSOptions, PeriodicXCDensityDomain

    module = importlib.import_module("vibeqc.pbc_bipole_" + driver)
    system = vq.PeriodicSystem(
        3, 10.0 * np.eye(3),
        [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])],
        multiplicity=1 if driver == "rks" else 3,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = PeriodicKSOptions()
    external = functional == "external"
    if external:
        from tests.test_xc_external_provider import (
            _QuadraticNonlocalProvider, _external_name,
        )
        provider = _QuadraticNonlocalProvider(scale=0.01)
        options.functional = _external_name(provider)
    else:
        options.functional = functional.split("+")[0]
    options.grid.n_radial = 20
    options.lattice_opts.cutoff_bohr = 12.
    options.lattice_opts.nuclear_cutoff_bohr = 12.
    options.max_iter = 3 if capped else 40
    options.conv_tol_energy = 1e-9
    options.conv_tol_grad = 1e-8
    options.initial_guess = InitialGuess.SAD
    options.damping = 0.1 if damped else 0.
    options.dynamic_damping = False
    options.use_diis = not damped
    xc_name = ("build_xc_periodic_kpoints" if driver == "rks"
               else "build_xc_periodic_uks_kpoints")
    fock_name = ("build_bipole_restricted_fock" if driver == "rks"
                 else "build_bipole_unrestricted_fock")
    original_xc = getattr(module, xc_name)
    original_fock = getattr(module, fock_name)
    original_check = module.bipole_terminal_check
    calls, builds, checks = [], [], []

    def observe_fock(*args, **kwargs):
        builds.append(kwargs.get("use_incremental", True))
        return original_fock(*args, **kwargs)

    def observe_xc(*args, **kwargs):
        digest = hashlib.sha256()
        for density in args[4:5] if driver == "rks" else args[4:6]:
            digest.update(np.asarray([c.index for c in density.cells],
                                     dtype=np.int64).tobytes())
            digest.update(np.asarray(density.blocks).tobytes())
        calls.append((len(builds), digest.hexdigest()))
        return original_xc(*args, **kwargs)

    def refuse_once(*args, **kwargs):
        answer = original_check(*args, **kwargs)
        if kwargs.get("phase") == "in_loop":
            checks.append(len(builds))
            if len(checks) == 1:
                return False, answer[1]
        return answer

    monkeypatch.setattr(module, xc_name, observe_xc)
    monkeypatch.setattr(module, fock_name, observe_fock)
    monkeypatch.setattr(module, "bipole_terminal_check", refuse_once)
    result = getattr(module, "run_pbc_bipole_" + driver)(
        system, basis, monkhorst_pack(system, list(mesh)), options,
        ewald_precision=1e-6, progress=False,
        xc_density_domain=(PeriodicXCDensityDomain.PERIODIC_LATTICE
                           if external else PeriodicXCDensityDomain.AUTO),
        dft_plus_u=([vq.HubbardSite(0, 0, U_ev=2.)]
                    if functional.endswith("+u") else None),
    )
    if capped:
        assert not result.converged
        assert result.n_iter == 3
        assert checks == [4]
        assert builds == [True, True, True, False]
        assert len(calls) == len(builds)
        return
    assert result.converged
    assert len(checks) == 2
    assert builds.count(False) == 2  # Both exact checks remain mandatory.
    assert len(builds) == result.n_iter + 2
    xc_by_build = dict(calls)
    first_exact = checks[0]
    assert first_exact in xc_by_build and checks[-1] in xc_by_build
    if damped or external:
        assert len(calls) == len(builds)
        if damped:
            assert xc_by_build[first_exact + 1] != xc_by_build[first_exact]
        if external:
            assert len(provider.calls) == len(calls)
    else:
        assert len(calls) == len(builds) - 1
        assert first_exact + 1 not in xc_by_build


@pytest.mark.parametrize("driver", ["rks", "uks"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("functional", ["pbe", "pbe0", "pbe0+u", "external"])
@pytest.mark.parametrize("damped", [False, True])
def test_refused_confirmation_reuses_only_unchanged_xc(
    monkeypatch, driver, mesh, functional, damped,
):
    _check_refused_confirmation_xc(monkeypatch, driver, mesh, functional, damped)


@pytest.mark.parametrize("driver", ["rks", "uks"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
def test_refused_confirmation_at_cap_keeps_exact_xc(monkeypatch, driver, mesh):
    _check_refused_confirmation_xc(
        monkeypatch, driver, mesh, "pbe0", False, capped=True,
    )


def _run(
    progress, max_iter: int = 40, use_diis: bool = True,
    driver: str = "rhf", options=None,
):
    import importlib
    from vibeqc._vibeqc_core import PeriodicKSOptions

    lattice = np.eye(3) * 7.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    opts = options
    if opts is None:
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


@pytest.mark.parametrize("driver", ["rks", "uks"])
@pytest.mark.parametrize("max_iter", [1, 40], ids=["capped", "converged"])
def test_final_density_reuses_xc_from_exact_fock(monkeypatch, driver, max_iter):
    """XC quadrature belongs to the exact build, including a capped return."""
    import importlib
    from contextlib import contextmanager

    module = importlib.import_module("vibeqc.pbc_bipole_" + driver)
    name = "build_xc_periodic" + ("_uks" if driver == "uks" else "") + "_kpoints"
    original_xc = getattr(module, name)
    original_phase = module.bipole_final_density_phase
    final_density_complete = False
    energies = []

    @contextmanager
    def observed_phase(*args, **kwargs):
        nonlocal final_density_complete
        with original_phase(*args, **kwargs):
            yield
        final_density_complete = True

    def observed_xc(*args, **kwargs):
        assert not final_density_complete, (
            "XC was recomputed after final-density Fock evaluation"
        )
        xc = original_xc(*args, **kwargs)
        energies.append(float(xc.e_xc))
        return xc

    monkeypatch.setattr(module, "bipole_final_density_phase", observed_phase)
    monkeypatch.setattr(module, name, observed_xc)
    result = _run(False, driver=driver, max_iter=max_iter)
    assert final_density_complete
    assert bool(result.converged) is (max_iter > 1)
    assert energies
    assert result.e_xc == pytest.approx(energies[-1], rel=0.0, abs=1e-14)
    rendered = vq.format_scf_trace(result, include_banner=False)
    xc_line = next(line for line in rendered.splitlines()
                   if "Exchange-correlation (XC)" in line)
    assert float(xc_line.split()[-1]) == pytest.approx(result.e_xc, abs=5e-10)


@pytest.mark.parametrize("driver,multiplicity", [("rks", 1), ("uks", 1), ("uks", 2)])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize("exit_mode", ["capped", "converged", "refused_confirmation"])
def test_smeared_terminal_energy_uses_committed_density_entropy(
    monkeypatch, driver, multiplicity, mesh, exit_mode,
):
    """Mermin entropy belongs to D, including before exact convergence (#383).

    Recover the occupations independently as eigenvalues of S**(1/2) D S**(1/2).
    Final Fock canonicalization must not pair E[D] with the next density's
    entropy. The capped case makes that distinction numerically substantial.
    """
    import importlib
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.pbc_bipole_common import bvk_torus_density_matrices

    atoms = [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 2.6]),
    ]
    if multiplicity == 2:
        atoms.append(vq.Atom(1, [2.4, 0., 0.6]))
    system = vq.PeriodicSystem(3, np.eye(3) * 12.0, atoms)
    system.multiplicity = multiplicity
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, mesh, use_symmetry=False)
    opts = PeriodicKSOptions()
    opts.functional = "pbe"
    opts.initial_guess = InitialGuess.HCORE
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.max_iter = 1 if exit_mode == "capped" else 80
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.smearing_temperature = 0.1
    opts.grid.n_radial = 24
    opts.grid.n_theta = 8
    opts.grid.n_phi = 16
    module = importlib.import_module("vibeqc.pbc_bipole_" + driver)
    run = getattr(module, "run_pbc_bipole_" + driver)
    run_kwargs = dict(progress=False, use_incremental_fock=False, ewald_precision=1e-6)
    if exit_mode == "refused_confirmation":
        baseline = run(system, basis, kmesh, opts, **run_kwargs)
        assert baseline.converged
        opts.max_iter = baseline.n_iter
    check = module.bipole_terminal_check
    exact = {}

    def record_check(*args, **kwargs):
        exact[kwargs["phase"]] = kwargs["exact_objective"]
        verdict, details = check(*args, **kwargs)
        if exit_mode == "refused_confirmation" and kwargs["phase"] == "in_loop":
            verdict = False
        return verdict, details

    monkeypatch.setattr(module, "bipole_terminal_check", record_check)
    result = run(system, basis, kmesh, opts, **run_kwargs)
    assert result.converged == (exit_mode == "converged")
    densities = ([result.density] if driver == "rks" else
                 [result.density_alpha, result.density_beta])
    degeneracy = 2.0 if driver == "rks" else 1.0
    entropy = 0.0
    for density in densities:
        d_k = bvk_torus_density_matrices(density, result.kpoints_cart, mesh)
        for weight, overlap, block in zip(result.kpoint_weights, result.overlap, d_k):
            s, u = np.linalg.eigh(overlap)
            sqrt_s = (u * np.sqrt(s)) @ u.conj().T
            f = np.linalg.eigvalsh(sqrt_s @ block @ sqrt_s) / degeneracy
            assert np.min(f) >= -1e-12 and np.max(f) <= 1.0 + 1e-12
            f = np.clip(f, 1e-15, 1.0 - 1e-15)
            entropy -= weight * degeneracy * np.sum(f*np.log(f) + (1-f)*np.log1p(-f))
    assert entropy > 1e-3  # A genuinely fractional state, not an integer control.
    assert result.entropy == pytest.approx(entropy, rel=0., abs=1e-12)
    assert result.free_energy == pytest.approx(
        result.energy - opts.smearing_temperature * entropy, rel=0., abs=1e-13,
    )
    if exit_mode != "capped":
        assert exact["in_loop"] == exact["post_loop"]


@pytest.mark.parametrize("driver", ["rks", "uks"])
@pytest.mark.parametrize("radius", [3.0, 8.0])
@pytest.mark.parametrize("periodic_grid", [True, False])
def test_auto_xc_keeps_density_reach_separate_from_grid_partition_radius(
    monkeypatch, driver, radius, periodic_grid,
):
    """The explicit XC density reach is independent of grid partition weights."""
    import importlib
    from vibeqc import _vibeqc_core as core

    module = importlib.import_module("vibeqc.pbc_bipole_" + driver)
    options = core.PeriodicKSOptions()
    options.use_periodic_becke = periodic_grid
    options.becke_image_radius_bohr = radius
    options.lattice_opts.becke_image_radius_bohr = 1.25
    options.grid.n_radial = 8
    options.grid.n_theta = 5
    options.grid.n_phi = 8
    grid_radii = []
    xc_radii = []
    original_grid = module.build_periodic_becke_grid
    xc_name = "build_xc_periodic" + ("_uks" if driver == "uks" else "") + "_kpoints"
    original_xc = getattr(module, xc_name)

    def observed_grid(*args, **kwargs):
        grid_radii.append(kwargs["image_radius_bohr"])
        return original_grid(*args, **kwargs)

    def observed_xc(*args, **kwargs):
        assert args[-1] == core.PeriodicXCDensityDomain.AUTO
        xc_radii.append(args[-2].becke_image_radius_bohr)
        return original_xc(*args, **kwargs)

    monkeypatch.setattr(module, "build_periodic_becke_grid", observed_grid)
    monkeypatch.setattr(module, xc_name, observed_xc)
    _run(False, driver=driver, max_iter=1, options=options)
    assert grid_radii == ([radius] if periodic_grid else [])
    assert xc_radii
    assert xc_radii == [1.25] * len(xc_radii)


@pytest.mark.parametrize("driver", ["rhf", "uhf", "rks", "uks"])
def test_final_density_domain_and_raw_components_316(monkeypatch, driver):
    """One physical settings record and raw returned-density sums across builds.

    This is a native/SCF acceptance witness. It is not part of the standalone
    source-only contract suite and requires calculation admission.
    """
    import importlib
    from vibeqc._vibeqc_core import PeriodicKSOptions

    module = importlib.import_module(
        "vibeqc.pbc_bipole" + ("" if driver == "rhf" else "_" + driver)
    )
    unrestricted = driver in ("uhf", "uks")
    builder = "build_bipole_" + ("unrestricted" if unrestricted else "restricted") + "_fock"
    original_fock = getattr(module, builder)
    original_one = module.build_bipole_one_electron_lattice
    one_electron = []
    builds = []

    def blocks(operator):
        return {tuple(int(x) for x in cell.index): np.asarray(block).copy()
                for cell, block in zip(operator.cells, operator.blocks)}

    def observe_one(*args, **kwargs):
        operators = original_one(*args, **kwargs)
        one_electron.append(operators)
        return operators

    def observe_fock(ctx, *densities, **kwargs):
        ctx.resolved_domain.require_fock_context(ctx)
        saved = [blocks(density) for density in densities]
        built = original_fock(ctx, *densities, **kwargs)
        builds.append((ctx.resolved_domain, kwargs.get("use_incremental", True), saved, built))
        return built

    monkeypatch.setattr(module, "build_bipole_one_electron_lattice", observe_one)
    monkeypatch.setattr(module, builder, observe_fock)
    opts = PeriodicKSOptions() if driver in ("rks", "uks") else PeriodicRHFOptions()
    assert opts.lattice_opts.pair_complete_1e is False  # Shared native default.
    result = _run(False, driver=driver, options=opts)
    assert result.converged
    assert result.resolved_domain.policy == "physical_pairs"
    assert builds and all(row[0] is result.resolved_domain for row in builds)
    assert any(row[1] for row in builds)
    assert any(not row[1] for row in builds)
    assert not builds[-1][1]
    returned = ([result.density_alpha, result.density_beta] if unrestricted
                else [result.density])
    for observed, density in zip(builds[-1][2], returned):
        actual = blocks(density)
        assert actual.keys() == observed.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], observed[key])

    # Independently accumulate the real-space block products, preserving the
    # native cell support and raw blocks (no Hermitian projection or clipping).
    def raw_contract(density, operator):
        return sum(float(np.sum(density[key] * value))
                   for key, value in blocks(operator).items())

    densities = [blocks(d) for d in returned]
    total_density = {key: sum(d[key] for d in densities) for key in densities[0]}
    _, kinetic, nuclear, _ = one_electron[-1]
    built = builds[-1][3]
    f2e = ([built.f2e_alpha_real, built.f2e_beta_real] if unrestricted
           else [built.f2e_real])
    e_two = built.e_2e_k_correction + .5 * sum(
        raw_contract(d, f) for d, f in zip(densities, f2e)
    )
    components = result.energy_components[-1]
    e_kin = raw_contract(total_density, kinetic)
    e_ne = raw_contract(total_density, nuclear)
    assert components.e_kinetic == pytest.approx(e_kin, rel=0., abs=1e-12)
    assert components.e_nuclear_attraction == pytest.approx(e_ne, rel=0., abs=1e-12)
    assert components.e_two_electron == pytest.approx(e_two, rel=0., abs=1e-12)
    reconstructed = (e_kin + e_ne + e_two + getattr(result, "e_xc", 0.)
                     + result.e_nuclear + result.e_dft_plus_u
                     + float(result.e_ext_el_spheropole or 0.))
    assert result.energy == pytest.approx(reconstructed, rel=0., abs=1e-12)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("functional,temperature", [("svwn", 0.), ("pbe0", 0.), ("svwn", 0.1)])
@pytest.mark.parametrize("orbital_controls", [False, True], ids=["plain", "mix-shift-mom"])
@pytest.mark.parametrize("capped", [False, True], ids=["confirmed", "capped"])
def test_rks_failed_density_closure_retries_acceleration(
    monkeypatch, mesh, functional, temperature, capped, orbital_controls,
):
    """An inaccurate orbital update must not disable acceleration indefinitely.

    Rotate one occupied/virtual pair on a provisional physical-Fock update.
    The metric-orthonormal candidate then fails density closure even though
    the evaluated state's energy and commutator pass. Retry only the orbital
    update, and require a fresh SCF evaluation before accepting convergence.
    """
    from vibeqc import pbc_bipole_rks as driver
    from vibeqc._vibeqc_core import PeriodicKSOptions

    system = vq.PeriodicSystem(
        3, np.eye(3) * 10.,
        [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, mesh, use_symmetry=False)
    opts = PeriodicKSOptions()
    opts.functional = functional
    opts.initial_guess = InitialGuess.SAD
    opts.lattice_opts.cutoff_bohr = 12.
    opts.lattice_opts.nuclear_cutoff_bohr = 12.
    opts.grid.n_radial = 20
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-8
    opts.smearing_temperature = temperature
    if orbital_controls:
        opts.fock_mixing = 0.25
        opts.level_shift = 0.3
    opts.max_iter = 40
    kwargs = dict(
        ewald_precision=1e-6, use_fock_symmetry_reduce=False,
        use_mom=orbital_controls and temperature == 0.,
    )
    reference = driver.run_pbc_bipole_rks(
        system, basis, kmesh, opts, progress=False, **kwargs,
    )
    assert reference.converged
    opts.max_iter = reference.n_iter + (0 if capped else 1)
    state = dict(eligible=False, injected=False)

    class ObserveIteration(ProgressLogger):
        def iteration(self, iteration, **values):
            state["eligible"] = (
                iteration > 1
                and abs(values["dE"]) < opts.conv_tol_energy
                and values["grad"] < opts.conv_tol_grad
            )
            return super().iteration(iteration, **values)

    diagonalize = driver._diag_in_orth_basis

    def displaced_trial(*args, **kwargs):
        coeffs, energies = diagonalize(*args, **kwargs)
        if state["eligible"] and not state["injected"]:
            state["injected"] = True
            coeffs = coeffs.copy()
            angle = 1e-4
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ])
            coeffs[:, :2] = coeffs[:, :2] @ rotation
        return coeffs, energies

    builds = []
    build = driver.build_bipole_restricted_fock

    def observe_build(*args, **kwargs):
        builds.append(kwargs.get("use_incremental", True))
        return build(*args, **kwargs)

    monkeypatch.setattr(driver, "_diag_in_orth_basis", displaced_trial)
    monkeypatch.setattr(driver, "build_bipole_restricted_fock", observe_build)
    stream = io.StringIO()
    result = driver.run_pbc_bipole_rks(
        system, basis, kmesh, opts,
        progress=ObserveIteration(stream=stream, verbose=4), **kwargs,
    )
    assert state["injected"]
    assert result.converged == (not capped)
    assert result.n_iter == opts.max_iter
    assert len(builds) == result.n_iter + 1
    assert builds.count(False) == 1  # Retry adds no Fock or exact build.
    assert result.energy == pytest.approx(reference.energy, rel=0., abs=1e-10)
    np.testing.assert_allclose(
        result.density.blocks, reference.density.blocks, rtol=0., atol=1e-10,
    )
    assert "density closure failed" in stream.getvalue()


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("functional", ["svwn", "pbe0"])
def test_rks_closed_physical_density_ignores_bad_extrapolation(
    monkeypatch, mesh, functional,
):
    """Keep canonical physical orbitals when the accelerated Fock is garbage."""
    from vibeqc import pbc_bipole_rks as driver
    from vibeqc._vibeqc_core import PeriodicKSOptions

    system = vq.PeriodicSystem(
        3, np.eye(3) * 10.,
        [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, mesh, use_symmetry=False)
    opts = PeriodicKSOptions()
    opts.functional = functional
    opts.initial_guess = InitialGuess.SAD
    opts.lattice_opts.cutoff_bohr = 12.
    opts.lattice_opts.nuclear_cutoff_bohr = 12.
    opts.grid.n_radial = 20
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 40
    kwargs = dict(progress=False, ewald_precision=1e-6, use_fock_symmetry_reduce=False)
    reference = driver.run_pbc_bipole_rks(system, basis, kmesh, opts, **kwargs)
    assert reference.converged
    extrapolate = driver.MultiKPeriodicSCFAccelerator.extrapolate_rhf
    poisoned = []

    def poison_stationary(self, focks, **kwargs):
        candidate = extrapolate(self, focks, **kwargs)
        if max(np.linalg.norm(e) for e in kwargs["error_k_list"]) < opts.conv_tol_grad:
            poisoned.append(True)
            # Reverse the orbital ordering without changing the input Focks.
            return [-np.asarray(f).copy() for f in focks]
        return candidate

    monkeypatch.setattr(
        driver.MultiKPeriodicSCFAccelerator, "extrapolate_rhf", poison_stationary,
    )
    result = driver.run_pbc_bipole_rks(system, basis, kmesh, opts, **kwargs)
    assert poisoned and result.converged
    assert result.n_iter == reference.n_iter
    assert result.energy == pytest.approx(reference.energy, rel=0., abs=1e-10)
    np.testing.assert_allclose(
        result.density.blocks, reference.density.blocks, rtol=0., atol=1e-10,
    )


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("functional,temperature", [("svwn", 0.), ("pbe0", 0.), ("svwn", 0.1)])
@pytest.mark.parametrize("capped", [True, False], ids=["capped", "resumed"])
def test_rks_exact_confirmation_rechecks_density_closure(
    monkeypatch, mesh, functional, temperature, capped,
):
    """A small exact-Fock change can pass stationarity but fail density closure.

    A stretched H2 cell amplifies a tiny diagonal operator change. Inject it
    first at the cold confirmation, then retain it on subsequent builds so a
    resumed SCF must reach the changed operator's actual fixed point.
    """
    from scipy.linalg import eigh
    from vibeqc import _vibeqc_core as core
    from vibeqc import pbc_bipole_rks as driver

    system = vq.PeriodicSystem(
        3, np.eye(3) * 20.,
        [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 6.])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, mesh, use_symmetry=False)
    opts = core.PeriodicKSOptions()
    opts.functional = functional
    opts.smearing_temperature = temperature
    opts.initial_guess = InitialGuess.SAD
    opts.lattice_opts.cutoff_bohr = 12.
    opts.lattice_opts.nuclear_cutoff_bohr = 12.
    opts.grid.n_radial = 20
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 30
    kwargs = dict(progress=False, ewald_precision=1e-6, use_fock_symmetry_reduce=False)
    baseline = driver.run_pbc_bipole_rks(system, basis, kmesh, opts, **kwargs)
    assert baseline.converged
    opts.max_iter = baseline.n_iter + (0 if capped else 8)
    build = driver.build_bipole_restricted_fock
    injected = []
    cold_builds = []

    def changed_exact_operator(*args, **kwargs):
        result = build(*args, **kwargs)
        cold = not kwargs.get("use_incremental", True)
        if cold:
            cold_builds.append(True)
        if injected or cold:
            blocks = [np.asarray(block).copy() for block in result.f2e_real.blocks]
            cells = list(result.f2e_real.cells)
            home = next(i for i, cell in enumerate(cells) if tuple(cell.index) == (0, 0, 0))
            blocks[home] += np.diag([3e-8, -3e-8])
            result.f2e_real = core.make_lattice_matrix_set(2, cells, blocks)
            injected.append(True)
        return result

    monkeypatch.setattr(driver, "build_bipole_restricted_fock", changed_exact_operator)
    result = driver.run_pbc_bipole_rks(system, basis, kmesh, opts, **kwargs)
    assert injected
    # Derive the candidate independently from returned physical F and S,
    # then inverse-transform it with the full mesh weights and cell phases.
    candidates = []
    for fock, overlap, occupations in zip(result.fock, result.overlap, result.occupations):
        _, coeffs = eigh(fock, overlap)
        candidates.append((coeffs * np.asarray(occupations)) @ coeffs.conj().T)
    cart = np.asarray([cell.r_cart for cell in result.density.cells])
    candidate_real = np.einsum(
        "k,kg,kij->gij", np.asarray(kmesh.weights),
        np.exp(-1j * np.asarray(kmesh.kpoints) @ cart.T), np.asarray(candidates),
    )
    closure = float(np.max(np.abs(candidate_real - np.asarray(result.density.blocks))))
    assert result.scf_trace[-1].grad_norm < opts.conv_tol_grad
    if capped:
        assert closure > opts.conv_tol_grad
        assert not result.converged
        assert len(cold_builds) == 1  # Finalization reuses the refused exact build.
    else:
        assert result.converged
        assert result.n_iter > baseline.n_iter
        assert closure < opts.conv_tol_grad


_EXACT_CLOSURE_CASES = [
    ('rhf', None, 0., 'restricted'),
    ('uhf', None, 0., 'alpha'), ('uhf', None, 0., 'beta'),
    ('uhf', None, .1, 'beta'),
    ('uks', 'svwn', 0., 'alpha'), ('uks', 'svwn', 0., 'beta'),
    ('uks', 'pbe0', 0., 'beta'), ('uks', 'svwn', .1, 'beta'),
]


@pytest.mark.parametrize('method,functional,temperature,spin,orbital_controls', [
    (*case, controls) for case in _EXACT_CLOSURE_CASES for controls in (False, True)
    # The thermal UKS mixed/shifted baseline does not converge on the unchanged
    # parent, so it cannot exercise an exact convergence confirmation.
    if not (case[0] == 'uks' and case[2] and controls)
])
@pytest.mark.parametrize('mesh_shape', [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize('capped', [True, False])
def test_hf_uks_exact_confirmation_rechecks_density_closure(
    monkeypatch, method, functional, temperature, spin, mesh_shape, capped,
    orbital_controls,
):
    """Reject inaccurate exact candidates, including separate spin channels.

    Plain SCF must recover; aggressive orbital controls may exhaust the
    bound, but may never claim a state whose physical residuals fail.
    """
    import importlib
    from scipy.linalg import eigh
    from vibeqc import _vibeqc_core as core

    module = importlib.import_module('vibeqc.pbc_bipole' + ('' if method == 'rhf' else '_' + method))
    run = getattr(module, 'run_pbc_bipole_' + method)
    system = vq.PeriodicSystem(
        3, 20. * np.eye(3),
        [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 6.])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    mesh = core.monkhorst_pack(system, mesh_shape, use_symmetry=False)
    opts = core.PeriodicKSOptions() if method == 'uks' else core.PeriodicRHFOptions()
    if method == 'uks':
        opts.functional = functional
        opts.grid.n_radial = 20
    if method != 'rhf':
        opts.smearing_temperature = temperature
    opts.initial_guess = core.InitialGuess.SAD
    opts.lattice_opts.cutoff_bohr = 12.
    opts.lattice_opts.nuclear_cutoff_bohr = 12.
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    if orbital_controls:
        opts.fock_mixing = .25
        opts.level_shift = .3
    kwargs = dict(progress=False, ewald_precision=1e-6,
                  use_fock_symmetry_reduce=False,
                  use_mom=orbital_controls and temperature == 0.)
    reference = run(system, basis, mesh, opts, **kwargs)
    assert reference.converged
    opts.max_iter = reference.n_iter + (0 if capped else 80)
    name = 'build_bipole_restricted_fock' if method == 'rhf' else 'build_bipole_unrestricted_fock'
    build = getattr(module, name)
    injected = []
    cold_builds = []
    delta = np.diag([6e-8, -6e-8]) if temperature else np.diag([3e-8, -3e-8])

    def changed_exact_operator(*args, **kwargs):
        result = build(*args, **kwargs)
        cold = not kwargs.get('use_incremental', True)
        if cold:
            cold_builds.append(True)
        if injected or cold:
            field = 'f2e_real' if method == 'rhf' else 'f2e_' + spin + '_real'
            matrix = getattr(result, field)
            cells = list(matrix.cells)
            blocks = [np.asarray(block).copy() for block in matrix.blocks]
            home = next(i for i, cell in enumerate(cells) if tuple(cell.index) == (0, 0, 0))
            blocks[home] += delta
            setattr(result, field, core.make_lattice_matrix_set(2, cells, blocks))
            if method == 'uhf':
                # UHF consumes the shared builder's assembled F(k) directly.
                # Restricted HF and UKS assemble it from F2e in their wrappers.
                field = 'f_' + spin + '_k_list'
                setattr(result, field, [np.asarray(f) + delta for f in getattr(result, field)])
            injected.append(True)
        return result

    monkeypatch.setattr(module, name, changed_exact_operator)
    result = run(system, basis, mesh, opts, **kwargs)
    assert injected
    channels = ([(result.fock, result.occupations, result.density)] if method == 'rhf'
                else [(getattr(result, 'fock_' + label),
                       getattr(result, 'occupations_' + label),
                       getattr(result, 'density_' + label)) for label in ('alpha', 'beta')])
    closures = []
    for focks, occupations, density in channels:
        candidates = []
        for fock, overlap, occ in zip(focks, result.overlap, occupations):
            _, coeffs = eigh(fock, overlap)
            candidates.append((coeffs * np.asarray(occ)) @ coeffs.conj().T)
        cart = np.asarray([cell.r_cart for cell in density.cells])
        candidate_real = np.einsum(
            'k,kg,kij->gij', np.asarray(mesh.weights),
            np.exp(-1j * np.asarray(mesh.kpoints) @ cart.T), np.asarray(candidates),
        )
        closures.append(float(np.max(np.abs(candidate_real - np.asarray(density.blocks)))))
    if capped:
        assert result.scf_trace[-1].grad_norm < opts.conv_tol_grad
        assert max(closures) > opts.conv_tol_grad
        assert not result.converged
        assert len(cold_builds) == 1
    else:
        assert result.n_iter > reference.n_iter
        if result.converged:
            assert result.scf_trace[-1].grad_norm < opts.conv_tol_grad
            assert max(closures) < opts.conv_tol_grad
        else:
            assert orbital_controls
            assert result.n_iter == opts.max_iter
            assert (result.scf_trace[-1].grad_norm >= opts.conv_tol_grad
                    or max(closures) >= opts.conv_tol_grad
                    or abs(result.scf_trace[-1].delta_e) >= opts.conv_tol_energy)
