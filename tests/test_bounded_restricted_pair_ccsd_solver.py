"""Ragged projected-CCSD numerical oracles, not physical PNO constructors."""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_ccsd_target import _legacy
from tests.test_bounded_restricted_ccsd_solver import _problem as _dense_problem
from tests.test_bounded_restricted_ccsd_solver import _run as _dense_run
from tests.test_bounded_restricted_ccsd_solver import _energy


def _problem(o=2, n=3, *, singles_ranks=None, pair_ranks=None, rotated=False,
             nonzero_initial=False, seed=734):
    p = _dense_problem(o, n, seed=seed)
    rng = np.random.default_rng(seed + 390)
    labels = list(combinations_with_replacement(range(o), 2))
    sr = [n]*o if singles_ranks is None else list(singles_ranks)
    pr = [n]*len(labels) if pair_ranks is None else list(pair_ranks)
    def frame(rank):
        q = np.linalg.qr(rng.normal(size=(n, n)))[0] if rotated else np.eye(n)
        return np.ascontiguousarray(q[:, :rank])
    sc, pc = [frame(rank) for rank in sr], [frame(rank) for rank in pr]
    st = [np.zeros(rank) for rank in sr]
    pt = [np.zeros((rank, rank)) for rank in pr]
    if nonzero_initial:
        st = [np.ascontiguousarray(.001*rng.normal(size=rank)) for rank in sr]
        for k, ((i, j), rank) in enumerate(zip(labels, pr, strict=True)):
            t = .001*rng.normal(size=(rank, rank))
            pt[k] = np.ascontiguousarray((t+t.T)/2 if i == j else t)
    p.update(n=n, labels=labels, sr=sr, pr=pr, sc=sc, st=st, pc=pc, pt=pt,
        foo=np.ascontiguousarray(p["fock"][:o, :o]),
        fvv=np.ascontiguousarray(p["fock"][o:, o:]),
        fov=np.ascontiguousarray(p["fock"][:o, o:]))
    return p


def _options(**changes):
    options = core._BoundedRestrictedPairCCSDSolverOptions()
    options.maximum_iterations = 80
    options.denominator_floor = 1e-8
    options.singles_residual_tolerance = options.doubles_residual_tolerance = 1e-12
    options.energy_tolerance = 1e-13
    options.coefficient_orthogonality_tolerance = 1e-12
    options.maximum_diagonal_update_antisymmetry_norm = 1e-12
    options.maximum_integral_work_units_per_call = 1
    for field, value in changes.items():
        setattr(options, field, value)
    return options


def _inventory(**changes):
    inventory = core._BoundedRestrictedPairCCSDSolverInventory()
    inventory.numerical_replicas = 1
    inventory.fixed_backend_margin_bytes_per_replica = 65536
    for field, value in changes.items():
        setattr(inventory, field, value)
    return inventory


_CAP_FIELDS = dict(
    maximum_occupied_count="n_occupied", maximum_common_virtual_dimension="common_virtual_dimension",
    maximum_pair_count="pair_count", maximum_singles_rank="maximum_singles_rank", maximum_pair_rank="maximum_pair_rank",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
    maximum_per_replica_inventoried_bytes="per_replica_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_inventoried_bytes",
    maximum_integral_calls="integral_calls_upper_bound", maximum_singles_calls="singles_calls_upper_bound",
    maximum_doubles_calls="doubles_calls_upper_bound", maximum_work_units="work_units_upper_bound",
)


def _caps(plan):
    caps = core._BoundedRestrictedPairCCSDSolverCaps()
    for field, report in _CAP_FIELDS.items():
        setattr(caps, field, getattr(plan, report))
    return caps


def _upper(p, options=None, inventory=None, transient=0):
    return core._plan_bounded_restricted_pair_ccsd_solver_upper(
        p["o"], p["n"], max(p["sr"]), max(p["pr"]),
        _options() if options is None else options, _inventory() if inventory is None else inventory,
        p["eri"].nbytes, transient,
    )


def _arguments(p):
    return p["foo"], p["fvv"], p["fov"], p["sc"], p["st"], p["pc"], p["pt"], p["eri"]


def _plan(p, options=None, inventory=None, caps=None, transient=0):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_upper(p, options, inventory, transient))
    return core._plan_bounded_restricted_pair_ccsd_solver(
        *_arguments(p), options, inventory, caps, transient,
    )


def _run(p, *, options=None, inventory=None, caps=None, callback=None, **kwargs):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_plan(p, options, inventory, transient=kwargs.get("provider_transient_bytes", 0)))
    return core._bounded_restricted_pair_ccsd_solve_diagnostic(
        *_arguments(p), options, inventory, caps, callback, **kwargs,
    )


def _local_result(p, result):
    return ([result.singles_copy(i) for i in range(p["o"])],
            [result.pair_copy(i, j) for i, j in p["labels"]])


def _expand(p, st, pt):
    o, n = p["o"], p["n"]
    t1, t2 = np.zeros((o, n)), np.zeros((o, o, n, n))
    for i, (c, t) in enumerate(zip(p["sc"], st, strict=True)):
        t1[i] = c@t
    for (i, j), c, t in zip(p["labels"], p["pc"], pt, strict=True):
        block = c@t@c.T
        if i == j:
            # Same physical diagonal-pair symmetry convention as the reader:
            # evaluate the upper representative, not an averaged operator.
            block = np.triu(block) + np.triu(block, 1).T
        t2[i, j], t2[j, i] = block, block.T
    fock = np.block([[p["foo"], p["fov"]], [p["fov"].T, p["fvv"]]])
    return dict(p, fock=np.ascontiguousarray(fock), t1=np.ascontiguousarray(t1), t2=np.ascontiguousarray(t2))


def _project(p, r1, r2):
    return ([c.T@r1[i] for i, c in enumerate(p["sc"])],
        [c.T@r2[i, j]@c for (i, j), c in zip(p["labels"], p["pc"], strict=True)])


def _projected_norms(p, rs, rp):
    max_s = max((float(np.max(abs(r), initial=0)) for r in rs), default=0)
    max_p = max((float(np.max(abs(r), initial=0)) for r in rp), default=0)
    norm_s = np.sqrt(sum(float(np.sum(r*r)) for r in rs))
    norm_p = np.sqrt(sum((1 if i == j else 2)*float(np.sum(r*r))
        for (i, j), r in zip(p["labels"], rp, strict=True)))
    return max_s, max_p, norm_s, norm_p


def _spin_residual(case):
    from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_energy, so_residuals

    o, v = case["o"], case["v"]
    n = o+v
    f = np.zeros((2*n, 2*n))
    f[0::2, 0::2] = f[1::2, 1::2] = case["fock"]
    off = f.copy()
    np.fill_diagonal(off, 0)
    eri = _spin_orbital_eri(case["factors"], n)
    occ, virt = slice(0, 2*o), slice(2*o, 2*n)
    eo, ev = np.diag(f)[occ], np.diag(f)[virt]
    d1 = eo[:, None]-ev[None, :]
    d2 = eo[:, None, None, None]+eo[None, :, None, None]-ev[None, None, :, None]-ev[None, None, None, :]
    t1 = np.zeros((2*o, 2*v))
    t1[0::2, 0::2] = t1[1::2, 1::2] = case["t1"]
    t2 = np.zeros((2*o, 2*o, 2*v, 2*v))
    for si in (0, 1):
        for sj in (0, 1):
            t2[si::2, sj::2, si::2, sj::2] += case["t2"]
            t2[si::2, sj::2, sj::2, si::2] -= case["t2"].transpose(0, 1, 3, 2)
    r1, r2 = so_residuals(f, off, eri, t1, t2, occ, virt, d1, d2)
    return r1[0::2, 0::2], r2[0::2, 1::2, 0::2, 1::2], so_energy(f, eri, t1, t2, occ, virt)


def _jacobi_oracle(p, *, iterations=100, fixed_iterations=False, spin=False):
    """Expand all ragged inputs, use independent residual, THEN project."""
    st, pt = [t.copy() for t in p["st"]], [t.copy() for t in p["pt"]]
    sd = [np.diag(c.T@p["fvv"]@c)-p["foo"][i, i] for i, c in enumerate(p["sc"])]
    pd = []
    for (i, j), c in zip(p["labels"], p["pc"], strict=True):
        e = np.diag(c.T@p["fvv"]@c)
        pd.append(e[:, None]+e[None, :]-p["foo"][i, i]-p["foo"][j, j])
    previous = None
    for iteration in range(1, iterations+1):
        case = _expand(p, st, pt)
        if spin:
            r1, r2, energy = _spin_residual(case)
        else:
            r1, r2 = _legacy(case)
            energy = _energy(case, case["t1"], case["t2"])
        rs, rp = _project(p, r1, r2)
        norms = _projected_norms(p, rs, rp)
        converged = previous is not None and abs(energy-previous) < 1e-14 and max(norms[:2]) < 1e-13
        if (not fixed_iterations and converged) or iteration == iterations:
            if not fixed_iterations:
                assert converged, "tiny independently projected CCSD Jacobi did not converge"
            return st, pt, energy, (r1, r2), norms
        st = [np.ascontiguousarray(t-r/d) for t, r, d in zip(st, rs, sd, strict=True)]
        pt = [np.ascontiguousarray(t-r/d) for t, r, d in zip(pt, rp, pd, strict=True)]
        for k, (i, j) in enumerate(p["labels"]):
            if i == j:
                pt[k] = np.ascontiguousarray((pt[k]+pt[k].T)/2)
        previous = energy
    raise AssertionError("unreachable oracle loop")


@pytest.mark.parametrize("o,n", [(1, 1), (1, 3), (2, 2), (3, 2), (4, 1)])
def test_identity_full_rank_matches_dense_ccsd_and_returned_residual(o, n):
    p = _problem(o, n)
    result = _run(p)
    assert result.final_snapshot.converged
    st, pt = _local_result(p, result)
    case = _expand(p, st, pt)
    if o <= 3:  # Existing direct dense-input safety cap stays unchanged.
        dense = _dense_run(_expand(p, p["st"], p["pt"]), supplied=True)
        assert dense.final_snapshot.converged
        np.testing.assert_allclose(case["t1"], dense.t1, atol=2e-12, rtol=1e-10)
        np.testing.assert_allclose(case["t2"], dense.t2, atol=2e-12, rtol=1e-10)
        assert result.final_snapshot.correlation_energy == pytest.approx(dense.final_snapshot.correlation_energy, abs=2e-13)
    expected_s, expected_p, energy, _, _ = _jacobi_oracle(p)
    for actual, expected in zip(st+pt, expected_s+expected_p, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-10)
    r1, r2 = _legacy(case)
    norms = _projected_norms(p, *_project(p, r1, r2))
    progress = result.final_snapshot
    for name, expected in zip(("singles_max_residual", "doubles_max_residual", "singles_residual_norm", "doubles_residual_norm"), norms, strict=True):
        assert getattr(progress, name) == pytest.approx(expected, abs=3e-14)
    assert progress.correlation_energy == pytest.approx(energy, abs=2e-13)
    assert progress.target_evaluations == progress.iteration*len(p["labels"])
    leaf_calls = result.memory.target.kernel.integral_calls_upper_bound-2*o*n**3+n**2
    assert progress.integral_calls == progress.iteration*(len(p["labels"])*leaf_calls+2*o**2*n**2)
    for i, j in p["labels"]:
        np.testing.assert_array_equal(result.pair_copy(j, i), result.pair_copy(i, j).T)
        if i == j:
            np.testing.assert_array_equal(result.pair_copy(i, i), result.pair_copy(i, i).T)


@pytest.mark.parametrize("singles_ranks,pair_ranks", [([3, 1], [1, 2, 0]), ([0, 2], [2, 1, 3]), ([2, 3], [0, 0, 0])])
def test_unequal_rotated_spaces_converge_to_independent_projected_fixed_point(singles_ranks, pair_ranks):
    p = _problem(singles_ranks=singles_ranks, pair_ranks=pair_ranks, rotated=True, nonzero_initial=True)
    result = _run(p)
    assert result.final_snapshot.converged
    st, pt = _local_result(p, result)
    expected_s, expected_p, energy, raw, _ = _jacobi_oracle(p)
    for actual, expected in zip(st+pt, expected_s+expected_p, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=1e-9)
    assert result.final_snapshot.correlation_energy == pytest.approx(energy, abs=2e-13)
    # Projected fixed point is NOT a false full-CCSD residual certificate.
    assert max(np.max(abs(raw[0])), np.max(abs(raw[1]))) > 1e-5
    spin_r1, spin_r2, spin_energy = _spin_residual(_expand(p, st, pt))
    norms = _projected_norms(p, *_project(p, spin_r1, spin_r2))
    assert max(norms[:2]) < 1.1e-12
    assert result.final_snapshot.correlation_energy == pytest.approx(spin_energy, abs=2e-14)


def test_full_rank_independent_singles_and_pair_rotations_preserve_common_solution():
    plain, rotated = _problem(), _problem(rotated=True)
    a, b = _run(plain), _run(rotated)
    assert a.final_snapshot.converged and b.final_snapshot.converged
    x = _expand(plain, *_local_result(plain, a))
    y = _expand(rotated, *_local_result(rotated, b))
    np.testing.assert_allclose(x["t1"], y["t1"], atol=3e-12, rtol=1e-10)
    np.testing.assert_allclose(x["t2"], y["t2"], atol=3e-12, rtol=1e-10)
    assert a.final_snapshot.correlation_energy == pytest.approx(b.final_snapshot.correlation_energy, abs=3e-13)


@pytest.mark.parametrize("truncated", [False, True])
def test_pure_python_spin_orbital_expand_project_jacobi_oracle(truncated):
    p = _problem(2, 2, singles_ranks=[2, 1] if truncated else None,
        pair_ranks=[1, 2, 0] if truncated else None, rotated=truncated)
    result = _run(p)
    expected_s, expected_p, energy, _, _ = _jacobi_oracle(p, spin=True)
    for actual, expected in zip(sum(_local_result(p, result), []), expected_s+expected_p, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=1e-9)
    assert result.final_snapshot.correlation_energy == pytest.approx(energy, abs=2e-13)


def test_zero_pair_ranks_do_not_remove_the_full_common_singles_product_from_energy():
    p = _problem(pair_ranks=[0, 0, 0], nonzero_initial=True)
    result = _run(p, options=_options(maximum_iterations=1))
    case = _expand(p, p["st"], p["pt"])
    expected = _energy(case, case["t1"], case["t2"])
    one_body_only = 2*np.einsum("ia,ia->", p["fov"], case["t1"])
    assert abs(expected-one_body_only) > 1e-10
    assert result.final_snapshot.correlation_energy == pytest.approx(expected, abs=2e-16)
    assert not result.final_snapshot.converged


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_iteration_exhaustion_returns_only_the_last_evaluated_ragged_snapshot(iterations):
    p = _problem(singles_ranks=[3, 1], pair_ranks=[1, 2, 0], rotated=True, nonzero_initial=True)
    events = []
    result = _run(p, options=_options(maximum_iterations=iterations), callback=events.append)
    assert not result.final_snapshot.converged
    assert result.final_snapshot.iteration == iterations
    assert [event.iteration for event in events] == list(range(1, iterations+1))
    expected_s, expected_p, energy, _, norms = _jacobi_oracle(p, iterations=iterations, fixed_iterations=True)
    for actual, expected in zip(sum(_local_result(p, result), []), expected_s+expected_p, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=2e-12)
    assert result.final_snapshot.correlation_energy == pytest.approx(energy, abs=3e-16)
    assert result.final_snapshot.singles_max_residual == pytest.approx(norms[0], abs=3e-15)
    assert result.final_snapshot.doubles_max_residual == pytest.approx(norms[1], abs=3e-15)


def test_converged_snapshot_does_not_compute_an_overflowing_unneeded_update():
    """Range-edge algebra witness; these loose controls are not chemistry defaults."""
    p = _problem(1, 1)
    p["factors"][:] = 0
    p["eri"][:] = 0
    p["foo"][:] = -5e-309
    p["fvv"][:] = 5e-309
    p["fov"][:] = 1e-154
    controls = dict(denominator_floor=1e-309, singles_residual_tolerance=1e155,
        doubles_residual_tolerance=1e155, energy_tolerance=3.0)
    terminal = _run(p, options=_options(maximum_iterations=2, **controls))
    assert terminal.final_snapshot.converged and terminal.final_snapshot.iteration == 2
    assert np.isfinite(terminal.final_snapshot.correlation_energy)
    assert terminal.final_snapshot.correlation_energy == pytest.approx(-2.0, abs=2e-14)
    assert terminal.singles_amplitude(0, 0) == pytest.approx(-1e154, rel=2e-14)
    # The current T and residual are finite and admitted. R/delta for a THIRD
    # snapshot would overflow, but it is irrelevant once snapshot2 converges.
    spare_iteration = _run(p, options=_options(maximum_iterations=3, **controls))
    assert spare_iteration.final_snapshot.converged and spare_iteration.final_snapshot.iteration == 2
    assert spare_iteration.payload_sha256 == terminal.payload_sha256
    np.testing.assert_array_equal(spare_iteration.singles_copy(0), terminal.singles_copy(0))
    np.testing.assert_array_equal(spare_iteration.pair_copy(0, 0), terminal.pair_copy(0, 0))


def test_all_zero_spaces_are_included_and_give_an_empty_projected_fixed_point():
    p = _problem(singles_ranks=[0, 0], pair_ranks=[0, 0, 0])
    result = _run(p)
    assert result.final_snapshot.converged and result.final_snapshot.iteration == 2
    assert result.final_snapshot.correlation_energy == 0
    assert result.minimum_denominator == result.maximum_denominator == 0
    assert result.final_snapshot.singles_residual_norm == result.final_snapshot.doubles_residual_norm == 0
    assert result.final_snapshot.target_evaluations == 2*3
    assert result.final_snapshot.integral_calls > 0
    assert all(x.size == 0 for x in sum(_local_result(p, result), []))
    with pytest.raises(IndexError, match="range"):
        result.singles_amplitude(0, 0)
    with pytest.raises(IndexError, match="range"):
        result.doubles_amplitude(0, 1, 0, 0)


def test_exact_ragged_inventory_and_count_only_upper_bound():
    p = _problem(singles_ranks=[3, 1], pair_ranks=[1, 2, 0])
    inventory = _inventory(numerical_replicas=3, external_node_bytes=197, other_live_bytes_per_replica=53)
    plan = _plan(p, inventory=inventory, transient=71)
    upper = _upper(p, inventory=inventory, transient=71)
    o, n, pairs = p["o"], p["n"], len(p["labels"])
    s, d = sum(p["sr"]), sum(rank**2 for rank in p["pr"])
    assert not plan.uniform_rank_upper_bound and upper.uniform_rank_upper_bound
    assert plan.amplitude_snapshot_bytes == plan.candidate_snapshot_bytes == 8*(s+d)
    assert plan.projected_fock_diagonal_bytes == 8*(s+sum(p["pr"]))
    assert plan.retained_record_bytes == 16*(o+pairs)
    assert plan.owned_snapshot_table_bytes == plan.borrowed_input_table_bytes == plan.amplitudes.borrowed_table_bytes
    borrowed = sum(a.nbytes for key in ("sc", "st", "pc", "pt") for a in p[key])
    assert plan.borrowed_initial_numerical_bytes == borrowed
    assert plan.borrowed_fock_bytes == 8*(o**2+n**2+o*n)
    assert plan.provider_retained_numerical_bytes == p["eri"].nbytes
    assert plan.provider_transient_numerical_bytes == 71
    assert plan.output_numerical_bytes == 8*(s+d)+16*(o+pairs)
    assert plan.peak_owned_numerical_bytes == 16*(s+d)+16*(o+pairs)+plan.projected_fock_diagonal_bytes+plan.target_owned_peak_bytes
    expected_per_replica = (plan.peak_owned_numerical_bytes+borrowed+plan.borrowed_fock_bytes+p["eri"].nbytes+71
        +plan.owned_snapshot_table_bytes+plan.borrowed_input_table_bytes+plan.control_storage_reservation_bytes+53+65536)
    assert plan.per_replica_inventoried_bytes == expected_per_replica
    assert plan.required_node_inventoried_bytes == 197+3*expected_per_replica
    for report in set(_CAP_FIELDS.values()):
        assert getattr(plan, report) <= getattr(upper, report)
    result = _run(p, inventory=inventory, provider_transient_bytes=71)
    assert result.final_snapshot.charged_work_units <= plan.work_units_upper_bound


@pytest.mark.parametrize("field", list(_CAP_FIELDS))
def test_each_exact_cap_minus_one_rejects_before_nan_scan_or_integral_callback(field):
    p = _problem()
    caps = _caps(_plan(p))
    setattr(caps, field, getattr(caps, field)-1)
    p["foo"][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap|ceiling|bound|limit"):
        _run(p, caps=caps, fail_before_call=0)


def test_hidden_provider_transient_and_external_node_bytes_are_rejected():
    p = _problem()
    caps = _caps(_plan(p))
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(p, caps=caps, provider_transient_bytes=1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(p, caps=caps, inventory=_inventory(external_node_bytes=1))


@pytest.mark.parametrize("fail_before", [0, 1, 100])
def test_integral_callback_exception_and_nonfinite_value_abort(fail_before):
    p = _problem()
    with pytest.raises(RuntimeError, match="injected integral callback failure"):
        _run(p, fail_before_call=fail_before)
    p["eri"][:] = np.inf
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="finite"):
        _run(p)


@pytest.mark.parametrize("field", ["foo", "sc", "pc", "st", "pt"])
def test_final_callback_mutation_of_initial_input_is_rejected(field):
    p = _problem(nonzero_initial=True)
    def mutate(event):
        if not event.converged:
            return
        if field == "foo":
            p[field][0, 0] += .01
        elif field in ("sc", "pc"):
            p[field][0][:, 0] *= -1
        elif field == "st":
            p[field][0][0] += .01
        else:
            p[field][1][0, 0] += .01
    with pytest.raises((ValueError, RuntimeError), match="changed|snapshot|immutable"):
        _run(p, callback=mutate)


def test_progress_copy_exception_pinned_list_owners_and_control_snapshot():
    p = _problem()
    def cancel(_event):
        raise RuntimeError("intentional pair CCSD progress cancellation")
    with pytest.raises(RuntimeError, match="intentional pair CCSD"):
        _run(p, callback=cancel)
    expected = _run(p)
    events = []
    options, inventory = _options(), _inventory()
    caps = _caps(_plan(p, options, inventory))
    def replace_and_mutate_controls(event):
        events.append(event)
        for key in ("sc", "st", "pc", "pt"):
            p[key][:] = [np.full_like(a, np.nan) for a in p[key]]
        options.maximum_iterations = 0
        inventory.numerical_replicas = 0
        caps.maximum_work_units = 0
        gc.collect()
    actual = _run(p, options=options, inventory=inventory, caps=caps, callback=replace_and_mutate_controls)
    assert actual.payload_sha256 == expected.payload_sha256
    assert [event.iteration for event in events] == list(range(1, actual.final_snapshot.iteration+1))
    assert events[0].iteration == 1 and not events[0].has_previous_energy


@pytest.mark.parametrize("defect", ["foo_symmetry", "fvv_symmetry", "coefficient", "diagonal_t", "nan", "gap"])
def test_invalid_numerical_input_is_not_silently_repaired(defect):
    p = _problem()
    if defect == "foo_symmetry":
        p["foo"][0, 1] += .01
    elif defect == "fvv_symmetry":
        p["fvv"][0, 1] += .01
    elif defect == "coefficient":
        p["sc"][0][:, 0] = 0
    elif defect == "diagonal_t":
        p["pt"][0][0, 1] = .01
    elif defect == "nan":
        p["st"][0][0] = np.nan
    else:
        p["fvv"][:] = -100*np.eye(p["n"])
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="symmetric|orthonormal|orthogonality|finite|denominator"):
        _run(p, fail_before_call=0)


@pytest.mark.parametrize("field,value", [
    ("maximum_iterations", 0), ("maximum_integral_work_units_per_call", 0),
    ("coefficient_orthogonality_tolerance", 1.0), ("coefficient_orthogonality_tolerance", np.nan),
    ("maximum_diagonal_update_antisymmetry_norm", np.nan), ("maximum_diagonal_update_antisymmetry_norm", -1.0),
    ("denominator_floor", 0.0), ("energy_tolerance", np.inf),
])
def test_invalid_explicit_controls(field, value):
    with pytest.raises(ValueError, match="positive|finite|invalid|declaration"):
        _run(_problem(), options=_options(**{field: value}))


def test_shape_owner_copy_boundaries_and_count_overflow():
    p = _problem()
    result = _run(p)
    copy = result.singles_copy(0)
    copy[:] = 12
    assert not np.array_equal(copy, result.singles_copy(0))
    copy = result.pair_copy(0, 1)
    copy[:] = 12
    assert not np.array_equal(copy, result.pair_copy(0, 1))
    with pytest.raises(IndexError, match="range"):
        result.pair_rank(2, 0)
    p["sc"][0] = np.asfortranarray(p["sc"][0])
    with pytest.raises(ValueError, match="contiguous"):
        _run(p)
    p = _problem()
    p["pc"].pop()
    with pytest.raises(ValueError, match="complete"):
        _run(p)
    p = _problem()
    p["st"][0] = p["st"][0].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        _run(p)
    for o, n, s, r in [(0, 1, 0, 0), (1, 0, 0, 0), (1, 2, 3, 1), (2**64-1, 2, 1, 1)]:
        with pytest.raises((ValueError, RuntimeError, OverflowError)):
            core._plan_bounded_restricted_pair_ccsd_solver_upper(o, n, s, r, _options(), _inventory())
