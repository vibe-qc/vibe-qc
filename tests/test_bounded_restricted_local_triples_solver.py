"""Ragged occupied-coupled Guo triples, independent tiny Galerkin oracles.

W/U are supplied numerical moments here, not physical input certificates.
Dense common amplitudes and linear matrices are restricted to test oracles.
"""

from __future__ import annotations

import gc
import itertools

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_triples_target import _adapt
from tests.test_bounded_restricted_triples_solver import (
    _problem as _common_problem, _moments, _run as _common_run,
    _linear_oracle as _common_linear, _canonical_spin_energy,
)


def _stabilize(tensor, occupied):
    """Assign each repeated-axis orbit once, making exact test symmetry."""
    result = tensor.copy()
    permutations = [p for p in itertools.permutations(range(3))
        if tuple(occupied[k] for k in p) == occupied]
    visited = set()
    for abc in np.ndindex(tensor.shape):
        if abc in visited:
            continue
        orbit = sorted({tuple(abc[k] for k in p) for p in permutations})
        value = sum(float(tensor[index]) for index in orbit)/len(orbit)
        for index in orbit:
            result[index] = value
        visited.update(orbit)
    return np.ascontiguousarray(result)


def _problem(o=2, n=3, *, ranks=None, seed=762):
    rng = np.random.default_rng(seed)
    labels = list(itertools.combinations_with_replacement(range(o), 3))
    ranks = [n]*len(labels) if ranks is None else list(ranks)
    assert len(ranks) == len(labels)
    raw = .025*rng.normal(size=(o, o))
    foo = np.ascontiguousarray(raw+raw.T-np.diag(np.linspace(1.1, 1.6, o)))
    raw = .02*rng.normal(size=(n, n))
    fvv = np.ascontiguousarray(raw+raw.T+np.diag(np.linspace(.35, 1.25, n)))
    coefficients, energies, w, u = [], [], [], []
    for label, rank in zip(labels, ranks, strict=True):
        q = np.linalg.qr(rng.normal(size=(n, n)))[0][:, :rank]
        ev, rotation = np.linalg.eigh(q.T@fvv@q)
        coefficients.append(np.ascontiguousarray(q@rotation))
        energies.append(np.ascontiguousarray(ev))
        w.append(_stabilize(.015*rng.normal(size=(rank,)*3), label))
        u.append(_stabilize(.002*rng.normal(size=(rank,)*3), label))
    return dict(o=o, n=n, labels=labels, ranks=ranks, c=coefficients, eps=energies,
        foo=foo, fvv=fvv, w=w, u=u, snapshot=37)


def _options(**changes):
    options = core._BoundedRestrictedLocalTriplesSolverOptions()
    options.maximum_iterations = 80
    options.denominator_floor = 1e-8
    options.residual_tolerance, options.energy_tolerance = 1e-12, 1e-13
    options.coefficient_orthogonality_tolerance = 1e-12
    options.maximum_projected_fock_error = 1e-11
    options.maximum_repeated_moment_defect_norm = 1e-12
    options.maximum_repeated_update_defect_norm = 1e-11
    for field, value in changes.items():
        setattr(options, field, value)
    return options


def _inventory(**changes):
    inventory = core._BoundedRestrictedLocalTriplesSolverInventory()
    inventory.numerical_replicas = 1
    inventory.fixed_backend_margin_bytes_per_replica = 65536
    for field, value in changes.items():
        setattr(inventory, field, value)
    return inventory


_CAPS = dict(maximum_occupied_count="n_occupied", maximum_common_virtual_dimension="common_virtual_dimension",
    maximum_triple_count="triple_count", maximum_rank="maximum_rank",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
    maximum_per_replica_inventoried_bytes="per_replica_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_inventoried_bytes",
    maximum_moment_visits="moment_visits_upper_bound", maximum_work_units="work_units_upper_bound")


def _caps(plan):
    caps = core._BoundedRestrictedLocalTriplesSolverCaps()
    for field, report in _CAPS.items():
        setattr(caps, field, getattr(plan, report))
    return caps


def _retained(p):
    return sum(a.nbytes for a in p["w"]+p["u"])


def _upper(p, options=None, inventory=None, transient=0):
    return core._plan_bounded_restricted_local_triples_solver_upper(p["o"], p["n"], max(p["ranks"]),
        _options() if options is None else options, _inventory() if inventory is None else inventory,
        _retained(p), transient, 256)


def _arguments(p):
    return p["foo"], p["fvv"], p["c"], p["eps"], p["w"], p["u"], p["snapshot"]


def _plan(p, options=None, inventory=None, caps=None, transient=0):
    options, inventory = _options() if options is None else options, _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_upper(p, options, inventory, transient))
    return core._plan_bounded_restricted_local_triples_solver(*_arguments(p), options, inventory, caps, transient)


def _run(p, options=None, inventory=None, caps=None, callback=None, **faults):
    options, inventory = _options() if options is None else options, _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_plan(p, options, inventory, transient=faults.get("provider_transient_bytes", 0)))
    return core._bounded_restricted_local_triples_solve_diagnostic(*_arguments(p), options, inventory, caps,
        progress=callback, **faults)


def _ordered(local, occupied):
    order = np.argsort(occupied, kind="stable")
    return local.transpose(tuple(np.argsort(order)))


def _lift(p, tensors):
    """Independent common-frame tensor. Stable sorting handles repetitions."""
    index = {label: k for k, label in enumerate(p["labels"])}
    out = np.zeros((p["o"],)*3+(p["n"],)*3)
    common = [np.einsum("ax,by,cz,xyz->abc", c, c, c, t)
        for c, t in zip(p["c"], tensors, strict=True)]
    for label in itertools.product(range(p["o"]), repeat=3):
        out[label] = _ordered(common[index[tuple(sorted(label))]], label)
    return out


def _action(p, tensors):
    t = _lift(p, tensors)
    f, h = p["foo"], p["fvv"]
    operated = (np.einsum("ad,ijkdbc->ijkabc", h, t)
        + np.einsum("bd,ijkadc->ijkabc", h, t) + np.einsum("cd,ijkabd->ijkabc", h, t)
        - np.einsum("il,ljkabc->ijkabc", f, t) - np.einsum("jl,ilkabc->ijkabc", f, t)
        - np.einsum("kl,ijlabc->ijkabc", f, t))
    return [np.einsum("ax,by,cz,abc->xyz", c, c, c, operated[label])
        for label, c in zip(p["labels"], p["c"], strict=True)]


def _energy(p, tensors):
    return sum((2-int(i == j)-int(j == k))*float(np.sum(_adapt(t)*(w+u)))
        for (i, j, k), t, w, u in zip(p["labels"], tensors, p["w"], p["u"], strict=True))


def _norms(p, residuals):
    maximum = max(float(np.max(abs(r), initial=0)) for r in residuals)
    norm = np.sqrt(sum(len(set(itertools.permutations(label)))*float(np.sum(r*r))
        for label, r in zip(p["labels"], residuals, strict=True)))
    return maximum, norm


def _linear_oracle(p):
    sizes = [r**3 for r in p["ranks"]]
    offsets = np.r_[0, np.cumsum(sizes)]
    dimension = int(offsets[-1])
    assert dimension <= 192  # Test-only dense matrix admission.
    def blocks(vector):
        return [vector[offsets[k]:offsets[k+1]].reshape((r,)*3) for k, r in enumerate(p["ranks"])]
    def flatten(values):
        return np.concatenate([x.ravel() for x in values])
    rhs = flatten(p["w"])
    if not dimension:
        return blocks(np.empty(0))
    matrix = np.column_stack([flatten(_action(p, blocks(unit))) for unit in np.eye(dimension)])
    vector = np.linalg.solve(matrix, -rhs)
    np.testing.assert_allclose(matrix@vector+rhs, 0, atol=3e-15)
    return blocks(vector)


@pytest.mark.parametrize("o,n,ranks", [(1, 2, [2]), (2, 3, [2, 1, 3, 0]),
    (3, 2, [2, 1, 0, 2, 1, 2, 0, 1, 2, 1]), (4, 1, [1]*20)])
def test_ragged_coupled_solution_matches_independent_common_frame_linear_operator(o, n, ranks):
    p = _problem(o, n, ranks=ranks)
    result = _run(p)
    assert result.final_snapshot.converged
    expected = _linear_oracle(p)
    actual = [result.amplitudes_copy(*label) for label in p["labels"]]
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, e, atol=5e-13, rtol=3e-10)
    residuals = [w+r for w, r in zip(p["w"], _action(p, actual), strict=True)]
    maximum, norm = _norms(p, residuals)
    assert result.final_snapshot.maximum_absolute_residual == pytest.approx(maximum, abs=3e-15)
    assert result.final_snapshot.residual_frobenius_norm == pytest.approx(norm, abs=8e-15)
    assert result.final_snapshot.triples_energy == pytest.approx(_energy(p, expected), abs=2e-14)
    assert result.final_snapshot.moment_visits == result.final_snapshot.iteration*sum(r > 0 for r in ranks)
    for label, tensor in zip(p["labels"], actual, strict=True):
        for permutation in set(itertools.permutations(label)):
            assert result.rank(*permutation) == tensor.shape[0]
            np.testing.assert_array_equal(result.amplitudes_copy(*permutation), _ordered(tensor, permutation))
        for axes in itertools.permutations(range(3)):
            if tuple(label[k] for k in axes) == label:
                np.testing.assert_array_equal(tensor, tensor.transpose(axes))


def test_full_rank_physical_common_moments_match_dense_coupled_triples_and_spin_energy():
    common = _common_problem(2, 3)
    w, u = _moments(common)
    p = _problem(2, 3)
    p["foo"], p["fvv"] = common["foo"], common["fvv"]
    for index, label in enumerate(p["labels"]):
        # Different complete quasi-canonical frames, with nontrivial axis order.
        order = np.roll(np.arange(3), index % 3)
        c = np.eye(3)[:, order]*np.array([-1., 1., -1.])
        p["c"][index] = np.ascontiguousarray(c)
        p["eps"][index] = np.ascontiguousarray(np.diag(common["fvv"])[order])
        p["w"][index] = np.ascontiguousarray(np.einsum("ax,by,cz,abc->xyz", c, c, c, w[label]))
        p["u"][index] = np.ascontiguousarray(np.einsum("ax,by,cz,abc->xyz", c, c, c, u[label]))
    result, dense = _run(p), _common_run(common)
    assert result.final_snapshot.converged and dense.final_snapshot.converged
    tensors = [result.amplitudes_copy(*label) for label in p["labels"]]
    restored = _lift(p, tensors)
    np.testing.assert_allclose(restored, dense.amplitudes, atol=6e-13, rtol=2e-10)
    np.testing.assert_allclose(restored, _common_linear(common, w), atol=6e-13, rtol=2e-10)
    assert result.final_snapshot.triples_energy == pytest.approx(_canonical_spin_energy(common), abs=3e-13)
    diagonal = dict(p, foo=np.ascontiguousarray(np.diag(np.diag(p["foo"]))))
    assert abs(_run(diagonal).final_snapshot.triples_energy-result.final_snapshot.triples_energy) > 1e-7


def test_all_equal_occupied_amplitudes_couple_despite_zero_energy_weight():
    p = _problem(2, 2)
    for index, label in enumerate(p["labels"]):
        if len(set(label)) != 1:
            p["w"][index][:] = p["u"][index][:] = 0
    result = _run(p)
    assert result.final_snapshot.converged
    assert np.linalg.norm(result.amplitudes_copy(0, 0, 1)) > 1e-7
    assert np.linalg.norm(result.amplitudes_copy(0, 1, 1)) > 1e-7
    expected = _linear_oracle(p)
    for label, tensor in zip(p["labels"], expected, strict=True):
        np.testing.assert_allclose(result.amplitudes_copy(*label), tensor, atol=5e-13)
    # The iii blocks drive neighbours but never acquire an energy multiplier.
    assert result.final_snapshot.triples_energy == 0


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_exhaustion_returns_only_evaluated_snapshot_and_raw_residual(iterations):
    p = _problem(2, 3, ranks=[2, 1, 3, 0])
    options = _options(maximum_iterations=iterations)
    events = []
    result = _run(p, options, callback=events.append)
    assert not result.final_snapshot.converged
    tensors = [np.zeros_like(w) for w in p["w"]]
    gaps = [e[:, None, None]+e[None, :, None]+e[None, None, :]-sum(p["foo"][i, i] for i in label)
        for label, e in zip(p["labels"], p["eps"], strict=True)]
    assert [event.iteration for event in events] == list(range(1, iterations+1))
    for index, event in enumerate(events):
        residual = [w+r for w, r in zip(p["w"], _action(p, tensors), strict=True)]
        maximum, norm = _norms(p, residual)
        assert event.triples_energy == pytest.approx(_energy(p, tensors), abs=3e-15)
        assert event.maximum_absolute_residual == pytest.approx(maximum, abs=3e-15)
        assert event.residual_frobenius_norm == pytest.approx(norm, abs=8e-15)
        if index+1 < iterations:
            tensors = [_stabilize(t-r/g, label) for label, t, r, g in
                zip(p["labels"], tensors, residual, gaps, strict=True)]
    for label, tensor in zip(p["labels"], tensors, strict=True):
        np.testing.assert_allclose(result.amplitudes_copy(*label), tensor, atol=3e-16, rtol=2e-12)


def test_zero_rank_all_spaces_need_no_moments_or_denominators():
    p = _problem(2, 3, ranks=[0]*4)
    plan = _plan(p)
    assert plan.total_amplitude_elements == plan.maximum_rank == plan.moment_visits_upper_bound == 0
    result = _run(p, caps=_caps(plan), fail_before_visit=0)
    assert result.final_snapshot.converged and result.final_snapshot.iteration == 2
    assert result.final_snapshot.moment_visits == result.final_snapshot.neighbour_visits == 0
    assert result.final_snapshot.triples_energy == 0
    assert result.minimum_denominator == result.maximum_denominator == 0
    assert all(result.amplitudes_copy(*label).shape == (0, 0, 0) for label in p["labels"])
    with pytest.raises(IndexError):
        result.amplitude(0, 0, 0, 0, 0, 0)


def test_only_exact_zero_occupied_couplings_skip_neighbours():
    p = _problem(2, 1)
    p["foo"][0, 1] = p["foo"][1, 0] = 0
    diagonal = _run(p)
    assert diagonal.final_snapshot.neighbour_visits == 0
    p["foo"][0, 1] = p["foo"][1, 0] = 1e-300
    tiny = _run(p)
    assert tiny.final_snapshot.neighbour_visits > 0


def test_exact_ragged_inventory_and_uniform_upper_replicas():
    p = _problem(2, 3, ranks=[2, 1, 3, 0])
    inventory = _inventory(numerical_replicas=3, external_node_bytes=123,
        other_live_bytes_per_replica=456)
    plan = _plan(p, inventory=inventory)
    upper = _upper(p, inventory=inventory)
    count = sum(r**3 for r in p["ranks"])
    borrowed = p["foo"].nbytes+p["fvv"].nbytes+sum(a.nbytes for a in p["c"]+p["eps"])
    assert plan.total_amplitude_elements == count
    assert plan.amplitude_snapshot_bytes == plan.candidate_snapshot_bytes == 8*count
    assert plan.borrowed_numerical_bytes == borrowed
    assert plan.moment_digest_bytes == 64*len(p["labels"])
    assert plan.output_numerical_bytes == plan.amplitude_snapshot_bytes+plan.retained_record_bytes
    assert plan.peak_owned_numerical_bytes <= upper.peak_owned_numerical_bytes
    assert plan.work_units_upper_bound <= upper.work_units_upper_bound
    assert plan.required_node_inventoried_bytes == 123+3*plan.per_replica_inventoried_bytes
    extra = _plan(p, inventory=inventory, transient=31)
    assert extra.per_replica_inventoried_bytes-plan.per_replica_inventoried_bytes == 31
    assert extra.required_node_inventoried_bytes-plan.required_node_inventoried_bytes == 93
    assert not plan.uniform_rank_upper_bound and upper.uniform_rank_upper_bound


@pytest.mark.parametrize("field", list(_CAPS))
def test_exact_caps_and_one_below_precede_numeric_scans_and_provider(field):
    p = _problem(2, 2)
    plan = _plan(p)
    caps = _caps(plan)
    assert _run(p, caps=caps).final_snapshot.converged
    setattr(caps, field, getattr(caps, field)-1)
    p["c"][0][0, 0] = np.nan
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _run(p, caps=caps, callback=events.append, fail_before_visit=0)
    assert not events


@pytest.mark.parametrize("fault", range(1, 8))
def test_provider_protocol_failure_does_not_publish(fault):
    with pytest.raises((ValueError, RuntimeError), match="moment|receiver|request|exactly|snapshot|extent|rank"):
        _run(_problem(), protocol_fault=fault)


@pytest.mark.parametrize("visit", [0, 2, 5])
def test_provider_exception_and_moments_changed_between_iterations_abort(visit):
    p = _problem()
    with pytest.raises(RuntimeError, match="injected moment"):
        _run(p, fail_before_visit=visit)
    assert _run(p).final_snapshot.converged
    def mutate(event):
        p["w"][1][0, 0, 0] += .001
    with pytest.raises((ValueError, RuntimeError), match="moment|immutable|changed|snapshot"):
        _run(p, callback=mutate)


@pytest.mark.parametrize("field", ["foo", "fvv", "c", "eps", "w", "u"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_inputs_and_moments_are_rejected(field, value):
    p = _problem()
    selected = p[field][0] if isinstance(p[field], list) else p[field]
    selected.flat[0] = value
    with pytest.raises((ValueError, OverflowError), match="finite|input|moment"):
        _run(p)


def test_explicit_quasi_canonical_and_repeated_moment_budgets_are_not_repairs():
    p = _problem(2, 2)
    p["eps"][1][0] += .001
    with pytest.raises(ValueError, match="Fock|canonical|project"):
        _run(p)
    p = _problem(2, 2)
    p["w"][1][0, 1, 0] += .001  # iij requires exchange of the first two axes.
    with pytest.raises(ValueError, match="moment|repeat|defect"):
        _run(p)
    one = _options(maximum_iterations=1, maximum_repeated_moment_defect_norm=1,
        maximum_repeated_update_defect_norm=0)
    result = _run(p, one)
    assert not result.final_snapshot.converged and result.final_snapshot.repeated_update_defect_norm == 0
    assert result.final_snapshot.maximum_repeated_moment_defect_norm > 0
    one.maximum_iterations = 2
    with pytest.raises(ValueError, match="update|repeat|defect|projection"):
        _run(p, one)
    raw = _run(p, _options(maximum_iterations=4, maximum_repeated_moment_defect_norm=1,
        maximum_repeated_update_defect_norm=1))
    assert not raw.final_snapshot.converged
    assert raw.final_snapshot.maximum_absolute_residual > 1e-5  # No projected-residual false convergence.


@pytest.mark.parametrize("field,value", [("maximum_iterations", 0), ("denominator_floor", 0),
    ("residual_tolerance", 0), ("energy_tolerance", np.nan), ("coefficient_orthogonality_tolerance", 1),
    ("maximum_projected_fock_error", np.nan), ("maximum_repeated_moment_defect_norm", -1),
    ("maximum_repeated_update_defect_norm", np.nan)])
def test_required_controls_no_defaults(field, value):
    with pytest.raises((ValueError, OverflowError)):
        _plan(_problem(), options=_options(**{field: value}))


def test_denominator_floor_is_strict_and_no_shift_is_introduced():
    p = _problem(1, 1)
    p["foo"][:] = 0
    p["fvv"][:] = .25
    p["eps"][0][:] = .25
    with pytest.raises(ValueError, match="denominator|floor"):
        _run(p, _options(denominator_floor=.75))
    result = _run(p, _options(denominator_floor=np.nextafter(.75, 0)))
    assert result.final_snapshot.converged and result.minimum_denominator == .75


def test_callbacks_controls_array_storage_and_owned_output_lifetime():
    p = _problem()
    options, inventory = _options(), _inventory()
    caps = _caps(_plan(p, options, inventory))
    baseline = _run(p, options, inventory, caps)
    def controls(event):
        options.maximum_iterations = 0
        inventory.numerical_replicas = 0
        caps.maximum_rank = 0
    copied = _run(p, options, inventory, caps, controls)
    assert copied.payload_sha256 == baseline.payload_sha256
    for field in ("foo", "c", "eps"):
        p = _problem()
        def mutate(event):
            selected = p[field][0] if isinstance(p[field], list) else p[field]
            selected.flat[0] += .001
        with pytest.raises((ValueError, RuntimeError), match="input|changed|immutable|Fock|orthogonal|snapshot"):
            _run(p, callback=mutate)
    p = _problem()
    def resize(event):
        p["w"][0].resize((1, 1, 1), refcheck=False)
    with pytest.raises(ValueError, match="storage|shape"):
        _run(p, callback=resize)
    p = _problem()
    def replace(event):
        p["c"][:] = [None]*len(p["c"])
        p["w"][:] = [None]*len(p["w"])
    # Native borrows the pinned original arrays, not future list entries.
    assert _run(p, callback=replace).final_snapshot.converged
    p = _problem()
    def cancel(event):
        raise RuntimeError("intentional local triples cancellation")
    with pytest.raises(RuntimeError, match="intentional local"):
        _run(p, callback=cancel)
    result = _run(p)
    original = result.amplitudes_copy(0, 0, 1)
    result.amplitudes_copy(0, 0, 1)[:] = 17
    del p
    gc.collect()
    np.testing.assert_array_equal(result.amplitudes_copy(0, 0, 1), original)
    with pytest.raises(IndexError):
        result.rank(4, 0, 0)


def test_shapes_alignment_and_no_implicit_array_coercion():
    p = _problem()
    p["c"][0] = np.asfortranarray(p["c"][0])
    with pytest.raises(ValueError, match="contiguous"):
        _plan(p)
    p = _problem()
    p["w"][0] = p["w"][0].astype(np.float32)
    with pytest.raises(ValueError, match="binary64"):
        _plan(p)
    p = _problem()
    p["c"][0] = p["c"][0].tolist()
    with pytest.raises(ValueError, match="arrays"):
        _plan(p)
    p = _problem()
    p["eps"].pop()
    with pytest.raises(ValueError, match="multiset"):
        _plan(p)
    p = _problem()
    raw = np.zeros(p["c"][0].nbytes+1, np.uint8)
    p["c"][0] = np.ndarray(p["c"][0].shape, dtype=np.float64, buffer=raw, offset=1)
    with pytest.raises(ValueError, match="aligned"):
        _plan(p)


@pytest.mark.parametrize("o,n,rank", [(0, 2, 1), (2, 0, 0), (2, 2, 3), (2**63, 2, 1), (2, 2**63, 1)])
def test_count_only_upper_rejects_invalid_or_overflowing_extents(o, n, rank):
    with pytest.raises((ValueError, OverflowError)):
        core._plan_bounded_restricted_local_triples_solver_upper(o, n, rank,
            _options(), _inventory(), 1024, 0, 256)
