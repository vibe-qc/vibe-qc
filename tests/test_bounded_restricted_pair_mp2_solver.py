"""Ragged pair-MP2 numerical references, not physical PNO/source constructors."""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement, product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _problem(o=3, n=3, ranks=None, *, same_basis=False, seed=924):
    rng = np.random.default_rng(seed)
    labels = list(combinations_with_replacement(range(o), 2))
    ranks = [n]*len(labels) if ranks is None else ranks
    off = .018*rng.normal(size=(o, o))
    f = off+off.T+np.diag(-np.linspace(1.0, 1.6, o))
    common_f = np.diag(np.linspace(.6, 1.3, n))
    gs, es, cs, common_gs = [], [], [], []
    for (i, j), r in zip(labels, ranks, strict=True):
        c = np.eye(n)[:, :r] if same_basis else np.linalg.qr(rng.normal(size=(n, n)))[0][:, :r]
        eps, u = np.linalg.eigh(c.T@common_f@c)
        c = c@u
        raw = .03*rng.normal(size=(n, n))
        if i == j:
            raw = (raw+raw.T)/2
        g = c.T@raw@c
        if i == j:
            g = (g+g.T)/2  # Exact diagonal pair symmetry, numerical fixture only.
        gs.append(np.ascontiguousarray(g))
        es.append(np.ascontiguousarray(eps))
        cs.append(np.ascontiguousarray(c))
        common_gs.append(raw)
    return dict(o=o, n=n, f=np.ascontiguousarray(f), labels=labels,
        ranks=list(ranks), g=gs, eps=es, c=cs, common_g=common_gs)


def _options(**changes):
    o = core._BoundedRestrictedPairMP2SolverOptions()
    o.maximum_iterations = 80
    o.denominator_floor = 1e-8
    o.residual_tolerance, o.energy_tolerance = 1e-12, 1e-13
    o.coefficient_orthogonality_tolerance = 1e-12
    o.maximum_diagonal_update_antisymmetry_norm = 1e-12
    for key, value in changes.items():
        setattr(o, key, value)
    return o


def _inventory(**changes):
    i = core._BoundedRestrictedPairMP2SolverInventory()
    i.numerical_replicas = 1
    i.fixed_backend_margin_bytes_per_replica = 65536
    for key, value in changes.items():
        setattr(i, key, value)
    return i


def _caps(p):
    caps = core._BoundedRestrictedPairMP2SolverCaps()
    fields = dict(maximum_occupied_count="n_occupied", maximum_common_virtual_dimension="common_virtual_dimension",
        maximum_pair_count="pair_count", maximum_pair_rank="maximum_pair_rank",
        maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
        maximum_per_replica_inventoried_bytes="per_replica_inventoried_bytes",
        maximum_node_inventoried_bytes="required_node_inventoried_bytes",
        maximum_coupling_slots="coupling_slots_upper_bound", maximum_work_units="work_units_upper_bound")
    for cap, report in fields.items():
        setattr(caps, cap, getattr(p, report))
    return caps


def _upper(p, options=None, inventory=None):
    return core._plan_bounded_restricted_pair_mp2_solver_upper(p["o"], p["n"], max(p["ranks"]),
        (_options() if options is None else options).maximum_iterations, _inventory() if inventory is None else inventory)


def _plan(p, options=None, inventory=None, caps=None):
    options, inventory = _options() if options is None else options, _inventory() if inventory is None else inventory
    return core._plan_bounded_restricted_pair_mp2_solver(p["f"], p["g"], p["eps"], p["c"],
        options, inventory, _caps(_upper(p, options, inventory)) if caps is None else caps)


def _run(p, *, options=None, inventory=None, caps=None, callback=None):
    options, inventory = _options() if options is None else options, _inventory() if inventory is None else inventory
    caps = _caps(_plan(p, options, inventory)) if caps is None else caps
    return core._bounded_restricted_pair_mp2_solve_diagnostic(p["f"], p["g"], p["eps"], p["c"],
        options, inventory, caps, callback)


def _blocks(result, p):
    return [result.pair_copy(i, j) for i, j in p["labels"]]


def _operator(p, blocks):
    """Projected full occupied equation, including both diagonal F terms.

    This explicitly includes k=i/j via C.T@C, independently of the native
    diagonal denominator extraction. No native overlap/residual is used.
    """
    lookup = {key: index for index, key in enumerate(p["labels"])}
    def source(i, j):
        index = lookup[min(i, j), max(i, j)]
        return index, blocks[index] if i <= j else blocks[index].T
    output = []
    for target, (i, j) in enumerate(p["labels"]):
        c, eps = p["c"][target], p["eps"][target]
        value = (eps[:, None]+eps[None, :])*blocks[target]
        for k in range(p["o"]):
            s, t = source(k, j)
            overlap = c.T@p["c"][s]
            value -= p["f"][k, i]*(overlap@t@overlap.T)
            s, t = source(i, k)
            overlap = c.T@p["c"][s]
            value -= p["f"][k, j]*(overlap@t@overlap.T)
        output.append(value)
    return output


def _projected_linear_oracle(p):
    dimensions = [r*r for r in p["ranks"]]
    offsets = np.r_[0, np.cumsum(dimensions)]
    size = int(offsets[-1])
    def unpack(x):
        return [x[offsets[k]:offsets[k+1]].reshape(r, r) for k, r in enumerate(p["ranks"])]
    if not size:
        return unpack(np.empty(0))
    columns = []
    for x in np.eye(size):
        columns.append(np.concatenate([a.ravel() for a in _operator(p, unpack(x))]))
    matrix = np.column_stack(columns)
    right = -np.concatenate([g.ravel() for g in p["g"]])
    return unpack(np.linalg.solve(matrix, right))


def _energy(p, blocks):
    return sum((1 if i == j else 2)*np.sum((2*g-g.T)*t)
        for (i, j), g, t in zip(p["labels"], p["g"], blocks, strict=True))


def _kronecker_oracle(p):
    o, n, f = p["o"], p["n"], p["f"]
    g = np.zeros((o, o, n, n))
    for (i, j), block in zip(p["labels"], p["g"], strict=True):
        g[i, j] = block
        g[j, i] = block.T
    identity = np.eye(o)
    foo = np.kron(f, identity)+np.kron(identity, f)
    answer = np.empty_like(g)
    eps = p["eps"][0]
    for a, b in product(range(n), repeat=2):
        answer[:, :, a, b] = np.linalg.solve((eps[a]+eps[b])*np.eye(o*o)-foo,
            -g[:, :, a, b].ravel()).reshape(o, o)
    return [answer[i, j] for i, j in p["labels"]]


@pytest.mark.parametrize("o,n", [(1, 1), (2, 3), (3, 2), (4, 4)])
def test_full_space_matches_independent_occupied_kronecker_equation(o, n):
    p = _problem(o, n, same_basis=True)
    events = []
    result = _run(p, callback=events.append)
    assert result.final_snapshot.converged
    wanted, actual = _kronecker_oracle(p), _blocks(result, p)
    for a, b in zip(actual, wanted, strict=True):
        np.testing.assert_allclose(a, b, atol=5e-13, rtol=3e-11)
    assert result.final_snapshot.correlation_energy == pytest.approx(_energy(p, wanted), abs=3e-13)
    assert len(events) == result.final_snapshot.iteration >= 2
    assert result.final_snapshot.coupling_slots == len(events)*len(p["labels"])*2*(o-1)
    assert result.final_snapshot.target_evaluations == len(events)*len(p["labels"])
    assert result.final_snapshot.input_checks == len(events)+1
    assert result.final_snapshot.charged_work_units <= result.memory.work_units_upper_bound


@pytest.mark.parametrize("ranks", [[3, 2, 1, 2, 0, 3], [1, 0, 2, 0, 1, 0], [0]*6])
def test_ragged_projected_linear_equation_reverse_sources_and_omitted_spaces(ranks):
    p = _problem(ranks=ranks)
    result = _run(p)
    expected, actual = _projected_linear_oracle(p), _blocks(result, p)
    assert result.final_snapshot.converged
    for (i, j), a, b in zip(p["labels"], actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=5e-13, rtol=3e-11)
        np.testing.assert_array_equal(result.pair_copy(j, i), a.T)
        assert result.pair_rank(j, i) == a.shape[0]
        if i == j:
            np.testing.assert_array_equal(a, a.T)
    residuals = [g+r for g, r in zip(p["g"], _operator(p, actual), strict=True)]
    maximum = max((np.max(abs(r)) for r in residuals if r.size), default=0)
    assert maximum <= 2e-12
    assert result.final_snapshot.maximum_absolute_residual == pytest.approx(maximum, abs=2e-15)
    assert result.final_snapshot.correlation_energy == pytest.approx(_energy(p, expected), abs=3e-13)
    assert result.final_snapshot.zero_rank_targets > 0
    if any(ranks):
        assert result.final_snapshot.zero_rank_sources > 0 and result.final_snapshot.transposed_sources > 0
    else:
        assert result.final_snapshot.correlation_energy == 0
        assert result.minimum_denominator == result.maximum_denominator == 0


def test_rank_resolved_inventory_and_uniform_upper_need_no_pair_payload():
    p = _problem(ranks=[3, 2, 1, 2, 0, 3])
    inventory = _inventory(numerical_replicas=2, external_node_bytes=123, other_live_bytes_per_replica=456)
    exact, upper = _plan(p, inventory=inventory), _upper(p, inventory=inventory)
    a, r, count = sum(x*x for x in p["ranks"]), max(p["ranks"]), len(p["ranks"])
    assert not exact.uniform_rank_upper_bound and upper.uniform_rank_upper_bound
    assert exact.amplitude_snapshot_bytes == exact.candidate_snapshot_bytes == 8*a
    assert exact.retained_pair_record_bytes == 16*count
    assert exact.borrowed_pair_table_bytes == 56*count
    assert exact.peak_owned_numerical_bytes == 16*a+16*count+32*r*r
    assert exact.initialization_phase_owned_bytes == 16*a+16*count+8*r*r
    assert exact.output_numerical_bytes == 8*a+16*count
    assert exact.borrowed_pair_numeric_bytes == sum(8*(rank*rank+rank+p["n"]*rank) for rank in p["ranks"])
    per_replica = (exact.peak_owned_numerical_bytes+exact.borrowed_fock_bytes+exact.borrowed_pair_numeric_bytes
        +exact.borrowed_pair_table_bytes+exact.fixed_control_storage_bytes+456+65536)
    assert exact.per_replica_inventoried_bytes == per_replica
    assert exact.required_node_inventoried_bytes == 123+2*per_replica
    for field in ("peak_owned_numerical_bytes", "required_node_inventoried_bytes", "work_units_upper_bound",
                  "overlap_scalar_products_upper_bound", "residual_scalar_products_upper_bound"):
        assert getattr(exact, field) <= getattr(upper, field)


@pytest.mark.parametrize("field", ["maximum_occupied_count", "maximum_common_virtual_dimension", "maximum_pair_count",
    "maximum_pair_rank", "maximum_owned_numerical_bytes", "maximum_per_replica_inventoried_bytes",
    "maximum_node_inventoried_bytes", "maximum_coupling_slots", "maximum_work_units"])
def test_one_below_exact_cap_rejects_before_nonfinite_payload_or_callback(field):
    p = _problem()
    caps = _caps(_plan(p))
    assert _run(p, caps=caps).final_snapshot.converged
    setattr(caps, field, getattr(caps, field)-1)
    p["g"][0].flat[0] = np.nan
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap|exceed"):
        _run(p, caps=caps, callback=events.append)
    assert not events


def test_iteration_exhaustion_returns_semicanonical_last_evaluated_snapshot():
    p = _problem()
    result = _run(p, options=_options(maximum_iterations=1))
    assert not result.final_snapshot.converged and not result.final_snapshot.has_previous_energy
    expected = []
    for (i, j), g, eps in zip(p["labels"], p["g"], p["eps"], strict=True):
        expected.append(-g/(eps[:, None]+eps[None, :]-p["f"][i, i]-p["f"][j, j]))
    for actual, wanted in zip(_blocks(result, p), expected, strict=True):
        np.testing.assert_allclose(actual, wanted, atol=1e-17, rtol=2e-15)
    assert result.final_snapshot.correlation_energy == pytest.approx(_energy(p, expected), abs=2e-15)
    assert result.final_snapshot.maximum_absolute_residual > 1e-6


def test_zero_fock_couplings_still_consume_all_ordered_slots():
    p = _problem(ranks=[3, 2, 1, 2, 0, 3])
    p["f"] = np.diag(np.diag(p["f"]))
    result = _run(p)
    assert result.final_snapshot.converged and result.final_snapshot.iteration == 2
    assert result.final_snapshot.coupling_slots == 2*6*4
    assert result.final_snapshot.overlap_scalar_products > 0


@pytest.mark.parametrize("mutation", ["fock", "diagonal_g", "nonfinite", "nonorthogonal", "gap"])
def test_invalid_algebra_is_not_silently_repaired(mutation):
    p = _problem()
    if mutation == "fock":
        p["f"][0, 1] += 1e-6
    elif mutation == "diagonal_g":
        p["g"][0][0, 1] += 1e-6
    elif mutation == "nonfinite":
        p["eps"][0][0] = np.nan
    elif mutation == "nonorthogonal":
        p["c"][0][0, 0] *= 2
    else:
        p["eps"][0][:] = -20
    with pytest.raises((ValueError, OverflowError), match="symmetric|finite|orthonormal|denominator"):
        _run(p)


@pytest.mark.parametrize("budget", [np.nan, -1.0, np.inf])
def test_diagonal_update_budget_requires_an_explicit_finite_choice(budget):
    with pytest.raises(ValueError, match="antisymmetry budget"):
        _run(_problem(), options=_options(maximum_diagonal_update_antisymmetry_norm=budget))


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.nan, np.inf, 1.0, 2.0])
def test_coefficient_orthogonality_tolerance_cannot_admit_zero_columns(tolerance):
    p = _problem(o=1, n=1)
    p["c"][0][:] = 0
    with pytest.raises(ValueError, match="positive finite|orthogonality tolerance"):
        _run(p, options=_options(coefficient_orthogonality_tolerance=tolerance))


def test_reported_diagonal_update_projection_has_an_enforced_boundary():
    p = _problem()
    result = _run(p)
    discarded = result.final_snapshot.maximum_diagonal_update_antisymmetry_norm
    assert 0 <= discarded < 1e-12
    exact = _run(p, options=_options(maximum_diagonal_update_antisymmetry_norm=discarded))
    assert exact.payload_sha256 == result.payload_sha256
    # An exact-zero discarded norm has no smaller valid nonnegative budget.
    if discarded > 0:
        with pytest.raises(ValueError, match="antisymmetry budget exceeded"):
            _run(p, options=_options(maximum_diagonal_update_antisymmetry_norm=np.nextafter(discarded, 0)))


def test_callback_cancel_final_mutation_and_pinned_input_list_owners():
    p = _problem()
    def cancel(event):
        raise RuntimeError("intentional pair MP2 cancellation")
    with pytest.raises(RuntimeError, match="intentional pair MP2 cancellation"):
        _run(p, callback=cancel)
    def final_mutate(event):
        if event.converged:
            p["g"][1].flat[0] += 1
    with pytest.raises(ValueError, match="input changed"):
        _run(p, callback=final_mutate)
    p = _problem()
    expected = _run(p)
    def replace_list_entries(event):
        p["g"][:] = [np.full_like(g, np.nan) for g in p["g"]]
        gc.collect()
    copied = _run(p, callback=replace_list_entries)
    assert copied.payload_sha256 == expected.payload_sha256


def test_overflow_counts_and_rank_zero_amplitude_index_are_rejected():
    with pytest.raises((ValueError, OverflowError), match="overflow|extent"):
        core._plan_bounded_restricted_pair_mp2_solver_upper(2**64-1, 2, 2, 2, _inventory())
    result = _run(_problem(ranks=[0]*6))
    with pytest.raises(IndexError, match="range"):
        result.amplitude(0, 0, 0, 0)
    with pytest.raises(IndexError, match="range"):
        result.pair_rank(3, 0)
