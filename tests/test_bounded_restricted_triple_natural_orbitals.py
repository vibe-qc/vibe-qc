"""Riplinger Eqs10-15 TNO geometry from explicitly supplied connected T2.

These are pure numerical oracles, not physical CCSD/source certification.
The direct tilde-T density below is independent of the native PSD algebra.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _options(**changes):
    options = core._BoundedRestrictedTNOOptions()
    options.occupation_cutoff = 0
    options.union_absolute_rank_cutoff = options.union_relative_rank_cutoff = 1e-12
    options.maximum_union_column_reconstruction_error = 1e-9
    options.input_orthonormality_tolerance = 1e-10
    options.eigensystem_relative_reconstruction_tolerance = 1e-10
    options.eigenvector_orthogonality_tolerance = 1e-10
    options.union_negative_absolute_tolerance = options.union_negative_relative_tolerance = 1e-12
    options.density_negative_absolute_tolerance = options.density_negative_relative_tolerance = 1e-12
    options.occupation_ambiguity_absolute_guard = 1e-13
    options.occupation_ambiguity_relative_guard = 1e-12
    for field in ("union", "density", "fock"):
        setattr(options, field+"_max_sweeps", 80)
        setattr(options, field+"_relative_eigensolver_tolerance", 1e-14)
    for field, value in changes.items():
        setattr(options, field, value)
    return options


def _inventory(**changes):
    inventory = core._BoundedRestrictedTNOInventory()
    inventory.backend_margin_bytes_per_replica = 65536
    for field, value in changes.items():
        setattr(inventory, field, value)
    return inventory


def _problem(occupied=(0, 1, 2), n=4, ranks=(2, 2, 2), seed=930):
    rng = np.random.default_rng(seed)
    i, j, k = occupied
    labels = [(i, j), (i, k), (j, k)]
    pairs = {}
    for label, rank in zip(labels, ranks, strict=True):
        if label in pairs:
            assert pairs[label][0].shape[1] == rank
            continue
        c = np.ascontiguousarray(np.linalg.qr(rng.normal(size=(n, n)))[0][:, :rank])
        t = .1*rng.normal(size=(rank, rank))
        if label[0] == label[1]:
            t = (t+t.T)/2
        pairs[label] = c, np.ascontiguousarray(t)
    raw = .1*rng.normal(size=(n, n))
    return dict(o=max(occupied)+1, occupied=occupied, n=n, labels=labels,
        c=[pairs[label][0] for label in labels], t=[pairs[label][1] for label in labels],
        fvv=np.ascontiguousarray(raw+raw.T+np.diag(np.linspace(.2, 1.4, n))))


def _arguments(p):
    return p["o"], p["occupied"], p["fvv"], p["c"], p["t"]


def _plan(p, options=None, inventory=None):
    return core._plan_bounded_restricted_triple_natural_orbitals(*_arguments(p),
        _options() if options is None else options, _inventory() if inventory is None else inventory)


_CAPS = dict(maximum_common_dimension="common_dimension", maximum_union_columns="union_columns",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes", maximum_node_bytes="total_node_bytes",
    maximum_control_storage_bytes_per_replica="control_storage_bytes_per_replica", maximum_work_units="work_units_upper_bound")


def _caps(plan):
    caps = core._BoundedRestrictedTNOCaps()
    for field, report in _CAPS.items():
        setattr(caps, field, getattr(plan, report))
    return caps


def _run(p, options=None, inventory=None, caps=None):
    options, inventory = _options() if options is None else options, _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_plan(p, options, inventory))
    return core._bounded_restricted_triple_natural_orbitals_diagnostic(*_arguments(p), options, inventory, caps)


def _oracle(p, cutoff=0):
    # All THREE occurrences enter the density, repeated edges included.
    density = np.zeros((p["n"], p["n"]))
    for (i, j), c, t in zip(p["labels"], p["c"], p["t"], strict=True):
        tilde = (4*t-2*t.T)/(1+int(i == j))
        pair = tilde@t.T+tilde.T@t
        np.testing.assert_allclose(pair, pair.T, atol=2e-15)
        density += c@pair@c.T/3
    unique = list(dict.fromkeys(p["labels"]))
    columns = np.column_stack([p["c"][p["labels"].index(label)] for label in unique])
    if not columns.shape[1]:
        return density, np.empty(0), np.empty((p["n"], 0)), np.empty(0), np.empty((p["n"], 0))
    q, singular, _ = np.linalg.svd(columns, full_matrices=False)
    rank = np.count_nonzero(singular > 1e-8)
    q = q[:, :rank]
    occupations, vectors = np.linalg.eigh(q.T@density@q)
    occupations, vectors = occupations[::-1], vectors[:, ::-1]
    selected = np.ones(rank, bool) if cutoff == 0 else occupations > cutoff
    c = q@vectors[:, selected]
    eps, rotation = np.linalg.eigh(c.T@p["fvv"]@c)
    return density, occupations, c@rotation, eps, q


@pytest.mark.parametrize("occupied,ranks", [((0, 1, 2), (2, 2, 2)),
    ((0, 0, 1), (1, 2, 2)), ((0, 1, 1), (2, 2, 1)), ((1, 1, 1), (2, 2, 2)),
    ((0, 1, 2), (0, 2, 1))])
def test_direct_contravariant_density_matches_tno_occupations_and_fock_projector(occupied, ranks):
    p = _problem(occupied, ranks=ranks)
    result = _run(p)
    density, occupations, c, eps, union = _oracle(p)
    assert tuple(result.occupied) == occupied
    assert result.union_rank == len(occupations) == result.retained_rank
    np.testing.assert_allclose(result.occupations_copy(), occupations, atol=4e-12, rtol=1e-10)
    actual = result.coefficients_copy()
    np.testing.assert_allclose(actual@actual.T, c@c.T, atol=5e-11, rtol=2e-10)
    np.testing.assert_allclose(actual.T@actual, np.eye(result.retained_rank), atol=2e-10)
    np.testing.assert_allclose(result.energies_copy(), eps, atol=5e-11, rtol=2e-10)
    np.testing.assert_allclose(actual.T@p["fvv"]@actual, np.diag(result.energies_copy()), atol=5e-11)
    assert result.union_column_reconstruction_frobenius_error < 1e-9
    assert result.output_numerical_bytes == 8*(p["n"]*result.retained_rank+result.retained_rank+result.union_rank)
    assert result.union_audit.relative_reconstruction_error <= 1e-10
    assert result.density_audit.relative_reconstruction_error <= 1e-10
    # Before F rotation, occupation eigenvalues belong to the density, not
    # individually to columns of the final ascending-energy output.
    assert np.trace(union.T@density@union) == pytest.approx(sum(occupations), abs=4e-12)


def test_antisymmetric_twelve_prefactor_and_diagonal_two_not_periodic_four():
    p = _problem(n=2, ranks=(2, 2, 2))
    c = np.eye(2)
    t = np.array([[0., .25], [-.25, 0.]])
    p["c"], p["t"] = [c, c, c], [t, t, t]
    np.testing.assert_allclose(_run(p).occupations_copy(), [.75, .75], atol=2e-14)
    p = _problem((0, 0, 0), n=2, ranks=(2, 2, 2))
    t = np.diag([.5, .25])
    p["c"], p["t"] = [c, c, c], [t, t, t]
    np.testing.assert_allclose(_run(p).occupations_copy(), [.5, .125], atol=2e-14)


def test_repeated_edge_storage_is_unique_but_density_occurrences_are_not_deduplicated():
    p = _problem((0, 0, 1), n=2, ranks=(1, 1, 1))
    c = np.array([[1.], [0.]])
    p["c"] = [c, c, c]
    diagonal, offdiagonal = np.array([[.25]]), np.array([[.5]])
    p["t"] = [diagonal, offdiagonal, offdiagonal]
    result = _run(p)
    assert result.memory.unique_edge_count == 2 and result.memory.union_columns == 2
    assert result.memory.borrowed_numerical_bytes == 8*(4+2*(2+1))
    expected = (2*.25**2+2*4*.5**2)/3
    assert result.occupations_copy()[0] == pytest.approx(expected, abs=3e-14)
    copied = dict(p, t=[diagonal, offdiagonal, offdiagonal.copy()])
    with pytest.raises(ValueError, match="repeated|identical|alias"):
        _plan(copied)
    copied = dict(p, c=[c, c, c.copy()])
    with pytest.raises(ValueError, match="repeated|identical|alias"):
        _plan(copied)


def test_zero_cutoff_completes_only_union_and_retains_null_density_modes():
    p = _problem(n=5, ranks=(2, 2, 2))
    c, t = np.eye(5)[:, :2].copy(), np.zeros((2, 2))
    p["c"], p["t"] = [c, c, c], [t, t, t]
    complete = _run(p)
    assert complete.union_rank == complete.retained_rank == 2 < p["n"]
    assert complete.usable
    np.testing.assert_array_equal(complete.occupations_copy(), 0)
    np.testing.assert_allclose(complete.coefficients_copy()@complete.coefficients_copy().T, c@c.T, atol=2e-12)
    empty = _run(p, _options(occupation_cutoff=1e-4))
    assert empty.union_rank == 2 and empty.retained_rank == 0 and not empty.usable
    assert empty.coefficients_copy().shape == (5, 0)
    assert empty.energies_copy().size == 0 and empty.occupations_copy().size == 2


def test_empty_union_admits_exact_zero_owned_and_column_caps_but_validates_fock():
    p = _problem(n=3, ranks=(0, 0, 0))
    plan = _plan(p)
    assert plan.union_columns == plan.peak_owned_numerical_bytes == plan.output_numerical_bytes_upper_bound == 0
    result = _run(p, caps=_caps(plan))
    assert result.union_rank == result.retained_rank == result.output_numerical_bytes == 0
    assert not result.usable and result.coefficients_copy().shape == (3, 0)
    p["fvv"][0, 0] = np.nan
    with pytest.raises(OverflowError, match="finite"):
        _run(p, caps=_caps(plan))


def test_occupation_selection_and_ambiguity_are_explicit_no_minimum_rank():
    p = _problem((0, 0, 0), n=2, ranks=(2, 2, 2))
    c, t = np.eye(2), np.diag([.5, .25])
    p["c"], p["t"] = [c]*3, [t]*3
    one = _run(p, _options(occupation_cutoff=.2))
    assert one.retained_rank == 1 and one.discarded_occupation_sum == pytest.approx(.125, abs=2e-14)
    np.testing.assert_allclose(one.coefficients_copy()@one.coefficients_copy().T, np.diag([1., 0.]), atol=2e-14)
    zero = _run(p, _options(occupation_cutoff=.6))
    assert zero.retained_rank == 0 and zero.discarded_occupation_sum == pytest.approx(.625, abs=2e-14)
    for cutoff in (.125, .5):
        with pytest.raises(RuntimeError, match="ambiguous"):
            _run(p, _options(occupation_cutoff=cutoff))


def test_union_rank_cutoff_cannot_silently_discard_relevant_original_columns():
    p = _problem(n=2, ranks=(1, 1, 1))
    p["c"] = [np.array([[1.], [0.]]), np.array([[np.cos(.01)], [np.sin(.01)]]), np.array([[1.], [0.]])]
    with pytest.raises(RuntimeError, match="original pair columns|discarded relevant"):
        _run(p, _options(union_absolute_rank_cutoff=.001))


def test_independent_pair_rotations_preserve_density_and_selected_projector():
    p = _problem(n=4, ranks=(2, 2, 2))
    baseline = _run(p)
    occupations = baseline.occupations_copy()
    cutoff = float((occupations[1]+occupations[2])/2)
    baseline = _run(p, _options(occupation_cutoff=cutoff))
    rng = np.random.default_rng(887)
    rotated = dict(p, c=[], t=[])
    for c, t in zip(p["c"], p["t"], strict=True):
        q = np.linalg.qr(rng.normal(size=(2, 2)))[0]
        rotated["c"].append(np.ascontiguousarray(c@q))
        rotated["t"].append(np.ascontiguousarray(q.T@t@q))
    result = _run(rotated, _options(occupation_cutoff=cutoff))
    np.testing.assert_allclose(result.occupations_copy(), baseline.occupations_copy(), atol=5e-12, rtol=2e-10)
    a, b = result.coefficients_copy(), baseline.coefficients_copy()
    np.testing.assert_allclose(a@a.T, b@b.T, atol=5e-11, rtol=2e-10)
    np.testing.assert_allclose(result.energies_copy(), baseline.energies_copy(), atol=5e-11)
    assert result.input_identity_sha256 != baseline.input_identity_sha256


def test_phase_inventory_unique_input_roles_and_replica_accounting():
    p = _problem(n=4, ranks=(2, 2, 2))
    inventory = _inventory(numerical_replicas=3, external_node_bytes=111,
        other_live_numerical_bytes_per_replica=222, other_live_control_bytes_per_replica=333)
    plan = _plan(p, inventory=inventory)
    n, m, u, r = 4, 6, 4, 2
    assert plan.borrowed_numerical_bytes == 8*(n*n+3*(n*r+r*r))
    assert plan.gram_phase_bytes == 40*m*m+8*m
    assert plan.union_phase_bytes == 8*n*u+16*m*m+8*m
    assert plan.density_phase_bytes == 8*n*u+16*u*u+8*u*r+16*u
    assert plan.density_eigen_phase_bytes == 8*n*u+40*u*u+8*u
    assert plan.selection_phase_bytes == 16*n*u+16*u*u+8*u
    semi = core.plan_restricted_pair_semicanonicalization(n, u)
    assert plan.semicanonical_phase_bytes == 8*n*u+8*u+semi.peak_owned_numerical_bytes
    assert plan.peak_owned_numerical_bytes == max(getattr(plan, name) for name in (
        "gram_phase_bytes", "union_phase_bytes", "density_phase_bytes", "density_eigen_phase_bytes",
        "selection_phase_bytes", "semicanonical_phase_bytes"))
    assert plan.control_storage_bytes_per_replica == plan.fixed_inventoried_object_bytes+333
    assert plan.total_node_bytes == 111+3*(plan.borrowed_numerical_bytes+plan.peak_owned_numerical_bytes
        +222+plan.control_storage_bytes_per_replica+65536)


@pytest.mark.parametrize("field", list(_CAPS))
def test_exact_caps_and_one_less_precede_any_floating_read(field):
    p = _problem()
    caps = _caps(_plan(p))
    assert _run(p, caps=caps).usable
    setattr(caps, field, getattr(caps, field)-1)
    p["t"][0][0, 0] = np.nan
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _run(p, caps=caps)


@pytest.mark.parametrize("field", ["fvv", "c", "t"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_borrowed_payloads_reject(field, value):
    p = _problem()
    (p[field][0] if isinstance(p[field], list) else p[field]).flat[0] = value
    with pytest.raises(OverflowError, match="finite"):
        _run(p)


def test_shape_orthogonality_diagonal_symmetry_and_no_coercion_gates():
    p = _problem((0, 0, 1), n=3, ranks=(2, 2, 2))
    p["t"][0][0, 1] += .01
    with pytest.raises(ValueError, match="diagonal|symmetric"):
        _run(p)
    p = _problem()
    p["c"][0][0, 0] += .01
    with pytest.raises(ValueError, match="orthonormal"):
        _run(p)
    p = _problem()
    p["fvv"][0, 1] += .01
    with pytest.raises(ValueError, match="symmetric"):
        _run(p)
    p = _problem()
    p["t"][0] = p["t"][0].astype(np.float32)
    with pytest.raises(ValueError, match="binary64"):
        _plan(p)
    p = _problem()
    p["c"][0] = np.asfortranarray(p["c"][0])
    with pytest.raises(ValueError, match="contiguous"):
        _plan(p)
    p = _problem()
    p["c"][0] = p["c"][0].tolist()
    with pytest.raises(ValueError, match="existing arrays"):
        _plan(p)
    p = _problem()
    p["occupied"] = (2, 0, 1)
    with pytest.raises(ValueError, match="sorted"):
        _plan(p)


@pytest.mark.parametrize("field,value", [("occupation_cutoff", np.nan), ("union_absolute_rank_cutoff", -1),
    ("union_relative_rank_cutoff", 1), ("maximum_union_column_reconstruction_error", 0),
    ("input_orthonormality_tolerance", 0), ("eigensystem_relative_reconstruction_tolerance", 1),
    ("eigenvector_orthogonality_tolerance", np.inf), ("density_negative_absolute_tolerance", -1),
    ("occupation_ambiguity_relative_guard", -1), ("union_max_sweeps", 0)])
def test_explicit_required_controls_no_unvalidated_defaults(field, value):
    with pytest.raises(ValueError):
        _plan(_problem(), _options(**{field: value}))


def test_backend_margin_and_replica_count_are_required():
    for field in ("backend_margin_bytes_per_replica", "numerical_replicas"):
        with pytest.raises(ValueError, match="margin|replica|positive"):
            _plan(_problem(), inventory=_inventory(**{field: 0}))


@pytest.mark.parametrize("exponent", [-400, 400])
def test_power_of_two_amplitude_scaling_preserves_representable_occupations(exponent):
    p = _problem((0, 0, 0), n=1, ranks=(1, 1, 1))
    c, t = np.ones((1, 1)), np.array([[np.ldexp(1., exponent)]])
    p["c"], p["t"] = [c]*3, [t]*3
    result = _run(p)
    assert result.occupations_copy()[0] == pytest.approx(np.ldexp(1., 2*exponent+1), rel=3e-14, abs=0)


def test_unrepresentable_density_and_mixed_scale_loss_fail_closed():
    p = _problem((0, 0, 0), n=1, ranks=(1, 1, 1))
    c, t = np.ones((1, 1)), np.array([[np.nextafter(0., 1.)]])
    p["c"], p["t"] = [c]*3, [t]*3
    with pytest.raises(OverflowError, match="underflow|rescal|nonzero|los"):
        _run(p)
    p = _problem((0, 0, 0), n=2, ranks=(2, 2, 2))
    c, t = np.eye(2), np.diag([1e300, 1e-300])
    p["c"], p["t"] = [c]*3, [t]*3
    with pytest.raises(OverflowError, match="scal|los|nonfinite"):
        _run(p)


def test_result_and_nested_semicanonical_ownership_and_signed_zero_identity():
    p = _problem((0, 0, 0), n=2, ranks=(2, 2, 2))
    c, t = np.eye(2), np.diag([.25, .125])
    p["c"], p["t"] = [c]*3, [t]*3
    result = _run(p)
    t[t == 0] = -0.
    duplicate = _run(p)
    assert duplicate.input_identity_sha256 == result.input_identity_sha256
    assert duplicate.result_identity_sha256 == result.result_identity_sha256
    original = result.coefficients_copy()
    result.coefficients_copy()[:] = 7
    options = result.options
    options.occupation_cutoff = 999
    assert result.options.occupation_cutoff == 0
    semi = result.semicanonical
    del p, result, duplicate
    gc.collect()
    np.testing.assert_array_equal(semi.coefficients_copy(), original)
