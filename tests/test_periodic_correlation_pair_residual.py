"""Bounded target-pair algebra against tiny independent dense MP2 equations."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


FIRST = core._PairResidualOccupiedLeg.FIRST
SECOND = core._PairResidualOccupiedLeg.SECOND


def _work(n, m):
    return 0 if m == 0 else n * m * m + n * n * m + n * n


def _controls(n, maximum_source=0, expected=0, *, target=(0, 1), noccupied=3,
              work_limit=None, floor=1e-12):
    result = core._PairResidualControls()
    result.occupied_label_count = noccupied
    result.target_first, result.target_second = target
    result.expected_coupling_count = expected
    result.maximum_source_dimension = maximum_source
    result.maximum_scalar_products = (_work(n, maximum_source) * expected
                                      if work_limit is None else work_limit)
    result.denominator_floor = floor
    return result


def _initialize(g, t, eps, controls=None, *, fii=-1.0, fjj=-1.2, cap=None):
    if controls is None:
        controls = _controls(len(g))
    if cap is None:
        cap = core._plan_pair_residual(len(g), controls.maximum_source_dimension,
                                       controls.expected_coupling_count).peak_owned_numerical_bytes
    return core._initialize_pair_residual(g, t, eps, fii, fjj, controls, cap)


def _add(accumulator, leg, k, stored, t, overlap, f):
    accumulator.accumulate(leg, k, *stored, np.ascontiguousarray(t),
                           np.ascontiguousarray(overlap), f)


def _dense_operator(t, occupied_fock, virtual_fock):
    # Independent full-tensor orbital operator; no extracted diagonals or
    # pair-overlap transforms appear in this oracle.
    return (np.einsum("ac,ijcb->ijab", virtual_fock, t)
            + np.einsum("ijac,cb->ijab", t, virtual_fock)
            - np.einsum("ik,kjab->ijab", occupied_fock, t)
            - np.einsum("jk,ikab->ijab", occupied_fock, t))


def _all_target_residuals(g, t, occupied_fock, virtual_fock, seed=914):
    no, _, nv, _ = t.shape
    eps, target_basis = np.linalg.eigh(virtual_fock)
    rng = np.random.default_rng(seed)
    source_bases = {(i, j): np.linalg.qr(rng.normal(size=(nv, nv)))[0]
                    for i in range(no) for j in range(i, no)}
    result = np.empty_like(t)
    transposed = 0
    for i in range(no):
        for j in range(no):
            control = _controls(nv, nv, 2 * (no - 1), target=(i, j), noccupied=no)
            transformed_g = np.ascontiguousarray(target_basis.T @ g[i, j] @ target_basis)
            transformed_t = np.ascontiguousarray(target_basis.T @ t[i, j] @ target_basis)
            accumulator = _initialize(transformed_g, transformed_t, np.ascontiguousarray(eps), control,
                                      fii=occupied_fock[i, i], fjj=occupied_fock[j, j])
            for leg, replaced in ((FIRST, i), (SECOND, j)):
                for k in range(no):
                    if k == replaced:
                        continue
                    ordered = (k, j) if leg == FIRST else (i, k)
                    stored = tuple(sorted(ordered))
                    basis = source_bases[stored]
                    source_t = basis.T @ t[stored] @ basis
                    overlap = target_basis.T @ basis
                    _add(accumulator, leg, k, stored, source_t, overlap, occupied_fock[replaced, k])
            done = accumulator.finish()
            transposed += done.diagnostics.transposed_source_count
            result[i, j] = target_basis @ done.residual_copy() @ target_basis.T
    assert transposed > 0
    return result


def test_full_space_fock_operator_both_occupied_legs_and_source_gauge_orientation():
    rng = np.random.default_rng(138)
    no, nv = 3, 3
    occupied = np.array([[-1.2, 0.11, -0.06], [0.11, -0.8, 0.07], [-0.06, 0.07, -1.1]])
    virtual = np.array([[0.8, 0.13, -0.04], [0.13, 1.3, 0.08], [-0.04, 0.08, 1.6]])
    raw = rng.normal(size=(no, no, nv, nv))
    t = 0.5 * (raw + raw.transpose(1, 0, 3, 2))
    raw = rng.normal(size=t.shape)
    g = 0.5 * (raw + raw.transpose(1, 0, 3, 2))
    expected = g + _dense_operator(t, occupied, virtual)
    actual = _all_target_residuals(g, t, occupied, virtual)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_dense_linear_mp2_solution_has_zero_coupled_residual_but_semicanonical_guess_does_not():
    rng = np.random.default_rng(42)
    no, nv = 3, 2
    occupied = np.array([[-1.2, 0.17, -0.06], [0.17, -0.8, 0.09], [-0.06, 0.09, -1.1]])
    virtual = np.array([[0.8, 0.13], [0.13, 1.3]])
    shape = (no, no, nv, nv)
    raw = rng.normal(size=shape)
    g = 0.5 * (raw + raw.transpose(1, 0, 3, 2))
    size = no * no * nv * nv
    unit = np.eye(size)
    operator = np.column_stack([_dense_operator(unit[:, i].reshape(shape), occupied, virtual).ravel()
                                for i in range(size)])
    t = np.linalg.solve(operator, -g.ravel()).reshape(shape)
    np.testing.assert_allclose(g + _dense_operator(t, occupied, virtual), 0.0, atol=2e-15)
    actual = _all_target_residuals(g, t, occupied, virtual)
    np.testing.assert_allclose(actual, 0.0, atol=2e-14)
    eps, basis = np.linalg.eigh(virtual)
    guess = np.empty_like(t)
    for i in range(no):
        for j in range(no):
            transformed = basis.T @ g[i, j] @ basis
            delta = eps[:, None] + eps[None, :] - occupied[i, i] - occupied[j, j]
            guess[i, j] = basis @ (-transformed / delta) @ basis.T
    assert np.linalg.norm(_all_target_residuals(g, guess, occupied, virtual)) > 0.05


@pytest.mark.parametrize("source_dimensions", [(1, 3, 0), (3, 1, 2), (0, 0, 0)])
def test_unequal_real_orthonormal_pair_spaces_reversed_amplitudes_and_zero_truncation(source_dimensions):
    rng = np.random.default_rng(109)
    n, common = 2, 5
    target_basis = np.linalg.qr(rng.normal(size=(common, n)))[0]
    g, t = rng.normal(size=(2, n, n))
    eps = np.array([0.7, 1.1])
    controls = _controls(n, max(source_dimensions), 3, target=(0, 2), noccupied=4)
    accumulator = _initialize(g, t, eps, controls)
    expected = g + (eps[:, None] + eps[None, :] + 2.2) * t
    slots = [(FIRST, 1, (1, 2), 0.18), (FIRST, 3, (2, 3), -0.21),
             (SECOND, 1, (1, 0), 0.13)]
    for m, (leg, k, stored, f) in zip(source_dimensions, slots):
        source_basis = np.linalg.qr(rng.normal(size=(common, m)))[0] if m else np.empty((common, 0))
        source_t = rng.normal(size=(m, m))
        overlap = target_basis.T @ source_basis
        expected_order = (k, 2) if leg == FIRST else (0, k)
        oriented = source_t if stored == expected_order else source_t.T
        expected -= f * overlap @ oriented @ overlap.T
        _add(accumulator, leg, k, stored, source_t, overlap, f)
    done = accumulator.finish()
    np.testing.assert_allclose(done.residual_copy(), expected, atol=3e-15, rtol=3e-15)
    assert done.diagnostics.accepted_coupling_count == 3
    assert done.diagnostics.transposed_source_count == 2
    assert done.diagnostics.zero_rank_source_count == source_dimensions.count(0)
    assert done.diagnostics.scalar_product_count == sum(_work(n, m) for m in source_dimensions)
    assert done.diagnostics.maximum_observed_source_input_bytes == 8 * (max(source_dimensions)**2 + n * max(source_dimensions))


def test_cross_source_compensation_preserves_small_term_during_large_cancellation():
    control = _controls(1, 1, 3, target=(0, 0), noccupied=4)
    accumulator = _initialize(np.zeros((1, 1)), np.zeros((1, 1)), np.ones(1), control,
                              fii=-1.0, fjj=-1.0)
    for k, amplitude in enumerate((1e16, 1.0, -1e16), 1):
        _add(accumulator, FIRST, k, (k, 0), np.array([[amplitude]]), np.ones((1, 1)), -1.0)
    result = accumulator.finish()
    assert result.residual(0, 0) == 1.0
    assert result.diagnostics.maximum_absolute_residual == result.diagnostics.residual_frobenius_norm == 1.0


def test_semicanonical_initializer_and_subnormal_denominator_conventions_match():
    g = np.array([[0.2, 0.3], [-0.4, 0.1]])
    eps = np.array([0.7, 1.1])
    initial = core._restricted_pair_semicanonical_mp2(g, eps, -1.0, -1.2, 1e-12, 32)
    t = initial.amplitudes_copy()
    result = _initialize(g, t, eps).finish()
    assert result.diagnostics.maximum_absolute_residual < 5e-17
    assert result.diagnostics.minimum_denominator == initial.minimum_denominator
    assert result.diagnostics.maximum_denominator == initial.maximum_denominator
    tiny = np.nextafter(0.0, 1.0)
    control = _controls(1, target=(0, 1), noccupied=2, floor=tiny)
    result = _initialize(np.array([[6 * tiny]]), np.array([[-1.0]]), np.array([3 * tiny]),
                         control, fii=0.0, fjj=0.0).finish()
    assert result.diagnostics.minimum_denominator == 6 * tiny
    assert result.residual(0, 0) == 0.0
    with pytest.raises(OverflowError, match="scaling loses range"):
        _initialize(np.ones((1, 1)), np.ones((1, 1)), np.array([3 * 2.0**-51]),
                    _controls(1, floor=2.0**-54), fii=2.0**1023, fjj=-2.0**1023)


def test_denominator_floor_is_strict_and_never_a_shift():
    g, t, eps = np.ones((1, 1)), np.ones((1, 1)), np.ones(1)
    for floor in (2.0, 3.0):
        with pytest.raises(ValueError, match="strictly exceed"):
            _initialize(g, t, eps, _controls(1, floor=floor), fii=0.0, fjj=0.0)
    result = _initialize(g, t, eps, _controls(1, floor=np.nextafter(2.0, 0.0)),
                         fii=0.0, fjj=0.0).finish()
    assert result.residual(0, 0) == 3.0


@pytest.mark.parametrize("n,m,k", [(1, 0, 0), (1, 1, 3), (2, 3, 4), (9, 2, 17)])
def test_exact_memory_and_work_inventory(n, m, k):
    plan = core._plan_pair_residual(n, m, k)
    assert plan.borrowed_target_input_bytes == 16 * n * n + 8 * n
    assert plan.maximum_borrowed_source_input_bytes == 8 * (m * m + n * m)
    assert plan.residual_bytes == plan.compensation_bytes == 8 * n * n
    assert plan.projection_workspace_bytes == 8 * n * m
    assert plan.peak_owned_numerical_bytes == 16 * n * n + 8 * n * m
    assert plan.output_numerical_bytes == 8 * n * n
    assert plan.maximum_scalar_products_per_source == _work(n, m)
    assert plan.maximum_total_scalar_products == k * _work(n, m)


def test_memory_and_source_work_admission_precede_numerical_reads():
    g = np.full((2, 2), np.nan)
    t, eps = np.zeros((2, 2)), np.ones(2)
    controls = _controls(2, 3, 1)
    peak = 16 * 4 + 8 * 2 * 3
    for cap in (0, peak - 1):
        with pytest.raises((ValueError, RuntimeError), match="byte cap"):
            _initialize(g, t, eps, controls, cap=cap)
    with pytest.raises(ValueError, match="finite"):
        _initialize(g, t, eps, controls, cap=peak)
    controls.maximum_scalar_products = _work(2, 3) - 1
    acc = _initialize(np.zeros((2, 2)), t, eps, controls)
    with pytest.raises((ValueError, RuntimeError), match="work budget"):
        _add(acc, FIRST, 2, (2, 1), np.full((3, 3), np.nan), np.full((2, 3), np.nan), 0.1)
    assert not acc.is_open
    with pytest.raises(RuntimeError, match="aborted"):
        acc.finish()


@pytest.mark.parametrize("failure", ["self", "duplicate", "out_of_order", "wrong_pair", "too_many", "rank"])
def test_arithmetic_stream_contract_failures_poison_accumulator(failure):
    n = 2
    expected = 1 if failure == "too_many" else 2
    acc = _initialize(np.zeros((n, n)), np.zeros((n, n)), np.ones(n),
                      _controls(n, 2, expected, target=(0, 1), noccupied=4))
    if failure in ("duplicate", "out_of_order", "too_many"):
        _add(acc, FIRST, 2, (2, 1), np.ones((2, 2)), np.eye(2), 0.2)
    if failure == "self":
        args = (FIRST, 0, (0, 1), np.ones((2, 2)), np.eye(2), 0.2)
    elif failure == "out_of_order":
        args = (FIRST, 1, (1, 1), np.ones((2, 2)), np.eye(2), 0.2)
    elif failure == "wrong_pair":
        args = (SECOND, 2, (1, 2), np.ones((2, 2)), np.eye(2), 0.2)
    elif failure == "rank":
        args = (FIRST, 2, (2, 1), np.ones((3, 3)), np.ones((2, 3)), 0.2)
    else:
        args = (FIRST, 2, (2, 1), np.ones((2, 2)), np.eye(2), 0.2)
    with pytest.raises((ValueError, RuntimeError)):
        _add(acc, *args)
    assert not acc.is_open
    with pytest.raises(RuntimeError):
        acc.finish()


def test_incomplete_provider_failure_and_finished_accumulators_cannot_publish_partial_results():
    def fresh():
        return _initialize(np.zeros((1, 1)), np.zeros((1, 1)), np.ones(1), _controls(1, 1, 1))

    incomplete = fresh()
    with pytest.raises(RuntimeError, match="incomplete"):
        incomplete.finish()
    assert not incomplete.is_open
    aborted = fresh()
    try:
        raise LookupError("synthetic source provider failed")
    except LookupError:
        aborted.abort()
    assert not aborted.is_open
    with pytest.raises(RuntimeError, match="aborted"):
        aborted.finish()
    acc = fresh()
    _add(acc, FIRST, 2, (2, 1), np.ones((1, 1)), np.ones((1, 1)), 0.3)
    result = acc.finish()
    assert not acc.is_open
    with pytest.raises(RuntimeError, match="finished"):
        acc.finish()
    assert result.residual(0, 0) == -0.3


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_target_and_source_values_are_never_silently_skipped(bad):
    for g, t, eps in ((np.array([[bad]]), np.ones((1, 1)), np.ones(1)),
                      (np.ones((1, 1)), np.array([[bad]]), np.ones(1)),
                      (np.ones((1, 1)), np.ones((1, 1)), np.array([bad]))):
        with pytest.raises(ValueError, match="finite"):
            _initialize(g, t, eps)
    for source_t, overlap, coupling in ((np.array([[bad]]), np.ones((1, 1)), 0.0),
                                       (np.ones((1, 1)), np.array([[bad]]), 0.0),
                                       (np.ones((1, 1)), np.ones((1, 1)), bad)):
        acc = _initialize(np.zeros((1, 1)), np.zeros((1, 1)), np.ones(1), _controls(1, 1, 1))
        with pytest.raises(ValueError, match="finite"):
            _add(acc, FIRST, 2, (2, 1), source_t, overlap, coupling)
        assert not acc.is_open


def test_numerical_overflow_aborts_and_product_underflow_is_visible():
    huge = np.finfo(float).max
    with pytest.raises(OverflowError, match="initialization"):
        _initialize(np.zeros((1, 1)), np.array([[huge]]), np.ones(1))
    acc = _initialize(np.zeros((1, 1)), np.zeros((1, 1)), np.ones(1), _controls(1, 1, 1))
    with pytest.raises(OverflowError, match="product overflow"):
        _add(acc, FIRST, 2, (2, 1), np.array([[2.0]]), np.ones((1, 1)), huge)
    assert not acc.is_open
    small = _initialize(np.zeros((1, 1)), np.zeros((1, 1)), np.ones(1), _controls(1, 1, 1))
    _add(small, FIRST, 2, (2, 1), np.array([[np.nextafter(0.0, 1.0)]]), np.array([[0.5]]), 1.0)
    result = small.finish()
    assert result.diagnostics.product_underflow_count == 1


def test_float64_only_boundary_shapes_alignment_no_copy_and_result_lifetime():
    g, t, eps = np.ones((2, 2)), np.zeros((2, 2)), np.ones(2)
    for bad in (g.astype(np.float32), g.astype(complex), np.asfortranarray(g), np.ones((2, 3))):
        with pytest.raises(ValueError, match="float64"):
            core._initialize_pair_residual(bad, t, eps, -1.0, -1.2, _controls(2), 1000)
    misaligned = np.ndarray((2, 2), dtype=np.float64, buffer=bytearray(33), offset=1)
    misaligned[:] = g
    with pytest.raises(ValueError, match="alignment"):
        _initialize(misaligned, t, eps)
    for array in (g, t, eps):
        array.flags.writeable = False
    acc = _initialize(g, t, eps, _controls(2, 2, 1))
    source = np.ones((2, 2), complex)
    with pytest.raises(ValueError, match="float64"):
        acc.accumulate(FIRST, 2, 2, 1, source, np.eye(2), 0.1)
    assert not acc.is_open
    finished = _initialize(g, t, eps).finish()
    del g, t, eps
    gc.collect()
    np.testing.assert_array_equal(finished.residual_copy(), np.ones((2, 2)))
    copy = finished.residual_copy()
    copy[:] = 0
    np.testing.assert_array_equal(finished.residual_copy(), np.ones((2, 2)))
    with pytest.raises(IndexError):
        finished.residual(2, 0)


def test_count_overflow_empty_target_invalid_slots_and_diagnostic_work_caps():
    with pytest.raises(ValueError, match="positive"):
        core._plan_pair_residual(0, 0, 0)
    for counts in ((2**32, 1, 1), (2, 2**63, 1), (16, 16, 2**63)):
        with pytest.raises((OverflowError, ValueError, RuntimeError)):
            core._plan_pair_residual(*counts)
    g, t, eps = np.ones((1, 1)), np.ones((1, 1)), np.ones(1)
    for changes in ({"occupied_label_count": 0}, {"target_first": 3},
                    {"expected_coupling_count": 5}, {"denominator_floor": 0.0}):
        control = _controls(1)
        for name, value in changes.items():
            setattr(control, name, value)
        with pytest.raises(ValueError):
            _initialize(g, t, eps, control)
    with pytest.raises(ValueError, match="diagonals"):
        _initialize(g, t, eps, _controls(1, target=(0, 0)), fii=-1.0, fjj=-2.0)
    with pytest.raises((ValueError, RuntimeError), match="limited"):
        _initialize(np.zeros((17, 17)), np.zeros((17, 17)), np.ones(17))
