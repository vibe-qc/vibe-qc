"""All-k unorthogonalized IAO charges and real differential, no optimization."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_bloch_iao import _ao_oracle, _inputs, _make as _iao
from tests.test_periodic_correlation_wannier import _reference


def _case(mesh=(3, 1, 1), frozen=(1,)):
    reference, cells, coefficients, overlaps, _ = _reference(mesh, frozen_indices=frozen)
    state = reference.state
    p = state.n_frozen_core + state.n_correlated_occupied
    r = state.n_effective_orbitals
    labels = np.array([7 if i % 2 == 0 else 901 for i in range(r)], dtype=np.uint64)
    points, b, d = [], [], []
    for k in range(len(cells)):
        s12, s22, _ = _inputs(coefficients[k], overlaps[k], r, seed=981 + k)
        points.append(_iao(reference, s12, s22, labels, point=k))
        active_mask = np.asarray(state.correlated_occupied_mask(k), bool)
        occupied = np.flatnonzero(active_mask | np.asarray(state.frozen_core_mask(k), bool))
        assert len(occupied) == p
        active = active_mask[occupied]
        _, _, bb, dd = _ao_oracle(coefficients[k], overlaps[k], s12, s22, occupied)
        b.append(bb[:, active])
        d.append(dd[:, active])
    rng = np.random.default_rng(912)
    gauges = np.array([np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
                       for _ in cells])
    return reference, points, gauges, np.array(b), np.array(d), labels


def _options():
    out = core._PeriodicCorrelationIAOPMOptions()
    out.gauge_unitarity_tolerance = 3e-12
    out.charge_normalization_tolerance = 2e-10
    return out


def _caps(plan):
    out = core._PeriodicCorrelationIAOPMCaps()
    out.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    out.maximum_work_units = plan.work_units
    return out


def _evaluate(case, *, gauges=None, caps=None, points=None):
    reference, owners, u, _, _, labels = case
    plan = core._plan_periodic_correlation_iao_pm(reference, len(labels))
    return core._evaluate_periodic_correlation_iao_pm(
        reference, owners if points is None else points, u if gauges is None else gauges,
        _options(), _caps(plan) if caps is None else caps,
    )


def _oracle(mesh, u, b, d, labels):
    cells = np.array(list(product(*(range(n) for n in mesh))))
    fourier = np.exp(2j * np.pi * cells @ (cells / np.array(mesh)).T) / len(cells)
    v = np.einsum("Rk,krj,kjn->Rrn", fourier, b, u)
    w = np.einsum("Rk,krj,kjn->Rrn", fourier, d, u)
    weights = np.zeros(v.shape, float)
    objective = 0.0
    normalization = np.zeros(u.shape[-1])
    for atom in np.unique(labels):
        on_atom = labels == atom
        charges = np.sum(v[:, on_atom].conj() * w[:, on_atom], axis=1).real
        objective += np.sum(charges**4)
        normalization += charges.sum(axis=0)
        weights[:, on_atom] = (4 * charges**3)[:, None, :]
    gradient = (np.einsum("Rk,krj,Rrn->kjn", fourier.conj(), b.conj(), weights * w)
                + np.einsum("Rk,krj,Rrn->kjn", fourier.conj(), d.conj(), weights * v))
    return objective, gradient, normalization


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("frozen", [(), (1,)])
def test_independent_original_ao_iao_and_finite_torus_charge_oracle(mesh, frozen):
    case = _case(mesh, frozen)
    actual = _evaluate(case)
    expected, gradient, normalization = _oracle(mesh, *case[2:])
    assert actual.objective == pytest.approx(expected, abs=8e-12, rel=2e-11)
    np.testing.assert_allclose(actual.gradient_copy(), gradient, atol=1e-11, rtol=3e-11)
    np.testing.assert_allclose(normalization, 1.0, atol=3e-12)
    assert actual.maximum_charge_normalization_residual < 3e-12
    assert actual.gradient_frobenius_norm == pytest.approx(np.linalg.norm(gradient), rel=3e-11)
    assert not hasattr(actual, "localization_converged")


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1), (2, 2, 2)])
def test_real_euclidean_gradient_matches_unitary_directional_finite_difference(mesh):
    case = _case(mesh)
    u = case[2]
    result = _evaluate(case)
    rng = np.random.default_rng(280)
    x = rng.normal(size=u.shape) + 1j * rng.normal(size=u.shape)
    h = x - x.swapaxes(1, 2).conj()
    h /= np.linalg.norm(h)
    tangent = u @ h
    predicted = np.vdot(result.gradient_copy(), tangent).real
    def displaced(step):
        rotations = []
        for point in h:
            values, vectors = np.linalg.eigh(-1j * point)
            rotations.append((vectors * np.exp(1j * step * values)) @ vectors.conj().T)
        return np.ascontiguousarray(u @ np.array(rotations))
    plus, minus = _evaluate(case, gauges=displaced(2e-5)), _evaluate(case, gauges=displaced(-2e-5))
    observed = (plus.objective - minus.objective) / 4e-5
    assert abs(predicted) > 1e-6
    assert observed == pytest.approx(predicted, abs=3e-9, rel=3e-7)


def test_full_real_euclidean_gradient_against_unconstrained_oracle_differential():
    case = _case()
    rng = np.random.default_rng(904)
    direction = rng.normal(size=case[2].shape) + 1j * rng.normal(size=case[2].shape)
    direction /= np.linalg.norm(direction)
    result = _evaluate(case)
    step = 2e-6
    plus = _oracle((3, 1, 1), case[2] + step * direction, *case[3:])[0]
    minus = _oracle((3, 1, 1), case[2] - step * direction, *case[3:])[0]
    observed = (plus - minus) / (2 * step)
    assert np.vdot(result.gradient_copy(), direction).real == pytest.approx(observed, abs=3e-9, rel=2e-7)


def test_one_orbital_euclidean_gradient_is_eight_u_not_a_convergence_norm():
    reference, _, c, s, _ = _reference((1, 1, 1), nactive=1)
    cross = np.ascontiguousarray(s[0] @ c[0, :, :1])
    points = [_iao(reference, cross, np.ones((1, 1), complex), np.array([17], np.uint64), point=0)]
    gauge = np.array([[[np.exp(0.31j)]]])
    plan = core._plan_periodic_correlation_iao_pm(reference, 1)
    result = core._evaluate_periodic_correlation_iao_pm(reference, points, gauge, _options(), _caps(plan))
    assert result.objective == pytest.approx(1.0, abs=3e-13)
    np.testing.assert_allclose(result.gradient_copy(), 8 * gauge, atol=3e-12)
    assert result.gradient_frobenius_norm == pytest.approx(8.0, abs=3e-12)
    # The ONLY unitary direction here is an irrelevant orbital phase.
    assert abs(np.vdot(result.gradient_copy(), 1j * gauge).real) < 3e-13


def test_common_translation_column_phases_and_permutation_preserve_objective():
    case = _case()
    u = case[2]
    base = _evaluate(case)
    factors = np.exp(-2j * np.pi * np.arange(3) / 3)
    shifted = np.ascontiguousarray(u * factors[:, None, None])
    assert _evaluate(case, gauges=shifted).objective == pytest.approx(base.objective, abs=3e-13)
    rotated = np.ascontiguousarray(u[:, :, ::-1] * np.exp(1j * np.array([0.23, 0.59])))
    assert _evaluate(case, gauges=rotated).objective == pytest.approx(base.objective, abs=3e-13)


def test_exact_live_owner_and_single_cell_workspace_inventory():
    case = _case()
    reference, points, gauges, _, _, labels = case
    p = core._plan_periodic_correlation_iao_pm(reference, len(labels))
    k, r, o = p.n_points, p.n_minimal, p.n_active
    assert p.gradient_bytes == p.gradient_compensation_bytes == gauges.nbytes == 16 * k * o**2
    assert p.cell_workspace_bytes == 72 * r * o + 8 * o
    assert p.active_index_bytes == 8 * k * o
    assert p.peak_owned_numerical_bytes == 32 * k * o**2 + 72 * r * o + 8 * o + 8 * k * o
    assert p.live_iao_numerical_bytes == sum(point.memory.output_numerical_bytes for point in points)
    dims = reference.dimensions
    assert p.required_node_memory_bytes == (dims.external_bytes + dims.shared_bytes
        + dims.per_rank_bytes + dims.localization_window_bytes_per_rank
        + p.peak_owned_numerical_bytes + p.borrowed_gauge_bytes
        + p.borrowed_owner_pointer_bytes + p.live_iao_numerical_bytes)


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_work_units"])
@pytest.mark.parametrize("zero", [True, False])
def test_caps_precede_untrusted_gauge_scan(field, zero):
    case = _case()
    caps = _caps(core._plan_periodic_correlation_iao_pm(case[0], len(case[-1])))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _evaluate(case, caps=caps, gauges=np.full_like(case[2], np.nan))


def test_points_require_exact_state_owner_and_correct_point_order():
    case = _case()
    other = _case()
    assert case[0].state.state_identity_sha256 == other[0].state.state_identity_sha256
    with pytest.raises(ValueError, match="point owners"):
        _evaluate(case, points=other[1])
    with pytest.raises(ValueError, match="point owners"):
        _evaluate(case, points=case[1][::-1])
    with pytest.raises(ValueError, match="unitary"):
        _evaluate(case, gauges=np.ascontiguousarray(0.9 * case[2]))


def test_gradient_copy_is_detached_and_no_gauge_dtype_conversion():
    case = _case()
    result = _evaluate(case)
    expected = result.gradient_copy()
    changed = result.gradient_copy()
    changed[:] = 0
    np.testing.assert_array_equal(result.gradient_copy(), expected)
    with pytest.raises(ValueError, match="complex128"):
        _evaluate(case, gauges=case[2].real.copy())
