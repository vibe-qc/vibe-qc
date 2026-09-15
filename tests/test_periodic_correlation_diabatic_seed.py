"""Tiny diabatic-gauge oracles, not SCF or chemical-localization benchmarks."""

from __future__ import annotations

import gc
import hashlib
import math
import struct
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_pair_topology import _make_budget, _make_dimensions
from tests.test_periodic_correlation_pao_domain import _reference as _reduced_reference
from tests.test_periodic_correlation_wannier import (
    _home,
    _make as _wannier,
    _options as _wannier_options,
    _reference,
)


def _options(reference=None, **changes):
    result = core._PeriodicCorrelationDiabaticSeedOptions()
    result.absolute_tolerance = 2e-11
    result.relative_tolerance = 2e-11
    result.unitarity_tolerance = 2e-11
    result.cholesky_absolute_floor = 1e-14
    result.cholesky_relative_floor = 1e-12
    result.singular_absolute_floor = 1e-14
    result.singular_relative_floor = 1e-12
    result.jacobi_max_sweeps = 64
    result.jacobi_relative_tolerance = 2e-15
    result.maximum_work_units = 2**63 - 1
    for name, value in changes.items():
        setattr(result, name, value)
    if reference is not None and "maximum_work_units" not in changes:
        result.maximum_work_units = _plan(reference, result).maximum_work_units
    return result


def _plan(reference, controls=None):
    if controls is None:
        controls = _options()
    state = reference.state
    return core._plan_periodic_correlation_diabatic_seed(
        state.mesh, state.n_basis, state.n_correlated_occupied,
        state.n_effective_orbitals, controls.jacobi_max_sweeps,
    )


def _make(reference, *, controls=None, cap=None):
    if controls is None:
        controls = _options(reference)
    if cap is None:
        cap = _plan(reference, controls).peak_owned_numerical_bytes
    return core._make_periodic_correlation_diabatic_seed(reference, cap, controls)


def _gauges(result):
    return np.array([result.point_gauge_copy(k) for k in range(result.memory.n_cells)])


def _native_wannier(reference, seed, cap=None):
    if cap is None:
        cap = core._plan_periodic_correlation_wannier(
            reference.state.mesh, reference.state.n_basis,
            reference.state.n_correlated_occupied,
        ).peak_owned_numerical_bytes
    return core._make_periodic_correlation_wannier_from_diabatic_seed(
        reference, seed, cap, _wannier_options(),
    )


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("ranks,workers", [(1, 1), (1, 2)])
def test_owner_aware_wannier_bridge_charges_entire_live_seed_without_changing_numerics(
    mesh, ranks, workers,
):
    reference, *_ = _reference(mesh, frozen_indices=(1,))
    dimensions, budget = reference.dimensions, reference.budget
    # Re-admission shares the existing state and adds its resident bytes once.
    dimensions.external_bytes -= reference.state.resident_bytes
    budget.mpi_ranks, budget.workers_per_rank = ranks, workers
    reference = core._make_periodic_correlation_admitted_reference(
        reference.state, dimensions, budget,
    )
    seed = _make(reference)
    raw = _wannier(reference, _gauges(seed))
    native = _native_wannier(reference, seed)
    np.testing.assert_array_equal(_home(native), _home(raw))
    assert native.wannier_identity_sha256 == raw.wannier_identity_sha256
    assert native.gauge_payload_sha256 == raw.gauge_payload_sha256
    assert native.diagnostics.live_diabatic_seed_index_bytes == seed.memory.retained_index_bytes
    assert raw.diagnostics.live_diabatic_seed_index_bytes == 0
    assert native.diagnostics.required_node_memory_bytes == (
        raw.diagnostics.required_node_memory_bytes
        + ranks * workers * seed.memory.retained_index_bytes
    )
    dims = reference.dimensions
    assert native.diagnostics.required_node_memory_bytes == (
        dims.external_bytes + dims.shared_bytes
        + ranks * (dims.per_rank_bytes + dims.localization_window_bytes_per_rank)
        + ranks * workers * (native.memory.peak_owned_numerical_bytes
                              + seed.memory.output_numerical_bytes)
    )
    for cap in (0, native.memory.peak_owned_numerical_bytes - 1):
        with pytest.raises((ValueError, RuntimeError), match="byte cap"):
            _native_wannier(reference, seed, cap)
    expected = _home(native)
    del seed, raw, reference
    gc.collect()
    np.testing.assert_array_equal(_home(native), expected)


def test_native_wannier_bridge_rejects_equal_content_from_a_distinct_reference_owner():
    reference, *_ = _reference()
    other, *_ = _reference()
    assert reference.state.state_identity_sha256 == other.state.state_identity_sha256
    seed = _make(reference)
    with pytest.raises(ValueError, match="exact admitted state allocation"):
        _native_wannier(other, seed)
    dims = reference.dimensions
    dims.external_bytes -= reference.state.resident_bytes
    dims.allocation_identity = "9" * 64
    other_allocation = core._make_periodic_correlation_admitted_reference(
        reference.state, dims, reference.budget,
    )
    with pytest.raises(ValueError, match="exact admitted state allocation"):
        _native_wannier(other_allocation, seed)


def _physical(result, coefficients):
    return np.array([
        coefficients[k][:, [result.active_band(k, i) for i in range(result.memory.n_active)]]
        @ result.point_gauge_copy(k)
        for k in range(result.memory.n_cells)
    ])


def _pivoted_cholesky(density, rank):
    """Independent dense test oracle, with direct Schur complement updates."""
    residual = density.real.copy()
    columns = []
    pivots = []
    for _ in range(rank):
        pivot = int(np.argmax(np.diag(residual)))
        column = residual[:, pivot] / np.sqrt(residual[pivot, pivot])
        columns.append(column)
        pivots.append(pivot)
        residual -= np.outer(column, column)
        residual[pivot, :] = 0.0
        residual[:, pivot] = 0.0
    return np.column_stack(columns), pivots


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1),
                                  (2, 2, 2), (4, 2, 2), (3, 2, 2)])
def test_seed_dense_cholesky_svd_oracle_and_translation_preserving_wannier_bridge(mesh):
    reference, cells, c, s, masks = _reference(mesh)
    result = _make(reference)
    u, physical = _gauges(result), _physical(result, c)
    active0 = c[0][:, masks[0]]
    anchor, pivots = _pivoted_cholesky(active0 @ active0.conj().T, 2)
    np.testing.assert_allclose(physical[0], anchor, atol=3e-14)
    assert [result.gamma_pivot(i) for i in range(2)] == pivots
    for k in range(len(cells)):
        active = c[k][:, masks[k]]
        left, singular, right = np.linalg.svd(active.conj().T @ anchor)
        expected = left @ right
        np.testing.assert_allclose(u[k], expected, atol=4e-13, rtol=4e-13)
        np.testing.assert_allclose(u[k].conj().T @ u[k], np.eye(2), atol=3e-13)
        np.testing.assert_allclose(physical[k].conj().T @ s[k] @ physical[k], np.eye(2), atol=3e-13)
        np.testing.assert_allclose(physical[k] @ physical[k].conj().T,
                                   active @ active.conj().T, atol=3e-13)
        # The polar objective reaches the singular-value sum; no Gamma-only
        # rotation, overlap-weighted similarity or transpose substitute.
        np.testing.assert_allclose(np.trace(u[k].conj().T @ active.conj().T @ anchor),
                                   singular.sum(), atol=4e-13)
    lookup = {tuple(cell): k for k, cell in enumerate(cells)}
    for k, cell in enumerate(cells):
        partner = lookup[tuple((-cell) % mesh)]
        np.testing.assert_allclose(physical[k], physical[partner].conj(), atol=3e-13)
    expected_home = np.einsum("Rk,kmi->Rmi",
                             np.exp(2j * np.pi * cells @ (cells / np.array(mesh)).T),
                             physical) / len(cells)
    wannier = _wannier(reference, np.ascontiguousarray(u))
    np.testing.assert_allclose(_home(wannier), expected_home, atol=4e-13)
    assert np.max(np.abs(expected_home.imag)) < 3e-13
    expected_self = math.prod(2 if axis % 2 == 0 else 1 for axis in mesh)
    assert result.diagnostics.self_inverse_point_count == expected_self
    assert result.diagnostics.sewn_pair_count == (len(cells) - expected_self) // 2
    assert result.diagnostics.polar_point_count == (len(cells) + expected_self) // 2 - 1
    assert result.diagnostics.charged_work_units == result.memory.maximum_work_units
    assert not hasattr(result, "localization_converged")


@pytest.mark.parametrize("mesh", [(2, 2, 2), (4, 2, 2), (3, 1, 1)])
def test_arbitrary_complex_degenerate_bloch_gauges_all_even_mesh_trim_frames(mesh):
    reference, cells, c, _, _ = _reference(mesh, nactive=3)
    original = _make(reference)
    rng = np.random.default_rng(3801)
    rotations = np.array([np.linalg.qr(rng.normal(size=(3, 3))
                                      + 1j * rng.normal(size=(3, 3)))[0]
                          for _ in cells])
    altered, _, changed_c, _, _ = _reference(mesh, nactive=3, canonical_rotations=rotations)
    result = _make(altered)
    # Raw U(-k)=conj(U(k)) is false for these arbitrary input gauges. The
    # invariant statement is C(-k)U(-k)=conj(C(k)U(k)).
    np.testing.assert_allclose(_physical(result, changed_c), _physical(original, c),
                               atol=2e-12, rtol=2e-12)
    for k in range(len(cells)):
        np.testing.assert_allclose(rotations[k] @ result.point_gauge_copy(k),
                                   original.point_gauge_copy(k), atol=2e-12)
    if mesh == (2, 2, 2):
        assert result.diagnostics.self_inverse_point_count == 8
        assert np.max(np.abs(_gauges(result).imag)) > 0.1
        assert np.max(np.abs(_physical(result, changed_c).imag)) < 2e-12


def test_noncontiguous_active_mask_and_frozen_core_are_not_mixed():
    mesh = (3, 1, 1)
    rng = np.random.default_rng(90)
    rotations = np.tile(np.eye(3, dtype=complex), (3, 1, 1))
    active = np.array([0, 2])
    for k in range(3):
        rotations[k][np.ix_(active, active)] = np.linalg.qr(
            rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        rotations[k, 1, 1] = np.exp(0.29j * (k + 1))
    reference, cells, c, s, _ = _reference(mesh, frozen_indices=(1,), canonical_rotations=rotations)
    result = _make(reference)
    for k in range(len(cells)):
        assert [result.active_band(k, i) for i in range(2)] == [0, 2]
        np.testing.assert_allclose(c[k, :, 1].conj() @ s[k] @ _physical(result, c)[k],
                                   np.zeros(2), atol=2e-13)
    broken = rotations.copy()
    broken[1] = np.eye(3, dtype=complex)[[1, 0, 2]]
    bad, *_ = _reference(mesh, frozen_indices=(1,), canonical_rotations=broken)
    with pytest.raises(ValueError, match="active projector.*time reversal"):
        _make(bad)


def test_reduced_retained_overlap_space_preserves_only_certified_active_columns():
    reference, _, c, s, _ = _reduced_reference((3, 1, 1), reduced=True)
    result = _make(reference)
    assert result.memory.n_effective_orbitals == 3 < result.memory.n_basis == 4
    assert result.memory.n_active == 1
    actual = _physical(result, c)
    for k in range(3):
        np.testing.assert_allclose(actual[k] @ actual[k].conj().T,
                                   c[k, :, :1] @ c[k, :, :1].conj().T, atol=2e-13)
        np.testing.assert_allclose(c[k, :, 1:].conj().T @ s[k] @ actual[k], 0.0, atol=2e-13)


def _rotating_reference(singular_value):
    """Two real TRIM points with a tunable occupied/anchor similarity rank."""
    mesh = (2, 1, 1)
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = "1" * 64
    data.periodic_dimension = 3
    data.mesh = mesh
    data.is_shift = (0, 0, 0)
    data.reciprocal_lattice = np.diag([2.0, 3.0, 5.0])
    data.converged = True
    data.n_basis = data.n_effective_orbitals = 3
    data.electrons_per_cell = 4
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    energies = np.array([-1.5, -1.5, 0.8])
    physical = []
    for k in range(2):
        c = np.eye(3, dtype=complex)
        if k:
            cosine = singular_value
            sine = np.sqrt(1.0 - cosine**2)
            c = np.array([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]], dtype=complex)
        data.add_kpoint(np.array([float(k), 0, 0]), 0.5, np.eye(3, dtype=complex),
                        c @ np.diag(energies) @ c.conj().T, c, energies,
                        np.array([2.0, 2.0, 0.0]), [0, 0, 0], [1, 1, 0], [0, 0, 1])
        physical.append(c)
    state = core._make_periodic_restricted_mean_field_state(data)
    dims = _make_dimensions(mesh, 2)
    dims.n_basis = dims.n_effective_orbitals = 3
    dims.n_home_total_occupied = 2
    dims.n_home_virtual = 1
    dims.n_auxiliary = 6
    dims.domain_ao_support_upper_bound = dims.domain_pao_upper_bound = dims.domain_pno_upper_bound = 3
    dims.domain_local_auxiliary_upper_bound = 6
    return core._make_periodic_correlation_admitted_reference(state, dims, _make_budget()), np.array(physical)


@pytest.mark.parametrize("singular_value", [0.0, 1e-10, 1e-6])
def test_deficient_procrustes_rank_rejects_without_dropping_active_bands(singular_value):
    reference, _ = _rotating_reference(singular_value)
    controls = _options(reference, singular_absolute_floor=1e-5, singular_relative_floor=1e-12)
    with pytest.raises(ValueError, match="similarity.*rank"):
        _make(reference, controls=controls)


def test_near_rank_boundary_remains_unitary_without_normal_equation_squaring():
    reference, c = _rotating_reference(1e-7)
    result = _make(reference, controls=_options(reference, singular_absolute_floor=1e-8,
                                               singular_relative_floor=1e-10))
    assert result.memory.n_active == 2
    np.testing.assert_allclose(_gauges(result), np.tile(np.eye(2), (2, 1, 1)), atol=2e-13)
    assert result.diagnostics.minimum_similarity_singular_value == pytest.approx(1e-7, rel=1e-8)
    assert np.max(np.abs(_physical(result, c).imag)) == 0.0


def test_gamma_cholesky_floor_is_strict_and_no_orbital_is_forced():
    reference, *_ = _reference((1, 1, 1))
    with pytest.raises(ValueError, match="Cholesky.*rank"):
        _make(reference, controls=_options(reference, cholesky_absolute_floor=0.9))
    result = _make(reference)
    assert result.diagnostics.minimum_cholesky_pivot > 0.0
    assert result.diagnostics.minimum_similarity_singular_value == 0.0
    assert result.diagnostics.polar_point_count == 0


def test_caps_and_stepwise_work_precede_relevant_reference_reads_and_stages():
    bad_reference, *_ = _reference(broken_tr=True)
    peak = _plan(bad_reference).peak_owned_numerical_bytes
    for cap in (0, peak - 1):
        with pytest.raises((ValueError, RuntimeError), match="byte cap"):
            _make(bad_reference, cap=cap)
    controls = _options(bad_reference)
    controls.maximum_work_units = _plan(bad_reference).reference_preflight_work_units - 1
    with pytest.raises((ValueError, RuntimeError), match="work budget"):
        _make(bad_reference, controls=controls)
    with pytest.raises(ValueError, match="time reversal"):
        _make(bad_reference, cap=peak)
    reference, *_ = _reference()
    plan = _plan(reference)
    for work in (plan.reference_preflight_work_units,
                 plan.reference_preflight_work_units + plan.anchor_work_units,
                 plan.maximum_work_units - 1):
        with pytest.raises((ValueError, RuntimeError), match="work budget"):
            _make(reference, controls=_options(reference, maximum_work_units=work))
    done = _make(reference)
    assert done.diagnostics.charged_work_units == plan.maximum_work_units


@pytest.mark.parametrize("mesh,b,n,effective", [((1, 1, 1), 3, 2, 3),
                                               ((2, 2, 2), 6, 4, 5),
                                               ((8, 8, 8), 45, 4, 44)])
def test_exact_fixed_workspace_no_placed_space_and_work_count_plan(mesh, b, n, effective):
    plan = core._plan_periodic_correlation_diabatic_seed(mesh, b, n, effective, 64)
    k = math.prod(mesh)
    assert plan.retained_gauge_bytes == 16 * k * n * n
    assert plan.retained_index_bytes == 8 * k * n + 8 * n
    assert plan.frame_workspace_bytes == 32 * b * n
    assert plan.polar_workspace_bytes == 144 * n * n
    assert plan.scalar_workspace_bytes == 16 * n + 24 * b
    assert plan.output_numerical_bytes == 16 * k * n * n + 8 * k * n + 8 * n
    assert plan.peak_owned_numerical_bytes == (plan.output_numerical_bytes + 32 * b * n
                                              + 144 * n * n + 16 * n + 24 * b)
    nself = math.prod(2 if axis % 2 == 0 else 1 for axis in mesh)
    assert plan.self_inverse_count == nself
    assert plan.maximum_work_units == (plan.reference_preflight_work_units + plan.anchor_work_units
                                       + ((k + nself) // 2 - 1) * plan.polar_work_units_per_point
                                       + k * plan.physical_work_units_per_point)


def test_seed_digest_ownership_non_aliasing_and_checked_getters():
    reference, _, c, _, _ = _reference((2, 2, 2))
    result = _make(reference)
    gauges = _gauges(result)
    domain = b"vibeqc.periodic.correlation.diabatic-seed.gauges"
    wire = struct.pack(">Q", len(domain)) + domain + struct.pack(">IQQ", 1, 8, 2)
    lanes = gauges.reshape(-1).view(np.float64).copy()
    lanes[lanes == 0.0] = 0.0
    wire += b"".join(struct.pack(">d", value) for value in lanes)
    assert result.gauge_payload_sha256 == hashlib.sha256(wire).hexdigest()
    assert _make(reference).seed_identity_sha256 == result.seed_identity_sha256
    assert result.diagnostics.required_node_memory_bytes >= (reference.state.resident_bytes
                                                             + result.memory.peak_owned_numerical_bytes)
    changed = _make(reference, controls=_options(reference, relative_tolerance=3e-11))
    assert changed.gauge_payload_sha256 == result.gauge_payload_sha256
    assert changed.seed_identity_sha256 != result.seed_identity_sha256
    copy = result.point_gauge_copy(0)
    copy[:] = 0
    np.testing.assert_array_equal(_gauges(result), gauges)
    del reference, changed, c
    gc.collect()
    np.testing.assert_array_equal(_gauges(result), gauges)
    assert result.state.n_basis == 3
    for getter, args in ((result.point_gauge_copy, (8,)), (result.gauge, (0, 2, 0)),
                         (result.active_band, (0, 2)), (result.gamma_pivot, (2,))):
        with pytest.raises(IndexError):
            getter(*args)


@pytest.mark.parametrize("field,bad", [("absolute_tolerance", np.nan),
                                       ("relative_tolerance", np.inf),
                                       ("unitarity_tolerance", 0.0),
                                       ("jacobi_relative_tolerance", 0.0),
                                       ("jacobi_max_sweeps", 0),
                                       ("maximum_work_units", 0)])
def test_invalid_explicit_controls_reject(field, bad):
    reference, *_ = _reference()
    controls = _options(reference)
    setattr(controls, field, bad)
    with pytest.raises(ValueError):
        _make(reference, controls=controls, cap=1_000_000)


def test_shift_dimension_overflow_and_tiny_diagnostic_caps_fail_closed():
    shifted, *_ = _reference((2, 1, 1), shift=(1, 0, 0))
    with pytest.raises(ValueError, match="Gamma-centered"):
        _make(shifted)
    for b, n, effective, sweeps in ((0, 1, 1, 64), (3, 0, 3, 64),
                                    (2, 3, 3, 64), (3, 2, 4, 64), (3, 2, 3, 0)):
        with pytest.raises(ValueError):
            core._plan_periodic_correlation_diabatic_seed((1, 1, 1), b, n, effective, sweeps)
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        core._plan_periodic_correlation_diabatic_seed((2, 2, 2), 2**32, 2**31, 2**32, 64)
    reference, *_ = _reference()
    with pytest.raises((ValueError, RuntimeError), match="limited"):
        _make(reference, controls=_options(reference, jacobi_max_sweeps=257))


def test_insufficient_jacobi_sweeps_are_not_reported_as_a_usable_seed():
    rng = np.random.default_rng(433)
    rotations = np.array([np.linalg.qr(rng.normal(size=(5, 5))
                                      + 1j * rng.normal(size=(5, 5)))[0] for _ in range(3)])
    reference, *_ = _reference(nactive=5, canonical_rotations=rotations)
    with pytest.raises(RuntimeError, match="Jacobi"):
        _make(reference, controls=_options(reference, jacobi_max_sweeps=1,
                                           jacobi_relative_tolerance=1e-16))
