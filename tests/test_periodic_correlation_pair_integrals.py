"""All-q native pair assembly from tiny actual Gaussian factor stores."""

from __future__ import annotations

import gc
import hashlib
import struct
import sys

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis
from tests.test_periodic_correlation_local_factors import (
    _factor, _native_store, _reciprocal_oracle, _selection as _local_selection,
)

pytestmark = pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="private POSIX factor store")


def _selection(b, **changes):
    s = core._PeriodicCorrelationPairIntegralSelection()
    s.virtual_count = b.space.retained_dimension
    s.virtual_block = 1
    for name, value in changes.items():
        setattr(s, name, value)
    return s


def _plan(b, s):
    return core._plan_periodic_correlation_pair_integral_block(
        b.reference, b.schedule, b.reader, b.wannier, b.domain, b.space, s)


def _caps(p):
    c = core._PeriodicCorrelationPairIntegralCaps()
    c.maximum_owned_numerical_bytes = p.peak_owned_numerical_bytes
    c.maximum_work_units = p.work_units_upper_bound
    c.maximum_factor_builds = p.factor_builds
    c.maximum_tile_visits = p.tile_visits_upper_bound
    return c


def _build(b, s=None, caps=None, gauge=None):
    s = _selection(b) if s is None else s
    return core._build_periodic_correlation_pair_integral_block(
        b.reference, b.schedule, b.reader, b.wannier,
        b.gauge if gauge is None else gauge, b.domain, b.space, s,
        _caps(_plan(b, s)) if caps is None else caps)


def _local(s, right=False):
    return _local_selection(
        occupied_index=s.occupied_j if right else s.occupied_i,
        occupied_cell=s.cell_j if right else s.cell_i,
        virtual_begin=s.virtual_begin, virtual_count=s.virtual_count,
        virtual_translation_cell=s.virtual_translation_cell)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1)])
def test_pair_block_matches_independent_reciprocal_kernel(tmp_path, mesh):
    nk = int(np.prod(mesh))
    gauge = np.exp(1j * np.linspace(0.19, 0.73, nk)).reshape(nk, 1, 1)
    b = _native_store(tmp_path, mesh=mesh, gauge=gauge, full_domain=True)
    s = _selection(b, cell_j=nk - 1, virtual_translation_cell=nk // 2)
    result = _build(b, s)
    expected = np.zeros((s.virtual_count, s.virtual_count), complex)
    for q in range(nk):
        _, reversed_density, kernel, _ = _reciprocal_oracle(b, q, _local(s), vo=True)
        _, right_density, _, _ = _reciprocal_oracle(b, q, _local(s, True))
        expected += reversed_density.conj().T @ kernel @ right_density / nk
    np.testing.assert_allclose(result.matrix_copy(), expected, atol=3e-12, rtol=5e-11)
    assert result.finite_image_reference and not result.real_orbital_integrals_certified
    assert result.store_identity_sha256 == b.reader.storage_identity_sha256
    assert result.factor_builds == result.memory.factor_builds
    assert result.tile_visits <= result.memory.tile_visits_upper_bound


def test_complex_pair_is_not_same_orientation_gram(tmp_path):
    b = _native_store(tmp_path, gauge=np.exp(1j * np.array([0.31, 0.67])).reshape(2, 1, 1))
    result = _build(b)
    expected = np.zeros((1, 1), complex)
    wrong = np.zeros_like(expected)
    for q in range(2):
        ov, vo = _factor(b, q).matrix_copy(), _factor(b, q, vo=True).matrix_copy()
        expected += vo.conj().T @ ov
        wrong += ov.conj().T @ ov
    np.testing.assert_allclose(result.matrix_copy(), expected, atol=1e-13)
    assert np.max(abs(expected - wrong)) > 1e-6
    assert np.max(abs(result.matrix_copy().imag)) > 1e-6


@pytest.mark.parametrize("block", [1, 2, 3, 8])
def test_virtual_block_partition_and_partial_blocks_preserve_pair(tmp_path, block):
    b = _native_store(tmp_path, mesh=(3, 1, 1), full_domain=True)
    full = _build(b, _selection(b, virtual_block=3)).matrix_copy()
    s = _selection(b, virtual_block=block)
    actual = _build(b, s)
    np.testing.assert_allclose(actual.matrix_copy(), full, atol=1e-13)
    nb = (s.virtual_count + block - 1) // block
    assert actual.factor_builds == 3 * b.schedule.shape.auxiliary_tile_count * nb * (1 + nb)


def test_ragged_auxiliary_tail_and_nonzero_virtual_slice_match_reciprocal_kernel(tmp_path):
    auxiliary = _basis([
        (0, (0.0, 0.0, 0.0), [0.6], [1.0], True),
        (0, (0.3, -0.2, 0.1), [0.9], [1.0], True),
        (0, (-0.1, 0.4, 0.2), [1.3], [1.0], True),
    ])
    b = _native_store(tmp_path, mesh=(3, 1, 1), full_domain=True,
                      auxiliary=auxiliary, auxiliary_block=2)
    assert b.schedule.shape.n_auxiliary == 3
    assert b.schedule.shape.auxiliary_tile_count == 2
    selection = _selection(b, cell_i=1, cell_j=2, virtual_begin=1,
                            virtual_count=2, virtual_block=1, virtual_translation_cell=1)
    result = _build(b, selection)
    expected = np.zeros((2, 2), complex)
    for q in range(3):
        _, left, kernel, _ = _reciprocal_oracle(b, q, _local(selection), vo=True)
        _, right, _, _ = _reciprocal_oracle(b, q, _local(selection, True))
        expected += left.conj().T @ kernel @ right / 3
    np.testing.assert_allclose(result.matrix_copy(), expected, atol=3e-12, rtol=5e-11)
    full = _build(b, _selection(b, cell_i=1, cell_j=2, virtual_block=3,
                                 virtual_translation_cell=1)).matrix_copy()
    np.testing.assert_allclose(result.matrix_copy(), full[1:3, 1:3], atol=3e-13)


def test_pair_planner_charges_outer_live_payload_above_inner_node_peak(tmp_path):
    b = _native_store(tmp_path, full_domain=True)
    p = _plan(b, _selection(b))
    assert p.retained_output_bytes == p.compensation_bytes == 16 * p.n_virtual**2
    extra = p.retained_output_bytes + p.compensation_bytes + p.retained_left_factor_bytes
    assert p.peak_owned_numerical_bytes == extra + p.maximum_factor.peak_owned_numerical_bytes
    assert p.required_node_memory_bytes == p.maximum_factor.required_node_memory_bytes + extra
    assert p.tile_visits_upper_bound == p.factor_builds * p.maximum_factor.tile_visits


@pytest.mark.parametrize("ranks,workers", [(1, 1), (1, 3)])
def test_pair_node_budget_charges_outer_payload_for_every_replica(tmp_path, ranks, workers):
    b = _native_store(tmp_path, full_domain=True)
    dimensions, budget = b.reference.dimensions, b.reference.budget
    dimensions.external_bytes -= b.reference.state_resident_bytes
    budget.mpi_ranks, budget.workers_per_rank = ranks, workers
    b.reference = core._make_periodic_correlation_admitted_reference(
        b.reference.state, dimensions, budget)
    selection = _selection(b)
    plan = _plan(b, selection)
    extra = plan.retained_output_bytes + plan.compensation_bytes + plan.retained_left_factor_bytes
    assert plan.required_node_memory_bytes == plan.maximum_factor.required_node_memory_bytes + ranks * workers * extra
    dimensions, budget = b.reference.dimensions, b.reference.budget
    dimensions.external_bytes -= b.reference.state_resident_bytes
    budget.memory_limit_bytes = plan.required_node_memory_bytes - 1
    b.reference = core._make_periodic_correlation_admitted_reference(
        b.reference.state, dimensions, budget)
    with pytest.raises((ValueError, RuntimeError), match="node memory"):
        _build(b, selection, _caps(plan))
    budget.memory_limit_bytes += 1
    b.reference = core._make_periodic_correlation_admitted_reference(
        b.reference.state, dimensions, budget)
    assert _build(b, selection, _caps(plan)).memory.required_node_memory_bytes == budget.memory_limit_bytes


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_work_units",
                                    "maximum_factor_builds", "maximum_tile_visits"])
@pytest.mark.parametrize("zero", [False, True])
def test_caps_precede_gauge_scan_and_factor_reads(tmp_path, field, zero):
    b = _native_store(tmp_path)
    s = _selection(b)
    caps = _caps(_plan(b, s))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    wrong = b.gauge.copy()
    wrong[0, 0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _build(b, s, caps, wrong)


@pytest.mark.parametrize("field,value", [("occupied_i", 1), ("occupied_j", 1),
    ("cell_i", 2), ("cell_j", 2), ("virtual_count", 0), ("virtual_begin", 100),
    ("virtual_block", 0), ("virtual_translation_cell", 2)])
def test_invalid_selection_fails_before_work(tmp_path, field, value):
    b = _native_store(tmp_path)
    with pytest.raises((ValueError, IndexError), match="selection|range|block"):
        _plan(b, _selection(b, **{field: value}))


def test_native_owner_outlives_inputs_and_diagnostic_copy_is_independent(tmp_path):
    b = _native_store(tmp_path)
    result = _build(b)
    matrix = result.matrix_copy()
    copy = matrix.copy()
    del b
    gc.collect()
    np.testing.assert_array_equal(result.matrix_copy(), copy)
    matrix[:] = 99
    np.testing.assert_array_equal(result.matrix_copy(), copy)
    domain = b"vibeqc.periodic.correlation.pair-integrals.payload"
    wire = struct.pack(">Q", len(domain)) + domain + struct.pack(">IQ", 1, len(copy))
    for value in copy.flat:
        wire += struct.pack(">dd", 0.0 if value.real == 0 else value.real,
                            0.0 if value.imag == 0 else value.imag)
    assert result.payload_sha256 == hashlib.sha256(wire).hexdigest()
