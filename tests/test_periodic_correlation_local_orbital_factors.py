"""Unified local factors: tiny actual Gaussian stores and independent oracles.

Mean-field matrices are synthetic exact eigenproblems, not chemical HF
references. No real-factor certificate or CCSD energy is inferred here.
"""

from __future__ import annotations

import gc
import hashlib
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis, _oracle as _ao_pair_oracle
from tests.test_periodic_correlation_density_factors import (
    _occupied_columns, _oo, _virtual, _virtual_columns, _vv,
)
from tests.test_periodic_correlation_factor_store import _populate, _writer
from tests.test_periodic_correlation_local_factors import (
    _caps, _factor, _native_store, _orbital_objects, _selection,
)
from tests.test_periodic_correlation_pao_space import _geometry
from tests.test_periodic_correlation_reciprocal_metric import (
    _accepted_metric_records, _analytic_s_fourier, _raw_metric_config, _two_s_metric_basis,
)


pytestmark = pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="native store requires POSIX")


def _plan(b, occupied, selected, q, begin=0, count=None):
    count = b.schedule.shape.n_auxiliary - begin if count is None else count
    return core._plan_periodic_correlation_local_orbital_factor_panel(
        b.reference, b.schedule, b.reader, b.wannier, b.domain, b.space,
        occupied, selected, q, begin, count)


def _make(b, rows, selected, q, *, begin=0, count=None, caps=None, gauge=None):
    rows = np.asarray(rows, dtype=np.uint64)
    count = b.schedule.shape.n_auxiliary - begin if count is None else count
    return core._build_periodic_correlation_local_orbital_factor_panel(
        b.reference, b.schedule, b.reader, b.wannier, b.gauge if gauge is None else gauge,
        b.domain, b.space, rows, selected, q, begin, count,
        _caps(_plan(b, len(rows), selected, q, begin, count)) if caps is None else caps)


def _oracle_image_bound(b, lattice):
    # |A^-1 R|_infinity <= max_d ||(A^-1)_d|| * |R|. A retained
    # translation has |R| <= cutoff + maximum AO-origin separation. The
    # extra label is an independent test-oracle guard, not a source-cut change.
    origins = np.array([shell.origin for shell in b.ao.shells()])
    diameter = np.linalg.norm(origins[:, None, :] - origins[None, :, :], axis=2).max()
    bound = int(np.ceil(np.linalg.norm(np.linalg.inv(lattice), axis=1).max()
                        * (b.reader.image_cutoff_bohr + diameter))) + 1
    assert bound <= 8, "tiny Gaussian oracle image enumeration exceeds its explicit bound"
    return bound


def _oracle(b, rows, selected, q, *, image_bound=None):
    records = _accepted_metric_records(b.sources[q])
    vectors = np.array([r[1] for r in records])
    weights = np.array([r[3] for r in records])
    f = np.array([_analytic_s_fourier(b.auxiliary, r[1], r[2]) for r in records]).T
    metric = (f.conj() * weights) @ f.T
    eigenvalues, u = np.linalg.eigh(metric)
    keep = eigenvalues > b.config.metric_absolute_eigenvalue_threshold
    w = (u[:, keep] / np.sqrt(eigenvalues[keep])) @ u[:, keep].conj().T
    nk, n = b.schedule.shape.n_kpoints, len(rows) + selected.count
    density = np.zeros((len(records), n, n), complex)
    lattice = 2 * np.pi * np.linalg.inv(b.reciprocal).T
    image_bound = _oracle_image_bound(b, lattice) if image_bound is None else image_bound
    for k in range(nk):
        ket = b.schedule.descriptor(q * b.schedule.shape.tiles_per_q + k * b.schedule.shape.tiles_per_k_bra).k_ket_index
        left = np.column_stack((_occupied_columns(b, k, rows), _virtual_columns(b, k, selected)))
        right = np.column_stack((_occupied_columns(b, ket, rows), _virtual_columns(b, ket, selected)))
        rho, _ = _ao_pair_oracle(b.ao, vectors, lattice,
            np.asarray(b.reference.state.kpoint_cartesian(ket)), b.reader.image_cutoff_bohr, image_bound)
        for mu in range(b.ao.nbasis):
            for nu in range(b.ao.nbasis):
                density += rho[mu, nu, :, None, None] * left[mu, None, :, None].conj() * right[nu, None, None, :] / nk
    return (w @ ((f.conj() * weights) @ density.reshape(len(records), -1)) / np.sqrt(nk)).reshape(b.auxiliary.nbasis, n, n)


def _compare_quadrants(b, rows, selected, q, values):
    o = len(rows)
    np.testing.assert_array_equal(values[:, :o, :o], _oo(b, rows, rows, q).tensor_copy())
    np.testing.assert_array_equal(values[:, o:, o:], _vv(b, selected, selected, q).tensor_copy())
    for i, (occupied, cell) in enumerate(rows):
        s = _selection(occupied_index=occupied, occupied_cell=cell, virtual_begin=selected.begin,
                       virtual_count=selected.count, virtual_translation_cell=selected.translation_cell)
        np.testing.assert_array_equal(values[:, i, o:], _factor(b, q, selection=s).matrix_copy())
        np.testing.assert_array_equal(values[:, o:, i], _factor(b, q, selection=s, vo=True).matrix_copy())


@pytest.mark.parametrize("mesh,q", [((1, 1, 1), 0), ((2, 1, 1), 1), ((3, 1, 1), 2), ((2, 2, 1), 3)])
def test_all_ordered_quadrants_and_actual_gaussian_oracle(tmp_path, mesh, q):
    nk = int(np.prod(mesh))
    gauge = np.exp(1j * np.linspace(0.17, 1.03, nk)).reshape(nk, 1, 1)
    b = _native_store(tmp_path, mesh=mesh, full_domain=True, cutoff=6.5, gauge=gauge)
    rows = [[0, nk - 1], [0, 0]] if nk > 1 else [[0, 0]]
    selected = _virtual(nk - min(2, nk), min(2, nk), nk // 2)
    result = _make(b, rows, selected, q)
    values = result.tensor_copy()
    _compare_quadrants(b, rows, selected, q, values)
    np.testing.assert_allclose(values, _oracle(b, rows, selected, q), atol=8e-12, rtol=6e-11)
    assert result.memory.tile_visits == b.schedule.shape.tiles_per_q
    assert result.ao_basis_identity_sha256 == b.reader.ao_basis_identity_sha256
    assert result.auxiliary_basis_identity_sha256 == b.reader.auxiliary_basis_identity_sha256
    assert result.source_identity_sha256 == b.sources[q].source_identity_sha256
    assert result.finite_image_reference and not result.ao_image_source_certified
    assert not result.density_symmetry_certified


def test_complex_ordered_entries_are_not_made_symmetric_or_real(tmp_path):
    gauge = np.exp(1j * np.array([0.2, 0.4, 1.3])).reshape(3, 1, 1)
    b = _native_store(tmp_path, mesh=(3, 1, 1), gauge=gauge, cutoff=6.5)
    values = _make(b, [[0, 0], [0, 2]], _virtual(translation=1), 1).tensor_copy()
    assert np.max(abs(values.imag)) > 1e-5
    assert np.linalg.norm(values - values.transpose(0, 2, 1)) > 1e-4
    assert np.linalg.norm(values - values.transpose(0, 2, 1).conj()) > 1e-4


def test_auxiliary_slicing_label_permutation_and_common_translation(tmp_path):
    b = _native_store(tmp_path, full_domain=True, cutoff=6.5)
    rows, selected = [[0, 0], [0, 1]], _virtual(0, 2)
    full = _make(b, rows, selected, 1)
    values = full.tensor_copy()
    slices = np.concatenate([_make(b, rows, selected, 1, begin=p, count=1).tensor_copy() for p in range(2)])
    np.testing.assert_array_equal(slices, values)
    reverse = _make(b, rows[::-1], selected, 1).tensor_copy()
    order = [1, 0, 2, 3]
    np.testing.assert_array_equal(reverse, values[:, order, :][:, :, order])
    translated = _make(b, rows[::-1], _virtual(0, 2, 1), 1).tensor_copy()
    np.testing.assert_allclose(translated, -values, atol=4e-13)
    other_q = _make(b, rows, selected, 0)
    assert other_q.local_basis_identity_sha256 == full.local_basis_identity_sha256
    assert other_q.source_identity_sha256 != full.source_identity_sha256
    assert other_q.identity_sha256 != full.identity_sha256


def _masked_store(tmp_path, reduced):
    reference, geometry = _geometry((2, 1, 1), reduced=reduced)
    auxiliary = _two_s_metric_basis()
    ao = _basis([(0, (0.1 * i, -0.06 * i, 0.03 * i), [0.55 + 0.1 * i], [0.7], True) for i in range(4)])
    dims = reference.dimensions
    dims.external_bytes -= reference.state_resident_bytes
    dims.n_auxiliary = auxiliary.nbasis
    dims.factor_ao_pair_block, dims.factor_auxiliary_block = 5, 1
    reference = core._make_periodic_correlation_admitted_reference(reference.state, dims, reference.budget)
    schedule = core._make_periodic_correlation_factor_stream_schedule(reference)
    config = _raw_metric_config(auxiliary, cutoff=2.0, reciprocal_block=2)
    config.ao_pair_block, config.auxiliary_block = 5, 1
    config.ao_basis_identity_sha256 = core._auxiliary_basis_content_identity_sha256(ao)
    config.backend_identity_sha256 = core._periodic_correlation_metric_factorization_backend_identity_sha256()
    config.publisher_mode = core._PeriodicCorrelationFactorPublisherMode.CANONICAL_SEQUENTIAL_EXACTLY_ONCE
    config.backing_mode = core._PeriodicCorrelationFactorBackingMode.DISK
    config.publisher_buffer_count, config.transpose_before_publish = 0, False
    config.codec_identity_sha256 = core._periodic_correlation_private_factor_store_codec_identity_sha256()
    config.codec = core._periodic_correlation_private_factor_store_codec_inventory()
    backend = config.backend
    backend.exact_extra_retained_bytes = 16 * auxiliary.nbasis * 5 + core._periodic_correlation_private_factor_store_receiver_bytes(schedule)
    backend.exact_extra_control_bytes = core._periodic_correlation_private_factor_store_fixed_control_bytes()
    backend.per_thread_fourier_transform_fixed_workspace_bytes = core._PERIODIC_AOPAIR_FOURIER_FIXED_NUMERIC_WORKSPACE_BYTES
    config.backend = backend
    sources = [core._make_periodic_correlation_reciprocal_metric_source_manifest(reference, schedule, config, auxiliary, q, 65536)
               for q in range(2)]
    census = core._make_periodic_correlation_factor_build_census(reference, schedule, config,
        [source.factor_build_q_record() for source in sources])
    b = SimpleNamespace(reference=reference, schedule=schedule, config=config, sources=sources, census=census,
                        ao=ao, auxiliary=auxiliary, reciprocal=np.diag([2.0, 3.0, 5.0]))
    b.store_caps = core._PeriodicCorrelationPrivateFactorStoreCaps()
    b.store_caps.maximum_file_bytes, b.store_caps.maximum_tile_bytes, b.store_caps.maximum_tile_count = 1048576, 8192, 128
    b.stream_caps = core._PeriodicCorrelationThreeCenterStreamCaps()
    for name, value in {"maximum_tile_count": 128, "maximum_logical_bytes": 65536, "maximum_tile_bytes": 8192,
                        "maximum_reciprocal_candidates_per_q": 65536, "maximum_image_candidates_per_tile": 20000,
                        "receiver_retained_numeric_bytes": core._periodic_correlation_private_factor_store_receiver_bytes(schedule)}.items():
        setattr(b.stream_caps, name, value)
    writer = _writer(b, tmp_path, cutoff=6.5)
    b.reader = writer.finish(_populate(b, writer, cutoff=6.5))
    objects = _orbital_objects(reference, domain_columns=[geometry.column(i) for i in range(geometry.domain_dimension)],
                               gauge=np.exp(1j * np.array([0.14, 0.39])).reshape(2, 1, 1))
    b.domain, b.space, b.wannier, b.gauge = objects.domain, objects.space, objects.wannier, objects.gauge
    return b


@pytest.mark.parametrize("reduced", [False, True])
def test_frozen_core_and_reduced_virtual_masks_reach_the_actual_gaussian_store(tmp_path, reduced):
    b = _masked_store(tmp_path, reduced)
    rows, selected = [[0, 1], [0, 0]], _virtual(0, min(2, b.space.retained_dimension), 1)
    values = _make(b, rows, selected, 1).tensor_copy()
    _compare_quadrants(b, rows, selected, 1, values)
    expected = _oracle(b, rows, selected, 1)
    np.testing.assert_allclose(values, expected, atol=9e-12, rtol=6e-11)
    if not reduced:
        # c_z=2*pi/5: fifth-cell images at 6.283... bohr are inside the
        # unchanged 6.5-bohr cutoff. The old +/-4 oracle silently omitted them.
        incomplete = _oracle(b, rows, selected, 1, image_bound=4)
        assert np.max(abs(expected - incomplete)) > 1e-5
        six = _oracle(b, rows, selected, 1, image_bound=6)
        np.testing.assert_array_equal(expected, six)
        np.testing.assert_array_equal(six, _oracle(b, rows, selected, 1, image_bound=8))
    state = b.reference.state
    assert state.n_frozen_core == 1
    assert state.n_effective_orbitals == (3 if reduced else 4)
    for k in range(2):
        all_occupied = np.asarray(state.frozen_core_mask(k), bool) | np.asarray(state.correlated_occupied_mask(k), bool)
        virtual = _virtual_columns(b, k, selected)
        np.testing.assert_allclose(state.coefficients(k)[:, all_occupied].conj().T @ state.overlap(k) @ virtual,
                                   0, atol=2e-12)


def test_exact_owned_borrowed_inventory_and_single_pass_work(tmp_path):
    b = _native_store(tmp_path, full_domain=True)
    p = _plan(b, 2, _virtual(0, 2), 1)
    n, nao, ab, o = 4, b.ao.nbasis, 2, 2
    assert p.retained_output_bytes == p.compensation_bytes == 16 * ab * n**2
    assert p.coefficient_panel_bytes == 32 * nao * n
    assert p.coefficient_scratch_bytes == 32 * nao
    assert p.retained_index_bytes == p.caller_index_bytes == 16 * o
    assert p.peak_owned_numerical_bytes == 32 * ab * n**2 + 32 * nao * n + 32 * nao + 16 * o
    assert p.tile_visits == b.schedule.shape.tiles_per_q
    assert p.reader_payload_bytes == b.schedule.shape.logical_bytes // b.schedule.shape.n_kpoints
    assert p.live_reader_numeric_bytes == 65536 + p.maximum_reader_tile_bytes
    assert p.live_domain_bytes == b.domain.memory.retained_domain_index_bytes + b.domain.memory.retained_matrix_bytes
    assert p.live_space_bytes == b.space.memory.output_numerical_bytes
    assert p.live_wannier_bytes == b.wannier.memory.retained_coefficient_bytes
    assert p.caller_gauge_bytes == b.gauge.nbytes
    d, neff, nk = b.domain.domain_dimension, b.reference.state.n_effective_orbitals, b.reference.state.n_kpoints
    vc = nao*d + nao**2 + 2*nao*neff + 4*nao
    cc = 2*nk*(2*vc + o*nao*(neff+1))
    assert p.work_units == nk*nao**2*ab*n**2 + 2*p.reader_payload_bytes//16 + 256*p.tile_visits + ab*n**2 + cc + b.gauge.size + 1 + 3*o + d
    dimensions, budget = b.reference.dimensions, b.reference.budget
    live = sum(getattr(p, name) for name in ("peak_owned_numerical_bytes", "caller_index_bytes", "caller_gauge_bytes",
        "live_wannier_bytes", "live_domain_bytes", "live_space_bytes", "live_reader_numeric_bytes", "live_reader_control_bytes"))
    assert p.required_node_memory_bytes == (dimensions.external_bytes + dimensions.shared_bytes
        + budget.mpi_ranks*(dimensions.per_rank_bytes + dimensions.localization_window_bytes_per_rank)
        + budget.mpi_ranks*budget.workers_per_rank*live)


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_work_units", "maximum_tile_visits", "maximum_reader_tile_bytes"])
@pytest.mark.parametrize("zero", [False, True])
def test_caps_precede_label_and_gauge_scans(tmp_path, field, zero):
    b = _native_store(tmp_path)
    caps = _caps(_plan(b, 1, _virtual(), 0))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    with pytest.raises(ValueError, match="cap"):
        _make(b, [[2**64 - 1, 0]], _virtual(), 0, caps=caps, gauge=np.full_like(b.gauge, np.nan))


def test_invalid_labels_slices_and_provenance_are_rejected(tmp_path):
    b = _native_store(tmp_path)
    for rows in ([[1, 0]], [[0, 2]], [[0, 2**64 - 1]]):
        with pytest.raises(IndexError, match="occupied index|modular"):
            _make(b, rows, _virtual(), 0)
    with pytest.raises(ValueError, match="duplicate"):
        _make(b, [[0, 0], [0, 0]], _virtual(), 0)
    for count in (0, 3, 2**64 - 1):
        with pytest.raises(IndexError, match="occupied count"):
            _plan(b, count, _virtual(), 0)
    for selected in (_virtual(count=0), _virtual(begin=1), _virtual(translation=2), _virtual(count=2**64 - 1)):
        with pytest.raises(IndexError, match="virtual slice|modular"):
            _plan(b, 1, selected, 0)
    with pytest.raises(ValueError, match="sealed Wannier gauge"):
        _make(b, [[0, 0]], _virtual(), 0, gauge=-b.gauge)
    other = _orbital_objects(b.reference, domain_columns=[[1, 1]])
    b.space = other.space
    with pytest.raises(ValueError, match="provenance"):
        _plan(b, 1, _virtual(), 0)


def test_live_node_budget_is_separate_from_worker_cap(tmp_path):
    b = _native_store(tmp_path)
    plan = _plan(b, 1, _virtual(), 0)
    dims, budget = b.reference.dimensions, b.reference.budget
    dims.external_bytes -= b.reference.state_resident_bytes
    budget.memory_limit_bytes = plan.required_node_memory_bytes - 1
    b.reference = core._make_periodic_correlation_admitted_reference(b.reference.state, dims, budget)
    with pytest.raises(ValueError, match="node memory"):
        _make(b, [[0, 0]], _virtual(), 0, caps=_caps(plan))


def test_foreign_equal_content_state_owner_is_not_accepted(tmp_path):
    b = _native_store(tmp_path)
    foreign_directory = tmp_path / "foreign"
    foreign_directory.mkdir(mode=0o700)
    other = _native_store(foreign_directory)
    assert b.reference.state.state_identity_sha256 == other.reference.state.state_identity_sha256
    b.reader = other.reader
    with pytest.raises(ValueError, match="provenance"):
        _plan(b, 1, _virtual(), 0)


def test_input_views_are_nonconverting_and_alignment_checked(tmp_path):
    b = _native_store(tmp_path)
    caps = _caps(_plan(b, 1, _virtual(), 0))
    def direct(rows, gauge):
        return core._build_periodic_correlation_local_orbital_factor_panel(
            b.reference, b.schedule, b.reader, b.wannier, gauge, b.domain, b.space,
            rows, _virtual(), 0, 0, 2, caps)
    with pytest.raises(ValueError, match="uint64"):
        direct(np.array([[0, 0]], np.int64), b.gauge)
    with pytest.raises(ValueError, match="contiguous"):
        direct(np.zeros((1, 4), np.uint64)[:, ::2], b.gauge)
    with pytest.raises(ValueError, match="complex128"):
        direct(np.array([[0, 0]], np.uint64), b.gauge.real.copy())
    unaligned = np.ndarray((1, 2), dtype=np.uint64, buffer=bytearray(17), offset=1)
    unaligned[:] = 0
    with pytest.raises(ValueError, match="aligned"):
        direct(unaligned, b.gauge)


def test_selection_payload_wire_and_detached_immutable_result(tmp_path):
    b = _native_store(tmp_path, full_domain=True)
    rows, selected = np.array([[0, 1], [0, 0]], np.uint64), _virtual(1, 1, 1)
    result = _make(b, rows, selected, 1)
    values = result.tensor_copy()
    def prefix(name):
        encoded = ("vibeqc.periodic.correlation.local-orbital-factors." + name).encode()
        return struct.pack(">Q", len(encoded)) + encoded + struct.pack(">I", 1)
    selection = prefix("selection") + struct.pack(">Q", 2)
    for row in rows:
        selection += struct.pack(">QQ", *row)
    selection += struct.pack(">QQQ", 1, 1, 1)
    assert hashlib.sha256(selection).hexdigest() == result.selection_identity_sha256
    basis = prefix("basis")
    for value in (b.reference.state.state_identity_sha256, b.reference.state.calculation_identity,
                  b.reference.dimensions.allocation_identity, result.selection_identity_sha256,
                  b.wannier.wannier_identity_sha256, b.wannier.gauge_payload_sha256,
                  b.domain.pao_domain_identity_sha256, b.space.pao_space_identity_sha256):
        encoded = value.encode()
        basis += struct.pack(">Q", len(encoded)) + encoded
    assert hashlib.sha256(basis).hexdigest() == result.local_basis_identity_sha256
    payload = prefix("payload") + struct.pack(">QQQQ", 1, 0, 2, 3)
    for z in values.ravel():
        payload += struct.pack(">dd", 0.0 if z.real == 0 else z.real, 0.0 if z.imag == 0 else z.imag)
    assert hashlib.sha256(payload).hexdigest() == result.payload_sha256
    result.tensor_copy()[:] = 47
    rows[:] = 0
    selected.begin = 0
    del b
    gc.collect()
    np.testing.assert_array_equal(result.tensor_copy(), values)
    assert result.occupied(0) == (0, 1)
    assert result.virtual_selection.begin == 1
    assert result.element(0, 1, 2) == values[0, 1, 2]
    with pytest.raises(IndexError):
        result.occupied(2)
