"""Tiny OO/VV verified-store consumers with independent reciprocal RI kernels.

No SCF, target chemistry, full-system tensor or caller-labelled factor input.
Diagonal densities retain the native omitted-zero-mode Coulomb Hamiltonian.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _oracle as _pair_oracle
from tests.test_periodic_correlation_local_factors import _caps, _native_store, _orbital_objects
from tests.test_periodic_correlation_reciprocal_metric import _accepted_metric_records, _analytic_s_fourier


pytestmark = pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="private store requires POSIX")


def _virtual(begin=0, count=1, translation=0):
    value = core._PeriodicCorrelationVirtualBlockSelection()
    value.begin, value.count, value.translation_cell = begin, count, translation
    return value


def _oo(b, left, right, q, *, begin=0, count=None, caps=None, gauge=None):
    left, right = np.asarray(left, dtype=np.uint64), np.asarray(right, dtype=np.uint64)
    count = b.schedule.shape.n_auxiliary - begin if count is None else count
    plan = core._plan_periodic_correlation_occupied_density_factor_block(
        b.reference, b.schedule, b.reader, b.wannier, len(left), len(right), q, begin, count)
    return core._build_periodic_correlation_occupied_density_factor_block(
        b.reference, b.schedule, b.reader, b.wannier, b.gauge if gauge is None else gauge,
        left, right, q, begin, count, _caps(plan) if caps is None else caps)


def _vv(b, left, right, q, *, begin=0, count=None, caps=None):
    count = b.schedule.shape.n_auxiliary - begin if count is None else count
    plan = core._plan_periodic_correlation_virtual_density_factor_block(
        b.reference, b.schedule, b.reader, b.domain, b.space, left, right, q, begin, count)
    return core._build_periodic_correlation_virtual_density_factor_block(
        b.reference, b.schedule, b.reader, b.domain, b.space, left, right, q, begin, count,
        _caps(plan) if caps is None else caps)


def _occupied_columns(b, k, rows):
    state = b.reference.state
    cells = np.array(list(product(*(range(n) for n in state.mesh))))
    fraction = cells[k] / np.array(state.mesh)
    active = np.asarray(state.correlated_occupied_mask(k), dtype=bool)
    coefficients = state.coefficients(k)[:, active] @ b.gauge[k]
    return np.column_stack([coefficients[:, index] * np.exp(-2j * np.pi * fraction @ cells[cell])
                            for index, cell in rows])


def _virtual_columns(b, k, selected):
    state = b.reference.state
    cells = np.array(list(product(*(range(n) for n in state.mesh))))
    fraction = cells[k] / np.array(state.mesh)
    c = state.coefficients(k)[:, np.asarray(state.virtual_mask(k), dtype=bool)]
    projector = c @ c.conj().T @ state.overlap(k)
    columns = []
    for d in range(b.domain.domain_dimension):
        cell, ao = b.domain.column(d)
        phase = np.exp(-2j * np.pi * fraction @ (cells[cell] + cells[selected.translation_cell]))
        columns.append(phase * projector[:, ao])
    x = b.space.coefficients_copy()[:, selected.begin:selected.begin + selected.count]
    return np.column_stack(columns) @ x


def _oracle(b, q, left, right, *, virtual=False):
    records = _accepted_metric_records(b.sources[q])
    vectors = np.array([r[1] for r in records])
    weights = np.array([r[3] for r in records])
    f = np.array([_analytic_s_fourier(b.auxiliary, r[1], r[2]) for r in records]).T
    metric = (f.conj() * weights) @ f.T
    values, u = np.linalg.eigh(metric)
    retained = values > b.config.metric_absolute_eigenvalue_threshold
    w = (u[:, retained] / np.sqrt(values[retained])) @ u[:, retained].conj().T
    inverse = (u[:, retained] / values[retained]) @ u[:, retained].conj().T
    kernel = (weights[:, None] * f.T) @ inverse @ (f.conj() * weights)
    lcount, rcount = (left.count, right.count) if virtual else (len(left), len(right))
    nk = b.schedule.shape.n_kpoints
    density = np.zeros((len(records), lcount, rcount), dtype=complex)
    lattice = 2 * np.pi * np.linalg.inv(b.reciprocal).T
    for k in range(nk):
        ket = b.schedule.descriptor(q * b.schedule.shape.tiles_per_q + k * b.schedule.shape.tiles_per_k_bra).k_ket_index
        cl = _virtual_columns(b, k, left) if virtual else _occupied_columns(b, k, left)
        cr = _virtual_columns(b, ket, right) if virtual else _occupied_columns(b, ket, right)
        rho, _ = _pair_oracle(b.ao, vectors, lattice, np.asarray(b.reference.state.kpoint_cartesian(ket)),
                              b.reader.image_cutoff_bohr, 4)
        for mu in range(b.ao.nbasis):
            for nu in range(b.ao.nbasis):
                density += rho[mu, nu, :, None, None] * cl[mu, None, :, None].conj() * cr[nu, None, None, :] / nk
    flattened = density.reshape(len(records), -1)
    expected = (w @ ((f.conj() * weights) @ flattened) / np.sqrt(nk)).reshape(b.auxiliary.nbasis, lcount, rcount)
    return expected, flattened, kernel


@pytest.mark.parametrize("mesh,q", [((1, 1, 1), 0), ((2, 1, 1), 1), ((3, 1, 1), 2)])
def test_oo_selected_translations_match_independent_gaussian_reciprocal_oracle(tmp_path, mesh, q):
    nk = int(np.prod(mesh))
    gauge = np.exp(1j * np.linspace(0.13, 0.74, nk)).reshape(nk, 1, 1)
    b = _native_store(tmp_path, mesh=mesh, gauge=gauge, cutoff=6.5)
    left = [[0, cell] for cell in range(nk)]
    right = [[0, nk - 1]]
    result = _oo(b, left, right, q)
    expected, _, _ = _oracle(b, q, left, right)
    np.testing.assert_allclose(result.tensor_copy(), expected, atol=5e-12, rtol=5e-11)
    assert result.left_occupied(0) == tuple(left[0])
    assert result.right_occupied(0) == tuple(right[0])
    assert result.source_identity_sha256 == b.sources[q].source_identity_sha256
    assert result.store_identity_sha256 == b.reader.storage_identity_sha256
    assert result.finite_image_reference and not result.ao_image_source_certified


@pytest.mark.parametrize("q", [0, 1, 2])
def test_vv_uses_separate_bra_ket_columns_of_one_sealed_space(tmp_path, q):
    b = _native_store(tmp_path, mesh=(3, 1, 1), full_domain=True, cutoff=6.5)
    left, right = _virtual(0, 2, 1), _virtual(1, 2, 2)
    result = _vv(b, left, right, q)
    expected, _, _ = _oracle(b, q, left, right, virtual=True)
    np.testing.assert_allclose(result.tensor_copy(), expected, atol=7e-12, rtol=5e-11)
    assert result.left_virtual().translation_cell == 1
    assert result.right_virtual().begin == 1
    assert result.memory.caller_gauge_bytes == result.memory.live_wannier_bytes == 0
    assert result.memory.live_domain_bytes > 0 and result.memory.live_space_bytes > 0


def test_general_complex_oo_vv_coulomb_uses_reversed_first_density(tmp_path):
    gauge = np.exp(1j * np.array([0.1, 0.4, 1.1])).reshape(3, 1, 1)
    b = _native_store(tmp_path, mesh=(3, 1, 1), gauge=gauge, cutoff=6.5)
    i, j, a = [[0, 0]], [[0, 1]], _virtual()
    actual = expected = naive = 0j
    for q in range(3):
        oo_reverse = _oo(b, j, i, q).tensor_copy().reshape(-1)
        oo_forward = _oo(b, i, j, q).tensor_copy().reshape(-1)
        vv = _vv(b, a, a, q).tensor_copy().reshape(-1)
        _, density_reverse, kernel = _oracle(b, q, j, i)
        _, density_vv, _ = _oracle(b, q, a, a, virtual=True)
        actual += np.vdot(oo_reverse, vv)
        naive += np.vdot(oo_forward, vv)
        expected += (density_reverse.conj().T @ kernel @ density_vv)[0, 0] / 3
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=5e-11)
    assert abs(actual.imag) > 1e-8
    assert abs(actual - naive) > 1e-8


def test_charged_diagonal_oo_vv_keep_omitted_zero_mode_without_chargeless_projection(tmp_path):
    b = _native_store(tmp_path)
    assert b.sources[0].zero_mode_excluded_count == 1
    for virtual in (False, True):
        selected = _virtual() if virtual else [[0, 0]]
        result = (_vv(b, selected, selected, 0) if virtual else _oo(b, selected, selected, 0)).tensor_copy()
        expected, _, _ = _oracle(b, 0, selected, selected, virtual=virtual)
        np.testing.assert_allclose(result, expected, atol=5e-12, rtol=5e-11)
        assert np.linalg.norm(result) > 1e-4


def test_oo_ordering_subsets_and_common_translation_are_exact(tmp_path):
    b = _native_store(tmp_path, cutoff=6.5)
    rows = [[0, 0], [0, 1]]
    full = _oo(b, rows, rows, 1).tensor_copy()
    permuted = _oo(b, rows[::-1], rows, 1)
    np.testing.assert_array_equal(permuted.tensor_copy(), full[:, ::-1, :])
    small = _oo(b, [rows[1]], [rows[0]], 1)
    np.testing.assert_array_equal(small.tensor_copy(), full[:, 1:2, 0:1])
    auxiliary = np.concatenate([_oo(b, rows, rows, 1, begin=p, count=1).tensor_copy() for p in range(2)])
    np.testing.assert_array_equal(auxiliary, full)
    translated = _oo(b, rows[::-1], rows[::-1], 1).tensor_copy()
    np.testing.assert_allclose(translated, -full, atol=5e-13)


def test_vv_slices_memory_and_translation_need_no_wannier_input(tmp_path):
    b = _native_store(tmp_path, full_domain=True)
    full = _vv(b, _virtual(0, 2), _virtual(0, 2), 1)
    sliced = np.concatenate([_vv(b, _virtual(i, 1), _virtual(0, 2), 1).tensor_copy() for i in range(2)], axis=1)
    np.testing.assert_array_equal(sliced, full.tensor_copy())
    translated = _vv(b, _virtual(0, 2, 1), _virtual(0, 2, 1), 1).tensor_copy()
    np.testing.assert_allclose(translated, -full.tensor_copy(), atol=5e-13)
    m = full.memory
    assert m.peak_owned_numerical_bytes == 32 * 2 * 2 * 2 + 16 * b.ao.nbasis * (2 + 2 + 2)
    assert m.live_reader_numeric_bytes == 65536 + m.maximum_reader_tile_bytes
    assert m.retained_index_bytes == m.caller_index_bytes == 0
    oo = _oo(b, [[0, 1]], [[0, 0], [0, 1]], 1)
    om = oo.memory
    assert om.peak_owned_numerical_bytes == 32 * 2 * 1 * 2 + 16 * b.ao.nbasis * 3 + 16 * 3
    assert om.caller_index_bytes == om.retained_index_bytes == 16 * 3
    copied = full.tensor_copy()
    copied[:] = 123
    assert not np.array_equal(full.tensor_copy(), copied)


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_work_units",
                                 "maximum_tile_visits", "maximum_reader_tile_bytes"])
def test_oo_caps_precede_invalid_index_and_gauge_scans(tmp_path, field):
    b = _native_store(tmp_path)
    p = core._plan_periodic_correlation_occupied_density_factor_block(
        b.reference, b.schedule, b.reader, b.wannier, 1, 1, 0, 0, 2)
    caps = _caps(p)
    setattr(caps, field, getattr(caps, field) - 1)
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _oo(b, [[2**64 - 1, 0]], [[0, 0]], 0, caps=caps, gauge=np.full_like(b.gauge, np.nan))


def test_occupied_aliases_duplicate_indices_and_gauge_mismatch_fail_closed(tmp_path):
    b = _native_store(tmp_path)
    for rows in ([[1, 0]], [[0, 2]], [[0, 2**64 - 1]]):
        with pytest.raises(IndexError, match="occupied index|modular"):
            _oo(b, rows, [[0, 0]], 0)
    with pytest.raises(ValueError, match="duplicate"):
        _oo(b, [[0, 0], [0, 0]], [[0, 0]], 0)
    with pytest.raises(ValueError, match="sealed Wannier gauge"):
        _oo(b, [[0, 0]], [[0, 0]], 0, gauge=-b.gauge)


def test_vv_rejects_invalid_slices_and_foreign_space_before_consuming(tmp_path):
    b = _native_store(tmp_path)
    for selected in (_virtual(count=0), _virtual(begin=1), _virtual(translation=2), _virtual(count=2**64 - 1)):
        with pytest.raises(IndexError, match="virtual slice|translation"):
            _vv(b, selected, _virtual(), 0)
    foreign = _orbital_objects(b.reference, domain_columns=[[1, 1]])
    b.space = foreign.space
    with pytest.raises(ValueError, match="provenance|domain"):
        _vv(b, _virtual(), _virtual(), 0)


def test_oo_selection_and_payload_digest_wire_is_independent(tmp_path):
    b = _native_store(tmp_path)
    result = _oo(b, [[0, 1], [0, 0]], [[0, 1]], 1)
    def prefix(name):
        value = name.encode()
        return struct.pack(">Q", len(value)) + value + struct.pack(">I", 1)
    selection = prefix("vibeqc.periodic.correlation.density-factors.selection")
    selection += struct.pack(">IQQQQQQQQ", 0, 2, 1, 0, 1, 0, 0, 0, 1)
    assert hashlib.sha256(selection).hexdigest() == result.selection_identity_sha256
    payload = prefix("vibeqc.periodic.correlation.density-factors.payload")
    payload += struct.pack(">IQQQQQ", 0, 1, 0, 2, 2, 1)
    for z in result.tensor_copy().reshape(-1):
        payload += struct.pack(">dd", 0. if z.real == 0 else z.real, 0. if z.imag == 0 else z.imag)
    assert hashlib.sha256(payload).hexdigest() == result.payload_sha256
