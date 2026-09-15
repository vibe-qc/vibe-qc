"""Tiny cross-PAO-space overlaps against independently placed finite tori.

The deliberately dense matrices are test-only, at most 32 AO functions.
No SCF, chemical calculation, factor store, or external QC runtime is used.
"""

from __future__ import annotations

import gc
import hashlib
import struct
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_pao_domain import (
    _make as _make_domain,
    _options as _domain_options,
)
from tests.test_periodic_correlation_pao_space import _geometry, _make as _make_space


def _selection(begin=0, count=1, translation=0):
    selected = core._PeriodicCorrelationVirtualBlockSelection()
    selected.begin, selected.count, selected.translation_cell = begin, count, translation
    return selected


def _caps(plan):
    caps = core._PeriodicCorrelationPAOOverlapCaps()
    caps.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    caps.maximum_work_units = plan.work_units
    return caps


def _plan(reference, ld, ls, left, rd, rs, right):
    return core._plan_periodic_correlation_pao_space_overlap(reference, ld, ls, left, rd, rs, right)


def _make(reference, ld, ls, left, rd, rs, right, *, caps=None):
    if caps is None:
        caps = _caps(_plan(reference, ld, ls, left, rd, rs, right))
    return core._make_periodic_correlation_pao_space_overlap(reference, ld, ls, left, rd, rs, right, caps)


def _spaces(mesh=(3, 1, 1), *, complex_case=False, reduced=False):
    reference, ld = _geometry(mesh, complex_case=complex_case, reduced=reduced)
    nk = reference.state.n_kpoints
    # Different support, ordering and (when Nk > 1) dimensions from the left.
    columns = list(dict.fromkeys([(0, 1), (nk - 1, 3), (nk // 2, 2), (0, 2)]))
    rd = _make_domain(reference, np.array(columns, np.uint64), options=_domain_options(
        require_time_reversal=not complex_case, require_real_matrices=not complex_case))
    return reference, ld, _make_space(reference, ld), rd, _make_space(reference, rd)


def _torus(reference):
    """Build S_R and Q_R by an independent unitary discrete Fourier matrix."""

    state = reference.state
    mesh = np.array(state.mesh)
    cells = np.array(list(product(*(range(n) for n in mesh))))
    nk, nao = len(cells), state.n_basis
    fourier = np.exp(2j * np.pi * cells @ (cells / mesh).T) / np.sqrt(nk)
    transform = np.kron(fourier, np.eye(nao))
    sk = np.zeros((nk * nao, nk * nao), complex)
    vk = np.zeros((nk * nao, nk * state.n_effective_orbitals), complex)
    for k in range(nk):
        rows = slice(k * nao, (k + 1) * nao)
        sk[rows, rows] = state.overlap(k)
        virtual = state.coefficients(k)[:, np.asarray(state.virtual_mask(k), dtype=bool)]
        vk[rows, k * state.n_effective_orbitals:k * state.n_effective_orbitals + virtual.shape[1]] = virtual
    sr = transform @ sk @ transform.conj().T
    placed_virtual = transform @ vk
    qr = placed_virtual @ placed_virtual.conj().T @ sr
    return cells, sr, qr


def _placed(reference, domain, space, selected, cells, qr):
    mesh = np.array(reference.state.mesh)
    labels = {tuple(cell): i for i, cell in enumerate(cells)}
    indices = []
    for d in range(domain.domain_dimension):
        cell, ao = domain.column(d)
        shifted = (cells[cell] + cells[selected.translation_cell]) % mesh
        indices.append(labels[tuple(shifted)] * reference.state.n_basis + ao)
    return qr[:, indices] @ space.coefficients_copy()[:, selected.begin:selected.begin + selected.count]


def _oracle(reference, ld, ls, left, rd, rs, right):
    cells, sr, qr = _torus(reference)
    a = _placed(reference, ld, ls, left, cells, qr)
    b = _placed(reference, rd, rs, right, cells, qr)
    return a.conj().T @ sr @ b


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 3, 1), (2, 2, 2)])
def test_independent_domains_translations_match_explicit_finite_torus(mesh):
    reference, ld, ls, rd, rs = _spaces(mesh)
    nk = reference.state.n_kpoints
    left = _selection(0, ls.retained_dimension, nk - 1)
    right = _selection(0, rs.retained_dimension, nk // 2)
    actual = _make(reference, ld, ls, left, rd, rs, right)
    np.testing.assert_allclose(actual.matrix_copy(), _oracle(reference, ld, ls, left, rd, rs, right),
                               atol=8e-12, rtol=8e-12)
    assert actual.matrix_copy().dtype == np.complex128
    assert actual.left_selection.translation_cell == nk - 1
    assert actual.right_selection.count == rs.retained_dimension


def test_complex_overlaps_preserve_conjugation_and_common_translation_covariance():
    reference, ld, ls, rd, rs = _spaces(complex_case=True)
    left, right = _selection(0, ls.retained_dimension, 0), _selection(0, rs.retained_dimension, 1)
    ab = _make(reference, ld, ls, left, rd, rs, right).matrix_copy()
    ba = _make(reference, rd, rs, right, ld, ls, left).matrix_copy()
    np.testing.assert_allclose(ab, ba.conj().T, atol=2e-12)
    np.testing.assert_allclose(ab, _oracle(reference, ld, ls, left, rd, rs, right), atol=8e-12)
    assert np.abs(ab.imag).max() > 1e-5
    left.translation_cell, right.translation_cell = 2, 0
    shifted = _make(reference, ld, ls, left, rd, rs, right).matrix_copy()
    np.testing.assert_allclose(shifted, ab, atol=3e-12)
    # Translating only one side is not a common phase or an identity map.
    left.translation_cell = 0
    one_side = _make(reference, ld, ls, left, rd, rs, right).matrix_copy()
    assert np.linalg.norm(one_side - ab) > 1e-3


def test_same_space_identity_slicing_and_reduced_retained_virtual_projector():
    for reduced in (False, True):
        reference, domain = _geometry((2, 1, 1), full=True, reduced=reduced)
        space = _make_space(reference, domain)
        r = space.retained_dimension
        selected = _selection(0, r)
        full = _make(reference, domain, space, selected, domain, space, selected).matrix_copy()
        np.testing.assert_allclose(full, np.eye(r), atol=7e-12)
        small = _make(reference, domain, space, _selection(r - 1, 1),
                      domain, space, _selection(0, r)).matrix_copy()
        np.testing.assert_array_equal(small, full[-1:])
        translated = _selection(0, r, 1)
        overlap = _make(reference, domain, space, selected, domain, space, translated).matrix_copy()
        np.testing.assert_allclose(overlap, _oracle(reference, domain, space, selected,
                                                   domain, space, translated), atol=7e-12)
        assert np.linalg.norm(overlap - np.eye(r)) > 0.1
        if reduced:
            assert reference.state.n_effective_orbitals < reference.state.n_basis
            assert r == 2  # one retained virtual band in each of two cells
    # Both occupied columns, including the frozen member, are absent from Q.
    assert reference.state.n_frozen_core == reference.state.n_correlated_occupied == 1


def test_exact_memory_work_and_address_based_live_owner_deduplication():
    reference, domain = _geometry((3, 1, 1))
    space = _make_space(reference, domain)
    columns = np.array([domain.column(i) for i in range(domain.domain_dimension)], np.uint64)
    domain2 = _make_domain(reference, columns)
    space2 = _make_space(reference, domain2)
    assert domain2.pao_domain_identity_sha256 == domain.pao_domain_identity_sha256
    assert space2.pao_space_identity_sha256 == space.pao_space_identity_sha256
    a, b = _selection(0, 2), _selection(0, 1)
    same = _plan(reference, domain, space, a, domain, space, b)
    distinct = _plan(reference, domain, space, a, domain2, space2, b)
    mixed = _plan(reference, domain, space, a, domain2, space, b)
    assert (same.unique_domain_owners, same.unique_space_owners) == (1, 1)
    assert (distinct.unique_domain_owners, distinct.unique_space_owners) == (2, 2)
    assert (mixed.unique_domain_owners, mixed.unique_space_owners) == (2, 1)
    assert distinct.live_domain_bytes == mixed.live_domain_bytes == 2 * same.live_domain_bytes
    assert distinct.live_space_bytes == 2 * same.live_space_bytes == 2 * mixed.live_space_bytes
    nk, nao, neff, d = reference.state.n_kpoints, reference.state.n_basis, reference.state.n_effective_orbitals, domain.domain_dimension
    assert same.output_bytes == same.compensation_bytes == 16 * 2
    assert same.coefficient_panel_bytes == 16 * nao * 3
    assert same.scratch_bytes == 32 * nao
    assert same.peak_owned_numerical_bytes == 32 * 2 + 16 * nao * 5
    column_work = nao * d + nao**2 + 2 * nao * neff + 4 * nao
    assert same.work_units == nk * (3 * column_work + nao**2 + 2 * (nao + 1)) + 2
    dims, budget = reference.dimensions, reference.budget
    for plan in (same, distinct, mixed):
        assert plan.required_node_memory_bytes == (dims.external_bytes + dims.shared_bytes
            + budget.mpi_ranks * (dims.per_rank_bytes + dims.localization_window_bytes_per_rank)
            + budget.mpi_ranks * budget.workers_per_rank
            * (plan.peak_owned_numerical_bytes + plan.live_domain_bytes + plan.live_space_bytes))
    np.testing.assert_array_equal(_make(reference, domain, space, a, domain, space, b).matrix_copy(),
                                  _make(reference, domain, space, a, domain2, space2, b).matrix_copy())


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_work_units"])
@pytest.mark.parametrize("missing", [False, True])
def test_explicit_byte_and_work_caps_are_strict(field, missing):
    reference, ld, ls, rd, rs = _spaces()
    selected = _selection()
    caps = _caps(_plan(reference, ld, ls, selected, rd, rs, selected))
    setattr(caps, field, 0 if missing else getattr(caps, field) - 1)
    with pytest.raises(ValueError, match="cap"):
        _make(reference, ld, ls, selected, rd, rs, selected, caps=caps)


def test_live_owners_are_charged_against_admitted_node_budget():
    reference, domain = _geometry((2, 2, 2), full=True)
    space = _make_space(reference, domain)
    columns = np.array([domain.column(i) for i in range(domain.domain_dimension)], np.uint64)
    domain2 = _make_domain(reference, columns)
    space2 = _make_space(reference, domain2)
    selected = _selection(0, space.retained_dimension)
    plan = _plan(reference, domain, space, selected, domain2, space2, selected)
    dims, budget = reference.dimensions, reference.budget
    dims.external_bytes -= reference.state_resident_bytes
    budget.memory_limit_bytes = plan.required_node_memory_bytes - 1
    limited = core._make_periodic_correlation_admitted_reference(reference.state, dims, budget)
    assert _plan(limited, domain, space, selected, domain2, space2, selected).required_node_memory_bytes == plan.required_node_memory_bytes
    with pytest.raises(ValueError, match="node memory"):
        _make(limited, domain, space, selected, domain2, space2, selected, caps=_caps(plan))


def test_selection_aliases_overflow_foreign_owner_and_wrong_space_fail_closed():
    reference, ld, ls, rd, rs = _spaces()
    for selected in (_selection(count=0), _selection(begin=ls.retained_dimension),
                     _selection(translation=reference.state.n_kpoints),
                     _selection(begin=2**64 - 1), _selection(count=2**64 - 1)):
        with pytest.raises(IndexError, match="selected rank|modular"):
            _plan(reference, ld, ls, selected, rd, rs, _selection())
    foreign, fd, fs, _, _ = _spaces()
    assert foreign.state.state_identity_sha256 == reference.state.state_identity_sha256
    with pytest.raises(ValueError, match="state owners"):
        _plan(reference, ld, ls, _selection(), fd, fs, _selection())
    with pytest.raises(ValueError, match="provenance"):
        _plan(reference, ld, ls, _selection(), rd, ls, _selection())
    empty = _make_domain(reference, np.empty((0, 2), np.uint64), cap=0)
    empty_space = _make_space(reference, empty, cap=0)
    with pytest.raises(ValueError, match="usable nonzero"):
        _plan(reference, ld, ls, _selection(), empty, empty_space, _selection())


def test_result_lifetime_detached_copies_and_independent_hash_wire():
    reference, ld, ls, rd, rs = _spaces(complex_case=True)
    left, right = _selection(1, 1, 2), _selection(0, 2, 1)
    result = _make(reference, ld, ls, left, rd, rs, right)
    expected = result.matrix_copy()

    def string(value):
        encoded = value.encode()
        return struct.pack(">Q", len(encoded)) + encoded

    payload = string("vibeqc.periodic.correlation.pao-overlap.payload") + struct.pack(">IQQQ", 1, 3, 1, 2)
    for value in expected.ravel():
        payload += struct.pack(">dd", 0.0 if value.real == 0 else value.real,
                               0.0 if value.imag == 0 else value.imag)
    assert hashlib.sha256(payload).hexdigest() == result.payload_sha256
    identity = string("vibeqc.periodic.correlation.pao-overlap.identity") + struct.pack(">I", 1)
    for value in (reference.state.state_identity_sha256, reference.state.calculation_identity,
                  reference.dimensions.allocation_identity, ld.pao_domain_identity_sha256, ls.pao_space_identity_sha256):
        identity += string(value)
    identity += struct.pack(">QQQ", left.begin, left.count, left.translation_cell)
    identity += string(rd.pao_domain_identity_sha256) + string(rs.pao_space_identity_sha256)
    identity += struct.pack(">QQQ", right.begin, right.count, right.translation_cell)
    identity += string(result.payload_sha256)
    identity += string("retained-virtual-projector;direct-S-overlap;full-BZ-1/Nk;independent-negative-cell-phases;complex-retained")
    assert hashlib.sha256(identity).hexdigest() == result.identity_sha256
    result.matrix_copy()[:] = 41
    result.left_selection.begin = 0
    left.begin, right.translation_cell = 0, 0
    del reference, ld, ls, rd, rs
    gc.collect()
    np.testing.assert_array_equal(result.matrix_copy(), expected)
    assert result.left_selection.begin == 1
    assert result.element(0, 1) == expected[0, 1]
    with pytest.raises(IndexError):
        result.element(1, 0)
