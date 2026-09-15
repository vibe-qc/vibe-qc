"""Tiny independent PAO finite-torus algebra; no SCF or target-size arrays."""

from __future__ import annotations

import gc
import hashlib
import struct
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_pair_topology import (
    _make_budget,
    _make_dimensions,
)


def _options(**changes):
    result = core._PeriodicCorrelationPAODomainOptions()
    for name in ("hermitian_absolute_tolerance", "hermitian_relative_tolerance",
                 "time_reversal_absolute_tolerance", "time_reversal_relative_tolerance",
                 "real_absolute_tolerance", "real_relative_tolerance"):
        setattr(result, name, 1e-12)
    result.require_time_reversal = True
    result.require_real_matrices = True
    for name, value in changes.items():
        setattr(result, name, value)
    return result


def _reference(mesh=(3, 1, 1), *, reduced=False, mixed=True, shift=(0, 0, 0),
               virtual_gauge=None, broken_tr=False, nonhermitian=0.0,
               broken_projector_tr=False):
    """Synthetic exact generalized eigenproblem, not a chemical reference.

    S=B^H B, C=B^-1 U, F=B^H U E U^H B. Positive-overlap AO directions
    excluded from retained C remain present in S and F for the reduced test.
    Occupied masks are noncontiguous with respect to their frozen partition.
    """

    cells = np.array(list(product(*(range(n) for n in mesh))))
    nk = len(cells)
    nao = 4
    neff = 3 if reduced else 4
    nvirtual = neff - 2
    energies = np.array([-1.5, -1.5, 0.8, 0.8])
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = "1" * 64
    data.reference_kind = core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    data.normalization = (core._PeriodicMeanFieldNormalizationConvention
                          .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS)
    data.periodic_dimension = 3
    data.mesh = mesh
    data.is_shift = shift
    data.reciprocal_lattice = np.diag([2.0, 3.0, 5.0])
    data.converged = True
    data.n_basis = nao
    data.n_effective_orbitals = neff
    data.electrons_per_cell = 4
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    u0 = np.linalg.qr(np.random.default_rng(914).normal(size=(nao, nao)))[0] if mixed else np.eye(nao)
    coeff, overlaps, focks = [], [], []
    for k, cell in enumerate(cells):
        fractional = (cell + 0.5 * np.array(shift)) / np.array(mesh)
        theta = 2 * np.pi * fractional.sum()
        b = np.diag(1.3 + 0.15 * np.cos(theta) + 0.1 * np.arange(nao)).astype(complex)
        if mixed:
            b += (0.06 * np.cos(theta) + 0.09j * np.sin(theta)) * np.triu(np.ones((nao, nao)), 1)
        if broken_tr and k == 1:
            b[0, 0] += 0.2
        u = u0.astype(complex)
        if broken_projector_tr and k == 1:
            # The full virtual Fock block is degenerate. Selecting a different
            # retained direction preserves S/F TR but breaks the retained Q.
            angle = 0.41
            u[:, 2:] = u[:, 2:] @ np.array([[np.cos(angle), -np.sin(angle)],
                                            [np.sin(angle), np.cos(angle)]])
        if virtual_gauge is not None:
            u[:, 2:neff] = u[:, 2:neff] @ virtual_gauge[k]
        s = b.conj().T @ b
        c = np.linalg.solve(b, u[:, :neff])
        f = b.conj().T @ u @ np.diag(energies) @ u.conj().T @ b
        f[0, 1] += nonhermitian
        frozen = np.zeros(neff, dtype=np.uint8)
        frozen[1] = 1
        active = np.zeros(neff, dtype=np.uint8)
        active[0] = 1
        virtual = np.zeros(neff, dtype=np.uint8)
        virtual[2:] = 1
        data.add_kpoint(data.reciprocal_lattice @ fractional, 1 / nk, s, f, c,
                        energies[:neff], np.r_[2.0, 2.0, np.zeros(nvirtual)],
                        frozen.tolist(), active.tolist(), virtual.tolist())
        coeff.append(c)
        overlaps.append(s)
        focks.append(f)
    state = core._make_periodic_restricted_mean_field_state(data)
    dims = _make_dimensions(mesh, 1, shift=shift)
    dims.n_basis = nao
    dims.n_effective_orbitals = neff
    dims.n_home_total_occupied = 2
    dims.n_home_virtual = nvirtual
    dims.n_auxiliary = 2 * nao
    dims.domain_ao_support_upper_bound = nao
    dims.domain_pao_upper_bound = nao
    dims.domain_pno_upper_bound = nao
    dims.domain_local_auxiliary_upper_bound = 2 * nao
    ref = core._make_periodic_correlation_admitted_reference(state, dims, _make_budget())
    return ref, cells, np.array(coeff), np.array(overlaps), np.array(focks)


def _make(reference, columns, *, options=None, cap=None):
    if cap is None:
        cap = core._plan_periodic_correlation_pao_domain(
            reference.state.mesh, reference.state.n_basis, len(columns)
        ).peak_owned_numerical_bytes
    return core._make_periodic_correlation_pao_domain(
        reference, columns, cap, _options() if options is None else options)


def _domain(cells, nao=4, *, full=False):
    if full:
        return np.array(list(product(range(len(cells)), range(nao))), dtype=np.uint64)
    return np.array([(0, 0), (len(cells) - 1, 2), (0, 3)], dtype=np.uint64)


def _placed_oracle(mesh, cells, c, s, f, columns):
    """Materialize only the tiny test torus, with a unitary Fourier matrix.

    This does not use the native Q, phase helper or reduced-domain sum.
    Build all occupied/virtual placed orbitals, then project selected BvK
    AOs in the real-space AO metric and contract against real-space S/F.
    """

    nk, nao, neff = c.shape
    phase = np.exp(2j * np.pi * cells @ (cells / np.array(mesh)).T)
    transform = np.kron(phase / np.sqrt(nk), np.eye(nao))
    sk = np.zeros((nk * nao, nk * nao), complex)
    fk = np.zeros_like(sk)
    cvk = np.zeros((nk * nao, nk * (neff - 2)), complex)
    for k in range(nk):
        sk[k * nao:(k + 1) * nao, k * nao:(k + 1) * nao] = s[k]
        fk[k * nao:(k + 1) * nao, k * nao:(k + 1) * nao] = f[k]
        cvk[k * nao:(k + 1) * nao, k * (neff - 2):(k + 1) * (neff - 2)] = c[k, :, 2:]
    sr = transform @ sk @ transform.conj().T
    fr = transform @ fk @ transform.conj().T
    virtual = transform @ cvk
    q = virtual @ virtual.conj().T @ sr
    indices = columns[:, 0].astype(int) * nao + columns[:, 1].astype(int)
    selected = q[:, indices]
    return selected.conj().T @ sr @ selected, selected.conj().T @ fr @ selected


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1),
                                  (2, 3, 1), (2, 2, 2)])
def test_full_placed_pao_oracle_fixes_phase_normalization_and_even_mesh(mesh):
    reference, cells, c, s, f = _reference(mesh)
    columns = _domain(cells)
    result = _make(reference, columns)
    expected_s, expected_f = _placed_oracle(mesh, cells, c, s, f, columns)
    np.testing.assert_allclose(result.overlap_copy(), expected_s, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(result.fock_copy(), expected_f, atol=3e-14, rtol=3e-14)
    assert np.array_equal(result.overlap_copy(), result.overlap_copy().conj().T)
    assert np.array_equal(result.fock_copy(), result.fock_copy().conj().T)
    assert result.diagnostics.time_reversal_compatible
    assert result.diagnostics.real_matrices_compatible


def test_full_rank_complement_removes_all_occupied_including_nonleading_frozen():
    reference, cells, c, s, f = _reference((1, 1, 1))
    columns = _domain(cells, full=True)
    result = _make(reference, columns)
    q = np.eye(4) - c[0, :, :2] @ c[0, :, :2].conj().T @ s[0]
    np.testing.assert_allclose(result.overlap_copy(), q.conj().T @ s[0] @ q, atol=3e-15)
    np.testing.assert_allclose(result.fock_copy(), q.conj().T @ f[0] @ q, atol=3e-15)
    # Incorrect removal of only the active occupied orbital leaves a core PAO.
    wrong = np.eye(4) - c[0, :, :1] @ c[0, :, :1].conj().T @ s[0]
    assert np.linalg.norm(wrong.conj().T @ s[0] @ wrong - result.overlap_copy()) > 1
    assert reference.state.n_frozen_core == 1


def test_reduced_retained_space_does_not_resurrect_discarded_positive_overlap_ao():
    reference, cells, c, s, f = _reference((1, 1, 1), reduced=True, mixed=False)
    result = _make(reference, _domain(cells, full=True))
    assert np.linalg.eigvalsh(s[0]).min() > 1
    assert reference.state.n_effective_orbitals == 3 < reference.state.n_basis
    assert result.overlap(3, 3) == 0
    assert result.fock(3, 3) == 0
    assert np.linalg.matrix_rank(result.overlap_copy(), tol=1e-12) == 1
    wrong = np.eye(4) - c[0, :, :2] @ c[0, :, :2].conj().T @ s[0]
    assert (wrong.conj().T @ s[0] @ wrong)[3, 3].real > 1
    expected = _placed_oracle((1, 1, 1), cells, c, s, f, _domain(cells, full=True))
    np.testing.assert_allclose(result.overlap_copy(), expected[0], atol=2e-15)


def test_virtual_unitary_gauge_is_not_a_time_reversal_requirement():
    reference, cells, *_ = _reference()
    columns = _domain(cells)
    original = _make(reference, columns)
    rng = np.random.default_rng(913)
    gauges = np.array([np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
                       for _ in cells])
    changed, *_ = _reference(virtual_gauge=gauges)
    rotated = _make(changed, columns)
    np.testing.assert_allclose(rotated.overlap_copy(), original.overlap_copy(), atol=2e-15)
    np.testing.assert_allclose(rotated.fock_copy(), original.fock_copy(), atol=2e-15)
    assert rotated.diagnostics.time_reversal_compatible
    assert rotated.state_identity_sha256 != original.state_identity_sha256


def test_translation_and_domain_permutation_covariance():
    mesh = (2, 3, 1)
    reference, cells, *_ = _reference(mesh)
    columns = _domain(cells)
    result = _make(reference, columns)
    by_cell = {tuple(cell): i for i, cell in enumerate(cells)}
    translated = columns.copy()
    translated[:, 0] = [by_cell[tuple((cells[int(index)] + (1, 2, 0)) % mesh)] for index in columns[:, 0]]
    shift_result = _make(reference, translated)
    np.testing.assert_allclose(shift_result.overlap_copy(), result.overlap_copy(), atol=2e-15)
    np.testing.assert_allclose(shift_result.fock_copy(), result.fock_copy(), atol=2e-15)
    assert shift_result.domain_index_sha256 != result.domain_index_sha256
    order = [2, 0, 1]
    reordered = _make(reference, np.ascontiguousarray(columns[order]))
    np.testing.assert_allclose(reordered.overlap_copy(), result.overlap_copy()[np.ix_(order, order)], atol=2e-15)
    assert [reordered.column(i) for i in range(3)] == [tuple(row) for row in columns[order]]


def test_empty_geometry_and_nonempty_rank_zero_domain_never_force_an_orbital():
    reference, *_ = _reference((1, 1, 1), mixed=False)
    empty = _make(reference, np.empty((0, 2), np.uint64), cap=0)
    assert empty.domain_dimension == 0
    assert empty.memory.peak_owned_numerical_bytes == 0
    assert empty.memory.temporary_column_bytes == 0
    assert empty.overlap_copy().shape == empty.fock_copy().shape == (0, 0)
    zero = _make(reference, np.array([[0, 0], [0, 1]], np.uint64))
    assert zero.domain_dimension == 2
    assert np.count_nonzero(zero.overlap_copy()) == 0
    assert np.count_nonzero(zero.fock_copy()) == 0
    with pytest.raises(IndexError):
        empty.column(0)


def test_complex_reference_gates_and_no_imaginary_discard():
    mesh = (3, 1, 1)
    reference, cells, c, s, f = _reference(mesh, broken_tr=True)
    columns = _domain(cells)
    with pytest.raises(ValueError, match="time reversal"):
        _make(reference, columns)
    options = _options(require_time_reversal=False, require_real_matrices=False)
    result = _make(reference, columns, options=options)
    expected_s, expected_f = _placed_oracle(mesh, cells, c, s, f, columns)
    np.testing.assert_allclose(result.overlap_copy(), expected_s, atol=3e-14)
    np.testing.assert_allclose(result.fock_copy(), expected_f, atol=3e-14)
    assert np.max(np.abs(result.overlap_copy().imag)) > 1e-4
    assert not result.diagnostics.time_reversal_compatible
    assert not result.diagnostics.real_matrices_compatible
    with pytest.raises(ValueError, match="real tolerance"):
        _make(reference, columns, options=_options(require_time_reversal=False))
    shifted, cells, *_ = _reference((2, 1, 1), shift=(1, 0, 0))
    with pytest.raises(ValueError, match="Gamma-centered"):
        _make(shifted, _domain(cells))


def test_selected_virtual_projector_has_its_own_gauge_invariant_tr_gate():
    mesh = (3, 1, 1)
    reference, cells, c, s, f = _reference(mesh, reduced=True, broken_projector_tr=True)
    columns = _domain(cells)
    with pytest.raises(ValueError, match="selected virtual projector"):
        _make(reference, columns)
    result = _make(reference, columns, options=_options(
        require_time_reversal=False, require_real_matrices=False))
    assert result.diagnostics.maximum_overlap_time_reversal_residual < 1e-12
    assert result.diagnostics.maximum_fock_time_reversal_residual < 1e-12
    assert result.diagnostics.maximum_selected_projector_time_reversal_residual > 0.01
    assert not result.diagnostics.time_reversal_compatible
    expected_s, expected_f = _placed_oracle(mesh, cells, c, s, f, columns)
    np.testing.assert_allclose(result.overlap_copy(), expected_s, atol=3e-14)
    np.testing.assert_allclose(result.fock_copy(), expected_f, atol=3e-14)


def test_raw_hermitian_defect_is_audited_then_correction_is_reported():
    reference, cells, c, s, f = _reference((1, 1, 1), nonhermitian=2e-12)
    columns = _domain(cells, full=True)
    with pytest.raises(ValueError, match="Hermitian audit"):
        _make(reference, columns, options=_options(hermitian_absolute_tolerance=1e-15,
                                                   hermitian_relative_tolerance=1e-15))
    result = _make(reference, columns, options=_options(hermitian_absolute_tolerance=1e-10))
    expected_s, raw_f = _placed_oracle((1, 1, 1), cells, c, s, f, columns)
    np.testing.assert_allclose(result.overlap_copy(), expected_s, atol=2e-15)
    np.testing.assert_allclose(result.fock_copy(), (raw_f + raw_f.conj().T) / 2, atol=2e-15)
    assert result.diagnostics.maximum_raw_fock_hermitian_defect > 1e-14
    assert result.diagnostics.maximum_fock_hermitization_correction > 1e-14


def test_memory_admission_precedes_domain_contents_and_exact_payload_is_reported():
    reference, cells, *_ = _reference()
    columns = _domain(cells)
    d, nao = len(columns), reference.state.n_basis
    plan = core._plan_periodic_correlation_pao_domain(tuple(reference.state.mesh), nao, d)
    assert plan.peak_owned_numerical_bytes == 16 * d + 32 * d * d + 48 * nao
    assert plan.caller_domain_index_bytes == plan.retained_domain_index_bytes == 16 * d
    assert plan.retained_matrix_bytes == 32 * d * d
    assert plan.temporary_column_bytes == 48 * nao
    bad = columns.copy()
    bad[0, :] = np.iinfo(np.uint64).max
    for cap in (0, plan.peak_owned_numerical_bytes - 1):
        with pytest.raises((ValueError, RuntimeError), match="byte cap"):
            _make(reference, bad, cap=cap)
    result = _make(reference, columns, cap=plan.peak_owned_numerical_bytes)
    dims, budget = reference.dimensions, reference.budget
    expected_node = (dims.external_bytes + dims.shared_bytes
                     + budget.mpi_ranks * (dims.per_rank_bytes + dims.localization_window_bytes_per_rank)
                     + budget.mpi_ranks * budget.workers_per_rank
                     * (plan.peak_owned_numerical_bytes + plan.caller_domain_index_bytes))
    assert result.diagnostics.required_node_memory_bytes == expected_node


@pytest.mark.parametrize("columns,pattern", [
    ([[0, 0], [0, 0]], "duplicate"), ([[3, 0]], "modular"), ([[0, 4]], "extents")])
def test_domain_duplicate_and_index_extents(columns, pattern):
    reference, *_ = _reference()
    with pytest.raises(ValueError, match=pattern):
        _make(reference, np.array(columns, np.uint64))


@pytest.mark.parametrize("name", ["hermitian_absolute_tolerance", "hermitian_relative_tolerance",
                                  "time_reversal_absolute_tolerance", "time_reversal_relative_tolerance",
                                  "real_absolute_tolerance", "real_relative_tolerance"])
@pytest.mark.parametrize("value", [-1e-12, 1.0, np.inf, np.nan])
def test_controls_are_explicit_finite_and_bounded(name, value):
    reference, cells, *_ = _reference((1, 1, 1))
    with pytest.raises(ValueError, match="tolerances"):
        _make(reference, _domain(cells), options=_options(**{name: value}))


def test_boundary_dtype_shape_contiguity_alignment_and_diagnostic_caps():
    reference, cells, *_ = _reference()
    columns = _domain(cells)
    for bad in (columns.astype(np.int64), columns.astype(np.float64), columns.ravel(),
                np.asfortranarray(columns), np.ones((3, 3), np.uint64)):
        with pytest.raises(ValueError, match="uint64"):
            core._make_periodic_correlation_pao_domain(reference, bad, 10000, _options())
    misaligned = np.ndarray((3, 2), dtype=np.uint64, buffer=bytearray(49), offset=1)
    misaligned[:] = columns
    with pytest.raises(ValueError, match="alignment"):
        _make(reference, misaligned)
    with pytest.raises((ValueError, RuntimeError), match="32 columns"):
        core._make_periodic_correlation_pao_domain(reference, np.zeros((33, 2), np.uint64), 0, _options())
    with pytest.raises(ValueError, match="positive Hermitian"):
        _make(reference, columns, options=core._PeriodicCorrelationPAODomainOptions())
    with pytest.raises(TypeError):
        core._make_periodic_correlation_pao_domain(reference, columns.tolist(), 10000, _options())


def test_count_only_overflow_and_exact_finite_domain_upper_bound():
    with pytest.raises(ValueError, match="n_basis"):
        core._plan_periodic_correlation_pao_domain((1, 1, 1), 0, 0)
    with pytest.raises(ValueError, match="unique"):
        core._plan_periodic_correlation_pao_domain((2, 1, 1), 3, 7)
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        core._plan_periodic_correlation_pao_domain((2, 1, 1), 2**63, 1)
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        core._plan_periodic_correlation_pao_domain((1, 1, 1), 2**32, 2**32)
    plan = core._plan_periodic_correlation_pao_domain((8, 8, 8), 100, 20)
    assert plan.maximum_unique_domain_columns == 51200
    assert plan.peak_owned_numerical_bytes == 16 * 20 + 32 * 20**2 + 48 * 100


def _wire_string(value):
    encoded = value.encode("ascii")
    return struct.pack(">Q", len(encoded)) + encoded


def test_fixed_digest_wire_readonly_lifetime_and_no_numeric_input_copy():
    reference, cells, *_ = _reference()
    columns = _domain(cells)
    columns.flags.writeable = False
    result = _make(reference, columns)
    version = struct.pack(">I", core._PERIODIC_CORRELATION_PAO_DOMAIN_CONTRACT_VERSION)
    wire = (_wire_string("vibeqc.periodic.correlation.pao.domain") + version
            + struct.pack(">III", *reference.state.mesh)
            + struct.pack(">QQ", reference.state.n_basis, len(columns))
            + columns.astype(">u8").tobytes())
    assert result.domain_index_sha256 == hashlib.sha256(wire).hexdigest()
    s, f = result.overlap_copy(), result.fock_copy()
    arrays = []
    for value in (s, f):
        lanes = value.view(np.float64).copy()
        lanes[lanes == 0] = 0.0
        arrays.append(lanes.astype(">f8").tobytes())
    matrix_wire = (_wire_string("vibeqc.periodic.correlation.pao.matrices") + version
                   + struct.pack(">Q", len(columns)) + b"".join(arrays))
    assert result.matrix_payload_sha256 == hashlib.sha256(matrix_wire).hexdigest()
    again = _make(reference, columns)
    assert again.pao_domain_identity_sha256 == result.pao_domain_identity_sha256
    altered_control = _make(reference, columns, options=_options(real_absolute_tolerance=2e-12))
    assert altered_control.matrix_payload_sha256 == result.matrix_payload_sha256
    assert altered_control.pao_domain_identity_sha256 != result.pao_domain_identity_sha256
    del reference, columns
    gc.collect()
    assert result.state.n_basis == 4
    np.testing.assert_array_equal(result.overlap_copy(), s)
    copy = result.fock_copy()
    copy[:] = 0
    np.testing.assert_array_equal(result.fock_copy(), f)
    with pytest.raises(IndexError):
        result.overlap(result.domain_dimension, 0)
