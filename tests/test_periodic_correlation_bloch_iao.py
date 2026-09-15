"""One-k Bloch-IAO algebra against tiny independent original-AO equations."""

from __future__ import annotations

import gc
import hashlib
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_pao_domain import _reference as _reduced_reference
from tests.test_periodic_correlation_wannier import _reference


def _options(**changes):
    out = core._PeriodicCorrelationBlochIAOOptions()
    for name in ("minimal_rank_absolute_floor", "depolarized_rank_absolute_floor", "iao_rank_absolute_floor"):
        setattr(out, name, 1e-14)
    for name in ("minimal_rank_relative_floor", "depolarized_rank_relative_floor", "iao_rank_relative_floor"):
        setattr(out, name, 1e-12)
    out.schur_negative_absolute_tolerance = 1e-11
    out.schur_negative_relative_tolerance = 1e-11
    out.validation_absolute_tolerance = 3e-10
    out.validation_relative_tolerance = 3e-10
    out.jacobi_max_sweeps = 64
    out.jacobi_relative_tolerance = 2e-15
    out.maximum_work_units = 2**63 - 1
    for key, value in changes.items():
        setattr(out, key, value)
    return out


def _plan(reference, r, controls=None):
    if controls is None:
        controls = _options()
    s = reference.state
    return core._plan_periodic_correlation_bloch_iao(
        s.n_basis, s.n_effective_orbitals, s.n_frozen_core + s.n_correlated_occupied,
        r, controls.jacobi_max_sweeps,
    )


def _provenance(reference, r, **changes):
    out = core._PeriodicCorrelationBlochIAOInputProvenance()
    out.cross_overlap_identity = "a" * 64
    out.minimal_basis_identity = "b" * 64
    out.caller_live_numerical_bytes = _plan(reference, r).borrowed_input_bytes
    for key, value in changes.items():
        setattr(out, key, value)
    return out


def _make(reference, s12, s22, labels, *, point=1, options=None, provenance=None, cap=None):
    options = _options() if options is None else options
    r = s12.shape[1]
    if provenance is None:
        provenance = _provenance(reference, r)
    if cap is None:
        cap = _plan(reference, r, options).peak_owned_numerical_bytes
    return core._make_periodic_correlation_bloch_iao(reference, point, s12, s22, labels,
                                                    provenance, cap, options)


def _inputs(y, s, r, *, seed=418, noise=0.15, include_discarded=False):
    """Synthetic physically realizable Gram blocks, not atomic basis data."""
    rng = np.random.default_rng(seed)
    e = y.shape[1]
    t = 0.25 * (rng.normal(size=(e, r)) + 1j * rng.normal(size=(e, r)))
    t[:r] += np.diag(np.linspace(1.0, 1.4, r))
    minimal_ao = y @ t
    if include_discarded:
        projected_complement = np.eye(len(s)) - y @ y.conj().T @ s
        minimal_ao += projected_complement @ (0.3 * rng.normal(size=minimal_ao.shape))
    cross = s @ minimal_ao
    minimal = minimal_ao.conj().T @ s @ minimal_ao + noise * np.eye(r)
    minimal = 0.5 * (minimal + minimal.conj().T)
    labels = np.array([17 + (i // 2) * 13 for i in range(r)], dtype=np.uint64)
    return np.ascontiguousarray(cross), np.ascontiguousarray(minimal), labels


def _orth(c, s):
    values, vectors = np.linalg.eigh(c.conj().T @ s @ c)
    return c @ ((vectors / np.sqrt(values)) @ vectors.conj().T)


def _ao_oracle(y, s, s12, s22, occupied_indices):
    """Original AO operators, not native retained-row case distinctions."""
    occupied = y[:, occupied_indices]
    pret = y @ y.conj().T @ s
    pocc = occupied @ occupied.conj().T @ s
    p12 = y @ y.conj().T @ s12
    p21 = np.linalg.solve(s22, s12.conj().T)
    depolarized = _orth(p12 @ p21 @ occupied, s)
    pdep = depolarized @ depolarized.conj().T @ s
    a = (pocc @ pdep + (pret - pocc) @ (pret - pdep)) @ p12
    g = a.conj().T @ s @ a
    d = a.conj().T @ s @ occupied
    b = np.linalg.solve(g, d)
    return a, g, b, d


def _ao_coefficients(result):
    return np.array([[result.ao_coefficient(mu, rho) for rho in range(result.memory.n_minimal)]
                     for mu in range(result.memory.n_basis)])


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("point", [0, 1, 2])
def test_complex_unorthogonalized_bloch_iao_matches_independent_original_ao_operators(r, point):
    reference, _, coefficients, overlap, _ = _reference()
    y, s = coefficients[point], overlap[point]
    inputs = _inputs(y, s, r)
    result = _make(reference, *inputs, point=point)
    expected_a, expected_g, expected_b, expected_d = _ao_oracle(y, s, *inputs[:2], [0, 1])
    a = _ao_coefficients(result)
    g, b, d = result.metric_copy(), result.occupied_coefficients_copy(), result.occupied_covariant_copy()
    np.testing.assert_allclose(a, expected_a, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(g, expected_g, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(b, expected_b, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(d, expected_d, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(a, y @ result.retained_coefficients_copy(), atol=1e-14)
    np.testing.assert_allclose(a @ b, y[:, :2], atol=3e-12)
    np.testing.assert_allclose(b.conj().T @ g @ b, np.eye(2), atol=3e-12)
    np.testing.assert_allclose(g @ b, d, atol=3e-12)
    assert np.max(np.abs(g - np.eye(r))) > 0.1  # no final atomic orthogonalization
    assert np.max(np.abs(a.imag)) > 0.01  # complex lanes retained
    assert result.point == point
    assert result.diagnostics.charged_work_units == result.memory.maximum_work_units
    assert not hasattr(result, "physical_source_certified")
    assert not hasattr(result, "localization_converged")


def test_all_occupied_includes_noncontiguous_frozen_partition_and_keeps_every_atom_label():
    reference, _, coefficients, overlaps, _ = _reference(frozen_indices=(1,))
    inputs = _inputs(coefficients[1], overlaps[1], 3)
    result = _make(reference, *inputs)
    assert [result.occupied_band(i) for i in range(3)] == [0, 1, 2]
    assert [result.atom_label(i) for i in range(3)] == inputs[2].tolist()
    a, g, b, d = _ao_oracle(coefficients[1], overlaps[1], *inputs[:2], [0, 1, 2])
    np.testing.assert_allclose(_ao_coefficients(result), a, atol=4e-12)
    np.testing.assert_allclose(result.occupied_coefficients_copy(), b, atol=4e-12)
    np.testing.assert_allclose(a @ b @ (a @ b).conj().T,
                               coefficients[1, :, :3] @ coefficients[1, :, :3].conj().T, atol=5e-12)
    assert result.state.n_frozen_core == 1


def test_arbitrary_occupied_gauge_does_not_rotate_the_atomic_iao_labels():
    reference, cells, c, s, _ = _reference(frozen_indices=(1,))
    inputs = _inputs(c[1], s[1], 4)
    result = _make(reference, *inputs)
    rng = np.random.default_rng(234)
    rotations = np.array([np.linalg.qr(rng.normal(size=(3, 3))
                                      + 1j * rng.normal(size=(3, 3)))[0] for _ in cells])
    changed, _, changed_c, _, _ = _reference(frozen_indices=(1,), canonical_rotations=rotations)
    altered = _make(changed, *inputs)
    np.testing.assert_allclose(_ao_coefficients(altered), _ao_coefficients(result), atol=5e-12)
    np.testing.assert_allclose(altered.occupied_coefficients_copy(),
                               result.occupied_coefficients_copy() @ rotations[1], atol=5e-12)
    np.testing.assert_allclose(_ao_coefficients(altered) @ altered.occupied_coefficients_copy(),
                               changed_c[1, :, :3], atol=5e-12)


def test_retained_projection_does_not_resurrect_discarded_positive_overlap_direction():
    reference, _, c, s, _ = _reduced_reference(reduced=True)
    inputs = _inputs(c[1], s[1], 3, include_discarded=True)
    result = _make(reference, *inputs)
    assert result.memory.n_effective == 3 < result.memory.n_basis == 4
    a, _, b, _ = _ao_oracle(c[1], s[1], *inputs[:2], [0, 1])
    np.testing.assert_allclose(_ao_coefficients(result), a, atol=5e-12)
    pret = c[1] @ c[1].conj().T @ s[1]
    np.testing.assert_allclose((np.eye(4) - pret) @ a, 0.0, atol=2e-13)
    np.testing.assert_allclose(a @ b, c[1, :, :2], atol=5e-12)
    assert np.linalg.norm((np.eye(4) - pret) @ np.linalg.solve(s[1], inputs[0])) > 0.05


def test_zero_schur_rank_is_valid_but_negative_schur_complement_is_not():
    reference, _, c, s, _ = _reference()
    inputs = _inputs(c[1], s[1], 2, noise=0.0)
    result = _make(reference, *inputs)
    assert abs(result.diagnostics.minimum_schur_eigenvalue) < 1e-13
    cross, minimal, labels = inputs
    with pytest.raises(ValueError, match="Schur-complement PSD"):
        _make(reference, cross, np.ascontiguousarray(minimal * 0.2), labels)


def test_single_occupied_single_minimal_and_shifted_one_point_algebra():
    reference, _, c, s, _ = _reference(nactive=1, shift=(1, 0, 0))
    inputs = _inputs(c[1], s[1], 1)
    result = _make(reference, *inputs)
    a, g, b, d = _ao_oracle(c[1], s[1], *inputs[:2], [0])
    np.testing.assert_allclose(_ao_coefficients(result), a, atol=3e-12)
    np.testing.assert_allclose(result.metric_copy(), g, atol=3e-12)
    np.testing.assert_allclose(result.occupied_coefficients_copy(), b, atol=3e-12)
    np.testing.assert_allclose(result.occupied_covariant_copy(), d, atol=3e-12)


def test_nonconvergence_and_overflowed_schur_tolerance_are_explicit_failures():
    reference, _, c, s, _ = _reference()
    with pytest.raises(RuntimeError, match="Jacobi"):
        _make(reference, *_inputs(c[1], s[1], 3), options=_options(jacobi_max_sweeps=1))
    cross, _, labels = _inputs(c[1], s[1], 2)
    huge = np.finfo(float).max
    minimal = np.eye(2, dtype=complex) * (huge / 2)
    with pytest.raises(OverflowError, match="tolerance overflow"):
        _make(reference, cross, minimal, labels,
              options=_options(schur_negative_absolute_tolerance=huge,
                               schur_negative_relative_tolerance=0.9))


def _from_retained_t(reference, y, s, t, minimal):
    cross = np.ascontiguousarray(s @ y @ t, dtype=complex)
    return cross, np.ascontiguousarray(minimal, dtype=complex), np.arange(t.shape[1], dtype=np.uint64)


def test_full_minimal_depolarized_and_final_atomic_ranks_fail_closed_separately():
    reference, _, c, s, _ = _reference()
    zero = _from_retained_t(reference, c[1], s[1], np.zeros((3, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="minimal overlap.*rank"):
        _make(reference, *zero)
    missing_occupied = _from_retained_t(reference, c[1], s[1],
                                        np.array([[1, 0], [0, 0], [0, 1]]), 2 * np.eye(2))
    with pytest.raises(ValueError, match="depolarized Gram.*rank"):
        _make(reference, *missing_occupied)
    missing_atomic = _from_retained_t(reference, c[1], s[1], np.diag([1, 1, 0]), 2 * np.eye(3))
    with pytest.raises(ValueError, match="final Gram.*rank"):
        _make(reference, *missing_atomic)


def test_depolarized_rank_cutoff_is_on_gram_eigenvalues_not_singular_values():
    reference, _, c, s, _ = _reference()
    alpha = 0.05
    t = np.array([[alpha, 0], [0, alpha], [0, 0]])
    inputs = _from_retained_t(reference, c[1], s[1], t, np.eye(2))
    with pytest.raises(ValueError, match="depolarized Gram.*rank"):
        _make(reference, *inputs, options=_options(depolarized_rank_absolute_floor=2e-5))
    done = _make(reference, *inputs, options=_options(depolarized_rank_absolute_floor=3e-6))
    assert done.diagnostics.minimum_depolarized_gram_eigenvalue == pytest.approx(alpha**4, rel=1e-11)
    assert done.diagnostics.minimum_iao_gram_eigenvalue == pytest.approx(alpha**2, rel=1e-11)


@pytest.mark.parametrize("b,e,p,r", [(3, 3, 2, 2), (4, 3, 2, 3), (70, 65, 10, 24)])
def test_exact_simultaneous_payload_and_borrowed_input_inventory(b, e, p, r):
    plan = core._plan_periodic_correlation_bloch_iao(b, e, p, r, 64)
    assert plan.borrowed_input_bytes == 16 * b * r + 16 * r * r + 8 * r
    assert plan.output_numerical_bytes == 16 * e * r + 16 * r * r + 32 * r * p + 8 * r + 8 * p
    assert plan.projection_workspace_bytes == 16 * e * r + 32 * e * p
    assert plan.matrix_workspace_bytes == 48 * r * r
    assert plan.scalar_workspace_bytes == 8 * r + 32 * b
    assert plan.peak_owned_numerical_bytes == (32 * e * r + 32 * e * p + 64 * r * r
                                              + 32 * r * p + 16 * r + 8 * p + 32 * b)
    assert plan.maximum_work_units == (plan.input_preflight_work_units + plan.projection_work_units
                                       + plan.construction_work_units + plan.validation_work_units)


def test_admission_precedes_numerical_reads_and_accounts_extra_live_input_owner_bytes():
    reference, _, c, s, _ = _reference()
    cross, minimal, labels = _inputs(c[1], s[1], 2)
    plan = _plan(reference, 2)
    cross[:] = np.nan
    for cap in (0, plan.peak_owned_numerical_bytes - 1):
        with pytest.raises((ValueError, RuntimeError), match="byte cap"):
            _make(reference, cross, minimal, labels, cap=cap)
    with pytest.raises(ValueError, match="inventory"):
        _make(reference, cross, minimal, labels,
              provenance=_provenance(reference, 2, caller_live_numerical_bytes=plan.borrowed_input_bytes - 1))
    with pytest.raises((ValueError, RuntimeError), match="work budget"):
        _make(reference, cross, minimal, labels,
              options=_options(maximum_work_units=plan.input_preflight_work_units - 1))
    with pytest.raises(ValueError, match="finite"):
        _make(reference, cross, minimal, labels)
    inputs = _inputs(c[1], s[1], 2)
    baseline = _make(reference, *inputs)
    extra = _make(reference, *inputs, provenance=_provenance(reference, 2,
                  caller_live_numerical_bytes=plan.borrowed_input_bytes + 4096))
    assert extra.diagnostics.required_node_memory_bytes >= baseline.diagnostics.required_node_memory_bytes + 4096
    with pytest.raises((ValueError, RuntimeError), match="node memory"):
        _make(reference, *inputs, provenance=_provenance(reference, 2, caller_live_numerical_bytes=2**62))
    for work in (plan.input_preflight_work_units,
                 plan.input_preflight_work_units + plan.projection_work_units,
                 plan.maximum_work_units - 1):
        with pytest.raises((ValueError, RuntimeError), match="work budget"):
            _make(reference, *inputs, options=_options(maximum_work_units=work))


def test_strict_existing_array_boundary_hermiticity_alignment_and_readonly_inputs():
    reference, _, c, s, _ = _reference()
    cross, minimal, labels = _inputs(c[1], s[1], 2)
    for bad in (cross.real.copy(), cross.astype(np.complex64), np.asfortranarray(cross)):
        with pytest.raises(ValueError, match="C-contiguous complex128"):
            _make(reference, bad, minimal, labels)
    with pytest.raises(ValueError, match="uint64"):
        _make(reference, cross, minimal, labels.astype(np.int64))
    bad_minimal = minimal.copy()
    bad_minimal[0, 1] += 1e-15j
    with pytest.raises(ValueError, match="exactly Hermitian"):
        _make(reference, cross, bad_minimal, labels)
    misaligned = np.ndarray(cross.shape, dtype=complex, buffer=bytearray(cross.nbytes + 1), offset=1)
    misaligned[:] = cross
    with pytest.raises(ValueError, match="alignment"):
        _make(reference, misaligned, minimal, labels)
    for array in (cross, minimal, labels):
        array.flags.writeable = False
    done = _make(reference, cross, minimal, labels)
    assert done.memory.n_minimal == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_cross_and_minimal_views_are_rejected(bad):
    reference, _, c, s, _ = _reference()
    cross, minimal, labels = _inputs(c[1], s[1], 2)
    broken = cross.copy()
    broken[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        _make(reference, broken, minimal, labels)
    broken = minimal.copy()
    broken[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        _make(reference, cross, broken, labels)


def _wire_string(value):
    data = value.encode("ascii")
    return struct.pack(">Q", len(data)) + data


def _wire_complex(array):
    lanes = np.asarray(array, dtype=complex).reshape(-1).view(float).copy()
    lanes[lanes == 0.0] = 0.0
    return b"".join(struct.pack(">d", x) for x in lanes)


def test_actual_payload_hashes_distinguish_caller_claims_and_result_retains_no_input_arrays():
    reference, _, c, s, _ = _reference()
    cross, minimal, labels = _inputs(c[1], s[1], 2)
    result = _make(reference, cross, minimal, labels)
    wire = (_wire_string("vibeqc.periodic.correlation.bloch-iao.input")
            + struct.pack(">IQQQ", 1, 1, 3, 2) + _wire_complex(cross) + _wire_complex(minimal)
            + b"".join(struct.pack(">Q", int(label)) for label in labels))
    assert result.input_payload_sha256 == hashlib.sha256(wire).hexdigest()
    arrays = (result.retained_coefficients_copy(), result.metric_copy(),
              result.occupied_coefficients_copy(), result.occupied_covariant_copy())
    wire = (_wire_string("vibeqc.periodic.correlation.bloch-iao.output") + struct.pack(">IQQQ", 1, 3, 2, 2)
            + b"".join(_wire_complex(a) for a in arrays)
            + b"".join(struct.pack(">Q", int(label)) for label in labels)
            + struct.pack(">QQ", 0, 1))
    assert result.output_payload_sha256 == hashlib.sha256(wire).hexdigest()
    alternate = _make(reference, cross, minimal, labels,
                       provenance=_provenance(reference, 2, cross_overlap_identity="c" * 64))
    assert alternate.input_payload_sha256 == result.input_payload_sha256
    assert alternate.output_payload_sha256 == result.output_payload_sha256
    assert alternate.iao_identity_sha256 != result.iao_identity_sha256
    assert result.declared_provenance.cross_overlap_identity == "a" * 64
    changed_labels = labels.copy()
    changed_labels[0] = 999
    relabeled = _make(reference, cross, minimal, changed_labels)
    assert relabeled.input_payload_sha256 != result.input_payload_sha256
    assert relabeled.atom_label(0) == 999
    cross[:] = 0
    minimal[:] = 0
    labels[:] = 0
    del reference, alternate, relabeled, c, s
    gc.collect()
    np.testing.assert_array_equal(result.retained_coefficients_copy(), arrays[0])
    assert result.atom_label(0) == 17
    assert result.state.n_basis == 3
    copy = result.metric_copy()
    copy[:] = 0
    np.testing.assert_array_equal(result.metric_copy(), arrays[1])
    for method, args in ((result.atom_label, (2,)), (result.occupied_band, (2,)), (result.ao_coefficient, (3, 0))):
        with pytest.raises(IndexError):
            method(*args)


def test_invalid_dimensions_controls_provenance_and_count_overflow_fail_closed():
    for b, e, p, r in ((3, 3, 0, 2), (3, 3, 2, 1), (3, 2, 2, 3), (3, 4, 2, 2)):
        with pytest.raises(ValueError):
            core._plan_periodic_correlation_bloch_iao(b, e, p, r, 64)
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        core._plan_periodic_correlation_bloch_iao(2**32, 2**32, 2, 2**31, 64)
    reference, _, c, s, _ = _reference()
    inputs = _inputs(c[1], s[1], 2)
    for changes in ({"jacobi_max_sweeps": 0}, {"jacobi_relative_tolerance": 0.0},
                    {"validation_absolute_tolerance": np.nan}, {"maximum_work_units": 0},
                    {"iao_rank_absolute_floor": 0.0, "iao_rank_relative_floor": 0.0}):
        with pytest.raises(ValueError):
            _make(reference, *inputs, options=_options(**changes), cap=1_000_000)
    with pytest.raises(ValueError, match="provenance"):
        _make(reference, *inputs, provenance=_provenance(reference, 2, cross_overlap_identity="not-a-digest"))
    with pytest.raises(IndexError, match="k-point"):
        _make(reference, *inputs, point=3)
    with pytest.raises((ValueError, RuntimeError), match="limited"):
        _make(reference, *inputs, options=_options(jacobi_max_sweeps=257))
