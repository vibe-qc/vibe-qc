"""Tiny complex canonical PAO-space tests; no SCF or chemistry calculations."""

from __future__ import annotations

import gc
import hashlib
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_pao_domain import (
    _domain,
    _make as _make_domain,
    _options as _domain_options,
    _reference,
)
from tests.test_periodic_correlation_pair_topology import _make_budget


def _options(**changes):
    result = core._PeriodicCorrelationPAOSpaceOptions()
    result.rank_absolute_cutoff = 1e-10
    result.rank_relative_cutoff = 1e-10
    result.negative_absolute_tolerance = 1e-10
    result.negative_relative_tolerance = 1e-10
    result.validation_absolute_tolerance = 1e-10
    result.validation_relative_tolerance = 1e-10
    result.max_sweeps = 100
    result.relative_eigensolver_tolerance = 1e-14
    for name, value in changes.items():
        setattr(result, name, value)
    return result


def _make(reference, domain, *, cap=None, options=None):
    if cap is None:
        cap = core._plan_periodic_correlation_pao_space(
            domain.domain_dimension, domain.domain_dimension).peak_owned_numerical_bytes
    return core._make_periodic_correlation_pao_space(
        reference, domain, cap, _options() if options is None else options)


def _geometry(mesh=(3, 1, 1), *, full=False, complex_case=False,
              dispersive=True, reduced=False, mixed=True):
    """Reuse a synthetic metric, but split virtual band energies for F tests.

    Reassembly is still an exact generalized eigenproblem. No orbital root is
    optimized, no integrals evaluated, and no external QC program is imported.
    """

    reference, cells, c, s, _ = _reference(mesh, broken_tr=complex_case,
                                           reduced=reduced, mixed=mixed)
    if dispersive:
        data = core._PeriodicRestrictedMeanFieldInput()
        data.calculation_identity = reference.state.calculation_identity
        data.reference_kind = core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
        data.normalization = (core._PeriodicMeanFieldNormalizationConvention
                              .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS)
        data.periodic_dimension = 3
        data.mesh = mesh
        data.is_shift = (0, 0, 0)
        data.reciprocal_lattice = np.diag([2.0, 3.0, 5.0])
        data.converged = True
        data.n_basis = 4
        data.n_effective_orbitals = c.shape[2]
        data.electrons_per_cell = 4
        data.reference_energy_per_cell = -5.0
        data.minimum_band_gap_hartree = 0.1
        data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
        neff = c.shape[2]
        for k, cell in enumerate(cells):
            fractional = cell / np.array(mesh)
            energies = np.array([-1.5, -1.5, 0.5 + 0.1 * np.cos(2 * np.pi * fractional.sum()), 1.3])[:neff]
            fock = s[k] @ c[k] @ np.diag(energies) @ c[k].conj().T @ s[k]
            data.add_kpoint(data.reciprocal_lattice @ fractional, 1 / len(cells), s[k], fock, c[k],
                            energies, np.r_[2.0, 2.0, np.zeros(neff - 2)],
                            [0, 1] + [0] * (neff - 2), [1, 0] + [0] * (neff - 2),
                            [0, 0] + [1] * (neff - 2))
        state = core._make_periodic_restricted_mean_field_state(data)
        dims = reference.dimensions
        # The old native state is replaced, not an additional persistent owner.
        dims.external_bytes -= reference.state_resident_bytes
        reference = core._make_periodic_correlation_admitted_reference(state, dims, _make_budget())
    columns = _domain(cells, full=full)
    domain = _make_domain(reference, columns, options=_domain_options(
        require_time_reversal=not complex_case, require_real_matrices=not complex_case))
    return reference, domain


def _oracle(domain, cutoff):
    s, f = domain.overlap_copy(), domain.fock_copy()
    values, u = np.linalg.eigh(s)
    retained = values > cutoff
    x = u[:, retained] / np.sqrt(values[retained])
    h = x.conj().T @ f @ x
    eps, v = np.linalg.eigh((h + h.conj().T) / 2)
    return values, u[:, retained] @ u[:, retained].conj().T, x @ v, eps


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 2, 1)])
@pytest.mark.parametrize("full", [False, True])
def test_metric_fock_and_right_oriented_retained_projector_against_numpy(mesh, full):
    reference, domain = _geometry(mesh, full=full)
    result = _make(reference, domain)
    values, projector, expected_c, expected_eps = _oracle(domain, result.diagnostics.effective_rank_cutoff)
    c, eps = result.coefficients_copy(), result.energies_copy()
    s, f = domain.overlap_copy(), domain.fock_copy()
    r = result.retained_dimension
    assert result.usable
    assert np.all(np.diff(eps) >= 0)
    np.testing.assert_allclose(result.overlap_eigenvalues_copy(), values, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(eps, expected_eps, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(r), atol=3e-12)
    np.testing.assert_allclose(c.conj().T @ f @ c, np.diag(eps), atol=3e-12)
    np.testing.assert_allclose(c @ c.conj().T @ s, projector, atol=3e-12)
    # C C^H is the metric inverse kernel, NOT the retained Euclidean projector.
    assert np.linalg.norm(c @ c.conj().T - projector) > 1e-2
    np.testing.assert_allclose(c @ c.conj().T, expected_c @ expected_c.conj().T, atol=3e-12)
    assert result.diagnostics.retained_projector_relative_residual < 1e-10
    assert result.diagnostics.final_metric_frobenius_residual < 1e-10


def test_aggressive_rank_truncation_checks_projected_equations_not_full_generalized_residual():
    reference, domain = _geometry((1, 1, 1), full=True)
    values = np.linalg.eigvalsh(domain.overlap_copy())
    threshold = (values[-1] + values[-2]) / 2
    result = _make(reference, domain, options=_options(rank_absolute_cutoff=threshold,
                                                      rank_relative_cutoff=0.0))
    assert result.retained_dimension == 1
    c, eps = result.coefficients_copy(), result.energies_copy()
    s, f = domain.overlap_copy(), domain.fock_copy()
    np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(1), atol=1e-12)
    np.testing.assert_allclose(c.conj().T @ f @ c, np.diag(eps), atol=1e-12)
    assert np.linalg.norm(f @ c - (s @ c) * eps) > 1e-3


def test_complex_domain_is_preserved_and_semicanonicalized_without_a_real_cast():
    reference, domain = _geometry(complex_case=True)
    result = _make(reference, domain)
    c = result.coefficients_copy()
    assert c.dtype == np.complex128
    assert np.abs(c.imag).max() > 1e-3
    np.testing.assert_allclose(c.conj().T @ domain.overlap_copy() @ c,
                               np.eye(result.retained_dimension), atol=2e-12)
    np.testing.assert_allclose(c.conj().T @ domain.fock_copy() @ c,
                               np.diag(result.energies_copy()), atol=2e-12)
    values, _, _, eps = _oracle(domain, result.diagnostics.effective_rank_cutoff)
    np.testing.assert_allclose(result.overlap_eigenvalues_copy(), values, atol=1e-13)
    np.testing.assert_allclose(result.energies_copy(), eps, atol=1e-12)


def test_reduced_scf_space_and_degenerate_fock_keep_the_selected_span():
    reference, domain = _geometry((1, 1, 1), full=True, reduced=True, mixed=False,
                                  dispersive=False)
    result = _make(reference, domain)
    assert result.retained_dimension == 1
    assert np.count_nonzero(result.coefficients_copy()[3]) == 0
    np.testing.assert_allclose(result.energies_copy(), [0.8], atol=1e-15)
    deg_reference, deg_domain = _geometry(full=True, dispersive=False)
    degenerate = _make(deg_reference, deg_domain)
    np.testing.assert_allclose(degenerate.energies_copy(), 0.8, atol=2e-14)
    c = degenerate.coefficients_copy()
    _, target, _, _ = _oracle(deg_domain, degenerate.diagnostics.effective_rank_cutoff)
    np.testing.assert_allclose(c @ c.conj().T @ deg_domain.overlap_copy(), target, atol=3e-12)


def test_strict_overlap_threshold_equality_and_relative_cutoff():
    reference, _, *_ = _reference((1, 1, 1), mixed=False)
    singleton = _make_domain(reference, np.array([[0, 2]], np.uint64))
    baseline = _make(reference, singleton)
    eigenvalue = baseline.overlap_eigenvalue(0)
    with pytest.raises(RuntimeError, match="zero retained virtual rank"):
        _make(reference, singleton, options=_options(rank_absolute_cutoff=eigenvalue,
                                                     rank_relative_cutoff=0.0))
    below = _make(reference, singleton, options=_options(
        rank_absolute_cutoff=np.nextafter(eigenvalue, 0.0), rank_relative_cutoff=0.0))
    assert below.retained_dimension == 1
    ref, domain = _geometry((1, 1, 1), full=True)
    values = np.linalg.eigvalsh(domain.overlap_copy())
    relative = 0.5 * (values[-2] + values[-1]) / values[-1]
    result = _make(ref, domain, options=_options(rank_absolute_cutoff=0.0,
                                                rank_relative_cutoff=relative))
    assert result.retained_dimension == 1
    assert result.diagnostics.effective_rank_cutoff == relative * result.overlap_eigenvalue(domain.domain_dimension - 1)


def test_empty_geometry_is_unusable_but_nonempty_zero_rank_is_an_error():
    reference, *_ = _reference((1, 1, 1), mixed=False)
    empty = _make_domain(reference, np.empty((0, 2), np.uint64), cap=0)
    result = _make(reference, empty, cap=0)
    assert not result.usable
    assert result.domain_dimension == result.retained_dimension == 0
    assert result.memory.peak_owned_numerical_bytes == result.memory.borrowed_domain_bytes == 0
    assert result.coefficients_copy().shape == (0, 0)
    assert result.energies_copy().size == result.overlap_eigenvalues_copy().size == 0
    occupied_only = _make_domain(reference, np.array([[0, 0], [0, 1]], np.uint64))
    with pytest.raises(RuntimeError, match="zero retained virtual rank"):
        _make(reference, occupied_only)
    with pytest.raises(IndexError):
        result.energy(0)


@pytest.mark.parametrize("n,r", [(0, 0), (1, 1), (4, 1), (4, 2), (4, 4), (16, 3)])
def test_exact_count_only_memory_phases(n, r):
    plan = core._plan_periodic_correlation_pao_space(n, r)
    assert plan.borrowed_domain_index_bytes == 16 * n
    assert plan.borrowed_domain_matrix_bytes == 32 * n * n
    assert plan.borrowed_domain_bytes == 16 * n + 32 * n * n
    phases = [32 * n * n + 8 * n]
    if r:
        phases += [16 * n * n + 16 * n * r + 8 * n,
                   16 * n * r + 16 * r * r + 24 * n,
                   16 * n * r + 48 * r * r + 8 * n + 8 * r,
                   32 * n * r + 16 * r * r + 8 * n + 8 * r,
                   32 * n * r + 24 * n + 8 * r]
    else:
        phases += [0] * 5
    names = ("overlap_factorization_phase_bytes", "compact_orthogonalizer_phase_bytes",
             "projected_fock_phase_bytes", "fock_factorization_phase_bytes",
             "rotation_phase_bytes", "validation_phase_bytes")
    assert [getattr(plan, name) for name in names] == phases
    assert plan.peak_owned_numerical_bytes == max(phases)
    assert plan.output_numerical_bytes == 16 * n * r + 8 * n + 8 * r


def test_two_stage_memory_admission_uses_actual_rank_and_charges_live_borrowed_domain():
    reference, domain = _geometry((1, 1, 1), full=True)
    baseline = _make(reference, domain)
    n, r = domain.domain_dimension, baseline.retained_dimension
    exact = core._plan_periodic_correlation_pao_space(n, r)
    worst = core._plan_periodic_correlation_pao_space(n, n)
    assert exact.peak_owned_numerical_bytes < worst.peak_owned_numerical_bytes
    result = _make(reference, domain, cap=exact.peak_owned_numerical_bytes)
    for cap in (0, exact.overlap_factorization_phase_bytes - 1,
                exact.peak_owned_numerical_bytes - 1):
        with pytest.raises((ValueError, RuntimeError), match="byte cap"):
            _make(reference, domain, cap=cap)
    dims, budget = reference.dimensions, reference.budget
    expected = (dims.external_bytes + dims.shared_bytes
                + budget.mpi_ranks * (dims.per_rank_bytes + dims.localization_window_bytes_per_rank)
                + budget.mpi_ranks * budget.workers_per_rank
                * (exact.peak_owned_numerical_bytes + exact.borrowed_domain_bytes))
    assert result.diagnostics.required_node_memory_bytes == expected
    assert result.memory.output_numerical_bytes == (result.coefficients_copy().nbytes
        + result.energies_copy().nbytes + result.overlap_eigenvalues_copy().nbytes)


@pytest.mark.parametrize("name", ["rank_absolute_cutoff", "rank_relative_cutoff",
                                  "negative_absolute_tolerance", "negative_relative_tolerance",
                                  "validation_absolute_tolerance", "validation_relative_tolerance"])
@pytest.mark.parametrize("value", [-1e-12, np.inf, np.nan])
def test_invalid_rank_negative_and_validation_controls(name, value):
    reference, domain = _geometry((1, 1, 1))
    with pytest.raises(ValueError, match="controls"):
        _make(reference, domain, options=_options(**{name: value}))


def test_zero_control_pairs_rank_overflow_and_eigensolver_failure_are_explicit():
    reference, domain = _geometry(full=True)
    for changes in ({"rank_absolute_cutoff": 0.0, "rank_relative_cutoff": 0.0},
                    {"negative_absolute_tolerance": 0.0, "negative_relative_tolerance": 0.0},
                    {"validation_absolute_tolerance": 0.0, "validation_relative_tolerance": 0.0}):
        with pytest.raises(ValueError, match="positive control"):
            _make(reference, domain, options=_options(**changes))
    with pytest.raises(ValueError, match="eigensolver"):
        _make(reference, domain, options=_options(max_sweeps=0))
    with pytest.raises(RuntimeError, match="overlap eigensolver"):
        _make(reference, domain, options=_options(max_sweeps=1, relative_eigensolver_tolerance=1e-30))
    with pytest.raises((ValueError, RuntimeError), match="200 sweeps"):
        _make(reference, domain, options=_options(max_sweeps=201))
    with pytest.raises(ValueError, match="rank"):
        core._plan_periodic_correlation_pao_space(2, 3)
    with pytest.raises((OverflowError, ValueError, RuntimeError)):
        core._plan_periodic_correlation_pao_space(2**32, 2**32)


def _wire_string(value):
    encoded = value.encode("ascii")
    return struct.pack(">Q", len(encoded)) + encoded


def test_fixed_payload_digest_domain_identity_and_borrowed_input_lifetime():
    reference, domain = _geometry(complex_case=True)
    result = _make(reference, domain)
    c, eps, values = result.coefficients_copy(), result.energies_copy(), result.overlap_eigenvalues_copy()
    wire = (_wire_string("vibeqc.periodic.correlation.pao.space.payload")
            + struct.pack(">IQQ", core._PERIODIC_CORRELATION_PAO_SPACE_CONTRACT_VERSION,
                          result.domain_dimension, result.retained_dimension))
    for array in (c.view(np.float64), eps, values):
        clean = array.copy()
        clean[clean == 0] = 0.0
        wire += clean.astype(">f8").tobytes()
    assert result.payload_sha256 == hashlib.sha256(wire).hexdigest()
    assert result.domain_index_sha256 == domain.domain_index_sha256
    assert result.pao_domain_identity_sha256 == domain.pao_domain_identity_sha256
    again = _make(reference, domain)
    assert again.pao_space_identity_sha256 == result.pao_space_identity_sha256
    changed = _make(reference, domain, options=_options(negative_absolute_tolerance=2e-10))
    assert changed.payload_sha256 == result.payload_sha256
    assert changed.pao_space_identity_sha256 != result.pao_space_identity_sha256
    wrong_reference, _ = _geometry(complex_case=False)
    with pytest.raises(ValueError, match="mismatch"):
        _make(wrong_reference, domain)
    del reference, domain
    gc.collect()
    np.testing.assert_array_equal(result.coefficients_copy(), c)
    assert result.state.n_basis == 4
    c[:] = 0
    assert np.count_nonzero(result.coefficients_copy()) > 0
    with pytest.raises(IndexError):
        result.coefficient(result.domain_dimension, 0)


def test_equal_content_distinct_state_owner_is_not_an_uncharged_borrow():
    first_reference, first_domain = _geometry((1, 1, 1))
    second_reference, _ = _geometry((1, 1, 1))
    assert (first_reference.state.state_identity_sha256
            == second_reference.state.state_identity_sha256)
    # Matching digests do not prove identical backing allocations. The native
    # reference budget accounts one state owner, not both live snapshots.
    with pytest.raises(ValueError, match="mismatch"):
        _make(second_reference, first_domain)
    assert _make(first_reference, first_domain).usable
