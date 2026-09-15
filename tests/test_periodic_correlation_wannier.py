"""Tiny independent finite-torus tests; no SCF, integrals or target-size arrays."""

from __future__ import annotations

import gc
import hashlib
import math
import struct
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_pair_topology import (
    _make_budget,
    _make_dimensions,
)


def _options(**overrides):
    value = core._PeriodicCorrelationWannierOptions()
    value.gauge_unitarity_tolerance = 1e-12
    value.time_reversal_absolute_tolerance = 1e-12
    value.time_reversal_relative_tolerance = 1e-12
    value.real_absolute_tolerance = 1e-12
    value.real_relative_tolerance = 1e-12
    value.require_time_reversal = True
    value.require_real_home_coefficients = True
    for name, setting in overrides.items():
        setattr(value, name, setting)
    return value


def _reference(mesh=(3, 1, 1), nactive=2, *, frozen_indices=(),
               shift=(0, 0, 0), canonical_rotations=None, broken_tr=False):
    """Synthetic exact RHF algebra, with k-dependent nonorthogonal AO metric.

    All occupied bands are exactly degenerate. B(k) supplies S=B^H B,
    C=B^-1 Q and F=B^H Q E Q^H B; Q may arbitrarily rotate the degenerate
    occupied block. These satisfy the native state's algebraic contract,
    without pretending to be an actual self-consistent chemical reference.
    """

    cells = np.array(list(product(*(range(n) for n in mesh))))
    nk = len(cells)
    n_frozen = len(frozen_indices)
    noccupied = nactive + n_frozen
    nao = noccupied + 1
    energies = np.r_[np.full(noccupied, -1.5), 0.8]
    occupations = np.r_[np.full(noccupied, 2.0), 0.0]
    masks = []
    coefficients, overlaps = [], []
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
    data.n_effective_orbitals = nao
    data.electrons_per_cell = 2 * noccupied
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    for k, cell in enumerate(cells):
        fractional = (cell + 0.5 * np.array(shift)) / np.array(mesh)
        theta = 2 * np.pi * fractional.sum()
        b = np.diag(1.3 + 0.15 * np.cos(theta) + 0.1 * np.arange(nao)).astype(complex)
        b += 0.06j * np.sin(theta) * np.triu(np.ones((nao, nao)), 1)
        if broken_tr and k == 1:
            b[0, 0] += 0.2
        q = np.eye(nao, dtype=complex)
        if canonical_rotations is not None:
            q[:noccupied, :noccupied] = canonical_rotations[k]
        s = b.conj().T @ b
        c = np.linalg.solve(b, q)
        f = b.conj().T @ q @ np.diag(energies) @ q.conj().T @ b
        frozen = np.zeros(nao, dtype=np.uint8)
        frozen[list(frozen_indices)] = 1
        active = np.r_[np.ones(noccupied, dtype=np.uint8), 0]
        active[frozen != 0] = 0
        virtuals = np.r_[np.zeros(noccupied, dtype=np.uint8), 1]
        data.add_kpoint(data.reciprocal_lattice @ fractional, 1 / nk,
                        s, f, c, energies, occupations,
                        frozen.tolist(), active.tolist(), virtuals.tolist())
        coefficients.append(c)
        overlaps.append(s)
        masks.append(active.astype(bool))
    state = core._make_periodic_restricted_mean_field_state(data)
    dims = _make_dimensions(mesh, nactive, shift=shift)
    dims.n_basis = nao
    dims.n_effective_orbitals = nao
    dims.n_home_total_occupied = noccupied
    dims.n_auxiliary = 2 * nao
    dims.domain_ao_support_upper_bound = nao
    dims.domain_pao_upper_bound = nao
    dims.domain_pno_upper_bound = nao
    dims.domain_local_auxiliary_upper_bound = 2 * nao
    reference = core._make_periodic_correlation_admitted_reference(state, dims, _make_budget())
    return reference, cells, np.array(coefficients), np.array(overlaps), masks


def _make(reference, gauges, options=None, cap=None):
    if cap is None:
        cap = core._plan_periodic_correlation_wannier(
            reference.state.mesh, reference.state.n_basis,
            reference.state.n_correlated_occupied,
        ).peak_owned_numerical_bytes
    return core._make_periodic_correlation_wannier(
        reference, gauges, cap, options if options is not None else _options()
    )


def _gauges(nk, nactive):
    return np.tile(np.eye(nactive, dtype=np.complex128), (nk, 1, 1))


def _home(result):
    return np.array([result.cell_coefficients_copy(cell) for cell in range(result.n_cells)])


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1),
                                  (2, 3, 1), (2, 2, 2), (3, 2, 2)])
def test_explicit_fourier_and_full_finite_torus_metric_orthogonality(mesh):
    reference, cells, c, s, masks = _reference(mesh)
    nk = len(cells)
    gauge = _gauges(nk, 2)
    result = _make(reference, gauge)
    w = _home(result)
    fractional = cells / np.array(mesh)
    phase = np.exp(2j * np.pi * cells @ fractional.T)
    cg = np.array([c[k][:, masks[k]] @ gauge[k] for k in range(nk)])
    expected = np.einsum("Rk,kmi->Rmi", phase, cg) / nk
    np.testing.assert_allclose(w, expected, atol=2e-15, rtol=2e-15)

    # Independently materialize a TINY complete torus only in this test.
    # S_(R,T) = Nk^-1 sum_k exp(+ik.(R-T)) S(k). This directly verifies the
    # coefficient 1/Nk normalization against nontrivial AO overlap blocks.
    sao = np.einsum("Rk,kab,Tk->RaTb", phase, s, phase.conj()) / nk
    all_orbitals = np.empty((nk, result.n_basis, nk, 2), dtype=complex)
    indices = {tuple(cell): i for i, cell in enumerate(cells)}
    for r, cell in enumerate(cells):
        for l, center in enumerate(cells):
            translated = indices[tuple((cell - center) % mesh)]
            for mu in range(result.n_basis):
                for i in range(2):
                    value = result.translated_coefficient(r, l, mu, i)
                    assert value == w[translated, mu, i]
                    all_orbitals[r, mu, l, i] = value
    placed = all_orbitals.reshape(nk * result.n_basis, nk * 2)
    metric = sao.reshape(nk * result.n_basis, nk * result.n_basis)
    np.testing.assert_allclose(placed.conj().T @ metric @ placed, np.eye(nk * 2), atol=3e-15)
    assert result.diagnostics.time_reversal_compatible
    assert result.diagnostics.real_home_coefficients_compatible
    assert result.diagnostics.maximum_home_imaginary_magnitude < 2e-15


def test_nontrivial_occupied_unitary_gauge_and_degenerate_band_rephasing():
    reference, cells, _, _, _ = _reference()
    theta = 0.37
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]], dtype=complex)
    gauge = np.tile(rotation, (len(cells), 1, 1))
    rotated = _home(_make(reference, gauge))
    canonical = _home(_make(reference, _gauges(len(cells), 2)))
    np.testing.assert_allclose(rotated, canonical @ rotation, atol=2e-15)
    rng = np.random.default_rng(71)
    arbitrary = np.array([np.linalg.qr(rng.normal(size=(2, 2))
                                       + 1j * rng.normal(size=(2, 2)))[0]
                          for _ in cells])
    regauged_reference, _, _, _, _ = _reference(canonical_rotations=arbitrary)
    result = _make(regauged_reference,
                   np.ascontiguousarray(arbitrary.conj().transpose(0, 2, 1)))
    np.testing.assert_allclose(_home(result), canonical, atol=2e-15)
    assert result.state_identity_sha256 != reference.state.state_identity_sha256
    assert result.diagnostics.time_reversal_compatible


def test_k_dependent_unitaries_preserve_complex_coefficients_without_claiming_localization():
    reference, cells, c, _, masks = _reference()
    gauges = _gauges(len(cells), 2)
    gauges[1] *= np.exp(0.31j)
    result = _make(reference, gauges, _options(
        require_time_reversal=False, require_real_home_coefficients=False))
    phase = np.exp(2j * np.pi * cells @ (cells / np.array((3, 1, 1))).T)
    cg = np.array([c[k][:, masks[k]] @ gauges[k] for k in range(len(cells))])
    np.testing.assert_allclose(_home(result), np.einsum("Rk,kmi->Rmi", phase, cg) / len(cells), atol=2e-15)
    assert not result.diagnostics.time_reversal_compatible
    assert not result.diagnostics.real_home_coefficients_compatible
    assert result.diagnostics.maximum_home_imaginary_magnitude > 0.01
    with pytest.raises(ValueError, match="time reversal"):
        _make(reference, gauges)
    with pytest.raises(ValueError, match="real-gauge"):
        _make(reference, gauges, _options(require_time_reversal=False))


def test_frozen_mask_is_not_assumed_to_be_a_leading_contiguous_core():
    reference, cells, c, _, masks = _reference(frozen_indices=(1,))
    result = _make(reference, _gauges(len(cells), 2))
    expected_home = np.mean(np.array([c[k][:, masks[k]] for k in range(len(cells))]), axis=0)
    np.testing.assert_allclose(result.cell_coefficients_copy(0), expected_home, atol=2e-15)
    assert result.state.n_frozen_core == 1
    assert result.n_home_occupied == 2
    with pytest.raises(ValueError, match="nactive"):
        _make(reference, _gauges(len(cells), 3))


def test_broken_reference_time_reversal_and_shift_fail_closed():
    reference, cells, *_ = _reference(broken_tr=True)
    with pytest.raises(ValueError, match="overlap/Fock"):
        _make(reference, _gauges(len(cells), 2))
    allowed = _make(reference, _gauges(len(cells), 2), _options(
        require_time_reversal=False, require_real_home_coefficients=False))
    assert not allowed.diagnostics.time_reversal_compatible
    assert allowed.diagnostics.maximum_overlap_time_reversal_residual > 0.1
    shifted, cells, *_ = _reference(mesh=(2, 1, 1), shift=(1, 0, 0))
    with pytest.raises(ValueError, match="Gamma-centered"):
        _make(shifted, _gauges(len(cells), 2))


def _wire_string(value):
    encoded = value.encode("ascii")
    return struct.pack(">Q", len(encoded)) + encoded


def _wire_complex(values):
    lanes = np.asarray(values).reshape(-1).view(np.float64).copy()
    lanes[lanes == 0.0] = 0.0
    return b"".join(struct.pack(">d", value) for value in lanes)


def test_native_digest_wire_immutable_ownership_and_numerical_memory_contract():
    reference, cells, *_ = _reference()
    gauges = _gauges(len(cells), 2)
    result = _make(reference, gauges)
    memory = result.memory
    assert memory.coefficient_count == len(cells) * result.n_basis * 2
    assert memory.retained_coefficient_bytes == 16 * memory.coefficient_count
    assert memory.temporary_gauged_coefficient_bytes == memory.retained_coefficient_bytes
    assert memory.peak_owned_numerical_bytes == 2 * memory.retained_coefficient_bytes
    assert memory.caller_gauge_bytes == gauges.nbytes
    assert result.diagnostics.required_node_memory_bytes >= (
        reference.state.resident_bytes + gauges.nbytes + memory.peak_owned_numerical_bytes)
    prefix = lambda domain: _wire_string(domain) + struct.pack(">I", 1)
    wire = (prefix("vibeqc.periodic.correlation.wannier.gauge")
            + struct.pack(">QQ", len(cells), 2) + _wire_complex(gauges))
    assert result.gauge_payload_sha256 == hashlib.sha256(wire).hexdigest()
    original = _home(result)
    wire = (prefix("vibeqc.periodic.correlation.wannier.coefficients")
            + struct.pack(">IIIQQ", *result.mesh, result.n_basis, 2)
            + _wire_complex(original))
    assert result.coefficient_payload_sha256 == hashlib.sha256(wire).hexdigest()
    options = result.options
    identity_wire = (prefix("vibeqc.periodic.correlation.wannier.identity")
                     + struct.pack(">I", reference.state.digest_version))
    for string in (result.state_identity_sha256, result.calculation_identity,
                   result.allocation_identity, result.gauge_payload_sha256,
                   result.coefficient_payload_sha256,
                   "inverse-bloch-home-coefficients-1/Nk;active-mask-ascending;exact-gamma"):
        identity_wire += _wire_string(string)
    identity_wire += struct.pack(">dddddII", options.gauge_unitarity_tolerance,
                                 options.time_reversal_absolute_tolerance,
                                 options.time_reversal_relative_tolerance,
                                 options.real_absolute_tolerance,
                                 options.real_relative_tolerance, 1, 1)
    assert result.wannier_identity_sha256 == hashlib.sha256(identity_wire).hexdigest()
    repeat = _make(reference, gauges)
    assert repeat.wannier_identity_sha256 == result.wannier_identity_sha256
    relaxed = _make(reference, gauges, _options(require_real_home_coefficients=False))
    assert relaxed.coefficient_payload_sha256 == result.coefficient_payload_sha256
    assert relaxed.wannier_identity_sha256 != result.wannier_identity_sha256
    gauges[:] = 0
    copy = result.cell_coefficients_copy(0)
    copy[:] = 800
    del reference, repeat
    gc.collect()
    np.testing.assert_array_equal(_home(result), original)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_gauge_fails_before_transform(bad):
    reference, cells, *_ = _reference()
    gauges = _gauges(len(cells), 2)
    gauges[0, 0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        _make(reference, gauges)


def test_nonunitary_and_invalid_binding_shapes():
    reference, cells, *_ = _reference()
    gauges = _gauges(len(cells), 2)
    bad = gauges.copy()
    bad[1, 0, 0] = 1.01
    with pytest.raises(ValueError, match="unitary"):
        _make(reference, bad)
    for bad in (gauges.real.copy(), gauges.astype(np.complex64),
                gauges[:, :, :1], gauges.transpose(0, 2, 1), gauges[0]):
        with pytest.raises(ValueError, match="C-contiguous complex128"):
            _make(reference, bad)
    with pytest.raises(TypeError):
        _make(reference, gauges.tolist())


@pytest.mark.parametrize("name", ["gauge_unitarity_tolerance",
                                  "time_reversal_absolute_tolerance",
                                  "time_reversal_relative_tolerance",
                                  "real_absolute_tolerance", "real_relative_tolerance"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1e-8, 1.0])
def test_invalid_explicit_tolerances(name, bad):
    reference, cells, *_ = _reference(mesh=(1, 1, 1), nactive=1)
    with pytest.raises(ValueError, match="tolerances"):
        _make(reference, _gauges(len(cells), 1), _options(**{name: bad}))


def test_zero_unknown_and_one_byte_below_cap_fail_before_gauge_traversal():
    reference, cells, *_ = _reference()
    gauges = _gauges(len(cells), 2)
    peak = core._plan_periodic_correlation_wannier((3, 1, 1), 3, 2).peak_owned_numerical_bytes
    gauges[0, 0, 0] = np.nan
    for cap in (0, peak - 1):
        with pytest.raises(ValueError, match="byte cap"):
            _make(reference, gauges, cap=cap)
    with pytest.raises(ValueError, match="positive"):
        _make(reference, _gauges(len(cells), 2),
               _options(gauge_unitarity_tolerance=0.0))


@pytest.mark.parametrize("mesh,nao,nactive", [
    ((0, 1, 1), 2, 1), ((1, 1, 1), 0, 1), ((1, 1, 1), 2, 0),
    ((1, 1, 1), 1, 2), ((1, 1, 1), 2**64 - 1, 1),
    ((1, 1, 1), 2**40, 2**40), ((2**30 - 1, 2**30 - 1, 2), 2, 1),
    ((1, 1, 1), 2**57, 1),
])
def test_count_only_invalid_candidate_byte_and_hash_domain_overflow(mesh, nao, nactive):
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        core._plan_periodic_correlation_wannier(mesh, nao, nactive)


def test_target_storage_shape_is_count_only_and_not_quadratic_in_cells():
    plan = core._plan_periodic_correlation_wannier((8, 8, 8), 200, 20)
    assert plan.coefficient_count == 512 * 200 * 20
    assert plan.peak_owned_numerical_bytes == 2 * 16 * 512 * 200 * 20
    assert plan.caller_gauge_bytes == 16 * 512 * 20 * 20


def test_scalar_and_cell_copy_indices_are_checked():
    reference, cells, *_ = _reference()
    result = _make(reference, _gauges(len(cells), 2))
    for indices in ((result.n_cells, 0, 0), (0, result.n_basis, 0),
                    (0, 0, result.n_home_occupied)):
        with pytest.raises(IndexError):
            result.coefficient(*indices)
    with pytest.raises(IndexError):
        result.cell_coefficients_copy(result.n_cells)
    with pytest.raises(IndexError):
        result.translated_coefficient(0, result.n_cells, 0, 0)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1), (2, 2, 2)])
def test_converged_native_optimizer_owner_bridge_has_no_gauge_copy_and_keeps_receipt(mesh):
    from tests.test_periodic_correlation_iao_optimizer import _case, _run

    case = _case(mesh, nactive=1, complex_gauge=True)
    reference = case["reference"]
    optimizer = _run(case)
    assert optimizer.converged
    gauges = optimizer.gauges_copy()
    receipt = optimizer.optimizer_identity_sha256
    # The bridge receives no seed/IAO numerical owners. Release them before
    # its live-inventory check; the retained reference is shared, not copied.
    del case
    gc.collect()
    plan = core._plan_periodic_correlation_wannier(mesh, reference.state.n_basis, 1)
    bridged = core._make_periodic_correlation_wannier_from_iao_optimizer(
        reference, optimizer, plan.peak_owned_numerical_bytes, _options())
    index_bytes = 8 * reference.state.n_kpoints
    assert bridged.diagnostics.live_optimizer_index_bytes == index_bytes
    assert bridged.diagnostics.live_diabatic_seed_index_bytes == 0
    assert bridged.localization_identity_sha256 == receipt
    del optimizer
    gc.collect()
    raw = _make(reference, gauges)
    assert raw.localization_identity_sha256 == ""
    assert raw.diagnostics.live_optimizer_index_bytes == 0
    assert bridged.wannier_identity_sha256 == raw.wannier_identity_sha256
    replicas = reference.budget.mpi_ranks * reference.budget.workers_per_rank
    assert bridged.diagnostics.required_node_memory_bytes == raw.diagnostics.required_node_memory_bytes + replicas * index_bytes
    for cell in range(reference.state.n_kpoints):
        np.testing.assert_array_equal(bridged.cell_coefficients_copy(cell), raw.cell_coefficients_copy(cell))


def test_optimizer_bridge_refuses_unconverged_and_foreign_owner_or_disabled_audits():
    from tests.test_periodic_correlation_iao_optimizer import _case, _options as optimizer_options, _run

    case = _case()
    reference = case["reference"]
    limited = _run(case, options=optimizer_options(maximum_iterations=1, riemannian_gradient_tolerance=1e-25))
    assert not limited.converged
    plan = core._plan_periodic_correlation_wannier(reference.state.mesh, reference.state.n_basis, 2)
    with pytest.raises(ValueError, match="converged native localization"):
        core._make_periodic_correlation_wannier_from_iao_optimizer(reference, limited, plan.peak_owned_numerical_bytes, _options())
    optimized = _run(case)
    assert optimized.converged
    other = _case()["reference"]
    with pytest.raises(ValueError, match="exact admitted state allocation"):
        core._make_periodic_correlation_wannier_from_iao_optimizer(other, optimized, plan.peak_owned_numerical_bytes, _options())
    for name in ("require_time_reversal", "require_real_home_coefficients"):
        with pytest.raises(ValueError, match="time-reversal and real-home audits"):
            core._make_periodic_correlation_wannier_from_iao_optimizer(reference, optimized,
                plan.peak_owned_numerical_bytes, _options(**{name: False}))
    for cap in (0, plan.peak_owned_numerical_bytes - 1):
        with pytest.raises(ValueError, match="owned byte cap"):
            core._make_periodic_correlation_wannier_from_iao_optimizer(reference, optimized, cap, _options())
