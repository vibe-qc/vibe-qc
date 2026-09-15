"""Tiny physical-projection and independent finite-torus occupied-Fock oracles."""

from __future__ import annotations

import gc
import hashlib
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_pair_topology import _make_budget, _make_dimensions
from tests.test_periodic_correlation_wannier import (
    _make as _make_wannier,
    _options as _wannier_options,
    _reference,
    _wire_complex,
    _wire_string,
)


def _options(**overrides):
    result = core._PeriodicCorrelationOccupiedFockOptions()
    result.hermiticity_absolute_tolerance = 1e-12
    result.hermiticity_relative_tolerance = 1e-12
    result.time_reversal_absolute_tolerance = 1e-12
    result.time_reversal_relative_tolerance = 1e-12
    result.real_absolute_tolerance = 1e-12
    result.real_relative_tolerance = 1e-12
    result.require_time_reversal = True
    result.require_real_blocks = True
    for name, value in overrides.items():
        setattr(result, name, value)
    return result


def _fixture(mesh=(3, 1, 1), *, frozen=False, perturbation=0.0,
             asymmetric=0.0, complex_gauge=False, allocation="4" * 64):
    # Reuse only the synthetic nonorthogonal C/S fixture. Replace its flat
    # degenerate bands by dispersive canonical levels, and use a skew B.
    original, cells, c, s, masks = _reference(mesh, frozen_indices=(1,) if frozen else ())
    state = original.state
    nk, nao = len(cells), state.n_basis
    noccupied = state.n_correlated_occupied + state.n_frozen_core
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = state.calculation_identity
    data.reference_kind = state.reference_kind
    data.normalization = state.normalization
    data.periodic_dimension = 3
    data.mesh = mesh
    data.is_shift = (0, 0, 0)
    data.reciprocal_lattice = np.array([[2.0, 0.3, -0.1], [0.0, 3.0, 0.2], [0.2, 0.0, 4.0]])
    data.converged = True
    data.n_basis = nao
    data.n_effective_orbitals = nao
    data.electrons_per_cell = 2 * noccupied
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    focks, energies = [], []
    for k, cell in enumerate(cells):
        fractional = cell / np.array(mesh)
        dispersion = 0.07 * np.cos(2 * np.pi * fractional.sum())
        eps = np.r_[-2.2 + 0.3 * np.arange(noccupied) + dispersion, 0.8]
        # Physical F differs from the stored canonical eigenvalues only by
        # an optional admitted residual, to detect an eps-only substitution.
        fmo = np.diag(eps).astype(complex)
        fmo[0, 0] += perturbation
        fmo[0, 1] += asymmetric
        sc = s[k] @ c[k]
        f = sc @ fmo @ sc.conj().T
        core_mask = np.zeros(nao, dtype=np.uint8)
        if frozen:
            core_mask[1] = 1
        virtual = np.r_[np.zeros(noccupied, dtype=np.uint8), 1]
        data.add_kpoint(data.reciprocal_lattice @ fractional, 1 / nk,
                        s[k], f, c[k], eps, np.r_[np.full(noccupied, 2.0), 0.0],
                        core_mask.tolist(), masks[k].astype(int).tolist(), virtual.tolist())
        focks.append(f)
        energies.append(eps)
    state = core._make_periodic_restricted_mean_field_state(data)
    dims = _make_dimensions(mesh, 2, allocation_identity=allocation)
    dims.n_basis = dims.n_effective_orbitals = nao
    dims.n_home_total_occupied = noccupied
    dims.n_auxiliary = 2 * nao
    dims.domain_ao_support_upper_bound = nao
    dims.domain_pao_upper_bound = nao
    dims.domain_pno_upper_bound = nao
    dims.domain_local_auxiliary_upper_bound = 2 * nao
    reference = core._make_periodic_correlation_admitted_reference(state, dims, _make_budget())
    theta = 0.39
    u = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]], dtype=complex)
    gauges = np.tile(u, (nk, 1, 1))
    if complex_gauge:
        gauges[1] = np.array([[np.cos(theta), 1j * np.sin(theta)],
                              [1j * np.sin(theta), np.cos(theta)]])
    woptions = _wannier_options(require_time_reversal=not complex_gauge,
                                 require_real_home_coefficients=not complex_gauge)
    wannier = _make_wannier(reference, gauges, woptions)
    return reference, wannier, gauges, cells, c, np.array(focks), masks, np.array(energies)


def _make(reference, wannier, gauges, options=None, cap=None):
    if cap is None:
        cap = core._plan_periodic_correlation_occupied_fock(
            reference.state.mesh, reference.state.n_basis,
            reference.state.n_correlated_occupied).peak_owned_numerical_bytes
    return core._make_periodic_correlation_occupied_fock(
        reference, wannier, gauges, cap, options if options is not None else _options())


def _blocks(result):
    return np.array([result.block_copy(r) for r in range(result.n_cells)])


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1),
                                  (2, 3, 1), (2, 2, 2), (3, 2, 2)])
def test_direct_physical_projection_fourier_and_translated_orbitals(mesh):
    reference, wannier, gauge, cells, c, f, masks, _ = _fixture(mesh, frozen=True)
    result = _make(reference, wannier, gauge)
    nk, nao, nocc = len(cells), reference.state.n_basis, 2
    phase = np.exp(2j * np.pi * cells @ (cells / np.array(mesh)).T)
    cg = np.array([c[k][:, masks[k]] @ gauge[k] for k in range(nk)])
    projected = np.array([cg[k].conj().T @ f[k] @ cg[k] for k in range(nk)])
    expected = np.einsum("Rk,kij->Rij", phase, projected) / nk
    blocks = _blocks(result)
    np.testing.assert_allclose(blocks, expected, atol=3e-15, rtol=2e-15)

    # Only this tiny test constructs placed matrices. This checks the sign
    # bra-minus-ket against the actual Wannier AO coefficients, not a second
    # implementation of the native modular-index accessor.
    fao = (np.einsum("Rk,kab,Tk->RaTb", phase, f, phase.conj()) / nk).reshape(nk * nao, nk * nao)
    placed = np.empty((nk, nao, nk, nocc), dtype=complex)
    for r in range(nk):
        for mu in range(nao):
            for l in range(nk):
                for i in range(nocc):
                    placed[r, mu, l, i] = wannier.translated_coefficient(r, l, mu, i)
    placed = placed.reshape(nk * nao, nk * nocc)
    actual_operator = placed.conj().T @ fao @ placed
    for bra in range(nk):
        for ket in range(nk):
            for i in range(nocc):
                for j in range(nocc):
                    assert result.placed_element(bra, ket, i, j) == pytest.approx(
                        actual_operator[bra * nocc + i, ket * nocc + j], abs=4e-15)
    indices = {tuple(cell): r for r, cell in enumerate(cells)}
    for r, cell in enumerate(cells):
        minus = indices[tuple((-cell) % mesh)]
        np.testing.assert_allclose(blocks[r], blocks[minus].conj().T, atol=2e-15)
    assert result.diagnostics.maximum_canonical_projection_discrepancy < 5e-15
    assert result.diagnostics.maximum_translation_hermiticity_residual < 2e-15
    assert result.diagnostics.time_reversal_compatible
    assert result.diagnostics.real_blocks_compatible


def test_direct_F_not_canonical_eigenvalue_substitution():
    fixture = _fixture(perturbation=2e-9)
    reference, wannier, gauge, cells, c, f, masks, eps = fixture
    result = _make(reference, wannier, gauge)
    physical = np.array([(c[k][:, masks[k]] @ gauge[k]).conj().T @ f[k]
                          @ (c[k][:, masks[k]] @ gauge[k]) for k in range(len(cells))])
    canonical = np.array([gauge[k].conj().T @ np.diag(eps[k, masks[k]]) @ gauge[k]
                           for k in range(len(cells))])
    difference = physical - canonical
    assert np.max(np.abs(difference)) > 1e-9
    assert result.diagnostics.maximum_canonical_projection_discrepancy == pytest.approx(
        np.max(np.abs(difference)), abs=1e-15)
    assert result.diagnostics.canonical_projection_discrepancy_frobenius == pytest.approx(
        np.linalg.norm(difference), abs=2e-15)
    np.testing.assert_allclose(result.block_copy(0), physical.mean(axis=0), atol=1e-15)
    assert np.max(np.abs(result.block_copy(0) - canonical.mean(axis=0))) > 1e-9


def test_hermiticity_gate_does_not_silently_average_admitted_operator():
    reference, wannier, gauge, *_ = _fixture(asymmetric=2e-10)
    with pytest.raises(ValueError, match="physical projection failed Hermiticity"):
        _make(reference, wannier, gauge)
    result = _make(reference, wannier, gauge,
                   _options(hermiticity_absolute_tolerance=1e-9))
    assert result.diagnostics.maximum_projected_hermiticity_residual > 1e-10
    block = result.block_copy(0)
    assert np.max(np.abs(block - block.conj().T)) > 1e-10


def test_complex_gauge_time_reversal_and_real_gates_remain_independent():
    reference, wannier, gauge, *_ = _fixture(complex_gauge=True)
    with pytest.raises(ValueError, match="time reversal"):
        _make(reference, wannier, gauge)
    with pytest.raises(ValueError, match="real-block"):
        _make(reference, wannier, gauge, _options(require_time_reversal=False))
    result = _make(reference, wannier, gauge,
                   _options(require_time_reversal=False, require_real_blocks=False))
    assert not result.diagnostics.time_reversal_compatible
    assert not result.diagnostics.real_blocks_compatible
    assert np.max(np.abs(_blocks(result).imag)) > 0.01


def test_exact_gauge_state_and_allocation_coherence_before_projection():
    reference, wannier, gauge, *_ = _fixture()
    changed = gauge.copy()
    changed[1] *= -1
    with pytest.raises(ValueError, match="gauge content differs"):
        _make(reference, wannier, changed)
    other, _, _, *_ = _fixture()
    assert other.state.state_identity_sha256 == reference.state.state_identity_sha256
    with pytest.raises(ValueError, match="identical immutable"):
        _make(other, wannier, gauge)
    dims = _make_dimensions((3, 1, 1), 2, allocation_identity="9" * 64)
    alternate = core._make_periodic_correlation_admitted_reference(reference.state, dims, _make_budget())
    with pytest.raises(ValueError, match="provenance"):
        _make(alternate, wannier, gauge)


def test_memory_plan_both_workspace_regimes_and_target_count_only():
    for mesh, nao, nocc in [((3, 2, 2), 4, 2), ((31, 1, 1), 3, 2), ((8, 8, 8), 200, 20)]:
        count = int(np.prod(mesh))
        plan = core._plan_periodic_correlation_occupied_fock(mesh, nao, nocc)
        assert plan.element_count == count * nocc**2
        assert plan.retained_block_bytes == 16 * count * nocc**2
        assert plan.peak_owned_numerical_bytes == plan.retained_block_bytes + 16 * max(2 * nao, max(mesh))
        assert plan.projection_workspace_bytes == 32 * nao
        assert plan.fourier_workspace_bytes == 16 * max(mesh)
        assert plan.caller_gauge_bytes == plan.retained_block_bytes
        assert plan.live_wannier_bytes == 16 * count * nao * nocc


def test_memory_caps_precede_nonfinite_traversal_and_outputs_are_independent():
    reference, wannier, gauge, *_ = _fixture()
    result = _make(reference, wannier, gauge)
    peak = result.memory.peak_owned_numerical_bytes
    bad = gauge.copy()
    bad[0, 0, 0] = np.nan
    for cap in (0, peak - 1):
        with pytest.raises(ValueError, match="byte cap"):
            _make(reference, wannier, bad, cap=cap)
    with pytest.raises(ValueError, match="non-finite"):
        _make(reference, wannier, bad)
    assert result.diagnostics.required_node_memory_bytes >= (
        reference.state.resident_bytes + result.memory.live_wannier_bytes
        + result.memory.caller_gauge_bytes + peak)
    original = _blocks(result)
    gauge[:] = 0
    copy = result.block_copy(0)
    copy[:] = 77
    del reference, wannier
    gc.collect()
    np.testing.assert_array_equal(_blocks(result), original)


def test_versioned_native_payload_and_identity_wires():
    reference, wannier, gauge, *_ = _fixture()
    result = _make(reference, wannier, gauge)
    prefix = lambda domain: _wire_string(domain) + struct.pack(">I", 1)
    wire = (prefix("vibeqc.periodic.correlation.occupied-fock.blocks")
            + struct.pack(">IIIQ", *result.mesh, result.n_home_occupied)
            + _wire_complex(_blocks(result)))
    assert result.block_payload_sha256 == hashlib.sha256(wire).hexdigest()
    wire = prefix("vibeqc.periodic.correlation.occupied-fock.identity")
    for value in (reference.state.state_identity_sha256, reference.state.calculation_identity,
                  result.allocation_identity, result.wannier_identity_sha256,
                  result.gauge_payload_sha256, result.block_payload_sha256,
                  "physical-AO-F-projection;inverse-bloch-1/Nk;bra-minus-ket;exact-gamma"):
        wire += _wire_string(value)
    wire += struct.pack(">ddddddII", *([1e-12] * 6), 1, 1)
    assert result.occupied_fock_identity_sha256 == hashlib.sha256(wire).hexdigest()
    again = _make(reference, wannier, gauge)
    assert again.occupied_fock_identity_sha256 == result.occupied_fock_identity_sha256
    diagnostic = _make(reference, wannier, gauge, _options(require_real_blocks=False))
    assert diagnostic.block_payload_sha256 == result.block_payload_sha256
    assert diagnostic.occupied_fock_identity_sha256 != result.occupied_fock_identity_sha256


@pytest.mark.parametrize("name", ["hermiticity_absolute_tolerance", "hermiticity_relative_tolerance",
                                  "time_reversal_absolute_tolerance", "time_reversal_relative_tolerance",
                                  "real_absolute_tolerance", "real_relative_tolerance"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1e-9, 1.0])
def test_invalid_explicit_controls(name, bad):
    reference, wannier, gauge, *_ = _fixture(mesh=(1, 1, 1))
    with pytest.raises(ValueError, match="tolerances"):
        _make(reference, wannier, gauge, _options(**{name: bad}))


def test_unset_controls_strict_array_boundary_and_scalar_indices():
    reference, wannier, gauge, *_ = _fixture()
    with pytest.raises(ValueError, match="explicit positive"):
        _make(reference, wannier, gauge, core._PeriodicCorrelationOccupiedFockOptions())
    for bad in (gauge.real.copy(), gauge.astype(np.complex64), gauge[:, :, :1],
                gauge.transpose(0, 2, 1), gauge[0]):
        with pytest.raises(ValueError, match="C-contiguous complex128"):
            _make(reference, wannier, bad)
    result = _make(reference, wannier, gauge)
    for indices in ((result.n_cells, 0, 0), (0, 2, 0), (0, 0, 2)):
        with pytest.raises(IndexError):
            result.element(*indices)
    with pytest.raises(IndexError):
        result.placed_element(0, result.n_cells, 0, 0)


@pytest.mark.parametrize("mesh,nao,nocc", [((0, 1, 1), 2, 1), ((1, 1, 1), 2, 0),
                                         ((1, 1, 1), 2**64 - 1, 1),
                                         ((1, 1, 1), 2**40, 2**40),
                                         ((1, 1, 1), 2**57, 1)])
def test_count_only_invalid_candidate_byte_hash_extents(mesh, nao, nocc):
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        core._plan_periodic_correlation_occupied_fock(mesh, nao, nocc)
