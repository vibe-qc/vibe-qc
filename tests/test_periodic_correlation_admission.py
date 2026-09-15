"""Native gate between certified periodic RHF state and correlation work."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


_CALCULATION_IDENTITY = "1" * 64
_ALLOCATION_IDENTITY = "4" * 64


def _state(*, is_shift: tuple[int, int, int] = (0, 0, 0)) -> object:
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = _CALCULATION_IDENTITY
    data.reference_kind = core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    data.normalization = (
        core._PeriodicMeanFieldNormalizationConvention
        .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS
    )
    data.periodic_dimension = 3
    data.mesh = (2, 1, 1)
    data.is_shift = is_shift
    data.reciprocal_lattice = np.diag([2.0, 3.0, 4.0])
    data.converged = True
    data.n_basis = 3
    data.n_effective_orbitals = 3
    data.electrons_per_cell = 4
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = (
        core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    )

    coefficients = np.eye(3, dtype=np.complex128)
    overlap = np.eye(3, dtype=np.complex128)
    coordinates = (
        [0.5 * is_shift[0], 1.5 * is_shift[1], 2.0 * is_shift[2]],
        [1.0 + 0.5 * is_shift[0],
         1.5 * is_shift[1], 2.0 * is_shift[2]],
    )
    for coordinate, energies in zip(
        coordinates,
        ([-1.5, -1.0, 0.5], [-1.4, -0.8, 0.7]),
        strict=True,
    ):
        fock = np.diag(energies).astype(np.complex128)
        data.add_kpoint(
            coordinate,
            0.5,
            overlap,
            fock,
            coefficients,
            energies,
            [2.0, 2.0, 0.0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        )
    return core._make_periodic_restricted_mean_field_state(data)


def _dimensions(*, is_shift: tuple[int, int, int] = (0, 0, 0)) -> object:
    dimensions = core._PeriodicCorrelationStaticDimensions()
    dimensions.allocation_contract_version = (
        core._PERIODIC_CORRELATION_STREAMED_RESOURCE_CONTRACT_VERSION
    )
    dimensions.calculation_identity = _CALCULATION_IDENTITY
    dimensions.allocation_identity = _ALLOCATION_IDENTITY
    dimensions.static_inventory_complete = True
    dimensions.periodic_dimension = 3
    dimensions.mesh = (2, 1, 1)
    dimensions.is_shift = is_shift
    dimensions.n_kpoints = 2
    dimensions.n_basis = 3
    dimensions.n_effective_orbitals = 3
    dimensions.n_auxiliary = 6
    dimensions.n_home_total_occupied = 2
    dimensions.n_home_occupied = 1
    dimensions.n_home_virtual = 1
    dimensions.n_spin_channels = 1
    dimensions.triples_requested = False
    dimensions.symmetry_representative_count = 2
    dimensions.symmetry_weight_sum = 2
    dimensions.expected_pair_candidate_count = 2
    dimensions.factor_k_bra_block = 1
    dimensions.factor_k_ket_block = 1
    dimensions.factor_q_block = 1
    dimensions.factor_auxiliary_block = 2
    dimensions.domain_ao_support_upper_bound = 3
    dimensions.domain_pao_upper_bound = 2
    dimensions.domain_pno_upper_bound = 2
    dimensions.domain_local_occupied_upper_bound = 1
    dimensions.domain_local_auxiliary_upper_bound = 6
    dimensions.pair_domain_metadata_upper_bytes = 1024
    dimensions.external_bytes = 17
    return dimensions


def _budget(
    *, memory: int = 1024**3, scratch: int = 1024**3, workers: int = 2
) -> object:
    budget = core._PeriodicCorrelationResourceBudget()
    budget.memory_limit_bytes = memory
    budget.scratch_limit_bytes = scratch
    budget.mpi_ranks = 1
    budget.workers_per_rank = workers
    return budget


def _admit(
    *, state: object | None = None, dimensions: object | None = None,
    budget: object | None = None
) -> object:
    return core._make_periodic_correlation_admitted_reference(
        _state() if state is None else state,
        _dimensions() if dimensions is None else dimensions,
        _budget() if budget is None else budget,
    )


def test_admitted_reference_owns_state_and_internally_sealed_static_plan():
    state = _state()
    dimensions = _dimensions()
    budget = _budget()
    state_bytes = state.resident_bytes

    admitted = _admit(state=state, dimensions=dimensions, budget=budget)

    assert admitted.contract_version == (
        core._PERIODIC_CORRELATION_ADMITTED_REFERENCE_CONTRACT_VERSION
    )
    assert admitted.state_resident_bytes == state_bytes
    assert admitted.state.calculation_identity == _CALCULATION_IDENTITY
    assert admitted.state.numerical_payload_sha256 == state.numerical_payload_sha256
    assert admitted.state.state_identity_sha256 == state.state_identity_sha256
    assert admitted.state.periodic_dimension == 3
    assert admitted.dimensions.periodic_dimension == 3
    assert admitted.state.mesh == [2, 1, 1]
    assert admitted.state.frozen_core_mask(0) == [1, 0, 0]
    assert admitted.state.correlated_occupied_mask(0) == [0, 1, 0]
    assert admitted.dimensions.external_bytes == 17 + state_bytes
    assert dimensions.external_bytes == 17
    assert admitted.budget.memory_limit_bytes == 1024**3
    assert admitted.plan.stage == (
        core._PeriodicCorrelationEstimateStage.STATIC_PREFLIGHT
    )
    assert admitted.plan.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_PAIR_DOMAIN_CENSUS
    )
    assert admitted.plan.calculation_identity == _CALCULATION_IDENTITY
    assert admitted.plan.allocation_identity == _ALLOCATION_IDENTITY
    assert admitted.plan.census_identity == ""
    assert not admitted.plan.pair_domain_census_complete
    assert not admitted.plan.triple_domain_census_complete

    expected_dimensions = _dimensions()
    expected_dimensions.external_bytes += state_bytes
    expected_plan = core._plan_periodic_correlation_static_resources(
        expected_dimensions, _budget()
    )
    baseline_plan = core._plan_periodic_correlation_static_resources(
        _dimensions(), _budget()
    )
    assert admitted.plan.modeled_peak_memory_bytes == (
        expected_plan.modeled_peak_memory_bytes
    )
    assert admitted.plan.required_memory_bytes == (
        expected_plan.required_memory_bytes
    )
    assert [
        admitted_phase.retained_bytes - baseline_phase.retained_bytes
        for admitted_phase, baseline_phase in zip(
            admitted.plan.phases, baseline_plan.phases, strict=True
        )
    ] == [state_bytes] * len(admitted.plan.phases)

    dimensions.external_bytes = 99
    budget.memory_limit_bytes = 1
    returned_dimensions = admitted.dimensions
    returned_dimensions.external_bytes = 123
    assert admitted.dimensions.external_bytes == 17 + state_bytes
    assert admitted.budget.memory_limit_bytes == 1024**3


def test_admitted_reference_and_state_have_independent_safe_lifetimes():
    state = _state()
    admitted = _admit(state=state)
    wrapped_state = admitted.state

    del state
    gc.collect()
    assert admitted.state.fock(1).shape == (3, 3)

    del admitted
    gc.collect()
    assert wrapped_state.calculation_identity == _CALCULATION_IDENTITY
    assert wrapped_state.correlated_occupied_mask(0) == [0, 1, 0]


def test_admitted_reference_accepts_matching_shifted_mesh():
    shift = (1, 0, 0)
    admitted = _admit(
        state=_state(is_shift=shift),
        dimensions=_dimensions(is_shift=shift),
    )
    assert admitted.state.is_shift == [1, 0, 0]
    np.testing.assert_array_equal(
        admitted.state.kpoint_cartesian(0), [0.5, 0.0, 0.0]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("calculation_identity", "2" * 64, "calculation identity"),
        ("periodic_dimension", 2, "periodic dimension"),
        ("mesh", (1, 2, 1), "mesh"),
        ("is_shift", (1, 0, 0), "shift"),
        ("n_kpoints", 1, "k-point count"),
        ("n_basis", 4, "AO-basis size"),
        ("n_effective_orbitals", 2, "effective-orbital count"),
        ("n_home_total_occupied", 1, "total occupied count"),
        ("n_home_occupied", 2, "correlated occupied count"),
        ("n_home_virtual", 2, "virtual count"),
        ("n_spin_channels", 2, "closed-shell only"),
    ],
)
def test_admitted_reference_rejects_state_dimension_mismatch(
    field: str, value: object, message: str
):
    dimensions = _dimensions()
    setattr(dimensions, field, value)
    with pytest.raises(ValueError, match=message):
        _admit(dimensions=dimensions)


def test_admitted_reference_rejects_symmetry_until_reconstruction_is_sealed():
    dimensions = _dimensions()
    dimensions.symmetry_reduction_requested = True
    dimensions.symmetry_mapping_identity = "7" * 64
    dimensions.symmetry_representative_count = 1
    with pytest.raises(ValueError, match="rejects symmetry reduction"):
        _admit(dimensions=dimensions)


def test_admitted_reference_rejects_null_state():
    with pytest.raises(ValueError, match="non-null"):
        core._make_periodic_correlation_admitted_reference(
            None, _dimensions(), _budget()
        )


def test_admitted_reference_adds_state_bytes_before_exact_memory_gate():
    exploratory = _admit(budget=_budget(workers=1))
    required = exploratory.plan.required_memory_bytes

    exact = _admit(budget=_budget(memory=required, workers=1))
    assert exact.plan.required_memory_bytes == required
    assert exact.plan.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_PAIR_DOMAIN_CENSUS
    )
    with pytest.raises(RuntimeError, match="MemoryExceeded"):
        _admit(budget=_budget(memory=required - 1, workers=1))


def test_admitted_reference_byte_accounting_overflow_fails_before_planning():
    dimensions = _dimensions()
    dimensions.external_bytes = 2**64 - 1
    with pytest.raises(OverflowError, match="byte accounting overflow"):
        _admit(dimensions=dimensions)


def test_admitted_reference_surfaces_planner_arithmetic_overflow():
    dimensions = _dimensions()
    dimensions.per_rank_bytes = 2**64 - 1
    with pytest.raises(OverflowError, match="ArithmeticOverflow"):
        _admit(dimensions=dimensions)


def test_admitted_reference_cannot_bypass_missing_memory_limit():
    with pytest.raises(
        RuntimeError,
        match=r"MissingMemoryLimit; required_memory_bytes=\d+",
    ):
        _admit(budget=_budget(memory=0))


def test_static_admission_does_not_require_scratch_before_census():
    admitted = _admit(budget=_budget(scratch=0))
    assert admitted.plan.required_scratch_bytes == 0
    assert admitted.plan.admission == (
        core._PeriodicCorrelationAdmissionCode.READY_FOR_PAIR_DOMAIN_CENSUS
    )
