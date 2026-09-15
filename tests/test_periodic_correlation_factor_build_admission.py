"""Native count-only admission gate for periodic numerical factor builds.

These tests exercise scalar inventories and tiny synthetic mean-field states.
The target-shaped probes construct no target state, q census, integral, or
factor tensor.
"""

from __future__ import annotations

import gc
import hashlib
import math
import resource
import struct
import sys
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


_CALCULATION_IDENTITY = "1" * 64
_ALLOCATION_IDENTITY = "2" * 64
_RECIPROCAL_LATTICE = np.diag([2.0, 3.0, 5.0])
_HUGE_LIMIT = 1 << 50
_FULL_RECIPROCAL_CENSUS_KAT = (
    "a0bb240fe23f1f1750b971480dd1c92b9819c627f194461dc683be0496dcca8a"
)
_RSGDF_CENSUS_KAT = (
    "3474c67131173f9ce1474ac7bb65882c18b1b21c3832cb501d07b97ea586465a"
)
_RSGDF_PLAN_KAT = (
    "0b174a532ffa54eaaadb0267c25a6c23ae2a3862c803b1fdab765f3d8326f42f"
)


class _CanonicalDigest:
    def __init__(self, domain: str, version: int) -> None:
        self._digest = hashlib.sha256()
        self.string(domain)
        self.u32(version)

    def u32(self, value: int) -> None:
        self._digest.update(struct.pack(">I", value))

    def i32(self, value: int) -> None:
        self._digest.update(struct.pack(">i", value))

    def u64(self, value: int) -> None:
        self._digest.update(struct.pack(">Q", value))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u64(len(encoded))
        self._digest.update(encoded)

    def binary64(self, value: float) -> None:
        assert math.isfinite(value)
        self._digest.update(struct.pack(">d", 0.0 if value == 0.0 else value))

    def finish(self) -> str:
        return self._digest.hexdigest()


def _rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def _translation_pair_count(mesh: tuple[int, int, int]) -> int:
    n_cells = math.prod(mesh)
    self_inverse = math.prod(2 if extent % 2 == 0 else 1 for extent in mesh)
    return (n_cells + self_inverse) // 2


def _state(
    mesh: tuple[int, int, int] = (2, 1, 1),
    n_basis: int = 3,
    *,
    calculation_identity: str = _CALCULATION_IDENTITY,
    reference_energy_per_cell: float = -5.0,
) -> object:
    n_kpoints = math.prod(mesh)
    energies = np.linspace(-1.0, 0.5, n_basis)
    occupations = np.zeros(n_basis)
    occupations[0] = 2.0
    overlap = np.eye(n_basis, dtype=np.complex128)
    fock = np.diag(energies).astype(np.complex128)
    coefficients = np.eye(n_basis, dtype=np.complex128)
    correlated = [1] + [0] * (n_basis - 1)
    virtual = [0] + [1] * (n_basis - 1)

    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = calculation_identity
    data.reference_kind = (
        core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    )
    data.normalization = (
        core._PeriodicMeanFieldNormalizationConvention
        .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS
    )
    data.periodic_dimension = 3
    data.mesh = mesh
    data.is_shift = (0, 0, 0)
    data.reciprocal_lattice = _RECIPROCAL_LATTICE
    data.converged = True
    data.symmetry_reduced_input = False
    data.symmetry_reconstructed_input = False
    data.n_basis = n_basis
    data.n_effective_orbitals = n_basis
    data.electrons_per_cell = 2
    data.reference_energy_per_cell = reference_energy_per_cell
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = (
        core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    )

    for address in product(*(range(extent) for extent in mesh)):
        fractional = np.asarray(
            [address[axis] / mesh[axis] for axis in range(3)]
        )
        data.add_kpoint(
            _RECIPROCAL_LATTICE @ fractional,
            1.0 / n_kpoints,
            overlap,
            fock,
            coefficients,
            energies,
            occupations,
            [0] * n_basis,
            correlated,
            virtual,
        )
    return core._make_periodic_restricted_mean_field_state(data)


def _dimensions(
    mesh: tuple[int, int, int] = (2, 1, 1),
    n_basis: int = 3,
    n_auxiliary: int = 5,
    auxiliary_block: int = 3,
    ao_pair_block: int = 4,
    *,
    calculation_identity: str = _CALCULATION_IDENTITY,
    allocation_identity: str = _ALLOCATION_IDENTITY,
) -> object:
    n_kpoints = math.prod(mesh)
    pair_count = _translation_pair_count(mesh)
    dimensions = core._PeriodicCorrelationStaticDimensions()
    dimensions.allocation_contract_version = (
        core._PERIODIC_CORRELATION_STREAMED_RESOURCE_CONTRACT_VERSION
    )
    dimensions.calculation_identity = calculation_identity
    dimensions.allocation_identity = allocation_identity
    dimensions.static_inventory_complete = True
    dimensions.periodic_dimension = 3
    dimensions.mesh = mesh
    dimensions.is_shift = (0, 0, 0)
    dimensions.n_kpoints = n_kpoints
    dimensions.n_basis = n_basis
    dimensions.n_effective_orbitals = n_basis
    dimensions.n_auxiliary = n_auxiliary
    dimensions.n_home_total_occupied = 1
    dimensions.n_home_occupied = 1
    dimensions.n_home_virtual = n_basis - 1
    dimensions.n_spin_channels = 1
    dimensions.triples_requested = False
    dimensions.symmetry_reduction_requested = False
    dimensions.symmetry_mapping_identity = ""
    dimensions.symmetry_representative_count = n_kpoints
    dimensions.symmetry_weight_sum = n_kpoints
    dimensions.expected_pair_candidate_count = pair_count
    dimensions.expected_triple_candidate_count = 0
    dimensions.factor_k_bra_block = 1
    dimensions.factor_k_ket_block = 1
    dimensions.factor_q_block = 1
    dimensions.factor_auxiliary_block = auxiliary_block
    dimensions.factor_ao_pair_block = ao_pair_block
    dimensions.domain_ao_support_upper_bound = n_basis
    dimensions.domain_pao_upper_bound = n_basis
    dimensions.domain_pno_upper_bound = n_basis
    dimensions.domain_local_occupied_upper_bound = 1
    dimensions.domain_local_auxiliary_upper_bound = n_auxiliary
    dimensions.pair_domain_metadata_upper_bytes = 8 + 64 * pair_count
    return dimensions


def _budget(
    *, memory: int = _HUGE_LIMIT, scratch: int = _HUGE_LIMIT
) -> object:
    budget = core._PeriodicCorrelationResourceBudget()
    budget.memory_limit_bytes = memory
    budget.scratch_limit_bytes = scratch
    budget.mpi_ranks = 1
    budget.workers_per_rank = 1
    return budget


def _reference_and_schedule(
    mesh: tuple[int, int, int] = (2, 1, 1),
    n_basis: int = 3,
    n_auxiliary: int = 5,
    auxiliary_block: int = 3,
    ao_pair_block: int = 4,
    *,
    memory: int = _HUGE_LIMIT,
    scratch: int = _HUGE_LIMIT,
    state: object | None = None,
    calculation_identity: str = _CALCULATION_IDENTITY,
    allocation_identity: str = _ALLOCATION_IDENTITY,
    reference_energy_per_cell: float = -5.0,
) -> tuple[object, object]:
    if state is None:
        state = _state(
            mesh,
            n_basis,
            calculation_identity=calculation_identity,
            reference_energy_per_cell=reference_energy_per_cell,
        )
    dimensions = _dimensions(
        mesh,
        n_basis,
        n_auxiliary,
        auxiliary_block,
        ao_pair_block,
        calculation_identity=calculation_identity,
        allocation_identity=allocation_identity,
    )
    reference = core._make_periodic_correlation_admitted_reference(
        state, dimensions, _budget(memory=memory, scratch=scratch)
    )
    schedule = core._make_periodic_correlation_factor_stream_schedule(
        reference
    )
    return reference, schedule


def _backend(*, complete: bool = True, short_range: bool = False) -> object:
    inventory = core._PeriodicCorrelationFactorBuildBackendInventory()
    inventory.complete = complete
    inventory.per_thread_short_range_fixed_workspace_bytes = (
        11 if short_range else 0
    )
    inventory.per_thread_fourier_transform_fixed_workspace_bytes = 13
    inventory.eigensolver_workspace_bytes = 17
    inventory.whitener_workspace_bytes = 19
    inventory.gemm_workspace_bytes = 23
    inventory.publisher_workspace_bytes = 29
    inventory.short_range_staging_memory_bytes = 31 if short_range else 0
    inventory.folded_source_memory_bytes = 37 if short_range else 0
    inventory.short_range_staging_scratch_bytes = 41 if short_range else 0
    inventory.folded_source_scratch_bytes = 43 if short_range else 0
    inventory.exact_extra_retained_bytes = 47
    inventory.exact_extra_control_bytes = 53
    inventory.exact_extra_scratch_bytes = 59
    return inventory


def _codec(*, complete: bool = True) -> object:
    inventory = core._PeriodicCorrelationFactorBuildCodecInventory()
    inventory.complete = complete
    inventory.fixed_header_bytes = 61
    inventory.fixed_footer_bytes = 67
    inventory.fixed_manifest_bytes = 71
    inventory.integrity_bytes_per_tile = 73
    inventory.rank_bytes_per_q = 79
    inventory.digest_bytes_per_tile = 83
    inventory.journal_header_bytes = 89
    inventory.journal_record_bytes_per_tile = 97
    inventory.checkpoint_record_bytes = 101
    inventory.checkpoint_records_per_generation = 3
    inventory.existing_generation_count = 2
    return inventory


def _config(
    *,
    producer: object | None = None,
    policy: object | None = None,
    backing: object | None = None,
    backend_complete: bool = True,
    codec_complete: bool = True,
    short_range_backend: bool = False,
) -> object:
    config = core._PeriodicCorrelationFactorBuildConfig()
    config.producer_mode = (
        core._PeriodicCorrelationFactorProducerMode
        .FULL_COULOMB_ALL_RECIPROCAL_REFERENCE
        if producer is None
        else producer
    )
    config.short_range_policy = (
        core._PeriodicCorrelationShortRangePolicy.DISABLED_ALL_RECIPROCAL
        if policy is None
        else policy
    )
    config.row_convention = (
        core._PeriodicCorrelationFactorRowConvention
        .ORIGINAL_AUXILIARY_AO_HERMITIAN_PRINCIPAL_PSEUDOINVERSE_SQUARE_ROOT
    )
    config.transfer_convention = (
        core._PeriodicCorrelationTransferRepresentativeConvention
        .CENTERED_HALF_OPEN_NEGATIVE_NYQUIST
    )
    config.backing_mode = (
        core._PeriodicCorrelationFactorBackingMode.DISK
        if backing is None
        else backing
    )
    config.publisher_mode = (
        core._PeriodicCorrelationFactorPublisherMode
        .RANDOM_ACCESS_EXACTLY_ONCE
    )
    config.ao_basis_identity_sha256 = "a" * 64
    config.auxiliary_basis_identity_sha256 = "b" * 64
    config.producer_identity_sha256 = "c" * 64
    config.backend_identity_sha256 = "d" * 64
    config.codec_identity_sha256 = "e" * 64
    config.metric_absolute_eigenvalue_threshold = 1.0e-10
    config.integral_absolute_screening_threshold = 1.0e-12
    config.reciprocal_energy_cutoff = 25.0
    config.short_range_real_space_cutoff = 0.0
    config.range_separation_omega = 0.0
    config.ao_pair_block = 4
    config.auxiliary_block = 3
    config.reciprocal_block = 7
    config.whitener_column_block = 2
    config.q_concurrency = 1
    config.native_threads = 1
    config.ao_pair_fourier_staging_required = True
    config.transpose_before_publish = True
    config.publisher_buffer_count = 2
    config.backend = _backend(
        complete=backend_complete, short_range=short_range_backend
    )
    config.codec = _codec(complete=codec_complete)
    return config


def _unravel(index: int, mesh: tuple[int, int, int]) -> tuple[int, int, int]:
    axis2 = index % mesh[2]
    rest = index // mesh[2]
    axis1 = rest % mesh[1]
    return rest // mesh[1], axis1, axis2


def _centered_transfer(
    index: int, mesh: tuple[int, int, int]
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    modular = _unravel(index, mesh)
    centered = tuple(
        value - extent if 2 * value >= extent else value
        for value, extent in zip(modular, mesh, strict=True)
    )
    doubled = tuple(2 * value for value in centered)
    wrap = tuple(
        (modular[axis] - centered[axis]) // mesh[axis]
        for axis in range(3)
    )
    return doubled, wrap


def _q_records(
    mesh: tuple[int, int, int], *, short_range: bool = False
) -> list[object]:
    records = []
    for q_index in range(math.prod(mesh)):
        record = core._PeriodicCorrelationFactorBuildQRecord()
        record.q_index = q_index
        doubled, wrap = _centered_transfer(q_index, mesh)
        record.centered_doubled_numerator = doubled
        record.centered_reciprocal_wrap = wrap
        record.base_reciprocal_vector_count = 5 + q_index
        record.tail_reciprocal_vector_count = 1 + (q_index % 3)
        record.zero_mode_excluded_count = int(q_index == 0)
        if short_range:
            record.short_range_metric_cell_count = 7 + q_index
            record.short_range_metric_task_count = 11 + q_index
            record.short_range_three_center_cell_pair_count = 13 + q_index
            record.short_range_three_center_task_count = 17 + q_index
        record.source_manifest_bytes = 101 + q_index
        records.append(record)
    return records


def _census_identity_oracle(
    reference: object, schedule: object, census: object
) -> str:
    wire = _CanonicalDigest(
        "vibeqc.periodic.correlation.factor-build.census",
        core._PERIODIC_CORRELATION_FACTOR_BUILD_ADMISSION_CONTRACT_VERSION,
    )
    wire.u32(reference.contract_version)
    wire.u32(schedule.contract_version)
    wire.u32(reference.state.digest_version)
    wire.u32(reference.dimensions.allocation_contract_version)
    wire.string(reference.state.state_identity_sha256)
    wire.string(reference.dimensions.calculation_identity)
    wire.string(reference.dimensions.allocation_identity)
    wire.string(schedule.schedule_identity_sha256)
    for component in reference.dimensions.mesh:
        wire.i32(component)
    for component in reference.dimensions.is_shift:
        wire.i32(component)

    shape = census.shape
    for field in (
        "n_kpoints",
        "n_basis",
        "n_auxiliary",
        "n_ao_pairs",
        "ao_pair_block",
        "auxiliary_block",
        "reciprocal_block",
        "whitener_column_block",
        "tile_count",
        "logical_element_count",
        "logical_bytes",
        "maximum_tile_bytes",
    ):
        wire.u64(getattr(shape, field))

    config = census.config
    for value in (
        config.producer_mode,
        config.short_range_policy,
        config.row_convention,
        config.transfer_convention,
        config.backing_mode,
        config.publisher_mode,
    ):
        wire.u32(int(value))
    for value in (
        config.ao_basis_identity_sha256,
        config.auxiliary_basis_identity_sha256,
        config.producer_identity_sha256,
        config.backend_identity_sha256,
        config.codec_identity_sha256,
    ):
        wire.string(value)
    for value in (
        config.metric_absolute_eigenvalue_threshold,
        config.integral_absolute_screening_threshold,
        config.reciprocal_energy_cutoff,
        config.short_range_real_space_cutoff,
        config.range_separation_omega,
    ):
        wire.binary64(value)
    for value in (
        config.ao_pair_block,
        config.auxiliary_block,
        config.reciprocal_block,
        config.whitener_column_block,
        config.q_concurrency,
        config.native_threads,
    ):
        wire.u64(value)
    wire.u32(int(config.ao_pair_fourier_staging_required))
    wire.u32(int(config.transpose_before_publish))
    wire.u64(config.publisher_buffer_count)

    backend = config.backend
    wire.u32(int(backend.complete))
    for field in (
        "per_thread_short_range_fixed_workspace_bytes",
        "per_thread_fourier_transform_fixed_workspace_bytes",
        "eigensolver_workspace_bytes",
        "whitener_workspace_bytes",
        "gemm_workspace_bytes",
        "publisher_workspace_bytes",
        "short_range_staging_memory_bytes",
        "folded_source_memory_bytes",
        "short_range_staging_scratch_bytes",
        "folded_source_scratch_bytes",
        "exact_extra_retained_bytes",
        "exact_extra_control_bytes",
        "exact_extra_scratch_bytes",
    ):
        wire.u64(getattr(backend, field))

    codec = config.codec
    wire.u32(int(codec.complete))
    for field in (
        "fixed_header_bytes",
        "fixed_footer_bytes",
        "fixed_manifest_bytes",
        "integrity_bytes_per_tile",
        "rank_bytes_per_q",
        "digest_bytes_per_tile",
        "journal_header_bytes",
        "journal_record_bytes_per_tile",
        "checkpoint_record_bytes",
        "checkpoint_records_per_generation",
        "existing_generation_count",
    ):
        wire.u64(getattr(codec, field))

    wire.u64(len(census.q_records))
    for record in census.q_records:
        wire.u64(record.q_index)
        for component in record.centered_doubled_numerator:
            wire.i32(component)
        for component in record.centered_reciprocal_wrap:
            wire.i32(component)
        for field in (
            "base_reciprocal_vector_count",
            "tail_reciprocal_vector_count",
            "zero_mode_excluded_count",
            "short_range_metric_cell_count",
            "short_range_metric_task_count",
            "short_range_three_center_cell_pair_count",
            "short_range_three_center_task_count",
            "source_manifest_bytes",
        ):
            wire.u64(getattr(record, field))
    return wire.finish()


def _plan_identity_oracle(reference: object, plan: object) -> str:
    wire = _CanonicalDigest(
        "vibeqc.periodic.correlation.factor-build.plan",
        core._PERIODIC_CORRELATION_FACTOR_BUILD_ADMISSION_CONTRACT_VERSION,
    )
    wire.string(plan.census_identity_sha256)
    wire.u64(reference.budget.memory_limit_bytes)
    wire.u64(reference.budget.scratch_limit_bytes)
    wire.u64(reference.budget.mpi_ranks)
    wire.u64(reference.budget.workers_per_rank)
    wire.u32(int(plan.admission))
    wire.string(plan.calculation_identity)
    wire.string(plan.allocation_identity)
    wire.string(plan.schedule_identity_sha256)
    for field in (
        "n_kpoints",
        "n_basis",
        "n_auxiliary",
        "n_ao_pairs",
        "ao_pair_block",
        "auxiliary_block",
        "reciprocal_block",
        "whitener_column_block",
        "tile_count",
        "logical_element_count",
        "logical_bytes",
        "maximum_tile_bytes",
    ):
        wire.u64(getattr(plan.shape, field))
    for field in (
        "total_base_reciprocal_vectors",
        "total_tail_reciprocal_vectors",
        "maximum_reciprocal_vectors_per_q",
        "total_short_range_metric_tasks",
        "total_short_range_three_center_tasks",
        "maximum_short_range_metric_tasks_per_q",
        "maximum_short_range_three_center_tasks_per_q",
        "logical_factor_bytes",
        "encoded_generation_bytes",
        "modeled_peak_memory_bytes",
        "required_memory_bytes",
        "modeled_scratch_bytes",
        "required_scratch_bytes",
    ):
        wire.u64(getattr(plan, field))
    for field in (
        "admitted_baseline_retained_bytes",
        "in_memory_output_bytes",
        "source_manifest_total_bytes",
        "source_manifest_maximum_bytes",
        "publisher_bitmap_bytes",
        "publisher_digest_table_bytes",
        "publisher_control_bytes",
        "folded_source_retained_bytes",
        "reciprocal_g_panel_bytes",
        "auxiliary_fourier_double_panel_bytes",
        "ao_pair_fourier_panel_bytes",
        "ao_pair_fourier_staging_bytes",
        "metric_matrix_bytes",
        "metric_eigenvector_bytes",
        "metric_eigenvalue_bytes",
        "whitener_matrix_bytes",
        "whitener_column_panel_bytes",
        "raw_three_center_panel_bytes",
        "whitening_input_panel_bytes",
        "whitening_output_panel_bytes",
        "transpose_staging_bytes",
        "publisher_buffer_bytes",
        "threaded_short_range_workspace_bytes",
        "threaded_fourier_workspace_bytes",
        "encoded_generation_bytes",
        "journal_bytes",
        "checkpoint_bytes",
        "disk_generation_bytes",
    ):
        wire.u64(getattr(plan.components, field))
    wire.u64(len(plan.phases))
    for phase in plan.phases:
        wire.u32(int(phase.phase))
        wire.u64(phase.retained_bytes)
        wire.u64(phase.phase_extra_bytes)
        wire.u64(phase.peak_memory_bytes)
    return wire.finish()


def _census(
    reference: object,
    schedule: object,
    *,
    config: object | None = None,
    q_records: list[object] | None = None,
) -> object:
    if config is None:
        config = _config()
    if q_records is None:
        q_records = _q_records(tuple(schedule.mesh))
    return core._make_periodic_correlation_factor_build_census(
        reference, schedule, config, q_records
    )


def _rsgdf_config(
    *,
    policy: object | None = None,
    backing: object | None = None,
) -> object:
    selected_policy = policy
    if selected_policy is None:
        selected_policy = (
            core._PeriodicCorrelationShortRangePolicy
            .DOUBLE_CELL_BVK_BLOCKED_TRANSFORM
        )
    config = _config(
        producer=core._PeriodicCorrelationFactorProducerMode.RANGE_SEPARATED_GDF,
        policy=selected_policy,
        backing=backing,
        short_range_backend=True,
    )
    if selected_policy == (
        core._PeriodicCorrelationShortRangePolicy
        .DOUBLE_CELL_RECOMPUTE_PER_K_PAIR_REFERENCE
    ):
        backend = config.backend
        backend.folded_source_memory_bytes = 0
        backend.folded_source_scratch_bytes = 0
        config.backend = backend
    config.short_range_real_space_cutoff = 8.0
    config.range_separation_omega = 0.4
    return config


def test_contract_enums_and_default_conventions_are_fixed() -> None:
    assert core._PERIODIC_CORRELATION_FACTOR_BUILD_ADMISSION_CONTRACT_VERSION == 1
    assert int(
        core._PeriodicCorrelationFactorProducerMode
        .FULL_COULOMB_ALL_RECIPROCAL_REFERENCE
    ) == 0
    assert int(
        core._PeriodicCorrelationFactorProducerMode.RANGE_SEPARATED_GDF
    ) == 1
    assert int(
        core._PeriodicCorrelationShortRangePolicy.DISABLED_ALL_RECIPROCAL
    ) == 0
    assert int(
        core._PeriodicCorrelationFactorBackingMode.DISK
    ) == 0
    assert int(
        core._PeriodicCorrelationFactorPublisherMode.RANDOM_ACCESS_EXACTLY_ONCE
    ) == 0

    config = core._PeriodicCorrelationFactorBuildConfig()
    assert config.row_convention == (
        core._PeriodicCorrelationFactorRowConvention
        .ORIGINAL_AUXILIARY_AO_HERMITIAN_PRINCIPAL_PSEUDOINVERSE_SQUARE_ROOT
    )
    assert config.transfer_convention == (
        core._PeriodicCorrelationTransferRepresentativeConvention
        .CENTERED_HALF_OPEN_NEGATIVE_NYQUIST
    )


def test_census_pins_centered_half_open_q_records_and_negative_nyquist() -> None:
    mesh = (4, 2, 1)
    reference, schedule = _reference_and_schedule(mesh=mesh)
    census = _census(reference, schedule, q_records=_q_records(mesh))

    assert [
        tuple(record.centered_doubled_numerator)
        for record in census.q_records
    ] == [
        (0, 0, 0),
        (0, -2, 0),
        (2, 0, 0),
        (2, -2, 0),
        (-4, 0, 0),
        (-4, -2, 0),
        (-2, 0, 0),
        (-2, -2, 0),
    ]
    assert [tuple(record.centered_reciprocal_wrap) for record in census.q_records] == [
        (0, 0, 0),
        (0, 1, 0),
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    ]
    assert [record.zero_mode_excluded_count for record in census.q_records] == [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


def test_odd_mesh_centers_upper_half_without_a_nyquist_point() -> None:
    mesh = (3, 1, 1)
    reference, schedule = _reference_and_schedule(mesh=mesh)
    census = _census(reference, schedule, q_records=_q_records(mesh))

    assert [
        tuple(record.centered_doubled_numerator)
        for record in census.q_records
    ] == [(0, 0, 0), (2, 0, 0), (-2, 0, 0)]
    assert [
        tuple(record.centered_reciprocal_wrap)
        for record in census.q_records
    ] == [(0, 0, 0), (0, 0, 0), (1, 0, 0)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "row_convention",
            core._PeriodicCorrelationFactorRowConvention
            .COMPACT_METRIC_EIGENMODES_UNSUPPORTED,
        ),
        (
            "transfer_convention",
            core._PeriodicCorrelationTransferRepresentativeConvention
            .UNSPECIFIED_UNSUPPORTED,
        ),
        (
            "publisher_mode",
            core._PeriodicCorrelationFactorPublisherMode
            .SEQUENTIAL_APPEND_UNSUPPORTED,
        ),
    ],
)
def test_census_rejects_unsupported_semantic_conventions(
    field: str, value: object
) -> None:
    reference, schedule = _reference_and_schedule()
    config = _config()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        _census(reference, schedule, config=config)


def test_census_rejects_noncanonical_centered_q_record() -> None:
    reference, schedule = _reference_and_schedule()
    records = _q_records((2, 1, 1))
    records[1].centered_doubled_numerator = (2, 0, 0)
    with pytest.raises(ValueError):
        _census(reference, schedule, q_records=records)

    records = _q_records((2, 1, 1))
    records[0].zero_mode_excluded_count = 0
    with pytest.raises(ValueError):
        _census(reference, schedule, q_records=records)


def test_census_identity_is_deterministic_and_sensitive_to_one_field() -> None:
    reference, schedule = _reference_and_schedule()
    first = _census(reference, schedule)
    second = _census(reference, schedule)
    changed_records = _q_records((2, 1, 1))
    changed_records[1].source_manifest_bytes += 1
    changed = _census(reference, schedule, q_records=changed_records)

    assert first.census_identity_sha256 == second.census_identity_sha256
    assert first.census_identity_sha256 == _census_identity_oracle(
        reference, schedule, first
    )
    assert first.census_identity_sha256 == _FULL_RECIPROCAL_CENSUS_KAT
    assert len(first.census_identity_sha256) == 64
    assert first.census_identity_sha256 != changed.census_identity_sha256


@pytest.mark.parametrize(
    ("backend_complete", "codec_complete"), [(False, True), (True, False)]
)
def test_census_requires_complete_backend_and_codec_inventory(
    backend_complete: bool, codec_complete: bool
) -> None:
    reference, schedule = _reference_and_schedule()
    config = _config(
        backend_complete=backend_complete, codec_complete=codec_complete
    )
    with pytest.raises(ValueError):
        _census(reference, schedule, config=config)


def test_all_reciprocal_reference_requires_zero_short_range_inventory() -> None:
    reference, schedule = _reference_and_schedule()
    census = _census(reference, schedule)
    assert all(
        record.short_range_metric_cell_count == 0
        and record.short_range_metric_task_count == 0
        and record.short_range_three_center_cell_pair_count == 0
        and record.short_range_three_center_task_count == 0
        for record in census.q_records
    )

    with pytest.raises(ValueError):
        _census(
            reference,
            schedule,
            q_records=_q_records((2, 1, 1), short_range=True),
        )

    with pytest.raises(ValueError):
        _census(
            reference,
            schedule,
            config=_config(short_range_backend=True),
        )


def test_rsgdf_requires_supported_double_cell_policy_and_nonzero_counts() -> None:
    reference, schedule = _reference_and_schedule()
    supported = (
        core._PeriodicCorrelationShortRangePolicy
        .DOUBLE_CELL_RECOMPUTE_PER_K_PAIR_REFERENCE,
        core._PeriodicCorrelationShortRangePolicy
        .DOUBLE_CELL_BVK_BLOCKED_TRANSFORM,
    )
    censuses = [
        _census(
            reference,
            schedule,
            config=_rsgdf_config(policy=policy),
            q_records=_q_records((2, 1, 1), short_range=True),
        )
        for policy in supported
    ]
    census = censuses[-1]
    assert census.config.producer_mode == (
        core._PeriodicCorrelationFactorProducerMode.RANGE_SEPARATED_GDF
    )
    assert all(
        record.short_range_three_center_cell_pair_count > 0
        for record in census.q_records
    )

    for policy in (
        core._PeriodicCorrelationShortRangePolicy.DISABLED_ALL_RECIPROCAL,
        core._PeriodicCorrelationShortRangePolicy.UNSUPPORTED_SINGLE_CELL_GAMMA_SUM,
    ):
        with pytest.raises(ValueError):
            _census(
                reference,
                schedule,
                config=_rsgdf_config(policy=policy),
                q_records=_q_records((2, 1, 1), short_range=True),
            )

    with pytest.raises(ValueError):
        _census(
            reference,
            schedule,
            config=_rsgdf_config(),
            q_records=_q_records((2, 1, 1)),
        )


def test_census_owns_state_and_copied_input_inventories() -> None:
    reference, schedule = _reference_and_schedule()
    config = _config()
    records = _q_records((2, 1, 1))
    census = _census(reference, schedule, config=config, q_records=records)
    state = census.state

    config.reciprocal_block = 1
    records[0].source_manifest_bytes = 999
    del reference, schedule
    gc.collect()

    assert census.config.reciprocal_block == 7
    assert census.q_records[0].source_manifest_bytes == 101
    assert state.fock(0).shape == (3, 3)


def test_cross_state_and_schedule_provenance_is_rejected() -> None:
    reference_a, schedule_a = _reference_and_schedule()
    reference_b, schedule_b = _reference_and_schedule(
        reference_energy_per_cell=-5.25
    )
    census_a = _census(reference_a, schedule_a)

    with pytest.raises(ValueError):
        _census(reference_a, schedule_b)
    with pytest.raises(ValueError):
        _census(reference_b, schedule_a)
    with pytest.raises(ValueError):
        core._plan_periodic_correlation_factor_build(
            reference_a, schedule_b, census_a
        )
    with pytest.raises(ValueError):
        core._plan_periodic_correlation_factor_build(
            reference_b, schedule_a, census_a
        )


def test_same_state_still_rejects_cross_allocation_schedule() -> None:
    state = _state()
    reference_a, schedule_a = _reference_and_schedule(state=state)
    reference_b, schedule_b = _reference_and_schedule(
        state=state, allocation_identity="3" * 64
    )
    census_a = _census(reference_a, schedule_a)

    with pytest.raises((ValueError, RuntimeError)):
        _census(reference_a, schedule_b)
    with pytest.raises((ValueError, RuntimeError)):
        core._plan_periodic_correlation_factor_build(
            reference_b, schedule_b, census_a
        )


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    return (value * numerator + denominator - 1) // denominator


def test_rsgdf_plan_matches_independent_component_phase_and_cap_oracle() -> None:
    reference, schedule = _reference_and_schedule()
    config = _rsgdf_config()
    records = _q_records((2, 1, 1), short_range=True)
    census = _census(
        reference, schedule, config=config, q_records=records
    )
    plan = core._plan_periodic_correlation_factor_build(
        reference, schedule, census
    )
    shape = schedule.shape

    tile_count = shape.tile_count
    n_kpoints = shape.n_kpoints
    n_auxiliary = shape.n_auxiliary
    reciprocal_block = config.reciprocal_block
    whitener_block = config.whitener_column_block
    ao_pair_block = shape.ao_pair_block
    maximum_tile_bytes = shape.maximum_tile_bytes
    backend = config.backend
    codec = config.codec

    baseline = (
        reference.dimensions.external_bytes
        + reference.dimensions.shared_bytes
        + reference.dimensions.per_rank_bytes
        + backend.exact_extra_retained_bytes
    )
    bitmap = (tile_count + 7) // 8
    digest_table = tile_count * codec.digest_bytes_per_tile
    publisher_control = (
        bitmap + digest_table + backend.exact_extra_control_bytes
    )
    folded_source = backend.folded_source_memory_bytes
    g_panel = 40 * reciprocal_block
    auxiliary_fourier = 2 * 16 * n_auxiliary * reciprocal_block
    ao_pair_fourier = 16 * ao_pair_block * reciprocal_block
    ao_pair_fourier_staging = ao_pair_fourier
    metric = 16 * n_auxiliary**2
    eigenvectors = metric
    eigenvalues = 8 * n_auxiliary
    whitener = metric
    whitener_columns = 16 * n_auxiliary * whitener_block
    raw_panel = 16 * n_auxiliary * ao_pair_block
    whitening_input = raw_panel
    whitening_output = raw_panel
    transpose = whitening_output
    publisher_buffers = config.publisher_buffer_count * maximum_tile_bytes
    threaded_short_range = (
        config.native_threads
        * backend.per_thread_short_range_fixed_workspace_bytes
    )
    threaded_fourier = (
        config.native_threads
        * backend.per_thread_fourier_transform_fixed_workspace_bytes
    )
    encoded = (
        shape.logical_bytes
        + codec.fixed_header_bytes
        + codec.fixed_footer_bytes
        + codec.fixed_manifest_bytes
        + tile_count * codec.integrity_bytes_per_tile
        + n_kpoints * codec.rank_bytes_per_q
    )
    generations = codec.existing_generation_count + 1
    journal = (
        codec.journal_header_bytes
        + tile_count * codec.journal_record_bytes_per_tile
    )
    checkpoint = (
        codec.checkpoint_record_bytes
        * codec.checkpoint_records_per_generation
        * generations
    )
    disk_generations = encoded * generations

    expected_components = {
        "admitted_baseline_retained_bytes": baseline,
        "in_memory_output_bytes": 0,
        "source_manifest_total_bytes": sum(
            record.source_manifest_bytes for record in records
        ),
        "source_manifest_maximum_bytes": max(
            record.source_manifest_bytes for record in records
        ),
        "publisher_bitmap_bytes": bitmap,
        "publisher_digest_table_bytes": digest_table,
        "publisher_control_bytes": publisher_control,
        "folded_source_retained_bytes": folded_source,
        "reciprocal_g_panel_bytes": g_panel,
        "auxiliary_fourier_double_panel_bytes": auxiliary_fourier,
        "ao_pair_fourier_panel_bytes": ao_pair_fourier,
        "ao_pair_fourier_staging_bytes": ao_pair_fourier_staging,
        "metric_matrix_bytes": metric,
        "metric_eigenvector_bytes": eigenvectors,
        "metric_eigenvalue_bytes": eigenvalues,
        "whitener_matrix_bytes": whitener,
        "whitener_column_panel_bytes": whitener_columns,
        "raw_three_center_panel_bytes": raw_panel,
        "whitening_input_panel_bytes": whitening_input,
        "whitening_output_panel_bytes": whitening_output,
        "transpose_staging_bytes": transpose,
        "publisher_buffer_bytes": publisher_buffers,
        "threaded_short_range_workspace_bytes": threaded_short_range,
        "threaded_fourier_workspace_bytes": threaded_fourier,
        "encoded_generation_bytes": encoded,
        "journal_bytes": journal,
        "checkpoint_bytes": checkpoint,
        "disk_generation_bytes": disk_generations,
    }
    assert {
        field: getattr(plan.components, field)
        for field in expected_components
    } == expected_components

    retained = baseline + publisher_control + folded_source
    manifest = max(record.source_manifest_bytes for record in records)
    expected_phase_extras = [
        (
            core._PeriodicCorrelationFactorBuildPhase.RECIPROCAL_METRIC,
            manifest
            + metric
            + g_panel
            + auxiliary_fourier
            + threaded_fourier,
        ),
        (
            core._PeriodicCorrelationFactorBuildPhase.SHORT_RANGE_METRIC,
            manifest
            + metric
            + threaded_short_range
            + backend.short_range_staging_memory_bytes,
        ),
        (
            core._PeriodicCorrelationFactorBuildPhase.METRIC_FACTORIZATION,
            manifest
            + metric
            + eigenvectors
            + eigenvalues
            + backend.eigensolver_workspace_bytes,
        ),
        (
            core._PeriodicCorrelationFactorBuildPhase.METRIC_WHITENER,
            manifest
            + eigenvectors
            + eigenvalues
            + whitener
            + 2 * whitener_columns
            + backend.whitener_workspace_bytes,
        ),
        (
            core._PeriodicCorrelationFactorBuildPhase.RECIPROCAL_THREE_CENTER,
            manifest
            + whitener
            + raw_panel
            + g_panel
            + auxiliary_fourier
            + ao_pair_fourier
            + ao_pair_fourier_staging
            + threaded_fourier,
        ),
        (
            core._PeriodicCorrelationFactorBuildPhase.SHORT_RANGE_THREE_CENTER,
            manifest
            + whitener
            + raw_panel
            + threaded_short_range
            + backend.short_range_staging_memory_bytes,
        ),
        (
            core._PeriodicCorrelationFactorBuildPhase.WHITENING,
            manifest
            + whitener
            + whitener_columns
            + whitening_input
            + whitening_output
            + backend.gemm_workspace_bytes,
        ),
        (
            core._PeriodicCorrelationFactorBuildPhase.TRANSPOSE,
            manifest + whitener + whitening_output + transpose,
        ),
        (
            core._PeriodicCorrelationFactorBuildPhase.PUBLISH,
            manifest
            + whitener
            + whitening_output
            + transpose
            + publisher_buffers
            + backend.publisher_workspace_bytes,
        ),
    ]
    assert [
        (
            phase.phase,
            phase.retained_bytes,
            phase.phase_extra_bytes,
            phase.peak_memory_bytes,
        )
        for phase in plan.phases
    ] == [
        (phase, retained, extra, retained + extra)
        for phase, extra in expected_phase_extras
    ]

    modeled_memory = retained + max(
        extra for _, extra in expected_phase_extras
    )
    modeled_scratch = (
        backend.short_range_staging_scratch_bytes
        + backend.folded_source_scratch_bytes
        + backend.exact_extra_scratch_bytes
        + disk_generations
        + journal
        + checkpoint
    )
    assert plan.total_base_reciprocal_vectors == sum(
        record.base_reciprocal_vector_count for record in records
    )
    assert plan.total_tail_reciprocal_vectors == sum(
        record.tail_reciprocal_vector_count for record in records
    )
    assert plan.maximum_reciprocal_vectors_per_q == max(
        record.base_reciprocal_vector_count
        + record.tail_reciprocal_vector_count
        for record in records
    )
    assert plan.total_short_range_metric_tasks == sum(
        record.short_range_metric_task_count for record in records
    )
    assert plan.total_short_range_three_center_tasks == sum(
        record.short_range_three_center_task_count for record in records
    )
    assert plan.maximum_short_range_metric_tasks_per_q == max(
        record.short_range_metric_task_count for record in records
    )
    assert plan.maximum_short_range_three_center_tasks_per_q == max(
        record.short_range_three_center_task_count for record in records
    )
    assert plan.logical_factor_bytes == shape.logical_bytes
    assert plan.encoded_generation_bytes == encoded
    assert plan.modeled_peak_memory_bytes == modeled_memory
    assert plan.required_memory_bytes == _ceil_ratio(modeled_memory, 3, 2)
    assert plan.modeled_scratch_bytes == modeled_scratch
    assert plan.required_scratch_bytes == _ceil_ratio(modeled_scratch, 5, 4)
    assert plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    assert census.census_identity_sha256 == _RSGDF_CENSUS_KAT
    assert plan.plan_identity_sha256 == _plan_identity_oracle(reference, plan)
    assert plan.plan_identity_sha256 == _RSGDF_PLAN_KAT


def test_disk_output_is_not_charged_to_ram_but_memory_output_is() -> None:
    reference, schedule = _reference_and_schedule()
    disk_census = _census(reference, schedule, config=_config())
    disk = core._plan_periodic_correlation_factor_build(
        reference, schedule, disk_census
    )

    memory_config = _config(
        backing=core._PeriodicCorrelationFactorBackingMode.MEMORY
    )
    memory_census = _census(reference, schedule, config=memory_config)
    memory = core._plan_periodic_correlation_factor_build(
        reference, schedule, memory_census
    )

    logical = schedule.shape.logical_bytes
    assert disk.components.in_memory_output_bytes == 0
    assert disk.components.disk_generation_bytes > logical
    assert memory.components.in_memory_output_bytes == logical
    assert memory.components.disk_generation_bytes == 0
    assert memory.modeled_peak_memory_bytes - disk.modeled_peak_memory_bytes == (
        logical
    )
    assert all(
        memory_phase.retained_bytes - disk_phase.retained_bytes == logical
        for disk_phase, memory_phase in zip(
            disk.phases, memory.phases, strict=True
        )
    )
    assert memory.modeled_scratch_bytes == (
        memory_config.backend.exact_extra_scratch_bytes
    )


def _factor_plan_with_budget(memory: int, scratch: int) -> object:
    reference, schedule = _reference_and_schedule(
        memory=memory, scratch=scratch
    )
    census = _census(reference, schedule)
    return core._plan_periodic_correlation_factor_build(
        reference, schedule, census
    )


def test_memory_and_scratch_budget_boundaries_are_exact() -> None:
    generous = _factor_plan_with_budget(_HUGE_LIMIT, _HUGE_LIMIT)

    exact_memory = _factor_plan_with_budget(
        generous.required_memory_bytes, _HUGE_LIMIT
    )
    short_memory = _factor_plan_with_budget(
        generous.required_memory_bytes - 1, _HUGE_LIMIT
    )
    assert exact_memory.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    assert short_memory.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.MEMORY_EXCEEDED
    )

    exact_scratch = _factor_plan_with_budget(
        _HUGE_LIMIT, generous.required_scratch_bytes
    )
    short_scratch = _factor_plan_with_budget(
        _HUGE_LIMIT, generous.required_scratch_bytes - 1
    )
    assert exact_scratch.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    assert short_scratch.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.SCRATCH_EXCEEDED
    )


def test_random_access_publisher_bitmap_rounds_up_to_a_full_byte() -> None:
    reference, schedule = _reference_and_schedule(
        mesh=(1, 1, 1),
        n_basis=3,
        n_auxiliary=5,
        auxiliary_block=3,
        ao_pair_block=5,
    )
    config = _config()
    config.ao_pair_block = 5
    census = _census(
        reference,
        schedule,
        config=config,
        q_records=_q_records((1, 1, 1)),
    )
    plan = core._plan_periodic_correlation_factor_build(
        reference, schedule, census
    )

    assert schedule.shape.tile_count == 4
    assert plan.components.publisher_bitmap_bytes == 1
    assert plan.components.publisher_bitmap_bytes == (
        schedule.shape.tile_count + 7
    ) // 8


def test_checked_plan_overflow_fails_closed_without_an_identity() -> None:
    reference, schedule = _reference_and_schedule()
    records = _q_records((2, 1, 1))
    records[0].source_manifest_bytes = (1 << 64) - 1
    records[1].source_manifest_bytes = 1
    census = _census(reference, schedule, q_records=records)
    plan = core._plan_periodic_correlation_factor_build(
        reference, schedule, census
    )

    assert plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ARITHMETIC_OVERFLOW
    )
    assert plan.plan_identity_sha256 == ""
    assert "overflow" in plan.failure_detail


@pytest.mark.parametrize(
    ("label", "mesh", "n_basis", "n_auxiliary"),
    [
        ("MgO", (8, 8, 8), 37, 111),
        ("diamond", (8, 8, 8), 36, 108),
        ("primitive corundum", (6, 6, 6), 196, 588),
    ],
)
def test_target_shapes_are_scalar_only_and_bounded_rss(
    label: str,
    mesh: tuple[int, int, int],
    n_basis: int,
    n_auxiliary: int,
) -> None:
    del label
    before = _rss_bytes()
    shape = core._estimate_periodic_correlation_factor_stream_shape(
        mesh, n_basis, n_auxiliary, 64, 256
    )
    after = _rss_bytes()

    n_kpoints = math.prod(mesh)
    assert shape.n_kpoints == n_kpoints
    assert shape.n_ao_pairs == n_basis**2
    assert shape.logical_element_count == (
        n_kpoints**2 * n_auxiliary * n_basis**2
    )
    assert shape.logical_bytes == 16 * shape.logical_element_count
    assert shape.maximum_tile_bytes == 16 * 64 * 256
    assert shape.two_buffer_workspace_bytes == 2 * shape.maximum_tile_bytes
    assert after - before < 16 * 1024**2
