"""Bounded diagnostics for the native one-q reciprocal metric.

Every successful numerical fixture is deliberately tiny.  The raw-metric
cases use at most four auxiliary functions, three k points and six accepted
reciprocal vectors.  They construct no three-center integral, SCF calculation,
factor tensor, target-system mesh, or target chemistry.
"""

from __future__ import annotations

import cmath
import gc
import hashlib
import math
import struct
from decimal import Decimal, localcontext
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


_ROOT = Path(__file__).resolve().parents[1]
_CALCULATION_IDENTITY = "1" * 64
_ALLOCATION_IDENTITY = "2" * 64
_HUGE_LIMIT = 1 << 50
_PRODUCER_IDENTITY_KAT = (
    "45d45e8026d486e65651ec27ec3d93e40c1792e3e459fb0bc49ec7ff9c343dae"
)
_GAMMA_SOURCE_IDENTITY_KAT = (
    "853f6163e386befb39ab57294eb8a107a79806c8d44852618168ea8e157e5f9b"
)
_PAYLOAD_DOMAIN = "vibeqc.periodic.correlation.reciprocal-metric.payload"


class _CanonicalDigest:
    """Independent implementation of the documented big-endian wire."""

    def __init__(self, domain: str, version: int) -> None:
        self._digest = hashlib.sha256()
        self.wire_bytes = 0
        self.string(domain)
        self.u32(version)

    def _update(self, encoded: bytes) -> None:
        self._digest.update(encoded)
        self.wire_bytes += len(encoded)

    def u32(self, value: int) -> None:
        self._update(struct.pack(">I", value))

    def i32(self, value: int) -> None:
        self._update(struct.pack(">i", value))

    def u64(self, value: int) -> None:
        self._update(struct.pack(">Q", value))

    def i64(self, value: int) -> None:
        self._update(struct.pack(">q", value))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u64(len(encoded))
        self._update(encoded)

    def binary64(self, value: float) -> None:
        assert math.isfinite(value)
        self._update(struct.pack(">d", 0.0 if value == 0.0 else value))

    def complex128(self, value: complex) -> None:
        self.binary64(float(value.real))
        self.binary64(float(value.imag))

    def finish(self) -> str:
        return self._digest.hexdigest()


def _binary64_fma(left: float, right: float, addend: float) -> float:
    """Use math.fma, with an exact-decimal fallback for Python 3.11/3.12."""
    fma = getattr(math, "fma", None)
    if fma is not None:
        return fma(left, right, addend)
    with localcontext() as context:
        context.prec = 2500
        exact = (
            Decimal.from_float(left) * Decimal.from_float(right)
            + Decimal.from_float(addend)
        )
    return float(exact)


def _auxiliary_basis(
    *,
    exponent: float = 0.75,
    coefficient: float = 0.625,
    shell_count: int = 1,
    name: str = "metric-diagnostic-aux",
) -> object:
    center = [0.0, 0.0, 0.0]
    molecule = core.Molecule([core.Atom(2, center)], 0, 1)
    shells = [
        core.ShellInfo(
            0,
            0,
            True,
            [exponent + 0.1 * shell_index],
            [coefficient],
            center,
        )
        for shell_index in range(shell_count)
    ]
    return core.BasisSet(
        molecule,
        shells,
        name,
        coefficients_pre_normalized=True,
    )

def _two_s_metric_basis(
    *,
    second_exponent: float = 1.0,
    second_center_x: float = 0.4,
    name: str = "raw-metric-two-s",
) -> object:
    """Two unnormalised primitive s functions with an analytic transform."""
    centers = ([0.0, 0.0, 0.0], [second_center_x, 0.0, 0.0])
    molecule = core.Molecule(
        [core.Atom(1, list(center)) for center in centers], 0, 1
    )
    shells = [
        core.ShellInfo(0, 0, True, [0.5], [1.0], centers[0]),
        core.ShellInfo(
            1, 0, True, [second_exponent], [0.75], centers[1]
        ),
    ]
    return core.BasisSet(
        molecule, shells, name, coefficients_pre_normalized=True
    )


def _many_s_metric_basis(shell_count: int) -> object:
    center = [0.0, 0.0, 0.0]
    molecule = core.Molecule([core.Atom(2, center)], 0, 1)
    shells = [
        core.ShellInfo(
            0,
            0,
            True,
            [0.5 + 0.001 * shell_index],
            [1.0],
            center,
        )
        for shell_index in range(shell_count)
    ]
    return core.BasisSet(
        molecule,
        shells,
        "raw-metric-cap-basis",
        coefficients_pre_normalized=True,
    )


def _cartesian_p_metric_basis() -> object:
    """A source-valid basis that the numerical Fourier leaf must reject."""
    center = [0.0, 0.0, 0.0]
    molecule = core.Molecule([core.Atom(2, center)], 0, 1)
    shell = core.ShellInfo(0, 1, False, [0.75], [1.0], center)
    return core.BasisSet(
        molecule,
        [shell],
        "raw-metric-cartesian-p",
        coefficients_pre_normalized=True,
    )


def _mixed_s_p_metric_basis() -> object:
    """Separated pure-spherical s/p shells for composition coverage."""
    centers = ([0.0, 0.0, 0.0], [0.35, -0.2, 0.1])
    molecule = core.Molecule(
        [core.Atom(1, list(center)) for center in centers], 0, 1
    )
    shells = [
        core.ShellInfo(0, 0, True, [0.6], [0.8], centers[0]),
        core.ShellInfo(1, 1, True, [0.9], [0.7], centers[1]),
    ]
    return core.BasisSet(
        molecule,
        shells,
        "raw-metric-mixed-s-p",
        coefficients_pre_normalized=True,
    )


def _translation_pair_count(mesh: tuple[int, int, int]) -> int:
    cell_count = math.prod(mesh)
    self_inverse = math.prod(
        2 if extent % 2 == 0 else 1 for extent in mesh
    )
    return (cell_count + self_inverse) // 2


def _state(
    mesh: tuple[int, int, int],
    reciprocal_lattice: np.ndarray,
    *,
    periodic_dimension: int = 3,
    n_basis: int = 2,
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
    data.calculation_identity = _CALCULATION_IDENTITY
    data.reference_kind = (
        core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    )
    data.normalization = (
        core._PeriodicMeanFieldNormalizationConvention
        .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS
    )
    data.periodic_dimension = periodic_dimension
    data.mesh = mesh
    data.is_shift = (0, 0, 0)
    data.reciprocal_lattice = reciprocal_lattice
    data.converged = True
    data.symmetry_reduced_input = False
    data.symmetry_reconstructed_input = False
    data.n_basis = n_basis
    data.n_effective_orbitals = n_basis
    data.electrons_per_cell = 2
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = (
        core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    )

    for address in product(*(range(extent) for extent in mesh)):
        fractional = np.asarray(
            [address[axis] / mesh[axis] for axis in range(3)]
        )
        data.add_kpoint(
            reciprocal_lattice @ fractional,
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
    mesh: tuple[int, int, int],
    *,
    periodic_dimension: int = 3,
    n_basis: int = 2,
    n_auxiliary: int = 1,
) -> object:
    n_kpoints = math.prod(mesh)
    pair_count = _translation_pair_count(mesh)
    dimensions = core._PeriodicCorrelationStaticDimensions()
    dimensions.allocation_contract_version = (
        core._PERIODIC_CORRELATION_STREAMED_RESOURCE_CONTRACT_VERSION
    )
    dimensions.calculation_identity = _CALCULATION_IDENTITY
    dimensions.allocation_identity = _ALLOCATION_IDENTITY
    dimensions.static_inventory_complete = True
    dimensions.periodic_dimension = periodic_dimension
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
    dimensions.factor_auxiliary_block = n_auxiliary
    dimensions.factor_ao_pair_block = n_basis * n_basis
    dimensions.domain_ao_support_upper_bound = n_basis
    dimensions.domain_pao_upper_bound = n_basis
    dimensions.domain_pno_upper_bound = n_basis
    dimensions.domain_local_occupied_upper_bound = 1
    dimensions.domain_local_auxiliary_upper_bound = n_auxiliary
    dimensions.pair_domain_metadata_upper_bytes = 8 + 64 * pair_count
    return dimensions


def _reference_and_schedule(
    mesh: tuple[int, int, int],
    reciprocal_lattice: np.ndarray,
    *,
    periodic_dimension: int = 3,
    n_auxiliary: int = 1,
    memory_limit: int = _HUGE_LIMIT,
    scratch_limit: int = _HUGE_LIMIT,
) -> tuple[object, object, object]:
    state = _state(
        mesh,
        reciprocal_lattice,
        periodic_dimension=periodic_dimension,
    )
    dimensions = _dimensions(
        mesh,
        periodic_dimension=periodic_dimension,
        n_auxiliary=n_auxiliary,
    )
    budget = core._PeriodicCorrelationResourceBudget()
    budget.memory_limit_bytes = memory_limit
    budget.scratch_limit_bytes = scratch_limit
    budget.mpi_ranks = 1
    budget.workers_per_rank = 1
    reference = core._make_periodic_correlation_admitted_reference(
        state, dimensions, budget
    )
    schedule = core._make_periodic_correlation_factor_stream_schedule(
        reference
    )
    return state, reference, schedule


def _config(auxiliary_basis: object, cutoff: float = 0.5) -> object:
    config = core._PeriodicCorrelationFactorBuildConfig()
    config.producer_mode = (
        core._PeriodicCorrelationFactorProducerMode
        .FULL_COULOMB_ALL_RECIPROCAL_REFERENCE
    )
    config.short_range_policy = (
        core._PeriodicCorrelationShortRangePolicy.DISABLED_ALL_RECIPROCAL
    )
    config.transfer_convention = (
        core._PeriodicCorrelationTransferRepresentativeConvention
        .CENTERED_HALF_OPEN_NEGATIVE_NYQUIST
    )
    config.auxiliary_basis_identity_sha256 = (
        core._auxiliary_basis_content_identity_sha256(auxiliary_basis)
    )
    config.producer_identity_sha256 = (
        core._periodic_correlation_reciprocal_metric_producer_identity_sha256()
    )
    config.reciprocal_energy_cutoff = cutoff
    config.short_range_real_space_cutoff = 0.0
    config.range_separation_omega = 0.0
    config.q_concurrency = 1
    return config


def _manifest(
    mesh: tuple[int, int, int],
    reciprocal_lattice: np.ndarray,
    q_index: int,
    *,
    cutoff: float = 0.5,
    periodic_dimension: int = 3,
) -> tuple[object, object, object, object, object, object]:
    basis = _auxiliary_basis()
    state, reference, schedule = _reference_and_schedule(
        mesh,
        reciprocal_lattice,
        periodic_dimension=periodic_dimension,
        n_auxiliary=int(basis.nbasis),
    )
    config = _config(basis, cutoff)
    source = (
        core._make_periodic_correlation_reciprocal_metric_source_manifest(
            reference,
            schedule,
            config,
            basis,
            q_index,
            core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_CANDIDATES,
        )
    )
    return source, state, reference, schedule, config, basis


def _raw_metric_config(
    auxiliary_basis: object,
    *,
    cutoff: float,
    reciprocal_block: int,
    factor_fourier_workspace_bytes: int = 0,
) -> object:
    """Complete, tiny all-reciprocal factor-build configuration."""
    config = _config(auxiliary_basis, cutoff)
    config.row_convention = (
        core._PeriodicCorrelationFactorRowConvention
        .ORIGINAL_AUXILIARY_AO_HERMITIAN_PRINCIPAL_PSEUDOINVERSE_SQUARE_ROOT
    )
    config.backing_mode = core._PeriodicCorrelationFactorBackingMode.MEMORY
    config.publisher_mode = (
        core._PeriodicCorrelationFactorPublisherMode
        .RANDOM_ACCESS_EXACTLY_ONCE
    )
    config.ao_basis_identity_sha256 = "a" * 64
    config.backend_identity_sha256 = "b" * 64
    config.codec_identity_sha256 = "c" * 64
    config.metric_absolute_eigenvalue_threshold = 1.0e-10
    config.integral_absolute_screening_threshold = 1.0e-12
    config.ao_pair_block = 4
    config.auxiliary_block = int(auxiliary_basis.nbasis)
    config.reciprocal_block = reciprocal_block
    config.whitener_column_block = int(auxiliary_basis.nbasis)
    config.native_threads = 1
    config.ao_pair_fourier_staging_required = False
    config.transpose_before_publish = False
    config.publisher_buffer_count = 1

    backend = core._PeriodicCorrelationFactorBuildBackendInventory()
    backend.complete = True
    backend.per_thread_fourier_transform_fixed_workspace_bytes = (
        factor_fourier_workspace_bytes
    )
    config.backend = backend

    codec = core._PeriodicCorrelationFactorBuildCodecInventory()
    codec.complete = True
    codec.fixed_header_bytes = 1
    codec.fixed_footer_bytes = 1
    codec.fixed_manifest_bytes = 1
    codec.integrity_bytes_per_tile = 1
    codec.rank_bytes_per_q = 1
    config.codec = codec
    return config


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 3, 1)])
@pytest.mark.parametrize("skew", [False, True])
def test_paired_source_conjugacy_audits_integer_inversion_both_directions(mesh, skew):
    reciprocal = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) if skew else np.eye(3)
    source, _, reference, schedule, config, basis = _manifest(mesh, reciprocal, 0, cutoff=2.0)
    for q in range(math.prod(mesh)):
        coordinates = np.array(np.unravel_index(q, mesh))
        opposite_q = int(np.ravel_multi_index(tuple((-coordinates) % mesh), mesh))
        left = core._make_periodic_correlation_reciprocal_metric_source_manifest(
            reference, schedule, config, basis, q, 65536,
        )
        right = core._make_periodic_correlation_reciprocal_metric_source_manifest(
            reference, schedule, config, basis, opposite_q, 65536,
        )
        left_id, right_id = left.source_identity_sha256, right.source_identity_sha256
        cap = max(left.candidate_count, right.candidate_count)
        assert core._require_periodic_correlation_reciprocal_source_conjugacy(left, right, cap) == left.accepted_vector_count
        assert core._require_periodic_correlation_reciprocal_source_conjugacy(right, left, cap) == right.accepted_vector_count
        assert left.source_identity_sha256 == left_id
        assert right.source_identity_sha256 == right_id
        # Independent explicit set, constructed only in this tiny oracle.
        records_left = _accepted_metric_records(left)
        records_right = _accepted_metric_records(right)
        right_by_p = {tuple(row[1]): row for row in records_right}
        for row in records_left:
            matched = right_by_p[tuple(-value for value in row[1])]
            assert matched[2:] == row[2:]


def test_mixed_nyquist_skew_cutoff_boundary_cannot_claim_conjugate_source():
    reciprocal = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # The two centered q norms differ. This fixed binary64 cutoff falls
    # between their v1 radial-slack membership limits for the exact pair
    # n=(1,0,0), nbar=(0,0,0). Lower-norm vectors keep both sources
    # nonempty, so the paired-source audit itself must reject this witness.
    # No source value or tolerance is changed.
    left, _, reference, schedule, config, basis = _manifest(
        (2, 3, 1), reciprocal, 4, cutoff=0.4027777777777386,
    )
    right = core._make_periodic_correlation_reciprocal_metric_source_manifest(
        reference, schedule, config, basis, 5, 65536,
    )
    assert left.radial_boundary_tolerance != right.radial_boundary_tolerance
    assert left.accepted_vector_count != right.accepted_vector_count
    for a, b in ((left, right), (right, left)):
        with pytest.raises(RuntimeError, match="unequal accepted-vector counts"):
            core._require_periodic_correlation_reciprocal_source_conjugacy(a, b, 65536)


def test_paired_source_audit_refuses_wrong_owner_q_cutoff_and_work_caps():
    left, _, reference, schedule, config, basis = _manifest((3, 1, 1), np.eye(3), 1, cutoff=2.0)
    right = core._make_periodic_correlation_reciprocal_metric_source_manifest(
        reference, schedule, config, basis, 2, 65536,
    )
    with pytest.raises(ValueError, match="opposite q addresses"):
        core._require_periodic_correlation_reciprocal_source_conjugacy(left, left, 65536)
    foreign, *_ = _manifest((3, 1, 1), np.eye(3), 2, cutoff=2.0)
    with pytest.raises(ValueError, match="matching live state owners"):
        core._require_periodic_correlation_reciprocal_source_conjugacy(left, foreign, 65536)
    config.reciprocal_energy_cutoff = 2.1
    wrong_cutoff = core._make_periodic_correlation_reciprocal_metric_source_manifest(
        reference, schedule, config, basis, 2, 65536,
    )
    with pytest.raises(ValueError, match="source geometry"):
        core._require_periodic_correlation_reciprocal_source_conjugacy(left, wrong_cutoff, 65536)
    for cap in (0, max(left.candidate_count, right.candidate_count) - 1):
        with pytest.raises(ValueError, match="explicit candidate cap"):
            core._require_periodic_correlation_reciprocal_source_conjugacy(left, right, cap)
    with pytest.raises(ValueError, match="tiny candidate limit"):
        core._require_periodic_correlation_reciprocal_source_conjugacy(left, right, 65537)


def _raw_metric_bundle(
    *,
    mesh: tuple[int, int, int] = (3, 1, 1),
    reciprocal_lattice: np.ndarray | None = None,
    basis: object | None = None,
    cutoff: float = 2.0,
    census_cutoff: float | None = None,
    reciprocal_block: int = 2,
    factor_fourier_workspace_bytes: int = 0,
    memory_limit: int = _HUGE_LIMIT,
    q_record_delta: tuple[int, int] | None = None,
) -> SimpleNamespace:
    """Build only the tiny scalar contracts needed by one raw metric."""
    if reciprocal_lattice is None:
        reciprocal_lattice = np.diag([1.0, 2.0, 3.0])
    if basis is None:
        basis = _two_s_metric_basis()
    state, reference, schedule = _reference_and_schedule(
        mesh,
        reciprocal_lattice,
        n_auxiliary=int(basis.nbasis),
        memory_limit=memory_limit,
    )
    source_config = _raw_metric_config(
        basis,
        cutoff=cutoff,
        reciprocal_block=reciprocal_block,
        factor_fourier_workspace_bytes=factor_fourier_workspace_bytes,
    )
    source_factory = (
        core._make_periodic_correlation_reciprocal_metric_source_manifest
    )
    candidate_cap = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_CANDIDATES
    )
    sources = [
        source_factory(
            reference,
            schedule,
            source_config,
            basis,
            q_index,
            candidate_cap,
        )
        for q_index in range(math.prod(mesh))
    ]
    records = [source.factor_build_q_record() for source in sources]
    if q_record_delta is not None:
        q_index, delta = q_record_delta
        records[q_index].base_reciprocal_vector_count += delta

    census_config = _raw_metric_config(
        basis,
        cutoff=cutoff if census_cutoff is None else census_cutoff,
        reciprocal_block=reciprocal_block,
        factor_fourier_workspace_bytes=factor_fourier_workspace_bytes,
    )
    census = core._make_periodic_correlation_factor_build_census(
        reference, schedule, census_config, records
    )
    plan = core._plan_periodic_correlation_factor_build(
        reference, schedule, census
    )
    return SimpleNamespace(
        basis=basis,
        state=state,
        reference=reference,
        schedule=schedule,
        source_config=source_config,
        sources=sources,
        census=census,
        plan=plan,
    )


def _source_identity_oracle(
    source: object,
    reference: object,
    schedule: object,
    config: object,
) -> tuple[str, int, int]:
    """Hash the documented source wire and independently enumerate records."""
    wire = _CanonicalDigest(
        "vibeqc.periodic.correlation.reciprocal-metric.source",
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_SOURCE_CONTRACT_VERSION,
    )
    wire.string(core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_PRODUCER_ID)
    wire.u32(core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_PRODUCER_VERSION)
    wire.string(
        core._periodic_correlation_reciprocal_metric_producer_identity_sha256()
    )
    wire.u32(reference.contract_version)
    wire.u32(schedule.contract_version)
    wire.u32(reference.state.digest_version)
    wire.u32(reference.dimensions.allocation_contract_version)
    wire.string(reference.state.state_identity_sha256)
    wire.string(reference.dimensions.calculation_identity)
    wire.string(reference.dimensions.allocation_identity)
    wire.string(schedule.schedule_identity_sha256)
    wire.u32(core._AUXILIARY_BASIS_CONTENT_DIGEST_VERSION)
    wire.string(config.auxiliary_basis_identity_sha256)
    wire.u32(source.periodic_dimension)
    for component in source.mesh:
        wire.i32(component)
    for component in source.is_shift:
        wire.i32(component)
    wire.u64(source.q_index)
    for component in source.centered_doubled_numerator:
        wire.i32(component)
    for component in source.centered_reciprocal_wrap:
        wire.i32(component)
    reciprocal_lattice = np.asarray(source.reciprocal_lattice)
    for value in reciprocal_lattice.ravel(order="C"):
        wire.binary64(float(value))
    for value in (
        source.reciprocal_energy_cutoff,
        source.maximum_reciprocal_radius,
        source.radial_boundary_tolerance,
        source.cell_volume_bohr3,
    ):
        wire.binary64(value)
    for component in source.lower_bounds:
        wire.i64(component)
    for component in source.upper_bounds:
        wire.i64(component)
    wire.u64(source.candidate_count)
    wire.u64(source.accepted_vector_count)
    wire.u64(source.zero_mode_excluded_count)

    assert (
        wire.wire_bytes
        == core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_SOURCE_FIXED_PREFIX_WIRE_BYTES
    )
    fractional = tuple(float(value) for value in source.q_fractional)
    exact_gamma = all(
        component == 0 for component in source.centered_doubled_numerator
    )
    membership_limit = (
        2.0 * source.reciprocal_energy_cutoff
        if exact_gamma
        else (
            source.maximum_reciprocal_radius
            + source.radial_boundary_tolerance
        )
        ** 2
    )
    accepted_count = 0
    for n0 in range(source.lower_bounds[0], source.upper_bounds[0] + 1):
        for n1 in range(source.lower_bounds[1], source.upper_bounds[1] + 1):
            for n2 in range(
                source.lower_bounds[2], source.upper_bounds[2] + 1
            ):
                if exact_gamma and n0 == 0 and n1 == 0 and n2 == 0:
                    continue
                shifted = tuple(
                    _binary64_fma(1.0, float(label), fractional[axis])
                    for axis, label in enumerate((n0, n1, n2))
                )

                def fixed_dot3(row: int) -> float:
                    return _binary64_fma(
                        float(reciprocal_lattice[row, 0]),
                        shifted[0],
                        _binary64_fma(
                            float(reciprocal_lattice[row, 1]),
                            shifted[1],
                            _binary64_fma(
                                float(reciprocal_lattice[row, 2]),
                                shifted[2],
                                0.0,
                            ),
                        ),
                    )

                px, py, pz = (fixed_dot3(row) for row in range(3))
                p2 = _binary64_fma(
                    px,
                    px,
                    _binary64_fma(
                        py, py, _binary64_fma(pz, pz, 0.0)
                    ),
                )
                if p2 > membership_limit:
                    continue
                weight = (4.0 * math.pi / source.cell_volume_bohr3) / p2
                for component in (n0, n1, n2):
                    wire.i64(component)
                for value in (px, py, pz, p2, weight):
                    wire.binary64(value)
                accepted_count += 1
    return wire.finish(), wire.wire_bytes, accepted_count


def _fixed_fma_squared_norm(
    reciprocal_lattice: np.ndarray,
    shifted_fractional: tuple[float, float, float],
) -> float:
    def dot3(row: np.ndarray) -> float:
        return _binary64_fma(
            float(row[0]),
            shifted_fractional[0],
            _binary64_fma(
                float(row[1]),
                shifted_fractional[1],
                _binary64_fma(
                    float(row[2]), shifted_fractional[2], 0.0
                ),
            ),
        )

    px, py, pz = (dot3(reciprocal_lattice[row]) for row in range(3))
    return _binary64_fma(
        px,
        px,
        _binary64_fma(py, py, _binary64_fma(pz, pz, 0.0)),
    )


def _accepted_metric_records(
    source: object,
) -> list[tuple[tuple[int, int, int], np.ndarray, float, float]]:
    """Independently replay the documented fixed-binary64 predicate."""
    reciprocal_lattice = np.asarray(source.reciprocal_lattice)
    fractional = tuple(float(value) for value in source.q_fractional)
    exact_gamma = all(
        component == 0 for component in source.centered_doubled_numerator
    )
    membership_limit = (
        2.0 * source.reciprocal_energy_cutoff
        if exact_gamma
        else (
            source.maximum_reciprocal_radius
            + source.radial_boundary_tolerance
        )
        ** 2
    )
    records = []
    for n0 in range(source.lower_bounds[0], source.upper_bounds[0] + 1):
        for n1 in range(source.lower_bounds[1], source.upper_bounds[1] + 1):
            for n2 in range(
                source.lower_bounds[2], source.upper_bounds[2] + 1
            ):
                label = (n0, n1, n2)
                if exact_gamma and label == (0, 0, 0):
                    continue
                shifted = tuple(
                    _binary64_fma(1.0, float(label[axis]), fractional[axis])
                    for axis in range(3)
                )

                def dot3(row: int) -> float:
                    return _binary64_fma(
                        float(reciprocal_lattice[row, 0]),
                        shifted[0],
                        _binary64_fma(
                            float(reciprocal_lattice[row, 1]),
                            shifted[1],
                            _binary64_fma(
                                float(reciprocal_lattice[row, 2]),
                                shifted[2],
                                0.0,
                            ),
                        ),
                    )

                p = np.asarray([dot3(row) for row in range(3)])
                p2 = _binary64_fma(
                    float(p[0]),
                    float(p[0]),
                    _binary64_fma(
                        float(p[1]),
                        float(p[1]),
                        _binary64_fma(float(p[2]), float(p[2]), 0.0),
                    ),
                )
                if p2 > membership_limit:
                    continue
                weight = (4.0 * math.pi / source.cell_volume_bohr3) / p2
                records.append((label, p, p2, weight))
    return records


def _analytic_s_fourier(basis: object, p: np.ndarray, p2: float) -> np.ndarray:
    """Independent L=0 formula; deliberately does not call a Fourier API."""
    values = []
    for shell in basis.shells():
        assert shell.l == 0
        assert shell.pure
        radial = math.fsum(
            float(coefficient)
            * (math.pi / float(exponent)) ** 1.5
            * math.exp(-p2 / (4.0 * float(exponent)))
            for exponent, coefficient in zip(
                shell.exponents, shell.coefficients, strict=True
            )
        )
        phase = cmath.exp(-1j * float(np.dot(p, shell.origin)))
        values.append(radial * phase)
    return np.asarray(values, dtype=np.complex128)


def _analytic_raw_metric(source: object, basis: object) -> np.ndarray:
    n_auxiliary = int(basis.nbasis)
    terms = [
        [[] for _ in range(n_auxiliary)] for _ in range(n_auxiliary)
    ]
    for _, p, p2, weight in _accepted_metric_records(source):
        fourier = _analytic_s_fourier(basis, p, p2)
        for row in range(n_auxiliary):
            for column in range(row, n_auxiliary):
                value = weight * fourier[row].conjugate() * fourier[column]
                terms[row][column].append(value)

    metric = np.zeros((n_auxiliary, n_auxiliary), dtype=np.complex128)
    for row in range(n_auxiliary):
        metric[row, row] = math.fsum(
            value.real for value in terms[row][row]
        )
        for column in range(row + 1, n_auxiliary):
            value = complex(
                math.fsum(item.real for item in terms[row][column]),
                math.fsum(item.imag for item in terms[row][column]),
            )
            metric[row, column] = value
            metric[column, row] = value.conjugate()
    return metric


def _leaf_composed_raw_metric(source: object, basis: object) -> np.ndarray:
    """Independent contraction around the separately tested Fourier leaf."""
    records = _accepted_metric_records(source)
    vectors = np.ascontiguousarray(
        np.asarray([record[1] for record in records]), dtype=np.float64
    )
    required = int(basis.nbasis) * len(records) * 16
    fourier = np.asarray(
        core._periodic_auxiliary_gaussian_fourier_panel(
            basis, vectors, required
        )
    )
    weights = np.asarray([record[3] for record in records])
    return np.einsum(
        "g,pg,qg->pq",
        weights,
        fourier.conj(),
        fourier,
        optimize=False,
    )


def _raw_metric_result(bundle: SimpleNamespace, q_index: int) -> object:
    return core._build_periodic_correlation_reciprocal_metric(
        bundle.reference,
        bundle.schedule,
        bundle.census,
        bundle.sources[q_index],
        bundle.basis,
    )


def _payload_identity_oracle(
    result: object, metric: np.ndarray
) -> tuple[str, int]:
    metric = np.asarray(metric, dtype=np.complex128, order="C")
    assert metric.shape == (result.n_auxiliary, result.n_auxiliary)
    wire = _CanonicalDigest(_PAYLOAD_DOMAIN, result.contract_version)
    wire.string(result.source_identity_sha256)
    wire.u64(result.q_index)
    wire.u64(result.n_auxiliary)
    wire.u64(result.n_auxiliary * result.n_auxiliary)
    wire.u64(result.metric_bytes)
    for value in metric.ravel(order="C"):
        wire.complex128(complex(value))
    return wire.finish(), wire.wire_bytes


def _assert_exact_hermitian(metric: np.ndarray) -> None:
    np.testing.assert_array_equal(metric, metric.conj().T)
    diagonal_imaginary = np.ascontiguousarray(metric.diagonal().imag)
    assert np.count_nonzero(diagonal_imaginary) == 0
    assert not np.any(np.signbit(diagonal_imaginary))


def _reciprocal_metric_phase(plan: object) -> object:
    phase_kind = (
        core._PeriodicCorrelationFactorBuildPhase.RECIPROCAL_METRIC
    )
    return next(phase for phase in plan.phases if phase.phase == phase_kind)


def test_producer_identity_and_source_wire_boundaries() -> None:
    producer_wire = _CanonicalDigest(
        "vibeqc.periodic.correlation.reciprocal-metric.producer",
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_SOURCE_CONTRACT_VERSION,
    )
    producer_wire.string(
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_PRODUCER_ID
    )
    producer_wire.u32(
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_PRODUCER_VERSION
    )
    actual = (
        core._periodic_correlation_reciprocal_metric_producer_identity_sha256()
    )
    assert actual == producer_wire.finish()
    assert actual == _PRODUCER_IDENTITY_KAT

    fixed = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_SOURCE_FIXED_PREFIX_WIRE_BYTES
    )
    record = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_ACCEPTED_VECTOR_WIRE_BYTES
    )
    maximum_sha_bytes = ((1 << 64) - 1) // 8
    maximum_count = (maximum_sha_bytes - fixed) // record
    estimator = core._periodic_correlation_reciprocal_metric_source_wire_bytes
    assert fixed == 812
    assert record == 64
    assert estimator(0) == fixed
    assert estimator(1) == fixed + record
    assert estimator(maximum_count) == fixed + record * maximum_count
    assert estimator(maximum_count) <= maximum_sha_bytes
    with pytest.raises(ValueError, match="SHA-256"):
        estimator(maximum_count + 1)
    with pytest.raises(ValueError, match="SHA-256"):
        estimator((1 << 64) - 1)


def test_certified_skew_plan_contains_legacy_omitted_fma_witness() -> None:
    reciprocal_lattice = np.asarray(
        [
            [0.7781018533990887, -0.7781018533990921, 0.0],
            [0.9660463621881401, -0.9660463621881372, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    squared_limit = 7.703246461005811e-26
    witness = (62.0, 62.0, 0.0)
    assert (
        _fixed_fma_squared_norm(reciprocal_lattice, witness)
        <= squared_limit
    )

    plan = core._plan_periodic_correlation_reciprocal_metric_enumeration(
        reciprocal_lattice, np.zeros(3), squared_limit
    )
    assert plan.lower_bounds[0] <= -62
    assert plan.lower_bounds[1] <= -62
    assert plan.upper_bounds[0] >= 62
    assert plan.upper_bounds[1] >= 62
    assert plan.lower_bounds[2] <= 0 <= plan.upper_bounds[2]
    assert plan.candidate_count == math.prod(
        upper - lower + 1
        for lower, upper in zip(
            plan.lower_bounds, plan.upper_bounds, strict=True
        )
    )
    assert (
        plan.candidate_count
        <= core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_CANDIDATES
    )

    # Reconstruct the former inverse-row reach and fixed +/-1 padding.  Its
    # ordinary binary64 2x2 determinant gives upper bound 61 on both nearly
    # dependent axes, so it cannot enumerate the accepted label above.
    determinant = (
        reciprocal_lattice[0, 0] * reciprocal_lattice[1, 1]
        - reciprocal_lattice[0, 1] * reciprocal_lattice[1, 0]
    )
    inverse_rows = (
        (
            reciprocal_lattice[1, 1] / determinant,
            -reciprocal_lattice[0, 1] / determinant,
        ),
        (
            -reciprocal_lattice[1, 0] / determinant,
            reciprocal_lattice[0, 0] / determinant,
        ),
    )
    legacy_upper_bounds = tuple(
        math.floor(math.sqrt(squared_limit) * math.hypot(*row)) + 1
        for row in inverse_rows
    )
    assert legacy_upper_bounds == (61, 61)


def test_admitted_floor_skew_plan_contains_large_witness_but_exceeds_cap() -> None:
    reciprocal_lattice = np.asarray(
        [
            [
                -0.50359090764391135,
                0.50359090764300329,
                -0.048070284863212132,
            ],
            [
                0.03573970777717228,
                -0.035739707777252695,
                0.99643701695803677,
            ],
            [
                -0.86320268247172638,
                0.86320268247225274,
                0.069300208866488452,
            ],
        ],
        dtype=np.float64,
    )
    squared_limit = 2.0 * 5.5403766338906144e-17
    witness = (10000.0, 10000.0, 0.0)
    assert (
        _fixed_fma_squared_norm(reciprocal_lattice, witness)
        <= squared_limit
    )

    plan = core._plan_periodic_correlation_reciprocal_metric_enumeration(
        reciprocal_lattice, np.zeros(3), squared_limit
    )
    assert plan.lower_bounds[0] <= -10000
    assert plan.lower_bounds[1] <= -10000
    assert plan.upper_bounds[0] >= 10000
    assert plan.upper_bounds[1] >= 10000
    assert plan.candidate_count > 65536


def test_enumeration_certificate_failure_modes_are_distinct() -> None:
    planner = core._plan_periodic_correlation_reciprocal_metric_enumeration
    with pytest.raises(RuntimeError, match="determinant_not_certified"):
        planner(np.diag([1.0, 1.0, 0.0]), np.zeros(3), 1.0)

    noncontractive = np.diag([1.0, 1.0, math.ldexp(1.0, -100)])
    with pytest.raises(
        RuntimeError, match="noncontractive_matvec_certificate"
    ):
        planner(noncontractive, np.zeros(3), 1.0)

    with pytest.raises(ValueError, match=r"\[-1/2,1/2\)"):
        planner(np.eye(3), np.asarray([0.5, 0.0, 0.0]), 1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        planner(np.eye(3), np.zeros(3), 0.0)


def test_diagonal_gamma_source_counts_wire_hash_and_factor_record() -> None:
    source, state, reference, schedule, config, _ = _manifest(
        (1, 1, 1), np.eye(3), 0
    )
    assert source.contract_version == 1
    assert source.periodic_dimension == 3
    assert tuple(source.mesh) == (1, 1, 1)
    assert tuple(source.is_shift) == (0, 0, 0)
    assert source.q_index == 0
    assert tuple(source.centered_doubled_numerator) == (0, 0, 0)
    assert tuple(source.centered_reciprocal_wrap) == (0, 0, 0)
    np.testing.assert_array_equal(source.q_fractional, np.zeros(3))
    np.testing.assert_array_equal(source.q_cartesian, np.zeros(3))
    assert source.maximum_reciprocal_radius == 1.0
    assert source.radial_boundary_tolerance == 0.0
    scaled_two_pi = 2.0 * math.pi
    expected_volume = _binary64_fma(
        _binary64_fma(scaled_two_pi, scaled_two_pi, 0.0),
        scaled_two_pi,
        0.0,
    )
    assert source.cell_volume_bohr3 == expected_volume
    assert source.zero_mode_excluded_count == 1
    assert source.accepted_vector_count == 6
    assert source.candidate_count == math.prod(
        upper - lower + 1
        for lower, upper in zip(
            source.lower_bounds, source.upper_bounds, strict=True
        )
    )

    expected_hash, expected_bytes, accepted_count = _source_identity_oracle(
        source, reference, schedule, config
    )
    assert accepted_count == source.accepted_vector_count
    assert expected_hash == source.source_identity_sha256
    assert source.source_identity_sha256 == _GAMMA_SOURCE_IDENTITY_KAT
    assert expected_bytes == source.source_wire_bytes
    assert expected_bytes == 812 + 6 * 64
    assert source.state.state_identity_sha256 == state.state_identity_sha256

    record = source.factor_build_q_record()
    assert record.q_index == 0
    assert tuple(record.centered_doubled_numerator) == (0, 0, 0)
    assert tuple(record.centered_reciprocal_wrap) == (0, 0, 0)
    assert record.base_reciprocal_vector_count == 6
    assert record.tail_reciprocal_vector_count == 0
    assert record.zero_mode_excluded_count == 1
    assert record.short_range_metric_cell_count == 0
    assert record.short_range_metric_task_count == 0
    assert record.short_range_three_center_cell_pair_count == 0
    assert record.short_range_three_center_task_count == 0
    assert record.source_manifest_bytes == 64


@pytest.mark.parametrize(
    ("q_index", "expected_numerator", "expected_wrap", "expected_q"),
    [
        (1, (2, 0, 0), (0, 0, 0), (1.0, 0.0, 0.0)),
        (2, (-2, 0, 0), (1, 0, 0), (-1.0, 0.0, 0.0)),
    ],
)
def test_diagonal_non_gamma_centering_counts_and_source_hash(
    q_index: int,
    expected_numerator: tuple[int, int, int],
    expected_wrap: tuple[int, int, int],
    expected_q: tuple[float, float, float],
) -> None:
    source, _, reference, schedule, config, _ = _manifest(
        (3, 1, 1), np.diag([3.0, 4.0, 5.0]), q_index
    )
    assert tuple(source.centered_doubled_numerator) == expected_numerator
    assert tuple(source.centered_reciprocal_wrap) == expected_wrap
    np.testing.assert_array_equal(source.q_cartesian, expected_q)
    assert source.radial_boundary_tolerance > 0.0
    assert source.zero_mode_excluded_count == 0
    assert source.accepted_vector_count == 1

    expected_hash, expected_bytes, accepted_count = _source_identity_oracle(
        source, reference, schedule, config
    )
    assert accepted_count == 1
    assert expected_hash == source.source_identity_sha256
    assert expected_bytes == source.source_wire_bytes == 812 + 64


def test_binding_and_caller_candidate_caps_fail_before_unbounded_work() -> None:
    source, _, reference, schedule, config, basis = _manifest(
        (1, 1, 1), np.eye(3), 0
    )
    hard_cap = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_CANDIDATES
    )
    assert hard_cap == 65536
    factory = (
        core._make_periodic_correlation_reciprocal_metric_source_manifest
    )
    with pytest.raises(ValueError, match="limited to 65536"):
        factory(reference, schedule, config, basis, 0, hard_cap + 1)
    with pytest.raises(ValueError, match="cap must be positive"):
        factory(reference, schedule, config, basis, 0, 0)
    with pytest.raises(ValueError, match="caller cap"):
        factory(
            reference,
            schedule,
            config,
            basis,
            0,
            source.candidate_count - 1,
        )
    with pytest.raises(IndexError, match="outside the full mesh"):
        factory(reference, schedule, config, basis, 1, hard_cap)


def test_dimension_upstream_state_and_auxiliary_basis_must_match() -> None:
    basis = _auxiliary_basis()
    _, reference, schedule = _reference_and_schedule(
        (1, 1, 1), np.eye(3), n_auxiliary=int(basis.nbasis)
    )
    config = _config(basis)
    factory = (
        core._make_periodic_correlation_reciprocal_metric_source_manifest
    )
    cap = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_CANDIDATES
    )

    _, _, other_schedule = _reference_and_schedule(
        (1, 1, 1), np.eye(3), n_auxiliary=int(basis.nbasis)
    )
    with pytest.raises(ValueError, match="same mean-field state object"):
        factory(reference, other_schedule, config, basis, 0, cap)

    _, two_dimensional_reference, two_dimensional_schedule = (
        _reference_and_schedule(
            (1, 1, 1),
            np.eye(3),
            periodic_dimension=2,
            n_auxiliary=int(basis.nbasis),
        )
    )
    with pytest.raises(ValueError, match="periodic dimension three"):
        factory(
            two_dimensional_reference,
            two_dimensional_schedule,
            config,
            basis,
            0,
            cap,
        )

    changed_basis = _auxiliary_basis(exponent=0.7500000000000001)
    with pytest.raises(ValueError, match="content identity"):
        factory(reference, schedule, config, changed_basis, 0, cap)

    larger_basis = _auxiliary_basis(shell_count=2)
    with pytest.raises(ValueError, match="size"):
        factory(reference, schedule, config, larger_basis, 0, cap)


def test_manifest_is_read_only_keeps_state_and_exposes_no_vector_list() -> None:
    source, state, reference, schedule, _, _ = _manifest(
        (1, 1, 1), np.eye(3), 0
    )
    with pytest.raises(TypeError):
        core._PeriodicCorrelationReciprocalMetricSourceManifest()
    with pytest.raises(AttributeError):
        source.q_index = 1
    for forbidden in (
        "accepted_vectors",
        "integer_labels",
        "reciprocal_vectors",
        "records",
        "vectors",
    ):
        assert not hasattr(source, forbidden)

    state_identity = state.state_identity_sha256
    del state, reference, schedule
    gc.collect()
    assert source.state.state_identity_sha256 == state_identity


def test_auxiliary_fourier_uses_the_source_fixed_fma_norm_witness() -> None:
    vector = np.asarray(
        [
            [
                -float.fromhex("0x1.3b028bc9537f8p+1"),
                -float.fromhex("0x1.7e660da0cfad4p+1"),
                -float.fromhex("0x1.6badad13c6c10p-1"),
            ]
        ],
        dtype=np.float64,
    )
    px, py, pz = (float(value) for value in vector[0])
    ordinary_p2 = px * px + py * py + pz * pz
    fixed_p2 = _binary64_fma(
        px, px, _binary64_fma(py, py, _binary64_fma(pz, pz, 0.0))
    )
    assert ordinary_p2.hex() == "0x1.ef8f2ea6a789fp+3"
    assert fixed_p2.hex() == "0x1.ef8f2ea6a78a0p+3"

    basis = _auxiliary_basis(exponent=1.0, coefficient=1.0)
    actual = np.asarray(
        core._periodic_auxiliary_gaussian_fourier_panel(
            basis, vector, 16
        )
    )[0, 0]
    radial_scale = math.pi ** 1.5
    fixed_expected = radial_scale * math.exp(-fixed_p2 / 4.0)
    ordinary_expected = radial_scale * math.exp(-ordinary_p2 / 4.0)
    assert fixed_expected != ordinary_expected
    assert actual.real == fixed_expected

    source = (_ROOT / "cpp/src/periodic_auxiliary_fourier.cpp").read_text()
    assert source.count("fixed_binary64_squared_norm(px, py, pz)") == 2
    assert "px * px + py * py + pz * pz" not in source


def test_raw_gamma_metric_matches_independent_two_s_oracle() -> None:
    bundle = _raw_metric_bundle(reciprocal_block=2)
    assert bundle.plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    source = bundle.sources[0]
    result = _raw_metric_result(bundle, 0)
    metric = np.asarray(result.matrix_copy())

    labels = [record[0] for record in _accepted_metric_records(source)]
    assert labels == [
        (-2, 0, 0),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (1, 0, 0),
        (2, 0, 0),
    ]
    assert (0, 0, 0) not in labels
    assert source.zero_mode_excluded_count == 1
    assert result.accepted_vector_count == source.accepted_vector_count == 6
    expected = _analytic_raw_metric(source, bundle.basis)
    np.testing.assert_allclose(metric, expected, rtol=2.0e-12, atol=2.0e-13)

    assert metric.shape == (2, 2)
    assert metric.dtype == np.complex128
    assert metric.flags.c_contiguous
    assert np.isfinite(metric).all()
    _assert_exact_hermitian(metric)
    assert result.self_conjugate_transfer
    assert result.conjugate_q_index == 0
    assert np.count_nonzero(metric.imag) == 0
    eigenvalues = np.linalg.eigvalsh(metric)
    tolerance = 64.0 * np.finfo(float).eps * float(np.max(metric.real))
    assert float(eigenvalues[0]) >= -tolerance


def test_raw_metric_q_minus_q_covariance_uses_exact_label_wrap() -> None:
    bundle = _raw_metric_bundle(reciprocal_block=2)
    positive = _raw_metric_result(bundle, 1)
    negative = _raw_metric_result(bundle, 2)
    positive_metric = np.asarray(positive.matrix_copy())
    negative_metric = np.asarray(negative.matrix_copy())

    positive_records = _accepted_metric_records(bundle.sources[1])
    negative_labels = {
        record[0] for record in _accepted_metric_records(bundle.sources[2])
    }
    positive_fractional = np.asarray(bundle.sources[1].q_fractional)
    negative_fractional = np.asarray(bundle.sources[2].q_fractional)
    reciprocal_shift = np.rint(
        positive_fractional + negative_fractional
    ).astype(int)
    assert tuple(reciprocal_shift) == (0, 0, 0)
    for label, _, _, _ in positive_records:
        conjugate_label = tuple(
            -label[axis] - int(reciprocal_shift[axis])
            for axis in range(3)
        )
        assert conjugate_label in negative_labels

    assert positive.conjugate_q_index == 2
    assert negative.conjugate_q_index == 1
    assert not positive.self_conjugate_transfer
    assert not negative.self_conjugate_transfer
    positive_expected = _analytic_raw_metric(
        bundle.sources[1], bundle.basis
    )
    negative_expected = _analytic_raw_metric(
        bundle.sources[2], bundle.basis
    )
    assert positive_metric[0, 1].imag < 0.0
    assert negative_metric[0, 1].imag > 0.0
    np.testing.assert_allclose(
        positive_metric, positive_expected, rtol=2.0e-12, atol=2.0e-13
    )
    np.testing.assert_allclose(
        negative_metric, negative_expected, rtol=2.0e-12, atol=2.0e-13
    )
    np.testing.assert_allclose(
        negative_metric,
        positive_metric.conj(),
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    _assert_exact_hermitian(positive_metric)
    _assert_exact_hermitian(negative_metric)
    assert positive.source_identity_sha256 != negative.source_identity_sha256
    assert positive.payload_identity_sha256 != negative.payload_identity_sha256


def test_raw_metric_composes_pure_spherical_p_shell_fourier_panel() -> None:
    basis = _mixed_s_p_metric_basis()
    bundle = _raw_metric_bundle(basis=basis, reciprocal_block=2)
    result = _raw_metric_result(bundle, 1)
    metric = np.asarray(result.matrix_copy())
    expected = _leaf_composed_raw_metric(bundle.sources[1], basis)

    assert metric.shape == (4, 4)
    assert np.max(np.abs(metric.imag)) > 1.0e-3
    _assert_exact_hermitian(metric)
    np.testing.assert_allclose(metric, expected, rtol=2.0e-12, atol=2.0e-13)

    gamma = _raw_metric_result(bundle, 0)
    gamma_metric = np.asarray(gamma.matrix_copy())
    assert gamma.self_conjugate_transfer
    assert gamma.maximum_self_conjugate_imaginary_residual == 0.0
    assert np.count_nonzero(gamma_metric.imag) == 0
    np.testing.assert_allclose(
        gamma_metric,
        _leaf_composed_raw_metric(bundle.sources[0], basis),
        rtol=2.0e-12,
        atol=2.0e-13,
    )

    nyquist_bundle = _raw_metric_bundle(
        mesh=(2, 1, 1), basis=basis, reciprocal_block=2
    )
    nyquist = _raw_metric_result(nyquist_bundle, 1)
    nyquist_metric = np.asarray(nyquist.matrix_copy())
    assert nyquist.self_conjugate_transfer
    assert nyquist.maximum_self_conjugate_imaginary_residual == 0.0
    assert np.count_nonzero(nyquist_metric.imag) == 0
    np.testing.assert_allclose(
        nyquist_metric,
        _leaf_composed_raw_metric(nyquist_bundle.sources[1], basis),
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_negative_nyquist_self_inverse_metric_is_bitwise_real() -> None:
    bundle = _raw_metric_bundle(mesh=(2, 1, 1), reciprocal_block=3)
    source = bundle.sources[1]
    result = _raw_metric_result(bundle, 1)
    metric = np.asarray(result.matrix_copy())

    assert tuple(source.centered_doubled_numerator) == (-2, 0, 0)
    labels = {record[0] for record in _accepted_metric_records(source)}
    assert labels == {
        (-1, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    }
    mapped_labels = {
        tuple(
            1 - label[0] if axis == 0 else -label[axis]
            for axis in range(3)
        )
        for label in labels
    }
    assert mapped_labels == labels
    assert result.self_conjugate_transfer
    assert result.conjugate_q_index == result.q_index == 1
    assert result.maximum_self_conjugate_imaginary_residual == 0.0
    _assert_exact_hermitian(metric)
    imaginary = np.ascontiguousarray(metric.imag)
    assert np.count_nonzero(imaginary) == 0
    assert not np.any(np.signbit(imaginary))
    np.testing.assert_allclose(
        metric,
        _analytic_raw_metric(source, bundle.basis),
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_positive_semidefinite_zero_metric_is_left_for_factorization() -> None:
    basis = _auxiliary_basis(exponent=1.0e-4, coefficient=1.0)
    bundle = _raw_metric_bundle(basis=basis)
    assert bundle.plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    result = _raw_metric_result(bundle, 0)
    metric = np.asarray(result.matrix_copy())

    np.testing.assert_array_equal(
        metric, np.zeros((1, 1), dtype=np.complex128)
    )
    assert result.maximum_self_conjugate_imaginary_residual == 0.0
    expected_identity, expected_wire_bytes = _payload_identity_oracle(
        result, metric
    )
    assert result.payload_identity_sha256 == expected_identity
    assert expected_wire_bytes == 169 + 16


def test_raw_metric_nonfinite_contraction_fails_closed() -> None:
    basis = _auxiliary_basis(exponent=0.75, coefficient=1.0e154)
    bundle = _raw_metric_bundle(basis=basis)
    assert bundle.plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    with pytest.raises(
        (ValueError, RuntimeError, OverflowError),
        match="non-finite|overflow|finite",
    ):
        _raw_metric_result(bundle, 0)


def test_reciprocal_panel_size_does_not_change_metric_or_payload_bits() -> None:
    bundles = [
        _raw_metric_bundle(reciprocal_block=block) for block in (1, 2, 4)
    ]
    results = [_raw_metric_result(bundle, 1) for bundle in bundles]
    matrices = [np.asarray(result.matrix_copy()) for result in results]

    for matrix in matrices[1:]:
        np.testing.assert_array_equal(matrix, matrices[0])
    assert len({result.source_identity_sha256 for result in results}) == 1
    assert len({result.payload_identity_sha256 for result in results}) == 1
    census_identities = {
        bundle.census.census_identity_sha256 for bundle in bundles
    }
    assert len(census_identities) == 3
    assert len({bundle.plan.plan_identity_sha256 for bundle in bundles}) == 3
    assert [result.reciprocal_panel_capacity for result in results] == [1, 2, 4]
    assert [result.completed_panel_count for result in results] == [4, 2, 1]


def test_payload_wire_provenance_and_matrix_copy_are_bounded() -> None:
    bundle = _raw_metric_bundle(reciprocal_block=2)
    source = bundle.sources[1]
    result = _raw_metric_result(bundle, 1)
    first = np.asarray(result.matrix_copy())
    expected_hash, expected_bytes = _payload_identity_oracle(result, first)

    assert result.contract_version == (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_RESULT_CONTRACT_VERSION
    )
    assert result.source_identity_sha256 == source.source_identity_sha256
    assert result.census_identity_sha256 == bundle.census.census_identity_sha256
    assert result.plan_identity_sha256 == bundle.plan.plan_identity_sha256
    assert result.auxiliary_basis_identity_sha256 == (
        core._auxiliary_basis_content_identity_sha256(bundle.basis)
    )
    assert result.payload_identity_sha256 == expected_hash
    assert expected_bytes == 169 + result.metric_bytes
    assert result.metric_bytes == 16 * result.n_auxiliary**2 == 64

    panel_capacity = min(
        bundle.census.config.reciprocal_block,
        result.accepted_vector_count,
    )
    assert result.reciprocal_panel_capacity == panel_capacity
    assert result.completed_panel_count == math.ceil(
        result.accepted_vector_count / panel_capacity
    )
    assert result.reciprocal_panel_bytes == 5 * 8 * panel_capacity
    assert result.maximum_auxiliary_fourier_panel_bytes == (
        16 * result.n_auxiliary * panel_capacity
    )
    assert result.maximum_weighted_fourier_panel_bytes == (
        16 * result.n_auxiliary * panel_capacity
    )
    assert result.admitted_reciprocal_metric_peak_bytes == (
        _reciprocal_metric_phase(bundle.plan).peak_memory_bytes
    )

    first[0, 0] += 1.0
    second = np.asarray(result.matrix_copy())
    assert second[0, 0] != first[0, 0]
    assert result.payload_identity_sha256 == expected_hash
    with pytest.raises(TypeError):
        core._PeriodicCorrelationReciprocalMetricResult()
    with pytest.raises(AttributeError):
        result.q_index = 0
    for forbidden in (
        "accepted_vectors",
        "integer_labels",
        "reciprocal_vectors",
        "weights",
        "fourier_panel",
    ):
        assert not hasattr(result, forbidden)

    state_identity = bundle.state.state_identity_sha256
    surviving_matrix = second.copy()
    del bundle, source
    gc.collect()
    assert result.state.state_identity_sha256 == state_identity
    del result
    gc.collect()
    np.testing.assert_array_equal(second, surviving_matrix)


def test_payload_wire_extent_is_preflighted_over_the_sha_domain() -> None:
    estimator = core._periodic_correlation_reciprocal_metric_payload_wire_bytes
    prefix = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_PAYLOAD_FIXED_PREFIX_WIRE_BYTES
    )
    assert prefix == 169
    assert estimator(0) == 169
    assert estimator(1) == 185
    assert estimator(2) == 233

    maximum_sha_bytes = ((1 << 64) - 1) // 8
    maximum_auxiliary = math.isqrt((maximum_sha_bytes - prefix) // 16)
    assert maximum_auxiliary == 379_625_062
    assert estimator(maximum_auxiliary) == 2_305_843_003_176_061_673
    assert estimator(maximum_auxiliary) <= maximum_sha_bytes
    with pytest.raises(ValueError, match="SHA-256"):
        estimator(maximum_auxiliary + 1)
    with pytest.raises(ValueError, match="SHA-256"):
        estimator((1 << 64) - 1)


def test_raw_metric_rejects_source_census_basis_and_state_mismatches() -> None:
    good = _raw_metric_bundle()

    wrong_record = _raw_metric_bundle(q_record_delta=(1, 1))
    assert wrong_record.plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    with pytest.raises((ValueError, RuntimeError), match="q record|source"):
        _raw_metric_result(wrong_record, 1)

    wrong_cutoff = _raw_metric_bundle(
        census_cutoff=math.nextafter(2.0, math.inf)
    )
    assert wrong_cutoff.plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    with pytest.raises(
        (ValueError, RuntimeError), match="source|cutoff|configuration"
    ):
        _raw_metric_result(wrong_cutoff, 1)

    changed_basis = _two_s_metric_basis(
        second_exponent=math.nextafter(1.0, math.inf)
    )
    with pytest.raises((ValueError, RuntimeError), match="basis|identity"):
        core._build_periodic_correlation_reciprocal_metric(
            good.reference,
            good.schedule,
            good.census,
            good.sources[1],
            changed_basis,
        )

    equivalent_but_distinct = _raw_metric_bundle()
    with pytest.raises((ValueError, RuntimeError), match="state|source"):
        core._build_periodic_correlation_reciprocal_metric(
            equivalent_but_distinct.reference,
            equivalent_but_distinct.schedule,
            equivalent_but_distinct.census,
            good.sources[1],
            equivalent_but_distinct.basis,
        )


def test_raw_metric_requires_admission_before_fourier_work() -> None:
    basis = _cartesian_p_metric_basis()
    factor_only_workspace = 1 << 20
    generous = _raw_metric_bundle(
        basis=basis,
        factor_fourier_workspace_bytes=factor_only_workspace,
    )
    assert generous.plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.ADMITTED
    )
    assert (
        generous.plan.components.threaded_fourier_workspace_bytes
        == factor_only_workspace
    )
    with pytest.raises(ValueError, match="Cartesian.*L>0"):
        _raw_metric_result(generous, 1)
    short = _raw_metric_bundle(
        basis=basis,
        factor_fourier_workspace_bytes=factor_only_workspace,
        memory_limit=generous.plan.required_memory_bytes - 1,
    )
    assert short.reference.budget.memory_limit_bytes >= (
        short.reference.plan.required_memory_bytes
    )
    assert short.plan.admission == (
        core._PeriodicCorrelationFactorBuildAdmissionCode.MEMORY_EXCEEDED
    )
    with pytest.raises((ValueError, RuntimeError), match="admission|admitted"):
        _raw_metric_result(short, 1)


def test_raw_metric_diagnostic_caps_precede_native_work() -> None:
    maximum_auxiliary = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_AUXILIARY
    )
    maximum_panel = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_PANEL_VECTORS
    )
    maximum_panel_elements = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_PANEL_ELEMENTS
    )
    maximum_terms = (
        core._PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_ACCUMULATION_TERMS
    )
    assert 2 <= maximum_auxiliary <= 64
    assert 4 <= maximum_panel <= 4096
    assert maximum_panel_elements >= maximum_auxiliary
    assert maximum_terms >= 6 * 3

    too_many_auxiliary = _raw_metric_bundle(
        basis=_many_s_metric_basis(maximum_auxiliary + 1)
    )
    with pytest.raises(ValueError, match="diagnostic|limited|auxiliary"):
        _raw_metric_result(too_many_auxiliary, 1)

    oversized_panel = _raw_metric_bundle(
        reciprocal_block=maximum_panel + 1
    )
    with pytest.raises(ValueError, match="diagnostic|limited|panel"):
        _raw_metric_result(oversized_panel, 1)

    panel_element_overflow = _raw_metric_bundle(
        basis=_many_s_metric_basis(maximum_auxiliary),
        reciprocal_block=maximum_panel_elements // maximum_auxiliary + 1,
    )
    assert panel_element_overflow.census.config.reciprocal_block <= (
        maximum_panel
    )
    with pytest.raises(ValueError, match="diagnostic|limited|panel"):
        _raw_metric_result(panel_element_overflow, 1)

    accumulation_overflow = _raw_metric_bundle(
        basis=_many_s_metric_basis(maximum_auxiliary),
        cutoff=40.0,
        reciprocal_block=maximum_auxiliary,
    )
    assert (
        maximum_auxiliary
        * maximum_auxiliary
        * accumulation_overflow.sources[1].accepted_vector_count
        > maximum_terms
    )
    with pytest.raises(ValueError, match="diagnostic|limited|accumulation"):
        _raw_metric_result(accumulation_overflow, 1)


def test_source_and_diagnostic_binding_are_registered_structurally() -> None:
    cmake = (_ROOT / "cpp/CMakeLists.txt").read_text()
    bindings = (_ROOT / "cpp/src/bindings.cpp").read_text()
    assert "src/periodic_correlation_reciprocal_metric.cpp" in cmake
    assert (
        '#include "periodic_correlation_reciprocal_metric_bindings.cpp"'
        in bindings
    )
    assert "bind_periodic_correlation_reciprocal_metric(m);" in bindings
