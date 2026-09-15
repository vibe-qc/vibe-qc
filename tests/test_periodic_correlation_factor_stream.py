"""Native periodic factor-stream schedule and synthetic transaction gates.

The source exercised here is deliberately nonphysical.  These tests pin only
the constant-space transport contract that a future numerical factor producer
must obey: exact regular-mesh momentum addressing, deterministic tiling,
bounded allocation, provenance identities, and write-once sink semantics.
"""

from __future__ import annotations

import gc
import hashlib
import math
import re
import resource
import struct
import sys
from collections import Counter
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


_CALCULATION_IDENTITY = "1" * 64
_ALLOCATION_IDENTITY = "4" * 64
_RECIPROCAL_LATTICE = np.diag([2.0, 3.0, 5.0])
_U64_MASK = (1 << 64) - 1
_PATTERN_NAME = "synthetic-nonphysical-test-pattern"

_SHAPE_FIELDS = (
    "n_kpoints",
    "n_basis",
    "n_auxiliary",
    "n_ao_pairs",
    "auxiliary_block",
    "ao_pair_block",
    "auxiliary_tile_count",
    "ao_pair_tile_count",
    "tiles_per_k_bra",
    "tiles_per_q",
    "tile_count",
    "logical_element_count",
    "logical_bytes",
    "maximum_tile_element_count",
    "maximum_tile_bytes",
    "two_buffer_workspace_bytes",
)


def _rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def _ceil_div(numerator: int, denominator: int) -> int:
    return numerator // denominator + int(numerator % denominator != 0)


def _unravel(index: int, mesh: tuple[int, int, int]) -> tuple[int, int, int]:
    m2 = index % mesh[2]
    rest = index // mesh[2]
    m1 = rest % mesh[1]
    m0 = rest // mesh[1]
    return m0, m1, m2


def _ravel(address: tuple[int, int, int], mesh: tuple[int, int, int]) -> int:
    return (address[0] * mesh[1] + address[1]) * mesh[2] + address[2]


def _shape_oracle(
    mesh: tuple[int, int, int],
    n_basis: int,
    n_auxiliary: int,
    auxiliary_block: int,
    ao_pair_block: int,
) -> dict[str, int]:
    n_kpoints = math.prod(mesh)
    n_ao_pairs = n_basis * n_basis
    resolved_ao_block = ao_pair_block or n_ao_pairs
    auxiliary_tiles = _ceil_div(n_auxiliary, auxiliary_block)
    ao_pair_tiles = _ceil_div(n_ao_pairs, resolved_ao_block)
    tiles_per_k_bra = ao_pair_tiles * auxiliary_tiles
    tiles_per_q = n_kpoints * tiles_per_k_bra
    tile_count = n_kpoints * tiles_per_q
    logical_elements = n_kpoints**2 * n_auxiliary * n_ao_pairs
    maximum_tile_elements = auxiliary_block * resolved_ao_block
    maximum_tile_bytes = 16 * maximum_tile_elements
    return {
        "n_kpoints": n_kpoints,
        "n_basis": n_basis,
        "n_auxiliary": n_auxiliary,
        "n_ao_pairs": n_ao_pairs,
        "auxiliary_block": auxiliary_block,
        "ao_pair_block": resolved_ao_block,
        "auxiliary_tile_count": auxiliary_tiles,
        "ao_pair_tile_count": ao_pair_tiles,
        "tiles_per_k_bra": tiles_per_k_bra,
        "tiles_per_q": tiles_per_q,
        "tile_count": tile_count,
        "logical_element_count": logical_elements,
        "logical_bytes": 16 * logical_elements,
        "maximum_tile_element_count": maximum_tile_elements,
        "maximum_tile_bytes": maximum_tile_bytes,
        "two_buffer_workspace_bytes": 2 * maximum_tile_bytes,
    }


def _descriptor_oracle(
    sequence_index: int,
    mesh: tuple[int, int, int],
    shape: object,
) -> tuple[int, ...]:
    within_q = sequence_index % shape.tiles_per_q
    q_index = sequence_index // shape.tiles_per_q
    within_k_bra = within_q % shape.tiles_per_k_bra
    k_bra_index = within_q // shape.tiles_per_k_bra
    ao_pair_tile = within_k_bra // shape.auxiliary_tile_count
    auxiliary_tile = within_k_bra % shape.auxiliary_tile_count

    q_address = _unravel(q_index, mesh)
    k_bra_address = _unravel(k_bra_index, mesh)
    raw_ket = tuple(
        k_bra_address[axis] + q_address[axis] for axis in range(3)
    )
    k_ket_address = tuple(
        raw_ket[axis] % mesh[axis] for axis in range(3)
    )
    wrap = tuple(raw_ket[axis] // mesh[axis] for axis in range(3))

    ao_pair_begin = ao_pair_tile * shape.ao_pair_block
    auxiliary_begin = auxiliary_tile * shape.auxiliary_block
    ao_pair_count = min(
        shape.ao_pair_block, shape.n_ao_pairs - ao_pair_begin
    )
    auxiliary_count = min(
        shape.auxiliary_block, shape.n_auxiliary - auxiliary_begin
    )
    return (
        sequence_index,
        q_index,
        k_bra_index,
        _ravel(k_ket_address, mesh),
        *wrap,
        ao_pair_begin,
        ao_pair_count,
        auxiliary_begin,
        auxiliary_count,
        ao_pair_count * auxiliary_count,
    )


def _descriptor_tuple(descriptor: object) -> tuple[int, ...]:
    return (
        descriptor.sequence_index,
        descriptor.q_index,
        descriptor.k_bra_index,
        descriptor.k_ket_index,
        *descriptor.k_ket_reciprocal_wrap,
        descriptor.ao_pair_begin,
        descriptor.ao_pair_count,
        descriptor.auxiliary_begin,
        descriptor.auxiliary_count,
        descriptor.element_count,
    )


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _U64_MASK
    value = (
        (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    ) & _U64_MASK
    value = (
        (value ^ (value >> 27)) * 0x94D049BB133111EB
    ) & _U64_MASK
    return (value ^ (value >> 31)) & _U64_MASK


def _synthetic_value(
    schedule: object,
    q_index: int,
    k_bra_index: int,
    auxiliary: int,
    ao_pair: int,
) -> complex:
    words = struct.unpack(">QQQQ", bytes.fromhex(_source_identity_oracle(schedule)))
    key = 0x6A09E667F3BCC909
    for word, salt in zip(
        words,
        (
            0xBB67AE8584CAA73B,
            0x3C6EF372FE94F82B,
            0xA54FF53A5F1D36F1,
            0x510E527FADE682D1,
        ),
        strict=True,
    ):
        key ^= _splitmix64(word ^ salt)
    key ^= _splitmix64(q_index ^ 0x243F6A8885A308D3)
    key ^= _splitmix64(k_bra_index ^ 0x13198A2E03707344)
    key ^= _splitmix64(auxiliary ^ 0xA4093822299F31D0)
    key ^= _splitmix64(ao_pair ^ 0x082EFA98EC4E6C89)
    key = _splitmix64(key)
    real = math.ldexp((key & 0xFFFFF) + 1, -20)
    imaginary = math.ldexp(((key >> 20) & 0xFFFFF) + 1, -20)
    if (key >> 40) & 1:
        imaginary = -imaginary
    return complex(real, imaginary)


def _payload_oracle(schedule: object, descriptor: object) -> list[complex]:
    return [
        _synthetic_value(
            schedule,
            descriptor.q_index,
            descriptor.k_bra_index,
            auxiliary,
            ao_pair,
        )
        for auxiliary in range(
            descriptor.auxiliary_begin,
            descriptor.auxiliary_begin + descriptor.auxiliary_count,
        )
        for ao_pair in range(
            descriptor.ao_pair_begin,
            descriptor.ao_pair_begin + descriptor.ao_pair_count,
        )
    ]


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
        if value == 0.0:
            value = 0.0
        self._digest.update(struct.pack(">d", value))

    def complex128(self, value: complex) -> None:
        self.binary64(value.real)
        self.binary64(value.imag)

    def finish(self) -> str:
        return self._digest.hexdigest()


def _schedule_identity_oracle(schedule: object) -> str:
    wire = _CanonicalDigest(
        "vibeqc.periodic.correlation.factor-stream.schedule",
        core._PERIODIC_CORRELATION_FACTOR_STREAM_CONTRACT_VERSION,
    )
    wire.u32(core._PERIODIC_CORRELATION_ADMITTED_REFERENCE_CONTRACT_VERSION)
    wire.u32(core._PERIODIC_CORRELATION_STREAMED_RESOURCE_CONTRACT_VERSION)
    wire.u32(core._PERIODIC_MEAN_FIELD_STATE_DIGEST_VERSION)
    wire.string(schedule.state_identity_sha256)
    wire.string(schedule.calculation_identity)
    wire.string(schedule.allocation_identity)
    for value in schedule.mesh:
        wire.i32(value)
    for value in schedule.is_shift:
        wire.i32(value)
    for value in (
        "gamma-centered-full-mesh-unreduced",
        "q,k-bra,ao-pair-tile,auxiliary-tile",
        "ao-pair=mu*n-basis+nu;nu-fastest",
        "synthetic-payload[opaque-row,ao-pair];ao-pair-fastest",
        "complex-ieee754-binary64",
        "q-index=regular-mesh-modular-residue;physical-q-gauge-undefined",
        "q=k-ket-k-bra;k-ket=k-bra+q-mod-G",
    ):
        wire.string(value)
    for field in _SHAPE_FIELDS:
        wire.u64(getattr(schedule.shape, field))
    return wire.finish()


def _source_identity_oracle(schedule: object) -> str:
    wire = _CanonicalDigest(
        "vibeqc.periodic.correlation.factor-stream.synthetic-source",
        core._PERIODIC_CORRELATION_SYNTHETIC_FACTOR_PATTERN_VERSION,
    )
    wire.string(schedule.schedule_identity_sha256)
    wire.string(_PATTERN_NAME)
    wire.string("splitmix64-source-bound-to-exact-binary-fractions")
    return wire.finish()


def _payload_identity_oracle(schedule: object) -> str:
    wire = _CanonicalDigest(
        "vibeqc.periodic.correlation.factor-stream.synthetic-payload",
        core._PERIODIC_CORRELATION_SYNTHETIC_FACTOR_PATTERN_VERSION,
    )
    wire.string(schedule.schedule_identity_sha256)
    wire.string(_source_identity_oracle(schedule))
    wire.string(_PATTERN_NAME)
    wire.u64(schedule.shape.tile_count)
    wire.u64(schedule.shape.logical_element_count)
    wire.u64(schedule.shape.logical_bytes)
    for sequence_index in range(schedule.shape.tile_count):
        descriptor = schedule.descriptor(sequence_index)
        wire.u64(descriptor.sequence_index)
        wire.u64(descriptor.q_index)
        wire.u64(descriptor.k_bra_index)
        wire.u64(descriptor.k_ket_index)
        for value in descriptor.k_ket_reciprocal_wrap:
            wire.i32(value)
        wire.u64(descriptor.ao_pair_begin)
        wire.u64(descriptor.ao_pair_count)
        wire.u64(descriptor.auxiliary_begin)
        wire.u64(descriptor.auxiliary_count)
        wire.u64(descriptor.element_count)
        for value in _payload_oracle(schedule, descriptor):
            wire.complex128(value)
    return wire.finish()


def _payload_wire_bytes_oracle(schedule: object) -> int:
    def encoded_string_size(value: str) -> int:
        return 8 + len(value.encode("utf-8"))

    return (
        encoded_string_size(
            "vibeqc.periodic.correlation.factor-stream.synthetic-payload"
        )
        + 4
        + 2 * (8 + 64)
        + encoded_string_size(_PATTERN_NAME)
        + 3 * 8
        + 84 * schedule.shape.tile_count
        + schedule.shape.logical_bytes
    )


def _receipt_identity_oracle(receipt: object) -> str:
    wire = _CanonicalDigest(
        "vibeqc.periodic.correlation.factor-stream.synthetic-receipt",
        core._PERIODIC_CORRELATION_SYNTHETIC_FACTOR_TRANSACTION_VERSION,
    )
    wire.u32(receipt.transaction_contract_version)
    wire.u32(receipt.synthetic_pattern_version)
    wire.string(_PATTERN_NAME)
    wire.u32(1 if receipt.synthetic_nonphysical else 0)
    wire.string(receipt.schedule_identity_sha256)
    wire.string(receipt.source_identity_sha256)
    wire.string(receipt.payload_identity_sha256)
    wire.u64(receipt.committed_tile_count)
    wire.u64(receipt.committed_element_count)
    wire.u64(receipt.committed_logical_bytes)
    return wire.finish()


def _translation_pair_count(mesh: tuple[int, int, int]) -> int:
    n_cells = math.prod(mesh)
    self_inverse = math.prod(2 if extent % 2 == 0 else 1 for extent in mesh)
    return (n_cells + self_inverse) // 2


def _make_state(
    mesh: tuple[int, int, int],
    n_basis: int,
    *,
    shift: tuple[int, int, int] = (0, 0, 0),
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
    data.reference_kind = core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    data.normalization = (
        core._PeriodicMeanFieldNormalizationConvention
        .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS
    )
    data.periodic_dimension = 3
    data.mesh = mesh
    data.is_shift = shift
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
            [
                (address[axis] + 0.5 * shift[axis]) / mesh[axis]
                for axis in range(3)
            ]
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


def _make_dimensions(
    mesh: tuple[int, int, int],
    n_basis: int,
    n_auxiliary: int,
    auxiliary_block: int,
    ao_pair_block: int,
    *,
    shift: tuple[int, int, int] = (0, 0, 0),
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
    dimensions.is_shift = shift
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


def _make_budget(*, workers: int = 1) -> object:
    budget = core._PeriodicCorrelationResourceBudget()
    budget.memory_limit_bytes = 1024**3
    budget.scratch_limit_bytes = 1024**3
    budget.mpi_ranks = 1
    budget.workers_per_rank = workers
    return budget


def _make_admitted_reference(
    mesh: tuple[int, int, int] = (2, 1, 1),
    n_basis: int = 2,
    n_auxiliary: int = 3,
    auxiliary_block: int = 2,
    ao_pair_block: int = 3,
    *,
    shift: tuple[int, int, int] = (0, 0, 0),
    calculation_identity: str = _CALCULATION_IDENTITY,
    allocation_identity: str = _ALLOCATION_IDENTITY,
    reference_energy_per_cell: float = -5.0,
    workers: int = 1,
) -> object:
    state = _make_state(
        mesh,
        n_basis,
        shift=shift,
        calculation_identity=calculation_identity,
        reference_energy_per_cell=reference_energy_per_cell,
    )
    dimensions = _make_dimensions(
        mesh,
        n_basis,
        n_auxiliary,
        auxiliary_block,
        ao_pair_block,
        shift=shift,
        calculation_identity=calculation_identity,
        allocation_identity=allocation_identity,
    )
    return core._make_periodic_correlation_admitted_reference(
        state, dimensions, _make_budget(workers=workers)
    )


def _make_schedule(**kwargs: object) -> object:
    reference = _make_admitted_reference(**kwargs)
    return core._make_periodic_correlation_factor_stream_schedule(reference)


def _tile(schedule: object, sequence_index: int) -> np.ndarray:
    descriptor = schedule.descriptor(sequence_index)
    return core._periodic_correlation_synthetic_factor_tile(
        schedule, descriptor, schedule.shape.maximum_tile_bytes
    )


def _caps(schedule: object) -> object:
    caps = core._PeriodicCorrelationSyntheticFactorExecutionCaps()
    caps.max_tile_bytes = schedule.shape.maximum_tile_bytes
    caps.max_workspace_bytes = schedule.shape.two_buffer_workspace_bytes
    caps.max_tile_count = schedule.shape.tile_count
    caps.max_logical_bytes = schedule.shape.logical_bytes
    return caps


@pytest.mark.parametrize(
    ("mesh", "n_basis", "n_auxiliary", "auxiliary_block", "ao_pair_block"),
    [
        ((1, 1, 1), 2, 3, 2, 0),
        ((2, 1, 1), 2, 3, 2, 3),
        ((2, 2, 1), 3, 5, 4, 4),
        ((3, 2, 1), 3, 5, 1, 8),
    ],
    ids=["single-k-full-pair", "ragged-both", "two-dimensional", "unit-aux"],
)
def test_shape_estimator_matches_independent_integer_oracle(
    mesh: tuple[int, int, int],
    n_basis: int,
    n_auxiliary: int,
    auxiliary_block: int,
    ao_pair_block: int,
) -> None:
    shape = core._estimate_periodic_correlation_factor_stream_shape(
        mesh, n_basis, n_auxiliary, auxiliary_block, ao_pair_block
    )
    expected = _shape_oracle(
        mesh, n_basis, n_auxiliary, auxiliary_block, ao_pair_block
    )
    assert {field: getattr(shape, field) for field in _SHAPE_FIELDS} == expected


def test_ragged_schedule_pins_q_k_pair_aux_order_and_wraps() -> None:
    schedule = _make_schedule()
    shape = schedule.shape

    assert shape.tile_count == 16
    assert [_descriptor_tuple(schedule.descriptor(index)) for index in range(4)] == [
        (0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 6),
        (1, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 3),
        (2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 2),
        (3, 0, 0, 0, 0, 0, 0, 3, 1, 2, 1, 1),
    ]
    assert _descriptor_tuple(schedule.descriptor(4)) == (
        4, 0, 1, 1, 0, 0, 0, 0, 3, 0, 2, 6
    )
    assert _descriptor_tuple(schedule.descriptor(8)) == (
        8, 1, 0, 1, 0, 0, 0, 0, 3, 0, 2, 6
    )
    assert _descriptor_tuple(schedule.descriptor(12)) == (
        12, 1, 1, 0, 1, 0, 0, 0, 3, 0, 2, 6
    )

    with pytest.raises(IndexError, match="sequence index"):
        schedule.descriptor(shape.tile_count)


def test_all_small_meshes_match_independent_modular_and_tile_oracle() -> None:
    for mesh in product(range(1, 4), repeat=3):
        schedule = _make_schedule(mesh=mesh)
        coverage: Counter[tuple[int, int, int, int]] = Counter()
        k_pairs: set[tuple[int, int, int]] = set()
        for sequence_index in range(schedule.shape.tile_count):
            descriptor = schedule.descriptor(sequence_index)
            assert _descriptor_tuple(descriptor) == _descriptor_oracle(
                sequence_index, mesh, schedule.shape
            )
            if (
                descriptor.ao_pair_begin == 0
                and descriptor.auxiliary_begin == 0
            ):
                k_pairs.add(
                    (
                        descriptor.q_index,
                        descriptor.k_bra_index,
                        descriptor.k_ket_index,
                    )
                )
            for auxiliary in range(
                descriptor.auxiliary_begin,
                descriptor.auxiliary_begin + descriptor.auxiliary_count,
            ):
                for ao_pair in range(
                    descriptor.ao_pair_begin,
                    descriptor.ao_pair_begin + descriptor.ao_pair_count,
                ):
                    coverage[
                        (
                            descriptor.q_index,
                            descriptor.k_bra_index,
                            auxiliary,
                            ao_pair,
                        )
                    ] += 1

        n_kpoints = math.prod(mesh)
        assert len(k_pairs) == n_kpoints**2
        assert len(coverage) == schedule.shape.logical_element_count
        assert set(coverage.values()) == {1}


def test_asymmetric_mesh_pins_transfer_sign_and_reciprocal_wrap() -> None:
    schedule = _make_schedule(mesh=(3, 2, 1))
    shape = schedule.shape
    q_index = _ravel((1, 1, 0), (3, 2, 1))
    k_bra_index = _ravel((2, 1, 0), (3, 2, 1))
    sequence_index = (
        q_index * shape.tiles_per_q
        + k_bra_index * shape.tiles_per_k_bra
    )
    descriptor = schedule.descriptor(sequence_index)

    assert descriptor.k_ket_index == _ravel((0, 0, 0), (3, 2, 1))
    assert descriptor.k_ket_reciprocal_wrap == [1, 1, 0]


def test_ao_pair_and_payload_offsets_are_exact_and_checked() -> None:
    schedule = _make_schedule(n_basis=3, ao_pair_block=4)
    for mu in range(3):
        for nu in range(3):
            ao_pair = 3 * mu + nu
            assert schedule.ao_pair_index(mu, nu) == ao_pair
            assert schedule.ao_pair_indices(ao_pair) == [mu, nu]

    descriptor = schedule.descriptor(2)
    for auxiliary in range(
        descriptor.auxiliary_begin,
        descriptor.auxiliary_begin + descriptor.auxiliary_count,
    ):
        for ao_pair in range(
            descriptor.ao_pair_begin,
            descriptor.ao_pair_begin + descriptor.ao_pair_count,
        ):
            expected = (
                (auxiliary - descriptor.auxiliary_begin)
                * descriptor.ao_pair_count
                + ao_pair
                - descriptor.ao_pair_begin
            )
            assert schedule.payload_offset(
                descriptor, auxiliary, ao_pair
            ) == expected

    with pytest.raises(IndexError):
        schedule.ao_pair_index(3, 0)
    with pytest.raises(IndexError):
        schedule.ao_pair_indices(9)
    with pytest.raises(IndexError):
        schedule.payload_offset(
            descriptor,
            descriptor.auxiliary_begin + descriptor.auxiliary_count,
            descriptor.ao_pair_begin,
        )


def test_synthetic_tiles_match_independent_splitmix64_pattern() -> None:
    schedule = _make_schedule()
    for sequence_index in range(schedule.shape.tile_count):
        descriptor = schedule.descriptor(sequence_index)
        payload = _tile(schedule, sequence_index)
        np.testing.assert_array_equal(
            payload, _payload_oracle(schedule, descriptor)
        )
        assert payload.dtype == np.complex128
        assert payload.flags.c_contiguous
        assert len(payload) == descriptor.element_count
        assert all(math.isfinite(value.real) for value in payload)
        assert all(math.isfinite(value.imag) for value in payload)

    descriptor = schedule.descriptor(0)
    with pytest.raises(RuntimeError, match="binding cap"):
        core._periodic_correlation_synthetic_factor_tile(
            schedule, descriptor, 16 * descriptor.element_count - 1
        )


def test_schedule_and_source_identities_match_independent_wire_format() -> None:
    schedule = _make_schedule()
    transaction = core._make_periodic_correlation_synthetic_factor_transaction(
        schedule
    )

    assert re.fullmatch(r"[0-9a-f]{64}", schedule.schedule_identity_sha256)
    assert schedule.schedule_identity_sha256 == _schedule_identity_oracle(schedule)
    assert transaction.source_identity_sha256 == _source_identity_oracle(schedule)


def test_synthetic_payload_is_bound_to_its_source_schedule() -> None:
    source_schedule = _make_schedule()
    sink_schedule = _make_schedule(allocation_identity="5" * 64)
    source_descriptor = source_schedule.descriptor(0)
    sink_descriptor = sink_schedule.descriptor(0)
    source_payload = _tile(source_schedule, 0)
    sink_payload = _tile(sink_schedule, 0)

    assert _descriptor_tuple(source_descriptor) == _descriptor_tuple(
        sink_descriptor
    )
    assert source_schedule.schedule_identity_sha256 != (
        sink_schedule.schedule_identity_sha256
    )
    assert not np.array_equal(source_payload, sink_payload)

    transaction = core._make_periodic_correlation_synthetic_factor_transaction(
        sink_schedule
    )
    with pytest.raises(ValueError, match="payload value"):
        transaction.accept(
            source_descriptor,
            source_payload,
            16 * source_descriptor.element_count,
        )
    assert transaction.state == (
        core._PeriodicCorrelationSyntheticTransactionState.FAILED
    )


def test_synthetic_payload_wire_length_is_exact_and_sha_bounded() -> None:
    schedule = _make_schedule()
    expected = _payload_wire_bytes_oracle(schedule)
    assert expected == (
        281 + 84 * schedule.shape.tile_count + schedule.shape.logical_bytes
    )
    assert core._estimate_periodic_correlation_synthetic_payload_wire_bytes(
        schedule.shape
    ) == expected

    oversized = core._estimate_periodic_correlation_factor_stream_shape(
        (1, 1, 1), 1 << 28, 1, 1, 1
    )
    assert oversized.logical_bytes == 1 << 60
    with pytest.raises((ValueError, RuntimeError), match="SHA-256"):
        core._estimate_periodic_correlation_synthetic_payload_wire_bytes(
            oversized
        )


def test_v1_identity_wire_known_answer_tuple() -> None:
    schedule = _make_schedule()
    transaction = core._make_periodic_correlation_synthetic_factor_transaction(
        schedule
    )
    receipt = core._execute_periodic_correlation_synthetic_factor_stream(
        schedule, _caps(schedule)
    )

    assert (
        schedule.schedule_identity_sha256,
        transaction.source_identity_sha256,
        receipt.payload_identity_sha256,
        receipt.receipt_identity_sha256,
    ) == (
        "e9de9061af86a8a67568651881765bfd1458a4fcbffcd76cb4327c502ba13e33",
        "0b5a14f9d77f56466c1ee138bd38efb29422cfc7e750e38725e28f7ebc4a5df4",
        "88d16a2e9975aa9edef4ba52b4a6ee6548f42993c52902294ac7e113cdfbcd8a",
        "0816208ce7baae34aed0da204c9e22733d10eaa56ab1645f5f236351fac19fc5",
    )


def test_manual_accept_requires_bounded_exact_numpy_storage() -> None:
    schedule = _make_schedule()
    descriptor = schedule.descriptor(0)

    class ExplodingIterable:
        iterated = False

        def __iter__(self) -> object:
            self.iterated = True
            raise AssertionError("binding must not consume an iterable")

    iterable = ExplodingIterable()
    rejected_payloads = (
        iterable,
        _payload_oracle(schedule, descriptor),
        np.arange(descriptor.element_count, dtype=np.float64),
        np.zeros(2 * descriptor.element_count, dtype=np.complex128)[::2],
    )
    for payload in rejected_payloads:
        transaction = (
            core._make_periodic_correlation_synthetic_factor_transaction(
                schedule
            )
        )
        with pytest.raises(ValueError):
            transaction.accept(
                descriptor, payload, 16 * descriptor.element_count
            )
        assert transaction.state == (
            core._PeriodicCorrelationSyntheticTransactionState.OPEN
        )
        transaction.abort()

    assert iterable.iterated is False


def test_schedule_identity_is_stable_canonical_and_provenance_sensitive() -> None:
    baseline = _make_schedule()
    repeated = _make_schedule()
    full_pair_alias = _make_schedule(ao_pair_block=4)
    worker_changed = core._make_periodic_correlation_factor_stream_schedule(
        _make_admitted_reference(workers=3)
    )
    allocation_changed = _make_schedule(allocation_identity="5" * 64)
    payload_changed = _make_schedule(reference_energy_per_cell=-5.25)
    mesh_changed = _make_schedule(mesh=(1, 2, 1))
    block_changed = _make_schedule(auxiliary_block=1)

    assert baseline.schedule_identity_sha256 == repeated.schedule_identity_sha256
    assert baseline.schedule_identity_sha256 != full_pair_alias.schedule_identity_sha256
    zero_pair_block = _make_schedule(ao_pair_block=0)
    assert zero_pair_block.schedule_identity_sha256 == (
        full_pair_alias.schedule_identity_sha256
    )
    assert baseline.schedule_identity_sha256 == worker_changed.schedule_identity_sha256
    for changed in (
        allocation_changed,
        payload_changed,
        mesh_changed,
        block_changed,
    ):
        assert baseline.schedule_identity_sha256 != changed.schedule_identity_sha256


@pytest.mark.parametrize(
    "shift",
    [shift for shift in product((0, 1), repeat=3) if shift != (0, 0, 0)],
)
def test_factor_schedule_rejects_every_shifted_mesh(
    shift: tuple[int, int, int],
) -> None:
    reference = _make_admitted_reference(shift=shift)
    with pytest.raises(ValueError, match="Gamma-centered"):
        core._make_periodic_correlation_factor_stream_schedule(reference)


def test_schedule_retains_the_immutable_mean_field_state() -> None:
    reference = _make_admitted_reference()
    schedule = core._make_periodic_correlation_factor_stream_schedule(reference)
    wrapped_state = schedule.state
    assert schedule.state_identity_sha256 == reference.state.state_identity_sha256

    del reference
    gc.collect()
    assert schedule.state.fock(1).shape == (2, 2)

    del schedule
    gc.collect()
    assert wrapped_state.correlated_occupied_mask(0) == [1, 0]


def test_manual_transaction_commits_once_and_matches_all_identity_wires() -> None:
    schedule = _make_schedule()
    transaction = core._make_periodic_correlation_synthetic_factor_transaction(
        schedule
    )
    open_state = core._PeriodicCorrelationSyntheticTransactionState.OPEN
    committed_state = core._PeriodicCorrelationSyntheticTransactionState.COMMITTED

    assert transaction.contract_version == (
        core._PERIODIC_CORRELATION_SYNTHETIC_FACTOR_TRANSACTION_VERSION
    )
    assert transaction.state == open_state
    assert transaction.next_sequence_index == 0
    assert transaction.accepted_tile_count == 0
    assert transaction.accepted_element_count == 0

    accepted_elements = 0
    for sequence_index in range(schedule.shape.tile_count):
        descriptor = schedule.descriptor(sequence_index)
        transaction.accept(
            descriptor,
            np.asarray(
                _payload_oracle(schedule, descriptor), dtype=np.complex128
            ),
            16 * descriptor.element_count,
        )
        accepted_elements += descriptor.element_count
        assert transaction.state == open_state
        assert transaction.next_sequence_index == sequence_index + 1
        assert transaction.accepted_tile_count == sequence_index + 1
        assert transaction.accepted_element_count == accepted_elements

    receipt = transaction.commit()
    repeated_receipt = transaction.commit()
    transaction.abort()
    assert transaction.state == committed_state
    assert receipt.receipt_identity_sha256 == repeated_receipt.receipt_identity_sha256
    assert receipt.transaction_contract_version == transaction.contract_version
    assert receipt.synthetic_pattern_version == (
        core._PERIODIC_CORRELATION_SYNTHETIC_FACTOR_PATTERN_VERSION
    )
    assert receipt.payload_kind == (
        core._PeriodicCorrelationFactorPayloadKind
        .SYNTHETIC_NONPHYSICAL_TEST_PATTERN
    )
    assert receipt.synthetic_nonphysical is True
    assert receipt.schedule_identity_sha256 == schedule.schedule_identity_sha256
    assert receipt.source_identity_sha256 == transaction.source_identity_sha256
    assert receipt.committed_tile_count == schedule.shape.tile_count
    assert receipt.committed_element_count == schedule.shape.logical_element_count
    assert receipt.committed_logical_bytes == schedule.shape.logical_bytes
    assert receipt.payload_identity_sha256 == _payload_identity_oracle(schedule)
    assert receipt.receipt_identity_sha256 == _receipt_identity_oracle(receipt)


def test_explicit_abort_is_idempotent_and_terminal() -> None:
    schedule = _make_schedule()
    transaction = core._make_periodic_correlation_synthetic_factor_transaction(
        schedule
    )
    aborted_state = core._PeriodicCorrelationSyntheticTransactionState.ABORTED

    transaction.abort()
    transaction.abort()
    assert transaction.state == aborted_state
    assert transaction.accepted_tile_count == 0
    with pytest.raises(RuntimeError):
        descriptor = schedule.descriptor(0)
        transaction.accept(
            descriptor,
            _tile(schedule, 0),
            16 * descriptor.element_count,
        )
    with pytest.raises(RuntimeError):
        transaction.commit()
    assert transaction.state == aborted_state


@pytest.mark.parametrize(
    "failure", ["out-of-order", "short", "corrupt", "cap"]
)
def test_accept_validation_failures_are_permanent_and_write_once(
    failure: str,
) -> None:
    schedule = _make_schedule()
    transaction = core._make_periodic_correlation_synthetic_factor_transaction(
        schedule
    )
    descriptor = schedule.descriptor(0)
    payload = _tile(schedule, 0)
    max_payload_bytes = schedule.shape.maximum_tile_bytes
    if failure == "out-of-order":
        descriptor = schedule.descriptor(1)
    elif failure == "short":
        payload = payload[:-1]
    elif failure == "corrupt":
        payload[0] += 1.0
    else:
        max_payload_bytes = 16 * descriptor.element_count - 1

    with pytest.raises((ValueError, RuntimeError)):
        transaction.accept(descriptor, payload, max_payload_bytes)
    assert transaction.state == (
        core._PeriodicCorrelationSyntheticTransactionState.FAILED
    )
    assert transaction.next_sequence_index == 0
    assert transaction.accepted_tile_count == 0
    assert transaction.accepted_element_count == 0
    transaction.abort()
    assert transaction.state == (
        core._PeriodicCorrelationSyntheticTransactionState.FAILED
    )
    with pytest.raises(RuntimeError):
        descriptor = schedule.descriptor(0)
        transaction.accept(
            descriptor,
            _tile(schedule, 0),
            16 * descriptor.element_count,
        )


def test_duplicate_tile_and_early_commit_fail_closed() -> None:
    schedule = _make_schedule()
    duplicate = core._make_periodic_correlation_synthetic_factor_transaction(
        schedule
    )
    descriptor = schedule.descriptor(0)
    duplicate.accept(
        descriptor,
        _tile(schedule, 0),
        16 * descriptor.element_count,
    )
    with pytest.raises((ValueError, RuntimeError)):
        duplicate.accept(
            descriptor,
            _tile(schedule, 0),
            16 * descriptor.element_count,
        )
    assert duplicate.state == core._PeriodicCorrelationSyntheticTransactionState.FAILED
    assert duplicate.accepted_tile_count == 1

    early = core._make_periodic_correlation_synthetic_factor_transaction(schedule)
    with pytest.raises(RuntimeError):
        early.commit()
    assert early.state == core._PeriodicCorrelationSyntheticTransactionState.FAILED


def test_one_shot_executor_obeys_exact_caps_and_matches_manual_payload() -> None:
    schedule = _make_schedule()
    receipt = core._execute_periodic_correlation_synthetic_factor_stream(
        schedule, _caps(schedule)
    )

    assert receipt.committed_tile_count == schedule.shape.tile_count
    assert receipt.committed_element_count == schedule.shape.logical_element_count
    assert receipt.committed_logical_bytes == schedule.shape.logical_bytes
    assert receipt.payload_identity_sha256 == _payload_identity_oracle(schedule)
    assert receipt.receipt_identity_sha256 == _receipt_identity_oracle(receipt)


@pytest.mark.parametrize(
    "field",
    [
        "max_tile_bytes",
        "max_workspace_bytes",
        "max_tile_count",
        "max_logical_bytes",
    ],
)
def test_one_shot_executor_checks_every_cap_before_allocation(field: str) -> None:
    schedule = _make_schedule()
    caps = _caps(schedule)
    setattr(caps, field, getattr(caps, field) - 1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        core._execute_periodic_correlation_synthetic_factor_stream(
            schedule, caps
        )

    missing = core._PeriodicCorrelationSyntheticFactorExecutionCaps()
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        core._execute_periodic_correlation_synthetic_factor_stream(
            schedule, missing
        )


@pytest.mark.parametrize(
    (
        "mesh",
        "n_basis",
        "n_auxiliary",
        "ao_pair_block",
        "expected_tiles",
        "expected_elements",
        "expected_bytes",
        "expected_maximum_tile_bytes",
        "expected_two_buffer_bytes",
    ),
    [
        (
            (8, 8, 8), 37, 111, 0,
            1_835_008, 39_835_140_096, 637_362_241_536,
            350_464, 700_928,
        ),
        (
            (8, 8, 8), 36, 108, 0,
            1_835_008, 36_691_771_392, 587_068_342_272,
            331_776, 663_552,
        ),
        (
            (6, 6, 6), 196, 588, 0,
            1_726_272, 1_053_894_094_848, 16_862_305_517_568,
            9_834_496, 19_668_992,
        ),
        (
            (8, 8, 8), 37, 111, 1024,
            3_670_016, 39_835_140_096, 637_362_241_536,
            262_144, 524_288,
        ),
        (
            (8, 8, 8), 36, 108, 1024,
            3_670_016, 36_691_771_392, 587_068_342_272,
            262_144, 524_288,
        ),
        (
            (6, 6, 6), 196, 588, 1024,
            65_598_336, 1_053_894_094_848, 16_862_305_517_568,
            262_144, 524_288,
        ),
    ],
    ids=[
        "eight-cubed-nao37-full-pair",
        "eight-cubed-nao36-full-pair",
        "six-cubed-nao196-full-pair",
        "eight-cubed-nao37-bounded-pair",
        "eight-cubed-nao36-bounded-pair",
        "six-cubed-nao196-bounded-pair",
    ],
)
def test_target_shapes_are_counted_without_states_tiles_or_payloads(
    mesh: tuple[int, int, int],
    n_basis: int,
    n_auxiliary: int,
    ao_pair_block: int,
    expected_tiles: int,
    expected_elements: int,
    expected_bytes: int,
    expected_maximum_tile_bytes: int,
    expected_two_buffer_bytes: int,
) -> None:
    before = _rss_bytes()
    shape = core._estimate_periodic_correlation_factor_stream_shape(
        mesh, n_basis, n_auxiliary, 16, ao_pair_block
    )
    after = _rss_bytes()

    assert shape.n_kpoints == math.prod(mesh)
    assert shape.tile_count == expected_tiles
    assert shape.logical_element_count == expected_elements
    assert shape.logical_bytes == expected_bytes
    assert shape.maximum_tile_bytes == expected_maximum_tile_bytes
    assert shape.two_buffer_workspace_bytes == expected_two_buffer_bytes
    assert after - before < 64 * 1024 * 1024


@pytest.mark.parametrize(
    "arguments",
    [
        ((0, 1, 1), 2, 3, 2, 3),
        ((1, 1, 1), 0, 3, 2, 0),
        ((1, 1, 1), 2, 0, 1, 0),
        ((1, 1, 1), 2, 3, 0, 0),
        ((1, 1, 1), 2, 3, 4, 0),
        ((1, 1, 1), 2, 3, 2, 5),
    ],
)
def test_shape_estimator_rejects_invalid_extents_before_work(
    arguments: tuple[object, ...],
) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        core._estimate_periodic_correlation_factor_stream_shape(*arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ((1_073_741_823, 5, 1), 2, 3, 2, 3),
        ((1, 1, 1), 1 << 32, 1, 1, 1),
    ],
    ids=["k-pair-square", "ao-pair-square"],
)
def test_shape_estimator_rejects_uint64_overflow_without_allocation(
    arguments: tuple[object, ...],
) -> None:
    with pytest.raises(OverflowError):
        core._estimate_periodic_correlation_factor_stream_shape(*arguments)
