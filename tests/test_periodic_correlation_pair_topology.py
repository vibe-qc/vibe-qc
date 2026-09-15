"""Exact translation-pair topology gates for native periodic correlation.

The numerical executor does not exist yet.  These tests pin the smaller
structural contract that must precede it: every unordered correlated occupied
pair in the finite Born-von Karman cell belongs to exactly one reference-cell
translation orbit.  The Python oracle below constructs those full placed-pair
orbits directly.  It deliberately does not use ``_RegularKMesh``, the native
count formula, or the older Python local-correlation pair planner.
"""

from __future__ import annotations

import gc
import hashlib
import math
import re
import resource
import struct
import sys
from itertools import combinations_with_replacement, product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


_CALCULATION_IDENTITY = "1" * 64
_ALLOCATION_IDENTITY = "4" * 64
_RECIPROCAL_LATTICE = np.diag([2.0, 3.0, 5.0])
_HOME_CELL = (0, 0, 0)

Cell = tuple[int, int, int]
PlacedPair = tuple[int, int]
TopologyRow = tuple[int, int, int, int]


def _rss_bytes() -> int:
    """Return peak RSS in bytes on both macOS and Linux."""

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def _cells(mesh: tuple[int, int, int]) -> tuple[Cell, ...]:
    """Finite translations in independent last-axis-fast product order."""

    return tuple(product(*(range(extent) for extent in mesh)))


def _linear_cell_index(cell: Cell, mesh: tuple[int, int, int]) -> int:
    return (cell[0] * mesh[1] + cell[1]) * mesh[2] + cell[2]


def _add_cells(left: Cell, right: Cell, mesh: tuple[int, int, int]) -> Cell:
    return tuple(
        (left[axis] + right[axis]) % mesh[axis]
        for axis in range(3)
    )  # type: ignore[return-value]


def _canonical_pair(left: int, right: int) -> PlacedPair:
    return (left, right) if left <= right else (right, left)


def _brute_force_pair_oracle(
    mesh: tuple[int, int, int], n_home_occupied: int
) -> tuple[tuple[TopologyRow, ...], dict[tuple[int, int, int], frozenset[PlacedPair]]]:
    """Partition the complete placed-pair set by explicit translations.

    A placed occupied index is ``cell_linear * B + home_orbital``.  Starting
    from every still-unassigned unordered pair, the oracle applies every
    finite translation to both endpoints.  Only after obtaining the full
    orbit does it choose the lexicographically first description with one
    endpoint in the home cell and ordered home-orbital labels.  Thus the
    oracle derives the ``L`` versus ``-L`` choice and stabilizer multiplicity
    from set membership rather than reproducing the native counting formula.
    """

    cells = _cells(mesh)
    cell_to_index = {cell: index for index, cell in enumerate(cells)}

    def encode(cell: Cell, orbital: int) -> int:
        return cell_to_index[cell] * n_home_occupied + orbital

    def decode(index: int) -> tuple[Cell, int]:
        cell_index, orbital = divmod(index, n_home_occupied)
        return cells[cell_index], orbital

    def translate(pair: PlacedPair, shift: Cell) -> PlacedPair:
        left_cell, left_orbital = decode(pair[0])
        right_cell, right_orbital = decode(pair[1])
        return _canonical_pair(
            encode(_add_cells(left_cell, shift, mesh), left_orbital),
            encode(_add_cells(right_cell, shift, mesh), right_orbital),
        )

    placed_occupied = len(cells) * n_home_occupied
    all_pairs = set(
        combinations_with_replacement(range(placed_occupied), 2)
    )
    unassigned = set(all_pairs)
    by_descriptor: dict[
        tuple[int, int, int], frozenset[PlacedPair]
    ] = {}

    while unassigned:
        seed = min(unassigned)
        orbit = frozenset(translate(seed, shift) for shift in cells)
        assert orbit <= all_pairs
        assert orbit <= unassigned
        unassigned.difference_update(orbit)

        anchored: set[tuple[int, int, int]] = set()
        for left, right in orbit:
            left_cell, left_orbital = decode(left)
            right_cell, right_orbital = decode(right)
            if left_cell == _HOME_CELL and left_orbital <= right_orbital:
                anchored.add(
                    (
                        left_orbital,
                        right_orbital,
                        _linear_cell_index(right_cell, mesh),
                    )
                )
            if right_cell == _HOME_CELL and right_orbital <= left_orbital:
                anchored.add(
                    (
                        right_orbital,
                        left_orbital,
                        _linear_cell_index(left_cell, mesh),
                    )
                )

        assert anchored
        descriptor = min(anchored)
        assert descriptor not in by_descriptor
        by_descriptor[descriptor] = orbit

    assert sum(map(len, by_descriptor.values())) == len(all_pairs)
    rows = tuple(
        (*descriptor, len(orbit))
        for descriptor, orbit in sorted(by_descriptor.items())
    )
    return rows, by_descriptor


def _make_state(
    mesh: tuple[int, int, int],
    n_home_occupied: int,
    *,
    shift: tuple[int, int, int] = (0, 0, 0),
    calculation_identity: str = _CALCULATION_IDENTITY,
    reference_energy_per_cell: float = -5.0,
) -> object:
    n_cells = math.prod(mesh)
    n_orbitals = n_home_occupied + 1
    orbital_energies = np.concatenate(
        (np.linspace(-2.0, -1.0, n_home_occupied), [0.5])
    )
    occupations = np.concatenate(
        (np.full(n_home_occupied, 2.0), [0.0])
    )
    coefficients = np.eye(n_orbitals, dtype=np.complex128)
    overlap = np.eye(n_orbitals, dtype=np.complex128)
    fock = np.diag(orbital_energies).astype(np.complex128)
    correlated = [1] * n_home_occupied + [0]
    virtual = [0] * n_home_occupied + [1]

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
    data.n_basis = n_orbitals
    data.n_effective_orbitals = n_orbitals
    data.electrons_per_cell = 2 * n_home_occupied
    data.reference_energy_per_cell = reference_energy_per_cell
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = (
        core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    )

    for cell in _cells(mesh):
        fractional = np.asarray(
            [
                (cell[axis] + 0.5 * shift[axis]) / mesh[axis]
                for axis in range(3)
            ],
            dtype=float,
        )
        data.add_kpoint(
            _RECIPROCAL_LATTICE @ fractional,
            1.0 / n_cells,
            overlap,
            fock,
            coefficients,
            orbital_energies,
            occupations,
            [0] * n_orbitals,
            correlated,
            virtual,
        )
    return core._make_periodic_restricted_mean_field_state(data)


def _make_dimensions(
    mesh: tuple[int, int, int],
    n_home_occupied: int,
    *,
    shift: tuple[int, int, int] = (0, 0, 0),
    calculation_identity: str = _CALCULATION_IDENTITY,
    allocation_identity: str = _ALLOCATION_IDENTITY,
    expected_pair_candidate_count: int | None = None,
    pair_domain_metadata_upper_bytes: int | None = None,
) -> object:
    n_cells = math.prod(mesh)
    n_orbitals = n_home_occupied + 1
    if expected_pair_candidate_count is None:
        expected_pair_candidate_count = len(
            _brute_force_pair_oracle(mesh, n_home_occupied)[0]
        )
    if pair_domain_metadata_upper_bytes is None:
        pair_domain_metadata_upper_bytes = (
            8 * n_home_occupied + 64 * expected_pair_candidate_count
        )

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
    dimensions.n_kpoints = n_cells
    dimensions.n_basis = n_orbitals
    dimensions.n_effective_orbitals = n_orbitals
    dimensions.n_auxiliary = 2 * n_orbitals
    dimensions.n_home_total_occupied = n_home_occupied
    dimensions.n_home_occupied = n_home_occupied
    dimensions.n_home_virtual = 1
    dimensions.n_spin_channels = 1
    dimensions.triples_requested = False
    dimensions.symmetry_reduction_requested = False
    dimensions.symmetry_mapping_identity = ""
    dimensions.symmetry_representative_count = n_cells
    dimensions.symmetry_weight_sum = n_cells
    dimensions.expected_pair_candidate_count = expected_pair_candidate_count
    dimensions.expected_triple_candidate_count = 0
    dimensions.factor_k_bra_block = 1
    dimensions.factor_k_ket_block = 1
    dimensions.factor_q_block = 1
    dimensions.factor_auxiliary_block = 1
    dimensions.domain_ao_support_upper_bound = n_orbitals
    dimensions.domain_pao_upper_bound = n_orbitals
    dimensions.domain_pno_upper_bound = n_orbitals
    dimensions.domain_local_occupied_upper_bound = n_home_occupied
    dimensions.domain_local_auxiliary_upper_bound = 2 * n_orbitals
    dimensions.pair_domain_metadata_upper_bytes = (
        pair_domain_metadata_upper_bytes
    )
    return dimensions


def _make_budget() -> object:
    budget = core._PeriodicCorrelationResourceBudget()
    budget.memory_limit_bytes = 1024**3
    budget.scratch_limit_bytes = 1024**3
    budget.mpi_ranks = 1
    budget.workers_per_rank = 1
    return budget


def _make_admitted_reference(
    mesh: tuple[int, int, int] = (2, 1, 1),
    n_home_occupied: int = 1,
    *,
    shift: tuple[int, int, int] = (0, 0, 0),
    calculation_identity: str = _CALCULATION_IDENTITY,
    allocation_identity: str = _ALLOCATION_IDENTITY,
    reference_energy_per_cell: float = -5.0,
) -> object:
    state = _make_state(
        mesh,
        n_home_occupied,
        shift=shift,
        calculation_identity=calculation_identity,
        reference_energy_per_cell=reference_energy_per_cell,
    )
    dimensions = _make_dimensions(
        mesh,
        n_home_occupied,
        shift=shift,
        calculation_identity=calculation_identity,
        allocation_identity=allocation_identity,
    )
    return core._make_periodic_correlation_admitted_reference(
        state, dimensions, _make_budget()
    )


def _make_topology(
    mesh: tuple[int, int, int] = (2, 1, 1),
    n_home_occupied: int = 1,
    **kwargs: object,
) -> object:
    admitted = _make_admitted_reference(mesh, n_home_occupied, **kwargs)
    return core._make_periodic_correlation_translation_pair_topology(admitted)


def _native_rows(topology: object) -> tuple[TopologyRow, ...]:
    return tuple(
        (
            topology.row(index).home_orbital,
            topology.row(index).partner_orbital,
            topology.row(index).translation_linear_index,
            topology.row(index).placed_multiplicity,
        )
        for index in range(topology.row_count)
    )


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1),
                                 (2, 2, 2), (2, 3, 2)])
@pytest.mark.parametrize("nocc", [1, 2, 3])
def test_ordered_placed_pair_resolution_reconstructs_endpoints_and_energy_counts(mesh, nocc):
    topology = _make_topology(mesh, nocc)
    cells = _cells(mesh)
    expected_rows, orbits = _brute_force_pair_oracle(mesh, nocc)
    row_for_unordered = {pair: row_index for row_index, row in enumerate(expected_rows)
                         for pair in orbits[row[:3]]}
    count = [0] * topology.row_count
    for l, m, i, j in product(range(len(cells)), range(len(cells)), range(nocc), range(nocc)):
        resolved = core._resolve_periodic_correlation_placed_pair(topology, i, l, j, m)
        row = topology.row(resolved.row_index)
        expected_index = row_for_unordered[tuple(sorted((l * nocc + i, m * nocc + j)))]
        assert resolved.row_index == expected_index
        shift = cells[resolved.common_translation_cell]
        partner_cell = _linear_cell_index(_add_cells(
            shift, cells[row.translation_linear_index], mesh), mesh)
        reconstructed = [(row.home_orbital, resolved.common_translation_cell),
                         (row.partner_orbital, partner_cell)]
        if resolved.transpose:
            reconstructed.reverse()
        assert reconstructed == [(i, l), (j, m)]
        count[resolved.row_index] += 1
    for index, actual_count in enumerate(count):
        weight = core._periodic_correlation_translation_pair_energy_weight(topology, index)
        assert weight * len(cells) == actual_count
        assert weight in (1, 2)
    assert sum(count) == (len(cells) * nocc)**2


def test_even_self_inverse_same_orbital_pair_reversal_changes_domain_translation():
    topology = _make_topology((2, 1, 1), 1)
    forward = core._resolve_periodic_correlation_placed_pair(topology, 0, 0, 0, 1)
    reverse = core._resolve_periodic_correlation_placed_pair(topology, 0, 1, 0, 0)
    assert forward.row_index == reverse.row_index
    assert not forward.transpose and not reverse.transpose
    assert (forward.common_translation_cell, reverse.common_translation_cell) == (0, 1)
    assert core._periodic_correlation_translation_pair_energy_weight(topology, forward.row_index) == 1


@pytest.mark.parametrize("args", [(1, 0, 0, 0), (0, 2, 0, 0),
                                  (0, 0, 1, 0), (0, 0, 0, 2)])
def test_placed_pair_resolution_rejects_alias_cells_and_inactive_labels(args):
    topology = _make_topology((2, 1, 1), 1)
    with pytest.raises(IndexError, match="out of range"):
        core._resolve_periodic_correlation_placed_pair(topology, *args)
    for row in (topology.row_count, 2**64 - 1):
        with pytest.raises(IndexError, match="out of range"):
            core._periodic_correlation_translation_pair_energy_weight(topology, row)


def _python_topology_identity(topology: object) -> str:
    """Independent encoder for the documented topology identity wire."""

    digest = hashlib.sha256()

    def u32(value: int) -> None:
        digest.update(struct.pack(">I", value))

    def u64(value: int) -> None:
        digest.update(struct.pack(">Q", value))

    def string(value: str) -> None:
        encoded = value.encode("utf-8")
        u64(len(encoded))
        digest.update(encoded)

    string("vibeqc.periodic.correlation.translation-pair-topology")
    u32(topology.contract_version)
    u32(core._PERIODIC_CORRELATION_ADMITTED_REFERENCE_CONTRACT_VERSION)
    u32(core._PERIODIC_CORRELATION_STREAMED_RESOURCE_CONTRACT_VERSION)
    u32(core._PERIODIC_MEAN_FIELD_STATE_DIGEST_VERSION)
    string(topology.state_identity_sha256)
    string(topology.calculation_identity)
    string(topology.allocation_identity)
    for value in topology.mesh:
        u32(value)
    for value in topology.is_shift:
        u32(value)
    u64(topology.n_cells)
    u64(topology.n_home_occupied)
    u64(topology.self_inverse_translation_count)
    u64(topology.row_count)
    u64(topology.placed_pair_count)
    string("unclassified")
    for row in _native_rows(topology):
        for value in row:
            u64(value)
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("mesh", "expected"),
    [
        ((2, 1, 1), ((0, 0, 0, 2), (0, 0, 1, 1))),
        ((3, 1, 1), ((0, 0, 0, 3), (0, 0, 1, 3))),
    ],
    ids=["even-two-cell-ring", "odd-three-cell-ring"],
)
def test_one_orbital_small_ring_rows_are_exact(
    mesh: tuple[int, int, int], expected: tuple[TopologyRow, ...]
) -> None:
    topology = _make_topology(mesh, 1)

    assert _native_rows(topology) == expected
    assert topology.row_count == len(expected)
    assert topology.placed_pair_count == sum(
        row[3] for row in expected
    )


def test_two_orbitals_pin_nested_row_order_and_even_stabilizers() -> None:
    topology = _make_topology((2, 1, 1), 2)

    assert _native_rows(topology) == (
        (0, 0, 0, 2),
        (0, 0, 1, 1),
        (0, 1, 0, 2),
        (0, 1, 1, 2),
        (1, 1, 0, 2),
        (1, 1, 1, 1),
    )
    assert topology.placed_pair_count == 10


@pytest.mark.parametrize("n_home_occupied", [1, 2, 3])
def test_all_small_meshes_match_full_placed_pair_orbits(
    n_home_occupied: int,
) -> None:
    """Exhaust every axis extent 1..4 without reusing native arithmetic."""

    for mesh in product(range(1, 5), repeat=3):
        expected_rows, expected_orbits = _brute_force_pair_oracle(
            mesh, n_home_occupied
        )
        topology = _make_topology(mesh, n_home_occupied)
        native_rows = _native_rows(topology)

        assert native_rows == expected_rows, (
            f"row mismatch for mesh={mesh}, B={n_home_occupied}"
        )
        assert topology.row_count == len(expected_orbits)
        assert topology.placed_pair_count == sum(
            len(orbit) for orbit in expected_orbits.values()
        )

        counts = core._estimate_periodic_correlation_translation_pair_counts(
            mesh, n_home_occupied
        )
        assert counts.cell_count == math.prod(mesh)
        assert counts.self_inverse_translation_count == sum(
            all((2 * cell[axis]) % mesh[axis] == 0 for axis in range(3))
            for cell in _cells(mesh)
        )
        assert counts.candidate_count == len(expected_rows)
        assert counts.placed_pair_count == sum(
            row[3] for row in expected_rows
        )


@pytest.mark.parametrize(
    (
        "mesh",
        "n_home_occupied",
        "expected_cells",
        "expected_self_inverse",
        "expected_candidates",
        "expected_placed_pairs",
        "expected_row_bytes",
    ),
    [
        ((8, 8, 8), 4, 512, 8, 4_112, 2_098_176, 131_584),
        ((8, 8, 8), 8, 512, 8, 16_416, 8_390_656, 525_312),
        ((6, 6, 6), 24, 216, 8, 62_304, 13_439_520, 1_993_728),
    ],
    ids=["eight-cubed-b4", "eight-cubed-b8", "six-cubed-b24"],
)
def test_target_shapes_are_counted_without_allocating_rows_or_states(
    mesh: tuple[int, int, int],
    n_home_occupied: int,
    expected_cells: int,
    expected_self_inverse: int,
    expected_candidates: int,
    expected_placed_pairs: int,
    expected_row_bytes: int,
) -> None:
    before = _rss_bytes()
    counts = core._estimate_periodic_correlation_translation_pair_counts(
        mesh, n_home_occupied
    )
    after = _rss_bytes()

    assert counts.cell_count == expected_cells
    assert counts.self_inverse_translation_count == expected_self_inverse
    assert counts.candidate_count == expected_candidates
    assert counts.placed_pair_count == expected_placed_pairs
    assert core._PERIODIC_CORRELATION_TRANSLATION_PAIR_ROW_BYTES == 32
    assert counts.candidate_count * 32 == expected_row_bytes
    assert after - before < 64 * 1024 * 1024


def test_count_estimator_rejects_invalid_and_overflowing_metadata() -> None:
    with pytest.raises(ValueError, match="n_home_occupied must be positive"):
        core._estimate_periodic_correlation_translation_pair_counts(
            (2, 1, 1), 0
        )
    with pytest.raises(RuntimeError, match="strictly positive divisions"):
        core._estimate_periodic_correlation_translation_pair_counts(
            (0, 1, 1), 1
        )
    with pytest.raises(OverflowError, match="home occupied pair count"):
        core._estimate_periodic_correlation_translation_pair_counts(
            (1, 1, 1), (1 << 64) - 1
        )


def test_admission_rejects_an_invented_pair_count_before_topology() -> None:
    state = _make_state((2, 1, 1), 1)
    dimensions = _make_dimensions(
        (2, 1, 1), 1, expected_pair_candidate_count=1
    )

    with pytest.raises(ValueError, match="exact translation-pair topology count"):
        core._make_periodic_correlation_admitted_reference(
            state, dimensions, _make_budget()
        )


def test_admission_rejects_pair_metadata_smaller_than_exact_worklist() -> None:
    mesh = (2, 1, 1)
    n_home_occupied = 1
    candidate_count = 2
    required_metadata = 8 * n_home_occupied + 64 * candidate_count
    state = _make_state(mesh, n_home_occupied)
    dimensions = _make_dimensions(
        mesh,
        n_home_occupied,
        pair_domain_metadata_upper_bytes=required_metadata - 1,
    )

    with pytest.raises(ValueError, match="structural minimum"):
        core._make_periodic_correlation_admitted_reference(
            state, dimensions, _make_budget()
        )


def test_pair_topology_rejects_shifted_mesh_after_valid_admission() -> None:
    admitted = _make_admitted_reference(
        (2, 1, 1), 1, shift=(1, 0, 0)
    )
    assert admitted.state.is_shift == [1, 0, 0]

    with pytest.raises(ValueError, match="requires the Gamma-centered"):
        core._make_periodic_correlation_translation_pair_topology(admitted)


def test_topology_seals_provenance_identity_and_retains_state_lifetime() -> None:
    admitted = _make_admitted_reference()
    topology = core._make_periodic_correlation_translation_pair_topology(
        admitted
    )
    wrapped_state = topology.state

    assert topology.contract_version == (
        core._PERIODIC_CORRELATION_TRANSLATION_PAIR_TOPOLOGY_CONTRACT_VERSION
    )
    assert topology.calculation_identity == _CALCULATION_IDENTITY
    assert topology.allocation_identity == _ALLOCATION_IDENTITY
    assert topology.state_identity_sha256 == admitted.state.state_identity_sha256
    assert topology.state.state_identity_sha256 == topology.state_identity_sha256
    assert topology.mesh == [2, 1, 1]
    assert topology.is_shift == [0, 0, 0]
    assert topology.n_cells == 2
    assert topology.n_home_occupied == 1
    assert topology.self_inverse_translation_count == 2
    assert topology.placed_pair_count == 3
    assert re.fullmatch(r"[0-9a-f]{64}", topology.topology_identity_sha256)
    assert topology.topology_identity_sha256 == (
        "2757f701f381e3d0358a562e60727af99909778f60ff72ffc1f86e9bbc4d1aa6"
    )
    assert topology.topology_identity_sha256 == _python_topology_identity(
        topology
    )

    del admitted
    gc.collect()
    assert topology.state.fock(1).shape == (2, 2)

    del topology
    gc.collect()
    assert wrapped_state.correlated_occupied_mask(0) == [1, 0]


def test_topology_identity_is_stable_and_separates_sealed_inputs() -> None:
    baseline = _make_topology()
    repeated = _make_topology()
    allocation_changed = _make_topology(allocation_identity="5" * 64)
    payload_changed = _make_topology(reference_energy_per_cell=-5.25)

    assert baseline.topology_identity_sha256 == repeated.topology_identity_sha256
    assert baseline.topology_identity_sha256 != (
        allocation_changed.topology_identity_sha256
    )
    assert baseline.topology_identity_sha256 != (
        payload_changed.topology_identity_sha256
    )
    assert _native_rows(baseline) == _native_rows(allocation_changed)
    assert _native_rows(baseline) == _native_rows(payload_changed)


def test_rows_stay_unclassified_and_have_no_numerical_or_domain_claims() -> None:
    topology = _make_topology()
    exact_public_surface = {
        "classification",
        "home_orbital",
        "partner_orbital",
        "placed_multiplicity",
        "translation_linear_index",
    }

    for index in range(topology.row_count):
        row = topology.row(index)
        assert row.classification == (
            core._PeriodicCorrelationPairClassification.UNCLASSIFIED
        )
        assert topology.classification(index) == (
            core._PeriodicCorrelationPairClassification.UNCLASSIFIED
        )
        assert {
            name for name in dir(row) if not name.startswith("_")
        } == exact_public_surface

    with pytest.raises(AttributeError):
        topology.row(0).placed_multiplicity = 99


def test_topology_row_and_classification_bounds_are_checked() -> None:
    topology = _make_topology()
    with pytest.raises(IndexError, match="row index is out of range"):
        topology.row(topology.row_count)
    with pytest.raises(IndexError, match="row index is out of range"):
        topology.classification(topology.row_count)
