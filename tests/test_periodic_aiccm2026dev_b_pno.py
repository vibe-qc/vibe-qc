"""Finite-torus PAO, PNO, and pair-orbit invariants for the B stream."""

from __future__ import annotations

from collections import Counter
from itertools import permutations, product

import numpy as np
import pytest

from vibeqc.periodic.chi.pno import (
    AICCM2026DevBPointPairPlan,
    AICCM2026DevBTranslationPairPlan,
    _enumerate_pair_orbits_python,
    _full_space_mp2_energy_python,
    _translation_permutations_python,
    _validate_permutations,
    build_pair_natural_orbitals,
    build_point_pair_plan,
    build_translation_pair_plan,
    enumerate_pair_orbits,
    full_space_noncanonical_mp2_energy,
    projected_atomic_orbitals,
    translation_permutations,
)


def _expanded_translation_pair_rows(
    plan: AICCM2026DevBTranslationPairPlan,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Expand a compact plan only for small-mesh oracle comparisons."""

    cells = tuple(product(*(range(value) for value in plan.mesh)))
    cell_index = {cell: index for index, cell in enumerate(cells)}
    output: list[tuple[tuple[int, int], ...]] = []
    for row in range(plan.n_representatives):
        band_i, band_j = (int(value) for value in plan.band_pairs[row])
        displacement = tuple(int(value) for value in plan.displacements[row])
        members: set[tuple[int, int]] = set()
        for cell in cells:
            target = tuple(
                (cell[axis] + displacement[axis]) % plan.mesh[axis]
                for axis in range(3)
            )
            occupied_i = cell_index[cell] * plan.n_bands + band_i
            occupied_j = cell_index[target] * plan.n_bands + band_j
            members.add(tuple(sorted((occupied_i, occupied_j))))
        assert len(members) == int(plan.multiplicities[row])
        output.append(tuple(sorted(members)))
    return tuple(output)


def _expanded_compact_pair_orbits(
    plan: AICCM2026DevBTranslationPairPlan,
) -> set[tuple[tuple[int, int], ...]]:
    return set(_expanded_translation_pair_rows(plan))


def _expanded_point_pair_orbits(
    plan: AICCM2026DevBPointPairPlan,
) -> set[tuple[tuple[int, int], ...]]:
    """Expand a point quotient only for small full-orbit comparisons."""

    translation_rows = _expanded_translation_pair_rows(plan.translation_plan)
    output: set[tuple[tuple[int, int], ...]] = set()
    for orbit_index in range(plan.n_representatives):
        start = int(plan.orbit_offsets[orbit_index])
        stop = int(plan.orbit_offsets[orbit_index + 1])
        members: set[tuple[int, int]] = set()
        for row_index in plan.member_indices[start:stop]:
            members.update(translation_rows[int(row_index)])
        assert len(members) == int(plan.combined_multiplicities[orbit_index])
        output.add(tuple(sorted(members)))
    return output


def _occupied_point_permutation(
    mesh: tuple[int, int, int],
    coordinate_operation: np.ndarray,
    band_permutation: np.ndarray,
) -> np.ndarray:
    """Build the full occupied permutation for one monomial point action."""

    cells = tuple(product(*(range(value) for value in mesh)))
    cell_index = {cell: index for index, cell in enumerate(cells)}
    n_bands = int(band_permutation.size)
    output = np.empty(len(cells) * n_bands, dtype=int)
    for source_cell_index, cell in enumerate(cells):
        mapped = coordinate_operation @ np.asarray(cell, dtype=int)
        target = tuple(
            int(mapped[axis]) % mesh[axis]
            for axis in range(3)
        )
        target_cell_index = cell_index[target]
        for source_band, target_band in enumerate(band_permutation):
            output[source_cell_index * n_bands + source_band] = (
                target_cell_index * n_bands + int(target_band)
            )
    return output


def _cubic_sp_point_actions(
    mesh: tuple[int, int, int],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return all 48 O_h coordinate and s,px,py,pz index actions."""

    coordinate_operations: list[np.ndarray] = []
    occupied_permutations: list[np.ndarray] = []
    for axis_permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            operation = np.zeros((3, 3), dtype=int)
            for source_axis, target_axis in enumerate(axis_permutation):
                operation[target_axis, source_axis] = signs[source_axis]
            band_permutation = np.asarray(
                (0, *(1 + axis for axis in axis_permutation)),
                dtype=int,
            )
            coordinate_operations.append(operation)
            occupied_permutations.append(
                _occupied_point_permutation(
                    mesh,
                    operation,
                    band_permutation,
                )
            )
    return tuple(coordinate_operations), tuple(occupied_permutations)


def _metric_orthonormal_coefficients(
    overlap: np.ndarray,
    count: int,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    orthogonalizer = eigenvectors / np.sqrt(eigenvalues)[None, :]
    return orthogonalizer[:, :count]


def test_full_domain_paos_are_virtual_and_metric_orthonormal() -> None:
    overlap = np.array(
        [
            [1.0, 0.13, 0.02, 0.00],
            [0.13, 1.0, 0.08, 0.01],
            [0.02, 0.08, 1.0, 0.17],
            [0.00, 0.01, 0.17, 1.0],
        ]
    )
    occupied = _metric_orthonormal_coefficients(overlap, 2)
    fock = np.diag([-1.1, -0.7, 0.3, 0.8])
    paos = projected_atomic_orbitals(occupied, overlap, fock)

    assert paos.rank == 2
    assert paos.discarded_rank == 2
    assert paos.orthonormality_error < 1e-12
    assert paos.occupied_leakage_error < 1e-12
    assert np.allclose(
        paos.coefficients.T @ overlap @ paos.coefficients,
        np.eye(2),
        atol=1e-12,
    )
    assert np.allclose(
        occupied.T @ overlap @ paos.coefficients,
        0.0,
        atol=1e-12,
    )


def test_domain_paos_remove_redundancy_by_relative_threshold() -> None:
    overlap = np.eye(5)
    occupied = np.eye(5)[:, :2]
    fock = np.diag([-1.0, -0.8, 0.2, 0.4, 0.9])
    paos = projected_atomic_orbitals(
        occupied,
        overlap,
        fock,
        ao_indices=[0, 2, 3],
    )
    assert paos.rank == 2
    assert paos.discarded_rank == 1
    assert np.allclose(paos.orbital_energies, [0.2, 0.4])


def test_pno_rank_is_nested_and_zero_threshold_is_exact_span() -> None:
    amplitudes = np.array(
        [
            [-0.20, 0.04, 0.00],
            [0.01, -0.05, 0.002],
            [0.00, 0.001, -0.003],
        ]
    )
    virtuals = np.eye(3)
    loose = build_pair_natural_orbitals(
        amplitudes,
        virtuals,
        diagonal_pair=False,
        occupation_threshold=1e-2,
    )
    tight = build_pair_natural_orbitals(
        amplitudes,
        virtuals,
        diagonal_pair=False,
        occupation_threshold=1e-5,
    )
    exact = build_pair_natural_orbitals(
        amplitudes,
        virtuals,
        diagonal_pair=False,
        occupation_threshold=0.0,
    )

    assert loose.retained_rank <= tight.retained_rank <= exact.retained_rank
    assert exact.retained_rank == 3
    assert exact.discarded_occupation == pytest.approx(0.0)
    assert np.all(exact.occupations >= 0.0)
    assert exact.hermiticity_error < 1e-14
    assert np.allclose(exact.coefficients @ exact.coefficients.T, np.eye(3))


def test_complex_pair_density_uses_adjoint_not_transpose() -> None:
    amplitudes = np.array(
        [[-0.2 + 0.1j, 0.03j], [0.01 - 0.02j, -0.05]],
        dtype=complex,
    )
    result = build_pair_natural_orbitals(
        amplitudes,
        np.eye(2, dtype=complex),
        diagonal_pair=True,
        occupation_threshold=0.0,
    )
    assert result.minimum_occupation >= -1e-12
    assert result.hermiticity_error < 1e-14
    assert np.allclose(
        result.coefficients.conj().T @ result.coefficients,
        np.eye(2),
    )


def test_full_space_mp2_audit_is_invariant_to_occupied_virtual_rotations() -> None:
    rng = np.random.default_rng(617)
    occupied = np.eye(5)[:, :2]
    virtual = np.eye(5)[:, 2:]
    fock = np.diag([-1.2, -0.8, 0.2, 0.5, 0.9])
    factors = rng.normal(scale=0.08, size=(7, 5, 5))
    factors = 0.5 * (factors + factors.swapaxes(1, 2))
    reference = full_space_noncanonical_mp2_energy(
        occupied,
        virtual,
        fock,
        factors,
    )
    occ_rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    vir_rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = full_space_noncanonical_mp2_energy(
        occupied @ occ_rotation,
        virtual @ vir_rotation,
        fock,
        factors,
    )
    assert rotated == pytest.approx(reference, abs=1e-14)


def test_native_full_space_mp2_kernel_matches_tensor_oracle() -> None:
    core = pytest.importorskip("vibeqc._vibeqc_core")
    if not hasattr(core, "aiccm2026dev_b_real_mp2_energy_from_lov"):
        pytest.skip("editable extension lacks the native χ-CCM real MP2 kernel")

    rng = np.random.default_rng(620)
    factors_ov = rng.normal(scale=0.05, size=(6, 2, 3))
    eps_occ = np.array([-1.1, -0.7])
    eps_vir = np.array([0.2, 0.5, 0.9])
    oracle = _full_space_mp2_energy_python(
        factors_ov,
        eps_occ,
        eps_vir,
        1e-12,
    )
    native = core.aiccm2026dev_b_real_mp2_energy_from_lov(
        np.ascontiguousarray(factors_ov),
        np.ascontiguousarray(eps_occ),
        np.ascontiguousarray(eps_vir),
        1e-12,
    )
    assert native == pytest.approx(oracle, abs=1e-14)


def test_translation_pair_orbits_partition_even_ring_with_fixed_half_pair() -> None:
    permutations = translation_permutations(1, (4, 1, 1))
    orbits = enumerate_pair_orbits(permutations, n_cells=4)

    assert sum(orbit.multiplicity for orbit in orbits) == 10
    assert len(orbits) == 3
    assert sorted(orbit.multiplicity for orbit in orbits) == [2, 4, 4]
    assert sum(orbit.per_cell_weight for orbit in orbits) == pytest.approx(2.5)
    half_cell = next(orbit for orbit in orbits if orbit.multiplicity == 2)
    assert half_cell.per_cell_weight == pytest.approx(0.5)


@pytest.mark.parametrize("n_bands", [1, 2, 3])
@pytest.mark.parametrize(
    "mesh",
    [
        (1, 1, 1),
        (3, 1, 1),
        (4, 1, 1),
        (2, 3, 1),
        (2, 2, 2),
    ],
)
def test_compact_translation_pair_plan_matches_full_orbit_oracle(
    n_bands: int,
    mesh: tuple[int, int, int],
) -> None:
    plan = build_translation_pair_plan(n_bands, mesh)
    oracle = enumerate_pair_orbits(
        translation_permutations(n_bands, mesh),
        n_cells=plan.n_cells,
    )

    assert _expanded_compact_pair_orbits(plan) == {
        orbit.members for orbit in oracle
    }
    assert int(np.sum(plan.multiplicities, dtype=object)) == plan.n_placed_pairs
    assert float(np.sum(plan.per_cell_weights)) == pytest.approx(
        plan.n_placed_pairs / plan.n_cells
    )


def test_compact_translation_pair_plan_handles_even_mesh_stabilizers() -> None:
    plan = build_translation_pair_plan(4, (8, 8, 8))
    same_band = plan.band_pairs[:, 0] == plan.band_pairs[:, 1]
    home = np.all(plan.displacements == 0, axis=1)
    self_inverse = np.all(
        (2 * plan.displacements) % np.asarray(plan.mesh) == 0,
        axis=1,
    )
    antipodal = same_band & self_inverse & ~home

    assert plan.n_cells == 512
    assert plan.n_occupied == 2_048
    assert plan.n_placed_pairs == 2_098_176
    assert plan.n_representatives == 4_112
    assert np.count_nonzero(same_band & home) == 4
    assert np.count_nonzero(antipodal) == 28
    assert np.all(plan.multiplicities[same_band & home] == 512)
    assert np.all(plan.multiplicities[antipodal] == 256)
    assert np.all(plan.per_cell_weights[antipodal] == 0.5)
    assert np.all(plan.per_cell_weights[~antipodal] == 1.0)
    assert float(np.sum(plan.per_cell_weights)) == pytest.approx(4_098.0)
    assert plan.payload_nbytes == 197_376
    assert plan.fingerprint == (
        "779e82a984a9d9686d5ea06d84f89ad5ea655e96618193f768c7db361a6e976c"
    )
    assert max(
        plan.band_pairs.size,
        plan.displacements.size,
        plan.multiplicities.size,
    ) < plan.n_occupied**2


def test_compact_translation_pair_plan_is_immutable_and_fingerprinted() -> None:
    first = build_translation_pair_plan(2, (2, 3, 1))
    repeated = build_translation_pair_plan(2, (2, 3, 1))
    changed = build_translation_pair_plan(2, (3, 2, 1))

    assert first.fingerprint == repeated.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert len(first.fingerprint) == 64
    assert set(first.fingerprint) <= set("0123456789abcdef")
    with pytest.raises(ValueError, match="read-only"):
        first.band_pairs[0, 0] = 1
    for array in (
        first.band_pairs,
        first.displacements,
        first.multiplicities,
    ):
        with pytest.raises(ValueError, match="WRITEABLE flag"):
            array.setflags(write=True)

    source_band_pairs = np.array(repeated.band_pairs, copy=True)
    source_displacements = np.array(repeated.displacements, copy=True)
    source_multiplicities = np.array(repeated.multiplicities, copy=True)
    detached = AICCM2026DevBTranslationPairPlan(
        mesh=repeated.mesh,
        n_bands=repeated.n_bands,
        band_pairs=source_band_pairs,
        displacements=source_displacements,
        multiplicities=source_multiplicities,
    )
    source_band_pairs[0, 0] = 1
    source_displacements[0, 0] = 1
    source_multiplicities[0] = 1
    assert detached.fingerprint == repeated.fingerprint
    assert np.array_equal(detached.band_pairs, repeated.band_pairs)
    assert np.array_equal(detached.displacements, repeated.displacements)
    assert np.array_equal(detached.multiplicities, repeated.multiplicities)


def test_compact_translation_pair_plan_rejects_malformed_payloads() -> None:
    valid = build_translation_pair_plan(2, (2, 1, 1))
    duplicate = np.array(valid.band_pairs, copy=True)
    duplicate_displacements = np.array(valid.displacements, copy=True)
    duplicate[1] = duplicate[0]
    duplicate_displacements[1] = duplicate_displacements[0]
    with pytest.raises(ValueError, match="unique canonical order"):
        AICCM2026DevBTranslationPairPlan(
            mesh=valid.mesh,
            n_bands=valid.n_bands,
            band_pairs=duplicate,
            displacements=duplicate_displacements,
            multiplicities=valid.multiplicities,
        )

    incomplete = np.array(valid.multiplicities, copy=True)
    incomplete[0] -= 1
    with pytest.raises(ValueError, match="stabilizer multiplicity"):
        AICCM2026DevBTranslationPairPlan(
            mesh=valid.mesh,
            n_bands=valid.n_bands,
            band_pairs=valid.band_pairs,
            displacements=valid.displacements,
            multiplicities=incomplete,
        )

    with pytest.raises(ValueError, match="must contain integers"):
        AICCM2026DevBTranslationPairPlan(
            mesh=valid.mesh,
            n_bands=valid.n_bands,
            band_pairs=np.asarray(valid.band_pairs, dtype=float),
            displacements=valid.displacements,
            multiplicities=valid.multiplicities,
        )
    with pytest.raises(ValueError, match="n_bands must be an integer"):
        build_translation_pair_plan(True, (2, 1, 1))
    with pytest.raises(OverflowError, match="cell count exceeds int64"):
        build_translation_pair_plan(1, (2**32, 2**32, 1))


@pytest.mark.parametrize(
    "mesh, point_specs",
    [
        (
            (3, 1, 1),
            (
                (np.diag([-1, 1, 1]), np.array([0, 1])),
                (np.eye(3, dtype=int), np.array([1, 0])),
            ),
        ),
        (
            (4, 1, 1),
            (
                (np.diag([-1, 1, 1]), np.array([1, 0])),
            ),
        ),
        (
            (2, 3, 1),
            (
                (np.diag([-1, 1, 1]), np.array([0, 1])),
                (np.diag([1, -1, 1]), np.array([1, 0])),
            ),
        ),
    ],
)
def test_compact_point_pair_plan_matches_full_placed_pair_oracle(
    mesh: tuple[int, int, int],
    point_specs: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> None:
    translation_plan = build_translation_pair_plan(2, mesh)
    point_actions = tuple(
        _occupied_point_permutation(mesh, operation, band_permutation)
        for operation, band_permutation in point_specs
    )
    plan = build_point_pair_plan(translation_plan, point_actions)
    oracle = enumerate_pair_orbits(
        (*translation_permutations(2, mesh), *point_actions),
        n_cells=translation_plan.n_cells,
    )

    assert _expanded_point_pair_orbits(plan) == {
        orbit.members for orbit in oracle
    }
    assert plan.parent_fingerprint == translation_plan.fingerprint
    assert plan.n_translation_representatives == (
        translation_plan.n_representatives
    )
    assert np.array_equal(
        np.sort(plan.member_indices),
        np.arange(translation_plan.n_representatives),
    )
    for orbit_index, representative in enumerate(plan.representative_indices):
        start = int(plan.orbit_offsets[orbit_index])
        stop = int(plan.orbit_offsets[orbit_index + 1])
        assert int(representative) == int(np.min(plan.member_indices[start:stop]))
    assert int(np.sum(plan.combined_multiplicities, dtype=object)) == (
        translation_plan.n_placed_pairs
    )
    assert float(np.sum(plan.per_cell_weights)) == pytest.approx(
        translation_plan.n_placed_pairs / translation_plan.n_cells
    )


def test_compact_point_pair_plan_rejects_invalid_or_non_normalizing_actions(
) -> None:
    one_cell = build_translation_pair_plan(2, (1, 1, 1))
    with pytest.raises(ValueError, match="not a permutation"):
        build_point_pair_plan(one_cell, [np.array([0, 0])])

    ring = build_translation_pair_plan(1, (4, 1, 1))
    non_normalizer = np.array([0, 2, 1, 3])
    assert np.array_equal(np.sort(non_normalizer), np.arange(4))
    with pytest.raises(ValueError, match="normaliz"):
        build_point_pair_plan(ring, [non_normalizer])


def test_compact_point_pair_plan_is_immutable_and_fingerprinted() -> None:
    translation_plan = build_translation_pair_plan(2, (4, 1, 1))
    identity = np.arange(translation_plan.n_occupied)
    inversion = _occupied_point_permutation(
        translation_plan.mesh,
        np.diag([-1, 1, 1]),
        np.array([1, 0]),
    )
    source_actions = [identity.copy(), inversion.copy()]
    first = build_point_pair_plan(translation_plan, source_actions)
    repeated = build_point_pair_plan(
        translation_plan,
        [inversion.copy(), identity.copy(), inversion.copy()],
    )
    changed = build_point_pair_plan(translation_plan, [identity.copy()])

    assert first.parent_fingerprint == translation_plan.fingerprint
    assert first.action_fingerprint == repeated.action_fingerprint
    assert first.fingerprint == repeated.fingerprint
    assert first.action_fingerprint != changed.action_fingerprint
    assert first.fingerprint != changed.fingerprint
    assert first.n_action_generators == 2
    for fingerprint in (first.action_fingerprint, first.fingerprint):
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")

    original_fingerprint = first.fingerprint
    source_actions[0][:] = 0
    source_actions[1][:] = 0
    assert first.fingerprint == original_fingerprint
    for array in (
        first.representative_indices,
        first.orbit_offsets,
        first.member_indices,
        first.combined_multiplicities,
    ):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0
        with pytest.raises(ValueError, match="WRITEABLE flag"):
            array.setflags(write=True)

    identity_plan = build_point_pair_plan(translation_plan, [identity])
    with pytest.raises(TypeError, match="build_point_pair_plan"):
        AICCM2026DevBPointPairPlan(
            translation_plan=translation_plan,
            representative_indices=np.array([0]),
            orbit_offsets=np.array([0, translation_plan.n_representatives]),
            member_indices=np.arange(translation_plan.n_representatives),
            combined_multiplicities=np.array([translation_plan.n_placed_pairs]),
            n_action_generators=1,
            action_fingerprint=identity_plan.action_fingerprint,
        )


def test_compact_point_pair_plan_accepts_band_dependent_cell_shift_normalizer(
) -> None:
    mesh = (3, 3, 1)
    n_bands = 2
    cells = tuple(product(*(range(value) for value in mesh)))
    cell_index = {cell: index for index, cell in enumerate(cells)}
    band_permutation = (1, 0)
    band_shifts = ((1, 0, 0), (0, 2, 0))
    occupied_action = np.empty(len(cells) * n_bands, dtype=int)
    for source_cell_index, cell in enumerate(cells):
        for source_band in range(n_bands):
            target_cell = tuple(
                (cell[axis] + band_shifts[source_band][axis]) % mesh[axis]
                for axis in range(3)
            )
            occupied_action[source_cell_index * n_bands + source_band] = (
                cell_index[target_cell] * n_bands
                + band_permutation[source_band]
            )
    assert not np.array_equal(occupied_action[:n_bands], np.arange(n_bands))

    translation_plan = build_translation_pair_plan(n_bands, mesh)
    plan = build_point_pair_plan(translation_plan, [occupied_action])
    oracle = enumerate_pair_orbits(
        (*translation_permutations(n_bands, mesh), occupied_action),
        n_cells=translation_plan.n_cells,
    )

    assert _expanded_point_pair_orbits(plan) == {
        orbit.members for orbit in oracle
    }
    assert int(np.sum(plan.combined_multiplicities, dtype=object)) == (
        translation_plan.n_placed_pairs
    )


def test_mgo_cubic_monomial_point_pair_ceiling_has_exact_census() -> None:
    mesh = (8, 8, 8)
    translation_plan = build_translation_pair_plan(4, mesh)
    coordinate_operations, occupied_actions = _cubic_sp_point_actions(mesh)
    plan = build_point_pair_plan(translation_plan, occupied_actions)

    assert len(coordinate_operations) == 48
    assert len({operation.tobytes() for operation in coordinate_operations}) == 48
    assert plan.n_action_generators == 48
    assert plan.n_cells == 512
    assert plan.n_occupied == 2_048
    assert plan.n_translation_representatives == 4_112
    assert plan.n_representatives == 260
    assert Counter(int(value) for value in np.diff(plan.orbit_offsets)) == {
        1: 2,
        3: 26,
        4: 3,
        6: 48,
        12: 87,
        24: 76,
        48: 18,
    }
    assert np.array_equal(
        np.sort(plan.member_indices),
        np.arange(translation_plan.n_representatives),
    )
    assert int(np.sum(plan.combined_multiplicities, dtype=object)) == 2_098_176
    assert float(np.sum(plan.per_cell_weights)) == pytest.approx(4_098.0)
    assert plan.parent_fingerprint == translation_plan.fingerprint
    assert plan.payload_nbytes == 39_144
    assert plan.payload_nbytes < translation_plan.n_placed_pairs * 8


def test_point_permutation_reduces_translation_unique_pair_orientations() -> None:
    translations = list(translation_permutations(1, (2, 2, 1)))
    # Exchange the two periodic axes: (x,y) -> (y,x).
    axis_exchange = np.array([0, 2, 1, 3])
    translation_only = enumerate_pair_orbits(translations, n_cells=4)
    with_point_group = enumerate_pair_orbits(
        [*translations, axis_exchange],
        n_cells=4,
    )
    assert len(with_point_group) < len(translation_only)
    assert sum(orbit.multiplicity for orbit in with_point_group) == 10


def test_native_pair_orbit_kernel_matches_python_oracle() -> None:
    core = pytest.importorskip("vibeqc._vibeqc_core")
    if not hasattr(core, "aiccm2026dev_b_pair_orbits"):
        pytest.skip("editable extension lacks the native χ-CCM pair-orbit kernel")

    permutations = [
        *translation_permutations(2, (2, 1, 1)),
        np.array([1, 0, 3, 2]),
    ]
    validated = _validate_permutations(permutations)
    oracle = _enumerate_pair_orbits_python(validated, n_cells=2)
    native = enumerate_pair_orbits(permutations, n_cells=2)

    assert [orbit.representative for orbit in native] == [
        orbit.representative for orbit in oracle
    ]
    assert [orbit.members for orbit in native] == [orbit.members for orbit in oracle]
    assert [orbit.per_cell_weight for orbit in native] == [
        orbit.per_cell_weight for orbit in oracle
    ]


def test_native_translation_permutation_kernel_matches_python_oracle() -> None:
    core = pytest.importorskip("vibeqc._vibeqc_core")
    if not hasattr(core, "aiccm2026dev_b_translation_permutations"):
        pytest.skip(
            "editable extension lacks the native χ-CCM translation-permutation kernel"
        )

    native = translation_permutations(3, (2, 1, 2))
    oracle = _translation_permutations_python(3, (2, 1, 2))
    assert len(native) == len(oracle)
    for got, expected in zip(native, oracle, strict=True):
        assert np.array_equal(got, expected)


def test_invalid_pair_mapping_fails_closed() -> None:
    with pytest.raises(ValueError, match="not a permutation"):
        enumerate_pair_orbits([np.array([0, 0])], n_cells=1)
