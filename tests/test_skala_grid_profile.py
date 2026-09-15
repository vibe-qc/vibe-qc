"""vibe-qc's accepted atomic-grid parity profile for Microsoft SKALA-1.1.

Reference values were generated once with PySCF 2.14.0, using its default
``Grids(level=3)`` configuration: atom-specific Treutler radial quadrature,
NWChem pruning, and Treutler sqrt(Bragg-radius) Becke adjustment. This matches
Microsoft's default inference/benchmark setup; the checkpoint itself supports
configurable grids. PySCF is an out-of-process oracle only and is not imported
by vibe-qc or this test.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import vibeqc as vq
import vibeqc._vibeqc_core as core


def _profile_options():
    options = vq.GridOptions()
    options.atomic_grid_profile = "pyscf-level3"
    return options


def _direction_order_digest(points: np.ndarray) -> str:
    """Return a platform-stable digest of the ordered unit directions."""
    radii = np.linalg.norm(points, axis=1)
    quantized = np.rint(points / radii[:, None] * 1e12).astype("<i8")
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


@pytest.mark.parametrize(
    ("atomic_number", "expected_points"),
    [
        (1, 9808),
        (6, 14118),
        (8, 14082),
        (11, 18100),
        (17, 18880),
        (82, 23228),
        (83, 23444),
        (103, 24378),
    ],
)
def test_level3_element_point_counts_match_pyscf_214(
    atomic_number, expected_points
):
    # Neutral odd-Z atoms are doublets; Molecule correctly rejects the
    # physically inconsistent default singlet before grid construction.
    multiplicity = 2 if atomic_number % 2 else 1
    molecule = vq.Molecule(
        [vq.Atom(atomic_number, [0.0, 0.0, 0.0])],
        multiplicity=multiplicity,
    )
    options = _profile_options()

    assert core.grid_atomic_point_counts(molecule, options) == [expected_points]
    grid = vq.build_grid(molecule, options)
    assert len(grid.weights) == expected_points


def test_level3_point_count_helper_preserves_variable_atom_order():
    molecule = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.4, 0.0]),
            vq.Atom(6, [0.0, -1.4, 0.0]),
        ],
        multiplicity=2,
    )
    assert core.grid_atomic_point_counts(molecule, _profile_options()) == [
        14082,
        9808,
        14118,
    ]


def test_level3_profile_overrides_generic_resolution_switches():
    molecule = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2
    )
    options = _profile_options()
    options.n_radial = 1
    options.angular = "product"
    options.n_theta = 1
    options.n_phi = 1
    options.partition = "stratmann"
    options.angular_pruning = "none"

    assert core.grid_atomic_point_counts(molecule, options) == [9808]
    assert len(vq.build_grid(molecule, options).weights) == 9808


def test_level3_hydrogen_radial_values_pruning_and_local_order_match_pyscf():
    molecule = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2
    )
    grid = vq.build_grid(molecule, _profile_options())
    points = np.asarray(grid.points)
    radii = np.linalg.norm(points, axis=1)
    sorted_values = np.sort(radii)
    sorted_radii = [float(sorted_values[0])]
    for radius in sorted_values[1:]:
        if abs(radius - sorted_radii[-1]) > 1e-11:
            sorted_radii.append(float(radius))
    assert len(sorted_radii) == 50
    np.testing.assert_allclose(
        sorted_radii[:3],
        [
            2.5481667399267265e-05,
            2.3414385822210056e-04,
            8.568538323199466e-04,
        ],
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        sorted_radii[-3:],
        [8.295040262440006, 9.731360469318938, 12.170112060262891],
        rtol=2e-14,
        atol=0.0,
    )

    # PySCF groups radial shells by ascending angular size. Inside each tier
    # it processes at most 12 radii at a time and flattens angular-major,
    # radial-minor. The 266-point tier occurs on both sides of the densest
    # radial region, so its 18 shell indices are deliberately noncontiguous.
    expected_radii: list[float] = []
    for angular_count, shell_indices in (
        (50, tuple(range(0, 15))),
        (86, tuple(range(15, 19))),
        (266, (19, 20, 21, 22, 23, *range(37, 50))),
        (302, tuple(range(24, 37))),
    ):
        tier_radii = [sorted_radii[index] for index in shell_indices]
        for batch_start in range(0, len(tier_radii), 12):
            batch = tier_radii[batch_start : batch_start + 12]
            expected_radii.extend(batch * angular_count)
    np.testing.assert_allclose(radii, expected_radii, rtol=2e-14, atol=0.0)

    # The first 50-point Lebedev tier begins with +x then -x. Each direction
    # owns the same first batch of 12 ascending radii.
    np.testing.assert_allclose(
        points[:12],
        np.column_stack((sorted_radii[:12], np.zeros((12, 2)))),
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        points[12:24],
        np.column_stack((-np.asarray(sorted_radii[:12]), np.zeros((12, 2)))),
        rtol=2e-14,
        atol=0.0,
    )

    # Raw angular weights sum to 4 pi in vibe-qc, while PySCF puts 4 pi in
    # the radial factor. Their per-shell raw volume is therefore identical.
    first_shell_weight = float(
        np.asarray(grid.atomic_weights)[np.arange(0, 12 * 50, 12)].sum()
    )
    assert first_shell_weight == pytest.approx(
        6.65326465346663e-13, rel=3e-14, abs=0.0
    )


@pytest.mark.parametrize(
    ("atomic_number", "expected_digest"),
    [
        (1, "811a64b174c4468b63a7726cae681b16ffd92ca7f438af673efa22e0c6cec050"),
        (11, "44e425c88ff00b671e4229e87a022d13937b0f06ea27bf3dd01f04c83da97f13"),
    ],
)
def test_level3_full_direction_sequence_matches_pyscf_214(
    atomic_number, expected_digest
):
    # H exercises 50/86/266/302 directions and Na exercises 50/86/350/434.
    # The digests were generated once from SkalaGrids in PySCF 2.14.0 after
    # quantizing ordered unit directions to 1e-12; no PySCF runtime is needed.
    molecule = vq.Molecule(
        [vq.Atom(atomic_number, [0.0, 0.0, 0.0])], multiplicity=2
    )
    points = np.asarray(vq.build_grid(molecule, _profile_options()).points)
    assert _direction_order_digest(points) == expected_digest


def test_heteronuclear_treutler_becke_weight_matches_pyscf_214():
    molecule = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(9, [0.0, 0.0, 1.7])]
    )
    grid = vq.build_grid(molecule, _profile_options())
    points = np.asarray(grid.points)
    target = np.array(
        [0.8489670538118455, 0.15647893654562037, 0.8489670538118455]
    )
    index = int(np.argmin(np.linalg.norm(points - target, axis=1)))
    assert np.linalg.norm(points[index] - target) < 2e-14
    partition_weight = float(grid.weights[index] / grid.atomic_weights[index])
    # Direct PySCF 2.14.0 reference at this exact H-owned point:
    # ``grids.weights[index] / grids.quadrature_weights[index]`` with
    # alignment=1 and sort_grids=False. The former 0.5014 reference was the
    # unadjusted homonuclear Becke value and omitted Treutler's heteroatomic
    # sqrt(BRAGG_RADII) correction.
    assert partition_weight == pytest.approx(
        0.35338339593999024, rel=3e-14, abs=0.0
    )


def test_periodic_heteronuclear_molecular_limit_is_exact():
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(9, [0.0, 0.0, 1.7])]
    molecule = vq.Molecule(atoms)
    system = vq.PeriodicSystem(1, np.diag([20.0, 25.0, 30.0]), atoms)
    options = _profile_options()

    molecular = vq.build_grid(molecule, options)
    periodic = vq.build_periodic_becke_grid(
        system, grid_options=options, image_radius_bohr=0.0
    )

    np.testing.assert_array_equal(periodic.points, molecular.points)
    np.testing.assert_array_equal(periodic.weights, molecular.weights)
    np.testing.assert_array_equal(periodic.atomic_weights, molecular.atomic_weights)
    assert periodic.atom_of_point == molecular.atom_of_point
    np.testing.assert_array_equal(periodic.atom_coords, molecular.atom_coords)
    assert periodic.atomic_numbers == molecular.atomic_numbers


def test_periodic_heteronuclear_profile_passes_image_atomic_numbers():
    lattice = np.diag([3.4, 30.0, 30.0])
    system = vq.PeriodicSystem(
        1,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(9, [1.7, 0.0, 0.0])],
    )
    options = _profile_options()
    from vibeqc.periodic_grid import _extended_partition_atom_data

    positions, numbers = _extended_partition_atom_data(system, 2.0)
    assert len(positions) > 2
    assert numbers[:2] == [1, 9]
    assert {1, 9} <= set(numbers[2:])

    expected = core.build_grid_periodic(
        system.unit_cell_molecule(), positions, numbers, options
    )
    actual = vq.build_periodic_becke_grid(
        system, grid_options=options, image_radius_bohr=2.0
    )
    np.testing.assert_allclose(actual.points, expected.points, rtol=0, atol=0)
    np.testing.assert_array_equal(actual.atomic_weights, expected.atomic_weights)

    wrong_numbers = numbers.copy()
    wrong_numbers[2:] = [1] * (len(numbers) - 2)
    wrong = core.build_grid_periodic(
        system.unit_cell_molecule(), positions, wrong_numbers, options
    )
    assert not np.allclose(expected.weights, wrong.weights, rtol=1e-13, atol=0)

    # A point-centered image domain has a different denominator from the
    # explicit home-centered list. Verify the H/F radius correction against
    # independent normalized products on the physical point neighborhood.
    max_point_radius = np.max(np.linalg.norm(np.asarray(actual.points), axis=1))
    image_bound = int(np.ceil((3*max_point_radius+2.)/3.4))+2
    centers = np.array([[3.4*n+offset, 0., 0.]
                        for n in range(-image_bound, image_bound+1) for offset in [0., 1.7]])
    radii = np.tile(np.sqrt([.35, .5]), 2*image_bound+1)
    selected = np.random.default_rng(772).choice(len(actual.weights), 96, replace=False)
    hetero_witnesses = 0
    for index in selected:
        point = np.asarray(actual.points)[index]
        owner = actual.atom_of_point[index]
        image_distances = np.linalg.norm(centers-point, axis=1)
        point_radius = max(2., 2*float(image_distances.min()))
        if np.linalg.norm(point-np.asarray(system.unit_cell[owner].xyz)) > point_radius:
            assert actual.weights[index] == 0.
            continue
        keep = np.flatnonzero(image_distances <= point_radius)
        distances = np.linalg.norm(centers[keep]-point, axis=1)
        weights = np.ones(len(keep))
        unadjusted = np.ones(len(keep))
        for i in range(len(keep)):
            for j in range(len(keep)):
                if i == j:
                    continue
                mu = (distances[i]-distances[j])/np.linalg.norm(centers[keep[i]]-centers[keep[j]])
                ratio = radii[keep[j]]/radii[keep[i]]
                adjustment = np.clip(.25*(ratio-1/ratio), -.5, .5)
                corrected = mu+adjustment*(1-mu*mu)
                for _ in range(3):
                    mu = .5*mu*(3-mu*mu)
                    corrected = .5*corrected*(3-corrected*corrected)
                weights[i] *= .5*(1-corrected)
                unadjusted[i] *= .5*(1-mu)
        position = np.flatnonzero(keep == 2*image_bound+owner)[0]
        reference = weights[position]/weights.sum()
        observed = actual.weights[index]/actual.atomic_weights[index]
        assert observed == pytest.approx(reference, rel=0, abs=3e-13)
        hetero_witnesses += abs(reference-unadjusted[position]/unadjusted.sum()) > 1e-5
    assert hetero_witnesses >= 5

    wrong_home_numbers = numbers.copy()
    wrong_home_numbers[1] = 1
    with pytest.raises(ValueError, match="home cell first"):
        core.build_grid_periodic(
            system.unit_cell_molecule(), positions, wrong_home_numbers, options
        )


def test_level3_profile_fails_above_pyscf_treutler_table():
    molecule = vq.Molecule([vq.Atom(104, [0.0, 0.0, 0.0])])
    with pytest.raises(ValueError, match=r"1\.\.103.*Z=104"):
        core.grid_atomic_point_counts(molecule, _profile_options())


def test_periodic_profile_requires_partition_atomic_numbers():
    molecule = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(9, [0.0, 0.0, 1.7])]
    )
    positions = [np.asarray(atom.xyz, dtype=float) for atom in molecule.atoms]
    with pytest.raises(ValueError, match="requires partition_atomic_numbers"):
        core.build_grid_periodic(molecule, positions, _profile_options())
