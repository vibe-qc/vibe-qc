"""Geometry-only regression tests for the semiempirical CCM slab studies."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import runpy
from typing import Any, Sequence

import numpy as np

_STUDY_DIR = Path(__file__).parents[1] / "studies" / "seccm-bulk3d"
_A = 4.212


def _load_study(name: str) -> dict[str, Any]:
    return runpy.run_path(str(_STUDY_DIR / name))


def _unlike_coordination(
    positions: list[np.ndarray] | np.ndarray,
    species: Sequence[int | str],
    lattice: list[np.ndarray] | np.ndarray,
    nearest_distance: float,
) -> list[int]:
    coords = np.asarray(positions, dtype=float)
    cell = np.asarray(lattice, dtype=float)
    shifts = [
        np.asarray(label, dtype=float) @ cell
        for label in product((-1, 0, 1), repeat=3)
    ]
    counts: list[int] = []
    for atom, center in enumerate(coords):
        unlike_distances = [
            float(np.linalg.norm(other + shift - center))
            for shift in shifts
            for other_index, other in enumerate(coords)
            if species[atom] != species[other_index]
        ]
        assert np.isclose(
            min(unlike_distances),
            nearest_distance,
            rtol=0.0,
            atol=1.0e-10,
        )
        counts.append(
            int(
                sum(
                    np.isclose(
                        distance,
                        nearest_distance,
                        rtol=0.0,
                        atol=1.0e-10,
                    )
                    for distance in unlike_distances
                )
            )
        )
    return counts


def _molecule_positions(
    molecule: Any, bohr_per_angstrom: float
) -> list[np.ndarray]:
    return [
        np.asarray(atom.xyz, dtype=float) / bohr_per_angstrom
        for atom in molecule.atoms
    ]


def test_b1_100_study_builders_extend_to_sixfold_bulk() -> None:
    scan = _load_study("scan_mgo100_slab.py")
    coords, species, translations, primitives = scan["mgo100_slab"](
        _A, 2
    )
    assert np.isclose(
        np.dot(primitives[0], primitives[1]), 0.0, atol=1.0e-12
    )
    assert np.allclose(
        [np.linalg.norm(vector) for vector in primitives],
        [_A / np.sqrt(2.0)] * 2,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.isclose(
        np.linalg.norm(np.cross(*primitives)),
        _A * _A / 2.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    fixtures = [
        (coords, species, [*translations, np.array([0.0, 0.0, _A])]),
    ]

    channel = _load_study("probe_slab_channel_isolation.py")
    molecule, topology = channel["_rocksalt_100_slab"](2)
    bohr = channel["BOHR"]
    channel_coords = _molecule_positions(molecule, bohr)
    channel_species = [atom.Z for atom in molecule.atoms]
    assert len(channel_coords) == len(coords)
    assert np.allclose(channel_coords, coords, rtol=0.0, atol=1.0e-12)
    assert channel_species == species
    assert np.allclose(
        np.asarray(topology.translations) / bohr,
        translations,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert topology.finite_group is not None
    assert np.allclose(
        np.asarray(topology.finite_group.primitive_vectors) / bohr,
        primitives,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert tuple(topology.finite_group.replicas) == (2, 2, 1)
    channel_planes: dict[float, dict[tuple[float, float], int]] = {}
    for position, atomic_number in zip(channel_coords, channel_species):
        plane = channel_planes.setdefault(round(float(position[2]), 12), {})
        site = (
            round(float(position[0]), 12),
            round(float(position[1]), 12),
        )
        plane[site] = atomic_number
    assert len(channel_planes) == 2
    lower, upper = (channel_planes[z] for z in sorted(channel_planes))
    lower_species = list(lower.values())
    upper_species = list(upper.values())
    assert lower_species.count(12) == lower_species.count(8) == 4
    assert upper_species.count(12) == upper_species.count(8) == 4
    assert lower.keys() == upper.keys()
    assert all(lower[site] != upper[site] for site in lower)
    fixtures.append(
        (
            channel_coords,
            channel_species,
            [
                *(
                    np.asarray(vector) / bohr
                    for vector in topology.translations
                ),
                np.array([0.0, 0.0, _A]),
            ],
        )
    )

    odd_molecule, _ = channel["_rocksalt_100_slab"](3)
    odd_channel_species = [atom.Z for atom in odd_molecule.atoms]
    assert len(odd_channel_species) == 24
    assert odd_channel_species.count(12) == odd_channel_species.count(8) == 12

    checkerboard = _load_study("probe_checkerboard_100.py")
    molecule, topology = checkerboard["rocksalt_100_slab"](2)
    bohr = checkerboard["BOHR"]
    fixtures.append(
        (
            _molecule_positions(molecule, bohr),
            [atom.Z for atom in molecule.atoms],
            [
                *(
                    np.asarray(vector) / bohr
                    for vector in topology.translations
                ),
                np.array([0.0, 0.0, _A]),
            ],
        )
    )

    defect = _load_study("defect_ladder_checkerboard.py")
    coords, species = defect["checkerboard_positions"](2, 2)
    t1, t2 = defect["_surface_primitives"]()
    fixtures.append(
        (
            coords,
            species,
            [2.0 * t1, 2.0 * t2, np.array([0.0, 0.0, _A])],
        )
    )

    xtb = _load_study("xtb_checkerboard_reference.py")
    species, coords, lattice = xtb["checkerboard_positions"]()
    fixtures.append(
        (
            coords,
            species,
            [*np.asarray(lattice[:2]), np.array([0.0, 0.0, _A])],
        )
    )

    for coords, species, lattice in fixtures:
        assert set(
            _unlike_coordination(coords, species, lattice, _A / 2.0)
        ) == {6}

    odd_coords, odd_species, _, _ = scan["mgo100_slab"](_A, 3)
    assert len(odd_coords) == 24
    assert odd_species.count(12) == odd_species.count(8) == 12


def test_b2_001_study_builder_extends_to_eightfold_bulk() -> None:
    study = _load_study("probe_checkerboard_100.py")
    molecule, topology = study["b2_001_slab"](2)
    bohr = study["BOHR"]
    lattice = [
        *(np.asarray(vector) / bohr for vector in topology.translations),
        np.array([0.0, 0.0, _A]),
    ]
    counts = _unlike_coordination(
        _molecule_positions(molecule, bohr),
        [atom.Z for atom in molecule.atoms],
        lattice,
        np.sqrt(3.0) * _A / 2.0,
    )
    assert set(counts) == {8}
    assert topology.finite_group is not None
    primitive_vectors = [
        np.asarray(vector) / bohr
        for vector in topology.finite_group.primitive_vectors
    ]
    assert np.allclose(
        primitive_vectors,
        [[_A, 0.0, 0.0], [0.0, _A, 0.0]],
        rtol=0.0,
        atol=1.0e-12,
    )
    positions = _molecule_positions(molecule, bohr)
    assert np.allclose(
        positions[4] - positions[0],
        [_A / 2.0, _A / 2.0, _A / 2.0],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_defect_ladder_uses_the_full_surface_supercell() -> None:
    study = _load_study("defect_ladder_checkerboard.py")
    for factory, args, expected_atoms in (
        (study["pristine"], (), 32),
        (study["o_vacancy"], (), 31),
        (study["vacancy_with_h2o"], (2.0,), 34),
    ):
        molecule, topology = factory(*args)
        assert topology.finite_group is not None
        assert topology.finite_group.order == 1
        assert len(molecule.atoms) == expected_atoms

    adsorbate, _ = study["vacancy_with_h2o"](2.0)
    substrate_z = [atom.xyz[2] for atom in adsorbate.atoms[:-3]]
    adsorbate_z = [atom.xyz[2] for atom in adsorbate.atoms[-3:]]
    assert min(substrate_z) == 0.0
    assert max(adsorbate_z) < 0.0


def test_synthetic_probe_dual_height_bounds_cover_skew_lattice_sphere() -> None:
    study = _load_study("probe_slab_jacobian.py")
    vector_1 = np.array([1.0, 0.0, 0.0])
    vector_2 = np.array([0.99, 0.05, 0.0])
    cutoff = 0.6
    bounds = study["_dual_height_bounds"](vector_1, vector_2, cutoff)

    included = [
        (i, j)
        for i in range(-20, 21)
        for j in range(-20, 21)
        if np.linalg.norm(i * vector_1 + j * vector_2) <= cutoff
    ]
    assert included
    assert max(abs(i) for i, _ in included) <= bounds[0]
    assert max(abs(j) for _, j in included) <= bounds[1]

    old_norm_bounds = (
        int(np.ceil(cutoff / np.linalg.norm(vector_1))) + 1,
        int(np.ceil(cutoff / np.linalg.norm(vector_2))) + 1,
    )
    assert any(
        abs(i) > old_norm_bounds[0] or abs(j) > old_norm_bounds[1]
        for i, j in included
    )
