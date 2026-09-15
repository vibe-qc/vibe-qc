"""Finite-group and complete static-site action tests for SECCM T1b."""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from vibeqc.semiempirical.seccm import (
    SECCMFiniteGroup,
    SECCMTopologyError,
    TopologyDiagnosticCode,
    bind_complete_translation_action,
    bind_finite_group,
    build_seccm_topology,
)


def _primitive_vectors(dimensionality: int, scale: float = 1.0):
    vectors = (
        np.array([2.0, 0.0, 0.0]),
        np.array([0.5, 1.5, 0.0]),
        np.array([0.2, 0.3, 1.7]),
    )
    return tuple(scale * vector for vector in vectors[:dimensionality])


def _padded_replicas(*active: int) -> tuple[int, int, int]:
    return (*active, *(1 for _ in range(3 - len(active))))


def _labels(replicas: tuple[int, int, int]):
    return tuple(product(*(range(replica) for replica in replicas)))


def _coords(primitive_vectors, replicas):
    return np.asarray(
        [
            sum(
                (
                    label[axis] * primitive_vectors[axis]
                    for axis in range(len(primitive_vectors))
                ),
                start=np.zeros(3),
            )
            for label in _labels(replicas)
        ]
    )


def _group_only(
    replicas: tuple[int, int, int],
    *,
    scale: float = 1.0,
    coords=None,
):
    dimensionality = sum(replica != 1 for replica in replicas)
    if dimensionality == 0:
        dimensionality = 1
    primitive_vectors = _primitive_vectors(dimensionality, scale)
    translations = [
        replicas[axis] * primitive_vectors[axis]
        for axis in range(dimensionality)
    ]
    if coords is None:
        coords = _coords(primitive_vectors, replicas)
    topology = build_seccm_topology(
        coords,
        translations,
        length_unit="bohr",
        geometry_quantum=1.0e-8,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=primitive_vectors,
        replicas=replicas,
        geometry_tolerance=1.0e-9,
        length_unit="BOHR",
    )


def _complete_action(
    replicas: tuple[int, int, int],
    *,
    scale: float = 1.0,
    atom_equivalence_keys=None,
):
    topology = _group_only(replicas, scale=scale)
    labels = topology.finite_group.labels
    if atom_equivalence_keys is None:
        atom_equivalence_keys = (("Z=6", "parameters=test-v1"),) * len(labels)
    return bind_complete_translation_action(
        topology,
        atom_primitive_site_ids=(0,) * len(labels),
        atom_cell_labels=labels,
        atom_equivalence_keys=atom_equivalence_keys,
        equivalence_key_schema="test-static-site-v1",
    )


@pytest.mark.parametrize(
    "replicas",
    [
        _padded_replicas(5),
        _padded_replicas(4, 3),
        _padded_replicas(3, 2, 5),
    ],
)
def test_finite_group_algebra_is_canonical_and_exact(replicas):
    group = _group_only(replicas).finite_group
    assert group.labels == _labels(replicas)
    assert group.order == np.prod(replicas)

    zero = (0, 0, 0)
    for left in group.labels:
        assert group.add(left, zero) == left
        assert group.add(left, group.inverse(left)) == zero
        for right in group.labels:
            assert group.subtract(group.add(left, right), right) == left


def test_even_nyquist_signed_representatives_are_geometric_views_only():
    group = SECCMFiniteGroup(
        primitive_vectors=_primitive_vectors(3),
        replicas=(4, 6, 2),
        geometry_tolerance=1.0e-9,
        length_unit="angstrom",
    )

    assert group.signed_representatives((2, 3, 1)) == tuple(
        product((-2, 2), (-3, 3), (-1, 1))
    )
    assert group.signed_representatives((3, 5, 0)) == ((-1, -1, 0),)
    assert len(group.labels) == group.order


@pytest.mark.parametrize("label", [(-1, 0, 0), (4, 0, 0), (1.0, 0, 0), (True, 0, 0)])
def test_noncanonical_group_labels_fail_without_modulo_normalization(label):
    group = _group_only((4, 1, 1)).finite_group
    with pytest.raises(SECCMTopologyError, match="noncanonical_cell_label"):
        group.add(label, (0, 0, 0))


def test_group_binding_copies_skew_vectors_and_normalizes_unit():
    topology = _group_only((2, 3, 1))
    group = topology.finite_group

    assert group.length_unit == "bohr"
    assert group.dimensionality == 2
    assert topology.finite_group_labels == group.labels
    with pytest.raises(ValueError, match="read-only"):
        group.primitive_vectors[0][0] = 3.0


@pytest.mark.parametrize(
    "primitive_vectors,replicas,match",
    [
        (_primitive_vectors(1), (2, 2, 1), "finite_group_dimension_mismatch"),
        (
            (np.array([2.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])),
            (2, 2, 1),
            "invalid_primitive_vectors",
        ),
        (_primitive_vectors(2), (2, 2, 2), "invalid_replicas"),
        (_primitive_vectors(2, 1.1), (2, 2, 1), "cyclic_translation_mismatch"),
    ],
)
def test_group_binding_rejects_invalid_lattice_contracts(
    primitive_vectors, replicas, match
):
    base_vectors = _primitive_vectors(2)
    topology = build_seccm_topology(
        _coords(base_vectors, (2, 2, 1)),
        [2 * base_vectors[0], 2 * base_vectors[1]],
    )
    with pytest.raises(SECCMTopologyError, match=match):
        bind_finite_group(
            topology,
            primitive_vectors=primitive_vectors,
            replicas=replicas,
            geometry_tolerance=1.0e-9,
            length_unit="bohr",
        )


def test_group_only_defect_topology_does_not_claim_an_atom_action():
    primitive = _primitive_vectors(1)
    defective_coords = _coords(primitive, (3, 1, 1))[:2]
    topology = _group_only((3, 1, 1), coords=defective_coords)

    assert topology.finite_group_bound
    assert not topology.complete_atom_action_bound
    assert topology.current_lattice_group_compatible
    assert topology.current_atom_geometry_group_covariant is None
    assert not topology.current_geometry_group_covariant
    assert not topology.record_orbits_currently_usable
    with pytest.raises(SECCMTopologyError, match="atom_action_not_bound"):
        topology.record_translation_orbits()


def test_complete_action_builds_exact_group_permutations():
    topology = _complete_action((3, 2, 1))
    group = topology.finite_group
    identity = tuple(range(len(topology.cells)))

    assert topology.complete_atom_action_bound
    assert topology.current_lattice_group_compatible
    assert topology.current_atom_geometry_group_covariant is True
    assert topology.atom_translation_permutation((0, 0, 0)) == identity
    for left in group.labels:
        left_permutation = topology.atom_translation_permutation(left)
        inverse = topology.atom_translation_permutation(group.inverse(left))
        assert tuple(inverse[index] for index in left_permutation) == identity
        for right in group.labels:
            right_permutation = topology.atom_translation_permutation(right)
            combined = topology.atom_translation_permutation(
                group.add(left, right)
            )
            assert tuple(
                right_permutation[index] for index in left_permutation
            ) == combined


@pytest.mark.parametrize(
    "keys,match",
    [
        (
            (("Z=6", "parameters=a-v1"), ("Z=6", "parameters=b-v1")),
            "atom_equivalence_key_mismatch",
        ),
        ((("Z=6",), object()), "invalid_atom_equivalence_key"),
        ((("Z=6",),), "missing_atom_equivalence_key"),
    ],
)
def test_complete_action_rejects_unknown_or_inequivalent_static_sites(keys, match):
    topology = _group_only((2, 1, 1))
    with pytest.raises(SECCMTopologyError, match=match):
        bind_complete_translation_action(
            topology,
            atom_primitive_site_ids=(0, 0),
            atom_cell_labels=((0, 0, 0), (1, 0, 0)),
            atom_equivalence_keys=keys,
            equivalence_key_schema="test-static-site-v1",
        )


def test_complete_action_rejects_missing_duplicate_and_noncanonical_copies():
    topology = _group_only((2, 1, 1))
    kwargs = {
        "atom_primitive_site_ids": (0, 0),
        "atom_equivalence_keys": (("Z=6",), ("Z=6",)),
        "equivalence_key_schema": "test-static-site-v1",
    }
    with pytest.raises(SECCMTopologyError, match="duplicate_cell_copy"):
        bind_complete_translation_action(
            topology,
            atom_cell_labels=((0, 0, 0), (0, 0, 0)),
            **kwargs,
        )
    with pytest.raises(SECCMTopologyError, match="noncanonical_cell_label"):
        bind_complete_translation_action(
            topology,
            atom_cell_labels=((0, 0, 0), (2, 0, 0)),
            **kwargs,
        )


def test_complete_action_rejects_coordinate_mismatch_and_invalid_schema():
    primitive = _primitive_vectors(1)
    mismatched_coords = _coords(primitive, (2, 1, 1))
    mismatched_coords[1, 0] += 1.0e-4
    topology = _group_only((2, 1, 1), coords=mismatched_coords)
    kwargs = {
        "atom_primitive_site_ids": (0, 0),
        "atom_cell_labels": ((0, 0, 0), (1, 0, 0)),
        "atom_equivalence_keys": (("Z=6",), ("Z=6",)),
    }
    with pytest.raises(
        SECCMTopologyError, match="coordinate_covariance_mismatch"
    ):
        bind_complete_translation_action(
            topology,
            equivalence_key_schema="test-static-site-v1",
            **kwargs,
        )
    with pytest.raises(
        SECCMTopologyError, match="invalid_equivalence_key_schema"
    ):
        bind_complete_translation_action(
            topology,
            equivalence_key_schema="unversioned",
            **kwargs,
        )


@pytest.mark.parametrize("replicas", [(3, 1, 1), (2, 3, 1), (2, 2, 3)])
def test_record_translation_maps_form_exact_orbits_for_skew_cells(replicas):
    topology = _complete_action(replicas)
    group = topology.finite_group
    record_keys = {
        (central, record_index)
        for central, cell in enumerate(topology.cells)
        for record_index in range(len(cell))
    }

    maps = {shift: topology.record_translation_map(shift) for shift in group.labels}
    assert all(set(mapping) == record_keys for mapping in maps.values())
    assert all(set(mapping.values()) == record_keys for mapping in maps.values())
    for left in group.labels:
        for right in group.labels:
            combined = maps[group.add(left, right)]
            assert all(
                maps[right][maps[left][record]] == combined[record]
                for record in record_keys
            )
    orbits = topology.record_translation_orbits()
    assert set().union(*(set(orbit) for orbit in orbits)) == record_keys
    assert sum(len(orbit) for orbit in orbits) == len(record_keys)


def test_record_shell_wrap_equation_covers_all_wrap_cases_and_nyquist_shift():
    topology = _complete_action((4, 1, 1))
    action = topology.translation_action
    shift = (2, 0, 0)
    mapping = topology.record_translation_map(shift)
    wrap_cases = set()

    for source, target in mapping.items():
        central, record_index = source
        image = topology.cells[central][record_index]
        central_wrap = (action.atom_cell_labels[central][0] + shift[0]) // 4
        origin_wrap = (action.atom_cell_labels[image.origin][0] + shift[0]) // 4
        wrap_cases.add((central_wrap, origin_wrap))
        translated = topology.cells[target[0]][target[1]]
        assert translated.image_shell_label[0] == (
            image.image_shell_label[0] + origin_wrap - central_wrap
        )

    assert wrap_cases == {(0, 0), (0, 1), (1, 0), (1, 1)}


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "missing_translated_record"),
        ("duplicate", "ambiguous_translated_record"),
        ("ownership", "translated_record_ownership_mismatch"),
        ("displacement", "translated_record_displacement_mismatch"),
    ],
)
def test_malformed_translated_records_fail_closed(mutation, match):
    topology = deepcopy(_complete_action((3, 1, 1)))
    translated = topology.record_translation_map((1, 0, 0))
    source, target = next(
        (source, target) for source, target in translated.items() if source != target
    )
    target_image = topology.cells[target[0]][target[1]]

    if mutation == "missing":
        topology.cells[target[0]].pop(target[1])
    elif mutation == "duplicate":
        topology.cells[target[0]].append(deepcopy(target_image))
    elif mutation == "ownership":
        target_image.ownership_multiplicity += 1
    else:
        target_image.disp = target_image.disp + np.array([1.0e-4, 0.0, 0.0])

    with pytest.raises(SECCMTopologyError, match=match):
        topology.record_translation_map((1, 0, 0))


def test_single_atom_rebuild_retains_reference_action_but_gates_orbits():
    topology = _complete_action((3, 1, 1))
    displaced_coords = np.array(topology.reference_coords, copy=True)
    displaced_coords[1, 0] += 1.0e-4

    rebuilt = topology.rebuild_displacements(displaced_coords)

    assert rebuilt.finite_group is topology.finite_group
    assert rebuilt.translation_action is topology.translation_action
    assert rebuilt.current_lattice_group_compatible
    assert rebuilt.current_atom_geometry_group_covariant is False
    assert not rebuilt.current_geometry_group_covariant
    assert not rebuilt.record_orbits_currently_usable
    assert rebuilt.atom_translation_permutation((1, 0, 0))
    assert TopologyDiagnosticCode.CURRENT_GEOMETRY_NOT_TRANSLATION_COVARIANT in {
        diagnostic.code for diagnostic in rebuilt.diagnostics()
    }
    with pytest.raises(
        SECCMTopologyError,
        match="current_geometry_not_translation_covariant",
    ):
        rebuilt.record_translation_orbits()

    restored = rebuilt.rebuild_displacements(topology.reference_coords)
    assert restored.current_lattice_group_compatible
    assert restored.current_atom_geometry_group_covariant is True
    assert restored.current_geometry_group_covariant
    assert restored.topology_fingerprint == topology.topology_fingerprint


def test_incompatible_cyclic_translation_sets_both_current_states_false():
    topology = _complete_action((3, 1, 1))
    incompatible_translation = np.array(topology.translations[0], copy=True)
    incompatible_translation[0] += 1.0e-4

    rebuilt = topology.rebuild_displacements(
        topology.reference_coords, [incompatible_translation]
    )

    assert not rebuilt.current_lattice_group_compatible
    assert rebuilt.current_atom_geometry_group_covariant is False
    assert not rebuilt.record_orbits_currently_usable

    restored = rebuilt.rebuild_displacements(
        topology.reference_coords, topology.translations
    )
    assert restored.current_lattice_group_compatible
    assert restored.current_atom_geometry_group_covariant is True


def test_fingerprints_include_bound_identity_but_not_current_validity():
    unbound_a = build_seccm_topology(
        _coords(_primitive_vectors(1), (3, 1, 1)),
        [3 * _primitive_vectors(1)[0]],
        length_unit="bohr",
        geometry_quantum=1.0e-8,
    )
    unbound_b = build_seccm_topology(
        _coords(_primitive_vectors(1, 2.0), (3, 1, 1)),
        [3 * _primitive_vectors(1, 2.0)[0]],
        length_unit="bohr",
        geometry_quantum=1.0e-8,
    )
    assert unbound_a.topology_fingerprint == unbound_b.topology_fingerprint

    bound_a = _complete_action((3, 1, 1))
    bound_b = _complete_action((3, 1, 1), scale=2.0)
    assert bound_a.topology_fingerprint == bound_b.topology_fingerprint
    assert (
        bound_a.reference_geometry_fingerprint
        != bound_b.reference_geometry_fingerprint
    )

    changed_keys = (("Z=6", "parameters=other-v1"),) * 3
    changed_action = _complete_action(
        (3, 1, 1), atom_equivalence_keys=changed_keys
    )
    assert (
        changed_action.translation_action.fingerprint
        != bound_a.translation_action.fingerprint
    )
    assert changed_action.topology_fingerprint != bound_a.topology_fingerprint

    displaced = np.array(bound_a.reference_coords, copy=True)
    displaced[1, 0] += 1.0e-4
    rebuilt = bound_a.rebuild_displacements(displaced)
    assert rebuilt.topology_fingerprint == bound_a.topology_fingerprint
    assert (
        rebuilt.reference_geometry_fingerprint
        == bound_a.reference_geometry_fingerprint
    )


def test_unicode_nfc_equivalence_keys_have_stable_identity():
    composed = (("role=caf\u00e9",),) * 2
    decomposed = (("role=cafe\u0301",),) * 2
    composed_topology = _complete_action(
        (2, 1, 1), atom_equivalence_keys=composed
    )
    decomposed_topology = _complete_action(
        (2, 1, 1), atom_equivalence_keys=decomposed
    )

    assert (
        composed_topology.translation_action.atom_equivalence_keys
        == decomposed_topology.translation_action.atom_equivalence_keys
    )
    assert (
        composed_topology.translation_action.fingerprint
        == decomposed_topology.translation_action.fingerprint
    )
