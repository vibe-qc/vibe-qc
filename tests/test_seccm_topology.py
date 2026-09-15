"""Focused contract tests for the Hamiltonian-independent SECCM topology."""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc.molecule import ANGSTROM_TO_BOHR
from vibeqc.semiempirical import seccm
from vibeqc.semiempirical.methods import msindo_ccm
from vibeqc.semiempirical.seccm import topology as topology_module
from vibeqc.semiempirical.seccm._adapter_common import (
    flatten_topology_records,
)
from vibeqc.semiempirical.seccm.topology import (
    SECCMImage,
    SECCMTopology,
    SECCMTopologyChangedError,
    SECCMTopologyError,
    SECCMTopologyProvenanceError,
    TopologyDiagnosticCode,
    bind_complete_translation_action,
    bind_finite_group,
    build_seccm_topology,
)


def _legacy_rund(value: float, digit: int) -> float:
    half = digit // 2
    rest = digit - half
    expo1 = 10.0**rest
    expo2 = 10.0**half
    sign = 1.0 if value >= 0.0 else -1.0
    integer = int(abs(value * expo1))
    remainder = abs(value * expo1) - integer
    rounded = float(int(remainder * expo2 + 0.5)) / expo2
    return sign * (integer + rounded) / expo1


def _legacy_records(coords, translations):
    """Frozen copy of the pre-extraction Python loop, including loop order."""
    coords = np.asarray(coords, float)
    translations = [np.asarray(vector, float) for vector in translations]
    dimensionality = len(translations)
    zero = np.zeros(3)
    a = translations[0] if dimensionality > 0 else zero
    b = translations[1] if dimensionality > 1 else zero
    c = translations[2] if dimensionality > 2 else zero
    vectors = []
    squared_lengths = []
    for ii, jj, kk in product((-1, 0, 1), repeat=3):
        if (ii, jj, kk) == (0, 0, 0):
            continue
        vector = ii * a + jj * b + kk * c
        vectors.append(vector)
        squared_lengths.append(float(vector @ vector))
    vectors = np.asarray(vectors)
    squared_lengths = np.asarray(squared_lengths)
    nonzero = np.array(
        [_legacy_rund(value, 1) != 0.0 for value in squared_lengths]
    )
    max_distance = float(np.max(0.5 * squared_lengths))

    image_translations = []
    for active_label in product((-1, 0, 1), repeat=dimensionality):
        label = (*active_label, *(0 for _ in range(3 - dimensionality)))
        translation = sum(
            (
                active_label[axis] * translations[axis]
                for axis in range(dimensionality)
            ),
            start=np.zeros(3),
        )
        image_translations.append((label, translation))

    cells = []
    for central in range(len(coords)):
        included = []
        for origin in range(len(coords)):
            for label, translation in image_translations:
                if origin == central and not np.any(translation):
                    continue
                position = coords[origin] + translation
                displacement = position - coords[central]
                if (
                    dimensionality == 3
                    and float(displacement @ displacement) > max_distance
                ):
                    continue
                projections = np.where(
                    nonzero,
                    (vectors @ displacement)
                    / np.where(nonzero, squared_lengths, 1.0),
                    0.0,
                )
                score = abs(float(np.max(projections)))
                if _legacy_rund(score, 4) <= 0.5:
                    included.append((origin, position, label))
        multiplicities = {}
        for origin, _, _ in included:
            multiplicities[origin] = multiplicities.get(origin, 0) + 1
        cells.append(
            [
                (
                    origin,
                    1.0 / multiplicities[origin],
                    position - coords[central],
                    label,
                    multiplicities[origin],
                )
                for origin, position, label in included
            ]
        )
    return cells


def _even_grid(dimensionality: int):
    translations = [
        np.eye(3)[axis] * 4.0 for axis in range(dimensionality)
    ]
    coords = []
    for indices in product((0, 1), repeat=dimensionality):
        position = np.zeros(3)
        for axis, index in enumerate(indices):
            position[axis] = 2.0 * index
        coords.append(position)
    return np.asarray(coords), translations


def _exhaustive_closest_labels(
    coords: np.ndarray,
    translations: list[np.ndarray],
    *,
    bound: int,
) -> dict[tuple[int, int], tuple[tuple[int, int, int], ...]]:
    """Independent finite-box nearest-image oracle for regression cells."""
    dimensionality = len(translations)
    labels = tuple(product(range(-bound, bound + 1), repeat=dimensionality))
    result = {}
    for central in range(len(coords)):
        for origin in range(len(coords)):
            candidates = []
            for active_label in labels:
                label = (*active_label, *(0 for _ in range(3 - dimensionality)))
                translation = sum(
                    (
                        active_label[axis] * translations[axis]
                        for axis in range(dimensionality)
                    ),
                    start=np.zeros(3),
                )
                distance = np.linalg.norm(
                    coords[origin] + translation - coords[central]
                )
                candidates.append((float(distance), label))
            minimum = min(distance for distance, _ in candidates)
            closest = tuple(
                sorted(
                    label
                    for distance, label in candidates
                    if abs(distance - minimum) <= 1.0e-12
                    and not (
                        central == origin and label == (0, 0, 0)
                    )
                )
            )
            result[(central, origin)] = closest
    return result


def _record_labels_by_pair(topology):
    return {
        (central, origin): tuple(
            image.image_shell_label
            for image in topology.cells[central]
            if image.origin == origin
        )
        for central in range(len(topology.cells))
        for origin in range(len(topology.cells))
    }


@pytest.mark.parametrize("dimensionality", [1, 2, 3])
def test_extraction_preserves_legacy_record_order_weights_and_displacements(
    dimensionality,
):
    coords, translations = _even_grid(dimensionality)
    legacy = _legacy_records(coords, translations)
    topology = build_seccm_topology(coords, translations)

    assert msindo_ccm.build_wigner_seitz is topology_module.build_wigner_seitz
    for shared_cell, legacy_cell in zip(topology.cells, legacy, strict=True):
        assert len(shared_cell) == len(legacy_cell)
        for image, expected in zip(shared_cell, legacy_cell, strict=True):
            origin, weight, displacement, label, multiplicity = expected
            assert image.origin == origin
            assert image.weight == weight
            assert np.array_equal(image.disp, displacement)
            assert image.image_shell_label == label
            assert image.ownership_multiplicity == multiplicity


@pytest.mark.parametrize(
    "dimensionality,expected_weights",
    [
        (1, {0.5}),
        (2, {0.25, 0.5}),
        (3, {0.125, 0.25, 0.5}),
    ],
)
def test_face_edge_corner_ties_have_fractional_ownership(
    dimensionality, expected_weights
):
    coords, translations = _even_grid(dimensionality)
    topology = build_seccm_topology(coords, translations)

    weights = {
        image.weight for cell in topology.cells for image in cell
    }
    assert weights == expected_weights
    assert topology.switching_margin == pytest.approx(0.0)
    assert topology.has_reference_ties
    assert all(
        image.weight == 1.0 / image.ownership_multiplicity
        for cell in topology.cells
        for image in cell
    )
    assert TopologyDiagnosticCode.NEAR_SWITCHING_SURFACE in {
        diagnostic.code for diagnostic in topology.diagnostics()
    }


def test_odd_replica_chain_has_no_fractional_ties():
    coords = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )
    topology = build_seccm_topology(coords, [[6.0, 0.0, 0.0]])

    assert not topology.has_reference_ties
    assert all(
        image.weight == 1.0
        for cell in topology.cells
        for image in cell
    )


def test_skew_cell_has_complete_reversal_map():
    translation_a = np.array([4.0, 0.0, 0.0])
    translation_b = np.array([1.0, 3.0, 0.0])
    primitive_a = translation_a / 2.0
    primitive_b = translation_b / 2.0
    coords = np.array(
        [
            np.zeros(3),
            primitive_a,
            primitive_b,
            primitive_a + primitive_b,
        ]
    )
    topology = build_seccm_topology(
        coords, [translation_a, translation_b]
    )

    reversal = topology.reversal_map()
    record_count = sum(len(cell) for cell in topology.cells)
    assert len(reversal) == record_count
    assert all(reversal[reverse] == record for record, reverse in reversal.items())
    assert topology.is_valid(len(coords))


def test_strongly_skew_2d_matches_exhaustive_closest_image_oracle():
    translation_a = np.array([1.0, 0.0, 0.0])
    translation_b = np.array([2.9, 0.1, 0.0])
    translations = [translation_a, translation_b]
    coords = np.array(
        [
            np.zeros(3),
            translation_a / 2.0,
            translation_b / 2.0,
            (translation_a + translation_b) / 2.0,
        ]
    )

    topology = build_seccm_topology(coords, translations)
    oracle = _exhaustive_closest_labels(coords, translations, bound=12)

    assert _record_labels_by_pair(topology) == oracle
    assert oracle[(0, 2)] == ((-7, 2, 0), (7, -3, 0))
    assert [
        np.linalg.norm(image.disp)
        for image in topology.cells[0]
        if image.origin == 2
    ] == pytest.approx([np.sqrt(0.125)] * 2, abs=1.0e-14)
    assert topology.is_valid(len(coords))
    assert len(topology.reversal_map()) == sum(
        len(cell) for cell in topology.cells
    )


def test_strongly_skew_3d_matches_exhaustive_closest_image_oracle():
    reduced = np.diag([4.0, 3.0, 2.0])
    unimodular_map = np.array([[1, 2, -1], [0, 1, 2], [0, 0, 1]])
    translations = unimodular_map @ reduced
    coords = np.array(
        [
            sum(
                (
                    index * reduced[axis] / 2.0
                    for axis, index in enumerate(indices)
                ),
                start=np.zeros(3),
            )
            for indices in product((0, 1), repeat=3)
        ]
    )

    topology = build_seccm_topology(coords, translations)
    oracle = _exhaustive_closest_labels(coords, list(translations), bound=8)

    assert _record_labels_by_pair(topology) == oracle
    assert any(
        abs(component) > 1
        for cell in topology.cells
        for image in cell
        for component in image.image_shell_label
    )
    assert topology.is_valid(len(coords))
    assert len(topology.reversal_map()) == sum(
        len(cell) for cell in topology.cells
    )


@pytest.mark.parametrize(
    "base_translations,unimodular_map",
    [
        (
            np.array([[4.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
            np.array([[1, 4], [0, 1]]),
        ),
        (
            np.array(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            ),
            np.array([[1, 4, -3], [0, 1, 5], [0, 0, 1]]),
        ),
    ],
    ids=["two-dimensional", "three-dimensional"],
)
def test_unimodular_skew_basis_preserves_cartesian_records_and_labels(
    base_translations,
    unimodular_map,
):
    dimensionality = len(base_translations)
    coords = np.array(
        [
            sum(
                (
                    index * base_translations[axis] / 2.0
                    for axis, index in enumerate(indices)
                ),
                start=np.zeros(3),
            )
            for indices in product((0, 1), repeat=dimensionality)
        ]
    )
    skew_translations = unimodular_map @ base_translations
    base = build_seccm_topology(coords, base_translations)
    skew = build_seccm_topology(coords, skew_translations)

    def canonical_records(topology, label_map):
        records = []
        for central, cell in enumerate(topology.cells):
            for image in cell:
                active_label = (
                    np.asarray(image.image_shell_label[:dimensionality])
                    @ label_map
                )
                records.append(
                    (
                        central,
                        image.origin,
                        tuple(int(value) for value in active_label),
                        image.ownership_multiplicity,
                        image.weight,
                        tuple(np.round(image.disp, 13)),
                    )
                )
        return sorted(records)

    assert canonical_records(base, np.eye(dimensionality, dtype=int)) == (
        canonical_records(skew, unimodular_map)
    )
    assert any(
        abs(component) > 1
        for cell in skew.cells
        for image in cell
        for component in image.image_shell_label[:dimensionality]
    )


def test_strongly_skew_complete_action_has_complete_record_orbits():
    translation_a = np.array([1.0, 0.0, 0.0])
    translation_b = np.array([2.9, 0.1, 0.0])
    coords = np.array(
        [
            np.zeros(3),
            translation_a / 2.0,
            translation_b / 2.0,
            (translation_a + translation_b) / 2.0,
        ]
    )
    topology = build_seccm_topology(
        coords,
        [translation_a, translation_b],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[translation_a / 2.0, translation_b / 2.0],
        replicas=(2, 2, 1),
        geometry_tolerance=1.0e-10,
        length_unit="bohr",
    )
    topology = bind_complete_translation_action(
        topology,
        atom_primitive_site_ids=(0, 0, 0, 0),
        atom_cell_labels=((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
        atom_equivalence_keys=(("X",),) * 4,
        equivalence_key_schema="test-site-v1",
    )

    orbits = topology.record_translation_orbits()
    all_records = {
        (central, record_index)
        for central, cell in enumerate(topology.cells)
        for record_index in range(len(cell))
    }
    assert all(len(orbit) == topology.finite_group.order for orbit in orbits)
    assert {record for orbit in orbits for record in orbit} == all_records


def test_native_builder_matches_exact_strongly_skew_python_records():
    indo = pytest.importorskip("vibeqc._vibeqc_core.semiempirical.indo")
    translation_a = np.array([1.0, 0.0, 0.0])
    translation_b = np.array([2.9, 0.1, 0.0])
    translations = np.array([translation_a, translation_b])
    coords = np.array(
        [
            np.zeros(3),
            translation_a / 2.0,
            translation_b / 2.0,
            (translation_a + translation_b) / 2.0,
        ]
    )
    topology = build_seccm_topology(coords, translations)

    native = indo._build_seccm_topology_records_for_testing(
        coords, translations
    )
    expected = [
        (central, record_index, image)
        for central, cell in enumerate(topology.cells)
        for record_index, image in enumerate(cell)
    ]

    assert native["valid"]
    assert len(native["records"]) == len(expected)
    for record, (central, record_index, image) in zip(
        native["records"], expected, strict=True
    ):
        assert record["central"] == central
        assert record["record_index"] == record_index
        assert record["origin"] == image.origin
        assert tuple(record["shell_label"]) == image.image_shell_label
        assert record["ownership_multiplicity"] == (
            image.ownership_multiplicity
        )
        assert record["weight"] == image.weight
        assert record["displacement"] == pytest.approx(
            image.disp, abs=1.0e-14
        )


def test_adapter_validates_bound_record_orbits_before_native_execution(
    monkeypatch,
):
    topology = build_seccm_topology(
        [[0.0, 0.0, 0.0]],
        [[4.0, 0.0, 0.0]],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[[4.0, 0.0, 0.0]],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-10,
        length_unit="bohr",
    )
    topology = bind_complete_translation_action(
        topology,
        atom_primitive_site_ids=(0,),
        atom_cell_labels=((0, 0, 0),),
        atom_equivalence_keys=(("H",),),
        equivalence_key_schema="test-site-v1",
    )
    molecule = Molecule([Atom(1, [0.0, 0.0, 0.0])], 0, 2)

    def fail_orbit_validation(_self):
        raise SECCMTopologyError("translated-record-orbit-sentinel")

    monkeypatch.setattr(
        SECCMTopology,
        "record_translation_orbits",
        fail_orbit_validation,
    )
    with pytest.raises(SECCMTopologyError, match="orbit-sentinel"):
        flatten_topology_records(
            molecule,
            topology,
            route_name="test-seccm",
        )


def test_adapter_preserves_zero_record_matrix_shapes():
    topology = build_seccm_topology(
        [[0.0, 0.0, 0.0]],
        [[20.0, 0.0, 0.0]],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[[20.0, 0.0, 0.0]],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-10,
        length_unit="bohr",
    )
    molecule = Molecule([Atom(2, [0.0, 0.0, 0.0])], 0, 1)

    records = flatten_topology_records(
        molecule,
        topology,
        route_name="test-seccm",
    )
    shell_labels = records[3]
    displacements = records[6]

    assert shell_labels.shape == (0, 3)
    assert shell_labels.dtype == np.int32
    assert displacements.shape == (0, 3)
    assert displacements.dtype == np.dtype(float)


def test_reversal_ambiguity_is_diagnosed_and_fails_closed():
    coords, translations = _even_grid(1)
    topology = deepcopy(build_seccm_topology(coords, translations))
    duplicate = topology.cells[0][0]
    topology.cells[0].append(
        SECCMImage(
            duplicate.origin,
            duplicate.weight,
            duplicate.disp.copy(),
            duplicate.image_shell_label,
            duplicate.ownership_multiplicity,
        )
    )

    codes = {diagnostic.code for diagnostic in topology.diagnostics()}
    assert TopologyDiagnosticCode.DUPLICATE_DISCRETE_RECORD in codes
    assert TopologyDiagnosticCode.AMBIGUOUS_REVERSE_IMAGE in codes
    with pytest.raises(SECCMTopologyError, match="ambiguous"):
        topology.reversal_map()


@pytest.mark.parametrize("delta", [-1.0e-3, 1.0e-3])
def test_exact_face_tie_continues_locally_without_reweighting(delta):
    coords, translations = _even_grid(1)
    topology = build_seccm_topology(coords, translations)
    displaced = coords.copy()
    displaced[1, 0] += delta

    rebuilt = topology.rebuild_displacements(
        displaced, max_tie_score_excursion=3.0e-4
    )

    assert rebuilt.topology_fingerprint == topology.topology_fingerprint
    assert [
        (image.image_shell_label, image.weight)
        for image in rebuilt.cells[0]
    ] == [
        (image.image_shell_label, image.weight)
        for image in topology.cells[0]
    ]


def test_exact_face_tie_rejects_overlarge_continuation():
    coords, translations = _even_grid(1)
    topology = build_seccm_topology(coords, translations)
    displaced = coords.copy()
    displaced[1, 0] += 0.5

    with pytest.raises(SECCMTopologyChangedError, match="exceeded"):
        topology.rebuild_displacements(
            displaced, max_tie_score_excursion=1.0e-2
        )


def test_reference_tie_requires_explicit_positive_finite_bound():
    coords, translations = _even_grid(1)
    topology = build_seccm_topology(coords, translations)

    for invalid in (None, 0.0, -1.0, np.inf):
        with pytest.raises(ValueError, match="positive finite"):
            topology.rebuild_displacements(
                coords, max_tie_score_excursion=invalid
            )


def test_previously_absent_image_entering_fails_closed():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    translations = [np.array([10.0, 0.0, 0.0])]
    topology = build_seccm_topology(coords, translations)
    displaced = coords.copy()
    displaced[1, 0] = 5.0

    with pytest.raises(SECCMTopologyChangedError, match="absent image entered"):
        topology.rebuild_displacements(displaced)


def test_public_rebuild_never_calls_private_unchecked_helper(monkeypatch):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    translations = [np.array([10.0, 0.0, 0.0])]
    topology = build_seccm_topology(coords, translations)
    displaced = coords.copy()
    displaced[1, 0] += 1.0e-3

    def forbidden(*_args, **_kwargs):
        raise AssertionError("public rebuild called unchecked helper")

    monkeypatch.setattr(
        topology_module, "_rebuild_displacements_unchecked", forbidden
    )
    topology.rebuild_displacements(displaced)


def test_rebuild_validates_replacement_translations_before_geometry_use():
    topology = build_seccm_topology(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]
    )

    with pytest.raises(ValueError, match="dimensionality"):
        topology.rebuild_displacements(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        )
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        topology.rebuild_displacements(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[10.0, 0.0]]
        )


def test_package_exports_only_reviewed_public_surface():
    assert set(seccm.__all__) == {
        "AtomEquivalenceKey",
        "DFTB0_SECCM_PARAMETER_SET",
        "DFTB0SECCMResult",
        "FiniteTranslationLabel",
        "ImageShellLabel",
        "SECCMCandidateClassification",
        "SECCMFiniteGroup",
        "SECCMImage",
        "SECCMTopology",
        "SECCMTopologyChangedError",
        "SECCMTopologyError",
        "SECCMTopologyProvenanceError",
        "SECCMTranslationAction",
        "TopologyDiagnostic",
        "TopologyDiagnosticCode",
        "WSNeighbor",
        "WignerSeitzCells",
        "bind_complete_translation_action",
        "bind_finite_group",
        "build_seccm_topology",
        "build_wigner_seitz",
        "SCCDFTB_SECCM_PARAMETER_SET",
        "SCCDFTBSECCMResult",
        "run_dftb0_seccm",
        "run_scc_dftb_seccm",
        "PM6_SECCM_PARAMETER_SET",
        "PM6SECCMResult",
        "run_pm6_seccm",
        "OMxSECCMResult",
        "run_omx_seccm",
        "GFN2_SECCM_PARAMETER_SET",
        "GFN2SECCMAttempt",
        "GFN2SECCMConvergenceError",
        "GFN2SECCMResult",
        "GFN2SECCMStateChange",
        "compare_gfn2_seccm_states",
        "run_gfn2_seccm",
    }
    assert not hasattr(seccm, "_rebuild_displacements_unchecked")
    assert not hasattr(msindo_ccm, "_rebuild_displacements_unchecked")


def _geometry_fingerprint(x: float, quantum: float = 1.0e-3) -> str:
    topology = build_seccm_topology(
        [[x, 0.0, 0.0]],
        [[4.0, 0.0, 0.0]],
        length_unit="bohr",
        geometry_quantum=quantum,
    )
    return topology.reference_geometry_fingerprint


def test_geometry_fingerprint_normalizes_signed_zero():
    assert _geometry_fingerprint(0.0) == _geometry_fingerprint(-0.0)


def test_geometry_fingerprint_uses_half_away_from_zero_quantization():
    quantum = 1.0e-3
    assert _geometry_fingerprint(0.5 * quantum) == _geometry_fingerprint(
        0.6 * quantum
    )
    assert _geometry_fingerprint(-0.5 * quantum) == _geometry_fingerprint(
        -0.6 * quantum
    )
    assert _geometry_fingerprint(0.5 * quantum) != _geometry_fingerprint(
        0.49 * quantum
    )
    assert _geometry_fingerprint(-0.5 * quantum) != _geometry_fingerprint(
        -0.49 * quantum
    )


def test_geometry_fingerprint_changes_only_after_quantized_geometry_changes():
    assert _geometry_fingerprint(0.1e-3) == _geometry_fingerprint(0.4e-3)
    assert _geometry_fingerprint(0.1e-3) != _geometry_fingerprint(0.6e-3)


def test_reference_arrays_are_copied_and_read_only():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    translation = np.array([10.0, 0.0, 0.0])
    topology = build_seccm_topology(coords, [translation])
    coords[0, 0] = 3.0
    translation[0] = 12.0

    assert topology.reference_coords[0, 0] == 0.0
    assert topology.translations[0][0] == 10.0
    with pytest.raises(ValueError, match="read-only"):
        topology.reference_coords[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        topology.translations[0][0] = 1.0


def test_prefiltered_candidates_have_nullable_score_fields():
    coords, translations = _even_grid(3)
    topology = build_seccm_topology(coords, translations)
    prefiltered = [
        candidate
        for candidate in topology.candidate_classifications
        if candidate.prefiltered
    ]
    assert prefiltered
    assert all(not candidate.legacy_inside for candidate in prefiltered)
    assert all(
        candidate.legacy_membership_score is None
        and candidate.signed_margin is None
        and candidate.absolute_margin is None
        for candidate in prefiltered
    )


def test_compatibility_construction_fails_provenance_operations_explicitly():
    compatibility = SECCMTopology(
        cells=[[SECCMImage(1, 1.0, np.ones(3))], []]
    )
    codes = {diagnostic.code for diagnostic in compatibility.diagnostics(2)}
    assert TopologyDiagnosticCode.MISSING_BUILDER_PROVENANCE in codes

    with pytest.raises(
        SECCMTopologyProvenanceError, match="missing_builder_provenance"
    ):
        _ = compatibility.topology_fingerprint
    with pytest.raises(SECCMTopologyProvenanceError):
        compatibility.reversal_map()
    with pytest.raises(SECCMTopologyProvenanceError):
        compatibility.rebuild_displacements(np.zeros((2, 3)))


def test_unbound_group_and_orbit_properties_fail_closed():
    topology = build_seccm_topology(
        [[0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]
    )
    assert not topology.finite_group_bound
    assert not topology.current_lattice_group_compatible
    assert topology.current_atom_geometry_group_covariant is None
    assert not topology.current_geometry_group_covariant
    with pytest.raises(NotImplementedError, match="explicit lattice binding"):
        _ = topology.finite_group_labels
    with pytest.raises(SECCMTopologyError, match="atom_action_not_bound"):
        _ = topology.translation_orbits


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"length_unit": "atomic"}, "bohr.*angstrom"),
        ({"geometry_quantum": 1.0e-8}, "explicit length_unit"),
        (
            {"length_unit": "bohr", "geometry_quantum": 0.0},
            "positive and finite",
        ),
    ],
)
def test_geometry_fingerprint_metadata_fails_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_seccm_topology(
            [[0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]], **kwargs
        )


# ---------------------------------------------------------------------------
# GitLab #669: the equidistant-image window is 1.0e-5 bohr, a physical length.
# ---------------------------------------------------------------------------

_GAP_CELL_EDGE_BOHR = 6.0


def _two_atom_gap_cell(gap_bohr: float, *, unit_scale: float = 1.0):
    """Two atoms in a cubic cell, the second ``gap_bohr / 2`` past the face
    centre, so its ``(0, 0, 0)`` and ``(-1, 0, 0)`` images differ in distance
    to the first atom by exactly ``gap_bohr``.  ``unit_scale`` is bohr per
    returned length unit."""
    edge = _GAP_CELL_EDGE_BOHR
    coords = np.array(
        [[0.0, 0.0, 0.0], [edge / 2.0 + gap_bohr / 2.0, 0.0, 0.0]]
    ) / unit_scale
    translations = [np.eye(3)[axis] * edge / unit_scale for axis in range(3)]
    return coords, translations


def _record_multiset(topology):
    return sorted(
        (central, image.origin, image.image_shell_label, round(image.weight, 12))
        for central, cell in enumerate(topology.cells)
        for image in cell
    )


def _rocksalt_mgo_2x2x2_bohr(dilation: float = 0.0, a_angstrom: float = 4.212):
    """The #592 fixture: rocksalt MgO, fcc primitive cell, 2x2x2 cyclic cluster
    (16 atoms), translations twice the primitive vectors.  ``dilation`` scales
    the coordinates only, so ties open by a controlled relative amount."""
    a = a_angstrom * ANGSTROM_TO_BOHR
    primitive = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    coords = []
    for indices in product(range(2), repeat=3):
        shift = np.asarray(indices, dtype=float) @ primitive
        coords.append(shift)
        coords.append(shift + 0.5 * a * np.array([1.0, 0.0, 0.0]))
    coords = np.asarray(coords) * (1.0 + dilation)
    translations = [2.0 * primitive[axis] for axis in range(3)]
    return coords, translations


def _in_angstrom(coords, translations):
    return coords / ANGSTROM_TO_BOHR, [t / ANGSTROM_TO_BOHR for t in translations]


@pytest.mark.parametrize(
    "gap_bohr, tied",
    [(0.9e-5, True), (1.1e-5, False), (1.8e-5, False), (2.0e-5, False)],
)
def test_equidistant_window_is_one_e_minus_five_bohr_in_every_length_unit(
    gap_bohr, tied
):
    """Identical physical coordinates classify identically in bohr and in
    angstrom on both sides of the 1e-5 bohr window.  Pre-fix an angstrom-tagged
    build applied the raw 1e-5 in angstrom (1.89e-5 bohr): the 1.1e-5 and
    1.8e-5 bohr gaps stayed tied in angstrom while the bohr build split them."""
    bohr = build_seccm_topology(*_two_atom_gap_cell(gap_bohr), length_unit="bohr")
    angstrom = build_seccm_topology(
        *_two_atom_gap_cell(gap_bohr, unit_scale=ANGSTROM_TO_BOHR),
        length_unit="angstrom",
    )
    assert _record_multiset(bohr) == _record_multiset(angstrom)
    weights = sorted({record[3] for record in _record_multiset(bohr)})
    assert weights == ([0.5] if tied else [1.0])
    assert bohr.has_reference_ties is tied
    assert angstrom.has_reference_ties is tied


def test_untagged_topology_keeps_the_bohr_convention():
    """``length_unit=None`` is the bohr convention of every vibe-qc route and of
    the native builder; its window is unchanged."""
    gap = 1.1e-5
    untagged = build_seccm_topology(*_two_atom_gap_cell(gap))
    bohr = build_seccm_topology(*_two_atom_gap_cell(gap), length_unit="bohr")
    assert _record_multiset(untagged) == _record_multiset(bohr)
    assert not untagged.has_reference_ties


@pytest.mark.parametrize("rebuilt_gap, enters", [(1.5e-5, False), (0.9e-5, True)])
def test_rebuild_displacements_judges_new_images_at_the_physical_window(
    rebuilt_gap, enters
):
    """Second call site: the frozen-topology rebuild re-enumerates the current
    closest images with the same window.  The reference gap 2.0e-5 bohr has no
    tie.  Closing it to 1.5e-5 bohr stays above the window, so no image newly
    enters in either unit (pre-fix the angstrom topology saw the second image
    inside its 1.89e-5 bohr window and failed closed); closing it to 0.9e-5
    bohr lets the second image in, and both units must fail closed with
    ``previously absent image entered``."""
    reference_gap = 2.0e-5
    bohr = build_seccm_topology(
        *_two_atom_gap_cell(reference_gap), length_unit="bohr"
    )
    angstrom = build_seccm_topology(
        *_two_atom_gap_cell(reference_gap, unit_scale=ANGSTROM_TO_BOHR),
        length_unit="angstrom",
    )
    bohr_coords = _two_atom_gap_cell(rebuilt_gap)[0]
    angstrom_coords = _two_atom_gap_cell(
        rebuilt_gap, unit_scale=ANGSTROM_TO_BOHR
    )[0]
    if enters:
        for topology, coords in ((bohr, bohr_coords), (angstrom, angstrom_coords)):
            with pytest.raises(
                SECCMTopologyChangedError, match="previously absent image entered"
            ):
                topology.rebuild_displacements(coords)
        return
    bohr_rebuilt = bohr.rebuild_displacements(bohr_coords)
    angstrom_rebuilt = angstrom.rebuild_displacements(angstrom_coords)
    assert _record_multiset(bohr_rebuilt) == _record_multiset(angstrom_rebuilt)
    assert _record_multiset(bohr_rebuilt) == _record_multiset(bohr)


@pytest.mark.parametrize("dilation", [5.0e-7, 1.0e-6, 1.2e-6])
def test_mgo_2x2x2_dilated_topology_is_unit_independent(dilation):
    """The issue's evidence.  A 1e-6 coordinate-only dilation of the #592 MgO
    cell moves coordinates by up to 1.194e-5 bohr (6.32e-6 Å): above the
    physical window, below the raw number read as angstrom.  Pre-fix this
    fixture classified as 408 records in bohr (weights 1/5, 1/4, 1/3, 1/2, 1;
    the issue's own bohr count) against 512 in angstrom (the issue: 510), and
    376 against 480 at 1.2e-6; at 5e-7 both units already agreed (512).  Both
    units must agree at every dilation."""
    coords, translations = _rocksalt_mgo_2x2x2_bohr(dilation)
    bohr = build_seccm_topology(coords, translations, length_unit="bohr")
    angstrom = build_seccm_topology(
        *_in_angstrom(coords, translations), length_unit="angstrom"
    )
    assert _record_multiset(bohr) == _record_multiset(angstrom)
    assert bohr.is_valid(len(coords)) and angstrom.is_valid(len(coords))


def test_mgo_2x2x2_undilated_topology_keeps_the_592_record_set():
    """The #592 closure: the undilated cell shares every equidistant image in
    both units (512 records, Evjen weights 1/6, 1/4, 1/2, 1)."""
    coords, translations = _rocksalt_mgo_2x2x2_bohr()
    bohr = build_seccm_topology(coords, translations, length_unit="bohr")
    angstrom = build_seccm_topology(
        *_in_angstrom(coords, translations), length_unit="angstrom"
    )
    assert _record_multiset(bohr) == _record_multiset(angstrom)
    assert len(_record_multiset(bohr)) == 512
    assert sorted({record[3] for record in _record_multiset(bohr)}) == [
        pytest.approx(1.0 / 6.0), 0.25, 0.5, 1.0
    ]


def test_seccm_topology_build_is_not_the_wall_clock_bottleneck():
    """Issue #423: the serial Python build used to dominate the SECCM route.

    The build enumerates every (central, origin) pair against a fixed
    5**dim candidate-label box.  Until 2026-09-07 it did that one candidate
    at a time in Python -- a `linalg.solve` per pair, a `norm` and a small
    matrix-vector product per candidate, and a dataclass allocation per
    candidate -- which put ~99 % of the build in one function and made the
    build, not the parallel SCC, the wall-clock bottleneck.  That is why
    GFN2-SECCM showed a 0.94-1.44x speedup on 20 threads: the dominant
    phase was serial Python.

    Measured on rocksalt MgO (this machine, scalar -> vectorised):

        atoms      build s        build s      build/SCC
                   (scalar)    (vectorised)   (vectorised)
           16        0.629           0.090           0.15
           54        7.062           0.871           0.04
          128       69.239           5.487           0.02

    The bound below is deliberately loose -- roughly 4x the measured time,
    so ordinary machine-to-machine variation cannot flake it -- but the
    scalar path took 69 s on the same cell and would fail it by an order of
    magnitude.  What it pins is that no per-candidate Python loop has come
    back.
    """
    import time

    coords, translations = _rocksalt_mgo(4)
    start = time.perf_counter()
    topology = build_seccm_topology(
        coords, translations, length_unit="bohr", geometry_quantum=1.0e-10
    )
    elapsed = time.perf_counter() - start

    assert len(coords) == 128
    assert elapsed < 25.0, f"SECCM topology build regressed: {elapsed:.1f}s"

    # Machine-independent companion: the candidate inventory is exactly the
    # pair grid times the fixed label box, minus the skipped self-image at
    # the origin.  A change here means the enumeration itself changed shape,
    # which no optimisation should do.
    assert len(topology._reference_candidates) == 128 * 128 * 125 - 128


def _rocksalt_mgo(n: int, a_bohr: float = 4.212 * 1.8897261254535):
    """n x n x n rocksalt MgO coordinates and cyclic translations, in bohr."""
    prim = [
        np.array([0.0, a_bohr / 2, a_bohr / 2]),
        np.array([a_bohr / 2, 0.0, a_bohr / 2]),
        np.array([a_bohr / 2, a_bohr / 2, 0.0]),
    ]
    basis = [np.zeros(3), np.array([a_bohr / 2, 0.0, 0.0])]
    coords = [
        i * prim[0] + j * prim[1] + k * prim[2] + site
        for i in range(n)
        for j in range(n)
        for k in range(n)
        for site in basis
    ]
    return coords, [n * vector for vector in prim]
