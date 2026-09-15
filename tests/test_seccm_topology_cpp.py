"""Parity tests for the shared C++ SECCM Wigner-Seitz topology."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from vibeqc.semiempirical.seccm.topology import build_seccm_topology


_ROOT = Path(__file__).resolve().parents[1]
_PRIMITIVE_VECTORS = (
    np.array([2.0, 0.0, 0.0]),
    np.array([0.55, 1.7, 0.0]),
    np.array([0.25, 0.35, 1.5]),
)


def _bridge():
    indo = pytest.importorskip("vibeqc._vibeqc_core.semiempirical.indo")
    return indo._build_seccm_topology_records_for_testing


def _translation_grid(
    dimensionality: int,
    replicas: int,
    *,
    skew: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if skew:
        primitive_vectors = _PRIMITIVE_VECTORS[:dimensionality]
    else:
        primitive_vectors = tuple(
            2.0 * np.eye(3)[axis] for axis in range(dimensionality)
        )
    coords = np.array(
        [
            sum(
                (
                    index * vector
                    for index, vector in zip(
                        indices, primitive_vectors, strict=True
                    )
                ),
                start=np.zeros(3),
            )
            for indices in product(range(replicas), repeat=dimensionality)
        ]
    )
    translations = np.array(
        [replicas * vector for vector in primitive_vectors]
    )
    return coords, translations


def _assert_python_cpp_parity(
    coords: np.ndarray, translations: np.ndarray
) -> dict:
    topology = build_seccm_topology(coords, translations)
    result = _bridge()(coords, translations)

    assert result["schema"] == "seccm-ws-cxx-records-v1"
    assert result["valid"] is topology.is_valid(len(coords))
    assert result["total_weights"] == [
        topology.total_weight(central) for central in range(len(coords))
    ]

    expected_records = [
        (central, record_index, image)
        for central, cell in enumerate(topology.cells)
        for record_index, image in enumerate(cell)
    ]
    assert len(result["records"]) == len(expected_records)
    for record, (central, record_index, image) in zip(
        result["records"], expected_records, strict=True
    ):
        assert record["central"] == central
        assert record["record_index"] == record_index
        assert record["origin"] == image.origin
        assert tuple(record["shell_label"]) == image.image_shell_label
        assert (
            record["ownership_multiplicity"]
            == image.ownership_multiplicity
        )
        assert record["weight"] == image.weight
        assert np.array_equal(record["displacement"], image.disp)
    return result


@pytest.mark.parametrize("dimensionality", [1, 2, 3])
@pytest.mark.parametrize("replicas", [2, 3], ids=["even", "odd"])
def test_cpp_records_match_python_for_skew_odd_and_even_groups(
    dimensionality,
    replicas,
):
    coords, translations = _translation_grid(
        dimensionality, replicas, skew=True
    )
    result = _assert_python_cpp_parity(coords, translations)

    assert result["valid"]


@pytest.mark.parametrize(
    "dimensionality,expected_weights",
    [
        (1, {0.5}),
        (2, {0.25, 0.5}),
        (3, {0.125, 0.25, 0.5}),
    ],
)
def test_cpp_records_pin_face_edge_corner_fractional_ownership(
    dimensionality,
    expected_weights,
):
    coords, translations = _translation_grid(
        dimensionality, 2, skew=False
    )
    result = _assert_python_cpp_parity(coords, translations)

    assert {record["weight"] for record in result["records"]} == (
        expected_weights
    )
    assert {
        record["ownership_multiplicity"] for record in result["records"]
    } == {round(1.0 / weight) for weight in expected_weights}


@pytest.mark.parametrize(
    "coords,translations,message",
    [
        (np.empty((0, 3)), np.eye(1, 3), "coordinates must have shape"),
        (np.zeros((1, 2)), np.eye(1, 3), "coordinates must have shape"),
        (
            np.array([[np.nan, 0.0, 0.0]]),
            np.eye(1, 3),
            "only finite values",
        ),
        (np.zeros((1, 3)), np.empty((0, 3)), "translations must have shape"),
        (np.zeros((1, 3)), np.zeros((1, 2)), "translations must have shape"),
        (np.zeros((1, 3)), np.zeros((4, 3)), "translations must have shape"),
        (
            np.zeros((1, 3)),
            np.array([[np.inf, 0.0, 0.0]]),
            "only finite values",
        ),
        (np.zeros((1, 3)), np.zeros((1, 3)), "nonzero and linearly independent"),
        (
            np.zeros((1, 3)),
            np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
            "nonzero and linearly independent",
        ),
        (
            np.zeros((1, 3)),
            np.array([[4.0, 0.0, 0.0], [4.0, 1.0e-13, 0.0]]),
            "nonzero and linearly independent",
        ),
        (
            np.zeros((1, 3)),
            np.array(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [4.0, 4.0, 0.0],
                ]
            ),
            "nonzero and linearly independent",
        ),
    ],
    ids=[
        "empty-coordinates",
        "coordinate-shape",
        "nonfinite-coordinates",
        "empty-translations",
        "translation-shape",
        "too-many-translations",
        "nonfinite-translations",
        "zero-translation",
        "dependent-2d",
        "near-dependent-2d",
        "dependent-3d",
    ],
)
def test_cpp_bridge_rejects_malformed_inputs_before_topology_build(
    coords,
    translations,
    message,
):
    with pytest.raises(ValueError, match=message):
        _bridge()(coords, translations)


def test_msindo_header_uses_shared_topology_aliases_only():
    source = (
        _ROOT
        / "cpp"
        / "include"
        / "vibeqc"
        / "semiempirical"
        / "methods"
        / "indo"
        / "ccm_engine.hpp"
    ).read_text()

    assert "using WSNeighbor = seccm::WSImage;" in source
    assert "using WignerSeitzCells = seccm::WSTopology;" in source
    assert "using seccm::build_wigner_seitz;" in source
    assert "inline WignerSeitzCells build_wigner_seitz" not in source
    assert "_image_translations" not in source
    assert "for (int ii = -1;" not in source


def _mgo_rocksalt_222_bohr(*, coordinate_scale: float = 1.0):
    """MgO B1 from the primitive fcc cell replicated 2x2x2 (16 atoms), bohr.

    ``coordinate_scale`` dilates the atomic coordinates but NOT the cyclic
    translations, which is exactly what a caller does when it converts
    coordinates with one Angstrom-to-bohr constant and translations with
    another (issue #592: CODATA-2014 vs CODATA-2018 differ by 4.4e-10).
    """
    from vibeqc.molecule import ANGSTROM_TO_BOHR

    a = 4.212
    primitive = np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])
    basis = np.array([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]])
    coords = []
    for i, j, k in product(range(2), repeat=3):
        offset = i * primitive[0] + j * primitive[1] + k * primitive[2]
        for site in basis:
            coords.append(site + offset)
    coords = np.array(coords) * ANGSTROM_TO_BOHR * coordinate_scale
    translations = 2.0 * primitive * ANGSTROM_TO_BOHR
    return coords, translations


def _record_multiset(records):
    return sorted(
        (
            int(record["central"]),
            int(record["origin"]),
            tuple(record["shell_label"]),
            round(float(record["weight"]), 12),
        )
        for record in records
    )


def test_equidistant_image_ties_survive_finite_precision_coordinates():
    """Issue #592: a 4.4e-10 relative coordinate/translation mismatch must not
    break the 4-, 2- and 6-fold equidistant-image ties of the MgO 2x2x2 cyclic
    cluster.  MSINDO ``neighbors.f`` shares images whose distance to the
    central atom lies within 1e-5 bohr of the minimum; the exact 256-eps search
    that replaced it turned 512 shared records into 320 unshared ones while the
    weight-sum validity rule still passed, and the MSINDO SECCM energy moved by
    6e-2 Ha and became labeling dependent (issues #592, #593)."""
    bridge = _bridge()
    exact_coords, translations = _mgo_rocksalt_222_bohr()
    perturbed_coords, _ = _mgo_rocksalt_222_bohr(coordinate_scale=1.0 - 4.403e-10)
    assert np.max(np.abs(perturbed_coords - exact_coords)) > 1.0e-9

    exact_cpp = bridge(exact_coords, translations)
    perturbed_cpp = bridge(perturbed_coords, translations)
    assert exact_cpp["valid"] and perturbed_cpp["valid"]
    exact_records = _record_multiset(exact_cpp["records"])
    assert len(exact_records) == 512
    assert {weight for _, _, _, weight in exact_records} == {
        1.0,
        0.5,
        0.25,
        round(1.0 / 6.0, 12),
    }
    assert _record_multiset(perturbed_cpp["records"]) == exact_records

    exact_py = build_seccm_topology(exact_coords, translations)
    perturbed_py = build_seccm_topology(perturbed_coords, translations)
    for python_topology, cpp_records in (
        (exact_py, exact_records),
        (perturbed_py, exact_records),
    ):
        python_records = sorted(
            (
                central,
                int(image.origin),
                tuple(image.image_shell_label),
                round(float(image.weight), 12),
            )
            for central, cell in enumerate(python_topology.cells)
            for image in cell
        )
        assert python_records == cpp_records
