"""Tests for the periodic MSINDO Cyclic Cluster Model (CCM).

The Wigner-Seitz cell construction is the foundation of the CCM: it defines each
cluster atom's periodic environment (which translational images of the other
atoms fall inside its WS cell, and with what partial-count weight).  The
non-circular ground truth is MSINDO itself — ``ccm_reference.json`` stores the
oracle's per-atom WS neighbour table (``PRINTOPTS=CCMWSC``) for a set of 1-D
cyclic clusters, generated out-of-process (CLAUDE.md §10) by
``examples/regression/msindo/gen_ccm_reference.py``.

This module pins the Python WS construction (``msindo_ccm.build_wigner_seitz``,
a port of ``neighbors.f``) to that oracle data.  The periodic Fock / Madelung
energy parity is the next increment (docs/user_guide/msindo.md, CCM).
"""

import json
import tomllib
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.molecule import ANGSTROM_TO_BOHR
from vibeqc.semiempirical.methods import msindo, msindo_ccm as ccm
from vibeqc.semiempirical.methods.msindo_ccm import (
    CCMOptions,
    WSNeighbor,
    WignerSeitzCells,
    _rund,
    build_wigner_seitz,
    ccm_gradient_fd,
    ccm_optimize,
    run_ccm,
)

_SYM2Z = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "F": 9, "Mg": 12,
          "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17}

_REF_PATH = (
    Path(__file__).parent.parent
    / "examples" / "regression" / "msindo" / "ccm_reference.json"
)
_REF = json.loads(_REF_PATH.read_text())
_CLUSTERS = {c["name"]: c for c in _REF["clusters"]}
_GRAD_CLUSTERS = {c["name"]: c for c in _REF.get("gradient_clusters", [])}


@pytest.fixture(scope="module")
def native_ccm_api():
    indo = pytest.importorskip("vibeqc._vibeqc_core.semiempirical.indo")
    param_path = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "vibeqc"
        / "semiempirical"
        / "methods"
        / "msindo_params.json"
    )
    return indo, indo.load_params_from_json(param_path.read_text())


def _coords(cluster):
    return [[x, y, z] for _sym, x, y, z in cluster["real_atoms"]]


@pytest.mark.parametrize("n,d", [(3, 1.28), (7, 1.28), (9, 1.3), (9, 1.375)])
@pytest.mark.parametrize("madelung", [False, True])
def test_carbon_chain_stable_under_coordinate_noise(n, d, madelung):
    """#249: a gapped final spectrum can still be an unstable SCF saddle."""
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n) * d
    translations = [[n * d, 0, 0]]
    values = [
        run_ccm([6] * n, coords * scale, translations, madelung=madelung,
                max_iter=400, conv_tol=1e-9)
        for scale in (1.0, 1.0 + 4.4e-10)
    ]
    for value in values:
        assert value.converged and value.n_iter <= 400
        assert value.stability_checked and value.stability_analysis_converged
        assert value.stability_eigenvalue >= -4e-6
    assert values[0].total_energy / n == pytest.approx(
        values[1].total_energy / n, abs=1e-8
    )


def test_carbon_chain_python_reference_stability(monkeypatch):
    monkeypatch.setattr(ccm, "_cpp_ccm_energy_kernel", lambda: None)
    n, d = 3, 1.28
    coords = [[i * d * (1 + 4.4e-10), 0, 0] for i in range(n)]
    result = run_ccm([6] * n, coords, [[n * d, 0, 0]], max_iter=400)
    assert result.converged and result.stability_checked
    assert result.stability_eigenvalue >= -4e-6
    assert result.total_energy / n == pytest.approx(-5.8507989864278, abs=1e-8)


def test_carbon_chain_run_job_reports_stability(tmp_path):
    molecule = vq.Molecule([
        vq.Atom(6, [i * 1.28 * ANGSTROM_TO_BOHR, 0, 0]) for i in range(3)
    ])
    result = vq.run_job(
        molecule, method="seccm", output=str(tmp_path / "stable_chain"),
        ccm_options=CCMOptions(translations=[[3.84, 0, 0]], madelung=False),
        citations=True, verbose=0,
    )
    assert result.stability_checked and result.stability_analysis_converged
    assert result.energy / 3 == pytest.approx(-5.8507989864278, abs=1e-8)
    assert "internal stability: stable" in (tmp_path / "stable_chain.out").read_text()
    assert "10.3390/molecules25051218" in (tmp_path / "stable_chain.bibtex").read_text()


def _oracle_origins(ws_cell):
    """Multiset of (1-based) neighbour origin atom-ids for one oracle WS cell —
    the lossless part of the dump (integers); validates the WS *topology*."""
    return Counter(ws_cell["neighbors"])


def _python_origins(cell):
    return Counter(n.origin + 1 for n in cell)


def _oracle_weight_multiset(ws_cell):
    """(origin id, weight) multiset; only lossless when the F6.2 dump weights are
    exact (1-D clusters: weights ∈ {0.5, 1.0})."""
    return Counter((nb, round(w, 4))
                   for nb, w in zip(ws_cell["neighbors"], ws_cell["weights"]))


def _python_weight_multiset(cell):
    return Counter((n.origin + 1, round(n.weight, 4)) for n in cell)


# --------------------------------------------------------------------------- #
# _rund — faithful port of MSINDO rund.f
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,digit,expected", [
    (0.5, 4, 0.5),
    (0.50003, 4, 0.5),       # rounds down to the boundary -> WS member
    (0.5005, 4, 0.5005),     # rounds up past 0.5 -> excluded
    (0.25, 4, 0.25),
    (0.8, 4, 0.8),
    (-0.4, 4, -0.4),
    (12.5, 1, 12.5),
    (0.0, 1, 0.0),
])
def test_rund_matches_fortran(value, digit, expected):
    assert _rund(value, digit) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# WS construction vs oracle
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", list(_CLUSTERS))
def test_ws_neighbour_origins_match_oracle(name):
    """Every atom's WS neighbour *topology* (which atoms, how many images each)
    matches the oracle — exact integer comparison, all dimensions."""
    cluster = _CLUSTERS[name]
    ws = build_wigner_seitz(_coords(cluster), cluster["translations"])
    assert len(ws.cells) == len(cluster["ws_cells"])
    for i, ocell in enumerate(cluster["ws_cells"]):
        assert ocell["atom"] == i + 1  # table is in atom order
        assert _python_origins(ws.cells[i]) == _oracle_origins(ocell), (
            f"{name} atom {i + 1}: WS neighbour origins differ from oracle"
        )


@pytest.mark.parametrize("name", [n for n, c in _CLUSTERS.items() if c["dim"] == 1])
def test_ws_weights_match_oracle_1d(name):
    """1-D WS weights match the oracle exactly (F6.2 dump is lossless for the
    weights 0.5/1.0 that occur in 1-D).  3-D weights (1/8, 1/4, …) round in the
    F6.2 dump, so they are validated to high precision through energy parity
    instead (``test_ccm_total_energy_matches_oracle``)."""
    cluster = _CLUSTERS[name]
    ws = build_wigner_seitz(_coords(cluster), cluster["translations"])
    for i, ocell in enumerate(cluster["ws_cells"]):
        assert _python_weight_multiset(ws.cells[i]) == _oracle_weight_multiset(ocell), (
            f"{name} atom {i + 1}: WS weights differ from oracle"
        )


@pytest.mark.parametrize("name", list(_CLUSTERS))
def test_ws_validity_rule(name):
    """neighbors.f:413 — round(total WS weight) == NATOMS-1 for every atom."""
    cluster = _CLUSTERS[name]
    natoms = len(cluster["real_atoms"])
    ws = build_wigner_seitz(_coords(cluster), cluster["translations"])
    assert ws.is_valid(natoms)
    for i in range(natoms):
        assert round(ws.total_weight(i)) == natoms - 1


def test_odd_chain_has_no_boundary_atoms():
    """An odd-N evenly-spaced chain places no atom on a WS face -> all weights 1."""
    cluster = _CLUSTERS["he5_a2.5_noewald"]
    ws = build_wigner_seitz(_coords(cluster), cluster["translations"])
    for cell in ws.cells:
        assert all(n.weight == 1.0 for n in cell)
        assert len(cell) == len(cluster["real_atoms"]) - 1


def test_ionic_chain_has_boundary_split():
    """The alternating H-F chain shares same-species neighbours across WS faces
    with weight 0.5 (the 1-D Madelung environment)."""
    cluster = _CLUSTERS["hfionic_a1.6_n2_noewald"]
    ws = build_wigner_seitz(_coords(cluster), cluster["translations"])
    half_weights = [n.weight for cell in ws.cells for n in cell if n.weight != 1.0]
    assert half_weights and all(w == 0.5 for w in half_weights)


def test_3d_cubic_cell_has_corner_edge_face_weights():
    """A 2×2×2 simple-cubic cluster shares neighbours across WS faces (1/2),
    edges (1/4) and corners (1/8); the per-atom weights sum to NATOMS-1."""
    cluster = _CLUSTERS["he_sc_2x2x2_a2.5_noewald"]
    natoms = len(cluster["real_atoms"])
    ws = build_wigner_seitz(_coords(cluster), cluster["translations"])
    weights = {round(n.weight, 3) for cell in ws.cells for n in cell}
    assert weights == {0.125, 0.25, 0.5}
    for i in range(natoms):
        assert ws.total_weight(i) == pytest.approx(natoms - 1, abs=1e-9)


def test_returns_wigner_seitz_cells_type():
    cluster = _CLUSTERS["hf_dilute_chain"]
    ws = build_wigner_seitz(_coords(cluster), cluster["translations"])
    assert isinstance(ws, WignerSeitzCells)
    # Dilute molecular-limit chain: each atom sees only its covalent partner.
    for cell in ws.cells:
        assert len(cell) == 1 and cell[0].weight == 1.0


# Issue #187.  A unimodular shear a2 <- a2 + n*a1 relabels the *same* surface
# lattice, so every quantity the 2-D Parry kernel computes must be unchanged.
# The pre-fix fixed +/-12 coefficient box was not cutoff-complete: on the H2O
# mesh below it moved madkonst[0,0] from the converged -0.390026492000 to
# -0.397660474499 (shear 13) and -0.398030563180 (shear 25), errors of 7.634
# and 8.004 mHa, and the required radius grew with the shear.
_MADKONST_2D_SHEARS = (0, 13, 25)

# Converged value: a radius sweep of the old fixed-box kernel on this cell
# reaches it from either skew representation (err 3e-16 by radius 60).
_MADKONST_2D_H2O_REFERENCE = -0.390026492000196


def _sheared_surface_basis(shear):
    """Unimodular a2 <- a2 + shear*a1 for the 10x10 bohr square surface mesh."""
    return [
        np.array([10.0, 0.0, 0.0]),
        np.array([10.0 * shear, 10.0, 0.0]),
    ]


@pytest.mark.parametrize("shear", _MADKONST_2D_SHEARS)
def test_madkonst_2d_is_invariant_under_unimodular_surface_shear(shear):
    """The 2-D Parry Madelung matrix is a sum over physical lattice vectors."""
    from vibeqc.semiempirical.seccm.topology import (
        bind_finite_group,
        build_seccm_topology,
    )

    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.8089, 0.0, 0.0], [-0.4576, 1.4316, 0.0]]
    )
    translations = _sheared_surface_basis(shear)
    topology = bind_finite_group(
        build_seccm_topology(
            coordinates,
            translations,
            length_unit="bohr",
            geometry_quantum=1.0e-10,
        ),
        primitive_vectors=translations,
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    madkonst = ccm._madkonst_2d(
        ccm._ewald_ws_cells(topology), topology.translations, len(coordinates)
    )

    assert madkonst[0, 0] == pytest.approx(
        _MADKONST_2D_H2O_REFERENCE, abs=1.0e-10
    )


@pytest.mark.parametrize("force_python", [False, True], ids=["native", "python"])
@pytest.mark.parametrize("shear", _MADKONST_2D_SHEARS)
def test_ccm2d_madelung_is_strong_skew_basis_invariant(
    monkeypatch, force_python, shear
):
    """A determinant-one surface shear cannot change the 2-D CCM energy."""
    if force_python:
        monkeypatch.setattr(ccm, "_cpp_ccm_energy_kernel", lambda: None)
    else:
        assert ccm._cpp_ccm_energy_kernel() is not None

    atomic_numbers = [3, 1]
    coordinates = [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]
    result = run_ccm(
        atomic_numbers,
        coordinates,
        _sheared_surface_basis(shear),
        madelung=True,
        conv_tol=1.0e-11,
    )

    assert result.converged
    assert result.total_energy == pytest.approx(
        -0.799734613226187, abs=1.0e-10
    )


@pytest.mark.parametrize("force_python", [False, True], ids=["native", "python"])
@pytest.mark.parametrize("shear", _MADKONST_2D_SHEARS)
def test_ccm2d_analytic_gradient_is_strong_skew_basis_invariant(
    monkeypatch, force_python, shear
):
    """The 2-D gradient shares the energy's inventory, so it is invariant too.

    Pre-fix this was the sharpest contracted probe: the neutral-cell Madelung
    *energy* is insensitive to the near-uniform part of the truncation error,
    but the gradient is not -- it moved by 3.5e-05 Ha/bohr at shear 13.
    """
    from vibeqc.semiempirical.methods import (
        msindo_ccm_gradient_analytic as gradient_module,
    )

    if force_python:
        monkeypatch.setattr(
            gradient_module, "_cpp_ccm_gradient_kernel", lambda: None
        )
        monkeypatch.setattr(ccm, "_cpp_ccm_energy_kernel", lambda: None)

    gradient = np.asarray(
        gradient_module.ccm_gradient_analytic(
            [3, 1],
            [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]],
            _sheared_surface_basis(shear),
            madelung=True,
        )
    )

    expected = np.array(
        [[-0.009557953129, 0.0, 0.0], [0.009557953129, 0.0, 0.0]]
    )
    np.testing.assert_allclose(gradient, expected, rtol=0.0, atol=1.0e-9)


def test_python_ccm2d_ewald_rejects_pathological_candidate_box():
    """Pathological surface bases fail closed before allocating a huge grid."""
    translations = np.array([[4.0, 0.0, 0.0], [4.0e6, 4.0, 0.0]])
    ews = [[WSNeighbor(origin=0, weight=1.0, disp=np.zeros(3))]]

    with pytest.raises(ValueError, match="safe candidate budget"):
        ccm._ewald_lattice_2d(ews, translations)


def test_native_ccm2d_madelung_fails_closed_on_pathological_surface_basis(
    native_ccm_api,
):
    """A pathological surface basis never returns a basis-dependent number.

    The compiled 2-D Ewald carries the same candidate-budget guard as the
    Python side, but a basis this ill-conditioned is rejected earlier by the
    SECCM closest-image enumeration.  Either way the contract that matters is
    fail-closed: the kernel must raise, not silently answer.
    """
    indo, params = native_ccm_api

    with pytest.raises(
        ValueError, match="safe candidate budget|too ill-conditioned"
    ):
        indo.run_ccm(
            [3, 1],
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0e6, 1.0, 0.0]],
            params,
            madelung=True,
            max_iter=2,
            conv_tol=1.0e-9,
            charge=0,
        )


@pytest.mark.parametrize("force_python", [False, True], ids=["native", "python"])
def test_ccm3d_madelung_is_strong_skew_basis_invariant(
    monkeypatch, force_python
):
    """A determinant-one lattice shear cannot change the 3-D Ewald result."""
    if force_python:
        monkeypatch.setattr(ccm, "_cpp_ccm_energy_kernel", lambda: None)
    else:
        assert ccm._cpp_ccm_energy_kernel() is not None

    atomic_numbers = [3, 1]
    coordinates = [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]
    cubic = np.eye(3) * 4.0
    skewed = cubic.copy()
    skewed[1] += 13.0 * skewed[0]

    reference = ccm.run_ccm(
        atomic_numbers,
        coordinates,
        cubic,
        madelung=True,
        conv_tol=1.0e-11,
    )
    transformed = ccm.run_ccm(
        atomic_numbers,
        coordinates,
        skewed,
        madelung=True,
        conv_tol=1.0e-11,
    )

    assert reference.converged and transformed.converged
    assert reference.total_energy == pytest.approx(
        -0.8059676829914101, abs=1.0e-11
    )
    assert transformed.total_energy == pytest.approx(
        reference.total_energy, abs=1.0e-10
    )
    np.testing.assert_allclose(
        transformed.density, reference.density, rtol=0.0, atol=1.0e-10
    )


def test_python_ccm3d_ewald_rejects_pathological_candidate_box():
    """Pathological bases fail closed before allocating an unbounded grid."""
    translations = np.array(
        [[4.0, 0.0, 0.0], [4.0e6, 4.0, 0.0], [0.0, 0.0, 4.0]]
    )
    ews = [[WSNeighbor(origin=0, weight=1.0, disp=np.zeros(3))]]

    with pytest.raises(ValueError, match="safe candidate budget"):
        ccm._ewald_lattice_3d(ews, translations)


def test_native_ccm3d_ewald_rejects_pathological_candidate_box(
    native_ccm_api,
):
    """The compiled kernel applies the same fail-closed allocation guard."""
    indo, params = native_ccm_api

    with pytest.raises(ValueError, match="safe candidate budget"):
        indo.run_ccm(
            [3, 1],
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
            [[1.0e6, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            params,
            madelung=True,
            max_iter=2,
            conv_tol=1.0e-9,
            charge=0,
        )


def test_is_valid_rejects_wrong_total_weight():
    """The validity checker flags a cell whose weights don't sum to NATOMS-1."""
    # 3 atoms but atom 0 only "sees" one neighbour with weight 1 -> invalid.
    bad = WignerSeitzCells(cells=[
        [WSNeighbor(1, 1.0, None)],
        [WSNeighbor(0, 1.0, None), WSNeighbor(2, 1.0, None)],
        [WSNeighbor(0, 1.0, None), WSNeighbor(1, 1.0, None)],
    ])
    assert not bad.is_valid(3)


# --------------------------------------------------------------------------- #
# Periodic INDO energy vs oracle (NOEWALD + 1-D Madelung)
# --------------------------------------------------------------------------- #

_ALL = list(_CLUSTERS)


def _run(cluster):
    Z = [_SYM2Z[s] for s, *_ in cluster["real_atoms"]]
    coords = [[x, y, z] for _s, x, y, z in cluster["real_atoms"]]
    return run_ccm(Z, coords, cluster["translations"], madelung=cluster["ewald"])


def test_direct_ccm_api_rejects_malformed_translations():
    z = [1, 1]
    coords = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
    four_vectors = [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 5.0, 0.0],
    ]

    with pytest.raises(ValueError, match="1, 2, or 3 vectors"):
        run_ccm(z, coords, four_vectors)

    with pytest.raises(ValueError, match="exactly three numeric components"):
        ccm_gradient_fd(z, coords, [[5.0, 0.0]])

    with pytest.raises(ValueError, match="exactly three numeric components"):
        ccm_optimize(z, coords, [["wide", 0.0, 0.0]], max_steps=1)


@pytest.mark.parametrize(
    "entrypoint", ["run_ccm", "ccm_gradient_fd", "ccm_gradient_analytic"]
)
def test_native_ccm_entrypoints_reject_mismatched_atom_arrays(
    native_ccm_api, entrypoint
):
    indo, params = native_ccm_api
    with pytest.raises(ValueError, match="same nonzero length"):
        getattr(indo, entrypoint)(
            [1, 1],
            [[0.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0]],
            params,
        )


@pytest.mark.parametrize(
    "translations, message",
    [
        ([], "one to three"),
        (
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
                [5.0, 5.0, 0.0],
            ],
            "one to three",
        ),
        ([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], "linearly independent"),
        ([[float("nan"), 0.0, 0.0]], "finite"),
    ],
)
@pytest.mark.parametrize(
    "entrypoint", ["run_ccm", "ccm_gradient_fd", "ccm_gradient_analytic"]
)
def test_native_ccm_entrypoints_reject_invalid_lattices(
    native_ccm_api, entrypoint, translations, message
):
    indo, params = native_ccm_api
    with pytest.raises(ValueError, match=message):
        getattr(indo, entrypoint)(
            [1, 1],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            translations,
            params,
        )


@pytest.mark.parametrize(
    "coords, message",
    [
        (
            [[float("inf"), 0.0, 0.0], [1.0, 0.0, 0.0]],
            "coordinates must be finite",
        ),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "coincident atoms",
        ),
    ],
)
@pytest.mark.parametrize(
    "entrypoint", ["run_ccm", "ccm_gradient_fd", "ccm_gradient_analytic"]
)
def test_native_ccm_entrypoints_reject_invalid_coordinates(
    native_ccm_api, entrypoint, coords, message
):
    indo, params = native_ccm_api
    with pytest.raises(ValueError, match=message):
        getattr(indo, entrypoint)(
            [1, 1], coords, [[5.0, 0.0, 0.0]], params
        )


@pytest.mark.parametrize(
    "controls, message",
    [
        ({"max_iter": 0}, "max_iter"),
        ({"conv_tol": 0.0}, "conv_tol"),
        ({"conv_tol": float("nan")}, "conv_tol"),
    ],
)
@pytest.mark.parametrize(
    "entrypoint", ["run_ccm", "ccm_gradient_fd", "ccm_gradient_analytic"]
)
def test_native_ccm_entrypoints_reject_invalid_scf_controls(
    native_ccm_api, entrypoint, controls, message
):
    indo, params = native_ccm_api
    with pytest.raises(ValueError, match=message):
        getattr(indo, entrypoint)(
            [1, 1],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0]],
            params,
            **controls,
        )


@pytest.mark.parametrize(
    "step", [0.0, -1.0e-3, float("inf"), float("nan")]
)
def test_native_ccm_fd_gradient_rejects_invalid_step(native_ccm_api, step):
    indo, params = native_ccm_api
    with pytest.raises(ValueError, match="step must be positive and finite"):
        indo.ccm_gradient_fd(
            [1, 1],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0]],
            params,
            step=step,
        )


def test_ccm_native_nonconvergence_does_not_fall_back_to_python(
    monkeypatch,
):
    class NonconvergedResult:
        converged = False

    def nonconverged_native(*args, **kwargs):
        return NonconvergedResult()

    monkeypatch.setattr(
        "vibeqc.semiempirical.methods.msindo_ccm._cpp_ccm_energy_kernel",
        lambda: (nonconverged_native, object()),
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        run_ccm(
            [1, 1],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0]],
        )


def test_direct_ccm_gradient_rejects_unsupported_elements():
    with pytest.raises(NotImplementedError, match="MSINDO engine supports"):
        ccm_gradient_fd(
            [55, 9],
            [[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]],
            [[5.0, 0.0, 0.0]],
        )


@pytest.mark.parametrize("step", [0.0, -1.0e-3, float("inf")])
def test_direct_ccm_gradient_rejects_invalid_fd_step_before_kernel(
    monkeypatch, step
):
    def forbidden_kernel():
        raise AssertionError("invalid FD step reached the native kernel")

    monkeypatch.setattr(
        "vibeqc.semiempirical.methods.msindo_ccm._cpp_ccm_gradient_fd_kernel",
        forbidden_kernel,
    )
    with pytest.raises(ValueError, match="positive and finite"):
        ccm_gradient_fd(
            [1, 1],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0]],
            step=step,
        )


@pytest.mark.parametrize("name", _ALL)
def test_ccm_total_energy_matches_oracle(name):
    """Periodic INDO total energy reproduces MSINDO to ~5e-9 Ha (NOEWALD and
    1-D Madelung clusters alike)."""
    cluster = _CLUSTERS[name]
    res = _run(cluster)
    assert res.converged
    assert res.total_energy == pytest.approx(
        cluster["reference"]["total_energy"], abs=5e-9)


@pytest.mark.parametrize("name", _ALL)
def test_ccm_energy_components_match_oracle(name):
    """Electronic, Madelung-nuclear (WS-weighted core-core) and binding all
    reproduce the oracle after using its derived geometry convention."""
    cluster = _CLUSTERS[name]
    ref = cluster["reference"]
    res = _run(cluster)
    assert res.binding_energy == pytest.approx(ref["binding_energy"], abs=5e-9)
    assert res.electronic_energy == pytest.approx(ref["electronic_energy"], abs=5e-9)
    assert res.madelung_nuclear_energy == pytest.approx(
        ref["madelung_nuclear_energy"], abs=5e-9)


@pytest.mark.parametrize("name", list(_GRAD_CLUSTERS))
def test_ccm_gradient_matches_oracle(name):
    """The fixed-WS finite-difference CCM nuclear gradient reproduces MSINDO's
    analytic gradient (CARTOPT ANALY GRADONLY) to finite-difference accuracy, in
    both the NOEWALD (1-D chain) and Ewald-Madelung (distorted MgO bulk) cases."""
    cluster = _GRAD_CLUSTERS[name]
    Z = [_SYM2Z[s] for s, *_ in cluster["real_atoms"]]
    coords = [[x, y, z] for _s, x, y, z in cluster["real_atoms"]]
    ref = cluster["gradient_ha_per_bohr"]
    g = ccm_gradient_fd(Z, coords, cluster["translations"],
                        madelung=cluster["ewald"])
    assert g.shape == (len(ref), 3)
    for i, gref in enumerate(ref):
        for d in range(3):
            assert g[i][d] == pytest.approx(gref[d], abs=2e-5), (
                f"{name} atom {i + 1} comp {d}: FD {g[i][d]} vs oracle {gref[d]}"
            )


def test_ccm_gradient_fixed_ws_beats_naive_at_boundary():
    """The fixed-WS gradient is essential where an atom sits on a WS face: in the
    H-F chain each F has a same-species image exactly at ±T/2, so a naive
    WS-rebuilding finite difference would straddle a membership flip and give a
    grossly wrong F force.  The fixed-WS gradient stays correct (~µHa/bohr)."""
    cluster = _GRAD_CLUSTERS["hfionic_chain_distorted"]
    Z = [_SYM2Z[s] for s, *_ in cluster["real_atoms"]]
    coords = [[x, y, z] for _s, x, y, z in cluster["real_atoms"]]
    g = ccm_gradient_fd(Z, coords, cluster["translations"], madelung=False)
    ref = cluster["gradient_ha_per_bohr"]
    # F atoms are indices 1 and 3; they would be off by ~1e-2 with a naive FD.
    for i in (1, 3):
        assert abs(g[i][0] - ref[i][0]) < 2e-5


def test_madelung_lowers_energy_vs_noewald():
    """On the same ionic chain, switching on the 1-D Madelung embedding shifts
    the energy (it is not a no-op) and reproduces the distinct oracle target."""
    Z = [1, 9]
    coords = [[0.0, 0, 0], [1.6, 0, 0]]
    trans = [[3.2, 0, 0]]
    noew = run_ccm(Z, coords, trans, madelung=False)
    mad = run_ccm(Z, coords, trans, madelung=True)
    assert abs(mad.total_energy - noew.total_energy) > 1e-3
    assert mad.total_energy == pytest.approx(-23.9259067974, abs=1e-6)


def test_3d_ewald_madelung_rocksalt():
    """3-D Ewald Madelung (madelkonst/madelsum) on the H-F rocksalt cell — the
    MgO structure type — reproduces the oracle and stabilises the cell vs
    NOEWALD by the long-range Madelung energy (~0.3 Ha)."""
    cluster = _CLUSTERS["hf_rocksalt_a4.0_madelung"]
    Z = [_SYM2Z[s] for s, *_ in cluster["real_atoms"]]
    coords = [[x, y, z] for _s, x, y, z in cluster["real_atoms"]]
    mad = run_ccm(Z, coords, cluster["translations"], madelung=True)
    noew = run_ccm(Z, coords, cluster["translations"], madelung=False)
    assert mad.total_energy == pytest.approx(-95.7626742774, abs=1e-6)
    assert mad.total_energy < noew.total_energy - 0.1  # Madelung-stabilised


def test_mgo_rocksalt_bulk():
    """Real **MgO** bulk (rocksalt Mg₄O₄ cell, CCM3D + Ewald Madelung) reproduces
    the oracle — the headline ionic-oxide CCM target (Mg parameters + 3-D WS +
    periodic Fock + Ewald Madelung end-to-end)."""
    cluster = _CLUSTERS["mgo_rocksalt_a4.21_madelung"]
    Z = [_SYM2Z[s] for s, *_ in cluster["real_atoms"]]
    coords = [[x, y, z] for _s, x, y, z in cluster["real_atoms"]]
    mad = run_ccm(Z, coords, cluster["translations"], madelung=True)
    noew = run_ccm(Z, coords, cluster["translations"], madelung=False)
    assert mad.converged
    assert mad.total_energy == pytest.approx(-66.2038374445, abs=1e-6)
    assert mad.total_energy < noew.total_energy - 0.1  # Madelung-stabilised oxide


def test_water_on_mgo_100_adsorption_energy():
    """The headline oxide-adsorption deliverable: a periodic H₂O adlayer on the
    MgO(100) surface (CCM2D + Ewald).  The fixed-geometry adsorption energy
    E[slab+H₂O] − E[slab] − E[H₂O(gas)] reproduces the oracle (each term to
    ~3e-8 → the difference to <1e-6), and water is bound (negative)."""
    def _ccm(name):
        c = _CLUSTERS[name]
        Z = [_SYM2Z[s] for s, *_ in c["real_atoms"]]
        coords = [[x, y, z] for _s, x, y, z in c["real_atoms"]]
        return run_ccm(Z, coords, c["translations"], madelung=True).total_energy

    e_sys = _ccm("mgo_100_water_a2.105_madelung")
    e_slab = _ccm("mgo_100_slab_a2.105_madelung")
    e_h2o = msindo.run_msindo(
        [8, 1, 1],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    ).total_energy
    e_ads = e_sys - e_slab - e_h2o
    # Oracle: -83.4685198921 - (-66.4395927257) - (-17.0182087674)
    e_ads_ref = -83.4685198921 - (-66.4395927257) - (-17.0182087674)
    assert e_ads == pytest.approx(e_ads_ref, abs=1e-6)
    assert e_ads < 0.0  # water binds to the surface


@pytest.mark.slow
def test_water_on_mgo_100_relaxation():
    """Geometry optimisation: relax the H₂O adsorbate on a frozen MgO(100) slab
    (CCM2D + Ewald) with the fixed-WS finite-difference gradient.  The energy must
    drop and the residual force on the adsorbate must converge below the target.
    (The relaxed-geometry energy matches the oracle single point to ~3e-8 — checked
    out-of-process; here we assert the optimiser itself works on our engine.)"""
    a = 2.105
    slab = [("Mg", 0, 0, 0), ("O", a, 0, 0), ("O", 0, a, 0), ("Mg", a, a, 0),
            ("O", 0, 0, a), ("Mg", a, 0, a), ("Mg", 0, a, a), ("O", a, a, a)]
    water = [("O", a, 0.0, a + 2.2), ("H", a + 0.76, 0.0, a + 2.76),
             ("H", a - 0.76, 0.0, a + 2.76)]
    atoms = slab + water
    Z = [_SYM2Z[s] for s, *_ in atoms]
    C = [[x, y, z] for _s, x, y, z in atoms]
    T = [[2 * a, 0, 0], [0, 2 * a, 0]]
    e0 = run_ccm(Z, C, T, madelung=True).total_energy
    Crel, final = ccm_optimize(Z, C, T, madelung=True, frozen=list(range(8)),
                               fmax=2e-3, max_steps=40)
    assert final.total_energy < e0 - 1e-4  # relaxation lowered the energy
    g = ccm_gradient_fd(Z, Crel, T, madelung=True, atoms=[8, 9, 10])
    assert float(np.max(np.abs(g[8:]))) < 3e-3  # forces converged


@pytest.mark.parametrize("name", [
    "mgo_100_water_a2.105_madelung", "mgo_100_nh3_a2.105_madelung"])
def test_polar_adsorbate_binds_mgo_100(name):
    """Polar adsorbates (H₂O, NH₃) bind the MgO(100) surface: the adsorption
    energy E[slab+ads] − E[slab] − E[ads(gas)] is negative.  Every term is
    oracle-validated (see ``test_ccm_total_energy_matches_oracle``); here we check
    the physically meaningful combination across the stated adsorbate set."""
    ads_cluster = _CLUSTERS[name]
    slab_atoms = _CLUSTERS["mgo_100_slab_a2.105_madelung"]["real_atoms"]
    n_slab = len(slab_atoms)
    real = ads_cluster["real_atoms"]
    T = ads_cluster["translations"]
    Z = [_SYM2Z[s] for s, *_ in real]
    C = [[x, y, z] for _s, x, y, z in real]
    e_sys = run_ccm(Z, C, T, madelung=True).total_energy
    e_slab = run_ccm([_SYM2Z[s] for s, *_ in slab_atoms],
                     [[x, y, z] for _s, x, y, z in slab_atoms], T,
                     madelung=True).total_energy
    # gas-phase adsorbate at the adsorbed geometry
    e_gas = msindo.run_msindo([_SYM2Z[s] for s, *_ in real[n_slab:]],
                              [[x, y, z] for _s, x, y, z in real[n_slab:]]).total_energy
    assert e_sys - e_slab - e_gas < 0.0


def test_dilute_chain_reduces_to_isolated_molecule():
    """A dilute HF chain (each WS cell holds only the covalent partner) must give
    exactly the isolated-molecule energy — cross-checked against the molecular
    engine on the same geometry, not just the stored number."""
    cluster = _CLUSTERS["hf_dilute_chain"]
    Z = [_SYM2Z[s] for s, *_ in cluster["real_atoms"]]
    coords = [[x, y, z] for _s, x, y, z in cluster["real_atoms"]]
    ccm = run_ccm(Z, coords, cluster["translations"], madelung=False)
    mol = msindo.run_msindo(Z, coords)
    assert ccm.total_energy == pytest.approx(mol.total_energy, abs=1e-7)


def test_mgo_100_surface_slab():
    """2-D **MgO(100)** bilayer slab (CCM2D + surface Ewald) reproduces the
    oracle — the oxide adsorption substrate.  Exercises the out-of-plane 2-D
    Ewald terms (the slab has two z-layers)."""
    cluster = _CLUSTERS["mgo_100_slab_a2.105_madelung"]
    Z = [_SYM2Z[s] for s, *_ in cluster["real_atoms"]]
    coords = [[x, y, z] for _s, x, y, z in cluster["real_atoms"]]
    res = run_ccm(Z, coords, cluster["translations"], madelung=True)
    assert res.converged
    # Tight-DELEN reference values isolate the geometry conversion from the
    # legacy executable's default 1e-8 energy-change stopping error.
    assert res.total_energy == pytest.approx(-66.4395927266, abs=1e-10)
    assert res.electronic_energy == pytest.approx(-152.2689865300, abs=1e-10)
    assert res.madelung_nuclear_energy == pytest.approx(85.8293938035, abs=1e-10)


def test_open_shell_ccm_rejected():
    """Odd valence-electron cluster (open shell) is not yet supported."""
    # 3 H atoms in a chain: 3 valence electrons -> open shell.
    with pytest.raises(NotImplementedError, match="closed-shell"):
        run_ccm([1, 1, 1], [[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]],
                [[7.5, 0, 0]], madelung=False)


# --------------------------------------------------------------------------- #
# run_job(method="ccm") wiring — v0.12.0 gate (CLAUDE.md §8.4 coverage)         #
#                                                                              #
# These pin the *runner* entry point (not the bare run_ccm engine, which the   #
# tests above cover): that run_job dispatches CCM, reproduces the engine       #
# energy, and lands the periodic-CCM citations (Peintinger & Bredow 2014 +     #
# Bredow, Geudtner & Jug 2001) on every user-facing citation surface AND in    #
# the .system manifest's internal-provenance [citations] section.              #
# --------------------------------------------------------------------------- #

# Smallest fast CCM cell: the 2-atom H-F ionic chain with 1-D Madelung
# embedding (same system as test_madelung_lowers_energy_vs_noewald). F sits
# exactly on the Wigner-Seitz face at T/2, so this cell also pins the
# atom/lattice unit-consistency fix in runner._run_ccm — a wrong Å↔bohr
# round-trip would flip F's WS membership and shift the energy by ~0.03 Ha.
_CCM_E2E_Z = [1, 9]
_CCM_E2E_COORDS_ANG = [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]
_CCM_E2E_TRANS = [[3.2, 0.0, 0.0]]
_CCM_E2E_REF_E = -23.9259067974  # oracle (MSINDO), abs 1e-6

# DOI / key fingerprints the periodic-CCM citation route must surface.
_CCM_CITE_KEYS = ("peintinger_ccm_2014", "bredow_geudtner_jug_ccm_2001")
_CCM_CITE_DOIS = (
    "10.1002/jcc.23550",
    "10.1002/1096-987X(20010115)22:1<89::AID-JCC9>3.0.CO;2-7",
)


def _ccm_e2e_molecule(charge: int = 0) -> vq.Molecule:
    """The H-F chain as a vibe-qc Molecule (bohr), built with vibe-qc's
    canonical Å→bohr constant — exactly the path real file I/O takes, so
    run_job's internal bohr→Å inversion must recover the same geometry the
    bare run_ccm engine call below uses."""
    from vibeqc.molecule import ANGSTROM_TO_BOHR as _A2B

    return vq.Molecule(
        [vq.Atom(z, [c * _A2B for c in xyz])
         for z, xyz in zip(_CCM_E2E_Z, _CCM_E2E_COORDS_ANG)],
        charge, 1,
    )


def test_ccm_run_job_reproduces_engine_energy(tmp_path: Path) -> None:
    """run_job(method="ccm") routes to msindo_ccm.run_ccm and returns the
    same energy as a bare engine call (to numerical noise) AND the MSINDO
    oracle reference — i.e. the runner wiring + Å↔bohr round-trip are exact."""
    bare = run_ccm(_CCM_E2E_Z, _CCM_E2E_COORDS_ANG, _CCM_E2E_TRANS,
                   madelung=True)
    res = vq.run_job(
        _ccm_e2e_molecule(),
        method="ccm",
        output=str(tmp_path / "hf_chain"),
        ccm_options=CCMOptions(translations=_CCM_E2E_TRANS, madelung=True),
        verbose=0,
    )
    assert float(res.energy) == pytest.approx(bare.total_energy, abs=1e-9)
    assert float(res.energy) == pytest.approx(_CCM_E2E_REF_E, abs=1e-6)


def test_ccm_run_job_charged_cell_uses_charged_cpp_path(monkeypatch, tmp_path: Path) -> None:
    """Charged CCM cells pass charge through to the C++ energy binding."""
    indo = pytest.importorskip("vibeqc._vibeqc_core.semiempirical.indo")

    param_path = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "vibeqc"
        / "semiempirical"
        / "methods"
        / "msindo_params.json"
    )
    params = indo.load_params_from_json(param_path.read_text())
    expected = indo.run_ccm(
        _CCM_E2E_Z,
        _CCM_E2E_COORDS_ANG,
        _CCM_E2E_TRANS,
        params,
        madelung=False,
        charge=2,
    )
    seen = {}
    original_run_ccm = indo.run_ccm

    def _recording_run_ccm(*args, **kwargs):
        seen["charge"] = kwargs.get("charge")
        return original_run_ccm(*args, **kwargs)

    monkeypatch.setattr(
        "vibeqc.semiempirical.methods.msindo_ccm._cpp_ccm_energy_kernel",
        lambda: (_recording_run_ccm, params),
    )
    res = vq.run_job(
        _ccm_e2e_molecule(charge=2),
        method="ccm",
        output=str(tmp_path / "hf_chain_charged"),
        ccm_options=CCMOptions(translations=_CCM_E2E_TRANS, madelung=False),
        verbose=0,
    )
    assert seen["charge"] == 2
    assert float(res.energy) == pytest.approx(expected.total_energy, abs=1e-9)


def test_ccm_run_job_charged_madelung_cell_fails_electroneutrality_gate(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="electroneutral cluster"):
        vq.run_job(
            _ccm_e2e_molecule(charge=2),
            method="ccm",
            output=str(tmp_path / "hf_chain_charged_madelung"),
            ccm_options=CCMOptions(
                translations=_CCM_E2E_TRANS, madelung=True
            ),
            verbose=0,
        )


def test_ccm_run_job_propagates_validated_wrapper_failures(
    monkeypatch, tmp_path: Path
) -> None:
    def fail_validated_wrapper(*args, **kwargs):
        raise RuntimeError("validated MSINDO CCM failure")

    monkeypatch.setattr(
        "vibeqc.semiempirical.methods.msindo_ccm.run_ccm",
        fail_validated_wrapper,
    )
    with pytest.raises(RuntimeError, match="validated MSINDO CCM failure"):
        vq.run_job(
            _ccm_e2e_molecule(),
            method="ccm",
            output=str(tmp_path / "hf_chain_wrapper_failure"),
            ccm_options=CCMOptions(
                translations=_CCM_E2E_TRANS, madelung=False
            ),
            verbose=0,
        )


def test_ccm_run_job_lands_citations_on_every_surface(tmp_path: Path) -> None:
    """The periodic-CCM papers reach the user-facing citation surface
    (.bibtex / .references / in-.out references block) AND the .system
    manifest's [citations] provenance section — and the always-on libint
    integral citation is suppressed (CCM uses analytic Slater integrals,
    not libint Gaussians). CLAUDE.md §8.3/§8.4/§8.5."""
    stem = tmp_path / "hf_chain"
    vq.run_job(
        _ccm_e2e_molecule(),
        method="ccm",
        output=str(stem),
        ccm_options=CCMOptions(translations=_CCM_E2E_TRANS, madelung=True),
        verbose=0,
    )

    # (1) user-facing surface: .bibtex / .references / .out all carry both
    # papers' DOIs; the dedicated .bibtex (no banner / linked-libs noise)
    # must NOT carry a libint citation.
    bibtex = stem.with_suffix(".bibtex").read_text()
    surface = "\n".join(
        stem.with_suffix(s).read_text(errors="replace")
        for s in (".bibtex", ".references", ".out")
        if stem.with_suffix(s).is_file()
    )
    for doi in _CCM_CITE_DOIS:
        assert doi in surface, f"CCM DOI {doi!r} missing from citation surface"
    assert "libint" not in bibtex.lower(), (
        "CCM must not cite libint — it uses analytic Slater (STO) integrals"
    )

    # (2) .system manifest [citations] section: full provenance view (every
    # entry regardless of print flag), libint excluded, both CCM papers in.
    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert "citations" in manifest, ".system manifest missing [citations]"
    cite_keys = {e["key"] for e in manifest["citations"]["entries"]}
    for key in _CCM_CITE_KEYS:
        assert key in cite_keys, f"{key!r} missing from .system [citations]"
    assert "parry_2d_ewald_1975" not in cite_keys
    assert "de_leeuw_perram_2d_ewald_1979" not in cite_keys
    assert "libint_valeev" not in cite_keys, (
        "libint citation must be suppressed for CCM in the .system manifest"
    )
    # [libraries] still records libint as *linked* — that is binary
    # provenance, not a scientific citation; the two surfaces are distinct.
    assert "libint" in manifest["libraries"]


@pytest.mark.parametrize("madelung", [False, True])
def test_ccm_run_job_slab_citations_follow_actual_2d_kernel(
    tmp_path: Path,
    madelung: bool,
) -> None:
    """A real CCM2D run cites the slab kernel iff it was selected."""
    stem = tmp_path / f"hf_sheet_madelung_{madelung}"
    vq.run_job(
        _ccm_e2e_molecule(),
        method="ccm",
        output=str(stem),
        ccm_options=CCMOptions(
            translations=[_CCM_E2E_TRANS[0], [0.0, 8.0, 0.0]],
            madelung=madelung,
        ),
        verbose=0,
    )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    cite_keys = {entry["key"] for entry in manifest["citations"]["entries"]}
    slab_keys = {
        "parry_2d_ewald_1975",
        "de_leeuw_perram_2d_ewald_1979",
    }
    if madelung:
        assert slab_keys <= cite_keys
    else:
        assert cite_keys.isdisjoint(slab_keys)

    # The unavailable Heyes primary record is not fabricated on either path.
    assert "10.1039/F29777301485" not in stem.with_suffix(".bibtex").read_text()


def test_ccm_run_job_unavailable_paths_are_clean(tmp_path: Path) -> None:
    """Every CCM boundary is a clean, explicit error — never a silent wrong
    answer (CLAUDE.md §7/§11). Missing lattice → ValueError; open-shell and
    run_job geometry optimisation → NotImplementedError; an element beyond
    the MSINDO engine's scope → NotImplementedError."""
    opts = CCMOptions(translations=_CCM_E2E_TRANS, madelung=True)

    # No ccm_options / no lattice: can't define the cyclic cluster.
    with pytest.raises(ValueError, match="ccm_options"):
        vq.run_job(_ccm_e2e_molecule(), method="ccm",
                   output=str(tmp_path / "a"), verbose=0)

    radius_opts = CCMOptions(cluster_radius=6.0)
    with pytest.raises(NotImplementedError, match="cluster_radius"):
        vq.run_job(_ccm_e2e_molecule(), method="ccm",
                   output=str(tmp_path / "radius"), ccm_options=radius_opts,
                   verbose=0)

    # Open-shell cell (RHF-only engine).
    open_shell = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0])], 0, 2,  # lone H: doublet
    )
    with pytest.raises(NotImplementedError, match="closed-shell"):
        vq.run_job(open_shell, method="ccm", output=str(tmp_path / "b"),
                   ccm_options=opts, verbose=0)

    # Geometry optimisation through run_job is not wired for CCM.
    with pytest.raises(NotImplementedError, match="optimi"):
        vq.run_job(_ccm_e2e_molecule(), method="ccm",
                   output=str(tmp_path / "c"), ccm_options=opts,
                   optimize=True, verbose=0)

    # Element beyond the MSINDO engine scope — clean unavailable.  The engine now
    # covers the full H–Xe table (Z 1–54), so the sentinel is Cs (Z=55, CsF closed
    # shell; no bundled params), matching test_msindo.test_unsupported_element.
    cs_cell = vq.Molecule(
        [vq.Atom(55, [0.0, 0.0, 0.0]), vq.Atom(9, [2.4 * 1.8897, 0.0, 0.0])],
        0, 1,
    )
    with pytest.raises(NotImplementedError, match="MSINDO engine supports"):
        vq.run_job(cs_cell, method="ccm", output=str(tmp_path / "d"),
                   ccm_options=opts, verbose=0)


# Published target: the article SECCM equation-of-state campaign (vq job
# 60a32b0a8630, vibe-qc 0.15.138, compute-managed) recorded E = -137.58937495890854 Ha
# for MgO B1, primitive fcc replicated 2x2x2, a = 4.212 A, madelung=False
# (vibeqc-article-generalsemiempirical, supporting-data/benchmarks/seccm-eos/
# 20260822-msindo-job60a32b0a8630/results/msindo_mgo_n2_bare.json).  That run
# used the exact-tie Wigner-Seitz construction (4-, 2- and 6-fold shared
# images); the native kernel reproduces it to 2.5e-12 on exact coordinates.
_MGO_222_SECCM_BARE_ARTICLE_ENERGY = -137.58937495890854
# CODATA-2014 Angstrom-to-bohr (1/0.52917721067), as hard-coded by the article's
# seccm_build.py; vibeqc.molecule uses CODATA-2018 (1/0.529177210903).  The
# ratio differs from 1 by -4.4e-10, so a Molecule built with one and read back
# with the other carries a 2.8e-9 A coordinate/translation mismatch (issue #592).
_ANGSTROM_TO_BOHR_CODATA_2014 = 1.8897261254578281


def _mgo_222_seccm_energy(anion_shift_angstrom, *, angstrom_to_bohr):
    from vibeqc.molecule import Atom, Molecule
    from vibeqc.semiempirical.runner import run_seccm

    a = 4.212
    primitive = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    basis = [
        (12, [0.0, 0.0, 0.0]),
        (8, [a / 2 + s for s in anion_shift_angstrom]),
    ]
    atoms = []
    for i, j, k in product(range(2), repeat=3):
        offset = [
            i * primitive[0][c] + j * primitive[1][c] + k * primitive[2][c]
            for c in range(3)
        ]
        for z, site in basis:
            atoms.append(
                Atom(z, [(site[c] + offset[c]) * angstrom_to_bohr for c in range(3)])
            )
    translations = [[2.0 * v for v in vector] for vector in primitive]
    molecule = Molecule(atoms, charge=0, multiplicity=1)
    result = run_seccm(
        molecule, CCMOptions(translations=translations, madelung=False)
    )
    assert result.converged
    return float(result.energy)


def test_seccm_mgo_222_invariant_under_anion_sublattice_lattice_translation():
    """Issues #592 / #593: translating the anion sublattice by the primitive
    vector a1 maps the 2x2x2 cyclic cluster onto itself, so the SECCM energy
    must not change, and coordinates carrying a 4.4e-10 conversion mismatch
    must still reproduce the exact-tie (article) energy.  At the parent the
    mismatched input lost 192 shared Wigner-Seitz records: -137.649 vs
    -137.661 Ha between the two labelings, 6e-2 Ha below the article value."""
    a1 = (0.0, 4.212 / 2, 4.212 / 2)
    exact = _mgo_222_seccm_energy(
        (0.0, 0.0, 0.0), angstrom_to_bohr=ANGSTROM_TO_BOHR
    )
    reference = _mgo_222_seccm_energy(
        (0.0, 0.0, 0.0), angstrom_to_bohr=_ANGSTROM_TO_BOHR_CODATA_2014
    )
    shifted = _mgo_222_seccm_energy(
        a1, angstrom_to_bohr=_ANGSTROM_TO_BOHR_CODATA_2014
    )
    assert exact == pytest.approx(_MGO_222_SECCM_BARE_ARTICLE_ENERGY, abs=1.0e-7)
    # The 4.4e-10 dilation of the coordinates alone is worth 2e-9 Ha.
    assert reference == pytest.approx(exact, abs=1.0e-7)
    assert shifted == pytest.approx(reference, abs=1.0e-8)
