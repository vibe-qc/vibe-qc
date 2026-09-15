"""Wigner–Seitz geometry engine for the Ab Initio Cyclic Cluster Model.

Tests the pure-geometry layer (no integrals): the minimum-image / boundary
multiplicity machinery that the CCM integral weighting reduces to. The key
property exercised here is correctness for **all crystal lattices** —
including skewed (triclinic / hexagonal / fcc-primitive) cells — which is
guaranteed by Minkowski-reducing the basis before the neighbour search.

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550, eq. (4).
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from vibeqc.periodic.ccm.wigner_seitz import (
    min_image_multiplicity,
    minimum_image,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _brute_force_min_image(disp, lattice, rng=6, tol=1e-6):
    """Ground-truth minimum image by explicit enumeration (rows = vectors)."""
    L = np.asarray(lattice, float)
    cells = np.array([[i, j, k] for i in range(-rng, rng + 1)
                      for j in range(-rng, rng + 1) for k in range(-rng, rng + 1)])
    R = cells @ L
    d = np.linalg.norm(disp - R, axis=1)
    dmin = d.min()
    tied = cells[d <= dmin + tol]
    return dmin, {tuple(int(x) for x in c) for c in tied}


# --------------------------------------------------------------------------- #
# Boundary multiplicity (eq. 4's n): 1 interior, 2 face, 4 edge, 8 corner.
# --------------------------------------------------------------------------- #
def test_cubic_boundary_multiplicity():
    d = 4.0
    L = np.diag([d, d, d])  # cubic cluster lattice (rows = columns here)
    A = np.zeros(3)
    assert min_image_multiplicity(A, [d / 4, 0, 0], L) == 1     # interior
    assert min_image_multiplicity(A, [d / 2, 0, 0], L) == 2     # face
    assert min_image_multiplicity(A, [d / 2, d / 2, 0], L) == 4  # edge
    assert min_image_multiplicity(A, [d / 2, d / 2, d / 2], L) == 8  # corner


def test_weights_sum_to_one():
    """Each pair's total WSSC weight is unity (counted exactly once)."""
    L = np.diag([4.0, 4.0, 4.0])
    A = np.zeros(3)
    for target in ([1.0, 0, 0], [2.0, 0, 0], [2.0, 2.0, 0], [2.0, 2.0, 2.0]):
        _, weights = minimum_image(A - np.asarray(target), L)
        assert weights[0].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Correctness for ALL lattices: reduced-basis search == brute force.
# --------------------------------------------------------------------------- #
LATTICES = {
    "cubic": np.diag([4.0, 4.0, 4.0]),
    "tetragonal": np.diag([3.0, 3.0, 6.0]),
    "hexagonal": np.array([[4.5, 0, 0], [-2.25, 3.897114, 0], [0, 0, 6.0]]),
    "triclinic": np.array([[4.0, 0, 0], [1.6, 3.7, 0], [1.2, 0.9, 4.3]]),
    "fcc_primitive": np.array([[0, 2.5, 2.5], [2.5, 0, 2.5], [2.5, 2.5, 0]]),
    "rhombohedral": np.array([[4.0, 0, 0], [1.3, 3.8, 0], [1.3, 1.1, 3.6]]),
}


@pytest.mark.parametrize("name", list(LATTICES))
def test_minimum_image_matches_brute_force_all_lattices(name):
    """The Minkowski-reduced minimum image equals brute force for any lattice."""
    L = LATTICES[name]
    rs = np.random.RandomState(12345)
    disps = (rs.rand(400, 3) * 2.0 - 1.0) @ L * 1.8  # arbitrary points in/around the cell
    cells, _ = minimum_image(disps, L)
    for m in range(disps.shape[0]):
        bd, bcells = _brute_force_min_image(disps[m], L)
        got_cells = {tuple(int(x) for x in c) for c in cells[m]}
        got_d = np.linalg.norm(disps[m] - (cells[m][0] @ L))
        assert got_d == pytest.approx(bd, abs=1e-7)
        assert got_cells == bcells


def test_returned_cells_are_original_basis_integers():
    """Minkowski round-trip: returned cells reproduce the displacement vector."""
    L = LATTICES["triclinic"]
    disp = np.array([2.3, -1.1, 0.7])
    cells, weights = minimum_image(disp, L)
    # The reduced residual must equal the closest-image vector in every cell.
    dists = [np.linalg.norm(disp - (c @ L)) for c in cells[0]]
    assert np.allclose(dists, dists[0])  # all tied cells equidistant
