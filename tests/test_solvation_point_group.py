"""Point-group operations and gradient symmetrization (#535).

The operations are found from the geometry alone, so these tests need no SCF:
they pin the operation counts of the standard point groups (the finder must
return every operation, including the reflection in a planar solute's own
plane, which shares the identity's atom permutation), the tolerance behaviour,
the fail-safe on a set that is not a group, and the exact properties of the
projection: it is idempotent, keeps a gradient's zero net force and zero net
torque, leaves a symmetric gradient alone and removes an antisymmetric one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc.solvation import point_group as pg

ANG = 1.8897261254578281

WATER = [(8, (0.0, 0.0, 0.0)), (1, (0.0, 0.75655, 0.63172)), (1, (0.0, -0.75655, 0.63172))]
WATER_ASYM = [(8, (0.0, 0.0, 0.0)), (1, (0.0, 0.757, 0.587)), (1, (0.0, -0.9, 0.587))]
_R = 0.94
AMMONIA = [(7, (0.0, 0.0, 0.0))] + [
    (1, (_R * math.cos(2 * math.pi * k / 3), _R * math.sin(2 * math.pi * k / 3), -0.38))
    for k in range(3)
]
_A = 0.629
METHANE = [(6, (0.0, 0.0, 0.0))] + [
    (1, (sx * _A, sy * _A, sz * _A))
    for sx, sy, sz in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
]
BENZENE = [
    (6, (1.3952 * math.cos(math.radians(60 * k + 90)), 1.3952 * math.sin(math.radians(60 * k + 90)), 0.0))
    for k in range(6)
] + [
    (1, (2.4824 * math.cos(math.radians(60 * k + 90)), 2.4824 * math.sin(math.radians(60 * k + 90)), 0.0))
    for k in range(6)
]


def _split(atoms):
    return [z for z, _ in atoms], np.array([p for _, p in atoms], dtype=float) * ANG


def _ops(atoms, tol=1e-4):
    z, x = _split(atoms)
    return pg.symmetry_operations(z, x, tol)


@pytest.mark.parametrize(
    "atoms, order",
    [(WATER, 4), (AMMONIA, 6), (METHANE, 24), (BENZENE, 24), (WATER_ASYM, 2)],
    ids=["water-C2v", "ammonia-C3v", "methane-Td", "benzene-D6h", "bent-water-Cs"],
)
def test_the_finder_returns_every_operation_of_the_point_group(atoms, order):
    ops = _ops(atoms)
    assert len(ops) == order
    z, x = _split(atoms)
    c = x.mean(axis=0)
    # The identity is first, and every operation really maps atoms onto atoms
    # of the same element.
    assert np.allclose(ops[0][0], np.eye(3)) and np.array_equal(ops[0][1], np.arange(len(z)))
    for Q, perm in ops:
        assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-12)
        assert [z[p] for p in perm] == z
        assert np.abs((x - c) @ Q.T - (x - c)[perm]).max() < 1e-9


def test_a_planar_solutes_own_plane_is_an_operation_with_the_identitys_permutation():
    """The reason handedness is part of an operation's identity: the reflection
    in the molecular plane and the identity move no atom."""
    ops = _ops(WATER)
    fixing = [Q for Q, perm in ops if np.array_equal(perm, np.arange(3))]
    assert len(fixing) == 2
    assert sorted(round(float(np.linalg.det(Q))) for Q in fixing) == [-1, 1]


def test_no_symmetry_and_no_group_to_speak_of_return_the_identity_alone():
    low = [(8, (0.0, 0.0, 0.0)), (1, (0.3, 0.757, 0.587)), (1, (-0.2, -0.9, 0.7)), (9, (1.0, 0.1, -0.4))]
    assert len(_ops(low)) == 1
    assert len(_ops([(8, (0.0, 0.0, 0.0))])) == 1                         # one atom
    assert len(_ops([(1, (0.0, 0.0, 0.0)), (17, (0.0, 0.0, 1.27))])) == 1  # linear
    z, x = _split(WATER)
    assert len(pg.symmetry_operations(z, x, 0.0)) == 1                    # disabled


def test_the_tolerance_accepts_a_near_symmetric_geometry_and_refuses_a_distorted_one():
    # A displacement of 3.8e-5 bohr of one hydrogen along the antisymmetric
    # direction: inside 1e-4, outside 1e-6.
    nudged = [WATER[0], WATER[1], (1, (0.0, -0.75655 + 2e-5, 0.63172))]
    assert len(_ops(nudged, 1e-4)) == 4
    # Past 1e-6 the hydrogen swap is lost, but the molecular-plane reflection
    # still holds exactly, and {E, sigma} is a group.
    assert len(_ops(nudged, 1e-6)) == 2
    # A real distortion (0.01 bohr) keeps only the plane.
    assert len(_ops([WATER[0], WATER[1], (1, (0.0, -0.75655 + 0.0053, 0.63172))], 1e-4)) == 2


def test_a_set_of_operations_that_is_not_a_group_is_refused_not_averaged():
    """The average of a set that is not closed is not a projector, so the
    finder returns the identity alone rather than a plausible wrong answer."""
    z, x = _split(WATER)
    ops = pg.symmetry_operations(z, x, 1e-4)
    assert pg._closed(ops)
    # Drop one operation of C2v: the rest is not closed (sigma_v' . C2 = sigma_v).
    broken = [op for k, op in enumerate(ops) if k != 3]
    assert not pg._closed(broken)


def _torque_and_force_free(x, rng):
    """A random gradient with zero net force and zero net torque."""
    c = x.mean(axis=0)
    cols = []
    for k in range(3):
        t = np.zeros_like(x)
        t[:, k] = 1.0
        cols.append(t.ravel())
        cols.append(np.cross(np.eye(3)[k], x - c).ravel())
    M = np.array(cols).T
    g = rng.normal(size=x.size)
    return (g - M @ np.linalg.lstsq(M, g, rcond=None)[0]).reshape(x.shape)


@pytest.mark.parametrize("atoms", [WATER, AMMONIA, METHANE, BENZENE], ids=["water", "ammonia", "methane", "benzene"])
def test_the_projection_is_idempotent_and_keeps_force_and_torque_zero(atoms):
    z, x = _split(atoms)
    c = x.mean(axis=0)
    g = _torque_and_force_free(x, np.random.default_rng(7))
    assert np.abs(g.sum(axis=0)).max() < 1e-12
    assert np.abs(np.cross(x - c, g).sum(axis=0)).max() < 1e-12
    s = pg.symmetrize_gradient(g, z, x)
    assert np.abs(s.sum(axis=0)).max() < 1e-12
    assert np.abs(np.cross(x - c, s).sum(axis=0)).max() < 1e-12
    assert np.abs(pg.symmetrize_gradient(s, z, x) - s).max() < 1e-14


def test_the_projection_keeps_the_symmetric_part_and_removes_the_rest():
    z, x = _split(WATER)
    symmetric = np.array([[0.0, 0.0, -0.2], [0.0, 0.05, 0.1], [0.0, -0.05, 0.1]])
    assert np.abs(pg.symmetrize_gradient(symmetric, z, x) - symmetric).max() < 1e-15
    # The tie artifact: a force on the oxygen along the hydrogens' axis, balanced
    # by the hydrogens, which the mirror symmetry forbids.
    spurious = np.array([[0.0, 1.4e-3, 0.0], [0.0, -7.0e-4, 0.0], [0.0, -7.0e-4, 0.0]])
    assert np.abs(pg.symmetrize_gradient(spurious, z, x)).max() < 1e-15


def test_a_solute_with_no_symmetry_is_returned_unchanged_bit_for_bit():
    low = [(8, (0.0, 0.0, 0.0)), (1, (0.3, 0.757, 0.587)), (1, (-0.2, -0.9, 0.7)), (9, (1.0, 0.1, -0.4))]
    z, x = _split(low)
    g = np.random.default_rng(2).normal(size=x.shape)
    out = pg.symmetrize_gradient(g, z, x)
    assert np.array_equal(out, g) and out is not g


def test_a_planar_solute_without_other_symmetry_changes_only_at_roundoff():
    """Bent asymmetric water keeps its molecular plane as an operation, so the
    projection runs; its gradient has no out-of-plane part, so only roundoff moves."""
    z, x = _split(WATER_ASYM)
    g = np.array([[1e-17, 0.036, -0.020], [-2e-17, -0.060, 0.005], [1e-18, 0.024, 0.015]])
    assert np.abs(pg.symmetrize_gradient(g, z, x) - g).max() < 1e-16


def test_the_solvent_model_validates_its_symmetry_tolerance():
    vq = pytest.importorskip("vibeqc")
    assert vq.SolventModel(epsilon=78.39).gradient_symmetry_tol_bohr == pg.DEFAULT_SYMMETRY_TOL_BOHR
    assert vq.SolventModel(epsilon=78.39, gradient_symmetry_tol_bohr=0).gradient_symmetry_tol_bohr == 0.0
    for bad in (-1e-4, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gradient_symmetry_tol_bohr"):
            vq.SolventModel(epsilon=78.39, gradient_symmetry_tol_bohr=bad)
