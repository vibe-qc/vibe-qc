"""Minimum-image convention in periodic NEB (interpolation + force kernel).

Periodic IDPP pair distances and the NEB force (tangent + spring) use
the minimum-image convention so a band whose images straddle a cell
boundary — the surface self-diffusion case, an adatom hopping across
the PBC — is handled correctly instead of being dragged the long way
across the cell. These tests pin:

* the minimum-image displacement helper (wrap, no-op, dim<3),
* MIC pair distances vs raw Cartesian,
* the MIC IDPP analytic gradient against finite differences (the
  selected lattice image is only locally constant, so this guards the
  gradient),
* that IDPP targets the physical (minimum-image) distance,
* that the driver spring shrinks to the minimum-image gap across a
  boundary, and is a *no-op* for a compact (non-crossing) band — so
  existing periodic NEB behaviour is unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.neb import (
    _idpp_value_and_grad,
    _minimum_image_diff,
    _neb_forces,
    _pair_distance_matrix,
    interpolate_idpp,
)

L12 = np.diag([12.0, 12.0, 12.0])


# ---------------------------------------------------------------------------
# minimum-image displacement helper
# ---------------------------------------------------------------------------


def test_minimum_image_diff_wraps_cross_boundary():
    # 11 bohr along x in a 12-bohr cell is -1 bohr to the nearest image.
    out = _minimum_image_diff(np.array([[11.0, 0.0, 0.0]]), L12, 3)
    assert np.allclose(out, [[-1.0, 0.0, 0.0]])


def test_minimum_image_diff_noop_for_small_displacement():
    # A displacement well within half a cell is returned unchanged.
    d = np.array([[1.0, -2.0, 3.0]])
    assert np.allclose(_minimum_image_diff(d, L12, 3), d)


def test_minimum_image_diff_dim2_leaves_nonperiodic_axis():
    # dim=2: x wraps (11 -> -1); the z axis (vacuum / non-periodic) is
    # outside the periodic column span and must be left untouched.
    out = _minimum_image_diff(np.array([[11.0, 0.0, 11.0]]), L12, 2)
    assert np.allclose(out, [[-1.0, 0.0, 11.0]])


def test_minimum_image_diff_skewed_column_lattice_matches_cvp_oracle():
    # PeriodicSystem stores lattice vectors in columns. For this tilted,
    # skewed 2-D cell, component-wise rounding gives the zero image for the
    # fractional displacement (0.49, 0.49), but the closest image subtracts
    # the second lattice column. A small brute-force CVP oracle pins both the
    # column convention and the neighbouring-integer search.
    lattice = np.array(
        [
            [4.0, 3.5, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 6.0],
        ]
    )
    periodic_vectors = lattice[:, :2]
    diff = 0.49 * (periodic_vectors[:, 0] + periodic_vectors[:, 1])
    candidates = [
        diff - periodic_vectors @ np.array([i, j])
        for i in range(-2, 3)
        for j in range(-2, 3)
    ]
    expected = min(candidates, key=np.linalg.norm)

    out = _minimum_image_diff(diff[None, :], lattice, 2)[0]

    np.testing.assert_allclose(expected, diff - periodic_vectors[:, 1])
    np.testing.assert_allclose(out, expected)


def test_minimum_image_diff_is_invariant_to_large_integer_shear():
    # The second column contains five copies of the first. Fractional rounding
    # in this unreduced basis misses the true (3, 0) image by more than one
    # integer cell. Minkowski reduction must recover the same physical image as
    # the equivalent reduced basis b' = b - 5a.
    lattice = np.array(
        [
            [4.0, 20.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 8.0],
        ]
    )
    reduced_lattice = lattice.copy()
    reduced_lattice[:, 1] -= 5 * reduced_lattice[:, 0]
    periodic_vectors = lattice[:, :2]
    diff = 0.49 * (periodic_vectors[:, 0] + periodic_vectors[:, 1])
    candidates = [
        diff - periodic_vectors @ np.array([i, j])
        for i in range(-6, 7)
        for j in range(-6, 7)
    ]
    expected = min(candidates, key=np.linalg.norm)

    unreduced = _minimum_image_diff(diff[None, :], lattice, 2)[0]
    reduced = _minimum_image_diff(diff[None, :], reduced_lattice, 2)[0]

    np.testing.assert_allclose(expected, diff - 3 * periodic_vectors[:, 0])
    np.testing.assert_allclose(unreduced, expected)
    np.testing.assert_allclose(reduced, expected)


# ---------------------------------------------------------------------------
# pair distances
# ---------------------------------------------------------------------------


def test_pair_distance_matrix_minimum_image():
    pos = np.array([[1.0, 6.0, 6.0], [11.0, 6.0, 6.0]])  # 10 apart, 2 via MIC
    assert _pair_distance_matrix(pos)[0, 1] == pytest.approx(10.0)
    assert _pair_distance_matrix(pos, L12, 3)[0, 1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# IDPP objective gradient
# ---------------------------------------------------------------------------


def _fd_grad(x0, target, n, lattice, dim, h=1e-6):
    g = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        ep, _ = _idpp_value_and_grad(xp, target, n, lattice, dim)
        em, _ = _idpp_value_and_grad(xm, target, n, lattice, dim)
        g[i] = (ep - em) / (2 * h)
    return g


def test_idpp_gradient_matches_fd_periodic():
    # Geometry near (but not on) the wrap boundaries, where the MIC
    # lattice-image selection is locally constant and the analytic gradient
    # must hold.
    x0 = np.array([[2.0, 3.0, 4.0], [9.5, 4.0, 5.0], [5.0, 10.5, 3.0]]).ravel()
    target = _pair_distance_matrix(
        np.array([[2.0, 3.0, 4.0], [9.0, 4.0, 5.0], [5.0, 9.0, 3.0]]), L12, 3
    )
    _, g_an = _idpp_value_and_grad(x0, target, 3, L12, 3)
    g_fd = _fd_grad(x0, target, 3, L12, 3)
    assert np.max(np.abs(g_an - g_fd)) < 1e-6


def test_idpp_gradient_matches_fd_molecular():
    # Regression: the lattice=None gradient (existing molecular path).
    x0 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.4, 0.3]]).ravel()
    target = _pair_distance_matrix(
        np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.5, 0.2]])
    )
    _, g_an = _idpp_value_and_grad(x0, target, 3)
    g_fd = _fd_grad(x0, target, 3, None, 3)
    assert np.max(np.abs(g_an - g_fd)) < 1e-6


# ---------------------------------------------------------------------------
# IDPP interpolation across a boundary
# ---------------------------------------------------------------------------


def _periodic(z_hopper_x):
    return vq.PeriodicSystem(
        3, L12,
        [vq.Atom(1, [3.0, 6.0, 6.0]),
         vq.Atom(1, [9.0, 5.0, 6.0]),
         vq.Atom(1, [z_hopper_x, 6.0, 6.0])],
    )


def test_idpp_targets_minimum_image_distance():
    prod = _periodic(11.2)
    P = np.array([list(a.xyz) for a in prod.unit_cell])
    # anchor0 <-> hopping atom: raw Cartesian is the long way (~8.2),
    # the physical minimum-image distance is short (~3.8).
    assert _pair_distance_matrix(P)[0, 2] > 7.0
    assert _pair_distance_matrix(P, L12, 3)[0, 2] < 4.0


def test_idpp_images_track_minimum_image_target():
    react, prod = _periodic(0.8), _periodic(11.2)
    path = interpolate_idpp(react, prod, n_images=5)
    dR = _pair_distance_matrix(
        np.array([list(a.xyz) for a in react.unit_cell]), L12, 3)
    dP = _pair_distance_matrix(
        np.array([list(a.xyz) for a in prod.unit_cell]), L12, 3)
    worst = 0.0
    for k in range(1, 6):
        t = k / 6.0
        dk = _pair_distance_matrix(
            np.array([list(a.xyz) for a in path[k].unit_cell]), L12, 3)
        worst = max(worst, float(np.max(np.abs(dk - ((1 - t) * dR + t * dP)))))
    assert worst < 0.2  # IDPP minimised the minimum-image objective


# ---------------------------------------------------------------------------
# driver: tangent + spring
# ---------------------------------------------------------------------------


def test_neb_spring_minimum_image_across_boundary():
    # Asymmetric 3-image band: image 1 is across the x-boundary from
    # image 0 (min-image gap 1.5) but close to image 2 (gap 1.0). The
    # Cartesian spring sees a bogus ~10.5-bohr previous segment.
    positions = [np.array([[0.5, 0.0, 0.0]]),
                 np.array([[11.0, 0.0, 0.0]]),
                 np.array([[10.0, 0.0, 0.0]])]
    energies = [0.0, 1.0, 0.0]
    grads = [np.zeros((1, 3))] * 3
    f_cart, _ = _neb_forces(positions, energies, grads, 0.1, None)
    f_mic, _ = _neb_forces(positions, energies, grads, 0.1, None,
                           lattice=L12, dim=3)
    assert np.linalg.norm(f_mic[0]) < 0.2 * np.linalg.norm(f_cart[0])


def test_neb_forces_noop_for_compact_band():
    # A band with small inter-image gaps (no boundary crossing): the
    # minimum image of each displacement is the displacement itself, so
    # the periodic force is bit-identical to the Cartesian one. This is
    # why existing (non-crossing) periodic NEB behaviour is unchanged.
    rng = [np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]),
           np.array([[1.2, 1.1, 0.9], [3.1, 3.0, 3.2]]),
           np.array([[1.5, 1.0, 1.1], [3.3, 2.9, 3.1]]),
           np.array([[1.7, 1.2, 1.0], [3.5, 3.1, 3.0]])]
    energies = [0.0, 0.5, 0.8, 0.2]
    grads = [np.ones((2, 3)) * 0.01 * i for i in range(4)]
    f_cart, t_cart = _neb_forces(rng, energies, grads, 0.1, None)
    f_mic, t_mic = _neb_forces(rng, energies, grads, 0.1, None,
                               lattice=L12, dim=3)
    for fc, fm in zip(f_cart, f_mic):
        assert np.allclose(fc, fm)
    for tc, tm in zip(t_cart, t_mic):
        assert np.allclose(tc, tm)
