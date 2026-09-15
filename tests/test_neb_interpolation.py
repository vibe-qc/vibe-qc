"""Tests for NEB interpolation primitives (Increment 1).

Scope: ``vibeqc.interpolate_linear`` and ``vibeqc.interpolate_idpp``
+ the ``NEBImage`` / ``NEBPath`` dataclasses. The full NEB driver
(tangent + spring + climbing image) is not exercised here — that's
later increments.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule, PeriodicSystem
from vibeqc.neb import (
    NEBImage,
    NEBPath,
    interpolate_idpp,
    interpolate_linear,
)


# ---- Small helpers --------------------------------------------------------


def _h2(r: float) -> Molecule:
    """H–H along z at separation ``r`` (bohr)."""
    return Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, r])])


def _h3_linear(rab: float, rbc: float) -> Molecule:
    """H–H–H collinear (doublet): H(0) — H(rab) — H(rab+rbc), along z."""
    return Molecule(
        [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, rab]),
            Atom(1, [0.0, 0.0, rab + rbc]),
        ],
        0,
        2,
    )


def _positions(system) -> np.ndarray:
    return np.array([list(a.xyz) for a in system.atoms], dtype=float)


def _min_pair_distance(system) -> float:
    pos = _positions(system)
    n = len(pos)
    pairs = [
        float(np.linalg.norm(pos[i] - pos[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return min(pairs) if pairs else float("inf")


# ---- Dataclasses ---------------------------------------------------------


class TestNEBImage:
    def test_default_slots_unset(self):
        img = NEBImage(system=_h2(1.4))
        assert img.energy is None
        assert img.gradient is None
        assert img.tangent is None

    def test_attributes_assignable(self):
        img = NEBImage(system=_h2(1.4))
        img.energy = -1.117
        img.gradient = np.zeros((2, 3))
        img.tangent = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        assert img.energy == pytest.approx(-1.117)
        assert img.gradient.shape == (2, 3)
        assert np.allclose(img.tangent[1], [0.0, 0.0, 1.0])


class TestNEBPath:
    def test_n_intermediate_excludes_endpoints(self):
        imgs = [NEBImage(_h2(r)) for r in (1.0, 1.2, 1.4, 1.6, 1.8)]
        path = NEBPath(images=imgs)
        assert path.n_images == 5
        assert path.n_intermediate == 3
        assert path.spring_constant == pytest.approx(0.1)
        assert path.climbing_image_index is None

    def test_energies_returns_nan_for_unset(self):
        imgs = [NEBImage(_h2(r)) for r in (1.0, 1.4, 1.8)]
        imgs[0].energy = -1.0
        imgs[2].energy = -1.05
        e = NEBPath(images=imgs).energies()
        assert e[0] == pytest.approx(-1.0)
        assert np.isnan(e[1])
        assert e[2] == pytest.approx(-1.05)


# ---- Linear interpolation -------------------------------------------------


class TestInterpolateLinear:
    def test_endpoints_preserved_identically(self):
        r = _h2(1.0)
        p = _h2(2.0)
        path = interpolate_linear(r, p, n_images=3)
        assert path[0] is r
        assert path[-1] is p
        assert len(path) == 5

    def test_midpoint_is_arithmetic_mean(self):
        r = _h2(1.0)
        p = _h2(2.0)
        path = interpolate_linear(r, p, n_images=1)
        mid = _positions(path[1])
        np.testing.assert_allclose(
            mid, 0.5 * (_positions(r) + _positions(p))
        )

    def test_three_images_evenly_spaced(self):
        r = _h2(1.0)
        p = _h2(4.0)
        path = interpolate_linear(r, p, n_images=3)
        bonds = [
            float(np.linalg.norm(_positions(s)[1] - _positions(s)[0]))
            for s in path
        ]
        # Endpoints: 1.0, 4.0. Three intermediate at 1.75, 2.5, 3.25.
        np.testing.assert_allclose(bonds, [1.0, 1.75, 2.5, 3.25, 4.0])

    def test_zero_images_returns_endpoints_only(self):
        r = _h2(1.0)
        p = _h2(2.0)
        path = interpolate_linear(r, p, n_images=0)
        assert len(path) == 2
        assert path[0] is r
        assert path[1] is p

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="n_images must be >= 0"):
            interpolate_linear(_h2(1.0), _h2(2.0), n_images=-1)

    def test_atom_count_mismatch_raises(self):
        r = _h2(1.0)
        p = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 2.0]),
             Atom(1, [0.0, 0.0, 4.0])],
            0, 2,
        )
        with pytest.raises(ValueError, match="matching atomic-number"):
            interpolate_linear(r, p, n_images=3)

    def test_atom_order_mismatch_raises(self):
        r = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(8, [0.0, 0.0, 2.0])], 0, 2)
        p = Molecule([Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 2.0])], 0, 2)
        with pytest.raises(ValueError, match="matching atomic-number"):
            interpolate_linear(r, p, n_images=3)

    def test_type_mismatch_raises(self):
        mol = _h2(1.0)
        sys = PeriodicSystem(
            3,
            np.eye(3) * 10.0,
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.0])],
        )
        with pytest.raises(ValueError, match="same system type"):
            interpolate_linear(mol, sys, n_images=3)


class TestInterpolateLinearPeriodic:
    def test_periodic_lattice_preserved(self):
        L = np.diag([10.0, 10.0, 10.0])
        r = PeriodicSystem(
            3, L,
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.0])],
        )
        p = PeriodicSystem(
            3, L,
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])],
        )
        path = interpolate_linear(r, p, n_images=2)
        for img in path:
            assert isinstance(img, PeriodicSystem)
            np.testing.assert_allclose(np.asarray(img.lattice), L)

    def test_mismatched_lattice_raises(self):
        r = PeriodicSystem(
            3, np.diag([10.0, 10.0, 10.0]),
            [Atom(1, [0.0, 0.0, 0.0])],
        )
        p = PeriodicSystem(
            3, np.diag([10.0, 10.0, 11.0]),
            [Atom(1, [0.0, 0.0, 0.0])],
        )
        with pytest.raises(ValueError, match="same lattice"):
            interpolate_linear(r, p, n_images=3)


# ---- IDPP interpolation ---------------------------------------------------


class TestInterpolateIDPP:
    def test_endpoints_preserved_identically(self):
        r = _h2(1.0)
        p = _h2(2.0)
        path = interpolate_idpp(r, p, n_images=3)
        assert path[0] is r
        assert path[-1] is p
        assert len(path) == 5

    def test_h2_stretch_path_matches_linear(self):
        """For a single-bond stretch, IDPP and linear paths must agree
        — there are no clashes to relieve and IDPP collapses to the
        linear interpolant in pair-distance space."""
        r = _h2(1.0)
        p = _h2(3.0)
        idpp = interpolate_idpp(r, p, n_images=3)
        lin = interpolate_linear(r, p, n_images=3)
        for a, b in zip(idpp, lin):
            np.testing.assert_allclose(_positions(a), _positions(b), atol=1e-4)

    def test_h3_collinear_no_atom_overlap(self):
        """H–H–H collinear: reactant has bond A–B short, product has
        bond B–C short. Linear interpolation between them places B
        nearly on top of itself (well, here on a straight line it's
        fine — but in the perpendicular case below it isn't)."""
        r = _h3_linear(1.5, 3.5)
        p = _h3_linear(3.5, 1.5)
        path = interpolate_idpp(r, p, n_images=5)
        for img in path:
            assert _min_pair_distance(img) > 0.9  # no atom-atom clash

    def test_idpp_differs_from_linear_in_2d(self):
        """Genuine 2-D rearrangement: a peripheral atom moves from one
        side of a base pair to the other. The linear path drives that
        atom through the inter-pair midline, compressing one of the
        new bonds. IDPP must find a curved trajectory and produce a
        per-image pair-distance series that differs measurably from
        the linear one."""
        # Reactant: A(0,0) - B(2,0) base pair, C bonded to A from
        # above (at y=+1.6). Product: same A,B; C bonded to B from
        # above. C swings through the AB midline along the way.
        r = Molecule(
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [2.0, 0.0, 0.0]),
                Atom(1, [0.0, 1.6, 0.0]),
            ],
            0, 2,
        )
        p = Molecule(
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [2.0, 0.0, 0.0]),
                Atom(1, [2.0, 1.6, 0.0]),
            ],
            0, 2,
        )
        lin = interpolate_linear(r, p, n_images=5)
        idpp = interpolate_idpp(r, p, n_images=5)
        # IDPP must not introduce overlaps and must move at least one
        # intermediate image away from where linear placed it.
        for img in idpp:
            assert _min_pair_distance(img) > 0.5
        displacements = [
            float(np.linalg.norm(_positions(a) - _positions(b)))
            for a, b in zip(idpp[1:-1], lin[1:-1])
        ]
        assert max(displacements) > 0.05

    def test_zero_images_returns_endpoints(self):
        r = _h2(1.0)
        p = _h2(2.0)
        path = interpolate_idpp(r, p, n_images=0)
        assert path == [r, p]

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="n_images must be >= 0"):
            interpolate_idpp(_h2(1.0), _h2(2.0), n_images=-1)


# ---- IDPP analytic gradient sanity ---------------------------------------


class TestIDPPGradient:
    """Finite-difference check on the IDPP objective's analytic
    gradient — guards against algebra errors in
    ``_idpp_value_and_grad``."""

    def test_analytic_matches_fd_on_h3(self):
        from vibeqc.neb import _idpp_value_and_grad, _pair_distance_matrix

        pos_r = np.array(
            [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]]
        )
        pos_p = np.array(
            [[0.0, 0.0, 0.0], [1.6, 0.3, 0.0], [3.0, 0.0, 0.0]]
        )
        d_R = _pair_distance_matrix(pos_r)
        d_P = _pair_distance_matrix(pos_p)
        target = 0.5 * (d_R + d_P)
        x0 = pos_r.ravel() + 0.05 * np.array(
            [0.1, -0.2, 0.0, 0.3, 0.05, 0.0, -0.1, 0.0, 0.0]
        )

        _, g_an = _idpp_value_and_grad(x0, target, n_atoms=3)
        eps = 1e-6
        g_fd = np.zeros_like(x0)
        for i in range(len(x0)):
            xp = x0.copy()
            xp[i] += eps
            xm = x0.copy()
            xm[i] -= eps
            ep, _ = _idpp_value_and_grad(xp, target, n_atoms=3)
            em, _ = _idpp_value_and_grad(xm, target, n_atoms=3)
            g_fd[i] = (ep - em) / (2 * eps)
        np.testing.assert_allclose(g_an, g_fd, rtol=1e-4, atol=1e-6)
