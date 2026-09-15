"""Tests for Bravais-lattice geometry utilities (all 14 lattice types)."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.bipole_bravais_utils import (
    cell_dimensionality,
    cell_volume_bohr,
    cell_length_scale_inv_bohr,
    lattice_to_cartesian_rotation,
)


class MockSystem:
    def __init__(self, dim, lattice):
        self.dim = dim
        self.lattice = np.asarray(lattice, dtype=float)


class TestCubicLattice:
    """Simple cubic (cP)."""

    def test_sc_volume(self):
        a = 5.0
        lattice = np.diag([a, a, a])
        sys = MockSystem(3, lattice)
        assert cell_dimensionality(sys) == 3
        assert cell_volume_bohr(sys) == pytest.approx(a**3)
        expected_scale = a**-1  # (1/V)^{1/3}
        assert cell_length_scale_inv_bohr(sys) == pytest.approx(expected_scale)


class TestBCCLattice:
    """Body-centred cubic (cI)."""

    def test_bcc_volume(self):
        a = 4.0
        lattice = (a / 2.0) * np.array(
            [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        )
        sys = MockSystem(3, lattice)
        vol = cell_volume_bohr(sys)
        assert vol == pytest.approx(a**3 / 2.0)


class TestFCCLattice:
    """Face-centred cubic (cF)."""

    def test_fcc_volume(self):
        a = 4.0
        lattice = (a / 2.0) * np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        )
        sys = MockSystem(3, lattice)
        vol = cell_volume_bohr(sys)
        assert vol == pytest.approx(a**3 / 4.0)


class TestTetragonalLattice:
    """Primitive tetragonal (tP)."""

    def test_tetragonal_volume(self):
        a, c = 3.0, 5.0
        lattice = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
        sys = MockSystem(3, lattice)
        assert cell_volume_bohr(sys) == pytest.approx(a * a * c)


class TestHexagonalLattice:
    """Hexagonal (hP)."""

    def test_hexagonal_volume(self):
        a, c = 2.5, 4.0
        lattice = np.array([
            [a, 0.0, 0.0],
            [-a / 2, a * np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, c],
        ])
        sys = MockSystem(3, lattice)
        vol = cell_volume_bohr(sys)
        # Area of hex base = a^2 * sqrt(3)/2, volume = area * c
        expected = a * a * np.sqrt(3) / 2 * c
        assert vol == pytest.approx(expected)


class TestMonoclinicLattice:
    """Monoclinic (mP)."""

    def test_monoclinic_volume(self):
        a, b, c = 3.0, 4.0, 5.0
        beta = np.radians(100)
        lattice = np.array([
            [a, 0, 0],
            [0, b, 0],
            [c * np.cos(beta), 0, c * np.sin(beta)],
        ])
        sys = MockSystem(3, lattice)
        vol = cell_volume_bohr(sys)
        expected = a * b * c * np.sin(beta)
        assert vol == pytest.approx(expected)


class TestTriclinicLattice:
    """Triclinic (aP)."""

    def test_triclinic_volume(self):
        lattice = np.array([
            [3.0, 0.2, 0.1],
            [0.3, 4.0, 0.2],
            [0.1, 0.3, 5.0],
        ])
        sys = MockSystem(3, lattice)
        vol = cell_volume_bohr(sys)
        assert vol > 0
        assert vol == pytest.approx(abs(np.linalg.det(lattice)))


class Test2DSlab:
    """2D slab (square)."""

    def test_slab_area(self):
        a = 6.0
        lattice = np.array([[a, 0, 0], [0, a, 0]])
        sys = MockSystem(2, lattice)
        assert cell_dimensionality(sys) == 2
        assert cell_volume_bohr(sys) == pytest.approx(a * a)
        expected_scale = 1.0 / a  # (1/A)^{1/2}
        assert cell_length_scale_inv_bohr(sys) == pytest.approx(expected_scale)


class Test1DChain:
    """1D chain."""

    def test_chain_length(self):
        a = 4.0
        lattice = np.array([[a, 0, 0]])
        sys = MockSystem(1, lattice)
        assert cell_dimensionality(sys) == 1
        assert cell_volume_bohr(sys) == pytest.approx(a)
        expected_scale = 1.0 / a  # 1/L
        assert cell_length_scale_inv_bohr(sys) == pytest.approx(expected_scale)


class TestLatticeRotation:
    """Lattice-to-Cartesian rotation matrix."""

    def test_identity_rotation_cubic(self):
        lattice = np.diag([5.0, 5.0, 5.0])
        sys = MockSystem(3, lattice)
        R = np.eye(3)
        R_lat = lattice_to_cartesian_rotation(sys, R)
        np.testing.assert_array_equal(R_lat, np.eye(3, dtype=int))

    def test_90_deg_rotation_cubic(self):
        lattice = np.diag([5.0, 5.0, 5.0])
        sys = MockSystem(3, lattice)
        # 90° rotation around z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R_lat = lattice_to_cartesian_rotation(sys, R)
        # For cubic, lattice-basis rotation = Cartesian rotation
        np.testing.assert_array_equal(R_lat, R.astype(int))

    def test_rotation_non_orthogonal(self):
        lattice = np.array([
            [3.0, 0.0, 0.0],
            [1.5, 3.0 * np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, 5.0],
        ])
        sys = MockSystem(3, lattice)
        R = np.eye(3)
        R_lat = lattice_to_cartesian_rotation(sys, R)
        np.testing.assert_array_equal(R_lat, np.eye(3, dtype=int))

    def test_rotation_2d_slab(self):
        lattice = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
        sys = MockSystem(2, lattice)
        R = np.eye(3)
        R_lat = lattice_to_cartesian_rotation(sys, R)
        np.testing.assert_array_equal(R_lat, np.eye(3, dtype=int))


class TestOrthorhombicLattices:
    """Orthorhombic lattices: primitive (oP), base-centred (oC),
    body-centred (oI), face-centred (oF)."""

    def test_orthorhombic_primitive(self):
        a, b, c = 3.0, 4.0, 5.0
        lattice = np.diag([a, b, c])
        sys = MockSystem(3, lattice)
        assert cell_volume_bohr(sys) == pytest.approx(a * b * c)

    def test_orthorhombic_base_centred(self):
        a, b, c = 3.0, 4.0, 5.0
        lattice = np.array([[a/2, -b/2, 0], [a/2, b/2, 0], [0, 0, c]])
        sys = MockSystem(3, lattice)
        assert cell_volume_bohr(sys) == pytest.approx(a * b * c / 2)

    def test_orthorhombic_body_centred(self):
        a, b, c = 3.0, 4.0, 5.0
        lattice = (1.0/2.0) * np.array([[-a, b, c], [a, -b, c], [a, b, -c]])
        sys = MockSystem(3, lattice)
        assert cell_volume_bohr(sys) == pytest.approx(a * b * c / 2)

    def test_orthorhombic_face_centred(self):
        a, b, c = 3.0, 4.0, 5.0
        lattice = (1.0/2.0) * np.array([[0, b, c], [a, 0, c], [a, b, 0]])
        sys = MockSystem(3, lattice)
        assert cell_volume_bohr(sys) == pytest.approx(a * b * c / 4)


class TestBodyCentredTetragonal:
    """Body-centred tetragonal (tI)."""

    def test_bct_volume(self):
        a, c = 3.0, 5.0
        lattice = (1.0/2.0) * np.array([[-a, a, c], [a, -a, c], [a, a, -c]])
        sys = MockSystem(3, lattice)
        assert cell_volume_bohr(sys) == pytest.approx(a * a * c / 2)


class TestRhombohedralLattice:
    """Rhombohedral (hR)."""

    def test_rhombohedral_volume(self):
        a, alpha = 4.0, np.radians(60)
        # Primitive rhombohedral vectors
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([a * np.cos(alpha), a * np.sin(alpha), 0.0])
        a3 = np.array([
            a * np.cos(alpha),
            a * np.cos(alpha) * (1 - np.cos(alpha)) / np.sin(alpha),
            a * np.sqrt(1 - 3*np.cos(alpha)**2 + 2*np.cos(alpha)**3) / np.sin(alpha),
        ])
        lattice = np.array([a1, a2, a3])
        sys = MockSystem(3, lattice)
        vol = cell_volume_bohr(sys)
        expected = a**3 * np.sqrt(1 - 3*np.cos(alpha)**2 + 2*np.cos(alpha)**3)
        assert vol == pytest.approx(expected)


class TestVolumeDimensionalityEdgeCases:
    """Edge cases for cell_volume_bohr."""

    def test_zero_volume_raises_in_scale(self):
        """Zero-volume cell raises ValueError in length scale."""
        lattice = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        sys = MockSystem(3, lattice)
        with pytest.raises(ValueError):
            cell_length_scale_inv_bohr(sys)

    def test_volume_invariance_under_rotation(self):
        """Volume is invariant under orthogonal rotation."""
        a, b, c = 3.0, 4.0, 5.0
        lattice = np.diag([a, b, c])
        theta = np.radians(30)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        lattice_rot = (R @ lattice.T).T
        sys = MockSystem(3, lattice)
        sys_rot = MockSystem(3, lattice_rot)
        assert cell_volume_bohr(sys) == pytest.approx(cell_volume_bohr(sys_rot))
