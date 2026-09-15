"""Tests for the spherical multipole moment buffer (BIPOLE-EXACT-ZONE incr 3a).

Validates the Cartesian→spherical conversion, buffer construction, and
on-demand retrieval against analytical reference values.
"""

from __future__ import annotations

import numpy as np
import pytest

# Tests are designed to be runnable without the C++ core by mocking
# the PairMultipoleMoments input.  Skip the full-buffer test if
# the C++ core is not importable.
try:
    import vibeqc as vq

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

from vibeqc.bipole_multipole import n_components
from vibeqc.bipole_spherical_moment_buffer import (
    SphericalMomentBuffer,
    build_spherical_moment_buffer,
    spherical_pair_moments,
)


def _mock_pair_moments(nbf=4, n_sh=2, n_cells=2):
    """Build a minimal PairMultipoleMoments for testing."""
    from vibeqc.bipole_pair_moments import PairMultipoleMoments

    cells = []
    for idx in [(0, 0, 0), (1, 0, 0)]:
        cells.append(
            type(
                "MockCell",
                (),
                {
                    "index": np.array(idx, dtype=np.int32),
                    "r_cart": np.array(idx, dtype=float),
                },
            )()
        )

    blocks = []
    centers = []
    for c in range(n_cells):
        comp_blocks = []
        for comp in range(10):
            mat = np.zeros((nbf, nbf), dtype=float)
            # Put non-zero values in the first shell-pair sub-block.
            if comp == 0:  # overlap = 1.0 for (0,0) shell pair
                mat[0, 0] = 1.0
                mat[1, 1] = 1.0
            elif comp in (1, 3):  # dipole x = 0.5, z = 0.3
                mat[0, 0] = 0.5 if comp == 1 else 0.3
            comp_blocks.append(mat)
        blocks.append(comp_blocks)
        # Centers: shell 0 at origin, shell 1 at (1,0,0) in home cell,
        # shifted by cell vector.
        g = np.array(cells[c].r_cart, dtype=float)
        sh_centers = np.zeros((n_sh, n_sh, 3), dtype=float)
        sh_centers[0, 1] = [0.5, 0.0, 0.0] + g * 0.5
        sh_centers[1, 0] = [0.5, 0.0, 0.0] + g * 0.5
        centers.append(sh_centers)

    return PairMultipoleMoments(
        nbf=nbf,
        L_max=2,
        cells=cells,
        blocks=blocks,
        centers=centers,
        shell_slices=[(0, 2), (2, 2)],
    )


def _mock_basis():
    """Build a minimal BasisSet mock for two s-shells."""
    from unittest.mock import MagicMock

    basis = MagicMock()
    sh0 = MagicMock()
    sh0.l = 0
    sh0.pure = True
    sh0.exponents = np.array([0.5, 2.0])
    sh0.origin = np.array([0.0, 0.0, 0.0])
    sh1 = MagicMock()
    sh1.l = 0
    sh1.pure = True
    sh1.exponents = np.array([0.3, 1.5])
    sh1.origin = np.array([1.0, 0.0, 0.0])
    basis.shells.return_value = [sh0, sh1]
    basis.nbasis = 2
    return basis


class TestSphericalMomentBuffer:
    """Test buffer construction and retrieval."""

    def test_buffer_construction(self):
        """Buffer builds without error and has entries for known pairs."""
        pair_mom = _mock_pair_moments(nbf=4, n_sh=2, n_cells=2)
        basis = _mock_basis()
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)

        assert buf.L_max == 2
        assert buf.n_sph == n_components(2)  # 9
        assert len(buf.cells) == 2

        # Home cell: shell pair (0,0) should exist.
        mom = buf.get_moments(0, 0, 0)
        assert mom is not None
        assert mom.shape == (2, 2, 9)

        centre = buf.get_center(0, 0, 0)
        assert centre is not None
        assert centre.shape == (3,)

    def test_buffer_transpose_entries(self):
        """(s1,s2) and (s2,s1) both have entries."""
        pair_mom = _mock_pair_moments(nbf=4, n_sh=2, n_cells=2)
        basis = _mock_basis()
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)

        mom_01 = buf.get_moments(0, 1, 0)
        mom_10 = buf.get_moments(1, 0, 0)
        assert mom_01 is not None
        assert mom_10 is not None
        # Transpose should match.
        np.testing.assert_allclose(
            mom_10, np.transpose(mom_01, (1, 0, 2)), atol=1e-14
        )

    def test_spherical_monopole_is_overlap(self):
        """Z_00 component = the Cartesian overlap (monopole)."""
        pair_mom = _mock_pair_moments(nbf=4, n_sh=2, n_cells=2)
        basis = _mock_basis()
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)

        mom_00 = buf.get_moments(0, 0, 0)
        assert mom_00 is not None
        # Monopole (index 0) should be exactly the overlap from Cartesian.
        # In our mock, overlap = 1.0 for (0,0) and (1,1).
        np.testing.assert_allclose(mom_00[0, 0, 0], 1.0, atol=1e-14)
        np.testing.assert_allclose(mom_00[1, 1, 0], 1.0, atol=1e-14)

    def test_spherical_moments_truncation(self):
        """spherical_pair_moments with L_truncate returns fewer components."""
        pair_mom = _mock_pair_moments(nbf=4, n_sh=2, n_cells=2)
        basis = _mock_basis()
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=4)

        mom_full = spherical_pair_moments(buf, 0, 0, 0)
        mom_l0 = spherical_pair_moments(buf, 0, 0, 0, L_truncate=0)
        mom_l1 = spherical_pair_moments(buf, 0, 0, 0, L_truncate=1)

        assert mom_full.shape[2] == n_components(4)  # 25
        assert mom_l0.shape[2] == n_components(0)  # 1
        assert mom_l1.shape[2] == n_components(1)  # 4

        # First components should match.
        np.testing.assert_allclose(
            mom_full[:, :, : n_components(1)],
            mom_l1,
            atol=1e-14,
        )

    def test_missing_pair_returns_none(self):
        """Non-existent pair returns None."""
        pair_mom = _mock_pair_moments(nbf=4, n_sh=2, n_cells=2)
        basis = _mock_basis()
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)

        assert buf.get_moments(99, 99, 0) is None
        assert buf.get_center(99, 99, 0) is None


@pytest.mark.skipif(not _CORE_AVAILABLE, reason="C++ core not importable")
class TestSphericalMomentBufferWithCore:
    """Full integration test using the real C++ multipole moments."""

    def test_real_system_buffer(self):
        """Build buffer for a real LiH cell and verify monopole agreement."""
        from vibeqc._vibeqc_core import (
            LatticeSumOptions,
            compute_multipole_moments_lattice,
        )
        from vibeqc.bipole_pair_moments import pair_center_moments

        ANG2BOHR = 1.0 / 0.529177210903
        a = 4.084 * ANG2BOHR
        lattice = (a / 2.0) * np.array(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )
        system = vq.PeriodicSystem(
            3,
            lattice,
            [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [a / 2, a / 2, a / 2])],
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

        lo = LatticeSumOptions()
        lo.cutoff_bohr = 6.0
        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0)
        )
        pair_mom = pair_center_moments(cart, basis)
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)

        assert buf.L_max == 2
        assert len(buf) > 0

        # Verify the overlap (monopole) agrees with the S matrix.
        from vibeqc._vibeqc_core import (
            compute_overlap_lattice,
        )

        lo_s = LatticeSumOptions()
        lo_s.cutoff_bohr = 6.0
        S_lat = compute_overlap_lattice(basis, system, lo_s)

        for cell_idx, cell in enumerate(buf.cells[:len(S_lat.cells)]):
            mom = buf.get_moments(0, 0, cell_idx)
            if mom is None:
                continue
            S_block = np.asarray(S_lat.blocks[cell_idx], dtype=float)
            b1, n1 = buf.shell_slices[0]
            b2, n2 = buf.shell_slices[0]
            S_sub = S_block[b1:b1 + n1, b2:b2 + n2]
            overlap_from_buffer = mom[:, :, 0]  # Z_00 = monopole = S
            # Only check elements that have non-zero overlap.
            mask = np.abs(S_sub) > 1e-10
            if mask.any():
                np.testing.assert_allclose(
                    overlap_from_buffer[mask],
                    S_sub[mask],
                    atol=1e-12,
                )
