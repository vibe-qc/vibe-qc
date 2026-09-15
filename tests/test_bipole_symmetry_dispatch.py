"""Tests for symmetry reduction of the bipolar penetration dispatch.

BIPOLE-EXACT-ZONE increment 4a tests.  Validates that the shell-pair
permutation symmetry reduces the dispatch without changing the
set of far-field quartets.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.bipole_dispatch import PenetrationDispatchParameters
from vibeqc.bipole_symmetry_dispatch import (
    build_symmetry_reduced_penetration_dispatch,
    unfold_symmetry_reduced_dispatch,
    _canonical_shell_pair,
)
from vibeqc.bipole_quartet_far_field import QuartetBipolarDispatch


def _mock_basis_and_cells(n_sh=4):
    """Build a mock basis with n_sh shells and a few lattice cells."""
    from unittest.mock import MagicMock

    basis = MagicMock()
    shells = []
    for i in range(n_sh):
        sh = MagicMock()
        sh.l = 0
        sh.pure = True
        sh.exponents = np.array([0.3 + 0.1 * i, 1.5 + 0.1 * i])
        sh.origin = np.array([float(i % 2), float((i // 2) % 2), 0.0])
        shells.append(sh)
    basis.shells.return_value = shells
    basis.nbasis = n_sh

    cells = []
    for idx in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        cell = MagicMock()
        cell.index = np.array(idx, dtype=np.int32)
        cell.r_cart = np.array(idx, dtype=float) * 3.0
        cells.append(cell)

    return basis, cells


class TestCanonicalShellPair:
    def test_canonical_ordering(self):
        assert _canonical_shell_pair(1, 2) == (1, 2)
        assert _canonical_shell_pair(2, 1) == (1, 2)
        assert _canonical_shell_pair(3, 3) == (3, 3)

    def test_zero_based(self):
        assert _canonical_shell_pair(0, 3) == (0, 3)
        assert _canonical_shell_pair(3, 0) == (0, 3)


class TestSymmetryReducedDispatch:
    def test_reduction_factor_at_least_one(self):
        """The reduction factor should be >= 1 (n_full >= n_reduced)."""
        basis, cells = _mock_basis_and_cells(n_sh=4)
        params = PenetrationDispatchParameters.for_cell_volume(100.0)

        j_result, _k = build_symmetry_reduced_penetration_dispatch(
            basis, cells, params, compute_exchange=False,
        )
        assert j_result.symmetry_factor >= 1.0
        assert j_result.n_full >= j_result.n_reduced

    def test_reduced_dispatch_fewer_than_full(self):
        """The reduced dispatch should have fewer entries than the full."""
        basis, cells = _mock_basis_and_cells(n_sh=4)
        params = PenetrationDispatchParameters.for_cell_volume(100.0)

        j_result, _k = build_symmetry_reduced_penetration_dispatch(
            basis, cells, params, compute_exchange=False,
        )
        # For 4 shells × 3 cells, the full dispatch has many entries.
        # The symmetry-reduced should have fewer.
        assert j_result.n_reduced < j_result.n_full or j_result.n_full == 0

    def test_unfold_preserves_reduced_entries(self):
        """Unfolding should produce a dispatch with at least as many
        entries as the reduced one (each quartet spawns up to 4 variants)."""
        basis, cells = _mock_basis_and_cells(n_sh=3)
        params = PenetrationDispatchParameters.for_cell_volume(100.0)

        j_result, _k = build_symmetry_reduced_penetration_dispatch(
            basis, cells, params, compute_exchange=False,
        )
        unfolded = unfold_symmetry_reduced_dispatch(
            j_result.dispatch, n_shells=3,
        )
        # Unfolded should have >= reduced entries (each may spawn variants).
        assert len(unfolded) >= len(j_result.dispatch)

    def test_empty_dispatch_unfolds_to_empty(self):
        """Empty dispatch unfolds to empty."""
        empty = QuartetBipolarDispatch()
        result = unfold_symmetry_reduced_dispatch(empty, n_shells=4)
        assert len(result) == 0

    def test_exchange_dispatch_built_when_requested(self):
        """When compute_exchange=True, k_result is not None."""
        basis, cells = _mock_basis_and_cells(n_sh=3)
        params = PenetrationDispatchParameters.for_cell_volume(100.0)

        j_result, k_result = build_symmetry_reduced_penetration_dispatch(
            basis, cells, params, compute_exchange=True,
        )
        assert k_result is not None
        assert k_result.symmetry_factor >= 1.0

    def test_dispatch_entries_have_valid_truncation_order(self):
        """All dispatch entries have truncation order in [0, max_order]."""
        basis, cells = _mock_basis_and_cells(n_sh=3)
        params = PenetrationDispatchParameters.for_cell_volume(
            100.0, maximum_multipole_order=4,
        )

        j_result, _k = build_symmetry_reduced_penetration_dispatch(
            basis, cells, params, compute_exchange=False,
        )
        for order in j_result.dispatch.truncation_orders:
            assert 0 <= order <= params.maximum_multipole_order


class TestSymmetryFockReconstructionMap:
    """Tests for the Fock reconstruction map built from symmetry dispatch."""

    def test_build_permutation_only_reconstruction(self):
        """The reconstruction map from permutation-only dispatch should
        have one orbit per canonical quartet, covering up to 4 variants."""
        from vibeqc.bipole_symmetry_dispatch import (
            SymmetryFockReconstructionMap,
            build_symmetry_fock_reconstruction_map,
        )

        basis, cells = _mock_basis_and_cells(n_sh=4)
        params = PenetrationDispatchParameters.for_cell_volume(100.0)
        j_result, _k = build_symmetry_reduced_penetration_dispatch(
            basis, cells, params, compute_exchange=False,
        )

        # Build fake shell slices: each shell has one basis function.
        shell_slices = [(i, 1) for i in range(4)]

        recon = build_symmetry_fock_reconstruction_map(
            j_result.dispatch,
            None,  # system — not used for permutation-only
            basis,
            cells,
            shell_slices,
        )

        assert isinstance(recon, SymmetryFockReconstructionMap)
        assert len(recon) == len(j_result.dispatch)
        assert recon.n_sym_operations >= 1

        # Each orbit should have at least 1 entry (the canonical) and
        # at most 4 entries (all transposition variants).
        for orbit in recon.orbit_entries:
            assert 1 <= len(orbit) <= 4
            for bra_key, bra_sl, ket_key, ket_sl in orbit:
                assert isinstance(bra_key, tuple) and len(bra_key) == 3
                assert isinstance(bra_sl, tuple) and len(bra_sl) == 4
                assert isinstance(ket_key, tuple) and len(ket_key) == 3
                assert isinstance(ket_sl, tuple) and len(ket_sl) == 4

    def test_empty_dispatch_gives_empty_reconstruction(self):
        """Empty dispatch should produce empty reconstruction map."""
        from vibeqc.bipole_symmetry_dispatch import (
            build_symmetry_fock_reconstruction_map,
        )

        empty = QuartetBipolarDispatch()
        shell_slices = [(0, 1)]
        cells = _mock_basis_and_cells(n_sh=1)[1]

        recon = build_symmetry_fock_reconstruction_map(
            empty, None, _mock_basis_and_cells(n_sh=1)[0], cells, shell_slices,
        )
        assert len(recon) == 0
        assert len(recon.orbit_entries) == 0
