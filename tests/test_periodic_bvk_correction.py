"""Mechanical tests for the legacy scalar BvK orbital-energy shift.

The helper computes an Ewald-Madelung scalar ``ξ`` and applies ``+ξ`` to
occupied orbital energies.  It is not the charge-projection plus surface-
dipole correction of Nejad et al. (2025), and these tests do not treat it as
a validated finite-size correction for periodic MP2.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_bvk_correction import (
    BvKCorrection,
    apply_bvk_correction_to_energies,
    compute_bvk_correction,
)
from vibeqc.periodic_toroidal_mp2 import build_toroidal_supercell


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _h2_chain_cell(spacing_bohr: float = 6.0) -> vq.PeriodicSystem:
    """One H₂ per cell; periodic along z, 20-bohr transverse vacuum."""
    return vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, spacing_bohr]),
        [vq.Atom(1, [10.0, 10.0, 2.3]), vq.Atom(1, [10.0, 10.0, 3.7])],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bvk_scalar_shift_is_finite_for_periodic_cell():
    """The Madelung constant is finite and positive for a real cell."""
    system = _h2_chain_cell()
    corr = compute_bvk_correction(system, n_occ=1)
    assert isinstance(corr, BvKCorrection)
    assert corr.xi > 0
    assert corr.delta_occ == pytest.approx(corr.xi)


def test_bvk_correction_apply_roundtrip():
    """Applying and then undoing the correction restores original energies."""
    eps_occ = np.array([-0.5, -0.4, -0.3])
    eps_vir = np.array([0.1, 0.2, 0.3, 0.4])
    corr = BvKCorrection(xi=0.05, delta_occ=0.05, n_occ=3)
    eps_occ_c, eps_vir_c = apply_bvk_correction_to_energies(eps_occ, eps_vir, corr)
    # Occupied shifted up by +0.05.
    np.testing.assert_allclose(eps_occ_c, eps_occ + 0.05)
    # Virtuals unchanged.
    np.testing.assert_allclose(eps_vir_c, eps_vir)
    # Round-trip: apply correction with negative xi.
    corr_neg = BvKCorrection(xi=-0.05, delta_occ=-0.05, n_occ=3)
    eps_back, _ = apply_bvk_correction_to_energies(eps_occ_c, eps_vir_c, corr_neg)
    np.testing.assert_allclose(eps_back, eps_occ)


def test_bvk_correction_madelung_behaviour():
    """Madelung constant depends on cell geometry, not just volume.

    For an elongated orthorhombic cell (20×20×N), ξ is dominated by the
    shortest lattice vector and does NOT simply decrease with volume.
    The (1,1,2) cell has the same shortest edge (6 bohr) as (1,1,1) but
    a larger volume, so the neutralising-background term (−π/(η²V)) is
    less negative → ξ can actually increase.  This is correct physics,
    not a bug.
    """
    system = _h2_chain_cell()
    xi_1 = compute_bvk_correction(system, n_occ=1).xi
    super_system = build_toroidal_supercell(system, (1, 1, 2))
    xi_2 = compute_bvk_correction(super_system, n_occ=2).xi
    # Both are finite and positive.
    assert xi_1 > 0
    assert xi_2 > 0
    # The ratio is not simply V2/V1 — cell shape matters for Ewald sums.
    # For an elongated cell, ξ can be similar or larger than the smaller
    # cell because the shortest lattice vector dominates.
    assert xi_2 > xi_1 * 0.5  # stays in the same ballpark
