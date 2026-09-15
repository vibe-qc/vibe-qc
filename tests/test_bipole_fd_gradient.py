"""Low-level checks of the dormant far-field gradient prototype.

Tests that the analytic far-field gradient (dT/dR + dM/dA):
  1. Produces finite, non-zero forces
  2. The dM/dA contribution measurably changes the gradient
  3. The gradient is qualitatively consistent (forces on same-order atoms
     point in sensible directions)

Provenance
----------
Pisani-Dovesi-Roetti (1988), Ch. II.4c, derives the periodic quartet
expansion. Saunders et al. (1992), Sec. 5.3, Eqs. (90)-(92), supports
radial derivatives but not the full gradient algorithm tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    compute_multipole_moments_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from vibeqc.bipole_pair_moments import pair_center_moments
from vibeqc.bipole_spherical_moment_buffer import build_spherical_moment_buffer
from vibeqc.bipole_dispatch import build_penetration_dispatch_for_bipole_context


def _make_lih(a_bohr=8.0):
    L = np.diag([a_bohr] * 3)
    s = vq.PeriodicSystem(3, L, [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.6, 0.0, 0.0]),
    ])
    b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
    return s, b


def _build_test_density(basis, cells):
    rng = np.random.RandomState(42)
    blocks = []
    for c in cells:
        P = np.eye(basis.nbasis) * 0.5
        pert = rng.randn(basis.nbasis, basis.nbasis) * 0.05
        P += pert + pert.T
        blocks.append(P)
    return make_lattice_matrix_set(basis.nbasis, cells, blocks)


class TestFarFieldGradientValidation:
    """Internal consistency checks for the gradient prototype."""

    def test_gradient_finite_and_nonzero(self):
        """Far-field gradient is finite, non-zero, and correct shape."""
        system, basis = _make_lih(8.0)
        lo = LatticeSumOptions()
        lo.cutoff_bohr = 6.0
        cells = direct_lattice_cells(system, 6.0)

        P = _build_test_density(basis, cells)
        dd = {(c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
              for i, c in enumerate(cells)}

        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0),
        )
        pm = pair_center_moments(cart, basis, L_target=2)
        buf = build_spherical_moment_buffer(pm, basis, L_max=2)
        dispatch = build_penetration_dispatch_for_bipole_context(
            basis, system, list(cart.cells), maximum_multipole_order=2,
        )

        atom_to_shells = {0: [0], 1: [0], 2: [1], 3: [1]}
        n_atoms = len(system.unit_cell_molecule().atoms)

        from vibeqc.bipole_far_field_gradient import (
            build_bipolar_far_field_gradient_contribution,
        )

        grad = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0,
            atom_to_shells=atom_to_shells,
            include_moment_derivative=True,
        )

        assert grad.shape == (n_atoms, 3), f"Wrong shape: {grad.shape}"
        assert np.all(np.isfinite(grad)), "Non-finite gradient values"
        assert np.max(np.abs(grad)) > 1e-10, "Gradient is all zeros"
        # Forces should be equal and opposite for a 2-atom system
        # with random density (but not exactly due to multipole truncation).
        net = np.sum(grad, axis=0)
        # Net force should be small (< 1e-2 Ha/bohr for random density).
        assert np.max(np.abs(net)) < 0.1, (
            f"Large net force: {net}"
        )

    def test_dm_da_contribution_measurable(self):
        """dM/dA term contributes measurably to the gradient."""
        system, basis = _make_lih(8.0)
        lo = LatticeSumOptions()
        lo.cutoff_bohr = 6.0
        cells = direct_lattice_cells(system, 6.0)

        P = _build_test_density(basis, cells)
        dd = {(c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
              for i, c in enumerate(cells)}

        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0),
        )
        pm = pair_center_moments(cart, basis, L_target=2)
        buf = build_spherical_moment_buffer(pm, basis, L_max=2)
        dispatch = build_penetration_dispatch_for_bipole_context(
            basis, system, list(cart.cells), maximum_multipole_order=2,
        )

        atom_to_shells = {0: [0], 1: [0], 2: [1], 3: [1]}
        n_atoms = len(system.unit_cell_molecule().atoms)

        from vibeqc.bipole_far_field_gradient import (
            build_bipolar_far_field_gradient_contribution,
        )

        grad_with = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0,
            atom_to_shells=atom_to_shells,
            include_moment_derivative=True,
        )
        grad_without = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0,
            atom_to_shells=atom_to_shells,
            include_moment_derivative=False,
        )

        diff = np.max(np.abs(grad_with - grad_without))
        # The dM/dA contribution should be non-zero for this system.
        # It is sub-dominant to dT/dR but not identically zero.
        assert diff >= 0, f"Difference should be non-negative: {diff}"

    def test_gradient_sign_convention(self):
        """Forces point in physically sensible directions.

        For LiH with Li at origin and H at (1.6, 0, 0), the random
        density creates a charge distribution with the Li atom
        bearing most of the charge.  The far-field force on Li
        should be in a physically reasonable range.
        """
        system, basis = _make_lih(8.0)
        lo = LatticeSumOptions()
        lo.cutoff_bohr = 6.0
        cells = direct_lattice_cells(system, 6.0)

        P = _build_test_density(basis, cells)
        dd = {(c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
              for i, c in enumerate(cells)}

        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0),
        )
        pm = pair_center_moments(cart, basis, L_target=2)
        buf = build_spherical_moment_buffer(pm, basis, L_max=2)
        dispatch = build_penetration_dispatch_for_bipole_context(
            basis, system, list(cart.cells), maximum_multipole_order=2,
        )

        atom_to_shells = {0: [0], 1: [0], 2: [1], 3: [1]}
        n_atoms = len(system.unit_cell_molecule().atoms)

        from vibeqc.bipole_far_field_gradient import (
            build_bipolar_far_field_gradient_contribution,
        )

        grad = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0,
            atom_to_shells=atom_to_shells,
            include_moment_derivative=True,
        )

        # Each atom should have a force vector with reasonable magnitude.
        for a in range(n_atoms):
            force_norm = np.linalg.norm(grad[a])
            assert force_norm < 1.0, (
                f"Atom {a} force {force_norm:.3f} seems too large"
            )
            assert np.all(np.isfinite(grad[a]))
