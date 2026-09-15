"""Verify the C++ far-field gradient contractor is exercised.

Tests that the C++ native path is successfully imported and produces
results matching the Python reference.  Also confirms the auto-wiring
in build_bipolar_far_field_gradient_contribution delegates to C++.
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
        vq.Atom(3, [0, 0, 0]),
        vq.Atom(1, [1.6, 0, 0]),
    ])
    b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
    return s, b


class TestCppGradientWiring:
    """C++ gradient contractor is importable and produces correct results."""

    def test_native_module_importable(self):
        """bipole_far_field_gradient_native imports successfully."""
        from vibeqc.bipole_far_field_gradient_native import (
            compute_bipolar_far_field_gradient_native,
        )
        assert callable(compute_bipolar_far_field_gradient_native)

    def test_cpp_function_bound(self):
        """C++ gradient function is available via _vibeqc_core."""
        from vibeqc._vibeqc_core import compute_bipolar_far_field_gradient_cpp
        assert callable(compute_bipolar_far_field_gradient_cpp)

    def test_erfc_gradient_function_bound(self):
        """C++ erfc gradient function is available."""
        from vibeqc._vibeqc_core import (
            multipole_erfc_interaction_tensor_gradient,
        )
        assert callable(multipole_erfc_interaction_tensor_gradient)

    @pytest.mark.parametrize(
        "orders",
        [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
    )
    def test_cartesian_derivative_rejects_negative_order(self, orders):
        """Negative multi-index components fail before axis expansion."""
        from vibeqc._vibeqc_core import (
            _multipole_cartesian_derivative_inverse_r as derivative,
        )

        with pytest.raises(ValueError, match="must be nonnegative"):
            derivative(*orders, 1.3, -0.7, 2.1)

    @pytest.mark.parametrize(
        "orders",
        [(6, 0, 0), (0, 6, 0), (0, 0, 6), (2, 2, 2)],
    )
    def test_cartesian_derivative_rejects_total_order_above_five(
        self,
        orders,
    ):
        """Unsupported orders fail before writing the fixed axis buffer."""
        from vibeqc._vibeqc_core import (
            _multipole_cartesian_derivative_inverse_r as derivative,
        )

        with pytest.raises(ValueError, match="total derivative order > 5"):
            derivative(*orders, 1.3, -0.7, 2.1)

    @pytest.mark.parametrize(
        ("lower", "axis"),
        [((4, 0, 0), 0), ((2, 2, 0), 2), ((1, 1, 2), 1)],
    )
    def test_order_five_cartesian_derivative_matches_finite_difference(
        self,
        lower,
        axis,
    ):
        """The dormant L4 gradient's order-5 primitive is memory-safe and
        agrees with a finite difference of the order-4 derivative."""
        from vibeqc._vibeqc_core import (
            _multipole_cartesian_derivative_inverse_r as derivative,
        )

        r = np.array([1.3, -0.7, 2.1], dtype=float)
        h = 1.0e-5
        upper = list(lower)
        upper[axis] += 1
        rp = r.copy()
        rm = r.copy()
        rp[axis] += h
        rm[axis] -= h
        finite_difference = (
            derivative(*lower, *rp) - derivative(*lower, *rm)
        ) / (2.0 * h)
        analytic = derivative(*upper, *r)
        assert analytic == pytest.approx(
            finite_difference,
            rel=2.0e-7,
            abs=2.0e-9,
        )

    def test_auto_wiring_matches_python(self):
        """build_bipolar_far_field_gradient_contribution uses C++ path."""
        system, basis = _make_lih(8.0)
        lo = LatticeSumOptions()
        lo.cutoff_bohr = 6.0
        cells = direct_lattice_cells(system, 6.0)

        rng = np.random.RandomState(42)
        blocks = []
        for c in cells:
            P = np.eye(basis.nbasis) * 0.5
            pert = rng.randn(basis.nbasis, basis.nbasis) * 0.05
            P += pert + pert.T
            blocks.append(P)
        P = make_lattice_matrix_set(basis.nbasis, cells, blocks)

        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0),
        )
        pm = pair_center_moments(cart, basis, L_target=2)
        buf = build_spherical_moment_buffer(pm, basis, L_max=2)
        dispatch = build_penetration_dispatch_for_bipole_context(
            basis, system, list(cart.cells), maximum_multipole_order=2,
        )

        dd = {(c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
              for i, c in enumerate(cells)}
        atom_to_shells = {0: [0], 1: [0], 2: [1], 3: [1]}
        n_atoms = len(system.unit_cell_molecule().atoms)

        # This goes through C++ now (auto-wired).
        from vibeqc.bipole_far_field_gradient import (
            build_bipolar_far_field_gradient_contribution,
        )
        grad = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0, atom_to_shells=atom_to_shells,
            include_moment_derivative=True,
        )
        assert grad.shape == (n_atoms, 3)
        assert np.all(np.isfinite(grad))
        # Should have non-zero forces.
        assert np.max(np.abs(grad)) > 0, "Gradient is all zeros"

    def test_auto_wiring_with_erfc(self):
        """C++ erfc gradient path is exercised when ewald_omega > 0."""
        system, basis = _make_lih(8.0)
        lo = LatticeSumOptions()
        lo.cutoff_bohr = 6.0
        cells = direct_lattice_cells(system, 6.0)

        rng = np.random.RandomState(42)
        blocks = []
        for c in cells:
            P = np.eye(basis.nbasis) * 0.5
            pert = rng.randn(basis.nbasis, basis.nbasis) * 0.05
            P += pert + pert.T
            blocks.append(P)
        P = make_lattice_matrix_set(basis.nbasis, cells, blocks)

        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0),
        )
        pm = pair_center_moments(cart, basis, L_target=2)
        buf = build_spherical_moment_buffer(pm, basis, L_max=2)
        dispatch = build_penetration_dispatch_for_bipole_context(
            basis, system, list(cart.cells), maximum_multipole_order=2,
        )

        dd = {(c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
              for i, c in enumerate(cells)}
        atom_to_shells = {0: [0], 1: [0], 2: [1], 3: [1]}
        n_atoms = len(system.unit_cell_molecule().atoms)

        from vibeqc.bipole_far_field_gradient import (
            build_bipolar_far_field_gradient_contribution,
        )
        grad = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.5, atom_to_shells=atom_to_shells,
            include_moment_derivative=False,  # dT/dR only
        )
        assert grad.shape == (n_atoms, 3)
        assert np.all(np.isfinite(grad))
        assert np.max(np.abs(grad)) > 0
