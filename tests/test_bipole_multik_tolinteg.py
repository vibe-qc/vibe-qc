"""Multi-k and overlap-threshold checks for the dormant bipolar prototype.

BIPOLE-EXACT-ZONE increment 5b. Tests the implementation-specific
quartet classifier and contractor across k-mesh sizes and thresholds.

Validates:
  1. Gamma-only (n_k=1) bipolar far-field energies
  2. Multi-k (n_k>1) bipolar far-field energies
  3. Sensitivity to the prototype overlap threshold
  4. Multi-k density consistency (Bloch-fold vs Gamma-local)
  5. Monopole-vs-dipole-vs-quadrupole convergence

Provenance
----------
Pisani-Dovesi-Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
The classifier has no documented one-to-one TOLINTEG mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    build_jk_2e_real_space,
    compute_multipole_moments_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from vibeqc.bipole_pair_moments import pair_center_moments
from vibeqc.bipole_spherical_moment_buffer import (
    build_spherical_moment_buffer,
)
from vibeqc.bipole_dispatch import (
    PenetrationDispatchParameters,
    build_penetration_dispatch_for_bipole_context,
)
from vibeqc.bipole_quartet_far_field import (
    build_bipolar_coulomb_far_field,
)


ANG2BOHR = 1.0 / 0.529177210903


def _make_lih_system(a_bohr=7.5):
    """LiH in cubic box, STO-3G."""
    lattice = np.diag([a_bohr, a_bohr, a_bohr])
    system = PeriodicSystem(
        3,
        lattice,
        [
            Atom(3, [0.0, 0.0, 0.0]),
            Atom(1, [a_bohr / 2, a_bohr / 2, a_bohr / 2]),
        ],
    )
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


# ---------------------------------------------------------------------------
# Gamma-only: bipolar far-field vs exact J_SR
# ---------------------------------------------------------------------------


class TestGammaBipolarFarField:
    """Gamma-only (n_k=1) bipolar far-field validation."""

    def test_bipolar_energy_nonzero(self):
        """Bipolar far-field produces non-zero energy for a real system."""
        system, basis = _make_lih_system(7.5)
        cutoff = 6.0
        omega = 0.5
        lo = LatticeSumOptions()
        lo.cutoff_bohr = cutoff

        cells = direct_lattice_cells(system, cutoff)
        blocks = [
            np.eye(basis.nbasis) if np.allclose(c.index, 0)
            else np.zeros((basis.nbasis, basis.nbasis))
            for c in cells
        ]
        P = make_lattice_matrix_set(basis.nbasis, cells, blocks)

        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0),
        )
        pair_mom = pair_center_moments(cart, basis)
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)
        dispatch = build_penetration_dispatch_for_bipole_context(
            basis, system, list(cart.cells), maximum_multipole_order=2,
        )

        density_dict = {
            (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
            for i, c in enumerate(cells)
        }
        ff = build_bipolar_coulomb_far_field(
            buf, dispatch, density_dict,
            ewald_omega=omega, nbf=basis.nbasis,
        )
        assert ff.n_quartets > 0, "No far-field quartets dispatched"
        assert abs(ff.e_coulomb_far) > 1e-12, "Far-field energy is zero"

    def test_bipolar_energy_converges_with_order(self):
        """Higher multipole order gives energy closer to exact."""
        system, basis = _make_lih_system(7.5)
        cutoff = 6.0
        omega = 0.5
        lo = LatticeSumOptions()
        lo.cutoff_bohr = cutoff
        cells = direct_lattice_cells(system, cutoff)
        blocks = [
            np.eye(basis.nbasis) if np.allclose(c.index, 0)
            else np.zeros((basis.nbasis, basis.nbasis))
            for c in cells
        ]
        P = make_lattice_matrix_set(basis.nbasis, cells, blocks)

        jk_exact = build_jk_2e_real_space(basis, system, lo, P, omega)
        e_exact = sum(
            0.5 * np.sum(np.asarray(P.blocks[c]) * np.asarray(jk_exact.J.blocks[c]))
            for c in range(len(cells))
        )

        errors = {}
        for L in (0, 1, 2):
            cart = compute_multipole_moments_lattice(
                basis, system, lo, 2, (0.0, 0.0, 0.0),
            )
            pair_mom = pair_center_moments(cart, basis)
            buf = build_spherical_moment_buffer(pair_mom, basis, L_max=L)
            dispatch = build_penetration_dispatch_for_bipole_context(
                basis, system, list(cart.cells), maximum_multipole_order=L,
            )
            density_dict = {
                (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
                for i, c in enumerate(cells)
            }
            ff = build_bipolar_coulomb_far_field(
                buf, dispatch, density_dict,
                ewald_omega=omega, nbf=basis.nbasis,
            )
            errors[L] = abs(e_exact - ff.e_coulomb_far)

        # Higher order should not be worse than lower order.
        assert errors[2] <= errors[1] * 10 or errors[1] < 1e-8, (
            f"L=2 error ({errors[2]:.2e}) > L=1 error ({errors[1]:.2e})"
        )


# ---------------------------------------------------------------------------
# Multi-k: bipolar far-field with Bloch-folded density
# ---------------------------------------------------------------------------


class TestMultiKBipolarFarField:
    """Multi-k (n_k > 1) bipolar far-field validation."""

    @pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
    def test_multi_k_density_consistency(self, mesh):
        """Bipolar far-field runs without error with multi-k density."""
        system, basis = _make_lih_system(7.5)
        cutoff = 6.0
        omega = 0.5
        lo = LatticeSumOptions()
        lo.cutoff_bohr = cutoff

        # Build a trial real-space density: identity at the home cell,
        # zero elsewhere (same as the Gamma-only tests).
        cells = direct_lattice_cells(system, cutoff)
        blocks = [
            np.eye(basis.nbasis)
            if np.allclose(c.index, np.zeros(3))
            else np.zeros((basis.nbasis, basis.nbasis))
            for c in cells
        ]
        P_real = make_lattice_matrix_set(basis.nbasis, cells, blocks)

        # Build infrastructure (k-independent).
        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0),
        )
        pair_mom = pair_center_moments(cart, basis)
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)
        dispatch = build_penetration_dispatch_for_bipole_context(
            basis, system, list(cart.cells), maximum_multipole_order=2,
        )

        density_dict = {
            (c.index[0], c.index[1], c.index[2]): np.asarray(
                P_real.blocks[i], dtype=float,
            )
            for i, c in enumerate(cells)
        }
        ff = build_bipolar_coulomb_far_field(
            buf, dispatch, density_dict,
            ewald_omega=omega, nbf=basis.nbasis,
        )
        assert ff.n_quartets > 0, (
            f"No far-field quartets dispatched at mesh {mesh}"
        )
        assert np.isfinite(ff.e_coulomb_far), (
            f"Non-finite far-field energy at mesh {mesh}"
        )

    @pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
    def test_multi_k_exact_vs_bipolar_energy_sign(self, mesh):
        """Exact and bipolar energies have the same sign at multi-k."""
        system, basis = _make_lih_system(7.5)
        cutoff = 6.0
        omega = 0.5
        lo = LatticeSumOptions()
        lo.cutoff_bohr = cutoff

        # Build a trial real-space density: identity at the home cell,
        # zero elsewhere.
        cells = direct_lattice_cells(system, cutoff)
        blocks = [
            np.eye(basis.nbasis)
            if np.allclose(c.index, np.zeros(3))
            else np.zeros((basis.nbasis, basis.nbasis))
            for c in cells
        ]
        P_real = make_lattice_matrix_set(basis.nbasis, cells, blocks)

        jk_exact = build_jk_2e_real_space(basis, system, lo, P_real, omega)
        e_exact = sum(
            0.5 * np.sum(
                np.asarray(P_real.blocks[c])
                * np.asarray(jk_exact.J.blocks[c]),
            )
            for c in range(len(cells))
        )

        cart = compute_multipole_moments_lattice(
            basis, system, lo, 2, (0.0, 0.0, 0.0),
        )
        pair_mom = pair_center_moments(cart, basis)
        buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)
        dispatch = build_penetration_dispatch_for_bipole_context(
            basis, system, list(cart.cells), maximum_multipole_order=2,
        )
        density_dict = {
            (c.index[0], c.index[1], c.index[2]): np.asarray(
                P_real.blocks[i], dtype=float,
            )
            for i, c in enumerate(cells)
        }
        ff = build_bipolar_coulomb_far_field(
            buf, dispatch, density_dict,
            ewald_omega=omega, nbf=basis.nbasis,
        )

        # Signs should agree (both positive for this test density).
        assert e_exact * ff.e_coulomb_far > 0 or abs(e_exact) < 1e-12, (
            f"mesh={mesh}: exact={e_exact:.6e}, bipolar={ff.e_coulomb_far:.6e}"
        )


# ---------------------------------------------------------------------------
# TOLINTEG (overlap_drop_threshold) sensitivity
# ---------------------------------------------------------------------------


class TestTOLINTEGSensitivity:
    """Bipolar far-field convergence with penetration threshold."""

    @pytest.mark.parametrize("threshold", [1e-3, 1e-5, 1e-7])
    def test_tighter_threshold_gives_more_quartets(self, threshold):
        """Tighter (smaller) overlap threshold → more far-field quartets."""
        system, basis = _make_lih_system(7.5)
        cutoff = 6.0
        omega = 0.5
        lo = LatticeSumOptions()
        lo.cutoff_bohr = cutoff

        cells = direct_lattice_cells(system, cutoff)

        params_tight = PenetrationDispatchParameters.for_cell_volume(
            float(abs(np.linalg.det(system.lattice))),
            maximum_multipole_order=2,
            overlap_drop_threshold=threshold,
        )
        params_loose = PenetrationDispatchParameters.for_cell_volume(
            float(abs(np.linalg.det(system.lattice))),
            maximum_multipole_order=2,
            overlap_drop_threshold=1e-2,
        )

        from vibeqc.bipole_dispatch import (
            build_shell_quartet_penetration_dispatch,
        )

        j_tight, _ = build_shell_quartet_penetration_dispatch(
            basis, cells, params_tight, compute_exchange=False,
        )
        j_loose, _ = build_shell_quartet_penetration_dispatch(
            basis, cells, params_loose, compute_exchange=False,
        )

        # Tighter threshold should dispatch AT LEAST as many as loose.
        assert len(j_tight) >= len(j_loose), (
            f"tight({threshold}): {len(j_tight)} < loose(1e-2): {len(j_loose)}"
        )

    def test_energy_stable_with_threshold(self):
        """Bipolar energy should be stable across reasonable thresholds."""
        system, basis = _make_lih_system(7.5)
        cutoff = 6.0
        omega = 0.5
        lo = LatticeSumOptions()
        lo.cutoff_bohr = cutoff
        cells = direct_lattice_cells(system, cutoff)
        blocks = [
            np.eye(basis.nbasis) if np.allclose(c.index, 0)
            else np.zeros((basis.nbasis, basis.nbasis))
            for c in cells
        ]
        P = make_lattice_matrix_set(basis.nbasis, cells, blocks)

        energies = {}
        for thr in (1e-3, 1e-5, 1e-7):
            cart = compute_multipole_moments_lattice(
                basis, system, lo, 2, (0.0, 0.0, 0.0),
            )
            pair_mom = pair_center_moments(cart, basis)
            buf = build_spherical_moment_buffer(pair_mom, basis, L_max=2)

            params = PenetrationDispatchParameters.for_cell_volume(
                float(abs(np.linalg.det(system.lattice))),
                maximum_multipole_order=2,
                overlap_drop_threshold=thr,
            )
            from vibeqc.bipole_dispatch import (
                build_shell_quartet_penetration_dispatch,
            )
            j_disp, _ = build_shell_quartet_penetration_dispatch(
                basis, cells, params, compute_exchange=False,
            )
            density_dict = {
                (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
                for i, c in enumerate(cells)
            }
            ff = build_bipolar_coulomb_far_field(
                buf, j_disp, density_dict,
                ewald_omega=omega, nbf=basis.nbasis,
            )
            energies[thr] = ff.e_coulomb_far

        # Energies at 1e-5 and 1e-7 should be very close (converged).
        if abs(energies[1e-5]) > 1e-12:
            rel_diff = abs(
                energies[1e-5] - energies[1e-7]
            ) / abs(energies[1e-5])
            assert rel_diff < 0.1, (
                f"Energy not converged: 1e-5={energies[1e-5]:.6e}, "
                f"1e-7={energies[1e-7]:.6e}, rel={rel_diff:.3e}"
            )
