"""Parity tests: C++ far-field gradient contractor vs Python reference.

Verifies that the C++ OpenMP gradient contractor produces matching
nuclear forces to the Python reference implementation.
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
from vibeqc.bipole_far_field_gradient_native import (
    compute_bipolar_far_field_gradient_native,
)
from vibeqc.bipole_far_field_gradient import (
    build_bipolar_far_field_gradient_contribution,
)

ANG2BOHR = 1.0 / 0.529177210903


def _make_mgo(a_bohr: float = 4.213 * 1.889725989):
    lattice = (a_bohr / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, [a_bohr / 2.0, a_bohr / 2.0, a_bohr / 2.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _make_lih(a_bohr: float = 8.0):
    lattice = np.diag([a_bohr, a_bohr, a_bohr])
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.6, 0.0, 0.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _setup(system, basis, cutoff, L_max):
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    cells = direct_lattice_cells(system, cutoff)

    cart = compute_multipole_moments_lattice(
        basis, system, lo, min(L_max, 3), (0.0, 0.0, 0.0),
    )
    pair_mom = pair_center_moments(cart, basis, L_target=min(L_max, 3))
    buf = build_spherical_moment_buffer(pair_mom, basis, L_max=L_max)
    dispatch = build_penetration_dispatch_for_bipole_context(
        basis, system, list(cart.cells), maximum_multipole_order=L_max,
    )
    return buf, dispatch, cells


def _build_density(basis, cells):
    rng = np.random.RandomState(42)
    blocks = []
    for c in cells:
        P = np.eye(basis.nbasis) * 0.5
        pert = rng.randn(basis.nbasis, basis.nbasis) * 0.05
        P += pert + pert.T
        blocks.append(P)
    return make_lattice_matrix_set(basis.nbasis, cells, blocks)


def _shell_to_atom_mgo(n_sh):
    return {sh: [0] if sh < n_sh // 2 else [1] for sh in range(n_sh)}


def _shell_to_atom_lih(n_sh):
    return {sh: [0] if sh < 2 else [1] for sh in range(n_sh)}


class TestCppGradientParity:
    """C++ far-field gradient matches Python to machine precision."""

    def test_lih_sto3g_bare_coulomb_gradient_L2(self):
        """LiH STO-3G L=2: C++ bare Coulomb gradient matches Python."""
        system, basis = _make_lih(8.0)
        buf, dispatch, cells = _setup(system, basis, 6.0, L_max=2)

        P = _build_density(basis, cells)
        dd = {(c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
              for i, c in enumerate(cells)}

        atom_to_shells = _shell_to_atom_lih(len(buf.shell_slices))
        n_atoms = len(system.unit_cell_molecule().atoms)

        grad_cpp = compute_bipolar_far_field_gradient_native(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0, atom_to_shells=atom_to_shells,
            include_moment_derivative=True,
        )
        grad_py = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0, atom_to_shells=atom_to_shells,
            include_moment_derivative=True,
        )

        max_diff = np.max(np.abs(grad_cpp - grad_py))
        assert max_diff < 1e-12, (
            f"Gradient mismatch: max diff = {max_diff:.3e}"
        )
        assert np.all(np.isfinite(grad_cpp))
        assert np.all(np.isfinite(grad_py))

    def test_mgo_sto3g_bare_coulomb_gradient_L2(self):
        """MgO STO-3G L=2: C++ bare Coulomb gradient matches Python."""
        system, basis = _make_mgo()
        buf, dispatch, cells = _setup(system, basis, 6.0, L_max=2)

        P = _build_density(basis, cells)
        dd = {(c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
              for i, c in enumerate(cells)}

        atom_to_shells = _shell_to_atom_mgo(len(buf.shell_slices))
        n_atoms = len(system.unit_cell_molecule().atoms)

        grad_cpp = compute_bipolar_far_field_gradient_native(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0, atom_to_shells=atom_to_shells,
            include_moment_derivative=True,
        )
        grad_py = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=0.0, atom_to_shells=atom_to_shells,
            include_moment_derivative=True,
        )

        max_diff = np.max(np.abs(grad_cpp - grad_py))
        assert max_diff < 1e-12, (
            f"Gradient mismatch: max diff = {max_diff:.3e}"
        )


    def test_lih_sto3g_erfc_screened_gradient_L2(self):
        """LiH STO-3G L=2: C++ erfc-screened gradient matches Python."""
        system, basis = _make_lih(8.0)
        buf, dispatch, cells = _setup(system, basis, 6.0, L_max=2)

        P = _build_density(basis, cells)
        dd = {(c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
              for i, c in enumerate(cells)}

        atom_to_shells = _shell_to_atom_lih(len(buf.shell_slices))
        n_atoms = len(system.unit_cell_molecule().atoms)

        omega = 0.5
        grad_cpp = compute_bipolar_far_field_gradient_native(
            buf, dispatch, dd, n_atoms,
            ewald_omega=omega, atom_to_shells=atom_to_shells,
            include_moment_derivative=False,  # dT/dR only for erfc parity
        )
        grad_py = build_bipolar_far_field_gradient_contribution(
            buf, dispatch, dd, n_atoms,
            ewald_omega=omega, atom_to_shells=atom_to_shells,
            include_moment_derivative=False,
        )

        max_diff = np.max(np.abs(grad_cpp - grad_py))
        assert max_diff < 1e-10, (
            f"Erfc gradient mismatch: max diff = {max_diff:.3e}"
        )
