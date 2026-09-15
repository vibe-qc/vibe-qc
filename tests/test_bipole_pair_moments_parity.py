"""Parity tests: C++ pair-centre moment shift vs Python reference.

Verifies shift_multipole_moments_to_pair_centres matches the Python
pair_center_moments to machine precision for L=2,3.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import compute_multipole_moments_lattice, LatticeSumOptions, shift_multipole_moments_to_pair_centres
from vibeqc.bipole_pair_moments import pair_center_moments as py_shift

ANG2BOHR = 1.0 / 0.529177210903


def _make_mgo(a_bohr: float):
    lattice = (a_bohr / 2.0) * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    atoms = [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a_bohr/2]*3)]
    return vq.PeriodicSystem(3, lattice, atoms)


class TestPairMomentsCppParity:
    def test_mgo_sto3g_l2_parity(self):
        """L=2: C++ matches Python to machine precision."""
        a = 4.213 * ANG2BOHR
        system = _make_mgo(a)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        lo = LatticeSumOptions(); lo.cutoff_bohr = 5.0
        cart = compute_multipole_moments_lattice(basis, system, lo, 2, (0,0,0))
        py = py_shift(cart, basis, L_target=2)
        cpp = shift_multipole_moments_to_pair_centres(cart, basis, 2, (0,0,0))
        md = max(np.max(np.abs(np.asarray(py.blocks[c][comp]) - np.asarray(cpp.blocks[c][comp])))
                 for c in range(len(py.cells)) for comp in range(10))
        assert md < 1e-12, f"L=2 max diff: {md:.2e}"

    def test_mgo_sto3g_l3_parity(self):
        """L=3: C++ matches Python to machine precision."""
        a = 4.213 * ANG2BOHR
        system = _make_mgo(a)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        lo = LatticeSumOptions(); lo.cutoff_bohr = 5.0
        cart = compute_multipole_moments_lattice(basis, system, lo, 3, (0,0,0))
        py = py_shift(cart, basis, L_target=3)
        cpp = shift_multipole_moments_to_pair_centres(cart, basis, 3, (0,0,0))
        md = max(np.max(np.abs(np.asarray(py.blocks[c][comp]) - np.asarray(cpp.blocks[c][comp])))
                 for c in range(len(py.cells)) for comp in range(20))
        assert md < 1e-12, f"L=3 max diff: {md:.2e}"

    def test_mgo_sto3g_l4_parity(self):
        """L=4: the C++ hexadecapole shift block matches Python on a
        full 35-component Cartesian set with nonzero degree-4 content.

        The 35-component input comes from the C++ isotropic L=4
        extension (extend_lattice_moments_to_L4) -- for parity purposes
        the physical meaning of the degree-4 values is irrelevant; both
        shifts receive the identical set and must agree on every
        component. This is the only lane that exercises the compiled
        hexadecapole block (production gates the native shift to L<=3
        and runs the merged L=4 route through the Python shift)."""
        from vibeqc._vibeqc_core import extend_lattice_moments_to_L4

        a = 4.213 * ANG2BOHR
        system = _make_mgo(a)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        lo = LatticeSumOptions(); lo.cutoff_bohr = 5.0
        cart3 = compute_multipole_moments_lattice(basis, system, lo, 3, (0,0,0))
        ext4 = extend_lattice_moments_to_L4(cart3, basis)
        assert int(ext4.L_max) == 4 and not ext4.spherical
        # The extension must supply nonzero degree-4 content, or this
        # parity test would pass vacuously.
        deg4_norm = max(
            np.max(np.abs(np.asarray(ext4.blocks[c][comp])))
            for c in range(len(ext4.cells)) for comp in range(20, 35)
        )
        assert deg4_norm > 1e-8, "no degree-4 content in extended set"
        py = py_shift(ext4, basis, L_target=4)
        cpp = shift_multipole_moments_to_pair_centres(ext4, basis, 4, (0,0,0))
        md = max(np.max(np.abs(np.asarray(py.blocks[c][comp]) - np.asarray(cpp.blocks[c][comp])))
                 for c in range(len(py.cells)) for comp in range(35))
        assert md < 1e-12, f"L=4 max diff: {md:.2e}"
