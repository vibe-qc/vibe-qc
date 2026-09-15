"""CAS-window selection tests — active_orbitals parameter.

Tests that the active space can be placed at an arbitrary orbital
index range, not only the lowest n_active after the core.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    build_hamiltonian_mo,
    casci,
    casscf,
    get_hf_orbital_provider,
)

H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)


def _ham(mol, basis_name):
    b = BasisSet(mol, basis_name)
    c = get_hf_orbital_provider(mol, b, method="rhf")
    return build_hamiltonian_mo(mol, b, c)


class TestCASWindow:
    """active_orbitals parameter for explicit active-space selection."""

    def test_casci_active_orbitals_equivalent(self):
        """Specifying active_orbitals=[2,3] with n_core=2 is the same as
        n_core=2, n_active_orb=2 without active_orbitals."""
        H = _ham(H2O, "sto-3g")
        e_default = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=2,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        e_window = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=2,
            nuclear_repulsion=H.nuclear_repulsion,
            active_orbitals=[2, 3],
        ).e_total
        assert abs(e_default - e_window) < 1e-12

    def test_casci_offset_window(self):
        """CASCI with active_orbitals=[4,5] (n_core=4 implied) same energy
        as n_core=4, n_active_orb=2 with default orbital ordering."""
        H = _ham(H2O, "sto-3g")
        e_default = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        e_window = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            active_orbitals=[4, 5],
        ).e_total
        assert abs(e_default - e_window) < 1e-12

    def test_cascf_active_orbitals(self):
        """CASSCF with active_orbitals works and converges."""
        H = _ham(H2O, "sto-3g")
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        res_win = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            active_orbitals=[4, 5],
        )
        assert res_win.converged and res_win.grad_norm < 1e-6
        assert abs(res.e_total - res_win.e_total) < 1e-12

    def test_noncontiguous_raises(self):
        """Non-contiguous active_orbitals raises ValueError."""
        H = _ham(H2O, "sto-3g")
        with pytest.raises(ValueError, match="contiguous"):
            casci(
                H.h1e,
                H.h2e,
                n_active_elec=2,
                n_active_orb=2,
                nuclear_repulsion=H.nuclear_repulsion,
                active_orbitals=[2, 5],  # gap
            )

    def test_empty_active_orbitals_raises(self):
        """Empty active_orbitals raises ValueError."""
        H = _ham(H2O, "sto-3g")
        with pytest.raises(ValueError, match="must not be empty"):
            casci(
                H.h1e,
                H.h2e,
                n_active_elec=2,
                n_active_orb=2,
                nuclear_repulsion=H.nuclear_repulsion,
                active_orbitals=[],
            )
