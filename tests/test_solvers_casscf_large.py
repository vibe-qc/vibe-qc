"""Larger active-space CASSCF robustness tests.

Covers N2 CAS(6,6) to ensure the optimizer handles many rotation pairs.
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

N2 = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.07])])
H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])


def _ham(mol, basis_name):
    b = BasisSet(mol, basis_name)
    c = get_hf_orbital_provider(mol, b, method="rhf")
    return build_hamiltonian_mo(mol, b, c)


class TestLargeActiveSpace:
    """CASSCF with N2 CAS(6,6) — many rotation pairs."""

    def test_n2_cas66_converges(self):
        """N2 CAS(6,6) — 6 electrons in (sigma, pi, pi*) active space.
        Uses NR mode for reliability with near-degenerate orbitals."""
        H = _ham(N2, "sto-3g")
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=6,
            n_active_orb=6,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
            orbital_step="nr",
            max_macro=50,
        )
        assert res.converged and res.grad_norm < 1e-6

    def test_n2_cas66_vs_casci(self):
        """CASSCF improves on CASCI for N2 CAS(6,6)."""
        H = _ham(N2, "sto-3g")
        e_casci = casci(
            H.h1e,
            H.h2e,
            n_active_elec=6,
            n_active_orb=6,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=6,
            n_active_orb=6,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
            orbital_step="nr",
            max_macro=50,
        )
        assert res.converged
        assert res.e_total <= e_casci + 1e-9  # CASSCF optimises, can't be worse

    def test_full_active_space_is_fci(self):
        """CAS(full space) = FCI for H2/6-31G."""
        H = _ham(H2, "6-31g")
        norb = H.norb
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=norb,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        fci = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=norb,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        assert abs(res.e_total - fci) < 1e-9
