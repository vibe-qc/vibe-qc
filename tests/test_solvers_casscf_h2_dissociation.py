"""H2 dissociation curve with CASSCF — the canonical test case.

A restricted Hartree-Fock calculation dissociates H2 incorrectly (it
converges to the ionic limit, not the neutral atoms).  CASSCF(2,2)
with a (σ, σ*) active space recovers the correct dissociation limit
because the two-configuration CI captures the static correlation.
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


def _h2_energy(r, basis="sto-3g", method="casscf"):
    """Single-point energy of H2 at bond length r (Bohr)."""
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, r])])
    b = BasisSet(mol, basis)
    C = get_hf_orbital_provider(mol, b, method="rhf")
    H = build_hamiltonian_mo(mol, b, C)
    if method == "casci":
        return casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
    return casscf(
        H.h1e,
        H.h2e,
        n_active_elec=2,
        n_active_orb=2,
        nuclear_repulsion=H.nuclear_repulsion,
    ).e_total


class TestH2Dissociation:
    """H2 dissociation — CASSCF(2,2) must describe bond-breaking correctly."""

    def test_equilibrium_energy(self):
        """CASSCF at equilibrium (~1.4 Bohr) gives reasonable energy."""
        e = _h2_energy(1.4, method="casscf")
        assert -1.2 < e < -1.0  # H2 equilibrium ~ -1.1 Ha with STO-3G

    def test_dissociation_limit(self):
        """CASSCF at large separation (~10 Bohr) approaches two H atoms."""
        e = _h2_energy(10.0, method="casscf")
        # Two H atoms in STO-3G: each ~ -0.466 Ha, total ~ -0.93 Ha
        assert -0.95 < e < -0.90

    def test_casscf_lower_than_casci(self):
        """CASSCF always lower or equal to CASCI at any geometry."""
        for r in [1.0, 1.4, 2.5, 5.0, 10.0]:
            e_cas = _h2_energy(r, method="casci")
            e_scf = _h2_energy(r, method="casscf")
            assert e_scf <= e_cas + 1e-9, f"r={r}: CASSCF {e_scf} > CASCI {e_cas}"

    def test_curve_physical(self):
        """CASSCF dissociation curve has a minimum near equilibrium (~1.4 Bohr)
        and rises as the bond stretches."""
        e_eq = _h2_energy(1.4, method="casscf")
        e_diss = _h2_energy(10.0, method="casscf")
        assert e_diss > e_eq  # Dissociated > equilibrium
        assert e_diss - e_eq > 0.05  # Binding energy ~0.1 Ha

    def test_fci_limit(self):
        """CASSCF(2,2) == FCI for H2/STO-3G (full space is 2e,2o)."""
        e_casscf = _h2_energy(1.4, method="casscf")
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        b = BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, b, method="rhf")
        H = build_hamiltonian_mo(mol, b, C)
        e_fci = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=H.norb,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        assert abs(e_casscf - e_fci) < 1e-9

    def test_near_degeneracy_region(self):
        """At stretched bond (~2.5 Bohr), the (σ,σ*) orbitals are near-
        degenerate and CASSCF must converge despite the small Hessian
        eigenvalues."""
        e = _h2_energy(2.5, method="casscf")
        assert e < -0.85  # Must be bound (not crazy positive)
