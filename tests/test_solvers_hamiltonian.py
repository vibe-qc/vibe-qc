"""Tests for Hamiltonian construction from AO integrals."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    build_hamiltonian_ao,
    build_hamiltonian_mo,
    canonical_orthogonalize,
    get_hf_orbital_provider,
    transform_hamiltonian,
)


@pytest.fixture
def h2_sto3g():
    """H₂ / STO-3G at R = 1.4 bohr."""
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "sto-3g")
    return mol, basis


class TestHamiltonianConstruction:
    def test_build_hamiltonian_ao(self, h2_sto3g):
        mol, basis = h2_sto3g
        ham = build_hamiltonian_ao(mol, basis)
        assert ham.norb == basis.nbasis
        assert ham.nelec == 2
        assert ham.ms2 == 0
        assert ham.nuclear_repulsion > 0
        assert ham.h1e.shape == (ham.norb, ham.norb)
        assert ham.h2e.shape == (ham.norb, ham.norb, ham.norb, ham.norb)

    def test_build_hamiltonian_mo(self, h2_sto3g):
        mol, basis = h2_sto3g
        C = get_hf_orbital_provider(mol, basis)
        ham = build_hamiltonian_mo(mol, basis, C)
        assert ham.norb == basis.nbasis
        # MO-basis h1e should be approximately diagonal (canonical HF)
        off_diag = ham.h1e - np.diag(np.diag(ham.h1e))
        assert np.max(np.abs(off_diag)) < 1e-6

    def test_transform_hamiltonian(self, h2_sto3g):
        mol, basis = h2_sto3g
        ham_ao = build_hamiltonian_ao(mol, basis)
        C = get_hf_orbital_provider(mol, basis)
        ham_mo = transform_hamiltonian(ham_ao, C)
        assert ham_mo.norb == basis.nbasis

    def test_canonical_orthogonalize(self, h2_sto3g):
        mol, basis = h2_sto3g
        from vibeqc._vibeqc_core import compute_overlap

        S = np.asarray(compute_overlap(basis))
        X = canonical_orthogonalize(S)
        # X^T S X = I
        I = X.T @ S @ X
        np.testing.assert_allclose(I, np.eye(X.shape[1]), atol=1e-12)
