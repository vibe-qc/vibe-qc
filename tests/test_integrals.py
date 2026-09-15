"""Structural and value checks on the AO integrals."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    BasisSet,
    compute_eri,
    compute_kinetic,
    compute_nuclear,
    compute_overlap,
)

from .conftest import GEOMETRIES, make_molecule


@pytest.fixture
def h2_sto3g():
    mol = make_molecule(GEOMETRIES["H2"])
    basis = BasisSet(mol, "sto-3g")
    return mol, basis


@pytest.fixture
def h2o_sto3g():
    mol = make_molecule(GEOMETRIES["H2O"])
    basis = BasisSet(mol, "sto-3g")
    return mol, basis


def test_h2_sto3g_basis_size(h2_sto3g):
    _, basis = h2_sto3g
    assert basis.nbasis == 2
    assert basis.nshells == 2


def test_h2o_sto3g_basis_size(h2o_sto3g):
    _, basis = h2o_sto3g
    # O: 1s, 2s, 2px, 2py, 2pz  => 5; H x 2: 1s each => 2; total 7
    assert basis.nbasis == 7


def test_overlap_is_symmetric_with_unit_diagonal(h2o_sto3g):
    _, basis = h2o_sto3g
    S = compute_overlap(basis)
    assert S.shape == (7, 7)
    np.testing.assert_allclose(S, S.T, atol=1e-13)
    np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-12)


def test_kinetic_is_symmetric(h2o_sto3g):
    _, basis = h2o_sto3g
    T = compute_kinetic(basis)
    np.testing.assert_allclose(T, T.T, atol=1e-13)


def test_nuclear_is_symmetric_and_negative_definite_diag(h2o_sto3g):
    mol, basis = h2o_sto3g
    V = compute_nuclear(basis, mol)
    np.testing.assert_allclose(V, V.T, atol=1e-13)
    # All on-site attractions must be negative (electron attracted to nuclei).
    assert np.all(np.diag(V) < 0.0)


def test_h2_sto3g_overlap_matches_szabo_ostlund(h2_sto3g):
    _, basis = h2_sto3g
    S = compute_overlap(basis)
    # Szabo & Ostlund Table 3.5: <1s_A|1s_B> = 0.6593 at R = 1.4 bohr.
    assert S[0, 1] == pytest.approx(0.6593, abs=5e-4)


def test_h2_sto3g_eri_matches_szabo_ostlund(h2_sto3g):
    _, basis = h2_sto3g
    eri = compute_eri(basis)
    # From Szabo & Ostlund Table 3.5, R = 1.4 bohr:
    assert eri[0, 0, 0, 0] == pytest.approx(0.7746, abs=5e-4)
    assert eri[0, 0, 1, 1] == pytest.approx(0.5697, abs=5e-4)
    assert eri[0, 0, 0, 1] == pytest.approx(0.4441, abs=5e-4)
    assert eri[0, 1, 0, 1] == pytest.approx(0.2970, abs=5e-4)


def test_eri_eight_fold_symmetry(h2o_sto3g):
    """(mu nu|la si) = (nu mu|la si) = (mu nu|si la) = (la si|mu nu) = ..."""
    _, basis = h2o_sto3g
    eri = compute_eri(basis)
    n = basis.nbasis
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    values = [
                        eri[i, j, k, l], eri[j, i, k, l],
                        eri[i, j, l, k], eri[j, i, l, k],
                        eri[k, l, i, j], eri[l, k, i, j],
                        eri[k, l, j, i], eri[l, k, j, i],
                    ]
                    assert max(values) - min(values) < 1e-13, (
                        f"8-fold symmetry broken at ({i},{j},{k},{l})"
                    )
