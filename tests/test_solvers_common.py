"""Tests for common interfaces (Hamiltonian, SolverResult)."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.solvers import (
    Hamiltonian,
    SolverOptions,
    SolverResult,
)
from vibeqc.solvers._common import (
    _antisymmetrize_h2e,
    _chemist_to_physicist,
    _physicist_to_chemist,
)


class TestHamiltonian:
    def test_construction(self):
        n = 4
        h1e = np.eye(n)
        h2e = np.zeros((n, n, n, n))
        ham = Hamiltonian(h1e=h1e, h2e=h2e, nuclear_repulsion=1.5)
        assert ham.norb == n
        assert ham.nuclear_repulsion == 1.5

    def test_norb_auto(self):
        h1e = np.eye(6)
        ham = Hamiltonian(h1e=h1e, h2e=np.zeros((6, 6, 6, 6)))
        assert ham.norb == 6

    def test_nelec_ms2(self):
        ham = Hamiltonian(
            h1e=np.eye(2),
            h2e=np.zeros((2, 2, 2, 2)),
            nelec=4,
            ms2=0,
        )
        assert ham.nelec == 4
        assert ham.ms2 == 0


class TestSolverResult:
    def test_defaults(self):
        result = SolverResult(energy=-1.0, method="test")
        assert result.converged is True
        assert result.energy_total == -1.0
        assert result.ci_coeffs is None

    def test_method_specific_fields(self):
        result = SolverResult(
            energy=-2.0,
            method="selected_ci",
            ci_coeffs=np.array([0.9, 0.1]),
            pt2_correction=-0.05,
            truncation_error=None,
            bond_dim=None,
            constraint_residual=None,
        )
        assert result.pt2_correction == -0.05
        assert result.energy == -2.0


class TestIntegralConversion:
    def test_physicist_to_chemist_roundtrip(self):
        """Test that (pq|rs) ↔ g_{pqrs} conversions are inverses."""
        n = 3
        h2e_phys = np.random.randn(n, n, n, n)
        chem = _physicist_to_chemist(h2e_phys)
        back = _chemist_to_physicist(chem)
        np.testing.assert_allclose(back, h2e_phys)

    def test_antisymmetrize(self):
        """Antisymmetrized tensor should satisfy <pq||rs> = -<qp||rs>
        when the original tensor has the correct particle-exchange symmetry."""
        n = 4
        # Build a tensor with physical symmetry: g_{pqrs} = g_{qpsr}
        h2e_raw = np.random.randn(n, n, n, n)
        h2e = 0.5 * (h2e_raw + h2e_raw.transpose(1, 0, 3, 2))
        g_as = _antisymmetrize_h2e(h2e)
        # <pq||rs> = -<qp||rs>
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        expected = -g_as[q, p, r, s]
                        np.testing.assert_allclose(
                            g_as[p, q, r, s], expected, atol=1e-14
                        )
