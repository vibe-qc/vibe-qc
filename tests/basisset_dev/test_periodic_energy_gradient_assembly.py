"""Build-free validation of the periodic (k-point) Pulay assembly.

The periodic counterpart of ``test_energy_gradient_assembly.py``: validates
``periodic_energy_gradient_fd`` against the finite difference of a mock
multi-k total energy, with no vibe-qc build (pure numpy + scipy). Exercises the
periodic-specific structure the molecular assembly does not — complex per-k
matrices, k-weights, the Re[] per-k contribution, and the −tr(W·∂S) Pulay term
through a per-k Hermitian *generalized* eigenproblem.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_linalg = pytest.importorskip("scipy.linalg")

from vibeqc.basis_optimization.periodic_energy_gradient import (  # noqa: E402
    periodic_energy_gradient_fd,
)


def _hermitian(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _pos_def(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a @ a.conj().T + n * np.eye(n)   # well-conditioned Hermitian PD


class _MockPeriodicProvider:
    """Non-interacting (G=0) mock band structure over a few k-points.

    Hcore(k, x) = A(k) + x·dA(k); S(k, x) = S0(k) + x·dS(k) (PD near x0). The
    per-k closed-shell density P(k) = 2 Σ_occ C_iC_i† comes from the Hermitian
    generalized eigenproblem (Hcore(k,x), S(k,x)); G(k) ≡ 0. The total energy
    Σ_k w_k Re tr(P(k)·Hcore(k)) is what the assembly must differentiate.
    """

    def __init__(self, n_bf=4, n_occ=1, seed=7):
        rng = np.random.default_rng(seed)
        self.kweights = np.array([0.5, 0.3, 0.2])
        self.nk = len(self.kweights)
        self.n_occ = n_occ
        self.A = [_hermitian(rng, n_bf) for _ in range(self.nk)]
        self.dA = [0.1 * _hermitian(rng, n_bf) for _ in range(self.nk)]
        self.S0 = [_pos_def(rng, n_bf) for _ in range(self.nk)]
        self.dS = [0.02 * _hermitian(rng, n_bf) for _ in range(self.nk)]

    def overlap(self, x):
        return [self.S0[k] + float(x[0]) * self.dS[k] for k in range(self.nk)]

    def hcore(self, x):
        return [self.A[k] + float(x[0]) * self.dA[k] for k in range(self.nk)]

    def gmatrix(self, x):
        return [np.zeros_like(self.A[k]) for k in range(self.nk)]

    def density(self, x):
        H = self.hcore(x); S = self.overlap(x)
        out = []
        for k in range(self.nk):
            _, C = scipy_linalg.eigh(H[k], S[k])
            occ = C[:, : self.n_occ]
            out.append(2.0 * occ @ occ.conj().T)   # closed-shell occupation 2
        return out

    def total_energy(self, x):
        P = self.density(x); H = self.hcore(x)
        return sum(self.kweights[k] * np.real(np.trace(P[k] @ H[k]))
                   for k in range(self.nk))


def test_periodic_assembly_matches_total_energy_fd():
    prov = _MockPeriodicProvider()
    x0 = np.array([0.0])
    g_an = periodic_energy_gradient_fd(prov, x0, delta=1e-4)
    h = 1e-5
    g_fd = (prov.total_energy(x0 + h) - prov.total_energy(x0 - h)) / (2.0 * h)
    np.testing.assert_allclose(g_an[0], g_fd, atol=1e-7, rtol=1e-6)


def test_periodic_two_electron_term():
    """½ Σ_k w_k Re tr(P·∂G) at frozen P, with G(k, x) linear and S, Hcore
    constant — isolates the two-electron contribution of the assembly."""
    rng = np.random.default_rng(3)
    n, nk = 3, 2

    class P2:
        kweights = np.array([0.6, 0.4])
        P = [2.0 * (lambda c: c @ c.conj().T)(rng.standard_normal((n, 1))
                                               + 1j * rng.standard_normal((n, 1)))
             for _ in range(nk)]
        S0 = [_pos_def(rng, n) for _ in range(nk)]
        H0 = [_hermitian(rng, n) for _ in range(nk)]
        dG = [_hermitian(rng, n) for _ in range(nk)]

        def overlap(self, x):
            return self.S0

        def hcore(self, x):
            return self.H0

        def gmatrix(self, x):
            return [self.H0[k] * 0 + float(x[0]) * self.dG[k] for k in range(nk)]

        def density(self, x):
            return self.P

    prov = P2()
    g = periodic_energy_gradient_fd(prov, np.array([0.0]), delta=1e-4)[0]
    expect = 0.5 * sum(prov.kweights[k] * np.real(np.trace(prov.P[k] @ prov.dG[k]))
                       for k in range(nk))
    np.testing.assert_allclose(g, expect, atol=1e-9, rtol=1e-7)


# ---------------------------------------------------------------------------
# Phase P2 (skeleton): the build-free contraction helper _periodic_pulay_contract
# ---------------------------------------------------------------------------


def test_pulay_contract_matches_fd_assembly_and_energy_fd():
    """``_periodic_pulay_contract`` (explicit per-k derivatives) agrees with the
    P0 FD-assembly path and with the mock total-energy FD.

    The mock's integrals are linear in x, so their exact derivatives are the
    constant slopes (dHcore = dA, dS = dS_mock, dG = 0). Feeding those to the
    contraction helper must reproduce ``periodic_energy_gradient_fd`` (which
    central-differences the integrals) and the analytic limit of the
    total-energy FD. Evaluated off x = 0."""
    from vibeqc.basis_optimization.periodic_energy_gradient import (
        _periodic_pulay_contract,
    )

    prov = _MockPeriodicProvider()
    x0 = np.array([0.37])  # off the reference point
    nk = prov.nk

    g_fd_assembly = periodic_energy_gradient_fd(prov, x0)[0]

    P = prov.density(x0)
    H = prov.hcore(x0)
    G = prov.gmatrix(x0)
    F = [H[k] + G[k] for k in range(nk)]          # G = 0 here, so F = Hcore
    dS = prov.dS
    dHcore = prov.dA
    dG = [np.zeros_like(prov.A[k]) for k in range(nk)]
    g_contract = _periodic_pulay_contract(P, F, prov.kweights, dS, dHcore, dG)

    h = 1e-5
    g_efd = (prov.total_energy(x0 + h) - prov.total_energy(x0 - h)) / (2.0 * h)

    np.testing.assert_allclose(g_contract, g_fd_assembly, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(g_contract, g_efd, atol=1e-7, rtol=1e-6)


def test_pulay_contract_two_electron_term():
    """``_periodic_pulay_contract`` isolates ½ Σ_k w_k Re tr(P·dG) when the
    one-electron derivatives vanish (dS = dHcore = 0)."""
    from vibeqc.basis_optimization.periodic_energy_gradient import (
        _periodic_pulay_contract,
    )

    rng = np.random.default_rng(11)
    n, nk = 3, 2
    w = np.array([0.6, 0.4])
    P = [2.0 * (lambda c: c @ c.conj().T)(rng.standard_normal((n, 1))
                                          + 1j * rng.standard_normal((n, 1)))
         for _ in range(nk)]
    F = [_hermitian(rng, n) for _ in range(nk)]       # W·dS = 0 since dS = 0
    dG = [_hermitian(rng, n) for _ in range(nk)]
    zero = [np.zeros((n, n), dtype=complex) for _ in range(nk)]

    g = _periodic_pulay_contract(P, F, w, zero, zero, dG)
    expect = 0.5 * sum(w[k] * np.real(np.trace(P[k] @ dG[k])) for k in range(nk))
    np.testing.assert_allclose(g, expect, atol=1e-12, rtol=1e-9)
