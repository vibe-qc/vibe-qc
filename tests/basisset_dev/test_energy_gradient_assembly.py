"""Phase-0 SCF-energy-gradient assembly tests (build-free).

Validates the pure-numpy assembly in ``basis_optimization.energy_gradient``
against a self-contained mock RHF: a 2-basis-function, 1-electron-pair
system with smooth model integrals. Because the SCF is solved self-
consistently, the analytic Pulay assembly (frozen P/W, integrals
finite-differenced) must reproduce the *full* SCF-energy finite difference.
This proves the energy-gradient math (W = ½·P·F·P, the −tr(W·∂S) Pulay
term, the G(P) build) without a built vibe-qc; the real
``VibeqcIntegralProvider`` path is validated separately on a build
(compute-reference). Run with:

    .venv/bin/python -m pytest tests/basisset_dev/test_energy_gradient_assembly.py -v --noconftest
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    import vibeqc  # noqa: F401
except Exception:
    PKG_PARENT = REPO_ROOT / "python"
    if str(PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(PKG_PARENT))
    for mod_name in list(sys.modules):
        # Keep vibeqc._vibeqc_core* registered even here: the single-phase-init
        # C extension never re-runs PyInit on re-import, so deleting those
        # entries would strip the pybind11 def_submodule registrations
        # (...semiempirical.{nddo,xtb,indo}) for the rest of the pytest process.
        # Reachable when ``import vibeqc`` fails *after* the extension itself
        # loaded.
        if mod_name == "vibeqc._vibeqc_core" or mod_name.startswith(
            "vibeqc._vibeqc_core."
        ):
            continue
        if mod_name == "vibeqc" or mod_name.startswith("vibeqc."):
            del sys.modules[mod_name]
    _shim = types.ModuleType("vibeqc")
    _shim.__path__ = [str(PKG_PARENT / "vibeqc")]
    sys.modules["vibeqc"] = _shim

from vibeqc.basis_optimization.energy_gradient import (  # noqa: E402
    build_g,
    electronic_energy,
    energy_gradient_fd,
    energy_weighted_density,
    fock,
)


# --------------------------------------------------------------------------
# A tiny self-contained RHF: 2 basis functions, 1 doubly-occupied MO.
# --------------------------------------------------------------------------


def _sym_eri(vals: dict[tuple, float]) -> np.ndarray:
    """Build a 2×2×2×2 chemist-notation ERI with full 8-fold symmetry.

    ``vals`` maps a canonical key to the integral value; every (μν|λσ) is
    assigned by reducing to its canonical key, so all permutation symmetries
    (μν|λσ)=(νμ|λσ)=(μν|σλ)=(λσ|μν) hold by construction.
    """

    def canon(m, n, l, s):
        bra = tuple(sorted((m, n)))
        ket = tuple(sorted((l, s)))
        return tuple(sorted((bra, ket)))

    eri = np.zeros((2, 2, 2, 2))
    for m in range(2):
        for n in range(2):
            for l in range(2):
                for s in range(2):
                    eri[m, n, l, s] = vals[canon(m, n, l, s)]
    return eri


# Fixed, physically-plausible base ERI (8-fold symmetric).
_BASE_ERI = _sym_eri({
    ((0, 0), (0, 0)): 0.80,
    ((1, 1), (1, 1)): 0.70,
    ((0, 0), (1, 1)): 0.50,
    ((0, 1), (0, 1)): 0.15,
    ((0, 0), (0, 1)): 0.05,
    ((0, 1), (1, 1)): 0.04,
})


def _model(t: float):
    """Smooth model integrals (S, Hcore, ERI) for the scalar parameter t.

    Each of the three depends on t in a *different* smooth way, so the
    gradient test exercises the ∂Hcore, ∂G and ∂S terms independently. The
    Hcore diagonal is split by ~1 Ha so the two functions are clearly
    non-degenerate (the lowest MO holds the pair throughout), keeping the
    one-pair SCF a clean, stable fixed point over the tested t range.
    """
    sigma = 0.35 * np.exp(-0.5 * t * t)  # overlap off-diagonal ∈ (0, 0.35]
    S = np.array([[1.0, sigma], [sigma, 1.0]])
    Hcore = np.array([
        [-1.6 + 0.10 * t, -0.25 * sigma],
        [-0.25 * sigma, -0.6 - 0.05 * t],
    ])
    eri = _BASE_ERI * (1.0 + 0.20 * t + 0.10 * t * t)
    return S, Hcore, eri


def _mock_scf(S, Hcore, eri, *, tol=1e-12, maxit=5000, damp=0.6):
    """Closed-shell RHF for one electron pair via the generalized eigenproblem.

    Density-damped fixed-point iteration to a tight P convergence, then a
    final clean diagonalisation so the returned (P, eps, C, F) are mutually
    consistent (F·C = S·C·ε, P = 2·c₀c₀ᵀ). Raises if it does not converge.
    """
    eps, C = sla.eigh(Hcore, S)  # core guess
    P = 2.0 * np.outer(C[:, 0], C[:, 0])
    for _ in range(maxit):
        F = fock(P, Hcore, eri)
        eps, C = sla.eigh(F, S)  # ascending; columns S-orthonormal
        P_new = 2.0 * np.outer(C[:, 0], C[:, 0])
        if np.linalg.norm(P_new - P) < tol:
            P = P_new
            break
        P = damp * P_new + (1.0 - damp) * P
    else:  # pragma: no cover - model is chosen to converge
        raise RuntimeError("mock SCF did not converge")
    # Consistent final set at the fixed point.
    F = fock(P, Hcore, eri)
    eps, C = sla.eigh(F, S)
    P = 2.0 * np.outer(C[:, 0], C[:, 0])
    return P, eps, C, F


class _MockProvider:
    def overlap(self, x):
        return _model(float(x[0]))[0]

    def hcore(self, x):
        return _model(float(x[0]))[1]

    def eri(self, x):
        return _model(float(x[0]))[2]

    def density(self, x):
        S, H, e = _model(float(x[0]))
        return _mock_scf(S, H, e)[0]


def _total_energy(t: float) -> float:
    S, H, e = _model(t)
    P = _mock_scf(S, H, e)[0]
    return electronic_energy(P, H, e)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_energy_weighted_density_identity():
    """W = ½·P·F·P must equal 2 Σ_occ ε_i C_iC_iᵀ at convergence."""
    S, H, e = _model(0.25)
    P, eps, C, F = _mock_scf(S, H, e)
    W_identity = energy_weighted_density(P, F)
    c0 = C[:, 0]
    W_explicit = 2.0 * eps[0] * np.outer(c0, c0)
    np.testing.assert_allclose(W_identity, W_explicit, atol=1e-9)


def test_build_g_symmetry_and_energy_consistency():
    """G(P) is symmetric and reproduces the SCF electronic energy."""
    S, H, e = _model(0.1)
    P, _, _, _ = _mock_scf(S, H, e)
    G = build_g(P, e)
    np.testing.assert_allclose(G, G.T, atol=1e-12)
    # E = tr(P·Hcore) + ½ tr(P·G)
    assert electronic_energy(P, H, e) == pytest.approx(
        float(np.trace(P @ H) + 0.5 * np.trace(P @ G))
    )


@pytest.mark.parametrize("t", [-0.4, -0.1, 0.0, 0.3, 0.7])
def test_assembly_matches_full_scf_finite_difference(t):
    """The frozen-P/W Pulay assembly == full re-SCF energy finite difference."""
    x0 = np.array([t])
    g_assembly = energy_gradient_fd(_MockProvider(), x0, delta=1e-5)
    h = 1e-6
    g_fd = (_total_energy(t + h) - _total_energy(t - h)) / (2.0 * h)
    assert g_assembly[0] == pytest.approx(g_fd, rel=1e-5, abs=1e-8)


def test_assembly_nontrivial():
    """Guard against a vacuous pass: the gradient is genuinely non-zero and
    each term contributes (S, Hcore, ERI all vary with t)."""
    g = energy_gradient_fd(_MockProvider(), np.array([0.3]), delta=1e-5)
    assert abs(g[0]) > 1e-3
