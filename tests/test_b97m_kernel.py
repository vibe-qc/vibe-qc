"""De-risk self-check for the generic B97M semilocal energy kernel
(``vibeqc.compute_b97m_semilocal_exc``, the core of the ωB97M(2) xDH
build).

The kernel evaluates the B97M (w,u) power-series semilocal energy for an
arbitrary coefficient/index set. If we feed it **ωB97M-V's own**
coefficients (libxc ``par_wb97m_v``), it must reproduce libxc's ωB97M-V
semilocal energy density — i.e. ``Functional("wb97m-v").eval_polarised_mgga``
— to ~1e-8 on synthetic (ρ, σ, τ) points. Passing this validates every
primitive (Slater-X, erf attenuation, PW92 + Stoll, the (w,u) variables
and series) before ωB97M(2)'s coefficients are trusted.

ωB97M-V coefficients + (w,u) indices are transcribed from libxc 7.0.0
``hyb_mgga_xc_wb97mv.c`` (``par_wb97m_v``) /
``hyb_mgga_xc_wb97mv.mpl`` (index sets), γ's from the same files.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Functional,
    compute_b97m_semilocal_exc,
    compute_b97m_semilocal_vxc,
)

OMEGA = 0.3
GAMMA_X, GAMMA_SS, GAMMA_OS = 0.004, 0.2, 0.006

# ωB97M-V coefficients with their (w_power, u_power) indices.
TERMS_X = [(0.85, 0, 0), (1.007, 0, 1), (0.259, 1, 0)]
TERMS_SS = [(0.443, 0, 0), (-1.437, 0, 4), (-4.535, 1, 0),
            (-3.39, 2, 0), (4.278, 4, 3)]
TERMS_OS = [(1.0, 0, 0), (1.358, 1, 0), (2.924, 2, 0),
            (-8.812, 2, 1), (-1.39, 6, 0), (9.142, 6, 1)]

K_FACTOR_C = 4.557799872345597137288163759599305358515


def _synthetic_points(n=60, seed=20260620):
    """Non-pathological spin-resolved (ρ, σ, τ) points, moderate density so
    the erf attenuation stays out of the low-density large-a tail (where
    the not-yet-transplanted enforce_smooth_lr smoothing would matter)."""
    rng = np.random.default_rng(seed)
    rho_a = rng.uniform(0.10, 0.80, n)
    rho_b = rng.uniform(0.10, 0.80, n)
    xa = rng.uniform(0.20, 1.20, n)   # reduced gradient x_σ
    xb = rng.uniform(0.20, 1.20, n)
    sigma_aa = (xa * rho_a ** (4.0 / 3.0)) ** 2
    sigma_bb = (xb * rho_b ** (4.0 / 3.0)) ** 2
    sigma_ab = 0.5 * np.sqrt(sigma_aa * sigma_bb)
    # reduced τ: t_σ = τ_σ/ρ_σ^(5/3) ∈ [1.5, 8]·... around K_FACTOR_C.
    ta = rng.uniform(2.0, 9.0, n)
    tb = rng.uniform(2.0, 9.0, n)
    tau_a = ta * rho_a ** (5.0 / 3.0)
    tau_b = tb * rho_b ** (5.0 / 3.0)
    return rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b


def test_b97m_kernel_reproduces_wb97mv_semilocal():
    ra, rb, saa, sab, sbb, ta, tb = _synthetic_points()

    mine = compute_b97m_semilocal_exc(
        ra, rb, saa, sab, sbb, ta, tb,
        OMEGA, GAMMA_X, GAMMA_SS, GAMMA_OS, TERMS_X, TERMS_SS, TERMS_OS)

    f = Functional("wb97m-v", 2)
    libxc_exc = f.eval_polarised_mgga(ra, rb, saa, sab, sbb, ta, tb)[0]

    # Per-point energy density ρ·ε_xc^semilocal must match libxc's ωB97M-V.
    max_abs = np.max(np.abs(mine - libxc_exc))
    np.testing.assert_allclose(mine, libxc_exc, rtol=1e-8, atol=1e-9), (
        f"max abs diff = {max_abs:.3e}")


def test_b97m_potential_reproduces_wb97mv():
    """The B97M semilocal POTENTIAL (finite-difference of the validated
    energy kernel) — needed for the self-consistent ωB97M(2) variant —
    reproduces libxc's ωB97M-V v_rho/v_sigma/v_tau when fed ωB97M-V's
    coefficients. This validates the SC-variant's new piece the same way
    the energy was (no ORCA anchor exists for SC-ωB97M(2))."""
    ra, rb, saa, sab, sbb, ta, tb = _synthetic_points()

    vra, vrb, vsaa, vsbb, vta, vtb = compute_b97m_semilocal_vxc(
        ra, rb, saa, sab, sbb, ta, tb,
        OMEGA, GAMMA_X, GAMMA_SS, GAMMA_OS, TERMS_X, TERMS_SS, TERMS_OS)

    f = Functional("wb97m-v", 2)
    _, lvra, lvrb, lvsaa, lvsab, lvsbb, lvta, lvtb = \
        f.eval_polarised_mgga(ra, rb, saa, sab, sbb, ta, tb)

    # FD potential vs libxc analytic potential: ~1e-7 (FD-limited).
    np.testing.assert_allclose(vra, lvra, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(vrb, lvrb, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(vsaa, lvsaa, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(vsbb, lvsbb, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(vta, lvta, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(vtb, lvtb, rtol=2e-6, atol=1e-7)
