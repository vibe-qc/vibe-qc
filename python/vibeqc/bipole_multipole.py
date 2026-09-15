"""BIPOLE multipole-far-pair branch -- Phase 2: multipole-multipole
Coulomb interaction tensor.

For two distributions characterised by their multipole moments
``M^A_{l1,m1}`` (at center O^A) and ``M^B_{l2,m2}`` (at center O^B),
separated by ``R = O^B - O^A``, the Coulomb interaction energy is::

    E_{AB} = S_{l1,m1,l2,m2}  M^A_{l1,m1} . T_{l1,m1; l2,m2}(R) . M^B_{l2,m2}

We use Stone's real solid-harmonic convention (Schmidt-semi-normalised,
with the ``√(4pi/(2l+1))`` factor absorbed) -- identical to the one
realised by ``_cart_to_sph.cartesian_to_spherical_matrix``, so that the
multipole moments and the solid harmonics ``Z_{l,m}`` are mutually
consistent to machine precision. In this convention a point charge ``q``
at the origin gives ``M_{0,0} = q`` and the monopole-monopole interaction
is exactly ``q1.q2/|R|``.

* Real solid harmonics ``Z_{l,m}(r)`` indexed ``(l, m)`` with
  ``l = 0, 1, 2, ...`` and ``m in [-l, +l]`` (``lm_index`` flat layout) --
  the natural objects for the multipole expansion of ``1/|r-R|``.
* Multipole moments ``M^A_{l,m} = ∫ r_A(r) . Z_{l,m}(r - O^A) dr``.
* The interaction tensor ``T_{l1,m1; l2,m2}(R)`` is assembled from the
  exact Cartesian Taylor expansion of ``1/|R - r1 + r2|`` (see
  ``multipole_interaction_tensor``) and converted to the spherical
  ``Z_{l,m}`` basis via the ``cartesian_to_spherical_matrix``
  pseudoinverse. Working in Cartesian space is convention-free and sticks
  to exact closed-form derivatives of ``1/|R|``, avoiding the
  spherical-harmonic coupling-coefficient (Wigner-3j) normalisation
  ambiguities entirely.

Background: Stone, *The Theory of Intermolecular Forces* (2nd ed., OUP
2013) ch. 3; the Cartesian multipole expansion of ``1/|r-R|`` is standard
(Jackson, *Classical Electrodynamics*, Sec.4.1). The target periodic
quartet decomposition is Pisani, Dovesi, and Roetti, *Hartree-Fock Ab
Initio Treatment of Crystalline Systems* (1988), Ch. II.4c,
Eqs. II.4.7-II.4.10, doi:10.1007/978-3-642-93385-1.

Module status: this interaction tensor belongs to a dormant prototype.
The public BIPOLE SCF routes fail closed on requests to enable the quartet
far field. The classifier, order map, and screened contractor have not been
traced to a primary-source derivation. Production total energies use the
separate exact Ewald-J split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "real_solid_harmonic",
    "irregular_real_solid_harmonic",
    "multipole_interaction_tensor",
    "screened_multipole_interaction_tensor",
    "sr_multipole_interaction_tensor",
    "multipole_pair_energy",
    "n_components",
    "lm_index",
]


# ---------------------------------------------------------------------
# Component indexing
# ---------------------------------------------------------------------
def n_components(L_max: int) -> int:
    """Number of (l, m) components for ``l = 0..L_max``: ``(L_max+1)^2``."""
    return (L_max + 1) ** 2


def lm_index(l: int, m: int) -> int:
    """Flat index for spherical-harmonic component (l, m). ``m in [-l, +l]``.

    Layout: ``[Y_{00}, Y_{1-1}, Y_{10}, Y_{1+1}, Y_{2-2}, ..., Y_{2+2}, ...]``.
    """
    if not (0 <= l):
        raise ValueError(f"l must be >= 0; got {l}")
    if not (-l <= m <= l):
        raise ValueError(f"m={m} out of range for l={l}")
    return l * l + l + m


# ---------------------------------------------------------------------
# Real solid harmonics
# ---------------------------------------------------------------------
def real_solid_harmonic(l: int, m: int, r: np.ndarray) -> float:
    """Real solid harmonic ``Z_{l,m}(r)`` in the ``cartesian_to_spherical_matrix``
    convention (Stone convention, Schmidt-semi-normalised, ``sqrt(4pi/(2l+1))``
    absorbed).

    Evaluated directly as a homogeneous polynomial in ``(x, y, z)`` using
    the conversion-matrix coefficients, guaranteeing that the multipole
    moments ``M_sph = C @ M_cart`` and the solid harmonics ``Z_{l,m}`` are
    mutually consistent to machine precision.

    For ``l=0..3`` the polynomial forms are:

    * ``Z_{0,0} = 1``
    * ``Z_{1,-1} = -sqrt2*y``, ``Z_{1,0} = z``, ``Z_{1,1} = sqrt2*x``
    * ``Z_{2,-2} = sqrt3*xy``, ``Z_{2,-1} = sqrt3*yz``,
      ``Z_{2,0} = (2z**2 - x**2 - y**2)/(2*sqrt2)``,
      ``Z_{2,1} = sqrt3*xz``, ``Z_{2,2} = (sqrt3/2)*(x**2 - y**2)``
    * ``Z_{3,...}`` -- see ``_cart_to_sph.py`` for the L=3 polynomial set.

    Parameters
    ----------
    l : int in {0, 1, 2, 3}
        Spherical-harmonic degree.
    m : int with ``|m| <= l``
        Spherical-harmonic order.
    r : array_like shape (3,)
        Cartesian vector.

    Returns
    -------
    float
        Value of ``Z_{l,m}(r)``.
    """
    r = np.asarray(r, dtype=float).reshape(3)
    x, y, z = float(r[0]), float(r[1]), float(r[2])

    # At origin, only Z_{0,0} is non-zero (homogeneous polynomials
    # of degree l>0 all vanish at r=0).
    if l == 0 and m == 0:
        return 1.0

    # Evaluate from the conversion-matrix coefficients so that
    # M_sph = C @ M_cart and Z_{l,m}(r) = sum C_{lm,ijk} * x^i y^j z^k
    # are exactly consistent.
    from ._cart_to_sph import cartesian_to_spherical_matrix as _C
    from .bipole_cell_moments import cartesian_component_indices as _cci

    C = _C(l)
    cart_idx_list = _cci(l)
    result = 0.0
    row = lm_index(l, m)
    for col, (i, j, k) in enumerate(cart_idx_list):
        coeff = C[row, col]
        if coeff == 0.0:
            continue
        result += coeff * (x**i) * (y**j) * (z**k)
    return result


def irregular_real_solid_harmonic(l: int, m: int, R: np.ndarray) -> float:
    """Irregular (singular) real solid harmonic ``I_{l,m}(R) = S_{l,m}(R̂) / |R|^{l+1}``.

    This is the kernel that appears in multipole-multipole interactions.
    For ``l = m = 0`` it reduces to ``1/|R|`` (the bare Coulomb).

    Parameters
    ----------
    l : int >= 0
    m : int with ``|m| <= l``
    R : array_like shape (3,)
        Separation vector. Must have ``|R| > 0``.

    Returns
    -------
    float
        Value of ``I_{l,m}(R)``.
    """
    R = np.asarray(R, dtype=float).reshape(3)
    radius = float(np.linalg.norm(R))
    if radius < 1e-30:
        raise ValueError("irregular_real_solid_harmonic singular at R=0")
    # I_{l,m}(R) = Z_{l,m}(R) / |R|^{2l+1}
    Z = real_solid_harmonic(l, m, R)
    return Z / radius ** (2 * l + 1)


# ---------------------------------------------------------------------
# Multipole interaction tensor T_{l1,m1; l2,m2}(R)
# ---------------------------------------------------------------------
def _cartesian_derivative_inv_r(gamma: Tuple[int, int, int], R: np.ndarray) -> float:
    """Exact Cartesian derivative ``d^gamma (1/|R|)`` for ``|gamma| <= 4``.

    ``gamma = (i, j, k)`` is the multi-index of partial derivatives w.r.t.
    ``(x, y, z)``; ``R = (x, y, z)`` and ``r = |R|``. The closed forms are
    the standard gradients of the bare Coulomb kernel (Jackson, Classical
    Electrodynamics, 3rd ed., Sec. 4.1), extended to fourth order for the
    prototype tensor. Pisani-Dovesi-Roetti (1988), p. 51, discusses
    total-order truncation of the periodic bipolar expansion.

    Each tensor is harmonic (``grad^2(1/r) = 0`` for ``r > 0``), so
    contracting any pair of repeated indices gives zero.  The order-4
    form is the rank-4 isotropic harmonic tensor:

        d_abcd (1/r) = [105 R_a R_b R_c R_d
          - 15 r^2 (δ_ab R_c R_d + δ_ac R_b R_d + ... (6 terms))
          + 3 r^4 (δ_ab δ_cd + δ_ac δ_bd + δ_ad δ_bc)] / r^9
    """
    order = sum(gamma)
    r = float(np.linalg.norm(R))
    if order == 0:
        return 1.0 / r
    # Expand the multi-index into a flat list of axes, e.g. (2,0,1)->[0,0,2].
    axes = [a for a, n in enumerate(gamma) for _ in range(n)]
    kron = lambda i, j: 1.0 if i == j else 0.0
    if order == 1:
        (a,) = axes
        return -R[a] / r ** 3
    if order == 2:
        a, b = axes
        return (3.0 * R[a] * R[b] - kron(a, b) * r ** 2) / r ** 5
    if order == 3:
        a, b, c = axes
        return (
            -15.0 * R[a] * R[b] * R[c]
            + 3.0 * r ** 2 * (
                kron(a, b) * R[c]
                + kron(a, c) * R[b]
                + kron(b, c) * R[a]
            )
        ) / r ** 7
    if order == 4:
        a, b, c, d = axes
        # All 6 pairwise-index terms for r^2*(δ_ij R_k R_l)
        pairs = [
            (a, b, c, d),
            (a, c, b, d),
            (a, d, b, c),
            (b, c, a, d),
            (b, d, a, c),
            (c, d, a, b),
        ]
        sum_15 = sum(
            kron(p0, p1) * R[p2] * R[p3]
            for (p0, p1, p2, p3) in pairs
        )
        # All 3 pairwise-double-Kronecker terms for r^4*(δ_ij δ_kl)
        sum_3 = (
            kron(a, b) * kron(c, d)
            + kron(a, c) * kron(b, d)
            + kron(a, d) * kron(b, c)
        )
        return (
            105.0 * R[a] * R[b] * R[c] * R[d]
            - 15.0 * r ** 2 * sum_15
            + 3.0 * r ** 4 * sum_3
        ) / r ** 9
    raise ValueError(f"derivative order {order} > 4 not supported")


# Hard clamp on the total Cartesian order ``|alpha| + |beta|`` retained in
# the bipolar Taylor expansion (see :func:`multipole_interaction_tensor`).
#
# The bare and the erf-screened tensor MUST share this clamp.  The
# short-range tensor is assembled by linearity as ``T_bare - T_erf``
# (:func:`sr_multipole_interaction_tensor`), so if the two are truncated at
# different orders the excess block carries ``-T_erf`` with no matching bare
# part.  That is not a small residual: on MgO/STO-3G it made the order-4
# block of the short-range tensor several times larger than every legitimate
# entry combined.
#
# Raising the clamp to total order 4 (within the truncation scheme discussed
# by Pisani-Dovesi-Roetti 1988, p. 51)
# needs derivatives of the kernel to order 4 for the tensor and order 5 for
# its gradient; the erf ladder currently refuses order 5.  See
# ``handovers/HANDOVER_BIPOLE_PRODUCTION.md``.  Keep this constant and its
# C++ twin (``kMaxBipolarTotalOrder`` in ``cpp/src/bipole_multipole.cpp``)
# in lock-step -- the two contractors are pinned against each other by
# ``tests/test_bipole_contractor_parity.py``.
_MAX_BIPOLAR_TOTAL_ORDER = 3


def multipole_interaction_tensor(
    L_max_A: int,
    L_max_B: int,
    R: np.ndarray,
) -> np.ndarray:
    """Build ``T_{l1,m1; l2,m2}(R)`` for ``l1 <= L_max_A``, ``l2 <= L_max_B``.

    Computed from the rigorous Cartesian Taylor expansion of
    ``1/|R - r1 + r2|`` and converted to spherical via the
    ``cartesian_to_spherical_matrix`` pseudoinverse.  This avoids
    Wigner-3j normalisation ambiguities entirely and is correct to
    machine precision for the spherical-harmonic convention used by
    the BIPOLE bipolar far-field.

    ``L_max_A``/``L_max_B`` set the output shape; the retained terms are
    additionally clamped to total order ``l1 + l2 <=
    _MAX_BIPOLAR_TOTAL_ORDER``, so blocks above that order are returned as
    zero even when the shape has room for them.

    Returns
    -------
    T : ndarray of shape (n_A, n_B) where ``n_A = (L_max_A+1)^2``
        and ``n_B = (L_max_B+1)^2``.
    """
    from math import factorial as _factorial

    R = np.asarray(R, dtype=float).reshape(3)

    L_max = max(L_max_A, L_max_B)
    if L_max > 4:
        raise ValueError("L_max > 4 not supported (hexadecapole)")

    from .bipole_cell_moments import cartesian_component_indices as _cart_idx
    from ._cart_to_sph import cartesian_to_spherical_matrix as _C_mat

    # --- Build Cartesian interaction tensor ----------------------
    # Bipolar Taylor expansion of the two-body kernel about R = O^B - O^A:
    #   1/|R - r1 + r2| = S_{a,b}  (-r1)^a/a! . r2^b/b! . d^{a+b}(1/|R|)
    # => E = S_{a,b} M^A_cart[a] . T_cart[a,b] . M^B_cart[b], with
    #   T_cart[a,b] = (-1)^{|a|} / (a! b!) . d^{a+b}(1/|R|)
    # and M_cart[g] = S_i q_i r_i^g the raw Cartesian monomial moments.
    # The Cartesian derivatives d^g(1/|R|) are taken exactly in closed
    # form (``_cartesian_derivative_inv_r``) -- convention-free, so the
    # spherical convention enters only through the basis change below.
    indices = _cart_idx(L_max)
    n_cart = len(indices)
    T_cart = np.zeros((n_cart, n_cart), dtype=float)

    for ia, (i1, j1, k1) in enumerate(indices):
        for ib, (i2, j2, k2) in enumerate(indices):
            L = i1 + i2 + j1 + j2 + k1 + k2
            if L > L_max or L > _MAX_BIPOLAR_TOTAL_ORDER:
                continue
            gamma = (i1 + i2, j1 + j2, k1 + k2)
            D_val = _cartesian_derivative_inv_r(gamma, R)

            # T_cart[alpha,beta] = (-1)^{|alpha|} / (alpha! beta!) * D^{alpha+beta}(1/|R|)
            sign = (-1.0) ** (i1 + j1 + k1)
            denom = (
                _factorial(i1) * _factorial(j1) * _factorial(k1)
                * _factorial(i2) * _factorial(j2) * _factorial(k2)
            )
            T_cart[ia, ib] = sign * D_val / denom

    # --- Convert to spherical via pseudoinverse ------------------
    C = _C_mat(L_max)
    C_pinv = np.linalg.pinv(C)
    T_sph = C_pinv.T @ T_cart @ C_pinv

    # Slice to requested L_max_A x L_max_B, zero-filling if smaller.
    n_A = n_components(L_max_A)
    n_B = n_components(L_max_B)
    T = np.zeros((n_A, n_B), dtype=float)
    n_A_full = n_components(L_max)
    n_B_full = n_components(L_max)
    T[: min(n_A, n_A_full), : min(n_B, n_B_full)] = T_sph[: n_A, : n_B]
    return T



def _boys(n: int, x: float) -> float:
    """Boys function ``F_n(x) = ∫_0^1 t^{2n} exp(-x t^2) dt``.

    Standard object of Gaussian integral technology (Boys, Proc. R. Soc.
    A 200, 542 (1950) -- shared-background, uncited per the CODATA rule).
    Evaluated through the regularised lower incomplete gamma
    ``F_n(x) = Γ(n+1/2)·P(n+1/2, x) / (2 x^{n+1/2})`` with the series
    limit ``F_n(0) = 1/(2n+1)``; the small-x branch avoids the 0/0.
    """
    if x < 1e-14:
        return 1.0 / (2 * n + 1)
    from scipy.special import gammainc  # noqa: PLC0415

    return float(
        math.gamma(n + 0.5) * gammainc(n + 0.5, x) / (2.0 * x ** (n + 0.5))
    )


def _cartesian_derivative_erf_r(
    gamma: Tuple[int, int, int], R: np.ndarray, mu: float
) -> float:
    """Exact Cartesian derivative ``d^gamma [erf(sqrt(mu)·r)/r]``, |gamma| <= 3.

    The screened kernel is a Boys function in disguise,
    ``erf(sqrt(mu) r)/r = 2 sqrt(mu/pi) · F_0(mu r^2)``, so its Cartesian
    derivatives follow the McMurchie-Davidson Hermite ladder over Boys
    orders -- Eq. (4.6) of McMurchie & Davidson, J. Comput. Phys. 26,
    218 (1978), doi:10.1016/0021-9991(78)90092-X:

        R^n_{000}    = (-2 mu)^n · 2 sqrt(mu/pi) · F_n(mu r^2)
        R^n_{t+1,u,v} = t · R^{n+1}_{t-1,u,v} + X · R^{n+1}_{t,u,v}

    (and cyclically for u -> Y, v -> Z), with
    ``d^{(t,u,v)} f = R^0_{t,u,v}``. This is the erf-screened analogue of
    the spherically-symmetric-kernel recursion in Saunders, Freyria-Fava,
    Dovesi, Salasco & Roetti, Mol. Phys. 77, 629 (1992), Sec. 5.3,
    Eqs. (90)-(92), where the bare-kernel radial factors
    ``(-1)^n (2n-1)!!/r^{2n+1}`` are replaced by Boys functions.
    Validated against central finite differences in
    ``tests/test_bipole_multipole.py``.
    """
    order = sum(gamma)
    if order > 4:
        raise ValueError(f"|gamma| <= 4 supported; got {gamma}")
    R = np.asarray(R, dtype=float).reshape(3)
    r2 = float(np.dot(R, R))
    pref = 2.0 * math.sqrt(mu / math.pi)

    # R^n_{000} for n = 0..order.
    base = {
        (0, 0, 0, n): ((-2.0 * mu) ** n) * pref * _boys(n, mu * r2)
        for n in range(order + 1)
    }

    def R_tuv(t: int, u: int, v: int, n: int) -> float:
        if t < 0 or u < 0 or v < 0:
            return 0.0
        key = (t, u, v, n)
        if key in base:
            return base[key]
        # Reduce the highest nonzero Cartesian index (order matters not;
        # the ladder is exact in every reduction order).
        if t > 0:
            val = (t - 1) * R_tuv(t - 2, u, v, n + 1) + R[0] * R_tuv(
                t - 1, u, v, n + 1
            )
        elif u > 0:
            val = (u - 1) * R_tuv(t, u - 2, v, n + 1) + R[1] * R_tuv(
                t, u - 1, v, n + 1
            )
        else:
            val = (v - 1) * R_tuv(t, u, v - 2, n + 1) + R[2] * R_tuv(
                t, u, v - 1, n + 1
            )
        base[key] = val
        return val

    return R_tuv(gamma[0], gamma[1], gamma[2], 0)


def screened_multipole_interaction_tensor(
    L_max_A: int,
    L_max_B: int,
    R: np.ndarray,
    mu: float,
) -> np.ndarray:
    """``T_{l1,m1; l2,m2}(R)`` for the SMOOTH kernel ``erf(sqrt(mu)·r)/r``.

    Identical bipolar Taylor assembly to
    :func:`multipole_interaction_tensor` (Cartesian derivatives +
    spherical basis change), with the bare ``1/|R|`` derivatives replaced
    by the Boys-ladder derivatives of the erf-screened kernel
    (:func:`_cartesian_derivative_erf_r`). Physically: the interaction
    tensor of two point multipoles through the long-range half of an
    Ewald split with splitting parameter ``sqrt(mu)``.

    The short-range (erfc) tensor follows by linearity as
    ``T_bare - T_erf`` -- use :func:`sr_multipole_interaction_tensor`.
    In the BIPOLE Ewald-split gauge that difference is the object a
    quartet-level bipolar far field must contract with pair moments:
    for two non-penetrating smeared distributions the effective
    ``mu`` is ``1/mu = 1/gamma_bra + 1/gamma_ket + 1/omega^2`` (the
    same reduced-exponent convention used by the SR pad and the
    charge-pair Schwarz screening envelopes).
    """
    from math import factorial as _factorial

    R = np.asarray(R, dtype=float).reshape(3)
    if not mu > 0.0:
        raise ValueError(f"mu must be positive; got {mu!r}")

    L_max = max(L_max_A, L_max_B)
    if L_max > 4:
        raise ValueError("L_max > 4 not supported (hexadecapole)")

    from .bipole_cell_moments import cartesian_component_indices as _cart_idx
    from ._cart_to_sph import cartesian_to_spherical_matrix as _C_mat

    indices = _cart_idx(L_max)
    n_cart = len(indices)
    T_cart = np.zeros((n_cart, n_cart), dtype=float)

    for ia, (i1, j1, k1) in enumerate(indices):
        for ib, (i2, j2, k2) in enumerate(indices):
            L = i1 + i2 + j1 + j2 + k1 + k2
            # Must match multipole_interaction_tensor exactly; see
            # _MAX_BIPOLAR_TOTAL_ORDER.
            if L > L_max or L > _MAX_BIPOLAR_TOTAL_ORDER:
                continue
            gamma = (i1 + i2, j1 + j2, k1 + k2)
            D_val = _cartesian_derivative_erf_r(gamma, R, mu)
            sign = (-1.0) ** (i1 + j1 + k1)
            denom = (
                _factorial(i1) * _factorial(j1) * _factorial(k1)
                * _factorial(i2) * _factorial(j2) * _factorial(k2)
            )
            T_cart[ia, ib] = sign * D_val / denom

    C = _C_mat(L_max)
    C_pinv = np.linalg.pinv(C)
    T_sph = C_pinv.T @ T_cart @ C_pinv

    n_A = n_components(L_max_A)
    n_B = n_components(L_max_B)
    T = np.zeros((n_A, n_B), dtype=float)
    T[:n_A, :n_B] = T_sph[:n_A, :n_B]
    return T


def sr_multipole_interaction_tensor(
    L_max_A: int,
    L_max_B: int,
    R: np.ndarray,
    mu: float,
) -> np.ndarray:
    """``T`` for the SHORT-RANGE kernel ``erfc-like`` ``1/r - erf(sqrt(mu)·r)/r``.

    By linearity of the bipolar Taylor assembly this is exactly
    ``multipole_interaction_tensor - screened_multipole_interaction_tensor``.
    This is the tensor a BIPOLE quartet-level bipolar far field contracts
    with pair moments in the Ewald-split gauge (the reciprocal J_LR
    channel already carries the erf half exactly), with the per-quartet
    effective ``1/mu = 1/gamma_bra + 1/gamma_ket + 1/omega^2``.
    """
    return multipole_interaction_tensor(
        L_max_A, L_max_B, R
    ) - screened_multipole_interaction_tensor(L_max_A, L_max_B, R, mu)


def multipole_pair_energy(
    M_A: np.ndarray,
    M_B: np.ndarray,
    R: np.ndarray,
    L_max_A: Optional[int] = None,
    L_max_B: Optional[int] = None,
) -> float:
    """Coulomb interaction energy ``E = M_A^T . T(R) . M_B`` between two
    distributions characterised by their multipole moments.

    Parameters
    ----------
    M_A, M_B : ndarray of shape (n_components(L_max),)
        Multipole moments in the ``lm_index`` flat layout.
    R : array_like shape (3,)
        Separation vector ``O^B - O^A``.
    L_max_A, L_max_B : int, optional
        Maximum l per side. If None, inferred from the moment-array shapes.

    Returns
    -------
    float
        Pairwise multipole-multipole Coulomb energy.
    """
    M_A = np.asarray(M_A, dtype=float).reshape(-1)
    M_B = np.asarray(M_B, dtype=float).reshape(-1)
    if L_max_A is None:
        L_max_A = int(round(math.sqrt(M_A.size))) - 1
        if (L_max_A + 1) ** 2 != M_A.size:
            raise ValueError(f"M_A size {M_A.size} is not (L_max+1)^2")
    if L_max_B is None:
        L_max_B = int(round(math.sqrt(M_B.size))) - 1
        if (L_max_B + 1) ** 2 != M_B.size:
            raise ValueError(f"M_B size {M_B.size} is not (L_max+1)^2")
    T = multipole_interaction_tensor(L_max_A, L_max_B, R)
    return float(M_A @ T @ M_B)
