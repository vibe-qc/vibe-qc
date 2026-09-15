"""Cartesian -> Stone-convention spherical multipole moment conversion.

For use by the BIPOLE multipole far-field builder.  The libint
``emultipole{1,2,3}`` output is in Cartesian form with (L_max+1)(L_max+2)
(L_max+3)/6 components.  The ``multipole_interaction_tensor`` from
:mod:`bipole_multipole` expects real spherical harmonic moments with
(L_max+1)^2 components.  This module bridges the two.

Supports L_max up to 12.  For L\u22644, analytical coefficients from the
real solid harmonic polynomials are used.  For L\u22655, a numerical
pseudoinverse is computed from the solid harmonics evaluated at a
regular point set.
"""

from __future__ import annotations

import numpy as np

from .bipole_cell_moments import cartesian_component_indices
from .bipole_multipole import lm_index, n_components

__all__ = ["cartesian_to_spherical_matrix"]


def cartesian_to_spherical_matrix(L_max: int) -> "np.ndarray":
    """Conversion matrix C: M_sph = C @ M_cart.

    Supports L_max in {0, 1, 2, 3, 4}.  Returns (n_sph, n_cart).
    For L_max = 4 the coefficients are computed numerically from the
    known real solid harmonics evaluated at a regular point set.
    """
    n_cart = len(cartesian_component_indices(L_max))
    n_sph = n_components(L_max)
    C = np.zeros((n_sph, n_cart), dtype=float)

    idx = cartesian_component_indices(L_max)

    def _cart_pos(i, j, k):
        return idx.index((i, j, k))

    def _set(l, m, cart_idx, coeff):
        C[lm_index(l, m), cart_idx] = coeff

    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)

    # L=0: monopole -- Z_00 = 1
    _set(0, 0, _cart_pos(0, 0, 0), 1.0)

    if L_max >= 1:
        # Z_1,-1 = -√2.y
        _set(1, -1, _cart_pos(0, 1, 0), -sqrt2)
        # Z_1,0 = z
        _set(1, 0, _cart_pos(0, 0, 1), 1.0)
        # Z_1,+1 = √2.x
        _set(1, 1, _cart_pos(1, 0, 0), sqrt2)

    if L_max >= 2:
        inv_2sqrt2 = 1.0 / (2.0 * sqrt2)
        half_sqrt3 = sqrt3 / 2.0
        # Z_2,-2 = √3.xy
        _set(2, -2, _cart_pos(1, 1, 0), sqrt3)
        # Z_2,-1 = √3.yz
        _set(2, -1, _cart_pos(0, 1, 1), sqrt3)
        # Z_2,0 = (2zz - xx - yy) / (2√2)
        _set(2, 0, _cart_pos(2, 0, 0), -inv_2sqrt2)
        _set(2, 0, _cart_pos(0, 2, 0), -inv_2sqrt2)
        _set(2, 0, _cart_pos(0, 0, 2), 2.0 * inv_2sqrt2)
        # Z_2,+1 = √3.xz
        _set(2, 1, _cart_pos(1, 0, 1), sqrt3)
        # Z_2,+2 = (√3/2).(xx - yy)
        _set(2, 2, _cart_pos(2, 0, 0), half_sqrt3)
        _set(2, 2, _cart_pos(0, 2, 0), -half_sqrt3)

    if L_max >= 3:
        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        half = 0.5

        # Z_3,-3 = (1/2).y.(3x^2 - y^2)  ->  (3/2).xxy - (1/2).yyy
        _set(3, -3, _cart_pos(2, 1, 0), 3.0 * half)
        _set(3, -3, _cart_pos(0, 3, 0), -half)

        # Z_3,-2 = (1/2).2.xyz = xyz
        _set(3, -2, _cart_pos(1, 1, 1), 1.0)

        # Z_3,-1 = (1/2).y.(4z^2 - x^2 - y^2)
        _set(3, -1, _cart_pos(2, 1, 0), -half)
        _set(3, -1, _cart_pos(0, 3, 0), -half)
        _set(3, -1, _cart_pos(0, 1, 2), 2.0)

        # Z_3,0 = [z^3 - (3/2).x^2z - (3/2).y^2z] / √3
        _set(3, 0, _cart_pos(2, 0, 1), -1.5 * inv_sqrt3)
        _set(3, 0, _cart_pos(0, 2, 1), -1.5 * inv_sqrt3)
        _set(3, 0, _cart_pos(0, 0, 3), 1.0 * inv_sqrt3)

        # Z_3,+1 = (1/2).x.(4z^2 - x^2 - y^2)
        _set(3, 1, _cart_pos(3, 0, 0), -half)
        _set(3, 1, _cart_pos(1, 2, 0), -half)
        _set(3, 1, _cart_pos(1, 0, 2), 2.0)

        # Z_3,+2 = (1/2).z.(x^2 - y^2)
        _set(3, 2, _cart_pos(2, 0, 1), half)
        _set(3, 2, _cart_pos(0, 2, 1), -half)

        # Z_3,+3 = (1/2).x.(x^2 - 3y^2)
        _set(3, 3, _cart_pos(3, 0, 0), half)
        _set(3, 3, _cart_pos(1, 2, 0), -3.0 * half)

    if L_max >= 4:
        # L=4 conversion (hexadecapole).  Coefficients derived from the
        # real solid harmonics in the Stone convention (Schmidt-semi-
        # normalised, √(4π/(2l+1)) factor absorbed) following the same
        # pattern: for |m| > 0 the leading factor is √(l+1)/2^{l-1}
        # = √5/8 for l=4.  The polynomial forms are constructed from
        # the Racah-normalised real spherical harmonics Y_{4,m}^R,
        # validated against the regular-solid-harmonic recurrence
        # Z_{l+1,m} = z·Z_{l,m} - ... in test_bipole_multipole.py.

        sqrt5 = np.sqrt(5.0)
        fac = sqrt5 / 8.0  # √(l+1) / 2^{l-1} for l=4

        # Z_{4,-4} = 4·fac·xy(x^2 - y^2)
        #           = 4·fac·(x^3 y - x y^3)
        _set(4, -4, _cart_pos(3, 1, 0), 4.0 * fac)
        _set(4, -4, _cart_pos(1, 3, 0), -4.0 * fac)

        # Z_{4,-3} = 4·fac·yz(3x^2 - y^2)
        #           = 12·fac·x^2 y z - 4·fac·y^3 z
        _set(4, -3, _cart_pos(2, 1, 1), 12.0 * fac)
        _set(4, -3, _cart_pos(0, 3, 1), -4.0 * fac)

        # Z_{4,-2} = 2·fac·xy(6z^2 - x^2 - y^2)
        #           = 12·fac·x y z^2 - 2·fac·x^3 y - 2·fac·x y^3
        _set(4, -2, _cart_pos(1, 1, 2), 12.0 * fac)
        _set(4, -2, _cart_pos(3, 1, 0), -2.0 * fac)
        _set(4, -2, _cart_pos(1, 3, 0), -2.0 * fac)

        # Z_{4,-1} = y · [3z·(2z^2 - 3(x^2+y^2))] · (4√5/?? )
        # Actually: Z_{4,-1} = 4·fac·y·(4z^3 - 3z(x^2+y^2))
        #           = 16·fac·y z^3 - 12·fac·x^2 y z - 12·fac·y^3 z
        _set(4, -1, _cart_pos(0, 1, 3), 16.0 * fac)
        _set(4, -1, _cart_pos(2, 1, 1), -12.0 * fac)
        _set(4, -1, _cart_pos(0, 3, 1), -12.0 * fac)

        # Z_{4,0} = (8z^4 - 24z^2(x^2+y^2) + 3(x^4 + 2x^2y^2 + y^4)) / 8
        # No √5 factor for m=0 (the zonal harmonic).
        _set(4, 0, _cart_pos(0, 0, 4), 1.0)
        _set(4, 0, _cart_pos(2, 0, 2), -3.0)
        _set(4, 0, _cart_pos(0, 2, 2), -3.0)
        _set(4, 0, _cart_pos(4, 0, 0), 0.375)
        _set(4, 0, _cart_pos(2, 2, 0), 0.75)
        _set(4, 0, _cart_pos(0, 4, 0), 0.375)

        # Z_{4,+1} = 4·fac·x·(4z^3 - 3z(x^2+y^2))
        _set(4, 1, _cart_pos(1, 0, 3), 16.0 * fac)
        _set(4, 1, _cart_pos(3, 0, 1), -12.0 * fac)
        _set(4, 1, _cart_pos(1, 2, 1), -12.0 * fac)

        # Z_{4,+2} = 2·fac·(x^2 - y^2)·(6z^2 - x^2 - y^2)
        #           = 12·fac·(x^2 z^2 - y^2 z^2) - 2·fac·(x^4 - y^4)
        _set(4, 2, _cart_pos(2, 0, 2), 12.0 * fac)
        _set(4, 2, _cart_pos(0, 2, 2), -12.0 * fac)
        _set(4, 2, _cart_pos(4, 0, 0), -2.0 * fac)
        _set(4, 2, _cart_pos(0, 4, 0), 2.0 * fac)

        # Z_{4,+3} = 4·fac·xz(x^2 - 3y^2)
        #           = 4·fac·x^3 z - 12·fac·x y^2 z
        _set(4, 3, _cart_pos(3, 0, 1), 4.0 * fac)
        _set(4, 3, _cart_pos(1, 2, 1), -12.0 * fac)

        # Z_{4,+4} = 2·fac·(x^4 - 6x^2y^2 + y^4)
        _set(4, 4, _cart_pos(4, 0, 0), 2.0 * fac)
        _set(4, 4, _cart_pos(2, 2, 0), -12.0 * fac)
        _set(4, 4, _cart_pos(0, 4, 0), 2.0 * fac)

    if L_max >= 5:
        # Numerical pseudoinverse for L >= 5.
        _extend_numerical(C, L_max)

    return C


def _extend_numerical(C: np.ndarray, L_max: int) -> None:
    """Extend the conversion matrix for L >= 5 using numerical fitting.

    Uses the real-solid-harmonic recurrence to compute Z_{l,m}(x,y,z)
    directly from Cartesian monomials, without calling back into
    cartesian_to_spherical_matrix (which would recurse).
    """
    from .bipole_multipole import lm_index, n_components

    n_cart = len(cartesian_component_indices(L_max))
    n_sph = n_components(L_max)

    # Generate Fibonacci sphere points.
    n_pts = max(500, 50 * L_max * L_max)
    points = np.zeros((n_pts, 3), dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_pts):
        y = 1.0 - (i / float(n_pts - 1)) * 2.0
        radius = np.sqrt(1.0 - y * y)
        theta = phi * i
        points[i, 0] = np.cos(theta) * radius
        points[i, 1] = y
        points[i, 2] = np.sin(theta) * radius

    # Evaluate Cartesian monomials at each point.
    cart_idx = cartesian_component_indices(L_max)
    M_cart = np.zeros((n_pts, n_cart), dtype=float)
    for p in range(n_pts):
        x, y, z = points[p]
        for c, (i, j, k) in enumerate(cart_idx):
            M_cart[p, c] = (x ** i) * (y ** j) * (z ** k)

    # Compute solid harmonics via recurrence: Z_{l,m} = r^l * Y_{l,m}(theta,phi)
    # Use the Cartesian representation from the known conversion matrix for L<=4
    # and extend via the polynomial recurrence.
    # For simplicity: compute solid harmonics numerically using the Stone
    # convention polynomial forms evaluated at each point.
    M_sph = np.zeros((n_pts, n_sph), dtype=float)
    for p in range(n_pts):
        x, y, z = points[p]
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                s = lm_index(l, m)
                if s < n_sph:
                    M_sph[p, s] = _solid_harmonic_direct(l, m, x, y, z)

    # Solve: M_sph = M_cart @ C^T → C^T = pinv(M_cart) @ M_sph
    C_new_T = np.linalg.pinv(M_cart) @ M_sph
    C_new = C_new_T.T

    # Overwrite rows for L >= 5, keeping analytical L=0-4.
    for l in range(5, L_max + 1):
        for m in range(-l, l + 1):
            s = lm_index(l, m)
            if s < n_sph:
                C[s, :] = C_new[s, :]


def _solid_harmonic_direct(l: int, m: int, x: float, y: float, z: float) -> float:
    """Real solid harmonic Z_{l,m}(x,y,z) in Stone convention.

    Uses the Cartesian polynomial representation directly.
    For low l, uses the analytical forms; for higher l, uses the recurrence
    Z_{l+1,m} = c1*z*Z_{l,m} - c2*r^2*Z_{l-1,m}.
    """
    # Pre-computed forms for l=0..4 match the analytical coefficients.
    # We only need this for the numerical fit, so approximate is OK.
    r2 = x*x + y*y + z*z
    if l == 0:
        return 1.0 if m == 0 else 0.0
    if l == 1:
        if m == -1: return -np.sqrt(2.0) * y
        if m == 0: return z
        if m == 1: return np.sqrt(2.0) * x
        return 0.0
    if l == 2:
        if m == -2: return np.sqrt(3.0) * x * y
        if m == -1: return np.sqrt(3.0) * y * z
        if m == 0: return (2*z*z - x*x - y*y) / (2.0 * np.sqrt(2.0))
        if m == 1: return np.sqrt(3.0) * x * z
        if m == 2: return np.sqrt(3.0) / 2.0 * (x*x - y*y)
        return 0.0
    # For l >= 3, use the recurrence from the analytical forms.
    # The recurrence for Stone-convention solid harmonics:
    # Z_{l,m} = N_{l,m} * P_{l}^{|m|}(cos theta) * R_{m}(phi) * r^l
    # where R_m(phi) = cos(m*phi) for m>=0, sin(|m|*phi) for m<0.
    # We compute this via spherical coordinates.
    if r2 < 1e-30:
        return 0.0
    r = np.sqrt(r2)
    ct = z / r  # cos(theta)
    st = np.sqrt(max(0.0, 1.0 - ct*ct))  # sin(theta)
    # Azimuthal part.
    if m >= 0:
        if st < 1e-15 and m > 0:
            Rm = 0.0
        else:
            # cos(m*phi) = Re((x + i*y)^m / (r*st)^m)
            # Compute via Chebyshev or direct formula.
            phi = np.arctan2(y, x) if st > 1e-15 else 0.0
            Rm = np.cos(m * phi)
    else:
        if st < 1e-15:
            Rm = 0.0
        else:
            phi = np.arctan2(y, x) if st > 1e-15 else 0.0
            Rm = np.sin(abs(m) * phi)
    # Legendre polynomial P_l^{|m|}(ct).
    P = _legendre(l, abs(m), ct)
    # Normalisation for Stone convention (Schmidt semi-normalised):
    # N_{l,m} = sqrt(2 * (l-|m|)! / (l+|m|)!) for m!=0, 1 for m=0.
    # With the sqrt(4pi/(2l+1)) factor absorbed.
    if m == 0:
        norm = 1.0
    else:
        from math import factorial
        norm = np.sqrt(2.0 * factorial(l - abs(m)) / factorial(l + abs(m)))
    return norm * P * Rm * (r ** l)


def _legendre(l: int, m: int, x: float) -> float:
    """Associated Legendre polynomial P_l^m(x)."""
    # Use recurrence: (l-m)P_l^m = x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m
    if m > l:
        return 0.0
    if l == m:
        # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^{m/2}
        from math import factorial
        val = 1.0
        for i in range(1, 2*m, 2):
            val *= i
        return val * ((1.0 - x*x) ** (m / 2.0)) * (1.0 if m % 2 == 0 else -1.0)
    if l == m + 1:
        # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
        return x * (2*m + 1) * _legendre(m, m, x)
    # Recurrence.
    return (x * (2*l - 1) * _legendre(l-1, m, x) - (l + m - 1) * _legendre(l-2, m, x)) / (l - m)
