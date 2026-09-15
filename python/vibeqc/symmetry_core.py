"""Phase SYM1: real-basis Wigner D-matrices for AO rotation.

When a 3D rotation R acts on a spherical-harmonic basis function of
angular momentum l, it mixes the (2l+1) functions of the same l
block-diagonally:

    R̂ . Y_{l, m}(r̂)  =  S_{m'}  D^l_{m', m}(R) . Y_{l, m'}(r̂)

``D^l(R)`` is the Wigner D-matrix of the rotation at angular momentum
l. Together with the spatial permutation of atom centers under the
full space-group operator ``(R, t)``, this gives the representation
of the space group on the AO basis -- the building block for every
symmetry-exploitation phase downstream (SYM2 real-space matrix
reduction, SYM4 IBZ Fock build, SYM5 symmetrized gradients).

vibe-qc forces ``set_pure(true)`` on every :class:`BasisSet`, so the
basis is always in real spherical harmonics. The Wigner D-matrices
we need are therefore real orthogonal (not unitary) -- we build them
by going through the complex representation and applying the
standard real<->complex unitary transform.

Ordering convention
-------------------

Within each shell the (2l+1) AOs are ordered by magnetic quantum
number from -l to +l ascending:

    Y_{l, -l},  Y_{l, -l+1},  ...,  Y_{l, 0},  ...,  Y_{l, l-1},  Y_{l, l}

This matches libint's pure-spherical ordering. The real spherical
harmonics themselves follow the Condon-Shortley convention:

    Y_{l, 0}^R  =  Y_{l, 0}^C                              (already real)
    Y_{l, m}^R  =  ((-1)^m / √2) . (Y_{l,  m}^C + Y_{l, -m}^C)   (m > 0)
    Y_{l,-m}^R  =  ((-1)^m / (i√2)) . (Y_{l, -m}^C - Y_{l, m}^C) (m > 0)

The public API is one function -- :func:`wigner_d_real` -- plus the
Euler-angle extractor and the real<->complex basis transform so users
can inspect each piece independently.

Validation
----------

Three identities, all exercised in ``tests/test_symmetry_core.py``:

* ``D^l(I) = I`` -- identity rotation gives the identity matrix.
* ``D^l(R).D^l(R)ᵀ = I`` -- real orthogonality.
* ``D^l(R₁ R₂) = D^l(R₁).D^l(R₂)`` -- representation homomorphism.

Plus direct matrix-element checks for:

* ``D^1(R) = P . R . Pᵀ`` where P is the permutation that puts the
  real-spherical (y, z, x) basis into the Cartesian (x, y, z) order
  our ``R`` uses. This is the classical "p orbitals rotate as the
  Cartesian vector does".
* Rotation by 2pi around any axis returns the identity for any l.
"""

from __future__ import annotations

import math
from math import factorial
from typing import Tuple

import numpy as np


__all__ = [
    "wigner_d_real",
    "wigner_d_complex",
    "wigner_small_d",
    "euler_angles_from_rotation",
    "real_spherical_to_complex_unitary",
]


# ---------------------------------------------------------------------------
# Euler-angle extraction (Z-Y-Z convention)
# ---------------------------------------------------------------------------

def euler_angles_from_rotation(
    R: np.ndarray,
) -> Tuple[float, float, float]:
    """Extract ZYZ Euler angles (a, b, g) from a **proper** rotation R
    (det = +1). Improper rotations (det = -1) are handled one level up
    in :func:`wigner_d_real` by factoring off the inversion.

    Convention: ``R = R_z(a) . R_y(b) . R_z(g)``. Returned angles
    lie in ``a in (-pi, pi]``, ``b in [0, pi]``, ``g in (-pi, pi]``.

    This is the convention most directly usable by the Wigner
    D-matrix: ``D^l(R) = D^l_z(a) D^l_y(b) D^l_z(g)`` with the two
    z-rotations trivially diagonal in the complex basis.

    At the b = 0 / b = pi "gimbal lock" poles a and g are not
    independently determined -- we set g = 0 and put all the z-rotation
    into a.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got {R.shape}")
    # Tolerance for proper-rotation sanity check.
    if abs(np.linalg.det(R) - 1.0) > 1e-6:
        raise ValueError(
            f"not a proper rotation (det = {np.linalg.det(R):.6f}); "
            "pass improper rotations to wigner_d_real, which factors "
            "off the inversion before calling this extractor"
        )

    # Extract from the R.e_z = (R_xz, R_yz, R_zz) column.
    # b = arccos(R_zz), clamped to avoid domain errors from roundoff.
    r33 = max(-1.0, min(1.0, R[2, 2]))
    beta = math.acos(r33)

    if abs(math.sin(beta)) < 1e-12:
        # b near 0 or pi -- gimbal lock; a and g are not independently
        # determined. Set g = 0 and put the whole z-rotation into a.
        # The two poles use different algebra:
        #   b ≈ 0 (R[2,2] ≈ +1):  R ≈ R_z(a + g). With g = 0,
        #     R[0,0] = cos a,   R[1,0] = sin a.
        #   b ≈ pi (R[2,2] ≈ -1):  R ≈ R_z(a - g) . R_y(pi). With g = 0,
        #     R[0,0] = -cos a,  R[1,0] = -sin a.
        # Disambiguating by the sign of R[2,2] keeps the extractor
        # correct at both poles.
        if R[2, 2] > 0.0:
            alpha = math.atan2(R[1, 0], R[0, 0])
        else:
            alpha = math.atan2(-R[1, 0], -R[0, 0])
        gamma = 0.0
    else:
        # Standard ZYZ extraction.
        alpha = math.atan2(R[1, 2], R[0, 2])
        gamma = math.atan2(R[2, 1], -R[2, 0])

    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# Complex Wigner D: small d (b-rotation) and full D (a, b, g)
# ---------------------------------------------------------------------------

def wigner_small_d(l: int, beta: float) -> np.ndarray:
    """Wigner "small d" matrix ``d^l_{m', m}(b)``.

    This is the rotation around the body y-axis in the complex
    spherical-harmonic basis -- a purely real (2l+1)x(2l+1) matrix.

    Formula (Wigner):

        d^l_{m'm}(b) = S_s (-1)^{m'-m+s} .
                        √((l+m')!(l-m')!(l+m)!(l-m)!) /
                        ((l+m-s)!(m'-m+s)! s! (l-m'-s)!)
                      . cos(b/2)^{2l-2s+m-m'} . sin(b/2)^{2s+m'-m}

    Sum over all s for which the factorial denominators are
    non-negative. For stability with large l we accumulate in
    ``float64`` -- factorials blow up around l ≈ 20 so this breaks for
    g-functions and beyond (l >= 5 is rare in practice).
    """
    if l < 0:
        raise ValueError(f"wigner_small_d: l must be >= 0, got {l}")
    n = 2 * l + 1
    d = np.zeros((n, n), dtype=float)
    cb = math.cos(beta / 2.0)
    sb = math.sin(beta / 2.0)

    for i in range(n):
        mp = i - l                   # m'
        for j in range(n):
            m = j - l                # m
            s_min = max(0, m - mp)
            s_max = min(l - mp, l + m)
            total = 0.0
            for s in range(s_min, s_max + 1):
                sign = (-1.0) ** (mp - m + s)
                num = math.sqrt(
                    factorial(l + mp) * factorial(l - mp)
                    * factorial(l + m) * factorial(l - m)
                )
                den = (factorial(l + m - s) * factorial(mp - m + s)
                       * factorial(s) * factorial(l - mp - s))
                # cos/sin exponents -- both >= 0 in the valid s range.
                cos_pow = 2 * l - 2 * s + m - mp
                sin_pow = 2 * s + mp - m
                term = sign * (num / den) * (cb ** cos_pow) * (sb ** sin_pow)
                total += term
            d[i, j] = total
    return d


def wigner_d_complex(
    l: int, alpha: float, beta: float, gamma: float,
) -> np.ndarray:
    """Full complex Wigner D-matrix in the ZYZ convention.

    ``D^l_{m',m}(a, b, g) = e^{-i m' a} . d^l_{m',m}(b) . e^{-i m g}``

    Returns a ``(2l+1, 2l+1)`` complex numpy array, with the rows
    indexed by m' and the columns by m, both running from -l to +l
    ascending.
    """
    small = wigner_small_d(l, beta)
    n = 2 * l + 1
    D = np.zeros((n, n), dtype=complex)
    # Phase pre-tables.
    ms = np.arange(-l, l + 1)
    row_phase = np.exp(-1j * ms * alpha)   # e^{-i m' a}, applied per row
    col_phase = np.exp(-1j * ms * gamma)   # e^{-i m g}, applied per column
    for i in range(n):
        for j in range(n):
            D[i, j] = row_phase[i] * small[i, j] * col_phase[j]
    return D


# ---------------------------------------------------------------------------
# Real <-> complex basis transform
# ---------------------------------------------------------------------------

def real_spherical_to_complex_unitary(l: int) -> np.ndarray:
    """Unitary matrix U mapping real spherical to complex spherical
    harmonics at angular momentum l.

    Derivation (Condon-Shortley convention, (-1)^m prefactor on the
    real harmonics for the "racah" sign convention used by most
    quantum chemistry codes and matched by libint):

    Starting from the identity ``(Y_{l,m}^C)* = (-1)^m . Y_{l,-m}^C``:

        Re[Y_{l,m}^C] = 1/2.(Y_{l,m}^C + (-1)^m . Y_{l,-m}^C)
        Im[Y_{l,m}^C] = (1/(2i)).(Y_{l,m}^C - (-1)^m . Y_{l,-m}^C)

    and defining the real spherical harmonics (m > 0):

        Y_{l, 0}^R  =  Y_{l, 0}^C
        Y_{l, m}^R  =  √2 . (-1)^m . Re[Y_{l, m}^C]
                   =  ((-1)^m / √2) . Y_{l, m}^C  +  (1 / √2) . Y_{l, -m}^C
        Y_{l,-m}^R  =  √2 . (-1)^m . Im[Y_{l, m}^C]
                   =  (-i(-1)^m / √2) . Y_{l, m}^C  +  (i / √2) . Y_{l,-m}^C

    Writing ``Y^R = U . Y^C`` where both vectors are column-ordered
    ``m = -l, -l+1, ..., +l`` ascending, U's entries follow directly
    from the formulas above.

    Then a complex operator D^C on complex harmonics pulls back to
    real harmonics as ``D^R = U . D^C . U^+`` (unitary change of basis).
    """
    n = 2 * l + 1
    U = np.zeros((n, n), dtype=complex)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for idx_real in range(n):
        m = idx_real - l   # signed m for the real basis index
        if m == 0:
            # Y_{l,0}^R = Y_{l,0}^C  -- identity entry on the diagonal.
            U[idx_real, l] = 1.0
        elif m > 0:
            # Y_{l, m}^R = ((-1)^m / √2) . Y_{l, m}^C  +  (1 / √2) . Y_{l,-m}^C
            sign = (-1.0) ** m
            U[idx_real, l + m] = sign * inv_sqrt2
            U[idx_real, l - m] = inv_sqrt2
        else:
            # m < 0; write am = |m| > 0.
            # Y_{l,-am}^R = (-i(-1)^am / √2) . Y_{l, am}^C  +  (i / √2) . Y_{l,-am}^C
            am = -m
            sign = (-1.0) ** am
            U[idx_real, l + am] = -1j * sign * inv_sqrt2
            U[idx_real, l - am] =  1j * inv_sqrt2
    return U


# ---------------------------------------------------------------------------
# Top-level: real Wigner D-matrix for a rotation matrix
# ---------------------------------------------------------------------------

def wigner_d_real(l: int, R: np.ndarray) -> np.ndarray:
    """Real orthogonal ``(2l+1)x(2l+1)`` Wigner D-matrix for the
    rotation R, acting on real spherical harmonics.

    Accepts both **proper** rotations (det R = +1: rotations about an
    axis) and **improper** rotations (det R = -1: reflections,
    inversions, rotoreflections). Improper R is factored as
    ``R = i . R_proper`` where ``i`` is the inversion and
    ``R_proper = -R`` has det = +1. Under inversion the real solid
    harmonic ``Y_{l,m}(r̂)`` transforms as ``(-1)^l Y_{l,m}(r̂)``, so

        D^l(R_improper) = (-1)^l . D^l(-R_improper)

    Pipeline for the (possibly factored) proper rotation:

    1. Extract ZYZ Euler angles (a, b, g) from R_proper.
    2. Compute the complex Wigner D-matrix D^C = D^l(a, b, g).
    3. Transform to the real basis: D^R = conj(U) . D^C . U^T, where
       U = :func:`real_spherical_to_complex_unitary`.

    The imaginary part of the final product is numerical noise
    (below ~1e-14 for l <= 4); we return the real part explicitly so
    callers don't have to care.

    Special-case ``l == 0`` shortcut: D^0 is trivially ±1 (sign
    depends on parity under inversion -- but Y_{0,0} is even-parity,
    so D^0(i) = +1, making D^0(R) = [[1]] regardless of R).

    Rejects matrices with det notin {+1, -1} (scalings, shears) with a
    clear error message.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")

    det = float(np.linalg.det(R))
    if l == 0:
        # Y_{0,0} ∝ 1 / √(4pi) has even parity under inversion.
        return np.array([[1.0]])

    # Classify and factor off the inversion if improper.
    if abs(det - 1.0) < 1e-6:
        R_proper = R
        inversion_sign = 1.0
    elif abs(det + 1.0) < 1e-6:
        R_proper = -R
        inversion_sign = (-1.0) ** l
    else:
        raise ValueError(
            f"R is not a rotation (det = {det:.6f}; must be ±1). "
            f"Expected a 3x3 orthogonal matrix; got a scaling or shear."
        )

    alpha, beta, gamma = euler_angles_from_rotation(R_proper)
    D_complex = wigner_d_complex(l, alpha, beta, gamma)
    U = real_spherical_to_complex_unitary(l)

    # Basis-change formula for operator matrix elements between two
    # orthonormal bases related by |new ν> = S_m U_{νm} |old m>:
    #
    #     D^new_{ν, mu} = <new ν| O |new mu>
    #                  = S_{m', m} U^*_{ν, m'} D^old_{m', m} U_{mu, m}
    #
    # In matrix form D^new = U^* . D^old . U^T.
    #
    # The commonly seen "U . O . U^+" is a similarity transform
    # (rotating the operator while holding the basis fixed); here we
    # want the change-of-basis transform, which uses U^* and U^T.
    # The two coincide only when U is real.
    D_real_complex = inversion_sign * (np.conj(U) @ D_complex @ U.T)
    # Imag part should be at numerical-noise level; check softly and drop.
    imag_max = np.abs(D_real_complex.imag).max()
    if imag_max > 1e-8:
        raise RuntimeError(
            f"wigner_d_real: residual imaginary part {imag_max:.3e} "
            f"exceeds tolerance at l = {l}. This usually means the "
            f"input matrix isn't a proper rotation (check det(R) and "
            f"orthonormality) or the Euler-angle extraction hit a "
            f"degenerate case."
        )
    return D_real_complex.real
