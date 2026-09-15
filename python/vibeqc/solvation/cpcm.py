"""CPCM (Conductor-like Polarisable Continuum Model) -- core math (S1b).

The conductor-screening CPCM equations on a discrete cavity
tessellation are

    A q = - f(e) . V        with    f(e) = (e - 1) / e

where ``A`` is the (n_pts, n_pts) cavity self-interaction matrix,
``V`` is the gas-phase molecular electrostatic potential evaluated at
the cavity surface points, and ``q`` is the apparent surface charge.
The total solvation energy is

    E_solv = (1/2) S_i q_i V_i.

This module assembles ``A`` and solves for ``q``; the SCF wiring
lives in :mod:`vibeqc.solvation.driver`.

References
----------
* Klamt-Schüürmann 1993 -- original COSMO.
* Cossi-Rega-Scalmani-Barone 2003 -- CPCM matrix layout used here.
* Scalmani-Frisch 2010 -- diagonal element from continuous-surface-
  charge (CSC) formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The screening factor lives in one place; see that module for why it is
# carried on a ScreeningModel rather than re-derived at each consumer
# (#546, #548). Re-exported here because ``vibeqc.solvation.cpcm
# .dielectric_factor`` is the long-standing public spelling.
from .screening import ScreeningModel, dielectric_factor  # noqa: F401

# Scalmani-Frisch (2010) diagonal constant.
#
#     A_ii = (CPCM_DIAG_ALPHA x √(4pi)) / √(w_i)
#          = (1.0694 x √(4pi))   / √(w_i)
#
# where ``w_i`` is the actual surface area element at the cavity point
# (bohr^2, so the dimensional units work out: 1 / √(bohr^2) = bohr⁻¹).
# The numerical factor 1.0694 is the self-energy of a uniformly charged
# disk relative to the equivalent Gaussian -- see Scalmani-Frisch 2010
# eq. (11) and Cossi-Scalmani-Mennucci-Tomasi 2003 eq. (18). The
# baked-in √(4pi) makes the diagonal dominant over typical nearest-
# neighbour 1/r off-diagonals on Lebedev-discretised spheres, which
# the bare 1.0694/√w_i form fails for moderately dense tessellations
# (110+ points per sphere).
#
# Matches PySCF (pyscf/solvent/pcm.py::PCM.get_F0) and Q-Chem (CPCM
# default; see Stratmann-Scuseria-Frisch JCP 109 8218).
import math as _math

CPCM_DIAG_ALPHA_BARE = 1.0694
CPCM_DIAG_ALPHA = CPCM_DIAG_ALPHA_BARE * _math.sqrt(4.0 * _math.pi)

# ---------------------------------------------------------------------
# Surface-charge representation
# ---------------------------------------------------------------------
#
# Two kernels, and which one is valid depends on the cavity, not on taste.
#
# ``point``    A_ij = 1/r_ij, A_ii = C_S sqrt(4 pi / a_i).
#              The FIXPVA form (Su & Li; Lange & Herbert 2010, Table III).
#              Singular as r_ij -> 0, so it is only safe on a discretization
#              that keeps segments apart. Lange & Herbert are explicit that
#              this is a precondition rather than an accident: "The FIXPVA
#              approach uses point charges, and this choice necessitates the
#              use of an alternative switching function, as close approach of
#              these point charges must be avoided" (p. 244111-8).
#
# ``gaussian`` A_ij = erf(zeta_ij r_ij)/r_ij, A_ii = zeta_i sqrt(2/pi)/F_i.
#              York & Karplus, J. Phys. Chem. A 103, 11060 (1999),
#              doi:10.1021/jp992097l, eq. 56; diagonal from Lange & Herbert,
#              J. Chem. Phys. 133, 244111 (2010), doi:10.1063/1.3511297,
#              eq. 3.7, which is the r_ij -> 0 limit of eq. 56 (their eq. 3.6).
#              Segments carry normalized spherical Gaussians instead of point
#              charges, so the kernel is finite at contact and A is the Gram
#              matrix of those charge distributions under the Coulomb inner
#              product -- positive definite by construction, for any geometry.
#
# Positive definiteness is not a numerical nicety. For C-PCM the response
# matrix is Q = -f A^-1, and Lange & Herbert eq. 2.25 show the polarization
# energy is a minimum only if Q is negative definite, i.e. only if A is
# positive definite. An indefinite A means the reaction field can *raise* the
# energy: the solve is not the variational solution of anything (#744).
#
# The Gaussian width is not a free parameter. York & Karplus eq. 61 sets
# zeta_i = zeta / sqrt(w_i) on the unit sphere, scaling as zeta_i(R) =
# zeta_i(1)/R; with a_i = w_i R^2 F_i that is
#
#     zeta_i = zeta sqrt(F_i / a_i)
#
# which needs only the segment area and switching factor, so it transfers to a
# cavity that has no Lebedev weights at all. Equating the two diagonals above
# gives the exact correspondence
#
#     zeta = C_S pi sqrt(2)
#
# and that is a real cross-check rather than a definition: Lange & Herbert's
# Lebedev C_S = 1.104 yields zeta = 4.9049, while York & Karplus fitted
# zeta = 4.901 to 4.907 for 110 to 1202 points by an entirely different route
# (exact Born energy of a conductor plus a uniform surface charge). Two
# papers, two parameterizations, four matching digits.
POINT_CHARGE = "point"
GAUSSIAN_CHARGE = "gaussian"

# York & Karplus 1999, Table 1: zeta optimized per angular quadrature level
# for a unit sphere. Flat to +-0.1 percent above 110 points, which is why a
# cavity with no Lebedev grid (a CFC) can use the dense limit honestly.
YORK_KARPLUS_ZETA: dict[int, float] = {
    14: 4.865, 26: 4.855, 50: 4.893, 110: 4.901, 194: 4.903,
    302: 4.905, 434: 4.906, 590: 4.905, 770: 4.899, 974: 4.907,
    1202: 4.907,
}
ZETA_DENSE_LIMIT = 4.907


def york_karplus_zeta(n_points_per_sphere: int | None = None) -> float:
    """The Gaussian width parameter for a discretization level.

    Returns the tabulated value for a Lebedev point count, or the dense limit
    for a cavity that is not a Lebedev grid. Interpolating between table rows
    would be false precision: the entries vary by 0.1 percent above 110 points
    and York & Karplus describe zeta as "roughly constant" across schemes.
    """
    if n_points_per_sphere is None:
        return ZETA_DENSE_LIMIT
    return YORK_KARPLUS_ZETA.get(int(n_points_per_sphere), ZETA_DENSE_LIMIT)


def gaussian_exponents(
    areas: np.ndarray, switching: np.ndarray | None = None, *, zeta: float
) -> np.ndarray:
    """``zeta_i = zeta sqrt(F_i / a_i)`` -- York & Karplus eq. 61, in the
    area form of Lange & Herbert eq. 4.1 so it needs no Lebedev weights."""
    a = np.asarray(areas, dtype=np.float64)
    F = np.ones_like(a) if switching is None else np.asarray(
        switching, dtype=np.float64
    )
    if np.any(a <= 0.0):
        raise ValueError(
            "gaussian_exponents: every segment area must be positive; a "
            "zero-area segment has no Gaussian width."
        )
    return float(zeta) * np.sqrt(F / a)


def _native_cpcm():
    try:
        from vibeqc import _vibeqc_core as _core
    except ImportError:
        return None
    required = (
        "cpcm_build_A_matrix",
        "cpcm_build_capped_A_matrix",
        "cpcm_dielectric_factor",
        "cpcm_solve_apparent_charges",
    )
    if not all(hasattr(_core, name) for name in required):
        return None
    return _core


@dataclass(frozen=True)
class CPCMResult:
    """Outcome of one apparent-surface-charge solve.

    Attributes
    ----------
    q : ndarray, shape (n_pts,)
        Apparent surface charges, atomic units.
    V : ndarray, shape (n_pts,)
        Total molecular electrostatic potential (electron +
        nuclear) at the cavity points, atomic units.
    e_solv : float, Hartree
        Solvation energy E_solv = 1/2 S_i q_i V_i.
    epsilon : float
        Solvent dielectric constant used.
    """

    q: np.ndarray
    V: np.ndarray
    e_solv: float
    epsilon: float

    @property
    def total_charge(self) -> float:
        return float(self.q.sum())


def build_A_matrix(cavity_points: np.ndarray,
                   cavity_weights: np.ndarray,
                   *,
                   representation: str = POINT_CHARGE,
                   switching: np.ndarray | None = None,
                   zeta: float | None = None,
                   n_points_per_sphere: int | None = None) -> np.ndarray:
    """Assemble the CPCM cavity self-interaction matrix A.

    Two surface-charge representations, documented at the top of this module.
    ``representation="point"`` (the default, and what every Lebedev-cavity
    result in vibe-qc was computed with) uses the FIXPVA point-charge form:
    diagonal ``C_S sqrt(4 pi / a_i)``, off-diagonal ``1/r_ij``.
    ``representation="gaussian"`` uses the York-Karplus spherical Gaussians:
    diagonal ``zeta_i sqrt(2/pi) / F_i``, off-diagonal
    ``erf(zeta_ij r_ij) / r_ij``.

    Pick by construction, not by preference. The point-charge kernel is
    singular at contact and is only safe where the discretization keeps
    segments apart -- for a Lebedev cavity that is the erf switching function.
    A cavity with no switching function, such as a CFC, has no such
    protection, and there the point-charge kernel makes ``A`` indefinite for
    roughly one grid spacing in six (#744), which breaks the variational
    condition that makes the solve meaningful at all. Prefer
    :func:`build_cavity_A_matrix`, which reads the choice off the cavity so
    no call site has to remember this.

    All inputs in atomic units (bohr for distances, bohr^2 for
    weights). The returned matrix is symmetric, positive-definite --
    suitable for ``numpy.linalg.solve`` or a Cholesky.

    Memory is O(n_pts^2) doubles. For 240 cavity points that's
    ~440 KB; for 5000 cavity points that's ~190 MB -- typical of
    "tight Gaussian-default cavity on a 50-heavy-atom molecule".
    Use ``solve_apparent_charges`` to factor + solve in one shot.
    """
    pts = np.asarray(cavity_points, dtype=np.float64)
    w = np.asarray(cavity_weights, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"cavity_points must be (n_pts, 3); got {pts.shape}"
        )
    if w.shape != (pts.shape[0],):
        raise ValueError(
            f"cavity_weights must be (n_pts,); got {w.shape}"
        )
    if representation == GAUSSIAN_CHARGE:
        return _build_gaussian_A_matrix(
            pts, w, switching, zeta, n_points_per_sphere
        )
    if representation != POINT_CHARGE:
        raise ValueError(
            f"build_A_matrix: unknown representation {representation!r} "
            f"(use {POINT_CHARGE!r} or {GAUSSIAN_CHARGE!r})."
        )

    native = _native_cpcm()
    if native is not None:
        return np.asarray(native.cpcm_build_A_matrix(pts, w), dtype=np.float64)

    # Pairwise Euclidean distances (bohr) -- full NxN construction is
    # the same cost as the matvec and avoids a second loop later.
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    # Off-diagonal: 1/r_ij. Diagonal will overwrite shortly.
    with np.errstate(divide="ignore"):
        A = 1.0 / dist
    # FIXPVA self-energy on the diagonal: A_ii = C_S sqrt(4 pi / a_i).
    # Finite for the smallest-weight (most heavily-switched) points, so the
    # diagonal never blows up -- but a positive diagonal is not the same as a
    # positive-definite matrix, which is what #744 is about.
    np.fill_diagonal(A, CPCM_DIAG_ALPHA / np.sqrt(w))
    return A


def _build_gaussian_A_matrix(pts, w, switching, zeta, n_points_per_sphere):
    """York-Karplus Gaussian kernel: eq. 56 off-diagonal, its contact limit
    on the diagonal.

    ``A`` here is the Gram matrix of normalized spherical Gaussians under the
    Coulomb inner product, which is positive definite for any set of distinct
    centres -- the Coulomb form is positive definite on charge densities, by
    Parseval. Where the switching function attenuates a segment the diagonal
    is *inflated* by ``1/F_i`` (York-Karplus' prescription, Lange & Herbert
    eq. 3.7), which only adds a non-negative diagonal and so cannot break
    that. Definiteness is therefore structural, not a property of the
    geometry, which is exactly what the point-charge kernel could not offer.
    """
    from scipy.special import erf as _erf

    if zeta is None:
        zeta = york_karplus_zeta(n_points_per_sphere)
    z = gaussian_exponents(w, switching, zeta=zeta)
    F = np.ones_like(w) if switching is None else np.asarray(
        switching, dtype=np.float64
    )

    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    # zeta_ij = zeta_i zeta_j / sqrt(zeta_i^2 + zeta_j^2)  (York-Karplus eq. 56)
    z_ij = np.outer(z, z) / np.sqrt(z[:, None] ** 2 + z[None, :] ** 2)
    x = z_ij * dist
    # Written as zeta_ij * (erf(x)/x) rather than erf(zeta_ij r)/r so the
    # r -> 0 limit is reachable in code and not only on paper. Being finite at
    # contact is the entire reason this kernel exists; computing it as
    # erf(0)/0 would hand back a NaN at the one geometry the fix is for. The
    # series form also avoids the cancellation in erf(x)/x for small x.
    small = x < 1e-8
    ratio = np.where(small, 2.0 / np.sqrt(np.pi), _erf(np.where(small, 1.0, x))
                     / np.where(small, 1.0, x))
    A = z_ij * ratio
    # Contact limit erf(x)/x -> 2/sqrt(pi) gives zeta_i sqrt(2/pi) at
    # zeta_ii = zeta_i/sqrt(2); the 1/F_i keeps the potential surface
    # continuous as a segment switches out.
    np.fill_diagonal(A, z * np.sqrt(2.0 / np.pi) / F)
    return A


def build_cavity_A_matrix(cavity, *, zeta: float | None = None) -> np.ndarray:
    """``A`` for a cavity, using the charge representation it declares.

    The one call site that should exist. A cavity knows whether its
    construction protects against close segment approach, so it -- not the
    caller -- decides which kernel is valid. Reading the choice off the cavity
    is what stops a third construction from silently inheriting a kernel whose
    precondition it does not meet, which is how #744 happened.
    """
    representation = getattr(cavity, "charge_representation", POINT_CHARGE)
    switching = getattr(cavity, "switching", None)
    return build_A_matrix(
        cavity.points,
        cavity.weights,
        representation=representation,
        switching=switching if representation == GAUSSIAN_CHARGE else None,
        zeta=zeta,
        n_points_per_sphere=getattr(cavity, "n_points_per_sphere", None),
    )


def solve_apparent_charges(
    A: np.ndarray,
    V_at_cavity: np.ndarray,
    *,
    epsilon: float,
    variant: str = "cpcm",
) -> CPCMResult:
    """Solve A q = -f(e) V for the apparent surface charges.

    Returns a :class:`CPCMResult` carrying ``q``, ``V``, the solvation
    energy ``1/2 q.V``, and the dielectric used. The matrix ``A`` is
    factorised by ``numpy.linalg.solve`` -- for repeated solves with
    the same cavity, prefer caching a Cholesky outside (the cavity is
    fixed for the duration of one SCF macro-iteration set).
    """
    V = np.asarray(V_at_cavity, dtype=np.float64).reshape(-1)
    if V.shape[0] != A.shape[0]:
        raise ValueError(
            f"solve_apparent_charges: V length {V.shape[0]} does "
            f"not match A shape {A.shape}"
        )
    native = _native_cpcm()
    if native is not None:
        try:
            res = native.cpcm_solve_apparent_charges(
                np.asarray(A, dtype=np.float64),
                V,
                float(epsilon),
                variant.lower(),
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        return CPCMResult(
            q=np.asarray(res.q, dtype=np.float64),
            V=np.asarray(res.V, dtype=np.float64),
            e_solv=float(res.e_solv),
            epsilon=float(res.epsilon),
        )

    f = dielectric_factor(epsilon, variant=variant)
    rhs = -f * V
    q = np.linalg.solve(A, rhs)
    e_solv = 0.5 * float(np.dot(q, V))
    return CPCMResult(q=q, V=V, e_solv=e_solv, epsilon=float(epsilon))
