"""Per-atom radial x angular quadrature grids for GAPW augmentation.

> **Experimental, infrastructure-only.** This is the M3b GAPW
> augmentation prep: per-atom Lebedev x radial grids that the
> hard / soft Hartree decomposition runs on (Lippert-Hutter 1999,
> CP2K's ``grid_atom`` family). Not yet wired into the SCF -- the
> M3b SCF (``periodic_gapw_j.run_periodic_rhf_gpw`` with
> ``v_ne_convention="smeared_erfc"``) builds V_ne via the smeared
> erfc/erf split on the FFT grid alone; the augmentation
> correction on top of that is the next chunk of work.

The atomic-grid object owns:

* a 1D **radial mesh** ``r_k`` with weights ``w_k`` such that
  ``∫_0^inf f(r) . r^2 dr  ≈  S_k f(r_k) . w_k``;
* a 2D **angular mesh** on the unit sphere ``Ω̂_l`` with weights
  ``w_l`` such that ``∫_S^2 g(Ω̂) dΩ̂  ≈  S_l g(Ω̂_l) . w_l``;
* a derived 3D **product grid** ``r_k . Ω̂_l`` with combined
  weights ``w_k . w_l`` for ``∫ f(r) . r^2 . g(Ω̂) dr dΩ̂``.

**Radial rule (Mura-Knowles 1996).** Maps the unit interval
``x in (0, 1)`` to ``r in (0, inf)`` via

    ``r(x)  =  -a . ln(1 - x^3)``,
    ``dr/dx = 3 . a . x^2 / (1 - x^3)``.

Using uniform sampling in ``x`` plus the volume element ``r^2``,
this gives a smooth radial mesh that hugs the nucleus (where x -> 0)
and the diffuse tail (where x -> 1) about equally. The scale
``a`` is chosen per-element. (Mura M. E. & Knowles P. J.,
*J. Chem. Phys.* 104, 9848 (1996); the CP2K developers credit
Lippert et al. 1999 for the equivalent log-transformed
Gauss-Chebyshev form.)

**Angular rule (Lebedev-Laikov).** Delegated to
``scipy.integrate.lebedev_rule``; weights normalised to ``S_l w_l
= 4pi`` matching the surface area of the unit sphere.

For now the grids are **unpartitioned** (per-atom, no Becke
weighting). GAPW only needs them inside an augmentation sphere
around each nucleus, so the partitioning that the molecular
Becke grid (``vibeqc._vibeqc_core.build_grid``) applies is the
wrong thing here -- we want raw atomic shells, soft-cut at the
augmentation radius if needed.

Audit reference: ``docs/design_periodic_gapw.md`` Sec. Q2 / D1 / D2.
The CP2K reference uses log-transformed Gauss-Chebyshev radial x
Lebedev angular (``grid_atom_type`` in ``grid_atom.F``). The
Mura-Knowles mapping we use here is numerically equivalent for
the integrands GAPW sees (smooth Gaussians); the
implementation is paraphrased from the *paper* -- no GPL source
reproduced.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.integrate import lebedev_rule as _scipy_lebedev_rule
    _HAS_LEBEDEV = True
except ImportError:  # pragma: no cover -- scipy is a hard runtime dep
    _HAS_LEBEDEV = False

from .periodic_gapw_grid import GAPWExperimentalWarning

__all__ = [
    "AtomicRadialGrid",
    "default_alpha_for_element",
    "lebedev_supported_orders",
    "real_spherical_harmonics",
    "compute_multipole_moments",
    "multipole_index",
    "multipole_index_pairs",
    "n_multipole_components",
]


# Mura-Knowles per-element radial scale a (bohr). Chosen so that the
# bulk of the radial weight sits in the chemically relevant range
# 0.1-4 bohr from the nucleus. Values for H-Ne calibrated against
# the Treutler-Ahlrichs ξ scales (treutler_xi in
# ``cpp/src/grid.cpp``); chemically equivalent to ~0.5 x ξ for the
# Mura-Knowles transform's slightly different mapping.
# Beyond Z = 10 we fall back to a Z-dependent heuristic that
# matches the empirical valence-shell scale.
_MURA_KNOWLES_ALPHAS_HE_NE = {
    1: 5.0,   # H
    2: 4.0,   # He
    3: 7.0,   # Li
    4: 5.0,   # Be
    5: 5.0,   # B
    6: 5.0,   # C
    7: 5.0,   # N
    8: 5.0,   # O
    9: 5.0,   # F
    10: 5.0,  # Ne
}


def default_alpha_for_element(Z: int) -> float:
    """Mura-Knowles radial scale ``a`` (bohr) for element ``Z``.

    Returns a tabulated value for H-Ne; for heavier elements
    interpolates linearly in ``Z`` between the Ne value and a
    capped value at Z = 30 (≈ 6 bohr) to keep the radial mesh
    well-resolved without spending nodes far outside the
    chemically relevant range.

    For production-quality work users should override per-element
    via the ``alpha`` parameter on
    :meth:`AtomicRadialGrid.from_element`.
    """
    if Z <= 0:
        raise ValueError(f"Z must be a positive integer, got {Z!r}")
    if Z in _MURA_KNOWLES_ALPHAS_HE_NE:
        return _MURA_KNOWLES_ALPHAS_HE_NE[Z]
    if Z <= 30:
        # Linear interpolate Ne (5.0) -> Zn-area (6.0).
        return 5.0 + (Z - 10) * (6.0 - 5.0) / 20.0
    return 6.0


def lebedev_supported_orders() -> tuple[int, ...]:
    """Return the Lebedev orders scipy.integrate.lebedev_rule supports.

    The Lebedev-Laikov family doesn't have a value at *every*
    odd order; the supported set is what
    :func:`scipy.integrate.lebedev_rule` accepts. Common production
    orders for GAPW (CP2K's defaults): 17, 23, 29, 35, 41.
    """
    # Empirically the scipy set is the standard Lebedev-Laikov tiers.
    return (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)


@dataclass(frozen=True)
class AtomicRadialGrid:
    """Per-atom radial x Lebedev quadrature grid.

    Attributes
    ----------
    centre_bohr
        ``(3,)`` Cartesian position of the atom this grid is
        centred on (bohr).
    r
        ``(n_radial,)`` radial mesh in bohr (strictly positive).
    w_r
        ``(n_radial,)`` radial weights such that ``∫_0^inf f(r) r^2
        dr ≈ S_k f(r_k) . w_k``. Note: the ``r^2`` Jacobian is
        already folded in; ``w_k`` already carries it.
    angular_xyz
        ``(n_angular, 3)`` Lebedev points on the unit sphere
        (unit-norm rows).
    w_a
        ``(n_angular,)`` angular weights such that ``∫_S^2 g(Ω̂) dΩ̂
        ≈ S_l g(Ω̂_l) . w_l``, ``S_l w_l = 4pi``.
    radial_alpha
        The Mura-Knowles ``a`` (bohr) used to build the radial
        mesh. Carried for audit / reproduction.
    lebedev_order
        The Lebedev algebraic order requested.
    """

    centre_bohr: np.ndarray
    r: np.ndarray
    w_r: np.ndarray
    angular_xyz: np.ndarray
    w_a: np.ndarray
    radial_alpha: float
    lebedev_order: int

    def __post_init__(self) -> None:
        centre = np.asarray(self.centre_bohr, dtype=float)
        if centre.shape != (3,):
            raise ValueError(
                f"centre_bohr must be shape (3,); got {centre.shape}"
            )
        r = np.asarray(self.r, dtype=float)
        w_r = np.asarray(self.w_r, dtype=float)
        if r.ndim != 1 or w_r.shape != r.shape:
            raise ValueError(
                f"r / w_r must be 1-D with equal length; got "
                f"r.shape={r.shape}, w_r.shape={w_r.shape}"
            )
        if not np.all(r > 0.0):
            raise ValueError(
                "r must be strictly positive (avoid r = 0 singular)"
            )
        ang = np.asarray(self.angular_xyz, dtype=float)
        w_a = np.asarray(self.w_a, dtype=float)
        if ang.ndim != 2 or ang.shape[1] != 3 or w_a.shape != (ang.shape[0],):
            raise ValueError(
                f"angular_xyz must be (n_angular, 3) and w_a "
                f"(n_angular,); got {ang.shape}, {w_a.shape}"
            )
        # Unit-vector pin (sanity).
        norms = np.linalg.norm(ang, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-10):
            raise ValueError(
                "angular_xyz rows must be unit vectors; got norms "
                f"in [{norms.min():.6f}, {norms.max():.6f}]"
            )
        if self.radial_alpha <= 0.0:
            raise ValueError(
                f"radial_alpha must be positive, got {self.radial_alpha!r}"
            )
        object.__setattr__(self, "centre_bohr", centre)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "w_r", w_r)
        object.__setattr__(self, "angular_xyz", ang)
        object.__setattr__(self, "w_a", w_a)

    @property
    def n_radial(self) -> int:
        return self.r.shape[0]

    @property
    def n_angular(self) -> int:
        return self.angular_xyz.shape[0]

    @property
    def n_points(self) -> int:
        return self.n_radial * self.n_angular

    def cartesian_points(self) -> np.ndarray:
        """Return ``(n_radial, n_angular, 3)`` Cartesian points
        ``r_k . Ω̂_l + R_atom``."""
        # outer product of r and Ω̂: shape (n_radial, n_angular, 3)
        return (self.r[:, None, None] * self.angular_xyz[None, :, :]
                + self.centre_bohr)

    def combined_weights(self) -> np.ndarray:
        """Return ``(n_radial, n_angular)`` product weights
        ``w_k . w_l``. The integral of a function ``f(r, Ω̂)`` is
        ``S_kl f(r_k, Ω̂_l) . w_k . w_l``."""
        return self.w_r[:, None] * self.w_a[None, :]

    def integrate(self, values: np.ndarray) -> float:
        """Quadrature: ``S_kl values[k, l] . w_k . w_l``.

        ``values`` is ``(n_radial, n_angular)`` evaluated at
        :func:`cartesian_points`.
        """
        if values.shape != (self.n_radial, self.n_angular):
            raise ValueError(
                f"values must have shape ({self.n_radial}, "
                f"{self.n_angular}); got {values.shape}"
            )
        return float(np.einsum("kl,k,l->", values, self.w_r, self.w_a))

    @classmethod
    def build(
        cls,
        centre_bohr: np.ndarray,
        *,
        n_radial: int = 50,
        alpha: float = 5.0,
        lebedev_order: int = 17,
    ) -> "AtomicRadialGrid":
        """Build a Mura-Knowles x Lebedev grid centred on
        ``centre_bohr``.

        Parameters
        ----------
        centre_bohr
            ``(3,)`` Cartesian atom position.
        n_radial
            Number of radial points (CP2K default 50, production 100+).
        alpha
            Mura-Knowles radial scale (bohr). Use
            :func:`default_alpha_for_element` for per-Z defaults.
        lebedev_order
            Lebedev algebraic order. 17 is sufficient for ℓ_max =
            8 spherical harmonics -- comfortably above the d / f
            level needed for typical GAPW projector multipoles.
        """
        if not _HAS_LEBEDEV:
            raise RuntimeError(
                "AtomicRadialGrid.build needs scipy.integrate.lebedev_rule "
                "(scipy >= 1.17). Install scipy or upgrade."
            )
        if n_radial < 4:
            raise ValueError(
                f"n_radial must be >= 4 to resolve a Gaussian, "
                f"got {n_radial!r}"
            )

        # Mura-Knowles radial transform.
        # x_i = i / (n_radial + 1) for i = 1, ..., n_radial -- interior
        # points avoiding both endpoints. dr/dx = 3.a.x^2 / (1 - x^3).
        # ∫_0^inf f(r) r^2 dr = ∫_0^1 f(r(x)) . r(x)^2 . (dr/dx) dx
        # Uniform sampling on x with weight 1/(n_radial + 1) gives the
        # composite-trapezoid rule; sufficient for smooth integrands.
        x = (np.arange(1, n_radial + 1, dtype=float)
             / (n_radial + 1))
        r = -alpha * np.log1p(-x ** 3)  # = -a . ln(1 - x^3)
        dr_dx = 3.0 * alpha * x ** 2 / (1.0 - x ** 3)
        # Weight for ∫_0^inf f(r) . r^2 dr ≈ S f(r_k) . w_k.
        w_r = (1.0 / (n_radial + 1.0)) * (r ** 2) * dr_dx

        # Lebedev angular grid. scipy.integrate.lebedev_rule returns
        # points as (3, n_points) and weights already normalised so
        # S_l w_l = 4pi (the unit-sphere surface area). Use as-is.
        points, weights = _scipy_lebedev_rule(lebedev_order)
        angular_xyz = points.T.copy()  # (n_points, 3)
        w_a = weights.copy()

        return cls(
            centre_bohr=np.asarray(centre_bohr, dtype=float),
            r=r,
            w_r=w_r,
            angular_xyz=angular_xyz,
            w_a=w_a,
            radial_alpha=alpha,
            lebedev_order=lebedev_order,
        )

    def evaluate_real_spherical_harmonics(self, lmax: int) -> np.ndarray:
        """Return ``(n_components, n_angular)`` real spherical harmonics
        ``Y_lm`` evaluated at this grid's Lebedev points.

        Components are ordered by ``multipole_index(l, m)`` --
        l = 0: (l=0, m=0); l = 1: (1, -1), (1, 0), (1, 1); ...
        ``n_components = (lmax + 1) ** 2``.
        """
        if lmax < 0:
            raise ValueError(f"lmax must be >= 0; got {lmax!r}")
        return real_spherical_harmonics(self.angular_xyz, lmax)

    @classmethod
    def from_element(
        cls,
        centre_bohr: np.ndarray,
        Z: int,
        *,
        n_radial: int = 50,
        lebedev_order: int = 17,
        quiet: bool = False,
    ) -> "AtomicRadialGrid":
        """Build with a per-element default ``a`` from
        :func:`default_alpha_for_element`.

        Emits :class:`GAPWExperimentalWarning` to remind callers
        the M3b augmentation infrastructure is experimental.
        """
        if not quiet:
            warnings.warn(
                f"AtomicRadialGrid.from_element(Z={Z}) -- M3b GAPW "
                f"augmentation grid infrastructure, experimental.",
                category=GAPWExperimentalWarning,
                stacklevel=2,
            )
        alpha = default_alpha_for_element(Z)
        return cls.build(
            centre_bohr,
            n_radial=n_radial,
            alpha=alpha,
            lebedev_order=lebedev_order,
        )


# ============================================================
# Real spherical harmonics & multipole moments
# ============================================================
#
# Convention: real spherical harmonics S_lm with normalisation
# ``∫_S^2 S_lm(Ω̂) . S_l'm'(Ω̂) dΩ̂ = d_ll' . d_mm'``. Compatible with
# the Stone convention used in CP2K's qs_rho0_methods for GAPW
# multipole compensators. The expression for each (l, m) up to
# l = 4 is hard-coded here -- sufficient for the multipole orders
# the typical GAPW augmentation needs (lmax = 4 covers s, p, d, f,
# g projector products).
#
# Higher-l would dispatch through scipy.special.sph_harm_y (which
# is complex-valued) plus a real-combination step; we ship the
# hard-coded path because it is bit-portable across scipy versions.


def multipole_index(l: int, m: int) -> int:
    """Linear index for the (l, m) component in a flat multipole array.

    Layout: l = 0 occupies index 0; l = 1 occupies indices 1, 2, 3
    for m = -1, 0, +1; l = 2 occupies indices 4 through 8 for
    m = -2, -1, 0, +1, +2; etc. In general ``idx = l^2 + l + m``.
    """
    if l < 0:
        raise ValueError(f"l must be >= 0; got {l!r}")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l; got l={l}, m={m}")
    return l * l + l + m


def multipole_index_pairs(lmax: int) -> list[tuple[int, int]]:
    """Return the ``(l, m)`` pairs in linear-index order up to
    ``lmax``."""
    return [
        (l, m)
        for l in range(lmax + 1)
        for m in range(-l, l + 1)
    ]


def n_multipole_components(lmax: int) -> int:
    """Number of (l, m) components up to ``lmax``: ``(lmax + 1)^2``."""
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0; got {lmax!r}")
    return (lmax + 1) ** 2


def real_spherical_harmonics(xyz: np.ndarray, lmax: int) -> np.ndarray:
    """Evaluate the real spherical harmonics ``S_lm`` on a batch of
    unit vectors, ordered by :func:`multipole_index`.

    Parameters
    ----------
    xyz
        ``(n_points, 3)`` unit-norm vectors (rows). The function
        does **not** normalise the input -- callers must pass
        unit vectors (Lebedev grids satisfy this by construction).
    lmax
        Maximum angular momentum, ``lmax >= 0``.

    Returns
    -------
    S
        ``((lmax + 1)^2, n_points)`` array of real spherical
        harmonic values.

    Convention
    ----------
    Stone-convention real ``S_lm`` with
    ``∫_S^2 S_lm . S_l'm' dΩ̂ = d_ll' . d_mm'``. The first few:

    * ``S_00 = 1/√(4pi)``
    * ``S_1,-1 = √(3/(4pi)) . y``,
      ``S_1, 0 = √(3/(4pi)) . z``,
      ``S_1, 1 = √(3/(4pi)) . x``
    * ``S_2,-2 = √(15/(4pi)) . xy``,
      ``S_2,-1 = √(15/(4pi)) . yz``,
      ``S_2, 0 = √(5/(16pi)) . (3 z^2 - 1)``,
      ``S_2, 1 = √(15/(4pi)) . zx``,
      ``S_2, 2 = √(15/(16pi)) . (x^2 - y^2)``
    * up through l = 4 hard-coded.

    Implementation references the standard Stone-convention table
    (Stone, *The Theory of Intermolecular Forces*, Appendix C).
    """
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0; got {lmax!r}")
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must be (n_points, 3); got {xyz.shape}"
        )
    n = xyz.shape[0]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    n_comp = n_multipole_components(lmax)
    S = np.empty((n_comp, n), dtype=float)

    pi = math.pi
    # For l <= 3 we use the hard-coded Cartesian closed forms
    # (bit-portable across scipy versions); for l >= 4 we route
    # through scipy.special.sph_harm_y and build the real
    # combinations.
    if lmax >= 4:
        from scipy.special import sph_harm_y
        # Convert unit Cartesian vectors to (th, phi). th in [0, pi],
        # phi in [-pi, pi].
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)
        for l in range(4, lmax + 1):
            for m in range(-l, l + 1):
                if m == 0:
                    # Y_l,0 is already real.
                    Y = sph_harm_y(l, 0, theta, phi)
                    S[multipole_index(l, 0)] = np.real(Y)
                elif m > 0:
                    # S_lm = √2 . (-1)^m . Re(Y_l,m).
                    Y = sph_harm_y(l, m, theta, phi)
                    S[multipole_index(l, m)] = (
                        math.sqrt(2.0) * ((-1) ** m) * np.real(Y)
                    )
                else:
                    # S_lm = √2 . (-1)^m . Im(Y_l,|m|).
                    Y = sph_harm_y(l, -m, theta, phi)
                    S[multipole_index(l, m)] = (
                        math.sqrt(2.0) * ((-1) ** m) * np.imag(Y)
                    )

    # l = 0
    S[multipole_index(0, 0)] = 1.0 / math.sqrt(4.0 * pi) * np.ones(n)

    if lmax >= 1:
        c1 = math.sqrt(3.0 / (4.0 * pi))
        S[multipole_index(1, -1)] = c1 * y
        S[multipole_index(1, 0)] = c1 * z
        S[multipole_index(1, 1)] = c1 * x

    if lmax >= 2:
        c2_off = math.sqrt(15.0 / (4.0 * pi))
        c2_z2 = math.sqrt(5.0 / (16.0 * pi))
        c2_xy = math.sqrt(15.0 / (16.0 * pi))
        S[multipole_index(2, -2)] = c2_off * x * y
        S[multipole_index(2, -1)] = c2_off * y * z
        S[multipole_index(2, 0)] = c2_z2 * (3.0 * z ** 2 - 1.0)
        S[multipole_index(2, 1)] = c2_off * z * x
        S[multipole_index(2, 2)] = c2_xy * (x ** 2 - y ** 2)

    if lmax >= 3:
        c3a = math.sqrt(35.0 / (32.0 * pi))
        c3b = math.sqrt(105.0 / (4.0 * pi))
        c3c = math.sqrt(21.0 / (32.0 * pi))
        c3d = math.sqrt(7.0 / (16.0 * pi))
        S[multipole_index(3, -3)] = c3a * y * (3.0 * x ** 2 - y ** 2)
        S[multipole_index(3, -2)] = c3b * x * y * z
        S[multipole_index(3, -1)] = c3c * y * (5.0 * z ** 2 - 1.0)
        S[multipole_index(3, 0)] = c3d * z * (5.0 * z ** 2 - 3.0)
        S[multipole_index(3, 1)] = c3c * x * (5.0 * z ** 2 - 1.0)
        S[multipole_index(3, 2)] = (c3b / 2.0) * z * (x ** 2 - y ** 2)
        S[multipole_index(3, 3)] = c3a * x * (x ** 2 - 3.0 * y ** 2)

    return S


def compute_multipole_moments(
    density_on_grid: np.ndarray,
    grid: AtomicRadialGrid,
    lmax: int,
) -> np.ndarray:
    """Compute the multipole moments ``Q_lm`` of a density on a
    per-atom grid.

    Definition (Stone convention; matches CP2K's
    ``calculate_rho_atom_coeff`` family):

        ``Q_lm = ∫ r(r) . r^l . S_lm(Ω̂_r) dr``

    where ``r`` is the radial distance from the grid centre and
    ``Ω̂_r`` is the unit vector. With normalised real spherical
    harmonics ``S_lm``, the integral has units of charge x length^l.

    Examples (sanity, Stone normalisation):

    * Normalised Gaussian ``(a/√pi)^3 exp(-a^2 r^2)`` centred at the
      grid centre: ``Q_00 = <r . S_00> = (1/√(4pi)) . ∫r
      = 1/√(4pi) ≈ 0.28209``. Higher l moments are zero by
      spherical symmetry.
    * A net charge ``q`` centred at the grid centre gives
      ``Q_00 = q / √(4pi)``.

    Parameters
    ----------
    density_on_grid
        ``(n_radial, n_angular)`` real-valued density evaluated
        at :meth:`AtomicRadialGrid.cartesian_points`.
    grid
        :class:`AtomicRadialGrid`.
    lmax
        Maximum angular momentum.

    Returns
    -------
    Q
        ``((lmax + 1)^2,)`` flat array of multipole moments in
        :func:`multipole_index` order.
    """
    if density_on_grid.shape != (grid.n_radial, grid.n_angular):
        raise ValueError(
            f"density_on_grid must be ({grid.n_radial}, "
            f"{grid.n_angular}); got {density_on_grid.shape}"
        )
    # Evaluate S_lm(Ω̂_l): (n_comp, n_angular).
    S = grid.evaluate_real_spherical_harmonics(lmax)
    # Build r^l factor per radial point: r_factor[l, k] = r_k^l.
    n_comp = n_multipole_components(lmax)
    r_factors = np.empty((n_comp, grid.n_radial), dtype=float)
    for l in range(lmax + 1):
        r_l = grid.r ** l
        for m in range(-l, l + 1):
            r_factors[multipole_index(l, m)] = r_l
    # Q_lm = S_kl r(r_k, Ω̂_l) . r_k^l . S_lm(Ω̂_l) . w_k . w_l
    # = einsum('cr,ca,c,r,a,ra->c', r_factors, S,
    #         density_on_grid)  -- but the simpler decomposition is
    # to first build the angular projection per radial shell.
    # Angular projection: A[lm, r] = S_l_ang S[lm, l_ang] . density[r, l_ang] . w_a[l_ang]
    A = np.einsum("ca,ra,a->cr", S, density_on_grid, grid.w_a)
    # Radial integral: Q[lm] = S_r A[lm, r] . r_r^l . w_r[r]
    Q = np.einsum("cr,cr,r->c", A, r_factors, grid.w_r)
    return Q
