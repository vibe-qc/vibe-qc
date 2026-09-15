"""Plane-wave / FFT grid infrastructure for the v0.10.x GAPW route.

> **Experimental.** This module is the M1 milestone of the gapw
> chat (docs/design_periodic_gapw.md). The :class:`PlaneWaveGrid`
> dataclass and the cutoff-driven sizing heuristics are stable; the
> downstream GPW driver (M2) and GAPW augmentation (M3) are not yet
> implemented. The :class:`PeriodicJKMethod.GPW` and ``GAPW`` enum
> values raise ``NotImplementedError`` until those land.

This module owns the **smooth real-space grid** that the GPW / GAPW
J build samples on. The actual FFT-Poisson solve runs in C++
(``cpp/src/fft_poisson.cpp``, already exposed as
:func:`vibeqc._vibeqc_core.solve_poisson_coulomb` /
:func:`solve_poisson_erf_screened`); this module is the Python
data model that owns the grid extents, the cutoff -> grid sizing
heuristic, the reciprocal-lattice metric, and the helpers for
mapping between fractional and Cartesian coordinates that the
collocation kernel will need.

Conventions
-----------

* **Fractional coordinates** ``s = A⁻¹ r`` are the grid's native
  coordinate system. Sample points sit at
  ``s = (i / nx, j / ny, k / nz)`` for ``i in [0, nx)`` etc. -- the
  same convention as :func:`solve_poisson_coulomb`.
* **Reciprocal-lattice columns** ``B = 2pi A⁻ᵀ``: ``b₁ = 2pi x (a₂ x a₃) / V``
  and cyclic. ``|G|^2 = (k_x b₁ + k_y b₂ + k_z b₃).(...)`` uses the
  full off-diagonal metric, so skew cells work without special
  cases.
* **FFT wrap-around** on each axis: ``k in {0, 1, ..., n/2 - 1,
  -n/2, ..., -1}``. Standard.
* **Atomic units**: lattice in bohr; cutoff in Hartree; grid
  spacing implicitly in bohr.

The grid auto-sizing follows the standard plane-wave-cutoff
heuristic: a kinetic-energy cutoff ``E_cut`` (Ha) corresponds to
``|G_max| = sqrt(2 E_cut)`` (atomic units; the kinetic energy of
a plane wave with wavevector G is ``1/2 |G|^2``). The number of grid
points along ``â_a`` is then
``n_a = ⌈|a_a| . |G_max| . k_safe / pi⌉`` (rounded up to an
FFT-friendly value), where the safety factor ``k_safe`` defaults
to 2 (the Nyquist criterion).

For an all-electron-aware default we expose
:func:`recommend_cutoff_from_basis` which probes a basis for the
tightest primitive exponent ``a_max`` and recommends a cutoff
``E_cut ~ K . a_max`` with a conservative ``K = 60`` (CP2K's
``REL_CUTOFF`` default). Users override via the ``cutoff_ha``
argument on :func:`make_grid`.

The FFT-friendly rounding uses the small set ``{2ᵃ . 3ᵇ . 5ᶜ}``;
FFTW3 (and other modern FFT libraries) hit their fastest
codepaths on these factor-only sizes.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "PlaneWaveGrid",
    "GAPWExperimentalWarning",
    "make_grid",
    "recommend_cutoff_from_basis",
    "nx_for_axis",
    "round_up_fft_friendly",
    "collocate_point_charges_on_grid",
    "point_charge_self_energy",
]


# Conservative default safety factor: 2 corresponds to the Nyquist
# criterion (two samples per shortest plane-wave period). Some codes
# go higher (3-4) for very high-derivative densities; CP2K's typical
# REL_CUTOFF=60 Ry against the tightest primitive is the calibration
# point -- see recommend_cutoff_from_basis for the basis-aware path.
_DEFAULT_SAFETY_FACTOR = 2.0

# K_max heuristic prefactor (CP2K's REL_CUTOFF). Cutoff ~ K * a_max
# where a_max is the tightest primitive exponent in the basis. The
# 60-Ha value gives ~µHa convergence on typical contracted Gaussian
# densities; tighter targets want K = 80 or 120. The unit here is Ha,
# not Ry -- the CP2K convention is Ry so doubled.
_DEFAULT_CUTOFF_K = 60.0


class GAPWExperimentalWarning(UserWarning):
    """Emitted when GAPW infrastructure is used.

    The GPW (pseudo-valence Gaussian-and-plane-waves) surface ships
    production-ready for RHF/RKS/UHF/UKS with pure DFT at Gamma and
    multi-k.  The all-electron GAPW augmentation, range-separated
    hybrids, and the OT solver are opt-in experimental until v0.23.0
    feature-completion parity.  Filter via
    ``warnings.filterwarnings("ignore",
    category=GAPWExperimentalWarning)`` if you need quiet output.
    """


def _warn_experimental(detail: str) -> None:
    warnings.warn(
        f"GAPW infrastructure: {detail}. "
        f"See docs/design_periodic_gapw.md and docs/user_guide/gapw.md.",
        category=GAPWExperimentalWarning,
        stacklevel=3,
    )


def round_up_fft_friendly(n: int, max_factor: int = 5) -> int:
    """Round ``n`` up to the next ``{2ᵃ . 3ᵇ . 5ᶜ}`` value.

    FFTW3 and other modern FFT libraries are fastest on sizes whose
    prime factorisation is contained in ``{2, 3, 5}`` (and sometimes
    ``7``). Default ``max_factor=5`` reproduces FFTW3's "good size"
    set; bump to ``7`` for slightly tighter coverage at the cost of
    higher RAM per grid call.

    Returns ``n`` unchanged when it's already a good size. Always
    returns an even number, since FFTW's real-to-complex transform
    is easier on even sizes (the Hermitian-conjugate symmetry maps
    cleanly onto ``n/2 + 1`` non-redundant slots).
    """
    if n < 2:
        return 2
    if n % 2 == 1:
        n += 1
    primes: tuple[int, ...]
    if max_factor >= 7:
        primes = (2, 3, 5, 7)
    elif max_factor >= 5:
        primes = (2, 3, 5)
    else:
        primes = (2, 3)

    def is_good(m: int) -> bool:
        for p in primes:
            while m % p == 0:
                m //= p
        return m == 1

    while not is_good(n):
        n += 2
    return n


def nx_for_axis(
    axis_length_bohr: float,
    cutoff_ha: float,
    safety_factor: float = _DEFAULT_SAFETY_FACTOR,
) -> int:
    """Return the number of grid points along a single lattice axis.

    Plane-wave cutoff convention: ``E_cut = 1/2 |G_max|^2`` so
    ``|G_max| = √(2 E_cut)`` in atomic units. The Nyquist criterion
    asks for >= 2 samples per wavelength of the highest plane wave,
    so ``n >= |a| . |G_max| / pi``. The safety factor scales that
    minimum.

    Returns an FFT-friendly (``{2ᵃ . 3ᵇ . 5ᶜ}``) value.
    """
    if cutoff_ha <= 0.0:
        raise ValueError(f"cutoff_ha must be positive, got {cutoff_ha!r}.")
    if axis_length_bohr <= 0.0:
        raise ValueError(
            f"axis_length_bohr must be positive, got {axis_length_bohr!r}."
        )
    G_max = math.sqrt(2.0 * cutoff_ha)
    n_min = axis_length_bohr * G_max * safety_factor / math.pi
    return round_up_fft_friendly(max(int(math.ceil(n_min)), 4))


@dataclass(frozen=True)
class PlaneWaveGrid:
    """Uniform 3D real-space grid spanning one unit cell, fractional-
    coordinate native.

    Attributes
    ----------
    lattice_bohr
        ``(3, 3)`` Cartesian lattice matrix whose **columns** are
        ``a₁, a₂, a₃`` in bohr. Same convention as the rest of the
        periodic stack (see :func:`solve_poisson_coulomb`).
    nx, ny, nz
        Grid extents along each lattice axis. Sample points sit at
        ``s = (i / nx, j / ny, k / nz)`` for the fractional
        coordinate ``s = A⁻¹ r``.
    cutoff_ha
        The kinetic-energy cutoff (Ha) from which the grid extents
        were derived, if any. ``None`` when the grid was constructed
        directly from ``(nx, ny, nz)`` without going through the
        cutoff heuristic.
    """

    lattice_bohr: np.ndarray
    nx: int
    ny: int
    nz: int
    cutoff_ha: float | None = None

    # ---------- construction helpers ------------------------------------

    def __post_init__(self) -> None:
        L = np.asarray(self.lattice_bohr, dtype=float)
        if L.shape != (3, 3):
            raise ValueError(f"lattice_bohr must be (3, 3); got shape {L.shape}.")
        if not np.isfinite(L).all():
            raise ValueError("lattice_bohr contains non-finite entries.")
        if abs(np.linalg.det(L)) < 1e-12:
            raise ValueError("lattice_bohr is singular (det ≈ 0).")
        for axis_name, n in (("nx", self.nx), ("ny", self.ny), ("nz", self.nz)):
            if not isinstance(n, int) or n < 2:
                raise ValueError(f"{axis_name} must be an integer >= 2; got {n!r}.")
        # Persist a numpy view of the lattice -- frozen=True forbids
        # ordinary assignment, so we go through object.__setattr__.
        object.__setattr__(self, "lattice_bohr", L)

    # ---------- derived properties --------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(nx, ny, nz)`` -- the numpy shape of a grid field."""
        return (self.nx, self.ny, self.nz)

    @property
    def n_points(self) -> int:
        """``nx * ny * nz`` -- total number of grid samples."""
        return self.nx * self.ny * self.nz

    @property
    def cell_volume_bohr3(self) -> float:
        """``|det(A)|`` -- unit-cell volume in bohr^3."""
        return float(abs(np.linalg.det(self.lattice_bohr)))

    @property
    def reciprocal_lattice_bohr_inv(self) -> np.ndarray:
        """``B = 2pi A⁻ᵀ`` -- reciprocal lattice (columns ``b₁, b₂, b₃``).
        ``b_i . a_j = 2pi d_ij``."""
        return 2.0 * math.pi * np.linalg.inv(self.lattice_bohr).T

    @property
    def axis_lengths_bohr(self) -> np.ndarray:
        """``(|a₁|, |a₂|, |a₃|)`` -- axis lengths in bohr."""
        return np.linalg.norm(self.lattice_bohr, axis=0)

    @property
    def voxel_volume_bohr3(self) -> float:
        """Volume per grid sample: ``V_cell / N_grid``."""
        return self.cell_volume_bohr3 / self.n_points

    # ---------- coordinate generation -----------------------------------

    def fractional_coords(self) -> np.ndarray:
        """Return ``(nx, ny, nz, 3)`` fractional coordinates ``s``.

        Generated lazily to avoid carrying a large array on the
        dataclass itself.
        """
        ix = np.arange(self.nx, dtype=float) / self.nx
        iy = np.arange(self.ny, dtype=float) / self.ny
        iz = np.arange(self.nz, dtype=float) / self.nz
        sx, sy, sz = np.meshgrid(ix, iy, iz, indexing="ij")
        return np.stack([sx, sy, sz], axis=-1)

    def cartesian_coords(self) -> np.ndarray:
        """Return ``(nx, ny, nz, 3)`` Cartesian coordinates ``r``.

        ``r = A . s`` per the column convention. The result lives in
        bohr.
        """
        s = self.fractional_coords()
        # Broadcasted matmul: r[..., :] = A @ s[..., :]
        return np.einsum("ij,xyzj->xyzi", self.lattice_bohr, s)

    def g_wrap(self, i: int, n: int) -> int:
        """FFT wrap-around: ``k = i if i < n/2 else i - n``."""
        return i if i < n // 2 else i - n

    def reciprocal_vectors(self) -> np.ndarray:
        """Return ``(nx, ny, nz, 3)`` reciprocal-space vectors ``G``
        on the FFT-wrapped grid. ``G[..., :]`` is in 1/bohr.

        Convention matches FFTW3's natural ordering, which the
        FFT-Poisson kernel inside :func:`solve_poisson_coulomb`
        consumes -- so this method is the right complement when a
        caller needs to manipulate r̃(G) or V(G) manually.
        """
        B = self.reciprocal_lattice_bohr_inv
        kx = np.array([self.g_wrap(i, self.nx) for i in range(self.nx)], dtype=float)
        ky = np.array([self.g_wrap(j, self.ny) for j in range(self.ny)], dtype=float)
        kz = np.array([self.g_wrap(k, self.nz) for k in range(self.nz)], dtype=float)
        kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing="ij")
        # G[..., :] = kx . b₁ + ky . b₂ + kz . b₃
        return (
            kxg[..., None] * B[:, 0]
            + kyg[..., None] * B[:, 1]
            + kzg[..., None] * B[:, 2]
        )

    # ---------- convenience -------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover -- formatting only
        ax, ay, az = self.axis_lengths_bohr
        cut = f"{self.cutoff_ha:.3f} Ha" if self.cutoff_ha else "n/a"
        return (
            f"PlaneWaveGrid({self.nx}x{self.ny}x{self.nz}, "
            f"|a|=({ax:.2f}, {ay:.2f}, {az:.2f}) bohr, cutoff={cut})"
        )


def make_grid(
    lattice_bohr: np.ndarray,
    *,
    cutoff_ha: float,
    safety_factor: float = _DEFAULT_SAFETY_FACTOR,
) -> PlaneWaveGrid:
    """Build a :class:`PlaneWaveGrid` sized to a given cutoff.

    Each axis's grid extent is set independently via
    :func:`nx_for_axis`; the result is FFT-friendly per axis.

    Emits :class:`GAPWExperimentalWarning` to remind the caller that
    the downstream GAPW driver is not yet wired up (the grid object
    itself is fine to use).
    """
    L = np.asarray(lattice_bohr, dtype=float)
    if L.shape != (3, 3):
        raise ValueError(f"lattice_bohr must be (3, 3); got shape {L.shape}.")
    axis_lengths = np.linalg.norm(L, axis=0)
    nx, ny, nz = (
        nx_for_axis(float(axis_lengths[i]), cutoff_ha, safety_factor) for i in range(3)
    )
    _warn_experimental(
        f"PlaneWaveGrid built at cutoff={cutoff_ha:.2f} Ha -> {nx}x{ny}x{nz}"
    )
    return PlaneWaveGrid(L, nx, ny, nz, cutoff_ha=cutoff_ha)


def recommend_cutoff_from_basis(
    exponents: Iterable[float],
    *,
    k_safe: float = _DEFAULT_CUTOFF_K,
    min_cutoff_ha: float = 1.0,
) -> float:
    """Recommend a plane-wave cutoff (Ha) from the tightest Gaussian
    primitive in a basis.

    The plane-wave grid must resolve the highest-frequency feature
    of the density ``r = S_muν D_muν chi_mu chi_ν``. The product of two
    Gaussian primitives of exponents ``a_a, a_b`` is a single
    Gaussian of exponent ``a_a + a_b`` (Gaussian-product theorem);
    the tightest density-Gaussian thus has exponent ``2 . a_max``
    where ``a_max`` is the tightest orbital primitive. We compute
    the cutoff as ``k_safe . a_max`` to stay calibration-consistent
    with CP2K's `REL_CUTOFF` knob (which is applied to the fused
    `ζ_p = a_a + a_b` after Gaussian-product fusion in CP2K's
    multigrid hot path). At `k_safe = 60` Ha, the resulting cutoff
    is roughly equivalent to CP2K's recommended `REL_CUTOFF =
    60 Ry ≈ 30 Ha` applied to `ζ_p ≈ 2 . a_max` -- so the default
    here is conservative by a factor of ~2 relative to CP2K's
    multigrid setting.

    The ``min_cutoff_ha`` floor avoids absurdly small grids on
    pathologically diffuse basis sets -- a cutoff of < 1 Ha would
    leave the grid unable to resolve the cell metric itself.

    Audit note (B2): vibe-qc has no multigrid + no Gaussian-product
    fusion yet, so this single number sets the *only* grid; CP2K's
    multigrid cascade routes each fused primitive to its
    appropriate sub-grid level via `gaussian_gridlevel`. M3b
    perf-track will likely add either fusion or multigrid.
    """
    exps_list = [float(e) for e in exponents if e > 0]
    if not exps_list:
        raise ValueError(
            "recommend_cutoff_from_basis: no positive exponents in the "
            "input. Pass the basis's primitive Gaussian exponents in "
            "bohr⁻^2."
        )
    alpha_max = max(exps_list)
    cutoff = k_safe * alpha_max
    return max(cutoff, min_cutoff_ha)


def collect_basis_exponents(basis) -> list[float]:
    """Walk a ``vibeqc.BasisSet`` and return every primitive exponent.

    Tolerant of slight API drift: tries the modern
    ``basis.shells / shell.alphas`` path first, then falls back to a
    duck-typed scan of any ``.alphas`` / ``.exponents`` attribute
    found on the basis tree. Returns an empty list rather than
    raising on an unrecognised shape -- the caller can decide whether
    that's fatal.
    """
    exps: list[float] = []
    for attr in ("shells", "primitives", "contractions"):
        shells = getattr(basis, attr, None)
        if shells is None:
            continue
        for shell in shells:
            for ea in ("alphas", "exponents", "alpha"):
                vals = getattr(shell, ea, None)
                if vals is None:
                    continue
                try:
                    exps.extend(float(v) for v in vals)
                    break
                except TypeError:
                    pass
            break
    return exps


def _diagonal_lattice(
    a: float, b: float | None = None, c: float | None = None
) -> np.ndarray:
    """Compose a diagonal lattice matrix from axis lengths (bohr)."""
    b = a if b is None else b
    c = a if c is None else c
    return np.diag([a, b, c]).astype(float)


# ============================================================
# Collocation -- point charges as Gaussian smearings
# ============================================================


def collocate_point_charges_on_grid(
    charges: Sequence[float],
    positions_bohr: np.ndarray,
    grid: PlaneWaveGrid,
    sigma_bohr: float,
    *,
    image_shells: int = 1,
) -> np.ndarray:
    """Lay each point charge ``q_i`` down on the grid as a normalised
    Gaussian of width ``sigma_bohr``, summing all contributions:

        ``r(r) = S_i q_i . (2pi s^2)^(-3/2) . exp(-|r - r_i|^2 / (2 s^2))``

    Each Gaussian integrates to ``q_i`` over all space, so the total
    integrated charge on the grid equals ``S_i q_i`` to within the
    finite-cell + finite-grid quadrature error. For positions inside
    the cell with ``sigma`` much smaller than the cell, that error
    is bounded by the Gaussian tails crossing the cell boundary --
    by default this function sums over the nearest ``image_shells``
    layer of periodic images so the standard ``[-L, L]`` box gets
    every fraction of the Gaussian back.

    Parameters
    ----------
    charges
        Per-site charge values (e). For nuclear repulsion these are
        the atomic numbers ``Z``; for Mulliken-style smearing of a
        density they could be Mulliken populations. Sign convention:
        positive ``q`` = positive charge (electrons would be ``-1``).
    positions_bohr
        ``(n_sites, 3)`` Cartesian positions in bohr. They need not
        lie inside the cell; the function wraps to the cell via the
        ``image_shells`` summation.
    grid
        :class:`PlaneWaveGrid` defining the cell + grid.
    sigma_bohr
        Gaussian width. Should satisfy ``sigma << min(axis_lengths)``;
        a typical Ewald-style choice is ``sigma ~ 0.5 / a_Ewald``.
        Must be positive.
    image_shells
        Number of neighbouring images along each axis to sum. The
        default ``1`` includes ``3^3 = 27`` cells centred on the home
        cell, which is overkill for ``sigma`` ~ 0.5 bohr in a
        10-bohr cell (Gaussian tail at the second shell is ~e⁻¹⁰⁰).
        Set to ``0`` for the strictly home-cell density (the
        Gaussian's tail outside the cell is then lost, ``∫r < S q``
        unless every charge is well inside the cell). Set to ``2``+
        for very small cells with large ``sigma``.

    Returns
    -------
    rho
        ``(nx, ny, nz)`` real-space density on the grid, in units
        of ``e / bohr^3``.

    Notes
    -----
    The collocation cost is ``O(n_sites . (2.image_shells+1)^3 . N_grid)``
    -- fine for an initial reference implementation; M2 will need a
    "compact-support" optimisation (limit each Gaussian to ~4s around
    its centre) to scale to large systems.
    """
    if sigma_bohr <= 0.0:
        raise ValueError(f"sigma_bohr must be positive, got {sigma_bohr!r}")
    if image_shells < 0:
        raise ValueError(f"image_shells must be >= 0, got {image_shells!r}")

    charges_arr = np.asarray(charges, dtype=float)
    positions = np.asarray(positions_bohr, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions_bohr must be (n_sites, 3); got shape {positions.shape}"
        )
    if charges_arr.shape != (positions.shape[0],):
        raise ValueError(
            f"charges shape {charges_arr.shape} doesn't match positions "
            f"({positions.shape[0]} sites)"
        )

    r = grid.cartesian_coords()  # (nx, ny, nz, 3)
    L = grid.lattice_bohr  # columns a1, a2, a3
    norm = 1.0 / (2.0 * math.pi * sigma_bohr**2) ** 1.5
    inv_two_sigma2 = 1.0 / (2.0 * sigma_bohr**2)

    rho = np.zeros(grid.shape, dtype=float)
    for q, r0 in zip(charges_arr, positions):
        for ix in range(-image_shells, image_shells + 1):
            for iy in range(-image_shells, image_shells + 1):
                for iz in range(-image_shells, image_shells + 1):
                    shift = ix * L[:, 0] + iy * L[:, 1] + iz * L[:, 2]
                    centre = r0 + shift
                    delta = r - centre
                    r2 = (delta * delta).sum(axis=-1)
                    rho += q * norm * np.exp(-r2 * inv_two_sigma2)
    return rho


def point_charge_self_energy(
    charges: Sequence[float],
    sigma_bohr: float,
) -> float:
    """Return the **per-cell** Gaussian self-interaction energy that
    must be subtracted from a periodic FFT-Poisson nuclear-repulsion
    result.

    Each Gaussian-smeared point charge ``q_i`` carries the spurious
    self-Hartree energy

        ``E_self_i = q_i^2 / (2 s √pi)``

    which is the all-space Hartree integral of one isolated unit-norm
    Gaussian times ``q^2``. The Ewald-style nuclear-repulsion observable
    we ultimately want excludes this self-term; the FFT-Poisson Hartree
    integral *includes* it, so the standard correction is

        ``E_nuc_corrected = E_H_grid - S_i E_self_i``

    (modulo the Madelung shift on charged cells; for nuclear repulsion
    in a charge-neutral neutralizing-background convention the
    Madelung term is the third Ewald piece and is constant per system,
    not per s.)
    """
    if sigma_bohr <= 0.0:
        raise ValueError(f"sigma_bohr must be positive, got {sigma_bohr!r}")
    q = np.asarray(charges, dtype=float)
    return float((q * q).sum() / (2.0 * sigma_bohr * math.sqrt(math.pi)))
