"""Sigma averaging and sigma profiles -- Klamt 1998 eq. 11, 14, 16.

COSMO gives a screening charge density per *segment*. COSMO-RS needs it per
*contact area*: the theory's whole premise is that molecules in a fluid pair
their surfaces in patches of a characteristic size, so what matters is the
charge density a neighbouring molecule actually sees, not the value on one
tessellation element. Klamt 1998 eq. 11 is the averaging that performs that
coarse-graining.

Units. The papers work in angstrom, e/angstrom^2 and kcal/mol, and the
published constants are only recognisable in those units, so the conversion
from vibe-qc's atomic units happens here, once, at the boundary -- not
scattered through the equations downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..cavity import ANG_TO_BOHR
from ..surface import MIN_SEGMENT_AREA_BOHR2, ConductorSurface

BOHR_TO_ANG = 1.0 / ANG_TO_BOHR

# Klamt 1998 eq. 14: the orthogonalisation coefficient relating the
# wider-radius average to the primary one. Determined by regression over the
# 1998 data set, quoted in the paper as 0.816.
SIGMA_PERP_ORTHOGONALISATION = 0.816


@dataclass(frozen=True)
class SegmentDescriptors:
    """Per-segment COSMO-RS descriptors, in Klamt units.

    Attributes
    ----------
    areas : ndarray (n_seg,), angstrom^2
    sigma : ndarray (n_seg,), e/angstrom^2
        Averaged screening charge density, eq. 11.
    sigma_perp : ndarray (n_seg,), e/angstrom^2
        Correlation descriptor, eq. 14. Describes how a segment's charge
        density relates to its surroundings; the misfit energy needs it
        because neighbouring densities are not uncorrelated.
    sigma_raw : ndarray (n_seg,), e/angstrom^2
        Unaveraged ``q/s``, kept for diagnostics and for the dielectric-energy
        descriptors of eq. 12/13.
    """

    areas: np.ndarray
    sigma: np.ndarray
    sigma_perp: np.ndarray
    sigma_raw: np.ndarray

    @property
    def total_area(self) -> float:
        return float(np.sum(self.areas))


def _average_weights(
    positions_ang: np.ndarray, areas_ang2: np.ndarray, r_av: float
) -> np.ndarray:
    """The eq. 11 kernel ``w_{mu nu}``, unnormalised.

        r_mu^2 = s_mu / pi
        w = (r_mu^2 r_av^2 / (r_mu^2 + r_av^2))
            * exp(-d^2 / (r_mu^2 + r_av^2))

    A Gaussian of width set by the *sum* of the source segment's own radius
    and the averaging radius, prefactored so that large segments carry
    proportionally more weight.
    """
    r2 = np.asarray(areas_ang2, dtype=np.float64) / math.pi
    rav2 = float(r_av) ** 2
    denom = r2[:, None] + rav2                       # (mu, nu) via broadcast
    diff = positions_ang[:, None, :] - positions_ang[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return (r2[:, None] * rav2 / denom) * np.exp(-d2 / denom)


def average_sigma(
    positions_ang: np.ndarray,
    areas_ang2: np.ndarray,
    sigma_raw: np.ndarray,
    r_av: float,
) -> np.ndarray:
    """Klamt 1998 eq. 11: area-weighted Gaussian average of ``sigma``.

    ``sigma_nu = sum_mu sigma*_mu w_{mu nu} / sum_mu w_{mu nu}``.
    """
    w = _average_weights(positions_ang, areas_ang2, r_av)
    num = np.einsum("m,mn->n", np.asarray(sigma_raw, dtype=np.float64), w)
    den = np.einsum("mn->n", w)
    out = np.zeros_like(num)
    ok = den > 0.0
    out[ok] = num[ok] / den[ok]
    return out


def segment_descriptors(
    surface: ConductorSurface, r_av: float
) -> SegmentDescriptors:
    """Build the averaged descriptors for one conductor surface.

    ``sigma_perp`` follows eq. 14, ``sigma_perp = sigma_wide - 0.816 sigma``,
    with ``sigma_wide`` averaged over radius ``2 r_av`` (eq. 14's "area of
    radius 2 r_av"). Orthogonalising against ``sigma`` is what makes it an
    *independent* descriptor rather than a near-copy: the two raw averages are
    strongly correlated.
    """
    areas_b = np.asarray(surface.areas, dtype=np.float64)
    keep = areas_b > MIN_SEGMENT_AREA_BOHR2

    positions = np.asarray(surface.positions, dtype=np.float64)[keep] * BOHR_TO_ANG
    areas = areas_b[keep] * (BOHR_TO_ANG ** 2)
    # sigma = q / s: converting the area to angstrom^2 converts the density.
    sigma_raw = np.asarray(surface.sigma, dtype=np.float64)[keep] / (
        BOHR_TO_ANG ** 2
    )

    sigma = average_sigma(positions, areas, sigma_raw, r_av)
    sigma_wide = average_sigma(positions, areas, sigma_raw, 2.0 * r_av)
    sigma_perp = sigma_wide - SIGMA_PERP_ORTHOGONALISATION * sigma
    return SegmentDescriptors(
        areas=areas, sigma=sigma, sigma_perp=sigma_perp, sigma_raw=sigma_raw
    )


@dataclass(frozen=True)
class SigmaProfile:
    """``p(sigma)`` -- area per unit screening charge density.

    Attributes
    ----------
    sigma_grid : ndarray (n_bin,), e/angstrom^2
        Bin centres.
    p : ndarray (n_bin,), angstrom^2 / (e/angstrom^2)
        Area density, so ``sum(p) * dsigma == total_area``. Keeping ``p`` an
        area *density* rather than an area histogram is what makes the
        sigma-potential integral grid-spacing independent.
    total_area : float, angstrom^2
    label : str
    """

    sigma_grid: np.ndarray
    p: np.ndarray
    total_area: float
    label: str = ""

    @property
    def dsigma(self) -> float:
        if self.sigma_grid.size < 2:
            return 1.0
        return float(self.sigma_grid[1] - self.sigma_grid[0])

    @property
    def normalized(self) -> np.ndarray:
        """``p'(sigma) = p(sigma) / A`` -- the probability density of eq. 17."""
        if self.total_area <= 0.0:
            return np.zeros_like(self.p)
        return self.p / self.total_area

    def area(self) -> float:
        """Integrated area; equals :attr:`total_area` up to binning."""
        return float(np.sum(self.p) * self.dsigma)


def default_sigma_grid(
    sigma_max: float = 0.025, n_bins: int = 51
) -> np.ndarray:
    """A symmetric grid over ``[-sigma_max, +sigma_max]`` in e/angstrom^2.

    The default span covers the range Klamt 1998 Figure 3 plots for water,
    methanol, acetone, pentane and benzene (about -0.02 to +0.02), with
    headroom so that strongly polar surfaces are not clipped. Clipping is not
    a rounding error here: a truncated tail silently removes exactly the
    hydrogen-bonding segments the model cares most about.
    """
    return np.linspace(-abs(sigma_max), abs(sigma_max), int(n_bins))


def sigma_profile(
    descriptors: SegmentDescriptors,
    sigma_grid: np.ndarray | None = None,
    *,
    label: str = "",
) -> SigmaProfile:
    """Bin segment areas onto a sigma grid, linearly spreading each segment.

    Linear (rather than nearest-bin) assignment keeps the profile continuous
    in the segment charges, which matters because the sigma potential and every
    property derived from it are then continuous too -- a nearest-bin histogram
    makes a chemical potential jump when a segment crosses a bin edge.

    Segments outside the grid are clamped onto the end bins and the total area
    is conserved; a lost segment would silently change ``A^X``.
    """
    grid = default_sigma_grid() if sigma_grid is None else np.asarray(
        sigma_grid, dtype=np.float64
    )
    if grid.size < 2:
        raise ValueError("sigma_profile: need at least two grid points.")
    d = float(grid[1] - grid[0])

    p = np.zeros(grid.size, dtype=np.float64)
    pos = np.clip(
        (np.asarray(descriptors.sigma, dtype=np.float64) - grid[0]) / d,
        0.0,
        grid.size - 1.0,
    )
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, grid.size - 1)
    frac = pos - lo
    areas = np.asarray(descriptors.areas, dtype=np.float64)
    np.add.at(p, lo, areas * (1.0 - frac))
    np.add.at(p, hi, areas * frac)

    return SigmaProfile(
        sigma_grid=grid, p=p / d, total_area=float(np.sum(areas)), label=label
    )


def mixture_sigma_profile(
    profiles: list[SigmaProfile], mole_fractions: np.ndarray
) -> SigmaProfile:
    """Klamt 1998 eq. 16: ``p_S(sigma) = sum_i x_i p^i(sigma) / sum_i x_i``.

    The ensemble profile of a mixture is the mole-fraction-weighted sum of its
    components' profiles. Note this weights by *molecules*, so a component with
    a large surface contributes proportionally more area, which is the intended
    behaviour: the ensemble is a bag of surface patches, not of molecules.
    """
    if not profiles:
        raise ValueError("mixture_sigma_profile: no components.")
    x = np.asarray(mole_fractions, dtype=np.float64)
    if x.shape != (len(profiles),):
        raise ValueError(
            f"mixture_sigma_profile: {x.size} mole fractions for "
            f"{len(profiles)} components."
        )
    if np.any(x < 0.0):
        raise ValueError("mixture_sigma_profile: negative mole fraction.")
    total_x = float(np.sum(x))
    if total_x <= 0.0:
        raise ValueError("mixture_sigma_profile: mole fractions sum to zero.")

    grid = profiles[0].sigma_grid
    for prof in profiles[1:]:
        if prof.sigma_grid.shape != grid.shape or not np.allclose(
            prof.sigma_grid, grid
        ):
            raise ValueError(
                "mixture_sigma_profile: components use different sigma grids; "
                "build them on one grid so the ensemble integral is defined."
            )

    p = np.zeros_like(profiles[0].p)
    area = 0.0
    for xi, prof in zip(x, profiles):
        p += xi * prof.p
        area += xi * prof.total_area
    return SigmaProfile(
        sigma_grid=grid,
        p=p / total_x,
        total_area=area / total_x,
        label="mixture",
    )


__all__ = [
    "BOHR_TO_ANG",
    "SIGMA_PERP_ORTHOGONALISATION",
    "SegmentDescriptors",
    "SigmaProfile",
    "average_sigma",
    "default_sigma_grid",
    "mixture_sigma_profile",
    "segment_descriptors",
    "sigma_profile",
]
