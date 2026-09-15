"""Gilat-Raubenheimer Brillouin-zone integration for metallic occupations.

Implements the Gilat-Raubenheimer (GR) quadrature for the T=0 Fermi-surface
integral that fixes fractional band occupations and the Fermi level on a
regular k-mesh. This is the integrator behind CRYSTAL's ``SHRINK IS ISP``
second (Gilat) net, and the parameter-free alternative to the temperature
broadening in :mod:`vibeqc.smearing`.

References
----------
* G. Gilat and L. J. Raubenheimer, "Accurate Numerical Method for Calculating
  Frequency-Distribution Functions in Solids", *Phys. Rev.* **144**, 390
  (1966), doi:10.1103/PhysRev.144.390
  (library: ``pbc/1966_gilat_accurate-numerical-method-calculating-frequency``).
* G. Gilat, "Analysis of Methods for Calculating Spectral Properties in
  Solids", *J. Comput. Phys.* **10**, 432 (1972),
  doi:10.1016/0021-9991(72)90046-0
  (library: ``pbc/1972_gilat_analysis-methods-calculating-spectral-properties``).

Method (occupations)
--------------------
The BZ is partitioned into one parallelepiped microcell per regular-mesh
k-point. Within the cell centred at ``k0`` the band is linearised (GR 1966),

    eps_n(k0 + dk) ~= eps_n(k0) + v_n(k0) . dk ,   v_n = grad_k eps_n ,

so the cell's occupied volume is the fraction below the Fermi level ``E_F``.
Writing the offset in the cell's fractional coordinates ``t_i`` in
``[-1/2, 1/2]`` (the i-th mesh axis), the linear band becomes

    eps - eps_0 = sum_i g_i t_i ,   g_i = v_n . (b_i / N_i) ,

where ``b_i`` are the reciprocal-lattice vectors and ``N_i`` the mesh sizes.
The directional slope ``g_i`` equals the periodic central difference of the
eigenvalue grid along mesh axis ``i`` (the energy change across one cell
width), so the eigenvalue grid alone determines the geometry -- the cell
Jacobian is constant and cancels in the *fraction*, hence the reciprocal-
lattice shape (including non-orthogonality) is fully carried by the ``g_i``.

The occupied fraction is then the volume of the unit box
``{t in [-1/2, 1/2]^3}`` below the plane ``sum_i g_i t_i = E_F - eps_0`` --
a box cut by a half-space, equivalently the CDF of a weighted sum of three
uniform variables (a piecewise cubic). That fraction ``f in [0, 1]`` times the
spin degeneracy is the GR occupation at the grid point; uniform mesh weights
``1/N`` reproduce the BZ integral. ``E_F`` is found by bisecting the total
electron count to the target. GR carries no smearing width and no electronic
entropy (sharp Fermi surface) -- that is the point of the method, versus
Fermi-Dirac smearing which needs a width and a 0 K extrapolation.

Validation: the box-cut-by-half-space volume in :func:`occupied_fraction` is
checked against direct Monte-Carlo cell sampling, and the GR electron count on
a coarse mesh is checked to be closer to the dense-mesh reference than naive
step counting, in ``tests/test_bz_integration_gilat.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Absolute slope floor (Hartree per cell width): protects denominators; far
# below any SCF energy tolerance.
_FLAT_TOL = 1e-12
# A direction is treated as flat (and its box dimension dropped) when its slope
# is below this fraction of the steepest slope in the same cell. Keeps the 3D
# closed form away from catastrophic cancellation; the dropped-direction error
# is bounded by this ratio.
_REL_TOL = 1e-9


def occupied_fraction(
    delta: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    g3: np.ndarray,
) -> np.ndarray:
    """Fraction of a microcell below the Fermi level for a linearised band.

    Returns ``Vol{ t in [-1/2, 1/2]^3 : g1 t1 + g2 t2 + g3 t3 < delta }``,
    where ``delta = E_F - eps_0`` and ``g_i`` are the directional band slopes
    across the cell (see module docstring). All arguments broadcast against
    each other; the result is clipped to ``[0, 1]``.

    Closed form (GR 1966; box cut by a half-space / CDF of a weighted sum of
    three uniforms). By the symmetry of ``t`` about 0 only ``A_i = |g_i|``
    enters. Shifting ``u_i = t_i + 1/2 in [0, 1]`` gives the half-space level
    ``y = delta + (A1 + A2 + A3) / 2`` and

        Vol = 1/(6 A1 A2 A3) *
              sum_{S subset {1,2,3}} (-1)^|S| relu(y - sum_{i in S} A_i)^3 ,

    valid for ``A_i > 0``. The 3D closed form cancels three O(y^3) cubes down
    to the answer, which is catastrophic in float64 when one slope is tiny
    relative to the others (a band nearly flat along that axis). So directions
    whose slope is negligible relative to the steepest are dropped and the
    matching lower-dimensional box formula is used instead -- 2D
    ``1/(2 A0 A1) sum (-1)^|S| relu(y - A_S)^2``, 1D ``clip(delta/A0 + 1/2)``,
    0D (fully flat cell) an exact step. The dropped-direction error is bounded
    by the slope ratio (``< _REL_TOL``), far below any SCF tolerance.
    """
    delta_b, a1_b, a2_b, a3_b = np.broadcast_arrays(
        np.asarray(delta, dtype=float),
        np.abs(np.asarray(g1, dtype=float)),
        np.abs(np.asarray(g2, dtype=float)),
        np.abs(np.asarray(g3, dtype=float)),
    )
    d = delta_b
    # sort slopes descending so A0 is the steepest direction in each cell
    a_sorted = np.sort(np.stack([a1_b, a2_b, a3_b], axis=-1), axis=-1)[..., ::-1]
    a0 = a_sorted[..., 0]
    a1 = a_sorted[..., 1]
    a2 = a_sorted[..., 2]

    # a direction is "dispersive" if its slope is non-negligible vs the
    # steepest; the count selects the box dimensionality per cell
    thresh = np.maximum(_FLAT_TOL, _REL_TOL * a0)
    n_disp = (a_sorted > thresh[..., None]).sum(axis=-1)

    a0f = np.maximum(a0, _FLAT_TOL)
    a1f = np.maximum(a1, _FLAT_TOL)
    a2f = np.maximum(a2, _FLAT_TOL)

    def relu2(x: np.ndarray) -> np.ndarray:
        x = np.maximum(x, 0.0)
        return x * x

    def relu3(x: np.ndarray) -> np.ndarray:
        x = np.maximum(x, 0.0)
        return x * x * x

    # 0D: flat cell -> step at delta = 0 (half-occupied exactly at E_F)
    f0 = np.clip(0.5 * np.sign(d) + 0.5, 0.0, 1.0)
    # 1D: only the steepest direction disperses
    f1 = np.clip(d / a0f + 0.5, 0.0, 1.0)
    # 2D: two dispersive directions (third axis integrates to a factor 1)
    y2 = d + 0.5 * (a0 + a1)
    f2 = np.clip(
        (relu2(y2) - relu2(y2 - a0f) - relu2(y2 - a1f) + relu2(y2 - a0f - a1f))
        / (2.0 * a0f * a1f),
        0.0,
        1.0,
    )
    # 3D: all three directions dispersive (well-conditioned since all > thresh)
    y3 = d + 0.5 * (a0 + a1 + a2)
    f3 = np.clip(
        (
            relu3(y3)
            - relu3(y3 - a0f)
            - relu3(y3 - a1f)
            - relu3(y3 - a2f)
            + relu3(y3 - a0f - a1f)
            + relu3(y3 - a0f - a2f)
            + relu3(y3 - a1f - a2f)
            - relu3(y3 - a0f - a1f - a2f)
        )
        / (6.0 * a0f * a1f * a2f),
        0.0,
        1.0,
    )

    return np.where(
        n_disp == 0,
        f0,
        np.where(n_disp == 1, f1, np.where(n_disp == 2, f2, f3)),
    )


def grid_band_slopes(
    eps_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional band slopes ``g_i`` on the regular k-mesh.

    ``g_i`` is the periodic central difference of ``eps_grid`` along mesh axis
    ``i`` -- the energy change across one cell width, ``~= v_n . (b_i / N_i)``
    (see module docstring). ``eps_grid`` has shape ``(N1, N2, N3, nband)``.

    A mesh axis of size 1 (e.g. a 2D slab folded into a 3D array, or a Gamma-
    only axis) carries no dispersion, so its slope is identically zero.
    """
    eps = np.asarray(eps_grid, dtype=float)
    if eps.ndim != 4:
        raise ValueError(
            "grid_band_slopes: eps_grid must have shape (N1, N2, N3, nband); "
            f"got {eps.shape}"
        )
    n1, n2, n3 = eps.shape[:3]

    # Central difference (eps(+1) - eps(-1)) / 2 across one cell width, with
    # periodic wrap (the BZ is periodic in the reciprocal lattice).
    g1 = 0.5 * (np.roll(eps, -1, axis=0) - np.roll(eps, 1, axis=0))
    g2 = 0.5 * (np.roll(eps, -1, axis=1) - np.roll(eps, 1, axis=1))
    g3 = 0.5 * (np.roll(eps, -1, axis=2) - np.roll(eps, 1, axis=2))

    if n1 == 1:
        g1 = np.zeros_like(eps)
    if n2 == 1:
        g2 = np.zeros_like(eps)
    if n3 == 1:
        g3 = np.zeros_like(eps)
    return g1, g2, g3


def gilat_dos(
    eps_grid: np.ndarray,
    energies: np.ndarray,
    spin_degeneracy: float = 2.0,
    step: float = 0.0,
) -> np.ndarray:
    """Gilat-Raubenheimer density of states ``D(E)`` on the full regular mesh.

    ``D(E) = d N(E) / dE``, where ``N(E)`` is the GR cell-quadrature count of
    states below ``E`` (the same occupied-fraction integral used for
    occupations). This is the original purpose of the method (Gilat &
    Raubenheimer 1966 -- accurate frequency-distribution / state-distribution
    functions) and the parameter-free DOS a CRYSTAL Gilat net produces.

    Evaluated by central finite difference of the (piecewise-cubic, hence
    smooth) cumulative count; ``step`` defaults to ~1/200 of the eigenvalue
    span. ``eps_grid`` is ``(N1, N2, N3, nband)``; returns ``D`` aligned with
    ``energies`` (states per unit energy per cell).

    Use post-SCF on converged eigenvalues -- e.g. after a smearing-converged
    metal SCF -- to get the parameter-free DOS without a smearing width.
    """
    eps = np.asarray(eps_grid, dtype=float)
    if eps.ndim != 4:
        raise ValueError("gilat_dos: eps_grid must have shape (N1, N2, N3, nband)")
    g = float(spin_degeneracy)
    g1, g2, g3 = grid_band_slopes(eps)
    n_cells = eps.shape[0] * eps.shape[1] * eps.shape[2]
    en = np.asarray(energies, dtype=float)

    h = float(step)
    if h <= 0.0:
        span = float(eps.max() - eps.min())
        h = (span / 200.0) if span > 0.0 else 1e-3

    def count_below(e_fermi: float) -> float:
        frac = occupied_fraction(e_fermi - eps, g1, g2, g3)
        return g * float(frac.sum()) / n_cells

    dos = np.array(
        [(count_below(e + h) - count_below(e - h)) / (2.0 * h) for e in en],
        dtype=float,
    )
    # DOS is physically non-negative; clamp tiny negative values. The slopes
    # come from energy-ordered bands (grid_band_slopes), so band crossings can
    # give a spurious central-difference slope and a small non-monotonicity in
    # the cumulative count -- a known band-ordering artifact of the simple GR
    # DOS, bounded away from the Fermi-surface integration that drives E_F.
    return np.maximum(dos, 0.0)


def gilat_raubenheimer_occupations(
    eps_grid: np.ndarray,
    n_electrons_per_cell: float,
    spin_degeneracy: float = 2.0,
    *,
    ef_tol: float = 1e-12,
    max_bisect: int = 200,
) -> Tuple[np.ndarray, float]:
    """Gilat-Raubenheimer fractional occupations + Fermi level on a regular mesh.

    Parameters
    ----------
    eps_grid
        Band eigenvalues on the *full* regular BZ mesh, shape
        ``(N1, N2, N3, nband)``, in Hartree. (IBZ-reduced meshes must be
        scattered to the full grid first -- ``eps_n(R k) = eps_n(k)`` -- which
        is the increment-2 wiring step.)
    n_electrons_per_cell
        Target electron count per unit cell (the BZ-integral constraint).
    spin_degeneracy
        ``2.0`` (default) closed-shell: occupations in ``[0, 2]``. ``1.0`` for
        a single open-shell spin channel (occupations in ``[0, 1]``).

    Returns
    -------
    (occ_grid, e_fermi)
        ``occ_grid`` has the shape of ``eps_grid`` with occupations in
        ``[0, spin_degeneracy]``; ``e_fermi`` is the Fermi level in Hartree.

    Notes
    -----
    No entropy is returned: GR is a sharp-Fermi-surface (T=0) method, so the
    Mermin ``-TS`` free-energy term is zero. The density build consumes
    ``occ_grid`` with uniform mesh weights ``1/(N1 N2 N3)``, exactly as it
    consumes Fermi-Dirac occupations with their k-weights.
    """
    eps = np.asarray(eps_grid, dtype=float)
    if eps.ndim != 4:
        raise ValueError(
            "gilat_raubenheimer_occupations: eps_grid must have shape "
            f"(N1, N2, N3, nband); got {eps.shape}"
        )
    g = float(spin_degeneracy)
    if g <= 0.0:
        raise ValueError(
            "gilat_raubenheimer_occupations: spin_degeneracy must be > 0"
        )

    n1, n2, n3, nband = eps.shape
    n_cells = n1 * n2 * n3
    target = float(n_electrons_per_cell)
    max_electrons = g * float(nband)
    if target < -1e-12 or target > max_electrons + 1e-12:
        raise ValueError(
            "gilat_raubenheimer_occupations: electron count "
            f"{target} is outside the band capacity [0, {max_electrons}]"
        )

    g1, g2, g3 = grid_band_slopes(eps)

    # Integer endpoints: the bisection cannot bracket an empty or completely
    # full manifold (E_F -> -/+ inf). Both are exact, so short-circuit.
    if target <= 1e-12:
        return np.zeros_like(eps), float(eps.min())
    if target >= max_electrons - 1e-12:
        return np.full_like(eps, g), float(eps.max())

    def electrons(e_fermi: float) -> float:
        # total = sum_cells (1/n_cells) * sum_bands g * f  =  g * sum(f) / n_cells
        frac = occupied_fraction(e_fermi - eps, g1, g2, g3)
        return g * float(frac.sum()) / n_cells

    # The cell linearisation spreads each level over +/- one cell width, so the
    # support extends a little past [eps.min(), eps.max()]; pad the bracket.
    span = float(eps.max() - eps.min())
    pad = max(1.0, span)
    lo = float(eps.min()) - pad
    hi = float(eps.max()) + pad
    for _ in range(int(max_bisect)):
        mid = 0.5 * (lo + hi)
        if electrons(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < ef_tol:
            break
    # Use the upper bracket: by construction electrons(hi) >= target, whereas
    # electrons(lo) < target. For a gapped/flat system the bisection squeezes
    # both onto a band energy, and a flat cell's occupation is a step there
    # (delta == 0 is a knife's edge: sign(0)=0 -> half-fill, and float noise can
    # drop it to the empty side -> a band that should be full reads as empty).
    # Taking hi keeps delta strictly positive for such a band, so it counts as
    # occupied and the electron count is satisfied. For a dispersive metal hi
    # and lo agree to ef_tol, so this is a no-op there.
    e_fermi = hi

    occ = g * occupied_fraction(e_fermi - eps, g1, g2, g3)
    return occ, e_fermi
