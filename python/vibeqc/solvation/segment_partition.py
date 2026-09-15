"""Smooth assignment of basis points to segments, with its adjoints.

The COSMO FINE Cavity groups iso-surface basis points onto a per-atom segment
grid by a hard nearest-centre ``argmin`` (Klamt & Diedenhofen 2018,
doi:10.1002/jcc.25342, workflow steps 5 and 6). That makes the cavity -- and
therefore the energy -- **discontinuous**: a basis point sitting on the
boundary between two segment centres flips at an infinitesimal displacement
and its *entire* area moves to the other segment. Measured on an asymmetric
solute: three of 1230 basis points reassign across one boundary, one segment
weight moves by 0.333 bohr^2, and the energy jumps by ~6e-7 Ha (#757).

The paper does not address this. Its own treatment of related discrete
problems is to *avoid* differentiating them -- COC gradients are taken
"neglecting the change of the surface areas by small movements of the atoms",
triple-segment area gradients are "neglected", and exact symmetric
coincidences are broken by "a small geometrical noise function applied to the
atomic positions" (p. 1652). None of that helps a generic boundary crossing.

So the fix comes from elsewhere, and it is the canonical one for exactly this
problem: Becke's fuzzy cells (A. D. Becke, *J. Chem. Phys.* **88**, 2547
(1988), doi:10.1063/1.454033), which replace a discrete point-to-centre
assignment with smooth weights summing to one. vibe-qc already uses this
scheme for DFT molecular grids (``cpp/src/grid.cpp``, ``becke_switch``); this
module applies the same construction to segment centres instead of atoms, at
the same published smoothing order ``k = 3``.

It introduces no fitted parameter. Becke's ``mu`` is a *ratio* of distances,
so the smoothing width is set by the spacing between segment centres -- the
only length in the problem -- and ``k = 3`` is Becke's own published choice
and vibe-qc's existing ``GridOptions.becke_k`` default. Picking a width by
seeing which value made the tests pass is precisely what CLAUDE.md section 7
forbids, and is unnecessary here.

Two properties matter downstream and are exact rather than approximate:

* ``sum_S W_{S,b} = 1`` for every basis point, so the total surface area is
  conserved exactly -- load-bearing, since the CFC's area feeds COSMO-RS
  sigma-profiles.
* consequently ``sum_S dW_{S,b}/dx = 0``, so the *total* area derivative is
  untouched by the partition. That is a free consistency check on the
  adjoints, needing no finite differences.

Adjoints are exposed as vector-Jacobian products, not as Jacobians. The full
``dW/dc`` tensor is ``(n_points, n_centres, n_centres, 3)``, which for one
oxygen's 336 basis points and 162 segments is 209 MB; the VJP is
``(n_points, n_centres)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BECKE_K",
    "POINT_CHUNK",
    "SmoothPartition",
    "becke_switch",
    "smooth_partition",
]

# Becke 1988's iterated smoothing order. Also ``GridOptions.becke_k``'s
# default in cpp/include/vibeqc/grid.hpp, and pinned equal to it by
# tests/test_solvation_segment_partition.py so the two cannot drift.
BECKE_K = 3

# Points are processed in chunks so the (points, centres, centres) working
# tensors stay bounded; the result does not depend on the chunk size.
#
# There is deliberately **no** nearest-neighbour truncation of the product. It
# is tempting -- a far centre contributes a factor very close to 1 -- but
# ``s(mu) = 0`` requires ``mu >= 1``, i.e. the triangle inequality saturated,
# so distant factors are near 1 and not equal to it: truncating at the 16
# nearest centres was measured to change the weights by 1.2e-3. Worse, the
# truncation is itself discontinuous, because the m-th and (m+1)-th nearest
# centres swap as the point moves. That would reintroduce a smaller copy of
# the very defect this module exists to remove (#757), which is the one
# trade this code cannot make.
POINT_CHUNK = 64


def becke_switch(mu: np.ndarray, k: int = BECKE_K) -> np.ndarray:
    """``s(mu) = (1 - p^k(mu))/2`` with ``p(mu) = (3 mu - mu^3)/2``.

    Becke 1988 eq. 19-21. Byte-for-byte the same recurrence as
    ``becke_switch`` in ``cpp/src/grid.cpp``, which is what makes the DFT grid
    and the cavity share one definition rather than two that agree today.
    """
    m = np.asarray(mu, dtype=np.float64)
    for _ in range(int(k)):
        m = 1.5 * m - 0.5 * m * m * m
    return 0.5 * (1.0 - m)


def _becke_switch_and_derivative(mu: np.ndarray, k: int):
    """``(s(mu), ds/dmu)``.

    ``ds/dmu = -(1/2) prod_i p'(mu_i)`` over the iterates, with
    ``p'(mu) = (3/2)(1 - mu^2)``. Accumulated along the same recurrence rather
    than differentiated symbolically, so the value and the derivative cannot
    disagree about which iterate they are at.
    """
    m = np.asarray(mu, dtype=np.float64)
    dp = np.ones_like(m)
    for _ in range(int(k)):
        dp = dp * 1.5 * (1.0 - m * m)
        m = 1.5 * m - 0.5 * m * m * m
    return 0.5 * (1.0 - m), -0.5 * dp


@dataclass(frozen=True)
class SmoothPartition:
    """Becke weights of basis points over segment centres, plus their VJP.

    Attributes
    ----------
    weights : ndarray (n_points, n_centres)
        ``W_{b,S}``, each row summing to one.
    """

    weights: np.ndarray
    _points: np.ndarray
    _centres: np.ndarray
    _k: int

    def row_sum_residual(self) -> float:
        """``max |sum_S W_{b,S} - 1|``. Exactly zero for a partition of unity."""
        return float(np.max(np.abs(self.weights.sum(axis=1) - 1.0)))

    def vjp(self, adjoint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(dG/d points, dG/d centres)`` for ``G`` with ``dG/dW = adjoint``.

        ``adjoint`` is ``(n_points, n_centres)``; results are ``(n_points, 3)``
        and ``(n_centres, 3)``.

        With ``W_S = P_S / sum_U P_U`` and ``P_S = prod_{T != S} s(mu_ST)``::

            dW_S      = W_S [ dlnP_S - sum_U W_U dlnP_U ]
            dlnP_S    = sum_{T != S} (ds/dmu)/s . dmu_ST
            dmu_ST/dt = (nhat_S - nhat_T) / R_ST
            dmu_ST/dc_S = (-nhat_S - mu_ST chat_ST) / R_ST
            dmu_ST/dc_T = (+nhat_T + mu_ST chat_ST) / R_ST

        Writing ``E_ST = b_S (ds/dmu)/(s R_ST)`` with ``b_S`` the reduced
        adjoint, and using that ``mu_ST chat_ST`` is *symmetric* under
        ``S <-> T`` (both factors change sign), the whole contraction collapses
        to two per-centre sums::

            g_S       = sum_T E_ST - sum_T E_TS
            dG/dt     = sum_S g_S nhat_S
            dG/dc_S   = -g_S nhat_S + sum_T (E_TS - E_ST) mu_ST chat_ST

        which is what makes this an O(n_centres^2) contraction per point
        rather than a materialized Jacobian.
        """
        a = np.asarray(adjoint, dtype=np.float64)
        W = self.weights
        b = W * (a - np.einsum("bS,bS->b", a, W)[:, None])

        t_all, c = self._points, self._centres
        n_pts, n_cen = t_all.shape[0], c.shape[0]
        adj_points = np.zeros((n_pts, 3), dtype=np.float64)
        adj_centres = np.zeros((n_cen, 3), dtype=np.float64)
        if n_cen == 1:
            return adj_points, adj_centres

        cvec = c[:, None, :] - c[None, :, :]
        R = np.linalg.norm(cvec, axis=2)
        eye = np.eye(n_cen, dtype=bool)
        Rsafe = np.where(eye, 1.0, R)
        chat = cvec / Rsafe[:, :, None]

        for lo in range(0, n_pts, POINT_CHUNK):
            hi = min(lo + POINT_CHUNK, n_pts)
            t = t_all[lo:hi]
            g = t[:, None, :] - c[None, :, :]
            d = np.linalg.norm(g, axis=2)
            nhat = g / np.maximum(d, 1e-300)[:, :, None]
            mu = (d[:, :, None] - d[:, None, :]) / Rsafe[None, :, :]
            sw, dsw = _becke_switch_and_derivative(mu, self._k)
            ratio = np.where(sw > 0.0, dsw / np.where(sw > 0.0, sw, 1.0), 0.0)
            ratio[:, eye] = 0.0
            E = b[lo:hi][:, :, None] * ratio / Rsafe[None, :, :]

            gS = E.sum(axis=2) - E.sum(axis=1)            # (chunk, n_cen)
            adj_points[lo:hi] = np.einsum("bS,bSi->bi", gS, nhat)
            adj_centres -= np.einsum("bS,bSi->Si", gS, nhat)
            F = (E.transpose(0, 2, 1) - E) * mu           # (chunk, n_cen, n_cen)
            adj_centres += np.einsum("bST,STi->Si", F, chat)
        return adj_points, adj_centres


def smooth_partition(
    points: np.ndarray,
    centres: np.ndarray,
    *,
    k: int = BECKE_K,
) -> SmoothPartition:
    """Becke fuzzy-cell weights of ``points`` over ``centres``.

    ``centres`` must be distinct; coincident centres make the partition
    ill-defined in the same way two coincident atoms do in Becke's original
    scheme, and are reported rather than silently regularized.
    """
    t = np.asarray(points, dtype=np.float64)
    c = np.asarray(centres, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"smooth_partition: points must be (n, 3), got {t.shape}")
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"smooth_partition: centres must be (m, 3), got {c.shape}")
    n_cen = c.shape[0]
    if n_cen == 0:
        raise ValueError("smooth_partition: no centres to partition over.")
    if n_cen == 1:
        return SmoothPartition(
            weights=np.ones((t.shape[0], 1)), _points=t, _centres=c, _k=int(k)
        )

    cvec = c[:, None, :] - c[None, :, :]
    R = np.linalg.norm(cvec, axis=2)
    eye = np.eye(n_cen, dtype=bool)
    if np.any((R < 1e-300) & ~eye):
        raise ValueError(
            "smooth_partition: two segment centres coincide, so the Becke "
            "ratio mu = (d_S - d_T)/|c_S - c_T| is undefined. Merge them "
            "before partitioning."
        )
    Rsafe = np.where(eye, 1.0, R)

    W = np.empty((t.shape[0], n_cen), dtype=np.float64)
    for lo in range(0, t.shape[0], POINT_CHUNK):
        hi = min(lo + POINT_CHUNK, t.shape[0])
        d = np.linalg.norm(t[lo:hi, None, :] - c[None, :, :], axis=2)
        mu = (d[:, :, None] - d[:, None, :]) / Rsafe[None, :, :]
        sw = becke_switch(mu, k)
        sw[:, eye] = 1.0                                  # no self factor
        P = np.prod(sw, axis=2)
        Z = P.sum(axis=1)
        if np.any(Z <= 0.0):
            raise ValueError(
                "smooth_partition: every cell function vanished for some "
                "point, so the partition cannot be normalized."
            )
        W[lo:hi] = P / Z[:, None]
    return SmoothPartition(weights=W, _points=t, _centres=c, _k=int(k))
