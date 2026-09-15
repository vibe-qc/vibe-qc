"""Smooth assignment of basis points to atoms, with its adjoints (#771).

Step 5 of the COSMO FINE Cavity workflow (Klamt & Diedenhofen 2018,
doi:10.1002/jcc.25342) assigns each basis grid point to "one of the atoms based
on the smallest relative distance" ``tau_a = |r - R_a| / rho_a``. That
``argmin`` is discrete, so a point on a boundary flips at an infinitesimal
displacement and its *whole* area moves to the other atom's segment grid.
Measured on an asymmetric solute in the default frame: 0.19 bohr^2 moves at
once, the secant across the crossing is 6.9 percent wrong, and the energy jumps
by 2.9e-08 Ha. Crossings are frequent -- scanning twelve coordinates over
+/-6e-03 bohr finds 30 of them.

This is the same defect :mod:`vibeqc.solvation.segment_partition` removed one
level down (#757), and the answer is the same in shape: replace the hard
assignment with weights that sum to one. What is *not* the same is the pair
coordinate, and that is the whole design problem.

Why not Becke's coordinate
--------------------------
The obvious construction -- Becke's ``mu_ab = (d_a - d_b)/R_ab`` with the
atomic size adjustment of his appendix A, which is what vibe-qc's DFT grids use
(``cpp/src/grid.cpp``) -- was built and measured, and it **fails**: it moves 13
to 16 percent of the total surface area between atoms. On water, oxygen loses
29 bohr^2 to the hydrogens. That is not a smoothing of step 5, it is a
different assignment rule.

The reason is structural rather than a tuning failure. Becke's scheme is built
for three-dimensional quadrature grids, where a point sits *near its own atom*;
a CFC basis point stands off every atom by a comparable distance, so ``mu`` is
small -- and the pair therefore "ambiguous" -- for atoms that are not remotely
in contention. The size adjustment is not the problem: it is exactly right
where it can be checked, since ``nu(u) = 0`` identically at the tau boundary on
the internuclear axis.

The coordinate that works
-------------------------
Measure the offset from the boundary in units of the **marching spacing**, the
one length the construction already carries::

    s_ab  = (tau_a - tau_b) / (1/rho_a + 1/rho_b)   signed bohr to tau_a = tau_b
    mu_ab = s_ab / h                                 h = the marching spacing
    W_a   = prod_{b != a} s(mu_ab), normalized

``s_ab`` is a first-order signed distance to the ``tau_a = tau_b`` surface --
the gradient of ``tau_a - tau_b`` has magnitude ``1/rho_a + 1/rho_b`` where the
two are opposed, which is where the boundary is -- so ``mu_ab`` says how many
grid cells the point sits from the boundary. Three properties follow, and all
three were measured before this was written:

* **It smooths the rule rather than replacing it.** The per-atom area budget
  moves 0.93 percent (asymmetric solute) and 1.33 percent (water) from the hard
  assignment at ``h = 0.40 A``.
* **The smoothing shrinks with the grid**: 0.33 percent at ``h = 0.20 A``. This
  is the property that makes the choice defensible rather than arbitrary. The
  ambiguity being smoothed is a *discretization-level* ambiguity -- whether a
  basis point is on one side of a boundary is only meaningful to within the
  spacing that placed it -- so the fuzzy rule converges to the hard rule in the
  same limit the cavity itself converges. A coordinate built from distances
  alone (Becke's, or the ratio ``(tau_a - tau_b)/(tau_a + tau_b)``) has no such
  limit: it smooths by the same fraction however fine the grid.
* **It stays affordable.** A basis point feeds 1.80 atoms on average at
  ``h = 0.40 A`` and 1.37 at ``0.20 A``, against 1.00 for the hard rule and
  *all of them* for the tau ratio. The cost of step 6 therefore grows by under
  a factor of two, and improves with refinement.

No fitted parameter is introduced: ``h`` is already an input to
:func:`~vibeqc.solvation.fine_cavity.build_fine_cavity`, and the cutoff is the
switching function's own constant.

The switching function is Stratmann, Scuseria & Frisch, *Chem. Phys. Lett.*
**257**, 213 (1996) section 11, not Becke's iterated polynomial, because it
reaches exactly 0 and 1 at ``|mu| = 0.64`` -- that compact support is what
bounds the cost above. It is the same function as ``stratmann_switch`` in
``cpp/src/grid.cpp``. Note that :mod:`~vibeqc.solvation.segment_partition`
deliberately refuses to *truncate* its own product; that is not in tension with
compact support here. Truncating a product over nearest neighbours is
discontinuous, because which centres are nearest changes as the point moves.
This support is compact by construction: the factor goes smoothly to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "STRATMANN_A",
    "AtomPartition",
    "atom_partition",
    "stratmann_switch",
]

# Stratmann, Scuseria & Frisch 1996 section 11. The cutoff at which the
# switching function reaches exactly 0 and 1; the paper's value, and the same
# constant as ``kStratmannA`` in cpp/src/grid.cpp.
STRATMANN_A = 0.64


def stratmann_switch(mu: np.ndarray) -> np.ndarray:
    """``s(mu) = (1 - g(mu/a))/2``, exactly 1 below ``-a`` and 0 above ``+a``.

    ``g(z) = (35 z - 35 z^3 + 21 z^5 - 5 z^7)/16`` is the unique degree-7
    polynomial reaching ``+-1`` at ``z = +-1`` with its first three derivatives
    vanishing there, so ``s`` is C3 at the join. Byte-for-byte the same
    polynomial as ``stratmann_switch`` in cpp/src/grid.cpp.
    """
    z = np.clip(np.asarray(mu, dtype=np.float64) / STRATMANN_A, -1.0, 1.0)
    z2 = z * z
    return 0.5 * (1.0 - z * (35.0 - z2 * (35.0 - z2 * (21.0 - 5.0 * z2))) / 16.0)


def _switch_and_derivative(mu: np.ndarray):
    """``(s(mu), ds/dmu)``.

    ``dg/dz = (35/16)(1 - z^2)^3`` exactly, so ``ds/dmu`` vanishes to third
    order at the cutoff and is identically zero outside it -- which is what
    makes the compact support smooth rather than clipped.
    """
    m = np.asarray(mu, dtype=np.float64)
    z = np.clip(m / STRATMANN_A, -1.0, 1.0)
    z2 = z * z
    s = 0.5 * (1.0 - z * (35.0 - z2 * (35.0 - z2 * (21.0 - 5.0 * z2))) / 16.0)
    ds = -0.5 * (35.0 / 16.0) * (1.0 - z2) ** 3 / STRATMANN_A
    return s, np.where(np.abs(m) < STRATMANN_A, ds, 0.0)


@dataclass(frozen=True)
class AtomPartition:
    """Weights of basis points over atoms, plus their vector-Jacobian product.

    Attributes
    ----------
    weights : ndarray (n_points, n_atoms)
        ``W_{b,a}``, each row summing to one, and exactly zero for an atom more
        than ``0.64 h`` of boundary-offset away.
    """

    weights: np.ndarray
    _points: np.ndarray
    _positions: np.ndarray
    _radii: np.ndarray
    _spacing: float

    def row_sum_residual(self) -> float:
        """``max |sum_a W_{b,a} - 1|``. Exactly zero for a partition of unity."""
        return float(np.max(np.abs(self.weights.sum(axis=1) - 1.0)))

    def vjp(self, adjoint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(dG/d points, dG/d atom positions)`` for ``G`` with ``dG/dW = adjoint``.

        ``adjoint`` is ``(n_points, n_atoms)``; the results are
        ``(n_points, 3)`` and ``(n_atoms, 3)``.

        The chain is shorter than the segment partition's, because this
        coordinate does not depend on the vector *between* two centres -- only
        on the two relative distances::

            dW_a       = W_a [ dlnP_a - sum_c W_c dlnP_c ]
            dlnP_a     = sum_{c != a} (ds/dmu)/s . dmu_ac
            dmu_ac     = k_ac (dtau_a - dtau_c),  k_ac = 1/(h (1/rho_a + 1/rho_c))
            dtau_a/dr  = +nhat_a / rho_a
            dtau_a/dR_a= -nhat_a / rho_a

        so with ``b_a`` the reduced adjoint and ``E_ac = b_a (ds/dmu)/s k_ac``,
        the whole contraction collapses to one per-atom sum::

            g_a      = sum_c E_ac - sum_c E_ca
            dG/dr    = sum_a g_a nhat_a / rho_a
            dG/dR_a  = -g_a nhat_a / rho_a

        which is also why ``sum_a dG/dR_a + dG/dr = 0`` holds identically: a
        rigid translation of the point *and* every atom cannot change a
        partition built from relative distances. The tests use that instead of
        finite differences.
        """
        a_in = np.asarray(adjoint, dtype=np.float64)
        w = self.weights
        pts = self._points
        pos = self._positions
        rho = self._radii
        n_at = pos.shape[0]

        adj_points = np.zeros_like(pts)
        adj_positions = np.zeros_like(pos)
        if n_at == 1:
            return adj_points, adj_positions

        g = pts[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(g, axis=2)
        nhat = g / np.maximum(d, 1e-300)[:, :, None]
        tau = d / rho[None, :]
        k = 1.0 / (self._spacing * (1.0 / rho[:, None] + 1.0 / rho[None, :]))
        mu = (tau[:, :, None] - tau[:, None, :]) * k[None, :, :]
        sw, dsw = _switch_and_derivative(mu)
        eye = np.eye(n_at, dtype=bool)
        ratio = np.where(sw > 0.0, dsw / np.where(sw > 0.0, sw, 1.0), 0.0)
        ratio[:, eye] = 0.0

        b = w * (a_in - np.einsum("ba,ba->b", a_in, w)[:, None])
        e = b[:, :, None] * ratio * k[None, :, :]
        gvec = e.sum(axis=2) - e.sum(axis=1)              # (n_points, n_atoms)
        scaled = gvec / rho[None, :]
        adj_points = np.einsum("ba,bai->bi", scaled, nhat)
        adj_positions = -np.einsum("ba,bai->ai", scaled, nhat)
        return adj_points, adj_positions


def atom_partition(
    points: np.ndarray,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
    grid_spacing_bohr: float,
) -> AtomPartition:
    """Smooth step-5 cells over atoms; see the module docstring for the design.

    ``grid_spacing_bohr`` is the marching spacing the basis points came from,
    which sets the width of the sharing band. It is not a free parameter: pass
    the spacing the cavity was actually built with, or the partition describes a
    grid that does not exist.
    """
    pts = np.asarray(points, dtype=np.float64)
    pos = np.asarray(atom_positions, dtype=np.float64)
    rho = np.asarray(atom_radii, dtype=np.float64)
    h = float(grid_spacing_bohr)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"atom_partition: points must be (n, 3), got {pts.shape}")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"atom_partition: atom positions must be (m, 3), got {pos.shape}"
        )
    if rho.shape != (pos.shape[0],):
        raise ValueError(
            f"atom_partition: need one radius per atom; got {rho.shape} for "
            f"{pos.shape[0]} atoms."
        )
    if np.any(rho <= 0.0):
        raise ValueError("atom_partition: radii must be positive.")
    if not (h > 0.0):
        raise ValueError(
            f"atom_partition: the marching spacing must be positive, got {h}. "
            "It sets the width of the sharing band, so there is no meaningful "
            "zero-spacing limit to fall back on."
        )

    n_at = pos.shape[0]
    if n_at == 1:
        return AtomPartition(
            weights=np.ones((pts.shape[0], 1)), _points=pts, _positions=pos,
            _radii=rho, _spacing=h,
        )

    d = np.linalg.norm(pts[:, None, :] - pos[None, :, :], axis=2)
    tau = d / rho[None, :]
    k = 1.0 / (h * (1.0 / rho[:, None] + 1.0 / rho[None, :]))
    mu = (tau[:, :, None] - tau[:, None, :]) * k[None, :, :]
    sw = stratmann_switch(mu)
    sw[:, np.eye(n_at, dtype=bool)] = 1.0                 # no self factor
    p = np.prod(sw, axis=2)
    z = p.sum(axis=1)
    if np.any(z <= 0.0):
        raise RuntimeError(
            "atom_partition: every cell function vanished for some basis "
            "point, so the partition cannot be normalized. The atom with the "
            "smallest relative distance always keeps a factor of at least one "
            "half per pair, so this means the product underflowed -- too many "
            "atoms for a plain product of pair factors."
        )
    return AtomPartition(
        weights=p / z[:, None], _points=pts, _positions=pos, _radii=rho,
        _spacing=h,
    )
