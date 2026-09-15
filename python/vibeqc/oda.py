"""Optimal Damping Algorithm (ODA) for periodic SCF density mixing.

Cancès & Le Bris, *Int. J. Quantum Chem.* **79**, 82 (2000) --
"Can we outperform the DIIS approach for electronic structure
calculations?"

At each SCF iter, instead of accepting the full step D_n -> D_naive
(from diagonalising F(D_n)) or applying a fixed-a damping
``D_n+1 = a.D_n + (1-a).D_naive``, ODA finds the **optimal** mixing
fraction l in [0, 1] that minimises a quadratic model of the energy
along the linear path ``D(l) = (1-l).D_n + l.D_naive``.

The quadratic model uses two Fock-matrix evaluations: F_n = F(D_n)
(already computed for the energy) and F_naive = F(D_naive) (one
extra Fock build per iter -- the cost of ODA). For RHF:

    E(D(l)) - E(D_n)  ≈  l.g0  +  1/2.l^2.(g1 - g0)
    g0  =  S_k w_k . Re tr( (D_naive(k) - D_n(k)) . F_n(k) )
    g1  =  S_k w_k . Re tr( (D_naive(k) - D_n(k)) . F_naive(k) )

The minimum on [0, 1] is

    l* = clamp(g0 / (g0 - g1), 0, 1)

If g1 - g0 <= 0 (energy decreasing at constant rate or accelerating),
the unconstrained minimiser is >= 1, so we take the full step l* = 1.
If g0 >= 0 (D_naive is uphill), the unconstrained minimiser is <= 0,
so we hold at l* = 0 (no update) -- should never happen if F_n was
correctly built and SCF is descending.

Why ODA on periodic ionic SCFs: vibe-qc's multi-k SCF on tight
ionic primitives (LiH/STO-3G at kmesh=(2,2,2)) without stabilisers
violently oscillates iter-to-iter between the right basin (~-8 Ha)
and progressively worse over-bound spikes (-15, -36, -53, -80 Ha by
iter 16; see commit-history diag artifacts). Fixed-a damping
smooths the spikes but introduces a directional bias that drifts the
SCF away from the basin. DIIS extrapolates toward an SCF fixed point
that doesn't match what MOM is tracking, so they fight. ODA's l
search adapts the mixing fraction per iter without committing to
either dynamic. Particularly robust when combined with MOM
(orthogonal: MOM picks which orbitals occupy; ODA picks how
aggressively to step toward those orbitals).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ODAStep:
    """Result of one ODA mixing step.

    Attributes
    ----------
    lam
        Selected mixing fraction l in [0, 1].
    g0
        Energy slope at D_n along the path: S_k w_k Re tr(ΔD(k) F_n(k)).
        Should be negative (descent direction) for a healthy SCF iter.
    g1
        Energy slope at D_naive along the path. ``g1 > g0`` (positive
        curvature) means the energy model has a finite minimiser on
        the line; ``g1 <= g0`` means a full step (l=1) is optimal.
    quadratic_minimum
        ``True`` when ``0 < l < 1`` -- ODA picked an interior point
        (the curvature term mattered). ``False`` when clamped.
    """
    lam: float
    g0: float
    g1: float
    quadratic_minimum: bool


def _bloch_sum_to_k(D_lat, k_cart: np.ndarray) -> np.ndarray:
    """Bloch-sum a LatticeMatrixSet at one k-point.

    Computes ``D(k) = S_g D(g) . e^{i k.R_g}`` where ``R_g`` is the
    Cartesian translation of cell ``g``. Returns the per-k (n_bf,
    n_bf) complex matrix. Mirrors `_bloch_sum_blocks` in
    `vibeqc.periodic_fock_multi_k`.
    """
    k = np.asarray(k_cart, dtype=float).reshape(3)
    cells = D_lat.cells
    blk0 = np.asarray(D_lat.blocks[0])
    out = np.zeros_like(blk0, dtype=complex)
    for g_idx in range(len(cells)):
        R_g = np.asarray(cells[g_idx].r_cart, dtype=float)
        phase = np.exp(1j * float(np.dot(k, R_g)))
        out = out + phase * np.asarray(D_lat.blocks[g_idx])
    return out


def compute_oda_lambda(
    D_n_lat,
    D_naive_lat,
    F_n_k_list: Sequence[np.ndarray],
    F_naive_k_list: Sequence[np.ndarray],
    k_points_cart: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    trust_lambda_max: float = 1.0,
) -> ODAStep:
    """Compute the optimal ODA mixing fraction l in [0, 1].

    Parameters
    ----------
    D_n_lat
        Real-space density (LatticeMatrixSet) used to build ``F_n``
        at this SCF iter -- i.e. the density entering the iter.
    D_naive_lat
        Real-space density built from the freshly-diagonalised MOs
        (the "naive" full-step density that standard SCF would accept).
    F_n_k_list
        Per-k Fock matrices computed at ``D_n``. Length matches
        ``k_points_cart``; each entry shape ``(n_bf, n_bf)`` complex.
    F_naive_k_list
        Per-k Fock matrices computed at ``D_naive`` -- the EXTRA
        Fock build that distinguishes ODA from cheaper accelerators.
    k_points_cart
        Cartesian k-vectors, shape ``(3,)`` each. Used to Bloch-sum
        ``D_n - D_naive`` per k for the energy-slope integrals.
    weights
        Per-k Brillouin-zone weights; must sum to 1.

    Returns
    -------
    :class:`ODAStep` carrying the selected ``lam`` plus diagnostics
    (``g0``, ``g1``, whether the minimum was interior).
    """
    n_k = len(F_n_k_list)
    if len(F_naive_k_list) != n_k:
        raise ValueError(
            f"compute_oda_lambda: F_naive_k_list has {len(F_naive_k_list)} "
            f"entries but F_n_k_list has {n_k}"
        )
    if len(k_points_cart) != n_k:
        raise ValueError(
            f"compute_oda_lambda: k_points_cart has {len(k_points_cart)} "
            f"entries but F_n_k_list has {n_k}"
        )
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_k,):
        raise ValueError(
            f"compute_oda_lambda: weights shape {w.shape} != ({n_k},)"
        )

    g0 = 0.0
    g1 = 0.0
    for k_idx in range(n_k):
        k_cart = np.asarray(k_points_cart[k_idx], dtype=float)
        D_n_k = _bloch_sum_to_k(D_n_lat, k_cart)
        D_naive_k = _bloch_sum_to_k(D_naive_lat, k_cart)
        delta_D = D_naive_k - D_n_k
        F_n_k = np.asarray(F_n_k_list[k_idx])
        F_naive_k = np.asarray(F_naive_k_list[k_idx])
        # tr(A.B) where A, B may be complex. For Hermitian A, B the
        # trace is real; np.real strips any rounding noise.
        g0 += float(w[k_idx]) * float(np.real(np.trace(delta_D @ F_n_k)))
        g1 += float(w[k_idx]) * float(np.real(np.trace(delta_D @ F_naive_k)))

    # Quadratic model: E(l) = l.g0 + 1/2.l^2.(g1 - g0). Pick the
    # minimiser on the trust-capped interval [0, trust_lambda_max].
    #
    #   curvature = g1 - g0
    #   * curvature > 0  (convex): unconstrained minimiser at
    #       l_c = g0 / (g0 - g1). Clamp to [0, trust_max].
    #   * curvature <= 0  (concave or linear): minimum lies at one of
    #       the interval endpoints (concave on a finite interval has
    #       no interior min). Pick the endpoint with lower model E.
    #       E(0) = 0, E(trust_max) = trust_max.g0 + 1/2.trust_max^2.(g1-g0).
    #
    # The earlier `lam_raw < 0 => l = 0` fallback was buggy: when
    # g0 < 0 and g1 < 0, lam_raw goes negative because the model is
    # concave (g1 - g0 < 0), and the correct choice is l = 1 (the
    # entire path is downhill) -- NOT l = 0 (which freezes the SCF
    # and leads to false convergence at the initial-guess basin).
    if not (0.0 < trust_lambda_max <= 1.0):
        raise ValueError(
            f"compute_oda_lambda: trust_lambda_max must be in (0, 1]; "
            f"got {trust_lambda_max}"
        )
    interior = False
    curvature = g1 - g0
    if curvature > 1e-14:
        lam_raw = g0 / (g0 - g1)
        if lam_raw <= 0.0:
            lam = 0.0
        elif lam_raw >= trust_lambda_max:
            lam = float(trust_lambda_max)
        else:
            lam = float(lam_raw)
            interior = True
    else:
        # Concave or linear on the interval -- endpoint is best.
        e_at_trust = (
            trust_lambda_max * g0
            + 0.5 * trust_lambda_max * trust_lambda_max * curvature
        )
        lam = float(trust_lambda_max) if e_at_trust < 0.0 else 0.0
    return ODAStep(lam=float(lam), g0=g0, g1=g1, quadratic_minimum=interior)


def oda_mix_densities(D_n_lat, D_naive_lat, lam: float):
    """In-place ODA mix: ``D_n <- (1-l).D_n + l.D_naive`` block-wise.

    Mutates ``D_n_lat`` via ``LatticeMatrixSet.set_block``. Returns
    the mutated ``D_n_lat`` for convenience.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"oda_mix_densities: lam must be in [0, 1]; got {lam}")
    if lam == 1.0:
        # Copy all blocks from D_naive into D_n (full step).
        n_cells = len(D_n_lat.cells)
        for g_idx in range(n_cells):
            D_n_lat.set_block(
                g_idx, np.asarray(D_naive_lat.blocks[g_idx]).copy(),
            )
        return D_n_lat
    if lam == 0.0:
        return D_n_lat
    n_cells = len(D_n_lat.cells)
    for g_idx in range(n_cells):
        blk_n = np.asarray(D_n_lat.blocks[g_idx])
        blk_naive = np.asarray(D_naive_lat.blocks[g_idx])
        D_n_lat.set_block(g_idx, (1.0 - lam) * blk_n + lam * blk_naive)
    return D_n_lat


__all__ = ["ODAStep", "compute_oda_lambda", "oda_mix_densities"]
