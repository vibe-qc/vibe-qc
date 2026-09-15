"""Density-space mixing for periodic SCF -- the v0.10.x metal-mixing program (D4).

Periodic *metals* are the motivation: a sharp Fermi surface makes the SCF map
oscillate (occupations flip as bands cross E_F each cycle), and linear density
damping + Fock-DIIS -- what the multi-k Ewald drivers ship -- cannot break the
resulting limit cycle (confirmed empirically for the Gilat-Raubenheimer net on
Be fcc: cold, heavy-damped, and warm-started runs all oscillate). Anderson /
Pulay density mixing extrapolates over a *history* of density residuals and can
converge where linear mixing only oscillates.

This module provides the engine-agnostic core: :class:`AndersonMixer`, a
pure-vector Anderson (Type-II / Pulay) mixer, and :class:`BroydenMixer`, a
limited-memory Broyden (second-method) quasi-Newton mixer with the same
interface. The multi-k driver flattens its per-k density matrices into a vector
via :func:`per_k_density_to_vector`, feeds ``(x_in, x_out)`` each iteration,
splits the returned next input back with :func:`vector_to_per_k_density`, and
rebuilds the real-space density from the mixed ``D(k)``. The Kerker
preconditioner (D4b, small-G charge-sloshing damping) builds on this.

References:
  * D. G. Anderson, "Iterative Procedures for Nonlinear Integral Equations",
    J. ACM 12, 547 (1965), doi:10.1145/321296.321305.
  * C. G. Broyden, "A class of methods for solving nonlinear simultaneous
    equations", Math. Comput. 19, 577 (1965), doi:10.2307/2003941. The
    limited-memory density-mixing form follows D. D. Johnson, Phys. Rev. B
    38, 12807 (1988), doi:10.1103/PhysRevB.38.12807.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class AndersonMixer:
    """Anderson (Type-II / Pulay) mixing on a flat real vector.

    For the SCF fixed-point map ``x_out = g(x_in)`` with residual
    ``f = x_out - x_in``, the next input is

        x_next = sum_i theta_i x_i  +  beta * sum_i theta_i f_i ,

    where ``theta`` minimises ``|| sum_i theta_i f_i ||`` subject to
    ``sum_i theta_i = 1`` over the last ``depth`` iterations (the DIIS/Pulay
    coefficient solve). ``depth = 1`` reduces exactly to linear mixing
    ``x_next = (1 - beta) x_in + beta x_out``.

    Parameters
    ----------
    depth:
        Maximum residual-history length (Anderson "m"). Larger captures more
        curvature but costs a bigger (and possibly ill-conditioned) solve.
    beta:
        Mixing parameter in ``(0, 1]``. Smaller is more conservative.
    """

    def __init__(self, depth: int = 8, beta: float = 0.5) -> None:
        if depth < 1:
            raise ValueError("AndersonMixer: depth must be >= 1")
        if not (0.0 < beta <= 1.0):
            raise ValueError("AndersonMixer: beta must be in (0, 1]")
        self.depth = int(depth)
        self.beta = float(beta)
        self._x: List[np.ndarray] = []
        self._f: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear the history (e.g. when restarting or switching phase)."""
        self._x.clear()
        self._f.clear()

    @property
    def history_size(self) -> int:
        return len(self._f)

    def update(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        """Return the next input given this iteration's input and SCF output.

        ``x_in`` is the density fed to the SCF map this iteration; ``x_out`` is
        the density it produced. Both are flattened to 1-D internally.
        """
        xi = np.asarray(x_in, dtype=float).reshape(-1)
        xo = np.asarray(x_out, dtype=float).reshape(-1)
        if xi.shape != xo.shape:
            raise ValueError(
                f"AndersonMixer: x_in {xi.shape} and x_out {xo.shape} differ"
            )
        f = xo - xi
        self._x.append(xi)
        self._f.append(f)
        if len(self._x) > self.depth:
            self._x.pop(0)
            self._f.pop(0)

        m = len(self._f)
        if m == 1:
            # First step (or post-reset): plain linear mixing.
            return xi + self.beta * f

        F = np.stack(self._f, axis=1)            # (n, m)
        B = F.T @ F                              # (m, m), Gram of residuals
        # Tikhonov regularisation keeps the constrained solve well-posed when
        # residuals become nearly linearly dependent near convergence.
        scale = float(np.trace(B)) / m
        if scale <= 0.0:
            return xi + self.beta * f
        B = B + 1e-10 * scale * np.eye(m)
        ones = np.ones(m)
        try:
            z = np.linalg.solve(B, ones)
        except np.linalg.LinAlgError:
            return xi + self.beta * f
        denom = float(ones @ z)
        if abs(denom) < 1e-300:
            return xi + self.beta * f
        theta = z / denom                        # sum(theta) = 1, min ||F theta||

        X = np.stack(self._x, axis=1)            # (n, m)
        x_next = X @ theta + self.beta * (F @ theta)
        return np.asarray(x_next, dtype=float)


class BroydenMixer:
    """Limited-memory Broyden (second method) mixing on a flat real vector.

    Quasi-Newton fixed-point solver for the SCF map ``x_out = g(x_in)`` with
    residual ``f(x) = g(x) - x``. It maintains an approximate *inverse*
    Jacobian ``G_k`` of ``f`` through Broyden's second (Type-II) rank-1
    updates and takes the quasi-Newton step ``x_next = x_k - G_k f_k``. The
    initial inverse Jacobian is ``G_0 = -beta.I``, so the first step is exactly
    linear mixing ``x_next = (1 - beta).x_in + beta.x_out`` -- identical to
    ``AndersonMixer(depth=1)`` and to ``BroydenMixer(depth=1)``.

    Stored in limited-memory form: only the last ``depth`` ``(Δf_i, u_i)``
    pairs are kept and ``G_k v`` is evaluated as
    ``G_0 v + S_i u_i (Δf_i.v)`` -- never an explicit ``nxn`` matrix, so the
    cost is ``O(depth.n)`` per step. This is the standard density-mixing
    Broyden of plane-wave/Gaussian SCF codes: it mixes the *density* directly
    and never re-diagonalises with a guessed occupation.

    Parameters
    ----------
    depth:
        Maximum number of retained secant pairs (Broyden history "m").
    beta:
        Linear-mixing parameter in ``(0, 1]`` (the ``-beta.I`` seed Jacobian).
        Smaller is more conservative.
    """

    def __init__(self, depth: int = 8, beta: float = 0.5) -> None:
        if depth < 1:
            raise ValueError("BroydenMixer: depth must be >= 1")
        if not (0.0 < beta <= 1.0):
            raise ValueError("BroydenMixer: beta must be in (0, 1]")
        self.depth = int(depth)
        self.beta = float(beta)
        self._df: List[np.ndarray] = []   # Δf_i = f_i - f_{i-1}
        self._u: List[np.ndarray] = []    # u_i  = (Δx_i - G_{i-1}Δf_i) / (Δf_i.Δf_i)
        self._x_prev: np.ndarray | None = None
        self._f_prev: np.ndarray | None = None

    def reset(self) -> None:
        """Clear the secant history (e.g. when restarting or switching phase)."""
        self._df.clear()
        self._u.clear()
        self._x_prev = None
        self._f_prev = None

    @property
    def history_size(self) -> int:
        return len(self._df)

    def _apply_G(self, v: np.ndarray) -> np.ndarray:
        """Apply the current limited-memory inverse Jacobian: G v."""
        out = -self.beta * v
        for df_i, u_i in zip(self._df, self._u):
            out = out + u_i * float(df_i @ v)
        return out

    def update(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        """Return the next input given this iteration's input and SCF output."""
        xi = np.asarray(x_in, dtype=float).reshape(-1)
        xo = np.asarray(x_out, dtype=float).reshape(-1)
        if xi.shape != xo.shape:
            raise ValueError(
                f"BroydenMixer: x_in {xi.shape} and x_out {xo.shape} differ"
            )
        f = xo - xi

        if self._x_prev is None:
            # First step (or post-reset): G_0 = -beta I -> linear mixing.
            self._x_prev = xi.copy()
            self._f_prev = f.copy()
            return xi + self.beta * f

        dx = xi - self._x_prev
        df = f - self._f_prev
        denom = float(df @ df)
        if denom <= 0.0 or not np.isfinite(denom):
            # Degenerate secant (no change in residual): fall back to linear.
            self._x_prev = xi.copy()
            self._f_prev = f.copy()
            return xi + self.beta * f

        # u_i uses G_{k-1} (the history *before* this pair is appended).
        u = (dx - self._apply_G(df)) / denom
        self._df.append(df)
        self._u.append(u)
        if len(self._df) > self.depth:
            self._df.pop(0)
            self._u.pop(0)

        self._x_prev = xi.copy()
        self._f_prev = f.copy()
        # Quasi-Newton step with the updated G_k (includes the new pair).
        return xi - self._apply_G(f)


# ---------------------------------------------------------------------------
# Per-k density-matrix <-> flat-vector bridge
# ---------------------------------------------------------------------------
#
# The mixers operate on a flat real vector. Multi-k periodic SCF's natural
# fixed-point variable is the list of per-k Hermitian AO density matrices
# ``D(k)`` -- they carry the total electron count exactly (the band occupations
# are re-solved to ``N`` every cycle) and a real linear combination of
# Hermitian matrices stays Hermitian. A driver flattens ``(D_in_per_k,
# D_out_per_k)`` to feed ``mixer.update`` and splits the result back, then
# rebuilds the real-space ``LatticeMatrixSet`` from the mixed ``D(k)`` with
# :func:`vibeqc.periodic_k_density.real_space_density_from_per_k_density`.
#
# Note: do *not* instead mix the real-space ``LatticeMatrixSet`` and Bloch-sum
# it back to per-k -- the lattice-sum cutoff cell list does not invert cleanly
# to the k-mesh, so ``bloch_sum(D_real, k)`` is an aliased/scaled proxy (the
# DIIS error vector tolerates the scaling; an actual density does not, and the
# per-k electron count blows up, over-binding the SCF to a nonphysical energy).


def per_k_density_to_vector(D_per_k: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of per-k complex Hermitian density matrices to a real vector.

    Real and imaginary parts of each ``(nbf, nbf)`` block are concatenated.
    """
    parts: List[np.ndarray] = []
    for D in D_per_k:
        A = np.asarray(D, dtype=complex)
        parts.append(A.real.reshape(-1))
        parts.append(A.imag.reshape(-1))
    return np.concatenate(parts).astype(float)


def vector_to_per_k_density(
    vec: np.ndarray, template: List[np.ndarray]
) -> List[np.ndarray]:
    """Inverse of :func:`per_k_density_to_vector`.

    ``template`` supplies the per-k block shapes. Each reconstructed block is
    Hermitised (``1/2(D + Dᴴ)``) -- a real linear combination of Hermitian inputs
    is already Hermitian to round-off, this just cleans the residual.
    """
    vec = np.asarray(vec, dtype=float).reshape(-1)
    out: List[np.ndarray] = []
    offset = 0
    for T in template:
        shape = np.asarray(T).shape
        size = int(np.prod(shape))
        re = vec[offset : offset + size].reshape(shape)
        offset += size
        im = vec[offset : offset + size].reshape(shape)
        offset += size
        D = re + 1j * im
        out.append(0.5 * (D + D.conj().T))
    if offset != vec.size:
        raise ValueError(
            f"vector_to_per_k_density: vector length {vec.size} does not match "
            "the per-k block layout"
        )
    return out


# ---------------------------------------------------------------------------
# Kerker preconditioner (D4b) -- long-wavelength charge-sloshing damping
# ---------------------------------------------------------------------------


class KerkerPreconditioner:
    """Kerker preconditioner for AO density-matrix SCF mixing on metals.

    The Kerker filter ``K(G) = |G|^2 / (|G|^2 + G₀^2)`` damps the long-wavelength
    (small-|G|) charge-density modes that cause charge sloshing -- the limit
    cycle that defeats Fock-DIIS / linear mixing on a sharp Fermi surface
    (Kerker, *Phys. Rev. B* 23, 3082 (1981)).

    A Gaussian-AO SCF carries the density as a matrix, not a charge density on
    a grid, so this applies the *genuine* reciprocal-space filter to the
    density-matrix **residual** by round-tripping through a plane-wave grid:

        r_avg = S_k w_k (D_out(k) - D_in(k))     # cell-average (home-cell) residual
        Δr(r) = collocate(r_avg)                 # AO -> charge density on the grid
        s(r)  = Δr - K.Δr                        # the small-|G| part Kerker removes
        dD    = S⁻¹ (∫ chi_muchi_ν s dr) S⁻¹          # project back, S⊗S density-fit metric
        r_K(k) = r(k) - g.dD                     # damp the long-wavelength residual

    **Why this is correct regardless of the round-trip's approximations
    (CLAUDE.md Sec.7).** It preconditions the *residual*: at the SCF fixed point
    every ``r(k) = 0`` so ``r_avg = 0`` => ``dD = 0`` => ``r_K = r = 0``. A
    preconditioner that vanishes with the residual cannot move the fixed point,
    so the converged energy is *exactly* the unpreconditioned SCF solution; the
    collocation grid, the S⊗S metric and the cell-average (Γ) charge density
    only change the convergence *path*, never the answer. With ``G₀ -> 0`` (or
    ``strength = 0``) it reduces to plain density mixing.

    **Empirical scope (honest caveat).** Kerker targets the long-wavelength
    charge sloshing of *large* metallic cells / extended free-electron-like
    states. On the minimal-basis Gaussian test systems exercised so far (STO-3G
    H-chains, Be-fcc-scale not yet run) it is convergence-*neutral* -- it
    reaches the same energy in a comparable iteration count, neither helping nor
    hurting materially -- because those cells do not exhibit the small-|G|
    instability it damps. It is provided as a correct, Sec.7-safe, opt-in tool, not
    a demonstrated speed-up on small cells.

    Parameters
    ----------
    basis, system
        The AO basis and :class:`PeriodicSystem`; used to build the
        plane-wave collocation grid + AO table once (iteration-invariant).
    overlap0
        The home-cell (g=0) AO overlap ``S(0)`` -- the metric for the
        density-fit back-projection.
    k0
        Kerker screening wave-vector ``G₀ = 2pi/l`` in 1/bohr. Larger damps a
        wider band of long-wavelength modes. Default 1.5.
    strength
        Damping fraction ``g in (0, 1]`` of the long-wavelength residual to
        remove. Default 1.0 (full Kerker).
    cutoff_ha
        Plane-wave cutoff sizing the collocation grid. The preconditioner only
        needs the long-wavelength modes resolved, so a modest cutoff suffices.
        Default 120 Ha.
    """

    def __init__(
        self,
        basis,
        system,
        overlap0: np.ndarray,
        *,
        k0: float = 1.5,
        strength: float = 1.0,
        cutoff_ha: float = 120.0,
    ) -> None:
        if not (k0 > 0.0):
            raise ValueError("KerkerPreconditioner: k0 must be > 0")
        if not (0.0 < strength <= 1.0):
            raise ValueError("KerkerPreconditioner: strength must be in (0, 1]")
        # Lazy imports: the GPW grid machinery is heavy + experimental, and the
        # mixer module is otherwise dependency-light (numpy only).
        from .kerker import kerker_kernel
        from .periodic_gapw_grid import make_grid
        from .periodic_gapw_j import build_gpw_collocation_cache

        self.basis = basis
        self.k0 = float(k0)
        self.gamma = float(strength)
        self._grid = make_grid(
            np.asarray(system.lattice, dtype=float), cutoff_ha=float(cutoff_ha)
        )
        self._cache = build_gpw_collocation_cache(basis, self._grid)
        # |G|^2 on the collocation grid -> the Kerker kernel K(G) (G=0 -> 0).
        g2 = np.sum(np.asarray(self._cache.recip) ** 2, axis=-1)
        self._kernel = kerker_kernel(g2, k0=self.k0)
        # S(0)⁻¹ for the S⊗S density-fit metric of the grid->matrix back-map.
        # pinv tolerates a near-singular overlap (linear-dependent basis).
        self._S0_inv = np.linalg.pinv(np.asarray(overlap0, dtype=float), rcond=1e-10)

    def precondition(
        self,
        residual_per_k: List[np.ndarray],
        weights: np.ndarray,
    ) -> List[np.ndarray]:
        """Return the Kerker-preconditioned per-k density-matrix residual."""
        from .periodic_gapw_j import (
            collocate_density_on_grid,
            project_potential_to_ao,
        )

        w = np.asarray(weights, dtype=float).reshape(-1)
        # Cell-average (home-cell) residual -- real symmetric.
        r_avg = np.zeros_like(np.asarray(residual_per_k[0], dtype=complex))
        for wk, r in zip(w, residual_per_k):
            r_avg = r_avg + float(wk) * np.asarray(r, dtype=complex)
        r_avg = np.real(0.5 * (r_avg + r_avg.conj().T))

        rho = collocate_density_on_grid(self.basis, r_avg, self._grid, cache=self._cache)
        # Long-wavelength part Kerker removes: (1 - K).Δr.
        small_g = rho - np.real(np.fft.ifftn(np.fft.fftn(rho) * self._kernel))
        proj = np.asarray(
            project_potential_to_ao(
                self.basis, small_g, self._grid, cache=self._cache
            )
        )
        # S⊗S-metric density fit: dD ≈ S⁻¹ (∫chichi.small_g) S⁻¹.
        dD = self._S0_inv @ proj @ self._S0_inv
        dD = 0.5 * (dD + dD.T)
        return [np.asarray(r) - self.gamma * dD for r in residual_per_k]
