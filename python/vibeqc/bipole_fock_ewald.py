"""BIPOLE multipole-far-pair branch -- Phase 5: Ewald J-split F^2e build.

Implements a CRYSTAL-equivalent two-electron Fock build using a
**Coulomb Ewald split**: short-range J via direct-space erfc-
screened ERIs, long-range J via reciprocal-space analytic sum.

Exchange (2026-06-10 revision): the original design built K as a
direct-space full-Coulomb sum on the claim that "exchange has no
long-range divergence after bra-ket overlap screening". That claim
is wrong for the finite-k-mesh density: the inverse-Bloch-folded
P(g) is BvK-periodic (at Γ-only it is *constant* across cells), so
the full-Coulomb exchange series carries a S_g 1/|g| tail and is
formally divergent -- any finite value is a truncation artefact of
the cell cutoff. The corrected convention extends the Ewald split
to exchange (see :func:`compute_K_long_range_gamma` and
:func:`probe_charge_madelung`)::

    K  =  K^{SR}(erfc w, direct)  +  K^{LR}(erf w, reciprocal, K!=0)
          +  (ξ_M - pi/(Vw^2)) . S.D.S          [exxdiv='ewald']

where the S.D.S term removes the K=0 component implicitly contained
in the absolutely-convergent direct K^{SR} sum (-pi/(Vw^2)) and adds
the probe-charge Ewald (Madelung) finite-size correction (+ξ_M),
matching PySCF's ``exxdiv='ewald'`` convention and the
mesh-converged limit. Validated element-wise against PySCF GDF
``vk`` on MgO/STO-3G Γ: |ΔK|_max 8e-4, ΔE_K -0.3 mHa, w-invariant
(2026-06-10, examples/regression/bipole_parity/).

J-split formula (unchanged)::

    F^{2e}_muν  =  J^{SR}_muν(w)  +  J^{LR}_muν(w)  -  1/2 K_muν

where:

* ``J^{SR}_muν(w)`` is built via existing
  ``build_fock_2e_real_space(a=1, w=w, exchange_scale=0)``
  -- pure J with erfc(w.r) screening, exponentially convergent
  as a real-space lattice sum.

* ``J^{LR}_muν(w)`` is built analytically in reciprocal space::

    J^{LR}_muν  =  (4pi / V) . S_{K != 0}  exp(-K^2/4w^2) / K^2
                              . Re[ r̂(K) . FT_muν*(K) ]

  with r̂(K) = S_{ls} P_ls . FT_ls(K) the cell density's
  Fourier coefficient (uses the Y_lm-corrected
  :func:`compute_cell_density_fourier`).

* ``K^{full}_muν`` is built as a direct-space full-Coulomb exchange
  component. The current multi-k BIPOLE driver obtains this from the
  native ``build_jk_2e_real_space`` component builder; the older
  Γ-only convenience wrapper below still reconstructs K from the
  legacy combined Fock builders for backward compatibility. Exchange
  has exponentially decaying contributions from the bra-ket overlap
  product, so the direct sum converges intrinsically.

The single Ewald w parameter is shared with V_ne and E_nn (one
shared Ewald state; ``w = a = (1/2.8).V^{1/3}``, the CRYSTAL default).

This differs from the empirically-broken
``run_rhf_periodic_multi_k_ewald3d`` Ewald composition (per memory
``reference_ewald_composition_bug_2026-05-17``): that path used
FFT-Poisson for J_LR with incorrect G=0 pinning + gauge mismatch.
Here J_LR is computed by an **analytic reciprocal-space sum** with
the same w as V_ne / E_nn, restoring gauge consistency.

Module status: 2026-05-17 -- Phase 5 of the BIPOLE multipole-far-
pair branch. Validates with Phase 6 (parity vs CRYSTAL14).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ._aopair_ft import (
    ao_pair_fourier_transform,
    ao_pair_fourier_transform_at_cells,
    ao_pair_fourier_transform_bloch,
)
from ._vibeqc_core import (
    BasisSet,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    build_fock_2e_real_space,
)
from .bipole_ext_el_pole import (
    _libint_ylm_correction_per_ao,
    compute_cell_density_fourier,
    compute_cell_density_fourier_lattice,
    compute_reciprocal_lattice_vectors,
    crystal_default_ewald_alpha,
)

__all__ = [
    "build_fock_2e_ewald_j_split_gamma",
    "compute_J_long_range_gamma",
    "compute_J_long_range_at_k",
    "compute_J_long_range_real_space_blocks",
    "compute_K_long_range_gamma",
    "compute_K_long_range_at_k",
    "build_k_exchange_long_range_cache",
    "compute_rho_hat_from_k_density",
    "compute_shifted_reciprocal_lattice_vectors",
    "probe_charge_madelung",
    "probe_charge_madelung_supercell",
    "exchange_q0_gauge_constant",
    "JLongRangeCache",
    "KExchangeLongRangeCache",
    "IncrementalJK",
]


@dataclass
class IncrementalJK:
    """Differential (incremental) builder for the corrected-gauge J_SR + K_SR.

    The two-electron Fock is linear in the density, so for any reference
    R: ``JK(D) = JK(R) + JK(D - R)``. Building from the iter-to-iter
    increment ``ΔD = D_n - D_{n-1}`` and accumulating exploits the C++
    builder's Schwarz x **density-envelope** screening
    (``cpp/src/periodic_fock.cpp``): as the SCF converges ``ΔD -> 0`` the
    envelope shrinks and progressively more ERI quartets fall below the
    screening threshold, so each increment build is cheaper than the
    full one (the single ``build_jk_2e_real_space`` traversal is ~99%
    of the corrected-gauge Fock-build wall -- 2026-06-14 profile). A full
    rebuild every ``reset_every`` iterations re-syncs the accumulator to
    the exact ``JK(D_n)`` and bounds accumulated screening error.

    Applies ONLY to the dominant direct erfc traversal (J_SR + K_SR);
    the reciprocal J^LR / K^LR pieces (<1% of the wall) are rebuilt
    fully each iter by the driver. The ΔD chain is correct only when the
    SAME density object the caller uses is passed every iteration, in
    order -- callers bypass it (full build) for ODA's extra naive build
    and the post-convergence rebuild, and :meth:`seed` re-syncs it to such
    a full build when the terminal check finds the chain has drifted
    (#116).
    """

    reset_every: int = 12
    _D_prev: Optional[dict] = field(default=None, repr=False)
    _J_acc: Optional[list] = field(default=None, repr=False)
    _K_acc: Optional[list] = field(default=None, repr=False)
    _cells: object = field(default=None, repr=False)
    _nbf: int = field(default=0, repr=False)
    _since_reset: int = field(default=0, repr=False)

    def build(self, density, build_jk):
        """Return accumulated ``(J_SR_lat, K_SR_lat)`` for ``density``.

        ``build_jk(density_lattice)`` must return an object with a ``.J``
        LatticeMatrixSet and a ``.K`` LatticeMatrixSet *or* ``.K = None``
        (the pure-DFT J-only build) on the operator cell list -- the
        ``build_jk_2e_real_space`` contract (or a J-only shim). When
        ``.K`` is ``None`` the returned ``K_SR_lat`` is ``None`` too.
        Returned lattices are fresh copies -- safe for the caller to
        mutate without corrupting the accumulator.
        """
        from ._vibeqc_core import make_lattice_matrix_set
        from .pbc_bipole_common import _cell_key

        do_full = self._D_prev is None or self._since_reset >= self.reset_every
        if do_full:
            jk = build_jk(density)
            has_k = getattr(jk, "K", None) is not None
            self._cells = list(jk.J.cells)
            self._nbf = int(jk.J.nbf)
            self._J_acc = [np.asarray(b, dtype=float).copy() for b in jk.J.blocks]
            self._K_acc = (
                [np.asarray(b, dtype=float).copy() for b in jk.K.blocks]
                if has_k
                else None
            )
            self._since_reset = 0
        else:
            delta = self._delta(density, make_lattice_matrix_set, _cell_key)
            jk = build_jk(delta)
            for c in range(len(self._J_acc)):
                self._J_acc[c] += np.asarray(jk.J.blocks[c], dtype=float)
                if self._K_acc is not None:
                    self._K_acc[c] += np.asarray(jk.K.blocks[c], dtype=float)
            self._since_reset += 1
        self._D_prev = {
            _cell_key(cell): np.asarray(b, dtype=float).copy()
            for cell, b in zip(density.cells, density.blocks)
        }
        J_lat = make_lattice_matrix_set(
            self._nbf, self._cells, [b.copy() for b in self._J_acc]
        )
        K_lat = (
            make_lattice_matrix_set(
                self._nbf, self._cells, [b.copy() for b in self._K_acc]
            )
            if self._K_acc is not None
            else None
        )
        return J_lat, K_lat

    def seed(self, density, J_lat, K_lat) -> None:
        """Re-sync the accumulator to an exact full build of ``density``.

        ``(J_lat, K_lat)`` must be the caller's own non-incremental
        ``build_jk(density)`` result (``K_lat`` is ``None`` for the J-only
        build). Afterwards the accumulator holds exactly what a full
        ``build`` of ``density`` would have stored, with the reset counter
        at zero, so the next :meth:`build` continues the ΔD chain from the
        exact operator instead of from the screening error the chain had
        accumulated. The BIPOLE drivers seed from their exact terminal
        confirmation rebuild when that rebuild disagrees with the
        incremental fixed point (GitLab #116): the loop then converges on
        the operator it is judged by, and the iteration that follows costs
        a ``ΔD = 0`` build. Blocks are copied, so the caller may keep
        mutating its own lattices.
        """
        from .pbc_bipole_common import _cell_key

        self._cells = list(J_lat.cells)
        self._nbf = int(J_lat.nbf)
        self._J_acc = [np.asarray(b, dtype=float).copy() for b in J_lat.blocks]
        self._K_acc = (
            [np.asarray(b, dtype=float).copy() for b in K_lat.blocks]
            if K_lat is not None
            else None
        )
        self._since_reset = 0
        self._D_prev = {
            _cell_key(cell): np.asarray(b, dtype=float).copy()
            for cell, b in zip(density.cells, density.blocks)
        }

    def _delta(self, density, make_lattice_matrix_set, _cell_key):
        blocks = []
        for cell, b in zip(density.cells, density.blocks):
            cur = np.asarray(b, dtype=float)
            prev = self._D_prev.get(_cell_key(cell))
            blocks.append(cur - prev if prev is not None else cur.copy())
        return make_lattice_matrix_set(
            int(density.nbf), list(density.cells), blocks
        )


def compute_J_long_range_gamma(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    omega: float,
    *,
    precision: float = 1e-8,
    pad_factor: float = 1.5,
) -> np.ndarray:
    """Long-range J Fock-matrix contribution at Γ via reciprocal-space sum.

    Γ-only thin wrapper around :func:`compute_J_long_range_at_k` with
    ``k_cart = (0,0,0)``. Returns the real part -- at Γ with a real-
    Hermitian density the imaginary part vanishes analytically (and
    is enforced to <= 1e-12 by the per-K ±K pairing).

    Kept as a public entry point for the Γ-only Phase 5 build
    (:func:`build_fock_2e_ewald_j_split_gamma`) and for backward
    compatibility with parity scripts that build J^LR(Γ) standalone.

    Parameters
    ----------
    P_real : LatticeMatrixSet
        Real-space density.
    basis : BasisSet
        AO basis.
    system : PeriodicSystem
        3D periodic system.
    omega : float > 0
        Ewald split parameter (bohr⁻¹). Must match V_ne / E_nn a.
    precision : float
        K-sum truncation precision.
    pad_factor : float
        K_max safety factor.

    Returns
    -------
    J_LR : ndarray of shape (nbf, nbf), real
        Long-range J Fock-matrix contribution at Γ.
    """
    F_LR_complex = compute_J_long_range_at_k(
        P_real,
        basis,
        system,
        omega,
        np.zeros(3, dtype=float),
        precision=precision,
        pad_factor=pad_factor,
    )
    # At Γ with real-Hermitian density, imag part is analytically zero;
    # symmetrise to kill numerical noise.
    J_LR = np.real(F_LR_complex)
    J_LR = 0.5 * (J_LR + J_LR.T)
    return J_LR


@dataclass
class JLongRangeCache:
    """Per-SCF-iter cache for the multi-k J^LR build.

    The shifted-ν AO-pair FT and the K-vector / kernel arrays depend
    only on ``(basis, system, omega, precision)`` -- invariant across
    SCF iters and across k-points within a single iter. The cache
    holds those invariant pieces so the per-k F_J^LR build only does
    the small einsum that contracts ``r̂(K)`` against the Bloch-
    summed bra-pair FT at the requested k.

    Attributes
    ----------
    K_vectors : (n_K, 3)
        Reciprocal-lattice vectors with ``0 < |K| <= K_max``.
    kernel : (n_K,)
        ``(4pi/V) . exp(-K^2/4w^2) / K^2`` Ewald-screened kernel.
    ft_per_cell : (n_g, nbf, nbf, n_K) complex
        Shifted-ν AO-pair FT for each lattice cell ``R_g``, with the
        libint Y_lm correction baked in. Indexed by the same cell
        order as ``LatticeMatrixSet.cells`` from the F_J^SR build.
    ft_bloch_cache : dict
        Lazily populated cache of ``S_g exp(-ik.R_g) FT_g`` tensors.
        These tensors are density-independent but used every time the
        multi-k SCF rebuilds ``r̂(K)`` from the current orbitals.
    cells_r_cart : (n_g, 3)
        Cartesian lattice vectors corresponding to ``ft_per_cell``'s
        first axis.
    omega : float
    """

    K_vectors: np.ndarray
    kernel: np.ndarray
    ft_per_cell: np.ndarray
    cells_r_cart: np.ndarray
    omega: float
    pair_cutoff_bohr: Optional[float] = None
    ft_bloch_cache: dict[bytes, np.ndarray] = field(
        default_factory=dict,
        repr=False,
    )


def _k_cache_key(k_cart: np.ndarray) -> bytes:
    """Stable binary key for a Cartesian k-vector."""

    return np.ascontiguousarray(
        np.asarray(k_cart, dtype=np.float64).reshape(3),
    ).tobytes()


def _bloch_pair_ft_from_cache(
    cache: JLongRangeCache,
    k_cart: np.ndarray,
) -> np.ndarray:
    """Return ``S_g exp(-ik.R_g) FT_g`` for ``k_cart`` using ``cache``.

    The shifted-cell AO-pair FT stack is expensive to build and the
    per-k Bloch contraction is also density-independent. Multi-k SCF
    revisits the same k-points every iteration, so materialising this
    contraction once per k-point removes a repeated ``n_cells`` tensor
    sum from the inner loop without changing the J^LR formula.
    """

    k_arr = np.asarray(k_cart, dtype=float).reshape(3)
    key = _k_cache_key(k_arr)
    ft_bloch = cache.ft_bloch_cache.get(key)
    if ft_bloch is None:
        phases_k = np.exp(-1j * (cache.cells_r_cart @ k_arr))
        ft_bloch = np.einsum(
            "g,gmnk->mnk",
            phases_k,
            cache.ft_per_cell,
            optimize=True,
        )
        cache.ft_bloch_cache[key] = ft_bloch
    return ft_bloch


def _build_j_long_range_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    cells_r_cart: np.ndarray,
    omega: float,
    precision: float,
    pad_factor: float = 1.5,
    *,
    K_max: Optional[float] = None,
    lattice_opts: Optional[LatticeSumOptions] = None,
) -> JLongRangeCache:
    """Construct a :class:`JLongRangeCache` for the given lattice setup.

    Computes K-mesh, kernel, and the shifted-ν AO-pair FT stack once
    so subsequent per-iter / per-k builds reuse them.

    Parameters
    ----------
    K_max : float, optional
        Explicit reciprocal-space cutoff in bohr⁻¹.  When provided,
        bypasses the precision-driven auto K_max (``2.w.√(-ln prec).pad``).
        Pass :func:`vibeqc.bipole_ext_el_pole.crystal_ewald_reciprocal_cutoff`
        to match CRYSTAL's reciprocal-space Ewald envelope (for output parity).
    """
    if system.dim != 3:
        raise ValueError(f"requires dim=3; got dim={system.dim}")
    if omega <= 0.0:
        raise ValueError(f"omega must be > 0; got {omega}")
    a = np.asarray(system.lattice, dtype=float)
    V = float(abs(np.linalg.det(a)))

    if K_max is not None and K_max > 0.0:
        _K_max = float(K_max)
    else:
        _K_max = (
            2.0
            * float(omega)
            * float(math.sqrt(-math.log(precision)))
            * float(pad_factor)
        )
    K_vectors = compute_reciprocal_lattice_vectors(system, _K_max)
    if K_vectors.shape[0] == 0:
        raise RuntimeError("_build_j_long_range_cache: K-list empty; check inputs")
    K2 = (K_vectors**2).sum(axis=1)
    kernel = (4.0 * math.pi / V) * np.exp(-K2 / (4.0 * omega * omega)) / K2

    R_g_arr = np.ascontiguousarray(cells_r_cart, dtype=float)
    ft_per_cell = ao_pair_fourier_transform_at_cells(
        basis,
        K_vectors,
        R_g_arr,
    )
    correction = _libint_ylm_correction_per_ao(basis)
    ft_per_cell = ft_per_cell * (
        correction[None, :, None, None] * correction[None, None, :, None]
    )

    pair_cutoff = (
        float(lattice_opts.cutoff_bohr)
        if lattice_opts is not None and lattice_opts.pair_complete_1e else None
    )
    if pair_cutoff is not None:
        from .lattice_screening import ao_pair_support_mask

        for cell, shift in enumerate(R_g_arr):
            ft_per_cell[cell] *= ao_pair_support_mask(basis, shift, pair_cutoff)[:, :, None]

    return JLongRangeCache(
        K_vectors=K_vectors,
        kernel=kernel,
        ft_per_cell=ft_per_cell,
        cells_r_cart=R_g_arr,
        omega=float(omega),
        pair_cutoff_bohr=pair_cutoff,
    )


def compute_rho_hat_from_k_density(
    D_k_list: Sequence[np.ndarray],
    k_points: Sequence[np.ndarray],
    weights: Sequence[float],
    cache: JLongRangeCache,
) -> np.ndarray:
    """Compute r̂_total(K) via the k-space density representation.

    The proper Bloch-formalism formula::

        r̂_total(K) = S_k w_k . S_{ls} D(k)_ls . FT^{(-k)}_ls(K)

    where ``FT^{(-k)}_ls(K) = S_g exp(-ik.R_g) . FT_ls(K; R_g)``
    is the inverse-Bloch AO-pair FT matching
    ``D(g) = S_k w_k exp(-ik.R_g)D(k)``.

    Equivalent to summing the real-space density blocks via
    ``S_g D(g).FT(K; R_g)`` ONLY when the lattice-cell count equals
    the k-mesh size and the inverse Bloch transform from D(k) to D(g)
    is exact. With cutoff-truncated lattices and Re[] taken in
    :func:`real_space_density_from_kpoints`, the two routes differ --
    only the k-space route preserves the proper k-mesh normalisation
    needed for SCF self-consistency at multi-k.

    Parameters
    ----------
    D_k_list : sequence of (nbf, nbf) complex
        Per-k density matrices ``D(k) = 2.C_occ.C_occ+(k)`` (RHF
        closed-shell convention).
    k_points : sequence of (3,) arrays
    weights : sequence of floats summing to 1
    cache : JLongRangeCache
        Cache holding ``ft_per_cell`` and ``cells_r_cart`` for the
        lattice over which to Bloch-sum.

    Returns
    -------
    rho_hat : (n_K,) complex
    """
    n_K = cache.K_vectors.shape[0]
    rho_hat = np.zeros(n_K, dtype=np.complex128)
    for k_idx, k in enumerate(k_points):
        w = float(weights[k_idx])
        ft_bloch = _bloch_pair_ft_from_cache(cache, k)
        D_k = np.asarray(D_k_list[k_idx])
        rho_hat = rho_hat + w * np.einsum("mn,mnk->k", D_k, ft_bloch)
    return rho_hat


def compute_J_long_range_at_k(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    omega: float,
    k_cart: np.ndarray,
    *,
    precision: float = 1e-8,
    pad_factor: float = 1.5,
    cache: Optional[JLongRangeCache] = None,
    rho_hat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Long-range J Fock-matrix contribution at crystal momentum k.

    Formula::

        F^{LR}_muν(k) = (4pi / V) . S_{K != 0}
            exp(-K^2/4w^2) / K^2 . r̂(K) . [FT^{(-k)}_muν(K)]*

    where:

    * ``r̂(K) = S_{ls, g} D_ls(g) . FT_ls(K; R_g)`` -- total cell-
      density Fourier coefficient summed over the lattice (correct
      shifted-ν FT, not the bra-pair-at-home approximation that
      the legacy Γ-only path used as an empirical shortcut).
    * ``FT^{(-k)}_muν(K) = S_g exp(-i k.R_g) . FT_muν(K; R_g)``. The
      minus sign appears before the final complex conjugation, so the
      resulting Fock matrix obeys vibe-qc's standard operator Bloch sum
      ``F(k) = S_g exp(+ik.R_g) . F(g)``; see
      :func:`vibeqc.pbc_bipole._bloch_sum_blocks`.

    This is the gauge-correct multi-k generalisation of
    :func:`compute_J_long_range_gamma`. At k = 0 with D non-zero only
    at g = 0 it reduces to the same closed form as the Γ-only path
    (verified by the parametrised test in
    ``tests/test_pbc_bipole_multik_ewald_split.py``).

    Parameters
    ----------
    P_real : LatticeMatrixSet
        Real-space density. The cell list MUST match the one used
        when ``cache.ft_per_cell`` was built (typically the cells
        from ``compute_overlap_lattice`` at the same cutoff).
    basis : BasisSet
        AO basis.
    system : PeriodicSystem
        3D periodic system (raises for dim != 3).
    omega : float > 0
        Ewald split parameter. MUST match V_ne / E_nn a.
    k_cart : (3,)
        Crystal momentum in Cartesian inverse bohr.
    precision, pad_factor
        K-mesh truncation parameters (used only when ``cache`` is
        ``None``; otherwise ``cache.K_vectors`` is taken as given).
    cache : JLongRangeCache, optional
        Pre-built cache from :func:`_build_j_long_range_cache`.
        Strongly recommended for SCF inner loops -- building the cache
        does ``n_cells`` shifted-ν FT computations, which is the
        bulk of the J^LR cost.

    Returns
    -------
    F_LR_k : (nbf, nbf) complex
        Per-k J^LR Fock-matrix contribution (Hermitian).
    """
    if system.dim != 3:
        raise ValueError(f"requires dim=3; got dim={system.dim}")
    if omega <= 0.0:
        raise ValueError(f"omega must be > 0; got {omega}")
    if P_real.nbf != basis.nbasis:
        raise ValueError(f"P_real.nbf={P_real.nbf} != basis.nbasis={basis.nbasis}")

    cells = list(P_real.cells)
    R_g_arr = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in cells],
        dtype=float,
    )
    if cache is None:
        cache = _build_j_long_range_cache(
            basis,
            system,
            R_g_arr,
            omega,
            precision,
            pad_factor,
        )
    else:
        if cache.ft_per_cell.shape[0] != len(cells):
            raise ValueError(
                f"cache cell count {cache.ft_per_cell.shape[0]} does not "
                f"match P_real cells ({len(cells)}); rebuild the cache"
            )

    K_vectors = cache.K_vectors
    n_K = K_vectors.shape[0]

    if rho_hat is None:
        # Fallback: real-space form r̂(K) = S_g D(g)_muν . FT_muν(K; R_g).
        # Used when no pre-computed k-space r̂ is supplied (e.g. Γ-only
        # short-circuit where D_used(g!=0) = 0 is enforced and the two
        # forms coincide). For multi-k SCF the caller should pass a
        # k-summed r̂ via :func:`compute_rho_hat_from_k_density` --
        # the real-space form here over-/under-counts when n_cells !=
        # n_k due to the Re[] in :func:`real_space_density_from_kpoints`.
        rho_hat = np.zeros(n_K, dtype=np.complex128)
        for c in range(len(cells)):
            P_block = np.asarray(P_real.blocks[c], dtype=float)
            if P_block.size == 0:
                continue
            rho_hat = rho_hat + np.einsum(
                "mn,mnk->k",
                P_block,
                cache.ft_per_cell[c],
            )

    F_LR_k = _compute_J_long_range_from_rho_linear(cache, k_cart, rho_hat)
    # Hermitise (numerical-noise cleanup; analytic Hermiticity holds
    # under the ±K pairing of the K-mesh + real-Hermitian density).
    F_LR_k = 0.5 * (F_LR_k + F_LR_k.conj().T)
    return F_LR_k


def _compute_J_long_range_from_rho_linear(
    cache: JLongRangeCache, k_cart: np.ndarray, rho_hat: np.ndarray,
) -> np.ndarray:
    """Unprojected HF contraction for finite-source operator comparisons.

    This is complex-linear in the supplied density Fourier coefficients.
    It adds no zero-mode, probe charge, spin factor or Hermitization. The
    caller owns the cache's finite support; this is not a source certificate.
    """
    ft_bloch_k = _bloch_pair_ft_from_cache(cache, k_cart)
    weighted = cache.kernel * rho_hat
    return np.einsum("k,mnk->mn", weighted, ft_bloch_k.conj())


def compute_J_long_range_real_space_blocks(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    omega: float,
    *,
    precision: float = 1e-8,
    pad_factor: float = 1.5,
    cache: Optional[JLongRangeCache] = None,
    rho_hat: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Long-range J operator blocks ``J^LR(g)`` on ``P_real``'s cell list.

    ``compute_J_long_range_at_k`` returns the Bloch-folded matrix needed
    for diagonalisation,

    ``J^LR(k) = S_g exp(+ik.R_g) J^LR(g)``.

    The lattice-sum energy path, however, contracts in real space:
    ``0.5.S_g tr[D(g)J^LR(g)]``.  This helper exposes those per-cell
    blocks so the SCF driver can use the same operator for both purposes
    without accidentally evaluating the energy from a Γ-folded matrix.

    When BOTH ``cache`` and ``rho_hat`` are supplied, ``P_real`` is not
    consulted at all: the blocks are emitted on the cache's cell list
    (the operator support). This is the Ewald-exchange-split Γ path,
    where the SCF density set is intentionally WIDER than the operator
    cell list (2x the traversal cutoff, so the C++ JK builder can
    resolve every P(b-a) difference) and r̂(K) is computed exactly from
    the Γ density matrix -- requiring the per-cell density here would
    force the FT cache onto the wide list for no numerical gain.
    """
    if system.dim != 3:
        raise ValueError(f"requires dim=3; got dim={system.dim}")
    if omega <= 0.0:
        raise ValueError(f"omega must be > 0; got {omega}")

    if cache is not None and rho_hat is not None:
        n_cells_out = cache.ft_per_cell.shape[0]
    else:
        if P_real.nbf != basis.nbasis:
            raise ValueError(
                f"P_real.nbf={P_real.nbf} != basis.nbasis={basis.nbasis}"
            )
        cells = list(P_real.cells)
        if cache is None:
            R_g_arr = np.array(
                [np.asarray(c.r_cart, dtype=float) for c in cells],
                dtype=float,
            )
            cache = _build_j_long_range_cache(
                basis,
                system,
                R_g_arr,
                omega,
                precision,
                pad_factor,
            )
        elif cache.ft_per_cell.shape[0] != len(cells):
            raise ValueError(
                f"cache cell count {cache.ft_per_cell.shape[0]} does not "
                f"match P_real cells ({len(cells)}); rebuild the cache "
                f"(or pass rho_hat to emit blocks on the cache's list)"
            )
        n_cells_out = len(cells)

    n_K = cache.K_vectors.shape[0]
    if rho_hat is None:
        rho_hat = np.zeros(n_K, dtype=np.complex128)
        for c in range(n_cells_out):
            P_block = np.asarray(P_real.blocks[c], dtype=float)
            if P_block.size == 0:
                continue
            rho_hat = rho_hat + np.einsum(
                "mn,mnk->k",
                P_block,
                cache.ft_per_cell[c],
            )

    weighted = cache.kernel * rho_hat
    blocks: list[np.ndarray] = []
    for c in range(n_cells_out):
        block = np.einsum(
            "k,mnk->mn",
            weighted,
            cache.ft_per_cell[c].conj(),
        )
        # The ±K reciprocal mesh makes each real-space block real up to
        # roundoff.  Keep the storage real so it behaves like the native
        # LatticeMatrixSet blocks used by the rest of the BIPOLE driver.
        blocks.append(np.real(block))
    return blocks


@dataclass
class EwaldJSplitFock:
    """Result of :func:`build_fock_2e_ewald_j_split_gamma`.

    Stores the F^2e (combined) at Γ along with the component pieces
    so the caller can inspect each contribution separately for
    debugging / per-component parity-checking.
    """

    F2e: np.ndarray  # (nbf, nbf) -- combined F^2e
    J_SR: np.ndarray  # (nbf, nbf) -- direct-space erfc-screened J
    J_LR: np.ndarray  # (nbf, nbf) -- reciprocal-space analytic J
    K_full: np.ndarray  # (nbf, nbf) -- direct-space K (full Coulomb)
    omega_bohr_inv: float


def build_fock_2e_ewald_j_split_gamma(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    omega: Optional[float] = None,
    precision: float = 1e-8,
    S_real: Optional[LatticeMatrixSet] = None,
) -> EwaldJSplitFock:
    """CRYSTAL-equivalent F^2e Fock matrix at Γ via Ewald J-split.

    Builds ``F^{2e} = J^SR + J^LR + J^bg - 1/2 K^full`` using:

      * ``J^SR`` from direct-space erfc-screened ERIs
        (build_fock_2e_real_space with w > 0, exchange_scale=0).
      * ``J^LR`` from reciprocal-space analytic sum
        (compute_J_long_range_gamma).
      * ``J^bg`` the jellium neutralising-background potential
        ``-pi N_e/(w^2V) . S(Γ)`` -- matches the BIPOLE driver's
        _build_fock_for_density.  Only applied when ``S_real``
        is provided (otherwise omitted for backward compatibility).
      * ``K^full`` extracted as
        ``2.(J^full_direct - (J^full_direct - 1/2K^full))``
        from two direct-space builds.

    For closed-shell RHF the K piece is multiplied by 0.5 internally.

    Parameters
    ----------
    P_real : LatticeMatrixSet
        Real-space density (typically k-averaged for multi-k SCF).
    basis : BasisSet
        AO basis.
    system : PeriodicSystem
        3D periodic system.
    lat_opts : LatticeSumOptions
        Real-space cutoff options for the direct-space builds.
        ``coulomb_method`` is ignored (driver-internal).
    omega : float, optional
        Ewald split parameter (bohr⁻¹). If None, uses CRYSTAL's
        ``crystal_default_ewald_alpha(V_cell)`` bounded below by
        ``sqrt(-ln precision) / lat_opts.cutoff_bohr`` (GitLab #674): the
        direct-space erfc build here walks the unpadded ``lat_opts`` cell
        list, so the split parameter must have decayed by that cutoff.
    precision : float
        K-sum truncation precision for the long-range piece.
    S_real : LatticeMatrixSet, optional
        Real-space overlap matrix blocks.  When provided, the jellium
        neutralising-background potential ``-pi N_e/(w^2V).S(Γ)`` is
        added to J^LR so the total ``F^2e`` matches the BIPOLE
        driver's convention.  When ``None``, the background is omitted
        (backward compatible; use for diagnostic component inspection).

    Returns
    -------
    EwaldJSplitFock
        F^2e + per-component matrices.
    """
    if system.dim != 3:
        raise ValueError(f"requires dim=3; got dim={system.dim}")
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    if omega is not None:
        omega_val = float(omega)
    else:
        from .pbc_bipole_common import default_ewald_alpha

        omega_val = default_ewald_alpha(
            V,
            real_cutoff_bohr=float(lat_opts.cutoff_bohr),
            tolerance=float(precision),
        )

    # --- Direct-space short-range J via erfc kernel.
    F_J_SR_lat = build_fock_2e_real_space(
        basis,
        system,
        lat_opts,
        P_real,
        0.0,  # exchange_scale = 0 -> pure J
        float(omega_val),  # omega > 0 -> erfc kernel
    )
    # Extract g=0 block.
    g0_idx = next(
        i
        for i, c in enumerate(F_J_SR_lat.cells)
        if (np.asarray(c.index) == np.array([0, 0, 0])).all()
    )
    J_SR_g0 = np.asarray(F_J_SR_lat.blocks[g0_idx], dtype=float)
    # Bloch sum to Γ: S_g F_real(g) . exp(i.0.R_g) = S_g F_real(g)
    J_SR = sum(
        np.asarray(F_J_SR_lat.blocks[i], dtype=float)
        for i in range(len(F_J_SR_lat.cells))
    )

    # --- Reciprocal-space long-range J.
    J_LR = compute_J_long_range_gamma(
        P_real,
        basis,
        system,
        omega_val,
        precision=precision,
    )

    # --- Jellium neutralising-background potential (G=0 term).
    # The BIPOLE driver's _build_fock_for_density adds this to J^LR
    # blocks per-cell.  For a SAD density (D(g!=0)=0), only the g=0
    # block contributes to the energy.  We add the background to the
    # Γ-folded J^LR using the g=0 overlap to match the per-cell
    # formulation exactly.
    if S_real is not None:
        g0_idx = next(
            i
            for i, c in enumerate(S_real.cells)
            if (np.asarray(c.index) == np.array([0, 0, 0])).all()
        )
        S_g0 = np.asarray(S_real.blocks[g0_idx], dtype=float)
        j_bg_potential = (
            -math.pi
            * float(system.n_electrons())
            / (float(omega_val) * float(omega_val) * V)
        )
        J_LR = J_LR + j_bg_potential * S_g0

    # --- Direct-space K extraction.
    # J_full_direct (no Ewald, no K)
    F_J_full_lat = build_fock_2e_real_space(
        basis,
        system,
        lat_opts,
        P_real,
        0.0,
        0.0,
    )
    J_full = sum(
        np.asarray(F_J_full_lat.blocks[i], dtype=float)
        for i in range(len(F_J_full_lat.cells))
    )
    # (J - 1/2K)_full_direct
    F_JmK_lat = build_fock_2e_real_space(
        basis,
        system,
        lat_opts,
        P_real,
        1.0,
        0.0,
    )
    J_minus_halfK_full = sum(
        np.asarray(F_JmK_lat.blocks[i], dtype=float)
        for i in range(len(F_JmK_lat.cells))
    )
    # K_full = 2.(J_full - (J - 1/2K)_full)
    K_full = 2.0 * (J_full - J_minus_halfK_full)

    # --- Combine.
    F2e = J_SR + J_LR - 0.5 * K_full
    F2e = 0.5 * (F2e + F2e.T)

    return EwaldJSplitFock(
        F2e=F2e,
        J_SR=J_SR,
        J_LR=J_LR,
        K_full=K_full,
        omega_bohr_inv=float(omega_val),
    )


def compute_K_long_range_gamma(
    cache: JLongRangeCache,
    D_gamma: np.ndarray,
) -> np.ndarray:
    """Γ-point long-range (erf) exchange via the reciprocal Ewald sum.

    The exchange analogue of the J^LR build, q = Γ - Γ = 0 channel::

        K^{LR}_muν = S_{K!=0} kernel(K) . [r̃(K)+ D r̃(K)]-sandwich
                  = S_{K!=0} (4pi/V).e^{-K^2/4w^2}/K^2
                        . S_{ls} r̃*_{mul}(K) D_ls r̃_{νs}(K)

    with ``r̃_{mul}(K) = S_g FT[(mu_0, l_g)](K)`` the Γ Bloch-summed
    shifted-ν AO-pair FT -- the same tables (and the same kernel,
    which excludes K = 0) the J^LR sum uses; exchange sandwiches the
    density between the two pair FTs instead of tracing against it.

    The K = 0 term of the *full* exchange kernel is handled
    separately by the caller: the absolutely-convergent direct-space
    K^SR(erfc w) sum implicitly contains its own K = 0 component
    (pi/(Vw^2)).S.D.S, which the caller removes and replaces with the
    exxdiv convention of choice (see :func:`probe_charge_madelung`).

    Parameters
    ----------
    cache : JLongRangeCache
    D_gamma : (nbf, nbf)
        Γ-point density matrix (the BvK representative; equals the
        home-cell block of the real-space density set at n_k = 1).

    Returns
    -------
    K_LR : (nbf, nbf) float
    """
    ft_bloch = _bloch_pair_ft_from_cache(cache, np.zeros(3))
    K_LR = np.einsum(
        "k,mlk,ls,nsk->mn",
        cache.kernel,
        ft_bloch.conj(),
        np.asarray(D_gamma, dtype=complex),
        ft_bloch,
        optimize=True,
    )
    K_LR = 0.5 * (K_LR + K_LR.conj().T)
    return np.real(K_LR)


def compute_shifted_reciprocal_lattice_vectors(
    system: PeriodicSystem,
    q_cart: np.ndarray,
    K_max_bohr_inv: float,
) -> np.ndarray:
    """Enumerate shifted reciprocal vectors ``{q + G : 0 < |q+G| <= K_max}``.

    ``G`` runs over the full reciprocal lattice *including* ``G = 0``;
    the zero VECTOR ``q + G = 0`` (possible only when ``q`` is itself a
    reciprocal-lattice vector, i.e. the diagonal ``k′ = k`` exchange
    channel) is excluded -- its K = 0 physics is handled by the
    exxdiv/Madelung correction, exactly as in the unshifted
    :func:`compute_reciprocal_lattice_vectors` sum.

    The output is invariant (as a set) under ``q -> q + G₀`` shifts; the
    enumeration order is deterministic for a given ``q`` input, so
    callers that cache per-``q`` tensors must canonicalise ``q`` first
    (see :class:`KExchangeLongRangeCache`).
    """
    if system.dim != 3:
        raise ValueError(
            "compute_shifted_reciprocal_lattice_vectors requires dim=3; "
            f"got dim={system.dim}"
        )
    if K_max_bohr_inv <= 0.0:
        raise ValueError(f"K_max must be positive; got {K_max_bohr_inv}")
    q = np.asarray(q_cart, dtype=float).reshape(3)
    a = np.asarray(system.lattice, dtype=float)
    b = 2.0 * math.pi * np.linalg.inv(a).T
    # |q+G| <= K_max implies |G| <= K_max+|q|.  Since a_i.G=2pi.n_i,
    # these component-wise bounds enumerate every possible reciprocal
    # coefficient on a skew lattice.
    g_radius = K_max_bohr_inv + float(np.linalg.norm(q))
    a_norms = np.linalg.norm(a, axis=0)
    n_max = np.ceil(g_radius * a_norms / (2.0 * math.pi)).astype(int)
    grids = [np.arange(-n, n + 1) for n in n_max]
    n1, n2, n3 = np.meshgrid(*grids, indexing="ij")
    idx = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=-1).astype(float)
    K = q[None, :] + idx @ b.T
    K2 = (K**2).sum(axis=1)
    keep = (K2 > 1e-12) & (K2 <= K_max_bohr_inv * K_max_bohr_inv)
    return K[keep]


@dataclass
class _QExchangeChannel:
    """Per-``q`` data for the multi-k LR-exchange build.

    ``K_vectors`` / ``kernel`` are the ``q``-shifted analogues of the
    :class:`JLongRangeCache` arrays; ``b_per_k`` lazily caches the
    Bloch-folded pair-FT tensor ``B^{(k′)}(q+G)`` per ``k′``.
    """

    K_vectors: np.ndarray
    kernel: np.ndarray
    b_per_k: dict[bytes, np.ndarray] = field(default_factory=dict, repr=False)


@dataclass
class KExchangeLongRangeCache:
    """Density-independent tables for the multi-k LR (erf) exchange.

    The multi-k exchange couples every ordered k-point pair ``(k, k′)``
    through the momentum transfer ``q = k - k′``: each channel needs
    the pair-FT tensor on the *shifted* reciprocal mesh ``{q + G}``.
    Channels with the same ``q`` modulo a reciprocal-lattice vector
    share their vector set and kernel; the per-``(q, k′)`` Bloch folds
    are cached individually. The ``q == 0`` (diagonal) channel reuses
    the :class:`JLongRangeCache` tables outright -- same vectors, same
    kernel, same cached fold tensors -- so at ``n_k = 1`` the multi-k
    entry point :func:`compute_K_long_range_at_k` reproduces
    :func:`compute_K_long_range_gamma` to machine precision.

    Memory note: each fold tensor is ``(nbf, nbf, n_K(q))`` complex and
    there are up to ``n_k^2`` of them (``n_k`` distinct ``q`` x ``n_k``
    ``k′`` folds). The per-cell FT stack for a ``q != 0`` channel is
    never materialised when the streaming C++ Bloch kernel is available
    (:func:`vibeqc._aopair_ft.ao_pair_fourier_transform_bloch`
    dispatch); the pure-Python fallback builds it in cell batches.
    """

    basis: BasisSet
    system: PeriodicSystem
    cells_r_cart: np.ndarray
    omega: float
    K_max: float
    j_cache: JLongRangeCache
    q_channels: dict[bytes, _QExchangeChannel] = field(
        default_factory=dict,
        repr=False,
    )

    def _canonical_q(self, q_cart: np.ndarray) -> tuple[bytes, np.ndarray]:
        """Wrap ``q`` to the first-BZ-adjacent representative.

        Fractional reciprocal coordinates ``f`` (``q = b @ f``) are
        rounded (Monkhorst-Pack differences are exact rationals; the
        rounding kills float noise) and wrapped to ``[-1/2, 1/2)`` so all
        ``q + G``-equivalent momentum transfers share one cache key and
        one deterministic vector enumeration.
        """
        a = np.asarray(self.system.lattice, dtype=float)
        q = np.asarray(q_cart, dtype=float).reshape(3)
        f = (a.T @ q) / (2.0 * math.pi)
        f = np.round(f, 9)
        f_wrapped = ((f + 0.5) % 1.0) - 0.5
        f_wrapped = np.round(f_wrapped, 9)
        b = 2.0 * math.pi * np.linalg.inv(a).T
        q_canonical = b @ f_wrapped
        return f_wrapped.tobytes(), q_canonical

    def channel_tables(
        self,
        q_cart: np.ndarray,
        k_prime_cart: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(kernel, B^{(k′)})`` for the channel ``q = k - k′``.

        ``B^{(k′)}_muν(q+G) = S_g exp(-i k′.R_g) . FT_muν(q+G; R_g)``
        with the libint Y_lm correction baked in -- the ``q``-shifted
        analogue of the ``ft_bloch`` tensors the J^LR build caches.
        """
        key, q_canonical = self._canonical_q(q_cart)
        is_q_zero = not np.any(np.frombuffer(key, dtype=float))
        return self._channel_tables_canonical(key, q_canonical, is_q_zero, k_prime_cart)

    def _channel_tables_on_mesh(
        self, mesh, target_index: int, source_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Private exact-address bridge to the correlation mesh diagnostic.

        Integer transfer labels select and center q before one binary64
        division. No Cartesian inverse fit or decimal rounding is used.
        This shares the HF Fourier and contraction implementation, but the
        existing SCF drivers still use ``channel_tables``. Neither route
        certifies real-space support or reciprocal cutoff boundary closure.
        The existing cache storage policy (up to all q/k panels) also applies.
        """
        from ._vibeqc_core import _RegularKMesh

        if not isinstance(mesh, _RegularKMesh):
            raise TypeError("mesh must be a native _RegularKMesh")
        transfer = mesh.transfer_address(mesh.transfer_index(source_index, target_index))
        centered = tuple(int(n) - (int(m) if int(n) >= int(m) // 2 else 0)
                         for n, m in zip(transfer, mesh.doubled_modulus))
        fractional = np.array([n / m for n, m in zip(centered, mesh.doubled_modulus)])
        reciprocal = 2.0 * math.pi * np.linalg.inv(np.asarray(self.system.lattice)).T
        # Separate exact-address and legacy Cartesian cache namespaces.
        key = b"regular-mesh/v1:" + repr((tuple(mesh.mesh), centered)).encode("ascii")
        return self._channel_tables_canonical(
            key, reciprocal @ fractional, not any(centered),
            reciprocal @ mesh.fractional_at(source_index),
        )

    def _channel_tables_canonical(
        self, key: bytes, q_canonical: np.ndarray, is_q_zero: bool,
        k_prime_cart: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if is_q_zero:
            # Diagonal channel: exactly the J^LR tables.
            ft_bloch = _bloch_pair_ft_from_cache(
                self.j_cache,
                np.asarray(k_prime_cart, dtype=float).reshape(3),
            )
            return self.j_cache.kernel, ft_bloch

        chan = self.q_channels.get(key)
        if chan is None:
            K_vectors = compute_shifted_reciprocal_lattice_vectors(
                self.system,
                q_canonical,
                self.K_max,
            )
            if K_vectors.shape[0] == 0:
                raise RuntimeError(
                    "KExchangeLongRangeCache: empty shifted K-list for "
                    f"q = {q_canonical}; K_max = {self.K_max}"
                )
            a = np.asarray(self.system.lattice, dtype=float)
            V = float(abs(np.linalg.det(a)))
            K2 = (K_vectors**2).sum(axis=1)
            kernel = (
                (4.0 * math.pi / V)
                * np.exp(-K2 / (4.0 * self.omega * self.omega))
                / K2
            )
            chan = _QExchangeChannel(K_vectors=K_vectors, kernel=kernel)
            self.q_channels[key] = chan

        k_prime = np.asarray(k_prime_cart, dtype=float).reshape(3)
        k_key = _k_cache_key(k_prime)
        B = chan.b_per_k.get(k_key)
        if B is None:
            from ._aopair_ft import ao_pair_fourier_transform_bloch

            # S_g exp(-ik′.R_g) FT_g -- the bloch helper applies +ik
            # phases, so request -k′.
            if self.j_cache.pair_cutoff_bohr is None:
                B = ao_pair_fourier_transform_bloch(
                    self.basis, chan.K_vectors, self.cells_r_cart, -k_prime,
                )
            else:
                from .lattice_screening import ao_pair_support_mask

                ft = ao_pair_fourier_transform_at_cells(
                    self.basis, chan.K_vectors, self.cells_r_cart,
                )
                for cell, shift in enumerate(self.cells_r_cart):
                    ft[cell] *= ao_pair_support_mask(
                        self.basis, shift, self.j_cache.pair_cutoff_bohr,
                    )[:, :, None]
                B = np.einsum("g,gmnk->mnk", np.exp(-1j * (self.cells_r_cart @ k_prime)), ft)

            correction = _libint_ylm_correction_per_ao(self.basis)
            B = B * (correction[:, None, None] * correction[None, :, None])
            chan.b_per_k[k_key] = B
        return chan.kernel, B


def build_k_exchange_long_range_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    j_cache: JLongRangeCache,
    *,
    K_max: float,
) -> KExchangeLongRangeCache:
    """Construct the multi-k LR-exchange cache sharing ``j_cache``'s state.

    The shifted-channel tables consume the same Ewald w and the same
    reciprocal envelope ``K_max`` as the J^LR cache (matched-envelope
    discipline: one shared Ewald state across V_ne / E_nn / J^LR / K^LR).
    """
    return KExchangeLongRangeCache(
        basis=basis,
        system=system,
        cells_r_cart=np.ascontiguousarray(j_cache.cells_r_cart, dtype=float),
        omega=float(j_cache.omega),
        K_max=float(K_max),
        j_cache=j_cache,
    )


def compute_K_long_range_at_k(
    x_cache: KExchangeLongRangeCache,
    k_cart: np.ndarray,
    k_points: Sequence[np.ndarray],
    weights: Sequence[float],
    D_k_list: Sequence[np.ndarray],
) -> np.ndarray:
    """Long-range (erf) exchange matrix at crystal momentum ``k``.

    Multi-k generalisation of :func:`compute_K_long_range_gamma` --
    every k-point pair contributes a momentum-transfer channel
    ``q = k - k′``::

        K^{LR}_muν(k) = S_{k′} w_{k′} S_{q+G != 0}
            (4pi/V) . e^{-|q+G|^2/4w^2} / |q+G|^2
            . [B^{(k′)}(q+G)]*_{mul} . D_{ls}(k′) . B^{(k′)}(q+G)_{νs}

    with ``B^{(k′)}_muν(K) = S_g exp(-ik′.R_g).FT_muν(K; R_g)`` the
    Bloch-folded shifted-ν AO-pair FT. Each ``G`` term is manifestly
    Hermitian for Hermitian ``D(k′)`` (sandwich form ``A* D Aᵀ``), so
    the result is Hermitian channel-by-channel.

    Derivation: Bloch-fold the real-space exchange lattice sum with
    ``D(g) = S_{k′} w_{k′} e^{-ik′.R_g} D(k′)`` and insert the
    continuous-FT representation of the erf kernel; the lattice sum
    over ket cells collapses the FT integral onto ``K = q + G``. The
    bra fold carries phase ``+k′`` conjugated; the ket fold reduces to
    the SAME ``B^{(k′)}`` tensor via ``FT_{sν}(K; R) =
    e^{-iK.R}.FT_{νs}(K; -R)`` and ``e^{i(k-q-G).R} = e^{ik′.R}``.
    Equivalently: this is the Γ-point exchange split of the BvK
    supercell, exactly unfolded -- the supercell reciprocal mesh is
    ``{q + G}`` and ``(1/V_sc).S_{K_sc}`` reproduces the
    ``w_{k′}.(1/V)`` weights. The ``q + G = 0`` term (diagonal channel
    only) is excluded here and handled by the exxdiv correction
    ``(ξ_M(supercell) - pi/(V_sc.w^2)).S(k)D(k)S(k)`` -- see
    :func:`probe_charge_madelung_supercell`. q->0 singularity treatment
    per Gygi & Baldereschi, Phys. Rev. B 34, 4405(R) (1986),
    doi:10.1103/PhysRevB.34.4405; probe-charge Ewald form per
    Sundararaman & Arias, Phys. Rev. B 87, 165122 (2013),
    doi:10.1103/PhysRevB.87.165122; Bloch AO-pair FT factorisation per
    Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119
    (2017), doi:10.1063/1.4998644.

    At ``n_k = 1`` (``k = k′ = Γ``) the single channel is ``q == 0``,
    which reuses the J-cache tables verbatim and reproduces
    :func:`compute_K_long_range_gamma` to machine precision (pinned in
    tests/test_bipole_fock_ewald_exchange.py).

    Parameters
    ----------
    x_cache : KExchangeLongRangeCache
    k_cart : (3,)
        The k-point at which the exchange matrix is assembled.
    k_points : sequence of (3,)
        The full (uniform-weight) Monkhorst-Pack mesh.
    weights : sequence of float
        k-point weights summing to 1.
    D_k_list : sequence of (nbf, nbf)
        Per-k density matrices (Hermitian; complex allowed).

    Returns
    -------
    K_LR_k : (nbf, nbf) complex
    """
    k_arr = np.asarray(k_cart, dtype=float).reshape(3)
    nbf = int(x_cache.basis.nbasis)
    K_LR = np.zeros((nbf, nbf), dtype=np.complex128)
    for k_idx, k_prime in enumerate(k_points):
        kp = np.asarray(k_prime, dtype=float).reshape(3)
        w = float(weights[k_idx])
        kernel, B = x_cache.channel_tables(k_arr - kp, kp)
        D_kp = np.asarray(D_k_list[k_idx], dtype=complex)
        K_LR = K_LR + w * _contract_K_long_range_channel_linear(kernel, B, D_kp)
    K_LR = 0.5 * (K_LR + K_LR.conj().T)
    return K_LR


def _contract_K_long_range_channel_linear(
    kernel: np.ndarray, B: np.ndarray, density: np.ndarray,
) -> np.ndarray:
    """One unweighted, unprojected HF exchange channel, complex-linear in D.

    Mesh weights, zero-mode and probe-charge terms belong to the caller.
    Keeping this contraction shared exposes arbitrary non-Hermitian density
    directions to the native correlation-source comparisons.
    """
    return np.einsum("x,mlx,ls,nsx->mn", kernel, B.conj(), density, B, optimize=True)


def probe_charge_madelung_supercell(
    system: PeriodicSystem,
    mesh: Sequence[int],
    alpha: Optional[float] = None,
    precision: float = 1e-12,
) -> float:
    """Probe-charge Ewald (Madelung) constant of the BvK supercell.

    The multi-k exchange G=0 correction uses the Madelung constant of
    the Born-von-Kármán supercell spanned by the Monkhorst-Pack mesh
    (lattice columns ``n_i.a_i``) -- the multi-k SCF *is* the Γ-point SCF
    of that supercell, exactly unfolded, and PySCF's
    ``_ewald_exxdiv_for_G0`` applies this same single constant
    ``madelung(cell, kpts)`` to every k-point. Reduces to
    :func:`probe_charge_madelung` at mesh = (1, 1, 1). Scales as
    ``ξ_M(N.cell) ≈ ξ_M(cell)/N^{1/3}`` for cubic meshes -- the
    finite-size exchange correction vanishing in the dense-mesh limit.

    Parameters
    ----------
    system : PeriodicSystem (dim = 3)
    mesh : (3,) ints
        Monkhorst-Pack mesh dimensions (n1, n2, n3).
    alpha, precision
        Forwarded to :func:`probe_charge_madelung` (a-invariant).
    """
    from ._vibeqc_core import Atom

    if system.dim != 3:
        raise ValueError(
            "probe_charge_madelung_supercell requires dim=3; "
            f"got dim={system.dim}"
        )
    n = [int(x) for x in mesh]
    if len(n) != 3 or any(x < 1 for x in n):
        raise ValueError(
            f"mesh must be three positive integers; got {list(mesh)}"
        )
    a = np.asarray(system.lattice, dtype=float)
    a_super = a * np.asarray(n, dtype=float)[None, :]
    # The Madelung routine consumes only the lattice; a neutral
    # closed-shell placeholder atom keeps PeriodicSystem construction
    # happy.
    probe_system = PeriodicSystem(
        3,
        a_super,
        [Atom(2, [0.0, 0.0, 0.0])],
    )
    return probe_charge_madelung(
        probe_system,
        alpha=alpha,
        precision=precision,
    )


def probe_charge_madelung(
    system: PeriodicSystem,
    alpha: Optional[float] = None,
    precision: float = 1e-12,
) -> float:
    """Probe-charge Ewald (Madelung) constant of the unit cell.

    ξ_M = -2.E_Ewald(single unit point charge + neutralizing
    background), the constant used for the exchange K = 0
    finite-size correction (PySCF ``exxdiv='ewald'`` equivalent:
    ``vk += ξ_M.S.D.S``). Treatment of the q -> 0 Coulomb
    singularity in exact exchange per F. Gygi & A. Baldereschi,
    Phys. Rev. B 34, 4405(R) (1986), doi:10.1103/PhysRevB.34.4405;
    probe-charge Ewald form per R. Sundararaman & T. A. Arias,
    Phys. Rev. B 87, 165122 (2013),
    doi:10.1103/PhysRevB.87.165122 (Sec. II.B).

    Standard Ewald split for a unit point charge at the origin::

        E = 1/2 S_{R!=0} erfc(a|R|)/|R|                    (real)
          + (2pi/V) S_{G!=0} e^{-G^2/4a^2}/G^2              (reciprocal)
          - a/√pi                                        (self)
          - pi/(2Va^2)                                    (background)

    a-invariant; pinned against PySCF ``tools.pbc.madelung`` =
    0.576295611599 on the MgO fcc primitive cell (a = 4.21 Å) in
    tests/test_bipole_fock_ewald_exchange.py.

    Parameters
    ----------
    system : PeriodicSystem (dim = 3)
    alpha : float, optional
        Ewald splitting parameter; defaults to the CRYSTAL-default
        a(V). Any a > 0 gives the same ξ_M to ``precision``.
    precision : float
        Real/reciprocal lattice-sum truncation target.

    Returns
    -------
    xi : float
        ξ_M in Hartree (positive for compact cells).
    """
    if system.dim != 3:
        raise ValueError(
            f"probe_charge_madelung requires dim=3; got dim={system.dim}"
        )
    a = np.asarray(system.lattice, dtype=float)
    V = float(abs(np.linalg.det(a)))
    alpha_val = (
        float(alpha) if alpha is not None else crystal_default_ewald_alpha(V)
    )
    if alpha_val <= 0.0:
        raise ValueError(f"alpha must be > 0; got {alpha_val}")
    ln_p = -math.log(float(precision))
    r_max = math.sqrt(ln_p) / alpha_val
    k_max = 2.0 * alpha_val * math.sqrt(ln_p)

    from ._vibeqc_core import direct_lattice_cells

    e_real = 0.0
    for cell in direct_lattice_cells(system, r_max):
        r = float(np.linalg.norm(np.asarray(cell.r_cart, dtype=float)))
        if r < 1e-12:
            continue
        e_real += 0.5 * math.erfc(alpha_val * r) / r

    K_vectors = compute_reciprocal_lattice_vectors(system, k_max)
    K2 = (K_vectors**2).sum(axis=1)
    e_recip = (2.0 * math.pi / V) * float(
        np.sum(np.exp(-K2 / (4.0 * alpha_val * alpha_val)) / K2)
    )
    e_self = -alpha_val / math.sqrt(math.pi)
    e_background = -math.pi / (2.0 * V * alpha_val * alpha_val)
    return -2.0 * (e_real + e_recip + e_self + e_background)


def exchange_q0_gauge_constant(
    xi_madelung: float,
    alpha: float,
    cell_volume: float,
    n_k: int,
) -> float:
    """The ONE definition of the corrected-split ``q + G = 0`` constant.

    ::

        c_g0 = ξ_M(BvK supercell)  -  pi / (alpha^2 . V_cell . n_k)

    ``+ξ_M`` is the probe-charge Ewald finite-size correction for the
    exchange singularity (PySCF ``exxdiv='ewald'``); ``-pi/(alpha^2 V n_k)``
    removes the ``q + G = 0`` component the absolutely-convergent direct
    ``K_SR(erfc)`` sum already carries, since
    ``lim_{K->0} (4 pi / K^2)(1 - e^{-K^2 / 4 alpha^2}) = pi / alpha^2``
    and the Bloch fold normalises by the BvK supercell volume
    ``V_sc = n_k . V_cell``. q -> 0 treatment per Gygi & Baldereschi,
    Phys. Rev. B 34, 4405(R) (1986), doi:10.1103/PhysRevB.34.4405;
    probe-charge Ewald form per Sundararaman & Arias, Phys. Rev. B 87,
    165122 (2013), doi:10.1103/PhysRevB.87.165122.

    **Why this is a function and not eleven copies.** The expression was
    hand-rolled at eleven call sites across five modules: four in
    :mod:`vibeqc.pbc_bipole_fock` (restricted/unrestricted x Gamma/multi-k),
    four in :mod:`vibeqc.bipole_gradient`, and one each in the ROHF, ROKS and
    corrected-exchange builders. Issue #478 landed
    the same failure mode one level up: two hand-rolled copies of the Ewald
    real-space cutoff desynced the analytic gradient from the energy
    (max|dg| = 1.27e-3 against a 1e-3 gate) when one was edited. A gauge
    constant that appears in both an energy and its gradient has to have a
    single definition or the pair silently stops being each other's
    derivative.

    ``n_k`` must be the size of the **full** Monkhorst-Pack mesh
    (``prod(mesh)``), never an irreducible-wedge count: ``V_sc`` is the
    Born-von-Karman supercell volume, which is a property of the mesh
    dimensions and not of how many representatives the caller chose to
    diagonalise. Pinned by
    ``tests/test_bipole_exchange_q0_gauge.py``.

    Parameters
    ----------
    xi_madelung
        ``probe_charge_madelung_supercell(system, mesh)`` at multi-k;
        ``probe_charge_madelung(system)`` at Gamma. Pass ``0.0`` for
        ``exchange_exxdiv='none'``.
    alpha
        The Ewald split parameter the driver already uses for its Hartree
        J and nuclear gauge (bohr^-1).
    cell_volume
        Primitive-cell volume (bohr^3) -- NOT the supercell volume; the
        ``n_k`` factor supplies the supercell scaling.
    n_k
        Number of k-points in the full Monkhorst-Pack mesh.
    """
    alpha_f = float(alpha)
    volume_f = float(cell_volume)
    n_k_i = int(n_k)
    if alpha_f <= 0.0:
        raise ValueError(
            f"exchange_q0_gauge_constant: alpha must be positive; got {alpha!r}"
        )
    if volume_f <= 0.0:
        raise ValueError(
            "exchange_q0_gauge_constant: cell_volume must be positive; got "
            f"{cell_volume!r}"
        )
    if n_k_i < 1:
        raise ValueError(
            f"exchange_q0_gauge_constant: n_k must be >= 1; got {n_k!r}"
        )
    return float(xi_madelung) - math.pi / (
        alpha_f * alpha_f * volume_f * float(n_k_i)
    )
