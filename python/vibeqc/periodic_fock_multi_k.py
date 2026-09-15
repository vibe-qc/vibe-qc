"""Phase 12e-c-4c-iii-a: multi-k Fock builder for the composed
EWALD_3D Coulomb dispatch.

Produces ``F(k)`` at every k-point in a user-supplied set, built
from a real-space density matrix ``D_real``
(:class:`LatticeMatrixSet`). The Hartree J is built by default from
the analytic reciprocal-space AO-pair Fourier transform
(:func:`vibeqc.ewald_composed.compute_j_ewald_3d_ft_lattice`) -- the
multi-k sibling of the Γ ``build_j_ewald_3d``. It is w-invariant and
its per-cell blocks Bloch-sum to the Γ analytic-FT J at the (1,1,1)
mesh, so the multi-k drivers stay in bit-parity with the Γ driver
(the F4 ``test_multi_k_at_gamma_mesh_matches_gamma_driver`` gate)
now that the Γ J is w-invariant (F2). The full-range exchange K
comes from the real-space ERI kernel
(:func:`vibeqc.build_jk_2e_real_space`). The decomposition at each
lattice cell ``g`` is

    F^{2e}(g)  =  J(g)  -  (a/2) K_full(g)

and F(k) is the Bloch sum ``F^{2e}(k) = S_g e^{i k.R_g} F^{2e}(g)``
plus the one-electron Hcore(k).

Cost (default analytic-FT backend): the Hartree J is one analytic
reciprocal-space contraction (no real-space J pass); the full-range
exchange K -- only needed when ``exchange_scale > 0`` -- is one
``build_jk_2e_real_space`` call at w = 0 (we keep ``K_full`` and drop
the ``J`` it also returns). Pure DFT (``exchange_scale == 0``) makes
no real-space 2e call at all. This drops the pre-F2 path's erfc
``J_SR`` real-space pass.

The pre-F2 Ewald split (real-space erfc ``J_SR`` + FFT-Poisson
``J_LR`` on the multi-cell density, w-dependent at finite grid) is
retained for diagnostics behind ``VIBEQC_J_EWALD3D_BACKEND=grid``.

This module ships the builder in isolation. The full multi-k SCF
driver (k-mesh iteration loop, occupation determination, D_real
reconstruction) lands in 12e-c-4c-iii-b.

Convention on k-points: the routine takes k-points in **Cartesian**
coordinates (bohr⁻¹), matching :func:`vibeqc.bloch_sum`. Callers
that have fractional/reduced k-points should convert via
``k_cart = 2pi . (B @ k_frac)`` where B = (lattice.T)⁻¹ is the
reciprocal-lattice matrix.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    build_fock_2e_real_space,
)
from .periodic_density import build_j_long_range_periodic

__all__ = [
    "build_periodic_fock_ewald3d_k",
    "build_periodic_j_ewald3d_k_from_k_density",
    "build_periodic_fock_slab_ewald2d_k",
    "build_fock_2e_ewald3d_blocks",
    "build_fock_2e_slab_ewald2d_blocks",
    "ewald_3d_j_blocks",
    "make_ewald_3d_lattice_j_cache",
    "make_slab_ewald_2d_lattice_j_cache",
    "slab_ewald_2d_j_blocks",
]


def make_ewald_3d_lattice_j_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    cells,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
):
    """Build the per-cell EWALD_3D Hartree-J cache for a fixed cell list,
    or return ``None`` when caching does not apply.

    Mirrors :func:`vibeqc.ewald_composed.make_ewald_3d_gamma_j_builder`'s
    backend handling for the multi-k path: for the default
    ``VIBEQC_J_EWALD3D_BACKEND=analytic_ft`` it precomputes the
    iteration-invariant per-cell AO-pair FT / dense G-mesh / ``4pi/G^2``
    kernel (:func:`vibeqc.ewald_composed.build_j_ewald_3d_ft_lattice_cache`,
    reading ``VIBEQC_J_EWALD3D_KE``); for the diagnostic ``grid`` backend it
    returns ``None`` (that path is not the per-iteration hotspot and is left
    uncached). Pass the result as the ``j_cache`` argument of
    :func:`build_periodic_fock_ewald3d_k` (it threads down to
    :func:`compute_j_ewald_3d_ft_lattice`); ``None`` reproduces the uncached
    per-iteration rebuild exactly.

    ``cells`` is the density's lattice cell list (``D_real.cells``), which is
    fixed across SCF iterations, so the cache built once is valid for the
    whole SCF. This is the multi-k sibling of the Γ E2 fix in
    ``docs/pbc_audit_2026-06.md``.
    """
    backend = os.environ.get("VIBEQC_J_EWALD3D_BACKEND", "analytic_ft").lower()
    if backend != "analytic_ft":
        return None
    if int(system.dim) != 3:
        # dim < 3 systems run the FFT-Poisson grid backend (the analytic-FT
        # J kernel is 3D-only -- see the dim guard in ewald_3d_j_blocks);
        # the grid path is uncached, so caching does not apply here either.
        return None
    from .ewald_composed import build_j_ewald_3d_ft_lattice_cache

    ke = float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0"))
    return build_j_ewald_3d_ft_lattice_cache(
        basis, system, cells, lattice_opts=lattice_opts, ke_cutoff=ke
    )


def _ewald_3d_lattice_j_cache_fits_memory_target(
    basis: BasisSet,
    system: PeriodicSystem,
    cells,
    *,
    ke_cutoff: Optional[float] = None,
) -> bool:
    """Return whether dense exact-J cache reuse fits its memory target.

    This planning probe builds only the reciprocal-vector list, not the
    ``(n_cells, nbf, nbf, n_G)`` AO-pair tensor. The cache target accounts
    for that tensor plus one similarly sized construction temporary. A cache
    that fits can retain the fast pre-SCF reuse path; a larger one must use
    the separately bounded per-iteration contraction.
    """
    if int(system.dim) != 3:
        return False
    from .aux_basis import rsgdf_dense_g_mesh
    from .ewald_composed import _ewald_j_ft_cell_chunk_size

    cell_list = list(cells)
    if not cell_list:
        return True
    ke = (
        float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0"))
        if ke_cutoff is None
        else float(ke_cutoff)
    )
    G_all = rsgdf_dense_g_mesh(system, ke)
    n_g = int(np.count_nonzero((G_all**2).sum(axis=1) > 0.0))
    cache_target_mib = float(
        os.environ.get("VIBEQC_J_EWALD3D_CACHE_MIB", "4096.0")
    )
    cell_batch = _ewald_j_ft_cell_chunk_size(
        int(basis.nbasis),
        n_g,
        target_mib=cache_target_mib,
    )
    return len(cell_list) <= cell_batch


def make_slab_ewald_2d_lattice_j_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    cells,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha: float = 0.4,
):
    """Build the per-cell ``SLAB_EWALD_2D`` Hartree-J cache."""
    from .ewald_composed_slab import build_j_slab_ewald_2d_lattice_cache

    return build_j_slab_ewald_2d_lattice_cache(
        basis,
        system,
        cells,
        lattice_opts=lattice_opts,
        alpha=float(alpha),
    )


def ewald_3d_j_blocks(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    omega: float,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    j_cache=None,
) -> List[np.ndarray]:
    """Per-cell EWALD_3D Hartree ``J(g)`` blocks for a real-space density.

    The **single shared entry** for the multi-k EWALD_3D Hartree J, so the
    RHF/RKS Fock builder (:func:`build_fock_2e_ewald3d_blocks`) and the
    open-shell UHF/UKS multi-k drivers all derive J from one path -- what
    keeps the F4 Γ<->multi-k bit-equivalence
    (``test_multi_k_at_gamma_mesh_matches_gamma_driver``) holding across
    every spin/driver variant.

    * Default (``VIBEQC_J_EWALD3D_BACKEND=analytic_ft``): the full periodic
      Hartree per cell via
      :func:`vibeqc.ewald_composed.compute_j_ewald_3d_ft_lattice` --
      w-invariant, ``G = 0`` omitted, no SR/LR split. Its blocks Bloch-sum
      to the Γ ``build_j_ewald_3d`` J at the (1,1,1) mesh.
    * Diagnostic (``=grid``): the pre-F2 Ewald split, real-space erfc
      ``J_SR`` (:func:`vibeqc.build_fock_2e_real_space`) plus FFT-Poisson
      ``J_LR`` (:func:`vibeqc.build_j_long_range_periodic`). w-dependent at
      finite grid; kept for cross-checks only.

    Returns a list of ``(n_bf, n_bf)`` numpy arrays, one per cell in
    ``D_real.cells``.
    """
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    n_cells = len(D_real.cells)
    backend = os.environ.get("VIBEQC_J_EWALD3D_BACKEND", "analytic_ft").lower()

    if backend == "analytic_ft" and int(system.dim) != 3:
        # The analytic-FT J kernel (compute_j_ewald_3d_ft_lattice) is
        # 3D-only -- its 4pi/G^2 G=0-omitted kernel needs a 3D G-mesh and
        # raises on dim < 3. 1D/2D periodicity stays on the FFT-Poisson
        # Ewald split that served all dims before the FT migration
        # (f7ee5832); that migration broke every dim<3 multi-k EWALD_3D
        # run (CI was collect-only, so the red went undetected until the
        # 2026-06-10 b4a6faba merge-drop audit re-ran the multi-k
        # accelerator-uniformity suite on its H₂-chain fixtures).
        backend = "grid"

    if backend == "analytic_ft":
        from .ewald_composed import compute_j_ewald_3d_ft_lattice

        ke = float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0"))
        # ``j_cache`` (an EwaldJFTLatticeCache) reuses the iteration-
        # invariant per-cell AO-pair FT / G-mesh / kernel across SCF
        # iterations; None rebuilds inline (unchanged one-shot behaviour).
        return compute_j_ewald_3d_ft_lattice(
            basis,
            system,
            D_real,
            float(omega),
            lattice_opts=opts,
            ke_cutoff=ke,
            cache=j_cache,
        )

    if backend != "grid":
        raise ValueError(
            "VIBEQC_J_EWALD3D_BACKEND must be 'analytic_ft' or 'grid'; "
            f"got {backend!r}."
        )

    # Legacy diagnostic backend: Ewald split J_SR + FFT-Poisson J_LR.
    J_SR_lms = build_fock_2e_real_space(
        basis,
        system,
        opts,
        D_real,
        0.0,
        float(omega),
    )
    J_LR_blocks = build_j_long_range_periodic(
        basis,
        system,
        D_real,
        omega=float(omega),
        grid_shape=grid_shape,
        origin=origin,
        spacing_bohr=spacing_bohr,
        output_cells=list(range(n_cells)),
        # Composition site (J_SR + J_LR = J_full): the erf kernel's G=0
        # finite part must be restored. This branch is also the FORCED
        # backend for dim < 3, so low-dimensional multi-k EWALD_3D runs
        # depend on it (GRID-BACKEND-CONVERGES-WRONG).
        restore_g0_finite_part=True,
    )
    return [
        np.asarray(J_SR_lms.blocks[g], dtype=float)
        + np.asarray(J_LR_blocks[g], dtype=float)
        for g in range(n_cells)
    ]


def slab_ewald_2d_j_blocks(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    alpha: float,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    j_cache=None,
) -> List[np.ndarray]:
    """Per-cell ``SLAB_EWALD_2D`` Hartree ``J(g)`` blocks."""
    from .ewald_composed_slab import compute_j_slab_ewald_2d_lattice

    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    return compute_j_slab_ewald_2d_lattice(
        basis,
        system,
        D_real,
        lattice_opts=opts,
        alpha=float(alpha),
        cache=j_cache,
    )


def _exchange_blocks_for(
    basis: BasisSet,
    system: PeriodicSystem,
    opts: LatticeSumOptions,
    D_real: LatticeMatrixSet,
    exchange_scale: float,
    exchange_assembly,
    full_range_alpha=None,
    symmetry_reduction=None,
) -> Optional[List[np.ndarray]]:
    """Coefficient-folded K blocks from either the legacy scalar
    ``exchange_scale`` or a full CAM ``exchange_assembly`` (which
    supersedes the scalar when given). None = no exchange needed.

    ``full_range_alpha`` puts the full-range arm on the Ewald-split
    short-range kernel; see
    :func:`vibeqc.periodic_screened_exchange.build_exchange_blocks`."""
    from .periodic_screened_exchange import (
        PeriodicExchangeAssembly,
        build_exchange_blocks,
    )

    exx = exchange_assembly
    if exx is None:
        exx = PeriodicExchangeAssembly(float(exchange_scale), 0.0, 0.0)
    return build_exchange_blocks(
        basis,
        system,
        opts,
        D_real,
        exx,
        full_range_alpha=full_range_alpha,
        symmetry_reduction=symmetry_reduction,
    )


def build_fock_2e_ewald3d_blocks(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    omega: float,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    exchange_scale: float = 1.0,
    exchange_assembly=None,
    full_range_alpha=None,
    symmetry_reduction=None,
    j_cache=None,
) -> List[np.ndarray]:
    """Return the real-space F^{2e}(g) blocks for the EWALD_3D split.

    For each cell ``g`` in ``D_real.cells`` (default analytic-FT backend):

        F^{2e}(g)  =  J(g)  -  (a/2) K_full(g)

    where ``a = exchange_scale`` (1.0 = RHF / Hartree-Fock; 0.0 = pure
    DFT, K skipped entirely; intermediate = hybrid functionals like
    B3LYP at a = 0.2). When ``exchange_assembly`` (a
    :class:`vibeqc.periodic_screened_exchange.PeriodicExchangeAssembly`)
    is given it supersedes ``exchange_scale`` and the exchange term
    generalises to the CAM form
    ``(c_full*K_full(g) + c_sr*K_erfc(g, omega_screen)) / 2`` --
    screened hybrids (hse06) ride the erfc arm.

    Internal mechanics (default ``VIBEQC_J_EWALD3D_BACKEND=analytic_ft``):

      - J(g) = :func:`vibeqc.ewald_composed.compute_j_ewald_3d_ft_lattice`
        -- the full periodic Hartree per cell from the analytic
        reciprocal-space AO-pair FT (G = 0 omitted, w-invariant, no
        SR/LR split). Its per-cell blocks Bloch-sum to the Γ
        ``build_j_ewald_3d`` J at the (1,1,1) mesh, so the multi-k
        drivers stay bit-equivalent to the Γ driver (F4 gate).
      - K_full(g) = ``build_jk_2e_real_space(..., 0)`` -- only built when
        ``a > 0``; single C++ pass returns both real-space blocks and we
        keep ``K_full`` directly (dropping the ``J`` it also returns).
        Same builder + w = 0 as the Γ driver's K, so the (1,1,1) Bloch
        sum matches it.

    For pure DFT (``exchange_scale == 0``) the K build is short-
    circuited and ``F^{2e}(g) = J(g)`` -- no real-space 2e call at all.

    The legacy diagnostic backend ``VIBEQC_J_EWALD3D_BACKEND=grid``
    restores the pre-F2 Ewald split: ``J_SR`` from libint erfc integrals
    (``build_fock_2e_real_space`` at w) plus ``J_LR`` from FFT-Poisson
    collocation (:func:`vibeqc.build_j_long_range_periodic`). That path
    is w-dependent at finite grid resolution and is kept for cross-checks
    only.

    Returns a list of ``(n_bf, n_bf)`` numpy arrays, one per cell in
    ``D_real.cells``.
    """
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    alpha = float(exchange_scale)
    n_cells = len(D_real.cells)

    # Hartree J(g) via the shared EWALD_3D J-blocks helper (analytic-FT
    # default; FFT-Poisson split behind VIBEQC_J_EWALD3D_BACKEND=grid).
    J_blocks = ewald_3d_j_blocks(
        basis,
        system,
        D_real,
        float(omega),
        lattice_opts=opts,
        grid_shape=grid_shape,
        origin=origin,
        spacing_bohr=spacing_bohr,
        j_cache=j_cache,
    )

    K_blocks = _exchange_blocks_for(
        basis, system, opts, D_real, alpha, exchange_assembly,
        full_range_alpha=full_range_alpha,
        symmetry_reduction=symmetry_reduction,
    )
    if K_blocks is None:
        # Pure DFT: F^{2e}(g) = J(g); no K build.
        return [np.asarray(J_blocks[g], dtype=float) for g in range(n_cells)]

    # Hybrid / HF: F^{2e}(g) = J(g) - K_HF(g)/2, with the assembly
    # coefficients already folded into K_HF. Same builders + omega
    # convention as the Γ driver's K, so the (1,1,1) Bloch sum matches.
    return [
        np.asarray(J_blocks[g], dtype=float) - 0.5 * K_blocks[g]
        for g in range(n_cells)
    ]


def build_fock_2e_slab_ewald2d_blocks(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    alpha: float,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    exchange_scale: float = 1.0,
    exchange_assembly=None,
    j_cache=None,
) -> List[np.ndarray]:
    """Return real-space ``F^{2e}(g)`` blocks for ``SLAB_EWALD_2D``.

    ``alpha`` is the 2D Ewald split parameter (``slab_ewald_alpha``),
    unrelated to the exchange coefficients. ``exchange_assembly``
    supersedes ``exchange_scale`` exactly as in
    :func:`build_fock_2e_ewald3d_blocks`.
    """
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    n_cells = len(D_real.cells)
    J_blocks = slab_ewald_2d_j_blocks(
        basis,
        system,
        D_real,
        float(alpha),
        lattice_opts=opts,
        j_cache=j_cache,
    )
    K_blocks = _exchange_blocks_for(
        basis, system, opts, D_real, float(exchange_scale), exchange_assembly
    )
    if K_blocks is None:
        return [np.asarray(J_blocks[g], dtype=float) for g in range(n_cells)]

    return [
        np.asarray(J_blocks[g], dtype=float) - 0.5 * K_blocks[g]
        for g in range(n_cells)
    ]


def _bloch_sum_blocks(
    blocks: Sequence[np.ndarray],
    cells,
    k_cart: np.ndarray,
) -> np.ndarray:
    """Python-side Bloch sum: F(k) = S_g e^{i k.R_g} F(g)."""
    k = np.asarray(k_cart, dtype=float).reshape(3)
    F_k = np.zeros_like(blocks[0], dtype=complex)
    for g_idx, block in enumerate(blocks):
        R_g = np.asarray(cells[g_idx].r_cart, dtype=float)
        phase = np.exp(1j * float(np.dot(k, R_g)))
        F_k = F_k + phase * block
    return F_k


def build_periodic_j_ewald3d_k_from_k_density(
    basis: BasisSet,
    system: PeriodicSystem,
    D_k_list: Sequence[np.ndarray],
    k_points_cart: Sequence[np.ndarray],
    weights: Sequence[float],
    cells,
    omega: float,
    *,
    ke_cutoff: Optional[float] = None,
    reciprocal_chunk_size: int = 512,
    cell_chunk_size: Optional[int] = None,
) -> List[np.ndarray]:
    """Build exact EWALD_3D Hartree ``J(k)`` from per-k densities.

    Unlike :func:`build_periodic_fock_ewald3d_k`, this entry never forms a
    :class:`LatticeMatrixSet` density or retains the all-cell/all-G AO-pair
    Fourier-transform tensor. The reciprocal and output-cell axes are
    contracted in bounded batches before the resulting ``J(g)`` blocks are
    Bloch-folded to every requested k-point.

    ``ke_cutoff=None`` preserves the driver-facing
    ``VIBEQC_J_EWALD3D_KE`` override used by the legacy lattice builder.
    """
    from .ewald_composed import compute_j_ewald_3d_ft_k_density_to_cells

    ke = (
        float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0"))
        if ke_cutoff is None
        else float(ke_cutoff)
    )
    cell_list = list(cells)
    # Sun et al., J. Chem. Phys. 147, 164119 (2017), Eqs. 16-17:
    # rho(G) contracts the Bloch AO-pair FT and J uses 4*pi/G^2 with G=0
    # omitted. Batching changes only evaluation order. On non-TRIM H2 in a
    # 12-bohr cell the bounded path differs from the legacy lattice route by
    # at most 7.9e-14 Ha per J block (random complex PSD D(k), 3x1x1 mesh).
    J_blocks = compute_j_ewald_3d_ft_k_density_to_cells(
        basis,
        system,
        D_k_list,
        k_points_cart,
        weights,
        cell_list,
        float(omega),
        ke_cutoff=ke,
        chunk_size=int(reciprocal_chunk_size),
        cell_chunk_size=cell_chunk_size,
    )

    k_list = [np.asarray(k, dtype=float).reshape(3) for k in k_points_cart]
    if not cell_list:
        shape = (int(basis.nbasis), int(basis.nbasis))
        return [np.zeros(shape, dtype=np.complex128) for _ in k_list]
    cell_r = [np.asarray(c.r_cart, dtype=float) for c in cell_list]
    try:
        from ._vibeqc_core import bloch_sum_multi_k as _bloch_mk
    except ImportError:
        return [_bloch_sum_blocks(J_blocks, cell_list, k) for k in k_list]
    result = _bloch_mk(J_blocks, cell_r, k_list)
    return [np.asarray(block) for block in result]


def build_periodic_fock_ewald3d_k(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    omega: float,
    k_points_cart: Sequence[np.ndarray],
    *,
    Hcore_k: Optional[Sequence[np.ndarray]] = None,
    lattice_opts: Optional[LatticeSumOptions] = None,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    exchange_scale: float = 1.0,
    exchange_assembly=None,
    full_range_alpha=None,
    symmetry_reduction=None,
    j_cache=None,
) -> List[np.ndarray]:
    """Compute F(k) at every k-point in ``k_points_cart`` using the
    composed EWALD_3D Coulomb dispatch.

    Parameters
    ----------
    basis, system
        AO basis and periodic system.
    D_real
        Real-space density matrix as a :class:`LatticeMatrixSet`.
    omega
        Ewald splitting parameter.
    k_points_cart
        Iterable of Cartesian k-vectors (bohr⁻¹), shape ``(3,)`` each.
    Hcore_k
        Optional pre-computed list of Hcore(k) matrices -- same
        length as ``k_points_cart``. When provided, the returned
        F(k) is the full Hamiltonian; otherwise only F^{2e}(k).
    lattice_opts, grid_shape, origin, spacing_bohr
        Forwarded to the short-range and long-range J builders.
    exchange_scale
        Fraction of HF exchange to retain (default 1.0 = full HF;
        0.0 = pure DFT, K skipped; 0.2 = B3LYP-like hybrid). Forwarded
        verbatim to :func:`build_fock_2e_ewald3d_blocks`, along with
        ``exchange_assembly`` (the CAM generalisation, which
        supersedes ``exchange_scale`` when given).

    Returns
    -------
    List of complex ``(n_bf, n_bf)`` matrices, one per k-point.
    """
    F_2e_blocks = build_fock_2e_ewald3d_blocks(
        basis,
        system,
        D_real,
        omega,
        lattice_opts=lattice_opts,
        grid_shape=grid_shape,
        origin=origin,
        spacing_bohr=spacing_bohr,
        exchange_scale=exchange_scale,
        exchange_assembly=exchange_assembly,
        full_range_alpha=full_range_alpha,
        symmetry_reduction=symmetry_reduction,
        j_cache=j_cache,
    )

    cells = D_real.cells
    cell_r = [np.asarray(c.r_cart, dtype=float) for c in cells]
    nbf = F_2e_blocks[0].shape[0]

    # Use the fused C++ multi-k Bloch sum when available.
    k_list = [np.asarray(k, dtype=float).reshape(3) for k in k_points_cart]
    try:
        from .._vibeqc_core import bloch_sum_multi_k as _bloch_mk
    except ImportError:
        pass
    else:
        if Hcore_k is not None:
            from .._vibeqc_core import assemble_fock_multi_k as _asm_mk

            hcore_list = [np.asarray(h, dtype=complex) for h in Hcore_k]
            result = _asm_mk(F_2e_blocks, cell_r, k_list, hcore_list)
        else:
            result = _bloch_mk(F_2e_blocks, cell_r, k_list)
        return [np.asarray(r) for r in result]

    # Python fallback: per-k Bloch sum.
    F_k_list: List[np.ndarray] = []
    for k_idx, k_cart in enumerate(k_points_cart):
        F_k = _bloch_sum_blocks(F_2e_blocks, cells, k_cart)
        if Hcore_k is not None:
            F_k = F_k + np.asarray(Hcore_k[k_idx], dtype=complex)
        F_k_list.append(F_k)
    return F_k_list


def build_periodic_fock_slab_ewald2d_k(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    alpha: float,
    k_points_cart: Sequence[np.ndarray],
    *,
    Hcore_k: Optional[Sequence[np.ndarray]] = None,
    lattice_opts: Optional[LatticeSumOptions] = None,
    exchange_scale: float = 1.0,
    exchange_assembly=None,
    j_cache=None,
) -> List[np.ndarray]:
    """Compute ``SLAB_EWALD_2D`` Fock matrices at every k-point."""
    F_2e_blocks = build_fock_2e_slab_ewald2d_blocks(
        basis,
        system,
        D_real,
        float(alpha),
        lattice_opts=lattice_opts,
        exchange_scale=exchange_scale,
        exchange_assembly=exchange_assembly,
        j_cache=j_cache,
    )

    cells = D_real.cells
    cell_r = [np.asarray(c.r_cart, dtype=float) for c in cells]
    k_list = [np.asarray(k, dtype=float).reshape(3) for k in k_points_cart]
    try:
        from .._vibeqc_core import bloch_sum_multi_k as _bloch_mk
    except ImportError:
        pass
    else:
        if Hcore_k is not None:
            from .._vibeqc_core import assemble_fock_multi_k as _asm_mk

            hcore_list = [np.asarray(h, dtype=complex) for h in Hcore_k]
            result = _asm_mk(F_2e_blocks, cell_r, k_list, hcore_list)
        else:
            result = _bloch_mk(F_2e_blocks, cell_r, k_list)
        return [np.asarray(r) for r in result]

    F_k_list: List[np.ndarray] = []
    for k_idx, k_cart in enumerate(k_points_cart):
        F_k = _bloch_sum_blocks(F_2e_blocks, cells, k_cart)
        if Hcore_k is not None:
            F_k = F_k + np.asarray(Hcore_k[k_idx], dtype=complex)
        F_k_list.append(F_k)
    return F_k_list
