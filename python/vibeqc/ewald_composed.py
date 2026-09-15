"""Phase 12e-c-4/F2: Ewald-3D Hartree J matrix.

Builds a single w-invariant periodic Hartree J matrix:

* The default backend evaluates the full periodic Hartree matrix in
  reciprocal space from the analytical AO-pair Fourier transform (the
  same machinery used by the periodic GDF and V_ne paths), with the
  ``G = 0`` mode omitted by the standard jellium convention.
* The legacy Ewald split, ``J_SR`` from libint erfc integrals plus
  ``J_LR`` from FFT-Poisson collocation, remains available for
  diagnostics via ``VIBEQC_J_EWALD3D_BACKEND=grid``.

The Ewald identity holds exactly at infinite basis / infinite grid:
``J(D)`` is independent of the splitting parameter w. At finite
resolution of the reciprocal-space FT mesh there is a small
Makov-Payne-like box-size correction that appears when the cell is
CHARGED (Q = ∫r dr != 0) -- see
``Makov-Payne correction`` below.

Use this builder directly for molecular-limit validation (where you
can compare against isolated-molecule ERIs) and as the reference J for
3D periodic Hartree-Fock SCF on neutral crystals.

Makov-Payne correction
----------------------

For a non-neutral density in a finite periodic cell, ``J``
differs from the isolated-molecule ``J_full`` by a **scalar-times-
overlap** shift

    a . S_muν,   a ≈ -a_M . Q / L

where Q = ∫r dr is the cell charge, L = V_cell^(1/3) is a
characteristic cell length, and a_M ≈ 2.837 is the simple-cubic
Madelung constant. a is w-independent to numerical precision -- it
represents the purely geometric self-interaction of the charge with
its uniform compensating background.

For real neutral crystals (Q_electrons + Q_nuclei = 0) this Makov-
Payne term cancels against the matching nuclear contribution and
``J`` is the correct periodic Hartree matrix. For molecular-
limit comparison against isolated J_full, the a.S correction must be
added explicitly -- the helper :func:`makov_payne_coefficient_cubic`
below returns a given a cubic cell side length and cell charge.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
    build_jk_gamma_molecular_limit,
)
from .ewald_j import auto_grid, build_j_long_range

# Default Ewald tolerance (matches C++ EwaldOptions::tolerance).
EWALD_DEFAULT_TOLERANCE = 1e-12


def auto_ewald_alpha(real_cutoff_bohr: float) -> float:
    """Derive Ewald splitting a from real-space cutoff.  **Deprecated.**

    Matches the C++ auto_alpha formula: a = √(-log(tol)) / real_cutoff,
    with tol = EWALD_DEFAULT_TOLERANCE (1e-12).

    .. deprecated:: v0.10.0
        Use :func:`vibeqc.bipole_ext_el_pole.crystal_default_ewald_alpha`
        instead, which returns CRYSTAL's a = 2.8/V^{1/3}.  The old
        geometry-based formula had no relationship to the nuclear Ewald
        a and broke gauge consistency between V_ne, E_nn, and J_LR.
        This function is retained for backward compatibility only and
        is no longer used by any EWALD_3D SCF driver.

    Parameters
    ----------
    real_cutoff_bohr
        Real-space Ewald cutoff in Bohr (default 25.0 for nuclear).

    Returns
    -------
    float
        The a parameter in bohr⁻¹.
    """
    t = -math.log(EWALD_DEFAULT_TOLERANCE)
    return math.sqrt(t) / real_cutoff_bohr


__all__ = [
    "auto_ewald_alpha",
    "build_j_ewald_3d",
    "build_j_ewald_3d_ft_gamma_cache",
    "build_j_ewald_3d_ft_lattice_cache",
    "compute_j_ewald_3d_ft_gamma",
    "compute_j_ewald_3d_ft_k_density_to_cells",
    "compute_j_ewald_3d_ft_lattice",
    "EwaldJFTGammaCache",
    "EwaldJFTLatticeCache",
    "makov_payne_coefficient_cubic",
    "make_ewald_3d_gamma_j_builder",
]


# Simple-cubic Madelung constant for the self-energy of a unit charge
# in a unit-volume cubic cell neutralised by a uniform background.
# Standard reference value; matches Fumi & Tosi (1964). Used only for
# the optional molecular-limit correction helper.
_SIMPLE_CUBIC_MADELUNG = 2.837297


def _ewald_j_ft_cell_chunk_size(
    nbf: int,
    n_g: int,
    *,
    target_mib: Optional[float] = None,
) -> int:
    """Return a bounded output-cell batch for the multi-k exact-J FT path."""
    n = max(1, int(nbf))
    g = max(1, int(n_g))
    if target_mib is None:
        target_mib = float(
            os.environ.get("VIBEQC_J_EWALD3D_FT_CELL_CHUNK_MIB", "512.0")
        )
    target_bytes = max(1.0, float(target_mib)) * 1024.0 * 1024.0
    # Account for the complex AO-pair FT tensor plus at least one similarly
    # sized temporary inside the FT / contraction stack.
    bytes_per_cell = (
        2.0 * float(np.dtype(np.complex128).itemsize) * n * n * g
    )
    return max(1, int(target_bytes // bytes_per_cell))


def makov_payne_coefficient_cubic(Q: float, L: float) -> float:
    """Return the Makov-Payne scalar a for a cubic cell of side ``L``
    (bohr) holding total charge ``Q`` (electrons; Q > 0 for an
    electron density).

    The correction ``a . S`` recovers the isolated-molecule J matrix
    from the periodic composition ``J_SR + J_LR`` -- see module
    docstring. The underlying formula is

        a = -a_M . Q / L,   a_M ≈ 2.837 (simple cubic).

    Only sensible for cubic cells. For general orthorhombic cells the
    Madelung constant varies with aspect ratio; use the w-invariance
    test (:func:`validate_ewald_identity`) to calibrate empirically.
    """
    return -_SIMPLE_CUBIC_MADELUNG * Q / L


def build_j_ewald_3d(
    basis: BasisSet,
    system: PeriodicSystem,
    D: np.ndarray,
    omega: float,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    ke_cutoff: Optional[float] = None,
) -> np.ndarray:
    """Composed Ewald-3D Hartree J matrix in the molecular-limit.

    Parameters
    ----------
    basis
        AO basis for the unit cell.
    system
        ``PeriodicSystem`` with the molecule wrapped in a (usually
        large) simulation cell.
    D
        Density matrix in the AO basis, shape ``(n_bf, n_bf)``. For
        UHF/UKS pass ``D_a + D_b``. Not symmetrized by this routine.
    omega
        Ewald splitting parameter (1 / bohr). The composed J is
        w-invariant up to numerical precision; ``w = 0.5`` is a
        reasonable default for most cells.
    lattice_opts
        :class:`LatticeSumOptions` for the short-range builder; default
        uses the builder's defaults (generally fine for molecular-limit
        cells).
    grid_shape, origin, spacing_bohr
        Legacy FFT-Poisson grid controls. They are ignored by the
        default analytical-FT backend and retained for the diagnostic
        ``VIBEQC_J_EWALD3D_BACKEND=grid`` path.
    ke_cutoff
        Kinetic-energy cutoff (Hartree) for the default analytical-FT
        G-mesh. ``None`` reads ``VIBEQC_J_EWALD3D_KE`` if set, else
        uses 200.0.

    Returns
    -------
    J : np.ndarray
        ``(n_bf, n_bf)`` symmetric periodic Hartree matrix (Hartree).
        For neutral crystals this is the physically-correct J. For
        molecular-limit comparison against isolated J_full, add the
        Makov-Payne ``a . S`` correction (see module docstring).
    """
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    backend = os.environ.get("VIBEQC_J_EWALD3D_BACKEND", "analytic_ft").lower()
    if backend == "analytic_ft":
        if ke_cutoff is None:
            ke = float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0"))
        else:
            ke = float(ke_cutoff)
        return compute_j_ewald_3d_ft_gamma(
            basis, system, D, omega, lattice_opts=opts, ke_cutoff=ke
        )
    if backend != "grid":
        raise ValueError(
            "VIBEQC_J_EWALD3D_BACKEND must be 'analytic_ft' or 'grid'; "
            f"got {backend!r}."
        )
    if opts.pair_complete_1e:
        raise NotImplementedError(
            "The diagnostic grid Hartree backend does not implement physical "
            "quartet support; use VIBEQC_J_EWALD3D_BACKEND=analytic_ft."
        )

    lat = np.asarray(system.lattice, dtype=float)
    if grid_shape is None:
        grid_shape = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape = (grid_shape, grid_shape, grid_shape)

    # Short-range piece: erfc(w r)/r in real space.
    jk_short = build_jk_gamma_molecular_limit(basis, system, opts, D, float(omega))
    J_sr = np.asarray(jk_short.J)

    # Long-range piece: erf(w r)/r via FFT Poisson. Passing ``system``
    # here is the v0.7 gauge fix: the AO basis is sampled with periodic
    # images so the FFT-Poisson density integrates to the right electron
    # count regardless of where the molecule sits in the cell. Without
    # ``system``, build_j_long_range falls back to the v0.6.x legacy
    # path which breaks translation invariance.
    J_lr = build_j_long_range(
        basis,
        D,
        lat,
        float(omega),
        grid_shape=grid_shape,
        origin=origin,
        spacing_bohr=spacing_bohr,
        system=system,
        # This is a COMPOSITION site: the returned object is the full
        # Hartree J, so the erf kernel's G=0 finite part must be present.
        # Without it J_SR + J_LR misses J_full by +pi Q^2/(2 w^2 V_cell)
        # -- the 14.5 mHa this backend disagreed with analytic_ft by
        # (GRID-BACKEND-CONVERGES-WRONG).
        restore_g0_finite_part=True,
    )

    return J_sr + J_lr


class EwaldJFTGammaCache:
    """Iteration-invariant machinery for the Γ analytic-FT Hartree J.

    Holds the three pieces of :func:`compute_j_ewald_3d_ft_gamma` that do
    **not** depend on the density matrix -- the AO-scaled Bloch AO-pair
    Fourier transform ``pair_ft`` (the dominant cost: it runs the
    McMurchie-Davidson / C++ FT kernel over the full ``(R_g, G)`` mesh),
    the ``4pi/G^2`` Coulomb kernel ``v_G``, and the inverse cell volume -- so
    an SCF loop can rebuild them **once** and redo only the O(n^2.n_G)
    density contraction each iteration. See
    :func:`build_j_ewald_3d_ft_gamma_cache` (constructor) and
    :func:`make_ewald_3d_gamma_j_builder` (the driver-facing closure).

    This is the EWALD sibling of the GDF driver's "build ``Lpq`` once,
    contract it per iteration" pattern -- the E2 efficiency fix in
    ``docs/pbc_audit_2026-06.md`` (the Γ EWALD per-iteration J rebuild was
    the dominant Γ-Ewald SCF cost, ~6 s/iteration on H₂/STO-3G/30-bohr).
    """

    __slots__ = ("pair_ft", "pair_ft_conj", "v_G", "inv_cell_volume", "nbf", "pair_cutoff")

    def __init__(
        self,
        pair_ft: np.ndarray,
        v_G: np.ndarray,
        inv_cell_volume: float,
        nbf: int,
        pair_cutoff=None,
    ) -> None:
        self.pair_ft = pair_ft            # (nbf, nbf, n_G), AO-scaled, complex
        self.pair_ft_conj = pair_ft.conj()
        self.v_G = v_G                    # (n_G,) = 4pi/G^2
        self.inv_cell_volume = inv_cell_volume
        self.nbf = nbf
        self.pair_cutoff = pair_cutoff


def _physical_pair_cutoff(lattice_opts):
    return (float(lattice_opts.cutoff_bohr)
            if lattice_opts is not None and lattice_opts.pair_complete_1e else None)


def _check_cache_pair_support(cache, lattice_opts):
    if lattice_opts is not None and cache.pair_cutoff != _physical_pair_cutoff(lattice_opts):
        raise ValueError("Hartree cache physical pair support differs from lattice options")


def _mask_physical_pair_ft(pair_ft, basis, translations, lattice_opts):
    """Apply the same finite AO-product support to source and output FTs."""
    if lattice_opts is not None and lattice_opts.pair_complete_1e:
        from .lattice_screening import ao_pair_support_mask

        for block, shift in zip(pair_ft, translations):
            block *= ao_pair_support_mask(
                basis, shift, lattice_opts.cutoff_bohr,
            )[:, :, None]
    return pair_ft


def build_j_ewald_3d_ft_gamma_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    ke_cutoff: float = 200.0,
) -> EwaldJFTGammaCache:
    """Precompute the density-independent pieces of the Γ analytic-FT J.

    Runs the iteration-invariant front half of
    :func:`compute_j_ewald_3d_ft_gamma` -- the dense G-mesh, the Bloch
    AO-pair FT ``pair_ft`` (with the per-L AO calibration applied), and
    the ``4pi/G^2`` kernel -- and returns them packaged as an
    :class:`EwaldJFTGammaCache`. Pass that cache back into
    :func:`compute_j_ewald_3d_ft_gamma` (or use
    :func:`make_ewald_3d_gamma_j_builder`) to reuse it across SCF
    iterations. The result is **bit-identical** to the uncached path; only
    the wasted per-iteration rebuild is removed.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import direct_lattice_cells
    from .aux_basis import _ao_scales_for_rsgdf, rsgdf_dense_g_mesh

    if system.dim != 3:
        raise ValueError(
            "build_j_ewald_3d_ft_gamma_cache: requires dim == 3; got "
            f"dim = {system.dim}."
        )
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    a_lattice = np.asarray(system.lattice, dtype=float)
    cell_volume = float(np.abs(np.linalg.det(a_lattice)))
    nbf = int(basis.nbasis)

    G_all = rsgdf_dense_g_mesh(system, float(ke_cutoff))
    G2 = (G_all**2).sum(axis=1)
    nonzero = G2 > 0
    G = G_all[nonzero]
    G2_kept = G2[nonzero]

    if G.shape[0] == 0:
        # No G != 0 modes: J is identically zero. Store empty arrays so the
        # contraction yields the (nbf, nbf) zero matrix (matches the
        # uncached ``np.zeros_like(D)`` early return).
        pair_ft = np.zeros((nbf, nbf, 0), dtype=np.complex128)
        v_G = np.zeros(0, dtype=float)
        return EwaldJFTGammaCache(pair_ft, v_G, 1.0 / cell_volume, nbf,
                                 _physical_pair_cutoff(opts))

    if opts.pair_complete_1e:
        from ._vibeqc_core import pair_complete_lattice_cells

        cells = pair_complete_lattice_cells(basis, system, float(opts.cutoff_bohr))
    else:
        cells = direct_lattice_cells(system, float(opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)

    if opts.pair_complete_1e:
        from ._aopair_ft import ao_pair_fourier_transform_at_cells

        pair_ft = np.zeros((nbf, nbf, len(G)), dtype=np.complex128)
        # Bound the extra image axis while retaining the physical pair mask.
        for shift in R_g:
            ft = ao_pair_fourier_transform_at_cells(basis, G, shift[None, :])
            pair_ft += _mask_physical_pair_ft(ft, basis, [shift], opts)[0]
    else:
        pair_ft = ao_pair_fourier_transform_bloch(
            basis, G, R_g, k_cart=np.zeros(3)
        )  # (n_orb, n_orb, n_G)

    # Per-L AO calibration: same convention used by the dense-FT GDF
    # and analytic V_ne paths.
    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_ft = pair_ft * np.outer(ao_scales, ao_scales)[:, :, None]

    v_G = 4.0 * np.pi / G2_kept
    return EwaldJFTGammaCache(pair_ft, v_G, 1.0 / cell_volume, nbf,
                             _physical_pair_cutoff(opts))


def compute_j_ewald_3d_ft_gamma(
    basis: BasisSet,
    system: PeriodicSystem,
    D: np.ndarray,
    omega: float,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    ke_cutoff: float = 200.0,
    cache: Optional[EwaldJFTGammaCache] = None,
) -> np.ndarray:
    """Γ-only Ewald-3D Hartree J via analytical reciprocal-space FT.

    Computes the full periodic Hartree matrix directly in reciprocal
    space with the ``G = 0`` Coulomb mode omitted. ``omega`` is accepted
    for API compatibility with the historical Ewald-split builder; the
    returned total J is independent of it.

    Pass ``cache`` (an :class:`EwaldJFTGammaCache` from
    :func:`build_j_ewald_3d_ft_gamma_cache`) to skip rebuilding the
    iteration-invariant AO-pair FT / G-mesh / kernel -- the SCF-loop fast
    path. When ``cache is None`` the cache is built inline, so a one-shot
    call is unchanged. Either way the returned J is bit-identical.
    """
    if cache is None:
        cache = build_j_ewald_3d_ft_gamma_cache(
            basis, system, lattice_opts=lattice_opts, ke_cutoff=ke_cutoff
        )
    D_arr = np.asarray(D, dtype=float)
    _check_cache_pair_support(cache, lattice_opts)

    # Density-dependent contraction only -- the cached ``pair_ft`` is the
    # analytic AO-pair Fourier transform r̂_muν(G), Eq. 16 of Sun,
    # Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119 (2017),
    # doi:10.1063/1.4998644. The G = 0-omitted periodic-Coulomb convention
    # (shared by PySCF pbc.df.AFTDF / GDF) is the ERI definition stated
    # around Eq. 17; Eq. 17 itself is the full mixed-density-fit 4-index
    # ERI.
    #   rho_D(G) = S_ls D_ls r̂_ls(G)
    #   J_muν = (1/Ω) S_{G!=0} (4pi/G^2) rho_D(G) conj(r̂_muν(G))
    # Validated: MgO STO-3G Γ EWALD_3D total energy is w-invariant for
    # w = 0.4, 0.6, 1.0 and targets PySCF KRHF exxdiv='ewald' at
    # -271.04994 Ha; LiH STO-3G Γ target -8.33527 Ha.
    rho_G = np.einsum("mn,mng->g", D_arr, cache.pair_ft)
    J = cache.inv_cell_volume * np.real(
        np.einsum("g,g,mng->mn", cache.v_G, rho_G, cache.pair_ft_conj)
    )
    J = 0.5 * (J + J.T)
    return J


def make_ewald_3d_gamma_j_builder(
    basis: BasisSet,
    system: PeriodicSystem,
    *,
    omega: float,
    lattice_opts: Optional[LatticeSumOptions] = None,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    ke_cutoff: Optional[float] = None,
):
    """Return a closure ``J(D)`` for the Γ EWALD_3D Hartree matrix that
    caches the iteration-invariant analytic-FT machinery once.

    The default ``analytic_ft`` backend precomputes the Bloch AO-pair FT,
    the dense G-mesh, and the ``4pi/G^2`` kernel a single time
    (:func:`build_j_ewald_3d_ft_gamma_cache`); each returned ``J(D)`` call
    then redoes only the O(n^2.n_G) density contraction. This mirrors the
    GDF driver building ``Lpq`` once and contracting it per iteration -- the
    E2 fix in ``docs/pbc_audit_2026-06.md`` (the Γ EWALD per-iteration J
    rebuild was the dominant Γ-Ewald SCF cost). The returned J is
    bit-identical to calling :func:`build_j_ewald_3d` each iteration.

    The diagnostic ``VIBEQC_J_EWALD3D_BACKEND=grid`` path is **not** cached
    (it is bisection-only and not the per-iteration hotspot): the closure
    forwards to :func:`build_j_ewald_3d` each call, so that env-gated path
    is byte-for-byte unchanged.
    """
    backend = os.environ.get("VIBEQC_J_EWALD3D_BACKEND", "analytic_ft").lower()
    if backend == "analytic_ft":
        if ke_cutoff is None:
            ke = float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0"))
        else:
            ke = float(ke_cutoff)
        opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
        cache = build_j_ewald_3d_ft_gamma_cache(
            basis, system, lattice_opts=opts, ke_cutoff=ke
        )

        def _build_j_cached(D: np.ndarray) -> np.ndarray:
            return compute_j_ewald_3d_ft_gamma(
                basis, system, D, omega, cache=cache
            )

        return _build_j_cached

    # grid (or any non-analytic_ft backend string): defer to
    # build_j_ewald_3d per call so the env-gated diagnostic path is
    # unchanged. build_j_ewald_3d validates the backend string itself.
    def _build_j_uncached(D: np.ndarray) -> np.ndarray:
        return build_j_ewald_3d(
            basis,
            system,
            D,
            omega=omega,
            lattice_opts=lattice_opts,
            grid_shape=grid_shape,
            origin=origin,
            spacing_bohr=spacing_bohr,
            ke_cutoff=ke_cutoff,
        )

    return _build_j_uncached


class EwaldJFTLatticeCache:
    """Iteration-invariant machinery for the multi-k (per-cell) analytic-FT
    Hartree J.

    Holds the AO-scaled per-cell AO-pair FT ``pair_at_cells``
    ``(n_cells, nbf, nbf, n_G)`` -- the dominant cost (the FT kernel over
    the whole ``(R_g, G)`` mesh) -- plus the ``4pi/G^2`` kernel ``v_G`` and
    the inverse cell volume, keyed to a **fixed lattice cell list** so a
    multi-k SCF loop redoes only the per-cell density contraction each
    iteration. See :func:`build_j_ewald_3d_ft_lattice_cache` (constructor)
    and the ``j_cache`` plumbing in :mod:`vibeqc.periodic_fock_multi_k`.

    The conjugate is **not** stored (unlike the small Γ cache): this tensor
    can be large and persisting a second copy risks OOM on big cells (see
    the OOM note in :func:`vibeqc._aopair_ft.ao_pair_fourier_transform_bloch`).
    The contraction conjugates inline -- exactly as the uncached path did --
    so the cache's resident memory equals one ``pair_at_cells``, the same
    tensor one uncached iteration already materialises.

    The cell list is captured at construction; reusing the cache with a
    density on a different cell list is a programming error (guarded by a
    cell-count check in :func:`compute_j_ewald_3d_ft_lattice`).
    """

    __slots__ = ("pair_at_cells", "v_G", "inv_cell_volume", "n_cells", "n_bf", "cell_keys", "pair_cutoff")

    def __init__(
        self,
        pair_at_cells: np.ndarray,
        v_G: np.ndarray,
        inv_cell_volume: float,
        n_cells: int,
        n_bf: int,
        cell_keys=None,
        pair_cutoff=None,
    ) -> None:
        self.pair_at_cells = pair_at_cells  # (n_cells, nbf, nbf, n_G) complex
        self.v_G = v_G                       # (n_G,) = 4pi/G^2
        self.inv_cell_volume = inv_cell_volume
        self.n_cells = n_cells
        self.n_bf = n_bf
        self.cell_keys = cell_keys
        self.pair_cutoff = pair_cutoff


def build_j_ewald_3d_ft_lattice_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    cells,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    ke_cutoff: float = 200.0,
) -> EwaldJFTLatticeCache:
    """Precompute the density-independent pieces of
    :func:`compute_j_ewald_3d_ft_lattice` for a fixed lattice cell list.

    ``cells`` is the lattice cell list the densities will use -- pass
    ``D_real.cells`` from the multi-k driver (it is fixed across SCF
    iterations, so the cache built from iteration 1 is valid for all).
    Returns an :class:`EwaldJFTLatticeCache` to feed back via the
    ``cache`` argument of :func:`compute_j_ewald_3d_ft_lattice`. The
    contraction is **bit-identical** to the uncached path.
    """
    from ._aopair_ft import ao_pair_fourier_transform_at_cells
    from .aux_basis import _ao_scales_for_rsgdf, rsgdf_dense_g_mesh

    if system.dim != 3:
        raise ValueError(
            "build_j_ewald_3d_ft_lattice_cache: requires dim == 3; got "
            f"dim = {system.dim}."
        )
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    cell_list = list(cells)
    n_cells = len(cell_list)
    nbf = int(basis.nbasis)
    a_lattice = np.asarray(system.lattice, dtype=float)
    cell_volume = float(np.abs(np.linalg.det(a_lattice)))

    G_all = rsgdf_dense_g_mesh(system, float(ke_cutoff))
    G2 = (G_all**2).sum(axis=1)
    nonzero = G2 > 0
    G = G_all[nonzero]
    G2_kept = G2[nonzero]

    if G.shape[0] == 0 or n_cells == 0:
        # No G != 0 modes (or no cells): J(g) == 0. Store an empty mesh so
        # the contraction returns the per-cell zero blocks (matches the
        # uncached early return).
        pair_at_cells = np.zeros((n_cells, nbf, nbf, 0), dtype=np.complex128)
        v_G = np.zeros(0, dtype=float)
        return EwaldJFTLatticeCache(
            pair_at_cells, v_G, 1.0 / cell_volume, n_cells, nbf,
            tuple(tuple(c.index) for c in cell_list),
            _physical_pair_cutoff(opts),
        )

    R_g = np.array([list(c.r_cart) for c in cell_list], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)

    # Per-cell AO-pair FT FT_muν(G; R_g): shape (n_cells, nao, nao, n_G).
    # S_g of it is the Γ (k=0) Bloch AO-pair FT -- the source of the F4
    # Γ<->multi-k bit-match.
    pair_at_cells = ao_pair_fourier_transform_at_cells(basis, G, R_g)
    _mask_physical_pair_ft(pair_at_cells, basis, R_g, opts)
    ao_scales = _ao_scales_for_rsgdf(basis)
    # In place: the tensor is (n_cells, nbf, nbf, n_G) -- the dominant
    # memory of this cache -- and an out-of-place multiply transiently
    # doubles it (measured 2.95 GB -> 5.9 GB peak on c-diamond
    # primitive/sto-3g, 177 cells; 2026-07-09).
    pair_at_cells *= np.outer(ao_scales, ao_scales)[None, :, :, None]

    v_G = 4.0 * np.pi / G2_kept
    return EwaldJFTLatticeCache(
        pair_at_cells, v_G, 1.0 / cell_volume, n_cells, nbf,
        tuple(tuple(c.index) for c in cell_list),
        _physical_pair_cutoff(opts),
    )


def compute_j_ewald_3d_ft_lattice(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real,
    omega: float,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    ke_cutoff: float = 200.0,
    cache: Optional[EwaldJFTLatticeCache] = None,
):
    """Lattice (per-cell) sibling of :func:`compute_j_ewald_3d_ft_gamma`.

    Returns the real-space Hartree J blocks ``J(g)`` -- one per cell in
    ``D_real.cells`` -- of the full periodic Hartree matrix, evaluated in
    reciprocal space with the ``G = 0`` mode omitted. This is the
    **multi-k generalisation** of the Γ analytic-FT J: Bloch-summing the
    blocks, ``J(k) = S_g e^{i k.R_g} J(g)``, gives the Hartree matrix at
    any crystal momentum, and at the Γ-only ``(1,1,1)`` mesh the sum
    bit-matches :func:`compute_j_ewald_3d_ft_gamma` (the F4
    ``test_multi_k_at_gamma_mesh_matches_gamma_driver`` parity gate).

    It shares the identical G-mesh (:func:`rsgdf_dense_g_mesh`), AO-pair
    L-calibration (:func:`_ao_scales_for_rsgdf`) and Coulomb kernel
    (``4pi/G^2``) as the Γ builder, and differs from it only in carrying
    the AO-pair FT *per cell* (``ao_pair_fourier_transform_at_cells``)
    rather than Bloch-summed at ``k = 0``. The Γ equality holds because
    ``S_g FT_muν(G; R_g)`` is exactly the ``k = 0`` Bloch AO-pair FT the Γ
    builder uses, and at ``(1,1,1)`` the reconstructed density blocks
    satisfy ``D_real(g) = D_Γ`` for every cell ``g``.

    ``omega`` is accepted for API compatibility with the historical
    Ewald-split builder; the returned J is independent of it -- the whole
    Hartree lives in the ``G != 0`` reciprocal sum, with no SR/LR split.
    This is what makes the migrated multi-k EWALD_3D path w-invariant
    (the e-e half of the F2 fix), matching the Γ driver.

    Parameters
    ----------
    basis, system
        AO basis and 3D periodic system.
    D_real
        Real-space density as a :class:`LatticeMatrixSet`. The returned
        blocks are aligned to ``D_real.cells`` so the multi-k Fock
        builder can Bloch-sum them directly.
    omega
        Ewald split parameter (accepted, unused; see above).
    lattice_opts
        :class:`LatticeSumOptions`; defaults to the builder defaults.
    ke_cutoff
        Kinetic-energy cutoff (Hartree) for the analytic-FT G-mesh --
        same convention and mesh as :func:`compute_j_ewald_3d_ft_gamma`.

    Returns
    -------
    list[np.ndarray]
        Per-cell ``(n_bf, n_bf)`` real Hartree J blocks (Hartree), one
        per cell in ``D_real.cells``, in the jellium (G=0-dropped) gauge.
    """
    if cache is None:
        cache = build_j_ewald_3d_ft_lattice_cache(
            basis,
            system,
            D_real.cells,
            lattice_opts=lattice_opts,
            ke_cutoff=ke_cutoff,
        )

    cells = list(D_real.cells)
    _check_cache_pair_support(cache, lattice_opts)
    n_cells = len(cells)
    if n_cells != cache.n_cells:
        raise ValueError(
            "compute_j_ewald_3d_ft_lattice: D_real has "
            f"{n_cells} cells but the cache was built for "
            f"{cache.n_cells}; rebuild the cache for this density's cell "
            "list."
        )

    if cache.cell_keys is not None and tuple(tuple(c.index) for c in cells) != cache.cell_keys:
        raise ValueError("compute_j_ewald_3d_ft_lattice: density cell ordering differs from cache")

    # Empty G-mesh (or no cells): J(g) == 0 per cell -- matches the uncached
    # early return.
    if cache.pair_at_cells.shape[-1] == 0 or n_cells == 0:
        nbf = cache.n_bf
        return [np.zeros((nbf, nbf), dtype=float) for _ in range(n_cells)]

    D_blocks = np.array(
        [np.asarray(b, dtype=float) for b in D_real.blocks]
    )  # (n_cells, nao, nao)

    # Density-dependent contraction only -- the cached ``pair_at_cells`` is
    # the per-cell analytic AO-pair Fourier transform FT_muν(G; R_g), Eq. 16
    # of Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119
    # (2017), doi:10.1063/1.4998644 (G = 0-omitted periodic-Coulomb
    # convention, shared by PySCF pbc.df.AFTDF / GDF, around Eq. 17).
    #   rho_D(G) = S_g S_ls D_real(g)_ls FT_ls(G; R_g)
    #   J(g)_muν  = (1/Ω) S_{G!=0} (4pi/G^2) rho_D(G) conj(FT_muν(G; R_g))
    # Bloch-summed at Γ this equals compute_j_ewald_3d_ft_gamma; at a
    # genuine k-mesh it is the correct multi-k Hartree (w-invariant).
    # Validated: bit-matches the Γ driver on H₂/30-bohr at kmesh (1,1,1)
    # (RKS + B3LYP + UKS) to < 1e-10 Ha -- F4 parity gate. The conjugate is
    # taken inline (not stored on the cache) to avoid a persistent second
    # copy of this potentially large tensor.
    rho_G = np.einsum("gmn,gmnk->k", D_blocks, cache.pair_at_cells)
    J_all = cache.inv_cell_volume * np.real(
        np.einsum("k,k,gmnk->gmn", cache.v_G, rho_G, cache.pair_at_cells.conj())
    )
    return [np.ascontiguousarray(J_all[g]) for g in range(n_cells)]


def compute_j_ewald_3d_ft_k_density_to_cells(
    basis: BasisSet,
    system: PeriodicSystem,
    D_k_list: Sequence[np.ndarray],
    k_points: Sequence[np.ndarray],
    weights: Sequence[float],
    output_cells,
    omega: float,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    ke_cutoff: float = 200.0,
    chunk_size: int = 512,
    cell_chunk_size: Optional[int] = None,
):
    """Exact G=0-dropped Hartree J blocks from per-k density matrices.

    This is the dense-reciprocal, unsplit sibling of the BIPOLE analytic
    ``J_SR(erfc) + J_LR(erf)`` composition. It uses the same AO-pair FT
    convention as :func:`compute_j_ewald_3d_ft_lattice`, but builds
    ``rho(G)`` directly from the BvK per-k density representation:

        rho(G) = sum_k w_k sum_mn D_mn(k)
            sum_g exp(-i k.R_g) FT_mn(G; R_g)

    The output is emitted only on ``output_cells`` so callers can keep the
    normal BIPOLE operator support while avoiding the finite real-space
    short-range envelope that caused the pure-RKS mHa Hartree offset.
    ``omega`` is accepted for API compatibility; the total exact J is
    independent of the Ewald split parameter.
    """
    _ = float(omega)
    if system.dim != 3:
        raise ValueError(
            "compute_j_ewald_3d_ft_k_density_to_cells requires dim == 3; "
            f"got dim = {system.dim}."
        )
    if len(D_k_list) != len(k_points) or len(k_points) != len(weights):
        raise ValueError(
            "compute_j_ewald_3d_ft_k_density_to_cells: D_k_list, "
            "k_points, and weights must have identical lengths"
        )
    cell_list = list(output_cells)
    n_cells = len(cell_list)
    nbf = int(basis.nbasis)
    if n_cells == 0:
        return []
    for idx, D_k in enumerate(D_k_list):
        D_arr = np.asarray(D_k)
        if D_arr.shape != (nbf, nbf):
            raise ValueError(
                "compute_j_ewald_3d_ft_k_density_to_cells: D_k_list["
                f"{idx}] has shape {D_arr.shape}, expected {(nbf, nbf)}"
            )
    chunk = max(1, int(chunk_size))
    explicit_cell_chunk = (
        None if cell_chunk_size is None else max(1, int(cell_chunk_size))
    )

    from ._aopair_ft import ao_pair_fourier_transform_at_cells
    from .aux_basis import _ao_scales_for_rsgdf, rsgdf_dense_g_mesh

    a_lattice = np.asarray(system.lattice, dtype=float)
    inv_cell_volume = 1.0 / float(np.abs(np.linalg.det(a_lattice)))
    G_all = rsgdf_dense_g_mesh(system, float(ke_cutoff))
    G2_all = (G_all**2).sum(axis=1)
    nonzero = G2_all > 0
    G_all = G_all[nonzero]
    G2_all = G2_all[nonzero]
    if G_all.shape[0] == 0:
        return [np.zeros((nbf, nbf), dtype=float) for _ in range(n_cells)]

    R_g = np.array([list(c.r_cart) for c in cell_list], dtype=float)
    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    J_all = np.zeros((n_cells, nbf, nbf), dtype=float)

    k_arrs = [np.asarray(k, dtype=float).reshape(3) for k in k_points]
    D_arrs = [np.asarray(D_k, dtype=np.complex128) for D_k in D_k_list]
    weights_arr = [float(w) for w in weights]

    for start in range(0, G_all.shape[0], chunk):
        stop = min(start + chunk, G_all.shape[0])
        G = G_all[start:stop]
        G2 = G2_all[start:stop]
        cell_chunk = (
            _ewald_j_ft_cell_chunk_size(nbf, G.shape[0])
            if explicit_cell_chunk is None
            else explicit_cell_chunk
        )
        active_cell_chunk = min(n_cells, cell_chunk)
        if active_cell_chunk >= n_cells:
            pair_at_cells = ao_pair_fourier_transform_at_cells(basis, G, R_g)
            _mask_physical_pair_ft(pair_at_cells, basis, R_g, lattice_opts)
            pair_at_cells *= pair_scales[None, :, :, None]

            rho_G = np.zeros(G.shape[0], dtype=np.complex128)
            for D_k, k_cart, weight in zip(D_arrs, k_arrs, weights_arr):
                phases = np.exp(-1j * (R_g @ k_cart))
                pair_bloch = np.einsum(
                    "g,gmnk->mnk",
                    phases,
                    pair_at_cells,
                    optimize=True,
                )
                rho_G += weight * np.einsum(
                    "mn,mnk->k",
                    D_k,
                    pair_bloch,
                    optimize=True,
                )

            v_G = 4.0 * np.pi / G2
            J_all += inv_cell_volume * np.real(
                np.einsum(
                    "k,k,gmnk->gmn",
                    v_G,
                    rho_G,
                    pair_at_cells.conj(),
                    optimize=True,
                )
            )
            continue

        rho_G = np.zeros(G.shape[0], dtype=np.complex128)
        for cell_start in range(0, n_cells, active_cell_chunk):
            cell_stop = min(cell_start + active_cell_chunk, n_cells)
            R_batch = R_g[cell_start:cell_stop]
            pair_batch = ao_pair_fourier_transform_at_cells(basis, G, R_batch)
            _mask_physical_pair_ft(pair_batch, basis, R_batch, lattice_opts)
            pair_batch *= pair_scales[None, :, :, None]
            for D_k, k_cart, weight in zip(D_arrs, k_arrs, weights_arr):
                phases = np.exp(-1j * (R_batch @ k_cart))
                pair_bloch = np.einsum(
                    "g,gmnk->mnk",
                    phases,
                    pair_batch,
                    optimize=True,
                )
                rho_G += weight * np.einsum(
                    "mn,mnk->k",
                    D_k,
                    pair_bloch,
                    optimize=True,
                )

        v_G = 4.0 * np.pi / G2
        for cell_start in range(0, n_cells, active_cell_chunk):
            cell_stop = min(cell_start + active_cell_chunk, n_cells)
            R_batch = R_g[cell_start:cell_stop]
            pair_batch = ao_pair_fourier_transform_at_cells(basis, G, R_batch)
            _mask_physical_pair_ft(pair_batch, basis, R_batch, lattice_opts)
            pair_batch *= pair_scales[None, :, :, None]
            J_all[cell_start:cell_stop] += inv_cell_volume * np.real(
                np.einsum(
                    "k,k,gmnk->gmn",
                    v_G,
                    rho_G,
                    pair_batch.conj(),
                    optimize=True,
                )
            )

    return [np.ascontiguousarray(J_all[g]) for g in range(n_cells)]
