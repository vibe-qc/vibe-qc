"""Rigorous 2D (slab) Ewald Hartree ``J`` via an Ewald-split assembly.

The ``CoulombMethod.SLAB_EWALD_2D`` analogue of
:func:`vibeqc.ewald_composed.make_ewald_3d_gamma_j_builder`. Builds the
Γ-point Hartree matrix ``J(D)`` for a system periodic in the plane and finite
along the normal, in the rigorous Parry / de Leeuw-Perram 2D-Ewald gauge -- the
electron-electron partner of the slab ``V_ne`` (``vibeqc.periodic_v_ne_slab``),
so the SCF total assembles bilinearly as
``E = E_nn + Tr[D.V_ne] + 1/2 Tr[D.J]``.

``J`` is built with the **same Ewald split as the slab ``V_ne``** (``1/r =
erf(ar)/r + erfc(ar)/r``); this is what makes the assembled total a-invariant
*and* convergent for all-electron cores. The three pieces, with the screening
parameter ``a`` shared with ``V_ne`` / ``E_nn``::

    J = J_short(erfc, real-space) + J_long(erf, recip g!=0) + J_g0(erf, slab)

* **J_short** -- the sharp short-range cusp ``erfc(a|r-r'|)/|r-r'|``, summed in
  **real space** by ``build_fock_2e_real_space(.., omega=a)`` against the
  homogeneous (full-Bloch, ``D(g)=D`` in every cell) Γ density -- the SAME
  density convention the reciprocal pieces use. libint integrates the cusp
  exactly, so no reciprocal cutoff has to resolve the core (the old
  full-reciprocal build needed ``ke >> 200`` and still under-converged the
  ``F 1s``-type core by ~0.1 Ha -- the v0.15 surface-dipole "bug" that was in
  fact this convergence failure; see ``handovers/HANDOVER_SLAB_EWALD_2D.md``).
* **J_long** -- the smooth long-range ``erf`` tail, a 1D ``G_z`` quadrature of
  the **damped** ``4pi/G^2 . e^{-G^2/4a^2}`` kernel over the in-plane
  ``g != 0`` reciprocal lattice (a 2D slab is a 3D crystal with ``L_z -> inf``,
  so ``S_{G_z} -> (L_z/2pi)∫dG_z`` and ``1/Ω = 1/(A L_z)`` give the
  ``ΔG_z/2piA`` prefactor). The Gaussian damping converges at a small ``|G|``
  independent of the AO core, unlike the undamped kernel.
* **J_g0** -- the ``g = 0`` slab channel against the z-resolved AO-pair density
  ``n_muν(z)``, with the **screened** Parry kernel ``g0(u) = u.erf(au) +
  e^{-a^2u^2}/(a√pi)`` (the ``erf`` partner of ``J_short``'s real-space cusp;
  its ``a -> inf`` limit is the bare ``|z|``). This is the SAME ``g0`` kernel
  ``V_ne``'s ``V_g0`` and the point-charge ``E_nn`` use, so all three blocks
  share one gauge::

    rho_n(z)  = S_ls D_ls n_ls(z)                  (electron number profile)
    J_g0_muν   = -(2pi/A) ∫dz ∫dz' n_muν(z) g0(z - z', a) rho_n(z')

The iteration-invariant reciprocal pieces -- the AO-pair FT on the (g, G_z)
mesh, the damped kernel, the z-profiles ``n_muν(z)`` and the ``g0`` convolution
-- are cached once (:class:`SlabEwald2DJCache`); each ``J(D)`` re-contracts the
density there and re-runs the (screened, so cheap) real-space ``J_short``.

References: Parry, Surf. Sci. **49**, 433 (1975); de Leeuw & Perram, Mol.
Phys. **37**, 1313 (1979). The 3D-FT Hartree it generalises:
Sun-Berkelbach-McClain-Chan, J. Chem. Phys. **147**, 164119 (2017),
doi:10.1063/1.4998644.

Validated (``tests/test_j_slab_ewald_2d.py``): a-invariance of ``½Tr[D.J]`` and
of the assembled bilinear total (the rigorous gate -- a is the split parameter
only); convergence to the same value the undamped reciprocal build approaches
as ``ke -> inf``; the polar (``M_z != 0``) bilinear total vs the
core-converged vacuum-extrapolated 3D-Ewald; cache<->uncached agreement;
symmetry. Un-gates nothing -- the SCF dispatch and C++ Γ drivers still raise
for SLAB_EWALD_2D.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
)
from .periodic_v_ne_slab import (
    _DEFAULT_ALPHA,
    _g0_kernel,
    _inplane_g_mesh,
    _slab_geometry,
    _z_resolved_pair_profiles,
    _z_resolved_pair_profiles_at_cells,
)

__all__ = [
    "SlabEwald2DJCache",
    "SlabEwald2DJLatticeCache",
    "build_j_slab_ewald_2d_gamma_cache",
    "build_j_slab_ewald_2d_lattice_cache",
    "compute_j_slab_ewald_2d_gamma",
    "compute_j_slab_ewald_2d_lattice",
    "make_slab_ewald_2d_gamma_j_builder",
]

# Reciprocal radius of the DAMPED erf long-range Hartree, |G| <= 2a√(-ln tol)
# (the radius at which the e^{-G^2/4a^2} damping reaches ``recip_tol``). Because
# the cusp is taken in real space (J_short), this radius is set by a -- NOT by
# the AO core -- so it stays small (|G| ~ 4 bohr⁻¹ at a=0.4) and core-independent.
_DEFAULT_RECIP_TOL = 1e-13


class SlabEwald2DJCache:
    """Iteration-invariant pieces of the Γ 2D-slab Ewald-split Hartree.

    Built once by :func:`build_j_slab_ewald_2d_gamma_cache`; reused across SCF
    iterations. Holds the damped long-range reciprocal kernel + AO-pair FT, the
    screened ``g0`` z-channel, and the ``alpha`` / cell-list needed to re-run
    the (screened, cheap) real-space ``J_short`` per density.
    """

    __slots__ = (
        "pair_ft",
        "pair_ft_conj",
        "v_G",
        "n_prof",
        "w_g0",
        "nbf",
        "alpha",
        "lat_opts",
        "overlap_template",
    )

    def __init__(self, pair_ft, v_G, n_prof, w_g0, nbf, alpha,
                 lat_opts, overlap_template):
        self.pair_ft = pair_ft          # r̂_muν(g!=0, G_z), (nbf, nbf, n_G)
        self.pair_ft_conj = pair_ft.conj()
        self.v_G = v_G                  # (n_G,) = (ΔG_z/2piA).4pi/G^2.e^{-G^2/4a^2}
        self.n_prof = n_prof            # n_muν(z), (nbf, nbf, n_z)
        self.w_g0 = w_g0                # (n_z, n_z) = -(2pi/A).dz.dz'.g0(z-z',a)
        self.nbf = nbf
        self.alpha = float(alpha)       # shared Ewald split parameter
        self.lat_opts = lat_opts        # cell list for the real-space J_short
        self.overlap_template = overlap_template  # LatticeMatrixSet for D(g)


class SlabEwald2DJLatticeCache:
    """Iteration-invariant pieces of multi-k slab Hartree ``J(g)`` blocks."""

    __slots__ = (
        "pair_at_cells",
        "pair_at_cells_conj",
        "v_G",
        "n_prof_at_cells",
        "w_g0",
        "nbf",
        "alpha",
        "lat_opts",
        "n_cells",
    )

    def __init__(
        self,
        pair_at_cells,
        v_G,
        n_prof_at_cells,
        w_g0,
        nbf,
        alpha,
        lat_opts,
        n_cells,
    ):
        self.pair_at_cells = pair_at_cells
        self.pair_at_cells_conj = pair_at_cells.conj()
        self.v_G = v_G
        self.n_prof_at_cells = n_prof_at_cells
        self.w_g0 = w_g0
        self.nbf = int(nbf)
        self.alpha = float(alpha)
        self.lat_opts = lat_opts
        self.n_cells = int(n_cells)


def build_j_slab_ewald_2d_gamma_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha: float = _DEFAULT_ALPHA,
    recip_tol: float = _DEFAULT_RECIP_TOL,
    n_gz: int = 401,
    z_pad: float = 8.0,
    z_spacing: float = 0.025,
    gz_profile_max: Optional[float] = None,
    profile_tol: float = 1e-7,
) -> SlabEwald2DJCache:
    """Precompute the density-independent pieces of the slab Γ Hartree.

    ``alpha`` is the Ewald split parameter; pass the SAME value the slab
    ``V_ne`` / ``E_nn`` use (the driver shares one ``alpha`` for bare-block
    gauge consistency). ``alpha <= 0`` selects :data:`_DEFAULT_ALPHA`.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from ._vibeqc_core import compute_overlap_lattice, direct_lattice_cells
    from .aux_basis import _ao_scales_for_rsgdf

    if system.dim != 2:
        raise ValueError(
            "build_j_slab_ewald_2d_gamma_cache: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}."
        )
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    if alpha <= 0.0:
        alpha = _DEFAULT_ALPHA
    nbf = int(basis.nbasis)
    area, nhat, b1, b2 = _slab_geometry(system)

    cells = direct_lattice_cells(system, float(opts.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    k_gamma = np.zeros(3)

    # ---- Long-range reciprocal mesh: in-plane g != 0 x 1D G_z, DAMPED kernel.
    # |G| <= 2a√(-ln recip_tol); the e^{-G^2/4a^2} damping (the erf tail) makes
    # this small and AO-core-independent -- the cusp is in J_short (real space).
    g_max = 2.0 * alpha * np.sqrt(-np.log(recip_tol)) + 1.0
    g_list = _inplane_g_mesh(b1, b2, g_max)
    Gz = np.linspace(-g_max, g_max, int(n_gz))
    dGz = float(Gz[1] - Gz[0])
    G_recip = (
        g_list[:, None, :] + Gz[None, :, None] * nhat[None, None, :]
    ).reshape(-1, 3)
    pair_ft = ao_pair_fourier_transform_bloch(basis, G_recip, R_g, k_gamma)
    pair_ft = pair_ft * pair_scales[:, :, None]
    G2 = (G_recip**2).sum(axis=1)
    v_G = (dGz / (2.0 * np.pi * area)) * (4.0 * np.pi / G2) * np.exp(
        -G2 / (4.0 * alpha**2)
    )

    # ---- g = 0 channel: SCREENED Parry g0(z-z',a) against the z-profiles.
    z_I = (
        np.array([list(a.xyz) for a in system.unit_cell], dtype=float) @ nhat
    )
    z_center = 0.5 * (float(z_I.min()) + float(z_I.max()))
    z_half = 0.5 * (float(z_I.max()) - float(z_I.min())) + z_pad
    z_grid, dz, n_prof = _z_resolved_pair_profiles(
        basis,
        system,
        opts,
        nhat=nhat,
        pair_scales=pair_scales,
        R_g=R_g,
        z_center=z_center,
        z_half=z_half,
        gz_profile_max=gz_profile_max,
        profile_tol=profile_tol,
        z_spacing=z_spacing,
        context="build_j_slab_ewald_2d_gamma_cache",
    )
    # w_g0[z, z'] = -(2pi/A).dz.dz'.g0(z-z',a) folds both quadrature weights and
    # the screened Parry kernel, so J_g0_muν = S_z n_muν(z) (w_g0 @ rho_n)[z],
    # rho_n(z') = S_ls D_ls n_ls(z'). g0 is the erf partner of J_short's
    # erfc real-space cusp (its a->inf limit is the bare |z|), shared with V_g0.
    g0_zz = _g0_kernel(z_grid[:, None] - z_grid[None, :], alpha)
    w_g0 = -(2.0 * np.pi / area) * (dz * dz) * g0_zz

    overlap_template = compute_overlap_lattice(basis, system, opts)

    return SlabEwald2DJCache(
        pair_ft, v_G, n_prof, w_g0, nbf, alpha, opts, overlap_template
    )


def build_j_slab_ewald_2d_lattice_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    cells,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha: float = _DEFAULT_ALPHA,
    recip_tol: float = _DEFAULT_RECIP_TOL,
    n_gz: int = 401,
    z_pad: float = 8.0,
    z_spacing: float = 0.025,
    gz_profile_max: Optional[float] = None,
    profile_tol: float = 1e-7,
) -> SlabEwald2DJLatticeCache:
    """Precompute the per-cell multi-k slab Hartree ``J(g)`` machinery."""
    from ._aopair_ft import ao_pair_fourier_transform_at_cells
    from .aux_basis import _ao_scales_for_rsgdf

    if system.dim != 2:
        raise ValueError(
            "build_j_slab_ewald_2d_lattice_cache: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}."
        )
    opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    if alpha <= 0.0:
        alpha = _DEFAULT_ALPHA
    cell_list = list(cells)
    n_cells = len(cell_list)
    nbf = int(basis.nbasis)
    area, nhat, b1, b2 = _slab_geometry(system)
    R_g = np.array([list(c.r_cart) for c in cell_list], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_scales = np.outer(ao_scales, ao_scales)

    g_max = 2.0 * float(alpha) * np.sqrt(-np.log(recip_tol)) + 1.0
    g_list = _inplane_g_mesh(b1, b2, g_max)
    Gz = np.linspace(-g_max, g_max, int(n_gz))
    dGz = float(Gz[1] - Gz[0])
    if g_list.shape[0] == 0:
        pair_at_cells = np.zeros((n_cells, nbf, nbf, 0), dtype=np.complex128)
        v_G = np.zeros(0, dtype=float)
    else:
        G_recip = (
            g_list[:, None, :] + Gz[None, :, None] * nhat[None, None, :]
        ).reshape(-1, 3)
        pair_at_cells = ao_pair_fourier_transform_at_cells(basis, G_recip, R_g)
        pair_at_cells = pair_at_cells * pair_scales[None, :, :, None]
        G2 = (G_recip**2).sum(axis=1)
        v_G = (dGz / (2.0 * np.pi * area)) * (4.0 * np.pi / G2) * np.exp(
            -G2 / (4.0 * float(alpha) ** 2)
        )

    z_I = (
        np.array([list(a.xyz) for a in system.unit_cell], dtype=float) @ nhat
    )
    z_center = 0.5 * (float(z_I.min()) + float(z_I.max()))
    z_half = 0.5 * (float(z_I.max()) - float(z_I.min())) + z_pad
    z_grid, dz, n_prof_at_cells = _z_resolved_pair_profiles_at_cells(
        basis,
        system,
        opts,
        nhat=nhat,
        pair_scales=pair_scales,
        R_g=R_g,
        z_center=z_center,
        z_half=z_half,
        gz_profile_max=gz_profile_max,
        profile_tol=profile_tol,
        z_spacing=z_spacing,
        context="build_j_slab_ewald_2d_lattice_cache",
    )
    g0_zz = _g0_kernel(z_grid[:, None] - z_grid[None, :], float(alpha))
    w_g0 = -(2.0 * np.pi / area) * (dz * dz) * g0_zz

    return SlabEwald2DJLatticeCache(
        pair_at_cells,
        v_G,
        n_prof_at_cells,
        w_g0,
        nbf,
        float(alpha),
        opts,
        n_cells,
    )


def compute_j_slab_ewald_2d_lattice(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha: float = _DEFAULT_ALPHA,
    cache: Optional[SlabEwald2DJLatticeCache] = None,
    **cache_kwargs,
) -> list[np.ndarray]:
    """Return per-cell slab Hartree ``J(g)`` blocks for a multi-k density."""
    from ._vibeqc_core import build_fock_2e_real_space

    if cache is None:
        cache = build_j_slab_ewald_2d_lattice_cache(
            basis,
            system,
            D_real.cells,
            lattice_opts=lattice_opts,
            alpha=alpha,
            **cache_kwargs,
        )
    cells = list(D_real.cells)
    n_cells = len(cells)
    if n_cells != cache.n_cells:
        raise ValueError(
            "compute_j_slab_ewald_2d_lattice: D_real has "
            f"{n_cells} cells but the cache was built for {cache.n_cells}; "
            "rebuild the cache for this density cell list."
        )
    if n_cells == 0:
        return []

    D_blocks = np.array(
        [np.asarray(b, dtype=float) for b in D_real.blocks]
    )
    J_all = np.zeros((n_cells, cache.nbf, cache.nbf), dtype=float)

    if cache.v_G.size:
        rho_G = np.einsum(
            "gmn,gmnk->k", D_blocks, cache.pair_at_cells, optimize=True
        )
        J_all += np.real(
            np.einsum(
                "k,k,gmnk->gmn",
                cache.v_G,
                rho_G,
                cache.pair_at_cells_conj,
                optimize=True,
            )
        )

    rho_n = np.einsum(
        "gls,glsz->z", D_blocks, cache.n_prof_at_cells, optimize=True
    )
    J_all += np.einsum(
        "gmnz,z->gmn",
        cache.n_prof_at_cells,
        cache.w_g0 @ rho_n,
        optimize=True,
    )

    j_sr_lat = build_fock_2e_real_space(
        basis, system, cache.lat_opts, D_real, 0.0, cache.alpha
    )
    for g in range(n_cells):
        J_all[g] += np.asarray(j_sr_lat.blocks[g], dtype=float)
    return [np.ascontiguousarray(J_all[g]) for g in range(n_cells)]


def compute_j_slab_ewald_2d_gamma(
    basis: BasisSet,
    system: PeriodicSystem,
    D: np.ndarray,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha: float = _DEFAULT_ALPHA,
    cache: Optional[SlabEwald2DJCache] = None,
    **cache_kwargs,
) -> np.ndarray:
    """Γ-point 2D-slab Hartree ``J(D)`` via the Ewald-split assembly.

    ``J = J_short(erfc, real) + J_long(erf, damped recip) + J_g0(erf, slab)``,
    all sharing the split parameter ``alpha`` (pass the slab ``V_ne`` /
    ``E_nn`` value). Pass ``cache`` (from
    :func:`build_j_slab_ewald_2d_gamma_cache`) to skip rebuilding the
    iteration-invariant reciprocal / z pieces -- the SCF-loop fast path.
    """
    from ._vibeqc_core import bloch_sum, build_fock_2e_real_space
    from .periodic_gradient import _gamma_density_lattice_set

    if cache is None:
        cache = build_j_slab_ewald_2d_gamma_cache(
            basis,
            system,
            lattice_opts=lattice_opts,
            alpha=alpha,
            **cache_kwargs,
        )
    D_arr = np.asarray(D, dtype=float)

    # ---- J_long, reciprocal g != 0 (DAMPED erf tail): rho_D(G) = S_ls D_ls r̂_ls(G);
    #   J_long_muν = Re S_G v_G rho_D(G) conj(r̂_muν(G)).
    rho_G = np.einsum("mn,mng->g", D_arr, cache.pair_ft)
    J = np.real(
        np.einsum("g,g,mng->mn", cache.v_G, rho_G, cache.pair_ft_conj)
    )
    # ---- J_g0, g = 0 slab (screened): rho_n(z) = S_ls D_ls n_ls(z);
    #   J_g0_muν = S_z n_muν(z) (w_g0 @ rho_n)[z].
    rho_n = np.einsum("ls,lsz->z", D_arr, cache.n_prof)
    J += np.einsum("mnz,z->mn", cache.n_prof, cache.w_g0 @ rho_n)

    # ---- J_short, erfc(a) cusp in REAL space against the homogeneous Γ density
    # D(g)=D (full-Bloch convention; the SAME density the reciprocal pieces
    # carry). erfc screening keeps the cell sum short, and libint integrates the
    # core cusp exactly -- the reason the assembled J converges where the
    # undamped full-reciprocal build did not. ``exchange_scale=0`` -> pure J.
    Dset = _gamma_density_lattice_set(
        cache.overlap_template, D_arr, homogeneous=True
    )
    j_sr_lat = build_fock_2e_real_space(
        basis, system, cache.lat_opts, Dset, 0.0, cache.alpha
    )
    J = J + np.real(bloch_sum(j_sr_lat, np.zeros(3)))

    J = 0.5 * (J + J.T)
    return J


def make_slab_ewald_2d_gamma_j_builder(
    basis: BasisSet,
    system: PeriodicSystem,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    alpha: float = _DEFAULT_ALPHA,
    **cache_kwargs,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a cached ``J(D)`` closure for the SCF loop.

    Builds the iteration-invariant :class:`SlabEwald2DJCache` once; each
    returned ``J(D)`` re-contracts the cached reciprocal / z pieces and re-runs
    the screened real-space ``J_short``. The driver-facing entry point (the
    SLAB_EWALD_2D analogue of
    :func:`vibeqc.ewald_composed.make_ewald_3d_gamma_j_builder`). Pass the
    ``alpha`` shared with the slab ``V_ne`` / ``E_nn``.
    """
    cache = build_j_slab_ewald_2d_gamma_cache(
        basis,
        system,
        lattice_opts=lattice_opts,
        alpha=alpha,
        **cache_kwargs,
    )

    def _build_j(D: np.ndarray) -> np.ndarray:
        return compute_j_slab_ewald_2d_gamma(
            basis, system, D, lattice_opts=lattice_opts, cache=cache
        )

    return _build_j
