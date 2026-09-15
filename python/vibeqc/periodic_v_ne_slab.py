"""Rigorous 2D (slab) Ewald electron-nuclear attraction ``V_ne``.

The ``CoulombMethod.SLAB_EWALD_2D`` analogue of
:func:`vibeqc.periodic_v_ne.compute_v_ne_ewald_3d_ft_gamma`. Builds the
Γ-point ``V_ne`` matrix for a system periodic in the plane (lattice columns
0, 1) and finite along the normal, in the rigorous Parry / de Leeuw-Perram
2D-Ewald gauge -- the *bare* (non-neutral nuclei, no neutralising sheet)
block that the SCF total assembles bilinearly,
``E = E_nn + Tr[D.V_ne] + 1/2 Tr[D.J]`` (handovers/HANDOVER_SLAB_EWALD_2D.md, "Option 1").

The split mirrors the 3D Ewald-FT path::

    V_ne = V_short + V_long(g!=0) + V_g0(slab)

* ``V_short`` -- libint erfc-screened nuclear attraction
  (:func:`compute_nuclear_erfc_lattice`, dim-aware; the sharp 1/r cusp,
  exact, no mesh).  Identical to the 3D path.

* ``V_long`` (in-plane ``g != 0``) -- the key 2D kernel.  A 2D slab is a 3D
  crystal with ``L_z -> inf``, so the discrete ``G_z`` reciprocal sum becomes a
  **continuous 1D integral with the same 3D Coulomb kernel** ``4pi/G^2``::

      V_long_muν = (ΔG_z / 2piA) . Re S_{g!=0} S_{G_z}
                    v_long(g, G_z) . conj(r̂_muν(g, G_z))
      v_long(G) = -S_I Z_I (4pi/G^2) e^{-G^2/4a^2} e^{-iG.R_I},  G = (g, G_z)

  i.e. the **3D V_ne FT formula verbatim** (same
  :func:`ao_pair_fourier_transform_bloch`, same ``pair_scales`` calibration,
  same damped Coulomb kernel) with the dense 3D G-mesh replaced by
  {2D in-plane reciprocal lattice ``g != 0``} x {dense 1D ``G_z`` quadrature}
  and ``1/Ω -> ΔG_z/(2piA)``.  Reciprocal-space => preserves the in-plane
  crystal point group exactly and is **independent of the in-plane cell
  size** -- unlike a molecular-grid quadrature, which cannot integrate the
  Γ-Bloch AO pair for a small cell (graphene-scale) because the periodic
  images fall outside the atom-centred grid.

* ``V_g0`` (the ``g = 0`` slab term, no 3D analogue) -- the ``g = 0`` channel
  has a ``1/G_z^2`` divergence at ``G_z = 0`` and is **not** the ``G_z``
  integral above; it is the analytic Parry slab term evaluated against the
  *z-resolved* AO-pair density
  ``n_muν(z) = (1/2pi) ∫ dG_z r̂_muν(0, G_z) e^{iG_z z}`` (the inverse z-FT of the
  ``g = 0`` AO-pair FT) by a 1D z-quadrature::

      V_g0_muν = (2pi/A) S_I Z_I ∫ dz n_muν(z) . g0(z - z_I),
      g0(u)   = u.erf(au) + e^{-a^2u^2}/(a√pi)

References: Parry, Surf. Sci. **49**, 433 (1975); de Leeuw & Perram, Mol.
Phys. **37**, 1313 (1979).  The 3D-FT V_ne kernel it generalises:
Sun-Berkelbach-McClain-Chan, J. Chem. Phys. **147**, 164119 (2017),
doi:10.1063/1.4998644.

The total ``V_ne`` is **a-invariant** (the rigorous Ewald-split correctness
gate): ``V_short``, ``V_long`` and ``V_g0`` each depend strongly on a; their
sum does not.  Validated to ~1e-9 in ``tests/test_v_ne_slab_ewald_2d.py``.

.. note::

   **Polar slabs (M_z != 0) are handled correctly.** ``V_g0`` is contracted
   against the z-resolved AO-pair density ``n_muν(z)`` from the ``g = 0``
   AO-pair FT (:func:`_z_resolved_pair_profiles`), which is the planar density
   of the **full Γ-Bloch** AO pair: its monopole is ``S`` (to ~1e-12) and its
   first moment is the **Bloch-summed** z-dipole ``Σ_g <mu,0|z|ν,R_g>`` (to
   ~1e-11 vs ``compute_multipole_moments_lattice``), *not* the home-cell
   ``<mu|z|ν>``. The two differ by O(7e-3) for diffuse functions at small
   in-plane spacing -- the physical cross-cell image dipole, which is exactly
   what the slab g=0 term needs. (An earlier handover mis-read that difference
   as a "surface-dipole bug"; the real ~0.1 Ha polar-slab error was the
   undamped-J core under-convergence, fixed by the Ewald split in
   :mod:`vibeqc.ewald_composed_slab`. See ``handovers/HANDOVER_SLAB_EWALD_2D.md``.)

The Gamma-only and multi-k RHF/RKS/UKS slab SCF paths consume this kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    EwaldOptions,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    bloch_sum,
    compute_nuclear_erfc_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
)

__all__ = [
    "SlabVneDecomposition",
    "SlabVneKCache",
    "build_v_ne_slab_ewald_2d_k_cache",
    "compute_v_ne_slab_ewald_2d_gamma",
    "compute_v_ne_slab_ewald_2d_k_matrix",
    "compute_v_ne_slab_ewald_2d_lattice",
]

# Default Ewald screening parameter for the 2D slab split. The result is
# a-invariant; a only balances real- (libint V_short) vs reciprocal-space
# (the g-/G_z-mesh) work. 0.4 keeps V_short converged inside a ~15-bohr
# cell list and the G_z mesh small. The SCF driver (Increment 4) MUST share
# one a across V_ne, J and E_nn (bare-block gauge consistency).
_DEFAULT_ALPHA = 0.4


@dataclass
class SlabVneDecomposition:
    """Per-component Γ-point ``V_ne`` for ``SLAB_EWALD_2D``.

    ``v_total = v_short + v_long + v_g0``. The components are exposed
    separately for the downstream BIPOLE dim=2 analytic surface gradient
    (Gap 2 in ``handovers/HANDOVER_SURFACE_REACTIONS.md``), which consumes the
    short / long / g0 decomposition term-by-term.
    """

    v_total: np.ndarray
    v_short: np.ndarray
    v_long: np.ndarray
    v_g0: np.ndarray
    alpha: float


@dataclass
class SlabVneKCache:
    """Density-independent pieces for slab ``V_ne(k)`` matrices."""

    v_short_lat: LatticeMatrixSet
    pair_at_cells: np.ndarray
    v_long_G: np.ndarray
    n_prof_at_cells: np.ndarray
    z_grid: np.ndarray
    dz: float
    nuclei_Z: np.ndarray
    z_I: np.ndarray
    g0_prefactor: float
    alpha: float


def _slab_geometry(system: PeriodicSystem):
    """In-plane area ``A``, slab normal ``n̂``, 2D reciprocal vectors."""
    lat = np.asarray(system.lattice, dtype=float)
    a1, a2 = lat[:, 0], lat[:, 1]
    cross = np.cross(a1, a2)
    area = float(np.linalg.norm(cross))
    if area < 1e-14:
        raise ValueError(
            "compute_v_ne_slab_ewald_2d_gamma: degenerate in-plane lattice "
            "(lattice columns 0 and 1 are collinear)."
        )
    nhat = cross / area
    # b_i . a_j = 2pi d_ij in-plane; b_i . n̂ = 0 (so g . n̂ = 0 for every g).
    b1 = 2.0 * np.pi * np.cross(a2, nhat) / area
    b2 = 2.0 * np.pi * np.cross(nhat, a1) / area
    return area, nhat, b1, b2


def _inplane_g_mesh(b1, b2, g_max):
    """2D in-plane reciprocal-lattice vectors with ``0 < |g| <= g_max``."""
    bmin = min(np.linalg.norm(b1), np.linalg.norm(b2))
    n = int(np.ceil(g_max / bmin)) + 1
    out = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if i == 0 and j == 0:
                continue
            g = i * b1 + j * b2
            if np.dot(g, g) <= g_max * g_max:
                out.append(g)
    return np.asarray(out, dtype=float).reshape(-1, 3)


def _g0_kernel(u, alpha):
    """Parry slab ``g = 0`` kernel ``g0(u) = u.erf(au) + e^{-a^2u^2}/(a√pi)``."""
    from scipy.special import erf

    return u * erf(alpha * u) + np.exp(-((alpha * u) ** 2)) / (
        alpha * np.sqrt(np.pi)
    )


def _g0_profile_meshes(
    basis,
    *,
    z_center,
    z_half,
    gz_profile_max=None,
    profile_tol=1e-7,
    z_spacing=0.025,
):
    """``G_z``-profile and z-grid meshes for the ``g = 0`` slab term.

    One mesh policy shared by the value builders
    (:func:`_z_resolved_pair_profiles`,
    :func:`_z_resolved_pair_profiles_at_cells`) and the rung-3 analytic
    gradient (:mod:`vibeqc.periodic_v_ne_slab_gradient`), so the
    derivative differentiates exactly the quadrature the SCF evaluated.

    Returns ``(Gz_p, dGz_p, z_grid, dz)``.
    """
    if gz_profile_max is None:
        a_max = max(
            float(np.max(np.asarray(sh.exponents))) for sh in basis.shells()
        )
        gz_profile_max = max(
            14.0, np.sqrt(8.0 * a_max * (-np.log(profile_tol))) + 2.0
        )
    # Nyquist: dG_z <= pi / max|z| so the z-DFT does not alias on the z-grid.
    dgz_nyq = 0.9 * np.pi / (abs(z_center) + z_half + 1.0)
    n_prof_pts = 2 * int(np.ceil(gz_profile_max / dgz_nyq)) + 1
    Gz_p = np.linspace(-gz_profile_max, gz_profile_max, n_prof_pts)
    dGz_p = float(Gz_p[1] - Gz_p[0])
    n_z = int(np.ceil(2.0 * z_half / z_spacing)) + 1
    z_grid = np.linspace(z_center - z_half, z_center + z_half, n_z)
    dz = float(z_grid[1] - z_grid[0])
    return Gz_p, dGz_p, z_grid, dz


def _z_resolved_pair_profiles(
    basis,
    system,
    lat_opts,
    *,
    nhat,
    pair_scales,
    R_g,
    z_center,
    z_half,
    gz_profile_max=None,
    profile_tol=1e-7,
    z_spacing=0.025,
    context="slab Ewald 2D",
):
    """z-resolved Γ-Bloch AO-pair density ``n_muν(z)`` on a z-grid.

    ``n_muν(z) = (1/2pi) ∫ dG_z r̂_muν(0, G_z) e^{iG_z z}`` -- the inverse z-FT of
    the ``g = 0`` AO-pair FT. This UNDAMPED FT must reach the AO-pair density
    bandwidth (~``√(8 a_max)`` for the tightest primitive exponent ``a_max``),
    so ``gz_profile_max`` defaults to ``√(8 a_max ln(1/profile_tol))``. The
    ``∫ n dz = S`` identity is asserted as a self-check (raises if the z-grid
    or G_z mesh is under-resolved). Shared by the slab ``V_ne`` (``V_g0``) and
    ``J`` (``J_g0``) builders.

    Returns ``(z_grid, dz, n_prof)`` with ``n_prof`` of shape
    ``(nbf, nbf, n_z)``.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch

    k_gamma = np.zeros(3)
    Gz_p, dGz_p, z_grid, dz = _g0_profile_meshes(
        basis,
        z_center=z_center,
        z_half=z_half,
        gz_profile_max=gz_profile_max,
        profile_tol=profile_tol,
        z_spacing=z_spacing,
    )
    G0 = Gz_p[:, None] * nhat[None, :]
    ft0 = ao_pair_fourier_transform_bloch(basis, G0, R_g, k_gamma)
    ft0 = ft0 * pair_scales[:, :, None]  # r̂_muν(0, G_z), (nbf, nbf, n_Gz)
    phase_z = np.exp(1j * np.outer(Gz_p, z_grid))  # (n_Gz, n_z)
    n_prof = (dGz_p / (2.0 * np.pi)) * np.real(
        np.einsum("mnk,kz->mnz", ft0, phase_z)
    )  # n_muν(z_grid), (nbf, nbf, n_z)

    S = np.real(
        bloch_sum(compute_overlap_lattice(basis, system, lat_opts), k_gamma)
    )
    S = 0.5 * (S + S.T)
    s_err = float(np.max(np.abs(np.einsum("mnz->mn", n_prof) * dz - S)))
    s_ref = max(float(np.max(np.abs(S))), 1.0)
    if s_err > 1e-4 * s_ref:
        raise RuntimeError(
            f"{context}: z-resolved AO-pair density fails ∫n dz = S by "
            f"{s_err:.2e} (rel {s_err / s_ref:.2e}); the z-grid / G_z-profile "
            "mesh is under-resolved. Increase z_pad / decrease z_spacing / "
            "raise gz_profile_max."
        )
    return z_grid, dz, n_prof


def _z_resolved_pair_profiles_at_cells(
    basis,
    system,
    lat_opts,
    *,
    nhat,
    pair_scales,
    R_g,
    z_center,
    z_half,
    gz_profile_max=None,
    profile_tol=1e-7,
    z_spacing=0.025,
    context="slab Ewald 2D",
):
    """Per-cell sibling of :func:`_z_resolved_pair_profiles`.

    Returns ``n_muν(g; z)`` whose Γ Bloch sum equals the z-resolved profile used
    by the Γ slab builders. Multi-k one-electron and Hartree slab terms Bloch
    sum these per-cell profiles at the requested k-point.
    """
    from ._aopair_ft import ao_pair_fourier_transform_at_cells

    Gz_p, dGz_p, z_grid, dz = _g0_profile_meshes(
        basis,
        z_center=z_center,
        z_half=z_half,
        gz_profile_max=gz_profile_max,
        profile_tol=profile_tol,
        z_spacing=z_spacing,
    )
    G0 = Gz_p[:, None] * nhat[None, :]
    ft0 = ao_pair_fourier_transform_at_cells(basis, G0, R_g)
    ft0 = ft0 * pair_scales[None, :, :, None]
    phase_z = np.exp(1j * np.outer(Gz_p, z_grid))
    n_prof = (dGz_p / (2.0 * np.pi)) * np.real(
        np.einsum("gmnk,kz->gmnz", ft0, phase_z)
    )

    S = np.real(
        bloch_sum(compute_overlap_lattice(basis, system, lat_opts), np.zeros(3))
    )
    S = 0.5 * (S + S.T)
    s_err = float(
        np.max(np.abs(np.einsum("gmnz->mn", n_prof) * dz - S))
    )
    s_ref = max(float(np.max(np.abs(S))), 1.0)
    if s_err > 1e-4 * s_ref:
        raise RuntimeError(
            f"{context}: per-cell z-resolved AO-pair density fails "
            f"Σ_g∫n(g,z) dz = S(Γ) by {s_err:.2e} "
            f"(rel {s_err / s_ref:.2e}); the z-grid / G_z-profile mesh is "
            "under-resolved. Increase z_pad / decrease z_spacing / raise "
            "gz_profile_max."
        )
    return z_grid, dz, n_prof


def build_v_ne_slab_ewald_2d_k_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    alpha: float = _DEFAULT_ALPHA,
    recip_tol: float = 1e-13,
    n_gz: int = 401,
    gz_profile_max: Optional[float] = None,
    profile_tol: float = 1e-7,
    z_pad: float = 8.0,
    z_spacing: float = 0.025,
) -> SlabVneKCache:
    """Precompute the per-cell pieces needed to evaluate slab ``V_ne(k)``."""
    from ._aopair_ft import ao_pair_fourier_transform_at_cells
    from .aux_basis import _ao_scales_for_rsgdf

    if system.dim != 2:
        raise ValueError(
            "build_v_ne_slab_ewald_2d_k_cache: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}."
        )
    alpha = float(alpha)
    if alpha <= 0.0:
        alpha = _DEFAULT_ALPHA
    area, nhat, b1, b2 = _slab_geometry(system)
    nuclei_Z = np.array([a.Z for a in system.unit_cell], dtype=float)
    nuclei_R = np.array([list(a.xyz) for a in system.unit_cell], dtype=float)
    z_I = nuclei_R @ nhat

    v_short_lat = compute_nuclear_erfc_lattice(basis, system, alpha, lat_opts)
    cells = list(v_short_lat.cells)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    n_cells = len(cells)
    nbf = int(basis.nbasis)
    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_scales = np.outer(ao_scales, ao_scales)

    g_max = 2.0 * alpha * np.sqrt(-np.log(recip_tol)) + 1.0
    g_list = _inplane_g_mesh(b1, b2, g_max)
    Gz = np.linspace(-g_max, g_max, int(n_gz))
    dGz = float(Gz[1] - Gz[0])
    if g_list.shape[0] == 0:
        pair_at_cells = np.zeros((n_cells, nbf, nbf, 0), dtype=np.complex128)
        v_long_G = np.zeros(0, dtype=np.complex128)
    else:
        G_recip = (
            g_list[:, None, :] + Gz[None, :, None] * nhat[None, None, :]
        ).reshape(-1, 3)
        pair_at_cells = ao_pair_fourier_transform_at_cells(basis, G_recip, R_g)
        pair_at_cells = pair_at_cells * pair_scales[None, :, :, None]
        G2 = (G_recip**2).sum(axis=1)
        damp = (4.0 * np.pi / G2) * np.exp(-G2 / (4.0 * alpha**2))
        phase = np.exp(-1j * (G_recip @ nuclei_R.T))
        v_long_G = (dGz / (2.0 * np.pi * area)) * (
            -(damp * (phase @ nuclei_Z))
        )

    z_center = 0.5 * (float(z_I.min()) + float(z_I.max()))
    z_half = 0.5 * (float(z_I.max()) - float(z_I.min())) + z_pad
    z_grid, dz, n_prof_at_cells = _z_resolved_pair_profiles_at_cells(
        basis,
        system,
        lat_opts,
        nhat=nhat,
        pair_scales=pair_scales,
        R_g=R_g,
        z_center=z_center,
        z_half=z_half,
        gz_profile_max=gz_profile_max,
        profile_tol=profile_tol,
        z_spacing=z_spacing,
        context="build_v_ne_slab_ewald_2d_k_cache",
    )

    return SlabVneKCache(
        v_short_lat=v_short_lat,
        pair_at_cells=pair_at_cells,
        v_long_G=v_long_G,
        n_prof_at_cells=n_prof_at_cells,
        z_grid=z_grid,
        dz=dz,
        nuclei_Z=nuclei_Z,
        z_I=z_I,
        g0_prefactor=2.0 * np.pi / area,
        alpha=float(alpha),
    )


def compute_v_ne_slab_ewald_2d_k_matrix(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    k_cart: np.ndarray,
    *,
    alpha: float = _DEFAULT_ALPHA,
    cache: Optional[SlabVneKCache] = None,
    **cache_kwargs,
) -> np.ndarray:
    """Return the rigorous slab ``V_ne(k)`` matrix for one Bloch k-point."""
    if cache is None:
        cache = build_v_ne_slab_ewald_2d_k_cache(
            basis, system, lat_opts, alpha=alpha, **cache_kwargs
        )
    k = np.asarray(k_cart, dtype=float).reshape(3)
    V_short = np.asarray(bloch_sum(cache.v_short_lat, k), dtype=complex)

    cells = list(cache.v_short_lat.cells)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    phases = np.exp(1j * (R_g @ k))

    if cache.v_long_G.size:
        # The g != 0 long-range block at crystal momentum k. With the module's
        # FT convention r̂_muν(g; G) = ∫ dr chi_mu(r) chi_ν(r - R_g) e^{-iG.r}
        # (:mod:`vibeqc._aopair_ft`) and the standard +ik Bloch sum,
        #
        #   V_long_muν(k) = S_g e^{+ik.R_g} ∫ dr chi_mu(r) chi_ν(r-R_g) V_long(r)
        #                 = S_G ṽ(G) S_g e^{+ik.R_g} r̂_muν(g; -G)
        #                 = S_G ṽ(G) S_g e^{+ik.R_g} conj(r̂_muν(g; G)),
        #
        # because V_long(r) = S_G ṽ(G) e^{+iG.r} carries e^{+iG.r} while the
        # FT convention is e^{-iG.r}, and the real-space AO pair is real so
        # r̂(g; -G) = conj(r̂(g; G)). The conjugation therefore acts on the
        # AO-pair FT ONLY -- it must NOT be applied to the assembled Bloch
        # sum, which would carry e^{-ik.R_g} and evaluate V_long(-k) instead
        # of V_long(k). The reciprocal slab kernel ṽ(G) itself was, and stays,
        # the Parry form -- Parry, Surf. Sci. 49, 433 (1975), sec. 3 (the
        # C -> inf limit that turns the discrete k_z sum into the 1D integral,
        # plus the separate g = 0 term); de Leeuw & Perram, Mol. Phys. 37,
        # 1313 (1979), sec. 3. What this block violated is the Bloch-sum
        # convention F(k) = S_g e^{+ik.R_g} F(g) that the rest of the code
        # uses (Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119
        # (2017), doi:10.1063/1.4998644, eq. 17), by conjugating the phase
        # together with the quantity being summed.
        #
        # Evaluated as conj(S_g e^{-ik.R_g} r̂(g; G)) so the conjugation runs
        # on the small (nbf, nbf, n_G) Bloch sum rather than on the whole
        # (n_cells, nbf, nbf, n_G) per-cell tensor.
        pair_bloch_conj = np.einsum(
            "g,gmnk->mnk", phases.conj(), cache.pair_at_cells, optimize=True
        ).conj()
        V_long = np.einsum(
            "k,mnk->mn", cache.v_long_G, pair_bloch_conj, optimize=True
        )
    else:
        V_long = np.zeros_like(V_short)

    n_prof_k = np.einsum(
        "g,gmnz->mnz", phases, cache.n_prof_at_cells, optimize=True
    )
    V_g0 = np.zeros_like(V_short)
    for Z, z_I in zip(cache.nuclei_Z, cache.z_I):
        ker = _g0_kernel(cache.z_grid - z_I, cache.alpha) * cache.dz
        V_g0 += cache.g0_prefactor * float(Z) * np.einsum(
            "mnz,z->mn", n_prof_k, ker, optimize=True
        )

    V = V_short + V_long + V_g0
    return 0.5 * (V + V.conj().T)


def compute_v_ne_slab_ewald_2d_gamma(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    ewald_options: Optional[EwaldOptions] = None,
    recip_tol: float = 1e-13,
    n_gz: int = 401,
    gz_profile_max: Optional[float] = None,
    profile_tol: float = 1e-7,
    z_pad: float = 8.0,
    z_spacing: float = 0.025,
) -> SlabVneDecomposition:
    """Γ-point ``V_ne`` for ``SLAB_EWALD_2D`` via the z-resolved Ewald FT.

    Parameters
    ----------
    basis, system, lat_opts
        AO basis, periodic system (``system.dim`` must be 2), and lattice-sum
        options. ``lat_opts.cutoff_bohr`` fixes the Bloch-sum cell list (used
        for both the libint ``V_short`` and the AO-pair FT);
        ``lat_opts.nuclear_cutoff_bohr`` the default ``V_short`` real cutoff.
    ewald_options
        Optional :class:`EwaldOptions`. ``alpha <= 0`` selects
        :data:`_DEFAULT_ALPHA`. The SCF driver passes an explicit shared a so
        ``V_ne`` / ``J`` / ``E_nn`` use one bare-block gauge.
    recip_tol
        Reciprocal sum is carried to ``|G| = 2a√(-ln recip_tol)`` (the radius
        at which the ``e^{-G^2/4a^2}`` damping reaches ``recip_tol``).
    n_gz
        Number of ``G_z`` quadrature points for the ``g != 0`` long-range sum
        (trapezoidal; the integrand is Gaussian-damped so it converges fast).
    gz_profile_max, profile_tol, z_pad, z_spacing
        ``G_z`` range and z-grid for the ``g = 0`` z-resolved AO-pair density
        ``n_muν(z) = (1/2pi)∫dG_z r̂(0,G_z) e^{iG_z z}``. ``gz_profile_max``
        defaults (``None``) to the bandwidth of the tightest AO-pair density
        -- ``√(8 a_max ln(1/profile_tol))`` for the largest primitive exponent
        ``a_max`` (an undamped FT, so it must reach the core bandwidth, unlike
        the damped ``g != 0`` sum). ``z_pad`` extends the z-grid beyond the
        nuclear slab; ``z_spacing`` sets its resolution. The ``∫ n dz = S``
        identity is asserted as a self-check (raises if under-resolved).
        For very tight all-electron cores this mesh grows; an analytic
        ``n_muν(z)`` (in-plane Gaussian integral) would be the future-proof
        replacement.

    Returns
    -------
    SlabVneDecomposition
        ``v_total`` plus the ``v_short`` / ``v_long`` / ``v_g0`` components,
        each a real symmetric ``(nbf, nbf)`` Γ matrix in libint AO ordering.
    """
    from ._aopair_ft import ao_pair_fourier_transform_bloch
    from .aux_basis import _ao_scales_for_rsgdf

    if system.dim != 2:
        raise ValueError(
            "compute_v_ne_slab_ewald_2d_gamma: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}."
        )

    if ewald_options is None:
        ewald_options = EwaldOptions()
        ewald_options.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr
    alpha = ewald_options.alpha
    if alpha <= 0.0:
        alpha = _DEFAULT_ALPHA

    area, nhat, b1, b2 = _slab_geometry(system)

    nuclei_Z = np.array([a.Z for a in system.unit_cell], dtype=float)
    nuclei_R = np.array([list(a.xyz) for a in system.unit_cell], dtype=float)
    z_I = nuclei_R @ nhat

    # ---- Bloch-sum cell list shared by V_short (libint) and the AO-pair FT.
    cells = direct_lattice_cells(system, lat_opts.cutoff_bohr)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    k_gamma = np.zeros(3)

    # ---- V_short via libint analytical erfc-screened nuclear attraction ----
    # (already electron-signed: -Z erfc(ar)/r integrated against the AOs).
    V_short_lat = compute_nuclear_erfc_lattice(basis, system, alpha, lat_opts)
    V_short = np.real(bloch_sum(V_short_lat, k_gamma))
    V_short = 0.5 * (V_short + V_short.T)

    # ---- V_long, in-plane g != 0: 1D G_z quadrature of the 3D Coulomb kernel.
    # The 2D slab is a 3D crystal with L_z -> inf: S_{G_z} -> (L_z/2pi) ∫ dG_z and
    # 1/Ω = 1/(A L_z) cancel L_z, giving the prefactor ΔG_z/(2piA). Excludes
    # the g = 0 line (its G_z=0 divergence is the slab term below).
    g_max = 2.0 * alpha * np.sqrt(-np.log(recip_tol)) + 1.0
    g_list = _inplane_g_mesh(b1, b2, g_max)
    Gz = np.linspace(-g_max, g_max, n_gz)
    dGz = float(Gz[1] - Gz[0])
    # Stack G = g + G_z n̂ for every (g, G_z): shape (n_g . n_gz, 3).
    G_recip = (
        g_list[:, None, :] + Gz[None, :, None] * nhat[None, None, :]
    ).reshape(-1, 3)
    ft = ao_pair_fourier_transform_bloch(basis, G_recip, R_g, k_gamma)
    ft = ft * pair_scales[:, :, None]  # r̂_muν(G), (nbf, nbf, n_G)
    G2 = (G_recip**2).sum(axis=1)
    # v_long(G) = -S_I Z_I (4pi/G^2) e^{-G^2/4a^2} e^{-iG.R_I} (the -Z gives the
    # attractive electron-operator sign, exactly as the 3D path).
    damp = (4.0 * np.pi / G2) * np.exp(-G2 / (4.0 * alpha**2))
    phase = np.exp(-1j * (G_recip @ nuclei_R.T))  # (n_G, n_atoms)
    v_long_G = -(damp * (phase @ nuclei_Z))
    V_long = (dGz / (2.0 * np.pi * area)) * np.real(
        np.einsum("k,mnk->mn", v_long_G, ft.conj())
    )
    V_long = 0.5 * (V_long + V_long.T)

    # ---- V_g0 slab term against the z-resolved AO-pair density n_muν(z) ------
    z_center = 0.5 * (float(z_I.min()) + float(z_I.max()))
    z_half = 0.5 * (float(z_I.max()) - float(z_I.min())) + z_pad
    z_grid, dz, n_prof = _z_resolved_pair_profiles(
        basis,
        system,
        lat_opts,
        nhat=nhat,
        pair_scales=pair_scales,
        R_g=R_g,
        z_center=z_center,
        z_half=z_half,
        gz_profile_max=gz_profile_max,
        profile_tol=profile_tol,
        z_spacing=z_spacing,
        context="compute_v_ne_slab_ewald_2d_gamma",
    )

    # V_g0_muν = (2pi/A) S_I Z_I ∫ dz n_muν(z) g0(z - z_I), electron-signed.
    V_g0 = np.zeros_like(V_short)
    for I in range(len(nuclei_Z)):
        ker = _g0_kernel(z_grid - z_I[I], alpha) * dz  # (n_z,)
        V_g0 += (2.0 * np.pi / area) * nuclei_Z[I] * np.einsum(
            "mnz,z->mn", n_prof, ker
        )
    V_g0 = 0.5 * (V_g0 + V_g0.T)

    v_total = V_short + V_long + V_g0
    v_total = 0.5 * (v_total + v_total.T)
    return SlabVneDecomposition(
        v_total=v_total,
        v_short=V_short,
        v_long=V_long,
        v_g0=V_g0,
        alpha=float(alpha),
    )


def compute_v_ne_slab_ewald_2d_lattice(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    ewald_options: Optional[EwaldOptions] = None,
    **gamma_kwargs,
) -> LatticeMatrixSet:
    """``LatticeMatrixSet`` ``V_ne`` for the Γ dispatch.

    Returns the libint ``V_short`` per-cell blocks with the smooth long-range
    Γ matrix (``V_long + V_g0``) folded into the home (``g = 0``) cell, so that
    ``bloch_sum(.., k = 0)`` reproduces
    :func:`compute_v_ne_slab_ewald_2d_gamma`'s ``v_total`` exactly. Dense
    k-mesh slab drivers use :func:`compute_v_ne_slab_ewald_2d_k_matrix`
    instead, because the z-resolved long-range terms must be Bloch-summed at
    each k-point rather than folded into one home-cell block.
    """
    decomp = compute_v_ne_slab_ewald_2d_gamma(
        basis, system, lat_opts, ewald_options=ewald_options, **gamma_kwargs
    )
    V_set = compute_nuclear_erfc_lattice(
        basis, system, decomp.alpha, lat_opts
    )
    # Find the home cell (r_cart ≈ 0) and add the Γ long-range part there.
    cells = V_set.cells
    home = min(
        range(len(cells)),
        key=lambda c: float(np.dot(cells[c].r_cart, cells[c].r_cart)),
    )
    home_block = np.asarray(V_set.blocks[home]) + decomp.v_long + decomp.v_g0
    V_set.set_block(home, np.ascontiguousarray(home_block, dtype=float))
    return V_set
