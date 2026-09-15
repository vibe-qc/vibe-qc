"""Analytic fixed-density gradient of the slab (dim=2) 2D-Ewald ``V_ne``.

Rung 3 of the slab GDF gradient ladder
(``handovers/HANDOVER_GDF_GRADIENT_DEFERRED.md`` § 6): the derivative of

    E_ne = S_k w_k Re Tr[ D(k) V_ne(k) ]

at fixed D(k), where ``V_ne(k)`` is EXACTLY what the slab GDF SCF evaluates
(:func:`vibeqc.periodic_v_ne_slab.compute_v_ne_slab_ewald_2d_k_matrix` on
the :func:`~vibeqc.periodic_v_ne_slab.build_v_ne_slab_ewald_2d_k_cache`
mesh policy, in the bare Parry / de Leeuw-Perram 2D-Ewald gauge; Parry,
Surf. Sci. **49**, 433 (1975); de Leeuw & Perram, Mol. Phys. **37**, 1313
(1979)).  ``V_ne = V_short + V_long + V_g0`` and every piece carries atom
positions on BOTH sides:

* nuclear side — the point charges in the erfc lattice sum (``V_short``),
  the structure-factor phase ``exp(-iG.R_I)`` (``V_long``), and the kernel
  offset ``g0(z - z_I)`` (``V_g0``);
* AO side — the Gaussian-product pair centres in the libint erfc integrals
  (``V_short``) and in the AO-pair Fourier transforms that build both
  reciprocal pieces (``V_long`` over the 2D g x G_z mesh, ``V_g0`` over the
  ``g = 0`` ``G_z``-profile mesh).

Derivative strategy per piece (each identity below was pinned numerically
at <= 2e-14 on the compact H2 (2,2,1) fixture before this landed):

* ``V_short`` — fold D(k) onto the real-space lattice
  (:func:`vibeqc.periodic_gdf_gradient._fold_hermitian_k_matrices_to_lattice`)
  and contract with the k-blind C++ erfc kernel
  ``nuclear_erfc_lattice_gradient_contribution`` (AO-centre + nuclear
  point-charge derivatives), exactly like the 3D multi-k rung 3b.

* ``V_long`` — reciprocal per-k form.  The k-matrix builder conjugates the
  Bloch-summed pair FT AFTER the ``exp(+ik.R_g)`` phases, so

      Re Tr[D(k) V_long(k)]
        = Re S_G v_long(G) S_ab (D(k)^T)_ab s_ab conj( FT^{(+k)}_ab(G) )

  with ``v_long(G) = -(dG_z / 2piA) (4pi/G^2) e^{-G^2/4a^2}
  S_I Z_I e^{-iG.R_I}`` (the 3D Coulomb kernel on the
  {in-plane g != 0} x {dense G_z} slab mesh).  AO side: the Bloch
  gweighted centre-derivative kernel
  (:func:`vibeqc._aopair_ft.ao_pair_fourier_transform_bloch_gradient_gweighted`)
  at ``k_cart = +k`` with ``Q_ab(G) = w_k v_long(G) (D(k)^T)_ab s_ab``.
  Nuclear side: ``d v_long(G)/dR_A = -(dG_z/2piA) Z_A (4pi/G^2)
  e^{-G^2/4a^2} (-iG) e^{-iG.R_A}`` contracted with the density-weighted
  conjugated Bloch pair FT.

* ``V_g0`` — the Parry ``g = 0`` slab kernel
  ``g0(u) = u erf(au) + e^{-a^2u^2}/(a sqrt(pi))``.  Nuclear side: the
  chain rule gives ``d g0(u)/du = erf(au)`` — the two Gaussian terms
  cancel exactly, the same cancellation the 2D-Ewald nn gradient's g = 0
  force uses (rung 2, ``cpp/src/ewald.cpp``
  ``ewald_2d_point_charge_gradient_with_background``) — contracted with
  the k-folded z-resolved AO-pair density.  AO side: over the symmetric
  ``G_z``-profile mesh, ``conj(F(G_z)) = F(-G_z)`` (real AO pairs) makes
  the z-quadrature ``S_z n(z) W(z)`` equal to
  ``(dG_z/2pi) S_j F(G_z_j) w~_j`` with ``w~_j = S_z W(z) e^{iG_z_j z}``
  (real after the j sum), so the AO derivative is the same Bloch gweighted
  kernel at ``k_cart = -k`` with
  ``Q_ab(j) = w_k (dG_z/2pi) (D(k)^T)_ab s_ab conj(w~_j)``.

The z-grid and G_z-profile meshes are rebuilt with the shared policy
(:func:`vibeqc.periodic_v_ne_slab._g0_profile_meshes`), and the cell list
comes from the same erfc lattice build the value cache uses, so the
derivative differentiates the SCF's own quadrature.  Mesh *placement*
(z-grid endpoints follow the nuclear extent) is deliberately not
differentiated: its energy dependence is the quadrature error itself.
"""

from __future__ import annotations

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    CoulombMethod,
    LatticeSumOptions,
    PeriodicSystem,
    compute_nuclear_erfc_lattice,
    nuclear_erfc_lattice_gradient_contribution,
)

__all__ = ["compute_v_ne_slab_ewald_2d_gradient"]


def _require_bloch_gradient_kernel_support(basis: BasisSet) -> None:
    """Fail loud where the value route would still run.

    The value builders fall back to the pure-Python AO-pair FT for
    Cartesian L > 0 shells or L > 6, but the centre-derivative kernel
    (``ao_pair_fourier_transform_bloch_gradient_gweighted``) is C++-only
    (pure-spherical, L <= 6).  Name the gap instead of failing deep inside
    the native call.
    """
    shells = basis.shells()
    max_l = max(int(sh.l) for sh in shells)
    all_pure = all(bool(sh.pure) or int(sh.l) == 0 for sh in shells)
    if not all_pure or max_l > 6:
        raise NotImplementedError(
            "compute_v_ne_slab_ewald_2d_gradient: the Bloch AO-pair FT "
            "centre-derivative kernel is C++-only (pure-spherical shells, "
            f"L <= 6); got max L = {max_l}, all_pure = {all_pure}. The "
            "slab V_ne VALUE route supports this basis via the Python "
            "fallback, but its derivative does not."
        )


def compute_v_ne_slab_ewald_2d_gradient(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    D_k_list,
    weights,
    k_cart_list,
    *,
    alpha: float = 0.0,
    recip_tol: float = 1e-13,
    n_gz: int = 401,
    gz_profile_max=None,
    profile_tol: float = 1e-7,
    z_pad: float = 8.0,
    z_spacing: float = 0.025,
) -> np.ndarray:
    """Fixed-density gradient of the slab V_ne energy, shape (n_atoms, 3).

    Differentiates ``S_k w_k Re Tr[D(k) V_ne(k)]`` with ``V_ne(k)`` built
    by :func:`vibeqc.periodic_v_ne_slab.compute_v_ne_slab_ewald_2d_k_matrix`
    on a :func:`~vibeqc.periodic_v_ne_slab.build_v_ne_slab_ewald_2d_k_cache`
    cache with the SAME keyword parameters (``alpha <= 0`` resolves to the
    forced slab default 0.4, exactly as the cache builder and the slab GDF
    driver's ``alpha=0.0`` call do).

    Parameters mirror the cache builder; ``D_k_list`` are the per-k
    Hermitian density matrices, ``weights`` / ``k_cart_list`` the k-mesh
    weights and Cartesian k-points the SCF used.
    """
    from ._aopair_ft import (
        ao_pair_fourier_transform_bloch,
        ao_pair_fourier_transform_bloch_gradient_gweighted,
    )
    from .aux_basis import _ao_scales_for_rsgdf
    from .periodic_gdf_gradient import _fold_hermitian_k_matrices_to_lattice
    from .periodic_v_ne import _vne_ft_g_chunk_size
    from .periodic_v_ne_slab import (
        _DEFAULT_ALPHA,
        _g0_kernel,
        _g0_profile_meshes,
        _inplane_g_mesh,
        _slab_geometry,
        _z_resolved_pair_profiles_at_cells,
    )

    if int(system.dim) != 2:
        raise ValueError(
            "compute_v_ne_slab_ewald_2d_gradient: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {int(system.dim)}."
        )
    if lat_opts.coulomb_method != CoulombMethod.SLAB_EWALD_2D:
        raise ValueError(
            "compute_v_ne_slab_ewald_2d_gradient: lat_opts must carry the "
            "SLAB_EWALD_2D gauge the slab GDF SCF uses for V_ne."
        )
    _require_bloch_gradient_kernel_support(basis)

    kpts = np.asarray(k_cart_list, dtype=float).reshape(-1, 3)
    w_arr = np.asarray(weights, dtype=float).reshape(-1)
    if w_arr.shape[0] != kpts.shape[0] or len(D_k_list) != kpts.shape[0]:
        raise ValueError(
            "compute_v_ne_slab_ewald_2d_gradient: D_k_list, weights and "
            f"k_cart_list must agree in length; got {len(D_k_list)}, "
            f"{w_arr.shape[0]}, {kpts.shape[0]}."
        )
    n_orb = int(basis.nbasis)
    densities = []
    for ik, D in enumerate(D_k_list):
        D = np.asarray(D, dtype=np.complex128)
        if D.shape != (n_orb, n_orb):
            raise ValueError(
                "compute_v_ne_slab_ewald_2d_gradient: D(k) must have shape "
                f"({n_orb}, {n_orb}); got {D.shape} at k index {ik}."
            )
        herm_err = float(np.max(np.abs(D - D.conj().T)))
        if herm_err > 1e-8:
            # The reciprocal-form identities used below assume D^H = D
            # (the SCF density is Hermitian by construction).
            raise ValueError(
                "compute_v_ne_slab_ewald_2d_gradient: D(k) must be "
                f"Hermitian; |D - D^H| = {herm_err:.2e} at k index {ik}."
            )
        densities.append(D)

    alpha = float(alpha)
    if alpha <= 0.0:
        alpha = _DEFAULT_ALPHA
    area, nhat, b1, b2 = _slab_geometry(system)
    nuclei_Z = np.array([a.Z for a in system.unit_cell], dtype=float)
    nuclei_R = np.array([list(a.xyz) for a in system.unit_cell], dtype=float)
    z_I = nuclei_R @ nhat
    n_atoms = len(system.unit_cell)

    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_scales = np.outer(ao_scales, ao_scales)

    # ---- V_short: erfc real-space lattice sum on the folded density -------
    # The fold template IS the value route's erfc lattice set, so the
    # gradient kernel walks the identical cell list (blocks are overwritten
    # by the fold; cells are untouched).
    v_short_lat = compute_nuclear_erfc_lattice(basis, system, alpha, lat_opts)
    cells = list(v_short_lat.cells)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    D_fold = _fold_hermitian_k_matrices_to_lattice(
        densities, w_arr, kpts, v_short_lat
    )
    gradient = np.asarray(
        nuclear_erfc_lattice_gradient_contribution(
            basis, system, D_fold, lat_opts, alpha
        ),
        dtype=np.float64,
    )

    # ---- V_long: {in-plane g != 0} x {dense G_z} reciprocal mesh ----------
    # Mesh policy mirrors build_v_ne_slab_ewald_2d_k_cache verbatim (the
    # mesh itself is geometry-independent: it depends on alpha and the
    # in-plane reciprocal lattice only).
    g_max = 2.0 * alpha * np.sqrt(-np.log(recip_tol)) + 1.0
    g_list = _inplane_g_mesh(b1, b2, g_max)
    if g_list.shape[0] > 0:
        Gz = np.linspace(-g_max, g_max, int(n_gz))
        dGz = float(Gz[1] - Gz[0])
        G_recip = (
            g_list[:, None, :] + Gz[None, :, None] * nhat[None, None, :]
        ).reshape(-1, 3)
        G2 = (G_recip**2).sum(axis=1)
        # v_long(G) = -(dG_z/2piA) (4pi/G^2) e^{-G^2/4a^2} S_I Z_I e^{-iG.R_I}
        # (Parry 1975 / de Leeuw-Perram 1979 slab reciprocal kernel; the
        # prefactor is the L_z -> inf limit dG_z/(2piA) of 1/Omega).
        pref_long = dGz / (2.0 * np.pi * area)
        damp = (4.0 * np.pi / G2) * np.exp(-G2 / (4.0 * alpha**2))
        phase = np.exp(-1j * (G_recip @ nuclei_R.T))  # (n_G, n_atoms)
        v_long_G = pref_long * (-(damp * (phase @ nuclei_Z)))

        g_chunk = _vne_ft_g_chunk_size(n_orb)
        n_G = G_recip.shape[0]
        for ik in range(kpts.shape[0]):
            w_k = float(w_arr[ik])
            k = kpts[ik]
            # Elementwise weight (D(k)^T)_ab s_ab (Hermitian D: D^T = conj D).
            D_weight = densities[ik].T * pair_scales
            for start in range(0, n_G, g_chunk):
                stop = min(start + g_chunk, n_G)
                G_c = G_recip[start:stop]
                phase_c = phase[start:stop]
                damp_c = damp[start:stop]
                v_long_c = v_long_G[start:stop]

                # Nuclear structure-factor derivative,
                #   d v_long(G)/dR_A = -pref Z_A (4pi/G^2) e^{-G^2/4a^2}
                #                       (-iG) e^{-iG.R_A},
                # contracted with the density-weighted, cell-wise conjugated
                # Bloch pair FT
                #   t_k(G) = S_ab (D^T s)_ab S_g e^{+ik.R_g} conj(r̂_ab(g; G)).
                #
                # The conjugation belongs to the AO-pair FT alone (the value
                # side derives this at
                # :func:`vibeqc.periodic_v_ne_slab.compute_v_ne_slab_ewald_2d_k_matrix`):
                # V_long(r) carries e^{+iG.r} while the FT convention is
                # e^{-iG.r}, so the sum needs r̂(g; -G) = conj(r̂(g; G)), and
                # the +ik Bloch phase must survive unconjugated. Conjugating
                # the assembled Bloch sum instead would differentiate
                # V_long(-k). ``ao_pair_fourier_transform_bloch`` returns
                # conj(.) of exactly this object when called at -k, which
                # keeps the streaming (no per-cell tensor).
                pair_ft_mk = ao_pair_fourier_transform_bloch(
                    basis, G_c, R_g, k_cart=-k
                )
                t_k = np.einsum(
                    "ab,abg->g", D_weight, pair_ft_mk.conj(), optimize=True
                )
                del pair_ft_mk
                d_structure = (
                    -pref_long
                    * nuclei_Z[:, None, None]
                    * damp_c[None, :, None]
                    * phase_c.T[:, :, None]
                    * (-1j * G_c[None, :, :])
                )
                gradient += w_k * np.real(
                    np.einsum("agx,g->ax", d_structure, t_k, optimize=True)
                )

                # AO-side pair-FT centre derivative (rung-1 Bloch kernel).
                # The kernel evaluates
                #   d/dR Re S_G S_ab Q_ab(G) conj(FT^{(+k')}_ab(G)),
                # so it must be driven at k' = -k to differentiate the same
                # S_g e^{+ik.R_g} conj(r̂_ab(g; G)) the value side builds.
                Q = w_k * D_weight[:, :, None] * v_long_c[None, None, :]
                gradient += ao_pair_fourier_transform_bloch_gradient_gweighted(
                    basis, G_c, R_g, -k, Q, n_atoms
                )

    # ---- V_g0: the Parry g = 0 slab term -----------------------------------
    z_center = 0.5 * (float(z_I.min()) + float(z_I.max()))
    z_half = 0.5 * (float(z_I.max()) - float(z_I.min())) + float(z_pad)
    Gz_p, dGz_p, z_grid, dz = _g0_profile_meshes(
        basis,
        z_center=z_center,
        z_half=z_half,
        gz_profile_max=gz_profile_max,
        profile_tol=profile_tol,
        z_spacing=z_spacing,
    )
    G0 = Gz_p[:, None] * nhat[None, :]
    g0_pref = 2.0 * np.pi / area

    # AO side: w~_j = S_z W(z) e^{iG_z_j z}, W(z) = (2pi/A) S_I Z_I
    # g0(z - z_I) dz.  Over the symmetric G_z mesh S_j F(j) w~_j is real
    # (conj(F(G_z)) = F(-G_z) for real AO pairs, W real), so the g = 0
    # z-quadrature is exactly the Bloch gweighted kernel at k_cart = -k
    # with Q_ab(j) = w_k (dG_z/2pi) (D^T s)_ab conj(w~_j).
    W_z = np.zeros_like(z_grid)
    for Z, zi in zip(nuclei_Z, z_I):
        W_z += g0_pref * float(Z) * _g0_kernel(z_grid - zi, alpha) * dz
    wtil = np.exp(1j * np.outer(Gz_p, z_grid)) @ W_z  # (n_Gz,)
    for ik in range(kpts.shape[0]):
        w_k = float(w_arr[ik])
        D_weight = densities[ik].T * pair_scales
        Q0 = (
            (w_k * dGz_p / (2.0 * np.pi))
            * D_weight[:, :, None]
            * np.conj(wtil)[None, None, :]
        )
        gradient += ao_pair_fourier_transform_bloch_gradient_gweighted(
            basis, G0, R_g, -kpts[ik], Q0, n_atoms
        )

    # Nuclear side: E_g0 = (2pi/A) S_I Z_I S_z n~(z) g0(z - z_I) dz with the
    # k-folded density profile n~(z) = S_g S_ab D_fold(g)_ab n_ab(g; z).
    # d g0(u)/du = erf(au) — the Gaussian terms cancel exactly (same
    # cancellation as the rung-2 nn g = 0 force, cpp/src/ewald.cpp) — and
    # du/dR_I = -nhat, so
    #   dE_g0/dR_I = -(2pi/A) Z_I [S_z n~(z) erf(a (z - z_I)) dz] nhat.
    from scipy.special import erf

    _, _, n_prof_at_cells = _z_resolved_pair_profiles_at_cells(
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
        context="compute_v_ne_slab_ewald_2d_gradient",
    )
    D_fold_blocks = np.array(
        [np.asarray(D_fold.blocks[c]) for c in range(len(cells))]
    )
    n_tilde = np.einsum(
        "gab,gabz->z", D_fold_blocks, n_prof_at_cells, optimize=True
    )
    for I in range(n_atoms):
        gradient[I] += (
            -g0_pref
            * float(nuclei_Z[I])
            * float(np.sum(n_tilde * erf(alpha * (z_grid - z_I[I]))) * dz)
        ) * nhat

    return gradient
