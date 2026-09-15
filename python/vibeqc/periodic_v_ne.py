"""V_ne dispatch for periodic SCF -- gauge-aligned with the FFT-Poisson J.

The default ``compute_nuclear_lattice`` is a bare libint direct-
truncated lattice sum: it includes the full ``S_A,g Z_A / |r-R_A-g|``
nuclear potential at each AO product, with no G = 0 omission and no
jellium compensation. For molecular-limit cases (vacuum-padded cell)
this is fine -- the truncation cutoff drops the diffuse interactions
and the result matches an isolated-molecule V_ne to mHa accuracy.

For **tight ionic crystals**, however, the bare lattice sum does
include nearby cells significantly, AND the matching Hartree J is
built via FFT-Poisson with the G = 0 reciprocal mode pinned to zero.
The two halves of the electron Coulomb interaction therefore live
in inconsistent gauges, and ``tr(D . V_ne) + 1/2 tr(D . J)`` fails to
cancel by hundreds of Ha.

LiH conventional rocksalt (a = 4.084 Å, STO-3G, EWALD_3D, iter-1
Hcore guess):

    E_V = tr(D . V_ne)  =  -1680 Ha        <- bare libint, no G=0
    E_J = 1/2 tr(D . J)   =     +16 Ha       <- FFT-Poisson, G=0=0
    expected E_V        ≈    -80 Ha        <- physical

This module dispatches V_ne on ``lat_opts.coulomb_method``:

* :pyattr:`CoulombMethod.EWALD_3D` ->
  :func:`compute_nuclear_lattice_ewald` -- uses the *same* G != 0
  reciprocal-space convention as the J build, plus a v_bg jellium
  shift (a constant) for the missing G = 0 contribution. Gauge-
  aligned with J and finite for tight crystals.
* :pyattr:`CoulombMethod.DIRECT_TRUNCATED` ->
  :func:`compute_nuclear_lattice` (unchanged -- the legacy bare-libint
  path stays for 1D/2D and for direct comparison against pre-v0.7
  behaviour).

Used by all 8 periodic SCF drivers (RHF/UHF/RKS/UKS x Γ-only/multi-k).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    CoulombMethod,
    EwaldOptions,
    GridOptions,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    build_grid,
    compute_nuclear_erfc_lattice,
    compute_nuclear_lattice,
    compute_nuclear_lattice_ewald,
    direct_lattice_cells,
)

__all__ = [
    "compute_nuclear_lattice_dispatch",
    "compute_v_ne_ewald_3d_ft_gamma",
    "compute_v_ne_ewald_3d_ft_lattice",
]


def _vne_ft_g_chunk_size(
    nbf: int,
    *,
    target_mib: Optional[float] = None,
) -> int:
    """Return a bounded ``G`` chunk for analytical EWALD_3D V_ne.

    ``ao_pair_fourier_transform_bloch`` returns a complex
    ``(nbf, nbf, n_G)`` tensor. Dense ionic cells can have hundreds of
    thousands of reciprocal vectors, so materialising the full tensor
    would be tens of GiB. Chunk on the reciprocal index instead and keep
    the per-call FT tensor under a conservative memory target.
    """
    n = max(1, int(nbf))
    if target_mib is None:
        target_mib = float(
            os.environ.get("VIBEQC_VNE_EWALD3D_FT_CHUNK_MIB", "256.0")
        )
    target_bytes = max(1.0, float(target_mib)) * 1024.0 * 1024.0
    # Account for the complex FT tensor plus at least one similarly sized
    # temporary inside NumPy/libint. The exact peak depends on BLAS/NumPy,
    # but this prevents the pathological full-G materialisation.
    bytes_per_g = 2.0 * float(np.dtype(np.complex128).itemsize) * n * n
    return max(1, int(target_bytes // bytes_per_g))


def _vne_ft_contracted_g_chunk_size(
    max_l: int,
    *,
    target_mib: Optional[float] = None,
) -> int:
    """Return a bounded ``G`` chunk for the weight-contracted V_ne kernel.

    :func:`_vne_ft_g_chunk_size` sizes the chunk against the dense
    ``(nbf, nbf, n_G)`` pair-FT tensor. The contracted kernel
    (``ao_pair_fourier_transform_weighted_per_cell``) never builds that
    tensor: its only ``n_G``-proportional storage is the shared
    ``(-iG)^t`` power tables (``3 x (2 max_l + 1)`` complex rows) plus one
    complex scratch row per OpenMP worker. The chunk is therefore
    independent of ``nbf`` -- keeping the old heuristic here would
    over-chunk a large basis by orders of magnitude (nbf = 500 would ask
    for 32-point chunks) and pay the shared-table rebuild on every one.
    """
    if target_mib is None:
        target_mib = float(
            os.environ.get("VIBEQC_VNE_EWALD3D_FT_CHUNK_MIB", "256.0")
        )
    target_bytes = max(1.0, float(target_mib)) * 1024.0 * 1024.0
    try:
        n_threads = max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except AttributeError:  # macOS / Windows have no sched_getaffinity
        n_threads = max(1, os.cpu_count() or 1)
    rows = 3.0 * (2.0 * max(0, int(max_l)) + 1.0) + float(n_threads) + 2.0
    bytes_per_g = rows * float(np.dtype(np.complex128).itemsize)
    return max(1, int(target_bytes // bytes_per_g))


def compute_v_ne_ewald_3d_ft_gamma(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    ewald_options: Optional[EwaldOptions] = None,
    ke_cutoff: float = 200.0,
    pair_ft_shared: "Optional[object]" = None,
    stream_pair_ft: bool = False,
) -> np.ndarray:
    """Γ-only Ewald-3D V_ne via analytical reciprocal-space FT.

    Replaces the real-space Becke-Lebedev molecular-grid V_long
    quadrature in :func:`compute_nuclear_lattice_ewald`
    (cpp/src/lattice_integrals.cpp)
    with the canonical analytical-FT formula per Sun-Berkelbach-Chan
    2017 (DOI 10.1063/1.4998644) / McClain-Sun-Chan-Berkelbach 2017
    (DOI 10.1021/acs.jctc.7b00049) / Lippert-Hutter-Parrinello 1997
    (DOI 10.1080/002689797170220).

    The Ewald split is:

        V_ne^total = V_short + V_long

        V_short_muν(g) = -S_{I,h} Z_I <chi_mu(0) | erfc(a r_{I,h})/r_{I,h}
                                     | chi_ν(g)>       (libint analytical)

        V_long_muν^Γ  = (1/Ω) S_{G!=0} v_long(G) . conj(r̂_muν^Γ(G))
                                                     (analytical FT)

    where

        v_long(G) = -S_I Z_I . (4pi/G^2) . exp(-G^2/(4a^2)) . exp(-iG.R_I)

        r̂_muν^Γ(G) = S_g ∫ chi_mu(r) chi_ν(r-g) e^{-iG.r} dr

    is the Γ-only Bloch-summed AO-pair Fourier transform, computed via
    the C++ kernel ``ao_pair_fourier_transform_bloch_cxx`` shipped by
    the integrals chat (commits 74f3eb4a + 590d022d).

    **Why this fix**: the existing real-space quadrature path
    integrates V_long on a Becke-Lebedev molecular grid (Treutler-
    Ahlrichs radial x Lebedev-Laikov angular x Becke fuzzy-cell
    partition). That grid does not exactly preserve the crystal point-
    group symmetry of a non-cubic primitive cell, so Hcore matrix
    elements that must vanish by site symmetry pick up a finite per-
    element residue (and the residue is not equal across symmetry-
    equivalent elements). Symptom on LiH primitive FCC + sto-3g:

        Hcore[Li-1s, Li-py/pz/px] = (-5.6e-3, +1.66e-2, -5.6e-3)
        PySCF reference            = (~2e-13 across all three)

    The symmetry-breaking error compounds via SCF self-consistency into
    a ~4 mHa total-energy error vs PySCF GDF / RSDF. The reciprocal-
    space sum is symmetric in G by construction -> preserves the C3
    symmetry exactly. Verified 2026-05-29 by direct V_ne substitution
    (vibe-qc SCF with PySCF's V_ne lands within +27 µHa of PySCF GDF).

    Parameters
    ----------
    basis
        AO basis for the unit cell (libint ordering).
    system
        Periodic system; ``system.dim`` must be 3.
    lat_opts
        :class:`LatticeSumOptions`. The ``cutoff_bohr`` and
        ``nuclear_cutoff_bohr`` propagate to V_short. ``coulomb_method``
        is ignored (this routine implements the EWALD_3D split).
    ewald_options
        Optional :class:`EwaldOptions`. If ``None``, defaults to
        ``EwaldOptions()`` with ``real_cutoff_bohr =
        lat_opts.nuclear_cutoff_bohr`` (matches
        :func:`compute_nuclear_lattice_dispatch`).
    ke_cutoff
        Kinetic-energy cutoff (Hartree) for the FT mesh. Default
        ``200`` -- converged for sto-3g LiH FCC to sub-µHa. The
        ``exp(-G^2/(4a^2))`` damping in ``v_long(G)`` makes V_long mesh-
        insensitive for any reasonable ``a``; ``ke_cutoff`` is here
        bounded by the orbital-pair FT bandwidth, not the nuclear
        Coulomb kernel.
    pair_ft_shared
        Optional precomputed dense-mesh AO-pair FT bundle
        (``vibeqc.aux_basis._RsgdfDensePairFT`` from
        ``_rsgdf_dense_pair_ft``). When the GDF driver builds the rsgdf
        cderi and this V_ne in the same SCF it passes one bundle to both,
        so the dominant Bloch AO-pair FT runs once instead of twice.
        ``None`` (default) builds it internally; a provenance mismatch
        (ke_cutoff / cutoff_bohr / n_ao) raises.
    stream_pair_ft
        If true, never materialise the dense ``(n_ao, n_ao, n_G)``
        AO-pair FT: sweep the G-mesh in :func:`_vne_ft_g_chunk_size`
        chunks with a per-chunk C++ pair FT, accumulating V_long. The
        memory-lean companion of the screened GDF fit
        (``handovers/HANDOVER_GDF_FIT_SCREENING.md``) -- the GDF driver
        selects it when ``fit_screen_threshold > 0``, where the fit
        build no longer produces a shared dense bundle to reuse.
        Conflicts with ``pair_ft_shared`` (which IS that dense
        materialisation): passing both raises.

    Returns
    -------
    V_ne : (n_orb, n_orb) ndarray
        Γ-point V_ne matrix in libint AO ordering, symmetrised.

    Notes
    -----
    Γ-only. For finite-k callers (BIPOLE), the per-cell decomposition
    is needed and this routine does not provide it. Defer to v0.12.0 +
    BIPOLE-chat scope.
    """
    from .aux_basis import _resolve_pair_ft_shared
    from ._vibeqc_core import bloch_sum

    if lat_opts.pair_complete_1e:
        if stream_pair_ft and pair_ft_shared is not None:
            raise ValueError(
                "compute_v_ne_ewald_3d_ft_gamma: stream_pair_ft cannot "
                "consume a precomputed pair_ft_shared bundle."
            )
        if pair_ft_shared is not None:
            _resolve_pair_ft_shared(
                pair_ft_shared, basis, system, ke_cutoff, lat_opts,
            )
        # A dense bundle built on the historical ball has no pair-support
        # provenance. Build the coupled per-cell split on its own support.
        blocks = compute_v_ne_ewald_3d_ft_lattice(
            basis, system, lat_opts, ewald_options=ewald_options,
            ke_cutoff=ke_cutoff, screen_rel=0.0,
        )
        value = np.real(bloch_sum(blocks, np.zeros(3)))
        return 0.5 * (value + value.T)

    if system.dim != 3:
        raise ValueError(
            "compute_v_ne_ewald_3d_ft_gamma: requires dim == 3; got "
            f"dim = {system.dim}."
        )

    # ---- Ewald split. For all-electron periodic systems, splitting the
    # nuclear Coulomb 1/r into erfc(a r)/r + erf(a r)/r lets the steep
    # core electrons be handled exactly by libint (V_short, no mesh
    # truncation) while the smooth long-range tail is computed by
    # symmetric reciprocal-space FT (V_long, no cubic-grid C3 break).
    # The a is chosen so V_short converges in a few unit cells AND
    # V_long is mesh-converged by ke ≈ 200-400 Ha for typical AO bases.
    #
    # G = 0 mode of V_long is DROPPED (PySCF convention -- jellium-
    # neutralised, no explicit v_bg shift). This matches PySCF's
    # df.GDF / df.RSDF get_nuc convention.
    a_lattice = np.asarray(system.lattice, dtype=float)
    cell_volume = float(np.abs(np.linalg.det(a_lattice)))

    if ewald_options is None:
        ewald_options = EwaldOptions()
        ewald_options.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr
    alpha = ewald_options.alpha
    if alpha <= 0.0:
        # a-selection heuristic: V_short via libint analytical exactly
        # handles the nuclear-cusp / steep-core region; V_long via FT
        # handles the smooth tail. Picking a larger pushes more work
        # into V_short (analytical) and reduces the V_long FT mesh
        # requirement.
        #
        # The OPTIMAL a scales with the steepest AO exponent: V_long FT
        # needs Gmax ≈ 2.√(max_AO_a) to resolve the AO pair density; if
        # we damp v_long(G) by exp(-G^2/(4a^2)) with a >= √(max_AO_a)/2,
        # the mesh truncation is hidden under the Gaussian damping.
        # In practice a = 2.0 works for Z <= 20 (Mg-1s a ≈ 31 -> √a/2 ≈
        # 2.8; conservative for sto-3g class basis, generous for
        # def2-svp-jk).
        alpha = 2.0

    # ---- V_short via libint analytical erfc-screened nuclear attraction ----
    V_short_lat = compute_nuclear_erfc_lattice(basis, system, alpha, lat_opts)
    V_short = np.real(bloch_sum(V_short_lat, np.zeros(3)))
    V_short = 0.5 * (V_short + V_short.T)

    # ---- V_long via analytical reciprocal-space FT (G != 0 only) -----------
    # Dense-mesh AO-pair FT r̂_muν(G) + the (G, |G|^2) it lives on. When the
    # GDF driver precomputes this it is SHARED with the rsgdf cderi build
    # (:func:`vibeqc.aux_basis.build_lpq_native_fft`) -- the dominant C++
    # ``ao_pair_fourier_transform_bloch`` cost, otherwise run twice per
    # SCF. See :class:`vibeqc.aux_basis._RsgdfDensePairFT`. The
    # ``stream_pair_ft`` mode never materialises it at all (chunked
    # per-G sweep below).
    if stream_pair_ft:
        if pair_ft_shared is not None:
            raise ValueError(
                "compute_v_ne_ewald_3d_ft_gamma: stream_pair_ft is the "
                "memory-lean mode and cannot consume a precomputed dense "
                "pair_ft_shared bundle (the bundle IS the (n_ao, n_ao, "
                "n_G) materialisation streaming avoids). Pass one or the "
                "other."
            )
        from .aux_basis import rsgdf_dense_g_mesh

        G_all = rsgdf_dense_g_mesh(system, float(ke_cutoff))
        G2_all = (G_all**2).sum(axis=1)
        nz = G2_all > 0
        G = G_all[nz]
        G2_kept = G2_all[nz]
        pair_ft = None
    else:
        _ft = _resolve_pair_ft_shared(
            pair_ft_shared, basis, system, ke_cutoff, lat_opts
        )
        G, G2_kept, pair_ft = _ft.G, _ft.G2, _ft.pair_ft

    # v_long(G) = -S_I Z_I . (4pi/G^2) . exp(-G^2/(4a^2)) . exp(-iG.R_I)
    nuclei_Z = np.array([a.Z for a in system.unit_cell], dtype=float)
    nuclei_R = np.array(
        [list(a.xyz) for a in system.unit_cell], dtype=float
    )                                            # (n_atoms, 3)
    damp = np.exp(-G2_kept / (4.0 * alpha**2))    # (n_G,) Gaussian decay
    phase = np.exp(
        -1j * np.einsum("ij,gj->ig", nuclei_R, G)
    )                                            # (n_atoms, n_G)
    v_long_G = -np.einsum("i,g,ig->g", nuclei_Z, damp, phase)
    v_long_G *= (4.0 * np.pi) / G2_kept

    # Contract: V_long_muν = (1/Ω) . S_{G!=0} v_long(G) . conj(r̂_muν(G))
    if stream_pair_ft:
        # Same Bloch cell list and per-L calibration the dense bundle
        # bakes in (_rsgdf_dense_pair_ft); the raw per-chunk C++ FT
        # carries neither, so both are applied here.
        from ._aopair_ft import ao_pair_fourier_transform_bloch
        from ._vibeqc_core import direct_lattice_cells
        from .aux_basis import _ao_scales_for_rsgdf

        cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
        R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
        if R_g.size == 0:
            R_g = np.zeros((1, 3), dtype=float)
        ao_scales = _ao_scales_for_rsgdf(basis)
        pair_scales = np.outer(ao_scales, ao_scales)
        nbf = int(basis.nbasis)
        g_chunk = _vne_ft_g_chunk_size(nbf)
        V_long_acc = np.zeros((nbf, nbf), dtype=np.complex128)
        k_gamma = np.zeros(3)
        for lo in range(0, len(G), g_chunk):
            ft = ao_pair_fourier_transform_bloch(
                basis, G[lo : lo + g_chunk], R_g, k_cart=k_gamma
            )
            V_long_acc += np.einsum(
                "g,mng,mn->mn",
                v_long_G[lo : lo + g_chunk],
                ft.conj(),
                pair_scales,
                optimize=True,
            )
        V_long = (1.0 / cell_volume) * np.real(V_long_acc)
    else:
        V_long = (1.0 / cell_volume) * np.real(
            np.einsum("g,mng->mn", v_long_G, pair_ft.conj())
        )
    V_long = 0.5 * (V_long + V_long.T)

    # ---- G = 0 mode correction (PySCF convention). --------------------
    # The libint-analytical V_short is a REAL-SPACE lattice sum that
    # implicitly includes the G=0 mode of v_short(r). FT-wise:
    #
    #   v_short(G) = -S_I Z_I . (4pi/G^2) . [1 - exp(-G^2/(4a^2))] . e^{-iG.R_I}
    #
    # The G->0 limit of [1 - exp(-G^2/(4a^2))]/G^2 is (1/(4a^2)) -- finite --
    # so v_short(G=0) = -pi . S_I Z_I / a^2 . (and the prefactor 1/V from
    # the periodic FT normalisation).
    #
    # PySCF's df.GDF / df.RSDF get_nuc convention drops the entire
    # G = 0 mode of v_full = v_short + v_long; that means both the
    # divergent monopole AND the finite v_short(G=0) tail are dropped.
    # Our V_short_libint includes that finite tail (it's a faithful
    # real-space integral of erfc(ar)/r). To match PySCF we subtract
    # v_short(G=0) . S_muν = (-pi.S Z / (a^2.V)) . S_muν from V_short.
    Z_total = float(nuclei_Z.sum())
    if abs(Z_total) > 1e-12:
        from ._vibeqc_core import compute_overlap_lattice
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        S = np.real(bloch_sum(S_lat, np.zeros(3)))
        S = 0.5 * (S + S.T)
        # v_short(G=0) is NEGATIVE for positive nuclear charge.
        # Subtracting v_short(G=0).S from V_short means ADDING
        # +(pi.S Z / (a^2.V)).S to V_ne (which flips the sign).
        v_short_G0 = -(np.pi / (alpha**2 * cell_volume)) * Z_total
        V_ne_correction = -v_short_G0 * S
    else:
        V_ne_correction = 0.0

    V_ne = V_short + V_long + V_ne_correction
    V_ne = 0.5 * (V_ne + V_ne.T)
    return V_ne


def compute_v_ne_ewald_3d_ft_lattice(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    ewald_options: Optional[EwaldOptions] = None,
    ke_cutoff: float = 200.0,
    screen_rel: float = 1e-12,
) -> LatticeMatrixSet:
    """Per-cell Ewald-3D V_ne via analytical reciprocal-space FT.

    The **lattice (per-cell) generalisation** of
    :func:`compute_v_ne_ewald_3d_ft_gamma`. Returns a
    :class:`LatticeMatrixSet` whose per-cell blocks ``V_ne(g)`` Bloch-
    sum to ``V_ne(k)`` at *any* crystal momentum, so it is a drop-in
    replacement for the grid-quadrature
    :func:`compute_nuclear_lattice_ewald` inside
    :func:`compute_nuclear_lattice_dispatch` -- Γ-only AND multi-k
    callers consume it unchanged via ``bloch_sum(V_lat, k)``.

    The per-cell block is the *same real-space object* the grid path
    approximates by quadrature, computed instead by symmetric analytic
    FT::

        V_ne_muν(g) = V_short_muν(g) + V_long_muν(g) + corr_muν(g)

        V_short_muν(g) : libint erfc-screened nuclear attraction
                        (compute_nuclear_erfc_lattice -- exact, no mesh)
        V_long_muν(g)  = (1/Ω) Re S_{G!=0} v_long(G) . conj(r̂_muν(G; R_g))
        corr_muν(g)    = -v_short(G=0) . S_muν(g)   (PySCF G=0 convention)

    with ``v_long(G) = -S_I Z_I (4pi/G^2) exp(-G^2/4a^2) exp(-iG.R_I)`` and
    ``r̂_muν(G; R_g) = ∫ chi_mu(r) chi_ν(r-R_g) e^{-iG.r} dr`` the per-cell
    AO-pair FT (bra at home, ket translated into cell g). The per-cell
    FT is obtained from the C++ Bloch kernel evaluated at a single-cell
    ``R_g`` list and ``k = 0`` (which collapses the Bloch sum to that
    one cell).

    **Why this replaces the grid path.** The grid quadrature integrates
    V_long on a Becke-Lebedev *molecular* grid whose point set does not
    exactly preserve the crystal point group of a non-cubic primitive
    cell (FCC, hexagonal, rhombohedral). Hcore elements that must
    vanish by site symmetry then pick up a finite, non-symmetry-
    equivalent residue (LiH primitive FCC / sto-3g: ``(Li-1s|V_ne|Li-
    px/py/pz) = (-5.6e-3, +1.66e-2, -5.6e-3)`` vs ``~2e-13`` from
    PySCF), which SCF self-consistency compounds into a ~4 mHa total-
    energy error. The reciprocal-space sum is symmetric in ``G`` by
    construction, so every per-cell block -- and hence the Bloch sum at
    every k -- preserves crystal symmetry exactly. See
    :func:`compute_v_ne_ewald_3d_ft_gamma` for the full derivation and
    references (Lippert-Hutter-Parrinello 1997 / Sun-Berkelbach-McClain-
    Chan 2017 / McClain-Sun-Chan-Berkelbach 2017).

    Summing the returned blocks over all cells (i.e. ``bloch_sum`` at
    ``k = 0``) reproduces :func:`compute_v_ne_ewald_3d_ft_gamma` to
    numerical noise (same formula, same cell + G meshes) --
    ``tests/test_periodic_v_ne_ewald_ft.py`` pins this equivalence.

    Parameters
    ----------
    basis, system, lat_opts
        As :func:`compute_v_ne_ewald_3d_ft_gamma`. ``system.dim`` must
        be 3.
    ewald_options
        Optional :class:`EwaldOptions`; defaults to ``real_cutoff_bohr =
        lat_opts.nuclear_cutoff_bohr``. ``alpha <= 0`` selects the
        auto-rule ``a = 2.0`` (same as the Γ-only routine).
    ke_cutoff
        Kinetic-energy cutoff (Hartree) for the FT mesh; default
        ``200`` (µHa-converged for light-atom cells like LiH FCC).
    screen_rel
        Cells whose overlap-block Frobenius norm is below
        ``screen_rel x ‖S(0)‖_F`` get ``V_long(g) = 0`` (their AO-pair
        density -- and hence its FT -- is negligible, exactly as the grid
        path produces ≈ 0 for distant cells). Set to ``0`` to compute
        every cell.

    Returns
    -------
    LatticeMatrixSet
        Per-cell V_ne blocks in libint AO ordering, on the same cell
        list as :func:`compute_nuclear_erfc_lattice`.
    """
    from .aux_basis import rsgdf_dense_g_mesh, _ao_scales_for_rsgdf
    from ._vibeqc_core import (
        ao_pair_fourier_transform_weighted_per_cell,
        compute_overlap_lattice,
    )

    if system.dim != 3:
        raise ValueError(
            "compute_v_ne_ewald_3d_ft_lattice: requires dim == 3; got "
            f"dim = {system.dim}. Use DIRECT_TRUNCATED for 1D / 2D."
        )

    a_lattice = np.asarray(system.lattice, dtype=float)
    cell_volume = float(np.abs(np.linalg.det(a_lattice)))

    if ewald_options is None:
        ewald_options = EwaldOptions()
        ewald_options.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr
    alpha = ewald_options.alpha
    if alpha <= 0.0:
        alpha = 2.0

    # ---- V_short per-cell (libint) -- also our LatticeMatrixSet skeleton.
    # compute_nuclear_erfc_lattice fixes the authoritative cell list
    # (direct_lattice_cells at lat_opts.cutoff_bohr) and nbf; we overwrite
    # each block in place with V_short + V_long + corr via set_block.
    V_set = compute_nuclear_erfc_lattice(basis, system, alpha, lat_opts)
    cells = V_set.cells
    n_cells = len(cells)
    V_short_blocks = V_set.blocks  # snapshot copy (pybind returns a fresh list)

    # ---- Per-cell overlap (same cell list) for the G=0 correction + screen.
    S_set = compute_overlap_lattice(basis, system, lat_opts)
    if len(S_set.cells) != n_cells:
        raise RuntimeError(
            "compute_v_ne_ewald_3d_ft_lattice: overlap and erfc cell "
            f"lists disagree ({len(S_set.cells)} vs {n_cells}); check "
            "LatticeSumOptions.cutoff_bohr."
        )
    S_blocks = S_set.blocks
    if any(
        not np.array_equal(np.asarray(s.index), np.asarray(v.index))
        or not np.array_equal(np.asarray(s.r_cart), np.asarray(v.r_cart))
        for s, v in zip(S_set.cells, cells)
    ):
        raise RuntimeError("compute_v_ne_ewald_3d_ft_lattice: cell ordering differs")

    # ---- v_long(G) on the dense FT mesh (G != 0; PySCF jellium G=0 drop).
    G_all = rsgdf_dense_g_mesh(system, float(ke_cutoff))
    G2 = (G_all**2).sum(axis=1)
    nonzero = G2 > 0
    G = G_all[nonzero]
    G2_kept = G2[nonzero]
    damp = np.exp(-G2_kept / (4.0 * alpha**2))
    # v_long(G) is damped by exp(-G^2/4a^2); points where the damping has
    # decayed below ~1e-14 contribute nothing -- drop them so an over-
    # large ke_cutoff (or a big vacuum box's dense reciprocal mesh) does
    # not inflate the per-cell FT cost. (The kept-set is symmetric in G,
    # preserving the crystal-symmetry property.)
    keep = damp > 1e-14
    G = G[keep]
    G2_kept = G2_kept[keep]
    damp = damp[keep]

    nuclei_Z = np.array([a.Z for a in system.unit_cell], dtype=float)
    nuclei_R = np.array(
        [list(a.xyz) for a in system.unit_cell], dtype=float
    )
    phase = np.exp(-1j * np.einsum("ij,gj->ig", nuclei_R, G))
    v_long_G = -np.einsum("i,g,ig->g", nuclei_Z, damp, phase)
    v_long_G *= (4.0 * np.pi) / G2_kept

    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    nbf = int(pair_scales.shape[0])
    g_chunk = _vne_ft_contracted_g_chunk_size(
        max((int(sh.l) for sh in basis.shells()), default=0)
    )

    # ---- G = 0 v_short tail subtracted to match PySCF's get_nuc (the
    # libint real-space erfc sum implicitly includes the finite
    # v_short(G=0) = -pi.SZ/a^2 mode that PySCF drops). See the Γ-only
    # routine for the full convention note.
    Z_total = float(nuclei_Z.sum())
    v_short_G0 = (
        -(np.pi / (alpha**2 * cell_volume)) * Z_total
        if abs(Z_total) > 1e-12
        else 0.0
    )

    # Screen against the LARGEST overlap block (the home cell, but found
    # by value rather than assuming cell index 0) so the threshold is a
    # true relative bound regardless of cell ordering.
    s_norms = np.array(
        [float(np.linalg.norm(np.asarray(b))) for b in S_blocks]
    )
    s_ref = float(s_norms.max()) if n_cells else 0.0
    screen = float(screen_rel) * s_ref if s_ref > 0.0 else 0.0

    # Cells whose AO-pair density (and hence its FT) is negligible get
    # V_long(g) = 0 -- the grid path likewise yields ~ 0 there. Only the
    # survivors reach the kernel.
    active = [c for c in range(n_cells) if s_norms[c] > screen]

    # V_long for every active cell in one pass. The kernel contracts
    # against v_long(G) inside its Hermite loop, so the dense
    # (nbf, nbf, n_G) pair-FT tensor is never built (it used to be
    # materialised per cell only to be contracted away), and because the
    # output is per-cell rather than Bloch-summed the whole
    # (shell-pair, cell) task space parallelises with no reduction. The
    # previous per-cell Python loop could only use the Bloch kernel's
    # shell-pair parallelism, which starves on a small basis with many
    # image cells (LiH/STO-3G: 16 shell pairs, 135 cells -> 2.97x on 8
    # threads).
    V_long_all = None
    if active:
        R_active = np.ascontiguousarray(
            np.array([list(cells[c].r_cart) for c in active], dtype=float)
        )
        # G is still swept in chunks so the shared (-iG)^t power tables
        # inside the kernel stay bounded on very dense reciprocal meshes.
        V_long_all = np.zeros((len(active), nbf, nbf), dtype=float)
        for start in range(0, len(G), g_chunk):
            stop = min(start + g_chunk, len(G))
            V_long_all += ao_pair_fourier_transform_weighted_per_cell(
                basis,
                np.ascontiguousarray(G[start:stop]),
                R_active,
                np.ascontiguousarray(v_long_G[start:stop]),
            )
        V_long_all *= pair_scales[None, :, :] / cell_volume

    active_pos = {c: i for i, c in enumerate(active)}
    for c in range(n_cells):
        S_b = np.asarray(S_blocks[c])
        corr = (-v_short_G0) * S_b if v_short_G0 != 0.0 else 0.0
        V_short_b = np.asarray(V_short_blocks[c])
        if c not in active_pos:
            new_block = V_short_b + corr
        else:
            new_block = V_short_b + V_long_all[active_pos[c]] + corr
        if lat_opts.pair_complete_1e:
            from .lattice_screening import ao_pair_support_mask

            new_block[~ao_pair_support_mask(
                basis, cells[c].r_cart, lat_opts.cutoff_bohr,
            )] = 0.0
        V_set.set_block(c, np.ascontiguousarray(new_block, dtype=float))

    return V_set


def compute_nuclear_lattice_dispatch(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    grid_options: Optional[GridOptions] = None,
    ewald_options: Optional[EwaldOptions] = None,
    ke_cutoff: Optional[float] = None,
) -> LatticeMatrixSet:
    """Build the V_ne matrix-set with gauge consistent with the J build.

    Dispatches on ``lat_opts.coulomb_method``:

    * ``EWALD_3D``: Ewald-split V_ne (G = 0 dropped in the reciprocal-
      space sum, finite v_short(G=0) tail subtracted -- PySCF get_nuc
      convention). Matches the gauge of :func:`build_j_ewald_3d`. The
      long-range half (V_long) is built by the **analytical
      reciprocal-space FT** :func:`compute_v_ne_ewald_3d_ft_lattice` by
      default (symmetric in G -> preserves crystal point-group symmetry
      exactly). The legacy Becke-Lebedev molecular-grid quadrature
      (:func:`compute_nuclear_lattice_ewald`) broke that symmetry on
      non-cubic primitive cells (FCC / hexagonal / rhombohedral) by
      ~mHa after SCF self-consistency -- see
      :func:`compute_v_ne_ewald_3d_ft_lattice`. Set the env var
      ``VIBEQC_VNE_EWALD3D_BACKEND=grid`` to restore the legacy grid
      path (diagnostic / bisection only).
    * ``DIRECT_TRUNCATED`` (and any non-EWALD): falls back to the bare
      libint lattice sum (legacy v0.6.x behaviour).

    Parameters
    ----------
    basis
        AO basis for the unit cell.
    system
        Periodic system; ``system.dim`` must be 3 for the EWALD_3D path.
    lat_opts
        :class:`LatticeSumOptions`; the ``coulomb_method`` field selects
        the dispatch and the cutoffs are propagated to the Ewald engine.
    grid_options
        Optional :class:`GridOptions`. Only consulted on the (env-gated)
        legacy ``grid`` backend for the molecular-grid V_long
        quadrature; ignored by the default analytical-FT path.
    ewald_options
        Optional explicit Ewald state for the ``EWALD_3D`` path.
        Supplying this is the preferred route when another energy
        component must share the same alpha / cutoffs, as in the
        CRYSTAL-style BIPOLE driver.
    ke_cutoff
        Kinetic-energy cutoff (Hartree) for the analytical-FT V_long
        mesh. ``None`` (default) reads ``VIBEQC_VNE_EWALD3D_KE`` if set,
        else uses 200.0 (µHa-converged for light-atom cells). Ignored
        by the legacy grid backend.
    """
    if lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
        if system.dim != 3:
            raise ValueError(
                "compute_nuclear_lattice_dispatch: EWALD_3D requires "
                f"dim == 3; got dim = {system.dim}. Use "
                "DIRECT_TRUNCATED for 1D / 2D systems."
            )
        if ewald_options is None:
            eopts = EwaldOptions()
            eopts.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr
        else:
            eopts = ewald_options

        backend = os.environ.get(
            "VIBEQC_VNE_EWALD3D_BACKEND", "analytic_ft"
        ).lower()
        if backend == "grid":
            # Legacy Becke-Lebedev molecular-grid V_long quadrature.
            # Retained behind an env var for diagnostics / bisection;
            # it breaks crystal symmetry on non-cubic primitive cells
            # (see compute_v_ne_ewald_3d_ft_lattice). Not the default.
            gopts = (
                grid_options if grid_options is not None else GridOptions()
            )
            grid = build_grid(system.unit_cell_molecule(), gopts)
            return compute_nuclear_lattice_ewald(
                basis, system, grid, lat_opts, eopts
            )

        if ke_cutoff is None:
            ke = float(os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0"))
        else:
            ke = float(ke_cutoff)
        return compute_v_ne_ewald_3d_ft_lattice(
            basis, system, lat_opts, ewald_options=eopts, ke_cutoff=ke
        )
    if lat_opts.coulomb_method == CoulombMethod.SLAB_EWALD_2D:
        # Rigorous Parry / de Leeuw-Perram 2D (slab) Ewald V_ne. Γ-valid
        # LatticeMatrixSet (libint V_short per cell + the z-resolved
        # reciprocal V_long and slab V_g0 folded into the home cell). See
        # vibeqc.periodic_v_ne_slab. Dense k-mesh slab drivers call the
        # direct V_ne(k) helper instead of this Γ-folded lattice set.
        if system.dim != 2:
            raise ValueError(
                "compute_nuclear_lattice_dispatch: SLAB_EWALD_2D requires "
                f"dim == 2; got dim = {system.dim}."
            )
        from .periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_lattice

        return compute_v_ne_slab_ewald_2d_lattice(
            basis, system, lat_opts, ewald_options=ewald_options
        )
    return compute_nuclear_lattice(basis, system, lat_opts)
