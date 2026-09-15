"""Phase 15c-3b: multi-k open-shell UKS SCF driver with composed
EWALD_3D Coulomb dispatch.

Open-shell DFT counterpart of:

  * :func:`run_uhf_periodic_multi_k_ewald3d` -- same per-spin
    DIIS / damping / orthogonalisation structure.
  * :func:`run_uks_periodic_gamma_ewald3d` -- Γ-only UKS Ewald
    (15c-3a).
  * :func:`run_rks_periodic_multi_k_ewald3d` -- multi-k closed-shell
    KS Ewald (15c-2).

Per SCF iteration:

    F_a(k) =  H_core(k) + Bloch_k[J(D_a + D_b, w) - K_HF(D_a)]
                       + Bloch_k[V_xc^a(g)]
    F_b(k) =  H_core(k) + Bloch_k[J(D_a + D_b, w) - K_HF(D_b)]
                       + Bloch_k[V_xc^b(g)]

with ``K_HF = c_full*K_full + c_sr*K_erfc(omega_screen)`` per the CAM
assembly of :func:`vibeqc.periodic_screened_exchange.resolve_periodic_exchange`
(pure DFT skips the K builds entirely; global hybrids keep the
full-range fraction; screened hybrids like hse06 ride the erfc arm;
LR-heavy range-separated functionals fail closed). The per-spin K
blocks come from ``build_jk_2e_real_space`` via
:func:`vibeqc.periodic_screened_exchange.build_exchange_blocks`
(mirrors the UHF multi-k pattern).

Density flow.  Multi-k carries proper LatticeMatrixSets
``D_a_real``, ``D_b_real`` (one-particle, no factor of 2). The
periodic UKS XC kernel :func:`build_xc_periodic_uks` consumes them
directly and returns LatticeMatrixSets ``V_xc^a(g)``, ``V_xc^b(g)``
which are Bloch-summed per k.

Energy formula (mirrors molecular UKS + the multi-k UHF Ewald
convention):

    E_elec  =  E_xc  +  S_k w_k . 1/2 Re tr[(D_a(k) + D_b(k)).H_core(k)]
                     +  S_k w_k . 1/2 Re tr[D_a(k).F_a^{HF}(k)]
                     +  S_k w_k . 1/2 Re tr[D_b(k).F_b^{HF}(k)]

where ``F_s^{HF}(k) = Bloch_k[J - a.K_s]`` is the Hartree-plus-HF
piece of F_s (V_xc reported through E_xc rather than a trace).

Scope.
  * Multi-k open-shell, ``multiplicity >= 1``; integer a/b occupations,
    or per-spin Fermi-Dirac fractional occupations via
    ``options.smearing_temperature`` (separate chemical potentials at
    fixed n_a / n_b; convergence on the free energy A = E - T.S).
  * Pure DFT, hybrid, and HF (a = 1, equivalent to UHF Ewald).
  * Per-spin Pulay DIIS with k-weighted Frobenius inner product.
  * Saunders-Hillier level shift via ``options.level_shift``.
  * Periodic Becke partition selectable via
    ``options.use_periodic_becke``.
  * <S^2> diagnostic on the Γ-block (or first) k-point -- same shortcut
    the multi-k UHF Ewald driver uses.
"""

from __future__ import annotations


from .guess import periodic_result_selection

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    apply_level_shift_k,
    BasisSet,
    bloch_sum,
    BlochKMesh,
    build_grid,
    build_xc_periodic_uks,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    CoulombMethod,
    Functional,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    LevelShiftDensity,
    nuclear_repulsion_per_cell,
    PeriodicKSOptions,
    PeriodicSystem,
    real_space_density_from_kpoints_fractional,
    SCFIteration,
    SpinlockMode,
)
from .ewald_j import auto_grid
from .guess import (
    initial_densities_open_shell,
    periodic_fock_guess_k,
)
from .mom import reorder_occupied_by_max_overlap as _mom_reorder
from .madelung import (
    madelung_energy_correction_for_lat as _madelung_energy_correction_for_lat,
)
from .periodic_fock_multi_k import (
    ewald_3d_j_blocks,
    make_ewald_3d_lattice_j_cache,
    make_slab_ewald_2d_lattice_j_cache,
    slab_ewald_2d_j_blocks,
)
from .periodic_grid import build_periodic_becke_grid
from .periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _damp_lattice_matrix,
    _diag_in_orth_basis,
    _g0_block,
    _MultiKPulayDIIS,
)
from .periodic_uhf_ewald import _spin_squared
from .periodic_k_density import (
    density_matrices_per_k as _density_matrices_per_k,
)
from .periodic_corrected_exchange import (
    CorrectedEwaldExchange,
    corrected_exchange_unavailable_reason,
    is_unsupported_gauge,
)
from .periodic_screened_exchange import (
    PeriodicExchangeAssembly,
    build_exchange_blocks,
    resolve_periodic_exchange,
)
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .smearing.fermi_dirac import (
    fermi_dirac_occupations_per_k as _fermi_dirac_occupations_per_k,
)

__all__ = [
    "PeriodicUKSMultiKEwaldResult",
    "run_uks_periodic_multi_k_ewald3d",
]


@dataclass
class PeriodicUKSMultiKEwaldResult:
    """Result of :func:`run_uks_periodic_multi_k_ewald3d`."""

    energy: float
    e_electronic: float
    e_nuclear: float
    e_xc: float
    e_coulomb: float
    e_hf_exchange: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float

    # a
    mo_energies_alpha: List[np.ndarray]
    mo_coeffs_alpha: List[np.ndarray]
    fock_alpha: List[np.ndarray]
    density_alpha: LatticeMatrixSet

    # b
    mo_energies_beta: List[np.ndarray]
    mo_coeffs_beta: List[np.ndarray]
    fock_beta: List[np.ndarray]
    density_beta: LatticeMatrixSet

    overlap: List[np.ndarray]
    hcore: List[np.ndarray]
    functional: str = ""
    scf_trace: List[SCFIteration] = field(default_factory=list)
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)
    # Smearing diagnostics (zero when smearing_temperature == 0).
    smearing_temperature: float = 0.0
    fermi_level_alpha: float = 0.0
    fermi_level_beta: float = 0.0
    bz_integration: Optional[str] = None
    entropy: float = 0.0
    free_energy: float = 0.0
    # True when hybrid exact exchange ran on a symmetry-reduced
    # Monkhorst-Pack mesh via the Pisani/Dovesi star unfolding; the
    # runner turns this into the kpoint_symmetry_unfolding citation.
    used_kpoint_symmetry_unfolding: bool = False
    # Snapshot of the lattice support actually used after truncation
    # optimization. Analytic derivatives must use this effective support.
    effective_lattice_opts: Optional[LatticeSumOptions] = None
    occupations_alpha: List[np.ndarray] = field(default_factory=list)
    occupations_beta: List[np.ndarray] = field(default_factory=list)

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_mesh: object = None
    restart_kpoints: object = None
    restart_weights: object = None

    # Preserve the physical per-k state independently of lattice cutoff.
    density_alpha_k: Optional[List[np.ndarray]] = None
    density_beta_k: Optional[List[np.ndarray]] = None


def _build_uks_fock_2e_blocks_ewald3d(
    basis: BasisSet,
    system: PeriodicSystem,
    D_alpha_real: LatticeMatrixSet,
    D_beta_real: LatticeMatrixSet,
    omega: float,
    exx,
    lat_opts: LatticeSumOptions,
    grid_shape_t: Tuple[int, int, int],
    origin: Optional[Sequence[float]],
    spacing_bohr: float,
    j_cache=None,
    d_total_buffer: Optional[LatticeMatrixSet] = None,
    full_range_alpha=None,
    symmetry_reduction=None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Per-cell F^{2e,s}(g) = J(D_total, w)(g) - K_HF(D_s)(g) blocks.

    Mirrors :func:`vibeqc.periodic_uhf_multi_k_ewald._build_uhf_fock_blocks_ewald3d`
    with the per-spin K generalised to the CAM assembly ``exx``
    (:class:`vibeqc.periodic_screened_exchange.PeriodicExchangeAssembly`;
    ``None`` or a no-exchange assembly = pure DFT / J-only, K builds
    skipped -- saving the real-space 2e builds per Fock evaluation).

    ``j_cache`` is an optional :class:`EwaldJFTLatticeCache` reused across
    SCF iterations for the analytic-FT Hartree J; ``None`` rebuilds inline
    (unchanged one-shot behaviour).

    ``d_total_buffer`` is an optional reusable ``LatticeMatrixSet``
    container (same cells / nbf) whose blocks are overwritten with
    ``D_a + D_b``; ``None`` builds a fresh container from the overlap
    template (recomputing the overlap lattice integrals just to obtain
    a container -- the one-shot behaviour)."""
    n_cells = len(D_alpha_real.cells)

    # D_total = D_a + D_b as a LatticeMatrixSet.
    D_total_real = (
        d_total_buffer
        if d_total_buffer is not None
        else compute_overlap_lattice(basis, system, lat_opts)
    )
    for g in range(n_cells):
        D_total_real.set_block(
            g,
            np.asarray(D_alpha_real.blocks[g], dtype=float)
            + np.asarray(D_beta_real.blocks[g], dtype=float),
        )

    if lat_opts.coulomb_method == CoulombMethod.SLAB_EWALD_2D:
        J_total_blocks = slab_ewald_2d_j_blocks(
            basis,
            system,
            D_total_real,
            float(omega),
            lattice_opts=lat_opts,
            j_cache=j_cache,
        )
    else:
        # J(D_total) per cell via the shared EWALD_3D J-blocks helper
        # (analytic-FT default -- w-invariant, Bloch-sums to the Γ J at the
        # (1,1,1) mesh; FFT-Poisson split behind VIBEQC_J_EWALD3D_BACKEND=grid).
        J_total_blocks = ewald_3d_j_blocks(
            basis,
            system,
            D_total_real,
            float(omega),
            lattice_opts=lat_opts,
            grid_shape=grid_shape_t,
            origin=origin,
            spacing_bohr=spacing_bohr,
            j_cache=j_cache,
        )

    if exx is None or not exx.needs_exchange:
        F_alpha_blocks: List[np.ndarray] = []
        F_beta_blocks: List[np.ndarray] = []
        for g in range(n_cells):
            J_total = np.asarray(J_total_blocks[g], dtype=float)
            F_alpha_blocks.append(J_total)
            F_beta_blocks.append(J_total)
        return F_alpha_blocks, F_beta_blocks

    # Hybrid path: per-spin coefficient-folded K blocks (full-range
    # and/or erfc-screened per the CAM assembly). Same builders +
    # omega convention as the Γ driver, so the (1,1,1) Bloch sum
    # matches it. Equality of the full-range routes is pinned in
    # tests/test_periodic_uhf_multi_k_ewald.py.
    K_alpha_blocks = build_exchange_blocks(
        basis, system, lat_opts, D_alpha_real, exx,
        full_range_alpha=full_range_alpha,
        symmetry_reduction=symmetry_reduction,
    )
    K_beta_blocks = build_exchange_blocks(
        basis, system, lat_opts, D_beta_real, exx,
        full_range_alpha=full_range_alpha,
        symmetry_reduction=symmetry_reduction,
    )

    F_alpha_blocks = []
    F_beta_blocks = []
    for g in range(n_cells):
        J_total = np.asarray(J_total_blocks[g], dtype=float)
        F_alpha_blocks.append(J_total - K_alpha_blocks[g])
        F_beta_blocks.append(J_total - K_beta_blocks[g])

    return F_alpha_blocks, F_beta_blocks


def _bloch_sum_blocks(
    blocks: Sequence[np.ndarray],
    cells,
    k_cart: np.ndarray,
) -> np.ndarray:
    k = np.asarray(k_cart, dtype=float).reshape(3)
    F_k = np.zeros_like(blocks[0], dtype=complex)
    for g_idx, block in enumerate(blocks):
        R_g = np.asarray(cells[g_idx].r_cart, dtype=float)
        phase = np.exp(1j * float(np.dot(k, R_g)))
        F_k = F_k + phase * block
    return F_k


def _bloch_sum_lms_at_k(
    lms: LatticeMatrixSet,
    k_cart: np.ndarray,
) -> np.ndarray:
    return np.asarray(bloch_sum(lms, np.asarray(k_cart, dtype=float).reshape(3)))


def run_uks_periodic_multi_k_ewald3d(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    options: Optional[PeriodicKSOptions] = None,
    *,
    omega: float = 0.0,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    canonical_orth_normalize_diag_first: bool = True,
    auto_optimize_truncation: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    initial_density_k: Optional[Sequence] = None,
    verbose: Optional[int] = None,
    bz_integration: Optional[str] = None,
) -> PeriodicUKSMultiKEwaldResult:
    """Multi-k open-shell periodic Kohn-Sham SCF with EWALD_3D Coulomb.

    Parameters
    ----------
    system, basis, kmesh
        Periodic system, AO basis, k-mesh.
    options
        Optional :class:`PeriodicKSOptions`. Reads ``functional``,
        ``grid``, ``use_periodic_becke``, ``becke_image_radius_bohr``,
        ``level_shift``, ``damping``, ``max_iter``, ``conv_tol_*``,
        ``diis_*``, ``initial_guess``, ``lattice_opts``.
        Positive ``smearing_temperature`` enables separate per-spin
        Fermi-Dirac occupations at fixed ``n_alpha`` / ``n_beta``.

    Returns
    -------
    :class:`PeriodicUKSMultiKEwaldResult`.
    """
    opts = options if options is not None else PeriodicKSOptions()
    from .guess import select_periodic_driver_guess
    guess = select_periodic_driver_guess(
        system, opts, route="ewald", method="UKS", driver="run_uks_periodic_multi_k_ewald3d",
        multi_k=True, restart_supplied=initial_density_k is not None,
    ).transport
    if initial_density_k is not None:
        if len(kmesh.kpoints) != int(np.prod(kmesh.mesh)):
            raise NotImplementedError("READ requires a complete full k mesh on the Ewald lattice adapter")
        if getattr(opts, "atomic_spins", None):
            raise ValueError("atomic_spins requires SAD without a restart density")
        if len(initial_density_k) != 2:
            raise ValueError("open-shell per-k READ requires alpha and beta block lists")
    if (getattr(opts, "initial_guess", None) == InitialGuess.READ) and initial_density_k is None:
        raise NotImplementedError(
            "periodic READ requires initial_density_k with a complete per-k "
            "density payload. Use run_periodic_job(read_from=...) to load "
            "a compatible result or archive; see docs/user_guide/initial_guess.md."
        )
    # SPINLOCK: this multi-k UKS driver implements PATTERN_HOLD (the AFM-seed
    # protection mode); SPIN_SCHEDULE's two-phase is Γ-Ewald-only in v1.
    from .spinlock_periodic import check_spinlock_support
    check_spinlock_support(
        opts, {SpinlockMode.PATTERN_HOLD}, "the multi-k UKS Ewald driver")
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError(
            "run_uks_periodic_multi_k_ewald3d: smearing_temperature must be >= 0"
        )

    # BZ-integration backend (opt-in). None / "smearing" keeps the temperature
    # path; "gilat" selects the Gilat-Raubenheimer net (computed per spin,
    # g=1.0) on the full or IBZ mesh. GR is T=0, so no finite smearing.
    if bz_integration not in (None, "smearing", "gilat"):
        raise ValueError(
            "run_uks_periodic_multi_k_ewald3d: bz_integration must be None, "
            f"'smearing', or 'gilat'; got {bz_integration!r}"
        )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise ValueError(
            "run_uks_periodic_multi_k_ewald3d: bz_integration='gilat' is a "
            "T=0 integrator; do not combine it with smearing_temperature > 0"
        )
    # Finite-T smearing and GR both give fractional occupations -> the
    # fractional per-spin density path (vs integer Aufbau slicing).
    use_fractional_density = (smearing_T > 0.0) or use_gilat
    # Fermi-Dirac smearing is supported per spin (separate chemical
    # potentials at fixed n_alpha / n_beta, mirroring the BIPOLE UKS
    # driver); convergence then runs on the free energy A = E - T.S.
    lat_opts: LatticeSumOptions = opts.lattice_opts
    slab_mode = lat_opts.coulomb_method == CoulombMethod.SLAB_EWALD_2D
    if slab_mode and system.dim != 2:
        raise ValueError(
            "run_uks_periodic_multi_k_ewald3d: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}"
        )
    plog = resolve_progress(progress, verbose=verbose)

    # ---- Force EWALD_3D gauge (gauge consistency; handover F4 2026-06-01) ----
    # This driver hard-codes the Hartree J to the Ewald-3D builder, so V_ne
    # (compute_nuclear_lattice_dispatch) and e_nuc (nuclear_repulsion_per_cell)
    # MUST share that gauge. Without the force, a default options object
    # (coulomb_method=DIRECT_TRUNCATED) makes nuclear_repulsion_per_cell return
    # the molecular 1/d sum and madelung_energy_correction_for_lat the bare-gauge
    # +a_M.Q_e^2/2L term; those only partially cancel (~0.74 mHa on H2/30-bohr),
    # so the SCF converged to a non-physical energy with no warning (CLAUDE.md
    # Sec.7). The RHF multi-k sibling already forces this (audit F1); extending it
    # here aligns e_nuclear with run_uks_periodic_gamma_ewald3d and zeroes the
    # now-redundant Madelung term (madelung_energy_correction_for_lat returns
    # 0.0 for EWALD_3D).
    # EWALD_3D V_ne (compute_nuclear_lattice_dispatch) is implemented only for
    # dim == 3 -- the 1D/2D Ewald variants raise (periodic_v_ne.py). So gate the
    # force on dim == 3; low-dim cells keep their DIRECT_TRUNCATED gauge (the
    # historical behaviour for these drivers on 1D/2D chains).
    if system.dim == 3 and lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
        plog.info(
            "coulomb_method forced to EWALD_3D for gauge consistency "
            f"(was {lat_opts.coulomb_method!r}); this driver's Hartree J "
            "is Ewald-3D and V_ne / e_nuc must match"
        )
        lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    # w must match the nuclear Ewald a (auto-selected from
    # nuclear_cutoff_bohr in the C++ ewald engine) so the jellium
    # background terms cancel exactly. Mirrors the override block in
    # the sibling Ewald drivers (commit 49f8ae91 / 433d3543). The
    # driver kwarg ``omega`` is retained for signature parity but is
    # overridden here; users override via ``opts.ewald_omega``.
    _ewald_tol = getattr(opts, "ewald_tolerance", 1e-12)
    _cutoff = getattr(opts, "ewald_cutoff_bohr", lat_opts.nuclear_cutoff_bohr)
    if slab_mode:
        if omega <= 0.0:
            omega = float(getattr(lat_opts, "slab_ewald_alpha", 0.4))
        if omega <= 0.0:
            omega = 0.4
        lat_opts.slab_ewald_alpha = float(omega)
    elif omega <= 0.0:
        _user_omega = getattr(opts, "ewald_omega", None)
        if _user_omega is not None and float(_user_omega) > 0.0:
            omega = float(_user_omega)
        else:
            from .pbc_bipole_common import default_ewald_alpha

            V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
            # GitLab #651: CRYSTAL's 2.8/V^(1/3), bounded below so that
            # erfc(omega r)/r has decayed to opts.ewald_tolerance at this
            # route's fixed real-space image cutoff (lat_opts.cutoff_bohr).
            omega = default_ewald_alpha(
                V_cell,
                real_cutoff_bohr=float(lat_opts.cutoff_bohr),
                tolerance=float(getattr(opts, "ewald_tolerance", 1e-12)),
            )

    lat = np.asarray(system.lattice, dtype=float)

    if slab_mode:
        grid_shape_t = (0, 0, 0)
    elif grid_shape is None:
        grid_shape_t = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape_t = (grid_shape, grid_shape, grid_shape)
    else:
        grid_shape_t = tuple(int(x) for x in grid_shape)
    if slab_mode:
        plog.info(
            f"UKS multi-k SLAB_EWALD_2D / functional={opts.functional!r}, "
            f"alpha = {float(omega):.3f}"
        )
    else:
        plog.info(
            f"UKS multi-k EWALD_3D / functional={opts.functional!r}, "
            f"omega = {float(omega):.3f}, "
            f"FFT grid {grid_shape_t[0]}x{grid_shape_t[1]}x{grid_shape_t[2]}"
        )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    from .options_dump import dump_active_settings

    dump_active_settings(
        plog,
        [
            ("PeriodicKSOptions", opts),
            ("LatticeSumOptions", lat_opts),
            (
                "Driver kwargs",
                {
                    "omega": float(omega),
                    "grid_shape": grid_shape_t,
                    "origin": origin,
                    "spacing_bohr": float(spacing_bohr),
                    "linear_dep_threshold": float(linear_dep_threshold),
                    "canonical_orth_normalize_diag_first": canonical_orth_normalize_diag_first,
                    "auto_optimize_truncation": auto_optimize_truncation,
                },
            ),
        ],
    )
    if plog.level >= 5:
        from .scf_log import format_basis_summary

        plog.write_raw(format_basis_summary(basis))

    n_elec = int(system.n_electrons())
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(
            f"run_uks_periodic_multi_k_ewald3d: multiplicity must be >= 1, got {mult}"
        )
    if (n_elec + mult - 1) % 2 != 0 or (n_elec - mult + 1) % 2 != 0:
        raise ValueError(
            f"run_uks_periodic_multi_k_ewald3d: (n_electrons={n_elec}, "
            f"multiplicity={mult}) cannot be split into integer a/b."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2

    # ---- Functional + DFT grid ------------------------------------------
    func = Functional(opts.functional, 2)  # spin-polarized
    # CAM exchange assembly (screened hybrids like hse06 ride the erfc
    # arm; LR-heavy RSH fails closed). omega_screen is unrelated to
    # this driver's ``omega`` (the Ewald split alpha).
    exx = resolve_periodic_exchange(
        func, where="run_uks_periodic_multi_k_ewald3d"
    )

    if opts.use_periodic_becke:
        grid = build_periodic_becke_grid(
            system,
            grid_options=opts.grid,
            image_radius_bohr=float(opts.becke_image_radius_bohr),
        )
    else:
        grid = build_grid(system.unit_cell_molecule(), opts.grid)

    k_points = list(kmesh.kpoints)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = len(k_points)
    if n_k == 0:
        raise ValueError("kmesh has no k-points")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"kmesh.weights must sum to 1; got {weights.sum():.6f}")
    plog.info(
        f"k-mesh: {n_k} k-point{'s' if n_k != 1 else ''}, "
        f"weights sum = {weights.sum():.4f}; "
        f"n_alpha = {n_alpha}, n_beta = {n_beta}"
    )

    # ---- Auto-optimise lattice truncation (default ON) -------------------
    if auto_optimize_truncation and lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
        from .eigs_preflight import (
            format_truncation_optimization_report,
            optimize_truncation,
        )

        k_arr = [np.asarray(k, dtype=float) for k in k_points]
        opt_rep = optimize_truncation(
            system,
            basis,
            lattice_opts=lat_opts,
            k_points_cart=k_arr,
        )
        if (
            opt_rep.n_evaluations > 1
            or opt_rep.optimized_lattice_opts.cutoff_bohr != lat_opts.cutoff_bohr
        ):
            plog.write_raw(format_truncation_optimization_report(opt_rep))
            if not opt_rep.converged:
                plog.warn("auto_optimize_truncation did not converge.")
            lat_opts = opt_rep.optimized_lattice_opts

    with plog.stage(
        "integrals_lattice", detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    ):
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        if slab_mode:
            from .periodic_v_ne_slab import build_v_ne_slab_ewald_2d_k_cache

            slab_vne_cache = build_v_ne_slab_ewald_2d_k_cache(
                basis,
                system,
                lat_opts,
                alpha=float(omega),
            )
            V_lat = None
        else:
            from .periodic_v_ne import compute_nuclear_lattice_dispatch

            V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    cells = list(S_lat.cells)
    n_cells = len(cells)

    # Fused C++ multi-k Bloch transforms (OpenMP-parallel over k) when
    # available; the per-k Python fold is the fallback. Consumed by the
    # per-iteration F_HF^s(g) -> F_HF^s(k) transforms in the SCF loop
    # below (same per-block accumulation order as ``_bloch_sum_blocks``).
    cell_r_list = [np.asarray(c.r_cart, dtype=float) for c in cells]
    k_arr_list = [np.asarray(k, dtype=float).reshape(3) for k in k_points]
    try:
        from ._vibeqc_core import bloch_sum_multi_k as _bloch_mk
    except ImportError:
        _bloch_mk = None

    S_k_list: List[np.ndarray] = []
    Hcore_k_list: List[np.ndarray] = []
    X_k_list: List[np.ndarray] = []
    # Per-k linear-dependence preflight; see periodic_rhf_multi_k_ewald
    # for the rationale (Searle et al., ARCHER eCSE04-16, 2017).
    from .linear_dependence import (
        PERIODIC_OVERLAP_HINT,
        scf_preflight_overlap_check,
    )

    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        if slab_mode:
            from .periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_k_matrix

            V_k = compute_v_ne_slab_ewald_2d_k_matrix(
                basis,
                system,
                lat_opts,
                k_arr,
                alpha=float(omega),
                cache=slab_vne_cache,
            )
        else:
            V_k = np.asarray(bloch_sum(V_lat, k_arr))
        H_k = T_k + V_k
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        scf_preflight_overlap_check(
            S_k,
            plog=plog,
            label=f"S(k={k_idx}, k_cart={k_arr.round(4).tolist()})",
            basis=basis,
            remediation_hint=PERIODIC_OVERLAP_HINT,
        )
        X_k, n_kept = _canonical_orthogonalizer_complex(
            S_k,
            linear_dep_threshold,
            normalize_diag_first=canonical_orth_normalize_diag_first,
        )
        if max(n_alpha, n_beta) > n_kept:
            raise RuntimeError(
                f"run_uks_periodic_multi_k_ewald3d: orth dropped too many "
                f"directions (n_a={n_alpha}, n_b={n_beta}, "
                f"n_kept={n_kept}) at k = {k_arr}"
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

    if (
        guess in (InitialGuess.SAP, InitialGuess.HUECKEL)
        and getattr(opts, "atomic_spins", None)
    ):
        raise RuntimeError(
            "GuessEngine: atomic_spins (ATOMSPIN broken-symmetry seed) "
            f"requires the SAD guess; resolved guess is {guess.name}"
        )
    fock_guess_per_k = guess in (InitialGuess.SAP, InitialGuess.HUECKEL)
    guess_fock_k = Hcore_k_list
    if fock_guess_per_k:
        guess_fock_k = list(
            periodic_fock_guess_k(
                system,
                basis,
                k_points,
                guess,
                lattice_opts=lat_opts,
                kinetic_lattice=T_lat,
                overlap_lattice=S_lat,
            )
        )

    # T_lat / V_lat are folded into Hcore(k), and SAP has consumed T_lat when
    # requested. Free the per-cell one-electron lattice integrals before the
    # SCF iterations (S_lat is still needed for its g0 block).
    del T_lat, V_lat

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    # ---- Initial guess --------------------------------------------------
    C_alpha_per_k: List[np.ndarray] = []
    eps_alpha_per_k: List[np.ndarray] = []
    C_beta_per_k: List[np.ndarray] = []
    eps_beta_per_k: List[np.ndarray] = []
    for F_guess_k, X_k in zip(guess_fock_k, X_k_list):
        C_a, eps_a = _diag_in_orth_basis(F_guess_k, X_k)
        C_b, eps_b = _diag_in_orth_basis(F_guess_k, X_k)
        C_alpha_per_k.append(C_a.astype(complex))
        eps_alpha_per_k.append(eps_a)
        C_beta_per_k.append(C_b.astype(complex))
        eps_beta_per_k.append(eps_b)

    def _integer_occupations(nbf: int, n_occ_each: int) -> List[np.ndarray]:
        occ_per_k = []
        for _ in range(n_k):
            occ = np.zeros(nbf, dtype=float)
            occ[:n_occ_each] = 1.0
            occ_per_k.append(occ)
        return occ_per_k

    def _occupations_per_spin(
        eps_spin_per_k: Sequence[np.ndarray],
        n_spin: int,
    ) -> Tuple[List[np.ndarray], float, float]:
        """Per-spin fractional occupations via Fermi-Dirac (mirrors the
        BIPOLE UKS driver): the closed-shell FD helper targets 2.n_spin
        electrons and returns occupations in [0, 2]; halving gives the
        single-spin occupations in [0, 1] at the same per-spin chemical
        potential, and the entropy halves with them. Returns
        (occ_per_k, mu_spin, entropy_spin)."""
        nbf = eps_spin_per_k[0].shape[0] if eps_spin_per_k else 0
        if use_gilat and n_spin > 0:
            # Gilat-Raubenheimer net for this spin channel (g=1.0, n_spin
            # electrons; sharp Fermi surface -> entropy 0). Handles full or
            # IBZ meshes via the shared auto-dispatch entry point.
            from .bz_integration import gilat_occupations_for_kmesh

            occ_gr, ef_gr = gilat_occupations_for_kmesh(
                system, kmesh, eps_spin_per_k, float(n_spin), spin_degeneracy=1.0
            )
            return occ_gr, float(ef_gr), 0.0
        if smearing_T <= 0.0 or n_spin == 0:
            return _integer_occupations(nbf, n_spin), 0.0, 0.0
        occ_double, mu, entropy_double = _fermi_dirac_occupations_per_k(
            eps_spin_per_k,
            weights,
            float(2 * n_spin),
            smearing_T,
        )
        occ = [np.asarray(o, dtype=float) * 0.5 for o in occ_double]
        return occ, float(mu), float(entropy_double) * 0.5

    def _spin_density(C_per_k_local, occ_per_k_local):
        """One-particle (no factor 2) real-space spin density via the
        C++ fractional-occupation builder. Mirrors the UHF multi-k
        Ewald driver convention."""
        return real_space_density_from_kpoints_fractional(
            C_per_k_local,
            occ_per_k_local,
            kmesh,
            cells,
        )

    occ_alpha_per_k, mu_alpha, entropy_alpha = _occupations_per_spin(
        eps_alpha_per_k, n_alpha
    )
    occ_beta_per_k, mu_beta, entropy_beta = _occupations_per_spin(
        eps_beta_per_k, n_beta
    )
    entropy = entropy_alpha + entropy_beta
    D_alpha_real = _spin_density(C_alpha_per_k, occ_alpha_per_k)
    D_beta_real = _spin_density(C_beta_per_k, occ_beta_per_k)

    initial_alpha_k = None
    initial_beta_k = None

    # Density-mode guesses: overwrite per-spin densities at g=0 with
    # the engine output (proportional spin split -- matches molecular
    # UHF behaviour). Closed-shell-like cases (n_a == n_b) reduce to
    # the even split that this driver previously used inline.
    if initial_density_k is not None:
        plog.info("initial guess: READ (complete per-k spin density)")
    elif fock_guess_per_k:
        # Common one-particle orbitals are occupied through the route's
        # existing per-spin global-k Aufbau/smearing policy.
        plog.info(
            f"initial guess: {guess.name} "
            "(one-particle Hamiltonian diagonalise at each k)"
        )
    else:
        seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
        split = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha,
            n_beta,
            seed_guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=lat_opts,
            # ATOMSPIN: per-atom +1/-1/0 seed -> broken-symmetry g=0 density
            # (Bloch-sums to a broken-symmetry D(k)). Empty/None = symmetric.
            atomic_spins=getattr(opts, "atomic_spins", None) or None,
            overlap=S_k_list, weights=weights,
        )
        if split is not None:
            plog.info(
                f"initial guess: {guess.name} "
                "(spin-split density via GuessEngine)"
            )
            D_a_sad, D_b_sad = split
            initial_alpha_k = [np.asarray(D_a_sad, dtype=complex).copy() for _ in k_points]
            initial_beta_k = [np.asarray(D_b_sad, dtype=complex).copy() for _ in k_points]
            zero_block = np.zeros_like(D_a_sad, dtype=float)
            for g_idx in range(len(D_alpha_real.cells)):
                is_g0 = (
                    D_alpha_real.cells[g_idx].index == np.array([0, 0, 0])
                ).all()
                D_alpha_real.set_block(
                    g_idx, D_a_sad if is_g0 else zero_block
                )
                D_beta_real.set_block(
                    g_idx, D_b_sad if is_g0 else zero_block
                )
        else:
            plog.info(
                f"initial guess: {guess.name} "
                "(Hcore-diagonalise at each k, spin-degenerate)"
            )

    # Cache the iteration-invariant per-cell Hartree-J machinery once. PATOM
    # consumes the same cache for its one HF-like in-field step before SCF.
    if slab_mode:
        j_cache = make_slab_ewald_2d_lattice_j_cache(
            basis,
            system,
            D_alpha_real.cells,
            lattice_opts=lat_opts,
            alpha=float(omega),
        )
    else:
        j_cache = make_ewald_3d_lattice_j_cache(
            basis, system, D_alpha_real.cells, lattice_opts=lat_opts,
        )
    # Reusable D_total container: the LatticeMatrixSet structure (cells,
    # nbf) is fixed for the whole SCF, so build the container once from
    # the overlap template and overwrite every block per Fock build.
    # Rebuilding it per iteration recomputed the overlap lattice
    # integrals just to obtain a container.
    d_total_buffer = compute_overlap_lattice(basis, system, lat_opts)

    # ---- Corrected-Ewald full-range exchange ------------------------------
    # Only the full-Coulomb (c_full) arm needs it: summed as a bare 1/r
    # kernel over a radial image ball it does not converge, because the SCF
    # density on a finite k mesh is BvK-torus periodic and D(g) does not
    # decay. Gate on c_full, NOT on needs_exchange -- a screened hybrid such
    # as hse06 is c_full = 0, c_sr = 0.25, i.e. entirely the erfc arm, which
    # is short-ranged and convergent already; keying on needs_exchange would
    # fail-close every dim=2 hse06 job for no reason.
    corrected_exchange = None
    if exx.c_full != 0.0:
        _unavailable = corrected_exchange_unavailable_reason(
            system, kmesh, slab_mode=slab_mode
        )
        if _unavailable is None:
            corrected_exchange = CorrectedEwaldExchange.build(
                basis,
                system,
                np.asarray(
                    [np.asarray(c.r_cart, dtype=float) for c in cells]
                ),
                kmesh,
                float(omega),
                where="run_uks_periodic_multi_k_ewald3d",
            )
        elif is_unsupported_gauge(system, slab_mode=slab_mode):
            # A gauge the split does not exist for (2D slab, dim != 3).
            # Live, pre-dates the fix: keep the historical kernel and say
            # so. Remaining gap, recorded in HANDOVER_OPEN_BUGS_V015.md.
            plog.info(
                "full-range exact exchange stays on the uncorrected "
                f"real-space image sum: {_unavailable}. Energies on this "
                "route are not lattice-cutoff converged."
            )
        else:
            # A mesh this code cannot serve yet. Fail closed rather than
            # let an IBZ run disagree with the same job on a full mesh.
            raise NotImplementedError(
                "run_uks_periodic_multi_k_ewald3d: this functional needs "
                f"full-range exact exchange (c_full = {exx.c_full:g}), but "
                f"{_unavailable}. Use a Monkhorst-Pack mesh (full, or "
                "symmetry-reduced with vq.attach_symmetry + "
                "use_symmetry=True), or a pure/screened functional."
            )



    # Wedge handle: set either by the corrected split (c_full path, which
    # carries its own unfolding) or standalone for a screened hybrid
    # (c_sr-only, e.g. hse06) on a supported symmetry-reduced MP mesh.
    # The Pisani/Dovesi density transformation applies to 2D slabs and
    # 3D crystals alike. The c_sr arm
    # has no k-space pieces, but its real-space K_erfc is FIRST order in
    # the density, and the plain weighted wedge fold uses each
    # representative in place of its orbit average -- a truncation-
    # asymmetry-level error on shift-bearing cells that pure DFT (second
    # order) never showed. The unfolded fold fixes that, and it also
    # unlocks the SYM3b orbit reduction for the 3D K_erfc build. Slabs
    # retain the raw truncated-operator convention used by their full mesh.
    wedge_unfolding = (
        corrected_exchange.unfolding
        if corrected_exchange is not None
        else None
    )
    if (
        wedge_unfolding is None
        and exx.c_sr != 0.0
        and (
            (slab_mode and int(system.dim) == 2)
            or (not slab_mode and int(system.dim) == 3)
        )
    ):
        _n_input = len(list(kmesh.kpoints))
        _mesh_dims = tuple(int(x) for x in getattr(kmesh, "mesh", (1, 1, 1)))
        _ir = np.asarray(
            getattr(kmesh, "ir_mapping", []), dtype=int
        ).reshape(-1)
        _n_full = int(np.prod(_mesh_dims))
        if _n_full > _n_input:
            _operations = getattr(
                getattr(system, "symmetry", None), "operations", None
            )
            if _ir.size != _n_full or not _operations:
                raise NotImplementedError(
                    "run_uks_periodic_multi_k_ewald3d: cannot reconstruct "
                    "the full-star density for this symmetry-reduced "
                    "Monkhorst-Pack mesh. Reattach current symmetry data "
                    "or run the corresponding full mesh."
                )
            from .periodic_k_symmetry import KMeshUnfolding

            try:
                wedge_unfolding = KMeshUnfolding.build(system, basis, kmesh)
            except ValueError as exc:
                # A representative-only fold is not a physical fallback:
                # it replaces each star member by the IBZ representative.
                raise NotImplementedError(
                    "run_uks_periodic_multi_k_ewald3d: cannot reconstruct "
                    "the full-star density for this symmetry-reduced "
                    "Monkhorst-Pack mesh. Reattach current symmetry data "
                    "or run the corresponding full mesh."
                ) from exc

    # SYM3b orbit reduction for the real-space exchange arms, wedge path
    # only (see the RHF driver's comment for the profile and the
    # symmetrized-answer rationale; here it applies per spin -- two
    # reduced builds per iteration, same mapping).
    exchange_symmetry_reduction = None
    if exx.c_full != 0.0 or exx.c_sr != 0.0:
        if wedge_unfolding is not None and not slab_mode:
            from ._vibeqc_core import direct_lattice_cells as _radial_cells
            from .bipole_symmetry_fock import (
                cell_orbit_mapping,
                representative_cell_indices,
            )

            # The mapping must index the K builder's OWN radial cell list
            # (direct_lattice_cells at the 2e cutoff), not the overlap
            # template ``cells``: under LatticeSumOptions.pair_complete_1e
            # (#429) the template is longer. Identical lists with the switch
            # off, so byte-identical there.
            _sym_mapping = cell_orbit_mapping(
                system, basis, list(_radial_cells(system, lat_opts.cutoff_bohr))
            )
            if _sym_mapping is not None:
                exchange_symmetry_reduction = (
                    _sym_mapping,
                    representative_cell_indices(_sym_mapping),
                )

    if guess == InitialGuess.PATOM:
        plog.info("initial guess: PATOM (SAD + one periodic in-field step)")
        F_alpha_blocks, F_beta_blocks = _build_uks_fock_2e_blocks_ewald3d(
            basis,
            system,
            D_alpha_real,
            D_beta_real,
            omega,
            # PATOM seed uses one full-HF in-field step regardless of
            # the functional (same convention as the Γ driver).
            PeriodicExchangeAssembly(1.0, 0.0, 0.0),
            lat_opts,
            grid_shape_t,
            origin,
            spacing_bohr,
            j_cache=j_cache,
            d_total_buffer=d_total_buffer,
        )
        C_alpha_per_k = []
        eps_alpha_per_k = []
        C_beta_per_k = []
        eps_beta_per_k = []
        for k_idx, k_cart in enumerate(k_points):
            F_a = _bloch_sum_blocks(F_alpha_blocks, cells, np.asarray(k_cart))
            F_b = _bloch_sum_blocks(F_beta_blocks, cells, np.asarray(k_cart))
            C_a, eps_a = _diag_in_orth_basis(
                F_a + Hcore_k_list[k_idx],
                X_k_list[k_idx],
            )
            C_b, eps_b = _diag_in_orth_basis(
                F_b + Hcore_k_list[k_idx],
                X_k_list[k_idx],
            )
            C_alpha_per_k.append(C_a)
            eps_alpha_per_k.append(eps_a)
            C_beta_per_k.append(C_b)
            eps_beta_per_k.append(eps_b)
        occ_alpha_per_k, mu_alpha, entropy_alpha = _occupations_per_spin(
            eps_alpha_per_k, n_alpha
        )
        occ_beta_per_k, mu_beta, entropy_beta = _occupations_per_spin(
            eps_beta_per_k, n_beta
        )
        entropy = entropy_alpha + entropy_beta
        D_alpha_real = _spin_density(C_alpha_per_k, occ_alpha_per_k)
        D_beta_real = _spin_density(C_beta_per_k, occ_beta_per_k)

    if guess == InitialGuess.PATOM:
        initial_alpha_k = None
        initial_beta_k = None

    if initial_density_k is not None:
        from .guess import periodic_restart_lattice_density, normalize_spin_density_k_guess
        initial_alpha_k, initial_beta_k = normalize_spin_density_k_guess(
            *initial_density_k, S_k_list, weights, n_alpha, n_beta,
        )
        D_alpha_real = periodic_restart_lattice_density(
            initial_alpha_k, S_k_list, weights, n_alpha, kmesh, D_alpha_real.cells, retained_k_system=system,
        )
        D_beta_real = periodic_restart_lattice_density(
            initial_beta_k, S_k_list, weights, n_beta, kmesh, D_beta_real.cells, retained_k_system=system,
        )
    D_alpha_prev: Optional[LatticeMatrixSet] = None
    D_beta_prev: Optional[LatticeMatrixSet] = None
    # Damped per-spin per-k densities, mirrored in lock-step with the
    # real-space damping above so the corrected-exchange k-space arms and
    # the real-space builds consume one identical density. Folding is
    # linear, so damping per-k with the same coefficients equals damping
    # the folds.
    D_a_used_per_k_prev: Optional[List[np.ndarray]] = None
    D_b_used_per_k_prev: Optional[List[np.ndarray]] = None

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_uks_periodic_multi_k_ewald3d: damping must be in "
            f"[0, 1); got {damping}"
        )

    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    # Single spin-coupled Pulay history over the concatenated a + b
    # per-k block lists (one coefficient set extrapolates both spins;
    # see the _AcceleratorState note in periodic_scf_accelerators.py).
    diis = (
        _MultiKPulayDIIS(max_subspace=int(opts.diis_subspace_size))
        if use_diis
        else None
    )
    level_shift = float(getattr(opts, "level_shift", 0.0))

    # SPINLOCK PATTERN_HOLD (open-shell magnetic convergence): hold the seeded
    # broken-symmetry occupied subspace per-k per-spin by maximum overlap (MOM)
    # with the previous cycle for the first ``spinlock_iterations`` cycles, then
    # release to aufbau -- protecting an ATOMSPIN seed from collapsing to the
    # symmetric solution on the multi-k path. Aufbau-mode only; skipped under
    # fractional smearing. (SPIN_SCHEDULE is handled as a two-phase run by
    # run_periodic_job, which restarts the released phase from the locked one.)
    spinlock_mode = getattr(opts, "spinlock_mode", SpinlockMode.OFF)
    spinlock_iterations = int(getattr(opts, "spinlock_iterations", 0))
    _pattern_hold = (
        spinlock_mode == SpinlockMode.PATTERN_HOLD
        and spinlock_iterations > 0
        and smearing_T == 0.0
    )
    # Previous-cycle occupied MOs per k per spin, for the MOM hold.
    C_alpha_occ_prev: List[Optional[np.ndarray]] = [None] * n_k
    C_beta_occ_prev: List[Optional[np.ndarray]] = [None] * n_k

    # Phase C1c -- quadratic SCF fallback (per-spin per-k Newton step).
    quadratic_fallback_iter = int(getattr(opts, "quadratic_fallback_iter", 0))
    quadratic_fallback_shift = float(getattr(opts, "quadratic_fallback_shift", 0.1))
    quadratic_fallback_max_step = float(
        getattr(opts, "quadratic_fallback_max_step", 0.1)
    )

    # ---- SCF loop -------------------------------------------------------
    scf_trace: List[SCFIteration] = []
    E_prev = 0.0
    F_alpha_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    F_beta_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    F_HF_alpha_k_list: List[np.ndarray] = list(F_alpha_k_list)
    F_HF_beta_k_list: List[np.ndarray] = list(F_beta_k_list)
    E_xc = 0.0
    E_coulomb_per_cell = 0.0
    E_hf_K_per_cell = 0.0
    scf_label = "SLAB_EWALD_2D" if slab_mode else "EWALD_3D"
    plog.banner(f"SCF (UKS multi-k {opts.functional!r}, {scf_label})")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    converged = False
    terminal_recheck_at_cap = False
    iter_idx = 0

    for iter_idx in range(1, int(opts.max_iter) + 1):
        # SPINLOCK PATTERN_HOLD: DIIS is suspended (no history recorded, no
        # extrapolation, damping stays live) while the hold is active. Fock
        # extrapolation across held-window iterates steers the SCF toward the
        # symmetric attractor by continuous orbital rotation -- a collapse the
        # occupation-selecting MOM hold cannot see -- and poisons the
        # post-release history with out-of-basin iterates. The history starts
        # fresh at release.
        hold_active = _pattern_hold and iter_idx <= spinlock_iterations
        diis_active = use_diis and iter_idx >= diis_start_iter and not hold_active

        D_a_cur_per_k = (
            initial_alpha_k if iter_idx == 1 and initial_alpha_k is not None
            else _density_matrices_per_k(C_alpha_per_k, occ_alpha_per_k)
        )
        D_b_cur_per_k = (
            initial_beta_k if iter_idx == 1 and initial_beta_k is not None
            else _density_matrices_per_k(C_beta_per_k, occ_beta_per_k)
        )
        if iter_idx > 1 and damping > 0.0 and not diis_active:
            D_alpha_used = _damp_lattice_matrix(
                D_alpha_real,
                D_alpha_prev,
                damping,
            )
            D_beta_used = _damp_lattice_matrix(
                D_beta_real,
                D_beta_prev,
                damping,
            )
            D_a_used_per_k = [
                damping * D_prev + (1.0 - damping) * D_cur
                for D_cur, D_prev in zip(
                    D_a_cur_per_k, D_a_used_per_k_prev
                )
            ]
            D_b_used_per_k = [
                damping * D_prev + (1.0 - damping) * D_cur
                for D_cur, D_prev in zip(
                    D_b_cur_per_k, D_b_used_per_k_prev
                )
            ]
        else:
            D_alpha_used = D_alpha_real
            D_beta_used = D_beta_real
            D_a_used_per_k = D_a_cur_per_k
            D_b_used_per_k = D_b_cur_per_k
        if wedge_unfolding is not None:
            # Symmetry-reduced wedge on the hybrid path (full-range OR
            # screened): the weighted wedge fold behind D_*_used is
            # wrong at multi-cell cutoffs (representative in place of
            # orbit average). J, K and XC all consume these sets, so
            # rebuild them exactly from the unfolded wedge densities
            # (Pisani, Dovesi & Roetti 1988, Eqs. II.7.21 / II.7.48).
            D_alpha_used = wedge_unfolding.fold_density(
                D_a_used_per_k, lat_opts
            )
            D_beta_used = wedge_unfolding.fold_density(
                D_b_used_per_k, lat_opts
            )

        # Per-spin 2e Fock blocks F^{2e,s}(g) = J(D_total) - K_HF(D_s).
        F_HF_alpha_blocks, F_HF_beta_blocks = _build_uks_fock_2e_blocks_ewald3d(
            basis,
            system,
            D_alpha_used,
            D_beta_used,
            omega,
            exx,
            lat_opts,
            grid_shape_t,
            origin,
            spacing_bohr,
            j_cache=j_cache,
            d_total_buffer=d_total_buffer,
            full_range_alpha=(
                None if corrected_exchange is None else float(omega)
            ),
            symmetry_reduction=exchange_symmetry_reduction,
        )

        # Periodic UKS XC: V_xc^s(g) lattice + scalar E_xc.
        xc = build_xc_periodic_uks(
            basis,
            system,
            grid,
            func,
            D_alpha_used,
            D_beta_used,
            lat_opts,
        )
        E_xc = float(xc.e_xc)

        # Bloch-sum F^{2e,s}(g) and V_xc^s(g) at every k, add Hcore(k).
        # F_HF^s(k) = S_g e^{i k.R_g} F^{2e,s}(g) at every k in one fused
        # call (OpenMP-parallel over k) when the C++ kernel is available;
        # per-k Python fold otherwise. V_xc(k) and Hcore(k) compose per k
        # below, unchanged.
        if _bloch_mk is not None:
            F_HF_alpha_k_all = [np.asarray(m) for m in _bloch_mk(
                F_HF_alpha_blocks, cell_r_list, k_arr_list)]
            F_HF_beta_k_all = [np.asarray(m) for m in _bloch_mk(
                F_HF_beta_blocks, cell_r_list, k_arr_list)]
        else:
            F_HF_alpha_k_all = [
                _bloch_sum_blocks(F_HF_alpha_blocks, cells, k)
                for k in k_arr_list
            ]
            F_HF_beta_k_all = [
                _bloch_sum_blocks(F_HF_beta_blocks, cells, k)
                for k in k_arr_list
            ]
        # Corrected-Ewald full-range exchange: the reciprocal and q+G=0
        # arms, subtracted from the Bloch-folded short-range blocks so
        # that BOTH the Fock and the energy trace below see one
        # consistent K. Prefactor is c_full with no 1/2: this driver
        # folds J_total - K_s on the one-particle spin density (unlike
        # the closed-shell J - K/2). The J-only reporting build further
        # down deliberately does NOT get these arms, which is what keeps
        # e_hf_K_per_cell = (E_HF - E_J) correct.
        if corrected_exchange is not None:
            _c_full = float(exx.c_full)
            # The damped per-k lists are the primary densities (the
            # real-space sets are folded from them); using them directly
            # drops both the BvK torus-coverage constraint and the
            # uniform-weight assumption of the inverse fold, and on a
            # symmetry-reduced mesh k_space_terms_all_k unfolds them to
            # the full BZ internally.
            _ka = corrected_exchange.k_space_terms_all_k(
                S_k_list, D_a_used_per_k
            )
            _kb = corrected_exchange.k_space_terms_all_k(
                S_k_list, D_b_used_per_k
            )
            F_HF_alpha_k_all = [
                f - _c_full * t for f, t in zip(F_HF_alpha_k_all, _ka)
            ]
            F_HF_beta_k_all = [
                f - _c_full * t for f, t in zip(F_HF_beta_k_all, _kb)
            ]

        F_alpha_k_list = []
        F_beta_k_list = []
        F_HF_alpha_k_list = []
        F_HF_beta_k_list = []
        for k_idx, k_cart in enumerate(k_points):
            k_arr = np.asarray(k_cart)
            F_HF_a_k = F_HF_alpha_k_all[k_idx]
            F_HF_b_k = F_HF_beta_k_all[k_idx]
            V_xc_a_k = _bloch_sum_lms_at_k(xc.V_alpha, k_arr)
            V_xc_b_k = _bloch_sum_lms_at_k(xc.V_beta, k_arr)
            F_a = Hcore_k_list[k_idx] + F_HF_a_k + V_xc_a_k
            F_b = Hcore_k_list[k_idx] + F_HF_b_k + V_xc_b_k
            F_a = 0.5 * (F_a + F_a.conj().T)
            F_b = 0.5 * (F_b + F_b.conj().T)
            F_alpha_k_list.append(F_a)
            F_beta_k_list.append(F_b)
            F_HF_alpha_k_list.append(F_HF_a_k)
            F_HF_beta_k_list.append(F_HF_b_k)

        # Energy + per-k errors.
        # E_elec = E_xc + S_k w_k [1/2 Re tr((D_a + D_b).H_k)
        #                          + 1/2 Re tr(D_a(k).F_HF_a(k))
        #                          + 1/2 Re tr(D_b(k).F_HF_b(k))]
        E_core_trace = 0.0
        E_HF_alpha_trace = 0.0
        E_HF_beta_trace = 0.0
        grad_norm_sum = 0.0
        error_alpha_k_list: List[np.ndarray] = []
        error_beta_k_list: List[np.ndarray] = []
        for idx in range(n_k):
            D_a_k = D_a_used_per_k[idx]
            D_b_k = D_b_used_per_k[idx]
            H_k = Hcore_k_list[idx]
            F_a_k = F_alpha_k_list[idx]
            F_b_k = F_beta_k_list[idx]
            F_HF_a_k = F_HF_alpha_k_list[idx]
            F_HF_b_k = F_HF_beta_k_list[idx]
            w = float(weights[idx])
            # NOTE: prefactor is 1.0 (not 1/2) because the per-spin
            # contribution below uses F_HF_s (Hartree + scaled-K only,
            # no Hcore). Compare the multi-k UHF Ewald driver, which
            # uses 1/2 on Hcore *and* uses the full F (Hcore included)
            # inside the per-spin terms -- equivalent total.
            E_core_trace += w * np.real(np.trace((D_a_k + D_b_k) @ H_k))
            E_HF_alpha_trace += w * 0.5 * np.real(np.trace(D_a_k @ F_HF_a_k))
            E_HF_beta_trace += w * 0.5 * np.real(np.trace(D_b_k @ F_HF_b_k))
            S_k = S_k_list[idx]
            FDS_a = F_a_k @ D_a_k @ S_k
            FDS_b = F_b_k @ D_b_k @ S_k
            err_a = FDS_a - FDS_a.conj().T
            err_b = FDS_b - FDS_b.conj().T
            error_alpha_k_list.append(err_a)
            error_beta_k_list.append(err_b)
            grad_norm_sum += w * float(
                np.sqrt(np.linalg.norm(err_a) ** 2 + np.linalg.norm(err_b) ** 2)
            )
        E_elec = (
            E_xc
            + float(E_core_trace)
            + float(E_HF_alpha_trace)
            + float(E_HF_beta_trace)
        )
        # Madelung-leak correction (v0.6.1). For UKS, total density
        # is D_a + D_b at the unit cell.
        _D_g0 = np.asarray(_g0_block(D_alpha_used)) + np.asarray(_g0_block(D_beta_used))
        _S_g0 = np.asarray(_g0_block(S_lat))
        E_madelung_fix = (
            0.0
            if slab_mode
            else _madelung_energy_correction_for_lat(_D_g0, _S_g0, system, lat_opts)
        )
        E_total = E_elec + e_nuc + E_madelung_fix
        # Free-energy formulation: with smearing the variational
        # quantity is A = E - T*S (per-spin entropies summed); dE and
        # the convergence check run on A so fractional occupations
        # can't oscillate the bare energy below the tolerance.
        free_energy = E_total - smearing_T * entropy

        dE = free_energy - E_prev
        # Divergence detection (v0.6.2).
        check_scf_divergence(
            "run_uks_periodic_multi_k_ewald3d",
            iter_idx,
            E_total,
            grad_norm_sum,
            dE,
        )
        diis_sub = 0
        if diis is not None:
            diis_sub = max(diis_sub, diis.subspace_size)
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm_sum),
                diis_subspace=diis_sub,
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm_sum),
            diis=diis_sub,
        )
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm_sum < float(opts.conv_tol_grad)
        )

        # Phase C1c gate.
        in_quadratic_phase = (
            quadratic_fallback_iter > 0 and iter_idx > quadratic_fallback_iter
        )

        new_C_alpha: List[np.ndarray] = []
        new_eps_alpha: List[np.ndarray] = []
        new_C_beta: List[np.ndarray] = []
        new_eps_beta: List[np.ndarray] = []

        if in_quadratic_phase:
            from .quadratic_scf import quadratic_step

            for idx in range(n_k):
                C_a, eps_a = quadratic_step(
                    F_alpha_k_list[idx],
                    C_alpha_per_k[idx],
                    eps_alpha_per_k[idx],
                    n_alpha,
                    shift=quadratic_fallback_shift,
                    max_step=quadratic_fallback_max_step,
                )
                C_b, eps_b = quadratic_step(
                    F_beta_k_list[idx],
                    C_beta_per_k[idx],
                    eps_beta_per_k[idx],
                    n_beta,
                    shift=quadratic_fallback_shift,
                    max_step=quadratic_fallback_max_step,
                )
                new_C_alpha.append(C_a)
                new_eps_alpha.append(eps_a)
                new_C_beta.append(C_b)
                new_eps_beta.append(eps_b)
        else:
            # Spin-coupled DIIS extrapolation: one history over the
            # concatenated a + b per-k block lists with duplicated
            # k-weights, one coefficient set applied to both spins.
            # Skipped entirely (not even recorded) while the PATTERN_HOLD
            # window is active; see the hold_active note at the loop head.
            if diis is not None and not hold_active:
                F_ex = diis.extrapolate(
                    list(F_alpha_k_list) + list(F_beta_k_list),
                    list(error_alpha_k_list) + list(error_beta_k_list),
                    list(weights) + list(weights),
                )
                if diis_active:
                    F_alpha_k_list = F_ex[:n_k]
                    F_beta_k_list = F_ex[n_k:]

            # Shift against the actual per-spin density used this cycle.
            # Fractional SAD, READ and damped densities retain their weights.
            if level_shift != 0.0:
                for idx in range(n_k):
                    S_k = S_k_list[idx]
                    D_a_k = D_a_used_per_k[idx]
                    D_b_k = D_b_used_per_k[idx]
                    F_alpha_k_list[idx] = apply_level_shift_k(
                        F_alpha_k_list[idx], S_k, D_a_k, level_shift,
                        LevelShiftDensity.SPIN)
                    F_beta_k_list[idx] = apply_level_shift_k(
                        F_beta_k_list[idx], S_k, D_b_k, level_shift,
                        LevelShiftDensity.SPIN)

            # Diagonalize per spin per k.
            for idx in range(n_k):
                C_a, eps_a = _diag_in_orth_basis(
                    F_alpha_k_list[idx],
                    X_k_list[idx],
                )
                C_b, eps_b = _diag_in_orth_basis(
                    F_beta_k_list[idx],
                    X_k_list[idx],
                )
                new_C_alpha.append(C_a)
                new_eps_alpha.append(eps_a)
                new_C_beta.append(C_b)
                new_eps_beta.append(eps_b)

        # SPINLOCK PATTERN_HOLD: for cycles 2..spinlock_iterations, reorder the
        # freshly diagonalised MOs per k per spin so the occupied subspace most
        # overlapping the previous cycle's occupied set comes first (MOM); the
        # column-order aufbau fill below then picks up the held broken-symmetry
        # pattern instead of pure-aufbau, protecting an ATOMSPIN seed. iter 1
        # sets the pattern by aufbau (no previous MOs yet).
        if _pattern_hold and 1 < iter_idx <= spinlock_iterations:
            for idx in range(n_k):
                if n_alpha > 0 and C_alpha_occ_prev[idx] is not None:
                    new_C_alpha[idx], new_eps_alpha[idx] = _mom_reorder(
                        new_C_alpha[idx], new_eps_alpha[idx],
                        S_k_list[idx], C_alpha_occ_prev[idx], n_alpha)
                if n_beta > 0 and C_beta_occ_prev[idx] is not None:
                    new_C_beta[idx], new_eps_beta[idx] = _mom_reorder(
                        new_C_beta[idx], new_eps_beta[idx],
                        S_k_list[idx], C_beta_occ_prev[idx], n_beta)
        # Record this cycle's occupied MOs for the next cycle's MOM hold.
        if _pattern_hold and iter_idx <= spinlock_iterations:
            for idx in range(n_k):
                C_alpha_occ_prev[idx] = (
                    new_C_alpha[idx][:, :n_alpha].copy() if n_alpha > 0 else None)
                C_beta_occ_prev[idx] = (
                    new_C_beta[idx][:, :n_beta].copy() if n_beta > 0 else None)

        C_alpha_per_k = new_C_alpha
        eps_alpha_per_k = new_eps_alpha
        C_beta_per_k = new_C_beta
        eps_beta_per_k = new_eps_beta

        occ_alpha_per_k, mu_alpha, entropy_alpha = _occupations_per_spin(
            eps_alpha_per_k, n_alpha
        )
        occ_beta_per_k, mu_beta, entropy_beta = _occupations_per_spin(
            eps_beta_per_k, n_beta
        )
        entropy = entropy_alpha + entropy_beta
        D_alpha_new = _spin_density(C_alpha_per_k, occ_alpha_per_k)
        D_beta_new = _spin_density(C_beta_per_k, occ_beta_per_k)
        D_alpha_prev = D_alpha_used
        D_beta_prev = D_beta_used
        D_a_used_per_k_prev = D_a_used_per_k
        D_b_used_per_k_prev = D_b_used_per_k
        D_alpha_real = D_alpha_new
        D_beta_real = D_beta_new

        if converged:
            # The provisional criteria describe D_*_used.  Rebuild the
            # physical, unshifted KS Fock on the newly diagonalised density,
            # then require its next density and thermodynamic objective to be
            # the same fixed point.  This includes finite-T/Gilat occupation
            # motion, which a canonical-orbital commutator cannot detect.
            D_a_terminal_input = _density_matrices_per_k(
                C_alpha_per_k, occ_alpha_per_k
            )
            D_b_terminal_input = _density_matrices_per_k(
                C_beta_per_k, occ_beta_per_k
            )
            if wedge_unfolding is not None:
                D_alpha_terminal = wedge_unfolding.fold_density(
                    D_a_terminal_input, lat_opts
                )
                D_beta_terminal = wedge_unfolding.fold_density(
                    D_b_terminal_input, lat_opts
                )
            else:
                D_alpha_terminal = D_alpha_real
                D_beta_terminal = D_beta_real

            F_HF_alpha_terminal_blocks, F_HF_beta_terminal_blocks = (
                _build_uks_fock_2e_blocks_ewald3d(
                    basis,
                    system,
                    D_alpha_terminal,
                    D_beta_terminal,
                    omega,
                    exx,
                    lat_opts,
                    grid_shape_t,
                    origin,
                    spacing_bohr,
                    j_cache=j_cache,
                    d_total_buffer=d_total_buffer,
                    full_range_alpha=(
                        None
                        if corrected_exchange is None
                        else float(omega)
                    ),
                    symmetry_reduction=exchange_symmetry_reduction,
                )
            )
            xc_terminal = build_xc_periodic_uks(
                basis,
                system,
                grid,
                func,
                D_alpha_terminal,
                D_beta_terminal,
                lat_opts,
            )
            if _bloch_mk is not None:
                F_HF_alpha_terminal = [
                    np.asarray(m)
                    for m in _bloch_mk(
                        F_HF_alpha_terminal_blocks,
                        cell_r_list,
                        k_arr_list,
                    )
                ]
                F_HF_beta_terminal = [
                    np.asarray(m)
                    for m in _bloch_mk(
                        F_HF_beta_terminal_blocks,
                        cell_r_list,
                        k_arr_list,
                    )
                ]
            else:
                F_HF_alpha_terminal = [
                    _bloch_sum_blocks(
                        F_HF_alpha_terminal_blocks, cells, k
                    )
                    for k in k_arr_list
                ]
                F_HF_beta_terminal = [
                    _bloch_sum_blocks(
                        F_HF_beta_terminal_blocks, cells, k
                    )
                    for k in k_arr_list
                ]
            if corrected_exchange is not None:
                c_full = float(exx.c_full)
                K_alpha_terminal = corrected_exchange.k_space_terms_all_k(
                    S_k_list, D_a_terminal_input
                )
                K_beta_terminal = corrected_exchange.k_space_terms_all_k(
                    S_k_list, D_b_terminal_input
                )
                F_HF_alpha_terminal = [
                    fock - c_full * exchange
                    for fock, exchange in zip(
                        F_HF_alpha_terminal, K_alpha_terminal
                    )
                ]
                F_HF_beta_terminal = [
                    fock - c_full * exchange
                    for fock, exchange in zip(
                        F_HF_beta_terminal, K_beta_terminal
                    )
                ]

            F_alpha_terminal: List[np.ndarray] = []
            F_beta_terminal: List[np.ndarray] = []
            for idx, k_cart in enumerate(k_points):
                k_arr = np.asarray(k_cart)
                V_xc_alpha = _bloch_sum_lms_at_k(
                    xc_terminal.V_alpha, k_arr
                )
                V_xc_beta = _bloch_sum_lms_at_k(
                    xc_terminal.V_beta, k_arr
                )
                F_a = (
                    Hcore_k_list[idx]
                    + F_HF_alpha_terminal[idx]
                    + V_xc_alpha
                )
                F_b = (
                    Hcore_k_list[idx]
                    + F_HF_beta_terminal[idx]
                    + V_xc_beta
                )
                F_alpha_terminal.append(0.5 * (F_a + F_a.conj().T))
                F_beta_terminal.append(0.5 * (F_b + F_b.conj().T))

            C_alpha_terminal: List[np.ndarray] = []
            eps_alpha_terminal: List[np.ndarray] = []
            C_beta_terminal: List[np.ndarray] = []
            eps_beta_terminal: List[np.ndarray] = []
            for idx in range(n_k):
                C_a, eps_a = _diag_in_orth_basis(
                    F_alpha_terminal[idx], X_k_list[idx]
                )
                C_b, eps_b = _diag_in_orth_basis(
                    F_beta_terminal[idx], X_k_list[idx]
                )
                if _pattern_hold and iter_idx <= spinlock_iterations:
                    if n_alpha > 0 and C_alpha_occ_prev[idx] is not None:
                        C_a, eps_a = _mom_reorder(
                            C_a,
                            eps_a,
                            S_k_list[idx],
                            C_alpha_occ_prev[idx],
                            n_alpha,
                        )
                    if n_beta > 0 and C_beta_occ_prev[idx] is not None:
                        C_b, eps_b = _mom_reorder(
                            C_b,
                            eps_b,
                            S_k_list[idx],
                            C_beta_occ_prev[idx],
                            n_beta,
                        )
                C_alpha_terminal.append(C_a)
                eps_alpha_terminal.append(eps_a)
                C_beta_terminal.append(C_b)
                eps_beta_terminal.append(eps_b)
            occ_alpha_terminal, _, entropy_alpha_terminal = (
                _occupations_per_spin(eps_alpha_terminal, n_alpha)
            )
            occ_beta_terminal, _, entropy_beta_terminal = (
                _occupations_per_spin(eps_beta_terminal, n_beta)
            )
            D_a_terminal_next = _density_matrices_per_k(
                C_alpha_terminal, occ_alpha_terminal
            )
            D_b_terminal_next = _density_matrices_per_k(
                C_beta_terminal, occ_beta_terminal
            )
            density_fixed_point_residual = max(
                [
                    float(np.max(np.abs(new - old)))
                    for new, old in zip(
                        D_a_terminal_next, D_a_terminal_input
                    )
                ]
                + [
                    float(np.max(np.abs(new - old)))
                    for new, old in zip(
                        D_b_terminal_next, D_b_terminal_input
                    )
                ],
                default=0.0,
            )
            E_core_terminal = 0.0
            E_HF_alpha_terminal = 0.0
            E_HF_beta_terminal = 0.0
            for idx in range(n_k):
                D_a = D_a_terminal_next[idx]
                D_b = D_b_terminal_next[idx]
                w = float(weights[idx])
                E_core_terminal += w * np.real(
                    np.trace((D_a + D_b) @ Hcore_k_list[idx])
                )
                E_HF_alpha_terminal += w * 0.5 * np.real(
                    np.trace(D_a @ F_HF_alpha_terminal[idx])
                )
                E_HF_beta_terminal += w * 0.5 * np.real(
                    np.trace(D_b @ F_HF_beta_terminal[idx])
                )
            if wedge_unfolding is not None:
                D_alpha_terminal_next = wedge_unfolding.fold_density(
                    D_a_terminal_next, lat_opts
                )
                D_beta_terminal_next = wedge_unfolding.fold_density(
                    D_b_terminal_next, lat_opts
                )
            else:
                D_alpha_terminal_next = _spin_density(
                    C_alpha_terminal, occ_alpha_terminal
                )
                D_beta_terminal_next = _spin_density(
                    C_beta_terminal, occ_beta_terminal
                )
            D_g0_terminal = np.asarray(
                _g0_block(D_alpha_terminal_next)
            ) + np.asarray(_g0_block(D_beta_terminal_next))
            E_madelung_terminal = (
                0.0
                if slab_mode
                else _madelung_energy_correction_for_lat(
                    D_g0_terminal, _S_g0, system, lat_opts
                )
            )
            free_energy_terminal = (
                float(xc_terminal.e_xc)
                + float(E_core_terminal)
                + float(E_HF_alpha_terminal)
                + float(E_HF_beta_terminal)
                + e_nuc
                + E_madelung_terminal
                - smearing_T
                * (entropy_alpha_terminal + entropy_beta_terminal)
            )
            converged = (
                abs(free_energy_terminal - free_energy)
                < float(opts.conv_tol_energy)
                and density_fixed_point_residual < float(opts.conv_tol_grad)
            )
            terminal_recheck_at_cap = (
                not converged and iter_idx >= int(opts.max_iter)
            )

        E_prev = free_energy
        if converged:
            break

    # ---- Final pass on converged D's ------------------------------------
    if converged or terminal_recheck_at_cap:
        D_a_final_per_k = _density_matrices_per_k(
            C_alpha_per_k, occ_alpha_per_k
        )
        D_b_final_per_k = _density_matrices_per_k(
            C_beta_per_k, occ_beta_per_k
        )
        if wedge_unfolding is not None:
            # One consistent exact fold for the reported densities and
            # both final builds (F_HF and the J-only reference).
            D_alpha_real = wedge_unfolding.fold_density(
                D_a_final_per_k, lat_opts
            )
            D_beta_real = wedge_unfolding.fold_density(
                D_b_final_per_k, lat_opts
            )
        F_HF_alpha_blocks, F_HF_beta_blocks = _build_uks_fock_2e_blocks_ewald3d(
            basis,
            system,
            D_alpha_real,
            D_beta_real,
            omega,
            exx,
            lat_opts,
            grid_shape_t,
            origin,
            spacing_bohr,
            j_cache=j_cache,
            d_total_buffer=d_total_buffer,
            full_range_alpha=(
                None if corrected_exchange is None else float(omega)
            ),
            symmetry_reduction=exchange_symmetry_reduction,
        )
        # J-only per-spin pair for reporting.
        if exx.needs_exchange:
            J_only_alpha_blocks, J_only_beta_blocks = _build_uks_fock_2e_blocks_ewald3d(
                basis,
                system,
                D_alpha_real,
                D_beta_real,
                omega,
                None,
                lat_opts,
                grid_shape_t,
                origin,
                spacing_bohr,
                j_cache=j_cache,
                d_total_buffer=d_total_buffer,
            )
        else:
            J_only_alpha_blocks = F_HF_alpha_blocks
            J_only_beta_blocks = F_HF_beta_blocks

        xc = build_xc_periodic_uks(
            basis,
            system,
            grid,
            func,
            D_alpha_real,
            D_beta_real,
            lat_opts,
        )
        E_xc = float(xc.e_xc)

        if _bloch_mk is not None:
            F_HF_alpha_k_all = [np.asarray(m) for m in _bloch_mk(
                F_HF_alpha_blocks, cell_r_list, k_arr_list)]
            F_HF_beta_k_all = [np.asarray(m) for m in _bloch_mk(
                F_HF_beta_blocks, cell_r_list, k_arr_list)]
            J_only_alpha_k_all = [np.asarray(m) for m in _bloch_mk(
                J_only_alpha_blocks, cell_r_list, k_arr_list)]
            J_only_beta_k_all = [np.asarray(m) for m in _bloch_mk(
                J_only_beta_blocks, cell_r_list, k_arr_list)]
        else:
            F_HF_alpha_k_all = [
                _bloch_sum_blocks(F_HF_alpha_blocks, cells, k)
                for k in k_arr_list
            ]
            F_HF_beta_k_all = [
                _bloch_sum_blocks(F_HF_beta_blocks, cells, k)
                for k in k_arr_list
            ]
            J_only_alpha_k_all = [
                _bloch_sum_blocks(J_only_alpha_blocks, cells, k)
                for k in k_arr_list
            ]
            J_only_beta_k_all = [
                _bloch_sum_blocks(J_only_beta_blocks, cells, k)
                for k in k_arr_list
            ]
        # Corrected-Ewald arms on F_HF only. J_only_* deliberately stays
        # uncorrected: e_hf_K_per_cell is (E_HF - E_J), so the J-only side
        # must carry no exchange at all for that difference to be the
        # corrected exchange energy.
        if corrected_exchange is not None:
            _c_full = float(exx.c_full)
            _ka = corrected_exchange.k_space_terms_all_k(
                S_k_list, D_a_final_per_k
            )
            _kb = corrected_exchange.k_space_terms_all_k(
                S_k_list, D_b_final_per_k
            )
            F_HF_alpha_k_all = [
                f - _c_full * x for f, x in zip(F_HF_alpha_k_all, _ka)
            ]
            F_HF_beta_k_all = [
                f - _c_full * x for f, x in zip(F_HF_beta_k_all, _kb)
            ]

        F_alpha_k_list = []
        F_beta_k_list = []
        F_HF_alpha_k_list = []
        F_HF_beta_k_list = []
        J_only_alpha_k_list: List[np.ndarray] = []
        J_only_beta_k_list: List[np.ndarray] = []
        for k_idx, k_cart in enumerate(k_points):
            k_arr = np.asarray(k_cart)
            F_HF_a_k = F_HF_alpha_k_all[k_idx]
            F_HF_b_k = F_HF_beta_k_all[k_idx]
            V_xc_a_k = _bloch_sum_lms_at_k(xc.V_alpha, k_arr)
            V_xc_b_k = _bloch_sum_lms_at_k(xc.V_beta, k_arr)
            F_alpha_k_list.append(
                0.5
                * (
                    (Hcore_k_list[k_idx] + F_HF_a_k + V_xc_a_k)
                    + (Hcore_k_list[k_idx] + F_HF_a_k + V_xc_a_k).conj().T
                )
            )
            F_beta_k_list.append(
                0.5
                * (
                    (Hcore_k_list[k_idx] + F_HF_b_k + V_xc_b_k)
                    + (Hcore_k_list[k_idx] + F_HF_b_k + V_xc_b_k).conj().T
                )
            )
            F_HF_alpha_k_list.append(F_HF_a_k)
            F_HF_beta_k_list.append(F_HF_b_k)
            J_only_alpha_k_list.append(J_only_alpha_k_all[k_idx])
            J_only_beta_k_list.append(J_only_beta_k_all[k_idx])

        final_C_alpha: List[np.ndarray] = []
        final_C_beta: List[np.ndarray] = []
        final_eps_alpha: List[np.ndarray] = []
        final_eps_beta: List[np.ndarray] = []
        E_core_trace = 0.0
        E_HF_alpha_trace = 0.0
        E_HF_beta_trace = 0.0
        E_J_alpha_trace = 0.0
        E_J_beta_trace = 0.0
        for idx in range(n_k):
            C_a, eps_a = _diag_in_orth_basis(
                F_alpha_k_list[idx],
                X_k_list[idx],
            )
            C_b, eps_b = _diag_in_orth_basis(
                F_beta_k_list[idx],
                X_k_list[idx],
            )
            # If the SCF converged while the PATTERN_HOLD window was still
            # active, the converged state is the MOM-held occupied set, which
            # need not be aufbau in its own Fock. Re-select by max overlap
            # with the held pattern so the reported energy / <S^2> / MOs
            # describe the state the SCF actually converged to (an aufbau
            # slice here would silently swap to a different state).
            if _pattern_hold and iter_idx <= spinlock_iterations:
                if n_alpha > 0 and C_alpha_occ_prev[idx] is not None:
                    C_a, eps_a = _mom_reorder(
                        C_a, eps_a, S_k_list[idx],
                        C_alpha_occ_prev[idx], n_alpha)
                if n_beta > 0 and C_beta_occ_prev[idx] is not None:
                    C_b, eps_b = _mom_reorder(
                        C_b, eps_b, S_k_list[idx],
                        C_beta_occ_prev[idx], n_beta)
            final_C_alpha.append(C_a)
            final_C_beta.append(C_b)
            final_eps_alpha.append(eps_a)
            final_eps_beta.append(eps_b)

        # The terminal Fock generally has a different spectrum from the
        # preceding SCF update. Fractional and Gilat occupations therefore
        # have to be solved again on these reported eigenvalues; carrying the
        # previous occupation arrays would mix two distinct states in the
        # energy, free energy, density, and output metadata.
        if use_fractional_density:
            occ_alpha_final, mu_alpha, entropy_alpha = _occupations_per_spin(
                final_eps_alpha, n_alpha
            )
            occ_beta_final, mu_beta, entropy_beta = _occupations_per_spin(
                final_eps_beta, n_beta
            )
            entropy = entropy_alpha + entropy_beta
        else:
            occ_alpha_final = occ_alpha_per_k
            occ_beta_final = occ_beta_per_k

        D_a_reported_per_k: List[np.ndarray] = []
        D_b_reported_per_k: List[np.ndarray] = []
        for idx in range(n_k):
            C_a = final_C_alpha[idx]
            C_b = final_C_beta[idx]
            if use_fractional_density:
                D_a_k = (C_a * occ_alpha_final[idx]) @ C_a.conj().T
                D_b_k = (C_b * occ_beta_final[idx]) @ C_b.conj().T
            else:
                C_a_occ = C_a[:, :n_alpha] if n_alpha > 0 else C_a[:, :0]
                C_b_occ = C_b[:, :n_beta] if n_beta > 0 else C_b[:, :0]
                D_a_k = C_a_occ @ C_a_occ.conj().T
                D_b_k = C_b_occ @ C_b_occ.conj().T
            D_a_reported_per_k.append(D_a_k)
            D_b_reported_per_k.append(D_b_k)
            w = float(weights[idx])
            E_core_trace += w * np.real(np.trace((D_a_k + D_b_k) @ Hcore_k_list[idx]))
            E_HF_alpha_trace += (
                w * 0.5 * np.real(np.trace(D_a_k @ F_HF_alpha_k_list[idx]))
            )
            E_HF_beta_trace += (
                w * 0.5 * np.real(np.trace(D_b_k @ F_HF_beta_k_list[idx]))
            )
            E_J_alpha_trace += (
                w * 0.5 * np.real(np.trace(D_a_k @ J_only_alpha_k_list[idx]))
            )
            E_J_beta_trace += (
                w * 0.5 * np.real(np.trace(D_b_k @ J_only_beta_k_list[idx]))
            )
        C_alpha_per_k = final_C_alpha
        C_beta_per_k = final_C_beta
        eps_alpha_per_k = final_eps_alpha
        eps_beta_per_k = final_eps_beta
        occ_alpha_per_k = occ_alpha_final
        occ_beta_per_k = occ_beta_final
        if wedge_unfolding is not None:
            D_alpha_real = wedge_unfolding.fold_density(
                D_a_reported_per_k, lat_opts
            )
            D_beta_real = wedge_unfolding.fold_density(
                D_b_reported_per_k, lat_opts
            )
        else:
            D_alpha_real = _spin_density(
                C_alpha_per_k, occ_alpha_per_k
            )
            D_beta_real = _spin_density(
                C_beta_per_k, occ_beta_per_k
            )
        E_elec = (
            E_xc
            + float(E_core_trace)
            + float(E_HF_alpha_trace)
            + float(E_HF_beta_trace)
        )
        # Madelung-leak correction (v0.6.1).
        _D_g0_f = np.asarray(_g0_block(D_alpha_real)) + np.asarray(
            _g0_block(D_beta_real)
        )
        _S_g0_f = np.asarray(_g0_block(S_lat))
        E_madelung_fix = (
            0.0
            if slab_mode
            else _madelung_energy_correction_for_lat(_D_g0_f, _S_g0_f, system, lat_opts)
        )
        E_total = float(E_elec) + e_nuc + E_madelung_fix
        E_coulomb_per_cell = float(E_J_alpha_trace + E_J_beta_trace)
        # tr(D.F_HF) = tr(D.J) - a.tr(D.K) (with the 1/2 prefactor inside
        # E_HF_*_trace), so HF_total - J_total = -a . 1/2 tr(D.K).
        E_hf_K_per_cell = float(
            (E_HF_alpha_trace - E_J_alpha_trace) + (E_HF_beta_trace - E_J_beta_trace)
        )

    # <S^2> from the Γ-block (or first) k-point -- same shortcut as
    # multi-k UHF Ewald.
    if n_alpha == 0 or n_beta == 0:
        s2 = 0.25 * (n_alpha - n_beta) * (n_alpha - n_beta + 2) + n_beta
    else:
        k0_idx = 0
        for i, k in enumerate(k_points):
            if np.allclose(np.asarray(k), 0.0):
                k0_idx = i
                break
        S_real = np.real(S_k_list[k0_idx])
        s2 = _spin_squared(
            n_alpha,
            n_beta,
            np.real(C_alpha_per_k[k0_idx]),
            np.real(C_beta_per_k[k0_idx]),
            S_real,
        )

    plog.converged(n_iter=iter_idx, energy=E_total, converged=converged)
    from .eigs_preflight import _copy_lattice_opts

    return PeriodicUKSMultiKEwaldResult(
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density_k is not None),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        used_kpoint_symmetry_unfolding=wedge_unfolding is not None,
        energy=E_total,
        e_electronic=float(E_elec),
        e_nuclear=e_nuc,
        e_xc=float(E_xc),
        e_coulomb=float(E_coulomb_per_cell),
        e_hf_exchange=float(E_hf_K_per_cell),
        n_iter=iter_idx,
        converged=converged,
        s_squared=float(s2),
        s_squared_ideal=0.25 * (mult - 1) * (mult + 1),
        mo_energies_alpha=eps_alpha_per_k,
        mo_coeffs_alpha=C_alpha_per_k,
        fock_alpha=F_alpha_k_list,
        density_alpha=D_alpha_real,
        density_alpha_k=_density_matrices_per_k(C_alpha_per_k, occ_alpha_per_k),
        density_beta_k=_density_matrices_per_k(C_beta_per_k, occ_beta_per_k),
        mo_energies_beta=eps_beta_per_k,
        mo_coeffs_beta=C_beta_per_k,
        fock_beta=F_beta_k_list,
        density_beta=D_beta_real,
        overlap=S_k_list,
        hcore=Hcore_k_list,
        functional=str(opts.functional),
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
        smearing_temperature=float(smearing_T),
        fermi_level_alpha=float(mu_alpha),
        fermi_level_beta=float(mu_beta),
        bz_integration=bz_integration,
        entropy=float(entropy),
        free_energy=float(E_total - smearing_T * entropy),
        effective_lattice_opts=_copy_lattice_opts(lat_opts),
        occupations_alpha=[
            np.asarray(occ, dtype=float).copy() for occ in occ_alpha_per_k
        ],
        occupations_beta=[
            np.asarray(occ, dtype=float).copy() for occ in occ_beta_per_k
        ],
    )
