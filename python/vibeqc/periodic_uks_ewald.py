"""Phase 15c-3a: Γ-point periodic UKS SCF driver using the composed
EWALD_3D Coulomb dispatch.

Open-shell DFT counterpart of :func:`run_rks_periodic_gamma_ewald3d`
(15c-1) and :func:`run_uhf_periodic_gamma_ewald3d` (15a). Per SCF
iteration:

    F_a  =  H_core  +  J_ewald(w, D_a + D_b)  -  K_HF(D_a)  +  V_xc^a
    F_b  =  H_core  +  J_ewald(w, D_a + D_b)  -  K_HF(D_b)  +  V_xc^b

where ``K_HF = c_full*K_full + c_sr*K_erfc(omega_screen)`` per the CAM
assembly of :func:`vibeqc.periodic_screened_exchange.resolve_periodic_exchange`
(global hybrids: ``c_full = hf_exchange_fraction``; screened hybrids
like hse06: pure erfc short-range exchange; LR-heavy range-separated
functionals fail closed). Compared to UHF Ewald (15a) which uses
``-K_s`` (full HF exchange per spin), here K is coefficient-scaled
for hybrids and skipped entirely for pure DFT.

V_xc comes from the new :func:`build_xc_periodic_uks` (Phase 15c-3
C++ addition), which returns separate V_a / V_b / E_xc from
spin-resolved densities. The functional is constructed with
``spin=2`` to take the libxc polarized path.

Density flow.  Γ-only molecular-limit cells have all electron density
in the home cell, so per-spin densities are wrapped in degenerate
LatticeMatrixSets (block 0 = D_s, others zero) before being handed
to ``build_xc_periodic_uks``. Same trick as the Γ-only RKS Ewald
driver -- exact in the molecular limit, the regime ``build_j_ewald_3d``
itself targets at Γ.

Energy formula. From the molecular UKS pattern (uks.cpp):

    E_elec  =  E_xc  +  tr((D_a + D_b).H_core)
                     +  1/2 tr(D_a.F_HF_a) + 1/2 tr(D_b.F_HF_b)

where ``F_HF_s = J(D_total) - a.K(D_s)`` is the HF-exchange piece
of F_s (V_xc reported through E_xc rather than a trace).

Scope.

  * Γ-only single k-point.
  * Open-shell, ``multiplicity >= 1``.
  * Pure DFT, hybrid, and HF (``a = 1``, equivalent to UHF Ewald).
  * DIIS on the block-diagonal [a, b] error vector with one rolling
    history per spin (matches UHF Ewald and the molecular UKS C++
    driver's convention).
  * Saunders-Hillier level shift via ``options.level_shift``.
  * Periodic Becke partition selectable via
    ``options.use_periodic_becke``.
  * The rigorous 2D slab sibling ``run_uks_periodic_gamma_ewald2d`` reuses this
    loop with slab ``V_ne``, ``E_nn``, and Hartree ``J`` selected by
    ``CoulombMethod.SLAB_EWALD_2D``.

The multi-k UKS Ewald driver lands in
:mod:`vibeqc.periodic_uks_multi_k_ewald` (15c-3b); it shares the
helpers with :mod:`vibeqc.periodic_uhf_multi_k_ewald` and adds the
periodic UKS XC.
"""

from __future__ import annotations

from .guess import periodic_result_selection

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    apply_level_shift,
    BasisSet,
    bloch_sum,
    build_grid,
    build_jk_gamma_molecular_limit,
    build_xc_periodic_uks,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    CoulombMethod,
    EwaldOptions,
    Functional,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    LevelShiftDensity,
    nuclear_repulsion_per_cell,
    PeriodicKSOptions,
    PeriodicSystem,
    SCFIteration,
    SpinlockMode,
)
from .ewald_composed import make_ewald_3d_gamma_j_builder
from .ewald_composed_slab import make_slab_ewald_2d_gamma_j_builder
from .ewald_j import auto_grid
from .guess import initial_densities_open_shell
from .madelung import (
    madelung_energy_correction_for_lat as _madelung_energy_correction_for_lat,
)
from .mom import reorder_occupied_by_max_overlap as _mom_reorder
from .periodic_grid import build_periodic_becke_grid
from .periodic_rhf_ewald import _canonical_orthogonalizer, _refuse_if_dense_ionic
from .periodic_scf_accelerators import DynamicDamping, PeriodicSCFAccelerator
from .periodic_screened_exchange import (
    build_exchange_gamma,
    resolve_periodic_exchange,
)
from .periodic_uhf_ewald import _integer_spin_occupations, _spin_squared
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence

__all__ = [
    "PeriodicUKSEwaldResult",
    "run_uks_periodic_gamma_ewald2d",
    "run_uks_periodic_gamma_ewald3d",
]


@dataclass
class PeriodicUKSEwaldResult:
    """Result of :func:`run_uks_periodic_gamma_ewald3d`.

    Spin-resolved counterpart of :class:`PeriodicRKSEwaldResult` with
    UHF-style ``s_squared`` diagnostic. All matrices are at Γ.
    """

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

    # a spin
    mo_energies_alpha: np.ndarray
    mo_coeffs_alpha: np.ndarray
    density_alpha: np.ndarray
    fock_alpha: np.ndarray

    # b spin
    mo_energies_beta: np.ndarray
    mo_coeffs_beta: np.ndarray
    density_beta: np.ndarray
    fock_beta: np.ndarray

    overlap: np.ndarray
    functional: str = ""
    scf_trace: List[SCFIteration] = field(default_factory=list)
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)
    occupations_alpha: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    occupations_beta: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def _density_set_gamma(
    template: LatticeMatrixSet,
    D: np.ndarray,
) -> LatticeMatrixSet:
    """Wrap the Γ-folded density ``D`` in a degenerate
    :class:`LatticeMatrixSet` matching the cell layout of ``template``:
    block 0 = D, every other block = zeros. Mutates ``template`` in
    place (LatticeMatrixSet's ``.blocks`` list is a transient copy;
    real mutation goes through ``set_block``)."""
    n_bf = D.shape[0]
    if n_bf != template.nbf:
        raise ValueError(
            f"_density_set_gamma: D has nbf={n_bf} but template has nbf={template.nbf}"
        )
    zero = np.zeros_like(D)
    for i in range(len(template)):
        template.set_block(i, D if i == 0 else zero)
    return template


def _resolve_slab_ewald_alpha(
    lat_opts: LatticeSumOptions,
    *,
    alpha: Optional[float],
    omega: Optional[float],
    driver: str,
) -> float:
    """Resolve the shared slab Ewald split parameter."""
    omega_alpha = None
    if omega is not None and float(omega) > 0.0:
        omega_alpha = float(omega)
    if alpha is None:
        alpha = omega_alpha
    elif omega_alpha is not None and abs(float(alpha) - omega_alpha) > 1e-14:
        raise ValueError(f"{driver}: alpha and positive omega alias disagree")
    if alpha is None:
        alpha = float(getattr(lat_opts, "slab_ewald_alpha", 0.4))
    alpha = float(alpha)
    if alpha <= 0.0:
        alpha = 0.4
    lat_opts.slab_ewald_alpha = alpha
    return alpha


def run_uks_periodic_gamma_ewald3d(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicKSOptions] = None,
    *,
    omega: float = 0.0,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    canonical_orth_normalize_diag_first: bool = True,
    auto_optimize_truncation: bool = True,
    allow_dense_ionic: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
) -> PeriodicUKSEwaldResult:
    """Γ-point open-shell periodic Kohn-Sham SCF with Ewald-3D Coulomb.

    Parameters mirror :func:`run_rks_periodic_gamma_ewald3d`.
    Multiplicity is read from ``system.multiplicity``; the driver
    handles closed-shell (mult=1, special case where a=b occupation)
    and open-shell systems uniformly.

    Returns
    -------
    :class:`PeriodicUKSEwaldResult`.
    """
    opts = options if options is not None else PeriodicKSOptions()
    from .guess import select_periodic_driver_guess
    selection = select_periodic_driver_guess(
        system, opts, route="ewald", method="UKS", driver="periodic_uks_ewald",
    )
    max_iter = int(opts.max_iter)
    if max_iter < 1:
        raise ValueError(
            "run_uks_periodic_gamma_ewald3d: max_iter must be >= 1; "
            f"got {max_iter}"
        )
    # SPINLOCK SPIN_SCHEDULE: two-phase (lock spin for spinlock_iterations
    # cycles, then release to the target multiplicity from that density).
    # Delegated before SCF setup; the sub-runs re-enter with spinlock OFF.
    if (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.SPIN_SCHEDULE
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    ):
        from .spinlock_periodic import run_spin_schedule

        return run_spin_schedule(
            lambda sysx, o: run_uks_periodic_gamma_ewald3d(
                sysx,
                basis,
                o,
                omega=omega,
                grid_shape=grid_shape,
                origin=origin,
                spacing_bohr=spacing_bohr,
                linear_dep_threshold=linear_dep_threshold,
                canonical_orth_normalize_diag_first=canonical_orth_normalize_diag_first,
                auto_optimize_truncation=auto_optimize_truncation,
                allow_dense_ionic=allow_dense_ionic,
                progress=progress,
                verbose=verbose,
            ),
            system,
            opts,
        )
    lat_opts = opts.lattice_opts
    slab_mode = lat_opts.coulomb_method == CoulombMethod.SLAB_EWALD_2D
    plog = resolve_progress(progress, verbose=verbose)

    if slab_mode:
        if system.dim != 2:
            raise ValueError(
                "run_uks_periodic_gamma_ewald2d: SLAB_EWALD_2D requires "
                f"dim == 2; got dim = {system.dim}"
            )
        if float(getattr(opts, "smearing_temperature", 0.0)) > 0.0:
            raise NotImplementedError(
                "run_uks_periodic_gamma_ewald2d: Fermi-Dirac smearing requires "
                "the multi-k Ewald machinery, which is not yet available for "
                "SLAB_EWALD_2D"
            )
        omega = _resolve_slab_ewald_alpha(
            lat_opts,
            alpha=None,
            omega=omega,
            driver="run_uks_periodic_gamma_ewald2d",
        )

    # ---- Force EWALD_3D gauge (gauge consistency; handover F4 2026-06-01) ----
    # This driver hard-codes the Hartree J to the Ewald-3D builder, so V_ne
    # (compute_nuclear_lattice_dispatch) and e_nuc (nuclear_repulsion_per_cell)
    # MUST share that gauge. Without the force, a default options object
    # (coulomb_method=DIRECT_TRUNCATED) makes nuclear_repulsion_per_cell return
    # the molecular 1/d sum and madelung_energy_correction_for_lat the bare-gauge
    # +a_M.Q_e^2/2L term; those only partially cancel (~0.74 mHa on H2/30-bohr),
    # so the SCF converged to a non-physical energy with no warning (CLAUDE.md
    # Sec.7). The closed-shell Γ / multi-k RHF/RKS siblings already force this
    # (audit F1); extending it here aligns e_nuclear with
    # run_rks_periodic_gamma_ewald3d and zeroes the now-redundant Madelung term
    # (madelung_energy_correction_for_lat returns 0.0 for EWALD_3D).
    # EWALD_3D V_ne (compute_nuclear_lattice_dispatch) is implemented only for
    # dim == 3 -- the 1D/2D Ewald variants raise (periodic_v_ne.py). So gate the
    # force on dim == 3; low-dim cells keep their DIRECT_TRUNCATED gauge (the
    # historical behaviour for these drivers on 1D/2D chains).
    if (
        not slab_mode
        and system.dim == 3
        and lat_opts.coulomb_method != CoulombMethod.EWALD_3D
    ):
        plog.info(
            "coulomb_method forced to EWALD_3D for gauge consistency "
            f"(was {lat_opts.coulomb_method!r}); this driver's Hartree J "
            "is Ewald-3D and V_ne / e_nuc must match"
        )
        lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    # Fail closed if periodic images of the basis overlap enough to make the
    # molecular-limit energy wrong (CLAUDE.md Sec.7); GDF/BIPOLE/GPW are correct
    # there. ``allow_dense_ionic=True`` bypasses for mechanics testing only.
    if not slab_mode:
        _refuse_if_dense_ionic(system, basis, lat_opts, allow_dense_ionic)

    # w must match the nuclear Ewald a (auto-selected from
    # nuclear_cutoff_bohr in the C++ ewald engine) so the jellium
    # background terms cancel exactly. Mirrors the override block in
    # run_rks_periodic_gamma_ewald3d (commit 49f8ae91). The driver
    # kwarg ``omega`` is retained for signature parity but overridden
    # here; users override via ``opts.ewald_omega`` when needed.
    _ewald_tol = getattr(opts, "ewald_tolerance", 1e-12)
    _cutoff = getattr(opts, "ewald_cutoff_bohr", lat_opts.nuclear_cutoff_bohr)
    if not slab_mode and omega <= 0.0:
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
            f"UKS Gamma SLAB_EWALD_2D / functional={opts.functional!r}, "
            f"alpha = {float(omega):.3f}"
        )
    else:
        plog.info(
            f"UKS Gamma EWALD_3D / functional={opts.functional!r}, "
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
                    "slab_ewald_alpha": float(omega) if slab_mode else None,
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

    n_elec = system.n_electrons()
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(
            f"run_uks_periodic_gamma_ewald3d: multiplicity must be >= 1; got {mult}"
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2
    if n_alpha + n_beta != n_elec:
        raise ValueError(
            "run_uks_periodic_gamma_ewald3d: n_electrons and multiplicity "
            f"are inconsistent (n_e={n_elec}, mult={mult})"
        )
    if n_beta < 0:
        raise ValueError(
            f"run_uks_periodic_gamma_ewald3d: n_beta < 0; n_e={n_elec}, mult={mult}"
        )

    # ---- Functional + DFT grid ------------------------------------------
    func = Functional(opts.functional, 2)  # spin-polarized
    # Exact-exchange assembly K_HF = c_full*K_full + c_sr*K_erfc(w_s);
    # global hybrids keep hf_exchange_fraction, screened hybrids
    # (hse06) get the pure erfc short-range kernel, LR-heavy RSH fails
    # closed. omega_screen is the functional's physical screening
    # parameter, NOT this driver's ``omega`` (the Ewald split alpha).
    exx = resolve_periodic_exchange(func, where="run_uks_periodic_gamma_ewald3d")

    if opts.use_periodic_becke:
        grid = build_periodic_becke_grid(
            system,
            grid_options=opts.grid,
            image_radius_bohr=float(opts.becke_image_radius_bohr),
        )
        # Periodic-Becke grid pairs with the Γ-torus density set (D in
        # every lattice block -> build_xc_periodic_uks cross-cell mode);
        # the home-cell-only set is the molecular-limit density that
        # pairs with the molecular grid below (the 2026-07-09 KRKS
        # finding class, spin-polarised flavour --
        # handovers/HANDOVER_AICCM_DIRECT_TORUS.md §4).
        from .periodic_rhf_gdf import _density_set_torus_gamma

        _set_xc_density = _density_set_torus_gamma
    else:
        grid = build_grid(system.unit_cell_molecule(), opts.grid)
        _set_xc_density = _density_set_gamma

    plog.info(
        f"UKS Gamma occupations: n_alpha = {n_alpha}, "
        f"n_beta = {n_beta} (multiplicity = {mult})"
    )

    # ---- Auto-optimise lattice truncation (default ON) -------------------
    if (
        not slab_mode
        and auto_optimize_truncation
        and lat_opts.coulomb_method == CoulombMethod.EWALD_3D
    ):
        from .eigs_preflight import (
            format_truncation_optimization_report,
            optimize_truncation,
        )

        opt_rep = optimize_truncation(system, basis, lattice_opts=lat_opts)
        if (
            opt_rep.n_evaluations > 1
            or opt_rep.optimized_lattice_opts.cutoff_bohr != lat_opts.cutoff_bohr
        ):
            plog.write_raw(format_truncation_optimization_report(opt_rep))
            if not opt_rep.converged:
                plog.warn(
                    "auto_optimize_truncation did not converge; SCF will run "
                    "with the last evaluated settings."
                )
            lat_opts = opt_rep.optimized_lattice_opts

    # ---- One-electron integrals at Γ ------------------------------------
    with plog.stage(
        "integrals_lattice", detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    ):
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        from .periodic_v_ne import compute_nuclear_lattice_dispatch

        eopts = None
        if slab_mode:
            eopts = EwaldOptions()
            eopts.alpha = float(omega)
            eopts.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr
        V_lat = compute_nuclear_lattice_dispatch(
            basis,
            system,
            lat_opts,
            ewald_options=eopts,
        )
    k_gamma = np.zeros(3)
    S = np.real(bloch_sum(S_lat, k_gamma))
    T = np.real(bloch_sum(T_lat, k_gamma))
    V = np.real(bloch_sum(V_lat, k_gamma))
    Hcore = T + V
    S = 0.5 * (S + S.T)
    Hcore = 0.5 * (Hcore + Hcore.T)

    # Linear-dependence preflight on S(Γ); see periodic_rhf_ewald.py.
    from .linear_dependence import scf_preflight_overlap_check

    scf_preflight_overlap_check(
        S,
        plog=plog,
        label="S(Γ)",
        basis=basis,
    )

    X, n_kept = _canonical_orthogonalizer(
        S,
        linear_dep_threshold,
        normalize_diag_first=canonical_orth_normalize_diag_first,
    )
    if max(n_alpha, n_beta) > n_kept:
        raise RuntimeError(
            "run_uks_periodic_gamma_ewald3d: canonical orthogonalisation "
            f"kept {n_kept} directions; need >= {max(n_alpha, n_beta)} "
            f"(n_a={n_alpha}, n_b={n_beta})."
        )

    use_davidson = getattr(opts, "use_davidson", False)
    dav_opts = getattr(opts, "davidson", None)
    dav_dim = getattr(opts, "davidson_min_dim", 100)
    use_dav = use_davidson and S.shape[0] >= dav_dim
    if use_dav and dav_opts is None:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    def diagonalise(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Fp = X.T @ F @ X
        Fp = 0.5 * (Fp + Fp.T)
        if use_dav and dav_opts is not None:
            from vibeqc._vibeqc_core import davidson_solve

            if dav_opts.n_eig == 0:
                dav_opts.n_eig = Fp.shape[0]
            if dav_opts.guess_vectors is not None:
                pass  # already set from previous iteration
            dres = davidson_solve(Fp, dav_opts)
            if not dres.converged:
                raise RuntimeError(
                    f"Davidson did not converge after {dres.n_iter} iters"
                )
            eps, Cp = dres.eigenvalues, dres.eigenvectors
            dav_opts.guess_vectors = Cp
        else:
            eps, Cp = np.linalg.eigh(Fp)
        return X @ Cp, eps

    # Cache the iteration-invariant Hartree-J machinery once so the PATOM
    # in-field step and the SCF loop use the same Ewald/slab J route.
    if slab_mode:
        j_build = make_slab_ewald_2d_gamma_j_builder(
            basis,
            system,
            lattice_opts=lat_opts,
            alpha=float(omega),
        )
    else:
        j_build = make_ewald_3d_gamma_j_builder(
            basis,
            system,
            omega=float(omega),
            lattice_opts=lat_opts,
            grid_shape=grid_shape_t,
            origin=origin,
            spacing_bohr=spacing_bohr,
        )

    # ---- Initial guess via the unified engine.
    # Bug fix (v0.9.x): this Γ-only UKS path previously ignored
    # ``opts.initial_guess`` and always used HCore. The engine call
    # respects the request; HCore reproduces the prior behaviour.
    guess = selection.effective
    seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
    # Seed the per-spin MO frame from Hcore for the quadratic-step
    # tracking (used even when SAD provides the start density).
    C_alpha, eps_alpha = diagonalise(Hcore)
    C_beta, eps_beta = diagonalise(Hcore)
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
        # READ restart (Γ-only): prior per-spin g=0 cell densities.
        read_density_alpha=getattr(opts, "read_density_alpha", None),
        read_density_beta=getattr(opts, "read_density_beta", None),
        read_path=getattr(opts, "read_path", ""),
        overlap=S,
    )
    if split is not None:
        D_alpha, D_beta = split
    else:
        D_alpha = (
            C_alpha[:, :n_alpha] @ C_alpha[:, :n_alpha].T
            if n_alpha > 0
            else np.zeros_like(Hcore)
        )
        D_beta = (
            C_beta[:, :n_beta] @ C_beta[:, :n_beta].T
            if n_beta > 0
            else np.zeros_like(Hcore)
        )
        D_alpha = 0.5 * (D_alpha + D_alpha.T)
        D_beta = 0.5 * (D_beta + D_beta.T)
    if guess == InitialGuess.PATOM:
        plog.info("initial guess: PATOM (SAD + one periodic in-field step)")
        D_total = D_alpha + D_beta
        J = j_build(D_total)
        jk_alpha = build_jk_gamma_molecular_limit(
            basis,
            system,
            lat_opts,
            2.0 * D_alpha,
            0.0,
        )
        jk_beta = build_jk_gamma_molecular_limit(
            basis,
            system,
            lat_opts,
            2.0 * D_beta,
            0.0,
        )
        K_alpha = 0.5 * np.asarray(jk_alpha.K)
        K_beta = 0.5 * np.asarray(jk_beta.K)
        F_alpha_seed = 0.5 * ((Hcore + J - K_alpha) + (Hcore + J - K_alpha).T)
        F_beta_seed = 0.5 * ((Hcore + J - K_beta) + (Hcore + J - K_beta).T)
        C_alpha, eps_alpha = diagonalise(F_alpha_seed)
        C_beta, eps_beta = diagonalise(F_beta_seed)
        D_alpha = (
            C_alpha[:, :n_alpha] @ C_alpha[:, :n_alpha].T
            if n_alpha > 0
            else np.zeros_like(Hcore)
        )
        D_beta = (
            C_beta[:, :n_beta] @ C_beta[:, :n_beta].T
            if n_beta > 0
            else np.zeros_like(Hcore)
        )
        D_alpha = 0.5 * (D_alpha + D_alpha.T)
        D_beta = 0.5 * (D_beta + D_beta.T)
    D_alpha_prev = D_alpha.copy()
    D_beta_prev = D_beta.copy()

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_uks_periodic_gamma_ewald3d: damping must be in [0, 1); got {damping}"
        )

    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )

    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    accel: Optional[PeriodicSCFAccelerator] = (
        PeriodicSCFAccelerator(opts) if use_diis else None
    )

    level_shift = float(getattr(opts, "level_shift", 0.0))

    # SPINLOCK PATTERN_HOLD: hold the seeded broken-symmetry occupied set by
    # maximum overlap (MOM) with the previous cycle for the first
    # ``spinlock_iterations`` cycles, then release. See periodic_uhf_ewald.
    spinlock_mode = getattr(opts, "spinlock_mode", SpinlockMode.OFF)
    spinlock_iterations = int(getattr(opts, "spinlock_iterations", 0))
    _pattern_hold = (
        spinlock_mode == SpinlockMode.PATTERN_HOLD and spinlock_iterations > 0
    )

    # Phase C1c -- quadratic SCF fallback (per-spin Newton step).
    quadratic_fallback_iter = int(getattr(opts, "quadratic_fallback_iter", 0))
    quadratic_fallback_shift = float(getattr(opts, "quadratic_fallback_shift", 0.1))
    quadratic_fallback_max_step = float(
        getattr(opts, "quadratic_fallback_max_step", 0.1)
    )

    # Track per-spin MO basis between iterations for the C1c step.
    C_alpha_prev_mo = C_alpha
    eps_alpha_prev_mo = eps_alpha
    C_beta_prev_mo = C_beta
    eps_beta_prev_mo = eps_beta

    # Pre-allocate degenerate LMS templates for D_a and D_b (V_xc input).
    D_alpha_set = compute_overlap_lattice(basis, system, lat_opts)
    D_beta_set = compute_overlap_lattice(basis, system, lat_opts)

    # ---- SCF loop -------------------------------------------------------
    scf_trace: List[SCFIteration] = []
    s_squared_ideal = 0.25 * (mult - 1) * (mult + 1)
    result = PeriodicUKSEwaldResult(
                 restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=e_nuc,
        e_xc=0.0,
        e_coulomb=0.0,
        e_hf_exchange=0.0,
        n_iter=0,
        converged=False,
        s_squared=0.0,
        s_squared_ideal=s_squared_ideal,
        mo_energies_alpha=np.empty(0),
        mo_coeffs_alpha=np.empty((0, 0)),
        density_alpha=D_alpha.copy(),
        fock_alpha=np.empty((0, 0)),
        mo_energies_beta=np.empty(0),
        mo_coeffs_beta=np.empty((0, 0)),
        density_beta=D_beta.copy(),
        fock_beta=np.empty((0, 0)),
        overlap=S,
        functional=str(opts.functional),
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
    )

    E_prev = 0.0
    scf_label = "SLAB_EWALD_2D" if slab_mode else "EWALD_3D"
    plog.banner(f"SCF (UKS Gamma {opts.functional!r}, {scf_label})")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    for iter_idx in range(1, max_iter + 1):
        if damper is not None:
            damping = damper.alpha
        # SPINLOCK PATTERN_HOLD: the accelerator is suspended (no history
        # recorded, no extrapolation, damping stays live) while the hold is
        # active. Fock extrapolation across held-window iterates steers the
        # SCF toward the symmetric attractor by continuous orbital rotation
        # -- a collapse the occupation-selecting MOM hold cannot see -- and
        # poisons the post-release history with out-of-basin iterates. The
        # history starts fresh at release.
        hold_active = _pattern_hold and iter_idx <= spinlock_iterations
        diis_active = (
            use_diis and iter_idx >= diis_start_iter and not hold_active
        )
        if iter_idx == 1 or damping == 0.0 or diis_active:
            D_alpha_used = D_alpha
            D_beta_used = D_beta
        else:
            D_alpha_used = damping * D_alpha_prev + (1.0 - damping) * D_alpha
            D_beta_used = damping * D_beta_prev + (1.0 - damping) * D_beta

        D_total = D_alpha_used + D_beta_used

        # Hartree J via Ewald (cached analytic-FT machinery).
        J = j_build(D_total)

        # Per-spin coefficient-folded K (pure DFT skips this).
        # K(D_s) = K(2.D_s) / 2 (RHF convention inside the builder).
        if exx.needs_exchange:
            K_alpha = 0.5 * build_exchange_gamma(
                basis, system, lat_opts, 2.0 * D_alpha_used, exx
            )
            K_beta = 0.5 * build_exchange_gamma(
                basis, system, lat_opts, 2.0 * D_beta_used, exx
            )
        else:
            K_alpha = None
            K_beta = None

        # V_xc via the periodic UKS XC kernel; density-set convention
        # paired with the grid choice above.
        _set_xc_density(D_alpha_set, D_alpha_used)
        _set_xc_density(D_beta_set, D_beta_used)
        xc = build_xc_periodic_uks(
            basis,
            system,
            grid,
            func,
            D_alpha_set,
            D_beta_set,
            lat_opts,
        )
        V_xc_alpha = np.real(bloch_sum(xc.V_alpha, k_gamma))
        V_xc_beta = np.real(bloch_sum(xc.V_beta, k_gamma))
        V_xc_alpha = 0.5 * (V_xc_alpha + V_xc_alpha.T)
        V_xc_beta = 0.5 * (V_xc_beta + V_xc_beta.T)
        E_xc = float(xc.e_xc)

        # F_s = Hcore + J - K_HF_s + V_xc^s (K carries the assembly
        # coefficients).
        if K_alpha is not None:
            F_HF_alpha = J - K_alpha
            F_HF_beta = J - K_beta
        else:
            F_HF_alpha = J
            F_HF_beta = J
        F_alpha = Hcore + F_HF_alpha + V_xc_alpha
        F_beta = Hcore + F_HF_beta + V_xc_beta
        F_alpha = 0.5 * (F_alpha + F_alpha.T)
        F_beta = 0.5 * (F_beta + F_beta.T)

        # Energy decomposition.
        E_core_trace = float(np.einsum("ij,ij->", D_total, Hcore))
        E_HF_alpha_trace = 0.5 * float(np.einsum("ij,ij->", D_alpha_used, F_HF_alpha))
        E_HF_beta_trace = 0.5 * float(np.einsum("ij,ij->", D_beta_used, F_HF_beta))
        E_elec = E_xc + E_core_trace + E_HF_alpha_trace + E_HF_beta_trace
        # Madelung-leak correction (v0.6.1).
        if lat_opts.coulomb_method in (
            CoulombMethod.EWALD_3D,
            CoulombMethod.SLAB_EWALD_2D,
        ):
            E_madelung_fix = 0.0
        else:
            E_madelung_fix = _madelung_energy_correction_for_lat(
                D_alpha_used + D_beta_used, S, system, lat_opts
            )
        E_total = E_elec + e_nuc + E_madelung_fix

        # Reporting decomposition: J vs HF-K.
        E_coulomb = 0.5 * float(np.einsum("ij,ij->", D_total, J))
        if K_alpha is not None:
            E_hf_K = -0.5 * float(
                np.einsum("ij,ij->", D_alpha_used, K_alpha)
            ) - 0.5 * float(np.einsum("ij,ij->", D_beta_used, K_beta))
        else:
            E_hf_K = 0.0

        # Per-spin orbital-gradient norms.
        FDS_alpha = F_alpha @ D_alpha_used @ S
        FDS_beta = F_beta @ D_beta_used @ S
        grad_alpha = FDS_alpha - FDS_alpha.T
        grad_beta = FDS_beta - FDS_beta.T
        grad_norm = float(
            np.sqrt(np.linalg.norm(grad_alpha) ** 2 + np.linalg.norm(grad_beta) ** 2)
        )

        dE = E_total - E_prev
        # Divergence detection (v0.6.2).
        check_scf_divergence(
            "run_uks_periodic_gamma_ewald3d",
            iter_idx,
            E_total,
            grad_norm,
            dE,
        )
        diis_sub = accel.subspace_size if accel is not None else 0
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=diis_sub,
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=diis_sub,
        )
        # Per-iter energy decomposition for cross-code parity comparison.
        E_kin_iter = float(np.einsum("ij,ij->", D_total, T))
        E_ne_iter = float(np.einsum("ij,ij->", D_total, V))
        plog.energy_decomposition(
            iter_idx,
            E_kin=E_kin_iter,
            E_ne=E_ne_iter,
            E_J=E_coulomb,
            E_xc=float(E_xc),
            E_K=float(E_hf_K),
            E_nuc=float(e_nuc),
            E_madelung=float(E_madelung_fix),
        )
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )

        # Phase C1c gate.
        in_quadratic_phase = (
            quadratic_fallback_iter > 0 and iter_idx > quadratic_fallback_iter
        )

        if in_quadratic_phase:
            from .quadratic_scf import quadratic_step

            C_alpha_new, eps_alpha_new = quadratic_step(
                F_alpha,
                C_alpha_prev_mo,
                eps_alpha_prev_mo,
                n_alpha,
                shift=quadratic_fallback_shift,
                max_step=quadratic_fallback_max_step,
            )
            C_beta_new, eps_beta_new = quadratic_step(
                F_beta,
                C_beta_prev_mo,
                eps_beta_prev_mo,
                n_beta,
                shift=quadratic_fallback_shift,
                max_step=quadratic_fallback_max_step,
            )
        else:
            # SCF-accelerator extrapolation. Every accelerator (DIIS
            # included) runs a single spin-coupled history -- one
            # coefficient set extrapolates both Focks; see the
            # _AcceleratorState note in periodic_scf_accelerators.py.
            # Skipped entirely (not even recorded) while the PATTERN_HOLD
            # window is active; see the hold_active note at the loop head.
            if accel is not None and not hold_active:
                F_alpha_ex, F_beta_ex = accel.extrapolate_uhf(
                    F_alpha,
                    F_beta,
                    error_alpha=grad_alpha,
                    error_beta=grad_beta,
                    density_alpha=D_alpha_used,
                    density_beta=D_beta_used,
                    energy=E_total,
                    mo_coeffs_alpha=C_alpha_prev_mo,
                    mo_coeffs_beta=C_beta_prev_mo,
                    mo_energies_alpha=eps_alpha_prev_mo,
                    mo_energies_beta=eps_beta_prev_mo,
                    n_alpha=n_alpha,
                    n_beta=n_beta,
                )
                if diis_active:
                    F_alpha = F_alpha_ex
                    F_beta = F_beta_ex

            # Shared Saunders-Hillier operator, F_s + b * (S - w * S D_s S).
            # The weight w is set by the density convention, not by taste.
            # D_alpha / D_beta are *spin* densities and idempotent in the S
            # metric (D_s S D_s = D_s), so S D_s S is exactly the projector
            # onto the occupied manifold and w = 1: occupied eigenvalues stay
            # put, virtuals rise by b. The closed-shell drivers carry the
            # total density D = 2P and therefore use w = 1/2 for the same
            # effect. This driver open-coded w = 1/2 on spin densities until
            # 2026-07-10, which left the occupied block shifted by +b/2 and
            # opened the gap by b/2 rather than b. It converged to the same
            # fixed point, so no energy assertion could see it. Routing
            # through the shared operator is what stops that recurring.
            if level_shift == 0.0:
                # Skip the pybind conversion of S and Dσ entirely on an
                # unshifted cycle, which is the default and common case.
                F_alpha_diag = F_alpha
                F_beta_diag = F_beta
            else:
                F_alpha_diag = apply_level_shift(
                    F_alpha, S, D_alpha_used, level_shift,
                    LevelShiftDensity.SPIN)
                F_beta_diag = apply_level_shift(
                    F_beta, S, D_beta_used, level_shift,
                    LevelShiftDensity.SPIN)

            C_alpha_new, eps_alpha_new = diagonalise(F_alpha_diag)
            C_beta_new, eps_beta_new = diagonalise(F_beta_diag)

        # SPINLOCK PATTERN_HOLD: reorder the occupied set by max overlap with
        # the previous cycle (MOM) for cycles 2..spinlock_iterations, then
        # release. iter 1 sets the pattern by aufbau (C_*_prev_mo is the prior
        # cycle's MOs). Mirrors periodic_uhf_ewald.
        if (
            _pattern_hold
            and 1 < iter_idx <= spinlock_iterations
            and C_alpha_prev_mo.shape[1] >= n_alpha
            and C_beta_prev_mo.shape[1] >= n_beta
        ):
            if n_alpha > 0:
                C_alpha_new, eps_alpha_new = _mom_reorder(
                    C_alpha_new, eps_alpha_new, S, C_alpha_prev_mo[:, :n_alpha], n_alpha
                )
            if n_beta > 0:
                C_beta_new, eps_beta_new = _mom_reorder(
                    C_beta_new, eps_beta_new, S, C_beta_prev_mo[:, :n_beta], n_beta
                )

        C_alpha_prev_mo = C_alpha_new
        eps_alpha_prev_mo = eps_alpha_new
        C_beta_prev_mo = C_beta_new
        eps_beta_prev_mo = eps_beta_new
        D_alpha_prev = D_alpha_used
        D_beta_prev = D_beta_used
        D_alpha = (
            C_alpha_new[:, :n_alpha] @ C_alpha_new[:, :n_alpha].T
            if n_alpha > 0
            else np.zeros_like(Hcore)
        )
        D_beta = (
            C_beta_new[:, :n_beta] @ C_beta_new[:, :n_beta].T
            if n_beta > 0
            else np.zeros_like(Hcore)
        )
        D_alpha = 0.5 * (D_alpha + D_alpha.T)
        D_beta = 0.5 * (D_beta + D_beta.T)

        # Assemble the externally visible state below from one
        # orbital-defined density and a physical Fock/energy evaluated on
        # that same density.  D_*_used can be damped and therefore cannot be
        # paired with the newly generated C_*_new orbitals.
        result.n_iter = iter_idx

        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

        if converged or iter_idx == max_iter:
            # Evaluate one internally consistent terminal state on every
            # exit path.  The last orbital/occupation state defines D; the
            # unshifted physical Fock, XC terms, and energy are rebuilt on
            # exactly that D.  At a non-fixed-point iteration cap the
            # orbitals do not diagonalise F[D], so report their Rayleigh
            # quotients in the rebuilt Fock as the orbital energies.
            D_total_f = D_alpha + D_beta
            J_f = j_build(D_total_f)
            if exx.needs_exchange:
                K_alpha_f = 0.5 * build_exchange_gamma(
                    basis, system, lat_opts, 2.0 * D_alpha, exx
                )
                K_beta_f = 0.5 * build_exchange_gamma(
                    basis, system, lat_opts, 2.0 * D_beta, exx
                )
            else:
                K_alpha_f = None
                K_beta_f = None
            _set_xc_density(D_alpha_set, D_alpha)
            _set_xc_density(D_beta_set, D_beta)
            xc_f = build_xc_periodic_uks(
                basis,
                system,
                grid,
                func,
                D_alpha_set,
                D_beta_set,
                lat_opts,
            )
            V_xc_a_f = np.real(bloch_sum(xc_f.V_alpha, k_gamma))
            V_xc_b_f = np.real(bloch_sum(xc_f.V_beta, k_gamma))
            V_xc_a_f = 0.5 * (V_xc_a_f + V_xc_a_f.T)
            V_xc_b_f = 0.5 * (V_xc_b_f + V_xc_b_f.T)
            E_xc_f = float(xc_f.e_xc)
            if K_alpha_f is not None:
                F_HF_a_f = J_f - K_alpha_f
                F_HF_b_f = J_f - K_beta_f
            else:
                F_HF_a_f = J_f
                F_HF_b_f = J_f
            F_alpha_f = Hcore + F_HF_a_f + V_xc_a_f
            F_beta_f = Hcore + F_HF_b_f + V_xc_b_f
            F_alpha_f = 0.5 * (F_alpha_f + F_alpha_f.T)
            F_beta_f = 0.5 * (F_beta_f + F_beta_f.T)
            C_a_f, eps_a_f = diagonalise(F_alpha_f)
            C_b_f, eps_b_f = diagonalise(F_beta_f)
            # Keep the look-ahead map in the MOM-held occupied basin while
            # PATTERN_HOLD is active. Comparing the held density against an
            # aufbau slice would measure a state swap, not its fixed-point
            # residual.
            if (
                _pattern_hold
                and iter_idx <= spinlock_iterations
                and C_alpha_prev_mo.shape[1] >= n_alpha
                and C_beta_prev_mo.shape[1] >= n_beta
            ):
                if n_alpha > 0:
                    C_a_f, eps_a_f = _mom_reorder(
                        C_a_f, eps_a_f, S, C_alpha_prev_mo[:, :n_alpha], n_alpha
                    )
                if n_beta > 0:
                    C_b_f, eps_b_f = _mom_reorder(
                        C_b_f, eps_b_f, S, C_beta_prev_mo[:, :n_beta], n_beta
                    )
            occ_alpha_f = _integer_spin_occupations(eps_a_f, n_alpha)
            occ_beta_f = _integer_spin_occupations(eps_b_f, n_beta)
            D_alpha_next = (C_a_f * occ_alpha_f) @ C_a_f.T
            D_beta_next = (C_b_f * occ_beta_f) @ C_b_f.T
            D_alpha_next = 0.5 * (D_alpha_next + D_alpha_next.T)
            D_beta_next = 0.5 * (D_beta_next + D_beta_next.T)
            density_fixed_point_residual = max(
                float(np.max(np.abs(D_alpha_next - D_alpha))),
                float(np.max(np.abs(D_beta_next - D_beta))),
            )
            E_core_f = float(np.einsum("ij,ij->", D_alpha + D_beta, Hcore))
            E_HF_a_f = 0.5 * float(np.einsum("ij,ij->", D_alpha, F_HF_a_f))
            E_HF_b_f = 0.5 * float(np.einsum("ij,ij->", D_beta, F_HF_b_f))
            E_elec_f = E_xc_f + E_core_f + E_HF_a_f + E_HF_b_f
            E_coulomb_f = 0.5 * float(np.einsum("ij,ij->", D_alpha + D_beta, J_f))
            if K_alpha_f is not None:
                E_hf_K_f = -0.5 * float(
                    np.einsum("ij,ij->", D_alpha, K_alpha_f)
                ) - 0.5 * float(np.einsum("ij,ij->", D_beta, K_beta_f))
            else:
                E_hf_K_f = 0.0
            if lat_opts.coulomb_method in (
                CoulombMethod.EWALD_3D,
                CoulombMethod.SLAB_EWALD_2D,
            ):
                E_madelung_fix_f = 0.0
            else:
                E_madelung_fix_f = _madelung_energy_correction_for_lat(
                    D_alpha + D_beta, S, system, lat_opts
                )
            result.energy = E_elec_f + e_nuc + E_madelung_fix_f
            result.e_electronic = E_elec_f
            result.e_xc = E_xc_f
            result.e_coulomb = E_coulomb_f
            result.e_hf_exchange = E_hf_K_f
            result.mo_energies_alpha = np.real(
                np.diag(C_alpha_new.T @ F_alpha_f @ C_alpha_new)
            )
            result.occupations_alpha = _integer_spin_occupations(
                result.mo_energies_alpha, n_alpha
            )
            result.mo_coeffs_alpha = C_alpha_new
            result.density_alpha = D_alpha
            result.fock_alpha = F_alpha_f
            result.mo_energies_beta = np.real(
                np.diag(C_beta_new.T @ F_beta_f @ C_beta_new)
            )
            result.occupations_beta = _integer_spin_occupations(
                result.mo_energies_beta, n_beta
            )
            result.mo_coeffs_beta = C_beta_new
            result.density_beta = D_beta
            result.fock_beta = F_beta_f
            result.s_squared = _spin_squared(
                n_alpha,
                n_beta,
                C_alpha_new,
                C_beta_new,
                S,
            )
            # ``converged`` above describes D_*_used, while the returned
            # state uses the freshly diagonalised D_*.  Damping can make
            # those materially different.  Verify the actual return
            # candidate before accepting it as an SCF fixed point.
            FDS_alpha_f = F_alpha_f @ D_alpha @ S
            FDS_beta_f = F_beta_f @ D_beta @ S
            grad_norm_f = float(
                np.sqrt(
                    np.linalg.norm(FDS_alpha_f - FDS_alpha_f.T) ** 2
                    + np.linalg.norm(FDS_beta_f - FDS_beta_f.T) ** 2
                )
            )
            terminal_converged = (
                converged
                and abs(result.energy - E_total) < float(opts.conv_tol_energy)
                and grad_norm_f < float(opts.conv_tol_grad)
                and density_fixed_point_residual < float(opts.conv_tol_grad)
            )
            if terminal_converged or iter_idx == max_iter:
                # Terminal rebuilding finalises the visible cycle; it is not
                # another SCF update. Replace (rather than append to) the
                # last trace entry so public trace output describes the same
                # state as the result while len(trace) remains n_iter. The
                # delta records the correction from the pre-update cycle.
                previous_trace = scf_trace[-1]
                scf_trace[-1] = SCFIteration(
                    iter=previous_trace.iter,
                    energy=float(result.energy),
                    delta_e=float(result.energy - previous_trace.energy),
                    grad_norm=float(grad_norm_f),
                    diis_subspace=previous_trace.diis_subspace,
                )
            if terminal_converged:
                result.converged = True
                plog.converged(
                    n_iter=result.n_iter,
                    energy=result.energy,
                    converged=True,
                )
                return result

    # Did not converge -- populate <S^2> on the final iterate.
    result.s_squared = _spin_squared(
        n_alpha,
        n_beta,
        result.mo_coeffs_alpha,
        result.mo_coeffs_beta,
        S,
    )
    plog.converged(
        n_iter=result.n_iter,
        energy=result.energy,
        converged=False,
    )
    return result


def run_uks_periodic_gamma_ewald2d(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicKSOptions] = None,
    *,
    alpha: Optional[float] = None,
    omega: Optional[float] = None,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    linear_dep_threshold: float = 1e-7,
    canonical_orth_normalize_diag_first: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
) -> PeriodicUKSEwaldResult:
    """Gamma-point UKS with rigorous 2D slab Ewald."""
    opts = options if options is not None else PeriodicKSOptions()
    lat_opts = opts.lattice_opts
    if system.dim != 2:
        raise ValueError(
            "run_uks_periodic_gamma_ewald2d: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}"
        )
    lat_opts.coulomb_method = CoulombMethod.SLAB_EWALD_2D
    slab_alpha = _resolve_slab_ewald_alpha(
        lat_opts,
        alpha=alpha,
        omega=omega,
        driver="run_uks_periodic_gamma_ewald2d",
    )
    return run_uks_periodic_gamma_ewald3d(
        system,
        basis,
        opts,
        omega=slab_alpha,
        grid_shape=grid_shape,
        origin=origin,
        spacing_bohr=spacing_bohr,
        linear_dep_threshold=linear_dep_threshold,
        canonical_orth_normalize_diag_first=canonical_orth_normalize_diag_first,
        auto_optimize_truncation=False,
        allow_dense_ionic=True,
        progress=progress,
        verbose=verbose,
    )
