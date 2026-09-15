"""BIPOLE-style periodic UKS driver in CRYSTAL's electrostatic gauge.

The open-shell DFT counterpart of :mod:`vibeqc.pbc_bipole_rks`.
Uses the same CRYSTAL-gauge Ewald J-split F^2e build for the Hartree
term and adds the spin-polarised libxc XC potential on the periodic
Becke grid. For hybrid functionals a fraction of HF exchange is
retained per spin channel via the native ``build_jk_2e_real_space``.

This driver keeps the same multi-k SCF scaffold as the UHF BIPOLE
driver: shared Ewald a across V_ne/E_nn/J^LR, analytic V_ne via
AO-pair Fourier transforms, real-space energy accounting, and
the maintained convergence accelerators (DIIS, MOM, level-shift schedule,
damping). The historical ODA hook fails closed.
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

from .periodic_k_density import real_space_density_from_per_k_density

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    apply_level_shift_k,
    BasisSet,
    BlochKMesh,
    EwaldOptions,
    Functional,
    GridOptions,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    LevelShiftDensity,
    PeriodicKSOptions,
    PeriodicSystem,
    PeriodicXCDensityDomain,
    SCFIteration,
    bloch_sum,
    build_grid,
    build_xc_periodic_uks,
    direct_lattice_cells,
    ewald_nuclear_repulsion,
    nuclear_repulsion_per_cell,
    real_space_density_from_kpoints_fractional,
)
from .bipole_ext_el_pole import compute_ext_el_spheropole
from .guess import (
    _coerce_periodic_driver_guess,
    initial_densities_open_shell,
    periodic_fock_guess_k,
)
from .level_shift_schedule import LevelShiftSchedule
from .mom import select_occupied_by_max_overlap as _mom_select
from .oda import compute_oda_lambda as _compute_oda_lambda
from .oda import oda_mix_densities as _oda_mix
from .pbc_bipole_common import (
    bipole_confirmation_phase,
    bipole_final_density_phase,
    unrestricted_occupations_per_spin,
    unrestricted_spin_density,
    unrestricted_split_k_density_list,
    PBCBipoleEnergyComponents,
    _bloch_sum_blocks,
    _bloch_sum_blocks_multi_k,
    _combine_density_sets,
    _compute_nuclear_lattice_ewald_reciprocal_ft,
    _copy_lattice_with_blocks,  # noqa: F401  (compat re-export, see tests/test_pbc_bipole_common.py)
    _crystal_ewald_options,
    _density_set_gamma_or_lattice,
    _emit_bipole_semantic_diagnostic,
    _expand_ibz_kmesh_for_ewald_j,
    _lattice_contract,
    _spin_occupations,
    _zero_cross_cell_density,
    bvk_torus_density_matrices,
    build_bipole_one_electron_lattice,
    prepare_bipole_lattice_options,
    raise_bipole_ewald_real_cutoff,
    resolve_bipole_ewald_alpha,
    reject_bipole_ecp_options,
    reject_bipole_fractional_legacy_gauge,
    reject_bipole_quartet_far_field,
    reject_bipole_lone_non_gamma_kpoint,
    reject_bipole_solver_options,
    reject_bipole_unsupported_ks_functional,
    unrestricted_bipole_commutator_norm,
    bipole_terminal_check,
    refresh_bipole_terminal_trace,
    resolve_fock_mixing,
    resolve_auto_fock_mixing,
    resolve_bipole_sr_image_extent,
    resolve_bipole_fock_symmetry,
    warn_bipole_charged_cell,
    warn_bipole_legacy_multik_gauge,
    home_cell_block,
    smearing_basin_warning,
    validate_bipole_kmesh,
)
from .pbc_bipole_fock import (
    _sr_density_cells,
    BipoleFockContext,
    BipoleScreenedExchangeExecution,
    build_bipole_unrestricted_fock,
)
from .periodic_grid import build_periodic_becke_grid
from .periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _damp_lattice_matrix,
    _diag_in_orth_basis,
)
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicUHFAccelerator,
)
from .periodic_screened_exchange import resolve_periodic_exchange
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .smearing import (
    SmearingOptions,
)
from .smearing import (
    closed_shell_periodic_occupations as _closed_shell_periodic_occupations,
)

__all__ = [
    "PBCBipoleUKSResult",
    "run_pbc_bipole_uks",
]


@dataclass
class PBCBipoleUKSResult:
    """Result of :func:`run_pbc_bipole_uks`."""

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

    mo_energies_alpha: List[np.ndarray]
    mo_coeffs_alpha: List[np.ndarray]
    fock_alpha: List[np.ndarray]
    density_alpha: LatticeMatrixSet

    mo_energies_beta: List[np.ndarray]
    mo_coeffs_beta: List[np.ndarray]
    fock_beta: List[np.ndarray]
    density_beta: LatticeMatrixSet

    overlap: List[np.ndarray]
    hcore: List[np.ndarray]
    runtime_backend: str = "pbc-bipole"
    scf_trace: List[SCFIteration] = field(default_factory=list)
    e_ext_el_spheropole: Optional[float] = None
    ewald_alpha_bohr_inv: Optional[float] = None
    # Absolute internal ket-image radius used by an erfc SR build, if any.
    # The analytic-gradient preview uses this to reject domain mismatch.
    sr_image_extent_bohr: Optional[float] = None
    # True when the Fock build used the group-invariant pair-resolved domain.
    pair_resolved_fock_domain: bool = False
    # Dudarev DFT+U contribution per unit cell (Hartree). 0 unless the
    # caller passed ``dft_plus_u=[HubbardSite(...)]``. Sum of per-spin
    # contributions for the open-shell UKS path.
    e_dft_plus_u: float = 0.0
    energy_components: List[PBCBipoleEnergyComponents] = field(
        default_factory=list,
    )
    # Gauge provenance (option (b)): see run_pbc_bipole_uhf.
    exchange_ewald_split: bool = False
    exchange_exxdiv: Optional[str] = None
    # Non-None when finite-T smearing straddled the HOMO-LUMO gap and may
    # have selected a near-metallic basin (ionic-Γ basin trap).
    basin_warning: Optional[str] = None
    functional: str = ""
    # Smearing diagnostics (zero when smearing_temperature == 0).
    smearing_temperature: float = 0.0
    fermi_level: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations_alpha: List[np.ndarray] = field(default_factory=list)
    occupations_beta: List[np.ndarray] = field(default_factory=list)
    # Cartesian k-points (bohr^-1) and weights this result spans, in the
    # same order as the per-k, per-spin ``mo_coeffs_*`` / ``mo_energies_*``
    # lists. Lets optional single-k output writers locate Gamma instead of
    # assuming the first k-point is Gamma. See PBCBipoleRHFResult.
    kpoints_cart: Optional[np.ndarray] = None
    kpoint_weights: Optional[np.ndarray] = None
    # Effective previous-Fock weight after the shared automatic default is
    # resolved. This can differ from ``options.fock_mixing``.
    fock_mixing: float = 0.0
    screened_exchange_execution: Optional[
        BipoleScreenedExchangeExecution
    ] = None
    # Corrected-gauge overlap-fold convergence diagnostic. Appended to
    # preserve the positional signature of the historical result fields.
    overlap_fold_drift: Optional[float] = None
    # BIPOLE-EXACT-ZONE increment 1b: radius (bohr) of the exact erfc
    # output zone when restricted below the operator cutoff (None =
    # exact zone == cutoff). Analytic-gradient consumers fail closed.
    exact_zone_bohr: Optional[float] = None
    # Fixed-Nalpha/Nbeta smearing solves one chemical potential per spin.
    # Appended to preserve positional compatibility; ``fermi_level`` remains
    # the historical singular alias.
    fermi_level_alpha: Optional[float] = None
    fermi_level_beta: Optional[float] = None

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_mesh: object = None
    restart_kpoints: object = None
    restart_weights: object = None


@dataclass
class _PBCBipoleUKSFockBuild:
    """Internal Fock bundle for one UKS density pair."""

    f2e_alpha_real: LatticeMatrixSet
    f2e_beta_real: LatticeMatrixSet
    f_alpha_k_list: List[np.ndarray]
    f_beta_k_list: List[np.ndarray]
    e_j_short_range: Optional[float] = None
    e_j_long_range: Optional[float] = None
    e_exchange: Optional[float] = None
    e_j_multipole: Optional[float] = None
    # k-space per-spin exchange correction (Ewald exchange split,
    # a_HF-scaled); lives in F_s(k) -- caller adds to E_2e.
    e_2e_k_correction: float = 0.0
    # The q+G=0 (Madelung / exxdiv='ewald') share of e_2e_k_correction on
    # its own, so the gauge is visible next to the total (#82).
    e_exchange_finite_size: Optional[float] = None
    # XC energy of the SAME grid quadrature whose per-spin V_xc entered
    # f_alpha/beta_k_list (None for the HF-like PATOM seed build).
    # Consuming this instead of re-running build_xc_periodic_uks keeps
    # the iteration's V_xc and e_xc from one build.
    e_xc: Optional[float] = None
    screened_exchange_execution: Optional[
        BipoleScreenedExchangeExecution
    ] = None


def run_pbc_bipole_uks(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    options=None,
    *,
    functional: Optional[str] = None,
    linear_dep_threshold: float = 1e-7,
    fock_mixing: Optional[float] = None,
    canonical_orth_normalize_diag_first: bool = True,
    level_shift_schedule: Optional[LevelShiftSchedule] = None,
    use_mom: bool = False,
    use_oda: bool = False,
    oda_trust_lambda_max: float = 1.0,
    use_incremental_fock: bool = True,
    use_ewald_j_split: Optional[bool] = None,
    ewald_omega: Optional[float] = None,
    ewald_precision: float = 1e-8,
    v_ne_grid_options: Optional[GridOptions] = None,
    use_multipole_far_field: Optional[bool] = False,
    use_exchange_ewald_split: Optional[bool] = None,
    exchange_exxdiv: str = "ewald",
    multipole_l_max: int = 2,
    use_fock_symmetry: Optional[bool] = None,
    use_fock_symmetry_reduce: Optional[bool] = None,
    sr_image_precision: Optional[float] = 1e-6,
    sr_image_extent_bohr: Optional[float] = None,
    exact_zone_bohr: Optional[float] = None,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    bz_integration: Optional[str] = None,
    init_alpha: Optional[Sequence[np.ndarray]] = None,
    init_beta: Optional[Sequence[np.ndarray]] = None,
    initial_density_k: Optional[Sequence] = None,
    dft_plus_u: Optional[List["HubbardSite"]] = None,
    xc_density_domain: PeriodicXCDensityDomain = (
        PeriodicXCDensityDomain.AUTO
    ),
) -> PBCBipoleUKSResult:
    """Multi-k open-shell UKS via the CRYSTAL-gauge BIPOLE scaffold.

    ``fock_mixing`` overrides ``options.fock_mixing`` when supplied and is
    forwarded through both phases of a spin schedule without rewriting the
    caller's options field.  A resolved zero can still activate the documented
    DIIS-off KS automatic value.

    ``sr_image_precision`` / ``sr_image_extent_bohr`` and the automatic
    ``use_fock_symmetry_reduce=None`` policy match
    :func:`run_pbc_bipole_rhf`. Spin-schedule recursion forwards all three
    controls unchanged.

    ``bz_integration="gilat"`` applies the T=0 Gilat-Raubenheimer microcell
    integral independently to the fixed alpha and beta populations on full
    or symmetry-reduced Monkhorst-Pack meshes. It has zero entropy and cannot
    be combined with a positive ``options.smearing_temperature``.

    ``dft_plus_u``: optional list of :class:`HubbardSite`. Same
    per-spin per-k convention as :func:`run_pbc_bipole_uhf` --
    ``F_s(k) += S(k) V_AO_s S(k)`` with
    ``V_AO_s = U_eff (1/2 - n_s)``; ``E_U_total`` is summed over
    spins and surfaced via ``result.e_dft_plus_u``.
    """
    reject_bipole_quartet_far_field(
        use_multipole_far_field,
        driver="run_pbc_bipole_uks",
    )
    reject_bipole_lone_non_gamma_kpoint(
        kmesh,
        driver="run_pbc_bipole_uks",
    )
    from ._vibeqc_core import PeriodicKSOptions as _PKSOpts

    caller_supplied_options = options is not None
    opts = options if caller_supplied_options else _PKSOpts()
    reject_bipole_ecp_options(
        opts,
        driver="run_pbc_bipole_uks",
        basis=basis,
        system=system,
    )
    reject_bipole_solver_options(opts, driver="run_pbc_bipole_uks")
    if int(opts.max_iter) < 1:
        raise ValueError("run_pbc_bipole_uks: max_iter must be at least 1")
    if use_oda:
        raise NotImplementedError(
            "run_pbc_bipole_uks: unrestricted ODA is unavailable because "
            "the current line search does not include the beta-spin energy "
            "direction; use DIIS instead"
        )
    requested_fock_mixing = resolve_fock_mixing(
        opts,
        fock_mixing,
        where="run_pbc_bipole_uks",
    )
    # SPINLOCK. SPIN_SCHEDULE (two-phase) is delegated before SCF setup;
    # PATTERN_HOLD reuses the per-k MOM machinery below (gated to the
    # spinlock window).
    from ._vibeqc_core import SpinlockMode
    from .spinlock_periodic import check_spinlock_support, run_spin_schedule

    _initial_guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", None),
        driver="run_pbc_bipole_uks",
        supported=periodic_guess_capabilities('bipole', 'UKS', dim=getattr(system, "dim", 3), multi_k=(len(kmesh.kpoints) > 1 or np.prod(kmesh.mesh) > 1), transport='lattice'),
        restart_supplied=init_alpha is not None or init_beta is not None or initial_density_k is not None,
    )
    _broken_state_symmetry_controls = []
    if getattr(opts, "atomic_spins", None):
        _broken_state_symmetry_controls.append("atomic_spins")
    if bool(use_mom):
        _broken_state_symmetry_controls.append("use_mom")
    if getattr(opts, "spinlock_mode", SpinlockMode.OFF) != SpinlockMode.OFF:
        _broken_state_symmetry_controls.append("spinlock")
    if bool(dft_plus_u):
        _broken_state_symmetry_controls.append("dft_plus_u")
    if init_alpha is not None or init_beta is not None or initial_density_k is not None:
        _broken_state_symmetry_controls.append("initial spin density")
    if _initial_guess in (InitialGuess.READ, InitialGuess.PATOM):
        _broken_state_symmetry_controls.append(_initial_guess.name.lower())
    if (
        bool(use_fock_symmetry) or use_fock_symmetry_reduce is True
    ) and _broken_state_symmetry_controls:
        raise NotImplementedError(
            "run_pbc_bipole_uks: explicit Fock symmetry enforcement or "
            "reduction cannot be combined with an electronically broken or "
            "unverified state control ("
            + ", ".join(_broken_state_symmetry_controls)
            + "). Disable Fock symmetry, or remove the state-selective "
            "control and opt in only for a symmetry-covariant density."
        )
    _spin_schedule_requested = (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.SPIN_SCHEDULE
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    )
    if _spin_schedule_requested:
        _schedule_ir = np.asarray(
            getattr(kmesh, "ir_mapping", []), dtype=int
        ).reshape(-1)
        _schedule_true_multik = (
            len(list(kmesh.kpoints)) > 1
            or _schedule_ir.size > 1
            or int(
                np.prod(np.asarray(getattr(kmesh, "mesh", (1, 1, 1))))
            ) > 1
        )
        if _schedule_true_multik:
            raise NotImplementedError(
                "run_pbc_bipole_uks: SPIN_SCHEDULE is Gamma-only. Its "
                "phase-2 restart currently carries only the home-cell spin "
                "densities and cannot reconstruct a complete multi-k state. "
                "Use PATTERN_HOLD or a Gamma mesh."
            )
        return run_spin_schedule(
            lambda sysx, o: run_pbc_bipole_uks(
                sysx,
                basis,
                kmesh,
                o,
                functional=functional,
                linear_dep_threshold=linear_dep_threshold,
                fock_mixing=requested_fock_mixing,
                canonical_orth_normalize_diag_first=canonical_orth_normalize_diag_first,
                level_shift_schedule=level_shift_schedule,
                use_mom=use_mom,
                use_oda=use_oda,
                oda_trust_lambda_max=oda_trust_lambda_max,
                use_incremental_fock=use_incremental_fock,
                use_ewald_j_split=use_ewald_j_split,
                ewald_omega=ewald_omega,
                ewald_precision=ewald_precision,
                v_ne_grid_options=v_ne_grid_options,
                use_multipole_far_field=use_multipole_far_field,
                use_exchange_ewald_split=use_exchange_ewald_split,
                exchange_exxdiv=exchange_exxdiv,
                multipole_l_max=multipole_l_max,
                use_fock_symmetry=use_fock_symmetry,
                use_fock_symmetry_reduce=use_fock_symmetry_reduce,
                sr_image_precision=sr_image_precision,
                sr_image_extent_bohr=sr_image_extent_bohr,
                exact_zone_bohr=exact_zone_bohr,
                progress=progress,
                verbose=verbose,
                bz_integration=bz_integration,
                dft_plus_u=dft_plus_u,
                xc_density_domain=xc_density_domain,
            ),
            system,
            opts,
        )
    check_spinlock_support(
        opts,
        {SpinlockMode.PATTERN_HOLD, SpinlockMode.SPIN_SCHEDULE},
        "the BIPOLE UKS driver",
    )
    if functional is not None:
        opts.functional = str(functional)
    if not getattr(opts, "functional", None):
        opts.functional = "pbe"
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError("run_pbc_bipole_uks: smearing_temperature must be >= 0")
    if bz_integration is not None:
        bz_integration = str(bz_integration).strip().lower()
    if bz_integration not in (None, "smearing", "gilat"):
        raise ValueError(
            "run_pbc_bipole_uks: bz_integration must be None, 'smearing', "
            f"or 'gilat'; got {bz_integration!r}"
        )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise ValueError(
            "run_pbc_bipole_uks: bz_integration='gilat' is a T=0 "
            "integrator; do not combine it with smearing_temperature > 0"
        )
    use_fractional_density = (smearing_T > 0.0) or use_gilat

    func = Functional(opts.functional, 2)  # spin-polarised
    reject_bipole_unsupported_ks_functional(
        func,
        driver="run_pbc_bipole_uks",
    )
    uses_external_xc = bool(getattr(func, "is_external", False))
    if uses_external_xc:
        from .periodic_external_xc import (
            _grid_options_for_external,
            _require_zero_temperature_external_xc,
        )

        _require_zero_temperature_external_xc(
            smearing_T,
            where="run_pbc_bipole_uks",
        )

        opts.grid = _grid_options_for_external(
            func,
            opts.grid if caller_supplied_options else None,
            where="run_pbc_bipole_uks",
        )
        if xc_density_domain != PeriodicXCDensityDomain.PERIODIC_LATTICE:
            raise ValueError(
                "run_pbc_bipole_uks: external XC requires "
                "xc_density_domain=PERIODIC_LATTICE"
            )
        if not bool(getattr(opts, "use_periodic_becke", False)):
            raise ValueError(
                "run_pbc_bipole_uks: external XC requires "
                "use_periodic_becke=True"
            )
    # CAM exchange assembly (see run_pbc_bipole_rks): alpha_hf is the
    # FULL-RANGE arm only; screened hybrids (hse06, c_full = 0) get
    # their erfc short-range K inside build_bipole_unrestricted_fock
    # (no seam, no far-field); LR-heavy RSH fails closed here.
    exx = resolve_periodic_exchange(func, where="run_pbc_bipole_uks")
    alpha_hf = float(exx.c_full)

    # CRYSTAL-style auto-FMIXING: fires only for DFT without DIIS.
    # Validated on the closed-shell twin (RKS Gap-B run matrix,
    # 2026-07-13); the shared Fock builder makes the damping behaviour
    # spin-independent. See resolve_auto_fock_mixing.
    fock_mixing_value = resolve_auto_fock_mixing(
        requested_fock_mixing,
        alpha_hf=alpha_hf,
        use_diis=bool(opts.use_diis),
        where="run_pbc_bipole_uks",
    )

    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)
    (
        use_ewald_j_split,
        use_ewald_j_split_auto,
        lat_opts_2e,
        lat_opts_1e,
    ) = prepare_bipole_lattice_options(system, lat_opts, use_ewald_j_split, plog)

    n_elec = int(system.n_electrons())
    n_alpha, n_beta = _spin_occupations(system)
    # ``_spin_occupations`` promotes the default multiplicity on an odd-
    # electron periodic cell to the minimum valid open shell without mutating
    # the caller's system. Report the multiplicity of that executed state.
    mult = int(n_alpha - n_beta + 1)

    # Open-shell guard moved after auto-disable resolution below.

    plog.info(
        f"PBC BIPOLE UKS (CRYSTAL-gauge) / cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    )
    if exx.is_screened:
        plog.info(
            f"  functional = {opts.functional}, screened exchange "
            f"K = {exx.c_sr:g}*K_erfc(omega={exx.omega_screen:g} bohr^-1)"
        )
    else:
        plog.info(
            f"  functional = {opts.functional}, hf_exchange_fraction = {alpha_hf}"
        )
    plog.info(f"  n_alpha = {n_alpha}, n_beta = {n_beta}, multiplicity = {mult}")

    _kmesh_ibz = kmesh
    _ir_mapping = np.asarray(getattr(kmesh, "ir_mapping", []), dtype=int).reshape(-1)

    k_points = list(_kmesh_ibz.kpoints)
    weights = np.asarray(_kmesh_ibz.weights, dtype=float)
    _input_true_multik = (
        len(k_points) > 1
        or _ir_mapping.size > 1
        or int(np.prod(np.asarray(getattr(kmesh, "mesh", (1, 1, 1))))) > 1
    )

    if use_ewald_j_split and _ir_mapping.size > 0:
        # IBZ inputs run on the EXPANDED full mesh (correctness: the
        # IBZ-native replication shortcut lacked the star AO rotations
        # D(R.k) = P(R).D(k).P(R)ᵀ and broke non-trivial crystals -- MgO
        # probe 2026-06-10, 8.25 Ha. See vibeqc.periodic_k_symmetry for
        # the transport groundwork and pbc_bipole.py for the full note).
        kmesh_full = _expand_ibz_kmesh_for_ewald_j(system, kmesh, plog)
        if len(list(kmesh_full.kpoints)) > len(k_points):
            kmesh = kmesh_full
            k_points = list(kmesh.kpoints)
            weights = np.asarray(kmesh.weights, dtype=float)
            _ir_mapping = np.asarray([], dtype=int)
        k_points_full = k_points
        weights_full = weights
    else:
        k_points_full = k_points
        weights_full = weights
    n_k = len(k_points)
    if n_k == 0:
        raise ValueError("kmesh has no k-points")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"kmesh.weights must sum to 1; got {weights.sum():.6f}")
    if (
        _input_true_multik
        and init_alpha is None
        and init_beta is None
        and initial_density_k is None
        and getattr(opts, "initial_guess", InitialGuess.AUTO) == InitialGuess.READ
    ):
        raise NotImplementedError(
            "run_pbc_bipole_uks: multi-k READ requires explicit complete "
            "real-space init_alpha/init_beta block sets; Gamma-only read "
            "densities cannot reconstruct the per-k state."
        )
    if use_ewald_j_split and n_k > 1 and _ir_mapping.size == 0:
        uniform_w = 1.0 / float(n_k)
        if not np.allclose(weights, uniform_w, atol=1e-9):
            raise ValueError(
                "use_ewald_j_split at multi-k requires uniform weights or IBZ mesh"
            )
    plog.info(f"k-mesh: {n_k} k-points, weights sum = {weights.sum():.4f}")

    # ---- Exchange/gauge resolution (option (b) Phase 4b, 2026-06-11) --
    # Mirrors run_pbc_bipole_uhf: per-spin exchange split at 3D Γ,
    # alpha_HF-scaled for hybrids; pure functionals get the gauge change
    # only (no exchange machinery).
    if exchange_exxdiv not in ("ewald", "none"):
        raise ValueError(
            f"run_pbc_bipole_uks: exchange_exxdiv must be 'ewald' or "
            f"'none'; got {exchange_exxdiv!r}"
        )
    _x_split_auto = use_exchange_ewald_split is None
    exchange_split_active = (
        bool(use_ewald_j_split) if _x_split_auto else bool(use_exchange_ewald_split)
    )
    if exchange_split_active and not use_ewald_j_split:
        raise ValueError(
            "run_pbc_bipole_uks: use_exchange_ewald_split=True requires "
            "the Ewald J split (use_ewald_j_split=True)."
        )
    # Multi-k split (option (b) Phase 3): needs the true Monkhorst-
    # Pack dimensions (q-channel tables, BvK-torus fold, supercell
    # Madelung); ad-hoc k-lists are rejected.
    _bvk_mesh: Optional[Tuple[int, int, int]] = None
    if exchange_split_active and n_k > 1:
        _mesh_attr = tuple(int(x) for x in getattr(kmesh, "mesh", (1, 1, 1)))
        if int(np.prod(_mesh_attr)) != n_k:
            if _x_split_auto:
                plog.info(
                    "  multi-k corrected exchange gauge needs a "
                    "Monkhorst-Pack mesh (BvK-torus fold + supercell ξ_M); "
                    "this ad-hoc k-list has no mesh metadata -> falling back "
                    "to the legacy gauge. Pass a monkhorst_pack(...) mesh "
                    "for the corrected gauge."
                )
                exchange_split_active = False
            else:
                raise ValueError(
                    "run_pbc_bipole_uks: the Ewald exchange split at multi-k "
                    "requires a Monkhorst-Pack BlochKMesh carrying its mesh "
                    f"dimensions (got mesh={_mesh_attr} for {n_k} k-points). "
                    "Build the mesh via monkhorst_pack(...); ad-hoc k-point "
                    "lists are not supported on the corrected gauge."
                )
        else:
            _bvk_mesh = _mesh_attr

    if uses_external_xc:
        validate_bipole_kmesh(
            kmesh,
            driver="run_pbc_bipole_uks external XC",
            require_complete=True,
        )
        if not exchange_split_active:
            raise NotImplementedError(
                "run_pbc_bipole_uks: periodic external XC requires the "
                "corrected Ewald exchange gauge. Leave "
                "use_exchange_ewald_split enabled and use a complete "
                "Monkhorst-Pack mesh or its valid expandable IBZ reduction."
            )

    reject_bipole_fractional_legacy_gauge(
        use_ewald_j_split=bool(use_ewald_j_split),
        exchange_split_active=exchange_split_active,
        fractional_occupations=use_fractional_density,
        driver="run_pbc_bipole_uks",
    )

    _ff_enable: Optional[bool] = False

    warn_bipole_legacy_multik_gauge(system, exchange_split_active, n_k, plog)
    warn_bipole_charged_cell(system, plog)
    _xi_madelung = 0.0
    if exchange_split_active and exchange_exxdiv == "ewald" and alpha_hf > 0.0:
        if n_k > 1:
            from .bipole_fock_ewald import probe_charge_madelung_supercell

            assert _bvk_mesh is not None
            _xi_madelung = probe_charge_madelung_supercell(system, _bvk_mesh)
        else:
            from .bipole_fock_ewald import probe_charge_madelung

            _xi_madelung = probe_charge_madelung(system)
    overlap_fold_drift: Optional[float] = None
    needs_pair_difference_xc = (
        xc_density_domain == PeriodicXCDensityDomain.PERIODIC_LATTICE
    )
    if needs_pair_difference_xc or getattr(opts, "use_periodic_becke", False):
        _xc_image_radius = float(
            getattr(opts, "becke_image_radius_bohr", 10.0)
        )
        # AUTO selects the density representation, not a separate grid reach.
        # Keep its finite AO bra images on the configured Becke partition.
        lat_opts_2e.becke_image_radius_bohr = _xc_image_radius
    if needs_pair_difference_xc:
        if _xc_image_radius <= 0.0:
            raise ValueError(
                "run_pbc_bipole_uks: explicit periodic-lattice XC needs "
                "becke_image_radius_bohr > 0"
            )
        if float(lat_opts_2e.cutoff_bohr) < _xc_image_radius:
            raise ValueError(
                "run_pbc_bipole_uks: XC cutoff_bohr must be at least "
                "becke_image_radius_bohr"
            )
    if exchange_split_active:
        plog.info(
            "  Gauge: corrected (full Bloch density, no spheropole"
            + (
                f"; per-spin exchange split exxdiv={exchange_exxdiv}, "
                f"alpha_HF={alpha_hf:g}"
                if alpha_hf > 0.0
                else "; pure functional"
            )
            + ")"
        )

    # Shared Ewald state
    ewald_options_1e = None
    omega_used = None
    ewald_cell_volume = None
    ewald_k_max: Optional[float] = None
    if system.dim == 3:
        from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff

        V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
        ewald_cell_volume = V_cell
        # #674: the K_SR erfc arm of the corrected exchange split is cut at
        # the Fock output cells (|g| <= lat_opts_2e.cutoff_bohr), so on that
        # route CRYSTAL's volume-only alpha is bounded below by
        # sqrt(-ln tol) / cutoff_bohr; the reciprocal envelope of J_LR / K_LR
        # follows the alpha in use (K_max / alpha stays CRYSTAL's 8.51).
        omega_used = resolve_bipole_ewald_alpha(
            V_cell,
            ewald_omega,
            lat_opts_2e,
            float(ewald_precision),
            erfc_exchange_arm_active=(exchange_split_active and exx.c_full != 0.0),
            plog=plog,
        )
        ewald_k_max = bipole_ewald_reciprocal_cutoff(V_cell, omega_used)
        # #478: the pinned alpha = 2.8/V^(1/3) makes erfc(alpha.R) at the
        # nearest image an L-INDEPENDENT 7.5e-5, so the 1e nuclear/Ewald
        # real-space sum must be sized from alpha, not from the AO cutoff.
        raise_bipole_ewald_real_cutoff(
            system,
            lat_opts_1e,
            omega_used,
            float(ewald_precision),
            plog,
        )
        ewald_options_1e = _crystal_ewald_options(
            lat_opts_1e,
            alpha_bohr_inv=omega_used,
            tolerance=float(ewald_precision),
            recip_cutoff_bohr_inv=ewald_k_max,
        )

    # Match the restricted driver: one padded ket-image ball converges the
    # Ewald J and screened-exchange erfc kernels, so the smaller omega owns
    # the radius.
    _sr_domain_omega = (
        min(float(omega_used), float(exx.omega_screen))
        if exx.is_screened and omega_used is not None
        else omega_used
    )
    _sr_image_extent = resolve_bipole_sr_image_extent(
        basis,
        system,
        lat_opts,
        lat_opts_2e,
        _sr_domain_omega,
        use_ewald_j_split=bool(use_ewald_j_split),
        sr_image_precision=sr_image_precision,
        sr_image_extent_bohr=sr_image_extent_bohr,
        plog=plog,
    )

    # DFT grid
    if getattr(opts, "use_periodic_becke", False):
        grid = build_periodic_becke_grid(
            system,
            grid_options=opts.grid,
            image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 10.0)),
        )
    else:
        grid = build_grid(system.unit_cell_molecule(), opts.grid)

    # One-electron integrals
    with plog.stage(
        "integrals_lattice", detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    ):
        S_lat, T_lat, V_lat, v_ne_lr_cache = build_bipole_one_electron_lattice(
            basis,
            system,
            lat_opts_1e,
            lat_opts_2e,
            ewald_options_1e,
            ewald_precision,
            ewald_k_max,
            v_ne_grid_options,
            plog,
            sym_log_style="rich",
        )
    cells = list(S_lat.cells)

    # SYM3b Fock-symmetry resolution -- before the density-list decision
    # (pair-resolved mode widens the density support; see
    # run_pbc_bipole_rhf).
    # A requested exact zone is not composable with the SYM3b reduced /
    # pair-resolved output domains yet (increment 1): let an AUTO
    # reduction yield to the explicit zone request; an explicit
    # use_fock_symmetry_reduce=True still reaches the resolver's
    # fail-closed raise below.
    _reduce_request = use_fock_symmetry_reduce
    if exact_zone_bohr is not None and _reduce_request is None:
        _reduce_request = False
    _fock_sym_map, _rep_cell_indices = resolve_bipole_fock_symmetry(
        system,
        basis,
        lat_opts_2e,
        use_fock_symmetry,
        _reduce_request,
        plog,
        per_spin=True,
        exchange_split_active=exchange_split_active,
        # Structural symmetry alone does not prove that an unrestricted
        # electronic state is symmetry-covariant.  Keep AUTO off for UKS;
        # explicit reduction remains an opt-in after the guard above.
        auto_reduce_safe=False,
    )
    _pair_mode = _fock_sym_map is not None and getattr(
        _fock_sym_map, "pair_resolved", False
    )

    from .pbc_bipole_common import resolve_bipole_exact_zone

    _exact_zone = resolve_bipole_exact_zone(
        system,
        lat_opts_2e,
        exact_zone_bohr,
        exchange_split_active=exchange_split_active,
        sr_image_extent=_sr_image_extent,
        pair_mode=_pair_mode,
        rep_cell_indices=_rep_cell_indices,
        plog=plog,
    )

    # ---- Lattice-fold convergence guard (gauge-independent) -------------
    # Shared guard (enforce_bipole_fold_support); hoisted above the gauge
    # condition 2026-08-06. The wide density list below is also required by
    # explicit periodic-lattice XC so every active pair difference exists.
    from .pbc_bipole_common import enforce_bipole_fold_support

    overlap_fold_drift = enforce_bipole_fold_support(
        basis,
        system,
        lat_opts_2e,
        method="uks",
        n_k=n_k,
        k_points=(k_points if n_k > 1 else None),
        exchange_split_active=exchange_split_active,
        plog=plog,
    )

    if exchange_split_active or needs_pair_difference_xc or lat_opts_2e.pair_complete_1e:
        if _pair_mode:
            # M3: pair-resolved density support (see run_pbc_bipole_rhf).
            cells_density = list(_fock_sym_map.density_domain.cells)
        else:
            cells_density = list(
                _sr_density_cells(basis, system, lat_opts_2e, _sr_image_extent)
            )

        if needs_pair_difference_xc:
            from .periodic_external_xc import (
                periodic_xc_difference_closed_cells,
            )

            exact_xc_cells = periodic_xc_difference_closed_cells(
                system,
                float(lat_opts_2e.cutoff_bohr),
                _xc_image_radius,
            )
            by_index = {
                tuple(np.asarray(cell.index, dtype=int)): cell
                for cell in (*cells_density, *exact_xc_cells)
            }
            cells_density = sorted(
                by_index.values(),
                key=lambda cell: (
                    float(np.dot(cell.r_cart, cell.r_cart)),
                    tuple(np.asarray(cell.index, dtype=int)),
                ),
            )

        label = "density" if exchange_split_active else "XC density"
        if _pair_mode:
            detail = (
                "pair-resolved Fock support plus atom-pair-complete XC "
                "differences"
                if needs_pair_difference_xc
                else "pair-resolved support at 2x cutoff; SR-masked privately"
            )
            plog.info(
                f"  {label} cell list: {len(cells_density)} cells "
                f"({detail})"
            )
        else:
            plog.info(
                f"  {label} cell list: {len(cells_density)} cells "
                "(atom-pair-complete difference support)"
            )
    else:
        cells_density = cells

    # Per-k matrices
    from .linear_dependence import scf_preflight_overlap_check

    S_k_list = []
    Hcore_k_list = []
    X_k_list = []
    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        V_k = np.asarray(bloch_sum(V_lat, k_arr))
        H_k = T_k + V_k
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        scf_preflight_overlap_check(S_k, plog=plog, label=f"S(k={k_idx})", basis=basis)
        X_k, n_kept = _canonical_orthogonalizer_complex(
            S_k,
            linear_dep_threshold,
            normalize_diag_first=canonical_orth_normalize_diag_first,
        )
        if max(n_alpha, n_beta) > n_kept:
            raise RuntimeError(
                f"run_pbc_bipole_uks: canonical orth at k={k_idx} dropped too many directions"
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

    # Nuclear repulsion
    e_nuc = (
        float(ewald_nuclear_repulsion(system, ewald_options_1e))
        if ewald_options_1e is not None
        else float(nuclear_repulsion_per_cell(system, lat_opts_1e))
    )

    # Initial guess
    guess = _initial_guess
    _atomic_spins = getattr(opts, "atomic_spins", None) or None
    _patom_seed_pending = guess == InitialGuess.PATOM
    fock_guess_per_k = (
        guess in (InitialGuess.SAP, InitialGuess.HUECKEL)
        and init_alpha is None
        and init_beta is None
        and _atomic_spins is None
    )
    guess_fock_k = Hcore_k_list
    if fock_guess_per_k:
        guess_fock_k = list(
            periodic_fock_guess_k(
                system,
                basis,
                k_points,
                guess,
                lattice_opts=lat_opts_2e,
                kinetic_lattice=T_lat,
                overlap_lattice=S_lat,
            )
        )

    C_alpha_per_k = []
    eps_alpha_per_k = []
    C_beta_per_k = []
    eps_beta_per_k = []
    for F_guess_k, X_k in zip(guess_fock_k, X_k_list):
        C_a, eps_a = _diag_in_orth_basis(F_guess_k, X_k)
        C_b, eps_b = _diag_in_orth_basis(F_guess_k, X_k)
        C_alpha_per_k.append(C_a.astype(complex))
        eps_alpha_per_k.append(eps_a)
        C_beta_per_k.append(C_b.astype(complex))
        eps_beta_per_k.append(eps_b)

    def _spin_density(C_per_k_local, n_occ_each):
        return unrestricted_spin_density(
            C_per_k_local,
            n_occ_each,
            n_k=n_k,
            kmesh=kmesh,
            cells_density=cells_density,
            exchange_split_active=(
                exchange_split_active or needs_pair_difference_xc
            ),
        )

    D_alpha_real = _spin_density(C_alpha_per_k, n_alpha)
    D_beta_real = _spin_density(C_beta_per_k, n_beta)

    # --- Smearing occupation helpers (open-shell: per-spin mu) ---
    from .smearing.fermi_dirac import fermi_dirac_occupations_per_k as _fd_per_k

    def _occupations_per_spin(eps_spin_per_k, n_spin):
        return unrestricted_occupations_per_spin(
            eps_spin_per_k,
            n_spin,
            smearing_T=smearing_T,
            weights=weights,
            system=system,
            kmesh=kmesh,
            bz_integration=bz_integration,
        )

    occ_alpha_per_k, mu_alpha, entropy_alpha = _occupations_per_spin(
        eps_alpha_per_k, n_alpha
    )
    occ_beta_per_k, mu_beta, entropy_beta = _occupations_per_spin(
        eps_beta_per_k, n_beta
    )
    entropy = entropy_alpha + entropy_beta

    if use_fractional_density:
        D_alpha_real = real_space_density_from_kpoints_fractional(
            C_alpha_per_k, occ_alpha_per_k, kmesh, cells_density
        )
        D_beta_real = real_space_density_from_kpoints_fractional(
            C_beta_per_k, occ_beta_per_k, kmesh, cells_density
        )
        if not exchange_split_active and not needs_pair_difference_xc:
            _zero_cross_cell_density(
                D_alpha_real, D_alpha_real.blocks[0].shape[0], n_k
            )
            _zero_cross_cell_density(
                D_beta_real, D_beta_real.blocks[0].shape[0], n_k
            )

    # Caller-supplied warm-start spin densities take precedence over
    # the SAD/Hcore guess engine -- see run_pbc_bipole_uhf for the
    # same contract. Used by the NEB driver for within-image density
    # warm-start for periodic NEB.
    if initial_density_k is not None:
        if init_alpha is not None or init_beta is not None:
            raise ValueError("supply either lattice or per-k restart densities, not both")
        if len(initial_density_k) != 2:
            raise ValueError("open-shell per-k READ requires alpha and beta block lists")
        from .guess import periodic_restart_lattice_density, normalize_spin_density_k_guess
        alpha_k, beta_k = normalize_spin_density_k_guess(
            *initial_density_k, S_k_list, weights, n_alpha, n_beta,
        )
        _read_alpha = periodic_restart_lattice_density(
            alpha_k, S_k_list, weights, n_alpha, kmesh, D_alpha_real.cells,
        )
        _read_beta = periodic_restart_lattice_density(
            beta_k, S_k_list, weights, n_beta, kmesh, D_beta_real.cells,
        )
        init_alpha, init_beta = _read_alpha.blocks, _read_beta.blocks
    if (init_alpha is not None) != (init_beta is not None):
        raise ValueError(
            "run_pbc_bipole_uks: init_alpha and init_beta must be "
            "provided together (both None or both populated)"
        )
    _constructed_alpha_k = _constructed_beta_k = None
    if init_alpha is not None and init_beta is not None:
        blocks_a = list(init_alpha)
        blocks_b = list(init_beta)
        if len(blocks_a) != len(D_alpha_real.cells):
            raise ValueError(
                f"run_pbc_bipole_uks: init_alpha has {len(blocks_a)} "
                f"blocks; expected {len(D_alpha_real.cells)}"
            )
        if len(blocks_b) != len(D_beta_real.cells):
            raise ValueError(
                f"run_pbc_bipole_uks: init_beta has {len(blocks_b)} "
                f"blocks; expected {len(D_beta_real.cells)}"
            )
        from .guess import real_lattice_restart_block, normalize_periodic_lattice_spin_restart
        for g_idx, (ba, bb) in enumerate(zip(blocks_a, blocks_b)):
            D_alpha_real.set_block(g_idx, real_lattice_restart_block(ba, basis.nbasis))
            D_beta_real.set_block(g_idx, real_lattice_restart_block(bb, basis.nbasis))
        D_alpha_real, D_beta_real = normalize_periodic_lattice_spin_restart(
            D_alpha_real, D_beta_real, S_k_list, weights, n_alpha, n_beta, kmesh,
        )
        initial_density_is_local = True
        density_from_c_per_k = False
    elif fock_guess_per_k:
        plog.info(
            f"initial guess: {guess.name} "
            "(per-k one-particle Hamiltonian diagonalisation; "
            "spin-resolved occupations)"
        )
        initial_density_is_local = False
        density_from_c_per_k = True
    else:
        # The shared open-shell engine owns both spin-balanced and
        # spin-polarized starts. An even total does not imply equal spins.
        _seed_guess = InitialGuess.SAD if _patom_seed_pending else guess
        D_guess = initial_densities_open_shell(
            system.unit_cell_molecule(),
            basis,
            n_alpha,
            n_beta,
            _seed_guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=lat_opts,
            atomic_spins=_atomic_spins,
            read_density_alpha=getattr(opts, "read_density_alpha", None),
            read_density_beta=getattr(opts, "read_density_beta", None),
            read_path=getattr(opts, "read_path", ""),
            overlap=S_k_list, weights=weights,
        )
        initial_density_is_local = D_guess is not None
        if D_guess is not None:
            D_a0, D_b0 = D_guess
            plog.info(f"initial guess: {guess.name} (spin densities via GuessEngine)")
            if n_k > 1:
                _constructed_alpha_k = [
                    np.asarray(D_a0, dtype=complex).copy() for _ in range(n_k)
                ]
                _constructed_beta_k = [
                    np.asarray(D_b0, dtype=complex).copy() for _ in range(n_k)
                ]
                D_alpha_real = real_space_density_from_per_k_density(
                    _constructed_alpha_k, kmesh, D_alpha_real.cells
                )
                D_beta_real = real_space_density_from_per_k_density(
                    _constructed_beta_k, kmesh, D_beta_real.cells
                )
            else:
                for g_idx, cell in enumerate(D_alpha_real.cells):
                    is_home = np.all(np.asarray(cell.index) == 0)
                    D_alpha_real.set_block(
                        g_idx, D_a0 if is_home else np.zeros_like(D_a0)
                    )
                    D_beta_real.set_block(
                        g_idx, D_b0 if is_home else np.zeros_like(D_b0)
                    )
        density_from_c_per_k = not initial_density_is_local
    if init_alpha is not None and init_beta is not None:
        _patom_seed_pending = False
    D_alpha_prev = None
    D_beta_prev = None

    # SCF aids
    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_pbc_bipole_uks: damping must be in [0,1); got {damping}"
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
    accel: Optional[MultiKPeriodicUHFAccelerator] = (
        MultiKPeriodicUHFAccelerator(opts) if use_diis else None
    )
    level_shift_static = float(getattr(opts, "level_shift", 0.0))
    if level_shift_schedule is not None and not isinstance(
        level_shift_schedule, LevelShiftSchedule
    ):
        raise TypeError(f"level_shift_schedule must be LevelShiftSchedule")

    # CRYSTAL-style FMIXING (per-spin, per-k).
    # Value resolved above in the options setup; log it here near the
    # SCF-loop preamble.
    if fock_mixing_value != 0.0:
        plog.info(
            f"fock mixing: CRYSTAL FMIXING "
            f"{100.0 * fock_mixing_value:.1f}% "
            "(previous Fock matrix weight)"
        )
    F_alpha_prev_mixed: Optional[List[np.ndarray]] = None
    F_beta_prev_mixed: Optional[List[np.ndarray]] = None

    # SPINLOCK PATTERN_HOLD reuses the per-k MOM machinery (gated to cycles
    # 2..spinlock_iterations) to hold the seeded broken-symmetry occupied set
    # then release -- protecting an ATOMSPIN seed.
    _spinlock_pattern_hold = (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.PATTERN_HOLD
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    )
    _spinlock_iters = int(getattr(opts, "spinlock_iterations", 0))
    if use_mom:
        plog.info("MOM: ON")
    elif _spinlock_pattern_hold:
        plog.info(f"SPINLOCK PATTERN_HOLD: MOM-hold for {_spinlock_iters} cycles")
    C_prev_occ_alpha_per_k = None
    C_prev_occ_beta_per_k = None
    if use_oda and use_diis:
        raise ValueError("use_oda and use_diis are mutually exclusive")
    if use_oda:
        plog.info(f"ODA: ON (trust lambda_max = {oda_trust_lambda_max})")

    # Ewald J-split cache
    j_lr_cache = v_ne_lr_cache
    if use_ewald_j_split:
        if system.dim != 3:
            raise ValueError(f"use_ewald_j_split requires dim=3")
        from .bipole_fock_ewald import _build_j_long_range_cache

        assert omega_used is not None
        if _pair_mode:
            # M3: J^LR blocks on the pair-resolved output template
            # (dedicated cache; never the radial V_ne one).
            j_lr_cache = _build_j_long_range_cache(
                basis,
                system,
                np.array(
                    [
                        np.asarray(c.r_cart, dtype=float)
                        for c in _fock_sym_map.cells
                    ],
                    dtype=float,
                ),
                omega_used,
                ewald_precision,
                K_max=ewald_k_max, lattice_opts=lat_opts_2e)
        elif j_lr_cache is None:
            cells_r_cart_arr = np.array(
                [np.asarray(c.r_cart, dtype=float) for c in cells], dtype=float
            )
            j_lr_cache = _build_j_long_range_cache(
                basis,
                system,
                cells_r_cart_arr,
                omega_used,
                ewald_precision,
                K_max=ewald_k_max, lattice_opts=lat_opts_2e)

    # Multi-k split: per-(k,k′) q-channel tables for the per-spin LR
    # exchange (hybrids only). Shares the J^LR w/K_max envelope.
    x_lr_cache = None
    if exchange_split_active and n_k > 1 and (alpha_hf > 0.0 or _patom_seed_pending):
        from .bipole_fock_ewald import build_k_exchange_long_range_cache

        assert j_lr_cache is not None and ewald_k_max is not None
        x_lr_cache = build_k_exchange_long_range_cache(
            basis,
            system,
            j_lr_cache,
            K_max=ewald_k_max,
        )
        plog.info(
            f"  K^LR q-channels: {n_k} distinct q = k-k′ shifts on the "
            f"shared K_max = {ewald_k_max:.2f} bohr⁻¹ envelope "
            f"(per spin, a_HF = {alpha_hf:g})"
        )

    def _split_k_density_list(density) -> List[np.ndarray]:
        return unrestricted_split_k_density_list(
            density, n_k=n_k, k_points=k_points, bvk_mesh=_bvk_mesh
        )

    # Incremental/differential J_SR(+K_SR) accumulators (opt-in; see
    # run_pbc_bipole_rhf). Hybrids build per-spin J_SR+K_SR (two
    # accumulators on D_a/D_b); pure functionals build J_SR on the total
    # density (one accumulator). Corrected gauge + DIIS only.
    incremental_jk_alpha = None
    incremental_jk_beta = None
    incremental_jk_total = None
    if use_incremental_fock:
        if exchange_split_active and not use_oda:
            from .bipole_fock_ewald import IncrementalJK

            if alpha_hf > 0.0:
                incremental_jk_alpha = IncrementalJK()
                incremental_jk_beta = IncrementalJK()
            else:
                incremental_jk_total = IncrementalJK()
            plog.info(
                "  incremental Fock (differential J_SR/K_SR via ΔD "
                "density-envelope screening): ON"
            )
        else:
            plog.info(
                "  incremental Fock requested but inactive "
                "(needs the corrected gauge + DIIS, not ODA)"
            )

    # (M2d) The UKS two-electron Fock assembly is routed through
    # pbc_bipole_fock.build_bipole_unrestricted_fock; the thin wrapper
    # below (defined after multipole + SYM3b resolution) adds the
    # per-spin V_xc and assembles F(k) in the driver.

    # ---- Multipole far-field config (resolve once before SCF loop) -----
    from .bipole_fock_multipole import (  # noqa: E402
        BipoleMultipoleConfig,
        resolve_multipole_config,
    )

    _mp_config = resolve_multipole_config(
        system,
        basis,
        lat_opts_2e,
        user_enable=_ff_enable,
        multipole_l_max=multipole_l_max,
    )
    if _mp_config.enabled:
        plog.info(
            f"  BIPOLE multipole far-field: ENABLED  "
            f"(L_max={_mp_config.L_max}, R_bipole={_mp_config.R_bipole:.1f} bohr, "
            f"n_cells={len(_mp_config.cache.cells) if _mp_config.cache else 0})"
        )
    else:
        plog.info(
            f"  BIPOLE multipole far-field: off  "
            f"(R_bipole={_mp_config.R_bipole:.1f} bohr, "
            f"cutoff={lat_opts_2e.cutoff_bohr:.1f} bohr)"
        )

    # SYM3b reduction is automatic for attached-symmetry crystals on the
    # corrected split; explicit choices were resolved before the density list.

    # ---- Disabled PDR 1988 Ch. II.4c quartet-prototype infrastructure ----
    from .bipole_far_field_infrastructure import (
        build_bipolar_far_field_infrastructure,
    )

    _spherical_buffer, _penetration_dispatch_j, _penetration_dispatch_k, _tensor_cache, _fock_kernel, _sym_recon = (
        build_bipolar_far_field_infrastructure(
            system,
            basis,
            lat_opts_2e,
            use_multipole_far_field=(
                (_ff_enable if _ff_enable is not None else False)
                and exchange_split_active
            ),
            exchange_split_active=exchange_split_active,
            multipole_l_max=multipole_l_max,
            ewald_omega=omega_used if use_ewald_j_split else 0.0,
            plog=plog,
        )
    )

    # ---- Shared unrestricted Fock-build context (M2d unification) --------
    # Bundle the per-run invariants the inline UKS Fock build captured.
    # Placed after multipole and symmetry resolution, before the PATOM
    # in-field step and SCF loop first call the wrapper.
    _fock_ctx = BipoleFockContext(
        basis=basis,
        system=system,
        lat_opts_2e=lat_opts_2e,
        use_ewald_j_split=use_ewald_j_split,
        exchange_split_active=exchange_split_active,
        n_k=n_k,
        omega_used=omega_used,
        ewald_precision=ewald_precision,
        ewald_cell_volume=ewald_cell_volume,
        n_elec=n_elec,
        xi_madelung=_xi_madelung,
        j_lr_cache=j_lr_cache,
        x_lr_cache=x_lr_cache,
        incremental_jk=None,
        rep_cell_indices=_rep_cell_indices,
        fock_sym_map=_fock_sym_map,
        mp_config=_mp_config,
        spherical_moment_buffer=_spherical_buffer,
        penetration_dispatch_j=_penetration_dispatch_j,
        penetration_dispatch_k=_penetration_dispatch_k,
        quartet_tensor_cache=_tensor_cache,
        far_field_fock_kernel=_fock_kernel,
        symmetry_reconstruction_map=_sym_recon,
        s_lat=S_lat,
        s_k_list=S_k_list,
        k_points=k_points,
        weights=weights,
        k_points_full=k_points_full,
        weights_full=weights_full,
        ir_mapping=_ir_mapping,
        bvk_mesh=_bvk_mesh,
        n_occ=n_alpha,
        plog=plog,
        sr_image_extent=_sr_image_extent,
        exact_zone_bohr=_exact_zone,
    )

    def _build_fock_for_density(
        D_alpha,
        D_beta,
        *,
        coeffs_alpha_for_rho,
        coeffs_beta_for_rho,
        use_incremental=True,
        patom_hf_like=False,
        reseed_incremental=False,
    ):
        fb = build_bipole_unrestricted_fock(
            _fock_ctx,
            D_alpha,
            D_beta,
            n_alpha=n_alpha,
            n_beta=n_beta,
            coeffs_alpha_for_rho=coeffs_alpha_for_rho,
            coeffs_beta_for_rho=coeffs_beta_for_rho,
            incremental_jk_alpha=incremental_jk_alpha,
            incremental_jk_beta=incremental_jk_beta,
            incremental_jk_total=incremental_jk_total,
            use_incremental=use_incremental,
            reseed_incremental=reseed_incremental,
            alpha_hf=alpha_hf,
            exchange_assembly=exx,
            patom_hf_like=patom_hf_like,
            use_reduced_j_only=True,
        )

        # XC potential -- open-shell UKS API: per-spin V_xc(g) + total
        # E_xc. PATOM is an HF-like in-field seed, independent of the
        # target functional, so skip XC for that one pre-SCF Fock build.
        xc_result = None
        if not patom_hf_like:
            D_xc_alpha = _density_set_gamma_or_lattice(S_lat, D_alpha)
            D_xc_beta = _density_set_gamma_or_lattice(S_lat, D_beta)
            xc_result = build_xc_periodic_uks(
                basis,
                system,
                grid,
                func,
                D_xc_alpha,
                D_xc_beta,
                lat_opts_2e,
                xc_density_domain,
            )

        # Per-k Fock: F_s(k) = F^2e_s(k) + V_xc,s(k) + Hcore(k) −
        # K_corr,s(k), hermitised after V_xc so a non-Hermitian Bloch
        # sum is symmetrised exactly as the pre-M2d inline build did.
        f_alpha_k_list = []
        f_beta_k_list = []
        F_a_2e_all = _bloch_sum_blocks_multi_k(
            fb.f2e_alpha_real.blocks, fb.f2e_alpha_real.cells, k_points
        )
        F_b_2e_all = _bloch_sum_blocks_multi_k(
            fb.f2e_beta_real.blocks, fb.f2e_beta_real.cells, k_points
        )
        for k_idx, k in enumerate(k_points):
            k_arr = np.asarray(k, dtype=float)
            F_a_2e = F_a_2e_all[k_idx]
            F_b_2e = F_b_2e_all[k_idx]
            if xc_result is None:
                Vxc_a_k = 0.0
                Vxc_b_k = 0.0
            else:
                Vxc_a_k = np.asarray(bloch_sum(xc_result.V_alpha, k_arr))
                Vxc_b_k = np.asarray(bloch_sum(xc_result.V_beta, k_arr))
            F_a = F_a_2e + Vxc_a_k + np.asarray(Hcore_k_list[k_idx], dtype=complex)
            F_b = F_b_2e + Vxc_b_k + np.asarray(Hcore_k_list[k_idx], dtype=complex)
            if fb.k_corr_alpha_per_k is not None:
                # Per-spin Ewald exchange split: one matrix per k (a
                # single Γ entry at n_k = 1).
                F_a = F_a - fb.k_corr_alpha_per_k[k_idx]
                F_b = F_b - fb.k_corr_beta_per_k[k_idx]
            f_alpha_k_list.append(0.5 * (F_a + F_a.conj().T))
            f_beta_k_list.append(0.5 * (F_b + F_b.conj().T))

        return _PBCBipoleUKSFockBuild(
            f2e_alpha_real=fb.f2e_alpha_real,
            f2e_beta_real=fb.f2e_beta_real,
            f_alpha_k_list=f_alpha_k_list,
            f_beta_k_list=f_beta_k_list,
            e_j_short_range=fb.e_j_short_range,
            e_j_long_range=fb.e_j_long_range,
            e_exchange=fb.e_exchange,
            e_j_multipole=fb.e_j_multipole,
            e_2e_k_correction=fb.e_2e_k_correction,
            e_exchange_finite_size=fb.e_exchange_finite_size,
            e_xc=(None if xc_result is None else float(xc_result.e_xc)),
            screened_exchange_execution=fb.screened_exchange_execution,
        )

    if _patom_seed_pending:
        plog.info("initial guess: PATOM (SAD + one BIPOLE in-field step)")
        patom_fock = _build_fock_for_density(
            D_alpha_real,
            D_beta_real,
            coeffs_alpha_for_rho=None,
            coeffs_beta_for_rho=None,
            use_incremental=False,
            patom_hf_like=True,
        )
        C_alpha_per_k = []
        eps_alpha_per_k = []
        C_beta_per_k = []
        eps_beta_per_k = []
        for idx in range(n_k):
            C_a, eps_a = _diag_in_orth_basis(
                patom_fock.f_alpha_k_list[idx],
                X_k_list[idx],
            )
            C_b, eps_b = _diag_in_orth_basis(
                patom_fock.f_beta_k_list[idx],
                X_k_list[idx],
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
        if use_fractional_density:
            D_alpha_real = real_space_density_from_kpoints_fractional(
                C_alpha_per_k, occ_alpha_per_k, kmesh, cells_density
            )
            D_beta_real = real_space_density_from_kpoints_fractional(
                C_beta_per_k, occ_beta_per_k, kmesh, cells_density
            )
            if not exchange_split_active and not needs_pair_difference_xc:
                _zero_cross_cell_density(
                    D_alpha_real, D_alpha_real.blocks[0].shape[0], n_k
                )
                _zero_cross_cell_density(
                    D_beta_real, D_beta_real.blocks[0].shape[0], n_k
                )
        else:
            D_alpha_real = _spin_density(C_alpha_per_k, n_alpha)
            D_beta_real = _spin_density(C_beta_per_k, n_beta)
        # Every first-cycle consumer must see the refined PATOM density,
        # including +U and acceleration on a repeated BvK cell list.
        _constructed_alpha_k = [
            (C * occ) @ C.conj().T
            for C, occ in zip(C_alpha_per_k, occ_alpha_per_k)
        ]
        _constructed_beta_k = [
            (C * occ) @ C.conj().T
            for C, occ in zip(C_beta_per_k, occ_beta_per_k)
        ]
        D_alpha_prev = None
        D_beta_prev = None
        initial_density_is_local = False
        density_from_c_per_k = True

    plog.banner("SCF (PBC BIPOLE UKS, direct-space)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    # ---- DFT+U setup (Increment 4d-bipole, UKS) -------------------------
    dft_plus_u_sites_cxx: List = []
    dft_plus_u_ao_groups: List[List[int]] = []
    if dft_plus_u:
        from ._vibeqc_core import _HubbardSiteCxx
        from .dft_plus_u import ao_group_indices

        ao_groups_map = ao_group_indices(basis)
        for site in dft_plus_u:
            key = (site.atom_index, site.l)
            if key not in ao_groups_map:
                raise ValueError(
                    f"run_pbc_bipole_uks: HubbardSite{key} has no AOs "
                    f"in the basis. Available channels: "
                    f"{sorted(ao_groups_map.keys())}"
                )
            dft_plus_u_sites_cxx.append(
                _HubbardSiteCxx(site.atom_index, site.l, site.U_eff_hartree)
            )
            dft_plus_u_ao_groups.append(ao_groups_map[key])

    def _apply_dft_plus_u(
        density_alpha: LatticeMatrixSet,
        density_beta: LatticeMatrixSet,
        fock_alpha_k: List[np.ndarray],
        fock_beta_k: List[np.ndarray],
        density_alpha_k=None,
        density_beta_k=None,
    ) -> float:
        """Apply spin-resolved +U to the density used for this Fock."""
        if not dft_plus_u_sites_cxx:
            return 0.0
        from ._vibeqc_core import _compute_dft_plus_u_multi_k_per_spin_cxx

        if density_alpha_k is not None:
            Pa_k_all, Pb_k_all = density_alpha_k, density_beta_k
        elif exchange_split_active:
            Pa_k_all = _split_k_density_list(density_alpha)
            Pb_k_all = _split_k_density_list(density_beta)
        else:
            Pa_k_all = _bloch_sum_blocks_multi_k(
                density_alpha.blocks, density_alpha.cells, k_points
            )
            Pb_k_all = _bloch_sum_blocks_multi_k(
                density_beta.blocks, density_beta.cells, k_points
            )
        P_alpha_k_for_u = [
            0.5 * (P_k + P_k.conj().T) for P_k in Pa_k_all
        ]
        P_beta_k_for_u = [
            0.5 * (P_k + P_k.conj().T) for P_k in Pb_k_all
        ]
        e_alpha, v_alpha = _compute_dft_plus_u_multi_k_per_spin_cxx(
            dft_plus_u_sites_cxx,
            dft_plus_u_ao_groups,
            S_k_list,
            P_alpha_k_for_u,
            list(weights),
        )
        e_beta, v_beta = _compute_dft_plus_u_multi_k_per_spin_cxx(
            dft_plus_u_sites_cxx,
            dft_plus_u_ao_groups,
            S_k_list,
            P_beta_k_for_u,
            list(weights),
        )
        v_alpha = np.asarray(v_alpha, dtype=complex)
        v_beta = np.asarray(v_beta, dtype=complex)
        for k_idx, S_k in enumerate(S_k_list):
            fock_a = fock_alpha_k[k_idx] + S_k @ v_alpha @ S_k
            fock_b = fock_beta_k[k_idx] + S_k @ v_beta @ S_k
            fock_alpha_k[k_idx] = 0.5 * (fock_a + fock_a.conj().T)
            fock_beta_k[k_idx] = 0.5 * (fock_b + fock_b.conj().T)
        return float(e_alpha) + float(e_beta)

    scf_trace = []
    energy_components = []
    def _exact_free_energy(d_alpha, d_beta, fb, e_dft_plus_u_value, entropy_value):
        """Free energy of the spin densities on the exact build ``fb`` (#116).

        Assembled exactly as the post-loop finalisation assembles the
        returned value (``fb.e_xc`` is the XC energy of the quadrature whose
        V_xc entered ``fb``), so the in-loop confirmation and the terminal
        refresh judge the same number.
        """
        d_tot = _combine_density_sets(basis, system, lat_opts_2e, d_alpha, d_beta)
        e_kin = _lattice_contract(d_tot, T_lat, operator_name="T")
        e_ne = _lattice_contract(d_tot, V_lat, operator_name="V_ne")
        e_2e = fb.e_2e_k_correction + 0.5 * (
            _lattice_contract(d_alpha, fb.f2e_alpha_real, operator_name="F2e")
            + _lattice_contract(d_beta, fb.f2e_beta_real, operator_name="F2e")
        )
        assert fb.e_xc is not None
        e_tot = (
            float(e_kin + e_ne + e_2e + float(fb.e_xc))
            + e_nuc
            + float(e_dft_plus_u_value)
        )
        if system.dim == 3 and not exchange_split_active:
            e_tot += compute_ext_el_spheropole(d_tot, basis, system, lat_opts)
        return float(e_tot) - smearing_T * float(entropy_value)

    E_prev = 0.0
    E_elec = 0.0
    e_xc = 0.0
    e_dft_plus_u = 0.0
    F_alpha_k_list = [np.zeros_like(H) for H in Hcore_k_list]
    F_beta_k_list = [np.zeros_like(H) for H in Hcore_k_list]
    converged = False
    iter_idx = 0
    #: Exact rebuild from the convergence confirmation (IID 514); reused by
    #: the post-loop finalisation so healthy rows pay nothing extra.
    _confirm_fb: Optional[_PBCBipoleUKSFockBuild] = None
    _confirm_e_dft_plus_u: Optional[float] = None

    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        # SPINLOCK PATTERN_HOLD: the accelerator (DIIS / EDIIS / ADIIS /
        # KDIIS -- whatever MultiKPeriodicUHFAccelerator resolved) is
        # suspended (no history recorded, no extrapolation, damping stays
        # live) while the hold is active. Fock extrapolation across
        # held-window iterates steers the SCF toward the symmetric attractor
        # by continuous orbital rotation -- a collapse the
        # occupation-selecting MOM hold cannot see -- and poisons the
        # post-release history with out-of-basin iterates. The history
        # starts fresh at release.
        hold_active = _spinlock_pattern_hold and iter_idx <= _spinlock_iters
        diis_active = (
            use_diis and iter_idx >= diis_start_iter and not hold_active
        )
        if iter_idx > 1 and damping > 0.0 and not diis_active:
            D_alpha_used = _damp_lattice_matrix(D_alpha_real, D_alpha_prev, damping)
            D_beta_used = _damp_lattice_matrix(D_beta_real, D_beta_prev, damping)
        else:
            D_alpha_used = D_alpha_real
            D_beta_used = D_beta_real

        d_used_is_damped = iter_idx > 1 and damping > 0.0 and not diis_active
        d_used_from_coeffs = (
            density_from_c_per_k
            and not (initial_density_is_local and iter_idx == 1)
            and not d_used_is_damped
        )
        fock_build = _build_fock_for_density(
            D_alpha_used,
            D_beta_used,
            coeffs_alpha_for_rho=(C_alpha_per_k if d_used_from_coeffs else None),
            coeffs_beta_for_rho=(C_beta_per_k if d_used_from_coeffs else None),
        )
        F_alpha_k_list = fock_build.f_alpha_k_list
        F_beta_k_list = fock_build.f_beta_k_list

        # ---- DFT+U: per-spin per-k Fock contribution -----------------
        # Apply the energy and potential to the same density used by the
        # base Fock builder. The converged-density rebuild below uses this
        # helper again so the returned energy/Fock/density remain one state.
        constructed_alpha_k = _constructed_alpha_k if iter_idx == 1 else None
        constructed_beta_k = _constructed_beta_k if iter_idx == 1 else None
        e_dft_plus_u = _apply_dft_plus_u(
            D_alpha_used,
            D_beta_used,
            F_alpha_k_list,
            F_beta_k_list,
            density_alpha_k=constructed_alpha_k,
            density_beta_k=constructed_beta_k,
        )

        # XC energy: from the SAME quadrature that produced this
        # iteration's per-spin V_xc (inside _build_fock_for_density on
        # D_alpha_used/D_beta_used) -- the loop used to run a second,
        # redundant build_xc_periodic_uks here every iteration just to
        # extract e_xc.
        assert fock_build.e_xc is not None
        e_xc = float(fock_build.e_xc)

        # Energy
        D_total_used = _combine_density_sets(
            basis, system, lat_opts_2e, D_alpha_used, D_beta_used
        )
        E_kin = _lattice_contract(D_total_used, T_lat, operator_name="T")
        E_ne = _lattice_contract(D_total_used, V_lat, operator_name="V_ne")
        E_2e = fock_build.e_2e_k_correction + 0.5 * (
            _lattice_contract(
                D_alpha_used, fock_build.f2e_alpha_real, operator_name="F2e_alpha"
            )
            + _lattice_contract(
                D_beta_used, fock_build.f2e_beta_real, operator_name="F2e_beta"
            )
        )
        E_elec = E_kin + E_ne + E_2e + e_xc

        # Gradient
        grad_norm_sum = 0.0
        error_alpha_k_list = []
        error_beta_k_list = []
        D_alpha_k_list = []
        D_beta_k_list = []
        D_k_split_guess_a: Optional[List[np.ndarray]] = None
        D_k_split_guess_b: Optional[List[np.ndarray]] = None
        if exchange_split_active and initial_density_is_local and iter_idx == 1:
            # BvK representatives (home block at Γ; exact torus fold at
            # multi-k) for local AND wide warm-start storage.
            D_k_split_guess_a = _split_k_density_list(D_alpha_used)
            D_k_split_guess_b = _split_k_density_list(D_beta_used)
        D_k_guess_fold_a: Optional[List[np.ndarray]] = None
        D_k_guess_fold_b: Optional[List[np.ndarray]] = None
        if (
            initial_density_is_local
            and iter_idx == 1
            and (D_k_split_guess_a is None or D_k_split_guess_b is None)
            and _constructed_alpha_k is None
        ):
            D_k_guess_fold_a = _bloch_sum_blocks_multi_k(
                D_alpha_used.blocks, D_alpha_used.cells, k_points
            )
            D_k_guess_fold_b = _bloch_sum_blocks_multi_k(
                D_beta_used.blocks, D_beta_used.cells, k_points
            )
        for idx in range(n_k):
            if initial_density_is_local and iter_idx == 1:
                if _constructed_alpha_k is not None:
                    D_a_k = _constructed_alpha_k[idx]
                    D_b_k = _constructed_beta_k[idx]
                elif D_k_split_guess_a is not None and D_k_split_guess_b is not None:
                    D_a_k = D_k_split_guess_a[idx]
                    D_b_k = D_k_split_guess_b[idx]
                else:
                    assert D_k_guess_fold_a is not None
                    assert D_k_guess_fold_b is not None
                    D_a_k = D_k_guess_fold_a[idx]
                    D_b_k = D_k_guess_fold_b[idx]
                D_a_k = 0.5 * (D_a_k + D_a_k.conj().T)
                D_b_k = 0.5 * (D_b_k + D_b_k.conj().T)
            else:
                C_a = C_alpha_per_k[idx]
                C_b = C_beta_per_k[idx]
                if not use_fractional_density:
                    C_a_occ = C_a[:, :n_alpha] if n_alpha > 0 else C_a[:, :0]
                    C_b_occ = C_b[:, :n_beta] if n_beta > 0 else C_b[:, :0]
                    D_a_k = C_a_occ @ C_a_occ.conj().T
                    D_b_k = C_b_occ @ C_b_occ.conj().T
                else:
                    # Fractional-occupation per-spin density
                    occ_a = np.asarray(occ_alpha_per_k[idx], dtype=float)
                    occ_b = np.asarray(occ_beta_per_k[idx], dtype=float)
                    C_a_full = np.asarray(C_a, dtype=np.complex128)
                    C_b_full = np.asarray(C_b, dtype=np.complex128)
                    D_a_k = (C_a_full * occ_a[None, :]) @ C_a_full.conj().T
                    D_b_k = (C_b_full * occ_b[None, :]) @ C_b_full.conj().T
            D_alpha_k_list.append(D_a_k)
            D_beta_k_list.append(D_b_k)
            S_k = S_k_list[idx]
            F_a_k = F_alpha_k_list[idx]
            F_b_k = F_beta_k_list[idx]
            err_a = F_a_k @ D_a_k @ S_k - (F_a_k @ D_a_k @ S_k).conj().T
            err_b = F_b_k @ D_b_k @ S_k - (F_b_k @ D_b_k @ S_k).conj().T
            error_alpha_k_list.append(err_a)
            error_beta_k_list.append(err_b)
            grad_norm_sum += float(weights[idx]) * float(
                np.sqrt(np.linalg.norm(err_a) ** 2 + np.linalg.norm(err_b) ** 2)
            )

        E_total = float(E_elec) + e_nuc + e_dft_plus_u

        # EXT EL-SPHEROPOLE -- uses total (alpha+beta) density.
        D_total_used = _combine_density_sets(
            basis, system, lat_opts_2e, D_alpha_used, D_beta_used
        )
        # EXT EL-SPHEROPOLE -- a 3D-Ewald-gauge correction, identically
        # zero in the direct (non-Ewald) gauge used for dim<3.
        if system.dim == 3 and not exchange_split_active:
            E_sphero = compute_ext_el_spheropole(D_total_used, basis, system, lat_opts)
            E_total += E_sphero
        else:
            E_sphero = None

        free_energy = E_total - smearing_T * entropy
        dE = free_energy - E_prev if iter_idx > 1 else 0.0
        check_scf_divergence(
            "run_pbc_bipole_uks", iter_idx, free_energy, grad_norm_sum, dE
        )

        diis_sub = max(
            accel.subspace_size if accel is not None else 0,
            accel.subspace_size if accel is not None else 0,
        )
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(free_energy),
                delta_e=float(dE),
                grad_norm=float(grad_norm_sum),
                diis_subspace=diis_sub,
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(free_energy),
            dE=float(dE),
            grad=float(grad_norm_sum),
            diis=diis_sub,
        )
        energy_components.append(
            PBCBipoleEnergyComponents(
                iter=int(iter_idx),
                e_total=float(E_total),
                e_electronic=float(E_elec),
                e_kinetic=float(E_kin),
                e_nuclear_attraction=float(E_ne),
                e_two_electron=float(E_2e),
                e_nuclear_repulsion=float(e_nuc),
                e_bielet_zone_ee=(None if use_ewald_j_split else float(E_2e)),
                e_j_short_range=fock_build.e_j_short_range,
                e_j_long_range=fock_build.e_j_long_range,
                e_exchange=fock_build.e_exchange,
                e_exchange_finite_size=fock_build.e_exchange_finite_size,
                e_ext_el_spheropole=E_sphero,
                e_dft_plus_u=float(e_dft_plus_u),
            )
        )
        plog.energy_decomposition(
            iter_idx,
            E_kin=float(E_kin),
            E_ne=float(E_ne),
            E_2e=float(E_2e),
            E_elec=float(E_elec),
            E_nuc=float(e_nuc),
        )

        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm_sum < float(opts.conv_tol_grad)
        )

        # SCF-accelerator extrapolation. Skipped entirely (not even
        # recorded) while the PATTERN_HOLD window is active; see the
        # hold_active note at the loop head.
        if accel is not None and not hold_active:
            if constructed_alpha_k is not None:
                density_alpha_k_list = constructed_alpha_k
                density_beta_k_list = constructed_beta_k
            elif exchange_split_active:
                # Unprojected Bloch fold: BvK representatives per k
                # (home block at Γ; exact torus fold at multi-k).
                # Mirrors the RHF accelerator fold; only the bridged
                # EDIIS/ADIIS modes consume these densities.
                density_alpha_k_list = _split_k_density_list(D_alpha_used)
                density_beta_k_list = _split_k_density_list(D_beta_used)
            else:
                density_alpha_k_list = _bloch_sum_blocks_multi_k(
                    D_alpha_used.blocks,
                    D_alpha_used.cells,
                    k_points,
                )
                density_beta_k_list = _bloch_sum_blocks_multi_k(
                    D_beta_used.blocks,
                    D_beta_used.cells,
                    k_points,
                )
            F_a_ex, F_b_ex = accel.extrapolate_uhf(
                F_alpha_k_list,
                F_beta_k_list,
                error_alpha_k_list=error_alpha_k_list,
                error_beta_k_list=error_beta_k_list,
                density_alpha_k_list=density_alpha_k_list,
                density_beta_k_list=density_beta_k_list,
                energy=free_energy,
                mo_coeffs_alpha_k_list=C_alpha_per_k,
                mo_coeffs_beta_k_list=C_beta_per_k,
                n_alpha=n_alpha,
                n_beta=n_beta,
                weights=list(weights),
                cells=cells,
                kpoints=list(k_points),
            )
            # On the converged iteration, diagonalise the *physical* Fock
            # F(D) -- not the extrapolated one -- so the final orbitals/
            # density stay on the converged fixed point. A near-machine-
            # zero DIIS error history (SCF nailing the solution in one
            # step) makes the Pulay B-matrix singular and its extrapolated
            # Fock garbage. See the RHF twin in pbc_bipole.py and
            # tests/test_pbc_bipole_diis_converged_basin.py.
            if diis_active and not converged:
                F_alpha_k_list = F_a_ex
                F_beta_k_list = F_b_ex

        # --- FMIXING (per-spin, per-k, after DIIS, before level-shift)
        # Skipped on the converged iteration (see DIIS note above).
        if fock_mixing_value != 0.0 and not converged:
            if F_alpha_prev_mixed is not None:
                for spin_list, prev_list in [
                    (F_alpha_k_list, F_alpha_prev_mixed),
                    (F_beta_k_list, F_beta_prev_mixed),
                ]:
                    for idx in range(n_k):
                        F_mixed = (1.0 - fock_mixing_value) * spin_list[
                            idx
                        ] + fock_mixing_value * prev_list[idx]
                        spin_list[idx] = 0.5 * (F_mixed + F_mixed.conj().T)
            F_alpha_prev_mixed = [
                np.asarray(F, dtype=complex).copy() for F in F_alpha_k_list
            ]
            F_beta_prev_mixed = [
                np.asarray(F, dtype=complex).copy() for F in F_beta_k_list
            ]

        # Level shift
        if level_shift_schedule is not None:
            level_shift_b = level_shift_schedule.at(iter_idx)
        else:
            level_shift_b = level_shift_static
        if level_shift_b != 0.0 and not converged:
            F_alpha_for_diag = []
            F_beta_for_diag = []
            for idx in range(n_k):
                S_k = S_k_list[idx]
                D_a_k = D_alpha_k_list[idx]
                D_b_k = D_beta_k_list[idx]
                F_a_shift = apply_level_shift_k(
                    F_alpha_k_list[idx],
                    S_k,
                    D_a_k,
                    level_shift_b,
                    LevelShiftDensity.SPIN,
                )
                F_b_shift = apply_level_shift_k(
                    F_beta_k_list[idx],
                    S_k,
                    D_b_k,
                    level_shift_b,
                    LevelShiftDensity.SPIN,
                )
                F_alpha_for_diag.append(0.5 * (F_a_shift + F_a_shift.conj().T))
                F_beta_for_diag.append(0.5 * (F_b_shift + F_b_shift.conj().T))
        else:
            F_alpha_for_diag = F_alpha_k_list
            F_beta_for_diag = F_beta_k_list

        # Diagonalise
        new_C_alpha = []
        new_eps_alpha = []
        new_C_beta = []
        new_eps_beta = []
        for idx in range(n_k):
            C_a, eps_a = _diag_in_orth_basis(F_alpha_for_diag[idx], X_k_list[idx])
            C_b, eps_b = _diag_in_orth_basis(F_beta_for_diag[idx], X_k_list[idx])
            new_C_alpha.append(C_a)
            new_eps_alpha.append(eps_a)
            new_C_beta.append(C_b)
            new_eps_beta.append(eps_b)

        # MOM (use_mom: every cycle; SPINLOCK PATTERN_HOLD: cycles
        # 2..spinlock_iterations, then release).
        _mom_this_iter = use_mom or (
            _spinlock_pattern_hold and 1 < iter_idx <= _spinlock_iters
        )
        if _mom_this_iter and C_prev_occ_alpha_per_k is not None:
            for idx in range(n_k):
                for spin, (C_k, eps_k, n_occ_spin, C_prev_occ_k) in enumerate(
                    [
                        (
                            new_C_alpha[idx],
                            new_eps_alpha[idx],
                            n_alpha,
                            C_prev_occ_alpha_per_k[idx],
                        ),
                        (
                            new_C_beta[idx],
                            new_eps_beta[idx],
                            n_beta,
                            C_prev_occ_beta_per_k[idx],
                        ),
                    ]
                ):
                    if n_occ_spin == 0:
                        continue
                    S_k = S_k_list[idx]
                    sel = _mom_select(C_k, S_k, C_prev_occ_k, n_occ_spin, eps_new=eps_k)
                    n_kept_idx = C_k.shape[1]
                    virt_mask = np.ones(n_kept_idx, dtype=bool)
                    virt_mask[sel] = False
                    virt_sel = np.where(virt_mask)[0]
                    virt_sel = virt_sel[np.argsort(np.real(eps_k[virt_sel]))]
                    order = np.concatenate([sel, virt_sel])
                    if spin == 0:
                        new_C_alpha[idx] = C_k[:, order]
                        new_eps_alpha[idx] = eps_k[order]
                    else:
                        new_C_beta[idx] = C_k[:, order]
                        new_eps_beta[idx] = eps_k[order]

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

        if use_fractional_density:
            D_alpha_new = real_space_density_from_kpoints_fractional(
                C_alpha_per_k, occ_alpha_per_k, kmesh, cells_density
            )
            D_beta_new = real_space_density_from_kpoints_fractional(
                C_beta_per_k, occ_beta_per_k, kmesh, cells_density
            )
            if not exchange_split_active and not needs_pair_difference_xc:
                _zero_cross_cell_density(
                    D_alpha_new, D_alpha_new.blocks[0].shape[0], n_k
                )
                _zero_cross_cell_density(
                    D_beta_new, D_beta_new.blocks[0].shape[0], n_k
                )
        else:
            D_alpha_new = _spin_density(C_alpha_per_k, n_alpha)
            D_beta_new = _spin_density(C_beta_per_k, n_beta)

        # Require both newly diagonalised spin densities to be the same fixed
        # point as D_used before accepting convergence. This includes the
        # occupation-only motion that a finite-temperature/Gilat commutator
        # cannot see.
        if converged:
            density_fixed_point_residual = max(
                (
                    float(np.max(np.abs(np.asarray(new) - np.asarray(old))))
                    for new_set, old_set in (
                        (D_alpha_new.blocks, D_alpha_used.blocks),
                        (D_beta_new.blocks, D_beta_used.blocks),
                    )
                    for new, old in zip(new_set, old_set)
                ),
                default=0.0,
            )
            converged = density_fixed_point_residual < float(opts.conv_tol_grad)

        # ODA
        if use_oda:
            fock_naive = _build_fock_for_density(
                D_alpha_new,
                D_beta_new,
                coeffs_alpha_for_rho=C_alpha_per_k,
                coeffs_beta_for_rho=C_beta_per_k,
                use_incremental=False,  # off the per-iter ΔD chain
            )
            oda_step = _compute_oda_lambda(
                D_alpha_used,
                D_alpha_new,
                F_alpha_k_list,
                fock_naive.f_alpha_k_list,
                [np.asarray(k) for k in k_points],
                weights,
                trust_lambda_max=oda_trust_lambda_max,
            )
            _oda_mix(D_alpha_used, D_alpha_new, oda_step.lam)
            _oda_mix(D_beta_used, D_beta_new, oda_step.lam)
            D_alpha_prev = D_alpha_real
            D_beta_prev = D_beta_real
            D_alpha_real = D_alpha_used
            D_beta_real = D_beta_used
            density_from_c_per_k = oda_step.lam == 1.0
            plog.info(
                f"  ODA: lambda = {oda_step.lam:.4f} (g0 = {oda_step.g0:+.3e}, g1 = {oda_step.g1:+.3e})"
            )
        else:
            D_alpha_prev = D_alpha_used
            D_beta_prev = D_beta_used
            D_alpha_real = D_alpha_new
            D_beta_real = D_beta_new
            density_from_c_per_k = True

        # Snapshot for next iter MOM (use_mom: every cycle; PATTERN_HOLD: while
        # inside the hold window).
        if use_mom or (_spinlock_pattern_hold and iter_idx <= _spinlock_iters):
            C_prev_occ_alpha_per_k = [
                np.asarray(C_alpha_per_k[idx][:, :n_alpha]).copy()
                if n_alpha > 0
                else np.zeros((C_alpha_per_k[idx].shape[0], 0), dtype=complex)
                for idx in range(n_k)
            ]
            C_prev_occ_beta_per_k = [
                np.asarray(C_beta_per_k[idx][:, :n_beta]).copy()
                if n_beta > 0
                else np.zeros((C_beta_per_k[idx].shape[0], 0), dtype=complex)
                for idx in range(n_k)
            ]

        if damper is not None:
            damper.update(free_energy)
        E_prev = free_energy
        if converged:
            # IID 514: confirm the provisional exit on the EXACT operator at
            # the density just committed (see run_pbc_bipole_rhf).
            # #115: announce the cold exact rebuild and make the loop's
            # provisional state durable BEFORE it runs; on a converged run
            # this is the expensive post-SCF phase, and the post-loop
            # finalisation reuses it at no cost.
            with bipole_confirmation_phase(
                plog,
                iter_idx=iter_idx,
                energy=float(scf_trace[-1].energy),
                n_k=n_k,
                nbf=basis.nbasis,
            ):
                _confirm_fb = _build_fock_for_density(
                    D_alpha_real,
                    D_beta_real,
                    coeffs_alpha_for_rho=(
                        C_alpha_per_k if density_from_c_per_k else None
                    ),
                    coeffs_beta_for_rho=(
                        C_beta_per_k if density_from_c_per_k else None
                    ),
                    use_incremental=False,
                    # #116: re-sync the incremental chains to this exact build,
                    # so a failed confirmation continues on the operator it is
                    # judged by (a no-op when the loop exits here).
                    reseed_incremental=True,
                )
            # Dudarev et al., Phys. Rev. B 57, 1505 (1998), Eq. (6):
            # the exact one-electron operator includes U_eff(1/2 - n).
            _confirm_e_dft_plus_u = _apply_dft_plus_u(
                D_alpha_real,
                D_beta_real,
                _confirm_fb.f_alpha_k_list,
                _confirm_fb.f_beta_k_list,
            )
            _confirm_grad = unrestricted_bipole_commutator_norm(
                _confirm_fb.f_alpha_k_list,
                _confirm_fb.f_beta_k_list,
                (
                    _split_k_density_list(D_alpha_real)
                    if exchange_split_active
                    else _bloch_sum_blocks_multi_k(
                        D_alpha_real.blocks, D_alpha_real.cells, k_points
                    )
                ),
                (
                    _split_k_density_list(D_beta_real)
                    if exchange_split_active
                    else _bloch_sum_blocks_multi_k(
                        D_beta_real.blocks, D_beta_real.cells, k_points
                    )
                ),
                S_k_list,
                weights,
            )
            # #116: judge the exact operator's free energy at the committed
            # densities here too -- the delta the post-loop refresh writes
            # into the terminal trace row -- so the loop cannot exit on a
            # state the terminal check would refuse.
            _confirm_ok, _ = bipole_terminal_check(
                plog,
                phase="in_loop",
                iter_idx=iter_idx,
                loop_objective=float(scf_trace[-1].energy),
                exact_objective=_exact_free_energy(
                    D_alpha_real,
                    D_beta_real,
                    _confirm_fb,
                    _confirm_e_dft_plus_u,
                    entropy,
                ),
                loop_grad_norm=float(scf_trace[-1].grad_norm),
                exact_grad_norm=float(_confirm_grad),
                conv_tol_energy=float(opts.conv_tol_energy),
                conv_tol_grad=float(opts.conv_tol_grad),
            )
            converged = _confirm_ok and abs(dE) < float(opts.conv_tol_energy)
            # At the iteration cap this is already the exact final-density
            # operator, even if confirmation failed. Preserve it for reuse.
            if not converged and iter_idx < int(opts.max_iter):
                _confirm_fb = None
                _confirm_e_dft_plus_u = None
        if converged:
            break

    screened_exchange_execution = fock_build.screened_exchange_execution
    final_e_xc = float(e_xc)
    final_e_2e = float(E_2e)
    final_e_exchange = float(fock_build.e_exchange or 0.0)

    # ---- Post-loop: evaluate the exact density returned to callers.  The
    # last diagonalisation commits a new density even when the iteration
    # budget is exhausted; always rebuild to keep the result internally
    # consistent.
    with bipole_final_density_phase(
        plog,
        iter_idx=iter_idx,
        energy=E_total,
        converged=converged,
        n_k=n_k,
        nbf=basis.nbasis,
        reused=(_confirm_fb is not None),
    ):
        _fb = (
            _confirm_fb
            if _confirm_fb is not None
            else _build_fock_for_density(
                D_alpha_real,
                D_beta_real,
                coeffs_alpha_for_rho=(
                    C_alpha_per_k if density_from_c_per_k else None
                ),
                coeffs_beta_for_rho=(
                    C_beta_per_k if density_from_c_per_k else None
                ),
                use_incremental=False,
            )
        )
    F_alpha_k_list = _fb.f_alpha_k_list
    F_beta_k_list = _fb.f_beta_k_list
    if _confirm_fb is not None:
        assert _confirm_e_dft_plus_u is not None
        e_dft_plus_u = _confirm_e_dft_plus_u
    else:
        e_dft_plus_u = _apply_dft_plus_u(
            D_alpha_real,
            D_beta_real,
            F_alpha_k_list,
            F_beta_k_list,
        )

    # Canonicalize each final physical spin Fock at the authoritative
    # returned densities, then refill occupations.  The SCF-loop eigenpairs
    # may describe a preceding or accelerated operator on iteration-limit
    # exits.
    C_alpha_per_k = []
    eps_alpha_per_k = []
    C_beta_per_k = []
    eps_beta_per_k = []
    for F_a, F_b, X_k in zip(
        F_alpha_k_list,
        F_beta_k_list,
        X_k_list,
    ):
        C_a, eps_a = _diag_in_orth_basis(F_a, X_k)
        C_b, eps_b = _diag_in_orth_basis(F_b, X_k)
        C_alpha_per_k.append(np.asarray(C_a, dtype=complex))
        eps_alpha_per_k.append(np.asarray(eps_a, dtype=float))
        C_beta_per_k.append(np.asarray(C_b, dtype=complex))
        eps_beta_per_k.append(np.asarray(eps_b, dtype=float))
    _mom_final = use_mom or (
        _spinlock_pattern_hold and 1 < iter_idx <= _spinlock_iters
    )
    if _mom_final and C_prev_occ_alpha_per_k is not None:
        for idx in range(n_k):
            for spin, (C_k, eps_k, n_occ_spin, C_prev_occ_k) in enumerate(
                [
                    (
                        C_alpha_per_k[idx],
                        eps_alpha_per_k[idx],
                        n_alpha,
                        C_prev_occ_alpha_per_k[idx],
                    ),
                    (
                        C_beta_per_k[idx],
                        eps_beta_per_k[idx],
                        n_beta,
                        C_prev_occ_beta_per_k[idx],
                    ),
                ]
            ):
                if n_occ_spin == 0:
                    continue
                sel = _mom_select(
                    C_k,
                    S_k_list[idx],
                    C_prev_occ_k,
                    n_occ_spin,
                    eps_new=eps_k,
                )
                virt_mask = np.ones(C_k.shape[1], dtype=bool)
                virt_mask[sel] = False
                virt_sel = np.where(virt_mask)[0]
                virt_sel = virt_sel[np.argsort(np.real(eps_k[virt_sel]))]
                order = np.concatenate([sel, virt_sel])
                if spin == 0:
                    C_alpha_per_k[idx] = C_k[:, order]
                    eps_alpha_per_k[idx] = eps_k[order]
                else:
                    C_beta_per_k[idx] = C_k[:, order]
                    eps_beta_per_k[idx] = eps_k[order]
    occ_alpha_per_k, mu_alpha, entropy_alpha = _occupations_per_spin(
        eps_alpha_per_k, n_alpha
    )
    occ_beta_per_k, mu_beta, entropy_beta = _occupations_per_spin(
        eps_beta_per_k, n_beta
    )
    entropy = entropy_alpha + entropy_beta

    screened_exchange_execution = _fb.screened_exchange_execution
    # The exact final-density Fock already carries this quadrature energy.
    assert _fb.e_xc is not None
    final_e_xc = float(_fb.e_xc)
    D_tot = _combine_density_sets(
        basis, system, lat_opts_2e, D_alpha_real, D_beta_real
    )
    E_kin_f = _lattice_contract(D_tot, T_lat, operator_name="T")
    E_ne_f = _lattice_contract(D_tot, V_lat, operator_name="V_ne")
    E_2e_f = _fb.e_2e_k_correction + 0.5 * (
        _lattice_contract(D_alpha_real, _fb.f2e_alpha_real, operator_name="F2e")
        + _lattice_contract(D_beta_real, _fb.f2e_beta_real, operator_name="F2e")
    )
    E_elec = E_kin_f + E_ne_f + E_2e_f + final_e_xc
    final_e_2e = float(E_2e_f)
    final_e_exchange = float(_fb.e_exchange or 0.0)
    E_total = float(E_elec) + e_nuc + e_dft_plus_u
    # Fresh E_total doesn't include spheropole -- add it (3D only; the
    # term is zero in the direct gauge used for dim<3).
    if system.dim == 3 and not exchange_split_active:
        E_sphero_final = compute_ext_el_spheropole(D_tot, basis, system, lat_opts)
        E_total += E_sphero_final
    else:
        E_sphero_final = None
    energy_components[-1] = PBCBipoleEnergyComponents(
        iter=int(iter_idx),
        e_total=float(E_total),
        e_electronic=float(E_elec),
        e_kinetic=float(E_kin_f),
        e_nuclear_attraction=float(E_ne_f),
        e_two_electron=float(E_2e_f),
        e_nuclear_repulsion=float(e_nuc),
        e_bielet_zone_ee=(
            None if use_ewald_j_split else float(E_2e_f)
        ),
        e_ext_el_spheropole=E_sphero_final,
        e_j_short_range=_fb.e_j_short_range,
        e_j_long_range=_fb.e_j_long_range,
        e_exchange=_fb.e_exchange,
        e_exchange_finite_size=_fb.e_exchange_finite_size,
        e_j_multipole=_fb.e_j_multipole,
        e_dft_plus_u=float(e_dft_plus_u),
    )

    # BZ-weighted complex <S^2> and the basin diagnostic must describe the
    # final canonical eigenpairs and occupations exposed by the result.
    from .periodic_k_gdf import _multi_k_s_squared

    s2 = _multi_k_s_squared(
        n_alpha,
        n_beta,
        C_alpha_per_k,
        C_beta_per_k,
        S_k_list,
        weights,
        occ_alpha_k=occ_alpha_per_k,
        occ_beta_k=occ_beta_per_k,
    )
    basin_warning = smearing_basin_warning(
        smearing_T,
        [
            (eps_alpha_per_k, occ_alpha_per_k, n_alpha, 1.0),
            (eps_beta_per_k, occ_beta_per_k, n_beta, 1.0),
        ],
        entropy,
        "run_pbc_bipole_uks",
    )
    if basin_warning is not None:
        plog.info("  WARNING: " + basin_warning)
        warnings.warn(basin_warning, UserWarning, stacklevel=2)

    # The converged-density rebuild above can change E_total. Keep the
    # reported thermodynamic objective on that same final density; computing
    # it before the rebuild returned a stale Mermin free energy for UKS.
    free_energy_final = E_total - smearing_T * entropy
    final_grad_norm = unrestricted_bipole_commutator_norm(
        F_alpha_k_list,
        F_beta_k_list,
        (
            _split_k_density_list(D_alpha_real)
            if exchange_split_active
            else _bloch_sum_blocks_multi_k(
                D_alpha_real.blocks, D_alpha_real.cells, k_points
            )
        ),
        (
            _split_k_density_list(D_beta_real)
            if exchange_split_active
            else _bloch_sum_blocks_multi_k(
                D_beta_real.blocks, D_beta_real.cells, k_points
            )
        ),
        S_k_list,
        weights,
    )
    _loop_objective = (
        float(scf_trace[-1].energy) if scf_trace else float(free_energy_final)
    )
    _loop_grad = (
        float(scf_trace[-1].grad_norm) if scf_trace else float(final_grad_norm)
    )
    refresh_bipole_terminal_trace(
        scf_trace,
        float(free_energy_final),
        grad_norm=final_grad_norm,
    )
    _terminal_ok, _ = bipole_terminal_check(
        plog,
        phase="post_loop",
        iter_idx=iter_idx,
        loop_objective=_loop_objective,
        exact_objective=float(free_energy_final),
        loop_grad_norm=_loop_grad,
        exact_grad_norm=float(final_grad_norm),
        conv_tol_energy=float(opts.conv_tol_energy),
        conv_tol_grad=float(opts.conv_tol_grad),
    )
    if converged:
        converged = _terminal_ok
    plog.converged(
        n_iter=iter_idx,
        energy=free_energy_final,
        converged=converged,
    )

    return PBCBipoleUKSResult(
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=init_alpha is not None or init_beta is not None or initial_density_k is not None),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=float(E_total),
        e_electronic=float(E_elec),
        e_nuclear=e_nuc,
        e_ext_el_spheropole=E_sphero_final,
        e_xc=final_e_xc,
        e_coulomb=final_e_2e - final_e_exchange,
        e_hf_exchange=final_e_exchange,
        n_iter=iter_idx,
        converged=converged,
        s_squared=float(s2),
        s_squared_ideal=0.25 * (mult - 1) * (mult + 1),
        mo_energies_alpha=eps_alpha_per_k,
        mo_coeffs_alpha=C_alpha_per_k,
        fock_alpha=F_alpha_k_list,
        density_alpha=D_alpha_real,
        mo_energies_beta=eps_beta_per_k,
        mo_coeffs_beta=C_beta_per_k,
        fock_beta=F_beta_k_list,
        density_beta=D_beta_real,
        overlap=S_k_list,
        hcore=Hcore_k_list,
        scf_trace=scf_trace,
        ewald_alpha_bohr_inv=omega_used,
        sr_image_extent_bohr=_sr_image_extent,
        pair_resolved_fock_domain=bool(_fock_sym_map is not None),
        e_dft_plus_u=float(e_dft_plus_u),
        energy_components=energy_components,
        exchange_ewald_split=bool(exchange_split_active),
        exchange_exxdiv=(exchange_exxdiv if exchange_split_active else None),
        overlap_fold_drift=overlap_fold_drift,
        exact_zone_bohr=_exact_zone,
        screened_exchange_execution=screened_exchange_execution,
        basin_warning=basin_warning,
        functional=str(opts.functional),
        smearing_temperature=smearing_T,
        fermi_level=float(mu_alpha) if n_alpha > 0 else float(mu_beta),
        fermi_level_alpha=(float(mu_alpha) if n_alpha > 0 else None),
        fermi_level_beta=(float(mu_beta) if n_beta > 0 else None),
        entropy=float(entropy),
        free_energy=float(free_energy_final),
        occupations_alpha=[np.asarray(o, dtype=float) for o in occ_alpha_per_k],
        occupations_beta=[np.asarray(o, dtype=float) for o in occ_beta_per_k],
        kpoints_cart=np.asarray(k_points, dtype=float).reshape(-1, 3),
        kpoint_weights=np.asarray(weights, dtype=float).reshape(-1),
        fock_mixing=fock_mixing_value,
    )
