"""BIPOLE-style periodic RKS driver in CRYSTAL's electrostatic gauge.

The DFT counterpart of :mod:`vibeqc.pbc_bipole`. Pure functionals use
the exact analytical-FT, G=0-dropped periodic Hartree J in the same
neutral-background convention as the GDF reference; hybrid functionals
retain the CRYSTAL-gauge Ewald split so the HF exchange correction can
share the native ``build_jk_2e_real_space`` component builder.

This driver keeps the same multi-k SCF scaffold as the RHF BIPOLE
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
    BasisSet,
    BlochKMesh,
    EwaldOptions,
    Functional,
    GridOptions,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicKSOptions,
    PeriodicXCDensityDomain,
    PeriodicSystem,
    SCFIteration,
    bloch_sum,
    build_grid,
    build_xc_periodic,
    direct_lattice_cells,
    ewald_nuclear_repulsion,
    nuclear_repulsion_per_cell,
    real_space_density_from_kpoints,
    real_space_density_from_kpoints_fractional,
)
from .bipole_ext_el_pole import compute_ext_el_spheropole
from .guess import (
    _coerce_periodic_driver_guess,
    initial_density_closed_shell,
    periodic_fock_guess_k,
)
from .level_shift_schedule import LevelShiftSchedule
from .mom import select_occupied_by_max_overlap as _mom_select
from .oda import compute_oda_lambda as _compute_oda_lambda
from .oda import oda_mix_densities as _oda_mix
from .pbc_bipole_common import (
    bipole_confirmation_phase,
    bipole_final_density_phase,
    PBCBipoleEnergyComponents,
    _bloch_sum_blocks,
    _bloch_sum_blocks_multi_k,
    _compute_nuclear_lattice_ewald_reciprocal_ft,
    _crystal_ewald_options,
    _density_set_gamma_or_lattice,
    _emit_bipole_semantic_diagnostic,
    _expand_ibz_kmesh_for_ewald_j,
    _lattice_contract,
    _zero_cross_cell_density,
    bloch_h_core_and_orthogonalizer,
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
    restricted_bipole_commutator_norm,
    bipole_terminal_check,
    refresh_bipole_terminal_trace,
    resolve_fock_mixing,
    resolve_auto_fock_mixing,
    resolve_bipole_sr_image_extent,
    resolve_bipole_fock_symmetry,
    resolve_incremental_jk,
    warn_bipole_exact_j_core_tail,
    warn_bipole_charged_cell,
    warn_bipole_legacy_multik_gauge,
    smearing_basin_warning,
    validate_bipole_kmesh,
)
from .pbc_bipole_fock import (
    _sr_density_cells,
    BipoleFockContext,
    BipoleOutputCellFarmingExecution,
    BipoleScreenedExchangeExecution,
    build_bipole_restricted_fock,
    split_k_density_list,
)
from .periodic_grid import build_periodic_becke_grid
from .periodic_rhf_multi_k_ewald import (
    _damp_lattice_matrix,
    _diag_in_orth_basis,
)
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicSCFAccelerator,
)
from .periodic_screened_exchange import resolve_periodic_exchange
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .smearing import (
    SmearingOptions,
)
from .smearing import (
    closed_shell_periodic_occupations as _closed_shell_periodic_occupations,
    occupations_are_per_k_integer_aufbau as _occupations_are_per_k_integer_aufbau,
)
from .symmetry_integrals_reduced import (
    compute_nuclear_lattice_reduced,
)

__all__ = [
    "PBCBipoleRKSResult",
    "run_pbc_bipole_rks",
]


@dataclass
class PBCBipoleRKSResult:
    """Result of :func:`run_pbc_bipole_rks`."""

    energy: float
    e_electronic: float
    e_nuclear: float
    e_xc: float
    e_coulomb: float
    e_hf_exchange: float
    n_iter: int
    converged: bool

    mo_energies: List[np.ndarray]
    mo_coeffs: List[np.ndarray]
    fock: List[np.ndarray]
    overlap: List[np.ndarray]
    hcore: List[np.ndarray]

    density: LatticeMatrixSet
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
    # caller passed ``dft_plus_u=[HubbardSite(...)]``.
    e_dft_plus_u: float = 0.0
    energy_components: List[PBCBipoleEnergyComponents] = field(
        default_factory=list,
    )
    functional: str = ""
    # Smearing diagnostics (zero when smearing_temperature == 0).
    smearing_temperature: float = 0.0
    fermi_level: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations: List[np.ndarray] = field(default_factory=list)
    # Gauge provenance (option (b)): True when the run used the
    # corrected gauge (full-Bloch density, no spheropole, a_HF-scaled
    # Ewald exchange split for hybrids).
    exchange_ewald_split: bool = False
    exchange_exxdiv: Optional[str] = None
    # Non-None when finite-T smearing straddled the HOMO-LUMO gap and may
    # have selected a near-metallic basin (ionic-Γ basin trap); carries
    # the warning message. See smearing_basin_warning.
    basin_warning: Optional[str] = None
    # Cartesian k-points (bohr^-1) and weights this result spans, in the
    # same order as the per-k ``mo_coeffs`` / ``mo_energies`` lists. Lets
    # optional Gamma-only / single-k output writers locate Gamma instead of
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
    # BIPOLE-EXACT-ZONE increment 1: radius (bohr) of the exact erfc
    # output zone when restricted below the operator cutoff (None =
    # exact zone == cutoff). Analytic-gradient consumers fail closed
    # on a set value (FD remains the production force path).
    exact_zone_bohr: Optional[float] = None
    output_cell_farming_execution: Optional[
        BipoleOutputCellFarmingExecution
    ] = None

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_mesh: object = None
    restart_kpoints: object = None
    restart_weights: object = None


@dataclass
class _PBCBipoleRKSFockBuild:
    """Internal Fock-build bundle for the BIPOLE RKS driver."""

    f2e_real: LatticeMatrixSet
    f_k_list: List[np.ndarray]
    e_j_short_range: Optional[float] = None
    e_j_long_range: Optional[float] = None
    e_exchange: Optional[float] = None
    e_j_multipole: Optional[float] = None
    # k-space exchange correction (Ewald exchange split, a_HF-scaled):
    # lives in F(k), so its energy must be added to the
    # lattice-contracted E_2e by the caller.
    e_2e_k_correction: float = 0.0
    # The q+G=0 (Madelung / exxdiv='ewald') share of e_2e_k_correction on
    # its own, so the gauge is visible next to the total (#82).
    e_exchange_finite_size: Optional[float] = None
    # XC energy of the SAME grid quadrature whose V_xc entered f_k_list.
    # Consuming this instead of re-running build_xc_periodic keeps the
    # iteration's V_xc and e_xc from one build (they used to come from
    # two independent quadratures per iteration -- one discarded).
    e_xc: Optional[float] = None
    screened_exchange_execution: Optional[
        BipoleScreenedExchangeExecution
    ] = None
    output_cell_farming_execution: Optional[
        BipoleOutputCellFarmingExecution
    ] = None


def run_pbc_bipole_rks(
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
    multipole_l_max: int = 2,
    use_exact_ft_j: bool = False,
    use_exchange_ewald_split: Optional[bool] = None,
    exchange_exxdiv: str = "ewald",
    use_fock_symmetry: Optional[bool] = None,
    use_fock_symmetry_reduce: Optional[bool] = None,
    sr_image_precision: Optional[float] = 1e-6,
    sr_image_extent_bohr: Optional[float] = None,
    exact_zone_bohr: Optional[float] = None,
    farm_output_cells: bool = False,
    output_cell_farming_strategy: str = "cyclic",
    output_cell_farming_task_kind: str = "direct-eri-output-cell",
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    initial_density: Optional[Sequence[np.ndarray]] = None,
    initial_density_k: Optional[Sequence] = None,
    dft_plus_u: Optional[List["HubbardSite"]] = None,
    bz_integration: Optional[str] = None,
    xc_density_domain: PeriodicXCDensityDomain = (
        PeriodicXCDensityDomain.AUTO
    ),
) -> PBCBipoleRKSResult:
    """Multi-k closed-shell RKS via the CRYSTAL-gauge BIPOLE scaffold.

    Parameters
    ----------
    system, basis, kmesh
        As in :func:`run_pbc_bipole_rhf`.
    options
        :class:`PeriodicKSOptions` or None for PBE defaults.
    functional
        XC functional name (overrides ``options.functional`` if given).
    fock_mixing
        Optional previous-Fock weight override in ``[0, 1)``.  The caller's
        ``options.fock_mixing`` field is not rewritten.  A resolved zero can
        still activate the documented DIIS-off KS automatic value.
    use_exchange_ewald_split, exchange_exxdiv
        Exchange/gauge convention (option (b), 2026-06-11). ``None`` =
        auto: the corrected gauge is ON at 3D Γ-only sampling under the
        Ewald J split -- full-Bloch density (no Γ-locality projection,
        which also fixes the XC grid density r(r): the projected
        density dropped the cross-cell AO products), no EXT
        EL-SPHEROPOLE term, and for hybrids (a_HF > 0) the Ewald
        exchange split ``K = K_SR(erfc w) + K_LR(reciprocal K!=0) +
        (ξ_M - pi/(Vw^2)).S.D.S`` scaled by a_HF. Pure functionals need
        no exchange machinery -- the gauge change alone applies (and
        the wasted full-Coulomb K traversal of the legacy path is
        skipped). See :func:`run_pbc_bipole_rhf` for the full
        convention notes. The corrected gauge is the default at BOTH
        Γ and multi-k (Phase-5 flip 2026-06-13): at multi-k it adds the
        q = k-k′ LR-exchange channels + BvK-supercell Madelung
        correction (requires a Monkhorst-Pack mesh; an ad-hoc k-list
        falls back to the legacy gauge under the auto default). Pass
        ``use_exchange_ewald_split=False`` for the legacy gauge.
    bz_integration
        ``None`` / ``"smearing"`` use the existing Aufbau or
        Fermi-Dirac occupation path. ``"gilat"`` selects the
        parameter-free Gilat-Raubenheimer net at T = 0; it cannot be
        combined with finite-temperature smearing.
    sr_image_precision, sr_image_extent_bohr
        M5 erfc SR image-domain controls; see
        :func:`run_pbc_bipole_rhf`. Pure semilocal RKS routes its Hartree J
        through the same padded SR+LR erfc composition as every other route
        (exact-FT retirement, 2026-07-18), so these controls apply to it.
        Screened hybrids pad the shared SR J base; their dedicated screened
        exchange traversal is a separate implementation path.
    use_exact_ft_j
        Opt-in cross-check oracle (default ``False``): route the pure
        semilocal Hartree J through the exact analytic-FT Ewald builder
        (``VIBEQC_J_EWALD3D_KE`` reciprocal cutoff, default 200 Ha) instead
        of the production padded SR+LR composition. The oracle's finite-ke
        reciprocal tail undercounts *absolute* totals on dense-core cells
        (~0.5-1 Ha at ke=200 on MgO-class bases; the run warns with the
        required ke estimate) while cancelling in energy differences. Pure
        functionals on the corrected split only; fails closed otherwise.
    use_fock_symmetry_reduce
        ``None`` auto-enables pair-resolved reduction for attached-symmetry
        crystals on the corrected split. See :func:`run_pbc_bipole_rhf` for
        the explicit opt-out and enforcement-only forms.
    farm_output_cells
        Distribute complete direct-ERI real-space output blocks across MPI
        ranks and gather them before incremental Fock bookkeeping. Every
        task retains the full internal translation sum. Generic BIPOLE
        callers leave this off; the chi-CCM restricted four-center selector
        supplies ``"chi-direct-output-cell"`` as its execution label.
        Screened hybrids fail closed because their separate J and screened-K
        phases require a multi-phase execution record.
    All other parameters
        As in :func:`run_pbc_bipole_rhf`.

    Returns
    -------
    PBCBipoleRKSResult
    """
    reject_bipole_quartet_far_field(
        use_multipole_far_field,
        driver="run_pbc_bipole_rks",
    )
    reject_bipole_lone_non_gamma_kpoint(
        kmesh,
        driver="run_pbc_bipole_rks",
    )
    from ._vibeqc_core import PeriodicKSOptions as _PKSOpts

    caller_supplied_options = options is not None
    opts = options if caller_supplied_options else _PKSOpts()
    guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", None),
        driver="run_pbc_bipole_rks",
        supported=periodic_guess_capabilities('bipole', 'RKS', dim=getattr(system, "dim", 3), multi_k=(len(kmesh.kpoints) > 1 or np.prod(kmesh.mesh) > 1), transport='lattice'),
        restart_supplied=initial_density is not None or initial_density_k is not None,
    )
    _patom_seed_pending = guess == InitialGuess.PATOM
    reject_bipole_ecp_options(
        opts,
        driver="run_pbc_bipole_rks",
        basis=basis,
        system=system,
    )
    reject_bipole_solver_options(opts, driver="run_pbc_bipole_rks")
    if int(opts.max_iter) < 1:
        raise ValueError("run_pbc_bipole_rks: max_iter must be at least 1")
    if not isinstance(farm_output_cells, (bool, np.bool_)):
        raise TypeError("run_pbc_bipole_rks: farm_output_cells must be bool")
    farm_output_cells = bool(farm_output_cells)
    if output_cell_farming_strategy not in ("block", "cyclic"):
        raise ValueError(
            "run_pbc_bipole_rks: output_cell_farming_strategy must be "
            "'block' or 'cyclic'"
        )
    if (
        not isinstance(output_cell_farming_task_kind, str)
        or not output_cell_farming_task_kind.strip()
    ):
        raise ValueError(
            "run_pbc_bipole_rks: output_cell_farming_task_kind must be a "
            "non-empty string"
        )
    if use_oda:
        raise NotImplementedError(
            "run_pbc_bipole_rks: ODA is unavailable because a mixed "
            "line-search density has no single orbital representation; "
            "use DIIS instead"
        )
    if functional is not None:
        opts.functional = str(functional)
    if not getattr(opts, "functional", None):
        opts.functional = "pbe"
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError("run_pbc_bipole_rks: smearing_temperature must be >= 0")
    if bz_integration is not None:
        bz_integration = str(bz_integration).strip().lower()
        if bz_integration not in ("smearing", "gilat"):
            raise ValueError(
                "run_pbc_bipole_rks: bz_integration must be None, "
                f"'smearing', or 'gilat'; got {bz_integration!r}"
            )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise NotImplementedError(
            "run_pbc_bipole_rks: bz_integration='gilat' is a "
            "sharp-Fermi-surface occupation backend and cannot be combined "
            "with finite-temperature smearing."
        )
    use_fractional_occupations = smearing_T > 0.0 or use_gilat
    func = Functional(opts.functional, 1)  # spin-unpolarised
    reject_bipole_unsupported_ks_functional(
        func,
        driver="run_pbc_bipole_rks",
    )
    uses_external_xc = bool(getattr(func, "is_external", False))
    if uses_external_xc:
        from .periodic_external_xc import (
            _grid_options_for_external,
            _require_zero_temperature_external_xc,
        )

        _require_zero_temperature_external_xc(
            smearing_T,
            where="run_pbc_bipole_rks",
        )

        opts.grid = _grid_options_for_external(
            func,
            opts.grid if caller_supplied_options else None,
            where="run_pbc_bipole_rks",
        )
        if xc_density_domain != PeriodicXCDensityDomain.PERIODIC_LATTICE:
            raise ValueError(
                "run_pbc_bipole_rks: external XC requires "
                "xc_density_domain=PERIODIC_LATTICE"
            )
        if not bool(getattr(opts, "use_periodic_becke", False)):
            raise ValueError(
                "run_pbc_bipole_rks: external XC requires "
                "use_periodic_becke=True"
            )
    # CAM exchange assembly. alpha_hf is the FULL-RANGE exact-exchange
    # fraction and drives the Ewald exchange-split machinery (xi_M,
    # K_LR q-channels, K_corr). Screened hybrids (hse06) have c_full
    # = 0 -- none of that machinery engages; their erfc short-range K
    # is added inside build_bipole_restricted_fock (no seam, no
    # far-field, per the CRYSTAL RSH treatment). LR-heavy RSH fails
    # closed in the resolver.
    exx = resolve_periodic_exchange(func, where="run_pbc_bipole_rks")
    alpha_hf = float(exx.c_full)
    if farm_output_cells and exx.is_screened:
        raise NotImplementedError(
            "run_pbc_bipole_rks: direct-ERI output-cell farming is not "
            "available for screened hybrids because the current single-phase "
            "execution record cannot attest both the base J and separate "
            "screened-K traversals"
        )

    # CRYSTAL-style auto-FMIXING: fires only for DFT without DIIS (see
    # resolve_auto_fock_mixing for the Gap-B measurement basis).
    requested_fock_mixing = resolve_fock_mixing(
        opts,
        fock_mixing,
        where="run_pbc_bipole_rks",
    )
    fock_mixing_value = resolve_auto_fock_mixing(
        requested_fock_mixing,
        alpha_hf=alpha_hf,
        use_diis=bool(opts.use_diis),
        where="run_pbc_bipole_rks",
    )

    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)
    (
        use_ewald_j_split,
        use_ewald_j_split_auto,
        lat_opts_2e,
        lat_opts_1e,
    ) = prepare_bipole_lattice_options(system, lat_opts, use_ewald_j_split, plog)

    plog.info(
        f"PBC BIPOLE RKS (CRYSTAL-gauge) / cutoff {lat_opts.cutoff_bohr:.2f} bohr"
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
    if use_gilat:
        plog.info("  bz integration = Gilat-Raubenheimer sharp-Fermi net")
    plog.info(
        f"  F^2e (J + V_xc{'+ K' if exx.needs_exchange else ''}) : "
        f"{'EWALD_J_SPLIT' if use_ewald_j_split else lat_opts_2e.coulomb_method.name}"
    )

    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            f"run_pbc_bipole_rks: closed-shell RKS requires even electron "
            f"count; got {n_elec}"
        )
    if system.multiplicity != 1:
        raise ValueError(
            f"run_pbc_bipole_rks: requires multiplicity=1; got {system.multiplicity}"
        )
    n_occ = n_elec // 2

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
        and initial_density is None
        and initial_density_k is None
        and getattr(opts, "initial_guess", InitialGuess.AUTO) == InitialGuess.READ
    ):
        raise NotImplementedError(
            "run_pbc_bipole_rks: multi-k READ requires an explicit complete "
            "real-space initial_density block set; a Gamma-only read_density "
            "cannot reconstruct the per-k Bloch density."
        )
    plog.info(
        f"k-mesh: {n_k} k-point{'s' if n_k != 1 else ''}, "
        f"weights sum = {weights.sum():.4f}"
    )

    # ---- Exchange/gauge resolution (option (b), 2026-06-11) ----------
    # Mirrors run_pbc_bipole_rhf: corrected gauge at 3D Γ under the J
    # split (full-Bloch density, no spheropole; a_HF-scaled Ewald
    # exchange split for hybrids). Multi-k: explicit opt-in runs the
    # q!=0 LR-exchange channels (Phase 3); auto stays legacy pending
    # certification.
    if exchange_exxdiv not in ("ewald", "none"):
        raise ValueError(
            f"run_pbc_bipole_rks: exchange_exxdiv must be 'ewald' or "
            f"'none'; got {exchange_exxdiv!r}"
        )
    _x_split_auto = use_exchange_ewald_split is None
    exchange_split_active = (
        bool(use_ewald_j_split) if _x_split_auto else bool(use_exchange_ewald_split)
    )
    if exchange_split_active and not use_ewald_j_split:
        raise ValueError(
            "run_pbc_bipole_rks: use_exchange_ewald_split=True requires "
            "the Ewald J split (use_ewald_j_split=True)."
        )
    # Multi-k split (option (b) Phase 3): q-channel tables, the
    # BvK-torus density fold, and the supercell Madelung correction
    # all need the true Monkhorst-Pack dimensions; ad-hoc k-lists
    # (mesh = (1,1,1) placeholders) are rejected.
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
                    "run_pbc_bipole_rks: the Ewald exchange split at multi-k "
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
            driver="run_pbc_bipole_rks external XC",
            require_complete=True,
        )
        if not exchange_split_active:
            raise NotImplementedError(
                "run_pbc_bipole_rks: periodic external XC requires the "
                "corrected Ewald exchange gauge. Leave "
                "use_exchange_ewald_split enabled and use a complete "
                "Monkhorst-Pack mesh or its valid expandable IBZ reduction."
            )

    reject_bipole_fractional_legacy_gauge(
        use_ewald_j_split=bool(use_ewald_j_split),
        exchange_split_active=exchange_split_active,
        fractional_occupations=use_fractional_occupations,
        driver="run_pbc_bipole_rks",
    )

    warn_bipole_legacy_multik_gauge(system, exchange_split_active, n_k, plog)
    warn_bipole_charged_cell(system, plog)
    _xi_madelung = 0.0
    if exchange_split_active and exchange_exxdiv == "ewald" and (alpha_hf > 0.0 or _patom_seed_pending):
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
                "run_pbc_bipole_rks: explicit periodic-lattice XC needs "
                "becke_image_radius_bohr > 0"
            )
        if float(lat_opts_2e.cutoff_bohr) < _xc_image_radius:
            raise ValueError(
                "run_pbc_bipole_rks: XC cutoff_bohr must be at least "
                "becke_image_radius_bohr"
            )
    if exchange_split_active:
        plog.info(
            "  Gauge: corrected (full Bloch density, no spheropole"
            + (
                f"; exchange split exxdiv={exchange_exxdiv}, "
                f"a_HF={alpha_hf:g}"
                + (
                    f", ξ_M = {_xi_madelung:.6f} Ha"
                    if exchange_exxdiv == "ewald"
                    else ""
                )
                if alpha_hf > 0.0
                else "; pure functional -- no HF exchange"
            )
            + ")"
        )
    # Exact-FT retirement (maintainer ruling 2026-07-13, landed
    # 2026-07-18): pure semilocal RKS defaults to the same padded SR+LR
    # Ewald-J composition as every other erfc route. The M4b
    # interaction-resolved screening plus the M5 ``sr_image_precision``
    # padding repaired the historical SR ket-image truncation that
    # motivated the exact-FT default, while the exact-FT builder's
    # finite-ke reciprocal core tail undercounts absolute totals on
    # dense-core cells by ~0.5-1 Ha at the default ke=200 (cancels in
    # differences; breaks absolute CRYSTAL/PySCF parity). The analytic-FT
    # builder stays available as the explicit ``use_exact_ft_j``
    # cross-check oracle and fails closed off its validated envelope.
    if use_exact_ft_j:
        if alpha_hf != 0.0 or exx.needs_exchange:
            raise ValueError(
                "run_pbc_bipole_rks: use_exact_ft_j is the pure-functional "
                "exact-FT J cross-check oracle; it does not apply to hybrid "
                f"or screened-hybrid functionals (a_HF = {alpha_hf:g}, "
                f"screened = {bool(exx.is_screened)})."
            )
        if not (use_ewald_j_split and exchange_split_active):
            raise ValueError(
                "run_pbc_bipole_rks: use_exact_ft_j requires the corrected "
                "exchange-split gauge under the Ewald J split (3D periodic "
                "Monkhorst-Pack sampling); the legacy gauge has no exact-FT "
                "J route."
            )
        if use_multipole_far_field:
            raise ValueError(
                "run_pbc_bipole_rks: use_exact_ft_j cannot be combined "
                "with the retired multipole far-field research artifact."
            )
    use_exact_ewald_j_for_pure_rks = bool(use_exact_ft_j)
    exact_j_ke_cutoff = 200.0
    exact_j_chunk_size = 512
    if use_exact_ewald_j_for_pure_rks:
        import os

        exact_j_ke_cutoff = float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0"))
        exact_j_chunk_size = max(
            1,
            int(os.environ.get("VIBEQC_J_EWALD3D_CHUNK", "512")),
        )
        plog.info(
            "  Pure-RKS Hartree J: exact analytic-FT Ewald cross-check "
            f"oracle (G=0 dropped, ke_cutoff={exact_j_ke_cutoff:g} Ha)"
        )
        warn_bipole_exact_j_core_tail(basis, exact_j_ke_cutoff, plog)

    # ---- Shared Ewald state (one a across V_ne / E_nn / J_LR) -------
    ewald_options_1e: Optional[EwaldOptions] = None
    omega_used: Optional[float] = None
    ewald_cell_volume: Optional[float] = None
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
            erfc_exchange_arm_active=(exchange_split_active and (exx.c_full != 0.0 or _patom_seed_pending)),
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

    _erfc_sr_build_active = (
        not use_exact_ewald_j_for_pure_rks or bool(exx.is_screened) or _patom_seed_pending
    )
    # The shared padded ket-image ball must converge both active erfc
    # kernels. HSE06's screened exchange (omega=0.11) can be longer-ranged
    # than the Ewald J kernel, so the smaller omega owns the radius.
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
        # The opt-in exact-FT J oracle has no erfc SR traversal to pad;
        # every production route (incl. pure RKS since the exact-FT
        # retirement) and screened hybrids pad the shared J/K domain.
        erfc_sr_build_active=_erfc_sr_build_active,
        plog=plog,
    )
    if farm_output_cells and (
        not exchange_split_active or _sr_image_extent is None
    ):
        raise NotImplementedError(
            "run_pbc_bipole_rks: direct-ERI output-cell farming currently "
            "requires the corrected 3D Ewald exchange split and a padded "
            "short-range image domain"
        )

    # ---- DFT grid ----------------------------------------------------
    # Wrapped in plog.stage so a job that stalls here (periodic Becke
    # partitioning over image cells can be minutes on dense/large cells)
    # shows ``[dft_grid]`` as its last live line instead of going dark --
    # the pre-SCF setup that walltimed P13 MgO BIPOLE/XC left no marker
    # localizing which stage was slow.
    with plog.stage(
        "dft_grid",
        detail=(
            "periodic Becke" if getattr(opts, "use_periodic_becke", False)
            else "molecular grid on unit cell"
        ),
    ):
        if getattr(opts, "use_periodic_becke", False):
            grid = build_periodic_becke_grid(
                system,
                grid_options=opts.grid,
                image_radius_bohr=float(
                    getattr(opts, "becke_image_radius_bohr", 10.0)
                ),
            )
        else:
            grid = build_grid(system.unit_cell_molecule(), opts.grid)

    # ---- Real-space one-electron integrals ---------------------------
    with plog.stage(
        "integrals_lattice",
        detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr",
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
            sym_log_style="plain",
        )
    cells = list(S_lat.cells)
    plog.info(f"n_cells in lattice sum = {len(cells)}")

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
        exchange_split_active=exchange_split_active,
        auto_reduce_safe=(
            not _erfc_sr_build_active or _sr_image_extent is not None
        ),
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
        method="rks",
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
                else "pair-resolved support at 2x cutoff"
            )
            plog.info(
                f"  {label} cell list: {len(cells_density)} cells "
                f"({detail}; SR tensor has "
                f"{_fock_sym_map.density_domain.n_triples} qualifying pairs)"
            )
        else:
            plog.info(
                f"  {label} cell list: {len(cells_density)} cells "
                "(atom-pair-complete difference support)"
            )
    else:
        cells_density = cells

    # ---- Per-k S(k), Hcore(k), X(k) ---------------------------------
    from .linear_dependence import (
        check_overlap_matrix,
        format_linear_dependence_report,
        raise_if_severe,
        scf_preflight_overlap_check,
    )

    S_k_list: List[np.ndarray] = []
    Hcore_k_list: List[np.ndarray] = []
    X_k_list: List[np.ndarray] = []
    overlap_reports = []
    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k, T_k, V_k, H_k, X_k, n_kept = bloch_h_core_and_orthogonalizer(
            S_lat,
            T_lat,
            V_lat,
            k_arr,
            linear_dep_threshold,
            canonical_orth_normalize_diag_first,
        )
        overlap_label = f"S(k={k_idx}, k_cart={k_arr.round(4).tolist()})"
        if n_k <= 16:
            report = scf_preflight_overlap_check(
                S_k,
                plog=plog,
                label=overlap_label,
                basis=basis,
            )
        else:
            report = check_overlap_matrix(
                S_k,
                basis=basis,
                label=overlap_label,
            )
            if report.severity != "ok":
                prefix = {
                    "warn": "WARN",
                    "error": "ERROR",
                    "critical": "CRITICAL",
                }[report.severity]
                plog.info(
                    f"[{prefix}] overlap [{overlap_label}]: "
                    f"nbf={report.n_basis}, "
                    f"min eig={report.min_eigenvalue:+.2e}, "
                    f"cond={report.condition_number:.2e}"
                )
                plog.write_raw(format_linear_dependence_report(report))
            raise_if_severe(report)
        overlap_reports.append(report)
        if n_occ > n_kept:
            raise RuntimeError(
                f"run_pbc_bipole_rks: canonical orth at k={k_idx} "
                f"dropped too many directions (n_occ={n_occ}, "
                f"n_kept={n_kept})"
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

    # ---- Nuclear repulsion -------------------------------------------
    if ewald_options_1e is not None:
        e_nuc = float(ewald_nuclear_repulsion(system, ewald_options_1e))
    else:
        e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts_1e))
    plog.info(f"E_nuc per cell = {e_nuc:+.10f} Ha")

    # ---- Initial guess -----------------------------------------------
    fock_guess_per_k = (
        initial_density is None
        and guess in (InitialGuess.SAP, InitialGuess.HUECKEL)
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
    C_per_k: List[np.ndarray] = []
    eps_per_k: List[np.ndarray] = []
    for F_guess_k, X_k in zip(guess_fock_k, X_k_list):
        C_k, eps_k = _diag_in_orth_basis(F_guess_k, X_k)
        C_per_k.append(C_k.astype(complex))
        eps_per_k.append(eps_k)
    n_occ_per_k = [n_occ] * n_k

    n_electrons_per_cell = float(n_elec)

    def _occupations_from_eps(
        eps_per_k_local: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], float, float]:
        if use_gilat:
            from .bz_integration import gilat_occupations_for_kmesh

            occ_gr, ef_gr = gilat_occupations_for_kmesh(
                system,
                kmesh,
                eps_per_k_local,
                n_electrons_per_cell,
                spin_degeneracy=2.0,
            )
            return occ_gr, float(ef_gr), 0.0
        # MOM permutes C(k) / eps(k) so its selected occupied states lead;
        # the T = 0 fill must occupy that block positionally rather than
        # re-pick by energy across the mesh (GitLab #725).
        return _closed_shell_periodic_occupations(
            eps_per_k_local,
            weights,
            n_electrons_per_cell,
            n_occ,
            smearing_T,
            fixed_occupied_subspace=bool(use_mom),
        )

    occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)

    def _fractional_density_needed(
        occ_per_k_local: Sequence[np.ndarray],
    ) -> bool:
        """Whether these occupations need the occupation-driven builder.

        The fixed ``C[:, :n_occ]`` slice builder is exact only for the legacy
        per-k integer Aufbau pattern. The T = 0 fill is one global Fermi level
        over the mesh (#85), so a band-overlap mesh can occupy a different
        number of bands per k (or a shared fractional group at mu), which no
        fixed slice reproduces; those occupations must be built from
        themselves, as the native multi-k GDF route does (#509).
        """
        if _occupations_are_per_k_integer_aufbau(occ_per_k_local, n_occ):
            return False
        reject_bipole_fractional_legacy_gauge(
            use_ewald_j_split=bool(use_ewald_j_split),
            exchange_split_active=exchange_split_active,
            fractional_occupations=True,  # runtime-discovered band-overlap fill
            driver="run_pbc_bipole_rks",
        )
        if use_mom or use_oda:
            raise NotImplementedError(
                "run_pbc_bipole_rks: the zero-temperature Brillouin-zone fill "
                "is not per-k-integer-aufbau on this mesh (the bands overlap "
                "the Fermi level), so there is no sharp occupied subspace for "
                "use_mom/use_oda. Disable use_mom/use_oda or add finite "
                "smearing."
            )
        return True

    # Per-k -> real-space density fold over the wide (2x-cutoff) cell list
    # is O(n_k x Ncells x Nbf^2) and silent; a prime multi-k cost for the
    # LDA (2,2,2) row, so bracket it for localization.
    with plog.stage(
        "initial_density_fold",
        detail=f"per-k SAD/Hcore -> real space over {len(cells_density)} cells",
    ):
        if _fractional_density_needed(occ_per_k):
            D_real = real_space_density_from_kpoints_fractional(
                C_per_k,
                occ_per_k,
                kmesh,
                cells_density,
            )
        else:
            D_real = real_space_density_from_kpoints(
                C_per_k,
                n_occ_per_k,
                kmesh,
                cells_density,
            )

    if not exchange_split_active and not needs_pair_difference_xc:
        _zero_cross_cell_density(D_real, basis.nbasis, n_k)

    # Caller-supplied warm-start density takes precedence over the
    # SAD/Hcore guess engine. Block ordering matches the canonical
    # ``direct_lattice_cells(kmesh)`` ordering -- same contract as the
    # RHF driver. Used by the NEB driver for within-image density
    # warm-start for periodic NEB.
    _constructed_density_k = None
    if initial_density_k is not None:
        if initial_density is not None:
            raise ValueError("supply either lattice or per-k restart densities, not both")
        from .guess import periodic_restart_lattice_density
        _read_lattice = periodic_restart_lattice_density(
            initial_density_k, S_k_list, weights, n_elec, kmesh, D_real.cells,
        )
        initial_density = _read_lattice.blocks
    if initial_density is not None:
        blocks_in = list(initial_density)
        if len(blocks_in) != len(D_real.cells):
            raise ValueError(
                f"run_pbc_bipole_rks: initial_density has "
                f"{len(blocks_in)} blocks; expected {len(D_real.cells)}"
            )
        from .guess import real_lattice_restart_block, normalize_periodic_lattice_restart
        for g_idx, block in enumerate(blocks_in):
            D_real.set_block(g_idx, real_lattice_restart_block(block, basis.nbasis))
        plog.info("initial guess: caller-supplied density (warm-start)")
        D_real = normalize_periodic_lattice_restart(D_real, S_k_list, weights, n_elec, kmesh)
        initial_density_is_local = True
        density_from_c_per_k = False
    elif fock_guess_per_k:
        plog.info(
            f"initial guess: {guess.name} "
            "(per-k one-particle Hamiltonian diagonalisation)"
        )
        initial_density_is_local = False
        density_from_c_per_k = True
    else:
        D_engine = initial_density_closed_shell(
            system.unit_cell_molecule(),
            basis,
            n_occ,
            InitialGuess.SAD if _patom_seed_pending else guess,
            is_periodic=True,
            periodic_system=system,
            lattice_opts=lat_opts_2e,
            # READ restart (Γ-only): prior g=0 cell density (pre-resolved from
            # read_from, or read + projected from read_path). Ignored unless READ.
            read_density=getattr(opts, "read_density", None),
            read_path=getattr(opts, "read_path", ""),
            overlap=S_k_list, weights=weights,
        )
        if D_engine is not None:
            plog.info(f"initial guess: {guess.name} (density via GuessEngine)")
            if n_k > 1:
                # Fold the selected D(k), including its image-cell blocks,
                # with the same inverse-Bloch convention as the SCF loop.
                # SAD need not be idempotent; it must be one consistent
                # density in real and reciprocal space (Sun 2017, Eq. 3).
                _constructed_density_k = [
                    np.asarray(D_engine, dtype=complex).copy() for _ in range(n_k)
                ]
                D_real = real_space_density_from_per_k_density(
                    _constructed_density_k, kmesh, D_real.cells
                )
            else:
                for g_idx, cell in enumerate(D_real.cells):
                    is_home = np.all(np.asarray(cell.index) == 0)
                    D_real.set_block(
                        g_idx, D_engine if is_home else np.zeros_like(D_engine)
                    )
        else:
            plog.info(f"initial guess: {guess.name} (Hcore-diag per k)")
        initial_density_is_local = D_engine is not None
        density_from_c_per_k = not initial_density_is_local
    D_real_prev: Optional[LatticeMatrixSet] = None

    # ---- SCF aids ----------------------------------------------------
    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_pbc_bipole_rks: damping must be in [0,1); got {damping}"
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
    accel: Optional[MultiKPeriodicSCFAccelerator] = (
        MultiKPeriodicSCFAccelerator(opts) if use_diis else None
    )

    level_shift_static = float(getattr(opts, "level_shift", 0.0))
    if level_shift_schedule is not None and not isinstance(
        level_shift_schedule,
        LevelShiftSchedule,
    ):
        raise TypeError(
            f"level_shift_schedule must be LevelShiftSchedule; "
            f"got {type(level_shift_schedule).__name__}"
        )

    # CRYSTAL-style FMIXING (same convention as RHF BIPOLE driver).
    # Value resolved above in the options setup; log it here near the
    # SCF-loop preamble.
    if fock_mixing_value != 0.0:
        plog.info(
            f"fock mixing: CRYSTAL FMIXING "
            f"{100.0 * fock_mixing_value:.1f}% "
            "(previous Fock matrix weight)"
        )
    F_k_prev_mixed: Optional[List[np.ndarray]] = None

    if use_mom:
        plog.info("MOM: ON")
    C_prev_occ_per_k: Optional[List[np.ndarray]] = None
    if use_oda and use_diis:
        raise ValueError("use_oda and use_diis are mutually exclusive")
    if use_oda:
        plog.info(f"ODA: ON (trust l_max = {oda_trust_lambda_max})")

    # ---- Ewald J-split cache -----------------------------------------
    j_lr_cache = v_ne_lr_cache
    if use_ewald_j_split and not use_exact_ewald_j_for_pure_rks:
        if system.dim != 3:
            raise ValueError("use_ewald_j_split requires dim=3")
        if n_k > 1 and _ir_mapping.size == 0:
            uniform_w = 1.0 / float(n_k)
            if not np.allclose(weights, uniform_w, atol=1e-9):
                raise ValueError(
                    "use_ewald_j_split at multi-k requires uniform "
                    "full-mesh weights or IBZ-reduced mesh with ir_mapping"
                )
        from .bipole_fock_ewald import _build_j_long_range_cache

        assert omega_used is not None
        if _pair_mode:
            # M3: the J^LR blocks must land on the pair-resolved output
            # template; dedicated cache (never the radial V_ne one).
            with plog.stage(
                "j_lr_cache",
                detail=(
                    f"Ewald long-range J kernel on the pair-resolved "
                    f"template (K_max={ewald_k_max} bohr^-1)"
                ),
            ):
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
        cells_r_cart_arr = np.array(
            [np.asarray(c.r_cart, dtype=float) for c in cells],
            dtype=float,
        )
        if j_lr_cache is None:
            # The reciprocal-space Ewald long-range J kernel precompute is
            # one of the heaviest silent pre-SCF stages (can be many
            # minutes on a tight lattice / large K_max). Bracket it so a
            # stalled job shows ``[j_lr_cache]`` as its last live line.
            with plog.stage(
                "j_lr_cache",
                detail=f"Ewald long-range J kernel (K_max={ewald_k_max} bohr^-1)",
            ):
                j_lr_cache = _build_j_long_range_cache(
                    basis,
                    system,
                    cells_r_cart_arr,
                    omega_used,
                    ewald_precision,
                    K_max=ewald_k_max, lattice_opts=lat_opts_2e)

    # Multi-k split: per-(k,k′) q-channel tables for the LR exchange
    # (hybrids only -- pure functionals carry no HF exchange). Shares
    # the J^LR cache's Ewald w and K_max envelope.
    x_lr_cache = None
    if exchange_split_active and n_k > 1 and (alpha_hf > 0.0 or _patom_seed_pending):
        from .bipole_fock_ewald import build_k_exchange_long_range_cache

        assert j_lr_cache is not None and ewald_k_max is not None
        # Per-(k,k') q-channel exchange tables: O(n_k^2) channels, the
        # heaviest multi-k hybrid setup stage -- bracket for localization.
        with plog.stage(
            "k_lr_cache",
            detail=f"{n_k} q-channels, K_max={ewald_k_max} bohr^-1",
        ):
            x_lr_cache = build_k_exchange_long_range_cache(
                basis,
                system,
                j_lr_cache,
                K_max=ewald_k_max,
            )
        plog.info(
            f"  K^LR q-channels: {n_k} distinct q = k-k′ shifts on the "
            f"shared K_max = {ewald_k_max:.2f} bohr⁻¹ envelope "
            f"(a_HF = {alpha_hf:g})"
        )

    # Incremental/differential J_SR(+K_SR) accumulator (opt-in; see
    # ``resolve_incremental_jk``). One accumulator on the total density:
    # hybrids build J_SR+K_SR (build_jk), pure functionals build J_SR only
    # (build_fock, K=None in the shim).
    incremental_jk = resolve_incremental_jk(
        use_incremental_fock=use_incremental_fock,
        exchange_split_active=exchange_split_active,
        use_oda=use_oda,
        plog=plog,
    )

    def _build_fock_for_density(
        density: LatticeMatrixSet,
        *,
        coeffs_for_rho: Optional[Sequence[np.ndarray]],
        use_incremental: bool = True,
        reseed_incremental: bool = False,
        patom_hf_like: bool = False,
    ) -> _PBCBipoleRKSFockBuild:
        """Build F^2e(g), add V_xc, and assemble F(k)."""
        fb = build_bipole_restricted_fock(
            _fock_ctx,
            density,
            coeffs_for_rho=coeffs_for_rho,
            alpha_hf=1.0 if patom_hf_like else alpha_hf,
            exchange_assembly=None if patom_hf_like else exx,
            use_incremental=use_incremental,
            reseed_incremental=reseed_incremental,
        )
        f2e_real = fb.f2e_real
        K_corr_per_k = fb.k_corr_per_k

        # ---- Build XC potential on the DFT grid ----------------------
        xc_result = None
        if not patom_hf_like:
            D_xc_set = _density_set_gamma_or_lattice(S_lat, density)
            xc_result = build_xc_periodic(
                basis,
                system,
                grid,
                func,
                D_xc_set,
                lat_opts_2e,
                xc_density_domain,
            )

        # ---- Assemble per-k Fock matrices ----------------------------
        f_k_list: List[np.ndarray] = []
        F2e_k_all = _bloch_sum_blocks_multi_k(
            f2e_real.blocks,
            f2e_real.cells,
            k_points,
        )
        for k_idx, k in enumerate(k_points):
            k_arr = np.asarray(k, dtype=float)
            Vxc_k = (0.0 if xc_result is None else
                     np.asarray(bloch_sum(xc_result.V_xc, k_arr)))
            F_k = (
                F2e_k_all[k_idx]
                + Vxc_k
                + np.asarray(Hcore_k_list[k_idx], dtype=complex)
            )
            if K_corr_per_k is not None:
                # Ewald exchange split: a_HF.(K_LR + G=0) per k (a
                # single Γ entry at n_k = 1).
                F_k = F_k - 0.5 * K_corr_per_k[k_idx]
            F_k = 0.5 * (F_k + F_k.conj().T)
            f_k_list.append(F_k)

        return _PBCBipoleRKSFockBuild(
            f2e_real=f2e_real,
            f_k_list=f_k_list,
            e_j_short_range=fb.e_j_short_range,
            e_j_long_range=fb.e_j_long_range,
            e_exchange=fb.e_exchange,
            e_j_multipole=fb.e_j_multipole,
            e_2e_k_correction=fb.e_2e_k_correction,
            e_exchange_finite_size=fb.e_exchange_finite_size,
            e_xc=None if xc_result is None else float(xc_result.e_xc),
            screened_exchange_execution=fb.screened_exchange_execution,
            output_cell_farming_execution=(
                fb.output_cell_farming_execution
            ),
        )

    # ---- Retired multipole far-field config -----------------------------
    from .bipole_fock_multipole import (  # noqa: E402
        BipoleMultipoleConfig,
        resolve_multipole_config,
    )

    # Explicit requests fail before setup. Keep a disabled config here for
    # the shared Fock context until the research-only implementation is
    # removed from that internal API.
    _ff_enable: Optional[bool] = False

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

    # ---- Disabled quartet-level far-field infrastructure ----------------
    from .bipole_far_field_infrastructure import (
        build_bipolar_far_field_infrastructure,
    )

    _spherical_buffer, _penetration_dispatch_j, _penetration_dispatch_k, _tensor_cache, _fock_kernel, _sym_recon = (
        build_bipolar_far_field_infrastructure(
            system,
            basis,
            lat_opts_2e,
            use_multipole_far_field=(
                _ff_enable and exchange_split_active
            ),
            exchange_split_active=exchange_split_active,
            multipole_l_max=multipole_l_max,
            ewald_omega=omega_used if use_ewald_j_split else 0.0,
            plog=plog,
        )
    )

    # ---- Shared restricted Fock-build context (M2 unification) ----------
    # The RKS wrapper adds V_xc after the shared two-electron builder returns.
    # _build_fock_for_density (above) references this via late binding; it is
    # only called from the SCF loop below, after this point.
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
        incremental_jk=incremental_jk,
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
        n_occ=n_occ,
        plog=plog,
        sr_image_extent=_sr_image_extent,
        exact_zone_bohr=_exact_zone,
        exact_j_for_pure_rks=use_exact_ewald_j_for_pure_rks,
        exact_j_ke_cutoff=exact_j_ke_cutoff,
        exact_j_chunk_size=exact_j_chunk_size,
        output_cell_farming_task_kind=(
            output_cell_farming_task_kind if farm_output_cells else None
        ),
        output_cell_farming_strategy=output_cell_farming_strategy,
    )

    def _split_k_density_list(density: LatticeMatrixSet) -> List[np.ndarray]:
        return split_k_density_list(_fock_ctx, density)

    if _patom_seed_pending:
        patom_fock = _build_fock_for_density(
            D_real, coeffs_for_rho=None, use_incremental=False,
            patom_hf_like=True,
        )
        C_per_k, eps_per_k = [], []
        for F_k, X_k in zip(patom_fock.f_k_list, X_k_list):
            C_k, eps_k = _diag_in_orth_basis(F_k, X_k)
            C_per_k.append(C_k.astype(complex))
            eps_per_k.append(eps_k)
        occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)
        if _fractional_density_needed(occ_per_k):
            D_real = real_space_density_from_kpoints_fractional(
                C_per_k, occ_per_k, kmesh, cells_density,
            )
        else:
            D_real = real_space_density_from_kpoints(
                C_per_k, [n_occ] * n_k, kmesh, cells_density,
            )
        if not exchange_split_active and not needs_pair_difference_xc:
            _zero_cross_cell_density(D_real, basis.nbasis, n_k)
        _constructed_density_k = [
            (C * occ) @ C.conj().T for C, occ in zip(C_per_k, occ_per_k)
        ]
        D_real_prev = None
        initial_density_is_local = False
        density_from_c_per_k = True

    plog.banner("SCF (PBC BIPOLE RKS, direct-space)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    scf_trace: List[SCFIteration] = []
    energy_components: List[PBCBipoleEnergyComponents] = []
    def _exact_free_energy(density, fb, e_dft_plus_u_value, entropy_value):
        """Free energy of ``density`` on the exact build ``fb`` (#116).

        Assembled exactly as the post-loop finalisation assembles the
        returned value (``fb.e_xc`` is the XC energy of the quadrature whose
        V_xc entered ``fb``), so the in-loop confirmation and the terminal
        refresh judge the same number.
        """
        e_kin = _lattice_contract(density, T_lat, operator_name="T")
        e_ne = _lattice_contract(density, V_lat, operator_name="V_ne")
        e_2e = (
            0.5 * _lattice_contract(density, fb.f2e_real, operator_name="F2e")
            + fb.e_2e_k_correction
        )
        assert fb.e_xc is not None
        e_tot = (
            float(e_kin + e_ne + e_2e + float(fb.e_xc))
            + e_nuc
            + float(e_dft_plus_u_value)
        )
        if system.dim == 3 and not exchange_split_active:
            e_tot += compute_ext_el_spheropole(density, basis, system, lat_opts)
        return float(e_tot) - smearing_T * float(entropy_value)

    E_prev = 0.0
    F_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    E_elec = 0.0
    e_xc = 0.0
    e_dft_plus_u = 0.0
    converged = False
    iter_idx = 0
    #: Exact rebuild from the convergence confirmation (IID 514); reused by
    #: the post-loop finalisation so healthy rows pay nothing extra.
    _confirm_fb: Optional[_PBCBipoleRKSFockBuild] = None
    _confirm_e_dft_plus_u: Optional[float] = None

    # ---- DFT+U setup (closed-shell BIPOLE RKS +U) ------------------------
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
                    f"run_pbc_bipole_rks: HubbardSite{key} has no AOs "
                    f"in the basis. Available channels: "
                    f"{sorted(ao_groups_map.keys())}"
                )
            dft_plus_u_sites_cxx.append(
                _HubbardSiteCxx(site.atom_index, site.l, site.U_eff_hartree)
            )
            dft_plus_u_ao_groups.append(ao_groups_map[key])

    def _apply_dft_plus_u(
        density: LatticeMatrixSet,
        fock_k_list: List[np.ndarray],
        density_k=None,
    ) -> float:
        """Apply +U energy/potential to the density used for this Fock."""
        if not dft_plus_u_sites_cxx:
            return 0.0
        from ._vibeqc_core import _compute_dft_plus_u_multi_k_per_spin_cxx

        if density_k is not None:
            P_k_all = density_k
        elif exchange_split_active:
            P_k_all = _split_k_density_list(density)
        else:
            P_k_all = _bloch_sum_blocks_multi_k(
                density.blocks, density.cells, k_points
            )
        P_sigma_k_for_u = [
            0.25 * (P_k + P_k.conj().T) for P_k in P_k_all
        ]
        e_sigma, v_ao = _compute_dft_plus_u_multi_k_per_spin_cxx(
            dft_plus_u_sites_cxx,
            dft_plus_u_ao_groups,
            S_k_list,
            P_sigma_k_for_u,
            list(weights),
        )
        v_ao = np.asarray(v_ao, dtype=complex)
        for k_idx, S_k in enumerate(S_k_list):
            fock = fock_k_list[k_idx] + S_k @ v_ao @ S_k
            fock_k_list[k_idx] = 0.5 * (fock + fock.conj().T)
        return 2.0 * float(e_sigma)

    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter

        D_used = D_real
        if iter_idx > 1 and damping > 0.0 and not diis_active:
            D_used = _damp_lattice_matrix(D_real, D_real_prev, damping)

        d_used_is_damped = iter_idx > 1 and damping > 0.0 and not diis_active
        d_used_from_coeffs = (
            density_from_c_per_k
            and not (initial_density_is_local and iter_idx == 1)
            and not d_used_is_damped
        )
        fock_build = _build_fock_for_density(
            D_used,
            coeffs_for_rho=(C_per_k if d_used_from_coeffs else None),
        )
        F2e_real = fock_build.f2e_real
        F_k_list = fock_build.f_k_list

        # ---- DFT+U: per-spin per-k Fock contribution (closed-shell).
        # Same pattern as run_pbc_bipole_rhf -- closed-shell uses
        # P_s = P_total/2 and E_U_total = 2 x E_s.
        constructed_density_k = _constructed_density_k if iter_idx == 1 else None
        e_dft_plus_u = _apply_dft_plus_u(
            D_used, F_k_list, density_k=constructed_density_k
        )

        # ---- XC energy -----------------------------------------------
        # From the SAME quadrature that produced this iteration's V_xc
        # (inside _build_fock_for_density on D_used) -- the loop used to
        # run a second, redundant build_xc_periodic here every iteration
        # just to extract e_xc.
        assert fock_build.e_xc is not None
        e_xc = float(fock_build.e_xc)

        # ---- Energy --------------------------------------------------
        E_kin = _lattice_contract(D_used, T_lat, operator_name="T")
        E_ne = _lattice_contract(D_used, V_lat, operator_name="V_ne")
        E_2e = (
            0.5
            * _lattice_contract(
                D_used,
                F2e_real,
                operator_name="F2e",
            )
            # k-space exchange correction (Ewald exchange split).
            + fock_build.e_2e_k_correction
        )
        E_elec = E_kin + E_ne + E_2e + e_xc

        grad_norm_sum = 0.0
        error_k_list: List[np.ndarray] = []
        D_k_list: List[np.ndarray] = []
        D_k_split_guess: Optional[List[np.ndarray]] = None
        if exchange_split_active and initial_density_is_local and iter_idx == 1:
            # BvK representative for local AND Bloch warm-start storage
            # (home block at Γ; exact torus fold at multi-k -- S_g over
            # the unprojected fold would overcount).
            D_k_split_guess = _split_k_density_list(D_used)
        D_k_guess_fold: Optional[List[np.ndarray]] = None
        if (
            initial_density_is_local
            and iter_idx == 1
            and D_k_split_guess is None
            and _constructed_density_k is None
        ):
            D_k_guess_fold = _bloch_sum_blocks_multi_k(
                D_used.blocks, D_used.cells, k_points
            )
        for idx in range(n_k):
            if initial_density_is_local and iter_idx == 1:
                if _constructed_density_k is not None:
                    D_k = _constructed_density_k[idx]
                elif D_k_split_guess is not None:
                    D_k = D_k_split_guess[idx]
                else:
                    assert D_k_guess_fold is not None
                    D_k = D_k_guess_fold[idx]
                D_k = 0.5 * (D_k + D_k.conj().T)
            else:
                C_k = C_per_k[idx]
                if _occupations_are_per_k_integer_aufbau(occ_per_k, n_occ):
                    C_occ = C_k[:, :n_occ]
                    D_k = 2.0 * (C_occ @ C_occ.conj().T)
                else:
                    # Fractional-occupation density:  D(k) = S_i n_i C_i C_i+
                    occ = np.asarray(occ_per_k[idx], dtype=float)
                    C_full = np.asarray(C_k, dtype=np.complex128)
                    D_k = (C_full * occ[None, :]) @ C_full.conj().T
            D_k_list.append(D_k)
            F_k = F_k_list[idx]
            w = float(weights[idx])
            S_k = S_k_list[idx]
            FDS = F_k @ D_k @ S_k
            grad = FDS - FDS.conj().T
            error_k_list.append(grad)
            grad_norm_sum += w * float(np.linalg.norm(grad))

        E_total = float(E_elec) + e_nuc + e_dft_plus_u

        # EXT EL-SPHEROPOLE -- a 3D-Ewald-gauge correction, identically
        # zero in the direct (non-Ewald) gauge used for dim<3, and
        # OMITTED under the corrected gauge (double-count -- see the
        # RHF driver note).
        if system.dim == 3 and not exchange_split_active:
            E_sphero = compute_ext_el_spheropole(D_used, basis, system, lat_opts)
            E_total += E_sphero
        else:
            E_sphero = None

        free_energy = E_total - smearing_T * entropy
        dE = free_energy - E_prev if iter_idx > 1 else 0.0

        check_scf_divergence(
            "run_pbc_bipole_rks",
            iter_idx,
            free_energy,
            grad_norm_sum,
            dE,
        )
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(free_energy),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm_sum),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(free_energy),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm_sum),
            diis=(accel.subspace_size if accel is not None else 0),
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

        physical_fock_stationary = (
            grad_norm_sum < float(opts.conv_tol_grad)
        )
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and physical_fock_stationary
        )

        # SCF-accelerator extrapolation. Full
        # {DIIS, KDIIS, EDIIS, EDIIS_DIIS, ADIIS} family wired here;
        # bridged modes route through the M2e stacked-real-block bridge
        # (see ``per_k_to_stacked_real_blocks`` in
        # ``periodic_scf_accelerators.py``).
        if accel is not None:
            if constructed_density_k is not None:
                density_k_list = constructed_density_k
            elif exchange_split_active:
                # Unprojected fold: BvK representative per k (home
                # block at Γ; exact torus fold at multi-k).
                density_k_list = _split_k_density_list(D_used)
            else:
                density_k_list = _bloch_sum_blocks_multi_k(
                    D_used.blocks, D_used.cells, k_points
                )
            F_ex_list = accel.extrapolate_rhf(
                F_k_list,
                error_k_list=error_k_list,
                density_k_list=density_k_list,
                energy=free_energy,
                mo_coeffs_k_list=C_per_k,
                n_occ=n_occ,
                weights=list(weights),
                cells=cells,
                kpoints=list(k_points),
            )
            # Once the physical Fock F(D) commutes with D, diagonalise that
            # Fock -- not the extrapolated one -- even if the lagging energy
            # difference still needs one confirmation iteration. Reapplying
            # DIIS at a stationary density can kick the SCF away before the
            # energy test catches up; a near-machine-zero error history also
            # makes the Pulay B-matrix singular. Fractional occupations still
            # pass through the density fixed-point check below after this
            # physical-Fock update. See the RHF twin in pbc_bipole.py and
            # tests/test_pbc_bipole_diis_converged_basin.py.
            if diis_active and not physical_fock_stationary:
                F_k_list = F_ex_list

        # --- FMIXING (CRYSTAL-style, after DIIS, before level-shift)
        # Skipped once the physical Fock is stationary (see DIIS note above).
        if fock_mixing_value != 0.0 and not physical_fock_stationary:
            if F_k_prev_mixed is not None:
                F_mixed_list: List[np.ndarray] = []
                for idx in range(n_k):
                    F_mixed = (1.0 - fock_mixing_value) * F_k_list[
                        idx
                    ] + fock_mixing_value * F_k_prev_mixed[idx]
                    F_mixed = 0.5 * (F_mixed + F_mixed.conj().T)
                    F_mixed_list.append(F_mixed)
                F_k_list = F_mixed_list
            F_k_prev_mixed = [np.asarray(F, dtype=complex).copy() for F in F_k_list]

        # Level shift
        if level_shift_schedule is not None:
            level_shift_b = level_shift_schedule.at(iter_idx)
        else:
            level_shift_b = level_shift_static
        if level_shift_b != 0.0 and not physical_fock_stationary:
            F_for_diag: List[np.ndarray] = []
            for idx in range(n_k):
                D_k = D_k_list[idx]
                S_k = S_k_list[idx]
                F_shift = (
                    F_k_list[idx]
                    + level_shift_b * S_k
                    - (level_shift_b / 2.0) * (S_k @ D_k @ S_k)
                )
                F_shift = 0.5 * (F_shift + F_shift.conj().T)
                F_for_diag.append(F_shift)
        else:
            F_for_diag = F_k_list

        # Diagonalise
        new_C_per_k = []
        new_eps_per_k = []
        for idx in range(n_k):
            C_k, eps_k = _diag_in_orth_basis(
                F_for_diag[idx],
                X_k_list[idx],
            )
            new_C_per_k.append(C_k)
            new_eps_per_k.append(eps_k)

        # MOM
        if use_mom and C_prev_occ_per_k is not None:
            for idx in range(n_k):
                C_k = new_C_per_k[idx]
                eps_k = new_eps_per_k[idx]
                S_k = S_k_list[idx]
                sel = _mom_select(
                    C_k,
                    S_k,
                    C_prev_occ_per_k[idx],
                    n_occ,
                    eps_new=eps_k,
                )
                n_kept_idx = C_k.shape[1]
                virt_mask = np.ones(n_kept_idx, dtype=bool)
                virt_mask[sel] = False
                virt_sel = np.where(virt_mask)[0]
                virt_sel = virt_sel[np.argsort(np.real(eps_k[virt_sel]))]
                order = np.concatenate([sel, virt_sel])
                new_C_per_k[idx] = C_k[:, order]
                new_eps_per_k[idx] = eps_k[order]

        C_per_k = new_C_per_k
        eps_per_k = new_eps_per_k

        occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)

        _fractional_occupations_here = _fractional_density_needed(occ_per_k)
        if _fractional_occupations_here:
            D_real_new = real_space_density_from_kpoints_fractional(
                C_per_k,
                occ_per_k,
                kmesh,
                cells_density,
            )
        else:
            D_real_new = real_space_density_from_kpoints(
                C_per_k,
                [n_occ] * n_k,
                kmesh,
                cells_density,
            )
        # Corrected gauge: the full Bloch fold IS the density the
        # builders need (see run_pbc_bipole_rhf); legacy keeps the
        # Γ-locality projection.
        if not exchange_split_active and not needs_pair_difference_xc:
            _zero_cross_cell_density(D_real_new, basis.nbasis, n_k)

        # Require the newly diagonalised density to be the same fixed point as
        # D_used before accepting convergence. This also catches the
        # occupation-only motion that a finite-temperature commutator cannot
        # see.
        if converged:
            density_fixed_point_residual = max(
                (
                    float(
                        np.max(
                            np.abs(
                                np.asarray(new_block)
                                - np.asarray(old_block)
                            )
                        )
                    )
                    for new_block, old_block in zip(
                        D_real_new.blocks, D_used.blocks
                    )
                ),
                default=0.0,
            )
            converged = density_fixed_point_residual < float(
                opts.conv_tol_grad
            )

        # ODA
        if use_oda:
            fock_naive = _build_fock_for_density(
                D_real_new,
                coeffs_for_rho=C_per_k,
                use_incremental=False,  # off the per-iter ΔD chain
            )
            oda_step = _compute_oda_lambda(
                D_used,
                D_real_new,
                F_k_list,
                fock_naive.f_k_list,
                [np.asarray(k) for k in k_points],
                weights,
                trust_lambda_max=oda_trust_lambda_max,
            )
            _oda_mix(D_used, D_real_new, oda_step.lam)
            D_real_prev = D_real
            D_real = D_used
            density_from_c_per_k = oda_step.lam == 1.0
            plog.info(f"  ODA: l = {oda_step.lam:.4f}")
        else:
            D_real_prev = D_used
            D_real = D_real_new
            density_from_c_per_k = True

        if use_mom:
            C_prev_occ_per_k = [
                np.asarray(C_per_k[idx][:, :n_occ]).copy() for idx in range(n_k)
            ]

        if damper is not None:
            damper.update(free_energy)
        E_prev = free_energy
        if converged:
            # IID 514: confirm the provisional exit on the EXACT operator at
            # the density just committed. The loop's gate measured F(D_prev)
            # on D_prev; at a marginal fixed point the true commutator
            # F(D_new) on D_new can still sit above the gate by one
            # iteration's step, and the post-loop terminal check would then
            # revoke the convergence with no way back. Confirming here lets
            # the loop run the extra iteration(s) it needs; the rebuild is
            # reused by the post-loop finalisation, so healthy rows pay
            # nothing extra.
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
                    D_real,
                    coeffs_for_rho=(C_per_k if density_from_c_per_k else None),
                    use_incremental=False,
                    # #116: re-sync the incremental chain to this exact build,
                    # so a failed confirmation continues on the operator it is
                    # judged by (a no-op when the loop exits here).
                    reseed_incremental=True,
                )
            # Dudarev et al., Phys. Rev. B 57, 1505 (1998), Eq. (6):
            # the exact one-electron operator includes U_eff(1/2 - n).
            _confirm_e_dft_plus_u = _apply_dft_plus_u(
                D_real,
                _confirm_fb.f_k_list,
            )
            _confirm_grad = restricted_bipole_commutator_norm(
                _confirm_fb.f_k_list,
                (
                    _split_k_density_list(D_real)
                    if exchange_split_active
                    else _bloch_sum_blocks_multi_k(
                        D_real.blocks, D_real.cells, k_points
                    )
                ),
                S_k_list,
                weights,
            )
            # #116: judge the exact operator's free energy at the committed
            # density here too -- the delta the post-loop refresh writes into
            # the terminal trace row -- so the loop cannot exit on a state
            # the terminal check would refuse.
            _confirm_ok, _ = bipole_terminal_check(
                plog,
                phase="in_loop",
                iter_idx=iter_idx,
                loop_objective=float(scf_trace[-1].energy),
                exact_objective=_exact_free_energy(
                    D_real, _confirm_fb, _confirm_e_dft_plus_u, entropy
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
    # final diagonalisation commits a new density even on max-iteration
    # exhaustion, so this rebuild is required for both success and failure.
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
                D_real,
                coeffs_for_rho=(C_per_k if density_from_c_per_k else None),
                use_incremental=False,
            )
        )
    F_k_list = _fb.f_k_list
    if _confirm_fb is not None:
        assert _confirm_e_dft_plus_u is not None
        e_dft_plus_u = _confirm_e_dft_plus_u
    else:
        e_dft_plus_u = _apply_dft_plus_u(D_real, F_k_list)

    # Canonicalize the exact physical operator at the authoritative returned
    # density.  Loop-local eigenpairs may belong to the preceding density or
    # to a DIIS/Fock-mixed operator, especially on a max-iteration return.
    C_per_k = []
    eps_per_k = []
    for F_k, X_k in zip(F_k_list, X_k_list):
        C_k, eps_k = _diag_in_orth_basis(F_k, X_k)
        C_per_k.append(np.asarray(C_k, dtype=complex))
        eps_per_k.append(np.asarray(eps_k, dtype=float))
    if use_mom and C_prev_occ_per_k is not None:
        for idx in range(n_k):
            C_k = C_per_k[idx]
            eps_k = eps_per_k[idx]
            sel = _mom_select(
                C_k,
                S_k_list[idx],
                C_prev_occ_per_k[idx],
                n_occ,
                eps_new=eps_k,
            )
            virt_mask = np.ones(C_k.shape[1], dtype=bool)
            virt_mask[sel] = False
            virt_sel = np.where(virt_mask)[0]
            virt_sel = virt_sel[np.argsort(np.real(eps_k[virt_sel]))]
            order = np.concatenate([sel, virt_sel])
            C_per_k[idx] = C_k[:, order]
            eps_per_k[idx] = eps_k[order]
    occ_per_k, fermi_level, entropy = _occupations_from_eps(eps_per_k)

    screened_exchange_execution = _fb.screened_exchange_execution
    # The exact final-density Fock already carries this quadrature energy.
    assert _fb.e_xc is not None
    final_e_xc = float(_fb.e_xc)
    E_kin_f = _lattice_contract(D_real, T_lat, operator_name="T")
    E_ne_f = _lattice_contract(D_real, V_lat, operator_name="V_ne")
    E_2e_f = (
        0.5 * _lattice_contract(D_real, _fb.f2e_real, operator_name="F2e")
        + _fb.e_2e_k_correction
    )
    E_elec = E_kin_f + E_ne_f + E_2e_f + final_e_xc
    final_e_2e = float(E_2e_f)
    final_e_exchange = float(_fb.e_exchange or 0.0)
    E_total = float(E_elec) + e_nuc + e_dft_plus_u
    # Fresh E_total doesn't include spheropole -- add it (3D only; the
    # term is zero in the direct gauge used for dim<3, and omitted
    # under the corrected gauge).
    if system.dim == 3 and not exchange_split_active:
        E_sphero_final = compute_ext_el_spheropole(D_real, basis, system, lat_opts)
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

    # ---- Ionic-Γ basin-health diagnostic (smearing straddle) -----------
    # Evaluate the diagnostic from the same final canonical eigenpairs and
    # occupations that are returned to callers.
    basin_warning = smearing_basin_warning(
        smearing_T,
        [(eps_per_k, occ_per_k, n_occ, 2.0)],
        entropy,
        "run_pbc_bipole_rks",
    )
    if basin_warning is not None:
        plog.info("  WARNING: " + basin_warning)
        warnings.warn(basin_warning, UserWarning, stacklevel=2)

    free_energy_final = E_total - smearing_T * entropy
    final_grad_norm = restricted_bipole_commutator_norm(
        F_k_list,
        (
            _split_k_density_list(D_real)
            if exchange_split_active
            else _bloch_sum_blocks_multi_k(
                D_real.blocks, D_real.cells, k_points
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

    return PBCBipoleRKSResult(
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density is not None or initial_density_k is not None),
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
        mo_energies=eps_per_k,
        mo_coeffs=C_per_k,
        fock=F_k_list,
        overlap=S_k_list,
        hcore=Hcore_k_list,
        density=D_real,
        scf_trace=scf_trace,
        ewald_alpha_bohr_inv=omega_used,
        sr_image_extent_bohr=_sr_image_extent,
        pair_resolved_fock_domain=bool(_fock_sym_map is not None),
        e_dft_plus_u=float(e_dft_plus_u),
        energy_components=energy_components,
        functional=str(opts.functional),
        smearing_temperature=smearing_T,
        fermi_level=float(fermi_level),
        entropy=float(entropy),
        free_energy=float(free_energy_final),
        occupations=[np.asarray(o, dtype=float) for o in occ_per_k],
        exchange_ewald_split=bool(exchange_split_active),
        exchange_exxdiv=(exchange_exxdiv if exchange_split_active else None),
        overlap_fold_drift=overlap_fold_drift,
        exact_zone_bohr=_exact_zone,
        output_cell_farming_execution=(
            _fb.output_cell_farming_execution
        ),
        screened_exchange_execution=screened_exchange_execution,
        basin_warning=basin_warning,
        kpoints_cart=np.asarray(k_points, dtype=float).reshape(-1, 3),
        kpoint_weights=np.asarray(weights, dtype=float).reshape(-1),
        fock_mixing=fock_mixing_value,
    )
