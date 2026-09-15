"""BIPOLE-style periodic UHF driver in CRYSTAL's electrostatic gauge.

This is the open-shell counterpart of :mod:`vibeqc.pbc_bipole`. It keeps
the same CRYSTAL-inspired composition:

* ``V_ne`` and ``E_nn`` share one explicit 3D Ewald state.
* The default 3D two-electron build uses ``J_SR(a) + J_LR(a)`` for the
  Hartree operator with that same alpha, plus full-range per-spin
  exchange from the direct real-space builder.
* Energies are evaluated by real-space lattice contractions so the
  first local SAD/Hcore cycle has the same accounting convention as the
  RHF BIPOLE driver.

The exact Ewald-J route is the production spin-unrestricted energy path.
Pisani-Dovesi-Roetti (1988), Ch. II.4c, is the source for the periodic
quartet expansion. The dormant classifier and add-back prototype do not
preserve the exact route's three-translation Fock domain, so this driver
rejects ``use_multipole_far_field=True``.
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    apply_level_shift_k,
    BasisSet,
    BlochKMesh,
    EwaldOptions,
    GridOptions,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    LevelShiftDensity,
    PeriodicRHFOptions,
    PeriodicSystem,
    SCFIteration,
    bloch_sum,
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
    _copy_lattice_with_blocks,
    _crystal_ewald_options,
    _emit_bipole_semantic_diagnostic,
    _expand_ibz_kmesh_for_ewald_j,
    _lattice_contract,
    _lattice_contract_blocks,
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
    unrestricted_bipole_commutator_norm,
    bipole_terminal_check,
    refresh_bipole_terminal_trace,
    resolve_fock_mixing,
    resolve_bipole_sr_image_extent,
    resolve_bipole_fock_symmetry,
    warn_bipole_charged_cell,
    warn_bipole_legacy_multik_gauge,
    home_cell_block,
    smearing_basin_warning,
)
from .pbc_bipole_fock import (
    _sr_density_cells,
    BipoleFockContext,
    BipoleUnrestrictedFockBuild,
    build_bipole_unrestricted_fock,
)
from .periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _damp_lattice_matrix,
    _diag_in_orth_basis,
)
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicUHFAccelerator,
)
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence

__all__ = [
    "PBCBipoleUHFResult",
    "run_pbc_bipole_uhf",
]


@dataclass
class PBCBipoleUHFResult:
    """Result of :func:`run_pbc_bipole_uhf`."""

    energy: float
    e_electronic: float
    e_nuclear: float
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
    # Absolute internal ket-image radius used by the erfc SR build. Analytic
    # gradients currently reject padded-domain results rather than silently
    # differentiating the historical traversal.
    sr_image_extent_bohr: Optional[float] = None
    # True when the Fock build used the group-invariant pair-resolved domain.
    pair_resolved_fock_domain: bool = False
    # Dudarev DFT+U contribution per unit cell (Hartree). 0 unless
    # the caller passed ``dft_plus_u=[HubbardSite(...)]``.
    e_dft_plus_u: float = 0.0
    energy_components: List[PBCBipoleEnergyComponents] = field(
        default_factory=list,
    )
    # Gauge provenance (option (b)): True when the run used the
    # corrected gauge (full-Bloch density, no spheropole, per-spin
    # Ewald exchange split).
    exchange_ewald_split: bool = False
    exchange_exxdiv: Optional[str] = None
    # Non-None when finite-T smearing straddled the HOMO-LUMO gap and may
    # have selected a near-metallic basin (ionic-Gamma basin trap).
    basin_warning: Optional[str] = None
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
    # Effective previous-Fock weight used by the SCF iteration.
    fock_mixing: float = 0.0
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


def run_pbc_bipole_uhf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    options: Optional[PeriodicRHFOptions] = None,
    *,
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
    use_exchange_ewald_split: Optional[bool] = None,
    exchange_exxdiv: str = "ewald",
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
) -> PBCBipoleUHFResult:
    """Multi-k open-shell UHF via the CRYSTAL-gauge BIPOLE scaffold.

    ``fock_mixing`` overrides ``options.fock_mixing`` when supplied and is
    forwarded through both phases of a spin schedule without rewriting the
    caller's options field.

    ``sr_image_precision`` / ``sr_image_extent_bohr`` and the automatic
    ``use_fock_symmetry_reduce=None`` policy match
    :func:`run_pbc_bipole_rhf`. Spin-schedule recursion forwards all three
    controls unchanged.

    ``bz_integration="gilat"`` applies the T=0 Gilat-Raubenheimer microcell
    integral independently to the fixed alpha and beta populations on full
    or symmetry-reduced Monkhorst-Pack meshes. It has zero entropy and cannot
    be combined with a positive ``options.smearing_temperature``.

    ``dft_plus_u``: optional list of :class:`HubbardSite`. When set,
    the Dudarev +U term is added per-spin per-k after the standard
    BIPOLE Fock build:

    * ``n_s^A_l = S_k w_k Re[(S(k) P_s(k) S(k))_{(A,l)}]`` per spin.
    * ``V_AO_s = U_eff (1/2 - n_s)`` (k-independent).
    * Per-k Fock contribution: ``F_s(k) += S(k) V_AO_s S(k)``.
    * Energy: ``E_U_total = S_s (U_eff/2)(tr n_s - tr n_s^2)`` per
      spin sum, reported via ``result.e_dft_plus_u``.
    """

    reject_bipole_quartet_far_field(
        use_multipole_far_field,
        driver="run_pbc_bipole_uhf",
    )
    reject_bipole_lone_non_gamma_kpoint(
        kmesh,
        driver="run_pbc_bipole_uhf",
    )

    opts = options if options is not None else PeriodicRHFOptions()
    reject_bipole_ecp_options(
        opts,
        driver="run_pbc_bipole_uhf",
        basis=basis,
        system=system,
    )
    reject_bipole_solver_options(opts, driver="run_pbc_bipole_uhf")
    if int(opts.max_iter) < 1:
        raise ValueError("run_pbc_bipole_uhf: max_iter must be at least 1")
    if use_oda:
        raise NotImplementedError(
            "run_pbc_bipole_uhf: unrestricted ODA is unavailable because "
            "the current line search does not include the beta-spin energy "
            "direction; use DIIS instead"
        )
    requested_fock_mixing = resolve_fock_mixing(
        opts,
        fock_mixing,
        where="run_pbc_bipole_uhf",
    )
    # SPINLOCK. SPIN_SCHEDULE (two-phase) is delegated before SCF setup;
    # PATTERN_HOLD reuses the per-k MOM machinery below (the same kernel as
    # use_mom), gated to cycles 2..spinlock_iterations.
    from .spinlock_periodic import check_spinlock_support, run_spin_schedule
    from ._vibeqc_core import SpinlockMode
    _initial_guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", None),
        driver="run_pbc_bipole_uhf",
        supported=periodic_guess_capabilities('bipole', 'UHF', dim=getattr(system, "dim", 3), multi_k=(len(kmesh.kpoints) > 1 or np.prod(kmesh.mesh) > 1), transport='lattice'),
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
            "run_pbc_bipole_uhf: explicit Fock symmetry enforcement or "
            "reduction cannot be combined with an electronically broken or "
            "unverified state control ("
            + ", ".join(_broken_state_symmetry_controls)
            + "). Disable Fock symmetry, or remove the state-selective "
            "control and opt in only for a symmetry-covariant density."
        )
    _spin_schedule_requested = (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF)
        == SpinlockMode.SPIN_SCHEDULE
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
                "run_pbc_bipole_uhf: SPIN_SCHEDULE is Gamma-only. Its "
                "phase-2 restart currently carries only the home-cell spin "
                "densities and cannot reconstruct a complete multi-k state. "
                "Use PATTERN_HOLD or a Gamma mesh."
            )
        return run_spin_schedule(
            lambda sysx, o: run_pbc_bipole_uhf(
                sysx, basis, kmesh, o,
                linear_dep_threshold=linear_dep_threshold,
                fock_mixing=requested_fock_mixing,
                canonical_orth_normalize_diag_first=canonical_orth_normalize_diag_first,
                level_shift_schedule=level_shift_schedule, use_mom=use_mom,
                use_oda=use_oda, oda_trust_lambda_max=oda_trust_lambda_max,
                use_incremental_fock=use_incremental_fock,
                use_ewald_j_split=use_ewald_j_split, ewald_omega=ewald_omega,
                ewald_precision=ewald_precision, v_ne_grid_options=v_ne_grid_options,
                use_multipole_far_field=use_multipole_far_field,
                multipole_l_max=multipole_l_max,
                use_exchange_ewald_split=use_exchange_ewald_split,
                exchange_exxdiv=exchange_exxdiv, use_fock_symmetry=use_fock_symmetry,
                use_fock_symmetry_reduce=use_fock_symmetry_reduce,
                sr_image_precision=sr_image_precision,
                sr_image_extent_bohr=sr_image_extent_bohr,
                exact_zone_bohr=exact_zone_bohr,
                progress=progress, verbose=verbose,
                bz_integration=bz_integration, dft_plus_u=dft_plus_u,
            ),
            system, opts,
        )
    check_spinlock_support(
        opts, {SpinlockMode.PATTERN_HOLD, SpinlockMode.SPIN_SCHEDULE},
        "the BIPOLE UHF driver")
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError("run_pbc_bipole_uhf: smearing_temperature must be >= 0")
    if bz_integration is not None:
        bz_integration = str(bz_integration).strip().lower()
    if bz_integration not in (None, "smearing", "gilat"):
        raise ValueError(
            "run_pbc_bipole_uhf: bz_integration must be None, 'smearing', "
            f"or 'gilat'; got {bz_integration!r}"
        )
    use_gilat = bz_integration == "gilat"
    if use_gilat and smearing_T > 0.0:
        raise ValueError(
            "run_pbc_bipole_uhf: bz_integration='gilat' is a T=0 "
            "integrator; do not combine it with smearing_temperature > 0"
        )
    use_fractional_density = (smearing_T > 0.0) or use_gilat
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
            "run_pbc_bipole_uhf: multi-k READ requires explicit complete "
            "real-space init_alpha/init_beta block sets; Gamma-only read "
            "densities cannot reconstruct the per-k state."
        )
    if use_ewald_j_split and n_k > 1 and _ir_mapping.size == 0:
        uniform_w = 1.0 / float(n_k)
        if not np.allclose(weights, uniform_w, atol=1e-9):
            raise ValueError(
                "use_ewald_j_split at multi-k requires uniform full-mesh "
                "weights or an IBZ-reduced Monkhorst-Pack mesh carrying "
                "ir_mapping metadata so the driver can expand it. "
                f"Got non-uniform weights = {weights.tolist()}."
            )

    # ---- Exchange/gauge resolution (option (b) Phase 4b, 2026-06-11) --
    # Mirrors run_pbc_bipole_rhf; per-spin exchange split at 3D Γ:
    #   K_s = K_SR(erfc w; D_s) + K_LR(D_s) + (ξ_M - pi/(Vw^2)).S.D_s.S
    # (PySCF UHF exxdiv convention: vk_s += ξ.S.D_s.S per spin).
    if exchange_exxdiv not in ("ewald", "none"):
        raise ValueError(
            f"run_pbc_bipole_uhf: exchange_exxdiv must be 'ewald' or "
            f"'none'; got {exchange_exxdiv!r}"
        )
    _x_split_auto = use_exchange_ewald_split is None
    exchange_split_active = (
        bool(use_ewald_j_split)
        if _x_split_auto
        else bool(use_exchange_ewald_split)
    )
    if exchange_split_active and not use_ewald_j_split:
        raise ValueError(
            "run_pbc_bipole_uhf: use_exchange_ewald_split=True requires "
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
                    "run_pbc_bipole_uhf: the Ewald exchange split at multi-k "
                    "requires a Monkhorst-Pack BlochKMesh carrying its mesh "
                    f"dimensions (got mesh={_mesh_attr} for {n_k} k-points). "
                    "Build the mesh via monkhorst_pack(...); ad-hoc k-point "
                    "lists are not supported on the corrected gauge."
                )
        else:
            _bvk_mesh = _mesh_attr

    reject_bipole_fractional_legacy_gauge(
        use_ewald_j_split=bool(use_ewald_j_split),
        exchange_split_active=exchange_split_active,
        fractional_occupations=use_gilat,
        driver="run_pbc_bipole_uhf",
    )

    _ff_enable: Optional[bool] = False

    warn_bipole_legacy_multik_gauge(system, exchange_split_active, n_k, plog)
    warn_bipole_charged_cell(system, plog)
    _xi_madelung = 0.0
    if exchange_split_active and exchange_exxdiv == "ewald":
        if n_k > 1:
            from .bipole_fock_ewald import probe_charge_madelung_supercell

            assert _bvk_mesh is not None
            _xi_madelung = probe_charge_madelung_supercell(system, _bvk_mesh)
        else:
            from .bipole_fock_ewald import probe_charge_madelung

            _xi_madelung = probe_charge_madelung(system)

    plog.info(
        f"PBC BIPOLE UHF (CRYSTAL-gauge) / cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    )
    overlap_fold_drift: Optional[float] = None
    if exchange_split_active:
        plog.info(
            f"  Gauge: corrected -- per-spin Ewald exchange split "
            f"(exxdiv={exchange_exxdiv}"
            + (
                f", ξ_M = {_xi_madelung:.6f} Ha"
                if exchange_exxdiv == "ewald"
                else ""
            )
            + "); full Bloch density; spheropole omitted"
        )
    plog.info(f"  n_alpha = {n_alpha}, n_beta = {n_beta}, multiplicity = {mult}")
    plog.info(
        f"  F^2e (J + K) : "
        f"{'EWALD_J_SPLIT' if use_ewald_j_split else lat_opts_2e.coulomb_method.name}"
        f"{' (auto)' if use_ewald_j_split_auto else ''}"
    )
    plog.info(
        f"k-mesh: {n_k} k-point{'s' if n_k != 1 else ''}, "
        f"weights sum = {weights.sum():.4f}"
    )

    ewald_options_1e: Optional[EwaldOptions] = None
    omega_used: Optional[float] = None
    ewald_cell_volume: Optional[float] = None
    ewald_k_max: Optional[float] = None
    if system.dim == 3:
        from .bipole_ext_el_pole import bipole_ewald_reciprocal_cutoff

        V_cell = float(
            abs(
                np.linalg.det(np.asarray(system.lattice, dtype=float)),
            )
        )
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
            erfc_exchange_arm_active=exchange_split_active,
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
        plog.info(
            f"  Ewald state: alpha = {omega_used:.6f} bohr^-1, "
            f"real_cutoff = {lat_opts_1e.nuclear_cutoff_bohr:.2f} bohr, "
            f"K_max = {ewald_k_max:.2f} bohr^-1, "
            f"tol = {float(ewald_precision):.0e}"
        )

    _sr_image_extent = resolve_bipole_sr_image_extent(
        basis,
        system,
        lat_opts,
        lat_opts_2e,
        omega_used,
        use_ewald_j_split=bool(use_ewald_j_split),
        sr_image_precision=sr_image_precision,
        sr_image_extent_bohr=sr_image_extent_bohr,
        plog=plog,
    )

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
            sym_log_style="rich",
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
        per_spin=True,
        exchange_split_active=exchange_split_active,
        # Structural symmetry alone does not prove that an unrestricted
        # electronic state is symmetry-covariant.  In particular ATOMSPIN
        # antiferromagnetic seeds can be projected into a different basin.
        # Keep AUTO off for UHF; explicit reduction remains an opt-in after
        # the broken-state guard above.
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
    # condition 2026-08-06. The wide density list below remains
    # corrected-gauge-only.
    from .pbc_bipole_common import enforce_bipole_fold_support

    overlap_fold_drift = enforce_bipole_fold_support(
        basis,
        system,
        lat_opts_2e,
        method="uhf",
        n_k=n_k,
        k_points=(k_points if n_k > 1 else None),
        exchange_split_active=exchange_split_active,
        plog=plog,
    )

    if exchange_split_active or lat_opts_2e.pair_complete_1e:
        if _pair_mode:
            # M3: pair-resolved density support (see run_pbc_bipole_rhf).
            cells_density = list(_fock_sym_map.density_domain.cells)
            plog.info(
                f"  density cell list: {len(cells_density)} cells "
                f"(pair-resolved support at 2x cutoff; SR-masked privately)"
            )
        else:
            cells_density = list(
                _sr_density_cells(basis, system, lat_opts_2e, _sr_image_extent)
            )
            plog.info(
                f"  density cell list: {len(cells_density)} cells "
                + ("(physical exchange density support)" if lat_opts_2e.pair_complete_1e
                   else "(2x cutoff)")
            )
    else:
        cells_density = cells

    from .linear_dependence import scf_preflight_overlap_check

    S_k_list: List[np.ndarray] = []
    Hcore_k_list: List[np.ndarray] = []
    X_k_list: List[np.ndarray] = []
    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        V_k = np.asarray(bloch_sum(V_lat, k_arr))
        H_k = T_k + V_k
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        scf_preflight_overlap_check(
            S_k,
            plog=plog,
            label=f"S(k={k_idx}, k_cart={k_arr.round(4).tolist()})",
            basis=basis,
        )
        X_k, n_kept = _canonical_orthogonalizer_complex(
            S_k,
            linear_dep_threshold,
            normalize_diag_first=canonical_orth_normalize_diag_first,
        )
        if max(n_alpha, n_beta) > n_kept:
            raise RuntimeError(
                f"run_pbc_bipole_uhf: canonical orth at k={k_idx} "
                f"dropped too many directions (n_alpha={n_alpha}, "
                f"n_beta={n_beta}, n_kept={n_kept})"
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

    if ewald_options_1e is not None:
        e_nuc = float(ewald_nuclear_repulsion(system, ewald_options_1e))
    else:
        e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts_1e))
    plog.info(f"E_nuc per cell ({lat_opts_1e.coulomb_method.name}) = {e_nuc:+.10f} Ha")

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

    def _spin_density(C_per_k_local, n_occ_each):
        return unrestricted_spin_density(
            C_per_k_local,
            n_occ_each,
            n_k=n_k,
            kmesh=kmesh,
            cells_density=cells_density,
            exchange_split_active=exchange_split_active,
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
        if not exchange_split_active:
            _zero_cross_cell_density(D_alpha_real, D_alpha_real.blocks[0].shape[0], n_k)
            _zero_cross_cell_density(D_beta_real, D_beta_real.blocks[0].shape[0], n_k)

    # Caller-supplied warm-start spin densities take precedence over
    # the SAD/Hcore guess engine. Both init_alpha and init_beta must
    # be provided together (or both None). Block ordering matches the
    # canonical ``direct_lattice_cells(kmesh)`` ordering -- same
    # contract as the RHF driver. Used by the NEB driver for
    # within-image density warm-start for periodic NEB.
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
            "run_pbc_bipole_uhf: init_alpha and init_beta must be "
            "provided together (both None or both populated)"
        )
    if init_alpha is not None and init_beta is not None:
        blocks_a = list(init_alpha)
        blocks_b = list(init_beta)
        if len(blocks_a) != len(D_alpha_real.cells):
            raise ValueError(
                f"run_pbc_bipole_uhf: init_alpha has {len(blocks_a)} "
                f"blocks; expected {len(D_alpha_real.cells)}"
            )
        if len(blocks_b) != len(D_beta_real.cells):
            raise ValueError(
                f"run_pbc_bipole_uhf: init_beta has {len(blocks_b)} "
                f"blocks; expected {len(D_beta_real.cells)}"
            )
        from .guess import real_lattice_restart_block, normalize_periodic_lattice_spin_restart
        for g_idx, (ba, bb) in enumerate(zip(blocks_a, blocks_b)):
            D_alpha_real.set_block(g_idx, real_lattice_restart_block(ba, basis.nbasis))
            D_beta_real.set_block(g_idx, real_lattice_restart_block(bb, basis.nbasis))
        plog.info("initial guess: caller-supplied spin densities (warm-start)")
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
            plog.info(f"initial guess: {guess.name} (g=0 spin densities)")
            D_a0, D_b0 = D_guess
            zero_a = np.zeros_like(D_a0, dtype=float)
            zero_b = np.zeros_like(D_b0, dtype=float)
            for g_idx in range(len(D_alpha_real.cells)):
                is_g0 = (
                    np.asarray(D_alpha_real.cells[g_idx].index, dtype=int)
                    == np.array([0, 0, 0])
                ).all()
                D_alpha_real.set_block(g_idx, D_a0 if is_g0 else zero_a)
                D_beta_real.set_block(g_idx, D_b0 if is_g0 else zero_b)
        else:
            plog.info(f"initial guess: {guess.name} (Hcore-diag per k)")
        density_from_c_per_k = not initial_density_is_local
    if init_alpha is not None and init_beta is not None:
        _patom_seed_pending = False
    D_alpha_prev: Optional[LatticeMatrixSet] = None
    D_beta_prev: Optional[LatticeMatrixSet] = None

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(f"run_pbc_bipole_uhf: damping must be in [0,1); got {damping}")
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
        level_shift_schedule,
        LevelShiftSchedule,
    ):
        raise TypeError(
            f"level_shift_schedule must be a LevelShiftSchedule or None; "
            f"got {type(level_shift_schedule).__name__}"
        )
    if level_shift_schedule is not None:
        plog.info(f"level_shift_schedule: {level_shift_schedule.as_list()}")

    # CRYSTAL-style FMIXING (per-spin, per-k).
    fock_mixing_value = requested_fock_mixing
    if fock_mixing_value != 0.0:
        plog.info(
            f"fock mixing: CRYSTAL FMIXING "
            f"{100.0 * fock_mixing_value:.1f}% "
            "(previous Fock matrix weight)"
        )
    F_alpha_prev_mixed: Optional[List[np.ndarray]] = None
    F_beta_prev_mixed: Optional[List[np.ndarray]] = None

    # SPINLOCK PATTERN_HOLD reuses the per-k MOM machinery (same kernel as
    # use_mom), gated to cycles 2..spinlock_iterations, to hold the seeded
    # broken-symmetry occupied set then release -- protecting an ATOMSPIN seed.
    _spinlock_pattern_hold = (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.PATTERN_HOLD
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    )
    _spinlock_iters = int(getattr(opts, "spinlock_iterations", 0))
    if use_mom:
        plog.info("MOM (Maximum Overlap Method): ON")
    elif _spinlock_pattern_hold:
        plog.info(f"SPINLOCK PATTERN_HOLD: MOM-hold for {_spinlock_iters} cycles")
    C_prev_occ_alpha_per_k = None
    C_prev_occ_beta_per_k = None
    if use_oda and use_diis:
        raise ValueError(
            "run_pbc_bipole_uhf: use_oda and use_diis are mutually exclusive"
        )
    if use_oda:
        if not (0.0 < oda_trust_lambda_max <= 1.0):
            raise ValueError(
                f"oda_trust_lambda_max must be in (0, 1]; got {oda_trust_lambda_max}"
            )
        plog.info(
            f"ODA (Optimal Damping): ON (+1 Fock build/iter, "
            f"trust lambda_max = {oda_trust_lambda_max})"
        )

    j_lr_cache = v_ne_lr_cache
    if use_ewald_j_split:
        if system.dim != 3:
            raise ValueError(
                f"use_ewald_j_split requires dim=3 (3D periodic). Got dim={system.dim}."
            )
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
                [np.asarray(c.r_cart, dtype=float) for c in cells],
                dtype=float,
            )
            j_lr_cache = _build_j_long_range_cache(
                basis,
                system,
                cells_r_cart_arr,
                omega_used,
                ewald_precision,
                K_max=ewald_k_max, lattice_opts=lat_opts_2e)
        elif j_lr_cache.ft_per_cell.shape[0] != len(cells):
            raise RuntimeError(
                "prebuilt V_ne/J^LR cache has a different cell count "
                f"({j_lr_cache.ft_per_cell.shape[0]}) from S_lat "
                f"({len(cells)})"
            )
        plog.info(
            f"  J^LR cache: {j_lr_cache.K_vectors.shape[0]} K-vectors, "
            f"{j_lr_cache.ft_per_cell.shape[0]} lattice cells"
        )

    # Multi-k split: per-(k,k′) q-channel tables for the per-spin LR
    # exchange (option (b) Phase 3). Shares the J^LR w/K_max envelope;
    # both spins consume the same density-independent tables.
    x_lr_cache = None
    if exchange_split_active and n_k > 1:
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
            f"shared K_max = {ewald_k_max:.2f} bohr⁻¹ envelope (per spin)"
        )

    def _split_k_density_list(density) -> List[np.ndarray]:
        return unrestricted_split_k_density_list(
            density, n_k=n_k, k_points=k_points, bvk_mesh=_bvk_mesh
        )

    # Per-spin incremental/differential J_SR+K_SR accumulators (opt-in;
    # see run_pbc_bipole_rhf). The corrected gauge builds jk_a + jk_b
    # separately, so each spin gets its own ΔD chain. Corrected gauge +
    # DIIS only.
    incremental_jk_alpha = None
    incremental_jk_beta = None
    if use_incremental_fock:
        if exchange_split_active and not use_oda:
            from .bipole_fock_ewald import IncrementalJK

            incremental_jk_alpha = IncrementalJK()
            incremental_jk_beta = IncrementalJK()
            plog.info(
                "  incremental Fock (per-spin differential J_SR/K_SR via "
                "ΔD density-envelope screening): ON"
            )
        else:
            plog.info(
                "  incremental Fock requested but inactive "
                "(needs the corrected gauge + DIIS, not ODA)"
            )

    # The UHF two-electron Fock assembly is routed through
    # pbc_bipole_fock.build_bipole_unrestricted_fock below.

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

    # ---- Shared unrestricted Fock-build context (M2c unification) --------
    # Bundle the per-run invariants the inline UHF Fock build captured.
    # The wrapper below is intentionally placed after multipole and symmetry
    # resolution, before the PATOM in-field step and SCF loop first call it.
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
        D_alpha: LatticeMatrixSet,
        D_beta: LatticeMatrixSet,
        *,
        coeffs_alpha_for_rho: Optional[Sequence[np.ndarray]],
        coeffs_beta_for_rho: Optional[Sequence[np.ndarray]],
        use_incremental: bool = True,
        reseed_incremental: bool = False,
    ) -> BipoleUnrestrictedFockBuild:
        return build_bipole_unrestricted_fock(
            _fock_ctx,
            D_alpha,
            D_beta,
            hcore_k_list=Hcore_k_list,
            n_alpha=n_alpha,
            n_beta=n_beta,
            coeffs_alpha_for_rho=coeffs_alpha_for_rho,
            coeffs_beta_for_rho=coeffs_beta_for_rho,
            occupations_alpha_per_k=occ_alpha_per_k,
            occupations_beta_per_k=occ_beta_per_k,
            smearing_temperature=smearing_T,
            incremental_jk_alpha=incremental_jk_alpha,
            incremental_jk_beta=incremental_jk_beta,
            use_incremental=use_incremental,
            reseed_incremental=reseed_incremental,
        )

    if _patom_seed_pending:
        plog.info("initial guess: PATOM (SAD + one BIPOLE in-field step)")
        patom_fock = _build_fock_for_density(
            D_alpha_real,
            D_beta_real,
            coeffs_alpha_for_rho=None,
            coeffs_beta_for_rho=None,
            use_incremental=False,
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
            if not exchange_split_active:
                _zero_cross_cell_density(
                    D_alpha_real, D_alpha_real.blocks[0].shape[0], n_k
                )
                _zero_cross_cell_density(
                    D_beta_real, D_beta_real.blocks[0].shape[0], n_k
                )
        else:
            D_alpha_real = _spin_density(C_alpha_per_k, n_alpha)
            D_beta_real = _spin_density(C_beta_per_k, n_beta)
        D_alpha_prev = None
        D_beta_prev = None
        initial_density_is_local = False
        density_from_c_per_k = True

    plog.banner("SCF (PBC BIPOLE UHF, direct-space)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    # ---- DFT+U setup (Increment 4d-bipole) ------------------------------
    # Translate user-facing HubbardSite objects to the C++ types + AO
    # index lists once per SCF call. ao_group_indices is geometry-
    # invariant (depends only on shell layout per atom Z + basis name),
    # so it's safe to pre-compute here.
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
                    f"run_pbc_bipole_uhf: HubbardSite{key} has no AOs "
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
    ) -> float:
        """Apply spin-resolved +U to the density used for this Fock."""
        if not dft_plus_u_sites_cxx:
            return 0.0
        from ._vibeqc_core import _compute_dft_plus_u_multi_k_per_spin_cxx

        if exchange_split_active:
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

    scf_trace: List[SCFIteration] = []
    energy_components: List[PBCBipoleEnergyComponents] = []
    def _exact_free_energy(d_alpha, d_beta, fb, e_dft_plus_u_value, entropy_value):
        """Free energy of the spin densities on the exact build ``fb`` (#116).

        Assembled exactly as the post-loop finalisation assembles the
        returned value, so the in-loop confirmation and the terminal
        refresh judge the same number.
        """
        d_tot = _combine_density_sets(basis, system, lat_opts_2e, d_alpha, d_beta)
        e_kin = _lattice_contract(d_tot, T_lat, operator_name="T")
        e_ne = _lattice_contract(d_tot, V_lat, operator_name="V_ne")
        e_2e = fb.e_2e_k_correction + 0.5 * (
            _lattice_contract(d_alpha, fb.f2e_alpha_real, operator_name="F2e")
            + _lattice_contract(d_beta, fb.f2e_beta_real, operator_name="F2e")
        )
        e_tot = float(e_kin + e_ne + e_2e) + e_nuc + float(e_dft_plus_u_value)
        if system.dim == 3 and not exchange_split_active:
            e_tot += compute_ext_el_spheropole(d_tot, basis, system, lat_opts)
        return float(e_tot) - smearing_T * float(entropy_value)

    E_prev = 0.0
    E_elec = 0.0
    e_dft_plus_u = 0.0
    F_alpha_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    F_beta_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    converged = False
    iter_idx = 0
    #: Exact rebuild from the convergence confirmation (IID 514); reused by
    #: the post-loop finalisation so healthy rows pay nothing extra.
    _confirm_fb: Optional[BipoleUnrestrictedFockBuild] = None
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
        # n_s = S_k w_k Re[(S(k) P_s(k) S(k))_(A,l)]; V_AO_s = U_eff
        # (1/2 - n_s); per-k Fock += S(k) V_AO_s S(k). The per-spin
        # densities P_s(k) come from Bloch-summing the same lattice
        # density that fock_build saw.
        e_dft_plus_u = _apply_dft_plus_u(
            D_alpha_used,
            D_beta_used,
            F_alpha_k_list,
            F_beta_k_list,
        )

        D_total_used = _combine_density_sets(
            basis,
            system,
            lat_opts_2e,
            D_alpha_used,
            D_beta_used,
        )
        E_kin = _lattice_contract(D_total_used, T_lat, operator_name="T")
        E_ne = _lattice_contract(D_total_used, V_lat, operator_name="V_ne")
        E_2e = fock_build.e_2e_k_correction + 0.5 * (
            _lattice_contract(
                D_alpha_used,
                fock_build.f2e_alpha_real,
                operator_name="F2e_alpha",
            )
            + _lattice_contract(
                D_beta_used,
                fock_build.f2e_beta_real,
                operator_name="F2e_beta",
            )
        )
        E_elec = E_kin + E_ne + E_2e

        grad_norm_sum = 0.0
        error_alpha_k_list: List[np.ndarray] = []
        error_beta_k_list: List[np.ndarray] = []
        D_alpha_k_list: List[np.ndarray] = []
        D_beta_k_list: List[np.ndarray] = []
        D_k_split_guess_a: Optional[List[np.ndarray]] = None
        D_k_split_guess_b: Optional[List[np.ndarray]] = None
        if exchange_split_active and initial_density_is_local and iter_idx == 1:
            # BvK representatives for local AND wide warm-start storage
            # (home block at Γ, exact torus fold at multi-k) -- a S_g
            # fold over the full cell list would overcount wide
            # warm-start blocks. SAD-local guesses give identical
            # values either way (only g = 0 contributes).
            D_k_split_guess_a = _split_k_density_list(D_alpha_used)
            D_k_split_guess_b = _split_k_density_list(D_beta_used)
        D_k_guess_fold_a: Optional[List[np.ndarray]] = None
        D_k_guess_fold_b: Optional[List[np.ndarray]] = None
        if (
            initial_density_is_local
            and iter_idx == 1
            and (D_k_split_guess_a is None or D_k_split_guess_b is None)
        ):
            D_k_guess_fold_a = _bloch_sum_blocks_multi_k(
                D_alpha_used.blocks, D_alpha_used.cells, k_points
            )
            D_k_guess_fold_b = _bloch_sum_blocks_multi_k(
                D_beta_used.blocks, D_beta_used.cells, k_points
            )
        for idx in range(n_k):
            if initial_density_is_local and iter_idx == 1:
                if D_k_split_guess_a is not None and D_k_split_guess_b is not None:
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
            FDS_a = F_a_k @ D_a_k @ S_k
            FDS_b = F_b_k @ D_b_k @ S_k
            err_a = FDS_a - FDS_a.conj().T
            err_b = FDS_b - FDS_b.conj().T
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
        # zero in the direct (non-Ewald) gauge used for dim<3, and
        # OMITTED under the corrected gauge (double-count).
        if system.dim == 3 and not exchange_split_active:
            E_sphero = compute_ext_el_spheropole(D_total_used, basis, system, lat_opts)
            E_total += E_sphero
        else:
            E_sphero = None

        free_energy = E_total - smearing_T * entropy
        dE = free_energy - E_prev if iter_idx > 1 else 0.0
        check_scf_divergence(
            "run_pbc_bipole_uhf",
            iter_idx,
            free_energy,
            grad_norm_sum,
            dE,
        )
        diis_sub = accel.subspace_size if accel is not None else 0
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

        # Skipped entirely (not even recorded) while the PATTERN_HOLD window
        # is active; see the hold_active note at the loop head.
        if accel is not None and not hold_active:
            if exchange_split_active:
                # Unprojected Bloch fold: S_g over the full cell list
                # overcounts -- BvK representatives per k instead (home
                # block at Γ; exact torus fold at multi-k). Mirrors
                # the RHF accelerator fold; only the bridged
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
            # step) makes the Pulay B-matrix singular; the degenerate
            # solve then returns a garbage extrapolated Fock whose Aufbau
            # diagonalisation can occupy the wrong orbital. See the RHF
            # twin in pbc_bipole.py and
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

        if level_shift_schedule is not None:
            level_shift_b = level_shift_schedule.at(iter_idx)
        else:
            level_shift_b = level_shift_static
        if level_shift_b != 0.0 and not converged:
            F_alpha_for_diag: List[np.ndarray] = []
            F_beta_for_diag: List[np.ndarray] = []
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

        new_C_alpha: List[np.ndarray] = []
        new_eps_alpha: List[np.ndarray] = []
        new_C_beta: List[np.ndarray] = []
        new_eps_beta: List[np.ndarray] = []
        for idx in range(n_k):
            C_a, eps_a = _diag_in_orth_basis(
                F_alpha_for_diag[idx],
                X_k_list[idx],
            )
            C_b, eps_b = _diag_in_orth_basis(
                F_beta_for_diag[idx],
                X_k_list[idx],
            )
            new_C_alpha.append(C_a)
            new_eps_alpha.append(eps_a)
            new_C_beta.append(C_b)
            new_eps_beta.append(eps_b)
        # --- MOM reorder (iter >= 2 only) ---
        # use_mom holds every cycle; SPINLOCK PATTERN_HOLD holds only cycles
        # 2..spinlock_iterations, then releases.
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
                    sel = _mom_select(
                        C_k,
                        S_k,
                        C_prev_occ_k,
                        n_occ_spin,
                        eps_new=eps_k,
                    )
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
            if not exchange_split_active:
                _zero_cross_cell_density(
                    D_alpha_new, D_alpha_new.blocks[0].shape[0], n_k
                )
                _zero_cross_cell_density(D_beta_new, D_beta_new.blocks[0].shape[0], n_k)
        else:
            D_alpha_new = _spin_density(C_alpha_per_k, n_alpha)
            D_beta_new = _spin_density(C_beta_per_k, n_beta)

        # The provisional energy/commutator criteria describe D_used. Keep
        # iterating unless both newly diagonalised spin densities represent
        # the same fixed point; otherwise a terminal rebuild would demote a
        # result after the loop had already stopped.
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

        # --- ODA mixing (extra Fock build) ---
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
                f"  ODA: lambda = {oda_step.lam:.4f} "
                f"(g0 = {oda_step.g0:+.3e}, g1 = {oda_step.g1:+.3e})"
            )
        else:
            D_alpha_prev = D_alpha_used
            D_beta_prev = D_beta_used
            D_alpha_real = D_alpha_new
            D_beta_real = D_beta_new
            density_from_c_per_k = True

        # Snapshot for next iter MOM (use_mom: every cycle; PATTERN_HOLD: while
        # inside the hold window, so cycle k+1 can hold against cycle k).
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
                    # #116: re-sync both incremental spin chains to this exact
                    # build, so a failed confirmation continues on the operator
                    # it is judged by (a no-op when the loop exits here).
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

    # The spin diagnostic is a Brillouin-zone average, not a Gamma-block
    # property. This complex-valued formulation also handles shifted meshes
    # and fractional per-spin occupations without dropping Bloch phases.
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

    # ---- Ionic-Gamma basin-health diagnostic (smearing straddle, per spin) -
    # See run_pbc_bipole_rks / smearing_basin_warning. Both spin channels
    # are checked; the minimum per-k gap across spins governs the trigger.
    basin_warning = smearing_basin_warning(
        smearing_T,
        [
            (eps_alpha_per_k, occ_alpha_per_k, n_alpha, 1.0),
            (eps_beta_per_k, occ_beta_per_k, n_beta, 1.0),
        ],
        entropy,
        "run_pbc_bipole_uhf",
    )
    if basin_warning is not None:
        plog.info("  WARNING: " + basin_warning)
        warnings.warn(basin_warning, UserWarning, stacklevel=2)

    # ---- Post-loop: evaluate the exact density returned to callers.  The
    # last diagonalisation commits a new density on both convergence and
    # max-iteration exhaustion; always rebuild so Fock, energy and density
    # describe one state.
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
    D_tot = _combine_density_sets(
        basis, system, lat_opts_2e, D_alpha_real, D_beta_real
    )
    E_kin_final = _lattice_contract(D_tot, T_lat, operator_name="T")
    E_ne_final = _lattice_contract(D_tot, V_lat, operator_name="V_ne")
    E_2e_final = _fb.e_2e_k_correction + 0.5 * (
        _lattice_contract(D_alpha_real, _fb.f2e_alpha_real, operator_name="F2e")
        + _lattice_contract(D_beta_real, _fb.f2e_beta_real, operator_name="F2e")
    )
    E_elec = E_kin_final + E_ne_final + E_2e_final
    E_total = float(E_elec) + e_nuc + e_dft_plus_u
    # Fresh E_total doesn't include spheropole -- add it (3D only;
    # zero in the direct dim<3 gauge, omitted under the corrected
    # gauge).
    if system.dim == 3 and not exchange_split_active:
        E_sphero_final = compute_ext_el_spheropole(D_tot, basis, system, lat_opts)
        E_total += E_sphero_final
    else:
        E_sphero_final = None
    energy_components[-1] = PBCBipoleEnergyComponents(
        iter=int(iter_idx),
        e_total=float(E_total),
        e_electronic=float(E_elec),
        e_kinetic=float(E_kin_final),
        e_nuclear_attraction=float(E_ne_final),
        e_two_electron=float(E_2e_final),
        e_nuclear_repulsion=float(e_nuc),
        e_bielet_zone_ee=(
            None if use_ewald_j_split else float(E_2e_final)
        ),
        e_ext_el_spheropole=E_sphero_final,
        e_j_short_range=_fb.e_j_short_range,
        e_j_long_range=_fb.e_j_long_range,
        e_exchange=_fb.e_exchange,
        e_exchange_finite_size=_fb.e_exchange_finite_size,
        e_j_multipole=_fb.e_j_multipole,
        e_dft_plus_u=float(e_dft_plus_u),
    )

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

    return PBCBipoleUHFResult(
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=init_alpha is not None or init_beta is not None or initial_density_k is not None),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=float(E_total),
        e_electronic=float(E_elec),
        e_nuclear=e_nuc,
        e_ext_el_spheropole=E_sphero_final,
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
        basin_warning=basin_warning,
        smearing_temperature=smearing_T,
        fermi_level=float(mu_alpha) if n_alpha > 0 else float(mu_beta),
        fermi_level_alpha=float(mu_alpha),
        fermi_level_beta=float(mu_beta),
        entropy=float(entropy),
        free_energy=float(free_energy_final),
        occupations_alpha=[np.asarray(o, dtype=float) for o in occ_alpha_per_k],
        occupations_beta=[np.asarray(o, dtype=float) for o in occ_beta_per_k],
        kpoints_cart=np.asarray(k_points, dtype=float).reshape(-1, 3),
        kpoint_weights=np.asarray(weights, dtype=float).reshape(-1),
        fock_mixing=fock_mixing_value,
    )
