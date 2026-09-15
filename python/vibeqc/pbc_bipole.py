"""BIPOLE-style periodic RHF driver in CRYSTAL's electrostatic gauge.

CRYSTAL's 3D periodic HF energy uses one shared Ewald state for the
point-charge tail terms and a separate screened real-space machinery for
the AO two-electron build. This driver mirrors that composition:

* ``V_ne`` and ``E_nn`` use ``EWALD_3D`` with one explicit
  ``EwaldOptions`` object -- a single shared Ewald state across the terms.
  The default 3D ``V_ne`` path evaluates the smooth reciprocal piece
  analytically with shifted AO-pair Fourier transforms.
* The optional ``use_ewald_j_split`` path builds
  ``J = J_SR(w) + J_LR(w)`` with the same a used by ``V_ne`` / ``E_nn``.
  ``J_LR`` is represented as real-space blocks for real-space
  (lattice-sum) energy contractions, and includes the electron-electron
  neutralising-background Fock potential ``-pi N_e /(a^2 V) . S(g)``.
* Exchange remains the full direct-space ``K`` from
  ``build_fock_2e_real_space``; no Madelung K shift is applied.
* Energies are always evaluated as real-space lattice contractions,
  ``S_g tr[D(g)H(g)] + 1/2S_g tr[D(g)F^2e(g)]``, not from a Γ-folded
  operator.

V_ne gauge placement
--------------------
CRYSTAL and vibe-qc use the same four-component Ewald decomposition
(real-space erfc, reciprocal-space K!=0 sum, self-energy, jellium
background), but place the G=0 correction differently:

* **CRYSTAL**: the jellium background ``-pi Q_n^2/(2 b^2 V)`` is added
  to the nuclear-repulsion term ``E_nn``.  ``V_ne`` includes only
  the K!=0 reciprocal sum; the G=0 term is handled implicitly through
  the total-energy cancellation.
* **vibe-qc**: the V_ne operator receives an explicit background
  ``+pi Q_n/(a^2 V) . S(g)``, and E_nn receives the standard
  ``-pi Q_n^2/(2 a^2 V)`` jellium term.  For a neutral cell these
  cancel exactly in E_total.  Per-component diagnostics (E_ne, E_nuc)
  therefore differ from CRYSTAL's ENECYCLE output by the background
  magnitude (~16 Ha for MgO/STO-3G), but the total energy is
  invariant.

This is still an algorithmic re-implementation, not a CRYSTAL wrapper,
and no external QC program is imported at runtime. Pisani-Dovesi-Roetti
(1988), Ch. II.4c, is the source for the periodic quartet expansion.
The dormant classifier and add-back prototype do not yet preserve the
exact route's three-translation Fock domain, so public drivers reject
``use_multipole_far_field=True``. The exact Ewald-J path is the supported
energy route. EXT EL-POLE and quartet-far-field parity remain uncertified.
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    EwaldOptions,
    GridOptions,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    SCFIteration,
    build_fock_2e_real_space,
    build_jk_2e_real_space,
    compute_nuclear_erfc_lattice,
    direct_lattice_cells,
    ewald_nuclear_repulsion,
    nuclear_repulsion_per_cell,
    real_space_density_from_kpoints,
)
from ._vibeqc_core import (
    monkhorst_pack as _native_monkhorst_pack,
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
from .periodic_rhf_multi_k_ewald import (
    _damp_lattice_matrix,
    _diag_in_orth_basis,
)
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicSCFAccelerator,
)
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .smearing._support import reject_unsupported_smearing_temperature

__all__ = [
    "PBCBipoleEnergyComponents",
    "PBCBipoleRHFResult",
    "run_pbc_bipole_rhf",
]


@dataclass
class PBCBipoleRHFResult:
    """Result of :func:`run_pbc_bipole_rhf`.

    Per-cell ``energy`` / ``e_electronic`` / ``e_nuclear`` and per-k
    matrices (``mo_energies``, ``mo_coeffs``, ``fock``, ``overlap``,
    ``hcore``) alongside the converged real-space ``density``. The last
    ``energy_components`` and ``scf_trace`` rows are refreshed after the
    terminal diagonalisation so they describe that returned density;
    ``initial_density_energy_components`` preserves the first density
    evaluation separately (CRYSTAL CYC0 for a legacy-gauge SAD run). For
    3D BIPOLE runs, ``ewald_alpha_bohr_inv`` records the single alpha used
    by V_ne / E_nn / optional J_LR.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool

    mo_energies: List[np.ndarray]
    mo_coeffs: List[np.ndarray]
    fock: List[np.ndarray]
    overlap: List[np.ndarray]
    hcore: List[np.ndarray]

    density: LatticeMatrixSet

    # Fields with defaults must come after all non-default fields
    # (Python 3.14 dataclass enforcement).
    runtime_backend: str = "pbc-bipole"
    e_ext_el_spheropole: Optional[float] = None
    scf_trace: List[SCFIteration] = field(default_factory=list)
    ewald_alpha_bohr_inv: Optional[float] = None
    # Absolute internal ket-image radius used by the erfc SR build. Analytic
    # gradients use this provenance to reproduce supported padded domains and
    # fail closed for combinations that have not been certified.
    sr_image_extent_bohr: Optional[float] = None
    # True when the Fock build used the group-invariant pair-resolved domain.
    # Gamma analytic gradients reproduce it; multi-k remains radial-only.
    pair_resolved_fock_domain: bool = False
    # Dudarev DFT+U contribution per unit cell (Hartree). 0 unless the
    # caller passed ``dft_plus_u=[HubbardSite(...)]``.
    e_dft_plus_u: float = 0.0
    energy_components: List[PBCBipoleEnergyComponents] = field(
        default_factory=list,
    )
    # Exchange convention provenance (option (b), 2026-06-10): True when
    # the run used the Ewald exchange split (K_SR(erfc) + K_LR(recip) +
    # G=0/Madelung correction, full-Bloch density, no spheropole term).
    # Analytic-gradient consumers use this flag to select the matching gauge
    # and refuse unsupported combinations.
    exchange_ewald_split: bool = False
    exchange_exxdiv: Optional[str] = None
    fock_mixing: float = 0.0
    # Cartesian k-points (bohr^-1) and weights this result spans, in the
    # same order as the per-k ``mo_coeffs`` / ``mo_energies`` lists. Carried
    # so optional Gamma-only / single-k output writers (molden, QVF
    # wavefunction) can locate the Gamma block instead of guessing that the
    # first k-point is Gamma. Mirrors the GDF multi-k result contract
    # (periodic_k_gdf.py). None for legacy results built without it.
    kpoints_cart: Optional[np.ndarray] = None
    kpoint_weights: Optional[np.ndarray] = None
    # Corrected-gauge overlap-fold convergence diagnostic. Appended to
    # preserve the positional signature of the historical result fields.
    overlap_fold_drift: Optional[float] = None
    # BIPOLE-EXACT-ZONE increment 1: radius (bohr) of the exact erfc
    # output zone when it was restricted below the operator cutoff
    # (None = exact zone == cutoff, the historical behavior). Far
    # operator cells carry J_LR + background only. Analytic-gradient
    # consumers fail closed on a set value (FD remains the production
    # force path).
    exact_zone_bohr: Optional[float] = None
    output_cell_farming_execution: Optional[
        "BipoleOutputCellFarmingExecution"
    ] = None
    n_electrons: int = 0
    occupations: List[np.ndarray] = field(default_factory=list)
    # Component breakdown evaluated on the density entering the first SCF
    # cycle.  This preserves CRYSTAL CYC0 / initial-guess evidence even
    # though ``energy_components[-1]`` is deliberately refreshed after the
    # terminal diagonalisation to match the density returned to callers.
    initial_density_energy_components: Optional[
        PBCBipoleEnergyComponents
    ] = None

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_mesh: object = None
    restart_kpoints: object = None
    restart_weights: object = None
    # Ewald accuracy used by the SCF; gradients must retain its finite sums.
    ewald_precision: float = 1e-8


@dataclass
class _PBCBipoleFockBuild:
    """Internal Fock-build bundle for one density in the BIPOLE driver."""

    f2e_real: LatticeMatrixSet
    f_k_list: List[np.ndarray]
    e_j_short_range: Optional[float] = None
    e_j_long_range: Optional[float] = None
    e_exchange: Optional[float] = None
    e_j_multipole: Optional[float] = None
    # Dormant PDR 1988 Ch. II.4c quartet-prototype diagnostic.
    e_j_bipolar_quartet: Optional[float] = None
    # k-space exchange correction (Ewald exchange split): the K_LR +
    # G=0/Madelung pieces enter F(k) directly (not the real-space f2e
    # blocks), so their energy contribution 1/2.S_k w_k Tr[D(k).ΔF(k)]
    # must be added to the lattice-contracted E_2e by the caller.
    e_2e_k_correction: float = 0.0
    # The q+G=0 (Madelung / exxdiv='ewald') share of e_2e_k_correction on
    # its own, so the gauge is visible next to the total (#82).
    e_exchange_finite_size: Optional[float] = None
    output_cell_farming_execution: Optional[
        "BipoleOutputCellFarmingExecution"
    ] = None


from .pbc_bipole_common import (
    bipole_confirmation_phase,
    bipole_final_density_phase,
    PBCBipoleEnergyComponents,
    _bloch_sum_blocks,
    _bloch_sum_blocks_multi_k,
    _cell_key,
    _compute_nuclear_lattice_ewald_reciprocal_ft,
    _crystal_ewald_options,
    _default_bipole_v_ne_grid_options,
    _density_set_gamma_or_lattice,
    _emit_bipole_semantic_diagnostic,
    _expand_ibz_kmesh_for_ewald_j,
    _lattice_contract,
    _lattice_contract_blocks,
    _zero_cross_cell_density,
    bloch_h_core_and_orthogonalizer,
    build_bipole_one_electron_lattice,
    bvk_torus_density_matrices,
    prepare_bipole_lattice_options,
    raise_bipole_ewald_real_cutoff,
    resolve_bipole_ewald_alpha,
    reject_bipole_ecp_options,
    reject_bipole_quartet_far_field,
    reject_bipole_lone_non_gamma_kpoint,
    reject_bipole_solver_options,
    restricted_bipole_commutator_norm,
    bipole_terminal_check,
    refresh_bipole_terminal_trace,
    resolve_fock_mixing,
    resolve_bipole_sr_image_extent,
    resolve_bipole_fock_symmetry,
    resolve_incremental_jk,
    warn_bipole_charged_cell,
    warn_bipole_legacy_multik_gauge,
    home_cell_block,
)
from .pbc_bipole_fock import (
    _sr_density_cells,
    BipoleFockContext,
    BipoleOutputCellFarmingExecution,
    build_bipole_restricted_fock,
)


def run_pbc_bipole_rhf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    options=None,
    *,
    linear_dep_threshold: float = 1e-7,
    fock_mixing: Optional[float] = None,
    canonical_orth_normalize_diag_first: bool = True,
    level_shift_schedule: Optional["LevelShiftSchedule"] = None,
    use_mom: bool = False,
    use_oda: bool = False,
    oda_trust_lambda_max: float = 1.0,
    use_incremental_fock: bool = True,
    use_ewald_j_split: Optional[bool] = None,
    ewald_omega: Optional[float] = None,
    ewald_precision: float = 1e-8,
    v_ne_grid_options: Optional[GridOptions] = None,
    use_multipole_diag: bool = False,
    use_multipole_far_field: Optional[bool] = False,
    multipole_l_max: int = 2,
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
    bz_integration: Optional[str] = None,
    dft_plus_u: Optional[List["HubbardSite"]] = None,
) -> PBCBipoleRHFResult:
    """Multi-k closed-shell RHF via the CRYSTAL-gauge BIPOLE scaffold.

    ``fock_mixing`` overrides ``options.fock_mixing`` when supplied and is
    the previous-Fock matrix weight in ``[0, 1)``.  The caller's options field
    is not rewritten.

    ``dft_plus_u``: optional list of :class:`HubbardSite`. When set,
    the Dudarev rotationally-invariant per-spin V_U is added to every
    per-k Fock matrix using the same per-spin Bloch-summed
    convention as :func:`run_pbc_bipole_uhf` (closed-shell:
    ``P_s = P_total / 2``, ``E_U_total = 2 x E_s``).  The +U
    energy lands on ``result.e_dft_plus_u``.

    Algorithm (real-space two-electron / bielectronic build):
      1. Real-space one-electron integrals S(g), T(g), V_ne(g) at
         ``opts.lattice_opts.cutoff_bohr``. For 3D systems V_ne uses
         the same Ewald a as E_nn.
      2. Bloch-sum to S(k), Hcore(k) per k-point; canonical-orth X(k).
      3. Initial guess via ``opts.initial_guess`` (default SAD).
      4. SCF iter:
         a. Build F^{2e}(g). With ``use_ewald_j_split=True`` this is
            ``J_SR(g;w) + J_LR(g;w) + V_bg.S(g) - 1/2K(g)`` where the
            exchange convention depends on ``use_exchange_ewald_split``
            (below). With the flag off, use the legacy direct-only
            ``build_fock_2e_real_space`` scaffold.
         b. Bloch-sum F^{2e}(g) -> F(k); add Hcore(k).
         c. Energy: E_elec = S_g tr[D(g)Hcore(g)]
            + 1/2S_g tr[D(g)F^2e(g)] in real-space block form
            (real-space lattice-sum convention).
         d. Optional DIIS extrapolation of F(k) via [F,DS] errors.
         e. Optional LEVSHIFT shift on F(k).
         f. Diagonalise F(k) -> C(k), e(k).
         g. Optional MOM reorder of occupied subspace.
         h. Rebuild D_real via real_space_density_from_kpoints.
         i. Commit the orbital-representable density. The historical ODA
            mixed-density hook fails closed at driver entry.
      5. E_total = E_elec + E_nuc.

    ``use_ewald_j_split`` defaults to ``None``. In that mode the
    driver automatically uses the CRYSTAL-gauge Ewald-J split for 3D
    systems and keeps the old direct-only path for dim < 3 diagnostic
    runs. Pass ``False`` explicitly only when you want the legacy
    direct-only F^2e scaffold for debugging. (Passing ``True`` on a
    dim < 3 system raises -- the Ewald split needs a 3D reciprocal
    lattice.)

    ``sr_image_precision`` controls the internal ket-image radius of every
    erfc short-range build. The default ``1e-6`` uses the conservative
    smeared-pair range and enables charge-pair Schwarz screening; pass
    ``None`` for the historical unpadded traversal (which also suppresses
    automatic Fock reduction unless explicitly requested). An explicit
    ``sr_image_extent_bohr`` is the M4a absolute-radius oracle and overrides
    the precision policy without changing the caller's screening flag.

    ``use_fock_symmetry_reduce=None`` auto-enables the group-invariant
    pair-resolved representative build when ``system`` has attached symmetry
    and the corrected Ewald exchange split is active. Pass ``False`` to use
    the unreduced radial build. Enforcement without reduction is the explicit
    diagnostic pair ``use_fock_symmetry=True`` and
    ``use_fock_symmetry_reduce=False``.

    ``farm_output_cells=True`` distributes complete direct-ERI real-space
    output blocks across MPI ranks and allgathers them before incremental
    Fock state is updated. Each task retains the full internal translation
    sum; this changes scheduling, not the Coulomb operator. The first
    supported composition is the corrected 3D Ewald split with a padded
    short-range domain. Generic BIPOLE callers leave it off; chi-CCM's
    restricted four-center selector supplies ``"chi-direct-output-cell"``
    as the execution label.

    ``use_exchange_ewald_split`` (2026-06-10 energy-assembly redesign;
    multi-k q!=0 channels 2026-06-11; multi-k default flip 2026-06-13)
    defaults to ``None`` = auto: ON for any 3D run under the Ewald J
    split (Γ AND multi-k), OFF otherwise. When ON, the exchange uses
    the Ewald split convention (module docstring of
    :mod:`vibeqc.bipole_fock_ewald`)::

        K(k) = K_SR(erfc w, direct) + K_LR(erf w, reciprocal, q+G!=0)
               + (ξ_M - pi/(V_sc.w^2)).S(k).D(k).S(k)

    with ``ξ_M`` the probe-charge Ewald (Madelung) constant of the
    BvK supercell (= the unit cell at Γ; ``V_sc = n_k.V``) when
    ``exchange_exxdiv='ewald'`` (the default; PySCF-equivalent) or 0
    when ``'none'``. At multi-k the LR term couples every k-point
    pair through the momentum-transfer channels ``q = k - k′`` (see
    :func:`vibeqc.bipole_fock_ewald.compute_K_long_range_at_k`). In
    this mode the SCF density is the full Bloch fold (the Γ-locality
    projection ``P(g!=0)=0`` is **not** applied), and the EXT
    EL-SPHEROPOLE term is omitted from the total -- at the corrected
    gauge it is a double-count (MgO Γ fixed-density audit,
    2026-06-10: the reassembled total matches PySCF GDF RHF to
    truncation with no spheropole term). When OFF (explicit
    ``False``), the legacy convention is kept: full-Coulomb
    direct-space K, Γ-locality projection at n_k = 1, spheropole term
    added -- known to mis-state absolute energies on tight ionic
    cells (kept only for the legacy-gauge analytic gradient + parity
    diagnostics). The corrected multi-k gauge needs a Monkhorst-Pack
    ``BlochKMesh`` carrying its ``mesh`` metadata; under the auto
    default an ad-hoc k-list (band path / explicit list) at multi-k
    falls back to the legacy gauge with a log note, and an explicit
    ``True`` with such a mesh raises.

    For dim < 3 the whole one- and two-electron Coulomb gauge falls
    back to ``DIRECT_TRUNCATED`` (no Ewald, no reciprocal sum), and the
    ``EXT EL-SPHEROPOLE`` correction -- a 3D-Ewald reciprocal-space
    (K=0 limit) term -- is identically zero, so it is omitted and
    ``e_ext_el_spheropole`` is ``None``. This is a diagnostic direct-sum
    path, not a 1-D/2-D Coulomb model: padding invariance and agreement with
    molecular RHF in the isolated-cell limit do not certify compact-cell
    cutoff convergence. Public periodic jobs and ``PeriodicSCFProvider``
    therefore refuse BIPOLE at dim < 3 (see
    ``tests/test_pbc_bipole_dim_lt3.py`` and IID 542).

    For 3D systems the default ``V_ne`` implementation is analytic:
    erfc-screened nuclear attraction from libint plus a reciprocal-space
    AO-pair Fourier-transform sum. Passing ``v_ne_grid_options`` opts
    into the older grid-quadrature long-range ``V_ne`` path for
    diagnostics.
    """
    reject_bipole_quartet_far_field(
        use_multipole_far_field,
        driver="run_pbc_bipole_rhf",
    )
    reject_bipole_lone_non_gamma_kpoint(
        kmesh,
        driver="run_pbc_bipole_rhf",
    )
    from ._vibeqc_core import PeriodicRHFOptions

    opts = options if options is not None else PeriodicRHFOptions()
    guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", None),
        driver="run_pbc_bipole_rhf",
        supported=periodic_guess_capabilities('bipole', 'RHF', dim=getattr(system, "dim", 3), multi_k=(len(kmesh.kpoints) > 1 or np.prod(kmesh.mesh) > 1), transport='lattice'),
        restart_supplied=initial_density is not None or initial_density_k is not None,
    )
    _patom_seed_pending = guess == InitialGuess.PATOM
    reject_bipole_ecp_options(
        opts,
        driver="run_pbc_bipole_rhf",
        basis=basis,
        system=system,
    )
    reject_bipole_solver_options(opts, driver="run_pbc_bipole_rhf")
    if int(opts.max_iter) < 1:
        raise ValueError("run_pbc_bipole_rhf: max_iter must be at least 1")
    if not isinstance(farm_output_cells, (bool, np.bool_)):
        raise TypeError("run_pbc_bipole_rhf: farm_output_cells must be bool")
    farm_output_cells = bool(farm_output_cells)
    if output_cell_farming_strategy not in ("block", "cyclic"):
        raise ValueError(
            "run_pbc_bipole_rhf: output_cell_farming_strategy must be "
            "'block' or 'cyclic'"
        )
    if (
        not isinstance(output_cell_farming_task_kind, str)
        or not output_cell_farming_task_kind.strip()
    ):
        raise ValueError(
            "run_pbc_bipole_rhf: output_cell_farming_task_kind must be a "
            "non-empty string"
        )
    if use_oda:
        raise NotImplementedError(
            "run_pbc_bipole_rhf: ODA is unavailable because a mixed "
            "line-search density has no single orbital representation; "
            "use DIIS instead"
        )
    requested_fock_mixing = resolve_fock_mixing(
        opts,
        fock_mixing,
        where="run_pbc_bipole_rhf",
    )
    if bz_integration is not None:
        bz_kind = str(bz_integration).strip().lower()
        if bz_kind != "smearing":
            raise NotImplementedError(
                "run_pbc_bipole_rhf: bz_integration only accepts None or "
                "'smearing'; parameter-free Gilat integration is available "
                "on the BIPOLE RKS route only."
            )
    reject_unsupported_smearing_temperature(
        opts,
        "run_pbc_bipole_rhf",
        detail=(
            "BIPOLE RHF requires integer occupations; use BIPOLE "
            "RKS/UHF/UKS for finite-temperature smearing."
        ),
    )
    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)
    # CRYSTAL-style gauge separation (per the EWALD_3D / BIPOLE audit):
    # V_ne and E_nn use Ewald with one shared alpha. F^{2e} uses the
    # direct lattice cell list for J_SR/K; the optional J_LR reciprocal
    # sum consumes the same alpha as the one-electron Ewald state.
    (
        use_ewald_j_split,
        use_ewald_j_split_auto,
        lat_opts_2e,
        lat_opts_1e,
    ) = prepare_bipole_lattice_options(system, lat_opts, use_ewald_j_split, plog)

    plog.info(f"PBC BIPOLE (CRYSTAL-gauge) / cutoff {lat_opts.cutoff_bohr:.2f} bohr")
    plog.info(
        f"  V_ne + E_nn  : {lat_opts_1e.coulomb_method.name}"
        f"   (Ewald gauge for point-charge tails)"
    )
    plog.info(
        f"  F^2e (J + K) : "
        f"{'EWALD_J_SPLIT' if use_ewald_j_split else lat_opts_2e.coulomb_method.name}"
        f"{' (auto)' if use_ewald_j_split_auto else ''}"
        f"   (direct J_SR/K cell list"
        f"{' + reciprocal J_LR' if use_ewald_j_split else ''})"
    )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")

    # Closed-shell sanity.
    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            f"run_pbc_bipole_rhf: closed-shell RHF requires even electron "
            f"count; got {n_elec}"
        )
    if system.multiplicity != 1:
        raise ValueError(
            f"run_pbc_bipole_rhf: requires multiplicity=1; got {system.multiplicity}"
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
        # IBZ-reduced input meshes are EXPANDED TO THE FULL MESH up
        # front and the whole SCF runs on the full mesh. The previous
        # "IBZ-native" shortcut diagonalised at the IBZ points and
        # replicated D(k) into each star without the AO rotation
        # D(R.k) = P(R).D(k).P(R)ᵀ -- exact only for trivial stars (the
        # He validation cells); on MgO/STO-3G [2,2,2] it left the SCF
        # unconverged 8.25 Ha from the full-mesh result (2026-06-10
        # probe; regression in tests/test_pbc_bipole_multik_ewald_split
        # pins full==IBZ equality). True IBZ-native reduction needs the
        # symmetry-adapted k-star transport -- groundwork + probe
        # findings live in vibeqc.periodic_k_symmetry.
        kmesh_full = _expand_ibz_kmesh_for_ewald_j(system, kmesh, plog)
        if len(list(kmesh_full.kpoints)) > len(k_points):
            plog.info(
                "  IBZ input mesh expanded to the full MP mesh for the "
                "whole SCF (correctness; IBZ-native reduction pending "
                "symmetry-adapted k-star transport)"
            )
            kmesh = kmesh_full
            k_points = list(kmesh.kpoints)
            weights = np.asarray(kmesh.weights, dtype=float)
            n_k = len(k_points)
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
            "run_pbc_bipole_rhf: multi-k READ requires an explicit complete "
            "real-space initial_density block set; a Gamma-only read_density "
            "cannot reconstruct the per-k Bloch density."
        )
    plog.info(
        f"k-mesh: {n_k} k-point{'s' if n_k != 1 else ''}, "
        f"weights sum = {weights.sum():.4f}"
    )

    # ---- Exchange Ewald-split resolution (option (b), 2026-06-10) ----
    # Corrected exchange convention K_SR(erfc) + K_LR(reciprocal) +
    # G=0/Madelung correction (bipole_fock_ewald module docstring).
    # The corrected gauge is the DEFAULT under the Ewald J split at BOTH
    # Γ and multi-k (Phase-5 flip, 2026-06-13): the q = k-k' != 0
    # LR-exchange channels (Phase 3, 2026-06-11/12) are parity-validated
    # (H₂ box [2,1,1] vs PySCF KRHF, supercell-unfolding identity to
    # +0.0001 mHa/cell; MgO [2,2,2] c8 -14.8 mHa vs legacy +3.9 Ha).
    # Pass use_exchange_ewald_split=False for the legacy gauge.
    if exchange_exxdiv not in ("ewald", "none"):
        raise ValueError(
            f"run_pbc_bipole_rhf: exchange_exxdiv must be 'ewald' or "
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
            "run_pbc_bipole_rhf: use_exchange_ewald_split=True requires "
            "the Ewald J split (use_ewald_j_split=True)."
        )
    # Multi-k split: the q-channel tables, the BvK-torus density fold,
    # and the supercell Madelung correction all need the true
    # Monkhorst-Pack dimensions. Ad-hoc k-lists carry mesh = (1,1,1)
    # placeholders (see the to_bloch_kmesh binding). Under the auto
    # default an ad-hoc multi-k mesh falls back to the legacy gauge
    # (no surprise breakage for explicit-k-list / band-path runs);
    # an explicit use_exchange_ewald_split=True still raises.
    _bvk_mesh: Optional[Tuple[int, int, int]] = None
    if exchange_split_active and n_k > 1:
        _mesh_attr = tuple(
            int(x) for x in getattr(kmesh, "mesh", (1, 1, 1))
        )
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
                    "run_pbc_bipole_rhf: the Ewald exchange split at multi-k "
                    "requires a Monkhorst-Pack BlochKMesh carrying its mesh "
                    f"dimensions (got mesh={_mesh_attr} for {n_k} k-points). "
                    "Build the mesh via monkhorst_pack(...); ad-hoc k-point "
                    "lists are not supported on the corrected gauge."
                )
        else:
            _bvk_mesh = _mesh_attr

    warn_bipole_legacy_multik_gauge(system, exchange_split_active, n_k, plog)
    warn_bipole_charged_cell(system, plog)

    # CRYSTAL-style shared Ewald state for all point-charge-tail terms.
    # V_ne, E_nn, and the optional reciprocal J^LR build must consume the
    # same alpha AND the same K_max -- because finite-cutoff G=0
    # cancellation requires matched reciprocal envelopes.
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
            f"  Ewald state: a = {omega_used:.6f} bohr⁻¹, "
            f"cutoff_real = {lat_opts_1e.nuclear_cutoff_bohr:.2f} bohr, "
            f"K_max = {ewald_k_max:.2f} bohr⁻¹, "
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
    if farm_output_cells and (
        not exchange_split_active or _sr_image_extent is None
    ):
        raise NotImplementedError(
            "run_pbc_bipole_rhf: direct-ERI output-cell farming currently "
            "requires the corrected 3D Ewald exchange split and a padded "
            "short-range image domain"
        )

    # Probe-charge Ewald (Madelung) constant for the exchange G=0
    # correction (exxdiv='ewald'; PySCF-equivalent). a-independent.
    # At multi-k the constant is the BvK-SUPERCELL Madelung -- the
    # multi-k SCF is the supercell Γ SCF exactly unfolded, and PySCF's
    # _ewald_exxdiv_for_G0 applies this same single madelung(cell,
    # kpts) value to every k-point.
    _xi_madelung = 0.0
    if exchange_split_active and exchange_exxdiv == "ewald":
        if n_k > 1:
            from .bipole_fock_ewald import probe_charge_madelung_supercell

            assert _bvk_mesh is not None
            _xi_madelung = probe_charge_madelung_supercell(system, _bvk_mesh)
        else:
            from .bipole_fock_ewald import probe_charge_madelung

            _xi_madelung = probe_charge_madelung(system)
    overlap_fold_drift: Optional[float] = None
    if exchange_split_active:
        plog.info(
            f"  Exchange: Ewald split -- K_SR(erfc w) + K_LR(reciprocal"
            + (f", {n_k}^2 (k,k′) q-channels" if n_k > 1 else "")
            + f") + G=0 correction (exxdiv={exchange_exxdiv}"
            + (
                f", ξ_M{'(supercell)' if n_k > 1 else ''} = "
                f"{_xi_madelung:.6f} Ha"
                if exchange_exxdiv == "ewald"
                else ""
            )
            + ")"
        )
        plog.info(
            "  Density: full Bloch fold (Γ-locality projection OFF); "
            "EXT EL-SPHEROPOLE omitted (gauge-consistent)"
        )

    # ---- Real-space one-electron integrals -------------------------------
    # S, T use cell-list-only cutoff (lat_opts_2e -- they're independent
    # of coulomb_method). V_ne uses lat_opts_1e so the EWALD_3D path is
    # taken on 3D systems (CRYSTAL-equivalent gauge).
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
            log_v_ne_ft=True,
        )
    cells = list(S_lat.cells)
    plog.info(f"n_cells in lattice sum = {len(cells)}")

    # ---- Lattice-fold convergence guard (gauge-independent) -------------
    # One shared guard for all four drivers; physics + thresholds in
    # enforce_bipole_fold_support. Hoisted above the gauge condition
    # 2026-08-06 (maintainer-approved): the k-fold eigenproblem consumes
    # S(k) in BOTH gauges, so the legacy path must not skip the guard.
    from .pbc_bipole_common import enforce_bipole_fold_support

    overlap_fold_drift = enforce_bipole_fold_support(
        basis,
        system,
        lat_opts_2e,
        method="rhf",
        n_k=n_k,
        k_points=(k_points if n_k > 1 else None),
        exchange_split_active=exchange_split_active,
        plog=plog,
    )

    # SYM3b Fock-symmetry resolution. Resolved BEFORE the density-list
    # decision: pair-resolved mode (M2/M3 of the pair-resolved truncation
    # workstream) stores the physical density on the group-invariant pair
    # support instead of the radial 2x-cutoff list. Direct SR J/K masks a
    # private copy to the qualifying tensor triples.
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
        auto_reduce_safe=(_sr_image_extent is not None),
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

    # Density cell list. Under the Ewald exchange split the SCF density
    # is stored on a 2x-cutoff list: the C++ JK builder's traversal
    # forms cell-pair differences |b-a| up to 2x the cutoff, and P(h)
    # lookups beyond the density's own list are silently skipped
    # (cpp/src/periodic_fock.cpp ``p_block``). With the non-decaying Γ
    # Bloch fold those dropped alive-overlap J/K terms are
    # SCF-exploitable: MgO/STO-3G c8 converged 0.70 Ha BELOW PySCF with
    # the electron count off by 0.43 in the reference metric
    # (2026-06-10 diagnosis). Traversal cost is unchanged -- the builder
    # derives its quartet cells from lat_opts; the density list only
    # feeds lookups. Operators (S/T/V/F2e blocks) stay on ``cells``;
    # the energy contraction iterates operator cells (see
    # ``_lattice_contract_blocks``).
    if _pair_mode:
        # M3: pair-resolved density support -- the same 2x-cutoff
        # criterion, but applied to atom PAIRS (|r_b + L.h - r_a| <=
        # 2*cutoff). The direct SR builder masks its private density copy
        # to the qualifying sub-blocks, so the tensor triple set remains
        # group-invariant (the radial list chops symmetry-equivalent
        # near-field couplings: 3 of MgO's 6 NN Mg-O couplings at c6).
        cells_density = list(_fock_sym_map.density_domain.cells)
        plog.info(
            f"  density cell list: {len(cells_density)} cells "
            f"(pair-resolved support at 2x cutoff; SR tensor has "
            f"{_fock_sym_map.density_domain.n_triples} qualifying pairs)"
        )
    elif exchange_split_active or lat_opts_2e.pair_complete_1e:
        cells_density = list(
            _sr_density_cells(basis, system, lat_opts_2e, _sr_image_extent)
        )
        plog.info(
            f"  density cell list: {len(cells_density)} cells "
            + ("(physical exchange density support)" if lat_opts_2e.pair_complete_1e
               else "(2x cutoff -- resolves every P(b-a) difference)")
        )
    else:
        cells_density = cells

    # Per-k S(k), Hcore(k), orthogonaliser X(k).
    from .linear_dependence import (
        check_overlap_matrix,
        format_linear_dependence_report,
        raise_if_severe,
        scf_preflight_overlap_check,
    )

    S_k_list: List[np.ndarray] = []
    T_k_list: List[np.ndarray] = []
    V_ne_k_list: List[np.ndarray] = []
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
                cond_str = (
                    f"{report.condition_number:.2e}"
                    if np.isfinite(report.condition_number)
                    else "+inf"
                )
                plog.info(
                    f"[{prefix}] overlap [{overlap_label}]: "
                    f"nbf={report.n_basis}, "
                    f"min eig={report.min_eigenvalue:+.2e}, "
                    f"cond={cond_str}, severity={report.severity}"
                )
                plog.write_raw(format_linear_dependence_report(report))
            raise_if_severe(report)
        overlap_reports.append(report)
        if n_occ > n_kept:
            raise RuntimeError(
                f"run_pbc_bipole_rhf: canonical orth at k={k_idx} "
                f"dropped too many directions (n_occ={n_occ}, n_kept={n_kept})"
            )
        S_k_list.append(S_k)
        T_k_list.append(T_k)
        V_ne_k_list.append(V_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)
    if n_k > 16:
        severity_rank = {"ok": 0, "warn": 1, "error": 2, "critical": 3}
        worst = max(
            overlap_reports,
            key=lambda r: severity_rank.get(r.severity, -1),
        )
        min_s = min(float(r.min_eigenvalue) for r in overlap_reports)
        max_cond = max(float(r.condition_number) for r in overlap_reports)
        cond_str = f"{max_cond:.2e}" if np.isfinite(max_cond) else "+inf"
        plog.info(
            f"overlap [k-mesh summary]: n_k={n_k}, nbf={basis.nbasis}, "
            f"min eig={min_s:+.2e}, max cond={cond_str}, "
            f"severity={worst.severity}"
        )

    # ---- Nuclear repulsion per cell --------------------------------------
    if ewald_options_1e is not None:
        e_nuc = float(ewald_nuclear_repulsion(system, ewald_options_1e))
    else:
        e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts_1e))
    plog.info(f"E_nuc per cell ({lat_opts_1e.coulomb_method.name}) = {e_nuc:+.10f} Ha")

    # ---- Initial guess ---------------------------------------------------
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
    def _rebuild_real_space_density(C_per_k_local):
        return real_space_density_from_kpoints(
            C_per_k_local,
            [n_occ] * n_k,
            kmesh,
            cells_density,
        )

    D_real = _rebuild_real_space_density(C_per_k)
    if not exchange_split_active:
        _zero_cross_cell_density(D_real, basis.nbasis, n_k)

    # Caller-supplied warm-start density takes precedence over both the
    # SAD/Hcore guess engine and the Hcore-diag fallback. The caller is
    # responsible for matching ``initial_density`` blocks against the
    # canonical ``direct_lattice_cells(kmesh)`` ordering (which is what
    # the SCF's ``D_real`` uses). Used by the NEB driver for within-
    # image density warm-start across outer iterations + within FD-
    # gradient displaced SCFs (periodic NEB warm-start milestone
    # follow-up).
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
                f"run_pbc_bipole_rhf: initial_density has {len(blocks_in)} "
                f"blocks; expected {len(D_real.cells)} (one per cell in "
                f"direct_lattice_cells(kmesh))"
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
        # SAD override (place SAD density at g=0; zeros elsewhere).
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
            plog.info(f"initial guess: {guess.name} (g=0 density from GuessEngine)")
            for g_idx in range(len(D_real.cells)):
                if (D_real.cells[g_idx].index == np.array([0, 0, 0])).all():
                    D_real.set_block(g_idx, D_engine)
                else:
                    D_real.set_block(g_idx, np.zeros_like(D_engine, dtype=float))
        else:
            plog.info(f"initial guess: {guess.name} (Hcore-diag per k)")
        initial_density_is_local = D_engine is not None
        density_from_c_per_k = not initial_density_is_local

    D_real_prev: Optional[LatticeMatrixSet] = None

    # ---- SCF aids: damping, accelerator family, LEVSHIFT, MOM, ODA ------
    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(f"run_pbc_bipole_rhf: damping must be in [0,1); got {damping}")

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
            f"level_shift_schedule must be a LevelShiftSchedule or None; "
            f"got {type(level_shift_schedule).__name__}"
        )
    if level_shift_schedule is not None:
        plog.info(f"level_shift_schedule: {level_shift_schedule.as_list()}")

    # CRYSTAL-style FMIXING: blend previous Fock into current before
    # diagonalisation.  Applied after DIIS, before level-shift.  Same
    # convention as the gamma EWALD_3D and GDF drivers.
    fock_mixing_value = requested_fock_mixing
    if fock_mixing_value != 0.0:
        plog.info(
            f"fock mixing: CRYSTAL FMIXING "
            f"{100.0 * fock_mixing_value:.1f}% "
            "(previous Fock matrix weight)"
        )

    if use_mom:
        plog.info("MOM (Maximum Overlap Method): ON")
    C_prev_occ_per_k: Optional[List[np.ndarray]] = None

    if use_oda and use_diis:
        raise ValueError(
            "run_pbc_bipole_rhf: use_oda and use_diis are mutually exclusive"
        )
    if use_oda:
        if not (0.0 < oda_trust_lambda_max <= 1.0):
            raise ValueError(
                f"oda_trust_lambda_max must be in (0, 1]; got {oda_trust_lambda_max}"
            )
        plog.info(
            f"ODA (Optimal Damping): ON (+1 Fock build/iter, "
            f"trust l_max = {oda_trust_lambda_max})"
        )

    # ---- Optional: Ewald J-split F^2e build (Phase 5 of BIPOLE branch) ---
    j_lr_cache = v_ne_lr_cache
    if use_ewald_j_split:
        # CRYSTAL-equivalent gauge: V_ne + E_nn use Ewald, F^2e uses
        # J^SR(direct erfc-screened) + J^LR(analytic reciprocal-sum) - 1/2K.
        # Single shared a between V_ne, E_nn, and J_LR (one shared
        # Ewald state).
        #
        # Multi-k J^LR uses Bloch-summed shifted-ν AO-pair FTs and a
        # k-space r̂(K). The operator is materialised as real-space
        # blocks below so both diagonalisation and real-space energy
        # accounting see the same long-range J.
        if system.dim != 3:
            raise ValueError(
                f"use_ewald_j_split requires dim=3 (3D periodic). Got dim={system.dim}."
            )
        if n_k > 1 and _ir_mapping.size == 0:
            # Non-uniform weights without ir_mapping: can't expand.
            uniform_w = 1.0 / float(n_k)
            if not np.allclose(weights, uniform_w, atol=1e-9):
                raise ValueError(
                    "use_ewald_j_split at multi-k requires uniform full-mesh "
                    "weights or an IBZ-reduced Monkhorst-Pack mesh carrying "
                    "ir_mapping metadata so the driver can expand it. "
                    f"Got non-uniform weights = {weights.tolist()}."
                )
        from .bipole_fock_ewald import (
            _build_j_long_range_cache,
            compute_J_long_range_real_space_blocks,
            compute_rho_hat_from_k_density,
        )

        assert omega_used is not None
        plog.info(
            f"Ewald J-split F^2e: ON (CRYSTAL-equivalent gauge); "
            f"w = {omega_used:.4f} bohr⁻¹, precision = {ewald_precision:.0e}"
        )
        # Pre-build the shifted-ν FT cache once -- invariant across SCF
        # iters + k-points within an iter.  Γ-only needs the same cache
        # for real-space energy blocks even though the Fock can be built
        # from the k=0 folded matrix.
        if _pair_mode:
            # M3: the J^LR blocks must land on the pair-resolved output
            # template (mapping.cells reaches past the radial list), so
            # a dedicated cache is built on it -- the V_ne cache stays
            # on the radial list and is never reused here.
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

    # Multi-k Ewald-exchange-split: per-(k,k′) q-channel tables for the
    # LR exchange (option (b) Phase 3). Shares the J^LR cache's Ewald w
    # and K_max envelope; the q == 0 diagonal channel reuses the J^LR
    # fold tensors outright.
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
            f"shared K_max = {ewald_k_max:.2f} bohr⁻¹ envelope"
        )

    def _split_k_density_list(density: LatticeMatrixSet) -> List[np.ndarray]:
        """Per-k density matrices for the Ewald-exchange-split paths.

        Exact for every density representation the SCF loop produces
        (orbital rebuilds, SAD/PATOM local guesses, caller warm-starts,
        damped and ODA-mixed densities): at Γ the BvK representative is
        the home-cell block; at multi-k the BvK-torus fold inverts the
        Bloch transform exactly (see ``bvk_torus_density_matrices``).
        """
        if n_k == 1:
            return [home_cell_block(density).astype(complex)]
        assert _bvk_mesh is not None
        return bvk_torus_density_matrices(density, k_points, _bvk_mesh)

    # Incremental/differential J_SR+K_SR accumulator (opt-in); see
    # ``resolve_incremental_jk`` for the gauge/ODA activation rationale.
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
    ) -> _PBCBipoleFockBuild:
        """Build F^2e(g) and F(k) for one real-space density.

        Thin wrapper over the shared restricted BIPOLE Fock builder
        (``pbc_bipole_fock.build_bipole_restricted_fock``, alpha_hf=1.0);
        the per-k Fock assembly (Bloch sum + Hcore + K_corr, then
        Hermitisation) stays here. See the builder for the gauge
        invariants and the K_corr derivation.

        ``coeffs_for_rho`` is supplied only when the density is exactly
        represented by the current per-k orbitals. ``use_incremental``:
        when False, force a full J_SR/K_SR build even if the incremental
        accumulator is active -- used for ODA's extra naive build and the
        post-convergence rebuild, which are off the per-iter ΔD chain.
        """
        fb = build_bipole_restricted_fock(
            _fock_ctx,
            density,
            coeffs_for_rho=coeffs_for_rho,
            alpha_hf=1.0,
            use_incremental=use_incremental,
            reseed_incremental=reseed_incremental,
        )
        f2e_real = fb.f2e_real
        K_corr_per_k = fb.k_corr_per_k

        f_k_list: List[np.ndarray] = []
        F2e_k_all = _bloch_sum_blocks_multi_k(
            f2e_real.blocks,
            f2e_real.cells,
            k_points,
        )
        for k_idx, k in enumerate(k_points):
            F_k = F2e_k_all[k_idx] + np.asarray(Hcore_k_list[k_idx], dtype=complex)
            if K_corr_per_k is not None:
                # Ewald exchange split: K_LR + G=0/Madelung pieces live
                # in k-space (one matrix per k; a single Γ entry at
                # n_k = 1).
                F_k = F_k - 0.5 * K_corr_per_k[k_idx]
            F_k = 0.5 * (F_k + F_k.conj().T)
            f_k_list.append(F_k)

        return _PBCBipoleFockBuild(
            f2e_real=f2e_real,
            f_k_list=f_k_list,
            e_j_short_range=fb.e_j_short_range,
            e_j_long_range=fb.e_j_long_range,
            e_exchange=fb.e_exchange,
            e_j_multipole=fb.e_j_multipole,
            e_2e_k_correction=fb.e_2e_k_correction,
            e_exchange_finite_size=fb.e_exchange_finite_size,
            e_j_bipolar_quartet=fb.e_j_bipolar_quartet,
            output_cell_farming_execution=(
                fb.output_cell_farming_execution
            ),
        )


    # ---- Retired multipole far-field config -----------------------------
    # Explicit requests fail before setup. Keep a disabled config here for
    # the shared Fock context until the research-only implementation is
    # removed from that internal API.
    from .bipole_fock_multipole import (  # noqa: E402
        BipoleMultipoleConfig,
        resolve_multipole_config,
    )

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

    # ---- SCF loop --------------------------------------------------------

    # SYM3b reduction is automatic for attached-symmetry crystals on the
    # corrected split; explicit enforcement/reduction choices are resolved
    # by the shared helper before the density-list decision.
    # ---- Shared restricted Fock-build context (M2 unification) ----------
    # Bundle the per-run invariants the inline Fock build used to capture,
    # so the heavy J^SR/J^LR/K assembly lives once in pbc_bipole_fock.
    # _build_fock_for_density (above) references this via late binding --
    # it is only ever called from the SCF loop below, after this point.
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
        output_cell_farming_task_kind=(
            output_cell_farming_task_kind if farm_output_cells else None
        ),
        output_cell_farming_strategy=output_cell_farming_strategy,
    )

    if _patom_seed_pending:
        patom_fock = _build_fock_for_density(
            D_real, coeffs_for_rho=None, use_incremental=False,
        )
        C_per_k, eps_per_k = [], []
        for F_k, X_k in zip(patom_fock.f_k_list, X_k_list):
            C_k, eps_k = _diag_in_orth_basis(F_k, X_k)
            C_per_k.append(C_k.astype(complex))
            eps_per_k.append(eps_k)
        D_real = _rebuild_real_space_density(C_per_k)
        if not exchange_split_active:
            _zero_cross_cell_density(D_real, basis.nbasis, n_k)
        D_real_prev = None
        initial_density_is_local = False
        density_from_c_per_k = True

    plog.banner("SCF (PBC BIPOLE, direct-space)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    # ---- DFT+U setup (closed-shell BIPOLE +U) ----------------------------
    # Same per-spin per-k convention as run_pbc_bipole_uhf -- for closed-
    # shell we use P_s = P_total/2 and the spin sum doubles E_s.
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
                    f"run_pbc_bipole_rhf: HubbardSite{key} has no AOs "
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
    ) -> float:
        """Apply +U energy/potential to the same density as the base Fock."""
        if not dft_plus_u_sites_cxx:
            return 0.0
        from ._vibeqc_core import _compute_dft_plus_u_multi_k_per_spin_cxx

        if exchange_split_active:
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

    scf_trace: List[SCFIteration] = []
    energy_components: List[PBCBipoleEnergyComponents] = []
    initial_density_energy_components: Optional[
        PBCBipoleEnergyComponents
    ] = None
    def _exact_total_energy(density, fb, e_dft_plus_u_value):
        """Total energy of ``density`` on the exact build ``fb`` (#116).

        Assembled exactly as the post-loop finalisation assembles the
        returned energy, so the in-loop confirmation and the terminal
        refresh judge the same number.
        """
        e_kin = _lattice_contract(density, T_lat, operator_name="T")
        e_ne = _lattice_contract(density, V_lat, operator_name="V_ne")
        e_2e = (
            0.5 * _lattice_contract(density, fb.f2e_real, operator_name="F2e")
            + fb.e_2e_k_correction
        )
        e_tot = float(e_kin + e_ne + e_2e) + e_nuc + float(e_dft_plus_u_value)
        if system.dim == 3 and not exchange_split_active:
            e_tot += compute_ext_el_spheropole(density, basis, system, lat_opts)
        return float(e_tot)

    E_prev = 0.0
    e_dft_plus_u = 0.0
    F_k_list: List[np.ndarray] = [np.zeros_like(H) for H in Hcore_k_list]
    F_k_prev_mixed: Optional[List[np.ndarray]] = None  # for fock_mixing
    E_elec = 0.0
    converged = False
    iter_idx = 0
    #: Exact rebuild from the convergence confirmation (IID 514); reused by
    #: the post-loop finalisation so healthy rows pay nothing extra.
    _confirm_fb: Optional[_PBCBipoleFockBuild] = None
    _confirm_e_dft_plus_u: Optional[float] = None

    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter
        E_j_short_range: Optional[float] = None
        E_j_long_range: Optional[float] = None
        E_exchange: Optional[float] = None
        E_j_multipole: Optional[float] = None

        # Damping (skip when DIIS active).
        D_used = D_real
        if iter_idx > 1 and damping > 0.0 and not diis_active:
            D_used = _damp_lattice_matrix(D_real, D_real_prev, damping)

        # --- F^{2e}(g) build.
        # Use the k-space r̂(K) route only when the real-space density
        # is exactly represented by C_per_k. Local SAD, fixed damping,
        # and ODA-mixed densities are real-space densities; for those,
        # J^LR must be built from the actual density blocks to avoid
        # using stale orbitals in the reciprocal-space piece.
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
        # (SYM3b Fock symmetrization happens inside _build_fock_for_density,
        # before the Bloch sum, so f2e_real and f_k_list are consistent.)
        F_k_list = fock_build.f_k_list
        E_j_short_range = fock_build.e_j_short_range
        E_j_long_range = fock_build.e_j_long_range
        E_exchange = fock_build.e_exchange
        E_j_multipole = fock_build.e_j_multipole

        # ---- DFT+U: per-spin per-k Fock contribution (closed-shell).
        # n_s = S_k w_k Re[(S(k) P_s(k) S(k))_(A,l)] with P_s = P_total/2;
        # V_AO_s = U_eff (1/2 - n_s); per-k Fock += S(k) V_AO_s S(k).
        # E_total_U = 2 x E_s (spin sum).
        e_dft_plus_u = _apply_dft_plus_u(D_used, F_k_list)

        # --- Per-cell electronic energy + [F,DS] error vectors.
        #
        # CRYSTAL's energy path contracts the real-space density against
        # real-space operator blocks: E = S_g D(g)H(g) + 1/2S_g D(g)F^2e(g).
        # This is essential at CYC0, where SAD is localised at g=0 and
        # Γ-folding T/V would incorrectly add cross-cell one-electron
        # blocks. k-space D(k) is still needed for error vectors,
        # level-shift projection, and the J^LR split path.
        E_kin = _lattice_contract(D_used, T_lat, operator_name="T")
        E_ne = _lattice_contract(D_used, V_lat, operator_name="V_ne")
        E_2e = (
            0.5
            * _lattice_contract(
                D_used,
                F2e_real,
                operator_name="F2e",
            )
            # k-space exchange correction (Ewald exchange split): the
            # K_LR + G=0/Madelung pieces live in F(k), not f2e_real.
            + fock_build.e_2e_k_correction
        )
        E_elec = E_kin + E_ne + E_2e
        grad_norm_sum = 0.0
        error_k_list: List[np.ndarray] = []
        D_k_list: List[np.ndarray] = []
        D_k_split_guess: Optional[List[np.ndarray]] = None
        if exchange_split_active and initial_density_is_local and iter_idx == 1:
            # Caller warm-starts may carry the full Bloch fold (D at
            # every cell) -- S_g over the cutoff list would overcount;
            # read the BvK representative instead (home block at Γ,
            # exact torus fold at multi-k).
            D_k_split_guess = _split_k_density_list(D_used)
        D_k_guess_fold: Optional[List[np.ndarray]] = None
        if (
            initial_density_is_local
            and iter_idx == 1
            and D_k_split_guess is None
        ):
            D_k_guess_fold = _bloch_sum_blocks_multi_k(
                D_used.blocks, D_used.cells, k_points
            )
        for idx in range(n_k):
            if initial_density_is_local and iter_idx == 1:
                # SAD/PATOM-style local guesses are stored explicitly as
                # D(g=0)=D_atom_sum and D(g!=0)=0. Their Bloch sum is the
                # same D at every k; using the Hcore-diag C(k) seed here
                # would make the energy/error vector inconsistent with
                # the Fock matrix that was just built from SAD.
                if D_k_split_guess is not None:
                    D_k = D_k_split_guess[idx]
                else:
                    assert D_k_guess_fold is not None
                    D_k = D_k_guess_fold[idx]
                D_k = 0.5 * (D_k + D_k.conj().T)
            else:
                # Multi-k (or legacy): D_k from previous iter's C.
                C_k = C_per_k[idx]
                C_occ = C_k[:, :n_occ]
                D_k = 2.0 * (C_occ @ C_occ.conj().T)
            D_k_list.append(D_k)
            H_k = Hcore_k_list[idx]
            F_k = F_k_list[idx]
            w = float(weights[idx])
            S_k = S_k_list[idx]
            FDS = F_k @ D_k @ S_k
            grad = FDS - FDS.conj().T
            error_k_list.append(grad)
            grad_norm_sum += w * float(np.linalg.norm(grad))

        E_total = float(E_elec) + e_nuc + e_dft_plus_u

        # EXT EL-SPHEROPOLE -- CRYSTAL's K=0 Ewald reciprocal-space
        # limit term, added to energy only (not the Fock matrix). It is a
        # 3D-Ewald-gauge correction and is identically zero in the direct
        # (non-Ewald) gauge used for dim<3, so it is absent there.
        # Under the Ewald exchange split it is omitted: at the corrected
        # gauge (full Bloch density + split exchange + v_bg.S) the total
        # already matches the reference assembly and the spheropole would
        # be a double-count (MgO Γ fixed-density audit, 2026-06-10).
        if system.dim == 3 and not exchange_split_active:
            E_sphero = compute_ext_el_spheropole(D_used, basis, system, lat_opts)
            E_total += E_sphero
        else:
            E_sphero = None

        dE = E_total - E_prev if iter_idx > 1 else 0.0

        check_scf_divergence(
            "run_pbc_bipole_rhf",
            iter_idx,
            E_total,
            grad_norm_sum,
            dE,
        )
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm_sum),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm_sum),
            diis=(accel.subspace_size if accel is not None else 0),
        )
        cycle_components = PBCBipoleEnergyComponents(
            iter=int(iter_idx),
            e_total=float(E_total),
            e_electronic=float(E_elec),
            e_kinetic=float(E_kin),
            e_nuclear_attraction=float(E_ne),
            e_two_electron=float(E_2e),
            e_nuclear_repulsion=float(e_nuc),
            e_bielet_zone_ee=(None if use_ewald_j_split else float(E_2e)),
            e_ext_el_spheropole=E_sphero,
            e_j_short_range=E_j_short_range,
            e_j_long_range=E_j_long_range,
            e_exchange=E_exchange,
            e_exchange_finite_size=fock_build.e_exchange_finite_size,
            e_j_multipole=E_j_multipole,
            e_dft_plus_u=float(e_dft_plus_u),
        )
        energy_components.append(cycle_components)
        if initial_density_energy_components is None:
            initial_density_energy_components = cycle_components
        plog.energy_decomposition(
            iter_idx,
            E_kin=float(E_kin),
            E_ne=float(E_ne),
            E_2e=float(E_2e),
            E_elec=float(E_elec),
            E_nuc=float(e_nuc),
        )

        # ---- Multipole far-field diagnostics (if enabled) -----------
        if use_multipole_diag and system.dim == 3 and not exchange_split_active:
            from .bipole_fock_multipole import (
                build_j_far_field_multipole,
                estimate_bipole_radius,
            )

            try:
                R_bipole = estimate_bipole_radius(
                    system,
                    basis,
                    L_max=multipole_l_max,
                )
                far_j = build_j_far_field_multipole(
                    D_used,
                    basis,
                    system,
                    lat_opts_2e,
                    L_max=multipole_l_max,
                    R_bipole=R_bipole,
                    cache=_mp_config.cache if _mp_config.enabled else None,
                )
                plog.info(
                    f"  BIPOLE far-field (L_max={multipole_l_max}, "
                    f"R_bipole={R_bipole:.1f} bohr): "
                    f"E_J_far = {far_j.e_j_far:+.6f} Ha, "
                    f"n_pairs = {far_j.n_cell_pairs}"
                )
            except Exception as exc:
                plog.info(
                    f"  BIPOLE far-field diagnostic failed: {type(exc).__name__}: {exc}"
                )

        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm_sum < float(opts.conv_tol_grad)
        )

        # --- SCF-accelerator extrapolation. The full
        # {DIIS, KDIIS, EDIIS, EDIIS_DIIS, ADIIS} family + dynamic_damping
        # is wired on the multi-k BIPOLE path: DIIS / KDIIS run natively
        # per-k (Pulay / orbital-rotation-gradient designs from M2c);
        # EDIIS / ADIIS / EDIIS_DIIS bridge through the stacked-real-block
        # representation landed in M2e (see
        # ``per_k_to_stacked_real_blocks`` in
        # ``periodic_scf_accelerators.py``).
        if accel is not None:
            if exchange_split_active:
                # Unprojected Bloch fold: S_g overcounts (see the +U
                # fold above) -- BvK representative per k instead.
                density_k_list = _split_k_density_list(D_used)
            else:
                density_k_list = _bloch_sum_blocks_multi_k(
                    D_used.blocks, D_used.cells, k_points
                )
            F_ex_list = accel.extrapolate_rhf(
                F_k_list,
                error_k_list=error_k_list,
                density_k_list=density_k_list,
                energy=E_total,
                mo_coeffs_k_list=C_per_k,
                n_occ=n_occ,
                weights=list(weights),
                cells=cells,
                kpoints=list(k_points),
            )
            # On the converged iteration, diagonalise the *physical* Fock
            # F(D_used) -- not the extrapolated one. At a true fixed point
            # F(D) commutes with D, so diagonalising the bare Fock
            # reproduces the converged density exactly and yields canonical
            # orbitals. Diagonalising an extrapolated Fock here can move the
            # solution off the fixed point: when the SCF lands essentially
            # on the solution in one step, the DIIS error history collapses
            # to machine-zero, the Pulay B-matrix goes singular, and the
            # degenerate solve returns large ±coefficients whose Fock
            # combination is numerical garbage. Its Aufbau diagonalisation
            # can occupy the wrong orbital -- the H₂/STO-3G RHF + [2,1,1]
            # spurious basins at bz≈1.449 (-0.347 Ha) and bz≈1.399
            # (-0.528 Ha); smooth -1.728 elsewhere; PySCF KRHF confirms a
            # single smooth solution. See
            # tests/test_pbc_bipole_diis_converged_basin.py.
            if diis_active and not converged:
                F_k_list = F_ex_list

        # --- FMIXING (CRYSTAL-style, after DIIS, before level-shift)
        # Skipped on the converged iteration (see DIIS note above): the
        # final diagonalisation must see the physical converged Fock.
        if fock_mixing_value != 0.0 and not converged:
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

        # --- LEVSHIFT (per-iter schedule or static)
        # Skipped on the converged iteration (see DIIS note above).
        if level_shift_schedule is not None:
            level_shift_b = level_shift_schedule.at(iter_idx)
        else:
            level_shift_b = level_shift_static
        if level_shift_b != 0.0 and not converged:
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

        # --- Diagonalise F(k) -> new C(k), e(k)
        new_C_per_k = []
        new_eps_per_k = []
        for idx in range(n_k):
            C_k, eps_k = _diag_in_orth_basis(F_for_diag[idx], X_k_list[idx])
            new_C_per_k.append(C_k)
            new_eps_per_k.append(eps_k)

        # --- MOM reorder (iter >= 2 only; falls through to Aufbau at iter 1)
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

        # --- Rebuild D_real.  At Γ-only, real_space_density_from_kpoints
        # produces P(g)=P(Γ) at every cell -- the correct Bloch fold. With
        # the Ewald exchange split that is exactly the density the
        # builders need (the erfc-screened K_SR sum is absolutely
        # convergent with non-decaying P). On the legacy path the
        # Γ-locality projection P(g!=0)=0 is kept: its full-Coulomb K
        # series would diverge with the unprojected fold.
        D_real_new = _rebuild_real_space_density(C_per_k)
        if not exchange_split_active:
            _zero_cross_cell_density(D_real_new, basis.nbasis, n_k)

        # Energy and commutator stationarity are evaluated on D_used. Before
        # accepting that provisional decision, require the just-diagonalised
        # density to represent the same fixed point. Otherwise the post-loop
        # rebuild can invalidate convergence after the loop has stopped.
        if converged:
            density_fixed_point_residual = max(
                (
                    float(np.max(np.abs(np.asarray(new) - np.asarray(old))))
                    for new, old in zip(D_real_new.blocks, D_used.blocks)
                ),
                default=0.0,
            )
            converged = density_fixed_point_residual < float(opts.conv_tol_grad)

        # --- ODA mixing (extra Fock build)
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
            plog.info(
                f"  ODA: l = {oda_step.lam:.4f} "
                f"(g0 = {oda_step.g0:+.3e}, g1 = {oda_step.g1:+.3e})"
            )
        else:
            D_real_prev = D_used
            D_real = D_real_new
            density_from_c_per_k = True

        # Snapshot for next iter's MOM
        if use_mom:
            C_prev_occ_per_k = [
                np.asarray(C_per_k[idx][:, :n_occ]).copy() for idx in range(n_k)
            ]

        if damper is not None:
            damper.update(E_total)
        E_prev = E_total
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
            # #116: the exact operator's ENERGY at the committed density is
            # judged here too -- the same delta the post-loop refresh writes
            # into the terminal trace row -- so the loop cannot exit on a
            # state the terminal check would refuse.
            _confirm_ok, _ = bipole_terminal_check(
                plog,
                phase="in_loop",
                iter_idx=iter_idx,
                loop_objective=float(scf_trace[-1].energy),
                exact_objective=_exact_total_energy(
                    D_real, _confirm_fb, _confirm_e_dft_plus_u
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

    # ---- Post-loop: evaluate the exact density returned to callers.  This
    # is mandatory even on max-iteration exhaustion: the loop diagonalises
    # and commits a new density after evaluating the prior one, so returning
    # the old Fock/energy would mix two distinct SCF states.
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
    E_kin_final = _lattice_contract(D_real, T_lat, operator_name="T")
    E_ne_final = _lattice_contract(D_real, V_lat, operator_name="V_ne")
    E_2e_final = (
        0.5
        * _lattice_contract(
            D_real,
            _fb.f2e_real,
            operator_name="F2e",
        )
        + _fb.e_2e_k_correction
    )
    E_elec = E_kin_final + E_ne_final + E_2e_final
    E_total = float(E_elec) + e_nuc + e_dft_plus_u
    # Fresh E_total doesn't include spheropole -- add it (3D only; the
    # term is zero in the direct gauge used for dim<3, and omitted
    # under the Ewald exchange split -- see the SCF-loop note).
    if system.dim == 3 and not exchange_split_active:
        E_sphero_final = compute_ext_el_spheropole(D_real, basis, system, lat_opts)
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
    _loop_objective = float(scf_trace[-1].energy) if scf_trace else float(E_total)
    _loop_grad = float(scf_trace[-1].grad_norm) if scf_trace else float(final_grad_norm)
    refresh_bipole_terminal_trace(
        scf_trace,
        float(E_total),
        grad_norm=final_grad_norm,
    )
    _terminal_ok, _ = bipole_terminal_check(
        plog,
        phase="post_loop",
        iter_idx=iter_idx,
        loop_objective=_loop_objective,
        exact_objective=float(E_total),
        loop_grad_norm=_loop_grad,
        exact_grad_norm=float(final_grad_norm),
        conv_tol_energy=float(opts.conv_tol_energy),
        conv_tol_grad=float(opts.conv_tol_grad),
    )
    if converged:
        converged = _terminal_ok

    plog.converged(n_iter=iter_idx, energy=E_total, converged=converged)

    return PBCBipoleRHFResult(
               restart_mesh=tuple(kmesh.mesh),
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density is not None or initial_density_k is not None),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=float(E_total),
        e_electronic=float(E_elec),
        e_nuclear=e_nuc,
        n_iter=iter_idx,
        converged=converged,
        mo_energies=eps_per_k,
        mo_coeffs=C_per_k,
        fock=F_k_list,
        overlap=S_k_list,
        hcore=Hcore_k_list,
        density=D_real,
        e_ext_el_spheropole=E_sphero_final,
        scf_trace=scf_trace,
        ewald_alpha_bohr_inv=omega_used,
        ewald_precision=float(ewald_precision),
        sr_image_extent_bohr=_sr_image_extent,
        pair_resolved_fock_domain=bool(_fock_sym_map is not None),
        e_dft_plus_u=float(e_dft_plus_u),
        energy_components=energy_components,
        exchange_ewald_split=bool(exchange_split_active),
        exchange_exxdiv=(exchange_exxdiv if exchange_split_active else None),
        overlap_fold_drift=overlap_fold_drift,
        exact_zone_bohr=_exact_zone,
        output_cell_farming_execution=(
            _fb.output_cell_farming_execution
        ),
        fock_mixing=fock_mixing_value,
        kpoints_cart=np.asarray(k_points, dtype=float).reshape(-1, 3),
        kpoint_weights=np.asarray(weights, dtype=float).reshape(-1),
        n_electrons=int(n_elec),
        occupations=[
            np.concatenate(
                [np.full(n_occ, 2.0), np.zeros(len(eps) - n_occ)]
            )
            for eps in eps_per_k
        ],
        initial_density_energy_components=(
            initial_density_energy_components
        ),
    )
