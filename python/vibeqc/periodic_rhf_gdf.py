"""Gamma-only periodic RHF/RKS with native Gaussian density fitting.

This is the first self-hosted GDF SCF driver in vibe-qc's periodic
stack. It uses the native C++ periodic 2c/3c auxiliary-integral kernels
through :func:`vibeqc.aux_basis.build_lpq_native`, then contracts the
resulting ``Lpq`` tensor in Python to build J and K.

Scope
-----

* \u0393-only RHF and closed-shell RKS.
* Native vibe-qc integrals only.
* Multi-k GDF is available via :func:`vibeqc.run_krhf_periodic_gdf`.

Algorithm selection
-------------------

* ``gdf_algorithm='bare'`` (default): direct image-summed 2c/3c ERIs
  with modrho rescaling. Works for 1D/2D/3D molecular-limit cells.
  For 3D cells where the GDF cutoff includes image cells, or when an
  internal finite-torus caller requires the periodic Hamiltonian, the
  driver falls back to EWALD_3D J + real-space K.
* ``gdf_algorithm='rsgdf'``: range-separated GDF (Ye & Berkelbach 2021,
  DOI 10.1063/5.0046617). Split 1/r = erfc(\u03c9r)/r + erf(\u03c9r)/r; SR
  in real space, LR in reciprocal space. Works for 1D/2D systems;
  on 3D systems the sparse G-mesh (~4 bohr\u207b\u00b9) cannot resolve
  compact aux primitives, producing \u223c231 mHa over-binding. A
  runtime warning is emitted for 3D RSGDF.

For production periodic SCF on 3D ionic crystals (LiH, MgO, etc.),
prefer :func:`vibeqc.run_krhf_periodic_gdf` with ``use_compcell=True``
(multi-k compcell GDF, validated at \u00b5Ha parity) or
:func:`vibeqc.run_pbc_gdf_rhf` (\u0393-only compcell with AFT correction
on by default).
"""

from __future__ import annotations

from .guess import periodic_guess_capabilities

from .guess import periodic_result_selection

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    apply_level_shift,
    BasisSet,
    bloch_sum,
    build_grid,
    build_jk_gamma_molecular_limit,
    build_xc_periodic,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    CoulombMethod,
    direct_lattice_cells,
    Functional,
    GridOptions,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    level_shift_at_iter,
    LevelShiftDensity,
    ewald_nuclear_repulsion,
    nuclear_repulsion_per_cell,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    SCFAccelerator,
    SCFIteration,
)
from ._vibeqc_core import monkhorst_pack as _mp_native
from .aux_basis import (
    _build_lpq_bloch_slab_truncated,
    _contract_slab_gdf_gamma,
    _contract_slab_gdf_multik_exchange,
    _slab_probe_charge_madelung_for_kmesh,
    build_lpq_native,
    default_aux_for,
    make_aux_basis_set,
    make_modrho_aux_basis,
)
from .ewald_composed import make_ewald_3d_gamma_j_builder
from .ewald_j import auto_grid
from .guess import initial_density_closed_shell
from .linear_dependence import scf_preflight_overlap_check
from .occupations import (
    aufbau_occupations_per_k as _aufbau_occupations_per_k,
)
from .occupations import (
    fermi_dirac_occupations_per_k as _fermi_dirac_occupations_per_k,
)
from .occupations import (
    hartree_to_kelvin_temperature as _hartree_to_kelvin_temperature,
)
from .smearing.apply import _global_aufbau_with_mu
from .options_dump import dump_active_settings
from .periodic_grid import build_periodic_becke_grid
from .periodic_k_density import real_space_density_from_per_k_density
from .periodic_rhf_ewald import (
    _canonical_orthogonalizer,
)
from .periodic_rhf_multi_k_ewald import _canonical_orthogonalizer_complex
from .periodic_scf_accelerators import (
    DynamicDamping,
    MultiKPeriodicSCFAccelerator,
    PeriodicSCFAccelerator,
)
from .periodic_screened_exchange import reject_unscreened_range_separated
from .progress import ProgressLogger, resolve_progress
from .symmetry_integrals_reduced import (
    compute_kinetic_lattice_reduced,
    compute_overlap_lattice_reduced,
)

__all__ = [
    "PeriodicRHFGDFResult",
    "run_rhf_periodic_gamma_gdf",
    "run_rks_periodic_gamma_gdf",
]


@dataclass
class PeriodicRHFGDFResult:
    """Result of :func:`run_rhf_periodic_gamma_gdf`.

    The ``backend`` field indicates the actual J/K mechanism:
      - ``"native-gamma-gdf"`` -- true density fitting via Lpq
        (1D/2D systems; 3D molecular-limit cells whose GDF cutoff
        includes only the home cell).
      - ``"ewald-jk-fallback"`` -- Ewald-3D J + real-space K
        (3D systems with image cells in the GDF cutoff, or an internal
        finite-torus request; see :mod:`vibeqc.pbc_gdf` for the
        compensated-cell GDF path)."""

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    scf_trace: List[SCFIteration] = field(default_factory=list)
    hcore: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    aux_basis_name: str = ""
    n_aux: int = 0
    n_fit: int = 0
    linear_dep_threshold: float = 1e-9
    gdf_algorithm: str = "bare"
    e_xc: float = 0.0
    e_coulomb: float = 0.0
    e_hf_exchange: float = 0.0
    functional: str = ""
    fock_mixing: float = 0.0
    level_shift: float = 0.0
    level_shift_warmup_cycles: int = 0
    smearing_temperature: float = 0.0
    fermi_level: float = 0.0
    entropy: float = 0.0
    free_energy: float = 0.0
    occupations: np.ndarray = field(default_factory=lambda: np.empty(0))
    backend: str = "native-gamma-gdf"

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


@dataclass
class _PrivateSlabKRHFGDFResult:
    """Internal result for the fail-closed multi-k slab-GDF validation route."""

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    mo_energies: List[np.ndarray]
    mo_coeffs: List[np.ndarray]
    density: List[np.ndarray]
    fock: List[np.ndarray]
    overlap: List[np.ndarray]
    hcore: List[np.ndarray]
    kpoints_cart: np.ndarray
    kpoint_weights: np.ndarray
    scf_trace: List[SCFIteration]
    functional: Optional[str] = None
    e_xc: float = 0.0
    e_coulomb: float = 0.0
    e_hf_exchange: float = 0.0
    aux_basis_name: str = ""
    n_aux: int = 0
    backend: str = "private-multik-slab-truncated-gdf"

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


# --- Ewald-J build parameters (Step 2 of the 2026-05-13 gauge fix) ---
# Defaults match `periodic_rks_ewald.py` (`omega = 0.5 / bohr`, FFT
# Poisson spacing `0.3 bohr`); the composed Ewald-3D J is w-invariant
# up to numerical precision (validated in `validate_ewald_identity`), so
# tuning matters only for cost.  Exposed here as constants so the choice
# is discoverable; can be promoted to function-level kwargs in a follow-up
# if user-tuning becomes needed.
_J_EWALD_OMEGA: float = 0.0
_J_EWALD_SPACING_BOHR: float = 0.3


def _check_energy_sanity(
    result: PeriodicRHFGDFResult,
    system: PeriodicSystem,
    plog: ProgressLogger,
) -> None:
    """Post-convergence sanity check: warn if the total energy is
    physically unreasonable for the given system.

    The bound is a loose hydrogenic upper limit (Z_sum^2 per cell).
    Values exceeding this are almost certainly from a gauge mismatch,
    missing Madelung correction, or divergent lattice sum -- not a
    physical SCF minimum.  This is an informational warning only;
    the caller decides whether to treat the result as usable."""
    z_sum_sq = sum(atom.Z**2 for atom in system.unit_cell)
    sane_bound = max(10.0 * z_sum_sq, 100.0)
    if abs(result.energy) > sane_bound:
        plog.warn(
            f"Total energy {result.energy:.6f} Ha is outside the "
            f"physically-reasonable range [{sane_bound:.0f} Ha] for "
            f"this cell (sum Z^2 = {z_sum_sq:.0f}). This usually "
            f"indicates a gauge mismatch, a missing Madelung/exxdiv "
            f"correction, or a divergent lattice sum. The result "
            f"may be meaningless -- consider a different JK backend "
            f"(e.g. run_pbc_gdf_rhf for 3D cells)."
        )


def _gauge_lat_opts_for_v_ne_and_e_nuc(
    src: LatticeSumOptions,
    system: PeriodicSystem,
) -> LatticeSumOptions:
    """Return a clone of ``src`` with the Coulomb gauge forced to Ewald-3D
    for the V_ne lattice sum and the nuclear-nuclear lattice sum, when
    the system is fully 3D-periodic.

    Rationale (gauge-regression fix, 2026-05-13):
    The native gamma GDF driver's three Coulomb-bearing pieces
    (V_ne, e_nuc, J / K via Lpq) must share a single gauge for the
    total energy to be physically meaningful.  PySCF
    (``exxdiv='ewald'``), CRYSTAL14, and vibe-qc's own ``EWALD_3D``
    direct path all use the Madelung/Ewald gauge -- that's the
    convention the now-sealed v0.8.0 STO-3G parity baseline
    (``examples/regression/crystal_parity/baseline_sto3g/PARITY_TABLE.md``)
    is calibrated against.

    The user-supplied ``lat_opts.coulomb_method`` controls **J/K**
    routing (DIRECT_TRUNCATED vs EWALD_3D vs DF backends), not
    one-electron / nuclear gauge -- those need to be Ewald-3D
    unconditionally on 3D-periodic systems.

    For ``dim < 3`` there is **no gauge to return**, so this fails closed
    (2026-07-10). Every periodic Coulomb kernel is defined only up to the
    conditional ``G+q=0`` constant ``c``, and a neutral cell's total energy is
    physical only when all three Coulomb channels share one ``c``::

        E_nn        ->  +½ c Z²
        Tr[D·V_ne]  ->  -  c Z N_e
        ½Tr[D·J]    ->  +½ c N_e²
        ------------------------------------------------
        sum         =  ½ c (N_e - Z)²  =  0     (neutral cell)

    This function used to pass ``src`` through unchanged for ``dim < 3``, handing
    back **bare** lattice sums for ``V_ne``/``E_nn`` while ``J`` came from the
    ``G+q=0``-dropped neutral cderi. Bare 1-D/2-D Coulomb sums are only
    conditionally convergent, so their ``c`` grows without bound with the
    lattice-sum cutoff and the uncancelled ``-½cZ²`` diverges: polyethylene
    (dim=1, sto-3g, (4,1,1)) ran ``E_total = -1226.78 -> -1368.41 -> -1518.55`` Ha
    as ``nuclear_cutoff_bohr`` went ``45 -> 90 -> 180``.

    Matching the two gauges would not have rescued it. For ``dim < 3`` the neutral
    cderi is not a Coulomb kernel either: its G-mesh pins every non-periodic axis
    at ``G_perp = 0`` (:func:`~vibeqc.aux_basis.rsgdf_dense_g_mesh`), so ``J`` is a
    transverse-uniform sheet term ``∝ 1/V`` that vanishes with the vacuum padding.
    A cutoff-independent total built on it would still be meaningless.

    The gauge-consistent low-D Hamiltonians live outside this 3-D-torus dispatch:
    ``dim == 1`` -> the mixed-boundary wire kernel
    (:func:`~vibeqc.periodic.ccm.lowd_scf.run_ccm_rhf_wire`); ``dim == 2`` -> the
    four-center route (bare ``1/r`` minimum image places all channels in one gauge)
    pending the slab partial-FT build.

    See the retired periodic-SCF gauge-regression forensic note in git
    history, Sec. "Fix path -- three steps" / Step 1.
    """
    dim = int(system.dim)
    if dim != 3:
        raise NotImplementedError(
            f"_gauge_lat_opts_for_v_ne_and_e_nuc: no consistent Coulomb gauge for a "
            f"dim = {dim} cell. The bare V_ne/E_nn lattice sums are conditionally "
            "convergent (the total diverges with nuclear_cutoff_bohr) and the neutral "
            "cderi they would pair with collapses to a transverse-uniform sheet term. "
            "Use run_ccm_rhf_wire for dim == 1; dim == 2 needs the slab mixed-boundary "
            "kernel (docs/aiccm2026dev_a_lowd_greens.md section 9) and is not "
            "implemented -- the four-center route is unaffected and remains available."
        )
    dst = LatticeSumOptions()
    for attr in dir(src):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(src, attr)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            setattr(dst, attr, value)
        except Exception:
            pass
    dst.coulomb_method = CoulombMethod.EWALD_3D
    return dst


def _lat_opts_with_nuclear_cutoff(
    src: LatticeSumOptions,
    nuclear_cutoff_bohr: float,
) -> LatticeSumOptions:
    """Return a clone of ``src`` with ``nuclear_cutoff_bohr`` overridden.

    Used by the Lpq-J/K (non-Ewald) path to clamp the V_ne / e_nuc
    nuclear lattice sum to the GDF integral cutoff, so all three
    Coulomb-bearing pieces (V_ne, e_nuc, J/K) sum over a single cell
    set (audit F3, 2026-05-31). See the call site for rationale.
    """
    dst = LatticeSumOptions()
    for attr in dir(src):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(src, attr)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            setattr(dst, attr, value)
        except Exception:
            pass
    dst.nuclear_cutoff_bohr = float(nuclear_cutoff_bohr)
    return dst


def _build_j_from_lpq(Lpq: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Closed-shell Coulomb matrix from a fitted periodic ERI factor."""
    rho = np.einsum("Lij,ij->L", Lpq, D, optimize=True)
    J = np.einsum("L,Lij->ij", rho, Lpq, optimize=True)
    return 0.5 * (J + J.T)


def _build_k_from_lpq(Lpq: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Closed-shell exchange matrix from a fitted periodic ERI factor.

    ``D`` is the total RHF density, including the factor of two for
    doubly occupied orbitals. The RHF Fock then uses ``J - 0.5 K``.
    """
    K = np.einsum("Lmk,kl,Lnl->mn", Lpq, D, Lpq, optimize=True)
    return 0.5 * (K + K.T)


def _density_set_gamma(
    template: LatticeMatrixSet,
    D: np.ndarray,
) -> LatticeMatrixSet:
    """Return a Γ-only density set with ``D`` in the home-cell block.

    ``build_xc_periodic`` treats a home-block-only set as a
    *molecular-limit* density (``ρ = χ_0 D χ_0``, no image-cell AO
    products -- see ``density_has_cross_cell_blocks`` in
    ``cpp/src/periodic_xc.cpp``). That pairs with the **molecular**
    Becke grid (``use_periodic_becke=False``, the v0.8.x convention);
    on the default periodic-Becke grid use
    :func:`_density_set_torus_gamma` instead -- the molecular-limit
    density on the one-cell periodic grid drops every image
    contribution to ρ(r) and mis-evaluates E_xc/V_xc by ~0.3 Ha on
    tight ionic cells (the 2026-07-09 KRKS finding,
    handovers/HANDOVER_AICCM_DIRECT_TORUS.md §4).
    """
    if D.shape[0] != template.nbf:
        raise ValueError(
            f"_density_set_gamma: D has nbf={D.shape[0]} but template "
            f"has nbf={template.nbf}"
        )
    zero = np.zeros_like(D)
    for i in range(len(template)):
        template.set_block(i, D if i == 0 else zero)
    return template


def _density_set_torus_gamma(
    template: LatticeMatrixSet,
    D: np.ndarray,
) -> LatticeMatrixSet:
    """Return the Γ-torus density set: ``D`` in **every** lattice block.

    The inverse Bloch fold of a single Γ point is cell-independent --
    ``D(R) = D_Γ`` for every direct-lattice cell ``R`` -- so the physical
    periodic density is the cross-cell sum
    ``ρ(r) = Σ_{a,s} χ_a(r) D χ_s(r) = |Σ_R χ(r-R)|-contracted``, which is
    what ``build_xc_periodic``'s cross-cell mode evaluates when the
    ``g != 0`` blocks are populated. This is the same convention the
    multi-k drivers feed via ``real_space_density_from_per_k_density``
    (their ``N_k = 1`` limit is exactly this set) and the convention the
    external references (PySCF KRKS) use at Γ.
    """
    if D.shape[0] != template.nbf:
        raise ValueError(
            f"_density_set_torus_gamma: D has nbf={D.shape[0]} but template "
            f"has nbf={template.nbf}"
        )
    for i in range(len(template)):
        template.set_block(i, D)
    return template


def _resolve_fock_mixing(options, override: Optional[float]) -> float:
    """Return previous-Fock weight for CRYSTAL-style FMIXING."""
    value = override
    if value is None:
        value = getattr(options, "fock_mixing", None)
    if value is None:
        percent = getattr(options, "fmixing_percent", None)
        if percent is not None:
            value = float(percent) / 100.0
    if value is None:
        value = 0.0
    value = float(value)
    if not (0.0 <= value < 1.0):
        raise ValueError(
            f"run_rhf_periodic_gamma_gdf: fock_mixing must be in [0, 1); got {value}"
        )
    return value


def _resolve_level_shift_warmup_cycles(
    options,
    *,
    level_shift: float,
    max_iter: int,
    override: Optional[int] = None,
) -> int:
    """Return the number of shifted startup cycles.

    ``None`` or ``-1`` means "auto": if a nonzero level shift is active,
    use up to five shifted startup cycles while leaving at least one
    final unshifted tail cycle. ``0`` keeps the historic persistent
    level-shift behaviour. Positive integers request an explicit warm-up
    length and are capped to preserve the unshifted tail cycle.
    """
    max_iter = int(max_iter)
    if abs(float(level_shift)) == 0.0 or max_iter <= 1:
        return 0

    raw = override
    if raw is None:
        raw = getattr(options, "level_shift_warmup_cycles", None)
    if raw is None:
        raw = -1
    raw = int(raw)
    if raw < -1:
        raise ValueError(
            "run_rhf_periodic_gamma_gdf: level_shift_warmup_cycles must "
            f"be >= 0 or -1 for auto; got {raw}"
        )
    if raw < 0:
        raw = 5
    if raw == 0:
        return 0
    return min(raw, max_iter - 1)


def _density_from_orbitals_and_occupations(
    C: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    """One-particle density for closed-shell occupations in ``[0, 2]``."""
    D = (C * np.asarray(occupations, dtype=float)[None, :]) @ C.T
    return 0.5 * (D + D.T)


def run_rhf_periodic_gamma_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
    *,
    functional: Optional[str] = None,
    aux_basis: Optional[str] = None,
    aux_drop_eta: float = 0.0,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    apply_modrho: bool = True,
    gdf_algorithm: str = "bare",
    rsgdf_omega: float = 0.4,
    fock_mixing: Optional[float] = None,
    level_shift_warmup_cycles: Optional[int] = None,
    symmetry_stabilize: bool = False,
    symmetry_reduce_fock: bool = False,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    _force_ewald_jk: bool = False,
    _slab_truncated_private: bool = False,
    _slab_ke_cutoff: float = 200.0,
) -> PeriodicRHFGDFResult:
    """Run Γ-only closed-shell periodic RHF/RKS via native GDF.

    Parameters mirror the older target API, but the implementation is
    now entirely vibe-qc owned. If ``functional`` is provided, the
    driver adds native libxc ``V_xc`` on the periodic/molecular Becke
    grid and uses the functional's exact-exchange fraction for hybrids.

    ``_force_ewald_jk`` is a private routing seam for callers whose finite
    cell is known to represent a fully periodic torus.  Such callers must
    retain the Ewald-J/real-space-K Hamiltonian even when the numerical
    lattice cutoff happens to include only the home cell.
    """
    from .pbc_gdf import _refuse_ecp_options

    _refuse_ecp_options(options, "run_rhf_periodic_gamma_gdf", system=system)
    opts = options if options is not None else PeriodicRHFOptions()
    from .guess import select_initial_guess, periodic_fock_guess_k
    guess = select_initial_guess(
        system.unit_cell_molecule(), opts.initial_guess, is_periodic=True,
        supported=periodic_guess_capabilities('gdf', 'RHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
        atomic_spins=getattr(opts, "atomic_spins", None),
        driver="run_rhf_periodic_gamma_gdf",
    ).effective
    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)

    func_name = functional or str(getattr(opts, "functional", "") or "")
    is_ks = bool(func_name)
    force_ewald_jk = bool(_force_ewald_jk)
    if force_ewald_jk and int(system.dim) != 3:
        raise ValueError(
            "run_rhf_periodic_gamma_gdf: _force_ewald_jk requires a "
            "fully 3D-periodic system"
        )
    use_private_slab_gdf = bool(_slab_truncated_private)
    if use_private_slab_gdf:
        if int(system.dim) != 2:
            raise ValueError(
                "private slab-truncated GDF requires a genuine dim=2 system"
            )
        if is_ks:
            raise NotImplementedError(
                "private slab-truncated GDF currently validates RHF only"
            )
        if lat_opts.coulomb_method != CoulombMethod.SLAB_EWALD_2D:
            raise ValueError(
                "private slab-truncated GDF requires "
                "coulomb_method=SLAB_EWALD_2D for V_ne and E_nuclear"
            )
    func = Functional(func_name, 1) if is_ks else None
    # This legacy Γ fallback builds the molecular-limit K full-range
    # only; screened hybrids must not silently run as full-range twins.
    reject_unscreened_range_separated(
        func, where="run_rhf_periodic_gamma_gdf"
    )
    alpha = float(func.hf_exchange_fraction) if func is not None else 1.0
    fock_mixing_value = _resolve_fock_mixing(opts, fock_mixing)
    level_shift = float(getattr(opts, "level_shift", 0.0))
    max_iter = int(opts.max_iter)
    warmup_cycles = _resolve_level_shift_warmup_cycles(
        opts,
        level_shift=level_shift,
        max_iter=max_iter,
        override=level_shift_warmup_cycles,
    )
    # Explicit per-iteration schedule (unified with the molecular
    # drivers). Empty ⇒ the warm-up step function below; non-empty ⇒
    # resolved per iteration by the shared C++ helper.
    _ls_schedule = list(getattr(opts, "level_shift_schedule", None) or [])
    smearing_T = float(getattr(opts, "smearing_temperature", 0.0))
    if smearing_T < 0.0:
        raise ValueError(
            "run_rhf_periodic_gamma_gdf: smearing_temperature must be >= 0"
        )

    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "run_rhf_periodic_gamma_gdf: closed-shell RHF/RKS requires even "
            f"electron count; got {n_elec}"
        )
    if system.multiplicity != 1:
        raise ValueError(
            "run_rhf_periodic_gamma_gdf: closed-shell RHF/RKS requires "
            f"multiplicity=1; got {system.multiplicity}"
        )
    n_occ = n_elec // 2

    aux_name = aux_basis or default_aux_for(basis.name)
    label = f"RKS {func_name}" if is_ks else "RHF"

    plog.info(
        f"{label} Gamma native GDF / "
        f"aux={aux_name}, algorithm={gdf_algorithm}, "
        f"cutoff={lat_opts.cutoff_bohr:.2f} bohr"
    )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    dim = int(system.dim)
    active_lengths = [
        float(np.linalg.norm(np.asarray(system.lattice, dtype=float)[:, i]))
        for i in range(dim)
    ]
    plog.info(
        "periodicity: "
        f"dim={dim}D, active lengths="
        + ", ".join(f"{x:.3f}" for x in active_lengths)
        + " bohr"
    )
    n_int_cells = len(direct_lattice_cells(system, lat_opts.cutoff_bohr))
    n_nuc_cells = len(direct_lattice_cells(system, lat_opts.nuclear_cutoff_bohr))
    plog.info(
        "lattice cells: "
        f"one-electron/GDF cutoff -> {n_int_cells}, "
        f"nuclear cutoff -> {n_nuc_cells}"
    )
    dump_active_settings(
        plog,
        [
            ("PeriodicKSOptions" if is_ks else "PeriodicRHFOptions", opts),
            ("LatticeSumOptions", lat_opts),
            (
                "GDF kwargs",
                {
                    "functional": func_name or None,
                    "hf_exchange_fraction": alpha,
                    "fock_mixing": fock_mixing_value,
                    "fmixing_percent": 100.0 * fock_mixing_value,
                    "level_shift": level_shift,
                    "level_shift_warmup_cycles": warmup_cycles,
                    "smearing_temperature": smearing_T,
                    "aux_basis": aux_name,
                    "aux_drop_eta": float(aux_drop_eta),
                    "linear_dep_threshold": float(linear_dep_threshold),
                    "gdf_linear_dep_threshold": float(gdf_linear_dep_threshold),
                    "apply_modrho": bool(apply_modrho),
                    "gdf_algorithm": gdf_algorithm,
                    "rsgdf_omega": float(rsgdf_omega),
                    "force_ewald_jk": force_ewald_jk,
                },
            ),
        ],
    )

    # ---- Functional + grid --------------------------------------------
    grid = None
    _set_xc_density = _density_set_gamma
    if is_ks:
        grid_options = getattr(opts, "grid", None)
        if grid_options is None:
            grid_options = GridOptions()
        if bool(getattr(opts, "use_periodic_becke", False)):
            grid = build_periodic_becke_grid(
                system,
                grid_options=grid_options,
                image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 0.0)),
            )
            # Periodic-Becke grid (one-cell partition of the crystal)
            # pairs with the *periodic* Γ-torus density (D in every
            # lattice block -> build_xc_periodic's cross-cell mode).
            # Feeding the home-cell-only set here evaluates the
            # molecular density on the periodic grid: exact in a vacuum
            # box, ~0.3 Ha wrong on tight ionic cells (the 2026-07-09
            # KRKS finding, HANDOVER_AICCM_DIRECT_TORUS.md §4).
            _set_xc_density = _density_set_torus_gamma
        else:
            # Molecular Becke grid (v0.8.x reproduction path) pairs with
            # the molecular-limit home-cell-only density.
            grid = build_grid(system.unit_cell_molecule(), grid_options)

    molecular_limit_gdf = int(system.dim) == 3 and n_int_cells == 1
    use_ewald_jk = int(system.dim) == 3 and (
        force_ewald_jk or not molecular_limit_gdf
    )
    if force_ewald_jk and molecular_limit_gdf:
        plog.info(
            "J/K routing: Ewald-3D forced for a finite periodic torus "
            "even though the lattice cutoff contains only the home cell"
        )

    # ---- One-electron integrals at Γ ----------------------------------
    #
    # Force V_ne (and e_nuc below) through the Ewald-3D gauge regardless
    # of the user-supplied coulomb_method.  See
    # ``_gauge_lat_opts_for_v_ne_and_e_nuc`` for rationale + the
    # 2026-05-13 gauge-regression handover doc.  ``coulomb_method`` on
    # the user-facing ``lat_opts`` still controls J/K routing.
    if use_ewald_jk:
        gauge_lat_opts = _gauge_lat_opts_for_v_ne_and_e_nuc(lat_opts, system)
        if gauge_lat_opts is not lat_opts:
            plog.info(
                "V_ne / e_nuc gauge: Ewald-3D "
                "(forced for 3D-periodic systems regardless of "
                f"lat_opts.coulomb_method={lat_opts.coulomb_method!r})"
            )
    elif use_private_slab_gdf:
        # The reciprocal fit, V_ne, and E_nuclear all use the same rigorous
        # 2D slab gauge. Unlike the legacy bare-Lpq route, the reciprocal fit
        # does not inherit the real-space integral cutoff, so the nuclear
        # cutoff must not be clamped to it.
        gauge_lat_opts = lat_opts
    else:
        # Lpq J/K path: 3D molecular-limit cell (GDF cutoff contains the
        # home cell only) or any dim<3 vacuum-padded wire/slab. J and K
        # come from the native Lpq tensor, which is summed over the GDF
        # integral cutoff `cutoff_bohr`. V_ne and e_nuc MUST sum nuclei
        # over the SAME cell set, or the long-range electron-nuclear
        # attraction is summed to image nuclei (out to the default
        # nuclear_cutoff_bohr=25) with no compensating electron-electron
        # repulsion (Lpq sees only cutoff_bohr=15) -- the neutral cell's
        # cancellation breaks and the energy over-binds with no warning
        # (CLAUDE.md Sec.7; audit F3 2026-05-31). H2 in an 18-bohr box came
        # back -1.783 Ha at the default cutoffs vs the isolated-H2
        # -1.117 Ha once the nuclear cutoff is clamped to cutoff_bohr.
        if lat_opts.nuclear_cutoff_bohr > lat_opts.cutoff_bohr:
            gauge_lat_opts = _lat_opts_with_nuclear_cutoff(
                lat_opts, lat_opts.cutoff_bohr
            )
            plog.info(
                "V_ne / e_nuc nuclear cutoff clamped to the GDF integral "
                f"cutoff {lat_opts.cutoff_bohr:.2f} bohr (was "
                f"{lat_opts.nuclear_cutoff_bohr:.2f} bohr) to match the "
                "Lpq J/K cell set and preserve neutral-cell cancellation"
            )
        else:
            gauge_lat_opts = lat_opts
    slab_fit = None
    with plog.stage(
        "integrals_lattice",
        detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr",
    ):
        _use_sym = system.symmetry is not None and system.symmetry.operations
        if _use_sym:
            ops = system.symmetry.operations
            plog.info(
                f"S/T integrals: symmetry-reduced path "
                f"(SG {system.symmetry.international_symbol}, "
                f"{system.symmetry.order} ops)"
            )
            _, S_blocks = compute_overlap_lattice_reduced(
                basis,
                system,
                lat_opts,
                ops,
            )
            S_lat = compute_overlap_lattice(basis, system, lat_opts)
            for i in range(len(S_lat)):
                S_lat.set_block(i, S_blocks[i])

            _, T_blocks = compute_kinetic_lattice_reduced(
                basis,
                system,
                lat_opts,
                ops,
            )
            T_lat = compute_kinetic_lattice(basis, system, lat_opts)
            for i in range(len(T_lat)):
                T_lat.set_block(i, T_blocks[i])
        else:
            S_lat = compute_overlap_lattice(basis, system, lat_opts)
            T_lat = compute_kinetic_lattice(basis, system, lat_opts)

        from .periodic_v_ne import compute_nuclear_lattice_dispatch

        V_lat = compute_nuclear_lattice_dispatch(basis, system, gauge_lat_opts)

    k_gamma = np.zeros(3)
    S = np.real(bloch_sum(S_lat, k_gamma))
    T = np.real(bloch_sum(T_lat, k_gamma))
    V = np.real(bloch_sum(V_lat, k_gamma))
    Hcore = 0.5 * ((T + V) + (T + V).T)
    S = 0.5 * (S + S.T)

    scf_preflight_overlap_check(S, plog=plog, label="S(Γ)", basis=basis)
    X, n_kept = _canonical_orthogonalizer(S, linear_dep_threshold)
    if n_occ > n_kept:
        raise RuntimeError(
            "run_rhf_periodic_gamma_gdf: canonical orthogonalisation "
            f"dropped too many directions (n_occ={n_occ}, n_kept={n_kept})"
        )

    use_davidson = getattr(opts, "use_davidson", False)
    dav_opts = getattr(opts, "davidson", None)
    dav_dim = getattr(opts, "davidson_min_dim", 100)
    use_dav = use_davidson and S.shape[0] >= dav_dim
    if use_dav and dav_opts is None:
        from vibeqc._vibeqc_core import DavidsonOptions

        dav_opts = DavidsonOptions()

    # ---- Native GDF factor --------------------------------------------
    mol = system.unit_cell_molecule()
    with plog.stage("aux_basis", detail=aux_name):
        aux = make_aux_basis_set(mol, aux_name=aux_name, drop_eta=float(aux_drop_eta))
    plog.info(f"aux basis: {aux_name}  ({aux.nbasis} BFs / {aux.nshells} shells)")

    with plog.stage(
        "native_lpq",
        detail=(
            f"2c/3c lattice ERIs, aux={aux.nbasis}, "
            f"fit_thr={gdf_linear_dep_threshold:.1e}"
        ),
    ):
        if gdf_algorithm == "rsgdf" and int(system.dim) == 3:
            plog.warn(
                "RSGDF on 3D systems is an experimental research feature "
                "(~230 mHa error on H2/12-bohr; architectural limitation -- "
                "the sparse G-mesh sized by the LR damping kernel cannot "
                "resolve compact aux primitives). Prefer "
                "gdf_algorithm='bare' (1D/2D) or the compcell path via "
                "run_pbc_gdf_rhf (3D). See aux_basis.py RSGDF LR status "
                "note for details."
            )
        if use_private_slab_gdf:
            if not apply_modrho:
                raise ValueError(
                    "private slab-truncated GDF requires apply_modrho=True"
                )
            modrho_aux = make_modrho_aux_basis(aux, mol)
            slab_fit = _build_lpq_bloch_slab_truncated(
                system,
                basis,
                modrho_aux,
                np.zeros(3),
                np.zeros(3),
                ke_cutoff=float(_slab_ke_cutoff),
                lat_opts=lat_opts,
                linear_dep_thr=float(gdf_linear_dep_threshold),
            )
            Lpq = slab_fit.factors
        else:
            Lpq = build_lpq_native(
                system,
                basis,
                aux,
                lat_opts=lat_opts,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                apply_modrho=bool(apply_modrho),
                algorithm=gdf_algorithm,
                rsgdf_omega=float(rsgdf_omega),
            )
    plog.info(
        f"Lpq:        {Lpq.shape[0]} fit vectors, "
        f"shape=({Lpq.shape[0]}, {Lpq.shape[1]}, {Lpq.shape[2]})"
    )

    if use_ewald_jk:
        # Ewald-gauge branch only: converged Ewald nuclear energy.
        # nuclear_repulsion_per_cell(EWALD_3D) truncates its real-space
        # sum at nuclear_cutoff_bohr, which on dense ionic cells is
        # unconverged at the 1e-5 Ha level with a spurious geometry
        # dependence (see pbc_gdf._pbc_gdf_gamma_setup). The
        # molecular-limit / dim<3 branches stay on the truncated
        # matched-cutoff sum: their J/K/V_ne are bare-gauge and rely on
        # the neutral-cell cancellation at the SAME cell set (audit F3
        # 2026-05-31 above) -- an Ewald e_nuc there is gauge-inconsistent.
        e_nuc = float(ewald_nuclear_repulsion(system))
    else:
        e_nuc = float(nuclear_repulsion_per_cell(system, gauge_lat_opts))

    # --- Steps 2 + 3 of the 2026-05-13 gauge fix: Ewald-3D J + K -------
    #
    # Step 2: for dim == 3 the Hartree J is built via the composed
    # Ewald-3D builder (short-range erfc/r + long-range FFT-Poisson of
    # erf/r), bringing J into the same Madelung/Ewald gauge as V_ne /
    # e_nuc (Step 1) and CRYSTAL14 / PySCF `exxdiv='ewald'`.
    #
    # Step 3 (revised): the K build *also* moves off the native modrho-
    # Lpq factor for dim == 3.  The 2026-05-14 forensic K-vs-K comparison
    # on LiH/STO-3G showed `_build_k_from_lpq` gives ||K|| ~5x too large
    # on tight ionic cells -- `build_lpq_native`'s native Lpq tensor is
    # wrong for tight cells where AO images overlap.  So for v0.8.0 K is
    # built via the same full-range real-space path the working EWALD_3D
    # driver uses (`build_jk_gamma_molecular_limit` at omega=0).
    #
    # The 2026-05-14 handover originally framed this molecular-limit
    # kernel as a builder bug ("misses g_l != g_s contributions"); the
    # 2026-05-15 cutoff-convergence sweep (retired forensic note in git
    # history) showed the
    # alternative "homogeneous-D triple-cell-sum" build diverges with
    # lattice cutoff and the molecular-limit kernel is in fact the
    # correct Γ-only periodic Fock build (the inverse-Bloch convention
    # P(g) = P_Γ . d_{g,0}).  The -0.9 Ha gap to CRYSTAL14 SHRINK 8 8
    # on LiH primitive is BZ-sampling residual, not a builder bug --
    # multi-k sampling is the fix for tight ionic crystals.
    #
    # Consequence: in v0.8.0 this "GDF" driver does NOT use density
    # fitting for J or K on tight dim == 3 cells -- it is Ewald-J +
    # real-space-K, identical in operation to
    # `run_rhf_periodic_gamma_ewald3d`.  In the molecular-limit regime
    # where the GDF cutoff contains only the home cell, the native Lpq
    # tensor remains the intended backend and exercises arbitrary 3D
    # lattice metrics without the tight-cell image-overlap pathology.
    #
    # Pre-compute the FFT grid once; it depends only on lattice geometry.
    j_ewald_grid_shape: Optional[Tuple[int, int, int]] = None
    # Iteration-invariant analytic-FT Hartree-J builder for the Ewald-JK
    # fallback path; built once below (None on the Lpq path).
    j_ewald_builder = None
    if use_ewald_jk:
        lat_arr = np.asarray(system.lattice, dtype=float)
        j_ewald_grid_shape = tuple(
            int(x) for x in auto_grid(lat_arr, _J_EWALD_SPACING_BOHR)
        )
        # Resolve the Ewald split w once (fixed geometry -> fixed w). The
        # analytic-FT J is w-invariant, so this matters only for the
        # diagnostic grid backend; resolving it here keeps that path
        # byte-identical to the former per-call resolution.
        _j_ewald_omega = _J_EWALD_OMEGA
        if _j_ewald_omega <= 0.0:
            from .bipole_ext_el_pole import crystal_default_ewald_alpha

            _V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
            _j_ewald_omega = crystal_default_ewald_alpha(_V_cell)
        # Build the density-INDEPENDENT analytic-FT machinery (the Bloch
        # AO-pair FT pair_ft, the dense G-mesh, and the 4pi/G^2 kernel -- an
        # EwaldJFTGammaCache) ONCE per SCF via make_ewald_3d_gamma_j_builder,
        # then reuse it every iteration; the closure redoes only the O(n^2.n_G)
        # density contraction. Previously _build_j called build_j_ewald_3d
        # each iteration, which rebuilt this cache every cycle (E2,
        # docs/pbc_audit_2026-06.md -- the GDF Lpq cache is the model this
        # mirrors). Bit-identical to the per-iteration build_j_ewald_3d call
        # (gated by test_ewald_j_cache.test_gamma_builder_matches_build_j_ewald_3d);
        # the diagnostic VIBEQC_J_EWALD3D_BACKEND=grid path stays uncached.
        j_ewald_builder = make_ewald_3d_gamma_j_builder(
            basis,
            system,
            omega=_j_ewald_omega,
            lattice_opts=lat_opts,
            grid_shape=j_ewald_grid_shape,
            origin=None,
            spacing_bohr=_J_EWALD_SPACING_BOHR,
        )
        plog.info(
            "J gauge: Ewald-3D "
            f"(omega={'auto' if _J_EWALD_OMEGA <= 0.0 else f'{_J_EWALD_OMEGA:.3f}'} / bohr, "
            f"FFT grid {j_ewald_grid_shape[0]}x"
            f"{j_ewald_grid_shape[1]}x{j_ewald_grid_shape[2]})"
        )
        plog.info(
            "K gauge: full-range real-space "
            "(molecular-limit kernel = correct Γ-only build; "
            "tight-ionic accuracy needs multi-k)"
        )
    elif molecular_limit_gdf:
        plog.info(
            "J/K backend: native Gamma GDF "
            "(3D molecular-limit cell; GDF cutoff contains home cell only)"
        )

    def _build_j(D_in: np.ndarray) -> np.ndarray:
        """Coulomb J for the current density.  For dim==3 use the Ewald-3D
        composed builder (gauge-correct), reusing the iteration-invariant
        analytic-FT machinery cached once in ``j_ewald_builder`` above; fall
        back to the Lpq-based J for dim<3 (vacuum-padded; bare gauge is fine)."""
        if use_ewald_jk:
            J_out = j_ewald_builder(D_in)
        elif use_private_slab_gdf:
            J_out, _ = _contract_slab_gdf_gamma(slab_fit, D_in, S)
        else:
            J_out = _build_j_from_lpq(Lpq, D_in)
        return np.real(0.5 * (J_out + J_out.conj().T))

    def _build_k(D_in: np.ndarray) -> np.ndarray:
        """Exchange K for the current density.  For dim==3 use the
        full-range real-space molecular-limit builder (the correct
        Γ-only build per the 2026-05-15 cutoff-convergence finding);
        fall back to the Lpq-based K for dim<3 (vacuum-padded -- the
        native Lpq tensor is fine when AO images don't overlap)."""
        if use_ewald_jk:
            jk = build_jk_gamma_molecular_limit(
                basis,
                system,
                lat_opts,
                D_in,
                0.0,
            )
            K_out = np.asarray(jk.K)
        elif use_private_slab_gdf:
            _, K_out = _contract_slab_gdf_gamma(slab_fit, D_in, S)
        else:
            K_out = _build_k_from_lpq(Lpq, D_in)
        return np.real(0.5 * (K_out + K_out.conj().T))

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

    def occupations_from_eps(
        eps: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        if smearing_T <= 0.0:
            occ = _aufbau_occupations_per_k([eps], n_occ)[0]
            return np.asarray(occ, dtype=float), 0.0, 0.0
        occ_list, mu, entropy = _fermi_dirac_occupations_per_k(
            [eps],
            [1.0],
            float(n_elec),
            smearing_T,
        )
        return np.asarray(occ_list[0], dtype=float), float(mu), float(entropy)

    D_engine = initial_density_closed_shell(
        mol,
        basis,
        n_occ,
        InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
        is_periodic=True,
        periodic_system=system,
        lattice_opts=lat_opts,
        # READ restart (Γ-only): prior g=0 cell density (pre-resolved from
        # read_from, or read + projected from read_path). Ignored unless READ.
        read_density=getattr(opts, "read_density", None),
        read_path=getattr(opts, "read_path", ""),
        overlap=S,
    )
    if D_engine is not None:
        plog.info(f"initial guess: {guess.name} (density via GuessEngine)")
        D = D_engine
        C0, eps0 = diagonalise(Hcore)
    else:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise)")
        C0, eps0 = diagonalise(Hcore)
        occ0, _, _ = occupations_from_eps(eps0)
        D = _density_from_orbitals_and_occupations(C0, occ0)
    if guess in (InitialGuess.SAP, InitialGuess.HUECKEL):
        guess_fock = periodic_fock_guess_k(
            system, basis, [np.zeros(3)], guess, lattice_opts=lat_opts,
        )[0].real
        C0, eps0 = diagonalise(guess_fock)
        occ0, _, _ = occupations_from_eps(eps0)
        D = _density_from_orbitals_and_occupations(C0, occ0)
    elif guess == InitialGuess.PATOM:
        C0, eps0 = diagonalise(Hcore + _build_j(D) - 0.5 * _build_k(D))
        occ0, _, _ = occupations_from_eps(eps0)
        D = _density_from_orbitals_and_occupations(C0, occ0)
    D_prev = D.copy()
    occ, fermi_level, entropy = occupations_from_eps(eps0)
    if smearing_T > 0.0:
        plog.info(
            "smearing: Fermi-Dirac kBT = "
            f"{smearing_T:.6g} Ha "
            f"({_hartree_to_kelvin_temperature(smearing_T):.1f} K)"
        )

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_rhf_periodic_gamma_gdf: damping must be in [0, 1); got {damping}"
        )
    if fock_mixing_value != 0.0:
        plog.info(
            "fock mixing: CRYSTAL FMIXING "
            f"{100.0 * fock_mixing_value:.1f}% "
            "(previous Fock/KS matrix weight)"
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
    if level_shift != 0.0:
        if warmup_cycles > 0:
            cycle_word = "cycle" if warmup_cycles == 1 else "cycles"
            plog.info(
                "level-shift warm-up: "
                f"{warmup_cycles} {cycle_word} at {level_shift:.3f} Ha; "
                "then restart unshifted"
            )
        else:
            plog.info(
                f"level shift: {level_shift:.3f} Ha applied at each diagonalization"
            )

    D_set = compute_overlap_lattice(basis, system, lat_opts) if is_ks else None

    plog.banner(f"SCF ({label} Gamma, native GDF)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    scf_trace: List[SCFIteration] = []
    result = PeriodicRHFGDFResult(
                 restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=float(e_nuc),
        e_xc=0.0,
        e_coulomb=0.0,
        e_hf_exchange=0.0,
        n_iter=0,
        converged=False,
        mo_energies=np.empty(0),
        mo_coeffs=np.empty((0, 0)),
        density=D.copy(),
        fock=np.empty((0, 0)),
        overlap=S,
        scf_trace=scf_trace,
        hcore=Hcore,
        aux_basis_name=aux_name,
        n_aux=int(aux.nbasis),
        n_fit=int(Lpq.shape[0]),
        linear_dep_threshold=float(gdf_linear_dep_threshold),
        gdf_algorithm=gdf_algorithm,
        functional=func_name,
        fock_mixing=fock_mixing_value,
        level_shift=level_shift,
        level_shift_warmup_cycles=warmup_cycles,
        smearing_temperature=smearing_T,
        fermi_level=float(fermi_level),
        entropy=float(entropy),
        free_energy=0.0,
        occupations=np.asarray(occ, dtype=float),
        backend=(
            "ewald-jk-fallback"
            if use_ewald_jk
            else (
                "private-slab-truncated-gdf"
                if use_private_slab_gdf
                else "native-gamma-gdf"
            )
        ),
    )

    E_prev = 0.0
    C_final = C0
    eps_final = eps0
    F_prev_mixed: Optional[np.ndarray] = None

    P_cache: list = []
    if symmetry_stabilize and system.symmetry is not None:
        from .symmetry_integrals import symmorphic_operations as _so
        from .symmetry_scf import build_ao_permutation_cache as _bpc

        ops = _so(system.symmetry.operations)
        if ops:
            P_cache = _bpc(system, basis, ops)
            plog.info(f"symmetry_stabilize: {len(P_cache)} symmorphic ops")

    for iter_idx in range(1, max_iter + 1):
        if damper is not None:
            damping = damper.alpha
        if warmup_cycles > 0 and iter_idx == warmup_cycles + 1:
            if accel is not None:
                accel = PeriodicSCFAccelerator(opts)
            F_prev_mixed = None
            plog.info("restart: unshifted Fock with fresh DIIS history")

        if _ls_schedule:
            active_level_shift = level_shift_at_iter(
                level_shift, warmup_cycles, _ls_schedule, max_iter, iter_idx
            )
        else:
            active_level_shift = (
                level_shift
                if (
                    level_shift != 0.0
                    and (warmup_cycles == 0 or iter_idx <= warmup_cycles)
                )
                else 0.0
            )
        diis_active = use_diis and iter_idx >= diis_start_iter
        D_used = (
            D
            if (iter_idx == 1 or damping == 0.0 or diis_active)
            else damping * D_prev + (1.0 - damping) * D
        )

        if symmetry_reduce_fock and system.symmetry is not None:
            from .symmetry_fock_reduced import compute_jk_gamma_reduced as _jk_red

            J, K = _jk_red(
                basis,
                system,
                lat_opts_2e,
                D_used,
                system.symmetry.operations,
            )
        else:
            J = _build_j(D_used)
            K = _build_k(D_used) if alpha != 0.0 else None
        if symmetry_stabilize and P_cache:
            from .symmetry_scf import symmetrize_matrix as _sym

            J = _sym(J, P_cache)
            if K is not None:
                K = _sym(K, P_cache)
        F_HF_part = J - 0.5 * alpha * K if K is not None else J

        E_xc = 0.0
        V_xc = 0.0
        if is_ks:
            _set_xc_density(D_set, D_used)
            xc_contrib = build_xc_periodic(
                basis,
                system,
                grid,
                func,
                D_set,
                lat_opts,
            )
            V_xc = np.real(bloch_sum(xc_contrib.V_xc, k_gamma))
            V_xc = 0.5 * (V_xc + V_xc.T)
            E_xc = float(xc_contrib.e_xc)

        F = Hcore + F_HF_part + V_xc
        F = 0.5 * (F + F.T)

        E_core = float(np.einsum("ij,ij->", D_used, Hcore))
        E_hf_trace = 0.5 * float(np.einsum("ij,ij->", D_used, F_HF_part))
        E_elec = E_xc + E_core + E_hf_trace
        E_total = E_elec + float(e_nuc)
        E_coulomb = 0.5 * float(np.einsum("ij,ij->", D_used, J))
        E_hf_K = (
            -0.25 * alpha * float(np.einsum("ij,ij->", D_used, K))
            if K is not None
            else 0.0
        )
        free_energy = E_total - smearing_T * entropy

        FDS = F @ D_used @ S
        grad = FDS - FDS.T
        grad_norm = float(np.linalg.norm(grad))
        dE = free_energy - E_prev
        converged = (
            iter_idx > 1
            and (warmup_cycles == 0 or iter_idx > warmup_cycles)
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )

        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(free_energy),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iter_idx,
            energy=float(free_energy),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )
        plog.energy_decomposition(
            iter_idx,
            E_kin=float(np.einsum("ij,ij->", D_used, T)),
            E_ne=float(np.einsum("ij,ij->", D_used, V)),
            E_J=E_coulomb,
            E_xc=E_xc if is_ks else None,
            E_K=E_hf_K,
            E_nuc=float(e_nuc),
        )

        if accel is not None:
            F_ex = accel.extrapolate_rhf(
                F,
                error=grad,
                density=D_used,
                energy=E_total,
                mo_coeffs=C_final,
                mo_energies=eps_final,
                n_occ=n_occ,
            )
            if diis_active:
                F = F_ex
        if fock_mixing_value != 0.0:
            if F_prev_mixed is not None:
                F_mixed = (
                    1.0 - fock_mixing_value
                ) * F + fock_mixing_value * F_prev_mixed
                F = 0.5 * (F_mixed + F_mixed.T)
            F_prev_mixed = F.copy()

        # Shared Saunders-Hillier operator: the weight on S·D·S is fixed by
        # the density convention, not by taste. ``D_used`` is the closed-shell
        # total density (occupations in {0, 2}), hence TOTAL. Returns F
        # untouched when the shift is 0.
        # Guarded: skip the pybind conversion of S and D entirely on an
        # unshifted cycle, which is the default and the common case.
        F_diag = (
            F if active_level_shift == 0.0
            else apply_level_shift(
                F, S, D_used, active_level_shift, LevelShiftDensity.TOTAL)
        )

        C_new, eps_new = diagonalise(F_diag)
        occ, fermi_level, entropy = occupations_from_eps(eps_new)
        D_prev = D_used
        D = _density_from_orbitals_and_occupations(C_new, occ)

        C_final = C_new
        eps_final = eps_new
        if damper is not None:
            damper.update(free_energy)
        E_prev = free_energy

        result.energy = E_total
        result.e_electronic = E_elec
        result.e_xc = E_xc
        result.e_coulomb = E_coulomb
        result.e_hf_exchange = E_hf_K
        result.n_iter = iter_idx
        result.mo_energies = eps_new
        result.mo_coeffs = C_new
        result.density = D_used
        result.fock = F
        result.fermi_level = float(fermi_level)
        result.entropy = float(entropy)
        result.free_energy = float(free_energy)
        result.occupations = np.asarray(occ, dtype=float)

        if converged:
            J_f = _build_j(D)
            K_f = _build_k(D) if alpha != 0.0 else None
            F_HF_f = J_f - 0.5 * alpha * K_f if K_f is not None else J_f
            E_xc_f = 0.0
            V_xc_f = 0.0
            if is_ks:
                _set_xc_density(D_set, D)
                xc_f = build_xc_periodic(
                    basis,
                    system,
                    grid,
                    func,
                    D_set,
                    lat_opts,
                )
                V_xc_f = np.real(bloch_sum(xc_f.V_xc, k_gamma))
                V_xc_f = 0.5 * (V_xc_f + V_xc_f.T)
                E_xc_f = float(xc_f.e_xc)
            F_f = 0.5 * ((Hcore + F_HF_f + V_xc_f) + (Hcore + F_HF_f + V_xc_f).T)
            C_f, eps_f = diagonalise(F_f)
            occ_f, fermi_level_f, entropy_f = occupations_from_eps(eps_f)
            E_core_f = float(np.einsum("ij,ij->", D, Hcore))
            E_hf_trace_f = 0.5 * float(np.einsum("ij,ij->", D, F_HF_f))
            E_elec_f = E_xc_f + E_core_f + E_hf_trace_f
            result.energy = E_elec_f + float(e_nuc)
            result.e_electronic = E_elec_f
            result.e_xc = E_xc_f
            result.e_coulomb = 0.5 * float(np.einsum("ij,ij->", D, J_f))
            result.e_hf_exchange = (
                -0.25 * alpha * float(np.einsum("ij,ij->", D, K_f))
                if K_f is not None
                else 0.0
            )
            result.mo_energies = eps_f
            result.mo_coeffs = C_f
            result.density = D
            result.fock = F_f
            result.fermi_level = float(fermi_level_f)
            result.entropy = float(entropy_f)
            result.free_energy = float(result.energy - smearing_T * entropy_f)
            result.occupations = np.asarray(occ_f, dtype=float)
            result.converged = True

            # Diagnostic: HOMO-LUMO gap check at convergence.  Mirrors
            # CRYSTAL14's `POSSIBLY CONDUCTING STATE - EFERMI(AU)`
            # warning that fires when the gap between the highest
            # occupied and lowest unoccupied MO is below threshold.
            # For a closed-shell Γ-only calc on a true insulator the
            # gap should be tens of mHa or larger; below ~1 mHa the
            # system is effectively metallic at this k-point and
            # SCF convergence may be fragile / sensitive to start
            # guess.  This is purely informational -- we don't fail
            # the SCF.  The 1e-3 Ha (~27 meV) threshold matches
            # CRYSTAL14's default reporting cutoff.
            _CONDUCTING_GAP_THRESHOLD_HA = 1e-3
            if n_occ > 0 and n_occ < eps_f.shape[0]:
                homo = float(eps_f[n_occ - 1])
                lumo = float(eps_f[n_occ])
                gap = lumo - homo
                if gap < _CONDUCTING_GAP_THRESHOLD_HA:
                    plog.info(
                        "POSSIBLY CONDUCTING STATE at convergence: "
                        f"HOMO-LUMO gap = {gap:.6f} Ha "
                        f"({gap * 27.2114:.3f} eV) "
                        f"below the {_CONDUCTING_GAP_THRESHOLD_HA:.0e} Ha threshold "
                        f"(HOMO={homo:.6f}, LUMO={lumo:.6f}). "
                        "For metals / small-gap systems consider Fermi smearing "
                        "(opts.smearing_temperature > 0) or denser k-mesh; "
                        "Γ-only RHF on a true metal can be ill-defined."
                    )

            plog.converged(
                n_iter=result.n_iter,
                energy=result.energy,
                converged=True,
            )
            _check_energy_sanity(result, system, plog)
            return result

    result.mo_coeffs = C_final
    result.mo_energies = eps_final
    result.converged = False
    plog.converged(
        n_iter=result.n_iter,
        energy=result.energy,
        converged=False,
    )
    return result


def _run_rhf_periodic_gamma_slab_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicRHFOptions] = None,
    *,
    aux_basis: Optional[str] = None,
    ke_cutoff: float = 200.0,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: Union[bool, ProgressLogger, None] = None,
) -> PeriodicRHFGDFResult:
    """Run the private Gamma RHF slab-GDF validation route.

    This entry is intentionally absent from ``__all__`` and the top-level
    package. Public slab-GDF dispatch remains fail-closed until compact and
    multi-k total-energy gates are complete.
    """
    return run_rhf_periodic_gamma_gdf(
        system,
        basis,
        options,
        aux_basis=aux_basis,
        linear_dep_threshold=linear_dep_threshold,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        _slab_truncated_private=True,
        _slab_ke_cutoff=ke_cutoff,
    )


def _run_krhf_periodic_slab_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: tuple[int, int, int],
    options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
    *,
    functional: Optional[str] = None,
    aux_basis: Optional[str] = None,
    ke_cutoff: float = 200.0,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
) -> _PrivateSlabKRHFGDFResult:
    """Run the private multi-k RHF/RKS slab-GDF validation route.

    The implementation is intentionally narrow: closed-shell zero-temperature
    RHF or RKS on a full Gamma-centered slab mesh, with no public dispatch,
    symmetry reduction, or optional post-SCF work. It exists to validate
    the signed J/K, BvK gauge, and finite-torus XC composition end to end before
    those pieces enter a production driver. Range-separated functionals remain
    fail-closed because this fitted exchange route is full-range only.
    """
    opts = options
    if opts is None:
        opts = PeriodicKSOptions() if functional else PeriodicRHFOptions()
    from .guess import _coerce_periodic_driver_guess, periodic_fock_guess_k
    guess = _coerce_periodic_driver_guess(
        opts.initial_guess, driver="private multi-k slab GDF",
        supported=periodic_guess_capabilities('gdf', 'RHF', dim=getattr(system, "dim", 3), multi_k=False, transport='k'),
    )
    from .pbc_gdf import _refuse_ecp_options
    _refuse_ecp_options(opts, "private multi-k slab GDF", system=system)
    func_name = functional or str(getattr(opts, "functional", "") or "")
    is_ks = bool(func_name)
    func = Functional(func_name, 1) if is_ks else None
    reject_unscreened_range_separated(
        func, where="_run_krhf_periodic_slab_gdf"
    )
    alpha = float(func.hf_exchange_fraction) if func is not None else 1.0
    lat_opts: LatticeSumOptions = opts.lattice_opts
    if int(system.dim) != 2:
        raise ValueError("private multi-k slab GDF requires a dim=2 system")
    if lat_opts.coulomb_method != CoulombMethod.SLAB_EWALD_2D:
        raise ValueError(
            "private multi-k slab GDF requires "
            "coulomb_method=SLAB_EWALD_2D"
        )
    mesh_raw = np.asarray(mesh, dtype=float)
    if (
        mesh_raw.shape != (3,)
        or not np.all(np.isfinite(mesh_raw))
        or not np.all(mesh_raw == np.rint(mesh_raw))
        or np.any(mesh_raw < 1)
    ):
        raise ValueError(
            "private multi-k slab GDF requires a positive integer "
            "(n1,n2,1) mesh"
        )
    mesh_values = np.rint(mesh_raw).astype(int)
    if int(mesh_values[2]) != 1:
        raise ValueError(
            "private multi-k slab GDF requires a mesh with n3 == 1"
        )
    mesh_tuple = tuple(int(value) for value in mesh_values)
    plog = resolve_progress(progress, verbose=verbose)
    n_elec = int(system.n_electrons())
    if n_elec % 2 != 0 or int(system.multiplicity) != 1:
        raise ValueError(
            "private multi-k slab GDF requires a closed-shell singlet"
        )
    if float(getattr(opts, "smearing_temperature", 0.0)) != 0.0:
        raise NotImplementedError(
            "private multi-k slab GDF does not yet support smearing"
        )
    n_occ = n_elec // 2

    kmesh = _mp_native(system, list(mesh_tuple), [0, 0, 0], False)
    kpoints = np.asarray(kmesh.kpoints, dtype=float)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = int(kpoints.shape[0])
    plog.banner(
        "run_krhf_periodic_slab_gdf  "
        f"{'RKS' if is_ks else 'RHF'}  kmesh={mesh_tuple}"
    )

    S_lattice = compute_overlap_lattice(basis, system, lat_opts)
    T_lattice = compute_kinetic_lattice(basis, system, lat_opts)
    from .periodic_v_ne_slab import (
        build_v_ne_slab_ewald_2d_k_cache,
        compute_v_ne_slab_ewald_2d_k_matrix,
    )

    v_ne_cache = build_v_ne_slab_ewald_2d_k_cache(
        basis,
        system,
        lat_opts,
        alpha=0.0,
    )
    cells = list(S_lattice.cells)
    overlaps: List[np.ndarray] = []
    hcores: List[np.ndarray] = []
    orthogonalizers: List[np.ndarray] = []
    for kpoint in kpoints:
        overlap = np.asarray(bloch_sum(S_lattice, kpoint), dtype=complex)
        kinetic = np.asarray(bloch_sum(T_lattice, kpoint), dtype=complex)
        nuclear = compute_v_ne_slab_ewald_2d_k_matrix(
            basis,
            system,
            lat_opts,
            kpoint,
            alpha=0.0,
            cache=v_ne_cache,
        )
        overlap = 0.5 * (overlap + overlap.conj().T)
        hcore = kinetic + nuclear
        hcore = 0.5 * (hcore + hcore.conj().T)
        X, n_kept = _canonical_orthogonalizer_complex(
            overlap, linear_dep_threshold
        )
        if n_occ > n_kept:
            raise RuntimeError(
                "private multi-k slab GDF overlap rank is smaller than the "
                "occupied space"
            )
        overlaps.append(overlap)
        hcores.append(hcore)
        orthogonalizers.append(X)

    molecule = system.unit_cell_molecule()
    aux_name = aux_basis or default_aux_for(basis.name)
    raw_aux = make_aux_basis_set(molecule, aux_name=aux_name)
    modrho_aux = make_modrho_aux_basis(raw_aux, molecule)
    fits = {}
    for i, k_bra in enumerate(kpoints):
        for j, k_ket in enumerate(kpoints):
            fits[(i, j)] = _build_lpq_bloch_slab_truncated(
                system,
                basis,
                modrho_aux,
                k_bra,
                k_ket,
                ke_cutoff=ke_cutoff,
                lat_opts=lat_opts,
                linear_dep_thr=gdf_linear_dep_threshold,
            )
    xi_bvk = _slab_probe_charge_madelung_for_kmesh(system, mesh_tuple)
    e_nuclear = float(nuclear_repulsion_per_cell(system, lat_opts))
    grid = None
    if is_ks:
        grid_options = getattr(opts, "grid", None) or GridOptions()
        if bool(getattr(opts, "use_periodic_becke", False)):
            grid = build_periodic_becke_grid(
                system,
                grid_options=grid_options,
                image_radius_bohr=float(
                    getattr(opts, "becke_image_radius_bohr", 0.0)
                ),
            )
        else:
            grid = build_grid(molecule, grid_options)

    def diagonalise(
        focks: List[np.ndarray],
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        coefficients = []
        energies = []
        for fock, X in zip(focks, orthogonalizers):
            transformed = X.conj().T @ fock @ X
            transformed = 0.5 * (transformed + transformed.conj().T)
            eps, coeff_transformed = np.linalg.eigh(transformed)
            coefficients.append(X @ coeff_transformed)
            energies.append(eps)
        return coefficients, energies

    def densities_from_coefficients(
        coefficients: List[np.ndarray],
        energies: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Per-k densities under ONE global zero-temperature Fermi level.

        Occupying ``coeff[:, :n_occ]`` at each k independently is a *per-k*
        Aufbau. When bands cross between k points it selects states that are
        not the globally lowest, so the density is built from the wrong
        subspace; on hBN/sto-3g that per-k selection flipped between
        adjacent iterations and the SCF oscillated for 200 cycles instead of
        converging. This applies the same weighted single-mu constraint that
        60104fc01 established for the 3D native multi-k GDF route.
        """
        if energies is None:
            # Only reachable for the pre-SCF seed, where the caller has no
            # Fock eigenvalues yet.
            occupations = [
                np.concatenate(
                    (
                        np.full(n_occ, 2.0, dtype=float),
                        np.zeros(max(0, coeff.shape[1] - n_occ), dtype=float),
                    )
                )
                for coeff in coefficients
            ]
        else:
            occupations, _ = _global_aufbau_with_mu(
                [np.asarray(np.real(eps), dtype=float) for eps in energies],
                np.asarray(weights, dtype=float),
                float(n_elec),
                occ_value=2.0,
            )
        densities = []
        for coeff, occ in zip(coefficients, occupations):
            weighted = coeff * np.asarray(occ, dtype=float)[None, :]
            density = weighted @ coeff.conj().T
            densities.append(0.5 * (density + density.conj().T))
        return densities

    def _is_per_k_integer_aufbau(occupations: List[np.ndarray]) -> bool:
        """Whether every k point has the legacy ``2[:n_occ], 0`` pattern."""
        for occ in occupations:
            actual = np.asarray(occ, dtype=float)
            expected = np.zeros_like(actual)
            expected[:n_occ] = 2.0
            if not np.array_equal(actual, expected):
                return False
        return True

    def _global_occupations(energies: List[np.ndarray]) -> List[np.ndarray]:
        occupations, _ = _global_aufbau_with_mu(
            [np.asarray(np.real(eps), dtype=float) for eps in energies],
            np.asarray(weights, dtype=float),
            float(n_elec),
            occ_value=2.0,
        )
        return occupations

    def build_j(densities: List[np.ndarray]) -> List[np.ndarray]:
        first = fits[(0, 0)]
        rho = np.zeros(first.factors.shape[0], dtype=complex)
        for j in range(n_k):
            fit_jj = fits[(j, j)]
            if not np.array_equal(fit_jj.metric_signs, first.metric_signs):
                raise RuntimeError(
                    "private multi-k slab GDF diagonal metric signatures "
                    "are inconsistent"
                )
            rho += float(weights[j]) * fit_jj.metric_signs * np.einsum(
                "Pmn,nm->P", fit_jj.factors, densities[j], optimize=True
            )
        coulomb = []
        for i in range(n_k):
            block = np.einsum(
                "P,Pmn->mn", rho, fits[(i, i)].factors, optimize=True
            )
            coulomb.append(0.5 * (block + block.conj().T))
        return coulomb

    def build_jk(
        densities: List[np.ndarray],
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        coulomb = build_j(densities)
        if alpha == 0.0:
            return coulomb, [np.zeros_like(block) for block in coulomb]
        exchange = _contract_slab_gdf_multik_exchange(
            fits,
            densities,
            overlaps,
            weights,
            bvk_probe_charge_madelung=xi_bvk,
        )
        return coulomb, [np.asarray(block) for block in exchange]

    coefficients, energies = diagonalise(hcores)
    # A finite slab Coulomb gauge can reorder the one-electron generalized
    # eigenvalues at different k-points even though the neutral Fock spectrum
    # is gauge invariant. Independently occupying Hcore(k) can therefore lock
    # compact cells into a discontinuous antibonding branch. Seed every point
    # with the Gamma occupied subspace, metric-normalized at that k; this is a
    # continuous-band analogue of the Gamma Hcore guess.
    if guess in (InitialGuess.SAP, InitialGuess.HUECKEL):
        focks = periodic_fock_guess_k(
            system, basis, kpoints, guess, lattice_opts=lat_opts,
            kinetic_lattice=T_lattice, overlap_lattice=S_lattice,
        )
        coefficients, energies = diagonalise(focks)
        densities = densities_from_coefficients(coefficients, energies)
    elif guess != InitialGuess.HCORE:
        density_engine = initial_density_closed_shell(
            molecule, basis, n_occ,
            InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
            is_periodic=True, periodic_system=system, lattice_opts=lat_opts,
            overlap=overlaps, weights=weights,
        )
        densities = [np.asarray(density_engine, dtype=complex).copy() for _ in range(n_k)]
        if guess == InitialGuess.PATOM:
            coulomb = build_j(densities)
            exchange = _contract_slab_gdf_multik_exchange(
                fits, densities, overlaps, weights,
                bvk_probe_charge_madelung=xi_bvk,
            )
            coefficients, energies = diagonalise([
                h + j - 0.5 * k for h, j, k in zip(hcores, coulomb, exchange)
            ])
            densities = densities_from_coefficients(coefficients, energies)
    elif is_ks:
        densities = densities_from_coefficients(coefficients)
    elif guess == InitialGuess.HCORE:
        densities = []
        gamma_occupied = coefficients[0][:, :n_occ]
        for overlap in overlaps:
            gram = gamma_occupied.conj().T @ overlap @ gamma_occupied
            gram = 0.5 * (gram + gram.conj().T)
            gram_values, gram_vectors = np.linalg.eigh(gram)
            if np.any(gram_values <= 0.0):
                raise RuntimeError(
                    "private multi-k slab GDF seed has zero norm"
                )
            gram_inv_sqrt = (
                gram_vectors / np.sqrt(gram_values)[None, :]
            ) @ gram_vectors.conj().T
            occupied = gamma_occupied @ gram_inv_sqrt
            density = 2.0 * occupied @ occupied.conj().T
            densities.append(0.5 * (density + density.conj().T))
    damping = float(opts.damping)
    if not 0.0 <= damping < 1.0:
        raise ValueError(
            "private multi-k slab GDF requires damping in [0,1)"
        )
    max_iter = int(opts.max_iter)
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
    e_previous = 0.0
    trace: List[SCFIteration] = []
    focks = [matrix.copy() for matrix in hcores]
    e_electronic = 0.0
    e_xc = 0.0
    e_coulomb = 0.0
    e_hf_exchange = 0.0

    for iteration in range(1, max_iter + 1):
        if damper is not None:
            damping = damper.alpha
        coulomb, exchange = build_jk(densities)
        density_real = None
        if is_ks:
            density_real = real_space_density_from_per_k_density(
                densities, kmesh, cells
            )
        two_electron = [
            J - 0.5 * alpha * K for J, K in zip(coulomb, exchange)
        ]
        vxc = [np.zeros_like(block) for block in hcores]
        e_xc = 0.0
        if is_ks:
            xc_contrib = build_xc_periodic(
                basis,
                system,
                grid,
                func,
                density_real,
                lat_opts,
            )
            e_xc = float(xc_contrib.e_xc)
            vxc = []
            for kpoint in kpoints:
                block = np.asarray(
                    bloch_sum(xc_contrib.V_xc, kpoint), dtype=complex
                )
                vxc.append(0.5 * (block + block.conj().T))
        focks = [
            0.5 * (F + F.conj().T)
            for F in (
                H + G + Vxc
                for H, G, Vxc in zip(hcores, two_electron, vxc)
            )
        ]
        e_electronic = 0.0
        e_coulomb = 0.0
        e_hf_exchange = 0.0
        grad_norm_sq = 0.0
        errors: List[np.ndarray] = []
        for weight, density, hcore, two_e, J, K, fock, overlap in zip(
            weights,
            densities,
            hcores,
            two_electron,
            coulomb,
            exchange,
            focks,
            overlaps,
        ):
            e_electronic += float(weight) * float(
                np.real(np.trace(density @ hcore))
                + 0.5 * np.real(np.trace(density @ two_e))
            )
            e_coulomb += 0.5 * float(weight) * float(
                np.real(np.trace(density @ J))
            )
            e_hf_exchange += -0.25 * alpha * float(weight) * float(
                np.real(np.trace(density @ K))
            )
            fds = fock @ density @ overlap
            error = fds - fds.conj().T
            errors.append(error)
            grad_norm_sq += float(weight) * float(np.linalg.norm(error) ** 2)
        e_electronic += e_xc
        energy = e_electronic + e_nuclear
        grad_norm = float(np.sqrt(grad_norm_sq))
        delta_e = energy - e_previous
        trace.append(
            SCFIteration(
                iter=iteration,
                energy=energy,
                delta_e=delta_e if iteration > 1 else 0.0,
                grad_norm=grad_norm,
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        plog.iteration(
            iteration,
            energy=energy,
            dE=delta_e if iteration > 1 else 0.0,
            grad=grad_norm,
            diis=(accel.subspace_size if accel is not None else 0),
        )
        converged = (
            iteration > 1
            and abs(delta_e) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )
        if converged:
            coefficients_final, energies_final = diagonalise(focks)
            plog.converged(
                n_iter=iteration,
                energy=energy,
                converged=True,
            )
            return _PrivateSlabKRHFGDFResult(
                       restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
                       guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                       restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
                energy=energy,
                e_electronic=e_electronic,
                e_nuclear=e_nuclear,
                n_iter=iteration,
                converged=True,
                mo_energies=energies_final,
                mo_coeffs=coefficients_final,
                density=densities,
                fock=focks,
                overlap=overlaps,
                hcore=hcores,
                kpoints_cart=kpoints,
                kpoint_weights=weights,
                scf_trace=trace,
                functional=func_name or None,
                e_xc=e_xc,
                e_coulomb=e_coulomb,
                e_hf_exchange=e_hf_exchange,
                aux_basis_name=aux_name,
                n_aux=int(raw_aux.nbasis),
                backend=(
                    "private-multik-slab-truncated-gdf-rks"
                    if is_ks
                    else "private-multik-slab-truncated-gdf-rhf"
                ),
            )

        diis_active = use_diis and iteration >= diis_start_iter
        if accel is not None:
            # Mirrors the 60104fc01 guard on the 3D native multi-k route:
            # KDIIS assumes a fixed occupied-orbital count at every k, which
            # global T=0 Aufbau no longer guarantees once bands overlap or a
            # Fermi degeneracy is fractionally filled. Fail closed rather
            # than extrapolate against an occupation pattern KDIIS cannot
            # represent.
            if (
                getattr(opts, "scf_accelerator", None) == SCFAccelerator.KDIIS
                and not _is_per_k_integer_aufbau(_global_occupations(energies))
            ):
                raise NotImplementedError(
                    "private multi-k slab GDF: KDIIS requires a fixed number "
                    "of occupied orbitals at every k point and cannot be "
                    "used after global T=0 Aufbau detects band overlap or a "
                    "fractionally occupied Fermi degeneracy. Use the default "
                    "EDIIS_DIIS/DIIS accelerator, or explicit "
                    "finite-temperature smearing for a metal."
                )
            extrapolated = accel.extrapolate_rhf(
                focks,
                error_k_list=errors,
                density_k_list=densities,
                energy=energy,
                mo_coeffs_k_list=coefficients,
                n_occ=n_occ,
                weights=list(weights),
                cells=cells,
                kpoints=list(kpoints),
            )
            if diis_active:
                focks = extrapolated
        new_coefficients, new_energies = diagonalise(focks)
        new_densities = densities_from_coefficients(
            new_coefficients, new_energies
        )
        if damping != 0.0 and not diis_active:
            densities = [
                damping * old + (1.0 - damping) * new
                for old, new in zip(densities, new_densities)
            ]
        else:
            densities = new_densities
        coefficients = new_coefficients
        energies = new_energies
        if damper is not None:
            damper.update(energy)
        e_previous = energy

    plog.converged(
        n_iter=max_iter,
        energy=e_electronic + e_nuclear,
        converged=False,
    )
    return _PrivateSlabKRHFGDFResult(
               restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy(),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=e_electronic + e_nuclear,
        e_electronic=e_electronic,
        e_nuclear=e_nuclear,
        n_iter=max_iter,
        converged=False,
        mo_energies=energies,
        mo_coeffs=coefficients,
        density=densities,
        fock=focks,
        overlap=overlaps,
        hcore=hcores,
        kpoints_cart=kpoints,
        kpoint_weights=weights,
        scf_trace=trace,
        functional=func_name or None,
        e_xc=e_xc,
        e_coulomb=e_coulomb,
        e_hf_exchange=e_hf_exchange,
        aux_basis_name=aux_name,
        n_aux=int(raw_aux.nbasis),
        backend=(
            "private-multik-slab-truncated-gdf-rks"
            if is_ks
            else "private-multik-slab-truncated-gdf-rhf"
        ),
    )


# Closed-shell RKS via GDF -- the RHF/GDF driver already dispatches to
# the KS branch when ``options`` is :class:`PeriodicKSOptions` (or
# ``functional=`` is set); this alias gives the call site the right
# name for what it does.
run_rks_periodic_gamma_gdf = run_rhf_periodic_gamma_gdf
