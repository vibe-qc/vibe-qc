"""Phase 15c: Γ-point periodic RKS SCF driver using the composed
Ewald-3D Coulomb dispatch.

DFT counterpart of :func:`run_rhf_periodic_gamma_ewald3d`. Replaces
the HF exchange ``-K(D)/2`` with the libxc XC potential
``V_xc[r(D)]`` from :func:`build_xc_periodic`, while keeping the
w-invariant Hartree J on the FFT-Poisson Ewald solver. For hybrid
functionals, exact exchange enters through the CAM assembly

    K_HF = c_full * K_full + c_sr * K_erfc(omega_screen)

resolved by :func:`vibeqc.periodic_screened_exchange.resolve_periodic_exchange`
(global hybrids: ``c_full = hf_exchange_fraction``; screened hybrids
like hse06: pure short-range erfc exchange; LR-heavy range-separated
functionals fail closed).

What this driver does
---------------------

At every SCF iteration:

    F  =  H_core  +  J_ewald(w, D)  +  V_xc[r(D)]  -  K_HF(D)/2

with H_core = T + V at Γ (Bloch-summed lattice integrals), S the
Γ-point overlap, and canonical-orthogonalisation as in the RHF
driver.

Density flow for V_xc
---------------------

``build_xc_periodic`` consumes a :class:`LatticeMatrixSet` density
that, at every grid point r, evaluates
``r(r) = S_g S_{muν} D(g)_{muν} chi_mu(r) chi_ν(r-g)``. For a Γ-only
molecular-limit cell, the physical density lives in the home cell
only (``D(g!=0) ≈ 0``), so we wrap the Γ-folded D in a degenerate
:class:`LatticeMatrixSet` with ``block[0] = D`` and zeros elsewhere.
This is consistent with the Γ-only Ewald assumption already enforced
by :func:`build_j_ewald_3d` (which expects molecular-limit cells --
see the docstring of :func:`run_rhf_periodic_gamma_ewald3d`).

Energy formula
--------------

Following the C++ RKS DIRECT_TRUNCATED driver (cpp/src/periodic_scf.cpp):

    E_elec  =  E_xc  +  tr(D . H_core)  +  1/2 tr(D . F_HF_part)

where ``F_HF_part = J - (a/2) K``. For pure DFT (``a = 0``) the K
trace is skipped entirely (and the K build is skipped too). For
a = 1, this reduces exactly to the RHF formula.

Validation targets
------------------

* **w-invariance**: total SCF energy stable across w in [0.3, 2.0]
  to ~µHa, exactly as the RHF Ewald driver.
* **Cross-check against DIRECT_TRUNCATED at molecular limit**: for
  a closed-shell molecule in a ~30 bohr vacuum box, the EWALD_3D
  and DIRECT_TRUNCATED RKS drivers must agree to within Makov-Payne
  finite-box corrections (~O(1/L^3), sub-mHa at this box size).

Slab-Ewald sibling
------------------

``run_rks_periodic_gamma_ewald2d`` reuses this SCF loop with the rigorous
2D-slab ``V_ne``, ``E_nn``, and Hartree ``J`` blocks selected by
``CoulombMethod.SLAB_EWALD_2D``. The FFT grid controls remain Ewald-3D-only.

Scope
-----

* **Γ-only** at a single k-point (0, 0, 0). Multi-k RKS Ewald is a
  follow-up, parallel to the RHF / UHF multi-k extensions.
* **Closed-shell** RKS (``multiplicity = 1``, even ``n_electrons``).
  The UKS counterpart lands separately.
* **DIIS** supported (default on); plain damping when ``use_diis =
  False``. Saunders-Hillier level shift through ``opts.level_shift``.
"""

from __future__ import annotations

from .guess import periodic_result_selection

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .periodic.exchange_convention import MOLECULAR_LIMIT
from ._vibeqc_core import (
    apply_level_shift,
    BasisSet,
    bloch_sum,
    build_grid,
    build_xc_periodic,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    CoulombMethod,
    EwaldOptions,
    Functional,
    GridOptions,
    InitialGuess,
    LatticeMatrixSet,
    LatticeSumOptions,
    level_shift_at_iter,
    LevelShiftDensity,
    nuclear_repulsion_per_cell,
    PeriodicKSOptions,
    PeriodicSystem,
    SCFIteration,
    XCKind,
)
from .ewald_composed import make_ewald_3d_gamma_j_builder
from .ewald_composed_slab import make_slab_ewald_2d_gamma_j_builder
from .ewald_j import auto_grid
from ._vibeqc_core import build_jk_gamma_molecular_limit
from .guess import _real_gamma_seed, initial_density_closed_shell
from .periodic_grid import build_periodic_becke_grid
from .periodic_rhf_ewald import _canonical_orthogonalizer, _refuse_if_dense_ionic
from .periodic_scf_accelerators import DynamicDamping, PeriodicSCFAccelerator
from .periodic_screened_exchange import (
    resolve_periodic_exchange,
)
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .fock_state import FockBuild
from .periodic_ks_fock import _GammaRKSBuilder, _gamma_ks_source_details
from .solvers.operators import OperatorFrame, _snapshot

__all__ = [
    "PeriodicRKSEwaldResult",
    "run_rks_periodic_gamma_ewald2d",
    "run_rks_periodic_gamma_ewald3d",
]


@dataclass
class PeriodicRKSEwaldResult:
    """Result of :func:`run_rks_periodic_gamma_ewald3d`.

    Structurally parallel to :class:`PeriodicKSResult` (the C++
    DIRECT_TRUNCATED RKS result) but carries Ewald-specific
    bookkeeping (``omega``, ``grid_shape``) and a separate XC-energy
    field. All matrices are at Γ (no Bloch sums).
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    e_xc: float
    e_coulomb: float
    e_hf_exchange: float
    n_iter: int
    converged: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    functional: str = ""
    scf_trace: List[SCFIteration] = field(default_factory=list)
    # Ewald-specific:
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None

    # Every Gamma-only Ewald/GDF driver builds exchange with the
    # molecular-limit kernel (home-cell density, image blocks zeroed); the
    # runner's manifest records this instead of the exxdiv knob (GitLab #23).
    exchange_q0: str = MOLECULAR_LIMIT
    fock_build: FockBuild | None = None
    occupations: np.ndarray | None = None
    next_density: np.ndarray | None = None
    density_fixed_point_residual: float | None = None
    evaluated_build_history: list[dict] = field(default_factory=list)

    @property
    def run_fields(self):
        if self.fock_build is None:
            return {}
        return dict(self.fock_build.output_fields(),
                    orbital_state_relation="density-defining-physical-rayleigh-quotients",
                    scf_stop_reason="converged" if self.converged else "iteration_limit",
                    density_fixed_point_residual=self.density_fixed_point_residual,
                    evaluated_build_history=[dict(row) for row in self.evaluated_build_history])



def _density_set_gamma(template: LatticeMatrixSet, D: np.ndarray) -> LatticeMatrixSet:
    """Return a degenerate :class:`LatticeMatrixSet` with ``D`` in
    the home-cell block and zeros elsewhere, copying the
    cells / nbf layout of ``template`` (typically the overlap
    lattice set).

    For Γ-only molecular-limit cells, the physical density lives in
    the home cell, so this wrapper is exact for the Ewald-Γ regime.
    Tight cells where image-cell density blocks are non-negligible
    need the multi-k driver.

    Mutates the template's ``blocks`` in place via
    :meth:`LatticeMatrixSet.set_block` (the only Python-visible way
    to mutate the C++ vector -- assigning ``.blocks[i]`` writes to a
    transient list and silently no-ops at the C++ level).
    """
    n_bf = D.shape[0]
    if n_bf != template.nbf:
        raise ValueError(
            f"_density_set_gamma: D has nbf={n_bf} but template has nbf={template.nbf}"
        )
    zero = np.zeros_like(D)
    for i in range(len(template)):
        template.set_block(i, D if i == 0 else zero)
    return template


def _empty_lattice_set_like(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts,
) -> LatticeMatrixSet:
    """Build a fresh LatticeMatrixSet with the canonical (cells, nbf)
    layout -- by computing a throwaway overlap lattice and re-using
    its structure. Cheap relative to the SCF itself, and robust
    against future cell-list changes."""
    return compute_overlap_lattice(basis, system, lat_opts)


def _resolve_slab_ewald_alpha(
    lat_opts: LatticeSumOptions,
    *,
    alpha: Optional[float],
    omega: Optional[float],
    driver: str,
) -> float:
    """Resolve the 2D slab Ewald split parameter.

    ``omega`` is accepted as a positive alias so the generic dispatcher keyword
    can keep working for the slab route.
    """
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


def run_rks_periodic_gamma_ewald3d(
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
) -> PeriodicRKSEwaldResult:
    """Γ-point closed-shell periodic Kohn-Sham SCF with Ewald-3D
    Coulomb.

    Hartree ``J`` uses the w-invariant composed split from
    :func:`build_j_ewald_3d`. ``V_xc`` is built on the periodic
    Becke grid via :func:`build_xc_periodic`. For hybrid
    functionals exact exchange is added per the CAM assembly of
    :func:`vibeqc.periodic_screened_exchange.resolve_periodic_exchange`
    (full-range K at w = 0 for global hybrids -- same K-build
    convention as the RHF Ewald driver -- plus/or the erfc-screened
    K at w = omega_screen for screened hybrids like hse06).

    Parameters
    ----------
    system
        :class:`PeriodicSystem` with a full-rank 3D lattice matrix.
    basis
        AO basis for the unit cell.
    options
        Optional :class:`PeriodicKSOptions`. Defaults: PBE
        functional, default DIIS, no level shift. The
        ``use_periodic_becke`` flag selects between the periodic
        Becke partition (default since v0.9.x; required for tight
        cells, reduces to the molecular partition in the
        molecular-limit regime) and the legacy molecular partition
        (silently wrong on tight cells; set False only to reproduce
        v0.8.x numerics).
    omega
        Ewald splitting parameter; result is w-independent at
        convergence to ~µHa across w in [0.3, 2.0].
    grid_shape, origin, spacing_bohr
        FFT-Poisson grid controls forwarded to :func:`build_j_ewald_3d`.
    linear_dep_threshold
        Overlap-eigenvalue cutoff for canonical orthogonalisation.
    allow_dense_ionic
        Internal escape hatch. The driver fails closed when periodic images
        of the basis overlap (max cross-cell AO overlap above ~0.3), where
        the molecular-limit density convention gives a ~2 Ha-wrong energy
        (CLAUDE.md Sec.7); use ``jk_method='gdf'``/``'bipole'``/``'gpw'``
        instead. Set ``True`` only to exercise the driver's mechanics on
        such a cell -- the returned energy is then NOT reliable.

    Returns
    -------
    :class:`PeriodicRKSEwaldResult`.
    """
    opts = options if options is not None else PeriodicKSOptions()
    max_iter = int(opts.max_iter)
    if max_iter < 0:
        raise ValueError("run_rks_periodic_gamma_ewald3d: max_iter must be >= 0 (0 evaluates the initial guess only)")
    from .guess import select_periodic_driver_guess
    selection = select_periodic_driver_guess(
        system, opts, route="ewald", method="RKS", driver="run_rks_periodic_gamma_ewald3d",
    )
    lat_opts = opts.lattice_opts
    slab_mode = lat_opts.coulomb_method == CoulombMethod.SLAB_EWALD_2D

    if slab_mode:
        if system.dim != 2:
            raise ValueError(
                "run_rks_periodic_gamma_ewald2d: SLAB_EWALD_2D requires "
                f"dim == 2; got dim = {system.dim}"
            )
        if float(getattr(opts, "smearing_temperature", 0.0)) > 0.0:
            raise NotImplementedError(
                "run_rks_periodic_gamma_ewald2d: Fermi-Dirac smearing requires "
                "the multi-k Ewald machinery, which is not yet available for "
                "SLAB_EWALD_2D"
            )
        omega = _resolve_slab_ewald_alpha(
            lat_opts,
            alpha=None,
            omega=omega,
            driver="run_rks_periodic_gamma_ewald2d",
        )
    # ---- Force EWALD_3D gauge ----
    elif lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
        lat_opts.coulomb_method = CoulombMethod.EWALD_3D
    # Tight-cell cutoff floor: small 3D cells need a large real-space cutoff
    # for the EWALD_3D lattice sums to converge (and for S(Γ) to be PSD).
    # This is an auto-tuning heuristic, so it is gated on
    # auto_optimize_truncation -- a user who explicitly opts out
    # (auto_optimize_truncation=False) gets their cutoff verbatim, and the
    # linear-dependence preflight below is then free to refuse a non-PSD
    # overlap rather than have it silently inflated away (CLAUDE.md Sec.7).
    if (
        not slab_mode
        and auto_optimize_truncation
        and lat_opts.cutoff_bohr < 18.0
        and system.dim == 3
    ):
        V_cell = float(abs(np.linalg.det(np.asarray(system.lattice))))
        if V_cell < 1000.0:
            lat_opts.cutoff_bohr = max(lat_opts.cutoff_bohr, 18.0)
            lat_opts.nuclear_cutoff_bohr = max(lat_opts.nuclear_cutoff_bohr, 25.0)

    # Fail closed if periodic images of the basis overlap enough to make the
    # molecular-limit energy wrong (CLAUDE.md Sec.7); evaluated at the resolved
    # lattice cutoff. GDF/BIPOLE/GPW are correct there.
    if not slab_mode:
        _refuse_if_dense_ionic(system, basis, lat_opts, allow_dense_ionic)

    plog = resolve_progress(progress, verbose=verbose)

    # ---- Ewald splitting parameter w -- must match nuclear a ----
    # When not specified explicitly, use the default Ewald splitting
    # a = 2.8.V^{-1/3} (matching the BIPOLE driver; CRYSTAL convention).
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
            f"RKS Gamma SLAB_EWALD_2D / functional={opts.functional!r}, "
            f"alpha = {float(omega):.3f}"
        )
    else:
        plog.info(
            f"RKS Gamma EWALD_3D / functional={opts.functional!r}, "
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

    # Closed-shell check.
    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "run_rks_periodic_gamma_ewald3d: closed-shell RKS "
            f"requires even electron count; got {n_elec}"
        )
    if system.multiplicity != 1:
        raise ValueError(
            "run_rks_periodic_gamma_ewald3d: closed-shell RKS requires "
            f"multiplicity=1; got {system.multiplicity}"
        )
    n_occ = n_elec // 2

    # ---- Functional ------------------------------------------------------
    func = Functional(opts.functional, 1)  # spin-unpolarised RKS
    # Exact-exchange assembly K_HF = c_full*K_full + c_sr*K_erfc(w_s).
    # Global hybrids give (hf_exchange_fraction, 0); screened hybrids
    # (hse06) give (0, 0.25) with omega_screen = 0.11; LR-heavy RSH
    # fails closed. NOTE: omega_screen is the functional's physical
    # screening parameter -- unrelated to the ``omega`` variable in
    # this driver, which is the numerical Ewald split alpha.
    exx = resolve_periodic_exchange(func, where="run_rks_periodic_gamma_ewald3d")

    # ---- Auto-optimise lattice truncation (default ON) -------------------
    # Find the loosest lattice / Schwarz settings that keep S(Γ) PSD,
    # so the user doesn't have to hand-tune ``cutoff_bohr`` /
    # ``schwarz_threshold`` to make the SCF converge on tight crystals.
    # See vibeqc.eigs_preflight.optimize_truncation. Short-circuits
    # at 1 evaluation when starting settings already pass the
    # preflight; non-short-circuit overhead is typically a handful of
    # extra S(Γ) builds, << SCF cost.
    if auto_optimize_truncation and lat_opts.coulomb_method == CoulombMethod.EWALD_3D:
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
                    "with the last evaluated settings. Check the report above "
                    "and consider vq.disambiguate_critical_overlap (basis vs "
                    "screening) and / or vq.make_basis(..., exp_to_discard=...)."
                )
            lat_opts = opt_rep.optimized_lattice_opts

    # ---- One-electron integrals at Γ -------------------------------------
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

    # ---- Linear-dependence preflight on S(Γ) -----------------------------
    # See periodic_rhf_ewald.py for the rationale.
    from .linear_dependence import scf_preflight_overlap_check

    scf_preflight_overlap_check(
        S,
        plog=plog,
        label="S(Γ)",
        basis=basis,
    )

    # ---- DFT grid --------------------------------------------------------
    # Built AFTER the linear-dependence preflight: on tight cells the
    # periodic Becke grid construction dominates setup (~25 s on the LiH
    # conventional cell), so a non-PSD-overlap abort (auto_optimize_
    # truncation=False) must not pay for it. The grid is not used until the
    # XC build below.
    if opts.use_periodic_becke:
        grid = build_periodic_becke_grid(
            system,
            grid_options=opts.grid,
            image_radius_bohr=float(opts.becke_image_radius_bohr),
        )
        # The periodic-Becke grid (one-cell partition of the crystal)
        # pairs with the Γ-torus density set (D in every lattice block,
        # build_xc_periodic's cross-cell mode). The home-cell-only set
        # makes build_xc_periodic fall back to its molecular-limit mode
        # (rho = chi_0 D chi_0, no image AO products): exact in a vacuum
        # box, ~0.3 Ha wrong on tight ionic cells (the 2026-07-09 KRKS
        # finding, the original investigation).
        from .periodic_rhf_gdf import _density_set_torus_gamma

        _set_xc_density = _density_set_torus_gamma
    else:
        grid = build_grid(system.unit_cell_molecule(), opts.grid)
        # Molecular Becke grid (v0.8.x reproduction) pairs with the
        # molecular-limit home-cell-only density.
        _set_xc_density = _density_set_gamma

    # ---- Canonical orthogonalisation -------------------------------------
    X, n_kept = _canonical_orthogonalizer(
        S,
        linear_dep_threshold,
        normalize_diag_first=canonical_orth_normalize_diag_first,
    )
    if n_occ > n_kept:
        raise RuntimeError(
            "run_rks_periodic_gamma_ewald3d: canonical orthogonalisation "
            f"dropped too many directions (n_occ = {n_occ}, "
            f"n_kept = {n_kept}); loosen linear_dep_threshold or pick "
            "a less redundant basis."
        )

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    from .solvers._scf import FockDiagonalizer

    n_required = n_kept if int(getattr(opts, "quadratic_fallback_iter", 0)) > 0 else n_occ
    seed_diagonalise = FockDiagonalizer(
        X, S, opts, kind="one_electron", n_required=n_required, real_only=True,
    )

    # ---- Initial guess via the unified engine ---------------------------
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

    guess = selection.effective
    seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
    D_engine = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        seed_guess,
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
        D = _real_gamma_seed(D_engine, "run_rks_periodic_gamma_ewald3d")
        C0, eps0 = seed_diagonalise(Hcore)  # MO trace for log symmetry
    else:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise)")
        C0, eps0 = seed_diagonalise(Hcore)
        D = 2.0 * C0[:, :n_occ] @ C0[:, :n_occ].T
        D = 0.5 * (D + D.T)
    # Pre-allocate a reusable LatticeMatrixSet for the V_xc density
    # input. We mutate its blocks each iteration via set_block().
    D_set = _empty_lattice_set_like(basis, system, lat_opts)

    # Memory cliff guard -- build_xc_periodic materialises AO values
    # per cell.  Fail fast on large systems instead of SIGKILL.
    from .periodic_rks_multi_k_ewald import _guard_legacy_periodic_xc_memory

    _guard_legacy_periodic_xc_memory(
        n_grid=int(np.asarray(grid.points).shape[0]),
        nbf=int(basis.nbasis),
        n_cells=len(D_set),
        is_gga=(func.kind == XCKind.GGA),
        functional=str(opts.functional),
    )

    import hashlib
    import json

    def source_details(*, hf_seed=False):
        return _gamma_ks_source_details(
            basis, system, lat_opts, opts, func, exx, grid, D_set,
            omega=omega, grid_shape=grid_shape_t, origin=origin,
            spacing_bohr=spacing_bohr, ewald_options=eopts, hf_seed=hf_seed)

    source_basis = source_details()
    frame = OperatorFrame(
        n=basis.nbasis, basis_id=hashlib.sha256(json.dumps(
            source_basis["shells"], sort_keys=True).encode()).hexdigest(),
        ordering=source_basis["ordering"], image_gauge="gamma-bloch-at-declared-shell-origins",
        overlap=S, X=X)
    X, S = frame.X, frame.overlap
    diagonalise = FockDiagonalizer(
        X, S, opts, kind="ks", n_required=n_required, real_only=True, frame=frame)
    build_fock = _GammaRKSBuilder(
        basis, system, lat_opts, T, V, frame, e_nuc, j_build,
        func, exx, grid, D_set, _set_xc_density, source_details)
    T, V, Hcore = build_fock.kinetic, build_fock.nuclear, build_fock.hcore
    if guess == InitialGuess.PATOM:
        # PATOM is a full-HF state, independent of the target XC and its
        # eigensolver recycle history.
        seed = build_fock(D, hf_seed=True)
        hf_diagonalise = FockDiagonalizer(
            X, S, opts, kind="fock", n_required=n_required, real_only=True, frame=frame)
        C0, eps0 = hf_diagonalise(seed.channels[0].operator)
        D = 2.0 * C0[:, :n_occ] @ C0[:, :n_occ].T
        D = 0.5 * (D + D.T)
    D_prev = D.copy()

    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_rks_periodic_gamma_ewald3d: damping must be in [0, 1); got {damping}"
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
    # Explicit per-iteration schedule (unified with the molecular
    # drivers). Empty ⇒ the constant `level_shift` above; non-empty ⇒
    # resolved per iteration by the shared C++ helper.
    _ls_schedule = list(getattr(opts, "level_shift_schedule", None) or [])
    _ls_max_iter = int(opts.max_iter)

    # Phase C1c -- quadratic SCF fallback (see periodic_rhf_ewald
    # for the full description).
    quadratic_fallback_iter = int(getattr(opts, "quadratic_fallback_iter", 0))
    quadratic_fallback_shift = float(getattr(opts, "quadratic_fallback_shift", 0.1))
    quadratic_fallback_max_step = float(
        getattr(opts, "quadratic_fallback_max_step", 0.1)
    )

    # Track MO basis between iterations so the C1c Newton step has a
    # current MO frame to operate in.
    C_prev_mo = C0
    eps_prev_mo = eps0

    # ---- SCF loop --------------------------------------------------------
    scf_trace: List[SCFIteration] = []
    result = PeriodicRKSEwaldResult(
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
        mo_energies=np.empty(0),
        mo_coeffs=np.empty((0, 0)),
        density=D.copy(),
        fock=np.empty((0, 0)),
        overlap=S.copy(),
        functional=str(opts.functional),
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
    )

    E_prev = 0.0
    scf_label = "SLAB_EWALD_2D" if slab_mode else "EWALD_3D"
    plog.banner(f"SCF (RKS Gamma {opts.functional!r}, {scf_label})")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    # Cache the iteration-invariant analytic-FT Hartree-J machinery once
    # (Bloch AO-pair FT, dense G-mesh, 4pi/G^2 kernel) so the SCF loop redoes
    # only the density contraction -- the GDF "build Lpq once" pattern; E2 in
    # docs/pbc_audit_2026-06.md. ``j_build(D)`` is bit-identical to a
    # per-iteration build_j_ewald_3d call (grid backend: uncached, unchanged).

    if max_iter == 0:
        # max_iter=0 is the documented NOITER path (docs/user_guide/
        # from_orca.md; the #392 contract of the molecular routes): evaluate
        # the initial guess once, explicitly unconverged, with no SCF
        # iteration. As there, the returned density is the guess itself and
        # the orbitals diagonalise its evaluated Fock.
        terminal = build_fock(D)
        channel = terminal.channels[0]
        C_next, _ = diagonalise(channel.operator)
        D_next = 2.0 * C_next[:, :n_occ] @ C_next[:, :n_occ].T
        D_next = .5 * (D_next + D_next.T)
        result.fock_build = terminal
        result.next_density = _snapshot(D_next)
        result.density_fixed_point_residual = float(np.max(np.abs(D_next - D)))
        result.energy, result.e_electronic = terminal.energy.total, terminal.energy.electronic
        terminal_components = dict(terminal.energy.components)
        result.e_xc = terminal_components["xc"]
        result.e_coulomb = terminal_components["coulomb"]
        result.e_hf_exchange = terminal_components["exchange"]
        result.mo_coeffs = C_next
        result.mo_energies = np.real(np.diag(C_next.T @ channel.operator.matrix @ C_next))
        result.occupations = np.zeros(C_next.shape[1])
        result.occupations[:n_occ] = 2.0
        result.density = channel.density.copy()
        result.fock = channel.operator.matrix.copy()
        result.n_iter, result.converged = 0, False
        plog.converged(n_iter=0, energy=result.energy, converged=False)
        return result

    for iter_idx in range(1, max_iter + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter
        D_used = (
            D
            if (iter_idx == 1 or damping == 0.0 or diis_active)
            else damping * D_prev + (1.0 - damping) * D
        )

        physical = build_fock(D_used)
        F = physical.channels[0].operator.matrix
        E_elec, E_total = physical.energy.electronic, physical.energy.total
        components = dict(physical.energy.components)
        E_xc, E_coulomb, E_hf_K = (components[key] for key in ("xc", "coulomb", "exchange"))
        E_madelung_fix = dict(physical.energy.corrections)["madelung"]
        FDS = F @ D_used @ S
        grad = FDS - FDS.T
        grad_norm = physical.commutator_norm

        dE = E_total - E_prev
        scf_trace.append(
            SCFIteration(
                iter=iter_idx,
                energy=float(E_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm),
                diis_subspace=(accel.subspace_size if accel is not None else 0),
            )
        )
        result.evaluated_build_history.append(dict(
            iteration=iter_idx, generation=physical.generation,
            build_id=physical.identity, density_id=physical.density_id))
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=(accel.subspace_size if accel is not None else 0),
        )
        # Per-iter energy decomposition (E_kin / E_ne / E_J / E_xc /
        # E_K / E_nuc / E_madelung). Always emitted at level 4+ for
        # cross-code (PySCF) parity comparison and bug localisation.
        E_kin_iter = components["kinetic"]
        E_ne_iter = components["nuclear_attraction"]
        plog.energy_decomposition(
            iter_idx,
            E_kin=E_kin_iter,
            E_ne=E_ne_iter,
            E_J=E_coulomb,
            E_xc=float(E_xc),
            E_K=E_hf_K,
            E_nuc=float(e_nuc),
            E_madelung=float(E_madelung_fix),
        )
        # Divergence detection (v0.5.6, generalised v0.6.2 to a shared
        # helper applied across all 8 periodic SCF entry points).
        check_scf_divergence(
            "run_rks_periodic_gamma_ewald3d",
            iter_idx,
            E_total,
            grad_norm,
            dE,
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

            C_new, eps_new = quadratic_step(
                F,
                C_prev_mo,
                eps_prev_mo,
                n_occ,
                shift=quadratic_fallback_shift,
                max_step=quadratic_fallback_max_step,
            )
        else:
            # SCF-accelerator extrapolation (DIIS / KDIIS / EDIIS /
            # EDIIS_DIIS / ADIIS). The convergence check above uses
            # the pre-extrapolation F so dE / grad_norm reflect raw
            # SCF progress. ``C_prev_mo`` / ``eps_prev_mo`` from the
            # previous diag give KDIIS its canonical-MO basis.
            if accel is not None:
                F_ex = accel.extrapolate_rhf(
                    F,
                    error=grad,
                    density=D_used,
                    energy=E_total,
                    mo_coeffs=C_prev_mo,
                    mo_energies=eps_prev_mo,
                    n_occ=n_occ,
                )
                if diis_active:
                    F = F_ex

            # Saunders-Hillier level shift.
            b = (
                level_shift_at_iter(
                    level_shift, 0, _ls_schedule, _ls_max_iter, iter_idx
                )
                if _ls_schedule
                else level_shift
            )
            # Shared Saunders-Hillier operator: the weight on S·D·S is
            # fixed by the density convention, not by taste. ``D_used`` is
            # the closed-shell total density (occupations in {0, 2}), hence
            # TOTAL. Returns F untouched when b == 0.
            # Guarded: skip the pybind conversion of S and D entirely on an
            # unshifted cycle, which is the default and the common case.
            F_diag = (
                F if b == 0.0
                else apply_level_shift(F, S, D_used, b, LevelShiftDensity.TOTAL)
            )

            solver_fock = physical.solver_operator(
                0, F_diag, modification="method-acceleration-level-shift")
            C_new, eps_new = diagonalise(solver_fock)

        C_prev_mo = C_new
        eps_prev_mo = eps_new
        D_prev = D_used
        D = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T
        D = 0.5 * (D + D.T)

        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

        result.n_iter = iter_idx
        if converged or iter_idx == max_iter:
            # The orbitals define the returned density. Evaluate the physical
            # Fock/energy on exactly that density, including at an iteration
            # cap. A further eigensolve is a look-ahead density, not this state.
            terminal = build_fock(D)
            channel = terminal.channels[0]
            C_next, _ = diagonalise(channel.operator)
            D_next = 2.0 * C_next[:, :n_occ] @ C_next[:, :n_occ].T
            D_next = .5 * (D_next + D_next.T)
            closure = float(np.max(np.abs(D_next - D)))
            result.fock_build = terminal
            result.next_density = _snapshot(D_next)
            result.density_fixed_point_residual = closure
            result.energy, result.e_electronic = terminal.energy.total, terminal.energy.electronic
            terminal_components = dict(terminal.energy.components)
            result.e_xc = terminal_components["xc"]
            result.e_coulomb = terminal_components["coulomb"]
            result.e_hf_exchange = terminal_components["exchange"]
            result.mo_coeffs = C_new
            result.mo_energies = np.real(np.diag(C_new.T @ channel.operator.matrix @ C_new))
            result.occupations = np.zeros(C_new.shape[1])
            result.occupations[:n_occ] = 2.0
            result.density = channel.density.copy()
            result.fock = channel.operator.matrix.copy()
            terminal_converged = (
                converged
                and abs(result.energy - E_total) < float(opts.conv_tol_energy)
                and terminal.commutator_norm < float(opts.conv_tol_grad)
                and closure < float(opts.conv_tol_grad))
            if terminal_converged or iter_idx == max_iter:
                previous = scf_trace[-1]
                scf_trace[-1] = SCFIteration(
                    iter=previous.iter, energy=result.energy,
                    delta_e=result.energy - previous.energy,
                    grad_norm=terminal.commutator_norm,
                    diis_subspace=previous.diis_subspace)
                result.evaluated_build_history[-1] = dict(
                    iteration=iter_idx, generation=terminal.generation,
                    build_id=terminal.identity, density_id=terminal.density_id)
            if terminal_converged:
                result.converged = True
                plog.converged(n_iter=result.n_iter, energy=result.energy, converged=True)
                return result

    result.converged = False
    plog.converged(
        n_iter=result.n_iter,
        energy=result.energy,
        converged=False,
    )
    return result


def run_rks_periodic_gamma_ewald2d(
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
) -> PeriodicRKSEwaldResult:
    """Gamma-point closed-shell RKS with rigorous 2D slab Ewald.

    ``V_ne``, ``E_nn``, and Hartree ``J`` all use the Parry/de Leeuw slab
    Ewald gauge and the shared ``slab_ewald_alpha`` split parameter. Hybrid
    functionals keep the existing full-range Gamma molecular-limit exchange
    build, as in the RHF slab driver.
    """
    opts = options if options is not None else PeriodicKSOptions()
    lat_opts = opts.lattice_opts
    if system.dim != 2:
        raise ValueError(
            "run_rks_periodic_gamma_ewald2d: SLAB_EWALD_2D requires "
            f"dim == 2; got dim = {system.dim}"
        )
    lat_opts.coulomb_method = CoulombMethod.SLAB_EWALD_2D
    slab_alpha = _resolve_slab_ewald_alpha(
        lat_opts,
        alpha=alpha,
        omega=omega,
        driver="run_rks_periodic_gamma_ewald2d",
    )
    return run_rks_periodic_gamma_ewald3d(
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
