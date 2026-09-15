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
from .guess import initial_density_closed_shell
from .madelung import (
    madelung_energy_correction as _madelung_energy_correction,
)
from .periodic_grid import build_periodic_becke_grid
from .periodic_rhf_ewald import _canonical_orthogonalizer, _refuse_if_dense_ionic
from .periodic_scf_accelerators import DynamicDamping, PeriodicSCFAccelerator
from .periodic_screened_exchange import (
    build_exchange_gamma,
    resolve_periodic_exchange,
)
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence

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
        # finding, handovers/HANDOVER_AICCM_DIRECT_TORUS.md §4).
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
        D = D_engine
        C0, eps0 = diagonalise(Hcore)  # MO trace for log symmetry
    else:
        plog.info(f"initial guess: {guess.name} (Hcore-diagonalise)")
        C0, eps0 = diagonalise(Hcore)
        D = 2.0 * C0[:, :n_occ] @ C0[:, :n_occ].T
        D = 0.5 * (D + D.T)
    if guess == InitialGuess.PATOM:
        # PATOM is one full-HF in-field step, independent of the target XC.
        J_seed = j_build(D)
        K_seed = np.asarray(build_jk_gamma_molecular_limit(
            basis, system, lat_opts, D, 0.0).K)
        C0, eps0 = diagonalise(Hcore + J_seed - 0.5 * K_seed)
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
        overlap=S,
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

    for iter_idx in range(1, int(opts.max_iter) + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and iter_idx >= diis_start_iter
        D_used = (
            D
            if (iter_idx == 1 or damping == 0.0 or diis_active)
            else damping * D_prev + (1.0 - damping) * D
        )

        # Hartree J via composed Ewald-3D (cached analytic-FT machinery).
        J = j_build(D_used)

        # Coefficient-folded exact exchange (full-range and/or erfc-
        # screened per the CAM assembly). None for pure DFT.
        K = build_exchange_gamma(basis, system, lat_opts, D_used, exx)

        # V_xc via libxc on the KS grid. The density-set convention is
        # paired with the grid choice above (torus set on the periodic
        # grid, home-cell-only on the molecular grid).
        _set_xc_density(D_set, D_used)
        xc_contrib = build_xc_periodic(
            basis,
            system,
            grid,
            func,
            D_set,
            lat_opts,
        )
        # V_xc lives in the LatticeMatrixSet -- Bloch-fold to Γ.
        V_xc = np.real(bloch_sum(xc_contrib.V_xc, k_gamma))
        V_xc = 0.5 * (V_xc + V_xc.T)
        E_xc = float(xc_contrib.e_xc)

        # F_HF_part = J - K_HF/2 (K already carries the assembly
        # coefficients).
        if K is not None:
            F_HF_part = J - 0.5 * K
        else:
            F_HF_part = J

        F = Hcore + F_HF_part + V_xc
        F = 0.5 * (F + F.T)

        # Energy: E_elec = E_xc + tr(D.Hcore) + 1/2 tr(D.F_HF_part)
        E_HF_trace = 0.5 * float(np.einsum("ij,ij->", D_used, F_HF_part))
        E_core_trace = float(np.einsum("ij,ij->", D_used, Hcore))
        E_elec = E_xc + E_core_trace + E_HF_trace
        # Madelung-leak correction (v0.6.1) -- disabled for EWALD_3D in
        # v0.7.0 since V_ne now dispatches to the gauge-aligned Ewald path.
        if lat_opts.coulomb_method in (
            CoulombMethod.EWALD_3D,
            CoulombMethod.SLAB_EWALD_2D,
        ):
            E_madelung_fix = 0.0
        else:
            E_madelung_fix = _madelung_energy_correction(
                D_used,
                S,
                system,
                nuclear_uses_ewald=False,
            )
        E_total = E_elec + e_nuc + E_madelung_fix

        # Coulomb / HF-exchange decomposition for reporting.
        E_coulomb = 0.5 * float(np.einsum("ij,ij->", D_used, J))
        E_hf_K = (
            -0.25 * float(np.einsum("ij,ij->", D_used, K))
            if K is not None
            else 0.0
        )

        # Orbital-gradient norm.
        FDS = F @ D_used @ S
        grad = FDS - FDS.T
        grad_norm = float(np.linalg.norm(grad))

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
        E_kin_iter = float(np.einsum("ij,ij->", D_used, T))
        E_ne_iter = float(np.einsum("ij,ij->", D_used, V))
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

            C_new, eps_new = diagonalise(F_diag)

        C_prev_mo = C_new
        eps_prev_mo = eps_new
        D_prev = D_used
        D = 2.0 * C_new[:, :n_occ] @ C_new[:, :n_occ].T
        D = 0.5 * (D + D.T)

        # Stash latest state.
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

        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

        if converged:
            # Final consistency pass on the fresh D (matches RHF
            # driver convention).
            J_f = j_build(D)
            K_f = build_exchange_gamma(basis, system, lat_opts, D, exx)
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
            if K_f is not None:
                F_HF_f = J_f - 0.5 * K_f
            else:
                F_HF_f = J_f
            F_f = Hcore + F_HF_f + V_xc_f
            F_f = 0.5 * (F_f + F_f.T)
            C_f, eps_f = diagonalise(F_f)
            E_HF_f = 0.5 * float(np.einsum("ij,ij->", D, F_HF_f))
            E_core_f = float(np.einsum("ij,ij->", D, Hcore))
            E_elec_f = E_xc_f + E_core_f + E_HF_f
            E_coulomb_f = 0.5 * float(np.einsum("ij,ij->", D, J_f))
            E_hf_K_f = (
                -0.25 * float(np.einsum("ij,ij->", D, K_f))
                if K_f is not None
                else 0.0
            )
            if lat_opts.coulomb_method in (
                CoulombMethod.EWALD_3D,
                CoulombMethod.SLAB_EWALD_2D,
            ):
                E_madelung_fix_f = 0.0
            else:
                E_madelung_fix_f = _madelung_energy_correction(
                    D,
                    S,
                    system,
                    nuclear_uses_ewald=False,
                )
            result.energy = E_elec_f + e_nuc + E_madelung_fix_f
            result.e_electronic = E_elec_f
            result.e_xc = E_xc_f
            result.e_coulomb = E_coulomb_f
            result.e_hf_exchange = E_hf_K_f
            result.mo_energies = eps_f
            result.mo_coeffs = C_f
            result.density = D
            result.fock = F_f
            result.converged = True
            plog.converged(
                n_iter=result.n_iter,
                energy=result.energy,
                converged=True,
            )
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
