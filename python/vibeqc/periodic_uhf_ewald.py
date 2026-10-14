"""Phase 15a: Γ-point periodic UHF SCF driver using the composed
Ewald-3D Coulomb dispatch.

Open-shell counterpart of :mod:`vibeqc.periodic_rhf_ewald`. Carries
two density matrices ``D_a``, ``D_b`` (one-particle, no factor 2)
with the standard UHF Fock construction

    F_a = H_core + J(D_a + D_b, w)  -  K(D_a)
    F_b = H_core + J(D_a + D_b, w)  -  K(D_b)

where the Hartree J uses the w-invariant Ewald split from
:func:`vibeqc.build_j_ewald_3d` (Phase 12e-c-4a) and the per-spin
exchange K comes from the full-range real-space builder
(:func:`vibeqc.build_jk_gamma_molecular_limit` at w = 0). DIIS runs
on the *block-diagonal* error vector ``[e_a, e_b]`` with one rolling
F history per spin -- same convention as the molecular UHF C++
driver.

Occupations from the molecule's ``multiplicity``:
    n_a = (n_e + mult - 1) / 2
    n_b = (n_e - mult + 1) / 2

Reports ``<S^2> = (n_a - n_b)(n_a - n_b + 2)/4
                 + n_b - S_{ij in occ} |<phi_i^a | phi_j^b>|^2``
as the standard spin-contamination diagnostic (UHF tests rely on
matching this to the molecular UHF driver in the molecular limit).

Scope (this commit)
-------------------

  - Γ-only (single k-point at the origin). Multi-k UHF lands in
    Phase 15b on top of the multi-k Ewald RHF substrate.
  - Closed-shell special case (multiplicity = 1) reproduces the RHF
    SCF result to ~µHa -- sanity check against
    :func:`run_rhf_periodic_gamma_ewald3d`.
  - DIIS supported (default on); per-spin damping in the no-DIIS
    fallback.
"""

from __future__ import annotations

from .guess import periodic_result_selection

import math
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .periodic.exchange_convention import MOLECULAR_LIMIT
from ._vibeqc_core import (
    apply_level_shift,
    BasisSet,
    bloch_sum,
    build_jk_gamma_molecular_limit,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    CoulombMethod,
    InitialGuess,
    LatticeSumOptions,
    level_shift_at_iter,
    LevelShiftDensity,
    nuclear_repulsion_per_cell,
    PeriodicRHFOptions,
    PeriodicSystem,
    SCFIteration,
    SpinlockMode,
)
from .ewald_composed import make_ewald_3d_gamma_j_builder
from .ewald_j import auto_grid
from .guess import initial_densities_open_shell
from .madelung import (
    madelung_energy_correction_for_lat as _madelung_energy_correction_for_lat,
)
from .mom import reorder_occupied_by_max_overlap as _mom_reorder
from .periodic_rhf_ewald import _canonical_orthogonalizer, _refuse_if_dense_ionic
from .periodic_scf_accelerators import DynamicDamping, PeriodicSCFAccelerator
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .solvers._scf import FockDiagonalizer
from .solvers.operators import OperatorFrame, _snapshot
from .fock_state import EnergyComponents, FockBuild, SourceProvenance
from .periodic_fock_source import _gamma_source_details

__all__ = [
    "PeriodicUHFEwaldResult",
    "run_uhf_periodic_gamma_ewald3d",
]


@dataclass
class PeriodicUHFEwaldResult:
    """Result of :func:`run_uhf_periodic_gamma_ewald3d`.

    Mirrors the molecular ``UHFResult`` shape with a separate a/b
    block, plus Ewald bookkeeping (``omega``, ``grid_shape``).
    All matrices are at Γ (no Bloch sums).
    """

    energy: float
    e_electronic: float
    e_nuclear: float
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

    # Every Gamma-only Ewald/GDF driver builds exchange with the
    # molecular-limit kernel (home-cell density, image blocks zeroed); the
    # runner's manifest records this instead of the exxdiv knob (GitLab #23).
    exchange_q0: str = MOLECULAR_LIMIT
    fock_build: FockBuild | None = None
    next_density_alpha: np.ndarray | None = None
    next_density_beta: np.ndarray | None = None
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


class _GammaUHFBuilder:
    """Synchronous method construction; completed states retain no builder.

    One J[Dalpha+Dbeta] and two existing K[2 Dspin]/2 source calls per build.
    Source drift is refused rather than reusing an invalid integral cache.
    Concurrent mutation of provider objects during a call is unsupported.
    """
    def __init__(self, basis, system, lattice_options, kinetic, nuclear, frame,
                 e_nuclear, j_build, source_details, *, method="UHF"):
        if method not in {"UHF", "ROHF"}:
            raise ValueError("Gamma spin HF builder requires UHF or ROHF")
        self.method = method
        self.basis, self.system, self.lattice_options = basis, system, lattice_options
        self.kinetic, self.nuclear = _snapshot(kinetic), _snapshot(nuclear)
        self.hcore = _snapshot(.5 * ((kinetic + nuclear) + (kinetic + nuclear).T))
        self.frame, self.e_nuclear, self.j_build = frame, e_nuclear, j_build
        self.source_details = source_details
        self.source = SourceProvenance.capture("libint2+periodic-ewald-j", **source_details())
        self.generation = 0

    def __call__(self, density_alpha, density_beta):
        if SourceProvenance.capture(self.source.provider, **self.source_details()) != self.source:
            raise ValueError("Gamma spin HF numerical source changed; rebuild its provider and metric frame")
        da, db = _snapshot(density_alpha), _snapshot(density_beta)
        if np.iscomplexobj(da) or np.iscomplexobj(db):
            raise NotImplementedError("Gamma Ewald UHF requires real spin densities")
        dt = da + db
        j = self.j_build(dt)
        ka, kb = (0.5 * np.asarray(build_jk_gamma_molecular_limit(
            self.basis, self.system, self.lattice_options, 2.0 * d, 0.0).K)
                  for d in (da, db))
        fa, fb = (self.hcore + j - k for k in (ka, kb))
        fa, fb = .5 * (fa + fa.T), .5 * (fb + fb.T)
        # Retain the method's HF half trace, including its summation order.
        electronic = (0.5 * float(np.einsum("ij,ij->", dt, self.hcore))
                      + 0.5 * float(np.einsum("ij,ij->", da, fa))
                      + 0.5 * float(np.einsum("ij,ij->", db, fb)))
        components = (
            ("kinetic", float(np.einsum("ij,ij->", dt, self.kinetic))),
            ("nuclear_attraction", float(np.einsum("ij,ij->", dt, self.nuclear))),
            ("coulomb", 0.5 * float(np.einsum("ij,ij->", dt, j))),
            ("exchange", -0.5 * float(np.einsum("ij,ij->", da, ka))
             - 0.5 * float(np.einsum("ij,ij->", db, kb))),
        )
        correction = _madelung_energy_correction_for_lat(
            dt, self.frame.overlap, self.system, self.lattice_options)
        state = FockBuild.from_dense(
            method=self.method, generation=self.generation, source=self.source,
            energy=EnergyComponents(electronic=electronic, nuclear=self.e_nuclear,
                                    components=components, corrections=(("madelung", correction),)),
            densities=(da, db), focks=(fa, fb), frames=(self.frame, self.frame),
            spins=("alpha", "beta"), kpoints=((0., 0., 0.),) * 2, weights=(1., 1.),
        )
        self.generation += 1
        return state


def _integer_spin_occupations(
    orbital_energies: np.ndarray,
    n_occupied: int,
) -> np.ndarray:
    """Return the fixed spin occupations matching an orbital array."""
    occupations = np.zeros(np.asarray(orbital_energies).size, dtype=float)
    occupations[:n_occupied] = 1.0
    return occupations


def _spin_squared(
    n_alpha: int,
    n_beta: int,
    C_alpha: np.ndarray,
    C_beta: np.ndarray,
    S: np.ndarray,
) -> float:
    """<S^2> for a UHF wavefunction.

    ``S^2_UHF = (n_a - n_b)(n_a - n_b + 2)/4 + n_b
              - S_{i in occ_a, j in occ_b} |<phi_i^a | phi_j^b>|^2``

    Matches the molecular C++ driver's convention. The exact
    eigenvalue for a pure spin state with multiplicity ``mult`` is
    ``S(S+1)`` where ``S = (mult - 1)/2``.
    """
    diff = n_alpha - n_beta
    s2 = 0.25 * diff * (diff + 2) + n_beta
    if n_alpha == 0 or n_beta == 0:
        return float(s2)
    Ca_occ = C_alpha[:, :n_alpha]
    Cb_occ = C_beta[:, :n_beta]
    overlap_ab = Ca_occ.T @ S @ Cb_occ
    s2 -= float(np.sum(np.abs(overlap_ab) ** 2))
    return float(s2)


def run_uhf_periodic_gamma_ewald3d(
    system: PeriodicSystem,
    basis: BasisSet,
    options: Optional[PeriodicRHFOptions] = None,
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
) -> PeriodicUHFEwaldResult:
    """Γ-point open-shell UHF SCF with the EWALD_3D Coulomb dispatch.

    Same interface as :func:`run_rhf_periodic_gamma_ewald3d` but
    with a/b occupations driven by the molecule's ``multiplicity``.
    Reuses the RHF driver's ``_canonical_orthogonalizer`` and the
    shared :class:`PeriodicSCFAccelerator` from
    ``periodic_scf_accelerators``; the Ewald J / real-space K builds
    are identical to the closed-shell path.

    Per-spin Fock solves use independent shared-solver histories and leave
    the caller's Davidson controls unchanged. Ordinary SCF may request only
    occupied roots; PATTERN_HOLD and quadratic updates require the complete
    retained spectrum. The HCORE seed follows the same dense/Davidson choice
    in a separate one-electron solve, without seeding either spin history.

    Parameters
    ----------
    system, basis, options
        See :func:`run_rhf_periodic_gamma_ewald3d`. ``options`` is a
        :class:`PeriodicRHFOptions` (UHF-specific options haven't
        diverged from RHF in vibe-qc's Python drivers -- same DIIS,
        damping, level shift, etc.).
    omega, grid_shape, origin, spacing_bohr, linear_dep_threshold
        Ewald + numeric controls. See the RHF driver.
    """
    opts = options if options is not None else PeriodicRHFOptions()
    from .guess import select_periodic_driver_guess
    selection = select_periodic_driver_guess(
        system, opts, route="ewald", method="UHF", driver="periodic_uhf_ewald",
    )
    max_iter = int(opts.max_iter)
    if max_iter < 1:
        raise ValueError(
            "run_uhf_periodic_gamma_ewald3d: max_iter must be >= 1; "
            f"got {max_iter}"
        )
    # SPINLOCK SPIN_SCHEDULE: two-phase (lock spin for spinlock_iterations
    # cycles, then release to the target multiplicity from that density).
    # Delegated before any SCF setup; the two sub-runs re-enter this driver
    # with spinlock_mode = OFF, so there is no further recursion.
    if (
        getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.SPIN_SCHEDULE
        and int(getattr(opts, "spinlock_iterations", 0)) > 0
    ):
        from .spinlock_periodic import run_spin_schedule

        return run_spin_schedule(
            lambda sysx, o: run_uhf_periodic_gamma_ewald3d(
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
    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)

    # ---- Force EWALD_3D gauge (gauge consistency; historical design record F4 2026-06-01) ----
    # This driver hard-codes the Hartree J to the Ewald-3D builder, so V_ne
    # (compute_nuclear_lattice_dispatch) and e_nuc (nuclear_repulsion_per_cell)
    # MUST share that gauge. Without the force, a default options object
    # (coulomb_method=DIRECT_TRUNCATED) makes nuclear_repulsion_per_cell return
    # the molecular 1/d sum and madelung_energy_correction_for_lat the bare-gauge
    # +a_M.Q_e^2/2L term; those only partially cancel (~0.74 mHa on H2/30-bohr),
    # so the SCF converged to a non-physical energy with no warning (CLAUDE.md
    # Sec.7). The closed-shell Γ / multi-k RHF/RKS siblings already force this
    # (audit F1); extending it here aligns e_nuclear with
    # run_rhf_periodic_gamma_ewald3d and zeroes the now-redundant Madelung term
    # (madelung_energy_correction_for_lat returns 0.0 for EWALD_3D).
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

    # Fail closed if periodic images of the basis overlap enough to make the
    # molecular-limit energy wrong (CLAUDE.md Sec.7); GDF/BIPOLE/GPW are correct
    # there. ``allow_dense_ionic=True`` bypasses for mechanics testing only.
    _refuse_if_dense_ionic(system, basis, lat_opts, allow_dense_ionic)

    # w must match the nuclear Ewald a (auto-selected from
    # nuclear_cutoff_bohr in the C++ ewald engine) so the jellium
    # background terms cancel exactly. Mirrors the override block in
    # run_rhf_periodic_gamma_ewald3d (commit 49f8ae91). The driver
    # kwarg ``omega`` is retained for signature parity but overridden
    # here; users override via ``opts.ewald_omega`` when needed.
    _ewald_tol = getattr(opts, "ewald_tolerance", 1e-12)
    _cutoff = getattr(opts, "ewald_cutoff_bohr", lat_opts.nuclear_cutoff_bohr)
    if omega <= 0.0:
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

    if grid_shape is None:
        grid_shape_t = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape_t = (grid_shape, grid_shape, grid_shape)
    else:
        grid_shape_t = tuple(int(x) for x in grid_shape)
    plog.info(
        f"UHF Gamma EWALD_3D / omega = {float(omega):.3f}, "
        f"FFT grid {grid_shape_t[0]}x{grid_shape_t[1]}x{grid_shape_t[2]}"
    )
    plog.info(f"basis: {basis.name}  ({basis.nbasis} BFs / {basis.nshells} shells)")
    # Active-settings dump.
    from .options_dump import dump_active_settings

    dump_active_settings(
        plog,
        [
            ("PeriodicRHFOptions", opts),
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

    # ---- Open-shell occupations ------------------------------------------
    n_elec = int(system.n_electrons())
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(
            f"run_uhf_periodic_gamma_ewald3d: multiplicity must be >= 1, got {mult}"
        )
    if (n_elec + mult - 1) % 2 != 0 or (n_elec - mult + 1) % 2 != 0:
        raise ValueError(
            f"run_uhf_periodic_gamma_ewald3d: (n_electrons={n_elec}, "
            f"multiplicity={mult}) cannot be split into integer "
            f"a/b occupations."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2
    if n_alpha < 0 or n_beta < 0:
        raise ValueError(
            f"run_uhf_periodic_gamma_ewald3d: invalid occupations "
            f"n_a={n_alpha}, n_b={n_beta} for n_e={n_elec}, "
            f"mult={mult}"
        )

    plog.info(
        f"UHF Gamma occupations: n_alpha = {n_alpha}, "
        f"n_beta = {n_beta} (multiplicity = {mult})"
    )

    # ---- Auto-optimise lattice truncation (default ON) -------------------
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
                    "with the last evaluated settings."
                )
            lat_opts = opt_rep.optimized_lattice_opts

    # ---- One-electron integrals at Γ -------------------------------------
    with plog.stage(
        "integrals_lattice", detail=f"S/T/V at cutoff {lat_opts.cutoff_bohr:.2f} bohr"
    ):
        S_lat = compute_overlap_lattice(basis, system, lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, lat_opts)
        from .periodic_v_ne import compute_nuclear_lattice_dispatch

        V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    k_gamma = np.zeros(3)
    S = np.real(bloch_sum(S_lat, k_gamma))
    T = np.real(bloch_sum(T_lat, k_gamma))
    V = np.real(bloch_sum(V_lat, k_gamma))
    Hcore = T + V
    S = 0.5 * (S + S.T)
    Hcore = 0.5 * (Hcore + Hcore.T)

    # Linear-dependence preflight on S(Γ); see periodic_rhf_ewald.py
    # for the rationale.
    from .linear_dependence import scf_preflight_overlap_check

    scf_preflight_overlap_check(
        S,
        plog=plog,
        label="S(Γ)",
        basis=basis,
    )

    # Canonical-orth on shared S (a and b share the basis orthogonaliser).
    X, n_kept = _canonical_orthogonalizer(
        S,
        linear_dep_threshold,
        normalize_diag_first=canonical_orth_normalize_diag_first,
    )
    if max(n_alpha, n_beta) > n_kept:
        raise RuntimeError(
            f"run_uhf_periodic_gamma_ewald3d: canonical orthogonalisation "
            f"kept {n_kept} directions; need >= {max(n_alpha, n_beta)} "
            f"(n_a={n_alpha}, n_b={n_beta})."
        )

    # State following and orbital rotations need every retained candidate,
    # while ordinary Aufbau SCF needs only the occupied space for each spin.
    needs_full_space = (
        (getattr(opts, "spinlock_mode", SpinlockMode.OFF) == SpinlockMode.PATTERN_HOLD
         and int(getattr(opts, "spinlock_iterations", 0)) > 0)
        or int(getattr(opts, "quadratic_fallback_iter", 0)) > 0
    )
    def source_details():
        return _gamma_source_details(
            basis, system, lat_opts, omega=omega, grid_shape=grid_shape_t,
            origin=origin, spacing_bohr=spacing_bohr)

    source_basis = source_details()
    frame = OperatorFrame(
        n=basis.nbasis, basis_id=hashlib.sha256(json.dumps(
            source_basis["shells"], sort_keys=True).encode()).hexdigest(),
        ordering=source_basis["ordering"], image_gauge="gamma-bloch-at-declared-shell-origins",
        overlap=S, X=X)
    X, S = frame.X, frame.overlap
    diagonalize_alpha = FockDiagonalizer(
        X, S, opts, spin="alpha", n_required=n_kept if needs_full_space else n_alpha,
        real_only=True, frame=frame,
    )
    diagonalize_beta = FockDiagonalizer(
        X, S, opts, spin="beta", n_required=n_kept if needs_full_space else n_beta,
        real_only=True, frame=frame,
    )

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    # Cache the iteration-invariant analytic-FT Hartree-J machinery once
    # (Bloch AO-pair FT, dense G-mesh, 4pi/G^2 kernel) so the PATOM
    # in-field step and the SCF loop reuse the same Ewald-J route.
    j_build = make_ewald_3d_gamma_j_builder(
        basis,
        system,
        omega=float(omega),
        lattice_opts=lat_opts,
        grid_shape=grid_shape_t,
        origin=origin,
        spacing_bohr=spacing_bohr,
    )
    build_fock = _GammaUHFBuilder(
        basis, system, lat_opts, T, V, frame, e_nuc, j_build, source_details)
    T, V, Hcore = build_fock.kinetic, build_fock.nuclear, build_fock.hcore

    # ---- Initial guess via the unified engine.
    # Bug fix (v0.9.x): this Γ-only UHF path previously ignored
    # ``opts.initial_guess`` entirely and always used HCore. The
    # engine call respects the request; HCore reproduces the prior
    # behaviour bit-identically.
    guess = selection.effective
    seed_guess = InitialGuess.SAD if guess == InitialGuess.PATOM else guess
    # Always seed the per-spin MO frame from Hcore -- used by the
    # quadratic-step tracking even when SAD provides the start density.
    # Preserve the configured seed solver, with a one-electron identity and
    # no shared warm state between the seed and density-dependent Focks.
    C_seed, eps_seed = FockDiagonalizer(
        X, S, opts, kind="one_electron", spin="restricted",
        n_required=n_kept if needs_full_space else max(n_alpha, n_beta),
        real_only=True,
    )(Hcore)
    C_alpha, eps_alpha = C_seed.copy(), eps_seed.copy()
    C_beta, eps_beta = C_seed.copy(), eps_seed.copy()
    del C_seed, eps_seed
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
        overlap=S.copy(),
    )
    if split is not None:
        D_alpha, D_beta = split
    else:
        # HCore fallback: per-spin densities from the leading
        # occupied columns of the Hcore-diagonalised MO frame.
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
        seed_build = build_fock(D_alpha, D_beta)
        C_alpha, eps_alpha = diagonalize_alpha(seed_build.channels[0].operator)
        C_beta, eps_beta = diagonalize_beta(seed_build.channels[1].operator)
        del seed_build
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
            f"run_uhf_periodic_gamma_ewald3d: damping must be in [0, 1); got {damping}"
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
    # resolved per iteration by the shared C++ helper, applied per-spin.
    _ls_schedule = list(getattr(opts, "level_shift_schedule", None) or [])
    _ls_max_iter = int(opts.max_iter)

    # SPINLOCK PATTERN_HOLD: hold the seeded broken-symmetry occupied set by
    # maximum overlap (MOM) with the previous cycle for the first
    # ``spinlock_iterations`` cycles, then release -- protecting an ATOMSPIN
    # seed from collapsing to the symmetric solution. Mirrors the molecular
    # UHF path (cpp/src/uhf.cpp) and the periodic multi-k UKS driver.
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

    # ---- SCF loop --------------------------------------------------------
    scf_trace: List[SCFIteration] = []
    result = PeriodicUHFEwaldResult(
                 restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0,
        e_electronic=0.0,
        e_nuclear=e_nuc,
        n_iter=0,
        converged=False,
        s_squared=0.0,
        s_squared_ideal=0.25 * (mult - 1) * (mult + 1),
        mo_energies_alpha=np.empty(0),
        mo_coeffs_alpha=np.empty((0, 0)),
        density_alpha=D_alpha.copy(),
        fock_alpha=np.empty((0, 0)),
        mo_energies_beta=np.empty(0),
        mo_coeffs_beta=np.empty((0, 0)),
        density_beta=D_beta.copy(),
        fock_beta=np.empty((0, 0)),
        overlap=S,
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
    )

    E_prev = 0.0
    plog.banner("SCF (UHF Gamma, EWALD_3D)")
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

        # Density damping per spin; skip when DIIS active.
        if iter_idx == 1 or damping == 0.0 or diis_active:
            D_alpha_used = D_alpha
            D_beta_used = D_beta
        else:
            D_alpha_used = damping * D_alpha_prev + (1.0 - damping) * D_alpha
            D_beta_used = damping * D_beta_prev + (1.0 - damping) * D_beta

        physical = build_fock(D_alpha_used, D_beta_used)
        alpha, beta = physical.channels
        F_alpha, F_beta = alpha.operator.matrix, beta.operator.matrix
        D_alpha_used, D_beta_used = alpha.density, beta.density
        E_elec, E_total = physical.energy.electronic, physical.energy.total

        # Orbital-gradient norms per spin.
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
            "run_uhf_periodic_gamma_ewald3d",
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
        result.evaluated_build_history.append(dict(
            iteration=iter_idx, generation=physical.generation,
            build_id=physical.identity, density_id=physical.density_id))
        plog.iteration(
            iter_idx,
            energy=float(E_total),
            dE=float(dE if iter_idx > 1 else 0.0),
            grad=float(grad_norm),
            diis=diis_sub,
        )
        # Per-iter energy decomposition for cross-code parity comparison.
        components = dict(physical.energy.components)
        E_madelung_fix = dict(physical.energy.corrections)["madelung"]
        plog.energy_decomposition(
            iter_idx,
            E_kin=components["kinetic"],
            E_ne=components["nuclear_attraction"],
            E_J=components["coulomb"],
            E_K=components["exchange"],
            E_nuc=float(e_nuc),
            E_madelung=float(E_madelung_fix),
        )
        converged = (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )

        # Phase C1c gate -- per-spin Newton step replaces both DIIS and
        # diagonalisation when the fallback is active.
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

            # Saunders-Hillier level shift per spin. UHF densities are
            # one-particle (no factor 2), so the projector onto the
            # occupied subspace is S.D_s.S (not 1/2.S.D.S as for closed
            # shell) and the shift formula is
            #     F_s_shift = F_s + b.S - b.(S.D_s.S)
            # The pre-update density D_s_used pins the shift to the
            # current iterate (matches the multi-k UHF and RHF/RKS
            # conventions).
            b = (
                level_shift_at_iter(
                    level_shift, 0, _ls_schedule, _ls_max_iter, iter_idx
                )
                if _ls_schedule
                else level_shift
            )
            # Shared Saunders-Hillier operator. D_alpha_used / D_beta_used are
            # *spin* densities (occupations in {0, 1}) and idempotent in the S
            # metric, so S·Dσ·S is the occupied projector and the weight is 1,
            # not the closed-shell ½. Returns F untouched when b == 0.
            if b == 0.0:
                # Skip the pybind conversion of S and Dσ entirely on an
                # unshifted cycle, which is the default and common case.
                F_alpha_diag = F_alpha
                F_beta_diag = F_beta
            else:
                F_alpha_diag = apply_level_shift(
                    F_alpha, S, D_alpha_used, b, LevelShiftDensity.SPIN)
                F_beta_diag = apply_level_shift(
                    F_beta, S, D_beta_used, b, LevelShiftDensity.SPIN)

            # Diagonalize + update.
            modification = "+".join(
                name for active, name in ((diis_active and accel is not None and not hold_active, "extrapolation"),
                                          (b != 0., "level_shift")) if active)
            op_alpha = (physical.solver_operator(0, F_alpha_diag, modification=modification)
                        if modification else alpha.operator)
            op_beta = (physical.solver_operator(1, F_beta_diag, modification=modification)
                       if modification else beta.operator)
            C_alpha_new, eps_alpha_new = diagonalize_alpha(op_alpha)
            C_beta_new, eps_beta_new = diagonalize_beta(op_beta)

        # SPINLOCK PATTERN_HOLD: after both the normal and quadratic branches
        # produce C_*_new, reorder the occupied set by max overlap with the
        # previous cycle (MOM) for cycles 2..spinlock_iterations, then release.
        # iter 1 sets the pattern by aufbau (C_*_prev_mo is the prior cycle).
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
        if n_alpha > 0:
            D_alpha = C_alpha_new[:, :n_alpha] @ C_alpha_new[:, :n_alpha].T
        else:
            D_alpha = np.zeros_like(Hcore)
        if n_beta > 0:
            D_beta = C_beta_new[:, :n_beta] @ C_beta_new[:, :n_beta].T
        else:
            D_beta = np.zeros_like(Hcore)
        D_alpha = 0.5 * (D_alpha + D_alpha.T)
        D_beta = 0.5 * (D_beta + D_beta.T)

        # The terminal payload is assembled below from one orbital-defined
        # density and a physical Fock/energy evaluated on that same density.
        # Do not expose the pre-update (and possibly damped) state here: its
        # density belongs to different orbitals than C_*_new.
        result.n_iter = iter_idx

        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

        if converged or iter_idx == max_iter:
            # Evaluate one internally consistent terminal state on every
            # exit path.  The last orbital/occupation state defines D; the
            # unshifted physical Fock and energy are then rebuilt on exactly
            # that D.  At a non-fixed-point iteration cap those orbitals do
            # not diagonalise F[D] (by definition), so the reported orbital
            # energies are their Rayleigh quotients in the rebuilt Fock.
            terminal = build_fock(D_alpha, D_beta)
            F_alpha_f, F_beta_f = (c.operator.matrix for c in terminal.channels)
            C_a_f, eps_a_f = diagonalize_alpha(terminal.channels[0].operator)
            C_b_f, eps_b_f = diagonalize_beta(terminal.channels[1].operator)
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
            result.fock_build = terminal
            result.next_density_alpha = _snapshot(D_alpha_next)
            result.next_density_beta = _snapshot(D_beta_next)
            result.density_fixed_point_residual = density_fixed_point_residual
            result.energy = terminal.energy.total
            result.e_electronic = terminal.energy.electronic
            result.mo_energies_alpha = np.real(
                np.diag(C_alpha_new.T @ F_alpha_f @ C_alpha_new)
            )
            result.occupations_alpha = _integer_spin_occupations(
                result.mo_energies_alpha, n_alpha
            )
            result.mo_coeffs_alpha = C_alpha_new
            result.density_alpha = terminal.channels[0].density.copy()
            result.fock_alpha = F_alpha_f.copy()
            result.mo_energies_beta = np.real(
                np.diag(C_beta_new.T @ F_beta_f @ C_beta_new)
            )
            result.occupations_beta = _integer_spin_occupations(
                result.mo_energies_beta, n_beta
            )
            result.mo_coeffs_beta = C_beta_new
            result.density_beta = terminal.channels[1].density.copy()
            result.fock_beta = F_beta_f.copy()
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
            grad_norm_f = terminal.commutator_norm
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
                result.evaluated_build_history[-1] = dict(
                    iteration=iter_idx, generation=terminal.generation,
                    build_id=terminal.identity, density_id=terminal.density_id)
            if terminal_converged:
                result.converged = True
                plog.converged(
                    n_iter=result.n_iter,
                    energy=result.energy,
                    converged=True,
                )
                return result

    # Did not converge -- populate <S^2> on the final iterate anyway.
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
