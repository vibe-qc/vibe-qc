"""Restricted open-shell Hartree--Fock for periodic systems --- Γ-point,
EWALD_3D Coulomb dispatch.

The periodic counterpart of :func:`vibeqc.run_rohf`: a spin-pure
restricted-open-shell determinant at the Γ point.  It reuses, verbatim,
the *validated* periodic two-electron machinery from the Γ UHF driver
(:mod:`vibeqc.periodic_uhf_ewald`) --- the Ewald-3D Hartree J
(:func:`make_ewald_3d_gamma_j_builder`), the real-space exchange K
(:func:`build_jk_gamma_molecular_limit`), the Madelung self-image
correction (:func:`madelung.madelung_energy_correction_for_lat`) and the
per-cell nuclear repulsion (:func:`nuclear_repulsion_per_cell`).  The
*only* change from UHF is the orbital update: instead of two independent
per-spin diagonalisations, the per-spin Fock matrices are combined into
Roothaan's single effective Fock
(:func:`vibeqc.rohf.roothaan_effective_fock`) and diagonalised once, with
the standard open-shell occupation rule
(:func:`vibeqc.rohf._roothaan_occupations`).

This split keeps the Sec.7-sensitive periodic physics (gauge / Madelung /
exxdiv) identical to the build-validated UHF path, while the
restricted-open-shell coupling + DIIS + occupation are the same code the
molecular ROHF driver uses (and unit-tested in ``tests/test_rohf.py``).

Energy is the per-cell UHF expression (ROHF and UHF share it) plus the
Madelung-leak fix::

    E = 1/2 Tr[Dt H] + 1/2 Tr[Da Fa] + 1/2 Tr[Db Fb] + E_madelung + E_nuc

Scope: Γ-point, EWALD_3D, Hartree--Fock. Multi-k ROHF/ROKS use their
dedicated Ewald drivers; this module keeps the Gamma-only compatibility entry
point.
"""

from __future__ import annotations


from .guess import periodic_result_selection

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    CoulombMethod,
    InitialGuess,
    LatticeSumOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    SCFIteration,
    bloch_sum,
    build_jk_gamma_molecular_limit,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    nuclear_repulsion_per_cell,
)
from .ewald_composed import make_ewald_3d_gamma_j_builder
from .ewald_j import auto_grid
from .guess import initial_densities_open_shell
from .madelung import (
    madelung_energy_correction_for_lat as _madelung_energy_correction_for_lat,
)
from .periodic_rhf_ewald import _canonical_orthogonalizer, _refuse_if_dense_ionic
from .progress import ProgressLogger, resolve_progress
from .scf_divergence import check_scf_divergence
from .rohf import (
    _commutator_error,
    _diis_extrapolate,
    _densities_from_occupations,
    _roothaan_occupations,
    roothaan_effective_fock,
)

__all__ = ["PeriodicROHFEwaldResult", "run_rohf_periodic_gamma_ewald3d"]


@dataclass
class PeriodicROHFEwaldResult:
    """Result of :func:`run_rohf_periodic_gamma_ewald3d`.

    Single spin-restricted orbital set (``mo_energies`` / ``mo_coeffs``),
    closed/open/virtual ``mo_occupations`` (2/1/0), and the per-spin
    densities.  Exposes the UHF-style ``*_alpha`` / ``*_beta`` aliases
    (identical spatial orbitals) so the periodic output writers handle it
    with no special-casing, exactly as the molecular ``ROHFResult`` does.
    ``s_squared`` is exact for a restricted-open determinant.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    mo_occupations: np.ndarray
    density: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    fock: np.ndarray
    fock_alpha: np.ndarray
    fock_beta: np.ndarray
    overlap: np.ndarray
    n_alpha: int
    n_beta: int
    scf_trace: List[SCFIteration] = field(default_factory=list)
    omega: float = 0.0
    grid_shape: Tuple[int, int, int] = (0, 0, 0)
    method: str = "rohf"

    # -- Unrestricted-view aliases (identical a/b spatial orbitals) --------
    @property
    def mo_energies_alpha(self) -> np.ndarray:
        return self.mo_energies

    @property
    def mo_energies_beta(self) -> np.ndarray:
        return self.mo_energies

    @property
    def mo_coeffs_alpha(self) -> np.ndarray:
        return self.mo_coeffs

    @property
    def mo_coeffs_beta(self) -> np.ndarray:
        return self.mo_coeffs

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None


def run_rohf_periodic_gamma_ewald3d(
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
) -> PeriodicROHFEwaldResult:
    """Γ-point restricted-open-shell HF SCF with the EWALD_3D dispatch.

    Same interface as :func:`vibeqc.run_uhf_periodic_gamma_ewald3d`; the
    a/b counts come from ``system.multiplicity``.  Returns a spin-pure
    determinant (``s_squared = S(S+1)`` exactly).
    """
    opts = options if options is not None else PeriodicRHFOptions()
    from .guess import select_periodic_driver_guess
    guess = select_periodic_driver_guess(
        system, opts, route="ewald", method="ROHF",
        driver="run_rohf_periodic_gamma_ewald3d",
    ).effective
    lat_opts: LatticeSumOptions = opts.lattice_opts
    plog = resolve_progress(progress, verbose=verbose)

    # ---- Force EWALD_3D gauge (gauge consistency; see the UHF driver) ----
    # This driver's Hartree J is Ewald-3D, so V_ne / e_nuc must share that
    # gauge or the SCF converges to a non-physical energy (CLAUDE.md Sec.7).
    if system.dim == 3 and lat_opts.coulomb_method != CoulombMethod.EWALD_3D:
        plog.info(
            "coulomb_method forced to EWALD_3D for gauge consistency "
            f"(was {lat_opts.coulomb_method!r})"
        )
        lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    _refuse_if_dense_ionic(system, basis, lat_opts, allow_dense_ionic)

    # w must match the nuclear Ewald a so the jellium backgrounds cancel
    # (mirrors the RHF/UHF drivers).
    if omega <= 0.0:
        _user_omega = getattr(opts, "ewald_omega", None)
        if _user_omega is not None and float(_user_omega) > 0.0:
            omega = float(_user_omega)
        else:
            from .pbc_bipole_common import default_ewald_alpha

            v_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
            # GitLab #651: CRYSTAL's 2.8/V^(1/3), bounded below so that
            # erfc(omega r)/r has decayed to opts.ewald_tolerance at this
            # route's fixed real-space image cutoff (lat_opts.cutoff_bohr).
            omega = default_ewald_alpha(
                v_cell,
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
        f"ROHF Gamma EWALD_3D / omega = {float(omega):.3f}, "
        f"FFT grid {grid_shape_t[0]}x{grid_shape_t[1]}x{grid_shape_t[2]}"
    )

    # ---- Open-shell occupations ------------------------------------------
    n_elec = int(system.n_electrons())
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(
            f"run_rohf_periodic_gamma_ewald3d: multiplicity must be >= 1, got {mult}"
        )
    if (n_elec + mult - 1) % 2 != 0:
        raise ValueError(
            f"run_rohf_periodic_gamma_ewald3d: (n_electrons={n_elec}, "
            f"multiplicity={mult}) cannot be split into integer a/b counts."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2
    plog.info(
        f"ROHF Gamma occupations: closed={n_beta}, open={n_alpha - n_beta} "
        f"(multiplicity = {mult})"
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
    S = 0.5 * (S + S.T)
    Hcore = 0.5 * ((T + V) + (T + V).T)

    from .linear_dependence import scf_preflight_overlap_check

    scf_preflight_overlap_check(S, plog=plog, label="S(Γ)", basis=basis)

    X, n_kept = _canonical_orthogonalizer(
        S, linear_dep_threshold,
        normalize_diag_first=canonical_orth_normalize_diag_first,
    )
    if n_alpha > n_kept:
        raise RuntimeError(
            f"run_rohf_periodic_gamma_ewald3d: canonical orthogonalisation "
            f"kept {n_kept} directions; need >= {n_alpha}."
        )

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))

    def diagonalize(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fp = X.T @ f @ X
        fp = 0.5 * (fp + fp.T)
        eps, cp = np.linalg.eigh(fp)
        return X @ cp, eps

    # ---- Initial guess ---------------------------------------------------
    c_hcore, _eps_h = diagonalize(Hcore)
    split = initial_densities_open_shell(
        system.unit_cell_molecule(), basis, n_alpha, n_beta,
        InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
        is_periodic=True,
        periodic_system=system,
        lattice_opts=lat_opts,
        atomic_spins=getattr(opts, "atomic_spins", None) or None,
        read_density_alpha=getattr(opts, "read_density_alpha", None),
        read_density_beta=getattr(opts, "read_density_beta", None),
        read_path=getattr(opts, "read_path", ""),
        overlap=S,
    )
    if split is not None:
        D_alpha, D_beta = split
    else:
        ca = c_hcore[:, :n_alpha]
        cb = c_hcore[:, :n_beta]
        D_alpha = ca @ ca.T if n_alpha > 0 else np.zeros_like(Hcore)
        D_beta = cb @ cb.T if n_beta > 0 else np.zeros_like(Hcore)
    D_alpha = 0.5 * (D_alpha + D_alpha.T)
    D_beta = 0.5 * (D_beta + D_beta.T)
    D_alpha_prev, D_beta_prev = D_alpha.copy(), D_beta.copy()

    damping = float(opts.damping)
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    level_shift = float(getattr(opts, "level_shift", 0.0))

    # ---- Cache the iteration-invariant Ewald-3D J machinery once ---------
    j_build = make_ewald_3d_gamma_j_builder(
        basis, system, omega=float(omega), lattice_opts=lat_opts,
        grid_shape=grid_shape_t, origin=origin, spacing_bohr=spacing_bohr,
    )

    def build_fock(da: np.ndarray, db: np.ndarray):
        """Periodic per-spin Fock + per-cell electronic energy (incl. the
        Madelung-leak fix), reusing the validated UHF two-electron path."""
        dt = da + db
        j = j_build(dt)
        # build_jk_gamma_molecular_limit returns K(D); feed 2.D_s and halve
        # to recover the per-spin exchange (the UHF-driver convention).
        ka = 0.5 * np.asarray(
            build_jk_gamma_molecular_limit(basis, system, lat_opts, 2.0 * da, 0.0).K
        )
        kb = 0.5 * np.asarray(
            build_jk_gamma_molecular_limit(basis, system, lat_opts, 2.0 * db, 0.0).K
        )
        fa = Hcore + j - ka
        fb = Hcore + j - kb
        fa = 0.5 * (fa + fa.T)
        fb = 0.5 * (fb + fb.T)
        e_elec = (
            0.5 * float(np.einsum("ij,ij->", dt, Hcore))
            + 0.5 * float(np.einsum("ij,ij->", da, fa))
            + 0.5 * float(np.einsum("ij,ij->", db, fb))
            + _madelung_energy_correction_for_lat(dt, S, system, lat_opts)
        )
        return fa, fb, e_elec

    if guess == InitialGuess.PATOM:
        fa, fb, _ = build_fock(D_alpha, D_beta)
        ca, _ = diagonalize(fa)
        cb, _ = diagonalize(fb)
        D_alpha = ca[:, :n_alpha] @ ca[:, :n_alpha].T
        D_beta = cb[:, :n_beta] @ cb[:, :n_beta].T
        D_alpha_prev, D_beta_prev = D_alpha.copy(), D_beta.copy()

    # ---- SCF loop (Roothaan single effective Fock) -----------------------
    scf_trace: List[SCFIteration] = []
    fock_hist: List[np.ndarray] = []
    err_hist: List[np.ndarray] = []
    e_prev = 0.0
    converged = False
    mo_coeffs = mo_energies = mo_occ = None
    focka = fockb = fock_eff = None
    n_iter = 0

    plog.banner("SCF (ROHF Gamma, EWALD_3D)")
    plog.info("  iter         energy (Ha)            dE          ||[F,DS]||   DIIS")

    for iter_idx in range(1, int(opts.max_iter) + 1):
        n_iter = iter_idx
        diis_active = use_diis and iter_idx >= diis_start_iter
        if iter_idx == 1 or damping == 0.0 or diis_active:
            da_used, db_used = D_alpha, D_beta
        else:
            da_used = damping * D_alpha_prev + (1.0 - damping) * D_alpha
            db_used = damping * D_beta_prev + (1.0 - damping) * D_beta

        focka, fockb, e_elec = build_fock(da_used, db_used)
        e_total = e_elec + e_nuc

        dt = da_used + db_used
        fock_eff = roothaan_effective_fock(focka, fockb, da_used, db_used, S)
        err = _commutator_error(fock_eff, dt, S, X)
        grad_norm = float(np.linalg.norm(err))

        dE = e_total - e_prev
        check_scf_divergence(
            "run_rohf_periodic_gamma_ewald3d", iter_idx, e_total, grad_norm, dE
        )

        diis_dim = 0
        if diis_active:
            fock_hist.append(fock_eff.copy())
            err_hist.append(err.copy())
            if len(fock_hist) > int(opts.diis_subspace_size):
                fock_hist.pop(0)
                err_hist.pop(0)
            diis_dim = len(fock_hist)
            if diis_dim > 1:
                fock_eff = _diis_extrapolate(fock_hist, err_hist)

        scf_trace.append(
            SCFIteration(
                iter=iter_idx, energy=float(e_total),
                delta_e=float(dE if iter_idx > 1 else 0.0),
                grad_norm=float(grad_norm), diis_subspace=diis_dim,
            )
        )
        plog.iteration(
            iter_idx, energy=float(e_total),
            dE=float(dE if iter_idx > 1 else 0.0), grad=float(grad_norm),
            diis=diis_dim,
        )

        if (
            iter_idx > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        ):
            converged = True
            mo_coeffs, mo_energies, mo_occ = _solve_and_occupy(
                fock_eff, focka, X, S, D_alpha, level_shift, n_alpha, n_beta,
            )
            break

        mo_coeffs, mo_energies, mo_occ = _solve_and_occupy(
            fock_eff, focka, X, S, D_alpha, level_shift, n_alpha, n_beta,
        )
        D_alpha_prev, D_beta_prev = D_alpha, D_beta
        D_alpha, D_beta = _densities_from_occupations(mo_coeffs, mo_occ)
        e_prev = e_total

    # ---- Final consistent energy + result --------------------------------
    focka, fockb, e_elec = build_fock(D_alpha, D_beta)
    e_total = e_elec + e_nuc
    fock_eff = roothaan_effective_fock(focka, fockb, D_alpha, D_beta, S)
    if mo_coeffs is None:
        mo_coeffs, mo_energies, mo_occ = _solve_and_occupy(
            fock_eff, focka, X, S, D_alpha, 0.0, n_alpha, n_beta,
        )

    spin = 0.5 * (n_alpha - n_beta)
    s_squared = spin * (spin + 1.0)

    return PeriodicROHFEwaldResult(
               restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
               guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=False),
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=float(e_total),
        e_electronic=float(e_elec),
        e_nuclear=float(e_nuc),
        n_iter=n_iter,
        converged=converged,
        s_squared=s_squared,
        s_squared_ideal=s_squared,
        mo_energies=mo_energies,
        mo_coeffs=mo_coeffs,
        mo_occupations=mo_occ,
        density=D_alpha + D_beta,
        density_alpha=D_alpha,
        density_beta=D_beta,
        fock=fock_eff,
        fock_alpha=focka,
        fock_beta=fockb,
        overlap=S,
        n_alpha=n_alpha,
        n_beta=n_beta,
        scf_trace=scf_trace,
        omega=float(omega),
        grid_shape=grid_shape_t,
    )


def _solve_and_occupy(fock_eff, focka, x, s, dma, level_shift, n_alpha, n_beta):
    """Diagonalise the effective Fock (optional virtual level shift) and
    assign ROHF occupations (closed by effective-Fock energy, open by a
    orbital energy)."""
    f = fock_eff
    if level_shift > 0.0:
        f = f + level_shift * (s - s @ dma @ s)
    fp = x.T @ f @ x
    fp = 0.5 * (fp + fp.T)
    eps, cp = np.linalg.eigh(fp)
    c = x @ cp
    # Alpha orbital energies for the open-shell selection: diag of Fa in MO.
    eps_alpha = np.einsum("pi,pq,qi->i", c, focka, c)
    occ = _roothaan_occupations(eps, eps_alpha, n_alpha, n_beta)
    return c, eps, occ
