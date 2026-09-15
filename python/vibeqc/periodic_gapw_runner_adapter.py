"""Adapter that converts a :class:`GpwScfResult` (the standalone GPW
SCF entry) to the duck-typed result shape that
:func:`vibeqc.periodic_runner.run_periodic_job` expects.

The runner consumes a result with at least these attributes:

* ``energy``, ``e_electronic``, ``e_nuclear``, ``e_coulomb``,
  ``e_xc``, ``e_hf_exchange``, ``free_energy`` (optional)
* ``n_iter``, ``converged``
* ``mo_energies``, ``mo_coeffs``, ``occupations``, ``density``
  (each may be a single array or a list of per-k arrays;
  Γ-only GPW returns single arrays)
* ``scf_trace`` -- a list of objects with ``iter, energy, delta_e,
  grad_norm, diis_subspace, newton_cg_iter, trah_level_shift``
  (the C++ ``SCFIteration`` shape)
* ``fock`` (optional), ``overlap`` (optional)

This module produces such a result from a
:class:`vibeqc.periodic_gapw_j.GpwScfResult`. The shape is duck-typed
(no inheritance), since GPW is iteratively a Python loop and the
existing runner code does ``getattr(result, 'foo', default)`` for
the optional fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import _vibeqc_core as _core
from .periodic_screened_exchange import reject_unscreened_range_separated


__all__ = [
    "GpwRunnerResult",
    "GpwRohfRunnerResult",
    "GpwOpenShellRunnerResult",
    "GpwUksMultiKRunnerResult",
    "gpw_result_to_runner_shape",
    "gpw_rohf_result_to_runner_shape",
    "gpw_uhf_result_to_runner_shape",
    "gpw_uks_result_to_runner_shape",
    "gpw_uks_multik_result_to_runner_shape",
    "gpw_roks_multik_result_to_runner_shape",
]


@dataclass
class GpwRunnerResult:
    """Duck-typed wrapper around a :class:`GpwScfResult` exposing the
    fields the periodic runner needs.

    Attributes are mutable so the runner's output-rendering code can
    write to optional fields if it needs to. Energy decomposition
    follows the GpwEnergyBreakdown convention:

    * ``e_coulomb`` = the GPW Hartree-J contribution ``1/2 tr(D.J)``.
    * ``e_xc`` = the XC energy read from the breakdown -- 0.0 for HF / RHF,
      the libxc XC energy for RKS / UKS.
    * ``e_hf_exchange`` = the (negative-signed) HF exchange
      contribution ``-(1/4) tr(D.K)``.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    e_coulomb: float
    e_xc: float
    e_hf_exchange: float
    n_iter: int
    converged: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    occupations: np.ndarray
    e_dft_plus_u: float = 0.0
    scf_trace: List[object] = field(default_factory=list)
    one_centre: Optional[str] = None
    molecular_limit_declared: bool = False

    restart_kpoints: object = None
    restart_weights: object = None
    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None


@dataclass
class GpwRohfRunnerResult:
    """Runner-facing shape for a Gamma GPW ROHF or ROKS result.

    ROHF has one spatial-orbital set with 2/1/0 occupations, while retaining
    per-spin densities and Focks for spin-resolved artefacts.  Keeping this
    adapter distinct from the unrestricted adapter prevents output consumers
    from inventing two unrelated orbital sets for a restricted wavefunction.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    e_coulomb: float
    e_xc: float
    e_hf_exchange: float
    n_iter: int
    converged: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    occupations: np.ndarray
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    mo_coeffs_alpha: np.ndarray
    mo_coeffs_beta: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    fock_alpha: np.ndarray
    fock_beta: np.ndarray
    occupations_alpha: np.ndarray
    occupations_beta: np.ndarray
    n_alpha: int
    n_beta: int
    s_squared: float
    s_squared_ideal: float
    e_dft_plus_u: float = 0.0
    method: str = "rohf"
    backend: str = "gpw-rohf"
    scf_trace: List[object] = field(default_factory=list)

    restart_kpoints: object = None
    restart_weights: object = None
    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None


def gpw_rohf_result_to_runner_shape(gpw_result) -> GpwRohfRunnerResult:
    """Expose a restricted-open-shell GPW result to ``run_periodic_job``.

    Unlike the RHF/UHF adapters, no Fock reconstruction is required: the
    standalone ROHF driver already returns its converged effective and
    per-spin Focks, overlap, restricted orbitals, and spin densities.
    """
    n_basis = int(np.asarray(gpw_result.mo_energies).size)
    occ_alpha = np.zeros(n_basis, dtype=float)
    occ_beta = np.zeros(n_basis, dtype=float)
    occ_alpha[: int(gpw_result.n_alpha)] = 1.0
    occ_beta[: int(gpw_result.n_beta)] = 1.0
    breakdown = gpw_result.breakdown
    coeffs = np.asarray(gpw_result.mo_coeffs)
    energies = np.asarray(gpw_result.mo_energies)
    return GpwRohfRunnerResult(
        restart_basis=getattr(gpw_result, "restart_basis", None),
        restart_lattice=getattr(gpw_result, "restart_lattice", None),
        guess_selection=getattr(gpw_result, "guess_selection", None),
        restart_kpoints=getattr(gpw_result, "restart_kpoints", None),
        restart_weights=getattr(gpw_result, "restart_weights", None),
        energy=float(gpw_result.energy),
        e_electronic=float(gpw_result.e_electronic),
        e_nuclear=float(gpw_result.e_nuclear),
        e_coulomb=float(breakdown.e_hartree),
        e_xc=float(getattr(breakdown, "e_xc", 0.0)),
        e_hf_exchange=float(breakdown.e_hf_exchange),
        n_iter=int(gpw_result.n_iter),
        converged=bool(gpw_result.converged),
        mo_energies=energies,
        mo_coeffs=coeffs,
        density=np.asarray(gpw_result.density),
        fock=np.asarray(gpw_result.fock),
        overlap=np.asarray(gpw_result.overlap),
        occupations=np.asarray(gpw_result.mo_occupations),
        mo_energies_alpha=energies,
        mo_energies_beta=energies,
        mo_coeffs_alpha=coeffs,
        mo_coeffs_beta=coeffs,
        density_alpha=np.asarray(gpw_result.density_alpha),
        density_beta=np.asarray(gpw_result.density_beta),
        fock_alpha=np.asarray(gpw_result.fock_alpha),
        fock_beta=np.asarray(gpw_result.fock_beta),
        occupations_alpha=occ_alpha,
        occupations_beta=occ_beta,
        n_alpha=int(gpw_result.n_alpha),
        n_beta=int(gpw_result.n_beta),
        s_squared=float(gpw_result.s_squared),
        s_squared_ideal=float(gpw_result.s_squared_ideal),
        e_dft_plus_u=float(getattr(gpw_result, "e_dft_plus_u", 0.0)),
        method=str(getattr(gpw_result, "method", "rohf")),
        backend=f"gpw-{getattr(gpw_result, 'method', 'rohf')}",
        scf_trace=list(getattr(gpw_result, "scf_trace", ()) or ()),
    )


def gpw_result_to_runner_shape(
    gpw_result,
    system,
    basis,
) -> GpwRunnerResult:
    """Convert :class:`GpwScfResult` to the runner's expected shape.

    Surfaces the driver's converged Fock and overlap directly when the
    result carries them (GPW, GAPW, and the C++-hosted driver all do);
    otherwise synthesises S and an HF-shaped Fock (Hcore + J - 1/2K at
    the converged density) as a fallback. Builds an occupations vector
    consistent with the closed-shell SCF and lifts the driver's
    ``scf_trace`` into the runner's ``SCFIteration`` shape (or a
    minimal flat trace for older drivers without one).
    """
    from .periodic_gapw_j import (
        _kinetic_lattice_gamma, _overlap_lattice_gamma,
        GpwJBuilder,
    )

    bd = gpw_result.breakdown
    D = np.asarray(gpw_result.density)
    n_basis = D.shape[0]

    # Closed-shell occupation: 2 for occupied, 0 for virtual. n_occ
    # derived from the converged density's electron count tr(D.S).
    S_stored = getattr(gpw_result, "overlap", None)
    if S_stored is not None:
        S = np.asarray(S_stored)
    else:
        S = _overlap_lattice_gamma(basis, system)
    n_elec = int(round(float(np.einsum("ij,ij->", D, S))))
    n_occ = n_elec // 2
    occupations = np.zeros(n_basis, dtype=float)
    occupations[:n_occ] = 2.0

    # Energy decomposition.
    e_xc = float(getattr(bd, "e_xc", 0.0))
    e_electronic = (
        bd.e_kinetic
        + bd.e_nuclear_attraction
        + bd.e_hartree
        + bd.e_hf_exchange
        + e_xc
    )
    e_nuclear = bd.e_nuclear_repulsion
    energy = float(getattr(gpw_result, "energy", e_electronic + e_nuclear))
    e_coulomb = bd.e_hartree
    e_hf_exchange = bd.e_hf_exchange
    e_dft_plus_u = float(getattr(gpw_result, "e_dft_plus_u", 0.0))

    # The Fock surfaced to post-SCF consumers. Preferred source: the
    # driver's own converged (post-DIIS) Fock -- the operator whose
    # eigenpairs are the returned mo_coeffs/mo_energies. GPW, GAPW,
    # and the C++-hosted driver all carry it. The reconstruction
    # below is a fallback for producers that predate the field; note
    # it is HF-shaped (Hcore + J - K/2 with the *un-augmented* GPW J,
    # no V_xc), so for KS/GAPW results it is only an approximation --
    # and it costs a fresh chi build plus a molecular-limit JK build
    # (~36% of a small GPW job's wall time).
    F_stored = getattr(gpw_result, "fock", None)
    if F_stored is not None:
        F = np.asarray(F_stored)
    else:
        T = _kinetic_lattice_gamma(basis, system)
        from .periodic_gapw_j import _ewald_v_ne_gamma
        V_ne = _ewald_v_ne_gamma(basis, system)
        Hcore = T + V_ne
        gpw = GpwJBuilder(basis, gpw_result.grid)
        J = gpw.build_J(D)
        lo = _core.LatticeSumOptions()
        lo.cutoff_bohr = 25.0
        jk_mol = _core.build_jk_gamma_molecular_limit(basis, system, lo, D)
        K = np.asarray(jk_mol.K)
        F = Hcore + J - 0.5 * K

    # Build the scf_trace from the standalone driver's per-iter
    # records (M3d). Each entry is a dict; we lift it into the
    # SCFIteration shape the runner's output code expects.
    raw_trace = getattr(gpw_result, "scf_trace", ()) or ()
    trace: List[_core.SCFIteration] = []
    if raw_trace:
        for step in raw_trace:
            trace.append(_core.SCFIteration(
                iter=int(step.get("iter", 0)),
                energy=float(step.get("energy", energy)),
                delta_e=float(step.get("delta_e", 0.0)),
                grad_norm=float(step.get("grad_norm", 0.0)),
                diis_subspace=0,
                newton_cg_iter=0,
                trah_level_shift=0.0,
            ))
    else:
        # Older driver path without scf_trace -- fall back to a flat
        # trace so the runner still has something to render.
        trace = [
            _core.SCFIteration(
                iter=it, energy=float(energy), delta_e=0.0,
                grad_norm=0.0, diis_subspace=0,
                newton_cg_iter=0, trah_level_shift=0.0,
            )
            for it in range(1, gpw_result.n_iter + 1)
        ]

    return GpwRunnerResult(
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        guess_selection=getattr(gpw_result, "guess_selection", None),
        restart_kpoints=getattr(gpw_result, "restart_kpoints", None),
        restart_weights=getattr(gpw_result, "restart_weights", None),
        energy=energy,
        e_electronic=e_electronic,
        e_nuclear=e_nuclear,
        e_coulomb=e_coulomb,
        e_xc=e_xc,
        e_hf_exchange=e_hf_exchange,
        e_dft_plus_u=e_dft_plus_u,
        n_iter=gpw_result.n_iter,
        converged=gpw_result.converged,
        mo_energies=np.asarray(gpw_result.mo_energies),
        mo_coeffs=np.asarray(gpw_result.mo_coeffs),
        density=D,
        fock=F,
        overlap=S,
        occupations=occupations,
        scf_trace=trace,
        one_centre=getattr(gpw_result, "one_centre", None),
        molecular_limit_declared=bool(
            getattr(gpw_result, "molecular_limit_declared", False)
        ),
    )


# ============================================================
# Open-shell (UHF / UKS) GPW adapter
# ============================================================


@dataclass
class GpwOpenShellRunnerResult:
    """Duck-typed wrapper around an open-shell GPW SCF result.

    Mirrors :class:`GpwRunnerResult` but adds the per-spin density,
    MO coefficient, MO energy, and occupation fields the runner's
    output / MO-summary code reaches for on UHF / UKS results
    (see :func:`vibeqc.periodic_runner._mo_summary`).

    ``density`` exposes the spin-summed AO density ``D_alpha +
    D_beta`` for the runner's molden / population paths that
    expect a single matrix; the per-spin matrices are available
    via ``density_alpha`` / ``density_beta``.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    e_coulomb: float
    e_xc: float
    e_hf_exchange: float
    n_iter: int
    converged: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    occupations: np.ndarray
    # Per-spin fields.
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    mo_coeffs_alpha: np.ndarray
    mo_coeffs_beta: np.ndarray
    density_alpha: np.ndarray
    density_beta: np.ndarray
    fock_alpha: np.ndarray
    fock_beta: np.ndarray
    occupations_alpha: np.ndarray
    occupations_beta: np.ndarray
    e_dft_plus_u: float = 0.0
    scf_trace: List[object] = field(default_factory=list)
    one_centre: Optional[str] = None
    molecular_limit_declared: bool = False

    restart_kpoints: object = None
    restart_weights: object = None
    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None


def _gpw_open_shell_to_runner(
    gpw_result,
    system,
    basis,
) -> GpwOpenShellRunnerResult:
    """Shared adapter body -- used by both UHF and UKS converters.

    Preserves stored F_alpha / F_beta + S when the producer exposes them, so
    callers (e.g., post-SCF analysis or Molden export) receive the operator
    that generated the reported eigenpairs. Older GPW-shaped objects fall
    back to reconstruction at their converged densities.
    """
    from .periodic_gapw_j import (
        _kinetic_lattice_gamma, _overlap_lattice_gamma,
        _ewald_v_ne_gamma, GpwJBuilder,
        collocate_density_on_grid,
    )

    bd = gpw_result.breakdown
    D_alpha = np.asarray(gpw_result.density_alpha)
    D_beta = np.asarray(gpw_result.density_beta)
    D_total = D_alpha + D_beta
    n_basis = D_alpha.shape[0]
    n_alpha = int(gpw_result.n_alpha)
    n_beta = int(gpw_result.n_beta)

    S_stored = getattr(gpw_result, "overlap", None)
    if S_stored is not None:
        S = np.asarray(S_stored)
    else:
        S = _overlap_lattice_gamma(basis, system)

    # Per-spin occupations: 1 for occupied, 0 for virtual.
    occ_alpha = np.zeros(n_basis, dtype=float)
    occ_alpha[:n_alpha] = 1.0
    occ_beta = np.zeros(n_basis, dtype=float)
    occ_beta[:n_beta] = 1.0
    # Combined "closed-shell-style" occupations for the runner's
    # single-vector path (alpha + beta -- sums to n_elec).
    occ_combined = occ_alpha + occ_beta

    # Energy decomposition.
    e_xc = float(getattr(bd, "e_xc", 0.0))
    e_electronic = (
        bd.e_kinetic
        + bd.e_nuclear_attraction
        + bd.e_hartree
        + bd.e_hf_exchange
        + e_xc
    )
    e_nuclear = bd.e_nuclear_repulsion
    energy = float(getattr(gpw_result, "energy", e_electronic + e_nuclear))
    e_coulomb = bd.e_hartree
    e_hf_exchange = bd.e_hf_exchange
    e_dft_plus_u = float(getattr(gpw_result, "e_dft_plus_u", 0.0))

    # GAPW UHF/UKS results carry the post-DIIS Focks that produced the
    # returned eigenpairs.  Use those exact augmented operators directly.
    # Reconstruct only for older GPW-shaped result objects: rebuilding first
    # would allocate another J/K/XC workspace and can fail for a route whose
    # stored augmented operator is already complete.
    F_alpha_stored = getattr(gpw_result, "fock_alpha", None)
    F_beta_stored = getattr(gpw_result, "fock_beta", None)
    if F_alpha_stored is not None and F_beta_stored is not None:
        F_alpha = np.asarray(F_alpha_stored)
        F_beta = np.asarray(F_beta_stored)
    else:
        T = _kinetic_lattice_gamma(basis, system)
        V_ne = _ewald_v_ne_gamma(basis, system)
        Hcore = T + V_ne
        gpw = GpwJBuilder(basis, gpw_result.grid)
        J = gpw.build_J(D_total)
        lo = _core.LatticeSumOptions()
        lo.cutoff_bohr = 25.0
        functional_name = getattr(bd, "functional", None)
        if functional_name is None:
            ex_frac = 1.0
            func = None
        else:
            func = _core.Functional(functional_name, 2)
            # Full-range-only per-spin K on this route.
            reject_unscreened_range_separated(
                func, where="_gpw_open_shell_to_runner"
            )
            ex_frac = float(func.hf_exchange_fraction)
        if ex_frac > 0.0:
            from .periodic_gapw_open_shell import _k_per_spin

            K_alpha = _k_per_spin(basis, system, D_alpha, lo)
            K_beta = _k_per_spin(basis, system, D_beta, lo)
        else:
            K_alpha = np.zeros_like(D_alpha)
            K_beta = np.zeros_like(D_alpha)
        F_alpha = Hcore + J - ex_frac * K_alpha
        F_beta = Hcore + J - ex_frac * K_beta
        dftu_sites = tuple(getattr(gpw_result, "dft_plus_u_sites", ()) or ())
        if dftu_sites:
            from .dft_plus_u import (
                _v_ao_per_spin as _dftu_v_ao_per_spin,
                ao_group_indices as _dftu_ao_group_indices,
            )

            dftu_ao_groups = _dftu_ao_group_indices(basis)
            F_alpha = F_alpha + S @ _dftu_v_ao_per_spin(
                dftu_sites,
                D_alpha,
                S,
                dftu_ao_groups,
            ) @ S
            F_beta = F_beta + S @ _dftu_v_ao_per_spin(
                dftu_sites,
                D_beta,
                S,
                dftu_ao_groups,
            ) @ S
        if func is not None:
            from .periodic_gapw_open_shell import (
                _evaluate_xc_polarised_on_grid,
                _project_vxc_polarised_to_ao,
            )

            rho_a_grid = collocate_density_on_grid(
                basis, D_alpha, gpw_result.grid
            )
            rho_b_grid = collocate_density_on_grid(
                basis, D_beta, gpw_result.grid
            )
            # basis + per-spin density matrices let meta-GGA functionals
            # build the per-spin t here (LDA/GGA ignore them); the returned
            # per-spin v_tau grids feed the reconstructed-Fock t term below.
            (
                _e_xc_re,
                v_xc_a_g, v_xc_b_g,
                v_sigma_aa_g, v_sigma_ab_g, v_sigma_bb_g,
                grad_a, grad_b,
                v_tau_a_g, v_tau_b_g,
            ) = _evaluate_xc_polarised_on_grid(
                rho_a_grid, rho_b_grid, gpw_result.grid, func,
                basis=basis,
                density_matrix_alpha=D_alpha,
                density_matrix_beta=D_beta,
            )
            V_xc_alpha = _project_vxc_polarised_to_ao(
                basis, "alpha", v_xc_a_g, gpw_result.grid,
                v_sigma_self=v_sigma_aa_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_a,
                grad_rho_other=grad_b,
            )
            V_xc_beta = _project_vxc_polarised_to_ao(
                basis, "beta", v_xc_b_g, gpw_result.grid,
                v_sigma_self=v_sigma_bb_g,
                v_sigma_cross=v_sigma_ab_g,
                grad_rho_self=grad_b,
                grad_rho_other=grad_a,
            )
            if v_tau_a_g is not None:
                from .periodic_gapw_j import _project_vtau_to_ao

                V_xc_alpha = V_xc_alpha + _project_vtau_to_ao(
                    basis, v_tau_a_g, gpw_result.grid
                )
                V_xc_beta = V_xc_beta + _project_vtau_to_ao(
                    basis, v_tau_b_g, gpw_result.grid
                )
            F_alpha = F_alpha + V_xc_alpha
            F_beta = F_beta + V_xc_beta

    # SCF trace lift to SCFIteration shape.
    raw_trace = getattr(gpw_result, "scf_trace", ()) or ()
    trace: List[_core.SCFIteration] = []
    if raw_trace:
        for step in raw_trace:
            trace.append(_core.SCFIteration(
                iter=int(step.get("iter", 0)),
                energy=float(step.get("energy", energy)),
                delta_e=float(step.get("delta_e", 0.0)),
                grad_norm=float(step.get("grad_norm", 0.0)),
                diis_subspace=0,
                newton_cg_iter=0,
                trah_level_shift=0.0,
            ))
    else:
        trace = [
            _core.SCFIteration(
                iter=it, energy=float(energy), delta_e=0.0,
                grad_norm=0.0, diis_subspace=0,
                newton_cg_iter=0, trah_level_shift=0.0,
            )
            for it in range(1, gpw_result.n_iter + 1)
        ]

    return GpwOpenShellRunnerResult(
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        guess_selection=getattr(gpw_result, "guess_selection", None),
        restart_kpoints=getattr(gpw_result, "restart_kpoints", None),
        restart_weights=getattr(gpw_result, "restart_weights", None),
        energy=energy,
        e_electronic=e_electronic,
        e_nuclear=e_nuclear,
        e_coulomb=e_coulomb,
        e_xc=e_xc,
        e_hf_exchange=e_hf_exchange,
        e_dft_plus_u=e_dft_plus_u,
        n_iter=gpw_result.n_iter,
        converged=gpw_result.converged,
        # Top-level (runner-default) views default to the alpha spin
        # for MO summary / molden export, which mirrors what
        # _mo_summary does (falls back to mo_energies_alpha when
        # mo_energies is absent). We expose mo_coeffs / mo_energies
        # as alpha so downstream code that ignores the per-spin
        # fields still sees something coherent.
        mo_energies=np.asarray(gpw_result.mo_energies_alpha),
        mo_coeffs=np.asarray(gpw_result.mo_coeffs_alpha),
        density=D_total,
        fock=F_alpha,
        overlap=S,
        occupations=occ_combined,
        mo_energies_alpha=np.asarray(gpw_result.mo_energies_alpha),
        mo_energies_beta=np.asarray(gpw_result.mo_energies_beta),
        mo_coeffs_alpha=np.asarray(gpw_result.mo_coeffs_alpha),
        mo_coeffs_beta=np.asarray(gpw_result.mo_coeffs_beta),
        density_alpha=D_alpha,
        density_beta=D_beta,
        fock_alpha=F_alpha,
        fock_beta=F_beta,
        occupations_alpha=occ_alpha,
        occupations_beta=occ_beta,
        scf_trace=trace,
        one_centre=getattr(gpw_result, "one_centre", None),
        molecular_limit_declared=bool(
            getattr(gpw_result, "molecular_limit_declared", False)
        ),
    )


def gpw_uhf_result_to_runner_shape(
    gpw_result,
    system,
    basis,
) -> GpwOpenShellRunnerResult:
    """Adapt a :class:`GpwUhfScfResult` to the runner's open-shell
    duck-type. Pure UHF -- no V_xc on the rebuilt Fock.
    """
    return _gpw_open_shell_to_runner(gpw_result, system, basis)


def gpw_uks_result_to_runner_shape(
    gpw_result,
    system,
    basis,
) -> GpwOpenShellRunnerResult:
    """Adapt a :class:`GpwUksScfResult` to the runner's open-shell
    duck-type. Preserves stored spin Focks when present; the compatibility
    fallback rebuilds V_xc from ``gpw_result.breakdown.functional``.
    """
    return _gpw_open_shell_to_runner(gpw_result, system, basis)


@dataclass
class GpwUksMultiKRunnerResult:
    """Duck-typed multi-k open-shell GPW result for the runner.

    Mirrors the attribute surface of
    :class:`vibeqc.periodic_k_gdf.PeriodicKUHFGDFResult` -- per-spin
    per-k LISTS -- so ``run_periodic_job``'s existing multi-k open-shell
    output machinery (MO summary, band-edge scan, density/molden/QVF
    writers) consumes a multi-k GPW UKS result exactly like a multi-k
    GDF UKS one. On the compact Bloch path the per-k MO count can be
    below ``n_basis`` (linear-dependence projection), which the
    list-shaped consumers already tolerate.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float

    mo_energies_alpha: List[np.ndarray]
    mo_coeffs_alpha: List[np.ndarray]
    density_alpha: List[np.ndarray]
    fock_alpha: List[np.ndarray]
    mo_energies_beta: List[np.ndarray]
    mo_coeffs_beta: List[np.ndarray]
    density_beta: List[np.ndarray]
    fock_beta: List[np.ndarray]

    overlap: List[np.ndarray]
    hcore: List[np.ndarray]
    kpoints_cart: np.ndarray
    kpoint_weights: np.ndarray
    occupations_alpha: List[np.ndarray]
    occupations_beta: List[np.ndarray]

    functional: Optional[str] = None
    e_xc: float = 0.0
    e_coulomb: float = 0.0
    e_hf_exchange: float = 0.0
    e_dft_plus_u: float = 0.0
    backend: str = "gpw-uks-multi-k"
    method: str = "uks"
    scf_trace: List[object] = field(default_factory=list)

    @property
    def energy_per_cell_ha(self) -> float:
        return float(self.energy)

    restart_kpoints: object = None
    restart_weights: object = None
    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None


def gpw_uks_multik_result_to_runner_shape(
    gpw_result,
    system,
    basis,
) -> GpwUksMultiKRunnerResult:
    """Adapt a :class:`GpwUksMultiKScfResult` to the runner's multi-k
    open-shell shape (see :class:`GpwUksMultiKRunnerResult`).

    Rebuilds the per-k pieces the GPW driver does not carry:

    * ``overlap`` / ``hcore`` per k -- Bloch sums of the one-electron
      lattice matrices in the same EWALD_3D V_ne gauge the driver used;
    * per-k per-spin densities ``D_s(k) = C_occ C_occ^H`` from the
      returned coefficients + integer occupations;
    * per-k per-spin Fock reconstructed on the solved subspace as
      ``F(k) = S(k) C(k) diag(eps) C(k)^H S(k)`` -- exact wherever the
      eigenproblem was solved (output/dump use only; the SCF itself never
      consumes it);
    * ``<S^2>`` via the shared multi-k spin-contamination diagnostic.
    """
    from .periodic_k_gdf import _multi_k_s_squared
    from .periodic_v_ne import compute_nuclear_lattice_dispatch

    kmesh = gpw_result.kmesh
    kpoints = np.asarray(kmesh.kpoints, dtype=float)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = kpoints.shape[0]
    n_alpha = int(gpw_result.n_alpha)
    n_beta = int(gpw_result.n_beta)

    from .periodic_gapw_j import (
        multik_one_electron_lattice_options, multik_v_ne_lattice_options,
    )
    lat_opts = multik_one_electron_lattice_options(basis, system)
    T_lat = _core.compute_kinetic_lattice(basis, system, lat_opts)
    S_lat = _core.compute_overlap_lattice(basis, system, lat_opts)
    lat_opts_v = multik_v_ne_lattice_options(basis, system)
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts_v)

    S_k: List[np.ndarray] = []
    hcore_k: List[np.ndarray] = []
    for ik in range(n_k):
        k = kpoints[ik]
        Sk = np.asarray(_core.bloch_sum(S_lat, k))
        Tk = np.asarray(_core.bloch_sum(T_lat, k))
        Vk = np.asarray(_core.bloch_sum(V_lat, k))
        S_k.append(0.5 * (Sk + Sk.conj().T))
        Hk = Tk + Vk
        hcore_k.append(0.5 * (Hk + Hk.conj().T))

    C_a_k = [np.asarray(C) for C in gpw_result.mo_coeffs_alpha_k]
    C_b_k = [np.asarray(C) for C in gpw_result.mo_coeffs_beta_k]
    e_a_k = [np.asarray(np.real(e), dtype=float) for e in gpw_result.mo_energies_alpha_k]
    e_b_k = [np.asarray(np.real(e), dtype=float) for e in gpw_result.mo_energies_beta_k]

    def _occ(n_occ: int, n_mo: int) -> np.ndarray:
        occ = np.zeros(n_mo, dtype=float)
        occ[: min(n_occ, n_mo)] = 1.0
        return occ

    occ_a_k = [_occ(n_alpha, C.shape[1]) for C in C_a_k]
    occ_b_k = [_occ(n_beta, C.shape[1]) for C in C_b_k]

    def _dens(C: np.ndarray, n_occ: int) -> np.ndarray:
        if n_occ <= 0:
            return np.zeros((C.shape[0], C.shape[0]), dtype=complex)
        Co = C[:, :n_occ]
        return Co @ Co.conj().T

    D_a_k = [_dens(C_a_k[ik], n_alpha) for ik in range(n_k)]
    D_b_k = [_dens(C_b_k[ik], n_beta) for ik in range(n_k)]

    def _fock(Sk: np.ndarray, C: np.ndarray, eps: np.ndarray) -> np.ndarray:
        # F = S C diag(eps) C^H S: satisfies F C = S C eps on the solved
        # subspace (C^H S C = I), which is all the output writers need.
        SC = Sk @ C
        F = (SC * eps[None, :]) @ SC.conj().T
        return 0.5 * (F + F.conj().T)

    F_a_k = [_fock(S_k[ik], C_a_k[ik], e_a_k[ik]) for ik in range(n_k)]
    F_b_k = [_fock(S_k[ik], C_b_k[ik], e_b_k[ik]) for ik in range(n_k)]

    s_z = 0.5 * (n_alpha - n_beta)
    s_sq_ideal = s_z * (s_z + 1.0)
    if getattr(gpw_result, "method", "uks") == "roks":
        s_sq = float(s_sq_ideal)
    else:
        s_sq = float(
            _multi_k_s_squared(
                n_alpha, n_beta, C_a_k, C_b_k, S_k, list(weights)
            )
        )

    bd = gpw_result.breakdown
    e_xc = float(getattr(bd, "e_xc", 0.0) or 0.0)
    e_electronic = float(
        bd.e_kinetic
        + bd.e_nuclear_attraction
        + bd.e_hartree
        + bd.e_hf_exchange
        + e_xc
    )

    return GpwUksMultiKRunnerResult(
               restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        guess_selection=getattr(gpw_result, "guess_selection", None),
        restart_kpoints=getattr(gpw_result, "restart_kpoints", None),
        restart_weights=getattr(gpw_result, "restart_weights", None),
        energy=float(gpw_result.energy),
        e_electronic=e_electronic,
        e_nuclear=float(bd.e_nuclear_repulsion),
        n_iter=int(gpw_result.n_iter),
        converged=bool(gpw_result.converged),
        s_squared=s_sq,
        s_squared_ideal=float(s_sq_ideal),
        mo_energies_alpha=e_a_k,
        mo_coeffs_alpha=C_a_k,
        density_alpha=D_a_k,
        fock_alpha=F_a_k,
        mo_energies_beta=e_b_k,
        mo_coeffs_beta=C_b_k,
        density_beta=D_b_k,
        fock_beta=F_b_k,
        overlap=S_k,
        hcore=hcore_k,
        kpoints_cart=kpoints,
        kpoint_weights=weights,
        occupations_alpha=occ_a_k,
        occupations_beta=occ_b_k,
        functional=getattr(bd, "functional", None),
        e_xc=e_xc,
        e_coulomb=float(bd.e_hartree),
        e_hf_exchange=float(bd.e_hf_exchange),
        backend=(
            "gpw-roks-multi-k"
            if getattr(gpw_result, "method", "uks") == "roks"
            else "gpw-uks-multi-k"
        ),
        method=str(getattr(gpw_result, "method", "uks")),
        scf_trace=list(gpw_result.scf_trace),
    )


def gpw_roks_multik_result_to_runner_shape(
    gpw_result,
    system,
    basis,
) -> GpwUksMultiKRunnerResult:
    """Adapt a restricted multi-k GPW ROKS result to runner shape."""
    return gpw_uks_multik_result_to_runner_shape(gpw_result, system, basis)
