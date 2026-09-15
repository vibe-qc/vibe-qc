"""Tiny pluggable Γ-point RHF SCF driver for the J-builder zoo.

Mirrors the structure of vibeqc.periodic_rhf_ewald.run_rhf_periodic_gamma_ewald3d
but lets the caller swap in any J-builder (EWALD3D / WOLF / PLAIN_EWALD / ADFT).

Scope:
- Γ-only, closed-shell RHF.
- Orthorhombic cells (inherited from FFT-Poisson constraint).
- DIIS or simple damping.
- Same SAD initial guess as the release path.

This is throwaway scaffolding — does not aim to match the release
driver feature-for-feature.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vibeqc import (
    BasisSet,
    InitialGuess,
    LatticeSumOptions,
    PeriodicSystem,
    bloch_sum,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    nuclear_repulsion_per_cell,
    sad_density,
)
from vibeqc.madelung import madelung_energy_correction
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch
from vibeqc import CoulombMethod

from j_builders import ADFTContext, build_jk


@dataclass
class SCFResult:
    energy: float
    e_elec: float
    e_nuc: float
    e_madelung: float
    converged: bool
    n_iter: int
    wall_s: float
    method: str
    notes: str = ""


def _canonical_orth(S: np.ndarray, threshold: float = 1e-7):
    """Build canonical orthogonalisation matrix X = U Σ^{-1/2} from S."""
    eigvals, eigvecs = np.linalg.eigh(0.5 * (S + S.T))
    mask = eigvals > threshold
    kept_vals = eigvals[mask]
    kept_vecs = eigvecs[:, mask]
    return kept_vecs / np.sqrt(kept_vals), int(mask.sum())


def _diis_extrapolate(F_history, e_history, n_keep: int = 6):
    """Pulay DIIS on a list of (F, error) pairs."""
    if len(F_history) < 2:
        return F_history[-1]
    # Trim to last n_keep
    F_history = F_history[-n_keep:]
    e_history = e_history[-n_keep:]
    n = len(F_history)
    B = np.zeros((n + 1, n + 1))
    B[-1, :-1] = -1.0
    B[:-1, -1] = -1.0
    for i in range(n):
        for j in range(n):
            B[i, j] = np.einsum("ij,ij->", e_history[i], e_history[j])
    rhs = np.zeros(n + 1)
    rhs[-1] = -1.0
    try:
        c = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        return F_history[-1]
    F_extrap = np.zeros_like(F_history[-1])
    for i in range(n):
        F_extrap += c[i] * F_history[i]
    return F_extrap


def run_rhf_method(
    method: str,
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    omega: float = 0.5,
    spacing_bohr: float = 0.3,
    alpha_wolf: float = 0.5,
    g_cutoff_factor: float = 8.0,
    aux_basis: Optional[BasisSet] = None,
    max_iter: int = 60,
    conv_tol_energy: float = 1e-7,
    damping: float = 0.5,
    use_diis: bool = True,
    diis_start_iter: int = 1,
    initial_guess: InitialGuess = InitialGuess.SAD,
    verbose: bool = False,
) -> SCFResult:
    """Γ-point RHF with pluggable J-method.

    For ``method == "ADFT"`` the caller must supply ``aux_basis``.
    """
    t0 = time.perf_counter()

    # Per-method gauge bookkeeping for the nuclear-attraction integrals:
    # EWALD_3D V_ne carries the same -α_M Q/L scalar shift that
    # ``build_j_ewald_3d`` produces by dropping G=0; the two cancel
    # in the SCF energy. WOLF and ADFT produce a "molecular" J with
    # no G=0 gauge shift, so the matching V_ne must be the bare
    # DIRECT_TRUNCATED lattice sum — otherwise the shift is unbalanced
    # and the SCF total drops by ~α_M Q²/(2L) ≈ 0.47 Ha for H₂ in a
    # 12-bohr box (the exact symptom we hit on first iteration).
    methods_gauge_aligned = {"EWALD3D", "PLAIN_EWALD"}
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()
        if method in methods_gauge_aligned:
            lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
        else:
            lattice_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED
        lattice_opts.cutoff_bohr = 12.0
        lattice_opts.nuclear_cutoff_bohr = 25.0

    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError("RHF requires closed-shell")
    n_occ = n_elec // 2

    # 1-e ints at Γ.
    S_lat = compute_overlap_lattice(basis, system, lattice_opts)
    T_lat = compute_kinetic_lattice(basis, system, lattice_opts)
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lattice_opts)
    k = np.zeros(3)
    S = np.real(bloch_sum(S_lat, k))
    T = np.real(bloch_sum(T_lat, k))
    V = np.real(bloch_sum(V_lat, k))
    S = 0.5 * (S + S.T)
    Hcore = 0.5 * ((T + V) + (T + V).T)

    X, n_kept = _canonical_orth(S, threshold=1e-7)
    if n_occ > n_kept:
        raise RuntimeError(
            f"Canonical orth dropped too many; n_occ={n_occ}, kept={n_kept}"
        )

    e_nuc = nuclear_repulsion_per_cell(system, lattice_opts)

    # ADFT context.
    adft_ctx = None
    if method == "ADFT":
        if aux_basis is None:
            raise ValueError("ADFT needs aux_basis")
        adft_ctx = ADFTContext.build(basis, aux_basis)

    # Initial guess.
    if initial_guess == InitialGuess.SAD:
        D = np.asarray(sad_density(system.unit_cell_molecule(), basis))
    else:
        Fp = X.T @ Hcore @ X
        eps, Cp = np.linalg.eigh(0.5 * (Fp + Fp.T))
        C = X @ Cp
        Cocc = C[:, :n_occ]
        D = 2.0 * Cocc @ Cocc.T

    # Madelung correction policy:
    # - EWALD3D / PLAIN_EWALD: J and V_ne both Ewald-gauged → already
    #   cancel, no correction.
    # - WOLF / ADFT: J and V_ne both gauge-clean (DIRECT_TRUNCATED) →
    #   no leak, no correction. Set 0 in both cases. We keep the
    #   fallback machinery in place so a future method that mixes
    #   gauges can opt back in by setting ``methods_need_madelung``.
    methods_need_madelung: set[str] = set()

    F_hist: list = []
    err_hist: list = []
    e_old = 0.0
    e_mad_last = 0.0
    converged = False
    n_iter = 0
    j_walls = []
    k_walls = []
    notes = ""

    for it in range(max_iter):
        n_iter = it + 1
        # Build J/K.
        try:
            J, K, t_jk = build_jk(
                method, basis, system, lattice_opts, D,
                omega=omega, spacing_bohr=spacing_bohr,
                alpha_wolf=alpha_wolf, g_cutoff_factor=g_cutoff_factor,
                adft_ctx=adft_ctx,
            )
        except Exception as e:
            notes = f"J-build failed at iter {it+1}: {e!r}"
            break
        j_walls.append(t_jk["J"])
        k_walls.append(t_jk["K"])

        F = Hcore + J - 0.5 * K
        F = 0.5 * (F + F.T)

        # DIIS error vector: F D S - S D F.
        err = F @ D @ S - S @ D @ F
        err_norm = np.linalg.norm(err)

        if use_diis and it >= diis_start_iter:
            F_hist.append(F.copy())
            err_hist.append(err.copy())
            F = _diis_extrapolate(F_hist, err_hist)

        # Diagonalise.
        Fp = X.T @ F @ X
        Fp = 0.5 * (Fp + Fp.T)
        eps, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        Cocc = C[:, :n_occ]
        D_new = 2.0 * Cocc @ Cocc.T

        # Damping.
        if damping > 0.0 and it > 0:
            D_new = (1.0 - damping) * D_new + damping * D

        # Energy.
        e_elec = 0.5 * np.einsum("ij,ji->", D_new, Hcore + F, optimize=True)
        if method in methods_need_madelung:
            e_mad = madelung_energy_correction(
                D_new, S, system, nuclear_uses_ewald=False,
            )
        else:
            e_mad = 0.0
        e_mad_last = e_mad
        e_total = float(e_elec) + e_nuc + e_mad

        if verbose:
            print(f"  iter {it+1:3d}  E = {e_total:.10f}  ΔE = {e_total-e_old:+.3e}  "
                  f"|err| = {err_norm:.3e}  J_wall = {t_jk['J']:.2f}s")

        if it > 0 and abs(e_total - e_old) < conv_tol_energy and err_norm < 1e-5:
            converged = True
            D = D_new
            e_old = e_total
            break

        e_old = e_total
        D = D_new

    wall = time.perf_counter() - t0

    return SCFResult(
        energy=float(e_old),
        e_elec=float(e_old - e_nuc - e_mad_last),
        e_nuc=float(e_nuc),
        e_madelung=float(e_mad_last),
        converged=converged,
        n_iter=n_iter,
        wall_s=wall,
        method=method,
        notes=notes,
    )
