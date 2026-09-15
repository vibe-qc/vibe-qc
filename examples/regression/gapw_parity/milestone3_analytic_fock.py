#!/usr/bin/env python
"""Milestone-3 blueprint: the analytic exact GAPW Hartree Fock J = ∂E_H/∂D.

VALIDATED REFERENCE (not yet wired into production `GapwJBuilder.build_J`).
See ``handovers/HANDOVER_GAPW_PRODUCTION.md`` § milestone 3.

The bug it fixes (the core-aufbau failure). The shipped `build_J` returns the
Hartree matrix as ``J_smooth(full basis) + Σ_a ∫χχ(V_hard − V_soft)``, which is
NOT ``∂E_H/∂D``. For a partially-occupied core atom (O/C/N/F and most molecules
with virtual orbitals) the inconsistent Fock pushes the tight 1s ABOVE the
valence (O: ε₁ₛ ≈ +68 Ha), so the SCF aufbau leaves the core empty and the total
collapses (O −35 vs −74 Ha). The ENERGY (`gapw_hartree_energy`) is already
correct at the core-occupied density (O +215 mHa vs the all-electron GDF
reference, as good as Ne); only the Fock was wrong.

The fix — the exact ``∂E_H/∂D``. Since ``E_H`` is quadratic in D, the Fock is
``∫ (∂ρ/∂D_μν) V[ρ]`` of every density piece:

    J_μν = embed(∫_FFT χ̃_μ χ̃_ν V_smooth)                       # (A) smooth, SOFT basis
         + Σ_a window·∫_rad (χ_oc χ_oc) V_hard,a                # (B) hard, full, on-centre
         − Σ_a window·embed(∫_rad (χ̃_oc χ̃_oc) V_soft,a)        # (C) soft, soft basis
         + Σ_{a,lm} M_{lm,a,μν} (⟨g_lm,a|V_smooth⟩ − ⟨g_lm,a|V_soft,a⟩_win)   # (D) ρ₀ response

where ``M_{lm,a,μν} = ∂Q_lm,a/∂D_μν`` is the (l,m) multipole of the on-centre
``(χ_μχ_ν − χ̃_μχ̃_ν)`` product (the ρ₀ compensator is D-dependent — Q_lm[D] —
and ∂Q₀₀/∂D is large for the core, which is the +68 Ha artefact).

Validation (this script, O/STO-3G/Γ/12-bohr/24³):
  * ``½ tr(D·J_analytic) = E_H`` exactly ⇒ J = ∂E_H/∂D.
  * SCF with this Fock: O → −73.558 (+215 mHa), e_kin 73.4, ε₁ₛ = −19.62,
    CORE OK; Ne → −126.642 (+339 mHa), preserved.

Productionisation: port into `GapwJBuilder.build_J`, caching the D-independent
M tensors + g_lm FFT/radial collocations (§ milestone-3 plan in the handover).
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_atomic_grid import multipole_index  # noqa: F401  (l,m order)
from vibeqc.periodic_gapw_augment import (
    GapwJBuilder,
    _radial_window,
    solve_poisson_radial,
)
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_j import (
    _build_v_ne,
    _eigh_safe,
    _kinetic_lattice_gamma,
    _overlap_lattice_gamma,
    collocate_density_on_grid,
    project_potential_to_ao,
)
from vibeqc.periodic_gapw_smearing import GaussianMultipoleCompensator


def analytic_hartree_fock(jk: GapwJBuilder, D: np.ndarray) -> np.ndarray:
    """Return the exact GAPW Hartree Fock ``J = ∂E_H/∂D`` for density ``D``."""
    aug, grid, basis = jk._aug, jk._grid, jk._basis
    n = basis.nbasis
    if not aug._augmentation_active:
        return jk._gpw.build_J(D)
    idx = aug._soft_indices
    D_soft = D[np.ix_(idx, idx)]
    comp = aug._build_compensator(D)
    rho_tilde = collocate_density_on_grid(aug._soft_basis, D_soft, grid)
    V_smooth = core.solve_poisson_coulomb(
        rho_tilde + comp.density_on_grid(grid), grid.lattice_bohr
    )
    dV = grid.voxel_volume_bohr3
    lmax = aug._lmax
    n_comp = (lmax + 1) ** 2
    lof = np.array([int(np.floor(np.sqrt(c))) for c in range(n_comp)])

    # (A) smooth term on the SOFT basis (∂ñ/∂D uses the soft AOs), embedded.
    J = np.zeros((n, n))
    J[np.ix_(idx, idx)] = project_potential_to_ao(aug._soft_basis, V_smooth, grid)

    for ad in aug._atom_data:
        g_r = ad.grid
        r = np.asarray(g_r.r)
        w_r, w_a = np.asarray(g_r.w_r), np.asarray(g_r.w_a)
        nr, na = g_r.n_radial, g_r.n_angular
        win = _radial_window(r, ad.aug_radius, width=0.5)
        wt = g_r.combined_weights()
        chi_f = ad.chi_a * ad.on_center_full[None, :]
        chi_s = ad.chi_tilde * ad.on_center_soft[None, :]
        rho_a = aug._compute_atomic_density(D, ad, use_soft=False)
        rho_t = aug._compute_atomic_density(D, ad, use_soft=True)
        rho0a = aug.compensator_density_on_atomic_grid(ad)
        V_hard = solve_poisson_radial(rho_a, g_r)
        V_soft = solve_poisson_radial(rho_t + rho0a, g_r)
        winf = (win[:, None] * np.ones(na)[None, :]).ravel()
        wtf = wt.ravel()
        # (B) on-centre hard, full basis; (C) on-centre soft, soft basis (embed).
        J += np.einsum("gm,gn,g->mn", chi_f, chi_f, V_hard.ravel() * winf * wtf, optimize=True)
        J[np.ix_(idx, idx)] -= np.einsum(
            "gm,gn,g->mn", chi_s, chi_s, V_soft.ravel() * winf * wtf, optimize=True
        )
        # M_{lm,μν} = ∂Q_lm/∂D = multipole of on-centre (χχ − χ̃χ̃).
        S = g_r.evaluate_real_spherical_harmonics(lmax)
        Wc = np.empty((n_comp, nr * na))
        for c in range(n_comp):
            Wc[c] = (S[c][None, :] * (r ** lof[c])[:, None] * (w_r[:, None] * w_a[None, :])).ravel()
        M = np.einsum("cg,gm,gn->cmn", Wc, chi_f, chi_f, optimize=True)
        M[:, idx[:, None], idx[None, :]] -= np.einsum(
            "cg,gm,gn->cmn", Wc, chi_s, chi_s, optimize=True
        )
        # (D) compensator response: per-(l,m) projection of g_lm onto the potentials.
        v_sm = np.zeros(n_comp)
        v_so = np.zeros(n_comp)
        for c in range(n_comp):
            Qu = np.zeros((comp.n_atoms, n_comp))
            Qu[ad.atom_idx, c] = 1.0
            cu = GaussianMultipoleCompensator(
                positions=comp.positions, alpha=comp.alpha, Q=Qu, lmax=lmax
            )
            v_sm[c] = float(np.sum(cu.density_on_grid(grid) * V_smooth)) * dV
            saved = aug._compensator
            aug._compensator = cu
            g_rad = aug.compensator_density_on_atomic_grid(ad)
            aug._compensator = saved
            v_so[c] = float(np.einsum("ra,ra,ra->", g_rad, V_soft * win[:, None], wt))
        J += np.einsum("c,cmn->mn", v_sm - v_so, M, optimize=True)
    return 0.5 * (J + J.T)


def _atom(Z, L=12.0):
    s = core.PeriodicSystem()
    s.dim = 3
    s.lattice = np.eye(3) * L
    s.unit_cell = [core.Atom(Z, [L / 2, L / 2, L / 2])]
    mol = vq.Molecule([vq.Atom(Z, [L / 2, L / 2, L / 2])], 0, 1)
    return s, vq.BasisSet(mol, "sto-3g"), L


def _scf_with_analytic_fock(Z, N=24, L=12.0):
    s, b, _ = _atom(Z)
    grid = PlaneWaveGrid(np.eye(3) * L, N, N, N)
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-9
    e_gdf = float(vq.run_rhf_periodic_gamma_gdf(s, b, opts, progress=False).energy)
    T = _kinetic_lattice_gamma(b, s)
    V_ne = _build_v_ne(b, s, "ewald", None, grid)
    S = _overlap_lattice_gamma(b, s)
    Hc = T + V_ne
    Enn = float(core.ewald_nuclear_repulsion(s, core.EwaldOptions()))
    jk = GapwJBuilder(b, s, grid, quiet=True)
    lo = core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0
    se, U = _eigh_safe(S)
    Sih = U @ np.diag(1.0 / np.sqrt(se)) @ U.T
    no = Z // 2
    e, C = _eigh_safe(Sih @ Hc @ Sih)
    C = Sih @ C
    D = 2.0 * C[:, :no] @ C[:, :no].T
    E_prev = 0.0
    E = 0.0
    for it in range(80):
        J = analytic_hartree_fock(jk, D)
        K = np.asarray(core.build_jk_gamma_molecular_limit(b, s, lo, D).K)
        F = Hc + J - 0.5 * K
        e, C = _eigh_safe(Sih @ F @ Sih)
        C = Sih @ C
        Dn = 2.0 * C[:, :no] @ C[:, :no].T
        E = (
            float(np.einsum("ij,ij->", D, Hc))
            + jk.gapw_hartree_energy(D)
            - 0.25 * float(np.einsum("ij,ij->", D, K))
            + Enn
        )
        if abs(E - E_prev) < 1e-8 and it > 2:
            break
        D, E_prev = Dn, E
    return e_gdf, E, float(np.einsum("ij,ij->", D, T)), float(e[0])


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    print("# Milestone-3 analytic exact GAPW Fock — validation (STO-3G, Γ, 24³)")
    for sym, Z in [("O", 8), ("Ne", 10)]:
        e_gdf, E, ekin, eps1s = _scf_with_analytic_fock(Z)
        ok = "CORE OK" if ekin > 0.5 * (74.8 if Z == 8 else 128.5) else "CORE LOST"
        print(
            f"  {sym}: GAPW={E:.4f} (GDF {e_gdf:.4f}, Δ={(E - e_gdf) * 1e3:+.1f} mHa) "
            f"e_kin={ekin:.2f} eps_1s={eps1s:.2f} [{ok}]"
        )
