"""Projector-based one-centre expansions for the GAPW augmentation.

> **Experimental, energy-evaluator only.** This is the validated
> construction from the 2026-07-29 oracle work
> (``examples/regression/gapw_parity/oracle_projector_augmentation_wip2.py``,
> ``handovers/HANDOVER_GAPW_PRODUCTION.md`` "NEXT MILESTONE"): it closes
> the H2O fixed-density Hartree identity to +9 mHa where the production
> block-restricted augmentation is -4213 mHa off. The production SCF
> does NOT use this module yet - the analytic Fock (dE/dD of this
> energy) has to land with the SCF port so ``1/2 tr(D J) = E_H`` holds
> (the milestone-3 lesson). Until then this is the reference energy
> evaluator the port must reproduce.

The construction (Lippert, Hutter & Parrinello, Theor. Chem. Acc. 103,
124 (1999), Sec. 4.2; Krack & Parrinello, PCCP 2, 2105 (2000),
Eqs. 11-25), specialised to vibe-qc's single-compensator arrangement:

    E_H[n] ~= E_sm[ntilde + n0]                          (FFT, zero-mean)
            + sum_A { E_rad[n1_A] - E_rad[ntilde1_A + n0_A] }  (radial)

with four load-bearing choices, each verified on H2O/STO-3G:

1. Augmentation spheres ONLY on atoms whose basis was actually pruned
   by ``softened_basis``. An unpruned atom's hard and soft one-centre
   content is identical, so its augmentation is exactly zero.
2. The one-centre density n1_A is built from ALL basis functions:
   on-centre AOs enter exactly through their own contraction
   coefficients; OFF-centre AOs enter through a region-weighted
   least-squares fit in a confined primitive set over a sphere whose
   radius follows the PRUNED-CONTENT support. The fit region must not
   reach another pruned atom's core.
3. The hard and soft expansions SHARE their off-centre coefficients
   (LHP99 Eq. 52), so the off-centre fits cancel exactly in
   delta_A = n1_A - ntilde1_A - n0_A.
4. Per-lm multipole compensation (default lmax 4) with the same
   numerically defined moments on the radial and FFT sides, and NO
   radial window - the confined expansion makes whole-grid integrals
   safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _vibeqc_core as _core
from .periodic_gapw_atomic_grid import real_spherical_harmonics

__all__ = [
    "AnalyticAugmentation",
    "OneCentreExpansion",
    "build_analytic_augmentation",
    "build_one_centre_expansions",
    "projector_hartree_energy",
    "projector_hartree_energy_and_fock",
    "projector_xc_correction",
    "pruned_atom_indices",
]

_DEFAULT_LMAX = 4
_DEFAULT_L_EXP = 3
_DEFAULT_N_CONF = 6
_DEFAULT_CONF_SCALE = 3.0
_DEFAULT_COMP_ALPHA = 2.25  # compensator Gaussian exponent (bohr^-2)


def pruned_atom_indices(basis, soft_basis) -> list[int]:
    """Atoms whose shell set differs between the hard and soft basis.

    Only these atoms carry an augmentation sphere: an unpruned atom's
    hard and soft one-centre content is identical, so its hard-soft
    augmentation difference is exactly zero.
    """
    def _sig(b, ia):
        out = []
        for sh in b.shells():
            if int(sh.atom_index) != ia:
                continue
            out.append((int(sh.l), tuple(round(float(e), 10)
                                         for e in sh.exponents)))
        return tuple(sorted(out))

    n_atoms = 1 + max(int(sh.atom_index) for sh in basis.shells())
    return [ia for ia in range(n_atoms)
            if _sig(basis, ia) != _sig(soft_basis, ia)]


@dataclass
class OneCentreExpansion:
    """D-independent one-centre expansion data for one pruned atom.

    ``prim_basis`` is the expansion primitive set (the atom's own
    primitives plus the confined all-l progression); ``c_hard`` /
    ``c_soft`` map the full / soft AO vectors into that set (off-centre
    columns shared between the two). ``chi`` caches the primitives
    evaluated on the atom's radial grid.
    """

    atom_index: int
    center: np.ndarray
    radius: float
    prim_basis: object
    c_hard: np.ndarray   # (n_prim, n_ao_hard)
    c_soft: np.ndarray   # (n_prim, n_ao_soft)
    chi: np.ndarray      # (n_grid_pts, n_prim) on the radial grid


def _make_prim_basis(prims, mol):
    shells = []
    for l, e, orig in prims:
        sh = _core.ShellInfo()
        sh.atom_index = 0
        sh.l = int(l)
        sh.pure = True
        sh.exponents = [float(e)]
        sh.coefficients = [1.0]
        sh.origin = list(orig)
        shells.append(sh)
    return _core.BasisSet(mol, shells, "gapw-projector-prims",
                          coefficients_pre_normalized=True)


def _ao_values(b, pts_flat):
    v = np.asarray(_core.evaluate_ao(b, pts_flat))
    if v.shape[0] == b.nbasis:
        v = v.T
    return v


def _pruned_support_radius(basis, soft_basis, ia) -> float:
    """Radius containing the pruned-content density support.

    The slowest pruned-x-kept primitive product on the atom decays as
    exp(-(a_pruned_min + a_kept_min) r^2); the radius where that falls
    to 1e-4 bounds the augmentation difference's support.
    """
    hard_exps = {}
    soft_exps = {}
    for b, store in ((basis, hard_exps), (soft_basis, soft_exps)):
        for sh in b.shells():
            if int(sh.atom_index) != ia:
                continue
            store.setdefault(int(sh.l), set()).update(
                round(float(e), 10) for e in sh.exponents)
    pruned = []
    kept = []
    for l, exps in hard_exps.items():
        kept_l = soft_exps.get(l, set())
        pruned.extend(exps - kept_l)
        kept.extend(kept_l)
    if not pruned:
        return 0.0
    a_pair = min(pruned) + (min(kept) if kept else min(pruned))
    return math.sqrt(math.log(1e4) / a_pair)


def build_one_centre_expansions(
    basis,
    soft_basis,
    system,
    atom_grids,
    *,
    l_exp: int = _DEFAULT_L_EXP,
    n_conf: int = _DEFAULT_N_CONF,
    conf_scale: float = _DEFAULT_CONF_SCALE,
) -> list[OneCentreExpansion]:
    """Build the D-independent expansion data for every pruned atom.

    ``atom_grids`` maps atom index -> AtomicRadialGrid (the same grids
    the augmentation uses). Raises if two pruned atoms' fit spheres
    would overlap (the H2O flat-region experiment showed that breaks
    the identity at the Ha level).
    """
    from .molecule import Molecule

    atoms = list(system.unit_cell)
    n_elec = int(sum(int(a.Z) for a in atoms))
    mol = Molecule(atoms, 0, 2 if n_elec % 2 else 1)

    pruned = pruned_atom_indices(basis, soft_basis)
    positions = np.asarray([list(a.xyz) for a in atoms], dtype=float)

    radii = {}
    for ia in pruned:
        r_support = _pruned_support_radius(basis, soft_basis, ia)
        # Floor at the oracle-validated robust window (1.4-2.0 bohr on
        # H2O): the 1e-4-amplitude support estimate runs slightly tight
        # (1.30 for O/STO-3G costs ~35 mHa vs the window).
        radii[ia] = max(r_support, 1.4)
    for i in pruned:
        for j in pruned:
            if i < j:
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if radii[i] + radii[j] > d:
                    raise NotImplementedError(
                        "gapw-projector: pruned atoms "
                        f"{i} and {j} are {d:.2f} bohr apart but their "
                        f"augmentation spheres need "
                        f"{radii[i]:.2f}+{radii[j]:.2f} bohr; "
                        "overlapping pruned spheres are not supported "
                        "yet (the fit region of one atom would reach "
                        "the other's core)."
                    )

    out = []
    for ia in pruned:
        orig = list(positions[ia])
        own = []
        seen = set()
        for sh in basis.shells():
            if int(sh.atom_index) != ia:
                continue
            for e in sh.exponents:
                key = (int(sh.l), round(float(e), 10))
                if key not in seen:
                    seen.add(key)
                    own.append((int(sh.l), float(e), orig))
        R_A = radii[ia]
        a_min = math.log(100.0) / (R_A ** 2) * conf_scale
        conf = []
        for i in range(n_conf):
            e = a_min * (2.0 ** i)
            for l in range(l_exp + 1):
                if (l, round(e, 10)) not in seen:
                    conf.append((l, float(e), orig))
        prims = own + conf
        pb = _make_prim_basis(prims, mol)
        n_g = pb.nbasis

        prim_pos = {}
        conf_cols = []
        k = 0
        for j, (l, e, _o) in enumerate(prims):
            for m in range(2 * l + 1):
                prim_pos[(l, round(e, 10), m)] = k
                if j >= len(own):
                    conf_cols.append(k)
                k += 1
        cc = np.asarray(conf_cols, dtype=int)

        g = atom_grids[ia]
        pts = np.asarray(g.cartesian_points()).reshape(-1, 3)
        wt = np.asarray(g.combined_weights()).ravel()
        r = np.linalg.norm(pts - np.asarray(orig), axis=1)
        mask = r <= R_A
        P = pts[mask]
        sw = np.sqrt(wt[mask])
        G = _ao_values(pb, P)
        A = G[:, cc] * sw[:, None]

        def _coeffs(target_b, exact_on_centre: bool):
            Phi = _ao_values(target_b, P)
            C = np.zeros((n_g, target_b.nbasis))
            col = 0
            for sh in target_b.shells():
                l = int(sh.l)
                n_m = 2 * l + 1
                if int(sh.atom_index) == ia:
                    for m in range(n_m):
                        for e_, c_ in zip(sh.exponents, sh.coefficients):
                            C[prim_pos[(l, round(float(e_), 10), m)],
                              col + m] = c_
                else:
                    for m in range(n_m):
                        b_vec = Phi[:, col + m] * sw
                        sol, *_ = np.linalg.lstsq(A, b_vec, rcond=1e-10)
                        C[cc, col + m] = sol
                col += n_m
            return C

        c_hard = _coeffs(basis, True)
        c_soft = _coeffs(soft_basis, True)

        # LHP99 Eq. 52: share the off-centre coefficients so the
        # off-centre fits cancel exactly in the hard-soft difference.
        hard_shells = list(basis.shells())
        hard_off = []
        off = 0
        for sh in hard_shells:
            hard_off.append(off)
            off += 2 * int(sh.l) + 1
        hk = 0
        col_s = 0
        for sh in soft_basis.shells():
            n_m = 2 * int(sh.l) + 1
            if int(sh.atom_index) != ia:
                while True:
                    hsh = hard_shells[hk]
                    if (int(hsh.atom_index) == int(sh.atom_index)
                            and int(hsh.l) == int(sh.l)
                            and abs(float(hsh.exponents[0])
                                    - float(sh.exponents[0])) < 1e-9):
                        break
                    hk += 1
                for m in range(n_m):
                    c_soft[:, col_s + m] = c_hard[:, hard_off[hk] + m]
                hk += 1
            col_s += n_m

        chi = _ao_values(pb, pts)
        out.append(OneCentreExpansion(
            atom_index=ia,
            center=np.asarray(orig),
            radius=R_A,
            prim_basis=pb,
            c_hard=c_hard,
            c_soft=c_soft,
            chi=chi,
        ))
    return out


def projector_hartree_energy(
    expansions: list[OneCentreExpansion],
    atom_grids,
    D: np.ndarray,
    D_soft: np.ndarray,
    rho_tilde_grid: np.ndarray,
    grid,
    *,
    lmax: int = _DEFAULT_LMAX,
    comp_alpha: float = _DEFAULT_COMP_ALPHA,
) -> float:
    """The projector-augmentation GAPW Hartree energy at density D.

    ``rho_tilde_grid`` is the soft density already collocated on the
    FFT grid (the caller owns the collocation cache); ``D_soft`` is the
    soft-AO block of D. Returns E_sm[ntilde+n0] + the windowless
    radial hard-soft augmentation with per-lm compensation.
    """
    from .periodic_gapw_augment import solve_poisson_radial

    e_aug = 0.0
    comp_specs = []
    for ex in expansions:
        g = atom_grids[ex.atom_index]
        n_r, n_a = g.n_radial, g.n_angular
        Dp_h = ex.c_hard @ D @ ex.c_hard.T
        Dp_s = ex.c_soft @ D_soft @ ex.c_soft.T
        chi = ex.chi
        n1 = np.einsum("ga,ab,gb->g", chi, Dp_h, chi).reshape(n_r, n_a)
        nt1 = np.einsum("ga,ab,gb->g", chi, Dp_s, chi).reshape(n_r, n_a)

        pts = np.asarray(g.cartesian_points()).reshape(n_r, n_a, 3)
        wt = np.asarray(g.combined_weights())
        r = np.sqrt(((pts - ex.center) ** 2).sum(axis=-1))
        Slm = g.evaluate_real_spherical_harmonics(lmax)

        Q = {}
        n0 = np.zeros_like(n1)
        ic = 0
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                q = float(((n1 - nt1) * (r ** l)
                           * Slm[ic][None, :] * wt).sum())
                Q[(l, m)] = q
                if abs(q) > 1e-12:
                    shape = (r ** l) * np.exp(-comp_alpha * r ** 2)
                    base = shape * Slm[ic][None, :]
                    mom = float((base * (r ** l)
                                 * Slm[ic][None, :] * wt).sum())
                    n0 += (q / mom) * base
                ic += 1
        comp_specs.append((ex.center, Q))

        V_h = solve_poisson_radial(n1, g, lmax=lmax)
        V_s = solve_poisson_radial(nt1 + n0, g, lmax=lmax)
        e_aug += (0.5 * float((n1 * V_h * wt).sum())
                  - 0.5 * float(((nt1 + n0) * V_s * wt).sum()))

    # Smooth side: ntilde + the same per-lm compensators.
    r_xyz = grid.cartesian_coords()
    rho0 = np.zeros(grid.shape)
    Lm = grid.lattice_bohr
    for (center, Q) in comp_specs:
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                for iz in (-1, 0, 1):
                    shift = (ix * Lm[:, 0] + iy * Lm[:, 1]
                             + iz * Lm[:, 2])
                    delta = r_xyz - (center + shift)
                    r2g = (delta ** 2).sum(axis=-1)
                    rg = np.sqrt(r2g)
                    om = delta / np.where(rg > 1e-30, rg, 1.0)[..., None]
                    Sg = real_spherical_harmonics(om.reshape(-1, 3), lmax)
                    ic = 0
                    for l in range(lmax + 1):
                        for m in range(-l, l + 1):
                            q = Q[(l, m)]
                            if abs(q) > 1e-12:
                                mom = (0.5 * math.gamma(l + 1.5)
                                       / comp_alpha ** (l + 1.5))
                                rho0 += ((q / mom) * (rg ** l)
                                         * np.exp(-comp_alpha * r2g)
                                         * Sg[ic].reshape(grid.shape))
                            ic += 1
    dV = grid.voxel_volume_bohr3
    rho_sm = rho_tilde_grid + rho0
    V_sm = _core.solve_poisson_coulomb(rho_sm, grid.lattice_bohr)
    e_sm = 0.5 * float(np.sum(rho_sm * V_sm)) * dV
    return e_sm + e_aug


def projector_hartree_energy_and_fock(
    expansions: list[OneCentreExpansion],
    atom_grids,
    D: np.ndarray,
    soft_indices: np.ndarray,
    rho_tilde_grid: np.ndarray,
    soft_chi_grid: np.ndarray,
    grid,
    *,
    lmax: int = _DEFAULT_LMAX,
    comp_alpha: float = _DEFAULT_COMP_ALPHA,
) -> tuple[float, np.ndarray]:
    """The projector Hartree energy AND its exact Fock ``J = dE/dD``.

    Every density map is linear in D, so E is a quadratic form and the
    Fock assembles from the energy's own intermediates:

        J = embed_soft(int chitilde_mu chitilde_nu V_sm)
          + sum_A [ C_h^T W_h C_h - embed_soft(C_s^T W_s C_s)
                    + sum_lm c_lm (Mh_lm - embed_soft(Ms_lm)) ]

    with ``W = <g_a g_b V>`` on the radial grid, ``M_lm = C^T m_lm C``
    the per-lm moment maps (``m_lm = <g_a g_b r^l S_lm>``) and
    ``c_lm = <shape_lm, V_sm>_FFT - <shape_lm, V_s>_rad`` the
    compensator-response coefficients. By construction
    ``1/2 tr(D J) = E`` exactly (both are the same quadratic form).

    ``soft_chi_grid`` is the soft-basis AO table on the FFT grid
    (``(n_grid_pts, n_soft)``, e.g. from the GapwJBuilder soft
    collocation cache); ``rho_tilde_grid`` the collocated soft density.
    """
    from .periodic_gapw_augment import solve_poisson_radial

    n_basis = D.shape[0]
    idx = np.asarray(soft_indices, dtype=int)
    D_soft = D[np.ix_(idx, idx)]

    e_aug = 0.0
    comp_specs = []      # (center, {lm: Q})
    per_atom = []        # intermediates for the Fock pass
    for ex in expansions:
        g = atom_grids[ex.atom_index]
        n_r, n_a = g.n_radial, g.n_angular
        Dp_h = ex.c_hard @ D @ ex.c_hard.T
        Dp_s = ex.c_soft @ D_soft @ ex.c_soft.T
        chi = ex.chi
        n1 = np.einsum("ga,ab,gb->g", chi, Dp_h, chi).reshape(n_r, n_a)
        nt1 = np.einsum("ga,ab,gb->g", chi, Dp_s, chi).reshape(n_r, n_a)

        pts = np.asarray(g.cartesian_points()).reshape(n_r, n_a, 3)
        wt = np.asarray(g.combined_weights())
        r = np.sqrt(((pts - ex.center) ** 2).sum(axis=-1))
        Slm = g.evaluate_real_spherical_harmonics(lmax)

        Q = {}
        moms = {}
        n0 = np.zeros_like(n1)
        ic = 0
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                q = float(((n1 - nt1) * (r ** l)
                           * Slm[ic][None, :] * wt).sum())
                Q[(l, m)] = q
                shape = (r ** l) * np.exp(-comp_alpha * r ** 2)
                base = shape * Slm[ic][None, :]
                mom = float((base * (r ** l)
                             * Slm[ic][None, :] * wt).sum())
                moms[(l, m)] = mom
                if abs(q) > 1e-12:
                    n0 += (q / mom) * base
                ic += 1
        comp_specs.append((ex.center, Q))

        V_h = solve_poisson_radial(n1, g, lmax=lmax)
        V_s = solve_poisson_radial(nt1 + n0, g, lmax=lmax)
        e_aug += (0.5 * float((n1 * V_h * wt).sum())
                  - 0.5 * float(((nt1 + n0) * V_s * wt).sum()))
        per_atom.append((ex, g, r, Slm, wt, moms, V_h, V_s))

    # ---- smooth side: rho0 on the FFT grid (pass 1) ------------------
    r_xyz = grid.cartesian_coords()
    rho0 = np.zeros(grid.shape)
    Lm = grid.lattice_bohr
    shifts = [ix * Lm[:, 0] + iy * Lm[:, 1] + iz * Lm[:, 2]
              for ix in (-1, 0, 1) for iy in (-1, 0, 1)
              for iz in (-1, 0, 1)]
    for (center, Q) in comp_specs:
        for shift in shifts:
            delta = r_xyz - (center + shift)
            r2g = (delta ** 2).sum(axis=-1)
            rg = np.sqrt(r2g)
            om = delta / np.where(rg > 1e-30, rg, 1.0)[..., None]
            Sg = real_spherical_harmonics(om.reshape(-1, 3), lmax)
            ic = 0
            for l in range(lmax + 1):
                for m in range(-l, l + 1):
                    q = Q[(l, m)]
                    if abs(q) > 1e-12:
                        mom = (0.5 * math.gamma(l + 1.5)
                               / comp_alpha ** (l + 1.5))
                        rho0 += ((q / mom) * (rg ** l)
                                 * np.exp(-comp_alpha * r2g)
                                 * Sg[ic].reshape(grid.shape))
                    ic += 1
    dV = grid.voxel_volume_bohr3
    rho_sm = rho_tilde_grid + rho0
    V_sm = _core.solve_poisson_coulomb(rho_sm, grid.lattice_bohr)
    e_sm = 0.5 * float(np.sum(rho_sm * V_sm)) * dV
    energy = e_sm + e_aug

    # ---- Fock assembly ----------------------------------------------
    J = np.zeros((n_basis, n_basis))

    # (1) soft-density response: int chitilde_mu chitilde_nu V_sm.
    V_flat = V_sm.reshape(-1)
    J_soft = (soft_chi_grid.T @ (soft_chi_grid * V_flat[:, None])) * dV
    J[np.ix_(idx, idx)] += J_soft

    # (2+3) per-atom radial responses + compensator response.
    for (ex, g, r, Slm, wt, moms, V_h, V_s) in per_atom:
        chi = ex.chi
        w_h = (V_h * wt).reshape(-1)
        w_s = (V_s * wt).reshape(-1)
        W_h = chi.T @ (chi * w_h[:, None])
        W_s = chi.T @ (chi * w_s[:, None])
        J += ex.c_hard.T @ W_h @ ex.c_hard
        J[np.ix_(idx, idx)] -= ex.c_soft.T @ W_s @ ex.c_soft

        # c_lm = <shape_lm/mom, V_sm>_FFT - <shape_lm/mom, V_s>_rad,
        # FFT part summed over images (pass 2 over the same shapes).
        c = {}
        ic = 0
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                shape_rad = ((r ** l) * np.exp(-comp_alpha * r ** 2)
                             * Slm[ic][None, :])
                c_rad = float((shape_rad * V_s * wt).sum()) / moms[(l, m)]
                c[(l, m)] = -c_rad
                ic += 1
        mom_fft = {
            (l, m): 0.5 * math.gamma(l + 1.5) / comp_alpha ** (l + 1.5)
            for l in range(lmax + 1) for m in range(-l, l + 1)
        }
        for shift in shifts:
            delta = r_xyz - (ex.center + shift)
            r2g = (delta ** 2).sum(axis=-1)
            rg = np.sqrt(r2g)
            om = delta / np.where(rg > 1e-30, rg, 1.0)[..., None]
            Sg = real_spherical_harmonics(om.reshape(-1, 3), lmax)
            ic = 0
            for l in range(lmax + 1):
                for m in range(-l, l + 1):
                    shape = ((rg ** l) * np.exp(-comp_alpha * r2g)
                             * Sg[ic].reshape(grid.shape))
                    c[(l, m)] += (float(np.sum(shape * V_sm)) * dV
                                  / mom_fft[(l, m)])
                    ic += 1

        # m_lm moment maps and their AO-space images.
        ic = 0
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                c_lm = c[(l, m)]
                if abs(c_lm) < 1e-14:
                    ic += 1
                    continue
                w_lm = ((r ** l) * Slm[ic][None, :] * wt).reshape(-1)
                m_lm = chi.T @ (chi * w_lm[:, None])
                J += c_lm * (ex.c_hard.T @ m_lm @ ex.c_hard)
                J[np.ix_(idx, idx)] -= c_lm * (
                    ex.c_soft.T @ m_lm @ ex.c_soft)
                ic += 1

    return energy, J


# ====================================================================
# Fit-free analytic-ERI augmentation (the variationally safe mode)
# ====================================================================


@dataclass
class AnalyticAugmentation:
    """D-independent data for the fit-free analytic-ERI augmentation.

    The formulation (validated 2026-07-30, see
    HANDOVER_GAPW_PRODUCTION.md):

        E_H_per[D] = E_sm_per[rho_tilde + rho0]
                   + 1/2 D . ERI_hard . D
                   - 1/2 D_soft . ERI_soft . D_soft
                   - <rho_tilde, V0>_grid
                   - sum_A q_A^2 sqrt(a/2)/sqrt(pi)
                   -  cross-compensator terms (multi-pruned-atom cells)

    with s-only compensators per pruned atom A carrying the Mulliken
    pruned charge ``q_A(D) = tr(W_A D) - tr(Wt_A D_soft)`` (linear in
    D). Every term is an exact ERI contraction, an analytic Gaussian
    integral, or a bounded-kernel grid quadrature: no fitted one-centre
    expansions anywhere, so the SCF has no fit-error self-repulsion
    hole to exploit (the failure mode of the projector-fit mode).

    Envelope: the free-space ERI difference assumes the
    molecular-limit Gamma regime (isolated cluster in a box whose
    density images do not overlap) - exactly the PW-limit atomization
    target. Dense-crystal cells need image sums of the difference
    terms (future work).
    """

    pruned: list[int]
    centers: np.ndarray          # (n_pruned, 3)
    eri_hard: np.ndarray         # (n, n, n, n)
    eri_soft: np.ndarray         # (ns, ns, ns, ns)
    w_hard: list[np.ndarray]     # per pruned atom: dq_A/dD (n, n)
    w_soft: list[np.ndarray]     # per pruned atom: dq_A/dD_soft (ns, ns)
    v0_grid: list[np.ndarray]    # per pruned atom: erf(sqrt(a) r)/r on FFT
    g_grid: list[np.ndarray]     # per pruned atom: unit compensator on FFT
    self_coeff: float            # sqrt(a/2)/sqrt(pi)
    cross_coeff: np.ndarray      # (n_pruned, n_pruned) off-diag comp-comp
    comp_alpha: float


def build_analytic_augmentation(
    basis,
    soft_basis,
    system,
    grid,
    *,
    comp_alpha: float = _DEFAULT_COMP_ALPHA,
) -> AnalyticAugmentation:
    """Build the D-independent data for the analytic-ERI augmentation."""
    from scipy.special import erf as _erf

    pruned = pruned_atom_indices(basis, soft_basis)
    atoms = list(system.unit_cell)
    positions = np.asarray([list(a.xyz) for a in atoms], dtype=float)
    centers = positions[pruned] if pruned else np.zeros((0, 3))

    eri_hard = np.asarray(_core.compute_eri(basis))
    eri_soft = np.asarray(_core.compute_eri(soft_basis))
    S_h = np.asarray(_core.compute_overlap(basis))
    S_s = np.asarray(_core.compute_overlap(soft_basis))

    # Pruned-charge partition. CRITICAL: sum_A q_A(D) must equal the
    # exact total tr(S_hard D) - tr(S_soft D_soft) for EVERY density,
    # not just physical ones - a per-atom Mulliken split does not sum
    # to that total (the unpruned atoms' Mulliken populations also
    # shift between the bases), and the resulting net-charge mismatch
    # in sigma = rho_tilde + rho0 is a Madelung-scale energy
    # inconsistency the SCF mines (measured: a -3.0 Ha H2O hole with
    # q(D*) 1.8 e short). D-independent fractions f_A of the exact
    # global dq operator conserve the total identically; for a single
    # pruned atom f = 1 is exact, for several the uniform split only
    # mislocates part of the (small) pruned charge, which enters the
    # energy only through the bounded image-moment terms.
    n_pruned = max(len(pruned), 1)
    w_hard = [S_h / n_pruned for _ in pruned]
    w_soft = [S_s / n_pruned for _ in pruned]

    # FFT-grid fields per pruned atom: the free-space erf potential of
    # a unit s-compensator (central image: molecular-limit regime) and
    # the periodic-image-summed unit compensator density for E_sm.
    r_xyz = grid.cartesian_coords()
    Lm = grid.lattice_bohr
    shifts = [ix * Lm[:, 0] + iy * Lm[:, 1] + iz * Lm[:, 2]
              for ix in (-1, 0, 1) for iy in (-1, 0, 1)
              for iz in (-1, 0, 1)]
    v0_grid = []
    g_grid = []
    for R in centers:
        d = r_xyz - R
        r = np.sqrt((d ** 2).sum(axis=-1))
        r = np.where(r > 1e-12, r, 1e-12)
        v0_grid.append(_erf(math.sqrt(comp_alpha) * r) / r)
        gg = np.zeros(grid.shape)
        for shift in shifts:
            d2 = ((r_xyz - (R + shift)) ** 2).sum(axis=-1)
            gg += (comp_alpha / np.pi) ** 1.5 * np.exp(-comp_alpha * d2)
        g_grid.append(gg)

    # Free-space compensator-compensator interactions (two s-Gaussians):
    # (q_A q_B) erf(sqrt(a/2) R_AB)/R_AB for A != B; the self term is
    # q^2 sqrt(a/2)/sqrt(pi).
    n_p = len(pruned)
    cross = np.zeros((n_p, n_p))
    for i in range(n_p):
        for j in range(n_p):
            if i == j:
                continue
            R_ab = float(np.linalg.norm(centers[i] - centers[j]))
            cross[i, j] = float(_erf(math.sqrt(comp_alpha / 2.0) * R_ab)
                                / R_ab)
    return AnalyticAugmentation(
        pruned=pruned,
        centers=centers,
        eri_hard=eri_hard,
        eri_soft=eri_soft,
        w_hard=w_hard,
        w_soft=w_soft,
        v0_grid=v0_grid,
        g_grid=g_grid,
        self_coeff=math.sqrt(comp_alpha / 2.0) / math.sqrt(math.pi),
        cross_coeff=cross,
        comp_alpha=comp_alpha,
    )


def analytic_hartree_energy_and_fock(
    aug: AnalyticAugmentation,
    D: np.ndarray,
    soft_indices: np.ndarray,
    rho_tilde_grid: np.ndarray,
    soft_chi_grid: np.ndarray,
    grid,
) -> tuple[float, np.ndarray]:
    """Energy + exact Fock of the analytic-ERI augmentation.

    Every term is quadratic in D through exact linear maps, so
    ``1/2 tr(D J) = E`` holds identically.
    """
    n = D.shape[0]
    idx = np.asarray(soft_indices, dtype=int)
    D_soft = D[np.ix_(idx, idx)]
    dV = grid.voxel_volume_bohr3

    # Pruned charges (linear in D).
    q = np.array([
        float(np.einsum("ij,ij->", aug.w_hard[i], D))
        - float(np.einsum("ij,ij->", aug.w_soft[i], D_soft))
        for i in range(len(aug.pruned))
    ])

    # Smooth periodic term.
    rho0 = np.zeros(grid.shape)
    for i, qi in enumerate(q):
        rho0 += qi * aug.g_grid[i]
    rho_sm = rho_tilde_grid + rho0
    V_sm = _core.solve_poisson_coulomb(rho_sm, grid.lattice_bohr)
    e_sm = 0.5 * float(np.sum(rho_sm * V_sm)) * dV

    # Exact free-space pieces.
    J_h = np.einsum("ijkl,kl->ij", aug.eri_hard, D)
    J_s = np.einsum("ijkl,kl->ij", aug.eri_soft, D_soft)
    e_h = 0.5 * float(np.einsum("ij,ij->", D, J_h))
    e_s = 0.5 * float(np.einsum("ij,ij->", D_soft, J_s))

    # Soft-density x compensator cross term and compensator
    # self/cross energies (free space).
    cross_t = np.array([
        float(np.sum(rho_tilde_grid * aug.v0_grid[i])) * dV
        for i in range(len(aug.pruned))
    ])
    e_cross = float(q @ cross_t)
    e_self = aug.self_coeff * float(q @ q)
    e_cc = 0.5 * float(q @ aug.cross_coeff @ q)

    energy = e_sm + e_h - e_s - e_cross - e_self - e_cc

    # ---- exact Fock ------------------------------------------------
    J = np.zeros((n, n))
    V_flat = V_sm.reshape(-1)
    J[np.ix_(idx, idx)] += (
        soft_chi_grid.T @ (soft_chi_grid * V_flat[:, None])) * dV
    J += J_h
    J[np.ix_(idx, idx)] -= J_s

    # V0 projections onto the soft AO pairs (for the cross term).
    for i, qi in enumerate(q):
        v0_flat = aug.v0_grid[i].reshape(-1)
        P_v0 = (soft_chi_grid.T
                @ (soft_chi_grid * v0_flat[:, None])) * dV
        J[np.ix_(idx, idx)] -= qi * P_v0

    # q(D)-chain terms: dE/dq_i * dq_i/dD.
    for i in range(len(aug.pruned)):
        g_flat = aug.g_grid[i].reshape(-1)
        dE_dq = (float(np.dot(g_flat, V_flat)) * dV     # from e_sm
                 - cross_t[i]                            # from -e_cross
                 - 2.0 * aug.self_coeff * q[i]           # from -e_self
                 - float(aug.cross_coeff[i] @ q))        # from -e_cc
        dq_dD = aug.w_hard[i].copy()
        dq_dD[np.ix_(idx, idx)] -= aug.w_soft[i]
        J += dE_dq * dq_dD

    return energy, J


# ====================================================================
# Projector-density XC augmentation (for the analytic mode's DFT path)
# ====================================================================


def _expansion_chi_gradient(ex: OneCentreExpansion, atom_grids):
    """Lazily cache the primitive-basis gradient table on the radial
    grid: a (3, n_pts, n_prim) array."""
    cached = getattr(ex, "_chi_grad", None)
    if cached is not None:
        return cached
    g = atom_grids[ex.atom_index]
    pts = np.asarray(g.cartesian_points()).reshape(-1, 3)
    out = _core.evaluate_ao_with_gradient(ex.prim_basis, pts)
    vals = [np.asarray(o) for o in out]
    grad = []
    for k in (1, 2, 3):
        v = vals[k]
        if v.shape[0] == ex.prim_basis.nbasis:
            v = v.T
        grad.append(v)
    grad = np.stack(grad)  # (3, n_pts, n_prim)
    ex._chi_grad = grad
    return grad


def projector_xc_correction(
    expansions: list[OneCentreExpansion],
    atom_grids,
    D: np.ndarray,
    D_soft: np.ndarray,
    soft_indices: np.ndarray,
    functional,
) -> tuple[dict, float]:
    """Per-atom XC augmentation on PROJECTOR one-centre densities.

    Replaces the block-restricted XC telescoping for the analytic
    mode's DFT path: on H2O/LDA at the fixed molecular density the
    block error is +1567.5 mHa, the projector-density error -70.5
    (2026-07-31 oracle). Returns ``({atom_index: dV_xc}, e_xc_aug)``
    with each ``dV_xc`` a full ``(n, n)`` AO matrix (the hard-side
    C^T W C minus the embedded soft side), and the energy

        e_xc_aug = sum_A int (exc[n1_A] - exc[nt1_A]) w

    with exc the libxc ENERGY DENSITY (rho included - summing exc*w is
    correct; multiplying by rho again double-counts). LDA and GGA;
    meta-GGA is not supported here.

    Fitted densities can be slightly negative in far tails; the
    density (and sigma) fed to libxc is clipped at zero. The clip
    boundary is a measure-zero kink identical in energy and Fock, so
    SCF consistency (F = dE/dD) is preserved to grid accuracy.
    """
    n = D.shape[0]
    idx = np.asarray(soft_indices, dtype=int)
    is_gga = getattr(functional, "kind", None) is not None and \
        functional.kind != _core.XCKind.LDA
    if getattr(functional, "kind", None) == _core.XCKind.MGGA:
        raise NotImplementedError(
            "projector_xc_correction: meta-GGA tau augmentation is not "
            "ported; use LDA/GGA or the block route."
        )

    corrections = {}
    e_xc_aug = 0.0
    for ex in expansions:
        g = atom_grids[ex.atom_index]
        wt = np.asarray(g.combined_weights()).ravel()
        chi = ex.chi
        Dp_h = ex.c_hard @ D @ ex.c_hard.T
        Dp_s = ex.c_soft @ D_soft @ ex.c_soft.T

        def _one_side(Dp):
            rho = np.einsum("ga,ab,gb->g", chi, Dp, chi)
            rho_c = np.clip(rho, 0.0, None)
            if is_gga:
                cg = _expansion_chi_gradient(ex, atom_grids)
                grad = np.stack([
                    2.0 * np.einsum("ga,ab,gb->g", chi, Dp, cg[k])
                    for k in range(3)
                ])  # (3, n_pts)
                sigma = (grad ** 2).sum(axis=0)
                exc, v_rho, v_sigma = functional.eval_unpolarised(
                    rho_c, sigma)
                exc = np.asarray(exc)
                v_rho = np.asarray(v_rho)
                v_sigma = np.asarray(v_sigma)
            else:
                exc, v_rho, _vs = functional.eval_unpolarised(
                    rho_c, np.zeros_like(rho_c))
                exc = np.asarray(exc)
                v_rho = np.asarray(v_rho)
                v_sigma = None
                grad = None
            dead = rho_c <= 0.0
            exc = np.where(dead, 0.0, exc)
            v_rho = np.where(dead, 0.0, v_rho)
            e = float((exc * wt).sum())
            W = chi.T @ (chi * (v_rho * wt)[:, None])
            if is_gga:
                v_sigma = np.where(dead, 0.0, v_sigma)
                for k in range(3):
                    t = 2.0 * v_sigma * grad[k] * wt
                    Wk = chi.T @ (cg[k] * t[:, None])
                    W += Wk + Wk.T
            return e, W

        e_h, W_h = _one_side(Dp_h)
        e_s, W_s = _one_side(Dp_s)
        e_xc_aug += e_h - e_s
        dV = ex.c_hard.T @ W_h @ ex.c_hard
        dV_s = ex.c_soft.T @ W_s @ ex.c_soft
        full = dV.copy()
        full[np.ix_(idx, idx)] -= dV_s
        corrections[ex.atom_index] = full
    return corrections, e_xc_aug


def projector_xc_correction_polarised(
    expansions: list[OneCentreExpansion],
    atom_grids,
    D_alpha: np.ndarray,
    D_beta: np.ndarray,
    D_soft_alpha: np.ndarray,
    D_soft_beta: np.ndarray,
    soft_indices: np.ndarray,
    functional,
) -> tuple[dict, dict, float]:
    """Spin-polarised per-atom XC augmentation on projector densities.

    The open-shell counterpart of :func:`projector_xc_correction`,
    mirroring CP2K's ``calculate_vxc_atom`` (qs_vxc_atom.F): the hard
    and soft one-centre densities are carried per spin through the
    same projector coefficients, the polarised functional is evaluated
    on both sides, and

        e_xc_aug = sum_A int (exc[n1_A^a, n1_A^b]
                              - exc[nt1_A^a, nt1_A^b]) w

    with per-spin Fock corrections assembled from ``v_rho^s`` plus the
    GGA ``d sigma / dD^s`` chain (``2 v_saa grad_a + v_sab grad_b``
    for the alpha channel and the mirrored form for beta). Returns
    ``({atom: dV_a}, {atom: dV_b}, e_xc_aug)``. ``functional`` must be
    the polarised (spin=2) libxc wrapper. LDA and GGA; meta-GGA is not
    ported. The same zero-clip convention as the closed-shell path
    applies, keyed on the total density.
    """
    idx = np.asarray(soft_indices, dtype=int)
    is_gga = getattr(functional, "kind", None) is not None and \
        functional.kind != _core.XCKind.LDA
    if getattr(functional, "kind", None) == _core.XCKind.MGGA:
        raise NotImplementedError(
            "projector_xc_correction_polarised: meta-GGA tau "
            "augmentation is not ported; use LDA/GGA or the block "
            "route."
        )

    corr_a: dict = {}
    corr_b: dict = {}
    e_xc_aug = 0.0
    for ex in expansions:
        g = atom_grids[ex.atom_index]
        wt = np.asarray(g.combined_weights()).ravel()
        chi = ex.chi
        Dp_h_a = ex.c_hard @ D_alpha @ ex.c_hard.T
        Dp_h_b = ex.c_hard @ D_beta @ ex.c_hard.T
        Dp_s_a = ex.c_soft @ D_soft_alpha @ ex.c_soft.T
        Dp_s_b = ex.c_soft @ D_soft_beta @ ex.c_soft.T

        def _one_side(Dp_a, Dp_b):
            rho_a = np.einsum("ga,ab,gb->g", chi, Dp_a, chi)
            rho_b = np.einsum("ga,ab,gb->g", chi, Dp_b, chi)
            rho_a_c = np.clip(rho_a, 0.0, None)
            rho_b_c = np.clip(rho_b, 0.0, None)
            if is_gga:
                cg = _expansion_chi_gradient(ex, atom_grids)
                grad_a = np.stack([
                    2.0 * np.einsum("ga,ab,gb->g", chi, Dp_a, cg[k])
                    for k in range(3)
                ])
                grad_b = np.stack([
                    2.0 * np.einsum("ga,ab,gb->g", chi, Dp_b, cg[k])
                    for k in range(3)
                ])
                s_aa = (grad_a ** 2).sum(axis=0)
                s_ab = (grad_a * grad_b).sum(axis=0)
                s_bb = (grad_b ** 2).sum(axis=0)
                exc, v_ra, v_rb, v_saa, v_sab, v_sbb = (
                    functional.eval_polarised(
                        rho_a_c, rho_b_c, s_aa, s_ab, s_bb)
                )
                v_saa = np.asarray(v_saa)
                v_sab = np.asarray(v_sab)
                v_sbb = np.asarray(v_sbb)
            else:
                zeros = np.zeros_like(rho_a_c)
                exc, v_ra, v_rb, _vsaa, _vsab, _vsbb = (
                    functional.eval_polarised(
                        rho_a_c, rho_b_c, zeros, zeros, zeros)
                )
                v_saa = v_sab = v_sbb = None
                grad_a = grad_b = None
            exc = np.asarray(exc)
            v_ra = np.asarray(v_ra)
            v_rb = np.asarray(v_rb)
            dead = (rho_a_c + rho_b_c) <= 0.0
            exc = np.where(dead, 0.0, exc)
            v_ra = np.where(dead, 0.0, v_ra)
            v_rb = np.where(dead, 0.0, v_rb)
            e = float((exc * wt).sum())
            W_a = chi.T @ (chi * (v_ra * wt)[:, None])
            W_b = chi.T @ (chi * (v_rb * wt)[:, None])
            if is_gga:
                v_saa = np.where(dead, 0.0, v_saa)
                v_sab = np.where(dead, 0.0, v_sab)
                v_sbb = np.where(dead, 0.0, v_sbb)
                for k in range(3):
                    t_a = (2.0 * v_saa * grad_a[k]
                           + v_sab * grad_b[k]) * wt
                    t_b = (2.0 * v_sbb * grad_b[k]
                           + v_sab * grad_a[k]) * wt
                    Wk_a = chi.T @ (cg[k] * t_a[:, None])
                    Wk_b = chi.T @ (cg[k] * t_b[:, None])
                    W_a += Wk_a + Wk_a.T
                    W_b += Wk_b + Wk_b.T
            return e, W_a, W_b

        e_h, W_h_a, W_h_b = _one_side(Dp_h_a, Dp_h_b)
        e_s, W_s_a, W_s_b = _one_side(Dp_s_a, Dp_s_b)
        e_xc_aug += e_h - e_s
        for (W_h, W_s, out) in (
            (W_h_a, W_s_a, corr_a),
            (W_h_b, W_s_b, corr_b),
        ):
            full = ex.c_hard.T @ W_h @ ex.c_hard
            full[np.ix_(idx, idx)] -= ex.c_soft.T @ W_s @ ex.c_soft
            out[ex.atom_index] = full
    return corr_a, corr_b, e_xc_aug
