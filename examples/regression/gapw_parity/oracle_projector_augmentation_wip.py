"""WORK-IN-PROGRESS oracle for the projector-based multipole-consistent
GAPW augmentation (the fix for the H2O -8.9 Ha public-route bug; see
HANDOVER_OPEN_BUGS_V015.md 2026-07-29 and HANDOVER_GAPW_PRODUCTION.md
"NEXT MILESTONE").

Follows Krack & Parrinello, PCCP 2, 2105 (2000), doi:10.1039/b001167n,
Eqs. 21-25 (projector one-centre expansion) and Eqs. 11-13 (multipole
screening), in the single-compensator (n0 == ntilde0,
FFT-representable) rearrangement that vibe-qc's GapwJBuilder uses:

    E_H[n] ~= E_sm[ntilde + n0]                        (FFT, zero-mean)
            + sum_A { E_rad[n1_A] - E_rad[ntilde1_A + n0_A] }   (radial)

Everything at the FIXED converged molecular RHF density of
H2O/STO-3G/12-bohr, so this is a pure functional-evaluation test. The
gauge-consistent target for E_H is ~35.713 Ha (exact free-space 47.318
minus the analytically verified image-gauge shift 11.605).

STATUS 2026-07-29 - identity NOT yet closed; empirical map of the
constructions tried (error of E_H_new vs the target):

  production block-restriction (on-centre AOs only) . . . -4213 mHa
  naive projection: expansion==projectors==atom's own
    exponents x all l (incl. diffuse) . . . . . . . . . . -1126 mHa
      (sane densities, but 0.417 e of representation charge
       leaks: smooth+comp integrates 9.583, not 10)
  KP2000 confined projectors + diffuse expansion  . . . . +3517 mHa
      (H one-centre charge explodes to 12.8: diffuse
       expansion functions fitted locally extrapolate
       wildly outside U_A, and the window-free radial
       integrals see all of it)
  split: exact on-centre + confined-only off-centre fit . -2020 mHa
      (all densities sane again; the off-centre tail fit
       in a 4-exponent confined set is simply not accurate
       enough at STO-3G)

Conclusions for the next iteration: (1) the off-centre fit quality
inside U_A is THE controlling error; fit it as a real least-squares
over the sphere region (radial-grid LSQ with a weight confined to
U_A), not through overlap tricks, and study convergence in the
confined-set size; (2) KP2000 validated at >=5 primitives per H -
STO-3G may be intrinsically marginal, so also check 6-31G* H2O where
the paper achieved -0.22 mHa; (3) charge closure of
(ntilde + n0 + projection error) must be monitored explicitly - a
0.4 e leak costs ~1 Ha at this box size.
"""
import math
import warnings

import numpy as np

warnings.simplefilter("ignore")
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_augment import (
    GapwJBuilder,
    collocate_density_on_grid,
    solve_poisson_radial,
    softened_basis,
)

L = 12.0
c = L / 2
POS = [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]]
ZS = [8, 1, 1]

system = core.PeriodicSystem()
system.dim = 3
system.lattice = np.eye(3) * L
system.unit_cell = [core.Atom(z, p) for z, p in zip(ZS, POS)]
mol = vq.Molecule([vq.Atom(z, p) for z, p in zip(ZS, POS)], 0, 1)
basis = vq.BasisSet(mol, "sto-3g")
o = vq.RHFOptions()
o.conv_tol_energy = 1e-10
D = np.asarray(vq.run_rhf(mol, basis, o).density)

eri = np.asarray(core.compute_eri(basis))
E_H_exact_free = 0.5 * float(np.einsum("ij,ijkl,kl->", D, eri, D))
print("exact free-space E_H[n] =", E_H_exact_free)

soft = softened_basis(basis, system)
S_full = np.asarray(core.compute_overlap(basis))
n_elec = float(np.einsum("ij,ij->", D, S_full))
print("tr(DS) =", n_elec)

# ---------------------------------------------------------------
# Projector one-centre expansion per atom (KP2000 Eqs. 21-25).
# One-centre basis on atom A: A's own primitives, uncontracted, per
# l channel (hard set). Projectors: identical exponents (alpha >=
# alpha_iso are their own isolated projectors; here we use the full
# overlap inversion since STO-3G exponent spreads are modest per l
# after the isolated-projector split below).
# ---------------------------------------------------------------
ALPHA_ISO = 20.0  # exponents above this act as isolated projectors

L_EXP = 2  # expansion-basis lmax (KP2000: same exponents for ALL l)

CONF_EXPS = [2.0, 4.0, 8.0, 16.0]  # confined off-centre expansion set

def atom_prim_shells(b, atom_index):
    """One-centre expansion primitives for atom `atom_index`.

    Split construction: the atom's OWN primitives per their own l (so
    on-centre AOs are represented exactly, however diffuse), plus a
    CONFINED all-l progression for the off-centre tails - confined so
    that the window-free radial integrals see decaying extrapolation
    outside U_A instead of a diffuse-function blow-up.
    """
    own = set()
    orig = None
    for sh in b.shells():
        if int(sh.atom_index) != atom_index:
            continue
        orig = list(sh.origin)
        for e in sh.exponents:
            own.add((int(sh.l), round(float(e), 10)))
    out = [(l, e, orig) for (l, e) in sorted(own)]
    for e in CONF_EXPS:
        for l in range(L_EXP + 1):
            if (l, round(e, 10)) not in own:
                out.append((l, e, orig))
    n_own = len([1 for _ in sorted(own)])
    return out, n_own

def make_prim_basis(prims, mol_):
    shells = []
    for l, e, orig in prims:
        sh = core.ShellInfo()
        sh.atom_index = 0  # unused for evaluation via merged basis
        sh.l = l
        sh.pure = True
        sh.exponents = [e]
        sh.coefficients = [1.0]
        sh.origin = orig
        shells.append(sh)
    return core.BasisSet(mol_, shells, "prim", coefficients_pre_normalized=True)

R_CONF = 1.51  # bohr (0.8 Angstrom, KP2000 Sec. 3)

def projector_coeffs(prims, target_basis, mol_, atom_index, center):
    """C'[a, mu]: on-centre AOs exactly on the atom's own primitives
    (LHP99 Eq. 36); off-centre AOs least-squares-fit INSIDE U_A on the
    confined expansion subset only, so the extrapolation outside the
    sphere decays. The fit metric is the overlap of confinement-scale
    projectors (same confined set), i.e. a local metric."""
    pb = make_prim_basis(prims, mol_)
    n_g = pb.nbasis

    # AO-index bookkeeping for the primitive list.
    prim_pos = {}
    conf_cols = []  # primitive indices belonging to the confined set
    k = 0
    for (l, e, orig) in prims:
        for m in range(2 * l + 1):
            prim_pos[(l, round(e, 10), m)] = k
            if any(abs(e - ce) < 1e-9 for ce in CONF_EXPS):
                conf_cols.append(k)
            k += 1

    merged_shells = list(pb.shells()) + list(target_basis.shells())
    mb = core.BasisSet(mol_, merged_shells, "merged",
                       coefficients_pre_normalized=True)
    S = np.asarray(core.compute_overlap(mb))
    S_gg = S[:n_g, :n_g]
    S_gphi = S[:n_g, n_g:]

    C = np.zeros((n_g, target_basis.nbasis))
    cc = np.asarray(conf_cols, dtype=int)
    S_cc = S_gg[np.ix_(cc, cc)]
    # least-squares in the confined metric: minimise ||chi - phi||
    # measured against the confined functions (their overlap row).
    col = 0
    for sh in target_basis.shells():
        l = int(sh.l)
        n_m = 2 * l + 1
        if int(sh.atom_index) == atom_index:
            for m in range(n_m):
                for e_, c_ in zip(sh.exponents, sh.coefficients):
                    C[prim_pos[(l, round(float(e_), 10), m)], col + m] = c_
        else:
            for m in range(n_m):
                rhs = S_gphi[cc, col + m]
                C[cc, col + m] = np.linalg.solve(S_cc, rhs)
        col += n_m
    return C, pb

def radial_density_from_prims(pb, Dp, grid_pts):
    """density(r) = sum_ab Dp_ab g_a(r) g_b(r) at grid points."""
    chi = np.asarray(core.evaluate_ao(pb, grid_pts.reshape(-1, 3)))
    if chi.shape[0] == pb.nbasis:
        chi = chi.T
    return np.einsum("ga,ab,gb->g", chi, Dp, chi)

# use GapwJBuilder for its radial grids + compensator alpha convention
grid = PlaneWaveGrid(np.eye(3) * L, 64, 64, 64)
jb = GapwJBuilder(basis, system, grid, quiet=True)
aug = jb._aug
idx = aug._soft_indices
D_soft_block = D[np.ix_(idx, idx)]

LMAX = 4  # multipole order for compensation + radial Poisson

e_aug_total = 0.0
comp_specs = []  # (position, alpha_exp, Q_lm dict) for the FFT side
for ia, ad in enumerate(aug._atom_data):
    prims_hard, _ = atom_prim_shells(basis, ia)
    prims_soft, _ = atom_prim_shells(soft, ia)
    C_hard, pb_hard = projector_coeffs(prims_hard, basis, mol, ia, POS[ia])
    # soft one-centre: same projector expansion but of the SOFT basis
    # (off-centre columns identical per LHP99 Eq. 52; on-centre columns
    # use pruned contractions). Expand soft AOs in the HARD primitive set
    # so hard/soft live on one radial basis.
    C_soft, _ = projector_coeffs(prims_hard, soft, mol, ia, POS[ia])

    Dp_hard = C_hard @ D @ C_hard.T
    Dp_soft = C_soft @ D_soft_block @ C_soft.T

    g = ad.grid
    pts = np.asarray(g.cartesian_points()).reshape(g.n_radial, g.n_angular, 3)
    wt = g.combined_weights()
    n1 = radial_density_from_prims(pb_hard, Dp_hard, pts).reshape(
        g.n_radial, g.n_angular)
    nt1 = radial_density_from_prims(pb_hard, Dp_soft, pts).reshape(
        g.n_radial, g.n_angular)

    q1 = float((n1 * wt).sum())
    qt1 = float((nt1 * wt).sum())
    print(f"atom {ia} (Z={ZS[ia]}): q(n1)={q1:.6f}  q(nt1)={qt1:.6f}  "
          f"dq={q1-qt1:+.6f}")

    # multipole moments of (n1 - nt1) up to LMAX on the radial grid
    Slm = g.evaluate_real_spherical_harmonics(LMAX)  # (ncomp, n_angular)
    center = np.array(POS[ia])
    rvec = pts - center
    r = np.sqrt((rvec ** 2).sum(axis=-1))
    Q = {}
    ic = 0
    for l in range(LMAX + 1):
        for m in range(-l, l + 1):
            mom = float(((n1 - nt1) * (r ** l) * Slm[ic][None, :] * wt).sum())
            Q[(l, m)] = mom
            ic += 1
    print(f"  |Q| l=0: {abs(Q[(0,0)]):.4f}  l=1: "
          f"{max(abs(Q[(1,m)]) for m in (-1,0,1)):.4f}  l=2: "
          f"{max(abs(Q[(2,m)]) for m in range(-2,3)):.4f}")
    comp_specs.append((center, Q))

    # compensator on the radial grid: per-lm Gaussian with alpha_c
    alpha_c = 2.25  # exponent (the production compensator alpha=1.5 width)
    n0 = np.zeros_like(n1)
    ic = 0
    for l in range(LMAX + 1):
        # N_l: normalisation so that the lm moment of
        # N r^l S_lm exp(-a r^2) equals 1
        # int r^l * N r^l exp(-a r^2) r^2 dr * int S_lm S_lm dOmega
        #   = N * Gamma(l + 3/2 + ...)  -- do it numerically on the grid.
        for m in range(-l, l + 1):
            if abs(Q[(l, m)]) < 1e-12:
                ic += 1
                continue
            shape = (r ** l) * np.exp(-alpha_c * r ** 2)
            base = shape * Slm[ic][None, :]
            mom = float((base * (r ** l) * Slm[ic][None, :] * wt).sum())
            n0 += (Q[(l, m)] / mom) * base
            ic += 1
    # windowless radial energies
    V_h = solve_poisson_radial(n1, g, lmax=LMAX)
    V_s = solve_poisson_radial(nt1 + n0, g, lmax=LMAX)
    e_h = 0.5 * float((n1 * V_h * wt).sum())
    e_s = 0.5 * float(((nt1 + n0) * V_s * wt).sum())
    e_aug_total += e_h - e_s
    print(f"  e_hard={e_h:.6f}  e_soft={e_s:.6f}  aug={e_h-e_s:+.6f}")

print("aug total =", e_aug_total)

# ---------------------------------------------------------------
# Smooth FFT term with the SAME per-lm compensators.
# ---------------------------------------------------------------
from vibeqc.periodic_gapw_atomic_grid import real_spherical_harmonics

rho_t_g = collocate_density_on_grid(soft, D_soft_block, grid,
                                    cache=jb._soft_collocation_cache)
r_xyz = grid.cartesian_coords()
rho0_g = np.zeros(grid.shape)
alpha_c = 2.25
Lmat = grid.lattice_bohr
for (center, Q) in comp_specs:
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                shift = ix * Lmat[:, 0] + iy * Lmat[:, 1] + iz * Lmat[:, 2]
                delta = r_xyz - (np.asarray(center) + shift)
                r2g = (delta ** 2).sum(axis=-1)
                rg = np.sqrt(r2g)
                rg_safe = np.where(rg > 1e-30, rg, 1.0)
                omega = delta / rg_safe[..., None]
                Sg = real_spherical_harmonics(omega.reshape(-1, 3), LMAX)
                ic = 0
                for l in range(LMAX + 1):
                    for m in range(-l, l + 1):
                        q = Q[(l, m)]
                        if abs(q) < 1e-12:
                            ic += 1
                            continue
                        # same numeric normalisation as the radial side:
                        # moment of (r^l exp(-a r^2) S_lm) over all space
                        # = int r^{2l+2} exp(-a r^2) dr = Gamma(l+3/2)/(2 a^{l+3/2})
                        mom = 0.5 * math.gamma(l + 1.5) / alpha_c ** (l + 1.5)
                        shape = (rg ** l) * np.exp(-alpha_c * r2g)
                        rho0_g += (q / mom) * shape * Sg[ic].reshape(grid.shape)
                        ic += 1

dV = grid.voxel_volume_bohr3
print("charge(rho_tilde FFT) =", float(rho_t_g.sum()) * dV)
print("charge(rho_0 FFT)     =", float(rho0_g.sum()) * dV)
rho_sm = rho_t_g + rho0_g
V_sm = core.solve_poisson_coulomb(rho_sm, grid.lattice_bohr)
e_smooth = 0.5 * float(np.sum(rho_sm * V_sm)) * dV
print("e_smooth =", e_smooth)

E_H_new = e_smooth + e_aug_total
print("E_H_new (projector + full-lm) =", E_H_new)
print("gauge-consistent target ~ 35.713 (exact_free 47.318 - 11.605)")
print("old block-restricted GAPW E_H = ~31.50 (-4.21 Ha error)")
print("new error vs target =", 1000 * (E_H_new - 35.713), "mHa (approx)")
