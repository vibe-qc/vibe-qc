"""Oracle v2 for the projector-based multipole-consistent GAPW
augmentation - the H2O Hartree identity CLOSES to +9 mHa (from
-4213 mHa in production).

Milestone step (a) COMPLETE (HANDOVER_GAPW_PRODUCTION.md "NEXT
MILESTONE"; v1 empirical map in oracle_projector_augmentation_wip.py).
Fixed converged molecular RHF density of H2O/STO-3G/12-bohr;
gauge-consistent periodic E_H target 35.7134 Ha (exact free-space
47.3181 minus the analytically verified image-gauge shift 11.6046).

THE VALIDATED CONSTRUCTION (all four ingredients matter):

1. Augmentation spheres ONLY on atoms whose basis was actually pruned
   (O). Unpruned atoms (H in mixed systems, per the hydrogen
   preservation rule) get NO sphere: their hard and soft one-centre
   content is identical and their augmentation is exactly zero.
   Production today puts spheres on every atom.
2. The sphere radius follows the PRUNED-CONTENT support (~1.7 bohr
   for O/STO-3G: slowest pruned-x-kept product exp(-5.4 r^2)), not a
   universal 3.5-bohr window, and the region-weighted LSQ fit of
   OFF-centre AOs runs over exactly that sphere. Letting one atom's
   fit region reach another PRUNED atom's core breaks the identity at
   the Ha level (the flat-2.2-bohr experiment).
3. Off-centre expansion coefficients are SHARED between the hard and
   soft one-centre densities (LHP99 Eq. 52): the off-centre fits then
   cancel exactly in delta_A = n1 - ntilde1 - n0, so delta_A is
   supported only where the pruned on-centre content lives, and the
   fit quality far from the centre barely matters.
4. Full-lm multipole compensation (LMAX=6 here; LMAX=4 is
   indistinguishable) with the same per-lm moments on the radial and
   FFT sides, and NO radial window: the integrals run over the whole
   grid, which the confined expansion makes safe.

Results at the fixed density (err = E_H vs the 35.7134 target):

  production block restriction  . . . . . . . . . -4213 mHa
  non-overlap spheres on ALL atoms (v2 first cut)   -221 mHa
  this construction, R_O=1.7, conf_scale=3  . . . .  +8.6 mHa
  robustness: R_O in [1.4, 2.0] x scale in [1, 3]:  +9..+44 mHa
  smooth-density charge closure . . . . . . . . . 10.005-10.009 / 10

Single-atom validation (same construction, which reduces for a lone
atom to exact on-centre + full-lm compensation + windowless
integrals; targets from the verified gauge relation
E_free - (dVne + E_nn) at the same box):

  Ne: +23.3 mHa (production windowed form: +53)
  O:  +16.3 mHa (production's -1.7 was window-luck cancellation)
  He:  +0.4 mHa
  charge closure exact (10.00000 / 8.00000 / 2.00000)

The remaining +16..+23 mHa on second-row atoms is radial-quadrature
and compensator-representation convergence, not formulation - the
same knobs the production port can converge.

Next steps: (b) port into the GapwAugmentation energy path (spheres
only on pruned atoms, shared off-centre fits, per-lm compensation, no
window) and re-validate the single atoms (He/O/Ne - no neighbours, so
the construction reduces to the de-windowed full-lm form) plus
LiH/H2O/H2 molecules; (c) the analytic Fock d(E_H)/dD of this energy
(per-lm M tensors + the off-centre-fit chain terms) keeping
1/2 tr(D J) = E_H; (d) delete the H2O strict xfail and tighten the
parity gates on honest oracles.
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
from vibeqc.periodic_gapw_atomic_grid import real_spherical_harmonics

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
E_H_free = 0.5 * float(np.einsum("ij,ijkl,kl->", D, eri, D))
TARGET = 35.713  # gauge-consistent periodic E_H (47.318 - 11.605)

soft = softened_basis(basis, system)

grid = PlaneWaveGrid(np.eye(3) * L, 64, 64, 64)
jb = GapwJBuilder(basis, system, grid, quiet=True)
aug = jb._aug
idx = aug._soft_indices
D_soft_block = D[np.ix_(idx, idx)]

LMAX = 6
L_EXP = 3
N_CONF = 6
R_REACH = 2.2  # fit-region outer radius (Becke weight handles neighbours)


def becke_weight(P, ia):
    """Becke cell weight of atom ia at points P (3rd-order switching)."""
    pos = np.asarray(POS)
    nA = len(POS)
    rA = np.stack([np.linalg.norm(P - pos[a], axis=1) for a in range(nA)])
    Pcell = np.ones((nA, P.shape[0]))
    def f(x):
        return 1.5 * x - 0.5 * x ** 3
    for a in range(nA):
        for b in range(nA):
            if a == b:
                continue
            dab = np.linalg.norm(pos[a] - pos[b])
            mu = (rA[a] - rA[b]) / dab
            sm = 0.5 * (1.0 - f(f(f(mu))))
            Pcell[a] *= sm
    tot = Pcell.sum(axis=0)
    tot = np.where(tot > 1e-30, tot, 1.0)
    return Pcell[ia] / tot

# Non-overlapping per-atom fit radii: half the nearest-neighbour
# distance (the H fit region must NOT reach into O's core - with a
# 2.2-bohr flat radius the H fit tried to reproduce O's core AOs and
# the identity broke at the -2 Ha level).
import itertools
_pos = np.asarray(POS)
import os
R_O = float(os.environ.get("R_O", "1.7"))
R_FIT_A = [R_O, None, None]
print("fit radii:", R_FIT_A)


def make_prim_basis(prims, mol_):
    shells = []
    for l, e, orig in prims:
        sh = core.ShellInfo()
        sh.atom_index = 0
        sh.l = l
        sh.pure = True
        sh.exponents = [e]
        sh.coefficients = [1.0]
        sh.origin = orig
        shells.append(sh)
    return core.BasisSet(mol_, shells, "prim",
                         coefficients_pre_normalized=True)


def ao_values(b, pts_flat):
    v = np.asarray(core.evaluate_ao(b, pts_flat))
    if v.shape[0] == b.nbasis:
        v = v.T
    return v  # (n_pts, nbasis)


def build_one_centre(conf_scale, ia, ad, target_b, prim_b):
    """Return (pb, C) for atom ia: exact on-centre + LSQ off-centre.

    The primitive (expansion) set always comes from ``prim_b`` (the
    HARD basis) so hard and soft one-centre densities share one radial
    basis; ``target_b`` supplies the AOs being expanded."""
    orig = list(POS[ia])
    own = []
    seen = set()
    for sh in prim_b.shells():
        if int(sh.atom_index) != ia:
            continue
        for e in sh.exponents:
            key = (int(sh.l), round(float(e), 10))
            if key not in seen:
                seen.add(key)
                own.append((int(sh.l), float(e), orig))
    # confined set adapted to this atom's fit radius:
    # alpha_min from exp(-a R^2) = 1e-2 containment, geometric x2.
    R_A = R_FIT_A[ia]
    a_min = math.log(100.0) / (R_A ** 2) * conf_scale
    conf_exps = [a_min * (2.0 ** i) for i in range(N_CONF)]
    conf = []
    for e in conf_exps:
        for l in range(L_EXP + 1):
            if (l, round(e, 10)) not in seen:
                conf.append((l, float(e), orig))
    prims = own + conf
    pb = make_prim_basis(prims, mol)
    n_g = pb.nbasis

    prim_pos = {}
    conf_cols = []
    k = 0
    for j, (l, e, org) in enumerate(prims):
        for m in range(2 * l + 1):
            prim_pos[(l, round(e, 10), m)] = k
            if j >= len(own):
                conf_cols.append(k)
            k += 1
    cc = np.asarray(conf_cols, dtype=int)

    g = ad.grid
    pts = np.asarray(g.cartesian_points()).reshape(-1, 3)
    wt = g.combined_weights().ravel()
    r = np.linalg.norm(pts - np.asarray(orig), axis=1)
    mask = r <= R_FIT_A[ia]
    P = pts[mask]
    sw = np.sqrt(wt[mask])

    G = ao_values(pb, P)
    Phi = ao_values(target_b, P)
    A = G[:, cc] * sw[:, None]

    C = np.zeros((n_g, target_b.nbasis))
    col = 0
    for sh in target_b.shells():
        l = int(sh.l)
        n_m = 2 * l + 1
        if int(sh.atom_index) == ia:
            for m in range(n_m):
                for e_, c_ in zip(sh.exponents, sh.coefficients):
                    C[prim_pos[(l, round(float(e_), 10), m)], col + m] = c_
        else:
            for m in range(n_m):
                b_vec = Phi[:, col + m] * sw
                sol, *_ = np.linalg.lstsq(A, b_vec, rcond=1e-10)
                C[cc, col + m] = sol
        col += n_m
    return pb, C


for conf_scale in (1.0, 2.0, 3.0):
    e_aug_total = 0.0
    comp_specs = []
    q_smooth_repr = 0.0
    ok = True
    for ia, ad in enumerate(aug._atom_data):
        if R_FIT_A[ia] is None:
            continue  # unpruned atom: no augmentation sphere
        pb, C_h = build_one_centre(conf_scale, ia, ad, basis, basis)
        _, C_s = build_one_centre(conf_scale, ia, ad, soft, basis)
        # LHP99 Eq. 52: off-centre coefficients are SHARED between the
        # hard and soft expansions (the functions coincide outside
        # their home atom), so the off-centre fits cancel exactly in
        # delta_A. Map soft AO columns -> hard AO columns.
        col_s = 0
        soft_shells = list(soft.shells())
        hard_shells = list(basis.shells())
        # build hard column offsets
        hard_off = []
        off = 0
        for sh in hard_shells:
            hard_off.append(off)
            off += 2 * int(sh.l) + 1
        # match soft shells to hard shells by (atom, l, origin, exps subset)
        hk = 0
        for sh in soft_shells:
            n_m = 2 * int(sh.l) + 1
            if int(sh.atom_index) != ia:
                # find the corresponding hard shell (same atom, same l,
                # same first exponent among that atom's shells)
                while True:
                    hsh = hard_shells[hk]
                    if (int(hsh.atom_index) == int(sh.atom_index)
                            and int(hsh.l) == int(sh.l)
                            and abs(float(hsh.exponents[0])
                                    - float(sh.exponents[0])) < 1e-9):
                        break
                    hk += 1
                for m in range(n_m):
                    C_s[:, col_s + m] = C_h[:, hard_off[hk] + m]
                hk += 1
            col_s += n_m

        Dp_h = C_h @ D @ C_h.T
        Dp_s = C_s @ D_soft_block @ C_s.T

        g = ad.grid
        pts = np.asarray(g.cartesian_points()).reshape(
            g.n_radial, g.n_angular, 3)
        wt = g.combined_weights()
        chi = ao_values(pb, pts.reshape(-1, 3))
        n1 = np.einsum("ga,ab,gb->g", chi, Dp_h, chi).reshape(
            g.n_radial, g.n_angular)
        nt1 = np.einsum("ga,ab,gb->g", chi, Dp_s, chi).reshape(
            g.n_radial, g.n_angular)

        q1 = float((n1 * wt).sum())
        qt1 = float((nt1 * wt).sum())
        if not (0.0 < q1 < 12.0 and 0.0 < qt1 < 12.0):
            ok = False

        center = np.asarray(POS[ia])
        rvec = pts - center
        r = np.sqrt((rvec ** 2).sum(axis=-1))
        Slm = g.evaluate_real_spherical_harmonics(LMAX)
        Q = {}
        ic = 0
        for l in range(LMAX + 1):
            for m in range(-l, l + 1):
                Q[(l, m)] = float(((n1 - nt1) * (r ** l)
                                   * Slm[ic][None, :] * wt).sum())
                ic += 1
        comp_specs.append((center, Q))
        q_smooth_repr += Q[(0, 0)] * math.sqrt(4 * math.pi) if False else 0.0

        alpha_c = 2.25
        n0 = np.zeros_like(n1)
        ic = 0
        for l in range(LMAX + 1):
            for m in range(-l, l + 1):
                q = Q[(l, m)]
                if abs(q) < 1e-12:
                    ic += 1
                    continue
                shape = (r ** l) * np.exp(-alpha_c * r ** 2)
                base = shape * Slm[ic][None, :]
                mom = float((base * (r ** l) * Slm[ic][None, :] * wt).sum())
                n0 += (q / mom) * base
                ic += 1
        V_h = solve_poisson_radial(n1, g, lmax=LMAX)
        V_s = solve_poisson_radial(nt1 + n0, g, lmax=LMAX)
        e_h = 0.5 * float((n1 * V_h * wt).sum())
        e_s = 0.5 * float(((nt1 + n0) * V_s * wt).sum())
        e_aug_total += e_h - e_s
        print(f"  atom {ia}: q1={q1:.4f} qt1={qt1:.4f} "
              f"dq={q1 - qt1:+.4f} aug={e_h - e_s:+.4f}")

    # smooth side
    rho_t_g = collocate_density_on_grid(soft, D_soft_block, grid,
                                        cache=jb._soft_collocation_cache)
    r_xyz = grid.cartesian_coords()
    rho0_g = np.zeros(grid.shape)
    alpha_c = 2.25
    Lm = grid.lattice_bohr
    for (center, Q) in comp_specs:
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                for iz in (-1, 0, 1):
                    shift = ix * Lm[:, 0] + iy * Lm[:, 1] + iz * Lm[:, 2]
                    delta = r_xyz - (center + shift)
                    r2g = (delta ** 2).sum(axis=-1)
                    rg = np.sqrt(r2g)
                    rs = np.where(rg > 1e-30, rg, 1.0)
                    om = delta / rs[..., None]
                    Sg = real_spherical_harmonics(om.reshape(-1, 3), LMAX)
                    ic = 0
                    for l in range(LMAX + 1):
                        for m in range(-l, l + 1):
                            q = Q[(l, m)]
                            if abs(q) < 1e-12:
                                ic += 1
                                continue
                            mom = 0.5 * math.gamma(l + 1.5) / alpha_c ** (l + 1.5)
                            rho0_g += (q / mom) * (rg ** l) * np.exp(
                                -alpha_c * r2g) * Sg[ic].reshape(grid.shape)
                            ic += 1
    dV = grid.voxel_volume_bohr3
    q_sm = float((rho_t_g + rho0_g).sum()) * dV
    V_sm = core.solve_poisson_coulomb(rho_t_g + rho0_g, grid.lattice_bohr)
    e_smooth = 0.5 * float(np.sum((rho_t_g + rho0_g) * V_sm)) * dV
    E_new = e_smooth + e_aug_total
    print(f"conf_scale={conf_scale}: q_smooth={q_sm:.4f}  e_sm={e_smooth:.4f}  "
          f"aug={e_aug_total:+.4f}  E_H={E_new:.4f}  "
          f"err={1000 * (E_new - TARGET):+.1f} mHa")
