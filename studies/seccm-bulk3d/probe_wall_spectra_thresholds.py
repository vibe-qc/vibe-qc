"""Solver-wall mechanism probe, part 2.

(1) Near-zero cyclic-overlap spectrum for the wall cells (Cu 2x2x4,
    Cu 4x4x4) vs the healthy cells (Cu 2x2x2).
(2) Structure of the violent negative Jacobian mode on 2x2x4: per-atom
    pattern and its weight on the small-eigenvalue overlap subspace.
(3) Screening-threshold sweep on the 2x2x4 bare map: does dropping the
    near-null WS-fold modes (which the production 1e-10 threshold keeps
    and amplifies by 1/sqrt(eig)) remove the wall?
"""

from __future__ import annotations

import sys

import math

import numpy as np

from probe_wall_map_mechanism import (
    BareMap,
    build,
    fermi_occ,
    gam3_free_params,
    gamma_matrix,
    one_shot,
)
from vibeqc._vibeqc_core.semiempirical import (
    SemiempiricalBasis,
    gfn2_enumerate_shells,
)


def setup(replicas, a):
    mol, topo = build(a, replicas)
    params = gam3_free_params()
    native, rec = one_shot(mol, topo, params)
    H0 = np.asarray(native.hamiltonian, dtype=float)
    S = np.asarray(native.overlap, dtype=float)
    f0 = np.asarray(native.shell_charges, dtype=float)
    n_occ = int(native.n_occ)
    n_basis = int(native.n_basis)
    basis = SemiempiricalBasis.build(mol, params, 0)
    shells = list(gfn2_enumerate_shells(basis, mol, params))
    ao_shell = np.zeros(n_basis, dtype=int)
    for si, sh in enumerate(shells):
        for mu in range(int(sh.bf_start), int(sh.bf_start) + int(sh.n_funcs)):
            ao_shell[mu] = si
    T = np.asarray(topo.translations)
    volume = abs(np.linalg.det(T))
    alpha = math.sqrt(math.pi) / volume ** (1.0 / 3.0)
    gamma = gamma_matrix(mol, params, rec, alpha)
    shell_atom = np.array([int(sh.atom_idx) for sh in shells])
    return dict(
        mol=mol,
        topo=topo,
        params=params,
        H0=H0,
        S=S,
        f0=f0,
        n_occ=n_occ,
        n_basis=n_basis,
        n_shells=len(shells),
        ao_shell=ao_shell,
        shell_atom=shell_atom,
        gamma=gamma,
    )


def s_spectrum(tag, S):
    ev = np.linalg.eigvalsh(S)
    low = ev[:14]
    print(
        f"[{tag}] S: n={S.shape[0]} min={ev.min():+.4e} "
        f"neg={(ev < 0).sum()} below_1e-2={(ev < 1e-2).sum()} "
        f"below_0.05={(ev < 0.05).sum()}"
    )
    print(f"[{tag}]   lowest 14: {np.array2string(low, precision=4)}")
    return ev


class ThresholdMap(BareMap):
    def __init__(self, ctx, temp, tau, label):
        self.tau = tau
        BareMap.__init__(
            self,
            ctx["H0"],
            ctx["S"],
            ctx["gamma"],
            ctx["ao_shell"],
            ctx["n_shells"],
            ctx["n_occ"],
            temp,
            label,
        )
        # redo the orthogonalizer with the requested threshold
        s_eval, s_evec = np.linalg.eigh(ctx["S"])
        keep = s_eval > tau
        self.X = s_evec[:, keep] / np.sqrt(s_eval[keep])
        self.screened = int((~keep).sum())


def newton_hunt(mapf, n, label, max_it=80):
    delta = 1.0e-4

    def jac(q, f_q):
        J = np.zeros((n, n))
        for k in range(n):
            e = np.zeros(n)
            e[k] = delta
            J[:, k] = (mapf(q + e) - f_q) / delta
        return J

    q = np.zeros(n)
    ok = False
    for _ in range(max_it):
        fq = mapf(q)
        r = fq - q
        rn = np.abs(r).max()
        if rn < 1.0e-9:
            ok = True
            break
        J = jac(q, fq)
        try:
            step = np.linalg.solve(J - np.eye(n), r)
        except np.linalg.LinAlgError:
            break
        damping, accepted = 1.0, False
        for _ in range(16):
            qn = q - damping * step
            if np.abs(mapf(qn) - qn).max() < rn:
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            qn = q + 0.1 * r
            if np.abs(mapf(qn) - qn).max() >= rn:
                break
        q = qn
    rfin = np.abs(mapf(q) - q).max()
    print(f"  [{label}] newton: converged={ok} |f(q)-q|_inf={rfin:.3e}")
    return ok, q


def main():
    # ---- (1) spectra across cells ----
    print("== overlap spectra ==", flush=True)
    ctx222 = setup((2, 2, 2), 3.615)
    s_spectrum("cu222", ctx222["S"])
    ctx224 = setup((2, 2, 4), 3.7958)
    ev224 = s_spectrum("cu224", ctx224["S"])
    print("building 4x4x4 (may take a while)...", flush=True)
    ctx444 = setup((4, 4, 4), 3.615)
    s_spectrum("cu444", ctx444["S"])

    # ---- (2) violent-mode structure on 2x2x4 ----
    print("== 2x2x4 violent mode structure (T=0.005) ==", flush=True)
    ctx = ctx224
    n = ctx["n_shells"]
    m = ThresholdMap(ctx, 0.005, 1.0e-10, "ws")
    m.calibrate(ctx["f0"])  # f0 was computed at T=0; recalibrate below
    nat_t, _ = one_shot(ctx["mol"], ctx["topo"], ctx["params"], temperature=0.005)
    m.calibrate(np.asarray(nat_t.shell_charges, dtype=float))
    f0v = m(np.zeros(n))
    delta = 1.0e-4
    J = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = delta
        J[:, k] = (m(e) - f0v) / delta
    w, V = np.linalg.eig(J)
    order = np.argsort(w.real)
    for idx in order[:3]:
        lam = w[idx].real
        mode = V[:, idx].real
        mode /= np.abs(mode).max()
        per_atom = np.zeros(16)
        np.add.at(per_atom, ctx["shell_atom"], mode)
        print(f"  lambda={lam:+.3f} per-atom sums (16 atoms, order i*8+j*4+k):")
        print(f"    {np.array2string(per_atom, precision=3)}")

    # weight of the violent mode on the small-S subspace: map the shell-space
    # mode to AO space (uniform within a shell), then project on the S
    # eigenvectors below 0.05.
    s_eval, s_evec = np.linalg.eigh(ctx["S"])
    small = s_evec[:, s_eval < 0.05]
    mode = V[:, order[0]].real
    ao_mode = mode[ctx["ao_shell"]]
    ao_mode /= np.linalg.norm(ao_mode)
    proj = np.linalg.norm(small.T @ ao_mode)
    print(f"  violent-mode AO-projection onto S-eigenspace(<0.05): {proj:.3f}")

    # ---- (3) screening-threshold sweep on the 2x2x4 bare map ----
    print("== threshold sweep, 2x2x4 bare map ==", flush=True)
    for temp in (0.005, 0.001):
        nat_t, _ = one_shot(
            ctx["mol"], ctx["topo"], ctx["params"], temperature=temp
        )
        f0_t = np.asarray(nat_t.shell_charges, dtype=float)
        base = ThresholdMap(ctx, temp, 1.0e-10, "cal")
        base.calibrate(f0_t)
        n0 = base.n0  # tau-independent reference occupations
        for tau in (1.0e-10, 1.0e-3, 1.0e-2, 3.0e-2, 6.0e-2, 0.1):
            mt = ThresholdMap(ctx, temp, tau, f"tau={tau}")
            mt.n0 = n0
            if mt.X.shape[1] < ctx["n_occ"]:
                print(f"  [T={temp} tau={tau}] kept < n_occ, skip")
                continue
            f0v = mt(np.zeros(n))
            Jt = np.zeros((n, n))
            for k in range(n):
                e = np.zeros(n)
                e[k] = delta
                Jt[:, k] = (mt(e) - f0v) / delta
            evt = np.sort(np.linalg.eigvals(Jt).real)
            dq = np.zeros(n)
            for _ in range(300):
                dq = 0.1 * mt(dq) + 0.9 * dq
            res = np.abs(mt(dq) - dq).max()
            print(
                f"  [T={temp} tau={tau:g}] screened={mt.screened} "
                f"J(0) in [{evt.min():+.2f}, {evt.max():+.2f}] "
                f"simple300 res={res:.3e}",
                flush=True,
            )
            newton_hunt(mt, n, f"T={temp} tau={tau:g}")


if __name__ == "__main__":
    sys.exit(main())
