"""Solver-wall fixed-point existence hunt (IID 130 / IID 101 family).

Settles whether the >30-atom GFN2-SECCM charge-solver wall is a solver
gap or a property of the map. For the bare WS-folded map (ewald_gamma,
aes off; optionally the real gam3 confinement) on fcc Cu cells:

  * grid-scan the symmetry-reduced uniform manifold (charge
    conservation makes it 2-dim) for a small-amplitude fixed point;
  * hunt the full-dimensional fixed point with exact-FD-Jacobian
    line-searched Newton, scipy hybr, and Newton-Krylov multi-start;
  * ladder the electronic temperature (smearing does NOT tame the
    charge-density-wave response: J(0) min goes from -14 at T=0.005
    to -23 at T=0.05 on 2x2x4).

Verdict recorded in handovers/HANDOVER_SECCM_BULK3D.md (2026-08-26):
the 2x2x4 and 4x4x4 maps have NO physical fixed point; every root any
method finds sits in the large-amplitude CDW basin (|dq_shell| 6.6-8.6
on 2x2x4; the 4x4x4 hunt diverges to |dq| ~ 3e2). The 2x2x2 control
finds its physical fixed point with the same tooling.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import optimize

from probe_wall_map_mechanism import BareMap, build, fermi_occ, one_shot, gamma_matrix
from probe_wall_spectra_thresholds import newton_hunt, setup


class Gam3Map(BareMap):
    """Bare map plus the real on-site third-order confinement term."""

    def __init__(self, *args, gam3_shell=None, **kw):
        BareMap.__init__(self, *args, **kw)
        self.g3 = gam3_shell

    def raw_pops(self, dq):
        V = self.gamma @ dq - self.g3 * dq * dq
        Vao = V[self.ao_shell]
        H = self.H0 + 0.5 * self.S * (Vao[:, None] + Vao[None, :])
        Ht = self.X.T @ H @ self.X
        eps, Ct = np.linalg.eigh(Ht)
        C = self.X @ Ct
        occ = fermi_occ(eps, self.n_occ, self.temp)
        D = (C * occ) @ C.T
        pops_ao = np.einsum("ij,ji->i", D, self.S)
        pops = np.zeros(self.n_shells)
        np.add.at(pops, self.ao_shell, pops_ao)
        return pops, eps


GAM3_L_SCALE = [1.0, 0.5, 0.25, 0.25]


def uniform_grid_scan(mapf, n_atoms, spa, lim=1.5, npts=31):
    """Grid the 2-dim uniform manifold (dq = (x, y, -x-y) per atom)."""
    best, bx = 1.0e9, None
    for x in np.linspace(-lim, lim, npts):
        for y in np.linspace(-lim, lim, npts):
            q = np.tile(np.array([x, y, -x - y]), n_atoms)
            fu = (mapf(q) - q).reshape(n_atoms, spa).mean(axis=0)
            r = np.abs(fu).max()
            if r < best:
                best, bx = r, (x, y)
    return best, bx


def hunt(mapf, n, label):
    okN, qN = newton_hunt(mapf, n, label)

    def F(q):
        return mapf(q) - q

    rows = []
    for seed in range(3):
        x0 = (
            np.zeros(n)
            if seed == 0
            else 0.05 * np.random.default_rng(seed).standard_normal(n)
        )
        try:
            s = optimize.root(F, x0, method="krylov", options={"maxiter": 300})
            rows.append((np.abs(F(s.x)).max(), np.abs(s.x).max()))
        except Exception:  # noqa: BLE001
            rows.append((float("nan"), float("nan")))
    print(
        f"  [{label}] newton={okN} krylov (res,|q|_inf): "
        f"{[(f'{r:.1e}', f'{q:.2f}') for r, q in rows]}",
        flush=True,
    )
    return okN


def run_cell(replicas, a, temps=(0.005, 0.001), grid=True):
    n_atoms = replicas[0] * replicas[1] * replicas[2]
    print(f"== fcc Cu {replicas} a={a} (bare map) ==", flush=True)
    ctx = setup(replicas, a)
    n, n_occ = ctx["n_shells"], ctx["n_occ"]
    spa = n // n_atoms
    for temp in temps:
        nat_t, _ = one_shot(
            ctx["mol"], ctx["topo"], ctx["params"], temperature=temp
        )
        f0_t = np.asarray(nat_t.shell_charges, dtype=float)
        m = BareMap(
            ctx["H0"],
            ctx["S"],
            ctx["gamma"],
            ctx["ao_shell"],
            n,
            n_occ,
            temp,
            "ws",
        )
        m.calibrate(f0_t)
        if grid:
            best, bx = uniform_grid_scan(m, n_atoms, spa)
            print(
                f"  T={temp}: best uniform-manifold residual {best:.4f} "
                f"at {bx}",
                flush=True,
            )
        hunt(m, n, f"T={temp}")


def main():
    run_cell((2, 2, 2), 3.615)
    run_cell((2, 2, 4), 3.7958)
    run_cell((4, 4, 4), 3.615, temps=(0.005,), grid=False)


if __name__ == "__main__":
    sys.exit(main())
