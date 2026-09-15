#!/usr/bin/env python3
"""RIJCOSX K-gradient term decomposition: analytic vs FD at fixed density.

Instrument for the RIJCOSX analytic-gradient parity defect
(glycine/def2-TZVP vs ORCA 6.1.1 EnGrad, 2026-07-29): decomposes the
chain-of-spheres exchange-gradient error into its neglected pieces by
finite-differencing the exchange energy

    E_K(R) = -(1/4) tr(D K_cosx(R))        (fixed converged density D)

with progressively more frozen ingredients:

  (a) full     : grid + overlap-fit Q rebuilt at each displaced geometry
                 -> the true explicit derivative dE_K/dR at fixed D
  (b) frozgrid : grid points/weights fixed in space, Q rebuilt
                 -> isolates the overlap-fit (Q) response
  (c) frozQ    : grid AND Q fixed
                 -> the surface the frozen-grid/frozen-Q analytic formula
                    (compute_cosx_k_gradient_contribution) claims to
                    differentiate

Comparing the analytic K-gradient contribution against (c) measures the
formula's internal error; (c)-(b) the Q-response term; (b)-(a) the
grid/weight-motion term.

Usage:
    python examples/regression/rijcosx_gradient_fd_decompose.py \
        [def2-svp|def2-tzvp] [legacy|1|2|3|4]

The second argument selects the COSX grid: ``legacy`` is the
``default_cosx_grid_options()`` product grid the gradient path uses for
``cosx_grid_level=0``; an integer selects the GridX tier
(``cosx_grid_options_for_level``) the SCF's AUTO default converges on.

Measured 2026-07-29 (H2O, this script, EVIDENCE.md of the fix worktree):

  def2-tzvp legacy grid : formula 3.2e-4, Q-response 1.5e-4,
                          grid motion 1.6e-4, total 5.7e-4 Ha/bohr
  def2-tzvp GridX tier 2: formula 2.4e-3, Q-response 2.8e-3,
                          grid motion 2.8e-4, total 5.5e-3 Ha/bohr
"""

from __future__ import annotations

import sys

import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import (
    GridOptions,
    build_cosx_q,
    build_grid,
    compute_cosx_k,
    compute_cosx_k_gradient_contribution,
    cosx_grid_options_for_level,
)

_A = 1.8897259886
H2O = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * _A, -0.613510 * _A]),
    (1, [0.0, -0.793353 * _A, -0.613510 * _A]),
]
ORB = sys.argv[1] if len(sys.argv) > 1 else "def2-tzvp"
AUX = {"def2-svp": "def2-svp-jk", "def2-tzvp": "def2-tzvp-jk"}[ORB]
GRIDMODE = sys.argv[2] if len(sys.argv) > 2 else "legacy"
H = 1e-4

Zs = [Z for Z, _ in H2O]
pos0 = [list(x) for _, x in H2O]


def make(positions):
    mol = vq.Molecule(
        [vq.Atom(int(Z), list(p)) for Z, p in zip(Zs, positions)]
    )
    return mol, vq.BasisSet(mol, ORB)


def cosx_grid_for(mol):
    if GRIDMODE == "legacy":
        g = GridOptions()
        g.n_radial = 35
        g.n_theta = 9
        g.n_phi = 18
        return build_grid(mol, g)
    return build_grid(mol, cosx_grid_options_for_level(int(GRIDMODE)))


def main() -> int:
    mol0, basis0 = make(pos0)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.max_iter = 200
    opts.density_fit = True
    opts.aux_basis = AUX
    opts.cosx = True
    rhf = vq.run_rhf(mol0, basis0, opts)
    assert rhf.converged
    D = np.asarray(rhf.density)

    grid0 = cosx_grid_for(mol0)
    Q0 = np.asarray(build_cosx_q(basis0, grid0))

    def e_k(positions, mode):
        mol, basis = make(positions)
        if mode == "full":
            K = compute_cosx_k(basis, D, cosx_grid_for(mol))
        elif mode == "frozgrid":
            K = compute_cosx_k(basis, D, grid0)
        elif mode == "frozQ":
            K = compute_cosx_k(basis, D, grid0, q_cached=Q0)
        else:
            raise ValueError(mode)
        return -0.25 * float((D * np.asarray(K)).sum())

    def fd(mode):
        g = np.zeros((len(pos0), 3))
        for a in range(len(pos0)):
            for c in range(3):
                pp = [list(p) for p in pos0]; pp[a][c] += H
                pm = [list(p) for p in pos0]; pm[a][c] -= H
                g[a, c] = (e_k(pp, mode) - e_k(pm, mode)) / (2 * H)
        return g

    g_an = np.asarray(compute_cosx_k_gradient_contribution(
        mol0, basis0, D, grid0, 1.0, np.empty((0, 0)), True))

    np.set_printoptions(precision=8, linewidth=120)
    g_a, g_b, g_c = fd("full"), fd("frozgrid"), fd("frozQ")
    print("== analytic (frozen grid+Q formula) ==\n", g_an)
    print("== FD full (grid+Q respond) ==\n", g_a)
    print("== FD frozen-grid (Q responds) ==\n", g_b)
    print("== FD frozen-grid+frozen-Q ==\n", g_c)
    print()
    print("max|analytic - FD(frozQ)|    =", np.abs(g_an - g_c).max(),
          " <- formula error at frozen level")
    print("max|FD(frozQ) - FD(frozgrid)| =", np.abs(g_c - g_b).max(),
          " <- Q-response term")
    print("max|FD(frozgrid) - FD(full)|  =", np.abs(g_b - g_a).max(),
          " <- grid/weight motion term")
    total = np.abs(g_an - g_a).max()
    print("max|analytic - FD(full)|     =", total,
          " <- total K-gradient error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
