"""Audit the GGA XC Pulay gradient term (cpp/src/gradient.cpp::
xc_pulay_gradient_rks_gga).

Decomposition strategy:
  * vibe-qc analytic RKS-PBE gradient = Pulay term only (no Becke-weight
    / grid-response derivative — documented omission in gradient.hpp).
  * PySCF RKS-PBE with grid_response=False = Pulay term only too.
    -> vibe-qc analytic vs PySCF(gr=False) isolates whether vibe-qc's
       Pulay term is CORRECT (should agree to grid-discretization error).
  * PySCF(gr=True) - PySCF(gr=False) = the grid-response term magnitude.
  * vibe-qc analytic vs vibe-qc FD = vibe-qc's own missing piece
    (should ~= grid-response term).

Run on H2O/def2-svp (low-l) and H2CO/def2-tzvp (exercises the GGA
Hessian path M2/M3 with f-shell basis functions), default + fine grid.
"""
from __future__ import annotations
import numpy as np
from vibeqc import (
    Atom, Molecule, BasisSet, RKSOptions, GridOptions,
    compute_gradient_rks, run_rks,
)

ANG = 1.0 / 0.529177210903

H2O = [(8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.793353 * ANG, -0.613510 * ANG]),
        (1, [0.0, -0.793353 * ANG, -0.613510 * ANG])]
H2CO = [(6, [0.0, 0.0, 0.0]),
        (8, [0.0, 0.0, 1.205 * ANG]),
        (1, [0.0, 0.943 * ANG, -0.587 * ANG]),
        (1, [0.0, -0.943 * ANG, -0.587 * ANG])]


def vibeqc_rks_grad(atoms, basis_name, functional, grid_opts):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, basis_name)
    opts = RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    opts.grid = grid_opts
    res = run_rks(mol, basis, opts)
    assert res.converged
    return np.array(compute_gradient_rks(mol, basis, res, grid_opts)), res.energy


def vibeqc_rks_energy(atoms, basis_name, functional, grid_opts):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, basis_name)
    opts = RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    opts.grid = grid_opts
    res = run_rks(mol, basis, opts)
    assert res.converged
    return res.energy


def vibeqc_fd_grad(atoms, basis_name, functional, grid_opts, h=1e-4,
                   only=None):
    """only: list of (atom, comp) to compute; None = all."""
    n = len(atoms)
    g = np.zeros((n, 3))
    pairs = only if only is not None else [
        (A, c) for A in range(n) for c in range(3)]
    for (A, c) in pairs:
        ap = [(Z, list(xyz)) for Z, xyz in atoms]
        am = [(Z, list(xyz)) for Z, xyz in atoms]
        ap[A][1][c] += h
        am[A][1][c] -= h
        ep = vibeqc_rks_energy(ap, basis_name, functional, grid_opts)
        em = vibeqc_rks_energy(am, basis_name, functional, grid_opts)
        g[A, c] = (ep - em) / (2 * h)
    return g


def pyscf_rks_grad(atoms, basis_name, xc, grid_level, grid_response):
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grid_level
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    g = mf.nuc_grad_method()
    g.grid_response = grid_response
    return g.kernel()


def run_case(name, atoms, basis_name, vq_func, ps_xc, grid_opts,
             ps_level, fd_only=None):
    print(f"\n=== {name} | {basis_name} | grid {grid_opts.n_radial}x"
          f"{grid_opts.n_theta}x{grid_opts.n_phi} / PySCF L{ps_level} ===")
    g_vq, _ = vibeqc_rks_grad(atoms, basis_name, vq_func, grid_opts)
    g_ps_nr = pyscf_rks_grad(atoms, basis_name, ps_xc, ps_level, False)
    g_ps_gr = pyscf_rks_grad(atoms, basis_name, ps_xc, ps_level, True)

    d_pulay = np.abs(g_vq - g_ps_nr).max()
    d_full = np.abs(g_vq - g_ps_gr).max()
    grid_resp = np.abs(g_ps_gr - g_ps_nr).max()
    print(f"  vibe-qc analytic vs PySCF(gr=False)  [Pulay parity]  : {d_pulay:.3e}")
    print(f"  PySCF(gr=True) - PySCF(gr=False)     [grid-resp size] : {grid_resp:.3e}")
    print(f"  vibe-qc analytic vs PySCF(gr=True)   [full-grad gap]  : {d_full:.3e}")

    if fd_only is not None:
        g_fd = vibeqc_fd_grad(atoms, basis_name, vq_func, grid_opts,
                              only=fd_only)
        for (A, c) in fd_only:
            print(f"  FD check atom {A} comp {c}: vibeqc_ana={g_vq[A,c]:+.6e} "
                  f"vibeqc_fd={g_fd[A,c]:+.6e} diff={g_vq[A,c]-g_fd[A,c]:+.2e} "
                  f"(diff ~= vibe-qc's own missing grid-response)")


if __name__ == "__main__":
    default = GridOptions()
    fine = GridOptions()
    fine.n_radial = 150; fine.n_theta = 25; fine.n_phi = 48

    # H2O / def2-svp — low-l, default + fine grid
    run_case("H2O RKS-PBE", H2O, "def2-svp", "PBE", "pbe,pbe",
             default, 4, fd_only=[(0, 1), (1, 2)])
    run_case("H2O RKS-PBE", H2O, "def2-svp", "PBE", "pbe,pbe",
             fine, 6, fd_only=[(0, 1), (1, 2)])

    # H2CO / def2-tzvp — exercises GGA Hessian path with f-shells
    run_case("H2CO RKS-PBE", H2CO, "def2-tzvp", "PBE", "pbe,pbe",
             default, 4, fd_only=[(1, 2)])
    run_case("H2CO RKS-PBE", H2CO, "def2-tzvp", "PBE", "pbe,pbe",
             fine, 6, fd_only=[(1, 2)])
