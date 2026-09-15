"""Measure actual DF gradient residuals to set data-driven test tolerances.

For each (molecule, basis, aux, method) in the DF gradient test suite,
report:
  * DF-analytic vs direct-analytic  max abs diff   (bounded by DF fit error)
  * DF-analytic vs DF-finite-diff   max abs diff   (bounded by FD floor;
                                                    isolates kernel)
"""
from __future__ import annotations
import numpy as np
from vibeqc import (
    Atom, Molecule, BasisSet, GradientOptions, GridOptions,
    RHFOptions, RKSOptions, UHFOptions, UKSOptions,
    compute_gradient, compute_gradient_rks,
    compute_gradient_uhf, compute_gradient_uks,
    run_rhf, run_rks, run_uhf, run_uks,
)

ANG = 1.0 / 0.529177210903

GEOMETRIES = {
    "H2":  [(1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.7414 * ANG])],
    "H2O": [(8, [0.0, 0.0, 0.0]),
            (1, [0.0, 0.793353 * ANG, -0.613510 * ANG]),
            (1, [0.0, -0.793353 * ANG, -0.613510 * ANG])],
    "CH4": [(6, [0.0, 0.0, 0.0]),
            (1, [+0.626 * ANG, +0.626 * ANG, +0.626 * ANG]),
            (1, [-0.626 * ANG, -0.626 * ANG, +0.626 * ANG]),
            (1, [-0.626 * ANG, +0.626 * ANG, -0.626 * ANG]),
            (1, [+0.626 * ANG, -0.626 * ANG, -0.626 * ANG])],
}
OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97 * ANG])]
H2CO = ([6, 8, 1, 1], np.array([
    (0.0, 0.0, 0.000), (0.0, 0.0, 1.205),
    (0.0, 0.943, -0.587), (0.0, -0.943, -0.587)]) * ANG)


def fd_grad(energy_at, positions, h=1e-4):
    n = len(positions)
    g = np.zeros((n, 3))
    for A in range(n):
        for c in range(3):
            pp = [list(p) for p in positions]; pp[A][c] += h
            pm = [list(p) for p in positions]; pm[A][c] -= h
            g[A, c] = (energy_at(pp) - energy_at(pm)) / (2 * h)
    return g


def probe_rhf(label, Zs, pos, orb, aux):
    mol = Molecule([Atom(int(z), list(p)) for z, p in zip(Zs, pos)])
    basis = BasisSet(mol, orb)
    o = RHFOptions(); o.conv_tol_energy = 1e-12; o.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, o)
    g_direct = np.array(compute_gradient(mol, basis, rhf))
    go = GradientOptions(); go.density_fit = True; go.aux_basis = aux
    g_df = np.array(compute_gradient(mol, basis, rhf, go))

    # DF-vs-FD: re-run DF-RHF at displaced geometry
    def df_energy(positions):
        m = Molecule([Atom(int(z), list(p)) for z, p in zip(Zs, positions)])
        b = BasisSet(m, orb)
        oo = RHFOptions(); oo.conv_tol_energy = 1e-12; oo.conv_tol_grad = 1e-10
        oo.density_fit = True; oo.aux_basis = aux
        hf = run_rhf(m, b, oo)
        return hf.energy
    g_fd = fd_grad(df_energy, [list(p) for p in pos])

    print(f"  {label:38s} DF-vs-direct={np.abs(g_df-g_direct).max():.2e}  "
          f"DF-vs-FD={np.abs(g_df-g_fd).max():.2e}")


def probe_rks(label, Zs, pos, orb, aux, func):
    mol = Molecule([Atom(int(z), list(p)) for z, p in zip(Zs, pos)])
    basis = BasisSet(mol, orb)
    o = RKSOptions(); o.functional = func
    o.conv_tol_energy = 1e-9; o.conv_tol_grad = 1e-7; o.max_iter = 200
    rks = run_rks(mol, basis, o)
    g_direct = np.array(compute_gradient_rks(mol, basis, rks, GridOptions()))
    go = GradientOptions(); go.density_fit = True; go.aux_basis = aux
    g_df = np.array(compute_gradient_rks(mol, basis, rks, GridOptions(), go))
    print(f"  {label:38s} DF-vs-direct={np.abs(g_df-g_direct).max():.2e}")


def probe_uhf(label, atoms, orb, aux):
    Zs = [z for z, _ in atoms]; pos = [list(p) for _, p in atoms]
    mol = Molecule([Atom(int(z), list(p)) for z, p in zip(Zs, pos)],
                   multiplicity=2)
    basis = BasisSet(mol, orb)
    o = UHFOptions(); o.conv_tol_energy = 1e-11; o.conv_tol_grad = 1e-9
    o.max_iter = 200
    uhf = run_uhf(mol, basis, o)
    g_direct = np.array(compute_gradient_uhf(mol, basis, uhf))
    go = GradientOptions(); go.density_fit = True; go.aux_basis = aux
    g_df = np.array(compute_gradient_uhf(mol, basis, uhf, go))
    print(f"  {label:38s} DF-vs-direct={np.abs(g_df-g_direct).max():.2e}")


def probe_uks(label, atoms, orb, aux, func):
    Zs = [z for z, _ in atoms]; pos = [list(p) for _, p in atoms]
    mol = Molecule([Atom(int(z), list(p)) for z, p in zip(Zs, pos)],
                   multiplicity=2)
    basis = BasisSet(mol, orb)
    o = UKSOptions(); o.functional = func
    o.conv_tol_energy = 1e-9; o.conv_tol_grad = 1e-7; o.max_iter = 500
    o.damping = 0.7
    uks = run_uks(mol, basis, o)
    g_direct = np.array(compute_gradient_uks(mol, basis, uks))
    go = GradientOptions(); go.density_fit = True; go.aux_basis = aux
    g_df = np.array(compute_gradient_uks(mol, basis, uks, options=go))
    print(f"  {label:38s} DF-vs-direct={np.abs(g_df-g_direct).max():.2e}")


if __name__ == "__main__":
    print("RHF DF gradient cases:")
    for m in ("H2", "H2O", "CH4"):
        atoms = GEOMETRIES[m]
        probe_rhf(f"{m}/def2-svp", [z for z, _ in atoms],
                  [list(p) for _, p in atoms], "def2-svp", "def2-svp-jk")
    probe_rhf("H2CO/def2-tzvp", H2CO[0], list(H2CO[1]),
              "def2-tzvp", "def2-tzvp-jk")

    print("RKS DF gradient cases:")
    for func, m in (("LDA", "H2O"), ("PBE", "H2O"),
                    ("B3LYP", "H2O"), ("LDA", "CH4")):
        atoms = GEOMETRIES[m]
        probe_rks(f"{func}/{m}/def2-svp", [z for z, _ in atoms],
                  [list(p) for _, p in atoms], "def2-svp",
                  "def2-svp-jk", func)

    print("UHF / UKS DF gradient cases:")
    probe_uhf("UHF/OH/def2-svp", OH, "def2-svp", "def2-svp-jk")
    probe_uks("UKS-PBE/OH/def2-svp", OH, "def2-svp", "def2-svp-jk", "PBE")
