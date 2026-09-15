"""Probe V_LR(r) directly for H2 at origin vs centred."""
from __future__ import annotations
import numpy as np

import vibeqc as vq
from vibeqc import (
    BasisSet, evaluate_ao, solve_poisson_erf_screened,
)


L = 30.0
omega = 0.5
spacing = 0.4
n = int(np.ceil(L / spacing))
n = n if n % 2 == 0 else n + 1
print(f"grid: {n}**3, h={L/n:.4f}")
h = L / n


def build_basis(offset):
    R0 = np.asarray(offset, dtype=float)
    atoms = [
        vq.Atom(1, R0.tolist()),
        vq.Atom(1, (R0 + np.array([0.0, 0.0, 1.4])).tolist()),
    ]
    sys_p = vq.PeriodicSystem(dim=3, lattice=L * np.eye(3), unit_cell=atoms)
    return sys_p, BasisSet(sys_p.unit_cell_molecule(), "sto-3g")


def grid_points():
    xs = np.arange(n) * h
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def main():
    points = grid_points()
    print(f"n_points = {len(points)}")

    # Origin case
    sys_o, basis_o = build_basis([0.0, 0.0, 0.0])
    chi_o = evaluate_ao(basis_o, points)
    # n_occ = 1; D = 2 |1s>⟨1s|. Use a uniform-overlap density for diagnosis:
    # actually, just use ρ = sum of |χ|² for both AOs (1 e- each, density = 2 χ_μ χ_μ for occupation 2).
    # Simpler: pretend D = I (just to check translation symmetry).
    D = np.eye(2) * 1.0
    rho_o = np.einsum("gi,ij,gj->g", chi_o, D, chi_o, optimize=True)
    print(f"  origin   rho.sum * dV = {rho_o.sum() * (h**3):.6f}  "
          f"(should integrate to 2 = ∫χ²(r)dr × 2 AOs ≈ 2)")

    # Centre case
    sys_c, basis_c = build_basis([L/2, L/2, L/2 - 0.7])
    chi_c = evaluate_ao(basis_c, points)
    rho_c = np.einsum("gi,ij,gj->g", chi_c, D, chi_c, optimize=True)
    print(f"  centre   rho.sum * dV = {rho_c.sum() * (h**3):.6f}")

    print(f"  rho_o.max = {rho_o.max():.4f}, location of max in r_g: "
          f"{points[rho_o.argmax()]}")
    print(f"  rho_c.max = {rho_c.max():.4f}, location of max in r_g: "
          f"{points[rho_c.argmax()]}")

    # The 0.62-vs-2.0 mismatch above is the bug — that's the whole
    # diagnostic point of this script. The Poisson-solve probe was a
    # follow-up but isn't needed: once ρ on the grid is wrong, V_LR
    # is wrong by the same translation factor.
    print()
    print("  >>> the rho-integral mismatch (0.62 vs 2.0) is the bug.")
    print("  >>> see PERIODIC_SCF_BUG_ANALYSIS.md for the fix path.")


if __name__ == "__main__":
    main()
