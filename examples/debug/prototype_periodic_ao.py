"""Prototype: does summing AO images cure the rho-integration mismatch?

If chi_periodic(r) = Σ_g χ(r - g) summed over a 3x3x3 image cube, the
grid integral of rho should hit 2.0 for both H2 at the origin and at
the centre. Quick check before plumbing the fix into the driver.
"""
from __future__ import annotations
import itertools
import numpy as np

import vibeqc as vq
from vibeqc import BasisSet, Atom, Molecule, evaluate_ao, PeriodicSystem


L = 30.0
spacing = 0.4
n = int(np.ceil(L / spacing))
n = n if n % 2 == 0 else n + 1
h = L / n


def grid_points():
    xs = np.arange(n) * h
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def shifted_basis(basis: BasisSet, sys_p: PeriodicSystem, dr) -> BasisSet:
    shifted = []
    for a in sys_p.unit_cell:
        x, y, z = a.xyz
        shifted.append(Atom(int(a.Z),
                            [float(x + dr[0]),
                             float(y + dr[1]),
                             float(z + dr[2])]))
    mol = Molecule(shifted, sys_p.charge, sys_p.multiplicity)
    return BasisSet(mol, basis.name)


def chi_periodic(basis, sys_p, points, image_radius=1):
    """Sum AO images within ±image_radius lattice vectors along each axis."""
    chi_p = np.zeros((points.shape[0], basis.nbasis), dtype=float)
    a, b, c = sys_p.lattice[0, 0], sys_p.lattice[1, 1], sys_p.lattice[2, 2]
    for ix, iy, iz in itertools.product(
        range(-image_radius, image_radius + 1), repeat=3
    ):
        if ix == 0 and iy == 0 and iz == 0:
            chi_p += evaluate_ao(basis, points)
        else:
            dr = (ix * a, iy * b, iz * c)
            chi_p += evaluate_ao(shifted_basis(basis, sys_p, dr), points)
    return chi_p


def build(offset):
    R0 = np.asarray(offset, dtype=float)
    atoms = [
        Atom(1, R0.tolist()),
        Atom(1, (R0 + np.array([0.0, 0.0, 1.4])).tolist()),
    ]
    sys_p = PeriodicSystem(dim=3, lattice=L * np.eye(3), unit_cell=atoms)
    basis = BasisSet(sys_p.unit_cell_molecule(), "sto-3g")
    return sys_p, basis


def main():
    points = grid_points()
    print(f"grid: {n}**3, h={h:.4f}")
    print(f"n_points = {len(points)}")
    print()

    for label, offset, ir in [
        ("origin,   no images",   [0.0, 0.0, 0.0],     0),
        ("centre,   no images",   [L/2, L/2, L/2-0.7], 0),
        ("origin,   ±1 images",   [0.0, 0.0, 0.0],     1),
        ("centre,   ±1 images",   [L/2, L/2, L/2-0.7], 1),
        ("origin,   ±2 images",   [0.0, 0.0, 0.0],     2),
        ("centre,   ±2 images",   [L/2, L/2, L/2-0.7], 2),
    ]:
        sys_p, basis = build(offset)
        chi_p = chi_periodic(basis, sys_p, points, image_radius=ir)
        # rho = chi_p (D=I) chi_p^T diagonal
        rho = np.einsum("gi,ij,gj->g", chi_p, np.eye(2), chi_p, optimize=True)
        N_e = rho.sum() * (h ** 3)
        print(f"  {label:25s}: rho_integrated = {N_e:.6f}")


if __name__ == "__main__":
    main()
