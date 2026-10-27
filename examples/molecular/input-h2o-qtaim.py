"""Water, QTAIM topology of the RHF/6-31G* density: critical points, densities and Laplacians.

System: H2O/6-31G* at the fixed geometry shared by these examples (bohr).
Method: ``run_job(method="rhf", basis="6-31g*", qtaim=True)`` runs the SCF
and the Bader topological analysis and stores the result in the QVF archive
for vibe-view; in this build the .out carries no QTAIM block and the
RHFResult (a C++ object) has no ``qtaim`` attribute. For the numbers the
script calls the same engine directly, ``vibeqc.qtaim_analysis(result,
basis, molecule)``, which evaluates the density, its gradient and its
Hessian on a grid (default spacing 0.15 bohr) and locates the (3,-3)
nuclear and (3,-1) bond critical points. Each bond critical point carries
``atom_pair``, the two atoms it connects, and that is the bonding
connectivity this script reports.

Each ``BondPath`` in ``qtaim.bond_paths`` is a traced gradient path: it
leaves the bond critical point along plus and minus the eigenvector of the
one positive Hessian eigenvalue and follows steepest ascent on rho to the two
nuclei, so the (N, 3) polyline (angstrom) runs from ``atoms[0]`` through the
critical point to ``atoms[1]``. The script prints each path's point count and
length next to the nuclear separation.

Files (next to this script):
    input-h2o-qtaim.qvf       wavefunction plus the QTAIM analysis for vibe-view
    input-h2o-qtaim.out       SCF, charges, dipole (no QTAIM block in this build)
    plus .molden / .system / .xyz / .population.* sidecars

Success: two bond critical points, one per O-H bond, at (0, +-0.629,
-0.493) angstrom with rho = 2.1285 e/angstrom^3 (0.3154 e/bohr^3),
Laplacian -37.85 e/angstrom^5 (-1.571 e/bohr^5) and ellipticity 0.024,
connecting atoms O1-H2 and O1-H3, plus three nuclear critical points (the
oxygen at the origin and one on each hydrogen). The negative Laplacian at
both BCPs is the shared-shell (covalent) signature. Each of the two bond
paths has 24 points and a length of 1.0032 angstrom against a nuclear
separation of 1.0030 angstrom (a straight bond). The grid search locates
the critical points to about half a grid spacing, so tighten
``grid_spacing`` for publication numbers. Tolerance: 1e-3 on rho, 0.05 on
the Laplacian at the BCPs (vibe-qc 0.17.x development build, Apple M5 Max,
10 to 40 s depending on machine load).
"""

from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem
BOHR = 0.52917721067          # angstrom per bohr
TO_AU_RHO = BOHR ** 3         # e/angstrom^3 -> e/bohr^3
TO_AU_LAP = BOHR ** 5         # e/angstrom^5 -> e/bohr^5

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])
BASIS = "6-31g*"
SYMBOLS = ["O", "H", "H"]

result = vq.run_job(mol, basis=BASIS, method="rhf", qtaim=True, output=str(HERE / STEM))
basis = vq.BasisSet(mol, BASIS)
qtaim = vq.qtaim_analysis(result, basis, mol)

cps = qtaim.critical_points
print(f"\n  QTAIM on RHF/{BASIS} water: {len(cps)} critical points, "
      f"grid spacing {qtaim.grid_spacing} bohr")
print(f"  {'type':5s} {'position (angstrom)':>26s} {'rho (e/A^3)':>12s} {'rho (au)':>9s} "
      f"{'lap (e/A^5)':>12s} {'lap (au)':>9s} {'ellipt.':>8s}  atoms")
for cp in sorted(cps, key=lambda c: (c.type != "bcp", c.position[1])):
    pos = "(" + ", ".join(f"{x:7.3f}" for x in cp.position) + ")"
    ell = f"{cp.ellipticity:8.4f}" if cp.ellipticity is not None else f"{'':8s}"
    pair = ("-".join(SYMBOLS[i] + str(i + 1) for i in sorted(cp.atom_pair))
            if cp.atom_pair is not None else "")
    print(f"  {cp.type:5s} {pos:>26s} {cp.rho:12.4f} {cp.rho * TO_AU_RHO:9.4f} "
          f"{cp.laplacian:12.3f} {cp.laplacian * TO_AU_LAP:9.4f} {ell}  {pair}")

bcps = [cp for cp in cps if cp.type == "bcp"]
bonds = sorted(tuple(sorted(cp.atom_pair)) for cp in bcps)
print(f"\n  {len(bcps)} bond critical points; connectivity from atom_pair: "
      + ", ".join(f"{SYMBOLS[i]}{i + 1}-{SYMBOLS[j]}{j + 1}" for i, j in bonds))
print(f"  Laplacian negative at all of them: {all(cp.laplacian < 0 for cp in bcps)} "
      "(shared-shell interaction)")
for bp in qtaim.bond_paths:
    pts = np.asarray(bp.path)
    length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    i, j = bp.atoms
    sep = np.linalg.norm(np.asarray(mol.atoms[i].xyz) - np.asarray(mol.atoms[j].xyz)) * BOHR
    print(f"  bond path {SYMBOLS[i]}{i + 1}-{SYMBOLS[j]}{j + 1}: {len(pts)} points, "
          f"length {length:.4f} A, nuclear separation {sep:.4f} A")
print(f"  E(RHF) = {result.energy:.8f} Ha")
