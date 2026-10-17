"""Hydrogen peroxide, a dihedral scan and a genuinely relaxed O-O bond scan.

System: HOOH at RHF/STO-3G, built from R(O-O) = 1.45 angstrom,
R(O-H) = 0.97 angstrom and an H-O-O angle of 100 degrees.
Method: ``vibeqc.relaxed_scan`` twice.

1. The H-O-O-H dihedral (atoms 0-1-2-3) at 0, 30, ..., 180 degrees, seven
   points. Note what "relaxed" means for a four-atom molecule: the atoms
   that define the scanned coordinate are always held fixed, and here every
   atom belongs to the dihedral, so nothing is left to relax and this pass
   is a rigid scan of the torsion at the fixed bond lengths and angles.
2. The O-O bond (atoms 1-2) from 1.35 to 1.55 angstrom, five points,
   started from the lowest-energy dihedral geometry. Only the two oxygens
   are constrained, so both hydrogens relax at every point; this is the
   relaxed scan the tutorial describes.

Files (next to this script):
    input-h2o2-relaxed-scan-dihedral.qvf   reaction-path archive, one frame per angle
    input-h2o2-relaxed-scan-oo-bond.qvf    the same for the bond scan
(relaxed_scan writes only the QVF; the tables below go to stdout.)

Success: the torsional profile has its minimum at 150 degrees on this
30 degree grid; the cis barrier (0 degrees) is 29.90 kcal/mol and the trans
barrier (180 degrees) 0.012 kcal/mol above that minimum. RHF/STO-3G with
frozen bond angles exaggerates the cis barrier badly (experiment: roughly 7 cis
and 1 trans, kcal/mol), which is the honest lesson of a rigid scan in a
minimal basis. The relaxed O-O scan is a smooth single-well curve with its
grid minimum at 1.40 angstrom. Tolerance: 0.05 kcal/mol on the barriers
(vibe-qc 0.17.x development build, Apple M5 Max, about 20 s for both scans including start-up).
"""

from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem
ANG = 1.8897259886
KCAL = 627.5094740631


def hooh(dihedral_deg: float) -> vq.Molecule:
    """H1-O1-O2-H2 with the O-O axis along z (atom order H, O, O, H; bohr)."""
    r_oo, r_oh, ang = 1.45 * ANG, 0.97 * ANG, np.deg2rad(100.0)
    d = np.deg2rad(dihedral_deg)
    o1 = np.array([0.0, 0.0, 0.0])
    o2 = np.array([0.0, 0.0, r_oo])
    h1 = np.array([r_oh * np.sin(ang), 0.0, -r_oh * np.cos(ang)])
    h2 = np.array([r_oh * np.sin(ang) * np.cos(d), r_oh * np.sin(ang) * np.sin(d),
                   r_oo + r_oh * np.cos(ang)])
    return vq.Molecule([vq.Atom(1, h1), vq.Atom(8, o1), vq.Atom(8, o2), vq.Atom(1, h2)])


# 1. Torsion, 7 points. All four atoms define the coordinate, so this is rigid.
angles_deg = np.linspace(0.0, 180.0, 7)
tors = vq.relaxed_scan(
    hooh(120.0), "sto-3g",
    coordinate=("dihedral", 0, 1, 2, 3),
    values=np.deg2rad(angles_deg),
    method="RHF",
    output=str(HERE / f"{STEM}-dihedral"),
)
e_min = float(tors.energies.min())
i_min = int(np.argmin(tors.energies))
print("\n  HOOH torsion, RHF/STO-3G (rigid: every atom is a constraint atom)")
print(f"  {'dihedral (deg)':>15s} {'E (Ha)':>16s} {'E - E_min (kcal/mol)':>21s}  converged")
for ang, e, ok in zip(angles_deg, tors.energies, tors.converged_flags):
    print(f"  {ang:15.1f} {e:16.8f} {(e - e_min) * KCAL:21.3f}  {ok}")
print(f"  minimum on this grid: {angles_deg[i_min]:.0f} degrees")
print(f"  cis barrier   (0 deg)   = {(tors.energies[0] - e_min) * KCAL:8.3f} kcal/mol")
print(f"  trans barrier (180 deg) = {(tors.energies[-1] - e_min) * KCAL:8.3f} kcal/mol")

# 2. O-O bond, 5 points, hydrogens relax; seeded from the torsional minimum.
r_oo_ang = np.linspace(1.35, 1.55, 5)
bond = vq.relaxed_scan(
    tors.geometries[i_min], "sto-3g",
    coordinate=("bond", 1, 2),
    values=r_oo_ang * ANG,
    method="RHF",
    output=str(HERE / f"{STEM}-oo-bond"),
)
e_bmin = float(bond.energies.min())
print("\n  HOOH O-O bond scan, RHF/STO-3G (both H atoms relax at each point)")
print(f"  {'R(O-O) (angstrom)':>18s} {'E (Ha)':>16s} {'E - E_min (kcal/mol)':>21s}  converged")
for r, e, ok in zip(r_oo_ang, bond.energies, bond.converged_flags):
    print(f"  {r:18.3f} {e:16.8f} {(e - e_bmin) * KCAL:21.3f}  {ok}")
print(f"  grid minimum at R(O-O) = {r_oo_ang[int(np.argmin(bond.energies))]:.2f} angstrom; "
      f"{tors!r}; {bond!r}")
