"""vibe-view periodic showcase: NaCl rocksalt at RKS-PBE / sto-3g.

Single periodic calculation that produces a .qvf carrying every
section vibe-view can render for a periodic system today. NaCl
rocksalt is the right system for a periodic showcase because it's
small (8 atoms in the conventional cell), runs in seconds, and
exercises every periodic-specific rendering path:

  * structure         a cubic conventional cell with 4 Na + 4 Cl,
                      alternating on the rock-salt lattice. The
                      unit-cell wireframe is drawn automatically,
                      and the replication controls tile the cell
                      in 3D.
  * volume.density    the electron density isosurface replicates
                      across the unit-cell boundaries, showing the
                      spherical ionic charge clouds around each
                      atom. With Nx=Ny=Nz=2 you see the full 2×2×2
                      supercell density.
  * atom_properties   Mulliken / Loewdin charges show the expected
                      ionic character: Na positive (red when
                      color-by-charge is on), Cl negative (blue).
  * citations         the BibTeX bundle the runtime assembled:
                      vibe-qc itself, libint, libxc, PBE
                      (Perdew-Burke-Ernzerhof 1996), sto-3g
                      (Hehre-Stewart-Pople 1969).

Run:

    ~/path/to/vibe-qc/.venv/bin/python examples/vibe_view/showcase_nacl_rocksalt.py

Output:

    output-nacl-showcase.qvf       <- open this with vibe-view
    output-nacl-showcase.out
    output-nacl-showcase.system
    output-nacl-showcase.population.{txt,json}
    output-nacl-showcase.density.xsf
    output-nacl-showcase.bibtex
    output-nacl-showcase.references
    output-nacl-showcase.xyz
    output-nacl-showcase.POSCAR
    output-nacl-showcase.cif
    output-nacl-showcase.structure.xsf

Then:

    vibe-view open output-nacl-showcase.qvf

vibe-view will print a startup banner, open the default browser at
http://127.0.0.1:8080, and show the periodic structure with the
unit-cell wireframe. The AppBar reads
``vibe-view — output-nacl-showcase (Na₄Cl₄) — rks/sto-3g``.

Walking the panels:

    1. structure   cubic rocksalt lattice, alternating Na/Cl.
                   Na in purple, Cl in green (Jmol colours).
                   The unit-cell wireframe outlines the
                   conventional cell. The "Periodic Replication"
                   card on the right lets you tile 2×2×2, 3×3×3.
    2. density     start at the default isovalue (0.05 e/bohr³).
                   Lower to 0.02 to see the spherical ionic
                   clouds. Increase replication to 2×2×2 to see
                   the periodic tiling of the density isosurface.
    3. atom_properties     color-coded charges in the 3D viewport.
                   Na is red (positive), Cl is blue (negative).
                   Switch between Mulliken and Loewdin in the
                   right-panel Method dropdown. Toggle "Color
                   atoms by charge" to see the ionicity.
    4. citations   BibTeX block ready to paste into a paper.

For the full guide see docs/user_guide/vibe_view.md and
docs/tutorial/vibe_view_walkthrough.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from vibeqc import (
    Atom,
    BasisSet,
    PeriodicSystem,
    run_periodic_job,
)

HERE = Path(__file__).resolve().parent

ANG2BOHR = 1.0 / 0.529177210903  # 1 Å in bohr

# NaCl rocksalt: conventional cubic cell, a = 5.64 Å.
# The conventional cell contains 4 Na + 4 Cl atoms on the rock-salt
# (NaCl) lattice — two interpenetrating FCC sublattices displaced by
# (a/2, 0, 0).

A_ANG = 5.64
CELL_ANG = [[A_ANG, 0.0, 0.0], [0.0, A_ANG, 0.0], [0.0, 0.0, A_ANG]]

# Na at corners + face centres: FCC sublattice 1
# Cl at edge midpoints + body centre: FCC sublattice 2
HALF = A_ANG / 2.0
ATOM_DATA = [
    # Na — FCC sublattice (corners + face centres)
    ("Na", [0.0, 0.0, 0.0]),
    ("Na", [0.0, HALF, HALF]),
    ("Na", [HALF, 0.0, HALF]),
    ("Na", [HALF, HALF, 0.0]),
    # Cl — FCC sublattice (displaced by a/2,0,0 relative to Na)
    ("Cl", [HALF, 0.0, 0.0]),
    ("Cl", [0.0, HALF, 0.0]),
    ("Cl", [0.0, 0.0, HALF]),
    ("Cl", [HALF, HALF, HALF]),
]

Z_BY_SYM = {
    "H": 1,
    "Li": 3,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ti": 22,
    "Zn": 30,
}


def main() -> None:
    print("=" * 72)
    print(" vibe-view periodic showcase: NaCl rocksalt / RKS-PBE / sto-3g")
    print("=" * 72)
    print()
    print("Producing output-nacl-showcase.qvf with:")
    print("  - structure (periodic, cubic cell, 8 atoms)")
    print("  - volume.density (periodic isosurface)")
    print("  - atom_properties (Mulliken / Lowdin charges)")
    print("  - citations")
    print()

    cell_bohr = np.array(CELL_ANG, dtype=float) * ANG2BOHR
    atoms = [
        Atom(Z_BY_SYM[sym], [float(x) * ANG2BOHR for x in pos_ang])
        for sym, pos_ang in ATOM_DATA
    ]
    system = PeriodicSystem(3, cell_bohr, atoms)
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")

    run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pbe",
        output=HERE / "output-nacl-showcase",
        use_diis=True,
        damping=0.0,
        fmixing_percent=30.0,
        max_iter=80,
        conv_tol_energy=1e-7,
        write_density=True,
        density_spacing_bohr=0.3,  # slightly coarser for faster run
        write_population_file=True,
        citations=True,
        output_qvf=True,
    )

    print()
    print("=" * 72)
    print(" Done. Now open the QVF in vibe-view:")
    print()
    print(f"   vibe-view open {HERE / 'output-nacl-showcase.qvf'}")
    print()
    print(" The browser opens at http://127.0.0.1:8080. Walk the sidebar")
    print(" sections in order; the docstring of this script has a panel-")
    print(" by-panel guide of what to look at in each one.")
    print()
    print(" Try the Periodic Replication card on the right:")
    print("   1. Set Nx=Ny=Nz=2, click Apply — see the 2x2x2 supercell.")
    print("   2. Activate the density section.")
    print("   3. Activate the atom_properties section, toggle")
    print("      color-by-charge to see Na (red +) / Cl (blue -).")
    print("=" * 72)


if __name__ == "__main__":
    main()
