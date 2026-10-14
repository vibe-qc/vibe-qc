"""H2 on Pt(111) surface, periodic ASE geometry fixture.

This script keeps a surface-adsorption setup reproducible without
running an SCF. The periodic ASE calculator exists
(``vibeqc.ase_periodic.VibeQCPeriodic``, Gaussian basis, Ewald or
Gamma-only GDF backend), but a 20-atom Pt slab needs an ECP basis and
a k-mesh, which is a heavy production job rather than an example, so the
script:

  1. Builds a Pt(111) 2×2 surface with ``ase.build.fcc111``,
     pure ASE, no calculator needed.
  2. Adds an H2 molecule above the surface (atop site).
  3. Prints the ``VibeQCPeriodic`` calculator + ``BFGS`` driver shape
     with the real import path and keyword names.
  4. Writes a local geometry artifact
     (``output-surface-h2-pt111.xyz``) for scratch inspection.

Run:
    .venv/bin/python examples/ase_workflows/surface-h2-pt111-singlepoint.py

(Prints the geometry and the calculator sketch, then exits; no SCF run.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.build import add_adsorbate, fcc111
from ase.io import write

HERE = Path(__file__).resolve().parent


def main() -> None:
    print("=" * 68)
    print(" ASE surface workflow:  H2 on Pt(111) geometry fixture")
    print("=" * 68)

    # 4-layer Pt(111) 2×2 slab, 10 Å vacuum.
    slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0, a=3.92)

    # H2 molecule centered over an atop site, 2.0 Å above the surface.
    add_adsorbate(slab, "H", height=2.0, position="ontop")
    add_adsorbate(slab, "H", height=2.0 + 0.74, position="ontop")

    print(f"  Atoms in cell:       {len(slab)}")
    print(f"  Cell (Å):")
    for v in slab.cell:
        print(f"     [{v[0]:8.4f}  {v[1]:8.4f}  {v[2]:8.4f}]")
    print(f"  PBC:                 {tuple(slab.pbc)}")
    print(f"  Pt atoms:            {sum(1 for s in slab.symbols if s == 'Pt')}")
    print(f"  H atoms:             {sum(1 for s in slab.symbols if s == 'H')}")

    xyz_path = HERE / "output-surface-h2-pt111.xyz"
    write(str(xyz_path), slab)
    print(f"\n  Geometry: {xyz_path.relative_to(HERE.parent.parent)}")

    # The production workflow, printed rather than run: the calculator
    # exists, but Pt needs an ECP basis and a converged k-mesh, which is
    # a production job rather than an example.
    print("\n  To relax this slab through the periodic ASE calculator:")
    print("  ")
    print("    from vibeqc.ase_periodic import VibeQCPeriodic")
    print("    from ase.optimize import BFGS")
    print("  ")
    print("    slab.calc = VibeQCPeriodic(")
    print("        basis='pob-tzvp', functional='pbe',")
    print("        kpts=(3, 3, 1),           # 3x3 in-plane, 1 normal")
    print("    )")
    print("    BFGS(slab).run(fmax=0.05)     # full surface relaxation")
    print("    print(slab.get_potential_energy())   # eV")
    print("  ")
    print("  VibeQCPeriodic keywords: basis, kpts, functional, charge,")
    print("  multiplicity, backend ('ewald' or 'gdf'), cutoff_bohr,")
    print("  conv_tol_energy, max_iter. See docs/user_guide/ase_integration.md")
    print("  and docs/user_guide/slabs_and_adsorbates.md.")
    print()
    print("  Geometry fixture written; SCF deliberately skipped.")


if __name__ == "__main__":
    main()
