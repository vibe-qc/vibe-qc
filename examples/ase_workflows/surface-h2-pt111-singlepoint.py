"""H2 on Pt(111) surface, periodic ASE geometry fixture.

This script keeps the surface-adsorption setup reproducible while
the periodic ASE calculator adapter is still being completed. The
native periodic drivers already provide energies and gradients, but
``ase.Atoms`` to ``PeriodicSystem`` dispatch is not yet a public ASE
calculator path. Until then, the script:

  1. Builds a Pt(111) 2×2 surface with ``ase.build.fcc111`` —
     pure ASE, no calculator needed.
  2. Adds an H2 molecule above the surface (atop site).
  3. Shows the intended ``VibeQCPeriodic`` calculator + ``BFGS``
     driver shape.
  4. Writes a local geometry artifact
     (``output-surface-h2-pt111.xyz``) for scratch inspection.

Run:
    .venv/bin/python examples/ase_workflows/surface-h2-pt111-singlepoint.py

(Will print the geometry, then exit cleanly with a "not yet
supported" notice, no SCF run.)
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

    # The intended workflow, as a comment-only sketch — keep here so
    # the next maintainer (or our future self) sees the full plan and
    # can wire it up the moment G1 + G2 ship.
    print("\n  Once the periodic ASE adapter lands, this script becomes:")
    print("  ")
    print("    from vibeqc.ase import VibeQCPeriodic           # v0.6+")
    print("    from ase.optimize import BFGS")
    print("  ")
    print("    slab.calc = VibeQCPeriodic(")
    print("        basis='pob-tzvp', functional='PBE', dispersion='d3bj',")
    print("        kmesh=[3, 3, 1],          # 3×3 in-plane, 1 normal")
    print("    )")
    print("    BFGS(slab).run(fmax=0.05)     # full surface relaxation")
    print("    print(slab.get_potential_energy())   # binding energy reference")
    print("  ")
    print("  See `docs/user_guide/ase_integration.md` for the")
    print("  current periodic-ASE integration status.")
    print()
    print("  vibe-qc periodic ASE calculator adapter not yet shipping.")
    print("    Setup written; SCF skipped.")


if __name__ == "__main__":
    main()
