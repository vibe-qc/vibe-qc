"""vibe-view periodic slab showcase: CO on MgO(001) at RKS-PBE / sto-3g.

A 1×1 conventional-cell MgO(001) monolayer with a CO molecule
adsorbed C-down above a surface Mg. This exercises several vibe-view
features that a bulk periodic example can't reach:

  * is_periodic  — 2D slab (periodic in xy, finite in z). The
    unit-cell wireframe shows the slab dimensions; replication in
    Nx/Ny tiles the surface while Nz stays 1.
  * mixed-Z     — Mg (Z=12), O (Z=8), C (Z=6) in one scene.
    Jmol colours: Mg silver, O red, C dark grey. The dangling-bond
    fix (atom names include index + replication slot) prevents
    same-symbol atoms from silently overwriting each other.
  * adsorbate   — CO sits above the surface. The C-O bond is visible
    as a cylinder. O in the adsorbate (red) is distinct from the
    surface O atoms in the MgO layer.
  * density     — periodic isosurface replicates in xy but not z.
    With Nx=Ny=2 you see the 2×2 surface cell with adsorbates on
    every site.
  * atom_properties — Mulliken / Löwdin charges show surface Mg
    partly positive, surface O partly negative, CO polarized with
    C→O charge transfer. The charge tints highlight the chemically
    distinct atoms.
  * citations   — BibTeX bundle: vibe-qc, PBE, sto-3g.

The calculation uses a single Γ-point SCF with sto-3g, no
smearing (MgO is insulating), so it finishes quickly even on a
laptop.

Run:

    ~/path/to/vibe-qc/.venv/bin/python examples/vibe_view/showcase_co_on_mgo.py

Output:

    output-co-on-mgo.qvf        <- open this with vibe-view
    output-co-on-mgo.out
    output-co-on-mgo.system
    output-co-on-mgo.bibtex
    output-co-on-mgo.references
    output-co-on-mgo.population.{txt,json}
    output-co-on-mgo.xyz
    output-co-on-mgo.POSCAR
    output-co-on-mgo.cif
    output-co-on-mgo.structure.xsf
    output-co-on-mgo.density.xsf

Then:

    vibe-view open output-co-on-mgo.qvf

vibe-view will open at http://127.0.0.1:8080 with the slab structure.
The AppBar reads ``vibe-view — output-co-on-mgo (CMg₂O₃) — rks/sto-3g``.

Walking the panels:

    1. structure   rotate to see the CO standing up from the flat
                   MgO layer. Mg atoms are silver, surface O red,
                   adsorbate C dark grey, adsorbate O red.
                   Spin/zoom to see the C-O bond cylinder.
    2. periodic    set Nx=Ny=2, click Apply. The 2×2 supercell
       replication  tiles the slab with adsorbates on every surface
                   Mg site — a textbook CO/MgO(001) model.
    3. density     activate the density section. Isosurface at
                   default 0.05 e/bohr³ shows the ionic charge
                   clouds. With Nx=Ny=2 you see the periodic tiling.
    4. atom_       color-coded charges: surface Mg slightly positive,
       properties  surface O negative. CO adsorbate: C slightly
                   positive, O negative (CO→surface charge transfer).
                   Switch between Mulliken and Löwdin.
    5. citations   BibTeX block ready to paste.

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

# ── MgO(001) slab geometry ──────────────────────────────────────────
# MgO rocksalt conventional lattice constant.
A_ANG = 4.211  # Å
A_BOHR = A_ANG * ANG2BOHR  # 7.957 bohr

# Vacuum gap between periodic slab images in z.
C_VAC_ANG = 18.0  # Å
C_VAC_BOHR = C_VAC_ANG * ANG2BOHR  # ~34.0 bohr

# Lattice: orthorhombic, periodic in xy, finite in z.
LATTICE_BOHR = np.array(
    [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, C_VAC_BOHR]],
    dtype=float,
)

# MgO monolayer: rocksalt (001) surface. In the 1×1 conventional cell
# there are 2 Mg and 2 O per layer, alternating in a checkerboard.
HALF = A_BOHR / 2.0  # 3.979 bohr
MG_Z = 0.0  # bottom of cell
O_Z = 0.0  # same plane as Mg for a flat monolayer

# CO adsorbate: C-down above the Mg at origin. C–Mg distance ~2.5 Å,
# C–O bond ~1.15 Å.
C_Z = 2.5 * ANG2BOHR  # ~4.72 bohr
O_ADS_Z = C_Z + 1.15 * ANG2BOHR  # ~6.90 bohr

ATOMS_BOHR = [
    # ── MgO monolayer ──
    Atom(12, [0.0, 0.0, MG_Z]),  # Mg at origin
    Atom(12, [HALF, HALF, MG_Z]),  # Mg at cell centre
    Atom(8, [HALF, 0.0, O_Z]),  # O at edge midpoint x
    Atom(8, [0.0, HALF, O_Z]),  # O at edge midpoint y
    # ── CO adsorbate (C-down above Mg at origin) ──
    Atom(6, [0.0, 0.0, C_Z]),  # C
    Atom(8, [0.0, 0.0, O_ADS_Z]),  # O (adsorbate)
]


def main() -> None:
    print("=" * 72)
    print(" vibe-view slab showcase: CO on MgO(001) / RKS-PBE / sto-3g")
    print("=" * 72)
    print()
    print("Producing output-co-on-mgo.qvf with:")
    print("  - structure (2D periodic slab, 6 atoms, mixed Mg/O/C)")
    print("  - volume.density (periodic isosurface)")
    print("  - atom_properties (Mulliken / Lowdin charges)")
    print("  - citations")
    print()

    # dim=2: xy periodic. The third lattice column is non-physical
    # bookkeeping, not a vacuum gap; jk_method defaults to "auto", which
    # routes a dim=2 slab to the vacuum-free SLAB_EWALD_2D gauge.
    system = PeriodicSystem(2, LATTICE_BOHR, ATOMS_BOHR)
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")

    run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pbe",
        output=HERE / "output-co-on-mgo",
        use_diis=True,
        damping=0.0,
        fmixing_percent=30.0,
        max_iter=80,
        conv_tol_energy=1e-7,
        write_density=True,
        density_spacing_bohr=0.3,
        write_population_file=True,
        citations=True,
        output_qvf=True,
    )

    print()
    print("=" * 72)
    print(" Done. Now open the QVF in vibe-view:")
    print()
    print(f"   vibe-view open {HERE / 'output-co-on-mgo.qvf'}")
    print()
    print(" The browser opens at http://127.0.0.1:8080. Walk the sidebar")
    print(" sections in order; the docstring of this script has a panel-")
    print(" by-panel guide of what to look at in each one.")
    print()
    print(" Things to try:")
    print("   1. Rotate to see CO standing up from the MgO layer.")
    print("   2. Set Nx=Ny=2, click Apply — 2x2 supercell with")
    print("      adsorbates on every surface Mg.")
    print("   3. Activate atom_properties, toggle color-by-charge.")
    print("   4. Activate density, lower isovalue to 0.02.")
    print("=" * 72)


if __name__ == "__main__":
    main()
