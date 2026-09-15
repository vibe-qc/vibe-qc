"""Generate crystal-lattice visualisation SVGs for the docs.

Renders representative materials for every common Bravais-family
lattice covered in docs/user_guide/crystal_lattices.md.  Outputs land
under docs/_static/lattices/<slug>.svg and are committed as static
documentation artifacts (re-run only when the lattice catalogue or
visual style changes).

Usage::

    .venv/bin/python scripts/docs/gen_lattice_figures.py

Requires ASE + matplotlib; both are vibe-qc dev-time dependencies.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import bulk, graphene
from ase.spacegroup import crystal
from ase.visualize.plot import plot_atoms

OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "_static" / "lattices"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _render(atoms: Atoms, slug: str, *, title: str, rep: tuple[int, int, int] = (2, 2, 2),
            rotation: str = "30x,-15y", radii: float = 0.5) -> None:
    """Plot one lattice; save SVG under docs/_static/lattices/."""
    super_atoms = atoms.repeat(rep)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_atoms(super_atoms, ax, radii=radii, rotation=rotation,
               show_unit_cell=2)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{slug}.svg", bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    print(f"wrote {slug}.svg")


def main() -> None:
    # ---- Cubic family ---------------------------------------------------

    # Simple cubic — α-Po. Textbook Bravais lattice.
    _render(bulk("Po", "sc", a=3.359), slug="cubic-simple-Po",
            title="Simple cubic (cP) — α-Po")

    # BCC — α-Fe (canonical metallic example).
    _render(bulk("Fe", "bcc", a=2.866), slug="cubic-bcc-Fe",
            title="Body-centered cubic (cI) — α-Fe (metal)")

    # FCC — Cu (canonical FCC metal).
    _render(bulk("Cu", "fcc", a=3.615), slug="cubic-fcc-Cu",
            title="Face-centered cubic (cF) — Cu (metal)")

    # Diamond / zincblende — Si (semiconductor).
    _render(bulk("Si", "diamond", a=5.431), slug="cubic-diamond-Si",
            title="Diamond / cF — Si (semiconductor)")

    # Rocksalt — NaCl (ionic).
    _render(bulk("NaCl", "rocksalt", a=5.640), slug="cubic-rocksalt-NaCl",
            title="Rocksalt — NaCl (ionic)")

    # Rocksalt — MgO (ionic, vibe-qc canonical periodic-SCF target).
    _render(bulk("MgO", "rocksalt", a=4.211), slug="cubic-rocksalt-MgO",
            title="Rocksalt — MgO (ionic; v0.8.0 canonical)")

    # CsCl (ionic, simple-cubic with 2-atom basis).
    cscl = crystal(symbols=["Cs", "Cl"],
                   basis=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
                   spacegroup=221, cellpar=[4.119, 4.119, 4.119, 90, 90, 90])
    _render(cscl, slug="cubic-cscl-CsCl", rep=(2, 2, 2),
            title="CsCl-type (cP + 2-atom basis) — CsCl (ionic)")

    # Cubic perovskite — SrTiO3 (ionic; ABX3 prototype).
    srtio3 = crystal(symbols=["Sr", "Ti", "O"],
                     basis=[(0.0, 0.0, 0.0),
                            (0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.0)],
                     spacegroup=221, cellpar=[3.905, 3.905, 3.905, 90, 90, 90])
    _render(srtio3, slug="cubic-perovskite-SrTiO3", rep=(2, 2, 2),
            title="Cubic perovskite — SrTiO₃ (ionic ABX₃)")

    # ---- Hexagonal / trigonal family -----------------------------------

    # HCP — Mg (canonical metal example).
    _render(bulk("Mg", "hcp", a=3.209, c=5.211),
            slug="hex-hcp-Mg", rep=(3, 3, 2),
            title="HCP (hP + 2-atom basis) — Mg (metal)")

    # Wurtzite — ZnO (ionic / wide-gap semiconductor).
    _render(bulk("ZnO", "wurtzite", a=3.250, c=5.207),
            slug="hex-wurtzite-ZnO", rep=(2, 2, 2),
            title="Wurtzite — ZnO (ionic / wide-gap semi)")

    # Trigonal corundum — α-Al2O3.
    al2o3 = crystal(symbols=["Al", "O"],
                    basis=[(0.0, 0.0, 0.352),
                           (0.306, 0.0, 0.25)],
                    spacegroup=167,
                    cellpar=[4.759, 4.759, 12.991, 90, 90, 120])
    _render(al2o3, slug="trig-corundum-Al2O3", rep=(2, 2, 1),
            title="Corundum (hR / R-3c) — α-Al₂O₃ (ionic insulator)")

    # Hexagonal α-quartz — SiO2.
    sio2 = crystal(symbols=["Si", "O"],
                   basis=[(0.4699, 0.0, 0.0),
                          (0.4145, 0.2662, 0.1184)],
                   spacegroup=154,
                   cellpar=[4.916, 4.916, 5.405, 90, 90, 120])
    _render(sio2, slug="trig-quartz-SiO2", rep=(2, 2, 1),
            title="α-Quartz (hP / P3₂21) — SiO₂ (insulator)")

    # Graphene — 2D hexagonal (visualised in a slab).
    grph = graphene(formula="C2", a=2.46, size=(2, 2, 1), vacuum=8.0)
    _render(grph, slug="hex-graphene-C", rep=(1, 1, 1),
            title="Graphene (hP 2D) — C (semimetal)", rotation="0x,0y")

    # ---- Tetragonal family ---------------------------------------------

    # Rutile — TiO2 (ionic / wide-gap insulator).
    tio2 = crystal(symbols=["Ti", "O"],
                   basis=[(0.0, 0.0, 0.0), (0.305, 0.305, 0.0)],
                   spacegroup=136, cellpar=[4.594, 4.594, 2.959, 90, 90, 90])
    _render(tio2, slug="tet-rutile-TiO2", rep=(2, 2, 3),
            title="Rutile (tP / P4₂/mnm) — TiO₂ (ionic insulator)")


if __name__ == "__main__":
    main()
