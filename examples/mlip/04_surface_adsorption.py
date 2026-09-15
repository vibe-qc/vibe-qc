"""Adsorption of water and ammonia on oxide + metal surfaces with MACE.

Requires the ``[mace]`` extra (Python <= 3.13) and ASE:

    pip install 'vibe-qc[mace]'

Run:

    python examples/mlip/04_surface_adsorption.py

For each surface this builds an ASE slab, relaxes the bare slab and the
slab + adsorbate with the MIT MACE-MPA-0 materials model (bottom layer
fixed), and prints the adsorption energy

    E_ads = E(slab + molecule) - E(slab) - E(molecule_gas)    [eV]

evaluated with the same model so the per-element reference energies
cancel. The materials model covers all 89 elements used here.

NB illustrative settings (thin slabs, loose fmax, a single adsorption
site) — converge slab thickness, supercell size, the site, and fmax for
production. The printed values are model-internal probes, not reference
adsorption energies or accuracy evidence. See docs/tutorial/mace_mlip.md.
"""

import numpy as np
from ase.build import add_adsorbate, bulk, fcc111, molecule, surface
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.spacegroup import crystal

from vibeqc.mlip.mace import mace_calculator

CALC = mace_calculator()  # MACE-MPA-0 (MIT materials, default)
_CATIONS = {12, 20, 13, 22, 78, 29}  # Mg Ca Al Ti Pt Cu


def energy(atoms, relax=True):
    a = atoms.copy()
    a.calc = CALC
    if relax:
        if a.cell.rank == 3 and a.pbc.any():
            z = a.positions[:, 2]
            a.set_constraint(FixAtoms(mask=z < z.min() + 1.2))
        BFGS(a, logfile=None).run(fmax=0.15, steps=80)
    return a.get_potential_energy()  # eV


def gas(name):
    m = molecule(name)
    m.set_cell([14.0, 14.0, 14.0])
    m.center()
    m.pbc = True
    return energy(m)


def make_slab(name):
    if name == "MgO(100)":
        return surface(bulk("MgO", "rocksalt", a=4.212), (1, 0, 0), 3, vacuum=8.0) * (2, 2, 1)
    if name == "CaO(100)":
        return surface(bulk("CaO", "rocksalt", a=4.811), (1, 0, 0), 3, vacuum=8.0) * (2, 2, 1)
    if name == "Al2O3(0001)":
        c = crystal(["Al", "O"], basis=[(0, 0, 0.35216), (0.69365, 0, 0.25)],
                    spacegroup=167, cellpar=[4.7607, 4.7607, 12.9947, 90, 90, 120])
        return surface(c, (0, 0, 1), 2, vacuum=8.0)
    if name == "TiO2(110)":
        c = crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3053, 0.3053, 0)],
                    spacegroup=136, cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90])
        return surface(c, (1, 1, 0), 3, vacuum=8.0) * (2, 1, 1)
    if name == "Pt(111)":
        return fcc111("Pt", size=(3, 3, 3), vacuum=8.0)
    if name == "Cu(111)":
        return fcc111("Cu", size=(3, 3, 3), vacuum=8.0)
    raise ValueError(name)


def adsorption_energy(slab, e_slab, ads_name, e_gas):
    sys = slab.copy()
    cat = [i for i, z in enumerate(sys.numbers) if z in _CATIONS]
    top = max(cat, key=lambda i: sys.positions[i, 2])
    add_adsorbate(sys, molecule(ads_name), height=2.1,
                  position=sys.positions[top, :2], mol_index=0)
    return energy(sys) - e_slab - e_gas


if __name__ == "__main__":
    print("# model=medium-mpa-0 loader=mace_mp license=MIT")
    print("# exploratory protocol; no external reference is included")
    e_h2o, e_nh3 = gas("H2O"), gas("NH3")
    print(f"{'surface':13s} {'E_ads(H2O)/eV':>14s} {'E_ads(NH3)/eV':>14s}")
    for name in ["MgO(100)", "CaO(100)", "Al2O3(0001)", "TiO2(110)", "Pt(111)", "Cu(111)"]:
        slab = make_slab(name)
        e_slab = energy(slab)
        ew = adsorption_energy(slab, e_slab, "H2O", e_h2o)
        en = adsorption_energy(slab, e_slab, "NH3", e_nh3)
        print(f"{name:13s} {ew:14.3f} {en:14.3f}")
