#!/usr/bin/env python3
"""Periodic MSINDO via the Cyclic Cluster Model (CCM): bulk, surface, adsorption.

The Cyclic Cluster Model (Bredow, Geudtner & Jug, *J. Comput. Chem.* **22**, 89
(2001); Peintinger & Bredow, *J. Comput. Chem.* **35**, 839 (2014)) treats a
crystal as a finite *cyclic* cluster (a supercell) in which every atom carries a
Wigner-Seitz cell describing its periodic environment, with the long-range
electrostatics added by a classical Ewald (Madelung) embedding.  vibe-qc's
re-implementation lives in ``vibeqc.semiempirical.methods.msindo_ccm`` and is
validated out-of-process against reference MSINDO (CLAUDE.md § 10) — every number
below reproduces the oracle to < 1 µHa.

``run_ccm(atomic_numbers, coords_angstrom, translations_angstrom, *, madelung=)``
takes the cluster geometry (Angstrom) and the 1–3 supercell lattice vectors;
``madelung=True`` switches on the Ewald embedding (1-D finite sum / 2-D surface /
3-D bulk, chosen by the number of lattice vectors).

This script demonstrates:
  * bulk MgO        — rocksalt Mg₄O₄ cell, CCM3D + Ewald Madelung
  * the MgO(100)    — a 2-D periodic slab, CCM2D + surface Ewald
    surface
  * H₂O adsorption  — a periodic water adlayer on MgO(100); the adsorption
                      energy E[slab+H₂O] − E[slab] − E[H₂O(gas)]

Run:

    .venv/bin/python examples/semiempirical/11_msindo_ccm.py
"""

from __future__ import annotations

import numpy as np

from vibeqc.semiempirical.methods import msindo
from vibeqc.semiempirical.methods.msindo_ccm import (
    ccm_gradient_fd,
    ccm_optimize,
    run_ccm,
)

HARTREE_TO_KCAL = 627.509474

_Z = {"H": 1, "O": 8, "Mg": 12}


def _zc(atoms):
    return [_Z[s] for s, *_ in atoms], [[x, y, z] for _s, x, y, z in atoms]


def rocksalt(cation: str, anion: str, a: float):
    """Conventional rocksalt cubic cell (4 cation + 4 anion), lattice ``a``."""
    fcc = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    atoms = [(cation, fx * a, fy * a, fz * a) for fx, fy, fz in fcc]
    atoms += [(anion, (fx + 0.5) * a, fy * a, fz * a) for fx, fy, fz in fcc]
    return atoms, [(a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)]


def mgo_100_slab(a: float):
    """MgO(100) bilayer slab, periodic in the surface plane."""
    atoms = [("Mg", 0, 0, 0), ("O", a, 0, 0), ("O", 0, a, 0), ("Mg", a, a, 0),
             ("O", 0, 0, a), ("Mg", a, 0, a), ("Mg", 0, a, a), ("O", a, a, a)]
    return atoms, [(2 * a, 0.0, 0.0), (0.0, 2 * a, 0.0)]


def main() -> None:
    print("Periodic MSINDO — Cyclic Cluster Model")
    print("=" * 60)

    # --- bulk MgO (rocksalt, 3-D Ewald Madelung) ---------------------------
    atoms, trans = rocksalt("Mg", "O", 4.21)
    Z, C = _zc(atoms)
    bulk = run_ccm(Z, C, trans, madelung=True)
    print(f"\nbulk MgO (rocksalt, CCM3D)      E = {bulk.total_energy:.10f} Ha"
          f"   ({bulk.n_iter} cycles)")

    # --- MgO(100) surface (2-D Ewald Madelung) -----------------------------
    slab_atoms, slab_trans = mgo_100_slab(2.105)
    Zs, Cs = _zc(slab_atoms)
    slab = run_ccm(Zs, Cs, slab_trans, madelung=True)
    print(f"MgO(100) slab (CCM2D)           E = {slab.total_energy:.10f} Ha"
          f"   ({slab.n_iter} cycles)")

    # --- H2O adsorbed on MgO(100) (periodic adlayer) -----------------------
    water = [("O", 2.105, 0.0, 2.105 + 2.2),
             ("H", 2.105 + 0.76, 0.0, 2.105 + 2.76),
             ("H", 2.105 - 0.76, 0.0, 2.105 + 2.76)]
    Zw, Cw = _zc(slab_atoms + water)
    sysw = run_ccm(Zw, Cw, slab_trans, madelung=True)
    print(f"MgO(100) + H2O adlayer (CCM2D)  E = {sysw.total_energy:.10f} Ha"
          f"   ({sysw.n_iter} cycles)")

    # gas-phase water (molecular MSINDO)
    gas = msindo.run_msindo(
        [8, 1, 1],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    )
    e_ads = sysw.total_energy - slab.total_energy - gas.total_energy
    print(f"\nH2O / MgO(100) adsorption energy = {e_ads:.6f} Ha"
          f"  = {e_ads * HARTREE_TO_KCAL:.2f} kcal/mol  (fixed geometry)")

    # CCM nuclear gradient (Ha/bohr) — the force on the adsorbate is what a
    # geometry optimiser follows to relax the binding site.
    grad = ccm_gradient_fd(Zw, Cw, slab_trans, madelung=True)
    fmax_water = float(np.max(np.abs(grad[len(slab_atoms):])))
    print(f"max |force| on the H2O atoms     = {fmax_water:.5f} Ha/bohr")

    # Relax the adsorbate on the (frozen) slab → the relaxed adsorption energy.
    print("\nrelaxing the H2O adsorbate (frozen slab) ...")
    _, relaxed = ccm_optimize(Zw, Cw, slab_trans, madelung=True,
                              frozen=list(range(len(slab_atoms))), fmax=2e-3)
    e_ads_rel = relaxed.total_energy - slab.total_energy - gas.total_energy
    print(f"relaxed adsorption energy        = {e_ads_rel:.6f} Ha"
          f"  = {e_ads_rel * HARTREE_TO_KCAL:.2f} kcal/mol")

    print("\nCite: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014).")


if __name__ == "__main__":
    main()
