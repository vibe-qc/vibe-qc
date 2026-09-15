#!/usr/bin/env python3
"""Example: SCF-level embedding potential + numerical forces (EXPERIMENTAL).

A closed-shell H-chain "surface" (bottom three atoms = substrate, top atom =
region I) demonstrating two features of the surface-embedding driver:

  1. ``sigma_scf=True`` — build the embedding potential ``Sigma_emb`` from the
     **SCF substrate Fock** instead of the bare Hcore (Ishida two-step,
     step 1). The substrate bands shift up by the mean-field potential and
     the complex-energy contour window auto-tracks them.
  2. ``EmbeddedSurfaceCalculator`` (ASE) — energy (eV) + **numerical forces**
     (eV/A), so any ``ase.optimize`` relaxer can drive the region-I geometry.

Companion examples: ``simple_adsorbate.py`` (Layer A + Layer B / Lloyd
adsorption energy) and ``open_shell_adsorbate.py`` (spin-polarised path).

The method is experimental and not yet validated to thick-slab GDF parity
(see ``docs/user_guide/surface_embedding.md``); treat the numbers as
qualitative.

Usage:
    .venv/bin/python examples/embedding/two_step_scf_and_forces.py
"""

from __future__ import annotations

import warnings

import numpy as np
from vibeqc._vibeqc_core import Atom, BasisSet, PeriodicSystem
from vibeqc.periodic_embedding import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    EmbeddedSurfaceExperimentalWarning,
    run_embedded_surface,
)

# Experimental method: silence the per-run warning for clean output
# (cf. the GAPW examples filtering GAPWExperimentalWarning).
warnings.simplefilter("ignore", EmbeddedSurfaceExperimentalWarning)

# --- A 4-atom H chain normal to the surface (vacuum along c) -----------
A_LAT, VAC, SPACING = 10.0, 30.0, 2.0  # bohr
lattice = np.array([[A_LAT, 0, 0], [0, A_LAT, 0], [0, 0, VAC]])
chain = [Atom(1, [A_LAT / 2, A_LAT / 2, 5.0 + i * SPACING]) for i in range(4)]
system = PeriodicSystem(dim=3, lattice=lattice, unit_cell=chain, multiplicity=1)
basis = BasisSet(system.unit_cell_molecule(), "sto-3g")

# Tags: bottom 3 atoms = semi-infinite substrate, top atom = region I.
tags = [TAG_SUBSTRATE] * 3 + [TAG_REGION_I]

# --- 1. One-shot embedding (Hcore Sigma_emb) ---------------------------
res_hcore = run_embedded_surface(
    system, basis, tags, surface_k_mesh=(1, 1), contour_n_nodes=40
)
print("1. One-shot embedding (Hcore Sigma_emb)")
print(f"     region-I occupation = {res_hcore.occupation:.4f} e-")
print(f"     band-structure E    = {res_hcore.band_energy:.4f} Ha")
print(f"     contour window      = [{res_hcore.e_bottom:.2f}, {res_hcore.e_fermi:.2f}] Ha")

# --- 2. SCF-level Sigma_emb (Ishida two-step, step 1) ------------------
res_scf = run_embedded_surface(
    system, basis, tags, surface_k_mesh=(1, 1), contour_n_nodes=40,
    sigma_scf=True,
)
print("\n2. SCF-level Sigma_emb (sigma_scf=True, Ishida step 1)")
print(f"     region-I occupation = {res_scf.occupation:.4f} e-")
print(f"     band-structure E    = {res_scf.band_energy:.4f} Ha")
print(f"     contour window      = [{res_scf.e_bottom:.2f}, {res_scf.e_fermi:.2f}] Ha")
print(f"     (shifted up ~{res_scf.e_fermi - res_hcore.e_fermi:.1f} Ha by the substrate mean field)")

# --- 3. ASE calculator: energy + numerical forces ----------------------
try:
    from ase import Atoms
    from ase.units import Bohr

    from vibeqc.periodic_embedding import EmbeddedSurfaceCalculator
except ImportError:
    print("\n3. [ase] extra not installed - skipping the ASE calculator demo.")
else:
    cell_ang = lattice * Bohr
    pos_ang = np.array([a.xyz for a in chain]) * Bohr
    ase_atoms = Atoms("H4", positions=pos_ang, cell=cell_ang, pbc=True)
    ase_atoms.set_tags(tags)

    # force_atom_indices=[3]: only the region-I atom is mobile, so the
    # numerical-force loop costs 6 embedded runs, not 6*N.
    ase_atoms.calc = EmbeddedSurfaceCalculator(
        basis_name="sto-3g", surface_k_mesh=(1, 1), contour_n_nodes=24,
        force_atom_indices=[3],
    )
    energy = ase_atoms.get_potential_energy()  # eV
    forces = ase_atoms.get_forces()  # eV / Ang
    print("\n3. ASE EmbeddedSurfaceCalculator (energy + numerical forces)")
    print(f"     total energy        = {energy:.4f} eV")
    print(f"     force on region-I z = {forces[3, 2]:+.4f} eV/Ang")
    print("     any ase.optimize relaxer can now drive the region-I geometry.")

# --- light self-checks (qualitative; the method is experimental) -------
assert res_hcore.occupation > 0.0
assert res_scf.occupation > 0.0
# The SCF mean field pushes the substrate bands up relative to Hcore.
assert res_scf.e_fermi > res_hcore.e_fermi
print("\nOK (qualitative checks passed).")
