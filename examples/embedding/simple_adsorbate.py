#!/usr/bin/env python3
"""Example: Green's-function embedding for an adsorbate on a periodic slab.

Demonstrates the complete Layer A + Layer B pipeline:
  1. Build a 2D periodic H-atom slab
  2. Tag substrate + region I atoms
  3. Run Layer A (embedded surface density)
  4. Extract the clean-surface Green function G0
  5. Compute the adsorption energy via Layer B (Lloyd's formula)

Usage:
    .venv/bin/python examples/embedding/simple_adsorbate.py
"""

from __future__ import annotations

import numpy as np
from vibeqc._vibeqc_core import Atom, BasisSet, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic_embedding import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    compute_layer_b_mock_delta_v,
    run_embedded_surface,
)
from vibeqc.periodic_embedding.scf2step import compute_region_i_gf_at_kz
from vibeqc.periodic_embedding.surface_sigma import build_surface_sigma


def build_h_slab(n_layers: int = 6, spacing: float = 2.0) -> PeriodicSystem:
    """Build a 2D-periodic H-atom slab (one atom per layer)."""
    a_lat = 10.0
    vacuum = 30.0
    z_slab = (n_layers - 1) * spacing
    c_vac = z_slab + vacuum
    lat = np.array([[a_lat, 0, 0], [0, a_lat, 0], [0, 0, c_vac]])
    shift_z = (c_vac - z_slab) / 2.0
    atoms = [
        Atom(1, [a_lat / 2, a_lat / 2, shift_z + i * spacing]) for i in range(n_layers)
    ]
    mult = 1 if n_layers % 2 == 0 else 2
    return PeriodicSystem(dim=2, lattice=lat, unit_cell=atoms, multiplicity=mult)


def main():
    print("=" * 60)
    print(" Green's-function embedding — adsorbate on H slab")
    print("=" * 60)

    n_layers = 6
    system = build_h_slab(n_layers=n_layers)
    mol = system.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * (n_layers - 2) + [TAG_REGION_I] * 2
    print(f"\nSystem: {n_layers}-layer H slab, dim={system.dim}")
    print(f"Basis: sto-3g, {basis.nbasis} AOs")
    print(f"Tags: {tags}")

    # --- Layer A: clean surface ---
    print("\n--- Layer A: embedded surface density ---")
    result_a = run_embedded_surface(
        system,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=40,
        e_bottom=-2.2,
        e_fermi=-1.0,
        scf_method="one_shot",
    )
    print(f"  Region I AOs: {result_a.n_i}")
    print(f"  Occupation:   {result_a.occupation:.6f} e-")
    print(f"  Band energy:  {result_a.band_energy:.6f} Ha")
    print(f"  Density diag: {np.diag(result_a.density_local)}")

    # --- Build G0 at Gamma ---
    region = result_a.region
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 15.0
    lat_opts.nuclear_cutoff_bohr = 15.0

    sigma = build_surface_sigma(system, basis, region, [0, 0], lat_opts=lat_opts)

    def g0_fn(z):
        return compute_region_i_gf_at_kz(
            system, basis, region, np.zeros(3), z, sigma.at(z), lat_opts=lat_opts
        )

    # --- Layer B: adsorbate perturbation ---
    print("\n--- Layer B: adsorption energy via Lloyd ---")
    delta_v = 0.3  # Ha, weak repulsive potential
    dv_pos = region.n_i - 1  # top region-I AO
    result_b = compute_layer_b_mock_delta_v(
        g0_fn,
        np.array([delta_v]),
        dv_loc=np.array([dv_pos]),
        e_range=(-3.0, 0.0),
        n_energies=3000,
        eta=0.02,
        e_fermi=-1.0,
    )
    print(f"  Delta_V = {delta_v:.2f} Ha on AO {dv_pos}")
    print(f"  Friedel sum (Delta_N):    {result_b.delta_n_total:.6f}")
    print(f"  Band energy change:      {result_b.band_energy_change:.6f} Ha")
    print(f"  |Delta_N_max|:            {np.max(np.abs(result_b.delta_n)):.4f}")

    print("\n" + "=" * 60)
    print(" Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
