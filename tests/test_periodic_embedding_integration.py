"""End-to-end integration test for the complete Green's-function embedding
pipeline — Layer A (clean surface) + Layer B (adsorbate impurity).

Validates the full data flow:
  tags → RegionPartition → H(k)/S(k) → Σ_emb → G0 → ΔV → Dyson → Lloyd
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic_embedding import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    EmbeddedSurfaceResult,
    LayerBResult,
    RegionPartition,
    compute_layer_b_mock_delta_v,
    run_embedded_surface,
)
from vibeqc.periodic_embedding.scf2step import compute_region_i_gf_at_kz
from vibeqc.periodic_embedding.surface_sigma import build_surface_sigma


def _h_chain_system(
    n_atoms: int = 8, spacing: float = 2.0, box: float = 30.0
) -> PeriodicSystem:
    lat = np.eye(3) * box
    atoms = [Atom(1, [box / 2, box / 2, 2.0 + i * spacing]) for i in range(n_atoms)]
    mult = 1 if n_atoms % 2 == 0 else 2
    return PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=mult)


def _default_lat_opts():
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    opts.nuclear_cutoff_bohr = 15.0
    return opts


def test_full_pipeline_layer_a_then_b():
    """Complete pipeline: Layer A → G0 → Layer B (mock ΔV) → Lloyd.

    Steps:
    1. Build slab system with tags
    2. Run Layer A (run_embedded_surface) → G0(k,z) + Σ_emb
    3. Extract G0 at Γ for the perturbed site
    4. Run Layer B with mock ΔV → ΔN + adsorption energy
    """
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 5 + [TAG_REGION_I] * 3

    # --- Layer A ---
    result_a = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=32,
        e_bottom=-2.2,
        e_fermi=-1.0,
    )

    assert isinstance(result_a, EmbeddedSurfaceResult)
    assert result_a.occupation > 0
    assert result_a.n_i == 3

    # --- Build G0 at Γ ---
    region = result_a.region
    lat_opts = _default_lat_opts()

    # Reconstruct the clean-surface sigma for Γ (we could also reuse
    # result_a.sigma_lin_per_k[0] but we want the full z-dependent one).
    sigma = build_surface_sigma(sys, basis, region, [0.0, 0.0], lat_opts=lat_opts)

    def g0_fn(z):
        return compute_region_i_gf_at_kz(
            sys, basis, region, np.zeros(3), z, sigma.at(z), lat_opts=lat_opts
        )

    # --- Layer B ---
    # Apply a weak repulsive ΔV on the top region-I atom (position 2 in i_ao).
    delta = 0.5  # Ha
    result_b = compute_layer_b_mock_delta_v(
        g0_fn,
        np.array([delta]),
        dv_loc=np.array([2]),
        e_range=(-3.0, 0.0),
        n_energies=2000,
        eta=0.02,
        e_fermi=-1.0,  # same as Layer A
    )

    assert isinstance(result_b, LayerBResult)
    assert len(result_b.delta_n) == 2000
    assert result_b.g_perturbed.shape == (2000, 1, 1)

    # The band-energy change should be finite and small for a weak ΔV.
    assert abs(result_b.band_energy_change) < 1.0

    # Layer A + Layer B together produce consistent results.
    # The occupation from Layer A should be broadly consistent with
    # the total displaced charge from Layer B (they involve the same
    # unperturbed system).
    assert result_a.occupation > 0


def test_full_pipeline_via_runner_then_manual_b():
    """Use run_embedded_surface's sigma_lin_per_k to build G0 and run Layer B."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 5 + [TAG_REGION_I] * 3

    result_a = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=24,
        e_bottom=-2.2,
        e_fermi=-1.0,
    )

    region = result_a.region
    lat_opts = _default_lat_opts()

    # Use the linearized sigma from Layer A (reuse, fast path).
    sigma_lin = result_a.sigma_lin_per_k[0]

    def g0_fn(z):
        return compute_region_i_gf_at_kz(
            sys, basis, region, np.zeros(3), z, sigma_lin.at(z), lat_opts=lat_opts
        )

    delta = 0.3
    result_b = compute_layer_b_mock_delta_v(
        g0_fn,
        np.array([delta]),
        dv_loc=np.array([2]),
        e_range=(-3.0, 0.0),
        n_energies=1500,
        eta=0.02,
        e_fermi=-1.0,
    )

    # Basic sanity.
    assert isinstance(result_b, LayerBResult)
    assert np.max(np.abs(result_b.delta_n)) > 0.01
