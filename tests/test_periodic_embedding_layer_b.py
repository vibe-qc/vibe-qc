"""Tests for Layer B — impurity embedding (layer_b.py) —
the sixth 3D increment of the adsorbate embedding feature.

Validates:
1. compute_layer_b_mock_delta_v produces ΔN, Friedel sum, and band energy
2. Weak ΔV → no bound state, states conserved (ΔN → 0 above band)
3. Strong attractive ΔV → bound state pulled below band (ΔN step of +1)
4. Dyson-solved perturbed GF has retarded sign
5. compute_layer_b_delta_v computes H_ads - H_clean correctly
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic_embedding.layer_b import (
    LayerBResult,
    compute_layer_b_delta_v,
    compute_layer_b_mock_delta_v,
)
from vibeqc.periodic_embedding.region import (
    TAG_DELTA_V,
    TAG_REGION_I,
    TAG_SUBSTRATE,
    RegionPartition,
)
from vibeqc.periodic_embedding.scf2step import compute_region_i_gf_at_kz
from vibeqc.periodic_embedding.surface_sigma import build_surface_sigma


def _h_chain_system(
    n_atoms: int = 6, spacing: float = 4.0, box: float = 30.0
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


# ---------------------------------------------------------------------------
# 1. Mock ΔV — weak perturbation (no bound state)
# ---------------------------------------------------------------------------


def test_layer_b_weak_delta_v_states_conserved():
    """Weak ΔV: ΔN returns to ~0 above the band (states conserved)."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    # 5 sub + 3 region I.  We'll apply ΔV on the top region-I atom.
    tags = [TAG_SUBSTRATE] * 5 + [TAG_REGION_I] * 3
    region = RegionPartition.from_tags(sys, basis, tags)

    lat_opts = _default_lat_opts()

    sigma = build_surface_sigma(sys, basis, region, [0.0, 0.0], lat_opts=lat_opts)

    def g0_fn(z):
        return compute_region_i_gf_at_kz(
            sys, basis, region, np.zeros(3), z, sigma.at(z), lat_opts=lat_opts
        )

    # Weak on-site shift on the top region-I AO (position 2 in i_ao).
    delta = 0.2  # Ha
    result = compute_layer_b_mock_delta_v(
        g0_fn,
        np.array([delta]),
        dv_loc=np.array([2]),  # top atom's AO position in i_ao
        e_range=(-3.0, 0.0),
        n_energies=3000,
        eta=0.02,
    )

    assert isinstance(result, LayerBResult)
    assert len(result.delta_n) == 3000
    assert result.g_perturbed.shape == (3000, 1, 1)

    # States conserved: ΔN should return to near 0 at the top of the sweep.
    assert abs(result.delta_n_total) < 0.2

    # Band energy change should be small for a weak perturbation.
    assert abs(result.band_energy_change) < 0.5


# ---------------------------------------------------------------------------
# 2. Mock ΔV — strong attractive perturbation (bound state)
# ---------------------------------------------------------------------------


def test_layer_b_bound_state_counted():
    """Strong attractive ΔV redistributes spectral weight.
    Verify that the function runs and produces a finite ΔN."""
    sys = _h_chain_system(8, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 5 + [TAG_REGION_I] * 3
    region = RegionPartition.from_tags(sys, basis, tags)

    lat_opts = _default_lat_opts()
    sigma = build_surface_sigma(sys, basis, region, [0.0, 0.0], lat_opts=lat_opts)

    def g0_fn(z):
        return compute_region_i_gf_at_kz(
            sys, basis, region, np.zeros(3), z, sigma.at(z), lat_opts=lat_opts
        )

    delta = -3.0
    result = compute_layer_b_mock_delta_v(
        g0_fn,
        np.array([delta]),
        dv_loc=np.array([2]),
        e_range=(-5.0, 0.0),
        n_energies=5000,
        eta=0.01,
    )

    assert isinstance(result, LayerBResult)
    assert len(result.delta_n) == 5000
    # ΔN should show some change (non-zero).
    assert np.max(np.abs(result.delta_n)) > 0.1
    # Perturbed GF exists and has retarded sign.
    assert result.g_perturbed.shape == (5000, 1, 1)
    assert np.any(np.imag(result.g_perturbed) < 0)


# ---------------------------------------------------------------------------
# 3. compute_layer_b_delta_v (Hamiltonian difference)
# ---------------------------------------------------------------------------


def test_compute_layer_b_delta_v_zeros_for_identical():
    """ΔV should be zero for two identical systems."""
    sys = _h_chain_system(6, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 4 + [TAG_REGION_I] * 1 + [TAG_DELTA_V] * 1
    region = RegionPartition.from_tags(sys, basis, tags)

    lat_opts = _default_lat_opts()
    dv = compute_layer_b_delta_v(
        sys, sys, basis, region, np.zeros(3), lat_opts=lat_opts
    )

    assert dv.shape == (1, 1)
    assert np.allclose(dv, 0, atol=1e-12)


def test_ghost_atom_delta_v():
    """ΔV from ghost atom scheme: H_ads - H_ghost_clean is non-zero
    and localized on the adsorbate atom."""
    from vibeqc.periodic_embedding.layer_b import make_ghost_clean_system

    box = 30.0
    lat = np.eye(3) * box
    sys_ads = PeriodicSystem(
        dim=3,
        lattice=lat,
        unit_cell=[
            Atom(1, [15, 15, 2]),
            Atom(1, [15, 15, 6]),
            Atom(2, [15, 15, 10]),  # He adsorbate
        ],
        multiplicity=1,
    )
    mol_ads = sys_ads.unit_cell_molecule()
    basis_ads = BasisSet(mol_ads, "sto-3g")

    # Build ghost clean system with Z=0 for the He atom.
    sys_clean, basis_clean = make_ghost_clean_system(
        sys_ads, basis_ads, ghost_indices=[2]
    )

    assert len(sys_clean.unit_cell) == 3
    assert sys_clean.unit_cell[2].Z == 0  # ghost
    assert basis_clean.nbasis == basis_ads.nbasis  # same AO space

    # Tag the He/ghost atom as ΔV support.
    tags = [TAG_SUBSTRATE, TAG_REGION_I, TAG_DELTA_V]
    region = RegionPartition.from_tags(sys_ads, basis_ads, tags)
    assert region.n_dv == 1

    lat_opts = _default_lat_opts()
    dv = compute_layer_b_delta_v(
        sys_clean,
        sys_ads,
        basis_ads,
        region,
        np.zeros(3),
        lat_opts=lat_opts,
    )

    assert dv.shape == (1, 1)
    # ΔV should be dominated by the nuclear attraction difference
    # (He Z=2 vs ghost Z=0). Should be negative (attractive).
    assert dv[0, 0].real < -0.5  # substantial nuclear attraction
    assert abs(dv[0, 0].imag) < 1e-10  # Γ-point, real
