"""Tests for the surface-projected embedding potential (surface_sigma.py) —
the third 3D increment of the adsorbate embedding feature.

Validates:
1. build_surface_sigma returns an EmbeddingPotential callable
2. Zero-padding from s_ao to i_ao is correct (non-zero only in s_ao block)
3. linearize_surface_sigma produces a valid LinearizedEmbeddingPotential
4. End-to-end: sigma(z) has correct shape and retarded sign
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic_embedding.embedding_potential import (
    EmbeddingPotential,
    LinearizedEmbeddingPotential,
)
from vibeqc.periodic_embedding.region import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    RegionPartition,
)
from vibeqc.periodic_embedding.surface_sigma import (
    _zero_pad_s_to_i,
    build_surface_sigma,
    linearize_surface_sigma,
)


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
# 1. zero_pad_s_to_i
# ---------------------------------------------------------------------------


def test_zero_pad_s_to_i_simple():
    """Zero-pad: sigma_s[0,0] goes to correct position in sigma_i."""
    sys = _h_chain_system(6, spacing=2.0)  # tight → s = i
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
    ]
    region = RegionPartition.from_tags(sys, basis, tags)

    # With spacing=2.0, both region-I atoms are within margin → s_ao = i_ao.
    assert region.n_s == region.n_i
    assert region.n_s == 2

    sigma_s = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    sigma_i = _zero_pad_s_to_i(sigma_s, region)

    assert sigma_i.shape == (2, 2)
    assert sigma_i[0, 0] == 1.0
    assert sigma_i[1, 1] == 4.0


def test_zero_pad_s_to_i_subset():
    """When s_ao ⊂ i_ao, only the s_ao block is non-zero."""
    sys = _h_chain_system(6, spacing=4.0)  # wider → s ⊂ i
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
    ]
    region = RegionPartition.from_tags(sys, basis, tags)

    # With spacing=4.0, n_i=2, n_s=1.
    assert region.n_i == 2
    assert region.n_s == 1

    sigma_s = np.array([[3.0 + 0.5j]], dtype=complex)
    sigma_i = _zero_pad_s_to_i(sigma_s, region)

    assert sigma_i.shape == (2, 2)
    # The non-zero entry is at the position of the single s_ao within i_ao.
    nz = np.flatnonzero(sigma_i)
    assert len(nz) == 1
    assert sigma_i.flat[nz[0]] == pytest.approx(3.0 + 0.5j)


# ---------------------------------------------------------------------------
# 2. build_surface_sigma
# ---------------------------------------------------------------------------


def test_build_surface_sigma_returns_embedding_potential():
    sys = _h_chain_system(6, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
    ]
    region = RegionPartition.from_tags(sys, basis, tags)
    lat_opts = _default_lat_opts()

    sigma = build_surface_sigma(
        sys, basis, region, [0.0, 0.0], lat_opts=lat_opts, label="Γ"
    )

    assert isinstance(sigma, EmbeddingPotential)
    assert sigma.label == "Γ"

    # Evaluate at a complex z inside the band.
    z = -1.3 + 0.1j
    sigma_z = sigma.at(z)
    assert sigma_z.shape == (region.n_i, region.n_i)
    # Retarded: Im Σ < 0 inside the band.
    diag_im = np.imag(np.diag(sigma_z))
    assert np.all(diag_im < 0)


# ---------------------------------------------------------------------------
# 3. linearize_surface_sigma
# ---------------------------------------------------------------------------


def test_linearize_surface_sigma_basic():
    sys = _h_chain_system(6, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
    ]
    region = RegionPartition.from_tags(sys, basis, tags)
    lat_opts = _default_lat_opts()

    z0 = -1.3 + 0.5j
    lin = linearize_surface_sigma(
        sys, basis, region, [0.0, 0.0], z0, lat_opts=lat_opts, label="Γ-lin"
    )

    assert isinstance(lin, LinearizedEmbeddingPotential)
    assert lin.label == "Γ-lin"

    # Evaluate at a nearby z.
    z = z0 + 0.01
    sigma_z = lin.at(z)
    assert sigma_z.shape == (region.n_i, region.n_i)

    # Compare to the full (non-linearized) sigma at slightly different z.
    full = build_surface_sigma(sys, basis, region, [0.0, 0.0], lat_opts=lat_opts)
    sigma_full = full.at(z)

    # Linearization should be close (second-order error).
    diff = np.max(np.abs(sigma_z - sigma_full))
    assert diff < 1e-2


def test_linearize_surface_sigma_second_order():
    """Linearization error scales as O(dz²)."""
    sys = _h_chain_system(6, spacing=2.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
    ]
    region = RegionPartition.from_tags(sys, basis, tags)
    lat_opts = _default_lat_opts()

    z0 = -1.3 + 0.5j
    lin = linearize_surface_sigma(sys, basis, region, [0.0, 0.0], z0, lat_opts=lat_opts)
    full = build_surface_sigma(sys, basis, region, [0.0, 0.0], lat_opts=lat_opts)

    def err(dz):
        return np.max(np.abs(lin.at(z0 + dz) - full.at(z0 + dz)))

    e_small = err(0.01)
    e_big = err(0.02)
    # Doubling dz should ~ quadruple the error (second order).
    assert e_big / e_small == pytest.approx(4.0, rel=0.5)
