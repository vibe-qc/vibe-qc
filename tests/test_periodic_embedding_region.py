"""Tests for the region partition module (region.py) — the first 3D
increment of the adsorbate Green's-function embedding feature.

Validates:
1. AO index mapping from atom tags via BasisSet.shells()
2. Dividing-plane S geometry (midpoint when substrate is below, offset otherwise)
3. s_ao selection (all region-I AOs for thin region I, subset for thick)
4. Error handling: invalid tags, wrong length, no region I
5. Tag-only atoms with zero AOs in the basis (e.g. ghost atoms for plane markers)
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, PeriodicSystem
from vibeqc.periodic_embedding.region import (
    TAG_DELTA_V,
    TAG_REGION_I,
    TAG_SUBSTRATE,
    RegionPartition,
)


def _h_chain_system(
    n_atoms: int = 4, spacing: float = 4.0, box: float = 30.0
) -> PeriodicSystem:
    """A vertical H-atom chain in a cubic box — minimal slab stand-in.

    Atoms are placed along z at ``spacing`` intervals starting at z=2,
    centred in x and y. Charge is zero (neutral). Multiplicity is chosen
    so the system is a singlet when possible (even n_atoms → mult=1,
    odd → mult=2). dim=3 so the BasisSet constructor is happy.
    """
    lat = np.eye(3) * box
    atoms = [Atom(1, [box / 2, box / 2, 2.0 + i * spacing]) for i in range(n_atoms)]
    mult = 1 if n_atoms % 2 == 0 else 2
    return PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=mult)


def _he_slab_system(
    n_layers: int = 3,
    layer_spacing: float = 4.0,
    vacuum: float = 30.0,
    a_lat: float = 8.0,
) -> PeriodicSystem:
    """A 2D-periodic He-atom slab with vacuum along z.

    Each layer is one He atom (2 electrons) at the cell centre. The slab
    is centred along z with vacuum above and below. dim=2, so the third
    lattice vector is long. He is closed-shell → multiplicity=1 always.
    """
    z_slab = (n_layers - 1) * layer_spacing
    c_vac = z_slab + vacuum
    lat = np.array([[a_lat, 0.0, 0.0], [0.0, a_lat, 0.0], [0.0, 0.0, c_vac]])
    shift_z = (c_vac - z_slab) / 2.0
    atoms = [
        Atom(2, [a_lat / 2, a_lat / 2, shift_z + i * layer_spacing])
        for i in range(n_layers)
    ]
    return PeriodicSystem(dim=2, lattice=lat, unit_cell=atoms)


# ---------------------------------------------------------------------------
# 1. Basic partition: AO count, atom counts, plane geometry
# ---------------------------------------------------------------------------


def test_region_partition_h_chain_4():
    """4 H-atom chain: bottom 2 = substrate, top 2 = region I."""
    sys = _h_chain_system(4, spacing=2.0)  # tight spacing so both layers within margin
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_SUBSTRATE, TAG_REGION_I, TAG_REGION_I]

    rp = RegionPartition.from_tags(sys, basis, tags)

    assert rp.n_i_atoms == 2
    assert rp.n_substrate_atoms == 2
    assert rp.n_dv == 0
    assert rp.n_i > 0
    # Both region-I atoms within 3 bohr of plane S (spacing=2):
    # z_i = 6,8; z_sub max = 4; plane S = 5.0; |6-5|=1, |8-5|=3 → both ≤ margin.
    assert rp.n_s == rp.n_i
    assert set(rp.s_ao) <= set(rp.i_ao)
    assert rp.s_plane_z == pytest.approx(5.0)
    assert np.all(np.diff(rp.i_ao) >= 0)


def test_region_partition_h_chain_6():
    """6 H-atom chain: bottom 2 = sub, middle 2 = region I, top 2 = ΔV.

    spacing=2.0 so both region-I atoms (z=6,8) are within 3 bohr of
    plane S (z=5.0).
    """
    sys = _h_chain_system(6, spacing=2.0)  # tight spacing
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
        TAG_DELTA_V,
        TAG_DELTA_V,
    ]

    rp = RegionPartition.from_tags(sys, basis, tags)

    assert rp.n_i_atoms == 2
    assert rp.n_dv > 0
    assert rp.n_substrate_atoms == 2
    # Plane S: (z_i.min=6 + z_sub.max=4) / 2 = 5.0
    assert rp.s_plane_z == pytest.approx(5.0)
    assert rp.n_s == rp.n_i  # both within margin
    assert len(set(rp.i_ao) & set(rp.dv_ao)) == 0
    assert all(a in set(rp.i_ao) for a in rp.s_ao)


def test_region_partition_no_substrate_below():
    """All atoms are region I — plane S placed at offset below lowest atom."""
    sys = _h_chain_system(4, spacing=4.0)  # 4 atoms → even → singlet
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_REGION_I, TAG_REGION_I, TAG_REGION_I, TAG_REGION_I]

    rp = RegionPartition.from_tags(sys, basis, tags)

    # Plane S = min_z(region I) - offset = 2.0 - 1.0 = 1.0
    assert rp.s_plane_z == pytest.approx(1.0)


def test_region_partition_custom_offset():
    """Custom s_plane_offset is respected when no substrate below."""
    sys = _h_chain_system(4, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_REGION_I, TAG_REGION_I, TAG_REGION_I, TAG_REGION_I]

    rp = RegionPartition.from_tags(sys, basis, tags, s_plane_offset=2.5)
    assert rp.s_plane_z == pytest.approx(2.0 - 2.5)


# ---------------------------------------------------------------------------
# 2. He slab (2D periodic, real slab geometry)
# ---------------------------------------------------------------------------


def test_he_slab_region_partition():
    """3-layer He slab: bottom 2 layers = substrate, top layer = region I."""
    sys = _he_slab_system(n_layers=3, layer_spacing=4.0, vacuum=30.0, a_lat=8.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_SUBSTRATE, TAG_REGION_I]

    rp = RegionPartition.from_tags(sys, basis, tags)

    assert rp.n_i_atoms == 1
    assert rp.n_substrate_atoms == 2
    assert rp.n_dv == 0
    assert rp.n_i > 0
    assert rp.n_s == rp.n_i
    # Plane S at the midpoint between top substrate and bottom region I.
    z_vals = np.array([a.xyz[2] for a in sys.unit_cell], dtype=float)
    z_i = z_vals[rp.i_atoms]
    z_sub = z_vals[rp.substrate_atoms]
    assert z_i.min() > z_sub.max()
    assert rp.s_plane_z == pytest.approx(0.5 * (z_i.min() + z_sub.max()))


def test_he_slab_with_adsorbate():
    """3-layer He slab + He adsorbate: sub, sub, region-I, ΔV."""
    c_vac = 2.0 * 4.0 + 30.0
    a_lat = 8.0
    lat = np.array([[a_lat, 0.0, 0.0], [0.0, a_lat, 0.0], [0.0, 0.0, c_vac]])
    z_slab = 2.0 * 4.0
    shift = (c_vac - z_slab) / 2.0
    atoms = [
        Atom(2, [a_lat / 2, a_lat / 2, shift + 0.0]),
        Atom(2, [a_lat / 2, a_lat / 2, shift + 4.0]),
        Atom(2, [a_lat / 2, a_lat / 2, shift + 8.0]),
        Atom(2, [a_lat / 2, a_lat / 2, shift + 8.0 + 2.0]),
    ]
    sys = PeriodicSystem(dim=2, lattice=lat, unit_cell=atoms)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_SUBSTRATE, TAG_REGION_I, TAG_DELTA_V]

    rp = RegionPartition.from_tags(sys, basis, tags)

    assert rp.n_i_atoms == 1
    assert rp.n_substrate_atoms == 2
    assert rp.n_dv > 0  # adsorbate has AOs
    assert len(set(rp.i_ao) & set(rp.dv_ao)) == 0  # disjoint


# ---------------------------------------------------------------------------
# 3. AO index arithmetic (validates the _AtomAOMap internals)
# ---------------------------------------------------------------------------


def test_ao_atom_map_consistency():
    """AO→atom map covers every AO exactly once, offsets are monotonic."""
    sys = _h_chain_system(6, spacing=4.0)  # even → singlet
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    nbf = basis.nbasis
    n_atoms = len(sys.unit_cell)

    from vibeqc.periodic_embedding.region import _AtomAOMap

    ao_map = _AtomAOMap.from_basis(basis, n_atoms)

    assert len(ao_map.ao_to_atom) == nbf
    assert len(ao_map.atom_ao_start) == n_atoms + 1
    assert np.all(ao_map.ao_to_atom >= 0)
    assert ao_map.atom_ao_start[0] == 0
    assert ao_map.atom_ao_start[-1] == nbf
    assert np.all(np.diff(ao_map.atom_ao_start) >= 0)


def test_ao_atom_map_counts_match_basis():
    """Total AO count from the map equals basis.nbasis."""
    sys = _he_slab_system(n_layers=3)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    n_atoms = len(sys.unit_cell)

    from vibeqc.periodic_embedding.region import _AtomAOMap

    ao_map = _AtomAOMap.from_basis(basis, n_atoms)
    assert ao_map.atom_ao_start[-1] == basis.nbasis


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------


def test_invalid_tags_raises():
    sys = _h_chain_system(4)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [0, 0, 1, 99]  # 99 not valid
    with pytest.raises(ValueError, match="Invalid tag"):
        RegionPartition.from_tags(sys, basis, tags)


def test_wrong_length_tags_raises():
    sys = _h_chain_system(4)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [0, 0, 1]  # too short
    with pytest.raises(ValueError, match="length"):
        RegionPartition.from_tags(sys, basis, tags)


def test_no_region_i_raises():
    sys = _h_chain_system(4)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [0, 0, 0, 0]  # all substrate
    with pytest.raises(ValueError, match="region I"):
        RegionPartition.from_tags(sys, basis, tags)


# ---------------------------------------------------------------------------
# 5. Basis name and n_ao_total carry-through
# ---------------------------------------------------------------------------


def test_basis_metadata_carried():
    sys = _h_chain_system(4)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    rp = RegionPartition.from_tags(sys, basis, [0, 0, 1, 1])
    assert rp.basis_name == "sto-3g"
    assert rp.n_ao_total == basis.nbasis


# ---------------------------------------------------------------------------
# 6. s_atoms selection logic (thick region I)
# ---------------------------------------------------------------------------


def test_thick_region_i_s_atoms_are_bottom_layer():
    """With a wide region I (3 atoms, spacing 6 bohr), only the bottom
    layer should be selected for s_ao (margin=3 bohr by default)."""
    spacing = 6.0
    # 6 atoms: 2 sub + 3 region I + 1 ΔV. 6 atoms → even → singlet.
    sys = _h_chain_system(6, spacing=spacing)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    # z: 2, 8 (sub), 14, 20, 26 (region I), 32 (ΔV)
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
        TAG_REGION_I,
        TAG_DELTA_V,
    ]

    rp = RegionPartition.from_tags(sys, basis, tags)

    # Plane S: (14 + 8) / 2 = 11.0
    assert rp.s_plane_z == pytest.approx(11.0)
    # Only the bottom region-I atom (z=14) is within 3 bohr of plane S.
    z_i = np.array([a.xyz[2] for a in sys.unit_cell], dtype=float)[rp.i_atoms]
    bottom_i_atom = rp.i_atoms[np.argmin(z_i)]
    assert len(rp.s_atoms) == 1
    assert rp.s_atoms[0] == bottom_i_atom
    assert rp.n_s < rp.n_i


def test_s_ao_is_subset_of_i_ao():
    """s_ao must always be a subset (not necessarily proper) of i_ao."""
    sys = _h_chain_system(6, spacing=4.0)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [
        TAG_SUBSTRATE,
        TAG_SUBSTRATE,
        TAG_REGION_I,
        TAG_REGION_I,
        TAG_DELTA_V,
        TAG_DELTA_V,
    ]
    rp = RegionPartition.from_tags(sys, basis, tags)
    s_set = set(rp.s_ao)
    i_set = set(rp.i_ao)
    assert s_set <= i_set
