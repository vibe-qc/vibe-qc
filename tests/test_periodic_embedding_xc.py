"""Tests for DFT XC integration (xc.py)."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, PeriodicSystem
from vibeqc.periodic_embedding.region import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    RegionPartition,
)
from vibeqc.periodic_embedding.xc import build_vxc_from_density


def _h_system():
    box = 30.0
    lat = np.eye(3) * box
    atoms = [
        Atom(1, [box / 2, box / 2, 2.0]),
        Atom(1, [box / 2, box / 2, 6.0]),
        Atom(1, [box / 2, box / 2, 10.0]),
    ]
    return PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=2)


def test_build_vxc_lda():
    sys = _h_system()
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_SUBSTRATE, TAG_REGION_I]
    region = RegionPartition.from_tags(sys, basis, tags)

    # Mock density on region I (1×1).
    d = np.array([[0.3]], dtype=float)
    vxc, exc = build_vxc_from_density(mol, basis, region, d, functional_name="LDA")

    assert vxc.shape == (1, 1)
    assert np.isfinite(vxc[0, 0])
    assert exc < 0  # XC energy is negative for finite density
    assert np.isfinite(exc)


def test_build_vxc_pbe():
    sys = _h_system()
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_SUBSTRATE, TAG_REGION_I]
    region = RegionPartition.from_tags(sys, basis, tags)

    d = np.array([[0.3]], dtype=float)
    vxc, exc = build_vxc_from_density(mol, basis, region, d, functional_name="PBE")

    assert vxc.shape == (1, 1)
    assert exc < 0
    # PBE should differ from LDA (sigma contribution).
    vxc_lda, _ = build_vxc_from_density(mol, basis, region, d, functional_name="LDA")
    assert vxc[0, 0] != pytest.approx(vxc_lda[0, 0], abs=1e-10)


def test_build_vxc_two_ao():
    sys = _h_system()
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_REGION_I, TAG_REGION_I]
    region = RegionPartition.from_tags(sys, basis, tags)

    d = np.array([[0.2, 0.05], [0.05, 0.3]], dtype=float)
    vxc, exc = build_vxc_from_density(mol, basis, region, d, functional_name="PBE")

    assert vxc.shape == (2, 2)
    assert np.allclose(vxc, vxc.T, atol=1e-10)  # symmetric
    assert exc < 0
