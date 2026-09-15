"""Tests for LDOS grid computation (ldos_grid.py)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic_embedding.ldos_grid import compute_ldos_grid, save_embedded_ldos
from vibeqc.periodic_embedding.region import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    RegionPartition,
)
from vibeqc.periodic_embedding.scf2step import compute_region_i_gf_at_kz
from vibeqc.periodic_embedding.surface_sigma import build_surface_sigma


def _h_system():
    box = 30.0
    lat = np.eye(3) * box
    atoms = [
        Atom(1, [box / 2, box / 2, 2.0]),
        Atom(1, [box / 2, box / 2, 6.0]),
        Atom(1, [box / 2, box / 2, 10.0]),
    ]
    return PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=2)


def test_compute_ldos_grid():
    sys = _h_system()
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_SUBSTRATE, TAG_REGION_I]
    region = RegionPartition.from_tags(sys, basis, tags)

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    opts.nuclear_cutoff_bohr = 15.0
    sigma = build_surface_sigma(sys, basis, region, [0, 0], lat_opts=opts)

    def g_fn(z):
        return compute_region_i_gf_at_kz(
            sys, basis, region, np.zeros(3), z, sigma.at(z), lat_opts=opts
        )

    energies = np.array([-1.3, -1.0])
    result = compute_ldos_grid(
        sys,
        basis,
        region,
        g_fn,
        energies,
        margin=2.0,
        spacing=0.5,
        eta=0.05,
    )

    assert "ldos" in result
    assert result["ldos"].shape[0] == 2  # 2 energies
    assert result["ldos"].dtype == np.float32
    assert "energies" in result
    assert "extent" in result
    assert result["ldos"].sum() > 0  # positive LDOS


def test_save_ldos():
    sys = _h_system()
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE, TAG_SUBSTRATE, TAG_REGION_I]
    region = RegionPartition.from_tags(sys, basis, tags)
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    opts.nuclear_cutoff_bohr = 15.0
    sigma = build_surface_sigma(sys, basis, region, [0, 0], lat_opts=opts)

    def g_fn(z):
        return compute_region_i_gf_at_kz(
            sys, basis, region, np.zeros(3), z, sigma.at(z), lat_opts=opts
        )

    result = compute_ldos_grid(
        sys,
        basis,
        region,
        g_fn,
        np.array([-1.3]),
        margin=2.0,
        spacing=1.0,
        eta=0.05,
    )

    with tempfile.TemporaryDirectory() as td:
        stem = str(Path(td) / "test")
        path = save_embedded_ldos(stem, result, metadata={"test": True})
        assert path.exists()
        loaded = np.load(path)
        assert "ldos" in loaded
        assert "metadata" in loaded
