"""Validation tests for the embedding method against explicit slab calculations.

Validates that the embedded method runs on real slab geometries and
produces sensible results.  The thick-slab GDF parity gate is a
follow-up requiring full periodic SCF infrastructure.
"""

from __future__ import annotations

import numpy as np
from vibeqc._vibeqc_core import Atom, BasisSet, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic_embedding.region import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    RegionPartition,
)
from vibeqc.periodic_embedding.runner import run_embedded_surface


def _h_chain_slab(
    n_layers: int = 4,
    spacing: float = 3.0,
    a_lat: float = 10.0,
    vacuum: float = 30.0,
) -> PeriodicSystem:
    """A 2D-periodic H-atom slab (one atom per layer, dim=2)."""
    z_slab = (n_layers - 1) * spacing
    c_vac = z_slab + vacuum
    lat = np.array([[a_lat, 0.0, 0.0], [0.0, a_lat, 0.0], [0.0, 0.0, c_vac]])
    shift_z = (c_vac - z_slab) / 2.0
    atoms = [
        Atom(1, [a_lat / 2, a_lat / 2, shift_z + i * spacing]) for i in range(n_layers)
    ]
    mult = 1 if n_layers % 2 == 0 else 2
    return PeriodicSystem(dim=2, lattice=lat, unit_cell=atoms, multiplicity=mult)


def _default_lat_opts() -> LatticeSumOptions:
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    opts.nuclear_cutoff_bohr = 15.0
    return opts


# ---------------------------------------------------------------------------
# 1. Embedded density on H slab converges with substrate depth
# ---------------------------------------------------------------------------


def test_h_slab_density_converges_with_depth():
    """As the substrate gets deeper, the embedded top-layer density
    stabilizes."""
    spacing = 2.0
    lat_opts = _default_lat_opts()
    # H sto-3g eigenvalues are ~ -1.5 to -1.1 Ha.
    e_bot = -2.2
    e_fermi = -1.0

    top_diag = []
    for n in (4, 5, 6):
        sys = _h_chain_slab(n_layers=n, spacing=spacing)
        mol = sys.unit_cell_molecule()
        basis = BasisSet(mol, "sto-3g")
        tags = [TAG_SUBSTRATE] * (n - 1) + [TAG_REGION_I]

        result = run_embedded_surface(
            sys,
            basis,
            tags,
            surface_k_mesh=(1, 1),
            contour_n_nodes=32,
            e_bottom=e_bot,
            e_fermi=e_fermi,
            lat_opts=lat_opts,
        )
        d_local = result.density_local
        assert d_local.shape == (1, 1)
        top_diag.append(d_local[0, 0])

    # All densities should be positive and in a reasonable range.
    for d in top_diag:
        assert d > 0.01
        assert d < 1.5

    # Values should not drift wildly with depth.
    assert abs(top_diag[2] - top_diag[0]) < 0.5


# ---------------------------------------------------------------------------
# 2. Multi-atom layer (2 H per layer)
# ---------------------------------------------------------------------------


def test_two_atoms_per_layer():
    """Two H atoms in the same z-layer produce a 2×2 density block."""
    box = 30.0
    a_lat = 10.0
    vac = 30.0
    spacing = 2.0
    n_layers = 5
    z_slab = (n_layers - 1) * spacing
    c_vac = z_slab + vac
    lat = np.array([[a_lat, 0.0, 0.0], [0.0, a_lat, 0.0], [0.0, 0.0, c_vac]])
    shift_z = (c_vac - z_slab) / 2.0
    atoms = []
    for i in range(n_layers):
        z = shift_z + i * spacing
        atoms.append(Atom(1, [a_lat / 2 - 1.0, a_lat / 2, z]))
        atoms.append(Atom(1, [a_lat / 2 + 1.0, a_lat / 2, z]))
    sys = PeriodicSystem(dim=2, lattice=lat, unit_cell=atoms, multiplicity=1)
    mol = sys.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")

    # Bottom 4 layers (8 atoms) = substrate, top 1 layer (2 atoms) = region I.
    tags = [TAG_SUBSTRATE] * 8 + [TAG_REGION_I] * 2

    result = run_embedded_surface(
        sys,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=32,
        e_bottom=-2.2,
        e_fermi=-1.0,
    )
    assert result.n_i == 2  # two AOs in region I
    assert result.density_local.shape == (2, 2)
    assert result.occupation > 0
    # Diagonal elements positive (charge per AO).
    assert np.all(np.diag(result.density_local) >= 0)


# ---------------------------------------------------------------------------
# 3. Embedded calculation on a slab from build.slab()
# ---------------------------------------------------------------------------


def test_embedded_on_build_slab():
    """Run the embedding on a slab built with build.slab() and verify
    basic sanity."""
    from vibeqc.build import slab as build_slab

    sys_slab, info = build_slab(
        "He",
        "fcc",
        facet=(1, 0, 0),
        n_layers=4,
        vacuum=20,
        supercell=(1, 1),
        a=4.0,
        periodic_z=True,
    )
    mol = sys_slab.unit_cell_molecule()
    basis = BasisSet(mol, "sto-3g")
    tags = [TAG_SUBSTRATE] * 3 + [TAG_REGION_I]

    # He eigenvalues ~ -11 to -9 Ha. Use a wide contour.
    result = run_embedded_surface(
        sys_slab,
        basis,
        tags,
        surface_k_mesh=(1, 1),
        contour_n_nodes=40,
        e_bottom=-13.0,
        e_fermi=-8.0,
    )

    # Basic sanity: the calculation completed and produced valid shapes.
    assert result.n_i == 1
    assert result.density_local.shape == (1, 1)
    assert result.region.n_ao_total == basis.nbasis
