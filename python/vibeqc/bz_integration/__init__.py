"""Brillouin-zone integration backends for metallic occupations.

Geometric (T=0) Fermi-surface integrators that produce fractional band
occupations + the Fermi level on a regular k-mesh, as an alternative to the
temperature-broadened :mod:`vibeqc.smearing` methods. The first backend is the
Gilat-Raubenheimer net (the integrator behind CRYSTAL's ``SHRINK IS ISP``
second/Gilat net); a Blochl linear-tetrahedron sibling may land alongside it.

These sit *downstream* of the J/K build and are engine-agnostic: BIPOLE/Ewald,
GDF, and GPW/GAPW multi-k drivers all produce per-k eigenvalues and can consume
the same backend.
"""

from __future__ import annotations

from .gilat import (
    gilat_dos,
    gilat_raubenheimer_occupations,
    grid_band_slopes,
    occupied_fraction,
)
from .grid import (
    eigenvalues_to_full_grid,
    expand_ibz_eigenvalues,
    fractional_kpoints,
    gilat_occupations_for_kmesh,
    gilat_occupations_on_kmesh,
    grid_to_kpoints,
    kpoint_grid_indices,
)

__all__ = [
    "gilat_dos",
    "gilat_raubenheimer_occupations",
    "grid_band_slopes",
    "occupied_fraction",
    "eigenvalues_to_full_grid",
    "expand_ibz_eigenvalues",
    "fractional_kpoints",
    "gilat_occupations_for_kmesh",
    "gilat_occupations_on_kmesh",
    "grid_to_kpoints",
    "kpoint_grid_indices",
]
