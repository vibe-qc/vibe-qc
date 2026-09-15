"""Macroscopic dipole-surface diagnostics for neutral periodic cells.

The lattice sum is **conditionally convergent for the dipole-dipole
piece** (depolarization-field problem), so its surface contribution depends
on the macroscopic sample shape and boundary condition.  The closed form in
this module is the spherical-sample depolarization term
``2 pi |mu|^2 / (3 V)`` from Saunders et al. (1992), Eqs. 20-21.  It is not
the conducting or tin-foil surface term, which is zero.  The spherical term
depends on the macroscopic sample boundary, not on whether the primitive cell
is cubic, tetragonal, or skew.

Module status: Phase 4b dipole diagnostic only.  Saunders et al. (1992),
Eqs. 116-130, define CRYSTAL's printed ``EXT EL-POLE`` through
shell-partitioned multipoles, an Ewald model, and the subtraction of point
multipole interactions inside an explicit penetration zone.  Neither this
dipole formula nor a reciprocal sum over the complete AO density determines
that decomposition.  Term-by-term CRYSTAL parity therefore remains gated
until the shell partition, multipole order, and penetration policy are part
of the implementation.

References
----------
* Saunders, V. R.; Freyria-Fava, C.; Dovesi, R.; Salasco, L.; Roetti, C.
  Mol. Phys. 77, 629 (1992), doi:10.1080/00268979200102671 -- periodic
  electrostatics and the BIPOLE multipole/penetration decomposition.
* Stone, A. J. "The Theory of Intermolecular Forces", 2nd ed.,
  OUP (2013), Sec.3 -- multipole-multipole interactions + periodic
  lattice sums.
* de Leeuw, S. W.; Perram, J. W.; Smith, E. R. Proc. R. Soc. Lond.
  A 373, 27 (1980) -- Ewald summation for arbitrary periodic
  multipole arrays.
"""
from __future__ import annotations

import math

import numpy as np

from ._vibeqc_core import PeriodicSystem
from .bipole_cell_moments import CellMultipoleMoments


__all__ = [
    "cell_volume_bohr3",
    "dipole_depolarization_surface_energy_spherical",
    "dipole_dipole_self_energy_cubic",
    "dipole_dipole_self_energy",
]


def cell_volume_bohr3(system: PeriodicSystem) -> float:
    """Cell volume in bohr^3 for a 3D periodic system.

    For 1D / 2D systems the "cell volume" is ill-defined -- those
    raise ``ValueError``.
    """
    if system.dim != 3:
        raise ValueError(
            f"cell_volume_bohr3 requires dim=3; got dim={system.dim}"
        )
    lattice = np.asarray(system.lattice, dtype=float)
    return float(abs(np.linalg.det(lattice)))


def dipole_depolarization_surface_energy_spherical(
    cell_moments: CellMultipoleMoments,
    V_cell_bohr3: float,
    *,
    charge_tolerance: float = 1e-8,
) -> float:
    """Spherical-sample dipole surface contribution for a neutral cell.

    For cell dipole moment ``mu`` and volume ``V``::

        E_surface = (2pi / 3V) . |mu|^2

    Saunders et al. (1992), Eqs. 20-21, assume an electroneutral lattice
    basis.  Neutrality makes ``mu`` origin-independent.  Electron-only cell
    moments are charged and must first be compensated, for example through
    ``compute_cell_multipole_moments(..., system=system)``.

    The primitive-cell metric does not select the macroscopic boundary.
    This function explicitly selects a spherical sample; the conducting or
    tin-foil surface contribution is zero, while other sample shapes require
    their own depolarization tensor.

    Parameters
    ----------
    cell_moments : CellMultipoleMoments
        Cell-level Cartesian moments (Phase 4a output). Must have
        ``L_max >= 1`` (contains the dipole components).
    V_cell_bohr3 : float
        Cell volume in bohr^3. Use ``cell_volume_bohr3(system)``.
    charge_tolerance : float
        Maximum allowed absolute monopole of the supplied cell moments.

    Returns
    -------
    float
        Spherical-sample dipole surface contribution in Ha.
    """
    if cell_moments.L_max < 1:
        raise ValueError(
            "dipole_depolarization_surface_energy_spherical requires "
            "L_max >= 1; "
            f"got {cell_moments.L_max}"
        )
    if not math.isfinite(V_cell_bohr3) or V_cell_bohr3 <= 0.0:
        raise ValueError(f"V_cell must be finite and positive; got {V_cell_bohr3}")
    if not math.isfinite(charge_tolerance) or charge_tolerance < 0.0:
        raise ValueError(
            "charge_tolerance must be finite and non-negative; "
            f"got {charge_tolerance}"
        )
    monopole = cell_moments["S"]
    if abs(monopole) > charge_tolerance:
        raise ValueError(
            "spherical dipole surface energy requires neutral cell moments; "
            f"got monopole {monopole} (tolerance {charge_tolerance}). "
            "Use nucleus-compensated moments."
        )
    mu_x = cell_moments["x"]
    mu_y = cell_moments["y"]
    mu_z = cell_moments["z"]
    mu_sq = mu_x * mu_x + mu_y * mu_y + mu_z * mu_z
    return (2.0 * math.pi / (3.0 * V_cell_bohr3)) * mu_sq


def dipole_dipole_self_energy_cubic(
    cell_moments: CellMultipoleMoments,
    V_cell_bohr3: float,
) -> float:
    """Compatibility name for the spherical-sample dipole surface term.

    The formula is independent of primitive-cell metric; ``cubic`` in this
    historical name does not impose a physical restriction.  New code should
    call :func:`dipole_depolarization_surface_energy_spherical`.
    """
    return dipole_depolarization_surface_energy_spherical(
        cell_moments,
        V_cell_bohr3,
    )


def dipole_dipole_self_energy(
    cell_moments: CellMultipoleMoments,
    system: PeriodicSystem,
    *,
    assume_cubic: bool = True,
) -> float:
    """Compatibility wrapper for the spherical-sample surface contribution.

    The spherical macroscopic boundary is explicit and applies for any 3D
    primitive-cell metric.  This delegates to
    :func:`dipole_depolarization_surface_energy_spherical`.

    Parameters
    ----------
    cell_moments : CellMultipoleMoments
        Phase 4a output.
    system : PeriodicSystem
        Supplies the 3D cell volume.
    assume_cubic : bool, default True
        Deprecated compatibility parameter with no effect.  Primitive-cell
        cubicity does not define the macroscopic sample boundary.

    Returns
    -------
    float
        Spherical-sample dipole surface contribution in Ha.
    """
    del assume_cubic
    V = cell_volume_bohr3(system)
    return dipole_depolarization_surface_energy_spherical(
        cell_moments,
        V,
    )
