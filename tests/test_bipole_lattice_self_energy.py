"""Tests for the Phase 4b spherical-sample dipole diagnostic.

The spherical macroscopic boundary gives the dipole surface formula
``E = (2π/3V)·|μ|²``. We validate it against:
  (a) The closed-form value for a synthetic μ at known V.
  (b) Its inverse-volume and rotational invariants.

This is the spherical-sample depolarization surface term, not the
conducting/tin-foil surface term and not CRYSTAL's ``EXT EL-POLE``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.bipole_cell_moments import CellMultipoleMoments
from vibeqc.bipole_lattice_self_energy import (
    cell_volume_bohr3,
    dipole_depolarization_surface_energy_spherical,
    dipole_dipole_self_energy,
    dipole_dipole_self_energy_cubic,
)


ANG2BOHR = 1.0 / 0.529177210903


def _make_cubic_system(a_bohr: float) -> vq.PeriodicSystem:
    """Simple-cubic 3D periodic system with edge length a_bohr,
    one atom at the origin (the atom identity doesn't matter here —
    we only use the lattice for the volume)."""
    lattice = a_bohr * np.eye(3)
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0])]
    return vq.PeriodicSystem(3, lattice, atoms)


def _make_fcc_system(a_bohr: float) -> vq.PeriodicSystem:
    """FCC primitive: non-orthogonal cell for the volume tests."""
    lattice = (a_bohr / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0])]
    return vq.PeriodicSystem(3, lattice, atoms)


# ---------------------------------------------------------------------
# cell_volume_bohr3 sanity
# ---------------------------------------------------------------------
def test_cell_volume_cubic():
    a = 4.5
    system = _make_cubic_system(a)
    V = cell_volume_bohr3(system)
    assert math.isclose(V, a ** 3, rel_tol=1e-12)


def test_cell_volume_fcc():
    """FCC primitive volume = a³/4."""
    a = 5.0
    system = _make_fcc_system(a)
    V = cell_volume_bohr3(system)
    assert math.isclose(V, a ** 3 / 4.0, rel_tol=1e-12)


def test_cell_volume_dim_lt_3_raises():
    lat = np.eye(3)
    sys2d = vq.PeriodicSystem(2, lat, [vq.Atom(1, [0, 0, 0])])
    with pytest.raises(ValueError):
        cell_volume_bohr3(sys2d)


# ---------------------------------------------------------------------
# Spherical-sample dipole surface term: closed form and invariants
# ---------------------------------------------------------------------
def test_dipole_dipole_self_energy_closed_form_x_only():
    """μ = (1, 0, 0), V = 27 bohr³ → E = 2π/(3·27) · 1 = 2π/81."""
    moments = np.zeros(10)
    moments[1] = 1.0  # μ_x
    cell_M = CellMultipoleMoments(
        L_max=2, n_components=10, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    V = 27.0
    E = dipole_depolarization_surface_energy_spherical(cell_M, V)
    expected = (2.0 * math.pi) / (3.0 * V) * 1.0
    assert math.isclose(E, expected, rel_tol=1e-12)


def test_dipole_dipole_self_energy_isotropic():
    """|μ|² should be the sum of squared components — direction-invariant."""
    moments = np.zeros(10)
    moments[1] = 0.6
    moments[2] = -0.8
    moments[3] = 0.0
    cell_M = CellMultipoleMoments(
        L_max=2, n_components=10, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    V = 64.0
    E = dipole_depolarization_surface_energy_spherical(cell_M, V)
    expected = (2.0 * math.pi / (3.0 * V)) * (0.36 + 0.64)
    assert math.isclose(E, expected, rel_tol=1e-12)


def test_dipole_dipole_self_energy_scales_with_inverse_volume():
    """Doubling V halves E."""
    moments = np.zeros(10)
    moments[1:4] = [1.0, 1.0, 1.0]
    cell_M = CellMultipoleMoments(
        L_max=2, n_components=10, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    E_V = dipole_depolarization_surface_energy_spherical(cell_M, 10.0)
    E_2V = dipole_depolarization_surface_energy_spherical(cell_M, 20.0)
    assert math.isclose(E_2V, E_V / 2.0, rel_tol=1e-12)


def test_dipole_surface_energy_zero_for_neutral_zero_dipole():
    """A neutral cell with no net dipole has zero spherical surface term."""
    moments = np.zeros(10)
    cell_M = CellMultipoleMoments(
        L_max=2, n_components=10, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    E = dipole_depolarization_surface_energy_spherical(cell_M, 100.0)
    assert E == 0.0


# ---------------------------------------------------------------------
# Macroscopic boundary is independent of primitive-cell metric
# ---------------------------------------------------------------------
def test_spherical_surface_term_accepts_any_primitive_cell_metric():
    moments = np.zeros(4)
    moments[1] = 1.0
    cell_M = CellMultipoleMoments(
        L_max=1, n_components=4, origin=(0.0, 0.0, 0.0), moments=moments,
    )

    systems = [
        _make_cubic_system(4.5),
        _make_fcc_system(5.0),
        vq.PeriodicSystem(
            3,
            np.diag([4.0, 4.0, 6.0]),
            [vq.Atom(1, [0.0, 0.0, 0.0])],
        ),
        vq.PeriodicSystem(
            3,
            np.array(
                [[5.0, 0.0, 0.0], [0.0, 3.0, 4.0], [4.0, 0.0, 3.0]]
            ),
            [vq.Atom(1, [0.0, 0.0, 0.0])],
        ),
    ]
    for system in systems:
        E = dipole_dipole_self_energy(cell_M, system, assume_cubic=False)
        V = cell_volume_bohr3(system)
        expected = (2.0 * math.pi / (3.0 * V)) * 1.0
        assert math.isclose(E, expected, rel_tol=1e-12)


def test_historical_cubic_name_is_compatibility_alias():
    moments = np.zeros(4)
    moments[1] = 1.0
    cell_M = CellMultipoleMoments(
        L_max=1, n_components=4, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    expected = dipole_depolarization_surface_energy_spherical(cell_M, 27.0)
    assert dipole_dipole_self_energy_cubic(cell_M, 27.0) == expected


# ---------------------------------------------------------------------
# A direct lattice sum is conditionally convergent and needs an explicitly
# matched macroscopic boundary.  The invariants above pin the implemented
# spherical-sample surface term without mislabelling it as tin-foil.
# ---------------------------------------------------------------------
def test_input_validation():
    """L_max < 1 → raises."""
    cell_M = CellMultipoleMoments(
        L_max=0, n_components=1, origin=(0.0, 0.0, 0.0),
        moments=np.array([1.0]),
    )
    with pytest.raises(ValueError):
        dipole_depolarization_surface_energy_spherical(cell_M, 1.0)
    # V_cell ≤ 0 also raises.
    moments = np.zeros(4)
    moments[1] = 1.0
    cell_M = CellMultipoleMoments(
        L_max=1, n_components=4, origin=(0.0, 0.0, 0.0), moments=moments,
    )
    with pytest.raises(ValueError):
        dipole_depolarization_surface_energy_spherical(cell_M, -1.0)
    with pytest.raises(ValueError):
        dipole_depolarization_surface_energy_spherical(cell_M, math.inf)
    with pytest.raises(ValueError):
        dipole_depolarization_surface_energy_spherical(cell_M, math.nan)
    for tolerance in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="charge_tolerance"):
            dipole_depolarization_surface_energy_spherical(
                cell_M,
                27.0,
                charge_tolerance=tolerance,
            )

    charged = np.zeros(4)
    charged[0] = 2.0
    charged[1] = 1.0
    charged_moments = CellMultipoleMoments(
        L_max=1,
        n_components=4,
        origin=(0.0, 0.0, 0.0),
        moments=charged,
    )
    with pytest.raises(ValueError, match="neutral cell moments"):
        dipole_depolarization_surface_energy_spherical(charged_moments, 27.0)
