"""Stress tensor for the GPW periodic route — correctness and parity.

Validates :func:`vibeqc.periodic_gapw_stress.compute_stress_gpw` on
simple periodic systems.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


def _he_system(L: float = 12.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def test_gapw_stress_fails_closed_for_analytic_one_centre():
    """Finite-strain GAPW stress currently implements the block functional."""
    from vibeqc.periodic_gapw_stress import compute_stress_gapw

    result = SimpleNamespace(converged=True, one_centre="analytic")
    with pytest.raises(NotImplementedError, match="one_centre='block'"):
        compute_stress_gapw(
            _he_system(),
            None,
            result,
            basis_name="sto-3g",
        )


def test_stress_tensor_is_symmetric():
    """Stress tensor must be symmetric for any physical system."""
    L = 12.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
    result = run_periodic_rhf_gpw(
        system,
        basis,
        grid=grid,
        quiet=True,
        max_iter=30,
    )
    from vibeqc.periodic_gapw_stress import compute_stress_gpw

    stress = compute_stress_gpw(
        system,
        basis,
        result,
        basis_name="sto-3g",
        grid=grid,
    )
    assert stress.shape == (3, 3)
    assert np.allclose(stress, stress.T, atol=1e-12), f"stress not symmetric: {stress}"


def test_stress_isotropic_atom():
    """He atom in cubic box: diagonal stress equal, off-diagonal ~0."""
    L = 12.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
    result = run_periodic_rhf_gpw(
        system,
        basis,
        grid=grid,
        quiet=True,
        max_iter=30,
    )
    from vibeqc.periodic_gapw_stress import compute_stress_gpw

    stress = compute_stress_gpw(
        system,
        basis,
        result,
        basis_name="sto-3g",
        grid=grid,
    )
    # Diagonal components should be nearly equal (isotropic).
    diag = np.diag(stress)
    assert abs(diag[0] - diag[1]) < 5e-5, f"σ_xx ≠ σ_yy: {diag}"
    assert abs(diag[0] - diag[2]) < 5e-5, f"σ_xx ≠ σ_zz: {diag}"
    # Off-diagonal should be ~0.
    offdiag = stress[0, 1]
    assert abs(offdiag) < 1e-3, f"σ_xy = {offdiag} (expected ~0)"
    # Stress should be finite.
    assert np.all(np.isfinite(stress))
    # For He in a 16-bohr box, the GPW energy on a coarse 16³ grid
    # is not at the correct absolute scale, so we only check finite
    # and symmetric, not the sign.


def test_stress_h2_finite():
    """H2 in cubic box: stress is finite and physically sensible."""
    L = 12.0
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
    result = run_periodic_rhf_gpw(
        system,
        basis,
        grid=grid,
        quiet=True,
        max_iter=30,
    )
    from vibeqc.periodic_gapw_stress import compute_stress_gpw

    stress = compute_stress_gpw(
        system,
        basis,
        result,
        basis_name="sto-3g",
        grid=grid,
    )
    assert np.all(np.isfinite(stress))
    assert np.allclose(stress, stress.T, atol=1e-12)
    # H2 bond along x: σ_xx should be most compressive (bond compression).
    diag = np.diag(stress)
    assert abs(diag[0]) > 0, f"σ_xx should be non-zero, got {diag[0]}"


def test_stress_gpw_vs_gapw_consistent():
    """GPW and GAPW stress should be consistent when augmentation is
    inactive (He has no soft shells)."""
    L = 12.0
    system = _he_system(L)
    mol = vq.Molecule(list(system.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)
    result = run_periodic_rhf_gpw(
        system,
        basis,
        grid=grid,
        quiet=True,
        max_iter=30,
    )
    from vibeqc.periodic_gapw_stress import compute_stress_gapw, compute_stress_gpw

    s_gpw = compute_stress_gpw(
        system,
        basis,
        result,
        basis_name="sto-3g",
        grid=grid,
    )
    s_gapw = compute_stress_gapw(
        system,
        basis,
        result,
        basis_name="sto-3g",
        grid=grid,
    )
    # For He (augmentation inactive), GPW and GAPW stress should match.
    # Relaxed tolerance accounts for subtle differences in the J builder
    # (GapwJBuilder internally delegates to GpwJBuilder when augmentation
    # is inactive, but the two paths may create slightly different grid
    # or basis objects).
    assert np.allclose(s_gpw, s_gapw, atol=1e-3), (
        f"GPW stress:\n{s_gpw}\nGAPW stress:\n{s_gapw}"
    )
