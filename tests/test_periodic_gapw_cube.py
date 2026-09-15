"""Tests for the periodic-GPW Gaussian-cube writer.

The molecular cube writer at :mod:`vibeqc.cube` builds its own
axis-aligned bounding box around a :class:`Molecule`; the
periodic variant lifts the box to the :class:`PeriodicSystem`
lattice and the voxel layout to the :class:`PlaneWaveGrid` the
GPW SCF actually ran on. Tests pin:

1. **Smoke** — a He STO-3G GPW SCF in a 16-bohr cube writes a
   parseable cube file whose integrated density matches the
   total electron count to GPW quadrature precision (≈ 1e-3,
   tighter than the molecular cube writer because we use the
   same grid the SCF saw — no resampling).

2. **Round-trip parse** — reading the cube file back and
   comparing the data block to a fresh
   :func:`collocate_density_on_grid` call recovers the same
   ``(nx, ny, nz)`` array to the cube file's printed precision
   (``%13.5e``, ~ 1e-5 relative).

3. **Dimensionality guard** — a 2D or 1D periodic system is
   rejected with a clear :class:`ValueError` rather than
   silently writing a cube with a degenerate voxel vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_cube import write_cube_density_periodic
from vibeqc.periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
)
from vibeqc.periodic_gapw_j import (
    collocate_density_on_grid,
    run_periodic_rhf_gpw,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Fixtures ------------------------------------------------------

def _he_periodic_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def _he_basis(L: float = 16.0):
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    return vq.BasisSet(mol, "sto-3g")


def _run_he_gpw(L: float = 16.0, n: int = 48):
    system = _he_periodic_system(L)
    basis = _he_basis(L)
    grid = PlaneWaveGrid(np.eye(3) * L, n, n, n)
    result = run_periodic_rhf_gpw(system, basis, grid=grid, quiet=True)
    return system, basis, result


# ---------- Cube parser (round-trip) --------------------------------------

def _parse_cube(path: Path):
    """Parse a minimal ASCII Gaussian cube. Returns
    ``(origin, voxels, n_atoms, data)``.

    Voxels is ``(3, 3)`` with row i = the i-th voxel vector.
    Data is the ``(nx, ny, nz)`` array — x outer, z inner.
    """
    with path.open() as f:
        f.readline()  # title
        f.readline()  # comment
        tok = f.readline().split()
        n_atoms = int(tok[0])
        origin = np.array([float(t) for t in tok[1:4]])
        voxel_lines = [f.readline().split() for _ in range(3)]
        n_axis = [int(t[0]) for t in voxel_lines]
        voxels = np.array([
            [float(t) for t in line[1:4]] for line in voxel_lines
        ])
        # Skip atom block.
        for _ in range(n_atoms):
            f.readline()
        # Read remaining numbers as a flat float array.
        flat = []
        for line in f:
            for tok in line.split():
                flat.append(float(tok))
        data = np.array(flat, dtype=float).reshape(n_axis)
        return origin, voxels, n_atoms, data


# ---------- Tests ---------------------------------------------------------

def test_write_cube_density_periodic_smoke_he(tmp_path: Path):
    """End-to-end: a converged He STO-3G GPW SCF writes a cube
    whose integrated density equals the total electron count
    within GPW quadrature precision."""
    system, basis, result = _run_he_gpw(L=16.0, n=48)
    assert result.converged

    path = tmp_path / "he.cube"
    out = write_cube_density_periodic(path, result, basis, system)

    assert out == path
    assert path.exists() and path.stat().st_size > 0

    # Parse the header back and verify shape + voxel vectors.
    origin, voxels, n_atoms, data = _parse_cube(path)
    assert n_atoms == 1
    np.testing.assert_allclose(origin, np.zeros(3), atol=1e-12)
    # Voxel vectors equal lattice columns / n_axis.
    expected_voxels = (np.eye(3) * 16.0) / 48
    # The cube emits voxel i = A[:, i] / n_i — row of voxels matrix is v_i.
    np.testing.assert_allclose(voxels, expected_voxels, atol=1e-6)
    assert data.shape == (48, 48, 48)

    # Integrated density ≈ tr(D · S). For He STO-3G that's 2 e exactly
    # (the basis has a single 1s function with unit overlap).
    voxel_vol = abs(np.linalg.det(expected_voxels))
    integrated = float(data.sum()) * voxel_vol
    # The molecular SCF tr(D·S) for He is exactly 2.0 e.
    assert integrated == pytest.approx(2.0, abs=1e-3)


def test_write_cube_density_periodic_roundtrip_matches_collocate(
    tmp_path: Path,
):
    """The cube data block, read back, equals a fresh
    :func:`collocate_density_on_grid` call to the cube file's
    printed precision (``%13.5e`` → ~1e-5 relative)."""
    system, basis, result = _run_he_gpw(L=16.0, n=32)

    path = tmp_path / "he_roundtrip.cube"
    write_cube_density_periodic(path, result, basis, system)
    _, _, _, data = _parse_cube(path)

    # Fresh collocation on the same grid.
    rho_fresh = collocate_density_on_grid(
        basis, np.asarray(result.density, dtype=float), result.grid,
    )
    assert data.shape == rho_fresh.shape
    # The cube format prints with %13.5e — 5 significant digits.
    # Use atol scaled to the max density so dense regions match to
    # ~5 sig-figs and the (essentially zero) tail matches to ~1e-10.
    rho_max = float(np.abs(rho_fresh).max())
    np.testing.assert_allclose(data, rho_fresh, rtol=1e-4,
                               atol=1e-5 * rho_max)


def test_write_cube_density_periodic_rejects_non_3d(tmp_path: Path):
    """1D / 2D periodic systems don't have three voxel vectors and
    must be rejected with a clear :class:`ValueError`."""
    system, basis, result = _run_he_gpw(L=16.0, n=32)

    # Build a sibling 2D system that shares the grid + density so
    # the only thing wrong is the dimensionality flag.
    sys2d = core.PeriodicSystem()
    sys2d.dim = 2
    sys2d.lattice = np.eye(3) * 16.0
    sys2d.unit_cell = list(system.unit_cell)

    path = tmp_path / "should_not_exist.cube"
    with pytest.raises(ValueError, match="3D"):
        write_cube_density_periodic(path, result, basis, sys2d)
    assert not path.exists()
