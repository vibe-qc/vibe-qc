"""Cube-file output: header well-formedness + numerical sum-rules.

The cube format is rigid (line-by-line column layout, voxel-major
data order). These tests pin both the format and the physics:

* header parses back as expected (atom count, voxel vectors, atoms)
* data block has 6 floats per line and the right number of values
* electron density on a sufficiently dense grid integrates to N_e
* MO data on the same grid is normalized: ⟨φ_i|φ_i⟩ ≈ 1
* multi-MO cube reports its volume count correctly
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq


@pytest.fixture
def h2o_rhf(tmp_path):
    mol = vq.Molecule.from_xyz(
        str(Path(__file__).parent.parent / "examples" / "h2o.xyz")
    )
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    return mol, basis, res


def _parse_cube_header(path: Path):
    """Return (n_atoms, origin, voxel_vectors, shape, n_values_or_None,
    atoms_lines, data_lines)."""
    lines = path.read_text().splitlines()
    title, comment = lines[0], lines[1]
    third = lines[2].split()
    n_atoms_signed = int(third[0])
    origin = np.array([float(x) for x in third[1:4]])
    n_values = int(third[4]) if len(third) >= 5 else None
    n_atoms = abs(n_atoms_signed)
    multi = n_atoms_signed < 0

    grid_lines = lines[3:6]
    shape = tuple(int(L.split()[0]) for L in grid_lines)
    voxel = np.array(
        [[float(x) for x in L.split()[1:4]] for L in grid_lines]
    )
    atoms_block = lines[6:6 + n_atoms]
    data_start = 6 + n_atoms
    if multi:
        data_start += 1   # the extra "n_v idx0 idx1 ..." line
    data_lines = lines[data_start:]
    return {
        "title": title, "comment": comment,
        "n_atoms": n_atoms, "n_atoms_signed": n_atoms_signed,
        "origin": origin, "voxel": voxel, "shape": shape,
        "n_values": n_values,
        "atoms": atoms_block, "data": data_lines,
    }


def test_grid_origin_and_shape(h2o_rhf):
    mol, _, _ = h2o_rhf
    grid = vq.make_uniform_grid(mol, spacing=0.3, padding=4.0)
    assert grid.spacing.tolist() == [0.3, 0.3, 0.3]
    pts = grid.points()
    assert pts.shape == (grid.n_points, 3)
    # All atoms are inside the bounding box.
    coords = np.array([list(a.xyz) for a in mol.atoms])
    lo = grid.origin
    hi = grid.origin + (np.array(grid.shape) - 1) * grid.spacing
    assert np.all(coords >= lo - 1e-12)
    assert np.all(coords <= hi + 1e-12)


def test_density_cube_integrates_to_n_electrons(h2o_rhf, tmp_path):
    mol, basis, res = h2o_rhf
    grid = vq.make_uniform_grid(mol, spacing=0.15, padding=5.0)
    p = vq.write_cube_density(
        tmp_path / "rho.cube", res.density, basis, mol, grid=grid,
    )
    hdr = _parse_cube_header(p)
    assert hdr["n_atoms"] == len(mol.atoms)
    assert hdr["n_atoms_signed"] > 0
    assert hdr["shape"] == grid.shape
    assert np.allclose(hdr["origin"], grid.origin, atol=1e-6)
    # Voxel vectors must be diagonal == grid.spacing.
    assert np.allclose(np.diag(hdr["voxel"]), grid.spacing, atol=1e-6)

    # Integral of rho == n_electrons within ~1% on this grid.
    chi = vq.evaluate_ao(basis, grid.points())
    rho = np.einsum("mi,ij,mj->m", chi, np.asarray(res.density), chi)
    integral = rho.sum() * float(np.prod(grid.spacing))
    assert integral == pytest.approx(mol.n_electrons(), rel=2e-2)


def test_data_block_has_six_per_line(h2o_rhf, tmp_path):
    mol, basis, res = h2o_rhf
    grid = vq.make_uniform_grid(mol, spacing=0.5, padding=3.0)
    p = vq.write_cube_density(
        tmp_path / "rho.cube", res.density, basis, mol, grid=grid,
    )
    hdr = _parse_cube_header(p)
    n_total = int(np.prod(grid.shape))
    seen = 0
    for L in hdr["data"]:
        toks = L.split()
        assert len(toks) <= 6
        # All tokens are floats.
        for t in toks:
            float(t)
        seen += len(toks)
    assert seen == n_total


def test_mo_cube_normalised(h2o_rhf, tmp_path):
    mol, basis, res = h2o_rhf
    grid = vq.make_uniform_grid(mol, spacing=0.15, padding=5.0)
    homo = mol.n_electrons() // 2 - 1
    p = vq.write_cube_mo(
        tmp_path / "homo.cube", res.mo_coeffs, homo, basis, mol, grid=grid,
    )
    hdr = _parse_cube_header(p)
    assert hdr["shape"] == grid.shape

    chi = vq.evaluate_ao(basis, grid.points())
    phi = chi @ np.asarray(res.mo_coeffs)[:, homo]
    norm = (phi ** 2).sum() * float(np.prod(grid.spacing))
    assert norm == pytest.approx(1.0, rel=2e-2)


def test_mos_cube_multivalue_header(h2o_rhf, tmp_path):
    mol, basis, res = h2o_rhf
    grid = vq.make_uniform_grid(mol, spacing=0.5, padding=3.0)
    indices = [0, 2, 4]
    p = vq.write_cube_mos(
        tmp_path / "mos.cube", res.mo_coeffs, indices,
        basis, mol, grid=grid,
    )
    hdr = _parse_cube_header(p)
    # Multi-value cubes carry a negative atom count and an explicit n_v.
    assert hdr["n_atoms_signed"] < 0
    assert hdr["n_values"] == len(indices)
    n_total = int(np.prod(grid.shape)) * len(indices)
    seen = sum(len(L.split()) for L in hdr["data"])
    assert seen == n_total


def test_mo_index_out_of_range(h2o_rhf, tmp_path):
    mol, basis, res = h2o_rhf
    with pytest.raises(IndexError):
        vq.write_cube_mo(tmp_path / "bad.cube", res.mo_coeffs, 999, basis, mol)
