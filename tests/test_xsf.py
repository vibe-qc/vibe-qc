"""XSF / BXSF writers — header well-formedness + correct unit conversion.

XSF wants ångström for all coordinates; BXSF wants 1/ångström and eV.
These tests check both conventions, plus the data-block traversal
order (XSF expects the first grid index to run fastest).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq


_BOHR_TO_ANGSTROM = 0.529177210903
_HARTREE_TO_EV = 27.211386245988


@pytest.fixture
def cubic_2atom():
    """A trivial 3D cubic cell with two atoms."""
    a = 5.6  # bohr
    atoms = [vq.Atom(11, [0, 0, 0]), vq.Atom(17, [a / 2, a / 2, a / 2])]
    return vq.PeriodicSystem(3, [[a, 0, 0], [0, a, 0], [0, 0, a]], atoms)


@pytest.fixture
def chain_2atom():
    """A 1D H2 chain — useful for non-cubic span tests."""
    a = 6.0
    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])]
    return vq.PeriodicSystem(1, [[a, 0, 0], [0, 30, 0], [0, 0, 30]], atoms)


def _graphene_slab(a3_bohr: float):
    """A dim=2 slab; lattice column 2 is synthesized padding, not a cell edge."""
    a = 4.6487  # bohr
    lattice = np.column_stack([
        [a, 0.0, 0.0],
        [0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
        [0.0, 0.0, a3_bohr],
    ])
    atoms = [vq.Atom(6, [0.0, 0.0, 0.0]), vq.Atom(6, [0.5 * a, 0.0, 0.0])]
    return vq.PeriodicSystem(2, lattice, atoms)


@pytest.fixture
def graphene_slab():
    return _graphene_slab(30.0)


@pytest.fixture
def hexagonal_1atom():
    """A skew hexagonal cell; lattice vectors are columns."""
    a = 5.0
    c = 18.0
    lattice = np.column_stack([
        [a, 0.0, 0.0],
        [0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
        [0.0, 0.0, c],
    ])
    return vq.PeriodicSystem(3, lattice, [vq.Atom(13, [0.0, 0.0, 0.0])])


def test_xsf_structure_units_and_layout(cubic_2atom, tmp_path):
    p = vq.write_xsf_structure(tmp_path / "s.xsf", cubic_2atom)
    text = p.read_text()
    assert "CRYSTAL" in text
    assert "PRIMVEC" in text
    assert "PRIMCOORD" in text

    # Lattice vector along x must equal 5.6 bohr in ångström.
    a_ang = 5.6 * _BOHR_TO_ANGSTROM
    primvec_line = text.splitlines()[2]   # first PRIMVEC row
    a_x = float(primvec_line.split()[0])
    assert a_x == pytest.approx(a_ang, rel=1e-6)

    # Atom Z values appear and are correct (Na, Cl).
    atom_block = text.splitlines()[7:9]
    assert int(atom_block[0].split()[0]) == 11
    assert int(atom_block[1].split()[0]) == 17


def test_xsf_structure_keyword_tracks_dimensionality(
    cubic_2atom, graphene_slab, chain_2atom, tmp_path,
):
    for system, keyword in [
        (cubic_2atom, "CRYSTAL"),
        (graphene_slab, "SLAB"),
        (chain_2atom, "POLYMER"),
    ]:
        p = vq.write_xsf_structure(tmp_path / f"{keyword}.xsf", system)
        assert p.read_text().splitlines()[0] == keyword


def test_xsf_volume_keyword_tracks_dimensionality(graphene_slab, tmp_path):
    data = np.zeros((3, 3, 3))
    p = vq.write_xsf_volume(
        tmp_path / "slab_vol.xsf", graphene_slab, data=data, name="rho",
    )
    assert p.read_text().splitlines()[0] == "SLAB"


def test_xsf_keyword_is_independent_of_lattice_geometry(tmp_path):
    """A synthesized a3 is indistinguishable from a real vacuum gap.

    The keyword must come from `dim` alone: growing a slab's non-physical a3
    must not turn it into a crystal, and a genuinely 3D cell with a long c
    must not turn into a slab.
    """
    for a3 in (30.0, 80.0):
        p = vq.write_xsf_structure(tmp_path / f"slab{a3}.xsf", _graphene_slab(a3))
        assert p.read_text().splitlines()[0] == "SLAB"

    tall_bulk = vq.PeriodicSystem(
        3, [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 80.0]], [vq.Atom(13, [0.0, 0.0, 0.0])],
    )
    p = vq.write_xsf_structure(tmp_path / "tall.xsf", tall_bulk)
    assert p.read_text().splitlines()[0] == "CRYSTAL"


def test_xsf_structure_writes_lattice_columns_for_hexagonal_cells(
    hexagonal_1atom, tmp_path,
):
    p = vq.write_xsf_structure(tmp_path / "hex.xsf", hexagonal_1atom)
    lines = p.read_text().splitlines()

    primvec = np.array([
        [float(x) for x in lines[i].split()]
        for i in range(2, 5)
    ])
    expected = np.asarray(hexagonal_1atom.lattice).T * _BOHR_TO_ANGSTROM
    np.testing.assert_allclose(primvec, expected, atol=1.0e-8)


def test_xsf_volume_data_count(chain_2atom, tmp_path):
    n = (4, 5, 6)
    rng = np.random.default_rng(0)
    data = rng.standard_normal(n)
    p = vq.write_xsf_volume(
        tmp_path / "vol.xsf", chain_2atom, data=data, name="rho",
    )
    text = p.read_text()
    assert "BEGIN_DATAGRID_3D_rho" in text
    assert "END_DATAGRID_3D_rho" in text

    # Count floats inside the data grid block.
    lines = text.splitlines()
    start = lines.index("BEGIN_DATAGRID_3D_rho")
    end = lines.index("END_DATAGRID_3D_rho")
    # Data lines start after the 5 header lines (n1 n2 n3, origin, 3× span).
    data_lines = lines[start + 6:end]
    n_floats = sum(len(L.split()) for L in data_lines)
    assert n_floats == int(np.prod(n))


def test_xsf_volume_traversal_first_index_fastest(cubic_2atom, tmp_path):
    """XSF wants i (first index) running fastest. Sentinel data with
    distinct values in each axis lets us verify."""
    n1, n2, n3 = 2, 3, 4
    data = np.zeros((n1, n2, n3))
    # Make every voxel uniquely identifiable.
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                data[i, j, k] = 100 * i + 10 * j + k
    p = vq.write_xsf_volume(
        tmp_path / "vol.xsf", cubic_2atom, data=data, name="s",
    )
    text = p.read_text()
    lines = text.splitlines()
    start = lines.index("BEGIN_DATAGRID_3D_s")
    end = lines.index("END_DATAGRID_3D_s")
    data_lines = lines[start + 6:end]
    flat = np.array([float(t) for L in data_lines for t in L.split()])
    # Expected order: k slowest, then j, then i fastest.
    expected = np.transpose(data, (2, 1, 0)).ravel()
    assert np.allclose(flat, expected)


def test_xsf_volume_default_span_uses_lattice_columns(hexagonal_1atom, tmp_path):
    data = np.zeros((2, 2, 2))
    p = vq.write_xsf_volume(
        tmp_path / "hex_vol.xsf", hexagonal_1atom, data=data, name="rho",
    )
    lines = p.read_text().splitlines()
    start = lines.index("BEGIN_DATAGRID_3D_rho")
    span = np.array([
        [float(x) for x in lines[start + i].split()]
        for i in range(3, 6)
    ])
    expected = np.asarray(hexagonal_1atom.lattice).T * _BOHR_TO_ANGSTROM
    np.testing.assert_allclose(span, expected, atol=1.0e-8)


def test_bxsf_units_and_shape(cubic_2atom, tmp_path):
    nkx, nky, nkz, nb = 3, 3, 3, 2
    energies = np.zeros((nkx, nky, nkz, nb))
    energies[..., 0] = -0.1   # Hartree
    energies[..., 1] = +0.1
    p = vq.write_bxsf(
        tmp_path / "x.bxsf", cubic_2atom, energies, e_fermi=0.0,
    )
    text = p.read_text()
    assert "BEGIN_INFO" in text
    assert "BANDGRID_3D_BANDS" in text

    # First band: every value must be -0.1 Ha = -2.7211 eV.
    lines = text.splitlines()
    band1_idx = next(i for i, L in enumerate(lines) if L.strip().startswith("BAND:") and L.strip().endswith("1"))
    band2_idx = next(i for i, L in enumerate(lines[band1_idx + 1:], band1_idx + 1) if L.strip().startswith("BAND:"))
    band1_data = np.array([
        float(t) for L in lines[band1_idx + 1:band2_idx] for t in L.split()
    ])
    assert band1_data.size == nkx * nky * nkz
    assert np.allclose(band1_data, -0.1 * _HARTREE_TO_EV, atol=1e-4)

    # Reciprocal lattice line: |b| = 2π / a in 1/ångström.
    a_ang = 5.6 * _BOHR_TO_ANGSTROM
    expected_b = 2.0 * np.pi / a_ang
    span_lines = lines[7:10]   # after fermi block + headers + nbands + nkmesh + origin
    # Find the spanning vectors: they're after the "0.0 0.0 0.0" origin line.
    origin_idx = next(i for i, L in enumerate(lines) if L.strip().startswith("0.0 0.0 0.0"))
    bvec1 = float(lines[origin_idx + 1].split()[0])
    assert bvec1 == pytest.approx(expected_b, rel=1e-6)


def test_bxsf_writes_reciprocal_lattice_columns_for_hexagonal_cells(
    hexagonal_1atom, tmp_path,
):
    energies = np.zeros((2, 2, 2, 1))
    p = vq.write_bxsf(
        tmp_path / "hex.bxsf", hexagonal_1atom, energies, e_fermi=0.0,
    )
    lines = p.read_text().splitlines()
    origin_idx = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith("0.0 0.0 0.0")
    )
    span = np.array([
        [float(x) for x in lines[origin_idx + i].split()]
        for i in range(1, 4)
    ])
    L_ang = np.asarray(hexagonal_1atom.lattice) * _BOHR_TO_ANGSTROM
    expected = (2.0 * np.pi * np.linalg.inv(L_ang).T).T
    np.testing.assert_allclose(span, expected, atol=1.0e-8)
