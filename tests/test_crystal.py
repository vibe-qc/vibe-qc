"""Crystal + space-group analysis via spglib wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vibeqc import (
    Crystal,
    SymmetriseResult,
    analyze_symmetry,
    read_cif,
    read_poscar,
    spglib_version,
    to_primitive,
    write_poscar,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
DATA = Path(__file__).parent / "data"


def test_spglib_version_nonempty():
    v = spglib_version()
    assert v
    # Expect "major.minor.patch" (tolerate additional suffix).
    parts = v.split(".")
    assert len(parts) >= 2 and parts[0].isdigit()


def test_nacl_primitive_is_fm3m():
    """NaCl rocksalt primitive (2-atom) cell → Fm-3m (#225), 48 operations."""
    crystal = read_poscar(DATA / "NaCl.poscar")
    assert crystal.n_atoms == 2
    sg = analyze_symmetry(crystal)
    assert sg.number == 225
    assert sg.international_symbol.startswith("Fm-3m")
    assert sg.order == 48
    # Na and Cl sit on distinct Wyckoff positions → each is its own
    # representative.
    assert list(sg.equivalent_atoms) == [0, 1]


def test_silicon_diamond_is_fd3m():
    """Silicon diamond primitive → Fd-3m (#227), 48 operations."""
    crystal = read_poscar(DATA / "Si.poscar")
    sg = analyze_symmetry(crystal)
    assert sg.number == 227
    assert sg.international_symbol.startswith("Fd-3m")
    assert sg.order == 48
    # Both Si atoms are symmetry-equivalent (same Wyckoff 8a).
    assert sg.equivalent_atoms[0] == sg.equivalent_atoms[1]


def test_mgo_conventional_reduces_to_2atom_primitive():
    """MgO in an 8-atom conventional cell → 2-atom Fm-3m primitive."""
    conv = read_poscar(DATA / "MgO_conventional.poscar")
    assert conv.n_atoms == 8

    prim = to_primitive(conv)
    assert prim.n_atoms == 2

    # Primitive space group must still be Fm-3m (#225).
    sg = analyze_symmetry(prim)
    assert sg.number == 225

    # Primitive-cell volume is 1/4 of the conventional FCC cell.
    v_conv = abs(np.linalg.det(conv.lattice))
    v_prim = abs(np.linalg.det(prim.lattice))
    assert v_prim == pytest.approx(v_conv / 4.0, rel=1e-9)


def test_identity_is_always_among_operations():
    sg = analyze_symmetry(read_poscar(DATA / "Si.poscar"))
    eye3 = np.eye(3, dtype=int)
    has_identity = any(
        np.array_equal(np.asarray(op.rotation), eye3)
        and np.allclose(np.asarray(op.translation), 0.0)
        for op in sg.operations
    )
    assert has_identity, "Identity operation missing from symmetry list"


def test_symmetry_ops_are_closed_under_composition():
    """The product of any two symmetry ops (rotation parts only) must be
    another rotation in the group. Quick check on the Si diamond lattice."""
    sg = analyze_symmetry(read_poscar(DATA / "Si.poscar"))
    rots = [np.asarray(op.rotation) for op in sg.operations]
    rot_set = {tuple(r.flatten().tolist()) for r in rots}
    # Test just the first 6 × all for speed.
    for A in rots[:6]:
        for B in rots:
            C = A @ B
            assert tuple(C.flatten().tolist()) in rot_set


def test_triclinic_p1_has_only_identity():
    """A random low-symmetry cell should resolve to P1 (#1), 1 op."""
    lattice_bohr = np.array(
        [[3.1 * ANGSTROM_TO_BOHR, 0.3, 0.2],
         [0.4,                   3.3 * ANGSTROM_TO_BOHR, 0.1],
         [0.5,                   0.7, 3.5 * ANGSTROM_TO_BOHR]],
    )
    frac = np.array([[0.0, 0.37], [0.0, 0.21], [0.0, 0.11]])
    crystal = Crystal(lattice_bohr, frac, [1, 6])
    sg = analyze_symmetry(crystal)
    assert sg.number == 1
    assert sg.order == 1


def test_crystal_ctor_mismatched_species_rejected():
    lattice = np.eye(3) * 5.0
    frac = np.array([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]])
    with pytest.raises(RuntimeError, match="fractional_coords"):
        Crystal(lattice, frac, [1])  # 2 positions but 1 species


def test_poscar_roundtrip(tmp_path):
    orig = read_poscar(DATA / "NaCl.poscar")
    out = tmp_path / "out.poscar"
    write_poscar(out, orig)
    got = read_poscar(out)

    assert np.allclose(orig.lattice, got.lattice, atol=1e-12)
    # Column order may differ because write_poscar groups by Z; check the
    # multiset of (Z, fractional coord tuple) matches up to translation.
    def _canonical(c):
        frac = np.asarray(c.fractional_coords) % 1.0
        return sorted(
            (z, round(frac[0, k], 10), round(frac[1, k], 10), round(frac[2, k], 10))
            for k, z in enumerate(c.species)
        )
    assert _canonical(orig) == _canonical(got)


def test_poscar_rejects_vasp4_format(tmp_path):
    """A POSCAR missing the species line (VASP 4) should give a clear error."""
    p = tmp_path / "vasp4.poscar"
    p.write_text(
        "vasp4\n"
        "1.0\n"
        "3.0 0 0\n"
        "0 3.0 0\n"
        "0 0 3.0\n"
        "1\n"                # counts line directly, no species line
        "Direct\n"
        "0.0 0.0 0.0\n"
    )
    with pytest.raises(ValueError, match="VASP 4"):
        read_poscar(p)


def test_cif_reader_p1_explicit_atoms(tmp_path):
    p = tmp_path / "mgo-p1.cif"
    p.write_text(
        "data_mgo\n"
        "_cell_length_a 4.210\n"
        "_cell_length_b 4.210\n"
        "_cell_length_c 4.210\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Mg1 Mg 0 0 0\n"
        "O1 O 0.5 0.5 0.5\n"
    )

    crystal = read_cif(p)

    assert isinstance(crystal, Crystal)
    assert crystal.n_atoms == 2
    assert list(crystal.species) == [12, 8]
    assert np.asarray(crystal.lattice)[0, 0] == pytest.approx(
        4.210 * ANGSTROM_TO_BOHR
    )
    assert np.asarray(crystal.fractional_coords)[0, 1] == pytest.approx(0.5)
    assert Crystal.from_cif(p).n_atoms == 2


def test_cif_reader_expands_symmetry_operations(tmp_path):
    p = tmp_path / "mgo-f-centered.cif"
    p.write_text(
        "data_mgo\n"
        "_cell_length_a 4.210(1)\n"
        "_cell_length_b 4.210(1)\n"
        "_cell_length_c 4.210(1)\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "'x, y, z'\n"
        "'x, y+1/2, z+1/2'\n"
        "'x+1/2, y, z+1/2'\n"
        "'x+1/2, y+1/2, z'\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Mg1 Mg 0 0 0\n"
        "O1 O 0.5(1) 0.5 0.5\n"
    )

    crystal = read_cif(p)

    assert crystal.n_atoms == 8
    assert sorted(crystal.species) == [8, 8, 8, 8, 12, 12, 12, 12]
    sg = analyze_symmetry(crystal)
    assert sg.number == 225


def test_cif_reader_symmetrise_keyword_returns_result(tmp_path):
    p = tmp_path / "mgo-f-centered.cif"
    p.write_text(
        "data_mgo\n"
        "_cell_length_a 4.210\n"
        "_cell_length_b 4.210\n"
        "_cell_length_c 4.210\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "'x, y, z'\n"
        "'x, y+1/2, z+1/2'\n"
        "'x+1/2, y, z+1/2'\n"
        "'x+1/2, y+1/2, z'\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Mg1 Mg 0 0 0\n"
        "O1 O 0.5 0.5 0.5\n"
    )

    result = read_cif(p, symmetrise=True, to_primitive=True)

    assert isinstance(result, SymmetriseResult)
    assert result.report.n_atoms_before == 8
    assert result.report.n_atoms_after == 2


def test_cif_reader_missing_cell_tag_rejected(tmp_path):
    p = tmp_path / "bad.cif"
    p.write_text("data_bad\n_cell_length_a 1\n")

    with pytest.raises(ValueError, match="missing CIF cell tags"):
        read_cif(p)
