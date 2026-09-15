"""Molecule geometry handling and XYZ parsing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import Atom, Molecule, PeriodicSystem, SymmetriseResult, from_xyz

from .conftest import ANGSTROM_TO_BOHR


def test_atom_construction():
    a = Atom(8, [0.0, 0.0, 0.0])
    assert a.Z == 8
    assert list(a.xyz) == [0.0, 0.0, 0.0]


def test_molecule_electron_count_neutral():
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 0.0, 1.8]),
                    Atom(1, [0.0, 1.8, 0.0])])
    assert mol.n_electrons() == 10
    assert mol.charge == 0
    assert mol.multiplicity == 1


def test_molecule_electron_count_cation():
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0])], charge=1, multiplicity=4)
    # Neutral O has 8 electrons; O+ has 7 (triplet or quartet depending).
    assert mol.n_electrons() == 7


def test_periodic_unit_cell_molecule_auto_doublet_for_odd_z_metals():
    for z in (47, 79):
        system = PeriodicSystem(
            3,
            np.eye(3) * 10.0,
            [Atom(z, [0.0, 0.0, 0.0])],
        )

        mol = system.unit_cell_molecule()

        assert mol.n_electrons() == z
        assert mol.multiplicity == 2


def test_periodic_unit_cell_molecule_preserves_compatible_multiplicity():
    system = PeriodicSystem(
        3,
        np.eye(3) * 10.0,
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        0,
        3,
    )

    mol = system.unit_cell_molecule()

    assert mol.n_electrons() == 2
    assert mol.multiplicity == 3


def test_molecule_rejects_negative_multiplicity():
    with pytest.raises(ValueError, match="multiplicity"):
        Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
                 multiplicity=0)


def test_molecule_rejects_inconsistent_mult_and_n_elec():
    # 2 electrons -> can be singlet (mult 1) or triplet (mult 3); doublet
    # (mult 2) would require 1 or 3 electrons.
    with pytest.raises(ValueError, match="inconsistent"):
        Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
                 multiplicity=2)


def test_nuclear_repulsion_h2():
    # Two protons at R = 1.4 bohr -> E_nuc = 1/1.4 Ha
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    assert mol.nuclear_repulsion() == pytest.approx(1.0 / 1.4, rel=1e-12)


def test_nuclear_repulsion_rejects_coincident_atoms():
    with pytest.raises(RuntimeError, match="coincide"):
        Molecule([Atom(1, [0.0, 0.0, 0.0]),
                  Atom(1, [0.0, 0.0, 0.0])]).nuclear_repulsion()


def test_xyz_parser_reads_positions_in_bohr(tmp_path):
    f = tmp_path / "h2.xyz"
    # 1.4 bohr * 0.529177210903 A/bohr = 0.7408480953 A
    f.write_text("2\ntitle line\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7408480953\n")
    mol = from_xyz(f)
    assert len(mol.atoms) == 2
    z_distance = mol.atoms[1].xyz[2] - mol.atoms[0].xyz[2]
    assert z_distance == pytest.approx(1.4, abs=1e-9)


def test_xyz_parser_case_insensitive_symbols(tmp_path):
    f = tmp_path / "h2o.xyz"
    f.write_text("3\ntitle\nO 0 0 0\nh 0 0 1\nH 0 1 0\n")
    mol = from_xyz(f)
    assert [a.Z for a in mol.atoms] == [8, 1, 1]


def test_xyz_parser_unknown_element_errors(tmp_path):
    f = tmp_path / "bad.xyz"
    f.write_text("1\ntitle\nXx 0 0 0\n")
    with pytest.raises(ValueError, match="unknown element"):
        from_xyz(f)


def test_xyz_parser_wrong_atom_count_errors(tmp_path):
    f = tmp_path / "bad.xyz"
    f.write_text("3\ntitle\nH 0 0 0\nH 0 0 1\n")  # claims 3, has 2
    with pytest.raises(ValueError, match="expected 3"):
        from_xyz(f)


def test_xyz_parser_charge_and_mult_arguments(tmp_path):
    f = tmp_path / "oh-minus.xyz"
    # OH- (10 electrons, neutral OH would have 9): charge=-1, mult=1
    f.write_text("2\nhydroxide\nO 0 0 0\nH 0 0 0.97\n")
    mol = from_xyz(f, charge=-1, multiplicity=1)
    assert mol.n_electrons() == 10
    assert mol.charge == -1


def test_extended_xyz_parser_returns_periodic_system(tmp_path):
    f = tmp_path / "mgo.xyz"
    f.write_text(
        "2\n"
        'Lattice="4.21 0 0 0 4.21 0 0 0 4.21" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Mg 0 0 0\n"
        "O 2.105 2.105 2.105\n"
    )

    system = from_xyz(f, periodic=True)

    assert isinstance(system, PeriodicSystem)
    assert system.dim == 3
    assert [atom.Z for atom in system.unit_cell] == [12, 8]
    assert np.asarray(system.lattice)[0, 0] == pytest.approx(
        4.21 * ANGSTROM_TO_BOHR
    )
    assert system.unit_cell[1].xyz[0] == pytest.approx(2.105 * ANGSTROM_TO_BOHR)


def test_extended_xyz_parser_periodic_mode_requires_lattice(tmp_path):
    f = tmp_path / "h2.xyz"
    f.write_text("2\nplain\nH 0 0 0\nH 0 0 0.74\n")

    with pytest.raises(ValueError, match="Lattice"):
        from_xyz(f, periodic=True)


def test_xyz_parser_explicit_lattice_selects_periodic_mode(tmp_path):
    f = tmp_path / "h2-chain.xyz"
    f.write_text("2\nplain\nH 0 0 0\nH 0 0 0.74\n")

    system = from_xyz(
        f,
        lattice=[[2.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
        dim=1,
    )

    assert isinstance(system, PeriodicSystem)
    assert system.dim == 1
    assert np.asarray(system.lattice)[0, 0] == pytest.approx(
        2.0 * ANGSTROM_TO_BOHR
    )


def test_extended_xyz_pbc_sets_prefix_dimensionality(tmp_path):
    f = tmp_path / "slab.xyz"
    f.write_text(
        "1\n"
        'Lattice="3 0 0 0 4 0 0 0 30" '
        'Properties=species:S:1:pos:R:3 pbc="T T F"\n'
        "C 0 0 0\n"
    )

    system = from_xyz(f, periodic=True)

    assert isinstance(system, PeriodicSystem)
    assert system.dim == 2


def test_extended_xyz_nonprefix_pbc_rejected(tmp_path):
    f = tmp_path / "bad.xyz"
    f.write_text(
        "1\n"
        'Lattice="3 0 0 0 4 0 0 0 30" '
        'Properties=species:S:1:pos:R:3 pbc="T F T"\n'
        "C 0 0 0\n"
    )

    with pytest.raises(ValueError, match="prefix-periodic"):
        from_xyz(f, periodic=True)


def test_extended_xyz_symmetrise_keyword_returns_result(tmp_path):
    f = tmp_path / "mgo.xyz"
    f.write_text(
        "2\n"
        'Lattice="4.21 0 0 0 4.21 0 0 0 4.21" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Mg 0 0 0\n"
        "O 2.105 2.105 2.105\n"
    )

    result = from_xyz(f, symmetrise=True)

    assert isinstance(result, SymmetriseResult)
    assert result.system.symmetry is not None
    assert result.report.n_atoms_before == 2
    assert result.report.n_atoms_after == 2
