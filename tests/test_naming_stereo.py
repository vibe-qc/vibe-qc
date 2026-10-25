"""Graphs from explicit bonds, and caller-supplied stereodescriptors in names (#537).

``MolecularGraph.from_bonds`` builds the graph a sketcher already knows
instead of perceiving bonds from coordinates. Optional R/S and E/Z labels
from the caller are validated, never computed, and appear in front of the
name with their locants. Names without descriptors do not change.
"""

from __future__ import annotations

import importlib

import pytest

from vibeqc_naming import MolecularGraph, build_graph, from_atoms_list, structure_from_name


@pytest.fixture(params=["vibeqc_naming.iupac_name", "vibeqc.naming.iupac_name"])
def naming_module(request):
    return importlib.import_module(request.param)


def _saturated(heavy, heavy_bonds, hydrogens):
    """Elements and bonds with explicit hydrogens appended after the heavy atoms."""
    elements = list(heavy)
    bonds = list(heavy_bonds)
    for atom, count in enumerate(hydrogens):
        for _ in range(count):
            bonds.append((atom, len(elements), 1))
            elements.append("H")
    return elements, bonds


# C[C@@H](O)CC, RDKit CIP label R on atom 1 (the #537 example).
BUTAN_2_OL = _saturated(
    ["C", "C", "O", "C", "C"],
    [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1)],
    [3, 1, 1, 2, 3],
)
# CC=CC
BUT_2_ENE = _saturated(["C", "C", "C", "C"], [(0, 1, 1), (1, 2, 2), (2, 3, 1)], [3, 1, 1, 3])
# C=CCC: terminal CH2 cannot carry E/Z
BUT_1_ENE = _saturated(["C", "C", "C", "C"], [(0, 1, 2), (1, 2, 1), (2, 3, 1)], [2, 1, 2, 3])
# Kekule benzene
BENZENE = _saturated(["C"] * 6, [(i, (i + 1) % 6, 2 if i % 2 == 0 else 1) for i in range(6)], [1] * 6)


def test_from_bonds_reproduces_the_coordinate_graph():
    """Same adjacency, rings, formula and electron count as bond perception."""
    for name in ("ethanol", "acetone", "benzene", "cyclohexane"):
        atoms = structure_from_name(name)
        perceived = from_atoms_list(atoms)
        given = MolecularGraph.from_bonds(
            [z for z, *_ in atoms],
            [(b.i, b.j, b.order) for b in perceived.bonds],
            positions=[xyz for _, *xyz in atoms],
        )
        assert given.formula == perceived.formula
        assert given.n_electrons == perceived.n_electrons
        assert given.is_organic == perceived.is_organic
        assert [sorted(n) for n in given.adjacency] == [sorted(n) for n in perceived.adjacency]
        assert sorted(sorted(r) for r in given.rings) == sorted(sorted(r) for r in perceived.rings)
        assert [b.distance for b in given.bonds] == pytest.approx(
            [b.distance for b in perceived.bonds], abs=1e-12
        )


def test_from_bonds_takes_symbols_or_numbers_and_needs_no_coordinates(naming_module):
    elements, bonds = BUTAN_2_OL
    by_symbol = MolecularGraph.from_bonds(elements, bonds)
    by_number = MolecularGraph.from_bonds([{"C": 6, "O": 8, "H": 1}[e] for e in elements], bonds)
    assert by_symbol.formula == by_number.formula == "C4H10O"
    assert naming_module.name_from_graph(by_symbol) == "butan-2-ol"
    assert all(b.distance == 0.0 for b in by_symbol.bonds)
    benzene = MolecularGraph.from_bonds(*BENZENE)
    assert naming_module.name_from_graph(benzene) == "benzene"


@pytest.mark.parametrize("label", ["R", "S"])
def test_butan_2_ol_carries_its_cip_label(naming_module, label):
    elements, bonds = BUTAN_2_OL
    graph = MolecularGraph.from_bonds(elements, bonds, atom_stereo={1: label})
    assert naming_module.name_from_graph(graph) == f"(2{label})-butan-2-ol"
    detailed = naming_module.name_from_graph_detailed(graph)
    assert detailed.name == f"(2{label})-butan-2-ol"


@pytest.mark.parametrize("label", ["E", "Z"])
@pytest.mark.parametrize("key", [(1, 2), (2, 1)])
def test_but_2_ene_carries_its_double_bond_label(naming_module, label, key):
    elements, bonds = BUT_2_ENE
    graph = MolecularGraph.from_bonds(elements, bonds, bond_stereo={key: label})
    assert graph.bond_stereo == {(1, 2): label}
    assert naming_module.name_from_graph(graph) == f"(2{label})-but-2-ene"


def test_names_without_descriptors_are_unchanged(naming_module):
    assert naming_module.name_from_graph(MolecularGraph.from_bonds(*BUTAN_2_OL)) == "butan-2-ol"
    assert naming_module.name_from_graph(MolecularGraph.from_bonds(*BUT_2_ENE)) == "but-2-ene"
    for name in ("ethanol", "acetone", "benzene"):
        assert naming_module.name_from_graph(from_atoms_list(structure_from_name(name))) == name


@pytest.mark.parametrize(
    "atom_stereo, message",
    [
        ({0: "R"}, "carries 3 hydrogens"),
        ({3: "S"}, "carries 2 hydrogens"),
        ({2: "R"}, "has 2 neighbours"),
        ({1: "X"}, "not one of R, S, r, s"),
        ({99: "R"}, "not an atom index"),
    ],
)
def test_atom_labels_that_cannot_apply_are_refused(atom_stereo, message):
    with pytest.raises(ValueError, match=message):
        MolecularGraph.from_bonds(*BUTAN_2_OL, atom_stereo=atom_stereo)


@pytest.mark.parametrize(
    "molecule, bond_stereo, message",
    [
        (BUT_2_ENE, {(0, 1): "E"}, "not joined by a double bond"),
        (BUT_2_ENE, {(1, 2): "cis"}, "not E or Z"),
        (BUT_1_ENE, {(0, 1): "Z"}, "not both hydrogen"),
        (BUT_2_ENE, {(1,): "E"}, "atom pair"),
    ],
)
def test_bond_labels_that_cannot_apply_are_refused(molecule, bond_stereo, message):
    with pytest.raises(ValueError, match=message):
        MolecularGraph.from_bonds(*molecule, bond_stereo=bond_stereo)


def test_a_label_on_a_name_without_locants_is_refused(naming_module):
    """Cyclohexanol's C1 passes the structural check, but the catalog records no
    locants for cyclohexanol, so the label has nowhere to go."""
    elements, bonds = _saturated(
        ["O", "C", "C", "C", "C", "C", "C"],
        [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1), (6, 1, 1)],
        [1, 1, 2, 2, 2, 2, 2],
    )
    graph = MolecularGraph.from_bonds(elements, bonds, atom_stereo={1: "R"})
    with pytest.raises(ValueError, match="no locant numbering is recorded"):
        naming_module.name_from_graph(graph)


def test_labels_set_after_construction_are_checked_at_naming(naming_module):
    graph = MolecularGraph.from_bonds(*BUTAN_2_OL)
    graph.atom_stereo = {1: "S"}
    assert naming_module.name_from_graph(graph) == "(2S)-butan-2-ol"
    graph.atom_stereo = {1: "Q"}
    with pytest.raises(ValueError, match="not one of"):
        naming_module.name_from_graph(graph)
    graph.atom_stereo = {}
    graph.bond_stereo = {(1, 3): "E"}
    with pytest.raises(ValueError, match="not joined by a double bond"):
        naming_module.name_from_graph(graph)


def test_a_coordinate_built_graph_has_no_descriptors_and_refuses_them_off_catalog(naming_module):
    """Coordinate graphs never gain labels; a label on a non-organic graph is refused."""
    graph = from_atoms_list(structure_from_name("ethanol"))
    assert graph.atom_stereo == {} and graph.bond_stereo == {}
    water = MolecularGraph.from_bonds(["O", "H", "H"], [(0, 1, 1), (0, 2, 1)])
    water.atom_stereo = {0: "R"}
    with pytest.raises(ValueError, match="only on catalog organic names"):
        naming_module.name_from_graph(water)


@pytest.mark.parametrize(
    "elements, bonds, kwargs, message",
    [
        (["C", "C"], [(0, 1, 1.5)], {}, "Kekule"),
        (["C", "C"], [(0, 0, 1)], {}, "itself"),
        (["C", "C"], [(0, 1, 1), (1, 0, 1)], {}, "given twice"),
        (["C", "C"], [(0, 2, 1)], {}, "not an atom index"),
        (["C", "Xx"], [(0, 1, 1)], {}, "unknown element symbol"),
        (["C", "C"], [(0, 1, 1)], {"positions": [(0.0, 0.0, 0.0)]}, "one \\(x, y, z\\)"),
    ],
)
def test_malformed_input_is_refused(elements, bonds, kwargs, message):
    with pytest.raises(ValueError, match=message):
        MolecularGraph.from_bonds(elements, bonds, **kwargs)


def test_build_graph_is_unchanged_by_the_shared_assembly():
    """build_graph now delegates to the same assembly as from_bonds."""
    atoms = structure_from_name("benzene")
    bohr = [(x / 0.529177210903, y / 0.529177210903, z / 0.529177210903) for _, x, y, z in atoms]
    graph = build_graph([z for z, *_ in atoms], bohr)
    assert graph.formula == "C6H6"
    assert graph.atom_stereo == {} and graph.bond_stereo == {}
    assert len(graph.rings) == 1 and len(graph.rings[0]) == 6
