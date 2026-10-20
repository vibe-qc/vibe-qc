"""The engine's public API as a consumer such as vibe-sketch calls it.

A smoke test of the packaged engine, not of the chemistry: the naming
rules are covered by the repository's ``tests/test_naming_*`` suites.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import vibeqc_naming
from vibeqc_naming import (
    MolecularGraph,
    NamedResult,
    name_from_atoms,
    name_from_atoms_detailed,
    name_from_graph,
    name_from_graph_detailed,
    structure_from_name,
)
from vibeqc_naming.smiles import smiles_from_name

WATER = [(8, 0.0, 0.0, 0.117), (1, 0.0, 0.757, -0.469), (1, 0.0, -0.757, -0.469)]


def _saturated(heavy, heavy_bonds, hydrogens):
    """Explicit hydrogens appended to a heavy-atom skeleton."""
    elements = list(heavy)
    bonds = list(heavy_bonds)
    for atom, count in enumerate(hydrogens):
        for _ in range(count):
            bonds.append((atom, len(elements), 1))
            elements.append("H")
    return elements, bonds


# CC(O)CC
BUTAN_2_OL = _saturated(
    ["C", "C", "O", "C", "C"],
    [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1)],
    [3, 1, 1, 2, 3],
)


def test_import_does_not_pull_in_vibeqc():
    """The engine runs without the compiled core, so importing it must not import ``vibeqc``."""
    package_dir = Path(vibeqc_naming.__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", "import sys, vibeqc_naming; print('vibeqc' in sys.modules)"],
        env={**os.environ, "PYTHONPATH": str(package_dir)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_structure_to_name():
    assert name_from_atoms(WATER) == "water"
    result = name_from_atoms_detailed(WATER)
    assert isinstance(result, NamedResult)
    assert result.name == "water"
    assert result.source is not None
    assert result.confidence is not None


def test_name_to_structure_and_smiles():
    pyridine = structure_from_name("pyridine")
    heavy = Counter(z for z, *_ in pyridine if z != 1)
    assert heavy == {6: 5, 7: 1}
    assert smiles_from_name("ethanol") == "CCO"


def test_graph_from_explicit_bonds_carries_a_descriptor():
    elements, bonds = BUTAN_2_OL
    assert name_from_graph(MolecularGraph.from_bonds(elements, bonds)) == "butan-2-ol"
    labelled = MolecularGraph.from_bonds(elements, bonds, atom_stereo={1: "R"})
    assert name_from_graph(labelled) == "(2R)-butan-2-ol"
    assert name_from_graph_detailed(labelled).name == "(2R)-butan-2-ol"
