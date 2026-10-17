"""Value-pinning tests for binary/compositional IUPAC naming.

The rewrite in commit ``99e50629`` inverted the element ordering in
``_build_compositional_name`` (electronegative element named first instead
of last) and dropped single-count element names to a bare lowercased
symbol — so ``H2O`` named as ``"o dihydride"`` rather than
``"dihydrogen oxide"``. The follow-up ``e28bcf23`` repaired the syntax and
circular import but deliberately left value correctness unpinned
(``tests/test_naming_iupac_importable.py`` only guards importability and
module scope). These tests close that gap.

The user-facing path is the standalone ``vibeqc_naming`` package
(``vibeqc.naming.report`` calls ``vibeqc_naming.name_from_atoms``); the
``vibeqc.naming.iupac_name`` shim carries a parallel copy, so we pin both
and assert they stay in lockstep.

Reference: IUPAC Red Book 2005, IR-4.4.2.2 (electropositive constituent
named first; electronegative constituent takes the ``-ide`` suffix).
"""

from __future__ import annotations

import importlib
import json
import zipfile

import pytest


@pytest.fixture(params=["vibeqc_naming.iupac_name", "vibeqc.naming.iupac_name"])
def naming_module(request):
    return importlib.import_module(request.param)


_PERIODIC_CASES = [
    pytest.param(
        [(11, 0.0, 0.0, 0.0), (17, 3.37, 3.37, 3.37)],
        [(6.74, 0, 0), (0, 6.74, 0), (0, 0, 6.74)],
        (True, True, True), "sodium chloride", "ClNa", "medium",
        id="bulk-3d",
    ),
    pytest.param(
        [(5, 0.0, 0.0, 0.0), (7, 1.45, 0.0, 0.0)],
        [(2.51, 0, 0), (-1.255, 2.174, 0), (0, 0, 20)],
        (True, True, False), "boron nitride slab", "BN", "medium",
        id="small-slab-2d",
    ),
    pytest.param(
        [(6, 0.0, 0.0, 0.0), (6, 1.42, 0.0, 0.0)],
        [(2.46, 0, 0), (-1.23, 2.13, 0), (0, 0, 20)],
        (True, True, False), "carbon slab", "C2", "medium",
        id="elemental-slab-2d",
    ),
    pytest.param(
        [(29, 2.0 * x, 2.0 * y, 0.0) for x in range(4) for y in range(2)],
        [(8, 0, 0), (0, 4, 0), (0, 0, 20)],
        (True, True, False), "copper slab", "Cu8", "medium",
        id="recognized-slab-2d",
    ),
]


@pytest.mark.parametrize("atoms,lattice,pbc,name,formula,confidence", _PERIODIC_CASES)
def test_lattice_naming(naming_module, atoms, lattice, pbc, name, formula, confidence):
    result = naming_module.name_from_atoms_with_lattice(atoms, lattice, pbc=pbc)
    assert isinstance(result, naming_module.NamedResult)
    assert result.name == name
    assert result.confidence == naming_module.Confidence(confidence)
    assert result.source == naming_module.NamingSource.INORGANIC_COMPOSITIONAL
    assert result.formula


@pytest.mark.parametrize("atoms,lattice,pbc,name,formula,confidence", _PERIODIC_CASES)
def test_periodic_qvf_naming(
    naming_module, tmp_path, atoms, lattice, pbc, name, formula, confidence
):
    # Self-contained archives exercise the public reader without the optional
    # conformance corpus, viewer, writer, or compiled extension.
    from vibeqc_naming.molecular_graph import ATOMIC_NUMBER_TO_SYMBOL

    path = tmp_path / "structure.qvf"
    structure = {
        "atoms": [
            {"atomic_number": z, "symbol": ATOMIC_NUMBER_TO_SYMBOL[z],
             "position": [x, y, z_coord]}
            for z, x, y, z_coord in atoms
        ],
        "pbc": pbc,
        "lattice_vectors": lattice,
    }
    manifest = {"sections": [{"id": "structure", "members": {
        "structure": {"path": "structure.json"}
    }}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("structure.json", json.dumps(structure))

    result = naming_module.name_from_qvf(path)
    assert isinstance(result, naming_module.NamedResult)
    assert result.name == name
    assert result.formula == formula
    assert result.confidence == naming_module.Confidence(confidence)
    assert result.source == naming_module.NamingSource.INORGANIC_COMPOSITIONAL


@pytest.mark.parametrize("lattice", [None, [(10, 0, 0), (0, 10, 0), (0, 0, 10)]])
@pytest.mark.parametrize("prefer_trivial", [True, False])
def test_lattice_fallback_preserves_naming_preference(naming_module, lattice, prefer_trivial):
    atoms = [(8, 0.0, 0.0, 0.0), (1, 0.76, 0.0, 0.59), (1, -0.76, 0.0, 0.59)]
    expected = naming_module.name_from_atoms_detailed(atoms, prefer_trivial=prefer_trivial)
    result = naming_module.name_from_atoms_with_lattice(
        atoms, lattice, prefer_trivial=prefer_trivial
    )
    assert result.name == expected.name
    assert result.formula == expected.formula
    if lattice is None:
        assert result == expected
    else:
        assert result.dimensionality == 3
        assert result.is_trivial == prefer_trivial

# The composed compositional name for each formula. These are the values a
# reader of the SCF ``.out`` would see for an un-named (non-trivial) binary.
_EXPECTED: dict[str, str] = {
    "H2O": "dihydrogen oxide",
    "CO2": "carbon dioxide",
    "NaCl": "sodium chloride",
    "Fe2O3": "diiron trioxide",
    "NH3": "trihydrogen nitride",
    "HCl": "hydrogen chloride",
    "SO3": "sulfur trioxide",
    "MgO": "magnesium oxide",
    "CaF2": "calcium difluoride",
    "Al2O3": "dialuminium trioxide",
    "LiH": "lithium hydride",
    "N2O": "dinitrogen oxide",
    "BN": "boron nitride",
    "SiC": "silicon carbide",
}


@pytest.mark.parametrize("formula, expected", sorted(_EXPECTED.items()))
def test_standalone_compositional_values(formula: str, expected: str) -> None:
    """The user-facing ``vibeqc_naming`` path names binaries correctly."""
    from vibeqc_naming.iupac_name import _compositional_name_from_formula

    assert _compositional_name_from_formula(formula) == expected


@pytest.mark.parametrize("formula, expected", sorted(_EXPECTED.items()))
def test_shim_compositional_values(formula: str, expected: str) -> None:
    """The ``vibeqc.naming.iupac_name`` shim names binaries correctly."""
    from vibeqc.naming.iupac_name import _compositional_name_from_formula

    assert _compositional_name_from_formula(formula) == expected


def test_electropositive_element_is_named_first() -> None:
    """Red Book IR-4.4.2.2: electropositive constituent precedes the -ide."""
    from vibeqc_naming.iupac_name import _compositional_name_from_formula

    # In every binary the electropositive element leads and the
    # electronegative one closes with an -ide suffix — never the reverse.
    name = _compositional_name_from_formula("NaCl")
    assert name.split() == ["sodium", "chloride"]
    assert not name.startswith("chlor")

    water = _compositional_name_from_formula("H2O")
    assert water.startswith("dihydrogen")
    assert water.endswith("oxide")


def test_shim_matches_standalone() -> None:
    """The shim and the standalone package name binaries identically."""
    from vibeqc.naming.iupac_name import (
        _compositional_name_from_formula as shim_name,
    )
    from vibeqc_naming.iupac_name import (
        _compositional_name_from_formula as std_name,
    )

    for formula in _EXPECTED:
        assert shim_name(formula) == std_name(formula), formula


# ---------------------------------------------------------------------------
# #538: RDKit has no structure-to-name function, so the external chain must
# not pretend to try a local RDKit backend.
# ---------------------------------------------------------------------------

_EXTERNAL_MODULES = ["vibeqc_naming.external", "vibeqc.naming.external"]
_WATER = [(8, 0.0, 0.0, 0.117), (1, 0.0, 0.757, -0.469), (1, 0.0, -0.757, -0.469)]
_METHANE = [
    (6, 0.0, 0.0, 0.0),
    (1, 0.629, 0.629, 0.629),
    (1, -0.629, -0.629, 0.629),
    (1, -0.629, 0.629, -0.629),
    (1, 0.629, -0.629, -0.629),
]


@pytest.mark.parametrize("module_name", _EXTERNAL_MODULES)
def test_name_from_rdkit_is_an_explicit_no_op(module_name):
    """``name_from_rdkit`` used to call ``Chem.MolToIUPACName``, which RDKit
    does not have, and swallow the ``AttributeError`` at debug level. It is
    kept for compatibility but says what it is: no name, with a warning."""
    external = importlib.import_module(module_name)
    with pytest.warns(DeprecationWarning, match="RDKit has no structure-to-name"):
        assert external.name_from_rdkit(_WATER) is None


@pytest.mark.parametrize("module_name", _EXTERNAL_MODULES)
def test_external_chain_does_not_try_a_local_rdkit_backend(module_name, monkeypatch):
    """The chain goes straight to the formula-based REST lookup."""
    external = importlib.import_module(module_name)
    rdkit_calls = []
    formulas = []

    def _rdkit(atoms):
        rdkit_calls.append(atoms)
        return "should not be used"

    def _formula(formula):
        formulas.append(formula)
        return (f"name for {formula}", "cactus")

    monkeypatch.setattr(external, "name_from_rdkit", _rdkit)
    monkeypatch.setattr(external, "name_from_formula_external", _formula)

    assert external.name_from_atoms_external(_METHANE) == ("name for CH4", "cactus")
    assert rdkit_calls == []
    assert formulas == ["CH4"]


@pytest.fixture(params=["standalone", "legacy"])
def naming539_api(request, monkeypatch):
    """Exercise both Python implementations without importing the native core."""
    if request.param == "standalone":
        return importlib.import_module("vibeqc_naming.iupac_name")
    import sys
    import types
    from pathlib import Path

    package = types.ModuleType("_naming539_legacy")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/vibeqc/naming")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = importlib.import_module(package.__name__ + ".iupac_name")
    # Relative modules are pure Python; they do not initialize vibeqc.
    return module


_NAMING539_STRUCTURES = [
    ('methane', 'C'),
    ('ethane', 'CC'),
    ('propane', 'CCC'),
    ('butane', 'CCCC'),
    ('pentane', 'CCCCC'),
    ('hexane', 'CCCCCC'),
    ('2-methylpentane', 'CC(C)CCC'),
    ('acetylene', 'C#C'),
    ('ethylene', 'C=C'),
    ('1,3-butadiene', 'C=CC=C'),
    ('cyclopropane', 'C1CC1'),
    ('cyclobutane', 'C1CCC1'),
    ('cyclohexane', 'C1CCCCC1'),
    ('benzene', 'c1ccccc1'),
    ('toluene', 'Cc1ccccc1'),
    ('phenol', 'Oc1ccccc1'),
    ('aniline', 'Nc1ccccc1'),
    ('pyridine', 'n1ccccc1'),
    ('naphthalene', 'c1ccc2ccccc2c1'),
    ('chlorobenzene', 'Clc1ccccc1'),
    ('biphenyl', 'c1ccc(-c2ccccc2)cc1'),
    ('formaldehyde', 'C=O'),
    ('methanol', 'CO'),
    ('acetaldehyde', 'CC=O'),
    ('ethanol', 'CCO'),
    ('dimethyl ether', 'COC'),
    ('acetone', 'CC(=O)C'),
    ('propanal', 'CCC=O'),
    ('formic acid', 'C(=O)O'),
    ('acetic acid', 'CC(=O)O'),
    ('propanoic acid', 'CCC(=O)O'),
    ('ethyl acetate', 'CCOC(C)=O'),
    ('glycine', 'NCC(=O)O'),
    ('acetonitrile', 'CC#N'),
    ('hydrogen cyanide', 'C#N'),
    ('methylamine', 'CN'),
    ('trimethylamine', 'CN(C)C'),
    ('cyclohexanol', 'OC1CCCCC1'),
    ('butan-2-ol', 'CC(O)CC'),
    ('ethane-1,2-diol', 'OCCO'),
    ('benzaldehyde', 'O=Cc1ccccc1'),
    ('benzoic acid', 'O=C(O)c1ccccc1'),
    ('tetrahydrofuran', 'C1CCOC1'),
    ('dichloromethane', 'ClCCl'),
]


def _naming539_atoms(smiles):
    chem = pytest.importorskip("rdkit.Chem")
    allchem = pytest.importorskip("rdkit.Chem.AllChem")
    mol = chem.AddHs(chem.MolFromSmiles(smiles))
    options = allchem.ETKDGv3()
    options.randomSeed = 7
    assert allchem.EmbedMolecule(mol, options) == 0
    allchem.MMFFOptimizeMolecule(mol)
    return [(a.GetAtomicNum(), *mol.GetConformer().GetAtomPosition(a.GetIdx()))
            for a in mol.GetAtoms()]


@pytest.mark.parametrize("name,smiles", _NAMING539_STRUCTURES)
def test_naming539_structure_identity(naming539_api, name, smiles):
    atoms = _naming539_atoms(smiles)
    result = naming539_api.name_from_atoms_detailed(atoms)
    assert result.name == name
    assert result.confidence.value == "high"
    # Naming must not depend on atom order, translation, or formula preference.
    transformed = [(z, x + 2, y - 3, zz + 1) for z, x, y, zz in reversed(atoms)]
    assert naming539_api.name_from_atoms(transformed, prefer_trivial=False) == name


def test_naming539_unsupported_structure_refuses_before_external(naming539_api, monkeypatch):
    atoms = _naming539_atoms("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    monkeypatch.setattr(naming539_api, "_try_external", lambda *_: pytest.fail("formula lookup must not invent structural identity"))
    with pytest.raises(ValueError, match="unsupported molecular connectivity"):
        naming539_api.name_from_atoms_detailed(atoms, with_external=True)


@pytest.mark.parametrize("charge,multiplicity", [(1, 1), (-1, 1), (0, 2), (0, 3)])
def test_naming539_electronic_state_refusal(naming539_api, charge, multiplicity):
    from types import SimpleNamespace
    from vibeqc_naming.structure_db import structure_from_name
    atoms = structure_from_name("water")
    for entry in (naming539_api.name_from_atoms, naming539_api.name_from_atoms_detailed):
        with pytest.raises(ValueError, match="charged or open-shell"):
            entry(atoms, charge=charge, multiplicity=multiplicity)
    graph = naming539_api._build_atoms_graph(atoms)
    for entry in (naming539_api.name_from_graph, naming539_api.name_from_graph_detailed):
        with pytest.raises(ValueError, match="charged or open-shell"):
            entry(graph, charge=charge, multiplicity=multiplicity)
    with pytest.raises(ValueError, match="charged or open-shell"):
        naming539_api.name_molecule_detailed(SimpleNamespace(charge=charge, multiplicity=multiplicity))


@pytest.mark.parametrize("smiles", ["[NH4+]", "[OH-]", "[CH3]"])
def test_naming539_unannotated_odd_electron_species_refuse(naming539_api, smiles):
    with pytest.raises(ValueError, match="odd-electron"):
        naming539_api.name_from_atoms_detailed(_naming539_atoms(smiles))


@pytest.mark.parametrize("name,smiles", [("acetone", "CC(C)=O"), ("acetic acid", "CC(=O)O"), ("glycine", "NCC(=O)O")])
def test_naming539_reverse_carbonyl_contract(name, smiles):
    chem = pytest.importorskip("rdkit.Chem")
    perception = pytest.importorskip("rdkit.Chem.rdDetermineBonds")
    from vibeqc_naming import structure_from_name, smiles_from_name
    canonical = lambda mol: chem.MolToSmiles(chem.RemoveHs(mol))
    expected = canonical(chem.MolFromSmiles(smiles))
    parsed = chem.MolFromSmiles(smiles_from_name(name))
    assert parsed is not None
    assert canonical(parsed) == expected
    atoms = structure_from_name(name)
    xyz = f"{len(atoms)}\n\n" + "\n".join(
        f"{chem.GetPeriodicTable().GetElementSymbol(z)} {x} {y} {zz}" for z, x, y, zz in atoms)
    perceived = chem.MolFromXYZBlock(xyz)
    perception.DetermineBonds(perceived, charge=0)
    assert canonical(perceived) == expected


@pytest.mark.parametrize("smiles", ["CC(C)=O", "CC(=O)O", "NCC(=O)O", "c1ccccc1", "c1ccc2ccccc2c1", "c1ccc(-c2ccccc2)cc1"])
def test_naming539_smiles_branches_and_ring_closures(smiles):
    chem = pytest.importorskip("rdkit.Chem")
    from vibeqc_naming import smiles_from_atoms
    atoms = _naming539_atoms(smiles)
    expected = chem.MolToSmiles(chem.MolFromSmiles(smiles))
    for ordered in (atoms, list(reversed(atoms))):
        actual = chem.MolFromSmiles(smiles_from_atoms(ordered))
        assert actual is not None
        assert chem.MolToSmiles(actual) == expected


@pytest.mark.parametrize("charge,multiplicity,expected", [(0, 1, True), (1, 1, False), (0, 2, False)])
def test_naming539_report_preserves_electronic_state(monkeypatch, charge, multiplicity, expected):
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from vibeqc_naming import structure_from_name

    # Load the pure report seam with a recording output sink, without a native
    # engine. This exercises the actual report function and actual naming API.
    root = types.ModuleType("_naming539_report")
    root.__path__ = []
    output = types.ModuleType(root.__name__ + ".output")
    writes = []
    output.write = writes.append
    monkeypatch.setitem(sys.modules, root.__name__, root)
    monkeypatch.setitem(sys.modules, output.__name__, output)
    path = Path(__file__).resolve().parents[1] / "python/vibeqc/naming/report.py"
    spec = importlib.util.spec_from_file_location(root.__name__ + ".naming.report", path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    atoms = [types.SimpleNamespace(Z=z, xyz=[v / 0.529177210903 for v in (x, y, zz)])
             for z, x, y, zz in structure_from_name("water")]
    molecule = types.SimpleNamespace(atoms=atoms, charge=charge, multiplicity=multiplicity)
    assert report.write_iupac_name(molecule) is expected
    assert writes == (["  IUPAC name: water\n"] if expected else [])
