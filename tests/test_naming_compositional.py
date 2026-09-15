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
