# vibeqc-naming

`vibeqc-naming` is vibe-qc's IUPAC naming engine as a distribution of its
own: structure to name and name to structure, in pure Python, with NumPy as
its only dependency. It exists so that tools such as vibe-sketch can name
molecules without building vibe-qc's compiled core (vibe-qc issue 536).

The import name is `vibeqc_naming`, the same package that every vibe-qc
wheel bundles and that `vibeqc.naming` re-exports. An environment with
vibe-qc installed needs nothing more.

## Install

From a vibe-qc checkout:

```bash
pip install ./vibeqc-naming
```

The `solids` extra adds spglib for the space-group analysis in
`name_solid_from_atoms`; the composition name needs no symmetry, so the
engine runs without it. Once the wheel is published beside vibe-qc's
releases, `pip install vibeqc-naming` resolves the same package.

`pip install -e ./vibeqc-naming` puts the checkout's `python/` directory on
`sys.path`, so edits under `python/vibeqc_naming` are picked up live.

## Use

```python
from vibeqc_naming import (
    MolecularGraph,
    name_from_atoms,
    name_from_atoms_detailed,
    name_from_graph,
    structure_from_name,
)
from vibeqc_naming.smiles import smiles_from_name

water = [(8, 0, 0, 0.117), (1, 0, 0.757, -0.469), (1, 0, -0.757, -0.469)]
name_from_atoms(water)                       # "water"
name_from_atoms_detailed(water).confidence   # provenance and confidence
structure_from_name("pyridine")              # [(7, x, y, z), (6, ...), ...]
smiles_from_name("ethanol")                  # "CCO"

# A graph from bonds a sketcher already knows, with a caller-supplied
# CIP descriptor that the name carries (butan-2-ol with R on atom 1).
graph = MolecularGraph.from_bonds(elements, bonds, atom_stereo={1: "R"})
name_from_graph(graph)                       # "(2R)-butan-2-ol"
```

The user guide, `docs/user_guide/naming.md` in the vibe-qc repository,
documents the engine, its confidence levels and its limits.

## Layout

The package source stays at `python/vibeqc_naming` in the vibe-qc
repository. `hatch_build.py` packages that directory from a checkout and
carries it into the sdist as `src/vibeqc_naming`, which the wheel
configuration selects when building from the sdist. The in-package test
scripts (`_test_*.py`, `_smoke_test.py`) do not ship.

The version line is independent of vibe-qc's: `pyproject.toml` and
`vibeqc_naming.__version__` carry the same number, and
`tests/test_distribution.py` pins them together.

## Tests

```bash
python -m pytest vibeqc-naming/tests
```

They run from a checkout without an install (the conftest puts `python/` on
the path) or against an installed wheel, which is what the release-candidate
CI job does. Building the wheel and sdist in the tests needs hatchling, the
`test` extra.

## License

MPL-2.0, as vibe-qc.
