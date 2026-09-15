(naming)=

# Molecular naming

vibe-qc can produce IUPAC names from 3D molecular structures and,
conversely, build 3D coordinates from chemical names. The naming
engine is a standalone, pure-Python package (`vibeqc_naming`) that
works without the C++ extension compiled.

## Quick start

```python
from vibeqc_naming import name_from_atoms, structure_from_name

# Structure → name
water = [(8, 0, 0, 0.117), (1, 0, 0.757, -0.469), (1, 0, -0.757, -0.469)]
name_from_atoms(water)
# → "water"

# Name → structure  
pyridine = structure_from_name("pyridine")
# → [(7, 1.40, 0.0, 0.0), (6, 0.70, 1.212, 0.0), ...]
```

## Naming from a calculation

When using {func}`vibeqc.runner.run_job`, the IUPAC name is printed
to the `.out` file by default:

```python
from vibeqc import Molecule
from vibeqc.runner import run_job

mol = Molecule.from_xyz("ethanol.xyz")
run_job(mol, basis="def2-SVP", method="rks", functional="PBE")
# .out file contains:
#   IUPAC name: ethanol
```

Pass `name_molecule=False` to suppress the name.

## NamedResult

The detailed API returns a {class}`~vibeqc_naming.NamedResult` with
provenance and confidence:

```python
from vibeqc_naming import name_from_atoms_detailed

result = name_from_atoms_detailed(water)
print(result.name)        # "water"
print(result.source)      # NamingSource.TRIVIAL_IUPAC
print(result.confidence)  # Confidence.HIGH
```

Confidence levels:

| Confidence | Meaning |
|---|---|
| `high` | Trivial/retained IUPAC name or confident systematic name |
| `medium` | Systematic name with some uncertainty |
| `low` | Formula-only compositional name or failed lookup |

## Built-in structure database

83 common molecules are available via {func}`~vibeqc_naming.structure_from_name`:

| Category | Examples |
|---|---|
| Small molecules | water, methane, ammonia, CO₂, HF, HCl, H₂, N₂, O₂ |
| Alkanes | ethane, propane, butane |
| Alkenes/alkynes | ethylene, acetylene |
| Aromatics | benzene, toluene, phenol, aniline |
| Heterocycles | pyridine, pyrrole, furan, imidazole |
| Functional groups | formaldehyde, formic acid, acetic acid, acetone, acetaldehyde |
| Alcohols/amines | methanol, ethanol, methylamine |
| Nucleobases | adenine, thymine, uracil, cytosine, guanine |
| Amino acids | glycine, alanine |
| Solvents | DMSO, acetonitrile, dichloromethane, THF |
| Others | hydrogen peroxide, phosphine, silane, diborane, nitromethane, cyclohexane, benzaldehyde, benzoic acid, dimethyl ether |

```python
from vibeqc_naming import structure_from_name, known_names

len(known_names())  # → 83
```

## Solids, slabs and periodic compositions

Use the solid engine for periodic structures. It names composition without
constructing a molecular bond graph, so ions disconnected within a unit cell,
metals, and supercells do not acquire fragment or coordination-complex names.
All 118 element symbols are supported.

```python
from vibeqc_naming import name_solid_from_atoms

atoms = [(11, 0.0, 0.0, 0.0), (17, 3.37, 3.37, 3.37)]
cell = [(6.74, 0, 0), (0, 6.74, 0), (0, 0, 6.74)]
result = name_solid_from_atoms(atoms, cell)
assert result.name == "sodium chloride"
assert result.formula == "ClNa"          # occupied unit-cell formula, Hill order
assert result.reduced_formula == "ClNa"  # simplest integer ratio, Hill order
assert result.display_formula == "NaCl" # chemical display order
assert result.name_kind == "systematic"
assert result.dimensionality == 3
```

Coordinates and lattice **rows** are in angstrom. `pbc` contains three booleans
paired with the lattice rows and defaults to `(True, True, True)`. An elongated
cell is still 3D unless the caller specifies reduced periodicity. Non-leading
periodic axes such as `(True, False, True)` are supported. The lattice must be
finite and full rank, including the bookkeeping rows for nonperiodic axes.

The result is a `SolidResult`, a `NamedResult` subclass that preserves `.name`,
`.source`, `.confidence`, `.formula`, and `.is_trivial`. It also carries weighted
`composition`, `reduced_formula`, `pbc`, `dimensionality`, optional structural
metadata, and `notes` explaining the origin or absence of that metadata.
Composition names have MEDIUM confidence: they do not establish phase identity.
`name_kind` distinguishes `systematic`, `common`, `formula_label`, and
`descriptive` output. Slab, chain, and adsorbate labels are descriptive.
`display_formula_order` is `compositional` where chemical ordering is supported,
or `hill` for a database formula used as a fallback. Existing Hill-order fields
retain their meaning.

| Composition / metadata | Default name |
|---|---|
| NaCl, any integer supercell | sodium chloride |
| Fe2O3 | diiron trioxide |
| UO2 | uranium dioxide |
| BaTiO3 | barium titanate |
| CuZn | copper zincide |
| NiSn | nickel stannide |
| Cu5Zn8 | pentacopper octazincide |
| Cr23C6 | tricosachromium hexacarbide |
| OCl2 | oxygen dichloride |
| An unrecognized multinary composition | Hill-formula solid |
| Carbon with `pbc=(True, True, False)` | carbon slab |
| Carbon with `pbc=(False, True, False)` | carbon chain |

The new API defaults to compositional names. `prefer_trivial=True` permits a
small list of composition names such as alumina, silica, magnesia and water.
It never identifies diamond, graphite, quartz, anatase or another polymorph from
formula alone. A formula label is a descriptive fallback, not a claim that the
sample is an alloy, a single phase, or a particular bonding arrangement.
Binary names follow the conventional element sequence in Red Book Table VI,
including oxygen before halogens, and the `-ide` names in Table IX. Multipliers
support ratios through 9999 without eliding vowels (for example, tetraoxide).
These names specify empirical composition, not oxidation states or molecular
units: `N2O4` reduces to `NO2` in this solid API. The molecular API retains its
existing behavior. Compositions containing elements 112-118, which are absent
from Table VI, still receive formula labels; their elemental names are supported.

### Occupancy, disorder and nonstoichiometry

```python
result = name_solid_from_atoms(
    [(26, 0, 0, 0), (8, 2, 0, 0)], cell, occupancies=[0.9, 1.0]
)
assert result.name == "Fe0.9O solid"
assert result.formula == "Fe0.9O"
assert result.reduced_formula == "Fe9O10"
```

Occupancies must lie in `[0, 1]`, with one value per atom. Alternative species
may occupy the same site if their occupancies sum to at most one; zero-occupancy
entries contribute no atoms. Weighted cell counts are retained in `composition`.
Partial-occupancy display counts are divided by the greatest common divisor of
the unweighted numbers of occupied entries for each element. This preserves the
site reference under exact repetition of the cell. It does not establish a
unique parent formula for arbitrary disorder models. Supply `formula_units=N`
to divide the weighted cell counts by a known number of formula units instead.
The divisor and its origin are recorded in `notes`. These labels do not assign
a defect charge or site ordering. For example, an extra iron site with occupancy
0.1 in FeO gives `Fe1.1O solid`; equal Cu/Zn sharing gives `Cu0.5Zn0.5 solid`.

For composition alone, use `name_solid_from_formula("Fe2O3")`. This accepts plain
elemental formulas and decimal counts; parentheses, charges, and symbolic
composition variables must first be resolved to counts.
This returns `SolidCompositionResult`, a `NamedResult` subclass carrying
`display_formula`, `display_formula_order`, and `name_kind` without structural
claims. Integer formulas reduce to the empirical ratio; decimal formulas retain
their supplied coefficient scale in the display and name. Thus `Fe1.1O` remains
`Fe1.1O solid`, while its database `formula` is `Fe11O10`.

### Symmetry and supplied structural labels

```python
result = name_solid_from_atoms(atoms, cell, analyze_symmetry=True)
print(result.space_group, result.space_group_number, result.crystal_system)
print(result.pearson_symbol, result.phase_descriptor)
print(result.notes)
```

Symmetry analysis uses optional [spglib](https://spglib.readthedocs.io/en/v2.7.0/api/autodoc/spglib.html)
and the full atomic structure, with `symprec=1e-5` angstrom by default. It runs
only for fully occupied 3D cells. Missing spglib or failed analysis leaves the
composition name available and adds a note. Structural metadata does not change
the composition name or identify a mineral; many structures share a space group.
`pearson_symbol` combines lattice system, centering, and the number of atoms in
the standardized conventional cell, as required by Red Book IR-11.5. Input
primitive cells and supercells therefore yield the same descriptor. For example,
fcc copper gives `Cu(cF4)` in `phase_descriptor`, and rock-salt NaCl gives
`NaCl(cF8)`. Rhombohedral cells use the hexagonal conventional atom count; side
centering is written `S`. These descriptors are absent when symmetry is skipped
or unavailable. They are plain-text representations of the printed notation.

Known phase labels and Miller indices can be supplied explicitly:

```python
result = name_solid_from_atoms(
    [(6, 0, 0, 0)], cell, pbc=(True, True, False),
    phase="graphite", miller_indices=(0, 0, 0, 1),
)
assert result.name == "carbon (graphite) (0 0 0 1) slab"
```

`notes` marks these labels as caller-supplied, not inferred or verified. Miller
indices refer to the **parent crystal basis**; three-index and four-index
Miller-Bravais notation are supported. Lattice dimensions alone are insufficient
to determine an exposed Miller plane.

### Adsorbates and existing entry points

For a 2D substrate, supply disjoint `adsorbate_groups` containing atom indices.
The remaining atoms define the substrate; each group is named by the molecular
engine. Molecular group coordinates must be unwrapped across periodic boundaries.
For example, a CO group on copper produces `carbon monoxide on copper` with
`prefer_trivial=True`. Automatic adsorbate partitioning is not part of the solid
engine because connectivity within a single cell is insufficient evidence.

`name_from_atoms_with_lattice` and `name_periodic_system` use the same engine and
accept `pbc` and the solid options above. Their existing `prefer_trivial=True`
default is retained. With no lattice, or with all PBC flags false, the atoms
entry point retains molecular naming. Unlike the old slab heuristic, a lattice
without PBC flags defaults to **3D**; pass the flags explicitly for slabs.
`detect_slab` and the legacy `name_slab_system` graph heuristic remain available,
but do not determine periodicity for the new engine.

`name_from_qvf` reads boundary conditions and optional per-atom `occupancy` from
the archive, returns a `SolidResult` for periodic structures, and accepts options
such as `analyze_symmetry=True`. The archive's PBC and occupancies cannot be
overridden through these options. Molecular archives retain their existing names.

This implements composition-based coverage and explicit structural descriptors,
following [IUPAC Red Book 2005, IR-5 and IR-11, Tables IV, VI and IX](https://iupac.org/wp-content/uploads/2016/07/Red_Book_2005.pdf)
and the [IUPAC numerical terms recommendations](https://iupac.qmul.ac.uk/misc/numb.html).
It is not exhaustive mineral/prototype recognition, automatic oxidation-state
assignment, magnetic-space-group naming, symbolic defect nomenclature, or
general multinary constituent identification. Unknown multinary compositions
remain explicitly marked Hill-formula labels rather than inferred ion names.

See {doc}`slabs_and_adsorbates` for periodic calculation setup.

## SMILES generation

The molecular graph can produce SMILES strings:

```python
from vibeqc_naming.smiles import smiles_from_name

smiles_from_name("benzene")    # → "C1CCCCC1"
smiles_from_name("acetylene")  # → "C#C"
```

## External backends

For complex molecules beyond the built-in engine, optional backends
provide authoritative IUPAC names:

```python
result = name_from_atoms_detailed(complex_atoms, with_external=True)
```

Backends tried in order: RDKit (local) → CACTUS REST → PubChem REST.
REST API results are cached in a 256-entry LRU to avoid repeated
network calls.

## API reference

```{eval-rst}
.. autosummary::
   :nosignatures:

   vibeqc_naming.name_from_atoms
   vibeqc_naming.name_from_atoms_detailed
   vibeqc_naming.name_molecule
   vibeqc_naming.name_molecule_detailed
   vibeqc_naming.name_from_graph
   vibeqc_naming.formula_to_name
   vibeqc_naming.name_from_qvf
   vibeqc_naming.structure_from_name
   vibeqc_naming.known_names
   vibeqc_naming.name_from_atoms_with_lattice
   vibeqc_naming.name_periodic_system
   vibeqc_naming.name_solid_from_atoms
   vibeqc_naming.name_solid_from_formula
```
