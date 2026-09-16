# Periodic systems

vibe-qc supports **1D, 2D, and 3D** periodicity through a single
`PeriodicSystem` type. Dimensionality is specified by `dim ∈ {1, 2, 3}`;
the lattice matrix is always a full-rank 3×3.

```{seealso}
* [`periodic_methods.md`](periodic_methods.md) - comparative tour
  of vibe-qc's periodic-SCF kernels (BIPOLE, GDF, GPW/GAPW), a
  bonding-organised example gallery, and the end-to-end
  surface-reactions workflow.
* [`crystal_lattices.md`](crystal_lattices.md) — the 14 Bravais
  lattices, with **visualisations + worked examples** for each
  common lattice (rocksalt, diamond, perovskite, HCP, wurtzite,
  rutile, corundum, α-quartz, graphene).
* [`k_points.md`](k_points.md) — Monkhorst-Pack mesh generation
  and Brillouin-zone sampling.
* [`multi_k_scf.md`](multi_k_scf.md) — multi-k SCF: current ship
  state (Γ-only native GDF), parity target, scope caveats.
* [`ewald.md`](ewald.md) — Ewald summation for periodic Coulomb.
```

## Born-von Karman PBC contract

The periodic stack follows the same lattice-general Born-von Karman
boundary-condition model used by CRYSTAL-style Gaussian crystal codes:

- The input lattice is an arbitrary full-rank matrix whose **columns**
  are the Cartesian direct-lattice vectors in bohr: `lattice[:, i]` is
  $\mathbf{a}_i$. There are no cubic-, hexagonal-, or orthorhombic-only
  branches in the public periodic method interfaces.
- **The orientation is not checkable from the matrix alone.** A
  row-oriented matrix (`lattice[i, :]` = $\mathbf{a}_i$, the ASE and
  Extended-XYZ convention) is still a full-rank lattice, so it is
  accepted and produces a *different, sheared* crystal that converges
  and reports plausible numbers. Cubic and FCC matrices are symmetric
  and hide this; hexagonal, monoclinic, and triclinic cells do not
  (issue #445, four independent instances). The volume cannot expose it
  either, because $\det L = \det L^{T}$. Two things do:
  * **Build from vectors, not a matrix.** `PeriodicSystem(dim,
    lattice_vectors=[a1, a2, a3], unit_cell=...)` and
    `vq.lattice_from_vectors(a1, a2, a3)` take the three vectors
    themselves and stack them into the columns for you, so there is no
    orientation to get wrong. Prefer this form whenever the lattice is
    written down as vectors (which is how every textbook writes it).
  * **Read the cell parameters back.** `vq.cell_parameters(sysp)` gives
    $|\mathbf{a}|, |\mathbf{b}|, |\mathbf{c}|, \alpha, \beta, \gamma$ as the
    engine consumes them, and `vq.nearest_neighbour_distance(sysp)` the
    shortest interatomic distance including images. Both move under a
    transpose where the volume does not, and the periodic `.out` prints
    them in its `Cell parameters` block, so a wrong cell is visible at
    the top of every output (h-BN: $a = 2.504$ Angstrom, $\gamma = 60^\circ$,
    B-N $1.446$ Angstrom; the row-fed cell in #108 printed $63.4^\circ$ and
    $0.857$ Angstrom). Assert these against literature when reviewing a
    lattice-related result; a volume check proves nothing.
    `nearest_neighbour_distance` is exact for any full-rank cell: it
    Minkowski-reduces the periodic vectors before searching, so a lattice
    written in a skewed basis reports the same distance as the same lattice
    written in a reduced one. It used to scan a fixed shell over the basis
    as supplied, which is not sufficient in general and reported a distance
    too large on a sheared cell (#128).
- `sysp.reciprocal_lattice()` is generated automatically as
  $2\pi A^{-T}$, so $a_i \cdot b_j = 2\pi\delta_{ij}$ for triclinic
  cells just as for cubic cells.
- Monkhorst-Pack, Gamma-centred, and explicit k-lists are stored in
  fractional reciprocal coordinates and converted through that
  reciprocal matrix.
- Real-space lattice sums enumerate integer Born-von Karman image
  vectors and then apply Cartesian cutoff / screening tolerances.
  Tight production crystals should use the GDF/FFTDF track; direct
  truncated sums remain a debug/reference path.
- Bravais centering is represented either by primitive vectors or by
  additional basis atoms in a conventional cell. The Coulomb, k-point,
  GDF, FFT, and SCF interfaces should not special-case `P`, `I`, `F`,
  `C`, or `R` centering labels.

Regression coverage pins this contract on representative triclinic,
monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, and cubic
metrics, plus CRYSTAL-style target structures: FCC diamond, BCC Fe,
hexagonal graphene, and Al2O3 stoichiometry in a skew hexagonal cell.
The native Γ-GDF RHF/RKS path is also smoke-tested on those 3D crystal
metrics and on non-orthogonal 1D / 2D embedding cells.

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem

# 1D chain — column 0 of lattice is the lattice vector, the other two
# columns are implicit vacuum axes with enough separation to decouple
# images.
sysp_1d = PeriodicSystem(
    dim=1,
    lattice=np.diag([4.0, 30.0, 30.0]),
    unit_cell=[Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])],
)

# 2D slab. Prefer vq.slab_2d: give only the two in-plane lattice
# vectors and the atoms at their real z. The third lattice column is
# synthesized for you and is NOT a vacuum gap; the SCF is invariant to
# it. Building a dim=2 PeriodicSystem by hand works too, but then the
# third column is still only bookkeeping.
sysp_2d = vq.slab_2d(
    [2.5, 0.0, 0.0],
    [0.0, 2.5, 0.0],
    [Atom(6, [0, 0, 0])],
)

# 3D bulk — all three columns are lattice vectors.
sysp_3d = PeriodicSystem(
    dim=3,
    lattice=np.diag([5.4, 5.4, 5.4]),       # Si conventional cubic, bohr
    unit_cell=[Atom(14, [0, 0, 0]), Atom(14, [0.25, 0.25, 0.25])],
)

# 2D monolayer from the three lattice VECTORS (orientation-free form, #445).
# a1, a2, a3 are stacked into the columns for you; a row-major literal
# like this one is exactly what the ``lattice=`` matrix form would have
# silently transposed on a non-symmetric cell.
a = 4.732  # h-BN, bohr
sysp_hbn = PeriodicSystem(
    dim=2,
    lattice_vectors=[
        [a, 0.0, 0.0],
        [a / 2, a * 3**0.5 / 2, 0.0],
        [0.0, 0.0, 30.0],
    ],
    unit_cell=[Atom(5, [0, 0, 0]), Atom(7, [a / 2, a / (2 * 3**0.5), 0])],
)
vq.cell_parameters(sysp_hbn).gamma      # 60.0
vq.nearest_neighbour_distance(sysp_hbn).distance_angstrom   # 1.446
```

## Accessors

```python
sysp.reciprocal_lattice()      # 3×3 matrix, columns = b_i with a_i · b_j = 2π δ_ij
sysp.n_electrons()             # total electrons per unit cell (Z - charge)
sysp.unit_cell_molecule()      # Molecule view for BasisSet() construction
```

## Space-group analysis

```python
from vibeqc import attach_symmetry
attach_symmetry(sysp)           # populates sysp.symmetry via spglib
print(sysp.symmetry)
# SpaceGroup(number=227, symbol='Fd-3m', order=48)
```

`SpaceGroup` carries `number` (International Tables), `international_symbol`,
`hall_number`, `point_group`, and the full list of symmetry operations.

## POSCAR I/O

```python
from vibeqc import read_poscar, write_poscar

sysp = read_poscar("tests/data/NaCl.poscar")   # returns Crystal
# To convert between Crystal and PeriodicSystem, see the next section.

standard = read_poscar(
    "tests/data/NaCl.poscar",
    symmetrise=True,
    to_primitive=True,
)
sysp_clean = standard.system                  # returns PeriodicSystem
```

## CIF I/O

```python
from vibeqc import read_cif

crystal = read_cif("MgO.cif")                 # returns Crystal

standard = read_cif(
    "MgO.cif",
    symmetrise=True,
    to_primitive=True,
)
sysp_clean = standard.system                  # returns PeriodicSystem
```

The CIF reader handles cell lengths/angles, fractional
`_atom_site_*` loops, uncertainty suffixes such as `4.210(1)`, and
standard CIF symmetry-operation loops. When symmetry operations are
present, asymmetric-unit sites are expanded to the full unit cell
before `Crystal` construction.

## Extended-XYZ I/O

Periodic Extended-XYZ stores the lattice in the comment line using the
ASE convention: `Lattice="a_x a_y a_z b_x ... c_z"` with lattice
vectors as rows in Ångström. vibe-qc converts that to the internal
column-vector bohr convention when periodic mode is requested:

```python
from vibeqc import from_xyz

sysp = from_xyz("MgO.xyz", periodic=True)     # returns PeriodicSystem

standard = from_xyz(
    "MgO.xyz",
    symmetrise=True,
    to_primitive=True,
)
sysp_clean = standard.system                  # returns PeriodicSystem
```

Plain molecular `from_xyz(path)` is unchanged and still returns a
`Molecule`. For plain XYZ files that carry no `Lattice="..."` tag,
pass an explicit 3×3 `lattice=` matrix in Ångström and, if needed,
`dim=1`, `dim=2`, or `dim=3`.
