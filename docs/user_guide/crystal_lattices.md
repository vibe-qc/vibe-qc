# Crystal lattices

A crystal is a **lattice** plus a **basis**: an infinite, periodic
arrangement of points where each point carries one or more atoms.
Every periodic calculation in vibe-qc, `PeriodicSystem`,
`run_rhf_periodic_scf`, `monkhorst_pack`, band-structure plots,
starts from one of the 14 Bravais lattices, decorated with the
basis atoms that turn it into a real material.

This page is the reference for **which lattice to pick, how to set
it up in vibe-qc, and what the Brillouin zone of each looks like**.
For the *sampling* of the BZ (Monkhorst-Pack, IBZ reduction, mesh
choice, band paths), see [k-point meshes](k_points.md).

## Lattice + basis = crystal

A Bravais lattice is the set of points

$$
\mathbf{R} = n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2 + n_3 \mathbf{a}_3
\quad \text{with } n_i \in \mathbb{Z},
$$

where the **lattice vectors** $\mathbf{a}_1, \mathbf{a}_2,
\mathbf{a}_3$ define the parallelepiped *unit cell*. The **basis** is
the list of atoms inside one unit cell:

$$
\boldsymbol\tau_j \quad j = 1, \ldots, N_\text{atoms in cell}.
$$

The full crystal is then $\{\boldsymbol\tau_j + \mathbf{R}\}$ for
every $\mathbf{R}$ on the lattice and every atom $j$ in the basis.

**Every Bravais lattice has a single point per unit cell.** Crystals
with 2+ atoms per cell are *not* multi-atom Bravais lattices, they're
a Bravais lattice with a multi-atom basis. Diamond is FCC + 2-atom
basis; NaCl is FCC + 2-atom basis; HCP is hexagonal + 2-atom basis.

In vibe-qc:

```python
import numpy as np
import vibeqc as vq

# 1. The three lattice vectors, in bohr. Give them as VECTORS
#    (``lattice_vectors=``): vibe-qc stacks them into the COLUMNS of the
#    lattice matrix for you. The ``lattice=`` matrix form reads columns
#    too, so a row-major literal handed to it is silently transposed --
#    harmless on this cubic cell, a different crystal on a hexagonal or
#    monoclinic one (issue #445; see periodic_systems.md).
a = 3.92 / 0.529177210903   # conventional cubic edge, Angstrom -> bohr
a1 = [a, 0.0, 0.0]
a2 = [0.0, a, 0.0]
a3 = [0.0, 0.0, a]

# 2. Basis atoms in CARTESIAN bohr (``Atom`` never takes fractional
#    coordinates): Na at the corner, Cl at the body centre (a/2, a/2, a/2).
sysp = vq.PeriodicSystem(
    dim=3,
    lattice_vectors=[a1, a2, a3],
    unit_cell=[
        vq.Atom(11, [0.0, 0.0, 0.0]),
        vq.Atom(17, [a / 2, a / 2, a / 2]),
    ],
)
vq.cell_parameters(sysp)      # a = b = c = 7.408 bohr, alpha = beta = gamma = 90
```

The `lattice` vectors, the choice of which atom sits at the origin,
and the orientation are all conventions. Different choices are
*equivalent* if related by a lattice translation; always *physically*
equivalent if related by a point-group operation.

## The 14 Bravais lattices

The Bravais classification groups every possible periodic 3D lattice
into 14 types, organized under the **7 crystal systems**:

| # | Crystal system | Bravais lattices | Constraint on $a, b, c$ | Constraint on $\alpha, \beta, \gamma$ |
|---|---|---|---|---|
| 1 | Triclinic | aP | none | none |
| 2 | Monoclinic | mP, mC (or mB) | none | $\alpha = \gamma = 90°$, $\beta \neq 90°$ |
| 3 | Orthorhombic | oP, oC, oI, oF | none | $\alpha = \beta = \gamma = 90°$ |
| 4 | Tetragonal | tP, tI | $a = b \neq c$ | $\alpha = \beta = \gamma = 90°$ |
| 5 | Trigonal (rhombohedral) | hR | $a = b = c$ | $\alpha = \beta = \gamma \neq 90°$ |
| 6 | Hexagonal | hP | $a = b \neq c$ | $\alpha = \beta = 90°$, $\gamma = 120°$ |
| 7 | Cubic | cP, cI, cF | $a = b = c$ | $\alpha = \beta = \gamma = 90°$ |

Letter codes: **P** = primitive, **I** = body-centered, **F** = face-
centered, **C** (or **A**, **B**) = base-centered (one face),
**R** = rhombohedral primitive.

The 14 unique lattices are:

```text
Cubic:           cP    cI (BCC)    cF (FCC)
Tetragonal:      tP    tI
Orthorhombic:    oP    oC          oI            oF
Hexagonal:       hP
Trigonal:        hR
Monoclinic:      mP    mC
Triclinic:       aP
```

Why 14 and not (say) 7 × 4 = 28? Some centrings are equivalent to
other Bravais lattices under a different choice of axes, e.g.
"face-centered tetragonal" is the same as "body-centered tetragonal"
with a 45° in-plane rotation.

```{important}
Periodic electronic-structure methods in vibe-qc must accept every
full-rank 3D lattice matrix from the 7 crystal systems. The centering
letters (`P`, `I`, `F`, `C`, `R`) are represented either by primitive
lattice vectors or by additional basis atoms in a conventional cell;
they are not special cases in the Coulomb, GDF, FFT, k-point, or SCF
interfaces. Regression tests cover representative triclinic,
monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, and cubic
metrics.
```

```{note}
The Bravais classification covers only **point groups of the
lattice**. The full classification of crystals (which adds the basis
+ point-group symmetries) gives the **230 space groups**, of which
the 14 Bravais lattices are the translational backbone. vibe-qc uses
[spglib](https://spglib.readthedocs.io/) to identify the space group
of a `PeriodicSystem` — `attach_symmetry(sysp)` returns the spglib
classification + the symmetry operations.
```

## Common materials lattices

Most QC use cases reduce to a small list of Bravais lattices with
specific basis decorations. Below: the name, primitive lattice
vectors (listed one per row in *units of the lattice constant $a$*;
hand them to `lattice_vectors=` as written, or stack them into the
**columns** of the `lattice=` matrix), and basis (in fractional
coordinates of the *primitive* lattice; convert to Cartesian bohr
before building `Atom`s). Each entry includes a **visualisation** (cell + 2×2×2
supercell, rotated for clarity) and links to **worked examples**
under [`examples/periodic/<system>/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic).

### Worked examples by material class

The bundled `examples/periodic/` directory ships standalone input
scripts (RHF + RKS-PBE + RKS-B3LYP for each material) plus
`.system` provenance manifests and, for MgO, paired CRYSTAL14
parity inputs. Use this index to find an example matching the
lattice and chemistry you're targeting:

| Class | Materials | Example directories |
|-------|-----------|----------------------|
| **Ionic, alkali / alkaline-earth halides + oxides** | LiH, NaCl, MgO | [LiH-rocksalt](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/LiH-rocksalt), [NaCl-rocksalt](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/NaCl-rocksalt), [MgO-rocksalt](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/MgO-rocksalt) |
| **Ionic, transition-metal oxides + wide-gap insulators** | TiO₂, ZnO, Al₂O₃, SiO₂ | [TiO2-rutile](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/TiO2-rutile), [ZnO-wurtzite](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/ZnO-wurtzite), [Al2O3-corundum](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Al2O3-corundum), [SiO2-alpha-quartz](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/SiO2-alpha-quartz) |
| **Semiconductors** | C-diamond, Si-diamond | [C-diamond](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/C-diamond), [Si-diamond](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Si-diamond) |
| **Noble-gas solids** | Ne FCC | [Ne-fcc](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Ne-fcc) |
| **Metals** | - | *Open at v0.8.x, see the [metals note](#a-note-on-metals) below.* |

```{note}
:name: a-note-on-metals

**Metals** — vibe-qc's metallic-smearing helpers landed in v0.8.0
(`run_*_periodic_gamma_gdf(..., smearing="0.01 Ha")`) but **multi-k
GDF is still Γ-only** at the public-API level (non-Γ meshes
deliberately raise; see [`multi_k_scf.md`](multi_k_scf.md)). True
metallic convergence needs a dense k-mesh, so a production
metals example doesn't ship yet. The metal-lattice **visualisations
below are correct**; the **example inputs slot is intentionally
empty** until the native multi-k loop lands. Tracked at
[`docs/roadmap.md` § native-multi-k GDF](../roadmap.md).
```

### Inline-example shortcut: MgO rocksalt

One full example inline so this page is self-contained, MgO
rocksalt at the conventional 8-atom cell, Γ-only RHF / pob-TZVP,
matching `examples/periodic/MgO-rocksalt/MgO-rocksalt-RHF-sto3g.py`:

```python
"""MgO rocksalt — Γ-only native GDF RHF.

Run::
    .venv/bin/python this_file.py
"""
import numpy as np
from vibeqc import Atom, BasisSet, PeriodicSystem, run_periodic_job

ANG2BOHR = 1.0 / 0.529177210903
a_ang = 4.211  # MgO experimental lattice constant (Å)

# Conventional cubic cell, 8 atoms (4 MgO formula units).
cell_bohr = np.eye(3) * a_ang * ANG2BOHR

atoms = [
    # Mg at corners (FCC sublattice)
    Atom(12, [0.0, 0.0, 0.0]),
    Atom(12, [0.0, a_ang/2, a_ang/2]),
    Atom(12, [a_ang/2, 0.0, a_ang/2]),
    Atom(12, [a_ang/2, a_ang/2, 0.0]),
    # O at face centers (interpenetrating FCC, offset by ½ diagonal)
    Atom(8, [a_ang/2, a_ang/2, a_ang/2]),
    Atom(8, [a_ang/2, 0.0, 0.0]),
    Atom(8, [0.0, a_ang/2, 0.0]),
    Atom(8, [0.0, 0.0, a_ang/2]),
]
atoms = [Atom(a.Z, [c * ANG2BOHR for c in a.xyz]) for a in atoms]
system = PeriodicSystem(3, cell_bohr, atoms)
basis = BasisSet(system, "sto-3g")

result = run_periodic_job(
    system, basis=basis,
    method="RHF",
    output="MgO-rocksalt-RHF-sto3g",
)
print(f"E = {result.energy:.6f} Ha/cell")
# Reference (v0.9.0 native GDF, sub-µHa vs PySCF.pbc.GDF):
# E ≈ -274.4321 Ha (RHF / sto-3g / Γ)
```

Every other lattice has the equivalent script under
`examples/periodic/<material>/`; the inline example is a
representative case to read top-to-bottom without leaving the
docs.

### Cubic family

#### Simple cubic (cP)

```{image} /_static/lattices/cubic-simple-Po.svg
:alt: Simple-cubic α-Po, 2×2×2 supercell
:width: 320px
:align: right
```

The textbook lattice, corners of a cube, one atom per cell.

```text
a₁ = (a, 0, 0)
a₂ = (0, a, 0)
a₃ = (0, 0, a)

Basis: (0, 0, 0)
```

Real materials: α-polonium is the only stable elemental simple-cubic
solid at ambient conditions. A useful **toy model** for periodic SCF
testing, large, isotropic, easy to converge.

```python
a = 3.50  # bohr
sysp = vq.PeriodicSystem(
    dim=3,
    lattice=a * np.eye(3),
    unit_cell=[vq.Atom(84, [0.0, 0.0, 0.0])],   # Po
)
```

#### Body-centered cubic / BCC (cI)

```{image} /_static/lattices/cubic-bcc-Fe.svg
:alt: BCC α-Fe, 2×2×2 supercell
:width: 320px
:align: right
```

Conventional cubic cell with extra atom at the body center. The
primitive cell is a rhombohedron one-half the volume of the
conventional cubic cell.

```text
Conventional cubic (2-atom basis):
  a₁ = (a, 0, 0); a₂ = (0, a, 0); a₃ = (0, 0, a)
  Basis: (0, 0, 0), (½, ½, ½)

Primitive (1-atom basis):
  a₁ = (a/2)·(-1,  1,  1)
  a₂ = (a/2)·( 1, -1,  1)
  a₃ = (a/2)·( 1,  1, -1)
  Basis: (0, 0, 0)
```

**Real materials**: α-Fe (3.16 Å), Na, K, Cr, Mo, W, V, Nb, Ta,
many transition metals and alkali metals. **Metal-typical**;
needs dense k-mesh + Fermi-Dirac smearing in production. See the
[metals note](#a-note-on-metals) above.

#### Face-centered cubic / FCC (cF)

```{image} /_static/lattices/cubic-fcc-Cu.svg
:alt: FCC Cu, 2×2×2 supercell
:width: 320px
:align: right
```

Conventional cubic cell with atoms at the corners + face centers.
4 conventional-cell atoms, primitive cell is one quarter the
volume.

```text
Conventional cubic (4-atom basis):
  a₁ = (a, 0, 0); a₂ = (0, a, 0); a₃ = (0, 0, a)
  Basis: (0, 0, 0), (½, ½, 0), (½, 0, ½), (0, ½, ½)

Primitive (1-atom basis):
  a₁ = (a/2)·(0, 1, 1)
  a₂ = (a/2)·(1, 0, 1)
  a₃ = (a/2)·(1, 1, 0)
  Basis: (0, 0, 0)
```

**Real materials**: Cu, Ag, Au, Al, Ni, Pt, Pd, γ-Fe, Pb, most
ductile FCC metals plus the noble metals. **Metal-typical** (see
the [metals note](#a-note-on-metals)). Closed-shell noble-gas FCC
solids (Ne, Ar) work today at Γ-only, see
[`examples/periodic/Ne-fcc/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Ne-fcc).

#### Diamond / zincblende (cF + 2-atom basis)

```{image} /_static/lattices/cubic-diamond-Si.svg
:alt: Diamond / zincblende Si, 2×2×2 supercell
:width: 320px
:align: right
```

FCC primitive lattice + 2-atom basis at $(0,0,0)$ and $(¼,¼,¼)$.
**Same** for diamond (both atoms = C), zincblende (one Zn, one S),
GaAs (Ga + As), AlP, BN cubic, etc.

```text
Primitive (FCC):
  a₁ = (a/2)·(0, 1, 1)
  a₂ = (a/2)·(1, 0, 1)
  a₃ = (a/2)·(1, 1, 0)
  Basis: (0, 0, 0), (¼, ¼, ¼)
```

```python
a = 6.74  # bohr (Si)
sysp = vq.PeriodicSystem(
    dim=3,
    lattice_vectors=(a / 2) * np.array([   # FCC primitive vectors, one per row
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]),
    unit_cell=[
        vq.Atom(14, [0.00, 0.00, 0.00]),          # Si at primitive-cell origin
        vq.Atom(14, [a / 4, a / 4, a / 4]),       # Si at fractional (1/4,1/4,1/4), Cartesian bohr
    ],
)
```

**Real materials** (semiconductor lattice): diamond (C,
a = 6.74 bohr), Si (a = 10.26 bohr), Ge (a = 10.69 bohr);
zincblende ZnS, GaAs, InP, InSb, AlP, AlAs, BN(c).

**Worked examples** ship under
[`examples/periodic/Si-diamond/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Si-diamond)
and
[`examples/periodic/C-diamond/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/C-diamond),
each provides RHF + RKS-PBE + RKS-B3LYP variants with pob-TZVP.

#### Rocksalt (cF + 2-atom basis)

```{image} /_static/lattices/cubic-rocksalt-MgO.svg
:alt: Rocksalt MgO, 2×2×2 supercell
:width: 320px
:align: right
```

Same FCC primitive lattice, but the second atom sits at $(½,½,½)$,
i.e. **two interpenetrating FCC lattices** offset by half the
conventional cube diagonal.

```text
Primitive (FCC):
  a₁ = (a/2)·(0, 1, 1)
  a₂ = (a/2)·(1, 0, 1)
  a₃ = (a/2)·(1, 1, 0)
  Basis: (0, 0, 0), (½, ½, ½)
```

**Real materials** (ionic lattice): NaCl (a = 10.65 bohr), KCl, LiF,
MgO, CaO, all the "alkali halide" / "alkaline-earth oxide"
prototypes. The Madelung-constant cross-validation in
[`examples/periodic/input-madelung-constants.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-madelung-constants.py)
runs on rocksalt + zincblende + CsCl.

**Worked examples** ship under
[`examples/periodic/MgO-rocksalt/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/MgO-rocksalt)
(canonical; also paired with CRYSTAL14 parity inputs),
[`examples/periodic/LiH-rocksalt/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/LiH-rocksalt),
and
[`examples/periodic/NaCl-rocksalt/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/NaCl-rocksalt).
Each provides RHF + RKS-PBE + RKS-B3LYP variants with pob-TZVP.
The MgO sto-3g RHF example is reproduced inline above.

#### CsCl (cP + 2-atom basis)

```{image} /_static/lattices/cubic-cscl-CsCl.svg
:alt: CsCl-type lattice, 2×2×2 supercell
:width: 320px
:align: right
```

Simple cubic (not BCC!) with one atom at the corner, one at the body
center. Distinct from BCC because the two basis atoms are
different elements, the center atom isn't related by translation
to the corner atom.

```text
a₁ = (a, 0, 0); a₂ = (0, a, 0); a₃ = (0, 0, a)
Basis: Cs at (0, 0, 0), Cl at (½, ½, ½)
```

**Real materials** (ionic): CsCl (a = 7.78 bohr), CsBr, NH₄Cl
(high-T), TlBr.

#### Cubic perovskite (cP + 5-atom basis)

```{image} /_static/lattices/cubic-perovskite-SrTiO3.svg
:alt: Cubic perovskite SrTiO3, 2×2×2 supercell
:width: 320px
:align: right
```

ABX₃, one A cation at the corner, one B cation at the center, three
X anions on the face centers.

```text
a₁ = (a, 0, 0); a₂ = (0, a, 0); a₃ = (0, 0, a)
Basis:
  A at (0, 0, 0)        # large cation, e.g. Sr, Ba, Ca
  B at (½, ½, ½)        # small cation, e.g. Ti, Zr, Mn
  X at (½, ½, 0)        # anion
  X at (½, 0, ½)
  X at (0, ½, ½)
```

**Real materials** (ionic, ABX₃, class includes both insulators and
metallic conductors depending on composition): SrTiO₃, BaTiO₃ (rt
cubic), CaTiO₃, KMnF₃ (above the ferro-distortion transition).
Ferroelectrics are typically *distorted* perovskites, the cubic
phase is the high-symmetry parent.

### Hexagonal family

#### Simple hexagonal / hP

```text
a₁ = (a, 0, 0)
a₂ = (-a/2, a√3/2, 0)
a₃ = (0, 0, c)

Basis: (0, 0, 0)
```

Real materials: rare on its own (graphene is hP **2D**); most
hexagonal solids have a 2-atom basis (HCP) or 4-atom (graphite).

#### Hexagonal close-packed / HCP (hP + 2-atom basis)

```{image} /_static/lattices/hex-hcp-Mg.svg
:alt: HCP Mg, 3×3×2 supercell
:width: 320px
:align: right
```

```text
Same lattice vectors as hP.
Basis: (0, 0, 0), (⅓, ⅔, ½)

For ideal HCP:  c/a = sqrt(8/3) ≈ 1.633
```

**Real materials** (metals): Mg (c/a = 1.624), Zn (c/a = 1.86,
non-ideal), Be, Ti (α-phase), Co (α), Os, Ru.
**Metal-typical** (see the [metals note](#a-note-on-metals)).

#### Wurtzite (hP + 4-atom basis)

```{image} /_static/lattices/hex-wurtzite-ZnO.svg
:alt: Wurtzite ZnO, 2×2×2 supercell
:width: 320px
:align: right
```

Hexagonal binary structure analogous to zincblende. ABAB stacking
of close-packed planes (vs ABCABC for zincblende).

```text
Same lattice vectors as hP.
Basis (X = small atom, Y = large atom):
  X at (0, 0, 0)
  X at (⅓, ⅔, ½)
  Y at (0, 0, u)            # u ≈ 3/8 ideal
  Y at (⅓, ⅔, ½ + u)
```

**Real materials** (ionic / wide-gap semiconductors): ZnO, GaN, AlN,
CdS, SiC (4H/6H polytypes), BN(h).

**Worked example** ships under
[`examples/periodic/ZnO-wurtzite/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/ZnO-wurtzite),
RHF + RKS-PBE + RKS-B3LYP variants with pob-TZVP.

#### Graphene (2D) and graphite (3D)

```{image} /_static/lattices/hex-graphene-C.svg
:alt: Graphene 2×2 supercell with vacuum
:width: 320px
:align: right
```

Graphene is a **2D** hexagonal lattice with a 2-atom basis (the two
sublattices A and B), a semi-metal. vibe-qc handles 2D systems as
genuine `dim=2` slabs. Build one with `vq.slab_2d`, which takes the two
in-plane lattice vectors and the atoms at their real `z`. There is no
vacuum gap: the third lattice column is synthesized bookkeeping and the
SCF energy does not depend on it.

```python
a = 4.6487  # bohr (graphene C-C ≈ 1.42 Å → a ≈ 2.46 Å)

# Two in-plane vectors + Cartesian-z atoms (bohr). No third vector.
sysp = vq.slab_2d(
    [a,       0.0,                 0.0],
    [a / 2,   a * np.sqrt(3) / 2,  0.0],
    [vq.Atom(6, [0.0,     0.0,                 0.0]),
     vq.Atom(6, [a / 2,   a * np.sqrt(3) / 6,  0.0])],
)
assert sysp.dim == 2
```

`jk_method="auto"` sends this to the vacuum-free `SLAB_EWALD_2D` gauge.
Passing a bulk route such as `jk_method="gdf"` raises.

Graphite is graphene stacked along $c$ with AB Bernal stacking;
4-atom basis.

### Tetragonal, orthorhombic, monoclinic, triclinic

These cover lower-symmetry structures: distorted perovskites,
elemental tin (β-Sn is tI), molecular crystals (most are
monoclinic or triclinic), high-Tc superconductors (oC, oI).

vibe-qc accepts arbitrary `lattice` vectors, so any of these works
out of the box. For symmetry analysis, `attach_symmetry(sysp)`
runs spglib and returns the space group regardless of which
crystal-system label you'd pick by eye.

#### Rutile (tP / P4₂/mnm), TiO₂

```{image} /_static/lattices/tet-rutile-TiO2.svg
:alt: Rutile TiO2, 2×2×3 supercell
:width: 320px
:align: right
```

The canonical tetragonal ionic insulator: 6-atom unit cell, two
Ti at $(0,0,0)$ and $(½,½,½)$, four O at $\pm(u, u, 0)$ and
$\pm(½ + u, ½ - u, ½)$ with $u ≈ 0.305$.

**Real materials** (ionic, wide-gap insulator / transition-metal
oxide): TiO₂-rutile, MgF₂, SnO₂, RuO₂ (metallic), IrO₂ (metallic).
The rutile structure type covers both wide-gap insulators
(TiO₂) and **metallic conductors** (RuO₂); the lattice geometry
is the same, the electronic-structure regime is different.

**Worked example** ships under
[`examples/periodic/TiO2-rutile/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/TiO2-rutile),
RHF + RKS-PBE + RKS-B3LYP with pob-TZVP.

#### Corundum (hR / R-3c), α-Al₂O₃

```{image} /_static/lattices/trig-corundum-Al2O3.svg
:alt: Corundum α-Al2O3, 2×2×1 supercell
:width: 320px
:align: right
```

Trigonal (rhombohedral). Both the hexagonal-settings (30-atom)
and rhombohedral-primitive (10-atom) conventions work in vibe-qc.

**Real materials** (ionic / wide-gap insulator): α-Al₂O₃ (sapphire),
Cr₂O₃, Fe₂O₃ (hematite), V₂O₃, Ti₂O₃. Sapphire is the canonical
example; the M₂O₃ family includes both insulators (Al₂O₃, Cr₂O₃)
and Mott-Hubbard insulators / metals (V₂O₃, Ti₂O₃).

**Worked example** ships under
[`examples/periodic/Al2O3-corundum/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Al2O3-corundum),
sto-3g RHF (provenance + smoke test).

#### α-Quartz (hP / P3₂21), SiO₂

```{image} /_static/lattices/trig-quartz-SiO2.svg
:alt: alpha-Quartz SiO2, 2×2×1 supercell
:width: 320px
:align: right
```

Trigonal SiO₂; 9-atom unit cell (3 SiO₂ formula units). Two
helical handednesses (α-quartz is chiral: P3₂21 left-handed,
P3₁21 right-handed); both spgroups occur naturally.

**Real materials** (ionic / wide-gap insulator): SiO₂ in its
α-quartz polymorph is the most common natural form.

**Worked example** ships under
[`examples/periodic/SiO2-alpha-quartz/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/SiO2-alpha-quartz),
RHF + RKS-PBE + RKS-B3LYP with pob-TZVP.

## Primitive vs conventional cells

The primitive cell is the **smallest** cell that tiles all of space
under lattice translations. For BCC + FCC lattices, the primitive
cell is *not* the cubic cell most textbooks draw, it's a rhombohedron
half the volume (BCC) or quarter the volume (FCC).

You'd think you always want the primitive cell, fewer atoms = faster
SCF. Two reasons to use the conventional cell instead:

1. **Symmetry.** Cubic cells make x, y, z directions explicit. The
   k-mesh is naturally aligned to the conventional cell's axes; band
   paths are conventionally specified in conventional-cell
   coordinates (Γ, X, M, R, K).
2. **Multi-atom bases.** Diamond, NaCl, etc. need 2-atom bases on
   the FCC primitive cell. Some users find it easier to build the
   conventional cell with a multi-atom basis (4 + 4 = 8 atoms for
   diamond) than to set up the primitive cell + 2-atom basis. The
   energy is identical; the multi-k cost is roughly 4× higher for
   the same physical system.

vibe-qc respects whichever cell you build. Use `detect_spacegroup`
for non-mutating inspection, `attach_symmetry` when downstream
k-point / integral code should see `system.symmetry`, and
`symmetrise` when an imported structure should be standardised before
calculation:

```python
import vibeqc as vq

sg = vq.detect_spacegroup(sysp, symprec=1e-4)
print(f"Space group: {sg.international_symbol} (#{sg.number})")
print(f"Hall number: {sg.hall_number}")
print(f"#operations: {sg.order}")

standard = vq.symmetrise(sysp, symprec=1e-4)
sysp_clean = standard.system
print(f"RMS cleanup: {standard.report.rms_displacement_bohr:.3e} bohr")

primitive = vq.symmetrise(sysp, symprec=1e-4, to_primitive=True).system
```

## The reciprocal lattice and the Brillouin zone

For lattice vectors $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$, the
**reciprocal lattice** is generated by

$$
\mathbf{b}_1 = 2\pi \frac{\mathbf{a}_2 \times \mathbf{a}_3}{V}
\quad
\mathbf{b}_2 = 2\pi \frac{\mathbf{a}_3 \times \mathbf{a}_1}{V}
\quad
\mathbf{b}_3 = 2\pi \frac{\mathbf{a}_1 \times \mathbf{a}_2}{V}
$$

with $V = \mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)$
the unit-cell volume. A general reciprocal-lattice vector is

$$
\mathbf{G} = m_1 \mathbf{b}_1 + m_2 \mathbf{b}_2 + m_3 \mathbf{b}_3
\quad m_i \in \mathbb{Z}.
$$

The **first Brillouin zone (BZ)** is the Wigner-Seitz cell of the
reciprocal lattice, the set of $\mathbf{k}$ closer to $\mathbf{G}=0$
than to any other reciprocal-lattice point. Bloch's theorem says
every electronic state in a periodic crystal can be labeled by a
$\mathbf{k}$ in this region.

The BZ shape depends on the Bravais lattice:

| Real-space lattice | BZ shape |
|---|---|
| Simple cubic (cP) | Cube |
| BCC (cI) | Rhombic dodecahedron |
| FCC (cF) | Truncated octahedron |
| Hexagonal (hP) | Hexagonal prism |
| Trigonal (hR) | Distorted hexagonal prism |

vibe-qc doesn't currently provide a BZ-shape visualization tool;
external packages like [seekpath](https://github.com/giovannipizzi/seekpath)
or [pymatgen](https://pymatgen.org/) do this well.

## High-symmetry points

These are the special $\mathbf{k}$-points where the BZ has extra
symmetry. Band structures conventionally trace a path through them.
The Setyawan-Curtarolo (2010) convention is what almost everyone
uses today:

| BZ shape | High-symmetry points |
|---|---|
| Cube (cP) | Γ, X, M, R |
| Rhombic dodecahedron (cI / BCC) | Γ, H, N, P |
| Truncated octahedron (cF / FCC) | Γ, X, K, L, U, W |
| Hexagonal prism (hP) | Γ, M, K, A, L, H |

The $\Gamma$ point is always at the BZ center, $\mathbf{k} = (0,0,0)$
in any units. Other points are at half- or quarter-cell positions
in reciprocal space.

For example, on FCC (using conventional-cell reciprocal vectors):

```text
Γ  =  (0,    0,    0   )         # BZ centre
X  =  (½,    0,    ½   )         # face centre of cube
L  =  (½,    ½,    ½   )         # body-diagonal corner of cube
K  =  (3/8,  3/8,  3/4 )         # midpoint of two adjacent X's
W  =  (½,    ¼,    ¾   )         # corner of square face
```

The standard FCC band path is `Γ → X → W → K → Γ → L → U → W → L
→ K`, which traces all of the BZ's irreducible piece.

## Setting up lattices in vibe-qc

```{tip}
For any non-trivial structure, **import from a CIF / POSCAR file**
rather than typing lattice vectors by hand. Vibe-qc reads:

* `read_cif(path)` / `Crystal.from_cif(path)` — Crystallographic
  Information File (the standard format for published structures);
  pass `symmetrise=True` for a standardised `PeriodicSystem`.
* `read_poscar(path)` — VASP 5 POSCAR; pass
  `symmetrise=True` to return a `SymmetriseResult` with a
  standardised `PeriodicSystem`.
* `from_xyz(path, periodic=True)` — Extended-XYZ with a
  `Lattice="..."` comment tag; pass `symmetrise=True` for the
  same standardisation pipeline.

The Materials Project (<https://materialsproject.org/>), Crystallography
Open Database (<https://crystallography.net/cod/>), and ICSD
(commercial) all export CIF directly.
```

For test systems and small examples, building by hand is
straightforward:

```python
import numpy as np
import vibeqc as vq

# --- FCC primitive (Cu, a = 3.61 Å) -----------------------------
a = 6.82  # bohr
fcc_primitive = (a / 2) * np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
])
cu_fcc = vq.PeriodicSystem(
    dim=3, lattice=fcc_primitive,
    unit_cell=[vq.Atom(29, [0, 0, 0])],
)

# --- Diamond (Si, a = 5.43 Å) -----------------------------------
a = 10.26  # bohr
si_diamond = vq.PeriodicSystem(
    dim=3, lattice=(a / 2) * np.array([
        [0, 1, 1], [1, 0, 1], [1, 1, 0],
    ]),
    unit_cell=[
        vq.Atom(14, [0.00, 0.00, 0.00]),
        vq.Atom(14, [0.25, 0.25, 0.25]),
    ],
)

# --- Rocksalt (NaCl, a = 5.64 Å) --------------------------------
a = 10.65  # bohr
nacl = vq.PeriodicSystem(
    dim=3, lattice=(a / 2) * np.array([
        [0, 1, 1], [1, 0, 1], [1, 1, 0],
    ]),
    unit_cell=[
        vq.Atom(11, [0.0, 0.0, 0.0]),
        vq.Atom(17, [0.5, 0.5, 0.5]),
    ],
)

# --- Hexagonal close packed (Mg, a = 3.21 Å, c = 5.21 Å) --------
a, c = 6.06, 9.85   # bohr
mg_hcp = vq.PeriodicSystem(
    dim=3, lattice=np.array([
        [a,    0.0,                 0.0],
        [-a/2, a * np.sqrt(3) / 2,  0.0],
        [0.0,  0.0,                 c  ],
    ]),
    unit_cell=[
        vq.Atom(12, [0.0,    0.0,    0.0]),
        vq.Atom(12, [1.0/3,  2.0/3,  0.5]),
    ],
)
```

## What's next

* For **how vibe-qc samples the BZ** of any of these lattices,
  Monkhorst-Pack meshes, Γ-only vs Γ-centered, IBZ reduction, mesh
  convergence, k-paths for band structures, see
  [k-point meshes](k_points.md).
* For **how to run a periodic SCF** on one of these lattices,
  EWALD_3D Coulomb dispatch, periodic Becke partition for DFT,
  level shifts and smearing for SCF stability, see
  [Ewald summation](ewald.md) and
  [SCF convergence](scf_convergence.md).
* For **band-structure plots** along the high-symmetry paths above,
  see [band structure](band_structure.md) and
  [band structure of an H-chain](../tutorial/band_structure.md).
* For **symmetry exploitation** in periodic SCF (compressing the
  lattice-matrix set under the space group), see
  [symmetry-aware periodic storage](../tutorial/symmetry_storage.md).

## References

* M. I. Aroyo, ed., *International Tables for Crystallography*,
  Vol. A, "Space-Group Symmetry" (8th ed., Wiley, 2024), the
  authoritative reference for space groups, Wyckoff positions,
  and crystallographic conventions.
* W. Setyawan and S. Curtarolo, "High-throughput electronic band
  structure calculations: Challenges and tools," *Comput. Mater.
  Sci.* **49**, 299 (2010), defines the Bravais-lattice-specific
  k-paths used today.
* C. Bradley and A. Cracknell, *The Mathematical Theory of
  Symmetry in Solids* (Oxford, 1972), the textbook reference for
  space-group representations.
* Spglib documentation, <https://spglib.readthedocs.io/>, the
  symmetry-finder vibe-qc uses internally; particularly the
  "Definition of unit cell" page on conventions.

## See also

* [`periodic_methods.md`](periodic_methods.md): comparative tour of
  the periodic-SCF kernels (BIPOLE, GDF, GPW/GAPW) the Bravais
  lattices feed into, bonding-organised example gallery, and the
  surface-reactions workflow.
