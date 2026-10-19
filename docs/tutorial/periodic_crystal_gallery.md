# A gallery of real crystal structures

**You'll learn:** the one shared pattern behind vibe-qc's real-crystal
example set, nine structure types, each run at RHF, B3LYP
({vibeqc-cite}`becke_1993`, {vibeqc-cite}`stephens_b3lyp_1994`), and PBE
({vibeqc-cite}`pbe_1996`), so
you can adapt any of them as a starting point for your own material.

This page is a map, not a derivation: for a full first-principles
walkthrough of *why* periodic Gaussian-basis SCF needs `pob-TZVP`
({vibeqc-cite}`peintinger_pob_tzvp_2013`), IBZ
k-mesh reduction, and how to read band structure/DOS/orbital cubes off
the result, see [Why solid-state calculations use
pob-TZVP](pob_tzvp.md) and the deep single-system walkthrough,
[Solid-state walkthrough: LiH rocksalt with
pob-TZVP](lih_pob_tzvp_solid_state.md). This page instead surveys the
other nine real structures in
[`examples/periodic/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/),
which have no tutorial coverage today, and shows the pattern once so you
don't have to read all ~27 nearly-identical scripts to find it.

## The shared pattern

Every script in the gallery below follows the same shape: a lattice
matrix and fractional-feeling Cartesian positions in Å, converted to
bohr, built into a `PeriodicSystem`, and run at Γ through the native GDF
route ({vibeqc-cite}`sun_berkelbach_gdf_2017`):

```python
from pathlib import Path
import numpy as np
from vibeqc import Atom, BasisSet, PeriodicSystem, run_periodic_job

ANG2BOHR = 1.0 / 0.529177210903

CELL_ANG = [[3.567, 0.0, 0.0], [0.0, 3.567, 0.0], [0.0, 0.0, 3.567]]
ATOM_DATA = [('C', [0.0, 0.0, 0.0]), ('C', [2.67525, 0.89175, 1.7835]), ...]

cell_bohr = np.array(CELL_ANG, dtype=float) * ANG2BOHR
Z_BY_SYM = {"C": 6, ...}   # every script carries its own small lookup
atoms = [Atom(Z_BY_SYM[sym], [x * ANG2BOHR for x in pos])
         for sym, pos in ATOM_DATA]

system = PeriodicSystem(3, cell_bohr, atoms)
basis = BasisSet(system.unit_cell_molecule(), "pob-tzvp")

run_periodic_job(
    system, basis,
    method="RHF",                 # or method="RKS", functional="b3lyp" / "pbe"
    output=Path(__file__).with_suffix(""),
    use_diis=True,
    fmixing_percent=30.0,         # mirrors CRYSTAL's default FMIXING
    max_iter=80,
    conv_tol_energy=1e-7,
)
```

`fmixing_percent=30.0` is there deliberately: it mirrors CRYSTAL's
default Fock/KS-matrix mixing ({vibeqc-cite}`dovesi_crystal14_2014`), which is part of why these scripts'
comments describe cross-checking against CRYSTAL14/PySCF references. The
route underneath is the native Γ-point GDF driver
(`run_rhf_periodic_gamma_gdf` / its RKS sibling), see [Choosing a
periodic method](periodic_methods_compared.md) if you need a different
route (multi-k, BIPOLE, GPW/GAPW).

## The nine structures

Each has its own directory under `examples/periodic/` with RHF,
RKS-B3LYP, and RKS-PBE variants sharing the pattern above (lattice
parameters below are the actual values each script's `CELL_ANG` uses -
verified by reading the scripts, not looked up separately):

| Structure | Type | Cell (Å) | Basis |
|---|---|---|---|
| [`C-diamond`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/C-diamond) | Diamond cubic carbon | cubic *a* = 3.567 | pob-TZVP |
| [`Si-diamond`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Si-diamond) | Diamond cubic silicon | cubic *a* = 5.431 | pob-TZVP |
| [`NaCl-rocksalt`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/NaCl-rocksalt) | Rocksalt (NaCl-type) | cubic *a* = 5.64 | pob-TZVP |
| [`MgO-rocksalt`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/MgO-rocksalt) | Rocksalt (NaCl-type) | cubic *a* = 4.21 (2-atom primitive) | pob-TZVP + a separate sto-3g Γ target |
| [`Ne-fcc`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Ne-fcc) | FCC solid neon | cubic *a* = 4.43 | pob-TZVP |
| [`SiO2-alpha-quartz`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/SiO2-alpha-quartz) | α-quartz | trigonal, *a* = 4.9134, *c* = 5.4052 | pob-TZVP |
| [`TiO2-rutile`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/TiO2-rutile) | Rutile | tetragonal, *a* = 4.5937, *c* = 2.9587 | pob-TZVP |
| [`ZnO-wurtzite`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/ZnO-wurtzite) | Wurtzite | hexagonal, *a* = 3.2495, *c* = 5.2069 | pob-TZVP |
| [`Al2O3-corundum`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/Al2O3-corundum) | α-corundum, space group *R*-3c (#167) | hexagonal, *a* = 4.7589, *c* = 12.991 | sto-3g (RHF only) |

(`LiH-rocksalt` also lives in this directory with the same three-method
pattern, but gets its own deep walkthrough, see [Solid-state
walkthrough: LiH rocksalt with pob-TZVP](lih_pob_tzvp_solid_state.md)
instead of repeating it here.)

### What the structure types imply

The nine entries are not an arbitrary list: they span the crystal systems
and bonding types a periodic Gaussian code has to get right, and each
structure's symmetry tells you in advance what the calculation can and
cannot show.

- **Diamond (C, Si)**, space group $Fd\bar 3m$, is a face-centred cubic
  lattice with a two-atom basis in which every atom sits on a site of
  tetrahedral ($T_d$) point symmetry. Covalent $sp^3$ bonding, no ionic
  charge transfer by symmetry, and an indirect gap: the valence-band maximum
  is at $\Gamma$ but the conduction-band minimum lies along $\Gamma$-$X$, so
  a band plot that stops at $\Gamma$ overestimates the gap. The atomic
  forces vanish identically by symmetry, so only the lattice constant can
  relax.
- **Rocksalt (NaCl, MgO, LiH)**, $Fm\bar 3m$, is two interpenetrating fcc
  lattices with every ion on an octahedral ($O_h$) site. It is the textbook
  ionic case: the Madelung constant is 1.7476, the Coulomb lattice sum is
  conditionally convergent and must be handled by Ewald summation
  ({vibeqc-cite}`ewald_lattice_sum_1921`; [Madelung constants](madelung_with_ewald.md)), and the forces again vanish
  by symmetry. Because the primitive cell has two atoms, the conventional
  cubic cell of eight holds four primitive cells; MgO ships both.
- **fcc neon**, $Fm\bar 3m$ with one atom per primitive cell (the shipped
  scripts use the four-atom conventional cubic cell, so their k-meshes are
  correspondingly smaller), is a closed-shell van der Waals solid: at the HF and semilocal-DFT levels shown
  here it is bound only weakly or not at all, since dispersion is missing,
  so it is a test of the machinery (one atom, high symmetry, a tiny
  irreducible wedge) rather than of the physics.
- **Alpha-quartz (SiO2)**, $P3_2 21$, is trigonal and chiral, with
  corner-sharing SiO4 tetrahedra and internal coordinates that are *not*
  fixed by symmetry: the Si-O bond length and the Si-O-Si angle can relax,
  so forces are non-zero and a fixed-geometry single point is a
  single point, not an equilibrium. The three-fold axis is also why a
  Gamma-centred k-mesh is required (an off-Gamma Monkhorst-Pack shift breaks
  the trigonal symmetry; `KPoints.monkhorst_pack` refuses it;
  {vibeqc-cite}`monkhorst_pack_1976`).
- **Rutile (TiO2)**, $P4_2/mnm$, is tetragonal with a free oxygen
  coordinate $u$; the two lattice parameters and $u$ together set the
  Ti-O distances. Ti$^{4+}$ has an empty $3d$ shell, so the gap is a
  charge-transfer gap from O $2p$ to Ti $3d$ and the semilocal functionals
  underestimate it.
- **Wurtzite (ZnO)**, $P6_3mc$, is hexagonal and polar: it has no inversion
  centre, so a spontaneous polarisation is allowed along $c$, the internal
  parameter $u$ is free, and the Zn $3d$ band sits only a few eV below the
  valence-band maximum where semilocal DFT places it too high. It also
  needs a Gamma-centred mesh.
- **Corundum (Al2O3)**, $R\bar 3c$, is rhombohedral (built here in the
  hexagonal setting with 30 atoms) with two free internal coordinates and
  the largest cell of the set; it is the reason the script builds the
  structure from space group and Wyckoff positions rather than by hand.

In short: the cubic structures test Ewald sums, k-mesh convergence and
symmetry reduction with no internal degrees of freedom; the trigonal,
tetragonal and hexagonal ones add free internal coordinates (and therefore
real forces), polarity, and the Gamma-centred-mesh requirement.

Two structures don't quite follow the plain pattern above:

- **`MgO-rocksalt`** additionally ships a CRYSTAL14 side-by-side
  comparison: `MgO-rocksalt-RHF-pobtzvp-mp8-gdf.py` builds the same
  2-atom FCC primitive cell CRYSTAL14 uses internally and runs it on an
  `8x8x8` k-mesh through the native multi-k GDF backend end to end,
  reporting the SCF energy, HOMO/LUMO gap, and IBZ-expansion metadata. A
  matching `.d12` CRYSTAL14 input and a `README-crystal14-parity.md` sit
  in the same directory for the actual parity comparison.
- **`Al2O3-corundum`** uses ASE's ({vibeqc-cite}`ase_2017`)
  `ase.spacegroup.crystal` to build the
  12 Al + 18 O conventional cell from the space group and Wyckoff
  parameters directly (`R`-3c, *a* = 4.7589 Å, *c* = 12.991 Å, from
  Springer Materials entry `sd_1400479`) rather than hand-listing atom
  positions, worth a look if you want to build a structure from its
  space group instead of pasting fractional coordinates by hand. It also
  only has an RHF/sto-3g variant, no B3LYP/PBE siblings.

## Adapting one for your own material

The fastest path to a new structure: copy the closest existing script
(same structure type, if one matches), replace `CELL_ANG` and
`ATOM_DATA` with your own lattice and basis, and keep everything else.
If your basis isn't pob-TZVP-covered for an element you need, see [Why
solid-state calculations use pob-TZVP](pob_tzvp.md#when-you-can-skip-pob)
for when a molecular basis is still safe to use instead.

## Next

- [Why solid-state calculations use pob-TZVP](pob_tzvp.md) for the basis
  theory these scripts all depend on.
- [Solid-state walkthrough: LiH rocksalt with
  pob-TZVP](lih_pob_tzvp_solid_state.md) for the deep single-system
  version of this pattern: k-mesh convergence, band structure, DOS, and
  orbital cubes.
- [Choosing a periodic method](periodic_methods_compared.md) for GDF vs.
  BIPOLE vs. GPW/GAPW, if Γ-point GDF isn't the right route for your
  system.
