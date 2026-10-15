# Symmetry-aware storage of lattice integrals (SYM3a)

For a periodic system with a non-trivial space group, many entries
of any lattice-summed integral are *redundant*: they are related to
one canonical representative by a symmetry operation. Pm-3m He at
6 bohr lattice has 48 point-group operations and ~19 cells in the
overlap lattice sum, yet only **3 orbits** of unique
$\mathbf{S}^{(g)}_{\mu\nu}$ entries. The other 16 are exact symmetry
images.

vibe-qc's Phase **SYM3a** ships orbit-reduced storage for the
periodic one-electron integrals (overlap, kinetic, nuclear
attraction). The wrappers below first compute the full lattice matrix,
then store representative sub-blocks for reconstruction on demand.
**Storage compression on a high-symmetry crystal can approach $|G|\times$**
(group order). These wrappers return both compressed and full forms, so
the call itself does not avoid the full build or its peak allocation.

This tutorial covers the storage API, the verify-symmetry diagnostic,
and the honest scope: SYM3a is **storage-only**. Kernel-level compute
reduction is available through the separate `symmetry_integrals_reduced`
APIs (Phase SYM3b), which evaluate representative cells before
reconstruction. Their scope and the shared infrastructure are described in
the [symmetry capability status](../symmetry_status.md). Storage ratios
alone do not establish fewer Fock contractions or a wall-clock speedup.

## The basic call

Three drop-in compressed variants of the standard one-electron
lattice builders:

```python
import vibeqc as vq
import numpy as np

# Simple cubic He, Pm-3m space group (#221), 48 point-group ops.
a = 6.0
sysp = vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(2, [0, 0, 0])])
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

opts = vq.LatticeSumOptions()
opts.cutoff_bohr = 10.0

# 1. Attach the space group via spglib (populates sysp.symmetry).
vq.attach_symmetry(sysp)
print(f"space group: {sysp.symmetry.international_symbol} "
      f"(#{sysp.symmetry.number}), |G| = {len(sysp.symmetry.operations)}")

# 2. Filter to symmorphic operations (zero fractional translation).
#    SYM3a's atom-pair-orbit construction requires symmorphic ops only;
#    non-symmorphic are dropped. For Pm-3m all 48 ops are symmorphic.
ops = vq.symmorphic_operations(sysp.symmetry.operations)

# 3. Compute overlap S(g) with orbit reduction.
reduced, full = vq.compute_overlap_lattice_with_orbits(
    basis, sysp, opts, ops,
)
print(f"  cells in lattice sum: {len(full.cells)}")
print(f"  orbit count:          {reduced.orbits.n_orbits}")
print(f"  compression ratio:    {reduced.orbits.compression_ratio:.2f}×")

# 4. Verify the full lattice-matrix set actually respects the symmetry
#    relations (sanity check for the cell-list / cutoff combination).
witness = vq.verify_lattice_matrix_set_symmetry(full, basis, sysp, ops,
                                                atol=1e-10)
print(f"  symmetry verified:    {witness['passes']}  "
      f"(max residual {witness['max_residual']:.1e})")
```

Output on this Pm-3m He system:

```
space group: Pm-3m (#221), |G| = 48
  cells in lattice sum: 19
  orbit count:          3
  compression ratio:    6.33×
  symmetry verified:    True  (max residual 1.1e-19)
```

The same call signature works for kinetic and nuclear-attraction
matrices:

```python
T_red, T_full = vq.compute_kinetic_lattice_with_orbits(basis, sysp, opts, ops)
V_red, V_full = vq.compute_nuclear_lattice_with_orbits(basis, sysp, opts, ops)
```

(Nuclear dispatches on `opts.coulomb_method`, orbit reduction works
identically for `DIRECT_TRUNCATED` and `EWALD_3D`, since the orbit
partition is operator-only and method-independent.)

## Standardise imported cells first

For a cell read from CIF/POSCAR/Extended-XYZ, small coordinate noise can
leave atoms slightly off their ideal Wyckoff positions. GS3 wraps the
standardise-then-compress workflow so users do not have to remember the
manual sequence `symmetrise` → rebuild basis → filter symmorphic ops:

```python
result = vq.compute_overlap_lattice_symmetrised_with_orbits(
    sysp_from_file,
    basis_name="sto-3g",
    options=opts,
    symprec=1e-3,
)

print(result.report.spacegroup_after.international_symbol)
print(result.reduced.compression_ratio)

# The result is also unpackable for the common reduced/full pair.
S_red, S_full = result
```

The kinetic and nuclear siblings have the same signature:

```python
T = vq.compute_kinetic_lattice_symmetrised_with_orbits(sysp, "sto-3g", opts)
V = vq.compute_nuclear_lattice_symmetrised_with_orbits(sysp, "sto-3g", opts)
```

Each result carries the standardised `PeriodicSystem`, the rebuilt
`BasisSet`, the `SymmetriseReport`, the symmorphic operations actually
used for compression, the `OrbitReducedLatticeMatrix`, and the full
`LatticeMatrixSet`.

## Compression scales with cell-list size

The compression ratio is bounded above by $|G|$ (the group order)
and approaches that bound as the cell list grows. For Pm-3m He, two
data points:

| Cell parameter | Cells in sum | Orbits | Compression |
|---|---:|---:|---:|
| 6.0 bohr | 19 | 3 | **6.33×** |
| 4.5 bohr | 33 | 5 | **6.60×** |

Tighter cells need more cells in the lattice sum (more image cells
falling within `cutoff_bohr`), so more orbits, but the ratio
keeps climbing toward $|G| = 48$. For a sufficiently extended cell
list, the only orbits left are the diagonal $g = 0$ block and a
small handful related by representative axes; the rest is symmetry
images.

For lower-symmetry cells the compression is correspondingly smaller:
a triclinic (P1) cell has $|G| = 1$ and compression = 1×. A 1D
chain with two H atoms per cell along x and full reflection symmetry
in (y, z) gives $|G| = 8$, useful but more modest than Pm-3m's 48.

## What's stored, what's not

`compute_overlap_lattice_with_orbits` returns a tuple
`(OrbitReducedLatticeMatrix, LatticeMatrixSet)`:

- The **`OrbitReducedLatticeMatrix`** holds:
  - `.orbits`, an `AtomPairOrbits` describing the partition of
    (source_atom, dest_atom, cell_index) triples into orbits, plus
    metadata (`n_orbits`, `n_triples`, `compression_ratio`).
  - `.representatives`, one matrix block per orbit (the canonical
    representative the others are symmetry-related to).
- The **`LatticeMatrixSet`** is the full uncompressed matrix, 
  exactly what the standard `compute_overlap_lattice` returns.
  Useful for downstream consumers that require full matrices. Separate
  compute-reduced APIs also reconstruct full matrices for such consumers.

To go from a reduced view back to a full one (for instance, after
loading the reduced view from disk):

```python
full_reconstructed = vq.reconstruct_lattice_matrix_set_c(
    reduced.representatives, reduced.orbits, basis,
)
# full_reconstructed equals the original 'full' to machine precision.
```

## When to use it

**SYM3a is most useful for analysis and disk I/O**:

- Saving lattice matrices to disk for later analysis (CRYSTAL-style
  `.f9`-equivalent files): write only the reduced representation;
  reconstruct on load.
- Inspecting the symmetry structure of a calculation: the
  `verify_lattice_matrix_set_symmetry` witness prints the maximum
  residual the symmetry relations are violated by, and is a
  diagnostic when something looks wrong with the lattice-sum cutoffs
  or the spglib analysis.
- Identifying candidate reductions to measure with the separate
  compute-reduced APIs.

`compute_*_lattice_with_orbits` continues to compute the full matrix first.
It does not transparently switch to representative-only integral evaluation.
Actual speedup depends on the admitted operations, representative cells,
reconstruction cost and downstream contractions, and must be measured.

## Theory

The sections below derive what an orbit is, why SYM3a uses atom-pair
orbits rather than single-cell ones, and how the verify-symmetry
witness checks that the stored representatives reproduce the full
lattice matrix.

### What an orbit is

The point group $\{R\}$ of the crystal (detected with spglib,
{vibeqc-cite}`togo_shinohara_tanaka_2024`) acts on the **(source atom,
destination atom, cell-index)** triple space:

$$
(A, B, \mathbf{g}) \;\longmapsto\; (R \cdot A, \, R \cdot B, \,
R \cdot \mathbf{g})
$$

(Translation parts of non-symmorphic operations are dropped, see
"Caveats" below.) Two triples that map to each other under any group
operation are in the **same orbit**. The matrix block
$\mathbf{S}^{(g)}_{[A,B]}$ for triple $(A, B, \mathbf{g})$ is then
related to the block of any orbit partner by the Wigner $\mathbf{D}$
matrices for the AOs on $A$ and $B$:

$$
\mathbf{S}^{(R \cdot \mathbf{g})}_{[R \cdot A, R \cdot B]} =
\mathbf{D}_A(R) \cdot \mathbf{S}^{(\mathbf{g})}_{[A, B]}
\cdot \mathbf{D}_B(R)^\top
$$

Pick one **representative** triple per orbit, store its block, and
every other block in the orbit is recoverable by one matrix
multiplication on each side. This is the symmetry treatment of
lattice-summed AO matrices in periodic LCAO codes described by
{vibeqc-cite}`dovesi_symmetry_lcao_1986`, with the Seitz action on AOs and
atom images of {vibeqc-cite}`zicovich_wilson_crystalline_orbitals_1998`. The compression ratio for storage is
exactly the average orbit size, bounded above by $|G|$.

### Why "atom-pair" orbits and not "single-cell"

The earlier SYM2b construction acts on **single cells** $\mathbf{g}$:
two cells are equivalent iff some operation maps one to the other.
That works for single-atom-basis cells but loses information for
multi-atom bases (the two H atoms in an H₂ chain unit cell are
distinguishable in the lattice-matrix block structure).

SYM2c (atom-pair orbits) acts on the (atom, atom, cell) **triple**
space, necessary for any cell with more than one atom. The
"compute_*_with_orbits" calls in SYM3a use SYM2c orbits internally;
that's why the underlying `AtomPairOrbits` class names (atoms, not
cells) are what the API exposes.

### The verify-symmetry witness

`verify_lattice_matrix_set_symmetry(lms, basis, system, ops, atol)`
returns a dict:

- `n_orbits`, number of orbit equivalence classes found.
- `compression_ratio`, `n_triples / n_orbits`.
- `max_residual`, maximum absolute difference between
  $\mathbf{D}_A(R) \mathbf{S}^{(g)}_{[A,B]} \mathbf{D}_B(R)^\top$
  and the directly-computed
  $\mathbf{S}^{(R \cdot g)}_{[R \cdot A, R \cdot B]}$ across all
  (orbit, operation) pairs.
- `passes`, whether `max_residual < atol`.

If `passes=False`, inspect the operator, AO maps and finite cell support.
Some triples may have orbit partners outside the cell list. For general
multi-atom cells, increasing an origin-centred radial cutoff does not by
itself guarantee closure under the atom-image shifts. A group-invariant
pair support may be required and defines a different finite approximation.
`require_closed=False` skips absent partners during orbit construction;
it does not repair the source or certify reconstruction. Retain the full
reference and verify reconstructed values before using the compressed form.

## Resources

- Pm-3m He at 6 bohr, sto-3g, cutoff 10 bohr: 19 cells, 3 orbits.
  Compute time: ~3 ms for `compute_overlap_lattice_with_orbits` on
  one core (the orbit identification dominates over the integral
  computation at this scale).
- The full `compute_overlap_lattice` allocation is retained while the
  compressed representation is constructed. Compression therefore adds
  temporary/retained storage during this call; peak-memory savings require
  a separately measured compute and storage plan.
- For larger systems (10+ atoms per cell, cutoff 15+ bohr), the
  orbit-identification cost becomes a few percent of the integral-
  build time, and the storage compression gives several MB of saved
  pickle / disk space per matrix.

## Caveats

- **Zero-fractional-translation subset.** These wrappers filter out
  operations with nonzero fractional translation. The shared periodic
  AO/Bloch transport supports more general operations, but that does not
  automatically extend this storage API or qualify its finite supports.
- **Storage-only wrappers.** The full lattice matrix is built before
  compression. The separate SYM3b APIs reduce selected one-electron work;
  neither a storage ratio nor their availability establishes a Fock-build
  speedup for every method.
- **Cell-list closure.** A symmetry orbit can have partners that
  fall outside `cutoff_bohr`; the default `require_closed=True`
  raises `ValueError` if any orbit isn't fully contained in the
  cell list. `require_closed=False` permits incomplete orbit membership
  by skipping absent partners; it is an exploratory option, not a
  production qualification or a promise of singleton fallback.
- **Spglib precision.** The space-group analysis uses spglib's
  default precision (`symprec=1e-5`). Heavily-strained cells or
  numerical relaxation residuals may push spglib into a lower-
  symmetry assignment; pass an explicit `symprec=` to
  `attach_symmetry` if needed.

## References

- **Bradley, C. J.; Cracknell, A. P.** *The Mathematical Theory of
  Symmetry in Solids: Representation Theory for Point Groups and
  Space Groups*. Oxford University Press (1972). The standard
  reference for crystallographic point groups, space groups, and
  the orbit-stabiliser construction vibe-qc uses for the atom-pair
  orbits.
- **Altmann, S. L.; Herzig, P.** *Point-group Theory Tables*. Oxford
  University Press (1994). The reference for the Wigner-D matrices
  in the *real* spherical-harmonic basis that vibe-qc's
  `wigner_d_real` returns; necessary for the AO-resolved orbit
  reconstruction.
- **Togo, A.; Shinohara, K.; Tanaka, I.** "Spglib: a software library
  for crystal symmetry search," *Sci. Technol. Adv. Mater.: Methods*
  **4**, 2384822 (2024) ({vibeqc-cite}`togo_shinohara_tanaka_2024`;
  preprint <https://arxiv.org/abs/1808.01590>, 2018). The library
  powering vibe-qc's `attach_symmetry` call.
- **Dovesi, R.; Saunders, V. R.; Roetti, C.; Orlando, R.;
  Zicovich-Wilson, C. M.; Pascale, F.; Civalleri, B.; Doll, K.;
  Harrison, N. M.; Bush, I. J.; D'Arco, P.; Llunell, M.;
  Causà, M.; Noël, Y.; Maschio, L.; Erba, A.; Rerat, M.;
  Casassa, S.** *CRYSTAL17 User's Manual*. University of Torino
  (2017). The canonical periodic-LCAO QC code; vibe-qc's
  symmetry-aware lattice-integral compression mirrors CRYSTAL's
  long-standing approach.

## Next

- The example script
  [`examples/periodic/input-nacl-symmetry.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-nacl-symmetry.py)
  walks through the precursor SYM2b / SYM2c machinery on NaCl
  (compress + reconstruct + Wigner-D rotation peek), which underpins
  the SYM3a integration this tutorial covers.
- [Symmetry capability status](../symmetry_status.md) distinguishes the
  available storage and compute APIs from shared transport and production
  qualification. [Periodic SCF convergence](periodic_scf_convergence.md)
  covers convergence controls separately from symmetry-reduction cost.
- [Ewald user guide](../user_guide/ewald.md), the orbit machinery
  composes cleanly with EWALD_3D nuclear attraction (the orbit
  partition is method-independent), so a high-symmetry 3D bulk SCF
  benefits from both.
