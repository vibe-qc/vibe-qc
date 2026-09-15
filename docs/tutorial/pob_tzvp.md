# Why solid-state calculations use `pob-TZVP`

Standard molecular basis sets, the Pople and Dunning families, 
were designed for *isolated molecules*. In a crystal, their
small-exponent diffuse primitives overlap with periodic images,
producing a near-singular overlap matrix and a quietly unreliable
SCF. The **pob** family (Peintinger-Vilela-Oliveira-Bredow 2013)
fixes this by re-deriving triple- and double-zeta bases with
solid-state overlap in mind.

## The one-liner

vibe-qc ships three pob bases in ``basis_library/custom/`` and
exposes them by name:

```python
from vibeqc import BasisSet
basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
```

Every solid-state tutorial in this series ([periodic HF](periodic_hf.md),
[periodic KS-DFT](periodic_dft.md), [band structure](band_structure.md))
uses ``pob-tzvp`` by default. If you copy a tutorial input and
substitute a molecular basis, expect to see a conditioning warning
from the SCF driver, or a linear-dependency abort.

## Why a molecular basis goes wrong in a crystal

A typical molecular triple-zeta basis, say ``def2-tzvp`` on oxygen, 
includes a smallest-exponent $s$ primitive around $\alpha \approx
0.1$ bohr⁻². For an isolated molecule, that $\alpha$ gives a
Gaussian centered on O with a radius $1/\sqrt{\alpha} \approx 3$ bohr
- fine, doesn't overlap meaningfully with the next O atom
typically $\sim 5$ bohr away.

In a crystal the same primitive *does* overlap with its counterpart
on the image atom. For NaCl with Na-Cl distance $\sim 5.3$ bohr,
the overlap between two $\alpha = 0.1$ primitives separated by
5.3 bohr is

$$
\langle \chi_\alpha(\mathbf{r}) | \chi_\alpha(\mathbf{r} - \mathbf{T}) \rangle
  = \bigl( \tfrac{2\alpha}{\pi} \bigr)^{3/2} \cdot \bigl( \tfrac{\pi}{2\alpha} \bigr)^{3/2} \, e^{-\alpha |\mathbf{T}|^2 / 2}
  = e^{-\alpha |\mathbf{T}|^2 / 2}
  \approx e^{-1.4}
  \approx 0.24,
$$

- a *quarter* of the self-overlap. Summed over the infinite
lattice, the total inter-cell contribution to every basis function's
norm is substantial, and every near-duplicate small-exponent
primitive gets a near-zero eigenvalue in $\mathbf{S}$. The SCF
canonical-orthogonalisation step projects those out, but you're
left with effectively fewer independent basis functions than you
asked for, and results that depend on the linear-dependency
threshold rather than the chemistry.

## What pob does

Peintinger-Vilela-Oliveira-Bredow (2013) took the def2-TZVP /
def2-DZVP molecular bases and, for every element from H through
Br, **systematically trimmed or replaced** the primitives with
$\alpha < \alpha_\text{min}^\text{solid-state}$, where
$\alpha_\text{min}^\text{solid-state}$ is chosen so that the
inter-cell overlap integral for typical crystalline densities
stays below a tolerance. The result is:

- **pob-TZVP**, triple-zeta valence + polarization, the most
  commonly used solid-state basis.
- **pob-TZVP-rev2**, a revised version with tighter polarization
  functions for specific element subsets.
- **pob-DZVP-rev2**, double-zeta equivalent, cheaper, for
  screening work.

Covers H through Br at the all-electron level. For elements 4d/5d/5f+
the bundled pob-TZVP-REV2 runtime basis carries ECP-based definitions
(the `Z+200` convention) that vibe-qc resolves and applies via
libecpint, validated against PySCF.pbc for PBE/pob-TZVP-REV2 + ECP.

## What you see in practice

Running a tutorial's H₂ molecular crystal ([input-h-chain-uniform.py](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-uniform.py))
with three bases and tracking SCF convergence:

| Basis | nbf / atom | n_iter | Conv. |
|---|---:|---:|---|
| STO-3G | 1 | 2 | ✓ (too small to matter) |
| 6-31G\* | 2 | oscillates | ✗ |
| pob-TZVP | 6 | ~8 | ✓ |

The `pob-TZVP` calculation converges smoothly because no basis
function is close to linear-dependent with any of its periodic
images. The `6-31G*` case oscillates on the same geometry, not
because the physics is wrong but because the basis is.

## When you can skip pob

- **Large unit cells** where the molecular-limit approximation is
  exact (30+ bohr separation between relevant atoms): any molecular
  basis works because image overlap is negligible.
- **Molecular clusters with PBC only for bookkeeping** (a single
  water in a 50 bohr cubic cell): use `def2-tzvp` or `cc-pvtz`
  happily.

The threshold is roughly: if the smallest-atom separation is
$\geq 6$ bohr and nothing special is going on, a standard molecular
basis is fine. Below that, switch to pob.

## Parsing your own solid-state basis

vibe-qc's ``basis_crystal`` module parses CRYSTAL-format per-element
files and emits libint-compatible `.g94` text:

```python
from vibeqc.basis_crystal import parse_crystal_atom_basis_file, emit_g94

atom = parse_crystal_atom_basis_file("my-basis/06_C")
with open("my-basis.g94", "w") as f:
    f.write(emit_g94([atom]))
```

Drop the resulting `my-basis.g94` into `basis_library/custom/` and
run `./scripts/setup_basis_library.sh` to pick it up. See
[`user_guide/basis_sets.md`](../user_guide/basis_sets.md) for the
full CRYSTAL-format reference.

## Resources

Depends sharply on system. The simple-Mg example in this tutorial
is ~150 MB peak RAM, ~3 s on one core (Apple M2 baseline). Real
metal-oxide / metal-hydride targets at pob-TZVP (8-20 atoms / cell,
[4,4,4] mesh) typically run **~1-10 minutes per SCF on 4-8 cores**;
defect supercells push into the hour-per-SCF regime where C1a
[level shifting](periodic_scf_convergence.md) pays for itself.

## References

- **pob-TZVP / pob-DZVP, original paper.** M. F. Peintinger, D. V.
  Oliveira, and T. Bredow, "Consistent Gaussian basis sets of
  triple-zeta valence with polarization quality for solid-state
  calculations," *J. Comput. Chem.* **34**, 451 (2013).
- **pob-TZVP-rev2 revision.** D. V. Oliveira, J. Laun, M. F.
  Peintinger, and T. Bredow, "BSSE-correction scheme for consistent
  Gaussian basis sets of double- and triple-zeta valence with
  polarization quality for solid-state calculations," *J. Comput.
  Chem.* **40**, 2364 (2019).
- **Linear-dependency canonical orthogonalisation.** P.-O. Löwdin,
  "On the non-orthogonality problem connected with the use of
  atomic wave functions in the theory of molecules and crystals,"
  *J. Chem. Phys.* **18**, 365 (1950).
- **CRYSTAL code**, for which pob was designed. A. Erba et al.,
  "CRYSTAL23: a program for computational solid state physics and
  chemistry," *J. Chem. Theory Comput.* **19**, 6891 (2023).

## Next

- [User guide, linear dependence + screening in periodic
  SCF](../user_guide/linear_dependence.md), the chapter the
  pob-* family was designed against. Walks through the v0.7
  diagnostic + fix stack (`vq.eigs_preflight`,
  `vq.disambiguate_critical_overlap`, `auto_optimize_truncation`,
  `vq.make_basis(..., exp_to_discard=...)`).
- [Basis-set convergence](basis_convergence.md) for the molecular
  side of the basis-set question, Dunning's cc-pVXZ hierarchy and
  the CBS extrapolation.
- The [Peierls dimerisation](peierls.md) tutorial uses `pob-tzvp`
  on a 1D H-chain where the diffuse-basis problem is especially
  acute.
