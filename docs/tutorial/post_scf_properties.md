# Post-SCF properties: charges, bonds, dipole

Once an SCF is converged, vibe-qc gives you the standard chemical
interpretation tools: atomic charges (Mulliken and Löwdin), Mayer bond
orders, and the molecular dipole moment. This tutorial walks through
them on water, then a short comparison on H-bonded systems.

These properties are emitted automatically by ``run_job`` when it has
enough information; you can also call the helpers directly to get the
raw arrays for plotting or downstream analysis.

## A single call

The simplest route is to let `run_job` emit the population analysis for
you: run water at HF/6-31G\* and the charges, bond orders, and dipole
land in the `.out` automatically.

```python
from vibeqc import Atom, Molecule, run_job

mol = Molecule([
    Atom(8, [ 0.0,  0.00,  0.00]),
    Atom(1, [ 0.0,  1.43, -0.98]),
    Atom(1, [ 0.0, -1.43, -0.98]),
])
run_job(mol, basis="6-31g*", method="rhf", output="water")
```

(Positions are in bohr. ``Molecule.from_xyz("examples/h2o.xyz")``
works too if you have an XYZ file.)

Open ``water.out``, after the SCF trace and orbital-energy table
you'll find (HF/6-31G\*, experimental geometry):

```
  Atomic charges
  ----------------------------------------------------
     #  elem    Mulliken      Löwdin
  ----------------------------------------------------
     1     O   -0.910152   -0.646928
     2     H    0.455076    0.323464
     3     H    0.455076    0.323464
  ----------------------------------------------------
          sum  -0.000000

  Bond orders (Mayer)
  ----------------------------------------
          atoms    bond order
  ----------------------------------------
  O1   — H2          0.7628
  O1   — H3          0.7628

  Dipole moment (origin: centre of mass)
  ----------------------------------------------------
     component     a.u. (e·bohr)       Debye
  ----------------------------------------------------
             x       -0.00000000     -0.0000
             y       -0.00000000     -0.0000
             z       -0.81069504     -2.0606
           |μ|        0.81069504      2.0606
```

## What the numbers mean

- **Mulliken charges** partition the electron density according to which
  basis function it was expanded in. Simple, interpretable, but
  basis-set dependent, a polarization function added to H shifts
  charge onto it.
- **Löwdin charges** partition in the orthonormalised AO basis
  (S^{-1/2} transform). Less basis-set sensitive, generally preferred
  for trend analysis across systems.
- **Mayer bond orders** generalise the Wiberg index to non-orthogonal
  bases. ``B_ij ≈ 1`` for single bonds, ``≈ 2`` for double, ``≈ 3``
  for triple. Values under ~0.1 are hidden by default.
- **Dipole moment** is computed with the origin at the center of mass,
  which makes it origin-invariant for neutral molecules. For ions pass
  an explicit ``origin=`` to ``dipole_moment()`` if you want a
  different reference.

## Pulling the arrays yourself

When you need raw data for plotting or fitting, call the helpers
directly, they take the same ``result`` object you get back from
``run_rhf`` / ``run_job``.

```python
import numpy as np
from vibeqc import (
    Atom, BasisSet, Molecule, run_rhf,
    mulliken_charges, loewdin_charges,
    mayer_bond_orders, dipole_moment,
)
from vibeqc.properties import prominent_bonds

mol = Molecule([
    Atom(8, [ 0.0,  0.00,  0.00]),
    Atom(1, [ 0.0,  1.43, -0.98]),
    Atom(1, [ 0.0, -1.43, -0.98]),
])
basis = BasisSet(mol, "6-31g*")
result = run_rhf(mol, basis)

q_mull = mulliken_charges(result, basis, mol)      # (n_atoms,) np.ndarray
q_lowd = loewdin_charges(result, basis, mol)       # same shape

bonds = mayer_bond_orders(result, basis, mol)      # (n_atoms, n_atoms)
print(prominent_bonds(bonds, mol, threshold=0.10)) # [(i, j, B_ij), ...]

mu = dipole_moment(result, basis, mol)
print(f"|μ| = {mu.total_debye:.3f} D")
```

## Basis-set dependence: a quick sweep

Mulliken charges drift significantly between basis sets, run the same
water molecule at three levels and print the oxygen charge:

```python
mol = Molecule([
    Atom(8, [ 0.0,  0.00,  0.00]),
    Atom(1, [ 0.0,  1.43, -0.98]),
    Atom(1, [ 0.0, -1.43, -0.98]),
])

for basis_name in ("sto-3g", "6-31g*", "def2-tzvp"):
    basis = BasisSet(mol, basis_name)
    r = run_rhf(mol, basis)
    q = mulliken_charges(r, basis, mol)
    print(f"{basis_name:12s}  q_O = {q[0]:+.3f}  μ (D) = "
          f"{dipole_moment(r, basis, mol).total_debye:.3f}")
```

Output (HF, experimental geometry):

```
sto-3g        q_O = -0.418  μ (D) = 1.725
6-31g*        q_O = -0.910  μ (D) = 2.061
def2-tzvp     q_O = -0.685  μ (D) = 2.008
```

Two lessons:

1. **Mulliken is sensitive to basis.** ``q_O`` swings half an electron
   across these sets, never compare absolute Mulliken charges between
   different bases, different codes, or different molecules unless you
   hold the basis fixed. Löwdin charges are more stable but not
   bulletproof either.
2. **Dipole moment is well-behaved.** ``6-31g*`` already overshoots
   slightly; TZVP pulls it back toward 2.0 D. The experimental
   gas-phase water dipole is 1.85 D. The residual over-estimate comes
   from missing electron correlation, rerun the sweep with
   ``method="rks", functional="PBE"`` and the dipole drops further.

## Following the H-bond in a water dimer

Run the dimer example (`[`examples/molecular/input-h2o-dimer-opt.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dimer-opt.py)`) and load
``output-h2o-dimer-opt.molden`` in Avogadro. The ``.out`` file already
lists the Mulliken charges at the optimized geometry, note the
donor-hydrogen charge is noticeably larger than the free hydrogens,
and the oxygen of the donor water is slightly more negative than the
acceptor. That's the H-bond polarization, visible in the numbers.

Same in Python:

```python
from vibeqc import Molecule, BasisSet, run_rhf, mulliken_charges
mol = Molecule.from_xyz("h2o_dimer_optimised.xyz")   # after optimisation
basis = BasisSet(mol, "6-31g*")
r = run_rhf(mol, basis)
q = mulliken_charges(r, basis, mol)
print("Charges on:", q)
```

!!! note "Citing the population analysis"
    A `run_job` that writes the population dump now also emits the
    defining-paper references, Mulliken (1955), Löwdin (1950) and Mayer
    (1983), into the job's `.references` / `.bibtex` siblings
    automatically. Hirshfeld (1977) joins them when the best-effort
    Hirshfeld charges actually compute, that column needs a Becke-
    Lebedev grid plus a SAD promolecule, so it is cited only when it
    reaches the `.out` "Atomic charges" table, not on every run. Keep
    them when you publish charges or bond orders; see
    [Citing vibe-qc](../citing.md).

## Theory

The sections below derive each property from the one-particle density
matrix: the density matrix itself, then Mulliken and Löwdin population
analysis, the Mayer bond order, and the dipole moment.

### The density matrix

Every post-SCF property in this tutorial derives from the
one-particle density matrix $\mathbf{P}$ that came out of the
[HF / DFT SCF](molecular_hf.md):

$$
P_{\mu\nu} = 2 \sum_{i \in \text{occ}} C_{\mu i} \, C_{\nu i}
$$

for a closed-shell system (factor 2 from the spin sum). The AO-basis
density is the single object that encodes all observables: every
expectation value $\langle \hat{O} \rangle = \operatorname{Tr}(\mathbf{P} \mathbf{O})$
where $\mathbf{O}$ is the AO-basis matrix of the operator.

### Mulliken population analysis

Mulliken starts from the observation that

$$
N = \operatorname{Tr}(\mathbf{P} \mathbf{S})
  = \sum_{\mu\nu} P_{\mu\nu} S_{\mu\nu}
$$

is the total electron count. Each term in the double sum is
assigned half to each basis function:

$$
n_\mu^\text{Mulliken} = \sum_\nu P_{\mu\nu} S_{\mu\nu},
\qquad
q_A = Z_A - \sum_{\mu \in A} n_\mu.
$$

It's the simplest partitioning imaginable and has the problem you'd
expect: it depends on which basis function the electron is "in,"
and two different bases that describe the same physics will give
different $q_A$. **Don't compare Mulliken charges across different
basis sets, different codes, or different method families.**

### Löwdin population analysis

Löwdin orthogonalises the AO basis first with $\mathbf{S}^{-1/2}$, the
symmetric Löwdin transformation, then partitions the trace in the
orthogonal basis:

$$
\tilde{\mathbf{P}} = \mathbf{S}^{1/2} \, \mathbf{P} \, \mathbf{S}^{1/2},
\qquad
n_\mu^\text{Löwdin} = \tilde{P}_{\mu\mu}.
$$

Orthogonalizing the basis first smooths out a lot of Mulliken's
worst basis-set sensitivity, though it's still not perfectly
stable. Löwdin is generally preferred for within-system trend
analysis and for comparing similar molecules.

### Mayer bond order

Mayer's bond order generalises the one-basis-function-per-atom
Wiberg index to general contracted bases. For closed-shell, the
bond order between atoms $A$ and $B$ is

$$
B_{AB} = \sum_{\mu \in A} \sum_{\nu \in B} (\mathbf{P} \mathbf{S})_{\mu\nu} \, (\mathbf{P} \mathbf{S})_{\nu\mu}.
$$

For idealised bonding patterns this reproduces chemical intuition:
$B \approx 1$ for single bonds, $\approx 2$ for double, $\approx 3$
for triple. Unlike raw orbital analyses, Mayer's index is
**rotationally invariant** and **converges to a finite limit as the
basis improves**, a genuine measure of "number of shared electron
pairs" rather than an artefact of how the basis partitions charge.

### Dipole moment

The electric dipole moment is a first-order response of the energy
to a uniform electric field:

$$
\mu_\alpha = \int \rho(\mathbf{r}) \, r_\alpha \, \mathrm{d}^3 r
  = -\sum_{\mu\nu} P_{\mu\nu} \, \langle \chi_\mu | r_\alpha | \chi_\nu \rangle
  \;+\; \sum_A Z_A \, R_{A,\alpha}.
$$

The electronic contribution comes from a trace against the dipole-
integral matrices $\mathbf{r}_\alpha$ that libint computes
analytically; the nuclear contribution is a classical sum over
point charges at atomic positions. The result is origin-independent
for neutral molecules (the two contributions' origin dependences
cancel), vibe-qc reports at the center of mass as a convention.

## Resources

Negligible, adds <100 MB and <0.1 s on top of whatever SCF you
ran. Mulliken / Löwdin / Mayer / dipole are all
$\mathcal{O}(N_\text{bf}^3)$ matrix products against the converged
density and ride free on top of any SCF you've already paid for.

## References

- **Mulliken population analysis.** R. S. Mulliken, "Electronic
  population analysis on LCAO-MO molecular wavefunctions. I,"
  *J. Chem. Phys.* **23**, 1833 (1955); parts II-IV in
  consecutive issues.
- **Löwdin orthogonalisation.** P.-O. Löwdin, "On the non-
  orthogonality problem connected with the use of atomic wave
  functions in the theory of molecules and crystals," *J. Chem.
  Phys.* **18**, 365 (1950).
- **Hirshfeld stockholder charges.** F. L. Hirshfeld, "Bonded-atom
  fragments for describing molecular charge densities," *Theor. Chim.
  Acta* **44**, 129 (1977). The best-effort Hirshfeld column (Becke-
  Lebedev grid + SAD promolecule) is cited only when it computes, see
  the population-analysis note above.
- **Mayer bond order.** I. Mayer, "Charge, bond order and valence
  in the ab initio SCF theory," *Chem. Phys. Lett.* **97**, 270
  (1983).
- **Dipole moment in quantum chemistry.** Textbook treatment in
  any of Szabo-Ostlund (chapter 3), Helgaker-Jørgensen-Olsen
  (chapter 4), or Jensen's *Introduction to Computational
  Chemistry*. The dipole operator's matrix elements are standard
  one-electron Gaussian integrals, evaluated by libint in vibe-qc.

## Next

- Dipole moment tracks the molecular geometry; see
  [geometry optimization](geometry_optimization.md) for relaxations.
- For vibrational contributions (IR intensities = dipole derivatives)
  see [vibrational frequencies](vibrational_frequencies.md) once
  the Hessian infrastructure lands.
