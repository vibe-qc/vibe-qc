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
land in the `.out` when those analyses succeed. Run the first block in the
installed engine environment from a run directory outside the checkout.
It is self-contained and uses an illustrative fixed geometry, not an
optimized structure. The raw-array example below is an alternative workflow;
you do not need to run both.

```python
from vibeqc import Atom, Molecule, run_job

mol = Molecule([
    Atom(8, [ 0.0,  0.00,  0.00]),
    Atom(1, [ 0.0,  1.43, -0.98]),
    Atom(1, [ 0.0, -1.43, -0.98]),
])
result = run_job(mol, basis="6-31g*", method="rhf", output="water")
if not result.converged:
    raise RuntimeError("Water RHF did not converge; inspect water.out")
```

Positions passed to `Atom` are in bohr. To read your own XYZ geometry in
angstrom, use `from_xyz` as described in [Molecules](../user_guide/molecules.md).
Keep the geometry, basis and method fixed when comparing charge models.

Open `water.out` after the SCF trace and orbital-energy table. The following
is an illustrative output excerpt from an earlier build, not a reference
value or a required match for your installation:

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
- **Dipole moment** defaults to the center of mass as its reported origin.
  A neutral molecule's dipole is origin-independent; for ions pass
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
if not result.converged:
    raise RuntimeError("Water RHF did not converge; inspect result.scf_trace")

q_mull = mulliken_charges(result, basis, mol)      # (n_atoms,) np.ndarray
q_lowd = loewdin_charges(result, basis, mol)       # same shape
np.testing.assert_allclose(q_mull.sum(), mol.charge, atol=1e-6, rtol=0)
np.testing.assert_allclose(q_lowd.sum(), mol.charge, atol=1e-6, rtol=0)

bonds = mayer_bond_orders(result, basis, mol)      # (n_atoms, n_atoms)
for i, j, order in prominent_bonds(bonds, mol, threshold=0.10):
    print(f"Atoms {i + 1}-{j + 1}: Mayer bond order {order:.4f}")

mu = dipole_moment(result, basis, mol)
print(f"|μ| = {mu.total_debye:.3f} D; origin = {mu.origin} bohr")
```

Array rows follow `mol.atoms` order. `prominent_bonds` returns zero-based
atom indices, so the loop adds one for display. Its threshold only filters
the printed list; it does not modify the full bond-order matrix. These
charge-sum checks test electron bookkeeping, not chemical accuracy. Use
Hirshfeld's grid-integration diagnostics separately; numerical quadrature
need not satisfy the same tolerance as an AO population trace.

Always use the molecule and basis that produced the density. For ECP systems,
the charge and dipole helpers also need effective nuclear charges; follow
[the ECP property recipe](../user_guide/properties.md). These helpers read
the supplied density. Passing an MP2 job result does not by itself request
properties of a correlated MP2 density.

## Basis-set dependence: a quick sweep

Mulliken charges drift significantly between basis sets, run the same
water molecule at three levels and print the oxygen charge:

```python
# doc-audit: skip - continues with imports from the raw-array example
mol = Molecule([
    Atom(8, [ 0.0,  0.00,  0.00]),
    Atom(1, [ 0.0,  1.43, -0.98]),
    Atom(1, [ 0.0, -1.43, -0.98]),
])

for basis_name in ("sto-3g", "6-31g*", "def2-tzvp"):
    basis = BasisSet(mol, basis_name)
    r = run_rhf(mol, basis)
    if not r.converged:
        raise RuntimeError(f"Water RHF/{basis_name} did not converge")
    q = mulliken_charges(r, basis, mol)
    print(f"{basis_name:12s}  q_O = {q[0]:+.3f}  μ (D) = "
          f"{dipole_moment(r, basis, mol).total_debye:.3f}")
```

Read the printed values as a basis-dependence study at one geometry.
Changing the basis changes the density and its partition among atoms. Keep
charge definitions and numerical settings consistent when comparing runs;
there is no universal charge-model ordering or guaranteed improvement in a
particular dipole component. To study method dependence, repeat with a
specified DFT functional at the same geometry and basis, and check its grid
convergence separately.

## Following the H-bond in a water dimer

The [geometry optimization tutorial](geometry_optimization.md) explains
how to run and check the water dimer optimization. A successful run of
[`examples/molecular/input-h2o-dimer-opt.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dimer-opt.py)
writes `output-h2o-dimer-opt.xyz` next to that script, using the same output
stem as its `.out` and `.molden` files. Check the optimizer's force target
and final SCF convergence before treating this as an optimized structure;
the example deliberately uses a loose demonstration threshold.

The final `.out` already contains population analysis. To repeat the
analysis separately, copy the final XYZ into your run directory and use:

```python
from vibeqc import from_xyz, BasisSet, run_rhf, mulliken_charges

mol = from_xyz("output-h2o-dimer-opt.xyz", charge=0, multiplicity=1)
basis = BasisSet(mol, "6-31g*")
r = run_rhf(mol, basis)
if not r.converged:
    raise RuntimeError("Dimer RHF did not converge; inspect r.scf_trace")
q = mulliken_charges(r, basis, mol)
for i, (atom, charge) in enumerate(zip(mol.atoms, q), start=1):
    print(f"Atom {i}, Z={atom.Z}: {charge:+.6f} e")
```

Identify the donor and acceptor from the coordinates rather than guessing
from the charge signs. To isolate the effect of complex formation, compare
against monomers with a documented geometry, basis and method; changing
several of them together makes a charge difference ambiguous.

```{admonition} Citing the population analysis
:class: note

A `run_job` that writes the population dump now also emits the
defining-paper references, Mulliken (1955), Löwdin (1950) and Mayer
(1983), into the job's `.references` / `.bibtex` siblings
automatically. Hirshfeld (1977) joins them when the best-effort
Hirshfeld charges actually compute, that column needs a Becke-
Lebedev grid plus a SAD promolecule, so it is cited only when it
reaches the `.out` "Atomic charges" table, not on every run. Keep
them when you publish charges or bond orders; see
[Citing vibe-qc](../citing.md).
```

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
density determines one-electron expectation values:
$\langle \hat{O} \rangle = \operatorname{Tr}(\mathbf{P} \mathbf{O})$
where $\mathbf{O}$ is the AO-basis matrix of the operator.

### Mulliken population analysis

Mulliken ({vibeqc-cite}`mulliken_1955`) starts from the observation
that

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

Löwdin ({vibeqc-cite}`lowdin_1950`) orthogonalises the AO basis first
with $\mathbf{S}^{-1/2}$, the symmetric Löwdin transformation, then partitions the trace in the
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

Mayer's bond order ({vibeqc-cite}`mayer_bond_order_1983`) generalises
the one-basis-function-per-atom Wiberg index
({vibeqc-cite}`wiberg_bond_index_1968`) to general contracted bases. For closed-shell, the
bond order between atoms $A$ and $B$ is

$$
B_{AB} = \sum_{\mu \in A} \sum_{\nu \in B} (\mathbf{P} \mathbf{S})_{\mu\nu} \, (\mathbf{P} \mathbf{S})_{\nu\mu}.
$$

For idealised bonding patterns this reproduces chemical intuition:
$B \approx 1$ for single bonds, $\approx 2$ for double, $\approx 3$
for triple. Unlike raw orbital analyses, Mayer's index is
**rotationally invariant**. Like Mulliken charges, though, it rests on
assigning basis functions to atoms, so it **depends on the basis** and
can become erratic with diffuse functions; compare values within one
basis family.

### Dipole moment

The electric dipole moment is a first-order response of the energy
to a uniform electric field:

$$
\mu_\alpha = -\int \rho(\mathbf{r}) \, r_\alpha \, \mathrm{d}^3 r
  + \sum_A Z_A \, R_{A,\alpha}
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

These analyses reuse the converged density, but their costs differ:
Mulliken uses an overlap contraction, Löwdin needs a matrix decomposition,
and dipoles require one-electron moment integrals. Hirshfeld additionally
builds a grid and promolecule. Treat these as additional analysis work when
planning larger jobs, rather than assuming a fixed memory or time overhead.

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
  see [vibrational frequencies](vibrational_frequencies.md).
