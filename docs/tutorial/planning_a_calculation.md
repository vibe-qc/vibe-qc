---
myst:
  html_meta:
    "description": "Plan a reliable vibe-qc calculation: choose the physical model, method, basis, numerical settings, memory route, convergence study, and provenance before spending compute time."
    "og:title": "Planning a reliable quantum-chemistry calculation"
---

# Planning a calculation

A quantum-chemistry input is a scientific model, not just a list of keywords.
This tutorial shows how to choose that model before running it and how to tell
whether the resulting number answers the question you intended to ask.

**You will learn how to:**

- translate a chemical question into an observable;
- choose a molecular or periodic model, method, and basis;
- separate model, basis, and numerical errors;
- estimate CPU and memory cost before submission;
- design a convergence study that changes one approximation at a time;
- retain enough provenance to reproduce the result.

## Start from the quantity, not the method

Write down the quantity you need before choosing a level of theory. Examples
include an optimized bond length, a reaction energy, an ionization energy, a
band gap, a vibrational frequency, or a charge-density difference.

Many useful quantities are differences:

$$
\Delta E = E_\text{products} - E_\text{reactants}.
$$

The cancellation in this expression is helpful only when every term uses the
same convention. Keep the method, basis, relativistic treatment, dispersion,
solvation, frozen-core choice, and numerical thresholds consistent unless the
protocol explicitly defines a correction.

For a calculated quantity $Q$, a useful planning model is

$$
\Delta Q_\text{total}
\approx
\Delta Q_\text{model}
+ \Delta Q_\text{basis}
+ \Delta Q_\text{numerical}
+ \Delta Q_\text{structure}
+ \Delta Q_\text{sampling}.
$$

These terms are not rigorously independent, but the decomposition prevents a
common mistake: tightening the SCF threshold while the dominant error comes
from the functional, basis, geometry, or k-point mesh.

## Step 1: choose the physical system

Use a `Molecule` for an isolated finite system. Use a `PeriodicSystem` when
the observable depends on an infinite crystal, surface, or wire.

Before constructing either object, record:

- geometry source and coordinate units;
- charge and spin multiplicity;
- periodic dimensionality and lattice-vector convention;
- whether symmetry, a supercell, vacuum padding, or a defect charge is part
  of the physical model;
- which atoms or lattice degrees of freedom may relax.

For molecules, vibe-qc's low-level coordinates are in bohr. File readers and
ASE interfaces can accept other units, but the conversion should be explicit
in the input or provenance. For periodic work, read
[Periodic systems](../user_guide/periodic_systems.md) and
[k-points](../user_guide/k_points.md) before treating a small unit cell as a
converged bulk model.

## Step 2: choose the electronic method

The method controls the physical approximation to electron exchange and
correlation.

| Question | Sensible starting point | What to check next |
|---|---|---|
| Closed-shell geometry or qualitative orbitals | RKS with a documented GGA or hybrid | grid, dispersion, basis convergence |
| Open-shell molecule | UHF/UKS or ROHF/ROKS | multiplicity, spin contamination, state ordering |
| Main-group reaction or binding energy | a validated hybrid or double hybrid | dispersion, basis, counterpoise, higher-level check |
| Single-reference benchmark | MP2, CCSD, or CCSD(T) | reference quality and basis convergence |
| Bond breaking or near-degenerate states | CASSCF or another multireference route | active space and state averaging |
| Crystal ground state | periodic HF or KS-DFT | k mesh, Coulomb route, basis, smearing |
| Metal | periodic KS-DFT | k mesh, electronic temperature, free-energy convention |

No row is a universal recommendation. The correct choice depends on the
elements, bonding pattern, observable, and accuracy target. The method pages
in the [user guide](../user_guide/index.md) explain the supported envelope and
known limitations for each route.

## Step 3: choose and converge the basis

In an atom-centered Gaussian calculation, every molecular orbital is expanded
in a finite basis:

$$
\phi_i(\mathbf r) = \sum_{\mu=1}^{N_\mathrm{bf}}
C_{\mu i}\,\chi_\mu(\mathbf r).
$$

Increasing $N_\mathrm{bf}$ gives the orbitals more flexibility, but also
increases cost and can create near-linear dependence in diffuse molecular or
periodic bases.

A practical molecular convergence ladder is double-zeta, triple-zeta, then
quadruple-zeta within one consistent family. For a property $Q_X$ at basis
cardinal number $X$, monitor

$$
\delta_X = Q_X - Q_{X-1}.
$$

Stop only when $|\delta_X|$ is small compared with the accuracy required for
the final conclusion. Total energies converge more slowly than many energy
differences, so converge the quantity you will actually report.

Periodic calculations need bases designed or tested for solids. Diffuse
functions that are harmless on an isolated molecule can become nearly
linearly dependent when repeated through a crystal. Start with the
[basis-set guide](../user_guide/basis_sets.md), then check the overlap
eigenvalues described in
[linear dependence](../user_guide/linear_dependence.md).

## Step 4: budget CPU time and memory

Method scaling is a warning label, not a runtime prediction. The leading
powers indicate how quickly a route becomes expensive as the orbital space
grows:

| Work | Typical leading dependence | Main practical variable |
|---|---:|---|
| Dense diagonalization | $O(N_\mathrm{bf}^3)$ | number of retained orbitals |
| Conventional HF exchange | $O(N_\mathrm{bf}^4)$ | basis functions and integral reuse |
| Direct SCF | screened shell-quartet work | locality, contractions, screening, iterations |
| Canonical MP2 | $O(N^5)$ | occupied and virtual spaces |
| CCSD | $O(N^6)$ | amplitudes and intermediates |
| CCSD(T) | $O(N^7)$ | triples workspace and occupied/virtual balance |

Conventional molecular SCF also stores a dense AO electron-repulsion tensor:

$$
M_\mathrm{ERI} = 8N_\mathrm{bf}^4\ \text{bytes}.
$$

Direct SCF avoids retaining that tensor and recomputes screened shell
quartets during Fock builds. That choice trades memory for integral work; it
does not change the Hartree-Fock model. The measured H2O2/cc-pVQZ example and
a safe crossover sweep are in
[Direct SCF or in-core integrals?](direct_scf_memory_tradeoff.md).

Use the same estimator as the high-level runner before launching a low-level
driver:

```python
from vibeqc import BasisSet, Molecule, RHFOptions, estimate_memory

mol = Molecule.from_xyz("h2o2.xyz")
basis = BasisSet(mol, "cc-pvqz")
options = RHFOptions()
estimate = estimate_memory(
    mol,
    basis,
    method="rhf",
    options=options,
)
print(estimate)
```

Compare the complete estimate with the memory allocated to the process, not
with the host's installed RAM. A scheduler, container, or cgroup may expose
only part of the machine. See [Memory budget](../user_guide/memory.md).

## Step 5: set numerical controls deliberately

Numerical thresholds should make their contribution smaller than the target
uncertainty. Tightening everything wastes time; leaving one noisy component
loose can make a smooth-looking result meaningless.

Check the controls relevant to the chosen method:

- SCF energy and commutator tolerances;
- DFT radial and angular grid;
- direct-SCF Schwarz threshold;
- density-fitting auxiliary basis and residual diagnostics;
- periodic real-space, reciprocal-space, and Ewald cutoffs;
- k-point mesh and symmetry reduction;
- smearing method, electronic temperature, and whether the reported quantity
  is energy or free energy;
- geometry force and displacement criteria.

The SCF convergence criterion can be written schematically as

$$
|E^{(n)}-E^{(n-1)}| < \tau_E,
\qquad
\|\mathbf F\mathbf P\mathbf S
  -\mathbf S\mathbf P\mathbf F\|_F < \tau_G.
$$

Both tests matter. A small energy change alone can occur during an oscillation
or near a flat region without proving that the density is self-consistent.

## Step 6: design the convergence study

A defensible study changes one approximation at a time:

1. Fix the geometry, charge, multiplicity, and method.
2. Converge numerical settings such as the DFT grid or periodic cutoffs.
3. Converge the molecular basis or periodic basis and k mesh.
4. Compare methods only after each one is internally converged.
5. Reoptimize the structure if the final observable depends strongly on
   geometry.

For a sequence $Q_1,Q_2,\ldots$, record both the value and increment:

| Level | Quantity | Change from previous | CPU time | Peak RSS |
|---|---:|---:|---:|---:|
| 1 | $Q_1$ | - | ... | ... |
| 2 | $Q_2$ | $Q_2-Q_1$ | ... | ... |
| 3 | $Q_3$ | $Q_3-Q_2$ | ... | ... |

This makes the scientific convergence and resource growth visible together.
The [basis convergence tutorial](basis_convergence.md) provides a complete
worked example.

## Step 7: inspect the result, not just the final energy

At minimum, check:

- `converged` and the final SCF residual;
- electron count, charge, multiplicity, and occupation pattern;
- orbital or overlap diagnostics appropriate to the method;
- spin contamination for unrestricted references;
- geometry and forces for optimized structures;
- method-specific diagnostics such as active-space occupations, pair-domain
  coverage, auxiliary-fit residuals, or smearing entropy;
- whether warnings or experimental-route gates appeared in the log.

A plausible energy with the wrong state, wrong units, or unconverged sampling
is not a successful calculation.

## Step 8: retain provenance

Use `run_job` or `run_periodic_job` for production work so the normal output
family is written together. Keep at least:

- the input script or settled QVF job container;
- `.out` for the human-readable calculation record;
- `.system` for machine-readable versions and settings;
- `.qvf` for the typed, checksummed result archive;
- `.references` and `.bibtex` for the citations selected by the actual route;
- geometry, trajectory, or checkpoint files needed to reproduce later steps.

Name the directory after the scientific question rather than the program
invocation. A useful structure is:

```text
project/
  inputs/
  runs/
    basis-dz/
    basis-tz/
    basis-qz/
  analysis/
  figures/
```

Do not run production inputs inside the source checkout. The
[examples guide](https://github.com/vibe-qc/vibe-qc/blob/main/examples/README.md)
shows how to copy a canonical input into a separate run directory.

## A compact planning worksheet

Before submitting, be able to fill every row:

| Decision | Your value |
|---|---|
| Observable and target uncertainty | |
| Geometry source and units | |
| Charge and multiplicity | |
| Molecular or periodic model | |
| Method and why it is appropriate | |
| Basis and convergence ladder | |
| Numerical grid, cutoffs, or k mesh | |
| SCF route and thresholds | |
| Estimated memory and allocated memory | |
| Expected runtime and thread count | |
| Diagnostics that define success | |
| Comparison or validation reference | |
| Files retained for provenance | |

## Next

- Build the mean-field foundation in
  [Molecular Hartree-Fock](molecular_hf.md).
- Compare density-functional choices in
  [Molecular DFT](molecular_dft.md) and
  [Functional comparison](functional_comparison.md).
- Plan solids with [Periodic methods compared](periodic_methods_compared.md).
- Learn the practical run-directory and naming conventions in
  [Good practices](../good_practices.md).
