---
myst:
  html_meta:
    "description": "Introduction to the vibe-qc tutorial course: what quantum chemistry computes, the electronic Schrodinger equation under the Born-Oppenheimer approximation, the molecular and periodic tracks, the shape of a calculation, and how to work through the tutorials."
    "og:title": "vibe-qc tutorials - introduction and course overview"
---

# Introduction: a course in computational quantum chemistry

These tutorials are written to be read as a **course**, not just a recipe
book. Every page is a complete, runnable example, but it also explains the
method it uses: where the equations come from, what the numbers mean, and
when the approach is the right one. If you only want to copy a code block
and get a number, you can; if you want to understand *why* that number is
what it is, the surrounding text is there for you.

You do not need to read the pages in order. But if you are new to the
field, the order in the sidebar is a deliberate learning path: each page
assumes the ones above it and builds on them.

## What quantum chemistry computes

Almost everything in this course comes back to one equation. For a
molecule with electrons at positions $\mathbf{r}_i$ and nuclei (charge
$Z_A$) clamped at positions $\mathbf{R}_A$, the **electronic
Hamiltonian** in atomic units (energies in Hartree, lengths in Bohr) is

$$
\hat{H}_\text{el}
  = -\tfrac{1}{2}\sum_i \nabla_i^2
    \;-\; \sum_{i,A}\frac{Z_A}{|\mathbf{r}_i-\mathbf{R}_A|}
    \;+\; \sum_{i<j}\frac{1}{|\mathbf{r}_i-\mathbf{r}_j|}.
$$

The three terms are the electronic kinetic energy, the
electron-nucleus attraction, and the electron-electron repulsion. Clamping
the nuclei is the **Born-Oppenheimer approximation**: nuclei are thousands
of times heavier than electrons, so the electrons relax essentially
instantaneously to any nuclear arrangement. The constant nuclear-repulsion
energy $\sum_{A<B} Z_A Z_B/|\mathbf{R}_A-\mathbf{R}_B|$ is added back at the
end.

We want the ground-state energy $E_0$ and wavefunction $\Psi$ that satisfy
$\hat{H}_\text{el}\Psi = E_0\Psi$. The **variational principle** gives us a
handle on it: for *any* normalised trial $\Psi$,

$$
E[\Psi] = \langle\Psi|\hat{H}_\text{el}|\Psi\rangle \;\ge\; E_0 .
$$

So minimising the energy over a family of trial wavefunctions can only
approach $E_0$ from above, never undershoot it. The methods in this course
are, in large part, different choices of that family and different ways of
doing the minimisation:

- **Hartree-Fock (HF)** takes $\Psi$ to be a single antisymmetrised
  product of one-electron orbitals (a Slater determinant) and minimises
  the energy over those orbitals. The electron-electron repulsion is felt
  only on average (mean field). See [molecular_hf](molecular_hf.md).
- **Density-functional theory (DFT / KS-DFT)** replaces the wavefunction
  with the electron density and an approximate exchange-correlation
  functional, recovering much of the missing physics at HF-like cost. See
  [molecular_dft](molecular_dft.md).
- **Correlated wavefunction methods** (MP2, coupled cluster, configuration
  interaction, multireference) systematically add the electron-electron
  correlation that mean-field methods miss, at higher cost. See
  [dlpno_local_correlation](dlpno_local_correlation.md) and
  [non_hf_solvers](non_hf_solvers.md).

Each of these gets its own tutorial that derives the working equations and
shows them running.

## Two tracks: molecular and periodic

vibe-qc has two families of systems, and the sidebar is split to match.

**Molecular** calculations treat a finite collection of atoms in vacuum.
The high-level entry point is `run_job`, and the lower-level drivers are
`run_rhf`, `run_rks`, `run_uhf`, and so on. Start at
[molecular_hf](molecular_hf.md).

**Periodic** calculations treat an infinite crystal, surface slab, or wire
through **Born-von-Karman periodic boundary conditions**: a unit cell is
tiled to fill space, and the orbitals carry a Bloch phase indexed by a
crystal momentum $\mathbf{k}$. The entry point is `run_periodic_job`.
Periodic boundary conditions are vibe-qc's headline focus, and **surface
chemistry** (slab energies, adsorption, reaction paths) is the flagship
workflow. Start at [periodic_hf](periodic_hf.md), and read
[kpoints_brillouin_bloch](kpoints_brillouin_bloch.md) for the physics of
$\mathbf{k}$ that the periodic track rests on.

The periodic track also has a choice the molecular track does not: **which
Coulomb-build route to use** (GDF, BIPOLE, or the plane-wave GPW/GAPW
family). That choice has its own comparison tutorial,
[periodic_methods_compared](periodic_methods_compared.md).

vibe-qc's **signature** periodic approach, though, is the
[cyclic cluster model](cyclic_cluster_model.md) (CCM): a real-space
alternative to k-point sampling and the project's original motivating
feature. CCM treats a crystal as a single finite cyclic cluster, which makes
it especially natural for defects and surfaces. It runs today at the
semi-empirical level, with ab-initio CCM (Hartree-Fock through coupled
cluster) the flagship roadmap.

## The shape of a calculation

Almost every tutorial follows the same three-step arc, so it is worth
seeing it once in the abstract:

1. **Define the system**, the atoms and their positions (a `Molecule`),
   or a `PeriodicSystem` with its lattice.
2. **Choose a basis and a method**, the set of functions the orbitals are
   expanded in, and how the energy is computed (HF, a DFT functional, a
   correlated method).
3. **Run the SCF and read the result**, the self-consistent field
   iteration solves the mean-field equations, and the returned object
   carries the energy, orbitals, and any properties you asked for.

The three lines hide several scientific decisions: what observable is being
predicted, how the basis and numerical grid will be converged, whether the
memory route fits the allocation, and which diagnostics define success.
[Planning a calculation](planning_a_calculation.md) turns those decisions
into a reusable worksheet before the method-specific lessons begin.

The "self-consistent" part is the heart of it: the equation for the
orbitals depends on the orbitals themselves (each electron moves in the
field of all the others), so it is solved by iteration to a fixed point.
How that iteration is accelerated and stabilised is itself a topic, covered
in [initial_guess_walkthrough](initial_guess_walkthrough.md) and
[ediis_diis_hybrid](ediis_diis_hybrid.md).

## How to read these tutorials

- **Every code block runs.** You can paste it into a Python interpreter on
  a working install.
- **The example files are downloadable.** Where a tutorial is built on a
  script, that script lives under `examples/`. Curated reference bundles
  with full logs, QVF archives where available, and vibe-view screenshots
  are linked from [example_outputs](../example_outputs.md). Run the input
  as-is, then change one thing at a time. The same catalog also carries the
  chi-CCM-B periodic QVF validation bundle with two inputs, sanitized
  manifests, validated QVF archives, and headless vibe-view captures.
- **Numbers are cited.** Where a tutorial compares against a literature or
  cross-code value, the source is named. vibe-qc also writes the citations
  for the methods you actually used into a `.bibtex` sidecar; see
  [auto_citations](auto_citations.md).

## Before you start

You need a working vibe-qc install. If you do not have one yet, follow the
[installation guide](../installation.md), and for the five-minute version
see the [quickstart](../quickstart.md). The molecular tutorials use the
built-in basis sets that ship with vibe-qc (STO-3G, 6-31G\*, cc-pVDZ); the
periodic tutorials use the [pob-\* basis sets](../user_guide/basis_sets.md)
designed to avoid linear dependence in crystals.

Throughout, energies are in **Hartree** and lengths in **Bohr** unless a
page says otherwise, the atomic units that make the Hamiltonian above
look as clean as it does.

When you are ready, open [molecular_hf](molecular_hf.md) for your first
calculation, or jump to [periodic_hf](periodic_hf.md) if you came here for
solids.
