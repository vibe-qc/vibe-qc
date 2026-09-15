# Molecular Hartree-Fock

Compute the restricted-Hartree-Fock energy of the water molecule in
the 6-31G* basis. This page is also the right place to learn *what*
vibe-qc actually solves under the hood, the theory section below
spells out the Roothaan-Hall equations, the Fock matrix, and Pulay's
DIIS acceleration.

```python
from vibeqc import Atom, Molecule, BasisSet, run_rhf

# Water, experimental geometry in bohr.
mol = Molecule([
    Atom(8, [0.0,  0.0,  0.0]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])
basis = BasisSet(mol, "6-31g*")
result = run_rhf(mol, basis)

print(f"E_RHF       = {result.energy:.10f}")
print(f"converged   = {result.converged}")
print(f"iterations  = {result.n_iter}")
print(f"MO energies = {result.mo_energies}")
```

Output:

```
E_RHF       = -76.0107424867
converged   = True
iterations  = 8
MO energies = [-20.55  -1.34  -0.71  -0.56  -0.50   0.21   0.31   ...]
```

## What just happened

1. **Building the basis.** `BasisSet(mol, "6-31g*")` looks up the
   6-31G\* contraction in libint's built-in library, places its
   shells on each atomic center, and records the resulting list of
   atomic orbitals.
2. **SCF with DIIS.** The driver diagonalises H_core for an initial
   guess, builds the Fock matrix from the density, then iterates using
   Pulay's DIIS extrapolation. Convergence tolerances default to
   `conv_tol_energy = 1e-8 Ha` and `conv_tol_grad = 1e-6`.
3. **Output.** The `RHFResult` carries the total energy, converged MO
   coefficients, orbital energies, density matrix, and the full SCF
   trace for post-analysis.

## Tuning convergence

Most of the time the defaults are fine. For tight benchmarks you may
want:

```python
from vibeqc import RHFOptions

opts = RHFOptions()
opts.conv_tol_energy = 1e-12
opts.conv_tol_grad = 1e-10
opts.use_diis = True
opts.initial_guess = vq.InitialGuess.SAD   # superposition of atomic densities

result = run_rhf(mol, basis, opts)
```

## Computing gradients

For geometry optimization or force evaluation, vibe-qc exposes analytic
HF gradients:

```python
from vibeqc import compute_gradient

grad = compute_gradient(mol, basis, result)   # (n_atoms, 3), Hartree / bohr
```

Gradients are analytic throughout; accuracy tracks the SCF convergence
(better than 1e-8 Ha/bohr at default tolerances).

## Theory

The sections below trace what the driver above actually solves: the
single-determinant ansatz, the Roothaan-Hall equations it leads to, the
self-consistent iteration that solves them, the DIIS acceleration that
makes it converge fast, and the accuracy ceiling that comes with a
single determinant.

### The variational ansatz

Hartree-Fock approximates the electronic ground-state wavefunction
as a single Slater determinant built from $N_\text{occ}$ orthonormal
spin-orbitals $\phi_i(\mathbf{r}, \sigma)$. The antisymmetric product

$$
\Psi_\text{HF}(\mathbf{r}_1 \sigma_1, \dots, \mathbf{r}_N \sigma_N) =
  \frac{1}{\sqrt{N!}}
  \det\!\bigl[\phi_i(\mathbf{r}_j, \sigma_j)\bigr]
$$

encodes Pauli antisymmetry exactly and returns the lowest energy any
single-determinant wavefunction can give, but it misses the
electron-electron *correlation* energy, the ~1% of the total that
methods like MP2, coupled cluster, and DFT try to recover.

### The Roothaan-Hall equations

Expand each orbital in a finite set of $N_b$ atomic-orbital basis
functions $\{\chi_\mu\}$:

$$
\phi_i(\mathbf{r}) = \sum_\mu C_{\mu i} \, \chi_\mu(\mathbf{r}).
$$

The variational minimisation of the HF energy with respect to the
$C_{\mu i}$ coefficients (subject to orthonormality $\mathbf{C}^\dagger
\mathbf{S} \mathbf{C} = \mathbf{I}$) yields the **Roothaan-Hall
equations**, a generalised eigenvalue problem in the AO basis:

$$
\mathbf{F} \, \mathbf{C} = \mathbf{S} \, \mathbf{C} \, \boldsymbol{\varepsilon},
$$

where $\mathbf{S}_{\mu\nu} = \braket{\chi_\mu | \chi_\nu}$ is the
overlap matrix, $\boldsymbol{\varepsilon}$ is the diagonal matrix of
orbital energies, and $\mathbf{F}$ is the **Fock matrix**:

$$
F_{\mu\nu} = h_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma}
  \Bigl[
    (\mu\nu \,|\, \lambda\sigma)
    - \tfrac{1}{2} \, (\mu\sigma \,|\, \lambda\nu)
  \Bigr].
$$

Here $h_{\mu\nu}$ is the one-electron core Hamiltonian (kinetic +
nuclear-attraction), $P_{\lambda\sigma} = 2 \sum_{i \in \text{occ}}
C_{\lambda i} C_{\sigma i}$ is the one-particle density matrix in the
closed-shell (RHF) case, and $(\mu\nu|\lambda\sigma)$ are the
two-electron repulsion integrals in chemists' notation. The two
terms inside the brackets are the classical **Coulomb** $J$ and
quantum-mechanical **exchange** $K$ contributions.

### Self-consistency via iteration

$\mathbf{F}$ depends on $\mathbf{P}$, which depends on $\mathbf{C}$,
which is the eigenvector we're trying to find, so the equations are
non-linear. The standard fix is the **self-consistent-field (SCF)**
iteration:

1. Guess an initial density $\mathbf{P}^{(0)}$ (vibe-qc defaults to
   the superposition-of-atomic-densities (SAD) guess, with an
   $H_\text{core}$ fallback).
2. Build $\mathbf{F}^{(n)}$ from $\mathbf{P}^{(n)}$.
3. Solve the generalised eigenvalue problem for
   $\mathbf{C}^{(n+1)}$ and $\boldsymbol{\varepsilon}^{(n+1)}$.
4. Form $\mathbf{P}^{(n+1)}$ from the occupied columns of
   $\mathbf{C}^{(n+1)}$.
5. Stop when $|E^{(n+1)} - E^{(n)}| < \texttt{conv\_tol\_energy}$
   *and* $\|[\mathbf{F}, \mathbf{P}\mathbf{S}]\|_\text{F} <
   \texttt{conv\_tol\_grad}$. The commutator vanishes at
   convergence: $[\mathbf{F}, \mathbf{P}\mathbf{S}] = 0$ iff
   $\mathbf{F}$ and the density share an eigenbasis.

### DIIS acceleration

Fixed-point iteration of the above recipe typically oscillates and
converges slowly. Pulay's **Direct Inversion in the Iterative
Subspace (DIIS)** extrapolates the Fock matrix instead of using
$\mathbf{F}^{(n)}$ directly:

$$
\mathbf{F}_\text{DIIS} = \sum_{i} c_i \, \mathbf{F}^{(i)},
\qquad
\text{minimize } \Bigl\| \sum_i c_i \, \mathbf{e}^{(i)} \Bigr\|^2
\text{ subject to } \sum_i c_i = 1,
$$

where $\mathbf{e}^{(i)} = [\mathbf{F}^{(i)}, \mathbf{P}^{(i)}
\mathbf{S}]$ is the commutator error vector at iteration $i$. Near
the solution DIIS converges quadratically in practice; at default
settings vibe-qc takes 8-12 iterations on most closed-shell
molecules, where naïve mixing would take dozens.

### Where HF helps, where it hurts

HF is exact for one-electron systems (H, He⁺, Li²⁺, …) and is the
starting point for the whole correlated-wavefunction hierarchy (MP2,
CCSD, CCSD(T)). It has a hard ceiling on accuracy because the single
determinant cannot represent the instantaneous electron-electron
correlation, *dynamic* correlation is missing entirely. For
benchmark-quality numbers you graduate to Kohn-Sham DFT (see
[Molecular DFT (RKS)](molecular_dft.md)), MP2, or coupled cluster.

## Resources

~70 MB peak RAM, <0.1 s on one core (Apple M2 baseline) for the
H₂O / 6-31G\* example. The Fock build is $\mathcal{O}(N_\text{bf}^4)$
in basis size, cc-pVTZ on the same H₂O is ~3 s, ~250 MB.

## References

Foundation papers and a canonical textbook.

- **Roothaan-Hall equations.** C. C. J. Roothaan, "New developments
  in molecular orbital theory," *Rev. Mod. Phys.* **23**, 69 (1951);
  G. G. Hall, "The molecular orbital theory of chemical valency.
  VIII," *Proc. R. Soc. Lond. A* **205**, 541 (1951).
- **Slater determinant.** J. C. Slater, "The theory of complex
  spectra," *Phys. Rev.* **34**, 1293 (1929).
- **Self-consistent field method.** D. R. Hartree, "The wave
  mechanics of an atom with a non-Coulomb central field," *Proc.
  Cambridge Philos. Soc.* **24**, 89 (1928); V. A. Fock, "Näherungs-
  methode zur Lösung des quantenmechanischen Mehrkörperproblems,"
  *Z. Phys.* **61**, 126 (1930).
- **DIIS extrapolation.** P. Pulay, "Convergence acceleration of
  iterative sequences. The case of SCF iteration," *Chem. Phys.
  Lett.* **73**, 393 (1980); P. Pulay, "Improved SCF convergence
  acceleration," *J. Comput. Chem.* **3**, 556 (1982).
- **Textbook.** A. Szabo and N. S. Ostlund, *Modern Quantum
  Chemistry: Introduction to Advanced Electronic Structure Theory*,
  Dover (1996). Chapter 3 is the clearest walk through HF; chapter
  5 covers MP2.

## Next

* Add exchange-correlation with [Kohn-Sham DFT](molecular_dft.md).
* Open-shell systems use the [UHF driver](open_shell.md).
* If your system is periodic, jump to the
  [periodic HF tutorial](periodic_hf.md).
