# Stage 1 Design Note, DFTB0 Molecular Prototype

**Date:** 2026-05-21 (original Stage 1)  
**Status:** Shipped in v0.9.0, superseded by 58-element set

---

## 1. Overview

Stage 1 implements a non-self-consistent tight-binding (DFTB0) energy
calculation for molecules. DFTB0 is the simplest member of the DFTB
family: no charge self-consistency, a minimal valence-only basis, and
a pairwise repulsive potential. This stage establishes the module
structure, data flow, and test infrastructure that later stages
(SCC-DFTB, periodic, gradients, stress) build upon.

---

## 2. Mathematical Model

### 2.1 Basis set

A minimal valence Slater-type orbital (STO) basis, expanded in 6
Gaussian primitives per STO (STO-6G). The orbital assignments per
element are:

| Element | Valence orbitals  | n_basis |
|---------|-------------------|---------|
| H       | 1s                | 1       |
| C       | 2s, 2pₓ, 2p_y, 2p_z | 4    |
| N       | 2s, 2pₓ, 2p_y, 2p_z | 4    |
| O       | 2s, 2pₓ, 2p_y, 2p_z | 4    |

STO exponents ζ are taken from the DFTB literature (Porezag et al.,
Phys. Rev. B 51, 12947, 1995). Each STO is expanded in 6 Gaussians
using the standard STO-NG least-squares fit coefficients from
Hehre, Stewart & Pople (J. Chem. Phys. 51, 2657, 1969).

### 2.2 Hamiltonian matrix H⁰

The non-self-consistent DFTB Hamiltonian H⁰ is constructed from
two-center Slater-Koster integrals evaluated at the interatomic
distance R_AB. For the Hückel-like simplification used in Stage 1:

```
H⁰_μν = ε_l                             (μ = ν, on-site)
H⁰_μν = ½ · κ · S_μν · (h_A + h_B)    (μ ≠ ν, two-center)
```

where:
- ε_l is the on-site orbital energy for shell l on atom A
- κ is a scaling constant (≈ 1.75 for DFTB)
- S_μν is the overlap matrix element computed from the STO-6G basis
- h_A, h_B are the average on-site energies for atoms A, B

The off-diagonal form is the Wolfsberg-Helmholtz approximation,
which captures the key physics that the hopping integral scales
with the overlap and the average of the on-site energies. This is
the same approximation used in extended Hückel theory and in the
non-SCC DFTB of Porezag et al.

### 2.3 Energy expression

The generalized eigenvalue problem:

```
H⁰ C = S C ε
```

yields MO energies ε_i. The electronic energy is:

```
E_elec = Σ_i n_i ε_i
```

where n_i = 2 for doubly-occupied closed-shell orbitals.

The repulsive energy is a sum of pairwise exponentials:

```
E_rep = Σ_{A<B} a_AB · exp(−b_AB · R_AB)
```

The total DFTB0 energy:

```
E_tot = E_elec + E_rep
```

### 2.4 Gradients (Stage 2, finite-difference for Stage 1)

For Stage 1, gradients are computed numerically via central finite
differences with h = 0.001 bohr. The analytic gradient formula
(deferred to Stage 2) is:

```
dE/dR_A = Σ_i n_i · dε_i/dR_A + dE_rep/dR_A

dε_i/dR_A = C⁺_i · (dH⁰/dR_A − ε_i · dS/dR_A) · C_i
```

---

## 3. Architecture and Data Flow

### 3.1 File layout

```
cpp/include/vibeqc/semiempirical/
├── parameters.hpp      — SemiempiricalParameters (on-site energies, repulsive pairs)
├── basis.hpp           — SemiempiricalBasis (minimal STO-NG construction)
├── hamiltonian.hpp     — SemiempiricalHamiltonianBuilder (S and H⁰ assembly)
└── dftb0.hpp           — DFTB0Result, run_dftb0()

cpp/src/semiempirical/
├── parameters.cpp      — Hardcoded parameter tables for H, C, N, O
├── basis.cpp           — Build BasisSet from SemiempiricalBasis
├── hamiltonian.cpp     — Build H⁰ and S matrices
└── dftb0.cpp           — DFTB0 energy driver

python/vibeqc/semiempirical/
├── __init__.py         — Module init, public API
├── model.py            — SemiempiricalModel ABC
├── parameters.py       — Python-side parameter access
├── basis.py            — Python-side basis helpers
└── dftb0.py            — DFTB0Model concrete class, finite-diff gradients
```

### 3.2 Data flow

```
User creates molecule (C++ Molecule)
        │
        ▼
Python DFTB0Model.__init__(mol)
        │
        ├──► SemiempiricalBasis.build(mol)  ──► libint BasisSet (STO-6G)
        │
        ▼
DFTB0Model.energy()
        │
        ├──► run_dftb0(mol, params)  [C++]
        │         │
        │         ├── SemiempiricalBasis.build(mol)       ──► BasisSet
        │         ├── SemiempiricalHamiltonianBuilder
        │         │     ├── build_overlap(basis)          ──► S (n×n)
        │         │     └── build_hamiltonian(basis, S, params) ──► H⁰ (n×n)
        │         ├── Eigen::GeneralizedSelfAdjointEigenSolver  ──► ε, C
        │         ├── E_elec = sum_i n_i * ε_i
        │         ├── E_rep = sum_{A<B} a_AB * exp(-b_AB * R_AB)
        │         └── DFTB0Result{energy, eps, C, density, S, H0}
        │
        └──► returns result
```

---

## 4. Pseudocode for Key Kernels

### 4.1 Hamiltonian builder

```
function build_hamiltonian(basis, overlap_S, params):
    n = basis.nbasis()
    H0 = zeros(n, n)

    for each shell pair (μ in atom A, ν in atom B):
        if A == B:
            # on-site: diagonal block
            for each orbital in shell:
                if orbital l == orbital l':
                    H0[μ, ν] = params.on_site_energy(atom_A.Z, l)
        else:
            # two-center: Wolfsberg-Helmholtz
            h_avg = (params.average_onsite(A.Z) + params.average_onsite(B.Z)) / 2
            kappa = 1.75
            H0[μ, ν] = 0.5 * kappa * S[μ, ν] * h_avg

    return H0
```

### 4.2 DFTB0 driver

```
function run_dftb0(mol):
    basis = SemiempiricalBasis.build(mol)        # STO-6G via libint
    params = SemiempiricalParameters.default()   # H, C, N, O table

    S = compute_overlap(basis)                   # via libint
    H0 = build_hamiltonian(basis, S, params)

    # Solve generalized eigenvalue problem
    solver = GeneralizedSelfAdjointEigenSolver(H0, S)
    eps = solver.eigenvalues()                   # ascending
    C = solver.eigenvectors()

    n_occ = min(mol.n_electrons / 2, n_basis)    # closed-shell; capped —
                                                 # the valence-minimal basis
                                                 # holds at most n_basis
                                                 # doubly-occupied MOs
    E_elec = 2 * sum(eps[0:n_occ])

    E_rep = 0
    for each atom pair (A, B), A < B:
        R = distance(atom_A, atom_B)
        a, b = params.repulsive_pair(atom_A.Z, atom_B.Z)
        E_rep += a * exp(-b * R)

    E_tot = E_elec + E_rep
    D = 2 * C_occ * C_occ^T

    return DFTB0Result{energy=E_tot, mo_energies=eps, mo_coeffs=C,
                       density=D, overlap=S, hamiltonian=H0}
```

### 4.3 Finite-difference gradient

```
function gradient(mol):
    h = 0.001  # bohr
    E0 = energy(mol)

    grad = zeros(n_atoms, 3)
    for atom i in 0..n_atoms-1:
        for component c in {x, y, z}:
            mol_plus = mol with atom i shifted by +h in direction c
            mol_minus = mol with atom i shifted by -h in direction c
            E_plus = energy(mol_plus)
            E_minus = energy(mol_minus)
            grad[i, c] = -(E_plus - E_minus) / (2*h)

    return grad
```

---

## 5. Parameter Tables

### 5.1 On-site energies (Hartree)

DFTB-approximate on-site energies from the literature:

| Element | ε_s     | ε_p     |
|---------|---------|---------|
| H       | −0.2066 | -       |
| C       | −0.5040 | −0.1943 |
| N       | −0.6809 | −0.2864 |
| O       | −0.8877 | −0.3321 |

### 5.2 STO exponents (a₀⁻¹)

| Element | ζ_s   | ζ_p   |
|---------|-------|-------|
| H       | 1.24  | -     |
| C       | 1.608 | 1.608 |
| N       | 1.924 | 1.924 |
| O       | 2.246 | 2.246 |

### 5.3 Repulsive pair parameters

Simple exponential form E_rep(R) = A · exp(−B · R):

| Pair   | A (Hartree) | B (bohr⁻¹) |
|--------|-------------|------------|
| H-H    | 8.0         | 2.0        |
| H-C    | 12.0        | 1.8        |
| H-N    | 12.0        | 1.8        |
| H-O    | 12.0        | 1.8        |
| C-C    | 20.0        | 1.6        |
| C-N    | 22.0        | 1.6        |
| C-O    | 22.0        | 1.6        |
| N-N    | 24.0        | 1.6        |
| N-O    | 24.0        | 1.6        |
| O-O    | 26.0        | 1.6        |

These are approximate, order-of-magnitude parameters tuned to give
bond lengths in the right neighborhood for small molecules. They are
placeholder values; proper DFTB repulsive potentials use splines
fitted to DFT reference data.

### 5.4 STO-6G expansion coefficients

Standard STO-NG coefficients (Hehre, Stewart & Pople, 1969) for
expanding one Slater function in 6 Gaussians. For the 2p set we
use the 3p-to-s coefficient mapping (a known limitation of
STO-6G in libint, p-type primitives are set up directly via
ShellInfo, so the expansion is correct).

---

## 6. Testing Plan

### 6.1 Unit tests (`test_semiempirical_dftb0.py`)

| Test | Description |
|------|-------------|
| `test_parameters_load` | Verify on-site energies and repulsive params for H, C, N, O |
| `test_basis_construction` | Build STO-6G basis for H₂O, check n_basis |
| `test_hamiltonian_builder` | Build H⁰ and S for H₂; check symmetry, dimensions |
| `test_dftb0_h2_energy` | H₂ DFTB0 energy is negative and reasonable |
| `test_dftb0_h2o_energy` | H₂O DFTB0 energy is negative and reasonable |
| `test_dftb0_ch4_energy` | CH₄ DFTB0 energy is negative and reasonable |
| `test_reproducibility` | Same molecule → same energy across calls |
| `test_orthogonal_energy` | Energy is invariant under global rotation |

### 6.2 Gradient tests

| Test | Description |
|------|-------------|
| `test_gradient_fd_h2` | Finite-difference gradient for H₂ |
| `test_gradient_fd_h2o` | Finite-difference gradient for H₂O |
| `test_gradient_sum_zero` | Sum of forces ≈ 0 (translational invariance) |
| `test_gradient_torque_zero` | Sum of torques ≈ 0 (rotational invariance) |

### 6.3 Integration test

| Test | Description |
|------|-------------|
| `test_geometry_optimization_h2o` | BFGS optimization of H₂O converges |

---

## 7. Limitations and Deferred Items

1. **No charge self-consistency (SCC).** The Hamiltonian is
   constructed once from neutral-atom densities. SCC with Mulliken
   charge-dependent Hamiltonians is Stage 3.

2. **No analytic gradients.** Gradients are finite-difference only.
   Analytic Slater-Koster derivatives are Stage 2.

3. **Simple repulsive potential.** The shipped R⁻¹² form with automatic
   default estimator is a physics-inspired placeholder. Real DFTB uses
   DFT-fitted splines. The shipped parameters are sufficient for screening
   and preoptimization, not for production energetics.

4. **Element coverage expanded.** The Stage 1 H/C/N/O set was extended
   to 58 elements (H-Rn s/p/d blocks) before the v0.9.0 release. See
   ``docs/semiempirical_roadmap.md`` for the full element table.

5. **No periodic support.** All code is molecular-only. Periodic
   k-points and Ewald summation are Stages 4-7.

6. **No dispersion.** Repulsive potential partially compensates,
   but proper D3/D4 dispersion is Stage 9.

7. **Wolfsberg-Helmholtz approximation.** Instead of proper
   tabulated two-center DFT integrals, we use the analytically
   simpler WH form. This is standard for DFTB0 when full SK tables
   are unavailable, but limits accuracy.
