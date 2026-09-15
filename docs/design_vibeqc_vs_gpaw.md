# vibe-qc GPW/GAPW vs GPAW Program -- Architectural Comparison

## 1. Orbital basis

| | vibe-qc GPW/GAPW | GPAW Program |
|---|---|---|
| **Basis type** | **Gaussian** (atom-centered, contracted GTOs via libint) | **Real-space grid** (uniform 3D finite-difference grid) + PAW projectors |
| **AO evaluation** | libint C++17 `evaluate_ao()` with 27-image periodic-image folding per grid point | Wavefunctions represented directly on the real-space grid; no AO basis |
| **Consequence** | Sparse analytic integrals (overlap, kinetic, nuclear attraction, ERI) are exact for the Gaussian basis; only the Coulomb J is done on the FFT grid | ALL operators (kinetic, potential, Hartree, XC) are finite-difference stencils on the same grid -- uniform treatment but finite-difference error |

**Fundamentally different philosophies**: vibe-qc is a **Gaussian-orbital code** borrowing the FFT-Poisson trick from GPW for the long-range Coulomb; GPAW is a **pure grid code** with PAW for all-electron accuracy.

## 2. Coulomb / Hartree solver

| | vibe-qc GPW/GAPW | GPAW Program |
|---|---|---|
| **Grid density** | Gaussian density ρ(r) collocated onto uniform FFT grid via AO evaluation at each grid point: `ρ(r_g) = Σ_μν D_μν χ_μ(r_g) χ_ν(r_g)` | Electron density computed directly on the FD grid from wavefunctions ψ_n(r) |
| **Poisson solve** | FFT → `4π/G²` → inverse FFT (same algorithm). C++ `solve_poisson_coulomb` kernel. | FFT → `4π/G²` → inverse FFT. Written in C with `_gpaw` C extension. |
| **AO projection** | Grid potential V_H(r_g) projected back to AO J matrix: `J_μν = Σ_g χ_μ(r_g) χ_ν(r_g) V_H(r_g) dV` | N/A -- no AO basis. Potential applied directly to wavefunctions on the grid. |
| **Collocation cache** | `GpwCollocationCache` stores χ(r_g) table built once per SCF, reused across iterations (95%+ of runtime saved) | No equivalent -- wavefunctions ARE the grid representation |

## 3. All-electron treatment

| | vibe-qc GAPW | GPAW Program |
|---|---|---|
| **Method** | **GAPW** (Gaussian-based): AO density on per-atom radial×Lebedev grids, augmented with atomic corrections for the hard (core+nodal) density | **PAW** (Blöchl 1994): Frozen atomic partial waves + projector functions. PAW datasets pre-computed for each element. |
| **Atomic data** | No frozen datasets -- uses the *same Gaussian basis* to model both hard (ρ_a) and soft (ρ̃_a) densities on atomic grids. Soft basis built by removing tight primitives (exponent > α_cut). | **Pre-computed PAW datasets** (setups) containing partial waves φ_i, pseudo partial waves φ̃_i, projector functions p̃_i, and compensation charges. Downloaded from gpaw-data. |
| **Correction form** | `E_H = E_H[ρ̃+ρ₀] + Σ_a (E_H[ρ_a] − E_H[ρ̃_a+ρ₀,a])` -- Hartree correction on atomic sphere | `E = E_pseudo + Σ_a Σ_ij Δρ_ij D_ij` -- PAW correction via atomic density matrix in the partial-wave basis |
| **Frozen core** | **All-electron** by construction (no pseudopotential). The Gaussian basis already describes the core. | Uses **frozen-core PAW** by default -- core states are pre-computed atomic orbitals, frozen during SCF |

## 4. SCF convergence

| | vibe-qc GPW/GAPW | GPAW Program |
|---|---|---|
| **Eigensolver** | DIIS (Pulay) + EDIIS/ADIIS on the Fock matrix. C++ SCF host (`periodic_gapw_cpp_host`). | Davidson iterative diagonalisation by default; also supports direct minimisation (etdm-fdpw), CG, and LCAO mode. |
| **Mixing** | Anderson/Pulay mixing of the density matrix. `auto_optimize_truncation` for convergence. | Pulay density mixing by default (`Mixer`); also `no-mixing` for direct-min. |
| **Smearing** | Fermi-Dirac, Gaussian, Marzari-Vanderbilt. `SmearingOptions` class. | Fermi-Dirac, Methfessel-Paxton, Marzari-Vanderbilt, cold smearing. |
| **Open-shell** | UHF/UKS/ROHF/ROKS available (post-M3). `GAPWExperimentalWarning` gates production use. | Spin-polarized DFT and HF available throughout. Hund's rule support for free atoms. |

## 5. GPU acceleration

| | vibe-qc | GPAW |
|---|---|---|
| **GPU** | No GPU support (CPU-only C++17 with Eigen/FFTW3) | **Yes** -- NVIDIA GPU support via CuPy (Python GPU arrays). Most operations run on GPU if available. |

## 6. Exchange (exact HF exchange)

| | vibe-qc GPW/GAPW | GPAW Program |
|---|---|---|
| **Hybrid functionals** | **Yes** -- libint analytic 4c-ERIs for HF exchange at Γ-point. BIPOLE route for multi-k. | **Yes** (PW mode) -- but expensive: O(N³) scaling. Usually combined with LCAO mode for hybrids. |
| **Range-separated** | Experimental (`periodic_gapw_range_sep`, behind `GAPWExperimentalWarning`). | Supported through libxc range-separated functionals. |
| **COSX (RIJCOSX)** | Γ-point COSX-K experimental; BIPOLE for multi-k. | No direct COSX equivalent; uses LCAO or pure grid modes. |

## 7. Functional coverage

| | vibe-qc GPW/GAPW | GPAW Program |
|---|---|---|
| **LDA/GGA** | ✓ Full libxc (500+ functionals) | ✓ Full libxc |
| **meta-GGA** | ✓ r²SCAN and others (M3c). Requires τ on the real-space grid. | ✓ Full libxc including r²SCAN. Requires τ. |
| **Hybrids** | ✓ (analytic ERI at Γ; BIPOLE at multi-k) | ✓ (but expensive in PW mode) |

## 8. Production readiness

| | vibe-qc GPW/GAPW | GPAW Program |
|---|---|---|
| **GPW** | Production for closed-shell Γ-point. Multi-k pure-DFT available. Analytic forces wired. | **Production since 2005.** Thousands of publications. |
| **GAPW** | **Experimental** (`GAPWExperimentalWarning`). Correct for molecular-limit systems; tight-cell ionic crystals have known issues. | **PAW is default and production.** Mature PAW datasets. |
| **Tests** | Growing test suite. GAPW parity against CP2K targeted. | Extensive test suite. Validated against VASP, ABINIT, QE. |

## 9. Method family overview

vibe-qc ships several periodic SCF routes covering different accuracy/cost trade-offs:

| Route | Basis | Coulomb | Exchange | All-electron | Status |
|---|---|---|---|---|---|
| **BIPOLE** | Gaussian | exact Ewald J split (direct erfc SR + reciprocal AO-pair-FT LR) | Analytic 4c-ERI | Yes (all-electron GTOs; ECPs unsupported) | Production exact route (RHF/UHF/RKS/UKS); quartet multipoles fail closed |
| **GDF** | Gaussian + density-fitting auxiliary basis | Ewald / FFT | Analytic 3c-ERI (DF-approximate) | Yes | Production (closed+open shell, multi-k) |
| **GPW** | Gaussian → real-space grid collocation | FFT-Poisson | Analytic 4c-ERI | No (smooth density only -- PP-like) | Production (Γ-point, closed-shell) |
| **GAPW** | Gaussian + per-atom radial grids | FFT-Poisson + atomic-sphere correction | Analytic 4c-ERI | **Yes** (all-electron via augmentation) | Experimental |
| **CCM** | Gaussian (INDO/MNDO parameterised) | Ewald (Madelung) | INDO Fock terms | No (valence-only, semiempirical) | Production (MSINDO) |

## 10. Summary

| Dimension | vibe-qc | GPAW |
|---|---|---|
| **Basis** | Gaussian (atom-centered) | Real-space grid (FD) |
| **Core treatment** | All-electron for BIPOLE/GDF and augmented GAPW; smooth-grid or valence models where selected | PAW (frozen-core, pre-computed datasets) |
| **Hartree** | FFT-Poisson on collocated density (GPW/GAPW), exact Ewald split (BIPOLE), or DF (GDF) | FFT-Poisson on grid density |
| **Exchange** | Analytic 4c-ERI (exact for Gaussians) or DF-approximate (GDF) | FD stencil on grid (or LCAO mode) |
| **Overall approach** | Gaussian quantum chemistry + GPW acceleration for Coulomb | Pure real-space DFT from the ground up |
| **Maturity** | GPW: production · GAPW: experimental · BIPOLE: production · GDF: production | Production (20 years) |

**TL;DR:** vibe-qc's GPW is a **Gaussian code accelerated by FFT**, converging the traditional quantum-chemistry world (libint, contracted GTOs, analytic ERI) with the plane-wave world (FFT-Poisson). GPAW is a **real-space finite-difference code** that happens to use PAW for all-electron accuracy. They share the FFT-Poisson solver and libxc, but differ in almost everything else. vibe-qc additionally offers BIPOLE and GDF routes that are entirely Gaussian-native -- no real-space grid involved -- which GPAW has no equivalent for.
