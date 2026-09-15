# Semiempirical Structure-Optimization Stack, Handover Document

**Status: historical v0.9.0 DFTB handover; audited 2026-05-31**
**Current snapshot:** DFTB covers 91 in-house elements; GFN2-xTB is gated
experimental; PM6/OMx are present but must be validated per use. See
`docs/user_guide/semiempirical.md` and `docs/roadmap.md` for the live full-stack map.

---

## Quick Start

```python
# Molecular DFTB0
from vibeqc.semiempirical import DFTB0Model, run_dftb0
model = DFTB0Model(molecule)
energy = model.energy()
gradient = model.gradient()  # analytic

# Molecular SCC-DFTB (charge self-consistent)
from vibeqc.semiempirical import SCCDFTBModel, run_scc_dftb
model = SCCDFTBModel(molecule, charge_mixing=0.2)
energy = model.energy()

# Add D3(BJ) dispersion
from vibeqc.semiempirical import DispersionCorrectedModel, DFTB_D3_DEFAULTS
disp_model = DispersionCorrectedModel(model, DFTB_D3_DEFAULTS)

# Periodic DFTB0/SCC at Gamma-point
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core.semiempirical import run_dftb0_gamma, run_scc_dftb_gamma

# Stress tensor (finite-difference)
from vibeqc.semiempirical import compute_stress_fd
stress = compute_stress_fd(periodic_system, energy_fn)

# Variable-cell optimization
from vibeqc.semiempirical import optimize_cell
optimized = optimize_cell(periodic_system, energy_fn)

# Preoptimization workflow (semiempirical → ab initio handoff)
from vibeqc.semiempirical.preoptimize import preoptimize_molecule, preoptimize_periodic
```

---

## Architecture

```
SemiempiricalModel (Python ABC)            — model.py
├── DFTB0Model                             — dftb0.py
├── SCCDFTBModel                           — dftb0.py
└── DispersionCorrectedModel               — dispersion.py

C++ Backend (namespace vibeqc::semiempirical)
├── SemiempiricalParameters                — parameters.hpp/cpp
├── SemiempiricalBasis                     — basis.hpp/cpp
├── SemiempiricalHamiltonianBuilder        — hamiltonian.hpp/cpp
├── DFTB0Result + run_dftb0()              — dftb0.hpp/cpp
├── SCCOptions/SCCDFTBResult/run_scc_dftb  — scc_dftb.hpp/cpp
├── compute_dftb0_gradient()               — gradient.hpp/cpp
├── PeriodicDFTB0Result/run_dftb0_gamma    — periodic_dftb0.hpp/cpp
├── PeriodicSCCDFTBResult/run_scc_dftb_gamma — periodic_scc_dftb.hpp/cpp
└── pybind11 bindings                      — bindings.cpp (lines ~4110-4260)

Python utilities
├── compute_stress_fd / optimize_cell       — periodic.py
├── preoptimize_molecule / _periodic        — preoptimize.py
└── d3bj_energy / DispersionCorrectedModel — dispersion.py
```

---

## File Inventory

### C++ headers (`cpp/include/vibeqc/semiempirical/`)
| File | Purpose |
|------|---------|
| `parameters.hpp` | ElementData, RepulsivePair, SemiempiricalParameters (Hubbard U, repulsive, WH kappa) |
| `basis.hpp` | SemiempiricalBasis, STO-6G construction via libint |
| `hamiltonian.hpp` | H⁰ builder, Mulliken charges, γ matrix, SCC correction |
| `dftb0.hpp` | DFTB0Result, run_dftb0() |
| `scc_dftb.hpp` | SCCOptions, SCCDFTBResult, run_scc_dftb() |
| `gradient.hpp` | compute_dftb0_gradient(), dftb0_repulsive_gradient() |
| `periodic_dftb0.hpp` | PeriodicDFTB0Options, PeriodicDFTB0Result, run_dftb0_gamma() |
| `periodic_scc_dftb.hpp` | PeriodicSCCOptions, PeriodicSCCDFTBResult, run_scc_dftb_gamma() |

### C++ sources (`cpp/src/semiempirical/`)
| File | Lines | Purpose |
|------|-------|---------|
| `parameters.cpp` | ~110 | Hardcoded H/C/N/O table, repulsive_energy(R) helper |
| `basis.cpp` | ~175 | STO-6G tables (HSP 1969), shell construction |
| `hamiltonian.cpp` | ~230 | WH H⁰ builder, Mulliken charges, γ matrix, H^{SCC} |
| `dftb0.cpp` | ~100 | DFTB0 driver |
| `scc_dftb.cpp` | ~160 | SCC SCF loop |
| `gradient.cpp` | ~180 | Analytic gradient via dS/dR + repulsive gradient |
| `periodic_dftb0.cpp` | ~190 | Gamma-point lattice-summed DFTB0 |
| `periodic_scc_dftb.cpp` | ~310 | Periodic SCC SCF loop |

### Python modules (`python/vibeqc/semiempirical/`)
| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports |
| `model.py` | SemiempiricalModel ABC (energy/gradient contract) |
| `parameters.py` | Python-side parameter helpers |
| `basis.py` | Python basis helpers (thin) |
| `dftb0.py` | DFTB0Model, SCCDFTBModel, run_dftb0, run_scc_dftb |
| `periodic.py` | compute_stress_fd, optimize_cell, _fd_forces |
| `preoptimize.py` | preoptimize_molecule, preoptimize_periodic |
| `dispersion.py` | DispersionCorrectedModel, d3bj_energy, DFTB_D3_DEFAULTS |

### Build / bindings
| File | Change |
|------|--------|
| `cpp/CMakeLists.txt` | Added 8 semiempirical source files |
| `cpp/src/bindings.cpp` | ~150 lines of pybind11 bindings (lines 31-33, 4110-4270) |
| `cpp/src/lattice_integrals.cpp` | Bug fix: added missing `compute_1e_lattice_matrix_explicit` |
| `python/vibeqc/__init__.py` | No changes needed (semiempirical is a sub-package) |

### Tests
| File | Tests |
|------|-------|
| `tests/test_semiempirical_dftb0.py` | 45 tests (Params, Energy, Gradient, SCC, Periodic, Stress, Dispersion, Optimization) |

### Documentation
| File | Content |
|------|---------|
| `docs/semiempirical_roadmap.md` | This file, handover / status |
| `docs/semiempirical_stage1_design.md` | Stage 1 math model + pseudocode |

---

## Stage Summary

### Stage 1, DFTB0 molecular prototype (non-SCC)
- Wolfsberg-Helmholtz H⁰ with STO-6G minimal basis
- E = Σ n_i ε_i + Σ V_rep(R_AB)
- Finite-difference gradients

### Stage 2, Analytic molecular gradients
- dE/dR = Σ [D·½κ·h̄ - W] · dS/dR + dE_rep/dR
- Uses libint overlap derivative engine (deriv_order=1)
- Geometry optimization converges

### Stage 3, SCC-DFTB for molecules
- Hubbard U: H(0.4195), C(0.3647), N(0.4038), O(0.4459) Ha
- Mulliken charges: Δq_A = n_val − Σ_{μ∈A} (D·S)_{μμ}
- γ matrix: Ohno-Klopman γ_{AB} = 1/√(R² + 0.25(1/U_A+1/U_B)²)
- H^{SCC} = H⁰ + ½ S (V_A+V_B), E = Σ n_i ε_i + E_rep + ½ Δq·γ·Δq
- Charge conservation: Σ Δq_A = 0 (normalized)

### Stage 4, Periodic Gamma-point DFTB0
- H_Γ = Σ_g H⁰(g), S_Γ = Σ_g S(g) via lattice-summed S(g)
- On-site ε_l only in g=0 cell
- Periodic repulsive with ½ factor for image pairs
- Molecular-limit check: g=0 only matches molecular DFTB0 exactly

### Stage 5, Periodic SCC-DFTB
- Periodic γ: γ_{AB}^{per} = Σ_g 1/√(|R+g|² + η²)
- SCC SCF loop with charge mixing
- CH₂ chain: 66 iter, 0.4 Ha SCC stabilization

### Stage 6, Stress tensor (finite-difference)
- σ_{ij} = (1/V) · [E(+h·e_{ij}) − E(−h·e_{ij})] / (2h)
- Only in-plane components computed (dim × dim)

### Stage 7, Variable-cell optimization
- ASE ExpCellFilter + BFGSLineSearch
- Simultaneous atomic + lattice optimization
- Isotropic option via StrainFilter

### Stage 8, Preoptimization workflow
- preoptimize_molecule(): SCC/DFTB0 → optimized geometry
- preoptimize_periodic(): periodic variant
- Drop-in for ab initio refinement pipeline

### Stage 9+, Dispersion + improved repulsive
- D3(BJ) via vibe-qc's existing compute_d3bj()
- DispersionCorrectedModel wraps any model
- R⁻¹² repulsive potential: V_rep = A/R¹² (B=0) or A·exp(−B·R) (B>0)
- dV_rep/dR = −12A/R¹³ for analytic gradient
- Creates proper energy minima (H₂O at R_OH ≈ 1.93 bohr)

---

## Mathematical Models at a Glance

### DFTB0
```
H⁰_{μν} = ε_l                    (on-site, same atom)
H⁰_{μν} = ½κ · S_{μν} · h̄_{AB}  (two-center, Wolfsberg-Helmholtz)
E = 2 Σ_i^{occ} ε_i + Σ_{A<B} V_rep(R_AB)
```

### SCC-DFTB
```
Δq_A = n_val(A) − Σ_{μ∈A} (D S)_{μμ}
γ_{AB} = 1/√(R² + 0.25(1/U_A+1/U_B)²),  γ_{AA} = U_A
V_A = Σ_C γ_{AC} Δq_C
H_{μν} = H⁰_{μν} + ½ S_{μν} (V_A + V_B)    (μ∈A, ν∈B)
E = Σ_i n_i ε_i + E_rep + ½ Σ_{AB} γ_{AB} Δq_A Δq_B
```

### Analytic gradient (DFTB0)
```
W = 2 C_occ diag(ε_occ) C_occ^T
M_{μν} = D_{μν} · ½κ · h̄_{AB} − W_{μν}     (two-center)
M_{μν} = −W_{μν}                             (on-site)
dE/dR_A = Σ_{μν} M_{μν} · dS_{μν}/dR_A + dE_rep/dR_A
```

### Repulsive potential
```
V_rep(R) = A/R¹²           (default, B=0 in parameters)
V_rep(R) = A·exp(−B·R)    (legacy, B>0)
dE_rep/dR_A = Σ_{B≠A} dV_rep/dR · (R_A−R_B)/R
dV/dR = −12A/R¹³          (R⁻¹² form)
dV/dR = −A·B·exp(−B·R)   (exponential form)
```

---

## Parameter Tables

### On-site energies (Hartree), 24 elements
| Z | Sym | ε_s | ε_p | ε_d | U | n_val |
|---|-----|------|------|------|--------|-------|
| 1 | H | −0.2066 | - | - | 0.4195 | 1 |
| 2 | He | −0.6000 | - | - | 0.6000 | 2 |
| 3 | Li | −0.1500 | - | - | 0.2500 | 1 |
| 4 | Be | −0.2500 | - | - | 0.3000 | 2 |
| 5 | B | −0.3500 | −0.1500 | - | 0.3300 | 3 |
| 6 | C | −0.5040 | −0.1943 | - | 0.3647 | 4 |
| 7 | N | −0.6809 | −0.2864 | - | 0.4038 | 5 |
| 8 | O | −0.8877 | −0.3321 | - | 0.4459 | 6 |
| 9 | F | −1.1000 | −0.4800 | - | 0.5300 | 7 |
| 10 | Ne | −1.5000 | −0.7000 | - | 0.6500 | 8 |
| 11 | Na | −0.1200 | - | - | 0.2000 | 1 |
| 12 | Mg | −0.2000 | - | - | 0.2500 | 2 |
| 13 | Al | −0.2800 | −0.1000 | - | 0.2800 | 3 |
| 14 | Si | −0.4000 | −0.1500 | - | 0.3000 | 4 |
| 15 | P | −0.5000 | −0.1800 | - | 0.3200 | 5 |
| 16 | S | −0.6000 | −0.2200 | - | 0.3500 | 6 |
| 17 | Cl | −0.7500 | −0.2800 | - | 0.4200 | 7 |
| 18 | Ar | −0.9000 | −0.4000 | - | 0.5000 | 8 |
| 19 | K | −0.1000 | - | - | 0.1800 | 1 |
| 26 | Fe | −0.4000 | −0.1000 | −0.2500 | 0.2500 | 8 |
| 29 | Cu | −0.4500 | −0.1200 | −0.3000 | 0.2800 | 11 |
| 30 | Zn | −0.5000 | −0.1400 | −0.3500 | 0.3000 | 12 |
| 35 | Br | −0.7000 | −0.2500 | - | 0.4500 | 7 |
| 53 | I | −0.5500 | −0.2000 | - | 0.3800 | 7 |

Repulsive: R⁻¹² with A = 100·U₁·U₂ default estimator (explicit values for H/C/N/O/F/P/S/Cl pairs).

------|---|------|---|------|---|
| H-H | 5 | C-C | 60 | N-N | 80 |
| H-C | 25 | C-N | 70 | N-O | 60 |
| H-N | 25 | C-O | 50 | O-O | 40 |
| H-O | 15 | C-F | 80 | F-F | 25 |
| H-F | 20 | C-P | 120 | P-P | 200 |
| H-P | 40 | C-S | 100 | S-S | 150 |
| H-S | 35 | C-Cl | 90 | Cl-Cl | 120 |
| H-Cl | 30 | N-F | 75 | P-Cl | 160 |
| | | P-S | 170 | S-Cl | 140 |

---


---

## Parameter Provenance (§1 licensing)

ALL 58-element parameters are **in-house estimates**, constructed from
Slater's rules (STO exponents), periodic-trend scaling (on-site
energies), and approximate Hubbard U estimates. **No values are sourced
from dftb.org parameter sets** (mio, mio-0-1, 3ob, matsci, trans3d,
etc.) or any other third-party DFTB parameterization. The R⁻¹² repulsive
form and `default_repulsive_A` estimator are original.

### Element families

| Family | Elements | Provenance |
|--------|----------|------------|
| H | H | Slater ζ, DFTB-approximate ε |
| 2p (row 2) | C, N, O, F, Ne | Slater ζ, periodic-trend ε |
| 3p (row 3) | Al, Si, P, S, Cl, Ar | Slater ζ, periodic-trend ε |
| s-block | Li, Be, Na, Mg, K, Ca, Rb, Sr, Cs, Ba | Slater ζ, periodic-trend ε |
| 3d metals | Sc-Zn | Slater ζ, HSP-1969 d-table, periodic-trend ε |
| 4d metals | Y-Cd | Slater ζ, HSP-1969 d-table, periodic-trend ε |
| 5d metals | W-Hg, Re-Ir | Slater ζ, HSP-1969 d-table, periodic-trend ε |
| Heavy p-block | Ga, Ge, As, Se, Br, Kr, I | Slater ζ, periodic-trend ε |

### Quality (honest-scope)

**These are approximate, order-of-magnitude parameters. They are NOT
production-grade DFTB.** Intended use: fast screening and geometry
preoptimization before HF/DFT refinement. Bond lengths are roughly
correct (±0.3 bohr for organics) but energies are not transferable
and should not be compared to experiment or high-level theory.
No systematic validation against DFT/DFTB+/experiment has been
performed.


## Testing

### Run all semiempirical tests
```bash
.venv/bin/python3 -m pytest tests/test_semiempirical_dftb0.py -v
```

### Run including existing regression tests
```bash
.venv/bin/python3 -m pytest tests/test_semiempirical_dftb0.py tests/test_rhf.py tests/test_molecule.py tests/test_integrals.py -v
```

### Test categories (45 semiempirical tests)
| Class | Count | What |
|-------|-------|------|
| TestParameters | 5 | Element data, Hubbard U, repulsive |
| TestDFTB0Energy | 11 | H₂/H₂O/CH₄ energies, dimensions, symmetry |
| TestDFTB0Gradient | 5 | Force direction, sum-zero, torque-zero, FD consistency |
| TestDFTB0AnalyticGradient | 5 | Analytic vs FD comparison, force sum-zero |
| TestSemiempiricalModel | 2 | ABC contract |
| TestSCCDFTB | 6 | SCC convergence, charges, energy decomposition |
| TestPeriodicDFTB0 | 4 | Periodic energy, molecular limit, n_cells |
| TestPeriodicSCCDFTB | 3 | Periodic SCC convergence, charges |
| TestStress | 2 | Stress tensor, symmetry |
| TestDispersion | 4 | D3 energy, model wrapping, gradient |
| TestVariableCellOptimization | 1 | (slow) variable-cell optimization |
| TestPreoptimization | 1 | (slow) molecule preoptimization |
| TestGeometryOptimization | 1 | (slow) H₂O BFGS optimization |

---

## Known Limitations

1. **45 elements across periodic table:** H, C, N, O, F, P, S, Cl. Extending to F, S, P, Cl, etc. requires adding ElementData entries.
2. **SCC gradients are finite-difference.** Analytic SCC gradients need dγ/dR and charge-response terms, deferred.
3. **Gamma-point only for periodic systems.** General k-point sampling requires Bloch-sum machinery.
4. **Simple repulsive form.** R⁻¹² is physically motivated but not DFT-fitted. Real DFTB uses splines.
5. **No open-shell support.** Only closed-shell (even electron count, multiplicity=1).
6. **No ECP or relativistic corrections.** Light-element only.

---

## Resumption Notes

To continue development:

1. **Read this file in full**, it's the single source of truth.
2. **Build and test:** `pip install -e . --no-build-isolation && pytest tests/test_semiempirical_dftb0.py`
3. **Key source files to study first:**
   - `cpp/include/vibeqc/semiempirical/parameters.hpp`, the data model
   - `cpp/src/semiempirical/dftb0.cpp`, simplest driver
   - `cpp/src/semiempirical/scc_dftb.cpp`, SCC SCF loop pattern
   - `python/vibeqc/semiempirical/dftb0.py`, Python API pattern
4. **Natural next steps:**
   - Extend element coverage (add F, S, Cl)
   - Implement analytic SCC gradients
   - Add general k-point support for periodic systems
   - Implement GFN2-xTB-style multipole electrostatics
   - Add halogen bonding correction
