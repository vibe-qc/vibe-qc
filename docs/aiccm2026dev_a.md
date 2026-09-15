---
myst:
  html_meta:
    "description": "The aiccm2026dev-a development line of the ab-initio Cyclic Cluster Model (AICCM) in vibe-qc: union-and-weight Γ-CCM HF/KS/MP2/CCSD(T), plus separately labelled neutral fitted-torus controls. Experimental."
---

# AICCM - Γ-CCM, the `aiccm2026dev-a` line (experimental)

```{warning}
**Experimental, and one of two parallel ab-initio CCM development lines.** This
page documents the **`aiccm2026dev-a`** four-center weighting and its method stack,
developed and validated independently for a prospective construction-level
comparison against a second line. The current reportable route map is empty.
It is separate, on purpose, from the side-by-side catalog in
[experimental features](experimental/index.md) and the
[CCM reference](user_guide/cyclic_cluster_model.md). The post-HF / padded-ERI
builders are research paths (kept out of `__all__`). The union-and-weight
four-center route defines Γ-CCM; the neutral GDF and real-Gamma solvers described
below are separate same-Hamiltonian controls and must not be relabelled as the
Γ-CCM construction. Numbers here are research-grade. Every public SCF driver
emits `vibeqc.AICCM2026DevAExperimentalWarning` (correlation, gradient, and
property drivers inherit it through the SCF they run); silence it with
`warnings.filterwarnings("ignore", category=vibeqc.AICCM2026DevAExperimentalWarning)`.
```

## What `aiccm2026dev-a` is

The ab-initio Cyclic Cluster Model (AICCM) computes a crystal's electronic
structure on a finite **Born-von-Karman torus** (the cluster with cyclic
boundary conditions), feeding vibe-qc's ordinary *molecular* method kernels a
**Wigner-Seitz-supercell (WSSC) weighted** set of integrals. The hard part is
the two-electron **four-center** term.

```{note}
**Terminology corrected by D89 (2026-07-15).** `aiccm2026dev-a` is **Γ-CCM**,
the union-and-weight/Wigner-Seitz integral-weighting approach within the
*variational finite-BvK-torus CCM* family. Its sibling **χ-CCM**
(`aiccm2026dev-b`) uses the finite-translation-group character construction.
They are distinct construction approaches. A declared common exchange-`q=0`
convention (`exxdiv="ewald"` where applicable) is a comparison constraint, not
an identification of the constructions. Their distinction is not a choice of
Coulomb kernels, and equality for a specified operator and route is evidence to
establish, not a naming premise. The library selector is
`method="aiccm2026dev-a"` (`run_ccm_scf(ccm, route="four-center")`); from the
runner every AICCM formulation is selected by
`run_periodic_job(method="aiccm", variant=...)`, and the bare spellings
`gamma`, `gamma-ccm`, `gamma_ccm` fail closed on both surfaces (ruling R1,
2026-08-21, recorded as D124, which supersedes the bare-`gamma` clause of
D89). The 2014 baseline is **AICCM** (Peintinger &
Bredow). A real-Gamma neutral-torus solver can be an internal representation
control for one already specified finite Hamiltonian without thereby
constituting evidence for the union-and-weight Γ-CCM approach.

**Binding clarified by D90 (2026-07-16).** A namespace, real-Gamma evaluation,
or reused solver cannot assign a construction identity. In particular, the
neutral-only UCCSD(T) and DLPNO APIs documented below are neutral fitted-torus
correlation controls, not Γ-CCM or χ-CCM results.
```

`aiccm2026dev-a` is the **symmetric four-center** (derived in `AICCM_ALGORITHM.md`
§13). It replaces the historical `union12` product weight
`ω_μν · ½(ω_μρ+ω_νρ) · ω_ρσ` - which singles out the ket anchor ρ and so breaks
the 8-fold permutational symmetry an SCF energy functional requires - with a
**symmetric bridge** and an **independent minimum-image fold**:

```
ω^sym_{μνρσ} = ω_μν · ω_ρσ · ¼(ω_μρ + ω_νρ + ω_μσ + ω_νσ)
```

with ρ and σ each folded to *their own* minimum image of the home bra (the
ket-pair weight ω_ρσ taken at the relative cell). The result is **exactly 8-fold
permutationally symmetric for any lattice** - a genuine variational ERI set -
verified to machine precision (≤3e-18) on 1-D chains and on 2-D hexagonal and
oblique lattices, where `union12` is 1.6e-3 (1-D) to 2.1e-2 (2-D) asymmetric. The
weighting rests only on the WSC minimum image, so it is lattice-general (the
reference cell is the WSC).

In 1-D and orthorhombic lattices the two weightings give the same energy (the WSC
symmetry hides the asymmetry); in non-orthorhombic ≥2-D they diverge - e.g. a 2-D
hex H lattice gives `aiccm2026dev-a` -0.400456 vs `union12` -0.399044 Ha/atom, and a 3-D
BCC lattice diverges by ~5 mHa/atom.

## Sizing the cluster: interaction radius (the real-space dual of a k-mesh)

The cluster can be sized two ways. The explicit mesh `nrep=(N1,N2,N3)` is the
advanced override; the **recommended** knob is a real-space **interaction range**
`R_c` - how far the WSSC interactions reach around every atom - from which the
minimal `nrep` is **derived automatically**:

```python
from vibeqc.periodic.ccm import CCMSystem, ccm_interaction_range_scan

# size by a real-space radius (bohr); nrep is derived (Å convenience too)
ccm = CCMSystem.from_interaction_range(unit, 12.0, "sto-3g")        # R_c = 12 bohr
ccm = CCMSystem(unit, basis="sto-3g", interaction_range_ang=6.5)    # R_c = 6.5 Å
print(ccm.nrep, ccm.wsc_inscribed_radius, ccm.kspacing_equiv)
```

This is the geometric **real-space dual of k-point sampling**: a cyclic cluster
of size `(N1,N2,N3)` defines a Born-von-Kármán torus whose compatible
characters form a Γ-centred `(N1,N2,N3)` mesh, and a radius `R_c` corresponds to
a uniform k-spacing **`Δk = π/R_c`**. This geometry does not equate the
union-and-weight and finite-character Hamiltonian constructions.

The derivation is rigorous (`docs/aiccm2026dev_a_followon.md` § A): the minimal
cluster is the one whose **supercell Wigner-Seitz cell encloses a sphere of
radius `R_c`** around every atom. The WS inscribed-sphere radius is
`r_in = λ₁/2` (half the shortest lattice vector); the per-direction rule
`N_i = ⌈2 R_c / d_i⌉` (`d_i` = unit-cell interplanar spacing) provably guarantees
`r_in ≥ R_c` because `λ₁ ≥ min_i N_i d_i`. It is tight for orthogonal cells and
conservative (never under-shoots) for oblique ones; `dim` caps non-periodic
directions.

A radius **convergence scan** - the CCM analogue of a k-point convergence study -
comes for free:

```python
scan = ccm_interaction_range_scan(unit, "sto-3g",
                                  radii_bohr=[6, 9, 12, 15, 18, 24, 30])
print(scan.converged_radius(1e-4))     # smallest R_c converged to 0.1 mHa/atom
```

On the 1-D H₂ chain (6-bohr cell, STO-3G, `aiccm2026dev-a`) E/atom converges to
0.1 mHa/atom by `R_c ≈ 9 bohr` ((3,1,1)) and to 10 µHa/atom by `R_c ≈ 18 bohr`
((6,1,1)). The demonstration across all 28 test-set crystals (every Bravais
lattice) is
[`studies/aiccm-2026/demo_interaction_range.py`](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/demo_interaction_range.py).

## The method stack

Every driver takes `method="aiccm2026dev-a"` (default is `"union12"`). The
four-center Coulomb/exchange carries the cyclic boundary conditions, and
correlation uses the CCM molecular-orbital integrals. The legacy libxc branch
evaluates XC on the molecular supercell grid. A registered pure full-grid
external provider instead uses the literal CCM numerical-integration
construction: a reference-cluster atom grid, translated periodic cluster AOs,
and an atomic partition normalized over the real and virtual cluster atoms
(Janetzko, Köster, and Salahub, *J. Chem. Phys.* **128**, 024102 (2008),
DOI `10.1063/1.2817582`, Eqs. 27-32).

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf, run_ccm_rhf_scalable
from vibeqc.periodic.ccm.uhf import run_ccm_uhf
from vibeqc.periodic.ccm.dft import run_ccm_rks, run_ccm_uks
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.ump2 import run_ccm_ump2
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.ri import (
    run_ccm_rhf_gdf, run_ccm_rks_gdf, run_ccm_rhf_rij, run_ccm_rhf_rijcosx)
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct

BOHR = 1.0 / 0.529177210903
pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]     # H4 alternating chain
unit = PeriodicSystem(3, np.diag([4.0 * BOHR, 40.0, 40.0]),
                      [Atom(1, p) for p in pos], charge=0, multiplicity=1)
ccm = CCMSystem(unit, (4, 1, 1), "sto-3g")

# --- Hartree-Fock and Kohn-Sham, with the four-center Coulomb ---------------
rhf = run_ccm_rhf(ccm, method="aiccm2026dev-a")          # closed-shell HF
sca = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a") # same, scalable C++ (reaches 3-D)
rks = run_ccm_rks(ccm, "pbe", method="aiccm2026dev-a")   # KS-DFT (libxc)
# A registered pure full-grid provider, including SKALA-1.1, uses the
# periodic reference-cluster grid. External-provider hybrids fail closed.
learned_rks = run_ccm_rks(ccm, "skala-1.1", method="aiccm2026dev-a")

# --- correlation: MP2 / UMP2 / CCSD(T), on the CCM MO integrals -------------
mp2 = run_ccm_mp2(ccm, method="aiccm2026dev-a")
cc  = run_ccm_ccsd(ccm, method="aiccm2026dev-a")         # CCSD + (T)
print(rhf.energy_per_atom, mp2.e_correlation, cc.e_correlation, cc.e_t)

# --- construction-specific weighted approximations --------------------------
cox = run_ccm_rhf_rijcosx(ccm)  # WSSC RI-J + chain-of-spheres K research route

# --- neutral fitted-torus controls (not Γ-CCM construction routes) ----------
ri  = run_ccm_rhf_gdf(ccm)      # character/Bloch evaluation of the control
rik = run_ccm_rks_gdf(ccm, "pbe")
dt  = run_ccm_rhf_direct(ccm)   # real-Gamma evaluation of the same control
                                # == run_ccm_rhf_gdf at a fully matched seam/fit
from vibeqc.periodic.ccm import run_ccm_rks_direct
dks = run_ccm_rks_direct(ccm, "pbe")    # k-free KS-DFT (pure + global hybrids;
                                        # periodic XC folded onto the torus,
                                        # hybrid seam scaled by a_x)
dhs = run_ccm_rks_direct(ccm, "hse06")  # EXPERIMENTAL: screened (HSE-class)
                                        # hybrids -- erfc-attenuated fitted
                                        # kernel + sigma_0 SDS zero mode, no
                                        # exchange-q=0 seam (exxdiv-independent);
                                        # full-range-arm RSH still fails closed
from vibeqc.periodic.ccm.direct import run_ccm_double_hybrid_direct
ddh = run_ccm_double_hybrid_direct(ccm, "b2plyp")
                                        # EXPERIMENTAL: closed-shell double
                                        # hybrid -- hybrid-KS SCF half + SCS-MP2
                                        # on the KS orbitals, one shared L
                                        # (BvK-ewald direct reference)
from vibeqc.periodic.ccm.direct import run_ccm_direct_gradient
grd = run_ccm_direct_gradient(ccm)      # EXPERIMENTAL: analytic forces via the
                                        # verified representation identity
                                        # (multi-k rsgdf gradient + per-run
                                        # energy-parity gate); dE_cell/dR.
                                        # HF/pure/global-hybrid KS, closed or
                                        # open shell; screened/DH fail closed
from vibeqc.periodic.ccm.direct import run_ccm_direct_optimize
opt = run_ccm_direct_optimize(ccm)      # EXPERIMENTAL: unit-cell relaxation
                                        # (fixed lattice) on the same verified
                                        # surface; steps ride the control
                                        # gradient, endpoints carry the full
                                        # direct-vs-control parity check

# --- the same control through the unified runner (EXPERIMENTAL, 2026-07-26) --
# jk_method='real-gamma' in run_periodic_job: the Gamma-centred k-mesh
# argument defines the BvK torus (nrep); results are per unit cell with the
# full artefact set, and the convention record (exchange_q0 + applicability,
# D86 executed values, lattice_vector_convention="columns", screened-exchange
# assembly) lands in the .system manifest and the
# x_vibeqc.real_gamma_convention QVF vendor section.
# run_periodic_job(cell, basis, method="RKS", functional="hse06",
#                  jk_method="real-gamma", kpoints=(2, 2, 2), output="rg")

# --- pick the evaluation route by keyword -----------------------------------
from vibeqc.periodic.ccm import run_ccm_scf
s1 = run_ccm_scf(ccm)                       # 2014 four-center lineage (default)
s2 = run_ccm_scf(ccm, route="four-center") # same lineage, explicit name
s3 = run_ccm_scf(ccm, route="neutral-bloch")  # neutral Γ-CCM construction,
                                            # Bloch representation
s4 = run_ccm_scf(ccm, route="real-gamma")  # same construction in a real
                                            # supercell
s5 = run_ccm_scf(ccm, route="real-gamma", functional="pbe0")
```

The `learned_rks` line additionally requires the optional `.[skala]` runtime,
Linux, and Python 3.11 through 3.13. It is an experimental fixed-geometry
single point; analytic derivatives and external-provider hybrids fail closed.

### Selecting the AICCM route by keyword

`run_ccm_scf(ccm, route=...)` selects either the neutral Γ-CCM construction
(ruling R1, 2026-08-21) or the 2014 union-and-weight four-centre lineage. The
two neutral producers are Fourier-related representations of one
block-circulant neutral Hamiltonian; the four-centre lineage is the object
Paper 1 diagnoses as non-variational:

| route keyword | what it runs | aliases |
|---|---|---|
| `"four-center"` | the 2014 union-and-weight/Wigner-Seitz four-centre lineage (`run_ccm_rhf`/`run_ccm_rks`), diagnosed non-variational in Paper 1; this is the default | `"4c"`, `"aiccm-hf"`, `"aiccm2026dev-a"` |
| `"neutral-bloch"` | the neutral Γ-CCM construction (Paper-1) through multi-k GDF | `"aiccm-ri"`, `"gdf"`, `"gdf-control"` |
| `"real-gamma"` | the same neutral Γ-CCM construction in the k-free real-Γ supercell representation; no k-mesh | `"aiccm-hf-direct"` |

The paper-facing spellings `"gamma"` / `"gamma-ccm"` / `"gamma_ccm"` fail
closed with a message naming both neutral producers (a single keyword cannot
choose between them), so they cannot silently select the four-centre lineage.
The runner applies the same rule: `run_periodic_job(method="aiccm",
variant=...)` names the formulation (`"real-gamma"`, `"neutral-bloch"`,
`"four-center"`, `"chi"`), `jk_method="gamma"` fails closed with the same
message, and the dev-era `jk_method` spellings resolve with a
`DeprecationWarning` (D124; user guide: [AICCM](user_guide/aiccm.md)).

For a one-page index of which route supports HF, KS, MP2, CCSD(T), DLPNO, and
analytic gradients, across this line and χ-CCM, see the
[AICCM route support matrix](experimental/aiccm2026dev_route_matrix.md).

Every route is **shell-aware**: an open-shell cluster (odd electron count, or
an explicit unit-cell multiplicity above the parity floor, e.g. a triplet
cell) dispatches to the unrestricted sibling automatically -
`run_ccm_uhf`/`run_ccm_uks` (four-center), `run_ccm_uhf_gdf`/`run_ccm_uks_gdf`
(neutral-bloch), `run_ccm_uhf_direct`/`run_ccm_uks_direct` (real-gamma).
Closed-shell
clusters keep the restricted drivers unchanged.

The public selector `PeriodicJKMethod.AICCM2026DEV_A` names the
union-and-weight four-centre lineage, while `.AICCM2026DEV_B` names
χ-CCM. The A-prefixed real-Gamma spellings are intentionally rejected because
they mix the dev-a four-centre lineage prefix with a neutral-construction
producer spelling. Select that producer as `run_ccm_scf(..., route="real-gamma")`, as
`run_periodic_job(..., jk_method="real-gamma")` (wired 2026-07-26,
EXPERIMENTAL - see the runner section above for the convention record it
writes), or through the test-set harness route `aiccm-hf-direct`. The
**four-centre lineage selector** `aiccm2026dev-a` computes through
`run_periodic_job(method="aiccm", variant="four-center")`; its dev-era
`jk_method="aiccm2026dev-a"` spelling resolves to the same arm with a
`DeprecationWarning`. Chi (χ-CCM, `aiccm2026dev-b`) stays a separate line
reached via `run_periodic_job` and is intentionally not a `run_ccm_scf` route.

| level | HF | KS | open-shell | correlation |
|---|---|---|---|---|
| **Γ-CCM 4-center** | `run_ccm_rhf` / `run_ccm_rhf_scalable`; high-level `variant="four-center"` | `run_ccm_rks`; libxc plus pure full-grid external providers through both the low-level and high-level APIs | `run_ccm_uhf`, `run_ccm_uks`; high-level `variant="four-center"`; `run_ccm_uks` and the runner accept the same pure full-grid external providers | `run_ccm_mp2`, `run_ccm_ump2`, `run_ccm_ccsd` (+(T)) with `method="aiccm2026dev-a"` |
| **neutral fitted-torus control (3c)** | `run_ccm_rhf_gdf`, `run_ccm_rhf_direct` (k-free, == GDF at `exxdiv="ewald"`), `run_ccm_rhf_ri_neutral` (strict-zero-mode) | `run_ccm_rks_gdf`, `run_ccm_rks_direct` (k-free; pure + global hybrids, periodic XC folded onto the torus - gated against `run_ccm_rks_gdf`, molecular RKS in the vacuum limit + external PySCF KRKS, see `test_ccm_rks_direct.py`), `run_ccm_uks_gdf` + `run_ccm_uks_direct` (open-shell KS: per-spin seam × `a_x`, spin-polarized periodic XC; collapses to `run_ccm_rks_direct` on closed shells; `test_ccm_uks_direct.py`); **EXPERIMENTAL**: both direct KS drivers additionally run screened HSE-class hybrids (erfc-attenuated fitted kernel + `σ₀·S·D·S` zero mode, exxdiv-independent, no seam; `test_ccm_rsh_direct.py`; full-range-arm RSH fails closed), and `run_ccm_double_hybrid_direct` composes closed-shell double hybrids (hybrid-KS SCF + SCS-MP2 on the KS orbitals, one shared `L`; `test_ccm_double_hybrid_direct.py`) | `run_ccm_uhf(cderi=L)` (strict-zero-mode), `run_ccm_uhf_gdf` (multi-k KUHF; per-unit-cell spin bookkeeping, see its warning), `run_ccm_uhf_direct` (k-free, BvK-ewald seam per spin, == `run_ccm_rhf_direct` on closed shells) | `run_ccm_ri_mp2` (one-call, closed or open shell), `run_ccm_ri_ccsd` (one-call, closed shell), `run_ccm_{mp2,ccsd,ump2}(cderi=L)`, and `run_ccm_uccsd` - density-fit from the neutral cderi `L`, **no dense n⁴ four-center**, reaches moderate 3-D. `reference="direct"` selects a complete folded/Ewald reference route, not a seam-only toggle. None is assigned Γ-CCM or χ-CCM identity. |
| **neutral-control RIJCOSX approximation (3c)** | `run_ccm_rhf_direct_rijcosx` (dim=3) | - | - | - |
| **Γ-CCM WSSC RIJCOSX research route (3c)** | `run_ccm_rhf_rijcosx` (bare research operator; excluded from reportable Γ/χ comparisons) | - | - | - |
| **neutral-control local correlation** | - | - | - | `ccm_dlpno_mp2`, `ccm_dlpno_ccsd` (union), `ccm_dlpno_ccsd_coupled` (per-pair-coupled CCSD(T)), open-shell `ccm_dlpno_ump2`, `ccm_dlpno_uccsd` - all on the neutral fitted-torus reference, with no Γ-CCM or χ-CCM identity assigned |

Properties (any converged CCM SCF): `ccm_homo_lumo_gap`, `ccm_mulliken_charges`,
`ccm_lowdin_charges`, `ccm_dipole`, `ccm_numerical_gradient` (see *Properties* below).

## Creating inputs

There are three ways to create a Γ-CCM union-and-weight input, from fastest to
most customisable.

### 1. The test-set runner (fastest)

The `studies/aiccm-2026/` directory ships a curated 28-system test set and a
single-command runner. Each system has a default cluster size, basis, and
dimensionality - override any parameter on the command line.

```bash
# Quick start - HF on the 1-D H chain
cd studies/aiccm-2026
python run_case.py h-chain aiccm-hf

# Change the cluster size
python run_case.py c-diamond aiccm-hf --nrep 2 2 2

# KS-DFT on a 2-D system
python run_case.py graphene aiccm-ks --functional pbe

# Run on a specific basis
python run_case.py mgo aiccm-ri --basis pob-tzvp-rev2

# Write results to a custom directory
python run_case.py lih-rocksalt aiccm-hf --out results/
```

Every run writes a `<system>__<route>.json` file with the energy, per-atom
energy, convergence flag, walltime, and correlation breakdown (for post-HF
routes). Aggregate all results with:

```bash
python compare.py results/  # per-system x per-route single-line summary

# Cross-approach --vs deltas are disabled by D89. Use compare_b.py only for
# explicit not-defined approach status or the separately named real-Gamma control.
```

### 2. Fleet batch generation

Generate the full benchmark matrix (up to 100+ jobs) and submit to the
vibe-queue scheduler:

```bash
cd studies/aiccm-2026

# Review the job plan
python make_jobs.py

# Submit everything to the fleet
python make_jobs.py | sh

# Local-only tier-A jobs (no scheduler)
python make_jobs.py --local | sh

# Single system, single tier
python make_jobs.py --system mgo
python make_jobs.py --tier B

# Periodic references at the CRYSTAL23-matched basis
python make_jobs.py --crystal-basis | sh
```

Or submit individual cases directly:

```bash
vq submit compute-small  -d studies/aiccm-2026/ -- python run_case.py h-chain   aiccm-hf
vq submit compute-study  -d studies/aiccm-2026/ -- python run_case.py c-diamond aiccm-ks
vq submit compute-study  -d studies/aiccm-2026/ -- python run_case.py mgo       gdf
```

### 3. From the qc-input-library

The [`qc-input-library`](https://vibe-qc.com/docs/)
repo's `aiccm2026testset/` directory provides an independent input generation
script that produces a full matrix of `input.cmd` files per (system, basis,
route, SCF-sweep-variant), organised for both the Γ-CCM and χ-CCM streams:

```bash
cd ~/gitlab/qc-input-library/aiccm2026testset

# Generate all 3-D inputs for both streams
python generate_inputs.py

# Dry-run to review
python generate_inputs.py --dry-run

# Tier A only, stream a only
python generate_inputs.py --tier A --stream a

# SCF convergence stress-test variants
python generate_inputs.py --scf-sweep

# Include 1-D and 2-D systems
python generate_inputs.py --all-dims
```

Each generated directory contains an `input.cmd` file ready for submission.

### 4. Building a system from scratch - worked examples

Every example below is a complete, runnable script. Copy-paste, adjust the
geometry and basis, and run.

---

#### Example 1 - 1-D H₂ chain: the full HF → MP2 → CCSD(T) stack

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf, run_ccm_rhf_scalable
from vibeqc.periodic.ccm.dft import run_ccm_rks
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd

BOHR = 1.0 / 0.529177210903
# H₂ unit cell: a=4.0 bohr, 40 bohr vacuum in y,z
lat = np.diag([4.0 * BOHR, 40.0, 40.0])
system = PeriodicSystem(
    1, lat,
    [Atom(1, [0.0, 0, 0]), Atom(1, [2.0 * BOHR, 0, 0])],
)
# 8-cell chain along x
ccm = CCMSystem(system, (8, 1, 1), "sto-3g")

rhf = run_ccm_rhf(ccm, method="aiccm2026dev-a")
scalable = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a")
rks = run_ccm_rks(ccm, "pbe", method="aiccm2026dev-a")
mp2 = run_ccm_mp2(ccm, method="aiccm2026dev-a")
cc = run_ccm_ccsd(ccm, method="aiccm2026dev-a", compute_triples=True)

print(f"{'RHF':>12s} E/atom = {rhf.energy_per_atom:.8f} Ha")
print(f"{'RKS-PBE':>12s} E/atom = {rks.energy_per_atom:.8f} Ha")
print(f"{'MP2':>12s} Ecorr  = {mp2.e_correlation:.8f} Ha")
print(f"{'CCSD':>12s} Ecorr  = {cc.e_ccsd_correlation:.8f} Ha")
print(f"{'CCSD(T)':>12s} Ecorr  = {cc.e_correlation:.8f} Ha, (T)={cc.e_t:.2e}")
```

---

#### Example 2 - 3-D diamond: Γ-CCM and neutral controls side by side

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf_scalable
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rhf_rijcosx

BOHR = 1.0 / 0.529177210903
a = 3.5670 * BOHR  # diamond lattice constant
# FCC primitive cell: 2 carbon atoms
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = PeriodicSystem(
    3, lat,
    [Atom(6, [0, 0, 0]), Atom(6, [a/4, a/4, a/4])],
)
ccm = CCMSystem(system, (2, 2, 2), "sto-3g")

# Four-center (scalable C++, reaches genuine 3-D)
hf_4c = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a")
# Neutral fitted-torus GDF control (not Γ-CCM)
hf_ri = run_ccm_rhf_gdf(ccm)
# WSSC RI-J + chain-of-spheres exchange research approximation
hf_rijcosx = run_ccm_rhf_rijcosx(ccm)

print(f"{'4-center':>12s} E/atom = {hf_4c.energy_per_atom:.8f} Ha")
print(f"{'RI (GDF)':>12s} E/atom = {hf_ri.energy_per_atom:.8f} Ha")
print(f"{'RIJCOSX':>12s} E/atom = {hf_rijcosx.energy_per_atom:.8f} Ha")
```

---

#### Example 3 - 3-D MgO rocksalt: neutral fitted-torus control

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rks_gdf

BOHR = 1.0 / 0.529177210903
a = 4.22389871 * BOHR  # MgO lattice constant
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = PeriodicSystem(
    3, lat,
    [Atom(12, [0, 0, 0]),
     Atom(8,  (lat @ np.array([0.5, 0.5, 0.5])).tolist())],
)
ccm = CCMSystem(system, (2, 2, 2), "sto-3g")

# This evaluates a separately declared neutral fitted-torus control.
# It is not an RI substitution for the union-and-weight Γ-CCM construction.
# Compare only with complete operator and build fingerprints.
hf_ri = run_ccm_rhf_gdf(ccm)
ks_ri = run_ccm_rks_gdf(ccm, "pbe")
print(f"{'RHF/GDF':>12s} E/atom = {hf_ri.energy_per_atom:.8f} Ha")
print(f"{'RKS-PBE/GDF':>12s} E/atom = {ks_ri.energy_per_atom:.8f} Ha")
```

---

#### Example 4 - 2-D graphene: hexagonal lattice

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf_scalable

BOHR = 1.0 / 0.529177210903
a = 2.46 * BOHR
lat = np.array([[a, 0, 0], [-a/2, a * np.sqrt(3)/2, 0], [0, 0, 50.0]])
system = PeriodicSystem(
    2, lat,
    [Atom(6, [-a/3, a/3, 0]), Atom(6, [a/3, -a/3, 0])],
)
# 3×3 supercell = 18 carbon atoms
ccm = CCMSystem(system, (3, 3, 1), "sto-3g")

# Scalable C++ four-center driver: this 18-atom 2-D cluster already pads
# past the dense-ERI guard of the padded run_ccm_rhf validation path.
# Converges to E/atom = -38.96503317 Ha; ~10 min on a laptop.
rhf = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a")
print(f"{'RHF':>12s} E/atom = {rhf.energy_per_atom:.8f} Ha")
```

---

#### Example 5 - Neutral reference + DLPNO local correlation

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import (
    CCMSystem, ccm_dlpno_mp2, ccm_dlpno_ccsd,
    ccm_eri_neutral, ccm_neutral_cderi,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd

lat = np.diag([6.0, 15.0, 15.0])
system = PeriodicSystem(
    3, lat, [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])],
)
ccm = CCMSystem(system, (2, 1, 1), "sto-3g")

# Neutral reference: g_eff = Σ_P L⊗L (RI density-fit of v_E)
g_neutral = ccm_eri_neutral(ccm, ke_cutoff=120.0)
L = ccm_neutral_cderi(ccm, ke_cutoff=120.0)

# SCF on the neutral four-center
scf_neutral = run_ccm_rhf(ccm, eri=g_neutral)
print(f"Neutral RHF E/atom = {scf_neutral.energy_per_atom:.8f} Ha")

# Canonical neutral-control MP2 / CCSD(T) (correctness oracle)
mp2_can = run_ccm_mp2(ccm, scf_neutral, eri=g_neutral)
cc_can = run_ccm_ccsd(ccm, scf_neutral, eri=g_neutral)
print(f"Canonical MP2     Ecorr = {mp2_can.e_correlation:.8f} Ha")
print(f"Canonical CCSD(T) Ecorr = {cc_can.e_correlation:.8f} Ha")

# DLPNO at zero truncation == canonical (hard gate, verified to machine ε)
dlpno_mp2 = ccm_dlpno_mp2(ccm, scf_neutral, cderi=L, localize="pm", tcut_pno=0.0)
dlpno_cc = ccm_dlpno_ccsd(ccm, scf_neutral, cderi=L, g_neutral=g_neutral,
                          localize="pm", tcut_pno=0.0)
print(f"DLPNO-MP2  (tc=0)  Ecorr = {dlpno_mp2.e_corr:.8f}  "
      f"Δ vs can = {dlpno_mp2.e_corr - mp2_can.e_correlation:+.2e}")
print(f"DLPNO-CCSD (tc=0)  Ecorr = {dlpno_cc.e_correlation:.8f}  "
      f"Δ vs can = {dlpno_cc.e_correlation - cc_can.e_correlation:+.2e}")

# PNO truncation (small cluster → full-dim; larger clusters gain)
dlpno_mp2_pno = ccm_dlpno_mp2(ccm, scf_neutral, cderi=L, localize="pm", tcut_pno=1e-6)
print(f"DLPNO-MP2  (tc=1e-6) Ecorr = {dlpno_mp2_pno.e_corr:.8f}")

# Reduced-scaling per-pair-coupled DLPNO-CCSD(T): each pair its own PNO basis,
# cross-pair PNO-overlap coupling, per-triple-TNO (T), local coupling_radius.
from vibeqc.periodic.ccm.dlpno_ccsd_coupled import ccm_dlpno_ccsd_coupled
coupled = ccm_dlpno_ccsd_coupled(ccm, scf_neutral, cderi=L, localize="pm",
                                 tcut_pno=0.0, compute_triples=True)  # (T) via TNO
print(f"coupled DLPNO-CCSD(T) Ecorr = {coupled.e_correlation:.8f}  "
      f"Δ vs can = {coupled.e_correlation - cc_can.e_correlation:+.2e}")
```

---

#### Example 6 - Open-shell neutral control: UHF + UCCSD(T)

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.uhf import run_ccm_uhf
from vibeqc.periodic.ccm.uccsd import run_ccm_uccsd
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi

lat = np.diag([8.0, 8.0, 8.0])
# Odd-electron neutral 3-D control: one Li atom (doublet)
system = PeriodicSystem(
    3, lat, [Atom(3, [0, 0, 0])], charge=0, multiplicity=2,
)
ccm = CCMSystem(system, (1, 1, 1), "sto-3g")

# UHF and UCCSD(T) on one matching neutral fitted-torus factor
L = ccm_neutral_cderi(ccm)
uhf = run_ccm_uhf(ccm, cderi=L)
print(f"UHF E/cell = {uhf.energy:.8f} Ha")

# Neutral-control UCCSD(T) - spin-resolved, closed-shell consistency verified
ucc = run_ccm_uccsd(ccm, uhf, cderi=L)
print(f"UCCSD(T) Ecorr = {ucc.e_correlation:.8f} Ha, (T) = {ucc.e_t:.2e}")
```

---

#### Example 7 - Properties: gap, charges, dipole, numerical gradient

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import (
    CCMSystem, ccm_homo_lumo_gap, ccm_mulliken_charges,
    ccm_lowdin_charges, ccm_dipole, ccm_numerical_gradient,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf

BOHR = 1.0 / 0.529177210903
# LiH chain: heteronuclear, good test for charge analysis
a = 3.2 * BOHR
lat = np.diag([a, 40.0, 40.0])
system = PeriodicSystem(
    1, lat,
    [Atom(3, [0, 0, 0]), Atom(1, [a/2, 0, 0])],
)
ccm = CCMSystem(system, (6, 1, 1), "sto-3g")
scf = run_ccm_rhf(ccm, method="aiccm2026dev-a")

# Fundamental gap
homo, lumo, gap = ccm_homo_lumo_gap(scf, ccm)
print(f"Gap: HOMO={homo:.4f}  LUMO={lumo:.4f}  Δ={gap:.4f} Ha")

# Per-cell Mulliken charges with translational-spread diagnostic
mulliken = ccm_mulliken_charges(scf, ccm)
lowdin = ccm_lowdin_charges(scf, ccm)
for i, (mq, lq) in enumerate(zip(mulliken.per_cell_charges,
                                   lowdin.per_cell_charges)):
    print(f"Atom {i}: Mulliken={mq:+.4f}  Löwdin={lq:+.4f}")
print(f"Mulliken translational spread = {mulliken.translational_spread:.4f}")

# Finite-cluster dipole (symmetry indicator for non-wrapping clusters)
mu = ccm_dipole(scf, ccm)
print(f"Dipole = ({mu.x:.4f}, {mu.y:.4f}, {mu.z:.4f}) e·bohr")

# Numerical gradient (finite-difference forces, Σ forces ≈ 0)
grad = ccm_numerical_gradient(ccm, method="aiccm2026dev-a")
for i, g in enumerate(grad.per_atom_gradient):
    print(f"Atom {i} force = ({g[0]:+.6f}, {g[1]:+.6f}, {g[2]:+.6f}) Ha/bohr")
print(f"Σ|F| = {np.linalg.norm(np.sum(grad.per_atom_gradient, axis=0)):.2e}")
```

---

#### Example 8 - Wannier localization + space-group symmetry

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import (
    CCMSystem, localise_ccm, localization_density_residual,
    analyze_ccm_symmetry, ccm_symmetry_invariance_residuals,
    ccm_symmetry_unique_atom_pairs, ccm_neutral_cderi,
    ccm_neutral_cderi_fold,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf_scalable

BOHR = 1.0 / 0.529177210903
a = 3.5670 * BOHR
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = PeriodicSystem(
    3, lat,
    [Atom(6, [0, 0, 0]), Atom(6, [a/4, a/4, a/4])],
)
ccm = CCMSystem(system, (2, 2, 2), "sto-3g")

# The scalable C++ four-center driver (the padded-ERI run_ccm_rhf is a
# small / 1-D validation path and its size guard rejects 3-D clusters).
# Converges to E/atom = -40.81651008 Ha; ~20 min on a laptop.
scf = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a")

# Wannier localization (Pipek-Mezey)
loc = localise_ccm(scf, ccm, method="pipek-mezey")
print(f"Objective: initial={loc.objective_initial:.6f}  final={loc.objective_final:.6f}")
print(f"Density residual (unitary-invariance gate) = "
      f"{localization_density_residual(scf, loc):.2e}")
# Note: a negative spread flags an orbital whose tail wraps the BvK torus
# (the position-operator descriptors are faithful only for non-wrapping
# orbitals; see localization_aliasing for the per-orbital diagnostic).
for i, (c, s) in enumerate(zip(loc.centers, loc.spreads)):
    print(f"  Wannier {i}: center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  "
          f"spread={s:.4f} bohr²")

# Space-group symmetry (spglib)
sym = analyze_ccm_symmetry(ccm)
print(f"Space group: {sym.number} ({sym.international_symbol})")
print(f"Cluster-invariant subgroup: {sym.cluster_order} of "
      f"{sym.crystal_order} crystal ops survive on this cluster")

res = ccm_symmetry_invariance_residuals(ccm, sym)   # dict, over res['n_ops'] ops
print(f"S-invariance residual: {res['overlap']:.2e}")   # ~0 (machine precision)
print(f"T-invariance residual: {res['kinetic']:.2e}")   # ~0 (machine precision)
print(f"V-invariance residual: {res['nuclear']:.2e}")   # non-zero: the bare union
                                                        # V breaks crystal symmetry

# Petite-list bookkeeping: symmetry-unique home atom pairs
pairs = ccm_symmetry_unique_atom_pairs(ccm, sym)
print(f"Unique atom pairs: {pairs.n_unique}/{pairs.n_total}  "
      f"(reduction factor: {pairs.reduction_factor:.1f}×)")

# Neutral (Ewald) GDF cderi: the RI-consistent three-center of the CCM, via
# the fold builder (per-q unit-cell fits folded onto the torus; the same
# object ccm_neutral_cderi builds, without its supercell-FFT memory wall).
L = ccm_neutral_cderi_fold(ccm)   # (1904, 80, 80) on this cluster

# Space-group pair-reduced supercell fit: only the orbit-representative
# atom-pair AO blocks of the petite list are fit, the rest are
# reconstructed by the group action (reduced-vs-unreduced SCF energy diff
# 5.5e-13 Ha at production defaults; pinned parity gates in
# tests/test_ccm_symmetry.py). Supercell-FFT-build path -- shown here on
# the primitive-cell cluster (seconds; 2x pair reduction on diamond); on
# large clusters prefer the memory-lean fold above until the fold's own
# symmetry reduction lands.
ccm1 = CCMSystem(system, (1, 1, 1), "sto-3g")
L_sym = ccm_neutral_cderi(ccm1, symmetry=analyze_ccm_symmetry(ccm1))
```

---

#### Example 9 - Interaction-radius convergence scan

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import ccm_interaction_range_scan

BOHR = 1.0 / 0.529177210903
lat = np.diag([4.0 * BOHR, 40.0, 40.0])
system = PeriodicSystem(
    1, lat,
    [Atom(1, [0, 0, 0]), Atom(1, [2.0 * BOHR, 0, 0])],
)

# The CCM analogue of a k-point convergence study
radii = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0, 30.0, 36.0]
scan = ccm_interaction_range_scan(system, "sto-3g", radii, method="aiccm2026dev-a")

print(f"{'R_c':>5s} {'nrep':>10s} {'n_at':>5s} {'E/atom':>14s} {'ΔE':>12s}")
prev = None
for r in scan:
    de = f"{r.energy_per_atom - prev:+.2e}" if prev is not None else "    --    "
    prev = r.energy_per_atom
    print(f"{r.radius_bohr:5.1f} {str(r.nrep):>10s} {r.n_atoms:5d}  "
          f"{r.energy_per_atom:14.8f} {de:>12s}")

print(f"\nConverged to 0.1 mHa/atom at R_c = {scan.converged_radius(1e-4)} bohr")
print(f"Converged to 10 µHa/atom at R_c = {scan.converged_radius(1e-5)} bohr")
```

---

#### Example 10 - Building a custom crystal from arbitrary lattice vectors

```python
import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem, nrep_for_interaction_range
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

BOHR = 1.0 / 0.529177210903

# Tetragonal TiO₂ rutile (P4₂/mnm), 6 atoms/cell
a, c = 4.5940 * BOHR, 2.9590 * BOHR
u = 0.305  # internal parameter
lat = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
frac = [(22, 0, 0, 0), (22, 0.5, 0.5, 0.5),
        (8, u, u, 0), (8, -u, -u, 0),
        (8, 0.5+u, 0.5-u, 0.5), (8, 0.5-u, 0.5+u, 0.5)]
atoms = [Atom(Z, (f1 * lat[0] + f2 * lat[1] + f3 * lat[2]).tolist())
         for Z, f1, f2, f3 in frac]
system = PeriodicSystem(3, lat, atoms)

# Size the cluster from a real-space interaction radius
ccm = CCMSystem.from_interaction_range(system, 10.0, "sto-3g")
print(f"TiO₂ rutile, derived nrep = {ccm.nrep}, {ccm.n_atoms} atoms")
# -> derived nrep = (3, 3, 4), 216 atoms

# SCF on this 216-atom all-electron cluster is a workstation job, not a
# laptop demo: the padded four-center run_ccm_rhf is rejected by the
# dense-ERI guard, and the production multi-k GDF route has to fit Ti's
# steep core functions on a large FFT grid (overnight-scale; 7+ h on an
# Apple-silicon laptop without completing). Uncomment on a workstation:
# rhf = run_ccm_rhf_gdf(ccm)
# print(f"RHF/GDF E/atom = {rhf.energy_per_atom:.8f} Ha")
```

---

#### Example 11 - Crystal from space-group symmetry (ASE bridge)

```python
import json, os, numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf
from ase.spacegroup import crystal as ase_crystal

BOHR = 1.0 / 0.529177210903

# AlN wurtzite (P6₃mc) - built from a grounded space-group spec
# (Specs exported from the CRYSTAL .d12 sources, in crystal_specs.json)
spec_path = os.path.join("studies/aiccm-2026", "crystal_specs.json")
with open(spec_path) as fh:
    specs = json.load(fh)
s = specs["aln-wurtzite"]

at = ase_crystal(
    symbols=[a[0] for a in s["asym"]],
    basis=[a[1:] for a in s["asym"]],
    spacegroup=int(s["sg"]),
    cellpar=list(s["cellpar"]),
    primitive_cell=True,
)
lat = np.asarray(at.cell.array, dtype=float).T * BOHR
atoms = [Atom(int(z), (np.asarray(p, dtype=float) * BOHR).tolist())
         for z, p in zip(at.numbers, at.positions)]
system = PeriodicSystem(3, lat, atoms)

ccm = CCMSystem(system, (2, 2, 2), "sto-3g")

# 32-atom 3-D cluster: the padded union-and-weight four-center run is out of
# reach of this dense example. The following is a neutral-torus GDF control,
# not a Γ-CCM construction result.
rf = run_ccm_rhf_gdf(ccm)
print(f"AlN wurtzite neutral-control RHF/GDF E/atom = {rf.energy_per_atom:.8f} Ha")
```

### All available routes

Every route is a single `python run_case.py <system> <route>` invocation.
The full `aiccm2026dev-a` feature matrix:

| route | what runs | example |
|---|---|---|
| `aiccm-hf` | HF, four-center cyclic cluster (no RI) | `python run_case.py h-chain aiccm-hf` |
| `aiccm-ks` | KS, four-center (`--functional`, default `pbe`) | `python run_case.py c-diamond aiccm-ks --functional pbe0` |
| `aiccm-ri` | legacy harness name for the neutral-torus multi-k GDF control; not Γ-CCM construction evidence | `python run_case.py mgo aiccm-ri` |
| `aiccm-ks-ri` | KS neutral-torus GDF control | `python run_case.py lih-rocksalt aiccm-ks-ri --functional pbe` |
| `aiccm-rijcosx` | HF with RIJCOSX (WSSC RI-J + chain-of-spheres K) | `python run_case.py graphene aiccm-rijcosx` |
| `aiccm-mp2` | MP2 correlation (UMP2 for open-shell) | `python run_case.py h-chain aiccm-mp2` |
| `aiccm-ccsd` | CCSD(T) correlation (closed-shell) | `python run_case.py h-chain aiccm-ccsd` |
| `aiccm-dlpno-mp2` | DLPNO-MP2 local correlation on the neutral reference | `python run_case.py h-chain aiccm-dlpno-mp2 --tcut-pno 1e-7` |
| `aiccm-dlpno-ccsd` | DLPNO-CCSD(T) (union-PNO) on the neutral reference | `python run_case.py h-chain aiccm-dlpno-ccsd` |
| `aiccm-uccsd` | Open-shell UCCSD(T) on the UHF-CCM neutral reference | `python run_case.py h-chain aiccm-uccsd` |
| `aiccm-properties` | Gap, Mulliken/Löwdin charges, dipole, forces + vibe-view `.qvf` | `python run_case.py c-diamond aiccm-properties` |
| `aiccm-viz` | HF + vibe-view: crystalline + Wannier orbitals (`.qvf`, `.cube`, `.molden`) | `python run_case.py c-diamond aiccm-viz` |
| `aiccm-localize` | **Demo:** Wannier localization (Pipek-Mezey) - density residual, centers, spreads | `python run_case.py h-chain aiccm-localize` |
| `aiccm-symmetry` | **Demo:** space-group symmetry - group, pair reduction, overlap/kinetic invariance | `python run_case.py c-diamond aiccm-symmetry` |
| `aiccm-pao` | **Demo:** DLPNO projected atomic orbitals - `n_pao`, occ-orthogonality | `python run_case.py h-chain aiccm-pao` |
| `bipole` | vibe-qc periodic reference (BIPOLE Ewald J/K) | `python run_case.py c-diamond bipole` |
| `gdf` | vibe-qc periodic reference (Gaussian density fitting) | `python run_case.py mgo gdf` |

**Sizing overrides** (any route): `--nrep i j k` (explicit), `--interaction-range R_c`
(bohr), `--interaction-range-ang R_c` (Å). **Periodic references**: `--kmesh i j k`
overrides the default MP mesh for `bipole`/`gdf` routes.

**Pre-flight check** - verify every system's cluster-overlap conditioning before
submitting the batch:

```bash
python testset.py --check
```

This builds each cluster at its benchmark `nrep` and reports the minimum
eigenvalue of `S^CCM`. The batch policy deliberately requires a positive,
full-rank overlap at the declared benchmark mesh; all 28 systems pass with
`min eig > 1e-3`. That is a campaign-admission rule, not a statement that every
CCM SCF must refuse a screenable direction. The SCF drivers canonically remove
overlap modes at or below `lindep_tol` and fail closed only when the retained
space cannot hold the occupied orbitals. The explicit
`CCMSystem.check_overlap_spectrum()` diagnostic remains strict and distinguishes
a materially indefinite finite-Γ overlap from numerical non-positivity and a
positive near-singularity.

## Construction routes and neutral-torus controls

* **Γ-CCM four-center** - the WSSC union-and-weight Coulomb/exchange. HF (`run_ccm_rhf`,
  and the scalable C++ `run_ccm_rhf_scalable` that reaches genuine 3-D) and KS
  (`run_ccm_rks`/`run_ccm_uks`, pure and hybrid libxc functionals) reproduce the
  molecular kernels exactly in the isolated limit. The same low-level RKS/UKS
  entry points accept registered pure full-grid providers on the periodic
  reference-cluster grid; external-provider hybrids fail closed. The public
  `run_periodic_job(method="aiccm", variant="four-center")` arm supports 3-D
  RHF/RKS/UHF/UKS and always selects the symmetric `aiccm2026dev-a` weighting;
  the legacy `jk_method="aiccm2026dev-a"` spelling reaches the same arm with a
  `DeprecationWarning`.
* **Neutral fitted-torus controls** - `run_ccm_rhf_gdf` /
  `run_ccm_rks_gdf` evaluate a specified neutral fitted-torus operator on its
  character/Bloch mesh. The **RI-consistent three-center** for that control is
  the neutral cderi `ccm_neutral_cderi`
  (`ccm_ri_tensors_neutral` / `ccm_ri_j_neutral` / `ccm_ri_k_neutral`): its RI-J/K
  reproduce the neutral four-center to machine ε. A **space-group pair
  reduction** of that fit (`ccm_neutral_cderi(ccm, symmetry=sym)`, routing
  through `ccm_symmetry_unique_atom_pairs` and the GDF builder's
  `fit_pair_list` seam) fits only orbit-representative atom-pair AO blocks
  and reconstructs the rest by the group action; reduced-vs-unreduced SCF
  energy diff 5.5e-13 Ha at production defaults, tensor parity < 1e-9
  cell-list-converged (gates in `tests/test_ccm_symmetry.py`; landed
  2026-07-10, see Example 8). The research route
  `run_ccm_rhf_rij` keeps the explicit eq-13 union three-center weighting on the
  bare-1/r integrals; it is exact in the isolated limit but ~few-% periodically -
  the bare-1/r four-center is *non-separable*, so no weighting of its 3-center is
  RI-consistent (§13.7). Use the neutral cderi (or `run_ccm_rhf_gdf`) only when
  the neutral fitted-torus control is the intended operator; this does not
  replace or validate the union-and-weight Γ-CCM construction.
* **Real-Gamma neutral control** - `run_ccm_rhf_direct`: the same neutral-kernel
  RI energy as `run_ccm_rhf_gdf`, computed **without any k-point machinery** -
  one real Γ-supercell SCF (a single real generalized eigenproblem over the
  `N_c·n_μ` supercell AOs). The multi-k driver's unit-cell lattice sums are
  folded onto the torus, RI-J/K contract the neutral cderi `L` in pure real
  algebra, and the exchange-`q=0` (BvK-Madelung, `exxdiv="ewald"`) seam is a
  derived Fock operator `K → K + ξ_N·S·D·S` (so `F = ∂E/∂D` holds; a
  deliberate `exxdiv=None` run sits above by exactly `ξ_N·N_e/2` per
  supercell). Parity with `run_ccm_rhf_gdf`: 1.7e-11 Ha/cell on the H₂
  anchor (external KRHF matched to 1.6e-11), ≤1e-8 on 1-D chains (TRIM and
  non-TRIM meshes) and ionic LiH. The result records `.exchange_q0`; compare
  routes only at matched conventions (`assert_matched_exchange_q0`). This
  character/real-Gamma parity is an internal same-Hamiltonian Fourier control,
  not Γ-CCM/χ-CCM approach evidence.
  The default `L` builder is the **fold** (`ccm_neutral_cderi_fold`): per-q
  *unit-cell* fits folded onto the torus - MB-scale memory (the one-shot
  supercell FFT fit needs GBs and is kept as `cderi_build="supercell"` for
  validation) and exact GDF parity even on ultra-diffuse bases (LiH (2,1,1):
  4e-14 vs the supercell fit's 1.35e-4 residual).
  On **tight-core** cells the parity above needs the RSGDF high-`|G|` tail,
  and both routes now resolve it the same way (`tail_ke_cutoff`, defaulting
  through `ccm_neutral_tail_ke_cutoff`). Until 2026-08-27 only the multi-k
  route could auto-size one (at `nrep=(1,1,1)` it delegates to the Γ fast
  path, which does), while the direct route had no plumbing for it, so the two
  ran different Hamiltonians and disagreed by **-4.99e-01 Ha/cell** on
  MgO/STO-3G (GitLab issue #307; the tailed side is the accurate one, 0.3 mHa
  from the PySCF MDF reference). Both result types now record the tail they
  actually applied as `.rsgdf_tail_ke_cutoff`, so a route comparison can check
  it the way it already checks `.exchange_q0`. At `N_c > 1` neither route
  auto-tails: the multi-k loop holds that class for parity (issue #146), and
  the direct route follows it rather than moving alone.
  The accelerated sibling `run_ccm_rhf_direct_rijcosx` swaps the exact
  RI-K for the periodic chain-of-spheres exchange (the validated M3b
  SR+LR `KPointCosxK` engine at Γ on the supercell) with the same seam -
  measured 22-30 µHa/cell from the exact route on genuine-3-D controls
  (the shared engine's composition floor). dim=3 only (the LR complement
  cannot compose on vacuum-collapsed meshes) and compact-basis only (the
  SR envelope criterion); both exclusions raise/warn loudly.
* **Γ-CCM RIJCOSX research route** - `run_ccm_rhf_rijcosx`: the WSSC RI-J
  Coulomb plus chain-of-spheres
  (COSX) seminumerical exchange (Neese 2009) on a supercell grid. Exchange is
  short-ranged (decays within the WSC), so the seminumerical K is a natural fit;
  the isolated limit reproduces molecular RIJCOSX to ~5 µHa.

## Local correlation: DLPNO-MP2 / DLPNO-CCSD(T)

Domain-based local pair natural orbitals (DLPNO) on the cyclic cluster, reusing
vibe-qc's validated molecular DLPNO machinery with the cyclic overlap `S^CCM` and
the **neutral** four-center reference:

```python
from vibeqc.periodic.ccm import CCMSystem, ccm_dlpno_mp2, ccm_dlpno_ccsd
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi

g = ccm_eri_neutral(ccm)              # neutral four-center g_eff = Σ_P L⊗L
L = ccm_neutral_cderi(ccm)            # its RI cderi
scf = run_ccm_rhf(ccm, eri=g)         # neutral SCF reference

mp2 = ccm_dlpno_mp2(ccm, scf, cderi=L, localize="pm", tcut_pno=1e-7)
cc  = ccm_dlpno_ccsd(ccm, scf, cderi=L, g_neutral=g, localize="pm", tcut_pno=1e-7)
print(mp2.e_corr, cc.e_correlation, cc.e_t)
```

* `ccm_dlpno_mp2` - pair domains (Mulliken `tcut_mkn`) → semicanonical PAOs →
  PNOs (occupation threshold `tcut_pno`) → coupled LMP2; canonical *or* localized
  (Pipek-Mezey / Boys) occupieds.
* `ccm_dlpno_ccsd` - subspace-projected DLPNO-CCSD(T): the per-pair PNOs are merged
  into a union virtual space and vibe-qc's CCM CCSD engine runs in `{occ, V_trunc}`.

**The reference is neutral, by necessity.** The bare-1/r Madelung background is
block-diagonal in occ/virt and shifts the orbital-energy *denominators*, so
independent bare-vs-neutral SCF→correlation do **not** agree (it is a
fixed-reference diagnostic, not a Madelung-invariance theorem). These DLPNO
controls therefore run on the neutral four-center / GDF reference. The hard gate: at no
truncation (`tcut_pno = tcut_mkn = 0`) DLPNO-MP2 and DLPNO-CCSD(T) reproduce the
canonical neutral-control MP2 / CCSD(T) on the *same* neutral four-center to machine ε
(`tests/test_ccm_dlpno_{mp2,ccsd}.py`).

## Open-shell correlation (UHF reference)

`run_ccm_ump2(method="aiccm2026dev-a")` is the union-and-weight Γ-CCM UMP2
route. Passing `cderi=L` instead selects the separately declared neutral
fitted-torus control. `run_ccm_uccsd` is currently neutral-control-only: it
drives vibe-qc's spin-orbital UCCSD(T) engine with the matching neutral cderi
transformed into the α/β MO bases. On a closed-shell cluster it reproduces
`run_ccm_ccsd(..., cderi=L)` to ≤1e-8 (the αα = ββ collapse), and a doublet
cluster converges with negative correlation (`tests/test_ccm_uccsd.py`). Its
API ancestry does not assign Γ-CCM or χ-CCM construction identity.

## Properties

One-particle properties on a converged CCM SCF (`vibeqc.periodic.ccm.properties`):

```python
from vibeqc.periodic.ccm import (ccm_homo_lumo_gap, ccm_mulliken_charges,
                                 ccm_lowdin_charges, ccm_dipole, ccm_numerical_gradient)
gap = ccm_homo_lumo_gap(scf, ccm)              # band/fundamental gap (RHF or UHF)
q   = ccm_mulliken_charges(scf, ccm)           # per-cell charges + transl. spread
mu  = ccm_dipole(scf, ccm)                     # finite-cluster dipole (see caveat)
grad = ccm_numerical_gradient(ccm)             # central-difference dE/dR; force = −grad
```

* **gap** - `ε_LUMO − ε_HOMO` over the supercell-Γ spectrum (= the gap at the
  k-points the cluster folds to Γ; spin-resolved for UHF/UKS). Well defined.
* **populations** - Mulliken / Löwdin with `S^CCM`; per-cell image-averaged
  charges, with `translational_spread` as a cyclic-symmetry diagnostic. Well
  defined.
* **dipole** - the *finite-cluster* electric dipole. ⚠ **not** the bulk
  polarization: the position operator is ill-defined on a torus (Resta 1998) and
  plain `⟨μ|r|ν⟩` wraps for boundary-crossing orbitals; faithful only for
  non-wrapping (dilute) clusters, and a symmetry indicator (≈ 0 for
  centrosymmetric) otherwise.
* **forces** - central-difference `dE/dR` per unit-cell atom (the displacement
  propagates to every image). It satisfies translational invariance (Σ forces ≈ 0)
  and reproduces vibe-qc's molecular analytic RHF gradient in the isolated limit
  to ~1e-7. It remains the gate the analytic gradient is validated against.
  The analytic WSSC-weighted gradient is **implemented**
  (`vibeqc.periodic.ccm.gradient_analytic`: `run_ccm_{rhf,uhf,rks,uks}_gradient`,
  `run_ccm_{mp2,ump2}_gradient`, `run_ccm_{ccsd,uccsd}_gradient` incl. (T)),
  agreeing with this finite-difference gate at 5e-6-class on the pinned
  chain fixtures; its four-center term is a dense `n_pad_ao⁴` path, so it
  is a small / 1-D / 2-D validation route until a scalable weighted
  ERI-gradient kernel lands. The Resta/Berry-phase bulk polarization is
  still a roadmap item.

See `docs/aiccm2026dev_a_followon.md` §§ 2, C, D for the derivations and gates.

## Union-and-weight versus neutral-torus control gap

The union-and-weight WSSC four-center and the neutral fitted-torus control can
differ by a Madelung-scale shift, especially for ionic 3-D crystals. That
difference compares two explicitly different finite constructions/operators;
it must not be erased by calling the neutral route the production Γ-CCM path.
The neutral four-center
`g_eff = (μν|v_E|ρσ)` is [`ccm_eri_neutral`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/periodic/ccm/neutral.py)
- the RI density-fit of `v_E` (§13.7), built from the periodic-Gamma GDF cderi
on the cluster supercell and exactly 8-fold symmetric. It is useful as a
neutral-torus control; it is not the union-and-weight construction by name.

The gap turns out to be a **pure mean-field (Hartree) background**. In the
molecular limit the neutral and bare four-centers differ by a rank-1 term,

```
g_eff[μν,ρσ] = (μν|ρσ)_bare − ξ · S_μν · S_ρσ ,   ξ = madelung_constant_for_cell(cluster)
```

because the AO-pair "charge" is `q_μν = ∫φ_μφ_ν = S_μν`. Two consequences settle
the question:

1. **HF / KS energy** - `ξ·S⊗S` is `∝ S` in the Fock (a uniform shift).
   `run_ccm_rhf_gdf` / `run_ccm_rks_gdf` evaluate the neutral fitted-torus
   control. The union-and-weight Γ-CCM result must be reported under its own
   construction identity; agreement or disagreement with this control is
   evidence to analyze, not a route substitution.
2. **Correlation (fixed reference only)** - the background is block-diagonal in
   the occ/virt split (`J`: `−ξN_e S`, `K`: `−ξ SDS`): it leaves the HF *orbitals*
   invariant but shifts the occ-virt *gap* by `O(ξ)`. In the MO basis
   `ξ·(CᵀSC)⊗(CᵀSC) = ξ·I⊗I` has no occupied-virtual element, so at a **fixed**
   reference it does not enter the MP2/CCSD *numerator*. It **does** shift every
   *denominator*, so **independent bare-vs-neutral SCF→correlation runs do NOT
   agree** - the neutral reference is required for ionic correlation. (This
   corrects an earlier overclaim that the bare four-center is Madelung-robust for
   correlation; the rank-1 result is a *fixed-reference diagnostic*, not an
   all-post-HF theorem - see the `aiccm2026dev-a` paper §7.)

For a neutral fitted-torus control study, take both the HF/KS energy and
correlation from the neutral reference so the finite operator is consistent.
Do not publish that result as Γ-CCM. `ccm_eri_neutral` is the small-cluster
analysis tool that makes the decomposition explicit; it can OOM on dense ionic
3-D systems, like the padded union-and-weight four-center. Quantitative ionic
Γ-CCM claims require construction-specific convergence and external validation.

## Tutorial: H4 chain, full hierarchy

The runnable demo is
[`examples/periodic/aiccm2026dev_a_demo.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_a_demo.py):

```console
$ python examples/periodic/aiccm2026dev_a_demo.py
=== 8-fold ERI symmetry (max permutation violation) ===
  1D H4 chain (4,1,1) : union12 1.63e-03   aiccm2026dev-a 0.00e+00
  2D hex      (2,2,1) : union12 2.12e-02   aiccm2026dev-a 3.10e-18
  2D oblique  (2,2,1) : union12 1.94e-02   aiccm2026dev-a 3.10e-18
=== Full stack on the 1D H4 chain (4,1,1), aiccm2026dev-a ===
  RHF     E/atom = -0.542676   (gold -0.542875)
  MP2     Ecorr  = -0.129514   /cell
  CCSD    Ecorr  = -0.190443   /cell
  CCSD(T) Ecorr  = -0.191100   /cell  [(T)=-6.57e-04]
=== 2D hexagonal lattice (2,2,1): aiccm2026dev-a vs union12 ===
  RHF E/atom : union12 -0.399044   aiccm2026dev-a -0.400456   (Δ=-1.413 mHa/atom)
```

Steps the demo walks through: (1) build the symmetric four-center and confirm it
is exactly 8-fold symmetric on three lattices while `union12` is not; (2) run the
HF → MP2 → CCSD(T) stack on the H4 chain (HF converges to the gold -0.542875
Ha/atom with cluster size); (3) show the non-orthorhombic ≥2-D divergence from
`union12`. For a separate neutral-control study, run
`run_ccm_rhf_gdf` and the relevant RIJCOSX research route under their own
machine identities. Do not present that side-by-side study as a 3-center
hierarchy of the Γ-CCM construction.

## Validation status

| piece | status |
|---|---|
| symmetric four-center | exact 8-fold symmetry (≤3e-18), 1-D / 2-D-hex / 2-D-oblique / 3-D |
| HF (4c), C++ scalable | == padded to µHa; 1-D → gold -0.542875; reaches 3-D |
| KS (4c), RKS/UKS | == molecular `run_rks`/`run_uks` (PBE/PBE0/B3LYP) to <1e-6 |
| MP2 / UMP2 | == molecular `run_mp2` / `run_ump2` (isolated) |
| CCSD(T) | == molecular DF-CCSD to the DF gap; (T) to ~1e-8 |
| neutral GDF control | molecular-limit side-by-side residual can reach the RI fitting scale; no global Γ-CCM equality claim |
| RIJCOSX | == molecular RIJCOSX (isolated) to ~5 µHa |
| neutral four-center (`ccm_eri_neutral`) | 8-fold symmetric; `= bare − ξ·S⊗S`, ξ = cell Madelung const |
| Madelung background (fixed ref) | block-diagonal in occ/virt → no correlation *numerator* term; shifts denominators, so independent bare-vs-neutral SCF differ (neutral ref required) |
| RI-consistent 3-center (`ccm_neutral_cderi`) | neutral RI-J/K == four-center to machine ε; bare union RI-J ~1e-3 (non-separable) |

| real-Γ control (`run_ccm_*_direct`) | SCF matrix RHF/UHF/RKS/UKS; == multi-k GDF at a matched `exxdiv` (1.7e-11 Ha/cell on the H₂ anchor, external PySCF KRHF to 1.6e-11); screened HSE-class hybrids exxdiv-independent by construction; double hybrids composed; analytic forces 5.6e-9 vs FD of the direct energy, fixed-lattice relaxation; DLPNO MP2/UMP2/CCSD(T) machine-exact vs canonical at zero truncation |

Tests: `tests/test_ccm_aiccm2026dev_a.py`, `tests/test_ccm_dft.py`,
`tests/test_ccm_ri.py`, `tests/test_ccm_neutral.py`. The real-Γ control
row is gated separately by `tests/test_ccm_direct.py`,
`test_ccm_rks_direct.py`, `test_ccm_uks_direct.py`,
`test_ccm_rsh_direct.py`, `test_ccm_double_hybrid_direct.py`,
`test_ccm_direct_gradient.py`, `test_ccm_direct_dlpno.py` and
`test_periodic_ccm_real_gamma_adapter.py` (the runner adapter).

**Madelung-scale gap - characterized for the compared operators** (see the
union-and-weight versus neutral-torus section above): the
ionic over-binding is, to leading order, a mean-field background `ξ·S⊗S`.
For a neutral-torus study, both the absolute HF/KS energy and correlation must
come from the same neutral reference: the background shifts the occ-virt gap,
so it leaves the correlation numerator unchanged only at a fixed reference.
This consistency rule does not relabel the neutral reference as Γ-CCM. The
neutral four-center
`ccm_eri_neutral` (the §13.7 density-fit of `v_E`) makes the leading-order
decomposition explicit; the non-rank-1 remainder is an open quantification.

**Neutral-control RI consistency - resolved** (§13.7): the RI-consistent three-center is the
neutral cderi `ccm_neutral_cderi` (its RI-J/K reproduce the four-center to
machine ε). The bare-1/r union RI-J (`run_ccm_rhf_rij`) is ~few-% because the
bare-1/r four-center is non-separable; the GDF route is RI-consistent for its
neutral fitted-torus operator, not for the union-and-weight construction.

**Open items** (research): a scalable dense-3-D neutral four-center (both
`ccm_eri_neutral`/`ccm_neutral_cderi` and the bare padded four-center are
small-cluster). The scalable GDF route remains a neutral-torus control and must
not be presented as the Γ-CCM production energy path.

Theory: `AICCM_ALGORITHM.md` §13. Reference: Peintinger & Bredow,
*J. Comput. Chem.* **35**, 839 (2014), doi:10.1002/jcc.23550.
