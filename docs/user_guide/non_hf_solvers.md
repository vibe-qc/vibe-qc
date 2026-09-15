(non-hf-solvers)=

# Non-mean-field solvers (`vibeqc.solvers`)

Developer documentation. Available since v0.9.0 (`vibeqc.solvers`).

## Overview

The `vibeqc.solvers` package provides wavefunction-based
electronic-structure methods that do **not** conceptually rely on a
Hartree-Fock reference state.  All solvers share a common interface:

```python
result = solver.solve(hamiltonian, options) -> SolverResult
```

where `hamiltonian` is a `Hamiltonian` object carrying one- and
two-electron integrals in an orthonormal spatial-orbital basis.
The orbital source (HF, canonical orthogonalisation, localised
orbitals, …) is a *separate concern* handled by the orbital-provider
layer.

## Architecture

```
vibeqc.solvers
├── _common.py          # Hamiltonian, SolverOptions, SolverResult
├── _hamiltonian.py     # AO → MO integral construction
├── _determinant.py     # Determinant / CSF data structures
├── _slater_condon.py   # Slater-Condon matrix-element rules
├── _selected_ci.py     # CIPSI-style selected-CI solver
├── _dmrg.py            # Two-site DMRG / MPS solver
├── _v2rdm.py           # Exact small-active-space 1/2-RDM solver
└── _transcorrelated.py # Similarity-transformed Hamiltonian
```

### Layers

1. **Integrals / orbital basis / Hamiltonian construction**
   (`_hamiltonian.py`): Builds `Hamiltonian` objects from vibe-qc's
   native C++ integral bindings.  Supports AO → MO transformation
   and canonical (Löwdin) orthogonalisation.  The `get_hf_orbital_provider`
   helper is a *convenience*, not a requirement.

2. **Many-body objects** (`_determinant.py`, `_slater_condon.py`):
   Determinant representations as tuples of occupied indices,
   excitation-rank logic, Slater-Condon matrix elements (diagonal,
   single, double), and full Hamiltonian-matrix construction.

3. **Solver backends** (`_selected_ci.py`, `_dmrg.py`, `_v2rdm.py`):
   Each solver implements `solve(Hamiltonian, Options) → SolverResult`.

4. **Diagnostics and benchmarking**: The `SolverResult` object carries
   method-specific diagnostics (PT2 corrections, truncation errors,
   constraint residuals, RDM objects) alongside energy traces.

5. **Public API / CLI exposure**: `vibeqc.solvers.__init__` exports all
   public classes and convenience functions.  The package is importable
   as `from vibeqc import solvers` or `from vibeqc.solvers import …`.

## `run_job` integration

The wavefunction solvers are reachable both as a standalone
`vibeqc.solvers` import **and** through the high-level
[`run_job`](output_files.md) entry point:

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("h2.xyz")
result = run_job(mol, basis="sto-3g", method="fci", output="output-h2-fci")
```

`run_job(method=…)` accepts `"fci"`, `"selected_ci"`, `"dmrg"`,
`"v2rdm"`, and `"transcorrelated_ci"` alongside the mean-field
`"rhf"` / `"uhf"` / `"rks"` / `"uks"`, plus the active-space
multireference methods `"casci"`, `"casscf"` (orbital-optimized),
`"nevpt2"`, and `"caspt2"`. `"caspt2"` is the internally-contracted
variant (Andersson 1992), validated against OpenMolcas `&CASPT2` to
≤2 µHa and un-gated. Small jobs use the explicit contracted determinant
engine; larger unshifted jobs automatically use the direct active-CI
signature-block path, validated through H2O/cc-pVDZ CAS(4,4). The older
strongly-contracted variant
(`caspt2(variant="sc")`) stays gated under
`VIBEQC_EXPERIMENTAL_MULTIREF=1`. Nonzero IPEA shifts use the canonical
eight-class explicit contraction and match OpenMolcas within 2 µHa on the
pinned single-state cases. IPEA is not yet supported with selected references,
the matrix-free `engine="cases"`, or multi-state CASPT2. The solver-specific
option objects are passed via the matching `run_job` kwargs
(`selected_ci_options=`, `dmrg_options=`, `v2rdm_options=`,
`transcorrelated_options=`, `casci_options=`, `caspt2_options=`,
`casscf_options=`); `caspt2_options` selects the
CASPT2 `variant` and the imaginary level shift / IPEA shift / frozen
core (`CASPT2Options(imaginary=…, n_frozen=…)`); `casci_options`
carries the multi-root request (`CASCIOptions(nroots=…)`) and the
determinant-count guard (`max_det`). An `active_space=(n_orbitals,
n_electrons)` kwarg applies the standard frozen-core / CAS partition
before the solver runs, the lowest `(nelec − n_elec)/2` orbitals
become a properly **dressed** doubly-occupied core (effective
one-electron term + `E_core` offset), so the reported energy is the
full CAS total energy, identical between the determinant solvers and
`casci` on the same window (see the
[active-space tutorial section](../tutorial/non_hf_solvers.md#active-spaces)
and `python/vibeqc/solvers/ACTIVE_SPACE.md` for the contract).
`run_job` builds the Hartree-Fock orbital basis internally (UHF
when `multiplicity > 1`, RHF otherwise) and hands the resulting
`Hamiltonian` to the solver.

## Where HF still lives

The HF reference is a **convenience default**, not a structural
dependency of the solver layer:

1. **`get_hf_orbital_provider`** in `_hamiltonian.py`, purely a
   convenience.  Callers can pass *any* orthonormal orbital basis via
   `transform_hamiltonian(ham_ao, C)` or construct the `Hamiltonian`
   directly.

2. **The `run_job` entry point** builds an HF orbital basis as the
   default orbital source before dispatching to a solver, but the
   solver itself only sees the abstract `Hamiltonian`. Supply a
   different orbital set by building the `Hamiltonian` yourself and
   calling the `solve_*` function directly.

3. **The v2RDM density path**, `_v2rdm.py` diagonalises the same
   determinant Hamiltonian used by the FCI/CASCI stack and contracts
   the ground-state CI vector to a spin-summed 1-RDM and 2-RDM.

The solver layer itself (`_selected_ci.py`, `_dmrg.py`, `_v2rdm.py`,
`_transcorrelated.py`) has **zero** HF-specific code.  All methods
operate on the abstract `Hamiltonian` interface.

## Method summaries

### Selected CI (`_selected_ci.py`)

Implements a CIPSI-style (Configuration Interaction using a
Perturbative Selection made Iteratively) algorithm:

1. Start from a reference determinant.
2. Generate all singles and doubles from the current space.
3. Estimate PT2 weights: ΔE ≈ |⟨D|H|Ψ₀⟩|² / (E₀ − ⟨D|H|D⟩).
4. Add top-N candidates to the variational space.
5. Diagonalise H in the expanded space.
6. Repeat until energy change < `conv_tol_energy` or `target_size`
   reached.

**Key options**: `target_size`, `pt2_threshold`, `do_pt2_correction`,
`spin_restricted`.

**Determinant basis.** `spin_restricted` defaults to **`False`** (since
issue #107, 2026-09-03): the solver walks `(alpha_occ, beta_occ)`
determinants, the full S_z sector of the active space, so a converged
run is the CAS / full-CI answer in that space. On N2/cc-pVDZ
CAS(6e,6o) the default ladder is variational and monotone in the
budget (`target_size` 6 / 20 / 50 / 100 gives E_var −109.01004 /
−109.01865 / −109.02172 / −109.02178 Ha, the last bit-identical to the
CASCI value −109.02177500 Ha; `tests/test_solvers_selected_ci.py`
computes that oracle in the same test).

`spin_restricted=True` is the **closed-shell, seniority-zero basis**
(doubly-occupied CI, DOCI; Bytautas et al. 2011, cited automatically on
that basis): each `Det` is a tuple of doubly-occupied spatial orbitals
excited by whole pairs and coupled through the pair-move integral
⟨ii|aa⟩. It is a closed subspace (no open-shell configuration is
representable), so on a system whose exact state has seniority > 0 it
converges to the DOCI energy, not the CAS energy. On N2/cc-pVDZ
CAS(6e,6o) that is −108.98507 Ha over all C(6,3) = 20 closed-shell
determinants, 36.7 mHa above CASCI; for two electrons (H2/STO-3G) the
singlet is seniority-zero and the two-determinant closed-shell space
*is* the FCI energy, −1.13727594 Ha. Until issue #639 (2026-09-03) the
pair-move coupling was scored with the one-electron Brillouin formula
and vanished for HF orbitals, so the restricted ladder froze at ndet=6,
−108.95947 Ha, and H2 returned the HF energy; before #107 that was the
default and was labelled `converged=True`. A restricted run that
exhausts its candidate pool short of `target_size` reports
`converged=False` with `stop_reason="space_exhausted"`, and any
restricted run that stops short of its budget warns that its limit is
the DOCI energy; do not read it as a CAS estimate.

**How a run stopped.** `SolverResult.stop_reason` is one of
`"target_size"` (the requested budget was spent), `"energy"` (the
ΔE criterion fired), `"space_exhausted"` (every candidate coupled to
the space is already in it; the connected space is complete), or
`"max_iter"`. The `.out` prints it next to `Converged:`. Candidates
whose second-order estimate is below one ulp of the current total
energy are never admitted (they cannot move the energy), so an
exhausted pool is reported as exhausted rather than as an energy
convergence over noise determinants.

For paper or SI benchmark rows, treat the default selector as a
convenience preset, not as a stable determinant-count contract.
Near-degenerate active spaces can rotate equivalent orbitals across
BLAS/thread builds or releases, which can change which determinant
weights sit just above `pt2_threshold`. Pin `SelectedCIOptions(...)`
and report both `ndet` and the PT2 correction when the selected
determinant budget is part of the comparison. The text `.out` report
prints the variational energy and PT2 correction whenever PT2 is
requested, including an explicit zero correction.

**Scaling**: O(N_det² + N_cand · n_occ² · n_vir²).  For small
active spaces (≤ 12 orbitals) this runs in seconds.

**Selected CI in an active space / as a CASSCF solver** (roadmap 25i):
`vibeqc.solvers.selected_casci` mirrors `casci()`'s interface (frozen
core, `ms2`, multi-root, `det_guess` warm start) but grows a selected
determinant subset by the multi-root CIPSI criterion instead of
enumerating the full CAS space; energies are variational at every
step, and 1-/2-RDMs of the truncated wavefunction are exact
(`make_rdm12` handles non-closed determinant lists via spill-aware
generator products), which is what the CASSCF orbital gradient needs.
Passing `ci_solver="selected_ci"` to `casscf()` (or
`CASSCFOptions(ci_solver="selected_ci", selected_ci_options=...)`
through `run_job`, method label suffix `_selci`) runs that kernel
inside every macro-iteration with the selected set warm-started
across iterations; in the full-selection limit it reproduces the
determinant CASSCF to <=1e-10 Ha (pinned on LiH and single-state
H2O in `tests/test_selected_casci.py`), and truncated runs approach
it variationally from above as `target_size` / `pt2_threshold`
tighten.  A selected-CI CASSCF also serves as the reference for
single-state `method="nevpt2"` / `method="caspt2"` (pass the same
`CASSCFOptions(ci_solver="selected_ci", ...)`): the MR-PT2 engines
are determinant-based (the strongly-contracted external groups for
NEVPT2, the `Ê_pr Ê_qs|0⟩` contracted space for IC-CASPT2), so they
build their zeroth order on the selected wavefunction directly,
without re-solving the full CAS or forming any high-order RDM.  The
result is the MR-PT2 of that reference and converges to the full-CAS
MR-PT2 as the selection tightens; at the full-selection limit it
reproduces the dense reference path exactly (hence OpenMolcas
`&CASPT2` to <=5e-6 Ha, pinned in `tests/test_solvers_mrpt_parity.py`).
Multi-state MS/XMS-CASPT2 still needs the exact CI backend (its model
space re-solves a multi-root CASCI over the full active space), as
does `method="mrci"`; both reject a selected reference with a clear
error.  `ci_solver="selected_ci"` also combines with
`spin_pure=True` for state averaging: each macro-iteration solves a
buffered root set in the selected space and keeps the lowest
`nroots` S^2-pure roots, exactly the dense filter's semantics (the
default `spin_complete` alpha/beta-swap closure makes the truncated
space spin-flip invariant, which symmetry-blocks singlet-triplet
mixing, so truncated roots stay spin-pure to well below the
classification tolerance; full-selection-limit parity with the
dense spin-pure SA-CASSCF and an OpenMolcas-pinned runner
composition are in `tests/test_selected_casci.py`).

Two backends share the kernel (`VIBEQC_SELECTED_CI_BACKEND=
auto|python|cpp`): the pure-Python engine (the validation oracle,
comfortable to ~10^3 selected determinants) and a C++ kernel
(sparse Slater-Condon Hamiltonian over SpinDet bitmasks, block
Davidson, pair-based truncated-list RDMs; machine-precision parity
with the Python oracle pinned in the test suite).  `auto` keeps
small full spaces on the Python path and dispatches actives whose
full determinant count exceeds 10^5 to C++.  This is the route past
the 2M-determinant direct-CI wall: N2/6-31G CAS(14,14) (11.8M full
determinants) runs as a selected CASCI in seconds at 20-50k
selected determinants, and the full selected-CI CASSCF converges there with
`orbital_step="nr"` in 8 macro-iterations and with the default
`"auto"` in 29 (~2 minutes either way).  `"auto"` hands over to
Newton-CG when the gradient drops below `switch_to_nr` OR when
Super-CI's energy progress stalls (`stall_to_nr`, default 1e-4
Ha/iteration over a 3-iteration window): the stall trigger covers
the stiff large-CAS surfaces where the first-order phase used to
plateau above the gradient threshold without handing over.  Keep
`significant_coeff` in
step with `pt2_threshold` when tightening: it truncates the
generator set, and the selection plateaus once the wavefunction
tail falls below it.  Note that CASSCF basin selection can differ
between CI backends on basin-rich systems (truncation perturbs the
early trajectory), as it does between codes.

For large actives (25+ orbitals), `select_eps` arms the heat-bath
prefilter (Holmes, Tubman & Umrigar 2016): a generator determinant's
contribution to candidate D is dropped when `|H_DI| * max_k|c_I^k| <
select_eps`, which both kernels apply with the identical predicate
(the Python kernel by brute force, the oracle; the C++ kernel by
walking presorted |H|-ordered double-excitation lists, stopping at
the first sub-threshold entry, since a double excitation's |H|
depends only on its four orbital indices).  The C++ kernel also
extends its sparse Hamiltonian incrementally across selection
cycles (each determinant's excitation rectangle is enumerated once
per solve, not once per cycle).  Measured on N2 actives at 20k
selected determinants: CAS(10,26) (cc-pVDZ window, 4.3e9 full dets)
11.0 s -> 3.9 s with `select_eps=3e-5` (energy cost 5 µHa);
CAS(10,40) (cc-pVTZ window, 4.3e11 full dets) runs in 11.6 s
(23.7 s at eps=0; cost 11 µHa).  Most of that win is the prefilter
plus the incremental Hamiltonian; the presorted walk's edge over a
brute-force prefilter grows with the generator count (1.24x at
`significant_coeff=1e-5`, 40 orbitals) and is the structural
enabler for the deterministic side of the semistochastic PT2 below.
`select_eps=0` (default) keeps exact CIPSI scoring.

**Epstein-Nesbet PT2 on the selected wavefunction**
(`vibeqc.solvers.selected_ci_pt2`, the SHCI perturbative stage:
Sharma, Holmes, Jeanmairet, Alavi & Umrigar, JCTC 13, 1595 (2017)).
Post-processing on a converged `selected_casci` result, per root:
the second-order correction over perturbers outside the selected
space, with contributions screened at `|H_ai c_i| >= eps2` (Eq. 5)
and the enumeration pruned by the heat-bath walk.  Two modes:

* deterministic (`n_samples=0`): exact at the given `eps2`; the
  perturber map holds every survivor, so memory grows as `eps2`
  shrinks (the SHCI memory bottleneck).
* semistochastic (`n_samples >= 2`): deterministic part at a loose
  `eps2_loose` plus the unbiased sampled difference estimator of
  `E2[eps2] - E2[eps2_loose]` (Eqs. 7-11; batches of `sample_size`
  determinants drawn with replacement from `p_i ~ |c_i|`, shared
  between both thresholds so the stochastic error largely cancels).
  Returns mean and standard error; fixed `seed` gives bitwise
  reproducibility per build.  `eps2_loose == eps2` has identically
  zero noise; `eps2_loose=inf` is the fully stochastic estimate.

The deterministic kernel is pinned against an independent
dense-matrix evaluation of Eq. 4 and the brute-force Python oracle
to machine precision, and a selected space that exhausts a symmetry
block correctly yields E2 = 0.  The kernel uses coherent perturber
numerators (contributions summed over all generators before
squaring); the legacy `method="selected_ci"` `do_pt2_correction`
estimate routes through the same machinery since 2026-06-11 (its
historical per-pair sum neglected cross-generator interference).
Through `run_job`, request the PT2 stage on a selected-CI CASSCF
with `CASSCFOptions(ci_solver="selected_ci",
pt2=SelectedCIPT2Options(...))`: per-root corrections surface on
`SolverResult.selected_pt2` and as a dedicated .out block, the
headline energy stays variational, and the Sharma-2017 citation
reaches the references.  Scale (N2, 20k selected determinants, single
core): CAS(10,26) deterministic `eps2=1e-6` (10.5M perturbers)
in 7 s; CAS(10,40) (26.9M perturbers) in 19 s; semistochastic
`eps2=1e-9` totals with ~2-5 µHa statistical error in 10-30 s.

### DMRG (`_dmrg.py`)

Implements a two-site DMRG sweep with MPS wavefunction:

- MPO construction via complementary-operator approach.
- Two-site variational optimisation (power iteration).
- SVD truncation with discarded-weight reporting.
- Bond-dimension schedule (e.g. [4, 8, 16, 32, 64]).

**Limitations** (minimal implementation):
- Python-level tensor contractions only, adequate for ≤ 12 orbitals
  and bond dimension ≤ 128.  Production use requires C++/GPU.
- No quantum-number symmetry blocking.
- The MPO is built as a placeholder; the two-site solver uses
  power iteration on the reduced wavefunction.

**Key options**: `bond_dim_schedule`, `n_sweeps`, `truncation_tol`,
`orbital_order`.

### Variational 2-RDM (`_v2rdm.py`)

Returns the exact small-active-space N-representable 1/2-RDM by
diagonalising vibe-qc's own determinant Hamiltonian and contracting
the ground-state CI vector.  This makes the reported energy
identical to FCI/CASCI in the same orbital space while still exposing
the spin-summed 1-RDM and 2-RDM through the v2RDM result object.

**Known limitation**: Q and G SDP feedback is not currently
implemented.  Requests whose `constraints` string contains `"q"` or
`"g"` raise `NotImplementedError` rather than returning a misleading
P-only number.  The `mu`, `mu_factor`, and convergence-tolerance
fields are retained for input compatibility, but the exact
small-space route converges in one deterministic diagonalisation.

**Key option**: `constraints="p"` for the current production route.

### Transcorrelated Hamiltonian (`_transcorrelated.py`)

Applies a similarity transformation H̃ = e^{−τ} H e^{τ} with a
correlator τ:

- **Simple Gaussian**: Damp two-electron integrals by
  (1 − γ·exp(−Δ)) based on orbital-index proximity.
- **Exponential Jastrow**: Longer-tailed damping,
  (1 − γ·(1 − exp(−βΔ))/β).
- **Custom**: Pass-through for user-supplied transformations.
- **NO2B**: Normal-ordered two-body approximation, discards
  the 3-body terms that arise from double commutators.
- **Symmetrisation**: Optional Hermitisation for standard
  eigensolvers.

**Diagnostics**: Reports norms of original/transformed 1e and 2e
terms, discarded 3-body norm, and anti-Hermitian part magnitude.

**Key options**: `form`, `gamma`, `no2b`, `symmetrize`,
`report_diagnostics`.

### CASCI / CASSCF / NEVPT2 / CASPT2

The active-space multireference methods are built on a common
determinant-basis FCI engine:

* **CASCI** (`method="casci"`), a full CI in the user-specified
  active space of *n* electrons in *m* orbitals, CAS(*n*,*m*).  Orbitals
  outside the active space are either frozen (doubly occupied core, properly
  dressed with an effective one-electron term) or empty (virtual).  CASCI
  with the full orbital space is identical to FCI.  Validated against PySCF
  to ~1e-12 Ha.

  **Multi-root CASCI**: pass `casci_options=CASCIOptions(nroots=N)` to
  `run_job` to solve for the N lowest states in one diagonalization.
  `result.energy` stays the ground root; the per-root total energies land
  in `result.root_energies` (per-root `<S^2>` in `result.root_s2`) and
  the .out solver block prints the root table.  The roots are the lowest
  eigenvectors of the fixed-M_s determinant sector with **no** S^2
  filtering, higher-spin states interleave (on H2 the triplet sits
  between the two lowest singlets), so read the `<S^2>` column to
  identify each state.  Wired in `tests/test_runner_casci_nroots.py`.

  ```python
  from vibeqc import Atom, Molecule, run_job
  from vibeqc.solvers import CASCIOptions

  h2 = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])  # bohr
  r = run_job(h2, basis="sto-3g", method="casci", active_space=(2, 2),
              casci_options=CASCIOptions(nroots=3))
  print(r.root_energies)  # [-1.13727594, -0.53180757, -0.16929174] Ha
  print(r.root_s2)        # [0.0, 2.0, 0.0] -- singlet, triplet, singlet
  ```

  The standalone solver takes the same knob directly, 
  `vibeqc.solvers.casci(h1e, h2e, ..., nroots=N)`, for workflows that
  build their own MO-basis Hamiltonian (`build_hamiltonian_mo`, see
  [Architecture](#architecture) above); its `CASCIResult.e_totals` /
  `ci_coeffs_all` carry all roots.

  **Backends**: small determinant spaces (<= 4000 determinants) use the
  dense Slater-Condon matrix + `eigh`; larger spaces dispatch
  automatically to the **C++ direct engine**, string-based sigma builds
  (Knowles-Handy 1984) with a block Davidson eigensolver (Davidson
  1975), which never materializes the Hamiltonian and reaches
  ~10^6-determinant spaces.  Force a backend with
  `VIBEQC_CASCI_BACKEND=python|cpp|auto`.  Both backends share the
  determinant ordering, phase conventions, and RDM conventions
  (parity-tested in `tests/test_casci_direct.py`).

* **CASSCF** (`method="casscf"`), orbital-optimized CASCI.  Minimises
  the CASCI energy with respect to orbital rotations between the
  inactive/active/virtual subspaces.  Two orbital update methods: **Super-CI**
  (diagonal Fock-based |H_diag| Hessian, one CASCI solve per iteration,
  robust far from convergence) and **Newton-CG** (`orbital_step="nr"`:
  trust-region truncated CG with Hessian-vector products by central FD
  of the analytic gradient, 2 CASCI solves per CG iteration,
  independent of the rotation-pair count; quadratically convergent).
  The default `"auto"` mode starts with Super-CI and switches to
  Newton-CG near convergence.  Backtracked line search (with a
  steepest-descent fallback) guarantees monotone energy descent.
  `tests/test_casscf_scale.py` pins Super-CI and Newton-CG to agree
  with each other to 1e-8 Ha on a unique-minimum LiH CAS(2,2) case:
  an internal cross-check between vibe-qc's own two orbital-update
  methods, not a PySCF comparison (that test file has no PySCF import).
  Its headline CAS(10,10) case (63 504 determinants per CI solve,
  converging in ~1.5 min on a laptop) is explicitly **not** compared
  to PySCF for equality: the file's own docstring records vibe-qc and
  PySCF `mcscf.CASSCF` converging to different stationary points on
  that system (8.13 mHa apart), and the regression deliberately pins
  vibe-qc's own value rather than asserting cross-code parity there.

  **State-averaged CASSCF**: pass `CASSCFOptions(nroots=2)` to `run_job`
  to minimise the weighted average over multiple states.  Per-root
  energies in `CASSCFResult.e_totals`.

  **Analytic nuclear gradient** (single-state, state-specific only;
  SA-CASSCF gradients are a later milestone).  vibe-qc computes the
  CASSCF energy gradient analytically as the complete derivative of the
  variational CASSCF energy -- density-contracted derivative integrals
  plus the generalized-Fock energy-weighted overlap term; a converged
  CASSCF is stationary in every orbital and CI parameter, so no
  z-vector / response term exists.  The gradient is surfaced on the
  returned `SolverResult.gradient` (shape `(n_atoms, 3)`, Hartree/bohr):

  ```python
  result = run_job(h2o, basis="sto-3g", method="casscf", active_space=(4,4))
  forces = -result.gradient  # F = -dE/dR
  ```

  The analytic vector agrees with a central finite difference of the
  converged CASSCF energy to 2.2e-7 Ha/bohr (H2/6-31G CAS(2,2)),
  2.5e-7 (H2/cc-pVDZ) and 1.1e-6 (N2/cc-pVDZ CAS(6,6) at R = 2.074 bohr,
  where the residual is the FD's own truncation error and the transverse
  components are 1.1e-16).  An earlier revision of this page
  described the path as "87 % of the full CP-MCSCF gradient" and offered
  `CASSCFOptions(compute_wz=True)` as an experimental W^z correction;
  the missing 12 % was a since-fixed defect in the energy-weighted
  density, the "correction" was a phantom term for a variational energy
  that produced spurious gradient components, and `compute_wz=True` is
  now a no-op alias of the analytic gradient (GitLab #516).  The
  Handy-Schaefer z-vector (Helgaker-Jorgensen-Olsen Sec. 12.5, Handy &
  Schaefer 1984) is used where it belongs: in the CASPT2 / NEVPT2
  relaxed gradients.  Validated in a three-way comparison:

  | Reference | Energy | Gradient |
  |---|---|---|
  | ORCA 6.1 EnGrad | Δ = 0.6 mHa | agrees with vibe-qc FD |
  | OpenMolcas &ALASKA | Δ = 1e-8 Ha | agrees with vibe-qc FD |
  | Full CASSCF FD (in-repo) | exact oracle | analytic = FD to ~2e-7 Ha/bohr |

  **Open-shell starting orbitals**: for multiplicity > 1 the CAS family
  starts from **UHF natural orbitals** (UNO-CAS, Pulay-Hamilton 1988), 
  one spin-restricted orbital set ordered by descending occupation, so
  the doubly-occupied core and active window are well-defined (on
  O2/STO-3G CAS(2,2) this lowers CASCI by 2.1 mHa and cuts CASSCF from
  27 to 4 iterations vs the old UHF-α reference).  Override with
  `run_job(cas_reference="rhf"|"uhf"|"uno"|"rohf")`.  **`cas_reference=
  "rohf"`** uses spin-restricted open-shell HF orbitals (Roothaan 1960):
  a spin-pure (⟨S²⟩ = S(S+1) exactly), contamination-free alternative to
  UNO, the orbitals come straight from the ROHF closed/open/virtual
  partition with no natural-orbital diagonalisation.

* **NEVPT2** (`method="nevpt2"`), strongly-contracted N-electron
  valence perturbation theory on a CASCI reference.  Validated
  against PySCF `mrpt.NEVPT` per excitation class to ~1e-9 Ha.

* **CASPT2** (`method="caspt2"`), internally-contracted (default)
  CASPT2 validated against OpenMolcas to <=2 uHa, with automatic
  explicit-small / direct-large dispatch.  The imaginary level shift
  (Forsberg-Malmqvist 1997, `imaginary=` in `CASPT2Options`) works on
  both paths, large intruder-plagued systems included (validated vs
  live OpenMolcas at cc-pVDZ scale).  The older strongly-contracted
  variant (`variant="sc"`) is gated behind
  `VIBEQC_EXPERIMENTAL_MULTIREF=1`. IPEA-shifted jobs use the canonical
  eight-class explicit engine and are OpenMolcas-pinned for single-state
  IC-CASPT2. Selected references, `engine="cases"`, and `multistate` reject
  nonzero IPEA explicitly.

  **Reference orbitals**: by default the PT2 reference is a CASCI on the
  HF orbitals (matching an OpenMolcas `RASSCF … CIonly` run).  Pass
  `casscf_options=` to optimize the orbitals first, the standard
  **CASSCF→CASPT2 / CASSCF→NEVPT2** composition.  The PT2 then runs on
  the CASSCF-converged integrals and lowest-root CI vector (the method
  label gains a `_casscf` suffix), and for a state-averaged reference
  the per-root CASSCF energies are surfaced in
  `SolverResult.root_energies`.  Validated against OpenMolcas full
  `&RASSCF` + `&CASPT2` to <=5e-6 Ha on unique-minimum systems (H2,
  N2); note that CASSCF is non-convex, on basin-rich cases like H2O
  CAS(4,4), different programs can converge to different stationary
  points (vibe-qc's optimizer lands at least as deep as the recorded
  OpenMolcas/PySCF basins there, see
  `tests/test_solvers_mrpt_parity.py::TestCASSCFReferencedCASPT2`).

  ```python
  # CASPT2 on a CASSCF(4,4) reference -- the standard composition
  run_job(h2o, basis="6-31g", method="caspt2", active_space=(4,4),
          casscf_options=CASSCFOptions())
  ```

  **Single-state analytic nuclear gradient:** set
  `CASPT2Options(compute_corr_grad=True)` on a state-specific, closed-shell
  CASSCF reference. For unshifted (`ipea=0`, `imaginary=0`) IC-CASPT2 jobs
  that remain on the explicit small-space engine, vibe-qc evaluates the exact
  stationary Hylleraas density, solves the coupled CASSCF orbital/CI response,
  and contracts the resulting Lagrangian density with analytic AO derivative
  integrals. The response right-hand sides and Hessian are built by internal
  parameter finite differences, but no displaced nuclear energies enter the
  returned gradient. H2/6-31G CAS(2,2) and core-containing LiH/STO-3G
  CAS(2,2) reproduce fully relaxed total-energy finite differences within
  `1e-7` Ha/bohr.

  IPEA or imaginary shifts, `engine="cases"`, the direct large-space engine,
  selected-CI references, and open-shell/state-averaged single-state runs are
  not silently mixed with this gradient: direct gradient requests fail with a
  capability error, while geometry optimizers keep those variants on their
  outer full-energy finite-difference route. MS/XMS-CASPT2 uses the separate
  analytic response described below.

* **MS-CASPT2 / XMS-CASPT2** (`caspt2_options=CASPT2Options(multistate=
  "ms"|"xms", nroots=N)`): multi-state CASPT2 for interacting electronic
  states (avoided crossings, conical-intersection regions, excited-state
  orderings), where independent single-state corrections distort the
  picture.  Per model state a single-state IC-CASPT2 is solved and the
  Finley effective Hamiltonian `H_eff[i,j] = <Phi_i|H|Phi_j> +
  <Phi_i|H|Psi_j(1)>` is symmetrized and diagonalized (Finley 1998); its
  eigenvalues land in `SolverResult.root_energies` (lowest =
  `result.energy`), the mixing matrix / effective Hamiltonian / per-state
  diagnostics in `result.multistate` and the `.out` solver block.

  `multistate="ms"` uses each state's own Fock operator (the OpenMolcas
  `MULTistate` convention); `multistate="xms"` first rotates the model
  states to diagonalize the state-averaged Fock and uses that single
  state-averaged H0 for all states (Granovsky 2011 / Shiozaki 2011;
  OpenMolcas `XMULtistate`); prefer XMS near degeneracies, MS for
  well-separated states.  The model space is the lowest `nroots`
  **spin-pure** CASCI roots of the reference basis (vibe-qc's determinant
  CI works in one M_s sector, so e.g. M_s=0 also contains triplets; the
  S^2 filter keeps the states a CSF-based code averages).  Validated
  against OpenMolcas `MULTistate`/`XMULtistate` to <=5e-6 Ha on
  HF-orbital references (LiH equilibrium + stretched avoided crossing,
  H2).  With `imaginary!=0` note the representation caveat documented in
  `caspt2()`: vibe-qc shifts the full nondiagonal H0, OpenMolcas its
  case-diagonal representation: identical without intruders, ~1e-4 Ha
  apart at strong intruders.  `multistate` requires `variant="ic"` and
  `ipea=0`.

  **Relaxed analytic nuclear gradient**
  (`CASPT2Options(compute_corr_grad=True)`, matching the single-state
  opt-in): SA-CASSCF-referenced MS/XMS runs surface the fully relaxed
  gradient of the target multi-state root on `result.gradient`
  (Ha/bohr).  The implementation solves the
  SA-CASSCF Lagrangian (the state-averaged energy, not the individual
  state, is orbital-stationary; Stalring, Bernhardsson & Lindh, Mol.
  Phys. 99, 103 (2001)) plus the state Lagrangian for the
  effective-Hamiltonian mixing (Celani & Werner, J. Chem. Phys. 119,
  5044 (2003); the OpenMolcas MCLR "SLag" mechanism), so orbital, CI
  and model-state-rotation responses are all included.  Validated
  against central-difference FD of the MS/XMS energy to ~1e-7 Ha/bohr
  (H2 + LiH mixed-state, both modes; `tests/test_casscf_gradient.py::
  TestMSCASPT2Gradient`) and cross-checked against OpenMolcas
  MCLR+ALASKA.  The model states must all be SA-averaged
  (`casscf_options.nroots >= caspt2_options.nroots`); CASCI-referenced
  multistate runs carry no gradient.  The semi-numerical response
  (finite differences of the MS pipeline over the orbital + external-CI
  parameters) is exact but scales with the determinant count, so the
  CI-response is currently gated to small active spaces
  (`max_det_response=64`); the analytic transition-RDM CI Lagrangian
  for larger spaces is on the roadmap.

  **State-pair nonadiabatic derivative coupling (under review):** set
  `CASPT2Options(nac_pair=(Q, P))` on an SA-CASSCF-referenced MS/XMS job.
  The `(n_atoms, 3)` coupling in bohr^-1 is returned as
  `result.nonadiabatic_coupling`; `result.nonadiabatic_pair` records the
  ordered physical-root pair. Reversing the pair reverses the vector.
  This route evaluates the full symmetrized MS-CASPT2 derivative coupling,
  not only the effective-Hamiltonian interstate term: every Cartesian
  component reoptimizes the SA-CASSCF reference at central displacements,
  reconstructs the internally contracted first-order CASPT2 wavefunctions,
  and uses cross-geometry AO overlaps for determinant-centre motion (Park and
  Shiozaki, JCTC 13, 2561 (2017)).

  The verified initial envelope is deliberately narrow: exact-CI,
  closed-shell singlet, equal-weight SA-CASSCF; `variant="ic"`,
  `engine="auto"`, `ipea=0`, `imaginary=0`, and `n_frozen=0`; at least two
  nondegenerate model states. Unsupported references, shifts, unequal
  weights, open shells, singular gaps, and failed state tracking raise a
  capability error instead of returning a partial vector. The default
  `nac_fd_step=1e-3` is in bohr. Because the numerical route performs
  `2 * 3N` displaced SA-CASSCF plus MS-CASPT2 solves, it targets the explicit
  small-space engine and is substantially more expensive than an energy or
  gradient.

  ```python
  # XMS-CASPT2 over 2 states on a SA2-CASSCF(2,2) reference
  run_job(lih, basis="sto-3g", method="caspt2", active_space=(2,2),
          casscf_options=CASSCFOptions(nroots=2),
          caspt2_options=CASPT2Options(multistate="xms"))
  # Full derivative coupling d[0,1] in bohr^-1 on the result object
  nac = run_job(
      lih, basis="sto-3g", method="caspt2", active_space=(2,2),
      casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
      caspt2_options=CASPT2Options(multistate="xms", nac_pair=(0, 1)),
  )
  print(nac.nonadiabatic_pair, nac.nonadiabatic_coupling)
  # ... or on the CASCI(HF-orbital) reference -- here caspt2_options.nroots
  # is the kwarg that sets the model-space size (no casscf_options needed)
  run_job(lih, basis="sto-3g", method="caspt2", active_space=(2,2),
          caspt2_options=CASPT2Options(multistate="ms", nroots=2))
  ```

  SA-CASSCF spin purity: state-averaged CASSCF (`nroots > 1`)
  averages the lowest `nroots` *spin-pure* CI roots by default
  (S^2-filtered to S = ms2/2, re-selected every macro-iteration,
  which also protects against spin-sector root flipping along the
  optimization).  This is the averaging a CSF-based code (OpenMolcas,
  MOLPRO) performs by construction, and what SA users almost always
  mean: the fixed-M_s determinant sector interleaves higher-spin
  roots (a triplet sits between the two lowest singlets on
  essentially every small CAS), so a raw sector average silently
  mixes spin states.  With the default, the standard SA-CASSCF and
  SA-CASSCF -> MS/XMS-CASPT2 workflows match OpenMolcas end to end
  (LiH SA2: singlet-averaged `&RASSCF` to <=4e-8 Ha, full `&RASSCF`
  + `MULTistate`/`XMULtistate` to <=5e-6 Ha; see
  `tests/test_ms_caspt2.py` / `tests/test_runner_ms_caspt2.py`).
  The .out root table prints per-root `<S^2>` so the averaged
  composition is always visible.  **Pre-2026-06-11 behavior**: the
  default was the raw M_s-sector average; pass
  `CASSCFOptions(spin_pure=False)` for that deliberate mixed-spin
  ensemble (it reproduces PySCF's `state_average_` with the default
  `direct_spin1` solver exactly; pinned in
  `tests/test_solvers_sacasscf.py`).  Single-state runs
  (`nroots=1`) are unaffected: they keep the raw lowest root of the
  sector.  `spin_pure` composes with the selected-CI CASSCF backend
  (`ci_solver="selected_ci"`); see the Selected CI section above.

* **MRCI** (`method="mrci"`), uncontracted multireference CI with
  singles and doubles (MR-CISD) from all CAS reference determinants.
  Includes the Davidson (+Q) size-extensivity correction.  Energy
  ladder: CASCI < CASSCF < MRCI < MRCI+Q < NEVPT2/CASPT2 < FCI.

**Example, CASSCF with run_job**:

```python
from vibeqc import Molecule, Atom
from vibeqc.runner import run_job
from vibeqc.solvers import CASSCFOptions
import numpy as np

h2o = Molecule([Atom(8, [0,0,0]), Atom(1, [0,1.43,-0.93]), Atom(1, [0,-1.43,-0.93])])

# Single-state CASSCF(2,2)
run_job(h2o, basis="sto-3g", method="casscf", active_space=(2,2))

# State-averaged over 2 states
run_job(h2o, basis="sto-3g", method="casscf", active_space=(2,2),
        casscf_options=CASSCFOptions(nroots=2))

# Analytic gradient: result.gradient (Ha/bohr)
result = run_job(h2o, basis="sto-3g", method="casscf", active_space=(4,4),
                 write_molden_file=False, citations=False)
forces = -result.gradient  # F = -dE/dR
print(f"Force norm: {np.linalg.norm(forces):.4f} Ha/bohr")
```

## Usage example

```python
from vibeqc import Atom, Molecule, BasisSet
from vibeqc.solvers import (
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    SelectedCIOptions,
    solve_selected_ci,
)

# Build the system
mol = Molecule(
    [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])],
)
basis = BasisSet(mol, "sto-3g")
C = get_hf_orbital_provider(mol, basis)  # any orbitals work
H = build_hamiltonian_mo(mol, basis, C)

# Run Selected-CI
opts = SelectedCIOptions(target_size=50, verbose=1)
result = solve_selected_ci(H, opts)
print(f"E = {result.energy:.10f} Ha  ({len(result.ci_labels)} determinants)")
```

## Testing

```bash
.venv/bin/python -m pytest tests/test_solvers_*.py -v
```

36 tests covering: common interfaces, determinant generation,
Slater-Condon rules, Hamiltonian construction, Selected-CI
convergence, v2RDM constraint enforcement, and transcorrelated
workflows.

## References

See individual module docstrings for method-specific references.
Key architectural references for each method are listed in the
`__init__.py` module docstring.
