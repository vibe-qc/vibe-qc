---
myst:
  html_meta:
    "description": "Common vibe-qc errors and how to fix them: SCF non-convergence, linear-dependence aborts, memory errors, missing basis-set entries, and the current known-issues table."
    "og:title": "vibe-qc, troubleshooting"
    "og:description": "Symptom → cause → workaround for the errors most users hit: linear dependence, SCF oscillation, memory aborts, basis-set gaps."
---

# Troubleshooting

When a vibe-qc run fails the error message is almost always concrete
and actionable, but knowing which knob to turn requires a bit of
context. This page collects the errors users actually hit during a
first month with the code, with the canonical fix for each. The
[homepage admonition](index.md) carries the current known-issues list
(the things that are bugs in vibe-qc rather than misconfigurations on
your end); this page covers the *recoverable* situations.

For "I think I found a bug" cases see
[CONTRIBUTING.md](CONTRIBUTING.md) § Filing a bug.

## Triage the failure first

Do not change several settings at once. Preserve the original input and sort
the symptom into one of these classes:

| Failure class | Evidence to collect | First destination |
|---|---|---|
| Environment or import | exact command, `sys.executable`, `vibeqc.__file__`, banner | [Installation common issues](installation.md#common-issues) |
| Input or model | geometry source and units, charge, multiplicity, basis coverage | [Planning a calculation](tutorial/planning_a_calculation.md) |
| Resource limit | memory estimate, allocation, thread count, CPU/wall time, peak RSS | [Memory budget](user_guide/memory.md) |
| Molecular SCF convergence | complete trace, residual trend, initial guess, overlap spectrum | [SCF convergence](user_guide/scf_convergence.md) |
| Periodic oscillation or impossible energy | minimal input, lattice, k mesh, Coulomb route, matched reference | preserve as a possible bug; see below |
| Missing or incomplete output | `.system`, `.out`, stderr, output stem, disk space | [Output files](user_guide/output_files.md) |

The first useful comparison changes one factor. If you simultaneously change
the basis, guess, damping, thresholds, and geometry, a successful rerun does
not identify the cause.

## SCF didn't converge

### Symptom, molecular

```text
  iter  85   -75.94731234   3.4e-04   2.1e-03   8
  iter  86   -75.94731201   3.3e-04   2.1e-03   8
  iter  87   -75.94731267   3.4e-04   2.1e-03   8
WARNING: SCF did NOT converge after 100 iterations
RuntimeError: SCF failed: max_iter (100) reached without convergence
```

The third column (`dE`) is bouncing instead of decreasing. The
fourth column ($\lVert\mathbf{e}\rVert_F$) is plateaued. The DIIS
subspace (the trailing integer) is full at 8.

### Cause

DIIS is oscillating between two near-degenerate density solutions.
Common triggers:

- Stretched bond / dissociation-limit geometry (closed-shell
  reference becomes a poor description of the multireference state).
- Transition-metal complex with ligand-field-degenerate d orbitals.
- Broken-symmetry singlet biradical.
- Wrong initial guess (e.g. AUTO picked SAP on a system where SAD
  starts closer to the true minimum).

### Fix

In v0.8.0 the default `scf_accelerator` is already `EDIIS_DIIS`
(see the [stiff-convergence tutorial](tutorial/ediis_diis_hybrid.md)),
if you're seeing this on a v0.8.x build, the default already fired.
The remaining levers, in order of bang-for-buck:

```python
# doc-audit: skip - option adjustments for an existing SCF setup
opts.level_shift   = 0.2          # Saunders-Hillier shift
opts.newton_threshold = 1.0       # Phase D2c Newton finalizer
opts.initial_guess = vq.InitialGuess.SAD  # if AUTO picked SAP
opts.max_iter      = 200          # last resort - buys time
```

If you specifically need the deprecated v0.7.x-style vanilla DIIS
behaviour for parity work:

```python
# doc-audit: skip - option adjustment for an existing SCF setup
opts.scf_accelerator = vq.SCFAccelerator.DIIS
```

The full decision tree is in
[user_guide/scf_convergence.md](user_guide/scf_convergence.md) §
SCF accelerators.

### Symptom, periodic

```text
  iter  18   -114.821342   1.2e-05   4.7e-04
  iter  19   -114.821339   1.1e-05   4.8e-04
  iter  20   -114.821345   1.2e-05   4.7e-04
```

For periodic SCF, an unexplained oscillation or impossible absolute energy is
a possible implementation bug, not permission to tune until the symptom
disappears. Preserve the original trace and construct the smallest reproducer
that keeps the same lattice, dimensionality, Coulomb route, basis character,
and k-point behavior. Compare it with an independent implementation using the
same conventions.

Do not use damping, level shift, quadratic fallback, or tighter thresholds to
paper over a gauge, Madelung, image-summing, symmetry, or Coulomb-route defect.
If a converged periodic result differs from a matched reference by more than
the route's documented accuracy envelope, report it even when the trace looks
smooth.

Smearing is appropriate only after the system is physically identified as
metallic or near-metallic. Then the k mesh, electronic temperature,
occupations, entropy term, and energy/free-energy convention form one
convergence study. See [Fermi-Dirac smearing](tutorial/fermi_dirac_smearing.md)
and [Periodic SCF convergence](tutorial/periodic_scf_convergence.md).

## "Canonical orth dropped too many basis directions"

### Symptom

```text
RuntimeError: Canonical orthogonalisation dropped 17 of 234 basis
directions (overlap eigenvalues below 1e-7). The basis set is
nearly linearly dependent on this geometry/cell. See
docs/user_guide/linear_dependence.md.
```

### Cause

The overlap matrix $\mathbf{S}$ has near-zero eigenvalues, two or
more basis-function combinations are nearly identical at the input
geometry. Most common triggers:

- Using a **molecular basis set in a periodic calculation** (e.g.
  `def2-tzvp` for solid NaCl). Diffuse functions overlap with
  periodic images. **Use the pob-* family for periodic systems**,
  see [Why solid-state calculations use `pob-TZVP`](tutorial/pob_tzvp.md).
- A **vDZP / dhf-* / x2c-* basis** without an explicit ECP center
  (the heavy-element basis is short by Z core electrons; without
  the ECP the SCF reaches for them via diffuse-function overlap).
- **Atoms placed at or very near the same coordinates** by an
  upstream geometry-conversion step (CIF → POSCAR rounding,
  duplicated atoms from a faulty supercell expansion).

### Fix

Pick the appropriate fix for the trigger:

```python
# doc-audit: skip - alternative repairs to apply to an existing periodic setup
# 1. Use a solid-state basis instead.
basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")

# 2. Filter out diffuse primitives at the BasisSet stage.
basis = vq.make_basis(sysp.unit_cell_molecule(), "def2-tzvp",
                      exp_to_discard=0.05)

# 3. Wire ECP centers explicitly for vDZP / dhf-* / x2c-*
#    (see docs/user_guide/ecp.md § ECPCenter recipe).
opts.ecp_centers = [
    vq.ECPCenter(Z=78, xyz=[0.0, 0.0, 0.0]),   # Pt
    ...
]
opts.ecp_library = "ecp60mdf"

# 4. Auto-optimise periodic cutoffs jointly with screening.
opts.auto_optimize_truncation = True    # default on v0.7+
```

[user_guide/linear_dependence.md](user_guide/linear_dependence.md)
walks through the v0.7 diagnostic stack
(`vq.eigs_preflight`, `vq.disambiguate_critical_overlap`,
`auto_optimize_truncation`) and the matching fix recipe in detail.

## Memory abort before the SCF starts

### Symptom

```text
vibe-qc estimates this calculation will require ~218.4 GB of memory:
    ERI tensor      186.0 GB
    ...
Available on this machine: 7.2 GB. ABORTING.

InsufficientMemoryError: Pass `memory_override=True` to `run_job` to proceed
anyway. In a low-level workflow, call `check_memory(..., allow_exceed=True)`.
```

### Cause

The pre-flight estimator (see [user_guide/memory.md](user_guide/memory.md))
sized the dense four-index ERI tensor or the MO transformation
buffers and added the configured headroom (1.5x by default); that
number exceeds the machine's
available RAM.

### Fix, in order of diagnosis

```python
# 1. If the dense AO ERI tensor dominates, use direct SCF.
opts.scf_mode = vq.SCFMode.DIRECT

# 2. Use density fitting when it is supported and appropriate to the route.
opts.density_fit = True
opts.aux_basis = vq.default_aux_basis_for(basis_name, kind="jk")

# 3. RIJCOSX can reduce hybrid-DFT exchange cost and storage.
opts.density_fit = True
opts.cosx = True

# 4. Reduce the basis only if the smaller basis still answers the question.
run_job(mol, basis="def2-svp", ...)

# 5. Override last. You accept the risk of swap-thrashing or an OOM kill.
run_job(mol, ..., memory_override=True)
```

Choose from the largest category in `estimate_memory`, not from basis count
alone. Direct SCF replaces the dense $8N_\mathrm{bf}^4$ AO tensor with
shell-pair scratch while retaining the same HF or KS model. Density fitting
and COSX use their own auxiliary basis and approximation controls, and a
post-SCF method can still allocate large transformed integrals or amplitudes
after a low-memory reference. See
[Direct SCF or in-core integrals?](tutorial/direct_scf_memory_tradeoff.md) and
[Density fitting](user_guide/density_fitting.md).

```{note}
The old v0.7.3 DF integral-kernel SIGSEGV on auxiliary bases with
`l >= 1` shells is fixed in current builds. The active DF-RHF suite runs
`def2-universal-jkfit` and `cc-pvdz-jkfit` and checks PySCF parity. If an
older checkout still segfaults below the Python boundary, it usually has a
stale libint install built without the required ERI2/ERI3 flags; follow the
clean rebuild recipe in [updating](updating.md#rebuild-one-native-dependency).
```

## `KeyError: 'Ne'` (or any other element) on basis load

### Symptom

```text
KeyError: "basis set 'pob-tzvp' has no entry for element 'Ne'"
```

### Cause

The bundled `.g94` for that basis set covers a subset of the
periodic table that doesn't include the requested element. Known
gaps:

| Basis | Missing | Mitigation |
|---|---|---|
| `pob-tzvp` / `pob-tzvp-rev2` | Ne | Use the def2-TZVP Ne block, or switch the whole calculation to def2-TZVP / cc-pVTZ |
| `pob-dzvp-rev2` | Si, Fe, and the transition metals (only the 19 lightest elements are parameterised) | Use `pob-tzvp-rev2` for silicates / TM systems |

### Fix

Either:

- Switch to a basis that covers the element (`def2-tzvp` /
  `cc-pvtz` for molecular work, `pob-tzvp-rev2` for periodic).
- Add the element by hand: copy the matching block from
  [BSE](https://www.basissetexchange.org/) into a `custom/*.g94`
  file (see [user_guide/basis_sets.md](user_guide/basis_sets.md)
  § Custom basis sets) and rebuild the library with
  `./scripts/setup_basis_library.sh`.

## "Explicit block-mode GAPW is Ha-scale wrong on a molecule like H₂O"

### Symptom

An older release, or a current calculation that explicitly requests
`one_centre="block"`, can converge cleanly but land Ha-scale away from the
all-electron reference when a core-bearing atom has multiple bonded
neighbours. H2O/STO-3G in a 12-bohr cube gives `-83.8226256933 Ha` in block
mode, `-8.859694 Ha` below the molecular RHF reference. This was the public HF
default before 2026-08-02.

### Cause

The block construction defines atom A's one-centre density using only the
A-centred AO block of the density matrix. The GAPW partition in the primary
method papers instead requires a representation of the **total** density
inside atom A's augmentation region, including the tails of basis functions
on other atoms and their bond-density products. Dropping those terms breaks
the hard-minus-soft Hartree identity. On the converged molecular density the
Hartree functional is already wrong by about `-4.213 Ha`; the self-consistent
minimisation then exploits that defective functional and doubles the total
error. This is a code/functional bug, not an SCF-convergence problem, and no
damping or convergence threshold can repair it.

### Fix / workaround

On current code, leave `one_centre="auto"` selected and declare the physical
envelope with `molecular_limit=True` on the standalone RHF/UHF driver, or
`gapw_molecular_limit=True` on `run_periodic_job`. RHF and UHF then use the
fit-free analytic-ERI augmentation; the H2O case above lands `+11.499 mHa`
from the molecular reference. Without that declaration HF fails before grid
or ERI allocation because no geometry-only rule can safely distinguish every
isolated molecule from a dense supercell. RKS and UKS continue to select block
mode automatically because safe multi-atom XC one-centre physics is still
open. Explicit block mode remains available only for legacy reproduction and
its previously validated narrow cases; explicit analytic is the expert opt-in
assertion of the molecular-limit envelope.

The analytic HF path is validated only for molecular-limit, single-Γ cells.
It does not establish dense-crystal or multi-k correctness. Use
`jk_method="gdf"` or `jk_method="bipole"` for compact all-electron crystals
and for molecular DFT. Direct GAPW analytic gradients and stress fail closed
for an analytic-mode result, and high-level GAPW RHF/UHF `optimize`,
`optimize_cell`, and `hessian` requests fail closed before they can switch
energy surfaces. The `VibeqcGAPW` ASE calculator uses full-SCF central finite
differences for analytic-mode forces. On a multi-k calculation it rejects
`one_centre` and `molecular_limit` instead of ignoring them. GAPW HF phonons
require explicit `gapw_kwargs={"one_centre": "block"}`; DFT phonons default
to block, while analytic/projector phonon requests fail closed. The regression
is
`tests/test_periodic_gapw_parity.py::test_gapw_aspherical_core_molecule_h2o_default_is_accurate`.

## "DF gradient disagrees by ~115 mHa/bohr"

### Symptom

Geometry optimisation drifts away from the true minimum when
`density_fit=True` is on; or hand-written FD gradients disagree
with `compute_gradient` by ~100 mHa/bohr on glycine / def2-TZVP.

### Cause

**Fixed in v0.8.0.** A libint engine-state leak in the 3c-ERI
gradient kernel (`compute_3c_eri_gradient_weighted` in
`cpp/src/df.cpp`). Two adjacent same-l heavy atoms (e.g. the
carboxyl O=C-O-H oxygens in glycine and formic acid) caused the
engine to leak derivative-buffer state across `compute()` calls.

### Fix

Upgrade to vibe-qc ≥ v0.8.0. The regression guard is
`tests/test_df_gradient.py::test_df_rhf_gradient_hcooh_def2_tzvp_matches_direct`.
`density_fit=True` gradients are now safe at def2-TZVP-class
basis sets.

## "Analytic RHF gradient is wrong"

### Symptom

Geometry optimisation walks to the wrong minimum on a system with
**f-shells AND ≥2 different second-row elements** (e.g. CO, CH₂O,
glycine + def2-TZVP). Magnitude can reach **161 mHa on glycine**
with a recent libint build.

### Cause

**Fixed in v0.8.0.** The `two_electron_gradient_contribution`
(direct 4-index ERI gradient) kernel relied on libint's internal
`DerivMapGenerator` unscramble path, which had a buggy
derivative-to-atom routing for high-l mixed-l shell quartets.
Fix C rewrites the kernel as a canonical 1/8 shell-quartet loop
with explicit l-canonical reorder before `engine.compute()`.

### Fix

Upgrade to vibe-qc ≥ v0.8.0. The regression guard is
`tests/test_gradient_f_bug.py::test_h2co_def2_tzvp_gradient_matches_pyscf`.
Direct analytic gradients with f-shells are now correct
(post-fix H2CO/def2-tzvp matches PySCF to ~5e-11 Ha/bohr).

## "Periodic (BIPOLE) forces / geometry optimisation look wrong"

### Symptom

A periodic geometry optimisation with `jk_method="bipole"` walks the
wrong way, or `compute_bipole_gradient_rhf/uhf/rks/uks` disagrees with a
finite-difference of the BIPOLE energy, including at Γ, and emits a
`UserWarning` about a "RESEARCH PREVIEW" gradient.

### Cause

The **analytic** BIPOLE gradient is still a gated research preview.  The
maintained RHF/UHF Gamma cases, maintained RHF/UHF multi-k cases, and
Gamma-local RKS/UKS LDA response regressions now track finite differences, but
broader KS functional/cell coverage, meta-GGA tau-Pulay, and multipole
far-field certification remain open.  RKS/UKS finite-temperature,
fractional-occupation, and multi-k analytic calls now fail fast to the
finite-difference path.  If a system sits outside the maintained surface, the
analytic driver can still disagree with the real BIPOLE total-energy finite
difference.

Since M5, erfc short-range energies use a larger precision-derived internal
ket-image domain and attached-symmetry crystals use a pair-resolved Fock
domain by default. The analytic derivative kernel does not yet differentiate
either traversal, so analytic gradient calls now fail closed when the SCF
result records one of them. Use the finite-difference path for the production
M5 energy. Setting `sr_image_precision=None` and
`use_fock_symmetry_reduce=False` is only a historical-domain diagnostic, not
the recommended force workflow.

### Fix

Use the exact finite-difference gradient, it is the default for all
production paths as of the 2026-05-31 fix:

```python
# doc-audit: skip - requires the periodic reproducer described above
# Geometry optimisation (FD forces by default - force_mode="fd")
from vibeqc.bipole_optimize import relax_atoms
result = relax_atoms(system, "sto-3g", kmesh, method="RHF")

# Standalone exact gradient (any method)
from vibeqc.bipole_gradient import compute_bipole_gradient_fd
g = compute_bipole_gradient_fd(system, "sto-3g", kmesh, opts,
                               method="RKS", functional="pbe")
```

`run_periodic_job(optimize=True)` and the periodic NEB driver already use
this path. If a displaced SCF point does not converge, the FD helper raises
instead of differentiating a failed iterate; pass
`require_converged=False` only when deliberately diagnosing a failed surface.
The analytic drivers remain available for research (`force_mode="analytic"`);
stay on the maintained regression surface if you compare them directly. The
full analytic gradient certification is tracked in `handovers/HANDOVER_BIPOLE_GRADIENT.md`;
the regression guards are in `tests/test_bipole_gradient.py`.

## CASPT2/NEVPT2 `compute_corr_grad=True` gradient has no PT2 correction (v0.15.17)

### Symptom

On v0.15.17, a CASPT2 or NEVPT2 analytic gradient requested with
`caspt2_options.compute_corr_grad=True` (or the NEVPT2 equivalent)
silently returns a gradient without the effective-density PT2
correction: an SA-CASPT2 gradient comes out identical to the
state-specific one. No warning or error is raised.

### Cause

The effective-density builder imported the `state_rdm12_cpp` C++
kernel outside its Python-fallback guard, and that kernel was never
exported from the compiled core, so the import always raised, and the
gradient driver's graceful-degradation handler silently dropped the
correction on every build.

### Fix

Fixed on `main` (2026-07-02): the import moved inside the fallback
guard and the pure-Python 2-RDM fallback was repaired (validated to
machine precision against a brute-force spin-orbital reference), so
the correction is always computed. Fixed again on current `main`
(2026-07-03): the restored C++ `state_rdm12_cpp` fast path now splits
the ket-side `E_rs|c>` and bra-side `E_sr|c>` branches correctly and
is re-enabled. Current `main` also warns if a requested PT2
effective-density, chain-rule, CI-Lagrangian, or explicit Lagrangian
correction has to fall back to a less complete gradient contribution.
On v0.15.17, treat `compute_corr_grad=True` gradients as CASSCF-level
only; regression guards live in `tests/test_casscf_gradient.py`
(`TestDyallChainRule`, `TestPT2Lagrangian`, `TestSASpecificity`, and
the degradation-warning regression) and `tests/test_pt2_density_rdm.py`.

## Multi-k GDF HF / hybrid-KS exchange wrong on diffuse bases in tight cells (v0.15.26 to v0.15.31)

### Symptom

On v0.15.26 through v0.15.31, `run_krhf_periodic_gdf` (and hybrid
`run_krks_periodic_gdf`) on a true multi-k mesh returns a wrong total
energy when the AO basis carries functions whose inter-cell overlap is
large, for example the ultra-diffuse Li 2sp shells of STO-3G on the
rocksalt LiH primitive cell. Measured on LiH/STO-3G at kmesh (2,1,1):
+2.04e-2 Ha/cell versus the same Hamiltonian's real-Gamma direct route
and versus pre-v0.15.26 releases. Gamma-only runs, pure (non-hybrid)
KS, and vacuum-padded cells (molecules in boxes, chains with vacuum)
are unaffected at the 1e-8 gate level.

### Cause

The v0.15.26 dense-core tail speedup mirrored shell-pair blocks of the
AO-pair Fourier transform at k = 0, using the transpose identity
`FT_mu,nu(G) == FT_nu,mu(G)`. That identity additionally requires
every evaluation frequency to be a reciprocal-lattice vector, which is
false for the momentum-shifted meshes `G+q` that the Bloch-pair GDF
fit evaluates for every `(k_bra != Gamma, k_ket = Gamma)` exchange
pair. The mirrored fit is silently wrong wherever inter-cell AO-pair
terms are non-negligible.

### Fix

Fixed on `main` (2026-07-10): the mirror is disabled unless the
evaluation frequencies are verified reciprocal-lattice vectors; the
Gamma-tail speedup itself is kept. Flagged for the next v0.15.x patch
release. On affected releases, cross-check any multi-k HF or hybrid-KS
number on diffuse-basis tight cells against `run_ccm_rhf_direct`
(real-Gamma route, unaffected) or upgrade to `main`. Regression pins:
`tests/test_rsgdf_dense_mesh_tail.py::test_gamma_mirror_disabled_on_momentum_shifted_mesh`
and the restored LiH (2,1,1) parity gates in `tests/test_ccm_direct.py`.

## Multi-k BIPOLE RHF raises `TypeError` instantly; BIPOLE RKS ignores `bz_integration="gilat"` (v0.15.23 to v0.15.30)

### Symptom

On v0.15.23 through v0.15.30, any `run_periodic_job(...,
method="RHF", jk_method="bipole", kpoints=...)` run at a multi-k mesh
fails in 0.0 s with

```text
TypeError: run_pbc_bipole_rhf() got an unexpected keyword argument 'bz_integration'
```

even though you never passed `bz_integration`. On the same releases,
`method="RKS"` with `jk_method="bipole"` *accepts*
`bz_integration="gilat"` and records it in the `.out`, but the SCF
silently runs with default Aufbau occupations - the Gilat-Raubenheimer
integration announced in those releases is inert on the BIPOLE RKS
route. (BIPOLE UHF/UKS + gilat correctly failed closed on those affected
releases; current `main` wires their existing per-spin kernel. GDF routes are
unaffected.)

### Cause

The commit wiring Gilat through GDF and BIPOLE RKS (`891acead`, first
in v0.15.23) put the new keyword on the wrong callsite in
`periodic_runner.py`: it passed `bz_integration=` unconditionally to
`run_pbc_bipole_rhf` (which did not accept it yet) and never passed it
to `run_pbc_bipole_rks` (which did).

### Fix

Fixed by `38c4afed`, shipped in v0.15.31 - upgrade, or pin a tag
outside the window. No energy computed on the affected tags is wrong:
the RHF route fails before any integral work, and the RKS route
computes a correct *default-occupation* result (only the requested
Gilat scheme is dropped). Regression pins:
`tests/test_periodic_runner_bipole_callsites.py` (live RHF multi-k
route + a static callsite/signature contract over every
`run_pbc_bipole_*` dispatch) and the strengthened forwarding assertion
in `tests/test_bz_integration_gilat_scf.py`. Full provenance:
BUG-PER-002 in `handovers/HANDOVER_OPEN_BUGS_V015.md`.

## `ImportError: cannot import name 'EEQOptions' from 'vibeqc._vibeqc_core'`

### Symptom

```text
ImportError: cannot import name 'EEQOptions' from
'vibeqc._vibeqc_core' (/path/to/_vibeqc_core.cpython-...so)
```

### Cause

You have a Python source tree from a newer commit but the
*compiled* `_vibeqc_core.so` is from an older one. Symbol mismatch.

### Fix

```sh
pip install -e . --no-build-isolation --force-reinstall
```

This rebuilds the C++ extension against the current Python sources.
On macOS the rebuild takes ~30-90 s; on Linux similar. If you're
on a worktree, [memory dictates a per-worktree venv](https://vibe-qc.com/)
so the parent venv doesn't get clobbered.

## `ModuleNotFoundError: No module named 'vibeqc'`

### Symptom

```text
ModuleNotFoundError: No module named 'vibeqc'
```

### Cause

You ran the script with the wrong Python, the system `python3`
instead of the venv-installed one.

### Fix

Either give the full path to the venv's interpreter:

```sh
~/path/to/vibe-qc/.venv/bin/python my_input.py
```

or activate the venv first:

```sh
source ~/path/to/vibe-qc/.venv/bin/activate
python my_input.py
```

## "Cell-list construction returned 0 cells"

### Symptom

```text
RuntimeError: cell-list construction returned 0 cells for cutoff
12.0 bohr on this lattice - check that the lattice matrix is full
rank and the cutoff is positive.
```

### Cause

Either a degenerate `lattice` (zero-rank column) or a non-positive
`lattice_opts.cutoff_bohr`. Common when a 1D-chain input has the
two vacuum axes accidentally set to 0 instead of a wide separation:

```python
# doc-audit: skip - deliberately broken diagnostic system with placeholder atoms
sysp = vq.PeriodicSystem(
    dim=1,
    lattice=np.diag([4.0, 0.0, 0.0]),    # <- bug - vacuum axes are 0
    unit_cell=[...],
)
```

### Fix

Set the non-periodic axes to a generous vacuum separation:

```python
# doc-audit: skip - corrected fragment reuses the diagnostic system's atoms
sysp = vq.PeriodicSystem(
    dim=1,
    lattice=np.diag([4.0, 30.0, 30.0]),  # 30 bohr vacuum decouples images
    unit_cell=[...],
)
```

## DFT / COSX wrong on a basis imported as Cartesian via `basis_toolkit` (v0.15.0 to v0.15.114)

### Symptom

A calculation on a basis built with
`vibeqc.basis_toolkit.to_libint_basis()` from an import whose
`harmonic_type` is Cartesian (the NWChem importer reads this from the
basis file, `basis "name" cartesian`; the g94 and CRYSTAL importers
accept it as an argument) produces wrong energies whenever the basis carries a
Cartesian shell with l >= 2. Measured on Be with an imported Cartesian d
shell, RKS/PBE: ~0.35 Ha over-binding and no SCF convergence. Named
bases (`vq.run_job(..., basis="cc-pvdz")` etc.) are completely
unaffected: that route forces spherical harmonics.

Affected surfaces by release:

* **v0.15.0 to v0.15.113**: every DFT energy, gradient, and Hessian on
  such a basis (the XC quadrature's AO evaluator applied a per-component
  normalization factor libint does not use), plus COSX exchange.
* **v0.15.114**: DFT quadrature fixed; COSX-only, and worse (2.4e-1
  relative error in K), because the evaluator half shipped one release
  before the COSX kernel half.

### Cause

Both AO evaluators and the COSX kernel multiplied mixed Cartesian
components by `sqrt((2l-1)!!/((2lx-1)!!(2ly-1)!!(2lz-1)!!))` on top of
libint's shell-wide axial normalization, which makes a d_xy density 3x
too large. Spherical shells were immune (the factor cancels through the
cart-to-pure transform), which is why ordinary named-basis coverage never
saw it. Full record: `handovers/HANDOVER_AO_CONVENTION.md` and the
CHANGELOG reclassification entry (2026-08-06).

### Fix

The evaluator half shipped in v0.15.114 (`a8130d61f`); the COSX kernel
half (`7fc952182`) is on `main` and ships with the next release. On
affected releases, work around it by importing the basis as spherical
(the g94 and CRYSTAL importers take `harmonic_type` as an argument; for
NWChem, change the file's `basis "name" cartesian` marker) or by using a
named basis. Regression pins: `tests/test_ao_convention_invariants.py`
(grid-vs-engine overlap parity, both shell kinds) and
`tests/test_cosx_cartesian_parity.py` (COSX vs exact-ERI exchange on a
Cartesian d shell).

## `vibeqc-cite: manifest missing` after a job

### Symptom

```text
$ vibeqc-cite output-h2o
error: manifest file not found: output-h2o.system
```

### Cause

Either the job hasn't finished writing the manifest (look for
`output-h2o.system.tmp`), or the path is wrong, or the run was
killed before the initial manifest landed.

### Fix

- Check `output-h2o.system` actually exists:
  `ls output-h2o.*`.
- Try the stem with the `.out` suffix, `vibeqc-cite output-h2o.out`
  also works (the CLI normalises via `.with_suffix(".system")`).
- For pre-v0.8.x runs (no manifest at all), use `vibeqc-cite`'s
  inability to find the manifest as a diagnostic, the run
  predates the citation surface. Re-run the SCF on a current
  build to get the bibliography.

## Native D4 dispersion (`backend="native"`) scope + accuracy

### What it is

`compute_d4(..., backend="native")` uses vibe-qc's own MPL-2.0 D4
implementation (no `dftd4` dependency). As of 2026-06-26 it is
**parity-validated for elements H-Ne**: its reference C6 (per-atom
Eq.-6 extraction over a correlated CPKS/PBE38 polarizability) agree
with the `dftd4` package to ~5 % (CH₄) / ~8 % (H₂O) and the CH₄-dimer
D4 energy to <0.05 kcal/mol. No warning is emitted. (The
`D4NativeExperimentalWarning` class is retained for back-compat but is
no longer raised -- earlier versions emitted it while the reference
data was being developed.)

### Limitation

The native reference catalogue covers **H, He, B, C, N, O, F, Ne**
only. `backend="native"` on a molecule containing any other element
raises a clear error (no reference data). For full periodic-table
coverage use the default `backend="dftd4"` (optional `dftd4` package,
bit-exact to upstream) -- every built-in `dispersion="d4"` route does.
Historical background is recorded in [Archived `HANDOVER_D4_NATIVE.md`](https://vibe-qc.com/docs/);
extending the catalogue past period 2 is the prerequisite for making
native the default backend.

## Things that are *not* errors

A handful of things look alarming but aren't:

- **`vibe-qc estimates this calculation will require ~218.4 GB`,
  `Proceeding (override)`**, you passed `memory_override=True` and
  the run is going through despite the estimate. Wait and see;
  the estimate's configured headroom plus the dense-ERI-tensor estimate
  is conservative.
- **`canonical_orth dropped 2 of 234 basis directions`** with a
  small number, vibe-qc reports any drop; 1-3 dropped basis
  directions out of hundreds is typical of a tight cell or large
  basis. Only worry if the drop exceeds ~5% of the basis count.
- **`# no citation route for basis 'foo'` in `.references`**, your
  basis isn't in the routing table. The job ran fine; the
  bibliography is just missing one entry that you'll need to add
  by hand. The fix is to extend
  [`database.toml`](user_guide/citations.md) in your next PR.
- **`Used 4 OpenMP threads`** even though you set `OMP_NUM_THREADS=16`
  - the system's actual core count (or the cgroup limit on a
  cluster node) wins over the env var. Sanity-check with `nproc`
  or `sysctl hw.physicalcpu` (macOS).

## BIPOLE absolute energies on tight ionic crystals (fixed, corrected gauge is the default at Γ and multi-k, all four drivers)

### Symptom

`jk_method="bipole"` total energies on dense ionic cells (MgO primitive
is the reference case) sit several hartree off converged references at
the same k-mesh, e.g. MgO/STO-3G RKS [2,2,2]: BIPOLE −266.1 Ha vs
PySCF-GDF −270.5 Ha (which independently agrees with converged CRYSTAL
to 5 mHa). The SCF *converges* cleanly, the stationary point's
absolute energy is what's off.

### Cause + status

Root-caused (2026-06-10, fixed-density component audit vs PySCF): (1)
the EXT EL-SPHEROPOLE term is a **double-count in vibe-qc's
explicit-jellium gauge**; (2) the Γ-locality density projection drops
real cross-cell contributions to T/V_ne/J on tight cells; (3) the
full-Coulomb direct exchange series is **formally divergent** with the
non-decaying finite-k-mesh Bloch density (any finite value is a cutoff
artefact).

**Fixed at Γ for all four drivers (RHF/RKS/UHF/UKS, on by default;
2026-06-10/11)** by the Ewald exchange split: ``K = K_SR(erfc ω,
direct) + K_LR(erf ω, reciprocal K≠0) + (ξ_M − π/(Vω²))·S·D·S``
(probe-charge Ewald / Madelung G=0 correction,
``exchange_exxdiv='ewald'``, PySCF-equivalent), full-Bloch density,
spheropole omitted. Validated out-of-process against PySCF GDF:
exchange element-wise to 8e-4 at the converged MgO density; H₂
12-bohr Γ to ~1 µHa (RHF) / ≤1 mHa (KS + open-shell variants). At
**multi-k** the corrected gauge is now the DEFAULT for all four
drivers too (option (b) Phase 3 + the Phase-5 flip, 2026-06-13):
the q = k−k′ LR-exchange channels + BvK-supercell Madelung
correction (α_HF-scaled for hybrid RKS/UKS, per spin for UHF/UKS),
validated on the H₂ box [2,1,1] vs PySCF (RHF +0.12 mHa at cutoff 7
/ −0.002 at 12, RKS SVWN/PBE0 +0.04/+0.06, UHF +0.03, UKS k-shift
0.013) and to +0.0001 mHa/cell against the explicitly-doubled
supercell at Γ. **Pass ``use_exchange_ewald_split=False`` for the
legacy gauge** (kept for the legacy-gauge analytic gradient + parity
diagnostics; it mis-states ionic absolute energies). The corrected
multi-k gauge needs a Monkhorst-Pack mesh; an ad-hoc k-list at
multi-k falls back to the legacy gauge under the auto default.

```{note}
**The S(Γ) fold-truncation preflight below is gauge-independent.** All
four BIPOLE drivers call the shared `pbc_bipole_common.
enforce_bipole_fold_support` unconditionally, hoisted above the gauge
branch, so switching to the legacy gauge above (or landing there via
the ad-hoc-k-list fallback) does **not** skip this safeguard — legacy
results record the measured `overlap_fold_drift` the same as the
corrected gauge, and the refusal message carries a suffix naming why
the guard applies regardless of gauge. (This was a real gap on earlier
`main` — the guard used to sit inside the corrected-gauge branch only
— fixed in `6299692ae`.)
```

### If your Γ ionic-crystal energies still look wrong

* **BIPOLE geometry optimisation is 3-D only.**
  `PeriodicSCFProvider` and the public periodic-job route refuse `dim=1` and
  `dim=2` before SCF. The low-level drivers expose `DIRECT_TRUNCATED` for
  diagnostics, but it is not a wire/slab Coulomb model and its compact-cell
  energy is cutoff-dependent. Raising the direct-space cutoff cannot make it
  a valid optimisation objective. For a 1-D wire use `jk_method='auto'` or
  `'gdf'`; for a 2-D surface use `'auto'`/`'slab_ewald_2d'`, or explicit GDF
  for its supported closed-shell slab envelope.
* **A refusal during a geometry optimisation is not always yours to fix.**
  The optimisers take trial steps, and a BFGS unit step in fractional
  coordinates can move an atom by a whole lattice vector, into a geometry
  the preflight legitimately refuses. Both periodic optimisers
  (`run_periodic_geomopt` and `bipole_optimize`) treat that as a recoverable
  penalty barrier and shrink the step; only a refusal at your *starting*
  geometry aborts, and that one really does mean the cutoff is too small.
  If an optimisation aborts with this message on its very first evaluation,
  raise the cutoff; if the message appears only in a trace of trial steps,
  nothing is wrong. `PeriodicSCFProvider` runs the same lattice sums as the
  BIPOLE drivers it wraps (15 bohr AO, 25 bohr nuclear, or your
  `scf_options.lattice_opts`); pass `cutoff_bohr=` only to tighten or loosen
  both sums deliberately.
* **Resolve the S(Γ) fold-truncation preflight before SCF.** The corrected
  gauge contracts full Bloch folds; diffuse AO tails (STO-3G Mg 3sp,
  outermost exponent ≈ 0.046) keep cross-cell overlaps alive past
  16 bohr. A drift above ``1e-2`` now fails before the expensive BIPOLE Fock
  build because that support can produce spurious metric-artifact states;
  raise ``bipole_cutoff_bohr`` until the drift is below ``1e-4`` for
  quantitative work. Moderate drift retains a truncation note. For example,
  MgO/STO-3G has ≈``5e-3`` drift at 12 bohr and needs at least 16 bohr for
  tight work; its 8-bohr drift is 0.39.
* **Check the SCF basin.** Minimal-basis ionic Γ SCF can have
  multiple solutions when the density is corrupted by an
  under-converged operator (the pre-v0.14 cross-cell XC bug era
  produced such basins). At a healthy density the fixture is benign:
  PySCF KRKS converges MgO/STO-3G to one basin from every standard
  guess (minao/1e/atom, Γ and [2,2,2], re-verified 2026-07-13), and
  the corrected-gauge BIPOLE RKS does the same. For absolute claims,
  warm-start from a reference density (``initial_density=``) or
  compare stationary points explicitly.
* **Keep KS smearing well BELOW the gap.** A minimal basis badly
  *underestimates* the gap (MgO/STO-3G: ~0.3 eV vs the ~7.8 eV
  experimental gap). A ``smearing_temperature`` whose window is
  comparable to that small gap fractionally occupies states across it
  and settles a **near-metallic basin hundreds of mHa off** the
  physical (gapped) solution, e.g. MgO RKS Γ at ``T = 0.01 Ha``
  (≈0.27 eV) landed +330 mHa with a 42 mHa entropy term (2026-06-13).
  The RKS/UKS drivers now emit a ``basin_warning`` (logged + on the
  result object) when finite-T smearing straddles the gap. The fix is
  a **sub-gap** smearing: reduce ``smearing_temperature`` until the
  ``basin_warning`` clears, or drop smearing once converged. For a gap
  as small as MgO/STO-3G's (where even ``0.002 Ha`` ≈ 0.05 eV still
  straddles, the converged gap there is only ~0.1-0.3 eV depending on
  the basin) that can mean ``≤ 0.001 Ha`` or no smearing at all.
  Usually no aid is needed at all: plain DIIS converges the MgO-class
  tight ionic cells monotonically in ~10 iterations (Gap-B validation
  2026-07-13), so the BIPOLE KS auto-FMIXING (30%) now engages only
  for DIIS-off runs, where it damps the bare Roothaan iteration
  (CRYSTAL-style). Smearing is for near-degeneracy, not as a blanket
  ionic-cell aid.

* **Keep the M5 SR image-domain default for absolute energies.** Every erfc
  short-range build now derives its absolute internal ket-image radius from
  `sr_image_precision=1e-6` and enables charge-pair Schwarz screening. This
  repairs the 515 mHa short-range Hartree loss measured on the MgO split
  fixture. `sr_image_precision=None` restores the old truncated behavior for
  reproducibility only and suppresses automatic Fock reduction on an active
  erfc SR route; it is not an accuracy setting. An explicit
  `sr_image_extent_bohr` overrides the precision-derived radius.

* **Pure semilocal RKS uses the padded SR+LR Hartree composition (2026-07-18).**
  The historical exact-FT Hartree default is retired: pure-functional RKS now
  routes its J through the same M5-padded SR+LR erfc composition with
  charge-pair Schwarz screening as every other route. This closes the
  exact-FT finite-ke reciprocal
  core-tail undercount (~0.5-1 Ha absolute on dense-core cells at the old
  ke=200 default). The analytic-FT builder remains available as the explicit
  ``use_exact_ft_j=True`` cross-check oracle on ``run_pbc_bipole_rks``; only
  that opt-in route still emits the dense-core reciprocal-tail warning and
  honors ``VIBEQC_J_EWALD3D_KE``.

### Remaining reference choices

Energy *differences* at fixed cell/density class (geometry scans,
relative energies on covalent/molecular-limit cells) are much less
affected, and H₂-class validation cells agree with references to
sub-mHa. For an independent implementation comparison, use the GDF
route and a converged reciprocal cutoff. Historical BIPOLE values
produced with the unpadded SR domain must be regenerated before
absolute comparison.

## Still stuck?

```
https://github.com/vibe-qc/vibe-qc/issues
```

When filing an issue, include:

1. The full error traceback.
2. The minimal Python script that reproduces it.
3. The `output-*.system` manifest (carries the hardware + library
   versions vibe-qc needs to reproduce).
4. The vibe-qc version (`python -c "from vibeqc import
   VIBEQC_VERSION; print(VIBEQC_VERSION)"`).

The maintainer or release chat will triage. For security
vulnerabilities follow
[SECURITY.md](https://github.com/vibe-qc/vibe-qc/blob/main/SECURITY.md)
instead of opening a public issue.
