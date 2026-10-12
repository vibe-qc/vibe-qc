(molecular_optimize)=
# Native Molecular Geometry Optimizer

See also the [tutorial](../tutorial/geometry_optimization.md) for the
ASE-driven BFGS walkthrough this page's native optimizer complements.

vibe-qc ships a standalone molecular geometry optimizer that needs **no
ASE**. It wraps analytic SCF nuclear gradients where they are validated and
uses central finite differences of the full energy for wavefunction methods
without an FD-tight analytic gradient. It is available both as a library
function and through `run_job`. A conservative steepest-descent + Brent
line-search backend is also available for systems where the Hessian
approximation misbehaves.

## Quick start

```python
from vibeqc import Molecule, Atom
from vibeqc.molecular_optimize import optimize_molecule

mol = Molecule([
    Atom(8, [ 0.00,  0.00,  0.00]),
    Atom(1, [ 0.00,  1.43, -0.98]),
    Atom(1, [ 0.00, -1.43, -0.98]),
])

result = optimize_molecule(
    mol,
    basis_name="def2-svp",
    method="rks",
    functional="PBE",
)
print(result.system)           # optimised Molecule (bohr)
print(result.energy)           # final energy (Ha)
print(result.n_iter)           # number of BFGS steps
print(result.converged)        # True / False
```

## Through `run_job`

Pass `optimizer_backend="native"` to bypass ASE:

```python
from vibeqc import run_job

run_job(
    mol,
    basis="def2-svp",
    method="rks",
    functional="PBE",
    optimize=True,
    optimizer_backend="native",   # ← no ASE needed
    output="h2o_opt",
)
```

The default `optimizer_backend="auto"` prefers ASE when installed
and falls back to the native path otherwise, so existing workflows
are unchanged.

## Backends

Three geometry-optimizer backends are available through
`optimizer_backend=` (both via `run_job` and directly):

| Backend | `optimizer_backend=` | How it works | Best for |
|---|---|---|---|
| ASE BFGS | `"ase"` (or `"auto"` when ASE installed) | ASE's `BFGSLineSearch`, quasi-Newton with Wolfe-condition line search | Routine use; fastest convergence near minimum |
| Native L-BFGS-B | `"native"` | scipy's L-BFGS-B, limited-memory BFGS with box constraints | No-ASE workflows; frozen atoms |
| Brent steepest-descent | `"brent"` | Steepest-descent direction + Brent 1-D line search per step | Flat/dispersion-bound PESs where Hessian extrapolation misbehaves |

**`fci`, `nevpt2`, `caspt2`, and `mrci` require `optimizer_backend="native"`
or `"brent"` explicitly.** The ASE calculator has no correlated-surface
route for these four (unlike `casci`/`casscf`, which it does route
correctly), so `optimizer_backend="ase"`, including the `"auto"` default
whenever ASE is installed, raises a `ValueError` rather than silently
optimizing the mean-field surface and reporting it under the requested
method (GitLab #357). Pass `optimizer_backend="native"` or `"brent"` to
reach the correlated FD surface described in
[Supported methods](#supported-methods) below; expect it to be
substantially slower than a mean-field optimization (minutes rather than
seconds even on a small molecule), since each FD gradient costs
`6 * n_atoms` correlated single points.

The Brent backend never takes an uphill step (each line search is a
rigorous 1-D minimisation) and needs no Hessian approximation.  Its
cost is higher per geometry step (multiple energy evaluations per line
search), but it is exceptionally robust on difficult surfaces.  Use it
when the quasi-Newton backends fail to converge.

```python
# Via run_job:
run_job(mol, basis="def2-svp", method="rks", functional="PBE",
        optimize=True, optimizer_backend="brent", output="h2o_brent")

# Direct API:
from vibeqc.molecular_optimize import optimize_molecule_brent
result = optimize_molecule_brent(mol, basis_name="def2-svp", method="rks",
                                  functional="PBE")
```

### When the optimizer does not converge

An optimization that stops short of the gradient threshold ends the job on
all three `optimizer_backend=` backends. The final single point, the
frequencies and the other post-SCF steps do not run, because they would
report the properties of a geometry that is not a stationary point.

For `"native"` and `"brent"` the `.out` closes the optimization with a
verdict. A converged run reads (H2O/STO-3G RHF from a stretched start)

```text
  Geometry optimization CONVERGED  (native L-BFGS-B, 4 iterations)
  final max|g| = 2.328e-05 Ha/bohr (threshold 9.723e-04)
  the largest gradient component is within the threshold

  Optimized geometry
```

and a run that did not converge reads

```text
  Geometry optimization NOT CONVERGED  (native L-BFGS-B, 3 iterations)
  final max|g| = 4.543e-03 Ha/bohr (threshold 9.723e-04)
  stopped at the step limit (3 of 3 steps)

  FATAL: geometry optimization (native L-BFGS-B) did not converge: stopped at the step limit (3 of 3 steps); final max|g| = 4.543e-03 Ha/bohr against the 9.723e-04 Ha/bohr threshold. The last geometry is in the output files.
  Last non-converged geometry written to h2o.xyz
```

`final max|g|` is the largest gradient component (Ha/bohr) at the last
geometry and the threshold is `fmax` converted from eV/Å. When scipy stops
for a reason other than the step limit, the line under it quotes that reason.

A run that did not converge also leaves the last geometry in the `.xyz`
(unless `write_xyz_file=False`), marks the manifest `crashed`, and records a
`geometry_optimization_nonconverged` event. `run_job` then raises:

| Backend | Error |
|---|---|
| `"ase"` | `RuntimeError`, "did not converge after N of M allowed steps"; the last geometry is also in the `.traj` |
| `"native"`, `"brent"` | `vibeqc.geomopt.GeomOptNotConvergedError`, a `RuntimeError` subclass |

`GeomOptNotConvergedError` carries the last geometry as `.system` and the
whole outcome, as a `vibeqc.geomopt.GeomOptResult` with the energy, gradient,
iteration count and trajectory, as `.result`:

```python
from vibeqc.geomopt import GeomOptNotConvergedError

try:
    run_job(mol, basis="def2-svp", method="rks", functional="PBE",
            optimize=True, optimizer_backend="native", max_opt_steps=20,
            output="h2o_opt")
except GeomOptNotConvergedError as err:
    print(err.result.gradient)       # flat (3N,) gradient, Ha/bohr
    restart_from = err.system        # last Molecule; continue from here
```

To get past a capped run, raise `max_opt_steps`, loosen `fmax`, restart from
the last geometry, or try another backend (the Brent backend above is slower
per step but takes no uphill step).

The library functions `optimize_molecule` and `optimize_molecule_brent` do not
raise. They return a `MolecularOptimizeResult` whose `converged` is `False`
and whose `message` says why it stopped, so a script that calls them directly
must check `converged` itself.

Releases before this change handled the ASE backend this way but not the
native and Brent backends: a capped run there wrote "Optimized geometry" and
went on to the single point and the frequencies as if it had converged.

## The native optimizer family (`geom_opt=`)

Beyond the three backends above, `run_job` exposes the uniform
`vibeqc.geomopt` framework through `geom_opt=`. Each optimizer walks
the same provider surface (SCF energy + analytic gradient, with
dispersion and solvation folded in when requested):

| `geom_opt=` | Algorithm | Target |
|---|---|---|
| `"sd"` | Steepest descent + line search | minimum |
| `"cg"` | Conjugate gradient | minimum |
| `"bfgs"` | BFGS quasi-Newton | minimum |
| `"lbfgs"` | Limited-memory BFGS | minimum |
| `"trust"` | Trust-region Newton | minimum or TS |
| `"rfo"` | Rational function optimization | minimum |
| `"gdiis"` | Geometry DIIS | minimum |
| `"fire"` | Fast inertial relaxation (FIRE) | minimum |
| `"ef"` | Eigenvector following | minimum or TS |
| `"prfo"` | Partitioned RFO | transition state |
| `"dimer"` | Dimer method (gradient-only saddle search) | transition state |

```python
run_job(mol, basis="def2-svp", method="rks", functional="PBE",
        optimize=True, geom_opt="bfgs", fmax=0.01,
        output="h2o_native_bfgs")

# Transition-state search (ef / prfo / dimer / trust):
run_job(mol, basis="def2-svp", method="rks", functional="PBE",
        optimize=True, geom_opt="ef", geom_target="transition_state",
        output="ts_search")
```

`geom_target="transition_state"` selects saddle-point mode;
`geom_opt="dimer"` *requires* it (a dimer search of a minimum is
refused with a ValueError). Per-optimizer tuning knobs pass through
`geom_opt_options={...}`, and the initial-Hessian policy through
`geom_hessian_init=` / `geom_hessian_update=`.

Every optimizer streams a per-step progress table into the job `.out`
while it runs, so long queue jobs can be monitored live:

```text
  step            E (Ha)           dE      max|g|      |step|  conv
  evaluating initial energy and gradient ...
     0      -74.96517778       --      1.9577e-02       --     gmax=1.958e-02 <= 4.500e-04 Ha/bohr [fail]
     1      -74.96565587   -4.781e-04  6.9972e-03   2.891e-02  gmax=6.997e-03 <= 4.500e-04 Ha/bohr [fail]
     2      -74.96579842   -1.426e-04  4.8506e-03   1.633e-02  gmax=4.851e-03 <= 4.500e-04 Ha/bohr [fail]
     3      -74.96589991   -1.015e-04  1.3428e-03   2.670e-02  gmax=1.343e-03 <= 4.500e-04 Ha/bohr [fail]
     4      -74.96590107   -1.152e-06  3.2652e-04   1.673e-03  gmax=3.265e-04 <= 4.500e-04 Ha/bohr [pass]
```

(H2O/STO-3G RHF, `geom_opt="bfgs"`.) The `conv` column shows each
active convergence criterion's current value, configured threshold, unit,
and ASCII `pass`/`fail` state. Multiple active criteria are separated by
semicolons; `rejected` markers identify trust-region step rejections. The
optimization stops once all active criteria pass.

When calling `run_geomopt` directly, `progress=True` also streams this trace
to the terminal. Pass a `ProgressLogger` for a caller-controlled live sink;
the persistent `.out` channel remains independent and no file handle is passed
through the optimizer.

Explicit MANTA RHF/UHF jobs also support `geom_opt` with independently selected
exact/RI Coulomb and exchange. Their geometry provider retains custom auxiliary
basis content and moves its centers with the owning atoms, including for the
final single point. This route currently uses HCORE at each geometry and refuses
unqualified derivative sources. See [selected HF geometry optimization](manta.md#selected-hf-geometry-optimization)
for configuration and resource limits.

### Coordinate systems (`geom_coords=`)

The native optimizer family walks Cartesian coordinates by default.
`geom_coords="dlc"` selects delocalised internal coordinates (bonds /
angles / torsions combined into a non-redundant set), which typically
converge in fewer steps for covalently bonded molecules:

```python
run_job(mol, basis="def2-svp", method="rks", functional="PBE",
        optimize=True, geom_opt="bfgs", geom_coords="dlc",
        output="h2o_dlc")
```

The DLC back-transform rebuilds the Wilson B-matrix pseudoinverse at
the current geometry each Newton iteration, wraps torsion residuals
onto the correct 2π branch, and seeds each step from the previous
geometry, so large steps and near-planar torsions (±π branch cut) stay
in the Newton convergence basin. The auto-generated primitive set does
not yet guarantee completeness for every topology: when it spans fewer
than the expected 3N−6 internal degrees of freedom, construction emits
a `RuntimeWarning` and the optimization may not fully converge; use
the default `geom_coords="cartesian"` for such systems.

Each step's SCF **warm-starts from the previous step's converged
density** (mean-field methods, gas phase): the provider hands the
prior result to the SCF as an `initial_guess=READ` restart, cutting
the per-step iteration count substantially since optimizer steps are
small. A warm-started SCF that fails to converge is retried once from
the cold default guess; a still-nonconverged SCF raises a clear error
instead of feeding an unconverged gradient to the optimizer. Opt out
(e.g. to reproduce older runs' iteration counts) with
`MolecularSCFProvider(..., warm_start=False)` when driving
`run_geomopt` directly.

For RKS and UKS, `MolecularSCFProvider` defaults to
`grid_level="orca-defgrid3"` and uses that same grid for the SCF energy
and analytic gradient. A caller-provided `rks_options=` or `uks_options=`
object keeps its own SCF grid when that grid was customised; an untouched
grid on it receives the `grid_level` preset (#663). A separate
`grid_options=` value remains an explicit analytic-gradient override. The
`optimize_molecule` and `optimize_molecule_brent` backends use the same
gradient-grid inheritance (the analytic gradient follows the SCF options'
grid). Both backends now accept `grid_level="orca-defgrid3"` and apply it
to absent or untouched KS options. A custom grid wins; pass
`grid_level="legacy"` for the old defaults. ROKS options and the selected
level reach every optimizer energy and finite-difference displacement,
including the ASE backend. Molecular NEB, dimer and IRC entry points accept
the same `grid_level` selection.

## Supported methods

| Method | Gradient | Notes |
|---|---|---|
| `rhf`, `uhf` | Analytic | All-electron, closed- and open-shell |
| `rks`, `uks` | Analytic / Central FD | Analytic for LDA, GGA, meta-GGA and global hybrids. Range-separated hybrids (HSE06, ωB97X, CAM-B3LYP, ωB97X-V, ...) and VV10-paired functionals (VV10, ωB97X-V, ωB97M-V) fall back to full-energy central FD because the analytic kernel omits their long-range exchange and VV10 terms (GitLab #571; see below) |
| `selected_ci`, `dmrg`, `v2rdm`, `transcorrelated_ci` | Central FD | 2-point finite difference on energy (`fd_step_bohr=0.005`) |
| `casci` | Central FD | 2-point FD on energy |
| `casscf` | Analytic / Central FD | Validated analytic gradient for state-specific, closed-shell CASSCF (v0.15.0); SA-CASSCF, open-shell, and `compute_wz="numerical"` fall back to full-energy FD (see the note below) |
| `caspt2` | Analytic / Central FD | Analytic IC-CASPT2 Lagrangian for state-specific closed-shell, unshifted, explicit-engine CASSCF references with `compute_corr_grad=True`; unsupported variants use the optimizer-owned full-energy FD fallback (see below) |
| `nevpt2` | Central FD | Relaxed full-energy central FD for CASSCF-referenced, gas-phase runs with `compute_corr_grad=True`; otherwise the optimizer owns the same full-energy FD fallback |
| `fci` | Central FD | 2-point FD on energy; full-valence unless `active_space=` truncates it |
| `mrci` | Central FD | 2-point FD on energy; CASCI or (with `casscf_options=`) orbital-optimized CASSCF reference |

`fci` and `mrci` reach `run_job(optimize=True)` only through
`optimizer_backend="native"` or `"brent"`; see the note above the
backends table.

**Range-separated and VV10 functionals walk the finite-difference
surface (GitLab #571).** The analytic RKS / UKS gradient kernel carries
exact exchange through the global fraction only and has no VV10 term, so
for HSE06, ωB97X, ωB97X-D, CAM-B3LYP, VV10, ωB97X-V, ωB97M-V and every other
functional for which `vibeqc.functional_gradient_terms_missing(name)` is
non-empty, all optimizer backends (ASE/BFGS, native L-BFGS-B, Brent, and
the `geom_opt=` family through `MolecularSCFProvider`) differentiate the
full SCF energy by central differences (`fd_step_bohr=0.005`) instead.
They share one decision (`MolecularSCFProvider.has_analytic_gradient` is
`False` and `.missing_gradient_terms` names the terms), so all of them
walk the same surface, and the `.out` carries one `Note:` line naming the
omitted terms and the FD step. Before this fix the optimizers used the
partial analytic gradient silently: on a bent O/H/H molecule in STO-3G it
sits 5.6e-3 (HSE06), 7.0e-4 (VV10) and 5.6e-2 (ωB97X-V) Ha/bohr off the
FD gradient, against a 2.1e-6 PBE noise floor. Frequencies for these
functionals fail closed (the FD Hessian finite-differences the analytic
gradient).

Dispersion corrections (D3-BJ) and implicit solvation (CPCM) are folded
into the energy **and the gradient** on every backend, so a solvated
optimization walks the solvated surface rather than the gas-phase one.

Releases before this fix did not: the ASE backend -- the default whenever
ASE is installed -- was handed no `solvent=` at all, and the native and
`geom_opt=` routes solvated the energy while differentiating the
gas-phase surface. On LiF/STO-3G in water, whose bond the reaction field
lengthens by 0.025 bohr, the default backend stopped 9e-06 bohr from the
*gas* minimum and the other two between the surfaces; the solvated
gradient at those reported geometries was 1.7x to 6.2x the run's own
convergence threshold. The energy printed was solvated throughout, so
nothing in the output said the geometry was not.

For a functional whose analytic gradient omits terms (the
`functional_gradient_terms_missing` list above), a solvated run takes the
same full-energy finite-difference route, differentiating the *solvated*
energy; `cpcm_gradient` refuses those functionals rather than adding a
reaction field to an incomplete gas-phase kernel.

Implicit solvation composes with the mean-field methods only
(`rhf` / `uhf` / `rks` / `uks`); requesting `solvent=...` with the
CAS-family methods or `rohf` raises a clear `ValueError` (there is
no CPCM composition for those methods, and silently optimizing the
gas-phase surface instead would misreport the result). For the same
reason `optimize=True` with `solvent=` and `method="msindo"` is refused:
MSINDO's COSMO route has no nuclear gradient, so its walk would follow
the gas-phase surface. Optimize it in gas phase, or use a mean-field
method with CPCM.

On **both** backends every per-step energy evaluation receives the
same solver options as the final single point, `active_space`,
`cas_reference`, and the wavefunction option structs
(`selected_ci_options`, `dmrg_options`, `v2rdm_options`,
`transcorrelated_options`, `casci_options`, `casscf_options`,
`caspt2_options`, `nevpt2_options`), so an
SA-CASSCF optimization (`casscf_options=CASSCFOptions(nroots=2)`)
walks the state-averaged surface its final energy is reported on, and
a `selected_ci` optimization keeps its active-space truncation at
every FD displacement instead of falling back to full-space CI.

As of v0.15.0 the state-specific CASSCF analytic gradient is a validated
full-energy derivative (`examples/regression/casscf_gradient_fd_reproducer.py`
passes to ~1e-7 Ha/bohr), and all three molecular backends (the L-BFGS-B
primary, the Brent backend `optimizer_backend="brent"`, and the uniform
`geom_opt` framework) use it for state-specific, closed-shell, default-
`compute_wz` CASSCF. They share one decision so they always walk the same
surface. Outside that validated envelope they fall back to full-energy central
FD: state-averaged CASSCF (`casscf_options=CASSCFOptions(nroots=2)`, whose
analytic gradient is only finiteness/translational-invariance checked),
open-shell CASSCF (the kernel is the closed-shell RHF formalism), and an
explicit `compute_wz="numerical"` request. `compute_wz=True` is a no-op
alias of the analytic gradient since the experimental W^z correction was
retired (GitLab #516) and stays on the analytic path.

CASPT2 optimizations use the runner-supplied analytic IC-CASPT2 Lagrangian
gradient when the run requests and can produce it: a state-specific,
closed-shell exact-CI CASSCF reference, gas phase, the unshifted explicit
small-space engine, and `compute_corr_grad=True`. IPEA/imaginary shifts,
`engine="cases"`, direct-large, selected-reference, and open-shell variants
stay on the optimizer's full-energy central FD route. MS/XMS-CASPT2 keeps its
separate SA-CASSCF analytic response route. NEVPT2 continues to use the
runner-supplied relaxed full-energy central FD when CASSCF-referenced, gas
phase, and opted in.

With the default `compute_corr_grad=False` the solver only computes the bare
CASSCF reference gradient, so every optimizer falls back to its outer
full-energy central FD rather than walking a surface inconsistent with the
reported PT2 energy. All three backends share the same capability decision.

```python
from vibeqc.solvers import CASPT2Options, CASSCFOptions

result = optimize_molecule(
    mol,
    basis_name="6-31g",
    method="caspt2",
    active_space=(2, 2),
    casscf_options=CASSCFOptions(),
    caspt2_options=CASPT2Options(compute_corr_grad=True),  # analytic IC gradient
)
```

## Trajectory collection

Set `record_trajectory=True` (the default) to collect per-step
geometries and energies:

```python
result = optimize_molecule(mol, basis_name="sto-3g", method="rhf")
for i, (frame, e) in enumerate(
    zip(result.trajectory_frames, result.trajectory_energies)
):
    print(f"  step {i}: E = {e:.8f} Ha")
```

When used through `run_job(optimize=True, output_qvf=True)`, the
trajectory is embedded in the QVF archive for vibe-view's animation
player, identical behaviour to the ASE backend.

## Convergence control

| Parameter | Default | Meaning |
|---|---|---|
| `conv_tol_grad` | `4.5e-4` | Gradient norm convergence (Ha/bohr) |
| `conv_tol_energy` | `1e-6` | Energy change tolerance (Ha) |
| `max_iter` | `100` | Maximum steps |

For DFT jobs where the SCF may struggle at intermediate geometries
(common with PBE + minimal basis sets), pass the appropriate
`rks_options` / `uks_options` with increased `max_iter`:

```python
from vibeqc import RKSOptions

rks_opts = RKSOptions()
rks_opts.max_iter = 80
rks_opts.use_diis = True

result = optimize_molecule(
    mol, basis_name="sto-3g", method="rks", functional="PBE",
    rks_options=rks_opts,
)
```

## API reference

```{eval-rst}
.. autoclass:: vibeqc.molecular_optimize.MolecularOptimizeResult
   :members:

.. autofunction:: vibeqc.molecular_optimize.optimize_molecule

.. autofunction:: vibeqc.molecular_optimize.optimize_molecule_brent

.. autofunction:: vibeqc.molecular_optimize.brent_minimize_1d
```
