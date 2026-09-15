# SCF convergence

vibe-qc's SCF surface is uniform across the four molecular drivers
(`run_rhf` / `run_uhf` / `run_rks` / `run_uks`). The periodic drivers
(`run_rhf_periodic_gamma`, multi-k variants) expose the same option
fields, and every Python periodic backend (Γ-Ewald, multi-k Ewald,
BIPOLE, GDF) now implements the full accelerator family + dynamic
damping (the rollout completed with the GDF M4 milestone); the only
remaining gap is C++ multi-k KDIIS. See
[SCF accelerators](#scf-accelerators-pick-one) below for the exact
coverage matrix. For the `solver=` keyword itself (Davidson, LOBPCG,
dense, and the experimental Jacobi-Davidson/GPLHR eigensolvers this
page assumes), see the [solver framework](solver_framework.md). The
user-facing selection knobs split into four groups:

* **Fock-build mode**, how the two-electron matrices are constructed
  each iteration (`opts.scf_mode`). Default is `SCFMode.AUTO`, which
  picks between in-core CONVENTIONAL and on-the-fly screened DIRECT
  based on basis-set size. See [SCF Fock-build modes](scf_modes.md).
* **Initial guess**, where the SCF starts from
  (`opts.initial_guess`). Default is `InitialGuess.AUTO`, which
  routes through the unified [`GuessEngine`](initial_guess.md):
  closed-shell light-atom molecular → SAP, open-shell /
  transition-metal / any periodic → SAD, with HCORE / PATOM /
  HUECKEL / MINAO / READ available as explicit selections. The
  richer story (theory, math, when to use which, references) lives
  in [Initial guesses](initial_guess.md); the brief summary here is
  enough to pick a knob.
* **Convergence aids**, composable knobs that stack freely
  (`damping`, `dynamic_damping`, `fock_mixing`, `level_shift`,
  `smearing_temperature` for periodic).
* **SCF accelerators**, mutually exclusive choices for how the Fock
  matrix is extrapolated each iteration (`scf_accelerator`).
* **Second-order finalizers**, mutually exclusive opt-in choices
  for how the asymptotic regime is closed out (`newton_threshold`,
  `soscf_threshold`, `trah_threshold`, `quadratic_fallback_iter`).

Quick example exercising every group:

```python
import vibeqc as vq
from vibeqc import RHFOptions

opts = RHFOptions()

# Stopping criteria
opts.max_iter = 100
opts.conv_tol_energy = 1e-8
opts.conv_tol_grad = 1e-6

# Initial guess (see docs/user_guide/initial_guess.md for details)
opts.initial_guess = vq.InitialGuess.AUTO     # AUTO / SAP / SAD / HCORE / PATOM / HUECKEL / MINAO / READ

# Convergence aids - compose freely
opts.damping = 0.5                            # initial density mixing α
opts.dynamic_damping = True                   # molecular default; adaptive α (Z-H 1979)
opts.fock_mixing = 0.0                        # CRYSTAL FMIXING
opts.level_shift = 0.0                        # Saunders-Hillier

# SCF accelerator - pick one (molecular default: EDIIS_DIIS)
opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS  # default; or DIIS / KDIIS / EDIIS / ADIIS

# Second-order finalizer - opt in for the asymptotic regime
opts.newton_threshold = 0.0                   # D2c full-Hessian Newton (off)
opts.soscf_threshold = 0.0                    # D2d Neese SOSCF (opt-in L-BFGS finalizer; off by default)
opts.trah_threshold = 0.0                     # D2e TRAH (trust-region AH, off)
opts.use_davidson = False                       # default; enable for large basis
```

All of these options exist on `UHFOptions`, `RKSOptions`, `UKSOptions`,
`PeriodicRHFOptions`, `PeriodicSCFOptions`, `PeriodicKSOptions`. The
auto-reducing level-shift controls (`level_shift_warmup_cycles` and
`level_shift_schedule`) are now uniform across all molecular and
periodic option structs; the periodic options additionally expose
`smearing_temperature` for metals / small-gap insulators.

---

## Initial guess

The unified `GuessEngine` (v0.9.x) dispatches every initial-guess
request to a concrete builder. Periodic driver options
(`PeriodicRHFOptions`, `PeriodicSCFOptions`, `PeriodicKSOptions`)
default to `InitialGuess.AUTO`, which inspects the system and picks
the right guess. Molecular driver options (`RHFOptions`, `UHFOptions`,
`RKSOptions`, `UKSOptions`) default to `InitialGuess.PATOM` (SAD density
plus one in-field re-polarisation step, a better molecular default).

| `opts.initial_guess` | Algorithm | Status | When to use |
|---|---|---|---|
| `InitialGuess.AUTO` | `GuessEngine::resolve_auto` picks per the table below | ✅ shipped | **Default**, let the engine decide. |
| `InitialGuess.SAD` | Superposition of atomic densities, fractional-occupation atomic SCF per unique element, summed at AO indices | ✅ shipped | Open-shell, transition / f-block atoms, any periodic system. AUTO routes here. |
| `InitialGuess.SAP` | Superposition of atomic potentials (Lehtola/Visscher/Engel 2020), sum tabulated atomic effective potentials, diagonalise `T + V_SAP` | ✅ shipped (v0.9.x) | Closed-shell molecules with light atoms. AUTO routes here. Cheaper than SAD (no per-element atomic SCF). |
| `InitialGuess.HCORE` | Diagonalise H_core; build D from occupied MOs | ✅ shipped | Diagnostic / back-compat. Slow on most systems; known to lock onto false minima for some open-shell cases (OH·/6-31G\*). |
| `InitialGuess.PATOM` | SAD density plus one in-field re-polarisation step (ORCA PAtom) | ✅ shipped (v0.9.x) | Transition-metal d-shell ordering; molecular only. |
| `InitialGuess.HUECKEL` | Parameter-free generalised Wolfsberg-Helmholz over computed AO energies | ✅ shipped (v0.9.x) | A shell-correct start with no atomic SCF; molecular only. |
| `InitialGuess.MINAO` | Free-atom ANO-RCC minimal-basis densities projected onto the target basis | ✅ shipped (v0.9.x) | Light cost; molecular only. |
| `InitialGuess.READ` | Restart from a prior result / `.qvf` / `.molden`, projected across basis/geometry | ✅ shipped (v0.9.x) | Geometry scans, NEB, continuing a job; molecular only. See [initial_guess](initial_guess.md). |

**AUTO dispatch table** (`GuessEngine::resolve_auto`):

| Hint                                 | AUTO resolves to |
|--------------------------------------|------------------|
| periodic, any                        | **SAD**          |
| molecular, open-shell                | **SAD**          |
| molecular, transition / f-block atom | **SAD**          |
| molecular, closed-shell, light atoms | **SAP**          |

```python
opts.initial_guess = vq.InitialGuess.AUTO    # default; engine picks
opts.initial_guess = vq.InitialGuess.SAD     # explicit SAD
opts.initial_guess = vq.InitialGuess.SAP     # explicit SAP (closed-shell light atoms)
opts.initial_guess = vq.InitialGuess.HCORE   # diagnostic / back-compat
```

The v0.6.2 periodic-HCORE calibration freeze was lifted in v0.9.x:
periodic drivers now default to `AUTO → SAD` for any periodic system
(fixes the NaCl/MgO bombing class of failure).

Where the SCF has a single stationary point, the guess changes only the
iteration count and the trace shape, not the converged energy. **Do not
read that as a guarantee.** Where the surface has several stationary
points, the guess selects which one you land on, and open-shell
transition metals are the standard case: on FeCl₃ / ROHF / def2-TZVP the
shipped SAD guess converges 0.520 Ha above the correct high-spin d⁵
state, while ORCA's model-potential guess starts in the right basin. See
[Convergence is not correctness](#convergence-is-not-correctness).

---

## Convergence aids, compose freely

### Periodic jobs: automatic strategy (`convergence="auto"`)

For periodic calculations, `run_periodic_job` selects these aids
**automatically by default** for `jk_method="bipole"` and
`jk_method="gdf"`: a cheap pre-SCF classifier profiles the system as
ionic-insulator (tight cell, large electronegativity spread, e.g.
MgO/NaCl), covalent-insulator (diamond/Si), metallic-candidate
(all-metal composition), or molecular-limit (vacuum-padded box), and
fills the knobs you did *not* set. Ionic KS cells get FMIXING 30 %
with integer occupations, metallic candidates get FMIXING 50 % plus
smearing (KS) or a level shift (HF), and molecular-limit and covalent
cells stay plain.

The `.out` file always states what happened in a "Convergence
strategy" block: `AUTO (default)` / `AUTO (requested)` / `manual` /
`off`, with the classification evidence and a `[auto]`/`[explicit]`
provenance tag + reason per knob. **Any knob you set explicitly is
never overridden**, and setting any knob (without
`convergence="auto"`) switches the whole job to manual mode. Pass
`convergence="off"` for today's plain defaults. For the GDF level-shift
and Fock-mixing exceptions described below, automatic choices are
checked against the selected driver's capabilities before dispatch:
an unsupported automatic value is set to zero with a
`capability-filtered` reason in this block, while an explicit nonzero
request for the same unsupported aid fails closed with an actionable
error.

```python
run_periodic_job(system, basis, method="RKS", functional="svwn",
                 jk_method="bipole", kpoints=(2, 2, 2))
# .out ->  Convergence strategy
#         convergence strategy: AUTO (default: ...)
#           profile: ionic-insulator
#             - tight 3D cell (126 bohr³, 63 bohr³/atom) with large
#               Pauling electronegativity spread delta-chi = 2.13 - ionic ...
#           smearing_temperature = 0   [auto] integer occupations ...
#           fock_mixing = 0.3   [auto] CRYSTAL-style FMIXING 30% ...
```


These are independent knobs. Set any subset; they all stack on top of
whichever SCF accelerator and finalizer you've selected.

### Static damping

Linear density mixing before the next Fock build:

```
D_used = α · D_prev + (1 − α) · D_new
```

```python
opts.damping = 0.5      # default; 0.0 disables
```

Skipped automatically once DIIS / KDIIS / EDIIS / EDIIS+DIIS takes
over (the accelerator does the same job better). Skipped during
the second-order phases (Newton / SOSCF / quadratic_fallback).

### Dynamic damping (Zerner-Hehenberger 1979)

Adapts α iteration-by-iteration based on the energy decrease:
increased toward `dynamic_damping_max` when the energy oscillates
upward, decreased toward `dynamic_damping_min` when the energy is
monotonically decreasing.

```python
opts.damping = 0.5               # initial α
opts.dynamic_damping = True      # adapt over iterations
opts.dynamic_damping_min = 0.0   # default; lower bound
opts.dynamic_damping_max = 0.95  # default; upper bound
```

When false (default), `damping` is held constant, bit-for-bit
back-compat with the pre-dynamic-damping path.

`ROHFOptions` / `ROKSOptions` default `dynamic_damping` to `False`
where the C++ drivers default it to `True`; the Roothaan loop also
skips density mixing entirely once DIIS is extrapolating, which with
the default `diis_start_iter = 1` is from the second iteration on. To
give damping room on a restricted-open case, push DIIS back
(`diis_start_iter = 11`). But read the ROHF warning under
[Diagnosing non-convergence](#diagnosing-non-convergence) first: on
high-spin transition-metal ROHF a damping-first schedule can converge
cleanly onto a *higher* solution.

### CRYSTAL FMIXING, static Fock-matrix mixing

```
F_diag = (1 − α_F) · F_current + α_F · F_previous
```

Independent of density damping. Stabilises the residual that
density damping leaves oscillating on tight ionic crystals.

```python
opts.fock_mixing = 0.0    # default; 0.30 mirrors CRYSTAL FMIXING 30
```

### Saunders-Hillier level shift

Adds `b · (S − w · S D S)` to F before diagonalisation, raising
virtual-orbital eigenvalues by `b` Hartree while leaving occupied
orbitals untouched. Inert at the converged density (the SCF fixed
point doesn't change), only the iteration dynamics do. Useful when
DIIS oscillates between near-degenerate occupied/virtual swaps on
small-HOMO-LUMO-gap insulators or transition-metal complexes.

The weight `w` is fixed by the density convention, not by taste:
`w = ½` for a closed-shell total density `D = 2P` (occupations in
{0, 2}), and `w = 1` for a single-spin density `D_σ` (occupations in
{0, 1}), which is idempotent in the overlap metric so that `S D_σ S`
is already the occupied-space projector. Both leave the occupied
eigenvalues exactly where they were and raise every virtual by
exactly `b`. Every driver, molecular and periodic, applies the shift
through the one shared operator `vibeqc.apply_level_shift` (or its
per-k Hermitian sibling `apply_level_shift_k`), selecting the weight
with `LevelShiftDensity.SPIN` / `LevelShiftDensity.TOTAL`; no driver
open-codes the expression.

```python
opts.level_shift = 0.0    # default; typical values: 0.1 - 0.5 Hartree
```

The same auto-reducing controls are available on **every** molecular
and periodic driver (RHF / UHF / RKS / UKS, ROHF / ROKS, and their
periodic counterparts), with identical names and semantics. A large
shift helps the opening iterations but slows the last-mile approach,
so vibe-qc releases it automatically rather than holding it forever.

The shift never reaches the **reported** orbital energies. Whichever
schedule is in force, the canonicalising diagonalisation that produces
`result.mo_energies` / `mo_coeffs` runs unshifted, so the orbital
table, the HOMO-LUMO gap and every consumer downstream of them (
Koopmans estimates, semicanonical ROHF-MP2 / ROHF-CCSD, `.molden`)
describe the converged Fock rather than the converged Fock plus `b`.
Unconverged runs are re-solved unshifted too: a diagnostic orbital
table has to show the Fock actually reached.

The shift is inert *at* a fixed point, but it is not neutral about
*which* fixed point you reach. A held shift reorders the frontier during
the approach and can walk the SCF into a different basin: FeCl₃ / ROHF /
def2-TZVP converges to −2640.6035 Ha unshifted and heads somewhere
below −2641.02 Ha under a persistent 0.3 Ha shift. Treat "same method,
same geometry, different shift" as "possibly a different state" and
compare energies and gaps.

**Warm-up then release** (`level_shift_warmup_cycles`):

```python
opts.level_shift = 0.3
opts.level_shift_warmup_cycles = -1   # default: auto
```

* `-1` (default): auto, shift the first few startup cycles (up to
  five), then release, always leaving at least one unshifted tail
  cycle so the final energy is the true unshifted fixed point.
* `0`: persistent, hold the shift at every iteration (the legacy
  constant-shift behaviour).
* `N > 0`: shift iterations `1..N`, then release.

**Explicit decay schedule** (`level_shift_schedule`, CRYSTAL
`LEVSHIFT B IRESET` style), a prescribed per-iteration curve that
supersedes `level_shift` + `level_shift_warmup_cycles` when set. Entry
`i` is the shift at iteration `i+1`; the last entry is reused for later
iterations (set it to `0.0` to release). Build one with the
`LevelShiftSchedule` helper and lower it onto the options struct:

```python
from vibeqc import LevelShiftSchedule
opts = LevelShiftSchedule.crystal_default().apply_to(vq.RHFOptions())
# [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]: decays to zero over seven iters
```

`LevelShiftSchedule` also offers `crystal_aggressive()` (heavier
opening, slower release for wide-gap ionic crystals like NaCl / LiF)
and `constant(b)`. Both the warm-up logic and the schedule are resolved
by one shared helper, `vibeqc.level_shift_at_iter`, so a given
`(level_shift, warmup, schedule)` triple behaves identically on the
molecular and periodic paths.

**Backend coverage.** The four molecular drivers, the shared Roothaan
ROHF / ROKS loop, the C++ direct-truncated periodic drivers, Γ-Ewald,
multi-k Ewald, and BIPOLE honour the shift. The C++ periodic form is applied per k as
`F(k) + b·S(k) − (b/2)·S(k)·P(k)·S(k)`. GDF coverage is
route-dependent: closed-shell non-Γ GDF and the closed-shell legacy Γ
fallback honour the shift, while pure exact-Γ GDF selected by an
explicit `gdf_method` and open-shell GDF do not. On those unsupported
routes AUTO filters its selected shift to zero and records why;
an explicit nonzero shift fails closed.

GDF Fock-mixing coverage is also route-dependent. Closed-shell
non-Γ GDF, the legacy closed-shell Γ fallback, and the open-shell
`kpoints=None` Γ drivers honour it. Exact-Γ RHF/GDF with an explicit
`gdf_method` and open-shell GDF jobs with an explicit k-mesh do not;
AUTO filters the value to zero and an explicit nonzero request fails
closed. Before v0.15.x the C++ periodic drivers exposed level-shift
fields but silently ignored them; that is fixed and regression-guarded
in `tests/test_periodic_level_shift.py`. The shift costs nothing when
it is not active: the `S·D·S` products are skipped entirely on any
iteration whose resolved shift is `0.0`, which is every iteration once
the warm-up releases.

### Smearing (periodic only)

Fermi-Dirac fractional occupations for metals / small-gap insulators.
Adds an electronic-entropy contribution to the free energy.

```python
opts.smearing_temperature = 0.0    # default; 0.001 - 0.02 Hartree typical
```

### Broken-symmetry hold (`SPINLOCK`, molecular UHF / UKS)

For open-shell magnetic systems, `opts.spinlock_mode` holds a
broken-symmetry spin pattern through the early iterations so the SCF
does not collapse to the symmetric solution. `PATTERN_HOLD` keeps a
seeded occupation in place by maximum overlap (pair it with
`atomic_spins`) and suspends the SCF accelerator for the hold window:
extrapolating across held iterates would rotate the occupied orbitals
toward the symmetric solution, a collapse the occupation-selecting hold
cannot prevent; `SPIN_SCHEDULE` locks the alpha/beta count difference
for `spinlock_iterations` cycles, then releases (CRYSTAL `SPINLOCK n
nstep`). Full API and examples are in the
[initial-guess guide](initial_guess.md), under "Multiplicity and the
initial guess".

---

## Internal stability analysis (molecular UHF/UKS, default on)

A converged SCF certifies only a *stationary point* of the energy; it
can still be a saddle point -- an excited SCF solution -- while every
convergence diagnostic looks clean. The character of the extremum is
decided by the lowest eigenvalue of the orbital-rotation Hessian: if it
is negative, rotating along the corresponding eigenvector lowers the
energy further (Lehtola 2020, Sec. 10; Seeger & Pople 1977). The
measured trigger for this feature was O2 (charge 0, multiplicity 5,
cc-pVDZ), where the default guess converged cleanly onto a saddle
**59 mHa above the ROHF energy of the same system** -- violating the
variational identity E(UHF) <= E(ROHF) -- while the true UHF minimum
was 17 mHa *below* ROHF.

After every molecular UHF convergence, and every molecular UKS convergence
whose complete orbital-Hessian response is supported, vibe-qc therefore
computes the lowest eigenvalue of the internal (real UHF -> UHF)
orbital-rotation Hessian matrix-free (Davidson iteration; one J + two K builds
per matvec). If the solution is internally unstable, the driver
line-searches the energy along the unstable rotation and restarts the
SCF from the rotated densities, holding the rotated occupation pattern
by fixed-reference maximum overlap for the first cycles (the plain
density-only restart is measurably recaptured by the original basin),
then re-checks -- up to `stability_max_retries` times. A restart that
runs out of iterations while still descending is chained into a few
fresh-DIIS restarts from its final densities before the driver gives
up: a fresh accelerator history started closer to the minimum converges
where one saturated history wanders. The `.out` records the verdict next to
the convergence line:

```
  converged in 15 iterations; E = -149.1799299198 Ha
  internal stability: stable after 1 corrective restart(s) -- the first
  SCF converged onto an excited solution (lowest Hessian eigenvalue ...)
```

If an instability cannot be escaped, the result keeps its converged
energy but the `.out` and a structured warning
(`scf_internal_instability`) say loudly that the energy is an excited
SCF solution.

The per-solve matvec cap remains user-bounded. Its default is 80 for UHF and
60 for UKS: the tighter residual needed to certify the eigenvalue sign takes
more than 64 products for FeCl3/cc-pVDZ. Exhausting the cap never means
"stable"; `stability_analysis_converged` remains false and the output reports
the solution character as unverified.

## Restricted-stability verdict (RHF/RKS, issue #144)

Closed-shell `run_rhf` / `run_rks` now run the same post-convergence
analysis, but as a **verdict only**: the lowest orbital-rotation Hessian
eigenvalue at the converged restricted solution is computed (the
Seeger-Pople singlet and triplet sectors; for KS the closed-shell
singlet/triplet fxc sectors of Bauernschmitt-Ahlrichs 1996) and a
negative verdict below `stability_tol` is reported **loudly** -
`result.internal_instability = True`, the negative
`result.stability_eigenvalue`, and the same
`scf_internal_instability` warning in the `.out`. The restricted route
does **not** rotate, escape, promote to UHF/UKS, or search other basins:
the policy for what to do with a negative verdict is still being
decided (see `agentic-loop/asks/ask-scf144-restricted-stability-contract-2026-08-27.md`),
and until then the verdict makes the wrong-root failure visible instead
of silent.

```python
opts = vq.RHFOptions()
opts.stability_check = True        # default; False = legacy single SCF
opts.stability_tol = 1e-4
opts.stability_davidson_max_iter = 60
res = vq.run_rhf(mol, basis, opts)
res.stability_checked            # analysis ran
res.stability_eigenvalue         # lowest Hessian eigenvalue
res.internal_instability         # True = excited restricted solution
```

Caveats:

* The check costs roughly one extra SCF's worth of Fock builds
  (typically 10-40 matvecs) even when the solution is already stable.
  Disable it with `stability_check = False` when throughput matters
  more than the guarantee.
* Deliberate state-targeting runs (`atomic_spins` broken-symmetry
  seeds, `spinlock_mode`) are **checked but never rotated away** from
  the state you asked for; the eigenvalue still lands on the result
  (`result.stability_eigenvalue`, `result.internal_instability`).
* UKS corrective following uses the complete KS energy, including
  exchange-correlation, and reconverges both signs of an internal unstable
  mode. Each signed branch first uses a fixed maximum-overlap hold, with an
  ordinary Aufbau retry if that hold does not yield a converged descent. The
  complete response is currently available for LDA, GGA, and their global
  hybrids on direct or density-fitted JK surfaces.
* DFT+U UHF jobs skip the check because the +U response is not part of the
  Hessian action. UKS meta-GGA, range-separated, VV10, DFT+U, active hybrid
  RIJCOSX, and coupled-solvent jobs skip the implicit automatic check until
  their missing response terms are available; their result has
  `stability_checked = False`. Explicitly assigning
  `uks_options.stability_check = True` fails closed instead of hiding an
  incomplete requested check. ROHF/ROKS do not run stability analysis yet.
* RHF/RKS restricted verdicts share the same fail-closed contract: DFT+U
  and hybrid-COSX RHF jobs, and DFT+U, meta-GGA, range-separated, VV10,
  and hybrid-COSX RKS jobs, skip the implicit check
  (`stability_checked = False`) and fail closed on an explicit request;
  coupled-solvent (CPCM) RHF/RKS runs disable the verdict because the
  reaction-field density response is not part of the Hessian.
* Because UHF now escapes excited solutions, open-shell UHF energies
  can be **lower** than in releases <= 0.15.112 wherever the old
  driver silently landed on a saddle (e.g. O2 triplet/STO-3G drops
  1.5 mHa to a symmetry-broken minimum).

---

## Iterative diagonalization (Davidson, Phase D3)

For large basis sets (>100 AO basis functions), the SCF Fock
diagonalization is an O(N^3) bottleneck. The **blocked Davidson
algorithm** replaces the full `SelfAdjointEigenSolver` with an
iterative subspace method.

### How to enable

```python
opts.use_davidson = True           # default False
opts.davidson_min_dim = 100        # skip below this AO count
```

For user convenience, the ``solver`` keyword on ``run_job()`` and
``run_periodic_job()`` selects the diagonalization method:
- ``solver="dense"`` :  numpy/scipy exact diagonalization (default)
- ``solver="davidson"`` :  blocked Davidson with eigenvector recycling
- ``solver="lobpcg"`` :  LOBPCG (fastest for large matrices, ~3-12x fewer
  iterations than Davidson)
- ``solver="lanczos"`` :  Lanczos with full reorthogonalization

This is equivalent to setting ``use_davidson=True`` manually on the options
struct, but provides a cleaner API.

### Tuning knobs

```python
from vibeqc import DavidsonOptions
dopts = DavidsonOptions()
dopts.conv_tol = 1e-7   # residual-norm threshold per eigenpair
dopts.max_iter = 200     # outer iterations
opts.davidson = dopts
```

### When it helps
* **Molecular RHF / UHF / RKS / UKS**: above ~500 AOs.
* **Periodic multi-k**: each k-point independently diagonalises its Fock matrix.
* **Below ~100 AOs**: full diagonalization is faster; the driver silently falls back.

### Eigenvector recycling
The Davidson solver automatically recycles the previous SCF iteration's
eigenvectors as its initial guess. Because the Fock matrix changes only
slightly between iterations, the warm-started Davidson typically converges
in 1-3 outer iterations per SCF cycle.

### Availability
All C++ SCF drivers (RHF, UHF, RKS, UKS, periodic Γ-only, periodic multi-k)
support Davidson. Python BIPOLE / GDF drivers will gain it after C++ migration.

### References
Davidson, E. R. *J. Comput. Phys.* 17, 87-94 (1975).
Liu, B. *NRCC Workshop*, LBL-8158 (1978).
Kresse, G. & Furthmüller, J. *Phys. Rev. B* 54, 11169 (1996).

---

## SCF accelerators, pick one

| `opts.scf_accelerator` | Algorithm | When to use |
|---|---|---|
| `SCFAccelerator.DIIS` | Pulay 1980/1982. The error starts from the AO commutator `e = F D S − S D F`. Molecular UKS transforms each spin block to `Xᵀ e X`, with `Xᵀ S X = I`, before measuring it or forming the Pulay matrix. | Standard organic-chemistry workhorse. |
| `SCFAccelerator.KDIIS` | Kollmar 1997 (exposed by ORCA as the opt-in `!KDIIS` keyword). Same Pulay machinery, error vector is the orbital-rotation gradient `g_{ai} = F^MO_{ai}` (occ-vir block in MO basis) instead of the AO commutator. vibe-qc adopts the error vector only, and still diagonalizes the extrapolated Fock; Kollmar's own scheme pairs it with a first-order-perturbation orbital update and is diagonalization-free. | Often robust where DIIS oscillates on transition-metal / open-shell cases where the AO-commutator is dominated by the wrong eigenmodes. |

The shared Pulay solve normalizes its error Gram block by its largest absolute
element before solving the bordered equations. This is an exact rescaling:
the extrapolation coefficients are unchanged and only the Lagrange multiplier
is rescaled. It prevents the numerical rank decision from changing merely
because a converging residual makes every Gram element small.
| `SCFAccelerator.EDIIS` | Energy-DIIS (Kudin/Scuseria/Cancès 2002). Minimises a quadratic energy functional on the convex hull of stored (F, D) pairs. | Diagnostic, pure EDIIS plateaus near convergence; prefer the hybrid below. |
| `SCFAccelerator.ADIIS` | Augmented-Roothaan-Hall DIIS (Hu & Yang 2010). The energy-functional sibling of EDIIS, minimises the ARH energy model expanded about the most recent iterate, same positive-simplex QP. | Diagnostic / comparison, Garza/Scuseria 2012 found it near-identical to EDIIS at the HF level; like pure EDIIS it plateaus near convergence. |
| `SCFAccelerator.EDIIS_DIIS` | **Default (molecular, v0.8.0).** Garza/Scuseria 2012 hybrid. Use EDIIS while `‖e‖_F > ediis_diis_switch_threshold` (default 0.1), then switch to plain DIIS for the asymptotic regime. | The recommended setting for transition-metal complexes, broken-symmetry singlets, and any case where plain DIIS plateaus or oscillates. Matches PySCF / ORCA defaults. |
| `SCFAccelerator.ADIIS_DIIS` | ADIIS analogue of the EDIIS_DIIS hybrid: use ADIIS while `‖e‖_F > ediis_diis_switch_threshold`, then switch to plain DIIS for the asymptotic regime (same switch threshold field). | The ADIIS sibling of the production hybrid; use when you prefer the ARH energy model far from convergence over EDIIS's convex-hull functional. Garza/Scuseria 2012 found ADIIS≈EDIIS at the HF level. |
| `SCFAccelerator.R_CDIIS` | Restarted commutator-DIIS (Chupin/Dupuy/Legendre/Séré 2021). Plain DIIS extrapolation, but the history depth grows until the stored error differences become nearly linearly dependent, then restarts. Restart aggressiveness is `diis_restart_tau` (τ ∈ (0,1); default 1e-4, smaller ⇒ deeper history). | When a fixed `diis_subspace_size` window is over- or under-sized for the problem; lets the depth self-tune and avoids the ill-conditioning of a saturated fixed window. |
| `SCFAccelerator.AD_CDIIS` | Adaptive-depth commutator-DIIS (same paper). Plain DIIS extrapolation, but the window keeps only recent iterates whose residual is within `1/δ` of the current one, so the depth shrinks smoothly near convergence. Controlled by `diis_adaptive_delta` (δ > 0; default 1e-4). | The smoother sibling of R_CDIIS: no post-restart slowdown; the paper's numerics suggest continuously adapting the depth gives the fastest convergence. |

EDIIS and ADIIS minimise a generally nonconvex quadratic exactly by visiting
every non-empty face of the coefficient simplex, following Kudin et al.
Eqs. (15)-(16). For EDIIS, ADIIS, EDIIS_DIIS, and ADIIS_DIIS,
`diis_subspace_size` must therefore be between 2 and 12 (default 8); larger
histories fail on first use instead of falling back to a local stationary
point. Plain DIIS, KDIIS, R-CDIIS, and AD-CDIIS do not use that exponential
solve and retain their existing history-depth range.

```python
opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS    # production hybrid
opts.ediis_diis_switch_threshold = 1e-1                # default, PySCF-style
opts.diis_subspace_size = 8                            # rolling-history cap

# Adaptive-depth commutator-DIIS (Chupin et al. 2021):
opts.scf_accelerator = vq.SCFAccelerator.AD_CDIIS      # or R_CDIIS
opts.diis_adaptive_delta = 1e-4                        # δ for AD_CDIIS
opts.diis_restart_tau = 1e-4                           # τ for R_CDIIS
```

### Per-backend accelerator support

The same `scf_accelerator` field is on every molecular and periodic
options class, but the set of values each backend actually
implements is narrower. Pinning a value that the chosen backend
does not support raises a clear ``NotImplementedError`` with a
pointer to the working route; nothing is silently downgraded.

| Backend                                 | DIIS | KDIIS | EDIIS | EDIIS_DIIS | ADIIS | dynamic_damping | Default              |
| --------------------------------------- | :--: | :---: | :---: | :--------: | :---: | :-------------: | -------------------- |
| Molecular RHF / UHF / RKS / UKS (C++)   | ✅   | ✅    | ✅    | ✅         | ✅    | ✅              | `EDIIS_DIIS` (v0.8.0)|
| Molecular ROHF / ROKS (Roothaan, Py)    | ✅   | ✅    | ✅    | ✅         | ✅    | ✅⁵             | `EDIIS_DIIS`         |
| C++ direct-truncated periodic (Γ)       | ✅   | ✅    | ✅    | ✅         | ✅    | ✅              | `EDIIS_DIIS`         |
| C++ direct-truncated periodic (multi-k) | ✅   | ❌¹   | ✅    | ✅         | ✅    | ✅              | `EDIIS_DIIS`         |
| Python Γ-Ewald (RHF/RKS/UHF/UKS)        | ✅   | ✅    | ✅    | ✅         | ✅    | ✅              | `EDIIS_DIIS`         |
| Python multi-k Ewald (RHF/RKS/UHF)      | ✅   | ✅³   | ✅⁴   | ✅⁴        | ✅⁴   | ✅              | `EDIIS_DIIS`         |
| Python BIPOLE (RHF/RKS/UHF/UKS)         | ✅   | ✅³   | ✅⁴   | ✅⁴        | ✅⁴   | ✅              | `EDIIS_DIIS`         |
| Python GDF (Γ + multi-k)                | ✅   | ✅³   | ✅⁴   | ✅⁴        | ✅⁴   | ✅²             | `EDIIS_DIIS`         |

`R_CDIIS` and `AD_CDIIS` (Chupin et al. 2021) ride the exact same code
path as `DIIS` (only the retained history window differs), so they are
available on **every** backend in the table above, molecular and
periodic, Γ and multi-k.

**One DIIS across every backend in the table above.** Every Pulay-family
extrapolation reachable through `scf_accelerator`, molecular or periodic,
Γ or multi-k, closed- or open-shell, is performed by the single
`vibeqc::DIIS` class in `cpp/src/diis.cpp`. The Python multi-k
accelerators (`_MultiKPulayDIIS`, `_MultiKKDIIS`, `_MultiKKDIISOpenShell`
in `python/vibeqc/periodic_scf_accelerators.py`) are thin adapters that
hand their per-k lists to `DIIS.extrapolate_blocks`; they hold no history
and solve no linear system of their own. The `√wₖ` block bridge (footnote
⁴ below) is what makes that possible: under it the C++ kernel's per-block
Frobenius inner product *is* the k-weighted Pulay inner product
`Σₖ wₖ Re Tr[eᵢ(k)† eⱼ(k)]`, so the history management, the
adaptive-depth policies and the Pulay solve are shared verbatim rather
than re-derived in numpy. Open-shell spin coupling follows from appending
the β blocks to the same lists: one B-matrix, one coefficient set, both
spins.

**The GAPW / GPW plane-wave drivers share the kernel, not the keyword.**
The `run_periodic_*_gapw` / `run_periodic_*_gpw` / `run_periodic_*_rsgapw`
family extrapolates through that same `vibeqc::DIIS`. Its closed-shell
drivers call `DIIS.extrapolate`; its open-shell drivers call
`DIIS.extrapolate_spin_coupled`, which stacks the two spins so one
B-matrix and one coefficient set drive both Fock matrices. No driver in
the family builds a B-matrix or solves a Pulay system of its own.

What they do not take is `scf_accelerator`. Each exposes `use_diis`,
`diis_subspace_size` and `diis_start_iter` instead, so plain Pulay DIIS
is the only accelerator selectable on them and the adaptive-depth
policies stay out of reach. EDIIS, ADIIS and KDIIS are not merely
unexposed there: they are separate classes (`cpp/src/ediis.cpp`,
`cpp/src/kdiis.cpp`) that need an energy history or an MO-basis orbital
gradient, neither of which these drivers pass down. Giving the family an
`scf_accelerator` keyword is tracked work, not a shipped claim. Read the
coverage matrix above as describing the `scf_accelerator` surface, which
the GAPW family does not join.

¹ KDIIS multi-k on the C++ direct-truncated path is still deferred
(per-k MO-basis projection not yet wired into the C++ kernel, 
`cpp/src/periodic_scf.cpp:406-411`); selecting it there raises a
clear error.

² The three Python GDF drivers (`run_rhf_periodic_gamma_gdf` Γ,
`run_pbc_gdf_rhf` Γ compensated-cell, `run_krhf_periodic_gdf` multi-k)
were migrated to the full accelerator family + `dynamic_damping` in
**M4**, the last Python periodic backend to do so. The legacy
`_reject_unsupported_python_accelerator` gate and the `_PulayDIIS`
alias were removed in the same change (the rollout M1 Γ-Ewald → M2
multi-k Ewald → M3 BIPOLE → M4 GDF is now complete; no Python periodic
backend silently downgrades or rejects a supported accelerator).

³ KDIIS on the Python multi-k Ewald path uses the per-k orbital-
rotation-gradient design landed in M2c (`MultiKPeriodicSCFAccelerator`
in `python/vibeqc/periodic_scf_accelerators.py`):
`g(k)_{ai} = (C(k)^† F(k) C(k))_{ai}` with a Pulay B-matrix
`B_{ij} = Σ_k w_k · Re Tr(g_i(k)^† g_j(k))`. This is a strict
extension of the molecular and Γ-point KDIIS error metric to a
k-weighted sum, and is what's used on every Python multi-k Ewald
driver today. Only the error metric is KDIIS-specific: the gradient
blocks go into the same `DIIS.extrapolate_blocks` call as everything
else (the Fock and error block lists are independent histories, so the
`n_vir × n_occ` gradient blocks sit happily alongside `nbf × nbf` Fock
blocks).

⁴ EDIIS / ADIIS / EDIIS_DIIS on the Python multi-k Ewald path use
the **stacked-real-block bridge** landed in M2e
(`per_k_to_stacked_real_blocks` / `stacked_real_blocks_to_per_k`
in `python/vibeqc/periodic_scf_accelerators.py`). Each per-k
Hermitian matrix `M(k) = M_R(k) + i M_I(k)` is mapped to two
real blocks `√w_k · M_R(k)` and `√w_k · M_I(k)`, stacked across
k. The C++ block-vector kernel's sum-of-Frobenius bilinear form
on a paired stack then evaluates to
`Σ_k w_k Re Tr[F(k) D(k)]`, exactly the periodic per-k energy
bilinear form used by `E_elec = ½ Σ_k w_k Re Tr[D(k)(Hcore(k) +
F(k))]` in the multi-k Ewald driver, so the EDIIS / ADIIS QP
linear term (E in Hartree) and quadratic cross-term are in
matching units and the QP minimum coincides with the per-k
energy minimum on the simplex.

The bridge supersedes the M2b Bloch (per-k ↔ per-cell)
bridge, which was found to misweight the QP cross-term: the
`inv_bloch` composition multiplies the per-cell inner product by
`N_g · w_k²` rather than `w_k`, so even on Bloch-Floquet matched
cells the QP coefficient set drifted from the per-k EDIIS
minimum, and convergence failed end-to-end on a kmesh=[2,1,1]
H₂ chain (commit `5c333372`). The stacked-real bridge avoids
the per-cell representation entirely. EDIIS / ADIIS are
asymptotically slower than DIIS for tight `conv_tol_grad`
(no quadratic-information correction past the convex-hull
guess); allow extra `max_iter` if pinning either of them as
the sole accelerator, or prefer `EDIIS_DIIS` (the hybrid
inherits DIIS's fast asymptote once the commutator-error norm
drops below `ediis_diis_switch_threshold`).

The C++ direct-truncated multi-k EDIIS / EDIIS_DIIS / ADIIS branches
maintain a **full per-cell history**: each iterate stores its complete
`LatticeMatrixSet` of Fock and density blocks, and the simplex-QP
cross-term `⟨F_i | D_j⟩ = Σ_g (F_i(g) ⊙ D_j(g)).sum()` is summed over
the real-space cell list. That sum is exactly the periodic energy
bilinear form `E_elec = ½ Σ_g (P(g) ⊙ [H(g) + F(g)]).sum()`, so the
EDIIS / ADIIS energy functional minimised by the QP coincides with
the SCF energy iterate-by-iterate, no Γ-fold approximation, no
uniform-per-cell distribution of a Γ-only extrapolation. (Plain DIIS
still extrapolates the Γ-folded Fock against a Γ-folded commutator
error; a per-cell commutator-error metric is a separate roadmap
item.)

⁵ The shared Roothaan ROHF/ROKS loop extrapolates the single
Roothaan effective Fock through the same `vibeqc::DIIS` /
`EDIIS` / `ADIIS` / `KDIIS` objects as everything else, so the
adaptive-depth policies apply unchanged. Two ROHF-specific
notes. `dynamic_damping` defaults to `False` here, against
`True` on the C++ drivers, and density mixing is skipped
entirely while DIIS is extrapolating, so with the default
`diis_start_iter = 1` neither `damping` nor `dynamic_damping`
does anything unless you push DIIS back. The Roothaan loop also
has **no second-order finalizer**: `soscf_threshold`,
`trah_threshold` and `newton_threshold` exist on
`ROHFOptions` for API parity with the C++ options structs but
are not implemented, so setting them has no effect. An ROHF
case that needs a second-order close-out has to reach the
asymptote on the accelerator alone.

---

## Second-order finalizers, pick zero or one

These are opt-in (default off, threshold 0). When activated, by the
SCF gradient norm dropping below the configured threshold, the SCF
swaps the diagonalize-F update for a step on the orbital-rotation
manifold. DIIS / KDIIS / EDIIS+DIIS / damping / level-shift / FMIXING
are all skipped during the second-order phase (the second-order step
is its own update mechanism).

The C1c quadratic fallback is implemented by the molecular and periodic
Ewald drivers. Experimental `aiccm2026dev-b` routes do not execute that
update and therefore require `quadratic_fallback_iter=0`; a positive value
fails closed before SCF.

The four finalizers are mutually exclusive. If more than one is
configured, the precedence is:

```
quadratic_fallback (C1c)  >  Newton (D2c)  >  TRAH (D2e)  >  SOSCF (D2d)
```

| Field | Algorithm | Hessian | Per-step cost | Convergence | When to use |
|---|---|---|---|---|---|
| `quadratic_fallback_iter > 0` | Single damped Newton step in MO space | Diagonal preconditioner only | One eigsolve, no Fock build | Linear | Cheap escape from DIIS oscillation; the C1c periodic-SCF fallback. |
| `newton_threshold > 0` | **Newton (D2c).** Full orbital Hessian via preconditioned CG. | Full Hessian, exact application via CG matvecs through `JKBuilder` (+ `XCKernelBuilder` for KS) | Multiple Fock builds per step (one per CG iter, ~5-15 typical); KS adds one W^XC matvec per CG iter | Quadratic in ~3-5 outer iters | The recommended finalizer for the asymptotic regime. **RHF / UHF / RKS / UKS, across LDA + GGA + hybrid-GGA** (UKS-GGA via the Phase 17e polarised GGA `f_xc` kernel). Meta-GGA second-order SCF still raises, τ-dependent polarised `f_xc` is unplumbed. |
| `trah_threshold > 0` | **TRAH (D2e Helmich-Paris 2022).** Same Hessian + CG as Newton, plus Powell-ρ adaptive trust radius. | Full Hessian, exact application | Same as Newton plus a ρ-driven trust-radius update each iter | Quadratic in the trust region; more robust than Newton when the initial trust is too large | Hard-convergence cases where Newton's fixed trust radius overshoots. Same coverage matrix as Newton. |
| `soscf_threshold > 0` | **SOSCF (D2d Neese 2000).** Augmented-Hessian eigsolve with diagonal-dominant Hessian. | Diagonal Hessian + AH λ-shift; carries V_xc only through F^MO | Single small eigsolve, no Fock build | Linear (plateaus for pure DFT, see note) | Cheaper than Newton when Newton's matvec is too expensive. **All four flavours; works at tight 1e-9 Ha for HF / hybrid functionals, plateaus at ~1e-4 Ha for pure DFT** (V_xc[ρ+δρ]−V_xc[ρ] is sizable in pure DFT but absent from SOSCF's diagonal-only Hessian, that's literally the motivation for Newton/TRAH). |

```python
# Recommended production stack: EDIIS+DIIS warm-up + Newton finalizer
opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS     # default
opts.newton_threshold = 1.0              # Fischer-Almlöf 1992 convention
opts.newton_opts.cg_tol = 1e-4
opts.newton_opts.trust_radius = 0.3

# Same recipe works for RKSOptions / UKSOptions - Newton/TRAH carry the
# XC contribution to the orbital Hessian automatically (the driver
# builds an XCKernelBuilder pinned at the current density each iter).
# Example (H₂O / cc-pVDZ / PBE on compute-reference): DIIS = 18 iter, Newton = 11 iter,
# ΔE = 4e-14 Ha. Same iter-count win on B3LYP and LDA.

# Cheaper alternative: TRAH instead of Newton (same matvec; adaptive radius)
opts.trah_threshold = 1.0
opts.trah_opts.initial_trust_radius = 0.3
opts.trah_opts.max_trust_radius = 0.7

# Cheap-per-step alternative: SOSCF (no Fock build inside the step)
opts.soscf_threshold = 1.0               # use 0.01 for pure-DFT UKS / RKS
opts.soscf_opts.trust_radius = 0.3
```

**KS extension (v0.8.0).** RKSOptions / UKSOptions expose the same
`newton_threshold` / `trah_threshold` / `soscf_threshold` / `*_opts`
fields as RHFOptions / UHFOptions. When activated, the RKS / UKS
driver builds an XC kernel pinned at the current SCF density and
passes it through to the Hessian matvec:

* **RKS**, `make_unpolarised_xc_kernel_builder` (closed-shell LDA /
  GGA / hybrid-GGA).
* **UKS**, `make_polarised_xc_kernel_builder`, which dispatches by
  functional kind: LDA → polarised-LDA builder, GGA / hybrid-GGA →
  polarised-GGA builder (the Phase 17e spin-polarised `f_xc` kernel,
  with the αβ coupling carried through `v2rho2_αβ`, `v2rhosigma_*`,
  `v2sigma2_*` and the `σ_αβ` cross term).

Meta-GGA second-order SCF (RKS or UKS) is still gated, the
τ-dependent `f_xc` is unplumbed and the kernel factory raises with a
clear roadmap pointer for that functional class.

The trace's `SCFIteration.newton_cg_iter` field records CG iters per
Newton step (0 on iterations that didn't enter Newton).

---

## Application order in the SCF inner loop

Every iteration runs the same pipeline; each enabled knob slots into a
fixed position. The standard CRYSTAL/ORCA order:

```
for iter in 1..max_iter:
    1. D_used  = damping(D_prev, D)              # static or dynamic α
    2. F       = Hcore + G[D_used]                # JKBuilder
    3. F       = fock_mixing(F, F_prev)           # CRYSTAL FMIXING
    4. F       = level_shift(F, S, D_used)        # Saunders-Hillier
    5. F_extra = accelerator.extrapolate(F, ...)  # DIIS | KDIIS | EDIIS | EDIIS+DIIS
    6. C, eps  = step(F_extra, ...)               # diagonalize OR Newton OR SOSCF OR quadratic
    7. D       = build_density(C)
    8. update dynamic damping α from energy decrease (if enabled)
    9. check convergence
```

The accelerator (step 5) is fed every iteration so its history builds
up; the extrapolated F replaces the raw F only once
`iter >= diis_start_iter`. The second-order finalizers (step 6,
`Newton` / `SOSCF`) replace the diagonalize-F step entirely once
their gradient threshold is crossed; in that case steps 5
(extrapolation) and step 1 (damping) are skipped, the second-order
step is its own update.

---

## Recommended workflows

| System | Recommended stack |
|---|---|
| **Routine closed-shell molecule** (organic, main-group) | `accelerator = EDIIS_DIIS` (default), default `damping = 0.5`, `initial_guess = AUTO → SAP`. The defaults work, no tweaking needed. |
| **Hard molecule** (TM complex, broken-symmetry, small-gap) | `accelerator = EDIIS_DIIS` (default), `newton_threshold = 1.0`, `level_shift = 0.2`. |
| **Open-shell** (UHF/UKS doublet, triplet) | Same as hard molecule + `max_iter = 250`. |
| **Periodic insulator** (small-medium cell) | `accelerator = DIIS`, `initial_guess = SAD` (opt in until v0.7), `damping = 0.5`. |
| **Periodic ionic crystal** (NaCl, MgO, …) | Add `fock_mixing = 0.5`, `level_shift = 0.3`. CRYSTAL's standard recipe. |
| **Periodic with charge sloshing** (metallic, small gap) | Add `dynamic_damping = True` and (when smearing is implemented for the path) `smearing_temperature = 0.005`. |
| **Anywhere DIIS oscillates** | Try `accelerator = KDIIS` first (cheap to swap); often robust where DIIS struggles. |

### When Newton/TRAH pay off (v0.8.0 benchmark)

A representative benchmark on compute-reference (Linux x86-64, OpenBLAS+LAPACKE,
4 CPUs), full table in `scripts/bench_scf_second_order.py`:

| System         | Functional | DIIS iter / s    | Newton iter / s  | Iter win | Wall win |
|----------------|------------|------------------|------------------|----------|----------|
| H₂O / cc-pVDZ  | LDA        | 19 / 2.5s        | 11 / 4.9s        | 1.7×     | 0.5×     |
| H₂O / cc-pVDZ  | PBE        | 18 / 2.7s        | 11 / 6.5s        | 1.6×     | 0.4×     |
| H₂O / cc-pVDZ  | B3LYP      | 18 / 3.6s        | 10 / 8.6s        | 1.8×     | 0.4×     |
| Benzene / 6-31G* | PBE      | 26 / 41.7s       | 18 / 84.2s       | 1.4×     | 0.5×     |
| Benzene / 6-31G* | B3LYP    | 18 / 30.0s       | 16 / 80.2s       | 1.1×     | 0.4×     |
| **OH• / cc-pVDZ** | UKS-LDA | **136 / 8.8s**  | **12 / 4.7s**   | **11×**  | **1.9×** |

Two patterns:

1. **Iter count always wins**, Newton's quadratic convergence kicks
   in near the SCF fixed point, so even on easy systems it saves
   5-10 iters. Energies match DIIS to 1e-12 Ha or tighter.

2. **Wall time depends on Fock-build cost vs CG iter count**. Each
   Newton step does ~5-15 Fock builds (one per preconditioned-CG
   iter). DIIS does one Fock build per iter. So Newton wins wall
   time only when the iter saving outweighs the per-iter Fock-build
   overhead. The benzene rows show the unfavourable case (small
   basis, no DF, Fock-build is cheap, DIIS already converges
   quickly). The OH• row shows the dramatic favourable case
   (DIIS struggles with open-shell radicals + small HOMO-LUMO gaps;
   Newton bypasses the oscillation entirely).

**Practical recipe**: for routine closed-shell organics keep the
default (EDIIS_DIIS, no second-order finalizer). Enable
`newton_threshold = 1.0` whenever DIIS takes more than ~30 iters
or the system has known SCF-difficulty markers (open-shell, near-
degeneracies, TM complexes, small gap, broken symmetry).

---

## Diagnosing non-convergence

Gaussian-basis molecular and periodic SCF result structs carry a full
`scf_trace`:

```python
for it in result.scf_trace:
    print(f"iter {it.iter:3d}  E = {it.energy:+.10f}  "
          f"dE = {it.delta_e:+.2e}  |grad| = {it.grad_norm:.2e}  "
          f"diis dim = {it.diis_subspace}  "
          f"newton CG = {it.newton_cg_iter}")
```

Pretty-format utilities:

```python
from vibeqc import format_scf_trace, log_scf_trace
print(format_scf_trace(result))
log_scf_trace(result)         # emits via Python logging
```

`format_scf_trace` and `log_scf_trace` are *post-mortem*, they render
the trace stored on the result object after the SCF returns. For *live*
per-iteration output during a long run (the typical
`nohup … > log 2>&1 & ; tail -f log` workflow on a remote box), pass
`progress=True` to any periodic SCF entry point or to `run_job`. See
[Output files → Progress logging](output_files.md#progress-logging) for
the full description and the entry-point coverage table.

Basis-free semiempirical result adapters currently expose the final
`n_iter` count but not per-iteration energies or residuals. Their compatible
`scf_trace` attribute is therefore an empty list. The formatters omit the
unavailable table and still report the final convergence verdict, iteration
count, and energy; they never manufacture DIIS telemetry.

If the energy oscillates: try `dynamic_damping = True`, increase
`damping` to 0.7, switch to `accelerator = KDIIS` or `EDIIS_DIIS`,
or add `level_shift = 0.2`.

### A stalled tail is usually the DIIS history, not the state

Read the trace before reaching for a convergence aid. The two failures
look alike in a one-line summary and want opposite treatments:

* `dE` and `|g|` both bouncing by orders of magnitude: a genuine
  oscillation. Damp it (level shift, damping, FMIXING).
* `dE` down at `1e-10` or below while `|g|` creeps down a decade per
  hundred iterations: **not** an oscillation. The energy is converged
  and the residual is stalling. Damping it further makes it slower.

The second case is fixed-depth DIIS going near-linearly-dependent: once
the subspace fills with near-identical iterates off a long plateau, the
Pulay system is ill-conditioned and the extrapolation stops paying.
The tell is that restarting the job from its own final density closes it
immediately, because that throws the history away.

One fix is an accelerator that throws the history away for you (see the
measured comparison below for what it does and does not buy):

```python
opts.scf_accelerator = vq.SCFAccelerator.R_CDIIS   # restarted commutator-DIIS
opts.diis_restart_tau = 1e-4                       # restart aggressiveness (default)
```

`R_CDIIS` restarts the history when the Pulay system goes near-singular;
`AD_CDIIS` instead shrinks the depth near convergence
(`diis_adaptive_delta`). Both are Chupin et al. 2021 and are available on
every driver, ROHF/ROKS included, but they answer different questions and
only the restart addresses a degenerate history: see the measured
comparison below before picking. `ROHFOptions` / `ROKSOptions` take the
accelerator as a lowercase string rather than the enum, so the same
selection there reads `opts.scf_accelerator = "r_cdiis"`.

**Worked case (BUG 118).** FeCl₃ / ROHF / multiplicity 6, high-spin d⁵
Fe(III), is the reference system. At def2-TZVP the shipped fixed-depth
run reached `dE = -1.9e-11` and `|g| = 4.2e-06` by iteration 300 and was
still four times short of `conv_tol_grad = 1e-6`, having crawled from
`|g| = 2.6e-05` at iteration 200. Restarting from that state converged in
11 iterations, which is what identifies the history as the problem.

Switching the accelerator fixes the cold run:

Iteration counts on this system are chaotic, so single runs prove
nothing. Measured over **two independent sets of 8 repeats** of the
identical FeCl₃ / def2-SVP input at a 300-iteration cap, on two
successive builds (the DIIS, level-shift and direct-JK paths are
identical between them):

| configuration | converged | median iters | spread |
|---|---|---|---|
| `ediis_diis` (shipped) | **8/16** | 114 / 217 | 63 to 245 |
| `r_cdiis` | **14/16** | 124 / 89 | 57 to 225 |
| `level_shift=0.6` held + `ediis_diis` | **16/16** | 55.5 / 55.5 | 50 to 59 |
| `level_shift=0.6` held + `r_cdiis` | **16/16** | 53 / 55 | 48 to 59 |

Two medians are quoted per row, one per set, because that is the honest
resolution: on the two unshifted rows the medians move by a factor of
two between sets, so eight samples do not pin an iteration count. Only
the converged-fraction and the shifted rows replicate.

Read the table carefully, because the obvious conclusion is the wrong
one. `r_cdiis` buys **reliability, not speed**: it converges within the
cap 14 times in 16 against 8, but its iteration count is not
distinguishable from the default's at this sample size. What actually
collapses both the count and the run-to-run scatter is **holding a level
shift** (16/16, median 55, spread 48 to 59 across both sets), and once
the shift is held the accelerator choice stops mattering (55 versus 55.5,
reproduced in both sets). The shift damps the plateau wandering that
amplifies floating-point noise into a 4x spread in the first place.

So: reach for the held shift first, and treat `r_cdiis` as insurance
rather than as the remedy.

```{warning}
All four rows above converge to the same def2-SVP solution. At
**def2-TZVP the solution the default reaches is the wrong one**, and no
accelerator fixes that. See
[Convergence is not correctness](#convergence-is-not-correctness)
immediately below before using any of this on a high-spin metal.
```

Reach for `R_CDIIS` rather than `AD_CDIIS` on this failure shape. The
two are not interchangeable: shrinking the depth near convergence is not
the same as throwing a degenerate history away, and on this system
`AD_CDIIS` performs like the fixed-depth default at both bases.

Whether the fixed-depth run lands inside a given cap is not
deterministic; repeats of the identical def2-SVP input took 79, 92, 99
and 204 iterations, because the plateau trajectory amplifies
thread-order floating-point noise. That is a good reason to fix the
history rather than to raise `max_iter` and hope.

Two things *not* to reach for on this class of system:

* **Damping first, DIIS later** (`damping = 0.5`,
  `diis_start_iter = 11`) converges quickly and cleanly, onto a
  different, ~0.10 Ha higher solution with a visibly different gap.
  A tidy-looking convergence trace is not evidence you found the same
  state; compare the energy and the frontier gap against a run that
  reached the same place another way.
* **`initial_guess='GWH'`** is unavailable to the Roothaan driver. It
  seeds from densities only, so the overlap-dependent guesses (GWH is
  spelled `HUECKEL` / `HUCKEL` here, plus `SAP`, `PATOM`, `MINAO`)
  raise a `ValueError` naming the supported set (`AUTO`, `SAD`,
  `HCORE`) rather than silently downgrading. `HCORE` is available but
  is the worse start for an open-shell metal: on FeCl₃ / def2-SVP it
  did not converge in 120 iterations and was drifting toward a
  different, ~0.96 Ha higher energy. That leaves SAD as the only
  practical guess here, which is precisely the limitation behind
  [Convergence is not correctness](#convergence-is-not-correctness):
  the model-potential guess that would start you in the right basin
  is the one the Roothaan driver cannot construct.

A level shift is worth trying when the *opening* iterations are wild,
but it is not the remedy for a stalled tail. What it *is* good for on
this system is reaching a different solution entirely, which is the
subject of the next section.

### Convergence is not correctness

A converged SCF is a stationary point, not necessarily the one you
want, and on open-shell transition metals vibe-qc's default SAD guess
can settle on the wrong one. FeCl₃ / ROHF / multiplicity 6 at
**def2-TZVP** is the worked case:

| | default (SAD + DIIS) | `level_shift=0.6` held + `r_cdiis` | ORCA 6.1.1 |
|---|---|---|---|
| E (Ha) | -2640.6034990 | **-2641.1238568** | **-2641.1238569** |
| iterations | 300, not converged | **48** (46 s) | 19 (91 s) |
| Mulliken Fe | +0.275 | (same) | **+1.363** |
| Mulliken Cl | +0.204, -0.239, -0.239 | (same) | -0.454 x3 |
| Mayer Fe-Cl | 0.726, 0.726, **0.189** | (same) | n/a |

The default lands **0.520 Ha (1366 kJ/mol) too high**. The tell is not
the energy, which you have nothing to compare against in a single run,
but the **broken symmetry**: all three Cl in planar FeCl₃ are
equivalent, and the default state gives them three different charges
and one nearly-severed Fe-Cl bond. The correct state has three
identical Cl and the five unpaired electrons on Fe (spin population
4.74, i.e. high-spin d⁵ Fe(III)).

This is a guess problem, not an implementation one: steered into the
right basin, vibe-qc reproduces ORCA to 7×10⁻⁸ Ha on the identical
ansatz (36 closed orbitals / 72 electrons plus 5 singly occupied,
integer occupations both sides). ORCA reaches it because its default
`PModel` model-potential guess starts there; **the Roothaan driver
cannot build that class of guess at all** (`SAP`, `PATOM`, `HUECKEL`
and `MINAO` are all rejected, see below), which leaves a held level
shift as the only in-tree lever.

The same system at **def2-SVP does not have this problem**: the
default converges to −2640.4655173 Ha against ORCA's −2640.4655183,
agreeing to 1 µHa, with three equivalent Cl. So a cheap-basis check
will not warn you. What to do on a high-spin metal:

* Check the symmetry of the population analysis and the bond orders
  against the molecular symmetry before trusting the number.
* Check that the spin sits where the chemistry says it should (Fe spin
  population near 5 for d⁵ Fe(III)).
* Re-run with `level_shift` held (0.3 to 1.0, `level_shift_warmup_cycles = 0`)
  and compare. A lower converged energy at the same multiplicity means
  the first run was on a higher stationary point.

If the energy diverges: check your initial guess (try
`initial_guess = SAD`) and geometry, a bad structure will never
converge. The periodic SCF additionally raises a "POSSIBLY CONDUCTING
STATE" warning when convergence stalls in a way consistent with
metallic character (try smearing).

---

## Parallelism

vibe-qc's compute-heavy kernels, molecular and periodic Fock builds,
analytic gradients, lattice-summed one-electron integrals, the Ewald
lattice sums, and the AO evaluation used by DFT, are parallelised
with OpenMP via a one-engine-per-thread pool.

**Three ways to set the thread count,** in increasing precedence:

1. `OMP_NUM_THREADS` environment variable, before launching Python:
   ```sh
   OMP_NUM_THREADS=8 python my-calc.py
   ```

2. `vibeqc.set_num_threads(n)` from Python, pinning the count for the
   rest of the process:
   ```python
   import vibeqc
   vibeqc.set_num_threads(8)
   print(vibeqc.get_num_threads())   # 8
   ```
   `n <= 0` restores the default (reads `OMP_NUM_THREADS` or falls
   back to the hardware logical-core count).

3. `num_threads=` keyword argument on `vibeqc.run_job`:
   ```python
   run_job(mol, basis="6-31g*", method="rhf", output="h2o",
           num_threads=4)
   ```
   The actual thread count is recorded in the output file as
   `Threads: 4  (OpenMP shared-memory parallelism)`.

**Every `.out` file includes a timing block** at the end:

```
Timings (wall clock, seconds)
----------------------------------------------------
SCF total                              0.326
SCF avg. per iteration                 0.036  (9 iters)
Job total                              0.328
Used 4 OpenMP threads.
```

Open-shell UHF and UKS jobs carry one extra row, because the
post-convergence [internal stability
analysis](#internal-stability-analysis-molecular-uhfuks-default-on)
above is not SCF-loop work:

```
SCF total                              7.902
SCF stability analysis                 4.117  (post-convergence)
SCF avg. per iteration                 0.420  (9 iters)
Job total                              7.910
```

`SCF avg. per iteration` divides the **loop** wall by the iteration count,
so `avg x n_iter + stability` reconstructs `SCF total`. Size a walltime
from `SCF total`, not from the average alone: on large UKS jobs the
stability phase has measured 0.8-3.0x the loop it certifies. The same
split is available machine-readably as the `scf.<method>.stability_analysis`
phase in the [perf log](output_files.md#performance-debugging) and as
`stability_wall_s` / `stability_cpu_s` on the result object.

For systematic benchmarking, ``scripts/bench.py`` in the repository
runs a small fixed suite across a sweep of thread counts and prints a
speedup table.

Good scaling requires enough work per thread, tiny test systems
(diatomics in minimal bases) won't show much because the OpenMP
start-up overhead dominates. Bigger molecules and periodic calculations
with many lattice cells benefit much more (up to near-linear in the
Fock build).
