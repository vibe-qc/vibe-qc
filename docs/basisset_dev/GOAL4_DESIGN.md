# Goal 4, Basis-set optimisation engine

> **⚠ SUPERSEDED** (2026-05-14).  This design document predates the
> `vibe-basis` standalone package architecture and is preserved for
> historical reference.  The active implementation lives in:
>
> * **`vibe-basis/`**, standalone basis-set optimization package
> * **`vibe_basis/recipes/pob_parity.py`**, Goal 8 Stage 0 driver
> * **`docs/basisset_dev/GOAL8_MPEI_TZVP.md`**, current design
>
> The in-process `python/vibeqc/basis_optimization/` scaffold
> described below was **not migrated** into `vibe-basis`, the
> optimization driver evolved into a transport/backend/recipe
> architecture that drives external SCF programs rather than
> linking against vibe-qc.

**Branch:** `basissetdev`. **Module:** `python/vibeqc/basis_optimization/` (superseded).

This is the design + status doc for Goal 4 of the `basissetdev`
branch: re-implement the M. F. Peintinger 2013 / Vilela Oliveira
2019 basis-set optimisation pipeline in vibe-qc, replacing the
historical CRYSTAL09 + MINUIT2 + python-wrapper stack with a
vibe-qc + iminuit + python stack.

## Module layout

```
python/vibeqc/basis_optimization/
├── __init__.py            ← public surface
├── parametrise.py         ← Basis ↔ flat parameter vector
├── objective.py           ← Objective protocol + MockEnergy + SinglePointEnergy
├── io.py                  ← TempBasisLibrary (writes candidate .g94)
├── drivers.py             ← optimize_scipy + optimize_minuit (common OptResult)
├── bdiis.py               ← optimize_bdiis (robust BDIIS/GDIIS + BFGS, same OptResult)
├── ld_diagnostics.py      ← overlap diagnostics + ld_penalty (min eig hinge) + condition_number_penalty (γ·ln κ)
├── gradients.py           ← analytic ∂S/∂α + penalty gradients (condition_number_penalty_gradient, ld_penalty_gradient)
└── recipes/
    ├── __init__.py
    ├── single_atom.py     ← Stage 1 (live SCF)
    ├── one_compound.py    ← Stage 2 (TBD)
    ├── multi_compound.py  ← Stage 3 (TBD)
    └── ld_aware.py        ← Stage 4 (TBD)
```

## Architectural pillars

### Parametrise: which numbers are free, how the optimiser sees them

`BasisParametrisation` stores an immutable reference basis (a dict
of `{symbol: CrystalAtomBasis}`) plus a list of `FreeSpec` entries.
Each `FreeSpec` identifies one number, element, shell index,
primitive index, field (`"exponent"`, `"coeff"`, `"coeff_s"`,
`"coeff_p"`), and how the optimiser sees it (`Transform.LOG` for
exponents, `Transform.LINEAR` for coefficients) plus optional
physical-space bounds.

`pack()` returns the current optimiser-space values; `unpack(x)`
returns a fresh deep-copy of the basis with free fields overwritten.
The reference is never mutated, so optimiser drivers can call
`unpack` repeatedly with arbitrary parameter vectors.

### Objective: what gets minimised

The `Objective` protocol is `Callable[[np.ndarray], float]` with a
`n_calls` counter for budget tracking. Two implementations ship:

* `MockEnergy`, pure-python quadratic, no vibe-qc dependency.
  Used by the architecture smoke test.
* `SinglePointEnergy`, runs vibe-qc SCF on one molecular system
  per call. Composes with `BasisParametrisation` and
  `TempBasisLibrary` to write the candidate basis, build a
  `vq.BasisSet`, and call `vq.run_rhf` / `run_uhf`.

Multi-system reducers (sum of weighted per-system energies) live
in `recipes/`, not in the objective module, that keeps the core
free of recipe-specific assumptions.

### IO: how the candidate basis reaches libint

vibe-qc's molecular SCF takes `vq.BasisSet(mol, name)`, where
`name` resolves to `$LIBINT_DATA_PATH/basis/<name>.g94`. There is
no in-process API to construct a BasisSet from raw exponents and
coefficients (yet). The optimisation engine sidesteps this by
managing a temp `$LIBINT_DATA_PATH` per optimiser run:

```
TempBasisLibrary __enter__:
    create /tmp/vibeqc-opt-XXXXXX/basis/
    set $LIBINT_DATA_PATH = /tmp/vibeqc-opt-XXXXXX

Per evaluation:
    write basis as opt-NNNNNN.g94 (unique suffix)
    return basis_name "opt-NNNNNN"
    SCF caller does: BasisSet(mol, "opt-NNNNNN")

TempBasisLibrary __exit__:
    restore $LIBINT_DATA_PATH
    rmtree the temp dir
```

The unique suffix per evaluation is to defeat any libint caching
keyed on basis name. Whether libint actually caches is a separate
investigation; the unique suffix costs nothing.

This is one file write (~few KB) per SCF call. For atomic / small-
molecule SCF (~100 ms each) the overhead is negligible. For
production multi-compound runs (~30 systems × 50 iterations =
1500 evals), at compute-reference-scale the file I/O is still <1 % of total
wall.

If profiling later identifies file I/O as a bottleneck, the
follow-on is to add a C++-side `BasisSet.from_arrays(exponents,
coefficients, ...)` constructor that bypasses the file system.

### Drivers: scipy + iminuit, common return type

`optimize_scipy` wraps `scipy.optimize.minimize`. Default method
`L-BFGS-B`, with bounds support and finite-difference numerical
gradients (since SCF gradients w.r.t. basis parameters are not
trivially available, they would be Pulay-like terms on the basis
function).

`optimize_minuit` wraps `iminuit.Minuit.migrad`, the modern
Python wrapper around the same Minuit2 / ROOT library that the
2013 paper used. Heavier dependency (C++ Minuit2 underneath, but
ships in iminuit's wheels for free), pays off when:

* HESSE error analysis at the minimum is wanted (parameter
  uncertainties → publishable basis-set errors).
* Many correlated parameters with different scales (the
  multi-compound recipe with N exponents per element).

Both return `OptResult` with `x`, `fun`, `success`, `message`,
`n_evaluations`, `wall_seconds`, `driver`, an optional `history`
list of `(x_k, f_k)` pairs, and the underlying driver object as
`raw`.

### BDIIS driver (`optimize_bdiis`)

`optimize_bdiis` is the third driver (same `OptResult` contract). It
ports CRYSTAL23's `OPTBASIS` optimiser, BDIIS, *Basis-set Direct
Inversion in the Iterative Subspace* (Daga, Civalleri, Maschio,
*JCTC* **16**, 2192 (2020)), which is geometry-DIIS (GDIIS; Császár &
Pulay 1984, robustified by Farkas & Schlegel 1999) applied to basis
exponents/coefficients. CRYSTAL's implementation is deliberately basic;
ours adds, in order of payoff:

* a **BFGS inverse-Hessian** accumulated across iterations, used both
  for the DIIS error vectors `eᵢ = −H⁻¹·gᵢ` and the quasi-Newton
  fallback step, curvature the raw-gradient GDIIS lacks;
* a **trust region + Armijo backtracking** (Farkas-Schlegel robust
  GDIIS): an accepted step never raises the objective, which is what
  actually contains the diffuse-exponent collapse that the
  condition-number penalty only discourages;
* **DIIS subspace conditioning**, a singular / runaway `B`-matrix
  extrapolation silently degrades to the quasi-Newton step rather than
  diverging into a region where the SCF won't converge;
* **box-projected** gradient convergence, so a minimum pinned on a
  parameter bound is recognised as converged (KKT), not reported stuck.

Gradients default to **central** finite differences (one order more
accurate than scipy's forward differences); an analytic gradient can be
supplied via `grad=`.

**Analytic penalty gradients** (`gradients.py`, implemented). The
overlap-based penalty terms are differentiated in closed form:
`condition_number_penalty_gradient` (∂/∂x of γ·ln κ) and
`ld_penalty_gradient` (∂/∂x of the λ_min hinge). The kernel is the
analytic same-center overlap derivative ∂S/∂α, from the closed-form
`s(α,β;l) = (2√(αβ)/(α+β))^(l+3/2)`, combined with Hellmann-Feynman
eigenvalue derivatives (∂λ = vᵀ·∂S·v) and the log-space chain rule. It
returns the optimiser-space gradient, so a BDIIS `grad=` can be assembled
as *analytic penalty grad + energy grad*. This matters because the
penalty is exactly where finite differences are worst (near linear
dependence κ is stiff). Scope: segmented (non-SP) shells; eigenvalues
assumed simple, both FD-verified in `test_basis_opt_gradients.py`.

Future work: the **analytic SCF-energy gradient** (the Pulay term
−tr(W·∂S/∂α) plus ∂h/∂α, ∂(μν|λσ)/∂α), this reuses the same ∂S/∂α
kernel but needs integral exponent-derivatives in the C++/libint layer,
so it is a build-level effort (scope with the maintainer, validate on
compute-reference). Design proposal: [`ENERGY_GRADIENT_DESIGN.md`](ENERGY_GRADIENT_DESIGN.md).
Also: EDIIS/ADIIS-style energy interpolation, per-shell trust radii.

### LD penalty + condition-number penalty

The pob papers' anti-linear-dependence strategy was a hard lower
bound on exponents (0.10 in PT2013, 0.15 / 0.12 in VO2019). Hard
bounds are awkward for gradient-based optimisers; a smooth penalty
on the smallest overlap eigenvalue is preferred. Sketch:

```
λ_min = min_eigenvalue(S(basis))
penalty = λ_LD * max(0, ε_LD - λ_min)^2
objective_total = objective_energy + penalty
```

with `ε_LD ≈ 1e-7` (the threshold above which canonical
orthogonalisation is unambiguous) and `λ_LD` tuned so the penalty
is invisible far from the threshold and dominant near it. The
penalty respects bounds in the `BasisParametrisation` so the user
can also impose a hard floor where they prefer (publication-style
reproducibility of pob-rev2's "0.15 lower threshold" rule).

**Status: implemented** in `ld_diagnostics.py` (`ld_penalty`,
`ld_penalty_from_atom`) and wired into the objective factories via
`use_ld_penalty=`.

Alongside it sits the **condition-number penalty**, the term the
CRYSTAL/OPTBASIS and VandeVondele objectives actually minimise:

```
Ω({α, d}) = E_tot({α, d}) + γ · ln κ({α, d})
```

with `κ = λ_max / λ_min` the overlap condition number (VandeVondele &
Hutter, *J. Chem. Phys.* **127**, 114105 (2007)). Unlike the `ld_penalty`
hinge, zero until `λ_min` crosses `ε_LD`, then quadratic, `ln κ`
contributes everywhere and shapes the whole search, growing without
bound as `λ_min → 0`. They compose (global `ln κ` conditioning + the
`ε_LD` hinge as a hard backstop). Implemented as
`condition_number_penalty` / `cond_penalty_from_atom` and exposed on the
objective factories via `use_cond_penalty=` / `cond_gamma=`. This is the
penalty `optimize_bdiis` is designed to optimise against.

## Stages and dependencies

| Stage | Description | Depends on |
|-------|-------------|------------|
| 1     | Single-atom HF, 1 free param, 1 system | nothing, laptop-runnable today |
| 2     | One-compound bulk solid (e.g. LiH) refit | REQUIREMENTS-PERIODIC R2 partial OK; full PW1PW gate on R4 |
| 3     | Multi-compound simultaneous fit (the pob recipe) | + Goal 3 test-set inputs |
| 4     | LD-aware multi-compound fit | + `ld_penalty.py` |

**Status refresh, 2026-05-13.** Codex's native GDF backend
(`periodic_rhf_gdf.py` / `periodic_k_gdf.py`) replaces the
retired in-process PySCF periodic path. Γ-only RHF / RKS,
`fock_mixing` (CRYSTAL-style FMIXING), and Fermi-Dirac
`smearing_temperature` are now wired through GDF and Ewald
multi-k. **This unblocks the Stage 2 LiH-only smoke** against
the cubic ionics (RHF; PW1PW-RKS still gates on R4 alias).
Stage 3 (non-cubic test set) still waits on R1 + R3 + R5.

Manifest § 10: the shared `validate_fraction_01` /
`mix_fock_matrices` helpers in `cpp/include/vibeqc/scf_mixing.hpp`
are the canonical SCF-mixing primitives across molecular +
periodic. Goal 4 reuses them via the runner kwargs
(`fock_mixing=`, `smearing_temperature=`), **never duplicate a
DIIS / FMixing / smearing implementation inside
`basis_optimization/`**.

## Stage 2 design pre-empt, one-compound bulk solid refit

The Stage-1 architecture (`SinglePointEnergy` + `optimize_scipy` +
`TempBasisLibrary`) is **already extensible to Stage 2**. The
remaining work is mechanical glue + a periodic SCF runner. Below
is the shovel-ready design so the moment R1-R5 land in the
periodic chat, Stage 2 lights up without rearchitecting.

### Target use case

LiH (rocksalt, 8-atom conventional cell, pob-TZVP) is the natural
first one-compound target:

* It's in the Phase-1 test-set inputs already (commit `6a77356`).
* The 1.0 starting basis has H s-diffuse exponent at 0.1795, the
  pob LD-aware compromise. The atomic optimum (Stage 1 result) is
  ~0.131. Stage 2 should fall *between* the two: less diffuse
  than 0.131 (because the periodic LD penalty pulls it tighter)
  but more diffuse than 0.1795 (because LiH-specific vs. multi-
  compound calibration).
* Reference acceptance target: PT2013 SI Table 2 lists
  E_HF(LiH) = −8.062 837 Ha per cell with CRYSTAL standard
  basis at converged k-mesh. Stage 2 should reproduce this to
  ~10 mHa with our optimised pob-TZVP-LiH (LiH is a soft
  sensitivity case, the published number won't move much).

### New module: `recipes/one_compound.py`

Mirrors `recipes/single_atom.py` but swaps molecular UHF for
periodic SCF. Pseudocode (updated 2026-05-13 to target Codex's
native GDF backend):

```python
def run_lih_smoke(
    *,
    perturbation: float = 1.5,
    bounds: tuple[float, float] = (0.05, 1.0),
    max_iter: int = 30,
    kmesh: tuple[int, int, int] = (4, 4, 4),
    functional: str = "rhf",        # PW1PW gates on R4 alias
    fock_mixing: float = 0.0,        # shared FMIXING helper
    smearing_temperature: float = 0.0,  # shared Fermi-Dirac helper
) -> SmokeResult:
    """Vary the diffuse-most H s exponent in pob-TZVP and minimise
    LiH/RHF total energy at converged k-mesh."""
    from vibeqc.basis_optimization.recipes._lih_geometry import lih_8atom
    sysp = lih_8atom()                         # PeriodicSystem from
                                               # examples/basisset_dev/inputs
    parametrisation = BasisParametrisation(
        atoms={"H": pob_tzvp_h(), "Li": pob_tzvp_li()},
        free=[FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
                       field="exponent", transform=Transform.LOG,
                       bounds=bounds)],
    )
    objective = PeriodicSinglePointEnergy(   # NEW
        parametrisation=parametrisation,
        system_factory=lambda: sysp,
        kmesh=kmesh,
        functional=functional,
        library=lib,
        # Γ-only path: vq.run_rhf_periodic_gamma_gdf
        # Multi-k path: vq.run_krhf_periodic_gdf (RHF) /
        #               vq.run_krks_periodic_gdf (RKS)
        scf_runner=vq.run_krhf_periodic_gdf,
        runner_kwargs=dict(
            fock_mixing=fock_mixing,
            smearing_temperature=smearing_temperature,
        ),
    )
    res = optimize_scipy(objective, x0=parametrisation.pack(),
                         bounds=parametrisation.optim_bounds(),
                         tol=1e-6, max_iter=max_iter)
    ...
```

The new `PeriodicSinglePointEnergy` is a sibling of
`SinglePointEnergy` in `objective.py`. Differences:

* Takes a `PeriodicSystem` (not `Molecule`) and a `kmesh`.
* Calls a periodic SCF runner, preferring the **native GDF**
  path (`run_rhf_periodic_gamma_gdf` for Γ-only,
  `run_krhf_periodic_gdf` / `run_krks_periodic_gdf` for multi-k).
  The Ewald multi-k runners are available too but the GDF
  path is the parity target against PySCF.pbc.GDF + CRYSTAL14.
* Energy returned is per-unit-cell, not per-molecule.
* Otherwise the parametrise → write_g94 → BasisSet → SCF →
  unpack pipeline is identical.
* **`fock_mixing` and `smearing_temperature` arrive via the
  runner kwargs**, no per-recipe re-implementation of CRYSTAL
  FMIXING or Fermi-Dirac smearing. The runners dispatch into the
  shared `scf_mixing.hpp` primitives.

### compute-reference fan-out for the multi-compound generalisation (Stage 3 prep)

Stage 2 runs on a single compound; one SCF per optimiser
evaluation. That fits the laptop *if* the cell is small enough.
Stage 3 is multi-compound, N systems × M iterations evaluations,
where each evaluation is independent of the others within an
iteration. That demands fan-out.

The vq + compute-reference integration is already in place
(`reference_vq_queue.md`). The stage 3-ready architecture:

```
Stage 3 driver loop (laptop)         compute-reference (vq daemon)
─────────────────────────────         ────────────────────────
parametrise.pack() → x_k
  │
  └─ for each system_i in test_set:
        emit candidate basis to a shared S3-style staging dir
        vq submit  examples/.../system_i_basis_xk.py  (--cpus N_i)
        ↓
                                       JOBID_i goes to pending
                                       max-jobs=1 dispatches one
                                       SCF runs, writes E_i to file
                                       ↓
        poll vq queue until all JOBID_i are completed
        vq fetch each output → laptop ./out/JOBID_i/
        read E_i from output files
        objective_value = Σᵢ wᵢ · E_i  +  λ · LD_penalty(min_eig(S))
        scipy.optimize → x_{k+1}
```

The MIGRAD / scipy MO loop stays local; only the SCF energy
evaluations cross the wire. The vq daemon's current `--max-jobs 1`
serializes the fan-out within an iteration, which is fine for
correctness; relaxing to `--max-jobs 4` (when v0.3 lands the RAM
budget) takes wall-clock down by ~Nᵢ.

### Multi-system objective in code

To live in `objective.py`:

```python
@dataclass
class MultiSystemEnergy:
    """Σ wᵢ · E_i(basis) over a list of systems, minimised jointly."""
    parametrisation: BasisParametrisation
    systems: list[SystemSpec]      # (factory, kmesh, weight, scf_runner)
    library: TempBasisLibrary
    transport: Transport            # Local | RemoteVqPlanetx
    n_calls: int = 0

    def __call__(self, x: np.ndarray) -> float:
        atoms = self.parametrisation.unpack(x)
        basis_name = self.library.write_g94(atoms,
                                            basis_name=f"opt-{self.n_calls:06d}")
        # Fan out via the chosen transport. Local = sequential
        # in-process SCF. RemoteVqPlanetx = vq submit + poll.
        results = self.transport.evaluate(basis_name, self.systems)
        self.n_calls += 1
        return sum(w * E for (E, w) in zip(results, self.weights))
```

`Transport` is the ABC that hides "where does each SCF run". Two
concrete implementations: `LocalTransport` (sequential, for
laptop-scale Stage 2 + smoke tests) and `RemoteVqPlanetxTransport`
(submits via vq, polls until done, fetches energies). This keeps
the optimiser code fan-out-agnostic.

### LD penalty (Stage 4)

Sketch from the prior pages of this design doc. Re-emphasised here
because it slots cleanly into `MultiSystemEnergy`:

```python
λ_min = min_eigenvalue(S(basis_at_x))
penalty = λ_LD * max(0, ε_LD - λ_min) ** 2
```

Computed once per `__call__` (one extra `compute_overlap` call;
trivially cheap relative to the SCF). λ_LD tuned so the gradient
contribution is negligible far from the threshold and dominant
near it. ε_LD = 1e-7 tracks the v0.7.0 default linear-dependence
threshold.

### What still blocks Stage 2 / 3 / 4

* **R1** (non-orthorhombic Ewald), every hexagonal / trigonal /
  tetragonal compound (≈32 in the pob set) is still refused at
  SCF entry. Stage 3 multi-compound can launch over the cubic
  subset already, but the full PT2013 / VO2019 calibration set
  waits on R1.
* **R2, partially unblocked.** Native GDF backend
  (`periodic_rhf_gdf.py` / `periodic_k_gdf.py`) landed; CRYSTAL
  FMIXING + Fermi-Dirac smearing live in the canonical
  `scf_mixing.hpp` helpers and are wired through the runner
  kwargs. Stage 2 LiH RHF smoke can run today via
  `run_krhf_periodic_gdf`. Outstanding: production-size 8×8×8
  benchmark + metallic FCC Cu smoke (the latter only matters for
  T12 metals in Goal 6).
* **R4** (PW1PW registry alias), PW1PW-specific Stage 2 still
  blocked. Workaround for the LiH smoke: run RHF + report
  Δ(opt - start) without comparing absolutely to the SI Table-1
  PW1PW number, OR use `hf_exchange_fraction=0.20` with PW91
  components if the periodic chat hasn't yet wired the
  three-component registry resolver.
* **R3** (broken-symmetry guess), the AFM TM oxide subset can't
  be optimised against until R3 lands. The pob recipe deliberately
  includes them, so Stage 3 needs R3 for full parity.
* **R5** (periodic geometry / lattice optimisation), the pob
  recipe optimises lattice constants alongside basis exponents,
  with the lattice as a "downstream" parameter. We can defer
  lattice opt to Stage 3+ and use experimental lattices in
  Stage 2 (the LiH-only smoke).
* **Goal 3 inputs** (Phase 3 done; Phase 4 hex/tet blocked on R1)
  - need at least the cubic ionics + cubic semiconductors
  available before Stage 3 multi-compound can run. ✅ Phase 1 +
  Phase 2 + Phase 3 cubic inputs landed (Phase 14h adds
  CRYSTAL14 `.d12` parity sidecars).

### Acceptance gates

* **Stage 2 LiH smoke**: optimised H diffuse exponent lands in
  (0.13, 0.18); LiH/pob-TZVP-LiH energy at 4×4×4 PW1PW agrees
  with PT2013 SI Table 2 (CRYSTAL std basis) to within ~10 mHa.
* **Stage 3 12-system PT2013-T4 fit**: rerun the 2013 calibration
  on the Phase 1 cubic ionics. Resulting pob-TZVP-vibeqc-2026
  exponents within 5 % of the published 2013 values, total energy
  drift < 1 mHa per cell vs. PT2013 SI Table 2.
* **Stage 4 LD-aware**: same as Stage 3 but with λ_LD active. No
  exponent below 0.10 (the published pob lower bound). Verifies
  the LD penalty is biting.

## Where work runs

Per `feedback_site_e.md`: optimisation runs target compute-reference. The
laptop loop is reserved for stage-1 smoke and architecture tests.
Stages 2-4 with full pob-style test sets generate
~600-1500 SCF evaluations per fit and need the 16-core / 128 GB
machine.

When the periodic features land, the recipe code adds a
`run_on_site_e=True` flag that bundles the parametrisation +
objective + driver, ships them to compute-reference via ssh, runs there,
and pulls the OptResult back. The MIGRAD loop stays local; only
the energy evaluations cross the wire.

## Testing

`tests/basisset_dev/test_basis_opt_stage1_arch.py`, pure-Python
smoke test, no vibe-qc dependency. Verifies:

* `pack ∘ unpack` is identity for log-space parameters.
* `unpack` modifies only the targeted field.
* Bounds are enforced.
* Field-name / shell-type validation works (SP shell rejects
  ambiguous `"coeff"`).
* `optimize_scipy` recovers the centre of a quadratic mock in 1-D
  and 3-D (different curvatures per axis).
* End-to-end: parametrise → MockEnergy → scipy driver → recover
  exponent in physical space.

Run with:

```sh
.venv/bin/python -m pytest tests/basisset_dev/test_basis_opt_stage1_arch.py -v --noconftest
```

(`--noconftest` is required because the parent `tests/conftest.py`
imports the C++-backed `vibeqc` symbols; the architecture tests
shim around that.)

The live SCF stage-1 smoke is `recipes.single_atom.run_h_smoke()`
- opt-in, requires a working `pip install -e .` install of vibe-qc
in the worktree's venv. Run with:

```sh
.venv/bin/python -c "from vibeqc.basis_optimization.recipes.single_atom \\
                     import run_h_smoke; print(run_h_smoke().summary())"
```

Expected: from a starting H-diffuse-s exponent of 0.1795 × 1.5 ≈
0.27, the optimiser pulls to a value that minimises atomic UHF
energy. The exact landing point is recipe-dependent; the smoke
asserts that `converged=True` and the energy at the optimum is
no worse than at the start.
