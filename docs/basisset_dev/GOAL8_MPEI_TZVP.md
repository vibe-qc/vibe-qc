# Goal 8, `mpei-TZVP`: a Hartree-Fock-optimized basis set

**Status:** active, opened 2026-05-13.
**Branch:** `basissetdev`.
**Owner:** M. F. Peintinger + basissetdev chat.
**Target output:** paper + bundled basis-set archive.

## 1. Motivation

`pob-TZVP`, `pob-TZVP-rev2`, and `pob-DZVP-rev2` were optimized
against **PW1PW** (Bredow-Gerson 2000, 20 % HF + 80 % PW91 exchange
+ PW91 correlation), a DFT-leaning hybrid. The published basis
exponents and contraction coefficients therefore carry a
**DFT-implicit** bias: any orbital relaxation or correlation effects
captured by PW1PW that aren't present in pure HF show up as
suboptimal Gaussian primitives when the basis is reused at the HF
level.

For users who need a pob-quality periodic Gaussian basis at the
**Hartree-Fock** level, cohesive energies of ionic crystals,
HF reference points for post-HF methods, screened-exchange
hybrids that flip the HF fraction higher, there is currently no
solid-state-targeted choice. `def2-TZVP` is molecular-optimized;
it usually works at the periodic Γ-point but lacks the pob
LD-aware exponent floor that makes multi-k SCF stable.

**Goal 8 ships `mpei-TZVP`**: same test set as pob, same
multi-compound joint optimization, but with **total HF energy** as
the objective. The exponent / coefficient deltas between
`pob-TZVP-rev2` (DFT-opt) and `mpei-TZVP` (HF-opt) are themselves
the **publication finding**, they quantify the DFT-vs-HF basis
bias on the standard cubic-ionic / cubic-semiconductor / TM-oxide
benchmarks.

## 2. Methodology continuity

Per the user decision (2026-05-13), Goal 8 follows the original
PT2013 recipe as closely as possible to keep the methodology
defensible:

| Choice | 2013 (pob-TZVP) | 2026 (mpei-TZVP) | Rationale |
|--------|-----------------|------------------|-----------|
| SCF engine | CRYSTAL09 | **CRYSTAL14** (compute-reference, via `vq`) | Same engine family; reproduce numerics |
| Optimizer | MINUIT2 (C++, Powell) | **NLopt BOBYQA** primary + **iminuit MIGRAD/HESSE** for errors | BOBYQA = Powell descendant; iminuit = same Minuit2 |
| Objective | Σ wᵢ E_PW1PW(basis) per compound | **Σ wᵢ E_HF(basis)** per compound | The HF switch, the only methodological change |
| LD handling | hard floor 0.10 (PT2013) / 0.15 (rev2) | **hard floor 0.15** (rev2 inheritance) | Optimizer bounds; revisit if MIGRAD bunches |
| Seed basis | from scratch with def2-TZVP / Stuttgart | **pob-TZVP-rev2** | User-confirmed; PoC efficiency over methodological purity |
| Test set | PT2013 T4, T5, T6, T8-T12, T14 | **PT2013 T4 + T8 + T10** for Stages 0-3 | Cubic subset; R1/R3 unblock the rest later |
| Reference | none, defines pob | **PT2013 SI Table 2 (HF totals)** | Validates Stage 0; mpei-TZVP totals are *new* numbers |

Everything else, multi-compound joint fit with equal weights, BSSE
counterpoise correction (rev2-style, Stage 4+), exponent /
coefficient parametrization in log-space, carries over verbatim.

## 3. Architecture

```
  Laptop (this worktree, basissetdev)              compute-reference (vq daemon, CRYSTAL14)
  ──────────────────────────────────                ──────────────────────────────
  optimizer (NLopt BOBYQA / iminuit MIGRAD)
        │
        │  per evaluation x_k = (exponents, coefs):
        │
        ├─ python/vibeqc/basis_optimization/mpei/
        │    └── candidate_basis.py
        │       ↓ writes per-element CRYSTAL basis files
        │       ↓ to ./mpei-cN/ basis/
        │
        ├─ examples/basisset_dev/_generator.py
        │    └── emit_crystal_d12(struct, basis="mpei-cN", method="rhf")
        │       ↓ generates .d12 for each test compound,
        │       ↓ pointing at ./mpei-cN/ basis/
        │
        ├─ for each test compound c_i:
        │    vq submit -d ./mpei-cN/ \
        │              --cpus 14 --wall-time-seconds 1800 -- \
        │              bash run-crystal.sh c_i.d12
        │       → compute-reference queue → CRYSTAL14 SCF → c_i.out
        │
        ├─ wait for all c_i to finish; vq fetch each output
        │    (poll loop with timeout; abort eval if any compound fails)
        │
        ├─ parse TOTAL ENERGY from each c_i.out (CRYSTAL14 output parser)
        │       ↓
        │     E_i = total HF energy per cell
        │
        ├─ compute scalar objective:
        │     L(x_k) = Σᵢ wᵢ · Eᵢ
        │
        │   (LD bound is a hard parameter bound on x_k, not a
        │    soft penalty - BOBYQA respects bounds natively;
        │    no in-process overlap-matrix call needed.)
        │
        └─ return L(x_k) to optimizer; next evaluation
```

Key properties of this architecture:

* **No vibe-qc SCF in the loop**, Goal 8 is independent of
  REQUIREMENTS-PERIODIC R1-R5 progress. vibe-qc's role is to
  *generate* CRYSTAL inputs (the Phase 14h `.d12` generator) and
  to *orchestrate* the compute-reference queue (the `vq` submit/poll/fetch
  pattern from `reference_crystal14_via_vq.md`). The SCF
  engine is exclusively CRYSTAL14.
* **No imports from CRYSTAL**, CRYSTAL is an external code per
  CLAUDE.md § 10. We read CRYSTAL output files, but the
  optimization driver is pure vibe-qc + iminuit + nlopt.
* **Hard bounds, not penalty**, the original pob's LD floor
  was a hard constraint (exponent ≥ 0.10 / 0.15). BOBYQA and
  iminuit both respect parameter bounds natively, so we don't
  need to compute min-eigenvalue(S) at each evaluation. This
  also keeps the methodology continuity with 2013 / 2019.

### 3.1 Why not vibe-qc as the SCF engine (2026-05-13)

vibe-qc is the home of the orchestration code (`basis_optimization/`,
the Phase 14h `.d12` generator, the compute-reference `vq` adapters), but
**not** the SCF engine for Goal 8. Per the user's explicit framing
on 2026-05-13:

1. **Too slow.** A single Γ-point periodic RHF on LiH (8-atom cubic
   cell, pob-TZVP) ran 30+ CPU-minutes locally during Stage-0
   pipeline scoping without converging, orders of magnitude
   slower than CRYSTAL14 on the same input (≤ 30 s on compute-reference).
   At ≈ 100 BOBYQA evals × 13 compounds for Stage 3, the vibe-qc
   wall-time budget would be measured in weeks of laptop CPU; the
   CRYSTAL14 + compute-reference budget is hours.
2. **No symmetry.** vibe-qc's `PeriodicSystem` expects the
   fully-expanded unit cell (every Wyckoff-equivalent atom written
   out explicitly). CRYSTAL14 expands from the asymmetric unit via
   the space group at parse time and exploits Fm-3m / Pm-3m / etc.
   symmetries during the SCF, a factor of 8-24× per cubic cell for
   the test set. The Phase 14h `.d12` generator already emits the
   asymmetric-unit form; using CRYSTAL14 collects this speedup
   for free.
3. **Periodic SCF still buggy.** The CLAUDE.md § 7 standing rule
   ("oscillating periodic SCF / impossible energies = bug, not a
   convergence-aid problem") records that vibe-qc periodic SCF has
   known open bugs across several lattice classes. The
   regression-suite chat owns the fix path; basissetdev does not
   take a dependency on it.

When vibe-qc periodic SCF reaches CRYSTAL14 parity (the
`REQUIREMENTS-PERIODIC` punch list closing R1-R5 + R7 + R9),
**Goal 8 can flip to a hybrid mode** where vibe-qc runs the
optimization fan-out and CRYSTAL23 runs only at checkpoints. The
architecture above already supports this: replace
``transport_vq.crystal_output_parser`` with a vibe-qc
``run_rhf_periodic_*_gdf`` call site, keep everything else.

Until then, **CRYSTAL23 is the external reference engine**.

### 3.2 compute-reference checkout for the CRYSTAL23 fan-out

Per the established compute-reference install convention (one work track
= one sibling checkout under `/home/USER/gitlab/`, see memory
note `reference_site_e_vibeqc_basis.md`), Goal 8 owns:

    /home/USER/gitlab/vibeqc-basis/

User-approved 2026-05-13. It mirrors the `vibeqc-dev` /
`vibeqc-release` / `vibeqc-queue` / `vibeqc-rsgdf` / etc.
pattern: historical dedicated clone of the archived monorepo on the `basissetdev`
branch, and a dedicated `vibe-basis/.venv` created with
`./vibe-basis/scripts/install.sh --extras all`. vq is managed separately
through its own lifecycle because the transport shells out to its CLI.

Setup is on-demand, i.e. when the first `pob_parity` recipe
is ready to run, not speculatively. The `vq` daemon (running
out of `vibeqc-queue/`'s install) is shared across all
sibling checkouts, so jobs submitted from `vibeqc-basis/` use
the same queue any other chat does.

## 4. Wall-time budget on compute-reference

Per `reference_crystal14_via_vq.md`, MgO at PBE/POB-TZVP with
SHRINK 8 8 ran in:
* Serial: 21 s
* Parallel (4 ranks): 9 s

Extrapolating to a Stage 3 multi-compound joint fit:

| Stage | Compounds | Evals (BOBYQA) | Wall time / eval | Total walltime |
|-------|-----------|----------------|------------------|----------------|
| 0 (pob-TZVP HF parity) | 13 | 1 | ~3 min | **3 min** |
| 1 (single-atom HF) | 1 (H + Li + …) | ~30 | < 1 s | **< 1 min** |
| 2 (LiH 1-compound) | 1 | ~50 (1 free param) | ~20 s | **17 min** |
| 3 (13 cubic ionics, joint) | 13 | ~100 (≈ 8 free params) | ~20 s × 13 = 4 min | **6.7 h** |
| 4 (37 cubic compounds w/o AFM) | 31 | ~150 | ~20 s × 31 = 10 min | **25 h** |

At `--max-jobs 1` (current daemon setting), Stage 3 is a one-day
job; Stage 4 is a 1-2 day job. Relaxing the daemon to
`--max-jobs 4` cuts these by ~3.5×.

For comparison: PT2013 in 2013 took **weeks** on the user's
academic cluster. We're now a factor of 50-100× faster on a single
2026 16-core box, before any parallelism.

## 5. Stages and acceptance gates

### Stage 0, pob-TZVP HF parity reproduction

**Purpose:** prove the CRYSTAL14 + vq + output-parser pipeline
works end-to-end *before* touching the basis. This is the
infrastructure smoke test.

**Procedure:**
1. Take the existing `pob-tzvp` basis from `python/vibeqc/basis_library/`.
2. Use `examples/basisset_dev/_generator.py emit_crystal_d12(...,
   method="rhf")` to emit `.d12` for the 13 cubic ionics (Phase 14h
   inventory).
3. `vq submit` each through `run-crystal.sh` on compute-reference.
4. Parse `TOTAL ENERGY` from each `.out`.
5. Compare to PT2013 SI Table 2 (HF) values.

**Acceptance gate:** Σᵢ |E_HF(vibeqc generator) − E_HF(PT2013 SI Table 2)|
< 0.1 mHa per cell, summed over the 13 cubic ionic test set.
Per-compound deltas printed for review.

If this fails, the pipeline is wrong (basis-file encoding,
SHRINK / TOLDEE / TOLINTEG defaults, or output parsing), fix
before proceeding to Stage 1+.

### Stage 1, single-atom HF (sanity)

**Purpose:** verify the optimizer + bound-handling work in a
trivial case where the answer is known analytically.

**Procedure:**
1. Pick H (1 free param: the diffuse-most s exponent).
2. CRYSTAL14 on an isolated H atom (gas-phase / cluster mode).
3. NLopt BOBYQA with bounds (0.05, 1.0); start at 0.27 (= 1.5 ×
   pob-TZVP value).
4. Minimize E_HF.

**Acceptance gate:** converged exponent within 1 % of the atomic
HF optimum (well-known literature value ~0.10-0.13 for H s
exponent in TZVP-class bases).

### Stage 2, LiH one-compound HF MIGRAD

**Purpose:** the publication-grade proof-of-concept. One
compound, few free parameters, gradient-free.

**Procedure:**
1. Seed basis: pob-TZVP-rev2 for Li + H.
2. Free parameters: the 3 valence-shell exponents that the rev2
   recipe identifies as LD-sensitive (specifically the H s
   diffuse-most + Li outermost s and p).
3. NLopt BOBYQA bounded: exponent floor 0.15, ceiling 5.0.
4. Objective: total HF energy of the LiH 8-atom cubic cell
   at SHRINK 8 8 + TOLDEE 8 + default TOLINTEG.
5. After BOBYQA converges, run iminuit MIGRAD at the BOBYQA optimum
   for HESSE error analysis (parameter uncertainties → publishable
   "mpei-TZVP-LiH exponents = 0.183 ± 0.008" style numbers).

**Acceptance gate:**
* BOBYQA converges (≤ 50 evals, gradient norm test below
  optimizer's default tolerance).
* mpei-TZVP-LiH total HF energy < pob-TZVP-rev2-LiH total HF
  energy by at least 0.1 mHa per cell (the variational
  improvement from the HF objective).
* All converged exponents above the 0.15 LD floor (i.e. MIGRAD
  doesn't bunch at the bound, if it does, the floor was too
  tight for the HF surface and we revisit).
* HESSE error matrix is positive-definite, guarantees we're at
  a true local minimum, not a saddle / boundary.

**Outreach gate (per user 2026-05-13): on Stage 2 success, reach
out to universities for collaboration on the full Stage 3 / Stage
4 production run.** Stage 2 is the smallest defensible result that
makes the methodology paper-worthy on its own (LiH paper figure
showing rev2 → mpei delta).

### Stage 3, multi-compound joint HF fit (13 cubic ionics)

**Purpose:** the actual mpei-TZVP-cubic-ionic basis. Same scope as
PT2013 T4 + PT2013 T14.

**Procedure:**
1. Seed basis: pob-TZVP-rev2 across the 8 elements that appear
   in the 13 compounds (H, Li, Na, K, F, Cl, Mg, Ca).
2. Free parameters: per-element valence-shell exponents, likely
   ~16-24 free total (2-3 per element). Coefficients held fixed
   for Stage 3; promoted to free in Stage 4 if needed.
3. Equal weights wᵢ = 1 across the 13 compounds (PT2013 convention).
4. Optimizer cascade:
   * NLopt BOBYQA bounded: rapid exploration. Stop when
     gradient norm < 1e-4 or evals = 200.
   * iminuit MIGRAD at the BOBYQA optimum: refine + HESSE error.
5. compute-reference fan-out: each evaluation spawns 13 `vq submit` calls.
   Daemon currently `--max-jobs 1`; bump to 4 if available.

**Acceptance gate:**
* All 13 compounds converge for every BOBYQA evaluation.
  (Any non-converged eval → infinite objective; BOBYQA will route
  around it. Persistent failures = LD bound too loose; tighten.)
* Final mpei-TZVP cubic-ionic Σwᵢ E_HF < pob-TZVP-rev2 Σwᵢ E_HF
  by at least 5 mHa total, i.e. the joint HF optimization
  genuinely improves over the rev2 baseline averaged across the
  13 compounds.
* HESSE diagonal: relative parameter uncertainty < 5 % for every
  free exponent (the publishable accuracy threshold).
* Reproducibility: re-running the BOBYQA from a +20 % perturbed
  start converges to the same exponents within 1 %.

### Stage 4, full test set (after R3 lands)

**Purpose:** extend mpei-TZVP to cover the full PT2013 T4 + T8 +
T10 test set (31 compounds excluding hex/tet, AFM TM oxides).
Add ATOMBSSE counterpoise correction (rev2-style) to make
mpei-TZVP-rev2 the publishable end product.

Detailed scope is opened after Stage 3 lands.

## 6. Module layout, `vibe-basis` (standalone package, 2026-05-13)

Goal 8 lives in **`vibe-basis/`**, a standalone Python package at
the top of the vibe-qc repository. It deliberately stays there after the
viewer, queue, and QVF format move to separate repositories.
vibe-basis is the modern home of every "drive external SCF
programs under optimization" piece, exactly what the user framed
on 2026-05-13: *"a basis-set optimizer as a separate module that
can run external programs, similar to vq."* Goal 8 is its first
production user.

The sketch below was the original 2026-05-13 plan; the actual current
layout (verified against the tree) has grown past it: `crystal14.py`
was retargeted to `crystal.py` (CRYSTAL23, v0.3.0), and `transports/`
and `recipes/` are no longer "planned," they're implemented:

```
vibe-basis/
├── pyproject.toml              ← hatchling, MPL-2.0, mirrors vq's
│                                  pyproject; entry point: vb
├── README.md                   ← scope + relationship to vibe-qc + vq
├── VERSIONING.md                ← version history
├── TUTORIAL.md                  ← the optimization-pipeline walkthrough
├── conftest.py                 ← src/ on sys.path for in-tree tests
├── src/vibe_basis/
│   ├── __init__.py
│   ├── cli.py                  ← top-level click CLI (``vb --version``, etc.)
│   ├── engine.py, cache.py, optimize.py, pipeline.py, workdir.py
│   ├── backends/               ← one module per external SCF program
│   │   ├── __init__.py
│   │   ├── crystal.py          ← targets CRYSTAL23; parse_output,
│   │   │                          probe_version, emit_input, emit_input_inline
│   │   └── crystal_atom.py
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── crystal23.py
│   │   └── gpaw.py
│   ├── transports/             ← how external SCF jobs reach a CPU
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── local.py
│   │   └── vq.py
│   ├── recipes/                ← end-to-end driver scripts
│   │   ├── __init__.py
│   │   ├── pob_parity.py
│   │   ├── remote.py
│   │   └── run_d12.py
│   └── io/
│       ├── __init__.py
│       ├── library_bridge.py
│       ├── references.py
│       └── structures.py
└── tests/                      ← lightweight Python-only suite
```

vibe-basis does **not** import vibe-qc. The two packages coexist
in the core repository and are independently installable:

* ``./vibe-basis/scripts/install.sh`` creates a dedicated environment with
  vibe-basis only; the package is not on PyPI yet.
* Optional extras: ``[nlopt]``, ``[iminuit]``, ``[scipy]``, and ``[all]``.
  The compatibility ``[vq]`` extra is dependency-free; install the external
  vq CLI from its separate vibe-queue repository through its own lifecycle.
* Academic collaborators get the optimizer without vibe-qc's
  libint / libxc / libecpint / FFTW3 footprint.

The vibe-qc-side bits that **stay in vibe-qc**:
* The 39-compound structure DB at `examples/basisset_dev/
  _structures.py` and the .d12 generator at `_generator.py`
  are basis-orchestration code that should also migrate to
  vibe-basis in a follow-on commit (queued in TODO). For now
  they remain in vibe-qc; vibe-basis recipes will import them
  via path-prepend until the migration lands.
* `docs/basisset_dev/` doc tree (this file, GOAL4_DESIGN.md,
  REQUIREMENTS-PERIODIC.md, etc.) stays in vibe-qc since
  basissetdev is a vibe-qc work-track tag.

Tests are lightweight Python-only, they don't require a
CRYSTAL14 installation. Real end-to-end smoke runs against
compute-reference live in `vibe-basis/src/vibe_basis/recipes/` (when wired)
to keep the suite laptop-fast.

## 7. Open questions parked for Stage 0 / 1

* **CRYSTAL14 thread-safety** under `--max-jobs > 1` per the v0.5.x
  daemon. Confirmed serial works; parallel-rank single CRYSTAL14
  works; **two CRYSTAL14s running in different vq slots** is
  unknown. Test in Stage 0.
* **CRYSTAL14 output stability** across runs (same input, same
  hardware, same OMP_NUM_THREADS, does it give bit-identical
  totals?). Affects MIGRAD HESSE error estimation. Verify in
  Stage 0.
* **vq output-parser robustness** to CRYSTAL14's known
  `STARVED`-mis-kill in v0.5.9 (already documented: needs
  `--wall-time-seconds N`). Build defensive parser that
  distinguishes "compound failed to converge" from "vq killed
  the job."

## 8. Implementation order (next 5 milestones for this chat)

1. **Stage 0 pipeline plumbing** (≈ 200 LoC Python):
   `crystal_output_parser.py` + `transport_vq.py` + the recipe
   driver `stage0_pob_parity.py`.
2. **Run Stage 0** on compute-reference; iterate until the 13-compound
   sum-of-deltas < 0.1 mHa.
3. **Stage 1 H atom**: 30 BOBYQA evals on compute-reference; ~minutes wall.
4. **Stage 2 LiH single-compound**: 50 BOBYQA evals + MIGRAD
   refinement; ~20 min wall.
5. **Write Stage-2 result-section paragraph + outreach email
   draft** (per user 2026-05-13 collaboration gate).

## 9. References

* M. F. Peintinger, D. Vilela Oliveira, T. Bredow, *J. Comput.
  Chem.* **34**, 451 (2013). [PT2013]
* D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow,
  *J. Comput. Chem.* **40**, 2364 (2019). [VO2019]
* T. Bredow, A. R. Gerson, *Phys. Rev. B* **61**, 5194 (2000).
  [PW1PW, explains why mpei-TZVP is needed: HF is a *different*
  basis optimum than PW1PW.]
* M. J. D. Powell, *The BOBYQA algorithm for bound constrained
  optimization without derivatives*, Cambridge NA Report
  NA2009/06.
* `reference_crystal14_via_vq.md` (chat memory).
* `reference_vq_queue.md` (chat memory).
* [REQUIREMENTS-PERIODIC.md](REQUIREMENTS-PERIODIC.md), for the
  R3 AFM gate that defers the TM-oxide subset.
* [GOAL4_DESIGN.md](GOAL4_DESIGN.md), for the prior in-process
  vibe-qc optimization design (Goal 4); Goal 8 takes a different
  SCF engine but inherits the parameter / driver design.
