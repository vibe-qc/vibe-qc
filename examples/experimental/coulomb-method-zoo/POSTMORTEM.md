# coulomb-method-zoo: POSTMORTEM

Experimental cross-pollination spike on alternative periodic Coulomb J
methods. Theme set by the original brief; this is the throwaway closeout
per `feedback_experimental_chat_policy`.

**Branch:** intended `experimental/coulomb-method-zoo`; that branch was
already checked out by another worktree at HEAD-of-main with no commits,
so the work landed on `claude/jolly-bohr-97bce6` instead. Commits are
prefixed `experiment:` and can be cherry-picked / fast-forwarded onto
the experimental branch by hand.

**Test system:** H₂ in a 12-bohr vacuum cubic box / sto-3g, 2 BFs/cell.
Chosen because (a) the existing `run_rhf_periodic_gamma_ewald3d` driver
is structurally broken on MgO (per `project_periodic_rhf_gdf_spike`
memory: oscillates, +241 Ha vs PySCF, 6229 s wall, confirmed by
killing a baseline run after 102 CPU minutes), and (b) the system
prompt's `run_rhf_periodic_gamma_gdf` driver does not exist on `main`.
A spike comparing alternative builders against a broken reference is
information-free, so we pivoted to molecular-limit H₂ where the
existing builders converge and a direct molecular reference
(`compute_eri × D`) is unambiguous.

**Reference:** `J_full = compute_eri × D` at the converged
molecular-RHF density, giving `½ Tr(D·J_full) = 1.34897753 Ha`.

## Headline results: J-builder accuracy at converged D

| method | ‖ΔJ‖_F vs J_full | ½ Tr(D·J) | ΔE_J vs reference | wall_s |
|---|---:|---:|---:|---:|
| `J_full` (compute_eri ref) | - | 1.34898 | (ref) | - |
| **EWALD3D** (ω=0.5, h=0.3) | 7.6e-1 | 0.90227 | −0.4467 | 0.33 |
| **PLAIN_EWALD** (ω=0.5, G/2ω≤8) | 7.6e-1 | 0.90227 | −0.4467 | 0.014 |
| **WOLF** (α=0.5) | 1.4e+0 | 0.49800 | −0.8510 | 0.004 |
| **ADFT** (aux=6-31g, n_aux=4) | **1.5e-2** | **1.33995** | **−0.0090** | 0.002 |

The 0.45 Ha offset between `EWALD3D`/`PLAIN_EWALD` and `J_full` is the
**Makov-Payne / G=0 gauge shift**: `J_ewald = J_full − (α_M Q/L)·S`.
It is not error; it is the gauge that gets cancelled by the matching
Ewald-aligned V_ne in periodic SCF (`madelung_energy_correction` does
the bookkeeping in the release path). The two Ewald methods agree
with each other to ~1e-7 ‖ΔJ‖_F across ω.

## Per-method verdicts

### PLAIN_EWALD: graduates as a sanity oracle

`PLAIN_EWALD` is the analytic G-vector sum for Gaussian-product
densities (no FFT, no real-space density grid). For an s-only basis
the per-pair Fourier transform is closed-form via the Gaussian-product
theorem; the implementation is ~50 LOC of NumPy.

**Result:** PLAIN_EWALD agrees with the existing FFT-Poisson `EWALD3D`
to ~1e-7 ‖ΔJ‖_F across ω ∈ [0.2, 2.0] when EWALD3D's grid is tight
(h ≤ 0.3 bohr). The grid sweep converges monotonically:

```
h=0.60 bohr    ‖J_E3D - J_PE‖_F = 3.96e-3
h=0.50          1.07e-3 .. wait .. 2.82e-4
h=0.40          1.52e-5
h=0.30          8.4e-7    (FFT noise floor)
h=0.15          8.2e-7    (no improvement)
```

**Use it for:** independent verification that future J builders
(native GDF, FMM, …) compute the right Coulomb energy on a Gaussian
density. Also a clean check that the `ewald_j` FFT-Poisson path has
no implementation bug, at h=0.3 it sits exactly at the analytic
limit. **Don't promote it to a production method**: at large basis
size the analytic G-vector sum scales as O(n_G · n_bf²) with no FFT
acceleration; the FFT-Poisson path wins asymptotically.

**Limitation:** s-shells only. Generalising to p/d/… is a
straightforward G-polynomial extension (Helgaker-Jørgensen-Olsen ch. 9)
but is roughly a half-day of work and not blocking anything in this
spike. Listed in the spinoff todos.

### ADFT: tentative graduate, blocked on libint AM ceiling

The variational Coulomb-fitting approach (Mintmire-Dunlap, Köster).
At converged molecular D and aux=6-31g:

- **‖ΔJ‖_F = 1.5e-2** (50× tighter than the EWALD gauge offset)
- **wall = 2 ms** (one Cholesky-solve plus two einsums)
- ΔE_J = −9 mHa (the variational fit error)

Aux quality is the dominant variable. With **rank-deficient** auxes
(sto-3g, sto-6g, only n_aux=2 against an AO product space of dim 3),
ADFT is forced into a 153 mHa fit error. With one extra contraction
(6-31g, n_aux=4) the error collapses by 17×.

**Blocker:** `vq.compute_2c_eri` segfaults on any aux containing
**p-shell or higher** functions. Confirmed on cc-pvdz (max_l=1,
n_aux=10), def2-svp (max_l=1), def2-svp-jk (max_l=2, n_aux=36). Only
s-only auxes are workable in the current build; for H this caps us
at 6-31g.

**This is NOT a libint build-config issue.** The vendored libint at
`third_party/libint/install/include/libint2/config.h` reports:

```
LIBINT_MAX_AM           = 5
LIBINT_ERI_MAX_AM_LIST  = 5,4,3   (deriv 0/1/2)
LIBINT_ERI2_MAX_AM_LIST = 5,4,3
LIBINT_ERI3_MAX_AM_LIST = 5,4,3
```

-- way more capable than what cc-pvdz / def2-svp-jk demand. The bug
is in vibe-qc's DF C++ wrapper (`cpp/src/df.cpp`) or its libint Engine
usage. **Same bug class as the tracked v0.7.3 DF SCF SIGSEGV**
(`examples/debug/df_smoke.py`, commit 196f205). This spike produces
a tighter reproducer that bypasses the SCF:

```python
import vibeqc as vq
mol = vq.Molecule([vq.Atom(1,[0,0,0]), vq.Atom(1,[1.4,0,0])], 0, 1)
vq.compute_2c_eri(vq.BasisSet(mol, 'cc-pvdz'))   # SIGSEGV (exit 139)
```

See `bug_compute_2c_eri_pshell.py` in this directory for the dev-chat
handoff version.

To make a real ADFT verdict we need this fixed in `cpp/src/df.cpp`,
not a libint rebuild. Separate dev-chat task not appropriate for this
spike.

**Tentative verdict:** ADFT is the only method in the zoo that
produces J directly (no Ewald gauge offset), at one-tenth the wall
of EWALD3D, with a controllable error knob (aux quality). It is
mathematically clean (variational stationarity → Pulay cancellation
in gradients → no surprise terms when we eventually do periodic
geometry opt) and composes naturally with the planned native-GDF
Method 1. **Recommend graduating to a follow-on chat once libint
is fixed**, with bench targets on LiH and second-row covalents and
a real def2-svp-jk aux comparison.

The deMon2k angle in the original brief, borrowing GEN-A2/A2* hand-
optimised auxes, remains attractive but is downstream of native
GDF landing in vibe-qc, not this spike. The pob-TZVP-aux paper path
discussed in chat is a separate research project per the existing
`pob_tzvp_aux_basis` memory.

### WOLF: rejected for Gaussian AOs

Wolf summation truncates the real-space Coulomb at R_c with a smooth
shifted-force kernel. The accuracy claim from Wolf et al. JCP 110,
8254 (1999) is ~1e-3 to 1e-4 on point-charge systems; the question
here was whether that survives on Gaussian AO densities.

**Sweep across α ∈ {0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0}** at
fixed cutoff_bohr=12:

```
α=0.1   ‖ΔJ‖_F = 2.16e-1   ΔE_J = -0.127 Ha
α=0.5   ‖ΔJ‖_F = 1.44e+0   ΔE_J = -0.851 Ha
α=5.0   ‖ΔJ‖_F = 2.26e+0   ΔE_J = -1.338 Ha
```

The "smooth shifted-force kernel" benefit doesn't materialise, the
‖ΔJ‖_F monotonically degrades with α, and even the best point
(α=0.1, mild screening) is only as accurate as the Ewald gauge offset.
Worse, α=0.1 is essentially "full Coulomb in real space inside the
12-bohr cutoff", at that point we're not really doing Wolf, we're
just lattice-summing 1/r and getting limited by the cutoff radius.

**Verdict: rejected.** Wolf was designed for point charges where the
shifted-force discontinuity correction has a clean interpretation;
on Gaussian AO products that benefit doesn't transfer cleanly, and
there's no parameter regime where it competes with EWALD3D or ADFT.
Don't promote to a production method, don't extend the implementation.

### EWALD3D: already in main, this spike doesn't change the verdict

The release-path J builder (`build_j_ewald_3d`). For our purposes here
it's the reference for Ewald-gauge methods and we observed nothing
that contradicts the existing release-path validation. The grid
sweep showed h=0.3 bohr is the appropriate spacing for sto-3g (FFT
noise floor); h=0.5 is too coarse (~mHa-level extra error in J), h=0.2
is wasted compute.

### Methods not implemented

- **PME** (Particle-Mesh Ewald with B-spline density spreading):
  would require new infrastructure (cardinal-B-spline density-spreading
  onto the FFT grid). Not done. The existing `EWALD3D` is "Ewald
  via direct rasterisation onto an FFT grid", which is the same
  infrastructure modulo the spreading kernel; PME is unlikely to
  produce a different *accuracy* answer at the same grid spacing.
  Where PME would win is on systems where the non-spread density
  needs aggressive grid resolution; on Gaussian AOs that's not
  obviously a bottleneck. **Out of scope** for this spike.
- **FMM / PFMM**: O(N) hierarchical multipole expansions. Continuous-
  FMM is on the official roadmap as Method 5 already; no need for
  this spike to duplicate that path. **Out of scope.**
- **Damped-shifted-force / SPME / wavelet-Poisson / multigrid /
  pseudospectral**: lower-priority items in the original brief; with
  WOLF showing the limitations of the "skip reciprocal space" family
  on Gaussian AOs, only the wavelet path looks distinct enough to
  be worth a separate spike. **Out of scope here.**

## What stays for the spinoff chat

```
[ ] Fix the compute_2c_eri segfault on p-shell auxes. NOT a libint
    config issue, libint is fine (LIBINT_MAX_AM=5, ERI2/ERI3 MAX_AM
    lists are 5,4,3). Bug is in vibe-qc's cpp/src/df.cpp DF wrapper
    or its libint Engine usage. Same bug class as the tracked v0.7.3
    DF SCF SIGSEGV (df_smoke.py, commit 196f205). Tighter reproducer
    in this directory: bug_compute_2c_eri_pshell.py.
[ ] bench_lih_jonly.py, same J-only structure on LiH (Li 1s + 2s,
    confirms ADFT scaling beyond a one-electron species).
[ ] bench_h2_scf.py, re-run the SCF benchmark with the fixed
    gauge pairing rule. The rule:
        EWALD3D   / PLAIN_EWALD  → CoulombMethod.EWALD_3D V_ne,
                                   Madelung correction = 0
        WOLF      / ADFT         → CoulombMethod.DIRECT_TRUNCATED
                                   V_ne with cutoff_bohr ≥ 30 (must
                                   exceed cell so lattice sum reaches
                                   isolated-molecule limit), Madelung
                                   correction = 0
    Verdict: do the SCF totals match the J-only ‖ΔJ‖_F predictions?
[ ] PLAIN_EWALD generalisation to p/d shells (G-polynomial factors).
    Validates the s-only path's correctness as a free byproduct.
[ ] WOLF α sweep on a larger system to see if there's a regime where
    the shifted-force form pays off (e.g. NaCl rocksalt vs molecular
    NaCl pair, where the LR ~ 1/r tail truly dominates).
[ ] Once libint is fixed: re-run the ADFT aux sweep with def2-svp-jk
    and def2-tzvp-jk. Expected: aux=def2-svp-jk hits ‖ΔJ‖_F < 1e-3,
    confirming ADFT graduation. If it doesn't, there's a bug.
[ ] Aux-basis automated optimisation paper, separate paper-scope chat
    per `project_pob_tzvp_aux_basis` memory. Sketched the design space
    in the chat: even-tempered parameterisation + Minuit2 (or analytic
    gradients) + Dunlap loss on a training set + linear-dependence
    barrier. AutoAux as a starting point. Not this chat.
```

## Files left in the worktree

```
examples/experimental/coulomb-method-zoo/
  README.md, math, scope, why we pivoted off MgO
  POSTMORTEM.md, this file
  j_builders.py, pluggable J-builders (EWALD3D, WOLF,
                         PLAIN_EWALD, ADFT)
  scf_driver.py, tiny pluggable Γ-RHF driver for the SCF
                         experiments (gauge-aware in WOLF/ADFT vs
                         EWALD branches; needs cutoff_bohr fix per
                         the spinoff todo)
  bench_h2.py, full SCF sweep for H₂; useful as a template
                         once the gauge bookkeeping is clean
  bench_h2_jonly.py, J-builder accuracy comparison at converged D
                         (the actual primary deliverable of this spike)
  results/
    bench_h2_jonly.json, JSON of all five sweeps
```

## What this spike learned that the next chat needs

1. **`run_rhf_periodic_gamma_ewald3d` is broken on tight bulk cells**, confirmed by the 102-CPU-minute MgO run that produced no output. The system-prompt description of `run_rhf_periodic_gamma_gdf` etc was aspirational; **don't trust periodic SCF benchmarks against vibe-qc reference until that driver lands**.
2. **vibe-qc's `compute_2c_eri` segfaults on any p-or-higher aux**, despite libint being correctly built for max_am=5. The bug is in `cpp/src/df.cpp` or the libint Engine usage there, not in libint itself. Same class as the tracked v0.7.3 DF SCF SIGSEGV. Hard blocker for any ADFT work that needs a richer aux than 6-31g (which on H is the only working ≥4-aux option).
3. **The Makov-Payne gauge shift between Ewald-gauge and molecular-gauge J builders is huge** in finite cells (~0.45 Ha for H₂ in a 12-bohr box). Comparing raw J matrices across methods without accounting for the V_ne pairing gives nonsensical answers, a trap that ate ~1 hour of debugging in this chat.
4. **PLAIN_EWALD is exactly EWALD3D** (to FFT-noise floor), which is a useful negative result: the FFT-Poisson path in `ewald_j` is correctly implemented; future J-builder bugs need to be looked for elsewhere.
5. **ADFT is the most promising lead** but the data point is currently one-system-one-aux. Don't over-claim until LiH + def2-svp-jk numbers exist.
