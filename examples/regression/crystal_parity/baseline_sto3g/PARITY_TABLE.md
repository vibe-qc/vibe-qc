# v0.8.0 STO-3G RHF parity baseline — CRYSTAL14 reference

CRYSTAL14 runs from `baseline_sto3g/*.d12` — all under defaults:
`SHRINK 8 8`, `TOLDEE 8`, no FMIXING, no LEVSHIFT.
Energies are **per primitive cell (1 formula unit)** as CRYSTAL14 prints them.

## Reference numbers (CRYSTAL14, sealed)

| System     | E_total/FU (Ha) | E_NN/FU (Ha)   | E_kin/FU (Ha) | Virial | Cycles | Status |
|------------|-----------------|----------------|---------------|--------|--------|--------|
| LiH        |    -7.93816810  |   -3.39397848  |    7.87893408 | 0.9963 | 12 | **clean** |
| MgO        |  -271.21814708  |  -73.08427668  |  267.12905642 | 0.9924 | 13 | **clean** |
| NaCl       |  -614.65306571  | -107.30632749  |  603.46676919 | 0.9908 | 11 | **clean** |
| LiF        |  -105.63898757† |   (n/a)        |   (n/a)       | (n/a)  | 11 | **accepted-at-CYC-11**; watchdog mis-killed at 305s on every retry; energy decomp + virial only printed in the final block CRYSTAL14 never reached |
| C-diamond  |   -74.87695367  |  -28.76942751  |   74.09986651 | 0.9948 | 13 | **clean** |
| Si-diamond |  -571.31994169  | -102.89353078  |  563.36902558 | 0.9930 | 14 | **clean** |

† = LiF intermediate at CYC 11 (DETOT = 1.14e-8, **stable to 1e-8 Ha**
across CYC 8–11). One or two more cycles would have stamped
`SCF ENDED`, but the daemon's CPU-starvation watchdog (`CPU < 5% for
≥ 300s`) keeps mis-killing the job at 305s elapsed every retry —
`mpirun`'s parent PID samples near-0% even though all 14 ranks are
running. Both `1c974bc2dd8c` and `9faaa790a947` resubmits hit the
same wall.  Decision: accept the 8-digit-stable CYC-11 value as the
parity reference; expected drift in cycles 12–13 is below the
v0.8.0 parity tolerance (10⁻⁶ Ha) anyway.  See "Known caveat"
section below.

"Clean" = converged under defaults, no `POSSIBLY CONDUCTING STATE`
warning, no SCF stabilizer needed.  **Five of six confirmed.**

## Band edges (insulator sanity check)

| System     | TOP VAL (Ha)  | BOTTOM VIRT (Ha) | Indirect gap (Ha / eV) | Insulating? |
|------------|---------------|------------------|-----------------------|-------------|
| LiH        | -0.2969 (K21) | +0.0309 (K21)    | 0.328 / 8.92           | yes |
| MgO        | -0.1474 (K11) | +0.3189 (K1)     | 0.466 / 12.69          | yes |
| NaCl       | -0.2992 (K12) | +0.4567 (K1)     | 0.756 / 20.57          | yes |
| LiF        | -0.3493 (K11) | +0.3723 (K1)     | 0.722 / 19.64          | yes |
| C-diamond  | -0.1223 (K1)  | +0.4367 (K1)     | 0.559 / 15.21          | yes |
| Si-diamond | -0.0612 (K1)  | +0.2491 (K1)     | 0.310 /  8.43          | yes |

(STO-3G overestimates band gaps wildly; here we just need them to be
finite and well-separated, which they are.)

## Method (CRYSTAL14)

```
RHF /STO-3G
SHRINK 8 8         (8×8×8 Monkhorst-Pack + Gilat secondary mesh)
TOLDEE 8           (10⁻⁸ Ha SCF tolerance)
no FMIXING, no LEVSHIFT, no BROYDEN
```

## vibe-qc results — EWALD_3D and GDF (gauge fix Steps 1+2+3′)

vibe-qc runs the **8-atom conventional cell at Γ-only**.  For
SG 225 / SG 227 that is a *4-primitive-cell* supercell sampled at
Γ — equivalent to a coarse 4-point primitive-cell BZ sampling,
markedly coarser than CRYSTAL14's `SHRINK 8 8`.

Energies below are **per 8-atom conventional cell** (the raw
vibe-qc number); the `/FU` column normalises (÷4 for rocksalt,
÷8 for diamond — 1 FU = 1 cation+anion pair, or 1 C/Si atom).

| System     | EWALD_3D /cell  | GDF /cell       | GDF − EWALD_3D | conv? | vibe-qc /FU | CRYSTAL14 /FU | Δ vs CRYSTAL14 |
|------------|-----------------|-----------------|----------------|-------|-------------|---------------|----------------|
| LiH        |   -29.33212849  |   -29.33212849  | **0.0** (bit-exact) | ✓ 11 |   -7.333    |   -7.93817    | +0.61 |
| MgO        |  -952.79185671  |  -952.79185671  | **0.0** (bit-exact) | ✓ 10 |  -238.198   | -271.21815    | +33.02 |
| NaCl       | -1820.58661638  | -1820.58660297  | 1.3e-5 (both unconverged) | ✗ 30 |  (n/a)      | -614.65307    | — |
| LiF        |   (wall cut)    |  -376.92174979  | — | ✗ 30 |  (n/a)      | -105.63899    | — |
| C-diamond  |   (wall cut)    |  -254.09516889  | — | ✓ 15 |  -31.762    |  -37.43848    | +5.68 |
| Si-diamond |   (wall cut)    |   (wall cut)    | — | — |  —          | -285.65997    | — |

### Reading the table

**1. The gauge fix is validated.** Where both vibe-qc paths
converge (LiH, MgO), **GDF reproduces EWALD_3D bit-for-bit**.
That was the whole point of the 2026-05-13/14 gauge regression
fix (Steps 1+2+3′ on `origin/main`; the in-tree handover brief
was removed in the 2026-05-17 archive sweep — recover via
`git log --diff-filter=D --name-only -- docs/`):
the native GDF driver's V_ne / e_nuc / J / K are now all in one
consistent Ewald gauge.  Pre-fix, LiH GDF diverged to −729 Ha.

**2. NaCl + LiF don't converge — SCF-acceleration hardness, not
a gauge bug.** NaCl fails to converge in *both* EWALD_3D and GDF
(and lands at the same energy to 1.3e-5 Ha — i.e. both are
bouncing around the same fixed point, just not tightly).  This
was the Anderson / Broyden / SOSCF story, then on the v0.8.x
roadmap; Anderson/Broyden mixing (2026-06-20) and a proper
quasi-Newton SOSCF (2026-07-08) have since shipped
(`python/vibeqc/periodic_density_mixing.py`, `SOSCFOptions`) —
exactly the class of problem the user said not to chase
alongside gauge bugs at the time, worth re-checking whether NaCl/LiF
converge with these accelerators now available.

**3. The `Δ vs CRYSTAL14` column is large and *not yet*
explained — a separate, open issue.** It is **not** the gauge
bug (the gauge fix is proven done by column 4).  It scales with
electron count (+0.6 Ha/FU LiH → +5.7 Ha/C C-diamond → +33 Ha/FU
MgO), which rules out BZ-sampling difference as the cause —
BZ sampling on an insulator does not scale with Z.  Working
hypothesis: **lattice-sum cutoff truncation** — the vibe-qc
sweep used the default `lattice_opts.cutoff_bohr` /
`nuclear_cutoff_bohr` with no explicit setting, and tight ionic
cells with core electrons need larger cutoffs.  It hits EWALD_3D
and GDF *equally* (they agree bit-exact), so it lives in the
**standard (no-fitting) layer**, not the DF layer.  Tracked as
the first concrete target of the Phase 4 EWALD_3D all-Bravais
audit — re-run with a cutoff convergence sweep before claiming
any CRYSTAL14 parity.

> **Bottom line:** the v0.8.0 gauge-regression fix is **done and
> validated**.  The vibe-qc↔CRYSTAL14 absolute-parity gap is a
> *different* problem (cutoff convergence + BZ sampling) and is
> Phase 4 work; it does not reopen the gauge fix.

## Job IDs (vq, compute-reference)

```
CRYSTAL14 baseline
  LiH         b173d2e3fa9a   completed
  MgO         7b4e94171cbe   completed
  NaCl        4c26ed4bc898   completed
  LiF         1c974bc2dd8c   STARVED (watchdog mis-kill at CYC 11)
  LiF (resub) 9faaa790a947   STARVED (same watchdog issue; accepted CYC-11)
  C-diamond   6427283bb895   completed
  Si-diamond  b553ffdcbee5   completed

vibe-qc EWALD_3D sweep   95952ca4842f   time_exceeded (LiH+MgO+NaCl done, LiF+ cut)
vibe-qc GDF sweep        5e076965b1f5   time_exceeded (LiH+MgO+NaCl+LiF+C done, Si cut)
LiH GDF A′ diagnostic    a604f94033a4   killed (redundant; local run confirmed first)
```

## Known caveat: vq CPU-starvation watchdog mis-kill on parallel CRYSTAL

The vq daemon has a CPU-starvation watchdog that SIGTERMs any
job whose parent-PID CPU% < 5.0 for ≥ 300s of elapsed wall time.
For parallel CRYSTAL14 the wrapper script `exec`s `mpirun` which
in turn spawns the 14 `Pcrystal` ranks; the daemon samples
`mpirun`'s parent CPU which sits near 0% (the parent waits on
its ranks) — so the watchdog can't tell that the 14 ranks are
running at full tilt.

Concrete evidence:

* `1c974bc2dd8c` (first LiF run): CYC 0–11 visible in `.out`,
  each ~25s, all 14 ranks busy. Killed at elapsed 307s.
* `9faaa790a947` (resubmit, identical input + longer wall):
  same pattern; killed at elapsed 308s.

This affects any CRYSTAL14 calculation needing more than ~290s.
For LiH/MgO/NaCl/C/Si baseline (each well under 290s) we just
escape it. For LiF and any subsequent denser-basis or larger-cell
parity work the workaround is one of:

1. **Accept the last-printed energy** (what we did here for LiF).
   When DETOT is already ≤ TOLDEE in absolute magnitude, the
   energy is correct to that tolerance.
2. **Run on the laptop locally** (slower but no daemon).
3. **Wait for queue chat** to ship a fix that samples the
   process *group*, not just the parent PID — feedback filed.

Tracked as queue feedback. Not a CRYSTAL14 bug; not a vibe-qc bug.
