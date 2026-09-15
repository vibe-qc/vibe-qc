# HF-on-`pob-TZVP-rev2` failure scout (2026-05-18)

**Branch:** `basissetdev`.
**Owner:** basissetdev chat.
**Context:** [`GOAL8_MPEI_TZVP.md`](GOAL8_MPEI_TZVP.md), motivation
for an HF-optimized basis-set sibling to `pob-TZVP-rev2`.

## TL;DR

3 of the 13 simplest closed-shell wide-gap ionic insulators in the
PT2013 cubic-ionic test set, **LiF, CaO, MgO**, **fail HF SCF on
`pob-TZVP-rev2`** through linear-dependence-driven DIIS collapse,
while **PBE and PW1PW on the same basis converge cleanly in 7-9
cycles** for the same compounds. This is the empirical evidence
that the rev2 basis, optimized for PW1PW, is not safe to reuse
at the HF level on the same test set, and motivates a separate
HF-optimized basis (the Goal 8 `mpei-TZVP` deliverable).

## Methodology

* **Test set:** the 13 cubic-ionic compounds of PT2013 Table 4
  (`vibe_basis.io.references.PT2013_T4_cubic_ionic_names`):
  LiCl, NaCl, LiF, NaF, KF, KBr, CaF₂, K₂O, MgO, CaO, LiH, NaH, KH.
* **Basis:** `POB-TZVP-REV2` (Vilela Oliveira, Laun, Peintinger,
  Bredow, *J. Comput. Chem.* **40**, 2364 (2019)).
* **SCF engine:** CRYSTAL23 (the `crystal23demo` binary at
  `/home/USER/bin/crystal23demo` on compute-reference, CRYSTAL14 doesn't
  recognize the `POB-TZVP-REV2` keyword, predating it by 5 years).
* **Methods compared:** HF (all 13); PBE + PW1PW (the 3 HF
  failures, as confirmation).
* **Defaults:** `SHRINK 8 8` (Pack-Monkhorst), `TOLDEE 8`,
  default `TOLINTEG` (Coulomb / exchange 6-cutoff family).
* **Transport:** `vq submit compute-reference -d <dir> --cpus 4
  --wall-time-seconds 900 -- bash run.sh`; the `run.sh` wrapper is
  one line of `crystal23demo < <name>.d12 > <name>.out 2>&1`.
* **Per-compound primitive cell:** 2 atoms (CRYSTAL builds from
  the asymm-unit form the `vibe_basis.backends.crystal14.emit_input`
  emitter produces, `crystal_spacegroup` + `crystal_asymm_unit`).

Decks were emitted via
`vibe_basis.backends.crystal14.emit_input(struct, basis='pob-tzvp-rev2',
method='hf' | 'pbe' | 'pw1pw')`. Raw outputs are under
`.claude/scout/_fetched/` (gitignored, transient, this report
is the permanent record).

## Results

### Hartree-Fock on `pob-TZVP-rev2`: full 13-compound run

| Compound | State | Cycles | E (Ha, primitive cell) | ΔE vs PT2013 SI T2 (mHa) |
|---|---|---|---|---|
| LiCl | converged   |  9 |  −467.093426 |  −5.958 |
| NaCl | converged   |  8 |  −621.503589 |  −7.645 |
| **LiF**  | **BLOW-UP at cyc 50** | 50 | (E_42 = +661.6 Ha) | - |
| NaF  | converged   | 22 |  −261.472793 | (PT2013 n-dash) |
| KF   | converged   | 11 |  −698.756062 | −51.547 |
| KBr  | converged   | 10 | −3171.700685 | (not in PT2013 T2) |
| CaF₂ | converged   | 13 |  −875.992414 | −47.124 |
| K₂O  | converged   | 13 | −1273.233784 | −48.364 |
| **CaO**  | **BLOW-UP at cyc 2** | 50 | −1464.44 (non-physical metallic basin) | −712 631 |
| LiH  | converged   | 10 |    −8.060246 | +2.591 |
| NaH  | converged   |  8 |  −162.455557 | −1.748 |
| KH   | converged   |  9 |  −599.744570 | −21.314 |
| **MgO**  | **BLOW-UP at cyc 50** | 50 | −245.476863 (last 49 cycles tracked −274.685, ≈3 mHa from PT2013 HF reference) | +29 205 |

10 converged, 3 failed.

### DFT confirmation runs (same basis, same systems, same defaults)

| Compound | Method | State | Cycles | E (Ha, primitive cell) |
|---|---|---|---|---|
| LiF | PBE     | converged | 7 | −107.442290 |
| LiF | PW1PW   | converged | 7 | −107.518397 |
| CaO | PBE     | converged | 9 | −752.687208 |
| CaO | PW1PW   | converged | 8 | −752.968294 |
| MgO | PBE     | converged | 9 | −275.318119 |
| MgO | PW1PW   | converged | 8 | −275.476167 |

6/6 converged. PBE/PW1PW for CaO lands at −752.7 Ha, the expected
energy ballpark for a 28-electron primitive cell, ≈712 Ha higher
than the HF SCF's non-physical −1464 Ha basin.

## Failure-mode anatomy

All three HF failures look like **linear-dependence-driven DIIS
extrapolation onto a non-physical eigenvector of the
near-singular overlap matrix**. Trace excerpts:

**MgO**: the textbook case. Tracks the right answer for 45+
cycles, then catastrophically jumps:

```
 CYC  45 ETOT(AU) -2.746853359092E+02 DETOT -3.13E-06 …
 CYC  46 ETOT(AU) -2.746853622106E+02 DETOT -2.63E-05 …
 CYC  47 ETOT(AU) -2.746853656540E+02 DETOT -3.44E-06 …
 CYC  48 ETOT(AU) -2.746852990921E+02 DETOT  6.66E-05 …
 CYC  49 ETOT(AU) -2.746849595747E+02 DETOT  3.40E-04 …  ← onset
 CYC  50 ETOT(AU) -2.454768633645E+02 DETOT  2.92E+01 …  ← jump (+29 Ha)
```

**LiF**: tracks the right answer to 7 decimal places for 38
cycles, then the DIIS subspace destabilises:

```
 CYC  38 ETOT(AU) -1.070842528457E+02 DETOT  1.91E-05 …
 CYC  39 ETOT(AU) -1.070842562884E+02 DETOT -3.44E-06 …
 CYC  40 ETOT(AU) -1.070180146296E+02 DETOT  6.62E-02 …  ← onset
 CYC  41 ETOT(AU) -6.101991142611E+01 DETOT  4.60E+01 …  ← jump (+46 Ha)
 CYC  42 ETOT(AU)  6.616344225348E+02 DETOT  7.23E+02 …  ← (+723 Ha)
 …
 CYC  49 ETOT(AU) -1.070816643507E+02 DETOT -8.64E+02 …  ← swings back
 CYC  50 ETOT(AU) -1.070842154332E+02 DETOT -2.55E-03 …
```

**CaO**: fails earliest. Cycle 0 lands at −751.6 Ha (close to
PT2013's −751.81 Ha unrevised pob-TZVP HF reference), cycle 2
jumps by −576 Ha into a non-physical metallic basin
(CRYSTAL annotates "POSSIBLY CONDUCTING STATE, EFERMI(AU) ..."
which is wrong: CaO is an 8 eV-gap insulator), and stays there:

```
 CYC   0 ETOT(AU) -7.516248894476E+02 DETOT -7.52E+02 …  ← physical start
 CYC   1 ETOT(AU) -7.618888584213E+02 DETOT -1.03E+01 …
 CYC   2 ETOT(AU) -1.337754381488E+03 DETOT -5.76E+02 …  ← collapse
 CYC   3 ETOT(AU) -1.396092086669E+03 DETOT -5.83E+01 …
```

All three failures share:

* **No legitimate physical difficulty.** Wide-gap, closed-shell,
  cubic Fm-3m ionic insulators are about the easiest periodic
  HF cases that exist. The same SCF engine, the same basis, the
  same defaults, but with a 20%-HF PW1PW or pure-DFT PBE
  functional, finishes in 7-9 cycles.
* **CRYSTAL's annotation of pathology.** All three runs print
  `POSSIBLY CONDUCTING STATE` warnings during the unstable
  cycles, a sign that the eigenstructure of the Fock matrix is
  going through a near-degeneracy that DIIS amplifies.
* **Sensitivity to the basis, not the functional.** Same
  geometry + same SHRINK + same TOLDEE + same TOLINTEG; the only
  variable is the basis-and-functional pairing.

## Implication

The `pob-TZVP-rev2` exponent/contraction set was optimised
against PW1PW (Bredow-Gerson 2000, 20 % HF + 80 % PW91), which
includes substantial dynamical screening. The same exponents at
the **pure HF** level encounter the unscreened 1/r long-range
exchange tail, which integrates over more lattice images and
drives the periodic overlap matrix closer to singularity at the
LD-floor exponent (0.15 in rev2). DIIS subspace eigenvalues
straddle the LD margin and extrapolation occasionally projects
onto the unphysical eigenvector, the SCF then collapses.

**This is exactly the failure pattern that motivates Goal 8.**
The same compounds (LiF / CaO / MgO) on the older `pob-TZVP`
basis (PT2013, LD floor 0.10) converged HF in the original paper
without these blow-ups, see PT2013 SI Table 2, which lists
LiF/CaO/MgO HF totals to 6 decimals. Tightening the rev2 floor
from 0.10 to 0.15 made the basis better-conditioned for **DFT
exchange-correlation** (the rev2 paper's design objective) but
moved several compounds across the LD margin for **pure HF
exchange**.

The fix is not a tighter SCF schedule (damping / level-shift /
TOLDEE tightening would only paper over the LD collapse, see
CLAUDE.md § 7 "oscillation = bug, not a convergence-aid
problem"). The fix is the same fix the rev2 paper applied for
DFT: re-optimise the exponents against the actual functional
that will be used. Goal 8's `mpei-TZVP` is that re-optimisation
for HF.

## Practical knock-on for Goal 8 Stage 0

[GOAL8 §5 Stage 0](GOAL8_MPEI_TZVP.md#stage-0-pob-tzvp-hf-parity-reproduction)
specifies "pob-TZVP HF parity reproduction", explicitly on the
**unrevised** `pob-TZVP` basis (PT2013), not rev2. This scout
confirms that targeting was correct: rev2 isn't safe as the
Stage-0 reproduction baseline because the LiF/CaO/MgO entries
would fail SCF independently of pipeline bugs. Stage 0 stays on
unrevised pob-TZVP for the pipeline smoke test; the rev2-vs-mpei
delta is a Stage 3 / Stage 4 publication finding, not a Stage 0
gate.

## Side notes

* **CRYSTAL output parser bug surfaced**, *fixed* (commit
  `b5d4f20`, milestone M2). The in-tree
  `vibe_basis.backends.crystal14.parse_output_file` had flagged
  cleanly-converged CRYSTAL23 outputs as `truncated=True`
  because the terminator changed from CRYSTAL14's bare line of
  capital E's to CRYSTAL23's `EEEEEEEEEE TERMINATION DATE …`
  form; the terminator regex was broadened.
* **`emit_input` deck title was stale**, *fixed* (milestone M3).
  It now emits "… (CRYSTAL parity)" rather than the
  engine-specific "(CRYSTAL14 parity)". M3 also added a
  `tolinteg` parameter so the emitter can set the tight
  `TOLINTEG 9 9 9 18 54` cutoffs this scout found decisive for
  HF SCF stability.
* **Scout artifacts** are at `.claude/scout/` (decks, wrappers,
  fetched workspaces, `analyze.py`, raw `results.txt` table).
  Gitignored, the report you are reading is the durable trace.

## Reproducing

```sh
# from a checkout on the basissetdev branch with vibe-basis installed:
./.venv/bin/python <<'EOF'
from pathlib import Path
from vibe_basis.backends.crystal14 import emit_input
from vibe_basis.io.structures import STRUCTURES
from vibe_basis.io.references import PT2013_T4_cubic_ionic_names

run_sh = (
    "#!/bin/bash\nset -euo pipefail\n"
    "crystal23demo < {name}.d12 > {name}.out 2>&1\n"
)
for name in PT2013_T4_cubic_ionic_names:
    d = Path(f".scout/{name}"); d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.d12").write_text(
        emit_input(STRUCTURES[name], basis='pob-tzvp-rev2', method='hf')
    )
    (d / "run.sh").write_text(run_sh.format(name=name))
    (d / "run.sh").chmod(0o755)
EOF

# then for each compound:
VQ=/path/to/vibe-queue/.venv/bin/vq
for d in .scout/*/; do
    $VQ submit compute-reference -d "$d" --cpus 4 --wall-time-seconds 900 \
         --job-name "$(basename $d)-hf-rev2" -- bash run.sh
done
```

Parse outputs with `vibe_basis.backends.crystal14.parse_output_file`
and classify on `parsed.converged` + the per-cycle `ETOT/DETOT`
trace (see `.claude/scout/analyze.py`).

-- basissetdev chat, 2026-05-18

---

## Follow-up (2026-05-19): three-basis × four-method scan on the failing compounds

After the 2026-05-18 scout flagged LiF / CaO / MgO as HF failures
on `pob-TZVP-rev2`, the user supplied three CRYSTAL `.d12` decks
(`MgO_seg_PW1PW.d12`, `CaO_seg_PW1PW.d12`, `LiF_seg_PW1PW.d12`)
with an inline segmented basis intended for these systems, and
asked for a four-method × three-basis comparison to isolate which
ingredient drives the HF instability, the basis itself, the
method, or the integral-truncation settings.

### Setup

* **Compounds:** LiF, CaO, MgO (same three as the original
  scout's HF failures).
* **Methods:** HF, PBE, PBE0, PW1PW (4).
* **Bases:** three variants (3):
  - **`seg`**: the inline segmented basis from the user's three
    .d12 files. More flexible than the pob-* keyword bases:
    Mg has 12 shells, Ca has 19 shells, O has 13 shells, F has
    14 shells, Li has 8 shells. Heavily diffuse on the metal.
  - **`pob-tzvp`**: CRYSTAL `BASISSET POB-TZVP` keyword.
  - **`pob-tzvp-rev2`**: CRYSTAL `BASISSET POB-TZVP-REV2`
    keyword (CRYSTAL23 only).
* **Identical everywhere else:** lattice constants from the
  user's .d12 files (LiF 3.972, CaO 4.796, MgO 4.189 Å, slightly
  contracted from experimental), `TOLINTEG 9 9 9 18 54`,
  `SHRINK 8 8`, default `TOLDEE` (= 6 in CRYSTAL), SCF-only
  (the `OPTGEOM/ENDOPT/ENDGEOM` block in the user's inputs was
  stripped, geometry optimisation would confound the SCF
  comparison).
* **Engine:** `crystal23demo` on compute-reference + compute-small (4 cpus, 20 min
  wall budget); 36 jobs via `vq`.

### Result: convergence grid

| compound / method | seg                 | pob-tzvp        | pob-tzvp-rev2   |
|---|---|---|---|
| **LiF**  / HF    | ✓ 6 cyc             | ✓ 6 cyc         | ✓ 6 cyc         |
| LiF  / PBE       | ✓ 6 cyc             | ✓ 6 cyc         | ✓ 6 cyc         |
| LiF  / PBE0      | ✓ 6 cyc             | ✓ 6 cyc         | ✓ 6 cyc         |
| LiF  / PW1PW     | ✓ 6 cyc             | ✓ 6 cyc         | ✓ 6 cyc         |
| **CaO**  / HF    | ✗ **SCF crash** + SIGTERM at cyc ≥9 | ✓ 7 cyc | ✓ 7 cyc |
| CaO  / PBE       | ✓ 8 cyc             | ✓ 8 cyc         | ✓ 8 cyc         |
| CaO  / PBE0      | ✓ 8 cyc             | ✓ 7 cyc         | ✓ 7 cyc         |
| CaO  / PW1PW     | ✓ 7 cyc             | ✓ 7 cyc         | ✓ 7 cyc         |
| **MgO**  / HF    | ✓ 8 cyc             | ✓ 7 cyc         | ✓ 7 cyc         |
| MgO  / PBE       | ✓ 8 cyc             | ✓ 8 cyc         | ✓ 7 cyc         |
| MgO  / PBE0      | ✓ 7 cyc             | ✓ 7 cyc         | ✓ 7 cyc         |
| MgO  / PW1PW     | ✓ 7 cyc             | ✓ 7 cyc         | ✓ 7 cyc         |

**35 of 36 jobs converged. The single failure is CaO HF on the
user's segmented basis.**

### Two findings that flip the original story

**Finding 1: `TOLINTEG` sensitivity rescues HF on the keyword
bases.** The 2026-05-18 scout used CRYSTAL's default
`TOLINTEG` (loose: 6 6 6 12 24-class cutoffs) with `TOLDEE 8`
and saw LiF, CaO, MgO **all** blow up under HF on rev2. The
2026-05-19 follow-up uses the user-recommended
`TOLINTEG 9 9 9 18 54` and the **same three compounds with the
same rev2 keyword converge cleanly in 7 cycles**. The HF
instability on rev2 is therefore **not** intrinsic to the
basis, it's a sensitivity to integral-cutoff settings that the
defaults don't satisfy.

| Compound | rev2 HF, default TOLINTEG (2026-05-18) | rev2 HF, TOLINTEG 9 9 9 18 54 (2026-05-19) |
|---|---|---|
| LiF | blow-up at cyc 41 (`+723 Ha` excursion) | converged at cyc 6, E=−107.085 Ha |
| CaO | blow-up at cyc 2 (-712 Ha into "POSSIBLY CONDUCTING") | converged at cyc 7, E=−751.798 Ha |
| MgO | blow-up at cyc 50 (after 49 cycles within 3 mHa of PT2013) | converged at cyc 7, E=−274.687 Ha |

This is a useful methodological fact for Goal 8: any
`mpei-TZVP` optimisation loop that uses HF on rev2-class
exponents **must** ship with tight TOLINTEG, or it will spuriously
fail evaluations that have nothing to do with the candidate
exponents being optimised.

**Finding 2: the user's segmented basis fails HF on CaO even
with tight TOLINTEG.** The lone failure in the 36-job grid is
CaO HF on `seg`:

```
 CYC   0 ETOT(AU) -7.520661260001E+02 DETOT -7.52E+02 …  ← physical
 CYC   1 ETOT(AU) -7.517545968528E+02 DETOT  3.12E-01 …
 CYC   2 ETOT(AU) -7.518043689240E+02 DETOT -4.98E-02 …
 CYC   3 ETOT(AU) -7.518119307761E+02 DETOT -7.56E-03 …
 CYC   4 ETOT(AU) -6.479390363955E+02 DETOT  1.04E+02 …  ← +104 Ha jump
 CYC   5 ETOT(AU) -7.850864019462E+02 DETOT -1.37E+02 …
 CYC   6 ETOT(AU) -7.912419254784E+02 DETOT -6.16E+00 …
 CYC   7 ETOT(AU) -7.718745827729E+02 DETOT  1.94E+01 …
 CYC   8 ETOT(AU) -6.611707575454E+02 DETOT  1.11E+02 …
 [job runs out the 20-min wall budget; crystal23demo segfaults
  during cleanup, last lines are stack-trace frames]
```

Same fingerprint as the original 2026-05-18 failures: cycle 0-3
near the right answer, then a multi-hundred-Ha jump that DIIS
can't recover from. The seg basis is heavily diffuse on Ca
(19 shells incl. 7 separate s-shells most with 1 contraction),
and the contracted small-exponent functions on Ca + the
diffuse-most O functions evidently produce a near-singular
overlap matrix at the HF level that DIIS occasionally projects
onto. PBE / PBE0 / PW1PW on the same seg basis for the same CaO
all converge in 7-8 cycles, same screening argument as before:
DFT exchange-correlation tames the LD margin where pure HF
doesn't.

This is a **basis-design** signal rather than a
TOLINTEG-tuning signal, no integral-cutoff tightening on CaO
HF + seg is going to fix a basis whose Ca shells are
overcomplete for HF.

### Variational depth comparison

For the 11 cases where all three bases converge a given
(compound, method), the seg basis is consistently more bound
than `pob-tzvp`, which is in turn more bound than
`pob-tzvp-rev2`. Per-compound differences for the methods
that succeed across all bases:

| (compound, method) | E(seg) − E(pob-tzvp) (mHa) | E(seg) − E(pob-tzvp-rev2) (mHa) |
|---|---|---|
| LiF, HF        |  −7.4 | −7.8 |
| LiF, PBE       | −12.2 | −13.7 |
| LiF, PBE0      | −10.6 | −11.6 |
| LiF, PW1PW     | −10.9 | −12.2 |
| CaO, PBE       | −10.3 | −37.7 |
| CaO, PBE0      |  −3.4 | −29.8 |
| CaO, PW1PW     |  −5.4 | −31.9 |
| MgO, HF        |  −7.9 |  −9.4 |
| MgO, PBE       | −15.0 | −16.9 |
| MgO, PBE0      | −12.3 | −14.1 |
| MgO, PW1PW     | −13.4 | −15.2 |

The user's seg basis is 4-38 mHa lower in energy than the pob
keyword bases, consistent with a more flexible (more shells,
more diffuse exponents) basis, but at the cost of a
near-singular overlap matrix that **specifically the HF
ansatz** can't tolerate on CaO.

### Revised implication for Goal 8

The 2026-05-18 conclusion ("rev2 is biased toward DFT") needs
qualification:

* **The rev2 *exponents/contractions* are not the problem for
  HF stability on simple ionics.** With tight TOLINTEG, rev2 HF
  converges in 7 cycles on LiF/CaO/MgO and gives sensible
  energies, even better-conditioned than the user's
  more-diffuse seg basis.
* **What rev2 *is* biased toward** is hidden in the small
  energy deltas vs PT2013 SI Table 2 (unrevised pob-TZVP HF):
  rev2 HF lands ~5-50 mHa above PT2013 for the test set, while
  rev2 PBE/PW1PW land close to where the rev2 paper reports
  them. The DFT-vs-HF energy difference at rev2 exponents is
  not the same as the DFT-vs-HF energy difference at PT2013 or
  at variationally-HF-optimal exponents.
* **The actual Goal 8 deliverable is the **delta** between
  rev2 exponents and HF-optimal exponents** at the BOBYQA /
  MIGRAD level, not a "rev2 is broken for HF" claim. The
  scout has reframed this from a stability concern to a
  variational-depth concern.
* **`mpei-TZVP` should be re-optimised against the same TOLINTEG
  the test inputs use** (9 9 9 18 54, the user's standard) to
  rule out cutoff sensitivity as a source of objective-function
  noise.

### Operational notes

* **CRYSTAL .d12 syntax for inline basis** needs an explicit
  `END` after both the geometry section (before the
  `<Z> <NSHELL>` header) and the basis section (after `99 0`).
  The `BASISSET` keyword self-closes both sections, so keyword
  bases don't need them. The build script
  (`.claude/scout/build_36.py`) emits the `END` lines only for
  the inline-basis variant.
* **`vq submit` consumes stdin** by default, which conflicts
  with a `while read … done` loop driving submits. Use
  `vq submit … < /dev/null` to guard the stdin pipe; otherwise
  the loop processes only the first line.
* **compute-small's `vq programs` reports `crystal23demo` as OK** because
  it checks the absolute path `/home/USER/bin/crystal23demo`,
  but the daemon's `$PATH` doesn't include `/home/USER/bin`.
  Job wrappers that invoke a bare `crystal23demo` succeed on
  compute-reference but fail instantly with exit-127 on compute-small. Workaround:
  use the absolute `/home/USER/bin/crystal23demo` in run.sh.

-- basissetdev chat, 2026-05-19
