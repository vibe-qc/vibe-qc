# `examples/basisset_dev/` — pob-paper test set as runnable inputs

Self-contained vibe-qc input files for every compound in the
calibration set behind pob-TZVP / pob-TZVP-rev2 / pob-DZVP-rev2.
Every file in `inputs/` bakes in the lattice vectors and fractional
coordinates from the published experimental reference; nothing is
fetched at runtime.

The full target set is ≈75 compounds across PT2013 + VO2019 — see
[`docs/basisset_dev/REQUIREMENTS-PERIODIC.md`](../../docs/basisset_dev/REQUIREMENTS-PERIODIC.md).
**Phases 1 + 2 + 3 ship 39 cubic compounds**: the 13 cubic
ionic compounds (PT2013 T4), 24 cubic semiconductors / carbides
/ nitrides / antifluorites / AFM rocksalt TM oxides (PT2013 T8
+ T10), and the 2 non-rocksalt cubic oxide / nitride structures
(Cu₂O cuprite + Cu₃N anti-ReO₃, PT2013 T10/T8). 6 of those are
antiferromagnetic; R3's blocker (a broken-symmetry initial guess)
shipped 2026-06-14/16 as ATOMSPIN, so those inputs' Γ-only RHF
placeholder runs are ready to be updated to a physical AFM UKS run
(see each docstring) — the remaining gate for a production-size
result is R2 (multi-k), not R3.

## Why this is here

The pob papers don't ship reproducible inputs — they ship lattice
constants in a paper table and assume the reader has CRYSTAL +
ICSD + a structure-database lookup. Re-running the calibration set
twelve years later required reconstructing each crystal structure
from the cited primary references. This directory makes that
work-once; every future run just opens the `.py` and reads.

## Layout

```
examples/basisset_dev/
├── README.md          ← this file
├── check_basisopt_install.py  ← preflight: can this env run the
│                                external-program path? (run first)
├── _structures.py     ← single source of truth: 39 entries with both
│                        full-cell + CRYSTAL14 asymm-unit views
├── _generator.py      ← emits two siblings per compound: .py + .d12
└── inputs/
    ├── lih_pob-tzvp_rhf.py     ← vibe-qc input
    ├── lih_pob-tzvp_rhf.d12    ← CRYSTAL14 parity sidecar (same compound)
    ├── nacl_pob-tzvp_rhf.py
    ├── nacl_pob-tzvp_rhf.d12
    └── … (39 .py + 33 .d12 in Phases 1+2+3)
```

Run `check_basisopt_install.py` first. `_generator.py` and the CRYSTAL
parity path both need `vibe-basis` importable, which a plain
`pip install -e .` does **not** give you — install vibe-qc's `basisopt`
extra (`uv pip install -e '.[basisopt]'`, or under pip
`pip install -e . && pip install -e vibe-basis/`). The preflight reports
what this environment can actually run and exits nonzero when the
external-program path is unusable, so it works as a guard at the top of
a campaign script.

The two underscore-prefixed files are tooling, not inputs to run.
Everything in `inputs/` is callable as a top-level Python script
(`.py`) or as a CRYSTAL14 input deck (`.d12`). The 6 AFM
compounds (MnO, FeO, CoO, NiO, α-MnS, α-MnSe) skip the `.d12`
emission until R3-aware ATOMSPIN handling lands.

## How to run on compute-reference (preferred)

Install [vibe-queue](https://github.com/vibe-qc/vibe-queue) separately
and follow the [queue setup guide](../../docs/user_guide/queue.md).
The `vq` command comes from that repository; `vibe-basis/` remains in this
core checkout. Wrapper paths below refer to the queue checkout on the
execution host. Each compound has **two** runnable inputs:

* `<compound>_pob-tzvp_rhf.py` — vibe-qc Γ-only RHF input.
* `<compound>_pob-tzvp_rhf.d12` — CRYSTAL14 parity input (same
  basis, same compound, in CRYSTAL's space-group form).

### Single vibe-qc run

```sh
JOBID=$(vq submit examples/basisset_dev/inputs/lih_pob-tzvp_rhf.py)
while ! vq status "$JOBID" 2>/dev/null \
        | grep -qE '^state:.*(completed|failed|killed|interrupted)'; do
    sleep 30
done
vq fetch "$JOBID" -o ./out
cat ./out/$JOBID/stdout.log
```

### CRYSTAL14 parity run (the in-house oracle)

```sh
mkdir mgo_crystal
cp examples/basisset_dev/inputs/mgo_pob-tzvp_rhf.d12 mgo_crystal/
JOBID=$(vq submit -d ./mgo_crystal --cpus 14 --wall-time-seconds 7200 -- \
    bash /home/USER/gitlab/vibe-queue/contrib/run-crystal.sh \
    mgo_pob-tzvp_rhf.d12)
# poll … fetch …
grep "TOTAL ENERGY" ./out/$JOBID/mgo_pob-tzvp_rhf.out
```

The `--wall-time-seconds` flag is required to work around the
v0.5.9 daemon watchdog regression that mis-kills bash-wrapped
jobs as `STARVED`.

### Why both

The 2013 paper printed total energies in SI Table 2 (HF) and
Table 1 (PW1PW). Diffing vibe-qc against those numbers tests
"do we match the published reference." Diffing vibe-qc against
**our own CRYSTAL14 run on the same input** tests "do we match
CRYSTAL on this exact geometry / basis / k-mesh / ITOL setup"
— this catches silent drift in vibe-qc's `PeriodicSystem`
construction or basis-file rounding **before** it shows up as
a Table-2 mismatch.

Per CLAUDE.md § 10, CRYSTAL is an **external code** — we
generate the `.d12`, `vq submit` ships it, CRYSTAL runs out
of process. No imports inside vibe-qc.

For the whole Phase-1 batch:

```sh
for f in examples/basisset_dev/inputs/*_rhf.py; do
    JOBID=$(vq submit "$f")
    echo "submitted $f → $JOBID"
done
```

The daemon currently runs `--max-jobs 1`, so the 13 jobs queue
serially. Each is small (≤12-atom unit cell, ≤44 basis functions).
Wall time per job: a few minutes to a few tens of minutes
depending on R1–R5 progress.

## How to run on the laptop (only for one-off smoke tests)

```sh
.venv/bin/python examples/basisset_dev/inputs/lih_pob-tzvp_rhf.py
```

For a tight set (LiH, MgO, NaCl) running locally is fine. **Do not
batch-run all 13 on the laptop** — the cumulative load is what
crashed the host on 2026-05-09. Either go through `vq` or run one
input at a time and watch RAM.

## Phase 1 inventory — cubic ionics (PT2013 T4)

| Compound | Type | Lattice (Å) | Cell atoms | Source | PT2013 SI Table 2 (HF) |
|----------|------|-------------|------------|--------|------------------------|
| LiCl | rocksalt | 5.130 | 8 | Aguayo 2004 | −467.087 468 |
| NaCl | rocksalt | 5.640 | 8 | Walker 1990 | −621.495 944 |
| LiF  | rocksalt | 4.027 | 8 | Dupre 1992  | −107.055 717 |
| NaF  | rocksalt | 4.632 | 8 | Rao 1990    | (SCF n/c)    |
| KF   | rocksalt | 5.347 | 8 | Chichagov 1994 | −698.704 515 |
| KBr  | rocksalt | 6.570 | 8 | VO2019 Ref. 24 | (no HF table) |
| MgO  | rocksalt | 4.217 | 8 | Walker 1990 | −274.681 754 |
| CaO  | rocksalt | 4.811 | 8 | Gajbhiye 2006 | −751.806 904 |
| LiH  | rocksalt | 4.083 | 8 | Hasegawa 2005 | −8.062 837 |
| NaH  | rocksalt | 4.890 | 8 | Sweeney 1993 | −162.453 809 |
| KH   | rocksalt | 5.704 | 8 | Guerin 1989 | −599.723 256 |
| CaF₂ | fluorite | 5.463 | 12 | Yang 1987 | −875.945 290 |
| K₂O  | antifluorite | 6.436 | 12 | Straumanis 1939 | −1273.185 420 |

## Phase 2 inventory — cubic semiconductors + TM compounds (PT2013 T8 + T10)

Diamond / zincblende / rocksalt-MX / antifluorite. The 6 AFM
compounds at the bottom emit a ``BLOCKED ON: REQUIREMENTS-PERIODIC R3``
stub in their docstring; basis loads fine but the spin-restricted
Γ-only RHF run converges to a non-physical solution. Wait for the
broken-symmetry guess (R3) to land.

| Compound | Type | Lattice (Å) | Cell atoms | Status |
|----------|------|-------------|------------|--------|
| C (diamond) | diamond | 3.567 | 8 | runnable |
| Si | diamond | 5.431 | 8 | runnable |
| Ge | diamond | 5.621 | 8 | runnable |
| AlP | zincblende | 5.421 | 8 | runnable |
| AlN | zincblende | 4.365 | 8 | runnable |
| GaAs | zincblende | 5.653 | 8 | runnable |
| GaP | zincblende | 5.448 | 8 | runnable |
| β-ZnS | zincblende | 5.400 | 8 | runnable |
| ZnSe | zincblende | 5.674 | 8 | runnable |
| β-BN | zincblende | 3.625 | 8 | runnable |
| β-SiC | zincblende | 4.358 | 8 | runnable |
| TiC | rocksalt | 4.328 | 8 | runnable |
| VC | rocksalt | 4.163 | 8 | runnable |
| TiN | rocksalt | 4.235 | 8 | runnable |
| VN | rocksalt | 4.137 | 8 | runnable |
| CrN | rocksalt | 4.135 | 8 | runnable (paramagnetic at room T) |
| Na₂Se | antifluorite | 6.825 | 12 | runnable |
| K₂S | antifluorite | 7.407 | 12 | runnable |
| MnO | rocksalt | 4.445 | 8 | **BLOCKED on R3** (AFM-II) |
| FeO (wüstite) | rocksalt | 4.326 | 8 | **BLOCKED on R3** (AFM-II) |
| CoO | rocksalt | 4.250 | 8 | **BLOCKED on R3** (AFM-II) |
| NiO | rocksalt | 4.195 | 8 | **BLOCKED on R3** (AFM-II) |
| α-MnS (alabandite) | rocksalt | 5.220 | 8 | **BLOCKED on R3** (AFM-II) |
| α-MnSe | rocksalt | 5.460 | 8 | **BLOCKED on R3** (AFM-II) |

## Phase 3 inventory — non-rocksalt cubic oxide / nitride

Two cubic compounds where the parent structure is NOT rocksalt /
zincblende / diamond: cuprite (Pn-3m, sg 224) and anti-ReO₃
(Pm-3m, sg 221). Both ship with the new asymmetric-unit helpers
(`_asymm_cuprite`, `_asymm_anti_reo3`) so the CRYSTAL14 .d12
sidecar emits identically to the higher-symmetry siblings.

| Compound | Type | Space group | Lattice (Å) | Cell atoms | Status |
|----------|------|-------------|-------------|------------|--------|
| Cu₂O | cuprite | Pn-3m (224) | 4.269 | 6 | runnable |
| Cu₃N | anti-ReO₃ | Pm-3m (221) | 3.817 | 4 | runnable |

## Acceptance criterion

When [`REQUIREMENTS-PERIODIC.md`](../../docs/basisset_dev/REQUIREMENTS-PERIODIC.md)
R2 (multi-k SCF at production sizes) lands, the SCF energy with
the **pob-TZVP** basis at converged Pack-Monkhorst grid should
agree with the PT2013 SI Table 2 reference to **≤ 1 mHa per unit
cell**. Until then, the Γ-only run stands as a "does the basis +
geometry parse and converge" smoke test, with the energy
differing by a basis-truncation error from the published value.

## What's NOT here yet (and why)

* **Hexagonal / tetragonal compounds** (PT2013 T5/T9/T11; VO2019
  T3/T7/T10) — blocked on R1 (non-orthorhombic Ewald).
* **Cubic semiconductors / TM oxides** (PT2013 T8/T10) — runnable
  with current vibe-qc; will land in Phase 2 of this generator.
* **Antiferromagnetic transition-metal oxides** (MnO, NiO, FeO,
  CoO, Cr₂O₃) — blocked on R3 (broken-symmetry initial guess).
* **PW1PW reference comparison** (PT2013 SI Table 1) — blocked
  on R4 (PW1PW alias).
* **Lattice optimisation** — blocked on R5; for now the inputs
  fix the lattice at the experimental value.
* **X23 molecular crystals** (VO2019 Table 14) — blocked on R11
  (D3 dispersion in periodic).

The structure database (`_structures.py`) will grow to cover all
≈75 compounds in lockstep with R1-R5 landing. Re-run
`_generator.py` after each database update.

## Regenerating

Edit `_structures.py` (add entries, fix typos, update lattice
constants if a better experimental reference appears), then:

```sh
.venv/bin/python examples/basisset_dev/_generator.py
```

The generator is idempotent — running it twice produces the same
files (modulo trailing newlines). Commit both the database edit
and the regenerated `inputs/` in the same change so they don't
drift.
