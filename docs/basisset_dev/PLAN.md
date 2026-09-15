# `basissetdev`, long-running plan

This branch hosts the basis-set development work driven by the
**basis-set main development chat**. It is **never merged into `main`
until the deliverables ship as papers**. As features land in `main`
(multi-k periodic SCF, hexagonal Ewald, hybrid functionals in periodic,
…), this branch is rebased onto the new `main`.

The work spans six goals, three of which are publication targets in
their own right.

## Author / motivation

The pob-* basis-set family was developed by M. F. Peintinger, D.
Vilela Oliveira, J. Laun, and T. Bredow at the Mulliken Center for
Theoretical Chemistry, Bonn. The current vibe-qc lead author (M. F.
Peintinger) is one of the original pob authors, and the optimisation
machinery used in the 2013 and 2019 papers, *"a python script which
runs single point calculations with CRYSTAL09/17 and uses the numerical
minimisation library MINUIT2 from the ROOT project"*, is the same
architecture this branch will reproduce inside vibe-qc.

## Goals

### Goal 1, verify the shipped pob basis sets ✅ (in progress)

Three pob bases ship in vibe-qc: `pob-TZVP`, `pob-TZVP-rev2`,
`pob-DZVP-rev2`. Verification audits the per-element CRYSTAL files
distributed by the Bredow group against the bundled `.g94` files,
plus spot-checks against the SI of the 2013 paper.

**Status:** [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md). Real
bug found and fixed in shipped pob-TZVP and pob-TZVP-rev2 (sulfur
d-polarisation column-swap, `exp=0` instead of `exp=0.5207`). All
three bases now byte-clean against upstream sources at 1e-15 tolerance.

### Goal 2, periodic-method requirements for the periodic chat

[REQUIREMENTS-PERIODIC.md](REQUIREMENTS-PERIODIC.md): the list of
periodic-SCF features needed before the full pob test set is runnable
in vibe-qc. Hand-off artefact for the periodic-methods main dev chat.

Major missing capabilities (memory + main-paper reading):

* hexagonal lattice geometry-opt (BeF₂, ScCl₃, MgBr₂, BeO, α-SiO₂,
  B₂O₃, Al₂O₃, NaNO₃, MgCO₃, FePO₄, NiAs, α-SiC, α-BN, B₄C, ScB₂,
  CoS, CuS, β-ZnS, GaF₃, GeO₂, V₂O₃, Cr₂O₃, ZnO)
* tetragonal (TiO₂ rutile/anatase)
* multi-k MP-grid SCF (CRYSTAL default ≈8×8×8 for the pob test set)
* PW1PW hybrid functional (CRYSTAL's 1-parameter PWGGA hybrid; not
  PBE0)
* WC GGA (used for metals)
* B3LYP / PBE0 / HSE in periodic (cross-method validation)
* stable open-shell UKS / UHF (MnO, FeO, CoO, NiO, Cr₂O₃, V₂O₃,
  CrCl₂, CrN, MnSe, MnS)
* simultaneous lattice + atomic-position optimisation
* D3 (Grimme) dispersion correction (X23 molecular crystals in 2019
  paper)
* counterpoise / BSSE for periodic crystals (CRYSTAL ATOMBSSE
  equivalent, required for re-running the rev2 BSSE optimisation)
* TOLINTEG / ITOL5 truncation-threshold control

### Goal 3, self-contained test-set inputs

`examples/basisset_dev/<compound>/{rhf,pw1pw,b3lyp}_<basis>.py` for
every compound in the 2012 + 2019 paper tables (≈70 compounds × ≈3
functionals = ≈200 inputs). Each input bakes lattice vectors and
fractional coordinates from the paper-cited references in directly,
no runtime CIF / ASE / Materials Project dependency.

Inputs whose required features are not yet in vibe-qc emit with a
`# BLOCKED ON: <feature>` stub at the top, so the gating against
`REQUIREMENTS-PERIODIC.md` is explicit.

### Goal 4, basis-set optimisation engine

New module `python/vibeqc/basis_optimization/`. Reproduces the 2012/2019
recipe: vibe-qc replaces CRYSTAL09/17 as the SCF engine, `iminuit`
(Python wrapper around the same Minuit2 from ROOT) replaces the
direct-MINUIT2 call.

Architecture:

```
basis_optimization/
├── parametrise.py      # ElementBasis ↔ flat parameter vector
│                       # (log exponents, coefficients, even-tempered
│                       #  constraints)
├── objective.py        # Σ_systems w_i · E_HF/PW1PW(basis)
│                       # + λ · LD_penalty(min_eig(S))
│                       # + μ · BSSE_penalty(counterpoise hydride dimers)
├── ld_penalty.py       # Smooth penalty on λ_min(S) below cutoff
├── drivers/
│   ├── scipy_driver.py # L-BFGS-B on log-exponents, prototyping
│   └── minuit_driver.py# iminuit MIGRAD + HESSE, production
├── recipes/
│   ├── pob_truncate.py # "drop primitives with α < threshold" recipe
│   │                   # (the original pob/rev2 starting point)
│   └── eichkorn.py     # for aux-basis design (Goal 5)
├── transports/
│   ├── local.py        # in-process SCF for stage-1 smoke tests
│   └── compute-reference.py      # ssh + vibe-queue fan-out for production
└── tests/
```

Stages (each is a publishable validation milestone):

1. Single-atom HF energy minimisation (smoke test against atomic HF
   references; runs on the laptop).
2. One-compound bulk optimisation: drop H's diffuse-most s in
   pob-TZVP, refit valence on LiH; verify reproduces the 2013
   recipe per-element. *(Requires Goal 2 features for periodic SCF.)*
3. Multi-compound simultaneous optimisation, the actual pob /
   pob-rev2 calibration (compute-reference, vibe-queue).
4. LD-aware multi-compound optimisation (smallest-overlap-eigenvalue
   penalty, the rev2 stability extension).

### Goal 5, bespoke pob aux basis (paper-worthy, scope decision pending)

The DF dev chat owns the AutoAux (Stoychev, JCTC 13, 554, 2017)
implementation. Once AutoAux lands, pob-* users get a working aux
basis automatically.

The open question for `basissetdev` is whether to *additionally*
design hand-tuned `pob-TZVP-jk`, `pob-TZVP-rev2-jk`, `pob-DZVP-rev2-jk`
sets (Eichkorn / Weigend recipe targeting the pob orbital bases
specifically) as a paper. See
[memory: pob-TZVP aux basis](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/basis_library/README.md).

### Goal 7, extend the bundled basis library (review of 2026-05-08)

The user delivered a literature review (
[REVIEW_BASIS_SETS_2026-05-08.md](REVIEW_BASIS_SETS_2026-05-08.md))
on 2026-05-08 with the directive *"make sure we ship all of these
basis sets"*. The full gap analysis + per-basis status / priority
table lives in
[ROADMAP_BASIS_LIBRARY.md](ROADMAP_BASIS_LIBRARY.md).

**Tooling shipped on basissetdev:**

* `scripts/basisset_dev/fetch_from_bse.py`, pulls missing bases
  from the Basis Set Exchange (`pip install basis-set-exchange`),
  writes to `python/vibeqc/basis_library/custom/`, runnable
  with `--priority {highest,high,medium,low,all}`.
* First fetch (priority=all, 87 bases) added pcseg-{0..4},
  aug-pcseg-{0..4}, pc-{0..4}, aug-pc-{0..4}, 6-31++G**, 6-31+G**,
  6-311+G**, 6-311+G(2d,p), 6-311++G**, def2-mTZVP, def2-mTZVPP,
  dhf-{SVP,SV(P),TZVP,TZVPP,QZVP,QZVPP}, x2c-TZVPall(-s/-2c),
  Grimme vDZP, LANL2DZ family, cc-pV(n+d)Z + aug- variants,
  jul-/jun-cc-pV(T+d)Z, cc-pCV{D,T,Q}Z + aug- variants,
  ANO-RCC-V{DZ,DZP,TZ,TZP,QZP}, ANO-R0..R3, Sadlej pVTZ, Sadlej+,
  pcS-{0..3}, aug-pcS-{1,2}, pcSseg-{0..2}, pcJ-{0..3},
  Sapporo-DKH3-{DZP,TZP,QZP}, Cologne DKH2, SARC-DKH2,
  SARC2-QZ{V,VP}-DKH2.

**Cross-cutting blockers** (captured in the roadmap status column):

* libecpint integration → blocks LANL2DZ family, vDZP, dhf-*,
  fifth-period pob, x2c-* (data only, files ship; library
  cannot use them yet).
* Relativistic Hamiltonian (X2C / DKH2 / ZORA) → blocks
  meaningful use of the dhf, x2c, Sapporo-DKH3, Cologne-DKH2,
  SARC families.
* NMR shielding / J-coupling kernels → blocks meaningful use
  of pcS, pcSseg, pcJ.
* Modern composite-method support (gCP + D3/D4) → required
  for def2-mTZVP, def2-mTZVPP, vDZP to act as 3c carriers.
* CP2K-style GPW / mixed Gaussian-PW basis → blocks MOLOPT.
* PAW / ONCV pseudopotential infrastructure → blocks
  PseudoDojo, SSSP, VASP-PAW (review's plane-wave column, 
  out-of-scope for vibe-qc today).

**Action items still open (after the first fetch):**

1. Update `python/vibeqc/basis_library/README.md` with the
   live dashboard listing every shipped basis + status flag.
2. Add `tests/basisset_dev/test_basis_library_load.py` that
   verifies every newly-fetched `.g94` parses under libint
   (needs working `pip install -e .` in the worktree venv).
3. File the libecpint and relativistic-Hamiltonian feature
   requests with the molecular dev chat.

### Goal 6, periodic metal basis (paper-worthy)

Long functions / low exponents collide with periodic linear
dependence. Plan:

* Corpus: Li, Be, Na, Mg, Al, Cu, Ag, Au, Ti, Fe, Ni, each with
  1-2 phases (bcc/fcc/hcp).
* Source basis: def2-TZVP for 3d/4d, possibly Sapporo-DKH3 or
  all-electron Stuttgart for 5d.
* Recipe: pob-style diffuse drop + Goal 4's LD-aware multi-compound
  optimisation.
* Cross-validation against CRYSTAL with the same k-mesh once multi-k
  lands; against PySCF.pbc.GDF in the meantime.
* Output: `pob-metal-tzvp` family + paper.

### Goal 8, **mpei-TZVP**: HF-optimized basis (paper-worthy, active)

The pob-TZVP / pob-TZVP-rev2 / pob-DZVP-rev2 lineage was developed
for **DFT** (PW1PW workhorse). The original optimization scripts
are lost, Goal 1 was the recreate-from-SI side. Goal 8 takes the
next step: a **Hartree-Fock-optimized** sibling, `mpei-TZVP`, on
the same test compounds with the same Powell-via-MINUIT2-style
recipe.

* Seed basis: **pob-TZVP-rev2** (user-confirmed 2026-05-13;
  faster PoC than def2-TZVP; the mpei-TZVP − rev2 exponent delta
  is itself the paper finding).
* SCF engine: **CRYSTAL14 on compute-reference via `vq`** (user-confirmed;
  matches PT2013 methodology continuity; the Phase 14h `.d12`
  generator stages the test set end-to-end).
* Reference: PT2013 SI Table 2 (HF totals) for the 13 cubic
  ionics; VO2019 equivalents for the wider test set.
* Optimizer: NLopt BOBYQA / COBYLA primary (derivative-free,
  Powell-family lineage = what PT2013 actually used; minimizes
  wall-time when each eval is minutes of CRYSTAL14 + queue
  latency), with iminuit MIGRAD + HESSE at the converged point
  for publication-grade parameter uncertainties.
* LD bound: hard floor on exponents at 0.15 (pob-rev2's rule,
  approximate transfer from DFT to HF; revisit if MIGRAD
  bunches at the bound).
* Stages 0-4 with explicit acceptance gates, see
  [GOAL8_MPEI_TZVP.md](GOAL8_MPEI_TZVP.md).
* University outreach gated on a successful Stage 2 (LiH
  one-compound HF MIGRAD that matches the PT2013 reference to
  ≤ 1 mHa per unit cell after MIGRAD converges).

Critically: **Goal 8 unblocks today.** It skips the entire
REQUIREMENTS-PERIODIC R4 (PW1PW alias) gate that holds Goal 4
back, because the HF objective doesn't need a functional. R3
(AFM broken-symmetry guess) still blocks 6 of the test
compounds; Stages 0-3 launch on the cubic-ionic + cubic-
semiconductor subset that doesn't need R3.

## Branch / worktree mechanics

* `basissetdev` is created from `main` and tracks the latest
  consolidated tip of the work.
* Each Claude chat operates in its own worktree on a `claude/...`
  branch. Work commits there and is fast-forwarded into `basissetdev`
  at the end of each milestone.
* When `main` advances (e.g. multi-k lands), rebase `basissetdev`
  onto `main` rather than merging.
* Memory of progress: this `PLAN.md` plus `VERIFICATION_REPORT.md`
  plus `REQUIREMENTS-PERIODIC.md`.

## Where heavy compute runs

* Single-atom + single-compound smoke tests: laptop (≤ a few
  minutes).
* Multi-compound optimisation, BSSE counterpoise loops, full test
  set runs: **compute-reference** (128 GB / 16 cores). The optimisation driver
  ships test-set evaluations through ssh + `vq` (vibe-queue) for
  per-compound parallelism. Network gating per the existing
  compute-reference memory note.
