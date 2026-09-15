# Roadmap — vibe-basis

`vibe-basis` is its own endeavour with its own version line
([`VERSIONING.md`](VERSIONING.md)), its own release cadence, and its own
success criterion. This file is its **engineering roadmap**.

It is deliberately *not* a section of
[`docs/roadmap.md`](../docs/roadmap.md): vibe-qc's roadmap tracks the
capabilities of a quantum-chemistry program, and this one tracks a driver that
designs basis sets. They move on different clocks. They are, however, **coupled
in one direction** — this roadmap consumes vibe-qc capabilities — and § 6 below
is the contract that keeps the two honest about it.

| | |
|---|---|
| **Current version** | 0.4.0 (M-0 landed; M0 API + vibe-qc engine landed, gate open) |
| **1.0.0 criterion** | first production basis published from this driver + frozen backend/transport API ([`VERSIONING.md`](VERSIONING.md)) |
| **Working tracker** | [`handovers/HANDOVER_BASISOPT.md`](../handovers/HANDOVER_BASISOPT.md) — current state, SHAs, blockers, asks |
| **vibe-qc's roadmap** | [`docs/roadmap.md`](../docs/roadmap.md) § "OPTBASIS-style basis optimiser" |

---

## 1. Mission

Design solid-state Gaussian basis sets by optimising against **cohesive
(atomization) energies**, cross-validated against an independent **plane-wave
limit** — and be decisively ahead of CRYSTAL23's `OPTBASIS`, with **vibe-qc
itself as the engine for every calculation.**

CRYSTAL23's `OPTBASIS` minimises Ω = E + γ·ln κ(S) for **one** crystal, over a
**fixed basis topology**, at a **fixed geometry**, driven by BDIIS (Daga,
Civalleri & Maschio, *JCTC* **16**, 2192 (2020)).

That characterisation is **verified against the CRYSTAL23 manual** (§ `OPTBASIS`,
p. 65; `~/gitlab/library/manuals/crystal23/crystal23.pdf`), not recalled:

* the objective is exactly Ω({α,d}) = E_tot({α,d}) + γ·ln κ({α,d}), with κ the
  overlap condition number, minimised by BDIIS;
* **`EXPONLY` is the default** — exponents only; coefficients require
  `COEFFONLY` or `ALLBDIIS`;
* which functions are optimised is chosen by **starring shells** in an
  explicitly written basis, so the topology is fixed *by construction*: there is
  no mechanism to add or remove a function;
* the worked example is a single crystal at a fixed geometry.

The manual's own assessment of the ceiling is worth quoting, because it is the
gap this roadmap aims at: *"The optimization of the entire basis set is a
formidable task and quite likely to be not very useful."*

**The optimiser axis is already won.** vibe-qc ships `optimize_bdiis` — the same
BDIIS, hardened past the CRYSTAL baseline (BFGS inverse-Hessian driving the DIIS
error vectors; trust-region + Armijo guard forbidding an objective increase;
quasi-Newton fallback on an ill-conditioned DIIS `B`; box-projected convergence)
— plus `condition_number_penalty` for OPTBASIS's own γ·ln κ term. So this
roadmap does not need to catch up on *how to minimise*. It changes **what is
minimised, over what, and against what truth.**

| # | Axis | CRYSTAL23 `OPTBASIS` | vibe-basis |
|:-:|------|----------------------|------------|
| 1 | **Objective** | total energy (+ κ penalty) | **cohesive energy vs a PW-limit reference** |
| 2 | **Basis topology** | fixed (starred shells only) | **searched** — d/f add-high / add-low / leave-one-out |
| 3 | **Systems** | one crystal | **joint multi-system**, weighted, held-out validation |
| 4 | **Geometry** | frozen | **re-relaxed per candidate basis** |
| 5 | **Thermochemistry** | none | **automated ZPE** |
| 6 | **Reference truth** | none | **PW-limit oracle**, validated across 28 systems |

---

## 2. Engine policy

**vibe-qc is the engine. External codes are fallback and cross-validation only.**

| Role | Engine | Status |
|------|--------|--------|
| **Primary — Gaussian** | **vibe-qc** (BIPOLE / GDF) | the target for every calculation |
| **Primary — plane wave** | **vibe-qc** (GPW / GAPW) | not there yet — dependency BOC-7 |
| Fallback — Gaussian | CRYSTAL23 | pob-lineage continuity; head-to-head baseline |
| Fallback — plane wave | GPAW | today's PW-limit oracle (28 systems, validated) |

One `EnergyEngine` protocol, four implementations. **No phase hard-codes an
engine, and no phase is *complete* until it runs on `VibeQcEngine`** — except
where an external result is deliberately the independent control (M5 rung 2).
Every reported number carries the engine that produced it.

---

## 3. Architecture

Two tiers, dependency strictly one-way — `vibe-qc → vibe-basis`, never the
reverse (guarded by `tests/test_no_vibeqc_dependency.py`):

| Tier | Home | Owns | Litmus test |
|:-:|------|------|-------------|
| **1** | `vibe-basis/` | optimizers · topology search · objective algebra · campaign engine · transports · reporting · external backends | *Would this work with an external code and no vibe-qc installed?* |
| **2** | `python/vibeqc/basis_optimization/` | analytic gradients · the vibe-qc engine · the cohesive pipeline on vibe-qc | *Does this need vibe-qc internals or the native build?* |

```
 L4  CAMPAIGN        [1]  budget · restart · report · provenance
 L3  TOPOLOGY SEARCH [1]  add-high · add-low · leave-one-out · recontract
 L2  CONTINUOUS OPT  [1]  BOBYQA / MIGRAD / BDIIS   (analytic gradients are [2])
 L1  OBJECTIVE       [1]  Σ w_i · loss( E_coh,i(basis) − E_coh,i(PW ref) )
 L0  PIPELINE        [2]  VibeQcEngine  ◄── PRIMARY
                     [1]  Crystal23Engine · GpawEngine  ◄── fallback + control
```

The L0 protocol landed in 0.3.0 as `vibe_basis.engine.EnergyEngine`. Engines
return an `EngineEnergy` carrying the engine name and version alongside the
number, because "MgO is −275.4776 Ha" is not a result — "CRYSTAL23 computed
−275.4776 Ha for MgO at RHF/pob-TZVP" is. A campaign that swaps engines as
easily as this one does must record which one produced each point, or it cannot
be audited afterwards.

Current engine status:

| Engine | Tier | State |
|--------|:----:|-------|
| `VibeQcEngine` | 2 | crystals, atoms, `relax`, `zero_point` (Γ phonons, M1) |
| `Crystal23Engine` | 1 | complete — emit, submit, parse, version-check |
| `GpawEngine` | 1 | declared, raises; wired to the validated PW worker at M5 |

An engine may additionally declare the optional capabilities `relax` and
`zero_point` (`EnergyEngine.supports()`). `VibeQcEngine` declares **`relax`**
(M0b, via the BIPOLE relaxer) and **`zero_point`** (M1, via
`vibeqc.basis_optimization.phonons`). `Crystal23Engine` declares neither, so a
pipeline on it records those stages as **skipped, with the engine named in the
reason** — a static-lattice cohesive energy that says it is one, rather than a
number that looks ZPE-corrected and is not.

Full rationale — why a separate package, why the arrow points this way, and why
the two `OptResult` classes are a deliberate duplication rather than a bug — is
in the working tracker, § 3.6.

---

## 4. Milestones

Each is independently landable and leaves `main` release-ready (CLAUDE.md § 14).
Gates are mechanical — a test, a number, or an artefact. Version column is the
vibe-basis bump the milestone earns.

| # | Milestone | Ver | Gate | Status |
|:-:|-----------|:---:|------|:------:|
| **M-0** | Boundary cleanup + `EnergyEngine` protocol + CRYSTAL14→23 retarget | 0.3.0 | one engine abstraction; import-direction guard green; CRYSTAL23 `.out` round-trips with a version probe | ✅ **done** |
| **M0** | Cohesive pipeline: relax → hybrid SP → ZPE → atoms → assemble | 0.4.0 | two-engine: CRYSTAL23 reproduces `E_pob` < 1 kJ/mol; then vibe-qc reproduces CRYSTAL23 < 2 kJ/mol | 🟡 **API (M0a) + `VibeQcEngine` (M0b) landed; gate open** |
| **M1** | Γ-point ZPE on the Gaussian route (FD Hessian **from energies** — see BOC-2) | 0.4.x | MgO + NaCl optical modes within 5 cm⁻¹ of the GAPW route; acoustic within 15 cm⁻¹ of zero | 🟡 **driver + engine landed; gate open** |
| **M2** | `CohesiveObjective` vs the PW limit | 0.5.0 | evaluating at pob-TZVP-REV2 reproduces MD −24.6 / MAD 35.4 kJ/mol (n=28) | ⬜ |
| **M3** | **Topology search — the d/f scan** | 0.6.0 | Mg-d in MgO + Si-d in SiC: terminates, ranked Pareto front, **rediscovers the shipped pob topology** on a total-energy objective | ⬜ |
| **M4** | Campaign orchestration on `vq` | 0.7.0 | 200-eval campaign survives a mid-run kill and resumes identically; zero writes into managed checkouts | ⬜ |
| **M5** | Cross-validation ladder (Gaussian ↔ PW limit) | 0.8.0 | ≥ 28 systems, rungs 1/2/2′/3 + extrapolation residual demonstrably smaller than the gap it measures | ⬜ |
| **M6** | Production run — the optimised basis | **1.0.0** | runs on `VibeQcEngine`; held-out MAD beats pob-TZVP-REV2's 35.4 kJ/mol by a stated margin at equal-or-lower primitive count | ⬜ |

Sequencing:

```
M-0 ─► M0 ──┬─► M1 ──┐
            │        ├─► M2 ─► M3 ─► M6
            └────────┘          │
                     M4 ────────┤
                     M5 ────────┘
```

M-0 first, and it is cheap. M1 may run in parallel with M2 (which stubs ZPE = 0
until M1 lands). M4 and M5 are independent of M3.

Per-milestone detail — deliverables, design notes, failure modes — lives in the
working tracker § 5, where it is kept current against `main`. This table is the
stable view; that one moves.

---

## 5. Dependencies on vibe-qc

None of these are vibe-basis's to fix. Each is owned by a vibe-qc dev chat and
tracked in that chat's handover.

| Id | Need | Blocks | Owner |
|----|------|--------|-------|
| BOC-1 | R13 Ewald-gauge dV(k)/dG(k) kernel → end-to-end periodic analytic basis gradient | *cost only* (~10× SCF without it) | periodic-SCF |
| BOC-2 | ~~BIPOLE analytic force accurate enough for FD phonons~~ **ANSWERED: it is a documented research preview, so M1 differences the *energy* instead. Not a bug; nothing owed by the periodic chat.** | — | — |
| BOC-3 | Ghost-atom BSSE for solids | counterpoise-corrected cohesive energies | periodic / basissetdev |
| BOC-4 | GPAW oracle breadth: more systems, PBE0/HSE06, Cs > 800 eV | M5 | vibeqc-gpaw |
| BOC-5 | PySCF reference backfill in `tests/test_periodic_ecp.py` | ECP-element cohesive energies | basissetdev |
| BOC-6 | Hybrid periodic frequency cost | M1 at hybrid level | periodic / perf |
| BOC-7 | **vibe-qc GPW/GAPW producing a converged PW-limit atomization** | **retiring the GPAW fallback — the "all calculations in vibe-qc" goal** | gapw |
| BOC-8 | vibe-qc ↔ CRYSTAL23 agreement on cohesive energies | M0 gate 2 | periodic-SCF |

**BOC-1 governs schedule. BOC-7 closes the mission.** BOC-8 is the honest risk —
total-energy parity does not imply cohesive-energy parity, and M0 is the first
time anyone measures it.

---

## 6. Sync contract with vibe-qc's roadmap

The two roadmaps are linked, not merged. The contract:

1. **vibe-qc's roadmap points here.** [`docs/roadmap.md`](../docs/roadmap.md)
   § "OPTBASIS-style basis optimiser" carries the pointer and records what
   shipped on the vibe-qc side (BDIIS, the κ penalty, the analytic gradients).
   It does **not** restate these milestones.
2. **This roadmap points there, once, per dependency.** § 5 is the only place
   vibe-qc capabilities appear. A capability that is not in § 5 is not a
   dependency, and adding one is how a vibe-qc need becomes visible.
3. **When a BOC dependency lands in vibe-qc, § 5 updates the same session.**
   Same cadence as CLAUDE.md § 5's lightweight-ongoing rule: don't queue it.
   The signal is a `main` sync (CLAUDE.md § 14) that touches a BOC owner's area.
4. **A vibe-basis release does not imply a vibe-qc release, or vice versa**
   ([`VERSIONING.md`](VERSIONING.md)). Milestones here earn vibe-basis bumps;
   they do not appear in vibe-qc's milestone list.
5. **Divergence is resolved toward the tracker.** If this file and the working
   tracker disagree about a gate, the tracker is current and this file is stale
   — fix this file. If this file and `docs/roadmap.md` disagree about what
   shipped in vibe-qc, `docs/roadmap.md` is authoritative.

---

## 7. Open questions for the maintainer

Carried here because they change the shape of M2/M3/M6, not just their schedule.

1. **Objective target** — the PW limit (isolates basis incompleteness) or
   experiment-minus-ZPE (physical, but folds in the functional's own error)?
   *Recommendation: PW limit; experiment stays a reported diagnostic.*
2. **BSSE** — counterpoise for solids does not exist (BOC-3). Without it the
   optimiser can lower the objective by *increasing* BSSE, which is exactly what
   diffuse `add-low` candidates will do. *Recommendation: a diffuseness penalty
   as proxy now; open BOC-3 properly only if the paper needs it.*
3. **Γ-only ZPE** — sufficient, or does the paper need q-mesh phonon ZPE?
4. **Element order** — main-group d-polarisation (Mg, Al, Si, S) first, 3d TM
   later? *Recommendation: yes — TM brings magnetism and is where the PW oracle
   is weakest.*
5. **Branch** — driver + pipeline on `main`, production campaign + resulting
   basis on `basissetdev` (CLAUDE.md § 4)? *Recommendation: yes.*
6. **Dependency policy** — promote vibe-basis to a **core** vibe-qc dependency
   (collapsing the duplicated `OptResult`), or keep it an extra and keep the two
   pinned by contract test? Adding it to core costs one small pure-Python
   package (click + numpy; click is *not* currently a vibe-qc dep). CLAUDE.md § 9
   says don't add dependencies casually, so this is the maintainer's call.
   *Recommendation: keep the extra; the contract test is proportionate for one
   nine-field dataclass.*

---

## 8. Papers: use the library, or the wishlist

Maintainer instruction (2026-07-26): **if you need a paper, look in
`~/gitlab/library` first; if it is not there, append it to
`~/gitlab/library/wishlist.txt`.** Do not go hunting for PDFs elsewhere.

The library is the PDF counterpart of vibe-qc's citation database
(`python/vibeqc/output/citations/database.toml`), organised into topic folders
with `INDEX.md` listing every file and `catalog.tsv` for lookup by DOI, author,
title, or citation key. It also holds vendor manuals under `manuals/`, including
**CRYSTAL23** — which is how § 1's characterisation of `OPTBASIS` and the
r2SCAN availability in § 4's M0 gate were verified rather than assumed.

Every paper this roadmap cites is already present:

| Topic | Paper | Key |
|---|---|---|
| the incumbent optimiser | Daga, Civalleri & Maschio 2020 | `daga_bdiis_2020` |
| pob-TZVP | Peintinger, Vilela Oliveira & Bredow 2013 | `peintinger_pob_tzvp_2013` |
| pob-rev2 | Vilela Oliveira, Laun, Peintinger & Bredow 2019 | `vilela_oliveira_pob_rev2_2019` |
| the κ penalty | VandeVondele & Hutter 2007 | `vandevondele_basisopt_2007` |
| r2SCAN | Furness et al. 2020 | `furness_r2scan_2020` |
| SCAN | Sun, Ruzsinszky & Perdew 2015 | `sun_scan_2015` |
| PW1PW | Bredow & Gerson 2000 | `bredow_gerson_pw1pw_2000` |

Per CLAUDE.md § 8, anything citable this workstream *adds* needs its
`database.toml` entry and route in the same merge; the PDF should land in the
library alongside, or go on the wishlist if it cannot be obtained.

## 9. See also

* [`README.md`](README.md) — what vibe-basis is, install, why a separate package.
* [`VERSIONING.md`](VERSIONING.md) — the version line and the 1.0.0 criterion.
* [`handovers/HANDOVER_BASISOPT.md`](../handovers/HANDOVER_BASISOPT.md)
  — the working tracker: current state, SHAs, per-milestone detail, progress log.
* [`docs/roadmap.md`](../docs/roadmap.md) — vibe-qc's own roadmap (§ 6 above).
* [`docs/basisset_dev/`](../docs/basisset_dev/) — the basis-set development
  documentation set, incl. `GOAL8_MPEI_TZVP.md` (the 1.0.0 target basis) and
  `OPTIMISING_A_BASIS.md` (the in-process molecular optimiser user guide).
* [`handovers/HANDOVER_GPAW_PW_REFERENCE.md`](../handovers/HANDOVER_GPAW_PW_REFERENCE.md)
  — the PW-limit oracle this roadmap consumes.
