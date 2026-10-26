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
| **Source version (2026-10-19)** | 0.12.0 (M-0 done; code for M0, M1, M2 (0.11.0) and M3 (0.12.0) landed; the gates of M0 to M3 are open, see § 4) |
| **1.0.0 criterion** | first production basis published from this driver + frozen backend/transport API ([`VERSIONING.md`](VERSIONING.md)) |
| **vibe-qc's roadmap** | [`docs/roadmap.md`](../docs/roadmap.md) § "OPTBASIS-style basis optimiser" |

---

## GitLab tracking and weekly reconciliation

Because this package lives in the vibe-qc repository, its GitLab milestones
live in that development project with the `vibe-basis` prefix. They do not
share QC's version sequence. Find these exact titles on its **Milestones** page:

| GitLab milestone title | Project milestone ID |
|---|---|
| vibe-basis M0 - cohesive pipeline qualification | 9 |
| vibe-basis M1 - Gamma ZPE qualification | 10 |
| vibe-basis M2 - cohesive objective qualification | 11 |
| vibe-basis M3 - topology search qualification | 12 |
| vibe-basis M4 - resumable campaign orchestration | 13 |
| vibe-basis M5 - cross-validation ladder | 14 |
| vibe-basis 1.0 - production basis and API freeze | 15 |

Milestones hold capability/release goals and their exit criteria. Issues hold
individual deliverables, defects and acceptance evidence. A shared prerequisite
may be linked from several milestones without moving its existing assignment.
An empty issue list or a 100% closed count is not proof that the exit criteria
passed. Keep future research outside a bounded release unless it is explicitly
included in that release's scope.

Review this roadmap and its GitLab milestones **every week** against current
main, tags, changelog, relevant issues/MRs and accepted numerical or functional
evidence. Correct drift in both places and date material status changes.
Distinguish implemented, qualified, release-ready and released; source landing
does not certify numerical accuracy, and qualification does not create a tag.

When the documented capability criteria pass, mark the milestone release-ready
and advance the next feature-release target under the existing version policy.
The release owner then selects the version and updates package metadata through
the normal release process, reusing accepted evidence and completing required
candidate checks. Close a release milestone only after its containing tag is
confirmed. Do not invent a SemVer-major jump, codename, paper date or scope
reduction to make a milestone appear complete. Companion releases remain
independent; experimental work retains its documented status.

The already shipped M0-M3 implementation versions remain historical facts.
Completing their outstanding acceptance alone does not earn another code
version. New features/fixes follow [VERSIONING.md](VERSIONING.md), updating both
version sites together; M4/M5 receive numbers when release scope is ready.
`1.0` remains the production-basis result plus API freeze, not completion of
this documentation pass. M-0 is already complete and needs no duplicate open
milestone. #224 remains the M2 evidence issue and is linked from M0; #225
remains the M3 evidence issue.

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
p. 65, checked in the configured literature library), not recalled:

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

**The continuous optimisers are implemented.** vibe-qc ships `optimize_bdiis` — the same
BDIIS, hardened past the CRYSTAL baseline (BFGS inverse-Hessian driving the DIIS
error vectors; trust-region + Armijo guard forbidding an objective increase;
quasi-Newton fallback on an ill-conditioned DIIS `B`; box-projected convergence)
— plus `condition_number_penalty` for OPTBASIS's own γ·ln κ term. So this
roadmap does not need to catch up on *how to minimise*. It changes **what is
minimised, over what, and against what truth.**

| # | Axis | CRYSTAL23 `OPTBASIS` | vibe-basis |
|:-:|------|----------------------|------------|
| 1 | **Objective** | total energy (+ κ penalty) | **cohesive energy vs a PW-limit reference** |
| 2 | **Basis topology** | fixed (starred shells only) | **search implemented; scientific gate open** — d/f add-high / add-low / leave-one-out |
| 3 | **Systems** | one crystal | **joint multi-system**, weighted, held-out validation |
| 4 | **Geometry** | frozen | **planned re-relaxation per candidate**; VibeQcEngine variable-cell relaxation currently refuses |
| 5 | **Thermochemistry** | none | **Γ-point ZPE driver implemented; cross-route gate open** |
| 6 | **Reference truth** | none | **28-system PW reference data**; integrated engine reproduction remains open |

---

## 2. Engine policy

**vibe-qc is the engine. External codes are fallback and cross-validation only.**

| Role | Engine | Status |
|------|--------|--------|
| **Primary — Gaussian** | **vibe-qc** (BIPOLE / GDF) | the target for every calculation |
| **Primary — plane wave** | **vibe-qc** (GPW / GAPW) | not there yet — dependency BOC-7 |
| Fallback — Gaussian | CRYSTAL23 | pob-lineage continuity; head-to-head baseline |
| Fallback — plane wave | GPAW | today's PW-limit oracle (28 systems, validated) |

One `EnergyEngine` protocol; two implemented engine adapters and a GPAW stub.
The GPW/GAPW capability is a remaining vibe-qc dependency, not a fourth
implemented adapter. **No phase hard-codes an
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
| `VibeQcEngine` | 2 | crystals and atoms; declared `zero_point` (Γ phonons, M1) and `counterpoise` (BOC-3). No `relax` capability; direct variable-cell relaxation fails closed. |
| `Crystal23Engine` | 1 | emit, submit, parse, version-check and counterpoise implemented; direct-binary `invocation="stdin"` added. Bare-atom Mg failure remains open (#554). |
| `GpawEngine` | 1 | declared, raises; wired to the validated PW worker at M5 |

Optional stages follow `EnergyEngine.supports()`. `VibeQcEngine` declares
`zero_point` and `counterpoise`; `Crystal23Engine` declares `counterpoise`.
Neither advertises `relax`: the current vibe-qc adapter refuses variable-cell
relaxation until a certified BIPOLE path exists. Unsupported pipeline stages
are recorded as skipped with the engine named, never presented as completed
relaxation or a ZPE correction. Static-lattice reference reproduction can
proceed without claiming the full re-relaxed mission is achieved.

The package boundary is described in [README.md](README.md); the import guard
and the two-tier source layout enforce it. Gate evidence belongs in the
corresponding issue, not in removed handover documents.

---

## 4. Milestones

Implementation lands independently of subsequent scientific acceptance;
follow [AGENTS.md](../AGENTS.md). A source version is not a gate verdict.
The original 0.4–0.8 milestone targets were overtaken by intervening releases:
M2 actually landed in 0.11.0 and M3 in 0.12.0. The table therefore records
implementation versions, not future bump promises. Unimplemented milestones
receive a version through the normal release process when their scope is ready.

| # | Milestone | Implementation version | Gate | Status |
|:-:|-----------|:---:|------|:------:|
| **M-0** | Boundary cleanup + `EnergyEngine` protocol + CRYSTAL14→23 retarget | 0.3.0 | one engine abstraction; import-direction guard green; CRYSTAL23 `.out` round-trips with a version probe | ✅ **done** |
| **M0** | Cohesive pipeline: relax → hybrid SP → ZPE → atoms → assemble | 0.4.0 | two-engine: CRYSTAL23 reproduces `E_pob` < 1 kJ/mol; then vibe-qc reproduces CRYSTAL23 < 2 kJ/mol | 🟡 **API + engine landed; reference-arithmetic rung 1a passes; engine reproduction and cross-engine gates open. Variable-cell relaxation unavailable.** |
| **M1** | Γ-point ZPE on the Gaussian route (FD Hessian **from energies** — see BOC-2) | Included by 0.12.0 | MgO + NaCl optical modes within 5 cm⁻¹ of the GAPW route; acoustic within 15 cm⁻¹ of zero | 🟡 **driver + engine landed; gate open** |
| **M2** | `CohesiveObjective` vs the PW limit | 0.11.0 | evaluating at pob-TZVP-REV2 reproduces MD −24.6 / MAD 35.4 kJ/mol (n=28) | 🟡 **objective landed (0.11.0); rung 2a passes**: the published pob energies through `CohesiveObjective` give −24.58 / 35.40 with the static-lattice and counterpoise conventions pinned (`tests/basisset_dev/test_m2_gate_published_pob.py`); rung 2b, an engine reproducing those energies, is open (#224); its `VibeQcEngine` half is no longer blocked on BOC-3 |
| **M3** | **Topology search — the d/f scan** | 0.12.0 | Mg-d in MgO + Si-d in SiC: terminates, ranked Pareto front, **rediscovers the shipped pob topology** on a total-energy objective | 🟡 **topology code landed (0.12.0); gate open** (#225) |
| **M4** | Campaign orchestration on `vq` | Unassigned | 200-eval campaign survives a mid-run kill and resumes identically; zero writes into managed checkouts | ⬜ |
| **M5** | Cross-validation ladder (Gaussian ↔ PW limit) | Unassigned | ≥ 28 systems, rungs 1/2/2′/3 + extrapolation residual demonstrably smaller than the gap it measures | ⬜ |
| **M6** | Production run — the optimised basis | **1.0.0** | runs on `VibeQcEngine`; held-out MAD beats pob-TZVP-REV2's 35.4 kJ/mol by a stated margin at equal-or-lower primitive count | ⬜ |

Sequencing:

```
M-0 ─► M0 ──┬─► M1 ──┐
            │        ├─► M2 ─► M3 ─► M6
            └────────┘          │
                     M4 ────────┤
                     M5 ────────┘
```

M-0 is complete. M1 qualification and M2 reproduction can proceed separately:
the published M2 reference is explicitly static lattice, with ZPE off and
counterpoise on. That convention is not a temporary stub waiting for M1.
M4 and M5 implementation need not wait for M3's scientific gate.

### Current evidence and next gates (2026-10-19)

- **M0/M2:** reference arithmetic and objective algebra pass, not a fresh
  28-system calculation. `test_m2_gate_published_pob.py` reproduces MD
  **+24.58** in candidate-minus-reference convention (−24.58 in the published
  GPAW-minus-pob convention) and MAD **35.40 kJ/mol**, n=28. The CRYSTAL
  reproduction needs the original 28 reference cells, or an explicit decision
  to use other cells with a revised comparison tolerance (#224).
- **BOC-3 is landed (e96c2a7c):** the ghost-cluster implementation removes
  the counterpoise capability blocker. The reported MgO comparisons agree
  within 2.7e-6 Ha for Mg in the tested bases and about 1e-7 Ha for O/STO-3G.
  The O/pob-TZVP-rev2 CRYSTAL comparison is incomplete. These checks do not
  establish the 28-system cohesive gate; cluster size and SCF cost remain
  practical constraints (#224).
- **Bare-atom failure (#554):** the default non-counterpoise CRYSTAL route
  gives an unphysical Mg energy, about 0.31 Ha too low (roughly 800 kJ/mol
  per Mg). Do not use that result as a valid cohesive reference. The tested
  counterpoise route avoids this defect; other closed-shell atoms remain
  untested. Preserve this scoped failure while running eligible work.
- **M1:** the energy-difference Γ-Hessian driver exists, but its MgO/NaCl
  comparison with GAPW remains open. Γ-only ZPE is not a converged q-mesh
  thermochemical correction.
- **M3 (#225):** the CRYSTAL stdin path is fixed (0bd5b77d) and one MgO
  PW1PW/pob-TZVP-rev2 point converged at −275.47676757571 Ha. That is a
  prerequisite check, not a topology scan. The candidate-relaxation recipe,
  focused contract check and MgO/SiC campaign remain. A pure-total-energy
  Pareto front is a weak rediscovery test because added functions generally
  lower variational energy; discriminating cohesive-objective evidence still
  requires M2's engine reproduction.
- **M4–M6:** cache/restart primitives and vq transport exist, but the
  200-evaluation kill/resume gate, integrated cross-validation ladder and
  production-basis result are not established. `GpawEngine` is still a stub.

The next feature release can collect qualified driver changes without waiting
for a production basis. **1.0 remains tied to the published production basis
and frozen backend/transport API**, not the article deadline or the current
0.12.0 number. A paper must identify the exact code and evidence it uses;
implemented searches alone do not support an improved-basis claim.

---

## 5. Dependencies on vibe-qc

None of these are vibe-basis's to fix. Each is owned by a vibe-qc dev chat and
documented in the dependency table below.

| Id | Need | Blocks | Owner |
|----|------|--------|-------|
| BOC-1 | R13 Ewald-gauge dV(k)/dG(k) kernel → end-to-end periodic analytic basis gradient | *cost only* (~10× SCF without it) | periodic-SCF |
| BOC-2 | ~~BIPOLE analytic force accurate enough for FD phonons~~ **ANSWERED: it is a documented research preview, so M1 differences the *energy* instead. Not a bug; nothing owed by the periodic chat.** | — | — |
| BOC-3 | ~~Ghost-atom BSSE for solids~~ **LANDED: `VibeQcEngine` declares `counterpoise` and computes the free atom in an `ATOMBSSE`-style ghost cluster (`vibeqc.basis_optimization.counterpoise`).** | — | — |
| BOC-4 | GPAW oracle breadth: more systems, PBE0/HSE06, Cs > 800 eV | M5 | vibeqc-gpaw |
| BOC-5 | PySCF reference backfill in `tests/test_periodic_ecp.py` | ECP-element cohesive energies | basissetdev |
| BOC-6 | Hybrid periodic frequency cost | M1 at hybrid level | periodic / perf |
| BOC-7 | **vibe-qc GPW/GAPW producing a converged PW-limit atomization** | **retiring the GPAW fallback — the "all calculations in vibe-qc" goal** | gapw |
| BOC-8 | vibe-qc ↔ CRYSTAL23 agreement on cohesive energies | M0 gate 2; full engine reproduction still open after BOC-3 | periodic-SCF |
| BOC-9 | Certified BIPOLE variable-cell relaxation exposed by `VibeQcEngine` | Re-relaxed M0 pipeline and per-candidate geometry mission; not static-lattice reference arithmetic | periodic-SCF / basis optimisation |

**BOC-1 affects cost; it does not block eligible calculations. BOC-7 closes
the all-vibe-qc mission.** BOC-8 requires measured cohesive-energy parity,
not inference from total-energy agreement. BOC-9 records an existing missing
capability; it does not introduce a new requirement for static-lattice gates.

---

## 6. Sync contract with vibe-qc's roadmap

The two roadmaps are linked, not merged. The contract:

1. **vibe-qc's roadmap points here.** [`docs/roadmap.md`](../docs/roadmap.md)
   § "OPTBASIS-style basis optimiser" carries the pointer and records what
   shipped on the vibe-qc side (BDIIS, the κ penalty, the analytic gradients).
   It does **not** restate these milestones.
2. **This roadmap points there, once, per dependency.** § 5 is the
   authoritative dependency inventory. Add a missing capability there when
   source or gate evidence identifies a concrete vibe-qc prerequisite.
3. **When a BOC dependency lands in vibe-qc, § 5 updates the same session.**
   Use the current [AGENTS.md](../AGENTS.md) sync policy and read the
   source and issue evidence when a relevant main change lands.
4. **A vibe-basis release does not imply a vibe-qc release, or vice versa**
   ([`VERSIONING.md`](VERSIONING.md)). Implementation changes earn vibe-basis bumps;
   they do not appear in vibe-qc's milestone list.
5. **Resolve divergence using evidence.** Read current source and dated
   numerical evidence in the relevant issues, then update this file. An old
   status comment is not a measured gate verdict. Reconcile shared capability
   claims with `docs/roadmap.md` against the same implementation and evidence.

---

## 7. Open questions for the maintainer

Carried here because they change the shape of M2/M3/M6, not just their schedule.

1. **Objective target** — the PW limit (isolates basis incompleteness) or
   experiment-minus-ZPE (physical, but folds in the functional's own error)?
   *Recommendation: PW limit; experiment stays a reported diagnostic.*
2. **BSSE** — counterpoise for solids exists on both engines since BOC-3
   landed, but a counterpoise atom is host-specific and costs one cluster SCF
   per element per system, so it is priced for reference comparisons rather
   than for every optimiser point. Without it in the objective the optimiser
   can lower the objective by *increasing* BSSE, which is exactly what diffuse
   `add-low` candidates will do. *Recommendation: a diffuseness penalty as
   proxy inside the loop; counterpoise atoms for the reference comparison and
   the final candidates.*
3. **Γ-only ZPE** — sufficient, or does the paper need q-mesh phonon ZPE?
4. **Element order** — main-group d-polarisation (Mg, Al, Si, S) first, 3d TM
   later? *Recommendation: yes — TM brings magnetism and is where the PW oracle
   is weakest.*
5. **Production integration** — keep driver and pipeline on `main`; choose
   the production-basis integration and artifact destination when the campaign
   is ready. Historical branch assignments do not override current ownership.
6. **Dependency policy** — promote vibe-basis to a **core** vibe-qc dependency
   (collapsing the duplicated `OptResult`), or keep it an extra and keep the two
   pinned by contract test? Adding it to core costs one small pure-Python
   package (click + numpy). Changing the dependency contract remains a
   maintainer decision under the current contributor policy.
   *Recommendation: keep the extra; the contract test is proportionate for one
   nine-field dataclass.*

---

## 8. Literature and reference evidence

Use the configured literature library, catalogues and source manuals following
[AGENTS.md](../AGENTS.md). Reuse prior reading; request missing references
through the library's intake process. Keep local library locations out of
product documentation. The CRYSTAL23 OPTBASIS manual and the papers below
supply the formulation and reference conventions; implementation or arithmetic
checks do not replace the pending engine comparisons.

Reference papers and citation keys:

| Topic | Paper | Key |
|---|---|---|
| the incumbent optimiser | Daga, Civalleri & Maschio 2020 | `daga_bdiis_2020` |
| pob-TZVP | Peintinger, Vilela Oliveira & Bredow 2013 | `peintinger_pob_tzvp_2013` |
| pob-rev2 | Vilela Oliveira, Laun, Peintinger & Bredow 2019 | `vilela_oliveira_pob_rev2_2019` |
| the κ penalty | VandeVondele & Hutter 2007 | `vandevondele_basisopt_2007` |
| r2SCAN | Furness et al. 2020 | `furness_r2scan_2020` |
| SCAN | Sun, Ruzsinszky & Perdew 2015 | `sun_scan_2015` |
| PW1PW | Bredow & Gerson 2000 | `bredow_gerson_pw1pw_2000` |

Add new scientific references to the citation database and the relevant
output route with the implementation, following the current contributor
policy. Retain the source paper, conventions and numerical evidence together
in the configured research records.

## 9. See also

* [`README.md`](README.md) — what vibe-basis is, install, why a separate package.
* [`VERSIONING.md`](VERSIONING.md) — the version line and the 1.0.0 criterion.
* [`docs/roadmap.md`](../docs/roadmap.md) — vibe-qc's own roadmap (§ 6 above).
* [`docs/basisset_dev/`](../docs/basisset_dev/) — the basis-set development
  documentation set, incl. `GOAL8_MPEI_TZVP.md` (the 1.0.0 target basis) and
  `OPTIMISING_A_BASIS.md` (the in-process molecular optimiser user guide).
