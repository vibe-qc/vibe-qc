---
orphan: true
---

# Full MANTA implementation work tracker

Development tracker **#617, MANTA full implementation alongside libint:
capability and claimable work tracker**, covers the full approved design.
MANTA stays an optional provider alongside libint. Provider selection,
Coulomb algorithm and exchange approximation remain independent; BUPO supplies
J, not K. The full goal remains unfinished.

The [engine capability matrix](manta_engine.md#3-capability-matrix),
[scientific contracts](manta_scientific_contract.md) and
[implementation and acceptance plan](manta_validation_plan.md) retain their
technical requirements. Issue numbers below refer to the development tracker.
The issues contain starting files, a first bounded increment, dependencies,
acceptance criteria and pickup instructions. Read their current notes before
claiming work; this map does not reassign an active owner.

## Release boundary and accepted starting point

The [bounded 0.18 controls](../user_guide/manta_release_controls.md) are qualified
at containing main `bc67d176`. Their
[benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md)
records source/native identities, complete calculations, strict refusals,
independent comparisons, cost and exclusions. Reuse that evidence for unchanged
behavior. It does not establish whole-method automatic accuracy or production
speedup. Candidate acceptance remains #557; the wider work below does not
enlarge that release gate.

Existing molecular scalar/range/FT, direct/RI/BUPO consumers and finite-source
periodic HF/KS implementations are the starting point, not work to recreate.
An old stage label is not evidence that a whole row is either absent or accepted.
Conditional matrix entries require an implemented same-source route or a
documented mathematical obstruction and explicit refusal.

## Implementation packages

Completed packages retain their source and acceptance links; read the current
issue state before claiming work. Remaining packages stay independently
claimable within their stated dependencies.

| Issue | Design stage | Remaining deliverable |
| --- | --- | --- |
| #618 | P0/P1 | role-specific capability query, planner-derived admission for the kernel, one-electron and kernel-free producers, and the limit inventory (deep signed contractions, ghost center) are on main; remaining provider capability and libint coexistence coverage continues |
| #619 | P1c | libint-backend range-separated exchange (exact J/K), the shared guesses, spin seeds, ECPs (energies and exact/RI geometries), post-SCF dispersion, RHF/UHF harmonic frequencies, same-source atomisation references, QTAIM topology, molecular smearing and DFT+U energies and gradients are on main; remaining molecular method (ROHF/ROKS) and response compositions continue |
| #620 | P2b | add bounded large-metric RI and RI-BUPO solves |
| #621 | P2a/P2b | implement certified incremental direct and RI-BUPO builds |
| #622 | P3b | stored-graph pair-tree metric bounds accepted at `27d9c24d`; actual contraction arithmetic remains #624 |
| #623 | P1/P3b | bound integral and stored-source construction arithmetic |
| #624 | P2/P3b | paired/retained Coulomb, evaluated assembly, direct K, BvK correction and whole-mesh fitted K accepted through `9e3a8c9a`; input/XC and remaining arithmetic errors remain |
| #625 | P3b | opt-in omitted-image overlap rank guard connects shared spectral bounds to fixed-source and support-selection admission; conditional metric assembly rank and finite inverse/projector response are implemented; full construction arithmetic and occupied-space stability remain |
| #626 | P3b | signed range admission and grouped reference bounds accepted through `da33a7dd`; XC and useful whole-method bounds remain |
| #627 | P3b | fixed-density certificate composition with no double counting, outward scalar summation, fail-closed bound admission and source/family-qualified full-Coulomb replacement and disjoint fitted-K composition for HF and signed global/range KS, plus an uncoupled second-order relaxation delimiter, are on main (`component_ledger`); relaxation and whole-method bounds, automatic selection remain |
| #628 | P3b | extend direct periodic BUPO electrostatics to complete KS compositions |
| #629 | P3a | qualify same-source BIPOLE primitive substitution |
| #630 | P3c | derive and implement slab and wire source families |
| #631 | P4 | complete union-and-weight and WSSC provider adapters |
| #632 | P4 | complete neutral Bloch and real-Gamma MANTA source coverage |
| #633 | P4 | complete finite-character chi MANTA source adapters |
| #634 | P4 | resolve conditional semiempirical long-range reuse |
| #635 | P5 | qualify source-specific integral and BUPO symmetry reduction |
| #636 | P6a | complete selected KS and BUPO molecular derivatives |
| #637 | P6a | complete periodic and CCM gradients and cell stress |
| #638 | P1b/P2/P3b/P6b | matched direct J/K baseline and shell-quartet reuse measured ([benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_direct_jk_cost.md)); kernel arithmetic, dispatch and BUPO crossover remain |
| #639 | P6b | qualify optional GPU, distributed and mixed-precision execution |

## Capability matrix coverage

Every row of the engine capability matrix maps to the issues below. A row
can have several consumers or conditional source proofs; completing one
consumer does not close the row. These are work assignments, not new claims
of supported methods or accepted accuracy.

| Kernel or consumer row | Implementation and acceptance issues |
| --- | --- |
| Molecular S/T/nuclear, cross-basis S and moments | #618; method composition #619 |
| Two-center metric and three-center coupling | #618; bounded solves #620; arithmetic and rank #623-#625 |
| Four-center Coulomb and direct J/K | #618, #619; incremental builds #621 |
| erf/erfc two-, three- and four-center kernels | #618; periodic range accuracy #626; source adapters #629-#633 |
| Molecular exact J and exact K | #618, #619, #616 |
| Molecular RI-J and RI-K | #619, #620, #624; independent K admission |
| Molecular BUPO-J and RI-BUPO-J | #619-#621; measured crossover #638 |
| Gamma and multi-k exact/RI J/K | #576, #607; accuracy #623-#627 |
| Periodic BUPO-J and RI-BUPO-J | #622-#628; large metrics #620 and cost #638 |
| Nuclear gradients and cell stress | #636, #637; separate derivative capability admission |

| Numerical source row | Implementation and acceptance issues |
| --- | --- |
| Molecule or isolated finite cluster | #618-#621, #616 |
| Three-dimensional Gamma | #576, #607, #622-#628 |
| Three-dimensional multi-k and shifted meshes | #576, #607, #622-#628 |
| One- and two-dimensional periodic | #630 |
| BIPOLE active physical or historical source | #629; shared domain/state owners #531 and #606 |
| CCM union-and-weight and WSSC four-center | #631 |
| Neutral CCM Bloch | #632 |
| Neutral CCM real-Gamma | #632 and existing source integration #519 |
| Finite-character chi four-center | #633 |
| Finite-character chi RI | #633 |
| Finite-character chi RIJCOSX | #633; preserve the existing COSX K source |
| Low-dimensional CCM | #630-#633; each named source needs its own proof |
| SECCM, MSINDO-CCM, PM6, OMx, DFTB and GFN2 | #634; conditional long-range subterm reuse only |

Source-qualified symmetry is #635 with shared admission #577/#608.
Molecular and periodic derivatives remain #636/#637; CPU dispatch and
accelerators remain #638/#639. Correlation tiles and factors reuse #581.
These obligations apply across the relevant rows and do not follow from a
passing scalar-kernel or Fock-action test. Conditional cells require the
named same-source implementation or a documented mathematical obstruction
and explicit refusal.

## Existing issues reused

| Issue | Responsibility |
| --- | --- |
| #616 | Molecular evaluated-state integration alongside the libint bridge |
| #607 | Periodic bounded native evaluated-state/solver adaptation |
| #576 | Complete multi-k HF/DFT/hybrid MANTA control and parity ledger |
| #581 | Molecular/periodic correlation tile and factor admission |
| #519 | Neutral real-Gamma RG-8 source and consumer integration |
| #577 | Cross-method symmetry and Fock parity |
| #567 | Existing citation-route defect |
| #557 | Bounded 0.18 candidate acceptance only |
| #643 | Explicit automatic reference/metric and component reservations qualified at `4736aceb`; original insufficient-cap refusals preserved |

Shared Fock construction and eigensolvers remain under #472; PBC/BIPOLE and
CRYSTAL23 parity under #593; full-SCF cost acceptance under #574; DLPNO under
#508. These issues are dependencies and consumers, not competing implementations.
MANTA provider symmetry uses the shared algebra and admission authority; no
transport test alone permits production reduction. Correlation needs qualified
integral tiles/factors, not merely a J action. Library/dependency defects retain
their existing tracker owners.

## Shared interface ownership

| Interface | MANTA work | Shared implementation and acceptance |
| --- | --- | --- |
| Molecular physical Fock and evaluated state | #619 and #616 | #472 and #594; use the same borrowed/owned native bridge as libint |
| Periodic physical state and eigensolution | #607 | #472; preserve bounded native execution, spin/k frames and physical versus solver-modified operators |
| Periodic sources and complete-method parity | #628, #629 and #630 | #593 and #576; BIPOLE domain/cancellation policy stays with #531, its Fock adapter with #606 |
| Source-qualified symmetry reduction | #635 | #577 and #608; shared algebra does not grant producer or broken-state admission |
| Complete-method cost | #638 | #574; reuse accepted source-matched controls and include setup, terminal work and peak memory |

The [operator pickup map](operator_consumers.md#tracker-pickup-map) and
[PBC roadmap](../roadmap.md#development-reconciliation-2026-10-19) link back to
this MANTA tracker. When changing a shared producer or adapter, record the
specific API, affected consumer issue and owned files on the work item. Link
the landed source and applicable evidence from both sides of that dependency.
Retain source/metric receipts separately from AO, spin, k-point and solver
frames. Auxiliary-metric factorization is not an orbital Fock eigensolve.
Coordinate the particular missing interface; do not wait for all parent issues
to close or create another state object or review chain.

The concrete shared bridge #594, complete-method parity #576, BIPOLE state
adapter #606 and cost acceptance #574 also link back to #617 and their
respective MANTA consumers. Update the implementing issue and the affected
consumer when an interface lands; reuse the same source and evidence links.

## Production readiness ledger

Snapshot of 2026-10-24 at containing main `4736aceb`, reconciling every
capability-matrix row with what is implemented, what carries recorded
numerical acceptance, what the manual supports within an explicit envelope
and what stays experimental. Closed-issue counts are not readiness; an `E`
or stage label in the [matrix](manta_engine.md#3-capability-matrix) does not
qualify its row. Four states are used. **Implemented**: the path exists
behind its method guards and its focused tests pass; no independent
numerical acceptance is claimed. **Validated**: a complete affected-method
calculation was compared with an independent reference (ORCA/CRYSTAL, an
independent NumPy/SciPy or Gaussian/Boys/Fourier construction, or a known
limit) with recorded source/native identity, inputs, tolerances and measured
errors, within the limits those records state. **Supported**: validated,
documented with its envelope (methods, sources, bases, limits, cost), refused
outside it before expensive work, and no longer labelled experimental for
that envelope. **Experimental**: everything else, including validated
components whose whole-method accuracy, cost or failure behaviour is not
established. Today no row reaches Supported: every MANTA route keeps the
experimental label ([manual](../user_guide/manta.md),
[engine guide](../user_guide/integral_engines.md),
[release controls](../user_guide/manta_release_controls.md)).

### Baseline refreshed on 2026-10-24

The bounded 0.18 MANTA contribution (#557) is a PASS for candidate
`5dfc97f0`: build 24463 and candidate pipeline 14816 succeeded, the final
native receipt's core SHA256 and its 17 dependency hashes match the verified
predecessor, and the receipt's native tree equals the candidate tree (tracker
#557, 2026-10-24). The accepted four-profile evidence remains the local
calculation set `vq9c085769c600` on its own local core: 8 converged
calculations, 10 strict physical refusals and 1 expected old-cap refusal,
reusing the original HF and continuous-grid PBE0 component and independent
references ([benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md)).
The local scientific core and the Linux candidate core are distinct binaries;
neither is a deployed runtime, and no cross-platform numerical parity is
claimed. That acceptance covers the four frozen profiles only (shifted
two-point Be RHF/PBE0 and a compact Gamma He RHF with RI-J, exact
full-Coulomb K, fixed supports, unreduced meshes). On 2026-10-24 the
`compact-tight` profile re-ran bitwise unchanged on main `d8f937fb` after the
#638 kernel change (all 1e-8 targets met; exchange 8.9e-12 / 3.0e-10 Eh,
nuclear 1.9e-17 Eh, kinetic 3.9e-13 Eh; local core, not a candidate binary).

### Kernel and consumer rows

| Row | State | Evidence and envelope | Open |
| --- | --- | --- | --- |
| Molecular S/T/nuclear, cross-basis S, moments | Implemented; Bloch S/T validated | One-electron operations through l=6 and moments to order 6 behind `one_electron_capability`; ECP stays its own provider. Periodic Bloch S/T parity ≤4.5e-15 ([benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_bloch_one_electron.md)); no hashed molecular acceptance page | #618 |
| 2c metric and 3c coupling | Implemented | `FittingTiles` auxiliary l≤7, orbital l≤6, full/erf/erfc, no derivative; dense metric/factor only | #620 bounded solves; #623-#625 arithmetic and rank |
| 4c Coulomb and direct J/K | Implemented; consistency-checked | Direct digestion through l=6 matches the libint reference to 1e-10 Ha (water/cc-pVDZ) and to 3e-12 Ha on four small fixtures with identical shell schedules; 9.5x slower than libint after shell-quartet reuse ([cost benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_direct_jk_cost.md)); screening none by design | #638 kernel arithmetic and dispatch; #621 incremental builds |
| erf/erfc 2c/3c/4c kernels | Implemented; periodic component bounds validated but loose | Kernels exist; the erfc quadrature target 2e-14 is not a rigorous bound; the libint backend refuses range-separated exchange. Periodic raw and pre-SCF signed range bounds accepted at `f1698506` and `1c06da0c` with matrix bounds 55-247 and energy bounds 1.8e4-8e4 Eh ([metric benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md)) | #626 useful bounds; #629-#633 source adapters |
| Molecular exact J and K | Implemented; consistency-checked | RHF/UHF/RKS/UKS gas phase through `run_*` and `run_job` with the shared guesses, spin seeds, ECPs, dispersion, atomisation, QTAIM, smearing, DFT+U energies and gradients, HF gradients, optimisation and RHF/UHF frequencies; refused: ROHF/ROKS, response, correlation, solvation, RKS/UKS and DFT+U Hessians. No hashed acceptance page and no matched ORCA comparison recorded | #619 remaining compositions; #616 evaluated-state bridge |
| Molecular RI-J and RI-K | Implemented; consistency-checked | All eight J×K selections; RI HF Lagrangian gradients including fitted ECP results; RI-J −29 µHa and RI-J/K +21 µHa versus exact on water/cc-pVDZ (manual, unhashed) | #619, #620, #624 |
| Molecular BUPO-J and RI-BUPO-J | Implemented; construction validated | Exact terminal confirmation; J error 8.6e-14 and energy 1.3e-13 on the benchmark cases, but no far block passes strict admission on water or the dimers and optimized libint is faster than every MANTA path ([BUPO benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_bupo.md)); gradients and incremental builds gated | #620, #621, #638 crossover |
| Gamma and multi-k exact/RI J/K | Implemented; component-validated on frozen profiles | Neutral all-electron 3-D cells with explicit fixed supports, HCORE/AUTO guess, no DIIS; release-control profiles accepted (above); fitted-K radii ≤1.9e-15 Fock and ≤9.4e-14 Eh accepted at `9e3a8c9a`; PySCF AFT parity 3.5e-13 Eh and complete RHF reference 1e-15 Eh (unhashed pages). No automatic accuracy profile: domain series versus GDF reach +0.74 mHa (Gamma) and −0.15 mHa (2,2,2) only at the largest domains | #576 complete parity; #607; #623-#627 |
| Periodic BUPO-J and RI-BUPO-J | Implemented (HF, exact reciprocal K); experimental RI multipoles | Public RHF/UHF only; RI multipoles labelled experimental, qualified for RI-J with exact K or pure KS; BUPO J ≤2.0e-13 / energy ≤2.8e-14 on sources `b6054324`, `4c5ceb97`; public job F ≤2.6e-13 / energy ≤1.9e-11 on `06731fc7`; RI accuracy page `82547576`: actual errors ≤1.3e-16 against bounds up to 8.8e-6; pair-tree metric bounds at `27d9c24d`; complete RHF slower than exact finite RI | #622-#628, #620, #638 |
| Nuclear gradients and cell stress | Implemented (molecular HF exact/RI incl. ECP, DFT+U); absent periodic | Molecular HF gradients validated against finite differences and the legacy gradient in tests, no hashed page; KS gradients, BUPO gradients, periodic gradients and stress refused | #636, #637 |

### Numerical-source rows

| Source | State | Evidence and envelope | Open |
| --- | --- | --- | --- |
| Molecule / finite cluster | Implemented | Rows 1-7 above; no hashed acceptance, experimental label | #616, #618-#621 |
| 3-D Gamma; 3-D multi-k and shifted meshes | Component-validated on frozen profiles | Release controls on `7917b560`, grid integration `9a868249`, corrected candidate `2d0fdb8c` and today's `d8f937fb` re-run; image 1e-6 and compact 1e-8 targets met; image-tight RI J 1.3e-9 / energy 2.5e-7; PBE0 versus independent reference 5.3e-15 Eh. Excludes whole-method relaxed energy, basis/k/XC convergence, bulk parity, symmetry and speedup | #576, #607, #622-#628 |
| 1-D / 2-D periodic | Not implemented | Periodic route is 3-D only; dimensionalities refused; low-dimensional boundaries excluded from the release scope | #630 |
| BIPOLE active/historical source | Not implemented | No MANTA adapter; supports do not substitute BIPOLE AC/BD domains | #629 (#531, #606 owners) |
| CCM union-and-weight / WSSC | Not implemented | CCM MANTA routes unqualified | #631 |
| CCM neutral Bloch | Not implemented | Same | #632 |
| CCM real-Gamma | Partial | `MANTAFitSource`, `MANTAOneElectronSource`, `MANTAXCSource` fold sources reproduce fitted HF/KS energies per cell under the experimental AICCM route; no hashed acceptance | #632, #519 |
| chi four_center / ri / rijcosx | Not implemented | No adapter documented | #633 |
| Low-dimensional CCM | Not implemented | | #630-#633 |
| SECCM / MSINDO-CCM / PM6 / OMx / DFTB / GFN2 | Not implemented | Conditional long-range reuse only | #634 |

### Cross-cutting

Symmetry: the finite-product algebra correction `d5a519cb` is an
experimental handoff, not producer qualification; the bounded release
controls do not depend on it (#635). Arithmetic: paired, evaluated-assembly
and fitted-K arithmetic are accepted through `9e3a8c9a`; input/XC
construction and remaining non-paired arithmetic stay open (#623, #624).
Automatic accuracy: the three fixed-density certificates are separate
ledgers with different reference objects and must not be summed into a
relaxed-energy tolerance; no relaxation, overlap-rank or k-mesh contribution
exists (#625, #627). Cost: MANTA is slower than libint on every measured
path and no BUPO crossover is established (#638). GPU, distributed and
mixed precision: not started (#639).

### What this means for a production verdict

Supported configurations: none without the experimental label. The
narrowest candidate envelope is molecular RHF/UHF/RKS/UKS with exact or RI J
and K on all-electron and ECP Hamiltonians, where results match the libint
route to 1e-10 Ha and gradients match finite differences, but it lacks a
hashed acceptance page, a matched ORCA comparison and acceptable cost. The
periodic envelope is the four frozen profiles with fixed supports, which
carry component certificates but no whole-method accuracy or bulk parity.
Failure behaviour is explicit: unsupported compositions refuse by name before
the basis is built, and unmet accuracy targets refuse before an archive is
written. Remaining restrictions are the open rows above; the release-owned
0.18 scope stays unchanged by this ledger.

## Parallel work and acceptance

Useful independent starts include the capability inventory, molecular job
composition, same-source BIPOLE adapter, conditional source proofs, derivative
primitives and benchmarks using already accepted APIs. Pair-tree stored-graph
metric bounds and the executed paired metric substep, including its exact-adjoint
J contribution, individual coupling actions and their paired J/energy
composition, including spin/k weighting, evaluated HF/KS assembly and
stored-input direct K, BvK correction, retained J and whole-mesh fitted K,
including signed evaluated Fock/energy composition, are accepted. Input/XC
construction, remaining
non-paired arithmetic and automatic
periodic accuracy remain the ongoing MANTA integration lane. The paired metric evidence is recorded in the
[metric benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#whole-mesh-fitted-exchange-arithmetic).
It does not close #624 or certify a complete arithmetic budget.
The [exchange fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#full-coulomb-fitted-exchange-component-certificate)
qualifies a separate full-Coulomb K component bound under #626. Its uniform
family bounds remain conservative. RI HF attachment is accepted at `2dfbd54d`;
Global-hybrid KS attachment is accepted at `2dfb076d`; raw erf/erfc component
bounds are accepted at `f1698506`; pre-SCF signed range composition is accepted
at `1c06da0c`. Optional grouped reference bounds are accepted at `da33a7dd`.
XC, further reference tightening and useful source selection remain #626/#627.
Claim a bounded scope and intended files on the selected issue before editing
shared producers, and preserve existing owners, jobs and checkout changes.
Unassigned does not mean unowned. A dependency is the specific needed capability,
not a wait for every umbrella issue to close.

Use existing producer, Fock, numerical and symmetry APIs. Keep physical source,
normalization, support, rank, zero-mode, charge and evaluated-state identities
explicit. No silent dense fallback, source substitution, default change or
removal of experimental labels follows from this tracker.

The implementing chat may implement, validate and accept its own increment.
There is no mandatory second reviewer or review-only hold. Land focused,
checked source promptly with ordinary CI skipped. Scientific acceptance and
issue closure require a complete affected-method calculation through `vq`,
matched to ORCA/CRYSTAL where suitable or to justified numerical/high-precision
references with limitations stated. Reuse valid evidence and a healthy runtime
containing the fix; record actual source/native identity, inputs, job/output
and comparison. Full CI is reserved for release candidates. Update the manual,
benchmark, citations and changelog when corresponding behavior ships.

Keep #617 open until every original matrix and stage obligation is
accounted for by accepted work or an expressly conditional obstruction/refusal.
Finishing #557 or one bounded work package does not complete the full goal.


The optional stored-operand fitting metric assembly guard advances #623/#625:
it feeds actual tile-source entry enclosures into the existing strict spectral
rank radius. Primitive/Fourier/phase/weight construction errors and inverse
response remain separate open work. It does not retire the full source-error,
automatic-accuracy or production-readiness obligations under #617. See
scientific contract Section 5.55 and the metric spectrum benchmark.


Finite retained-projector and truncated-inverse response now consumes the
conditional #623 assembly perturbation in the shared inverse certificate
(#625, contract 5.56). It includes partial-rank rotation and moving relative
cutoffs, and composes the stored-factor error without changing factors.
Coupling/source construction and J/K/energy propagation remain open; this
adds no whole-method accuracy or production verdict.


The optional #627 metric-action target can now include the #623/#625
conditional metric-assembly inverse response, propagated through the shared
explicit or supported compressed coupling norms and fixed density support.
Separate source/stored components avoid double counting; default physical
sources and the global ledger are unchanged. Primitive/coupling construction,
K propagation and total relaxed accuracy remain open (contract 5.57).


Conditional metric-source K response (#626) now has separate optional raw
component targets across actual Gamma and ordered q owners, including
physical range kernels (contract 5.58). It consumes the shared #623/#625
assembly/inverse proof and coupling norm producer. Existing fitting/signed
aggregate/global ledgers stay distinct. Source operands, coupling, metric
domain/screening response and relaxed accuracy remain unfinished under #617.


The #623 short-range metric Bloch phase is now enclosed from exact stored
q/R vectors, including dot-product and sine/cosine evaluation, by the shared
bounded Taylor/angle-doubling proof (contract 5.59). Existing optional metric
assembly/rank and J/K source-response targets consume it. Physical sources
and states remain unchanged. Primitive/Fourier/coupling-phase/weight and
q/R construction, domains and whole-method accuracy remain unfinished.


The #623 reciprocal metric-weight construction proof (contract 5.60) now
includes stored-vector squares, stored range parameters and exp evaluation,
using the shared interval producer and existing physical kernel/zero modes.
Metric rank/inverse and conditional J/K targets consume the expanded proof.
The reciprocal prefactor, Fourier/coupling construction, domains, relaxation
and useful automatic whole-method accuracy remain unfinished under #617.


The #623 reciprocal metric prefactor proof now includes pi and the exact
stored-lattice determinant (contract 5.61), consumed by existing weight,
rank/inverse and conditional J/K producers. Strict unresolved-volume refusal
preserves the physical cell. Primitive/Fourier/coupling and vector generation,
domains, relaxation and useful automatic whole-method accuracy remain open.


The #623 auxiliary Fourier construction enclosure (contract 5.62) now covers
primitive Gaussian moments/decay, phase, signed contraction and arithmetic on
stored representation coefficients across auxiliary l=0..7. The existing
metric/rank/inverse/J/K chain consumes it; numerical panels remain unchanged.
Basis/representation data generation, short-range/coupling construction,
vector generation, domains and useful relaxed accuracy remain open.


The #623 two-center metric construction proof (contract 5.63) encloses radial
seeds, primitive prefactors, center shifts, Hermite contraction and application
of stored pure transforms through l=7. It feeds shared rank/inverse/J/raw-K
bounds without changing numerical source or states. Basis/representation and
vector-data generation, three-center/AO-pair coupling construction, domains,
screening and relaxation remain unfinished; #623/#625/#617 stay experimental.


The #623 positive erfc radial proof (contract 5.64) tightens the short-range
construction component without changing its reference or numerical producer.
Bernstein positive-series panels and explicit tails avoid full/long-range
subtraction; the ordinary Boys proof remains shared and unchanged. Existing
metric/rank/inverse/J/raw-K consumers propagate the result. Useful automatic
accuracy, coupling/data construction, domains and relaxation remain open.


The #623 representation construction proof (contract 5.65) removes the stored
pure-matrix premise from auxiliary Fourier and two-center metric certificates.
Exact rational-polynomial/sqrt bounds feed shared rank/inverse/J/raw-K without
changing numerical transforms or states. Basis/vector data generation, coupling,
domains/screening and relaxed response remain open; no symmetry permission is
inferred. #623/#625/#617 remain experimental and unfinished.


The #623 orbital-pair Fourier construction component (contract 5.66) adds a
shared certificate through orbital l=6 for Gaussian products, exact product
centers, phase/moments, contractions and both pure transforms. It retains the
numerical source and is currently a separate diagnostic. Full three-center
coupling composition and propagation, stored input generation, domain/screening
and relaxed response remain open. No existing metric/Fock/symmetry admission
inherits this component automatically; #623/#617 remain experimental.


The #623 streamed three-center construction component (contract 5.67)
covers the exact stored auxiliary/orbital Gaussian integral through l=7/6/6
for Coulomb/erf/erfc. Shared radial, elementary and representation proofs feed
per-tile Hermite/Coulomb construction bounds without changing numerical tiles.
Full periodic coupling composition and action propagation remain open, along
with stored-input/image generation, domains/screening and relaxed accuracy.
This component does not expand existing metric/Fock or symmetry admissions.

The #623 finite periodic coupling diagnostic (contract 5.68) composes shared
three-center, Fourier, phase, lattice-weight and representation proofs over
literal selected integer image sets, including translated centers and the
finite-zero term. Separate tile/forward/adjoint certificates compare unchanged
numerical routes directly with that unscreened finite reference. It closes
the finite construction composition gap; propagation into J/K/RI-BUPO/SCF
admission, input/domain completeness and relaxed accuracy remain open. Existing
Fock/PBC/symmetry interfaces and metric budgets are preserved. #623/#617 remain
experimental and unfinished; overlapping component budgets must not be added.

The retained-source discrepancy #687 is repaired by sharing the existing
budgeted primitive selector between MANTA and libint, including dense and
streamed paths. Screened source identities and split-call admission reflect
the retained primitive set; unscreened sources stay unchanged (contract 5.69).
Finite coupling/metric composition into J/K/RI-BUPO/SCF remains the next open
accuracy increment; the primitive repair does not close #617 or that work.
