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
| #618 | P0/P1 | complete provider capability and libint coexistence coverage |
| #619 | P1c | finish molecular method, guess and response compositions |
| #620 | P2b | add bounded large-metric RI and RI-BUPO solves |
| #621 | P2a/P2b | implement certified incremental direct and RI-BUPO builds |
| #622 | P3b | stored-graph pair-tree metric bounds accepted at `27d9c24d`; actual contraction arithmetic remains #624 |
| #623 | P1/P3b | bound integral and stored-source construction arithmetic |
| #624 | P2/P3b | paired/retained Coulomb, evaluated assembly, direct K, BvK correction and whole-mesh fitted K accepted through `9e3a8c9a`; input/XC and remaining arithmetic errors remain |
| #625 | P3b | connect metric and overlap stability to automatic error admission |
| #626 | P3b | complete fitted/range exchange and XC accuracy contributions |
| #627 | P3b | deliver useful automatic whole-method periodic RI-BUPO accuracy |
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
| #638 | P1b/P2/P3b/P6b | establish useful CPU dispatch and BUPO crossover |
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
| #643 | Existing automatic reference refinement/component budget refusals |

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
