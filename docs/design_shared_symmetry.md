# Shared symmetry architecture and production-reduction gates

The shared group, space and diagnostic contracts are implemented, with
molecular and periodic consumers. This page develops the rationale of the
original 2026-09-06 design into the remaining method-integration gates.
It does not enable a solver or change numerical acceptance thresholds.
See the [developer reference](developer_shared_symmetry.md) for implemented
APIs, the [capability status](symmetry_status.md) for legacy scope, and the
[chi symmetry reference](experimental/aiccm2026dev_b_symmetry.md) for that
consumer's finite-source diagnostics.
The [October literature assessment](symmetry_literature.md) records the new
papers, inspected versions, supplements and their implementation implications.

## Architecture

Use the shared symmetry module with numerical backend adapters. A unified
Fock-engine refactor is not a prerequisite for these contracts. Symmetry has uses
outside Fock construction: occupied/virtual spaces, density fitting, local
correlation tensors, nuclear displacements and response equations. A common
Fock engine is therefore neither necessary nor sufficient for this work.

The reusable object is an action of an admitted group on an explicitly named
space. It is not a universal recipe for replacing every method's computation
by a scalar multiplicity. Share representation, orbit and transport machinery;
require each operator and approximation to establish its own admissibility.

## What can be reused now

| Existing source | Reusable capability | Boundary that must remain explicit |
| --- | --- | --- |
| `symmetry_shared.py`, `cpp/src/symmetry_shared.cpp` | Exact group/cocycle validation, compact block actions, orbit stabilizers, typed scatter/gather and operator/subspace audits | Shared evidence never authorizes production reduction. Native budgets are logical payload/work bounds, not fleet memory guarantees. |
| `symmetry_core.py`, `symmetry_ao.py` | Atom/radial matching, pure-shell Wigner rotations and compact molecular pure/Cartesian actions through `l=6` | Cartesian blocks share the periodic polynomial kernel and are generally not Euclidean-unitary. The legacy dense API requires pure shells beyond s. |
| `cpp/src/periodic_ao_bloch_transport.cpp`, `kmesh_address` primitives | Bounded native Seitz action, full atom-image shifts, Bloch phases, mesh compatibility, scalar time reversal; Cartesian and pure shells through l=6 | One panel action is not an operator or state symmetry certificate. Spinor and magnetic actions require additional representations. |
| `cpp/src/periodic_mean_field_state.cpp`, `cpp/src/periodic_orbital_sewing.cpp` | Immutable numerical reference, metric/Roothaan audits, frozen/active/virtual mask leakage checks | Current sewing explicitly reports physical source symmetry as uncertified. A converged snapshot does not prove arbitrary-density operator equivariance. |
| `symmetry_lattice_c.py`, `pair_resolved_truncation.py` | Atom-pair/image action, geometric pair-distance domains, shell masks and orbit utilities | Lower-level atom matching carries fractional translations, but several callers still filter them out. Audit the actual function, not its historical symmorphic-only docstring. |
| `bipole_symmetry_fock.py`, native domain-aware J/K builders | Full masked builds, separate output/density/internal domains and representative reconstruction | Validate unreconstructed full builds first. Existing high-level mapping filters nonzero fractional translations. Changing domain membership changes the finite approximation. |
| `periodic_k_symmetry.py` | Phase-aware k-star density expansion and inverse folding; star maps restricted to an admitted operation subset, which may include glides and screws (the fractional translation travels in the atom-image shift, with no separate `exp(-i k.tau)` factor) | Its default star map is the zero-fractional-translation subset, for consumers that have not qualified fractional translations on their source; representative k points alone do not reduce every two-electron contraction. The multi-k GDF exchange wedge admits every operation on the SR/LR source and equal-shift operations on the classic source; other drivers' wiring needs its own audit. |
| `symmetry_integrals_reduced.py`, `symmetry_fock_reduced.py`, `symmetry_salc.py` | Representative one-electron integrals, Gamma J/K prototype, point-group projection ingredients | Gamma, point-group and pure-shell assumptions cannot be silently generalized to multi-k chi. |
| `symmetry_abelian.py` | Complex one-dimensional irrep products, conjugates and bounded finite-ERI J/K block execution | Experimental opt-in adapter; full input tensor required, no integral skipping or production SCF routing, no antiunitary/projective character admission. |
| `symmetry_quartets.py` | Complete-source-audited molecular shell orbits, stabilizers and streamed representative-tile J/K | Experimental bounded shell schedule using full AO blocks. An initial all-ordered-tile audit is mandatory; repeated builds request representatives from the same immutable provider. No default solver or periodic routing. |
| `symmetry_quartets_streaming.py` | Pair-orbit/stabilizer enumeration without a retained quartet inventory | An opt-in schedule retaining `O(order * n_shells**2)` index arrays; complete source audit and arbitrary-density contractions remain quartic. Full AO transport, not irreducible-product or CC recoupling. |
| `symmetry_quartets_native.py`, `cpp/src/symmetry_quartets.cpp` | Native four-leg transport and arbitrary-complex-density J/K accumulation per representative tile | Provider-independent real-tile algebra. Python retains source admission, schedule and digest checks; native execution is opt-in and does not qualify a source, screen or state. |
| `symmetry_quartets_batch.py` | Independent density channels sharing each representative tile and angular transport | NumPy and native executors return per-channel J/K and residual bounds with one aggregate work receipt. Batch membership does not establish electronic-state symmetry or enable new integral sources. |
| `periodic/chi/symmetry.py` | Finite-torus actions using shared exact algebra checks, occupied-space action and native snapshot bridge | Dense diagnostic witnesses and count caps are not a scalable production representation or execution plan. |
| `periodic/chi/pno.py` | Compact translation-pair census, even-mesh stabilizers, conditional point-pair quotient | Orbit counts alone provide neither orbital phases nor local-space transport or production pair skipping. |
| `periodic/ccm/symmetry.py` | Cluster compatibility and independent small-system AO checks | Explicit supercell matrices do not scale to the target; WSSC/union-weight and chi finite-character Hamiltonians must retain separate operator identities. |

Paths in this table are relative to `python/vibeqc/` unless prefixed by `cpp/`.
Reuse means tests and implementations can be consolidated incrementally; it
does not mean all present contracts are interchangeable.

## Why a single unconditional reduction recipe would be wrong

1. **Geometry, operator and electronic state are different tests.** A crystal
   operation can preserve the geometry while an image cutoff breaks the
   numerical Hamiltonian, as in the chi half-translation diagnostics. For a
   density-dependent Fock builder the relevant operator relation is
   equivariance, `F[g.D] = g.F[D]`, with the
   appropriate covariant/contravariant AO transformations. Invariance of one
   final `F[D]` additionally requires an invariant density. A broken-symmetry
   UHF state can use a smaller subgroup than its geometry. Averaging a Fock
   matrix is not evidence that its producer satisfies either relation.
2. **A localized orbital need not map to one other orbital.** Symmetry can mix
   several occupied orbitals. Casassa et al. explicitly discuss the arbitrary
   orientations of Boys-localized MgO sp3 combinations. Their Eq. (3) allows
   matrix-valued actions within localized subsets. Independent pair domains
   and PNO thresholds need not commute with that mixing. Use a validated
   compatible gauge or covariant block spaces; a pair permutation is a special
   case, not the default theorem.
3. **Different tensors have different actions.** AO matrices, three-center
   factors, occupied-pair amplitudes and triples carry different index types,
   conjugations, momentum relations and stabilizers. Auxiliary-metric rank
   cuts, frozen-core masks, PAO/PNO/TNO projectors and near-degenerate retained
   spaces all need compatible transport. CCSD residual couplings and the
   occupied-Fock-coupled triples correction must survive reconstruction.
4. **Gradients are derivatives of the same approximate energy.** Force vectors
   and displacement representations can share symmetry machinery, but this
   does not supply missing Pulay, kernel, fitting, grid or correlation-response
   terms. A displaced geometry generally has a smaller symmetry group. An
   energy invariant at the high-symmetry geometry does not validate every
   displacement derivative, and zero forces on ideal MgO are a weak test.

Dovesi (1986), Sections 3-4, especially printed p. 1763, identifies both the
reuse of symmetry relations across integral types and the loss of Fock
symmetry under truncated Coulomb/exchange sums. Casassa et al. (2006), printed
pp. 727-728, Eq. (3), gives the more general localized-space transformation.
References: [Dovesi, DOI 10.1002/qua.560290608](https://doi.org/10.1002/qua.560290608),
[Casassa et al., DOI 10.1007/s00214-006-0119-z](https://doi.org/10.1007/s00214-006-0119-z).

## Shared contract and adapter responsibilities

The [developer reference](developer_shared_symmetry.md) describes the current
internal classes and functions. Their common responsibilities are:

* **Group and mesh:** geometry/basis identity, admitted subgroup, Seitz
  composition, integer atom images, reciprocal wraps, antiunitary flags and
  finite-torus compatibility. Preserve full integer shifts until the relevant
  character is evaluated. A fractional translation in a nonprimitive cell
  must not be discarded merely because older helpers call it nonsymmorphic.
* **Space action:** compact atom/shell maps and small rotation blocks provide
  AO actions and occupied/selected-space audits. Auxiliary, local-pair
  and response adapters must supply their own tensor and projector semantics.
  Transport blocks on demand. Avoid dense all-supercell AO matrices
  and explicit lists of every translated pair. Retain native allocation/work
  admission before materialization.
* **Operator qualification:** operator convention, source revision, numerical
  supports, screening policy and tolerances, plus state/subspace checks.
  Record what is established analytically and what is only numerically
  tested. Finite probe densities alone do not prove a universal bound.
  Density-dependent screening must also respect the chosen action and budget.
* **Orbit execution:** representatives, stabilizers, exact weights and typed
  scatter/gather, with consistent adjoints. Reconstruction must account for
  operations stabilizing a representative, not just multiply by orbit size.
  An admitted group and a transport map alone cannot authorize pair skipping.
* **Method adapters:** BIPOLE/four-center, GDF/RI, molecular SCF, chi local
  correlation and derivatives each declare their source and tensor contracts.
  Equivalent representation infrastructure must not erase different finite
  Hamiltonians, q=0 conventions, fitting metrics or derivative semantics.

This supports one implementation of the common mathematics with several
operator adapters. It does not promise equal acceleration for all methods or
systems. Identity-only execution is the correct outcome where no larger
electronic/source symmetry has been qualified.

## Implemented foundation

The shared native group/space interfaces and two source-distinct consumers,
molecular integrals and periodic chi transport, are present. They reuse the
same exact group validation and compact action contracts. Molecular Cartesian
and periodic polynomial rotations also share one native kernel. The SALC
consumer uses common group and conjugacy checks. Metric, selected-space,
group-law and operator-leakage diagnostics build on these primitives.

The historical chi finite-source defect has a quantitative diagnosis and
opt-in finite-source witnesses. The default historical source remains subject
to refusal; those witnesses do not certify the full physical Hamiltonian or
representative-only correlation. Similarly, a complete whole-mesh J/K producer
does not by itself qualify symmetry reduction. These are distinct acceptance
scopes, not reasons to reimplement the shared foundation.

## Remaining production acceptance gates

1. **Qualify a declared numerical source.** For the selected method, identify
   the actual finite supports, screening, q=0 convention and density/state
   dependence. Validate unreconstructed J/K or fitted factors first, against
   an independent same-convention reference and nonsymmetric trial densities.
   A finite-source witness is not a default-Hamiltonian migration. Changes to
   support membership require separate physical/convergence acceptance. Keep
   scientific tolerances unchanged and retain an unreduced fallback.
2. **Add and qualify method-specific tensor adapters.** The bounded SR/LR GDF
   retained-mode diagnostic now connects actual fit batches to the shared
   subspace audit; it does not admit a subgroup or enable production work
   reduction. General GDF/RI factors, auxiliary metrics
   and rank cuts need index/conjugation/momentum contracts in addition to AO
   transport. Admit only operations preserving the chosen finite source.
   Reuse existing planner repairs and assess their independent acceptance
   before adding another implementation. Producer redesign, routing and cost
   decisions remain separate from the common mathematical layer.
3. **Transport retained local spaces and contractions.** Bind the actual
   occupied gauge and frozen/active spaces to PAO, auxiliary,
   PNO and TNO representations. Compare representative and unreduced MP2,
   CCSD residuals/energies, and coupled triples on small odd/even meshes.
   Include dense occupied mixing, fractional translations, degenerate retained
   subspaces, negative leakage cases and nonsymmetric trial densities.
4. **Establish scalable execution.** Where global factor/tensor construction
   remains, replace it with bounded local production data access. Validate
   increasing MgO meshes through `vq`, recording source
   and runtime identities, energy components, approximation settings, peak
   memory and actual contraction counts. Run 8x8x8 only after memory and
   small-mesh numerical gates pass.
5. **Validate derivatives separately.** Assemble chi RHF total analytic
   gradients using its own declared operator/support; compare with
   displaced-geometry total-energy finite
   differences and translation/rotation sum rules. Add other SCF backends and
   relaxed post-HF derivatives as separately validated capabilities. Symmetry
   transport can be shared throughout; derivative formulas remain method-owned.

Orbit-count reductions describe planner cardinalities. They do not establish
CCSD(T) energy accuracy, memory feasibility or speedup. A production acceptance
must include the actual data-access and contraction schedule, source/runtime
identity, numerical reference and resource measurements. Keep experimental
CCM/chi capabilities labelled until their method-specific gates are satisfied.

The stored molecular P0 mask now has a complete opt-in NH3/cc-pVDZ RHF
acceptance calculation through `vq`, with identical-mask and unmasked ordered
source controls, arbitrary complex non-Hermitian trial-density covariance,
final evaluated-state closure and an ORCA energy comparison. The
[measured status](symmetry_status.md#complete-screened-molecular-scf-acceptance)
records the source/runtime scope and actual work, timing and memory. This
closes the bounded molecular SCF calculation gap; scalable producer
certificates, method-owned dispatch and each additional source or derivative
remain separate production gates.

## Complex and non-Abelian tensor contracts

Keep four separate admissions: geometry/basis closure, numerical-operator
equivariance, electronic-state invariance and retained-subspace covariance.
For the shared coefficient action `C' = U K^a C`, density and operator laws
are

```text
D' = U K^a(D) U^dagger
F' = U^-dagger K^a(F) U^-1
K^a(U^dagger F_target[D'] U) = F_source[D]
U^dagger S_target U = K^a(S_source)
```

The last line is metric covariance, not Euclidean unitarity of a Cartesian
rotation. No final-matrix symmetrization establishes the third line. Probe
the raw builder on nonsymmetric densities, both spin channels and relevant
complex directions. Full-source tensor selection provides stronger evidence
for that *fixed finite array*, but still cannot authenticate the producer or
its next density-dependent screening decision.

In the new finite-ERI adapter the convention is
`E[p,q,r,s] = (p* q | r* s)`. Its allowed blocks obey
`conj(chi_p) chi_q conj(chi_r) chi_s = 1`; its J and K contractions are
`J[p,q] = sum E[p,q,r,s] D[s,r]` and
`K[p,s] = sum E[p,q,r,s] D[q,r]`. The common conjugate of the entire
selection product is equivalent for the trivial target, but changing only
one leg or using real eightfold ERI symmetry is wrong. For nontrivial
operator/transition targets the variance and target character must both be
declared. Scalar time reversal conjugates the tensor and its basis/labels;
it is not another unitary row of the character table. Magnetic spinor
corepresentations and projective little groups need separate adapters.

A general non-Abelian tensor implementation must provide all of:

1. Full irrep matrices and a common partner gauge. Express a space as
   `direct_sum_alpha (multiplicity_alpha tensor V_alpha)`. Repeated copies
   admit dense mixing. Character traces alone do not fix the partners.
2. Clebsch-Gordan maps for each ordered, variance-aware product
   `V_alpha^(*) tensor V_beta^(*) = sum_gamma N_ab^gamma V_gamma`, with an
   explicit multiplicity index whenever `N_ab^gamma > 1`. Validate their
   isometry, completeness and intertwining equations against the actual
   representation matrices.
3. Reduced/full tensor conversions with documented normalization, partner
   ordering and duals. Scalar invariant operators have Schur form
   `A_gamma tensor I_dimension_gamma` in a unitary retained metric. A
   partner-degenerate state is a separate requirement for using this form
   for density-dependent Fock matrices.
4. Recoupling maps between `(pq)(rs)`, `(pr)(qs)` and `(ps)(qr)`, including
   conjugation, exchange signs and missing partner reconstruction. CC spin
   adaptation and residual terms cannot simply transpose a reduced array.
5. Resource plans: pair multiplicities are
   `M_gamma = sum_ab m_a m_b N_ab^gamma`. Reduced invariant blocks store
   `sum_gamma M_gamma^2` scalars, versus
   `sum_gamma dimension_gamma M_gamma^2` if all equal partner blocks are
   retained, before the larger unsymmetrized storage. Add CG maps,
   multiplicity gauges, recoupling workspace, indices and full-tensor
   temporaries. Charge both representations when conversion needs them
   simultaneously; benchmark conversion as well as contractions.

The C3v formulas in Hellmann and Gauss are a first oracle for this interface,
not a substitute for it. Keep non-Abelian execution unsupported until these
maps and complete small CC residual comparisons exist.

## MANTA/BUPO and unified Fock interface boundary

The subsequent implemented consumer is `ShellQuartetPlan`, which combines spatial
shell actions with the eight permutations of **real molecular AO** ERIs.
It transports full shell blocks and therefore also admits noncommuting C3v
actions, without implementing non-Abelian reduced products or CC recoupling.
It enumerates every distinct ordered shell quartet once and retains its
representative transporter and stabilizer. Stabilizers constrain the whole
angular tensor; they are not scalar multiplicities to apply to a density.

Admission evaluates every ordered quartet of the finite provider source and
compares it against the reconstructed representative, including stabilizer
checks. This setup cost is explicit. Repeated builds evaluate representatives
only, check their snapshots and contract every required density block.
The complete-source Frobenius residual bounds each J/K reconstruction error
by that residual times the density Frobenius norm, excluding arithmetic
roundoff. This gate does not assume an invariant density or symmetrize a
result. Source changes require new admission; incomplete or incompatible
sources retain an explicit unreduced schedule.

The initial MANTA adapter consumes the P0 libint `ReferenceJKBuilder`
source, its exact shell snapshot and a private reusable tile session. The
provider owns kernels, normalization and its zero-precision screening
policy. Symmetry owns the schedule, four-leg inverse-action transport and
J/K digestion. Calls on one plan are serial because its workspace is private
but reusable. Integral calls, requested primitive quartets, null tiles,
contraction terms and transport terms have separate counts. The bounded
all-shell inventory and initial complete audit are not a scalable
producer-side proof or a replacement for MANTA's production planner.
The complete factory reservation precedes radial/coordinate snapshots,
group construction and the retained native tile session. Stabilizers use
one byte per group operation with eight permutation bits, including when
many operations fix every shell. Logical reservation, retained array
payload and whole-process RSS are distinct measurements.

The optional streaming schedule replaces the complete retained quartet
inventory with unordered shell pairs `X = {(a,b): a <= b}`. For each G-orbit
representative `a` in X, its stabilizer `H_a` acts on possible right pairs.
Within a right-pair orbit `G/H_b`, these are the double cosets
`H_a \ G / H_b`. Their cardinalities depend on
`H_a intersect t H_b t^-1`, as in Davidson (1975), Eq. (11). They are computed
from the actual permutations, never from an Abelian constant multiplicity.

Pair exchange needs another quotient. Different G pair-orbits are ordered;
within one orbit, a transporter taking `a` to `b` is inverted to bring the
exchanged left pair back to `a`, and its right partner is minimized under
`H_a`. This is the inverse-double-coset redundancy discussed after Davidson's
Eq. (17). A separate Burnside census checks the number of quartet orbits:

`N = sum_g (fix_X(g)**2 + fix_X(g*g)) / (2*order)`.

The exchange term uses `g*g`, including for non-Abelian groups; replacing
it with `fix_X(g)` miscounts. Each yielded representative expands to at
most `8*order` ordered members with the full angular stabilizer. Only one
such orbit is live in the iterator. Shared real pure/Cartesian actions
still transport every angular leg; pair indices do not supply partner
functions or non-Abelian reduced tensors.

An ordered digest of representative keys and values replaces the stored
per-quartet digests. It is checked before either completed J/K target is
returned; a changed representative produces no partial result. It can
detect a changed generic callback only on evaluated representatives. As
before, reuse requires an immutable producer; a digest is not a proof of
source equivariance. Every ordered source tile and every representative
stabilizer is still checked during initial admission. Neither the memory
improvement nor geometry closure licenses sampled admission or screening.

Streaming regenerates the index schedule during each build, so its memory
saving can cost time. The stored schedule remains the default. A future
automatic policy must account for setup reuse, memory and measured execution
cost without changing the admitted source or scientific tolerance.

The optional native executor compiles angular transport, permutation indexing
and J/K accumulation together, with one call per representative. It consumes
the existing provider tile and admitted topology; it owns no integral kernel.
Inverse coefficient blocks `W=U^-1` act on all four legs, then real-ERI
permutation axes select the target indices. It accumulates
`J[p,q] += E[p,q,r,s] D[s,r]` and `K[p,s] += E[p,q,r,s] D[q,r]` directly for
arbitrary complex density matrices. Neither Hermiticity nor state invariance
is assumed, and no final matrix projection repairs the result.

Native construction snapshots density and inverse blocks. Tile/member
buffers are borrowed only during a serial call. The kernel validates shapes,
indices, finite values and checked resource arithmetic, records executed
contraction/transport terms and poisons its partial targets after a native
failure. Completed vectors own their storage. Its low-level output is still
only algebra: it cannot authenticate the supplied group, tensor variance,
complete orbit coverage or provider. The Python adapter checks the source,
representative digests and full execution census before returning either
immutable J/K target. Complete admission remains on the established path.

The combined live-plan and native reservation is checked before packing
angular arrays or creating the accumulator. It includes borrowed/owned
density panels, targets/exports and tile scratch. MANTA's engine inventory
remains charged through the source plan. This boundary allows later MANTA
tile providers to reuse the executor only after their own source admission;
new fitting, range-separated or complex kernels are not implicitly enabled.

Independent densities can share this execution through `jk_batch` with
shape `(batch, n, n)`. Each channel retains the same contravariant density
and covariant J/K conventions: `D_b -> U D_b U^dagger`, and
`U^dagger J_b[g.D_b] U = J_b[D_b]`, likewise for K. Channels may be complex,
non-Hermitian, zero or symmetry-breaking. There is no relation assumed
between channels. Each representative is evaluated, checked and rotated
once; its ordered values contract into every channel before the next tile.
For B densities, source calls and transport terms equal one build, while
scalar products and block contractions equal B single builds.

The batch owns its density snapshots before calling the producer, returns
immutable per-channel targets and computes one source reconstruction bound
`epsilon_source * ||D_b||_F` per channel. An invalid channel, nonfinite
arithmetic or late source/digest failure refuses the entire batch. Logical
memory grows with `B*n**2` for densities/targets, not `B*n**4`; all inputs,
snapshots, exports and the live source plan are admitted before allocation.
No channels are silently dropped or split into independent source epochs.

The method consumer owns batch order, spin normalization, exchange scaling,
energy assembly and response interpretation. In an unrestricted HF example,
`J_total = J_alpha + J_beta` and `F_sigma = h + J_total - K_sigma`; no
extra closed-shell factor enters the batch kernel. The work receipt belongs
to the complete batch, so attaching identical copies to each spin and adding
them would overcount source work. Frozen Fock actions must capture the
completed targets, with solver generation/spin annotations kept separate
from the admitted source identity. This is a reusable linear contraction
boundary, not production SCF dispatch or a nuclear-gradient implementation.

Target selection now accepts an immutable per-density tuple of `j`, `k` or
`jk`. Omitting it preserves both outputs for every density. Returned J and K
panels are compact, in input-channel order, with explicit `j_channels` and
`k_channels` mappings. An absent target family has shape `(0,n,n)` and owns
no matrix payload. Every input must request at least one target; no dummy
channel or unrequested contraction is evaluated to complete an output shape.

For an unrestricted build, the caller can supply
`[D_alpha+D_beta, D_alpha, D_beta]` and request `('j','k','k')`. This executes
three contractions per ordered integral instead of the four required by
`[D_alpha,D_beta]` with both outputs. Integral calls, stabilizer handling and
angular transport remain identical. It owns an extra total-density input,
so fewer output panels do not imply a lower total memory reservation. The
method still owns the addition, spin factors and physical Fock assembly.

Count-only admission charges every input density and only the requested
target panels. Native construction checks a minimal one-target-per-density
reservation before inspecting selectors, then checks the exact selected
reservation before payload scans or snapshots. The selector itself is owned
by the accumulator. Error bounds remain indexed by input channel; a compact
output uses the bound at its channel mapping. Source/digest failure or a
nonfinite requested contraction still refuses the whole batch. Omitting an
unused target is an algebraic request, not a screening policy, electronic
selection rule or strengthened source certificate.

Native selected-target execution now resolves permutation-dependent matrix
strides once per member and target family, sharing matrix addresses across
the requested density channels. The scalar loop visits only the selected
family and preserves each output's contribution
order. It needs no permuted tile, added density snapshot or target panel.
Scalar-shell tiles bypass the four-axis loops with direct matrix addresses.
Both-target requests retain paired J/K accumulation. This specializes the
existing contraction algebra; it does not change the admitted source, orbit
schedule, error bound or cost-based dispatch policy.

This is a proposed integration contract, not another integral engine. The
planned MANTA producer owns recurrence, kernel evaluation, quadrature,
screening, CD/RI construction and BUPO hierarchies. Symmetry owns admitted
actions, tensor variance, representative/stabilizer maps and selection
metadata. Fock unification owns source selection, density/spin dispatch,
energy assembly and fallback. The coordinated P0 reference provider and
the [MANTA engine](design/manta_engine.md) and
[scientific contract](design/manta_scientific_contract.md) are now present.
They remain owned by the integral engine workstream; this adapter does not
duplicate their kernels. The operator-contract and operator-consumer files
remain absent here. Their workstream versions were inspected read-only for
the interface names below, which remain proposed integration boundaries.

| Existing/proposed boundary | Symmetry integration |
| --- | --- |
| Proposed `KernelRequest`, `IntegralTileView`, `TensorLayout` | Retain exact shell images, AO order, axes, strides and conjugation. Share small shell actions and variance-aware selection; do not attach scalar orbit weights without stabilizer resolution. |
| Proposed `NumericalSourceContract` and `BuildReceipt` | Bind source, support, screening, auxiliary rank and qualified symmetry-plan identities. The shared `OperatorContract` remains a compact declared projection, not a replacement or authentication token. |
| Proposed `BuildRequest`, `DensityEpochView`, `prepare`/`execute` | Qualify the complete spin/k density epoch and finite schedule. Integral tiles may expire at the next provider call; retained symmetry blocks and completed J/K panels must own their data. |
| Existing molecular `JKBuilder` seam | Future admitted providers feed direct consumers here. The complex-Abelian consumer retains dense-source allowed blocks; the quartet consumer streams tiles after a capped complete source audit. Neither is the production data model. Initial libint-backed P0 capability must not be relabeled as native MANTA kernels. |
| Existing `HermitianOperator`, `FockOperator`, `OperatorMetadata` in the Fock workstream | Method assembly creates a fixed density-epoch operator. Source receipts stay beside the method result; generation/spin/k/symmetry annotations do not establish source symmetry. Never capture a live mutable builder during a solve. |
| Draft `FockBuild`, `FockChannel`, `SourceProvenance` and `OperatorFrame` in the Fock workstream | A method assembles compact J/K results into physical spin/k Focks and its own complete energy before constructing an immutable evaluated state. Preserve separate source, density, energy, frame and solver-role identities. The draft is not imported or a landed dependency here; identity completeness and ownership need independent acceptance. |
| Existing explicit `X`, `S`, `reduced_diagonal` boundary | Retained-space transport must bind the actual metric frame and its column order. Supply `diag(X^dagger F X)` for transformed callback Davidson; do not infer it from an AO diagonal or infer partner alignment from degenerate eigenvectors. |

The evaluated-state boundary must cover energy components/corrections and
operator kind as well as D/F values. A source digest or build ID alone is not
a symmetry certificate. A density batch constructs different operators;
applying one fixed operator to multiple vectors is a different operation.
Complex non-Hermitian transition densities remain valid contraction probes
but cannot become physical Hermitian `FockChannel` states without the
method's additional requirements. Owning immutable results does not establish
metric, occupied/frozen or retained-subspace covariance.

PBC integration additionally retains BIPOLE's physical AC/BD exchange domains
and wider K output support. It must not reuse the refused legacy projector.
Carry actual k/quadrature weights, full AO image gauges and separate density
epochs/output mappings, together with finite domains, screening and Ewald
zero-mode/Madelung conventions. Validate raw unreconstructed J/K/XC before
any Hermitization or symmetry reconstruction. A saved crystal Fock passing
an operator/solver crossing establishes neither source equivariance nor
finite-source convergence. The present molecular executor does not admit
these periodic producers. Snapshot copies must be included in the producer's
memory reservation before any bounded MANTA integration.

The operator consumer inventory deliberately leaves auxiliary metrics,
natural-orbital/PNO selection and non-Hermitian CC Jacobians outside Fock
eigensolver policy. Preserve those distinctions. Non-Hermitian trial J/K
results used here test the linear contraction; they must not be passed to a
Hermitian eigensolver. Conversely, a converged projected eigensolve does not
qualify the discarded space or its symmetry.

MANTA must **provide** a source descriptor binding kernel and conventions,
geometry/radials, scalar/spin type, full finite image supports and weights,
screening rules, approximation thresholds, auxiliary metric/rank or CD pivot
policy, and derivative support. Batch outputs must identify their ordered
legs, conjugations, auxiliary gauge and momentum labels. Return evaluated
and screened shell-quartet/three-center counts, contraction counts, retained
and peak workspace bytes, and timings separated by planning, evaluation,
transport and contraction. Current Python ERI value counts are not native
quartet-evaluation counts.

MANTA must **consume** candidate operations admitted on that source,
compact metric-aware shell actions, representative keys with stabilizers,
selection rules and typed gather/scatter destinations. Only a producer
capable of honoring these requests may skip integral calls. Exposing a
full array and blocking it afterward, as in the complex-Abelian increment,
saves storage and contractions but no integrals. The later quartet consumer
skips representative calls only after its complete finite-source audit.
The same batch interface must serve
fitting, SCF, correlation and response; it must not encode a symmetric SCF
density as a universal premise.

For BUPO, the [2025 paper](https://doi.org/10.1021/acs.jpca.4c07415), Sections
2.4-2.6, distinguishes terminating natural-center multipoles from truncated
translated expansions and builds enclosing spheres around bounded objects.
A geometry symmetry does not imply that a clustering/tie-breaking algorithm
produces an equivariant bubble tree. An admitted action must preserve leaf
membership, expansion centers, radii/extents, child-parent relations,
multipole orders, translations and each near/far decision. Rotations mix
multipole components. Stabilizers act on whole multipole blocks. Unequal
trees require a qualified common representation or rejection to the full
source; silently rebuilding a larger symmetric domain changes the source.
Density screening must obey the same orbit-consistent rule in reduced and
unreduced builds (including any orbit maximum bound). A posteriori averaging
cannot repair a source with unequal thresholds or bubble membership.

For multi-k batches, require Seitz composition and the lattice cocycle,
unreduced integer atom-image shifts, reciprocal wraps, AO-centered Bloch
phases, k-star weights and momentum conservation before any orbit
execution. Preserve distinct CCM union-weight/WSSC and chi finite-character
Hamiltonians, supports and weights. Auxiliary, occupied/virtual and frozen
spaces, PAO/PNO/TNO projectors must intertwine the same admitted actions;
dense occupied mixing can couple several pair domains. Rank cuts crossing
a degenerate multiplet or noncovariant local domains must reject reduction.

The next production gates are therefore producer-side admitted integral
batches and cost-based dispatch, followed by retained factor and correlation
consumers. Gradient and stress receipts remain separate for every source:
response, Pulay, fitting/metric, grid, cutoff/support and strain derivatives
must be supplied and tested at displaced geometries or strained cells.

## Fixed whole-orbit screening of the audited molecular source

`QuartetScreening` adds a density-independent omission policy to the stored
P0 molecular quartet audit. It requires the immutable native reference
producer and tile session. A custom callback's self-declared source identity
cannot justify skipping future reads. Stored plans use retained orbit norms;
streaming plans can build a compact certificate during the same full audit.
Primitive screening precision stays zero. A null tile contributes its observed numerical zero to the finite
source audit, without becoming a certified analytic zero.

For each complete ordered shell orbit O, admission records
`b_O = sqrt(sum_{t in O} ||E_t||_F^2)` and the corresponding reconstruction
residual `r_O`. No dense AO four-index tensor is retained. The stored policy sorts by `(b_O,
representative)` and omits whole orbits while their aggregate Frobenius norm
stays within an explicit `maximum_omitted_norm`. The streaming policy visits
orbits in deterministic pair-stabilizer schedule order and omits each orbit
if adding its norm fits the remaining aggregate budget. It can choose a
different mask and does not claim the stored policy's smallest-first ordering. The mask is fixed before
any trial density is supplied. A per-representative cutoff, average bound
or state-derived permutation mask is not this policy. In particular,
nonunitary Cartesian transport prevents inferring `b_O` from a single
representative and orbit multiplicity.

Let M denote this complete-orbit mask. The screened reference is the actual
ordered source `M E`; the executed tensor reconstructs only retained orbits.
Its difference from the unscreened source has disjoint retained and omitted
supports, hence the exact-arithmetic bound is

```text
B_tensor = hypot(sqrt(sum_retained r_O^2), sqrt(sum_omitted b_O^2))
||J_exec[D] - J_E[D]||_F <= B_tensor ||D||_F
||K_exec[D] - K_E[D]||_F <= B_tensor ||D||_F
```

The bound against the identical screened source uses only retained residuals.
Floating-point norm, transport and contraction roundoff are separate. The
API does not change the source acceptance tolerance or pick an automatic
screening tolerance. The method must accept the resulting density-specific
matrix bounds and form its own spin-weighted Fock and energy bounds.

For coefficient action U, `D' = U D U^dagger`, while
`F' = U^-dagger F U^-1`. Complete-orbit masks preserve the same tensor
equivariance as the audited source within its measured residuals. Tests
therefore compare `U^dagger F[D'] U` with `F[D]` on arbitrary complex,
non-Hermitian trial densities. No electronic-state projection, matrix
Hermitization or Euclidean-unitarity assumption enters this admission.
Electronic-state and retained-subspace symmetry still require separate tests.

The certificate binds the producer, full ordered-source bytes, group and
inverse shell actions, audit tolerance, actual mask and deterministic policy.
Different masks identify different numerical sources; policies choosing the
same mask have the same screened-source identity but different policy IDs.
Result bounds follow input density order, with existing explicit compact
J/K output maps. The JSON-compatible receipt excludes live source objects
and densities. A Fock adapter may copy the fixed receipt into source
provenance, keeping density-specific bounds beside the evaluated build.
Neither record certifies an electronic state or a solver's retained frame.

The streaming certificate retains one bit per orbit and one digest over the
retained representatives. Each later build skips omitted representatives
before calling the producer, reconstructs retained members, then checks that
digest before exporting any target. The audit identity binds all ordered
source bytes, shell actions, group multiplication, tolerance and schedule;
it is independent of the requested omission budget. Equal masks within that
audit keep the same source identity when the cap changes. Stored and streamed
audit identities are distinct; no cross-schedule identity equivalence is
claimed. An explicit `screening=None` still builds the full audited source.

The live schedule retains `O(|G| n_shells**2)` index arrays plus at most one
quartet orbit. The additional mask uses `ceil(N_orbits/8)` bytes, so its
worst-case asymptotic storage is still quartic in shell count. There are no
retained per-quartet member lists, norms or hashes, and no global sort
workspace. Count-only admission includes both mask construction buffers and
the attached certificate even when a later build requests no screening.
The complete ordered audit is still mandatory. This increment reduces
retained metadata; it does not supply scalable producer-side prescreening.

MANTA's current `IntegralProvider` source identity and evaluate/evaluate_batch
seam does not supply a general rigorous omission certificate. Its narrower
extent and BUPO ledgers cannot be relabeled as one. The eventual producer
certificate must cover the actual kernel, ordered tensor legs, immutable
finite domain/weights, primitive and density screening, auxiliary metric,
rank/pivot policy and operation-compatible bound closure. Symmetry consumes
those bounds and supplies representatives, stabilizers and mask closure;
MANTA owns their computation, screening implementation and bubble hierarchy.
For density-dependent screening, both reduced and unreduced builds must use
the identical policy and transform its density-dependent bounds covariantly.

This implementation admits real molecular ERIs with complex densities.
It does not admit complex integral sources, antiunitary operations, RI/CD
factors, local projectors, periodic/BIPOLE/CCM/chi sources or nuclear
derivatives. Existing Seitz/image/phase/k-weight/momentum and finite-domain
requirements remain prerequisites for any periodic extension. In particular,
BIPOLE physical AC/BD and wider exchange support cannot inherit a molecular
mask. Reusing a certificate at a displaced geometry or on a different audit
is refused. Gradients and stress need their own response, Pulay, fitting
and support/mask derivative acceptance. Production routing remains disabled.

## Producer bounds before integral admission

`MolecularCoulombBounds` binds MANTA's existing absolute-product envelopes to
an exact immutable molecular P0 source and basis snapshot. It calls the native
pair-envelope producer, owns immutable Q/V arrays, and retains source and
normalization identities. It implements no integral recurrence, Gaussian
envelope, bubble hierarchy or alternative BUPO engine. The envelope producer
owns the mathematical Q/V bounds and their numerical evaluation.

For every AO product in shell pair AB, MANTA supplies
`Q_AB >= integral |phi_a phi_b|` and
`V_AB >= sup_x integral |phi_a(r) phi_b(r)| / |x-r| dr`. Therefore

```text
|(ab|cd)| <= min(Q_AB V_CD, Q_CD V_AB)
b_ABCD = sqrt(n_A n_B n_C n_D) min(Q_AB V_CD, Q_CD V_AB)
b_orbit = sqrt(sum_members b_ABCD^2)
```

The adapter takes outward positive products and combines every ordered member
bound, including nonunitary Cartesian AO actions. Positive underflow never
becomes a certified zero; overflow refuses admission. These are analytic
full-Coulomb integral bounds with floating-point headroom, not interval bounds
on libint arithmetic. An extent's chosen radius is not a finite support cutoff
here: only full-space mass/potential bounds are consumed. The current adapter
admits only real molecular, zero-derivative, zero-primitive-screening P0.

An explicit `bounds=` argument on spatial streaming qualification lets the
fixed schedule-order policy omit an entire orbit before reading any of its
integrals when adding its bound fits the tensor omission budget. Every retained
member and stabilizer is still numerically audited with the original source
tolerance. Bound queries, bound construction, skipped tiles and actual integral
calls are reported separately. No state density enters this decision.

A screening bound is not an equivariance certificate. The resulting plan
qualifies the retained tensor, and its omission norm is an upper bound rather
than a measured norm. The same Fock/energy contraction bounds apply, excluding
arithmetic roundoff. An unmasked build from this partial audit is refused;
callers must use the unreduced producer or perform explicit full requalification.
A changed basis, source, kernel or support cannot reuse the bound certificate.
Stored and completely audited streaming paths remain unchanged fallbacks.

This avoids a complete integral audit only when the envelopes can omit orbits.
It still visits every ordered member's bound, retains the compact orbit mask,
and audits every retained member. Tighter kernel-specific or hierarchical bounds,
producer-certified equivariance, default dispatch, periodic finite supports,
fitting/rank cuts, local projectors and nuclear derivative/stress admission
remain separate work. The shared FockBuild schema is unchanged; the method
receipt explicitly labels `retained-orbits` qualification and the producer-bound
error domain. A molecular certificate cannot qualify distinct CCM/chi sources.

## Hierarchical producer-bound candidates

`MolecularHierarchicalBounds` consumes MANTA's existing `BubbleTree`, native
unordered `BubbleTraversal` and `BubbleErrorModel(order=0)`. It implements no
clustering, radial envelope, multipole kernel or BUPO digestion. Every node
carries a uniform envelope over all descendant AO products. For disjoint far
nodes A and B the entry bound is

```text
min(Q_A V_B, Q_B V_A, Q_A |T_00| Q_B + epsilon_AB)
```

Here `epsilon_AB` is MANTA's complete compact-remainder, Gaussian-penetration
and full-moment-tail ledger. The interaction estimates a magnitude only;
no approximate integral replaces a retained integral. Near/always-near nodes
use Q/V, and a native radial overflow leaves Q/V available. Geometry alone,
the compact remainder alone, or a pointwise extent threshold is insufficient.
Analytic approximation bounds remain separate from arithmetic roundoff.

For a node define `W_A = sum_pairs (2-delta_ab) n_a n_b`. Its self block
contains `W_A^2` ordered AO entries and a disjoint A/B block contains
`2 W_A W_B`. The node's uniform entry bound times the square root of this
integer count gives a full tensor bound. The native traversal supplies a
disjoint exhaustive pair partition; accepted candidate blocks cumulatively
spend the fixed tensor budget. Rejected far candidates descend using MANTA's
traversal, eventually retaining a near block.

The native tree is deliberately not assumed equivariant. The symmetry plan
omits an orbit only when every ordered member is covered by accepted blocks.
This intersection can discard some candidates; it never completes an orbit
by silently adding uncovered members. Per-member bounds retain Cartesian
component sizes and the usual aggregate cap. All retained members/stabilizers
still undergo numerical source qualification. A partial plan requires its
exact fixed mask and preserves the unreduced/full-audit fallbacks.

The adapter stores accepted node pairs and canonical-pair ancestor chains.
An orbit-local cache reuses covered bounds only for identical unordered
real-ERI shell-pair pairs. It does not identify spatially related members,
whose bounds can differ. The cache holds at most 128 entries, then falls
back to actual coverage queries; it is discarded after each orbit. Every
member still contributes its bound in the original accumulation order.
Empty candidate sets need no ancestor-map lookup. Numerical candidates,
whole-orbit masks, upper rounding and source identities are unchanged.
Receipts distinguish native node-bound/error calls, membership lookups,
integral reads and contractions. Native construction stays bounded; worst-case
candidate storage remains quadratic in shell-pair count, and orbit iteration
still occurs. No subquartic planning or production speedup follows from using
a tree. Tighter candidates may reduce numerical work, but dense systems need
measured cost-aware dispatch. The molecular full-Coulomb P0 source, shared
FockBuild schema and all separate periodic/retained-space/derivative gates
remain unchanged.

## Molecular native-host construction increment

The next gap after the fixed-source screened SCF experiment was consumer
integration. The quartet algebra already transported full shell tensors and
contracted arbitrary densities, but a standalone SCF diagnostic did not make
that source usable by the native method drivers or the common evaluated-state
object. `symmetry_molecular_hf` now supplies those two explicit boundaries.

| Literature-to-code capability | Existing foundation | Current consumer and remaining gap |
| --- | --- | --- |
| Davidson representatives/stabilizers and Dupuis-King shell transport | P0 qualification, stored or streaming orbits, selected density channels, producer-bound omissions | Real native RHF/UHF `JKBuilder` consumes fixed masks; producer equivariance/support attestations and cost-aware default dispatch remain missing |
| Pitzer/SALC and non-Abelian partner representations | Metric-aware Cartesian/pure AO actions and full-tensor C3v reconstruction | No irreducible-product CC tensor storage, multiplicity recoupling or reduced/full CC conversion is inferred from HF admission |
| Complex Abelian tensor variance | Complex-density quartet and finite-ERI block contractions | Real native SCF bridge explicitly refuses complex D; complex integral and antiunitary method admissions remain separate |
| Cholesky/symmetry analytic gradients and periodic symmetry literature | Source, state and retained-space contracts are distinct | RI/CD ranks, local projectors, periodic finite supports, gradients and stress still need method-specific source and derivative acceptance |

`ScreenedMolecularJK` never turns a geometry label into a numerical-source
certificate. It consumes the existing exact native P0 producer through
`QuartetScreening`, with a complete audit or producer-bound omissions and a
retained-member audit. Stored and streaming schedules retain that admission.
The mask is fixed over SCF iterations;
method convergence cannot loosen it. Native RHF uses one fused J/K traversal.
Native UHF consumes the selected `(J[Dtotal], K[Da], K[Db])` batch through
`JKBuilder.build_jk_uhf` for NOITER, ordinary iterations and the terminal
rebuild. The explicit HF state adapter consumes the same selected batch.
The generic joint-build default preserves the original J0/K0/K1 cache slots
and call order. Stateless stability energies and Hessian response keep their
separate calls. No new integral recurrence or bubble hierarchy is
implemented here; producer bounds/support and their closure belong to MANTA.

The method adapter forms `F = H + J[D] - K[D]/2` for RHF and
`Fspin = H + J[Da + Db] - K[Dspin]` for UHF. Its energy uses total density for
one-electron/Hartree terms and the appropriate closed-shell or spin exchange
factor. Owned H/D and explicit S/X frames feed the shared `FockBuild`; physical
and solver-modified operators retain their existing distinction. Per-density
matrix/energy bounds are separate from immutable source provenance. This is
an explicit molecular integration, not a migration of every native SCF result
or output path to `FockBuild`, and not default runner routing.

A method may supply a rectangular retained frame, but doing so does not prove
that the group preserves that space. No new orbital truncation or symmetry
projection occurs. Multi-k Seitz/image/wrap/phase/weight and momentum contracts,
CCM versus chi finite Hamiltonians, auxiliary/local projector covariance,
response, Pulay, fitting and support derivatives are unchanged acceptance
requirements for their respective consumers.

The joint UHF boundary returns separate physical J, K-alpha and K-beta
matrices, leaving the method's spin factors, energy, one-center corrections
and shared FockBuild assembly intact. An optimized producer must admit all
channel buffers and independent density-dependent bounds before work. It may
reuse integrals or auxiliary factors, but must preserve the actual source and
screening policy for each requested channel. MANTA's existing implementations
inherit the unchanged sequential default until they supply such an override;
this consumer does not implement another integral engine. The symmetry adapter
uses its immutable whole-orbit mask for every channel and accepts independent,
noninvariant spin densities. Batching certifies neither electronic-state nor
retained-subspace symmetry. No periodic finite source or derivative is admitted
through this molecular real-AO interface.

## Cost routing within one qualified molecular source

The next production increment adds an ordered fallback for the exact fixed
mask and an opt-in measured route choice. It does not add a new symmetry
identity. Davidson stabilizers and Dupuis-King full-shell reconstruction
already supply the reduced candidate. MolSym's RHF supplement Table S1
motivates measuring execution: fewer representatives need not mean less time.
The ordered candidate evaluates every retained ordered tile through the same
MANTA P0 session and contracts the same selected density channels. It skips
omitted tiles and applies no angular reconstruction or density projection.

Producer qualification remains deliberately narrow: exact immutable native
ReferenceJKBuilder construction fixes real, molecular, full-Coulomb,
derivative-zero, primitive-precision-zero support and shell normalization.
The adapter checks each tile's provider, kernel, range parameter and identity.
The qualified basis and complete or retained-only numerical audit bind the
mask. A copied source string, geometry group or caller declaration cannot
qualify another producer. The new qualification record describes this
existing evidence; it is not a producer-issued algebraic equivariance seal
and does not eliminate the retained quartet audit.

For each requested channel signature, calibration contracts fixed arbitrary
nonsymmetric trial densities through both routes, alternating their order.
The discrepancy must satisfy the retained reconstruction residual times the
density norm plus the unchanged source tolerance. The omission allowance
cannot excuse cross-route disagreement because both paths omit the same
entries. Calibration is separate from physical build counts and records all
integral, contraction, transport, logical memory and timing costs. At most
five native HF signatures and five paired probes per signature are retained. Unmeasured signatures,
no measured advantage and insufficient expected reuse choose ordered
execution. Native reconstruction wins only if the fastest ordered sample
minus the slowest reduced sample, times expected remaining builds, exceeds
the entire measured calibration time. This conservative heuristic is not a
future timing guarantee. Already-paid qualification cost is reported
separately; pre-qualification dispatch remains missing.

Execution decisions live in MolecularJKReceipt, outside the fixed screening
receipt and FockBuild source. Numerical contraction order can change FockBuild
bytes and hence its identity; source/policy identity and physical agreement,
not forced equality of state hashes, certify compatibility. The user still
selects the source, omission cap, frame and scientific tolerances. No public
method default, complex-source SCF, RI/CD, BUPO, periodic, CCM/chi, retained
projector, CC residual or derivative admission follows from this routing.
MANTA must eventually supply versioned immutable kernel/domain/screening and
auxiliary-rank descriptors plus operation-specific support/equivariance
attestations. Its bounds and bubble hierarchy remain producer-owned. Periodic
admission must retain complete Seitz/image/wrap/phase/weight/momentum data and
the distinct finite Hamiltonians before any cost comparison is meaningful.

## Source-owned molecular method entry and output state

The explicit public HF entry closes the gap between a callable JKBuilder and
a usable method result. It constructs and qualifies the actual P0 source,
checks basis centers against the molecule, validates overlap/Hcore covariance,
and invokes the existing native RHF/UHF loop. Exact method-local J/K snapshots
now supply the returned evaluated density in the shared FockBuild when their
source, channel order and density bytes match. A miss uses a terminal cold build. No solver rewrite,
state symmetrization, different finite support or implicit engine substitution
is involved. Capture work and any fallback terminal contraction are reported
separately. Separate native RHF J and K calls are joined only on the same
source and exact density bytes; UHF retains its complete spin-channel order.

At an iteration cap the native host's next-step orbitals can describe a
different density from the one whose energy was evaluated. The wrapper owns
physical D/F/energy and diagonalizes `X.T S D S X` only to obtain an exact
output representation with occupations. It preserves fractional densities;
`diag(C.T F C)` supplies honestly labeled Rayleigh energies. Reconstruction
and metric checks establish output consistency, not state stationarity,
canonical eigenpairs or symmetry of a truncated/degenerate retained space.
The actual retained-frame bytes enter the existing frame identity, while
source, density and build identities remain distinct.

Fock unification supplies the existing run_fields/QVF seam and method-declared
orbital labels. MANTA supplies the exact source, envelopes and hierarchy;
this entry adds no producer kernels or algebraic attestations. PBC methods,
CCM/chi Hamiltonians, fitted/complex sources, local/auxiliary projector
transport, correlation and analytic derivatives remain outside this admission.
Zero-copy native evaluated-state capture and pre-qualification cost selection
remain production work. Method-local exact J/K reuse already avoids the
additional terminal build in the admitted cases, while retaining an explicit
cold fallback and independent FockBuild ownership.

Explicit prepared sources now amortize the existing qualification over
independent molecular states. The preparation owns the native source and
fixed mask, immutable one-electron matrices and exact geometry/basis signature;
it contains no electronic or retained-space state. Each method rebuilds its
own retained frame and creates fresh physical counters. Only five bounded
scalar route profiles may be reused. Preparation, newly measured routes and
physical execution have separate costs, avoiding fictitious savings from
omitted setup accounting. A process-local lease protects the shared native
workspace; changed nuclear/radial content refuses before integral work.
Reusable evidence is not an algebraic equivariance attestation for an unseen
source. That producer contract, and cost admission before a fresh audit,
remain necessary for scalable default routing.


The standard molecular job now accepts that explicit source request and
retains physical FockBuild provenance in its normal outputs. An executable
QVF carries operations, error policy, resource bounds and cost policy, never
an audit seal or process-local prepared handle. It reconstructs and qualifies
its own producer before execution. This makes the existing measured route
usable without moving MANTA source authority into the job parser. Unsupported
method compositions and unqualified legacy retries refuse.

Output consumers must distinguish density-defining natural orbitals from
canonical Fock eigenvectors. Occupation order does not define a spectral gap:
Rayleigh quotient tables are labeled, canonical frontier diagnostics are
suppressed, and symbolic frontier cube requests refuse. This is an output
representation correction, not a new electronic-state symmetry assumption.
QVF coordinate unit round trips can change floating-point bytes: a fresh
source identity is expected, and validation must use the reloaded numerical
source. The MANTA producer contract and all periodic domain/Seitz/momentum
admissions remain independent of this molecular interface increment.


The cost-routing boundary now distinguishes admission of a numerical source
for ordered execution from admission of angular reconstruction. A fixed mask
may be constructed without integral reads from an exact immutable molecular
P0 producer and native full-space omission bounds. It authorizes evaluation
of every retained ordered tile only. The existing retained-member/stabilizer
audit remains required for representative reconstruction. Geometry, an
asymmetric bubble hierarchy or a finite-domain certificate alone supplies
neither that audit nor producer-issued equivariance authority.

This distinction enables an explicit ordered route and an early auto-routing
cost proof for short expected use: when the number of expected calls cannot
repay even the ordered halves of paired calibration, neither calibration nor
the angular audit is performed. The mask, scalar tolerance and state remain
unchanged. Source identity binds producer plus actual mask independently of
audit evidence and cost policy; the evaluated Fock object preserves that
separation. Longer-use auto routing still measures both qualified executors.
Production default routing still needs producer operation/support/roundoff
attestation for cheap reduced admission. MANTA owns that producer contract;
this consumer adds no kernels, extents or bubble implementation.

The molecular fitted-source increment now consumes MANTA's exact retained
metric factor and one owned three-center snapshot. Complete complex-Abelian
AO and auxiliary spaces are adapted without changing the producer rank cut.
Selection acts on conjugated AO and auxiliary characters, with full auxiliary
partner mixing and inverse AO variance retained. Actual allowed three-index
blocks drive arbitrary-density J/K; an unreduced contraction of the identical
selected factors is available. Full-source selection/reconstruction error is
measured before reduced work is permitted. The shared native RHF/UHF bridge
and FockBuild consume this fixed source without asserting electronic-state
invariance. Setup still evaluates all ordered fitting tiles.

MANTA remains responsible for factor export, symmetry-compatible screening,
operation/support and roundoff authority, and bubble bounds. A future retained
factor export can replace the snapshot traversal only when it identifies the
same finite source and metric policy. The consumer must not infer such
permission from the geometry or a periodic domain certificate. No changes
are made to Seitz/image/wrap phases, k-star weights, momentum constraints,
CCM/chi supports or Hamiltonians. RI derivative acceptance still requires
metric/fitting/Pulay/response terms, moving supports and independent displaced
geometry checks; this density-contraction increment supplies none of them.

For ill-conditioned but retained native RI factors, ideal metric
orthonormality must not substitute for the finite producer source. The
auxiliary audit now preserves G = F^dagger M F, checks transported G and
coefficient containment with a QR coordinate solve, and records the defect
from identity separately. A unitary polar correction of the assembled irrep
rotation preserves F F^dagger; it does not whiten F, recut M or repair the
electronic density. Partner relations, leakage, isometry and final tensor
error still pass the original thresholds. This removes false rejections of
large JK-fitting bases without weakening source qualification.

Prepared RI factor routing now uses the same measured payback rule as the
quartet consumer. The choice is between dense and blocked contractions of
identical selected factors. Five bounded per-channel scalar profiles may be
reused across fresh method states; neither probe timing nor executor choice
enters source or evaluated-state identity. Fixed nonsymmetric paired probes
must agree before selection, short-use requests can prove probe repayment
impossible and skip all probe work, and unknown signatures retain dense
execution. Preparation remains separate from physical contraction costs.
This is explicit developer-level routing, not default method/job dispatch
or permission to omit source qualification. MANTA producer work and periodic
finite-domain admission are unchanged.


## RI source admission at the molecular method boundary

The next literature-to-code gap after measured RI routing is integration,
not new tensor mathematics. Melega Eqs. (3)-(5) and Kitsaras/Stopkowicz
preprint Eqs. (16)-(17) already motivate the implemented finite factor identity
and conjugated Abelian products. Their existing source and version limitations
remain as documented above. Davidson/Dupuis-King quartet execution continues
as a separate qualified source; neither the gradient paper nor geometric
characters grant an RI gradient, CC or periodic acceptance claim.

The coherent consumer increment is explicit RHF/UHF and executable job/QVF
admission through the existing prepared source and FockBuild. A new immutable
RI request specifies auxiliary basis, complete complex characters and native
rank controls. Actual one-electron tensors must pass covariant invariance;
source preparation and repeated SCF costs are accounted separately. No solver,
integral kernel, metric factorization or state projection is duplicated.

MANTA continues to own finite integral generation, metric rank and factor
construction. The remaining producer capability is a qualified immutable
factor export, with identity, finite support, rank/cutoff policy and screening
closure evidence, so the consumer can avoid an independent complete coupling
snapshot. BUPO hierarchy and omission bounds must bind that same source; an
AO geometry action or cost profile cannot substitute for this evidence.
The consumer must still test arbitrary trial densities and compare actual
J/K, not only a converged symmetric state. Native bounded block execution and
automatic workload estimates remain separate performance increments.

Fock unification consumes the existing evaluated result, with a distinct RI
source identity and fitting-specific setup receipt. RI and quartet bridges
now share method-local exact-input J/K capture. Its key is the qualified
selected-source identity, channel order and exact density bytes; neither
spatial equivalence nor converged-state symmetry permits a match. Capture
copies and comparisons are reported separately from contractions. Periodic
Parity retains its own finite-domain, Seitz/Bloch, momentum and k-star
qualification: this molecular entry grants no multi-k, CCM or chi admission.
Likewise non-Abelian partner/multiplicity resolution, local/frozen retained
spaces, response, fitting/Pulay/support derivatives and stress still require
their separate contracts and numerical evidence.


### Producer factor ownership before reduced generation

The molecular RI producer now exports the actual retained fitted coefficients
with a native identity and executed-work receipt. This resolves the ownership
gap between MANTA's exact metric factor and the symmetry consumer's independent
raw coupling snapshot, and removes that raw array from default preparation.
It does not resolve the remaining algebraic/roundoff certificate for symmetry
representatives or screened factor generation. The existing ordered source
has no declared finite-precision pair-exchange policy, so the exporter keeps
its complete ordered coupling pass. Treating its geometry as permission to
mirror missing values would silently change the numerical source.

The consumer can compare the native export with the explicit ordered-tile
fallback and independent integrals without changing cutoffs or final source
error limits. Both selected and dense execution consume one identified owned
factor payload. MANTA owns integral evaluation, rank selection and export;
the shared symmetry layer owns spatial/retained-space qualification and complex
block adaptation; FockBuild owns the evaluated method state. Periodic Parity
continues to require its full image/domain/momentum contracts separately.

### Native execution of qualified RI blocks

The next execution increment compiles the selected factor blocks into an
immutable native layout. It adds no source authority: the existing factory
still checks the producer export, retained auxiliary Gram and rank, complex
irrep products and aggregate factor-selection error before compilation.
The selected numerical source identity excludes this execution choice.

The blocked route contracts one auxiliary panel at a time, without a dense
rank-by-AO-by-AO reconstruction or exchange intermediate. It evaluates
`J = sum_A B_A tr(B_A^dagger D)` and `K = sum_A B_A D B_A^dagger` for arbitrary
complex nonsymmetric D. Complete factor blocks may be rectangular and irrep
groups may be empty. AO density/operator transport remains metric-aware and
outside the contraction kernel. No conjugation of D, state projection or
antiunitary operation is introduced.

The native snapshot costs one additional selected-factor copy. Count-only
resource admission precedes descriptor/payload reads, copying and contraction.
Actual complex products and auxiliary-panel contractions are reported; the
existing cost router compares complete AO calls against the same-factor dense
fallback, including transport and native dispatch. The former NumPy blocked
path remains an explicit low-level reference. MANTA still owns factor
generation, bounds, screening and bubble hierarchies. This change does not
remove any source integral calls or qualify periodic/CC/derivative consumers.

### Route calibration follows method demand

Automatic route selection must not spend work on channels the actual solver
never requests. The method now obtains a profile at the first physical
request for each signature; the qualified source and physical density remain
unchanged. The probes are still nonsymmetric and use the identical numerical
source, so an invariant state cannot substitute for operator qualification.
Only successful scalar profiles may be shared by prepared sources.

Native NOITER supplies an exact one-evaluation workload and therefore skips
new probes. This local cost bound must not overwrite a profile or policy for
a future iterative calculation on the same prepared source. An iterative
SCF cap is not a convergence forecast or a general bound on response and
line-search calls. Such estimates and producer operation/screening closure
remain separate production requirements. FockBuild ownership, MANTA integral
and hierarchy ownership, and periodic finite-domain admission are unchanged.

MANTA's fixed-metric periodic coupling-only screening now supplies a selected
coupling identity, an unscreened finite reference identity and fixed-density
J/energy approximation bounds. Its paired forward/adjoint maps use the same
ordered image partition; the metric, rank and finite factor remain fixed.
The current consumer admits Coulomb J and refuses fitted K under a positive
coupling-only screen. See the [producer contract](design/manta_scientific_contract.md).
This is useful producer evidence, but it is not an operation certificate.
Symmetry reuse must additionally bind each admitted operation to the actual
ordered support and selection policy, auxiliary and AO actions, reciprocal
and atom-image transport, and an operation/pair error budget. Integral,
factor, phase and contraction roundoff remain outside the current screen
bound. Representative generation therefore still requires separate closure
and arithmetic evidence; neither the paired map nor its scalar error bound
allows the consumer to omit an orbit audit. The producer owns that evidence
and hierarchy implementation; symmetry consumes it through the shared source
contract without reimplementing screening.

### Complete matrix irreps and reduced invariant maps

The experimental `symmetry_products` layer now resolves ordinary unitary
representations using complete matrix projectors. It requires a complete
inequivalent set of matrix irreps in the admitted group's exact operation
order. It checks group multiplication, unitarity, character orthogonality,
`sum d_gamma**2 = |G|`, integer product multiplicities and conjugate closure.
Nonzero lattice cocycles and antiunitary operations refuse. Complex variance
conjugates the relevant matrix representation; no realification substitutes
for a complex irrep.

Columns have order `(irrep, multiplicity, partner)`. For each irrep, `P00`
selects multiplicity seeds and `Pi0` generates partners. Full intertwining
and orthogonality checks validate the resulting basis. Its identity includes
the chosen multiplicity gauge. Two bases with matching characters therefore
cannot silently exchange reduced arrays.

An invariant map between two resolved spaces has the form
`T = U_left direct_sum(A_gamma tensor I_dgamma) U_right^dagger`.
The factory reads the full supplied tensor, retains a specified partner of
each multiplicity block, reconstructs every partner and checks the complete
Frobenius residual at the unchanged explicit tolerance. It does not average
or Hermitize the source. The residual bounds an arbitrary-input application
by `epsilon * ||x||_F` in these orthonormal coordinates; arithmetic roundoff
remains separate. AO users must carry their metric-aware density/operator
transformations and corresponding norm factors themselves.

Composition executes only `A_gamma B_gamma` in multiplicity spaces, without
an extra irrep-dimension factor or a full intermediate. Its inner gauge must
match exactly; an explicit recoupling map carries different gauges and leg
permutations. Full conversion restores all partners before a four-index
re-sort, as required by Hellmann-Gauss's missing-element discussion. This
bounded reference path retains dense basis matrices and uses dense audits;
their storage and transformation costs cannot be called free. Full CC spin
adaptation/residuals, local/frozen projectors, periodic domains, gradients and
stress are not supplied by this algebra.
