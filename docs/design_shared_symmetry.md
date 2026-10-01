# Shared symmetry architecture and production-reduction gates

The shared group, space and diagnostic contracts are implemented, with
molecular and periodic consumers. This page develops the rationale of the
original 2026-09-06 design into the remaining method-integration gates.
It does not enable a solver or change numerical acceptance thresholds.
See the [developer reference](developer_shared_symmetry.md) for implemented
APIs, the [capability status](symmetry_status.md) for legacy scope, and the
[chi symmetry reference](experimental/aiccm2026dev_b_symmetry.md) for that
consumer's finite-source diagnostics.

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
| `periodic_k_symmetry.py` | Phase-aware k-star density expansion and inverse folding; star maps restricted to an admitted operation subset | It selects a zero-fractional-translation subset; representative k points alone do not reduce every two-electron contraction. The multi-k GDF exchange wedge admits operations against its executed source; other drivers' wiring needs its own audit. |
| `symmetry_integrals_reduced.py`, `symmetry_fock_reduced.py`, `symmetry_salc.py` | Representative one-electron integrals, Gamma J/K prototype, point-group projection ingredients | Gamma, point-group and pure-shell assumptions cannot be silently generalized to multi-k chi. |
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
