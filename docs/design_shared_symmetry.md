# Shared symmetry infrastructure: implementation audit and proposal

Status: proposal, 2026-09-06. Audited source: `30ed36af4` on `main`.
This document does not enable a solver or change its numerical acceptance gates.
The immediate use case is chi issue #704 and the subsequent local-correlation
and analytic-gradient work recorded in
`handovers/HANDOVER_AICCM2026DEV_B.md`.

## Recommendation

Develop a shared symmetry module with numerical backend adapters. Keep the
unified Fock-engine refactor after publication, as requested. Symmetry has uses
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
| `symmetry_core.py`, `symmetry_ao.py` | Molecular/periodic atom matching and pure-shell rotations | Match AO ordering, radial contractions, origins and metric conventions; do not presume Cartesian AO rotations are Euclidean-unitary. |
| `cpp/src/periodic_ao_bloch_transport.cpp`, `kmesh_address` primitives | Bounded native Seitz action, full atom-image shifts, Bloch phases, mesh compatibility, scalar time reversal; Cartesian and pure shells through l=6 | One panel action is not an operator or state symmetry certificate. Spinor and magnetic actions require additional representations. |
| `cpp/src/periodic_mean_field_state.cpp`, `cpp/src/periodic_orbital_sewing.cpp` | Immutable numerical reference, metric/Roothaan audits, frozen/active/virtual mask leakage checks | Current sewing explicitly reports physical source symmetry as uncertified. A converged snapshot does not prove arbitrary-density operator equivariance. |
| `symmetry_lattice_c.py`, `pair_resolved_truncation.py` | Atom-pair/image action, geometric pair-distance domains, shell masks and orbit utilities | Lower-level atom matching carries fractional translations, but several callers still filter them out. Audit the actual function, not its historical symmorphic-only docstring. |
| `bipole_symmetry_fock.py`, native domain-aware J/K builders | Full masked builds, separate output/density/internal domains and representative reconstruction | Validate unreconstructed full builds first. Existing high-level mapping filters nonzero fractional translations. Changing domain membership changes the finite approximation. |
| `periodic_k_symmetry.py` | Phase-aware k-star density expansion and inverse folding | It selects a zero-fractional-translation subset; representative k points alone do not reduce every two-electron contraction. Audit current driver wiring separately. |
| `symmetry_integrals_reduced.py`, `symmetry_fock_reduced.py`, `symmetry_salc.py` | Representative one-electron integrals, Gamma J/K prototype, point-group projection ingredients | Gamma, point-group and pure-shell assumptions cannot be silently generalized to multi-k chi. |
| `periodic/chi/symmetry.py` D127-D130 | Finite-torus actions, exact image cocycle/group checks, actual occupied-space action and native snapshot bridge | Dense diagnostic witnesses and count caps are not a scalable production representation or execution plan. |
| `periodic/chi/pno.py` D123/D126 | Compact translation-pair census, even-mesh stabilizers, conditional point-pair quotient | No orbital phases or local-space transport, and no production pair skipping. |
| `periodic/ccm/symmetry.py` | Cluster compatibility and independent small-system AO checks | Explicit supercell matrices do not scale to the target; WSSC/union-weight and chi finite-character Hamiltonians must retain separate operator identities. |

Paths in this table are relative to `python/vibeqc/` unless prefixed by `cpp/`.
Reuse means tests and implementations can be consolidated incrementally; it
does not mean all present contracts are interchangeable.

## Why a single unconditional reduction recipe would be wrong

1. **Geometry, operator and electronic state are different tests.** A crystal
   operation can preserve the geometry while an image cutoff breaks the
   numerical Hamiltonian, as in #704. For a density-dependent Fock builder the
   relevant operator relation is equivariance, `F[g.D] = g.F[D]`, with the
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
Both source PDFs were resolved through the companion library catalog and read.
References: [Dovesi, DOI 10.1002/qua.560290608](https://doi.org/10.1002/qua.560290608),
[Casassa et al., DOI 10.1007/s00214-006-0119-z](https://doi.org/10.1007/s00214-006-0119-z).

## Proposed shared contract

The following are responsibilities, not committed public class names:

* **Group and mesh:** geometry/basis identity, admitted subgroup, Seitz
  composition, integer atom images, reciprocal wraps, antiunitary flags and
  finite-torus compatibility. Preserve full integer shifts until the relevant
  character is evaluated. A fractional translation in a nonprimitive cell
  must not be discarded merely because older helpers call it nonsymmorphic.
* **Space action:** compact atom/shell maps and small rotation blocks, plus
  adapters for AO, auxiliary, occupied, virtual, local-pair and response
  spaces. Transport blocks on demand. Avoid dense all-supercell AO matrices
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

## Delivery sequence and acceptance gates

1. Finish #704 source-domain diagnosis on unreconstructed J/K. Compare radial
   output support with the existing atom-pair distance domain at fixed alpha,
   density and internal domain; then establish support convergence and
   source-matched SCF/derivative behavior. Do not widen a tolerance to pass
   one fixture, and do not call an output-domain experiment a production fix.
2. Establish the shared native group/space interfaces by reusing the existing
   Bloch action and chi group logic. Integrate two real consumers before
   declaring the contract general: chi/BIPOLE plus GDF/RI or a molecular
   consumer. Chi and BIPOLE share a producer and are not independent backend
   evidence by themselves. Keep production switches off until their gates pass.
3. Bind the actual occupied gauge and frozen/active spaces to PAO, auxiliary,
   PNO and TNO representations. Compare representative and unreduced MP2,
   CCSD residuals/energies, and coupled triples on small odd/even meshes.
   Include dense occupied mixing, fractional translations, degenerate retained
   subspaces, negative leakage cases and nonsymmetric trial densities.
4. Replace global factor/tensor construction with bounded local production
   data access. Validate increasing MgO meshes through `vq`, recording source
   and runtime identities, energy components, approximation settings, peak
   memory and actual contraction counts. Run 8x8x8 only after memory and
   small-mesh numerical gates pass.
5. Assemble chi RHF total analytic gradients first using its own declared
   operator/support; compare with displaced-geometry total-energy finite
   differences and translation/rotation sum rules. Add other SCF backends and
   relaxed post-HF derivatives as separately validated capabilities. Symmetry
   transport can be shared throughout; derivative formulas remain method-owned.

The present MgO four-active-band plan has 2,098,176 placed unordered pairs,
4,112 translation representatives and 260 conditional point representatives.
Those counts are covered by planning tests. They do not establish 8x8x8
CCSD(T) energy, memory feasibility or speedup; current global intermediates
are the separate scaling obstruction documented in D123.

## Proposed separate task brief

Develop shared symmetry infrastructure for vibe-qc, starting from the existing
native AO Bloch transport, orbital sewing, atom-pair support and chi finite-torus
group implementations. Own common representations, compact orbit execution and
source/subspace qualification interfaces. Coordinate the #704 domain boundary
and the periodic correlation owner through the existing issue/handover process.
Integrate two source-distinct consumers with unchanged unreduced numerical
references before expanding coverage. Preserve fractional translations,
antiunitary semantics, AO metrics, even-mesh stabilizers and bounded allocations.
Keep operator-specific screening, q=0/finite-Hamiltonian conventions, PNO/TNO
approximations and analytic derivatives in adapters. No full Fock-engine rewrite,
production symmetry enabling, gradient enabling, or MgO performance claim
without the corresponding local and fleet gates. Confirm the registered topic
checkout and ownership before implementation; do not create an ad-hoc clone.
