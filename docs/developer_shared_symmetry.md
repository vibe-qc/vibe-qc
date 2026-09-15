# Shared symmetry infrastructure

The experimental `vibeqc.symmetry_shared` module separates mathematical group
and space actions from qualification of a numerical operator. It is internal
infrastructure under issue #717. It does not enable a solver, pair skipping,
IBZ construction, or analytic derivatives.

Issue numbers on this page refer to the archived `mpei/vibeqc` monorepo.
The implementation now lives entirely in the separate `vibe-qc` repository;
it requires no sibling viewer, queue or QVF checkout.

## Group and space contracts

`FiniteGroup.from_table` admits a supplied finite multiplication table through
native `vibeqc::SymmetryGroup`. Products act on the right first. It validates
identity, unique two-sided inverses, associativity, and scalar antiunitary
parity. Optional integer rotations and lattice cocycles describe
`g h = T(ell(g,h)) product(g,h)`; their exact product and twisted cocycle laws
are checked without reducing integer images modulo a mesh. This validates the
supplied algebra, not completeness or geometric symmetry.
`from_integer_rotations` constructs the exact rotation quotient and
`conjugacy_classes` uses the admitted multiplication/inverse table. The SALC
character-table consumer uses these shared checks instead of a separate
floating-inverse conjugation implementation. Duplicate or nonclosed rotation
lists are refused; this does not admit fractional translations to that
point-group-only consumer. Geometry, basis
closure and whole-mesh compatibility remain adapter responsibilities.
Conjugacy-class admission includes the retained native group inventory as
well as its own temporary storage.

`BlockSpaceAction` represents a semilinear coefficient action `U K^a`, with
one destination for each block and a small, possibly dense complex matrix
inside each block. `K` is scalar conjugation. A block may represent several
mixed occupied orbitals, rather than one signed orbital. Source and target
`SpaceIdentity` values explicitly name geometry, basis, space and gauge;
these are caller declarations, not authenticated content digests.

The native kernel applies a panel without constructing a dense all-space
matrix. Block-permutation validation uses a linear-size destination lookup;
its admission work scales with the packed blocks and coefficient panel. It makes no assumption that Cartesian AO transformations are
Euclidean-unitary. Its real-inner-product adjoint is `K^a U^dagger`.
Typed matrix operations distinguish:

- Coefficients: `C_target = U conjugate_if_a(C_source)`.
- Density pushforward: `D_target = U conjugate_if_a(D_source) U^dagger`.
- Operator pullback: `F_source = conjugate_if_a(U^dagger F_target U)`.

The last two preserve the real density/operator pairing even for a
nonorthogonal AO representation. They do not substitute `U` for
`U^{-dagger}`. A transport block alone is not an invertibility, metric,
group-representation or retained-subspace certificate.

## Bounded orbit operations

`orbit_of` is the only ordinary constructor for `Orbit`; manually supplied
or replaced orbit records are refused. It constructs one orbit on demand,
checking the supplied action's
multiplication law on that orbit. It retains every operation reaching each
member, the representative stabilizer, and the exact orbit weight. It does
not enumerate all translated pairs. It retains the admitted group, and
scatter/gather require matching operation count and scalar antiunitary grading.
The caller owns the size and cost of callback keys, which must retain stable
equality and hash semantics.

`scatter_orbit` applies the supplied block actions and rejects inconsistent
stabilizer transports. It neither averages the representative nor replaces
matrix transport by a scalar multiplicity. `gather_orbit_adjoint` is the
real-inner-product adjoint of the selected scatter maps: every orbit member
contributes once. It is not an inverse or an orbit average. Tensor index,
conjugation, momentum and approximation contracts remain method-owned.

## Evidence is separate from authorization

`OperatorContract` records an operator identity, source revision, numerical
support, screening policy and convention. `QualificationEvidence` distinguishes
an analytic argument, a finite numerical test and an unestablished relation.
Even a successful test or a recorded analytic argument has
`production_reduction_authorized == False` in this module. Method adapters
must implement their own complete admission gates.

`audit_equivariance` compares `pull(F_target[push(D)])` with `F_source[D]` for
one explicitly named trial density. For an invertible action this tests Fock equivariance
on that density. It does not establish a universal error bound, invariant
state, or symmetry of density-dependent screening. The builder owns its
resource admission and finite-Hamiltonian definition.
The audit validates and snapshots the first returned operator before the
second builder call, so backends may reuse their output workspace without
overwriting the unreduced reference. This snapshot joins the byte admission.
When source and target `SpaceIdentity` values differ, callers must supply a
`target_builder` implementing the declared operator in the target gauge.
The source builder is reused only when both named spaces agree. Matching
matrix dimensions alone does not establish that one formula works in both
gauges. The two callbacks still require method-owned source qualification;
this interface does not identify different finite Hamiltonians as equivalent.

`audit_metric_panels` accepts the common `SpaceAction` protocol and compares
source and transported Gram matrices on supplied coefficient columns. It
preserves scalar antiunitary conjugation. A panel witness does not replace
the existing native frozen/active/virtual orbital-sewing audits.

`audit_subspace_panels` adds a common retained-subspace diagnostic for molecular
compact actions and the native chi Bloch panel adapter. Callers supply two
equal-rank, nonempty panels, their ambient metrics, and separate
`SpaceIdentity` declarations for the retained spaces. Geometry and basis
declarations must match the corresponding ambient action. Retained-column
gauges may differ and may mix the entire subspace. The caller selects the
columns and orthonormalizes them; this interface does not select frozen,
active, PAO, PNO or TNO ranks or thresholds.

Both supplied panels must be metric-orthonormal within `metric_tolerance` in
Frobenius norm, with that tolerance less than one. After transporting the
source panel to `V`, the diagnostic solves the small target Gram system

```text
(C_target^dagger S_target C_target) M = C_target^dagger S_target V
```

The immutable result retains `M` as `mixing`. It reports two distinct
`QualificationEvidence` records: `metric` compares transported and original
Gram matrices, including scalar conjugation for antiunitary actions;
`containment` measures `||V - C_target M||_F / ||V||_F` in coefficient space.
The Gram solve accounts for admitted finite orthonormality error. A zero
transport has zero containment residual but fails the metric witness.
Coefficient leakage cannot vanish through cancellation in a null or
indefinite direction of a supplied metric. Its magnitude depends on the
declared ambient AO gauge; it is invariant under unitary changes of the
retained-column gauges. The scalar antiunitary flag remains explicit:
retained coordinates transform as `M conjugate_if_a(X)`.

Admission precedes array scans and snapshots. Metrics and reference panels
are snapshotted before calling the action, so reused backend workspaces cannot
overwrite the comparison. Work scales as `O(n^2 r + n r^2)` using the already
supplied metrics and small rank-space matrices. No dense ambient action or
projector is constructed. This orchestration budget is separate from retained
action storage, backend execution and BLAS workspace admission.

`SubspaceTransportEvidence` is constructed only by the audit. Its
`passed_probe` requires both numerical relations to pass, while
`production_reduction_authorized` is always false. A passing result does not
authenticate the caller's identities, establish global metric positivity,
test a full group representation, prove operator equivariance or validate a
PNO selection rule. Existing native orbital-sewing and method acceptance gates
remain required. Tests use independent molecular overlap/core-Hamiltonian
orbitals and directly summed periodic Bloch overlaps, including half
translations, time reversal, dense retained-space mixing and mismatched-space
negatives.

### Retained-space group laws

`audit_group_transport` checks the complete inventory of local
`SubspaceTransportEvidence` records against an admitted `FiniteGroup`.
The caller supplies distinct named retained spaces, a destination table
`destinations[g,s]`, and one audited transport `transports[g][s]` per operation
and source space. The destination table must satisfy the identity and
composition laws exactly. Contracts, source/target names, scalar antiunitary
grading and ranks must agree throughout each orbit. Disjoint orbits may have
different ranks.

For products that act with `h` first, the numerical relation is

```text
M[g,h.s] conjugate_if_a(g)(M[h,s]) = omega[g,h,s] M[gh,s]
omega[g,h,s] = exp(-2 pi i k_(gh.s) . ell[g,h])
```

The phase uses the final target's rational `BlochCharacter` and the admitted
group's full integer lattice cocycle. A nonzero cocycle requires explicit
characters, including a zero character at Gamma. Characters canonicalize
reciprocal-integer shifts; integer products and rational reduction precede
floating phase evaluation, preserving even very large lattice images.
The factors must also satisfy the scalar antiunitary cocycle law exactly:

```text
omega[g,h,j.s] omega[gh,j,s]
    = conjugate_if_a(g)(omega[h,j,s]) omega[g,hj,s]
```

The factory-created `GroupTransportEvidence` snapshots destinations and
retains the immutable local witnesses. `passed_probe` requires every local
metric/containment probe plus the maximum identity and composition Frobenius
residuals to pass their stated tolerances. It records the space or operation
triple responsible for each maximum. No phases are fitted, and no mixing is
averaged or repaired. Admission covers native group storage, retained mixing,
destination snapshots, factor storage and conservative control/work
reservations. It excludes identity strings, allocator overhead and BLAS
workspace. Only small retained-space matrices are multiplied.

These are finite numerical group-law diagnostics. Declared characters and
space names do not authenticate a physical mesh, a mean-field snapshot or a
selected orbital space. Operator equivariance, rank selection, approximation
policy and production execution remain separate gates;
`production_reduction_authorized` stays false. Physical tests exercise
noncommuting molecular site permutations on a degenerate core-Hamiltonian
eigenspace, and periodic half translations with time reversal at odd-mesh and
zone-boundary k points using directly summed Bloch overlap metrics.
Inconsistent operation phases and discarded lattice images fail the group
probe even when every individual metric/containment witness passes.

### Binding a native periodic state

`audit_periodic_state_subspace` connects these shared diagnostics to an
immutable native `_PeriodicRestrictedMeanFieldState`, including states built
by the chi RHF snapshot bridge. Callers select `frozen_core`,
`correlated_occupied` or `virtual`; the actual band indices come from the
state's masks at source and target k. Empty selections are refused. The
function first runs native orbital sewing with the caller's options,
inventory and caps. Native cross-mask leakage, transported Roothaan,
energy-intertwining and full-AO/retained-scope gates remain in force.
It then applies native AO transport to the actual selected coefficient panel
and runs the shared metric and relative coefficient-leakage checks. Passing
native sewing does not override a stricter failing shared probe.

The factory-created `PeriodicStateSubspaceEvidence` retains native `sewing`
and shared `transport` evidence, plus source/target indices, band tuples,
exact Bloch characters and immutable operation descriptors. Its `state`
property returns the native immutable owner. Space names bind its numerical
state digest, k index and selected bands, as well as hashes of the supplied
geometry and full ordered AO contraction descriptors. Source contracts bind
the state's numerical payload. `source_character` and `target_character`
come from native doubled mesh addresses, including shifts; callers do not
round Cartesian k coordinates or provide a separate phase declaration.

For a whole-group audit, collect each bridge's `transport` by operation and
source k, use its `target_index` for the destination table, and take the
space names and `source_character` from the identity-operation row. This
retains the same state namespace across all operations. Evidence from a
different immutable state, changed basis or geometry cannot be substituted
under those names. Actual chi RHF tests exercise this path on complete odd
and even meshes, with bond-centred inversion and scalar time reversal.
The odd-mesh half-translation group retains its full lattice phases; dropping
those phases fails. The existing even-mesh half-translation stationarity
failure is still rejected by native sewing before group evidence is built.

The aggregate budget reserves native planning and both execution passes,
state storage, descriptor/hash storage and the shared panel work before
extracting matrix payloads. Native tiny diagnostic limits still apply.
Python allocator overhead and unreported library workspace are excluded.
The operation is snapshotted; geometry and basis are borrowed and must not
be modified concurrently. Observable context changes during construction
are rejected. Mutating later SCF results or returned matrix copies cannot
retarget the retained immutable evidence.

These names bind the supplied numerical state and context; they do not
prove that its F/S/C payload was generated from that geometry/basis or
authenticate its calculation declaration. Source qualification and method
acceptance remain separate. `production_reduction_authorized` stays false.

### Binding the operation inventory

`audit_periodic_state_group` consumes a tuple of state-bridge rows indexed by
operation, with each row covering the complete native mesh in k-index order.
All records must retain the same immutable state owner, geometry/basis
context and mask selection. Each row must use one consistent operation and
the group's scalar antiunitary grading. Missing points, reordered sources,
mixed owners and operations that change with k are refused.

This layer checks the supplied Seitz inventory before testing orbital mixing.
Rotations multiply exactly as integers. The translation relation is

```text
W[g] tau[h] + tau[g] - tau[gh] - ell[g,h] = 0
```

Its residual is measured in fractional-coordinate Euclidean norm, with an
explicit `seitz_tolerance` from zero through `1e-6`. Rational arithmetic on
the supplied binary64 translations precedes the norm comparison, preserving
the full integer cocycle and avoiding cancellation from premature floating
conversion. This tolerance concerns fractional Seitz matching; native AO
geometry checks retain their separate Cartesian tolerances. Operations with
the same rotation and grading must be distinct modulo integer translations.
Scalar time reversal may share the spatial part of a unitary operation.

The table and cocycle are re-admitted natively with the recorded rotations,
so their exact action on the lattice images is also checked. The underlying
`audit_group_transport` then derives names, destinations, characters and
mixing from the bridge inventory. A failing local or matrix group probe
remains a failure; the wrapper does not rerun sewing or repair a result.
Gamma-point orbital panels alone cannot establish these facts: tests show
an incorrect rotation table, duplicated cosets and dropped lattice images
passing panel-only group checks or becoming invisible in phases, then being
rejected by the operation inventory.

`PeriodicStateGroupEvidence` is factory-created and retains `declared_group`,
`group_transport` (whose group uses the recorded rotations), and `bridges`.
It reports the largest fractional Seitz residual and its operation pair,
plus aggregate byte/work admission. The inventory counts retained native
sewing, mixing and descriptors, the common state once, both group owners,
and conservative rational/control and composition work. Caller-owned
strings, allocator overhead and BLAS workspace are excluded. No claim of
crystallographic completeness, physical-source qualification or production
reduction follows from this finite numerical witness.

### Checking a method-selected subspace

`audit_selected_group_transport` takes parent group evidence and one complex
column panel per parent space. The panels express the method's selection in
the parent's retained coordinates and must be Euclidean-orthonormal. It
accepts `GroupTransportEvidence`, `PeriodicStateGroupEvidence`, or a previous
`SelectedGroupTransportEvidence`. Periodic inputs retain the native state
and checked operation inventory through their parent wrapper.

For operation `g` at source `s`, the local relation is

```text
M[g,s] conjugate_if_antiunitary(g)(Q[s]) = Q[g.s] N[g,s]
```

The common panel audit checks metric preservation and relative coefficient
leakage separately. The induced mixing `N` then undergoes the complete group
audit using the parent's destinations, table, contract and full rational
Bloch characters. Ranks must agree within each orbit; independent orbits
may have different nonzero ranks. Dense mixing and complex column gauges
are supported. No eigenvector, cutoff, rank or compatible domain is chosen,
fitted or repaired by this function.

The result retains immutable selection snapshots and automatically names
each selected space using the actual columns, parent space and explicit
`selection_identity`. This string declares the method's policy; the audit
does not authenticate its implementation. `passed_probe` requires both the
parent and the new selected-space audit to pass. Pass the wrapper itself
when applying another selection so earlier failures remain attached, even
when a later cut removes the block that exposed them.

These residuals concern retained coordinates. They are neither a fresh AO
metric/source test nor an accumulated error bound relative to the original
AO space. Physical tests expose a rank-one cut through a noncommuting
molecular degenerate eigenspace and a periodic zone-boundary selection that
loses compatibility with half translation and scalar time reversal.

Byte/work admission includes the parent inventory, column snapshots,
induced mixing and conservative local/group intermediates. Wrapper parents
are conservatively charged at their recorded admission peak. Identity strings,
their serialization, allocator overhead and BLAS workspace are excluded. Method-owned
PAO/PNO/TNO generation, approximation acceptance and production reduction
remain separate; `production_reduction_authorized` stays false.

### Checking retained operators

`audit_group_operators` checks a supplied static operator matrix at every
retained space of `GroupTransportEvidence`, `PeriodicStateGroupEvidence`, or
`SelectedGroupTransportEvidence`. Supply square, C-contiguous `complex128`
arrays in the parent's orthonormal coordinates. Independent orbits may have
different ranks. General complex linear operators are accepted; this audit
does not require or establish Hermiticity.

For operation `g` at source `s`, it measures the absolute Frobenius residual
of the intertwining relation

```text
A[g.s] M[g,s] = M[g,s] conjugate_if_antiunitary(g)(A[s])
```

The tolerance therefore has the operator's units. Dense mixing, distinct
target gauges and antiunitary coordinate conjugation are retained. The
parent already checks the full rational Bloch characters in the group law;
there is no additional character multiplier in this one-operation relation.

Pass a separate `OperatorContract` for the supplied matrix family. It need
not equal the parent's transport contract: an overlap-based transport probe
and a projected core Hamiltonian describe different numerical objects.
`GroupOperatorEvidence` retains immutable matrix snapshots, a numerical
`QualificationEvidence` for each operation/source pair, and the complete
parent wrapper. `passed_probe` requires every local residual and the parent
to pass, even when a scalar operator commutes with a failed representation.

For a method-owned orthonormal panel `C`, a covariant AO operator is supplied
as `C.conj().T @ A_AO @ C`; after selecting columns `Q`, use
`Q.conj().T @ A_retained @ Q`. Tests use directly computed molecular core and
periodic Bloch kinetic integrals, as well as actual native snapshot Fock
matrices in occupied and virtual spaces. Perturbations breaking covariance
fail without changing the parent action.

This checks static projected matrices. It does not establish operator
leakage outside the retained space, authenticate the declared source, or
test a density-dependent builder on arbitrary densities. In particular,
passing a snapshot Fock audit does not qualify its producer or a local
correlation approximation. `production_reduction_authorized` stays false.
Byte/work admission precedes scans and copies and includes the retained
parent inventory, matrix snapshots, numerical records and matrix workspace.
Wrapper parents are conservatively charged at their recorded peak; identity
strings, allocator overhead and BLAS workspace are excluded.

## Initial consumers

The chi finite-torus group builder uses the shared native validator for its
final exact algebra checks. Its existing Seitz/AO residual audits, workspace
and group-count gates, fingerprints, result fields, and scientific acceptance
thresholds remain in place. `ChiAOBlochSpaceAction` provides a diagnostic
adapter to the existing native Bloch transport. It retains the native
whole-mesh, fractional-translation, pure/Cartesian shell and allocation gates.
Its borrowed descriptors are checked by the native kernel on every call;
its declared space identities are not immutable mean-field snapshots.

The molecular consumer in `symmetry_ao.py` admits supplied Cartesian point-group
operations through the same native group validator and builds compact pure-shell
actions using the existing Wigner rotations. It validates exact radial
contractions and basis origins. This initial molecular adapter accepts pure
shells through `l=6`; Cartesian molecular shells are explicitly refused.
The existing dense AO matrix builder and the compact adapter share one
radial-channel matching and Wigner-block implementation. The dense API keeps
its requested dense output, while compact callers transport panels. Both
refuse mismatched radial contractions and match reordered/repeated channels.
This does not change molecular SCF routing. Its numerical reference is the
unreduced molecular integral builder, independent of the periodic producer.

The dormant BIPOLE multipole-kernel adapter follows the same admission
boundary: its legacy shell/cell orbit inventory cannot authorize operator
reconstruction. Nonidentity reconstruction is refused. Checked identity
metadata applies the already-stored unreduced kernel; it does not fill entries
omitted by a reduced producer. Its Python-to-native adapter explicitly converts
stored row-major AO-pair indices to the existing native column-major convention,
with the Python unreduced contraction retained as an independent reference.
These repairs do not qualify or enable the dormant multipole source or change
the active erfc engine.

## Resource and validation boundaries

`Budget` is explicit: native planners check logical simultaneously live
payload and work counts before payload scans, snapshots or output allocation.
Checked integer arithmetic rejects overflow. Kernel output allocations are
transferred to NumPy without a second full panel. Python matrix and orbit
orchestration adds its own live-array reservations. These counts are not
allocator/RSS, external-callback, BLAS-thread or replicated-node guarantees.
Fleet-sized execution still requires the corresponding method's inventory.

`tests/test_symmetry_shared.py` covers exact group and cocycle laws, malformed
inputs, budget refusal, complex mixing, nonorthogonal metrics, scalar time
reversal and adjoints, odd/even stabilizers, source-probe failures, molecular
S/T/V and trial-density Fock covariance, and chi half-translation panel
transport. Existing chi group and native Bloch/sewing tests remain the
consumer reference gates. Test results and build identity are recorded in
the shared-symmetry workstream handover; existence of a test is not evidence
that a full method or target-sized calculation passed.

The initial interfaces do not supply GDF/RI tensor adapters, PAO/PNO/TNO
projector transport, reduced MP2/CCSD/triples contractions, production pair
skipping, source-domain repair for #704, or derivatives. Those capabilities
require separate source and retained-space validation. No MgO performance or
8x8x8 feasibility claim follows from these primitives.

## Sources

Dovesi, [DOI 10.1002/qua.560290608](https://doi.org/10.1002/qua.560290608),
Sections 3-4, gives the integral/atom-image transport relations and explains
how finite Coulomb/exchange truncation can break Fock symmetry. Casassa et al.,
[DOI 10.1007/s00214-006-0119-z](https://doi.org/10.1007/s00214-006-0119-z),
Section 2, Eq. (3), describes matrix-valued localized-space actions.
The `shared_symmetry` numerical citation route resolves both existing entries.
