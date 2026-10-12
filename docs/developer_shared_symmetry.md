# Shared symmetry infrastructure

The experimental `vibeqc.symmetry_shared` module separates mathematical group
and space actions from qualification of a numerical operator. The shared
contracts and molecular/periodic consumers described below are implemented.
This module does not enable pair skipping, IBZ construction, or analytic
derivatives. The explicit molecular HF adapter below can supply screened J/K
to the existing native SCF drivers; default runner dispatch is unchanged.
The separate experimental `symmetry_abelian` consumer now performs bounded
finite-ERI block J/K contractions; it does not change production routing.

See the [capability status](symmetry_status.md) for the distinction between
shared infrastructure and legacy reductions, and the
[architecture and production gates](design_shared_symmetry.md) for method
integration requirements. The implementation lives entirely in `vibe-qc`;
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
Frobenius norm is accumulated after scaling the real and imaginary residual
components with real division, including subnormal scales. The same component
scaling is used for relative retained-space and selected-operator leakage.
Each norm in a ratio uses its own component scale; binary exponents are combined
only at the final division. This preserves tiny relative defects beside large
retained components, even when unscaled norms are not representable. Nonzero
ratios that cannot be represented are refused. This preserves tiny nonzero defects and finite large residuals without changing the absolute
tolerance. Nonfinite arithmetic or an unrepresentable final norm is refused.
The
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

### Checking operator leakage through a selection

`audit_selected_operators` links an existing `GroupOperatorEvidence` to a
`SelectedGroupTransportEvidence` that references the exact same group parent.
The operator matrices and selected columns are taken from those immutable
snapshots. A separately constructed, merely identically named parent is
refused. The operator contract is inherited from the supplied matrix audit.

At each space, the audit computes the selected operator coordinates and
relative leakage using

```text
G = Q.conj().T @ Q
G B = Q.conj().T @ A @ Q
leakage = ||A Q - Q B||_F / ||A Q||_F
```

The small Gram solve keeps admitted finite orthonormality error from becoming
spurious leakage. A separate solve and residual apply the same relation to
`A.conj().T`. Checking both directions establishes a *reducing subspace* in
the parent's Euclidean retained coordinates: for a non-Hermitian operator,
invariance under `A` alone does not exclude coupling into the selection from
discarded directions. A zero action has zero leakage; scaled norms preserve
tiny nonzero failures and avoid squared overflow. `leakage_tolerance` is
dimensionless, whereas `covariance_tolerance` is absolute in operator units.

The compressed matrices undergo `audit_group_operators` with the selected
group action, retaining dense mixing, antiunitary conjugation and the
parent's Bloch group-law evidence. For example, once the caller has supplied
`operator_evidence` and `selection_evidence`:

```python
result = audit_selected_operators(
    operator_evidence, selection_evidence,
    covariance_tolerance=1e-10, leakage_tolerance=1e-10,
    probe_identity="method-selected operator subspace", budget=budget,
)
```

`SelectedOperatorEvidence` retains the complete operator parent, selection,
compressed `operator_transport`, and separate `containment` and
`adjoint_containment` records. Every parent, covariance and leakage check must
pass. Pass this wrapper itself when checking a further selection; extracting
only its compressed operator audit would discard the earlier leakage record.
Tests include a symmetric zero compression hiding unit leakage, an invariant
line that fails the adjoint check, tiny and large operator scales, and a
second selection that removes the offending block but preserves the earlier
failure. Molecular core, periodic kinetic and native snapshot Fock cases
exercise the same interface.

Admission precedes payload work and conservatively reserves the retained
parent peaks, projection/Gram workspace and nested covariance audit. Identity
strings, allocator overhead and BLAS workspace remain excluded. These checks
concern a selection within an already retained coordinate space; they do not
bound leakage beyond that original space or accumulated metric error relative
to the AO basis. Operator-source qualification, rank policy, approximation
acceptance and production reduction remain method-owned.

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
operations through the same native group validator and builds compact actions
for pure and Cartesian shells through `l=6`, including mixed bases. Pure shells
use the existing Wigner rotations; Cartesian shells share the native polynomial
substitution kernel with periodic AO/Bloch transport. Explicit budgets admit
metadata, packed blocks and each distinct Cartesian rotation before evaluation.
It validates exact radial contractions and basis origins.
The dense AO matrix builder and compact adapter share radial-channel matching
and pure-shell rotations. Both refuse mismatched radial contractions and match
reordered/repeated channels. The legacy dense API still requires pure shells
beyond s. Cartesian compact actions generally are not Euclidean-unitary:
physical overlap covariance uses `U.conj().T @ S @ U`, and operator pullback
and density pushforward retain their distinct covariant/contravariant rules.
Direct Gaussian values under proper/improper operations and unreduced molecular
overlap, kinetic and nuclear matrices provide independent numerical witnesses.
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
consumer reference gates. Independent acceptance is recorded with the
corresponding tracker change, with build identity and execution evidence
retained outside the product checkout. Existence of a test is not evidence
that a full method or target-sized calculation passed.

Beyond the bounded SR/LR GDF diagnostic below, these interfaces do not supply
production GDF/RI tensor execution, PAO/PNO/TNO
projector transport, reduced MP2/CCSD/triples contractions, production pair
skipping, method-specific finite-source covariance qualification, or
derivatives. Those capabilities require separate source and retained-space
validation. No MgO performance or
8x8x8 feasibility claim follows from these primitives.

## Sources

Dovesi, [DOI 10.1002/qua.560290608](https://doi.org/10.1002/qua.560290608),
Sections 3-4, gives the integral/atom-image transport relations and explains
how finite Coulomb/exchange truncation can break Fock symmetry. Casassa et al.,
[DOI 10.1007/s00214-006-0119-z](https://doi.org/10.1007/s00214-006-0119-z),
Section 2, Eq. (3), describes matrix-valued localized-space actions.
The `shared_symmetry` numerical citation route resolves both existing entries.

## SR/LR GDF retained auxiliary tensor diagnostic

`vibeqc.symmetry_gdf.audit_range_separated_gdf_transport` connects shared
retained-space audits to the existing private
`_build_lpq_range_separated_shared_q` producer. Both independently built
source and target batches must use `retain_fit_state=True`. This is an
experimental library diagnostic. Neither periodic SCF nor CCM invokes it
for pair skipping, and `production_reduction_authorized` remains false.

| Contract | Scope of this adapter |
| --- | --- |
| Producer | Explicit SR/LR image cutoffs, reciprocal sphere, screening and absolute metric threshold; retained source controls must agree |
| Candidate operation | Caller supplies named auxiliary, bra-factor and ket-factor actions; no subgroup or source-support admission is inferred |
| Retained space | Actual full metric eigensystems and strict `lambda > threshold` masks; equal positive ranks, full orthonormal eigenvectors and shared leakage/isometry probes |
| Tensor | `L = lambda^(-1/2) E^dagger T`, with destination/source factor-index actions and one scalar conjugation |
| Reference | Independent full target raw tensor and compact factors, with both original factorizations checked |
| Execution | Reconstruct one supplied batch for comparison; no representative planner or global pair inventory |
| Remaining gates | Finite-source/group admission, state symmetry, group/stabilizer closure across a complete inventory, production fallback and measured cost/accuracy |

For retained source and target eigenvectors `Es`, `Et`, the shared subspace
audit obtains `M = Et^dagger Uaux K^a(Es)`, accounting for admitted finite
orthonormality error through its Gram solve. The compact factor-row map is

```text
W = diag(lambda_target^(-1/2)) M diag(lambda_source^(1/2))
L_target[r,a,b] = sum W[r,s] Ubra[a,m] Uket[b,n] K^a(L_source[s,m,n])
```

The two AO legs are **factor-index actions**, not density pushforward or
operator pullback. In the classic pure-shell Bloch convention the three legs
carry auxiliary `+q`, bra `+k_bra` and ket `-k_ket` atom-image phases. The
caller must preserve its actual tensor convention, integer images and
reciprocal wraps. One pair of AO actions applies to every member of a batch;
separate audits are required when the phases depend on its k pair.
Nonorthogonal AO legs are allowed; whitened auxiliary isometry is measured
separately. Arbitrary dense gauges within a degenerate retained eigenspace
are permitted. Rank disagreement is refused, including at a threshold
crossing. No eigenvalue, projector or tensor is averaged or repaired.

The result retains separate full-metric covariance, retained metric/leakage,
whitened-isometry, source/target factorization, raw-tensor and compact-factor
witnesses. All must pass `passed_probe`. Failure of a parent cannot disappear
because a tensor is zero or lies outside the leaking direction. The immutable
`mixing` and `reconstructed_factors` snapshots are finite diagnostic outputs;
they are not an execution certificate. The caller's `OperatorContract`,
space names and mutable producer fit state are declarations, not authenticated
source or geometry receipts. `source_digest` and `target_digest` bind the
snapshotted numerical batches, including their eigenvector gauges, momentum
arrays and reciprocal vectors. Mapping those reciprocal supports under the
operation remains caller-owned; the fit state does not store the reciprocal
cutoff selector itself. These digests do not authenticate how those arrays
were produced.

The scalar tolerance bounds relative Frobenius residuals except the shared
retained-space metric and orthonormality checks, which use absolute Frobenius
norms. The adapter checks both complete borrowed batches and reserves their
snapshots, aggregate retained outputs and conservative intermediates before
array scans or callbacks. Exact `required_bytes`/`required_work` budgets pass;
one-short limits refuse. Action storage/execution, allocator overhead and
BLAS workspace are separate. There is no all-AO-pair Gram materialization.
These logical budgets are not peak-RSS or fleet scalability measurements.

Tests include independently built finite SR/LR raw tensors and compact
factors for proper/improper operations and scalar time reversal, exact
auxiliary rank loss, complex nonsymmetric trial contractions, dense retained
gauges, nonorthogonal AO legs, failed-parent controls and stabilizer negatives.
This does not accept broader production reduction, a default-Hamiltonian
migration, correlation energies, gradients or the independent CCM star gate.

## Complex Abelian finite-ERI block consumer

`vibeqc.symmetry_abelian.AbelianCharacters` accepts a shared `FiniteGroup`
and a complete complex128 character table with irreps on rows and operations
on columns. It checks distinct unit-modulus characters, the group law,
orthogonality and unique product/conjugate closure. Arbitrary row and
operation order is allowed. Non-Abelian groups, antiunitary grading and
nonzero lattice cocycles are refused. Real two-dimensional combinations of
conjugate irreps are not accepted as one-dimensional characters.
`allows(labels, variance, target=...)` uses +1 for a character and -1 for
its conjugate. The default target is the trivial irrep. This small selection
interface can be shared by producers and future contraction adapters; it
does not qualify a tensor or a source by itself.

`AbelianERIPlan` is an experimental molecular finite-array consumer:

```python
from vibeqc.symmetry_abelian import AbelianCharacters, AbelianERIPlan

# group is a previously admitted ordinary finite Abelian group.
# eri is complete complex128 E[p,q,r,s] = (p* q | r* s), already adapted.
# labels is a contiguous int64 irrep label for each orbital.
irreps = AbelianCharacters(group, characters, budget=budget)
plan = AbelianERIPlan(
    eri, labels, irreps, space=space_identity, contract=operator_contract,
    source_tolerance=1e-10, budget=budget,
)
result = plan.jk(density, budget=budget)
# Spin-summed RHF: F = H + result.j - 0.5 * result.k
```

The tolerance in this example is explicit caller policy, not a new SCF
tolerance. Construction scans every forbidden block and requires its
aggregate Frobenius norm to be at most `source_tolerance`. Failure raises
`ValueError` and leaves the full source available to the caller. Accepted
roundoff-sized entries are omitted without averaging. `omitted_norm` bounds
each J/K Frobenius error by `omitted_norm * norm(density, 'fro')`, exposed as
`result.error_bound`. This bounds the omitted-block contribution in exact
arithmetic, excluding contraction roundoff; it is not an accumulated SCF
energy bound. A large trial density can amplify the omission, and the
consumer must enforce its own final observable tolerance. Character and
source tolerances are separate.

Density may be non-Hermitian, complex, spin-resolved or symmetry-breaking.
All its blocks participate; no diagonal-irrep or invariant-state assumption
is made. With `C` mapping adapted orbitals to AOs, density transforms as
`D_AO = C D C^dagger` and returned operators as `C^dagger F_AO C`, including
Cartesian bases with a nontrivial overlap metric. Metric normalization,
representation construction and electronic-state qualification remain caller
responsibilities. The complete input must be consistent with the ordered
chemist convention; no real-only eightfold ERI permutation is used.

The plan stores immutable blocks and indices, a numerical SHA-256 digest,
the named space and operator contract. These declarations do not
authenticate the producer. The plan's `production_reduction_authorized`
is always false, and its `citation_numerics` route is
`complex_abelian_tensor_blocks`. No automatic citation is emitted by an
unchanged production driver.

`stored_elements`, `dense_elements` and `result.scalar_products` count
actual retained values and scalar multiply-accumulate terms; each stored
ERI participates once in J and once in K. `block_contractions` counts executed
einsum calls, including structurally allowed blocks whose numerical values
happen to vanish. `retained_bytes` counts NumPy array payload (ERI blocks,
labels, indices and character/product tables), excluding Python metadata,
native group inventory, allocator and BLAS overhead. `required_bytes` and
`required_work` are conservative construction admission limits, including
the full borrowed tensor and scan/snapshot workspace. Runtime admission
conservatively retains that construction reservation plus matrix workspace.
It is deliberately not a streamed, large-system execution plan.

The benchmark script prints measurements without writing job artifacts:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_symmetry_abelian.py --molecule ammonia --basis cc-pvdz --repeats 31
```

Run each case in a fresh process for its process peak RSS. The script uses
the independent test fixtures, measures complete source preparation and
plan construction separately, then alternates full and blocked contractions.
It reports full ERI values materialized and explicitly marks native quartet
evaluations as uninstrumented. The full integral call is unchanged. See the
[measured status](symmetry_status.md) for both wins and overhead-dominated
cases, and the [literature assessment](symmetry_literature.md) for the
preprint/journal distinction and remaining CC/derivative requirements.

## Source-audited molecular quartet consumer

`symmetry_quartets.ShellQuartetPlan` adds an experimental streaming consumer
for ordered real AO integral tiles. Its modes are `none` (all ordered
quartets), `permutation` (the eight real-ERI permutations) and `spatial`
(permutations combined with the admitted molecular group). Complex densities
are supported, including non-Hermitian probes and independent spin channels.
Complex integral kernels and complex AO actions require another adapter;
they must not be passed through real eightfold permutation symmetry.

The MANTA factory is lazy and requires its actual native P0 libint provider:

```python
from vibeqc.manta import ReferenceJKBuilder
from vibeqc.symmetry_quartets import manta_quartet_requirements, qualify_manta_quartets

provider = ReferenceJKBuilder(basis)
factory_budget = manta_quartet_requirements(
    molecule, basis, provider, len(rotations), budget=budget,
)
plan = qualify_manta_quartets(
    molecule, basis, rotations, translations, provider=provider,
    source_tolerance=1e-10, budget=factory_budget, mode="spatial",
)
result = plan.jk(complex_density, budget=factory_budget)
```

The factory checks the provider's exact ordered shell snapshot against the
supplied basis, builds existing metric-aware pure/Cartesian actions and
admits the complete finite numerical source. It requires molecular
full-Coulomb real zeroth-derivative tiles, zero primitive screening precision,
and the provider's unchanged immutable source identity. Antiunitary actions,
lattice cocycles, displaced/incompatible sources and noncovariant screening
are refused. A null provider buffer is a recorded numerical zero, not a
certificate of an analytic zero. The complete audit still compares it with
its orbit partners.

`manta_quartet_requirements` uses native extents before shell snapshots or
group/action construction. Its returned `Budget` includes the complete
factory workspace and admits exact-cap construction. One byte or work unit
less is refused before snapshots or native engine construction. This is a
conservative logical reservation, not measured allocation or an RSS cap.
Provider construction has its own MANTA admission and occurs beforehand.

The full audit performs `n_shells**4` tile requests before representative
execution is enabled. It never hides this cost in a build-time speedup.
`qualification_work` records setup calls, requested primitive quartets,
null tiles and integral values received. `result.work` records the same
actual counts for one subsequent build. Cached-array reference providers
report zero native calls. `scalar_products` counts executed J/K terms;
`transport_scalar_products` counts four-leg matrix transport separately.
No symmetry of the density is assumed, so this increment reduces integral
calls, not the full arbitrary-density contraction term count.

Representative values are copied before another provider call and their
digests checked at each build. Every output is owned and read-only. The
native adapter retains one tile session after aggregate admission, so calls
on a plan must be serial. `required_bytes` includes the provider's modeled
peak inventory, shell schedule, transport scratch and target matrices;
`retained_array_bytes` counts only compact shell actions and offsets, not
Python schedule metadata, the provider or process RSS. This is a capped
prototype with a complete shell inventory, not a large-system memory claim.
Stabilizers retain one byte per group operation, with one bit per real-ERI
permutation; iteration produces pairs lazily. Highly nonfaithful shell
actions therefore do not allocate an unbudgeted Python pair for every
stabilizer element. Factory requirements additionally cover snapshots and
group/action construction that are no longer live in a completed plan.

`source_residual_norm` is the aggregate Frobenius difference between the
direct ordered source and its reconstruction. Each J/K omitted contribution
is bounded in exact arithmetic by
`source_residual_norm * norm(density, 'fro')`, returned as `error_bound`.
Arithmetic roundoff and accumulated SCF, CC, gradient or stress errors are
separate. Generic `QuartetSource` callbacks are diagnostic declarations;
only a producer with immutable numerical source state can justify reuse.

Pass `schedule="streaming"` to both `manta_quartet_requirements` and
`qualify_manta_quartets` to select `StreamingShellQuartetPlan`. The default
`schedule="stored"` preserves the earlier implementation. Both use the
same complete source audit, tile adapter, real AO tensor convention and
scientific tolerances. The modes `none`, `permutation` and `spatial` are
available with either schedule; `none` retains the literal source fallback.

The reusable `symmetry_quartets_streaming.PairQuartetSchedule` accepts a
`FiniteGroup` and a strict int64 array of shell permutations, shape
`(order, n_shells)`. It validates index composition and retains unordered
pair maps and transporters, with quadratic shell storage. It qualifies
only index algebra. Angular actions, basis/source identity and numerical
equivariance are admitted separately by the plan. Its iterator yields
`QuartetTopology` objects containing representatives, members and lazy
stabilizers; `len(schedule)` uses an independent Burnside census. It is
reiterable, not an array supporting random quartet access. Callers that
retain yielded objects must budget their own storage.

`plan.orbits.retained_array_bytes` reports the streaming pair-index payload;
`plan.retained_array_bytes` remains the angular-action/offset payload.
The plan retains one aggregate representative digest. A build validates it
at completion before returning results, so a changed later representative
is also refused. Diagnostic callback identity strings do not authenticate
unchanged nonrepresentatives; native immutable-source admission remains
essential. The memory model includes borrowed group inventory, temporary
pair maps, angular scratch and simultaneously live orbit objects. It does
not remove the full audit, bound libint internals by RSS, or reduce the
arbitrary-density contraction count.

`scripts/benchmark_symmetry_quartets.py --molecule ammonia --basis cc-pvdz
--mode spatial --repeats 7` measures the public tile adapter. Run each of
`none`, `permutation` and `spatial` in a fresh process with one requested
OpenMP/BLAS thread. It records full admission separately, constructs an
independent all-ordered-tile oracle, and reports actual native calls,
requested primitive quartets, contraction/transport terms, logical bytes,
array payload and process RSS. It does not benchmark MANTA's native J/K
executor or a complete SCF job. See the [measured status](symmetry_status.md).
Add `--schedule streaming` for the pair-stabilizer schedule. For a separate
Python allocation probe, use `--trace-admission-memory`; tracing changes
admission timing, and its live/peak values omit opaque native allocations.
Never compare traced setup timing against an untraced run.

## Native quartet transport and contraction

Both stored and streaming plans accept `executor="native"` on `jk`.
The default remains `executor="numpy"`. The compiled path consumes the
same real tiles, complete source audit and representative digest checks.
It supports arbitrary complex Hermitian or non-Hermitian trial densities,
with independent spin calls. It does not add complex integral or
antiunitary support to the real eightfold schedule.

```python
from vibeqc.symmetry_quartets_native import native_quartet_requirements

execution_budget = native_quartet_requirements(plan, budget=budget)
result = plan.jk(complex_density, budget=execution_budget, executor="native")
assert result.executor == "native"
```

The returned exact budget includes the already live admitted plan and the
native execution reservation. One byte or work unit less is refused before
angular packing, native construction or source calls. Plan admission is
still a prerequisite, and its factory budget is separate. A build lacking
the new native bindings raises an explicit unavailable-executor error;
the caller can select `executor="numpy"` without changing the source or
scientific tolerance.

The native accumulator snapshots density and packed inverse shell matrices.
It handles all member permutations within each representative call, without
allocating a complete AO ERI tensor or per-member J/K matrices. It reports
scalar products, block contractions and angular transport products
separately; these counts use the same definitions as the NumPy executor.
The provider receipts still supply actual integral and primitive-request
counts. Reordering floating-point sums can change roundoff, which remains
separate from the source reconstruction bound.

The low-level `_SymmetryQuartetAccumulator` is an internal algebra utility,
not a source-admission API. It requires exact aligned C-contiguous int64,
float64 and complex128 arrays; it does not force-cast inputs. Indices and
counts are checked, and native failures prevent exporting partial targets.
Calls retain the GIL and are serial. Only the plan adapter adds complete
source/digest/census admission and immutable Python results. No solver
default changes or periodic/derivative claims follow from this kernel.

Selected targets use one resolved matrix-stride pair per member and target
family. J/K family dispatch happens outside the scalar tile traversal, and
matrix addresses are shared across requested density channels. Each selected
output receives contributions in the same order
as the paired both-target loop; no transpose/permuted tile is allocated.
Scalar-shell tiles use direct matrix addresses without four-axis loops.
Only fixed-size stack metadata is added, covered by the existing reservation.
Tests compare all eight permutations on unequal shell axes with an independent
four-leg einsum and check exact selected/both-target agreement. A nonfinite
requested contribution still poisons the entire result, including already
completed channels. An absent target is never evaluated.

Add `--executor native` to `scripts/benchmark_symmetry_quartets.py` to
measure it with either schedule. The reported aggregate execution bytes
include both live plan and native scratch; they are logical reservations,
not an RSS promise. The complete-source admission timing is unchanged in
meaning and remains separate from repeated-build timing.

## Shared-density quartet batches

Both schedules expose `jk_batch(densities, budget=..., executor=...)`.
Use an aligned C-contiguous complex128 array of shape `(batch, n, n)` with
at least one density. NumPy is the default; native execution is explicit.
Neither executor assumes Hermiticity, occupations, spin balance or an
invariant state. A batch of one retains its leading dimension.

```python
from vibeqc.symmetry_quartets_batch import quartet_batch_requirements

execution_budget = quartet_batch_requirements(
    plan, densities.shape[0], budget=budget, executor="native")
result = plan.jk_batch(densities, budget=execution_budget, executor="native")
j_total = result.j[0] + result.j[1]  # example: alpha and beta densities
```

`result.j` and `result.k` are immutable batch arrays; `error_bounds` gives
one source-error bound per input density, excluding arithmetic roundoff.
`source_identity` names the shared admitted source. `work` counts the one
shared traversal, while `scalar_products` and `block_contractions` sum all
channels. `transport_scalar_products` counts only the reused angular work.
Do not multiply the provider receipt by the number of returned channels.
The method owns channel labels and HF/KS exchange factors.

An optional snapshot crossing exercises the actual separate Fock-workstream
draft without importing its package or native core:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/check_symmetry_fock_contract.py --operator-root "$OPERATOR_CHECKOUT"
```

Set `OPERATOR_CHECKOUT` to the existing checkout containing
`python/vibeqc/fock_state.py` and `python/vibeqc/solvers/operators.py`.
The diagnostic reads those two files into an isolated module namespace,
records their hashes, and writes JSON to stdout. It changes neither checkout.
Complete ordered P0 tiles independently reference the method-assembled UHF
trial Focks/energies for water, ammonia and Cartesian s+d, with full and empty
beta channels, both schedules and both executors. It checks immutable state
ownership across a subsequent build. This is an optional diagnostic against
an explicit draft, not a package dependency or acceptance of its identity,
solver, periodic, converged-state or production-memory semantics.

Count-only reservation includes the live plan and batch inputs, owned
density snapshots, J/K targets and immutable exports. Exact limits succeed;
one-short limits fail before packing, snapshots or source evaluation. A
malformed density, overflow in any channel or changed late representative
returns no batch. Native input snapshots and the NumPy owned density copy
prevent source callbacks from mutating the density epoch during execution.
The low-level native accumulator accepts rank-two or rank-three densities;
its single-channel export is rank two, and the public batch adapter restores
the leading singleton axis. Its output still needs complete source admission.

For measured comparisons use `--densities 2 --batch-policy shared` and
`--densities 2 --batch-policy separate` in fresh benchmark processes with
the same executor, mode and schedule. The separate case sums actual source
and contraction counts from independent calls. Its logical execution budget
describes one active call; benchmark-owned batch buffers and the independent
dense oracle are reported separately and are also reflected in process RSS.
This reuses integral work for spin/response trials without implementing
screening, a response solver or nuclear derivatives.

Use `targets` to omit unneeded contractions and storage:

```python
densities = np.stack([d_alpha+d_beta, d_alpha, d_beta])
targets = ('j', 'k', 'k')
execution_budget = quartet_batch_requirements(
    plan, 3, budget=budget, executor="native", targets=targets)
result = plan.jk_batch(
    densities, budget=execution_budget, executor="native", targets=targets)
assert result.j_channels == (0,) and result.k_channels == (1, 2)
f_alpha = h + result.j[0] - result.k[0]
f_beta = h + result.j[0] - result.k[1]
```

Pass the same immutable selector tuple to admission and execution. Each
entry must be `j`, `k` or `jk`; `None` requests both for every input. Outputs
are compact and follow `j_channels`/`k_channels`, not their raw array index
in the input batch. `error_bounds` always follows input order. J-only or
K-only batches return an empty `(0,n,n)` array for the absent family; no
zero-filled matrix is allocated in its place. Single-target low-level native
exports remain rank two and the public adapter restores the leading axis.

Actual scalar and block counts scale with the requested target count, while
source and angular counts stay fixed for the same admitted plan. Both
executors skip the unused contractions. Selectors are snapshotted, and
malformed selectors, insufficient caps, or a late source failure return no
partial result. All supplied densities are still validated and snapshotted.
The API neither builds `D_alpha+D_beta` for the caller nor inserts HF/KS
spin factors. Explicit both-target calls remain the reference fallback.

The benchmark's `--target-policy j` and `--target-policy k` select one
family for each density; `--target-policy spin --densities 2` constructs
the total/alpha/beta request above. These options require
`--batch-policy shared`. Compare `spin` with `all --densities 2` on the same
executor/schedule. Reports include target-array bytes and mappings. J-only
and K-only measurements do not report a complete interaction energy.
All shared `production_reduction_authorized` flags remain false.

## Source-qualified fixed-orbit screening

Stored spatial P0 plans can explicitly omit complete audited shell orbits:

```python
from vibeqc.symmetry_quartets_screening import QuartetScreening

screen = QuartetScreening(plan, maximum_omitted_norm=tensor_error_budget,
                          budget=budget)
execution_budget = quartet_batch_requirements(
    plan, densities.shape[0], budget=budget, executor="native",
    targets=targets, screening=screen)
result = plan.jk_batch(densities, budget=execution_budget, executor="native",
                       targets=targets, screening=screen)
receipt = screen.receipt
```

`tensor_error_budget` is an explicit AO tensor Frobenius omission budget;
there is no default value or change to the source tolerance. The constructor
uses already measured complete-orbit norms, not another integral traversal.
`quartet_screening_requirements(plan, budget=...)` returns exact construction
limits before sorting or creating the mask. Execution admission includes
the still-live original plan, certificate, sort reservation and ordinary
batch workspace. Reservations are conservative and do not shrink with the
mask. Screening saves integral calls and contractions, not the retained
stored schedule or J/K output shapes. RSS is measured separately.

`screen.keep` is an immutable indexable sequence aligned with `plan.orbits`.
Stored masks use a tuple; compact streamed masks use one packed bit per orbit.
`omitted_representatives`, `omitted_ordered_tiles` and
`omitted_integral_values` describe the fixed mask. `result.work` records only
actual calls, while contraction and transport counters record only executed
work. Qualification work still includes every ordered shell tile.
`result.screening` retains the certificate and `result.source_identity`
identifies its screened numerical source. The unscreened producer identity
is separately retained in the certificate. `screen.receipt` returns a fresh
JSON-compatible record for a method's source provenance.

`result.error_bounds[i]` bounds either requested contraction for input i
against the unscreened numerical source, excluding arithmetic roundoff.
Against the identical masked source, multiply `screen.retained_residual_norm`
by the input density norm. The mask and source ID do not depend on density.
Keep density-specific bounds outside a fixed source identity; methods apply
their own spin/exchange factors when propagating bounds to Fock and energy.
Never index a bound by compact output slot without its channel mapping.

Only the exact immutable native P0 producer/session and a complete spatial
audit are supported. Mutable callbacks, permutation-only plans, foreign
certificates and malformed budgets are refused. A streaming plan must request
its fixed mask during admission, as described below.
An all-omitted mask returns correctly shaped zeros with zero integral and
contraction work. Scientific source tolerances and the unscreened, streaming
and NumPy fallbacks remain unchanged; choosing another source requires
explicit method acceptance. A complete audit is still mandatory before a
mask can be built, so this is not scalable producer-side prescreening.

A compact streaming certificate is created during the original full audit:

```python
plan = qualify_manta_quartets(
    molecule, basis, rotations, translations, provider=provider,
    source_tolerance=source_tolerance, budget=budget, schedule="streaming",
    maximum_omitted_norm=tensor_error_budget)
screen = plan.screening
result = plan.jk_batch(densities, targets=targets, budget=budget,
                       executor="native", screening=screen)
```

Pass the same `maximum_omitted_norm` to `manta_quartet_requirements` for the
count-only factory reservation. The cap is explicit, and is valid only with
`mode="spatial", schedule="streaming"`. Without it, `plan.screening` is
`None`. There is no retroactive streaming mask constructor: choosing another
cap requires another full audit. The streaming policy is deterministic
schedule-order first-fit, not the stored smallest-first policy. Compare each
result against its own identical masked source. Both use the same complete
member norms in the original AO coordinates, including Cartesian actions.

`screen.receipt["mask_payload_bytes"]` reports the packed payload, excluding
Python object overhead; the existing reservations include that overhead.
The pair schedule remains quadratic, but the extra one-bit-per-orbit mask
remains quartic in the worst case. No per-orbit norm, member inventory or
hash is retained. A single retained-representative digest is checked before
any result returns. `screening=None` explicitly executes the full source,
and still reserves the live attached mask. Neither path changes tolerances.

The benchmark adds `--screen-norm-budget VALUE` for stored spatial shared
batches. It constructs independent ordered source J/K before and after the
identical mask, reports both errors and the independently measured omitted
norm, and separates mask construction from full audit and repeated builds.
The optional Fock draft crossing accepts the same flag for 12 stored-plan
cases, including empty beta channels. It carries only the fixed receipt in
`SourceProvenance`; per-density bounds remain separate in its diagnostic.

## Omitting audit reads with producer envelopes

```python
from vibeqc.symmetry_quartets_bounds import MolecularCoulombBounds

bounds = MolecularCoulombBounds(basis, provider, budget=budget)
plan = qualify_manta_quartets(
    molecule, basis, rotations, translations, provider=provider,
    source_tolerance=source_tolerance, budget=budget, schedule="streaming",
    maximum_omitted_norm=tensor_error_budget, bounds=bounds)
screen = plan.screening
result = plan.jk_batch(densities, targets=targets, budget=budget,
                       executor="native", screening=screen)
```

`molecular_coulomb_bounds_requirements` reserves native pair-envelope generation,
Python exports, immutable snapshots and the live provider before bound evaluation.
Pass `bounds=` also to `manta_quartet_requirements`; combined admission includes
the live bound table and all bound queries. Q/V come from MANTA's native absolute
Cartesian polynomial envelopes; there is no caller-declared array or callback
route. The adapter checks the original basis against the producer's snapshot.

`bounds.receipt` reports shell/primitive pairs, envelope evaluations, zero
integral-shell calls during envelope generation, memory/work reservations and
its exact source identity. `plan.bound_queries` counts actual ordered-member
bound requests. `plan.qualification_work` counts only retained integral reads;
adding `screen.omitted_ordered_tiles` gives the full quartet census.
`screen.omitted_norm` is now an upper bound, explicitly identified in its
receipt along with `qualification_scope="retained-orbits"`. No libint roundoff
bound or operator-equivariance proof is inferred from the envelope.

Partial plans require their exact `plan.screening` argument on every batch.
Both `screening=None` and the unmasked `plan.jk` convenience path refuse;
use the unreduced provider or a new full audit as fallback. The existing
`ScreenedMolecularJK` bridge supplies the required certificate automatically.
Changing the cap requires requalification; it never retunes a running SCF.

## Native hierarchical candidate bounds

```python
from vibeqc.symmetry_quartets_hierarchy import MolecularHierarchicalBounds

bounds = MolecularHierarchicalBounds(
    basis, provider, maximum_omitted_norm=tensor_error_budget,
    leaf_capacity=1, opening=0.65, budget=budget)
# Pass bounds and the SAME tensor_error_budget to the streaming factory above.
```

`molecular_hierarchical_bounds_requirements` performs a conservative census
before native envelopes, trees and Python snapshots. The bound object owns
its native tree/error model, immutable arrays and sparse candidate map.
Changing the source or cap requires reconstruction. The native default extent
criteria choose a radius; complete Gaussian-tail terms still enter each far
bound. Neither the radius nor the opening parameter cuts off the source.

`bounds.receipt` reports native node/error queries, tightened node bounds,
accepted candidate blocks, candidate entry count/norm, pair-envelope work,
identities and conservative memory/work. Its payload-byte fields exclude
Python/native object overhead, which is included conservatively in admission.
`plan.bound_queries` counts actual ordered-member coverage requests, with
short-circuit refusal when one member is uncovered; `bound_lookup_probes`
counts ancestor-map probes. `hierarchy_closure_refusals` counts orbits retained
because coverage was not established, including wholly uncovered orbits.
Only `screen.omitted_*` counts actual whole-orbit omission. Candidate coverage
alone is not a source-symmetry certificate.

The completely audited and Q/V-only paths remain explicit alternatives.
Native and NumPy batches plus the molecular HF bridge consume the resulting
fixed mask without a new Fock or solver interface.

## Explicit molecular HF consumer

`vibeqc.symmetry_molecular_hf.ScreenedMolecularJK` implements the existing
native `JKBuilder` interface for an immutable P0 `QuartetScreening`
certificate. Construct a stored mask with `QuartetScreening`, or pass the
streaming plan's inline `plan.screening` certificate. No caller-declared
callback can enter this bridge. Both constructor and each build perform resource
admission before density packing or integral reads.

```python
from vibeqc.symmetry_molecular_hf import ScreenedMolecularJK

jk = ScreenedMolecularJK(screening, budget=budget, executor='native')
result = vq.run_rhf_scf_with_jk(basis, n_electrons, S, Hcore, E_nuc, jk, options)
```

The caller supplies the native AO ordering and the matching one-electron
Hamiltonian, metric and electron count. The builder reduces the admitted
integral source and evaluates arbitrary full AO densities. It does not require
an invariant electronic state or certify a retained orbital subspace.
`build_g_rhf` fuses J and K; zero exact-exchange fraction requests J only.
`build_K_pair` shares one traversal. The current native UHF driver invokes J,
K-alpha and K-beta separately, so its per-iteration work is three traversals.
A method calling `build_channels((Da + Db, Da, Db), targets=('j', 'k', 'k'))`
shares one traversal instead. No density cache or pair-permutation assumption
connects those channels.

This real native interface requires finite `float64` matrices and refuses
complex inputs before any integral work, even if their imaginary part is
zero. Existing complex-density quartet contractions remain available directly.
No final symmetrization repairs a failed source. Range-separated K remains
unsupported; runtime Schwarz tuning and incremental state are inactive and
cannot change the fixed mask. Resetting solver state preserves work counters.

`ScreenedMolecularHF(jk, frame=frame, hcore=H, nuclear_energy=E_nuc)` is the
separate method construction adapter. `evaluate((D,), method='RHF',
generation=...)` uses total closed-shell density; UHF takes `(Da, Db)`.
It returns a `MolecularHFEvaluation` with a shared immutable `FockBuild`,
scalar execution receipt, Fock error bounds and electronic-energy error bound.
The source record binds the certificate, actual H bytes, nuclear energy and
metric/retained-frame identity. Density-specific bounds do not enter that
fixed source identity. H and density inputs are owned snapshots. No solver
shift, DIIS Fock, occupation rule or convergence decision is introduced.

Bounds compare with the audited unmasked source in the original AO
coordinates and exclude arithmetic roundoff. They describe the chosen approximation; the method/user still owns
its scientific acceptance threshold. J/K logical byte admission includes the
bridge buffers. `FockBuild.storage_nbytes` reports additional owned arrays,
not peak RSS or a hard bound on the complete SCF process. `last_build` and
`totals` retain only scalar receipts, including actual shell and primitive
requests, contraction counts, elapsed time and logical byte reservations.
Calls are synchronous; concurrent use of one builder is unsupported.
