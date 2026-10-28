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

The MANTA factory is lazy and requires an actual immutable native molecular
provider. Full-Coulomb `ReferenceJKBuilder` (libint) and `DirectJKBuilder`
(native MD) enter the same complete numerical audit:

```python
from vibeqc.manta import ReferenceJKBuilder
from vibeqc.symmetry_quartets import manta_quartet_requirements, qualify_manta_quartets

provider = ReferenceJKBuilder(basis)
# DirectJKBuilder(basis) from vibeqc.manta selects the separately audited MD source.
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

The native plan must expose its kernel and range parameter; an older runtime
without that metadata refuses admission. Native MD has distinct source and
Fock provenance. It cannot borrow libint's Q/V or hierarchical omission bounds,
or enter ordered-source-only qualification. Both stored and streaming full
audits support a fixed whole-orbit mask, representative reconstruction and an
ordered fallback from the identical producer. This remains an experimental
low-level adapter; the public prepared-source entry still selects libint.

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

Only an exact immutable native molecular producer/session and a complete spatial
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
counts actual ancestor-map probes. `bound_lookup_cache_hits` counts coverage
queries avoided by reusing identical real-ERI pair labels within one orbit;
subtract it from `bound_queries` to get actual coverage calls.
`bound_lookup_cache_peak_entries` reports peak cache occupancy, capped at 128.
Exhausting the cache falls back to ordinary queries; it cannot omit members.
Empty candidate sets perform no ancestor-map probes. All ordered member
bounds are accumulated in their original order, including cache hits.
`hierarchy_closure_refusals` counts orbits retained
because coverage was not established, including wholly uncovered orbits.
Only `screen.omitted_*` counts actual whole-orbit omission. Candidate coverage
alone is not a source-symmetry certificate.

The cache is discarded after each orbit and reserves 36864 conservative
bytes before auditing. It identifies only permutation/exchange of the same
shell-pair labels, not spatially related members: a generic MANTA tree is not
assumed equivariant. It retains no new all-quartet inventory or integral data.

The completely audited and Q/V-only paths remain explicit alternatives.
Native and NumPy batches plus the molecular HF bridge consume the resulting
fixed mask without a new Fock or solver interface.

### Streaming execution skips omitted image expansion

The compact pair schedule separates representative enumeration from image
expansion. After the exact source-bound mask is validated, `retained(keep)`
expands only retained representatives. Complete qualification still uses the
full iterator and audits all required ordered members and stabilizers; a mask
never becomes an equivariance certificate. Stored schedules keep their existing
execution path, and ordered/NumPy/native streaming consumers share selection.

Image expansion uses the established real-ERI permutation order and constructs
one descriptor per unique image. Duplicate stabilizer images still contribute
their permutation bits, and the orbit-stabilizer census remains mandatory.
No representative/member inventory or evaluated tiles are cached. Persistent
storage remains quadratic in shell count with one orbit live at a time. The
existing conservative memory/work reservations, source digest order and
floating-point contraction order are unchanged.

## Explicit molecular HF consumer

`vibeqc.symmetry_molecular_hf.ScreenedMolecularJK` implements the existing
native `JKBuilder` interface for an immutable molecular `QuartetScreening`
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
`build_K_pair` shares one traversal. `build_jk_uhf(Da, Db)` returns
`(J[Da+Db], K[Da], K[Db])` through the existing selected three-channel batch.
Native UHF uses this joint entry point for NOITER, ordinary iterations and
the terminal rebuild. The generic `JKBuilder` implementation preserves the
original ordered calls to J slot 0, alpha K slot 0 and beta K slot 1, so
incremental builders keep independent caches. No density cache, spin equality
or pair-permutation assumption connects the symmetry adapter's channels.
Hessian response and the stateless stability line search retain their existing
individual calls; UKS and periodic drivers are not migrated by this increment.

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

### Opt-in measured molecular routing

Use `executor='ordered'` for direct evaluation of every retained ordered tile,
or `executor='auto'` for conservative measured routing of the same immutable
mask. Both use MANTA's existing tile session and the native contraction
accumulator. The ordered route has zero angular transport work; it does not
materialize the ERI tensor or add an all-quartet inventory.

```python
jk = ScreenedMolecularJK(screening, budget=budget, executor='auto')
routes = [jk.calibrate_routes(targets=t, expected_builds=30)
          for t in (('j', 'k', 'k'), ('j',), ('k',))]
result = vq.run_uhf_scf_with_jk(basis, n_alpha, n_beta, S, Hcore, E_nuc,
                             jk, options)
```

Calibration is explicit, before SCF, and accepts one to five paired probes
(default two). Reserve its memory and cumulative work before constructing
trial densities. Its immutable receipt contains measured samples, peak
logical reservation, modeled work, numerical discrepancy, selected executor
and producer/audit/source/policy identities. Probes do not update `totals` or
`last_build`. `route_calibrations` retains at most the five native HF signatures
(`j`, `k`, `jk`, paired K, joint UHF); recalibration replaces only that
signature. Unmeasured signatures use ordered execution. The bounded scalar
receipt cache reserves 131072 bytes. Calibrate standalone J/K as well as
joint UHF when native stability/response calls are expected. A failed numerical calibration
publishes no reduced winner. Explicit `native` and `numpy` choices remain.

`last_build.route` reports the actual choice. FockBuild source capture does
not include route diagnostics. The qualification record states whether the
source had a complete audit or producer-bound omissions with a retained
member audit; it is descriptive evidence, not an alternative admission API.
The producer constructor fixes the supported molecular domain, and every
tile also checks full-Coulomb kernel/range metadata. Custom callbacks, other
native providers and periodic domains cannot enter by copying that record.
The lower-level `ordered_quartet_jk` supports complex nonsymmetric densities
against this real source; the native HF bridge still requires real float64.

### Public molecular RHF/UHF entry

`run_rhf` and `run_uhf` accept an explicit `symmetry_reduction` configuration.
The caller supplies complete operations and the fixed source tolerance and
omission cap; a geometry label or Boolean is not a source certificate.

```python
reduction = vq.MolecularSymmetryOptions(
    rotations, translations, source_tolerance=1e-10,
    maximum_omitted_norm=1e-12, budget=budget,
    screening='hierarchical', executor='auto', expected_builds=30,
)
result = vq.run_uhf(molecule, basis, symmetry_reduction=reduction)
physical = result.fock_build
```

This entry constructs the exact immutable MANTA P0 producer, its source-bound
hierarchy (or `screening='audit'`), compact fixed mask and existing native HF
host. `native`, `numpy` and `ordered` remain explicit executor choices.
Automatic routing calibrates each HF channel signature used by the host,
including standalone J/K stability calls. Kernel, fitting or source selection
is never inferred from cost. Source failures raise; they cannot fall back to
a different Hamiltonian or a different omission cap.

The current method scope is all-electron molecular RHF/UHF with HCORE or
supplied same-basis READ densities. When no method options are supplied this
explicit entry selects HCORE; supplied options must select their guess
explicitly. ECP, COSX, legacy density fitting, other selected engines,
projected/file READ, FRAGMO, DFT+U and spin/multi-guess schedules are refused.
Omitting `symmetry_reduction` preserves existing dispatch. `run_job` accepts
the same explicit configuration or prepared source for RHF/UHF single points.
Post-SCF consumers remain outside this admission.

Overlap and Hcore are computed for the actual molecule/basis, then checked
as covariant AO tensors under the admitted inverse coefficient action W:
`W.T @ A @ W = A`. This includes nonorthogonal Cartesian shells. The retained
output frame uses the native host's unscaled-overlap eigenvalue cutoff
convention; reconstructing its density is checked explicitly. It neither
projects the state onto a spatial irrep nor qualifies that retained space
for symmetry reduction.

The method first looks for owned J/K evaluations on the native host's exact
returned density. A source or density mismatch uses an explicitly counted
terminal cold build. `MolecularSymmetryResult` exposes that
owned physical energy/D/F and `fock_build`, with ordinary native convergence
and trace diagnostics forwarded. At a cap or NOITER it does not advance the
density to the next solver step. Natural orbitals and their actual occupations
reconstruct the evaluated density, including fractional READ/damped states;
reported orbital energies are physical-Fock Rayleigh quotients. They are
labeled `natural` in QVF, not advertised as canonical eigenpairs. The UHF
Slater-determinant spin diagnostic is exposed only for idempotent densities.

`result.run_fields` supplies source/frame/build identities and stop-state
metadata through the existing QVF output interface. `symmetry_provenance`
is a fresh JSON-compatible copy containing source and one-electron audits,
calibration, SCF and terminal work, timings, bounds and storage observations.
The budget reserves an additional dense method workspace before source
construction, but is not a hard bound on native SCF DIIS, stability or LAPACK
memory. The method-owned wrapper deliberately offers no native unwrapping
hook: unsupported post-SCF consumers cannot silently assume an unmasked
reference. Analytic RHF/UHF gradients refuse this source until response and
support derivatives receive separate acceptance.

### Reusable prepared molecular source

For repeated states at exactly the same nuclear geometry and basis, prepare
the qualified source explicitly:

```python
source = vq.prepare_molecular_symmetry(
    molecule, basis, symmetry_reduction=reduction,
)
setup = source.preparation_receipt
rhf = vq.run_rhf(molecule, basis, symmetry_reduction=source)
uhf = vq.run_uhf(molecule, basis, symmetry_reduction=source)
```

`PreparedMolecularSymmetry` owns the immutable producer, exact existing audit,
fixed mask and checked S/H matrices. It holds no density, orbitals, Fock,
occupied/virtual spaces or convergence history. Nuclear identities, positions,
atom order and every shell's representation, radial coefficients and exponents
must match before a method reads any tile. Charge and multiplicity may differ:
they select a state of this all-electron Hamiltonian. The native host and
method wrapper continue to validate each state's electron counts and options.

Automatic execution caches at most five scalar calibration receipts, each
bound to this source and its own HF channel signature. RHF and UHF may reuse
their common J/K signatures; a previously unseen signature is calibrated.
Each method gets fresh physical counters. Cached timing is an estimate from
earlier work in the same process, not a guarantee that performance is stable.
Choose a fresh preparation to measure routes again. Explicit native/ordered
execution still avoids calibration altogether.

`preparation_receipt` records the once-paid audit, source and one-electron
costs. A prepared method call reports zero new audit/one-electron work and
only newly performed calibration in `routes`; `reused_routes` names cached
signatures. `prepared_source` references the original preparation receipt.
Count that preparation once when totaling several calculations. Source,
policy, frame, density and build identities retain their existing meanings.

The process-local handle refuses pickling and use after process forking.
One nonblocking lease protects its native tile workspace throughout a method;
overlapping calls refuse and exceptions release the lease. Independent
prepared handles may execute independently. Changing geometry, basis,
operations or screening requires a new preparation. This is reuse of actual
qualification evidence, not bypass of an audit for a fresh source or a new
producer-issued algebraic certificate. It grants no periodic, fitted,
complex-source, local-space or derivative capability.

### Job and QVF source requests

The normal output pipeline can consume this source explicitly:

```python
result = vq.run_job(
    molecule, method='uhf', basis='cc-pvdz',
    symmetry_reduction=reduction, output='qualified',
)
```

Absent method options select HCORE. Supplied options retain their guess and
are copied before runner adjustments. Same-basis READ arrays can be passed
through native method options. Source or method failure never retries through
the legacy SAD/integral path. Ordinary convergence failure still raises and
does not publish a completed QVF. Geometry optimization, Hessians, correlation,
response, alternate engines, fitting, ECP and state schedules need separate
admissions. An automatic method choice that resolves to correlation refuses.

The QVF option `symmetry_reduction` is a JSON record with required `rotations`
(ng by 3 by 3), `translations` (ng by 3), `source_tolerance`,
`maximum_omitted_norm` and `budget` (`maximum_bytes`, `maximum_work`). Optional
`screening`, `executor` and `expected_builds` have the same constructor rules.
Finite numeric types, dimensions, field names and resource reservations are
checked before allocating operation arrays. Source identities, audits and
producer seals cannot be supplied in this record. Use
`vibeqc.symmetry_molecular_job.source_request(reduction)` to create it for
`write_pending_qvf(..., options={'symmetry_reduction': request})`; method
controls can be supplied through the existing `rhf_options`/`uhf_options`
records, with an explicit supported guess.

Execution always constructs and qualifies a fresh source. A completed QVF
records the configuration even when the in-process calculation used a prepared
handle; it never serializes a live producer or grants authority to a saved
audit. As with other completed job recipes, this source record does not encode
all solver controls or a READ starting density. Use a pending container with
explicit method controls for a prescribed execution; the submitted spec is
preserved verbatim. Evaluated D/F/energy and source/frame/state identities
remain output data, distinct from the requested calculation.

Natural-orbital tables use actual occupations and identify physical-Fock
Rayleigh quotients. Canonical frontier labels, HOMO-LUMO gap warnings and
symbolic frontier cube requests are inapplicable; explicit MO cube indices
remain available where the writer supports that spin representation. QVF
retains `orbital_kind='natural'`. Default populations and occupied-space
localization consume the evaluated density/orbitals without asserting a
symmetry action on localized projectors. Source and physical work receipts do
not count those output analyses as integral savings.


### Exact terminal J/K reuse

A method-local `ScreenedMolecularJK` or `MolecularRIJK` fork can retain one
full-channel J/K snapshot and one pending J snapshot through their shared
capture implementation. Public qualified RHF/UHF entries enable
this capture; the prepared source remains free of densities and matrices.
Full UHF channels retain total-density J and ordered alpha/beta K inputs.
Native converged RHF requests J and K separately: they can be joined only
when their actual input bytes and immutable source match exactly. No density
norm tolerance, spatial operation, spin permutation or electronic-state
symmetry can establish a match. A single changed bit, including signed zero,
misses. Standalone response requests cannot by themselves replace a full UHF
capture or create a different physical state.

`ScreenedMolecularHF.evaluate(..., reuse_evaluated=True)` validates the source
and every density before attempting reuse. The default remains a cold
calculation. A successful reuse returns a receipt with zero new quartet,
primitive, density-contraction and transport work;
`reused_from_builds` identifies the one or two original physical build
numbers in that method's counter stream. The method still assembles its
owned physical FockBuild, energy components, density/frame metadata and
unchanged error bounds. No native result matrix or digest alone is trusted.
A mismatch, explicit reset or missing snapshot invokes the existing cold
path. A failed full-channel replacement discards its old capture.

Captured input/output arrays have immutable byte owners, independent of both
native inputs and returned writable J/K arrays. At most eight real AO matrices
are retained (full UHF plus pending J is conservatively reserved even though
that pairing is unused). A `4096 + 128*n_AO**2` byte reservation admits
capture, comparison and overlapping copies before tile reads. Capture work
and timing are separate from integral/contraction counts; `terminal_capture`
reports copied and compared values, retained bytes and capture time. The
reservation is logical and does not bound the native SCF's full memory.
Forks copy no captures, and process-local prepared-source leasing is unchanged.

This removes redundant terminal integral/contraction work when an exact
previous evaluation is available. It does not supply MANTA's missing
operation/support attestation, skip an unseen-source audit, or admit a new
complex integral kernel, periodic or derivative source. Fitted-source reuse
requires its already qualified selected-factor identity; a parent metric or
producer identity alone is insufficient. Additional native capture could
reduce snapshot copies, but is not needed to trust this exact reuse.


## Cost admission before source audit

With hierarchical screening, explicit `executor="ordered"` now constructs the
same complete-orbit mask using only MANTA's immutable molecular P0 producer
and its native omission bounds. Preparation reads no integral tile. Every
retained ordered tile is evaluated during J/K execution, with identity angular
actions; no representative reconstruction is authorized. Low-level callers
can request `qualify_manta_quartets(..., schedule="streaming", bounds=bounds,
qualification="ordered-source", ...)`. This requires exact native producer
and bound types. NumPy/native reduced executors reject the resulting plan.
`qualification="equivariance"` remains the low-level default and audits every
retained member and angular stabilizer.

Auto routing skips paired probes if `expected_builds <= repetitions`. Each
probe pair already pays for one ordered build, and even a free reduced build
cannot save more than the fastest ordered sample per subsequent call. This
makes repayment under the existing conservative timing criterion impossible.
The default two repetitions therefore select ordered-only qualification
before integral work when `expected_builds` is 1 or 2 and screening is
hierarchical. Higher estimates retain measured routing and full retained
source qualification. An ordered-only handle cannot later upgrade itself
through calibration: request a newly audited source for reduced execution.
Audit-based screening still needs its complete integral reads to choose its
mask, even when probe work is skipped. These rules do not predict SCF iteration
counts or guarantee faster execution for a badly underestimated run length.

Receipts distinguish `qualification_scope="ordered-source"` and
`equivariance_qualified=False` from numerical equivariance evidence. The
reported zero transport residual means no reconstruction error is incurred;
it is not a measured equivariance residual. Omission bounds still exclude
arithmetic roundoff. Counters show zero audit and probe calls separately from
all physical ordered work. Logical reservations remain conservative.

Streamed numerical-source identity now uses version 2: immutable producer,
shell layout, complete schedule descriptor and actual mask. Audit identity
and policy identity remain separate. Equal masks from ordered and audited
preparations have equal integral-source identities, while their qualification
evidence differs. Molecular HF source provenance also uses version 2 and
binds that integral source, frame, core Hamiltonian and nuclear energy without
including cost or audit receipts. Existing saved output identities are not
rewritten; executable QVF requests always qualify a fresh source. Neither
identity nor serialized evidence can construct an admitted handle.

## Source-qualified molecular RI factors

`prepare_abelian_molecular_ri` in `vibeqc.symmetry_ri` is an explicit,
experimental consumer of MANTA `RIJKBuilder` and `FittingTiles`. Supply the
molecule, orbital and auxiliary bases, complete Abelian operations/characters,
unchanged native `RIOptions`, `source_tolerance` (at most 1e-8), and a `Budget`.
It snapshots every ordered three-center tile once. MANTA owns primitive
integrals, normalization, the Coulomb metric, lower-triangle convention and
absolute/relative rank cut. No bubble hierarchy or integral kernel is copied.
This path does not yet reduce fitting integral generation.

The AO transformation C spans the full positive-overlap space. Auxiliary
columns W = F R rotate the producer's exact retained metric factor F by a
unitary R. Full metric invariance, finite retained Gram isometry and coefficient
leakage are separate checks. AO panels retain the shared orthonormal audit.
Native auxiliary factors preserve their actual Gram matrix F^dagger M F;
finite deviation from identity is recorded, not removed by whitening. A QR
coordinate solve checks containment without applying the ill-conditioned
Coulomb metric twice. The final auxiliary rotation is made unitary by its
polar factor, then all partner relations and source error are checked again
at the original tolerance. This rotates the factor gauge without changing
the native Gram tensor F F^dagger or its rank policy. A rank cut splitting a degenerate partner space
refuses; the consumer never repairs it by changing rank. Whole retained
spaces may mix by matrices. Metric-aware Cartesian and pure actions use the
shared AO layer. Non-Abelian and antiunitary actions are not admitted here.

With L = inverse(C), factors use B[A,p,q] = sum W[P,A] C[u,p]* T[P,u,v] C[v,q].
The allowed product is chi(p)* chi(q) chi(A) = 1. The finite Gram tensor is
E[p,q,r,s] = sum B[A,p,q] B[A,s,r]*, so J = sum B tr(B^dagger D) and
K = sum B D B^dagger. Density is never conjugated or projected. Transform
D to L D L^dagger and the result back to L^dagger F L; replacing the inverse
by C^dagger S would silently discard finite orthonormality error.

Qualification compares selected factors pulled back to AO coordinates with
all producer factors. Its tensor Frobenius bound includes factor omission,
finite coordinate reconstruction and the auxiliary rotation's unitarity
residual. Arithmetic roundoff is explicitly outside that bound. Source
identity binds producer, factor, coordinate transformations, labels and
actual stored blocks; a geometry or identity string alone grants no access.
All payloads are owned and immutable, and caller policy is snapshotted.

`source.jk(complex_density, budget=..., targets='jk', executor='blocked')`
accepts nonsymmetric complex trial densities. `executor='dense'` reconstructs
the identical selected factors for a fallback/comparison. Neither executor
recomputes integrals. Receipts separate complex scalar products, block calls,
AO transport, factor/transform/native matrix bytes, conservative live-memory
reservations, metric/three-center calls and primitive requests, and setup time.
Smaller product counts do not imply lower elapsed time in this Python engine.

`MolecularRIJK(source, budget=..., executor=...)` implements the existing
native real-AO RHF/UHF JKBuilder interface. `ScreenedMolecularHF` also accepts
this bridge and creates the same shared physical FockBuild, spin channels,
energy components and source/state separation. Standalone density response
may break spatial symmetry. Method overlap truncation and occupied/virtual
spaces remain method-owned. This is a developer API, not default job/QVF
routing; failure leaves the explicit native unreduced RI producer available.
Periodic kernels, magnetic complex integrals, CC residuals, geometry gradients
and stress require their own admission and numerical evidence.

### Measured RI routing

`MolecularRIJK(source, budget=..., executor='auto')` starts with dense
contraction of the same selected factors. Call
`calibrate_routes(targets=('j', 'k', 'k'), expected_builds=100)` explicitly for
an expected UHF workload; the five admissible native signatures are `('j',)`,
`('k',)`, `('jk',)`, `('k', 'k')` and `('j', 'k', 'k')`. Unmeasured signatures
keep the dense fallback. Explicit `dense` and `blocked` remain available.
Calibration never changes the factor mask, metric cut, source tolerance or
state, and cannot authorize an unqualified producer.

Two paired fixed nonsymmetric probes alternate executor order by default;
at most five pairs are allowed. Every output must agree at the unchanged
source tolerance, without adding a factor-omission allowance since both
routes use identical factors. A failed comparison publishes no route.
The same conservative cost rule used by quartet routing selects blocked
execution only when `(minimum_dense - maximum_blocked) * expected_builds`
exceeds the complete measured calibration time. If expected use is no greater
than the number of pairs, even the baseline halves cannot repay; calibration
performs zero contractions. These are workload estimates, not guarantees
under changing machine load, and no hidden recalibration occurs during SCF.

`route_calibrations` contains at most five immutable scalar profiles. Sample
receipts report executed products, block calls, transport, memory and time;
none is counted as a physical build. Actual build receipts identify the
selected executor and route while shared FockBuild source identity stays
independent of cost evidence. `fork()` retains the immutable source and scalar
profiles with fresh physical counters and no density/Fock history. Native
integral and metric work remains preparation-owned and is not repeated by
routing. Repeated-build work reservations now charge execution only; the
source receipt continues to report the complete preparation work separately.


### Qualified RI method and job entry

`MolecularRISymmetryOptions` admits the preceding finite molecular RI source
through `run_rhf`, `run_uhf` and `run_job`. It is explicit and experimental;
ordinary method defaults are unchanged. For an already specified complete
Abelian operation/character table:

```python
reduction = vq.MolecularRISymmetryOptions(
    rotations=rotations, translations=translations, characters=characters,
    auxiliary_basis="def2-universal-jkfit", source_tolerance=1e-10,
    budget=Budget(512 << 20, 10**12), executor="auto", expected_builds=100)
prepared = vq.prepare_molecular_symmetry(molecule, "def2-svp",
    symmetry_reduction=reduction)
result = vq.run_job(molecule, basis="def2-svp", method="rhf",
    symmetry_reduction=prepared)
```

A named auxiliary basis is mandatory. The snapshotted native `absolute_cutoff`,
`relative_cutoff`, `negative_relative_tolerance` and `mode_block` are explicit
configuration fields, defaulting to 1e-12, 1e-10, 1e-10 and 8. Do not also set
legacy `density_fit`, `aux_basis` or COSX options. Only real all-electron
molecular RHF/UHF with HCORE or same-basis supplied-density READ is admitted.
The prepared handle guards geometry, AO ordering, basis data, process and
exclusive use. Independent states reuse source factors and scalar cost
profiles, with fresh physical counters. No density or occupied space is
cached in it.

The actual S and H are checked for `U^dagger A U = A`, with
`U = C diag(chi) inverse(C)`. This is the covariant-operator counterpart of
`D -> U D U^dagger`; no matrix is symmetrized to pass. Method overlap rank
and natural output orbitals remain separate from source symmetry. The common
`MolecularSymmetryResult` owns the terminal density, physical FockBuild,
energy and density-defining natural orbitals, including capped/zero-step
states. RI terminal evaluation uses the shared exact-input capture described
above. Matching real AO densities reuse the evaluated J/K, even when the
source's adapted factors are complex. `terminal_capture` reports retained
bytes, copies and comparisons; the terminal receipt identifies original
builds and zero new factor-contraction/AO-transport work. A miss keeps the
counted cold fallback. Each method fork reserves capture storage before new
route probes; neither the prepared handle nor its scalar profiles retain
electronic-state snapshots.

Execution provenance has `source_kind='abelian-ri'`, `fitting_source`, and
`fitting_setup_work`. The first retains immutable preparation evidence; the
last is zero for a reused handle. It does not pretend fitting calls are
quartet audits. The common SCF/terminal work receipts separate contractions,
AO transport and logical memory. The one-electron receipt is separate.

QVF stores only the declarative request: operations, character real/imaginary
panels, auxiliary name, cutoffs and resources. Loading strictly validates the
schema and resources before numeric allocation; execution qualifies a fresh
producer. The container carries neither source authority nor timing profiles.
Geometry unit conversion may change last bits, so re-execution can legitimately
produce a different source digest. No copied digest certifies equivalence.

Post-SCF, response, gradients, stress, ECP, solvation, DFT+U, periodic kernels
and antiunitary symmetry remain refused on this path. Use an explicitly chosen
unreduced source when qualification fails. These gates do not change the
separate MANTA integral-engine or periodic Fock dispatch.


### Consume producer-owned RI factors

`prepare_abelian_molecular_ri(..., factor_export='producer')` now uses the
parent `RIJKBuilder.export_factors` result. Its factory type, source/factor
identities, kernel, normalization, shape and exact executed census are checked
before admission. Native B is rotated into complex auxiliary partners as
`B_adapted[A,u,v] = sum_B R[B,A] B_native[B,u,v]`; R is not conjugated at
this coefficient transformation. The existing AO changes of variance,
retained-Gram/isometry checks and final tensor-error bound are unchanged.
The native export identifies its rounded coefficients; arithmetic roundoff is
still outside the selection-error bound.

The default preparation no longer constructs a second Python fitting provider
or stores a raw three-center array. `factor_export='ordered-tiles'` explicitly
selects the former independent tile assembly for comparison/fallback. Neither
path silently substitutes for a failed source qualification. Both evaluate
all ordered coupling tiles; fewer integral calls are not claimed. Native
producer, coupling and metric-factor identities must agree between paths,
while their separately rounded/adapted factor source identities can differ.
Reduced and dense contractions always compare the identical selected payload.

`MolecularRISymmetryOptions.factor_export` carries the same selection through
methods, prepared handles and declarative QVF requests. The preparation
receipt adds `factor_export`, `factor_export_receipt` and
`raw_coupling_array_bytes`; existing fitting/SCF/terminal accounting remains
separate. Native source export does not certify point-group, broken-state,
periodic, derivative or local-subspace admission.

### Native RI block executor

`AbelianMolecularRI.jk(executor='blocked')` uses the immutable native
`_SymmetryRIBlocks` layout compiled during successful source preparation.
`executor='numpy'` selects the previous blocked implementation as a low-level
reference; `executor='dense'` reconstructs the identical selected factors.
Public molecular method options and automatic routing keep their existing
`blocked`, `dense` and `auto` values. The automatic candidate is now native,
with unchanged per-channel arithmetic agreement and measured payback gates.

The native constructor checks complete AO/auxiliary offset partitions,
nonempty block axes, unique in-range keys, exact packed extents and finite
complex values before copying. Empty irrep groups and an empty selected
tensor are valid algebra inputs; they do not override the molecular source
factory's nonempty retained-space requirement. Native arrays require exact
aligned C-contiguous int64/complex128 storage. No implicit conversion or
Hermiticity assumption is made. Each contraction owns fresh outputs; failures
cannot publish partial results or modify reusable factors.

The preparation receipt adds `block_executor`, `native_factor_copy_bytes`
and `native_block_plan_bytes`. Resource reservations include borrowed/owned
packed factors, descriptor validation, output matrices and bounded Eigen
workspace. Exchange holds one auxiliary panel, not the complete auxiliary
rank. `scalar_products` retains its complex-product meaning;
`block_contractions` counts actual auxiliary-panel trace, scale and matrix
products, so it differs from NumPy's batched-einsum call count. Integral,
factor-setup, AO-transport, calibration and terminal-capture costs remain
separate. Numerical source identity, retained rank and error tolerance do
not change when the executor changes.

### Calibrate only requested molecular routes

Public RHF/UHF automatic routing now installs a method-local request hook on
the fresh J/K builder. The first actual channel signature reuses a valid
prepared scalar profile or runs the existing paired nonsymmetric probes.
Unrequested signatures cause no calibration work. Invalid density descriptors
and resource requests refuse before probes; failed probes publish no profile.
Prepared sources retain only successful scalar receipts, never the request
hook, density, spin state, Fock matrix or method counters. Standalone builders
keep explicit calibration and the unmeasured baseline fallback.

Native `max_iter=0` evaluates one initial-density J/K map and reuses it at
termination. Its method-local cost horizon is therefore one, even when the
caller supplied a larger iterative estimate. A new calibration is skipped;
that temporary decision is not saved onto the prepared source. Already paid
profiles can still be reused. The separate orbital optimizer's schedule is
not inferred from its iteration cap. General iterative workload prediction
remains open, and explicit `expected_builds` keeps its existing meaning.

Provenance adds `route_calibration='on-demand'`, `single_evaluation_bound`,
`scf_elapsed_seconds`, `scf_calibration_seconds` and
`terminal_calibration_seconds`. `scf_seconds` excludes newly paid calibration
within the solver; `scf_elapsed_seconds` includes it. Route receipts still
hold probe work and timings, separately from physical and terminal receipts.
`routes` and `reused_routes` now list only signatures actually requested.
Calibration receipts also record `targets`, including zero-probe decisions
whose empty sample list cannot identify the signature.

### Ordinary irreducible products and multiplicity maps

`IrrepMatrices(group, matrices, budget=..., tolerance=...)` accepts a tuple
of exact complex128 arrays `(operation, partner, partner)`. The complete
matrix irreps are immutable. `products[a,b,c]` records integer multiplicity;
`conjugates` identifies conjugate irreps. `product(a,b,variance=(1,-1),...)`
conjugates only the second tensor leg before resolving the product. Direct
`resolve(matrices,...)` admits any complete unitary retained representation,
including repeated copies and matrix mixing between equivalent functions.

`ResolvedRepresentation.basis` maps coupled coordinates to the supplied
orthonormal frame. Offsets delimit irreps; each block orders multiplicity
before partner. The identity includes the actual partner basis, not just
characters. The matrices must be in orthonormal coordinates: Cartesian AO
actions require the existing metric-aware transformation first.
`basis_inverse` is the actual finite inverse, not an assumed adjoint.
Composition therefore preserves the intermediate coordinate map, and its
error bound includes the measured frame conditioning. `retained_bytes`
counts owned coordinates and metadata separately from construction scratch.
For `delta = ||U^H U-I||_F < 1`, composition bounds the physical matrix norm
by the coupled Frobenius norm times
`sqrt((1+delta_target)/(1-delta_source))`. This covers the admitted finite
frame rather than assuming a unit condition number. It still excludes
arithmetic error in the contractions. The source tolerance and complete
reconstruction audit remain independent requirements.

`InvariantTensorMap(matrix,target=...,source=...,source_tolerance=...,budget=...)`
audits the whole finite matrix against its reconstructed nonredundant blocks.
`apply` accepts arbitrary complex column batches and returns contraction and
basis-transport products separately. `compose` contracts only multiplicity
blocks and records actual `composition_products`. It requires the identical
resolved intermediate gauge and propagates source-error bounds, excluding
arithmetic error. `to_full` restores every partner. `recoupling` audits an
explicit tensor-leg permutation between two resolved spaces; multiplicities
can mix through dense matrices. Cross-pair four-index re-sorts must restore,
transpose and qualify the tensor with the new pair representations.

Every operation admits count-only memory/work before dense payload copies.
`required_bytes` is a conservative logical reservation; `stored_elements`
counts reduced tensor entries only and excludes the dense basis matrices.
Full audits, conversions and AO transports remain material costs. The module
is reusable finite tensor algebra, with no new public SCF dispatch, producer
attestation, state projection, projective symmetry or derivative admission.

### Compiled streamed quartet images

`PairQuartetSchedule.iter_representatives(keep=None)` streams selected keys
without angular image objects. After the usual source and mask admission,
native `quartet_jk_batch` uses `_SymmetryQuartetAccumulator.accumulate_orbit`
when available. Its inputs are the representative int64 vector, real tile,
int64 `(operation, 4)` mapped shell indices, identity operation and mode
(0 identity, 1 real-ERI permutations, 2 spatial plus permutations). Arrays
must have exact aligned C-contiguous storage. Invalid descriptors, payloads
or arithmetic poison the accumulator, including previously computed targets.

Expansion uses the existing real-ERI permutation table, identity first and
then ascending operation IDs, preserving first-occurrence order. The duplicate
set does not reorder arithmetic. The orbit-stabilizer census is checked before
the existing transport/contraction routine runs. This is bounded index algebra;
complete group, AO and numerical-source qualification remains the caller's
obligation. Streaming digest, final source identity and expected work census
are checked before publishing any result. No density symmetry is assumed.

Stored plans, NumPy execution and ordered execution keep their current paths.
Missing native support uses explicit checked image rows. Complete source
qualification still expands every required orbit in Python and reads the
ordered source; the execution optimization cannot bypass that audit.

The existing 2,048-byte-per-operation logical metadata reservation covers the
four-index map and contiguous copy, up to eight unique set keys/nodes per
operation and their six-int64 member rows. This is a conservative logical
reservation, not a portable allocator/RSS bound. Existing index-work budgets
remain charged; no quartet-sized persistent allocation is introduced.
Integral, contraction and AO transport counters retain their meanings. The
status document records paired elapsed cost, complete source work and process
memory separately from the number of Python images eliminated.

### Native molecular source consumption

For a streamed `_MantaSource`, a compatible runtime can use
`_SymmetryQuartetSourceAccumulator` around the existing algebra accumulator.
It pins the immutable MANTA builder and owns one additional `Workspace`.
Only exact native libint/MD real molecular four-center, derivative-zero,
full-Coulomb providers qualify; their precision-zero constructor contract
remains required. Kernel/request metadata, source identity and the complete
AO shell partition are checked before workspace allocation. Fixed-size source
and digest descriptors are checked before string copies. Every tile's shape,
extent and finiteness are checked before contraction. Null buffers
become explicit zero tiles; allocated zero-valued tiles keep their actual
non-null receipt. No density screening or batching kernel is introduced.

`consume_orbit` reads one representative through `IntegralProvider.evaluate`,
hashes its big-endian int64 key and native float64 tile bytes in the existing
wire format, and invokes the same orbit expansion and AO contraction before
the borrowed view expires. `finish` checks the complete representative count,
digest and immutable source identity before releasing targets. Descriptor,
source, digest or arithmetic failure invalidates partial targets. Abandoning
an unfinished bridge invalidates its algebra accumulator too. The binding
keeps that accumulator alive for the bridge's entire lifetime.

The count-only source planner conservatively charges another full producer
memory reservation, one maximum-size zero tile and controls; its work model
includes the producer kernel cap and digest/census work. This is an upper
reservation, not additional integral calls or an RSS prediction. The live audit
session remains charged by the plan. Native requirements choose this optional
bridge only if it fits alongside the plan, targets and mask. Molecular methods
subtract their bridge and capture storage before making the same choice at
planning and execution. Smaller previously valid budgets retain the checked
Python reader with native contraction. No payload is copied or evaluated to
make that resource decision.

Stored/generic sources, ordered execution and older runtimes retain their
existing readers. Complete numerical qualification and all scientific error
limits are unchanged. Shared Fock identities, MANTA producer ownership and
periodic admission are unaffected; this bridge cannot grant an unqualified
source permission to reconstruct angular images.

### Producer-owned molecular source support

`core._manta.plan_molecular_source_support(provider,operations,budget)` inspects
only immutable source metadata and reserves native/Python input storage and
inspection work before any operation values are read or copied.
`certify_molecular_source_support(provider,rotations,translations,destinations,
geometry_tolerance,budget)` requires C-contiguous float64 operations and int64
shell maps. It copies admitted inputs safely, including unaligned arrays.
The native certificate has no Python constructor or mutable attributes.

Each map must be bijective. Native shell exponents and contracted coefficients
must match exactly, including single-ulp differences; angular momentum,
pure/Cartesian convention and AO extent must also match. Orthogonality and
center residuals use the explicit geometry tolerance. The producer does not
certify the group law, representation matrices or arithmetic equivariance.
The certificate records the maximum center residual and actual shell/radial
inspection counts; it evaluates no integrals. Its versioned identity includes
the source, operation order, map and tolerance, with canonical big-endian
integer/binary64 encoding and signed-zero normalization.

`qualify_manta_quartets` reserves this component alongside actions and the
quartet plan, obtains it before creating an evaluation session, and stores it
on `_MantaSource.support_certificate`. Session construction preserves the
certificate. The complete ordered-tile/stabilizer audit and all source
thresholds remain unchanged. Missing native support retains that audit;
incompatible support refuses qualification before integral evaluation.
`qualified_molecular_source` exposes `support_certificate_identity` separately
from physical source/mask/state identities and revalidates the live provider.

The reservation includes the live provider, metadata copies, seen-set scratch
and conservative controls. `additional_bytes` is the incremental logical cap;
`scalar_terms` is an inspection-work cap. `shell_checks` and `radial_values`
count actual metadata comparisons, never integral calls or contraction savings.
The certificate keeps its immutable provider alive. Its arithmetic permission
is explicitly false: future MANTA source/screening/bubble/error certificates
must compose with this evidence before a consumer can skip any full audit.

### Bind certified maps once per native build

A compatible `MolecularSourceSupport` owns its int64 destination inventory and
provides `validate_maps(destinations)`. Validation accepts exact aligned
C-contiguous int64 arrays and checks dimensions before scanning values. It
allocates no replacement inventory. The source-support constructor still
accepts unaligned input by copying it after count-only admission. Its logical
reservation now includes input, binding snapshot and retained native maps
concurrently, adding eight bytes per operation/shell pair to the former cap.

Both quartet-plan constructors validate available map evidence before creating
an evaluation session. The native batch path calls
`_SymmetryQuartetSourceAccumulator.bind_support(certificate,destinations,
identity,mode)` once. A mismatched source/map, invalid descriptor, repeated or
late binding invalidates the accumulator and its algebra targets. A bound
bridge accepts `consume_key(representative)` and rejects explicit-map overrides.
The same checked representative digest, null-tile receipt, source identity and
final census still guard export. Certificate and provider lifetimes are native.

The reusable native `(operations,4)` scratch map replaces the transient Python
map inside the existing per-operation accumulator metadata reservation. No
quartet inventory or integral cache is added. Source/workspace budget fallback
is unchanged; an unavailable binding API uses `consume_orbit` with explicit
checked maps. Integral and scalar-product counters retain their meanings.
Profile map conversions separately from numerical work and include complete
qualification, transport, reconstruction and terminal costs in comparisons.

### Inspect molecular quartet construction error

The explicit experimental native API is
`core._manta.plan_molecular_quartet_construction(provider,key,budget)` followed by
`certify_molecular_quartet_construction(provider,key,budget)`. The key contains
four native shell indices. Both entry points require an immutable native
MANTA or reference builder with real molecular four-center full Coulomb and
four single-contraction s channels. The other basis shells may have higher
angular momentum; selecting one of them refuses the certificate.

The immutable result exposes `evaluated`, `lower`, `upper`, `error_bound`,
`null_tile`, `key`, `source_identity`, `plan` and `validate(provider)`. Its
`qualification_scope` is `finite-molecular-s-quartet-construction` and
`arithmetic_equivariance_certified` is false. It does not mutate the builder,
change its physical source identity, select a mask or authorize a reduced
calculation. The existing full-audit fallback is the only qualified path.

`plan.shell_calls` counts one actual source evaluation;
`primitive_products` counts all primitive reference terms and requested native
primitive products, not measured libint internal evaluations. Scalar/kernel
counts are conservative admission caps, not measured floating-point work.
`additional_bytes` includes another native workspace reservation and constant
proof/receipt controls. `modeled_peak_bytes` includes the live builder, other
live bytes and backend margin too. These logical reservations are not RSS.
No all-quartet array or integral cache is retained. Measure native evaluation,
proof and total setup costs separately when evaluating future audit savings.

The shared `coulomb_boys_enclosure` wrapper exposes the existing verified radial
reference to this proof without changing the numerical radial kernel. Broader
MANTA radial improvements must remain shared; do not duplicate an integral
engine or transfer a Coulomb certificate to another kernel. The exact stored
basis data define the reference. Normalization generation, higher angular
transport, geometry mismatch and source-wide permission remain open.
