# Hermitian operator and Fock/KS architecture

Design and migration issue: #472. Audit baseline: `f14e9ef5`.
Literature cutoff: 2026-10-05. This document describes an incremental migration,
not a claim that every eigensystem in the codebase is a Fock problem.

## Decision

Use a Hermitian operator as the shared mathematical contract. A Fock or
Kohn-Sham operator is a one-particle specialization. CI Hamiltonians and
Hermitian TDA response matrices share the algebra but retain their own kinds.
Full RPA/Casida pencils, non-Hermitian EOM-CC Jacobians and indefinite metrics
need distinct contracts. In particular, Hermitian storage is not an assertion
that the physical SCF state is stationary or stable.

The builder owns integrals, density dependence, image sums and exchange/XC
approximations. The operator owns a fixed representation. The solver owns its
iteration space. The occupation layer owns electron counts, smearing, Fermi
levels and symmetry ensembles. The accelerator owns DIIS, level shifts and
orbital rotations. None may silently substitute for another layer.

The Python interface is additive in `vibeqc.solvers.operators`:
`HermitianOperator`, `FockOperator`, `OperatorMetadata`, `solve_operator`,
`solver_capabilities` and `subspace_overlap`. Historical eigenproblem
descriptors remain available. `solve_eigenproblem` also accepts a new operator.
The native implementation boundary is internal:
`cpp/include/vibeqc/hermitian_operator.hpp`. Existing SCF and result signatures
remain intact; a native solve borrows const matrices synchronously, whereas
Python snapshots copy their arrays. Neither layer retains a factorization
cache across solves. This deliberately avoids a Python callback in a native
SCF loop and avoids a new ABI for method-specific Fock builders.

## Construction and evaluated states

The construction/state continuation of #472 originated in draft !203 at
`b0047d47`; subsequent main landings supersede that implementation baseline.
Shared eigensolution does not establish a common evaluated state for every
method. The coverage table in
[operator_consumers.md](operator_consumers.md) tracks these claims separately.
Its [release boundary](operator_consumers.md#release-boundary-and-remaining-algebra-ownership)
also separates full migration from the bounded 0.18 qualification in #557.

`vibeqc.fock_state.FockBuild` links the existing `FockOperator` to the complete
evaluated spin/k density epoch, method energy components and a separate
`SourceProvenance`. Its first producer is Gamma Ewald UHF, including PATOM,
ordinary SCF and terminal confirmation. The method still constructs
`H + J[Dalpha + Dbeta] - K[Dspin]`, owns all spin factors, and evaluates the
existing HF energy expression. `EnergyComponents` checks the method-supplied
decomposition, owns immutable scalar values and rejects nonfinite complete
totals, including overflow; it never derives KS energy by applying a half
trace to XC.
There is no change to the provider, kernel, support, occupations or tolerances.

An immutable `OperatorFrame` owns S and rectangular X, validates `X^H S X = I`
once, and records basis identity, AO ordering and shell-image gauge. Both spin
channels reuse these buffers. Its identity also covers S/X values and column
ordering. Reuse of a previous subspace checks this identity in addition to
kind, spin, physical k point and symmetry block. Coordinate changes require
explicit transport and a newly declared frame; equality of matrix dimensions
or build generations is insufficient. K weights remain in state sums and
never scale eigenvectors. This first frame implementation stores explicit
S/X; matrix-free overlap operations are not advertised.

`FockBuild.from_dense` owns frozen D/F buffers and shares the frame. The
completed state captures no mutable builder or integral workspace. Array
storage uses immutable owners, including refusal to re-enable NumPy writes.
The builder is synchronous and must own its inputs or refuse changed source
settings before reusing caches. Gamma UHF checks resolved lattice, basis,
nuclear, kernel and environment-selected Hartree controls at every build.
Concurrent mutation of a builder's source during a call is unsupported.
Generation increases only after a completed build; a terminal confirmation
build can advance it without advancing the public SCF iteration.

The full build identity includes the operator kind and complete normalized
energy record, including named components and corrections. Equal D/F buffers
with different energy records therefore identify different evaluated states.
Density, source and frame identities remain separate; solver modifications
retain the identity of their physical parent. Public construction and
`dataclasses.replace` recheck the content digest and reject stale identities;
use `from_dense` to construct a changed physical state. This adds one linear
verification pass over the immutable dense snapshots, with bounded byte
chunks and no additional retained arrays. Native borrowed-view kernels remain
outside this owned dense adapter.

Residual aggregation remains method-owned. `FockBuild.commutator_norm` reports
`sqrt(sum_c w_c ||F_c D_c S_c - S_c D_c F_c||_F^2)` over its channels. BIPOLE
RKS instead uses `sum_k w_k ||F_k D_k S_k - S_k D_k F_k||_F` for its existing
convergence diagnostic. These quantities are not interchangeable. A migration
must preserve the method's aggregation, thresholds and exact-confirmation
checks rather than substituting the generic state's diagnostic in its gates.

Source identity is separate from frame identity. Gamma's source record
includes actual ordered shells and centers, lattice and dimensionality,
resolved finite-domain/screening settings, Hartree backend and cutoff,
zero-mode and molecular-limit exchange conventions, grid controls and method
coefficients. This is a settings record, not a native-binary attestation,
symmetry certificate or finite-support convergence claim. Native build
provenance remains part of validation evidence. MANTA receipts and BIPOLE's
resolved-domain records remain authoritative for their own sources.

A solver-modified operator has `role="solver"`, a modification description,
and the same build/density identity as its physical parent. DIIS and level
shifts never replace the stored physical Fock. ROHF/ROKS effective operators
retain a distinct role; their physical spin operators require separate
construction adapters. Main's #487 finalization repair is preserved; remaining
real-only admission work stays tracked under #494.

Gamma 3D RHF/UHF and Gamma 3D/slab RKS/UKS return `result.fock_build`,
a separately stored next density and a density-closure residual. Its established terminal policy is
preserved: rebuild on the density defined by the returned orbitals, report
physical Rayleigh quotients, and accept convergence only after the terminal
energy, commutator and closure checks. At an iteration cap those orbitals need
not diagonalize the evaluated physical Fock. Legacy result arrays remain
writable copies. SCF history records identify the evaluated build of each
reported cycle, replacing the last identity together with terminal history.
The existing `vibeqc.output` route carries the scalar identities, energy
components and state relation into QVF provenance; restart densities still
come from the evaluated orbital/occupation state.

The Gamma RKS and UKS producers call the existing joint libxc evaluator once
per physical density state. UKS evaluates the polarized XC energy and both
spin potentials together, with one total-density Hartree build and the
existing per-spin `K[2 D_spin]/2` exchange convention. It records `E_xc` directly alongside kinetic, nuclear,
Coulomb and coefficient-folded HF exchange energies; a half trace of the KS
potential is not the XC energy. Pure, global-hybrid and screened-hybrid
exchange retain the existing libint assembly. PATOM uses a separate full-HF
build and solver history. UKS keeps separate alpha/beta solver histories
through SCF and terminal look-ahead; active Davidson controls require the
occupied space, or all retained orbitals for MOM and quadratic steps.
Caller options and guesses remain untouched. The source receipt binds resolved functional
components, physical exchange screening, XC quadrature bytes and options,
cell-density convention, and the 3D or open-normal slab boundary. The Ewald
split remains a separate numerical parameter. Both supported boundaries have
zero separate Madelung correction. Existing HF source identities, BIPOLE
resolved domains and MANTA bounded native admission remain unchanged.

The dense adapter owns two D/F spin pairs plus diagonals and one shared frame.
`storage_nbytes` counts these retained arrays, excluding Python objects,
integral caches, temporary matrices, legacy export copies and eigensolver
workspace. It is not a peak-memory bound. The method does not retain full
matrix histories. The allocation benchmark distinguishes retained storage,
observed allocation peaks and wall time, without claiming matrix-free SCF.

Native bounded kernels may implement the same semantics with synchronous
borrowed views or owning handles and their admitted solver. No generic Python
snapshot is inserted into MANTA's hot loops. The native restricted periodic
mean-field state in `periodic_mean_field_state.hpp` remains the qualified
full-mesh reference for its existing downstream consumers; it is not replaced
by an unevaluated orbital projector. An adapter must preserve that contract's
uniform full-mesh, retained-space and reference-kind admission.

## Representations and ownership

Python dense and sparse snapshots copy input values and the diagonal. Legacy
metric arguments are copied too; an explicit `OperatorFrame` shares its
already validated immutable overlap and orthogonalizer. Dense arrays have
immutable owners. Callers must not replace sparse storage internals or
deliberately re-enable writes to sparse backing buffers.
Sparse input stays sparse. Callback owners guarantee a fixed linear Hermitian
action throughout a solve, declare dimension/dtype, and supply their own
diagonal for Davidson. Hermiticity cannot be certified for an arbitrary
callback without an exhaustive check; the interface treats it as a producer
contract and checks the returned action shape, type and finiteness.

For transformed sparse/callback Davidson, supply explicit X and
`reduced_diagonal = diag(X^H A X)`. The reduced diagonal is copied, finite,
real and sized to the retained rank. It describes this exact X, including its
column ordering and gauge; it cannot be inferred from the AO diagonal.
Overlap-only canonical reduction therefore remains unsupported for callback
Davidson. No action is invoked to validate or reconstruct a missing diagonal.
Dense operators compute their reduced diagonal internally and refuse this
extra field. X and S remain explicit arrays; this is not a matrix-free metric
or a bounded-memory overlap factorization.

`apply` accepts one vector or a block. A supplied `matmat` performs a block
application; otherwise columns are applied individually. Both real and complex
native Davidson now accept a block callback, so a
supplied `matmat` receives a whole search/correction block in one Python call.
A vector-only callback is still applied column by column. Vector application
counts remain the sum of block widths, not the number of Python calls. Explicit `block`
access is available for preconditioner construction. Callback block access,
custom block preconditioners and matrix-free overlap actions are deferred;
the current API refuses unsupported requests rather than probing unit vectors.

Callback return buffers may be temporary workspace, valid until the next
callback. `apply` returns an owned action and copies each vector before a
vector-only callback is called for the next column. This requires storage for
the vector or block result, but never reconstructs the full operator. Producers
must not mutate an input vector or reuse output concurrently before the call
returns. Callback thread safety remains the producer's responsibility.

Metadata records operator kind, basis identity, generation, spin, k-point,
symmetry block and optional occupation annotations. It neither creates symmetry
blocks nor assigns occupations. A new Fock build creates a new snapshot and
generation. Captured mutable callback state is never made safe by incrementing
a counter during a solve. Cached builder data must be invalidated by its owner
when density, geometry, basis, k-mesh or approximation settings change.

Integral providers and density-dependent consumers remain below this boundary.
Completed integral tiles can be ephemeral, while a method's fixed Fock action
must own its density epoch, coefficients, numerical supports and execution
plan. A const builder or a receipt identifier does not guarantee immutable
captures. Provider/source receipts stay beside the method result, separate
from solver frame annotations. This boundary also applies to the
MANTA provider work; no integral backend or physical source is enabled by a
solver capability change.

A producer batch over several densities is distinct from operator `matmat`,
which applies one fixed operator to several trial vectors. Density batches
retain input/output ordering and per-density J/K identities beside their
shared source/traversal receipt. Each completed Hermitian Fock has its own
frozen build state and spin history. Arbitrary response-density contractions
may be non-Hermitian and do not enter `FockOperator` merely because their
integral provider also builds physical Focks.

The experimental MANTA direct batch and streamed RI spin-pair builders can
already feed explicit Fock snapshots through this boundary. Their current
real symmetric density admission and source/metric receipts stay with the
builders. Crossing tests form per-spin `H + J_total - K_spin`, check against
independent libint integral contractions and generalized LAPACK, and verify
snapshot lifetime across subsequent density builds. The small RI auxiliary
space in these tests establishes algebraic consistency, not fitting accuracy.
No production default or integral-provider capability changes follow from
these tests. MANTA's complete auxiliary-metric decomposition remains a
producer-side rank factorization, separate from orbital eigensolution.

Opt-in molecular MANTA RHF/UHF uses the existing native SCF drivers through
`JKBuilder` and therefore reaches `solve_fock_matrix` directly. The provider
selects the integral source and J/K approximation, while the SCF options select
dense or Davidson orbital solves. Full-spectrum exact/RI fixtures check the
returned physical Fock spectra, AO residuals and metric normalization against
generalized LAPACK. Invalid Davidson subspace controls must still raise through
this entry point. The native unrestricted driver's spin-specific recycling
remains follow-up work, separate from the Python Gamma UHF migration.

MANTA's finite periodic spin-Fock producer supplies complex Hermitian
`F[spin,k,AO,AO]` snapshots and matching electronic energies. Each spin/k solve
has its own operator metadata; a whole-mesh density change creates new Fock
snapshots while source receipts remain producer-owned. Coulomb, erf and erfc
crossing tests compare generalized spectra, metric normalization and recycled
subspaces without changing the producer's kernel or zero-mode policy.

The finite periodic MANTA RHF driver intentionally retains the specialized
`solve_bounded_periodic_rhf` Jacobi solver. Its pre-admitted work and memory
exclude the copying/validation allocations of the generic operator adapter.
Tests compare its exported F/S/C/energies against generalized LAPACK and the
common operator contract, including reduced overlap rank and shifted complex
meshes under strict-zero-mode and BvK-Ewald exchange policies. These checks do
not establish convergence of the finite integral supports to an infinite sum.

An AO shell-image change is a coordinate change, not a new density iteration.
For the producer convention, `U_mu=exp(+i k.delta_mu)` transforms matrices as
`U M U^H` and coefficients as `U C`. A consumer must use a different basis/frame
identity and explicitly transport a previous subspace; generation alone cannot
authorize reusing coefficients from the old gauge. The bounded driver's last
evaluated density also remains distinct from the density constructed from its
newly diagonalized orbitals until the measured closure tolerance is met.

The selected periodic job preserves this distinction through public result
properties, retained scalar iteration history and QVF. Successive capped jobs
are checked against the independent generalized HCORE projector followed by
common-operator Fock solves. Same-density energies, density closure and metric
commutators refer to each evaluated epoch. Output transport to QVF's nuclear
centers applies `U^H D U`; it neither changes the native state nor substitutes
the next orbital projector. Combined stopped-state and independent AO-image
checks cover both admitted exchange zero-mode policies on shifted meshes.

Legacy Python adapters still copy and validate X/S for each Fock solve.
Gamma Ewald UHF now shares a validated frame across evaluated builds and
solver modifications. Neither dense adapter satisfies the specialized
memory-bounded drivers' storage contracts.

The stateless internal `solve_dense_fock` adapter handles multi-k GPW/GAPW
and fixed-density band paths. It accepts either an AO matrix with both X/S,
or an already orthogonalized matrix without either. The latter is important
for projected DIIS: no fictitious full AO operator is reconstructed. Its
coefficients and residuals remain in the orthogonal frame, explicitly labelled
in metadata and diagnostics; the method applies X to form AO coefficients.
Spin, k-point and iteration generation are annotations on each solve. No
all-k metric copies are retained by the adapter, and weights never enter
orbital normalization. A supplied X must be orthonormal in the actual S;
an inverse built by flooring an unresolved metric direction cannot bypass
that check.

The real Roothaan loop used by molecular ROHF/ROKS and Gamma GPW ROHF/ROKS
also owns one `FockDiagonalizer`. Saunders-Hillier shifts remain a method
operation on the AO effective Fock and are removed on the existing exit
paths. The solver snapshots Davidson controls and recycles its own subspace.
The loop requires the complete retained spectrum for occupations and reported
orbitals; a partial Davidson request is rejected before building a Fock.
Guest-Saunders canonicalization uses separate dense solves for closed, open
and virtual MO blocks, labelled as coordinate partitions rather than spatial
symmetries. It never globally sorts across those blocks, and sizes rotations
by the retained orbital count, including rectangular AO coefficient arrays.
Fractional-shell ROKS keeps its Roothaan spectrum and bypasses these rotations.

Gamma Ewald ROHF also uses `FockDiagonalizer` for its effective-Fock solves,
including the SCF and final fallback paths. Its dense default remains;
explicit Davidson requests now follow the shared size threshold and controls,
with one restricted-open-shell warm subspace. It requires all retained roots
because the open-shell occupation rule ranks all non-core candidates by their
alpha orbital energies. Root counts outside that retained space are refused
before J/K construction. The HCORE seed stays with the guess producer. Virtual level
shifts, alpha-energy selection and density updates remain in the method;
the solver does not remove shifts. Finalization preserves main's #487 repair:
it retains the tested determinant, canonicalizes within equal-occupation
blocks of the unshifted physical effective Fock and keeps capped returns
internally consistent. These method-owned steps remain outside eigensolution.

Gamma Ewald UHF uses separate alpha/beta `FockDiagonalizer` instances for
PATOM in-field Focks, SCF iteration and unshifted terminal consistency checks.
The caller's native Davidson options and guesses are never modified or shared
between the two recycling histories. Ordinary Aufbau iteration may request
only the occupied space; PATTERN_HOLD and quadratic updates require every
retained candidate, enforced before J/K construction. HCORE preserves the
configured dense/Davidson choice in a separate one-electron solve. Its result
is copied into independent starting spin frames without seeding either Fock
history. Damping, level-shift schedules, MOM selection, quadratic
rotations and terminal density/physical Rayleigh-quotient reporting remain
method-owned. A terminal consistency solve may advance its spin's private
solver generation without advancing the public SCF iteration count.
This real-only driver opts into the adapter's `real_only` capability: complex
Davidson guesses are rejected before J/K, and complex frames/Focks are refused.
The generic complex-capable adapter remains available to consumers using
adjoints. Audit #494 tracks the remaining real-only consumers; a complex
orbital frame must not enter a method that builds densities with a transpose.

Post-SCF semicanonicalization also uses `solve_dense_fock`: Brueckner CCD
rotates only active occupied and virtual blocks, FNO-CCSD rotates the retained
natural-orbital virtual space, and the reference UMP2/ROHF-MP2 kernel rotates
occupied and virtual blocks separately for each spin. Each solve has a named
orthonormal MO partition and returns its complete ascending spectrum. The
methods never sort across partitions or change their denominators. Brueckner
keeps frozen columns and singleton blocks unchanged; the MP2 kernel skips
empty blocks and retains the transformed occupied-virtual Fock coupling that
produces singles. FNO density diagonalization and truncation remain method
operations. Its Fock result and work arrays are released before the final
native CC call; this does not impose a memory bound on the dense FNO setup.

Molecular DLPNO-MP2 and DLPNO-UMP2 also share dense solves for the retained
pair-PNO Fock blocks. Basis labels identify the pair and canonical/localized
source frame; unrestricted labels use indices in the active occupied window
and separate alpha/beta metadata. Restricted labels include the frozen-core
offset, matching the result's pair keys. Density eigenvalues still select
PNOs with the original strict threshold and minimum-one policy. The shared
helper also serves experimental neutral CCM UMP2 and its UCCSD pair assembly,
with separate CCM frame labels. This does not change their physical-support
status. PAO metric/Fock preparation, CC pair kernels and bounded-memory
implementations remain separate. The migrated paths already form dense pair
matrices; the adapter adds temporary snapshots and residual work, without
retaining an operator cache across pairs or claiming a requested-memory cap.

## Numerical contract

For a standard problem, solve `A Y = Y e`, with `Y^H Y = I`. For an AO problem,
use `C = X Y`, `X^H S X = I`, and solve `X^H A X Y = Y e`. Every complex
transformation uses an adjoint. X can be rectangular or nonsymmetric. Previous
AO coefficients map to reduced coordinates by solving `X Y_previous = C_previous`
in least squares, not by solving with the transposed X.

When the Python operator supplies S without X, canonical reduction retains
overlap eigenvalues above `max(cutoff, cutoff * max(abs(s)))`. The default
cutoff is `1e-12`. Negative directions beyond that cutoff are rejected, not
silently discarded. The requested root count must fit the retained rank.
At the strict entry point, `n_roots=0` requests all retained roots; a Lanczos
`ncv` smaller than the requested count is refused before applying the operator.

An explicit retained `X` does not exempt the full overlap from positive
semidefinite admission. Owned `OperatorFrame` construction checks its spectrum
with an absolute/relative allowance of 1e-12; separate S/X operators use their
`overlap_cutoff`, matching implicit reduction. Negative directions beyond that
allowance are refused even when X omits them. Null directions and roundoff-sized
negative eigenvalues remain admissible when the supplied X is metric-orthonormal.
This adds one overlap eigenspectrum check at frame construction, reused by all
operators sharing that frame; it is not a bounded-memory native kernel change.

Existing production methods retain their own overlap policies: molecular SCF
passes its rank-reduced X, Bloch native solves retain their near-singular
refusal, and DFTB0 uses a strict SPD Cholesky reduction with no rank deletion.
Those differences are deliberate method policy, visible at the common boundary.

`residuals` contains `||A C_i - e_i S C_i||_2` when S is available, and
`projected_residuals` contains `||X^H A X Y_i - e_i Y_i||_2`. With X alone,
the AO metric is unspecified and the reported residual frame is `orthogonal`.
For a truncated metric the full AO residual can be nonzero even for an exact
projected solution. The result exposes both; it never presents projected
convergence as an exact solution in discarded directions.

Strict operator solves require all requested projected residuals to be at
most `tol`, a successful backend status, and Frobenius metric-orthogonality
error at most `1e-7`. They raise on failure. Counts are observed callback
vector applications, including diagnostic applications; unavailable native
counts are null. They are not inferred from iteration counts. Dense native
SCF uses a `1e-7` absolute residual check; Davidson uses its configured
`conv_tol`. Native diagnostics are currently checked internally, not added to
every historical SCF result or output schema.

A strict residual/orthogonality refusal raises `SolverConvergenceError`, a
`RuntimeError` subclass. Its scalar diagnostics distinguish the original backend
status and residual maximum from the independently recomputed projected residuals
and metric-orthogonality error, with the actual unchanged tolerances. For a dense
problem it also retains the exact input matrix, supplied X/S, materialized reduced
matrix, Ritz vectors, returned vectors, residual vectors and Gram matrix, provided
the complete numeric snapshot fits eight MiB. Larger snapshots are explicitly
omitted; allocation failure also leaves a scalar-only refusal. Sparse and callback
operators are never densified or applied again for capture. Failed NaN/Inf values
remain in the numeric evidence, with nonfinite scalar summaries encoded as null.

The arrays are immutable and are not an accepted solution, restart or SCF state.
With X alone, S is unknown: orthogonality refers to the retained Ritz vectors and
no generalized AO residual is claimed. The independently evaluated projected
residual and the backend residual may differ in finite precision because their
matrix-product associations differ. Neither is substituted for the other and no
relative-residual waiver is introduced. A caller can explicitly preserve a refusal
through `vibeqc.output.eigensolver.write_solver_failure(path, error)`; the atomic
NPZ contains numeric arrays and `diagnostics_json`, and loads with
`allow_pickle=False`. Collection does not write files or change acceptance by itself.

Selection and ordering are separate. Dense selection supports smallest/largest
algebraic, smallest/largest magnitude, both ends and nearest to sigma. Both
ends assigns the extra root to the upper end for an odd count. Ties use the
stable order of the enumerated ascending spectrum. Returned dense roots are
ascending in energy. Iterative algebraic methods support SA/LA only through
the strict interface; experimental interior backends retain distance targeting.
No deterministic orientation is promised within a degenerate eigenspace.

Small residuals certify eigenpair accuracy, not completeness of an extremal
or interior root set. `selection_verified` is true only for the enumerating
dense backend. `require_selection=True` refuses uncertified iterative methods.
Warm-start overlaps do not confer that certificate. `subspace_overlap` provides
overlap amplitudes and principal cosines for a whole degenerate cluster;
automatic state following, crossings, and occupation reassignment remain
explicit caller decisions. A previous result must have the same declared
operator kind, basis, spin, k-point and symmetry block. Generation may change
across SCF.

## Correlation consumer boundary

DLPNO migration is tracked by #580 under #472, with periodic/MANTA composition
through #577. The [correlation-source admission contract](../design_dlpno_correlation_source.md#shared-state-admission-and-remaining-integration)
separates an evaluated reference Fock from certification of its correlation
factors. Preserve the actual exchange zero mode, finite support and fit;
shared operator/frame identities do not equate different Hamiltonians.

A solver replacement must retain generalized residual and metric-orthogonality
checks in the declared retained space, occupations, occupied projectors,
independent spin partitions and frozen-core boundaries. Compare degenerate
subspaces rather than individual columns. An accepted eigensolve does not
certify SCF stationarity: stopped-state physical Rayleigh quotients are not
canonical eigenvalues, and a correlation consumer must explicitly admit or
refuse the reference it receives. No production correlation claim or new
bounded 0.18 release gate follows from a shared solve.

## Capabilities and compatibility

The strict capability table describes executable behavior, not the old broad
`supports_*` flags. Dense accepts explicit real/complex matrices. Real and complex Davidson
accept explicit, sparse and callback operators with a supplied diagonal. Their
native small-partial-spectrum callback fallback would construct a dense matrix,
so the strict interface refuses that dimensional regime. Full-spectrum
iteration naturally needs square subspace storage. Transformed callbacks
still require a separately supplied reduced-frame operator and diagonal.
LOBPCG, JD and harmonic-Davidson remain explicit-only because their native
callback adapters materialize A or are unavailable. Lanczos supports
real/complex actions but has no restarted or block/warm-start implementation;
it remains experimental. A dense reduction of sparse/callback input is never
implicit in the strict interface.

Nondefault options unsupported by a selected strict backend are errors. In
particular, `preconditioner='none'` is not passed to a kernel that always uses
its diagonal, nor is a guess accepted by a backend that ignores it. Defaults
of irrelevant historical option fields are neutral. Solver keywords in
`run_job` support dense and Davidson molecular SCF; LOBPCG now raises instead
of silently executing Davidson. The generic eigensolver offers LOBPCG for
explicit real problems. Method-specific Davidson size thresholds still apply.

Historical descriptor/strategy and direct pybind entry points are retained
for compatibility. Their callback probing/materialization and incomplete
option validation are listed as migration debt in the consumer inventory.
New integrations must use the strict interface. Removing those old behaviors
or changing pybind signatures needs a separately reviewed compatibility
increment. The fixes to complex generalized reduction, magnitude selection
and complex real-backend refusal apply to historical descriptors immediately.

## GPLHR decision

`9ac1817` is in the audit baseline. Its signed shifted-diagonal denominator,
distance-based harmonic selection, retained correction blocks, stagnation
status and singular-pencil refusal remain present in `cpp/src/gplhr.cpp`.
Its search space grows and restarts. It does not implement the published
locally harmonic recurrence, Schur extraction, projected preconditioned
residual sequence and previous-direction block. The generalized
non-Hermitian formulation in Vecharynski, Yang and Xue is also absent.

Retain the algorithm as experimental **harmonic-Davidson**, with
`harmonic_davidson` as an accurate alias and `gplhr` retained for compatibility.
Keep the `zuev_gplhr_2015` citation as provenance, not proof of equivalence.
Do not implement the full GPLHR algorithm merely to justify the old name.
Dense nearest-root selection is the supported reference for small problems;
for larger interior problems this increment does not designate an iterative
production default. The existing JD implementation is also experimental.

The published full recurrence uses a locally bounded trial space containing
current Schur vectors, preconditioned residual blocks, higher residual-like
blocks and retained directions. Its harmonic Schur extraction handles general
pencils and targets eigenvalues near a shift without exact shift inversion.
The local growing-space variant shares ideas but not that recurrence. See
[Zuev et al.](https://doi.org/10.1002/jcc.23800) and
[Vecharynski et al.](https://doi.org/10.1137/15M1027413).

The seeded `n=60,120,300`, coupling `0.01,0.05`, sigma `0.4248` reproduction
remains runnable in `examples/experimental/interior_eigensolvers.py`. The
hard case must retain its failure or expose an independently verified root;
loosening tolerances or accepting a nearby converged eigenpair is not a fix.

## Migration increments and acceptance

1. Record the design on the tracker before broad edits. Done in #472.
2. Establish snapshots, capabilities, residual/metric diagnostics and native
   real/complex solve boundary. Preserve method defaults and ownership.
3. Integrate molecular RHF/RKS/UHF/UKS, native Bloch and periodic SCF, the
   shared Python multi-k Ewald/GDF helper, DFTB0/UDFTB0, and iterative TDA.
4. Validate independent spectral references and representative physical
   consumers. The unit and end-to-end commands are in the user guide.
5. Integrate the Gamma Ewald and GDF HF/KS helpers through the internal
   `FockDiagonalizer` adapter. It retains method-owned X/S and size thresholds,
   snapshots options without mutating callers, checks occupied-space coverage,
   and gives each unrestricted spin its own recycled subspace.
6. Share the complex Davidson iteration between explicit and callback inputs,
   and dispatch real/complex block callbacks without reconstructing A.
7. Integrate Gamma GPW/GAPW RHF/RKS/UHF/UKS and range-separated GAPW
   Fock solves, including final external-XC canonicalization. Preserve the
   existing full-rank orthogonalizers, solver defaults, spin populations and
   physical support gates. Compare spectra independently with generalized
   LAPACK and converged SCF energies/densities with the previous dense solve.
8. Integrate multi-k GPW RKS/UKS/ROKS, GAPW RKS and fixed-density band paths
   through stateless dense solves. Keep AO and projected DIIS coordinate
   frames distinct. Preserve global filling, restricted-open-shell orbital
   ordering, method-owned overlap cutoffs and physical refusal gates.
9. Integrate the real Roothaan SCF and Guest-Saunders Fock subblocks. Preserve
   occupations, fractional shells, level-shift exits and separate block
   ordering. Check dense/Davidson parity, option ownership, metric rank
   reduction and downstream correlated references.
10. Integrate Brueckner, FNO and reference UMP2/ROHF-MP2 semicanonical Fock
    blocks. Check frozen/occupied/virtual subspace invariance, empty spin
    partitions, off-diagonal Fock and DF tensor rotations, independent spectra,
    correlated energies and FNO setup lifetime at the native boundary.
11. Integrate dense DLPNO-MP2/UMP2 PNO Fock blocks and the shared experimental
    CCM caller. Check pair/spin metadata, retained density subspaces, threshold
    and minimum-rank policy, degenerate virtual gauges, pair energy corrections
    and coupled amplitudes against the previous solve and canonical limits.
12. Admit generalized sparse/callback Davidson with explicit X and its reduced
    diagonal. Check real/complex spectra against generalized LAPACK, original
    and projected residuals, retained-rank fallback refusal, scratch reuse,
    copied frame data and frozen factorized actions across build generations.
13. Integrate Gamma Ewald ROHF effective-Fock solves. Check dense/Davidson
    policy and input ownership, shifted generalized spectra, rectangular
    retained frames, complete-spectrum refusal, alpha-energy occupation
    selection and converged parity with the previous iteration.
14. Continue remaining Fock-family migrations with energy, occupation and
    degeneracy checks. Do not replace specialized CI, response,
    metric-factorization or bounded-memory kernels mechanically.

The [consumer inventory](operator_consumers.md) records the status and reason
for each audited source family and lists static call sites. Deferred migrations
are explicit remaining work, not an assertion that the whole repository is
unified. The [literature assessment](eigensolver_literature.md) separates
published evidence from proposals. User-facing diagnostics are formatted in
`vibeqc.output.eigensolver`; solver modules write no user files. Full SCF log
and QVF diagnostic schema integration is a later compatibility increment.
The portable formatter retains operator frame, generation, spin, k-point,
symmetry and occupation annotations in a nested `operator_metadata` record.
Its existing flat `operator_kind` field is preserved, and absent annotations
remain null rather than being inferred from the eigenvalues.


Operator metadata stores k points as three finite real scalars and occupation
annotations as a one-dimensional finite real scalar tuple. Nested and complex
arrays are refused; caller-owned vectors are copied into immutable scalars.
Empty occupation tuples remain valid, and fractional occupations are retained
without changing any method-owned filling or spin policy.


Root-tracking `subspace_overlap` validates a supplied full metric's shape,
finiteness, Hermiticity and positive semidefiniteness before computing principal
cosines. Separate normalization of the two subspaces cannot qualify an
indefinite metric. The default absolute/relative negative-eigenvalue allowance
is 1e-12, as for owned frames. Complex rotations within a degenerate cluster
and metric-null components retain their quotient-space interpretation; this
check does not reorder orbitals or occupations. A supplied dense metric incurs
one eigenvalue check per comparison.

Warm starts with a supplied overlap use `X^H S C` to project seed orbitals
into the retained metric space. This preserves the projection under changes
of AO coordinates, including seeds outside the retained space. With X alone,
Euclidean least-squares coordinate recovery remains the available convention.
This changes only the initial seed, not occupations or root-selection policy.

The legacy `GeneralizedEigenProblem` reduction also checks the supplied
overlap's declared shape, finiteness and Hermiticity before diagonalization.
A valid lower triangle does not qualify an invalid full metric. Positive
semidefinite rank reduction and its existing cutoff remain unchanged.

For an explicit generalized Hamiltonian, shape, finiteness and Hermiticity
are checked in the original AO space before reduction. A valid retained block
does not qualify invalid matrix elements in discarded directions. This check
does not probe callbacks or change the retained-rank residual convention.

The legacy generalized adapter copies each callback action before requesting
the next column when reporting original-space residuals. Callbacks may reuse
workspace between calls without aliasing the residual columns; no additional
operator applications or dense operator reconstruction are required.

Legacy generalized reduction rechecks complex capability after applying the
overlap transformation. A real original Fock does not imply real retained
coordinates. Real-only backends refuse such reductions before dispatch;
complex-capable backends retain their existing behavior.

Generalized callback actions must return a finite vector of the declared AO
dimension. The adapter validates and owns each action before both retained
projection and original-space residual reporting, refusing malformed output
instead of allowing broadcasting or nonfinite diagnostics.

Legacy generalized results qualify backend convergence against independently
computed retained-space residuals and metric orthogonality (1e-7 tolerance).
The returned `converged` flag can downgrade a backend claim but cannot promote
a backend failure. `n_converged` counts residual-qualified roots. Original AO
residuals remain reported separately and need not vanish outside a truncated
retained space. No additional callback evaluations are used.

The legacy generalized adapter requires exactly the requested eigenpair count
and matching retained-space vector shape before back-transformation. A default
request means all retained roots, not all original AO directions. Partial or
extra backend roots cannot be reported as an accepted complete result.

With a real overlap, a generalized callback can still represent a complex
operator. Real-only backends reject callback actions with nonzero imaginary
components before numeric conversion. Complex storage with exactly zero
imaginary components remains valid; complex-capable backends retain the full
action. This check does not probe additional vectors.

Generalized solves require a finite, positive residual tolerance before any
backend dispatch or callback evaluation, matching the strict operator API.
Infinity cannot disable residual qualification; NaN, zero and negative
tolerances are invalid requests rather than unconverged calculations.

Generalized warm-start vectors are validated in AO coordinates before
least-squares reduction: a finite two-dimensional block with the declared AO
row count and at least the requested number of columns. Default requests use
the retained rank. Valid seeds and caller options remain unmodified.

SCF convenience adapters interpret a default full-spectrum request in the
retained rank of X, rather than the original AO dimension. Explicit solver
root counts also determine how many previous orbital columns are recycled.
This aligns rectangular-X solves and full-spectrum warm starts with the shared
operator contract; it does not change method-owned occupations.

Generalized root requests must be nonnegative integers and cannot exceed the
retained overlap rank. Zero continues to request all retained roots. These
checks run before backend dispatch or callback evaluation, rather than relying
on native coercion or rejecting an inconsistent result afterward.

Generalized warm starts obey the same complex capability rule as strict
operators: real-only backends refuse complex seed arrays before conversion,
including complex storage with zero imaginary components. Complex-capable
backends preserve phase and retained-space seed information.

The generalized adapter uses the shared solver capability table to admit
warm-start requests. Backends without warm-start support refuse supplied
seeds before reduction or dispatch instead of silently ignoring them.
Davidson and Hermitian Davidson continue to accept supported seeds.

Legacy dispatch validates the effective interior shift before backend work.
An explicit `SolverOptions.sigma` overrides the interior descriptor shift;
otherwise the descriptor supplies it. NaN and infinite effective shifts are
refused without mutating caller options. Valid selection semantics are unchanged.
