# χ-CCM finite-torus space-group symmetry

Status: experimental, diagnostic only. The default is `off` and is unchanged.

## Scope

The χ-CCM (`aiccm2026dev-b`) cyclic cluster has a finite translation group. Crystal
space-group operations can reduce quantities on that group only if they also
preserve the chosen cyclic-cluster shape. This implementation obtains the
three-dimensional space group, Wyckoff letters, site-symmetry symbols, and
fractional operations from spglib. It then constructs the exact compatible
subgroup, atom and cell mappings, and reciprocal-net orbits.

No integral is skipped in this increment. `symmetry_mode="diagnostic"` runs
the unchanged full-net SCF and attaches a symmetry witness to the result.
`symmetry_mode="integrals"` raises `NotImplementedError`. The latter guard is
intentional: AO sewing matrices at general k and shell-quartet petite-list
scattering are not yet validated against the full build.

## Restricted native snapshot bridge

D130 (`build_aiccm2026dev_b_restricted_snapshot` in
`vibeqc.periodic.chi.symmetry`) connects a full-character chi RHF result to
the shared immutable native restricted mean-field state. Required keyword
arguments are `declared_calculation_identity` (a lowercase SHA-256-shaped
caller provenance declaration), `frozen_core_bands` (explicit zero-based
occupied indices) and `minimum_band_gap_hartree` (finite and positive).
The initial envelope is all-electron, zero-smearing 3D four-center RHF with
at least one correlated occupied and one retained virtual orbital per k.
RI/RIJCOSX, RKS, open-shell, ECP and lower-dimensional inputs are not admitted.

The bridge forwards the actual recorded k coordinates and weights in their
original order, and the original Fock, overlap, coefficients, energies and
occupations. Missing blocks are not synthesized. It does not diagonalize or
repair F, S or C. Inputs must be representable in binary64; the native weight
gate canonicalizes accepted weight roundoff to exact uniform `1/Nk`, as on
the shared route. Exact frozen indices, identical at every k, select masks
even within a degenerate manifold; energy cutoffs do not substitute for them.
The native state's fixed metric, Hermiticity, occupation, electron-count,
Roothaan, ordering and insulating-gap checks run unchanged. Consequently
ordinary SCF convergence with looser thresholds is not sufficient admission.

The factory-only result retains `state`, `finite_torus_convention`,
`declared_calculation_identity` and `estimated_snapshot_bytes`. Its
`physical_source_symmetry_certified` property is always false. The native
calculation digest hashes the versioned bridge declaration, supplied caller
identity, complete chi convention and sorted frozen selection. Native payload
and state digests additionally bind the actual matrices/masks and native
validation contract. This is content identification, not authentication of
the caller's source, basis or threshold declarations. No source can be
relabelled as chi merely because the native numerical gate passed.

Before copying a payload or constructing native input, the bridge checks
512-k/256-AO diagnostic bounds and the explicit numerical snapshot cap
(`max_snapshot_bytes`, default 256 MiB). For `K` points, `b` AOs, `r` retained
orbitals and native resident-state estimate `R`, the conservative additional
explicit numerical inventory is
`3 R + 16 (12 b^2 + 12 b r + 12 r^2) + 64 K (r+4)` bytes. This excludes
caller-owned inputs, basis metadata, Python/control/allocator overhead and
BLAS internals; it is not an RSS or complete application memory guarantee.
The downstream native sewing leaf has stricter tiny-binding limits and its
own explicit resource inventory, which must be admitted independently.

`snapshot.state` is accepted by the internal
`_make_periodic_orbital_sewing(state, basis, system, operation, source_index,
time_reversal, options, inventory, caps)` diagnostic. That shared leaf
retains full complex frozen/active/virtual matrices and tests transported
Roothaan equations and mask closure without relaxing the specified budgets.
Its one-operation witness is not a whole-group or density-response proof;
it does not authenticate the physical chi integral source or enable
representative-only CCSD(T). No production correlation gate consumes the
new wrapper, and analytic total gradients remain gated.

Actual two-/three-character He2/6-31G chi four-center fixtures test identity,
scalar time reversal and bond-centred inversion sewing. Single frozen bands
are checked against the actual C/S projection: selecting one member of a
degenerate pair need not give an invariant subspace, and mask leakage must
be refused. The whole occupied space is a different scope.

D130 exposed issue #704: on diagonal `(8,16,16)` bohr, He at x=0 and 4,
the geometric half-cell translation had nonzero-k Fock covariance residuals
`5.97e-6` / `5.02e-6` Ha for two/three characters, despite original Roothaan
residuals below `2e-11`. D131 isolates the dominant term to raw V_ne erfc
support: at nuclear radius 15 its two-character covariance is `5.46e-6`
and non-Hermiticity `1.65e-6`; the reciprocal nuclear residual is below
`5e-10`. Extending only the nuclear sum removes that defect without matrix
averaging, alpha changes or output-cell widening.

For primitive product exponent p=a+b and screening alpha, Saunders et al.
(1992), Eq. (66), gives the direct prototype factor
`[erf(sqrt(p)*d)-erf(sqrt(q)*d)]/d`, with
`q^-1=p^-1+alpha^-2`. D131 pads the nuclear cell-origin radius by
`R_output + max_shell,nucleus |O_shell-R_nucleus|` and uses
`erfcinv(tolerance)*sqrt(1/(2*a_min)+1/alpha^2)` for its smeared range,
retaining any larger caller radius and the previous point-charge bound.
Relative positions make this independent of a common origin translation.
Angular polynomials, contraction amplitudes and summation multiplicities
still need convergence checks: this is not a total-error certificate.
The energy and analytic nuclear-attraction derivative share this image rule.

After this partial repair, three-character F covariance is below `1e-9`
and the native `1e-8` half-translation sewing gate passes. Two characters
retain about `1.64e-7` Ha and still fail that same gate. Issue #704 remains
open; no full-source certificate or representative-only energy is enabled.
These are implementation checks, not absolute-energy benchmarks.

This integration uses the shared Dovesi (1986), Eqs. (37)-(43), and
Casassa et al. (2006), Sec. 2 matrix-valued orbital action contracts already
cited by the native sewing primitive. It does not introduce a new electronic
method, normalization, empirical default or separate SALWF construction.

D127 additionally exposes a compact action of one finite-mesh-compatible
spatial operation on real-torus AO coefficient rows. This library utility is
independent of the SCF symmetry mode and remains diagnostic: no SCF or
correlation path consumes it.

## Finite-cluster compatibility

Let the primitive translation lattice be

\[
\Lambda = \{A\mathbf r : \mathbf r\in\mathbb Z^3\},
\tag{1}
\]

where the columns of \(A\), measured in meters, are the primitive lattice
vectors. A diagonal Born-von Karman cluster with
\(N=\operatorname{diag}(N_1,N_2,N_3)\) identifies translations modulo

\[
\Lambda_N = \{AN\mathbf z : \mathbf z\in\mathbb Z^3\}.
\tag{2}
\]

A space-group operation is written \(g=\{W\mid\mathbf w\}\), where \(W\) is
integer-valued in fractional coordinates and \(\mathbf w\) is fractional.
It descends to an automorphism of \(\Lambda/\Lambda_N\) exactly when

\[
W\Lambda_N=\Lambda_N.
\tag{3}
\]

For an invertible crystallographic \(W\), Equation (3) is equivalent to

\[
N^{-1}WN\in\mathbb Z^{3\times3}.
\tag{4}
\]

Indeed, \(WN\mathbf z=N\mathbf z'\) must hold for every integer
\(\mathbf z\), so \(\mathbf z'=N^{-1}WN\mathbf z\) is integer for every
integer input. Conversely, integrality of this matrix maps every generator of
\(N\mathbb Z^3\) back into that sublattice. Since \(W^{-1}\) is another
space-group operation and satisfies the same condition in the compatible
set, the inclusion is an equality. The implementation checks Equation (4)
with integer divisibility, not floating-point rounding.

An anisotropic cluster may therefore retain only a subgroup of the crystal
space group. This is mathematically valid and is reported explicitly. Setting
`symmetry_require_full_group=True` instead refuses such a shape.

## Atom and cell mapping, including screw and glide translations

For fractional atom coordinate \(\mathbf f_a\), element-preserving atom
matching determines a unique atom \(p_g(a)\) and integer vector
\(\mathbf q_{g,a}\) from

\[
W\mathbf f_a+\mathbf w
=\mathbf f_{p_g(a)}+\mathbf q_{g,a}.
\tag{5}
\]

The residual is evaluated in Cartesian distance,

\[
\epsilon_{g,a}
=\left\|A\left(W\mathbf f_a+\mathbf w
-\mathbf f_{p_g(a)}-\mathbf q_{g,a}\right)\right\|_2,
\tag{6}
\]

and must not exceed the requested spglib tolerance. The public vibe-qc API
expresses that tolerance in bohr; the corresponding SI distance is
\(a_0\epsilon_{g,a}\), with \(a_0\) the Bohr radius in meters.

An atom in cyclic cell \(\mathbf r\) maps as

\[
(a,\mathbf r)\mapsto
\left(p_g(a),\;W\mathbf r+\mathbf q_{g,a}\pmod{\mathbf N}\right).
\tag{7}
\]

Keeping \(\mathbf q_{g,a}\) is essential for screw and glide operations.
Discarding it gives correct-looking atom permutations but wrong image-cell
labels.

The plan uses wrapped primitive-cell representatives. If the supplied system
instead stores atom \(a\) at
\(\widetilde{\mathbf f}_a=\mathbf f_a+\mathbf n_a\), with
\(\mathbf n_a\in\mathbb Z^3\), the AO-action factory recovers the integer
reference-cell offset and uses

\[
 \widetilde{\mathbf q}_{g,a}
 =\mathbf q_{g,a}+W_g\mathbf n_a-\mathbf n_{p_g(a)}.
\]

The action payload fingerprints both \(\mathbf n_a\) and this corrected shift.

## Compact real-torus AO action

Let primitive AO \(\mu\) belong to atom \(a(\mu)\), and let
\(P^g_{\nu\mu}\) be the real-solid-harmonic AO action with destination rows
and source columns. D127 defines the full finite-torus action by

\[
 [\mathcal U_g]_{(\mathbf S,\nu),(\mathbf R,\mu)}
 =P^g_{\nu\mu}\,
 \delta_{\mathbf S,\,
 W_g\mathbf R+\widetilde{\mathbf q}_{g,a(\mu)}
 \pmod{\mathbf N}} .
\tag{8}
\]

Equation (8) uses the active `+q` scatter fixed by Equation (5) and its
reference-offset correction above. The implementation
stores only \(P^g\) and the destination-cell table for each source cell and
atom. Its retained action storage is therefore
\(O(n_{\rm AO}^2+N_c n_{\rm atom})\), rather than the dense action's
\(O((N_c n_{\rm AO})^2)\). The mesh, operation, AO action, atom shifts, AO
ownership, and cell map are immutable and share one SHA-256 fingerprint.
The fingerprint identifies that compact action payload, not a complete
system, radial-basis, localized-state, run, build, or stable spglib-operation
identity. Operation indices are local to the plan's enumerated order.

The factory recomputes the Cartesian atom-mapping residual and independently
caps it at \(10^{-5}\) bohr. A plan records its observed residual but not the
`symprec` that admitted it, so a diagnostically valid plan built with looser
`symprec` may contain an operation that this action factory rejects.
Finite-mesh compatibility alone does not guarantee action admission.

The factory validates basis-shell origins and exact ordered radial-shell
equality across mapped atoms, although the radial exponents and contraction
coefficients are intentionally absent from the payload fingerprint. The
supported angular envelope is scalar orbital AOs with spherical
\(l\geq1\) shells. Cartesian \(p\)-and-higher shells fail closed; the
Cartesian/spherical distinction is immaterial for \(s\). Applying the action
transforms AO rows only; it does not transform localized-orbital columns or
establish their own reference-cell offsets, centres, phases, or local domains.
Execution uses atom-block-sized temporaries in addition to the returned
matrix, while never materializing the dense torus action.

For affine operations \(g\) and \(h\), with \(h\) applied first,

\[
 W_{gh}=W_gW_h,\quad
 \pi_{gh}=\pi_g\circ\pi_h,\quad
 \widetilde{\mathbf q}_{gh,a}
 =W_g\widetilde{\mathbf q}_{h,a}
 +\widetilde{\mathbf q}_{g,\pi_h(a)},\quad
 P_{gh}=P_gP_h.
\tag{9}
\]

Representatives whose fractional translations are stored modulo primitive
lattice vectors acquire an integer translation factor. If \(k\) is the stored
representative with \(W_k=W_gW_h\), define

\[
 \boldsymbol\ell_{gh}=W_g\mathbf w_h+\mathbf w_g-\mathbf w_k
 \in\mathbb Z^3.
\]

The corresponding representative law is

\[
 \mathcal U_g\mathcal U_h
 =T_{\boldsymbol\ell_{gh}}\mathcal U_k.
\]

This is the primitive-translation cocycle of the chosen coset section. D127
retains the atom shifts needed to derive it, and a diamond fixture pins one
nonzero instance of this law, but D127 does not expose or certify the complete
group table. Time reversal is antiunitary and is not treated as another real
linear action.

Under χ localization's
\(e^{+2\pi i\mathbf k\cdot\mathbf R}\) convention, Equation (8) sends
\(\mathbf k\) to \(\mathbf k'=W_g^{-T}\mathbf k\) and implies the sewing phase

\[
 B_g(\mathbf k)_{\nu\mu}
 =P^g_{\nu\mu}
 e^{-2\pi i\mathbf k'\cdot
 \widetilde{\mathbf q}_{g,a(\mu)}}.
\tag{10}
\]

The next χ symmetry gate must use this action with the actual torus overlap
and localized occupied coefficients, including their column/reference-offset
map. It must distinguish signed/phased monomial gauges, which may eventually
feed D126's scalar support permutations, from general matrix-valued
petal/flower representations, which require matrix-valued domain and amplitude
transport. D127 neither emits/tests the Bloch sewing matrix nor performs
Casassa's a posteriori SALWF construction and petal/flower/bunch
classification.

The small dense fixtures reuse the trusted primitive AO rotation and
independently materialize only the torus scatter/factorization. The skew-cell
metric fixture is synthetically symmetrized under that action; it is not the
physical χ torus overlap. Both distinctions remain part of the next
localized-occupied covariance gate.

## Occupied-space action

D128 connects the compact D127 AO action to an
`AICCM2026DevBLocalizationResult` through
`build_aiccm2026dev_b_occupied_symmetry_action(action, localization)`.
For overlap-orthonormal occupied coefficients C and their supplied overlap S,
the occupied action has destination rows and source columns:

```text
D_g = C^dagger S U_g C
U_g C = C D_g
D_g^dagger D_g = I
```

The factory checks the cell order, positive-definite Hermitian metric,
`U_g S U_g^dagger = S`, occupied orthonormality, closure and unitarity.
It does not repair or symmetrize C or S. Metric residuals are Frobenius norms
relative to `||S||_F`; closure is relative to `max(||C||_F, 1)`;
orthonormality, unitarity and monomial residuals are absolute Frobenius norms.
The default tolerance is `1e-8`, with admitted values in `(0, 1e-6]`.

Casassa et al. (2006), DOI
[10.1007/s00214-006-0119-z](https://doi.org/10.1007/s00214-006-0119-z),
Eq. (3) permits matrix-valued orbital mixing; Eq. (6) identifies the favorable
single-petal sign-permutation case. Here arbitrary column phases generalize
the latter to a phase-permutation matrix. A unique dominant destination in
each column is admitted only if the full Frobenius distance to that matrix
passes the tolerance. The computed D matrix is always retained without
rounding. General mixing returns `permutation=None` and `phases=None`.
No SALWF generation, petal/flower partition or irreducible decomposition is
performed. The result's `citation_numerics` tokens can be passed to the shared
citation assembler by explicit diagnostic callers.

The immutable result binds the D127 action fingerprint, numerical C/S input
digest, computed occupied matrix and tolerance. It does not authenticate the
full SCF/system/basis/build provenance, stationarity or localization
optimality. Independent per-operation checks do not constitute a complete
group or nonsymmorphic cocycle seal. No result is fed to D126 or an energy
solver; occupied reference offsets, PAO/auxiliary/PNO/TNO and amplitude maps
remain required before reduction is enabled. Analytic gradients are unchanged.

The diagnostic consumes the existing dense localizer payload. With
`n = N_c nbf` and `m = N_c nocc`, admission reserves
`16 * (8*n*n + 12*n*m + 12*m*m)` bytes for explicit numeric arrays before
copying inputs or applying the AO action. The default cap is 512 MiB. This
conservative inventory excludes caller inputs and BLAS/LAPACK internal work;
it is not an RSS bound or an MgO 8-cubed resource estimate. The retained
occupied matrix is still quadratic in m. A future production transport must
use home-cell blocks and independently admitted native resources.

## Spatial group and occupied cocycle

D129 adds `build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan)`.
It uses all supplied compatible spatial operations, not the plan's time-reversal
flag. A supplied subgroup is permitted if closed; this does not certify that
the supplied list exhausts the crystal space group. There is one representative
per coset modulo primitive translations, with a zero-translation identity.
For destination/source actions (h acts first), the output stores:

```text
products[g,h] = k
lattice_cocycle[g,h] = ell
U_g U_h = T_ell U_k
q_gh,a = W_g q_h,a + q_g,pi_h(a)
ell(g,h) = q_gh,a - q_k,a             (independent of atom a)
ell(g,h) + ell(gh,j) = W_g ell(h,j) + ell(g,hj)
```

Products, quotient inverses, associativity and the twisted integer cocycle are
checked for every pair/triple. The whole-cell shifts are not discarded modulo
the finite mesh. A quotient inverse can therefore compose to a nonzero
translation on the torus. Atom reference-cell offsets cancel in the cocycle;
changing the Seitz coset representatives instead changes it by the expected
integer coboundary. Primitive AO products and Cartesian Seitz products have
independent numerical residual checks (Frobenius and bohr respectively), with
default tolerance `1e-8`, admitted range `(0, 1e-6]`. This supplements the
atom-image transformations described by Dovesi (1986), Eqs. (25)-(43), DOI
[10.1002/qua.560290608](https://doi.org/10.1002/qua.560290608).

`build_aiccm2026dev_b_occupied_group_witness(group, localization)` projects all
spatial operations through D128 and translations through the same C/S. It
checks metric covariance and occupied closure/unitarity for translations,
`D(T_a)^N_a = I`, commuting translation generators,
`D_g D(T_a) = D(T_Wg*a) D_g`, identity and every spatial product including its
lattice translation. General orbital mixing remains a matrix, not a scalar
pair permutation. Full group-law residuals are absolute Frobenius norms;
individual metric and closure residuals follow the D128 normalization above.

Tests compare primitive-diamond group products with independent AO scatters
and native Seitz Bloch transport. For the positive AO Bloch sum convention,
composition carries `exp(-2*pi*i*k_target.ell)` exactly once. Two- and
three-point axes test real and genuinely complex phases; these are transport
tests, not correlated energies. A physical three-cell He2 occupied calculation
pins a half translation whose square is a whole-cell translation, including
after an arbitrary complex occupied gauge change.

The group retains compact D127 actions and `O(|G|^2)` tables, not a table over
all `N_c |G|` placed operations. It admits at most 192 spatial cosets, defaults
to 256 MiB explicit-array inventory and 10 million checked pair/triple relations.
The occupied witness defaults to 512 MiB and 100,000 group-product relations;
its dense C/S and retained `|G|` occupied matrices are still small-system
diagnostics. Relation caps are not floating-operation or elapsed-time bounds;
byte inventories exclude caller inputs, Python-object overhead, allocator and
BLAS/LAPACK internals and are not RSS guarantees. The occupied inventory also
reserves the group's inventory and retained phase/permutation metadata.

Both results are immutable and provide `citation_numerics` tokens. Their hashes
bind enumerated action payloads, computed tables/matrices and tolerances, not
a complete physical system/basis/build provenance. Neither checks Fock
stationarity, frozen/active/virtual separation, antiunitary sewing or
PAO/auxiliary/PNO/TNO/amplitude covariance. No result enters the D126 pair
quotient or an energy solver yet. Full symmetry-reduced multi-k DLPNO-CCSD(T)
and analytic total gradients remain unfinished.

## Reciprocal-net orbits

The reciprocal characters of the cyclic translation group are

\[
\mathbf k_{\mathbf m}
=\left(m_1/N_1,m_2/N_2,m_3/N_3\right),
\quad 0\le m_i<N_i.
\tag{11}
\]

The reciprocal action follows from phase preservation,
\(\mathbf k'\mathbin{\cdot}W\mathbf r
=\mathbf k\mathbin{\cdot}\mathbf r\), and is therefore

\[
\mathbf k'=W^{-T}\mathbf k\pmod{\mathbb Z^3}.
\tag{12}
\]

Equation (4) guarantees closure of Equation (12) on the finite net. For a
closed-shell nonmagnetic reference, time reversal additionally identifies
\(\mathbf k\) and \(-\mathbf k\). If orbit \(\mathcal O_\alpha\) has size
\(|\mathcal O_\alpha|\), its exact integration weight is

\[
\omega_\alpha
=\frac{|\mathcal O_\alpha|}{N_1N_2N_3},
\qquad \sum_\alpha\omega_\alpha=1.
\tag{13}
\]

The diagnostic records these orbits but SCF continues to use all
\(N_1N_2N_3\) points. Thus enabling diagnostics cannot change the energy.

## Safe Gamma projection

At Gamma all cell-translation Bloch phases are one. The existing vibe-qc
real-solid-harmonic Wigner matrices and atom permutation then define a real
orthogonal AO action \(U_g\). An AO matrix can be projected by the Reynolds
operator

\[
\mathcal P_G[M]=\frac{1}{|G_N|}\sum_{g\in G_N}U_gMU_g^T.
\tag{14}
\]

Group closure proves idempotency:

\[
\mathcal P_G[\mathcal P_G[M]]
=\frac{1}{|G_N|^2}\sum_{g,h}U_{gh}MU_{gh}^T
=\mathcal P_G[M],
\tag{15}
\]

because every product occurs exactly \(|G_N|\) times. The implementation
exposes Equation (14) as a diagnostic utility and reports
\(\max_g\|U_gMU_g^T-M\|_F\) for Gamma Fock and density matrices. It does not
insert the projection into DIIS or SCF.

At general k, the nonsymmorphic translation and the atom-dependent image-cell
shift in Equation (5) generate Bloch sewing phases. Equation (10) derives the
phase for χ localization's declared Bloch-sum convention, but D127 does not
implement or test that sewing matrix; other backends may use different
Bloch-sum conventions. Omitting or transplanting a phase without matching its
convention breaks the space-group representation. General-k AO symmetrization
therefore remains blocked until the exact executing backend implements its
sewing matrix and passes full-net Fock and energy parity tests.

## Shell-pair and quartet orbit diagnostics

For each operation, primitive shells map by atom permutation while retaining
their within-atom shell slot and angular momentum. The diagnostic partitions
intrinsic-unique shell pairs

\[
 (a,b)\equiv(b,a)
\]

and intrinsic eightfold shell quartets

\[
 ((a,b),(c,d))
 \equiv ((b,a),(c,d))
 \equiv ((c,d),(a,b))
\]

into point-group orbits. Every pair or quartet must occur in exactly one
orbit. The SCF output reports representative/full counts. Explicit quartet
enumeration is skipped above 24 primitive shells to avoid a diagnostic
fourth-power memory cost.

These are mapping tables, not an accelerated integral build. A
representative quartet cannot be scattered at general k using only its shell
orbit: the AO Wigner blocks, nonsymmorphic sewing phases, density transform,
and orbit stabilizer must all enter the contraction. `integrals` therefore
continues to fail closed.

## Usage

```python
result = vibeqc.run_aiccm2026dev_b_rhf(
    system,
    basis,
    mesh=(2, 2, 2),
    backend="ri",
    symmetry_mode="diagnostic",
)

symmetry = result.aiccm2026dev_b_symmetry
print(symmetry.plan.international_symbol)
print(symmetry.plan.n_operations_compatible)
print(symmetry.plan.n_kpoints_irreducible)

from vibeqc.periodic.chi.symmetry import (
    build_aiccm2026dev_b_real_torus_ao_action,
)

action = build_aiccm2026dev_b_real_torus_ao_action(
    system, basis, symmetry.plan, operation_index=0
)
transformed = action.apply(real_torus_coefficients)
```

The SCF symmetry diagnostic that supplies `symmetry.plan` is requested through
the standard front door with `method="aiccm"`, `variant="chi"`, and
`aiccm_symmetry="diagnostic"`. The compact AO action itself is a low-level χ
library utility and is not consumed by runner execution. The dev-era
`jk_method="aiccm2026dev-b"` spelling remains only as a deprecated
compatibility selector.

## Decisions and open questions

1. The feature is off by default. The off path does not invoke spglib or add
   result attributes.
2. A compatible subgroup is accepted because it is an exact symmetry of the
   finite torus. Users can require the full group explicitly.
3. Three-dimensional spglib is not applied to wires or slabs. A vacuum-padded
   3D classification is not a rod or layer group, so 1D and 2D diagnostics
   fail closed pending the correct group treatment.
4. Fractional translations and atom-dependent lattice shifts are retained in
   every mapping table. Nonsymmorphic operations are not filtered out.
5. Gamma AO projection is available for validation only. It does not alter
   the SCF fixed point or energy.
6. General-k AO sewing, symmetry-unique shell pairs, shell quartets, and
   libint representative scattering remain open. The `integrals` option is a
   hard error until an enabled-versus-disabled parity test passes for Fock and
   total energy.
7. Degeneracy classification by little-group irreducible representations is
   not yet implemented. The present k-orbit reduction alone does not label
   bands by irreducible representation.
8. Compact real-torus AO actions are available as immutable diagnostic
   primitives. They do not attest a localized occupied gauge, transform local
   orbital columns or their reference offsets, transform local domains, reduce
   pair work, or enable `symmetry_mode="integrals"`. Their fingerprint is not
   a complete system, radial-basis, localized-state, run, or build fingerprint.
