---
orphan: true
---

# Design, native vibe-qc GDF (Method 1 of the periodic JK roadmap)

## Range-separated source contract (2026-09)

The original design below is historical. The bulk 3D `rsgdf` selector now
builds a real-space short-range fit plus a reciprocal long-range fit using
`cpp/src/periodic_gdf_short_range.cpp`. Gamma and general-k drivers share
this source, its physical auxiliary basis and its zero-mode convention.
The old exponent-sized reciprocal repair is not used on this route.
Production integration acceptance is still in progress; the source tests
alone do not establish dense-cell SCF accuracy or whole-job memory bounds.

The 3D automatic one-electron image policy bounds the omitted overlap and
kinetic lattice sums using absolute Gaussian envelopes, including angular
polynomials and signed-contraction magnitudes. A positive overlap matrix
is not a convergence criterion: the independent NaCl/LANL2DZ reference has
a positive Gamma overlap at 15 bohr with a maximum matrix error of 0.315.
The resolved AO domain now reaches S/T, nuclear attraction, ECP blocks and
their force partners. Explicit flat image domains remain caller-controlled;
projector/nuclear-image cutoffs are separate and still need convergence.
The bound targets raw S/T errors of 1e-12 and does not certify SCF energy
accuracy or a well-conditioned basis.

Before image enumeration, setup reserves a conservative lattice-box bound
for retained and temporary one-electron matrices, Bloch matrices, nuclear
images and Fourier workspace. Dry-run uses the same calculation. This
replaces the old overlap-signature search, which allocated trial overlaps
before admitting the setup memory. Full-process memory and physical energy
acceptance remain separate numerical gates.


For `q = k_ket - k_bra`, the short-range three-center source is

```text
T_SR(P,mu,nu;k_bra,k_ket)
  = sum_(R,T) exp(+i k_ket.R - i q.T)
      (P_T | erfc(omega*r)/r | mu_0 nu_R).
```

Both translations are independent. Anchoring both P and mu while summing
only nu does not produce the periodic three-center integral, even at
Gamma. This is Ye and Berkelbach, JCP 154, 131104 (2021),
doi:10.1063/5.0046617, Eqs. (12)-(13), after translating the first AO to
the home cell. The two-center metric uses
`sum_T exp(+i q.T) (P_0 | erfc(omega*r)/r | Q_T)`.

The long-range term uses unnormalized Fourier integrals and the weight
`(4*pi/volume) exp(-|G+q|^2/(4*omega^2))/|G+q|^2`. Its singular mode is
omitted. At reciprocal-equivalent q=0, subtract the finite short-range
zero mode exactly once: `pi/(volume*omega^2)` times the auxiliary charges
for the metric, or auxiliary charge times the Bloch AO overlap for the
three-center tensor. No exchange finite-size correction is included in
these integral sources.

The replacement source uses the same finite AO-pair image domain in SR,
LR and the zero-mode overlap. AO-center separation is bounded by the
explicit pair cutoff; the auxiliary SR image domain is bounded by its
distance to the segment joining the AO centers. Every primitive Gaussian
product center lies on that segment. These are finite-domain definitions,
not certified error estimates: both cutoffs must be converged.

Image candidates are streamed. Shell triples own disjoint output slices,
so real-space memory contains a shared-q output batch and a shell block
per worker, rather than a tensor per image pair or per worker. LR work
uses bounded AO-pair and reciprocal-vector panels. Pair tiles contain at
most 16 pairs and shrink to expose several tasks per worker on small cells;
a fixed 16-pair tile would expose only 11 tasks for 13 AOs at Gamma.
Its workers own disjoint
`(k, AO-pair)` slices and are limited by the available workspace. Metric
columns are independent work items; neither operation reduces full tensors
across threads.

With `A` auxiliary functions, `N` AOs and `K` momenta in a shared-q batch,
the output reservation is `16*(A*A + K*A*N*N)` bytes. Fourier workspace is
`16*A*(128+1) + 8*128 + workers*(16*16*128 + W_pair)`, where `W_pair` is
the fixed numerical workspace reported by the native AO-pair primitive.
The short-range phase admits the prototype engine, worker engines, shifted
shell copies and contracted shell blocks separately. Its estimate uses the
linked libint primitive-data type and recurrence-stack query, with a 64 KiB
per-engine allowance for parameter/evaluator allocations. The reduced-center
braket is supplied to the engine constructor: selecting it after construction
retains an unnecessary `nprim^4` primitive-data allocation in libint.

The SCF reservation queries `gdf_short_range_workspace_bytes` for the actual
orbital/auxiliary bases and requested thread count, including derivative
engines when forces are requested. The query and execution use one engine
memory calculation and limit workers by shell-task count. The source no
longer imposes a fixed 64 MiB ceiling on the SR team. It retains at least that
much requested workspace for Fourier panels; if the remaining process budget
is smaller, admission reduces the team before allocating engines. Dry-run
estimates also charge this linked-kernel requirement and the bounded XC phase,
using native grid point counts without building a grid.

These caps admit numerical arrays and an engine workspace estimate. Borrowed
bases, global library/runtime state and allocator overhead remain separate;
this is not a whole-process RSS limit. The source contract also does not by
itself bound the SCF driver's retained k-pair factor cache.

The private Python integration builder admits reciprocal enumeration, the
binding's Eigen copy of the reciprocal input, raw M/T, queried LAPACK
workspace, eigenvectors and whitened factors. Reciprocal points are counted
and filled in bounded lines without materializing the enclosing integer
box. The shared-q cache builder reserves the full retained factor cache
plus one source build before constructing pairs. With an exchange wedge of
`I` bras among `K` k points, the pair count is `I*K + K-I`, because Hartree
needs the remaining diagonal pairs too. This is a per-process numerical
budget with a bookkeeping allowance; it does not distribute or spool a
cache that exceeds the budget.

The accepted-state exchange contraction distributes complete bra k-point
rows across workers. Every row retains the full weighted ket sum. Workers
share one 32 MiB numerical workspace budget, subdivided before allocation;
adding workers does not multiply that allowance. MPI partitions the bra
rows before local threading and gathers them in the requested order. The
package defaults BLAS to one thread, so matrix-level threads do not compete
with this k-point parallelism. Whole-node scaling still requires measurement.

`examples/debug/gdf_multik_profile_hot_path.py` measures the actual production
cache, combined native source, exchange and XC calls in separate processes
for each requested thread count. It records wall time, process CPU time,
peak RSS, native core hash and accepted energy. Run it through `vq` on a
full node; `--threads 16 32 64 --require-wide-scaling` requires at least a
1.2-fold native-source speedup at both wider teams relative to 16 threads,
with energy spread below `1e-8` Ha. This explicit performance gate supplements
the independent energy/force references; it does not establish their accuracy.

Within an admitted factor cache, shared-q groups subdivide when their source
batch would exceed the remaining workspace budget. The first admitted batch
builds and diagonalizes M; later batches use the same eigenvectors/whitener
and ask the native source for T only. The source fingerprint and all physical
fit parameters must match before metric reuse. Only preallocation admission
failures permit subdivision. Numerical failures and actual allocation errors
propagate rather than being retried as smaller calculations. Subdivision
does not change a pair's primitive/image/reciprocal accumulation order.

Gamma and multi-k Hartree assembly use the same auxiliary Gram contraction:
`rho_P = sum_k w_k sum_mn conj(L_kPmn) D_kmn`, followed by
`J_kmn = sum_P L_kPmn rho_P`. The density indices are not transposed in the
first expression. Writing it as `tr(L_P D)` is equivalent only when each
individual factor is Hermitian. A complex unitary change of auxiliary
coordinates preserves the fitted operator but removes that property. The
shared contraction remains invariant and uses bounded auxiliary panels
instead of a conjugated copy of the entire cache. The exchange Madelung
correction remains a separate subsequent operation.

The optional short-range screen uses an absolute shell envelope, including
angular momentum and signed contractions. For a primitive polynomial of
order `l`, choose `a = min(alpha)/2` (`a = min(alpha)` for s shells), and use

```text
r^l exp(-alpha*r^2)
  <= [l/(2*e*(alpha-a))]^(l/2) exp(-a*r^2).
```

Each regular solid-harmonic component is bounded by `r^l`, by the spherical
harmonic addition theorem in libint's normalization. Summing absolute
contracted coefficients gives a positive Gaussian envelope. Interactions
between these envelopes bound the erfc integrals using

```text
[erf(eta1*R)-erf(eta2*R)]/R
  = (2/sqrt(pi)) integral_(eta2)^(eta1) exp(-t^2*R^2) dt
  <= (2/sqrt(pi)) (eta1-eta2) exp(-eta2^2*R^2).
```

For three-center integrals the Gaussian product contributes its additional
`exp(-beta*R_mu_nu^2)` factor. A requested `integral_screen_error` is divided
by a conservative census of the finite image domain (both independent
translations for three-center integrals), bounding the sum of omitted raw
contributions per tensor element. Zero disables the screen. This budget
covers screening within the stated cutoffs, not finite-cutoff tails or the
amplification of source error by metric inversion. The screen can shrink
auxiliary image enumeration to its Gaussian-envelope ball while preserving
the original capsule intersection.

Long contractions are evaluated in fixed four-primitive blocks. Each block
retains its original coefficients without renormalization. This bounds
libint's cubic primitive-data workspace per three-center engine independently
of the longest contraction; the workspace model separately charges shifted
shell and block storage. Changing worker count does not change block order.

The private weighted SR derivative sources reuse the same image traversal
and screening membership. They compute atomic derivatives of
`Re sum(weight * integral)` directly into per-worker atom gradients, with
no atom-by-integral derivative tensor. The derivative engine and accumulator
storage have their own admission estimate. Weights are borrowed in the
documented contiguous layout rather than copied by the binding. Discrete
cutoff/screen membership is held fixed during differentiation. The combined
weighted source also differentiates the reciprocal contribution and the
finite zero-mode overlap on the energy source's AO-pair distance domain.
It streams 128 reciprocal vectors and seven AO-pair components (value and
the two center derivatives), then contracts into atom gradients. Auxiliary
center derivatives multiply the Fourier value by `-i*p`; the conjugate
auxiliary in the three-center source therefore contributes `+i*p`.
The original normalized polynomial Gaussian is differentiated before
contraction, so raising angular momentum does not renormalize the function
(Helgaker and Taylor, 1992, doi:10.1007/BF01132826, Eq. 15).
The private fit builder can retain its original T, eigensystem, kept modes,
reciprocal vectors and integration parameters for differentiation. These
arrays share their existing storage; the source reservation includes their
overlap with whitening. A content fingerprint binds the state to the cell
and normalized orbital/auxiliary bases. The gradient rejects a changed
geometry or basis instead of contracting an old fit with new integrals.

The J/K response uses that exact state. With the thresholded metric inverse
`H = U h(lambda) U^H`, the Hartree objective is `0.5*v^H*H*v`, where
`v = sum_k w_k T(k) D(k)^T`. Exchange uses
`-0.25*sum_(i,j) w_i*w_j Tr[G(i,j)*H]` with the AO-density sandwich in
`G`. The divided differences of `h` include both retained and discarded
modes. The resulting unconjugated M/T weights feed the combined native
derivative, including its finite zero-mode overlap term. For unrestricted
densities, J uses the sum of spin channels and each spin exchange uses
twice the restricted prefactor. Native source and fit-response finite
differences have passed. Production drivers retain the source cache for
this derivative; full-SCF and moving-grid XC acceptance remain separate
checks of the assembled force.

The private cutoff planner applies the same envelopes to omitted domains.
For a lattice with smallest singular value `sigma`, a uniform bound valid
at any center is

```text
Theta(c) = sum_n exp(-c |A*n-r|^2)
         <= (1 + sqrt(pi/c)/sigma)^3.
sum_(|A*n-r|>R) exp(-c |A*n-r|^2)
         <= exp(-c*R^2/2) Theta(c/2).
```

The first inequality follows by replacing the lattice norm with
`sigma*|n-A^-1*r|`, separating the three integer sums, and bounding each
one-dimensional Gaussian sum by one plus its integral. Poisson summation
places its maximum at an integer shift. Splitting the exponent gives the
second inequality. Thus neither estimate assumes orthogonal cells or an
atom at the origin.

The planner bounds the SR metric tail, both independent SR three-center
tails, the LR reciprocal tail and the AO-image tail in the LR and zero-mode
overlap. An omitted auxiliary center lies outside the envelope product
center's ball because that center lies on the AO segment. For reciprocal
vectors outside radius `G`, it uses `1/|p|^2 <= 1/G^2` and the same Gaussian
tail inequality on the reciprocal lattice. The full LR weight sum used in
the AO-image bound instead uses a lower bound on the smallest nonzero
`|G+q|`; the finite SR zero mode is included separately at q=0.

Each contribution receives one eighth of the requested absolute raw-integral
error; the finite-domain screen receives another eighth. This leaves slack
beyond the five contributions to a three-center element. The resulting
estimate applies before whitening. It does not certify energy error or
stability of the retained metric rank. Production defaults still use the
old route while this planner and the replacement route undergo validation.

Exchange contraction also panels the auxiliary axis. The factored-density
path reserves three transformed-factor arrays, a possible contiguous input
panel and two AO-matrix temporaries within a 32 MiB numerical workspace.
The dense fallback uses the same cap with four panel arrays. Retained
factors, density factorizations, MPI gather storage and returned matrices
are separate. A cap smaller than one panel is refused before contraction.
Gamma uses the same dense panel contraction, including complex auxiliary
coordinates, rather than retaining a conjugate copy of the complete tensor.

The legacy integral-rescaling and rebuilt-basis modrho helpers now share
one conversion. With libint radial coefficients `c`, define
`s = sum_p c_p integral_0^infinity r^(2L+2) exp(-alpha_p*r^2) dr`.
Their scale is `sqrt(2L+1)/(4*pi*s)`, including the conversion between
normalized spherical harmonics and libint's regular solid harmonics.
The moment against `r^L Y_Lm` is then `1/sqrt(4*pi)`; an s function has
unit charge. Omitting the conversion in only the integral-rescaling path
changes the metric spectrum and therefore an absolute rank threshold.

## Private common-source MDF implementation

`python/vibeqc/periodic_mdf.py` builds private Gamma and general-k MDF factors
from the same SR/LR metric `M` and three-center tensor `T` described above.
For a caller-selected sphere of nonzero `p = G+q`, define
`w(p) = 4*pi/(volume*|p|^2)`, auxiliary Fourier integrals `A(P,p)`, and
Bloch AO-pair Fourier integrals `B(k,mn,p)`. The residual fit is

```text
M_res(P,Q) = M(P,Q) - sum_p w(p) conj(A(P,p)) A(Q,p)
T_res(k,P,mn) = T(k,P,mn) - sum_p w(p) conj(A(P,p)) B(k,mn,p)
L = concatenate(whiten(M_res) T_res, sqrt(w) B)
```

The native `_compute_gdf_plane_wave_projection` binding evaluates these PW
blocks with bounded Fourier panels and the same AO-pair image domain and
native basis normalization as SR/LR. Both sources omit the singular mode;
there is no second zero-mode correction. The caller supplies the auxiliary
basis and all finite domains explicitly. An empty PW span leaves a Gaussian
fit. Negative residual metric modes beyond the configured threshold and
roundoff allowance are errors; the legacy MDF threshold floor is not applied.

The private cache reserves retained Gaussian and PW factors before source
construction. It groups equal transfers, subdivides admitted batches, and
reuses one residual metric eigensystem per q. Geometry, both bases, transfer,
and physical source settings identify that eigensystem. Allocation failures
propagate; only explicit preallocation rejections permit subdivision. Native
workspace, vector copies, raw outputs, eigensolver workspace, and joined
factors are charged separately from borrowed bases and runtime overhead.

This cache has a distinct MDF type and is absent from public SCF selectors and
package exports. Production MDF, symmetry transport, and MDF analytic derivatives
have not been switched to it. `tests/test_periodic_mdf_source.py` registers
independent s-Gaussian PW witnesses, the no-PW limit, metric identity,
subdivision, and allocation-failure cases in the `pbc-gdf` lane. These new
numerical tests have not been run: calculation-based acceptance is deferred
during the repository deployment transition. Compiler syntax checks alone
do not establish scientific correctness.

The private `_build_mdf_jk` consumer now connects this cache to the shared
panelled Coulomb and exchange contractions. It accepts a spin-summed density
or separate alpha/beta densities, verifies the geometry, both bases and
ordered k mesh, and reserves retained cache/density storage, returned
matrices and contraction scratch before applying the operator. Every
exchange bra retains the full weighted ket sum. Partial bra selections
return matrices in the requested order and omit the incomplete exchange
energy; Hartree still uses every diagonal pair. The energies use restricted
`E_K = -sum_k w_k Tr(D_k K_k)/4` or separate-spin
`E_K = -sum_(s,k) w_k Tr(D_sk K_sk)/2`, with no Madelung correction.
The builder returns read-only factor arrays and a read-only pair mapping.

Additional pending witnesses compare the fixed-density interface against
explicit four-index contractions, complex fitting-coordinate rotations and
density variations of its returned energy. This integration does not enable
an SCF selector or establish MDF force compatibility.

The private native `_compute_gdf_plane_wave_projection_gradient_weighted`
primitive differentiates the real contraction of arbitrary unconjugated
weights with `M_PW`, `T_PW`, and `F_PW`. It shares the SR/LR derivative's
AO-center derivatives and image walker. Auxiliary-center derivatives use
`d A(P,p)/d R_P = -i p A(P,p)`; AO-center derivatives act on the complete
Bloch pair integral. The direct PW factor response is `sqrt(w) dB`.
Singular vectors contribute neither values nor derivatives, and there is
no finite SR zero-mode term in this PW primitive.

This routine holds the cell, q, k, reciprocal vectors and image membership
fixed. It is an atomic integral derivative, not a stress, cutoff derivative,
or complete MDF force. Borrowed metric/tensor/factor weights remain outside
its workspace cap. The implementation keeps one auxiliary Fourier panel,
seven complex values per AO pair and reciprocal point in a fixed panel,
and an atom-gradient array per admitted worker. It never constructs an
atom-by-three-center derivative tensor. Selected-vector offsets are preserved
across panels, including when a caller supplies zero vectors.

Pending tests compare each weighted term with displaced source values for
mixed s/p bases at Gamma and general k, check translation invariance of M/T,
exercise a second reciprocal panel and zero-vector omission, and reject
incompatible weight storage. These numerical checks remain unrun.

`retain_fit_state=True` now preserves the raw residual T, full metric
eigensystem, retained-mode mask and exact SR/LR and PW reciprocal lists.
These arrays share their admitted storage. The private response builder
uses the Gaussian truncated-inverse divided differences, including
retained/discarded projector rotation, and a separate direct PW response.
Modes on the metric threshold are refused because their derivative is
undefined. The atomic J/K response is assembled as

```text
dE = WM:dM_SRLR + WT:dT_SRLR - WM:dM_PW - WT:dT_PW + WF:dF_PW.
```

The private `_compute_mdf_cache_gradient` rebuilds one batch at a time from
the energy cache's recorded controls and checks the rebuilt factors before
differentiating them. A first pass over all diagonal pairs collects both
raw Hartree sources, `sum_k w_k T_res(k):D(k).T` and
`sum_k w_k F_PW(k):D(k).T`. Every second-pass batch uses these full sources;
the Gaussian Hartree metric response appears once. Exchange uses each
pair's complete weighted energy contribution. Separate spin channels share
Hartree of the total density and use their respective exchange derivatives.
An incomplete exchange pair set is refused.

The cache, density, optional spin total, source rebuild, response weights,
binding copies, native workspace and atom-gradient output are admitted
together. Temporary factor views are released before the next source build.
Pending tests cover raw M/T/PW energy variations with a discarded metric
mode, split Hartree batches, subtraction signs, source lifetime and
displaced-cell fixed-density energies. None has been run during the
calculation pause. This remains a private fixed-density two-electron derivative.

The existing multi-k RHF/UHF gradient assemblers now recognize an explicitly
supplied private MDF cache. Its adapter admits packing the driver density
lists and calls the combined MDF fit derivative once, using J of the total
density and the correct exchange fraction for each spin convention. The
assemblers retain their existing one-electron, nuclear, overlap-Lagrangian,
ECP and Madelung terms. The KS wrappers reuse this composition and add their
existing XC terms. Gamma is represented as a one-point mesh.

The caller must supply the accepted SCF W/S matrices, resolved one-electron
and nuclear/projector options, Madelung value and, for KS, XC provenance.
The MDF cache owns its physical auxiliary basis and execution budgets. Its
budget covers the fitted-derivative phase and density packing; other SCF
arrays and one-electron/XC workspaces retain their existing policies. The
UHF one-electron spin sum is allocated after the fitted derivative completes.

Pending composition witnesses check that the fit is counted once, hybrid
and spin factors reach the correct terms, ECP options are forwarded, and
packing is admitted before allocation. Public MDF SCF/gradient selectors
still use the legacy route and its existing capability guards. Full-SCF
stationarity, force finite differences, whole-job memory and numerical
acceptance of the new composition remain unverified.

The private `_evaluate_mdf_mean_field` now assembles physical Focks and
component-resolved total energies from that same MDF cache. It accepts the
resolved Hcore and overlap matrices, nuclear energy, Madelung value and
optional paired XC potential/energy. A nonunit exchange fraction requires
both XC inputs. The caller is responsible for evaluating XC at the supplied
density and preserving the one-electron gauge and projector provenance.
The evaluator includes XC energy once and applies the restricted or
separate-spin Ewald exchange correction to both Focks and energy. Full-mesh
exchange rows are returned in k order; zero exchange only needs diagonal
factors. Its returned Focks are read-only, and its result retains the exact
cache for the private gradient composition.

This phase reserves the borrowed Hcore, overlap, density and XC arrays,
retained factors, J/K and Fock outputs, and contraction/assembly scratch
before contraction. Other SCF arrays and external one-electron/XC builders
remain outside this phase's budget. Pending tests differentiate the total
energy with respect to complex Hermitian densities using a nonlinear model
XC functional, compare restricted and balanced-spin HF, and check admission
and diagonal-only operation. These tests have not run. The evaluator does
not implement an SCF iteration, certify stationarity, construct W, or enable
a public MDF route.

The existing bulk SCF cache hook now accepts the explicit private
`_MdfScfSource(plane_wave_cutoff)` object in `_lpq_cache_builder`. The
`gdf_method='rsgdf'` host driver supplies its common SR/LR cutoff planner,
SCF memory headroom, one-electron gauge and iteration machinery. After
planning, the injection builds the Gaussian/PW cache with its own complete
retention reservation. It does not pass MDF factors through the Gaussian
rank bound or relabel them as an RSGDF cache. RHF/RKS, UHF/UKS and the ROHF
energy driver accept the private hook; Gamma uses the same one-point
general engine. ROHF retains its occupied-factor exchange contraction,
linear bare-K damping and Roothaan effective Fock construction. Its returned
physical spin Focks and energy components use the accepted spin densities.

The injection requires 3D, full-mesh GDF exchange and no IBZ transport.
Existing functional and gradient guards still apply. Its source identity
and combined fit rank survive the gradient handoff, which returns the
energy cache for the combined MDF response. Progress and returned backend
metadata identify private, numerically unvalidated MDF and retain its
resolved LR cutoff. Ordinary selectors and `gdf_method='mdf'` are unchanged.

Pending integration tests cover planned-domain and memory-budget forwarding,
rejection before unsupported driver setup, retained-cache gradient handoff,
and real RHF/UHF/ROHF assembly against the private evaluator using synthetic
factors at Gamma and general k. ROHF cases include both spin channels,
damped iterations, the Ewald exchange shift and effective Fock assembly.
ROHF now canonicalizes the accepted effective Fock on both converged and
iteration-limit returns, including a first density-guess iteration. It
reports integer refill occupations without replacing the accepted spin
densities or rebuilding their energy. Pending witnesses check the retained
eigenproblem and preservation of those densities. ROHF gradients and property
validation remain separate work; effective orbitals do not diagonalize the physical
spin Focks and cannot use the RHF/UHF overlap-Lagrangian helper directly.
The private `_rohf_effective_spectrum=True` property diagnostic now keeps
these operators separate. Both spin channels share the returned effective
eigenpairs and energy axis; each channel's pair weights use its physical
spin Fock. Integrated populations contract the accepted spin density,
including orbital coherence, rather than a canonical refill. Returned
payloads label the energy operator, projection operator, shared orbital
basis and unvalidated status. Such curves are physical-spin-Fock projections
on effective energies, not physical-spin-Fock eigenspectra. The QVF writer
preserves all five diagnostic labels in DOS, PDOS, COOP and COHP sections,
and in the two pair-metadata members. It requires complete nonempty string
labels whenever a payload supplies any of them, before writing spectral
members. Ordinary payloads without labels retain their existing representation.
Pending archive witnesses check section/member agreement, member hashes,
schema validation and rejection before writes. Public ROHF property output
remains guarded until consumer interpretation and numerical acceptance are
complete. Pending numerical witnesses cover complex operators,
non-idempotent densities, nonuniform k weights, eigenpair rejection and
memory admission; no new numerical witness has run.
These tests have not run; the synthetic factors isolate assembly and do
not validate the native source. Full-SCF
stationarity, source convergence, forces, XC integration and whole-job
memory acceptance remain outstanding.

Private RHF/UHF MDF force assembly now constructs its overlap Lagrangian from the
accepted density and returned physical Fock. The terminating SCF step keeps
the density used for energy while reporting canonical physical eigenpairs;
refilling those orbitals can produce a different density. In the retained
S-orthonormal orbital space the new helper forms

```text
P = C^H S D S C
G = C^H F C
W = C (P G + G P)/2 C^H
```

At stationarity this is the usual occupation-weighted energy density. It
keeps the accepted fractional occupations, including small contributions
and coherent density within degenerate orbital spaces, and sums the spin
channels without another spin factor or k weight. It uses no overlap inverse
and does not refill or truncate the accepted density.

Before returning W, the helper checks finite Hermitian state arrays,
S-orthonormality, support of D in the retained orbital space, and agreement
of the reported eigenpairs with the physical Fock. The default relative
state-consistency tolerance is `1e-8` (absolute for S-orthonormality). The
weighted AO `F D S - S D F` norm must meet the driver's `conv_tol_grad`.
Cache, borrowed state arrays, W and serial workspace are admitted before
matrix work. This check applies only to private MDF; existing public routes
retain their assembly. It does not establish occupation-response or changing
retained-space derivatives, nor numerical force acceptance.

Pending tests exercise restricted and separate spins, nonorthogonal AOs,
retained overlap subspaces, degenerate states, tiny fractional occupations,
the `W -> W + lambda D_total` energy-origin shift, inconsistent-state refusal
and memory admission. These numerical tests remain unrun.

The separate private ROHF helper uses the molecular determinant expression
`W = Da Fa Da + Db Fb Db` with complex AO matrices. It preserves occupied
off-diagonal physical-Fock elements that an effective-eigenvalue refill
would omit. Shared retained orbitals supply coordinates for state checks;
they need not diagonalize either spin Fock. The helper requires Hermitian
inputs, S-orthonormality, retained-density support, idempotent spin projectors
and `Pa Pb = Pb`. Its stationarity condition is the k-weighted norm of
`[Ga, Pa] + [Gb, Pb]` in the shared retained space. Individual spin
commutators can be nonzero and cancel. This norm has its own explicit
tolerance; it is not silently equated to the driver's effective-Fock norm.

Cache, borrowed state, returned W and serial workspace are admitted before
matrix work. Returned AO blocks are read-only and contain no k weights or
extra spin factors. Fractional or damped states outside the determinant
tolerance are rejected. Pending witnesses cover complex nonorthogonal and
truncated spaces, shared-coordinate invariance, energy-origin shifts, an
independent overlap variation and invalid-state/admission checks. They have
not run. The helper is not yet wired to a complete periodic ROHF gradient;
source response, one-electron/Pulay composition and full force acceptance
remain outstanding.

The streamed MDF gradient now admits its response after reciprocal
enumeration but before raw integral evaluation. A shared reservation covers
the Gaussian response, direct PW weights, Fourier panels, reciprocal and
ket-vector binding copies, native scratch and two atom-gradient outputs.
The factor-rank estimate alone could miss a large LR vector list and fail
only after an expensive source rebuild. The source builder now receives the
separate response budget and full density mesh size, checks actual vector
counts, and raises the typed admission error so the caller can reduce the
batch. A single-pair refusal still propagates; ordinary allocation failures
are never treated as subdivision requests.

The derivative consumer checks the same reservation before weight or native
work. These execution controls do not change physical source identity and
are refreshed from the current gradient budget on each rebuild. Pending
tests cover large LR lists, exact admission boundaries, the real response
allocator with mocked native derivatives, batch subdivision and propagation
of actual allocation failures. They have not run during the calculation
pause.

Gradient response groups now distinguish mesh-index diagonals from exchange
pairs with the same transfer. Repeated or reciprocal-equivalent k entries
can have `q=0` even when `i != j`. The previous transfer-only grouping put
these exchange pairs into the Hartree source pass, which then rejected a
cache that the energy builder could consume. The diagonal response now
collects the complete weighted Hartree source and counts its metric term
once; exchange includes every pair in both groups with the existing spin
and mesh weights. Each group rebuilds the same physical q source within
the existing memory bounds.

Pending tests force single-pair batches on repeated and reciprocal-shifted
meshes and check Hartree/exchange pair accounting for both spin conventions.
A separate unrun native-source witness compares fixed-density energies and
gradients before and after replication of a k point. This does not widen
the public SCF gradient mesh guards or establish numerical acceptance.

Native SR metric, SR three-center and shared SR/LR/PW derivative workers now
allocate and zero their atom-gradient matrices in place after admission.
The previous vector fill constructor first materialized a full temporary
gradient and copied it into each worker, so initialization could exceed the
reservation by one atom-gradient matrix. The new initialization keeps the
same zero values, worker count and contraction order while removing that
temporary. Compiler syntax checks do not establish numerical or RSS
acceptance; the existing derivative witnesses remain pending.

## Returned density and accepted SCF state

The bulk multi-k GDF drivers support `return_lattice_density=True`; the
public runner requests it for QVF density and XSF artifacts. The producer
validates its full uniform unshifted or half-shifted Monkhorst-Pack mesh,
then returns a complete BvK residue set plus inverse cells. It checks
Hermiticity, time reversal and an unprojected round trip against each
accepted per-k density before attaching either spin channel. Non-time-reversal
states are refused for this real representation, rather than changed by a
real-part projection. No overlap cutoff defines the density's cell set.

The optional transform admits its numerical arrays, native/Python block
copies and bookkeeping before cell construction. Its default cap is 128 MiB,
configurable with `lattice_density_memory_bytes`; the reported
`density_lattice_reserved_peak_bytes` excludes other SCF storage. It never
constructs a cell-by-k phase table or enumerates a lattice sphere.

The artifact reader checks the returned BvK representation independently.
Its analytic charge uses the accepted Bloch overlap: a minimal complete
residue set is a representation of the infinite density, not a finite
real-space overlap truncation. Independent AO-image and grid-charge
convergence remain required. Finite-temperature GDF likewise evaluates
entropy from the accepted metric density `X^H S D S X`; a subsequent Fock
orbital refill does not change the entropy paired with the returned energy.

The acceptance tests compare raw SR integrals with an independent analytic
s-Gaussian Boys-function oracle and combined integrals with an independent
bare reciprocal sum. They check finite-domain convergence, atom-image
covariance at nonzero q, reciprocal relabelling, angular-shell normalization,
and split-parameter invariance. A tight spherical Gaussian self-integral
test spans exponents 20 to 160,000 with the same compact reciprocal mesh. Full
SCF, dense-core accuracy, factor-cache memory and node scaling remain
separate acceptance gates for the production route.

## Historical initial implementation

**Status (2026-05-09, feature/native-gdf-chain)**: in progress.
Slice 1 (C++ periodic 2c/3c ERI kernels) and slice 2 (Python aux loader
+ Cholesky-fitted Lpq) have shipped on the `feature/native-gdf-chain`
branch. **Slice 3 (modrho compensation) is empirically confirmed
mandatory**, see ["Slice 3, modrho compensation is required"](#slice-3-modrho-compensation-is-required) below.
Slices 4-8 (V_ne via aux, driver rewrite, parity sweep, docs) are
queued behind slice 3.

**Goal**: drop the PySCF dependency from the GDF spike. Build the Lpq
tensor entirely from vibe-qc's libint pipeline + an in-house auxiliary
basis manager.

## What Lpq is, recap

```
M_PQ      = (P|Q)_periodic           ← 2c periodic ERI on aux basis
(μν|P)_pp = Σ_T (μ_0 ν_0 | P_T)      ← 3c periodic ERI w/ ket image sum
Lpq[L, μ, ν] = Σ_P (μν|P)_pp · M^{-1/2}_{PL}      (Cholesky factor)
```

Then for closed-shell RHF (or KS):

```
ρ_L  = Σ_λσ Lpq[L, λ, σ] · D_λσ
J_μν = Σ_L Lpq[L, μ, ν] · ρ_L
K_μν = Σ_L (Lpq[L] @ D @ Lpq[L].T)_μν
F_2e = J − ½ K  (+ Madelung correction for exxdiv='ewald')
```

## Pieces vibe-qc needs to add

### 1. C++, 3c and 2c molecular ERI engines

Wraps libint2's existing 3-center and 2-center engines.

`cpp/include/vibeqc/aux_eri.hpp` (new):

```cpp
// 3c molecular: (μν|P) tensor (n_ao, n_ao, n_aux), no lattice sum.
Eigen::Tensor<double, 3> compute_3c_eri(
    const BasisSet& ao_basis,
    const BasisSet& aux_basis);

// 2c molecular: (P|Q) matrix (n_aux, n_aux), no lattice sum.
Eigen::MatrixXd compute_2c_eri(const BasisSet& aux_basis);
```

`cpp/src/aux_eri.cpp`:
- Use `libint2::Operator::coulomb` with `Engine::compute(s1, s2, s3)`
  for 3c and `Engine::compute(s1, s2)` for 2c.
- Mirror the molecular `compute_eri` already in
  `cpp/src/integrals.cpp` for the threading + buffer layout.
- Output is ROW-major libint AO order; permutation to PySCF order
  (if needed) lives in Python.

### 2. C++, periodic versions

`cpp/include/vibeqc/aux_eri.hpp`:

```cpp
// Periodic 3c: Σ_T (μ_0 ν_0 | P_T) tensor (n_ao, n_ao, n_aux).
Eigen::Tensor<double, 3> compute_3c_eri_lattice(
    const BasisSet& ao_basis,
    const BasisSet& aux_basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts);

// Periodic 2c: Σ_T (P_0 | Q_T) matrix (n_aux, n_aux).
Eigen::MatrixXd compute_2c_eri_lattice(
    const BasisSet& aux_basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts);
```

Implementation: mirror `cpp/src/lattice_integrals.cpp` cell loops with
shifted shells, accumulate. Schwarz screening on 3c can prune many
quartet calls.

For the long-range part of the periodic 2c metric (where the `1/r`
sum diverges in real space), apply Ewald-style splitting:
  `(P|Q)_periodic = (P|Q)_SR(ω) + (P|Q)_LR(ω)`
where SR is real-space erfc(ω)/r and LR is reciprocal-space erf(ω)/r.
The LR part is a small G-mesh sum, NOT an FFT (the aux basis is
small enough that direct G-summation works).

### 3. C++, pybind11 bindings

Add to `cpp/src/bindings.cpp`:

```cpp
m.def("compute_3c_eri",         &compute_3c_eri,
      "AO 3-center ERI tensor (μν|P).");
m.def("compute_2c_eri",         &compute_2c_eri,
      "AO 2-center ERI matrix M_PQ = (P|Q).");
m.def("compute_3c_eri_lattice", &compute_3c_eri_lattice,
      "Periodic 3-center ERI tensor with image sum on the ket.");
m.def("compute_2c_eri_lattice", &compute_2c_eri_lattice,
      "Periodic 2-center ERI metric matrix.");
```

### 4. Python, auxiliary basis manager

`python/vibeqc/aux_basis.py` (new):

```python
def make_aux_basis_set(ao_basis: BasisSet, *, aux_name: Optional[str] = None,
                      drop_eta: float = 0.0) -> BasisSet:
    """Build an auxiliary BasisSet for an AO basis.

    Mirrors PySCF's pyscf.df.addons.predefined_auxbasis +
    pyscf.pbc.df.df.make_modrho_basis:
      - If aux_name is None, look up the standard JKfit aux for the
        AO basis (sto-3g → universal-jkfit, def2-tzvp → def2-jkfit, ...).
      - If aux_name is a string, load it from the libint .g94 path.
      - Drop primitives with exponents below drop_eta to avoid
        linear-dependent metric (matches PySCF make_modrho_basis).
      - Renormalize aux primitives so the monopole moment ∫ χ_P dr =
        √(1/(4π)). This convention simplifies the compensated-charge
        algebra (see PySCF make_modrho_basis docstring).
    """
    ...
```

Reference: `pyscf/pbc/df/df.py:make_modrho_basis` (lines 64-120).

### 5. Python, Lpq builder

In `python/vibeqc/periodic_rhf_gdf.py`, replace
`_lpq_tensor_from_pyscf` with:

```python
def _lpq_tensor_native(system, ao_basis, *, aux_name=None,
                      lat_opts=None, linear_dep_thr=1e-9):
    """Build Lpq(L, μ, ν) entirely with vibe-qc machinery.

    1. Build aux basis from ao_basis (or load aux_name).
    2. Compute periodic 2c metric: M[P, Q] = Σ_T (P_0 | Q_T)
       via compute_2c_eri_lattice (or with Ewald split if needed).
    3. Compute periodic 3c: T[μ, ν, P] = Σ_T (μ_0 ν_0 | P_T)
       via compute_3c_eri_lattice.
    4. Cholesky factor M = L L^T (Lehtola-style pivoted Cholesky for
       robustness against the diffuse-aux linear dependence).
    5. Solve L · X = T.reshape(naux, nao*nao) → Lpq.
    """
    aux_basis = make_aux_basis_set(ao_basis, aux_name=aux_name)
    M = compute_2c_eri_lattice(aux_basis, system, lat_opts)
    T = compute_3c_eri_lattice(ao_basis, aux_basis, system, lat_opts)

    # Pivoted Cholesky tolerates near-singular M (diffuse aux).
    from scipy.linalg import cho_factor, cho_solve
    L_chol, low = cho_factor(M, lower=True)
    Lpq = cho_solve((L_chol, low), T.reshape(M.shape[0], -1))
    return Lpq.reshape(M.shape[0], T.shape[0], T.shape[1])
```

### 6. Validation

Hard test: compare `_lpq_tensor_native(cell, basis)` to the existing
`_lpq_tensor_from_pyscf(cell)` elementwise. Should match to ~1e-10
(Cholesky residual, identical aux basis, identical integrals).

Then drop the PySCF Lpq path. The full SCF chain becomes self-hosted:

```
S, T, V_ne   from vibe-qc compute_*_lattice (already)
Lpq          from native vibe-qc 3c/2c        (shipped for Γ)
cell blocks  from compute_*_eri_lattice_blocks (multi-k storage boundary)
J, K          from native einsum on Lpq        (already for Γ)
V_xc          from native libxc on Becke grids
```

PySCF and CRYSTAL are no longer in-process periodic backends. They are
external reference programs whose outputs are parsed for parity checks.
The remaining native multi-k GDF work is the k-dependent metric/tensor
phase assembly, IBZ-weighted J/K contractions, and disk-backed storage.

## Estimated effort

- 3c/2c molecular bindings: 2-3 hours.
- Periodic versions with Schwarz screening: 4-6 hours.
- Pybind bindings + CMake: 30 min.
- Aux basis manager + JKfit table loading: 2-3 hours.
- Lpq builder + Cholesky: 1 hour.
- Validation + debug: 4-8 hours (typical: aux normalization, linear-
  dep handling for diffuse bases, Schwarz threshold tuning).

Total: **1.5-2 focused days**. Doable in one follow-up session.

## Auxiliary basis library, what to ship

For sto-3g and minimal bases: PySCF auto-generates "even-tempered"
aux when no explicit aux is registered. We can do the same: emit
even-tempered Gaussians from the AO exponent range. Quick fix that
unblocks the full sweep.

For the Bredow-group periodic bases (pob-tzvp, pob-tzvp-rev2): we
already have auxiliary basis design as a paper-worthy item (see
memory: `project_pob_tzvp_aux_basis.md`). Ship a **prototype**
auto-aux for v0.7.x; the proper hand-tuned aux is a separate
deliverable.

For def2-* bases: copy PySCF's def2-jkfit / def2-jkfit-tight tables
into vibe-qc's `basis_library/aux/`.

## Slice 3, modrho compensation is required

The image-summed periodic 2c metric `M_PQ = Σ_T (P_0 | Q_T)` on a
non-charge-compensated aux basis is **mathematically divergent** in
3D, the leading monopole-monopole term sums as `Σ 1/|T|` over a 3D
lattice, which is the classical Madelung divergence. This was an
unresolved question at design time; the empirical confirmation lives
in `examples/debug/scratch_lpq_native_smoke.py` and
`examples/debug/scratch_lpq_native_energy.py`. On MgO/sto-3g with
def2-universal-jkfit (PySCF's auto-pick):

| cutoff (bohr) | ‖M‖_F | min eig(M) | Cholesky |
|---:|---:|---:|---:|
| 4  | 1.6e+3  | +4.8e−6 | ✓ |
| 8  | 6.0e+3  | −7.6e+1 | ✗ |
| 12 | 1.2e+4  | −2.2e+1 | ✗ |
| 16 | 1.8e+4  | −9.5e+0 | ✗ |
| 20 | 3.4e+4  | −4.4e+0 | ✗ |

‖M‖_F grows monotonically with cutoff (no convergence) and the
metric loses positive-definiteness past the Cholesky-OK cutoff (~4
bohr, barely larger than the 8 bohr cubic cell, so the SR aux
contributions dominate while the LR divergence has not yet kicked in).

Iter-1 SCF energy with the SAD initial density:

| Lpq source | E_iter1 (Ha) | comment |
|---|---:|---|
| (A) PySCF Lpq baseline | −1084.108 | converges to −1085.231 |
| (B) Native cut=4 (Cholesky) | −270.229 | off by ~800 Ha |
| (C) Native cut=20 (SVD pseudo) | −12,653.152 | catastrophic |

so the bare lattice sum is unusable for SCF. **Modrho compensation
is structurally required**, not an optimisation.

### Modrho, algorithmic note

PySCF's `pyscf.pbc.df.df.make_modrho_basis` rescales each contracted
aux shell so that its monopole moment is `√(1/(4π))`:
```
s_i  = Σ_p c[p, i] · gaussian_int(2L+2, α_p)
c[p, i] *= √(1/(4π)) / s_i
```
For l > 0 the renormalisation is conventional (the actual electric
monopole vanishes by symmetry); for l = 0 it caps each S-shell's
monopole at the same finite value. Combined with PySCF's separate
charge-compensation on the AO-pair side
(`pyscf.pbc.df.df_builder._CCGDFBuilder.make_j3c`, smooth Gaussians
that cancel the AO pair's monopole), the resulting (modified-aux |
compensated-AO-pair) integral converges in real space.

For vibe-qc to reproduce this:

1. **Ship a way to construct a `BasisSet` with custom contraction
   coefficients.** Two options:
   - (preferred) Add `BasisSet(Molecule, vector<ShellInfo>)` C++
     constructor that builds `libint2::Shell`s directly. Bypasses
     the `.g94` round-trip; avoids file I/O races; clean API.
   - (workaround) Write a temp `.g94` file under
     `python/vibeqc/basis_library/basis/<unique_name>.g94` with
     denormalised coefficients, load via `BasisSet(mol, name)`,
     clean up. Works without C++ changes but ugly. The denormalisation
     dance (libint applies primitive norm at load; PySCF's modrho
     coeffs are post-normalised; the conversion factor must be exact)
     is error-prone.
2. **Implement modrho in `vibeqc.aux_basis`** using vibe-qc's own
   `gaussian_int(2l+2, α)` helper (port the 3-line PySCF function).
   Apply per-shell rescaling on the libint-internal coefficients.
3. **Implement compensated-AO-pair handling on the 3c side.** The
   compensating smooth Gaussians become a separate `compcell`
   BasisSet; the periodic 3c is then computed on
   `(aux | AO_pair − comp)` and the missing `(aux | comp)` piece is
   added back analytically (single G-mesh sum, see PySCF's
   `_CCGDFBuilder.make_j3c` ~600 LOC for the canonical algorithm).

Estimated additional effort: **1-2 days** (modrho + compcell + a
Python-side reciprocal-space sum for the compensating term + parity
validation). The slice-1 C++ kernels are reusable as-is.

### Resumption checklist for the next session

1. Read this section + check
   [Archived `handovers/HANDOVER_GDF_OUTSTANDING.md`](https://vibe-qc.com/docs/)
   for historical context; check current work against the new `mpei/vibe-qc` repository.
2. Confirm slice-1 baseline still passes:
   `.venv/bin/python examples/debug/scratch_aux_eri_lattice_smoke.py`.
3. Confirm slice-3b (modrho) reduces metric ill-conditioning by ~2
   orders of magnitude but does NOT make the metric SPD past
   cutoff=4 bohr, this is the open question for slice 3c-d.

### Update 2026-05-09, slices 3a and 3b shipped

**Slice 3a (BasisSet from explicit ShellInfo)**, committed
(`0ed5854`). Adds `BasisSet(molecule, shells, name,
coefficients_pre_normalized=True)` C++ constructor + Python-mutable
`ShellInfo` so modrho can run in pure Python without `.g94` file
round-trips. Round-trip on H2/sto-3g shows S and ERI agree exactly;
scaling all coefficients by 2 produces overlap × 4, validating that
the no-renorm path is honoured.

**Slice 3b (modrho normalisation)**, committed (`5778e39`). Direct
port of PySCF `make_modrho_basis`:
```text
s_i = Σ_p c[p, i] · gaussian_int(2L+2, α_p)
c[p, i] *= √(1/(4π)) / s_i
```
plus an optional `drop_eta` primitive cull. Wired into
`make_aux_basis_set` via `compensate=True` (default).

Empirical effect on MgO/sto-3g/def2-universal-jkfit (image-summed
2c metric, increasing cutoff):

| cutoff (bohr) | min eig(M) raw | min eig(M) modrho | min eig(M) modrho+drop=0.2 |
|---:|---:|---:|---:|
| 4  | +4.8e−6  | +3.5e−7  | +5.3e−6  |
| 8  | −7.6e+1  | −7.2e−1  | −3.6e−1  |
| 12 | −2.2e+1  | −1.0e−1  | −2.3e−2  |
| 16 | −9.5e+0  | −2.6e−2  | −4.5e−3  |
| 20 | −4.4e+0  | −8.6e−3  | −4.7e−4  |

Modrho improves conditioning by **~2 orders of magnitude**, but the
metric remains non-SPD past cut=4 bohr. The 3D Madelung divergence
is not cured by normalisation alone, this is the expected behaviour
per the PySCF algorithm structure: modrho is a normalisation
convention that simplifies downstream algebra; the actual convergence
treatment lives in the AO-pair-side compensating-charge mechanism
(slice 3d below).

### Slice 3c-d, open algorithmic questions

Two paths are plausible from here; **the literature sweep
commissioned in parallel should resolve which is correct:**

#### (i) Per-shell precision-truncated lattice cutoff

Hypothesis: PySCF's `pbc_intor("int2c2e")` on the modrho'd auxcell
returns a **finite** matrix (max |M| ≈ 20 on MgO/sto-3g) by using
its `cell.precision`-controlled image cutoff via
`_estimate_rcut(es, l, c_max, precision)`. Each aux shell contributes
to images only out to its own precision-justified radius; diffuse
primitives have small radii so their long-range divergent
contribution is naturally truncated.

If true, vibe-qc's bare 2c lattice sum on modrho aux + per-shell
rcut should reproduce PySCF's `pbc_intor("int2c2e")` matrix to
~1e-10. Then the metric is "PSD by truncation", physically accurate
to the precision target, not to infinity.

Implementation cost: small. Add per-shell-pair rcut to the cell loop
in `compute_2c_eri_lattice` / `compute_3c_eri_lattice`.

#### (ii) Full AO-pair compensating-charge mechanism

The textbook fix per Sun et al. 2017 § II.B: introduce a "compcell"
of smooth Gaussians (one per atom × per l, with global
`smooth_eta`). For each AO pair (μ, ν) at center A, compute its
multipole moments; subtract a matching combination of compcell
Gaussians; compute (aux | compensated AO pair) in real space
(converges); add back analytical (aux | smooth-Gaussian) terms in
reciprocal space.

Substantially more code:
- Compcell `BasisSet` construction (`make_chgcell` in PySCF).
- AO-pair multipole moments (libint `emultipoleN`, already used in
  `compute_dipole`; extend to higher l).
- (aux | compcell) integrals, special case of the existing 3c lattice
  with compcell as the orbital pair.
- Reciprocal-space (aux | smooth) sum, small G-mesh, analytic
  Fourier transforms of Gaussians.

Estimate: 2-3 days, plus parity validation.

#### Decision rule

Try (i) first, small change, may suffice for sub-µHa parity if our
hypothesis about PySCF's truncation is right. If (i) doesn't reach
sub-µHa, the lit sweep's PySCF algorithm walkthrough should clarify
exactly what (ii) needs to look like.

### Resumption checklist (updated 2026-05-09)

1. Read this section + the handover docs.
2. Verify slices 1-3b still work:
   - `examples/debug/scratch_aux_eri_lattice_smoke.py` (slice 1
     structural check).
   - Quick modrho check via `make_aux_basis_set(mol,
     aux_name="def2-universal-jkfit", compensate=True)` and a manual
     ‖M‖_F sweep, should reproduce the table above.
3. Implement per-shell rcut (path (i), start here).
4. Compare vibe-qc's 2c metric to PySCF's `auxcell.pbc_intor(int2c2e)`
   elementwise. If matched to ~1e-10, ship as slice 3c.
5. If not matched, defer to lit-sweep results for the proper
   compcell algorithm and implement path (ii) as slice 3d.
6. Once Lpq parity is achieved, resume slices 4 (V_ne via aux), 5
   (driver rewrite), 6 (`_basis_g94` cleanup), 7 (full parity sweep),
   8 (docs + PR).
