# Experimental chi translation local MP2

`vibeqc.periodic.chi.local_mp2.run_chi_local_mp2` connects an admitted
classic physical-pair RI RHF `ChiHFReference` to the shared translation-aware
local-MP2 engine. This is an explicit experimental library route. Chi remains
experimental and D93 fitted/post-HF remains not-qualified. Source tests do not
establish scientific acceptance, production threshold accuracy or scaling.

The [classic RHF/MP2 convergence study](chi_mp2_convergence.md) separates
finite-source algebraic acceptance from physical-support, basis and mesh
convergence, with fixed gates and independently measured reference differences.

## Reference and usage

Prepare a reference with `prepare_chi_hf_reference` using the explicit controls
in the [fitted-support guide](chi_fitted_support.md), or restore an accepted
classic RHF archive as described in [restart](chi_restart.md). The source must
be a neutral three-dimensional singlet, use full Coulomb RI with unscreened
physical pairs, and carry a complete uniform unshifted character mesh, tight
stationarity, integer gapped occupations and its original physical input
binding. The adapter never reruns SCF, refits factors or changes supports.
This extends the D6-D7 block routes with sealed physical-pair
transport. It reuses the same `CorrelationSource`, `CyclicAOOperator`,
factor-provider, selection, PAO/PNO and iterative solver interfaces.
Per-character frozen and active projectors are preserved without introducing
D6's global energy sorting or its full virtual-orbital payload.

```python
from vibeqc.periodic.chi.local_mp2 import run_chi_local_mp2

result = run_chi_local_mp2(
    reference, frozen_core=0,
    tcut_mkn=0.0, tcut_pno=0.0,
    max_iter=250, conv_tol_energy=1e-13, conv_tol_residual=1e-11,
    damping=1.0, max_transport_bytes=256 * 1024**2,
    factor_provider="streamed", max_factor_bytes=64 * 1024**2,
    factor_auxiliary_tile=16, factor_cache_bytes=1024**2,
    compact_storage=True, max_selection_bytes=64 * 1024**2,
    output=output_dir / "local-mp2",
)
raw = result.raw_e_corr_per_cell
pno_estimate = result.pno_correction_per_cell
converged = result.converged
```

`output` writes `.out`, `.system`, `.bibtex` and `.references` through
`vibeqc.output`. Without a stem, active text/structured output channels receive
the same result. The manifest records the retained factor source, consumed
reference and adapter identities, runtime, actual overlap/Fock/factor
transform audits, electron counts, domain and PNO settings, selected spaces,
solver controls, convergence and method convention. `chi_correlation_record`
provides the same structured payload to other output consumers.

`frozen_core` counts lowest explicit occupied spatial orbitals **per character**.
The adapter preserves the frozen and active projectors separately and orders
all frozen columns first in the shared source. The engine receives the total
torus frozen count, `prod(mesh) * frozen_core`. Scalar ECP references retain
their executed ECP identity, full metadata and distinct physical, ECP-removed,
explicit, frozen-explicit and correlated electron counts. ECP cores are already
absent. This interface does not claim [ECP acceptance](chi_ecp.md) or close #530.

UHF, ROHF, SR, KS/hybrid, reduced or shifted meshes, symmetry-reduced references,
fractional occupations, metallic gaps, overlap-truncated AO spaces, charged
cells and lower dimensions refuse. An old `ChiAcceptedReference` or historical
torus Hamiltonian cannot substitute for this sealed physical-pair reference.
Accepted HF restart is reusable; local-MP2 amplitude restart is not implemented.

## Scoped new-source acceptance

Full-domain/zero-PNO admission passes for the accepted H2/cc-pVDZ Gamma source
and LiH radius-24 `(4,1,1)` source, with independent LiH frozen 0/1 partitions.
Bounded and prepared transport agree exactly in the recorded raw energies,
residuals and PAO/PNO ranks; maximum canonical error is 7.55e-15 Ha/cell.
H2 uses AO seeds and LiH the explicit cosine option. The
[convergence record](chi_mp2_convergence.md#local-admission-on-the-newly-admitted-sources)
contains runtime pins, controls, residuals and separate corrections. This
accepts the finite-source full-space route, not nonzero truncation thresholds,
physical convergence or general crystal accuracy. No method fix was needed.

The [LiH support follow-up](chi_mp2_convergence.md#lih-support-extension-and-local-truncation-probes)
adds full-space admission on seven larger-support sources, preserves the
reciprocal-40 factor-recovery refusal, and measures independent domain/PNO
errors on the original radius-24 source. PNO cutoffs 1e-6 and 1e-7 fail the
10-microhartree raw target for its all-electron partition, even though separate
PNO estimates reduce the corrected errors. No production cutoff is inferred.

## Explicit occupied seed frames

The default `seed_basis="ao"` greedily selects translated families of
normalized primitive AOs and applies symmetric Lowdin orthogonalization.
An admitted occupied projector need not admit this particular frame: an AO
may have no projection onto the occupied space at one character even when
it has a large projection at another. A refused frame is not evidence of
an invalid HF reference or a failed MP2 iteration.

`seed_basis="cosine"` explicitly uses real DCT-II mixtures of the normalized
primitive AOs, re-normalized in the home-cell overlap metric, as candidate
seeds. The same primitive mixing is used at every translated cell. Projection,
complete Gram spectra, the strict `seed_lindep` test, deterministic greedy
order, symmetric Lowdin construction and global admission checks are retained.
Active and frozen partitions are selected and checked independently. No
automatic fallback, rank relaxation, canonical occupied array or change of
SCF Fock is introduced. This option can also refuse a common frame.

This changes only the seed matrix in the existing projected-seed construction
of Riplinger and Neese, Sec. II B1, with the finite-character transform of
Nejad et al., Sec. II, Eqs. (2)-(10). It is an experimental deterministic frame,
not optimized Wannier localization or a production localization default.
Mixed seeds can alter domains and PNO truncation errors; compare them
separately from the full-space raw limit. Provenance records the basis recipe
and selected cosine indices; `active_seed_aos` and `frozen_seed_aos` are empty
for mixtures, so cosine indices are never presented as individual AO labels.
Compact preparation adds `8*(5*b*b+2*b)` bytes to its controlled payload
reservation for a primitive basis of size `b`. The existing exclusions from
that reservation remain in force.

## Exact finite-source transport

The overlap and Fock home-row blocks are computed directly as

$$B_D = N^{-1}\sum_k e^{-ikD} A(k),\qquad A_{R,T}=B_{T-R}.$$

All displacements remain. The adapter supplies `CyclicAOOperator` objects and
checks reality, cyclic symmetry, positive overlap symbols and forward recovery
of every original Bloch matrix. It does not build square supercell AO overlap
or Fock matrices. The actual SCF Fock, including its BvK exchange zero-mode
term, enters the amplitude equations unchanged. The retained Coulomb factors
supply occupied-virtual numerators; the overlap-only zero-mode constant has
zero occupied-virtual matrix elements. Canonical comparisons must keep the
same SCF Fock, equivalent to `keep_exxdiv=True` in references that expose that
choice. No historical torus source or finite-size correction is substituted.

Complex Bloch orbitals are admitted only through a verified real-space
transformation. With unitary Fourier-folded occupied coefficients Z, the real
span `[Re Z, Im Z]` must have exactly the original occupied rank and unit
nonzero metric-Gram eigenvalues. Compact preparation evaluates its complete
spectrum through character blocks, described below. The normalized real-span
projector must reproduce the original **complex** metric projector at every
AO entry and displacement. Active and frozen spaces are checked independently,
including at self-inverse characters. Arbitrary orbital phases and character
ordering are allowed; a complex projector is not. The preceding canonical
reference's complete metric, stationarity and physical-source admission is
unchanged. Conventional D9-D12 adapters keep their complete eigenspace replay.

The default `factor_provider="streamed"` borrows the immutable retained Bloch
operator. `factor_provider="dense"` explicitly selects the existing native
home-auxiliary transform as a small-system oracle. There is no automatic
fallback. Both must recover **every** retained complex AO factor entry within
`1e-10` absolute tolerance, including canonical auxiliary null modes. Real
and symmetric admission uses the same tolerance. Incompatible auxiliary
gauges refuse; no retained rank, support or source identity changes.

For N cells, define the unitary orbital transform and the retained factor
convention by

$$c_{k\mu p}=N^{-1/2}\sum_R e^{-ikR}C_{R\mu p},\qquad
B_{tP,R\mu,S\nu}=N^{-2}\sum_{ij}
e^{ik_i(R-t)-ik_j(S-t)}L_{ij,P\mu\nu}.$$

The two inverse AO transforms each supply $1/N$. Contracting with two orbital
columns supplies $\sqrt N$ from each AO sum, leaving

$$X_{q,Plr}=N^{-1}\sum_{k_j-k_i=q}
c_{i,l}^{\dagger}L_{ij,P}c_{j,r},\qquad
B^{\rm MO}_{tP,lr}=\sum_q e^{iqt}X_{q,Plr}.$$

Character orthogonality gives $\sum_t e^{i(q+q')t}=N\delta_{q',-q}$.
For the admitted real pair inputs, $X_{-q}=X_q^*$, so the actual pair kernel is

$$(ia|jb)=N\sum_{qP}X_{q,Pia}X^*_{q,Pjb}.$$

The conjugation is on the second pair factor. This is also the canonical
retained-operator convention $L_{ia}L_{bj}^{\dagger}$ after expanding the
orbital coefficients. No extra per-cell energy division occurs in the
provider; the shared solver applies its existing orbit weights and division
by N. Nonorthogonal AO bases require no added overlap in this covariant
coefficient contraction. Complex orbital inputs to `mo_transform` retain the
left-coefficient conjugation and explicit auxiliary-cell sum.

Admission folds one primitive AO pair and an auxiliary tile at a time using
character/displacement tiles of width d, with d at most 8 and strictly below
N for N greater than one. It retains linear `d*N*p` slabs, never an additional
`N*N` character table or `N*N*p` audit tile. Define

$$H_{RjP}=N^{-1}\sum_i e^{ik_iR}L_{ij,P\mu\nu},\qquad
B_{RSP}=N^{-1}\sum_j e^{-ik_jS}H_{RjP}.$$

For each target character tile I, every displacement block is reconstructed.
The separately transformed reversed AO block supplies
$A_{RSP}=\tfrac12(\operatorname{Re}B_{RSP}+
\operatorname{Re}B^{\nu\mu}_{SRP})$. Every imaginary entry and every real
symmetry difference is measured before this average. Recovery then uses

$$W_{ISP}=\sum_R e^{-ik_IR}A_{RSP},\qquad
\widehat L_{IJP}=\sum_S e^{ik_JS}W_{ISP}.$$

Every retained entry of $\widehat L-L$ is checked, including every null-space
component. The inverse sums run over the complete i axis followed by the
complete j axis. The forward R sum accumulates in original displacement order;
the S reduction spans its full axis. Tiling changes storage and recomputation,
not normalization, conjugation, the real transpose average or any tolerance.
All norms are maximum absolute entries in their original representation.
Reconstructing B again for each I costs extra work; it is not a speed claim.

The one-entry immutable phase cache holds only a bounded rectangle and is
bound to the sealed source identity, character order and mesh. Pair and MO
contractions use the same generator, including complex left conjugation and
the auxiliary-cell phase product. The phase translation law still visits
every character, origin and cyclic shift, including self-inverse cells.
The sealed provider's constructive covariance follows from this checked phase
action and the complete factor recovery.
`factor_audit="all-origins"` on the shared engine still checks actual pair
contractions at every origin. No sampled energy replaces a source audit.

These transformations follow Nejad et al. (2025), Sec. II, Eqs. (2)-(6),
[doi:10.1063/5.0290816](https://doi.org/10.1063/5.0290816). The existing shared
engine uses its projector/Lowdin occupied frames and Mulliken atom domains,
not the paper's optimized Wannier/OSV/local-fitting procedure. Its occupied
coupling residual follows Sec. V, Eqs. (42)-(44). PNO densities and cross-pair
metric transport retain Riplinger and Neese (2013), Sec. II B3, Eqs. (23)-(25),
and II C, Eq. (30), [doi:10.1063/1.4773581](https://doi.org/10.1063/1.4773581).
The [D1-D5 design](design_dlpno_correlation_source.md) explains their density
normalization differences and even-mesh pair multiplicities.

## Approximations, convergence and memory

`tcut_mkn` and `tcut_pno` are independent. Zero means full AO domains and no
PNO truncation. Positive domain cutoffs use absolute Mulliken populations;
positive PNO cutoffs use the shared `pno_norm="mp2"` density. `lindep` and
`seed_lindep` retain the shared PAO rank and occupied seed conditioning controls.
For a canonical-limit comparison, also verify that each full-domain PAO rank
equals the total explicit virtual dimension and converge `lindep` if needed.
Zero domain/PNO cutoffs do not override numerical conditioning cutoffs.
No pairs or occupied Fock couplings are screened. Domains that are empty, or
PNO spaces that violate a self-inverse pair stabilizer, refuse explicitly.

`raw_e_corr_per_cell` is the iterated raw pair energy. The separately reported
`pno_correction_per_cell` is the semicanonical discarded-PNO estimate **inside
the selected domains**. It does not repair missing domains or occupied-Fock
coupling. `e_corr_per_cell` and `e_total_per_cell` include that named estimate.
There is no domain correction. The raw residual and energy change are tested
on the same amplitude state. Reaching `max_iter` returns `converged=False`,
including in artifacts; a correction cannot turn it into convergence.

Each metric/Fock stores `N*b*b` real numbers for b AOs per primitive cell.
The streamed provider removes the additional `8*a*(N*b)**2` byte home-factor
array and its immutable snapshot. The retained Bloch source still stores
`16*N*N*a*b*b` bytes and is borrowed without copying. Dense oracle preparation
retains the original allocations and is outside the new factor budget.
`compact_storage=True` (the chi default) retains only selected occupied
home columns, with `C[R mu,T i] = C[(R-T) mu,0 i]`. Active and frozen frames
remain separate. Home columns are constructed directly from the cyclic symmetric
Lowdin matrix function. Complete source-partition, metric and translation
audits use verified cyclic displacement blocks at the existing `1e-10`
maximum-entry tolerance; unverified inputs retain streamed column checks. Selection and pair spaces share the same immutable home buffers.
No coefficient or occupied tail is screened.

Pair storage reuses the shared PAO recipes `V = E_D A - C B`, where C is the
active/frozen union. The builder compares every reconstructed PAO and rotated
PNO coefficient with the dense algebra, then checks the complete metric,
occupied exclusion and self-inverse translation stabilizer as before. It
retains A, B and domain indices, reconstructing one pair at a time for factor
contractions. Native preparation receives neither a dense selected occupied
frame nor a dense pair-AO pool. Active and frozen home columns have independent
cell-major indexing. It checks every occupied Gram and translation entry,
then streams occupied columns for recipe validation and stable pair links. Its stable selected-column branch reconstructs one source
pair at a time and includes that rectangle in its existing workspace bound.
Recipe/source hashes bind prepared transport reuse without reconstructing
AO bases during iteration or provenance recording.

Use `compact_storage=False` for the dense coefficient oracle, independently
of `factor_provider`. Shared generic translation APIs retain their opt-in
`compact_storage=True` interface. `np.asarray(selection.occupied)` explicitly
reconstructs a detached complete matrix. `spaces.pair_coefficients[row]`
returns a detached pair matrix; neither output is cached in compact mode.
The AO-reference solver requires explicit dense coefficient storage.

`coefficient_storage` reports retained selected/recipe/pair-AO bytes, dense
equivalents and the measured complete pair reconstruction defect. The count
covers the pair build; the automatic compact selection shares its selected
buffers. The [bounded storage ladder](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/chi_pair_storage.md)
also counts the dense selection's separate snapshots. This is a retained
storage reduction, not a peak-memory cap. PAO/PNO projected matrices, occupied Fock, amplitudes and
prepared transport links remain dense. Complete selected frames and their
shared SC/FC arrays are absent on the compact chi path. The PAO builder
caches only cyclic metric-applied home columns; it evaluates the current
pair Fock through the original four-term expression, retaining both
occupied/virtual cross terms, including admitted Fock asymmetry. Full domains can produce projected
matrices with the full AO dimension. All-displacement actions and the factor
round-trip audit still perform substantial work. The existing shared compact
orbital-audit and reconstruction-recipe implementations for other adapters remain unchanged.

### Cyclic occupied preparation

Let $U_{Rk}=e^{ikR}/\sqrt N$ and $Q(k)$ be one admitted occupied
metric projector, with m bands per cell. For the chosen home AO indices J,
let $E_J$ select those columns and $D_{jj}=\sqrt{S_{0j,0j}}$.
The translated normalized seeds have symbol

$$B(k)=Q(k)E_JD^{-1},\qquad G(k)=B(k)^\dagger S(k)B(k).$$

The dense seed Gram is unitarily similar to the direct sum of all $G(k)$.
Hermitianization commutes with that transform. Its smallest and largest
eigenvalues are therefore the global extrema across every character block,
with no factor of N. The existing admission remains strictly
$\lambda_{\min}>\mathrm{seed\_lindep}\lambda_{\max}$.
Greedy candidate order, minimum-eigenvalue score and the tolerance-sized tie
envelope are unchanged. Finite-precision $Q^\dagger S Q$ is retained;
idempotence is never used to replace it with $S Q$.

For $G(k)=V_k\Lambda_kV_k^\dagger$, form its unique symmetric inverse
root $H(k)=V_k\Lambda_k^{-1/2}V_k^\dagger$ using the complete spectrum.
No eigenvector phase, ordering within a degeneracy or additional cutoff
selects a gauge. The selected home columns are

$$C_{R\mu,0i}=N^{-1}\sum_k e^{ikR}[B(k)H(k)]_{\mu i},
\qquad C_{R\mu,Ti}=C_{(R-T)\mu,0i}.$$

There are two unitary transforms, hence $1/N$ rather than $1/\sqrt N$.
The inverse transform checks every imaginary home entry before taking its
real part. At self-inverse characters the symbols are real; their complete
real matrix function is retained, including degenerate bands. Active and
frozen partitions are selected separately. `compact_storage=False` keeps the
explicit dense selector and real-span oracle; no budget failure chooses it.

The chi source real-span audit also has an exact character decomposition.
For a partition's original Bloch columns c, its block is
$A_k=[c_k,c_{-k}^*]/\sqrt2$. The corresponding metric block is
$M_k=[S_kc_k,(S_{-k}c_{-k})^*]/\sqrt2$; keeping this expression preserves
admitted source asymmetry instead of assuming it vanishes. At self-inverse
characters use $[\Re c_k,\Im c_k]$ and
$[\Re(S_kc_k),\Im(S_kc_k)]$ explicitly. The union of the Hermitianized
$A_k^\dagger M_k$ spectra is the original real-span Gram spectrum.
Every block must have m unit and m zero eigenvalues at `1e-10`, with no new
rank cutoff. A conjugate pair is reconstructed as a complete real/imaginary
column pair; self-inverse characters use the entire real block. Small
coefficient-space recipes are local to this call and included in its symbol
charge. Verified complete character representations evaluate these audits through
small real-character and cyclic displacement blocks. Unverified tables retain
the exhaustive real-column replay.

### Bounded reference phases and greedy reuse

Compact reference preparation generates positive-phase rectangles of at most
`N*d` entries on demand, with no retained phase cache. Inverse transforms tile
displacements and retain the complete character reduction; forward recovery
tiles characters and retains the complete displacement reduction. Both use
the original conjugations and normalization. Every overlap/Fock entry and
occupied-projector displacement remains audited. Exhaustive threshold replay
requests one phase row at a time. Immutable source, mesh and character-order
bindings prevent reuse across references; generic supplied tables still need
verification and otherwise retain exhaustive replay. The explicitly selected
dense factor oracle may construct its complete table under its separate scope.

For each active or frozen selection independently, define

$$H(k)=D^{-1}Q(k)^\dagger S(k)Q(k)D^{-1}.$$

Every candidate Gram is the principal submatrix `H(k)[J,J]`. One primitive
`N*b*b` cache therefore replaces repeated candidate contractions. This is an
algebraic rearrangement of the existing projected-seed criterion, not
idempotence substitution or screening. All candidate spectra and their global
extrema remain evaluated in the original order. A conservative roundoff guard
replays the entire original greedy step near strict rank admission or the tie
envelope. The winning conditioning score and final symmetric Lowdin gauge
use the original selected-column contraction. This preserves the original
arithmetic where it determines a boundary decision; no tolerance is loosened.
The cache is bound to projector, metric, mesh and partition identities and is
released on return or failure. It is never a translated-seed Gram.

`greedy_gram_work` records primitive builds, candidate eigenspectra, original
contractions, boundary replays and released bytes independently for each
partition. Reference-phase provenance records tile, generated entries, zero
cache/table bytes and its immutable binding. Gamma pays extra setup/cache
cost; ambiguous candidates can require both cached evaluation and replay.
The primitive cache remains quadratic in primitive AO count, and retained
source factors, downstream amplitudes and transport remain separate costs.
These changes preserve experimental scope and do not expand the bounded 0.18
release gate; production periodic DLPNO remains deferred under #557.

### Complete cyclic audit norms

For a selected partition X with immutable home columns $h_X[R]$, define
$c_X(k)=\sum_R e^{-ikR}h_X[R]$. The complete Gram and projector symbols are

$$g_{XY}(k)=c_X(k)^\dagger S(k)c_Y(k),\qquad
p_X(k)=c_X(k)c_X(k)^\dagger S(k).$$

Their home-row blocks are $N^{-1}\sum_k e^{-ikD}g_{XY}(k)$ and
$N^{-1}\sum_k e^{-ikD}p_X(k)$. Every displacement D is reconstructed.
Subtract the identity only at D=0 for a same-partition Gram; the cross Gram
has no identity subtraction. The largest real entry over all D is exactly
the original complete selected-orbital norm. Compare each projector block
with its own frozen or active source block before taking that same maximum.
The right factor is S, not its adjoint, preserving admitted metric asymmetry.
Neither symbol maxima, Frobenius norms nor a sample of D=0 is substituted.

The source real-span frame uses a different orbital basis. For a conjugate
pair, its real columns are $\sqrt2[\Re(U_kF_k),\Im(U_kF_k)]$.
With $g=F_k^\dagger M_k$, the original real Gram is
$[\Re g,\Im g;-\Im g,\Re g]$. Its norm is the maximum of the real
and imaginary entries separately, not the complex magnitude of g.
Self-inverse characters retain the complete real block. Both the original
metric-frame action and the admitted real S action are checked. Distinct
character pairs are orthogonal by the verified complete character table.
The normalized-span projector uses $F_kM_k^\dagger$ and is compared with
$c_k(S_kc_k)^\dagger$ at every AO entry and displacement, including its
imaginary component. Conjugate partners use the same recipe; degenerate
subspaces are never split.

Only exact immutable column recipes and cyclic operators with matching mesh,
cell-major ordering and source bindings receive structural reductions.
Character tables are checked against their actual mesh coordinates and
phases. For these representations every origin is a displacement permutation;
generator checks on all home entries therefore cover every complete column.
Generic dense/custom inputs retain the exhaustive streamed checks. A result
within a conservative floating-point guard of the admission threshold replays
the original arithmetic; the guard does not alter the acceptance tolerance.

A preparation context reuses at most one overlap and two occupied symbol
tensors plus sixteen measured scalar defects. Source, overlap, Fock, partition
and home-content identities bind reuse. No complete C, SC, FC, AO projector
or translated-seed Gram is cached. The context expires on return or exception; rebinding its source or budget
clears both symbol and scalar caches.
Provenance names the formulation, all measured metric/projector/cross-partition
defects, displacement counts, boundary replays and bounded cache counters.
Equivalent checks are reported as evaluations, not skipped scientific checks.
Source stationarity, factor audits, pair stabilizers and tolerances are unchanged.
The native transport boundary retains its independent streamed occupied checks
under its separate budget; cached Python measurements do not bypass that boundary.

`max_selection_bytes` preflights a conservative controlled numerical-payload
reservation for compact chi occupied-source realification, cyclic selection
and its admission. For N cells, b primitive AOs and m total occupied bands,
the recorded byte charges are `256*N*b*b` for symbols/transforms/copies,
`256*N*b*m` for home outputs/snapshots/reconstruction,
`96*N*b*b` for exhaustive audit tiles, `128*N*(b*b+b*m+m*m)` for
cyclic audit intermediates, `16*N*(b*b+b*m)` for the bounded symbol cache,
128 bytes for scalar measurements, `128*N*d` for bounded reference phases
with `d=min(8,max(1,N//2))`, `16*N*b*b` for the partition-local primitive
candidate-Gram cache,
and `384*(2*b)**2 + 128*b` for
eigensolver inputs/outputs/copies, `128*N*(b+m+3)` for indices/phases and
`128*np.getbufsize()` for NumPy iterator buffers. Each phase preflights
before its affected allocations. Cached symbols are included in this charge
even when reused by consumers during the same preparation. No cache persists
after the preparation returns.
The `accounted_peak_bytes` field is this conservative reservation, not a
measured process peak. The unavoidable selected-home output is `8*N*b*m`
bytes; the two source-projector outputs retain `16*N*b*b` bytes.

This cap excludes existing SCF/source arrays, canonical reference admission,
LAPACK/FFT/BLAS internal scratch,
Python objects, allocator overhead, already returned buffers and downstream
domain/PAO/PNO/native transport/solver work. The native transport cap separately
charges its home-metric cache and streamed reconstruction buffers; Python and
native binding copies are outside the default prepared transport cap. Full-domain projected
matrices, occupied Fock, pair amplitudes and the overlap graph can still be
quadratic or larger. Canonical adapter momentum metadata also retains an
`N*N*N` conservation index outside this cap. The [bounded selector ladder](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/chi_occupied_preparation.md)
reports both time and storage. No total-memory bound or linear scaling follows.

`max_factor_bytes` bounds numerical array payload during each serial streamed
provider call, including preparation's factor audit, coefficient conversion
and Fourier buffers, auxiliary tiles, temporary products, cached phases,
audit-local reuse and returned arrays until return. Calls on the same provider cannot overlap.
Returned pair matrices are unavoidable output storage; their size is charged
before allocation and reported as `max_output_bytes`. Once returned, arrays
retained by downstream consumers are outside this per-call cap. The provider
uses `einsum(optimize=False)` with explicit output arrays, so it makes no
native factor-binding copies or BLAS packing allocations. The ledger reserves
eight complex NumPy iterator buffers at the current `np.getbufsize()` setting.

For auxiliary width p and character/displacement width d, preparation charges
`192*N*d*p` audit bytes, `128*N*d` phase bytes and the iterator allowance.
The audit charge covers concurrently live recovery/home slabs, inverse
half-transforms, gathered source entries, reversed blocks, products and
finite/absolute-value temporaries. The phase charge covers two live blocks
and a replacement's construction plus immutable snapshot. No full table is
cached even when the requested cache ceiling is large.

After choosing this same tile plan, preparation may reserve another
`16*N*d*p` bytes to reuse its first displacement slab. This is the immutable
real transpose average, stored only after measuring the original imaginary
and symmetry defects. The reservation covers its snapshot alongside the
original and its subsequent lifetime alongside the remaining audit work.
Reuse requires spare workspace, more than one target tile, and room for both
this slab and the largest phase cache block under `factor_cache_bytes`.
It never reduces the chosen tile widths to make a cache fit. The slab belongs
to one source, AO pair and auxiliary tile, and is released before moving to
the next pair or returning.

With another `16*N*d*p` bytes of spare workspace and cache capacity, a second
immutable slab retains raw complex reversed-AO values `B_nu,mu[S,R]` for
the first S tile and all R. Later R and target-character tiles reuse these
values when measuring symmetry and forming the real transpose average.
The raw snapshot is never realified or averaged, so reuse does not assume
symmetry or conceal imaginary defects. Every original home slab still
undergoes the complete imaginary check. Remaining reverse blocks and home
slabs outside the first R tile are recomputed as before.

The baseline audit reservation already covers a `d*N*p` reverse output and
its half-transform. The second extra reservation covers its immutable copy
during construction and its retained lifetime alongside the first cache and
other audit work. Both slabs plus the largest phase snapshot must fit the
cache ceiling. Both are released per AO/auxiliary tile and before return;
d remains capped at eight and below N for non-Gamma meshes.
Every source entry is still recovered in the original sum order, including
null modes. Neither a complete home tensor nor a quadratic audit array returns.

For l left and r right columns, contractions retain the existing charges:
`64*N*b*(l+r)` conversion/Fourier bytes, `p*(64*l*r+32*l*b)` auxiliary work,
and either `48*r*r` pair-output/work bytes or `32*N*a*l*r` MO-output/work
bytes, plus bounded phases and the iterator allowance. Planning tries d from
`min(8,max(1,N//2))` down to one, choosing the largest fitting p up to
`factor_auxiliary_tile`. Infeasible budgets refuse before workspace allocation.
The minimum audit reservation is `iterator_allowance + 320*N` bytes; a later
contraction can still refuse because its output and Fourier buffers need more.

`factor_cache_bytes` permits one immutable phase block and the two optional
audit-local slabs described above. Zero disables caching; an insufficient
ceiling or workspace leaves either the first slab alone or complete
recomputation, with the same minimum and tile plan.
Cache replacement is charged, and a smaller call plan evicts an oversized
previous phase entry before allocation. No home factors survive preparation;
orbital arrays are never cached. `audit_reuse_workspace_bytes` records the
extra reservation, and `audit_reuse_stores` / `audit_reuse_hits` count slab
snapshots / subsequent uses. The corresponding `audit_reverse_reuse_*`
fields separately record the raw reversed-AO reservation, stores and hits.
`cache_bytes` reports only the phase block still
retained after return. Diagnostics also report the actual tile dimensions, minimum
reservation, accounted peak, cache traffic, generated phase entries, repeated
inverse blocks/entries, complete recovered entries and all measured defects.
The [bounded benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/chi_streamed_factors.md)
compares minimum and larger budgets with the former tiled and complete-table oracles.

The cap excludes pre-existing SCF factors, caller coefficients, reference
admission and orbital preparation (bounded separately by `max_selection_bytes`),
already returned buffers, downstream
PAO/PNO/solver allocations, Python metadata, allocator overhead and process
RSS. It is not a total-memory bound. `factor_provider_record` identifies the
preparation; `factor_provider_execution` additionally reports final cache
hits, largest tile/output, accounted peak and contraction imaginary defects.
Source hashing borrows contiguous canonical factor buffers rather than
copying a full block to bytes.

For the default prepared backend, `max_transport_bytes` is enforced by the shared native preparation before
allocating the prepared transport payload and its accounted workspace. It does
not cover SCF, this adapter, PAO/PNO preparation, Python metadata, binding
conversion copies of remaining inputs, solver output copies, allocator
overhead or total process memory. The adapter's `storage`
record reports retained array bytes only. No new total-memory guarantee or
production-scaling claim is made.

## Bounded transport links

`transport_backend="bounded"` explicitly selects native on-demand link access
with `compact_storage=True`. The default `"prepared"` retains all numerical
links as the small-system performance oracle. Both use the same compiled
Jacobi solver, complete pair inventory, occupied-Fock coupling order,
denominators, residual and half-cell stabilizers. No link is screened, even
when its Fock coefficient is zero.

For complete recipes $V_p=E_{D_p}A_p-CB_p$, a translated link is
$L_{pq}(d)=V_p^T S T_d V_q$. The stable branch forms the target metric from
selected S rows and the complete occupied projection, reconstructs one source
pair, then accumulates cell blocks in the prepared order. It does not replace
this contraction with the cancellation-prone two-term projected identity.
The prepared stability bound still selects the evaluation per target, using
all source links. Admitted metric asymmetry and every active/frozen tail
remain. This reschedules Riplinger and Neese Eq. (30) inside Nejad
Eqs. (42)-(44), without adopting either paper's pair screening.

Native preparation independently audits the metric, occupied columns,
recipes, pair metrics, stabilizers and every numerical link. Bounded mode
immediately discards each checked link. It owns immutable copies of metric
blocks, home columns, recipes and ordering, bound to the complete Python
source identity. During iteration a single-entry cache reuses an immediately
repeated link; other requests reconstruct it. A second single-entry cache
reuses the stable target metric across links with the same target. Specifically,
$M_p=A_p^T E_{D_p}^T S-(SCB_p)^T$ retains admission's original evaluation
order, including admitted metric asymmetry. Only $M_p$ is reused: each source
pair and cell-block product is evaluated as before. Target replacement releases
the old matrix first. Both caches belong to one solve over one immutable native
payload and are destroyed on success or exception. No all-pair AO pool or retained
numerical-link graph exists in this mode. Repeated reconstruction can be
substantially slower than prepared iteration, especially for stable links.

The bounded transport cap includes native route/link metadata, recipe and
metric snapshots, binding conversion copies, one link with live replacement,
selected-column and pair reconstruction buffers, and two-sided contraction
workspace. Target reuse additionally reserves `8*n_AO*max_pair_rank` bytes
plus its native cache descriptor, including coexistence with nonstable link
work. The single retained target requires no second live replacement; it is
conservatively charged separately from the existing stable reconstruction
rectangles. A separate Python reservation covers Fock outputs, home stacking,
recipe/metric hashing copies and one-pair contractions before allocation.
The exact accounted minimum is reported as `required_bytes`; smaller caps
refuse without fallback. Library packing scratch, interpreter/allocator
object overhead, pre-existing source data, downstream PAO/PNO preparation,
amplitudes/residuals and returned solver copies remain outside this cap.
It is not a total-process-memory guarantee.

Provenance separates numerical link bytes, recipe snapshots, route metadata,
Python workspace, retained/reserved payload, link and target-metric construction
counts, cache hits and evictions. `target_metric_cache_bytes` reports the added
reservation; `cache_bytes_after_solve=0` records the solve-local lifetime. The [transport benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/chi_pair_storage.md)
separates preparation and iteration from pair construction and reports traced
Python peaks separately from native reservations. Occupied Fock, projected
domains, routing metadata and amplitude arrays retain their previous scaling.
Chi stays experimental, D93 fitted/post-HF stays not-qualified, and #557's
production periodic DLPNO deferral is unchanged.

## Reproducible Validator handoff

[`chi_translation_mp2.py`](../examples/periodic/chi_translation_mp2.py) prepares
H2 or LiH/STO-3G with explicit finite classic supports, or reuses an accepted
HF archive through `--reference`. Stage it through the configured `vq` service
with a runtime containing this increment and the prepared native translation
solver. Keep output outside the checkout and inspect existing job identities
before submitting. Suggested cases:

```text
--case h2 --mesh 1 --output RESULTS/h2-m1
--case h2 --mesh 3 --output RESULTS/h2-m3
--case lih --mesh 4 --frozen-core 0 --output RESULTS/lih-m4
--reference RESULTS/lih-m4/hf.vqr --frozen-core 1 --output RESULTS/lih-m4-core
```

The short support and individual AO seeds are diagnostic defaults, not
guaranteed admissible LiH references or frames. For the even-mesh LiH
fixture, explicitly use `--one-electron-cutoff-bohr 12 --seed-basis cosine`
and retain `--seed-basis cosine` when reusing its archive. Wider one-electron
support defines a different finite Hamiltonian; compare only with a canonical
calculation consuming that same archive. A positive metric and an insulating
accepted reference are still required. The support option applies only to
fresh HF preparation; a reused archive retains its recorded support.

For example, copy the single example script to an external `CASE_DIR`, select
the verified registered runtime interpreter as `RUNTIME_PYTHON`, and submit:

```text
vq submit --dir CASE_DIR --cpus 4 --mem-mb 8192 --time-limit 01:00:00 -- \
  RUNTIME_PYTHON chi_translation_mp2.py --case h2 --mesh 3 --output results
```

These resource values are initial reservations, not measured requirements.
Use an installed runtime whose source contains the delivery commit recorded
on #508. Record its native identity from the result manifest; an older runtime
cannot qualify this adapter. A reused archive must also be staged into the
case directory. No source checkout or environment copy is needed.

The example runs both factor providers and both coefficient storage modes
by default, compares their raw energies,
residuals and separately reported PNO corrections, and records their storage
ledgers. Select only `--factor-providers streamed` for sizes beyond the dense
oracle's scope and `--coefficient-storage compact` to omit dense coefficients.
`--max-factor-bytes`, `--factor-auxiliary-tile` and
`--factor-cache-bytes` reproduce budget experiments. `--max-selection-bytes`
controls the separate occupied-preparation reservation. Use
`--transport-backend bounded --coefficient-storage compact` and
`--max-transport-bytes` for bounded link cases; reuse the same archived
reference for the prepared comparison. Record the SHA-256 of
the installed `vibeqc.dlpno.translation_bloch_factors` module and pass it as
`--expected-provider-sha256` to refuse a runtime with different source. The
coefficient implementation is checked with `--expected-storage-sha256`
(SHA-256 of `vibeqc.dlpno.translation_storage`); `--expected-core-sha256`
checks the loaded native module. `--expected-preparation-sha256` checks
`vibeqc.dlpno.translation_preparation`. `--expected-implementation-sha256`
binds all thirteen occupied-preparation/consumer modules in a deterministic
digest; use it with the native hash to identify the containing runtime.
The result records these hashes and
the actual loaded native identity. A prepared case is not
a scientific verdict or evidence that a future managed runtime contains it.

The full-space raw solution is compared with independently evaluated
same-source canonical MP2 at `2e-9` Ha/cell, with residual `1e-11` and energy
change `1e-13` gates; all three are checked and the last energy change is
recorded. Tighten these if residual accumulation approaches the
energy tolerance. The example varies domain thresholds at zero PNO cutoff,
then PNO cutoffs at full domains, recording raw errors, corrections, ranks,
iterations and refusals. Threshold errors need not be monotonic.

For a containing-runtime comparison of prepared and bounded transport, keep
the accepted archive bytes, source/Fock/factor identities, seed basis and
frozen partition fixed. Run each LiH frozen partition independently; a pass
with one frozen orbital does not qualify the all-electron partition. Reuse
the accepted canonical and independent same-source reference evidence, then
check the changed consumer against it. Record the loaded Python and native
identities separately from the archive producer. An earlier prepared-backend
pass does not qualify a later bounded implementation.

Compare representative amplitudes, residual tensors and iteration traces as
well as raw energies and separate PNO estimates. Pair builds, selected active
and frozen projectors, ranks and electron counts must agree between backends.
Keep the full-space canonical gate separate from domain and PNO errors;
truncated rows still need their own convergence verdict. None of these checks
permits omitting an occupied-Fock coupling or relaxing a source audit.

Inspect `stable_selected_column_links`, `target_metric_constructions` and
`target_metric_cache_hits` to establish whether a case exercises target reuse.
A case with zero stable links checks the nonstable path only. Gamma overhead
or a small finite-source timing does not establish scaling. Record retained
links, the target reservation and the documented transport budget scope;
`cache_bytes_after_solve=0` is a lifetime check, not a total-process memory
measurement. Keep scientific results and immutable runtime receipts on the
acceptance issue. This protocol remains outside the 0.18 release gate.

Source checks exercise a nonorthogonal finite model with occupied-Fock
coupling, independently evaluated dense canonical MP2, complex Bloch gauges,
frozen core, translation covariance and even-mesh self-inverse multiplicities.
They also reject source mismatch and unsupported inputs and forbid historical
dense reconstruction. These are implementation evidence. Validator must record
the actual containing-source/native runtime and matched conventions, reuse
qualifying #492/#530 operands, and assess physical support/cell convergence and
ORCA/CRYSTAL comparisons separately. No existing scientific job was replaced by
this development increment.

On the deterministic nonorthogonal source model, the largest full-space raw
error across meshes `(1,1,1)`, `(3,1,1)`, `(4,1,1)` with one frozen orbital
per character, and `(2,2,1)` was `5.52e-15` Ha/cell. The largest final raw
residual was `1.78e-13`; the multi-cell cases required eight iterations with
nonzero occupied-Fock coupling. These values are algebra checks, not results
for H2 or LiH. The separate three-character threshold experiments measured:

| Domain cutoff | PNO cutoff | Raw error (Ha/cell) | PNO estimate (Ha/cell) | PAO/PNO rank ranges |
|---|---|---|---|---|
| 0 | 0 | `+1.72e-16` | `-1.14e-18` | 6 / 6 |
| 0.05 | 0 | `+4.34e-3` | below `1e-18` | 1-2 / 1-2 |
| 0.01 | 0 | `+4.34e-3` | below `1e-18` | 1-2 / 1-2 |
| 0 | `1e-4` | `+1.01e-3` | `-1.03e-3` | 6 / 1-3 |
| 0 | `1e-6` | `+1.90e-6` | `-1.96e-6` | 6 / 4-6 |

All five solves converged below `2e-14` raw residual. The PNO estimates are
reported independently and do not enter the full-space raw comparison.

The occupied audits use verified cyclic blocks. Compact reference phases are
bounded and greedy primitive-Gram contractions are reused. Pair reconstruction
remains dense one pair at a time, and the default prepared overlap graph remains
retained; bounded transport can instead reconstruct links during iteration. Repeated inverse-block reconstruction in the complete factor audit
belongs to the separate factor work. Optimized localization, calibrated
screening, unrestricted equations and local CC remain separate work.


## Internal CCSD algebra follow-up

The [translation-native restricted CCSD residual evaluator](translation_ccsd_residual.md)
reuses this preparation and bounded transport to evaluate complete native
singles/doubles residuals and raw correlation energy at arbitrary amplitudes.
It shares the molecular PNO contractions and is not an iterative CCSD method.
Its CC workspace accounting and pending scientific acceptance are separate
from the MP2 capability recorded here.
