# Experimental chi translation local MP2

`vibeqc.periodic.chi.local_mp2.run_chi_local_mp2` connects an admitted
classic physical-pair RI RHF `ChiHFReference` to the shared translation-aware
local-MP2 engine. This is an explicit experimental library route. Chi remains
experimental and D93 fitted/post-HF remains not-qualified. Source tests do not
establish scientific acceptance, production threshold accuracy or scaling.

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

Admission folds one primitive AO pair and an auxiliary tile at a time into
four reusable `N*N*tile` complex arrays. It compares every real-space entry
with the separately transformed reversed AO pair, measures every imaginary
entry, and forward-transforms the admitted real transpose average to compare
every retained Bloch entry. This includes all character pairs and null modes.
The phase translation law is checked for every character, origin and cyclic
shift, including even-mesh self-inverse cells. The sealed provider's
constructive covariance then follows from the exact auxiliary-cell sum.
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
buffers. The [bounded storage ladder](../benchmarks/chi_pair_storage.md)
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
charge. Verified complete character tables now evaluate these audits through
small real-character and cyclic displacement blocks. Unverified tables retain
the exhaustive real-column replay.

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
or translated-seed Gram is cached. The context expires on return or exception.
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
128 bytes for scalar measurements, `64*N*N` for phase validation/snapshots,
and `384*(2*b)**2 + 128*b` for
eigensolver inputs/outputs/copies, `128*N*(b+m+3)` for indices/phases and
`128*np.getbufsize()` for NumPy iterator buffers. Each phase preflights
before its affected allocations. Cached symbols are included in this charge
even when reused by consumers during the same preparation. No cache persists
after the preparation returns.
The `accounted_peak_bytes` field is this conservative reservation, not a
measured process peak. The unavoidable selected-home output is `8*N*b*m`
bytes; the two source-projector outputs retain `16*N*b*b` bytes.

This cap excludes existing SCF/source arrays, the adapter's shared character
phase table, canonical reference admission, LAPACK/FFT/BLAS internal scratch,
Python objects, allocator overhead, already returned buffers and downstream
domain/PAO/PNO/native transport/solver work. The native transport cap separately
charges its home-metric cache and streamed reconstruction buffers; Python and
native binding copies are outside that transport cap. Full-domain projected
matrices, occupied Fock, pair amplitudes and the overlap graph can still be
quadratic or larger. The [bounded selector ladder](../benchmarks/chi_occupied_preparation.md)
reports both time and storage. No total-memory bound or linear scaling follows.

`max_factor_bytes` bounds numerical array payload during each serial streamed
provider call, including preparation's factor audit, coefficient conversion
and Fourier buffers, auxiliary tiles, temporary products, cached phases and
returned arrays until return. Calls on the same provider cannot overlap.
Returned pair matrices are unavoidable output storage; their size is charged
before allocation and reported as `max_output_bytes`. Once returned, arrays
retained by downstream consumers are outside this per-call cap. The provider
uses `einsum(optimize=False)` with explicit output arrays, so it makes no
native factor-binding copies or BLAS packing allocations. The ledger reserves
eight complex NumPy iterator buffers at the current `np.getbufsize()` setting.

For auxiliary tile width p, preparation accounts for `96*N*N*p` tile bytes,
plus `64*N*N` bytes for the phase table and immutable snapshot, and the
iterator allowance. For l left and r right
columns, a contraction accounts for `64*N*b*(l+r)` conversion/Fourier bytes,
`p*(64*l*r+32*l*b)` tile bytes, and either `48*r*r` pair-output/work bytes or
`32*N*a*l*r` complete-MO-output/work bytes, plus phases and iterator allowance.
These conservative charges include temporary absolute-value/finiteness
buffers. The largest fitting p is selected up to `factor_auxiliary_tile`;
an impossible one-auxiliary tile fails before the affected allocation.
`factor_cache_bytes` permits one complete phase table if it fits, otherwise
phases are recomputed. No factors or orbital arrays are cached. The cache is
already included in the phase charge; raising its ceiling cannot exceed the
total factor cap. A minimum audit tile still grows as N squared.

The cap excludes pre-existing SCF factors, caller coefficients, reference
admission and orbital preparation, already returned buffers, downstream
PAO/PNO/solver allocations, Python metadata, allocator overhead and process
RSS. It is not a total-memory bound. `factor_provider_record` identifies the
preparation; `factor_provider_execution` additionally reports final cache
hits, largest tile/output, accounted peak and contraction imaginary defects.
Source hashing borrows contiguous canonical factor buffers rather than
copying a full block to bytes.

`max_transport_bytes` is enforced by the shared native preparation before
allocating the prepared transport payload and its accounted workspace. It does
not cover SCF, this adapter, PAO/PNO preparation, Python metadata, binding
conversion copies of remaining inputs, solver output copies, allocator
overhead or total process memory. The adapter's `storage`
record reports retained array bytes only. No new total-memory guarantee or
production-scaling claim is made.

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
controls the separate occupied-preparation reservation. Record the SHA-256 of
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
change `1e-13` targets. Tighten these if residual accumulation approaches the
energy tolerance. The example varies domain thresholds at zero PNO cutoff,
then PNO cutoffs at full domains, recording raw errors, corrections, ranks,
iterations and refusals. Threshold errors need not be monotonic.

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

The occupied audits now use verified cyclic blocks. The next bounded
increment is to reduce repeated candidate Gram construction in greedy seed
selection while preserving its conditioning scores and tie envelope. Pair reconstruction remains dense
one pair at a time, and the prepared overlap graph remains retained. The
complete source audit can also be tiled over characters to lower its minimum
N-squared workspace. Optimized
localization, calibrated screening, unrestricted equations and local CC remain
separate work.
