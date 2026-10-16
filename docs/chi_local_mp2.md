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
    compact_storage=True,
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
nonzero metric-Gram eigenvalues. The constructed real frame must reproduce
the original **complex** metric projector, checked in AO row blocks. Active
and frozen spaces are checked independently, including at self-inverse
characters. Arbitrary orbital phases are allowed; a complex projector is not.

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
remain separate. Every original selected column is compared with its
reconstruction at the existing `1e-10` maximum-entry tolerance; the complete
source partition, metric and translation audits run on the reconstructed
frames. Selection and pair spaces share the same immutable home buffers.
No coefficient or occupied tail is screened.

Pair storage reuses the shared PAO recipes `V = E_D A - C B`, where C is the
active/frozen union. The builder compares every reconstructed PAO and rotated
PNO coefficient with the dense algebra, then checks the complete metric,
occupied exclusion and self-inverse translation stabilizer as before. It
retains A, B and domain indices, reconstructing one pair at a time for factor
contractions. Native preparation receives no dense pair-AO pool or binding
copy of that pool. Its stable selected-column branch reconstructs one source
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
storage reduction, not a peak-memory cap. Complex folded reference columns,
real candidate spans, selector Gram matrices, transient complete selected
frames, shared SC/FC, PAO/PNO projected matrices, occupied Fock, amplitudes and
prepared transport links remain dense. Full domains can produce projected
matrices with the full AO dimension. All-displacement actions and the factor
round-trip audit still perform substantial work. The existing shared compact
orbital-audit and reconstruction-recipe implementations remain unchanged.

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
`--factor-cache-bytes` reproduce budget experiments. Record the SHA-256 of
the installed `vibeqc.dlpno.translation_bloch_factors` module and pass it as
`--expected-provider-sha256` to refuse a runtime with different source. The
coefficient implementation is checked with `--expected-storage-sha256`
(SHA-256 of `vibeqc.dlpno.translation_storage`); `--expected-core-sha256`
checks the loaded native module. The result records all three hashes and
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

The next bounded increment is to remove transient complete selected occupied
frames and the full selector Gram eigensystem, while preserving complete
projector and seed-conditioning audits. Pair reconstruction remains dense
one pair at a time, and the prepared overlap graph remains retained. The
complete source audit can also be tiled over characters to lower its minimum
N-squared workspace. Optimized
localization, calibrated screening, unrestricted equations and local CC remain
separate work.
