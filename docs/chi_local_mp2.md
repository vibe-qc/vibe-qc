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
`TranslationHomeFactors`, selection, PAO/PNO and iterative solver interfaces.
Per-character frozen and active projectors are preserved without introducing
D6's global energy sorting or its full virtual-orbital payload.

```python
from vibeqc.periodic.chi.local_mp2 import run_chi_local_mp2

result = run_chi_local_mp2(
    reference, frozen_core=0,
    tcut_mkn=0.0, tcut_pno=0.0,
    max_iter=250, conv_tol_energy=1e-13, conv_tol_residual=1e-11,
    damping=1.0, max_transport_bytes=256 * 1024**2,
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

The existing native home-auxiliary factor transform is reused. It must produce
real symmetric factors, and a forward transform must reproduce **every**
retained complex AO factor block within `1e-10` absolute tolerance. This checks
the full canonical auxiliary frame, including its null space, without refitting
or changing retained ranks. Incompatible auxiliary gauges refuse. Small
floating-point imaginary/symmetry residuals are measured and recorded; a
significant imaginary component is never discarded.

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
Home-auxiliary factors still store `a*(N*b)**2` real numbers for a primitive
auxiliary functions, plus the retained Bloch source. The factor transform and
immutable snapshot can coexist temporarily. Complex folded occupied columns,
real candidate spans and their metric Gram matrices, selected orbitals, pair
AO coefficients, PAO/PNO projected matrices, occupied Fock, amplitudes and
prepared transport links remain dense. Full domains can produce projected
matrices with the full AO dimension. All-displacement actions and the factor
round-trip audit still perform substantial work.

`max_transport_bytes` is enforced by the shared native preparation before
allocating the prepared transport payload and its accounted workspace. It does
not cover SCF, this adapter, PAO/PNO preparation, Python metadata, solver output
copies, allocator overhead or total process memory. The adapter's `storage`
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

The next bounded increment is to reduce the remaining occupied/factor and pair
AO payloads while preserving these same-source and raw-limit audits. Optimized
localization, calibrated screening, unrestricted equations and local CC remain
separate work.
