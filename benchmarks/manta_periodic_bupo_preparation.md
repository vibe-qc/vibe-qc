# Complete native BUPO HF source preparation

Run `python benchmarks/manta_periodic_bupo_preparation.py` in the checkout's
built environment. Each fixture uses a separate process, one numerical
thread and at least three repetitions. Timings separate source preparation
from the complete HF run over the newly prepared source. Preparation builds
AO images, moments, hierarchy, finite Ewald support, the admitted near/far
partition and certified reciprocal exchange spheres. No density or SCF
history is retained by the prepared handle.

The fixtures are compact synthetic all-electron cells, matching the
independently validated cutoff cases. The shifted hydrogen UHF case has
an empty beta channel. The shifted lithium case is capped after one
evaluated density and is explicitly nonconverged. These inputs are cost
probes, not general accuracy profiles.

The cumulative source reservation sums existing stage peak models and
therefore double-counts shared parent storage and workspaces. It remains
live through HF. The final planned peak also reserves candidate exchange
support; actual-support modeling uses the generated vectors. Neither is
an RSS prediction. RSS includes Python, libraries and the explicit-source
fixture retained for comparison. There is no speedup or scaling claim.

Scientific acceptance compares the prepared route with original explicit
source construction and independent Gaussian/Fourier/eigensolver results
for identical physical inputs. Existing valid reference artifacts may be
reused when their source identities, settings and arrays match. There is
no matched ORCA/CRYSTAL custom finite-support certificate; no external
production-parity claim follows from these tests. Evaluated-density
integral bounds exclude basis, mesh, relaxed-state and roundoff errors.

## Measured construction and HF cost

Three repetitions per process, one numerical thread, ARM64 macOS and
Python 3.14.8. The host also ran unrelated work; these are not idle-host
or comparative performance measurements. The medians are reported
separately, so their sum need not equal the median complete time.

| Case | Preparation (ms) | HF (s) | Complete (s) | Source reservation (MB) | Planned peak (MB) | Actual-support model (MB) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rhf-gamma | 10.63 | 0.553 | 0.563 | 192.69 | 497.03 | 455.93 | 218.00 |
| uhf-shifted-empty-beta | 38.41 | 1.956 | 1.995 | 192.69 | 812.34 | 648.09 | 232.08 |
| uhf-shifted-capped | 92.70 | 3.997 | 4.086 | 197.42 | 1398.33 | 1016.93 | 262.14 |

The fixtures use reciprocal cutoffs 500, 500 and 900 hartree, respectively,
with 115312, 461416 and 1114900 exchange vectors. The first two converge;
the lithium timing covers its explicitly capped one-step state.

Timing native SHA-256:
`c2610fd0a4620dad12ef7a8e9ea1dae053009d07291b4c58c6a5c4263398a515`.
This runtime includes the prepared-source implementation on base
`486ba8a9`. The prepared-source implementation stayed byte-identical after integration
with subsequent main changes. The test inventory was regenerated from the
merged classification table.
Scientific acceptance below uses the integrated runtime and records its
own identity. The earlier valid timing evidence is retained.

## Scientific acceptance

The integrated source `f0af381d` was exercised through one isolated local
vq calculation job, using one CPU and a 2 GiB memory allocation. Native
SHA-256: `d5d04e063e3edce13ef524cb0629a62a3e0270d6f4ceb37979d962304920f464`.
Ten executed calculations cover prepared and explicit construction for
Gamma helium RHF, skew-cell helium RHF, shifted hydrogen empty-beta UHF,
shifted lithium UHF and capped lithium UHF. The complete command took
115.71 seconds, used 115.36 CPU seconds and reached 316.27 MiB peak RSS.

All density, Fock, core Hamiltonian and overlap arrays, energies and final
physical source/state identities match the explicit controls and the
previously accepted cutoff results exactly. The independent Gaussian,
ordered polynomial Fourier and generalized-eigenproblem references from
that acceptance were reused only after exact physical-input, source and
numerical-state checks. Their generator and input test files are unchanged.
Fetched artifacts were assessed separately with NumPy, without native QC.

Maximum Fock difference is 2.621e-13, same-density energy difference
1.765e-11 hartree, relaxed energy difference 1.892e-11 hartree and density
difference 9.750e-10. The original tolerances remain 3e-11 for Fock/energy
and 3e-9 for density, with the existing relative allowances. The largest
energy-trace difference is 5.152e-12 hartree. Every evaluated density passes
its electrostatic, S/T and full-exchange targets. Two deliberately
insufficient exchange cutoffs fail those targets. Capped states retain
`ITERATION_LIMIT`, their evaluated density/Fock/energy and no convergence
confirmation. Shared Fock consumers preserve the same data.

Sixteen new preparation cases, twelve existing cutoff cases, nine original
HF adapter cases and two inventory/citation cases pass. Three relevant
checks were repeated after integration and the required local basis-overlay
refresh. No full CI or separate review chain was used. Automatic job
selection, whole-method profiles and source-qualified symmetry remain open;
these results do not bound basis, mesh, relaxed-state/rank, Madelung
numerical or floating-point errors.

The prepared-source implementation lands as `60bb3aa8` after a
concurrent main update. All numerical implementation and reference files
remain byte-identical to the accepted source above. The incoming citation
changes affect chi methods only; the test inventory was regenerated.
