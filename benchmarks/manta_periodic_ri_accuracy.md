# Explicit periodic RI-BUPO accuracy and cost

This benchmark measures the shared version-two isolated erfc error model:
joint polynomial/Gaussian radial bounds and interior absolute-moment bounds.
It preserves the exact finite metric and references but may accept a different
fixed multipole partition at a given requested coupling tolerance. Approximate
source identities therefore change. The derivation is in the scientific
contract; Gaussian tails, arithmetic and physical-source qualifications remain
separate. No automatic accuracy profile is selected.

## Reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/manta_periodic_ri_accuracy.py --workers 3 --repeats 3
```

Three sequential processes measure one cold and three repeated calls.
Fixed-density rows use the same two-AO, six-auxiliary-function, two-k-point
rank-six metric as the pair-tree action benchmark. Orders two, four and six
use requested coupling tolerances 1e-4, 1e-6 and 1e-8. Exact finite RI-J is
the reference. Metric/builder preparation time is separate from action time;
all relevant native/source hashes, work counters and admitted peaks are saved.

Complete shifted Be RHF rows compare the exact finite coupling with orders
two, four and six at 1e-8. They include source setup, SCF and managed output;
each repeat is a new complete job, not reuse of a converged density. Relaxed
energy and density differences from the exact-coupling calculation are
reported separately from the fixed-density approximation bounds. These small
fixtures establish neither bulk ORCA/CRYSTAL parity nor a production crossover.


## Measurement and scientific acceptance

Calculation and benchmark source is `8254757647c1e7a071fe252b8fcbe1e72d74d8a0`. Native SHA-256 is
`4ee74cfd69a82a3beefd753c5ce4a24ad6e396f6735aa1097e54f82460695644`. Three sequential processes, each with one cold and
three repeated calls, give the following median times with one computation
thread. Exact finite RI-J takes 19.014 ms cold and
19.351 ms repeated. Metric/builder setup is about 1.2 ms.

| Order | Requested coupling error | Cold ms | Warm ms | Max J error | J bound | Energy bound (hartree) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1e-04 | 70.366 | 18.681 | 3.038e-13 | 7.595e-02 | 7.079e-06 |
| 2 | 1e-06 | 70.567 | 19.385 | 5.451e-13 | 1.093e-03 | 1.006e-07 |
| 2 | 1e-08 | 71.296 | 19.962 | 7.494e-16 | 5.694e-06 | 5.239e-10 |
| 4 | 1e-04 | 139.910 | 42.637 | 4.283e-15 | 6.960e-02 | 6.480e-06 |
| 4 | 1e-06 | 141.334 | 43.119 | 6.538e-15 | 9.251e-04 | 8.513e-08 |
| 4 | 1e-08 | 140.084 | 43.149 | 9.282e-17 | 8.298e-06 | 7.635e-10 |
| 6 | 1e-04 | 1178.856 | 1173.139 | 7.119e-14 | 9.459e-02 | 8.844e-06 |
| 6 | 1e-06 | 1185.851 | 1177.934 | 3.275e-17 | 6.366e-04 | 5.858e-08 |
| 6 | 1e-08 | 1193.352 | 1186.721 | 2.777e-17 | 8.903e-06 | 8.192e-10 |

At the strictest tolerance, the evaluated coupling bounds are 2.48e-10 to
3.63e-10. Between 22,572 and 22,700 accepted far blocks and 1,872 to 2,116
multi-pair groups remain active. Actual fixed-density energy errors are
1.73e-17 to 1.32e-16 hartree, distinct from the much larger conservative
bounds. The J bound remains sensitive to the retained metric. A requested
coupling tolerance must not be presented as a J or whole-method tolerance.

Admitted peak bytes at orders two/four/six are 232,567,992 / 325,602,984 /
676,487,528. Measured process peak RSS across all cases is 523-557 MB. At
order six, recorded partition reuse is zero and partition builds reach
2,544, versus 636 builds and 1,908 reuses at orders two/four. This bounded
cache behavior accompanies the high order-six cost and needs further work;
raising order does not automatically improve time to solution.

All complete RHF jobs below converge in nine iterations. Orders two and four
use coupling tolerance 1e-8 and respectively retain 464 and 416 far blocks.
Repeated times include complete source preparation, SCF and managed output.

| Order | Cold ms | Repeated complete job ms | Relaxed energy difference (hartree) | Fixed-density energy bound (hartree) |
|---:|---:|---:|---:|---:|
| Exact finite RI | 186.186 | 109.156 | 0.000e+00 | 0.000e+00 |
| 2 | 153.063 | 155.225 | 1.421e-14 | 1.623e-06 |
| 4 | 167.992 | 168.325 | 1.066e-14 | 2.066e-06 |

Relaxed density differences are below 3.6e-14. Relaxed differences are observed
comparisons, not certified by the fixed-density bounds. Complete RHF is slower
than exact finite RI on this small fixture. These measurements establish
explicit accuracy behavior, not an automatic profile or general speedup.

The focused evidence comprises 685 passing scientific checks, one skip and
12 documentation checks. Thirty-one checks are new: independent quadrature of
the joint radial supremum, compact signed point distributions, interior moment
caps, diffuse-tail retention, boundary/positive-underflow cases and two tight
public RHF calculations. Prior lattice, Gaussian, raw map, resource, rank,
energy derivative and shared Fock gates remain in force. The latest BIPOLE
automatic density-qualified symmetry and native per-cell Fourier changes were
integrated before rebuilding. The 134 directly affected erfc/pair-tree checks
were then rerun on the final native runtime; unaffected earlier checks are
reused with their original provenance.

Complete local vq acceptance passes nine public SCF states and fourteen raw
action cases. Public cases cover shifted Be RHF, empty-beta H UHF, converged
and capped Li UHF, Be PBE RKS, converged and capped Li PBE UKS, plus Be RHF at
orders two/four and coupling tolerance 1e-8. Capped cases remain explicitly
nonconverged. Raw cases cover Gamma/nonzero q, orders zero/two/four, signed
Cartesian/spherical d shells, physical erfc and four strict-tolerance maps.
All 1,877 input pins, 81 managed outputs and 23 result archives verify.

A separate NumPy-only assessment of the fetched arrays passes. Maximum Fock,
density and energy differences from independent approximate-source references
are 4.83e-15, 1.52e-9 and 1.14e-13 hartree, respectively. Maximum raw-map and
complex-duality disagreements are 1.26e-15 and 5.03e-15. Actual approximation
errors against the exact finite source satisfy the component and Euclidean
action bounds and the propagated fixed-density J/energy bounds. The seventeen
predecessor cases retain bitwise-identical exact metric/reference arrays and
matching exact identities; approximate coupling identities change as required
because their accepted partitions can change.

No matched ORCA/CRYSTAL reference is available for these custom finite domains.
References use independent Gaussian/Boys/Fourier formulas, libint erfc and
NumPy retained-metric/SciPy SCF calculations; KS shares libxc. The approximate
partition is not independently reimplemented. Neese et al., DOI
`10.1021/acs.jpca.4c07415`, equations 29-36, remain the full-moment action
reference. The new isolated erfc bound is the explicit Taylor derivation in
the scientific contract, not a harmonic Coulomb remainder attributed to that
paper. Finite lattice complements retain their previous remainder formula.

The shared producer API, exact metric, method-owned K/XC and unified Fock
composition are preserved. No new BIPOLE/CRYSTAL parity or symmetry-reduction
permission is inferred for the approximate source. Domain/basis errors,
roundoff and density relaxation remain outside this approximation ledger.
Automatic whole-method accuracy, production crossover and the rest of the
approved MANTA 0.18.x capability matrix remain unfinished.


## Compact partition storage measurement

The lossless storage increment uses source `6bb8d7b46778d0168aa3dc22f593926df80eb900` and native SHA-256
`f88387edf5cf77d0fbc736a29c28b44c381b3b1e46960fd0af56c9fce1a3274e`. It retains only allowed total-degree coefficients,
restores the same dense matrix before contraction, stores the actual accepted
antichain and evicts according to actual charged entry bytes. At order six,
924 coefficients replace 7056 stored matrix entries. The existing 64 MiB
partition cap, finite source, partitions, metric and numerical identities stay
fixed. An oversized entry is used without retention or duplicate construction.

The same three-process, three-repeat, one-thread protocol gives these results
at requested coupling error 1e-8. Previous values are the accepted measurements
above, not a fresh run with a second runtime.

| Order | Previous cold ms | Compact cold ms | Previous warm ms | Compact warm ms |
|---:|---:|---:|---:|---:|
| 2 | 71.296 | 69.257 | 19.962 | 19.415 |
| 4 | 140.084 | 138.663 | 43.149 | 44.220 |
| 6 | 1193.352 | 429.179 | 1186.721 | 166.377 |

Repeated order-six RI-J is about 7.1 times faster on this fixture; the other
two tolerances show 6.8-6.9 times improvement. The first call has 636 partition
builds and 1908 reuses instead of 2544 builds and no reuse. Separate tests
verify that repeated order-six builds have no new partitions or trials.
Measured process peak RSS is 389 MB, versus 523-557 MB in the prior sweep.
Conservative admitted peak bytes are unchanged, including the complete-entry
construction workspace. Packing/expansion work is explicitly admitted.
Orders two/four remain close to their earlier times; the data do not establish
a broad crossover or universal performance benefit.

Actual fixed-density J/energy errors and their conservative bounds are exactly
unchanged across all nine approximate benchmark rows. Exact finite RI-J takes
18.781 ms repeated. The complete-job benchmark now includes order six:

| Complete RHF | Repeated complete job ms | Relaxed energy difference (hartree) |
|---|---:|---:|
| Exact finite RI | 103.173 | 0.000e+00 |
| Order 2, coupling tolerance 1e-8 | 146.773 | 1.421e-14 |
| Order 4, coupling tolerance 1e-8 | 158.090 | 1.066e-14 |
| Order 6, coupling tolerance 1e-8 | 210.813 | 1.066e-14 |

All four complete RHF rows converge in nine iterations. Relaxed differences
remain observations rather than fixed-density certificates. Complete RHF is
still slower than exact finite RI on this small fixture; automatic accuracy
selection and production crossover remain required work.

Scientific acceptance passes 373 focused checks, including 13 new cases for
compact storage, actual-size retention, eviction, oversized-entry reuse,
orders eight/twelve and complete order-six RHF. Existing concurrent calls,
all resource limits, independent finite references, raw maps, shared Fock,
metric ranks and source/gauge checks pass. Twelve documentation checks pass
after adding required bullet markers to three incoming changelog fragments.
The current shared Fourier batching change was integrated before rebuilding.

Complete local vq acceptance passes ten public HF/KS states and sixteen raw
cases on this same pinned runtime. It adds order-six RHF and strict-tolerance
Gamma/nonzero-q raw maps to the prior cases. All 1877 source/reference pins,
90 managed outputs and 26 result archives verify. A separate NumPy-only
assessment confirms all error bounds and independent SCF references; 340
arrays across all 23 predecessor cases are bitwise unchanged, including
couplings, metric, Fock, densities, energies and saved spin densities.
Numerical coupling, exact-reference and fixed-factor identities are unchanged.

The prior Gaussian/Boys/Fourier, libint and NumPy/SciPy references and their
limitations still apply; no matched ORCA/CRYSTAL reference is available for
these custom finite domains. No new BIPOLE/CRYSTAL parity, method or symmetry
permission follows from storage reuse. The full MANTA 0.18.x matrix remains
unfinished.


## Componentwise retained-factor error transport

Source `38c261a0896e30085377827b3b89cf5630629ab5` and native SHA-256
`029f82d39935a5fbd6c643fabf41dbde7297bf374aae238b160875ad8cfce428` retain the same numerical coupling,
metric and paired RI maps as the compact-storage checkpoint. The shared raw
producer now exports componentwise omission and exact-reference auxiliary
column envelopes. RI propagates these through the actual retained factor,
keeping its previous scalar bounds whenever tighter. The scientific contract
derives the positive triangle inequalities; no ideal inverse or cancellation
is assumed. This changes diagnostics and their admitted workspace, not the
partition or numerical source identity.

The same three-process, three-repeat, one-thread protocol gives:

| Order | Coupling tolerance | Warm ms | Max J error | J bound | J-bound tightening factor | Energy bound (hartree) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1e-04 | 19.288 | 3.038e-13 | 5.301e-03 | 14.33 | 1.744e-06 |
| 2 | 1e-06 | 20.076 | 5.451e-13 | 1.869e-04 | 5.85 | 7.688e-08 |
| 2 | 1e-08 | 20.474 | 7.494e-16 | 5.816e-07 | 9.79 | 1.689e-10 |
| 4 | 1e-04 | 44.463 | 4.283e-15 | 8.094e-03 | 8.60 | 3.021e-06 |
| 4 | 1e-06 | 45.470 | 6.538e-15 | 1.364e-04 | 6.78 | 4.606e-08 |
| 4 | 1e-08 | 45.648 | 9.282e-17 | 1.323e-06 | 6.27 | 5.460e-10 |
| 6 | 1e-04 | 174.722 | 7.119e-14 | 1.114e-02 | 8.49 | 4.013e-06 |
| 6 | 1e-06 | 172.151 | 3.275e-17 | 9.385e-05 | 6.78 | 3.262e-08 |
| 6 | 1e-08 | 169.071 | 2.777e-17 | 1.503e-06 | 5.92 | 6.383e-10 |

All nine measured J/energy errors, evaluated coupling bounds and forward L2
bounds are exactly unchanged from the compact-storage measurement. J bounds
are 5.8-14.3 times tighter and energy bounds 1.28-4.06 times tighter. At coupling
tolerance 1e-8, J bounds are 5.82e-7 to 1.51e-6 and energy bounds 1.69e-10 to
6.39e-10 hartree. These remain much larger than the observed finite-reference
errors. They are not an automatic whole-method tolerance.

Strict order-two/four/six repeated RI-J times are 20.474/45.648/169.071 ms,
versus 19.415/44.220/166.377 ms previously. Exact RI-J takes 19.333 ms. These
small timing increases include the new diagnostic work; they do not establish
a general performance change. Peak RSS remains about 389 MB. Both existing
action and accuracy benchmark suites pass; no extra coupling sweep or dense
auxiliary-by-pair object is introduced. Plans explicitly admit the new vectors,
Gamma mates, exports and positive propagation work.

Complete shifted Be RHF still converges in nine iterations in every row:

| Order | Repeated complete job ms | J bound | Energy bound (hartree) | Relaxed energy difference (hartree) |
|---:|---:|---:|---:|---:|
| Exact finite RI | 98.299 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 2 | 145.424 | 6.165e-06 | 1.608e-06 | 1.421e-14 |
| 4 | 157.030 | 7.986e-06 | 2.046e-06 | 1.066e-14 |
| 6 | 210.112 | 1.195e-05 | 3.043e-06 | 1.066e-14 |

Complete-job J bounds improve by about four times, but strict-case energy
bounds improve by only about one percent. They remain 1.61e-6 to 3.05e-6
hartree; observed relaxed differences are near 1e-14 hartree and are not
certified by a fixed-density bound. Complete approximate RHF remains slower
than exact finite RI. Better source-to-result error allocation and automatic
profile selection remain necessary.

Acceptance includes 383 scientific checks, of which 13 are new, and eleven
documentation checks. The new cases cover five producer routes, arbitrary
complex Gamma/non-Gamma inputs, zero and 1e180-scaled inputs, owned exports,
and adversarial forward errors at retained ranks zero/one/six. The existing
memory census was updated to include the new linear vectors while retaining
its tenfold separation from full-tensor storage. No numerical gate was relaxed.

Local vq passes ten public HF/KS states and sixteen raw cases, including tight
order-six RHF/maps, signed angular shells and physical erfc. All 1,878 source
pins, 90 managed outputs and 26 result archives verify. A NumPy-only
assessment independently checks every exported component and column bound,
reconstructs retained-factor error propagation and checks fixed-density J and
energy against exact finite references. Across all 26 predecessor cases,
364 arrays are bitwise unchanged, including couplings, metric, densities,
Focks, energies and saved spin state; coupling/factor identities also match.
Only the propagated J/energy bounds change. Fock, density and energy agreement
with independent approximate-source SCF references remains within 4.83e-15,
1.52e-9 and 1.14e-13 hartree respectively.

There is no matched ORCA/CRYSTAL reference for these custom finite domains.
The existing independent Gaussian/Boys/Fourier, libint erfc and NumPy/SciPy
references remain applicable; KS shares libxc and the approximate partition
is not independently reimplemented. Basis/domain errors, arithmetic and SCF
relaxation are outside this omission ledger. Unified Fock, exact K/XC,
BIPOLE/CRYSTAL conventions and source-qualified symmetry keep their existing
owners and acceptance gates. Automatic accuracy and the remaining approved
MANTA 0.18.x matrix remain unfinished.
