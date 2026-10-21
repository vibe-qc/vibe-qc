# MANTA mixed exact/RI periodic HF

This increment supplies both RI-J/exact-K and exact-J/RI-K to the existing
bounded restricted and unrestricted numerical leaves. Explicit sources,
cutoff factories and public jobs jointly admit both components. The qualified
subset is neutral all-electron 3D HF on uniform full regular meshes with
integer occupations and retained virtual space. Fitted KS, physical range
fitting and the remaining capability matrix stay open.

```sh
.venv/bin/python benchmarks/manta_periodic_mixed_hf.py --repeats 5
```

Eight isolated workers each use one numerical thread and verify a fresh
native build from their own checkout. Every worker measures a cold public
job, five repeated complete public jobs and five native cutoff/SCF calls.
Each call reconstructs its fixed fitting source; there is no cross-run cache.
Temporary standard outputs are removed. Numerical references and QVF
validation/readback occur outside the public timings.

Native SHA256: `4630216af266bdcfd7a757e31d851bc9d6146e15beeb113a669d18e1d9d07286`.

Scientific output adapter SHA256: `79130f3de50f3af226ef65876b83f08b904a2b6b840ac39cd80de8fd8d6f9d19`.


## Finite source and independent reference

The skew cell has lattice rows `(6, .2, .1)`, `(0, 6.2, .15)`, `(0, 0, 6.4)`
in bohr and one nucleus at `(.1, -.15, .2)`. RHF uses Be with four electrons;
UHF uses Li with alpha/beta populations 2/1. Both have three normalized s AOs
with exponents `(2.4, .7, .2)` and three s auxiliaries `(0.4, 1.4, 4.8)`.
The full 2x1x1 mesh is half shifted along its first reciprocal direction.
Both strict-zero-mode and separately admitted BvK-ewald exchange are tested.
AO pair radius is 3 bohr, auxiliary radius 8 bohr, numerical fitting omega
0.65, nuclear alpha 0.6, nuclear image radius 4 bohr and reciprocal kinetic
cutoff 6 hartree. These define this experiment, not an accuracy profile.

The shared Gamma list contains 168 primed G vectors. Exact K uses
`G+k_target-k_source` with 168 vectors in each diagonal and 170 in each
off-diagonal segment, for 676 total. RI K uses the opposite transfer
convention and empty diagonals, for 340 pair vectors plus the Gamma source.
RI-J/exact-K retains one rank-3 metric; exact-J/RI-K retains three. Direct
and fitted finite supports are kept separate, with no parity assumption.

Independent libint raw metric/three-center blocks, NumPy positive fitting
factors, Gaussian Fourier quadrature for direct J/K and SciPy SCF occupations
supply the reference. Separate one-electron/nuclear and probe-charge Ewald
witnesses check those components. Every composed Fock is checked at the
actual evaluated native density, while the density and total energy are
also compared with the independently converged solution.

| Quantity | Maximum absolute difference |
| --- | ---: |
| Overlap | 3.330669e-16 |
| HCORE | 3.552714e-15 |
| Same-density Fock | 5.140333e-14 |
| Converged density | 2.777001e-10 |
| Energy components | 1.019185e-13 |
| Converged total energy, hartree | 1.119105e-13 |
| Native versus public D/F/H/S | 0, bitwise |
| QVF versus evaluated spin density | 0, bitwise |

Every worker emits nine standard artifacts and validates its QVF. RHF QVF
contains equal halves of the evaluated total density. The method ledger
keeps nuclear, one-electron, J, uncorrected K and exchange-zero-mode terms
separate. The correction is applied once, at the same density as both sources.

## Complete native and public costs

In the case names, J or K denotes the fitted component. The complementary
component is direct exact reciprocal. All times are milliseconds.

| Case | Native median [min, max] | Cold public | Repeated public median [min, max] | Iterations | Total energy, hartree |
| --- | ---: | ---: | ---: | ---: | ---: |
| rhf-J-strict | 28.964 [27.936, 29.308] | 304.258 | 148.340 [142.521, 157.609] | 8 | -10.834409112561 |
| rhf-J-BvK | 47.761 [46.930, 49.499] | 336.942 | 171.752 [165.967, 180.849] | 14 | -11.439576155056 |
| rhf-K-strict | 74.850 [71.084, 77.453] | 352.584 | 190.416 [188.709, 197.091] | 9 | -11.432304368038 |
| rhf-K-BvK | 97.307 [96.496, 99.753] | 362.780 | 219.398 [214.392, 223.777] | 13 | -12.037471410533 |
| uhf-J-strict | 40.396 [39.569, 42.824] | 316.468 | 160.780 [157.401, 161.257] | 12 | -5.809818644952 |
| uhf-J-BvK | 53.926 [53.205, 56.038] | 322.556 | 177.076 [172.477, 186.927] | 15 | -6.263693926823 |
| uhf-K-strict | 97.659 [96.055, 98.027] | 368.474 | 217.315 [212.945, 225.815] | 13 | -6.142622912968 |
| uhf-K-BvK | 109.810 [108.435, 112.830] | 375.122 | 235.823 [229.908, 236.061] | 15 | -6.596498194840 |

Different component choices define different finite Hamiltonians and can
require different iteration counts. These timings establish no speedup or
large-mesh crossover. They also do not compare the converged-cutoff accuracy
of the direct and fitted approximations.

## Admission and work

| Case | Generated bound, bytes | Executed bound, bytes | Public bound, bytes | Retained fitting, bytes | Actual shell calls | Terminal direct shell calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rhf-J-strict | 93,820,702 | 93,577,838 | 96,444,334 | 109,488 | 18,864 | 396 |
| rhf-J-BvK | 94,038,878 | 93,796,014 | 96,662,510 | 109,488 | 31,284 | 396 |
| rhf-K-strict | 94,090,478 | 93,794,126 | 96,685,198 | 328,656 | 50,466 | 198 |
| rhf-K-BvK | 94,308,654 | 94,012,302 | 96,903,374 | 328,656 | 70,482 | 198 |
| uhf-J-strict | 93,835,904 | 93,593,040 | 96,496,336 | 109,464 | 27,144 | 396 |
| uhf-J-BvK | 94,054,080 | 93,811,216 | 96,714,512 | 109,464 | 33,354 | 396 |
| uhf-K-strict | 94,105,632 | 93,809,280 | 96,737,152 | 328,584 | 70,482 | 198 |
| uhf-K-BvK | 94,323,808 | 94,027,456 | 96,955,328 | 328,584 | 80,490 | 198 |

Public aggregate admission ranges from 91.976 to 92.464 MiB.


The joint two-electron reservation is about 40.7 to 41.0 MiB, including the
retained fitting source, streamed response, direct producer and composition
scratch. `plan.fock` alone describes the fitting subplan. The enclosing plan
also counts S/T/V, solver state, scalar history, domain construction, output
and every possible solver build plus terminal confirmation. Generated domain
boxes overestimate retained vectors and source workspace. These are modeled
conservative bounds, not measured RSS. Actual shell counts include fitting
metric setup once and both components at every evaluated/confirmation density;
the terminal direct column is one direct build, not aggregate work. For these
single-primitive fixtures actual primitive counts equal shell calls.

## Additional scientific and integration acceptance

The 152 cases in `tests/test_manta_periodic_mixed_hf.py` cover He/Be restricted
and H/Li unrestricted states, Gamma/unshifted/shifted meshes, both mixtures
and exchange conventions, every early capped recurrence, truncated/zero-rank
fits and all five exact/one-short native/domain/public budget boundaries.
Multicenter AO and auxiliary image shifts check complex physical Focks and
nuclear-frame QVF transport. Equivalent explicit/generated sources retain
bitwise equal D/F and source identities. Omitted public exchange keeps its
exact default; it never silently enables RI K.

Thirty-two saved prechange full-RI states preserve matrices, orbital spectra,
energies, original source/projector identities and existing admission counts
bitwise. New component receipts and aggregate two-electron fields are additive.

Owned exports cross the landed FockBuild API with the complete method energy,
independent source/frame/density identities and physical operator action.
No extra dense Fock snapshot is allocated inside the bounded solver. Shared
symmetry transport and BIPOLE source/terminal-density contracts coexist with
this increment. The mixed GDF/reciprocal source does not replace BIPOLE AC/BD
supports, establish CRYSTAL parity or authorize symmetry reduction.
