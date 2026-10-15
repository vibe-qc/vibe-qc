# Explicit fitted periodic KS acceptance and cost

Measured on 2026-10-10 with FP64 and one thread. Twelve isolated workers
each perform a cold native solve, five complete native solves and five
selected scientific solves. Each call rebuilds the fitting source and its
projectors; projectors are fixed within SCF and terminal confirmation.
The selected wrapper admits and constructs output provenance. These timings
do not include full public jobs, generated cutoffs/grids or artifact writing.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python benchmarks/manta_periodic_ri_ks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python benchmarks/manta_periodic_ri_ks.py --conditioning
```

The explicit skew cell, shifted two-point mesh, orbital and auxiliary bases,
finite reciprocal cube and image domains come from the regression fixture.
The timing grid is a 10-by-10-by-10 tensor Gauss-Legendre cell quadrature.
RKS uses Be and UKS uses Li with PBE for pure KS and PBE0 for globals.
Empty-beta H uses BLYP for pure KS and 0.8 B88 + LYP + 0.2 exact exchange
for globals. Every global timing uses the method-owned BvK correction.
J denotes RI-J/exact-K; K denotes exact-J/RI-K; JK denotes both fitted.
These deliberately finite sources are not common Hamiltonians or calibrated
accuracy settings. Differences between their energies do not measure a bug
or establish any speedup. Strict-zero-mode and Gamma cases have separate tests.

## Complete solve times

All times are seconds. Selected results match native D/F/H/S, coefficients
and orbital energies bit for bit. The last confirmation build is not an
additional SCF iteration.

| Profile | Cold native | Native median | Native min to max | Selected median | Iterations / builds |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure-rks | 0.062207 | 0.059046 | 0.058600 to 0.061161 | 0.061246 | 9 / 10 |
| pure-uks | 0.094600 | 0.092815 | 0.089452 to 0.096087 | 0.094234 | 13 / 14 |
| pure-empty | 0.024072 | 0.023675 | 0.022330 to 0.024232 | 0.024108 | 5 / 6 |
| J-rks | 0.058162 | 0.055122 | 0.053808 to 0.055197 | 0.055189 | 8 / 9 |
| J-uks | 0.076924 | 0.074585 | 0.071848 to 0.077820 | 0.073421 | 10 / 11 |
| J-empty | 0.044268 | 0.043318 | 0.042817 to 0.045415 | 0.045657 | 9 / 10 |
| K-rks | 0.096490 | 0.094032 | 0.091050 to 0.101039 | 0.093165 | 9 / 10 |
| K-uks | 0.110565 | 0.108868 | 0.108632 to 0.109571 | 0.108288 | 10 / 11 |
| K-empty | 0.064556 | 0.062111 | 0.060666 to 0.064733 | 0.062166 | 9 / 10 |
| JK-rks | 0.080326 | 0.080343 | 0.078715 to 0.080712 | 0.084474 | 8 / 9 |
| JK-uks | 0.106526 | 0.104752 | 0.103171 to 0.107076 | 0.107626 | 10 / 11 |
| JK-empty | 0.060173 | 0.059285 | 0.058605 to 0.060948 | 0.060611 | 9 / 10 |

## Independent numerical acceptance

Raw libint fitting metrics/couplings, independent NumPy spectral projection,
Gaussian Fourier quadrature for direct terms and Gaussian Bloch values and
gradients with libxc values define an independent functional. A SciPy
generalized eigensolve/SCF iteration supplies the density reference. The
reference does not call MANTA contraction or SCF routines. Nuclear energy
is checked through the existing independent Ewald reference.

Every same-density energy component and full spin/k Fock is compared;
converged energies use independently converged densities. Maximum observed
errors across the twelve timing profiles are:

| Quantity | Maximum absolute difference |
| --- | ---: |
| S | 3.330669e-16 |
| H | 2.664535e-15 |
| D | 1.644477e-10 |
| F | 1.221245e-14 |
| components | 2.550737e-14 |
| energy | 1.953993e-14 |
| grid_populations | 3.552714e-15 |

Energies and Fock entries are in atomic units. The benchmark gates are
1e-8 for D and 1e-9 for other quantities. Maximum native final Fock and energy
confirmation errors are 2.23e-16 and 1.78e-15 respectively. Same-grid correctness does not
certify an infinite lattice, complete fitting basis or converged quadrature.
The maximum integrated grid-population error against the expected electron
count is 2.672815e-01, separately from reference contraction agreement.

## Joint resource model and executed work

Models are conservative admission bounds, not process RSS certificates.
Native bounds include retained fit owners, complementary direct sources, XC,
solver, history, temporary embedding and maximum allowed repeated builds.
Selected bounds additionally reserve provenance. RI-J has only the Gamma
owner; RI-K retains all three Gamma/ordered-transfer owners on this mesh.

| Profile | Metric owners | Native peak bytes | Selected peak bytes | Retained fit bytes | Shell calls | Primitive pairs | Primitive points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure-rks | 1 | 88209839 | 88287663 | 105264 | 26658 | 13698 | 810000 |
| pure-uks | 1 | 88212086 | 88289910 | 105240 | 37242 | 19098 | 1134000 |
| pure-empty | 1 | 88167034 | 88244858 | 106600 | 10180 | 4996 | 324000 |
| J-rks | 1 | 104341825 | 104436033 | 105264 | 26604 | 14940 | 729000 |
| J-uks | 1 | 104345758 | 104439966 | 105240 | 32472 | 18216 | 891000 |
| J-empty | 1 | 105337498 | 105431706 | 106600 | 18116 | 9476 | 540000 |
| K-rks | 3 | 104606401 | 104725185 | 315792 | 53112 | 40152 | 810000 |
| K-uks | 3 | 104608590 | 104727374 | 315720 | 58386 | 44130 | 891000 |
| K-empty | 3 | 105600410 | 105719194 | 319800 | 32484 | 23844 | 540000 |
| JK-rks | 3 | 87702675 | 87821459 | 315792 | 46542 | 34878 | 729000 |
| JK-uks | 3 | 87704874 | 87823658 | 315720 | 56802 | 42546 | 891000 |
| JK-empty | 3 | 88710902 | 88829686 | 319800 | 31844 | 23204 | 540000 |

The independent reference and Python/runtime overhead contribute to process
peak RSS, which ranged from 172.7 to 175.8 MiB.
The source counts and actual work vary with orbital/auxiliary size and SCF
iterations. Per-transfer fitting workspaces remain conservative; these small
fixtures do not establish an optimized scaling crossover.

## Exactly empty-spin conditioning

The separate order-7 H witnesses retain pure PBE and global PBE0/B3LYP.
The minority column is the maximum Fock-entry difference against independent
Gaussian/libxc contraction. Sensitivity is the largest minority-XC change
after multiplying the majority grid density by 1 plus or minus one machine
epsilon at fixed gradients. This measures the existing libxc FP64 endpoint,
not a uniform error bound or a new regularization. Energy and occupied-spin
gates remain strict. The pure-PBE tests also isolate the full observed
difference in the XC-only source, leaving fitted J agreement at roundoff.

| Source / functional | k points | Minority difference | One-epsilon sensitivity | Occupied difference | Largest component difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure / pbe | 1 | 3.575838e-06 | 7.341882e-06 | 5.551115e-16 | 1.665335e-16 |
| pure / pbe | 2 | 1.535976e-06 | 1.797744e-06 | 1.332268e-15 | 3.330669e-16 |
| JK / pbe0 | 1 | 2.929081e-06 | 1.583474e-06 | 3.194667e-14 | 1.644518e-14 |
| JK / pbe0 | 2 | 1.907746e-06 | 3.982466e-06 | 1.176836e-14 | 6.356027e-15 |
| JK / b3lyp | 1 | 3.489548e-11 | 5.856415e-11 | 2.539635e-14 | 1.315614e-14 |
| JK / b3lyp | 2 | 9.037548e-11 | 6.142564e-11 | 9.603429e-15 | 5.093148e-15 |

The B88/LYP empty-spin timing profiles retain full matrix acceptance.
PBE endpoint conditioning is not inferred from energy or density agreement.

## Regression and integration scope

The fitted KS suite checks converged independent SCF, every capped epoch,
fourth-order complex density-energy derivatives, strict/BvK scaling, frozen
full/truncated/zero projectors, exact and one-short native/selected caps,
functional/source refusals, real output serialization and citations.
Multicenter independent AO and auxiliary image shifts cross shared
BlockSpaceAction transport. Redundant AO shells test rectangular S/X frames
and complete physical Fock exports through the landed FockBuild API.
The generic object receives the method energy and evaluated density; an HF
half trace or next orbital projector is not substituted.

Fifty-six saved direct pure/global/range KS states retain numerical arrays,
spectra, energies, source identities and original resource counts bitwise.
Evaluated-density and Fock snapshot hashes intentionally adopt version 2
semantics separating source selection from the density epoch.

Fitted cutoff/grid factories and public jobs remain required work. These
finite sources preserve the independent BIPOLE/CRYSTAL and source-symmetry
qualification boundaries and introduce no reduction or parity claim.

Native binary SHA256: `5063b87f25d9fd9f9cd98e1b55cfc56aa25da54c245838b7255b380dbedc91d2`.
Scientific adapter SHA256: `bdd26c0f1a7735b75ec36c3921bd6f7fcc8511edebec5a1c7e564fabd9092bc3`.
