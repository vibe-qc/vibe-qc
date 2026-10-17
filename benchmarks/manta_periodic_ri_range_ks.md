# Bounded fitted range-separated KS

This benchmark runs 18 isolated workers, each with one thread and five complete
native solves plus five selected Python solves. All use neutral all-electron
Be RKS or Li UKS, three orbital and three auxiliary s functions, a shifted
`(2, 1, 1)` mesh, 343 per-cell quadrature points, numerical fitting omega 0.65,
3 bohr AO-pair and 8 bohr auxiliary cutoffs. Physical ranges and coefficients
come from HSE06, CAM-B3LYP or the explicit 0.4 HSE06 + 0.6 CAM-B3LYP mixture.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ri_range_ks.py
```

`J` means RI-J/exact-K, `K` exact-J/RI-K, and `JK` RI-J/RI-K. Every fitted
kernel retains a separate metric/projector. The method uses the canonical
Coulomb-plus-erfc decomposition, keeps bare-Coulomb J separate and adds a
singular correction only for the resolved long-range coefficient. HSE06
therefore has no Madelung correction. These different exact/RI finite sources
need not agree with each other; their timings are not a speedup claim.

| Fit | Spin | Functional | Native median (ms) | Selected median (ms) | Evaluated iterations | Retained metric channels | Total energy (hartree/cell) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| J | RKS | hse06 | 50.724 | 50.839 | 10 | 1 | -11.70227769973592 |
| J | RKS | cam-b3lyp | 65.445 | 64.468 | 11 | 1 | -11.52271324289145 |
| J | RKS | mixed | 65.760 | 67.348 | 10 | 1 | -11.59442689606535 |
| J | UKS | hse06 | 58.032 | 57.545 | 11 | 1 | -6.41091836798650 |
| J | UKS | cam-b3lyp | 57.887 | 60.325 | 12 | 1 | -6.27090561626603 |
| J | UKS | mixed | 87.515 | 94.884 | 12 | 1 | -6.32682946782104 |
| K | RKS | hse06 | 97.373 | 96.846 | 10 | 3 | -12.32374444259574 |
| K | RKS | cam-b3lyp | 194.475 | 176.996 | 11 | 6 | -12.12693898897000 |
| K | RKS | mixed | 230.762 | 217.983 | 10 | 9 | -12.20557846509247 |
| K | UKS | hse06 | 92.367 | 87.874 | 11 | 3 | -6.78116434008634 |
| K | UKS | cam-b3lyp | 182.956 | 176.816 | 11 | 6 | -6.62955267085557 |
| K | UKS | mixed | 254.624 | 261.106 | 11 | 9 | -6.69011929382253 |
| JK | RKS | hse06 | 117.408 | 118.941 | 10 | 4 | -11.77039072372076 |
| JK | RKS | cam-b3lyp | 268.359 | 206.268 | 11 | 7 | -11.57470523683406 |
| JK | RKS | mixed | 278.156 | 283.513 | 10 | 10 | -11.65288289629589 |
| JK | UKS | hse06 | 139.651 | 141.713 | 11 | 4 | -6.44851389641256 |
| JK | UKS | cam-b3lyp | 226.513 | 248.088 | 12 | 7 | -6.29959435895324 |
| JK | UKS | mixed | 292.146 | 289.800 | 12 | 10 | -6.35907992069865 |

Native and selected S/H/D/F, orbitals and coefficients are bitwise equal.
Every converged solve passed an independent terminal energy/Fock rebuild.
Libint raw short-range integrals, Gaussian Fourier sources, NumPy fixed
projectors and contractions, independent libxc quadrature contractions, and
SciPy fixed points provide the reference. Maximum errors are 6.58e-14 hartree
in total energy, 2.17e-10 in density, 3.24e-14 in same-density Fock and
6.64e-14 hartree in energy components.

Modeled native peaks range from 104,147,877 to 166,540,562 bytes. Retained
provider bytes range from 105,248 to 1,052,648, including the functional
solver's energy scalar. Every physical kernel's source, metric setup,
workspace, repeated response and confirmation is admitted. These conservative
bounds are not process RSS; retained projectors are not a global dense B tensor.
The selected route also reserves provenance before copying source arrays.

The coarse grid population differs from metric populations by 0.0115 to
0.0769 electrons. Neither agreement with the finite-source reference nor
native confirmation establishes grid/domain convergence. No matched ORCA or
CRYSTAL reference exists for these explicit finite supports, so external parity
remains untested. Public range-fitting jobs and generated method domains are
still gated.

The focused suite also covers Gamma, other long-range hybrids, empty beta,
signed and cancelled terms, weighted functional derivatives, fixed truncated
or empty projectors, capped evaluated epochs, rectangular overlap frames,
complex multicentre image transport through shared symmetry actions, shared
Fock state, resource boundaries and owned provenance. Sixteen saved prior
fitted pure/global and direct range KS states retain all previous numerical
arrays, plans and identities exactly.

Native core SHA-256:
`d2e73f46d7fd5852e9bc2f491b5c4465dde05396181498067bb07ee1ee86696e`.
Selected Python adapter SHA-256:
`dce8816297885adc3f51bee631c06dda62ad85fbb87eb1aadd494c4a23ce59b4`.
