# Public fitted range-KS job benchmark

This acceptance measures full `run_periodic_job` RKS/UKS with RI-J/exact-K,
exact-J/RI-K and RI-J/RI-K for HSE06, CAM-B3LYP and a signed mixed-range
functional. It extends the [generated-source benchmark](manta_periodic_ri_range_ks_domains.md)
with complete job admission, standard output and QVF. Its cells, basis sets,
shifted two-k mesh, numerical fitting split, physical exchange terms and
finite grid/cutoff controls are identical to that benchmark.

Reproduce after a fresh native build:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ri_range_ks_job.py
```

Eighteen separate one-thread workers each measure a cold public job, five
repeated full public jobs and five native generated solves. Each solve builds
new grid, domains, physical fitting projectors and SCF state. Public times
include file output, provenance, citations and QVF; no native source is cached
between repetitions. No other numerical work from this task ran during the
benchmark. These illustrative wall times compare interface cost for the same
finite source, not speedups between different physical fitting choices.

Every result converges and passes physical terminal confirmation. Native/public
D, F, H, S, coefficients, orbital energies and total energies agree bitwise.
QVF validates and recovers the evaluated density bitwise, with equal spin halves
for RKS and distinct spin channels for UKS. The independent reference uses raw
libint/Gaussian Fourier fitting sources, NumPy factors, Gaussian Bloch/libxc XC,
independent nuclear energy and SciPy fixed points, as documented in the
source benchmark. It does not use the native MANTA Fock/SCF implementation.

| Profile | Energy (hartree) | Iterations | Native median (s) | Public median (s) | Public plan (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| J-rks-hse06 | -11.846566509415 | 9 | 0.0792 | 0.1745 | 102.646 |
| J-rks-cam | -11.650232709431 | 12 | 0.0968 | 0.2164 | 119.089 |
| J-rks-mixed | -11.728739754906 | 11 | 0.1188 | 0.2086 | 136.325 |
| J-uks-hse06 | -6.517614810076 | 11 | 0.1114 | 0.2397 | 102.685 |
| J-uks-cam | -6.375853128857 | 12 | 0.1185 | 0.2237 | 119.130 |
| J-uks-mixed | -6.432522847586 | 12 | 0.1253 | 0.2218 | 136.367 |
| K-rks-hse06 | -12.298467361624 | 9 | 0.1219 | 0.2083 | 102.912 |
| K-rks-cam | -12.088095118653 | 12 | 0.2160 | 0.3296 | 128.082 |
| K-rks-mixed | -12.172214731484 | 11 | 0.2774 | 0.3418 | 154.045 |
| K-uks-hse06 | -6.774174621765 | 12 | 0.1639 | 0.2719 | 102.950 |
| K-uks-cam | -6.622148217868 | 12 | 0.2236 | 0.3350 | 128.119 |
| K-uks-mixed | -6.682892387255 | 12 | 0.3687 | 0.4638 | 154.082 |
| JK-rks-hse06 | -11.894496351480 | 9 | 0.1278 | 0.2523 | 111.345 |
| JK-rks-cam | -11.686584391711 | 12 | 0.2595 | 0.3182 | 136.515 |
| JK-rks-mixed | -11.769712824577 | 11 | 0.3132 | 0.4186 | 162.477 |
| JK-uks-hse06 | -6.543175697164 | 12 | 0.1672 | 0.2752 | 111.382 |
| JK-uks-cam | -6.395180774276 | 12 | 0.2503 | 0.3499 | 136.552 |
| JK-uks-mixed | -6.454323089482 | 12 | 0.3019 | 0.4487 | 162.514 |

`J` means RI-J/exact-K, `K` exact-J/RI-K, and `JK` both fitted. `cam` denotes
CAM-B3LYP; `mixed` is 0.4 HSE06 plus 0.6 CAM-B3LYP. The physical range terms
retain separate projectors and remain distinct from numerical fitting omega.

| Independent quantity | Maximum absolute error |
| --- | ---: |
| Overlap | 3.330669e-16 |
| HCORE | 3.552714e-15 |
| Converged density | 2.150382e-10 |
| Evaluated Fock | 3.397282e-14 |
| Functional components | 6.650236e-14 |
| Converged energy (hartree) | 6.750156e-14 |
| Grid populations | 4.884981e-15 |

Maximum physical confirmation differences are 4.440894e-16 in F and
1.776357e-15 hartree in energy. The 1,320-point grid's maximum population
error is 0.143 to 0.249 electrons; same-grid independent agreement does not
qualify quadrature convergence. No matched ORCA/CRYSTAL finite-source
reference is available, and external parity is untested.

Public plans range from 107,632,517 to 170,408,690 bytes. Separate job/output
reservations are 2,854,208 to 3,022,080 bytes and selected scientific evidence
reservations 94,208 to 241,664 bytes. Each job writes nine files totalling
85,283 to 120,141 bytes; serialized integral provenance uses 10,104 to 24,506
bytes. Measured process peaks are 183.7 to 188.8 MiB. Allocation/work plans
remain conservative contracts rather than process RSS estimates. The JSON
records source/runtime hashes, work bounds and output sizes.

Focused tests also cover Gamma and strict-zero-mode cases, empty beta,
LC-wPBE, capped evaluated states, truncated/empty metrics, every joint budget
boundary, preflight refusal, immutable results, rectangular overlap frames,
shared Fock states and complex AO/auxiliary image transport into QVF's nuclear
frame. Dry runs and synthetic output failure test writer lifecycle. These
checks preserve the shared operator and symmetry seams without enabling
source reduction, BIPOLE AC/BD parity, gradients or automatic accuracy profiles.
