# Generated fitted range-separated KS benchmark

This finite-source acceptance covers independently selected RI-J/exact-K,
exact-J/RI-K and RI-J/RI-K through generated cutoff domains and atom-centred
quadrature. HSE06, CAM-B3LYP and a signed mixed-range functional exercise
restricted and unrestricted states. Public range-fitting jobs, automatic
accuracy profiles, external parity and symmetry reduction remain unqualified.

Reproduce after a fresh native build:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ri_range_ks_domains.py
```

Eighteen separate one-thread workers each measure a cold native solve and five
repeated native plus five repeated selected solves. Every solve regenerates
quadrature, domains, all fitting/projector owners and SCF. The selected call
includes scientific provenance construction, without active file output or QVF.
No numerical test work from this task ran alongside the benchmark. Wall times
are illustrative; different finite sources and iteration counts do not define
a speedup comparison. The JSON records native/adapter hashes and source,
domain and grid identities.

The cell has lattice rows `(6, .2, .1), (0, 6.2, .15), (0, 0, 6.4)` bohr with
a shifted `(2, 1, 1)` mesh. Be RKS and Li UKS use three s orbital and three s
auxiliary functions. The mixed functional is the one defined in
`tests/test_manta_periodic_range_separated_ks.py`: 0.4 HSE06 plus 0.6 CAM-B3LYP,
with canonical physical K terms 0.39 Coulomb, 0.1 erfc(0.11) and
-0.276 erfc(0.33). Numerical fitting omega is 0.65, distinct from physical ranges.
The pair radius is 3 bohr, auxiliary radius 8 bohr, reciprocal kinetic cutoff
6 hartree and XC image radius 11 bohr. The grid has 12 radial points, Lebedev
order 17, partition radius 8 bohr and 1,320 points. These are bounded witnesses,
not recommended accuracy settings.

There is one orbital translation, 168 nonzero Gamma vectors and 27 XC images.
Exact K has 676 target-minus-source vectors including primed diagonal segments;
RI K has 340 ket-minus-bra off-diagonal vectors and empty diagonal segments.
The method resolves physical kernels and keeps their projectors separate.
The fitting source owns finite erfc Gamma zero, while the method owns the
long-range BvK correction. HSE06 has no singular correction. The cases retain
one to four fitting sources, one to ten metric channels, all with rank three.

The independent reference reconstructs raw libint/Gaussian Fourier metrics and
couplings, NumPy projectors/contractions, Gaussian Bloch/libxc grid values,
nuclear energy and a SciPy fixed point. It does not use MANTA's Fock contraction
or SCF iteration. D, F, S, H, all functional components and grid populations
are compared at the evaluated state; converged energies are also compared
against independently converged densities.

| J/K profile | Spin | Functional | Energy (hartree) | Iterations | Native median (s) | Selected median (s) | Admission (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| J | RKS | hse06 | -11.846566509415 | 9 | 0.1003 | 0.0989 | 100.176 |
| J | RKS | cam-b3lyp | -11.650232709431 | 12 | 0.1359 | 0.1267 | 116.634 |
| J | RKS | mixed | -11.728739754906 | 11 | 0.1316 | 0.1302 | 133.886 |
| J | UKS | hse06 | -6.517614810076 | 11 | 0.1386 | 0.1352 | 100.180 |
| J | UKS | cam-b3lyp | -6.375853128857 | 12 | 0.1488 | 0.1518 | 116.640 |
| J | UKS | mixed | -6.432522847586 | 12 | 0.1918 | 0.1717 | 133.893 |
| K | RKS | hse06 | -12.298467361624 | 9 | 0.1404 | 0.1391 | 100.405 |
| K | RKS | cam-b3lyp | -12.088095118653 | 12 | 0.2639 | 0.2605 | 125.635 |
| K | RKS | mixed | -12.172214731484 | 11 | 0.3416 | 0.3294 | 151.658 |
| K | UKS | hse06 | -6.774174621765 | 12 | 0.1929 | 0.2082 | 100.407 |
| K | UKS | cam-b3lyp | -6.622148217868 | 12 | 0.2819 | 0.2820 | 125.637 |
| K | UKS | mixed | -6.682892387255 | 12 | 0.3872 | 0.3851 | 151.660 |
| JK | RKS | hse06 | -11.894496351480 | 9 | 0.1591 | 0.1495 | 108.855 |
| JK | RKS | cam-b3lyp | -11.686584391711 | 12 | 0.3251 | 0.3112 | 134.085 |
| JK | RKS | mixed | -11.769712824577 | 11 | 0.3697 | 0.3573 | 160.108 |
| JK | UKS | hse06 | -6.543175697164 | 12 | 0.2122 | 0.2142 | 108.857 |
| JK | UKS | cam-b3lyp | -6.395180774276 | 12 | 0.3011 | 0.3033 | 134.087 |
| JK | UKS | mixed | -6.454323089482 | 12 | 0.4069 | 0.4077 | 160.110 |

`J` means RI-J/exact-K, `K` exact-J/RI-K, and `JK` both fitted.
Every state converges and passes physical terminal confirmation. Generated,
explicit and selected D/F/H/S, coefficients, orbital energies and total energy
agree bitwise.

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
1.776357e-15 hartree in energy. The coarse grid's electron-population deviation
is 0.143 to 0.249 electrons. Independent agreement on that finite grid does
not establish quadrature or cutoff convergence. No matched ORCA/CRYSTAL source
is available for these controls; external parity is untested.

Count-only joint admission spans 100.176 to 160.110 MiB, while actual executed
native plans use 104,684,101 to 167,144,946 bytes. Selected plans additionally
reserve their output evidence, reaching 104,778,309 to 167,386,610 bytes.
Retained fitting/source owners plus the functional solver's energy scalar use
109,472 to 1,095,464 bytes. Generated domains use 198,080 to 263,936 bytes;
the retained quadrature allocation is 133,344 bytes. Serialized evidence uses
9,697 to 21,365 bytes. Plans are conservative allocation/work contracts,
not process RSS promises; measured process peaks are 176.0 to 182.5 MiB.

The focused test module adds Gamma and empty-beta states, skew-cell sphere
checks, every exact/one-short aggregate resource boundary, invalid-source
refusal before payload generation, selected output ownership, independent
geometry/projector identities, shared Fock snapshots, rectangular overlap
frames and complex multicentre AO/auxiliary image transport. These witnesses
preserve the common operator and symmetry interfaces without authorizing
source skipping or claiming BIPOLE AC/BD parity.
