# Public fitted periodic KS benchmark

This acceptance measures complete `run_periodic_job` execution for pure RI-J
and all three fitted global RKS/UKS choices. Each job retains the actual
functional energy and evaluated spin density through standard output and QVF.
Physical range fitting, automatic accuracy profiles, source-qualified symmetry
reduction and BIPOLE/CRYSTAL parity remain separate requirements.

Reproduce with the current checkout's fresh native build:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ri_ks_job.py
```

Twelve isolated workers each measure a cold public job, five repeated public
jobs and five native generated-source solves. Every call rebuilds the grid,
physical domains, fixed fitting sources and SCF. Public calls include complete
admission, evidence, history, citations, output delivery and QVF serialization.
They overwrite their own temporary output outside the checkout. The final QVF
is validated and its saved spin density compared bitwise with the evaluated
native state. Timing was collected without concurrent test work from this
task. Iterations and finite Hamiltonians differ across profiles, so this is
not a speedup comparison.

The scientific fixtures and independent references are those of the
[generated fitted KS benchmark](manta_periodic_ri_ks_domains.md): a skew 3D
cell, shifted `(2, 1, 1)` mesh, Be RKS, Li UKS and empty-beta H; PBE/PBE0,
with BLYP or global B88/LYP for strict empty-spin full-matrix witnesses.
The reciprocal kinetic cutoff is 6 hartree, pair radius 3 bohr, XC image
radius 11 bohr, fitting omega 0.65 and auxiliary radius 8 bohr. The coarse
atom-centred grid has 12 radial points, Lebedev order 17, partition radius
8 bohr and 1,320 points. All global cases use the method-owned BvK correction.
RI-J retains one metric, while RI-K retains three; exact and fitted K retain
their distinct transfer orientations and diagonal supports.

Libint raw fitting, NumPy projectors/contractions, Gaussian Fourier and
Gaussian Bloch/libxc evaluation, plus SciPy SCF independently reconstruct
S, H, F, each energy component and the converged density. The nuclear energy
is independently reconstructed. No MANTA density contraction or SCF iteration
is used by that reference.
These custom finite fitting, reciprocal and quadrature supports have no matched
ORCA or CRYSTAL reference here; external-program parity remains untested.

| Source and spin | Iterations | Native median (s) | Public median (s) | Public modeled peak (MiB) | Output bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure-rks | 9 | 0.0814 | 0.2241 | 87.130 | 76095 |
| pure-uks | 13 | 0.1198 | 0.2682 | 87.167 | 86815 |
| pure-empty | 5 | 0.0309 | 0.1657 | 87.075 | 78777 |
| J-rks | 8 | 0.0797 | 0.2177 | 102.712 | 86187 |
| J-uks | 11 | 0.1095 | 0.2579 | 102.751 | 95236 |
| J-empty | 9 | 0.0615 | 0.2090 | 103.648 | 89659 |
| K-rks | 8 | 0.1173 | 0.2583 | 102.944 | 87889 |
| K-uks | 11 | 0.1679 | 0.3042 | 102.981 | 96792 |
| K-empty | 9 | 0.0840 | 0.2187 | 103.878 | 91348 |
| JK-rks | 8 | 0.1123 | 0.2414 | 86.816 | 86770 |
| JK-uks | 11 | 0.1595 | 0.2914 | 86.853 | 95677 |
| JK-empty | 9 | 0.0846 | 0.2171 | 87.763 | 90187 |

`J` denotes RI-J/exact-K, `K` exact-J/RI-K and `JK` both fitted. Pure cases
have no K. All native and public states converge and confirm. D, F, H, S,
coefficients, orbital energies and total energies agree bitwise between the
interfaces. Saved QVF contains the exact evaluated UKS spin densities or RKS
equal halves. Each job delivers nine files; output size includes all artifacts.

| Independent quantity | Maximum absolute error |
| --- | ---: |
| Overlap | 3.330669e-16 |
| HCORE | 3.552714e-15 |
| Converged density | 2.006539e-10 |
| Same-density Fock | 1.443290e-14 |
| Energy components | 2.550737e-14 |
| Converged total energy | 2.486900e-14 |
| Integrated grid populations | 7.105427e-15 |

Maximum native confirmation differences are
2.220446e-16 in F and
1.776357e-15 hartree in energy.
Coarse-grid population deviation is 0.015 to 0.248 electrons;
same-grid reference agreement does not establish quadrature accuracy.

Public modeled peak storage is 86.816 to 103.878 MiB. The job
reservation is 2774272.000 to 2825472.000 bytes; separate scientific Python
evidence reservation is 77824.000 to 118784.000 bytes. These owners
include maximum-iteration history/output, basis metadata and density conversion,
even if convergence occurs earlier. Count-only native grid/domain admission
before those public reservations is 84.315 to 101.406 MiB.

These conservative ownership models are not RSS. Whole-worker peak RSS is
182.172 to 186.547 MiB, including Python, libraries, owned exported
states, output processing and the independent reference. Serialized evidence
is 8206.000 to 13411.000 bytes and is not its live-memory bound.

`tests/test_manta_periodic_ri_ks_job.py` additionally covers Gamma and shifted
meshes, both singular conventions, LDA/B3LYP descriptor routing, capped and
truncated/empty-projector states, lossless evaluated-density QVF, independent
AO/auxiliary image shifts and nuclear-frame conversion, rectangular S/X
frames, immutable result/evidence exports, citations, all five exact/one-short
complete job caps, dry runs and failed output delivery. Shared FockBuild and
BlockSpaceAction checks preserve method energy and actual D/F without granting
a symmetry certificate. Sixteen saved existing public direct KS/RI HF states
retain their numerical arrays, native plans and identities exactly.

Native core SHA-256: `c9b81d47943d442eb5bb04026f06827b3ff9c60630bf00156bdf0c0b1891e84d`.
Scientific adapter SHA-256: `f0c0407ee49a8c981041ee16f055136b66d1e007ef3ff97b4239a979dec053e5`.
Public job adapter SHA-256: `ceb147af0b675c04686175119eacd8f8c65e7581a1dc0029fc409c0da7d2f010`.
The benchmark checks native checkout ownership and freshness before running.
This is focused finite-source acceptance, not a full numerical lane or release
gate.
