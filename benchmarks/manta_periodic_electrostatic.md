# Matched periodic electronic and nuclear electrostatics

The native `PeriodicElectrostaticSource` uses one finite zero-average Ewald
kernel for electron-electron, electron-point and point-point energies. It
requires native infinite-domain and full AO-image electronic bounds, and
returns J+V with a total electrostatic energy/error ledger. This is a
component calculation; production HF/KS selection and RI-BUPO remain open.

## Reproduce

```console
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/manta_periodic_electrostatic.py --repeats 5
python -m pytest tests/test_manta_periodic_electrostatic.py -q
```

Three isolated single-thread workers, five repeats each, Python 3.14.8,
NumPy 2.5.3, Darwin arm64. Native SHA256:
`c050c8657b20afcd0e8d4663cab91b0ceeab918e919f10750e5d33c059e53880`.
Other task-owned tests/builds were stopped during timing. Small timing
fluctuations are not evidence of a speedup.

| Source | AO / canonical pairs / points | Nuclear setup (ms) | Combined build (ms) | Electronic build (ms) | Model (MB) | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Sparse signed s | 2 / 7 / 2 | 5.607 | 43.917 | 44.022 | 135.39 | 168.14 |
| Sparse signed s/p | 4 / 7 / 2 | 9.243 | 98.518 | 102.307 | 135.51 | 166.84 |
| Complete tight neutral s | 1 / 14 / 1 | 15.713 | 116.470 | 114.825 | 137.50 | 167.52 |

Setup includes the fixed nuclear potential, overlap, point energy and bounds.
Build includes charge admission, the original BUPO calculation, component
traces and error aggregation. Underlying electronic values and energies are
bitwise identical to calling the electronic source alone. Model bytes are
explicit admission estimates, not process RSS or a measured allocator peak.

The sparse fixtures use signed s or s/p contractions, a skew cell of roughly
4 bohr, origin and plus/minus x AO images, eta 0.45, real cube 2 and reciprocal
cube 3. Point charges are +1.25 and -0.3, and densities are general Hermitian
matrices, not SCF states. Uniform background is selected explicitly. Their
energy bounds, 2.07e6 and 5.35e6 hartree, are intentionally loose and do not
meet a production accuracy target.

The tight fixture is one s primitive of exponent 4 in a cubic 6-bohr cell,
AO image cube 1, eta 0.4, real cube 4 and reciprocal cube 8. Its electron
charge is normalized to one against the native overlap and its point charge
is +1. Neutral admission includes omitted-image charge uncertainty
1.77948e-18. Measured total energy is -2.063159076115948 hartree, with energy
bound 9.94722e-15 hartree and J+V bound 4.15305e-16. These analytical bounds
exclude roundoff; numerical comparisons use separately declared allowances.

## References and scope

Independent finite references integrate Cartesian Gaussian polynomials using
Laplace quadrature and characteristic functions. The ordered point Ewald
reference explicitly resolves real, reciprocal, self and gauge terms. Tight
full-source references use all ordered AO images and the bare nonzero-G
Coulomb series for electron-electron and electron-point terms, plus a
separately converged point Ewald energy. They do not call native integrals.

Tests include signed s/p/d contractions, Cartesian/pure representations,
cubic/skew cells, nonzero k points, zero weights, two physical eta values,
independent reference refinement, charged background, neutral admission,
missing source/target images, density derivatives, zero points, point
permutations, immutable exports and all resource caps. Prior electronic
BUPO/domain/image validation is reused; composition does not change that
finite electronic map or its fixed partition.

There is no suitable matched ORCA/CRYSTAL arbitrary-density electrostatic
component profile. These Gaussian/Ewald limits provide numerical acceptance
without an external SCF parity claim. Roundoff, exchange, fitted sources,
production accuracy profiles, geometry derivatives and lower-dimensional
boundaries remain separate requirements. The shared Fock/UHF, physical
BIPOLE domains and qualified symmetry interfaces are unchanged.

## Queued scientific acceptance

Source `06469f2df61dde040092bf4b4279572e8a5da0b9` was exercised by local
vq job `305a15fd692f`: 18 complete calculations, 11.419 seconds, one CPU,
512 MiB memory request and 600-second limit. The native digest is recorded
above. All 242 implementation/reference file digests and the runtime agree
before and after the job; all 18 fetched numerical artifacts match their
original workspace hashes. A separate NumPy-only assessment recomputed
component traces, bare-series energies and finite differences from those
artifacts. No remote fleet access was used.

The cases include six finite signed s/p/d/cell combinations, a mixed
accepted-far BUPO partition, two tight neutral eta values, a charged
background, a small-image source, a zero-weight derivative base and four
perturbations, an omitted central target, and an empty nuclear source.
Both neutral-policy refusal cases were also checked. The finite mixed case
reuses qualified electronic BUPO evidence and independently resolves its
matched nuclear components; it does not claim full-source SCF parity.

| Check | Largest measured discrepancy |
|---|---:|
| Finite nuclear matrix vs independent integrals | 1.990e-13 |
| Finite attraction energy component | 1.564e-13 hartree |
| Finite point Ewald component | 1.333e-15 hartree |
| Native energy vs recomputed weighted trace | 5.330e-15 hartree |
| Tight full-source energy vs recomputed bare reference | 1.226e-13 hartree |
| Tight full-source J+V vs bare reference | 2.554e-15 |
| Reported / recomputed density derivative | 5.580e-11 / 2.915e-11 hartree |

Numerical allowances are 3e-11 for component, matrix and energy comparisons
and 2e-8 for finite differences, separate from the analytical bounds.
Reported finite-component energies and the independently recontracted arrays
agree within those allowances. Re-summing a streamed bare reference in a
different order contributes up to 1.08e-13 hartree of the total discrepancy.

The equal-population tight neutral calculations give total energy bounds
1.276e-16 at eta 0.4 and 6.925e-13 hartree at eta 0.7; both energy and matrix
bounds meet 1e-8. The benchmark above instead uses a general Hermitian mesh
density rescaled to one electron, so its charge/bound ledger differs. The
small-image source has energy bound 0.557 hartree and fails neutral charge
admission despite retained charge normalization. Its background calculation
remains within its explicitly loose bounds. The omitted-target example has
attraction energy error 0.6404 hartree, matrix error 0.4904 and point-domain
energy error 5.898e-7 hartree, each within its reported bound.

There are 21 new focused checks plus 88 Gaussian-domain/electronic-image
regressions, all passing. One fixture's arbitrary lower bound on a visible
point-domain error was corrected from 1e-6 to 100 times the comparison
allowance; its measured 5.898e-7 error is well resolved. No upper acceptance
tolerance changed. Manifest, changelog, citation routing, Sphinx and privacy
checks pass. No full CI or release qualification is claimed.

The source landed as `8f8743104a71c708779e31aa01c2cc16a1044286` after rebasing
onto the concurrent BIPOLE density-image screening update. All 242 pinned
MANTA implementation/reference files and native bytes remain identical to
the calculation above. The incoming screening code is a separate path and
is not claimed compiled or revalidated by this runtime. Integrated manifest,
changelog and citation checks pass; qualifying numerical evidence is reused.
