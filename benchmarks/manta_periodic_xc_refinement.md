# Frozen-density periodic XC grid assessment

Measured 2026-10-09 in three fresh CPU workers, each with one cold run and
five repetitions. OpenMP, OpenBLAS and Accelerate use one thread. Timings
include joint admission, three grid generations, all XC contractions and
comparison norms. They are elapsed observations, not a speedup claim.

```sh
.venv/bin/python benchmarks/manta_periodic_xc_refinement.py
```

The fixed source is a cubic 6-bohr cell with one He center, two normalized
s Gaussians (exponents 0.8 and 0.25), the 27 neighboring images, two non-TRIM
k points at plus/minus 0.13 inverse bohr on x, weights 0.4/0.6 and complex
Hermitian densities. The empty-beta fixture sets the BLYP beta density to
zero. These fixed densities are integration witnesses, not physical SCF
solutions. Each assessment uses Becke grids `(radial, Lebedev order, partition
radius in bohr)` of `(16,29,8)`, `(24,35,8)` and `(40,59,12)`, respectively
4832, 10416 and 48080 points. The last grid is the explicit reference.

| Profile | Cold (s) | Median (s) | Range (s) | Driver reservation (bytes) | Modeled peak (bytes) |
|---|---:|---:|---:|---:|---:|
| LDA RKS | 0.196588 | 0.189328 | 0.187803 to 0.189743 | 147940 | 18631637 |
| PBE UKS | 0.258379 | 0.253529 | 0.249278 to 0.255212 | 148644 | 18685205 |
| Empty-beta BLYP | 0.247244 | 0.248403 | 0.245174 to 0.250170 | 148656 | 18685221 |

All three perform 26784 shell calls and 3419712 primitive/grid products.
The peak already includes the driver reservation; it is not measured RSS.
Scalar/kernel bounds are conservative admission counts, approximately
3.92e13/1.25e15 per complete assessment. Every requested grid contributes
to work, while only the reference and one candidate potential coexist.

Independent polynomial Gaussian and libxc point contractions use the same
generated points to check the numerical operation. The maximum errors across
all profiles are 4.441e-16 hartree in XC energy, 1.444e-15 in the returned
reference potential, 4.885e-15 electrons in populations and 3.442e-15 in the
reported difference norms. This is same-grid correctness, not grid accuracy.

| Profile | 64-to-80 tensor energy change (Ha) | 64-to-80 tensor potential change (Ha) | Selected reference energy error vs tensor 80 (Ha) | Selected reference potential max error vs tensor 80 (Ha) |
|---|---:|---:|---:|---:|
| LDA RKS | 6.662e-15 | 2.887e-15 | 1.868e-7 | 1.375e-7 |
| PBE UKS | 4.441e-16 | 4.525e-14 | 9.379e-7 | 5.275e-7 |
| Empty-beta BLYP | 7.289e-12 | 1.395e-11 | 2.252e-7 | 4.376e-7 |

The independent tensor-cell integration uses Gauss-Legendre rules and
analytic Gaussian values/gradients in bounded Python chunks. It uses neither
the native AO evaluator nor the atomic-grid producer. Even this relatively
smooth fixture shows a measurable error in the chosen atomic reference.
Its comparison row is exactly zero by construction. For empty-beta BLYP,
the 16-point radial grid has an energy difference of 1.043e-5 hartree but a
maximum spin/k potential difference of 1.997e-4 hartree. Energy and each
potential need separate checks; no accuracy profile is qualified here.

The native core SHA-256 was
`60ba349fd49120208e5083617db10d2505752026808e57adf71df35758164736`.

## Physical HeH/PBE reference adequacy

```sh
.venv/bin/python benchmarks/manta_periodic_xc_refinement.py --physical-refinement
```

This slower diagnostic freezes the confirmed HeH/PBE UKS density from the
physical KS-grid fixture: three s AOs, the full shifted three-point mesh and
the explicit 11-bohr XC-image sphere. That density was solved on the small
12-radial/order-17 grid; all subsequent evaluations use exactly that same
density, functional and source frame. Four Becke grids vary radial/angular
resolution and partition radius independently. No new SCF relaxation is mixed
into the grid differences.

| Radial points | Lebedev order | Partition radius (bohr) | Total points | Signed energy difference from selected reference (Ha) | Maximum spin/k potential difference (Ha) |
|---|---:|---:|---:|---:|---:|
| 40 | 59 | 8 | 96160 | -2.259429e-6 | 1.828769e-5 |
| 80 | 95 | 8 | 491840 | -2.466189e-7 | 7.295958e-6 |
| 80 | 95 | 16 | 491840 | -1.867900e-7 | 7.296610e-6 |
| 120 | 95 | 20 | 737760 | 0 | 0 |

One complete native assessment took 212.275 seconds with a modeled peak of
115313329 bytes, using the core hash above. This is a single diagnostic run,
not a repeated performance benchmark. Independent tensor-cell references
then use orders 64, 96 and 128 without the native AO or grid producer.

| Tensor refinement | Energy change (Ha) | Maximum complex spin/k potential change (Ha) | Maximum population change (electrons) |
|---|---:|---:|---:|
| 64 to 96 | 6.585473e-8 | 8.252382e-6 | 2.132e-14 |
| 96 to 128 | 1.063786e-7 | 3.758120e-6 | 1.653e-13 |

The selected atomic reference differs from tensor 128 by 6.924901e-8 hartree
in energy and 2.368949e-6 hartree in potential. These are differences between
two references that still need refinement, not reliable continuum errors.
Energy changes are nonmonotonic, and population stability near 1e-13 does not
qualify potentials changing by several microhartree. A stronger independent
integration witness is required before accepting an automatic tight profile.
The following adaptive witness addresses reference convergence for this fixed
fixture; it does not qualify the atomic grid or an automatic profile.

## Independent adaptive reference

```sh
.venv/bin/python benchmarks/manta_periodic_xc_refinement.py --adaptive-reference
```

The optional witness uses SciPy's array-valued
[adaptive cubature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html)
on the fractional cell, with polynomial Gaussian values/gradients and libxc
point evaluation. It integrates energy, both populations and the real and
imaginary parts of every spin/k/AO potential entry at once. The cell Jacobian
is explicit; k weights enter the density, not each k potential. Same-point
tests against native integration and complex-density energy derivatives
check this vector integrand for restricted/unrestricted LDA/PBE and empty-beta
BLYP. No native grid or AO evaluator supplies the HeH s-Gaussian witness.

The fixed HeH/PBE density and finite image source are identical to the
physical probe above. Degree-seven Genz-Malik subdivision and two product
Gauss-Kronrod rules vary both the rule and requested absolute component
tolerance. SciPy error estimates are diagnostics rather than rigorous bounds;
the subdivision cap and actual status are always reported. A capped result
is not relabeled converged. This optional benchmark uses SciPy 1.18.1 with
`integrate.cubature`, adds no product runtime dependency, and takes several
minutes on one CPU thread.

| Rule | Absolute component tolerance | Subdivisions / cap | Actual status | Points evaluated | Largest estimated component error | Elapsed (s) |
|---|---:|---:|---|---:|---:|---:|
| Genz-Malik | 1e-7 | 6000 / 6000 | not converged | 4368091 | 1.343075e-7 | 65.859 |
| Gauss-Kronrod 15 | 2e-8 | 137 / 300 | converged | 7781021 | 1.981420e-8 | 92.488 |
| Gauss-Kronrod 21 | 2e-9 | 98 / 300 | converged | 15324770 | 1.998308e-9 | 180.668 |

| Comparison | XC energy difference (Ha) | Largest complex spin/k potential difference (Ha) | Largest population difference (electrons) |
|---|---:|---:|---:|
| Genz-Malik to Gauss-Kronrod 15 | 1.030e-10 | 1.856e-10 | 1.687e-10 |
| Gauss-Kronrod 15 to Gauss-Kronrod 21 | 3.038e-12 | 5.645e-10 | 1.333e-15 |

The tighter reference XC energy is -1.535281640539257 hartree. These
inter-rule differences strengthen this fixed-fixture reference substantially
relative to the uniform tensor sequence. They are observed numerical
agreement, not a universal error bound. The script retains complete complex
potential estimates for later comparisons. An atomic-grid comparison must
still use the fixed-state reference in both energy and potential; reference
convergence does not establish convergence of a new SCF density, image support
or Hamiltonian.

Re-evaluating the native 120-radial/order-95/20-bohr atomic reference at this
same density gives errors of 2.988e-8 hartree in XC energy, 1.065e-6 hartree
in the largest complex potential entry and 1.304e-8 electrons in population
against the tighter adaptive estimate. Both spin channels have potential
errors near 1e-6 hartree. This separates the remaining atomic-grid error from
the much smaller observed adaptive-reference differences. The atomic grid
still does not qualify a tight energy-and-potential profile. The physical
mode now retains its full complex reference potential for this comparison.

## Evaluated-build compatibility

The actual draft `OperatorFrame`/`FockBuild` modules in the operator
workstream were inspected and executed read-only with the MANTA checkout's
dense solver. The base revision was `7b19b5474a05`, with uncommitted module
hashes `04469508826b8ed8130c08d89a84f268efc53ac62482e504603bf3a799a89980`
for `solvers/operators.py` and
`a54bd861d1d894f9b3c5feb3a02a379a4c6fb85debbfc1a37530b75c02d3c895`
for `fock_state.py`. This identifies a checked draft, not a landed dependency.

Nine public KS jobs (H2 RKS, HeH UKS and empty-beta H/BLYP at caps 1, 2 and
converged termination) supplied 45 physical spin/k channels. Their method
energies, evaluated densities and physical Focks map directly to the draft
build; full S/X frames are shared between spins. Mutating input exports and
constructing solver-shifted variants leaves physical snapshots unchanged.
Energy agreement is 5.552e-17 hartree, density/action errors are zero,
and spectrum, residual, orthogonality and next-projector errors are below
2.806e-15. A capped evaluated density differs from its next projector by
0.11049, so conflating these states would fail this witness. Retained dense
array storage is at most 2736 bytes for these tiny exports, not a general
memory bound or a native-loop integration claim.

The refinement API remains a component diagnostic, and the bounded native
SCF loop retains its own storage contract. Neither this export check nor the
shared periodic grid qualifies BIPOLE/CRYSTAL source parity, symmetry/IBZ
reduction, finite-image convergence or general production accuracy profiles.

## Explicit hybrid semilocal assessment

Measured 2026-10-10 with the same frozen states, three atomic grids and
independent tensor-cell integration described above. Three fresh workers,
each with one cold assessment and five repeats, use one BLAS/OpenMP/Accelerate
thread. These timings include all three grids and semilocal evaluations;
exchange and SCF are not part of this diagnostic.

```sh
.venv/bin/python benchmarks/manta_periodic_xc_refinement.py --hybrid
```

The profiles use PBE0 RKS, CAM-B3LYP UKS and empty-beta LC-BLYP UKS.
`assess_periodic_hybrid_xc_refinement` shares the bounded grid comparator but
explicitly marks its partial component. Each child retains either the 0.25
full-Coulomb exchange requirement or the complete signed Coulomb/erfc list.
The separate pure-XC entry point continues to refuse hybrids.

| Profile | Cold (s) | Median (s) | Range (s) | Driver reservation (bytes) | Modeled peak (bytes) |
|---|---:|---:|---:|---:|---:|
| PBE0 RKS | 0.264502 | 0.258675 | 0.253293 to 0.273597 | 135664 | 17623013 |
| CAM-B3LYP UKS | 0.282774 | 0.274816 | 0.266026 to 0.278049 | 136428 | 17624437 |
| Empty-beta LC-BLYP | 0.263720 | 0.266209 | 0.261553 to 0.272813 | 136644 | 17624725 |

Each executes 26784 shell calls and 3419712 primitive/grid products. Metadata
allowances reflect the bundled libxc component counts. Exchange remains
outside this declared semilocal-only diagnostic. Peak memory includes the driver reservation and
retained reference/candidate; it is a numerical model, not process RSS.
No speedup comparison is claimed.

Independent Gaussian/libxc integration on the actual grid points gives
maximum errors of 1.111e-16 hartree in energy, 4.997e-16 in reference potential,
4.885e-15 electrons in populations and 3.117e-15 in the reported energy and
potential differences.

| Profile | 64-to-80 tensor energy change (Ha) | 64-to-80 tensor potential change (Ha) | Selected reference energy error vs tensor 80 (Ha) | Selected reference potential max error vs tensor 80 (Ha) |
|---|---:|---:|---:|---:|
| PBE0 RKS | 3.220e-15 | 3.936e-14 | 9.237e-7 | 4.924e-7 |
| CAM-B3LYP UKS | 1.310e-11 | 2.145e-11 | 7.889e-8 | 2.768e-7 |
| Empty-beta LC-BLYP | 4.251e-12 | 8.909e-12 | 7.256e-8 | 4.376e-7 |

The nonzero reference errors reinforce that the zero comparison row does not
certify grid accuracy. Exchange cutoffs, their zero-mode convention, source
images and density relaxation require separate assessments. General
functional/grid accuracy profiles remain open.

The 39 new tests cover independent RKS/UKS contractions, HSE06, CAM-B3LYP,
LC-omegaPBE, omegaB97X, multiple ranges, cancelled metadata and empty spin.
They also check energy-derivative consistency, every spin/k defect, exact
joint resource caps and one-short refusals, ownership, layouts and shell-image
gauges. Five complete global/range-separated hybrid builds at the same
HCORE density verify that changing only the XC grid gives the same full
Fock and energy difference. Their exchange, singular correction and other
physical components stay identical. Unsupported external providers cannot
execute. The broader regression set passed 403 tests before the final
external-provider refusal check, which also passed; Sphinx succeeded.

These partial potentials are source contributions for the method-owned build,
not complete FockBuild snapshots. No operator adapter, BIPOLE source change or
symmetry admission is introduced. The native binary SHA-256 is
`ba791a55e2d40ea8d5cd7b80bd2df358e2c1ea767c3f6f4a826787af63bd8fe6`.
