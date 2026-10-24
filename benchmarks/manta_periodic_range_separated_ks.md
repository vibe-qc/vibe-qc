# Explicit finite range-separated periodic KS

Run `python benchmarks/manta_periodic_ks.py --range-separated --hybrid-q0 strict-zero-mode`
and repeat with `--hybrid-q0 BvK-ewald` in a fresh own-checkout editable
runtime. Each command starts three fresh workers with one BLAS/OpenMP/Accelerate
thread, one cold execution and five timed repeats. The previous result is
released before each admitted call. Keep operational JSON outside the checkout.

The profiles are H2/HSE06 RKS, unequal-spin HeH/CAM-B3LYP UKS and empty-beta
H/LC-BLYP UKS. They share the skew cell, shifted three-point full mesh,
26 nonzero Hartree G vectors, origin-only physical pair translations,
3 bohr pair cutoff, 27 canonical XC images and 1000 tensor-cell quadrature
points of the explicit KS fixture. Base exchange domains have 26 Q vectors
on each diagonal transfer and 27 off diagonal, 240 total. Each erfc component
adds three exact zero vectors with their finite analytic kernel values and
ordinary source-k weights. These deliberately finite sources are not
grid-, cutoff- or mesh-converged reference Hamiltonians.

The independent reference uses fixed functional coefficients, Gaussian pair
Fourier quadrature, explicit analytic reciprocal kernels, AO polynomials and
spatial derivatives, libxc integrands, libint one-electron matrices and an
independent probe-charge Ewald sum. SciPy generalized eigensolves iterate
the independent SCF to a 1e-12 density change. Acceptance requires density
agreement within 2e-9 and same-density Fock, total and individual energy
components, and grid populations within 2e-10. This compares the identical
canonical Coulomb-plus-erfc finite source, not an alternative range-dependent
singularity prescription. The oracle is a correctness reference, not an
optimized production timing baseline.

## Measured qualification

On 2026-10-09, 2442 focused regression checks passed, including the first
70 range-separated cases, existing pure/global-hybrid KS and public jobs,
semilocal and screened exchange, functional solvers, citations and test gates.
Review then added an independent transfer-source work charge for the
degenerate case where all exchange coefficients cancel. After rebuilding,
209 focused checks passed, including all 71 range-separated cases and the
pure/global drivers and hybrid domains. Sphinx succeeded. The measurements
below use that final build, after the test workload completed.

| Requested policy | Profile | Cold (s) | Median (s) | Range (s) | CPU median (s) | Iterations / physical builds | Modeled peak (bytes) | RSS (MiB) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| strict-zero-mode | rks | 0.024294 | 0.023217 | 0.022859 to 0.024144 | 0.023174 | 8 / 9 | 95273665 | 166.0 |
| strict-zero-mode | uks | 0.065625 | 0.062938 | 0.062636 to 0.072205 | 0.062797 | 14 / 15 | 112260005 | 165.8 |
| strict-zero-mode | empty-spin | 0.040752 | 0.039836 | 0.039368 to 0.040364 | 0.039813 | 14 / 15 | 112132909 | 165.6 |
| BvK-ewald | rks | 0.023391 | 0.023486 | 0.022853 to 0.023678 | 0.023442 | 8 / 9 | 95273665 | 164.6 |
| BvK-ewald | uks | 0.083131 | 0.082122 | 0.080730 to 0.082575 | 0.081855 | 18 / 19 | 112525349 | 165.5 |
| BvK-ewald | empty-spin | 0.048386 | 0.047178 | 0.045682 to 0.048615 | 0.047091 | 17 / 18 | 112398253 | 164.8 |

HSE06 has zero long-range Coulomb coefficient, so its actual exchange policy
is `not-applicable` for either request. Its D/F and energy are bitwise
identical; no Madelung setup or correction is performed. CAM-B3LYP and LC-BLYP
use long-range fractions 0.65 and 1, respectively, for the selected correction.
The finite erfc zero contributes in both requested policies.

Maximum independent errors across the six workers were 1.433e-10 in density,
4.358e-14 Ha in any Fock element, 3.908e-14 Ha in total electronic energy,
4.439e-14 Ha in individual energy components and 8.882e-16 electrons in
quadrature populations. Separate regression witnesses qualify the nuclear
energy, zero-mode spin/k factors, complex density derivatives, multiple ranges,
rank truncation, cancelled components, image gauges and capped evaluated epochs.
Source and grid population differences remain explicit finite-fixture
diagnostics; population agreement alone does not certify potential accuracy.

| Requested policy | Profile | Shell calls | Primitive pairs | Primitive points |
| --- | --- | ---: | ---: | ---: |
| strict-zero-mode | rks | 8592 | 816 | 486000 |
| strict-zero-mode | uks | 24894 | 5454 | 1215000 |
| strict-zero-mode | empty-spin | 15384 | 2424 | 810000 |
| BvK-ewald | rks | 8592 | 816 | 486000 |
| BvK-ewald | uks | 31518 | 6894 | 1539000 |
| BvK-ewald | empty-spin | 18456 | 2904 | 972000 |

Aggregate admission includes all component kernels, both retained Q layouts,
receipt ownership, spin matrices, functional workspace, evaluated-state
history, solver and final physical confirmation. Source traversal remains
charged when no exchange leaf survives cancellation. Exact simultaneous and
one-short memory/shell/primitive/scalar/kernel checks pass before poisoned
payloads are read. This is a modeled numerical bound; process RSS includes
Python, native libraries, fixtures and the independent oracle. These small
measurements claim no performance advantage or scalable crossover.

Native binary SHA256: `a091b364d87e569d3489b63881eee365567905f7ac333e35e704144ec0ad924a`.

## Shared Fock object compatibility

The actual draft FockBuild/OperatorFrame modules consumed nine exported
range-separated states: the three profiles above at iteration caps 1, 2 and
100. All 45 spin/k blocks preserved physical S/X frames, owned evaluated D/F,
component energies, k weights and solver-modification parentage. Density and
operator-action errors were exactly zero; maximum energy, spectrum, residual,
metric and next-projector errors were 4.441e-16, 2.887e-15, 2.908e-15,
2.673e-15 and 2.776e-15, respectively. The maximum capped density versus next
orbital projector separation was 0.120579, confirming that the adapter did
not silently replace the evaluated density with the newly diagonalized state.

The recheck used `fock_state.py` SHA256
`d2f9476e8ab36e69413e80f014280ee0a3e75f1bec44f853cd650d81d0650210`
and `solvers/operators.py` SHA256
`04469508826b8ed8130c08d89a84f268efc53ac62482e504603bf3a799a89980`,
including the draft's owned scalar energy and complete build-identity fixes.
Only exported snapshots entered this check; no unlanded module or Python
adapter was added to the native bounded solver. BIPOLE AC/BD source matching,
CRYSTAL parity and method-owned symmetry admission remain independent gates.
Cutoff/grid factories have the separate qualification below. Public jobs and
production accuracy profiles remain required increments.

## Cutoff and joint atomic-grid factories

Add `--hybrid-source cutoffs` or `--hybrid-source grid` to each of the two
range-separated benchmark commands above. The cutoff profile retains the
1000-point explicit tensor grid. Both profiles use a reciprocal kinetic-energy
cutoff of 2 Ha, a physical pair cutoff of 3 bohr and a canonical XC image
radius of 11 bohr. The shared certified producer generates 264 nonzero base
exchange vectors and 27 XC images; each erfc kernel adds its three finite
zero vectors in the method. Source owners reserve 162752 modeled bytes.

The atomic profile uses 12 radial points, Lebedev order 17, an 8 bohr periodic
Becke partition radius, and 2640 points for H2/HeH or 1320 for H. Its additional
retained grid allowance is 233920 or 133344 bytes, respectively. All domain
and grid work is included in each complete timed solve. These are source
choices, not automatic accuracy profiles or equal-grid performance comparisons.

The rebuilt native implementation passed 497 focused checks across physical
HF/KS, cutoff domains, joint grids, public pure/global-hybrid jobs and test
gates, including 62 new range-separated domain/grid cases. Three additional
geometry-identity and cancelled-exchange tests then passed, for 65 new cases.
Sphinx succeeded. Measurements used twelve fresh single-thread workers, each
with one cold execution and five repeats after the test workload completed.

| Source | Requested policy | Profile | Cold (s) | Median (s) | Range (s) | Iterations / builds | Realized / admitted modeled peak (bytes) | RSS (MiB) |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| cutoffs | strict-zero-mode | rks | 0.016051 | 0.015936 | 0.015703 to 0.016093 | 6 / 7 | 95444465 / 95570609 | 166.1 |
| cutoffs | strict-zero-mode | uks | 0.061879 | 0.062428 | 0.061738 to 0.062527 | 15 / 16 | 112433829 / 112625573 | 165.6 |
| cutoffs | strict-zero-mode | empty-spin | 0.036537 | 0.036378 | 0.035698 to 0.036799 | 14 / 15 | 112306733 / 112467277 | 165.1 |
| cutoffs | BvK-ewald | rks | 0.016512 | 0.016414 | 0.016011 to 0.016771 | 6 / 7 | 95444465 / 95570609 | 164.2 |
| cutoffs | BvK-ewald | uks | 0.072889 | 0.073182 | 0.072289 to 0.073951 | 18 / 19 | 112699173 / 112890917 | 165.5 |
| cutoffs | BvK-ewald | empty-spin | 0.043796 | 0.042455 | 0.042218 to 0.043806 | 17 / 18 | 112572077 / 112732621 | 165.1 |
| grid | strict-zero-mode | rks | 0.042960 | 0.043334 | 0.042641 to 0.043699 | 6 / 7 | 95783345 / 95909489 | 175.5 |
| grid | strict-zero-mode | uks | 0.143843 | 0.144402 | 0.142812 to 0.146694 | 14 / 15 | 112772709 / 112964453 | 176.0 |
| grid | strict-zero-mode | empty-spin | 0.046241 | 0.045810 | 0.045554 to 0.046377 | 14 / 15 | 112460557 / 112621101 | 167.7 |
| grid | BvK-ewald | rks | 0.043763 | 0.042884 | 0.042595 to 0.043221 | 6 / 7 | 95783345 / 95909489 | 174.8 |
| grid | BvK-ewald | uks | 0.184728 | 0.190331 | 0.186562 to 0.199461 | 18 / 19 | 113038053 / 113229797 | 175.8 |
| grid | BvK-ewald | empty-spin | 0.056792 | 0.055052 | 0.054653 to 0.055473 | 17 / 18 | 112725901 / 112886445 | 167.9 |

Maximum independent cutoff-profile errors were 1.269e-10 in density,
4.353e-14 Ha in Fock, 3.820e-14 Ha in electronic energy, 4.436e-14 Ha in
individual components and 2.665e-15 electrons in grid populations. The
corresponding atomic-grid maxima were 1.604e-10, 4.341e-14 Ha, 3.997e-14 Ha,
4.436e-14 Ha and 3.109e-15 electrons. Every result includes a physical
confirmation build. Nuclear terms have their independent Ewald witness.

The tests separately verify integer-sphere membership, conjugacy, exact
boundaries, AO-image gauge transport, both grid partitions, mixed ranges,
owned exports and capped states. All five jointly exact resource caps admit;
each one-short cap refuses before poisoned arrays or invalid partition
generation. Geometric identity remains separate from functional identity.
Even a composite with cancelled exchange retains admission for its generated
base source. RSS is measured, not bounded by the modeled numerical allowance.

The same patched FockBuild/OperatorFrame modules also consumed nine joint-grid
states, 45 spin/k blocks, retaining domain and quadrature provenance. D/F
actions and total energy were exact; maximum spectrum, residual, metric and
next-projector errors were 2.443e-15, 2.478e-15, 2.425e-15 and 2.332e-15.
The maximum capped density/next-projector separation was 0.114365. This remains
an export check, with no adapter inside the bounded native solver.

Native binary SHA256: `e49b857cb7567f0a3fd20c1d82f9c599d8a4f06811625cb4356d767439d51f91`.
Public range-separated jobs now have separate artifact-delivery acceptance in
`benchmarks/manta_periodic_ks_job.md`. Continuum accuracy, BIPOLE CRYSTAL parity
and symmetry-reduced execution remain separately gated.
