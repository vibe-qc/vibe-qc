# Explicit finite physical periodic KS

Explicit range-separated profiles have their own
[qualification report](manta_periodic_range_separated_ks.md).

Run `python benchmarks/manta_periodic_ks.py` in the fresh own-checkout editable
runtime. Three fresh workers use one BLAS/OpenMP/Accelerate thread, record cold
execution and five timed repeats, and release the previous result before the
next admitted call. Keep operational JSON outside the product checkout.

Profiles are H2/PBE RKS, unequal-spin HeH/PBE UKS and empty-beta H/BLYP UKS.
They use a skew cell, a shifted three-point full mesh, 26 nonzero reciprocal
vectors, origin-only physical pair translations, a 3 bohr pair cutoff,
27 canonical XC images and 1000 tensor Gauss-Legendre cell points. The grid
batch is 64. These deliberately finite fixtures qualify the composed
Hamiltonian; their energies are not grid- or source-converged reference data.

Independent Gaussian quadrature constructs Hartree. Explicit AO polynomials
and analytic spatial derivatives with libxc construct XC, while libint and
Gaussian quadrature supply the one-electron witness. SciPy eigensolves iterate
the independent SCF to a 1e-12 density change. Timed results must agree within
2e-9 in density and 2e-10 in same-density Fock, energy and grid population.
The independent oracle time includes full-grid temporary owners and is not
a production speed baseline.

Aggregate modeled admission includes all source leaves, simultaneous Hartree
and XC results, density copies, the bounded functional solver, physical S/H,
iteration history, exported results and terminal confirmation. Peak RSS also
includes Python, libraries, fixtures and the independent full-grid oracle.
No process RSS cap, asymptotic speedup or production accuracy profile is claimed.

## Measured qualification

On 2026-10-08 the fresh own-checkout build passed 740 focused checks,
including all 38 physical KS tests, physical RHF/UHF regression, semilocal XC,
the joint functional solver, citations and repository gates. Two optional
PySCF checks were deselected. Sphinx succeeded. Two rank-truncation cases and an explicit terminal-confirmation failure
case ran after the 737-check combined invocation, in the complete 38-case
KS file. Exact caps include external live memory and the retained energy
scalar; exported snapshot lifetime and shell-image gauge checks also pass.

The isolated three-worker measurements follow the regression workload.
Each worker runs one cold calculation and five repeats.

| Profile | Cold (s) | Median (s) | Range (s) | CPU median (s) | Iterations / physical builds | Modeled peak (bytes) | XC panel (bytes) | RSS (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| rks | 0.011998 | 0.011230 | 0.010927 to 0.011643 | 0.011167 | 3 / 4 | 79371939 | 43008 | 167.3 |
| uks | 0.054072 | 0.054147 | 0.052928 to 0.054620 | 0.053802 | 9 / 10 | 79431321 | 55296 | 167.6 |
| empty-spin | 0.036728 | 0.035553 | 0.035242 to 0.036347 | 0.035488 | 10 / 11 | 79329213 | 43008 | 165.3 |

Maximum independent errors were 1.149e-10 in converged density,
3.664e-15 Ha in same-density Fock, 6.218e-15 Ha in energy and
8.882e-16 electrons in quadrature populations. Independent oracle times
were 0.040, 0.151 and 0.080 s, respectively. Nuclear energies have a separate
explicit Ewald-sum regression witness. Each converged result includes one
additional physical build at its returned density.

Grid populations differ from fixed metric populations by 0.000729, 0.001884
and 0.015101 electrons for these finite fixtures. Those differences are
reported diagnostics, not oracle errors or evidence of a converged grid.
A separate refinement study found that a HeH GGA grid with charge agreement
of 5.42e-11 electrons could still have a 1.09e-4 Ha potential error. Even
order80-to96 refinement changed energy by only 1.21e-9 Ha while changing the
maximum potential element by 1.24e-6 Ha. Pair-cutoff metric and uncut grid
populations can also differ after quadrature refinement. Source, quadrature
energy, quadrature potential and solver convergence therefore need separate
acceptance gates in future grid/domain factories.

Native binary SHA256: `de70a96e8d572ed7d49a3b4fd4a6b587a045249600ddc6006f7c87b85cc06387`.

A separate reviewer compiled the actual driver and bindings with their
physical dependencies in an isolated module. All 310 snapshotted source/header
files matched this implementation, including explicit-copy snapshot bindings.
The 131 driver/planner invocations included 64 completed driver runs,
33 refused driver runs, five admitted plans and 29 refused plans. Five
profiles covered H2 LDA/PBE RKS, equal-spin H2 PBE UKS, unequal-spin HeH PBE
UKS and empty-beta H BLYP UKS. The first six evaluated epochs and converged
states agreed with an independent Gaussian-quadrature/polynomial/libxc and
SciPy reference: maximum energy error 1.49e-14 Ha, Fock error 5.25e-14 Ha,
spectrum error 8.31e-14 Ha and density error 9.84e-11. Metric orthogonality
and individual AO-image gauge errors were below 3.0e-15 and 2.0e-15.

Confirmation added exactly one physical call without changing evaluated
history. Tightening both confirmation tolerances to 1e-30 correctly kept the
overall result unconfirmed despite numerical convergence. All five jointly
exact caps admitted; 50 one-short planning/execution cases refused poisoned
arrays before payload work. C/F/unaligned arrays, batches 1/13/512, all six
owned array exports and standalone snapshot/memory/history lifetimes passed.
Two 5-AO/rank-4 physical cases passed exact caps with 123456 external live
bytes. This review qualifies the finite driver composition and its binding,
not public jobs, process RSS or grid/support convergence. Existing S/H
producers and libxc internals retain their separate qualification evidence.


## Explicit-source global hybrids

Run `python benchmarks/manta_periodic_ks.py --hybrid-q0 strict-zero-mode`
and repeat with `--hybrid-q0 BvK-ewald`. Each command starts three fresh
single-thread workers, with one cold run and five repeats. The full shifted
three-point mesh, physical G/translation supports and 1000-point cell
quadrature match the finite pure-KS fixture above. Target/source exchange
supports have 26 vectors on the diagonal and 27 off diagonal. PBE0 supplies
RKS H2 and unequal-spin UKS HeH; empty-beta H uses a custom 20% exact-exchange,
80% B88 plus full LYP hybrid. The finite sources are not continuum references.

Independent analytic Gaussian Fourier panels now also build each spin's
exchange. The reference supplies its own known mixing fraction, independent
probe-charge Ewald, explicit AO/libxc contraction and SciPy fixed-point SCF.
Acceptance includes every Fock element and separate one-electron, Hartree,
semilocal, uncorrected exchange and zero-mode correction energies. It also
checks the converged density and grid populations. Full source refinement,
production grid profiles and comparative speed claims remain open.

Measurements on 2026-10-09, after the 363-check physical KS/HF regression
run, are below. RSS includes the interpreter, libraries, fixtures and the
independent oracle. The modeled bound jointly reserves explicit source
copies, J/K/XC providers, Madelung setup/scratch, solver and confirmation.

| Zero mode | Profile | Cold (s) | Median (s) | Range (s) | CPU median (s) | Iterations / builds | Modeled peak (bytes) | RSS (MiB) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| strict-zero-mode | rks | 0.018367 | 0.017828 | 0.017643 to 0.018093 | 0.017832 | 7 / 8 | 95231117 | 165.6 |
| strict-zero-mode | uks | 0.052249 | 0.054076 | 0.052113 to 0.055026 | 0.054065 | 13 / 14 | 95304383 | 164.5 |
| strict-zero-mode | empty-spin | 0.030069 | 0.031195 | 0.030760 to 0.032769 | 0.031191 | 12 / 13 | 96238267 | 163.5 |
| BvK-ewald | rks | 0.012726 | 0.012192 | 0.011832 to 0.012547 | 0.012191 | 5 / 6 | 95496461 | 163.6 |
| BvK-ewald | uks | 0.055368 | 0.054697 | 0.053039 to 0.057362 | 0.054693 | 14 / 15 | 95569727 | 164.5 |
| BvK-ewald | empty-spin | 0.033312 | 0.031720 | 0.030401 to 0.032026 | 0.031721 | 12 / 13 | 96503611 | 163.7 |

Maximum independent errors were 1.499e-10 in converged density,
9.603e-15 Ha in the same-density Fock, 1.243e-14 Ha
in electronic energy, 1.707e-14 Ha in individual energy
components and 1.998e-15 electrons in grid populations.
The independent oracle took 0.036 to 0.145 s.
This measures this finite implementation; it is not a speed comparison with
another production method or an accuracy qualification of its supports.

The 56-case hybrid regression separately covers built-in B3LYP, custom LDA,
RKS/UKS equivalence, both zero-mode factors, complex energy derivatives,
source identity changes, owned exports, AO-image gauges, rank truncation,
capped evaluated epochs and all five exact/one-short resource caps. Empty-spin
PBE0 and B3LYP retain separate backend-conditioning witnesses: small changes
in occupied density at the exactly polarized endpoint can amplify the
minority potential. No regularization or relaxed blanket Fock tolerance is
introduced. Strict empty-spin full-matrix tests use the B88/LYP hybrid above.

Nine native hybrid snapshots (three spin profiles at iteration caps 1, 2 and
100) were exported into the actual shared FockBuild/OperatorFrame draft.
All 45 spin/k channels preserved physical actions and evaluated densities
exactly. Maximum energy error was 4.441e-16 Ha; spectrum, metric, residual and
next-projector errors were below 2.611e-15. The capped evaluated density and
next projector differed by up to 0.1343, confirming their separation. This
checks the inspected draft export seam, not a landed generic runtime adapter
or a Python allocation inside the bounded loop. BIPOLE parity and symmetry
qualification are separate workstreams.

Native binary SHA256: `387686a4c34e040e80678e959af5df246a6c4d5346b3cb3c76f2403833428599`.
Inspected draft operator and state module SHA256 values:
`04469508826b8ed8130c08d89a84f268efc53ac62482e504603bf3a799a89980` and
`a54bd861d1d894f9b3c5feb3a02a379a4c6fb85debbfc1a37530b75c02d3c895`.


## Cutoff and joint atomic-grid global hybrids

Run the same script with `--hybrid-q0 BvK-ewald --hybrid-source cutoffs`
and `--hybrid-q0 BvK-ewald --hybrid-source grid`. Each starts three fresh
single-thread workers with one cold run and five repeats. The species,
functionals and shifted three-point full mesh are unchanged from the explicit
hybrid study. Reciprocal kinetic cutoff 2 Ha, physical pair cutoff 3 bohr,
nuclear cutoff 4 bohr and XC-image radius 11 bohr select origin-only physical
pair translations, 264 target/source Q vectors and 27 XC images. The cutoff
mode keeps the 1000-point tensor cell grid. Joint-grid mode uses 12 radial
points, Lebedev order 17 and Becke partition radius 8 bohr, producing 2640 points
for H2/HeH and 1320 for H. These are deliberately finite consistency fixtures,
not recommended production grid profiles.

The timed calls include source generation and, for `grid`, atomic quadrature
construction. Independent Gaussian/libxc contractions and SciPy SCF use the
exact exported finite supports after timing. Both whole-request preflight
and realized-SCF modeled peaks are reported: the former uses enclosing box
bounds before numerical enumeration. RSS also includes Python, libraries,
fixtures and the full-grid independent witness, so it is a different quantity.

Measurements on 2026-10-09 follow 365 passing focused regressions, including
all 44 new domain/grid checks. Six optional PySCF tests were excluded. The old
pure benchmark mode also passed a three-repeat smoke check.

| Source | Profile | Cold (s) | Median (s) | Range (s) | CPU median (s) | Iterations / builds | Preflight peak (bytes) | Realized plan peak (bytes) | RSS (MiB) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| cutoffs | rks | 0.011402 | 0.010044 | 0.009683 to 0.010167 | 0.010045 | 4 / 5 | 95764893 | 95664669 | 165.9 |
| cutoffs | uks | 0.056641 | 0.057954 | 0.056893 to 0.059112 | 0.057952 | 15 / 16 | 95869359 | 95737935 | 165.1 |
| cutoffs | empty-spin | 0.028586 | 0.028129 | 0.027816 to 0.028575 | 0.028034 | 12 / 13 | 96772043 | 96671819 | 164.4 |
| grid | rks | 0.027966 | 0.028190 | 0.027629 to 0.028684 | 0.028189 | 4 / 5 | 96103773 | 96003549 | 174.3 |
| grid | uks | 0.137175 | 0.134837 | 0.133320 to 0.136980 | 0.134595 | 14 / 15 | 96208239 | 96076815 | 174.9 |
| grid | empty-spin | 0.037621 | 0.037554 | 0.036882 to 0.038024 | 0.037538 | 12 / 13 | 96925867 | 96825643 | 166.8 |

Maximum independent errors: density 1.569e-10, Fock 9.770e-15 Ha,
electronic energy 1.110e-14 Ha, individual energy components
1.708e-14 Ha and grid populations 5.995e-15 electrons.
Independent SCF/reference work took 0.035 to 0.277 s.
This is not a matched-accuracy speed comparison with the earlier explicit
supports or another production code.

The tests additionally prove integer sphere membership on skew Gamma and
shifted meshes, opposite-transfer conjugacy, exact reciprocal boundaries,
shared HF pair/G/Q identity, separate XC image identity, owned exports,
AO-image gauge transport, both zero-mode policies and both grid partitions.
All five caps are tested jointly at equality and one unit below, including
external live bytes. A coincident-center test proves full refusal before
partition generation. Capped states preserve the evaluated D/F/E.

The joint-grid hybrid export also passed nine snapshot crossings into the
same actual shared FockBuild draft used above: 45 spin/k channels, with D and
operator-action error 0, energy error 4.441e-16 Ha, and spectrum, residual,
metric and next-projector errors below 2.649e-15. The maximum capped evaluated
D versus next-projector separation was 0.12484. No generic Python adapter was
inserted in the native loop, and no BIPOLE or symmetry producer was enabled.

Native binary SHA256: `3a0b7e18b7a7c195bf55ac936d82f3a8486f34a349b8e3365e58d22b538ddf4a`.
