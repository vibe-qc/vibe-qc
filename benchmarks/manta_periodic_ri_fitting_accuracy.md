# Complete periodic RI fitting target admission

Run `benchmarks/manta_periodic_ri_fitting_accuracy.py` in the built checkout.
Two complete RHF/UHF profiles compare enabled and disabled full-Coulomb
fitting admission at unchanged sources. Each uses one cold pair and two
alternating warm pairs, with bitwise S/H/F, density and energy checks.
Prepared reference construction and preliminary fixture setup are outside
the measured intervals; every enabled method still reserves their costs.

The certificate uses the actual Gamma RI producer and canonical overlap
density family before SCF. J/energy targets are deliberately loose to
qualify enclosure and integration, not a recommended practical preset.
Fitting, selected rank, coupling and source-domain differences are included
together. Floating arithmetic, rank stability and density relaxation are
excluded. Useful automatic auxiliary/reference selection remains open.

Source `71c9c45fce409d03dc42412012f7128f0139f25c`; native SHA256
`73c8d65eebb7105d37ccb5584097612b6128a1cb6890787e99d83cac65008e88`. All 1,903 source/reference pins and native hash
remain unchanged through local `vq` acceptance. Six profiles and their
controls converge, with bitwise S/H/F, density, energy and physical source
identity equality. QVF and fitting-accuracy citations pass.

| Profile | Iterations | Energy (Eh) | Uniform J bound | Coulomb-energy bound (Eh) |
|---|---:|---:|---:|---:|

| RHF | 9 | -8.745245080287464 | 18.882522316 | 1052.998515741 |
| UHF | 13 | -3.676437147719352 | 18.882522316 | 1052.998515720 |
| RKS | 9 | -9.680715568887706 | 18.880626229 | 1053.398334634 |
| UKS | 11 | -4.133630833057696 | 18.880626229 | 1053.398334613 |
| PBE0 | 9 | -9.479314352774018 | 18.880626229 | 1053.398334634 |
| HSE06 | 9 | -9.660618685880872 | 18.880626229 | 1053.398334634 |

Each certificate uses 18 real Hermitian probes and canonical weighted
L1 limit 784.535934. The selected-rank/coupling/domain discrepancy is
included together with the fitting error. At the converged HF densities,
independent full Gaussian Fourier sums give J errors 0.00133437037428
(RHF) and 0.000918829461021 (UHF), and Coulomb-energy errors
0.0172326000345 and 0.00712691087670 Eh. Cubes 36/40 converge within
1.34e-15 in J and 2.67e-15 Eh in energy; sampled Fourier panels also agree
with shifted-contour quadrature. Independent finite-source Fock, energy
and SciPy SCF density differences are below 5.0e-15, 1.12e-13 Eh and
3.23e-10. KS acceptance checks unchanged-source controls; independent KS
regression references are reused. No matched ORCA/CRYSTAL custom-domain
reference is available.

Twenty-one new checks cover all six methods, shared certificate identity,
all five caps, pre-SCF refusal, count-only dry runs, policy/frame rejection,
fitted-K ownership and combined explicit/canonical family provenance.
Shared regressions expose and fix the range exchange-only citation guard.
Two fresh benchmark processes produce four warm samples per profile:

| Profile | Plain wall (s) | Certified wall (s) | Plain CPU (s) | Certified CPU (s) |
|---|---:|---:|---:|---:|
| RHF | 0.143179 | 0.401276 | 0.136978 | 0.395163 |
| UHF | 0.168178 | 0.424271 | 0.162357 | 0.417603 |

The certificate/reference peak reservation is 553,066,098 bytes,
including 188,120,112 retained reference bytes and conservative overlapping
parent reservations. Benchmark RSS is 261.80 MiB; complete acceptance
with independent references peaks at 417.83 MiB. Reference setup is
outside timings but included in admission. These small fixtures establish
correctness and overhead, not a production speedup or practical accuracy
preset. Useful auxiliary/reference selection, arithmetic, rank stability,
relaxation and all remaining original capabilities are still required.


After integrating the shared Cartesian-power Fourier cache, source
`a508d404496b48dd40bf6bb2c7e9184a9bc83668` is rebuilt with native SHA256
`e9f6fb7156f7e47a2d0d57bc6c4f4524fdb0b838d31561471256d0debb6e7298`. A second source-pinned local `vq` calculation passes
the same six methods, controls, independent references and two fresh
benchmark processes. All complete certificates and energies equal the
initial acceptance exactly. The 21 admission cases and two focused
incoming Fourier-cache cases pass. Initial benchmark timings above retain
their original runtime provenance; the integration check adds no scope.


## Kernel-capped s-pair reference envelopes

Source `8fda8ceb874499b6502b90874bf41df48ba280a6`, native SHA256
`2880ee4160cc6ddef02960bc7fb886024dee1f3938a5b9d8391736386acc76a3`, retains the tightest common primitive exponent
scaling for plain s-s masses within the original global kernel ceiling.
This avoids loosening mixed angular enclosures while tightening constant
pair images. Physical supports, J/K values and occupations are unchanged.
Source-bound identities change to identify the refined envelope.

Local `vq` passes all six RI profiles and complete BUPO HF. All six RI
energies match the previous acceptance exactly; RHF/UHF and BUPO density,
Fock and energy arrays also match exactly. Independent full Fourier,
finite-source SCF and compensated BUPO references pass at unchanged
comparison tolerances. No matched local ORCA/CRYSTAL reference is available.

| Profile | Uniform J bound | Coulomb-energy bound (Eh) | Reference J remainder |
|---|---:|---:|---:|
| RHF | 0.01492061007 | 0.8708116085 | 0.0002079824073 |
| UHF | 0.01492061007 | 0.8708116085 | 0.0002079824073 |
| RKS | 0.006677999757 | 1.270630502 | 0.0002079824073 |
| UKS | 0.006677999757 | 1.270630502 | 0.0002079824073 |
| PBE0 | 0.006677999757 | 1.270630502 | 0.0002079824073 |
| HSE06 | 0.006677999757 | 1.270630502 | 0.0002079824073 |

RHF/UHF now admit fixed J/energy targets 0.02/1 Eh before SCF, with the
same six retained reference pairs. Observed Coulomb J/energy discrepancies
remain 0.00133437037428/0.0172326000345 for RHF and
0.000918829461021/0.00712691087670 for UHF. The finite selected-fitting
map discrepancy now dominates the reference remainder. These are
same-density Coulomb bounds, not total or relaxed-state energy accuracy.
Eight new checks and 77 affected checks pass; ten gradient/Fourier checks
retain their separate formulas. A mixed s/p comparison reconstructs the
original half-Gaussian bound independently and verifies nonincrease.

Two fresh benchmark processes use the same methodology as above:

| Profile | Plain wall (s) | Certified wall (s) | Plain CPU (s) | Certified CPU (s) |
|---|---:|---:|---:|---:|
| RHF | 0.140845 | 0.397845 | 0.135170 | 0.392182 |
| UHF | 0.164899 | 0.423711 | 0.159394 | 0.417921 |

Modeled reservations are unchanged. Useful automatic auxiliary/reference
selection and the full arithmetic/rank/relaxation and method matrix remain
unfinished. This tighter shared producer is a prerequisite, not completion.


## Automatic metric cutoff selection

Source `b5a0594eac4cc2320d1a712312f9a1bdb2fa90e0`, native SHA256
`0cf2af1a17108a97e75c6695b96189e11f6e279adc0b2886a106ca0a7e3287dc`, selects a metric cutoff against the full-Coulomb
fitting certificate before SCF. Shared coupling/reference producers,
canonical density families and physical exchange are preserved. The
accepted Gamma owner is kept directly; source identities identify its
selected cutoff. Auxiliary and reference domains remain explicit.

Local `vq` converges RHF, UHF, RKS, UKS, PBE0 and HSE06 after rejecting
an initial rank-zero metric. All six states match explicitly selected
controls bitwise. Independent HF full-Fourier, shifted-contour quadrature,
finite-source Gaussian/libint and SciPy SCF references pass unchanged
tolerances. A KS validation control initially retained different relative
cutoff settings; correcting the control restored source-identity agreement
without changing numerical states or product code. Passing HF evidence
was reused. No matched local ORCA/CRYSTAL custom-domain reference exists.

The default reduction factor 0.1 also passes a complete RHF calculation:

| Trial | Resolved cutoff | Rank | Uniform J bound | Coulomb-energy bound (Eh) |
|---|---:|---:|---:|---:|
| 1 | 3.660290962 | 0 | 11.50358545 | 2637.568603 |
| 2 | 0.3660290962 | 1 | 3.110519906 | 452.9792919 |
| 3 | 0.03660290962 | 1 | 3.110519906 | 452.9792919 |
| 4 | 0.003660290962 | 2 | 0.4071248355 | 36.96774724 |
| 5 | 0.0003660290962 | 3 | 0.01492061007 | 0.8708116085 |

Targets are 0.02 and 1 Eh. The first passing cutoff is retained above the
declared floor; no monotonic-error assumption replaces a certificate.
The final state matches the independently validated fixed-rank state
bitwise. Twenty-two new checks cover six methods, first success, gradual
search, resolved relative cutoffs, all five resource caps and refusals;
160 focused checks pass including documentation checks.

Run `python benchmarks/manta_periodic_ri_fitting_accuracy.py --metric-selection`
for the two-candidate search versus the explicit certified final metric.
Two fresh processes each run three repetitions per method, alternating
order. Medians of six measurements are:

| Profile | Automatic wall (s) | Fixed wall (s) | Automatic CPU (s) | Fixed CPU (s) |
|---|---:|---:|---:|---:|
| RHF | 0.629903 | 0.385673 | 0.623493 | 0.379533 |
| UHF | 0.653760 | 0.408241 | 0.647635 | 0.402133 |

The two-candidate certificate reserves 781850714 bytes,
1944312912 shell calls and
21727155992785918 kernel terms. These are conservative
upper reservations including rejected probes and replacement owners, not
measured work. Peak benchmark process RSS was
274382848 bytes. Search adds setup cost; no performance
speedup is claimed. Fitted K and separate component-accuracy policies refuse
this optional search; their fixed-cutoff paths remain available. Useful
automatic auxiliary/domain/reference selection and the complete MANTA
arithmetic, rank, relaxation and method matrix remain unfinished.


Source `e3244797ae8be997e728df20974ee11576ca9a27`, native SHA256
`39774c21fa03891c727cee82eb385d5c9f3baca32551c340e6a75455afc09046`, additionally reserves the cubic Gamma consistency
checks after replacement. Two-candidate scalar/kernel reservations become
460055885367013/21727155993228286; memory and shell counts are unchanged.
Six focused resource/gradual checks and a local `vq` calculation of both
two- and five-trial RHF searches pass with identical states and certificate
identities. Numerical comparisons and timings above retain their recorded
source/runtime; this accounting adjustment adds no accuracy scope.


## Automatic shared reference selection

Source `c8a2bb129e07bbd91fe894c1d669bb3e25563849`, native SHA256
`e2da917d9fd6101c3668b495a4953400c077add91e9112b401c899dc2c5218b5`, prepares and refines the shared BUPO reference
inside complete RI HF/KS jobs. The actual Gamma owner and fixed canonical
family are used throughout. Default reference controls grow their domains
until the reference remainder is tight enough for the targets. A following
metric search rejects rank zero and retains the first fully certified map.

Local `vq` accepts all six profiles with bitwise explicit selected-reference
controls, identical physical Fock source identities and valid QVF output:

| Profile | Reference trials | Uniform J bound | Coulomb-energy bound (Eh) | Total energy (Eh) |
|---|---:|---:|---:|---:|
| RHF | 4 | 0.01515674911 | 0.9426209762 | -8.745245080287 |
| UHF | 4 | 0.01515674911 | 0.9426209762 | -3.676437147719 |
| RKS | 4 | 0.00687218021 | 1.342439824 | -9.680715568888 |
| UKS | 4 | 0.00687218021 | 1.342439824 | -4.133630833058 |
| PBE0 | 4 | 0.00687218021 | 1.342439824 | -9.479314352774 |
| HSE06 | 4 | 0.00687218021 | 1.342439824 | -9.660618685881 |

The RHF default search uses these reference domains:

| Trial | Pair cube | Real cube | Reciprocal cube | Pair cutoff (bohr) | Reference J remainder |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 1 | 3 | 416.1765714 |
| 2 | 1 | 2 | 2 | 4.5 | 1.56881174 |
| 3 | 2 | 3 | 3 | 6.75 | 0.03492662507 |
| 4 | 3 | 4 | 4 | 10.125 | 0.0002738839928 |

Final HF states and fitted J match the prior independently validated arrays
bitwise and retain the same RI source identities. Reused full-Fourier cubes
36/40, shifted-contour quadrature, finite Gaussian/libint and SciPy SCF
evidence therefore applies to the physical states. The new certificates
also enclose those independent full-Coulomb values. KS uses explicit selected
controls and existing independent regression evidence. No matched local
ORCA/CRYSTAL custom-domain reference is available. The 455 focused checks
include all five parent/internal search caps and explicit refusals.

Run `python benchmarks/manta_periodic_ri_fitting_accuracy.py --reference-selection`.
Two fresh processes with `--repetitions 1` each run RHF and UHF, always
an automatic search followed by the selected explicit-reference job.
Medians of the two measurements per profile are:

| Profile | Automatic wall (s) | Fixed wall (s) | Fixed reference preparation (s) | Automatic CPU (s) | Fixed CPU (s) |
|---|---:|---:|---:|---:|---:|
| RHF | 63.526001 | 57.408125 | 0.005293 | 63.448362 | 57.365870 |
| UHF | 61.756201 | 58.169917 | 0.004934 | 61.714143 | 58.118741 |

Automatic timings include all rejected reference preparation and probes;
fixed-job timings exclude the separately reported reference preparation.
The benchmark leaves metric refinement disabled to isolate reference search.
Peak benchmark process RSS was 270270464 bytes. This is a
conservative accuracy search with substantial setup overhead, not a speedup
or production crossover claim.

The declared default search cap is 1 GiB, 10^12 shell calls, 10^14 primitive
products, 10^17 scalar and 10^18 kernel terms. Each admitted stage/probe is
charged cumulatively, with rejected work retained. Output separates these
caps from the actual admitted cost bounds. Auxiliary and physical fitting
domain selection, arithmetic/rank/relaxation ledgers and the rest of the
MANTA capability matrix remain unfinished. Same-density Coulomb bounds do
not establish relaxed total-energy accuracy.


Final source `c01c7ace969fc193c17ee84d7c70c179e863e113`, native SHA256
`e2931aa8c46eb3e4ada26b49db88b527d3fa5b9c6c6a4a5722075bcc81175caf`, additionally admits the minimum search setup
in count-only planning and reserves any larger search backend margin while
preserving the original RI leaf plan. Four new checks and local `vq`
calculations with default and larger margins pass with identical accepted
RHF states and certificate identities. Reference provenance now contains
all preparation controls and the backend margin. The numerical evidence
and timings above retain their original source/runtime; they are reused
without repeating the unchanged search. No accuracy scope is extended.


## Automatic physical fitting-domain selection

Run `python benchmarks/manta_periodic_ri_fitting_accuracy.py --domain-selection
--repetitions 1` twice in fresh single-threaded processes. Each process runs
RHF and UHF automatic selection followed by a fixed selected-domain control,
using the same original physical S/T, nuclear and exact-exchange inputs.
Controls prepare the selected reference before the timed fixed calculation;
that preparation is reported separately. Both paths require the same
certificate and bitwise identical states. The benchmark disables optional
metric search; the separate scientific acceptance includes nested metric
selection and four KS variants.

| Method | Automatic mean / s | Fixed selected mean / s |
|---|---:|---:|
| RHF | 0.762728 | 0.402091 |
| UHF | 0.718412 | 0.411383 |

Peak resident memory across the two fresh processes: 274825216 bytes.
The initial 8-vector source fails the 0.02 Eh J / 2 Eh same-density Coulomb
energy targets. Three domain trials select pair/auxiliary supports 6.75/9
bohr and a complete cube-two reciprocal union with 124 vectors. A fixed
canonical density family is used throughout. The measured overhead is
selection/certification work on a synthetic compact fixture; no molecular
or solid-state speedup, large-system scaling or total-energy accuracy is
claimed. Auxiliary-basis and remaining full-method selection are unfinished.

Scientific/benchmark source: `5a1c6e09cdfa62e41027fa3a04ebdb1cce78d9d6`.
Native SHA256: `f804e122683d326694e1fd9b92f5a14227079f16964ac2ab57c31c80758d8475`.
Integrated shared-XC runtime source: `616c6e6b07c345372d5a767886280e4f91c16c27`.
Integrated native SHA256: `7f8b62c8746a513429cf81ac6e475414404708680c89dd2f2c463bac69fca338`.
The integrated runtime passes the four affected KS controls, preserving
energies and certificates. Full independent HF evidence is reused because
the merged change affects the shared XC projection implementation only.
Job IDs, host identity, complete inputs, output hashes and failed diagnostic
attempts are retained outside the public checkout.


## Shared product auxiliary candidates

Run `.venv/bin/python benchmarks/manta_periodic_ri_fitting_accuracy.py
--auxiliary-generation --repetitions 1` in each of two fresh processes.
The compact HF fixture starts from a three-function seed and enriches it
with the shared primitive-product generator, at levels zero and one.
Generation is timed separately. Complete jobs include automatic reference
and physical-domain certification; compare the prepared object with the
identical plain basis. Both states must converge and agree bitwise in
energy, density, Fock/core/overlap matrices. This is a bookkeeping and source
cost comparison, not automatic selection among auxiliary candidates.

| Method | Level | Functions | Generation / s | Prepared job / s | Plain basis job / s |
|---|---:|---:|---:|---:|---:|
| RHF | 0 | 8 | 0.00001758 | 1.571956 | 1.534193 |
| RHF | 1 | 12 | 0.00001919 | 2.003866 | 2.029171 |
| UHF | 0 | 8 | 0.00001231 | 1.540903 | 1.533074 |
| UHF | 1 | 12 | 0.00001583 | 2.021241 | 2.010982 |

The table reports means over two fresh processes, one pair per level/method
in each process, prepared first. Maximum RSS is 275349504
bytes. Fixture setup is outside timing. Level-zero generator plan reserves
145248 bytes and 270336 modeled work units; level one reserves 163680 bytes
and 352256 work units. The typed job additionally reserves those full costs
and 32 kernel-work units per modeled scalar-work unit. These are conservative
reservations, not measured operation counts. No performance crossover or
speedup is claimed.

Scientific/benchmark source `6d090f59a8c0eaeb02580d095128ea7b39d99eeb`,
native SHA256 `0b8ca3c1d6570f5bd5a5f6b12ee92227c8e6968ddb28bfb21e2787649500c124`. Integrated XC scheduling source
`1c7feff4902367f89ac28d6c9635a9b083642ee7`, native SHA256
`1685d3fe5c065866f2894dc1f4a90413f104f958a05ea4c8b411ca336e0806c5`, preserves the four KS energies and
certificates in local queue calculations. Eight original complete states
pass, including independent HF finite Gaussian/libint M/T, NumPy factors,
Fock/SCF and full Fourier witnesses. Maximum independent Fock error is
1.12e-14 Eh and density error 2.85e-10. No matched local ORCA/CRYSTAL
custom-domain reference was available; selected-source controls qualify KS.

Eight-function HF candidates retain rank eight; twelve-function candidates
retain rank ten at cutoff 1e-12. The fixed-density full-Coulomb J/energy
bounds change from 0.00205124/0.481827 to 0.00205079/0.481811 Eh. Same-family
seed-basis bounds were 0.0153705/1.10547 Eh in the preceding increment.
Observed J errors are 6.30e-5 to 8.58e-5 Eh and Coulomb energy errors
4.62e-4 to 8.57e-4 Eh. These are synthetic component acceptance results;
finite physical supports, arithmetic/rank stability and relaxation still
prevent a whole-method energy claim. Auxiliary candidate selection remains
unfinished.


## Automatic auxiliary selection

Run `.venv/bin/python benchmarks/manta_periodic_ri_fitting_accuracy.py
--auxiliary-selection --repetitions 1` in two fresh processes. The same
compact HF fixtures now start from the seed and select level zero or one
using the actual full-Coulomb fitting certificate. Each pair runs automatic
selection first, then an explicit regenerated selected basis and reference,
with the selected physical domains and metric. Fixture setup is outside
job timing; explicit source/reference preparation is reported separately.
Both jobs must converge with bitwise matching energy, matrices and certificate.
The benchmark uses the original positive metric cutoff; scientific acceptance
also exercises a rank-zero starting cutoff and nested metric refinement.

| Method | Selected level | Automatic / s | Fixed / s | Fixed preparation / s |
|---|---:|---:|---:|---:|
| RHF | 0 | 2.562639 | 1.414002 | 0.000652 |
| RHF | 1 | 4.641450 | 2.054764 | 0.000964 |
| UHF | 0 | 2.411359 | 1.416947 | 0.000716 |
| UHF | 1 | 4.301350 | 1.954686 | 0.000654 |

Means cover two fresh processes with one pair per method and selected level
in each. Peak RSS is 274530304 bytes. The automatic case
includes sequence generation, reference/domain search, all rejected basis
trials and SCF on the retained source. The fixed case still verifies the
same full-Coulomb certificate. Search caps are conservative reservations,
not measured work counts. No performance crossover or speedup is claimed.

Scientific and benchmark source `3a48c2ab6436fdedda94d1fa7230add7805fe685`,
native SHA256 `e9a27c6ab3cd1bc1701bd81a2cdc34a640ef0c5a4e1945002196fc29e463493d`. Eight complete
source-pinned local queue states pass, including nested metric refinement.
Four HF states and fitted operators match the preceding source
`6d090f59a8c0eaeb02580d095128ea7b39d99eeb` bitwise, allowing reuse of its
independent SciPy SCF and Fourier cube 36/40 evidence. Independent finite
Gaussian/libint M/T and NumPy factors were checked again for the selected
owners. No matched local ORCA/CRYSTAL custom-domain reference was available;
selected-source controls and shared-functional regressions qualify KS.

J/energy bounds remain 0.00205124/0.481827 Eh at level zero and
0.00205079/0.481811 Eh at level one. The tighter energy target forces the
latter selection without changing the fixed density family. Observed HF
errors remain inside those bounds. This is fixed-density fitting/metric
acceptance; arithmetic, rank stability, density relaxation and whole-method
accuracy remain unfinished.


## Reference energy enclosures

Run `.venv/bin/python benchmarks/manta_periodic_ri_fitting_accuracy.py
--reference-energy --repetitions 1` in two fresh processes for the compact
reference-energy profiles. Add `--tight-profiles` to reproduce the more
expensive tight profiles. Automatic and explicit selected-source controls
must converge with bitwise equal energy, matrices and certificates.
Preparation of the fixed basis/reference is timed separately.

| Method | Profile | Samples | Automatic / s | Fixed / s | Fixed preparation / s |
|---|---|---:|---:|---:|---:|
| RHF | reference-energy | 2 | 4.579769 | 1.763817 | 0.000682 |
| RHF | tight | 1 | 173.081919 | 19.197212 | 0.002432 |
| UHF | reference-energy | 2 | 4.495756 | 1.720904 | 0.000743 |
| UHF | tight | 1 | 165.665429 | 21.085893 | 0.002356 |

Reference-energy rows are means from two fresh processes, automatic first.
Tight rows reuse one measured automatic/fixed pair per method from the
source-pinned scientific run; they are single observations, not a performance
distribution. Fixture setup is outside timing. Benchmark peak RSS is
277479424 bytes; the complete scientific/reference process peaks at
435535872 bytes. The latter includes independent validation work and
must not be interpreted as the method's isolated memory use. No speedup or
crossover claim follows from these small fixtures.

The reference-energy profile sets J/energy targets 0.001/0.05 Eh and allows
no reference refinement. The initial reference's entrywise energy bound
passes the fraction test where the former scalar maximum-J estimate fails.
The tighter profile requests 1e-6/1e-4 Eh and allows bounded shared reference,
domain and auxiliary refinement. Both still require the final combined
full-Coulomb bounds; component energy receipts are separate enclosures and
must not be added to the combined bound.

Scientific/benchmark source `aaa480c2cfa99208d61951f9c66412a628a1724c`,
native SHA256 `506d0a2972f58edf9d533c6aec3fd1593d36a5a58d94dc61c3b86a6f226331c1`. All eight complete
states pass, with bitwise fixed-source controls and independent finite
Gaussian/libint, NumPy factor, complete HF Fock/SciPy SCF and Fourier36/40
references. No matched local ORCA/CRYSTAL custom-domain reference was
available; qualified shared-functional tests and selected-source controls
cover KS. Source/job/output details remain in the private acceptance packet.

These are fixed-density Coulomb fitting/metric enclosures. Physical sources,
unified Fock and symmetry permissions retain their contracts. Arithmetic,
rank stability, density relaxation and full-method accuracy remain unfinished.


### Shared reference batches

Run `python benchmarks/manta_periodic_ri_fitting_accuracy.py --reference-batches
--repetitions 1 --tight-profiles` for paired complete automatic RHF/UHF
searches with batch capacities eight and one. The public benchmark alternates
execution order by repetition. The measurements below reuse the local
scientific acceptance pairs, alternating order by profile, including four
additional KS methods. They are one observation per state and do not establish
a general speedup distribution. Fixture preparation is outside timing.

| Method | Profile | Batch eight / s | Scalar / s | Ratio |
|---|---|---:|---:|---:|
| RHF | reference-energy | 3.181345 | 4.426414 | 1.39 |
| UHF | reference-energy | 3.119834 | 4.442231 | 1.42 |
| RKS | reference-energy | 3.463921 | 4.869762 | 1.41 |
| UKS | reference-energy | 3.817727 | 5.164777 | 1.35 |
| PBE0 | reference-energy | 3.503849 | 4.859226 | 1.39 |
| HSE06 | reference-energy | 3.069171 | 4.658990 | 1.52 |
| RHF | tight | 32.003415 | 160.801624 | 5.02 |
| UHF | tight | 32.041835 | 161.586276 | 5.04 |

All 16 calculations converge. Energy, density, Fock, core/overlap, physical
source and fitting-certificate identities agree bitwise within every pair.
The saved independent HF states, fitted J, reciprocal inventory and entrywise
bounds from the prior reference-energy qualification also agree bitwise.
Direct reference evaluations are shared; approximate BUPO partitions keep
scalar fallback. Conservative cold work reservations are unchanged except
for added batch setup and live workspace, so actual work savings do not
refund search budgets. These are fixed-density fitting bounds, not full
relaxed-energy guarantees.

Source `1e78b4651f0bddc44a42c01640506f2524a22ee6`, native SHA256 `d38d06a5c303e2b680c2e99c17262680edafe40ca30252f3022b75cf110f72ee`.
Peak RSS for the complete scientific process is 302039040 bytes;
it includes both control paths and witness comparisons. No matched local
ORCA/CRYSTAL custom-domain reference is available. Reused independent
Gaussian/libint, NumPy/SciPy and Fourier36/40 evidence remains tied to the
bitwise matched HF results; KS uses complete scalar controls and previously
qualified shared-functional evidence. Operational pins remain outside Git.

Integrated shared XC spin-field reuse is also checked on source `5326cccb9ac7e6e221c4fec2a645722b6f304f90`,
native SHA256 `86847691bfa78359f5a0de752c9b31c37af6a6cfb626fb01d25d18363d505a5e`. Four affected MANTA KS profiles each pass scalar
and batch-eight complete calculations, preserving the prior energies and
certificate identities. The 137 focused torus checks pass, with four existing
expected empty-spin GGA/MGGA potential-parity failures. These additional
calculations qualify integration; the timing table retains the original
paired scientific measurements. An incoming native periodic RKS READ-state fixture still fails the AO-image
density-domain closure guard; that separate driver is not qualified by
these MANTA runs.

For either final tight HF certificate, 18 Hermitian probes are processed in
three batches (8, 8, 2). Actual direct shell evaluations fall from 13504590
to 2250765, and Gaussian majorant-pair evaluations from 10098 to 1683. These
counts describe the final certificate, while paired timings above include
the complete automatic search and SCF. Source preparation and other RI work
explain why elapsed time does not scale exactly with the sixfold reduction.


### Component-directed reference refinement

Run `python benchmarks/manta_periodic_ri_fitting_accuracy.py --reference-components
--repetitions 1 --tight-profiles` to compare complete automatic RHF/UHF jobs
using component-directed and joint reference refinement. Both use reference
batches of eight. These measurements reuse the scientific acceptance pairs,
alternating execution order by profile and including four KS methods. They
are one observation per state, not a general speedup distribution. Fixture
preparation is outside timing. Initial references already pass for soft
profiles, so those pairs do not exercise refinement savings.

| Method | Profile | Selective / s | Joint / s | Ratio |
|---|---|---:|---:|---:|
| RHF | reference-energy | 3.306847 | 3.351435 | 1.01 |
| UHF | reference-energy | 3.203920 | 3.138878 | 0.98 |
| RKS | reference-energy | 3.631782 | 3.670426 | 1.01 |
| UKS | reference-energy | 3.822462 | 3.848185 | 1.01 |
| PBE0 | reference-energy | 3.757227 | 3.690991 | 0.98 |
| HSE06 | reference-energy | 3.737436 | 3.704369 | 0.99 |
| RHF | tight | 14.140936 | 36.378957 | 2.57 |
| UHF | tight | 13.322969 | 34.046472 | 2.56 |

RHF tight selects pair/real/reciprocal cubes 3/3/6 instead of 3/5/7. Final-certificate direct shell evaluations are 584397 versus 2250765. Combined J/energy bounds are 5.09053958e-07/2.37393657e-05 Eh against unchanged 1e-6/1e-4 targets.

UHF tight selects pair/real/reciprocal cubes 3/3/6 instead of 3/5/7. Final-certificate direct shell evaluations are 584397 versus 2250765. Combined J/energy bounds are 5.09053943e-07/2.37393649e-05 Eh against unchanged 1e-6/1e-4 targets.

All eight paired profiles converge with bitwise identical physical states,
energies and Fock source identities. Each selective state also passes a
complete fixed selected-source control, including certificate identity.
Different selected reference domains can have different certificates; each
must pass its own combined targets. Three additional default/real/reciprocal
searches and their fixed controls converge. The default path advances pair,
reciprocal and real controls in five steps; the twelve-step default is a
bounded allowance, not a convergence guarantee. Saturation and explicit
low-limit refusals remain tested.

The four HF states and fitted J arrays match the saved independent
Gaussian/libint, NumPy/SciPy and Fourier36/40 witnesses bitwise. The new
bounds enclose those saved full-Coulomb discrepancies. No matched local
ORCA/CRYSTAL custom-domain reference is available; KS uses selected-source
controls and previously qualified shared-functional evidence. Nonzero
finite approximation envelopes are independently checked at producer level;
these complete selection profiles use direct reference partitions.

Source `603c44ebc749ed859447d239975adc89a947dfe2`, native SHA256 `f5369bd574a27912eb89cc51b71b6eb6a2ff5d82e22cd01fcc76f20373980d44`.
The integrated Libxc spin-fraction patch uses dynamic-library SHA256
`194ffe4dc47c817fa70560f5706e27710054cd25e830d1ed2e872c51ef819cf2`. HF witnesses remain bitwise matched;
KS controls compare the integrated runtime, without assuming prior-library
bitwise identity.
The 30 complete calculations and reference comparisons use a peak RSS of
314081280 bytes. Resource reservations include extra component
arrays and contractions; rejected work is not refunded. This increment
preserves unified Fock, BIPOLE/CRYSTAL and symmetry permissions. It still
excludes arithmetic, rank stability, relaxation and full-method accuracy.
