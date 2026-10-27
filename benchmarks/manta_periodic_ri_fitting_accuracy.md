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

## Explicit automatic-reference search reservations

The two bounded reference-budget profiles retain their original scientific
inputs and targets. Their earlier default-cap refusals were insufficient
reservations, not a demonstrated lifetime-accounting defect. The ledger uses
maximum live memory and cumulative work, including rejected references and
metric replacement. Its complete producer peaks intentionally overestimate
some overlapping workspaces; they are not process RSS or executed-operation
measurements. No global default or producer formula changed.

Both profiles use the existing compact Be RHF fixture: three orbital and
three auxiliary s functions, a shifted `(2, 1, 1)` mesh, finite RI pair and
auxiliary supports of 3 and 12 bohr, reciprocal energy cutoff 6, nuclear
cutoff 4 bohr, independent exact exchange with the BvK Ewald correction,
100 maximum SCF iterations and energy tolerance `1e-12` Eh. The lattice,
normalized shell data and all selection controls remain in the shared
`reference_budget_job` fixture. Complete-job memory stays at 4 GiB.

| Profile | Explicit search cap change | Charged peak bytes | Charged scalar terms | Charged kernel terms |
|---|---|---:|---:|---:|
| joint-metric | Scalar `2e17`, kernel `1.5e18` | 830,846,576 | 142,975,645,354,679,624 | 1,188,278,268,440,494,904 |
| component | Memory 1,100 MiB | 1,080,741,426 | 30,090,297,715,179,275 | 319,849,056,473,565,346 |

The remaining search caps keep their defaults. In particular, joint-metric
keeps 1 GiB memory, while component keeps `1e17` scalar and `1e18` kernel
terms. Shell/primitive charges are respectively 4,899,132,000/4,899,442,464
for joint-metric and 8,522,623,566/8,522,624,679 for component. Original
joint-metric scalar and kernel reservations are insufficient; the component
peak exceeds 1 GiB by 6,999,602 bytes. The original scalar/memory refusals
remain explicit regression checks, with no computed QVF emitted. Simultaneous
exact charged caps admit each profile, and reducing each cap individually
by one refuses.

Joint-metric prepares four references with pair/real/reciprocal cube radii
`(0,1,1)`, `(1,2,2)`, `(2,3,3)`, `(3,4,4)` and pair cutoffs 3, 4.5, 6.75
and 10.125 bohr. It retains that reference while changing the metric cutoff
from `1e6` to `1e-12`, rank zero to three. Component uses one reference with
cube radii `(1,3,5)` and pair cutoff 3 bohr. Both use eta 0.4, near cube
zero, zero BUPO multipole tolerance and full-Coulomb targets J `0.02`,
Coulomb energy `1` Eh. Component additionally retains its separate explicit
L1 density limit `1e6` and component targets `1e8`; that policy remains
distinct from the canonical-overlap full-Coulomb family.

| Profile | Full-Coulomb J bound | Same-density Coulomb-energy bound (Eh) | Automatic wall (s) | Fixed selected wall (s) |
|---|---:|---:|---:|---:|
| joint-metric | 0.015156749106 | 0.942620976188 | 21.807 | 9.969 |
| component | 0.014920610066 | 0.870811608533 | 0.275 | 0.280 |

Each profile and its explicitly prepared selected-source control converges
at `-8.745245080287464` Eh with bitwise equal energy, density, Fock, overlap
and core Hamiltonian, equal physical source and certificate identities,
and valid QVF output. The component certificate separately reports
J `5.427465291735353e-5` and Coulomb energy `27.13732645868294` Eh for its
much larger explicitly requested density family; these are not additive
with the canonical full-Coulomb bounds.

The one-process, one-thread sample above includes validation/output capture;
fixed reference preparation adds 0.0049 s and 0.0005 s respectively. Process
high-water RSS was 627,261,440 bytes for this sample, not incremental profile
memory. These are actual costs for tiny controls, not a speedup claim.
Reproduce with `--reference-budgets --repetitions 1` and retain the emitted
source commit, native hash, selection receipts and timings.

The qualification used Python source `ccfc31f3` plus the recorded fixture
and benchmark patch (SHA256
`ba71ba9e643188ed96047d6a646ea11ab4719a3c815fc20e1bb38a0eb19ec7f8`),
with native source `2c6aeac7` and binary SHA256
`128b1892cc35b242da6c8bb1deee7770ba8abedc02ff35f3ccc5db3a7eab04fc`.
Their native source trees are identical. The shared Fock identity fix is
included in the tested Python source. The energy, density, Fock and RI
source identity also match the previously accepted independent-reference
RHF state bitwise, reusing Gaussian/Boys/Fourier/libint and SciPy SCF
checks. No matched ORCA/CRYSTAL custom-domain reference is available.

MANTA remains experimental. These certificates exclude integral-contraction
arithmetic, rank stability and density relaxation. They do not cover total
relaxed energy, the complete exchange/XC/one-electron/nuclear error budget,
bulk CRYSTAL parity, symmetry reduction, GPU/distributed execution or general
automatic accuracy. The broader production programme remains open.

## Outward fixed-density ledger composition

The consumer ledger now encloses the exact sum of its stored, disjoint
component bounds. Each positive binary64 addition advances toward positive
infinity. Zero terms and a single stored bound remain exact. This follows
the outward-arithmetic convention of scientific contract Sections 5.38-5.40
and Rump (2010), DOI `10.1017/S096249291000005X`; it introduces no new physical
approximation. For example, ordinary summation of `1` and three `2**-53`
terms returns `1`, below the exact stored sum. Exact rational comparisons
cover permutations of this case, widely separated magnitudes, subnormals,
zero and the largest finite single bound.

Negative, non-finite and missing requested energy bounds, invalid supplied
matrix bounds and non-finite outward sums now refuse ledger qualification.
The composed bound is absent, the relaxation delimiter is not evaluated,
and invalid energy values become `null` in the failed JSON-safe ledger.
No bad bound is clipped to zero or interpreted as an unrequested component.

A complete local queued one-thread qualification reproduced the accepted
physical states and original component certificates bitwise. Each profile
also matched a complete control with its optional certificates disabled.

| Profile | Energy (Eh) | Outward component bound (Eh) | Selected wall (s) | Control wall (s) |
|---|---:|---:|---:|---:|
| ordinary-rhf | -8.745245080287464 | 151491.2441428105 | 0.240 | 0.145 |
| compact-tight | 0.6667155727018081 | 3.0134188493711834e-10 | 2.329 | 2.308 |
| reference-component | -8.745245080287464 | 27.13732645868294 | 0.270 | 0.144 |

The first two configurations are unchanged `make_job` profiles in
`benchmarks/manta_release_controls.py`; run its `--help` for output/profile
selection. The third is the explicit `component` profile from
`reference_budget_job` above, with automatic RI-BUPO reference preparation
and its separate component-density policy. Its ledger sum contains only the
finite-coupling certificate; it does not silently add the overlapping
full-Coulomb certificate. The unchanged full-Coulomb receipt remains separate.
The ordinary profile's large bound is an admission control, not useful
material accuracy. The single component bound is unchanged; the two
multi-component sums exceed their exact stored sums by approximately
4.13e-11 and 6.32e-26 Eh respectively.

Tested Python/native source was `d57dd90d` plus the consumer/test patch;
native SHA256 was
`5aa2b2efec93c06989adb11e4a68a4e16c3ccf456ed338037cbc1435bf676acf`.
The native code did not change for this increment. The queue receipt reports
9.998 s total wall, 9.517 CPU seconds and 612 MiB process high-water RSS,
including six selected/control calculations and 29 focused checks (one
additional compact calculation). These costs are not a speedup claim.

Existing Gaussian/Boys/Fourier/libint and NumPy/SciPy acceptance is reused
through exact physical-state/source/certificate comparisons. Exact rational
summation independently checks the new composition arithmetic. No matched
ORCA/CRYSTAL custom-domain reference is available. All producer exclusions,
including integral construction and contraction arithmetic, remain explicit;
this change encloses only the sum of stored certificates. Relaxation remains
an uncertified estimate, and no whole-method energy guarantee, symmetry
permission or production-readiness claim follows.

## Full-Coulomb ledger replacement

The periodic RI RHF/UHF ledger now consumes the existing requested
full-Coulomb RI-BUPO certificate from scientific contract Sections 5.20-5.22.
It records one `coulomb` contribution and replaces the overlapping
finite-coupling contribution. It does not add their bounds or their separately
minimized subcomponents. The original receipts remain in `fitting`.

The existing native fitting receipt now identifies its weighted-entry/L1
family using the same byte convention as the kinetic and nuclear image
certificates. The ledger checks the selected Gamma RI source, canonical
family, literal full-zero-average Coulomb qualification and requested-target
success before composition. Physical certificates must have matching
canonical family identities. Failed checks give no composed bound; diagnostic
receipts without requested targets do not replace the finite-coupling term.
The native plan charges 512 extra peak bytes and `4096*(n_k*n_AO^2+8)` scalar
terms, with eight times that kernel allowance, when an entry envelope is
present. This bounds hashing and retained identity storage; it changes no
source, physical bound or numerical certificate identity.

| Profile | Energy (Eh) | Ledger component sum (Eh) | Selected wall (s) | Control wall (s) |
|---|---:|---:|---:|---:|
| RHF full only | -8.745245080287464 | 0.8708116085333024 | 0.330 | 0.141 |
| RHF full plus coupling | -8.745245080287464 | 0.8708116085333024 | 0.261 | 0.140 |
| RHF full plus physical certificates | -8.745245080287464 | 151492.11495441906 | 0.264 | 0.144 |
| UHF full only | -3.6764371477193523 | 0.8708116085156151 | 0.284 | 0.167 |

The compact reference inputs and physical RI supports are those of
`reference_selection_job(..., compact=True)` and the explicit `component`
reservation above. RHF uses Be and UHF uses Li, with the same existing compact
orbital/auxiliary fixture families and shifted two-point mesh. The full-plus-
physical case copies the ordinary RHF kinetic, nuclear and exact-exchange
accuracy controls into that job through `full_coulomb_ledger_job`; it does
not alter physical supports. The full-Coulomb J/energy targets stay at
`0.02` and `1` Eh. The separately requested coupling certificate retains its
explicit density limit `1e6` and targets `1e8`; its 27.13732645868294 Eh
bound is not added to the canonical full-Coulomb contribution. Large ordinary
physical-component bounds are admission controls, not useful material accuracy.

All eight selected/control calculations converge and agree bitwise in
energy, density, Fock, overlap, core Hamiltonian and physical source. Prior
accepted RHF/UHF states are also bitwise unchanged. RHF numerical full-Coulomb
certificate/source identities and J/energy bounds match the previous
component-profile receipt. The combined case's new native family identity
matches its kinetic and nuclear identities. Output QVF checks pass. An
impossible explicit full-Coulomb target refuses before computed output;
source/family/qualification/invalid-bound cases refuse ledger composition.
Existing complete-plan and exact/one-short search checks include the new
identity charge.

This sample used source `2c5f547d` plus the recorded native/consumer/test patch,
native SHA256
`2992c45a87670d142ae3d124b6777cb19eaac0c1cddf52b517941462b2e0fad1`.
The one-thread times include job/output work and are not a speedup claim.
The whole queued acceptance, including 50 focused checks and their complete
calculations, used 121.772 s wall, 121.541 CPU seconds and 574.969 MiB
process high-water RSS.
Reproduce the composed profile and refusal checks in
`tests/test_manta_periodic_ri_fitting_accuracy.py` with `-k full_coulomb_ledger`.
The existing `--reference-budgets` benchmark exports the full and coupling
receipts and the updated ledger for its component profile.

Independent Gaussian/Boys/Fourier/libint and NumPy/SciPy evidence is reused
through exact physical-state and certificate comparisons. No matched
ORCA/CRYSTAL custom-domain reference is available. Only the covered omitted
Hartree-domain exclusion is removed from a qualified full-Coulomb composition.
Integral construction and contraction arithmetic, rank/overlap stability,
basis and k-mesh errors, density relaxation and other component exclusions
remain. The RHF relaxation delimiter includes the J entry bound but remains
uncertified. MANTA remains experimental; this is neither a total relaxed-
energy guarantee nor a change to frozen release inputs or symmetry permission.

## Fitted-exchange ledger composition

The existing `exchange_admission_job` fixture now qualifies the fitted-K
contribution to the RHF/UHF component ledger. It uses the accepted shifted
(2,1,1) skew-cell Be RHF/Li UHF profiles: the three-s orbital/auxiliary
fixtures, 6 Eh reciprocal cutoff, 3 bohr AO-pair support, 4 bohr nuclear
support, alpha 0.6, fitting omega 0.65, 12 bohr auxiliary support and fixed
BvK correction. Reference AO/reciprocal cube radii remain 1/8, completion
radius 2, with raw K/energy targets 1000 and 1e6 Eh. The joint cases retain
the existing full-Coulomb J certificate. None of these loose targets is a
useful whole-method accuracy claim.

| Profile | SCF energy (Eh) | Composed bound (Eh) | Selected wall (s) | Control wall (s) |
| --- | ---: | ---: | ---: | ---: |
| rhf-False | -9.615420772487335 | 71997.87156514583 | 3.359 | 0.133 |
| rhf-True | -9.615420772487335 | 71998.74237675438 | 3.488 | 0.199 |
| uhf-False | -4.094056282195062 | 71997.87156370551 | 3.380 | 0.157 |
| uhf-True | -4.094056282195062 | 71998.74237531405 | 3.518 | 0.226 |

All eight selected/control calculations converge. Energy, density, Fock,
core Hamiltonian, overlap, coefficients and orbital energies agree bitwise
between each pair. Accepted earlier energies, spin densities, whole-mesh RI
source identities, fitted-K numerical certificates and bounds also remain
bitwise identical. The canonical-family identity agrees with the joint J
receipt. QVF output checks pass. Exact rational sums verify outward stored
bound aggregation. Mismatched source/family, overlapping exact/fitted K,
invalid bounds and unsupported scaling/reference claims refuse composition.

The native automatic attachment exports the original weighted-family identity;
raw probe certificates do not acquire a canonical claim. The combined ledger
counts one fitted-K energy contribution, without an extra restricted factor.
Its new resource charge is tested at exact caps and one below. HF, global-KS
and mixed-range native cap checks preserve the shared attachment API. The
existing impossible-target refusal and partial-rank/source checks pass.

Source is `f3cfd7ed` plus the recorded native/consumer/test patch; native
SHA256 is
`f0f25657fcb6e96508def413324ba58ea702a7da78939f1de6c3c52ffc0a5598`.
The complete successful queued acceptance used 81.375 s wall, 80.950 s CPU
and 581.891 MiB largest-process peak RSS on one thread. An initial harness
attempt cost 5.051 s wall and stopped after comparing the whole-method
identity with an RI identity; correcting the identity field required no
product-code change. This cost is not a speedup measurement.

Reproduce the composition and complete-state assertions with
`tests/test_manta_periodic_exchange_bounds.py -k
"fitted_exchange_ledger or exchange_admission_preserves_complete_and_stopped_hf"`.
The existing fitted-exchange benchmark and QVF receipts retain the component
reference evidence. Accepted independent libint/Gaussian-contour Fourier and
NumPy method errors (below 1.12e-13 Eh energy and 4.85e-15 Fock) are reused
through unchanged identities, states and numerical certificates. No matched
ORCA/CRYSTAL reference exists for these custom finite supports.

The reference is full Coulomb exchange with its reciprocal zero mode omitted.
Source/BvK corrections, response and assembly arithmetic, restricted embedding
arithmetic, source construction, rank/overlap perturbations, basis/BZ, XC and
relaxation remain excluded. KS/range ledger aggregation remains subsequent.
MANTA stays experimental; this is a fixed-density component sum, not a total
relaxed-energy guarantee, bulk CRYSTAL parity or production speedup.

## Global-hybrid and physical-range KS ledgers

The accepted PBE0, HSE06 and CAM-B3LYP RKS/UKS profiles now expose the
fixed-density component ledger through their existing KS provenance. The
unchanged fixtures use the shifted (2,1,1) skew cell, three-s orbital/auxiliary
basis, 6 Eh reciprocal cutoff, 3 bohr pair support, 4 bohr nuclear support,
fitting omega 0.65, 12 bohr auxiliary support and the existing BvK convention.
Their explicit quadrature, occupations and physical supports remain those
of `exchange_admission_job` and `range_admission_job`. Raw/aggregate exchange
targets remain 1000 matrix and 1e6 Eh energy; these deliberately loose controls
do not establish useful automatic accuracy. Joint cases retain the accepted
full-Coulomb J target and canonical family.

| Profile | Joint J | SCF energy (Eh) | Fixed-density component sum (Eh) | Selected / control wall (s) |
| --- | --- | ---: | ---: | ---: |
| pbe0-rks | False | -9.700417784506898 | 17999.4852320514 | 3.291 / 0.128 |
| pbe0-rks | True | -9.700417784506898 | 18000.7558136691 | 3.281 / 0.184 |
| pbe0-uks | False | -4.145245061675722 | 17999.4852316913 | 3.207 / 0.160 |
| pbe0-uks | True | -4.145245061675722 | 18000.7558133090 | 3.472 / 0.203 |
| hse06-rks | False | -9.880935008763963 | 17999.4938755731 | 3.232 / 0.147 |
| hse06-rks | True | -9.880935008763963 | 18000.7644571909 | 3.419 / 0.242 |
| hse06-uks | False | -4.284658734208046 | 17999.4938752131 | 3.519 / 0.200 |
| hse06-uks | True | -4.284658734208046 | 18000.7644568308 | 3.602 / 0.234 |
| cam-b3lyp-rks | False | -9.672681162936970 | 79917.7151197828 | 7.110 / 0.208 |
| cam-b3lyp-rks | True | -9.672681162936970 | 79918.9857014005 | 7.332 / 0.234 |
| cam-b3lyp-uks | False | -4.143618577079909 | 79917.7151181840 | 7.086 / 0.216 |
| cam-b3lyp-uks | True | -4.143618577079909 | 79918.9856998017 | 6.791 / 0.259 |
| pbe0-rks (a=-0.2, stopped) | False | -7.851130058273347 | 14399.5881856411 | 3.365 / 0.110 |
| pbe0-uks (a=-0.2, stopped) | False | -2.801531634124281 | 14399.5881853531 | 3.431 / 0.099 |

Twelve selected/control pairs converge (24 calculations); two additional
negative-coefficient pairs stop deliberately after one iteration (four
calculations). Every pair agrees bitwise in energy, iterations, density,
Fock, H, S, coefficients and orbital energies. Accepted earlier energies,
spin densities, source identities, numerical exchange certificates and
bounds are unchanged. Global hybrids use one outward absolute coefficient
multiplication; signed range methods use one native aggregate contribution.
Children are not added twice. Joint J/K canonical-family identities match.
Output QVF checks and exact-rational stored-sum comparisons pass.

The implementation uses the existing native receipt; no native numerical
algorithm, physical support, source, Fock/eigensolver or symmetry policy
changes. Native SHA256 is
`145fe8fdc735483d9980c147c83cc7e095400451e0361e55010ca76a1618fc2a`,
built from source `732ee398`; the tested Python/test patch is recorded with
the queued inputs. This identifies the actual local runtime, not a deployed
release runtime. Native global/range capacity refusals and strict impossible
targets retain their original meaning. Missing physical components, source,
kernel, range, signed coefficient, canonical family, invalid/underbounded
aggregate or overlapping exchange claims refuse the ledger.

Independent accepted libint/Gaussian Fourier/NumPy method evidence is reused
through exact state/source/certificate comparisons: maximum discrepancies in
these receipts are below 5.78e-14 Eh energy and 2.84e-15 Fock. Shared libxc is
not independently validated. No matched ORCA/CRYSTAL reference exists for
these custom finite supports. The prior continuous-grid release acceptance
remains separate from these explicit-quadrature profiles.

Reproduce the public profiles with `exchange_admission_job` and
`range_admission_job` in `tests/test_manta_periodic_exchange_bounds.py`.
Focused checks select `ks_ledger`, `exchange_ks_admission` and
`range_admission`; the benchmark helpers retain the original independent
reference comparisons. These results certify only requested stored-input
fixed-density components. XC/grid errors, source construction, source/BvK
corrections, executed response/assembly and restricted embedding arithmetic,
rank/overlap, basis/BZ and relaxation remain excluded. No KS relaxation
estimate is supplied. MANTA remains experimental, with no total relaxed-energy,
bulk CRYSTAL parity or general speedup claim.

The main local queued acceptance passed 111 focused checks in addition to
the complete calculations. Total queue cost was 128.083 s wall, 126.605 CPU
seconds and 586.969 MiB largest-process peak RSS on one thread. Times include
output and validation work; they are not a speedup measurement.

A final supplement verifies three-component mixed-range RKS/UKS ledgers,
with four additional converged selected/control calculations and the accepted
independent source/contraction oracle. Both prior physical energies and
source/certificate identities remain identical. Sixty supplemental checks
include native exact-cap ledger export and exact-rational extreme scaling:
zero, subnormal products, finite maximum coefficients and overflow refusal.
The supplement cost 60.549 s wall, 60.106 CPU seconds and 573.344 MiB
largest-process peak RSS. These checks reuse the primary profile evidence;
no unchanged full-profile calculations were repeated for the scalar edge fix.

## Conditional overlap-image rank admission

The #625 increment adds an opt-in guard driven by the existing image-overlap
producer, shared canonical projection/Jacobi and outward spectral certificate.
The development native binary on source baseline `edf68abf` plus this increment
has SHA256
`63ba1f512cce88f89e256bde8b10a655d8c0f4c5fcde8f00f0142a68f7c17fef`.
This identifies tested development code, not a deployed runtime or release.

Twelve complete local vq calculations cover five explicit selected/control
pairs and one automatic-support/manual pair. Every calculation converged;
energy, iteration count and evaluated density/Fock/H/S are identical within
each explicit pair. The explicit cases use 10-bohr AO pair supports,
12.1-bohr auxiliary supports and completion cube 3. These are the existing
custom-basis image fixtures with their original physical kernels, charge,
spin, mesh, quadrature and SCF controls. No tolerance was loosened.

| Profile | Energy (Eh) | Iterations | Image/projection norm bound | Minimum rank radius | Selected wall (s) |
|---|---:|---:|---:|---:|---:|
| RHF Be | -8.745246181091739 | 9 | 7.56e-13 | 8.97e-3 | 0.586 |
| UHF H | 0.09295229253660428 | 10 | 6.21e-16 | 1.91e-2 | 0.294 |
| RKS Be | -9.680726572313460 | 9 | 7.56e-13 | 8.97e-3 | 0.497 |
| PBE0 Be | -9.479322790838660 | 9 | 7.56e-13 | 8.97e-3 | 0.548 |
| HSE06 Li | -4.178171904827243 | 10 | 7.56e-13 | 8.97e-3 | 0.731 |
| Automatic ordinary RHF | -8.745245080287464 | 9 | 1.15e-4 | 8.97e-3 | included below |

The primary vq command took 10.058 seconds including controls, reference
construction and focused checks. It ended with one test-harness assertion
failure after all calculations passed: the raw support selector returns an
unqualified result at its limit, whereas the test expected an exception.
The corrected test preserves this existing API contract; the calculation
outputs are retained and reused rather than repeated.

The RHF independent Gaussian quadrature/libint/Fourier/NumPy construction
has maximum overlap error 2.78e-17, Fock error 4.75e-15, density error 1.87e-10
and energy error 8.80e-14 Eh. Its extended-image overlap has the same retained
rank. The observed image difference, 2.04e-26, lies below the 7.56e-13 entry
bound; this comparison retains a 3e-12 arithmetic allowance and is not a
construction-roundoff proof. Decimal two-by-two spectra and independent
perturbed eigensolves separately exercise the strict radius, adaptive cutoff,
clusters, wrong rank, projection displacement and boundary refusal. A
near-cutoff support request drives physical AO support growth; a fixed-source
request refuses before SCF. Resource checks include exact and one-short caps.

There is no matched ORCA/CRYSTAL reference for these custom finite-domain
fixtures. Existing component references and unaffected state evidence remain
valid. The guard does not cover integral construction, occupied-space or
density-family changes, total relaxed energy, bulk CRYSTAL parity or speedup.
The current-grid/old-grid and automatic/explicit distinctions remain unchanged.

Final source qualification at `8fb49acb` preserves the latest shared Fock,
BIPOLE and eigensolver changes. A cached rebuild after rebase produced the
same native SHA256 above. Three final-source RHF/PBE0/HSE06 calculations also
match the original energies, densities and rank certificates exactly. Their
outputs were reused after correcting a reporting-only spin-array shape
assertion and satisfying the repository's stale-core timestamp check; neither
condition changed the scientific implementation.

The final vq acceptance passes all nine outstanding strict/cap checks and
verifies the reused archives and source/native hashes. Its measured acceptance
work costs 1.412 seconds wall, 1.365 seconds CPU and 612.75 MiB peak process
RSS; the queue's command wall time, including startup, is 6.030 seconds.
The two preceding reporting/admission attempts cost 3.016 and 6.036 seconds
of command wall time. These costs are reported without a speedup claim.
The near-cutoff request uses an explicit absolute cutoff of 0.0089: supports
3.0 and 4.5 bohr have perturbation bound 1.145e-4 versus radius 6.565e-5 and
refuse; 6.75 bohr has bound 1.711e-8 versus radius 6.565e-5 and passes.
A one-attempt search returns unqualified. The public job layer retains its
existing refusal of an unqualified selection. Earlier passing spectral and
physical comparisons are reused; 52 documentation/changelog checks pass.
