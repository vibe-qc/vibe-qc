# Stored metric spectral enclosure benchmark

`manta_metric_spectrum.py` measures the shared certificate on synthetic complex
Hermitian matrices with two, eight and 32 functions. The eigensolve occurs
outside the timed region. Three consecutive observations measure the residual,
orthogonality, eigenvalue and rank enclosure; this is not a complete QC timing
or an asymptotic performance qualification. Work scales cubically, and every
periodic RI owner now pays this audit cost once. No factor changes are made.

| Dimension | Wall seconds, three observations | Eigenvalue bound | Rank radius |
| --- | --- | --- | --- |
| 2 | 4.91599e-06, 2.37499e-06, 1.00001e-06 | 1.22125e-15 | 1.01874 |
| 8 | 2.7333e-05, 2.63751e-05, 2.60411e-05 | 1.03251e-13 | 0.556161 |
| 32 | 0.00162208, 0.00161717, 0.00165071 | 1.5038e-12 | 12.578 |

Complete periodic RI-BUPO checks use the existing reference-energy profiles,
with paired fixed selected-source controls. All sixteen calculations converge.
The table records one automatic calculation per profile; it is not a timing
distribution or a before/after speed claim.

| Method | Profile | Wall seconds | Iterations | Energy / Eh |
| --- | --- | --- | --- | --- |
| rhf | reference-energy | 2.913710 | 10 | -7.142711309871 |
| uhf | reference-energy | 2.825746 | 10 | -2.600317249155 |
| rks | reference-energy | 3.141376 | 9 | -9.281975051461 |
| uks | reference-energy | 3.189987 | 11 | -3.908126100059 |
| pbe0 | reference-energy | 3.077823 | 9 | -8.774718842496 |
| hse06 | reference-energy | 3.080846 | 9 | -8.956216604105 |
| rhf | tight | 11.832726 | 10 | -7.142707212914 |
| uhf | tight | 12.016111 | 10 | -2.600315310703 |

Scientific source `a070f8860811789fd07fcc48f814ff2ee585fec2`. Across 8 exported metric
receipts, all spectrum and rank flags qualify. The largest eigenvalue enclosure
is 5.07371922e-14; the smallest strict rank radius
is 1.32231251e-09. These are absolute metric units,
not SCF energy tolerances. Tight HF combined J/energy targets remain 1e-6/1e-4 Eh.

Fixed controls reproduce energies, densities, Fock matrices, physical source
identities and full-Coulomb certificates bitwise. Four HF states, fitted J and
reciprocal inventories also match accepted independent Gaussian/libint,
NumPy/SciPy and Fourier36/40 evidence exactly, allowing those references to be
reused. No matched local ORCA/CRYSTAL custom-domain reference is available.
KS uses fixed-source controls and retains that external-comparison limitation.

Independent Decimal eigenvalue witnesses and arbitrary Hermitian perturbations
check the spectral proof. Cutoff crossings, deficient eigenvectors, underflow,
overflow, all/full retained spaces and resource boundaries are covered.
The bounds apply to the exact stored matrix, excluding integral construction,
factor/inverse or Fock arithmetic, overlap rank and density relaxation. They
are diagnostic; automatic rank-aware selection and production-scale cost
qualification remain unfinished. See scientific contract Section 5.31.


## Rank-driven fitting admission

`manta_metric_spectrum.py --rank-admission --repetitions 1` now exercises
complete periodic searches whose initial J/energy bounds pass but whose
initial stored metric rank is unresolved. The optional 1e-10 perturbation
request forces selection to continue through the existing shared producer
APIs. Each search has a complete fixed selected-source control, and the
provenance records every rejected rank request. These are one-observation
fixture timings, not a production-size cost or speedup claim.

| Method | Search advanced by rank requirement | Trials | Wall seconds | Selected radius | Energy / Eh |
| --- | --- | --- | --- | --- | --- |
| rhf | fixed | 2 | 0.406294 | 0.00072091729 | -8.745245080287 |
| uhf | fixed | 2 | 0.444001 | 0.00072091729 | -3.676437147719 |
| rhf | reference | 2 | 0.404230 | 0.00072091729 | -8.745245080287 |
| rhf | domain | 2 | 0.147560 | 2.60474476e-06 | -7.233807671871 |
| rhf | auxiliary | 2 | 0.677852 | 0.000445744506 | -8.728381639757 |

Scientific source `fa746e2f2791aaa42f4a546c21fdea3ad1dafbc3`. All five searches and their controls
converge with bitwise matching energy, density, Fock, physical source and
full-Coulomb certificate. Reference readiness is allowed to pass while the
provisional rank fails, so the reference is not enlarged to repair a metric
rank problem. Four searches with an unattainable request refuse before SCF.

Eight existing soft/tight RI-BUPO profiles also converge with the new 1e-10
rank request and fixed selected-source controls, bringing the total to 26
complete calculations. Four HF states, fitted J and reciprocal inventories
match accepted independent Gaussian/libint, NumPy/SciPy and Fourier36/40
references exactly. All eight energies equal the prior accepted values.
No matched local ORCA/CRYSTAL custom-domain reference is available; KS retains
the fixed-source and shared-functional comparison limitation.

Strict equality with the certified radius is refused. None preserves the
old acceptance policy, while zero requires a resolved rank. The requested
margin is not a measured integral error or a total-energy tolerance. This
extends the earlier diagnostic with optional automatic admission; it does
not complete integral/factor/Fock arithmetic, all-q rank qualification,
relaxed accuracy or production-scale validation. See contract Section 5.32.


## Stored factor inverse error bounds

Scientific source `f7e3c00b47adf1cb34bc88264b9f8d70e796ad03`. Sixteen complete local periodic
calculations (eight profiles and fixed selected-source controls) converge.
All eight exported selected metric inverse certificates qualify. Physical
energies, densities, Fock matrices, sources and full-Coulomb certificate
identities remain bitwise equal to controls. Four HF states and fitted J
arrays also match accepted independent Gaussian/libint, NumPy/SciPy and
Fourier references exactly. No matched ORCA/CRYSTAL custom-domain reference
is available; KS retains the fixed-control comparison limitation.

| Method | Profile | Wall seconds | Inverse error bound |
| --- | --- | --- | --- |
| rhf | reference-energy | 4.725194 | 59624.185 |
| uhf | reference-energy | 4.507778 | 59624.185 |
| rks | reference-energy | 5.129166 | 94.7213132 |
| uks | reference-energy | 4.944817 | 94.7213132 |
| pbe0 | reference-energy | 5.042157 | 94.7213132 |
| hse06 | reference-energy | 4.457929 | 94.7213132 |
| rhf | tight | 18.339180 | 86917.294 |
| uhf | tight | 20.869079 | 86917.294 |

These are absolute inverse-operator units, not Eh. Bounds grow for nearly
dependent auxiliary functions; they are not fitted J or SCF energy error
bounds. Integral construction and coupling/Fock contraction roundoff remain
excluded. The full-Coulomb certificate still has `roundoff_included=false`.

The combined synthetic spectrum/factor audit uses `--inverse`; eigensolves
are outside the timed region. Three observations per dimension follow.
These small cases qualify accounting, not production scaling.

| Dimension | Wall seconds, three observations | Inverse error bound |
| --- | --- | --- |
| 2 | 1.0209042e-05, 4.6669738e-06, 1.7910497e-06 | 4.4703913e-15 |
| 8 | 6.9124973e-05, 6.5541943e-05, 6.5290951e-05 | 3.83574427e-13 |
| 32 | 0.0048388341, 0.004341042, 0.004856334 | 6.0846042e-14 |

Independent 100-digit Decimal two-by-two inverse witnesses cover actual
stored complex entries, perturbed vectors/factors, full/truncated/empty
spaces and scales 1e-100 through 1e100. Further checks cover clusters,
Gamma/non-Gamma owners, unresolved ranks, overflow and all admission caps.
See scientific contract Section 5.33 for the finite divided-difference
proof; no first-order approximation is treated as a finite error bound.


## Coupling-projected metric action admission

Scientific source `79564c6b1be9ad6ccfd67b40de5acc81b8395ce9`. Twenty-eight complete local
calculations and four deliberate pre-SCF refusals pass. Eight soft/tight
HF/KS profiles request the same numerical metric-action targets as their
full-Coulomb J/energy targets. Fixed selected-source controls reproduce
energy, density, Fock matrices and physical identities bitwise. Four HF
states, fitted J and reciprocal inventories also match accepted independent
Gaussian/libint, NumPy/SciPy and Fourier references exactly. No matched
ORCA/CRYSTAL custom-domain reference is available; KS retains its fixed-source
comparison limitation.

| Method | Profile | Wall seconds | Metric J bound | Metric energy bound / Eh |
| --- | --- | --- | --- | --- |
| rhf | reference-energy | 3.416713 | 5.36318505e-10 | 1.58130338e-07 |
| uhf | reference-energy | 3.375801 | 5.36318505e-10 | 1.58130338e-07 |
| rks | reference-energy | 3.821058 | 1.34079592e-10 | 3.95325816e-08 |
| uks | reference-energy | 3.779387 | 1.34079592e-10 | 3.95325816e-08 |
| pbe0 | reference-energy | 3.597991 | 1.34079592e-10 | 3.95325816e-08 |
| hse06 | reference-energy | 3.645003 | 1.34079592e-10 | 3.95325816e-08 |
| rhf | tight | 13.851030 | 5.36318524e-10 | 1.5813034e-07 |
| uhf | tight | 14.062252 | 5.36318523e-10 | 1.5813034e-07 |

Soft full-Coulomb and metric targets are 1e-3/5e-2; tighter HF targets
are 1e-6/1e-4. These are separate conditions, not additive contributions to
a verified total SCF error. Actual integral and Fock-contraction roundoff
remain excluded. The projected bound uses the actual coupling directions;
the preceding inverse-operator values are not J/energy tolerances.

Five further cases reject their initial unresolved metric-action bound,
advance the bounded metric/domain/auxiliary search and retain the actual
passing source. A sixth case exercises nonzero ordered multipole coupling.
All six match their fixed-source controls bitwise.

| Method | Search | Nonzero ordered multipoles | Trials | Metric J bound | Metric energy bound / Eh |
| --- | --- | --- | --- | --- | --- |
| rhf | fixed | False | 2 | 5.23748402e-13 | 1.54424146e-10 |
| uhf | fixed | False | 2 | 5.23748402e-13 | 1.54424146e-10 |
| rhf | reference | False | 2 | 5.23748402e-13 | 1.54424146e-10 |
| rhf | domain | False | 2 | 5.23748402e-13 | 1.54424146e-10 |
| rhf | auxiliary | False | 2 | 4.40844507e-06 | 0.00107960902 |
| rhf | fixed | True | 2 | 5.23748402e-13 | 1.54424146e-10 |

Those six transition cases use loose 1e8 fitting and metric-action targets
to isolate qualification from the existing physical-domain choice. Four
1e-100 metric-action requests refuse before SCF. Independent 100-digit
Decimal tests cover full/truncated/empty spectra, complex coupling columns,
arbitrary factor errors, J and energy envelopes. An ill-conditioned fixture
shows a projected bound more than 1,000 times tighter than the worst-direction
inverse estimate. All public and aggregate resource caps are exercised.

Reproduce transition cases with `manta_metric_spectrum.py --metric-action`.
Timings are single observations, not a production scaling claim. Explicit
screened/ordered-multipole tiles are supported; compressed/global moment
and pair-tree actions retain the new policy guard. Contract Section 5.34
gives the finite residual proof and the remaining arithmetic exclusions.


## Joint total and component density limits

Scientific source `093642f86a2f98637adc04b5a30c133614222e98` uses the joint density-family
dual bound in contract Section 5.34. The same stored RI factors and coupling
producer are retained. Six complete local calculations (RHF, UHF, RKS and
their selected-source controls) pass with source/native pins before and
after execution. Energy and state match controls bitwise; HF states and J
reuse accepted independent Gaussian/libint, NumPy/SciPy and Fourier evidence.
There is no matched ORCA/CRYSTAL custom-domain reference. RKS retains its
fixed-source comparison limitation. Earlier focused checks are reused.

| Method | Previous metric J bound | Joint metric J bound | Previous metric energy / Eh | Joint metric energy / Eh |
| --- | --- | --- | --- | --- |
| rhf | 5.36318505e-10 | 5.36318505e-10 | 1.58130338e-07 | 1.58130338e-07 |
| uhf | 5.36318505e-10 | 5.36318505e-10 | 1.58130338e-07 | 1.58130338e-07 |
| rks | 1.34079592e-10 | 1.34079592e-10 | 3.95325816e-08 | 3.95325816e-08 |

The synthetic uneven-density fixture reduces the J bound by
14.6377 times and the energy bound by 214.262
times relative to the previous separate-limit expression. The latter
comparison is a float64 diagnostic; independent 100-digit primal allocation
checks verify the new enclosure. Reproduce with
`manta_metric_spectrum.py --density-family`. These are bound improvements,
not a production scaling claim.

The supported explicit-coupling configuration and arithmetic exclusions
are unchanged. The optional numerical metric conditions remain separate
from full-Coulomb fitting differences. Compressed action arithmetic, integral
and actual Fock contraction roundoff and relaxed SCF energy remain open.


## Compressed metric-action admission

Scientific source `d0e485994ebb37e2eb382a544f696c07f9fedc35` qualifies ordered compressed hierarchy and
global-panel coupling norms through the shared producer and inverse action.
Eighteen complete calculations (six RHF/UHF/UKS profiles, fixed-source and
materialized controls) and six pre-SCF strict-target refusals pass through
the local queue with source/native pins before and after execution. Enabling
the metric condition leaves energy, density, Fock, overlap, core Hamiltonian
and source identity bitwise unchanged against the fixed-source control.

| Method | Action | Metric J bound | Metric energy bound / Eh | Compressed nodes | Wall time / s |
| --- | --- | --- | --- | --- | --- |
| rhf | ordered | 1.7985871e-07 | 4.5673436e-05 | 1252 | 0.2437 |
| rhf | global | 1.7985871e-07 | 4.5673436e-05 | 1252 | 0.1749 |
| uhf | ordered | 1.7985871e-07 | 4.5673436e-05 | 1252 | 0.1853 |
| uhf | global | 1.7985871e-07 | 4.5673436e-05 | 1252 | 0.1867 |
| uks | ordered | 5.399447e-07 | 0.00013987621 | 204 | 0.2024 |
| uks | global | 5.399447e-07 | 0.00013987621 | 204 | 0.2019 |

The global HF and UKS cases record 36 and 16 pair-panel reuses respectively.
The largest materialized-control energy difference is 5.33e-15 Eh and the
largest state-matrix difference is 2.85e-14. These numerical controls exercise
the same physical partition through different contraction routes; they do
not bound actual contraction roundoff. Independent 100-digit Decimal
metric-action checks, complex/shifted/angular materialized-column witnesses
and all five resource boundaries pass. The focused and shared checks record
331 successful executions after repairing an incoming changelog-format error.

Reproduce with `manta_metric_spectrum.py --compressed-action --repetitions 1`.
The complete fixtures retain loose fitting targets of 1e8 and numerical
metric targets of 1e30; 1e-100 metric targets refuse. Reported bounds are
measured admission results, not achieved total SCF accuracy. Norm-only
propagation can be much looser than the explicit projected-column bound.
These small timings are single observations, not a scaling claim.

Existing inverse/spectrum, shared Gaussian/libint and Fourier evidence is
reused for unchanged components. No matched ORCA/CRYSTAL custom-domain
reference is available; no previous HF state is transferred to the changed
compressed source. Pair-tree metric bounds, integral construction and actual
compressed/Fock contraction roundoff, relaxed SCF energy and the original
remaining capability matrix remain open. Contract Section 5.35 defines the
exact stored-moment comparison and its exclusions. Unified Fock,
BIPOLE/CRYSTAL conventions and symmetry permissions are unchanged.

## Global pair-tree metric-action admission

Scientific source `27d9c24d1193e0ef3679aa2bbf4bbd7e15b54132` extends the
stored-graph norm audit to global ordered pair trees. Native SHA256 is
`3a4bb2d145865b130aff78d6f87baabdc18c3ca49b721d5cc51d903baadffeac`.
The local `vq` run completed six converged calculations and three pre-SCF
strict-target refusals, with source and native pins checked before and after.
RHF, UHF and UKS admission each retains bitwise energy, density, Fock, overlap,
core Hamiltonian and source identity against its metric-policy-disabled control.

| Method | Energy / Eh | Iterations | Metric J bound | Metric energy bound / Eh | Compressed nodes | Wall time / s |
| --- | --- | --- | --- | --- | --- | --- |
| rhf | -8.745244677860905 | 9 | 1.79858709e-07 | 4.56734364e-05 | 760 | 0.2610 |
| uhf | -3.676437098543107 | 13 | 1.79858709e-07 | 4.56734364e-05 | 760 | 0.2125 |
| uks | -4.133637051752787 | 11 | 5.39944701e-07 | 1.39876214e-04 | 120 | 0.2171 |

Each metric audit returns 18 column norms. Independent materialized coupling
references use a separate NumPy metric solve and contractions, finite-source
one-electron/exchange references and shared libxc for KS. The maximum energy
difference is 9.86e-14 Eh; Fock entries differ by at most 4.52e-15. Evaluated
unified Fock energies agree within 1e-12 Eh. No matched ORCA/CRYSTAL oracle is
available for these custom finite supports. Shared integral/Fourier and
verified inverse evidence is reused for unchanged components.

The focused and shared suites pass 360 checks, including 80-digit Decimal
norms of materialized columns, complex and shifted frames, signed angular
panels, near-only and compressed paths, full/truncated/zero metric ranks,
strict targets and all five resource caps. The comparison graph contains the
stored panels, translations, phases, interactions and near/residual values;
it is not a proof of the rounded exported tensor or executed action arithmetic.
The producer retains its masks and Gamma conjugate completion. Production
auditing uses bounded vectors and the existing tree workspace, with no dense
auxiliary-by-AO-pair coupling tensor.

Reproduce with `manta_metric_spectrum.py --pair-tree-action --repetitions 1`.
The queued process took 1.8902 s and peaked at 278233088 bytes RSS, including
all calculations and reference checks. Table times include admitted setup and
SCF for each selected case; they are single observations, not a speedup or
scaling claim. Targets of 1e30 qualify the representation transition;
1e-100 targets refuse. Integral construction, actual compressed/Fock
roundoff, relaxed SCF energy and useful whole-method automatic accuracy remain
open. This extension does not enlarge the frozen 0.18 release controls or
grant new BIPOLE/CRYSTAL parity or symmetry permissions.

## Executed paired metric arithmetic

Scientific source `3d2e1db27a832f693101d68bebf96bcff0f688c1` adds a separate
actual-input interval audit of the stored-factor projection, reconstruction
and Coulomb squared-norm energy. Native SHA256 is
`c6f5d02a20dc0e941f2072712365f2e072b4e9aaa7f7c8c3fae880333873755e`.
A local `vq` run passes six complete calculations and three strict-target
refusals, with source/native pins checked before and after. The same pair-tree
fixtures and independent references as above are reused. Energy, density,
Fock, overlap, core Hamiltonian and source identity remain bitwise equal to
the metric-policy-disabled controls; unified evaluated Fock energies agree.

| Method | Projection bound | Coefficient bound | Metric energy roundoff bound / Eh | Audited wall / s | Control wall / s |
| --- | --- | --- | --- | --- | --- |
| rhf | 1.43218770e-14 | 4.65405492e-13 | 2.13162821e-14 | 0.2402 | 0.1867 |
| uhf | 8.93729535e-15 | 2.85993451e-13 | 1.15463195e-14 | 0.2020 | 0.2011 |
| uks | 5.35682609e-15 | 1.95399252e-13 | 9.76996262e-15 | 0.2064 | 0.2062 |

These values bound the executed metric substep relative to exact arithmetic
on its stored factor and actual forward-action output. They are neither a
whole-Coulomb roundoff bound nor the pre-SCF inverse-action target. The latter
still reports the fixed-density-family bounds in the preceding table. The
coefficient error must still propagate through the coupling adjoint to bound
J. Full-Coulomb receipts retain `roundoff_included=false`.

The maximum independent energy difference is 9.86e-14 Eh; the maximum Fock
entry difference is 4.52e-15. No suitable matched ORCA/CRYSTAL oracle exists
for these custom finite supports. The independent materialized coupling,
separate NumPy metric/contractions and finite-source one-electron/exchange
checks remain the reference, with shared libxc for KS. Existing integral and
Fourier acceptance is reused, not claimed as a new external comparison.

The focused suites pass 180 checks; shared source, Fock, admission and docs
checks pass 173, with the sole unrelated changelog-format failure corrected
and its ten checks rerun successfully. Exact rational complex witnesses test
projection, reconstruction and energy enclosures, including cancellation,
subnormals, zero rank, deliberately inaccurate numerical witnesses,
nonfinite/overflow refusal and exact/one-short resource budgets. Retained-mode
and fused J/K paths do not claim an evaluated arithmetic audit.

Reproduce with `manta_metric_spectrum.py --pair-tree-action --repetitions 1`.
The queued payload took 1.7604 s and peaked at 279527424 bytes RSS, including
all calculations and references. Table times include setup and SCF; the
control disables the entire metric policy, so their difference does not
isolate the new arithmetic audit. These single observations establish no
speedup or scaling claim. Conservative plans reserve maximum-rank audit
resources before payload execution.

Contract Section 5.36 defines this bounded component of work package #624.
Coupling forward/adjoint, weighted-density, final J/K/Fock and integral
construction arithmetic remain open, along with useful automatic accuracy.
The full MANTA goal and #624 remain unfinished. Shared Fock/eigensolver,
BIPOLE/CRYSTAL and symmetry authority and the frozen 0.18 scope are unchanged.

## Metric arithmetic contribution to J

Scientific source `8555f080dad87f17ca12e8bb88b9e29a5de05e17` propagates the
executed coefficient error through the exact stored coupling adjoint, using
its setup-time maximum column norm. Native SHA256 is
`276c85a9cc00064b6af8e87658267d4951cd4181c65da7b9c32785eff0624401`.
Local `vq` acceptance passes six complete RHF/UHF/UKS calculations and three
strict refusals with all source/native/output pins verified. The same fixtures,
bitwise controls, independent references and external-reference limitation
as the preceding increment apply. Maximum independent energy and Fock entry
differences remain 9.86e-14 Eh and 4.52e-15 respectively.

| Method | Stored adjoint norm bound | Metric J roundoff contribution | Audited wall / s | Control wall / s |
| --- | --- | --- | --- | --- |
| rhf | 0.250885214 | 1.167633563e-13 | 0.2400 | 0.1868 |
| uhf | 0.250885214 | 7.175152813e-14 | 0.2015 | 0.1991 |
| uks | 0.263722314 | 5.153114299e-14 | 0.2067 | 0.2080 |

The 173 focused checks cover exact rational complex adjoint witnesses, zero
and retained ranks, explicit/ordered/global/pair-tree sources, strict resource
caps, receipt provenance and shared Fock behavior. Build shell/primitive and
coupling-pass counts stay equal to their controls. The norm is retained from
existing admission; no extra integral sweep or dense coupling tensor is added.
Payload time is 1.7720 s with peak RSS 279330816 bytes. Times are individual
observations, not isolated propagation overhead or a scaling claim.

Reproduce with `manta_metric_spectrum.py --pair-tree-action --repetitions 1`.
`metric_j_roundoff_evaluated` qualifies only the metric contribution through
the exact adjoint. It excludes rounding in executed coupling forward/adjoint,
weighted density and final Fock assembly; full-Coulomb `roundoff_included`
remains false. Pre-SCF inverse-action targets and the frozen 0.18 scope are
unchanged. Work package #624 and the full goal remain open.

## Executed coupling forward and adjoint arithmetic

Scientific source `2177681933faa7e8f1c37e6c03be7ff84075baaa` adds interval
replay to the existing explicit, ordered, global-panel and pair-tree actions.
Native SHA256 is
`29e43f5ae9b261d1d3d5a03bc6667a90a53857fdf1659dd1609a46ebdb2b1c0d`.
Local `vq` acceptance passes six complete RHF/UHF/UKS selected/control
calculations and three strict refusals with source/native/output pins verified.
Physical outputs remain bitwise equal to controls and evaluated shared Fock
energies agree. Existing source-matched independent references are reused;
maximum energy/Fock differences remain 9.86e-14 Eh and 4.52e-15. No matched
ORCA/CRYSTAL oracle is available for these custom finite supports.

| Method | Forward action roundoff bound | Adjoint action roundoff bound | Audited wall / s | Control wall / s |
| --- | --- | --- | --- | --- |
| rhf | 4.352074257e-14 | 1.665334537e-15 | 0.2469 | 0.1875 |
| uhf | 3.197442311e-14 | 9.159339953e-16 | 0.2044 | 0.2020 |
| uks | 3.463895837e-14 | 7.771561172e-16 | 0.2085 | 0.2061 |

The new raw-action tests pass 34 cases, including exact rational explicit
contractions and complex adjoint-pairing discrepancies enclosed without an
added tolerance. Compressed checks cover non-Gamma/Gamma frames, orders zero
and two, signed angular panels, cancellation, zero/subnormal inputs and near
blocks. Each audited result is bitwise equal to its ordinary action. All five
resource caps are checked before NaN input. Reused materialized-source and
integral/Fourier comparisons cover unchanged physical producer conventions.

The first integration run passed 347 checks and exposed one large-source
count-planning overflow. Charging replay work separately from integral
construction fixes it; 95 targeted planning, cap, arithmetic and complete-method
checks then pass, including the large-source regression. The correction does
not change the numerical schedule or widen test tolerances.

Reproduce with `manta_metric_spectrum.py --pair-tree-action --repetitions 1`.
Payload time is 1.7902 s and peak RSS 278462464 bytes.
Table times include setup and SCF; their difference includes the full metric
policy, not isolated action replay. These single observations establish no
speedup or scaling claim. Replay borrows source data and adds bounded interval
vectors, with no new partition or dense coupling tensor. Its work census
charges contraction replay rather than multiplying integral-generation work.

Contract Section 5.37 defines exact arithmetic on the stored graph as the
comparison object. Forward error still needs propagation through the metric
and adjoint before combination with the metric J and actual adjoint errors.
Integral/source construction, weighted-density and final J/K/Fock arithmetic
remain excluded; full-Coulomb `roundoff_included=false` is unchanged. #624 and
the full MANTA goal remain open; shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL and
symmetry authority and frozen 0.18 scope are preserved.


## Composed paired J and energy arithmetic

Scientific source `0c8d17c45c961bf858c971d83092c7a8a8210de7` composes the accepted
forward, fixed-factor metric and adjoint enclosures for the actual stored
weighted input and the same stored coupling graph. Native SHA256 is
`1d482ad2d37f16ce22b2c05a313382aedd2ca5446b5b746759a55a89443487c9`.
Local `vq` passes six complete RHF/UHF/UKS selected/control calculations and
three strict refusals. Source/native pins and retrieved output hashes agree.
All physical arrays, energies and source identities are bitwise equal to
controls; the shared evaluated Fock energy agrees. Independent materialized
coupling with separate NumPy metric/contractions and finite-source
one-electron/exchange witnesses yields maximum energy/Fock differences of
9.86e-14 Eh and 4.52e-15. KS uses the shared libxc. No matched ORCA/CRYSTAL
reference is available for these custom finite supports.

| Method | Paired J arithmetic bound | Paired energy arithmetic bound / Eh | Audited wall / s | Control wall / s |
| --- | --- | --- | --- | --- |
| rhf | 7.886059002e-12 | 5.950795412e-13 | 0.2527 | 0.1986 |
| uhf | 6.005812813e-12 | 2.549072065e-13 | 0.2189 | 0.2126 |
| uks | 2.968284127e-12 | 1.731947918e-13 | 0.2168 | 0.2196 |

The 243 focused checks pass exact rational perturbed-input factor actions,
full explicit paired J/energy comparisons without additive tolerances,
complex cancellation and subnormal witnesses, zero rank/input, exact/one-short
resource caps, large-source planning, strict refusals and shared Fock controls.
Ordered/global/pair-tree HF/KS paths preserve receipts and standard provenance.
Accepted producer and integral/Fourier evidence is reused for unchanged parts.

Reproduce with `manta_metric_spectrum.py --pair-tree-action --repetitions 1`.
Payload time is 1.8870 s and peak RSS 278478848 bytes.
Times are individual observations including setup and SCF, not isolated
composition cost or a speedup claim. The combined bound uses per-RHS errors
and the already-admitted source norm; no source sweep or dense coupling tensor
is added. Metric arithmetic is counted once. Energy uses the projected-vector
bound, since the numerical energy does not consume the adjoint.

Contract Section 5.38 defines this post-build comparison. Weighted-input and
integral/source construction, final J/K/Fock accumulation and relaxed-SCF
accuracy remain excluded. Pre-SCF inverse targets and full-Coulomb
`roundoff_included=false` remain unchanged. This advances #624 but leaves
it and the full MANTA matrix open. Shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL,
symmetry authority and frozen 0.18 scope remain unchanged.


## Spin-summed and k-weighted input arithmetic

Scientific source `e30df649a0a58a999cf7609e88161cfc170b77ed` extends the paired
comparison to exact spin summation and multiplication by stored k weights.
Native SHA256 is `923dbe51a5e024cc3c969bfbc9034b113bf99ca9b9c68197a9db9ecf10f09aa4`.
Local `vq` passes six complete RHF/UHF/UKS selected/control calculations and
three strict refusals with verified source/native and retrieved output hashes.
Physical arrays, energies and source identities remain bitwise equal to their
controls; evaluated shared Fock energies agree. Reused independent materialized
coupling, separate NumPy metric/contractions and finite-source one-electron/K
references give maximum energy/Fock differences of 9.86e-14 Eh / 4.52e-15.
KS uses shared libxc. No matched ORCA/CRYSTAL reference is available for these
custom finite supports.

| Method | Weighted input error bound | Paired J arithmetic bound | Paired energy arithmetic bound / Eh | Audited wall / s | Control wall / s |
| --- | --- | --- | --- | --- | --- |
| rhf | 7.105427358e-15 | 1.119545781e-11 | 8.215650382e-13 | 0.2376 | 0.1845 |
| uhf | 7.105427358e-15 | 7.979570038e-12 | 3.321787290e-13 | 0.1992 | 0.1952 |
| uks | 7.105427358e-15 | 4.201501069e-12 | 2.335909244e-13 | 0.2042 | 0.2030 |

The initial focused run passed 365 checks. Sixteen new cases stopped at a
work-count assertion comparing first-use source setup with a prepared source.
After warming both controls consistently, 34 targeted cases pass, including
nine additional spin-cancellation cases. No numerical tolerance was changed.
The combined coverage qualifies 390 distinct checks. Exact rational paired
references cover two spins, non-power-of-two stored weights, zero weights,
cancellation and zero density; explicit and compressed producer checks cover
input uncertainty, Gamma/non-Gamma frames, orders zero/two, all resource caps,
invalid inputs and zero-radius equivalence to the previous audit.

The producer carries input rectangles through the existing graph, including
Gamma conjugate/permuted inputs, without additional source passes. Scope flags
identify the input contribution in q/Fock receipts and standard provenance.
Input and metric errors are propagated once. Plans charge added storage/work;
physical state, source identity, solver policy and symmetry admission are
unchanged. No dense coupling tensor is introduced.

Reproduce with `manta_metric_spectrum.py --pair-tree-action --repetitions 1`.
Payload time is 1.7298 s; peak RSS is 278396928 bytes.
Individual setup/SCF times are observations, not isolated audit overhead or a
speedup claim. Contract Section 5.39 defines the comparison at stored spin
densities and weights. Errors in density generation, integral/source
construction, final J/K/Fock assembly and relaxed SCF remain excluded.
Full-Coulomb `roundoff_included=false`, existing pre-SCF inverse targets,
frozen 0.18 scope and shared Fock/BIPOLE/CRYSTAL/symmetry authority remain
unchanged. #624 and the full MANTA goal stay open.


## Evaluated Fock and energy assembly arithmetic

Scientific source `d53648176b38403a67dd3f96ec038daef3f8aee4` adds a shared stored-component
assembly audit to the existing native HF/KS adapters. Native SHA256 is
`18b548401dbfa652971971d0d19521b12b2469ce0f3368be8364545025207fc8`.
Local `vq` qualifies 18 selected/control calculations: 12 converged and six
intentionally iteration-capped states, across RHF, UHF, RKS, UKS, PBE0 and
HSE06. Three strict pre-SCF metric refusals also pass. Source/native pins and
171 retrieved output hashes match; all physical arrays, energies and source
identities are bitwise equal to their unaudited controls.

The RHF/UHF/UKS cases reuse independent materialized coupling, separate NumPy
metric/contractions, finite-source one-electron/exchange references and the
shared libxc KS functional. Maximum energy/Fock differences are 9.86e-14 Eh
and 4.52e-15. Shared evaluated Fock energies agree. RKS and global/range
compositions additionally qualify assembly and zero-mode correction for both
converged and stopped states using bitwise controls and prior component
acceptance. No matched ORCA/CRYSTAL reference is available for these custom
finite supports; these comparisons do not establish infinite-source accuracy.

| Method/state | Assembly Fock bound | Assembly energy bound / Eh | Combined paired Fock bound | Combined paired energy bound / Eh |
| --- | --- | --- | --- | --- |
| rhf converged | 1.665334537e-16 | 9.059419881e-14 | 1.119562435e-11 | 9.121592370e-13 |
| uhf converged | 4.440892099e-16 | 5.107025913e-14 | 7.980014128e-12 | 3.832489881e-13 |
| uks converged | 3.330669074e-16 | 1.776356839e-14 | 4.201834136e-12 | 2.513544928e-13 |
| rks converged | 8.326672685e-17 | 3.907985047e-14 | 7.486348455e-12 | 7.114309142e-13 |
| rks capped | 1.110223025e-16 | 3.730349363e-14 | 6.272535905e-12 | 6.590283874e-13 |
| pbe0 converged | 2.220450498e-16 | 1.136868377e-13 | 7.070672424e-12 | 7.371880884e-13 |
| pbe0 capped | 1.942890293e-16 | 9.947598301e-14 | 6.272619172e-12 | 7.212008768e-13 |
| hse06 converged | 1.665343400e-16 | 7.638334409e-14 | 7.397172459e-12 | 7.167599847e-13 |
| hse06 capped | 1.665334537e-16 | 7.460698725e-14 | 6.272591416e-12 | 6.963318810e-13 |

The 803 focused/shared checks include exact rational comparisons with signed
exchange, complex entries, cancellation, gradual underflow, deliberately
perturbed public witnesses, all three raw resource caps, finite-input refusal,
and complete HF/KS compositions. Ten focused citation checks pass. The helper
compares its intervals to the actual final evaluated Fock, so callback
reconstruction and Hermitian projection are included. Persistent interval
storage and per-call work are admitted before SCF. No numerical source sweep,
dense coupling tensor or additional confirmation call is introduced.

The primary three-method benchmark is reproducible with
`manta_metric_spectrum.py --pair-tree-action --repetitions 1`; the additional
six-method converged/capped controls are in
`test_assembly_complete_methods_and_stopped_states`. Payload time for all
queued cases is 3.8375 s and peak RSS is
280477696 bytes. These whole-process observations are not isolated
audit overhead or a speedup claim.

Contract Section 5.40 defines the exact stored-component comparison and the
single composition with accepted paired Coulomb arithmetic. H/K/XC/nuclear
and correction construction, density generation, relaxed SCF and other
non-paired routes remain excluded. Full-Coulomb `roundoff_included=false`,
pre-SCF inverse targets and the bounded 0.18 release scope remain unchanged.
#624 and full MANTA #617 stay open; shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL
and source-qualified symmetry retain their ownership.


## Direct periodic exchange contraction arithmetic

Scientific source `c7f61c14030d736d151e05fbce4ee3aaf822496b` audits direct exchange at the
actual stored Bloch Fourier panels, density channels and reciprocal factors.
Native SHA256 is `e3e12611398b35ec3606fcb8d9dca5230a3c2129adde322f116b07bf562c91cb`.
Local `vq` qualifies 26 selected/control calculations: 16 converged and ten
intentionally iteration-capped states, across RHF/UHF, pure RKS/UKS, PBE0,
HSE06 and restricted/unrestricted CAM-B3LYP. Three strict pre-SCF metric
refusals pass. Source/native pins and 243 retrieved output hashes match.
Physical arrays, energies and source identities remain bitwise against their
unaudited controls. The signed CAM-B3LYP components exercise positive Coulomb
and negative erfc coefficients through the shared method audit.

The independent RHF/UHF/UKS reference differences remain below 9.86e-14 Eh
for energy and 4.52e-15 for Fock. These reuse materialized coupling, separate
NumPy metric/contractions, finite-source one-electron/exchange and shared libxc
KS references. No matched ORCA/CRYSTAL reference exists for these custom finite
supports. Other method controls reuse accepted source components; this does
not qualify infinite-source or relaxed-SCF accuracy.

| Method/state | Direct-K Fock contribution | Direct-K energy contribution / Eh | Combined Fock bound | Combined energy bound / Eh |
| --- | --- | --- | --- | --- |
| rhf converged | 4.029286182e-15 | 2.589388913e-13 | 1.119965363e-11 | 1.171098128e-12 |
| uhf converged | 4.173167467e-15 | 1.579175198e-13 | 7.984187295e-12 | 5.411665079e-13 |
| uks converged | 0.000000000e+00 | 0.000000000e+00 | 4.201834136e-12 | 2.513544928e-13 |
| rks converged | 0.000000000e+00 | 0.000000000e+00 | 7.486348455e-12 | 7.114309142e-13 |
| rks capped | 0.000000000e+00 | 0.000000000e+00 | 6.272535905e-12 | 6.590283874e-13 |
| pbe0 converged | 1.020836232e-15 | 6.339275915e-14 | 7.071693260e-12 | 8.005808475e-13 |
| pbe0 capped | 1.065871463e-15 | 5.667237383e-14 | 6.273685043e-12 | 7.778732506e-13 |
| hse06 converged | 1.268706961e-15 | 7.731421653e-14 | 7.398441166e-12 | 7.940742012e-13 |
| hse06 capped | 1.263742475e-15 | 7.001605737e-14 | 6.273855159e-12 | 7.663479384e-13 |
| cam-rks converged | 4.239151846e-15 | 2.743365756e-13 | 7.341775972e-12 | 1.059486299e-12 |
| cam-rks capped | 4.386921828e-15 | 2.462235657e-13 | 6.277089361e-12 | 1.004727936e-12 |
| cam-uks converged | 4.349909026e-15 | 1.686376061e-13 | 4.156713010e-12 | 4.795000530e-13 |
| cam-uks capped | 4.287987671e-15 | 1.605944371e-13 | 3.840603841e-12 | 4.772300437e-13 |

For pure KS the direct-K contribution is zero and its scope flag is false.
Raw producer energy bounds include contraction and final weighted trace;
the method composition instead propagates K-entry errors through the actual
exchange coefficients and method density. It does not add the raw producer
energy bound, which would count trace arithmetic twice.

The 894 focused/shared scientific checks include 50 exact-rational stored-panel
cases over Coulomb/erf/erfc, two panel sizes, signed/cancellation-heavy densities,
subnormal values, zero density and zero weights. Three additional checks admit
each raw memory/scalar/kernel cap exactly and reject one-short caps before
nonfinite payloads. Exact rational perturbed-K references qualify method
propagation for signed coefficients and both spin counts; zero radii recover
the original assembly bounds. Complete positive/negative range compositions,
stopped states, existing phase/image and shared Fock checks pass. Ten citation
and five documentation checks pass. Native source values, work schedule and
source/epoch identities are unchanged; the audit adds interval work and bounded
storage, with no second Fourier sweep or four-index AO tensor.

The primary three-method benchmark is reproducible with
`manta_metric_spectrum.py --pair-tree-action --repetitions 1`. The other method
controls are `test_assembly_complete_methods_and_stopped_states` and
`test_direct_exchange_signed_range_composition`. The complete queued payload
takes 5.3747 s with peak RSS 279429120 bytes.
These are whole-process observations, not isolated audit overhead or a speedup
claim. Contract Section 5.41 specifies the stored-panel/factor reference and
composition. Fourier/kernel-factor construction, density generation and source
embedding, XC/correction construction, fitted/retained/fused K arithmetic,
rank and relaxed-SCF errors remain excluded. Full-Coulomb roundoff, useful
automatic accuracy and the full capability matrix remain unfinished. The
shared Fock/eigensolver, BIPOLE/CRYSTAL and symmetry authorities and bounded
0.18 release scope are unchanged. #624 and full MANTA #617 remain open.


## Stored exchange zero-mode correction arithmetic

Scientific source `19a8cd49d3100212324db9d6555e4d261c9b70af` extends the evaluated assembly audit
to correction construction C0 = -a m S D S at the stored overlap, source spin
density, Madelung coefficient and method exchange scale. Native SHA256 is
`a5f04ebcf4c25a8cb7398a05c02e671ca6b8081aad204f21ae3db7a5c5c5c2d0`.
The shared interval helper encloses both matrix products and coefficient
multiplication, preserving the physical compensated calculation. Per-entry
complex L1 errors propagate once through the existing Fock and weighted
energy assembly. This does not bound construction of S or m themselves.

Local `vq` qualifies 26 selected/control calculations, 16 converged and ten
intentionally capped, across RHF/UHF, pure RKS/UKS, PBE0, HSE06 and restricted/
unrestricted CAM-B3LYP. Three strict pre-SCF metric refusals pass. All 1938
scientific source pins and 243 retrieved output hashes match. Energies,
physical arrays and source identities are bitwise equal to unaudited controls.
Independent RHF/UHF/UKS energy and Fock discrepancies stay below 9.86e-14 Eh
and 4.52e-15, respectively. These reuse materialized coupling, separate NumPy
metric/contractions, finite-source one-electron/exchange and shared libxc
references. No matched ORCA/CRYSTAL reference is available for these custom
finite supports; other method controls reuse accepted component references.

| Method/state | Correction Fock contribution | Correction energy contribution / Eh | Combined Fock bound | Combined energy bound / Eh |
| --- | --- | --- | --- | --- |
| rhf converged | 6.591949209e-17 | 5.323901074e-15 | 1.119971955e-11 | 1.176422029e-12 |
| uhf converged | 8.673617380e-17 | 3.307432114e-15 | 7.984274031e-12 | 5.444739400e-13 |
| uks converged | 0.000000000e+00 | 0.000000000e+00 | 4.201834136e-12 | 2.513544928e-13 |
| rks converged | 0.000000000e+00 | 0.000000000e+00 | 7.486348455e-12 | 7.114309142e-13 |
| rks capped | 0.000000000e+00 | 0.000000000e+00 | 6.272535905e-12 | 6.590283874e-13 |
| pbe0 converged | 1.734723476e-17 | 1.365688393e-15 | 7.071710608e-12 | 8.019465359e-13 |
| pbe0 capped | 1.474514955e-17 | 1.116030482e-15 | 6.273699789e-12 | 7.789892811e-13 |
| hse06 converged | 0.000000000e+00 | 0.000000000e+00 | 7.398441166e-12 | 7.940742012e-13 |
| hse06 capped | 0.000000000e+00 | 0.000000000e+00 | 6.273855159e-12 | 7.663479384e-13 |
| cam-rks converged | 4.163336342e-17 | 3.291130219e-15 | 7.341817605e-12 | 1.062777429e-12 |
| cam-rks capped | 3.816391647e-17 | 2.844085573e-15 | 6.277127524e-12 | 1.007572022e-12 |
| cam-uks converged | 4.510281038e-17 | 2.001172532e-15 | 4.156758113e-12 | 4.815012256e-13 |
| cam-uks capped | 4.163336342e-17 | 1.823397526e-15 | 3.840645474e-12 | 4.790534412e-13 |

The 862 focused/shared scientific checks include 24 exact-rational independent
four-index correction expansions with non-Hermitian overlaps, signed/zero
scales, cancellation and subnormal products, plus deliberately perturbed
witnesses. Eight tests compose these exact references into Fock and energy;
zero radii recover the previous assembly bounds. Exact/one-short raw resource
caps, nonfinite and malformed input refusals, strict-zero-mode exclusions,
converged/capped HF/KS and signed range compositions pass. Ten citation and
five documentation checks pass. Pure KS and HSE06 have no long-range BvK
correction in these fixtures and retain zero contributions. The correction
trace arithmetic is already included in assembly and is not added again.

The three-method benchmark uses `manta_metric_spectrum.py --pair-tree-action
--repetitions 1`; the complete-method tests and exact-rational diagnostics are
in `test_manta_inverse_action.py`. Parent plans charge all audit work and
scratch before SCF. No new Fourier sweep, physical source, Fock object,
eigensolver policy or symmetry permission is introduced.

The whole queued payload takes 6.4070 s with peak RSS
284852224 bytes; this is not an isolated audit-overhead or speedup
claim. Contract Section 5.42 defines the scope. Overlap/Madelung and other
input construction, density embedding, retained/fused RI, XC, rank and
relaxed-SCF accuracy remain excluded. Full-Coulomb `roundoff_included` remains
false. #624 and full MANTA #617 remain open; the bounded #557 release scope
and shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL and symmetry authorities are
unchanged.


## Retained-mode periodic RI arithmetic

Scientific source `63a80bf1f18630b504d61d2e120e12158320745c` audits the actual streamed
F^dagger T mode blocks, spin/k weighted Coulomb response and fused q-channel
B D B^dagger exchange at stored factor/tile inputs. Native SHA256 is
`30a1deb7ed608e47c2c5ad6416a6526792452fe7c6b72ceb8a94b8ae68c5dfcf`.
The retained J/energy bounds propagate once through existing evaluated HF/KS
assembly. Fused K entry bounds remain per q channel; complete fitted-K cross-q
aggregation is not yet qualified. Integral, factor and physical-source
construction errors remain separate.

Local `vq` passes 26 selected/control calculations, 16 converged and ten
intentionally capped, across RHF/UHF, pure RKS/UKS, PBE0, HSE06 and restricted/
unrestricted CAM-B3LYP. Three strict pre-SCF target refusals and four additional
fixed-q J/K calculations with exact rational references pass. The 1939
scientific source pins and 243 retrieved output hashes match. Physical arrays,
energies and source identities are bitwise equal to unaudited controls.
Independent RHF/UHF/UKS energy/Fock differences are below 1.20e-13 Eh/4.86e-15.
These reuse materialized coupling, separate NumPy metric/contractions,
finite-source one-electron/exchange and shared libxc references. No matched
ORCA/CRYSTAL reference exists for these custom finite supports. The incoming
0.17.26 release metadata changed only the version among the pinned inputs;
all numerical source and the tested native runtime are retained unchanged.

| Method/state | Combined Fock bound | Combined energy bound / Eh |
| --- | --- | --- |
| rhf converged | 1.222390220e-14 | 4.862647129e-13 |
| uhf converged | 9.217742614e-15 | 2.660332258e-13 |
| uks converged | 3.941291737e-15 | 6.838973832e-14 |
| rks converged | 6.328271240e-15 | 1.536548666e-13 |
| rks capped | 6.106226635e-15 | 1.509903313e-13 |
| pbe0 converged | 7.554031396e-15 | 2.888291970e-13 |
| pbe0 capped | 7.270109975e-15 | 2.709512250e-13 |
| hse06 converged | 7.763277782e-15 | 2.646995655e-13 |
| hse06 capped | 7.425480261e-15 | 2.547571687e-13 |
| cam-rks converged | 1.072236485e-14 | 5.347211664e-13 |
| cam-rks capped | 1.064233468e-14 | 4.977576088e-13 |
| cam-uks converged | 8.645730075e-15 | 3.122905491e-13 |
| cam-uks capped | 8.492957377e-15 | 2.889832594e-13 |

| Fixed-q source | Spins | Retained J bound | Fused K bound | Coulomb energy bound / Eh |
| --- | --- | --- | --- | --- |
| nonzero transfer (K only) | 1 | 0.000000000e+00 | 2.438819979e-13 | 0.000000000e+00 |
| nonzero transfer (K only) | 2 | 0.000000000e+00 | 2.874478607e-13 | 0.000000000e+00 |
| Gamma | 1 | 1.307494822e-14 | 1.240063607e-14 | 1.243449788e-14 |
| Gamma | 2 | 1.237364671e-14 | 1.363326119e-14 | 2.664535259e-15 |

The 1094 passing focused/shared scientific checks include 38 exact-rational
stored-factor/tile cases with complex q channels, independent spin, two block
sizes, signed/cancellation-heavy and subnormal/zero densities, zero weights
and zero/truncated/full rank. Twelve complete-method converged/capped controls
qualify retained assembly. Counts reserve actual interval storage and only
reachable arithmetic routes; mutually exclusive paired/retained buffers are
not added together. The bounded-memory census and exact/one-short resource
checks pass. Fifty-two affected checks pass again on the final native build;
ten citation and five documentation checks pass.

Two automatic-reference profiles separately refuse their scalar-work or memory
budgets. Both reproduce with the periodic RI producer from prior main
`5c1aae28`, where retained auditing is absent. They remain tracked in #643 and
are not included in passing acceptance. No numerical tolerance or cap was
relaxed. This limits those automatic-selection profiles, not the fixed-source
retained qualification above.

Reproduce the three-method benchmark with
`manta_metric_spectrum.py --retained-action --repetitions 1`; raw exact-rational
cases are in `test_manta_periodic_ri_q.py`, and complete-method controls in
`test_manta_inverse_action.py`. The mode audit shares the physical producer's
existing tile sweeps and allocates only the admitted block and response
intervals. The numerical contractions, identities, shared Fock/eigensolver,
BIPOLE/CRYSTAL and symmetry authorities are unchanged.

The whole queued payload takes 4.6364 s with peak RSS
280297472 bytes. These observations include controls and independent
references; they do not measure isolated audit overhead or demonstrate a
speedup. Contract Section 5.43 defines the reference. Complete fitted-K
aggregation, integral/factor construction, density embedding, rank, XC and
relaxed-SCF errors remain excluded. Full-Coulomb `roundoff_included=false`,
#624 and full #617 stay open. The bounded #557 release scope is unchanged.


## Whole-mesh fitted exchange arithmetic

Scientific source `9e3a8c9ab510a8d46bb343b93b6c0c394c97c4e5` extends the stored-factor q-channel
exchange radii through the existing ordered-q accumulation and the evaluated
HF/global/range KS Fock and energy assembly. Native SHA256 is
`1fb3bd2640dcd4554f98b6f0fb4c48860fc0e0f6ea298e5a417867261c234fd0`.
Physical integral sweeps, coefficients, evaluated states and source identities
are unchanged. Scientific contract Section 5.44 specifies the stored-input
reference and separate fitted/direct/correction contributions.

Local `vq` passes 32 selected/control calculations: 16 converged and 16
intentionally capped, spanning RHF/UHF and restricted/unrestricted PBE0,
HSE06 and CAM-B3LYP. Every evaluated state is compared with independent
libint/Gaussian integral sources, NumPy factors/contractions and finite-source
one-electron terms; KS reuses the shared libxc implementation. Maximum energy
and Fock differences are 1.12e-13 Eh and 4.85e-15. All physical arrays, energies
and source identities are bitwise equal to unaudited controls. Shared evaluated
Fock exports preserve method energy. The 1939 scientific input pins and all
288 retrieved output hashes match the tested source and runtime.

Six additional raw whole-mesh witnesses cover Gamma and four-point shifted
meshes with Coulomb, erf and erfc K. Independent exact binary-rational
expansion of stored factors, coupling tiles and weighted spin densities
lies inside every K entry radius. The focused checks additionally cover
empty/truncated/full metrics, block sizes, zero and subnormal densities,
cancellation, signed/zero coefficients, independent resource boundaries and
source-qualified image transport. There are 1428 distinct passing scientific
checks and 27 passing documentation/citation checks. Two known automatic
reference budget refusals remain excluded under #643; their earlier failure
evidence is retained. A pre-existing downloads changelog formatting failure
was repaired without changing its content, and the check then passed.

| Profile | Energy (Eh) | Fitted K Fock radius | Fitted K energy radius (Eh) |
| --- | --- | --- | --- |
| rhf | -9.615420772487 | 1.8874e-15 | 9.3506e-14 |
| uhf | -4.094056282195 | 1.8319e-15 | 5.4606e-14 |
| pbe0-rks | -9.700417784507 | 3.8858e-16 | 2.1094e-14 |
| pbe0-uks | -4.145245061676 | 3.9899e-16 | 1.2306e-14 |
| hse06-rks | -9.880935008764 | 4.9960e-16 | 2.8577e-14 |
| hse06-uks | -4.284658734208 | 4.8572e-16 | 1.6508e-14 |
| cam-b3lyp-rks | -9.672681162937 | 1.7576e-15 | 9.1804e-14 |
| cam-b3lyp-uks | -4.143618577080 | 1.6652e-15 | 5.4069e-14 |

The queued payload took 11.7348 seconds and peaked at
292552704 bytes RSS. This includes reference construction, output
and exact-rational witnesses, so it is not a production timing or speedup.
Reproduce the complete-method benchmark with
`.venv/bin/python -m benchmarks.manta_metric_spectrum --fitted-exchange --repetitions 1`.
Exact-rational cases are in `test_manta_periodic_ri_fock.py`; complete-method,
capped-state and signed assembly checks are in `test_manta_inverse_action.py`.

No matched ORCA/CRYSTAL result exists for these custom finite supports.
The reference holds density, weights, factors and source tiles fixed;
source construction, metric/rank perturbations, restricted density embedding,
XC construction and relaxed-SCF errors remain excluded. Earlier q-channel,
retained-J, direct-K and correction evidence is reused, not requalified as a
whole-method accuracy claim. #624 and full #617 remain open; bounded #557,
shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL and symmetry admission are unchanged.


## Full-Coulomb fitted exchange component certificate

Implementation `93f6ec00`, qualified at integrated source
`7107d2e7e5b7bdfe1b6aad72c700c1f837fbce59`, compares the actual shared RI exchange map
with the existing finite reciprocal exchange producer and its full-image/
full-reciprocal remainder. Native SHA256 is
`9ac72c480902b91ad443beedc3e9cd79e9937927c7e10635079492bcdceba68a`.
Hermitian density probes establish finite discrepancy and reference remainder
without assuming a positive or variational fit residual. Existing probe
arithmetic bounds are included. Scientific contract Section 5.45 defines the
unweighted spin-absolute family, energy pairing and explicit exclusions.

Local `vq` passes two converged RHF/UHF calculations, four additional raw
family cases (Gamma/multi-k and full/truncated/empty fits), and four strict
target refusals. The component owner exactly matches each completed method's
fitting-source identity. Independent Gaussian-contour Fourier quadrature
compares finite and enlarged image/reciprocal supports; complete-method
libint/Gaussian and NumPy witnesses agree within 1.12e-13 Eh in energy and
4.85e-15 in Fock entries. Source/native and all 1941 scientific input pins
plus 18 retrieved output hashes are verified. There is no matched ORCA/CRYSTAL
reference for these custom finite supports.

A supplementary queued case at source `7e5f6760` retains one of two auxiliary
modes at cutoff 0.5 in every q channel. Its measured K discrepancy
0.0026491575 is enclosed by 63.3481821; an impossible target refuses. The
original cutoff-10 fixture discarded every mode, so it did not test partial
rank. The focused tests now assert the actual full, partial and empty ranks.
All 27 focused certificate/citation checks pass. Only this test fixture changed
after the two complete HF calculations; their product source, native binary
and independent references are unchanged and their acceptance is reused.

There are 539 passing affected scientific checks on the final native runtime,
including 26 certificate/resource/family tests, and 28 documentation/citation
checks. Ten changelog checks also pass after integrating the independently
validated CRYSTAL timing correction; that change does not affect the numerical
source. Prior exchange-tail and stored-input arithmetic acceptance is reused.

| Method | K matrix bound | Measured K difference | Exchange energy bound (Eh) | Measured energy difference (Eh) |
| --- | --- | --- | --- | --- |
| RHF | 222.76471 | 0.0010644351 | 71997.872 | 0.0064138067 |
| UHF | 222.76471 | 0.0007723072 | 71997.872 | 0.0058187129 |

Both cases use a reference image cube of radius 1 and reciprocal cube of
radius 8, with AO completion radius 2. The canonical occupied-space density
bounds are converted outward to the component's unweighted convention.
The original radius-0/radius-3 reference gave an RHF matrix bound near 3287;
the expanded reference reduces this to about 223. The result is still very
conservative: the finite-discrepancy maximum is about 0.203, and the reference
remainder maximum is about 222.749. These are uniform family bounds, not error
estimates fitted to the observed SCF density. Useful admission thresholds,
automatic reference/auxiliary selection and complete-method attachment remain
open; no production accuracy or speedup claim follows from this enclosure.

The queued payload took 27.4182 seconds and peaked at
284540928 bytes RSS, including calculations, reference construction
and output. Reproduce with
`.venv/bin/python -m benchmarks.manta_metric_spectrum --exchange-fitting --repetitions 1`.
The benchmark reserves the source and probe work through the existing plans;
these conservative plans are not measured runtime/RSS predictions.

The certificate holds the input source integrals and factors fixed. Source
construction, factor/rank perturbation, restricted density embedding, physical
range/XC certificates, BvK/source correction errors and SCF relaxation remain
separate. A computed K response also needs its own arithmetic radius. #626,
#627 and full #617 stay open; #643 resource failures are unchanged. Shared
Fock/eigensolver, PBC/BIPOLE/CRYSTAL, symmetry and bounded #557 scope are preserved.


## Preconstruction exchange-certificate capacity

Count-only resource admission in `a8a57eff`, qualified at source
`c89aef19da43a873f61eee2bf02a2bb853d115fa`, shares the realized certificate aggregation
and original pair/exchange-tail planners. Native SHA256 is
`b704f2435b755fb01d4863204bec898c7cf0b767fbd3ead16f886a652b7b1a22`. It includes reference-pair
preparation and optionally retained J-certificate storage; the caller separately
charges RI setup. Scientific contract Section 5.46 specifies this boundary.

All 45 affected checks pass, including 18 new capacity cases spanning full,
partial and empty ranks, screened and unscreened pairs, five exact/one-short
caps, invalid counts, overflow and retained J lifetime. An initial invalid-basis
test fixture used a non-atomic shell origin; the fixture was corrected and its
check passed without changing product code. Fifty-two documentation checks pass.
Unchanged tail-source numerical acceptance is reused.

Two complete local `vq` RHF/UHF calculations pass using the new capacity
before constructing certificate owners. Energies, iteration counts, source
identities, certificate identities and numerical bounds match the previous
accepted cases bit for bit. Independent Gaussian-contour/libint/NumPy checks
and all 1941 input pins plus 18 output hashes pass. There is no matched
ORCA/CRYSTAL reference for these custom supports. The queue payload took
23.8328 seconds and peaked at 279232512 bytes RSS.

| Method | Capacity bytes | Realized certificate bytes | Capacity shell calls | Realized certificate shell calls |
| --- | --- | --- | --- | --- |
| RHF | 81560690 | 72866644 | 677033625 | 677033256 |
| UHF | 81560690 | 72866644 | 677033625 | 677033256 |

Capacity also bounds scalar, kernel and primitive work. Its additional work
covers pair preparation and any pair entries later removed by screening;
these conservative byte/work bounds are not measured runtime estimates.
Reproduce with the existing `--exchange-fitting --repetitions 1` benchmark.
This is a prerequisite for complete-method automatic attachment, which remains
unfinished under #626/#627. It does not tighten the conservative physical
bounds or change Fock/eigensolver, BIPOLE/CRYSTAL or symmetry semantics.


## Pre-SCF fitted exchange admission for RI HF

Source `2dfbd54dc150343a8fdd98f70af97c4c1f453326` attaches the full-Coulomb K component
certificate to the actual periodic RI RHF/UHF owner before SCF. Native SHA256:
`171d54e46e7da21c6822ac6b27ab893d456eb996af0434ed433eff52793b4c78`. Public
`PeriodicFittingOptions.full_coulomb_exchange_accuracy` requests positive raw
matrix/energy targets; the same canonical occupied-space family is converted
outward to the component convention and checked on every evaluated density.
Section 5.47 defines source, lifetime and arithmetic boundaries.

Local `vq` passes eight converged selected/control calculations (RHF/UHF,
standalone K and joint J/K controls), four deliberately iteration-capped
partial-rank selected/control calculations and two pre-SCF target refusals.
Energies, densities, physical Focks, coefficients and orbital energies are
bitwise equal to controls. The partial fits retain two of three auxiliary
modes. All 1942 input pins and 108 retrieved output hashes are verified.
Independent libint/Gaussian-contour/NumPy witnesses agree within 1.12e-13 Eh
in complete HF energy and 4.85e-15 in Fock entries. No matched ORCA/CRYSTAL
reference exists for these custom finite supports; previous full-domain tail,
arithmetic and raw-certificate acceptance is reused.

There are 184 passing focused/shared scientific checks, including standalone
and joint policies, stopped states, Gamma/shifted meshes, actual partial ranks,
strict/unsupported controls and resource boundaries. Six accounting checks,
including execution at the joint native caps, pass again on the final runtime.
Fifty-two documentation checks pass. A test fixture initially combined a J-only
coupling approximation with RI K; clearing that unrelated approximation made
its intended capacity checks valid. The final accounting revision additionally
charges assembly/correction audits activated by K accuracy.

| Method | Joint J control | Attached K bound | Attached exchange energy bound (Eh) | Measured K difference |
| --- | --- | --- | --- | --- |
| RHF | False | 222.7647141 | 71997.87157 | 0.001064435074 |
| RHF | True | 222.7647141 | 71997.87157 | 0.001064435074 |
| UHF | False | 222.7647141 | 71997.87156 | 0.0007723072003 |
| UHF | True | 222.7647141 | 71997.87156 | 0.0007723072003 |

The deliberately loose targets are 1000 for raw K entries and 1e6 Eh for its
exchange energy. They demonstrate admission and refusal behavior, not useful
automatic accuracy. Reference image/reciprocal cubes remain fixed at radii 1/8,
with AO completion radius 2. The response arithmetic radius is separate from
these raw-component targets; source construction, BvK/source corrections, other
Fock components and relaxation remain excluded. KS jobs explicitly refuse this
control pending their own attachment. No production speedup or total-energy
claim is made.

The queued payload took 82.7582 seconds with peak RSS
277856256 bytes. Reproduce the converged ladder with
`.venv/bin/python -m benchmarks.manta_metric_spectrum --exchange-admission --repetitions 1`.
The separate physical source, Fock/eigensolver and symmetry identities remain
unchanged. #626/#627 and full #617 stay open; #643 and bounded #557 are unchanged.


## Pre-SCF fitted exchange admission for global-hybrid RI KS

Source `2dfb076d226b4cec806691b74189818004b69f19` extends the existing full-Coulomb K
certificate helper to the actual RI global-hybrid RKS/UKS owner. Native SHA256:
`19ed180fad84483d3e3dfee296f63c3f83f510fa770d19665969a948b7fa1abd`. Contract Section 5.48 defines raw
component targets, the canonical spin family, physical exchange scaling and
separate arithmetic/exclusion ledgers. No Fock, XC or orbital solver is duplicated.

Local `vq` passes twelve converged selected/control calculations, eight
iteration-capped selected/control calculations and two strict pre-SCF target
refusals. These cover PBE0 RKS/UKS with standalone K and joint J/K policies,
exact-J/fitted-K compositions and an admitted custom negative exchange fraction
of -0.2. Energies, densities, physical Focks, coefficients and orbital energies
are bitwise identical to controls. All 1942 input pins and
144 retrieved output hashes are verified.

Independent libint/Gaussian Fourier sources and NumPy contractions, using the
same libxc functional values, agree within 4.18e-14 Eh for energy and 1.07e-14
for Fock entries across these cases. Exact-J/fitted-K independent SCF densities
agree within 2.89e-11. No matched ORCA/CRYSTAL reference exists for these custom
finite supports; prior raw K full-domain/image and arithmetic evidence is reused.
The independently implemented contractions do not independently validate libxc.

There are 678 passing affected scientific checks and
52 passing documentation/changelog checks. These include
native execution at the simultaneous resource caps, signed scaling, stopped
states, density-family admission and unsupported pure/range/exact-K refusals.
The initial RKS test added an extra spin dimension; correcting its fixture to
the existing KS spin layout resolved four failures without changing numerical
tolerances or product behavior. The integrated native contains the incoming
periodic XC density-image reuse change.

| Profile | Joint J policy | Exchange fraction | Raw K bound | Scaled K bound | Raw exchange energy bound (Eh) |
| --- | --- | --- | --- | --- | --- |
| pbe0-rks | False | 0.25 | 222.76516 | 55.69129 | 71997.941 |
| pbe0-rks | True | 0.25 | 222.76516 | 55.69129 | 71997.941 |
| pbe0-uks | False | 0.25 | 222.76516 | 55.69129 | 71997.941 |
| pbe0-uks | True | 0.25 | 222.76516 | 55.69129 | 71997.941 |

The raw targets are 1000 for K and 1e6 Eh for exchange energy. The loose bounds
are an attachment qualification, not useful automatic accuracy. Scaling by the
absolute physical exchange fraction does not alter those raw targets or include
computed assembly/correction arithmetic. Source construction, factor/rank
perturbation, XC, other Fock components and relaxed-SCF error remain excluded.
Pure and physical range-separated KS remain unsupported for this control.
Source selection and useful whole-method bounds remain open under #626/#627;
source/rank work and the original matrix are still required. Shared Fock,
PBC/BIPOLE/CRYSTAL and symmetry ownership and bounded #557 scope are unchanged.

Reproduce this attachment benchmark with:

```sh
.venv/bin/python -m benchmarks.manta_metric_spectrum --exchange-ks-admission --repetitions 1
```


## Physical range exchange component enclosures

Source `f1698506b284ca951bab93ad035bc5be44e8722d` extends the existing shared exchange-tail
source and Hermitian RI probe certificate to physical erf/erfc. Native SHA256:
`c604c121e8563ea94433621c7ddca8d1f6f006f808a99ae080ff0a65f16d2d0e`. Contract Section 5.49 derives the
nonzero-Q Coulomb majorant and explicit finite screened-zero accounting. Kernel,
range, source and support identities remain distinct; the RI owner and its
range-specific projector are unchanged. There is no new exchange engine.

Local `vq` passes ten converged selected/control calculations: HSE06 and
CAM-B3LYP RKS/UKS and a reused PBE0 Coulomb control. Six certificates reconstruct
exactly the range methods' actual component-source identities. Four additional
erf fixed-family cases exercise truncated/empty fits; two impossible component
targets refuse. Selected/control method states are bitwise equal. The earlier
PBE0 source, certificate identity and numerical bounds remain bitwise unchanged.
All 1942 source pins, eighteen Coulomb-control output hashes and the range
calculation result digest are verified. There are 488 passing affected scientific
checks and 52 documentation/changelog checks, including finite-zero inclusion/
omission, zero-range tail limits, source/range refusals and exact resource caps.

Independent libint/Gaussian Fourier sources, NumPy fitting/contraction and shared
libxc agree within 5.16e-14 Eh in complete energy and 3.15e-14 in Fock entries.
Physical component differences compare independent Gaussian-contour Fourier
quadrature on enlarged image/reciprocal support. There is no matched ORCA or
CRYSTAL reference for these custom finite supports. Libxc values are shared,
not independently qualified by these comparisons; existing tail/arithmetic
acceptance is reused.

| Functional | Spins | Kernel | Range | Component bound | Measured matrix difference | Energy bound (Eh) |
| --- | --- | --- | --- | --- | --- | --- |
| hse06 | 1 | erfc | 0.11 | 1046251.9 | 0.13401541 | 12412937 |
| hse06 | 2 | erfc | 0.11 | 738986.51 | 0.13056508 | 6192629.2 |
| cam-b3lyp | 1 | coulomb | 0 | 1009151.9 | 0.027577675 | 11548222 |
| cam-b3lyp | 1 | erfc | 0.33 | 1009151.9 | 0.025557779 | 11548222 |
| cam-b3lyp | 2 | coulomb | 0 | 726348.03 | 0.027147587 | 5982622.3 |
| cam-b3lyp | 2 | erfc | 0.33 | 726348.03 | 0.025382021 | 5982622.3 |

The diagnostic chooses an explicit fixed L1 family containing the evaluated
SCF density; it does not admit a canonical family before SCF. Nonzero range
kernels deliberately retain conservative Coulomb tail majorants. Bounds do not
certify total Fock or relaxed SCF error. Source construction, factor/rank
perturbation, other components, XC and method correction/assembly remain outside
this component result. Automatic signed range-method composition and useful
accuracy selection remain open under #626/#627. Existing bounded #557 controls,
shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL and symmetry contracts are unchanged.

The initial test reused a fixture without explicit K selection and requested
zero-range RI fits that the existing producer refuses. It now preserves those
guards and tests zero-range limits at the tail source. The complete-method
reconstruction uses the native reciprocal-lattice k frame, preserving exact
identity checks rather than replacing them with numerical tolerance.

Reproduce the complete-method component benchmark with:

```sh
.venv/bin/python -m benchmarks.manta_metric_spectrum --range-exchange-bounds --repetitions 1
```


## Pre-SCF signed physical range exchange admission

Scientific source `1c06da0cdc2c82a83ee3fdd7c6c38480723f8780` uses native SHA256
`f4eb29abdb694507d0a3c12703014ba6f0b0b3c8b0312cf8b2fe18db5aaa1212`. Optional
`PeriodicFittingOptions.range_exchange_accuracy` admits the signed physical
exchange-component sum before range-separated RKS/UKS SCF. It reuses the actual
RI owners, shared canonical spin family, Hermitian probe/tail certificates and
count-only capacities. Each component is scaled outward by the absolute value
of its physical coefficient; coefficients in the evaluated functional remain
unchanged. Reference erfc includes the finite screened zero. Raw child evidence
has no individual target; the combined matrix/energy targets are enforced once.

Local queued jobs `3d2adf894083` and `b58cd6baa15b` qualify
22 converged and eight deliberately stopped selected/control calculations,
plus two strict pre-SCF target refusals. Coverage includes HSE06/CAM-B3LYP
RKS/UKS with standalone and joint J/K policies, mixed physical ranges with
exact J/fitted K at exact native caps, and the accepted PBE0 regression.
All selected/control states are bitwise equal; prior Coulomb source,
certificate and bounds remain bitwise unchanged. All 1942 source
pins and 240 scientific output hashes are verified.
There are 468 passing affected scientific checks and
52 documentation checks, including one-short caps,
family checks, unsupported-policy refusals and stopped-state receipts.

Independent libint/Gaussian Fourier sources, NumPy contractions and shared
libxc give complete HSE/CAM energy/Fock errors of approximately
5.77e-14 Eh/3.04e-15.
Mixed-range independent SCF density error is approximately
6.26e-11.
No matched ORCA/CRYSTAL reference exists for the custom finite supports;
shared libxc is not independently validated. Existing physical-tail, raw-probe
and arithmetic acceptance is reused without repeated qualification.

| Functional | Spin | Matrix bound | Energy bound (Eh) | Iterations |
| --- | --- | --- | --- | --- |
| hse06-rks | restricted | 55.691319 | 17999.494 | 8 |
| hse06-uks | unrestricted | 55.691319 | 17999.494 | 10 |
| cam-b3lyp-rks | restricted | 247.26933 | 79917.715 | 8 |
| cam-b3lyp-uks | unrestricted | 247.26933 | 79917.715 | 11 |

These remain conservative fixed-family bounds. Response/assembly arithmetic,
source/BvK correction, source construction, factor/rank perturbation, XC, other
Fock components and nonlinear SCF relaxation are excluded from the requested
targets. This is automatic exchange-component admission, not useful whole-method
accuracy. #626/#627 and the full #617 matrix remain open; #643 resource failures
and bounded #557 scope are unchanged. Shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL
and source-qualified symmetry authority remain with their existing owners.

The initial queue job completed the twelve HSE/CAM cases before a reporting
script requested a public-result attribute on the native result. The corrected
script uses its native solver snapshot and reuses those verified passing cases;
only remaining mixed-range/control/refusal work is executed in the supplement.
No numerical source, tolerance or product code changed for that correction.

Reproduce the complete-method benchmark with:

```sh
.venv/bin/python -m benchmarks.manta_metric_spectrum --range-exchange-admission --repetitions 1
```


## Positive Fourier-decay groups in exchange reference bounds

Scientific source `da33a7dd74fdc2c52aad091f97fed737ce9d225a` uses native SHA256
`c02be212156d03e5ec555d2d91ab4e931bba9cacac6ff7516eb44d0407c42ffa`. The optional
`fourier_decay_bins` control partitions the original positive primitive
Fourier envelopes by exponent. Cross-group image and reciprocal products use
the sum of their two decay rates. The physical source and calculation are
unchanged. One bin retains the previous values and identities; additional
bins tighten the same reference remainder under explicit memory/work caps.
See scientific contract Section 5.51 for the positive-sum derivation.

Local queued job `2e91e06933fd` passes 23 converged and eight deliberately
stopped calculations, three strict target refusals and four independent
complex-density Fourier remainder cases. Complete methods include RHF/UHF,
PBE0, HSE06 and CAM-B3LYP with independent spins and joint/standalone policies.
All selected/control energies, densities, Focks, coefficients and orbital
energies are bitwise equal. Finite-fit discrepancy bounds and physical RI,
pair and finite-reference identities are unchanged; grouped bound identities
are deliberately distinct. The prior one-bin Coulomb certificate and bounds
remain bitwise equal. All 1942 source pins and
288 scientific output hashes are verified.

There are 485 passing scientific checks and
52 documentation checks. Coverage includes all five
one-short caps before vector payload reads, exact native mixed-range method
caps, empty bins, invalid counts, complex densities, signed contractions,
pure/Cartesian angular functions and finite erfc zero. The initial eight-bin
fixture exposed repeated charging of the shared owner/backend margin. The
planner now reserves these once and explicitly counts only additional group
storage; the same case passes without raising its budget or weakening a check.

Independent libint/Gaussian Fourier/NumPy comparisons give complete energy and
Fock discrepancies of approximately 1.11e-13 Eh
and 4.84e-15. Raw remainder witnesses use
independent shifted-contour Gauss-Hermite polynomial integration and enlarged
image/reciprocal supports. Shared libxc is not independently validated;
there is no matched ORCA/CRYSTAL reference for the custom finite supports.
Existing tail, probe and arithmetic evidence is reused.

| Profile | Single-bin matrix bound | Four-bin matrix bound | Single-bin energy bound (Eh) | Four-bin energy bound (Eh) |
| --- | --- | --- | --- | --- |
| rhf | 222.76471 | 157.44892 | 71997.872 | 17571.105 |
| uhf | 222.76471 | 157.44892 | 71997.872 | 17571.105 |
| pbe0-rks | 222.76516 | 157.45365 | 71997.941 | 17571.174 |
| pbe0-uks | 222.76516 | 157.45365 | 71997.941 | 17571.174 |
| hse06-rks | 55.691319 | 39.363368 | 17999.494 | 4392.8022 |
| hse06-uks | 55.691319 | 39.363368 | 17999.494 | 4392.8022 |
| cam-b3lyp-rks | 247.26933 | 174.77355 | 79917.715 | 19504.004 |
| cam-b3lyp-uks | 247.26933 | 174.77355 | 79917.715 | 19504.004 |

The fixed HSE06 component targets 50 matrix / 5000 Eh pass with four bins and
refuse with one. These are deliberately loose component targets that witness
strict improvement; they do not establish chemically useful precision.
Construction, correction, response/assembly arithmetic, rank, XC, other Fock
components and SCF relaxation retain their exclusions. Useful automatic
whole-method accuracy and the remaining #617 matrix are unfinished. Existing
shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL and symmetry authority is unchanged;
real #643 failures and bounded #557 scope are preserved.

Reproduce the complete-method benchmark with:

```sh
.venv/bin/python -m benchmarks.manta_metric_spectrum --decay-group-admission --repetitions 1
```


## Conditional metric assembly rank guard

Implementation `e1e80ed6`, qualified source `7b42701d`. The optional
`PeriodicFittingOptions.metric_assembly_rank` guard passed eleven complete
calculations: five audit-enabled HF/KS states reuse accepted physical-state
and reference receipts; three fitted-exchange states have paired controls.
All converge. Energy, density and Fock identities match the accepted or
paired states bitwise. Nineteen exported metric-channel receipts qualify,
including non-Gamma transfer owners and physical erf/erfc kernels. This is
conditional rank stability under stored-operand assembly roundoff, not a
bound on inverse response or total relaxed energy.

| Profile | Iterations | Energy / Eh | Wall seconds | Largest metric perturbation bound | Smallest strict radius |
| --- | --- | --- | --- | --- | --- |
| rhf-Be | 9 | -8.745246181092 | 0.654875 | 6.3183879e-14 | 0.00080101903 |
| uhf-H | 10 | 0.092952292537 | 0.314093 | 6.3183879e-14 | 0.00080101903 |
| rks-Be | 9 | -9.680726572313 | 0.561135 | 8.8128819e-14 | 0.00075905603 |
| pbe0-Be | 9 | -9.479322790839 | 0.608533 | 8.8128819e-14 | 0.00075905603 |
| hse06-Li | 10 | -4.178171904827 | 0.644550 | 8.8128819e-14 | 0.00075905603 |
| fitted-rhf | 9 | -9.615420772487 | 3.605708 | 7.8800022e-14 | 0.00080101921 |
| fitted-hse06-uks | 10 | -4.284658734208 | 3.705230 | 1.9797173e-13 | 0.00075905603 |
| fitted-cam-b3lyp-rks | 8 | -9.672681162937 | 6.541460 | 1.0281958e-13 | 0.00075905603 |

The five first profiles use AO image radius 10 bohr, auxiliary radius 12.1
bohr and split 0.65 inverse bohr. The fitted profiles use AO radius 3 bohr,
auxiliary radius 12 bohr for RHF and 8 bohr for KS. All use reciprocal cutoff
6, nuclear radius 4 bohr, Ewald alpha 0.6 and a shifted 2x1x1 mesh. KS retains
the accepted continuous grid: 12 radial points, Lebedev order 17, radius 8
bohr and XC image radius 11 bohr. Old-grid and automatic-selection results
are not substituted for these explicit configurations. The fixtures record
the actual cell, custom basis contractions, spin, thresholds and source IDs.

The complete validation command took 30.917928 s wall and 30.652510 s CPU,
with 598.30 MiB peak RSS. Table timings include each audited calculation,
not paired controls or references. Single observations on small fixtures
establish neither production-size scaling nor speedup. The reproducer is
`manta_metric_spectrum.py --metric-assembly --repetitions 1` for the first
five profiles; fitted cases reuse `exchange_admission_job` and
`range_admission_job` with the flag enabled.

The accepted independent RHF Gaussian/libint/Fourier/NumPy construction is
reused through matching state identities: Fock error 4.75e-15 and energy
error 8.80e-14 Eh. Independent fitted RHF/HSE06/CAM-B3LYP source contractions
have maximum Fock error 4.85e-15 and energy error 1.12e-13 Eh. KS references
share libxc; no matched external ORCA/CRYSTAL custom finite-domain result is
available. Decimal stored-operand reconstruction spans coefficient scales
1e-30, 1 and 1e30. Strict cutoff refusal, exact and one-short resource caps,
and all three physical kernels pass. All 132 source checks, 104 public job,
preparation/documentation checks and 118 shared solver checks pass.

Actual qualified native SHA256:
`ae9e8d651a1e3a28d4dc7657204ce817f14e23d406fc0a3dbc7fddc53724f5b0`.
This records a qualified source/runtime, not a release or deployed candidate.
Primitive/Fourier/phase/kernel-weight evaluation, screening, omitted domains,
retained-space rotation, inverse response, XC/grid, basis/BZ and relaxed energy
remain excluded. MANTA remains experimental; #623/#625/#617 remain open.


## Finite retained-space and inverse response to metric assembly error

Implementation `49b10c64`, qualified tree `86d31ddc`. The existing metric
assembly option now records the shared finite response certificate from
contract 5.56. Ten complete calculations converge: eight reuse accepted
physical-state/reference receipts and a partial-rank RHF case has a paired
control. All physical energies, densities, Fock/source/factor identities and
previous stored inverse certificates remain unchanged. Twenty channel
responses qualify, including non-Gamma and physical erf/erfc owners.

| Profile | Rank/dimension | Largest projector response | Largest inverse response | Largest stored-factor-plus-source bound | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| rhf-Be | 3/3 | 0 | 1.9694747e-07 | 2.0257372e-07 | 0.583851 |
| uhf-H | 3/3 | 0 | 1.9694747e-07 | 2.0257372e-07 | 0.295474 |
| rks-Be | 3/3 | 0 | 3.0591442e-07 | 3.1612966e-07 | 0.494563 |
| pbe0-Be | 3/3 | 0 | 3.0591442e-07 | 3.1612966e-07 | 0.549041 |
| hse06-Li | 3/3 | 0 | 3.0591442e-07 | 3.1612966e-07 | 0.609612 |
| fitted-rhf | 3/3 | 0 | 2.195825e-07 | 2.2654524e-07 | 3.188800 |
| fitted-hse06-uks | 3/3 | 0 | 6.0051763e-07 | 6.2047381e-07 | 3.346511 |
| fitted-cam-b3lyp-rks | 3/3 | 0 | 3.4421395e-07 | 3.5225365e-07 | 6.227179 |
| partial-rank-rhf | 2/3 | 5.0922141e-12 | 1.9878342e-10 | 2.0454815e-10 | 0.512272 |

The first eight configurations, supports and continuous grids are exactly
those in the preceding assembly section. The partial-rank case uses the
same RHF fixture with absolute metric cutoff 0.01 and relative cutoff zero;
it converges in nine iterations to -8.934797433221 Eh and retains two of
three auxiliary directions. Projector bounds compare exact spectral
projectors, not rounded exported eigenvector products. Inverse bounds have
inverse-metric units, not energy units. Full-rank projectors have zero
rotation because they remain the identity; inverse response is nonzero.

The command took 17.194207 s wall, 17.067545 s CPU and 595.61 MiB peak RSS.
Table timings include each selected calculation, excluding the extra control
and test overhead. These small-fixture observations establish no speedup or
production-size scaling. The existing `--metric-assembly` reproducer also
checks the new response flags; the reduced-rank control changes only the
explicit cutoff as above.

Independent 250-digit Decimal two-by-two inverses/projectors cover empty,
partial and full rank at scales 1e-80, 1 and 1e80, with a deliberately
perturbed stored factor. Clustered complex spectra and moving relative
cutoffs use independent eigensolves. A 400-digit scalar witness checks the
smallest positive subnormal perturbation; equality/crossing and exact or
one-short resource limits are tested. All 136 focused source checks pass,
including the 19 response/periodic checks run inside the validation job;
104 public preparation/job/documentation checks also pass.

Matching source, factor, state and archive identities reuse the previous
Gaussian/libint/Fourier/NumPy RHF and fitted-exchange acceptance. The new
partial-rank complete-method comparison is a fixed-factor control, with its
response independently checked by the analytic spectral witnesses. No
matched ORCA/CRYSTAL custom finite-domain comparison is available; the KS
reference still shares libxc. Primitive/Fourier/phase/weight construction,
missing domains, screening, coupling changes, J/K/Fock/energy propagation,
overlap/occupied-space response and total relaxed energy remain excluded.
MANTA remains experimental and #623/#625/#617 stay open.

Actual qualified native SHA256:
`b2037f7921184d4b29cafc88fe72e7a52768b229e4094380924b8704a67a194d`.
This is source/runtime qualification, not release or deployment evidence.


## Conditional metric-source contribution to Coulomb action targets

Implementation `a7a00f5a`, qualified tree `bdbe3132`. Explicit
`metric_action_include_assembly=True` consumes the shared finite inverse
response and the existing stored-coupling column norms (contract 5.57).
It requires `metric_assembly_rank=True` and positive action targets.
The existing stored-factor action bound and the source-only response are
added outward; the stored-factor error is counted once. This is a
fixed-density-family bound conditional on the stored source operands.
It leaves the full-Coulomb fitting certificate and global component ledger
unchanged because those have different reference objects.

Eleven complete calculations converge: three fitted profiles reuse their
accepted physical states and references, and four selected/control pairs
exercise explicit, local compressed, global compressed and pair-tree
couplings. All compared energies, densities, Fock and source identities are
bitwise unchanged. The new assembly flag is explicit; this is not evidence
of automatic useful-accuracy selection.

| Profile | Iterations | Energy / Eh | Stored J bound | Assembly J bound | Combined energy bound / Eh |
| --- | --- | --- | --- | --- | --- |
| fitted RHF | 9 | -9.615420772487335 | 5.2374840e-13 | 6.2959716e-06 | 1.59880325e-03 |
| fitted HSE06 UKS | 10 | -4.284658734208046 | 2.9199316e-13 | 1.0655462e-05 | 2.76036733e-03 |
| fitted CAM-B3LYP RKS | 8 | -9.672681162936970 | 2.9199316e-13 | 1.0655462e-05 | 2.76036733e-03 |
| local compressed RHF | 9 | -8.745244727379031 | 1.7985871e-07 | 6.2959716e-06 | 1.64447651e-03 |
| global compressed RHF | 9 | -8.745244727379037 | 1.7985871e-07 | 6.2959716e-06 | 1.64447651e-03 |
| global pair-tree UHF | 13 | -3.676437098543107 | 1.7985871e-07 | 6.2959716e-06 | 1.64447651e-03 |
| explicit pure RKS | 9 | -9.680725483733205 | 2.9199316e-13 | 1.0655462e-05 | 2.76036733e-03 |

The three fitted configurations, physical supports and continuous XC grids
are identical to the preceding assembly/response acceptance. The four
paired configurations use `source_action_job` and its existing
`retained_metric_job` / `compressed_metric_job` fixtures. Compressed cases
use order zero, coupling tolerance 1, hierarchy enabled, and the indicated
global/pair-tree controls. All action targets are deliberately loose at
1e30, while full-Coulomb fitting targets remain 1e8. These admission controls
do not demonstrate chemically useful whole-method accuracy. Their exact
cell, custom basis, density limits, physical supports and source identities
are retained in the calculation receipts. No old-grid result is substituted.

The validation command took 17.806053 s wall and 17.674577 s CPU, with
578.36 MiB peak RSS. The queue's whole-process receipt measured 21.062373 s
wall and 18.660737 s CPU, including command startup. These are small-fixture
cost observations, not speedup or scaling evidence. The benchmark option
`--metric-source-action --repetitions 1` reproduces five basic paired
configurations; the recorded acceptance instead uses the four pairs above
and three previously accepted fitted fixtures.

All 223 focused source checks and 104 public job, preparation and
documentation checks pass. Seventeen source-action checks also pass inside
the validation job, including independent 180-digit Decimal fixed-coupling
J/energy witnesses for empty, partial and full rank, component density
limits, exact/one-short resource caps and target boundaries. Missing guards
and missing positive targets refuse. A complete-method request with action
targets 1e-100 refuses before an accepted calculation. Unresolved initial
rank remains a strict refusal; rank-boundary recovery is not added.

The fitted states reuse accepted Gaussian/libint/Fourier/NumPy reference
comparisons: maximum Fock error 4.85e-15 and energy error 1.12e-13 Eh.
The compressed and explicit new pairs are fixed-source controls; the new
bound itself is checked against the independent Decimal witnesses. No
matched external ORCA/CRYSTAL custom finite-domain reference is available.
KS references share libxc and do not independently validate XC.

Actual qualified native SHA256:
`e5cfb110caf30a82ce0fd532d7b19bbedb76f444b02165f41bbc60ca935b32ad`.
Source, dependency, native and output hashes were checked. This qualifies
that source/runtime, not a deployed release. Primitive/Fourier/phase/weight
evaluation, coupling construction, metric screening and omitted domains,
K source response, executed Fock/energy arithmetic, overlap/occupied-space
response, XC/grid, basis/BZ error and total relaxed energy remain excluded
from this component. MANTA stays experimental; #623/#625/#627/#617 remain open.


## Conditional metric-source contribution to fitted exchange

Implementation `88d8b48d`, qualified tree `ab2790ba`. Contract 5.58 adds
separate raw K component targets for the conditional metric-assembly inverse
response. It preserves physical states, the original fitting certificate,
signed range aggregation and the global ledger. The shared norm producer
uses a conservative complex/auxiliary L1 envelope, bounding Euclidean column
norms. Every actual Gamma and ordered transfer owner supplies its own
response, kernel and weight; no channel is inferred from another.

Nine public calculations converge: three fitted states reproduce accepted
source, factor, density/Fock and energy receipts, and three selected/control
pairs cover UHF and both PBE0 spin variants. Seven raw physical-component
receipts qualify, covering 21 metric channels. The CAM-B3LYP rows below are
its separate physical children before signed method coefficients.

| Profile / raw kernel | Iterations | Energy / Eh | K entry bound | Component energy bound / Eh | Selected wall seconds |
| --- | --- | --- | --- | --- | --- |
| fitted RHF / Coulomb | 9 | -9.615420772487335 | 6.8277603e-06 | 2.1808688e-03 | 3.254010 |
| fitted HSE06 UKS / erfc | 10 | -4.284658734208046 | 4.2506993e-05 | 1.3602598e-02 | 3.173423 |
| fitted CAM-B3LYP RKS / Coulomb | 8 | -9.672681162936970 | 1.1537506e-05 | 3.6777476e-03 | 6.144256 |
| fitted CAM-B3LYP RKS / erfc | 8 | -9.672681162936970 | 1.1140471e-05 | 3.5504835e-03 | same calculation |
| fitted UHF / Coulomb | 16 | -4.094056282195062 | 6.8277603e-06 | 2.1808688e-03 | 3.206495 |
| fitted PBE0 RKS / Coulomb | 8 | -9.700417784506898 | 1.1537506e-05 | 3.6777476e-03 | 3.150546 |
| fitted PBE0 UKS / Coulomb | 10 | -4.145245061675722 | 1.1537506e-05 | 3.6777476e-03 | 3.183922 |

The first three configurations reuse the exact physical supports, custom
bases and continuous grids of the preceding accepted fitted profiles, with
J metric-source admission still enabled. The other three use
`metric_exchange_job`: the existing fitted-exchange fixture, explicit
`metric_assembly_rank=True`, and paired raw metric-source targets of 1e20.
Existing exchange fitting targets remain 1000/1e6; the separate Coulomb
policy is unchanged. These deliberately loose controls exercise admission,
not automatic useful accuracy. All inputs, physical domains, density families,
source identities and outputs are retained in the hashed receipts. Old-grid
results are not substituted for these continuous-grid configurations.

All 76 focused source checks pass. Seven 180-digit Decimal witnesses use
actual stored coupling tiles and independently perturbed canonical metrics;
they cover unequal and zero weights, non-Gamma transfers, full/partial/empty
rank and Coulomb/erf/erfc kernels. They check observed K/energy response
against the component enclosure and independently reconstruct its positive
coefficient bound. Exact and one-short resource limits and target boundaries
pass. Fourteen checks also run inside the calculation job, including a
complete-method 1e-100 target refusal and missing-proof/invalid-target
refusals. Public preparation/job and documentation checks pass after fixing
the new changelog fragment's required bullet prefix (103 original passing
checks and all 15 final documentation checks).

Accepted Gaussian/libint/Fourier/NumPy reference receipts are reused for the
three unchanged fitted states: maximum Fock error 4.85e-15 and energy error
1.12e-13 Eh. New UHF/PBE0 comparisons are fixed-source complete-method
controls; the new component has independent Decimal perturbation witnesses.
No matched external ORCA/CRYSTAL custom finite-domain reference is available;
KS references share libxc. This does not establish bulk CRYSTAL parity.

The complete validation command took 33.232373 s wall, 33.104205 s CPU and
602.81 MiB peak RSS. Per-profile timings above exclude controls and test
overhead. `--metric-source-exchange --repetitions 1` reproduces six paired
profiles; accepted unchanged fitted references need not be recomputed.
No speedup or production-size scaling follows from these observations.

Actual qualified native SHA256:
`7b08b7f6eb93f215643b18bae234332b59e7eb9d025ae77c051bcac7c5129228`.
Source/native/dependency/output hashes were checked before and after the job.
This is qualified source/runtime evidence, not a deployed release. Primitive,
Fourier/phase/weight and coupling construction, metric domain/screening,
stored-factor and executed K/Fock/energy arithmetic, BvK correction, XC/grid,
overlap/occupied-space response and total relaxed energy remain outside this
component. MANTA remains experimental; #626/#625/#623/#617 remain open.


## Short-range metric Bloch-phase construction bound

Source and qualified tree `2425cd61` extend the optional assembly/rank audit
with the exact short-range metric phase on stored momentum and translation
vectors (contract 5.59). The proof includes dot-product and transcendental
evaluation error. It uses outward Taylor/Lagrange intervals and binary angle
doubling without assuming correctly rounded libm. The numerical phases,
physical metric and SCF path are unchanged. Other source-construction errors
remain excluded; this is one bounded increment of the experimental programme.

Six complete calculations converge and exactly reuse the preceding accepted
six selected states: energy, iteration count, physical source and final
density/Fock identities, QVF spin densities, original full-Coulomb fitting
certificate and global component ledger. The exact inputs and physical
supports equal those of the preceding section. This includes the current
continuous-grid KS configurations, explicit profiles and deliberately loose
raw K targets of 1e20; J metric-source admission remains on for the three
`fitted-` profiles. This is not evidence for automatic profile selection.
Twenty-three actual metric-owner receipts qualify, including 21 K channels
and the separate Coulomb owners in the range methods. Each records
`assembly_rank.phase_included=True`; the expanded proof changes the audit
bounds and identities, not the physical states.

| Profile / raw kernel | Iterations | Energy / Eh | K entry bound | Component energy bound / Eh | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| fitted-rhf / Coulomb | 9 | -9.615420772487335 | 6.82777105e-06 | 2.18087222e-03 | 3.251702 |
| fitted-hse06-uks / erfc | 10 | -4.284658734208046 | 4.25070491e-05 | 1.36026154e-02 | 3.164084 |
| fitted-cam-b3lyp-rks / Coulomb | 8 | -9.672681162936970 | 1.15375622e-05 | 3.67776559e-03 | 6.130153 |
| fitted-cam-b3lyp-rks / erfc | 8 | -9.672681162936970 | 1.11405126e-05 | 3.55049680e-03 | 6.130153 |
| uhf / Coulomb | 16 | -4.094056282195062 | 6.82777105e-06 | 2.18087222e-03 | 3.235263 |
| pbe0-rks / Coulomb | 8 | -9.700417784506898 | 1.15375622e-05 | 3.67776559e-03 | 3.128147 |
| pbe0-uks / Coulomb | 10 | -4.145245061675722 | 1.15375622e-05 | 3.67776559e-03 | 3.137369 |

The CAM-B3LYP entries are separate physical children before signed method
coefficients; the same calculation time appears on both rows. Across the
23 metric owners, perturbation bounds range from 6.3184e-14 to 1.9798e-13,
and conditional inverse-response bounds from 1.9694e-7 to 6.0052e-7.
These are component statements for fixed stored couplings and declared
density families, not bounds on total relaxed energy.

Thirty-five focused phase/assembly checks pass, including independent
800-digit AGM pi, argument reduction and scalar trigonometric series.
Witnesses cover cancellation, underflow, very large angles, pi neighbours,
arbitrary returned phases and exact Gamma. Six independently contracted
stored-integral sums cover ordinary/large momenta and signed coefficients
scaled by 1e-30, 1 and 1e30. Audit-on/off metric values agree bitwise.
Exact/one-short resource limits and nonfinite/overflow refusals pass.
The complete local calculation job passes 36 checks, including those 35
and complete-method missing-proof, invalid-target and 1e-100 target refusals.
Separate source (91), J/K consumer (30), and public preparation/documentation
(104) checks pass; these groups overlap and are not a unique-test total.

The first three states reuse their accepted Gaussian/libint/Fourier/NumPy
reference comparisons (maximum Fock error 4.85e-15 and energy error
1.12e-13 Eh). UHF/PBE0 reuse accepted fixed-source controls. The new phase
contribution has independent high-precision witnesses; the stored primitive
integrals themselves are not validated by the phase reconstruction. There
is no matched ORCA/CRYSTAL custom finite-domain reference; KS comparisons
share libxc. No bulk CRYSTAL parity or general speedup follows.

The harness took 23.754112 s wall, 23.615582 s CPU and 581.69 MiB peak RSS;
the queue measured 25.038255 s wall and 24.576410 s CPU for the process.
Per-profile times exclude the focused checks and startup. Reproduce the
current six paired controls with `--metric-source-exchange --repetitions 1`
and the phase checks with `tests/test_manta_fourier.py` and
`tests/test_manta_periodic_gdf_tiles.py`, selecting `bloch_phase or metric_assembly`.
Unchanged accepted reference calculations need not be repeated.

Actual qualified native SHA256:
`b62b8761b48e28ba53b42e8335abf5010ba4f9600b23bd0ad8bc540b82b15248`.
Source/native/dependency/reused-output pins agree before and after the job,
and all generated output hashes were verified after retrieval. This receipt
qualifies the stated source/runtime, not a later deployed release.
Primitive/Boys/radial/contraction/pure-transform evaluation, Fourier/coupling
phases, weights, q/R generation, source domains/screening, coupling
construction, stored-factor/executed arithmetic, BvK correction, XC/grid,
overlap/occupied-space and density relaxation remain separate. The overall
source-construction-complete flag stays false. Shared Fock, PBC/BIPOLE and
symmetry contracts remain unchanged; #623/#625/#617 remain open.


## Reciprocal metric weight construction bound

Source/qualified tree `313d1e74` extends the preceding phase audit with the
reciprocal metric weight proof of contract 5.60. Exact stored momentum/range
values and stored reciprocal prefactor define the conditional reference.
Squared norms, range squares, divisions, exp evaluation and cancellation
are enclosed; pi/cell-volume/prefactor generation and Fourier panels remain
conditional. Physical weights and SCF states are unchanged. The broader
reference produces larger bounds; this is increased coverage, not a claim
that the observed energy became more accurate.

Six complete calculations converge. All six exactly reproduce the prior
selected energy, iterations, source/final-density/final-Fock identities,
QVF spin densities, full-Coulomb certificate and component ledger. Profile
inputs and physical supports equal those of the preceding accepted phase
receipt, including current continuous grids. Explicit profiles retain raw
K targets 1e20 and existing fitting targets; the first three fitted profiles
retain J source-action admission. No automatic-selection or old-grid result
is substituted. All 23 metric owners report qualified rank and inverse
response with both phase and weight included; 21 belong to raw K components.

| Profile / raw kernel | Iterations | Energy / Eh | K entry bound | Component energy bound / Eh | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| fitted-rhf / Coulomb | 9 | -9.615420772487335 | 1.26932593e-05 | 4.05428278e-03 | 4.443554 |
| fitted-hse06-uks / erfc | 10 | -4.284658734208046 | 6.76318936e-05 | 2.16385020e-02 | 4.288444 |
| fitted-cam-b3lyp-rks / Coulomb | 8 | -9.672681162936970 | 2.34810335e-05 | 7.48480934e-03 | 8.062569 |
| fitted-cam-b3lyp-rks / erfc | 8 | -9.672681162936970 | 2.78127706e-05 | 8.86390431e-03 | 8.062569 |
| uhf / Coulomb | 16 | -4.094056282195062 | 1.26932593e-05 | 4.05428278e-03 | 4.243850 |
| pbe0-rks / Coulomb | 8 | -9.700417784506898 | 2.34810335e-05 | 7.48480934e-03 | 4.264282 |
| pbe0-uks / Coulomb | 10 | -4.145245061675722 | 2.34810335e-05 | 7.48480934e-03 | 4.269489 |

CAM-B3LYP rows are separate physical children before signed coefficients,
with the same calculation time repeated. Metric perturbation bounds span
1.2282e-13 to 2.9492e-13; conditional inverse-response bounds span
3.8283e-7 to 8.9458e-7. These remain fixed-coupling component statements,
not total relaxed-energy budgets or useful automatic whole-method accuracy.

All 77 focused weight/phase/assembly checks pass. The 42 new checks include
250-digit Decimal exponential/rational witnesses for Coulomb/erf/erfc,
finite-zero terms, equal and adjacent ranges, both signs of cancellation,
singular/nonsingular momenta, underflow, large arguments and shifted returned
witnesses. Signed stored-Fourier Gram reconstructions use coefficient scales
1e-30, 1 and 1e30. Exact/one-short resource caps, unresolved pole boundaries,
invalid inputs and unrepresentable denominators refuse. Existing finite-zero
contraction witnesses now use the expanded conditional weight reference.
The local calculation job passes 78 checks, including a complete-method
1e-100 target refusal. Separate source (133), J/K consumer (30) and public
preparation/documentation (104) checks pass; these groups overlap.

Accepted Gaussian/libint/Fourier/NumPy references are reused for the first
three states (maximum Fock error 4.85e-15, energy error 1.12e-13 Eh).
UHF/PBE0 reuse accepted fixed-source controls. Decimal witnesses independently
check the new weight component, not the primitive/Fourier values. No matched
ORCA/CRYSTAL custom finite-domain reference is available; KS references share
libxc. No bulk parity or general speedup is established.

Harness cost is 31.894397 s wall, 29.628515 s CPU and 584.59 MiB peak RSS.
The queue process receipt records 33.187914 s wall and 30.775280 s CPU.
The existing `--metric-source-exchange --repetitions 1` reproducer exercises
the expanded proof; select `metric_weight or metric_assembly or bloch_phase`
in the GDF/Fourier tests for its scalar and contracted witnesses. Reuse
unchanged accepted references instead of recomputing them.

Actual qualified native SHA256:
`244d418945ba780045898b8bce71aa706525802fa39ece6fb6c534f8e2ba93d9`.
Source/native/dependency/reused-output pins match before and after the job;
retrieved outputs also match their hashes. This identifies qualified source
and runtime, not a deployed release. Primitive/Boys/radial/contraction/pure
transforms, Fourier/coupling construction, pi/volume/prefactor and vector
generation, domains/screening, stored-factor/executed arithmetic, BvK, XC/grid,
basis/BZ, overlap/occupied-space/density relaxation and total relaxed energy
remain separate. The complete-source flag stays false. Shared Fock/PBC/
BIPOLE/symmetry contracts are preserved; MANTA and #617 remain experimental.


## Lattice reciprocal metric prefactor bound

Source/qualified tree `9f7d54f0` removes the stored-prefactor premise from
the metric audit (contract 5.61). The reference includes mathematical pi and
the determinant of the exact stored lattice, propagated through the shared
weight, rank/inverse and optional J/raw-K producers. The numerical source,
factors and SCF path stay unchanged. Lattice/vector generation and primitive/
Fourier/coupling construction remain outside this component.

Six complete calculations converge and reproduce accepted energy, iterations,
physical source/final density/final Fock identities, QVF spin densities,
original full-Coulomb certificate and component ledger exactly. All inputs
and physical supports equal the preceding accepted weight receipt, including
current continuous grids. Explicit profiles keep raw K targets 1e20 and the
original fitting policies; the first three fitted profiles retain J source
admission. This does not qualify automatic selection or old-grid results.
All 23 actual metric-owner receipts qualify with phase, weight and prefactor
included; 21 are K channels and two are separate range-method J owners.

| Profile / raw kernel | Iterations | Energy / Eh | K entry bound | Component energy bound / Eh | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| fitted-rhf / Coulomb | 9 | -9.615420772487335 | 1.35714203e-05 | 4.33480365e-03 | 3.690878 |
| fitted-hse06-uks / erfc | 10 | -4.284658734208046 | 7.78015443e-05 | 2.48931493e-02 | 3.671964 |
| fitted-cam-b3lyp-rks / Coulomb | 8 | -9.672681162936970 | 2.55583303e-05 | 8.14720090e-03 | 6.752700 |
| fitted-cam-b3lyp-rks / erfc | 8 | -9.672681162936970 | 2.89610214e-05 | 9.22985290e-03 | 6.752700 |
| uhf / Coulomb | 16 | -4.094056282195062 | 1.35714203e-05 | 4.33480365e-03 | 3.763792 |
| pbe0-rks / Coulomb | 8 | -9.700417784506898 | 2.55583303e-05 | 8.14720090e-03 | 3.684623 |
| pbe0-uks / Coulomb | 10 | -4.145245061675722 | 2.55583303e-05 | 8.14720090e-03 | 3.537035 |

The CAM-B3LYP rows are separate physical children before signed coefficients,
with the same calculation time repeated. Metric perturbation bounds span
1.2952e-13 to 3.4356e-13 and inverse-response bounds span 4.0374e-7 to
1.0422e-6. These larger component enclosures cover more construction error;
they do not indicate any change in numerical energy accuracy or establish
useful automatic whole-method accuracy.

All 116 focused prefactor/weight/phase/assembly checks pass. Thirty-nine new
checks compare bounded Machin pi and interval determinants with independent
AGM pi and Decimal determinants for all three physical kernels and both
zero conventions. Cases cover skew cells, reversed orientation, cancellation
and lattice scales from 1e-100 to 1e100; unresolved volume signs, nonfinite
inputs, overflow and exact/one-short resource caps refuse. Existing signed
stored-Fourier Gram and finite-zero contraction witnesses now use the true
pi and lattice-determinant reference. The local calculation job passes 117
checks including the complete-method strict target refusal. Separate source
(172), J/K consumer (30) and public preparation/documentation (104) checks
pass; these groups overlap.

The first three states reuse accepted Gaussian/libint/Fourier/NumPy reference
comparisons (maximum Fock error 4.85e-15, energy error 1.12e-13 Eh).
UHF/PBE0 reuse accepted fixed-source controls. New high-precision witnesses
validate the prefactor contribution, not primitive/Fourier accuracy. There
is no matched ORCA/CRYSTAL custom finite-domain reference; KS references
share libxc. No bulk CRYSTAL parity or general speedup is demonstrated.

Harness cost is 27.216897 s wall, 26.950769 s CPU and 583.88 MiB peak RSS;
the queue process measured 28.173710 s wall and 28.044980 s CPU. The existing
`--metric-source-exchange --repetitions 1` reproducer exercises this proof.
Select `metric_prefactor or metric_weight or metric_assembly or bloch_phase`
in the existing GDF/Fourier tests for scalar/contracted witnesses. Reuse
unchanged accepted reference calculations.

Actual qualified native SHA256:
`1017eba02ec5ce0072ae88eaf3df263c1ea09ba2d4c2fff8def332ce178e7e12`.
Source/native/dependency/reused-output pins agree before and after the job,
and fetched outputs match their hashes. This identifies qualified source
and runtime, not a deployed release. Lattice/vector generation, primitive/
Boys/radial/contraction/pure transforms, Fourier/coupling construction,
domains/screening, stored-factor/executed arithmetic, BvK, XC/grid, basis/BZ,
overlap/occupied-space/density relaxation and total relaxed energy remain
separate. The complete-source flag remains false. Shared Fock/PBC/BIPOLE/
symmetry contracts are preserved; #623/#625/#617 stay open and experimental.


## Auxiliary Fourier construction bounds

Source/qualified tree `cd9b5002` extends the optional metric audit to exact
auxiliary Gaussian Fourier evaluation (contract 5.62), conditional on stored
basis and representation coefficients and stored vectors. The existing
numerical Fourier producer is unchanged. Its independent interval moment
recurrence encloses primitive normalization seed, decay, center phase,
signed contraction and arithmetic applying the stored pure transform.
Generation of normalized basis/representation data remains excluded.

Six complete calculations reproduce the preceding accepted prefactor states:
energy, iterations, source/final density/final Fock identities, QVF spin
densities, full-Coulomb certificate and component ledger agree exactly.
Inputs and physical supports also agree exactly, including current continuous
grids. Explicit profiles retain original fitting policies and raw K targets
1e20; the first three fitted profiles retain J source admission. Automatic
selection and old-grid results are not qualified by this evidence. All 23
metric-owner receipts include the auxiliary Fourier construction, phase,
weight and prefactor bounds and qualify rank/inverse response. There are 21
K channels and two additional range-method J owners.

| Profile / raw kernel | Iterations | Energy / Eh | K entry bound | Component energy bound / Eh | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| fitted-rhf / Coulomb | 9 | -9.615420772487335 | 1.92127811e-05 | 6.13682628e-03 | 3.335209 |
| fitted-hse06-uks / erfc | 10 | -4.284658734208046 | 1.21436325e-04 | 3.88542565e-02 | 3.315189 |
| fitted-cam-b3lyp-rks / Coulomb | 8 | -9.672681162936970 | 3.92716674e-05 | 1.25191002e-02 | 6.228361 |
| fitted-cam-b3lyp-rks / erfc | 8 | -9.672681162936970 | 3.86848691e-05 | 1.23288387e-02 | 6.228361 |
| uhf / Coulomb | 16 | -4.094056282195062 | 1.92127811e-05 | 6.13682628e-03 | 3.244463 |
| pbe0-rks / Coulomb | 8 | -9.700417784506898 | 3.92716674e-05 | 1.25191002e-02 | 3.189936 |
| pbe0-uks / Coulomb | 10 | -4.145245061675722 | 3.92716674e-05 | 1.25191002e-02 | 3.186942 |

CAM-B3LYP children are physical components before signed coefficients; their
shared calculation time is repeated. Metric perturbation bounds span
1.7598e-13 to 5.3556e-13 and inverse-response bounds span 5.4854e-7 to
1.6246e-6. Expanded construction coverage increases conservative component
bounds without changing the numerical states. These loose component budgets
do not establish useful automatic whole-method or total relaxed-energy
accuracy.

Fifty-eight new construction checks pass. An independent 250-digit closed
finite moment sum uses factorial coefficients, AGM pi, Decimal sqrt/exp and
the earlier independent phase oracle rather than the native recurrence.
It covers Cartesian/pure l=0..7, signed coefficient scales 1e-30 to 1e30,
diffuse/tight primitives, deep cancellation and representable s-shell
exponents 1e-200 and 1e200. Evaluated panels equal the existing producer
bitwise. Exact/one-short storage, scalar, kernel, shell and primitive caps
and nonfinite input refusals pass. Representation data are shared conditional
operands, not an independent validation of their generation.

The local calculation job passes 175 construction/prefactor/weight/phase/
assembly checks, including complete-method strict target refusal. Separate
source (297) and J/K consumer (30) checks pass. Public preparation and docs
checks report 103 passes and one unrelated existing documentation-test
failure: the CONTRIBUTING navigation fixture expects the obsolete label
`testing policy and inventory`, while the current link says `testing policy`.
This does not affect the MANTA inputs or calculations. Check groups overlap.

The first three states reuse accepted Gaussian/libint/Fourier/NumPy reference
comparisons (maximum Fock error 4.85e-15, energy error 1.12e-13 Eh); UHF/PBE0
reuse accepted fixed-source controls. No suitable matched ORCA/CRYSTAL
custom finite-domain reference is available. KS references share libxc;
new Decimal witnesses isolate the auxiliary construction contribution.
There is no bulk CRYSTAL parity or general speedup claim.

Harness cost is 25.168816 s wall, 25.029990 s CPU and 583.16 MiB peak RSS;
the queue process measured 25.891201 s wall and 25.882443 s CPU. Reproduce
with the existing `--metric-source-exchange --repetitions 1` benchmark and
select `auxiliary_fourier_construction or metric_assembly` in the Fourier
and GDF tests. Reuse unchanged accepted independent reference calculations.

Actual qualified native SHA256:
`edf14fcc5fe61b1702e0db94a1bcc11cf010a046dcf5d14708978522d5ff4e58`.
Source/native/dependency/reused-output pins match before and after the job;
fetched outputs are hash-verified. This identifies qualified source/native,
not a deployed release. Short-range integral and AO-pair Fourier/coupling
construction; basis/representation and lattice/vector data generation;
domains/screening; stored-factor/executed arithmetic; BvK, XC/grid, basis/BZ;
overlap/occupied-space/density relaxation and total relaxed energy remain
separate. Complete-source remains false. Shared Fock/PBC/BIPOLE/symmetry
contracts are preserved; #623/#625/#617 remain open and experimental.


## Two-center metric integral construction bounds

Source/qualified tree `cb02db0d` extends the optional metric audit to
short-range two-center integral construction (contract 5.63): radial seeds,
primitive prefactors, center-plus-image addition, Hermite contraction and
arithmetic applying stored pure transforms. The same proof is available for
Coulomb/erf/erfc through auxiliary l=7. Numerical MD integrals and physical
source/factor identities are unchanged. Exact stored basis/representation
coefficients and image/vector data define the reference; their generation,
coupling construction and domains/screening remain outside this component.

Six complete calculations reproduce the accepted Fourier-construction states
exactly: energy, iterations, physical source/final density/final Fock IDs,
QVF spin densities, original full-Coulomb certificate and component ledger.
Inputs and physical supports, including current continuous grids, are equal.
Explicit profiles retain original fitting policies and loose raw K targets
1e20; the first three fitted profiles retain J source admission. This does
not qualify automatic selection or substitute old-grid evidence. All 23
metric-owner receipts qualify rank/inverse response with integral, Fourier,
phase, weight and prefactor coverage: 21 K channels and two range J owners.

| Profile / raw kernel | Iterations | Energy / Eh | K entry bound | Component energy bound / Eh | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| fitted-rhf / Coulomb | 9 | -9.615420772487335 | 7.37036515e-03 | 2.35381538e+00 | 5.279003 |
| fitted-hse06-uks / erfc | 10 | -4.284658734208046 | 4.97863981e-03 | 1.59239547e+00 | 5.003716 |
| fitted-cam-b3lyp-rks / Coulomb | 8 | -9.672681162936970 | 2.48668918e-03 | 7.92642098e-01 | 9.615512 |
| fitted-cam-b3lyp-rks / erfc | 8 | -9.672681162936970 | 2.23455583e-03 | 7.12152786e-01 | 9.615512 |
| uhf / Coulomb | 16 | -4.094056282195062 | 7.37036515e-03 | 2.35381538e+00 | 5.118549 |
| pbe0-rks / Coulomb | 8 | -9.700417784506898 | 2.48668918e-03 | 7.92642098e-01 | 4.953189 |
| pbe0-uks / Coulomb | 10 | -4.145245061675722 | 2.48668918e-03 | 7.92642098e-01 | 4.962024 |

CAM-B3LYP children precede signed coefficients; their shared calculation time
is repeated. Metric perturbation bounds span 1.9123e-11 to 8.8535e-11 and
inverse-response bounds span 5.8781e-5 to 2.7597e-4. The resulting raw K
energy-component bounds are deliberately conservative, reaching 2.354 Eh.
Passing these loose test targets is not useful automatic whole-method
accuracy. Broader error coverage does not imply changed numerical states;
tighter radial/cancellation and response bounds remain production work.

All 94 fitting tests pass, including 62 new construction witnesses. Independent
180-digit alternating radial integrals and closed finite Cartesian derivative
sums avoid the native positive series and derivative recurrence. They cover
all angular orders, Cartesian/pure representations, full/erf/erfc kernels,
signed coefficient scales, diffuse/tight primitives, repeated cancellation,
zero range, coincident centers and high angular pairs. An 800-digit large-T
witness checks the conservative gamma-integral enclosure. Stored pure
coefficients are conditional data, not independently qualified generation.
Exact/one-short memory, scalar, kernel, shell and primitive caps pass.
Existing independent libint fitting comparisons also pass.

The local calculation job passes 237 construction/prefactor/weight/phase/
assembly/refusal checks, including complete-method strict target refusal.
Separate metric/spectral (172) and J/K consumer (30) checks pass. Public
preparation/docs have 102 passing checks; the fragment-format failure was
fixed and all 10 fragment checks then pass. The known unrelated obsolete
CONTRIBUTING link-label test was explicitly excluded. Groups overlap.

Accepted Gaussian/libint/Fourier/NumPy references are reused for the first
three states (maximum Fock error 4.85e-15, energy error 1.12e-13 Eh).
UHF/PBE0 use accepted fixed-source controls. No matched ORCA/CRYSTAL custom
finite-domain reference is available; KS references share libxc. New scalar
and finite-derivative witnesses validate the integral construction component.
No bulk CRYSTAL parity or general speedup is established.

Harness cost is 40.488584 s wall, 39.209728 s CPU and 589.83 MiB peak RSS;
the queue process measured 41.855936 s wall and 40.720531 s CPU. The existing
`--metric-source-exchange --repetitions 1` benchmark exercises this proof;
select `metric_construction` in the fitting tests for independent witnesses.
Reuse unchanged accepted physical-reference calculations.

Actual qualified native SHA256:
`a98333da1d3e5890f81e5228e06281888a130fb237f336a2c0fc3a2201d4b09c`.
Source/native/dependency/reused-output pins agree before and after the job,
and fetched output hashes agree. This identifies qualified source/native,
not a deployed release. Basis/representation and lattice/vector data
generation; AO-pair Fourier/three-center coupling construction; domains and
screening; stored-factor/executed Fock/energy arithmetic; BvK, XC/grid,
basis/BZ; overlap/occupied-space/density relaxation and total relaxed energy
remain separate. Complete-source stays false. Shared Fock/PBC/BIPOLE/symmetry
contracts are preserved; #623/#625/#617 remain open and experimental.


## Positive short-range radial construction bounds

Qualified source `87e3fa9b` replaces the audit-only erfc difference of Boys
enclosures with positive panel integrals (contract 5.64). Bernstein products,
positive exponential series and explicit geometric/omitted-tail bounds avoid
cancellation while retaining the same exact stored-operand reference. The
ordinary Boys proof and numerical MD radial/integral producer are unchanged.

Six complete calculations exactly reproduce the preceding integral-construction
receipt: energies, iterations, physical source/final density/final Fock IDs,
QVF spin densities, full-Coulomb certificate and component ledger. Exact inputs
and physical supports match, including current continuous grids. Explicit
profiles retain original fitting policies and raw K targets 1e20; the first
three fitted profiles retain J source admission. No automatic-selection or
old-grid evidence substitution occurs. All 23 metric-owner receipts qualify,
including 21 K channels and two separate range-method J owners.

| Profile / raw kernel | Iterations | Energy / Eh | K entry bound | Component energy bound / Eh | Previous/new bound ratio | Wall seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fitted-rhf / Coulomb | 9 | -9.615420772487335 | 2.40418897e-05 | 7.67918765e-03 | 306.56 | 7.363285 |
| fitted-hse06-uks / erfc | 10 | -4.284658734208046 | 1.39530512e-04 | 4.46415625e-02 | 35.68 | 5.548650 |
| fitted-cam-b3lyp-rks / Coulomb | 8 | -9.672681162936970 | 4.84172632e-05 | 1.54342672e-02 | 51.36 | 8.525534 |
| fitted-cam-b3lyp-rks / erfc | 8 | -9.672681162936970 | 4.68778286e-05 | 1.49399350e-02 | 47.67 | 8.525534 |
| uhf / Coulomb | 16 | -4.094056282195062 | 2.40418897e-05 | 7.67918765e-03 | 306.56 | 5.566291 |
| pbe0-rks / Coulomb | 8 | -9.700417784506898 | 4.84172632e-05 | 1.54342672e-02 | 51.36 | 3.959953 |
| pbe0-uks / Coulomb | 10 | -4.145245061675722 | 4.84172632e-05 | 1.54342672e-02 | 51.36 | 3.954322 |

CAM-B3LYP rows are physical children before signed coefficients; their shared
calculation time is repeated. The raw K bounds improve by factors 35.68 to
306.56. Metric perturbation bounds span 2.2697e-13 to 6.0581e-13; inverse
response bounds span 7.0747e-7 to 1.8377e-6. The largest reported raw K energy
component is now 0.044642 Eh. This is a sharper certificate of the same
numerical states, not a change to measured energy errors or useful automatic
total relaxed-energy accuracy. Original component budgets are unchanged.

The local `vq` job passes 247 construction/prefactor/weight/phase/assembly
checks, including complete-method strict target refusal. All 62 earlier
construction witnesses pass, as do ten new positive-radial checks. Independent
180/800-digit alternating radial integrals test cancellation and tiny values;
a half-line Gaussian integral minus a small-argument series checks very large
T, with an explicit tail bound. Cases include sub-ulp erfc widths and panel
count boundaries. Evaluated integral tiles remain bitwise unchanged. The
ordinary Boys proof is textually unchanged for shared producer reuse.
Separate metric/spectral (172), J/K (30) and public preparation/docs (103)
checks pass. The known unrelated obsolete CONTRIBUTING link-label test is
explicitly excluded. Check groups overlap; no full-suite claim is made.

Accepted Gaussian/libint/Fourier/NumPy comparisons are reused for the first
three states (maximum Fock error 4.85e-15, energy error 1.12e-13 Eh).
UHF/PBE0 use accepted fixed-source controls. No matched ORCA/CRYSTAL custom
finite-domain reference is available; KS references share libxc. New independent
radial witnesses isolate proof accuracy and do not certify basis data generation.

Harness cost is 41.843619 s wall, 40.928844 s CPU and 588.05 MiB peak RSS;
queue process cost is 44.009036 s wall and 42.740007 s CPU. The preceding
harness took 40.488584 s wall and 39.209728 s CPU. These single measurements
do not demonstrate a speedup. Reproduce through the existing
`--metric-source-exchange --repetitions 1` benchmark and select
`positive_radial or metric_construction` in the fitting tests.

Actual qualified native SHA256:
`5bd3b928ddc250e5b2d9105307b50dc3e52eebbab93cf3899cb30e744e914538`.
Source/native/dependency/reused-output pins match before and after the job;
retrieved output hashes and exact configurations are verified. This is a
qualified source/native receipt, not a deployed-runtime or release claim.
Basis/representation and lattice/vector data generation, AO-pair Fourier/
three-center coupling construction, source domains/screening, stored-factor/
executed arithmetic, BvK, XC/grid, basis/BZ and overlap/occupied/density
relaxation remain separate. Complete-source remains false. Shared Fock/PBC/
BIPOLE/symmetry ownership is preserved; #623/#625/#617 remain experimental.


## Pure representation coefficient construction bounds

Qualified source `9e5cd724` removes the rounded pure-transform coefficient
premise from auxiliary Fourier and two-center metric construction proofs
(contract 5.65). The same shared representation enclosure covers mathematical
real Racah coefficients through l=7. Normalized basis coefficients and
lattice/vector data remain stored reference inputs. Numerical transforms,
integrals, factors and SCF paths are unchanged; no symmetry permission follows.

Six complete calculations exactly reproduce the preceding positive-radial
receipt: energies, iterations, physical source/final density/final Fock IDs,
QVF spin densities, original full-Coulomb certificate and component ledger.
Inputs and physical supports, including current continuous grids, match.
Explicit profiles retain original fitting policies and raw K targets 1e20;
first three fitted profiles retain J source admission. No automatic-selection
or old-grid evidence substitution occurs. All 23 metric-owner receipts qualify
with representation construction included, covering 21 K channels and two
separate range-method J owners.

| Profile / raw kernel | Iterations | Energy / Eh | K entry bound | Component energy bound / Eh | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| fitted-rhf / Coulomb | 9 | -9.615420772487335 | 2.55244645e-05 | 8.15276101e-03 | 4.535346 |
| fitted-hse06-uks / erfc | 10 | -4.284658734208046 | 1.50718898e-04 | 4.82215462e-02 | 3.536911 |
| fitted-cam-b3lyp-rks / Coulomb | 8 | -9.672681162936970 | 5.15840852e-05 | 1.64438244e-02 | 6.752791 |
| fitted-cam-b3lyp-rks / erfc | 8 | -9.672681162936970 | 4.93295351e-05 | 1.57212950e-02 | 6.752791 |
| uhf / Coulomb | 16 | -4.094056282195062 | 2.55244645e-05 | 8.15276101e-03 | 4.532985 |
| pbe0-rks / Coulomb | 8 | -9.700417784506898 | 5.15840852e-05 | 1.64438244e-02 | 3.425682 |
| pbe0-uks / Coulomb | 10 | -4.145245061675722 | 5.15840852e-05 | 1.64438244e-02 | 3.442992 |

CAM-B3LYP children precede signed coefficients; their shared calculation time
is repeated. Metric perturbation bounds span 2.39520560e-13 to 6.56060489e-13;
inverse-response bounds span 7.46598186e-07 to 1.99006140e-06. These are
conditional fixed-density component budgets, not total relaxed-energy accuracy.
The complete-source flag remains false.

Nineteen new representation checks compare outward coefficient bounds with
exact rational Legendre recurrence/differentiation/homogenization and 250-digit
square roots, covering Cartesian/pure l=0..7, immutable receipts, invalid
angular orders and exact/one-short storage/scalar/work caps. Existing SciPy
spherical-harmonic witnesses independently check the real normalization and
component order. Fourier/integral witnesses now use mathematical coefficients
from this separate oracle, removing the shared rounded-transform premise.
The local calculation job reports `266 passed, 498 deselected in 4.59s`; passing source/consumer checks
and the known excluded stale CONTRIBUTING navigation fixture are recorded in
the acceptance receipt. Groups overlap; no full-suite claim is made.

The first three states reuse accepted Gaussian/libint/Fourier/NumPy references
(maximum Fock error 4.85e-15, energy error 1.12e-13 Eh); UHF/PBE0 use accepted
fixed-source controls. No matched ORCA/CRYSTAL custom finite-domain reference
is available; KS references share libxc. Coefficient construction witnesses
do not establish numerical angular equivariance or omitted-member accuracy.

Harness cost: 31.522495 s wall, 31.414864 s CPU,
587.41 MiB peak RSS. Queue cost: 32.241826 s wall,
32.244272 s CPU. No speedup claim. The existing
`--metric-source-exchange --repetitions 1` benchmark exercises this proof;
select `representation_construction` in the MD tests for coefficient witnesses.

Actual qualified native SHA256:
`b4556eaa96511de4021276dc1cc0404a3faaa9f0e88f6d4f396db6ccc042eb09`.
Source/native/dependency/reused-output pins match before and after the job;
fetched outputs and exact configurations are verified. This identifies
qualified source/native, not a deployed release. Normalized basis and
lattice/vector data generation, AO-pair Fourier/three-center coupling
construction, source domains/screening, stored-factor/executed arithmetic,
BvK, XC/grid, basis/BZ and overlap/occupied/density relaxation remain separate.
Shared Fock/PBC/BIPOLE/symmetry ownership is preserved; #623/#625/#617 remain
experimental, without automatic total-accuracy, bulk parity or release claims.
