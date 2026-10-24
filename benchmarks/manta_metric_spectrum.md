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
