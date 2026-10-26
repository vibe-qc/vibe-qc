---
orphan: true
---

# MANTA implementation, acceptance and benchmark plan

This plan accompanies the [architecture](manta_engine.md) and
[scientific contracts](manta_scientific_contract.md). The stage table specifies
the full 0.18.x implementation target; the progress descriptions and manual
identify qualified increments. The reference-test inventory is not evidence that every
proposed backend already passes it. Reproducible benchmark reports live under
`benchmarks/`. No numerical source default changes in this plan.

## 1. Ordered implementation increments

| Stage | Concrete deliverables | Exit evidence / permitted next step |
| --- | --- | --- |
| P0: shared interfaces and references | Add provider capabilities, lossless shell/image/layout adapter, source/receipt schema and a libint tile adapter. Connect a diagnostic consumer to the existing molecular `JKBuilder` seam. Record fixed operator ownership. | Identical libint integrals/J/K through old and new boundaries, no duplicate normalization or changed thresholds; unsupported combinations rejected before setup. Snapshot lifetime and preflight budget gates pass. No new production default. |
| P1a: molecular primitive kernels | Implement FP64 Boys and MD recurrence, S/T/V, cross-basis S, moments, 2c/3c/4c Coulomb for s/p. Direct J/K and a small capped reference materializer. | Analytic/high-precision and libint parity, arbitrary-density contractions and explicit memory/work receipts. Selectable experimental `manta` on the admitted subset. |
| P1b: full molecular kernel envelope | Extend pure/Cartesian scalar orbital l through 6, auxiliary l through 7, general/partial-general contractions, batched multiple targets, erf/erfc and FT; optimize only after parity. | All admitted classes tested, contraction schemes agree, full/erf/erfc identities hold, zero-screening and all permutations work. Unsupported classes use declared fallback or refusal. |
| P1c: method and RI adapters | Add native `JKBuilder` integration, spin slots, direct and streamed RI J/K, metric admission, output/citations and explicit backend selection. | RHF/UHF/RKS/UKS parity and immutable solve capture, method energy components and terminal-density checks. ROHF/ROKS and response are individually admitted, not inferred from closed-shell tests. |
| P2a: molecular BUPO-J | Extent envelopes, moments/translations, deterministic hierarchy, disjoint near/far traversal, paired contraction, error accounting. Near field can use either provider. | Compare arbitrary-density J to the same unscreened/direct source; isolate tails, multipole, screening and summation errors. No exchange acceleration claim. |
| P2b: molecular RI-BUPO-J | Paired forward/adjoint coupling, streamed metric solves, fit/rank controls, incremental builds with full confirmations. | Exact RI versus approximate-pass comparisons, adjoint/energy-derivative tests and chain-drift safeguards. Measure actual crossover versus optimized RI and Split-J. |
| P3a: periodic primitive substitution | MANTA shell/FT tiles inside current Gamma and whole-mesh erfc/reciprocal/GDF sources. Keep BIPOLE domains and exchange seams. | Same-source S/T/V/J/K and energy-component parity on general complex trial densities, shifted/non-TRIM meshes and distorted cells. No molecular-limit shortcut. |
| P3b: periodic BUPO-J and RI-BUPO-J | Finite Ewald coefficients, compact/full-Gaussian remainders, typed q=0 hierarchy, complete finite direct J, a fixed admitted BUPO partition, retained-source infinite-Ewald errors and native all-pair-image electronic J/energy bounds and a matched electronic/point-nuclear charge, background and self policy are implemented. An explicit bounded RHF/UHF consumer now combines matched BUPO electrostatics with native kinetic and finite exact exchange, enforcing electrostatic targets at every density. Native full-image S/T enclosures now supply pre-SCF matrix admission and per-density kinetic energy admission. Explicit full-Coulomb exchange certification now jointly bounds omitted AO images and reciprocal vectors, including the overlap-dependent fixed-Madelung correction, at every evaluated spin density. A native reciprocal-cutoff factory preserves the original BUPO source while selecting certified exchange spheres and deriving their completion cube. Complete native AO/Ewald/tree/near construction now uses explicit cube policies, cumulative staged admission and a prepared source without electronic state. Typed public RHF/UHF jobs now retain the complete prepared source, managed output and evaluated-density QVF state. An exact paired q=0 RI J contraction now supplies a two-pass baseline over the unchanged retained metric and finite coupling source. The paired baseline now reuses unweighted complex forward/adjoint actions owned by the existing GDF tile producer, preserving ordered finite domains. A coupling-only screened baseline now preserves the exact finite metric and propagates source-owned omission bounds through paired J/energy actions. Ordered erfc shell-leaf multipoles now reuse the same producer support, native extents/moments and paired actions, with evaluated Gaussian error bounds and exact direct fallback. The selected auxiliary-image hierarchy now has shared Gaussian preparation and compressed forward/downward and adjoint/upward moment actions over the same q-aware coupling partition. A bounded source-owned cache now reuses selected trees and certified partitions across actions and SCF. Call-wide pair charges and auxiliary inputs, plus single auxiliary/pair projections, now reuse these producer panels across shell triples. An opt-in global AO-pair hierarchy now supplies paired moment actions and per-shell-block error ledgers. Shared isolated erfc radial and interior-moment bounds qualify explicit tighter tolerances. Shared componentwise omission and column-magnitude envelopes now tighten J/energy bounds through the actual retained metric factor; automatic whole-method profiles and general production crossover remain. | Independent Gaussian integration, unreduced ordered Bloch sums, complete reciprocal Coulomb limits, positive image envelopes, near transfers, domain enlargement and weighted energy derivatives qualify the declared electronic response. Diffuse/skew-cell whole-method accuracy remains a method gate. |
| P3c: low-dimensional sources | Explicit slab/wire Green functions and open-direction boundary policies with dimensional stress conventions. | Same-dimensional independent references; 3D vacuum extrapolation is supplementary, not the defining reference. Keep unsupported combinations refused. |
| P4: CCM and chi adapters | Individual adapters for union/WSSC, neutral Bloch, neutral real-Gamma and chi four-center/RI/RIJCOSX. Prove weight factorization before BUPO aggregation; use streamed direct terms otherwise. Inventory model-specific SECCM electrostatics. | Preserve finite domains/weights and experimental gates. Same-H Fourier equivalence only where applicable. No global supercell AO matrices or four-index tensors in production. Conditional capability decisions documented per source. |
| P5: qualified symmetry reduction | Method-owned admission over shell orbits, stabilizers, bubble/fit/support policies; transport in generation and digestion; explicit identity-only path. | Independent unreduced same-source comparisons, arbitrary-density equivariance, failed-parent/rank/phase negative tests, measured executed-work reduction. Existing diagnostic success alone does not enable reduction. |
| P6a: derivatives | Nuclear gradients for molecular exact/RI then BUPO, periodic and CCM sources; stress as a distinct extension. Include Pulay, metric/projector, moment/translation, domain, gauge and seam terms. | Distorted-geometry finite differences, sum rules, cell strain and topology/rank crossing diagnostics for each selected approximation. Nonvariational sources retain their restrictions. |
| P6b: further acceleration | Specialized recurrence DAGs, GPU batches, distributed target tiles, optional mixed precision and measured kernel dispatch. | Same scientific gates and end-to-end error budget; host/device transfer and replicated memory included. No imported ORCA/LibintX/EXESS speedup claims or unreviewed dependencies. |

P0-P2 can land without waiting for a universal periodic Fock refactor.
The periodic raw short-range fitting increment reuses the existing GDF image
selector with MANTA 2c/3c kernels. Analytic Boys double sums, same-source
libint metric/coupling blocks and general complex fitted-response trials
check Gamma and non-TRIM transfers. Signed long contractions, pure/Cartesian
angular shells, independent high-l radial/Fourier integrals, image gauges,
reciprocal relabeling, engaged envelope screening and simultaneous exact/
one-short resource caps retain their separate acceptance. The output owns
only raw SR components; complete RSGDF assembly and method selection remain
required. Its report is `benchmarks/manta_periodic_fitting.md`.

The subsequent complete Coulomb source assembles SR and tiled MANTA Fourier
blocks with the existing jellium kernel. Independent analytic finite sums and
converged bare-Coulomb reciprocal references validate Gamma and non-TRIM
transfers, two numerical splits, finite-zero subtraction independent of list
membership, and complex trial responses. Polynomial quadrature checks the
Cartesian and high-angular Fourier assembly. Gauges, left-handed cells,
engaged SR screening and joint caps remain explicit. Fitted SCF is qualified
separately below. The complete-source report is
`benchmarks/manta_periodic_gdf.md`.

The immutable tile source now reconstructs that dense source bitwise while
admitting only one metric or coupling shell tile. Its fixed-projector
q-channel RI consumer is checked against independent Gaussian/Boys/Fourier
sums and libint blocks, including signed angular contractions, arbitrary
complex indefinite spin densities, nonuniform weights, image gauges,
conjugate transfers and weighted energy derivatives. Converged numerical
splits agree with an independent bare-Coulomb fit. Redundant, truncated and
zero-rank auxiliary spaces, significant negative-metric refusal, owned
snapshots, source identities and all five exact/one-short resource caps have
separate tests. Count-only memory witnesses admit streaming below the size of
one full coupling tensor; they do not execute the large numerical workload.
The report is `benchmarks/manta_periodic_ri_q.md`.

The whole-mesh fitted Fock consumer now passes independent Gaussian/libint
checks on every matrix and energy component, weighted energy derivatives,
mesh reordering, and full/truncated-projector image transport through the
shared symmetry density-push/operator-pull interface. Opposite transfers
must preserve the actual finite reciprocal support and conjugate metric,
projector and inverse before the consumer is admitted. Arbitrary trial spin
densities remain independent; no spatial or time-reversal reduction is used.

Fixed independent J/K selection additionally passes separate Gaussian/libint
matrix and component-energy references, weighted energy variations and
reciprocity for full, truncated and zero-rank fits. Gamma/non-TRIM meshes,
nonuniform/zero weights, exact component decomposition, all five joint caps
and one-short refusal are tested. J-only owns no exchange-transfer metric;
its actual metric/coupling calls and complete admitted bound are smaller on
the multik witness. Independent image transport and owned exports cross the
landed shared Fock API with explicit source flags and no symmetry admission.
Eighteen prechange joint J/K states preserve matrices, scalar receipts,
projectors and admission counts bitwise. Timing and modeled-memory evidence
is in `benchmarks/manta_periodic_ri_targets.md`. Trial-density diagnostics also
compose each RI target with the complementary direct reciprocal source and
check independent responses, energies and derivatives with both transfer
conventions preserved. Mixed HF acceptance is described below. Fitted KS,
physical range fits and BIPOLE parity still require their own acceptance.

Physical finite-source RI UHF now reuses S/T/V, nuclear Ewald, the fixed-spin
bounded solver and the method-owned exchange correction. Independent libint
raw fits, NumPy contractions and SciPy SCF check Gamma/two-point/shifted H and
Li states, unequal/empty spins, the restricted limit, all energy components,
and every capped epoch. AO and auxiliary image transports check complex
physical snapshots. All five aggregate caps include every retained projector,
metric construction once, repeated coupling passes, scalar history and terminal
confirmation. Selected scientific output adds admitted per-transfer evidence
without changing the caller's budget.

An offline compatibility probe also admitted 24 actual RI UHF states (72 spin/k
channels) into the then-inspected FockBuild draft. Evaluated density, method energy,
fixed source, S/X frame, physical action, metric populations, immutable storage
and solver-parent identities were preserved. Capped states keep their distinct
next orbital projector. That probe does not land or duplicate the draft API,
replace method convergence criteria, qualify BIPOLE/CRYSTAL parity, or authorize
symmetry reduction. The report is `benchmarks/manta_periodic_ri_uhf.md`.
The RI UHF cutoff factory now passes independent integer enumeration of skew,
odd/even and shifted reciprocal spheres, exact-boundary/empty-source checks,
pair-complete AO image membership and independent AO/auxiliary gauge transport.
Explicit-source and generated runs have identical D/F and scientific source
identities. Independent libint/NumPy/SciPy checks cover Gamma and shifted H/Li
SCF with both exchange corrections. All five exact/one-short caps include
generation, auxiliary sources, projectors and repeated solver work. Its
geometric receipt remains distinct from the fitted physical source.
The report is `benchmarks/manta_periodic_ri_domains.md`.

Public RI UHF job selection now preserves the same independently checked
finite source, every fitting control and metric projector, explicit J/K
choices, evaluated spin history and lossless QVF density. Joint source/output
admission, dry runs, output delivery failures, truncated/zero-rank fits,
independent AO/auxiliary images and capped states have separate checks.
Repository tests now check the 24 public states against the landed FockBuild
API, preserving method energy, physical action, source/density/frame separation
and solver-parent identities without adding dense snapshots to the bounded
native solve. The
report is `benchmarks/manta_periodic_ri_job.md`.

Restricted RI HF now reuses the same immutable fitting/physical source while
running the bounded RHF numerical leaf. Independent libint metric/coupling
and one-electron/nuclear sources, NumPy contractions and SciPy restricted
SCF cover He/Be, Gamma, full/shifted meshes and both exchange corrections.
Capped recurrences, negative/empty/truncated auxiliary spaces, redundant AO
spaces, equal-spin embedding, exact/one-short native/domain/output caps,
independent AO/auxiliary images and nuclear-frame QVF have explicit checks.
Shared FockBuild crossings preserve the restricted total density and retained
S/X frame; no source symmetry certificate is inferred. Singlet UHF has the
same fitted source but its own solver/storage and method identity. Sixteen
existing UHF capped/converged states remain bitwise unchanged after sharing
RI HF composition. Native and public benchmark costs are recorded separately
in `benchmarks/manta_periodic_ri_rhf.md`. Fitted KS, physical
range-separated fitting and the remaining capability matrix stay open.

Mixed exact/RI RHF and UHF now pass independent libint raw-fitting, Gaussian
Fourier quadrature and NumPy/SciPy SCF references for both component choices,
Gamma/full/shifted meshes, empty/unequal spins and both exchange corrections.
The 152-case mixed suite also checks every capped recurrence, all five
native/generated/public resource boundaries, truncated and empty fits,
multicenter AO/auxiliary image transport, the two K-domain conventions,
evaluated QVF densities and the landed FockBuild interface. The complete
method plan admits retained fitting plus direct response and composition
workspace; `plan.fock` alone is only the fitting subplan. Thirty-two prechange
complete RI states retain arrays, spectra, energies, source/projector IDs and
existing resource counts bitwise. Eight isolated benchmark workers measure
complete native/public jobs and independently converged references in
`benchmarks/manta_periodic_mixed_hf.md`. Mixed finite-source qualification
does not establish fitted KS, physical range fitting, BIPOLE parity or a
source-specific symmetry reduction.

Explicit fitted pure/global KS now passes independent libint raw-fitting,
Gaussian Fourier/direct and libxc quadrature contractions with NumPy factors
and SciPy SCF. Pure RI-J LDA/PBE/BLYP and all three global exact/RI choices
cover Gamma/shifted meshes, restricted/unequal/empty spins and strict/BvK
corrections. Capped recurrences preserve the evaluated density and frozen
projectors. Fourth-order complex congruence energy differences check the
functional Fock derivative; the HF half trace is explicitly rejected as an
energy reference. Empty PBE/PBE0/B3LYP endpoints keep separate measured
minority-potential sensitivity witnesses, with strict occupied-spin and
energy checks. This is the existing libxc endpoint limit, not a widened
uniform matrix tolerance or a regularized functional.

Truncated/zero fitting spaces, independent multicenter AO/auxiliary image
shifts, redundant AO shells and rectangular S/X frames preserve all physical
components. Exports cross the actual shared FockBuild and BlockSpaceAction
interfaces with method-owned energy and no inferred symmetry certificate.
Source changes at the same HCORE state preserve the evaluated-density
identity while changing source/Fock identities. Fifty-six saved direct
pure/global/range KS states retain arrays, spectra, energies, source hashes
and original resource counts bitwise; density/snapshot IDs intentionally
adopt the separated version 2 contract.

All five native and selected scientific caps admit exactly and refuse one
short before poisoned array copies. Functional/source restrictions remain
fail-closed. Twelve isolated benchmark workers, each with five repeated
native and selected scientific solves, report complete fixed-source errors
and conservative resource models in `benchmarks/manta_periodic_ri_ks.md`.
The selected timing includes provenance construction but no full public job
or artifact writing.

Generated fitted KS adds independent integer membership/conjugacy checks for
Gamma and shifted meshes, raw-fitting/functional SCF on generated supports,
all exact/one-short domain/grid/output caps, capped/truncated states, empty
spins, rectangular overlap frames and independent AO/auxiliary images.
Owned exports preserve the shared Fock and BlockSpaceAction contracts.
Twelve saved generated direct KS states preserve numerical arrays, plans and
identities exactly after sharing the factories. The generated/explicit/selected
states agree bitwise. `benchmarks/manta_periodic_ri_ks_domains.md` measures
complete source and quadrature construction in twelve isolated workers with
five repeats per interface.

Public fitted KS adds independent pure/global functional SCF, explicit J/K
selection, complete output-budget boundaries, capped/truncated states and
lossless evaluated-density QVF for restricted, unequal and empty spins.
Independent AO/auxiliary image shifts, rectangular overlap frames, immutable
result/evidence exports, citations, dry runs and artifact-delivery failures
preserve the method contract. Sixteen saved public direct KS/RI HF states
retain numerical arrays, native plans and identities exactly. The twelve-worker
`benchmarks/manta_periodic_ri_ks_job.md` measures complete native and public jobs,
including standard output and QVF, with five repeats each. Physical range fits,
BIPOLE/CRYSTAL parity, source-qualified reduction and the remaining approved
matrix stay required.

The initial public surface covers the P0 reference boundary and the P1
molecular MD one-electron matrices, streamed 2c/3c fitting tiles and four-center
tiles/direct J/K, including erf/erfc kernels and batched AO-pair/auxiliary
Fourier panels, multi-density direct digestion and a streamed dense-metric
RI-J/K reference, plus bounded sharing across general and partial contraction
channels, analytic one-electron basis/charge/origin derivatives and analytic
2c/3c/4c shell-center derivative tiles and a bounded all-electron molecular HF
Lagrangian gradient consumer, plus explicit RHF/UHF/RKS/UKS method selection with
executed-provider and terminal-density evidence, independent exact/RI range
exchange sources and the native XC/VV10 composition. A bounded absolute
three-dimensional shell-pair extent/tail envelope and bounded Cartesian
multipole translations/interactions and a deterministic streamed bubble
partition support a low-level molecular BUPO-J consumer. Independent
point-charge Legendre expansions check the multipole algebra,
including the compact-support remainder; Gaussian tails remain separate.
Small exhaustive inventories prove that near/far blocks partition each
unordered shell-pair interaction once, including when candidates are refined.
Shared primitive shell-pair moment panels through degree twelve and optional
full absolute radial envelopes support streaming forward/reverse consumers;
independent polynomial quadrature validates the complete panels.
Single-auxiliary panels through l=7 and degree twelve, and shell-centered
full/tail extent envelopes, are validated against direct polynomial
quadrature and radial gamma integrals. These retain the single-function
distribution required by RI-BUPO rather than substituting an orbital product.
A Gaussian error model combines compact truncation, exact tail penetration
and the full-versus-compact moment correction. Independent libint and analytic
Gaussian interactions validate its per-integral approximation ledger, separate
from floating-point contraction errors. The paired J consumer passes arbitrary
density, linearity, reciprocity, energy-derivative, near-only and strict
far-field comparisons against independent dense integral contractions. Its
selected RHF/UHF/RKS/UKS adapter confirms the returned density against an
independent exact J while preserving the chosen exchange and XC treatment.
Loose negative controls must fail that confirmation even when the approximate
SCF itself is stationary. Performance benchmarks remain a separate gate;
BUPO does not supply exchange or derivatives.
Pair-bubble/single-auxiliary interactions reuse the same explicit Gaussian
ledger, with independent three-center libint and unequal-Gaussian analytic
checks. This establishes coupling-tile admission, not a complete RI-J error
or fitting-conditioning certificate.
A native paired three-center coupling now streams the same density-independent
partition in forward and adjoint passes, with atomic auxiliary leaves and pair
bubble translations. Independent libint/libcint comparisons through orbital
l=6/auxiliary l=7, full-matrix adjoint tests, exact near limits, signed linearity,
positive-metric energy derivatives and exact aggregate resource boundaries
validate that primitive. A bounded RI-BUPO J consumer now combines both passes
with the existing exact dense metric and fixed positive retained subspace.
It reports explicit exact-RI confirmation, separate pass defects, propagated
J/energy coupling bounds and discarded-space metric residuals. Independent
dense references, duplicate/nearly-dependent and zero-rank spaces, energy
finite differences and combined resource boundaries validate this composition.
Selected RHF/UHF/RKS/UKS can now use RI-BUPO J with independent exact/RI K
and native XC/range exchange. Terminal confirmation rebuilds exact RI-J with
the identical auxiliary/metric source; loose stationary and NOITER controls
must fail. Receipts include executed paired-source and conditioning evidence.
Large streamed metric solves and selected RI-BUPO derivatives remain gates.
QVF file and memory producers preserve executed molecular engine receipts in
an optional vendor section, including failed exact terminal confirmation, with
hash validation, coexistence and source-ownership tests. Selected molecular
static populations, bond indices, natural orbitals and dipoles use the executed
one-electron provider with content checks and separate work receipts. Selected
single-point jobs retain J/K and range/XC sources through dispatch and RKS READ
retries, output/citations and failed-confirmation checkpoints. IAO populations
and closed-shell IBO/Boys/Pipek-Mezey localization use the selected main,
minimal-reference and cross-basis overlaps and dipole moments. Independent
libint comparisons, occupied-density preservation, source refusals and QVF
exports qualify these property paths. Atomic SAD promolecules, response and
other job compositions remain gated.
Independent shell-tile sessions admit only provider/workspace resources and
one quartet, leaving aggregate work and density buffers to the scheduler;
the original ordered J/K plans retain their whole-traversal admission.
These interfaces
do not complete the remaining job-composition,
large-metric, periodic or derivative gates;
the [manual](../user_guide/manta.md) records what is actually selectable.
Symmetry foundations can advance independently, but P5 execution depends on
the actual P2/P3/P4 source to be reduced. Derivative primitives can be developed
earlier; a user-facing force/stress capability requires the complete P6 gate.

Use incremental, additive changes. Add a manifest classification through
`scripts/test_gate/update_suite_manifest.py` for each new test file. Add
user-visible changes as `changelog.d/` fragments when behavior ships, not as
claims in released changelog sections. The implementing chat may validate and
accept its own work; no separate reviewer or review-only hold is required.
Scientific acceptance and closure require an affected-method calculation
through `vq` with a suitable matched external or justified numerical reference.
Reuse valid evidence and record the actual source/native runtime. Ordinary
landings skip CI; full CI belongs to release candidates. Before a large refactor,
record its bounded scope on the [full MANTA work tracker](manta_work_tracker.md).

The dense RI metric response now shares the RI positive-eigenprojector
factorization. Independent displaced eigensolves validate inverse and
projector derivatives, including nonzero discarded modes, degenerate blocks,
zero rank and absolute/relative cutoffs. Frobenius adjoints, orthogonal
covariance, fixed-density fitting energies, exact factor identity, numerical
rank-crossing refusals and resource boundaries are qualified separately from
complete nuclear gradients. The cutoff guard is a numerical admission policy,
not an interval spectral certificate.

A fixed-density RI Coulomb gradient consumer now streams the fitting RHS and
three-center derivative contraction and contracts the exact metric adjoint.
Independent libint energy differences validate full and truncated fitting
spaces; range kernels, separate orbital/auxiliary ownership, component-wise
finite differences, translation/rotation sum rules and aggregate resource
boundaries cover the derivative composition. This qualifies the RI Coulomb
energy component, not complete SCF forces, exchange derivatives or stress.

The paired-spin RI exchange energy gradient now forms its metric adjoint in
bounded raw auxiliary blocks, including discarded modes, and streams the
three-center derivative contraction. Independent libint energy and component
finite differences, full/erf/erfc kernels, spin additivity, block invariance,
zero rank, geometric sum rules and aggregate work/memory admission qualify
this component. Receipts match the RI exchange source and spin-density
identities and expose the reference schedule's repeated integral passes.
Complete selected RI SCF force composition remains a separate gate.

A complete low-level molecular HF RI Lagrangian now combines independently
selected exact/RI J and K derivatives with nuclear, core and overlap Pulay
terms. Component-wise independent libint finite differences cover pure and
Cartesian, full and truncated metrics; relaxed RHF/UHF differences and
independent full-RI force comparisons include auxiliary response. Executed
counts prove that an all-RI derivative requests no four-center tiles. The
combined admission covers every component before copying input matrices.
The generic selected RHF/UHF adapters now retain an immutable auxiliary basis,
RI policy and budget. They verify geometry, terminal density and rebuilt RI
source identities. Relaxed-energy differences exercise both public and native
gradient routes for independent exact/RI J and K selections. Custom same-name
auxiliary bases, truncated metrics, caller option mutations, derivative-budget
refusals and success-only output receipts qualify context preservation.
Selected KS and BUPO derivatives remain separate gates.

The molecular HF geometry source now preserves orbital/auxiliary shell content
and atom maps while rebuilding centers at each geometry. Displaced RHF/UHF
energies validate all exact/RI J/K compositions; native optimizer walks and
complete jobs verify stationary final forces, source identities, final custom
auxiliary placement and derivative citations. The source snapshots mutable
SCF/fitting options and budgets and uses explicit HCORE starts. Projected READ
warm starts and other derivative compositions remain separate gates.

The finite Bloch Fourier primitive now streams translated shell panels across
an explicit image list and shares them across ket momenta. Independent Gaussian
quadrature, physical libint overlap at zero argument, reciprocal relabeling,
global-origin phases, momentum-transfer adjoints and exact physical-support
boundaries qualify its convention. The older raw Fourier table's normalized
solid harmonics require an explicit angular conversion before reference
comparison, matching the existing RSGDF AO scaling helper. That difference is
not hidden by a fitted numerical rescaling;
complete periodic/CCM sources still require normalization-consistent method
qualification. Aggregate work/storage caps and actual generation/folding
receipts cover bounded batches and the opt-in periodic Fourier helper adapters.

The finite three-dimensional reciprocal Hartree consumer now composes this
provider over a complete density epoch with explicit nonnegative normalized
k weights and unique inversion-closed image/G supports. Tests compare pure
and Cartesian results with independent Gaussian quadrature, differentiate
the Hartree energy with respect to one complex Hermitian density block, and
check Gamma reduction, nonuniform mesh permutation, time reversal, reciprocal
relabeling and origin covariance. A normalized scalar Gaussian supplies a
closed-form reciprocal-cutoff convergence reference. Existing Ewald J
cell/k interfaces agree at the same finite source and physical normalization.
Admission covers cumulative kernel work, all live output/input owners and
the Python grid generator; low caps fail before payload/grid allocation.
Raw imaginary/Hermiticity residuals are checked before legacy real-block
conversion. This does not qualify nuclear/K composition, periodic SCF,
CCM, periodic BUPO, forces or stress.

The one-electron point-charge provider admits erf/erfc kernels and analytic
bra, ket and charge-position derivatives at fixed range. Independent libint
erfc nuclear matrices, optional libcint Gaussian-charge matrices through
l=6, and high-precision scalar Gaussian integrals check the radial convention.
The latter retain relative accuracy checks on erfc tails below the range at
which full-minus-erf subtraction rounds to zero. Range partition, translation
sum rules, center finite differences, resource boundaries and unchanged
default Coulomb source identities gate the primitive. This does not itself
admit a periodic nuclear sum or a derivative of the splitting parameter.

Finite overlap and kinetic producers now retain real cell blocks while sharing
translated MD shell tiles across all Bloch momenta. Independent libint lattice
blocks, l=6 polynomial quadrature and a converged analytic Gaussian chain check
the physical convention. Tests cover image-relabeling covariance with regenerated
pair-complete supports, reciprocal relabeling, asymmetric/repeated image domains,
raw residuals, exact cutoff boundaries and aggregate admission. The explicit
cell adapter preserves the standard LatticeMatrixSet in dimensions one through
three and emits executed source/work receipts. It does not enable complete
periodic SCF or symmetry reduction on its own.

The finite three-dimensional nuclear operator now composes erfc point-nucleus
images, damped reciprocal attraction and the positive overlap correction for
zero-average G=0 treatment. Its physical nuclear sphere uses the existing
midpoint-centered selector. Independent libint short-range blocks, Gaussian
quadrature reciprocal blocks, analytic overlap corrections and a converged
scalar plane-wave series check each component and the total. Tests exercise
split convergence, basis/nuclear image relabeling, origin shifts, complex Bloch
folding, l=6 screened attraction, actual image/primitive-charge counts and
aggregate resource boundaries. The explicit lattice adapter agrees with the
existing Ewald operator at matched finite domains and emits execution evidence.
This qualifies the operator, not complete periodic SCF, exchange, forces or stress.

A finite reciprocal full-Coulomb exchange consumer now accepts explicit
per-transfer Q supports and an immutable all-k/all-density snapshot. Its
B-dagger-D-B convention agrees with independent Gaussian quadrature, analytic
scalar Gaussian sums and the existing negative-source-phase contraction after
physical AO normalization. Reversed transfer supports are checked so that the
weighted density response is self-adjoint; perturbing one k density verifies
the exchange-energy derivative and changes responses across the mesh. Separate
spin/trial channels share Fourier work but retain separate responses/identities.
Tests cover image-gauge density pushforward, reciprocal relabeling, arbitrary
complex Hermitian densities, raw residuals, batching and aggregate admission.
The explicit source omits Q=0 and adds no Madelung correction. It does not
qualify singularity-corrected exchange or complete periodic SCF.

The same consumer now supports explicit erf/erfc kernels. A positive-range
erfc request includes its finite zero mode only when supplied in the domain;
full Coulomb/erf retain the omitted-zero convention. Independent quadrature,
energy derivatives, high-precision positive-tail tests and a scalar Gaussian
real/reciprocal Poisson identity qualify this distinction. Kernel/range and
actual zero-mode work are recorded; no singularity correction is inferred.

The native spin Fock consumer now freezes H+J[D_alpha+D_beta]-a*K[D_spin]
and the corresponding electronic-energy components in one owned epoch.
Independent quadrature, spin/k energy derivatives, restricted factors,
time reversal and atom-image gauge tests qualify the composition. Source,
density and evaluated-payload identities remain distinct; aggregate admission
reserves both component peaks and enclosing owners before copying inputs.
The H matrix is explicitly supplied and authenticated, not asserted to be a
MANTA nuclear source. This is a fixed matrix snapshot; complete physical
periodic SCF and correction policies remain separate qualification stages.

The neutral all-electron three-dimensional RHF driver now composes physical
S/T/V, point-charge Ewald energy and the whole-mesh reciprocal J/K source.
The bounded numerical solver retains the last evaluated density/Fock pair,
checks the occupied/virtual gap and projected eigenproblem, and requires a
fresh terminal Fock/energy confirmation. Explicit strict zero-mode omission
and a separately admitted BvK probe-charge Ewald correction are qualified
against independent quadrature/libint components and optional refined PySCF
AFT calculations. Full regular Gamma, multi-k and shifted meshes are covered;
cutoff convergence remains distinct from SCF convergence.

The cutoff factory admits physical AO-pair image spheres and certified
reciprocal kinetic-energy spheres before allocation, checks reversed
momentum-transfer conjugacy, and preserves independent domain/source
identities. Its selected `run_periodic_job` adapter now provides separate
engine/J/K controls, explicit domains, output memory reservations, standard
artifacts and executed provenance. QVF preserves evaluated densities on
converged and capped exits and transports independently displaced AO shells
into the archive's nuclear-center gauge. Capability refusals and output
delivery failures retain the shared job lifecycle semantics. Native-source
and complete-job benchmarks separately report numerical error, time, work
and memory. General periodic KS, RI/BUPO, reduced-dimensional boundaries, CCM,
qualified symmetry reduction and periodic derivatives remain open stages;
these RHF results do not qualify them.

The physical RHF driver also retains scalar snapshots of every evaluated
iteration through the bounded solver's existing observer contract. Native
copies and Python/QVF records are admitted against the maximum iteration
count; matrix histories are not retained. Independently capped runs check
each energy/residual/gap snapshot, and confirmation calls do not create
extra iterations. Selected jobs deliver the complete history after the
solve while retaining the evaluated terminal density separately.

The bounded UHF numerical leaf is now a separate fixed-spin solver with
`D[2,K,n,n]`, shared physical `S/H[K,n,n]`, independent alpha/beta occupations
and a single whole-density callback per iteration. It shares the RHF leaf's
audited Jacobi, overlap and metric routines. Complex NumPy fixed-point
references qualify unequal/empty spin populations, spin exchange, AO phases,
per-spin band gaps, metric idempotency and last-evaluated-density exits.
Exact-cap and one-byte-short cases check owned and enclosing resource budgets.
The explicit-support physical UHF driver now composes this leaf with the
neutral 3D all-electron S/T/V and nuclear Ewald source, whole-mesh spin J/K,
and separately admitted BvK correction. Independent quadrature/libint UHF
fixed points validate empty and nonempty beta channels, Gamma and shifted
meshes, component energies, shell-image gauge transport and same-density
confirmation. Physical RHF/UHF share the spin-independent source receipts;
the full UHF identity additionally binds its fixed spin populations.
Refined Gamma and shifted-mesh hydrogen cases also compare S/H, both spin
densities and Focks, nuclear and total energies, and the Madelung constant
against optional PySCF AFT KUHF. The reference uses explicit fixed-spin
occupations to avoid its empty-beta diagnostic bug, with supercell electron
counts set explicitly. These are finite-basis checks, not spin-stability
or basis-convergence claims. Cutoff-generated UHF now shares the certified
geometric source with RHF and admits a separate complete spin-aware physical
plan. Independent integer enumeration, spin SCF, shell-image transport and
exact-budget refusals qualify this route. Selected public UHF jobs now deliver separate evaluated spin densities,
owned spin orbitals/occupations, complete scalar spin history and standard
artifacts. Independent job-level integral/SCF comparisons and QVF readback
cover empty beta, unequal populations, capped states and shifted AO gauges.
Job resource admission and delivery failures preserve the shared lifecycle.
Pure-KS jobs are qualified below; general KS and later stages remain open.

A separate joint functional numerical contract now reuses the bounded
restricted/unrestricted iteration kernels while preserving the HF APIs.
The provider supplies potential and interaction energy from one evaluated
D. Nonlinear polynomial fixed points and energy derivatives distinguish this
from the HF half-trace formula; capped, cancelled and missing-output cases
check epoch consistency. This is groundwork for periodic KS, not physical
XC quadrature or selected KS job qualification.

## 2. Independent reference hierarchy

Agreement with one backend does not establish correctness if both share a
normalization or contraction bug. Use all applicable levels:

1. Analytic ssss, overlap, kinetic, point-charge and Gaussian moment formulas;
   high-precision one-dimensional Boys quadrature and small Gaussian Coulomb
   cases independent of the production recurrence.
2. Existing libint integral tiles, with a separately checked AO convention;
   small independent NumPy contractions over full tensors, not the same
   MANTA digestion loop reused as an oracle.
3. Existing unreduced native/Python lattice and reciprocal builders under
   matched supports, weights, metric ranks, background and q=0 conventions.
   Distinguish finite-source parity from convergence to an infinite source.
4. Independent external programs only in optional tests or regression
   subprocesses, never in `python/vibeqc/` or `cpp/`. Pin basis coefficients,
   pure/Cartesian order, pseudopotentials, auxiliary basis, exchange treatment,
   numerical cutoffs and gauge. A converged SCF comparison without those
   matches is a scientific comparison, not an implementation identity test.

For periodic Coulomb use an independently accumulated high-accuracy Ewald or
reciprocal reference; validate nuclear, J and combined electrostatic energy
separately. A direct bare lattice cutoff is not a valid 3D Coulomb reference
without its surface/neutrality convention. For CCM use explicit small-domain
weighted sums as oracles; a generic periodic program cannot certify a
different finite Hamiltonian. Published lattice-sum numbers in Kudin 2004
Tables I-II provide an additional convention-converted witness.

Existing tests to inspect and extend, preserving their established thresholds:

| Layer | Existing files |
| --- | --- |
| Molecular primitives and fit | `tests/test_integrals.py`, `tests/test_df_core.py`, `tests/test_df_rhf.py`, `tests/test_df_uhf.py`, `tests/test_df_gradient.py` |
| Frozen operators | `tests/test_operator_contract.py` |
| Periodic finite kernels/resources | `tests/test_periodic_gaussian_source_context.py`, `tests/test_periodic_gaussian_metric.py`, `tests/test_periodic_gaussian_three_center.py`, `tests/test_periodic_gaussian_fock.py` |
| Whole-mesh and GDF | `tests/test_periodic_fock_multi_k.py`, `tests/test_rsgdf_shared_q_batch.py`, `tests/test_gdf_cross_route_parity.py`, `tests/test_periodic_gdf_memory_preflight.py` |
| BIPOLE identity and final density | `tests/test_bipole_jk_domains.py`, `tests/test_bipole_domain_contract.py`, `tests/test_bipole_final_density_phase.py`, `tests/test_pbc_bipole_multik_ewald_split.py` |
| CCM/chi | `tests/test_ccm_bloch_reference.py`, `tests/test_ccm_direct.py`, `tests/test_periodic_ccm_neutral_bloch_adapter.py`, `tests/test_chi_transform_inputs.py`, `tests/test_chi_fitted_controls.py`, `tests/test_chi_hf_reference.py` |
| Symmetry | `tests/test_symmetry_shared.py`, `tests/test_symmetry_integrals_reduced.py`; the symmetry workstream's retained-GDF tests after integration |

## 3. Numerical acceptance suite

The numbers below are **proposed initial small-fixture gates**, to be
confirmed with independent references. They do not relax existing tests or
promise these tolerances for arbitrary ill-conditioned inputs. Fixtures have
bounded norms and non-singular retained spaces. If a gate fails, establish
conditioning/reference error before proposing a change; do not silently widen
it. Publish absolute max, absolute Frobenius and scaled Frobenius residuals.

| Test family | Required cases and measurements | Initial gate |
| --- | --- | --- |
| Normalization and integral tiles | All s/p classes and representative classes through advertised l; mixed pure/Cartesian, same-center, far/diffuse/tight, signed and deep contractions, ghosts; 2c/3c/4c; every coincidence/permutation | Element error <= `1e-11 + 5e-12 * abs(reference)` a.u.; additionally report norms and analytic ssss/S diagonals. Higher-conditioning cases use high-precision witnesses. |
| Range and moment/FT kernels | omega near zero, intermediate and large; full=erf+erfc; analytic finite SR Q=0; moments at several origins, translation composition and derivative orders; FT conjugation and q wraps | Same integral gate; zero/large limits and exact low-order identities tested independently |
| Arbitrary-density J/K | Seeded PSD, Hermitian indefinite, sparse, random complex, spin-separated and small delta densities; batches of 1/2/8 targets; response non-Hermitian mode separately | Max error <= `2e-10` Eh for fixed bounded fixtures versus independent full contractions; linearity/adjoint and raw Hermiticity <= `1e-11` in their stated norms |
| Energy consistency | Central difference of quadratic J/K energy with nonsymmetric-in-space Hermitian perturbations; density-pair reciprocity; spin factors and separate exchange ranges | Directional derivative agrees to `1e-8` Eh on the step-size plateau; cross-density bilinear residual <= `1e-10` before any symmetrization |
| BUPO-J | Exact near-only, far-only admitted pairs, translated and rotated geometries, signed contractions, diffuse always-near bucket, repartitioned trees, reordered tasks | At strict target, max delta-J <= `1e-8` Eh and absolute delta-EJ <= `1e-8` Eh for small fixtures; tighten each approximation until the same-source analytic limit is recovered |
| RI-BUPO-J | Exact B/V baseline; approximate forward only, adjoint only and paired passes; rank-degenerate/near-threshold auxiliary spaces; small/large solve residuals | Same BUPO gate relative to exact RI; forward/adjoint dot test <= `1e-11` relative. Separately report exact-RI minus exact-4c error, never charge it to BUPO |
| Incremental builds | Alternating and cancelling deltas, different alpha/beta slots, restart, threshold/rank/kernel/tree changes, at least two rebuild intervals | Final full build agrees with independent nonincremental map within its allocated error; forced drift prevents convergence and reseeds state |
| Periodic finite-source parity | Gamma, 2x1x1, 3x1x1, 2x2x2 and shifted meshes; non-TRIM complex densities; one k perturbed at a time; nonuniform normalized quadrature outside finite-character tests | Raw J/K max error <= `1e-9` Eh for small matched sources; component energies <= `1e-8` Eh/cell; no final averaging to conceal phase defects |
| Periodic convergence | Independent real/G/eta, near-cell, pair-image, multipole order and mesh ladders; diffuse/skew ionic cells, neutral and explicitly compensated charged controls | Same-source errors meet the specified budget; successive independent converged controls change total energy < `1e-7` Eh/cell and J/K < `1e-7` Eh before claiming convergence for that fixture |
| Cell identity/gauge | Shift atom by lattice vector; change unit-cell origin; equivalent supercell folding and k labeling; G=0 background and nuclear self removal; full versus SR exchange seam | Covariant matrices and invariant matched energies satisfy the periodic gates; expected potential-zero changes are identified component by component |
| CCM identities | Explicit weighted small sources, WSSC ties on odd/even meshes, all route names, neutral real/Bloch Fourier equivalence, chi finite characters and fitted projectors | Same-source matrix/energy gates; intentionally different sources must remain distinguishable; unsupported weight factorization/dimensions refuse |
| Symmetry | Proper/improper operations, stabilizers, fractional translations, scalar TR, Cartesian metric, dense auxiliary gauges, broken-state trial densities, orbit-adjoint contraction | Reduced/unreduced max delta-J/K <= `1e-10` Eh on bounded fixtures, separately from approximation error; every deliberately invalid source/rank/action is rejected or runs identity-only |
| End-to-end SCF | H2O, stretched/open-shell molecule, signed/general/diffuse basis; ionic and covalent distorted solids; each experimental source separately; HF, pure KS, hybrid, screened hybrid, smeared metal where supported | Same basin/occupations and original convergence tolerances; total energy <= `1e-8` Eh (per cell for periodic fixtures), component errors reported, density subspaces compared in S rather than arbitrary orbital gauges |
| Gradients | Distorted molecules/cells, selected exact/RI/BUPO energy, multiple displacement steps; total and component finite differences; tree/rank/support crossings | Max force error <= `3e-6` Eh/bohr on a stable FD plateau; translation/rotation sum rules; separate reports for crossing cases and nonvariational sources |
| Stress | Hydrostatic and shear strains, skew cells, fixed fractional k, nuclear/background/seam terms, finite-temperature free energy where applicable | Max error <= `3e-6` Eh/bohr^3 for 3D fixtures; dimensional units for 1D/2D; compare several strain steps and cutoff ladders |

Convergence ladders must vary one approximation at a time and then test their
combination. Finite-difference energy noise must be well below the targeted
force times displacement (or stress times volume and strain). Smearing tests
compare the appropriate free energy and occupations. Degenerate orbital
rotations are assessed as subspaces; energy ordering alone is not a state
identity check.

For every budgeted path test exact-cap acceptance, one-byte/work-unit-short
refusal, integer overflow, cancellation, invalid dtype/layout, NaNs and stale
source identity before expensive allocation. Verify that tile views cannot
outlive their workspace and that mutating a builder after snapshot creation
does not change a captured operator. Concurrent density mutation is prohibited;
use ownership or the existing checked borrowed-input pattern. Force a fallback
and verify its source identity, memory admission and output counts.

Initial local test ceilings: analytic/tile fixtures 60 seconds and 512 MiB per
case; small J/K and periodic cases 15 minutes and 4 GiB per process. These are
proposed admission limits, not measured runtimes. Medium numerical and scaling
cases need a separately sized plan after small gates pass. No full native lane,
large all-q inventory or large supercell run is required by this documentation
change. Actual implementation tests must rebuild and verify the loaded native
extension belongs to the edited checkout.

## 4. Benchmarks with attributable speedups

Benchmark the requested scientific accuracy, not only a kernel's throughput.
Use reproducible public input definitions, pinned source/build/compiler/BLAS,
CPU/GPU model, thread/rank settings and numerical controls. Record operational
logs, machine identifiers and absolute paths outside the product checkout.
Only sanitized portable summaries and approved fixtures belong in the tree.
Acquire licensed literature SI inputs on demand unless redistribution has
been reviewed. SI geometry presence does not confer a license to bundle it.

Workloads should cover:

* Low-l segmented molecules where libint can win; deep general contractions;
  high-l quartet and 3c classes; multiple response right sides.
* Compact water/organic clusters, extended glycine chains and a diffuse
  charged molecule. Vary shape and basis separately from atom count.
* Gamma and multi-k covalent/ionic solids, anisotropic cells and slabs/wires
  only after their source gate. Vary cell size, k mesh and support separately.
* Small and growing finite sources for each CCM/chi route, including cases
  where weight structure prevents efficient BUPO aggregation.

For each workload run these matched variants, including no-win cases:

| Variant | Provider | J algorithm | Symmetry | Question answered |
| --- | --- | --- | --- | --- |
| A | libint | existing exact/direct or RI | off | Baseline including existing optimized digestion |
| B | libint | new tile/consumer exact or RI | off | Interface/digestion overhead or benefit |
| C | MANTA | same exact or RI algorithm as B | off | Integral-provider contribution |
| D | MANTA | BUPO or RI-BUPO matched to C | off | Coulomb approximation/traversal contribution at measured accuracy |
| E | MANTA | same as D | qualified reduction | Symmetry contribution for the same numerical source |
| F | libint near field | same BUPO plan as D | off | Provider versus BUPO interaction/crossover |

Measure A/B, B/C, C/D and D/E separately; show A/E only as total observed
benefit. Also compare to the best *existing* admissible route. Approximate
sources with different RI/K choices do not form a speedup ratio without
showing their accuracy difference. End-to-end timings are not products of
microbenchmark speedups.

Each record contains setup and total wall/CPU time, SCF iterations and final
confirmation time; basis/extent/tree setup; E/R/Boys formation; primitive and
contracted transforms; screening; near-field integrals; moments/M2M/M2L/L2L;
metric setup/solve; RI forward/adjoint passes; J/K digestion; reciprocal work;
symmetry planning/transport; XC; eigensolve; output; synchronization, I/O and
host/device transfers. Report peak process-tree RSS and device memory, measured
workspace high-water mark, per-rank/worker replication and spill volume.

Actual work counts include candidate/retained shell and primitive pairs,
quartets/triples/metric tiles, screened terms, executed GEMMs and recurrence
classes, bubble visits/accepted interactions/orders, near-cell and reciprocal
vectors, q/k contractions, fit iterations, symmetry representatives and
reconstructions, incremental/full rebuilds and fallback tiles. Do not present
planner cardinalities as executed work. Explain time components that overlap
instead of adding overlapping timers.

Run one warm-up and at least three measured repeats for stable medium cases;
report median and range, with cold setup separately. Longer cases may use
fewer repeats with that limitation visible. Confirm accuracy on the timed
run's actual output, including J/K, energy components and terminal density.
Control thread counts and node contention; an isolated rerun distinguishes a
timeout from a numerical regression. Show absolute seconds and memory even
when a speedup is below one. Fit scaling only over the measured regime with
fixed accuracy, and disclose dense metric/solver bottlenecks and diffuse
near-field growth. No ORCA speedup is transferred to vibe-qc.

## 5. Decisions and review checkpoints

Adopt now as design constraints: explicit provider selection, separate J/K
algorithms, fixed solver snapshots, source-specific periodic/CCM identities,
bounded production storage, paired RI approximation, unreduced reference and
visible provenance. These do not change a public API or an existing source.

Before the relevant implementation stage, review the proposed public option
names/schema, qualified l/derivative envelope, calibrated accuracy profiles,
periodic tail representation, compatible CCM weight factorizations and the
large-metric solver/preconditioner. A new hard dependency, licensing decision,
public API break or physical-source/default change belongs to the maintainer.
Ordinary additive provider implementation and focused validation can proceed
within the agreed stage. Preserve the libint reference throughout the rollout.


## Periodic direct-Bloch local-exchange increment

The finite local-exchange source is qualified separately from physical KS
SCF. `tests/test_manta_periodic_local_exchange.py` compares complex direct
Bloch densities and potentials with independently evaluated Gaussian
polynomials and libxc LDA exchange. Its gates cover weighted k/spin energy
derivatives, restricted spin scaling, empty spin, shell-image gauge changes,
source identity/ownership, image and grid refinement, and exact/one-short
resource admission. No real inverse-Bloch density assumption is made.

This leaf admits explicit 3D image/grid supports and l<=4; it alone does not
close the periodic KS capability row. The following increments qualify pure
LDA/GGA physical composition, grid/domain generation and public job outputs.
General XC, production accuracy profiles and other unqualified stages
remain required.


The subsequent native LDA/GGA component shares that finite Bloch support and
adds bounded AO gradients, spin-density invariants and their exact adjoint.
Independent Gaussian polynomial derivatives and libxc point evaluations,
complex real/imaginary energy differences, mixed-spin gradients, restricted
limits, empty spins, image gauges and exact resource boundaries qualify it.
The finite physical periodic KS composition below consumes this joint
functional energy contract, preserves the same k weights and confirms the
terminal evaluated density against the complete finite physical source.

The explicit physical pure-LDA/GGA KS driver now composes the qualified
Hartree and semilocal sources with physical S/T/V and nuclear energy.
The same-density joint energy contract applies to both restricted and
fixed-spin unrestricted occupations. Independent Gaussian/libxc contractions
and SciPy eigensolves check full SCF, capped evaluated epochs, separate
energies and final physical confirmation. Exact/one-short aggregate budgets
include both sources and the functional solver's retained energy scalar.
The finite Hartree pair support and canonical XC image support remain distinct;
grid charge, XC energy and XC potential require independent refinement.
Public pure-KS jobs have their own acceptance below. General functional
profiles are not inferred from this finite-source increment.

The explicit global-hybrid adapter separately adds full-Coulomb exchange to
the semilocal component on the same evaluated density. Resolved method
components supply the coefficient; the pure API still refuses hybrids.
Independent Gaussian pair Fourier integrals and AO gradient contractions,
libint one-electron matrices, libxc integrands and SciPy fixed-point solves
check PBE0, B3LYP and custom LDA/GGA hybrids. Both zero-mode policies have
independent probe-charge and spin-factor witnesses. Complex congruence
energy differences check the Fock derivative, including k weights and
conjugation. Capped epochs, source/image gauges, rank truncation and all five
exact/one-short aggregate resource limits pass. Separate empty-spin PBE0
and B3LYP tests measure libxc endpoint sensitivity without changing runtime
regularization or claiming uniform minority-potential accuracy. Full-matrix
empty-spin acceptance uses a well-conditioned B88/LYP global hybrid.

Exported hybrid snapshots were checked against the actual shared FockBuild
draft, including physical S/X frames, owned evaluated densities, separate
method energy components, solver-modified operator parentage and next
orbital projectors. This qualifies export compatibility with that inspected
draft, not a landed runtime adapter or an in-loop Python allocation contract.
No symmetry-reduced or BIPOLE producer is admitted by this evidence.
General periodic accuracy qualification remains required. Explicit range
separation has the separate acceptance below. Reproducible finite-source timing
and independent errors are recorded in
`benchmarks/manta_periodic_ks.md` in the repository.

The explicit range-separated adapter now independently qualifies complete
finite-source RKS/UKS for HSE06, CAM-B3LYP, LC-omegaPBE, omegaB97X, mixed ranges
and empty-spin LC-BLYP. The reference uses fixed published coefficients,
independent Gaussian Fourier panels and analytic screened kernels, libxc AO
contractions and SciPy SCF. Tests isolate the finite erfc zero and its k weight,
the long-range coefficient of the Madelung term, and the absence of that term
for screened-only exchange. Both requested zero-mode choices, complex density
energy derivatives, capped evaluated states, next projectors, closed-shell
limits, retained-rank solves, cancelled components and source-image gauges
pass. All five exact and one-short aggregate budgets include every kernel
and the expanded transfer source; invalid numerical arrays remain unread
after a resource refusal. Unsupported functional families stay gated.

The actual shared FockBuild/OperatorFrame draft also consumed nine exported
range-separated states, including early capped exits and converged results.
All 45 spin/k operators preserved physical frames, evaluated density,
component energy, immutable ownership and solver-modification parentage.
Independent generalized spectra and next projectors agree. This qualifies
export compatibility with that draft, not a landed in-loop adapter or a
symmetry certificate. Canonical Coulomb-plus-erfc zero-mode semantics are
explicit; no equivalence to another finite-mesh singularity prescription is
inferred. Timings and errors are recorded in
`benchmarks/manta_periodic_range_separated_ks.md`. Range-separated cutoff/grid
generation and public output have separate acceptance below; source refinement and
production accuracy profiles remain required. BIPOLE/CRYSTAL and symmetry
gates retain their own sources.

The range-separated cutoff and joint-grid factories reuse the qualified
HF/global-hybrid geometric producer, with complete range-separated admission.
Independent integer spheres check Gamma and shifted skew-cell G/Q support,
exact boundaries and distinct pair/XC image domains. HSE06, CAM-B3LYP, mixed
ranges and empty-spin LC-BLYP match explicit composition bitwise and an
independent Gaussian/libxc and SciPy SCF reference. Every erfc leaf retains
its finite diagonal zero; domain receipts describe the common nonzero base.
Both spin conventions, singularity choices and grid partitions, capped
evaluated states, AO-image gauges and standalone receipt lifetimes pass.
All five jointly exact and one-short caps precede payload copies or invalid
grid generation. The grid, geometric source and all exchange components
remain charged together. A further nine-state/45-block check against the
current FockBuild draft preserves domain/quadrature provenance and evaluated
state ownership. Public jobs have separate acceptance; no symmetry admission
follows from these factories.

Hybrid cutoff and joint-grid factories now have independent acceptance too.
They reuse the HF geometric planner and certified reciprocal generator with
complete hybrid admission; no HF solver proxy is used. Independent integer
enumeration covers skew Gamma and shifted multi-k G/Q spheres, exact cutoff
boundaries, pair-complete source translations and distinct canonical XC images.
Identical pair/G/Q supports retain the same child geometry identity as HF.
RKS, UKS, unequal/empty spins and both zero-mode policies match explicit source
composition and independent full SCF, Fock and individual energy terms.
Both atomic partition choices and capped evaluated states are covered. All
five whole-request exact/one-short caps precede enumeration or poisoned array
copies, and a coincident-center fixture proves joint refusal before partition
generation. Grid/source ownership is charged through the full hybrid solve.
These tests do not establish automatic grid accuracy, BIPOLE parity or a
symmetry-reduced producer. Public output and QVF qualification follows below.

Pure-KS cutoff generation now reuses the pair-complete direct source and
certified reciprocal source, with a separately selected canonical XC image
sphere. Count-only admission bounds all traversal, source ownership and the
complete physical KS solver; no exchange-transfer tables are constructed.
Independent integer spheres, generated/explicit equality, independent SCF,
AO-image gauge transport and exact/one-short budgets qualify these domains.
The standalone cutoff interface keeps caller-supplied quadrature; the joint
producer below supplies it explicitly, including in selected pure-KS jobs.
The empty-beta PBE witness also
requires a separate minority-potential conditioning check: the observed
libxc fp64 endpoint response amplifies one-epsilon input perturbations even
when energy and occupied-spin results agree tightly.

A bounded periodic atomic quadrature adapter now qualifies a finite 3D source:
shared generic radial/Lebedev rules and periodic Becke/Stratmann partitions,
canonical nuclei and folded points, owned exports and count-only work/memory
admission including OpenMP neighbor scratch. Independent radial, angular and
integer-image partition reconstruction checks the generated weights. Separate
Gaussian/libxc contractions check the same-grid energy/potential, and two
independent tensor-cell resolutions qualify the refinement reference.
Observed grid errors remain explicit: a larger grid is not an automatic
accuracy qualification, and partition-radius refinement matters separately.
This does not close automatic profiles or low-dimensional grid rows.


Joint grid/domain/KS admission now reserves every retained grid owner and
explicit source copy, subtracts grid work before downstream admission, and
checks the whole request before generating numerical points. Independent
finite-source SCF and bitwise equality with explicit composition cover RKS,
UKS, empty-spin BLYP, both partitions, converged and capped epochs. Exact
simultaneous and one-short memory/shell/primitive/scalar/kernel limits cover
the joint plan; invalid partition generation is not reached after aggregate
refusal. Source receipts and histories retain their existing semantics.


Selected public RKS/UKS jobs now use the joint grid/domain/SCF source with
explicit finite XC controls. Job-level independent Gaussian/libxc contraction
and SciPy SCF references cover PBE restricted and fixed-spin states, the
restricted UKS limit and empty-beta BLYP. QVF validation and exact density
readback cover the actual evaluated state, including capped epochs; separate
generalized eigensolves check terminal orbitals without reconstructing energy
from them. Gauge tests distinguish source-frame arrays from nuclear-frame QVF
and require a changed frame identity. Owned exports remain independent.

Before numerical generation, selected KS jobs reject unsupported functional
components, exchange sources, BIPOLE controls, symmetry reduction and each
insufficient resource cap. Grid-only dry runs and artifact delivery failures
preserve the standard lifecycle. Functional/grid citations, child source
identities and raw component energies survive output. Whole-job benchmarks
include artifact writing and report independent errors with modeled memory.
This is finite-source job acceptance, not BIPOLE/CRYSTAL parity, a certified
symmetry source, or general functional/grid accuracy qualification.

Selected global-hybrid RKS/UKS jobs now pass independent Gaussian/libxc
contractions and SciPy SCF checks under both exchange zero-mode conventions.
PBE0, B3LYP and custom LDA/GGA components cover Gamma, shifted meshes,
restricted, unequal and empty spins. Method-scaled exchange, its separate
correction and the unscaled K child receipt survive manifest, structured log
and QVF delivery. Component sums refer to the evaluated density, including
capped exits; independent generalized eigensolves verify the separate next
projector. Nuclear-frame QVF readback and source-frame Fock/gauge checks
preserve that distinction. All simultaneous resource caps pass exactly and
refuse one short after reserving job and provenance owners. Unsupported
functional components, BIPOLE and symmetry requests fail before generation;
dry-run and artifact-failure checks preserve the existing output lifecycle.
The public-job benchmark times complete artifact delivery and independently
checks SCF, every energy component and lossless QVF density. Finite-source
acceptance does not establish BIPOLE/CRYSTAL parity or a symmetry certificate.

Selected public range-separated RKS/UKS jobs now pass independent full SCF,
same-density energy/Fock, evaluated-epoch QVF and component-provenance checks.
HSE06, CAM-B3LYP, LC-omegaPBE, omegaB97X, distinct-range composites and
empty-spin LC-BLYP cover Gamma and shifted meshes. Cancelled descriptors
retain complete method admission. Screened-only jobs report the requested
convention separately from its inactive actual correction, with identical
source/D/F under changes to inactive controls. Signed component energies,
raw child K receipts and complete semilocal requirements survive JSON output.
All five exact/one-short job budgets include component receipt and serialized
owners; unsupported kernels and sources fail before generation. Capped states,
image gauges, immutable public exports, dry runs and delivery failures retain
their contracts. Nine public states also pass the actual FockBuild draft
adapter check with full delivered provenance. Method-owned convergence norms
and confirmation criteria remain native; a generic operator diagnostic does
not replace them. Whole-job benchmarks include standard artifact delivery.

The frozen-density grid assessment now has its own native aggregate admission
and an explicitly partial hybrid entry point. Independent Gaussian/libxc
contractions check global and range-separated LDA/GGA components, including
multiple ranges and empty spin. Complete hybrid builds at the same HCORE
density independently confirm that only the semilocal Fock/energy changes
when the grid changes. Pure-entry refusal, complete hybrid admission,
cancelled descriptors, source/gauge ownership and exact/one-short aggregate
resource caps retain their gates. Exchange and singular corrections remain
with the complete method, and partial potentials are never labelled Fock.

The pure grid assessment also has independent contraction checks.
LDA/PBE restricted and unrestricted states,
BLYP, empty-beta BLYP and complex non-TRIM k matrices check signed XC energy,
each spin population, the reference potential, and every spin/k maximum-entry
and Frobenius difference. All five resource caps pass exactly and refuse one
short before copying invalid numerical inputs. Tests cover count overflow,
retained reference and external storage, batch-independent source identities,
separate fixed-density identity, owned exports and AO-image gauge transport.

Its benchmark separates same-grid correctness from reference convergence.
The chosen atomic reference still has measurable energy and potential errors
against independent tensor-cell integration. The optional physical HeH/PBE
probe holds a confirmed finite-source UKS density fixed across four atomic
grids and three independent tensor resolutions. It tests reference adequacy,
not another SCF sequence; a stable charge integral cannot qualify a changing
minority-spin potential. No automatic accuracy profile follows from these
diagnostics. Image, functional, boundary, BIPOLE parity and source-specific
symmetry gates remain independent.

The harder HeH/PBE reference is now also tested with independent adaptive
cell integration. The benchmark uses Genz-Malik and two product
Gauss-Kronrod rules with explicit subdivision caps; energy, populations and
all real/imaginary spin/k potential entries participate in adaptation. The
vector witness is checked against same-point native contractions and complex
density energy derivatives, including restricted LDA/PBE and empty-beta BLYP.
Actual convergence status and inter-rule differences remain distinct from
estimated errors. This strengthens the fixed-fixture reference without
claiming an automatic grid profile or changing a product quadrature source.

### Finite lattice compact remainder

`tests/test_manta_lattice_remainder.py` compares the native positive integral
envelopes with independently evaluated incomplete-gamma formulas, including
optional 70-digit references. Signed compact point clouds test complete
finite Ewald cross energies, every charged pair and order selection. Near-set
transfers, cubic/skew cells, origin limits, support contact, charge/radius
scaling, covariance, positive underflow, source identity and exact/one-short
resource caps exercise the conditional contract. Previously accepted
coefficient matrices and identities remain unchanged.

The [remainder benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_lattice_remainder.md)
separates bound cost from coefficient construction and independent validation.
The recorded `vq` calculation covers 24 finite Ewald energies, all nine
charged pairs per case and four adaptive-order calculations. Every measured
error is below its admitted bound; at order 12 the maximum cross-energy
error is 2.38e-15 Eh. Source and native identities accompany the benchmark.
These compact finite-source calculations have no matched ORCA/CRYSTAL case.
Gaussian tails, omitted Ewald images and numerical roundoff are excluded from
this Taylor ledger; no production periodic BUPO or symmetry omission follows.

### Finite lattice Gaussian error ledger

`tests/test_manta_lattice_gaussian.py` checks all four native orbital-pair
and auxiliary-envelope combinations against independent analytic Gaussian
convolutions. Signed off-center p/d contractions use bivariate polynomial
Gaussian integration and refined radial quadrature, independent of native
Taylor coefficients and moment recurrences. Tests also cover complete
neutral three-Gaussian energies, separate compact/penetration/full-moment
terms, an explicit tight-source error target, absolute norms for neutral
components, units and density scaling, retained-moment requirements,
immutable identities, positive underflow and exact/one-short joint budgets.
Previously accepted coefficient matrices and identities remain unchanged.

The [Gaussian ledger benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_lattice_gaussian.md)
measures joint coefficient, remainder and error construction. The recorded
`vq` calculation passed 24 finite Gaussian energies, all nine pairs per case,
four signed angular combinations and four tight-source target calculations.
At order 12 the largest energy error was 8.89e-16 Eh. All four tight-source
cases met a 1e-8 bound at order 5. Diffuse-source bounds can be conservative
and need not decrease with order; the consumer must check the actual bound.
The benchmark records source and native identity and measured errors.

No matched ORCA/CRYSTAL reference exists for this arbitrary finite-source
profile. This ledger excludes omitted lattice/pair-image domains and
floating-point errors. Its envelope fingerprints do not authorize a Fock
source or symmetry omission. Production periodic BUPO/RI-BUPO, complete
accuracy profiles and point-nuclear self conventions remain separate work.

### Infinite Ewald cube-domain remainders

`tests/test_manta_lattice_domain.py` checks the complete integer-cube
omission bounds against independent infinite positive shell envelopes,
explicit real/reciprocal refinement and complete neutral two-ion Ewald
energies with regular point self. Signed compact charge clouds and translated
copies verify the finite periodicity-defect bound. Rotations, handedness,
skew cells, scaled units, zero norms, positive subnormals, kernel identity,
near transfer, invalid/incomplete domains and count-only resource refusal
are covered. Native matrix coefficients and compact remainders preserve the
saved prior bitwise fixtures.

The [domain-bound benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_lattice_domain.md)
records native bound construction and complete ionic energy comparisons.
Adaptive cube enlargement must meet an explicit 1e-8 Eh bound, with measured
error also below that target. The high-cutoff reference is an independent
SciPy erfc/cosine sum, including the self correction in Ewald (1921), IV.1,
Eqs. 38-40. A finite periodicity defect is measured rather than set to zero.
This qualifies conditional compact-source domain errors, not unbounded
Gaussian lattice errors, AO pair-image convergence or a full SCF profile.

Actual local `vq` evidence includes 24 fixed complete ionic energies, eight
adaptive target calculations and two signed compact translation checks.
All eight 1e-8 Eh targets are met at cube radii 5 through 8; the largest
accepted bound is 9.657e-9 Eh and measured error 3.89e-16 Eh. Explicit
finite periodicity defects are covered by conservative real-domain bounds.
Source/native pins, fetched numerical artifacts and reference limitations
are recorded in the benchmark.

### Periodic pair-image density source

`tests/test_manta_periodic_pairs.py` compares native shifted-shell moment
panels and density contractions with independent polynomial Gaussian
quadrature, including signed contractions and pure/Cartesian l=6 shells.
An unreduced ordered Bloch sum independently checks the q=0 conjugate-pair
identity. A nonreciprocal Fourier counterexample preserves its boundary.
General complex Hermitian densities, non-TRIM momenta, nonuniform/zero
weights, weighted adjoints and complete quadratic energy derivatives are
tested without assuming time reversal between k points. Ownership, source
identity, canonical image order, origin shifts, retained norms, zero density,
invalid domains and all five joint resource caps have separate gates.

The [periodic pair-source benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_pairs.md)
separates immutable source setup, forward density moments and adjoint AO
responses. Its complete finite Fourier Coulomb calculations use native
moments with a declared Taylor density approximation. Independent ordered
Gaussian Fourier integrals measure that approximation error; retained
absolute radial moments bound it. Same-order polynomial quadrature checks
the source itself. A converging diagnostic Taylor approximation does not
establish an SCF accuracy profile. The molecular extent helper refactor
preserves every field and identity in the saved regression fixtures.

The complete local calculation evidence covers 24 finite Fourier Coulomb
energies and four fixed quadratic density functionals through `vq`. Native
same-order Fourier discrepancies are at most 4.75e-15, half-trace errors
5.00e-16 Eh and finite-difference derivative errors 5.41e-15. All explicit
Fourier/energy bounds cover their errors. The expanded density set still
has order-12 truncation up to 2.264e-5 Eh; no universal convergence ratio or
SCF accuracy target is inferred. Source/runtime provenance and measured
results are recorded in the benchmark.

Periodic traversal, source-bound aggregate Coulomb error control, full charge
policy, omitted-image/lattice convergence and production Fock integration
remain required. The molecular bubble-tree inventory guard is unchanged.
No spatial symmetry, exchange or nonperiodic translation identity is inferred.

### Physical range fitting sources

`tests/test_manta_periodic_range_fitting.py` compares dense and streamed erfc/erf
metrics and couplings with libint short-range and independent Gaussian Fourier
references, including signed angular contractions. It checks physical/numerical
range separation, finite-zero ownership, the complementary-kernel identity,
image gauge and transfer conjugacy, fixed RI contractions, source/frame/density
identity separation, immutable controls and all five resource refusals.
Saved default Coulomb matrices, identities and counts remain unchanged.

Four complete finite Gamma SCF diagnostics combine the native fixed Coulomb-J,
physical-range K and semilocal XC components outside the bounded driver. Their
full functional energies and states are compared with independent
libint/Gaussian/NumPy responses and SciPy fixed points. These deliberately
specified finite functionals have no matched ORCA or CRYSTAL reference; external
parity is untested. They do not qualify production fitted range KS composition,
public jobs, source reduction, derivatives or a converged accuracy profile.
The measured source and calculation results are in the
[range-fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_range_fitting.md).

`tests/test_manta_periodic_ri_range_fock.py` extends physical-range acceptance to
one, two and four k points, including zero/nonuniform weights, signed exchange
coefficients, positive truncated and empty projectors, weighted variation and
reciprocity, image transport, exact resource boundaries and screened-J refusal.
Four complete shifted two-k functional SCF diagnostics independently compare
the whole-mesh native range K with per-q libint/Gaussian/NumPy contractions.
They retain separate bare-Coulomb J and actual functional energy. The original
Coulomb Fock baselines remain bitwise unchanged. See the
[whole-mesh range benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_range_fock.md).

### Bounded fitted range KS

`tests/test_manta_periodic_ri_range_ks.py` independently reconstructs each
Coulomb/erfc metric and coupling using libint and Gaussian Fourier references,
retains fixed NumPy projectors, and compares complete native SCF states with
SciPy fixed points. RI-J/exact-K, exact-J/RI-K and RI-J/RI-K cover Gamma and
shifted two-k RKS/UKS, HSE06, CAM-B3LYP, mixed physical ranges, other long-range
hybrids and empty beta spin. Tests retain actual functional energy, signed and
cancelled terms, finite screened zero and method-owned singular correction.

Further witnesses cover functional derivatives, capped evaluated epochs,
truncated/empty metrics, rectangular overlap spaces, immutable exports,
all five exact/one-short resource caps before payload copies, and selected
Python provenance admission. Shared Fock snapshots and complex multicentre
AO/auxiliary image transport keep the operator and symmetry seams qualified
for these sources. Sixteen saved prior fitted pure/global and direct range KS
states, plans and identities are unchanged. The
[bounded range KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_range_ks.md)
records 18 isolated complete calculations. No matched ORCA/CRYSTAL finite-source
reference is available; external parity and domain/grid convergence remain
untested. Generated range-fitting methods are qualified separately below;
public jobs follow below, while source reduction still needs its own acceptance.

### Generated fitted range KS

`tests/test_manta_periodic_ri_range_ks_domains.py` checks integer reciprocal
spheres in skew cells, full transfer reversal and kernel-independent geometry.
Complete generated RKS/UKS on Gamma and shifted meshes covers HSE06, CAM-B3LYP,
mixed physical ranges, exact/RI J/K choices and empty beta. Generated and
explicit matrices, spectra, energies and physical-source identities agree
bitwise. Independent libint/Gaussian/NumPy/libxc/SciPy SCFs compare complete
evaluated matrices, functional components and grid populations.

All five joint resource boundaries, selected output reservation, invalid-source
refusal before payload generation, owned arrays, geometry/projector identity
separation, shared Fock snapshots, rectangular overlap frames and complex
multicentre AO/auxiliary image transport are exercised. These tests do not
grant source-qualified symmetry skipping. The
[generated range KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_range_ks_domains.md)
records isolated complete native and selected calculations. Grid/cutoff
convergence and external ORCA/CRYSTAL parity remain unqualified.

### Public fitted range-KS jobs

`tests/test_manta_periodic_ri_range_ks_job.py` compares complete public RKS/UKS
calculations with native generated sources and independent functional SCFs.
HSE06, CAM-B3LYP and mixed ranges cover Gamma/shifted meshes and strict/BvK
conventions. Separate witnesses cover LC-wPBE, empty beta, truncated/empty
projectors, capped evaluated states and every exact/one-short resource limit.
Unqualified selections refuse before generation or calculated artifacts.

Returned results, manifests, logs and QVF retain the same energy, density,
frame and per-kernel source/projector evidence. QVF schema validation and
bitwise density recovery, complex AO/auxiliary image transport, rectangular
overlap frames and immutable shared Fock states guard the common interfaces.
Dry runs and synthetic delivery failures check writer lifecycle. The
[public range-KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_range_ks_job.md)
measures complete jobs and records independent comparisons, output sizes and
resource reservations. These finite-source calculations do not qualify an
accuracy profile, external parity, source reduction or derivatives.

### Full Gaussian infinite-domain remainder

`tests/test_manta_gaussian_domain.py` checks native pointwise Gaussian
majorants against every signed angular Cartesian/pure component, periodic
pair translations and large-coordinate product centers. All pair/auxiliary
source combinations compare their omitted Ewald errors with full Gaussian
convolutions and independent reciprocal Coulomb sums. Angular fixtures also
use polynomial Gaussian quadrature. Complete neutral smooth-charge energies
include Gaussian self integrals and adapt the cube to a declared 1e-8 Eh bound.
A native moment/finite-coefficient calculation combines the finite Gaussian
ledger and new domain bound with matching physical kernel identities.

The 48 focused checks include owned immutable sources, zero/subnormal and
unrepresentable bounds, exact resource caps, canonical identities, reciprocal
symmetry and payload-free resource refusal. Six previous extent inventories
and four compact-domain receipts remain bitwise unchanged. The
[measured benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_gaussian_domain.md)
reports six isolated workers and the external-reference limitation. This
primitive does not qualify omitted AO pair images, density aggregation,
whole-method accuracy, periodic traversal, symmetry or derivatives.

### Periodic bubble source aggregation

`tests/test_manta_periodic_bubbles.py` checks the native hierarchy against
independent multinomial and Gauss-Hermite moments, absolute radial bounds,
weighted forward/reverse identities and energy derivatives through order 12.
Signed arbitrary complex whole-mesh densities, nonuniform weights, image
cutoffs, Cartesian/pure shells, deterministic reordering, source lifetime,
owned exports and zero/subnormal cases remain explicit. Exhaustive index
expansion proves that accept/refine traversals cover every retained unordered
pair interaction once. Molecular inventory validation is unchanged.

Complete finite Fourier Coulomb actions compare to unreduced ordered Gaussian
Bloch sums. The tests distinguish source pairing from Taylor error and compare
the latter to absolute full-moment bounds. Joint resource caps and shape-only
refusal protect hierarchy construction and driver copies. The
[hierarchy benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bubbles.md)
records timings, runtime provenance, references and limitations. This advances
P3b aggregation and traversal geometry; kernel-dependent admission, omitted
pair-image error and production periodic BUPO accuracy remain open.

### Complete canonical finite Ewald direct response

`tests/test_manta_periodic_ewald.py` compares every complex J element and
separate real, reciprocal and gauge energies with independent full Gaussian
convolutions. Signed contractions, Cartesian/pure p and d components,
coincident Gaussian self integrals and cubic/skew cells are included.
Angular references use bivariate Gauss-Hermite and Laplace quadrature,
with quadrature refinement. The reference uses a stable erf difference
near coincident Gaussian centers and an erfc difference at long distance.

General complex Hermitian whole-mesh densities, nonuniform and zero weights,
two-step energy derivatives, linearity and cross-k responses test the weighted
action. Reciprocal energies also compare to unreduced ordered Bloch Fourier
sums. Enlarged real/reciprocal domains converge to an independent reciprocal
Gaussian Coulomb limit at two eta values. This is evidence for the tested
finite sources; it is not a general omitted-domain or accuracy-profile proof.

The 31 focused tests include exact caps, count-only refusal before payload
reads, strict/unaligned arrays, immutable ownership, subnormal-scale response,
domain reordering and origin/reciprocal/time-reversal covariance. Three old
lattice matrices/identities and one pair contraction/identity remain bitwise
unchanged. The
[direct Ewald benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ewald.md)
records measurements, runtime and actual calculation evidence. No matched
ORCA/CRYSTAL profile is available for this arbitrary-density folded finite
quadratic form. Production kernel-admitted BUPO traversal, pair-image and
aggregate error controls, full charge policy and SCF selection remain open.

### Bound-admitted finite periodic BUPO response

`tests/test_manta_periodic_bupo.py` exercises accepted and direct blocks of
one complete finite Coulomb calculation. Full signed Gaussian convolutions
provide an independent scalar reference. Cartesian/pure angular sources
compare every complex J element and energy with the already independently
qualified native direct Ewald source. Cubic/skew cells, nonzero pair images,
nonuniform/zero mesh weights and raw moments through order 12 are covered.
The fixed partition gives linear J and the weighted derivative of a single
quadratic energy, tested with two finite-difference steps.

The tests check aggregate energy and per-AO J bounds, density scaling,
subnormal response, immutable source matching, exact resource caps and
refusal before payload reads. Periodically overlapping candidates refine
even when their central-cell geometry is separated. Zero tolerance and wide
Gaussian extents with inadequate error bounds use the complete direct action.
Moving identical real images between bare-near and complementary-tail terms
preserves the finite kernel and full Cartesian trace contributions.

The [finite BUPO benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bupo.md)
records actual calculation evidence, timings and source/runtime provenance.
Four saved direct-source calculations are bitwise unchanged by the shared
block-helper extraction, and the existing direct and Gaussian-error checks
pass. These bounds compare with the same retained finite canonical kernel;
they exclude floating-point roundoff, omitted pair images and infinite-domain
error. They do not qualify production BUPO/RI-BUPO profiles, the complete
electronic+nuclear charge policy, exchange, source reduction or derivatives
with respect to geometry.

### Retained-source infinite-Ewald error aggregation

`tests/test_manta_periodic_bupo_domain.py` compares complete complex J and
energy with independently integrated bare reciprocal Gaussian Coulomb limits.
Scalar and Cartesian/pure angular sources, both cells and boundary-crossing
pair images are included. Reciprocal-reference refinement is checked.
Tight mixed BUPO partitions compare with enlarged independently integrated
Ewald domains. An independent dense contraction of the public primitive
domain bounds verifies diagonal half-energy factors, coefficient norms and
both Bloch scatter arms, including same-AO nonzero images.

Density scaling, zero weights/density, positive underflow, exact resource
caps and immutable owned exports remain explicit. Domain checks refuse
incomplete cubes and inadequate omitted-center separation. Near transfers
leave the domain-bound identity and values unchanged. A complete atomic
periodic pair source enlarges its domains until both energy and J bounds
fall below 1e-8 at two eta values. This target concerns the retained source,
not omitted pair images or a production SCF profile.

Four saved finite BUPO responses, bounds and identities remain bitwise
unchanged. Existing Gaussian-domain and finite-BUPO checks pass. The
[domain-bound benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bupo_domain.md)
records calculations, runtime provenance and measured overhead. The
infinite-minus-finite bound includes the finite kernel's covariance error
relative to the limiting retained periodic source. Omitted AO pair images,
roundoff, full-cell charge policy, production selection, RI and geometry
derivatives remain open.


### Matched electronic and point-nuclear composition

`tests/test_manta_periodic_electrostatic.py` resolves signed s/p/d Gaussian
attraction against independent Laplace quadrature and Gaussian characteristic
functions, and point energies against independent ordered Ewald sums. Cubic
and skew cells, pure and Cartesian components, original mesh weights, zero
weights, point permutations and zero charges are covered. A tight neutral
source is compared with a complete ordered-image bare reciprocal electronic
reference plus independently converged point Ewald energy at two physical
eta values. Separate image and reciprocal refinements check the reference.

The checks distinguish retained charge from omitted-image uncertainty,
neutral refusal from explicit uniform background, Gaussian self interaction
from point self removal, and J+V from half-trace energy. Whole-mesh density
derivatives preserve the fixed electronic partition. Deliberately missing
central targets and small Ewald domains exercise attraction/point bounds;
all five resource limits precede payload reads. Source/result arrays are
owned and receipt fields immutable. The electronic source itself retains
its previous qualification and independent numerical evidence.

The [electrostatic benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_electrostatic.md)
records local queued calculation evidence, references, measured errors,
setup/build cost and remaining limitations. No matched ORCA/CRYSTAL
arbitrary-density component profile is available; this is not an external
SCF parity claim. Production BUPO accuracy-profile/SCF selection, q-aware
RI-BUPO and the remaining method matrix stay open.

Native reciprocal-cutoff selection over that immutable BUPO HF source now
passes 12 focused and 38 related checks, plus five complete queue-interface
calculation scenarios. Independent integer enumeration validates every
executed sphere; Gaussian/Fourier integration and separate SCF validate
Gamma/skew RHF, shifted empty-beta/doublet UHF and a capped doublet. Ten
accepted native runs preserve factory/explicit-control D/F/H/S/E bitwise.
Maximum Fock, same-density energy, relaxed-energy and density discrepancies
are 2.621e-13, 1.765e-11, 1.892e-11 and 9.750e-10 in atomic units, within
the unchanged comparison tolerances. Two insufficient cutoffs refuse their
accuracy targets. Shifted odd meshes, left-handed/skew cells, source
immutability, opposite support, all five resource caps, output provenance
and evaluated shared-Fock state retain their separate checks.
The [cutoff report](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bupo_domains.md)
records complete timing/memory and reference limitations. This selects only
exchange spheres over a supplied source. Automatic AO/Ewald selection,
whole-method profiles and q-aware RI remain open. The public job route
below composes the subsequent native source preparation.


### Public BUPO HF job acceptance

The typed public job increment composes the already accepted prepared
native BUPO source. Five actual `vq` calculations preserve the accepted
RHF/UHF D/F/H/S/E and source identities exactly. Independently integrated
Gaussian/Fourier/eigensolver references are reused with pinned artifacts
and unchanged physical inputs. Maximum same-density energy and Fock
differences are 1.765e-11 hartree and 2.621e-13; evaluated spin-density QVF
round trips have zero discrepancy for the accepted five cases. Capped
states remain nonconverged and unconfirmed. All evaluated-density integral
targets pass without tolerance changes. The shared Fock state contract and
unreduced periodic domain ownership are preserved; molecular symmetry
consumer qualification does not authorize periodic reduction.

Focused checks additionally cover independent basis-image gauge transport,
dry-run output discovery without source construction, and exact/one-short
complete-job resource caps. See the
[public job benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bupo_job.md)
for source/runtime provenance, measured costs and reference limitations.
Automatic whole-method profiles, q-aware RI-BUPO, lower-dimensional
Green functions, CCM/chi adapters, producer-qualified symmetry, derivatives
and acceleration remain open matrix items.

## Exact paired periodic RI Coulomb baseline

The fixed q=0 J-only source now executes two raw coupling sweeps when its
retained-mode schedule needs more than two. Rank, native metric factor,
finite Gram matrix, phase convention and physical support stay fixed.
The actual local `vq` run covers seven complete public states and seven
retained-mode controls: RHF, UHF including empty beta, pure RKS/UKS and
capped HF/KS states. Independent raw integrals, Gaussian Fourier sums,
factorization and SCF pass with maximum Fock and same-density energy
differences 5.152e-14 and 9.415e-14 hartree. Evaluated-density QVF and shared
Fock state checks pass without granting source symmetry.

The rank-six trial benchmark reduces coupling evaluations from six passes
to two under the unchanged conservative admission. Twenty-five new tests
cover independent full/truncated/zero-rank fits, weighted adjoints and
energy derivatives, image gauges, target separation and all five caps.
See the [paired RI benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_paired.md)
for measurements, provenance and the lack of a matched ORCA/CRYSTAL finite
source reference. This is the exact baseline for later paired approximate
couplings; approximate periodic RI-BUPO and the rest of the matrix remain.

### Shared periodic coupling action acceptance

The exact paired RI baseline now consumes the producer-owned forward and
adjoint actions on `PeriodicGDFTiles`. Independent complex transition inputs
at Gamma and nonzero transfer, Coulomb/erfc/erf kernels, angular contractions,
image frames, optional metric storage and all five action resource caps pass.
An actual local `vq` calculation checks seven complete HF/KS paired states
and seven retained-mode controls against independent finite-source SCF
references. Shared Fock and QVF evaluated states remain consistent. The
[report](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_gdf_actions.md) records measured
errors, unchanged acceptance gates, runtime provenance and action costs.
This establishes the source boundary for the next approximation; approximate
periodic RI-BUPO and the other required 0.18.x capabilities remain open.


## Fixed-metric periodic coupling screening

The first approximate periodic RI-J coupling uses the existing ordered GDF
Gaussian-envelope selector. Its zero-screen finite metric and native retained
factor remain unchanged. Paired actions supply density-specific J and energy
omission bounds; positive coupling screening refuses fitted exchange.
Focused qualification covers independent libint short-range and Gaussian
Fourier contractions, complex adjoints, nonuniform/zero weights, retained
ranks zero through six, energy derivatives, joint admission and complete
evaluated HF states. Public KS uses the same J source and keeps its own XC.

The [screening acceptance record](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_screen.md)
records calculation evidence and measured costs. These bounds exclude
arithmetic, finite-domain, RI basis and relaxed-density errors. This is an
experimental bounded screening baseline, not completed hierarchical
RI-BUPO or an automatic production accuracy profile. P3b and the full 0.18.x
capability matrix remain open.


## Ordered periodic RI multipole integration

The experimental `coupling_multipole_error` path replaces admitted far erfc
image interactions inside the existing GDF producer. Validate raw complex
maps at Gamma and nonzero q, dense/tiled equivalence, signed Cartesian/pure
shells, image gauges, reciprocal relabelling, exact metric/factor identity,
rank-zero and weighted limits, energy derivatives and all joint resource caps.
Public RHF/UHF and RKS/UKS states must retain their evaluated shared Fock,
independent exchange/XC, QVF density and approximation receipt. An actual
local `vq` calculation is required for acceptance, with source/native pins
and independent finite integrals and SCF algebra. ORCA/CRYSTAL comparisons
remain unavailable for this custom finite-domain test; no bulk parity claim
is inferred. See the
[multipole benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_multipole.md).

The shell-leaf prototype is not the completed hierarchy. Bubble aggregation,
upward/downward translations, useful automatic error allocation and large-cell
scaling remain part of P3b and the full 0.18.x goal.


The ordered GDF auxiliary-image hierarchy is an additional P3b increment.
Validation reuses the exact finite-domain and paired-map references, and
adds nontrivial multi-image nodes, upward translation counts, image/reciprocal
gauge checks, density-independent partitions and joint tree/moment/envelope
admission. Public HF and KS calculations must exercise the hierarchy through
vq and retain the shared Fock receipt. The global density-compressed hierarchy,
global AO-pair aggregation, automatic accuracy and source-qualified symmetry
remain open; this bounded increment does not close P3b or the full design.


Producer-owned Gaussian preparation is shared across periodic RI paired
actions and complete SCF builds. `tests/test_manta_periodic_ri_preparation.py`
checks bitwise cold/warm/streaming/dense equality, concurrent first publication,
independent source lifetimes, unchanged identities and exact metrics, full
storage/work admission before bad inputs, and lazy always-near behavior.
Public HF/KS tests check reused preparation in the evaluated Fock receipt;
grouped-node HF uses auxiliary cutoff 12 bohr. Complete changed-method
acceptance still runs through local vq with independent finite references.
This closes repeated Gaussian preparation. Compressed actions and bounded
partition reuse are qualified below; global AO-pair aggregation and automatic
whole-method accuracy remain open within P3b and the full approved design.


Compressed hierarchical periodic RI actions now qualify the input-moment and
output-field path over the existing selected auxiliary-image tree. Every
operator column and complex adjoint is compared with independently materialized
tiles; additional checks cover angular signed contractions, atom/reciprocal
gauges, density-independent partitions, positive magnitude envelopes, all
resource limits, fixed metric/rank and weighted energy derivatives. Public
HF/KS calculations exercise compression through vq and retain shared Fock,
energy and saved-density evidence. Benchmarks compare materialized and
compressed actions on the same source. This advances P3b but leaves global
AO-pair aggregation and automatic accuracy open; bounded partition reuse is
qualified below.


`tests/test_manta_periodic_ri_partition.py` qualifies source-owned bounded
partition reuse for compressed and materialized actions at Gamma/nonzero q
and orders zero/two/four. It checks bitwise cold/warm/reconstructed maps and
dense tiles, preserved identities/error ledgers, zero and arbitrary complex
inputs, concurrent publication, source lifetime isolation, eviction and
oversized fallback, cold/warm resource admission before invalid inputs,
inactive exact/leaf/reciprocal routes and public HF/KS Fock receipts.

243 affected scientific source checks pass. Benchmarking compares explicit
zero and 64-MiB capacities on the same source and metric, recording actual
runtime/content pins, first and warm calls, partition work and admitted peaks.
Acceptance requires complete local vq calculations and independent assessment
of fetched arrays; the benchmark record contains that evidence. External
ORCA/CRYSTAL parity remains unavailable for the custom finite-domain fixtures.
Global AO-pair aggregation, useful automatic accuracy and the other approved
0.18.x capabilities remain open. See the
[partition benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_partition.md).


Call-wide periodic RI moment actions add 40 focused source checks in
`tests/test_manta_periodic_ri_global.py`. They compare both complex maps with
per-tile compression and separately materialized coupling, including every
operator column, Gamma/nonzero-q, orders zero/two/four, signed Cartesian and
spherical d shells, atom/reciprocal gauges, linear and zero inputs, fixed
metric ranks and weighted energy derivatives. Concurrent calls verify private
input state. Source and parent resource checks run before invalid inputs;
inactive routes reserve no global storage. Public HF/KS checks retain the
shared evaluated Fock and new execution counters.

Benchmark the unchanged source and metric with the policy both disabled and
enabled, recording native/content pins, first and repeated calls, action work
and admitted peaks. Complete local vq calculations must then exercise the
changed HF/KS path and independently assess fetched arrays, approximation
bounds and output density. Custom finite-domain references use independent
Gaussian/Boys/Fourier and libint integrals, NumPy contraction and SciPy SCF;
they do not supply ORCA/CRYSTAL bulk parity. See the
[global-action benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_global.md).
This qualifies call-wide gather/scatter only. Global pair-bubble traversal,
automatic accuracy and the remaining full 0.18.x matrix remain open.


`tests/test_manta_periodic_gdf_pair_tree.py` qualifies the shared ordered GDF
pair tree independently of a new Fock route. It compares every pair/image
with an unreduced integer sphere, every classified node and complete partition
with brute-force capsules, and the full auxiliary union with independent image
enumeration. It checks cutoff equality, separate AO/auxiliary image frames,
reciprocal relabeling, signed angular extents, source lifetime, concurrent
queries and every resource cap before preparation. Exact s-Gaussian coupling
reconstructed through the native tree masks and phases must agree with libint
over the complete finite support.

Local vq acceptance now passes four complete finite-source calculations through
the native masks and phases, seven public HF/KS states and six raw complex-action
regressions. Independent assessment of fetched arrays verifies both complete
Coulomb maps and retained-metric responses. All 1,874 source/reference pins and
63 managed output hashes match; 361 source checks and five documentation checks
pass. Maximum tree-source coupling and response discrepancies are 1.111e-15 and
1.195e-14; public Fock/energy/density remain within their existing gates.
Gaussian/Boys/Fourier, libint and independent NumPy/SciPy references qualify
these finite fixtures; no matched ORCA/CRYSTAL reference is available. Those source-geometry calculations used the earlier per-pair hierarchy;
the subsequent global-action acceptance is recorded below. Automatic accuracy
and the remaining P3b and full 0.18.x gates remain open. See the
[pair-tree benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_gdf_pair_tree.md).


`tests/test_manta_periodic_ri_pair_tree.py` and the corresponding local vq
calculation now qualify the opt-in global pair-tree contraction. All 61 new
checks and 611 affected checks pass, including both complex maps, independent
finite-source error witnesses, Gamma conjugate-support refusal, dense/raw
exports, physical erfc, gauges, fixed metric ranks, energy derivatives,
concurrent call-private state and complete resource boundaries. Seven complete
public HF/KS states and ten raw cases pass on the same pinned native runtime;
63 managed outputs and 17 fetched arrays verify. The independently contracted
Fock, energy, density and approximation ledgers retain their existing gates.
No matched ORCA/CRYSTAL finite-domain reference is available. Benchmark and
provenance are in the [pair-tree action record](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_pair_tree.md).
This is an explicit pair-tree by auxiliary-image traversal. Useful automatic
accuracy, dual-tree/general production scaling and all other unqualified
0.18.x capabilities remain required.


The same pair-tree suite now qualifies per-shell-block error and magnitude
ledgers. Its 80 checks include 19 new sparse/complex-input, duplicate-output,
large-scale Euclidean-bound and fixed-rank propagation cases; 437 affected
checks and 12 documentation checks also pass. Complete local vq acceptance
passes the same seven public HF/KS states and ten raw cases on the new native
runtime. An independent NumPy-only assessment verifies both action norms and
all fixed-density energy/J bounds. Against the accepted predecessor, 146
arrays and the numerical source/factor identities are unchanged. Updated
bounds and costs are in the [pair-tree action record](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_pair_tree.md).
These tighter bounds still do not establish automatic useful accuracy;
the unchanged custom-domain reference limitations and remaining full matrix
requirements apply.


### Joint erfc radial and interior-moment bounds

The shared isolated erfc producer now qualifies a joint polynomial/Gaussian
radial envelope and an interior absolute-moment scale. Its explicit Taylor
derivation is in the scientific contract. This changes accepted partitions,
so approximate source identities are versioned; exact metric and finite
reference identities remain unchanged. Gaussian penetration and full-moment
tails are retained, and finite lattice complement bounds are unchanged.

The increment has 685 passing scientific checks (31 new), one skip and
12 documentation checks. After integrating current BIPOLE symmetry/Fourier
work and rebuilding, 134 directly affected checks pass on the final runtime.
Complete local vq acceptance adds nine public HF/KS states and fourteen raw
cases, including tight Gamma/nonzero-q actions and complete RHF at orders
two/four with requested coupling error 1e-8. A NumPy-only fetched-array
assessment verifies component/Euclidean action bounds, propagated fixed-density
J/energy bounds, independent SCF references and managed output. Seventeen
predecessor cases retain bitwise-identical exact reference arrays.

The [explicit accuracy benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_accuracy.md)
records source/native provenance, timings, measured errors, conservative bounds
and reference limitations. Fixed-density energy bounds reach below 8.2e-10
hartree on its small fixture, while complete RHF remains slower than exact RI.
Coupling tolerance is not whole-method accuracy; no automatic profile, general
production speedup, external bulk parity or additional symmetry permission is
claimed. All remaining required MANTA 0.18.x capabilities stay open.


### Lossless high-order partition retention

The global pair-tree source now retains only structurally allowed interaction
coefficients and its actual accepted antichain. A single expansion restores
the same dense matrix before contraction. The existing separately admitted
construction workspace permits cache charging and eviction by actual entry
bytes; oversized entries are used once without retention. The numerical
source, partition decisions and all error ledgers remain unchanged.

All 373 focused scientific checks pass, including 13 new cache/high-order
cases; twelve documentation checks pass. Complete local vq acceptance passes
ten public HF/KS states and sixteen raw cases, adding order-six RHF and
strict-tolerance Gamma/nonzero-q maps. A separate NumPy-only assessment finds
340 result arrays bitwise unchanged across all 23 accepted predecessor cases.
The [accuracy and cost record](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_accuracy.md)
provides source/native provenance, references and remaining limitations.
Order-six repeated RI-J improves about sevenfold on the bounded fixture, while
complete RHF remains slower than exact RI. This qualifies lossless storage
reuse, not automatic whole-method accuracy or a production crossover. The full
remaining 0.18.x capability matrix stays required.


### Componentwise retained-factor bounds

The shared GDF actions now expose per-output omission and exact-reference
auxiliary-column magnitude envelopes. Periodic RI propagates these through
the actual retained metric factor and keeps the older scalar bounds when
tighter. Numerical values, partitions, fixed metric/projector and source
identities are unchanged. Gamma envelopes follow the same conjugate pair
completion as the source; no new symmetry permission is inferred.

383 scientific checks and eleven documentation checks pass. Thirteen new
cases cover five producer routes, complex Gamma/non-Gamma maps, zero and
large inputs, owned exports and adversarial errors at ranks zero/one/six.
Local vq passes ten public HF/KS states and sixteen raw cases. Independent
NumPy assessment verifies component bounds, exact-reference column magnitudes
and retained-factor error transport. All 1,878 source pins, 90 outputs and
26 archives verify. Across the same 26 predecessor cases, 364 numerical and
reference arrays remain bitwise identical, with matching source/factor IDs.

The [accuracy benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_accuracy.md)
records provenance, limits and cost. Fixed-density J bounds improve by
5.8-14.3 times and energy bounds by 1.28-4.06 times on its small rank-six
fixture. Complete strict RHF energy bounds improve only about one percent;
automatic useful accuracy and production crossover are not established.
No matched ORCA/CRYSTAL custom-domain reference is available. Existing
Gaussian/Boys/Fourier, libint and NumPy/SciPy references apply; KS shares libxc.
Basis/domain, arithmetic and relaxation errors remain separate. All other
unqualified MANTA 0.18.x capabilities remain required.


### Fixed-envelope automatic periodic RI coupling selection

The selector admits all trial work and overlapping source storage, then uses
shared producer omission/magnitude envelopes and the actual retained metric
factor to certify J and Coulomb-energy targets for a declared AO density L1
limit. It fixes the source before SCF, checks the limit on every evaluated
density and refuses unsupported routes or unmet targets. The public receipt
records selected/requested controls and the certificate. Independent K/XC,
unified Fock and source-qualified symmetry ownership remain unchanged.

899 focused checks pass, including 25 new cases for metric ranks zero/one/six,
complex weighted densities, exact resource limits, invalid/exhausted controls,
HF/KS, global-hybrid and physical range composition. Local vq passes ten public
states and four hybrid/range cases. Independent NumPy assessment verifies the
envelope at 160 additional Hermitian densities, all source/output pins and
unchanged exact-reference metrics/couplings. The
[accuracy and cost benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_accuracy.md)
records source/native provenance, references and memory/cost limitations.

Both 1e-6 J/energy targets pass for the complete RHF fixture after two source
trials, but its automatic runs remain slower than exact finite RI. No matched
ORCA/CRYSTAL custom-domain reference is available; existing independent finite
Gaussian/Boys/Fourier, libint and NumPy/SciPy witnesses apply. This qualifies
automatic coupling selection only. Whole-method error, useful performance,
remaining source adapters, lower-dimensional boundaries, source-qualified
symmetry, derivatives/stress and every other unqualified 0.18.x row stay open.


### Overlap-derived automatic periodic RI density limits

The complete HF/KS path now obtains a fixed density L1 envelope from the
shared bounded solver's canonical overlap and occupation contract when the
requested limit is zero. Raw RI builders retain the explicit-limit contract.
Count-only admission covers the added eigenanalysis, work and owned exports.
The method selects its coupling source once and still checks every density.

712 focused checks pass, including 26 new cases for independent complex
occupied subspaces, overlap rank/gauge/scaling, resource refusal and complete
method equivalence. Local vq passes ten public HF/KS states plus four
global-hybrid/physical-range cases. A separate NumPy assessment reconstructs
the envelope, checks 480 independent occupied subspaces and 160 further
Hermitian densities, verifies source/output pins and reuses unchanged exact
metric/coupling references. Automatic and explicit resolved-limit states
match bit for bit. Unified Fock, physical K/XC, BIPOLE/CRYSTAL and symmetry
contracts are preserved.

The RHF benchmark derives L=784.535933 and reaches the 1e-6 J/energy targets
through exact finite-coupling fallback at orders two/four/six. The UHF hydrogen
case retains nonzero approximate far blocks. The
[accuracy and cost benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_accuracy.md)
records source/native provenance, numerical comparisons and the conservative
runtime cost. No matched ORCA/CRYSTAL custom-domain reference is available;
Gaussian/Boys/Fourier, libint and NumPy/SciPy witnesses apply, with shared
libxc for KS. Basis/domain, metric solve, arithmetic and relaxation errors
remain separate. Automatic whole-method accuracy and production performance
are unfinished, along with every other unqualified full-design capability.


### Componentwise automatic periodic RI allocation

The shared canonical-overlap helper now supplies an entrywise envelope for
the weighted spin-total density as well as its total L1 limit. Source-owned
absolute action and omission envelopes preserve shell-block information,
Gamma conjugate-orbit transport and exact residuals. The selector propagates
them through the actual retained factor and probes the same adjoint partition.
It admits all work/storage before setup, fixes both contracts before SCF,
and rejects any evaluated density exceeding either constraint. Explicit
positive L without entry limits retains the scalar selector.

884 numerical checks and 48 documentation checks pass. Local vq passes ten
public HF/KS states and four hybrid/range cases. Independent NumPy assessment
checks 480 occupied subspaces, 160 further densities, both constraint families
and the full certificate. Exact reference arrays and metric/factor identities
are unchanged. Explicit resolved total/entry controls reproduce automatic
states bit for bit. No new Fock ownership, physical K/XC, BIPOLE/CRYSTAL or
symmetry policy is introduced. The
[accuracy and cost benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_accuracy.md)
records source/native provenance and matched scalar controls.

The small RHF fixture accepts looser tolerances but still falls back to exact
finite coupling and pays extra setup cost. UHF hydrogen retains approximate
far blocks under both requested targets. No matched ORCA/CRYSTAL custom-domain
reference is available; finite Gaussian/Boys/Fourier, libint and NumPy/SciPy
witnesses apply, with shared libxc for KS. Basis/domain, metric solve,
arithmetic and relaxation remain separate errors. Automatic useful whole-method
accuracy and production performance remain open with the other required rows.


### Automatic RI trial selection without premature exact fallback

Measured fixed-source curves exposed a certified approximate interval that
the former target-ratio step skipped. The selector now uses a less aggressive
bounded proposal and handles positive-double underflow. Shared producer
bounds still decide every acceptance, and source/metric identities remain
fixed before SCF. No additional physical source or symmetry policy is added.

1,120 numerical checks and 48 documentation checks pass. Ten new cases cover
three starting tolerances at orders two/four/six and an underflowing estimate.
Local vq passes ten public HF/KS states and four hybrid/range cases; separate
NumPy assessment verifies all certificates, 480 occupied subspaces, 160 further
densities, numerical reference identities and source/output provenance.
The three RHF orders now retain nonzero far blocks while meeting both 1e-6
targets in two trials, with explicit-source state equivalence. Exact finite RI
remains faster on this small case. The
[accuracy and cost benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_accuracy.md)
records measured bounds, costs and source/native identities.

No matched ORCA/CRYSTAL custom-domain reference is available; independent
finite Gaussian/Boys/Fourier, libint and NumPy/SciPy witnesses apply, with
shared libxc for KS. This is finite-coupling allocation, not whole-method
accuracy. Basis/domain, metric solve, arithmetic, density relaxation, useful
performance and every other unqualified full-design capability remain open.


### Complete periodic RI image-accuracy admission

A separate pre-SCF ledger now reuses the shared full-image overlap/kinetic
producer and canonical density envelope in RI HF, pure KS, global hybrids
and physical range compositions, including exact J with RI K. Explicit
completion cubes can be smaller than retained inventories while preserving
a conservative positive exterior bound. Same-support physical states and
unified Fock identities are unchanged. All five resource caps, invalid
controls, immutable target snapshots and tight-target refusals are checked.

Source `b33fa2d7be9c40621964c6f17c3a1326232836b4` passes 46 image checks and 1,904 affected regressions.
Local vq acceptance covers twelve public states, independent Gaussian
quadrature, NumPy/Decimal contraction and 1536 occupied subspaces.
A tighter-support complete RHF comparison uses independent finite integral
construction and SciPy SCF. Existing hybrid/range reference evidence is
reused; no matching external custom-domain reference is available. The
[RI image benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_image_accuracy.md)
records measured bounds, overhead, resource provenance and limitations.

This qualifies only S/T image errors and kinetic energy in the same finite
canonical density family. Nuclear, two-electron, metric/solve, arithmetic,
overlap-rank changes and relaxed-state accuracy remain outside the ledger.
Automatic whole-method RI-BUPO profiles, useful performance crossover,
low-dimensional sources, conditional CCM/chi adapters, qualified symmetry,
remaining derivatives/stress and acceleration rows remain unfinished.


### Uniform canonical exchange envelope

The shared producer is exercised on signed contracted s and Cartesian p
sources, complex occupied subspaces, unequal and empty spin populations,
rank truncation, original mesh order, and strict uniform-weight refusal.
Independent contour-shifted Gaussian quadrature compares retained exchange
with enlarged AO and reciprocal sums. Complete local RI HF calculations
exercise the new bound at their evaluated states; these do not yet impose a
complete RI exchange target. Count-only exact and one-short caps precede
input values, typed reuse preserves the certificate, and exports own their
read-only arrays. The occupied-space benchmark records constructor, overlap
analysis and reused-envelope costs separately. No production speedup or
whole-method accuracy follows from this prerequisite.

Local vq acceptance of this increment passed eight cases and four complete
finite HF calculations, including three RI and one BUPO case. Independent Fourier/occupied-space references and
70-digit positive contractions checked 576 random spin-pair states. The
benchmark records actual source/native identities and limitations. A missing
import in three test wrappers was corrected separately; numerical code and
the source-pinned calculation payload were unaffected. That prerequisite did not yet impose complete RI exchange targets;
the adapter below now adds them for qualified full-Coulomb methods.


### RI exchange and fixed-BvK admission

Exercise optional pre-SCF targets in RHF, unequal-spin UHF, empty beta and
restricted/unrestricted global hybrids. Enabled/disabled controls must retain
bitwise-identical evaluated states and source identities, including capped
iterations and output densities. Reconstruct BvK positive bounds independently
and compare finite/full Gaussian overlaps at fixed occupied subspaces.
Admit exact and one-short memory, shell, primitive, scalar and kernel budgets
before poisoned numerical payloads. Refuse unqualified fitted K, pure KS and
physical range sources. Test the explicit AO cube's retained-support guard
and preserve standalone default-source identities against accepted evidence.
A compact complete RI calculation must pass tight targets; loose diffuse
bounds must remain visible rather than be relabeled as practical accuracy.

This increment passed local queue acceptance on eight complete RI states,
512 random spin-pair subspaces, independent Gaussian/BvK witnesses and
70-digit positive contractions. Two prior standalone source certificates
and one PBE0 state were reused with bitwise comparisons. Independent finite
HF Focks/energies and converged SCF references passed. The compact strict
and fixed-BvK sources both meet 1e-8 matrix and energy targets. Source
checks pass 1,066 affected cases. The
[RI exchange benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_exchange_accuracy.md)
records exact implementation/native identities, measured errors and costs.
No matching ORCA/CRYSTAL custom finite-domain reference is available.
This leaves the whole-method error budget and remaining capability matrix
unfinished; ordinary diffuse-source bounds are still loose.


### Automatic physical component supports before RI SCF

Source `d70e86fa5c690c40364248d0c15ef8b6778a1a59` qualifies bounded source selection through the
shared cutoff, overlap-envelope, S/T, nuclear and exact-exchange producers.
Twenty-one focused tests exercise multi-step growth, complete RI HF/KS
routes, unchanged explicit-source states, all five resource caps, owned
option snapshots, dry runs, hard limits and retained source refusals.
420 affected shared-producer/RI checks pass. A separate documentation and
citation run exposes only an existing unrelated changelog-format defect.

Local `vq` acceptance runs a converged compact RHF calculation and tight
diffuse RHF/UHF calculations capped at one evaluated iteration. Independent
Gaussian S/T, polynomial/Boys/Fourier nuclear sources, libint finite RI
metric/coupling, NumPy factors, analytic Fourier exchange and SciPy
occupations check their complete finite states. The scalar-Gaussian Fourier
reference is cross-checked with general shifted-contour quadrature. The
compact case additionally passes independent SCF closure and bitwise
automatic/explicit source equivalence. QVF and shared Fock exports agree.

All requested 1e-8 component bounds pass, including diffuse sources whose
initial nuclear attraction and exchange estimates were much larger. The
diffuse source requires nearly one million exact-exchange vectors; no useful
production crossover follows. The unchanged auxiliary 12-bohr boundary case
still refuses its non-conjugate capsule support; acceptance explicitly uses
12.1 bohr rather than altering the shared selector. No matched ORCA/CRYSTAL
custom-domain reference is available. Source/native identity, exact sizes,
measured error and automatic preparation overhead are in the
[support-selection benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_support_selection.md).

This is automatic physical component selection, not a relaxed total-error
RI-BUPO profile. Finite fitting/metric, arithmetic, rank stability,
relaxation, XC and basis/k error ledgers, useful performance, low-dimensional
sources, CCM/chi, source-qualified symmetry, derivatives/stress and remaining
acceleration gates stay required by the full 0.18.x goal.


A compact local `vq` integration calculation also passes after merging
incoming BIPOLE lattice-key exports and weighted Fourier moment reuse and
refreshing the native runtime. Its S/H/F, density, energy and source
certificates match the independently qualified compact state bitwise.
The benchmark records both native identities; the earlier diffuse
scientific evidence is reused without duplicate calculations.


### Nested periodic RI metric-factor truncation bounds

Source `522b81f9651db8e382d328b313d92ecbc4e28e8b` adds optional combined coupling and nested
stored-factor bounds to the existing Gamma J-only accuracy selector.
The selected factor, shared finite-source producer, fixed density family
and all five resource caps retain their existing contracts. A finer
reference cutoff adds positive modes from the same stored eigensystem;
this does not certify an ideal inverse or an unfitted Coulomb operator.

Twenty-two focused cases pass: independent finer-rank RI references for
complex indefinite densities, Decimal90 positive contractions, empty and
zero-rank limits, partial reference spaces, invalid-cutoff and unmet-target
refusals, all five resource caps and complete HF/KS output compatibility.
265 shared producer/RI/support-selection regressions and 468 changelog and
citation checks pass. The earlier unrelated changelog-format issue is
resolved on incoming main.

Local `vq` acceptance runs three converged calculations: full and truncated
RHF, and truncated UHF. Independent Gaussian/libint/Fourier finite sources,
NumPy factors and SciPy SCF check the selected and reference states.
Coupling-reference Fock/energy differences are below 5.0e-15/1.12e-13;
independent densities agree within 4.81e-10. The nonzero metric differences
are covered, though the bounds remain very conservative. QVF and shared
Fock exports agree, and manual selected-source control states agree bitwise.
No matched ORCA/CRYSTAL custom-domain reference is available.

The [metric-bound benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_metric_bounds.md)
records runtime identity, measured errors, bounds, timing and limitations.
Finite fitting-basis/domain, eigensolver, arithmetic, rank-stability and
relaxation terms remain open, as do useful automatic truncation and all
other unqualified rows of the approved MANTA 0.18.x matrix.


### Full-Coulomb periodic RI fitting enclosure

Source `9bfd02610edfacb6974b6f254c1d05acbb5d2199` adds an explicit immutable certificate comparing the
selected stored RI map with full zero-average Coulomb. It reuses the
existing RI and BUPO builders; the BUPO reference must carry both infinite
Ewald-domain and omitted AO-image bounds. A complete real Hermitian probe
basis spans the fixed weighted density family. No exact Hilbert projection
or variational fitting sign is assumed for separately truncated sources.

Sixteen focused checks and 118 shared RI/BUPO regressions pass. Local `vq`
acceptance converges RHF and UHF and compares their final fixed-density J
and Coulomb energy with independent full Gaussian Fourier sums. Reciprocal
cubes 36/40 converge within 1.34e-15/2.67e-15; sampled Fourier transforms
agree with shifted-contour quadrature. Independent finite-source HF
Fock/energy checks pass at 8e-11 and SciPy SCF density error is below
3.23e-10. QVF/shared Fock and bitwise checked/plain J controls pass.
Strict source identity caught a validation-fixture reciprocal-frame
rounding mismatch; using the native reciprocal lattice fixes it without
weakening the guard. No matched ORCA/CRYSTAL custom-domain reference is
available.

Observed J/energy errors up to 0.001335/0.017233 Eh are enclosed by
46.038/23018.970 Eh. These large bounds establish the enclosure, not useful
automatic accuracy. The
[fitting-bound benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_bounds.md)
records exact source/native identities, inputs, timing and limitations.
The certificate includes selected fit, rank, coupling and domain
discrepancies together; do not add those earlier component bounds again.
Floating arithmetic, rank stability and density relaxation are excluded.
Automatic complete-job attachment, useful auxiliary/domain selection and
the remaining full MANTA 0.18.x capability matrix remain unfinished.


### Shared BUPO electronic reference preparation

Source `0a4dde646b159d6eace78da3a1b06a551cd67209` extracts the original cumulatively admitted electronic
source construction from the HF preparation consumer. RI fitting receives
the same immutable full-domain/full-image source directly, together with
the budget remaining after its preparation. Native mesh/source identities
are preserved and no exchange, nuclear or SCF policy enters this factory.

Fourteen new and 32 existing focused cases pass. Local `vq` acceptance
converges RI RHF/UHF and BUPO RHF. Independent Gaussian/Fourier and SciPy
SCF references pass, with unchanged RI fitting errors/bounds. BUPO Fock,
energy and density differences are below 8.89e-15, 2.74e-14 and 9.12e-11.
The established compensated ordered-Fourier exchange reference resolves
an initial unbuffered-contraction mismatch without changing tolerances.
No matched ORCA/CRYSTAL custom-domain reference is available.

The [preparation benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bupo_preparation.md#shared-electronic-reference-for-ri-fitting)
records source/runtime identity, resources and measured costs. No new
accuracy proof is introduced. Full-job RI fitting admission still needs
shared Gamma ownership, pre-SCF canonical density bounds, fixed targets
and joint count-only admission. Useful automatic selection and all
unqualified original matrix rows remain required.


### Complete periodic RI fitting admission

Source `71c9c45fce409d03dc42412012f7128f0139f25c` attaches the shared full-Coulomb fitting certificate
to the actual Gamma RI producer before RHF/UHF and pure/global/range KS
SCF. Count-only joint plans include the shared reference, all probes,
canonical density analysis and later fixed-family validation. Fitted K,
XC, physical Fock identities and symmetry authority are unchanged.

Twenty-one new checks pass, and the shared regression's exchange-only
range-KS citation guard is fixed. Local `vq` acceptance converges all six
methods with bitwise plain/certified states and source identities. HF
independent full Fourier errors are enclosed, and finite-source Fock,
energy and density comparisons pass. KS uses unchanged-source controls
and reuses existing independent regression evidence. No matched local
ORCA/CRYSTAL custom-domain reference is available.

The [complete fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md)
records runtime identity, measured errors, bounds and overhead. The J
bounds near 18.883 and energy bounds near 1053.4 Eh are still conservative.
Targets are deliberately loose; this proves pre-SCF enforcement for an
explicit shared reference, not useful automatic accuracy selection or a
relaxed-state energy guarantee. Arithmetic/rank/relaxation and the complete
unqualified MANTA 0.18.x capability matrix remain required.


### Kernel-capped constant-pair image envelopes

Source `8fda8ceb874499b6502b90874bf41df48ba280a6` tightens the shared BUPO/RI reference image
mass producer while preserving its global kernel constant. Plain s-s
primitive exponents use t=min(1,A/(a+b)); other derivations remain intact.
Eight new and 77 affected checks pass, including independent reconstruction
of the old mixed-angular enclosure and tighter complete RI targets. Ten
existing gradient/Fourier checks pass. Local `vq` converges six RI methods
and BUPO HF; independent Fourier/finite-source/SCF checks and previous-state
bitwise comparisons pass. No matched local ORCA/CRYSTAL reference exists.
The [fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#kernel-capped-s-pair-reference-envelopes)
records bounds, runtime and costs. The unchanged finite fitting discrepancy
now dominates this reference remainder; useful automatic source/auxiliary
selection and all other unqualified MANTA capabilities remain required.


### Full-Coulomb metric cutoff selection

Source `b5a0594eac4cc2320d1a712312f9a1bdb2fa90e0` passes 160 focused checks, including
22 new cases and documentation checks. Local `vq` accepts six converged
RI HF/KS profiles plus a gradual five-trial RHF search. Every candidate
uses the same full reference; the selected source is retained for SCF.
Independent HF numerical references, explicit selected controls and
unchanged-state comparisons pass. The initial KS control-identity failure
was resolved by matching cutoff settings, with passing HF evidence reused.
No matched local ORCA/CRYSTAL custom-domain reference is available.

The [fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#automatic-metric-cutoff-selection)
records the rejected and accepted ranks, bounds, costs and runtime.
This implements bounded metric selection, with explicit auxiliary and
reference domains. Their automatic selection and all remaining unqualified
MANTA capabilities stay required.


### Automatic periodic RI reference selection

Source `c8a2bb129e07bbd91fe894c1d669bb3e25563849` passes 455 focused checks. Local `vq`
accepts default shared-reference refinement followed by metric refinement
for RHF, UHF, RKS, UKS, PBE0 and HSE06. Complete states, source identities
and full certificates match explicit selected-reference controls. Prior
independent HF numerical evidence is reused after exact source/state/hash
checks; new bounds enclose the same full-Coulomb values. No matched local
ORCA/CRYSTAL custom-domain reference exists. Fixed-cutoff fitted K and
separate component policies retain their contracts; symmetry grants and
BIPOLE/CRYSTAL conventions are unchanged.

The [reference-search benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#automatic-shared-reference-selection)
records measured cost, reference trials and native identity. Count plans
reserve a declared cap; actual stages charge cumulative admitted costs,
including rejected work. Dry runs claim no accuracy. Automatic physical
fitting-domain and auxiliary selection and every unqualified original
MANTA capability remain required.


Final accounting source `c01c7ace969fc193c17ee84d7c70c179e863e113` adds four passing
setup/margin checks and two accepted local `vq` RHF calculations. Default
and larger search margins preserve states/certificates. Complete preparation
controls are recorded. Together with reused evidence above, 459 focused
checks pass; no additional numerical accuracy scope is claimed.


### Automatic physical fitting-domain selection

The shared GDF producer now enlarges the actual RI J pair, auxiliary-image
and reciprocal supports under pre-SCF full-Coulomb admission. Complete
plans reserve the largest allowed candidate for every SCF call; rejected
domain/reference/metric work remains charged. This does not change the
bases, kernel convention, complementary physical sources, unified Fock
ownership or symmetry permissions. See scientific contract Section 5.25.

The source-qualified local acceptance used six converged RHF/UHF/RKS/UKS/
PBE0/HSE06 calculations, each with three physical domains and two metric
trials per domain. Every preceding candidate failed at least one target;
the selected pair/auxiliary radii were 6.75/9 bohr, with reciprocal cube two
and 124 vectors. All selected-source controls preserve energies, densities,
Fock/core/overlap matrices and certificate identities bitwise. The four KS
profiles also pass on the runtime containing the incoming shared periodic
XC projection reuse. Native sources and complete state exports retain
their existing qualifications.

For the two HF fixtures, the full-rank J bound decreases from 0.525835 to
0.106848 and then 0.0153705 Eh; its same-density Coulomb energy bound drops
from 145.965 to 26.1934 and then 1.10547 Eh. The configured 0.02/2 targets
pass only at the third physical domain. The deliberately coarse starting
reciprocal support belongs to a synthetic acceptance fixture; these are
not recommended production targets or converged total energies.

| State | Observed full-Coulomb J error / Eh | J bound / Eh | Observed Coulomb energy error / Eh | Energy bound / Eh |
|---|---:|---:|---:|---:|
| RHF | 0.00132026 | 0.0153705 | 0.0189208 | 1.10547 |
| UHF | 0.000909658 | 0.0153705 | 0.00816172 | 1.10547 |

Independent finite Gaussian/libint M/T arrays and NumPy factors agree with
the accepted source. Independent complete HF Fock errors are below
7.0e-15 Eh; SciPy SCF densities agree within 1.53e-9. Independently summed
full reciprocal cubes 36/40 agree within 7e-16 Eh in J, after checking
Gaussian Fourier panels against shifted-contour quadrature. No matched
local ORCA/CRYSTAL custom-domain reference was available. KS uses bitwise
selected-source controls and qualified shared-producer/functional tests;
the external-reference limitation remains explicit.

The 33 new domain tests, 77 existing fitting-admission tests, 667 affected
shared/documentation checks and 50 incoming XC checks pass. Coverage includes
all five complete and staged resource caps, rejected work, nested metric
resets, first-pass retention, limits, initial-support nesting, shared
producer union/integral/action checks and component-contract refusals.
One test initially inspected the wrong nested plan and was corrected;
scientific tolerances were unchanged.

The two fresh-process benchmark pairs measure roughly 0.763/0.402 seconds
for automatic/fixed RHF and 0.718/0.411 seconds for UHF, with peak RSS about
275 MB. This is a bounded selection cost, not a speedup or crossover claim.
The [benchmark record](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#automatic-physical-fitting-domain-selection)
contains the exact procedure and source/runtime pins. Auxiliary-basis
selection, remaining numerical/relaxation ledgers and every other
unqualified MANTA 0.18.x capability remain unfinished.


## Shared product auxiliary candidates

The shared generator and prepared periodic job adapter in scientific
contract Section 5.26 are qualified for the admitted sources. Local
source-pinned calculations cover RHF/UHF/RKS/UKS/PBE0/HSE06 with level-zero
candidates and two finer level-one HF cases. All eight converge and retain
their full-Coulomb certificates. Plain selected-basis controls preserve
energy, density, Fock/core/overlap matrices, physical source identity and
certificate identity bitwise. Standard output includes generation receipts
and the Stoychev2017 citation; QVF validates. The four affected KS profiles
also pass on the merged XC scheduling runtime with unchanged energies and
certificate identities.

The compact HF seed has three functions. Level zero produces eight
functions and retains rank eight; level one produces twelve and retains
rank ten at the original 1e-12 metric cutoff. The KS seed contains one exact
generated duplicate, giving seven functions at level zero. All profiles
select the same 6.75/9 bohr pair/auxiliary radii and reciprocal cube two.
The preceding seed-basis HF certificate had J/energy bounds
0.0153705/1.10547 Eh. Level-zero enrichment gives 0.00205124/0.481827 Eh;
level one gives 0.00205079/0.481811 Eh. These conservative synthetic-fixture
targets are not production accuracy recommendations or converged total
energies. Increasing basis size does not imply monotonic observed error.

| State and level | Observed full-Coulomb J error / Eh | Observed Coulomb energy error / Eh |
|---|---:|---:|
| RHF, zero | 8.57946e-5 | 8.56740e-4 |
| UHF, zero | 6.29797e-5 | 4.62476e-4 |
| RHF, one | 8.57972e-5 | 8.56730e-4 |
| UHF, one | 6.29831e-5 | 4.62472e-4 |

Independent finite Gaussian/libint metric/coupling arrays, NumPy factors,
full HF Focks and SciPy SCF agree. Maximum Fock error is 1.12e-14 Eh and
density error is 2.85e-10. Full reciprocal cubes 36/40 agree within
1.12e-15 Eh in J and 2.67e-15 Eh in energy, after checking Fourier panels
against shifted-contour quadrature. All observed errors lie inside the
reported fixed-density bounds. No matched local ORCA/CRYSTAL custom-domain
reference was available. KS uses bitwise selected-source controls and
qualified shared-producer/functional tests; that limitation remains.

The 827 focused generator, periodic job, shared-method and repository
checks pass, plus 86 incoming XC checks. Coverage includes exact seed
preservation, nested grids, independent angular-product spaces through
orbital f, normalized shells, the metadata angular ceiling, overflow,
source matching, four generation caps and complete job memory/work charges.
Test fixture parity, citation routing/output paths, changelog format and
test classification were corrected; no numerical tolerance was relaxed.

Two fresh benchmark processes include generation separately and compare
complete prepared/plain-basis jobs. Level-zero RHF costs about 1.57/1.53 s;
level one about 2.00/2.03 s. Generator time is below 20 microseconds on
these tiny fixtures; periodic source construction and solving dominate.
Peak process RSS is about 275 MB. The
[benchmark record](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#shared-product-auxiliary-candidates)
provides exact reproduction and runtime pins. No speedup claim follows.
Automatic selection among auxiliary candidates, remaining numerical and
relaxation ledgers, and all other unqualified 0.18.x capabilities remain
unfinished. Unified Fock, BIPOLE/CRYSTAL and symmetry authority are preserved.


## Automatic periodic RI auxiliary selection

Scientific contract Section 5.27 is qualified by eight complete local queue
calculations: RHF/UHF/RKS/UKS/PBE0/HSE06 at generated level zero, plus two
HF cases selecting level one. Every calculation starts from the seed and
resets the original metric at each auxiliary/domain trial. The deliberately
rank-zero initial cutoff is refined inside each trial. Both targets must
pass on the same canonical density family and shared reference. The first
six states select after twelve trials; finer HF cases use eighteen. Fixed
selected-source controls reproduce energy, density, Fock/core/overlap,
physical source and certificate identities bitwise. QVF validates and
standard output records the actual selected basis and generator citation.

The compact HF seed has three functions. Targets 0.005 Eh in J and 0.7 Eh
in same-density Coulomb energy select eight functions, rank eight. Tightening
the energy target to 0.48182 Eh selects twelve functions, rank ten, at
cutoff 1e-12. Their bounds remain 0.00205124/0.481827 and
0.00205079/0.481811 Eh respectively. These conservative synthetic targets
exercise the controller; they are not production tolerance recommendations.
No monotonic convergence assumption or metric repair is introduced.

Independent finite Gaussian/libint M/T and NumPy RI factors agree directly.
The four HF states, fitted J arrays, selected vector arrays and entry bounds
match the preceding accepted generator calculations bitwise, with the same
RI/reference source identities. That match permits reuse of their independent
SciPy SCF, full-Fock and Fourier cube 36/40 witnesses. Observed full-Coulomb
errors remain 6.30e-5 to 8.58e-5 Eh in J and 4.62e-4 to 8.57e-4 Eh in
Coulomb energy, within the reported bounds. No matched local ORCA/CRYSTAL
custom-domain reference was available. KS uses bitwise selected-source
controls and the qualified shared-functional tests; this limitation remains.

The 913 focused/shared/repository checks pass, including 31 new selection
checks and 90 incoming torus/adjoint checks. Coverage includes fixed fitting
supports without domain search, retained seed or generated owner, finer
candidates, metric resets, immutable captured controls, source mismatch,
shared integral parity, sequence lifetimes, all five cumulative search and
complete capacity caps, limit refusals and component-contract refusals.
No scientific tolerance was relaxed. Incoming release bookkeeping preserves
the numerical source; runtime/source pins and job outputs are retained in
the private acceptance record.

The updated benchmark compares automatic selection with the identical fixed
selected source and reference in two fresh processes, with preparation
reported separately. It measures controller overhead, not a speedup.
[Reproduction and numerical results](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#automatic-auxiliary-selection)
include source/runtime pins. Unified Fock composition, BIPOLE/CRYSTAL
conventions and symmetry authority are preserved. Arithmetic, rank stability,
relaxation, useful whole-method accuracy and the remaining 0.18.x matrix
are still unfinished.


## Reference energy enclosures and tighter periodic RI profiles

Scientific contract Section 5.28 is qualified by eight complete local queue
calculations. RHF/UHF/RKS/UKS/PBE0/HSE06 use J/energy targets 0.001/0.05 Eh
and permit no reference refinement. All six retain the initial shared BUPO
reference because its entrywise energy enclosure passes the fraction test;
the former L1*maximum-J estimate would refuse these same controls. The
actual fitted source still has to pass both combined targets. This prevents
unnecessary reference growth without weakening scientific acceptance.

Two tighter HF profiles use J/energy targets 1e-6/1e-4 Eh, automatically
refining the shared reference and selecting the generated auxiliary basis
and physical fitting domains. The search retains the same fixed canonical
density family throughout. Pair/auxiliary fitting radii stop at 6.75/9 bohr;
the reciprocal cube can grow to four. These controls qualify compact
synthetic fixtures, not a general production default or whole-method bound.

| State | Profile | Observed full-Coulomb J error / Eh | J bound / Eh | Observed Coulomb energy error / Eh | Energy bound / Eh |
|---|---|---:|---:|---:|---:|
| RHF | reference-energy | 5.27954e-07 | 0.000209896 | 4.1168e-06 | 0.0148146 |
| UHF | reference-energy | 3.61145e-07 | 0.000209896 | 1.94592e-06 | 0.0148146 |
| RHF | tight | 1.1979e-09 | 5.08334e-07 | 1.98388e-08 | 2.3501e-05 |
| UHF | tight | 2.57568e-09 | 5.08334e-07 | 7.46476e-09 | 2.3501e-05 |

Every selected-source control reproduces energy, density, Fock/core/overlap,
physical source and certificate identities bitwise; QVF validates. Direct
independent finite Gaussian/libint M/T and NumPy factors agree. Complete HF
Focks differ by at most 3.09e-14 Eh and independent SciPy SCF
densities by at most 2.85e-10. Full reciprocal cubes 36/40 agree within
4.72e-16 Eh in J and 4.44e-15 Eh in energy, after a shifted-contour
check of Gaussian Fourier panels. Observed finite-fitting and reference
energy discrepancies separately lie inside the new component bounds.
No matched local ORCA/CRYSTAL custom-domain reference was available. KS uses
bitwise selected-source controls and qualified shared-functional regressions;
the external-reference limitation remains explicit.

The 364 focused checks pass: 159 fitting/reference/domain/auxiliary
regressions, eight complete new profiles, 93 repository/documentation checks
and 104 incoming XC torus/adjoint checks. Coverage includes complex indefinite
densities, zero entries/weights, quadratic energy scaling, resource caps,
first-pass retention, unchanged fixed-state behavior and shared-source limits.
No scientific tolerance was relaxed. Incoming XC residue-index reuse is
included in the pinned calculation runtime.

[Benchmark reproduction and timings](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#reference-energy-enclosures)
record controller costs. Expensive tight-calculation timings are reused from
the scientific acceptance runs; no duplicate calculation is needed merely
for timing. Unified Fock, BIPOLE/CRYSTAL and symmetry authority are preserved.
Fixed-density Coulomb fitting acceptance does not cover integral arithmetic,
rank stability, density relaxation or the remaining MANTA 0.18.x matrix.


### Bounded periodic reference batches

Six reference-energy profiles (RHF, UHF, RKS, UKS, PBE0 and HSE06) and two
tighter HF profiles pass complete local calculations with both batch-eight
and scalar reference evaluation. All 16 states converge and each pair has
bitwise equal energy, density, Fock, core/overlap, source and certificate
identities. Tight J/energy targets remain 1e-6/1e-4 Eh; no tolerance changes.
The HF states, fitted J arrays, reciprocal inventory and entrywise bounds
also match the previous independently validated reference-energy evidence
bitwise. Its Gaussian/libint, NumPy/SciPy and Fourier36/40 comparisons are
therefore reused. No matched local ORCA/CRYSTAL custom-domain reference was
available; KS retains the complete scalar controls and shared-functional
qualification limitation.

The 31 new checks cover C/F-contiguous batches, complex indefinite angular
inputs, zero weights and reciprocal inventory, exact scalar physical/bound
receipts, explicit approximate-partition fallback, invalid inputs, all five
resource caps and off-diagonal probe pairs crossing batch boundaries.
Actual shared kernel/domain counts fall while conservative cold reservations
remain charged. Full-source identities and symmetry permissions are intact.
See scientific contract Section 5.29 and the
[paired benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#shared-reference-batches).
Source `1e78b4651f0bddc44a42c01640506f2524a22ee6` qualifies this bounded increment; arithmetic, rank
stability, relaxation and remaining MANTA matrix rows are still unfinished.

Integrated shared XC spin-field reuse is also checked on source `5326cccb9ac7e6e221c4fec2a645722b6f304f90`,
native SHA256 `86847691bfa78359f5a0de752c9b31c37af6a6cfb626fb01d25d18363d505a5e`. Four affected MANTA KS profiles each pass scalar
and batch-eight complete calculations, preserving the prior energies and
certificate identities. The 137 focused torus checks pass, with four existing
expected empty-spin GGA/MGGA potential-parity failures. These additional
calculations qualify integration; the timing table retains the original
paired scientific measurements. An incoming native periodic RKS READ-state fixture still fails the AO-image
density-domain closure guard; that separate driver is not qualified by
these MANTA runs.


### Reference component bounds and directed refinement

Source `603c44ebc749ed859447d239975adc89a947dfe2` qualifies separate finite BUPO, real/reciprocal-domain and
pair-image J/energy enclosures over the same fixed density family. Independent
Gaussian domain increments cover nonzero real and reciprocal errors for
scalar and angular bases. A finite approximate BUPO action is compared with
direct Ewald; batch/scalar bounds, zero-weight families, quadratic energy
scaling, explicit limits and fixed-source controls pass. Existing metric,
domain and auxiliary resource admission remains enforced.

Thirty complete local calculations pass: six soft HF/KS and two tight HF
profiles each use selective, joint and selected-source controls, plus three
directed-domain profiles and their fixed controls. All are converged. Paired
physical states are bitwise equal; fixed controls also reproduce certificate
identities. Tight combined J/energy targets stay 1e-6/1e-4 Eh. Changed
reference domains have independently assessed certificates, not assumed
identity with the former joint path.

Independent Gaussian/libint, NumPy/SciPy and Fourier36/40 HF evidence is
reused only after matching physical state, fitted J and reciprocal inventory.
New bounds enclose the saved discrepancies. No matched local ORCA/CRYSTAL
custom-domain reference is available. Complete KS selected-source controls
retain the shared-functional qualification limitation. Approximation-directed
complete searches are not claimed by these direct-partition profiles.

See scientific contract Section 5.30 and the
[component benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_fitting_accuracy.md#component-directed-reference-refinement).
Arithmetic, rank stability, relaxation and all remaining MANTA capability
rows stay unfinished. Unified Fock, BIPOLE/CRYSTAL and symmetry authority
remain unchanged; this is not a relaxed total-energy guarantee.


### Stored metric spectrum and rank enclosure

Source `a070f8860811789fd07fcc48f814ff2ee585fec2` adds a shared outward-rounded residual and
orthogonality audit of actual periodic RI metrics. Independent Decimal
2-by-2 eigenvalues across 300 orders of magnitude, complex clustered metrics,
Hermitian perturbations, cutoff ambiguity, deficient eigenvectors and
resource-admission checks exercise the bound and its refusal conditions.

Sixteen complete local calculations pass: RHF, UHF, pure RKS/UKS, global and
range-separated KS, plus tight RHF/UHF, each with a fixed selected-source
control. The 8 exported metric certificates all qualify their spectrum
and rank. Physical states, sources and full-Coulomb certificates match the
controls bitwise. Four HF states/J arrays match prior independent Gaussian,
libint, NumPy/SciPy and Fourier36/40 references exactly. No matched local
ORCA/CRYSTAL custom-domain reference is available; KS retains the fixed-source
and shared-functional comparison limitation.

See the [spectral benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md)
for measured bounds, timings and scope. The diagnostic leaves factors,
unified Fock, BIPOLE/CRYSTAL conventions and symmetry authority unchanged.
It does not certify integral/factor/Fock arithmetic or relaxed SCF energy.
Automatic rank-aware selection and all other unfinished capability rows remain.


### Optional fitting rank-margin admission

Source `fa746e2f2791aaa42f4a546c21fdea3ad1dafbc3` qualifies strict stored-Gamma rank margins in
fixed and bounded metric/domain/auxiliary fitting admission. Focused checks
cover empty/partial/full retained spaces, exact threshold boundaries,
invalid controls, None versus zero, independent reference readiness and
pre-SCF exhaustion. The Section 5.31 numerical proof and its accepted Decimal
and perturbation witnesses are reused without changing the eigensolver audit.

Twenty-six complete local calculations pass. Five searches advance solely
because the initial rank request fails despite passing J/energy bounds;
metric, domain and auxiliary transitions retain the actual first passing
owner. Their fixed controls match energy/state/source/full-Coulomb certificate
bitwise. Eight earlier soft/tight HF/KS profiles also pass with a 1e-10 rank
request and selected controls, retaining prior energies. Four HF states/J
arrays match accepted independent Gaussian/libint, NumPy/SciPy and Fourier
references exactly. Four unattainable searches refuse before SCF. No matched
ORCA/CRYSTAL custom-domain reference is available; KS retains its fixed-source
and shared-functional comparison limitation.

See the [rank-admission benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#rank-driven-fitting-admission).
Unified Fock, BIPOLE/CRYSTAL conventions and symmetry authority are unchanged.
This is an enforced declared norm margin for Gamma J, not an automatically
estimated integral error, all-q or relaxed total-energy guarantee. The
remaining original MANTA capability rows stay unfinished.


### Stored factor inverse enclosure

Source `f7e3c00b47adf1cb34bc88264b9f8d70e796ad03` qualifies the Section 5.33
stored-factor inverse diagnostic through sixteen complete local calculations
and independent Decimal inverse witnesses. Shared Fock/rank/fitting resource
checks pass; full/truncated/empty spaces, complex clusters, arbitrary factor
perturbations, overflow and pre-payload admission are covered.

All eight selected metric certificates qualify. Selected controls remain
bitwise identical in energy, density, Fock matrix and physical identities.
Four HF profiles reuse accepted independent Gaussian/libint, NumPy/SciPy and
Fourier references after exact state/J/inventory comparison. No matched
ORCA/CRYSTAL custom-domain reference is available; KS retains its fixed-source
comparison limitation. The bound measures an absolute inverse-operator
error, and does not change the separate full-Coulomb roundoff exclusion.
See the [inverse benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#stored-factor-inverse-error-bounds).
Integral and Fock-action arithmetic, relaxed energy and the remaining MANTA
capability matrix still require implementation and acceptance.


### Coupling-projected metric action admission

Source `79564c6b1be9ad6ccfd67b40de5acc81b8395ce9` qualifies optional numerical metric
J/energy targets on explicit coupling tiles. The finite residual proof in
Section 5.34 is checked against independent 100-digit Decimal calculations,
full/truncated/empty spaces, complex columns, factor perturbations, density
scaling, all resource caps and fixed-source state controls. The selected
suite passes 755 checks; no full CI is required for this development landing.

Twenty-eight complete local calculations and four pre-SCF refusals pass.
Eight earlier soft/tight profiles meet matching fitting and numerical metric
targets. Four HF states/J/inventories match the accepted independent Gaussian,
libint, NumPy/SciPy and Fourier references exactly. Five action-driven searches
and one nonzero ordered-multipole case retain the actual passing owner and
match controls bitwise. No matched ORCA/CRYSTAL custom-domain reference is
available; KS retains its fixed-source comparison limitation. See the
[metric-action benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#coupling-projected-metric-action-admission).

Compressed/global moment and pair-tree action arithmetic remains unqualified
for this additional policy. Integral and actual Fock-contraction roundoff,
relaxed energy and the original remaining capability matrix remain open.
Unified Fock, physical BIPOLE/CRYSTAL conventions and symmetry permissions
are preserved; the full-Coulomb receipt still excludes roundoff.


### Joint density-family metric admission

Source `093642f86a2f98637adc04b5a30c133614222e98` uses a verified dual support bound
for total and per-entry density limits. Independent 100-digit primal
allocation checks cover ties, nearly equal capacities, zero limits, large
and small scales, permutations and public resource boundaries. The focused
metric and periodic admission checks are reused after a source-preserving
rebase. Six complete local RHF/UHF/RKS calculations with fixed-source
controls pass on the integrated runtime. HF reuses the accepted independent
reference arrays; no matched ORCA/CRYSTAL custom-domain reference is available,
and RKS remains a fixed-source comparison. See the
[joint-density benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#joint-total-and-component-density-limits).
This qualifies tighter bounds within the existing explicit-coupling scope;
it does not qualify compressed actions or the remaining full MANTA matrix.


### Compressed metric-action admission

Source `d0e485994ebb37e2eb382a544f696c07f9fedc35` adds producer-owned outward
column norms for ordered compressed hierarchy and global-panel sources.
331 successful focused/shared checks cover independent 100-digit Decimal
metric action, complex and signed-angular materialized-column witnesses,
fixed-source controls, all five resource caps and existing Fock interfaces.
Eighteen complete local queued calculations and six strict-target refusals
pass. Source/native pins are preserved; previous acceptance is reused only
for unchanged components. No matched ORCA/CRYSTAL custom-domain reference is
available. See the
[compressed metric benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#compressed-metric-action-admission).
The accepted complete profiles use unreduced symmetry and loose transition
targets. This does not authorize new symmetry reductions or qualify pair-tree
bounds, integral/contraction roundoff, relaxed SCF energy or production
whole-method accuracy. Full MANTA remains unfinished.


### Composed stored-input paired arithmetic

Source `0c8d17c4` composes the accepted forward, fixed-factor metric and
adjoint enclosures into paired J and energy bounds at the actual stored
weighted input. The fitting owner reuses its source-qualified column norm;
no new numerical route, source sweep or dense coupling tensor is introduced.
243 focused checks and six complete local queued RHF/UHF/UKS calculations
with three strict refusals pass. See the
[paired arithmetic benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#composed-paired-j-and-energy-arithmetic)
and scientific contract Section 5.38 for scope, references and limitations.
Weighted-input and stored-source construction, final J/K/Fock assembly,
relaxed-SCF accuracy and the remaining capability matrix are unfinished.
Full-Coulomb roundoff remains excluded; this does not change the pre-SCF
inverse target, shared Fock/eigensolver, BIPOLE/CRYSTAL or symmetry authority.


### Paired arithmetic from stored spin densities

Source `e30df649` extends paired J/energy bounds to exact spin summation and
multiplication by the stored k weights. Per-entry input errors feed the
existing coupling interval graph, including Gamma completion, then the shared
fixed-factor metric composition. Explicit scope flags distinguish the extended
comparison. Exact rational and same-source controls qualify 390 distinct
checks after correcting first-use cache accounting in a new fixture. Six
complete local queued RHF/UHF/UKS calculations and three strict refusals pass.
See the [weighted-input benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#spin-summed-and-k-weighted-input-arithmetic)
and contract Section 5.39. Physical outputs stay bitwise, with no new source
sweep or dense coupling tensor. Final J/K/Fock assembly, density/source
construction and relaxed-SCF accuracy remain excluded; full-Coulomb roundoff
and the remaining capability matrix are still unfinished.


### Evaluated HF/KS assembly arithmetic

Source `d5364817` adds a shared stored-component Fock/energy audit inside the
existing native HF/KS adapters. It covers weighted traces, final callback
reconstruction and Hermitian projection, including iteration-capped states,
and composes accepted paired Coulomb arithmetic once. Exact rational and
complete-method controls pass within 803 focused/shared checks. Eighteen
local queued selected/control calculations and three strict refusals pass;
physical states, energy and source identities remain bitwise unchanged.
See the [assembly benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#evaluated-fock-and-energy-assembly-arithmetic)
and scientific contract Section 5.40. Source/K/XC/correction construction,
density generation, non-paired execution and relaxed-SCF accuracy remain
excluded. The full capability matrix and whole-method arithmetic budget
remain unfinished; shared Fock/eigensolver, PBC and symmetry authority is
unchanged.


### Direct periodic exchange arithmetic

Source `c7f61c14` adds an optional interval audit to the existing reciprocal
B^dagger D B exchange producer at stored Bloch panels and reciprocal factors.
Per-entry errors propagate through actual method exchange fractions and density
traces once, alongside the accepted paired Coulomb and evaluated assembly
bounds. Fifty exact-rational raw cases and signed method composition checks
pass within 894 focused/shared scientific checks. Twenty-six local queued
selected/control calculations and three strict refusals pass with bitwise
physical states and source identities. See the
[direct exchange benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#direct-periodic-exchange-contraction-arithmetic)
and contract Section 5.41. Fourier/kernel-factor construction, fitted/retained/
fused K arithmetic, XC/correction construction and relaxed-SCF accuracy remain
excluded. Shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL and symmetry authority is
unchanged; full-Coulomb roundoff and the remaining capability matrix stay open.


### Stored exchange zero-mode correction arithmetic

Source `19a8cd49` adds the shared interval SDS/scalar audit to the actual BvK
correction in admitted paired RI HF/KS. Twenty-six local queued selected/control
calculations and three strict refusals pass with bitwise physical states and
source identities. The 862 focused/shared scientific checks include independent
exact-rational four-index expansion, signed/zero scales, non-Hermitian raw
inputs, cancellation, subnormal products, perturbed witnesses and correction
error propagation into energy/Fock exactly once. Resource boundaries, malformed
input, strict-zero-mode exclusions and converged/capped methods pass. See the
[correction benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#stored-exchange-zero-mode-correction-arithmetic)
and contract Section 5.42. The reference holds S, D, Madelung and exchange scale
fixed. Their physical construction, retained/fused RI/XC, rank and relaxed-SCF
errors remain outside it. This does not close #624 or the full matrix, change
shared Fock/eigensolver or PBC/BIPOLE/symmetry policy, or expand #557.


### Retained-mode periodic RI arithmetic

Source `63a80bf1` audits retained F^dagger T, weighted J/energy and fused
q-channel K at stored factor/tile inputs without changing physical sweeps.
Complete HF/KS consume the retained J/energy bounds through the shared assembly
ledger, separately from paired-route bounds. Local queued acceptance passes
26 selected/control calculations, four raw q-channel calculations and three
strict refusals; 1094 focused/shared scientific checks pass, with 52 affected
checks repeated after resource accounting corrections. Two prior automatic
reference budget refusals remain tracked in #643 and are not claimed as passed.
See the [retained arithmetic benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#retained-mode-periodic-ri-arithmetic)
and contract Section 5.43. Complete fitted-K cross-q aggregation, input/factor
construction, rank, XC and relaxed-SCF accuracy remain outside this increment.
Shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL, symmetry and #557 scope are unchanged;
#624 and the full matrix remain unfinished.


### Whole-mesh fitted exchange arithmetic

Source `9e3a8c9a` propagates existing q-channel K radii through ordered-q
accumulation and signed HF/global/range KS assembly. Local queued validation
passes 32 selected/control calculations and six exact-rational whole-mesh
cases. There are 1428 distinct scientific and 27 documentation/citation checks;
independent energy/Fock differences are below 1.12e-13 Eh/4.85e-15 and physical
controls are bitwise identical. See contract Section 5.44 and the
[fitted exchange benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#whole-mesh-fitted-exchange-arithmetic).
Input/factor construction, rank, XC and relaxed-SCF errors remain separate;
#643 budget refusals remain unresolved. #624 and the full #617 matrix stay
open. Shared producer/Fock/eigensolver, PBC/BIPOLE/CRYSTAL, symmetry admission
and bounded #557 scope are unchanged.


### Full-Coulomb fitted exchange component certificate

Source `7107d2e7` qualifies a fixed-family full-Coulomb fitted-K component
bound using shared RI/reciprocal producers and existing image/domain tails.
Two complete local queued RHF/UHF calculations, four raw family cases and
four strict refusals pass; 539 final-native scientific and 28 documentation/
citation checks pass. See contract Section 5.45 and the
[exchange fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#full-coulomb-fitted-exchange-component-certificate).
A supplementary case at `7e5f6760` verifies an actually truncated metric;
the complete HF evidence is reused with unchanged product source and native.
The bounds remain very conservative. Automatic method attachment and useful
reference/auxiliary selection, physical range/XC bounds and source/rank/SCF
errors remain #623/#625/#626/#627. This does not close the full #617 matrix,
change shared state or symmetry admission, resolve #643, or enlarge #557.


### Exchange-certificate preconstruction admission

Source `c89aef19` qualifies count-only certificate capacity through the existing
pair, RI and exchange-tail plans (contract Section 5.46). All 45 affected checks
and two complete local queued RHF/UHF cases pass. The earlier numerical
certificates, energies and source identities are bitwise preserved; see the
[capacity benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#preconstruction-exchange-certificate-capacity).
Reference preparation and retained J storage are charged before construction;
RI setup remains the method caller's responsibility. Automatic method attachment
and useful accuracy remain open under #626/#627; the full matrix is unfinished.


### Pre-SCF RI HF fitted exchange admission

Source `2dfbd54d` attaches the shared full-Coulomb K component certificate to
periodic RI RHF/UHF before SCF, with canonical density-family checks and joint
J/K reservations. Eight converged and four partial-rank stopped selected/control
calculations plus two strict refusals pass through local `vq`; 184 scientific
checks pass, with six accounting checks repeated on the final runtime.
See contract Section 5.47 and the
[admission benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#pre-scf-fitted-exchange-admission-for-ri-hf).
Raw component targets remain very loose; executed-response arithmetic and
source/BvK/rank/SCF errors stay separate. KS attachment, physical range/XC and
useful selection remain #626/#627. This does not close the full #617 matrix or
expand bounded #557. Shared Fock/eigensolver, PBC/BIPOLE/CRYSTAL and symmetry
contracts are preserved.


### Pre-SCF signed range exchange admission

Source `1c06da0c` attaches the signed physical exchange-component bound before
RI range-separated RKS/UKS SCF through shared component owners and the canonical
spin family. It retains absolute-coefficient outward propagation, finite erfc
zero accounting, joint J/K reservations and separate response arithmetic.
Twenty-two converged and eight stopped selected/control calculations and two
strict refusals pass through local `vq`; 468 affected scientific checks pass.
See contract Section 5.50 and the
[range admission benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#pre-scf-signed-physical-range-exchange-admission).
Existing physical states and Coulomb certificates remain bitwise unchanged.
Bounds are conservative; useful accuracy, XC, construction/rank/relaxation and
all remaining matrix requirements remain open. The shared Fock/eigensolver,
PBC/BIPOLE/CRYSTAL and symmetry contracts and bounded #557 scope are preserved.


### Grouped Fourier exchange reference bounds

Source `da33a7dd` adds optional positive Fourier-decay groups to the
existing exchange reference producer and RI accuracy controls. The same
primitive envelopes are partitioned by exponent, with all cross-group terms
and their separate decays bounded. Physical states and the default one-bin
certificates are unchanged. Twenty-three converged and eight stopped
calculations, three refusals and four independent complex-density remainder
cases pass through local `vq`; 485 scientific checks pass. See contract5.51
and the [decay-group benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#positive-fourier-decay-groups-in-exchange-reference-bounds).
HSE06 component bounds improve from about55.69/17999.49 to39.36/4392.80,
while remaining conservative. Useful whole-method accuracy, XC, source/rank/
relaxation and the full capability matrix remain unfinished. Shared method,
solver, PBC/BIPOLE/CRYSTAL, symmetry, #643 and bounded #557 scope are preserved.


The stored-operand metric assembly rank increment (contract 5.55) is accepted
at source `e1e80ed6`, qualified tree `7b42701d`: eleven converged public
calculations, nineteen qualified channel receipts, exact state/control reuse,
Decimal assembly witnesses and strict cutoff/resource refusals. The metric
spectrum benchmark records physical supports, measured bounds, independent
reference boundary, actual cost and native identity. Construction of the
stored operands, inverse response and total relaxed energy remain outside
this conditional proof; #623/#625/#617 stay open.


The finite inverse/projector response increment (contract 5.56) is accepted
at source `49b10c64`, qualified tree `86d31ddc`: ten converged public
calculations and twenty qualified metric-channel responses, including an
explicit partial-rank RHF control. Decimal analytic and independent complex
perturbation witnesses cover rank limits, rotation, clusters, moving cutoffs,
subnormals and strict/resource refusals. See the metric spectrum benchmark
for actual native identity, supports, bounds, cost and reference limits.
This does not qualify coupling construction, J/K/energy propagation or total
relaxed accuracy; the broader #617 programme remains open.


The conditional metric-source Coulomb action increment (contract 5.57) is
accepted at source `a7a00f5a`, qualified tree `bdbe3132`: eleven converged
public calculations and one strict complete-method refusal through local
`vq`. Three fitted profiles reuse accepted states and independent reference
receipts; four selected/control pairs preserve physical states bitwise.
The shared inverse-response norm-family producer supplies a source-only
contribution, added outward to the stored-factor action audit when explicitly
requested. Explicit, compressed and pair-tree coupling producers are reused.
All 223 focused source and 104 public/preparation/documentation checks pass;
17 source-action checks also pass inside the calculation job. See the
[metric-source benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#conditional-metric-source-contribution-to-coulomb-action-targets)
for actual native identity, component bounds, physical supports, costs and
reference limits. This does not alter the global ledger or establish source
operand accuracy, K response, total relaxed energy, bulk CRYSTAL parity or
speedup. The broader #617 programme remains open and experimental; shared
Fock, PBC/BIPOLE and symmetry contracts are preserved.


The conditional metric-source K response (contract 5.58) is accepted at
source `88d8b48d`, qualified tree `ab2790ba`: nine converged public
calculations, seven raw component receipts and 21 actual metric channels.
Three fitted profiles reuse accepted state/reference receipts; UHF and both
PBE0 spin variants match their controls. Seventy-six focused source checks
pass; fourteen also run through local `vq`, including a strict complete-method
refusal. Independent 180-digit Decimal metric perturbations cover weights,
complex spin densities, rank limits and physical range kernels. See the
[exchange metric-source benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#conditional-metric-source-contribution-to-fitted-exchange)
for actual native identity, inputs, component bounds, costs and reference
limits. Separate raw targets do not alter the fitting/signed/global ledgers
or establish total relaxed accuracy. Shared Fock/PBC/BIPOLE/symmetry contracts
remain intact; MANTA and the unfinished #617 programme remain experimental.


The short-range metric Bloch-phase increment (contract 5.59) is accepted at
source/qualified tree `2425cd61`: six converged complete calculations reuse
accepted physical states and references, with 23 qualified metric-owner
receipts carrying the phase contribution. The local `vq` job passes 36
phase/assembly/refusal checks. Independent 800-digit phase and contracted
stored-integral witnesses validate the new contribution; primitive integral
accuracy remains outside that comparison. Source (91), J/K consumer (30)
and public preparation/documentation (104) checks pass. See the
[phase benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#short-range-metric-bloch-phase-construction-bound)
for actual native identity, supports, component bounds, costs and reference
limits. This narrows the earlier stored-phase premise for the short-range
metric only. No total relaxed-energy, automatic useful accuracy, bulk parity
or speedup claim is added. MANTA and the unfinished #617 programme remain
experimental; shared Fock/PBC/BIPOLE/symmetry contracts are preserved.


Reciprocal metric weight construction (contract 5.60) is accepted at
source/qualified tree `313d1e74`: six complete calculations reproduce accepted
states and references, with 23 qualified metric owners including the new
weight contribution. The local `vq` job passes 78 weight/phase/assembly and
strict-refusal checks; separate source (133), J/K (30) and public
preparation/documentation (104) checks pass. Independent Decimal exp/rational
and contracted stored-Fourier witnesses cover physical kernels, range
cancellation, finite-zero terms, underflow and resource boundaries. See the
[weight benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_metric_spectrum.md#reciprocal-metric-weight-construction-bound)
for actual native identity, physical supports, bounds, cost and reference
limits. Pi/volume/prefactor, primitive/Fourier/coupling construction, domains
and relaxation remain excluded. The broader #617 programme stays open;
no total relaxed-accuracy, bulk parity or speedup claim is added.
