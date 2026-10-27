# Experimental translation-native restricted CCSD residual

The internal `vibeqc.dlpno.translation_ccsd.evaluate_translation_ccsd` entry
point evaluates complete restricted singles and doubles residuals and the raw
correlation energy at arbitrary admissible amplitudes. It is not an iterative
CCSD driver, a new runner method or a scientific acceptance result. Chi remains
experimental, D93 fitted/post-HF remains not-qualified, and #557's production
periodic DLPNO deferral is unchanged.

## Shared equations and supported scope

The molecular compiled PNO engine and the translation evaluator call the same
C++ residual contraction templates in `cpp/src/dlpno_ccsd.cpp`. The molecular
view retains its existing factors and PNO data. The translation view requests
individual placed operands through an explicit callback interface, with native
contractions for every residual term. It does not instantiate the molecular
solver, construct a canonical virtual frame, fold global occupied/virtual DF
blocks, or retain a pool of placed pair bases or amplitudes.

Supported inputs are the existing admitted real restricted translation source,
sealed home-auxiliary or retained-Bloch factors, compact occupied columns,
complete projected PAO/PNO recipes, and a complete strong-pair inventory.
Frozen and active projectors remain independent. All source/reference and
native transport admission checks still apply; source admission does not
establish scientific qualification. Generic unsealed factors, dense pair
storage, unrestricted references and incomplete/screened pair inventories
are outside this entry point. There are no triples or independent-spin terms.

The equations follow Riplinger and Neese,
[doi:10.1063/1.4773581](https://doi.org/10.1063/1.4773581), Sec. II A and II C,
Eqs. (3)-(18) and (30), and the existing spin-integrated DF-CCSD engine.
DePrince and Sherrill,
[doi:10.1021/ct400250u](https://doi.org/10.1021/ct400250u), Sec. II,
Eqs. (3)-(24), provide the factorized CCSD background; their SI gives numerical
DF/CD/FNO comparisons, not additional translation equations. This engine
retains explicit singles contractions rather than adopting that paper's
non-Hermitian singles-transformed factors. The storage change introduces no
new CC approximation. It does not adopt optimized Wannier orbitals, pair
screening or the papers' calibrated truncation recipes.

## Translation contract and normalization

An active occupied index is `(cell, band)`. Singles are supplied once per home
active band in that band's diagonal-pair space. Doubles are supplied once per
canonical unordered translation orbit, in the existing representative PNO
gauge. Reversal transposes the doubles tensor; translation shifts its basis.
Self-inverse pair tensors must satisfy the existing stabilizer sewing relation
`T = U @ T.T @ U.T`. Inputs that violate it are refused, not repaired.

For a home target, every occupied contraction visits every cell and active
band once. Residual terms carry no energy orbit weights. Every contracted
virtual index stays in its amplitude's PNO space. Free indices are transported
with the existing stable two-sided native overlap reconstruction, including
complete occupied-projection tails. Frozen occupieds contribute to those
projection tails but never enter active excitation sums. The SCF Fock,
including its exchange zero mode and off-diagonal occupied entries, is retained.

For a finite system with N cells, divide the molecular correlation energy by
N. Each home singles term then occurs once per active band. Each unordered
pair contributes its translation-orbit multiplicity times `(2-delta_ij)/N`.
Ordinary off-diagonal families have weight two; diagonal and same-band
self-inverse families have weight one. These weights come from the independently
audited native transport inventory. The energy uses the molecular engine's
pair-projected singles product in tau; it is not a second energy expression.
The build's MP2 discarded-PNO estimate is returned separately as
`pno_mp2_correction_per_cell` and is never added to the raw CC energy or residual.

## Internal interface and resource scope

Call the evaluator with a source, its matching `TranslationPNOBuild`, home
singles and representative doubles. A matching bounded transport can be reused;
otherwise it is prepared explicitly. `terms=True` returns cumulative snapshots
for the bare, occupied/virtual ladder, Fock, ring, and singles contraction
groups. Differences between successive snapshots isolate the groups. They
are diagnostic arrays, not partial-equation execution modes.

The native boundary obtains representative indices, bands and energy weights
from the owned transport payload and independently checks the occupied Fock
against that payload. Operand shapes and finite values are checked natively.
Python binds source, partitions, factors, recipes and amplitudes before calling
it. The callback only supplies numerical operands; the CC equations remain
compiled and shared with molecular execution.

The evaluator preflights a separate `max_workspace_bytes` numerical-payload
reservation. It includes residual/term outputs and conversion copies, live
native target/source operands, and the target's complete occupied-pair overlap
workspace. The Python reservation covers four cached operands, two cached AO
frames, live reconstruction/contraction consumers and conversion copies.
Caches belong to one evaluation, release before replacement, and clear on
success or failure. No reusable amplitude-dependent cache survives a call.

This first implementation still has quadratic occupied-pair overlap scratch,
full occupied Fock, complete representative projected-domain data, and logical
full auxiliary dimensions on individual transformed blocks. Callback dispatch
and repeated source-pair/factor reconstruction can be expensive. The CC cap
is distinct from MP2 transport and factor-provider budgets. Source/build/input
storage, source admission, separately budgeted transport and provider work,
external library scratch and container/allocator overhead are excluded.
It is not a process-memory bound or a scaling qualification.

## Verification and remaining work

The source ratchet compares arbitrary nonzero singles/doubles and named
contraction groups with the unreduced compiled molecular engine using identical
placed PNO gauges and factors. It independently compares full-domain,
zero-PNO residuals and energy with canonical CCSD, including singles-only and
doubles-only cases. Nonorthogonal metrics, frozen partitions, off-diagonal
occupied Fock, odd/even multidimensional meshes, self-inverse pairs and admitted
complex Bloch sources are covered. Dense folding exists only in these explicit
small-system oracles. Allocation guards prohibit implicit dense occupied/AO
operator conversion in the translation path.

Production-driver dispatch, iteration/DIIS over representative singles and
pairs, convergence/nonconvergence reporting, screened strong/weak-pair support,
and method-specific corrected CC output remain subsequent increments. No
converged translation-native CCSD result is claimed by this delivery. Matched
external-reference and containing-runtime vq scientific acceptance remain
separate from the executable residual checks.

The [snapshot benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/chi_ccsd_residual.md) records reservations,
Python traced peaks and the measured reconstruction regression through eight
cells. The [archive replay payload](../examples/periodic/chi_translation_ccsd_residual.py)
accepts an existing source-admitted HF archive, explicit frozen partition and
seed basis, and optional native/evaluator SHA-256 pins. It compares arbitrary
amplitudes with a separately capped canonical oracle and writes a residual
receipt through the output interface. Reuse the accepted H2 Gamma/odd and LiH
even archives recorded in #591; select LiH cosine seeds and test frozen core
zero and one independently. Existing reference jobs should not be duplicated.
A containing runtime must satisfy the pins before vq execution. This payload
is a residual comparison, not a converged CCSD scientific-acceptance case.
