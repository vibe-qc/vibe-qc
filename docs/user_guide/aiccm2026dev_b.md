---
myst:
  html_meta:
    "description": "Experimental χ-CCM / aiccm2026dev-b: finite-character (Γ-centred character-mesh) CCM for 3D finite-torus HF, Kohn-Sham, RI-MP2, and local-PNO CCSD(T), with four-center, RI, and RIJCOSX backends and explicit scientific caveats."
---

# χ-CCM / aiccm2026dev-b (experimental)

For opt-in fitted classic/SR RI RHF construction, independent Coulomb controls,
immutable source identities and supported-scope refusals, see the
[fitted support contract](../chi_fitted_support.md). Its implementation does not
confer independent numerical acceptance or production qualification.

## Explicit direct physical support

All-electron, three-dimensional direct RHF can select
`support_policy="physical_pairs"` in `run_aiccm2026dev_b_rhf`, or
`chi_support_policy="physical_pairs"` in
`run_periodic_job(method="aiccm", variant="chi", aiccm_backend="four_center")`.
The explicitly selected reproduction mode is `"historical_index_ball"`.
Omitting the keyword preserves the existing route selection and does not
request the new support record. Generic native defaults are unchanged.

The physical mode uses the shared periodic pair and quartet kernels. S, T,
and both halves of V_ne use physical AO-pair separation. The erfc nuclear
sources are centered on the geometric AO midpoint. J retains AB/CD products;
K retains AC/BD products and their wider output/density enclosure. Both
products obey the pair radius, and their midpoint separation obeys the
interaction radius. The long-range terms share the resolved Ewald envelope,
with the existing q=0/Madelung convention. E_nn uses its own pair-complete
point-charge sum, including self and background corrections. These are
consistent conventions, not a claim that the distinct operator domains are
identical. See Sharma and Beylkin, JCTC 17 (2021), 3916,
DOI: 10.1021/acs.jctc.0c01195, and the shared
[BIPOLE domain contract](../bipole_domain_contract.md).

`result.executed_support.as_dict()` records ordered S/T/V_ne, density and
terminal Fock cell labels and Cartesian vectors, actual J/V_ne reciprocal
arrays, all K transfer channels including q=0, full basis fingerprint,
requested/resolved settings, and normalization. Nuclear-repulsion domains
are captured during the actual energy traversal. Electron-nuclear membership
is a replay of the same native source selector on the returned V_ne shell
pairs; it is identified as geometric support before numerical screening.
Quartet cell lists are enclosures with explicit predicates and screening
controls, not a log claiming every enclosed quartet was evaluated. The
`.system` manifest stores the record and its SHA-256 identity.

This opt-in recorder uses unreduced full-character traversal and refuses an
estimated provenance size above two million terms before SCF. Native record
exports also enforce a bounded term budget. A refusal never lowers a physical
cutoff or emits a partial support record. Large-system streaming provenance
is not implemented. Fitted sources, ECP, other spin/functional methods,
post-HF, lower dimensions and non-diagonal SCF extension matrices are outside
this increment. Explicit unsupported combinations fail; no alternate source
is substituted. Skew three-dimensional Bravais cells with diagonal cyclic
extensions are distinct from non-diagonal extension matrices.

Converge AO-pair, quartet-interaction, nuclear and reciprocal supports
independently using their **executed** sets. Nuclear requests can be resolved
from Ewald precision, so an unchanged executed set is not a convergence step.
Compare fixed-density J/K and separate nuclear/electronic components as well
as totals. Overlap closure, an old energy pin, and agreement with a shared
builder do not establish physical accuracy. No cell-specific cutoff is a
universal default. Independent imported-runtime numerical acceptance remains
required; chi stays experimental and D93 v1 fitted/post-HF stays not-qualified.


For small-system symmetry studies, D128 can now project a compact AO action
into a χ localization result using its actual overlap. The diagnostic keeps
phases and general orbital mixing, and rejects a metric or occupied subspace
that fails covariance. It does not yet reduce DLPNO work or enable analytic
gradients. The [occupied-action specification](../experimental/aiccm2026dev_b_symmetry.md#occupied-space-action)
describes its API, tolerance, provenance and dense-memory limits, including
why cutoff convergence must be checked before changing a symmetry tolerance.
D129 extends these diagnostics to every product of a supplied closed spatial
set, retaining the nonsymmorphic whole-cell translations and testing their
occupied-space representation. It is not yet permission to omit equivalent
pairs or triples; see the [group contract](../experimental/aiccm2026dev_b_symmetry.md#spatial-group-and-occupied-cocycle).

D130 provides a small-system [native snapshot bridge](../experimental/aiccm2026dev_b_symmetry.md#restricted-native-snapshot-bridge)
from full-mesh, all-electron four-center chi RHF to the shared numerical
orbital-sewing checks. It requires actual recorded occupations and Fock data,
explicit frozen bands and an insulating-gap floor. It can refuse a converged
SCF whose residuals are too loose for the native contract. It does not enable
symmetry-reduced DLPNO-CCSD(T), RI-source certification or analytic gradients.

D131 corrects insufficient short-range nuclear-image support in the shared
four-center source. A three-character symmetry regression now passes the
unchanged orbital-sewing checks; the two-character negative control remains
gated. A frozen band selected inside a degenerate pair is not automatically
symmetry invariant. Full symmetry-reduced multi-k DLPNO-CCSD(T) is still under
development.

`aiccm2026dev-b` is the dev-era code selector for χ-CCM[^xccm-convention], the
finite-translation-group character approach to the variational finite-BvK-torus
CCM. The name is prose only: select it with
`run_periodic_job(method="aiccm", variant="chi", ...)` (see the
[AICCM page](aiccm.md)); the legacy `jk_method="aiccm2026dev-b"` keeps
resolving with a `DeprecationWarning`. Its sibling is
[`aiccm2026dev-a`](../aiccm2026dev_a.md), Γ-CCM, which uses the
union-and-weight/Wigner--Seitz integral-weighting construction. The selectors,
core mathematics, tests, examples, and historical design records remain separate so comparison
of the two approaches stays auditable.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention, currently
    `exchange_q0="bvk-ewald"` for the production χ-CCM path. Their distinction
    is not a choice of Coulomb kernels, and equality for a specified
    operator/route is evidence to establish, not a naming premise.

The [AICCM route support matrix](../experimental/aiccm2026dev_route_matrix.md)
indexes which route on either line supports HF, KS, MP2, CCSD(T), DLPNO, and
analytic gradients.

D89 records the B identity explicitly:
`ccm_approach="chi-ccm"`,
`ccm_construction="finite-translation-group-character"`, and
`evaluation_representation="gamma-centred-character-mesh"`. These immutable
fields appear on B diagnostics/results and in fleet, QVF vendor, and `.system`
metadata. QVF wrapper schema version 2 keeps the older human-readable
`representation` field from version 1 for compatibility, but
`evaluation_representation` is normative. The exact real-Gamma Fourier form
of the same χ-defined Hamiltonian is an evaluation control, not a Γ-CCM
identity. Pre-D89 records are not numerically retracted solely for lacking the
new fields, but they cannot be used as construction-comparison evidence.

```{warning}
The 3D SCF routes remain active experimental diagnostics, but their numerical
two-electron supports are not all absolute-energy converged. Current
four-center RHF, PBE, and PBE0 fleet rows preserve the low-level backend and
record the exact M5 precision/policy plus the resolved padded erfc image
radius. Their support is eligible only when the backend and policy match, the radius
is finite and beyond the executed electronic cutoff, and the duplicated
physical cutoffs agree. Eligibility is not numerical qualification. Every
fitted RI/RIJCOSX/MDF and post-HF chi result remains permanently
`not-qualified` in D93 v1. These rows provide convergence, representation or
route-plumbing evidence; both audit and comparison commands reject them as
quantitative rows. Missing or malformed support metadata also fails closed.

D103 independently binds corrected-gauge direct overlap-fold support. A drift
above `1e-2` stops before the first Fock build; a direct run with
`1e-4 < drift <= 1e-2` can execute but remains `not-qualified`; qualification
requires `drift <= 1e-4`, the fixed maximum-element diagnostic evaluated at
`1.5` times the base cutoff, and exact matches to the executed electronic
cutoff and character mesh. Fitted and post-HF routes record this support as
exactly `not-applicable`.

D102 records the historical shifted-pair Ewald cutoff defect. Its exact
`vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v1` evidence is permanently
`not-qualified`; the repair cannot upgrade stored v1 data. Commit `e578b86c`
repaired the helper with centered-displacement interplanar pair bounds. D104
replaces the blanket hold only for fresh exact
`vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v2` records. Qualification
binds `implementation="centered-displacement-interplanar-pair-bounds"`, repair
commit `e578b86c00268a172b651cae29c7845836e31e17`, finalized
producer/core/probe identity, verified repair ancestry, and a same-process
canary. It requires `repair_is_ancestor=true`. The exact
`vibeqc.ewald.shifted-pair-canary/v1` fixture is
`mgo-point-charges-basis-shift-37a1/c18`: at an 18-bohr cutoff it shifts the
oxygen point charge by 37 lattice vectors along `a1` and requires the absolute
energy difference to be at or below `1e-10` Ha. Audit and comparison bind that
evidence back to D77-clean finalized provenance. D93, D103, and D104 are
independent: passing any one does not supply the other two.
```

```{warning}
D101 provides a dedicated high-level χ route-control producer for the pinned
1D/2D/3D campaign, but it does not qualify RI support. A successful record is
always `status="ok"`, `evidence_role="route-plumbing-diagnostic"`,
`quantitative_status="not-qualified"`, and
`comparison_status="no-gamma-chi-construction-comparison-defined"`. Its
campaign JSON deliberately contains no absolute energy. The current producer
accepts only D77 clean-checkout provenance because the normative D91 bundle
transport/producer/auditor contract is not implemented. The compute-managed runtime
audited on 2026-08-11 was bundle-shaped and lagged main, but its
replaceable artifact path/checksum and digest-free launcher do not satisfy D91
immutability. D101 rejects that transport even after a routine bundle refresh;
the interpreter selected by `run.sh` must resolve to either a
separate D77-clean Git checkout or a future implemented D91 contract.
Here `status="ok"` means only that execution and the internal contract audit
succeeded. It does not override `quantitative_status` or create a comparison
verdict. Its nuclear equality remains a same-helper routing check and is not
the repair-specific D104 canary.
```

```{warning}
D99 pins shared fitted-route behavior at the χ-CCM-B boundary. Since
`ddbe859d1`, the supported 3D Gamma-only fitted paths and all multi-k RI and
RIJCOSX RHF, RKS, UHF, and UKS paths use the shared
`ewald_nuclear_repulsion` helper, not `nuclear_repulsion_per_cell`.
That routing statement remains true, but the earlier convergence claim does
not: `a8b9ac2f8` recorded the shifted-pair cutoff defect in the shared helper.
Its unshifted translation preselection could omit an image whose shifted pair
distance was inside the requested cutoff. Commit `e578b86c` repaired that
enumeration. Pre-repair and D102 v1 records remain revision-bound; fresh rows
must satisfy D104 v2. Gamma-only here names the one-character evaluation path
within χ-CCM, not the union-and-weight Γ-CCM construction.

From `6ce142339` until 2026-09-29 compact dense-core MDF calculations in the
MgO/STO-3G class failed closed through the B selector; the shared GDF layer
admits them since its auxiliary-image sum (GitLab #172), and B inherits that. The validated vacuum-padded MDF
envelope is unchanged, but every fitted row remains D93 `not-qualified`.
Commit `ad96a2f63` supplies a bounded shared-q auxiliary Fourier-transform
cache with bit-identical results and memory/performance effects only. B also
passes `ibz_native=False` and `compute_gradient=False` explicitly on every
fitted restricted and open-shell dispatch. Generic IBZ or GDF-gradient
defaults therefore cannot replace the complete unreduced character mesh or
open a total-gradient route. All 1D/2D SCF and analytic total gradients remain
fail closed; D83, D88, and D98 are unchanged.

D106 additionally exposes `rsgdf_tail_ke_cutoff` as diagnostic transport on
RI and RIJCOSX with `gdf_method="rsgdf"`. When a tail is supplied, the base
cutoff must be positive and finite and the tail must be finite and strictly
larger; it is forwarded exactly and recorded on the result. Four-center and
MDF combinations fail before SCF. Gamma-only RHF/RI also fails when a null
input would extend the shared automatic dense-core tail above the base;
provide the extended value explicitly or use a nontrivial character mesh. An
automatic value equal to the base builds no complementary shell and remains
inactive. A successful null B result therefore means no complementary tail
executed. D93 v1 campaign
producers still require a null tail, so this API path does not qualify a fitted
result or admit an absolute energy.
```

```{warning}
D88 corrects a lattice-convention defect in pre-D88 χ-CCM-B records.
`PeriodicSystem.lattice` stores lattice vectors as columns, so Cartesian
translations are `system.lattice @ n` and the BvK lattice is
`system.lattice @ diag(mesh)`. New convention records serialize
`lattice_vector_convention="columns"`, and successful fleet payloads record
`primitive_lattice_bohr`. The fleet audit binds the exact BvK matrix to those
fields; the Γ/χ comparator reports no comparison when the binding is absent or
inconsistent. Because the Madelung, probe-charge, and
fleet helpers are shared, all affected pre-D88 Γ-CCM and χ-CCM records require
rerun under the fingerprint rule; no missing convention field is inferred. A
symmetric lattice that commutes with `diag(mesh)` is numerically outside this
defect class, but its old record is not newly attested. The shared builders for
`graphene`, `mgo-slab`, `ice-ih`, `co2-dryice`, and `sio2-quartz` require fresh
Γ and χ fingerprints. Ordinary pre-D88 periodic GDF exact-exchange and BIPOLE
J/K records for affected lattice/mesh combinations also require audit and
rerun.

For the skew audit matrix `[[7,.4,.2],[.3,8,.5],[.1,.6,9]]` bohr and mesh
`(2,3,1)`, the corrected positive Madelung convention gives
`xi_M=0.138352993811598`. The pre-D88 fitted helper gave
`0.143621616230995`, an RHF two-electron seam overbinding of 5.268622
mHa/cell, while the pre-D88 four-center probe gave `0.138913224426180`.
The theory article already uses the column convention correctly. D88 is a
code and fingerprint repair, not a kernel, sign, or theory change. The 1D/2D
absolute-energy guard and analytic-total-gradient guard remain fail closed.
D89 records that Γ-CCM and χ-CCM are distinct approaches, not representation
aliases. This warning holds the declared exchange-q=0 convention fixed and
does not assign different Coulomb kernels to the approach names.
```

χ-CCM defines a finite Born--von Karman translation group from a real-space
lattice extension. Its full Γ-centred character net is then derived
exactly. It minimizes RHF/RKS or UHF/UKS energy over
translation-commuting idempotent spin densities. Wigner--Seitz weights select
only tied representatives of one translation class. They are not multiplied
into an otherwise non-periodized three- or four-center tensor.

## Run it

```python
result = vq.run_periodic_job(
    system,
    basis,
    method="RKS",
    functional="pbe0",
    jk_method="aiccm2026dev-b",
    aiccm_lattice_extension=(2, 1, 1),
    aiccm_backend="ri",  # "four_center", "ri", or "rijcosx"
    rsgdf_ke_cutoff=200.0,
)

check = result.aiccm2026dev_b
print(result.converged, result.n_iter)
print(check.density_idempotency_error, check.electron_count_error)
print(check.coulomb_kernel, check.exchange_q0, check.boundary_model)
print(check.exact_exchange_assembly)
```

This is a route and invariant check, not a fleet attestation. Do not quote the
returned absolute energy unless a fresh producer record independently passes
D77, D93, D103, and D104.

Every successful χ-CCM-B SCF result owns an
`AICCM2026DevBExactExchangeAssembly` at
`result.aiccm2026dev_b.exact_exchange_assembly`; the convenience attribute
`result.exact_exchange_assembly` refers to the same object. Its `c_full`,
`c_sr`, and `omega_screen_bohr_inv` fields come from the shared live periodic
exchange resolver. The immutable schema is
`vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1`; the resolver field records
`vibeqc.periodic_screened_exchange.resolve_periodic_exchange`.
`screened_exchange_applicability` is active exactly when `c_sr` is nonzero.
Inactive routes use `screened_exchange_assembly="not-applicable"`. The
implemented four-center HSE route uses `short-range-direct`; the distinct
`full-range-minus-long-range` convention is not implemented by χ-CCM-B.
RHF/UHF records carry `(1, 0, 0)`, pure DFT carries `(0, 0, 0)`, a global
hybrid carries its nonzero full-range fraction and no screened arm, and an
HSE-type record carries zero full-range fraction plus a positive screened
fraction and physical inverse-bohr omega. An active screened label is attached
only after the BIPOLE K-erfc branch emits matching execution evidence. That
guard does not independently validate the numerical matrix or qualify its
support.
Scalar run fields in `.system`, B band metadata, and the
`x_vibeqc.aiccm2026dev_b_convention` QVF payload preserve the complete
descriptor. The `.out` invariant block reports its schema, screened
applicability, and assembly. The QVF wrapper payload uses schema version 2;
its nested exact-exchange object has the independent v1 schema above.

For vibe-view-ready periodic archives, leave `output_qvf=True` (the default).
χ-CCM-B writes torus-periodic density grids over the full BvK cell and, for
restricted records, torus-periodic orbital grids. Current periodic QVF output
also includes a `wavefunction.gto` section built from the Gamma character (or
the k=0 block of a multi-character result) as an additive raw-coefficient
payload. The precomputed torus grids remain the periodic visualization
contract. Add `qvf_wannier_centers=True` to also localize the occupied
finite-torus space and embed an `x_ccm.wannier_centers` overlay in Angstrom /
Angstrom-squared units; unrestricted records currently emit the spin-summed
density plus alpha/beta Wannier centres, not spin-resolved orbital grids.

`aiccm_wigner_seitz_shells=2` is the radius-style alternative. It produces
five primitive translations in every active direction, from (-2) through
(+2), and hence an odd cyclic extension of five. The legacy `kpoints=` tuple
is retained as an exact alias, but it is not a second convergence parameter.

Mesh, lattice-extension and Wigner-Seitz shell counts must be integers.
Python and NumPy integer inputs are accepted; floats (including `2.0`),
booleans and strings are rejected before constructing cells or k-points.
Orders must be positive and shell counts nonnegative. Scalar counts repeat
over active directions; omitted inactive directions use one cell and zero
shells. Explicit inactive entries must use those same values.

## Creating inputs

There are three ways to create a χ-CCM input, from fastest to most
customisable.

### 1. The test-set runner (fastest)

The `studies/aiccm-2026/` runner covers every implemented closed-shell route and
lists unsupported theory gaps explicitly.

```bash
cd studies/aiccm-2026

# Quick start - 3-D MgO RHF/RI route smoke; default KE is not quantitative
python run_case_b.py mgo rhf-ri

# Four-center diagnostic only; absolute values are revision-bound
python run_case_b.py c-diamond rhf-4c

# RKS/PBE with RI backend on diamond
python run_case_b.py c-diamond rks-pbe-ri

# Hybrid functional with RIJCOSX
python run_case_b.py bn-zb rks-pbe0-rijcosx

# Post-HF: canonical RI-MP2 on 3-D LiH rocksalt (3-D only)
python run_case_b.py lih-rocksalt ri-mp2

# DLPNO local correlation (exact limit = canonical oracle)
python run_case_b.py lih-rocksalt dlpno-mp2 --local-mode exact

# Override cluster size and basis
python run_case_b.py mgo rhf-ri --mesh 2 2 2 --basis pob-tzvp-rev2
```

Every run writes `<system>__b-<route>.json` with energy, orbital properties,
SCF diagnostics (idempotency, electron count, commutator norm, DIIS subspace,
accelerator settings), and for post-HF routes: correlation components, pair
counts, T1 norm, PNO correction, and finite-torus factor residuals.
The separate `run_case_cmp_b.py` producer owns only the pinned D101 χ
route-control campaign. Invoke `run.sh --comparison-b` for system `3d` with
`--expected-source-sha <full-origin-main-sha>` and `--vq-host compute-managed`; the 3D
record must pass first, and its exact fetched JSON is then supplied with
`--validation-record` for `1d` and `2d`. All three systems remain declared
`dim=3`. The result filename is
`d101-cmp1d2d3d-<system>-chi-rhf-ri.json`, and `audit_cmp_b.py` checks it
without rendering energy. This is not the generic fleet producer and is not a
Γ-CCM comparison route.

```bash
python audit_cmp_b.py \
  --expected-source-sha <full-origin-main-sha> \
  d101-cmp1d2d3d-3d-chi-rhf-ri.json
```

For a `1d` or `2d` record, add
`--validation-record d101-cmp1d2d3d-3d-chi-rhf-ri.json`. The auditor requires
the exact fetched parent file and recomputes its SHA and identifiers.

D101 filenames are deliberately outside the quantitative fleet globs and are
not inputs to `audit_b.py`, `compare_b.py`, or curation. Its GDF method and
cutoff evidence is a B-selector-resolved setting forwarded into the fitted
driver, not backend-owned telemetry or numerical-support qualification.
`declared_model="3d-periodic-in-vacuum"` describes the chain or monolayer
geometry inside a vacuum-padded cell; `boundary_model="3d-periodic"` describes
the fully 3D periodic Green function. This is not an isolated wire/slab
exception to D72/O1. The asserted compute-managed target and numeric Slurm job id record
operator intent and scheduler context, not cryptographic proof of the host
alias.

`scf_options.fock_mixing` is the requested input, whereas
`convergence_diagnostics.fock_mixing` is the executed effective previous-Fock
weight after backend defaults are resolved. The values can differ for a
DIIS-off four-center KS run, where a request of `0.0` can execute as `0.30`.
This is provenance only; it does not change the SCF algorithm or energy.
Pre-D86 records in that case are not convergence-fingerprint-complete.

For direct Python SCF calls, a non-`None` `fock_mixing=` keyword overrides
`options.fock_mixing`; otherwise the options field supplies the request. The
selected request must lie in `[0, 1)`, and resolving it does not rewrite the
caller's options field. All four `four_center` routes and multi-cell fitted
RHF/RKS can execute a nonzero request. Three-dimensional Gamma-only RI RHF
accepts resolved zero but rejects nonzero because mixing would switch to the
legacy cutoff-selected Gamma GDF fallback; the other one-cell fitted
restrictions are unchanged. Fitted RI/RIJCOSX UHF/UKS do not implement a
previous-Fock mixing loop and therefore
reject a nonzero request explicitly.
An explicit keyword zero overrides a nonzero options value, although that
requested zero can still resolve to executed `0.30` on a DIIS-off four-center
KS route under the documented automatic rule. D87 changes convergence-control
selection and validation on supported execution routes, not their finite
Hamiltonian or backend mixing formula. The Gamma explicit-zero correction
deliberately restores the declared operator instead of preserving the old
route-selection bug. Successful pre-D87 direct calls with differing
keyword/options values, plus fitted RI/RIJCOSX UHF/UKS calls with a nonzero
options-only request, have
incomplete request fingerprints. Pre-D87 Gamma-only RI RHF calls where either
input source was nonzero, including an explicit-zero keyword over nonzero
options, followed the legacy operator and are not χ-CCM-B results. The 1D/2D
absolute-energy and analytic total-gradient fail-closed policies remain in
force. D72 now runs before every backend-specific Gamma and mixing guard,
closing a former 1D one-cell RI-RHF escape. Any value from that route is
invalid and must not be reported.

For `jk_method="aiccm2026dev-b"`,
`run_periodic_job(dynamic_damping=...)` controls the underlying SCF adaptive
damping flag explicitly. `None` preserves the selected method option default;
`True` or `False` overrides it, and `False` disables adaptive damping updates.
Other J/K selectors currently fail closed on an explicit value because not all
of their dispatches preserve the option. A static `damping=0.0` request alone
does not disable an independently enabled dynamic controller, so fingerprinted
routes such as D101 pass `dynamic_damping=False` as a separate pin.

Unrestricted level shifting is currently fail-closed on every backend. The
four-center UHF/UKS loop applies a restricted-density half coefficient where
a unit-occupation spin projector requires the full subtraction, while the
fitted UHF/UKS loop does not execute the request. Consequently,
`PeriodicRHFOptions` or `PeriodicKSOptions` passed to a B UHF/UKS call must
have `level_shift=0` and an empty or all-zero `level_shift_schedule`.
The public `run_periodic_job(..., level_shift=...)` surface forwards the
resolved value into the B options object and therefore enforces the same gate.
Restricted static level shifting remains available on four-center routes and
on fitted routes with at least two characters. Restricted scheduled shifting
is available only on fitted routes with at least two characters. Gamma-only
RHF/RI is the D107 exception: a nonzero static shift would select the legacy
cutoff-selected Gamma GDF fallback instead of the declared fitted periodic
operator, while a nonzero schedule is not executed by the pure Gamma PBC-GDF
path. D108 rejects every nonempty restricted four-center options-level
schedule because the χ wrapper does not transport it into the BIPOLE driver.
Even an all-zero schedule must supersede a simultaneous static shift under the
public contract, so treating it as inert would execute the wrong control. For
Gamma-only RI, use `level_shift=0` with an empty or all-zero schedule. For a
static four-center shift, use an empty schedule. Use at least two cyclic cells
for scheduled restricted shifting.

D110 makes the separate four-center static warm-up control executable. With a
nonzero `options.level_shift` and an empty explicit schedule,
`options.level_shift_warmup_cycles=-1` selects up to five shifted startup
cycles, a positive integer requests that many startup cycles, and the resolved
length is capped to leave one unshifted tail cycle. An active auto or positive
warm-up requires `max_iter >= 2`; a smaller budget fails before SCF because it
cannot contain both a shifted startup and an unshifted tail. Zero requests the
persistent static-shift mode even for a one-cycle budget. When the static shift
is zero, the warm-up modifier is dormant and uninterpreted. The wrapper does
not modify the options object: the
options field remains the request, while
`result.aiccm2026dev_b.level_shift_warmup_cycles` records the executed
effective length. With an empty schedule, diagnostic zero means persistent for
a nonzero static shift and inactive for a zero shift. On a fitted route with a
nonempty explicit schedule, zero means that the warm-up modifier was
superseded; the caller's `options.level_shift_schedule` remains the
authoritative request. Inactive unrestricted routes also record executed
warm-up zero instead of the raw default `-1`.
`run_periodic_job(level_shift=...)` inherits the options default and therefore
uses auto release on a restricted four-center route. Set
`level_shift_warmup_cycles=0` on a directly constructed periodic options
object when persistent shifting is specifically intended. This generated
warm-up schedule does not enable the D108 explicit options-level schedule
surface; every nonempty explicit four-center schedule still fails closed.

Pre-D107 records with a nonzero static Gamma RI shift are not χ results. Gamma
RHF/RI and derived post-HF records with either a nonzero static or scheduled
shift are request-fingerprint-invalid and revision-bound. Pre-D108 restricted
four-center records with a nonempty schedule are likewise request-fingerprint-
invalid and revision-bound because the schedule did not execute. These guards
are applied after the common dimension check, so 1D/2D inputs still report the
missing wire/slab Hamiltonian rather than a convergence-control error.
Pre-D110 declared-3D restricted four-center RHF/RKS `ok` or `not_converged`
records with a nonzero static shift, an empty explicit schedule, and auto or
positive warm-up are also request-fingerprint-invalid and revision-bound. The
old wrapper executed a persistent shift while recording a finite release
request. Only a run extending beyond the resolved warm-up proves that the
intended release point was crossed, so the quarantine does not itself prove an
energy change. Error/unsupported records, zero-shift dormant warm-ups, and an
explicit persistent warm-up of zero remain valid. Current public χ post-HF
builders use fitted RHF references and are outside this four-center defect
class.

Quadratic SCF fallback is also fail-closed on every χ backend. Set
`quadratic_fallback_iter=0`, which is the default. A positive value formerly
looked like an active C1c orbital-rotation request in campaign metadata, but
the BIPOLE and fitted GDF/COSX loops did not read the activation, denominator
shift, or maximum-step fields. D109 now rejects a positive activation before
SCF and rejects a negative activation as invalid. The dormant
`quadratic_fallback_shift` and `quadratic_fallback_max_step` values do not
activate a fallback while the iteration is zero. Existing dimension, Gamma-RI
mixing/shift, restricted four-center schedule, and unrestricted level-shift
errors keep their earlier precedence. Closed-shell χ MP2 and local-correlation
calls inherit the same guard through their RHF reference. A pre-D109 3D
`ok`/`not_converged` SCF or derived post-HF record that accepted a positive
activation is conservatively request-fingerprint-invalid and revision-bound
because its backend neither supported nor attested that conditional request.
This does not prove that the trigger was reached or that the energy changed;
only `iterations > quadratic_fallback_iter` shows that the intended trigger
was crossed. Unsupported/error records remain valid failure evidence, and
nondefault shift/step values with activation zero remain valid and dormant.
This convergence-control correction does not change the finite Hamiltonian,
the distinct Γ-CCM and χ-CCM constructions, their declared common exchange-q=0
comparison convention, or the analytic-total-gradient fail-close.

### 2. Fleet batch generation

Generate the full benchmark matrix for inspection. The current coverage
profile remains preview-only because every emitted job uses a fitted SCF or
post-HF route that is D93 `not-qualified`:

```bash
cd studies/aiccm-2026

# Coverage: runnable fitted SCF + post-HF on 3D LiH, with direct/low-D skips
python make_jobs_b.py --profile coverage

# First AICCM-paper χ-CCM route matrix
python make_jobs_b.py --profile paper1

# Same route/mesh matrix at the CRYSTAL-matched orbital basis
python make_jobs_b.py --profile paper1 --basis pob-tzvp-rev2

# All closed-shell systems through all nine SCF paths
python make_jobs_b.py --profile scf

# All valid post-HF inputs
python make_jobs_b.py --profile posthf

# Diamond-family RKS/PBE/RI convergence investigation
python make_jobs_b.py --profile investigate

# Focused command previews
python make_jobs_b.py --profile scf --system mgo --host compute-large
python make_jobs_b.py --profile posthf --system lih-rocksalt --host compute-large
```

The coverage profile emits ten runnable fitted LiH jobs, 18 explicit
lower-dimensional failures, and three direct LiH/STO-3G support failures at
the measured 15-bohr drift of `7.0985e-2`. Runnable does not mean quantitative.
Other direct commands may be run as diagnostics; their support eligibility
requires fresh finalized D77 provenance and independent D93, D103 and D104
contracts. Eligibility alone is not qualification. Fitted and post-HF results
remain permanently not-qualified in v1.

Each emitted job claims 4 CPU slots / 8 GB memory. Select an enabled external
`AICCM_SITE_CONFIG` profile before using host defaults, or choose an explicit
remote queue alias with `--host`. Use `--local` for local rendering. The
`paper_host`, `tier_hosts` and `posthf_host` settings select the corresponding
targets; names such as `compute-large` in these examples are placeholders.
Lower-dimensional anchors remain explicit fail-closed coverage until O1
supplies a shared wire/slab Coulomb kernel. See the study's
[site configuration guide](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/site-configuration.md).

### 3. Building a system from scratch - worked examples

Every example is a complete, runnable script. Copy-paste, adjust the geometry
and basis, and run. Printed fitted and post-HF energies are local diagnostics
and remain permanently not-qualified in v1. Direct-route support eligibility
requires independent D77, D93, D103 and D104 evidence.

---

#### Example 1 - 3-D H₂: all three backends side by side

```python
import numpy as np
import vibeqc as vq

system = vq.PeriodicSystem(
    3, np.diag([8.0, 12.0, 12.0]),
    [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

common = dict(
    jk_method="aiccm2026dev-b",
    aiccm_lattice_extension=(2, 1, 1),
    max_iter=40, progress=False,
    citations=False, write_xyz_file=False, output_qvf=False,
)

for backend in ("four_center", "ri", "rijcosx"):
    r = vq.run_periodic_job(system, basis, method="RHF",
                            aiccm_backend=backend, **common)
    d = r.aiccm2026dev_b
    print(f"{'RHF/' + backend:>14s} E/cell = {r.energy:.12f} Ha  "
          f"idem = {d.density_idempotency_error:.1e}")
```

This example checks dispatch and internal consistency. It is not a license to
interpret the three printed finite-cutoff values as independently converged
absolute energies.

---

#### Example 2 - 1-D polymer chain: RI backend with vacuum padding

```python
import numpy as np
import vibeqc as vq

# Polyethylene-like 1-D chain: 2 CH₂ units along x, 20 bohr vacuum in y,z
system = vq.PeriodicSystem(
    1, np.diag([5.0, 20.0, 20.0]),
    [vq.Atom(6, [0, 0, 0]), vq.Atom(1, [0.9, 0, 0]),
     vq.Atom(1, [-0.9, 0, 0]),
     vq.Atom(6, [2.5, 0, 0]), vq.Atom(1, [3.4, 0, 0]),
     vq.Atom(1, [1.6, 0, 0])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# 1-D: all current χ-CCM-B SCF backends fail closed until O1 lands.
for backend in ("ri", "rijcosx"):
    try:
        vq.run_periodic_job(
            system, basis, method="RKS", functional="pbe",
            jk_method="aiccm2026dev-b",
            aiccm_lattice_extension=(4, 1, 1),
            aiccm_backend=backend,
            max_iter=60, progress=False,
        )
    except NotImplementedError as exc:
        print(f"{backend} is intentionally blocked: {exc}")
```

---

#### Example 3 - 2-D slab: MgO(001) surface

```python
import numpy as np
import vibeqc as vq

a = 4.21  # Angstrom
# 2-D slab: periodic in xy, 50 bohr vacuum in z
lat = np.array([[a/2, a/2, 0], [-a/2, a/2, 0], [0, 0, 50.0]])
system = vq.PeriodicSystem(
    2, lat,
    [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [0, a/2, 0])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

try:
    vq.run_periodic_job(
        system, basis, method="RHF",
        jk_method="aiccm2026dev-b",
        aiccm_lattice_extension=(2, 2, 1),
        aiccm_backend="ri",
        max_iter=60, progress=False,
    )
except NotImplementedError as exc:
    print(f"2-D χ-CCM-B RI is intentionally blocked: {exc}")
```

---

#### Example 4 - Diamond: hybrid functional comparison (PBE vs PBE0)

```python
import numpy as np
import vibeqc as vq

a = 3.5670
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = vq.PeriodicSystem(
    3, lat,
    [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [a/4, a/4, a/4])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

for func, backend in [("pbe", "ri"), ("pbe0", "ri"),
                       ("pbe", "rijcosx"), ("pbe0", "rijcosx")]:
    r = vq.run_periodic_job(
        system, basis, method="RKS", functional=func,
        jk_method="aiccm2026dev-b",
        aiccm_lattice_extension=(2, 2, 2),
        aiccm_backend=backend,
        max_iter=80, progress=False,
    )
    gap = r.aiccm2026dev_b.fundamental_gap
    print(f"{'RKS/' + func + '/' + backend:>20s} E/atom = {r.energy / 2:.8f} Ha",
          f"gap = {gap:.4f}" if gap else "")
```

---

#### Example 5 - LiH rocksalt: full post-HF stack

```python
import numpy as np
import vibeqc as vq
from vibeqc.periodic.chi.posthf import (
    run_aiccm2026dev_b_mp2,
    run_aiccm2026dev_b_dlpno_mp2,
    run_aiccm2026dev_b_dlpno_ccsd_t,
)

a = 4.0840
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = vq.PeriodicSystem(
    3, lat,
    [vq.Atom(3, [0, 0, 0]),
     vq.Atom(1, (lat @ np.array([0.5, 0.5, 0.5])).tolist())],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
ext = (2, 2, 2)

# Canonical RI-MP2 (exact finite-torus oracle)
mp2 = run_aiccm2026dev_b_mp2(system, basis, lattice_extension=ext)
print(f"Canonical RI-MP2  Ecorr = {mp2.e_correlation:.8f} Ha")

# DLPNO-MP2 at the exact (no-truncation) limit == canonical
for mode in ("exact", "pno"):
    d_mp2 = run_aiccm2026dev_b_dlpno_mp2(
        system, basis, lattice_extension=ext, local_mode=mode,
    )
    print(f"DLPNO-MP2 ({mode:>5s}) Ecorr = {d_mp2.e_corr:.8f} Ha  "
          f"npairs = {d_mp2.n_pairs}")

# DLPNO-CCSD(T) -- exact limit and truncated
for mode in ("exact", "pno"):
    d_cc = run_aiccm2026dev_b_dlpno_ccsd_t(
        system, basis, lattice_extension=ext, local_mode=mode,
    )
    print(f"DLPNO-CCSD(T) ({mode:>5s}) Ecorr = {d_cc.e_correlation:.8f} Ha  "
          f"(T) = {d_cc.e_t:.2e}")
```

---

#### Example 6 - Local correlation: localization modes and PNO truncation

```python
import numpy as np
import vibeqc as vq
from vibeqc.periodic.chi.posthf import run_aiccm2026dev_b_dlpno_mp2

a = 4.0840
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = vq.PeriodicSystem(
    3, lat,
    [vq.Atom(3, [0, 0, 0]),
     vq.Atom(1, (lat @ np.array([0.5, 0.5, 0.5])).tolist())],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# Compare localization methods at the exact limit
for localise in ("pm", "wannier", "iao", "none"):
    r = run_aiccm2026dev_b_dlpno_mp2(
        system, basis, lattice_extension=(2, 1, 1),
        localise=localise, local_mode="exact",
    )
    print(f"DLPNO-MP2 l={localise:>7s} Ecorr = {r.e_corr:.8f} Ha")

# PNO truncation sweep
print("\nPNO truncation sweep (Pipek--Mezey):")
for tcut in (0.0, 1e-8, 1e-7, 1e-6, 1e-5):
    r = run_aiccm2026dev_b_dlpno_mp2(
        system, basis, lattice_extension=(2, 1, 1),
        localise="pm", local_mode="pno", tcut_pno=tcut,
    )
    print(f"  tcut_pno={tcut:.0e}  Ecorr = {r.e_corr:.8f}  "
          f"npairs = {r.n_pairs}")
```

---

#### Example 7 - Properties: band structure, Mayer bond orders, charges

```python
import numpy as np
import vibeqc as vq
from vibeqc.periodic.chi.properties import (
    derive_aiccm2026dev_b_scf_properties,
    aiccm2026dev_b_band_structure,
    aiccm2026dev_b_mayer_bond_orders,
)

a = 3.5670
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = vq.PeriodicSystem(
    3, lat,
    [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [a/4, a/4, a/4])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# SCF first
result = vq.run_periodic_job(
    system, basis, method="RHF",
    jk_method="aiccm2026dev-b",
    aiccm_lattice_extension=(2, 2, 2),
    aiccm_backend="ri",
    max_iter=80, progress=False,
)

# One-particle properties
props = derive_aiccm2026dev_b_scf_properties(result, system, basis)
print(f"HOMO = {props.homo:.4f} Ha, LUMO = {props.lumo:.4f} Ha")
print(f"Gap = {props.gap:.4f} Ha = {props.gap * 27.2114:.2f} eV")
print("Mulliken charges:", props.mulliken_charges)
print(f"Density idempotency: {props.density_idempotency:.2e}")

# Band structure (folded-Γ spectrum on the torus)
bands = aiccm2026dev_b_band_structure(
    system, basis, result,
    k_path_labels=["Γ", "X", "W", "K", "Γ", "L", "U"],
)
print(f"Band path: {len(bands.k_points)} k-points, "
      f"{bands.n_bands} bands")

# Mayer bond orders (primitive cell)
bonds = aiccm2026dev_b_mayer_bond_orders(result, system, basis)
for (i, j), bo in bonds.bond_orders.items():
    print(f"  Bond ({i},{j}): BO = {bo:.4f}")
```

---

#### Example 8 - restricted SCF convergence tuning

```python
import numpy as np
import vibeqc as vq

a = 4.0840
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = vq.PeriodicSystem(
    3, lat,
    [vq.Atom(3, [0, 0, 0]),
     vq.Atom(1, (lat @ np.array([0.5, 0.5, 0.5])).tolist())],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

base = dict(
    method="RHF", jk_method="aiccm2026dev-b",
    aiccm_backend="ri", aiccm_lattice_extension=(2, 2, 2),
    max_iter=100, progress=False,
)

# Default EDIIS+DIIS hybrid (from iter 2, subspace 8)
r_def = vq.run_periodic_job(system, basis, **base)
print(f"Default hybrid: {r_def.energy:.8f} Ha  "
      f"{r_def.n_iter} iters")

# Larger hybrid subspace (exact EDIIS face enumeration supports up to 12)
r_diis = vq.run_periodic_job(
    system, basis, diis_subspace_size=12, **base,
)
print(f"Hybrid n=12:  {r_diis.energy:.8f} Ha  "
      f"{r_diis.n_iter} iters")

# Delay acceleration until iteration 5
r_delayed = vq.run_periodic_job(
    system, basis, diis_start_iter=5, **base,
)
print(f"Hybrid start 5: {r_delayed.energy:.8f} Ha  "
      f"{r_delayed.n_iter} iters")

# Restricted level shift on this multi-character fitted route
r_ls = vq.run_periodic_job(
    system, basis,
    level_shift=0.3,
    **base,
)
print(f"LS 0.3:       {r_ls.energy:.8f} Ha  "
      f"{r_ls.n_iter} iters")

# Damping only, no DIIS
r_damp = vq.run_periodic_job(
    system, basis, damping=0.3, use_diis=False, **base,
)
print(f"Damp 0.3:     {r_damp.energy:.8f} Ha  "
      f"{r_damp.n_iter} iters")
```

---

#### Example 9 - Wigner-Seitz shell sizing (radius-style)

```python
import numpy as np
import vibeqc as vq

a = 3.5670
lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
system = vq.PeriodicSystem(
    3, lat,
    [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [a/4, a/4, a/4])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# Wigner-Seitz shell sizing: shells=1 -> 3 translations (-1,0,+1)
# shells=2 -> 5 translations (-2,-1,0,+1,+2) per active direction
for shells in (1, 2, 3):
    r = vq.run_periodic_job(
        system, basis, method="RHF",
        jk_method="aiccm2026dev-b",
        aiccm_wigner_seitz_shells=shells,
        aiccm_backend="ri",
        max_iter=40, progress=False,
    )
    d = r.aiccm2026dev_b
    print(f"shells={shells}  mesh={d.character_mesh_shape}  "
          f"E/atom = {r.energy / 2:.8f} Ha")
```

---

#### Example 10 - Direct SCF API vs run_periodic_job

```python
import numpy as np
import vibeqc as vq
from vibeqc.periodic.chi.scf import run_aiccm2026dev_b_rks

system = vq.PeriodicSystem(
    3, np.diag([8.0, 12.0, 12.0]),
    [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# Via run_periodic_job (high-level, I/O, output plan, citations)
r1 = vq.run_periodic_job(
    system, basis, method="RKS", functional="pbe",
    jk_method="aiccm2026dev-b",
    aiccm_lattice_extension=(2, 1, 1),
    aiccm_backend="ri",
)

# Via the direct SCF API (low-level, returns SCF result + diagnostics,
# emits AICCM2026DevBExperimentalWarning)
r2 = run_aiccm2026dev_b_rks(
    system, basis, aiccm_lattice_extension=(2, 1, 1),
    functional="pbe", aiccm_backend="ri",
)

print(f"run_periodic_job: E = {r1.energy:.12f} Ha")
print(f"Direct SCF API:   E = {r2.energy:.12f} Ha")
print(f"Convention: {r2.finite_torus_convention.coulomb_kernel}")
```

### All available routes

Every route is a single `python run_case_b.py <system> <route>` invocation.
The complete closed-shell χ-CCM method matrix:

| route | method | ERI backend | example |
|---|---|---|---|
| `rhf-4c` | RHF | four-center (3-D only) | `python run_case_b.py c-diamond rhf-4c` |
| `rhf-ri` | RHF | pair-resolved 3-center RI-J/RI-K (3-D only) | `python run_case_b.py mgo rhf-ri` |
| `rhf-rijcosx` | RHF | RI-J + COSX exchange (3-D only) | `python run_case_b.py bn-zb rhf-rijcosx` |
| `rks-pbe-4c` | RKS/PBE | four-center (3-D only) | `python run_case_b.py c-diamond rks-pbe-4c` |
| `rks-pbe-ri` | RKS/PBE | RI (3-D only) | `python run_case_b.py mgo rks-pbe-ri` |
| `rks-pbe-rijcosx` | RKS/PBE | RIJCOSX (3-D only) | `python run_case_b.py bn-zb rks-pbe-rijcosx` |
| `rks-pbe0-4c` | RKS/PBE0 | four-center (3-D only) | `python run_case_b.py c-diamond rks-pbe0-4c` |
| `rks-pbe0-ri` | RKS/PBE0 | RI (3-D only) | `python run_case_b.py mgo rks-pbe0-ri` |
| `rks-pbe0-rijcosx` | RKS/PBE0 | RIJCOSX (3-D only) | `python run_case_b.py bn-zb rks-pbe0-rijcosx` |
| `ri-mp2` | canonical RI-MP2 (3-D only) | pair-resolved finite-torus RI | `python run_case_b.py lih-rocksalt ri-mp2` |
| `dlpno-mp2` | DLPNO-MP2 (3-D only) | exact real representation of RI torus | `python run_case_b.py lih-rocksalt dlpno-mp2` |
| `dlpno-ccsd` | DLPNO-CCSD (3-D only) | exact real representation of RI torus | `python run_case_b.py lih-rocksalt dlpno-ccsd` |
| `dlpno-ccsd-t` | DLPNO-CCSD(T) (3-D only) | exact real representation of RI torus | `python run_case_b.py lih-rocksalt dlpno-ccsd-t` |

Post-HF `--local-mode` controls:

- `--local-mode exact` (default) - disables PNO and pair truncations; the
  accuracy oracle that reproduces the canonical finite-torus limit.
- `--local-mode pno` - exercises the current PNO approximation.

**Localization options** (post-HF): `localise="pm"` (PBC-safe Pipek-Mezey,
default), `localise="wannier"`, `localise="iao"`, `localise="none"` (canonical
occupieds - use with `--local-mode exact` for the validation limit).

### Audit B, inspect approach status, and compare controls or CRYSTAL

```bash
# Audit B records first. Reportable-status 1-D/2-D records fail this gate.
python studies/aiccm-2026/audit_b.py results-b/

# A true Γ-CCM/χ-CCM approach delta is currently not defined.
python studies/aiccm-2026/compare_b.py results-b/

# Optionally add the separately attested neutral-torus real-Gamma control.
python studies/aiccm-2026/compare_b.py results-b/ \
    --real-gamma-control-results results-control/

# With a CRYSTAL23 reference column
python studies/aiccm-2026/compare_b.py results-b/ \
    --real-gamma-control-results results-control/ \
    --crystal-refs studies/aiccm-2026/crystal_refs_b.json --csv comparison.csv
```

Only compare calculations with the same geometry, orbital basis, functional,
and reciprocal mesh. The cheap default cross-stream pass uses each registry
basis, usually STO-3G; the article/CRYSTAL pass should be emitted explicitly
with `--basis pob-tzvp-rev2` so its JSON records advertise the matched basis.
`compare_b.py` refuses old lower-dimensional B records with `status="ok"` or
`status="not_converged"`; these pre-guard absolute energies remain failure
evidence and are not table data.
The audit and comparator also require the recorded
`exchange_q0_applicability` to match the selected route. RHF, the post-HF
routes, and PBE0 require `active`; PBE requires `inactive`. A missing field or
an active/inactive contradiction fails closed before either an energy table or
a real-Gamma representation control is emitted.

D95 added top-level fleet `exact_exchange_assembly` beside
`finite_torus_convention`. The separation prevents the declared
`exchange_q0="bvk-ewald"` family from being mistaken for an always-live seam.
The recorded `resolver`, `c_full`, `c_sr`, and physical
`omega_screen_bohr_inv` must match the route-resolved live object, and
full-range q=0 applicability is active exactly when `c_full` is nonzero. A
screened-only HSE-type assembly therefore has inactive full-range
applicability but active screened applicability. These are separate facts.

D96 made the historical serialized object part of the fleet acceptance gate.
D98 replaces its unversioned four-field shape with the exact seven-key v1
schema. `audit_b.py` and `compare_b.py` validate the schema, resolver, finite
non-boolean coefficients, screened applicability and assembly, and route
values against an independent registry oracle: RHF and post-HF use
`(1,0,0)`, PBE uses `(0,0,0)`, and PBE0 uses `(0.25,0,0)`. Every current fleet
route requires screened inactive/not-applicable. Missing, legacy, malformed,
or route-inconsistent assemblies fail closed for successful and non-converged
records. Unsupported and error rows remain explicit failure evidence and need
not claim a successful assembly. HSE remains outside the fleet.

D93 adds an independent two-electron-support gate. Successful and
non-converged rows must carry the exact v1 support payload. Current
four-center RHF/PBE/PBE0 rows can pass from their preserved low-level backend,
exact M5 precision/policy, executed radius, and consistent physical cutoffs.
Every current fitted or post-HF row is deliberately
`not-qualified`, so neither `audit_b.py` nor `compare_b.py` will place its
absolute energy in a table. In particular, the optional B `rhf-ri` versus A
`aiccm-hf-direct` real-Gamma representation control remains unavailable even
if its older provenance scaffold would otherwise match.

D103 adds the exact
`vibeqc.aiccm2026dev-b.overlap-fold-support/v1` payload. Four-center rows bind
the executed maximum overlap-fold drift, the electronic cutoff, the exact
diagnostic identifier, the fixed `1.5` reference-cutoff factor, the executed
character mesh, the `1e-4` quantitative target, and the `1e-2` runtime stop
threshold. The payload is qualified only at or below the quantitative target
and must agree with `direct_lattice_cutoffs` and the row mesh; moderate support
can execute but is rejected by both tools. Fitted and post-HF rows use the
exact `not-applicable` form with null drift, cutoff, factor, and mesh.

D104 then requires every successful or non-converged row to carry the exact v2
shared-Ewald support object described above. Fresh D77-clean records can
qualify when the repair ancestry and same-process canary pass. Historical v1,
missing, malformed, unqualified, or provenance-inconsistent evidence fails
closed. This removes D102's blanket hold only for fresh post-repair evidence;
it does not qualify D93 or D103.

`run_case_cmp.py` emits D104 v2 for its neutral fitted-torus Bloch/GDF
same-helper control, but it remains overall
`quantitative_status="not-qualified"`. It is not the mapped real-Gamma route.
The mapped A-owned `aiccm-hf-direct` producer currently emits no D104 v2
evidence, so `compare_b.py` rejects that optional control.

D101 applies that same nonqualification to its dedicated high-level χ route.
Its internal total/electronic/nuclear decomposition audit must pass, but the
serialized campaign record contains no total or per-atom energy. An `ok`
calculation status is therefore never a quantitative or comparison verdict;
the accompanying evidence-role, quantitative-status, and comparison-status
fields above are mandatory. D101 emits neither the historical v2
`comparison_input` nor a `comparison_contract`.
Do not recover or copy a χ energy from `.out`, `.system`, the SCF progress log,
or any runner side artifact; those files are nonreportable diagnostics under
D93.

The reportable Γ/χ approach map is empty. B `rhf-ri` versus A
`aiccm-hf-direct` is a neutral-torus representation control, not evidence for
the union-and-weight Γ-CCM approach. The A RI and RIJCOSX harness routes call
the same multi-k GDF SCF drivers as B, so their agreement is a common-path
regression check. The direct route assembles and minimizes the real
Γ-supercell neutral-torus SCF problem but intentionally reuses the common per-q
RSGDF fit and one-electron primitives; it does not independently validate that
shared RI machinery. Equality of the Γ-CCM and χ-CCM approaches for any
specified operator and route remains evidence to establish under the common
exchange-q=0 convention.

Even this control remains fail-closed. Comparator contract v2 is an
incomplete validation scaffold and must not be emitted unchanged. It
requires both producer JSON records to embed the same canonical
`comparison_input`, with its hash in `comparison_contract.input_sha256`, and
validates that payload against the result and contract. It records
`ao_linear_dependence_threshold` separately from
`auxiliary_metric_linear_dependence_threshold`, and it requires requested and
reported `smearing_temperature=0.0` Ha so a finite-temperature SCF energy cannot
be labelled as a comparison delta. Comparator v2 does not bind Fock mixing,
and D86 leaves it unchanged and incomplete. A superseding comparison contract
must bind requested and executed accelerator controls. It also requires matching clean source,
native-core, host, package-version, and successful composite producer-process
attestation. The cached `aiccm-host-probe/v2` result is only a preflight. Each
runner binds it to the copied benchmark payload and to the identity of the
process producing the energy, records linked native-library versions, and
always runs a cheap shifted-mesh native-versus-Python Fourier-transform canary
in that process. If the native-core path SHA256 differs from preflight, the
producer reruns the full v2 API, reciprocal-cutoff, and LiH direct-versus-GDF
checks in process; the cheap canary alone cannot approve a changed core. It
then re-reads the identity, libraries, and payload immediately before writing
the result and accepts only an exactly stable result identity.

The core SHA256 identifies bytes currently at the imported module's filesystem
path, not the native image already mapped into memory. The same-process
numerical checks cover loaded behavior without claiming a cryptographic mapped
image identity. The comparator independently validates every nested schema,
digest, check version, and tolerance and compares the B and real-Gamma control
records by current producer identity, not by preflight history or their
intentionally different payload digests.

On an immutable-bundle host where clean-checkout identity is structurally
unavailable, D91 defines a possible future `attestation="bundle"` alternative.
The producer would have to checksum the artifact, prove its full source SHA is
resolvable on `origin/main`, record package, native-core, and native-library
identities, pass the full in-process numerical probe and loaded-core canary,
bind its producer payload, and recheck every identity layer before writing the
result. A reconstructed clean checkout instead records
`attestation="git-checkout"`. The current B producer, curator, and comparator
accept only D77 clean-checkout attestations, so no B bundle row is presently
reportable. The D91 alternative remains blocked until a producer creates an
authoritative immutable-artifact manifest and proves its full source SHA
resolvable on `origin/main`; D95 exchange coefficients do not supply either
identity proof. A future exception cannot upgrade an older record or replace any
numerical-input, two-electron-support, or construction-binding requirement.

The legacy `--a-results` option is accepted only as an alias for
`--real-gamma-control-results`. It does not identify the supplied records as
Γ-CCM and should not appear in new commands.

Neither runner emits `aiccm2026-gamma-chi/v2`, and that schema is not
sufficient for producer emission. A neutral-torus character/real-Gamma factor
audit found and fixed one prerequisite in the shared RSGDF builder. The builder
now enumerates the
physical shifted support `0 < |G+q| <= sqrt(2 E_cut)` instead of shifting an
already truncated `|G|` ball. This makes reciprocal-equivalent transfer labels
share one support while preserving time reversal at even-mesh Nyquist
channels. Canonical χ-CCM post-HF factor transforms now match the neutral-torus
folded fitted-Gram control to the numerical floor on both `(2,1,1)` and
`(3,1,1)` meshes; separate gates cover compact fitted-Gram and canonical-factor
relabelling. This is a same-Hamiltonian Fourier control, not Γ-CCM construction
evidence.

The q=0 RSGDF path is byte-preserved, but the finite nonzero-q support changed.
Treat pre-D78 records that built nonzero-q factors as revision-bound: RHF/UHF
and hybrid RI with exact GDF exchange, neutral-torus folds, and RI post-HF/full-pair
caches. Semilocal RI and RIJCOSX SCF use only q=0 RSGDF factors for J and are
numerically unaffected by this support change.

The acceptance schema still has open work. It must separate the common Bloch
pair-density phase from the producer-specific factor pipeline. Γ uses
canonical auxiliary-AO builder blocks, then an unfolded-k fold, real `q/-q`
stack, and null-row prune; χ SCF self-contracts compact metric eigenmodes.
The schema must establish q-resolved retained auxiliary-projector agreement
together with gauge-invariant fitted-Gram agreement, not only matching ranks,
thresholds, or projectors. For the current A route it must also require
full/no-drop at every B character and verify the
finite-Fourier overlap relation, so each character retains all primitive-cell
AO directions. If projected spaces are supported later, it must compare
Fourier-related retained-space projectors rather than rank counts alone. A
later `RUN.meta` or other curation sidecar cannot retroactively attest the
numerical input, build, probe, payload, or producer process that produced an
existing energy. Existing rows are not upgraded to this contract.

### From the qc-input-library

The `qc-input-library` has independent input generation for both streams:

```bash
cd /path/to/qc-input-library/aiccm2026testset

# Generate χ-CCM inputs for all 3-D systems
python generate_inputs.py --stream b

# Tier A only with SCF convergence variants
python generate_inputs.py --stream b --tier A --scf-sweep
```

## Electron-repulsion backends
The direct SCF APIs are `run_aiccm2026dev_b_rhf`,
`run_aiccm2026dev_b_rks`, `run_aiccm2026dev_b_uhf`, and
`run_aiccm2026dev_b_uks`. Every invocation emits
`AICCM2026DevBExperimentalWarning`.

Restricted 3D `four_center` RHF and non-screened RKS now farm complete
direct-ERI output blocks when launched with multiple MPI ranks. Without
symmetry-representative reduction, pure semilocal RKS farms the padded J-only
traversal; global hybrids farm the fused padded J/K traversal. A masked
symmetry-representative pure route may use the shared fused builder and discard
its K result. The χ selector uses cyclic assignment, keeps the full
M5 internal translation sum inside every task, and gathers complete blocks
before incremental Fock bookkeeping. XC grid work, reciprocal long-range
Hartree, long-range exchange, the BvK q=0 seam, character transforms, and
diagonalization remain replicated and are not multiplied by the rank count.
The numerical result on every rank is the same complete result; in serial,
the scheduling seam is bit-identical to the prior native traversal.

The executed contract is available as
`result.output_cell_farming_execution` and as
`result.aiccm2026dev_b.direct_output_cell_farming`. It records the task kind
`chi-direct-output-cell`, strategy, MPI world size and rank, global and local
task counts, all per-rank counts, ordered-cell fingerprint, and the fact that
the internal translation sum is complete. An `active` value of false means
the χ scheduling path executed in a one-rank world, not that a different
operator was selected.

This MPI layer is restricted to χ RHF and non-screened RKS with
`four_center`. HSE-type screened RKS stays serial because its base J and
separate screened-K traversals cannot both be described by the current
single-phase execution record; the generic driver rejects an explicit farming
request. UHF, UKS, RI, RIJCOSX, post-HF contractions,
and analytic gradients do not yet use this χ-owned output-cell layer. The
work unit is a radial direct-ERI AO
output task, possibly restricted by a symmetry shell mask. It is not a Γ-CCM
union-and-weight WSC and is not automatically one finite-group residue. D114
assigns no physical Wigner weight.

D117 separately distributes the density-diagnostic inverse transform over
actual χ finite-group residues. Before scheduling, it groups all tied
minimum-image representatives in the zero-offset cell set used by the density
diagnostic. Their equal alias weights must sum to one. This does not attest
the offset-dependent representatives for AO pairs, quartets, or local
domains. One task then evaluates that residue's complete finite-character
sum; no additional Wigner weight is applied. The complete mesh and uniform
dual character net, residue coverage, translation congruences, and normalized
weights are checked before active ranks compare a construction fingerprint.
Complete blocks are then gathered in canonical residue order, including an
empty parcel from any oversubscribed rank. Restricted and unrestricted χ
results expose the executed record as
`result.residue_inverse_transform_execution` and
`result.aiccm2026dev_b.residue_inverse_transform`. It reports task kind
`chi-residue-inverse-transform`, unique residue keys, representative
multiplicities and weight sums, local and per-rank task ownership, character
mesh and labels, representative scope, ordered-key and construction
fingerprints, and whether execution was multi-rank.

This is genuine χ residue scheduling, but only for SCF density diagnostics.
It does not distribute J/K, XC, diagonalization, localization, QVF/property
transforms, post-HF contractions, or gradients. It is not Γ-CCM WSC farming,
does not change the Coulomb or exchange-q=0 convention, and does not imply a
DLPNO distance truncation.
Use `progress=False` and keep artifact writing on rank zero in hand-written
MPI scripts until the shared output layer has an explicit rank-ownership
contract.

The bundled campaign has two narrow exceptions to writing that plumbing by
hand. `bash run.sh --d114-mpi` validates RHF, while
`bash run.sh --d116-rks-mpi` validates PBE RKS. Both perform the host probe
once before launch, require `mpi4py`, start exactly two ranks, and compare a
replicated unfarmed BIPOLE calculation with the farmed χ selector on
c-diamond. The D116 profile fixes STO-3G, a `(2,2,2)` mesh, 15-bohr cutoffs,
one SCF cycle, zero Fock mixing, and symmetry off. Every rank must reproduce
the D114 RHF energy, Fock matrices, and density blocks exactly. D116 uses the
versioned `d116-rks-mpi-validation/v2` acceptance contract: absolute total,
electronic, nuclear, and XC energy differences must be at most `1e-10` Ha,
and maximum elementwise Fock and density differences must be at most `1e-12`.
Array shapes, convergence and iteration state, executed zero Fock mixing, D83
exchange-q=0 applicability `inactive`, task census and canonical ordering,
launcher settings, and source/core identities still agree exactly. Its JSON
keeps `exact_parity` as an observation and uses `contract_parity` as the v2
pass condition. Existing v1 D116 files retain their original exact meaning.
Only rank zero writes the milestone-specific JSON using an atomic replace. These
records are parallel implementation-parity checks and must not be cited as
independent absolute-energy benchmarks or whole-SCF scaling results.
For hybrid execution the wrapper owns both MPI ranks and divides the total
`VQ_CPUS` allocation evenly between them. Do not pass `--scheduler-tasks`:
Torque does not enlarge the node allocation from that field, and the wrapper
rejects any nonempty `VQ_SCHEDULER_TASKS`. Intel MPI uses an OpenMP-sized
compact pinning domain; Open MPI uses an explicit per-rank processing-element
map and core binding. Unknown launchers fail closed. The artifact records that
policy, the requested
and native thread counts, thread-library caps, and the fixed Ewald/seam and SCF
controls used in the parity calculation.

Installing the optional `[mpi]` extra provides MPI capability but does not
activate it. Ordinary `python` and `import vibeqc` processes remain serial and
do not initialize `mpi4py.MPI`. A recognized launcher rank or a wrapper's
explicit required-size contract activates and binds `MPI.COMM_WORLD`;
`mpi_available()` becomes true only for a bound world with more than one rank.
The D114 wrapper requires an exact two-rank world, so
a missing MPI runtime or a launcher/communicator mismatch stops the job rather
than running duplicated serial calculations.

For RI and RIJCOSX, the high-level `run_periodic_job` surface forwards the
same fitting controls as the direct APIs: `gdf_method`, `rsgdf_ke_cutoff`,
optional `rsgdf_tail_ke_cutoff`, and `mdf_ke_cutoff`. A non-null tail is
accepted only with `gdf_method="rsgdf"`; the base must be positive and finite,
and the tail must be finite and strictly above it. The value is printed in the
`.out` file and recorded as `aiccm_resolved_rsgdf_tail_ke_cutoff`. This is a
D106 transport diagnostic, not D93 support qualification. The current fleet
schema still requires a null tail. The explicit fitted-source API below
separately records requested and executed support. SR execution clears the
inactive classic tail; explicit classic execution uses the general engine
and does not auto-select a Gamma tail.

Full-grid external XC providers have a separate, deliberately narrow χ-CCM
contract. They are accepted for 3D RKS and UKS only with
`aiccm_backend="four_center"`, capability protocol version 1, a supported
complete grid profile, and zero exact-exchange fraction. The complete
Gamma-centred finite-torus mesh is retained, the density and potential use the
explicit periodic-lattice domain, and a provider-required profile is resolved
before SCF setup. Microsoft SKALA-1.1 satisfies this provider shape through
its pinned PySCF-level-3 profile. RI and RIJCOSX external XC, every 1D/2D
request, and external-provider hybrids fail closed. This is an experimental
vibe-qc adaptation; it is not a published periodic SKALA validation result.

Global hybrids such as PBE0 remain available through all three 3D backends.
HSE06 is currently available only through 3D `four_center`, where its exchange
is the screened `0.25 K_erfc(omega=0.11)` operator and the full-range BvK seam
is inactive. D98 records screened applicability `active` and assembly
`short-range-direct` only after matching execution evidence from the direct
K-erfc branch; it does not infer either from the `hse06` name. RI and RIJCOSX
fail closed for HSE06 and every other
range-separated functional because the χ fitted-backend contract has no
B-owned screened-COSX validation or provenance. The shared generic COSX
backend can build HSE exchange, but that does not silently widen χ support;
nor do the fitted backends replace HSE06 by PBE0. The four-center HSE route is
algebraically wired and now uses the shared M5 padded screened-exchange
traversal, but HSE is not a fleet route and remains without route-specific
quantitative validation. These hybrid statements concern ordinary built-in
functionals; they do not widen the external-provider contract above.

Every χ-CCM result carries an explicit finite-torus convention descriptor. Current
production records `coulomb_kernel="3d-periodic-g0"`,
`exchange_q0="bvk-ewald"`, the boundary model, the character mesh, and the BvK
Madelung supercell. Finite-N HF, MP2, CCSD(T), and DLPNO numbers are comparable
only at the same declared convention. A strict-zero-mode exchange reference is a
different finite-N Hamiltonian with the same thermodynamic target, not a harmless
label change. If a finite solid leaves a non-rank-1 Madelung remainder after the
leading molecular-limit term is identified, treat it as finite-size physics of
the same periodic kernel, not as an RI error or an adjustable gauge.

Canonical RI-MP2 is available in 3D through
`run_aiccm2026dev_b_mp2(system, basis, lattice_extension)`. It uses B's own RI-RHF
orbitals and pair-resolved three-center tensors. It does not route through the
older Gamma-supercell post-HF helper because that helper has a different
finite HF energy and exchange q=0 convention. MP2 fails closed in 1D/2D until
their long-range gauges are matched.
The canonical MP2 implementation streams each pair-resolved AO-space `Lpq`
block directly into the occupied-virtual `Lov` factors, so it does not hold the
full AO-space pair cache at the same time as the MO-space factors.

The same declared 3D χ-CCM convention is available through
`run_aiccm2026dev_b_dlpno_mp2`, `run_aiccm2026dev_b_dlpno_ccsd`, and
`run_aiccm2026dev_b_dlpno_ccsd_t`. These routes inverse-transform the
pair-resolved RI factors to the complete real finite torus; they do not run a
different Gamma-supercell SCF. The total finite-torus correlation energy is
divided by the number of cyclic cells exactly once.

The finite-torus setup uses χ-CCM native OpenMP kernels for the one-body
matrix inverse transform, the pair-resolved RI-factor inverse transform, and
the three-index AO-to-MO contractions consumed by the local-correlation
drivers. The RI-factor kernel avoids the former six-dimensional complex
temporary. D111 also retains only the exact home-auxiliary representative
`H[P,R,S]` of the full translated factor family. On an `N_c`-cell cyclic
torus, this changes retained real AO-factor storage from
`8 A N_c^3 M^2` to `8 A N_c^2 M^2` bytes for primitive auxiliary and AO ranks
`A` and `M`. A native circulant transform reconstructs every logical
auxiliary translation while producing the same full MO-factor shape expected
by the restricted and unrestricted local solvers. The boundary requires the
complete cyclic translation group, its dual character mesh, and the common
full primitive-auxiliary frame used by the canonical factor builder; it rejects
nondual group inputs rather than applying the compression outside that
contract. Density and property residue blocks use a native finite-character
inverse Bloch transform. The
finite-translation occupied-index permutation table and occupied-pair orbit
partition used by local-correlation diagnostics also run in native
finite-group kernels, with the Python closures kept as regression oracles.
This accelerates setup and reduces retained AO-factor memory, but it is not
yet translation-representative local-correlation scaling or a universal
total-peak-memory bound.
For complete-domain local-PNO exact-limit corrections, the real-torus MP2
audit consumes the same circulant transform and streams the `L[P,i,a]`
contraction in native OpenMP code rather than reconstructing the dense AO
factor or storing the full `ijab` tensor. The momentum-pair cache, logical MO
factors, and all-pair/all-triple correlated contractions remain; D111 changes
neither the finite Hamiltonian nor any local approximation.

The planned production target is primitive-cell MgO on the `(8, 8, 8)` finite
translation group and its complete dual character mesh, with a triple-zeta
orbital basis adapted to periodic calculations.
It is **not runnable yet**. At the working frozen-core count of four active
occupied bands per cell, the current route would create 2,048 correlated
occupied orbitals and 2,098,176 placed pairs, then request global fitted-MO
tensors measured in tens to hundreds of TiB. A large-memory node cannot repair
that algorithm.

D123 provides the first allocation-safe piece of the replacement: a compact,
fingerprinted pure-translation pair plan.

```python
from vibeqc.periodic.chi.pno import build_translation_pair_plan

plan = build_translation_pair_plan(4, (8, 8, 8))
assert plan.n_placed_pairs == 2_098_176
assert plan.n_representatives == 4_112
assert plan.payload_nbytes == 197_376
assert plan.per_cell_weights.sum() == 4_098.0
```

The 4,112 rows include four home diagonals and 28 half-weight same-band
antipodal representatives required by the even mesh. The plan stores no orbit
members and no dense translation-permutation table. It currently changes no
energy path: all χ local-correlation wrappers still evaluate their historical
full workload and retain the existing screening refusals.

D126 adds a second planner-only quotient when a caller supplies a proposed
monomial occupied-space support permutation:

```python
from vibeqc.periodic.chi.pno import build_point_pair_plan

# Supplied by a future, independently validated covariance provider.
point_plan = build_point_pair_plan(plan, occupied_permutations)
assert point_plan.parent_fingerprint == plan.fingerprint
assert point_plan.n_placed_pairs == plan.n_placed_pairs
```

Each supplied occupied-index support permutation must normalize the finite
translation group; otherwise the call fails closed. The factory-only result
partitions D123 row indices, combines their exact multiplicities, and remains
immutable and fingerprinted. This validates the permutation and normalizer
algebra, not its physical provenance: orbital signs and phases are neither
carried nor fingerprinted. The planner does not materialize placed pairs and
is not consumed by an MP2 or coupled-cluster solver.

For an idealized symmetry-adapted MgO `s, p_x, p_y, p_z` occupied support
model, the permutations induced by all 48 signed-axis operations of `O_h`
reduce the 4,112 translation rows to 260 conditional work rows while
preserving the exact 2,098,176 placed-pair and 4,098 per-cell censuses. This
is not yet a symmetry speedup for a real MgO calculation. Cubic-anion
localized orbitals can be arbitrary `sp3` mixtures, so their symmetry action
need not be a scalar permutation. Casassa's procedure transforms such LWFs
a posteriori into SALWF petals, flowers, and bunches whose within-flower
representation is generally matrix-valued; a flower is not synonymous with a
dense action. The current χ localization supplies neither a monomial-gauge
seal nor that general representation, domain transformations, phases, or
amplitude scatter needed to execute the quotient.

D127 provides the next lower-level building block: a compact spatial action
on real-torus AO coefficient rows. This is a low-level χ library utility;
normal SCF calculations select `method="aiccm", variant="chi"`, but no runner
execution consumes this action.

```python
from vibeqc.periodic.chi.symmetry import (
    build_aiccm2026dev_b_real_torus_ao_action,
    build_aiccm2026dev_b_symmetry_plan,
)

symmetry = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
operation = build_aiccm2026dev_b_real_torus_ao_action(
    system, basis, symmetry, operation_index=0
)
transformed_coefficients = operation.apply(real_torus_coefficients)
```

The coefficient matrix must have `N_c * basis.nbasis` rows in the same
C-order cell blocking as χ localization. The returned matrix has the same
shape. Internally the object stores the primitive AO rotation and a linear
`(N_c, n_atom)` table of atom-dependent destination cells. It does not build
the full `(N_c * nbf)` squared action. Every numeric field is immutable, and
`fingerprint` binds the mesh, operation, AO rotation, atom reference-cell
offsets, corrected atom shifts, AO ownership, and cell map.

The cell convention is an active `R -> W R + q_a` scatter, where
`W f_a + w = f_pi(a) + q_a` for wrapped atom coordinates. If the supplied
system stores atom `a` in an equivalent cell with integer offset `n_a`, the
action uses `q_a + W n_a - n_pi(a)`; those offsets and corrected shifts are
exposed and fingerprinted.

The factory recomputes the atom-mapping residual and caps it independently at
`1e-5` bohr. Because a symmetry plan records its residual but not its input
`symprec`, a plan built with a looser `symprec` can be valid diagnostically
while this action factory rejects one of its operations. The basis-shell
origins and exact ordered radial-shell equality across mapped atoms are also
validated. Spherical `l >= 1` orbital AOs and `s` functions are supported;
Cartesian `p`-and-higher shells fail closed. The radial values themselves
are intentionally absent from the payload fingerprint.

The small dense tests reuse the trusted primitive AO rotation and
independently materialize the torus scatter/factorization. They cover proper
and improper rotations, a nonsymmorphic diamond operation with a nonzero
representative cocycle, an unwrapped atom representative, mixed odd/even
anisotropic meshes, and a skew cell. The metric check uses a synthetic
invariant metric, not the χ torus overlap. Time reversal is not a spatial
operation and is not included by this API.

The fingerprint identifies this compact action payload only. It is not a full
system, radial-basis, localized-state, run, build, or stable spglib-operation
fingerprint; operation indices are local to the plan's enumeration. Applying
the action allocates the result plus atom-block-sized temporaries. It
transforms AO rows only, not localized-orbital columns, their own reference
offsets, centres, phases, or local domains.

This utility is still diagnostic. It does not transform the current
localized occupieds into a certified group representation, and it does not
feed `build_point_pair_plan`, MP2, CCSD, or triples. The next gate must form
the occupied sewing matrices with the actual torus overlap, retain phases and
the primitive-translation cocycle, and distinguish a signed/phased monomial
gauge from a general matrix-valued petal/flower representation. D127 does not
perform Casassa's SALWF construction or classification. Only after that seal,
and after covariant local-domain maps exist, can the D126 quotient execute
physical work.

The implementation ladder is documented in
[the design note](../design_aiccm2026dev_b.md#production-target-mgo-on-the-8-x-8-x-8-finite-translation-group).
In short, production requires a covariant k-local occupied gauge,
minimum-image PAO and auxiliary domains, streamed charge-constrained local RI,
representative DLPNO-MP2, sparse representative CCSD, graph-generated TNO
triples, distributed checkpoints, and only then optional batched GPU kernels.
For rocksalt MgO, the exact point/space-group quotient layered on top of D123's
translation quotient is the primary scaling bet. It requires covariant maps
for occupied orbitals and every local domain; geometric displacement folding
alone is not a valid symmetry reduction.

Here `direct` describes streamed storage and execution, separately from the
`four_center`, RI-JK, or RIJCOSX operator-realisation choice. The intended
production reference is streamed RI-JK after the high-G fitted-route defects
#121 and #142 are closed. Four-center BIPOLE remains the small-mesh operator
oracle, and RIJCOSX is a separately converged exchange approximation. ORCA's
SHARK engine is not required and is not a vibe-qc dependency.

D100 also keeps occupied-localization diagnostics low rank. Density
invariance, one-particle-energy invariance, and translation-projector errors
are evaluated by a native kernel from occupied-space Gram matrices and
`C^H F C`, without materializing either full complex AO projector or a dense
projector--Fock product. Diagnostic scratch therefore scales as
`O(n_AO n_occ + n_occ^2)` instead of `O(n_AO^2)`. The former dense NumPy
formula remains a private parity oracle. A streamed AO-pair fallback preserves
small residuals when the fast Gram-norm identity is close to exact
cancellation, still without storing a projector. This changes only the
diagnostic evaluation: localized coefficients, objectives, PNO thresholds,
domains, and correlation energies are unchanged.

`run_aiccm2026dev_b_ccsd` and `run_aiccm2026dev_b_ccsd_t` select the
canonical-occupied, complete-domain, zero-threshold limit explicitly. They
use the same finite-torus contractions and convention descriptor as the local
route and are validation oracles, not a second Hamiltonian.

The default occupied localization is PBC-safe Pipek--Mezey. Molecular Boys
localization is rejected. The B-only `localise="wannier"` and
`localise="iao"` options feed the finite-torus localized occupieds into the
same PAO/PNO pipeline. The Wannier spread is currently a projected circular
AO-centre approximation; the IAO cross overlap is not yet periodized across
the supercell boundary. PNO truncation is available, but pair-distance
screening, finite occupied-coupling radii, and local auxiliary fitting fail
closed until their domains use minimum-image periodic distances. Use
`localise="none"`, zero PNO/domain thresholds, all pairs, and complete
occupied coupling to reach the canonical finite-torus validation limit.
At that MP2 limit, the result exposes `raw_local_e_corr_per_cell` and the
independent `complete_space_correction_per_cell` audit.

The unrestricted 3D counterparts are `run_aiccm2026dev_b_ump2`,
`run_aiccm2026dev_b_uccsd_t`, `run_aiccm2026dev_b_dlpno_ump2`, and
`run_aiccm2026dev_b_dlpno_uccsd_t`. Alpha and beta occupied projectors are
localized independently. The full-domain UCCSD(T) implementation is the
explicitly cost-capped O(N^6) correctness oracle from the DLPNO stack, not a
claim of production reduced scaling. The truncated route uses PNO subspaces,
but representative-only pair propagation remains disabled.

The finite-torus local correlation wrappers keep their historical all-electron
`n_frozen=0` defaults. With custom solver options, frozen core is supported
with `localise="none"`. The complete-space DLPNO-MP2 correction uses the
solver's resolved active occupied space, while PAO virtuals remain orthogonal
to all occupied orbitals, including frozen core. Explicit integer counts refer
to the full torus, and published chemical-core selectors resolve on that
full-torus molecule in the solver.

Nonzero frozen core with Wannier, IAO, or Pipek-Mezey localization is rejected
before electronic work. These localizers currently rotate the entire occupied
space, so removing the first orbitals afterwards would freeze a different
projector. This guard covers restricted and unrestricted local MP2 and
CCSD(T). Published selectors with a zero core count remain compatible with
localization. The restriction does not establish physical multi-k validation
or production readiness of the experimental chi route.

Every canonical and DLPNO post-HF entry point requires
`lattice_cutoff_bohr`, `rsgdf_ke_cutoff`, and
`gdf_linear_dep_threshold` to be positive finite real numbers. Invalid values,
including booleans, NaN, and infinities, fail before the RI-RHF or RI-UHF
reference calculation starts. These are numerical-support controls for the
same finite-character Hamiltonian; accepting a different value does not name
a different χ construction or Coulomb convention.

`derive_aiccm2026dev_b_scf_properties` reports electron and spin counts,
Mulliken charge/spin populations, finite-net band gaps, idempotency, and
spin contamination. The returned object carries the same finite-torus
convention descriptor as the parent SCF result. `aiccm2026dev_b_band_structure`
stores the descriptor in `bands.metadata["finite_torus_convention"]`, and
`aiccm2026dev_b_mayer_bond_orders` does the same for the primitive-cell Mayer
table, so direct Python analyses and the B-owned JSON summaries cannot be
mistaken for a strict-zero-mode or isolated-wire/slab Hamiltonian.

The high-level runner uses that same finite-torus property path for population
sidecars. On every supported character mesh,
`write_population_file=True` writes `.population.txt` and `.population.json`;
the capability-aware default does so as well. These files are built from the
full character-resolved density and overlap blocks. They are not a molecular
analysis of only the Gamma block.

Periodic QVF output from
`run_periodic_job(..., jk_method="aiccm2026dev-b", output_qvf=True)` emits the
visual `structure.pbc`, `structure.lattice_vectors`, `volume.density`, and
restricted `volume.orbital` sections over the full BvK torus cell. For mesh
sizes larger than `(1, 1, 1)`, the writer builds the visual supercell, folds
the finite-character density into the matching AO density matrix, and emits
Gamma-character HOMO/LUMO orbital grids in that same cell for restricted
records. Unrestricted records emit the spin-summed density grid and, with
`qvf_wannier_centers=True`, alpha/beta Wannier-centre overlays; spin-resolved
orbital grids remain open. The archive provenance preserves the represented
spin polarization: for a primitive multiplicity \(M\) repeated over \(N_c\)
torus cells, the visual BvK system records multiplicity
\(N_c(M-1)+1\). The root provenance describes that visual system; the
archive's `job.spec` separately records the primitive requested charge and
multiplicity. Restricted singlets therefore remain singlets. A shared
periodic-output update additionally emits
`wavefunction.gto` from the Gamma character, or the k=0 block of a
multi-character result. That raw primitive-cell coefficient payload is
additive; it does not replace the torus-periodic density/orbital grids or the
cyclic AO image sums used to sample them. The restart-only
`x_vibeqc.bloch_wavefunction` payload remains suppressed for χ. The grid
sampler applies AO image sums on the declared periodic axes and leaves vacuum
axes unwrapped. The archive also carries the first-party
`x_vibeqc.aiccm2026dev_b_convention` vendor JSON section, advertised by the
root `x_vibeqc` extension marker, with the method selector, mesh, backend, and
finite-torus convention descriptor used for the SCF result.

The generic periodic QVF DOS and bonding-property rebuild is deliberately
inactive for χ-CCM. That shared path constructs a fixed-cutoff Ewald/HF-like
lattice operator from the converged density; it does not reuse the executed
finite-character J/K assembly and, for KS jobs, omits the XC potential.
Labelling its DOS, projected DOS, COOP, COHP, or Mayer arrays as properties of
the χ Hamiltonian would therefore be wrong. An explicit `coop_cohp=True`
request with `jk_method="aiccm2026dev-b"` fails before SCF until a route-owned
implementation consumes the converged finite-character Hamiltonian. The
B-owned population sidecar and Mayer analysis, direct Python band/property
APIs, torus density/orbital grids, convention payload, and optional Wannier
overlay remain available; no generic surrogate property section is inserted
into the archive.

Periodic cell dipoles, Berry-phase polarization, analytic response, and
correlated properties without a relaxed correlated one-particle density fail
closed.

## QVF visualization fixtures

`run_periodic_job(..., jk_method="aiccm2026dev-b", output_qvf=True)` writes
vibe-view-ready periodic QVF archives. For finite-torus visualization the
archive stores the full BvK display cell in `structure.lattice_vectors` and
ships precomputed torus-aligned `volume.density` / `volume.orbital` grids.
The chi-CCM-B writer may also include `x_ccm.wannier_centers`, which vibe-view
draws as an optional overlay.

Two sanitized visualization fixtures are bundled with the docs:

The files were generated at pre-D89 source `d746d238` (shown as `d746d23` in
their archived output). They validate only the visualization pipeline. Because
they lack the normative D89 approach identity fields, they are not
construction-comparison or numerical-validation evidence.

| Case | Input | Output | QVF | vibe-view capture |
|---|---|---|---|---|
| 3D vacuum-padded H-chain, RI, `aiccm_lattice_extension=(4,1,1)` | [input](../_static/examples/chi-ccm-b-qvf/input-chi-ccm-b-hchain-ri-n4-wannier.py) | [`.out`](../_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-hchain-ri-n4-wannier.out) | [`.qvf`](../_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-hchain-ri-n4-wannier.qvf) | [Wannier overlay](../_static/examples/chi-ccm-b-qvf/hchain-wannier-overlay.png) |
| 3D H2-pair, RI, `aiccm_lattice_extension=(2,1,1)` | [input](../_static/examples/chi-ccm-b-qvf/input-chi-ccm-b-h2pair-3d-ri-n2-wannier.py) | [`.out`](../_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-h2pair-3d-ri-n2-wannier.out) | [`.qvf`](../_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-h2pair-3d-ri-n2-wannier.qvf) | [Wannier overlay](../_static/examples/chi-ccm-b-qvf/h2pair-wannier-overlay.png) |

The full capture set and sanitized `.system` manifests are summarized in the
{download}`chi-CCM-B fixture README <../_static/examples/chi-ccm-b-qvf/README.md>`.

## Gradients and forces

χ-CCM-B now exposes an explicit gradient-status surface:
`aiccm2026dev_b_gradient_status(result)` reports the finite-torus convention,
backend, lattice extension, and the still-open derivative terms for the result.
The first independently checkable component helpers return only fixed pieces
of the 3-D Ewald one-electron electrostatics for results that declare the
`3d-periodic-g0` / `bvk-ewald` convention:
`compute_aiccm2026dev_b_ewald_nuclear_gradient(system, result)` differentiates
the nuclear-repulsion term, while
`compute_aiccm2026dev_b_ewald_electron_nuclear_gradient(...)` differentiates
`sum_g Tr[D(g) V_ne(g)]` at a caller-supplied fixed real-torus density and
lattice cutoff. The convenience bundle
`compute_aiccm2026dev_b_ewald_electrostatic_gradient_components(...)` returns
those two arrays plus their fixed-density sum, with the Ewald alpha and cutoffs
recorded in the returned component object.
`compute_aiccm2026dev_b_ewald_electrostatic_energy_components(...)` returns
the matching fixed-density scalar energy pieces for central-difference audits
of that component bundle. For the explicit fixed-density electron-nuclear and
electrostatic bundle helpers, the supplied density `LatticeMatrixSet.cells`
must exactly match the lattice-cell list implied by `lattice_options`, its AO
dimension must match the operator template, and it must carry one correctly
shaped AO block per cell; mismatches fail closed rather than mixing cutoffs
or AO spaces in the audit pair.

The kinetic part is available as a second explicit fixed-density audit pair.
`compute_aiccm2026dev_b_fixed_density_kinetic_energy(...)` evaluates
`sum_g Tr[D(g) T(g)]`, while
`compute_aiccm2026dev_b_fixed_density_kinetic_gradient(...)` returns its
analytic AO-centre derivative with the supplied real-torus density blocks held
fixed. The same exact cell-index, Cartesian-translation, AO-dimension, and
block-shape guards apply.
The derivative is tested against central differences of the matching scalar on
displaced bases, so this is an independently checkable component rather than
an inferred total force.

The overlap constraint is available as a third explicit audit pair.
`compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian(...)`
evaluates
`-sum_g,mu,nu W_mu,nu(g) S_mu,nu(g)` for a caller-supplied real-torus
energy-weighted density, while
`compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient(...)`
returns its analytic AO-centre derivative with the numerical `W(g)` blocks
and lattice held fixed. The scalar is a Lagrangian companion, not a separately
additive electronic energy. The same exact support guards apply, and the test
uses real but individually nonsymmetric residue blocks to pin the Frobenius
orientation and sign against central differences.

The restricted fixed-input algebra feeding that overlap audit is available as
`compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(...)`.
It accepts explicit \(S(q)\), occupied \(C_o(q)\), and candidate variational
\(F(q)\) blocks, checks them against the complete character net and the exact
overlap support selected by `lattice_options`, and evaluates
`Lambda(q) = C_o(q)^H F(q) C_o(q)` followed by
`W(q) = 2 C_o(q) Lambda(q) C_o(q)^H`. The returned real-space
`LatticeMatrixSet` can be passed directly to the D80 scalar and derivative.
Hermiticity, occupied overlap orthonormality, occupied-unitary covariance,
time reversal, the inverse-transform sign, and the real residue are pinned by
independent tests on even, odd, and multi-axis meshes. Nondual or unbounded
character coordinates, nonfinite derived algebra, broken occupied-projector
sewing, and a non-real inverse-transform residue fail closed.

This helper deliberately says `fixed-input`. It does not take stored orbital
energies, infer occupations, rebuild the final unmixed and unshifted physical
Fock, prove stationarity or backend variational closure, or attest the object
as belonging to the executed SCF state. It is not an SCF Pulay wrapper and
does not enable total gradients.

The unrestricted companion is
`compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(...)`.
It accepts the common explicit \(S(q)\) plus separate alpha/beta occupied
coefficients and candidate variational Focks. The coefficient columns are
unit-occupied, zero-temperature integer-occupation columns; fractional
occupations are outside this audit. For each spin it evaluates
`Lambda_s(q) = C_o,s(q)^H F_s(q) C_o,s(q)` and
`W_s(q) = C_o,s(q) Lambda_s(q) C_o,s(q)^H = D_s(q) F_s(q) D_s(q)`, then
inverse-transforms `W_alpha(q) + W_beta(q)` onto the same exact D80 support.
There is no factor of two in one spin block and no alpha-beta cross term.

Alpha and beta ranks may differ, but each is constant across the complete
character net. Either one may be zero through explicit zero-column occupied
blocks, while both zero fail closed. The candidate Fock for a zero-rank spin
is still explicit and must pass finite, Hermitian, and per-spin sewing checks.
Overlap orthonormality, occupied projectors, candidate Focks, and \(W_s\) are
checked separately in each spin. Each spin's inverse-character imaginary
residue is also checked before the sum, so spin errors cannot cancel.
Independent occupied rotations leave the result unchanged.

For `C_alpha=C_beta=C` and `F_alpha=F_beta=F`, the unrestricted construction
reduces exactly to the restricted one:
`W_alpha + W_beta = 2 C Lambda C^H = D F D / 2`, with the restricted
spin-summed density `D=2 C C^H`. The summed real-space object can be passed
directly to the D80 scalar and derivative with its numerical blocks fixed.

Like the restricted helper, this is not an SCF Pulay wrapper. It does not
infer occupations, read stored orbital energies or densities, rebuild or
attest the final physical spin Focks, prove stationarity or backend closure,
or bind the inputs to an executed SCF state. D85 binding and all remaining
explicit exchange-kernel, RI, XC, and response derivatives are still needed
before any total gradient can be exposed.

The restricted active BvK exchange seam is available as a fourth fixed-input
audit pair.
`compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(...)`
evaluates
`-eta/4 sum_q w_q Tr[D(q) S(q) D(q) S(q)]` for a caller-supplied
spin-summed restricted character density. The matching `..._gradient(...)`
helper evaluates its analytic AO-centre derivative at fixed density, eta, and
primitive lattice. It requires an active restricted RHF/RKS 3D result, the
complete unreduced uniform unshifted character net, Hermitian and
time-reversal-consistent density blocks, and a positive finite explicit
`operator_coefficient_eta`. The result records only whether the full-range
seam is active, while D95 separately records the route-resolved hybrid fraction but
does not bind the numeric eta, density, basis, or overlap support. The helper
therefore does not infer that coefficient from the descriptor or a method
name. Pure and HSE records with
inactive applicability refuse the seam component rather than returning a zero
that could be mistaken for a completed screened-exchange derivative. Negative
full-range fractions are outside this first component contract. The
public scalar and gradient are checked on an odd three-character skew-cell
projector, including the factor of two from differentiating both overlap
factors, transform orientation, eta linearity, translational sum, and central
differences.

The unrestricted active BvK exchange seam is available as a fifth fixed-input
audit pair.
`compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(...)`
evaluates
`-eta/2 sum_q w_q sum_s Tr[D_s(q) S(q) D_s(q) S(q)]` from separately
supplied alpha and beta character densities. The matching `..._gradient(...)`
helper evaluates
`-eta sum_q w_q sum_s Tr[D_s(q) S(q) D_s(q) dS(q)]` by
inverse-transforming
`eta [D_alpha S D_alpha + D_beta S D_beta]` onto the overlap support. It
requires an active 3D UHF result or a nonempty `UKS/<functional>` result, the
complete unreduced uniform unshifted character net, independently finite,
Hermitian, and time-reversal-consistent alpha and beta blocks, and a positive
finite explicit `operator_coefficient_eta`. A zero beta channel is accepted.
The spin blocks are contracted separately, so no alpha-beta exchange term is
introduced. With `D_alpha=D_beta=D/2`, the scalar, derivative, and
inverse-transformed overlap weight reduce exactly to the restricted D92
helpers.

Active `exchange_q0_applicability` declares only that the route resolves a
nonzero full-range exact-exchange arm. D95 records that arm's numerical
fraction, but does not bind the caller's eta or remaining state/support
inputs, so the unrestricted helper does not infer eta from UHF, a UKS
functional label, applicability, or the descriptor. Restricted records fail
the method guard; production pure and HSE records carry inactive applicability
and fail the applicability guard without a functional-label reclassification.
Broken-spin and closed-shell skew-cell tests pin the separate-spin contraction, eta
linearity, translation sum, exact D92 reduction, and analytic central
difference.

There is deliberately no SCF-bound kinetic, overlap/Pulay, or seam wrapper yet.
Current SCF results do not retain the resolved one-electron lattice cutoff
needed to prove that a reconstructed kinetic operator has the same cell
support as the SCF operator. Constructing the stationary energy-weighted
density also requires the D85 binding: a rebuilt final physical variational
Fock, occupied coefficients and integer occupations, retained-space
projectors, and structure, basis, torus, backend, gauge, and numerical-support
digests. It is formed from `Lambda = C_occ^H F_var C_occ`, not accepted from
unattested stored orbital energies. The BvK exchange seam or screened-exchange
kernel contributes to that physical Fock but has a separate explicit
fixed-density derivative. Neither seam pair binds its supplied density, eta,
basis, or overlap support to the SCF result; neither may be hidden inside or
double-counted with the overlap term. The public overlap
helpers therefore prove only the fixed-`W` skeleton.
Callers doing a derivation audit must therefore pass the fixed density or
energy-weighted density and the known matching `lattice_options` explicitly
instead of relying on a default cutoff.
All component helpers now also bind the result to the active finite torus:
the result mesh, character mesh, and BvK repetitions must agree, and the
recorded column-vector BvK lattice must equal
`system.lattice @ diag(mesh)` within `1e-12` bohr. SCF-density folding accepts
only the complete unreduced, unshifted Γ-centred character net with uniform
weights. This guard does not attest atoms, basis identity, operator cutoffs,
or the origin of caller-supplied density blocks.
`compute_aiccm2026dev_b_scf_density_lattice(...)` inverse-Bloch folds the
stored SCF k-density onto the same lattice-cell list as
`compute_overlap_lattice`, so a converged B result can feed those
fixed-density component helpers without hand-written density reconstruction.
For restricted records, the fold first checks for singlet multiplicity and a
non-negative even effective electron count. For unrestricted records, it checks
that the effective electron count and multiplicity imply non-negative integer
alpha/beta occupations. The stored k-density blocks must also match the active
AO basis dimension before they are inverse-Bloch folded.
`compute_aiccm2026dev_b_scf_ewald_electrostatic_energy_components(...)`
performs that fold and returns the same fixed-density electrostatic energy
pieces at the stored SCF density.
`compute_aiccm2026dev_b_scf_ewald_electrostatic_gradient_components(...)`
performs that fold and returns the nuclear, electron-nuclear, and
fixed-density electrostatic sum at the stored SCF density.
These helpers are useful for derivation tests, but they are not total forces.
`compute_aiccm2026dev_b_gradient(result, ...)` and
`run_aiccm2026dev_b_gradient(result, ...)` fail closed with a
`NotImplementedError` until the derivative of the declared χ-CCM Hamiltonian is
derived and validated. D95 adds coefficient provenance but does not add D85
state binding, low-level operator attestation, screened exchange, density or
orbital response, RI/RIJCOSX three-center or metric response, or a
total-gradient wrapper.
Γ-CCM and χ-CCM remain distinct approaches compared at a declared common
exchange-q=0 convention; their names do not select different Coulomb kernels.

This guard is intentional. The sibling union-and-weight Γ-CCM gradient
differentiates its direct-torus WSSC molecular-kernel energy. A separate
neutral fitted-torus real-Gamma/GDF representation-control gradient also does
not acquire χ construction identity from its representation or shared
primitives. Production χ-CCM-B numbers declare
`coulomb_kernel="3d-periodic-g0"` and
`exchange_q0="bvk-ewald"`, so a χ-CCM-B force must include the derivative of
that finite-character Hamiltonian, including the BvK exchange seam and the
RI/RIJCOSX metric and three-center response when those backends are selected.
RKS and UKS additionally require the exchange-correlation quadrature and grid
derivative.
For HSE06 four-center records, the corresponding missing exchange term is the
screened erfc-kernel derivative rather than a full-range seam contribution.
Substituting either non-B-qualified gradient would silently change or leave
unattested the Hamiltonian behind the forces. Geometry optimizers should
therefore treat χ-CCM-B analytic forces as not implemented rather than falling
back to a sibling or representation-control route. The high-level
`run_periodic_job(..., jk_method="aiccm2026dev-b")` surface rejects
`optimize=True` and `hessian=True` for the same reason.

Space-group analysis is opt in with `symmetry_mode="diagnostic"` on the
direct APIs or `aiccm_symmetry="diagnostic"` on `run_periodic_job`. It
attaches the spglib group, exact cluster-compatible subgroup, atom/cell maps,
and irreducible k orbits without changing the SCF build. The `integrals` mode
fails closed until general-k sewing matrices and shell-quartet scatter pass
energy and Fock parity.

## Electron-repulsion backends

| Backend | Coulomb | Exact exchange | Current purpose |
|---|---|---|---|
| `four_center` | direct periodic four-center build, 3D only | direct periodic four-center build, 3D only | needs qualified D93, D103, and fresh D104 support |
| `ri` | pair-resolved periodic three-center fit, 3D only | pair-resolved fitted exchange, 3D only | scalable route; D93 holds absolute values |
| `rijcosx` | same RI-J, 3D only | chain-of-spheres exchange, 3D only | trial route; D93 holds absolute values |

`aux_basis` is meaningful only for `ri` and `rijcosx`. The `four_center`
operator is complete and contains no density fit, so an explicit auxiliary
basis raises before SCF rather than being silently ignored. Omitting the
keyword keeps the direct route unchanged.

For 3D `four_center` jobs, the neutral Ewald J split resolves the nuclear
real-space cutoff no larger than the electronic J/K cutoff. With library
defaults, both therefore resolve to 15 bohr rather than the raw 15/25 bohr
option pair. Benchmark JSON records expose the actual values in
`direct_lattice_cutoffs`; RI and RIJCOSX records set that field to `null`.
The χ wrappers explicitly set `use_multipole_far_field=False`. This matches
the shared BIPOLE contract: the quartet far-field prototype is unavailable,
and explicit `True` raises before setup because the prototype does not preserve
the exact three-translation Fock domain. This pin changes no Coulomb convention
and applies to RHF, RKS, UHF, and UKS.
That pair describes coherent physical density/nuclear support. The larger
internal ket-image traversal for smeared AO-pair charges is now supplied by
the shared M4b/M5 erfc builder and returned as `sr_image_extent_bohr`.
With the shared BIPOLE physical default, this radius bounds pair-midpoint
separation; the historical QQR policy instead uses an absolute ket-image
radius. Chi labels the policy from the producer's immutable `resolved_domain`,
and emits no domain attestation when that record is absent or its extent
disagrees with the reported radius. Caller options do not identify the
executed domain because the BIPOLE driver resolves its default without
mutating those options. Current D93 fleet acceptance admits only the
historical QQR policy, so physical-domain results remain `not-qualified`.
Earlier unpadded records remain revision-bound. Since the
2026-07-18 exact-FT retirement, pure restricted semilocal RKS also routes its
Hartree J through the padded SR+LR composition; the finite-KE analytic-FT
partial sum remains only as the explicit ``use_exact_ft_j`` oracle. Records
from symmetry-enabled pair-domain runs before `2fd23eff` are separately
revision-bound because their stored SCF density was pair-masked globally.

The separate D103 overlap-fold record binds the maximum drift over the
executed character mesh to the same electronic cutoff. Its versioned formula
is the maximum absolute element of the base-cutoff versus `1.5`-times-cutoff
Bloch-fold difference, maximized over that mesh. A value above `1e-2` fails
before Fock construction; `1e-4 < drift <= 1e-2` remains executable but not
qualified; `drift <= 1e-4` satisfies this support component. Passing D103 does
not replace D93 or the independent D104 shared-Ewald contract.

RI and RIJCOSX result objects can now record an explicitly transported RSGDF
tail, but current D93 fleet records still require null because there is no
accepted B-owned tail-convergence attestation. They serialize the requested
RSGDF or MDF base support, runtime backend, and absence of an accepted tail or
COSX-domain contract. Pre-D106 Gamma RHF/RI records that inherited the shared
automatic tail are revision-bound. Current execution records distinguish
requested support from the selected engine and its executed tail. Under D93,
all fitted and all post-HF chi results remain `not-qualified` permanently in
v1, including results that bind a common SCF/correlation factor source. D107 separately rejects
either shift form on Gamma-only RHF/RI: a static shift would leave the fitted
periodic operator, while a scheduled shift would not execute. D108 rejects the
same ignored request on restricted four-center routes whenever the schedule is
nonempty. D110 separately lowers an empty-schedule static-shift warm-up into
the BIPOLE schedule and records its resolved executed length. The qualification
gate is unchanged: `audit_b.py` and `compare_b.py` reject unqualified rows
before an absolute energy can enter a table.

RI, RIJCOSX, and four-center currently run only for 3D cells. Lower-dimensional
χ-CCM-B SCF fails closed until the neutral wire/slab Green function and the
matching RI/GDF support are implemented and anchor-tested. The derived
character net is Γ-centred. RIJCOSX currently requires at least two cyclic
cells. Post-HF remains 3D-only until the lower-dimensional long-range gauges
are matched.

## Dimensional Coulomb conventions

The finite translation-group identity is dimension independent. The Coulomb
kernel is not. The current 3D four-center route uses the neutral Ewald J split
and the Ewald exact-exchange finite-size correction. The 1D and 2D direct
four-center fallback is a direct-truncated active-lattice kernel, not the
neutral wire/slab finite-torus Green function. The lower-dimensional neutral-RI
mesh is also not a Coulomb kernel because it pins every transverse reciprocal
component at \(G_\perp=0\). `aiccm2026dev-b` therefore blocks all
lower-dimensional SCF backends before returning an absolute energy.

Consequently, lower-dimensional χ-CCM-B is not an absolute-energy parity
backend today. Matched 1D wire and 2D slab Green functions remain an acceptance
item. The code does not disguise this mismatch with damping or a fitted offset.

Future isolated low-dimensional support needs a shared mixed-boundary kernel:
a 2D slab Green function with isolated transverse boundary, a 1D wire Green
function with isolated transverse boundary, and matching exchange q=0 and
self-potential conventions derived from that same kernel. The 3D
`exchange_q0="bvk-ewald"` seam must not be imported into those routes as if it
were universal.

## What is checked on every result

- Wigner--Seitz representative weights sum to one for every translation class.
- A restricted density obeys $DSD=2D$; unrestricted spin densities obey
  $D^\sigma S D^\sigma=D^\sigma$ independently.
- The weighted total and spin-resolved electron counts match the cell.
- The inverse Bloch transform has a negligible imaginary residual for the
  Γ-centred character mesh.

These checks establish internal consistency, not thermodynamic-limit accuracy.
Converge the cyclic lattice extension and compare against a matched reciprocal-space
calculation before using a number quantitatively.

## Separate examples and theory

- [Worked χ-CCM tutorial](../tutorial/aiccm2026dev_b.md)
- [χ-CCM occupied localization](../tutorial/aiccm2026dev_b_localization.md)
- [χ-CCM symmetry diagnostics](../experimental/aiccm2026dev_b_symmetry.md)
- [`examples/periodic/aiccm2026dev_b_demo.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_b_demo.py)
- [`examples/periodic/aiccm2026dev_b_mp2.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_b_mp2.py)
- [`examples/periodic/aiccm2026dev_b_local_correlation.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_b_local_correlation.py)
- [`examples/regression/benchmark_aiccm2026dev_b.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/benchmark_aiccm2026dev_b.py)
- [Complete χ-CCM fleet inputs](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README_B.md)
- [Pinned 1D/2D/3D route-control campaign](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/COMPARISON_1D2D3D_2026-07-15.md)
- [Independent derivation](../design_aiccm2026dev_b.md)
- [Decisions and open questions](../aiccm2026dev_b_decisions.md)

The [Archived comparative manuscript](https://vibe-qc.com/docs/)
is the detailed theory record. It includes the historical ab-initio weighting,
the Janetzko--Köster--Salahub deMon2k KS-ADFT construction, both 2026
development streams, and the reciprocal-space reference formulation.

## Explicit fitted source

`run_periodic_job(method="aiccm", variant="chi", aiccm_backend="ri",
fit_source="classic")` selects the classic shared-q FFT fit. The same selector
is accepted by the χ RHF, RKS, UHF and UKS library drivers, with `backend="ri"`
or `"rijcosx"`. `fit_source="auto"` preserves the existing default, including
the 3D SR/LR source. `fit_source="sr"` selects that source explicitly.
This is a choice of fitted operator: classic and SR energies are separate
baselines. Neither source is numerically qualified. D93 permanently keeps all
fitted χ and all χ post-HF results **not-qualified** in v1.
Their result objects expose `quantitative_status="not-qualified"`.

Explicit source selection requires 3D RSGDF. Missing values such as `None`,
unknown names, four-center/MDF combinations and private fit adapters are
refused. An analytic gradient of the explicit classic fit is available with
`classic_support_policy="historical_index_ball"`, the support the neutral
Gamma-CCM producers pin; on the physical-pair support it is refused (#404).
The generic GDF front-door defaults are unchanged.
For χ reference campaigns, the front door accepts `conv_tol_energy=1e-11` and
`conv_tol_grad=1e-9`; omitting the latter preserves its existing orbital tolerance.
An energy tolerance alone does not establish orbital convergence.

The explicit classic profile retains canonical modrho auxiliary factors from
SCF. The χ MP2 and local/canonical CC library drivers accept
`fit_source="classic"` and consume those same factors. Set
`lattice_cutoff_bohr` equal to `options.lattice_opts.cutoff_bohr`; inconsistent
support is refused. `gdf_linear_dep_threshold` and an explicit
`rsgdf_tail_ke_cutoff` reach the SCF source. The independent pair-by-pair canonical
builder remains available internally as an operator oracle. Canonical storage
can be larger than the reduced eigenmode frame. The explicit classic result
retains its full AO-pair cache through correlation and for the result's lifetime;
this trades memory for reuse compared with independently streamed factors.
This profile preserves the
current independently resolved one-electron domain, including its S/T image-tail
bound, the BvK q=0 exchange seam, hybrid scaling and primitive-cell normalization.

The historical post-HF `auto` behavior still rebuilds its independent classic
fit after the default SCF. Its `executed_fit_source` identifies that independent
classic shared-q engine, while `reference_fit_source` identifies SCF separately.
Both independent classic post-HF consumers group all ordered pairs by the
shared canonical reciprocal-transfer helper. They use bounded batches without
symmetry reduction. The internal `batch_memory_bytes` control defaults to
64 MiB for three batch-dependent three-center buffers per admitted pair
(accumulation, canonical factors, and transform workspace), with a one-pair
minimum. The shared engine separately bounds reciprocal chunks and its
auxiliary Fourier cache; this control is not a limit on total process memory.
The AO consumer retains the complete cache. The streamed consumer transforms
each batch to occupied-virtual factors and releases its AO factors before the
next batch. Returned caches keep deterministic row-major pair order.
The unchanged `_build_canonical_lpq_cache_pairwise` and
`_build_canonical_lov_cache_pairwise` remain internal numerical oracles.
Chunk widths can change factor-element rounding; operator and energy identity
must be checked separately. Cutoffs, metric thresholds, canonical auxiliary
frame, source selection and D93 status are unchanged.

The streaming auto MP2 descriptor labels its emitted factors `occupied-virtual`;
their hash identifies that projection. Full retained factors are labelled
`ao-pair`. Explicit `sr` post-HF is refused
until a compatible canonical SR frame is implemented and checked. The χ
correlation front door remains library-only.

Results expose an immutable `executed_fit_source` descriptor and a separate
`rsgdf_fit_request`. The descriptor records engine/version, normalized orbital
and auxiliary fingerprints, resolved numerical support, metric threshold,
auxiliary frame, momentum-transfer grouping, k points, reciprocal cutoff,
executed tail and factor digest. `process_provenance` identifies the imported
release, source revision and SHA-256 of the loaded native extension.
The `.system` output includes executed source metadata, requested settings,
typed mesh/lattice arrays and the loaded native hash. An absent executed tail is
recorded separately from the requested tail. The SR engine does not execute the
classic reciprocal tail, and Γ admission does not consult that inactive policy.
An explicit classic Γ call uses the general finite-character engine with current
one-electron support; it does not inherit a legacy Γ auto-tail.

The alternating H4 two-cell regression separates historical energy from current
routing. Its original `-2.1648043358` Ha/cell pin belongs to classic fitting
on the historical index ball with a flat 15-bohr one-electron domain,
reproduced explicitly through `run_krhf_periodic_gdf(..., fit_source="classic",
classic_support_policy="historical_index_ball", rcut_strategy="flat")`.
That finite-support value is not a reference for the current χ default, which
uses the SR/LR fit and independently grows the one-electron domain. The current
χ test checks energy, AO matrices and executed source metadata against the
shared engine with matching settings, and retains the coarse CCM comparisons.
These routing checks do not provide independent absolute-energy acceptance or
change the fitted source's `not-qualified` status.


### Reusing an accepted restricted reference (experimental)

`vibeqc.periodic.chi.reference.prepare_chi_reference` builds a process-local
`ChiAcceptedReference` for several restricted library consumers. Pass it as
`reference=` to canonical MP2, DLPNO-MP2, CCSD/CCSD(T), or their DLPNO variants:

```python
from vibeqc import PeriodicRHFOptions
from vibeqc.periodic.chi.reference import prepare_chi_reference
from vibeqc.periodic.chi.posthf import (
    run_aiccm2026dev_b_mp2, run_aiccm2026dev_b_dlpno_mp2,
)

options = PeriodicRHFOptions()
options.conv_tol_energy = 1e-11
options.conv_tol_grad = 1e-9
controls = dict(fit_source="classic", rsgdf_ke_cutoff=200.0,
                lattice_cutoff_bohr=15.0, gdf_linear_dep_threshold=1e-9)
reference = prepare_chi_reference(system, basis, (2, 1, 1), options, **controls)
mp2 = run_aiccm2026dev_b_mp2(
    system, basis, (2, 1, 1), options, reference=reference, **controls)
local_mp2 = run_aiccm2026dev_b_dlpno_mp2(
    system, basis, (2, 1, 1), options, reference=reference, **controls)
```

The consumer must request the same geometry, orbital and auxiliary bases,
mesh, source, tail, and numerical support. Explicit SCF options must match;
omitting them with a supplied reference accepts its recorded SCF controls.
The builder's default SCF tolerances are energy `1e-11` and commutator `1e-9`.
It checks achieved energy change, accepted-density and orbital-density
commutators, `FC-SCε`, metric orthogonality and density agreement before
admission. Changed states, factor arrays, descriptors or process build
identities fail closed. Raw SCF result objects are not accepted as transport.

`auto` retains the existing SR SCF and separate classic correlation source.
`classic` borrows the already canonical, attested SCF factors; no auxiliary
frame conversion or replacement of an SR fit by a classic fit is performed.
The reference stores full character AO-pair factors and the existing home-cell
auxiliary representation, without constructing an Nc-fold real auxiliary
copy. Retaining factors can use more memory than a single cold streamed MP2
call. `reference.build_stages` and each result's `reference_execution` identify
built/reused stages and their sources. Caller-owned factor mappings are not
cleared. Omit `reference` for the independent cold path.

The object carries separate SCF/correlation fit descriptors, requested and
executed tails, actual one-electron support, finite-torus and exchange
conventions, orbital residuals and the process release/source/native-core
triple. It is process-local and does not provide disk or distributed caching.
Unrestricted, fractional-occupation, ECP and non-RHF references are unsupported.
RIJCOSX preserves its iterated Fock/orbitals as `iterated_fock`,
`iterated_mo_coeffs` and `iterated_mo_energies`, alongside its corrected returned
Fock/orbitals. Its `energy_operator` is `iterated`, while `returned_operator`
is `post-convergence-one-center-corrected`. The energy and accepted density
retain their existing meaning. A small iterated commutator cannot attest the
corrected state, and this transport refuses it as a stationary correlation
reference.

All fitted and post-HF results remain **not-qualified** (D93). The correlation
front door, gradients, numerical support and truncation defaults are unchanged.
At zero cuts, inspect both `raw_local_e_corr_per_cell` and the returned MP2
energy: the latter includes the existing canonical complete-space correction.

### Optional SR q-batch MPI execution

The admitted SR fit used by `run_periodic_job(method="aiccm", variant="chi",
aiccm_backend="ri", fit_source="sr")` can distribute its complete q groups.
Set `VIBEQC_SR_Q_MPI=1` under an MPI launcher; the default is `0`, including
when MPI is installed. Default serial helper calls remain noncollective,
including calls made on only one rank. The optional `[mpi]` extra supplies
`mpi4py`. This switch changes ownership of the existing SR integral batches. It preserves
the selected fit, auxiliary eigenmode frame, ordered bra pairs, metric
threshold, screening and one-electron support. It does not enable public SR
post-HF support or change the permanent D93 **not-qualified** status.

For a total budget of eight OpenMP threads, a two-rank invocation is:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 \
VIBEQC_SR_Q_MPI=1 VIBEQC_SR_Q_TOTAL_THREADS=8 \
mpiexec -n 2 python calculation.py
```

Choose launcher binding appropriate to the allocation. Each rank must call
the same calculation, with the same source and numerical controls. Give
each rank a separate output stem when using the front door. The source
builder collectively checks its SCF memory preflight, resolved domain,
loaded release/source/native-core identities and thread requests before
partitioning. A caught preparation or native-build error is reported on all
ranks before ordered gather. Ranks owning no q groups still participate.
Private MDF and symmetry pair-cache adapters are outside this SR loop and
are refused when this switch is enabled.

This mode replicates the complete gathered factor cache and subsequent SCF
state on every rank. Its memory admission reserves local retention,
serialization and gathered-copy overlap, plus one native workspace per rank.
SCF admission divides available memory conservatively across the world
before subtracting each rank's retained SCF and one-electron state. It is not
a distributed-memory SCF implementation. Increasing rank count can increase
memory use without increasing throughput, particularly beyond the q count.

`result.rsgdf_q_execution` and `run.rsgdf_q_execution` in the `.system` file
report world size, q ownership, per-rank task counts, actual local build and
ordered-gather intervals, retained/gathered memory, process provenance and
the loaded thread pools when available. Timers report both `perf_counter`
and realtime clocks with load samples. Differences between clocks and gaps
in external monitoring require investigation before treating an interval
as an end-to-end wall bound. Task counts alone are not a scaling result.

For repeated runs on the same source and domain, set
`VIBEQC_SR_Q_COST_PROFILE` to an external JSON file containing
`domain_sha256` from the execution record and `q_build_seconds`, a list of
measured build times in its ordered q sequence. The scheduler assigns the
largest measured costs first to the least loaded rank using
[Graham's longest-processing-time rule](https://epubs.siam.org/doi/10.1137/0117039),
then restores the original q and pair ordering. All ranks must read the same
profile; a domain mismatch is refused. Keep the originating process provenance and timing
records with that profile. Without a profile the existing block partition is
used. Serial execution remains the default; a measured benefit depends on
integral costs, gathering and host contention.

### Finite-character transform input contract

`inverse_bloch_transform(M, k, R, weights=None)` requires finite nonempty
square matrix blocks `(n_k, nbf, nbf)`, finite real fractional coordinates
`(n_k, 3)`, and integral translations `(n_R, 3)` in signed 64-bit range.
Integral real translations are accepted; fractional, nonfinite, boolean and
out-of-range values are refused before integer conversion or native dispatch.
Weights have shape `(n_k,)`, are finite and nonnegative, and sum to one.
Zero weights and nonuniform normalized weighted sums remain supported.

The optional keyword `extension_matrix=S` declares the complete character
group of the column-lattice extension `A_super = A S`. It requires a
nonsingular integral 3 by 3 matrix with addressable group order, exactly
`abs(det S)` distinct dual characters (`k S` integral modulo reciprocal
translations), and uniform weights `1/abs(det S)`. The same uniform contract
applies to the distributed residue transform and occupied localization.
Non-diagonal and orientation-reversing matrices are valid for this algebraic
check; this does not add non-diagonal SCF or Hamiltonian support. Round-trip
reconstruction additionally requires a complete set of translation residues.

Native transform entry points also validate raw values before narrowing.
Large integer translations use modular phase arithmetic, preserving character
parity without converting the translation to a floating-point integer.
Fractional reciprocal coordinates are interpreted modulo primitive reciprocal
lattice integers. Finite representatives of any magnitude remain valid when
their represented value reduces to an admitted character. For example,
`1e300` is an integral binary floating-point value and represents the trivial
character; it is not an overflow of the signed-integer translation input.
Fractions already lost when creating a floating-point input cannot be recovered.
The home-auxiliary factor transform applies this same convention and still
requires complete, distinct dual characters. Nonfinite reciprocal coordinates
raise `ValueError` before native evaluation, including direct native calls;
they are not treated as reciprocal-lattice representatives.
The same raw integer checks apply to cyclic meshes, orbital counts and
permutation indices in the home-auxiliary MO, occupied-pair and Mayer
transform entry points. Mesh and output-size products are checked before
allocation. Real MO and Mayer inputs must be finite real arrays; use the
complex MO entry point for complex coefficients instead of discarding their
imaginary parts. Empty occupied or virtual MO spaces remain valid zero-width
outputs. These admission checks do not change the physical support.
These checks leave chi experimental and D93 v1 fitted/post-HF not-qualified.

## Explicit KS energy contract

The opt-in contract supports RKS and UKS with Libxc LDA, GGA and
kinetic-energy-density meta-GGA functionals, including supported global,
screened and range-separated hybrids. Semilocal examples include LDA, PBE and r2SCAN.
It remains experimental. Implementation, SCF stationarity and independent
scientific acceptance are separate; fitted results remain D93 not-qualified.

```python
from vibeqc import PeriodicKSOptions, run_aiccm2026dev_b_rks
from vibeqc.periodic.chi.xc import ChiXCControls

options = PeriodicKSOptions()
options.conv_tol_energy = 1e-11
options.conv_tol_grad = 1e-9
xc = ChiXCControls(density_image_radius_bohr=24.0,
                   partition_radius_bohr=12.0, block_size=128)
result = run_aiccm2026dev_b_rks(
    system, basis, "pbe", (3, 1, 1), options,
    support_policy="physical_pairs", xc_controls=xc,
)
```

The radii above are example convergence controls, not universal sufficient
cutoffs. Vary the density-image radius separately from the partition radius
and radial/angular grid resolution. Record changes in `xc_grid_electrons`,
XC energy and the XC operator. An unchanged result proves no convergence if
the executed support did not change. Successive differences are measurements,
not rigorous error bounds. Coulomb and one-electron support retain their
separate controls. No 10-bohr density cap is imposed.

For RI use `backend="ri"`, explicit `fit_source="classic"` or `"sr"`,
`coulomb_controls=ChiCoulombControls(...)`, and the same `xc_controls`.
Classic and SR remain distinct finite constructions. Gamma and complete
complex cyclic meshes use the same XC density assembly. The corresponding
UKS driver takes paired alpha/beta densities for `initial_density_k`.
The public `run_periodic_job` keywords are `chi_xc_controls`,
`chi_support_policy`, `chi_coulomb_controls`, and `chi_ks_reference`.

| Capability | Explicit KS contract |
|---|---|
| LDA / GGA / tau meta-GGA | RKS and UKS energies and variational operators |
| Coulomb sources | Direct, classic RI, SR/LR RI, each with explicit support |
| Density | Complex character matrices with occupation and per-cell weights |
| Occupations | Neutral insulating cells, integer fixed population per character |
| Geometry | All-electron 3D, diagonal cyclic extensions, including skew cells |
| Reference reuse | Process-local, sealed, identical KS request and executed state |
| Global, screened, range-separated hybrids | Kernel-specific direct, classic RI and SR fitted exchange |
| Laplacian, nonlocal XC, Yukawa, multiple screening ranges | Refused before quadrature allocation |
| Forces, ECP, correlation, persistent restart | Separate unsupported contracts |
| Scientific qualification | Pending independent native acceptance |

### Hybrid kernels and source identity

Use the same library or public-runner controls above with, for example,
`functional="pbe0"`, `"hse06"`, `"cam-b3lyp"` or `"wb97x"`.
Admission reads the executed Libxc component capabilities, weights and
exchange kernels. It requires 3D XC energies and first derivatives; a
potential-only or kinetic functional cannot stand in for an XC energy.
Hybrid tau meta-GGAs use the same tau adjoint as the semilocal path.
VV10, double hybrids, external providers, negative short- or long-distance
exchange fractions and multiple distinct screening ranges remain unsupported.

| Functional class | Executed exchange kernel | Direct | Classic RI | SR fitted |
|---|---|---|---|---|
| PBE0 | 0.25 / r | Implemented | Implemented | Implemented |
| HSE06 | 0.25 erfc(0.11 r) / r | Implemented | Implemented | Implemented |
| CAM-B3LYP | [0.65 - 0.46 erfc(0.33 r)] / r | Implemented | Implemented | Implemented |
| wB97X | [1 - 0.842294 erfc(0.3 r)] / r | Implemented | Implemented | Implemented |
| Eligible hybrid tau meta-GGA | Native component coefficients and range | Implemented | Implemented | Implemented |

Every cell in this matrix is experimental and awaits independent native
scientific acceptance. Existing HF and semilocal prerequisite gates remain
in force; source tests and stationary SCF alone do not qualify a result.

For spin-summed restricted density D, the exchange operator and energy are
`F_x = -K(D)/2` and `E_x = -sum_k w_k Tr[D_k K(D)_k]/4`. For unrestricted
spin densities they are `F_x,s = -K(D_s)` and
`E_x = -sum_s,k w_k Tr[D_s,k K(D_s)_k]/2`. K already includes the functional
coefficients. Hartree always uses full Coulomb. The Libxc semilocal pieces
already contain their hybrid subtraction and are not rescaled a second time.

SR fitting is a numerical split with its own parameter eta. It does not
select a physically screened interaction. Each physical kernel uses matching
two- and three-center integrals and its own metric/rank selection. The finite
fitted CAM operator is explicitly a full fit plus a signed erfc fit. Separate
fits are nonlinear in the kernel, so separately fitted full, erfc and erf
operators need not obey an exact additive identity at finite auxiliary rank.
The raw kernel integrals do obey the corresponding identity.

All fitted metrics omit p=0. Full exchange restores the BvK Ewald overlap
term `xi S D S`. Physical erfc exchange restores
`pi/(omega^2 V_BvK) S D S`, and full-minus-erfc long-range exchange has
`[xi - pi/(omega^2 V_BvK)] S D S`. Direct real-space erfc sums already
contain their finite zero mode, so no extra Madelung term is added to them.
This declared long-range convention must be matched when comparing another
implementation. At fixed finite BvK volume the erfc zero term diverges as
omega tends to zero; only the nonzero-momentum kernel approaches full
Coulomb in that limit. The omega-to-infinity erfc limit vanishes, including
its zero term. Coefficient limits retain the explicitly selected semilocal
correlation and are not automatically pure HF or pure PBE.

The accepted final Fock and energy are rebuilt without shifts or mixing.
Reference reuse binds backend/component identity, coefficients, physical range,
spin/occupations, grid/images, source/support, rank and factor bytes. Output
records `executed_chi_exchange` and coefficient-weighted exchange components
through the normal `vibeqc.output` manifest. Direct routes require real Gamma
densities and time-reversal paired complex characters; they refuse incompatible
densities rather than replacing them with a real projection.

Bloch AOs are summed over a point-centered physical sphere for each AO
center. Values and gradients use the identical image mask. Grid and image
loops are streamed; a candidate-budget refusal never truncates the image
sum. The finite support boundary is not differentiated with respect to
nuclear positions. This energy support therefore does not provide forces.
The convention is `tau = sum_i occupation_i |grad psi_i|^2 / 2`, and the
potential differentiates the executed energy without an additional tau floor.
Libxc supplies pointwise functional values and derivatives; the chi density
and AO operator assembly are independently constructed. See the
[Libxc derivative conventions](https://libxc.gitlab.io/manual/libxc-5.1.x/).

`result.executed_chi_xc` records resolved functional IDs/weights, Libxc
version, grid identity, density-image policy and measured integrated charge.
The output manifest carries that record alongside Coulomb support and final
SCF acceptance. Use `prepare_chi_ks_reference` from `periodic.chi.reference`
to prepare a tight process-local reference; pass it as `ks_reference` to an
identical library request. Changed functional, spin, occupations, grid,
images, Coulomb source, controls, runtime or mutated result bytes reject
reuse. A KS reference is not an HF post-HF reference.

The public runner accepts `chi_grid_options=GridOptions(...)` for the same
quadrature settings as the library `PeriodicKSOptions.grid`. Its density-image
and partition radii remain separate in `chi_xc_controls`. Direct Coulomb
cutoff overrides currently require the library options; public direct parity
uses the common default Coulomb options.

Direct support recording bounds the stored nuclear membership: shell-pair/cell
mappings, unique AO midpoints and selected atom/image labels. The limit remains
2,000,000 entries. Native image selection runs in bounded midpoint panels and
uses the same geometric predicate as the nuclear-attraction operator. A loose
product of enclosing image boxes is not the stored record size. Paneling
preserves all membership labels, their order and the canonical v1 JSON identity.
An actual record above the limit refuses before SCF without reducing any radius.

### R5.1 same-source canonical MP2 and UMP2

The developer entry point
`vibeqc.periodic.chi.canonical_mp2.evaluate_same_source_mp2` consumes a sealed
`ChiHFReference` made by `prepare_chi_hf_reference` with explicit fitted Coulomb
controls and `fit_source="classic"` or `"sr"`. It borrows that reference's actual
canonical AO factors. It never builds a classic fit for an SR determinant.
This developer entry point does not stamp public admission. Use the public
entry points below for admitted MP2/UMP2. The legacy post-HF routes, their
explicit-SR refusals and historical `auto` behavior are unchanged.

The public library entry point
`vibeqc.periodic.chi.correlation.run_chi_mp2(reference, frozen_core=0,
auxiliary_batch_size=64)` and the runner's `method="aiccm", variant="chi",
correlation="mp2"` adapter admit the explicit finite-source contract below.
Use `aiccm_backend="ri"`, `fit_source="classic"` or `"sr"`, explicit
`chi_coulomb_controls=ChiCoulombControls(...)`, `conv_tol_energy=1e-11`
and `conv_tol_grad=1e-9` (or tighter tolerances). The controls must select
`physical_pairs`, no reciprocal tail, and zero integral and fit screening.
The route requires a 3D all-electron full-Coulomb HF energy single point.
Implicit controls, the default four-center backend, historical support,
smearing, constrained occupations, reduced characters, restarted jobs,
optimization and response requests refuse before SCF. The final state must
also pass the achieved stationarity, source and occupation checks below.
Use `scf_reference="uhf"` to request UMP2 explicitly, including on a singlet;
an open-shell cell otherwise selects UHF automatically. The correlation
selector stays `"mp2"` for either reference. Public UMP2 currently requires
`fit_source="classic"`; SR UMP2 remains gated pending independent
complex-spin operator acceptance.

The runner accepts `chi_frozen_core` and `chi_auxiliary_batch_size`, with
the same meaning as the library arguments. Omitting them uses zero frozen
orbitals and panels of 64. These keywords refuse on every other route.
Nonzero public frozen counts currently require explicit classic RHF.
SR and UHF frozen active spaces remain gated pending applicable native
acceptance; all-electron MP2/UMP2 uses zero frozen counts.
The adapter seals the already executed HF result and borrows its factors;
it does not run another SCF or refit. Its output-layer integration carries
actual spin-resolved correlation and total energies, source/reference identity,
rank and stationarity records to `.out` and `.system`, never an SCF-only
success for a correlation request. The `.system` source record preserves
momentum coordinates as TOML numbers, including with NumPy 2.
As on other periodic correlated runners,
the result's `energy` is HF; `e_correlation` and `e_total_correlated` name
the post-HF energies. Kernel preparation and output tests are not native
acceptance. There is no environment switch to bypass these checks.

| Source | Reference | Method | Implemented | Publicly admitted | Independently accepted native | Production qualified |
|---|---|---|---|---|---|---|
| Explicit classic | RHF | MP2 | Kernel and public adapter | Scoped contract | Finite native controls | No |
| Explicit SR | RHF | MP2 | Kernel and public adapter | Scoped contract | Finite native controls | No |
| Explicit classic | UHF | UMP2 | Kernel and public adapter | Scoped contract | Finite native controls | No |
| Explicit SR | UHF | UMP2 | Kernel and gated adapter | No | Partial; complex triplet criterion open | No |
| Direct | RHF/UHF | MP2/UMP2 | No adapter | No | No | No |
| Any | ROHF or KS/hybrid | Correlation | Later increment | No | No | No |

CCSD(T), ROHF-MP2 and ECP correlation are separate later increments. Existing
legacy library methods retain their documented scope; this table describes
only the new same-source route. Chi remains experimental and D93 v1 fitted and
post-HF results remain not-qualified.

The canonical adapter retains each q record, metric eigenvalues and keep mask,
rank, factor hash, basis and auxiliary identities, executed support, kernel and
zero-mode convention. Current three-dimensional sources use positive retained
metric modes. Signed metrics and eigenmode-only frames refuse. The full
canonical auxiliary dimension is preserved even when the retained rank is
smaller. No whitening, rank selection or refit occurs in correlation.

For complex characters the Coulomb tensor uses
`(ia|jb) = sum_P L(ki,ka)[P,i,a] * conj(L(kb,kj)[P,b,j])`.
Both factors have the same momentum transfer. This does not require phase
matching of independently diagonalized q and minus-q metrics. Momentum is
conserved exactly on the complete unshifted finite character group. Ordered
same-spin pairs carry a factor of one quarter after antisymmetrization;
opposite-spin pairs carry one. Primitive-cell factors give an overall
`Nk**-3` energy normalization. The external overlap-only exchange zero-mode
restoration vanishes in occupied-virtual matrix elements of the admitted
orthonormal canonical state; its effect on HF orbital energies is retained.
These are the conventional Moller-Plesset equations (1934), with periodic
normalization described by McClain et al. (2017); output citation routes are
`chi-same-source-mp2` and `chi-same-source-ump2`.

Admission requires the sealed stationary full-exchange RI HF state, achieved
energy and energy-rebuild errors at most `1e-11` hartree, and orbital and density
residuals at most `1e-9`. Canonical eigenpairs, metric orthogonality and orbital
versus reported density are checked again. Smearing, fractional occupations,
metallic gaps, noncanonical states, ECP and KS/hybrid references refuse.
Denominators must be strictly below `-denominator_tolerance` (default `1e-12`
hartree); no clamping or regularization occurs.

Explicit fitted RHF and UHF use fixed integer populations for their initial
orbital fill and intermediate SCF trials. A degenerate HCORE seed can therefore
evolve into a gapped determinant instead of remaining a fractional ensemble.
Convergence and terminal reporting still refill the physical unshifted Fock
globally across the mesh and require integer Aufbau populations at every
character. A final metallic or fractional state remains unsupported. This
trial rule does not change ordinary GDF or KS occupations.

`frozen_core` is an explicit nonnegative count of lowest occupied orbitals per
character and spin, or a pair of counts for UHF. RHF counts must agree. A core
boundary through a degenerate occupied subspace refuses. Returned active-space
triples are `(frozen, active occupied, virtual)` for alpha and beta; retained
overlap dimensions are preserved. `auxiliary_batch_size` controls factor panels;
one occupied pair's virtual matrix is stored at a time. It is not a process
memory cap or a new distributed implementation. Existing batching and reference
transport remain in place.

`vibeqc.output.chi_correlation.chi_correlation_record(result)` supplies SCF,
correlation and total energies, spin components, active spaces, exact source and
reference identities, convergence evidence, citation route and admission status.
The successful public result stamps its admission policy separately from
`quantitative_status="not-qualified"`; a direct developer evaluation records
`public_admission="not-requested: developer evaluation"`. A correlation failure
cannot become a successful SCF-only correlated job.

Native finite controls compare the retained operator against a separate
Gaussian construction, rebuild stationary HF from exported raw operands,
and compare spin-resolved correlation against an explicit spin-orbital
tensor. H2 covers classic/SR, RHF/UHF, Gamma and complex characters,
cold/reused factors, panels 1/7/64 and same-root closed-shell collapse.
Classic LiH adds restricted frozen-core bookkeeping and genuine triplet
UMP2, including nonzero alpha-alpha, beta-beta and alpha-beta terms on the
complex mesh. Zero same-spin channels in a Gamma determinant remain explicit.

The original SR LiH auxiliary radius of 10 bohr gives an indefinite metric
and remains refused. Supplementary SR triplet measurements use a separately
recorded 32-bohr auxiliary domain with the same remaining finite controls
and rank threshold. The Gamma measurement passes, but the complex
triplet comparison exceeds the inherited independent Fock tolerance
of `1e-9`. Reference-integral accuracy may contribute to this unresolved
comparison, so public SR UMP2 remains gated. These measurements do not
make the original 10-bohr source valid or establish physical support
convergence. Negative modes are never sign-clipped.

Source comparisons retain operator tolerance `2e-8` and the inherited
independent Fock tolerance `1e-9`. Correlation tolerance remains `1e-10`
hartree per cell. The separate raw-state HF prerequisites above also apply.
They establish the experimental capability envelope; physical basis/support
convergence, ORCA/CRYSTAL parity, production qualification and completion of
the whole R5 program remain separate. Broader source/support ladders and
ionic obligations remain open. Array-level oracle agreement alone is not
native acceptance.
