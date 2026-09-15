# χ-CCM / aiccm2026dev-b fleet inputs

This is the executable test-set layer for χ-CCM[^xccm-convention], the
finite-character (Γ-centred character-mesh) CCM implementation selected by
`aiccm2026dev-b`. It intentionally reuses this directory's canonical
`testset.py`, so Γ-CCM and χ-CCM receive identical cells, atoms, basis labels,
and cyclic sizes. It does not call or import Γ-CCM's electronic-structure
implementation.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention, currently
    `exchange_q0="bvk-ewald"` for the production χ-CCM path. Their distinction
    is not a choice of Coulomb kernels, and equality for a specified
    operator/route is evidence to establish, not a naming premise.

> **D89 approach identity.** Successful B records identify
> `ccm_approach="chi-ccm"`,
> `ccm_construction="finite-translation-group-character"`, and
> `evaluation_representation="gamma-centred-character-mesh"`. These immutable
> identifiers are serialized in `finite_torus_convention`, QVF vendor metadata,
> and `.system` run metadata. QVF wrapper schema version 2 retains the legacy
> version 1 `representation` string for compatibility;
> `evaluation_representation` is normative. The real-Gamma Fourier control is
> not assigned a Γ-CCM identity.
> Pre-D89 records are not numerically retracted solely for missing these fields,
> but cannot serve as construction-comparison evidence. The reportable Γ/χ
> approach map is currently empty.

> **Numerical-support status.** D93 gives every completed fleet row an exact
> `vibeqc.aiccm2026dev-b.two-electron-support/v1` payload. Current
> four-center RHF, PBE, and PBE0 rows qualify only when the producer records
> the executed low-level backend, exact M5 `sr_image_precision=1e-6` policy,
> physical cutoffs, and a finite resolved absolute `sr_image_extent_bohr`
> beyond the electronic cutoff. Its duplicated cutoff fields must agree. RI,
> RIJCOSX, MDF, and
> post-HF rows record their base/runtime support but remain `not-qualified`
> until high-G RSGDF-tail, COSX, MDF, and correlation-factor contracts are
> established. `audit_b.py` and `compare_b.py` reject missing, malformed,
> internally contradictory, route-inconsistent, or unqualified quantitative
> rows. D103 adds an independent overlap-fold gate for every direct row, and
> D104 adds independent post-repair shared-Ewald evidence. A fresh direct row
> must satisfy D93, D103, and D104; a fresh fitted or post-HF row can satisfy
> D104 but remains held by its D93 route-specific support. Unsupported and
> error records remain explicit failure evidence. HSE uses the shared padded direct route
> but is not in this fleet and still needs route-specific validation.

> **D99 fitted-route baseline.** Since `ddbe859d1`, every 3D fitted χ-B
> supported Gamma-only fitted path and every multi-k restricted/open-shell
> route uses `ewald_nuclear_repulsion` instead of
> `nuclear_repulsion_per_cell`. This pins routing, not convergence. D102
> recorded the historical shifted-pair cutoff defect in the shared Ewald
> helper. Commit `e578b86c` repaired its image enumeration with centered
> displacement and interplanar pair bounds. Pre-repair rows and every stored
> D102 v1 row remain revision-bound; fresh rows must carry qualified D104 v2
> evidence. Gamma-only names the
> one-character χ evaluation path here and does not confer union-and-weight
> Γ-CCM identity. Since
> `6ce142339`, compact dense-core MDF jobs in the MgO/STO-3G class fail closed
> through the B selector. The validated vacuum-padded MDF envelope is
> unchanged, while all fitted rows remain D93 `not-qualified`.
> `ad96a2f63` supplies a bounded shared-q auxiliary Fourier-transform cache
> with bit-identical results and memory/performance effects only. All fitted B
> RHF/RKS/UHF/UKS dispatches explicitly set `ibz_native=False` and
> `compute_gradient=False`, preserving the complete unreduced character mesh
> and total-gradient fail-close against generic-driver defaults. No 1D/2D SCF
> or total-gradient route opens, and D83, D88, and D98 remain unchanged.
> The campaign producer `run_case_cmp.py` is correspondingly labelled as a
> neutral fitted-torus Bloch/GDF representation control and separately
> recomputes its executed nuclear scalar through the same shared Ewald helper.
> That equality is dispatch evidence, not a convergence proof. The producer
> emits D104 v2 evidence for the repaired helper but remains overall
> `quantitative_status="not-qualified"` because its fitted support is not
> qualified. It is not the mapped real-Gamma control.

> **D103 direct-support stop.** Since `9633ee6bc`, corrected-gauge
> `four_center` RHF/RKS/UHF/UKS fails before the first Fock build when its
> measured overlap-fold drift exceeds `1e-2`. Moderate
> `1e-4 < drift <= 1e-2` support still executes but is not quantitative.
> Every completed fleet row now carries the exact
> `vibeqc.aiccm2026dev-b.overlap-fold-support/v1` payload: direct rows bind the
> executed maximum drift and electronic cutoff and qualify only at
> `drift <= 1e-4`; fitted rows are exactly `not-applicable`. `audit_b.py` and
> `compare_b.py` reject missing, malformed, cutoff-inconsistent,
> mesh-inconsistent, or unqualified direct evidence. The payload fixes
> `diagnostic="max-abs-element-overlap-bloch-fold-drift/v1"` and
> `reference_cutoff_factor=1.5`; its `character_mesh_shape` must equal the
> executed row mesh. The χ selector inherits the runtime stop
> without a bypass. This contract is independent of D93 and D104; satisfying
> one does not satisfy either of the others.

> **D102 history and D104 post-repair support.** Historical completed rows
> carry exact
> `vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v1` evidence with
> `qualification="not-qualified"`, the affected implementation label, and
> `repair_commit=null`. Version 1 has no qualified state. Clearing a global
> stop or reading it after the repair cannot upgrade a stored v1 row.
> D104 replaces the blanket hold only for freshly produced exact
> `vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v2` evidence. Its exact keys
> are `schema`, `qualification`, `reason`, `implementation`, `repair_commit`,
> `producer_commit`, `core_build_id`, `probe_attestation_id`,
> `repair_is_ancestor`, and `canary`. They bind
> `implementation="centered-displacement-interplanar-pair-bounds"`, repair
> commit `e578b86c00268a172b651cae29c7845836e31e17`, the finalized producer,
> core, and probe identities, verified repair ancestry, and a same-process
> canary. Qualification requires D77-clean provenance,
> `repair_is_ancestor=true`, and the exact
> `vibeqc.ewald.shifted-pair-canary/v1` fixture
> `mgo-point-charges-basis-shift-37a1/c18` at an 18-bohr cutoff to remain
> invariant when the oxygen point charge is shifted by 37 lattice vectors along
> `a1`, with an absolute energy difference at or below `1e-10` Ha. Audit and
> comparison bind the v2 identities back to finalized provenance.
> D104 attests only the shared Ewald repair and cannot supply D93 or D103.
> `run_case_cmp.py` emits this v2 object for its Bloch/GDF same-helper control
> but remains nonquantitative. The mapped A-owned real-Gamma control,
> `aiccm-hf-direct`, currently emits no D104 v2 evidence and is therefore
> unavailable to `compare_b.py`.

> **D88 lattice-fingerprint status.** `PeriodicSystem.lattice` stores direct
> lattice vectors as columns. χ-CCM-B therefore uses Cartesian shifts `A @ n`
> and the BvK lattice `A @ diag(mesh)`; the runner records
> `primitive_lattice_bohr` and `lattice_vector_convention="columns"`. The fleet
> audit binds the exact BvK matrix to those fields, while the comparator defines
> no Γ/χ comparison if the binding is missing or inconsistent. All affected pre-D88 Γ-CCM and χ-CCM
> records must be rerun under the fingerprint rule; no missing convention field
> is inferred or retrofitted. A symmetric lattice that commutes with
> `diag(mesh)` is numerically outside this defect class, but its old record is
> not newly attested. The shared builders for `graphene`, `mgo-slab`, `ice-ih`,
> `co2-dryice`, and `sio2-quartz` require fresh Γ and χ fingerprints. Ordinary
> pre-D88 periodic GDF exact-exchange and BIPOLE J/K records for affected
> lattice/mesh combinations also require audit and rerun. For the
> skew audit `A=[[7,.4,.2],[.3,8,.5],[.1,.6,9]]` bohr and mesh `(2,3,1)`, the
> corrected positive code convention is `xi_M=0.138352993811598`; the pre-D88
> fitted helper gave `0.143621616230995` and the pre-D88 four-center probe gave
> `0.138913224426180`. The fitted error is 5.268622 mHa/cell of overbinding for
> a two-electron RHF seam. The article theory already uses columns correctly:
> D88 repairs code and fingerprints, not the kernel, signed-Madelung
> convention, or theory. All 1D/2D absolute-energy routes and analytic total
> gradients remain fail closed. D89 records that Γ-CCM and χ-CCM are
> distinct CCM approaches, not representation aliases; their comparison holds
> the declared exchange-q=0 convention fixed and does not assign different
> Coulomb kernels by name.

The χ-CCM runner covers every currently implemented closed-shell route:

| Route | Method | ERI treatment |
|---|---|---|
| `rhf-4c` | RHF | native four-center periodic J/K, 3D only |
| `rhf-ri` | RHF | pair-resolved 3-center RI-J/RI-K, 3D only |
| `rhf-rijcosx` | RHF | RI-J plus COSX exchange, 3D only |
| `rks-pbe-4c`, `rks-pbe0-4c` | pure/hybrid KS | four-center, 3D only |
| `rks-pbe-ri`, `rks-pbe0-ri` | pure/hybrid KS | RI, 3D only |
| `rks-pbe-rijcosx`, `rks-pbe0-rijcosx` | pure/hybrid KS | RIJCOSX, 3D only |
| `ri-mp2` | canonical MP2 (3D only) | momentum-conserving finite-torus RI |
| `dlpno-mp2` | local-PNO MP2 (3D only) | exact real representation of the RI torus |
| `dlpno-ccsd` | local-PNO CCSD (3D only) | exact real representation of the RI torus |
| `dlpno-ccsd-t` | local-PNO CCSD(T) (3D only) | exact real representation of the RI torus |

The default post-HF `--local-mode exact` disables PNO and pair truncations. It
is the accuracy oracle. `--local-mode pno` exercises the current PNO
approximation, while periodic distance screening and local fitting remain
disabled until minimum-image domains are implemented.

In ORCA-style shorthand, the current vibe-qc `rijcosx` backend is the
RIJCOSX/GridX route: RI-J plus grid/COSX exchange. There is no separate public
`gridx` selector in this χ-CCM harness.

## Generate and review

Select an enabled external `AICCM_SITE_CONFIG` profile before using remote
host defaults; see [site-configuration.md](site-configuration.md). Names such as
`compute-large` below are placeholders. Use `--host` for an explicit remote
queue alias, or `--local` to render local commands without a site profile.

From the repository root:

```sh
# First AICCM paper χ-CCM matrix: uniform H chain, H2-pair chain, diamond,
# MgO, Al2O3 corundum, and NaCl through HF, RI/COSX, PBE-RI, PBE0,
# and DLPNO-CCSD(T), with fail-closed skips printed as comments.
python studies/aiccm-2026/make_jobs_b.py --profile paper1

# Same route/mesh matrix at the CRYSTAL-matched orbital basis. Use this for
# article tables against pob-TZVP-REV2 CRYSTAL references; keep the STO-3G pass
# as the cheap cross-stream smoke.
python studies/aiccm-2026/make_jobs_b.py --profile paper1 --basis pob-tzvp-rev2

# 10 runnable jobs on 3D LiH, plus 18 fail-closed 1D/2D rows and 3 direct
# LiH/STO-3G rows blocked by their measured critical 15-bohr overlap fold.
python studies/aiccm-2026/make_jobs_b.py --profile coverage

# All closed-shell systems through all nine SCF paths.
python studies/aiccm-2026/make_jobs_b.py --profile scf

# All valid post-HF inputs, or the complete valid Cartesian matrix.
python studies/aiccm-2026/make_jobs_b.py --profile posthf
python studies/aiccm-2026/make_jobs_b.py --profile full

# Diamond-family RKS/PBE/RI convergence investigation with residual traces.
python studies/aiccm-2026/make_jobs_b.py --profile investigate
```

The current coverage commands are previews only: all ten emitted jobs use
fitted SCF or post-HF routes that remain D93 `not-qualified`, while its three
direct LiH/STO-3G rows fail the D103 runtime stop. D102 no longer imposes a
blanket hold on a fresh post-repair row. A direct command may be run as a
diagnostic, but no result is reportable unless its finalized D77 provenance and
its independent D93, D103, and D104 contracts all pass. The energy-free D101
route diagnostic below is a separate, nonquantitative campaign.

The `paper1` profile reads `paper_host` from the external site configuration.
Select an explicit alternative with `--host` only after its producer runtime
satisfies the required
D77 clean-checkout or implemented D91 bundle contract. Keep the route matrix
fixed when moving between hosts; only the queue target should change. Use
`--basis pob-tzvp-rev2` for basis-matched article/CRYSTAL runs; without an
override, the registry basis remains the default.

The coverage profile keeps its 1D H-chain and 2D graphene anchors as explicit
unsupported rows and would submit only the remaining ten runnable fitted 3D
LiH jobs to the configured coverage target; its post-HF
LiH gate uses a tractable `2x1x1` finite torus. In the larger profiles, only
3D runnable rows are emitted. Override their fleet target with `--host`.
Each emitted command contains the basis, cyclic mesh, SCF thresholds, RSGDF
cutoff, and local-correlation mode explicitly; `vq` copies the complete
`studies/aiccm-2026/` directory into its workspace.

Each emitted job currently claims 4 CPU slots and 8 GB of memory. That is
appropriate for the compact coverage jobs, but not for every paper-1 crystal.
In particular, do not submit Al2O3/STO-3G at the 2 x 1 x 1 character mesh with
the generic claim: the one-character RI reference already exceeded 10 GB.
Treat that first common PBE/RI cell as a 32 GB, 48-hour job after the generator
gains a resource override or per-system sizing. `run.sh --b` locates the
development checkout's virtual-environment interpreter on each fleet layout,
runs the cached numerical preflight, and then launches the B producer. The
producer completes the composite attestation in the process holding the result,
so directory submissions do not accidentally use a bare system Python or rely
only on a probe from an earlier process.

Useful focused command previews are:

```sh
python studies/aiccm-2026/make_jobs_b.py --profile coverage --system lih-rocksalt --host compute-large
python studies/aiccm-2026/make_jobs_b.py --profile scf --system mgo --host compute-large
python studies/aiccm-2026/make_jobs_b.py --profile posthf --system lih-rocksalt --host compute-large
python studies/aiccm-2026/make_jobs_b.py --profile investigate --system c-diamond
```

Open-shell `nio-afm`, Gamma-only RI-RKS/RIJCOSX cases, and every 1D/2D SCF or
post-HF route are listed as unsupported rather than submitted. All 1D/2D SCF
and post-HF routes remain explicit coverage rows, but none is runnable. This
is intentional for the paper-1 closed-shell route matrix: the direct fallback
does not provide a neutral lower-dimensional Green function, the shared
neutral-RI/GDF mesh collapses transverse reciprocal structure and is not a
Coulomb kernel, and post-HF is validated only for the declared 3D Hamiltonian.

## D114/D116 two-rank implementation validation

The ordinary `run.sh --b` producer is intentionally serial: launching it
under `mpirun` would let every rank write the same JSON path. Use the dedicated
rank-safe path for the restricted four-center D114 seam:

```sh
bash run.sh --d114-mpi --mesh 2 2 2 --basis sto-3g \
  --cutoff-bohr 15.0 --max-iter 1
```

The D116 PBE-RKS extension uses the same proven two-rank launcher contract but
a distinct immutable validation profile:

```sh
bash run.sh --d116-rks-mpi
```

That profile fixes c-diamond/STO-3G, the `(2,2,2)` character mesh, 15-bohr
electronic and nuclear cutoffs, one SCF cycle, zero Fock mixing, and symmetry
off. Attempts to change those scientific inputs fail closed. Each rank runs
the unfarmed PBE BIPOLE reference and the farmed χ selector. Total,
electronic, nuclear, and XC energies must agree within `1e-10` Ha, and every
Fock-matrix and density-block element must agree within `1e-12`. Array shapes,
convergence and iteration state, executed zero Fock mixing, D83 exchange-q=0
applicability `inactive`, task census and order, launcher controls, and
source/core identities remain exact. Rank zero alone writes
`chi-d116-c-diamond-rkspbe4c-mpi2.json` with the versioned D116 v2 contract.
The JSON retains observational `exact_parity`, reports normative
`contract_parity`, and embeds the acceptance bounds. Existing D116 v1 files
retain their exact-contract meaning.

The wrapper runs the source/core probe once, refuses a runtime without
`mpi4py` or an MPI launcher, and starts exactly two ranks. Each rank evaluates
the same replicated unfarmed BIPOLE reference and farmed χ calculation with
`progress=False`. Energy, electronic energy, every Fock matrix, and every
density block must agree exactly for D114 and within the v2 bounds above for
D116. The gathered records must prove active cyclic ownership on both ranks,
a complete internal translation sum, and one common scheduling fingerprint.
Only rank zero atomically writes
`chi-d114-c-diamond-rhf4c-mpi2.json` for the RHF profile. Both artifacts are
implementation-parity evidence, not independent absolute-energy benchmarks.

For a compute-cluster `--cpus 20` allocation, the wrapper owns both MPI ranks and divides
the total `VQ_CPUS` allocation evenly, giving ten OpenMP threads per rank. Do
not pass `--scheduler-tasks`: Torque does not expand the node allocation from
that field, so the wrapper rejects any nonempty `VQ_SCHEDULER_TASKS`. Intel MPI
receives an OpenMP-sized compact pinning domain; Open MPI receives an explicit
two-ranks-per-node processing-element map with core binding. Unknown
launchers, inconsistent resource fields, a native thread-count mismatch, and
unbound hybrid execution fail closed. The JSON records the executed resource
contract and fixed SCF/Ewald seam controls.

## Pinned D101 χ route-control diagnostic

`run_case_cmp_b.py` is the dedicated high-level χ producer for
`COMPARISON_1D2D3D_2026-07-15.md`. It calls only
`run_periodic_job(..., method="RHF", functional=None,
jk_method="aiccm2026dev-b", aiccm_backend="ri")` and writes
`d101-cmp1d2d3d-<system>-chi-rhf-ri.json` under schema
`vibeqc.aiccm2026dev-b.campaign-1d2d3d/v1`. It is intentionally separate from
`run_case_cmp.py`, the neutral fitted-torus Bloch/GDF representation control,
and from the quantitative fleet/curation filenames.

Do not run the following command under the current compute-managed managed runtime. Once
the interpreter selected by `run.sh` (its managed wrapper or an explicit
`VIBEQC_PYTHON`) resolves to a current D77-clean Git checkout, run the crystal
validation first through the narrow launcher selector:

```bash
git fetch origin
SHA=$(git rev-parse origin/main)
vq submit --host compute-managed --job-name d101-chi-3d-ri -d studies/aiccm-2026/ -- \
  bash run.sh --comparison-b 3d --expected-source-sha "$SHA" \
  --vq-host compute-managed
```

Only a contract-valid nonquantitative fetched 3D record with the same stable D77
producer tuple (source, version, core, attestor, libraries, and payload) may be
supplied through `--validation-record` to the `1d` and `2d` invocations. The
Slurm node hostname and full per-job attestation digest may differ.
Those names identify a chain and monolayer embedded in vacuum; all three
systems remain declared `dim=3`, with exact campaign geometry, mesh, and
numerical pins. The producer binds exact current `origin/main`, required repair
ancestors, tracked payload bytes, complete character residues, the D83/D88
convention and BvK lattice, executed zero Fock mixing, exchange assembly,
the shared Ewald nuclear scalar, and the internal energy decomposition.
The serialized GDF method and cutoff evidence is a B-selector-resolved setting
forwarded into the fitted driver, not backend-owned telemetry or D93 support
qualification. Madelung and nuclear values are separately recomputed through
the same shared helpers used by SCF, so they are dispatch/assembly evidence and
not independent kernel implementations.

The same-helper nuclear check is not an independent convergence audit.
Historical commit `a8b9ac2f8` records the 3D shifted-pair cutoff defect;
`e578b86c` repairs it with centered displacement and interplanar pair bounds.
That repair does not retroactively qualify pre-repair absolute energies or any
stored D102-v1 object. D101 remains a nonquantitative route diagnostic because
it serializes no energy and its fitted RI support is D93 `not-qualified`.

`declared_model="3d-periodic-in-vacuum"` describes a vacuum-padded geometry;
`boundary_model="3d-periodic"` names the fully 3D periodic Green function.
This pair is intentional and does not open an isolated wire/slab convention or
bypass D72/O1. `--vq-host compute-managed` is an operator assertion and the numeric
Slurm job id proves scheduler context; the current vq environment does not
export a cryptographic target-alias binding.

A successful record always carries all four fields together:

```text
status = ok
evidence_role = route-plumbing-diagnostic
quantitative_status = not-qualified
comparison_status = no-gamma-chi-construction-comparison-defined
```

`status=ok` means only that execution and the internal D101 contract audit
succeeded. It is not quantitative acceptance and is not a comparison verdict.

D93 bars fitted RI absolute energies, so the campaign JSON contains no total
or per-atom energy. The ordinary runner output and SCF log remain
nonreportable diagnostic artifacts; do not copy an energy from `.out`,
`.system`, the progress log, or another runner side file. Check a record with:

```bash
python audit_cmp_b.py --expected-source-sha <full-origin-main-sha> \
  <record.json>
```

For a `1d` or `2d` record, also pass
`--validation-record d101-cmp1d2d3d-3d-chi-rhf-ri.json`; the auditor rejects a
child record unless it can recompute the exact parent-file SHA and identifiers.
This dedicated auditor does not print an energy. D101 files are deliberately
excluded from generic fleet globs, `audit_b.py`, `compare_b.py`, and curation.
D101 accepts only D77 clean-checkout attestation because the normative D91
bundle transport/producer/auditor contract is not implemented. The current
compute-managed deployment is stale and cannot produce an accepted D101 record: its
managed development runtime is bundle-shaped, but its replaceable artifact
path/checksum and digest-free launcher do not satisfy D91 immutability. D101
also rejects bundle transport even after a routine refresh to the right SHA.
Do not submit this campaign until either the interpreter selected by
`run.sh` resolves to a separate D77-clean Git checkout or the D91 implementation
exists. D101 creates no Γ/χ
approach map and emits no historical v2 comparison contract.

## Outputs and properties

Every job writes `<system>__b-<route>.json` to `$VQ_WORKDIR`. In addition to
the energy and timing, SCF records contain the electronic and nuclear energy,
HOMO, LUMO, fundamental gap, finite-group density idempotency, electron count,
Wigner-Seitz partition error, imaginary residual, final SCF energy step,
final commutator norm, DIIS subspace, accelerator settings, and a compact
SCF trace tail. Post-HF records contain the HF and correlation components,
same/opposite-spin MP2 components when applicable, triples correction, pair
counts, T1 norm, PNO correction, and finite-torus factor residuals where the
method provides them.

Every completed row also carries `two_electron_support`. Four-center records
store the physical cutoffs, preserved low-level runtime backend, exact M5
domain policy and precision, and resolved absolute image radius. The nested
cutoffs must exactly match `direct_lattice_cutoffs`. Fitted records store the
active RSGDF or MDF
base cutoff and explicitly null missing tail support; RIJCOSX and post-HF add
their unbound exchange or correlation status. Use `--rsgdf-ke-cutoff` or
`--mdf-ke-cutoff` to change the corresponding recorded fleet input.

Every completed row additionally carries the exact D103
`overlap_fold_support` object. A four-center record stores the measured
corrected-gauge maximum overlap-fold drift, its executed electronic cutoff,
the exact diagnostic identifier, the fixed `1.5` reference-cutoff factor, its
executed character mesh, the `1e-4` quantitative target, and the `1e-2`
runtime stop threshold. It is qualified only at or below the quantitative
target. RI, RIJCOSX, and post-HF records serialize the same schema as exactly
`not-applicable`, with null drift, cutoff, factor, and mesh fields.

Every freshly produced successful or non-converged fleet row also carries the
exact D104 `ewald_shifted_pair_support` v2 object. Qualification binds the
repair commit, implementation, finalized producer/core/probe identity, repair
ancestry, and passing same-process MgO shift canary. Audit and comparison reject
missing, malformed, unqualified, provenance-inconsistent, or old v1 evidence.
Historical v1 objects remain permanently `not-qualified`.

Every successful or non-converged fleet row also carries the exact seven-key
`vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1` object. It records the live
resolver, `c_full`, `c_sr`, physical inverse-bohr screening parameter,
screened-branch applicability, and screened assembly. The independent route
oracle requires all current RHF, PBE, PBE0, and post-HF routes to be screened
`inactive` / `not-applicable`; HSE is not a fleet route. Legacy unversioned
four-field objects, extra keys, and active or full-range-minus-long-range
screened labels fail before an energy or control can render. The separate D83
`exchange_q0_applicability` still describes only the full-range q=0 seam and
is active exactly when `c_full` is nonzero.

`scf_options.fock_mixing` records the requested input.
`convergence_diagnostics.fock_mixing` records the executed effective
previous-Fock weight after backend defaults are resolved. The values can differ
for a DIIS-off four-center KS run, where requested `0.0` can execute as `0.30`.
This provenance correction does not change the SCF algorithm or energy.
Pre-D86 records in that case are not convergence-fingerprint-complete.

Direct Python callers use one D87 override rule: a non-`None`
`fock_mixing=` keyword overrides `options.fock_mixing`, otherwise the options
field supplies the request, and the caller's mixing field is not rewritten.
All `four_center` methods and multi-cell fitted RHF/RKS accept a nonzero value
in `[0, 1)`. Three-dimensional Gamma-only RI RHF accepts resolved zero but
rejects nonzero rather than switching to the legacy molecular-limit GDF
operator. Fitted RI/RIJCOSX UHF/UKS reject a nonzero value because those
backends have no previous-Fock mixing loop. An explicit keyword zero overrides
a nonzero options value, while
DIIS-off four-center KS may still resolve requested zero to the executed
automatic `0.30` recorded by D86. The fleet runner is closed-shell and
options-only, so its route inputs are unchanged. Successful pre-D87 direct
calls with conflicting keyword/options values, plus fitted RI/RIJCOSX UHF/UKS
calls with a nonzero options-only request, have incomplete request fingerprints.
Pre-D87 Gamma-only RI RHF calls where either input source was nonzero,
including an explicit-zero keyword over nonzero options, followed the legacy
operator and are not χ-CCM-B results. On supported execution routes, mixing
changes convergence control rather than the finite Hamiltonian; the Gamma
explicit-zero correction instead restores the declared operator.
The common D72 dimension guard runs before every backend-specific Gamma or
mixing guard. This closes a former 1D one-cell RI-RHF escape; any value from
that route is invalid and must not be reported as a χ-CCM-B absolute energy.

These are diagnostic properties of the computed wavefunction. Periodic dipole
moments and localized charge populations are not emitted because their gauge
and cell-partition conventions have not yet been derived for B; inventing a
molecular convention here would make the comparison less sound.

## Store local results in the input library

For local smoke runs, `curate.py` launches one or more B routes through the
same `run.sh --b` preflight and producer-process attestation used by queued
jobs, then stores their JSON records in the nested `aiccm2026testset` layout:

```sh
python studies/aiccm-2026/curate.py \
  --stream aiccm2026dev-b \
  --testset-root ../qc-input-library/aiccm2026testset \
  lih-rocksalt rhf-ri,rhf-4c --basis sto-3g --mesh 2 1 1

# Preserve one intentional O1 negative record without running SCF.
python studies/aiccm-2026/curate.py \
  --stream aiccm2026dev-b \
  --testset-root ../qc-input-library/aiccm2026testset \
  uniform-h-chain rhf-ri --basis sto-3g --mesh 8 1 1
```

The output path is
`<testset-root>/aiccm2026dev-b/<system>/<basis>/<route>/`. Successful records
and intentional fail-closed `status="unsupported"` records are both stored
as self-describing `result.json` files with a small `RUN.meta` sidecar. The
curator stores a producer-attested successful record only after independently
validating its D77 composite producer attestation. Curation does not establish
reportability: audit and comparison still require qualified D93, D103, and
D104 evidence for each applicable row. An explicit probe skip or
other untrusted success remains in the scratch output and cannot overwrite a
trusted curated row. Intentional unsupported rows contain no reportable energy
and may still be stored as untrusted negative coverage. The curator fills
missing selector, system, route, and basis identity fields before
storage, without changing the scientific result payload. Direct local curation
copies the producer host, vibe-qc version, full source commit, clean/dirty
state, compiled-core build time and path SHA256, composite-attestation ID, and
payload ID from the produced JSON into `RUN.meta`; it does not replace them
with the curator workstation's identity. Attestation remains `not-recorded`
unless it came from the producing run. The default target can also be set with
`AICCM2026_TESTSET_ROOT`; if neither is supplied, the helper looks for a sibling
`qc-input-library/aiccm2026testset` checkout.

To store already-fetched vq workspaces without rerunning the calculations,
ingest the fetched result tree instead:

```sh
python studies/aiccm-2026/curate.py \
  --stream aiccm2026dev-b \
  --testset-root ../qc-input-library/aiccm2026testset \
  --ingest-results ~/vibeqc-runs/aiccm-b-paper1
```

Ingest mode imports only records matching the selected stream, preserves the
scientific JSON payload while normalizing those identity fields, and marks the
`RUN.meta` return code as `not-recorded` because the subprocess status belongs
to the fetched queue job. Producer provenance is copied only when the fetched
JSON itself carries it; otherwise the host, version, commit, and core build are
written as `not-recorded`. The ingesting workstation is never misidentified as
the machine/build that produced a fetched energy. An exact match with an
existing `result.json` is a successful no-op that preserves its `RUN.meta`;
duplicate batch identities and differing existing records fail closed. Ingest
and curation never retroactively upgrade an older result to the current
producer-process attestation.

## SCF troubleshooting

Treat a periodic SCF iteration cap as a method or implementation question
until the residual trace proves otherwise. A converged-looking density and a
reasonable energy are not enough for B acceptance. For the current RKS/PBE/RI
diamond-family investigation, inspect `convergence_diagnostics.final_grad_norm`,
`final_delta_e_ha`, and `scf_trace_tail` before changing the Hamiltonian or
declaring an accelerator fix.

The fleet runner records one subtle control interaction explicitly: in the
current periodic GDF loop, static density damping is bypassed once DIIS starts.
A command such as `--damping 0.5` with the default `--diis-start 2` is therefore
not a damping-only experiment after the first DIIS iteration. To isolate
damping, use `--no-diis` or delay DIIS with a larger `--diis-start`; to test
globalized Fock extrapolation, use `--scf-accelerator EDIIS_DIIS` and compare
the residual history. Likewise, inspect the diagnostic Fock-mixing value rather
than inferring the executed accelerator from the requested value alone.

## Audit B, inspect approach status, and compare controls or CRYSTAL

Before making comparison tables, audit either the fetched B JSON records or
the curated B subtree. Raw run filenames and nested `result.json` records are
discovered recursively:

```sh
python studies/aiccm-2026/audit_b.py results-b/
python studies/aiccm-2026/audit_b.py results-b/ --csv audit-b.csv
python studies/aiccm-2026/audit_b.py \
  ../qc-input-library/aiccm2026testset/aiccm2026dev-b/
```

The audit fails if a successful or non-converged record does not declare the
exact D89 identity
`ccm_approach="chi-ccm"`,
`ccm_construction="finite-translation-group-character"`, and
`evaluation_representation="gamma-centred-character-mesh"`; if it does not
exactly match its route, method, backend, and functional declaration against
`b_routes.ROUTES`; if it does not declare `coulomb_kernel="3d-periodic-g0"` and
`exchange_q0="bvk-ewald"`; or if it has `dim != 3` (including missing legacy
dimension metadata). It also verifies the column-vector lattice convention and
the recorded `A_BvK = A_primitive @ diag(character_mesh)` relation. A record
that is `not_converged` or `error` needs an explicit allow flag, and a retracted
record is never numerical comparison data. A successful B record must say
exactly `status="ok"`, `converged=true`, and carry a finite numeric energy;
contradictory terminal statuses fail even when their ordinary status allow flag
is present. D93 additionally requires the exact v1 two-electron-support
payload for every successful or non-converged row. Missing, malformed,
route-inconsistent, internally contradictory, or `not-qualified` support is
an audit failure. D103 independently requires exact overlap-fold evidence and
cutoff agreement for every four-center row. The D102/D104 compatibility gate
then requires the exact post-repair v2 Ewald support bound to finalized
provenance. It rejects every historical v1 row and every missing, malformed,
unqualified, or identity-inconsistent v2 row. Unsupported
and error records need no successful-support assertion.

Unsupported lower-dimensional records and retracted failure evidence remain
visible in the audit table without becoming numerical data. `compare_b.py`
independently refuses retracted or reportable-status lower-dimensional B
records, non-converged or non-finite energies, duplicate system/route records,
route-identity contradictions, and missing or inexact D89 construction
identity. It independently applies the D93 and D103 support gates and the D104
per-record Ewald support gate, which permanently rejects historical D102 v1
evidence, before emitting an absolute-energy table. Select exactly
one basis and character mesh for each
route/functional before comparison rather than allowing one curated record to
overwrite another silently.

The Γ-CCM/χ-CCM approach comparison is always reported as `not-defined`.
There is no current route map from the union-and-weight Γ-CCM construction to
the finite-character χ-CCM construction, so `compare_b.py` never emits an
approach energy or approach delta. The B calculation status remains separate
and is not a comparison verdict.

The comparator has a scaffold for one neutral fitted-torus Fourier control: B
`rhf-ri` versus `aiccm-hf-direct`. D93 currently keeps every fitted B row
`not-qualified`, so this control is not reportable even if its older
provenance checks would pass. A future qualified pair must be finite converged
successes and agree on method, functional, basis, B character mesh versus the
control `nrep`, and normalized `bvk-ewald` exchange-q=0 convention. It must
also supersede the historical `aiccm2026-gamma-chi/v2` numerical-input
contract and carry complete producer attestation. That legacy schema name is
retained only for control provenance; it is not a Γ-CCM construction identity
and is not sufficient for producer emission in its current form.

For the control, each producer must embed the canonical `comparison_input`,
whose exact hash equals `comparison_contract.input_sha256`. The comparator
validates the system, ordered lattice and atoms, method, functional, bases,
mesh, 3-D operator, auxiliary fit, AO and auxiliary thresholds, phase
convention, zero-temperature occupations, clean source, native core, linked
libraries, and composite producer-process attestation. Missing or contradictory
evidence leaves every `real_gamma_control_*` value and the
character-minus-real-Gamma delta explicitly `not-defined`.

The generated B fleet commands use `run.sh --b`, which creates an atomic
private temporary directory for the preflight. In the process holding the
result, each producer always runs a cheap shifted-mesh native-versus-Python
AO-pair Fourier-transform canary. If the current core-path SHA256 differs from
preflight, that process reruns the full v2 API, reciprocal-cutoff, and LiH
direct-versus-GDF checks; passing the cheap canary alone is insufficient. It
re-reads the current identity, linked libraries, and payload immediately before
writing and requires exact stability. The path SHA256 describes bytes currently
at the imported module's filesystem path, not bytes already mapped into the
process. The numerical checks attest loaded behavior without claiming a
cryptographic mapped-image identity. `compare_b.py` independently validates
the nested schemas, digests, check versions, and tolerances. It matches the B
and real-Gamma control records on their current producer identity, not on
preflight history or on their stream-specific payload digests, which
intentionally differ. Raw JSON does not bypass these gates.

The reportable Γ/χ approach map is empty. B `rhf-ri` versus A
`aiccm-hf-direct` remains a neutral-torus representation control, not evidence
for the union-and-weight Γ-CCM approach. The previously mapped A RI and RIJCOSX
routes call
the same multi-k GDF SCF drivers as their B counterparts. Their agreement is a
useful common-path regression check, but it cannot be reported as a Γ/χ
approach delta. The direct Γ-supercell solver independently assembles
and minimizes the fitted Hamiltonian, but deliberately reuses the same per-q
RSGDF fit objects and one-electron lattice-sum primitives. Its comparison checks
the neutral-torus Γ-supercell versus character/Bloch SCF representation, not
the common RI builder or the union-and-weight Γ-CCM construction. The research
WSSC `aiccm-rijcosx`, KS pairs, and all four-center pairs remain excluded.
Approach equality for a specified operator and route remains evidence to
establish under the common exchange-q=0 convention.

No existing curated row satisfies that complete paired contract. In
particular, do not use the 2026-07-08 c-diamond `aiccm-ri` value: a
verified-current rerun shifted it by about 24.5 mHa/atom, so the old row is not
reproducible evidence. Treat the shift as a provenance retraction, not a Γ/χ
result. Verified 2026-07-10 reruns strongly support the shifted-mesh mirror bug
as the explanation for this exact-exchange row: the mirror-insensitive direct
route lands on the post-fix value within 5e-5 Ha/atom, while pure PBE is
unchanged to 3e-13 Ha/atom. The attribution remains an inference because the
old row lacks build attestation and was not rerun on old and fixed cores with
an otherwise identical setup. Those runs predate embedded self-attestation and
the A RI route shares B's multi-k driver, so they still do not qualify as
construction-comparison evidence.

Contract v2 is currently a comparator acceptance schema only. Both runners now
emit symmetric composite producer-process provenance, but neither emits
`comparison_input` or its v2 `comparison_contract`. Under D89 the reportable
approach map is empty; the `rhf-ri` / `aiccm-hf-direct` neutral-torus
representation control is not a Γ-CCM construction comparison. Do not stamp
the current synthetic
`auxiliary_phase_convention`: the legacy real-Gamma control fold uses a
canonical auxiliary-AO frame,
while B self-contracts compact auxiliary eigenmodes. The common contract must
first define the Bloch pair-density phase separately from this representation
gauge and record the factor frames without requiring them to be identical. It
must also replace threshold-only AO equality. For the current control route, every B
character must retain all primitive-cell AO directions: full/no-drop is the
acceptance condition. If projected spaces are allowed later, the producers
must compare Fourier-related retained-space projectors rather than rank counts
alone. A `RUN.meta` file or other sidecar written during later curation cannot
retroactively attest the numerical input, build, host probe, payload, or
producer process that produced an existing energy. Existing rows remain under
the contract they actually recorded.

Fetch the B and real-Gamma control JSON workspaces into separate directories,
or point at their separate curated stream subtrees, then run:

```sh
python studies/aiccm-2026/compare_b.py results-b/ \
  --real-gamma-control-results results-control/
python studies/aiccm-2026/compare_b.py results-b/ \
  --real-gamma-control-results results-control/ \
  --crystal-refs studies/aiccm-2026/crystal_refs_b.json --csv comparison.csv
```

The legacy `--a-results` option remains an alias for
`--real-gamma-control-results`; it does not turn the control records into
Γ-CCM results and should not appear in new instructions.

Copy `crystal_refs_b.template.json` to an untracked results file and populate
only energies from actual CRYSTAL runs. Values are energy per primitive-cell
atom in Hartree and are method-specific (`rhf`, `pbe`, `pbe0`). The `.d12`
provenance for each geometry remains in `testset.py` and every JSON record.
Only compare calculations with the same geometry, orbital basis, functional,
and reciprocal mesh. The cheap default cross-stream pass uses each registry
basis, usually STO-3G; the article/CRYSTAL pass should be emitted explicitly
with `--basis pob-tzvp-rev2` so its JSON records advertise the matched basis.

Rerun any CRYSTAL value produced by `crun.sh` before vibe-qc commit `5b488e57`.
The earlier wrapper passed the wrong input form and could extract null or
garbage-scale energies; the fixed wrapper uses the `INPUT` file and selects the
reported `E(AU)` value precisely. The committed reference file is still only a
null template, so no published numeric anchor is grandfathered through this
correction.
