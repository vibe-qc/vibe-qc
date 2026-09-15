# Finite-torus route-control campaign: 1D / 2D / 3D on compute-managed (2026-07-15)

One test system per dimensionality, one pinned numerical realization, three
routes. The campaign organizes a neutral-torus real-Gamma control, a neutral
Bloch/GDF control, and a one-sided χ-CCM finite-character route diagnostic
under the fingerprint rule (theory doc §14.4). Under D89/D90, neither control is bound to the
union-and-weight Γ-CCM construction, so this campaign defines no Γ-CCM/χ-CCM
approach delta. Issued by the theory chat at the maintainer's request and
reclassified by the χ-CCM owner before results were admitted.

## 0. Non-negotiable ground rules

1. **All three cells are declared fully three-dimensional** (`dim=3`,
   periodic-in-vacuum for the chain and the monolayer). Declared 1D/2D cells
   fail closed in χ-CCM and are barred cutoff/padding artifacts in the -a
   routes (theory doc §14.3). Do not "fix" this by declaring low dimensions.
   For the embedded objects, `declared_model="3d-periodic-in-vacuum"`
   describes the vacuum-padded geometry while `boundary_model="3d-periodic"`
   selects the fully 3D periodic Green function. The fields are intentional,
   not an isolated wire/slab exception to D72/O1.
2. **Current main only.** The build must postdate D88 (column-vector lattice
   repair), 85b5afe5 (column-norm bound in all three RSGDF mesh builders),
   e578b86c (centered shifted-pair Ewald enumeration), and the COSX one-center
   correction. Record the fingerprints of §14.4:
   clean full source SHA, native-core SHA256, versioned probe pass. A run
   from a host package that lags main is not reportable. Validate ONE job
   (System 3D below) on compute-managed and check its fingerprint before submitting
   the rest (host version skew is a known hazard).
3. **Same Hamiltonian convention everywhere**: neutral 3-D torus kernel,
   `exxdiv="ewald"`, and the recorded `exchange_q0_applicability` must read
   `active` (RHF). A matching descriptor is necessary but does not by itself
   prove that two route skeletons realize an identical discretized Hamiltonian.
4. **No silent parameter substitution.** If your route cannot honor a pinned
   parameter exactly, stop and report which one, per the fail-closed culture.
   Do not run a nearby variant.
5. **Submission**: via `vq submit --host compute-managed` with an explicit
   `--job-name` per the recipe table in vibe-queue docs. Never hand-edit the
   managed checkouts. Payload carries your submitted source.

## 1. Test systems (geometry pinned in bohr, column-vector lattice)

### System 1D: LiH chain, declared 3-D periodic-in-vacuum
- Lattice (columns): a1 = (6.0000, 0, 0), a2 = (0, 30.0000, 0),
  a3 = (0, 0, 30.0000)
- Atoms: Li (0.0000, 15.0000, 15.0000), H (3.0000, 15.0000, 15.0000)
- `PeriodicSystem(3, ...)`, charge 0, closed shell (4 e per cell)
- BvK mesh / lattice extension: (8, 1, 1)

### System 2D: h-BN monolayer, declared 3-D periodic-in-vacuum
- a = 2.504 Angstrom. Lattice (columns): a1 = (4.7319, 0, 0),
  a2 = (-2.3660, 4.0979, 0), a3 = (0, 0, 30.0000)
- Atoms: B (0.0000, 2.7319, 15.0000), N (2.3659, 1.3660, 15.0000)
- `PeriodicSystem(3, ...)`, closed shell (12 e per cell)
- BvK mesh / lattice extension: (3, 3, 1)
- Note: the lattice matrix is non-orthogonal by design. This deliberately
  exercises the post-753b7ec4/85b5afe5 reciprocal boxes and the post-D88
  lattice convention. That is a feature of the test, not an accident.

### System 3D: LiH rocksalt, fcc primitive cell
- a = 4.083 Angstrom. Lattice (columns): a1 = (0, 3.8579, 3.8579),
  a2 = (3.8579, 0, 3.8579), a3 = (3.8579, 3.8579, 0)
- Atoms: Li (0.0000, 0.0000, 0.0000), H (3.8579, 3.8579, 3.8579)
- `PeriodicSystem(3, ...)`, closed shell (4 e per cell)
- BvK mesh / lattice extension: (2, 2, 2)
- The symmetric lattice commutes with diag(2,2,2), so this system is outside
  the D88 numerical defect class and doubles as the compute-managed validation job.

## 2. Pinned numerical realization (identical for every route)

- Method: **RHF** only. No DFT, no post-HF, no smearing
  (`smearing_temperature = 0.0`).
- Basis: **STO-3G** on every atom.
- Auxiliary basis (RI routes): the route's documented default, but the
  resolved auxiliary basis NAME must be recorded and must be identical
  between the neutral Bloch/GDF run and the χ-CCM RI run for the same system.
- `rsgdf_ke_cutoff = 200.0` (record the configured/source-resolved value and
  the B-selector-resolved value forwarded into the fitted driver when exposed;
  this is not backend-owned telemetry or D93 support qualification).
- Convergence: `conv_tol_energy = 1e-10` Ha, `max_iter = 128`.
- Converger: DIIS on, subspace 8. `damping = 0.0`. Level shift 0.0.
  `fock_mixing = 0.0` requested, and the EXECUTED value recorded per D86
  must also be 0.0. If your route resolves a different executed value,
  that is a stop-and-report condition (rule 0.4).
- Linear dependence: AO threshold 1e-7, auxiliary metric threshold 1e-9
  (the D74 v2 values), recorded separately as source-resolved route defaults
  when the result object does not expose them.
- The original request did not pin an initial guess. Record requested and
  executed semantics for every route. The χ-RI runner explicitly requests
  `HCORE`, matching the fixed multi-k GDF initialization; convergence-trajectory
  or iteration-count comparisons remain undefined unless the other routes bind
  the same choice.
- Exchange q=0: `exxdiv="ewald"`, applicability `active`.

## 3. Route assignments

| Route | Owner chat | Entry point |
|---|---|---|
| neutral-torus real-Gamma control | -a direct-torus chat | `run_ccm_rhf_direct(ccm, exxdiv="ewald")` |
| neutral Bloch/GDF control | -a production chat | `run_ccm_rhf_gdf(ccm)` |
| χ-CCM (finite-character RI) | -b chat | `run_periodic_job(..., jk_method="aiccm2026dev-b", aiccm_backend="ri")` |

Each chat runs all three systems on its own route. This defines six possible
control numbers plus three nonquantitative χ diagnostic receipts, not nine
reportable numbers. Numerical-support and producer-attestation gates still
decide whether either control route may contribute a number.

The χ route has the dedicated D101 producer `run_case_cmp_b.py`, selected by
`run.sh --comparison-b`. It binds the exact section-1 and section-2 inputs,
calls only the high-level B entry point shown above, and requires System 3D to
complete its internal contract audit first. Each later embedded-object run
must consume the exact fetched contract-valid nonquantitative 3D JSON as
`--validation-record`; its file digest, vq job, Slurm job, and D77
producer-attestation identity become part of the later record. All three
systems remain declared `dim=3`, so the B low-dimensional fail-close must not
fire.

The current compute-managed managed runtime cannot satisfy D101. Once the interpreter
selected by `run.sh` (its managed wrapper or an explicit `VIBEQC_PYTHON`)
resolves to a current D77-clean Git checkout, the first D101 submission recipe
is:

```bash
git fetch origin
SHA=$(git rev-parse origin/main)
vq submit --host compute-managed --job-name d101-chi-3d-ri -d studies/aiccm-2026/ -- \
  bash run.sh --comparison-b 3d --expected-source-sha "$SHA" \
  --vq-host compute-managed
```

The entry-point column names the function, not a literal bare call. Every
route MUST pass the section-2 pins explicitly (the bare defaults differ, for
example `PeriodicRHFOptions()` carries damping 0.5, conv 1e-8, max_iter 100)
and must confirm an empty pin-violation list in its report. A bare-default
run is not a campaign row.

Walltime, partition, node count, and thread count on compute-managed are NOT pinned
parameters. Pick whatever compute-managed partition fits the job (the 1D vacuum-cell
mesh is large, use a longer partition rather than the devel cap). If no
compute-managed partition fits at all, escalate to the maintainer, do not move hosts
unilaterally.

## 4. Reporting format and qualification

Report into your drop-box/handover and back to the theory chat. For a route
whose numerical support is independently qualified, include:

- E_total per cell in Ha, full double precision, and E per atom;
- Iteration count, converged flag, executed fock_mixing, resolved
  aux basis, resolved ke cutoff, lindep thresholds;
- Convention record: exchange_q0 and exchange_q0_applicability;
- Fingerprints: source SHA (clean tree), native-core SHA256, probe pass id,
  vq job id, host (compute-managed), date; and
- Wall time and peak memory if the queue reports them.

D101 is deliberately different. D93 leaves its fitted RI support
`not-qualified`, so its campaign JSON contains neither E_total nor per-atom
energy. The producer audits total/electronic/nuclear assembly internally and
records only whether that audit passed. Its successful status must always be
read as the complete tuple
`status="ok"`, `evidence_role="route-plumbing-diagnostic"`,
`quantitative_status="not-qualified"`, and
`comparison_status="no-gamma-chi-construction-comparison-defined"`.
`audit_cmp_b.py` revalidates this tuple and the nested evidence without
printing an energy. Auditing a `1d` or `2d` file requires the exact fetched 3D
JSON through `--validation-record`, so the auditor can recompute its file SHA
and parent identifiers. Ordinary runner and SCF-log energies are nonreportable
diagnostic artifacts. Do not recover or copy a χ energy from `.out`, `.system`,
the progress log, or any other runner side artifact: the D101 JSON is the only
campaign receipt for this route.
Its Madelung and nuclear checks are separate recomputations through the same
shared helpers used by SCF. They are dispatch and assembly evidence, not
independent kernel or Ewald implementations.
Historical commit `a8b9ac2f8` records the 3D Ewald shifted-pair cutoff defect
in that shared nuclear helper. Commit `e578b86c` repairs the image enumeration
with centered displacement and interplanar pair bounds. The repair does not
retroactively qualify an affected absolute-energy record or any stored D102-v1
object; a fresh quantitative row needs its own applicable D104-v2 evidence.
The same-helper equality still cannot test convergence. The energy-free D101
diagnostic remains intact but nonquantitative because its fitted RI support is
D93 `not-qualified`.

Attestation ruling (theory chat, 2026-07-16, interpreting theory doc 14.4):
for an immutable-bundle deployment on compute-managed, where the D77 git-identity check
is structurally unsatisfiable, a bundle attestation can satisfy the
validated-producer-build layer only when its schema binds the artifact to a
full source SHA proven resolvable on `origin/main`, independently verifies the
artifact and native-core checksums, runs the full in-process v2 numerical
checks and loaded-core canary, binds the producer payload and native-library
identities, rechecks every identity layer before serialization, and records
`attestation="bundle"` rather than `"git-checkout"`. D91 codifies this
normative contract, but its transport/producer/auditor implementation remains
prospective. The current B producer, curator, and comparator implement only D77
clean-checkout attestation, so no B bundle campaign row is reportable yet. A
no-rerun ruling can apply only to a row already carrying and passing the
complete schema; no current B campaign row does.
The current compute-managed managed development runtime is both an immutable bundle and
behind `origin/main`. D101 rejects that transport even after a routine refresh
to the correct SHA because the D91 implementation is absent. Do not submit
D101 until the interpreter selected by `run.sh` resolves to a separate
D77-clean Git-checkout runtime or the D91 implementation lands.
`--vq-host compute-managed` remains an operator assertion. A numeric `SLURM_JOB_ID`
proves Slurm scheduler context, but vq currently exports no authoritative
target-alias field, so those two values do not cryptographically prove host
identity.

The exchange-q=0 convention record may be DERIVED from executed evidence
(routing log, nonzero BvK Madelung constant, exact-exchange fraction) on
routes that do not yet stamp it as a result field, with the derivation mode
recorded and fail-closed when evidence is missing. The χ route reads its D83
operator-coefficient-derived descriptor directly. Commit `3c6863d5` added
fields to the neutral Bloch/GDF wrapper, but that wrapper still infers hybrid
applicability from `e_hf_exchange`; an energy is not the live full-range
operator coefficient and can vanish while the arm remains present. RHF is
unambiguous in that wrapper because the route sets applicability active
directly, but the A-side general KS field still needs the lower driver to
expose the executed full-range coefficient or seam state before it is described
as a native D83 implementation. The wrapper also writes an empty
`exchange_q0` when applicability is inactive, whereas D83 retains the
`bvk-ewald` family label and marks only its applicability inactive. Both the
live coefficient and persistent family label need repair on the A-side KS
wrapper; neither changes the native χ descriptor used in this campaign.

## 5. What the comparison can and cannot claim (theory-chat framing)

- This route set contains no implementation explicitly bound to the
  union-and-weight Γ-CCM construction. It therefore cannot test Γ-CCM/χ-CCM
  construction coincidence. A character-versus-real-Gamma comparison can be
  admitted only as a representation control for one already specified
  neutral-torus Hamiltonian, while neutral Bloch/GDF versus χ-RI is largely a
  common-path regression because both use the shared multi-k GDF driver.
  Neither validates the shared kernel; the external KRHF anchor remains
  separate.
- D101 emits neither the historical v2 `comparison_input` nor a
  `comparison_contract`. No matching descriptor, route label, or shared driver
  ancestry proves identical discretized Hamiltonians, and this campaign has no
  Γ/χ construction-comparison field to interpret.
- Historical planning magnitudes are direct vs GDF at the
  RI-fitting error (about 5e-5 Ha/atom covalent, up to about 8e-4 Ha/atom
  ionic), and neutral Bloch/GDF versus χ-RI at the common-path agreement floor
  when both sides share one realization. These are diagnostic priors, not
  acceptance bands or fingerprinted construction evidence. D101 cannot test
  them while D93 holds; do not tune any route toward them.
- These are declared 3-D periodic-in-vacuum models for the chain and the
  monolayer. No isolated-wire or isolated-slab accuracy claim follows from
  them, and none may be made (theory doc §14.3).
