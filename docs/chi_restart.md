# Experimental chi checkpoint and restart

Persistent restart preserves the executed classic chi source for neutral,
all-electron or [scalar ECP](chi_ecp.md), three-dimensional RI RHF, UHF and
shared-orbital ROHF references.
It also continues canonical MP2, UMP2, Knowles RMP2, RHF/UHF CCSD and the defined
streamed triples correction on that reference. Complete unshifted characters,
integer occupations and the existing method-specific public admissions still
apply. Chi remains experimental; D93 fitted/post-HF remains not-qualified.
Scientific acceptance in #518, #521 and #523 remains separate.

The four states have distinct meanings:

| State | Meaning |
|---|---|
| SCF guess | Density/orbitals are an initial guess; renewed convergence is required. |
| Accepted stationary reference | The loader rebuilds physical Focks and energy from retained factors and verifies the method's stationary occupied spaces before issuing a new process-local seal. |
| Unfinished correlation | Complete work units or CC iterations are saved; the calculation has no completed correlation result yet. |
| Completed result | All required work and convergence checks passed; reloading still checks source and space compatibility. |

A serialized acceptance flag or checksum never establishes scientific
acceptance. Loading an accepted reference does not iterate SCF, evaluate new
integrals, refit factors, change support or retained rank, or substitute another
source. An incompatible accepted reference raises an error; it is never
silently downgraded to a guess.

## Library use

Use the same `system`, `basis`, `mesh`, HF options, auxiliary basis and explicit
classic physical-pair controls on both runs. The following assumes those
inputs have already been defined as in the [fitted-support guide](chi_fitted_support.md).
Place calculation artifacts outside the source checkout.

```python
from vibeqc.periodic.chi.reference import prepare_chi_hf_reference
from vibeqc.periodic.chi.correlation import run_chi_mp2

reference = prepare_chi_hf_reference(
    system, basis, mesh, options,
    method="RHF", backend="ri", fit_source="classic",
    coulomb_controls=controls, aux_basis="def2-SVP-JKFIT",
    checkpoint_path=output_dir / "hf.vqr",
)
result = run_chi_mp2(
    reference, frozen_core=0, auxiliary_batch_size=64,
    checkpoint_path=output_dir / "mp2.vqr", checkpoint_interval=20,
)
```

A subsequent process reconstructs the saved reference and continues the saved
correlation prefix. The original exact HF request is checked before reuse.

```python
reference = prepare_chi_hf_reference(
    system, basis, mesh, options,
    method="RHF", backend="ri", fit_source="classic",
    coulomb_controls=controls, aux_basis="def2-SVP-JKFIT",
    restart_from=output_dir / "hf.vqr",
)
result = run_chi_mp2(
    reference, frozen_core=0, auxiliary_batch_size=64,
    resume_from=output_dir / "mp2.vqr", checkpoint_interval=20,
)
```

`vibeqc.periodic.chi.checkpoint.save_chi_hf_reference(path, reference,
system=system, basis=basis)` also saves an already prepared reference.
`load_chi_hf_reference(path, system=system, basis=basis,
expected_method="RHF", mode="accepted")` validates and reseals one without
calling a driver. Supply `expected_request_identity` when checking an exact
request directly, or let `prepare_chi_hf_reference(restart_from=...)` perform
the normal driver's exact request check. The saved reference must carry the
original complete input binding; an arbitrary historical result object does
not acquire that binding by being passed to the writer.

The explicit alternative `load_chi_hf_reference(..., mode="guess")` returns a
`ChiHFGuess` with `density_k`, `mo_coeffs`, `physical_identity` and
`convergence_required=True`. It is not a `ChiHFReference` and cannot be passed
to correlation as one. This mode still verifies archive integrity and physical
compatibility; it does not salvage damaged data. A caller may explicitly use
its density with a compatible RHF/UHF driver's `initial_density_k` option and
must obtain a new convergence result. The current ROHF driver continues to
refuse initial-density guesses. The public persistent-reference path below
loads accepted references only and has no automatic guess fallback.

`run_chi_ccsd` and the developer `evaluate_same_source_mp2`,
`evaluate_same_source_rohf_mp2` and `evaluate_same_source_cc` entries expose
the same `checkpoint_path`, `resume_from` and positive integer
`checkpoint_interval` arguments. CC equation controls remain in
`ChiCCOptions`. When only `resume_from` is supplied, new complete checkpoints
replace that same archive. Supplying `checkpoint_path` writes resumed progress
to the new destination while retaining the original archive.

## Public single-point jobs

Use the existing `run_periodic_job(method="aiccm", variant="chi", ...)`
route with `aiccm_backend="ri"`, `fit_source="classic"`, explicit
`chi_coulomb_controls`, and the established tight HF thresholds. The explicit
path options are:

| Keyword | Action |
|---|---|
| `chi_reference_checkpoint` | Save the accepted HF reference and its retained factors after SCF or validated reuse. |
| `chi_reference_from` | Load and validate an accepted HF reference before the exact request check. |
| `chi_correlation_checkpoint` | Publish correlation iterations/work to this archive. |
| `chi_correlation_resume` | Continue correlation from this archive, using the separately supplied saved or sealed HF reference. |
| `chi_checkpoint_interval` | Publish after this many complete CC iterations or correlation work units; defaults to 1. |

For example, add `chi_reference_checkpoint=output_dir / "hf.vqr"` and
`chi_correlation_checkpoint=output_dir / "corr.vqr"` to a fresh admitted job.
On the subsequent identical job, use `chi_reference_from` and
`chi_correlation_resume` for those paths. A process-local `chi_hf_reference`
is also valid for correlation resume, but cannot accompany
`chi_reference_from`. HF and correlation archives require distinct paths.
The [ROHF/RMP2 example](../examples/periodic/chi_rohf_rmp2.py) exposes
`--checkpoint` and `--resume` for separate-process runs.
Completion is published even when the final work count is not an interval
multiple; CC also publishes an iteration-limit boundary before raising.

These options do not replace the generic QVF `read_from`/`restart_from`
mechanism and cannot be combined with it, geometry optimization or cell
optimization. A nondefault interval requires an explicit checkpoint/resume
path. Omitting the new options preserves historical behavior.

The `.out` and `.system` records distinguish a fresh HF state, reused
process-local reference, restored validated reference, resumed correlation,
and restored completed correlation result. They report actual total CC
iterations, resumed work, complete coverage and the DIIS reuse policy.
Reconstructing a stationary HF reference reports zero new SCF iterations.
The library guess object explicitly reports that convergence is required.
Checkpoint filesystem paths are not physical provenance.

## Portable identity and storage contract

The conventional suffix is `.vqr`; the writer uses the exact requested path
without appending an extension. The shared `vibeqc.output.checkpoint` writer
uses the atomic publication convention of the existing output infrastructure.
A restart archive is a ZIP64-capable, uncompressed container with
`manifest.json` and raw numeric `arrays/<name>` entries. It is not a QVF
visualization file. Small metadata uses `vibeqc.restart.archive/v1`.
Metadata larger than one MiB uses the explicitly supported
`vibeqc.restart.archive/v2` envelope: the complete finite JSON is stored in
`metadata.json`, with its byte count and SHA-256 in the manifest. Executed
source and support records are preserved exactly, including large complex
character calculations. The manifest remains limited to 32 MiB. The reader
checks the uncompressed member's actual byte count and checksum before JSON
decoding; both supported envelopes return the same method state.
The manifest records each array's shape, dtype, byte count and SHA-256 checksum,
and a checksum of its own canonical finite JSON payload. Plain numeric
boolean, integer, floating and complex arrays are accepted; object arrays,
executable deserialization and pickle are not used.

HF uses `vibeqc.chi.hf-restart/v1` for all-electron states and
`vibeqc.chi.hf-restart/v2` for scalar ECP states. ECP v2 adds complete primitive
parameters, effective charges/counts and retained one-electron component
arrays. Changed-ECP inputs and a v2 state relabeled v1 refuse. The correlation
method schemas remain
`vibeqc.chi.canonical-mp2-restart/v1` and `vibeqc.chi.canonical-cc/v1`.
Unknown envelope or method versions, incompatible descriptors, duplicate or
missing members, truncated arrays, nonfinite values and checksum mismatches
raise errors. Existing version-one archives remain directly readable; neither
supported envelope changes the physical identity or migrates a method schema.
A future format change
requires an explicit supported migration; changing a schema string manually
is not a migration.

HF storage includes geometry/cell, charge and spin, normalized orbital shells
and basis identity, the all-electron or explicit ECP electron convention,
complete characters and
weights, one-electron and nuclear data, fitted factors, auxiliary metric
spectra/retained ranks, kernel/q=0/support records, orbitals, energies,
occupations, physical Focks, method/canonicalization and numerical controls.
The loader recomputes the request digest from the archived inputs and checks
requested controls against the executed source and support records. Versions
one and two require portable basis labels and refuse filesystem paths in those
labels instead of rewriting the source identity. Loading checks dimensions,
electron/spin counts, metric orthonormality,
projectors and the Fock/energy reconstruction. ROHF additionally preserves
its shared closed/open/virtual spaces and generalized stationarity; it is
never reconstructed from independently optimized UHF orbitals.

`portable_reference_identity` binds this physical state and executed source.
Process IDs, current runtime telemetry, timing and paths do not define that
identity. Restoring in another process issues a new process-local seal.
Correlation separately binds canonical energies/frames, active and frozen
spaces, retained ranks, momentum ordering, numerical controls and the
reference identity. Scientific runtime provenance remains in normal output;
a portable checksum is an integrity check, not a reference calculation or a
proof of physical accuracy.

Publication writes a temporary sibling, flushes and synchronizes it, then
atomically replaces the destination. A failed or interrupted write leaves the
last complete destination available; a leftover temporary file is never
loaded as a checkpoint. Requested checkpoint I/O failures propagate rather
than reporting unsaved progress as successful. Numeric read/write staging
uses at most one MiB panels. Readers allocate the retained numeric arrays,
and the reference factors, canonical frames, CC amplitudes, DIIS history,
checkpoint snapshots, metadata and library overhead still require memory.
The loader temporarily holds decoded factors alongside the operator's immutable
copies, with an additional factor byte copy while computing its identity.
Physical validation also retains densities, Focks and J/K arrays. This is not
a constant bound on total process memory or a distributed cache.

## Deterministic correlation continuation

MP2/UMP2 saves a deterministic prefix over spin sectors, character triples
and occupied pairs, with separate aa/bb/ab accumulators and exact component
coverage counters. Knowles RMP2 first records alpha/beta singles as separate
units, then its doubles; its shared frozen projector and spin-semicanonical
conventions are bound to the archive. Resume skips exactly the completed
prefix and applies the existing `Nk^-3` doubles and `Nk^-1` singles
normalizations once. Incomplete coverage cannot be a completed result.

CCSD saves the next amplitudes at an unfinished iteration boundary, the
cumulative iteration number and convergence trace, preceding energy, and up
to eight compatible DIIS vectors/errors. It validates their dimensions,
finiteness, antisymmetry and spin/momentum spaces on resume. DIIS history is
reused and reported as `reused-compatible-history`; it is not silently reset.
`max_iter` may increase to continue iteration-limited work. Tolerances,
denominator control, frozen/canonical spaces, auxiliary batch size and the
other bound solver controls must match. A completed CC state is rechecked
against its physical residual, energy and convergence thresholds.

Triples saves converged CC amplitudes alongside a partial complex sum and the
exact lexicographic prefix of unique occupied/virtual triples. Its coverage
includes forbidden zero-contribution units; counts and a rolling prefix
digest verify that skipped work is neither omitted nor repeated. A partial
sum is never returned or labelled as a completed triples correction. Final
coverage and the existing finite, real-energy checks must pass. The specified
CC and triples equations remain those in the [CC guide](chi_ccsd.md).

## Scope and verification

Persistent restart currently covers the classic HF and canonical correlation
routes above, including classic scalar ECPs. It does not add SR restart,
spin-orbit ECPs, KS/hybrid correlation,
fractional occupations, ROHF-CC, changed-source continuation, distributed
checkpointing or new guess support. Existing SR metric/operator and public
frozen-space restrictions remain. The chi/CCM construction firewall is
unchanged; a neutral CCM result cannot be substituted for a chi reference.

Focused source checks cover separate-process HF/solver round trips,
interrupted publication, corruption and incompatible inputs, mutation,
complex characters, ROHF shared spaces, and uninterrupted versus resumed
correlation components, CC amplitudes/residuals and complete triples sums.
Independent finite-space fixtures check the equations and coverage. Resume
preserves the defined work order; cross-process/platform comparisons use the
method's existing numerical criteria rather than a general promise of bitwise
equality. The reference rebuild requires energy agreement within `1e-11` Ha
and Fock/projector residuals within `1e-9`; CC uses its unchanged explicit
energy/residual tolerances. These checks do not constitute native scientific
acceptance. Complete `vq` restart
calculations and comparisons belong to Validator; their proposed cases and
runtime/source evidence are recorded once on the increment's tracking issue.
