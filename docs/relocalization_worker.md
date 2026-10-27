# Standalone relocalization worker

vibe-qc owns the localization process. The consumer needs no import of vibe-qc
and the backend environment needs no installation of vibe-view. Protocol **1**,
worker API **1.0.0**, is **new in the 0.17.5 release**.
An existing `0.17.4` or earlier release installation does not
imply that it contains this worker. Pin the implementing commit and negotiate
`protocol_version`, `worker_version` and live capabilities: a reported package
version alone does not prove that this worker is installed. Minimum worker
version is `1.0.0`, protocol version exactly `1`.

## Install and run

Use Python 3.11 or newer and a full vibe-qc installation, including the compiled
`_vibeqc_core`, native libraries and bundled basis data (including Huzinaga MINI).
The ordinary package dependencies suffice. No viewer, PySCF, GPU or extra Python
localization dependency is needed. See [installation](installation.md) for
platform prerequisites. In a checkout containing this implementation:

```sh
./scripts/install.sh --current --extras none --venv .venv
.venv/bin/python -m vibeqc_relocalize --probe
```

**Wire requests are one compact JSON object on one line.** The checked-in JSON
fixtures are indented for review, so compact them before feeding the worker:

```sh
.venv/bin/python -c 'import json,sys; print(json.dumps(json.load(sys.stdin)))' \
  < tests/data/relocalize/h2.request.json | .venv/bin/python -m vibeqc_relocalize
```

For a client, the exact
argv is `[backend_python, "-m", "vibeqc_relocalize"]`, and the probe argv is
`[backend_python, "-m", "vibeqc_relocalize", "--probe"]`. The installed console
script `vibe-qc-relocalize` accepts the same arguments. Store the absolute Python
executable path globally in the consumer and launch without a shell. A backend
console-script path is also usable if the consumer omits `-m vibeqc_relocalize`.

The lightweight `vibeqc_relocalize` package is deliberately outside `vibeqc`:
`python -m vibeqc.relocalize` would import `vibeqc.__init__` before it could report
a missing native core. The supported launcher catches that failure and returns
a structured capability report. A missing launcher itself is a process-launch
failure, which the consumer must explain.

For source development, use an isolated venv and rebuild after native changes:

```sh
VIBEQC_REQUIRE_VENDORED=ON .venv/bin/pip install -e . --no-build-isolation
```

Do not run an installer that switches branches when pinning a particular commit;
bootstrap prerequisites as described in the contributor setup, then use the pip
command above on that checked-out commit.

Keep the backend's native installation tree in place. The current CMake build
records paths to its native dependency libraries; building a wheel does not
bundle those libraries into a portable runtime. A wheel installed into another
venv on the same host can use that native installation, but copying the wheel
alone to another machine is not a supported deployment procedure. Install
vibe-qc with its native prerequisites on the destination and probe there.

## Transport and capability negotiation

One request and one terminal event per process. UTF-8 JSONL on stdin/stdout;
max request 16 MiB. Write the request, newline, and close stdin. The worker does
not wait for EOF after that line. Additional lines are not additional jobs;
start another process. Diagnostics, Python warnings and native fd 1 output are
redirected to stderr. Drain both streams concurrently. The caller owns timeout
and cancellation (terminate the process, discard incomplete output). No signal
or native crash can guarantee a terminal event.

Every event contains `protocol`, `protocol_version`, `worker_version`, `id`,
`event`. Normal localization emits `started` (`stage="validating"`) then exactly
one `result` or `error`. A capabilities request emits only `capabilities`.
`id` is an opaque string of 1 to 128 characters; it is echoed after envelope
validation, otherwise null. Unknown input fields, duplicate keys and non-finite
numbers are errors. UTF-16/32 input is rejected; the transport is strictly
UTF-8. Numeric arrays reject boolean and string elements, including arrays
mixing them with numbers; there is no boolean-to-number coercion. Do not
interpret stderr as a protocol channel.

```json
{"protocol":"vibeqc.relocalize","protocol_version":1,"id":"probe-1","operation":"capabilities"}
```

`--probe` does the same without stdin and uses null id. The report includes
backend distribution version, `native_core_ready`, per-method `ready`,
`fresh_rhf.ready`, supported dimensions/spins/coefficient types/k-point classes,
required metadata and resource limits. It actually builds a Gaussian basis,
computes native overlap/dipole/reference integrals, exercises all three
molecular localizers with two occupied orbitals, solves a small RHF witness,
and exercises the periodic complex native Fourier/projector path. Per-method
failure disables that method with `readiness_error`. The periodic witness is a
small synthetic metric problem, not an accuracy qualification of periodic SCF.
`native_core_ready` certifies the native integral witness; use each method's
`ready` for its additional kernels. A functioning import alone is insufficient.

Exit codes: `0` successful result or at least one ready method on probe; `2`
structured request/refusal error; `3` computational failure or no ready method.
Inspect the terminal event as well as the exit code. A partial probe may return
0 with some methods unavailable; never enable a method solely from global
`ready`. Reprobe after backend path or installation changes.

```json
{"protocol":"vibeqc.relocalize","protocol_version":1,"worker_version":"1.0.0","id":"job-1","event":"error","error":{"code":"invalid_source","message":"supplied requires orbitals; fresh_rhf forbids them (no fallback)","field":"orbitals"}}
```

Stable error codes include `invalid_json`, `invalid_request`,
`unsupported_protocol`, `unsupported_operation`, `request_too_large`,
`backend_unavailable`, `unsupported_method`, `unsupported_basis`,
`incomplete_basis`, `unsupported_spin`, `unsupported_occupations`,
`unsupported_complex`, `unsupported_source`, `invalid_source`,
`unsupported_periodic_method`, `experimental_opt_in_required`,
`unsupported_periodic_representation`, `unsupported_pbc`, `unsupported_kmesh`,
`unsupported_complex_metric`, `incomplete_band_manifold`,
`unsupported_metallic_manifold`, `nonorthonormal_input`, `resource_limit`,
`scf_not_converged`, `numerical_validation_failed`, `computation_failed`.
`field` can be null; message text is diagnostic, not a stable API.

## Supported worker methods

| Method | Molecular real | Molecular complex | Periodic Gamma | Complete cyclic k mesh | Spin/occupation |
|---|---|---|---|---|---|
| `ibo` | Yes | No | No | No | Restricted singlet, all occupied 2 |
| `boys` | Yes | No | No | No | Restricted singlet, all occupied 2 |
| `pipek-mezey` | Yes | No | No | No | Restricted singlet, all occupied 2 |
| `aiccm-wannier` | No | No | Real and complex, experimental | Real and complex, experimental finite-BvK only | Restricted singlet, all occupied 2 |

No worker route supports UHF/UKS, ROHF, spinors, noncollinear spin, ECPs,
fractional occupations or metallic disentanglement. All three molecular
methods describe populations and charges with MINI IAOs, so all share its
all-electron and Z <= 86 limits even when their objective is Boys or Mulliken PM.
These methods are different criteria, not interchangeable periodic aliases.

The periodic route requires `allow_experimental: true`, dimension 1, 2 or 3,
periodic axes first, a full unshifted cyclic mesh with uniform weights, an
isolated occupied manifold, and a time-reversal-compatible **overlap**. It
preserves genuinely complex coefficients, including at Gamma. It does not
require making each orbital real. It does not accept arbitrary infinite-crystal,
shifted, irreducible or band-path samples. Gamma alone cannot stand in for an
archive containing a full k mesh. Finite-torus localization is not a general
Marzari-Vanderbilt/Wannier90 interpolation or disentanglement implementation.

## Request contract

All lengths are **bohr**, primitive exponents **bohr^-2**, band energies
**hartree**, atomic charges **electron-charge units**. Positions are Cartesian.
There is no implicit unit conversion and there are no inferred occupations.
`source` defaults to `supplied`; only `source="fresh_rhf"` runs a new RHF solve
(maximum 100 SCF iterations, molecular only). `fresh_rhf` forbids `orbitals`.
A bad supplied subspace is rejected; it is never orthogonalized or replaced.

The envelope requires `protocol`, integer `protocol_version=1`, string `id`,
`operation="localize"`, `method`, `system`, `basis`. Optional keys are `source`,
`orbitals`, `periodic`, `allow_experimental`. Nested objects also reject unknown
keys. The reviewable, executable examples are in `tests/data/relocalize/`.

| Field | Definition |
|---|---|
| `system.kind` | `molecular` or `periodic`; periodic object required exactly for periodic |
| `system.atomic_numbers` | Integer list `[natom]`, 1..118, no ghosts; 1..256 atoms |
| `system.positions_bohr` | `[natom,3]`, finite real values |
| `system.charge` | Integer total charge per molecule or primitive cell |
| `system.multiplicity`, `system.spin` | Exactly `1`, `restricted` |
| `basis.ao_convention` | Exactly `qvf-gto-v1` |
| `basis.uses_ecp` | Required, exactly false; caller must verify provenance |
| `basis.name` | Optional label; used to load a named basis only for fresh RHF without shells |
| `basis.shells` | Ordered contracted-shell list, required for supplied orbitals |
| shell `center`, `l`, `pure` | Zero-based atom index; integer angular momentum 0..6; boolean spherical/Cartesian |
| shell `exponents`, `coefficients` | Same-length nonempty primitive lists, positive exponents; QVF Appendix A.1 raw coefficients |
| `orbitals.coefficients` | Encoded array, row per occupied orbital |
| `orbitals.occupations` | `[nocc]` molecular; `[nk,nocc]` periodic, every value exactly 2 |

The entire occupied subspace is required: `nocc=(sum(Z)-charge)/2`, positive
and no greater than `nao`. No virtual coefficient rows are accepted. Molecular
coefficient shape is `[nocc,nao]`; periodic shape is `[nk,nocc,nao]`. Gamma still
has a leading `nk=1` axis. There are at most 512 AOs (and 128 total AOs in a
periodic torus), 512 shells and 128 primitives per shell.

Encoded arrays are objects `{"encoding":"real","data":[...]}` or
`{"encoding":"complex_split_last_axis","data":[...]}`. Complex encoding adds
a final axis of length 2 containing `[real,imag]`, as in QVF Bloch members.
No complex-to-real cast occurs. Molecular requests reject the complex encoding
even when every imaginary component is zero; the consumer may explicitly
choose real encoding only if it has proved the data are real.

### Exact AO convention

Shell list order is AO block order. Within a spherical shell, use libint
STANDARD real solid harmonics `m=-l,...,+l`; a p shell is `y,z,x`. Cartesian
order is descending `lx`, then descending `ly`, with `lz=l-lx-ly`. Per-shell
`pure` is authoritative, including mixed spherical/Cartesian bases.

For every primitive the worker multiplies archived raw coefficient `c` by
`N=(2 alpha/pi)^(3/4) (4 alpha)^(l/2) / sqrt((2l-1)!!)`, then constructs the
native basis with pre-normalized coefficients. It applies no additional
contraction normalization, spherical permutation, or Cartesian component
correction. This exactly reverses `output.formats.qvf._basis_shell_payload`.
The shell center is the corresponding system atom position. Diffuse,
Cartesian, custom and mixed bases keep their archived coefficients; a basis
name alone cannot establish their identity. Very ill-conditioned AO metrics
are refused (smallest eigenvalue must exceed both 1e-14 and 1e-10 times the
largest). Input `C^H S C` must agree with identity to Frobenius error < 1e-7.

### Periodic data

The `periodic` object requires all of:

| Field | Definition |
|---|---|
| `representation` | `finite_bvk`; an assertion of the source Hamiltonian/metric representation, not a conversion request |
| `lattice_bohr` | `[3,3]`, primitive lattice vectors in **columns**, invertible including inactive embedding axes |
| `dimension`, `pbc` | 1..3; `[true,false,false]`, `[true,true,false]` or `[true,true,true]` |
| `mesh` | Three positive integers; inactive axes equal 1; product `nk` |
| `kpoints_fractional` | `[nk,3]`, fractional reciprocal coordinates in `[0,1)`, lexicographic `ix/nx,iy/ny,iz/nz` order, z fastest |
| `weights` | `[nk]`, each exactly `1/nk` within 1e-12 |
| `bloch_convention` | `exp(+2pi*i*k.R)`; lattice translation phase, no extra atom-position phase |
| `overlap` | Encoded `[nk,nao,nao]` Hermitian positive-definite S(k), in the identical AO order/normalization and phase convention as C(k) |
| `orbital_energies_hartree` | `[nk,nband]` sorted reference energies, `nocc < nband <= nao`; must include at least the first virtual band |

The indirect reference gap must exceed 1e-6 hartree. Energies certify the
**declared source** occupied manifold; the worker does not rebuild the Fock
operator to authenticate them. It likewise trusts the caller's basis and
finite-BvK provenance. Orthonormality alone cannot authenticate AO ordering or
the Hamiltonian. `S(-k)=conj(S(k))` is required within 1e-10, including a real
Gamma metric. The existing native transform additionally refuses imaginary
residue in the real-space metric. Supplied C may remain genuinely complex.

No Fock, density, ERI or SCF restart object is needed for this worker route.
Fock matrices would enable an additional one-particle-energy check, but are
not accepted in protocol 1. The occupied projector checks do not constitute an
independent total-energy calculation. For IAO-based periodic methods, matched
minimal-basis S22(k), target/reference S12(k), atom maps and Gaussian/image
provenance are additionally needed; those methods are not exposed here.

## Result contract

`event="result"` carries a `result` object. Its coefficients always have rows
as orbitals. It includes `method`, `source`, `scf_performed`, backend version,
AO convention, spin, occupations, encoded coefficients and encoded rotation,
atom populations, charges, `n_centres`, `centre_threshold=0.1`, centroids and
explicit descriptor model names. There are **no orbital energies** for the
localized mixture. `citation_features` names the existing feature routes in
`vibeqc.output.citations/database.toml` (IBO, Foster-Boys, PM or Wannier); the
periodic route retains its explicit approximation label. `convergence="not_certified_by_kernel"` is intentional:
the existing Jacobi interfaces do not expose a common convergence certificate.
A result certifies the stated numerical invariants, not a global localization
maximum or attainment of the iteration stopping threshold.

Molecular `representation="molecular_ao"`:

- Coefficients `[nocc,nao]`, occupations `[nocc]`, rotation `[nocc,nocc]`.
  Internally, AO-by-orbital matrices obey `C_after=C_before @ rotation`.
- `atom_populations[nocc,natom]` are probabilities in the MINI IAO partition;
  rows sum to 1. All three methods report `population_model="iao-mini"`.
- `charges[natom]` are nuclear Z minus IAO electron populations, with positive
  values meaning electron deficiency. They sum to the system charge.
- `centroids_bohr[nocc,3]` are position expectations; `n_centres[nocc]` counts
  atoms with orbital population greater than 0.1.

Periodic `representation="finite_torus_ao"`:

- Let `N=prod(mesh)`. Coefficients are `[N*nocc,N*nao]`, occupations
  `[N*nocc]`, and rotation `[N*nocc,N*nocc]`. These are localized orbitals
  across the **whole finite torus**, not per-k primitive-cell orbitals.
- `translations[N,3]` enumerates integer cell translations in lexicographic
  order, z fastest. AO columns are cell-major, then the original shell/AO
  order; atom columns likewise repeat the primitive atoms cell-major. Their
  positions are `r_atom + lattice_bohr @ translation`. Returned orbital rows
  are optimizer order; do not infer one home-cell orbital per band or automatic
  translation equivalence from their indices.
- The canonical torus coefficients before rotation are
  `C0[(t,mu),(r,n)] = (1/N) sum_k exp(+2pi i k.t) Ck[mu,n] exp(-2pi i k.r)`.
  The rotation acts on these columns, not directly on the stacked k rows.
  The torus metric uses the same Fourier transform of S(k).
- Populations `[N*nocc,N*natom]` and charges `[N*natom]` use the Lowdin finite
  torus partition, with `population_model=charge_model="lowdin_finite_torus"`.
  They must not be labelled IAO charges. Charges sum to `N*system.charge`.
- `centroids_bohr`, `spreads_bohr2` and `centroid_model` describe the existing
  **projected circular AO-centre approximation**. They are not exact periodic
  position expectations. Preserve `aliasing_fraction`, `aliasing_detected`,
  `experimental=true`, `reference_gap_hartree` and `warnings` in the overlay.

`validation` reports input/output orthonormality, the S-norm of the residual
outside the supplied subspace, and the occupied rotation's unitarity error,
with tolerance 1e-7. The periodic input value is measured after the unitary
Bloch-to-torus transform, with each original block also checked before use.
Periodic density and translation-projector errors are also reported. Tests
independently compare full density kernels `C C^H` and reconstruct the discrete
Fourier transform; they do not merely re-read these reported errors.

## Audit of existing vibe-qc implementations

The following code predates the worker and remains unmodified:

| Implementation | What it can do with archived occupied data | Limits / additional inputs |
|---|---|---|
| `iao.analyse_localization` and `localise` | Molecular IBO, Foster-Boys, Mulliken PM without SCF | Real closed-shell orbitals, exact Gaussian basis, overlap/dipoles and MINI cross-overlap rebuilt natively; IBO explicitly casts to float and Boys/PM use real algebra |
| `periodic_localise.localise_periodic_gamma` | Real Gamma PM using supplied `mo_coeffs` and periodic `overlap`; Boys uses home-cell moments | PM criterion is overlap/population based; Boys and position moments are valid only for non-wrapping states. The wrapper uses `real_if_close(...).real`, so it is not a safe complex consumer. The new worker never calls it |
| `periodic.ccm.localize.localise_ccm` | Localization within the selected finite Gamma-CCM supercell occupied space | Delegates to the legacy Gamma wrapper; does not make primitive Gamma equivalent to general k meshes |
| `periodic.chi.localization.localize_aiccm2026dev_b_occupied_blocks` | Complex finite cyclic-mesh localization without any SCF result class, `wannier` or `iao` objective | Full C(k), S(k), lattice, mesh, AO centers/atom map; IAO additionally needs target/reference and reference overlaps on the torus. Complex unitary rotations, native Fourier and projector audits. Experimental projected-position objective |
| `localize_aiccm2026dev_b_unrestricted_occupied` | Independently rotates alpha/beta occupied spaces | Existing result-object API requires converged B-stream SCF provenance, spin blocks, overlap/Fock/energies. Not exposed by protocol 1 |
| Native `_localize_periodic_gaussian_occupied` | Connected finite-image Gaussian Bloch IAO -> diabatic seed -> time-reversal-preserving IAO-PM -> Wannier chain, with supplied admitted reference, no SCF rerun | Restricted mean-field state, occupied/frozen/active/virtual masks, k data, Fock/energies/occupations, matching target/minimal Gaussian basis, cell, finite-image policy and memory/work admission. It explicitly reports `hf_basis_source_authenticated=false` and `infinite_image_tail_certified=false`. A bounded reference implementation, not a ready archive adapter |

The connected native tests exercise Gamma, odd `(3,1,1)` and even `(2,2,2)`
meshes and compare native IAO-PM objectives with independent AO oracles. That
is real implemented multi-k functionality, but it is not support for arbitrary
archived QVF states in this worker. Its IAO-PM p=4 convention is also distinct
from the molecular Mulliken p=2 method named `pipek-mezey` in the viewer.

No native refactor or new localization algorithm is introduced here. The new
periodic method selects the existing `wannier` block API. Its numerical model
and finite-torus limits stay visible. General periodic Boys needs periodic
position/link operators, usually neighboring-k overlaps M(k,b), consistent
Bloch gauge and a suitable optimizer. Molecular dipoles are not a substitute.
A general IAO/IBO archive adapter needs matched minimal-reference matrices or
a reproducible periodic Gaussian integral/image convention, not only C(k).

## QVF sufficiency and separate format proposal

This audit uses vibe-qc's current producer
`python/vibeqc/output/formats/qvf.py`, the vendored specification and archives
under `docs/_static/examples/`. It changes neither the qvf repository nor any
vendored schema.

- `wavefunction.gto` has exact ordered shells, purity, real or split-complex
  row-per-orbital coefficients, spin and occupations. A molecular restricted,
  all-electron archive with that section, authoritative charge/multiplicity,
  and geometry is sufficient. The worker rebuilds its metric and reference
  integrals without SCF. Volume-only orbitals or missing occupied rows are not
  sufficient. Fractional natural orbitals, ECP provenance gaps and unknown AO
  conventions must be disabled, not guessed.
- `x_vibeqc.bloch_wavefunction` carries restricted per-k complex coefficients,
  occupations, energies and k coordinates. The current producer can include
  `k_weight`; older archives may omit it. Its newer unrestricted restart flavor
  stores alpha/beta **density matrices**, which are not archived orbital gauges
  and must not be treated as supplied occupied coefficients.
- Structure JSON already has row-vector lattice data in **angstrom**, PBC and
  dimensionality. This worker expects a transposed column-vector lattice in
  bohr. Atomic positions likewise need conversion: divide angstrom by
  `0.529177210903`. Do not convert shell exponents, already in bohr^-2.
- Neither generic molecular `wavefunction.gto` nor the current Bloch extension
  guarantees periodic S(k), mesh/phase/image-policy provenance, or matched
  minimal-reference matrices. Rebuilding a basis **by name** is insufficient.
  Even with exact shells, finite-BvK and infinite image sums may have different
  metrics. Do not infer finite-BvK provenance from the existence of a k mesh.
- Inspected examples: `h2o-rhf/output-h2o-rhf.qvf` has a molecular GTO section;
  `mgo-route-gdf-bipole/output-mgo-rhf-sto3g-gdf-k222.qvf` has eight complex Bloch
  blocks but lacks overlap blocks and integration weights. The two checked-in
  `chi-ccm-b-qvf` examples have no generic/Bloch occupied coefficient section
  suitable for this worker. A rendered Wannier volume is not sufficient data.

**Proposal for the qvf repository, not implemented here:** standardize an
optional relocalization/restart payload (or version the existing extension)
linking the exact structure and basis with immutable identities, explicit AO
normalization/order, ECP/electron-count and spin/occupation semantics,
Hamiltonian representation (`finite_bvk`, infinite-crystal, etc.), lattice
orientation/units, complete mesh and shift, ordered full k points and weights,
Bloch phase convention, and S(k) in lossless split-complex encoding. Preserve
source band energies/gap provenance and finite-image cutoffs or boundary
policy. Use explicit absent/unknown metadata, never implicit defaults.

For future IAO localization add the reference-basis identity/atom map,
S12(k)/S22(k) or enough authenticated Gaussian/image metadata to rebuild them.
For general Wannier/Boys add neighboring-k link overlaps and their gauge/
reciprocal-shift conventions. Optional Fock blocks support an energy-invariance
check. Unrestricted orbital localization needs alpha/beta coefficient and
occupation blocks; a spin-density-only restart is a separate capability.
The format owner must choose the extension/version and producer migration.
No mandatory QVF change is needed for the molecular restoration.

## Exact remaining vibe-view changes

1. Persist an absolute backend Python path globally, independent of the
   viewer environment. Probe using the exact argv above, drain both pipes,
   check protocol/worker versions and method readiness, and surface failures.
2. Build the request from the **active archive geometry and exact shell data**,
   carrying explicit charge, multiplicity, spin, occupations and ECP status.
   Select the complete occupied block, preserve AO order and complex encoding,
   and convert structure coordinates/lattice units. Do not call a fresh RHF
   when coefficients are absent or rejected. Offer that as a separately named
   action which submits `source="fresh_rhf"` and marks its result as new SCF.
   Geometry edits invalidate the archived subspace unless compatible new
   orbitals exist; passing an overlap check alone does not establish provenance.
3. Replace `sys.executable -m vibeview.relocalize` with the configured backend
   argv. Use unique request IDs, one process per operation, and cancel obsolete
   work on file reload or geometry/basis edits. Accept only a matching terminal
   result for the same archive/geometry snapshot. Neither stale overlays nor a
   parseable but unsuccessful process exit may replace the active result.
4. Enable IBO/Boys/PM only for supported molecular real restricted cases. Keep
   periodic controls disabled unless the full experimental finite-BvK contract
   is present and the user opts in. Never relabel AICCM as periodic Boys/IBO/PM.
   The existing archives audited here cannot enable that periodic route.
5. Adapt result rows and occupations as supplied. Molecular overlays can use
   the existing renderer after renaming `centroids_bohr` to its internal field.
   Display charge/population models, fresh-SCF provenance and validation data.
   No localized orbital energy exists. Keep overlays separate from canonical
   archive data and allow clearing/reverting them.
6. A periodic overlay requires a separate finite-torus AO/atom adapter using
   `translations`, full complex coefficients and the returned shape; the
   current restricted primitive-cell adapter is insufficient. Display
   experimental status, approximate center model and aliasing flags. Do not
   squeeze the leading k/cell structure into molecular rows or `.real`.
7. Add viewer subprocess contract tests with these fixtures, including missing
   backend, partial readiness, malformed JSON, missing terminal event, timeout,
   nonzero exit, cancellation, stale IDs/geometry and population-model labels.
   Share fixtures or copy them with their worker/protocol version pinned.

See `tests/data/relocalize/README.md` for fixture provenance and numerical test
commands. The worker only returns process events; it creates no job artifacts
and does not modify the input archive.

## Implementation verification

Initially verified in an isolated macOS arm64 Python 3.14.7 environment, with a native
extension rebuilt from this checkout and no installed vibe-view:

- `tests/test_relocalization_worker.py`: **61 passed**. This includes supplied
  molecular water orbitals, the existing water QVF, positive Gamma/complex
  finite-torus fixtures, independent projector/orthonormality checks, explicit
  RHF, strict rejections, UTF-8/type enforcement and broken/partial backend
  readiness.
- Localization/AO regression command in the fixture README: **111 passed,
  1 skipped, 2 deselected**. The skip needs the separately installed viewer.
  The two excluded tests run full periodic SCFs; the initial supplemental run
  was stopped before replacing it with the documented localization-focused
  command. No full-suite or release-gate qualification is claimed.
- `tests/test_binding_sanity.py`: **2 passed**.
- `tests/test_test_gate_lanes.py`: **72 passed**, after installing its required
  `pytest-timeout` test plugin in the fresh environment.
- Ruff on the worker, launcher, worker tests and fixture generator: clean.
  Both new Markdown documents parse with MyST without warnings. Staged privacy,
  prose and whitespace checks pass.
- Installed `--probe` and stdin capability requests report all four methods
  ready. `python -I -m vibeqc_relocalize` handles both periodic fixtures through
  JSONL with native diagnostics isolated from stdout.
- A normal wheel was also built with `pip wheel . --no-deps
  --no-build-isolation`, installed with its base dependencies into a fresh venv,
  and exercised from outside the checkout. Both module and console-script
  probes, all three positive fixtures and both rejection fixtures passed.
  Python modules and the native extension loaded from that venv's
  `site-packages`; vibe-view was absent. This validates a separate installed
  environment using this host's native libraries, not wheel portability.

Integration with post-0.17.4 main repeated the worker, localization, AO,
periodic-kernel, binding and test-policy checks, adding the new IAO population
suite and released-changelog tests: **330 passed, 3 skipped, 2 deselected**.
One unrelated policy assertion failed because upstream `test_kpoints.py`
imports experimental CCM helpers without the module marker its checker expects.
The same offender was verified directly in the upstream revision; both that
file and the checker are unchanged by the worker. The three skips require
optional PySCF or vibe-view, and the two excluded full periodic SCFs are the
ones named above. The fresh probe still reports native, RHF and all four
localization methods ready. All 191 released changelog sections match their pins.

Across the three saved positive fixtures, the largest output orthonormality
error is about 1.0e-15, subspace residual 2.8e-16, and unitary error 1.2e-15
(rounded upward). The complex periodic result has an imaginary coefficient
magnitude about 0.344; preservation is not achieved by dropping imaginary data.
Expected finite-torus aliasing warnings remain visible in results.
