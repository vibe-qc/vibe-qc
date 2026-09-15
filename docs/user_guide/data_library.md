# The `data_library/` parameter-storage standard

Every numerical parameter shipped with vibe-qc, basis sets, ECPs,
dispersion damping constants, EEQ atomic-charge parameters, gCP
geometric-counterpoise fits, integration grids, custom XC-functional
alias definitions, lives under
[`python/vibeqc/data_library/`](https://github.com/vibe-qc/vibe-qc/tree/main/python/vibeqc/data_library).
That single tree is the canonical source of truth: it carries the
publication citation, DOI, redistribution license, and per-element
coverage matrix for every parameter set vibe-qc consumes.

This page is the user-facing summary. The full normative standard is
in the directory's [`README.md`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/data_library/README.md);
the per-kind schema is in
[`_schema.toml`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/data_library/_schema.toml).

## Why this exists

vibe-qc made three architectural decisions about parameter data, in
order of priority:

1. **Ship everything that's published + redistributable.**
   Optional-dependency packages (`dftd3`, `dftd4`) remain as backends
   for users who already have them, but vibe-qc itself bundles the
   reference numerical data. No paper should fail to reproduce
   because the user forgot a `pip install`.
2. **License + citation discipline is mechanical, not vibes.** Every
   `.toml` file carries `metadata.citation`, `metadata.doi`, and
   `metadata.license` as structurally required fields. CI rejects
   any new parameter file missing those. The `.system` job manifest
   + the `.out` SCF log surface them at run time so the citation
   chain reaches downstream papers.
3. **Extending coverage is a one-file diff.** Adding the gCP
   parameters for a new element at an existing basis is dropping a
   `[elements.<symbol>]` table into the right `.toml`. Adding a new
   basis is dropping a whole `.toml`. No C++ recompile for cold-path
   data (gCP, alias definitions, custom hybrid mixes).

## Directory layout

```
data_library/
├── README.md         # The normative standard (read this first to extend).
├── _schema.toml      # Per-kind body schema specification.
├── basis/            # libint-native .g94 basis-set files.
├── ecp/              # Effective core potentials.
├── gcp/              # Kruse-Grimme gCP parameters.
└── (Phase 2 — future, not yet migrated)
    ├── dispersion/   # D3-BJ + D4 functional damping params.
    ├── eeq/          # EEQ atomic-charge parameters.
    ├── functional/   # Custom XC alias definitions (e.g. PW1PW, PBEh-3c).
    └── lebedev/      # Lebedev quadrature grids.
```

## TOML file shape

Every parameter file under `data_library/` follows the same top-level
structure:

```toml
[metadata]
name = "def2-svp"                     # canonical lowercase id (matches file stem)
kind = "gcp_parameters"               # schema discriminator
citation = "Kruse & Grimme, J. Chem. Phys. 136, 154101 (2012)"
doi = "10.1063/1.3700154"
license = "Open data: numerical results from a publication"
source = "Tables 2 + 3 of the cited paper"
status = "partial"                    # complete | partial | pending
coverage = "H, C, N, O, F"
note = "..."

[parameters]
sigma = 0.4048
alpha = 0.5263
beta  = 1.4009

[variants.some-method]                # optional, see below
eta   = 1.40858
alpha = 0.29083

[elements.H]
e_mis  = 0.000539
n_virt = 4.0
zeta   = 1.0000

[elements.C]
e_mis  = 0.027326
# ...
```

Element keys are atomic symbols (`H`, `C`, `Fe`, ...). The loader is
case-insensitive on the symbol. Per-kind body fields are documented
in `_schema.toml`, for gCP, they're `(sigma, eta, alpha, beta)` at the
top level plus `(e_mis, n_virt, zeta)` per element.

An optional `[variants.<method>]` block overrides any of the four fit
constants for one method while reusing the file's `[elements.*]` tables
verbatim. This exists because the constants are per `(method, basis)`
pair while the per-element tables are per basis, exactly as in
`gcp.f90`. Several composites can share one parent basis without sharing
a fit: PBEh-3c, B3LYP-3c and HSE-3c all sit on def2-mSVP, and HSE-3c's
constants are its own. Select one with
`vq.gcp_params_for(basis, variant=...)` or `vq.compute_gcp(..., variant=...)`;
`vq.available_gcp_variants(basis)` lists them. An unregistered variant name
raises rather than falling back to `[parameters]`.

## Coverage statuses

* **`complete`**, every element declared in `coverage` is populated.
  Safe for production use.
* **`partial`**, some elements present, others raise the kind's
  domain-specific data-missing exception (e.g. `vibeqc.GCPDataMissing`)
  with the missing-element list + contribute-via-PR pointer.
* **`pending`**, metadata + fit constants are registered so the
  dispatcher knows about the basis, but `[elements.*]` blocks have
  not yet been bundled. The kind's loader raises a clear
  "<source> parameters not yet bundled; see <citation>" error so
  users hit an actionable wall, not a wrong-answer trap.

gCP coverage has grown well past the v0.9.0 first cut: def2-SVP,
def2-mSVP, def2-mTZVP, def2-mTZVPP, def2-TZVP, and MINIX are now
`status = "complete"`. Only vDZP and MINIS remain pending stubs.

## Extending coverage

Three pathways, in increasing scope:

### One element at an existing basis

1. Open the relevant `data_library/gcp/<basis>.toml` (or
   `data_library/<kind>/<name>.toml` for other kinds).
2. Add a new `[elements.<symbol>]` table populated from the cited
   source.
3. If this completes the file's `coverage` statement, bump
   `status = "complete"`.
4. Add a regression test under `tests/test_<kind>.py` exercising
   the new element.

### A whole new basis (or parameter set)

1. Copy `data_library/<kind>/_template.toml` to a new
   `<name>.toml`.
2. Fill in the `[metadata]` block, include the publication
   citation + DOI, double-check the license posture against
   `docs/license.md` § 1.
3. Populate `[parameters]` and `[elements.*]`.
4. The Python loader picks the new file up at next module import.
5. Open a PR.

### One-off run with un-bundled data

If you want to evaluate gCP against your own private parameter set,
pass an explicit `params=` to `vq.compute_gcp` (bypasses the
registry):

```python
custom = vq.GCPParams(
    basis_name="my-private-basis",
    sigma=0.3845, alpha=0.7400, beta=1.4366,
    e_mis={6: 0.0273, 7: 0.0457, 8: 0.0721},
    n_virt={6: 9.0, 7: 9.0, 8: 9.0},
    zeta={6: 5.67, 7: 6.66, 8: 7.66},
    citation="my internal reference",
)
res = vq.compute_gcp(mol, params=custom)
```

## Hot-path data, Phase 2

The `data_library/` standard is **read-only at runtime** for the
files vibe-qc consumes today. Files that need to live in compiled
C++ for inner-loop performance (D3-BJ coefficient lookup, EEQ
partial charges, Lebedev grids) will migrate to the same standard
in v0.9.x patches via a small generator pipeline:

* `data_library/dispersion/<functional>.toml` holds the
  authoritative parameter set.
* `scripts/generate_cpp_data.py <kind> <name>` regenerates
  `cpp/src/<kind>_data.cpp` from the TOML at maintenance time
  (NOT at every build, the generated `.cpp` is committed so the
  provenance shows up in `git blame`).
* The C++ inner loop continues to read the compiled table, so
  there's zero runtime cost.

This mirrors the existing
[`setup_basis_library.sh`](https://github.com/vibe-qc/vibe-qc/blob/main/scripts/setup_basis_library.sh)
regeneration pattern for basis sets: TOML source-of-truth, `.cpp`
generated cache, both committed.

## Citation surface

When you publish results, `vq.run_job(...)` puts the bundled
citations into the `.out` file's "Parameter provenance" section:

```
Parameter provenance
----------------------------------------------------
gCP (def2-mSVP):
  Kruse & Grimme, J. Chem. Phys. 136, 154101 (2012)
  + Grimme et al., J. Chem. Phys. 143, 054107 (2015)
  (DOI: 10.1063/1.3700154, 10.1063/1.4927476)
```

Copy that block into your paper's Methods section verbatim, every
parameter file vibe-qc consumed for that run is named.
