# `vibeqc.data_library` — parameter storage standard

This directory is the **single source of truth** for all non-code
parameter data shipped with vibe-qc: basis sets, ECPs, dispersion
damping constants, EEQ atomic-charge parameters, gCP geometric-
counterpoise fits, integration grids, and custom XC-functional
alias definitions.

The standard exists so that:

1. **License + citation discipline is mechanical, not vibes.** Every
   parameter file carries its originating publication, DOI, and the
   redistribution license inline — vibe-qc's SCF log surfaces the
   bundled citation string per-job for downstream papers (CLAUDE.md
   § 1 + § 8).
2. **Adding / extending parameters does not require a C++ rebuild**
   for cold-path data (gCP, alias definitions, less-used basis sets).
   Hot-path data (D3-BJ coefficients, EEQ alphas, Lebedev grids) is
   still compiled into the C++ core for inner-loop performance, but
   the TOML file under this directory is the source-of-truth — a
   small generator script (`scripts/generate_cpp_data.py`) emits the
   `.cpp` table from the TOML at build time. Mirrors the existing
   `setup_basis_library.sh` regeneration pattern for basis sets.
3. **External contributors can extend coverage** without touching
   compiled code: drop a new `.toml` file with the right metadata
   into the appropriate sub-directory, register it in the relevant
   Python loader (`gcp.py`, etc.), open a PR with the originating
   publication referenced. The standard makes "I added the
   parameters for element X at basis Y" a one-file diff.

## Directory layout

```
data_library/
├── README.md                  # This file — the standard, in full.
├── _schema.toml               # Cross-cutting TOML schema spec.
├── basis/                     # Basis-set definitions (.g94 files).
├── ecp/                       # Effective core potentials.
├── gcp/                       # Kruse-Grimme 2012 gCP parameters.
│   ├── _template.toml         # Schema + worked example.
│   ├── def2-svp.toml
│   ├── def2-tzvp.toml
│   └── ...
├── dispersion/                # D3-BJ + D4 damping params (FUTURE Phase 2).
├── eeq/                       # EEQ atomic charge parameters (FUTURE Phase 2).
├── functional/                # Custom XC-functional alias definitions (FUTURE Phase 2).
└── lebedev/                   # Lebedev quadrature grids (FUTURE Phase 2).
```

**v0.9.0 status note**: `basis/` and `ecp/` are still loaded from
their legacy paths `python/vibeqc/basis_library/` and
`python/vibeqc/ecp_library/` respectively. The diagram above shows
the post-Phase-2 target; migrating the basis loader to the new
`data_library/basis/` path is a separate v0.9.x MR that updates
`vibeqc._basis_g94`, `vibeqc.basis_filter`, and the C++
`LIBINT_DATADIR` define together. The placeholder `basis/` and
`ecp/` entries in the diagram above are aspirational — they will
land in Phase 2 alongside the dispersion / EEQ / functional /
Lebedev migrations.

## TOML file contract

Every parameter file under `data_library/` follows the same
top-level shape:

```toml
# Per-publication metadata — REQUIRED for every parameter file.
[metadata]
name = "def2-svp"                   # lowercase canonical id
kind = "gcp_parameters"             # type-discriminator; see kinds list below
citation = "Author et al., Journal Vol, Page (Year)"
doi = "10.xxx/xxxxx"                # empty string if no DOI (e.g. SI-only data)
license = "..."                     # full SPDX-compatible license string or
                                     #   "Open data: numerical results from a
                                     #    publication; redistribution is
                                     #    standard scientific practice"
source = "Table N of <citation>"    # which page/table/file the numbers come from
status = "complete"                  # complete | partial | pending
coverage = "H–Ar"                    # human-readable element-coverage statement
note = ""                            # free-form additional context

# Body — format defined per kind. See _schema.toml for the catalogue.
[parameters]
# kind-specific top-level constants
...

[elements.H]
# kind-specific per-element data
...
[elements.C]
...
```

### Required metadata fields

| Field      | Type   | Notes                                             |
|---         |---     |---                                                |
| `name`     | string | Lowercase canonical identifier; the key the     |
|            |        | Python loader resolves on. Must match the file  |
|            |        | stem (`def2-svp.toml` → `name = "def2-svp"`).   |
| `kind`     | string | Schema discriminator. See list below.            |
| `citation` | string | Single-line bibliographic citation.              |
| `doi`      | string | DOI of the originating publication; "" if N/A. |
| `license`  | string | Redistribution license. SPDX-friendly or a       |
|            |        | clear free-text statement.                        |
| `status`   | string | `complete` / `partial` / `pending`.             |

### Status values

* `complete` — all elements declared in `coverage` are populated.
  Safe for production use. The most common state for shipped files.
* `partial` — some elements are missing; the file declares which
  are present in `coverage`. Loader populates the present entries
  and raises `<Kind>DataMissing` for missing elements (same shape
  as `vibeqc.gcp.GCPDataMissing`).
* `pending` — metadata only, no parameter rows yet. File acts as a
  registration stub so the dispatcher knows about the basis but the
  numbers are not yet bundled. Loader raises a clear "<source>
  parameters not yet bundled; see <citation>" error.

## Catalogue of `kind` values

| `kind`              | Schema      | Loader module                  |
|---                  |---          |---                             |
| `gcp_parameters`    | `_schema.toml § gcp` | `vibeqc.gcp`           |
| *(future)* `d3bj_damping` | `_schema.toml § d3bj`    | `vibeqc.dispersion` |
| *(future)* `d4_damping`   | `_schema.toml § d4`      | `vibeqc.dispersion_d4` |
| *(future)* `eeq_alpha`    | `_schema.toml § eeq`     | `vibeqc.eeq` (Python wrapper around the C++ EEQ) |
| *(future)* `xc_alias`     | `_schema.toml § xc_alias`| `vibeqc.functional_aliases` (NEW) |
| *(future)* `lebedev_grid` | `_schema.toml § lebedev` | `vibeqc.grid` (C++-side via codegen) |

`basis/` and `ecp/` keep the existing `.g94` and `.ecp` formats —
those are libint-native and don't benefit from a TOML wrapper.

## License + citation surface

When `vibeqc.run_job(...)` or any of the lower-level entry points
consume a parameter file, the file's `[metadata]` block is recorded
into the `.system` job manifest and the `.out` file's "Parameter
provenance" section. Users writing a paper that uses, e.g., the
PBEh-3c composite at def2-mSVP will see:

```
Parameter provenance
----------------------------------------------------
gCP (def2-mSVP):
  Kruse & Grimme, J. Chem. Phys. 136, 154101 (2012)
  + Grimme et al., J. Chem. Phys. 143, 054107 (2015)
  (DOI: 10.1063/1.3700154, 10.1063/1.4927476)
Basis (def2-mSVP):
  Weigend & Ahlrichs, Phys. Chem. Chem. Phys. 7, 3297 (2005)
  + Grimme et al., J. Chem. Phys. 143, 054107 (2015)
  (DOI: 10.1039/B508541A, 10.1063/1.4927476)
Dispersion (D3-BJ for PBEh-3c):
  Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011)
  (DOI: 10.1002/jcc.21759)
```

The citation hardness comes from the `metadata.citation` + `metadata.doi`
fields being structurally required; no parameter file lands on `main`
without them.

## Extending the data library

To add a new parameter set:

1. Determine which sub-directory it belongs in (`gcp/`, `dispersion/`,
   ...) — open an issue if none of the existing kinds fit.
2. Copy `_template.toml` from that sub-directory into a new
   `<name>.toml`.
3. Fill in the `[metadata]` block first; check the cited paper has
   a redistributable-numerical-results posture (most published
   parameter tables do — they're numerical scientific results, not
   creative works).
4. Populate `[parameters]` and `[elements.*]` from the cited source.
   Be explicit about where each number came from (Table number,
   equation number, or SI page).
5. Add a regression test under `tests/test_<kind>.py` that loads
   the new file and exercises a simple correctness check (e.g. an
   atom-pair gCP energy known from the original paper).
6. Update the relevant Python loader to register the new file.
7. Open a PR; CI will verify the schema + ensure `status` is
   `complete` if you claim full element coverage.

For hot-path data that needs to live in compiled C++, also:

8. Run `python scripts/generate_cpp_data.py <kind> <name>` to emit
   the matching `cpp/src/<kind>_data.cpp` table.
9. Commit both the TOML and the generated `.cpp` (the build does
   not regenerate at build time — committing the `.cpp` makes the
   provenance auditable via `git blame`).

## Phase 1 status (v0.9.0)

**Landed**:

* `gcp/` — Kruse-Grimme 2012 geometric counterpoise parameters.
  def2-svp + def2-tzvp populated (H/C/N/O/F); skeleton files for
  minix / def2-mSVP / def2-mTZVP / def2-mTZVPP / vDZP marked
  `status = "pending"` until each per-element table is verified
  against the originating publication or `mctc-gcp` source.

**Pending Phase 2** (multi-session — see `docs/roadmap.md` §
"data_library Phase 2"):

* Migrate `cpp/src/dispersion_params.cpp` (D3-BJ damping) to
  `data_library/dispersion/` + codegen.
* Migrate `cpp/src/eeq_charges_data.cpp` to `data_library/eeq/` +
  codegen.
* Migrate `cpp/src/xc.cpp` `resolve_alias` dict to
  `data_library/functional/` + a runtime loader.
* Migrate `cpp/src/lebedev_data.cpp` to `data_library/lebedev/` +
  codegen.

The Phase 2 migrations are deliberately staged separately from
Phase 1 because each one needs a generator script + matching CI
job; landing them piecewise keeps each MR small enough to review.
