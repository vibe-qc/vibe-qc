---
orphan: true
---

# Announcement, `vibeqc.output` + citation database (2026-05-18)

**For**: all v0.8.x dev chats, molecular, periodic, basissetdev,
density-fitting, ECP, dispersion, perf, vq queue, docs.

**Subject**: New `vibeqc.output` module and citation-database
discipline. Action required when your chat ships a new method /
functional / basis / library.

---

## TL;DR

* New package on `main`: `python/vibeqc/output/`, unified output,
  logger, and citation surface. Full design contract in
  [`docs/design_output_module.md`](design_output_module.md).
* New file the dev chats must update:
  `python/vibeqc/output/citations/database.toml`, single source of
  truth for every paper that vibe-qc auto-cites in `.out` /
  `.bibtex` / `.references`.
* AGENTS.md gained **rule 8, "Citation database ownership"**. Read
  it; it's the contract.
* CI gate lives in `tests/test_citations.py`, its
  `_REQUIRED_FUNCTIONALS` / `_REQUIRED_BASIS_SETS` lists will fail
  the build when your feature lands without a route.

**End-of-day update (2026-05-18 evening)**: the seven writer modules
that previously lived at scattered top-level paths
(`vibeqc.structured_log`, `vibeqc.perf`, `vibeqc.crash_dump`,
`vibeqc.scf_log`, `vibeqc.system_info`, `vibeqc.io.molden`,
`vibeqc.io.trajectory`) have been relocated to
`python/vibeqc/output/formats/`. Backward-compat re-export shims
stay at every old path, function/class identity is preserved
across both, so existing `from vibeqc.X import Y` imports keep
working unchanged. New code should prefer the canonical
`from vibeqc.output.formats.X import Y` path (or the
`vibeqc.output` top-level re-exports).

**Also as of evening 2026-05-18**: dispatcher infrastructure
(`vibeqc.output.Dispatcher` + `default_dispatcher()` with 10
pre-registered adapters + `OutputWriter.dispatch_role`), CIF
writer (`vibeqc.output.write_cif`), and two operator CLIs, 
`vibeqc-cite` (already in TL;DR above) and `vibeqc-outputs`
(reads any `.system` manifest and prints status + per-file
table; `--strict` exits 3 on missing always-on files for CI
gates). Plain console-scripts; usable from any Python shell
without a full vibe-qc install.

## What changed

`run_job` now writes a richer `{stem}.system` manifest with two new
sections, `[plan]` (declared at job start, before any compute) and
`[outputs]` (rewritten as each artefact lands, status flips to
`"complete"` / `"crashed"` at job end). vq reads this to know which
files to fetch and to detect crashed jobs.

The same call also emits `{stem}.bibtex` and `{stem}.references`
sibling files, assembled from `database.toml`, and appends a `##
References` block to `{stem}.out`. The user no longer has to
cross-reference `docs/citing.md` by hand.

## What your dev chat needs to do

When your PR adds, removes, or renames one of the items below, your
PR must also patch `database.toml` in the **same merge**:

| You touched … | Database update |
|---|---|
| A new functional (libxc ID, hybrid mix, dispersion-corrected) | Add `[entries.<paper_key>]` for each defining paper. Add a route under `[routes.functionals]` keyed by the lowercase functional name. Extend `tests/test_citations.py::_REQUIRED_FUNCTIONALS`. |
| A new basis set (bundled `.g94` or otherwise) | Add `[entries.<paper_key>]`. Add a route under `[routes.basis_sets]` keyed by the lowercase basis name. Extend `_REQUIRED_BASIS_SETS`. |
| A new ECP | Add the libecpint reference if not already present; the route under `[routes.libraries]` already exists. |
| A new dispersion model (D4 native, gCP, …) | Add `[entries.<paper_key>]`. Route under `[routes.methods]` keyed by `"d4"` / `"gcp"` / etc. |
| A new SCF accelerator (SOSCF, TRAH, ADIIS) | Same shape: `[entries.…]` + `[routes.methods]` entry. |
| A new linked native library (e.g. someone vendors BLAS for SCF) | Add `[entries.…]`. Add a route under `[routes.libraries]`. Wire the runtime conditional in the corresponding driver's `assemble_from_plan(..., uses_<lib>=True)` call. |

The database is **commented**. Look at the existing entries, copy
the shape, adjust the fields. The required fields are: `kind`,
`bibtex_key`, `authors`, `title`. Common optional fields:
`journal`, `volume`, `issue`, `pages`, `year`, `doi`, `url`, `notes`.

`bibtex_key` must be unique across the whole database, convention is
`<first_author_lastname>_<short_subject>_<year>` (e.g.
`grimme_d3bj_2011`, `weigend_ahlrichs_def2_2005`).

## Where to look

* Schema + routing semantics, [`docs/design_output_module.md` §
  Citation database](design_output_module.md).
* Bundled database, [`python/vibeqc/output/citations/database.toml`](../python/vibeqc/output/citations/database.toml).
* CI gate, [`tests/test_citations.py`](../tests/test_citations.py),
  particularly `test_required_functional_has_a_route` and
  `test_required_basis_has_a_route`.
* The AGENTS.md clause, [`AGENTS.md` § "Citation database
  ownership"](AGENTS.md) (rule 8 in "Project rules an agent
  MUST follow").

## Manifest update template (for your chat's drop-box file)

Drop this paragraph into your chat's
`.release-status/v0.8.x/<chat-id>.md` once you've absorbed the rule
and intend to honour it on the next feature you ship:

```markdown
### Citation discipline — acknowledged

I will keep `python/vibeqc/output/citations/database.toml` in sync
with every method / functional / basis / library my chat lands on
`main`. The matching `tests/test_citations.py` parametrised entry +
the route in the database will land in the same PR, not a follow-up.
I will not hand-edit `docs/citing.md` or
`docs/user_guide/functionals.md § Citations` to add new references —
the database is the source of truth and Phase O7 renders the docs
from it.
```

## What this is NOT (yet)

* Periodic drivers (`periodic_runner.py`) are not yet wired into
  `OutputWriter`. Phase O5 handles that. Until then, periodic jobs
  still use the original scattered writer call sites.
* Not all post-SCF property writers (population analysis, cube,
  fchk, band structure data) are present. Phase O6 / O3 / O5 / O7
  follow.
* The vq integration (`JobSpec.expected_outputs`, `--vibeqc-dry-run`,
  `vq fetch --outputs-only`) is a separate PR in
  the separate vibe-queue repository, Phase O4.
* The `{stem}.xyz` final-geometry writer arrives in Phase O3.

If your chat needs one of the above sooner than the listed phase,
flag it on the v0.8.0 release chat's drop-box thread; the output
chat (`claude/mystifying-mcnulty-38ab6e`) can reprioritise.

## Questions

If you're unsure whether your feature needs a database update, the
test of last resort is: would a user citing this feature in a paper
need to cite a specific publication? If yes, the database needs an
entry. If you're not sure who the right person to ask is, ping the
output-module chat directly via the release-chat drop-box.
