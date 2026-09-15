# Generating and reviewing the bundled basis library

The directory `python/vibeqc/basis_library/basis/` is **mechanically
generated**. End users get every shipped basis set on `pip install`
because the directory is committed to the repo and packaged into the
wheel, but the canonical sources live elsewhere and the directory is
re-assembled by a build-time script. Treating it as hand-edited code
hides meaningful changes inside hundreds of `.g94` diffs.

This page documents the generation pipeline and the review checklist
that every PR touching `basis_library/basis/` should pass.

## Pipeline at a glance

```
third_party/libint/install/share/libint/<ver>/basis/*.g94   (standard)
python/vibeqc/basis_library/custom/*.g94 + *.ecp            (vibe-qc-specific)
                       │
                       ▼
            scripts/setup_basis_library.sh
                       │  copy standard first, custom last (custom wins on name clash)
                       ▼
            scripts/basisset_dev/split_ecp_g94.py
                       │  split BSE-format ECP-bearing files into
                       │  <name>.g94 (orbital-only, libint-loadable)
                       │  <name>.ecp (verbatim ECP blocks for libecpint)
                       ▼
python/vibeqc/basis_library/basis/*.g94 + *.ecp             (output, committed)
```

### Source directories

* **`third_party/libint/install/share/libint/<libint-version>/basis/`**, 
  the standard set libint installs with itself (Pople, Dunning, def2,
  Karlsruhe, ANO, …). Built by `scripts/build_libint.sh` as part of
  `scripts/setup_native_deps.sh`. We do not edit these files; bumping
  libint refreshes them.
* **`python/vibeqc/basis_library/custom/`**, vibe-qc-specific bases:
  every BSE-fetched set (Phase 14), every vibe-qc-engineered set
  (Phase 16 mpei-tzvp), every basis we ship that libint does not
  bundle. **`.g94` files in this directory, plus any pre-split
  `.ecp` sidecars, are the source of truth for the custom side.**
  The `.g94` files must carry an
  `! Originating publication: …` header so per-record attribution
  survives the pipeline (see Citation policy below).

### Generator scripts

* **`scripts/setup_basis_library.sh`**, the entry point. Idempotent;
  safe to re-run. Wipes `basis/`, copies libint's set in, copies
  custom `.g94` files and pre-split `.ecp` sidecars on top (so a
  custom file with the same stem wins), then invokes the splitter.
* **`scripts/basisset_dev/split_ecp_g94.py`**, splits BSE-format
  `.g94` files that bundle orbital blocks together with `<Sym>-ECP`
  blocks. libint2 cannot parse the ECP blocks
  (`"invalid angular momentum label"`), so each affected file becomes
  an orbital-only `<name>.g94` plus a `<name>.ecp` sidecar that
  vibe-qc reads via libecpint at runtime. The splitter is idempotent
  on already-split files.
* **`scripts/basisset_dev/fetch_from_bse.py`**, Phase-14 fetcher
  that pulls BSE entries into `custom/` with a provenance header.
  Build-time only; not on the runtime path.

### Expected output

* Every `python/vibeqc/basis_library/basis/*.g94` is libint-loadable
  (no `<Sym>-ECP` blocks left in the file).
* Every ECP-bearing basis has a sister
  `python/vibeqc/basis_library/basis/<name>.ecp` sidecar carrying
  the verbatim ECP blocks. The canonical inventory pinned by tests
  is documented in [`tests/test_basis_ecp_sidecars.py`](../../tests/test_basis_ecp_sidecars.py)
  (EXPECTED_ECP_BASES).

The `.ecp` sidecar is read by libecpint via `vq.parse_sidecar_path`
/ `vq.auto_ecp_centers` for Phase-14e auto-population. An accidental
deletion of a sidecar silently breaks ECP SCF for that basis, the
regression checks below guard against it.

## When to regenerate

* After **upgrading libint** (the standard set may have shifted).
* After **adding or modifying a file under `custom/`**.
* After **changing the splitter** in `scripts/basisset_dev/split_ecp_g94.py`.

```sh
./scripts/setup_basis_library.sh
```

The output directory (`basis/`) is committed. Do not edit files in
`basis/` by hand, your changes will be wiped on the next run of the
setup script. Edit `custom/` instead.

## Citation policy (rule 8)

Every bundled basis must reach the user as a citable reference. Two
mechanisms, either one suffices for a given basis but at least one
must be in place:

1. A route entry under `[routes.basis_sets]` in
   `python/vibeqc/output/citations/database.toml`, mapping the
   basis name to one or more `[entries.<key>]` records. This drives
   the SCF log and the auto-rendered `docs/citing.md`.
2. An inline `! Originating publication: …` (or `! Citation: …` /
   `! Cite: …`) header line in the `.g94` source. The setup pipeline
   preserves the file header verbatim, so per-record provenance
   survives the `custom/` → `basis/` promotion. Required for any
   custom basis the route table does not cover.

### Header-key convention

Two header keys appear in the bundle today:

* **`! Originating publication: …`**, the BSE fetcher
  (`scripts/basisset_dev/fetch_from_bse.py`) emits this on every
  fetched file. Preferred for any newly-added basis.
* **`! Cite: …`**, historical key, retained on the pob-* set
  (`pob-tzvp.g94`, `pob-tzvp-rev2.g94`, `pob-dzvp-rev2.g94`) and on
  a handful of hand-curated custom sources.

Both forms are accepted by the citation-coverage regex in
[`tests/test_basis_citation_coverage.py`](../../tests/test_basis_citation_coverage.py).
Don't churn existing files just to swap keys, preserving the
header verbatim is the contract. **Whichever key is present, the
line must carry a concrete reference** (authors + journal + DOI),
not a pointer like `! Cite: see README.md`. The README is not
parsed; a pointer leaves the header unable to attribute the basis
on its own, which is the failure mode that motivated the
post-22aface3 audit (May 2026).

When you add or rename a basis, update **the same merge** per
[AGENTS.md § 8](../AGENTS.md): add the `[entries.*]` block, the
matching `[routes.basis_sets]` line, and extend the `_REQUIRED_*`
list in `tests/test_citations.py` if the basis is a primary user
target.

## Licensing policy (rule 1)

Before pulling a new basis into `custom/`, **verify redistribution
terms**. Check:

* the originating publication's license / copyright statement,
* the BSE entry's "Role" / "Notes" / per-record license string,
* any third-party `LICENSE` or `COPYING` files shipped alongside.

If the terms are unclear or restrictive: **do not bundle.** Use the
on-demand fetcher pattern (modeled on `vqfetch`), pull from source
on first use, cache locally, surface the per-record provenance +
license string in the SCF log + `.system` manifest. The full
per-component inventory is at [`docs/license.md`](../license.md).

## Reviewing a PR that touches `basis_library/basis/`

Generated-data changes are easy to over-trust because the diff is
huge. Use this checklist:

1. **Did the PR change `custom/` or `setup_basis_library.sh` /
   `split_ecp_g94.py`?** If not, the `basis/` diff is suspect, 
   someone hand-edited a generated file. Push back.
2. **Run the diff helper** (Phase 14, this milestone):

   ```sh
   .venv/bin/python scripts/basisset_dev/diff_basis_library.py origin/main HEAD
   ```

   It enumerates added / removed / modified `.g94` and `.ecp` files,
   flags sidecar-pair changes, and prints provenance-header diffs.
   The output is a few dozen lines even for a full regeneration.
3. **For every added basis**: confirm its source (`custom/` file or
   libint version), its provenance header, its license, and a
   citation entry per § Citation policy above.
4. **For every removed basis**: confirm the removal is intentional
   and documented in the CHANGELOG `[Unreleased]`. ECP sidecar
   removals are especially easy to miss.
5. **For every modified `.g94`**: read the diff *for the file*, not
   just the summary. A coefficient change is a different review than
   an added element block.
6. **Run the integrity tests** (Phase 14, this milestone):

   ```sh
   .venv/bin/python -m pytest tests/test_basis_ecp_sidecars.py \
                              tests/test_basis_no_maintainer_paths.py \
                              tests/test_basis_citation_coverage.py
   ```

   These run without rebuilding native dependencies and finish in
   under a second on the current bundle.
7. **Check the per-file headers** don't leak maintainer home paths
   (`/Users/<name>/…`, `/home/<name>/…`), `test_basis_no_maintainer_paths.py`
   gates this in CI but the pre-commit hook
   ([`.githooks/pre-commit`](../../.githooks/pre-commit)) is the
   first line of defence per clone.

## See also

* [`docs/user_guide/basis_sets.md`](../user_guide/basis_sets.md), 
  user-facing survey of what we ship and when to pick what.
* [`docs/license.md`](../license.md), full per-component licensing
  inventory.
* [`tests/test_basis_ecp_sidecars.py`](../../tests/test_basis_ecp_sidecars.py)
  - pinned ECP sidecar contract.
* [`scripts/setup_basis_library.sh`](../../scripts/setup_basis_library.sh)
  - the regeneration entry point.
* [`scripts/basisset_dev/split_ecp_g94.py`](../../scripts/basisset_dev/split_ecp_g94.py)
  - ECP block splitter.
* [`AGENTS.md`](../AGENTS.md) § 8, citation database ownership.
* [`CLAUDE.md`](../CLAUDE.md) § 1, § 8, § 12, licensing, citation,
  privacy.
