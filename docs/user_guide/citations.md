---
myst:
  html_meta:
    "description": "vibe-qc auto-assembles citation lists on every job, BibTeX, plain-text reference list, and an in-output ## References block. The database covers vibe-qc itself, libint, libxc, spglib, libecpint, FFTW3, ASE, and every functional / basis set / dispersion model / SCF accelerator that ships on main."
    "og:title": "Automatic citations in vibe-qc"
    "og:description": "The TOML-backed citation database emits .bibtex and .references next to every .out, no more cross-referencing docs/citing.md by hand."
---

# Automatic citations

vibe-qc emits a complete reference list on every job. Alongside the
familiar [`{stem}.out`](output_files.md) text log, every successful
[`run_job`](output_files.md) call also produces:

* **`{stem}.bibtex`**, `@article` / `@software` entries, one per cited
  work, in citation order. Drop into `\bibliography{output-h2o.bibtex}`
  and let biber resolve everything.
* **`{stem}.references`**, Chicago-style numbered list, human-readable.
  Open it in a plain-text editor when you want to glance at what to
  cite without firing up LaTeX.
* **A `## References` block at the bottom of `{stem}.out`**, the same
  list embedded in the text log so a reviewer reading the output knows
  what software stack produced the numbers.

The references are assembled from a single source of truth, the
[`database.toml`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/output/citations/database.toml)
that ships with the package. Routing rules in the same file translate
"this job used B3LYP, def2-TZVP, D3(BJ), and is periodic" into the
ordered list of papers the user must cite.

```{admonition} What this replaces
:class: tip

Older vibe-qc workflows pointed users at [`docs/citing.md`](../citing.md)
and asked them to cross-reference functionals, basis sets, dispersion
models, and linked libraries by hand. The auto-citation surface (landed
v0.8.x) does that cross-reference for you on every run. `citing.md`
remains as a backup reference for ad-hoc citations and for the canonical
software citation; the runtime database is now authoritative for
everything else.
```

## What you get on disk

After running this minimal job:

```python
# input-water-pbe.py
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("water.xyz")
run_job(
    mol,
    basis="6-31g*",
    method="rks",
    functional="PBE",
    dispersion="d3bj",
    output="output-water-pbe",
)
```

the working directory contains:

```
output-water-pbe.out          # text log — ends with ## References block
output-water-pbe.bibtex       # BibTeX entries (one per cited work)
output-water-pbe.references   # plain-text numbered list
output-water-pbe.molden       # MOs
output-water-pbe.xyz          # final geometry
output-water-pbe.system       # TOML manifest — declares the plan + status
```

The bottom of `output-water-pbe.out` reads (lines hard-wrapped at 78
columns to match the SCF-trace layout):

```text
## References

Please cite the references below when reporting results
from this run. The corresponding BibTeX entries are
written to the .bibtex sibling.

  [1] Peintinger, Michael F. (2026). vibe-qc: a quantum-chemistry code for
     molecules and solids. [Software v0.9.0, MPL-2.0]. <https://vibe-qc.com/>

  [2] Valeev, Edward F. Libint: A library for the evaluation of molecular
     integrals of many-body operators over Gaussian functions. [Software].
     <https://github.com/evaleev/libint>

  [3] Ditchfield, R., Hehre, W. J., and Pople, J. A. (1971). Self-Consistent
     Molecular-Orbital Methods. IX. An Extended Gaussian-Type Basis for
     Molecular-Orbital Studies of Organic Molecules. Journal of Chemical
     Physics, 54(2), 724--728. doi:10.1063/1.1674902

  [4] Hariharan, P. C. and Pople, J. A. (1973). The influence of polarization
     functions on molecular orbital hydrogenation energies. Theoretica Chimica
     Acta, 28(3), 213--222. doi:10.1007/BF00533485

  [5] Lehtola, Susi, Steigemann, Conrad, et al. (2018). Recent developments in
     libxc - A comprehensive library of functionals for density functional
     theory. SoftwareX, 7, 1--5. doi:10.1016/j.softx.2017.11.002

  [6] Perdew, John P., Burke, Kieron, and Ernzerhof, Matthias (1996).
     Generalized Gradient Approximation Made Simple. Physical Review Letters,
     77(18), 3865--3868. doi:10.1103/PhysRevLett.77.3865

  [7] Pulay, Péter (1980). Convergence acceleration of iterative sequences.
     The case of SCF iteration. Chemical Physics Letters, 73(2), 393--398.
     doi:10.1016/0009-2614(80)80396-4

  [8] Pulay, Péter (1982). Improved SCF convergence acceleration. Journal of
     Computational Chemistry, 3(4), 556--560. doi:10.1002/jcc.540030413

  [9] Grimme, Stefan, Antony, Jens, et al. (2010). A consistent and accurate
     ab initio parametrization of density functional dispersion correction
     (DFT-D) for the 94 elements H-Pu. Journal of Chemical Physics, 132(15),
     154104. doi:10.1063/1.3382344

  [10] Grimme, Stefan, Ehrlich, Stephan, and Goerigk, Lars (2011). Effect of
     the damping function in dispersion corrected density functional theory.
     Journal of Computational Chemistry, 32(7), 1456--1465.
     doi:10.1002/jcc.21759
```

and `output-water-pbe.bibtex` (excerpt) contains:

```bibtex
% vibe-qc auto-generated BibTeX entries — one per cited
% reference, in citation order. The corresponding software
% citation for vibe-qc itself is the first entry.

@software{peintinger_vibeqc,
  author      = {Peintinger, Michael F.},
  title       = {vibe-qc: a quantum-chemistry code for molecules and solids},
  year        = 2026,
  url         = {https://vibe-qc.com/},
  version     = {0.8.0},
  license     = {MPL-2.0},
  note        = {Always cite. A peer-reviewed publication is forthcoming; this software citation is the canonical reference until then.}
}

@article{perdew_burke_ernzerhof_1996,
  author      = {Perdew, John P. and Burke, Kieron and Ernzerhof, Matthias},
  title       = {Generalized Gradient Approximation Made Simple},
  journal     = {Physical Review Letters},
  volume      = 77,
  number      = 18,
  pages       = {3865--3868},
  year        = 1996,
  doi         = {10.1103/PhysRevLett.77.3865}
}

@article{grimme_d3bj_2011,
  author      = {Grimme, Stefan and Ehrlich, Stephan and Goerigk, Lars},
  title       = {Effect of the damping function in dispersion corrected density functional theory},
  journal     = {Journal of Computational Chemistry},
  volume      = 32,
  number      = 7,
  pages       = {1456--1465},
  year        = 2011,
  doi         = {10.1002/jcc.21759}
}
```

## How the routing works

The database is structured into two halves: an `[entries.<key>]` table
per citable reference and a `[routes.<category>]` table that maps from
"what the user requested" to "which entries fire". The runtime
assembler walks the routes table in a fixed order:

1. **Software**, `vibeqc_software` always fires first.
2. **Integrals**, `libint_valeev` always fires second.
3. **Basis set**, keyed lookup on the lowercased basis name. The
   `6-31g*` route fires both Ditchfield 1971 (the split-valence paper)
   and Hariharan-Pople 1973 (the polarisation extension); `cc-pVDZ`
   fires Dunning 1989; the pob-rev2 family fires both Peintinger 2013
   and Vilela Oliveira 2019.
4. **Functional**, if a functional is set, `_libxc_always` fires
   (Lehtola 2018) plus the per-functional entries. `b3lyp` (and the
   flavor spellings `b3lyp5` / `b3lyp/g` / `b3lypg`) fires Becke
   1993, Lee-Yang-Parr 1988, Stephens 1994, VWN 1980, and Hertwig-
   Koch 1997 (the which-B3LYP-is-which paper, cite it so your
   Methods section pins the VWN flavor); `pbe0` fires PBE 1996 plus
   Adamo-Barone 1999; `pw1pw` fires PW91 1992 plus Bredow-Gerson
   2000.
5. **SCF accelerator**, cited from what the SCF executed, not from the
   options object you passed (issue #682): every DIIS-family run fires
   Pulay 1980 + 1982; `ediis` adds Kudin-Scuseria-Cancès 2002; the
   default `EDIIS_DIIS` hybrid adds Kudin 2002 and Garza-Scuseria 2012
   only when at least one SCF cycle actually took the EDIIS branch of
   its commutator-norm switch, and cites Pulay alone when every applied
   step was DIIS; `adiis` / `adiis_diis` add Hu-Yang 2010 by the same
   rule; `kdiis` adds Kollmar 1997. The per-iteration record is
   `SCFIteration.accelerator_step` on the result's `scf_trace`
   (0 none, 1 DIIS, 2 KDIIS, 3 EDIIS, 4 ADIIS). Two runs that dispatch
   the same SCF record the same set, whether or not an options object
   was supplied.
6. **Dispersion**, `d3` fires Grimme 2010; `d3bj` adds Grimme 2011;
   `d4` fires the full three-paper set the upstream `dftd4` authors
   ask users to cite, Caldeweyher 2017 (precursor), 2019 (method),
   and 2020 (periodic extension), **plus the damping-parameter fit
   paper** when the functional's D4 parameters were published
   separately from the method paper: r²SCAN-D4 adds Ehlert 2021,
   the r²SCAN hybrids add Bursch 2022, ωB97X-D4 adds Najibi-Goerigk
   2020, LC-ωPBE adds Friede 2023, and the revDSD / revDOD double
   hybrids add Santra-Sylvetsky-Martin 2019, so your Methods
   section cites the paper the damping parameters actually come
   from. The per-functional mapping lives in
   `vibeqc/dispersion_d4_parameters.py` (the inline `doi=` fields)
   and is mirrored by the `[routes.dispersion_params]` table in the
   citation database.
7. **Conditional libraries**, `spglib` (Togo-Tanaka 2018) fires for
   periodic jobs; `libecpint` (Shaw-Hill 2017) when an ECP is in use;
   `fftw3` (Frigo-Johnson 2005) when the FFT-Poisson backend ran;
   `ase` (Larsen 2017) when the ASE Calculator or BFGS path was taken.

Each entry appears only once, in first-fire order, even when multiple
routes pull it in (e.g. Lee-Yang-Parr fires for both B3LYP and B2PLYP
but appears once if both somehow ran in the same job).

## OpenTrustRegion: cite the optimizer that ran

An executed OpenTrustRegion molecular SCF selects
`greiner_opentrustregion_2026`, including issue 2, in the `.out` References
block, `.references` and `.bibtex` files:

```{vibeqc-cite-entry} greiner_opentrustregion_2026
```

The plain-text style may shorten the author list with "et al."; BibTeX
retains all four authors, including the Unicode spelling of Høyvik. The
reference appears once per bibliography. Native SCF does not cite it;
OpenTrustRegion does not implicitly receive DIIS references. The selection
uses the executed backend report, rather than merely testing whether the
optional library was compiled in. For the mathematical explanation, see
[OpenTrustRegion theory](opentrustregion_theory.md).

## `vibeqc-cite`: reprint citations from an already-run job

`pip install -e .` registers a `vibeqc-cite` console script that
reads `{stem}.system`, walks the citation database, and either prints
the references to stdout or rewrites the `.bibtex` / `.references`
siblings. When `[[citations.entries]]` are recorded, their keys, order and
visibility select the bibliography; the current database supplies the full
bibliographic metadata. This preserves runtime choices such as OpenTrustRegion
and avoids inventing a default DIIS citation during regeneration. A malformed
entry or unknown recorded key produces an error before files are rewritten.
Three workflows it covers:

* **Older manifests with a `[plan]` but no recorded citation entries**,
  assemble from the available method / basis / functional fields. Those
  fields alone cannot recover runtime-only choices. Manifests without a
  `[plan]` are rejected with a diagnostic rather than guessed.
* **Generated outputs** being copied between machines without the
  `.bibtex` / `.references` siblings, regenerate them locally without
  re-running the SCF.
* **Tutorials and docs** that want to show "here are the references
  this run cited" without embedding the output verbatim.

CLI surface:

```sh
# Print the plain-text reference list to stdout (default):
vibeqc-cite output-h2o

# Print only the BibTeX entries to stdout:
vibeqc-cite output-h2o --bibtex-only

# Write {stem}.bibtex + {stem}.references next to the manifest:
vibeqc-cite output-h2o --write

# Write only the .bibtex sibling:
vibeqc-cite output-h2o --write --bibtex-only
```

The stem can carry any suffix (`output-h2o`, `output-h2o.out`,
`output-h2o.system` all work); the CLI normalises via
`Path.with_suffix(".system")` internally. Exit codes: `0` on
success, `1` on missing / malformed manifest, `2` on database load
error.

Equivalent invocation without the console-script shim:

```sh
python -m vibeqc.output.citations.cli output-h2o --write
```

## Inspecting and assembling citations manually

The same machinery is available as a public Python API. Use it when
you want to print the bibliography ahead of a run, in a tutorial, or
when stitching citations into a manuscript via Python:

```python
from vibeqc.output import OutputPlan
from vibeqc.output.citations import (
    load_default_database,
    write_bibtex,
    write_references,
    format_references_block,
)

# Build the plan the way run_job would.
plan = OutputPlan.from_run_job_kwargs(
    output="output-h2o-pbe",
    method="RKS",
    basis="6-31g*",
    functional="PBE",
)

# Load the bundled database.
db = load_default_database()

# Assemble citations for a periodic PBE/pob-TZVP/D3BJ job with ASE
# optimisation. Boolean flags below mirror what the runner detects
# from job state.
citations = db.assemble_from_plan(
    plan,
    dispersion="d3bj",
    periodic=True,
    uses_ase=True,
)

for c in citations:
    print(f"[{c.bibtex_key}]  {' and '.join(c.authors)} — {c.title}")

# Write the same files run_job would have written.
write_bibtex("preview", citations)        # → preview.bibtex
write_references("preview", citations)    # → preview.references

# Or get the .out block as a string:
print(format_references_block(citations))
```

`citations.warnings` lists routing gaps (e.g. an unrouted basis name)
without raising, the same gaps appear at the bottom of the
`.references` file as `# --- citation routing warnings ---` lines so
they are visible to the user but never crash a job.

## Extending the database

When you add a new functional, basis set, ECP, dispersion model, or
linked library to vibe-qc you must extend the database **in the same
merge**. The contract is codified in
[Archived `AGENTS.md` § "Citation database ownership"](https://vibe-qc.com/docs/)
and enforced by
[`tests/test_citations.py`](https://github.com/vibe-qc/vibe-qc/blob/main/tests/test_citations.py)
- `_REQUIRED_FUNCTIONALS` and `_REQUIRED_BASIS_SETS` fail the build
when a registered feature has no route.

A new entry looks like:

```toml
# python/vibeqc/output/citations/database.toml

[entries.heyd_scuseria_ernzerhof_hse_2003]
kind        = "article"
bibtex_key  = "heyd_scuseria_ernzerhof_2003"
authors     = ["Heyd, Jochen", "Scuseria, Gustavo E.", "Ernzerhof, Matthias"]
title       = "Hybrid functionals based on a screened Coulomb potential"
journal     = "Journal of Chemical Physics"
volume      = 118
issue       = 18
pages       = "8207--8215"
year        = 2003
doi         = "10.1063/1.1564060"
```

and the matching route (under the right category) wires it up:

```toml
[routes.functionals]
"hse06" = ["pbe_1996", "heyd_scuseria_ernzerhof_hse_2003"]
```

Required fields are `kind`, `bibtex_key`, `authors`, and `title`. Use
the `kind` vocabulary `article` / `book` / `software` / `phdthesis` /
`misc`. The `bibtex_key` must be unique across the whole database;
convention is `<first_author_lastname>_<short_subject>_<year>` (e.g.
`grimme_d3bj_2011`, `weigend_ahlrichs_def2_2005`).

The
`vibeqc-cite-block` Sphinx directive (Phase O7, queued) will
render `docs/citing.md` and `docs/user_guide/functionals.md`'s
citations sections directly from the database so the published docs
never drift from what the runtime emits.

### Templated fields

Two template tokens are substituted at load time:

* `{{VIBEQC_VERSION}}`, the running package version. Used only by
  the `vibeqc_software` entry's `version` field so each released
  archive's citation reports its own version.
* `{{VIBEQC_YEAR}}`, the calendar year. Used by the same entry's
  `year` field.

If you need either, set `version_template` or `year_template` instead
of `version` / `year`. Other fields are taken literally.

### basissetdev sibling database

The historical set of 87 BSE-fetched bases is documented in the
[Archived `basissetdev` branch](https://vibe-qc.com/docs/)
notes. That branch was not transferred to the new core repository. Its
optional citation file, `python/vibeqc/output/citations/database_basissetdev.toml`,
is not shipped here; the registry loads it automatically when present. The schema is
identical to `database.toml`; entry-key collisions across the two
files are a load-time error.

## Periodic jobs (Phase O5)

`run_periodic_job` writes the same family of citation siblings as
`run_job` as of v0.8.x Phase O5, `.bibtex`, `.references`, and the
`## References` block in `.out`. The plan additionally declares the
periodic-specific geometry artefacts: **extended-XYZ** (ASE-style
with the lattice in the comment line), **VASP POSCAR**, and **XSF**
structure block. The `spglib` route fires automatically for any
periodic job; the `fftw3` route fires when the FFT-Poisson backend
ran; the `ase` route fires when the periodic ASE Calculator was
used.

## Dry-run pre-flight (`vq submit`'s hook)

Passing `dry_run=True` to `run_job` (or exporting `VIBEQC_DRY_RUN=1`)
short-circuits the call after the method resolves but before any
compute. The runner writes a one-shot `{stem}.system` manifest with
`[outputs].status = "dry_run"`, prints the declared-artefacts
summary to stdout, and returns `None`. No basis-set construction, no
memory estimate, no SCF.

This is the entry point [`vq submit`](queue.md) uses to learn which
files a job will produce before scheduling, when the daemon
receives a Python script that imports `run_job`, it runs the script
once with `VIBEQC_DRY_RUN=1`, parses the resulting `[plan]` section
out of the manifest, and uses it to populate `JobSpec.expected_outputs`.
For users it's also a fast way to confirm "what will this run
write?" without paying the SCF cost:

```sh
VIBEQC_DRY_RUN=1 python input-water.py
# → prints the plan, exits 0, leaves output-water.system on disk.
```

Reading the manifest back:

```python
import tomllib
with open("output-water.system", "rb") as f:
    sys = tomllib.load(f)
print(sys["outputs"]["status"])    # "dry_run"
for f in sys["plan"]["files"]:
    print(f"{f['role']:<10} {f['path']}  ({'always' if f['always'] else 'cond'})")
```

## The `.system` manifest

Each job's manifest carries a `[plan]` section that declares every
artefact the job will write *before* compute starts, and an
`[outputs]` section that fills in as files land. The
[`vq` queue](queue.md) reads this to know which files to fetch back
and to detect crashed jobs. A successful PBE/D3BJ water job ends with:

```toml
[outputs]
finished_at_iso = "2026-05-18T10:42:03Z"
status          = "complete"

[[outputs.files]]
path         = "output-water-pbe.out"
written      = true
bytes        = 4231
sha256       = "ab12cd34..."
wall_time_s  = 0.082

[[outputs.files]]
path         = "output-water-pbe.bibtex"
written      = true
bytes        = 1872
sha256       = "..."
wall_time_s  = 0.003

# ... etc
```

When the SCF crashes the writer flips `status = "crashed"`,
timestamps the crash, and rewrites the `.system` atomically so vq's
liveness detection sees the failed state. The plan section is never
mutated after job start, its purpose is to be the contract the
runtime is held to.

## API reference

```python
from vibeqc.output import OutputPlan, PlannedFile, OutputWriter
from vibeqc.output.citations import (
    Citation,
    CitationDatabase,
    AssembledCitations,
    DatabaseError,
    load_database,           # load explicit paths
    load_default_database,   # bundled DB(s)
    assemble,                # convenience: load_default + assemble
    write_bibtex,            # → {stem}.bibtex
    write_references,        # → {stem}.references
    format_references_block, # → embedded ## References text
)
```

The runtime side is small on purpose: a frozen `OutputPlan`
describes the artefact set, the bundled database holds the entries
and routes, and `assemble_from_plan(plan, **flags)` returns an
ordered `AssembledCitations` you hand to the two writers. Everything
beyond that is just rendering.

## Validation

[`tests/test_citations.py`](https://github.com/vibe-qc/vibe-qc/blob/main/tests/test_citations.py)
pins the contract:

- the bundled DB loads without errors;
- every route references a real entry (load-time validation);
- the first cited entry is always vibe-qc itself, libint always fires
  second;
- the regression suite's parametrised `_REQUIRED_FUNCTIONALS` (LDA,
  PBE, PBE0, B3LYP, PW91, B2PLYP) and `_REQUIRED_BASIS_SETS` (the
  v0.8.0-on-main set) fail loud when the DB drifts from what the
  test suite actually exercises;
- assembled lists have no duplicate keys.

When CI fails on `test_required_functional_has_a_route` it is
telling you: you added a functional but did not add a route. The fix
is to follow the [extending the database](#extending-the-database)
recipe above in the same PR, not to weaken the test.

## See also

- [Input scripts and output files](output_files.md), the full
  per-format reference for everything `run_job` writes.
- [Citing vibe-qc](../citing.md), the canonical software citation
  plus backup table for hand-citing on older versions.
- [`docs/design_output_module.md`](https://github.com/vibe-qc/vibe-qc/blob/main/docs/design_output_module.md)
  - full design rationale and the v0.8.x → v1.0 roadmap for the
  output module.
- [Archived `AGENTS.md` § "Citation database ownership"](https://vibe-qc.com/docs/)
  - the dev-chat-side contract for keeping the database in sync.
