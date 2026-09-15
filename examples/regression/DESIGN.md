# Handover: vibe-qc automated regression / parity test suite

You are picking up the build-out of a comprehensive automated test
suite for `vibe-qc` at `~/vibeqc`. This isn't
unit tests — vibe-qc already has those (~700 in `tests/`). This is a
separate, parallel infrastructure that exists to **continuously verify
the code's scientific correctness** by running a curated set of
physics calculations and comparing against reference results.

## What you're building (in waves)

### Wave 1 — Linear-dep + screening test suite (the immediate ask)

Inputs + runner + comparator for these systems, two basis sets each:

* NaCl rocksalt
* MgO rocksalt
* Al₂O₃ corundum
* LiH rocksalt

Bases: `sto-3g` (small, available everywhere) and `pob-dzvp-rev2`
(Vilela Oliveira 2019, ships with vibe-qc — the right basis for solids).

For each of the 8 (system × basis) combinations:

1. Build the periodic system + basis using the new `vq.make_basis(...)`
   API.
2. Run RKS-LDA SCF with `auto_optimize_truncation=True` (default in the
   v0.7 branch — the screening optimiser tunes cutoffs jointly).
3. If SCF doesn't converge, fall back to retry strategies in this order:
   * tighten DIIS, switch to OT-style direct minimisation,
   * apply `make_basis(..., exp_to_discard=0.1)` if PSD diagnostic
     flags critical,
   * increase `cutoff_bohr` / `nuclear_cutoff_bohr` manually past
     `optimize_truncation`'s ceiling.
4. Run the same calculation through PySCF.pbc (already in `.venv`).
5. Compute `ΔE = E_vibeqc - E_pyscf`, classify pass / fail by published
   basis-set-error tolerance per system.

### Wave 2 — All methods, all flavours

Same harness, expanded matrix:

* RHF / UHF / RKS / UKS / (later ROHF)
* MP2 (where applicable)
* Periodic + molecular comparisons
* Multi-k-point sampling, not just Γ
* Functionals: LDA, PBE, B3LYP, ωB97X (where vibe-qc has the XC kernels)

### Wave 3 — Dual-target runner

The same suite must execute against:

* The user's installed `vibe-qc` package (`pip show vibe-qc`)
* The current dev tree (`pip install -e .` from a checked-out branch)

Output: a side-by-side report showing both versions' results vs PySCF,
so a regression in dev can be caught before release.

## Where to put the code

* **Main artefact:** new directory `examples/regression/` (this dir).
  Multi-file infrastructure, not a one-shot script.
* **Per-system inputs:**
  `examples/regression/systems/<NaCl|MgO|Al2O3|LiH>/<basis>/{input.py,
  reference_pyscf.py, expected.json}`.
* **Runner:** `examples/regression/run_suite.py` — invoked as
  `python -m examples.regression.run_suite [--target dev|release|both]
  [--systems all|<list>] [--bases all|<list>]`.
* **Branch:** suggest `feature/automated-test-suite` off `main` —
  you'll merge it when wave 1 is solid.

## What's already in tree to lean on

* `python/vibeqc/eigs_preflight.py` — `vq.eigs_preflight`,
  `vq.optimize_truncation`, `vq.disambiguate_critical_overlap`. The
  screening / optimiser stack just landed in v0.7 work and is the
  foundation of wave 1.
* `python/vibeqc/basis_filter.py` —
  `vq.make_basis(mol, name, exp_to_discard=0.1)`,
  `vq.format_basis_filter_report(rep)`. Use the formatter to log every
  dropped primitive into the verbose output the user asked for.
* `python/vibeqc/orthogonalisation.py` — opt-in canonical / Lehtola
  pivoted-Cholesky / symmetric methods if SCF is fighting linear
  dependence.
* `python/vibeqc/options_dump.py` — `format_options(opts, title=...)`
  and `dump_active_settings(plog, ...)` for printing every active SCF
  setting at startup. Use this in the verbose log per system.
* `python/vibeqc/basis_library/basis/pob-tzvp.g94`, `pob-tzvp-rev2.g94`,
  `pob-dzvp-rev2.g94` — the pob bases ship in-tree. Just
  `vq.BasisSet(mol, "pob-dzvp-rev2")`.
* `examples/ase_compare/cross_code_regression/` — five comparison scripts
  against PySCF and ORCA from earlier v0.7 work. The CSV format
  defined there
  (`(system, basis, code, energy, n_iter, wall_s, note)`) is the right
  schema to reuse.
* `tests/conftest.py` — the `ORCA_DEFAULT_CONV_TOL_*` constants from
  the molecular ORCA-parity work. Match those defaults so the same SCF
  settings drive vibe-qc, PySCF, and ORCA where applicable.

## What's NOT in tree but you'll need

* **PySCF.pbc input templates** for each test system. Standard
  pyscf.pbc.dft.RKS construction; rocksalt geometries readily available
  from materials project / textbook references. Mind the basis name
  mapping (PySCF uses `gth-szv`, `gth-dzvp` for the Gaussian-pseudo-
  potential lineage; you may need to match basis at the ECP level, not
  just name).
* **Published reference values** for each system — Hartree-Fock and
  DFT total energies per cell. Sources: original pob paper (Peintinger
  et al. 2013, JCC 34, 451), pob-rev2 (Vilela Oliveira 2019, JCC 40,
  2364). LDA/PBE references from CRYSTAL benchmark sets, MOLOPT
  validation papers, or recompute via PySCF.pbc with a tight basis as
  the canonical reference.
* **Geometry data**: lattice constants and atomic positions in
  fractional coordinates per system. NaCl 5.640 Å, MgO 4.211 Å, LiH
  4.084 Å (already in this codebase's xfail test), Al₂O₃ R-3c trigonal
  a=4.760 c=12.991 (more complex; defer to wave 1.5 if it slows you
  down).

## Required behaviours of the runner

* **Verbose logging** — for every SCF, dump `format_options(opts)`,
  `format_basis_filter_report(filter_rep)` if filtering happened, the
  EIGS preflight per k-point, the truncation-optimiser report, the
  SCF iteration trace, and the final energy decomposition. **No
  silent settings changes.** This is non-negotiable per the v0.7
  transparency directive.
* **Convergence-retry ladder** — capture exceptions / non-convergence,
  log the diagnosis (`vq.disambiguate_critical_overlap` if S is
  non-PSD; `divergence_check` if energies blew up), pick the next
  strategy, retry. Each retry is logged separately so the
  failure → recovery → success path is auditable.
* **Comparator** — final report per system: `{vibeqc_E_dev,
  vibeqc_E_release, pyscf_E, delta_dev, delta_release,
  basis_set_error_estimate, verdict}`. Write CSV (one row per
  (system, basis, code, version)) for downstream regression-tracking
  dashboards.
* **Dual-target invocation** — `--target dev` and `--target release`
  should both work without code changes. Use `subprocess` to invoke a
  separate `python -c '...'` against each environment, OR drive both
  via `importlib.metadata` and parametrise the import path. The dev
  install lives at `~/vibeqc` (editable); the
  release install lives wherever `pip show vibe-qc` reports.

## Bigger picture: this becomes the dashboard

In wave 2/3 the same harness will track:

* All 8 SCF method types (RHF/UHF/RKS/UKS/...)
* Restricted vs unrestricted vs (later) restricted-open-shell
* Multi-k vs Γ-only
* Hybrid functionals, MP2, gradients, Hessians
* Both molecular and periodic
* Both small (CI-friendly, < 30s) and large (overnight, < 8h) test sets

So design the runner with the matrix in mind from the start. Per-system
files with a `system.py` / `expected.json` schema scale to thousands of
entries; a giant single-file dispatch doesn't.

## Out of scope for this chat

* Performance benchmarking (wall-clock, memory, FLOPs).
* GPU validation.
* Cross-platform CI (this is for the user's Linux box; CI workflows
  come later).
* Integration with `tests/test_*.py` pytest collection (the regression
  suite is parallel infrastructure, not pytest).

## Working style

* **Lead with the schema.** Define the per-system / per-basis /
  per-method input format first, then build the runner against it.
  Inputs are forever; runners can be rewritten.
* **One system end-to-end before all of them.** Get NaCl/sto-3g
  passing through the full stack (input → SCF → PySCF compare → CSV)
  before wiring up MgO and friends. Catches every architectural bug at
  minimum cost.
* **Verbose logs are documents.** Treat the verbose log as the
  artefact a domain expert reads to verify correctness — not a debug
  dump. Match the housestyle of `format_basis_filter_report` and
  `format_truncation_optimization_report`: titled blocks, aligned
  columns, one citation reminder per data-modifying step.
* **Don't write tests for the test runner itself** at first. The
  runner IS the test. Add unit tests once it's proven.

Today's date is 2026-05-02. Branch off main as
`feature/automated-test-suite` (or pick your own name). v0.7 work is
on `feature/v0.7-pyscf-pbc-parity` — that branch will be merged before
you reach wave 2, so just rebase on main when you start each new wave.
