---
myst:
  html_meta:
    "description": "vqfetch, pull crystal structures from open databases (OPTIMADE / Materials Project / COD / NOMAD) into vibe-qc with full per-record provenance."
    "og:title": "vibe-qc, external structures (vqfetch)"
    "og:description": "Round-trip recipe: vqfetch → emit SPEC + input script → SCF. Five canonical structures. Per-record license + DOI provenance. CC0 / CC-BY 4.0 sources."
---

# External structures (`vqfetch`)

`vqfetch` is the v0.8.0 console-script that pulls crystal
structures from open databases and emits two artefacts:

1. A **regression `PeriodicSpec` module** (under
   `examples/regression/systems/periodic/`) so the structure
   becomes part of the regression matrix.
2. An **executable vibe-qc input script** (under `examples/`)
   so you can run an SCF on the fetched cell with one command.

The pull preserves **full per-record provenance**: source DB,
ID, permalink URL, original DOI (where available), license
string, and fetched-at timestamp. This means a `vqfetch`-pulled
structure is publication-ready out of the box, no "where did
this come from" auditability gap.

## Sources

| Source | Default per-record license | Use it for | CLI subcommand |
|--------|---------------------------|------------|----------------|
| **OPTIMADE** federation | per-provider (varies) | formula-based federated query across providers | `vqfetch optimade --formula MgO` |
| **Materials Project** | CC-BY 4.0 | computed structures; routed through MP's OPTIMADE endpoint | `vqfetch mp --id mp-1265` |
| **COD** (Crystallography Open Database) | CC0 / public domain | experimentally determined CIFs | `vqfetch cod --id 1011027` |
| **NOMAD** | CC-BY 4.0 (data); CC0 (metadata) | computed materials data | use `--provider nomad` with `optimade` |
| **Canonical set** | (whichever the slug's primary provider applies) | the five round-trip-verified structures used as smoke tests | `vqfetch canonical mgo_rocksalt` |
| **Candidate discovery** | (whichever the provider specifies) | interactive search: list all matching structures with dedup + ranking | `vqfetch list-candidates --formula MgO` |

The `list-candidates` subcommand (v0.13.x, VFETCH-X1) queries the
OPTIMADE federation and displays a tabular report of distinct
candidate structures ranked by space-group plurality, so you can
pick the right polymorph before pulling:

```sh
$ vqfetch list-candidates --formula MgO
provider     id               sg           a (Å)  atoms  merged
-----------------------------------------------------------------------
mp           mp-1265          Fm-3m        4.1940      8       0
mp           mp-1009127       Pm-3m        2.7521      2       0
mp           mp-1191789       P6_3mc       3.2322      4       0

3 candidate(s) shown. Pick one:
  vqfetch optimade --formula MgO
  # OR id-lookup:  vqfetch optimade --optimade-id mp/mp-1265
```

Each row is a structurally distinct polymorph. When two providers
return the same structure, they're merged into one row with the
`merged` count tracking duplicates. The table is ranked by
space-group plurality, the most common space group across all
returned structures floats to the top.

Flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--formula` | (required) | `chemical_formula_reduced`, e.g. `"MgO"`. |
| `--provider` | `None` (federated) | Narrow to one provider: `--provider mp`. |
| `--max-results` | `10` | Maximum candidates to display. |
| `--no-dedup` | off | Show structurally-identical entries separately. |
| `--no-cache` | off | Bypass cache reads (still writes through). |
| `--cache-only` | off | Refuse live HTTP; fail on cache miss. |

The full license inventory is in
{ref}`external-data licensing <external-data-licensing>`.

## Install

`vqfetch` is part of the optional `[fetch]` extra:

```sh
pip install -e '.[fetch]'   # development install
# OR
pip install 'vibe-qc[fetch]' # once published
```

This pulls in `optimade>=1.0,<2`, `ase>=3.22`,
`beautifulsoup4>=4.12,<5`, and `lxml>=4.9`. Without the extra,
`vqfetch` will not be on `$PATH`.

## Quick start: round-trip MgO from the canonical set

```sh
# 1. Fetch the canonical MgO entry → emit SPEC + input script.
vqfetch canonical mgo_rocksalt --quick

# Output (one path per line; both are written to disk):
# examples/regression/systems/periodic/mgo_rocksalt.py
# examples/input-mgo_rocksalt-sto-3g.py

# 2. Run the SCF on compute-reference via vq (LDA + sto-3g converges in ~13 iters):
vq submit examples/input-mgo_rocksalt-sto-3g.py --cpus 16 --wall-time-seconds 14400
# returns a 12-char jobid; poll with `vq status <jobid>`.
```

The SCF run produces:

- `output-mgo_rocksalt-sto-3g.out`, banner, SCF trace, energy
  breakdown, orbital table, plus wall-clock timings.
- `output-mgo_rocksalt-sto-3g.system`, runtime manifest (TOML).
  Records vibe-qc version, host OS, OMP threads, library versions,
  and SCF wall time.
- `output-mgo_rocksalt-sto-3g.perf`, per-phase timing breakdown.

Reference: live compute-reference round-trip on 2026-05-09 produced
**E = −950.4204308512 Ha** (13 SCF iters, ~2h 20m on 16 cores).

## Five canonical structures (round-trip-verified)

The `canonical` subcommand walks a hand-curated five-structure
table that the v1 acceptance harness round-trips end-to-end on
every commit:

```sh
vqfetch canonical mgo_rocksalt    # MgO via Materials Project mp-1265
vqfetch canonical nacl_rocksalt   # NaCl via Materials Project mp-22862
vqfetch canonical lih_rocksalt    # LiH via Materials Project mp-23703
vqfetch canonical si_diamond      # Si via Materials Project mp-149
vqfetch canonical c_diamond       # C  via Materials Project mp-66
```

Use `--quick` to drop the recommended basis to `sto-3g` for a
fast smoke test (default behaviour is the heuristic-recommended
basis per § Recommended basis below).

The table itself lives at
[`python/vibeqc/fetch/canonical_set.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/fetch/canonical_set.py)
with per-entry expected space group + conventional-cell atom
count + lattice constant for sanity-check assertions in the
smoke harness.

## Per-record provenance

Every fetched record carries the following fields (visible in
the emitted `PeriodicSpec` module's `Provenance` dataclass and
echoed into the SCF run header):

```python
Provenance(
    source_db="OPTIMADE/mp",                                   # "OPTIMADE/<provider>" | "COD" | "CCCBDB" | "manual"
    source_id="mp-1265",                                       # provider-specific id
    source_url="https://optimade.materialsproject.org/structures/mp-1265",
    original_reference="10.17188/1199994",                     # DOI when known, "" otherwise
    license="CC-BY-4.0",
    fetched_at="2026-05-09T21:30:00Z",
    fetcher_version="0.1.0",                                   # vibeqc.fetch.__version__
    notes="",                                                  # free-form; aliases when dedup runs
)
```

When you re-run a calculation later, the provenance bundle
travels with the SPEC. **For published work**, cite the source
DB per its terms of use (Materials Project: cite per their
terms; COD: CC0, citation appreciated, not legally required;
NOMAD: cite the contributing author + NOMAD).

## Cache + offline mode

vqfetch caches every successful fetch on disk per XDG (default
root: `$XDG_CACHE_HOME/vibeqc/fetch/`, falling back to
`~/.cache/vibeqc/fetch/`). Subdirectories per source DB
(`OPTIMADE_mp/`, `COD/`, `CCCBDB/`, …), one JSON file per
`(source_db, source_id)`. Repeated calls do **not** re-hit the
API:

- **30 days** TTL for OPTIMADE / MP / NOMAD / CCCBDB.
- **Infinite** TTL for COD (CIFs are immutable post-publication).

Two relevant flags:

* `--no-cache`, bypass cache reads but still write through
  after a live fetch. Useful when you suspect the upstream
  record has been updated.
* `--cache-only`, refuse live HTTP entirely; fail fast if the
  record isn't in the cache. Useful for offline / reproducible
  runs (e.g. on cluster compute nodes without network).
  Equivalent env var: `VIBEQC_FETCH_CACHE_ONLY=1`.

Override the cache root with `$VIBEQC_FETCH_CACHE_ROOT` for
tests or per-project caches.

## Common flags

All structure subcommands accept the same emission flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--basis` | (heuristic; see below) | Override the recommended basis (e.g. `--basis pob-tzvp`). |
| `--method` | `rks-lda` (periodic), auto-`rhf` for molecules | SCF method baked into the emitted input script. Choices: `rhf`, `rks-lda`, `rks-pbe`, `rks-blyp`, `rks-b3lyp`. |
| `--quick` | off | Force the recommended basis to `sto-3g` regardless. Smoke-test mode. |
| `--out` | `examples/regression/systems/{periodic,molecules}/` | Output directory for the emitted SPEC module. |
| `--input-script` | `examples/` | Output directory for the emitted input script. |
| `--no-cache` | off | Bypass cache reads (still writes through). |
| `--cache-only` | off | Refuse live HTTP. |
| `--slug` | (auto-generated or canonical-set slug) | Override the emitted SPEC's `id` field and the output filename. |

## Recommended basis (heuristic)

When `--basis` is not specified, vqfetch picks per heuristic
([`python/vibeqc/fetch/heuristics.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/fetch/heuristics.py)):

| System type | Default basis | Reason |
|-------------|---------------|--------|
| Periodic (any composition) | `pob-tzvp` | Bredow / Peintinger-Vilela-Oliveira-Bredow periodic-tuned TZVP; the standard solid-state reference. For transition-metal systems: verify ECP coverage in your build. |
| Molecule | `def2-svp` | Standard molecular split-valence; bump to `def2-tzvp` manually if affordable. |
| `--quick` mode | `sto-3g` | Smoke-test minimum. |

Override at any time via `--basis <name>`. The bundled basis
inventory + per-family citations are at
[`docs/license.md`](../license.md#3a-bundled-basis-sets).

## SCF defaults baked into the emitted script

The fetcher picks SCF stability knobs per the regression-suite
patterns:

| Trigger | `default_initial_guess` | `default_damping` |
|---|---|---|
| Cell contains any of Li / Na / K / Mg / Ca / Al / Cl / transition metal | `SAD` | `0.85` |
| Otherwise (rare-gas, light-covalent: H / B / C / N / O / Si / F / …) | `HCORE` | `0.5` |

k-mesh seed by cell size:

| atoms in cell | `default_kmesh` |
|---|---|
| ≤ 4 | (4, 4, 4) |
| 5-20 | (2, 2, 2) |
| > 20 | (1, 1, 1) |

These are **seeds for a convergence study**, not converged
values. The emitted script includes a `# TODO: k-mesh
convergence` comment referencing
`examples/input-k-mesh-convergence.py`.

## What the emitted files look like

After `vqfetch canonical mgo_rocksalt --quick`:

**`examples/regression/systems/periodic/mgo_rocksalt.py`**, a
drop-in `PeriodicSpec` the regression suite picks up
automatically:

```python
"""mgo_rocksalt — fetched rocksalt (OPTIMADE/mp/mp-1265).

Auto-generated by vibeqc.fetch (fetcher version 0.1.0).
Do not edit by hand — re-run the fetcher to regenerate.
"""
from __future__ import annotations
from examples.regression.core.spec import AtomFrac, PeriodicSpec, Provenance

SPEC = PeriodicSpec(
    id="mgo_rocksalt",
    family="rocksalt",
    lattice_ang=(
        (4.19400279, 0.0, 0.0),
        (0.0, 4.19400279, 0.0),
        (0.0, 0.0, 4.19400279),
    ),
    space_group="Fm-3m",
    atoms=(
        AtomFrac(symbol="Mg", z=12, frac=(0.0, 0.0, 0.0)),
        AtomFrac(symbol="Mg", z=12, frac=(0.0, 0.5, 0.5)),
        AtomFrac(symbol="Mg", z=12, frac=(0.5, 0.0, 0.5)),
        AtomFrac(symbol="Mg", z=12, frac=(0.5, 0.5, 0.0)),
        AtomFrac(symbol="O",  z=8,  frac=(0.5, 0.5, 0.5)),
        AtomFrac(symbol="O",  z=8,  frac=(0.5, 0.0, 0.0)),
        AtomFrac(symbol="O",  z=8,  frac=(0.0, 0.5, 0.0)),
        AtomFrac(symbol="O",  z=8,  frac=(0.0, 0.0, 0.5)),
    ),
    default_kmesh=(2, 2, 2),
    default_initial_guess="SAD",
    default_damping=0.85,
    recommended_basis="sto-3g",     # --quick was set; default is pob-tzvp
    is_open_shell=False,
    provenance=Provenance(
        source_db="OPTIMADE/mp",
        source_id="mp-1265",
        source_url="https://optimade.materialsproject.org/structures/mp-1265",
        original_reference="",
        license="CC-BY-4.0",
        fetched_at="2026-05-09T21:30:00Z",
        fetcher_version="0.1.0",
        notes="",
    ),
)
```

**`examples/input-mgo_rocksalt-sto-3g.py`**, runnable SCF script
mirroring the style of `examples/input-mgo-pob-tzvp.py`:

```python
"""mgo_rocksalt (rocksalt) — auto-generated by vibeqc.fetch."""
import os, time
from pathlib import Path
import numpy as np
import vibeqc as vq
from vibeqc.progress import ProgressLogger

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
LATTICE_ANG = np.array([[4.194, 0, 0], [0, 4.194, 0], [0, 0, 4.194]])
MG_FRAC = [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
O_FRAC  = [(0.5,0.5,0.5), (0.5,0,0), (0,0.5,0), (0,0,0.5)]

plog = ProgressLogger(log_path="output-mgo_rocksalt-sto-3g.out", verbose=True)
with vq.perf_log("output-mgo_rocksalt-sto-3g.perf"):
    lat_bohr = LATTICE_ANG * ANGSTROM_TO_BOHR
    unit_cell  = [vq.Atom(12, [fx*lat_bohr[0,0], fy*lat_bohr[1,1], fz*lat_bohr[2,2]])
                  for fx, fy, fz in MG_FRAC]
    unit_cell += [vq.Atom(8,  [fx*lat_bohr[0,0], fy*lat_bohr[1,1], fz*lat_bohr[2,2]])
                  for fx, fy, fz in O_FRAC]
    system = vq.PeriodicSystem(dim=3, lattice=lat_bohr, unit_cell=unit_cell)
    basis  = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    vq.attach_symmetry(system, symprec=1e-4)

    opts = vq.PeriodicKSOptions()
    opts.functional = "LDA"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.damping = 0.85
    opts.initial_guess = vq.InitialGuess.SAD
    kpts = vq.KPoints.monkhorst_pack(system, [2, 2, 2], symmetry=True)
    result = vq.run_rks_periodic_scf(system, basis, kpts, opts, progress=plog)
```

The emitted script uses vibe-qc's native periodic-SCF API
(`vq.PeriodicSystem`, `vq.PeriodicKSOptions`,
`vq.run_rks_periodic_scf`, …), not a higher-level wrapper.
That keeps every SCF knob visible at the call site, which is
the right tradeoff for a benchmark workload that the user will
inevitably want to tweak.

## Combining with the regression suite

Once a SPEC lands in `examples/regression/systems/periodic/`,
the regression suite picks it up automatically. Run a focused
matrix entry:

```sh
python -m examples.regression.run_suite \
    --systems mgo_rocksalt --bases sto-3g --methods rks-lda
```

…and the new system shows up alongside the hand-curated test
set, with the per-source provenance footer included in the
generated `summary.md`.

## Multi-candidate Python API

For interactive workflows (notebooks / scripts) where you want
to see ALL polymorphs a formula returns and pick by hand:

```python
from vibeqc.fetch import fetch_optimade

candidates = fetch_optimade(
    formula="MgO",
    max_results=10,         # walk all hits, not just the first
    dedup=True,             # structurally-identical entries merged
)                           # → list[PeriodicSpec] sorted by space-group
                            #   plurality + larger-cell tiebreak

for spec in candidates:
    p = spec.provenance
    print(f"{p.source_db}/{p.source_id}: sg={spec.space_group} "
          f"a={spec.lattice_ang[0][0]:.3f} Å  natoms={len(spec.atoms)}")
```

Structurally identical entries across providers get merged
into one survivor, `Provenance.notes` carries the alias list
(`"also at: OPTIMADE/oqmd/oqmd-1234, OPTIMADE/aflow/aflow:abc"`).
Dedup hashes the formula + cell + sorted-rounded fractional
positions to 1e-4 Å resolution. Non-orthorhombic entries are
silently dropped in the multi-candidate path (the single-pick
path raises with the full rejection list).

A CLI surface for this (`vqfetch list-candidates`) is in the
v0.8.x maintenance line.

## When NOT to use vqfetch

* **You already have a CIF on disk.** Use ASE's CIF reader
  directly: `ase.io.read("my.cif")` then build a
  `PeriodicSystem` from the result. vqfetch's value is the
  fetch + provenance + emission round-trip, not the CIF
  parsing itself.
* **You want a structure NOT in any open database.**
  vqfetch's sources cover ~99% of well-studied solids; for
  exotic / proprietary structures, hand-build the
  `PeriodicSystem` and document provenance manually.
* **You need a non-orthorhombic cell** (monoclinic /
  triclinic). vqfetch standardises through
  `spglib.standardize_cell(..., to_primitive=False)`; if the
  conventional setting is also non-orthorhombic, it refuses
  with a clear message rather than silently misconfigure the
  Ewald lattice sum. Triclinic-Ewald support is on the
  roadmap.
* **You need cell relaxation.** vqfetch emits a static-SCF
  script; geometry optimisation is a follow-on step the user
  drives.

## Deferred to the v0.8.x maintenance line

These items are designed but not shipped in v0.8.0:

* **`vqfetch list-candidates`** CLI subcommand (Python
  multi-candidate API shipped in v0.8.0; the CLI surface
  follows).
* **WebBook fallback** for structures missing from MP / COD.
* **Materials Project property fields** beyond what OPTIMADE
  surfaces (`client_mp.py` is currently a thin OPTIMADE pass-
  through; band gap / formation energy / magnetisation pull
  follows when the v0.8.x line lands).
* **NOMAD raw-archive hook** (`client_nomad.py` is a stub
  today; pulls the input-archive URL into `Provenance.notes`
  when it lands).
* **Cross-provider consistency check** (alert when MP / COD /
  NOMAD disagree on the same canonical structure).
* **Bulk-sweep tooling** (`vqfetch --bulk <list-file>`) for
  populating the cache with a curated reference set.
* **`runner_crystal.py`** for the regression suite, runs
  CRYSTAL14 via vq subprocess (textbook external-codes
  pattern); unlocks three-way periodic parity
  (vibe-qc / PySCF.pbc / CRYSTAL) for fetched SPECs.

These are tracked in `docs/roadmap.md` under the v0.8.x
maintenance window, none blocking the v0.8.0 tag.

## See also

* [`reference_data.md`](reference_data.md), vqfetch's
  CCCBDB integration for **experimental reference data**
  (atomization energies, ΔHf°, vibrational frequencies, IE).
* [`external_codes.md`](external_codes.md), vibe-qc's policy
  on external programs vs vendored libraries; how parity
  runs against PySCF / ORCA / CRYSTAL are wired.
* [`docs/license.md`](../license.md), full per-source
  licensing + bundled-data inventory.
