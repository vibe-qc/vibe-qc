# Versioning — vibe-basis

`vibe-basis` carries its **own** version line, independent of vibe-qc's.
This matches how the other co-located sibling packages in this monorepo
work: `vq` (`vibe-queue/`) is at 0.24.0 and `vibeview` (`vibe-view/`) is
at 2.14.1 while vibe-qc is at 0.15.119. A vibe-qc release does not imply a
vibe-basis release, and vice versa.

## Why independent

vibe-basis is a **driver**, not a quantum-chemistry program (see
`README.md` § "Why a separate package?"). Its compatibility surface is
the external programs it drives (CRYSTAL23, later ORCA / Gaussian) and
its own public API — neither of which moves when vibe-qc's SCF internals
do. Tying its number to vibe-qc's would make every vibe-qc patch look
like a vibe-basis change to anyone reading a run's provenance.

## Scheme — SemVer, currently pre-1.0

`MAJOR.MINOR.PATCH`.

| Bump | When |
|---|---|
| **MAJOR** | Breaking change to the public API (`vibe_basis.backends`, `.transports`, `.optimize`, `.io`, `.recipes` signatures) or to the `vb` CLI surface. Also: any change to an optimizer default that moves the converged parameters of an existing recipe. |
| **MINOR** | New backend, transport, optimizer, recipe, or `vb` subcommand. New optional extra. New packaging surface (e.g. becoming installable via vibe-qc's `[basisopt]` extra). Behaviour additions that leave existing runs unchanged. |
| **PATCH** | Bug fixes, parser robustness against real-world output, docs, tests, internal refactors with no API or numerical effect. |

**While the version is `0.x`, MINOR carries breaking changes** — the
standard SemVer 0.x allowance. Treat a `0.x` MINOR bump as "read the
CHANGELOG entry before upgrading a running optimization".

**1.0.0 criterion:** cut 1.0.0 when the first production basis set is
published from this driver (`mpei-TZVP`, see
`docs/basisset_dev/GOAL8_MPEI_TZVP.md`) **and** the
`backends` / `transports` `(emit_input, run, parse_output)` triple is
frozen. Publishing a basis set is a scientific commitment to
reproducibility; the API that produced it should stop moving at the same
moment.

## Single source of truth

Two places carry the number and **must** match:

* `vibe-basis/pyproject.toml` → `[project] version`
* `vibe-basis/src/vibe_basis/__init__.py` → `__version__`

`vb --version` reads the second (`cli.py`'s `click.version_option`), so
a drift between them makes the CLI lie about what is installed.
`tests/test_versioning.py` pins the two together — it fails the moment
one is bumped without the other.

## When to bump

This follows the standing decision of 2026-07-23, documented in
[`docs/release_process.md`](../docs/release_process.md) § "Sibling
modules at vibe-qc release time". vibe-basis does not get its own
policy here; this section only records how that policy lands on this
module.

* **The release chat owns the backstop.** At every vibe-qc `vX.Y.Z`
  tag it diffs each sibling since its last version-bump commit and
  bumps anything where user-visible work landed, in the tree it tags,
  so nothing ships version-stale. That ownership is theirs, not the
  implementing chat's.
* **The implementing chat may — and should — bump proactively.** If
  you land user-visible work here, bumping in the same merge is
  encouraged and saves the backstop a decision. Patch for fixes, minor
  for features; docs-only or refactor-only ranges get no bump, the
  same rule vibe-qc itself follows.
* **Either way, update both version sites together.**
  `docs/release_process.md` § 3 lists them for vibe-basis as
  `pyproject.toml` and `src/vibe_basis/__init__.py`;
  `tests/test_versioning.py` now enforces that pair mechanically, so a
  half-bump fails a test rather than shipping.

The version reaches the release record automatically:
`vibe-queue/scripts/make_release_report.py` reads
`vibe-basis/pyproject.toml` at the peeled release commit and emits it as
`sibling_versions_at_release.vibe_basis`.

Historical note for anyone reading old release commits: vibe-basis sat
at `0.1.0` from spin-up until 2026-07-26, and the backstop commits
record *"vibe-basis unchanged (0.1.0, no commits since its last bump)"*.
That was accurate — the module genuinely had not moved.

## Deployment status

vibe-basis is **not** a managed vq program and has no fleet rollout
lane — by design, because it is a client that submits *to* the fleet
rather than a payload that runs on it. Since v0.2.0 it is installable
into the vibe-qc venvs via vibe-qc's `[basisopt]` extra, which is what
makes its version verifiable on a host (`importlib.metadata`) rather
than merely present as source. The full reasoning is in
[`handovers/HANDOVER_VIBE_BASIS_DEPLOYMENT.md`](../handovers/HANDOVER_VIBE_BASIS_DEPLOYMENT.md).

## History

| Version | Date | What |
|---|---|---|
| 0.12.0 | 2026-09-09 | **M3 (code).** `vibe_basis.topology`: an immutable `BasisTopology` (element -> shells, mirroring `CrystalShell` field for field) with `add_shell` / `drop_shell` / `split_contraction`, the roadmap's four-move scan (`baseline`, `add-high` at a1*r, `add-low` at an/r, `leave-one-out` per shell) with r the local even-tempered ratio, `pareto_front` over accuracy/cost/conditioning, and `greedy_scan` with the plateau rule and an optional beam. Candidate evaluation and the `BasisParametrisation` bridge are injected, keeping Tier 1 free of vibeqc. Mutators return new topologies: a search branches, and in-place edits would corrupt the parent. **M3's gate (scan Mg-d in MgO and Si-d in SiC, rediscovering the pob topology on total energy) is NOT run** -- it needs an L2 relaxation per candidate. Additive only. |
| 0.11.0 | 2026-09-09 | **M2 (code).** `vibe_basis.objective`: `CohesiveObjective` scores candidate cohesive energies against a reference with per-system weights, four losses (RMSD default, MAD, max-|delta|, Huber) and a deterministic held-out split whose error is always reported alongside the training one. `load_pw_reference` reads the `pwref_merged.json` layout; the path is the caller's, since Tier 1 cannot reach into vibe-qc's `studies/`. Penalties are injected rather than imported, keeping the `ld_penalty` seam on the Tier-2 side. A candidate that fails on a system cannot be silently dropped: there is no `skip` policy, because dropping unconverged systems would make the objective improve as the basis got worse enough to break them. **M2's own gate (reproducing MD -24.6 / MAD 35.4 kJ/mol at pob-TZVP-REV2 over n=28) is NOT run** -- it needs 28 periodic cohesive runs. Additive only; no existing run changes. |
| 0.10.0 | 2026-08-08 | Adds a complete standalone macOS/Linux lifecycle: safe install, Git-aware update, rollback-capable reinstall, and marker-guarded uninstall. The default `standard` profile includes SciPy, `[test]` now matches the test suite, and the compatibility `[vq]` extra no longer asks public indexes for an unpublished package; vq stays a separately managed CLI or can be installed from the co-located checkout with `--with-vq` on Python 3.12+. |
| 0.9.0 | 2026-07-28 | **Counterpoise decks work on non-symmorphic space groups.** `emit_input_atom_counterpoise` now emits `ORIGIN` + `TRASREMO` before `ATOMBSSE`, as the reference set's own diamond decks do. Without them, `ATOMBSSE` on a non-symmorphic group leaves symmetry operators that cannot close (`MULTIP: SYMMOPS DO NOT FORM A GROUP`), which killed every SG-227 system (C-diamond, Si, Ge) in the ghost-shell sweep at every rung. Emitted unconditionally: measured bit-identical on a symmorphic system, so there is nothing to gate on. Opt out with `trasremo=False`. |
| 0.8.0 | 2026-07-26 | **Decks now match the pob reference protocol.** `TOLINTEG 9 9 9 18 54` is emitted by default (every one of the reference set's 1759 decks uses it; we emitted none). `HUGEGRID` is now **off** by default and opt-in via `hugegrid=True`: it was added on the manual's recommendation without checking that no reference deck uses it. The grid choice was also asymmetric between the bulk and atom emitters, which does not cancel in a cohesive energy; all three now agree, pinned by a test. An emitted single point reproduces the reference LiCl bulk energy at r2SCAN to 0.012 kJ/mol. Breaking: default deck contents changed. |
| 0.7.0 | 2026-07-26 | **Every emitted CRYSTAL deck was malformed.** First run against a real CRYSTAL23 found five input-parse defects: missing `ENDBS` after the ` 99 0` basis sentinel; blank lines read as empty keywords; space group 1 written with one lattice parameter instead of six; `SHRINK 0 0` on a 3D box; and no r2SCAN in the functional list, so the functional the pob reference set uses was rejected. SCAN-family decks now also request `HUGEGRID` per the manual. Pinned by `tests/test_deck_accepted_by_crystal.py`, which feeds a deck to CRYSTAL when one is installed and asserts **positively** that a basis was built. |
| 0.6.0 | 2026-07-26 | **Counterpoise free atoms**, matching the published pob protocol. `emit_input_atom_counterpoise` emits an `ATOMBSSE` deck: the atom computed in the ghost basis of its own crystal, which is how every free-atom deck in the cohesive reference set is built, so a cohesive energy assembled from bare atoms was not the same quantity as the published one. `EnergyEngine.atom_energy` gains `host=`, `"counterpoise"` joins the capability set, `CohesivePipeline` gains `counterpoise=` and `CohesiveResult.counterpoise_applied`. Cache keys include the host, which **deliberately disables cross-system atom sharing** for counterpoise atoms: MgO's oxygen and CaO's oxygen are genuinely different calculations. Breaking: `atom_energy` implementations must accept `host`. |
| 0.5.0 | 2026-07-26 | **Correctness fix, changes numbers.** The isolated-atom CRYSTAL deck emitted no `SYMMREMO`, no `UHF`/`SPIN` and no `SPINLOCK`, so every open-shell free atom was computed spin-restricted and spherical. Those SCFs converge cleanly, so the error was silent, and it corrupts exactly the free-atom half of a cohesive energy: the plane-wave work measured 37 kJ/mol for spin-restricting one bromine reference. `emit_input_atom` now emits all three for an open shell, gains `n_unpaired` / `spinlock_cycles`, exports `GROUND_STATE_MULTIPLICITY`, and **raises** for an untabulated element instead of guessing. `Crystal23Engine` reports `atom_reference` describing the occupancy rather than just the box. Per CRYSTAL23 manual: *"UHF and SPINLOCK must be used to define a reasonable orbital occupancy."* |
| 0.4.0 | 2026-07-26 | **M0a** — the cohesive-energy pipeline API. `pipeline.CohesivePipeline` (relax → single point → ZPE → isolated atoms → assemble, engine-neutral) returning a `CohesiveResult` that records per-stage status and whether ZPE was applied; `cache.EnergyCache`, content-addressed with an append-only JSONL journal, sharing free-atom energies across every system in a campaign and resuming after a kill. `EnergyEngine` gains optional `relax` / `zero_point` capabilities plus `supports()`, so an engine that lacks one causes a *recorded skipped stage* rather than a silently defaulted value. M0's remaining work is its two-engine acceptance gate, which is evidence rather than API and earns no further bump. |
| 0.3.0 | 2026-07-26 | **M-0.** `EnergyEngine` protocol + `EngineEnergy` provenance record; `Crystal23Engine` (moved here from vibe-qc's `calculators.py`, where it was tier-1 code on the tier-2 side) and a declared-but-unbuilt `GpawEngine`. CRYSTAL backend **retargeted to CRYSTAL23**: `backends/crystal14{,_atom}.py` → `backends/crystal{,_atom}.py`, plus `probe_version()` and `CrystalEnergyResult.crystal_version` so a result records which CRYSTAL produced it. `vb parse` takes `crystal` (canonical) with version-suffixed aliases and prints the detected version. Emitted `.d12` titles now say "CRYSTAL23 parity". Import-direction guard added (`tests/test_no_vibeqc_dependency.py`). Breaking: the `backends.crystal14*` module paths are gone (0.x MINOR allowance). |
| 0.2.0 | 2026-07-26 | Installable into the vibe-qc venvs via vibe-qc's `[basisopt]` extra; `vibeqc.basis_optimization`'s six `vibe_basis`-importing modules become importable wherever vibe-qc is installed. This versioning scheme adopted. |
| 0.1.0 | 2026-05-13 | Spun up as a standalone driver: CRYSTAL14 backend, `local` + `vq` transports, structure/reference databases, `vb` CLI, pob-parity Stage 0 recipe. |
