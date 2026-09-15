# CRYSTAL23 demo system inputs

Reference inputs from the official CRYSTAL distribution
([crystal.unito.it/test_demo/inputs/](https://crystal.unito.it/test_demo/inputs/),
linked from [crystalsolutions.eu/try-it.html](https://www.crystalsolutions.eu/try-it.html)).
Mirrored here so the queue chat — and any of the other dev chats —
have an in-repo set of small reference systems for smoke testing
the `crystal23demo` + `properties23demo` binaries (v0.6.3+).

## What runs on the demo

CRYSTAL23 demo (`~/bin/crystal23demo`, `~/bin/properties23demo` on
compute-small, compute-reference, and the user's dev laptop) is **feature-complete
relative to the paid v23 license** — the only restriction is a
**10-atom-per-primitive-cell limit**. Every input below sits at or
below that limit (some by symmetry expansion of a smaller
asymmetric unit; see § "Atom-count caveat" if a run errors out
with "demo limit exceeded").

The demo is **serial-only** — no `Pcrystal23demo` ships. Run via
the wrapper's `--demo` flag (which rejects `--np`).

## The 15 systems

| File | Dimensionality | Asym-unit atoms | Description |
|---|---|---|---|
| `be_sto3g.d12` | bulk (3D) | 1 | Beryllium bulk, STO-3G (hex, sg 194) |
| `bebulk.d12` | bulk (3D) | 1 | Beryllium bulk, extended Gaussian basis |
| `mgo_sto3g.d12` | bulk (3D) | 2 | MgO bulk (rocksalt), STO-3G |
| `mgo_bulk.d12` | bulk (3D) | 2 | MgO bulk (rocksalt), extended basis |
| `nio_sto3g.d12` | bulk (3D) | 2 | NiO bulk ferromagnetic, STO-3G |
| `nio.d12` | bulk (3D) | 2 | NiO bulk ferromagnetic, extended basis |
| `sibulk_sto3g.d12` | bulk (3D) | 1 | Si bulk (diamond structure), STO-3G |
| `sibulk.d12` | bulk (3D) | 1 | Si bulk, extended basis |
| `diamond_sto3g.d12` | bulk (3D) | 1 | C diamond, STO-3G |
| `diamond.d12` | bulk (3D) | 1 | C diamond, 6-21G modified + polarization |
| `graphite_sto3g.d12` | slab (2D) | 1 | Graphite monolayer, STO-3G |
| `mgo001.d12` | bulk (3D) | 2 | MgO (001) surface, monolayer model |
| `mgo001co.d12` | bulk (3D) | 2 | CO adsorbed on MgO (001) surface |
| `sn_polym_sto3g.d12` | polymer (1D) | 2 | (SN)x polymer, STO-3G |
| `urea_bulk_321G.d12` | bulk (3D) | 5 | Urea bulk, 3-21G |

All inputs are SCF (`.d12`). The page does not ship a PROPERTIES
post-processing input (`.d3`) — use any of the above to seed
`fort.9`, then write your own `.d3` for the post-processing step.

## Atom-count caveat

The demo limit is **10 atoms per *primitive* cell**, and the
asymmetric unit (the count CRYSTAL parses from the input file's
geometry block) is generally smaller than the symmetry-expanded
primitive cell. The systems above are listed on the official
try-it page as demo-runnable, so the curated atom-count fits
under the cap even after symmetry expansion. If you run a custom
input and CRYSTAL exits with "exceeded demo limit," the
asymmetric-unit count was deceiving.

## Submit-via-vq

Install the separate [vibe-queue repository](https://github.com/vibe-qc/vibe-queue)
using the [queue setup guide](../../docs/user_guide/queue.md). The wrapper paths
below refer to its checkout on the execution host; adapt them to that host.

Single-input submission, with the wrapper choosing the demo
binary via `--demo` (which forces serial, rejects `--np`):

```sh
vq submit <host> -d ./examples/crystal23_demo/<system>/ \
    --cpus 1 --wall-time-seconds 3600 -- \
    env CRYSTAL23DEMO_BIN=/home/USER/bin/crystal23demo \
    bash /home/USER/gitlab/vibe-queue/contrib/run-crystal.sh \
    --demo <system>.d12
```

Per `chat-onboarding.md` § "Common job patterns" — the `env`
prefix sets the absolute binary path because the systemd-user
daemon's PATH on compute-small/compute-reference does NOT include `~/bin/` (where
the demo binary lives). Same recipe for the laptop, just dispatch
locally instead of `--host compute-small`/`compute-reference`.

For PROPERTIES23 demo post-processing (write your own `.d3` per
the CRYSTAL manual):

```sh
vq submit <host> -d ./mycalc \
    --cpus 1 --wall-time-seconds 1800 -- \
    env PROPERTIES23DEMO_BIN=/home/USER/bin/properties23demo \
    bash /home/USER/gitlab/vibe-queue/contrib/run-crystal.sh \
    --demo --properties myinput.d3
```

## Why these are here

Three audiences:

1. **Smoke tests.** A queue chat verifying `crystal23demo` after a
   fleet config change (v0.6.3+) needs a known-good input
   under the demo cap. `mgo_sto3g.d12` (the smallest extended
   system, 2 atoms, STO-3G) runs in seconds and exercises the
   3D-periodic SCF path.
2. **Cross-code reference for vibe-qc parity work.** The
   `crystal_parity/` regression suite uses small CRYSTAL inputs
   (LiH, MgO, NaCl, …) for vibeqc-vs-CRYSTAL14 comparisons. The
   v23 demo inputs are independent: same chemistry, official
   reference, validates that vibe-qc's results converge to the
   CRYSTAL family across CRYSTAL14 → CRYSTAL23 too.
3. **Operator handhold.** New chats / collaborators landing in the
   project get a directory of "runs immediately on the demo" inputs
   without having to chase URLs and CRYSTAL-manual conventions.

## Provenance

Inputs fetched 2026-05-17 from `https://crystal.unito.it/test_demo/inputs/`
(the Univ. of Torino mirror — Roetti / Dovesi's home institution
and the canonical CRYSTAL distribution since the early 1990s).
Linked from the CrystalSolutions try-it page
`https://www.crystalsolutions.eu/try-it.html`. Files are
unmodified — same byte contents as the upstream copies on the
fetch date.

## License + redistribution

Per CLAUDE.md § 1: these inputs are reference test cases from a
third-party project. Their use in these vibe-qc examples is **as test
fixtures** (smoke + parity), not as bundled-and-redistributed
content. The CRYSTAL inputs themselves are short text files
documenting standard test systems; the relevant licensing is the
CRYSTAL23 demo binary's own terms (use of the demo for any
purpose is explicitly permitted under the 10-atom-cell cap per
the try-it page). If a future release tightens the redistribution
story, switch to on-demand fetch (the `vqfetch` pattern from
CLAUDE.md § 1 "on-demand fetcher pattern").
