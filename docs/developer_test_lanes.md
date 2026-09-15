# Developer Test Lanes

vibe-qc has outgrown a single "run everything" development loop. The executable
gate uses isolated pytest targets: usually whole files, and for the blocking
release-core profile selected pytest node ids where a whole file is too slow.
The manifest partitions those targets into lanes so a chat runs the tests for
the area it touched.

The lane manifest is:

```text
scripts/test_gate/lane_manifest.json
```

The lane-aware runner is:

```sh
PY=.venv/bin/python
WT=$PWD

$PY scripts/test_gate/run_full_suite.py --wt "$WT" --list-lanes
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --list-profiles
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --lane pbc-gdf --dry-run
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --changed-since v0.15.41 \
    --list-affected-lanes
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" --lane smoke
```

## Rule

For ordinary development:

1. Run `smoke` only for native-binding or test-gate policy changes.
2. Run every area lane that matches the files you changed. This is the normal
   pre-push requirement; the complete repository inventory is not.
3. For release and patch candidates, run the blocking `release-core` profile
   and `--changed-since <base> --impact-mode release`. The release impact gate
   runs only explicitly selected cheap release-impact lanes; the clean native
   build and import supply T0.
4. Run advisory release-profile lanes separately only for nonblocking status on
   non-AICCM areas. AICCM/CCM/SECCM are experimental manual research lanes and
   are intentionally absent from release advisory profiles.
5. Use the default `--impact-mode dev` or explicit `--lane <area>` while
   developing, when the changed scientific area needs the full calculation
   lane. Do not make those calculation lanes release blockers by default.
6. Run `full-fast` only when the change touches shared infrastructure, core
   numerical primitives, pytest/gate machinery, final pre-tag confidence, or
   when no recent nightly covers the tree.
7. Run `slow-nightly` and `external-reference` before release tags, or when
   those areas are touched.
8. Keep AICCM/CCM/SECCM out of release, advisory-profile, and automatic impact
   gates. Run those lanes only by explicit manual research command.

This means AICCM real-Gamma/Gamma/chi, CCM, MSINDO-CCM, and SECCM are not part
of release testing or automatic impact testing. PBC lanes are not run for
molecular-only work. Molecular semiempirical and periodic semiempirical lanes
are separate. GAPW, BIPOLE, GDF, embedding, and external reference-code lanes
are independent unless a change crosses those boundaries.

## Policy Metadata

Every lane in `scripts/test_gate/lane_manifest.json` carries the release-policy
metadata that other chats should consume:

- `method_maturity`: exactly one of `production`, `verified`,
  `under-review`, or `experimental`.
- `lane_class`: exactly one of `pre-cut blocking`, `post-cut evaluation`, or
  `implementing-chat-only`.
- `global_items`: the release policy, bug, or gated-item identifiers connected
  to the lane.
- `target_release`: the release line the lane currently supports, written as
  `vMAJOR.MINOR.x` for a release line (for example `v0.18.x`) or
  `vMAJOR.MINOR` for a forward track (for example `v2.0`). Any other form is
  rejected when the manifest loads.
- `required_full_calculation`: the full input/output matched-reference
  calculation that is required before a scientific value can be promoted.
- `scientific_acceptance`: always `false` for pytest lanes.

A `target_release` older than the repository's current line, read from the
`[project] version` in `pyproject.toml`, is stale. Forward tracks are never
stale. Stale lanes whose owning chats have not yet retargeted them are listed
in `policy.known_stale_target_release`, which works like
`known_reds_baseline.json`: a stale lane missing from the list, a listed lane
that is no longer stale, or a listed name that is not a lane needs action. When
a chat retargets its lane, it removes the lane from that list in the same
change.

Staleness is reported on stderr whenever a lane run loads the manifest, but it
never fails that run, because every version bump makes the previous line stale
at once. The enforceable form is
`scripts/test_gate/run_full_suite.py --wt . --check-lane-manifest`, which exits
non-zero when action is needed. It imports nothing from `vibeqc`, so any CI job
can run it.

The runner validates those fields when it loads the manifest, and
`tests/test_test_gate_lanes.py` pins the contract. Pytest lanes and CI are
mandatory change-safety evidence, but they never accept a method or paper
value. Full production-shaped calculations with complete input, verbose
vibe-qc output, raw reference output, artifacts, and matched-reference analysis
remain the scientific evidence.

`pre-cut blocking` lanes are the default one-hour patch-release critical path.
`post-cut evaluation` lanes run on touch, in advisory/nightly/final confidence
contexts, or after the immutable cut as selected by the agentic-loop policy.
`implementing-chat-only` lanes are experimental and must not enter release,
advisory-profile, or automatic impact gates until their owner promotes them.

## Lanes

| Lane | Use it for |
| --- | --- |
| `smoke` | Native-binding or test-gate policy changes. |
| `release-core-cheap` | Blocking release profile lane for cheap shipped, non-experimental feature coverage. Uses selected pytest node ids for otherwise heavy files. |
| `output-docs` | `vibeqc.output`, citations, QVF, manifest, docs-symbol, golden output, and formatting work. |
| `molecular-scf-dft` | Molecular HF/DFT, grids, SCF convergence, molecular gradients, Hessians, properties, dispersion. |
| `molecular-correlation` | MP2, CC, DLPNO, CAS, CI, TDDFT, excited-state, and solver work. |
| `basis-ecp-integrals` | Basis sets, ECPs, integral kernels, DF core, symmetry integrals, and basissetdev architecture. |
| `semiempirical-molecular` | Molecular MSINDO, PM6/OMx, GFN2/xTB, DFTB0, runner/output, PES, MD, NEB, and route-plan tests. |
| `semiempirical-periodic` | Periodic and k-point semiempirical adapters, DFTB smearing/convergence/FD, and periodic native route checks. |
| `semiempirical` | Aggregate semiempirical lane. Prefer a narrower semiempirical sublane unless the refactor crosses all of them. |
| `pbc-core` | Periodic core excluding GDF, BIPOLE, GAPW/GPW, embedding, and megacell. |
| `pbc-gdf` | Periodic GDF, RSGDF, MDF/compcell, AO-pair FT, and k-point GDF. |
| `pbc-bipole` | BIPOLE Ewald split, screened exchange, pair-resolved SR domains, and BIPOLE-backed periodic NEB. |
| `pbc-gapw-gpw` | GAPW/GPW, plane-wave grids, augmentation, stress, phonons, PW reference, FFT-Poisson. |
| `embedding-surfaces-neb` | Periodic embedding, slabs, surface reactions, ASE periodic calculators, and NEB paths. |
| `external-reference` | Out-of-process PySCF/ORCA/CRYSTAL/CP2K/GPAW parity boundaries and `examples/regression` runner tests. |
| `full-fast` | Non-slow shipped-feature pytest files, excluding experimental AICCM/CCM/SECCM. |
| `slow-nightly` | Slow/heavy shipped-feature tests on a large box, excluding experimental AICCM/CCM/SECCM. |

## Examples

Molecular runner or SCF change:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane smoke --out smoke.jsonl
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane molecular-scf-dft --out molecular-scf-dft.jsonl
```

Periodic GDF change:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --lane pbc-gdf --dry-run
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane smoke --out smoke.jsonl
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane pbc-gdf --out pbc-gdf.jsonl
```

Molecular semiempirical change:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --lane semiempirical-molecular --dry-run
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane smoke --out smoke.jsonl
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane semiempirical-molecular --out semiempirical-molecular.jsonl
```

Periodic semiempirical change:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --lane semiempirical-periodic --dry-run
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane smoke --out smoke.jsonl
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane semiempirical-periodic --out semiempirical-periodic.jsonl
```

Patch or release-candidate blocking gate:

```sh
BASE=v0.15.41
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --tier T1 --out blocking.jsonl
$PY scripts/test_gate/gate_verdict.py blocking.jsonl \
    scripts/test_gate/known_reds_baseline.json --wt "$WT" --py "$PY"
$PY scripts/test_gate/run_full_suite.py --wt "$WT" \
    --changed-since "$BASE" --impact-mode release --list-affected-lanes
$PY scripts/test_gate/run_full_suite.py --wt "$WT" \
    --changed-since "$BASE" --impact-mode release --dry-run
```

The blocking gate is the **T1 tier**, matching what `build-test` runs on a
tag or a `release-candidate/*` branch, so a local pre-flight and CI cannot
disagree about what green means.

Do **not** pair `--profile release-core --profile-mode blocking` with
`gate_verdict.py` here. That selection includes `tests/test_binding_sanity.py`,
which is tier T2, and the blocking verdict accepts only T0/T1, so the run
aborts with `blocking verdict accepts only T0/T1 records; found T2` even when
every selected test passes. If a binding-sanity check should in fact be
release-blocking, retier it to T1 rather than loosening the verdict. The
profile form is still fine for `--dry-run` inspection and for
`--profile-mode advisory`, neither of which feeds the blocking verdict.

The two `--impact-mode release` calls stay informational: they name the area
lanes the change touched, which the owning dev chat runs on touch.

Advisory nonblocking status for non-AICCM lanes:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --profile release-core --profile-mode advisory --out release-advisory.jsonl
```

Do not include `release-advisory.jsonl` in `blocking.jsonl`. AICCM/CCM/SECCM
are not part of this profile.

Final pre-tag shipped-feature confidence:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane full-fast --out fast.jsonl
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane slow-nightly --out slow.jsonl
cat fast.jsonl slow.jsonl > all.jsonl
$PY scripts/test_gate/gate_verdict.py all.jsonl \
    scripts/test_gate/known_reds_baseline.json --reverify --wt "$WT" --py "$PY"
```

## Regression Suite Under `examples/regression`

The `examples/regression` suite is a calculation-parity suite with full
artefacts, external-code subprocess runners, and paper/SI candidate outputs.
It is not the default pytest lane for every code change.

Use it when the work changes:

- cross-code parity infrastructure,
- paper or SI numbers,
- external reference-code runners,
- output artefact retention for calculations,
- public scientific claims that need full input/output evidence.

Routine unit and integration coverage should go through the pytest lanes first.
Run `examples/regression` only for the subset that matches the touched method,
system family, or publication claim.

## Maintenance

When adding a new expensive test, do one of these in the same change:

- keep it cheap enough for the relevant area lane,
- mark it `@pytest.mark.slow` and make sure it belongs in `slow-nightly`,
- add or update a lane glob or explicit `nodes` entry in
  `scripts/test_gate/lane_manifest.json`,
- document why the test is experimental or external-reference only.

When a lane's file list becomes noisy, update the manifest rather than telling
every chat to remember a private command. The manifest is the contract.
