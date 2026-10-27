# What validation actually runs

The default development policy favors quick feedback. A routine MR is a bounded
check, not a claim that every feature works. Run a small reproduction for the code
you changed. Broad numerical validation uses a qualified development build and
appropriate compute resources; it need not block every implementation merge.

## Inventory and counting

The figures below are a historical snapshot, not a current suite count. Use
the manifest and lane inventory for current file membership.

Snapshot at source `0819b1771c84` (2026-09-26): **933 test files and 15,028
`test_*` function/method definitions** under `tests/`. Definitions are not collected
pytest cases: parametrization expands them, and dependencies can skip modules.
A full collected count requires a matching native runtime. No native build was
started just to count tests. `vibe-basis/tests/`, hook tests and website tests are
separate from these totals.

The exhaustive `scripts/test_gate/suite_manifest.json` classifies those files:

| Tier | Files | Purpose | Default CI execution |
|---|---:|---|---|
| T1 | 12 | Blocking sentinels and policy/contract checks | Candidate or explicit full validation |
| T2 | 847 | Broad regression, numerical and reference inventory | Selected manually; not the default MR gate |
| T3 | 75 | Experimental/research inventory | Explicit research runs |
| T0 | 0 pytest files | Compile/import and runtime provenance | Opt-in native MRs and full validation |

In the measured release baseline, T1 produced **294 passed and two skipped**:
one skip was a whole module, not an executed passing case. The sum of its file
execution times was 128.3 seconds; files run concurrently, so this is not wall
time. The combined native build/test job took 56.7 minutes. Compilation, setup
and producer/consumer work are additional to T1.

## Types of validation

| Type | Concrete examples | Does it run chemistry? |
|---|---|---|
| Static and repository contracts | Privacy, syntax compilation, source inventory, changelog pins, basis coverage | No SCF calculation |
| Unit/algebra tests | Tensor operations, solver residuals, conventions, option validation; mocks around native calls | Usually no complete calculation |
| Small numerical regression | H2/H/STO-3G RHF/UHF/RKS/UKS and MP2 fixed points in `test_release_sentinels.py`; gradient finite differences; Gamma Molden export | Yes, real small SCF/correlation/gradient calculations |
| Broad numerical/reference tests | Molecular correlation, periodic GDF/GAPW/BIPOLE, basis/ECP, cross-code parity | Often multiple real calculations; may need large memory or an external reference |
| Producer/consumer integration | QC-generated output consumed by the pinned viewer | Some tests require the native QC runtime |
| Native build | C++/pybind11 build plus dependency setup and import provenance | Builds the implementation; compilation alone proves no scientific result |
| Other build checks | Sphinx docs; Astro website tests/build/output verification; vibe-basis tests and install lifecycle | No production calculation campaign |

T1 itself is mixed. Basis/source/changelog files are contracts; derivative guard
files contain both mocked/refusal tests and numerical checks. A single function
can run several methods, so test count is not calculation count. Large periodic,
CC, scaling and reference acceptance remains a separate measured activity with
inputs, runtime identity, tolerances and a reference comparison. Green CI alone
is not that verdict. Experimental T3 work does not become a routine release gate.

## Execution policy

The current [repository policy](https://github.com/vibe-qc/vibe-qc/blob/main/AGENTS.md)
separates local development checks from candidate qualification:

| Work | Required evidence |
|---|---|
| Documentation prose | Focused text, local-link and anchor check |
| Non-scientific code | One focused functional check |
| Scientific implementation | Focused source/functionality check before landing; complete calculation and reference comparison for scientific acceptance |
| Native changes | Rebuild the affected native code for checks that execute it; verify loaded runtime identity |
| Ordinary development build, push or merge | Local checks; skip CI with `[skip ci]` and push option `ci.skip` where supported |
| Daily release candidate, when new commits exist | Full candidate CI under the authorized release owner |

Do not request manual, API or scheduled CI for development refs. A pipeline
configuration's available jobs are not an instruction to run them on each
change. Candidate qualification includes native build/runtime checks, T1,
pinned viewer contracts, vibe-basis, schemas and documentation; consult the
[release process](release_process.md) for the current candidate workflow.

For scientific acceptance, submit a complete calculation through `vq` and
compare against ORCA or CRYSTAL when available. Record matched inputs,
source/native runtime identity, measured values, units and tolerances.
Justify alternative numerical evidence if the external reference is
unavailable. A runner may validate its own implementation and reuse qualifying
prior evidence. Pending calculations neither count as PASS nor block unrelated
ready work or the daily candidate build.

## Commands

```sh
# Cheap policy checks, no native extension required.
python3 -m unittest discover -s scripts/ci/tests -p 'test_*.py'
python3 scripts/test_gate/run_full_suite.py --wt . --tier T1 --dry-run

# Affected small tests on an already qualified runtime.
.venv/bin/python -m pytest tests/test_<area>.py -q

# Full blocking release test tier, not the entire research inventory.
.venv/bin/python scripts/test_gate/run_full_suite.py --wt . \
  --py .venv/bin/python --tier T1 --out <external-evidence>/blocking.jsonl

# Inventory expansion, only on a runtime matching this source tree.
.venv/bin/python -m pytest tests --collect-only -q
```

See [developer test lanes](developer_test_lanes.md) for targeted lane commands.
Avoid launching all T2/T3 calculations or a new native build just for routine
feedback or counting. The oldest performance entries in the manifest are
historical measurements, not current runtime promises.

## Avoid repeated companion tests

The QC candidate keeps the complete T1 numerical tier. Its viewer invocation
selects real producer/native integration: pending and settled containers,
attachment downloads, writer/reader kind agreement, live QC optimization,
wavefunction parity and gradients, relocalization, manifest metadata and UHF
spin density. Viewer-only plot rendering, mock live services and synthetic
in-memory archives run in the viewer's own suite, rather than a second time
inside the expensive QC build job. The complete viewer test files remain
available for focused debugging.

## Candidate compilation reuse

Candidates configure and build in a stable ABI-scoped CMake directory and use
ccache, keyed by the compiler, options and included source/header contents.
Dependency recipes retain their separate authenticated cache. A previous
same-profile cache may seed it; library stamps and source receipts still
force every changed dependency to rebuild before use. The pip cache
is shared across candidate names; pip still resolves declared requirements.
Compiler caches are saved even when a later test fails, so fixing a Python
regression does not discard successfully compiled translation units.

Set `VIBEQC_CLEAN_BUILD=1` explicitly for an uncached diagnostic. This uses a
separate per-pipeline directory with ccache disabled, leaving normal cached
objects intact. Both modes retain source/core identity, freshness, dependency
hashes, T1 and producer/consumer checks. Memory-aware build concurrency is
unchanged. `test-results/native-build-seconds.txt` and the before/after ccache
statistics distinguish installation/build time and cache reuse from total
job time. An unchanged source can require no compiler calls when CMake's
existing objects are reusable; zero ccache calls alone is not a cache failure.
