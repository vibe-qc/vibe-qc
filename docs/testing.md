# What validation actually runs

The default development policy favors quick feedback. A routine MR is a bounded
check, not a claim that every feature works. Run a small reproduction for the code
you changed. Broad numerical validation uses a qualified development build and
appropriate compute resources; it need not block every implementation merge.

## Inventory and counting

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
inputs, runtime identity, tolerances and an independent verdict. Green CI alone
is not that verdict. Experimental T3 work does not become a routine release gate.

## Execution policy

| Trigger | What runs |
|---|---|
| Python-only, test-only or CI-YAML-only MR | Privacy, policy tests, Python syntax and T1 selection checks; relevant companion/tooling jobs; no native rebuild |
| C++/native build input MR | Cheap gates by default; set `VIBEQC_BUILD_NATIVE=1` to add incremental compile/import |
| Main push | Privacy, and website verification when applicable |
| Release candidate | Clean native build, T1, pinned viewer contracts, vibe-basis and schema checks, docs |
| Manual web pipeline with `VIBEQC_FULL_VALIDATION=1` | Explicit clean native/T1/viewer validation plus companion gates |

This intentionally defers native compiler errors, Python runtime errors and
Python/native mismatches to focused local work, a development build or release
validation. There is no default C++ compilation gate on an MR. Opt in with
`VIBEQC_BUILD_NATIVE=1` for an MR touching native inputs, or use full validation.

Routine docs edits no longer build the entire Sphinx site. MR docs builds are
limited to docs configuration/theme/build scripts and CI configuration; candidates,
release and web pipelines still build the site. A prose cross-reference can
therefore fail later. QVF schema conformance is path-filtered on MRs, and CI-YAML
edits no longer exercise the unrelated vibe-basis installation lifecycle.
CI edits no longer rebuild the native core merely because the YAML changed.
Superseded validation jobs are interruptible; candidate auto-cancellation remains
disabled. Builds that are still needed remain serialized by the native resource
group.

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
