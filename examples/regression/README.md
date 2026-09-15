# vibe-qc regression / parity test suite

Continuous parity-checking against PySCF (and ORCA, for molecules) on a
curated set of physics calculations. **Not** unit tests — this is the
parallel infrastructure that verifies vibe-qc's *scientific*
correctness, not its API contracts.

See [`DESIGN.md`](DESIGN.md) for the original design rationale and
[`REFERENCES.md`](REFERENCES.md) for test-set provenance. The historical
handover file has been retired; this README is the current operator guide.

For ordinary development, start with the partitioned pytest lanes in
[`scripts/test_gate/lane_manifest.json`](../../scripts/test_gate/lane_manifest.json)
and [`docs/developer_test_lanes.md`](../../docs/developer_test_lanes.md).
This calculation-parity suite is for full input/output evidence, paper/SI
numbers, external-code runner changes, and method/system claims that need
cross-code artefacts. It is not the default cost paid for unrelated molecular,
PBC, AICCM, GAPW, GDF, or output-only changes.

## Run

```sh
# Cheap end-to-end smoke case:
.venv/bin/python -m examples.regression.run_suite --smoke

# Everything that's currently enabled (molecules + Ne FCC + ionic rocksalts):
.venv/bin/python -m examples.regression.run_suite

# A subset, by id:
.venv/bin/python -m examples.regression.run_suite \
    --systems h2,h2o,ne_atom \
    --bases sto-3g \
    --methods rhf,rks-lda

# Put artifacts somewhere explicit:
.venv/bin/python -m examples.regression.run_suite \
    --output-root ~/vibeqc-runs \
    --systems h2 \
    --bases sto-3g \
    --methods rhf

# Paper/SI candidate run: keep a validated QVF beside each vibe-qc row:
.venv/bin/python -m examples.regression.run_suite \
    --output-qvf \
    --systems h2,h2o,benzene \
    --bases sto-3g \
    --methods rhf,rks-b3lyp

# List the current active case set without creating a run:
.venv/bin/python -m examples.regression.run_suite --list-cases

# Create metadata only; useful before launching a long queue job:
.venv/bin/python -m examples.regression.run_suite --dry-run
```

Run directories default to `~/vibeqc-runs/<run_id>`. Override that with
`--output-root PATH` or `VIBEQC_RUNS_DIR=PATH`; the suite no longer
defaults into the checkout. Precedence is explicit CLI `--output-root`,
then `VIBEQC_RUNS_DIR`, then `~/vibeqc-runs`.

`--output-qvf` writes `cases/<case>/vibeqc/vibeqc.qvf` for each
successful vibe-qc calculation. This is intentionally opt-in because
QVF archives can become large for dense periodic cases. Use it for
paper, SI, and viewer-facing reruns where the calculation must carry
full input, raw text output, the `.system` manifest, parsed JSON, and a
validated visualization archive in one case directory. For molecular
MP2/UMP2 rows, the QVF contains the converged SCF reference
wavefunction used by the correlated step; the final MP2 parity energy is
still the value in `results.csv`, `summary.md`, and `parsed.json`.

Designed to **run autonomously** end-to-end on a remote workstation
(`compute-reference` etc.) - no interactive babysitting. Each case is independent;
one case erroring out doesn't abort the suite. **Per-case subprocess
isolation** (v0.7.2+) means even a C-level crash in one runner (PySCF
segfault, ORCA non-zero exit, libxc abort) only kills its own
subprocess; the dispatcher captures it as an `error` row and continues.

## Running long jobs over ssh

The natural "run a multi-hour suite from my laptop, drop the ssh, come
back later" pattern *does not work* with plain `nohup` + `&` + `disown`
on systemd-managed Linux boxes (Manjaro, recent Arch, RHEL 9+, modern
Ubuntu). Even with `setsid` and `nohup`, the python process inherits
its ssh session's cgroup (`/user.slice/user-N.slice/session-N.scope`),
which systemd cleans up — killing every process in it — when the ssh
session ends. The suite then dies silently a couple of minutes after
ssh disconnect, well before reaching its first long periodic SCF.

**The fix is two steps**:

```sh
# 1. ONE-TIME: enable lingering for your user, so the user manager
#    (`user@<UID>.service`) keeps running across login sessions. Without
#    this, transient services from step 2 still die at session end.
loginctl enable-linger "$USER"

# 2. EVERY-RUN: launch the suite as a transient service under the
#    user manager. `systemd-run --user --unit=NAME` places it in
#    `/user.slice/user-N.slice/user@N.service/app.slice/NAME.service`,
#    which is independent of any login session.
export XDG_RUNTIME_DIR=/run/user/$(id -u)            # required for systemd --user over non-interactive ssh
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus

systemctl --user reset-failed vibeqc-suite.service 2>/dev/null
systemd-run --user --unit=vibeqc-suite \
    --working-directory=/path/to/vibe-qc \
    --setenv=ORCA_PATH=/path/to/orca_install \
    --property=StandardOutput=file:/tmp/regression-stdout.log \
    --property=StandardError=file:/tmp/regression-stdout.log \
    -- /path/to/vibe-qc/.venv/bin/python -u -m examples.regression.run_suite \
        --output-root ~/vibeqc-runs \
        --systems all
```

Notes:
* `ORCA_PATH` (or `ASE_ORCA_COMMAND` / `ORCA_COMMAND` / a directory on
  `$PATH`) is required for ORCA molecular cases. Non-interactive ssh
  shells skip `~/.zshrc` so the interactive `$PATH` is gone — pass the
  ORCA install path explicitly via `--setenv`.
* `python -u` makes stdout line-buffered, so progress lines appear in
  the file as they're emitted instead of accumulating in a 4 KB buffer
  that would be lost on crash.

**While the suite runs**:

```sh
# is the unit alive?
systemctl --user is-active vibeqc-suite.service

# detailed status (memory, CPU, runtime)
systemctl --user status vibeqc-suite

# tail the dispatcher's progress lines
tail -f /tmp/regression-stdout.log

# tail a specific case's verbose log
RUN=$(ls -dt ~/vibeqc-runs/2026* | head -1)
tail -f $RUN/verbose/lih_rocksalt__sto-3g__rks-lda__1x1x1.log

# stop early (will run __exit__ handlers cleanly via SIGTERM)
systemctl --user stop vibeqc-suite
```

The unit auto-cleans itself once the suite exits (transient unit; no
file leftover in `/etc/systemd/`). The run dir, `summary.md`, `results.csv`,
and `verbose/` artefacts persist under `~/vibeqc-runs/` unless
`--output-root` points elsewhere.

Standalone diagnostics under this tree follow the same artifact-root
rule when they create run products. For example, the ORCA parity matrix
and speed benchmark accept `--output-root` / `--run-id`; their reports,
vq workspaces, fetched raw outputs, and regenerated cache copies default
under `~/vibeqc-runs/<run_id>/`. Updating committed fixture caches is an
explicit opt-in on those tools.

## Outputs (per `run_id`)

```
~/vibeqc-runs/<run_id>/
├── manifest.json        # run metadata, planned cases, completed rows
├── results.csv          # one row per (case, code, target) — the dashboard fixture
├── summary.md           # markdown report — paste into the dev chat
├── env.json             # git SHA, vibe-qc / pyscf / ase / orca versions, machine
├── verbose/
│   └── <case>.log       # per-case persistent log: active settings, EIGS preflight,
│                        # truncation-optimizer report, SCF iteration trace, energy
│                        # decomposition, plus pyscf + (for molecules) orca sections.
│                        # Treat as a *document* a domain expert reads to verify
│                        # correctness — not a debug dump.
└── cases/
    └── <case>/
        ├── vibeqc/      # input.json, vibeqc.out, vibeqc.system, parsed JSON,
        │                # vibeqc.qvf when --output-qvf was requested, plus
        │                # periodic production sidecars as vibeqc_job.*
        └── reference/
            ├── pyscf/   # input.json, runner.py, stdout/stderr, parsed JSON
            ├── orca/    # ORCA .inp/.out/.err plus stdout/stderr, parsed JSON
            ├── crystal/ # optional: CRYSTAL deck, fetched raw .out, logs, parsed JSON
            └── cp2k/    # optional: CP2K input, raw .out, logs, parsed JSON
```

The `summary.md` is the **bug-hunt + perf-optimization hand-back**: paste
it (or pieces of it) into the development chat. The `## Action items`
section at the bottom is auto-generated from rows that exceeded
tolerance, marginal Δ, eigs_preflight severity ≥ error, or > 5×
vibeqc/pyscf wall ratio.

The `verbose/` and `cases/` directories are kept for manual inspection.
Rows marked `unavailable` are intentional skips or host-capability gaps,
not numeric failures.

`results.csv` carries both raw and size-normalized accuracy fields. The
raw `delta_ha_vs_ref` / `delta_mha_vs_ref` columns catch whole-system
failures. The normalized `delta_mha_per_atom_vs_ref`,
`delta_mha_per_electron_vs_ref`, and
`delta_mha_per_basis_function_vs_ref` columns make large molecules and
large-basis cases comparable to small smoke cases. `summary.md` renders
the same information in the per-case table plus a dedicated
`Accuracy ranking` section, sorted by absolute mHa error.

## Layout

```
examples/regression/
├── DESIGN.md                             # historical design + wave plan
├── README.md                             # this file
├── REFERENCES.md                         # test-set and basis provenance
├── run_suite.py                          # CLI entry — autonomous runner
├── core/
│   ├── output_paths.py                   # shared ~/vibeqc-runs resolution
│   ├── spec.py                           # PeriodicSpec / MoleculeSpec / MethodSpec / ExpectedRef
│   ├── case.py                           # CodeRow / CSV io / env snapshot
│   ├── compare.py                        # ΔE classification (pass / marginal / fail)
│   ├── runner_vibeqc.py                  # vibe-qc periodic + molecular SCF + verbose log
│   ├── runner_pyscf.py                   # pyscf.pbc + pyscf.gto reference at matching settings
│   ├── runner_orca.py                    # ORCA molecular reference via ASE
│   ├── runner_crystal.py                 # optional CRYSTAL periodic reference
│   ├── runner_cp2k.py                    # optional CP2K periodic reference
│   └── report.py                         # markdown summary generator
├── systems/
│   ├── periodic/                         # one PeriodicSpec module per system
│   │   ├── ne_fcc.py                     # active easy reference
│   │   ├── lih_rocksalt.py               # active ionic rocksalt
│   │   ├── nacl_rocksalt.py              # active ionic rocksalt
│   │   └── mgo_rocksalt.py               # active ionic rocksalt
│   └── molecules/                        # one MoleculeSpec module per system,
│       ├── h2.py                         # smoke-test anchor
│       ├── h2o.py                        # method-coverage workhorse
│       └── s22_*.py                      # S22 specs; 10 active, 12 data-only
├── methods/
│   └── catalog.py                        # MethodSpec dict
├── expected/                             # one .json per (system, basis, method)
│   └── <system_id>__<basis>__<method_id>.json
├── parity_matrix_orca/                   # standalone ORCA diagnostics/fixtures
├── crystal_parity/ bipole_parity/ gapw_parity/
│                                           # standalone bug-hunt diagnostics
└── output/                               # legacy empty stub; not a default output target
```

## Adding a new case

1. Drop a `systems/periodic/<id>.py` (or `systems/molecules/<id>.py`)
   exporting `SPEC: PeriodicSpec` (or `MoleculeSpec`).
2. (Optional) `expected/<system_id>__<basis>__<method_id>.json` to
   override the default 5 mHa-tolerance / pyscf-at-runtime reference.
3. Add the `(system_id, basis, method_id)` triple to either
   `WAVE1_MOLECULE_CASES` or `WAVE1_PERIODIC_CASES` in `run_suite.py`.

## Reference selection

* **Molecules**: ORCA is the primary reference when available
  (`vibeqc.benchmark.find_orca_command` resolves the binary). PySCF is
  the fallback. vibe-qc is compared against the primary.
* **Periodic**: PySCF.pbc is the default reference. `--include-crystal`
  adds CRYSTAL via `vq submit`; `--include-cp2k` adds a local CP2K
  subprocess when the installed basis/method combination is supported.
  Missing programs and unsupported combinations are reported as
  `unavailable`, not as numeric failures. At matching basis + functional
  + grid + cutoffs, closed-shell wide-gap systems should be µHa-scale.
  Dense ionic / minimal-basis rows are active bug-hunt rows: judge them
  in both raw mHa/cell and mHa/atom, and do not treat a raw mHa number
  as meaningful without the size-normalized context.

The "primary" reference is recomputed every run, so no version-skew
between hand-curated reference values and current pyscf releases. Per
spec, that's the trade-off: shorter feedback loop, slightly noisier
agreement.

## Convergence

Periodic SCF for ionic / deep-core systems (Na, Mg, Cl, Li, transition
metals) requires SAD initial guess + damping ≥ 0.7 to dodge the
v0.7 SCF driver's HCORE-divergence diagnostic. These are set in each
`PeriodicSpec.default_initial_guess` / `default_damping`. If a system
still doesn't converge, the verbose log records the SCF driver's
diagnostic message verbatim — read it, then update the spec or add a
retry-ladder rung in `runner_vibeqc.py`.

## Status

`--list-cases` currently reports 44 active cases: 40 molecular and 4
periodic.

* **Molecules**: H2, Ne, HF, CH4, NH3, H2O, H2CO, benzene, O2, O3, and
  10 representative S22 dimers. Coverage includes RHF, UHF, RKS/UKS
  LDA/PBE/BLYP/B3LYP, MP2/UMP2, and selected density-fitting lanes.
  Three-way comparison is vibe-qc, PySCF, and ORCA when ORCA is
  available.
* **Periodic**: Ne FCC plus LiH / NaCl / MgO rocksalt, all `sto-3g` /
  `rks-lda`. Default comparison is vibe-qc + PySCF.pbc, with optional
  CRYSTAL and CP2K rows behind CLI flags. PySCF.pbc cell setup can be
  slow on first invocation; budget accordingly on remote machines.
* **Deferred**: pob-dzvp-rev2 cross-code basis mapping, larger corundum
  and X23-style cells, interaction-energy assembly for S22/S66-style
  datasets, and broader multi-k / hybrid periodic coverage.
* **Deferred to wave 3**: dual-target invocation
  (`--target dev|release|both`).
