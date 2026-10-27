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

# Prepare the existing periodic cases for the public BIPOLE route:
.venv/bin/python -m examples.regression.run_suite \
    --systems ne_fcc --periodic-jk-method bipole --dry-run
```

`--periodic-jk-method gdf|bipole` selects the vibe-qc periodic route;
the default remains `gdf`. The selection crosses subprocess isolation and
is recorded in each planned periodic case in `manifest.json`, the vibe-qc
`input.json` and the verbose log. For BIPOLE, the declared direct and nuclear
cutoffs and Ewald splitting parameter are passed to `run_periodic_job` as
`bipole_cutoff_bohr`, `bipole_nuclear_cutoff_bohr` and `ewald_omega`. The
case input records those route controls. Preflight options are labelled
separately from the public driver's resolved calculation settings.

The periodic runner dispatches RHF/UHF/RKS/UKS requests through the public
driver. This does not expand the curated case list or establish a magnetic
state, basis, functional or reference match. Those require explicit case
definitions and numerical evidence. Reference runners keep their own method
and applicability checks. GDF-only `--rsgdf-tail-ke-cutoff` is rejected with
the BIPOLE selector; GDF-specific static parity holds apply only to GDF.

### Explicit periodic matrices

Use a versioned case file to run the same system on multiple meshes and
select the primary reference for each comparison:

```sh
.venv/bin/python -m examples.regression.run_suite \
    --periodic-matrix examples/regression/matrices/bipole_restricted_starter.json \
    --dry-run
```

The starter contains LiH RHF, PBE and B3LYP restricted controls at Gamma and
2x2x2. Its one-microhartree-per-cell threshold is a requested comparison
target, not established accuracy. It is an incomplete starting set for
convergence and reference qualification. It does not qualify unrestricted
magnetic states (see the charge and spin requests below), larger bases, ECPs,
physical slabs/wires or real-checkpoint
SKALA. Each starter row requires a CRYSTAL23 program banner. The configured
wrapper selects the executable; this requirement does not change it. A
missing, conflicting or different banner rejects the comparison even when
the energy converged. Matched physical inputs must still be established
before claiming parity. The unrestricted reference refusal remains in force.

Schema version 1 has exactly three top-level fields: `schema_version: 1`,
`jk_method` (`gdf` or `bipole`) and a nonempty `cases` list. Each case requires
`system_id`, a registry `basis` name, catalog `method_id`, three positive
integer `kmesh` counts, finite nonnegative `tolerance_ha`, a nonempty
`tolerance_rationale` and `reference_code` (`pyscf`, `crystal` or `cp2k`).
Optional `reference_version` requires an exact version token such as
`CRYSTAL23` and is currently supported only with `reference_code: crystal`.
Omitting it preserves legacy version-unconstrained comparisons. Optional
`notes` describe the case's purpose and limitations. The selected
system module supplies the geometry and remaining defaults. Unknown fields,
duplicate case identities and malformed requests are rejected before any
calculation. Shift and checkpoint overrides are not implemented by this
schema and cannot be silently supplied as extra fields.

#### Lattice cutoffs

A row may declare `lattice_cutoff_bohr` and `nuclear_cutoff_bohr`, which
replace the system module's `default_cutoff_bohr` and
`default_nuclear_cutoff_bohr`. These are the BIPOLE direct and nuclear
cutoffs, and the lattice-sum cutoffs on the GDF route. The nuclear cutoff may
not be smaller than the lattice cutoff. Declared cutoffs are part of the
case identity (`..._1x1x1__cut28_ncut28`), so a cutoff ladder can hold the
same case at several cutoffs. They cross process isolation explicitly and
appear in the manifest and the case's `input.json`.

BIPOLE refuses to enter the SCF when the S(k) lattice-fold drift exceeds
1e-2, and calls it converged only below 1e-4. A test checks every row of
every shipped BIPOLE matrix against 1e-4 at its declared cutoff. The LiH
starter declares 28 bohr: the STO-3G Li 2sp tail gives a drift of 2.5e-1 at
the system default of 12 bohr, 5.3e-4 at 24 and 1.1e-5 at 28. One Gamma RHF
at 28 bohr took 47 minutes locally.

#### Charge and spin requests

A row may override the system's electronic state with `charge`,
`multiplicity` and `atomic_spins`; the same fields exist on `PeriodicSpec`.
Every count is per primitive cell and identical on every cell of the
Born-von Karman supercell, so a mesh of N_k points carries
N_k * (multiplicity - 1) unpaired electrons in total. That is vibe-qc's
periodic `multiplicity`. PySCF's k-point classes count electrons over the
whole supercell, so the PySCF runner sets `nelec` to
(N_k * n_alpha, N_k * n_beta); `cell.spin` alone fills a different, mixed
state. `atomic_spins` are initial -1/0/+1 tags for an unrestricted SAD guess,
separate from the total spin count. The state is part of a case's identity
(`..._uhf__2x1x1__mult3`), crosses process isolation explicitly, and each
row records the resolved per-cell `n_alpha` and `n_beta`. A comparison fails
when the two codes report different counts, or when only one side confirms
a polarized state.

These requests are refused before any calculation:

- a nonzero `charge`, because no neutralizing-background convention is
  declared for matched comparisons;
- a restricted method (`rhf`, `rks-*`) with a multiplicity above 1, atomic
  spins or `is_open_shell`;
- `is_open_shell` without a multiplicity or atomic spins that describe the
  state;
- an electron count whose parity cannot carry the multiplicity;
- nonzero float `magnetic_moments`, which no runner consumes;
- `atomic_spins` without the SAD guess, or for the PySCF reference, which has
  no equivalent initial state here.

A secondary PySCF row that cannot represent the state reports `unavailable`.
The CRYSTAL runner still refuses unrestricted methods.
`matrices/open_shell_starter.json` pairs RHF, the UHF singlet limit and the
per-cell UHF triplet of `h2_box12` at Gamma and 2x1x1. It is a
state-transport control, not a qualified magnetic material.

#### Physical-input qualification

A shared basis or functional name does not prove that two codes solved the
same Hamiltonian. Each periodic case writes `input_qualification.json`, built
from what each code actually built or was submitted
(`core/input_identity.py`). It compares these fields:

- **geometry**: the lattice metric (Gram matrix), handedness, and atomic
  numbers with fractional positions. These do not depend on the Cartesian
  orientation each code chooses.
- **basis**: per atom, each contracted shell's angular momentum, pure/Cartesian
  form, exponents and contraction. The contraction is normalized over
  normalized primitives, so coefficient conventions cancel.
- **ecp**: per-atom core counts and resolved scalar Gaussian potential terms.
  Local and angular-projector channels, actual radial powers, exponents and
  coefficients must match (1e-12 absolute plus relative tolerance for the
  floating terms). Identical primitives are combined without normalization.
  Zero-core potentials remain operators. Older count-only records, spin-orbit
  operators and unparsed CRYSTAL pseudopotentials stay `unverified`.
- **xc**: the exact-exchange fraction, omega, and the energy density with
  its rho and sigma derivatives. Each code evaluates its own resolved
  functional on the same fixed (rho, sigma) sample. B3LYP with VWN5 and with
  VWN-RPA therefore differ whatever they are called.

Each code is compared against the primary reference, and each row's
`input_qualification` column records the result:

- `qualified`: every compared field matched.
- `unverified`: a field could not be established, with the reason.
- `mismatch`: the inputs differ. The row becomes an `error` even when the
  energies agree, because equal energies from different Hamiltonians are not
  parity.

The record lists the conventions it does not compare: grids, XC domains,
Coulomb/exchange and zero-mode treatment, fitting basis, screening and
lattice-sum cutoffs, convergence and smearing, and spin-orbit ECP operators. Paired
timing ratios are withheld unless the inputs are qualified.

The scalar ECP identity follows Shaw and Hill, JCP 147, 074108 (2017),
Sec. I Eqs. (1)-(3), DOI:10.1063/1.4986887: the effective charge, local
potential and angular-projector terms are separate. The stored radial power
is the power in the potential; both integral libraries store it with an
additional two from radial integration. The vibe-qc record uses the same
ECP resolver as the public periodic driver; PySCF uses its built integral
arrays. These records qualify resolved inputs, not the subsequent integral
execution or SCF accuracy. Complete numerical acceptance remains separate.

vibe-qc and PySCF records come from the built objects. CRYSTAL's record
starts from the submitted `INPUT.d12`. Its `a b c alpha beta gamma` and
fractional coordinates fix a right-handed cell, so a left-handed spec is a
mismatch. HF is verified by the absence of a DFT block.

When exactly one fetched `INPUT.out` exists, what the run reports it used
replaces the deck's request for the remaining fields:

- **geometry**: the printed lattice vectors and Cartesian coordinates, to
  12 digits.
- **basis**: `LOCAL ATOMIC FUNCTIONS BASIS SET`, compared per element.
  CRYSTAL prints only four significant digits, so the comparison tolerates
  6e-4 relative on exponents and 2e-3 on canonical coefficients, and the
  field records that precision. A different contraction scheme or scale
  factor fails; a change below the printed digits cannot be detected.
- **ecp**: no pseudopotential section means an all-electron run.

Without an output, the internal-library `BASISSET` and its ECP stay
`unverified`.

A DFT keyword is resolved from the manual, because the outputs do not print
the functional's components. The standalone exchange-correlation keywords of
a CRYSTAL DFT block are `SVWN`, `BLYP`, `PBEXC` and `B3LYP`, among others
(CRYSTAL23 manual pp. 93 and 134-135; CRYSTAL14 p. 84). Each is mixed from
its documented libxc parts and evaluated on the shared sample:

- `SVWN` is Slater + VWN5.
- `B3LYP` is `EXCHANGE BECKE`, `CORRELAT LYP`, `HYBRID 20`,
  `NONLOCAL 0.9 0.81` on VWN5. That is 0.08 Slater + 0.72 B88 exchange,
  0.19 VWN5 + 0.81 LYP correlation, and 20% HF.

The field records that source. Any other keyword stays `unverified`. A bare
`PBE` is only an `EXCHANGE`/`CORRELAT` sub-keyword, so the runner now writes
`PBEXC` for `rks-pbe`. The sealed `crystal_demos/mgo_pbe_sto3g.d12` keeps
the bare `PBE` its reference energy was computed with. That run agreed with
vibe-qc's full PBE to a few mHa, so CRYSTAL23 evidently read it as full PBE.

Lattice rows in `PeriodicSpec` are the lattice vectors, and r = frac · L.
Both runners now build that crystal. vibe-qc uses the orientation-free
`lattice_vectors=` form, and PySCF gets positions computed from rows.
Previously, a non-symmetric row matrix gave each code a different, sheared
cell (#445). Every shipped system has a symmetric matrix, so earlier results
are unaffected. The PySCF runner also attaches the ECP that PySCF ships for
a basis, as vibe-qc does, so a valence-only basis is not run all-electron.

`--periodic-matrix` replaces the curated case list and cannot be combined
with case filters or `--smoke`. A conflicting route selector is rejected.
List and dry-run modes need no reference executable; execution requires
`--include-crystal` or `--include-cp2k` when a row selects that reference.
An unavailable selected reference remains an error/unavailable comparison;
the harness does not substitute another code. All included reference rows
are compared against the selected primary reference using that row's
explicit tolerance, without inheriting a legacy Gamma expected file.

CRYSTAL attempts write `reference.json` beside the submitted deck, retaining
the requested/reported version, wrapper, job identity and terminal state,
resource limits, and SHA256 hashes of the submitted input and fetched output.
This receipt connects the local artifacts and job; it does not inspect the
remote executable bytes or attest that the remote program read identical
input bytes. Keep these operational artifacts in the configured external
output directory. Program banners are parsed independently of case titles.

The planned-case manifest records the mesh, route, reference/version, tolerance,
rationale and notes. Meshes propagate through isolated and direct reference
runners, and each mesh has a distinct output directory, including failed
cases. Existing CSV/JSON results and the summary renderer are reused.
Energy agreement alone does not prove matching basis primitives, physical
domains, spin states, k-point coordinates, runtime identity or timing scope.

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
│   ├── spec.py                           # re-exports PeriodicSpec / MoleculeSpec / MethodSpec / ExpectedRef from vibeqc_spec
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
├── parity_periodic_dlpno_vs_pyscf.py     # DLPNO-CCSD(T) on crystals vs PySCF KRCCSD(T)
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

### Physical identity is narrower than matching names

Input qualification requires resolved `domain`, `integration` and
`electrostatics` definitions in addition to geometry, orbital basis, scalar
ECP and XC. Missing or merely requested conventions are `unverified`, even
when the older four fields match. Existing producers without those observations
therefore retain diagnostic energies but cannot supply qualified paired timing.

Each convention record names its `evidence` (`resolved`), nonempty `source`,
and `definition`. Domain records identify dimensionality, periodic axes,
per-cell/per-molecule normalization and neutralizing-background convention.
Integration records bind exact point and weight SHA-256 identities, coordinate
units, domain and partition definition; equal grid nicknames are insufficient.
Electrostatic records name the Coulomb and exchange kernels, each zero-mode
prescription and the boundary condition. Different definitions are mismatches;
matching explicit records only qualify those named fields. The result always
retains `not_compared` and `full_physical_equivalence=false`: fitting bases,
screening/cutoffs, convergence/occupations and spin-orbit operators still require
separate evidence. Different converged grids may be numerically comparable,
but that needs a declared convergence study rather than relabeling their
identities as equal.

Explicit CRYSTAL scalar `INPUT` ECPs are read according to CRYSTAL23 manual
pp. 84–85, Eqs. (3.16)–(3.19): effective charge `ZNUC`, local `M` terms,
projected `M0` through `M4` terms, then exponent, coefficient and **actual**
radial power. No integral-library `n-2` conversion is applied to CRYSTAL's
`NKL`. Supported explicit general-basis decks retain the normalized operator
request, separately labelled `evidence=submitted`; matching requests alone
cannot establish execution identity. Library pseudopotentials, spin-orbit
`INPSOC`, ambiguous atomic numbering and malformed terms remain unverified or
refused. Output printout grammars are not guessed from a keyword or core count.
The output/parser and actual matched numerical acceptance remain distinct.

For declared-matrix CRYSTAL23 performance assessment, use the
[paired performance protocol](../../docs/developer_pbc_performance.md).
It supplements the existing accuracy-filtered timing table with repeated
full-calculation receipts, resource matching, uncertainty and missing-case gates.
