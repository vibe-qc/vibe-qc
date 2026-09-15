# vibe-basis — basis-set optimization driver

`vibe-basis` is a standalone Python package that drives **external
quantum-chemistry programs** under a basis-set optimization loop.
It is the modern successor to the home-grown CRYSTAL09 + MINUIT2
+ python pipeline that produced the `pob-TZVP` (Peintinger et al.
2013) and `pob-TZVP-rev2` / `pob-DZVP-rev2` (Vilela Oliveira et
al. 2019) basis sets.

## Scope

* **Inputs:** a basis-set parametrization, a test-set of compounds,
  an objective function (typically Σ wᵢ · E_SCF), bounds + LD
  rules, an optimizer choice.
* **Engines:** every engine implements one protocol,
  `vibe_basis.engine.EnergyEngine`, so which program computes a
  number is configuration rather than structure. **vibe-qc is the
  primary engine**; **CRYSTAL23** (Gaussian) and **GPAW** (plane
  wave) are fallbacks and independent cross-checks. See
  [`ROADMAP.md`](ROADMAP.md) § 2 for the policy and
  `vibe_basis/engine.py` for why the primary engine lives on the
  vibe-qc side. Future backends: ORCA, Gaussian, NWChem, Psi4.
* **Optimizers:**
  * **NLopt BOBYQA / COBYLA** — derivative-free Powell descendants
    that match what MINUIT2 actually called under the hood in
    2013. Recommended for minutes-per-eval external-SCF runs.
  * **iminuit MIGRAD + HESSE** — the same Minuit2 / ROOT
    minimizer the original recipe used, modern Python wrapping.
    Used at the BOBYQA optimum for publication-grade parameter
    uncertainties.
  * **scipy.optimize** — L-BFGS-B, trust-region, Nelder-Mead. Best
    when evaluations are cheap.
* **Transports** (how external SCF jobs reach a CPU):
  * `local` — direct subprocess to a locally-installed program.
  * `vq` — submit via the [vibe-queue](https://github.com/vibe-qc/vibe-queue) cross-machine
    job queue (target: compute-host-d for this lab's CRYSTAL setup).
* **Outputs:** the optimized basis as a per-element CRYSTAL /
  Gaussian / NWChem / BSE file, parameter uncertainty estimates,
  per-compound energy table, optimizer trajectory for the paper.

## Why a separate package?

The optimizer is a **driver** for external programs — it doesn't
compute SCF energies itself. That's the same architectural shape
as `vq` (which drives external compute jobs to remote daemons),
and a different architectural shape from `vibe-qc` (which IS a
quantum-chemistry program). Keeping `vibe-basis` separate gives:

* **Clean install for collaborators** — the source lifecycle installer
  creates a dedicated environment with NumPy + Click + the selected optimizer
  profile; no vibe-qc / libint / libxc dependency. vibe-basis is not on PyPI
  yet, so bare `pip install vibe-basis` does not resolve.
* **License clarity** — MPL-2.0, no transitive GPL-via-FFTW3
  caveat (vibe-qc carries that).
* **Engine-neutral abstraction** — the `backends/` namespace
  forces every supported program to implement the same
  ``(emit_input, run, parse_output)`` triple, so adding
  ORCA / Gaussian later is a single new module rather than a
  rewrite.
* **Standalone CLI** — `vb --version`, `vb --help`, and
  `vb parse crystal OUTPUT` without touching vibe-qc's Python imports. Fit,
  parity, and test-set CLI commands remain planned.

## Install

Standalone (no vibe-qc needed):

```sh
./vibe-basis/scripts/install.sh                  # standard: SciPy + CLI
./vibe-basis/scripts/install.sh --extras all     # every optimizer
# Install the separate vibe-queue project for remote submission.
```

| Profile | Contents |
| --- | --- |
| `core` | Click, NumPy, and `vb` |
| `standard` | Core plus SciPy; default |
| `optimizers` | SciPy, NLopt, and iminuit |
| `all` | All optimizer backends |
| `test` | SciPy and pytest |

The standalone environment lives at `vibe-basis/.venv`, does not build
vibe-qc, and works on macOS and Linux. Its full lifecycle is:

```sh
./vibe-basis/scripts/update.sh       # fast-forward current branch + rebuild
./vibe-basis/scripts/reinstall.sh    # rebuild without changing Git
./vibe-basis/scripts/uninstall.sh    # remove only the checkout-owned venv
```

The [toolset lifecycle guide](../docs/toolset_lifecycle.md) compares these
commands with the sibling vibe-qc, vibe-view, and vq installers, including
Python requirements, rollback behavior, and uninstall data retention.

Run any command with `--dry-run` to inspect its target first. The `vq`
transport shells out to the independently managed `vq` CLI; it does not
import a Python package. Install vq through `./scripts/install.sh` in its independent vibe-queue
checkout. The former co-located `--with-vq` recipe no longer applies.
Update and reinstall preserve the recorded profile, vq choice, and base Python
unless you override them. vibe-basis has no pre-marker adoption mode: inspect
and move an unmarked environment aside, or select a new `--venv` path.

From a vibe-qc checkout, as vibe-qc's `[basisopt]` extra:

```sh
./scripts/install.sh --extras basisopt       # fresh root environment
./scripts/reinstall.sh --extras basisopt     # existing root environment
```

For an advanced manual pip environment, install both local projects with that
environment's interpreter:

```sh
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e './vibe-basis[all]'
```

Installing it this way is what makes
`vibeqc.basis_optimization`'s CRYSTAL recipe modules importable —
`calculators` and `recipes.crystal_stage1..3` / `crystal_objective` /
`production` import `vibe_basis` directly.

**vibe-basis is not a managed vq program** and has no fleet rollout
lane. It does have its own release test lane. It is a *client*: it emits
decks and submits them through the
`vq` CLI, and what runs on the compute host is CRYSTAL, not this
package. The reasoning is recorded in
[`handovers/HANDOVER_VIBE_BASIS_DEPLOYMENT.md`](../handovers/HANDOVER_VIBE_BASIS_DEPLOYMENT.md).

## Roadmap

vibe-basis is its own endeavour, with its own roadmap:
[`ROADMAP.md`](ROADMAP.md). It carries the mission (beat CRYSTAL23's
`OPTBASIS` on cohesive energies, with vibe-qc as the engine), the
engine policy, milestones M-0 through M6 against the version line
below, the dependencies it has on vibe-qc, and the sync contract with
[`docs/roadmap.md`](../docs/roadmap.md).

That file also carries the **tier contract** (§ 3): which side of the
vibe-qc / vibe-basis boundary owns what. Short version: anything that
would still work with an external code and no vibe-qc installed lives
here; anything needing vibe-qc internals or the native build lives in
`python/vibeqc/basis_optimization/`. The arrow never reverses, and
`tests/test_no_vibeqc_dependency.py` enforces it.

Day-to-day state (current progress, SHAs, blockers, asks) is in the
working tracker,
[`handovers/HANDOVER_BASISOPT.md`](../handovers/HANDOVER_BASISOPT.md).

## Versioning

vibe-basis carries its **own** SemVer line, independent of vibe-qc's —
as `vq` and `vibeview` do. Bump it in the same merge as the change that
earns it, keeping `pyproject.toml` and `src/vibe_basis/__init__.py` in
lockstep (`tests/test_versioning.py` pins this). The scheme, the
per-bump rules and the 1.0.0 criterion are in
[`VERSIONING.md`](VERSIONING.md).

## Status

* **2026-08-08** (0.10.0): Complete standalone lifecycle for macOS and
  Linux, including rollback-capable replacement and marker-guarded removal;
  dedicated release CI; test dependencies now match the suite; the external
  `vq` CLI no longer makes public package resolution unsatisfiable.
* **2026-07-26** (0.3.0): **M-0 complete.** One engine abstraction
  (`engine.EnergyEngine` + `EngineEnergy`, which carries the engine and
  its version alongside every number), with `Crystal23Engine` and a
  declared-but-unbuilt `GpawEngine` under `engines/`; `VibeQcEngine`,
  the primary, lives on the vibe-qc side because it needs the native
  build. The CRYSTAL backend is **retargeted to CRYSTAL23**:
  `backends/crystal{,_atom}.py`, a `probe_version()` reading the
  start-up banner, and a `crystal_version` on every parsed result, so
  which CRYSTAL produced a number is recorded rather than assumed.
  `vb parse crystal <out>` now prints the detected version. Breaking:
  the `backends.crystal14*` paths are gone.
* **2026-07-26** (0.2.0): Installable into the vibe-qc venvs via
  vibe-qc's `[basisopt]` extra, so the version is verifiable wherever
  vibe-qc is installed rather than merely present as source. Adopted an
  independent version line (`VERSIONING.md`); Python floor lowered to
  match vibe-qc's `>=3.11` so the extra resolves across vibe-qc's whole
  supported range. Confirmed as a **deliberate non-target** for fleet
  rollout.
* **2026-05-14**: Backend parser + emitter (`crystal14.py`), two
  transport implementations (`local` and `vq`), 39-compound
  crystal-structure database (`io/structures.py`), PT2013 reference
  energies (`io/references.py`), and the **Goal 8 Stage 0 driver**
  (`recipes/pob_parity.py`) all landed.  78 tests pass.
  The recipe driver wires the full pipeline: structure → emit .d12 →
  transport.submit → parse output → compare to reference.
* **2026-05-13**: Skeleton + first backend module
  (`backends/crystal14.py` — output parser) lands. Tests pass.
  No optimizer yet; no transport yet; no CLI surface beyond
  `vb --version` + `vb parse crystal14 <out>`.

## Running on a remote machine

For the maintainer's own setup, heavy CRYSTAL runs go to a
shared remote box (compute-host-d, 128 GB / 16 cores) via the
[vibe-queue](https://github.com/vibe-qc/vibe-queue) `vq`
job queue. The convention is to keep one dedicated clone per
work track:

    /home/<user>/gitlab/vibeqc-basis/

with the `basissetdev` branch checked out and `vibe-basis[all]`
installed in a per-clone venv. The separately installed `vq` daemon shared across
sibling checkouts dispatches CRYSTAL jobs locally on that
host. Academic collaborators with their own cluster setups can
swap `vq` for a `[slurm]` or `[pbs]` extra once those transports
land (planned, not yet implemented).

See `docs/basisset_dev/GOAL8_MPEI_TZVP.md` in the vibe-qc
repository for the design + acceptance gates for the first
production use of vibe-basis: the `mpei-TZVP` Hartree-Fock-
optimized basis set (paper-targeted, 2026-Q3).

## License

Mozilla Public License 2.0 — see `LICENSE` (inherited from the
parent vibe-qc repository, where vibe-basis deliberately remains).

## Citation

When `vibe-basis` produces basis sets used in published work,
cite:

* The basis paper itself (e.g. mpei-TZVP — Peintinger 2026, in
  preparation).
* The external SCF program used (CRYSTAL23 / ORCA / …).
* The PT2013 / VO2019 methodology lineage:
  * M. F. Peintinger, D. Vilela Oliveira, T. Bredow,
    *J. Comput. Chem.* **34**, 451 (2013).
  * D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow,
    *J. Comput. Chem.* **40**, 2364 (2019).
  * J. Laun, T. Bredow, *J. Comput. Chem.* **43**, 839 (2022) — the
    fifth-period extension.
* If counterpoise-corrected free atoms were used (`counterpoise=True`,
  i.e. CRYSTAL `ATOMBSSE`): S. F. Boys, F. Bernardi, *Mol. Phys.*
  **19**, 553 (1970), doi:10.1080/00268977000101561.
