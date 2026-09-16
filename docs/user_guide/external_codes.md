# External QC codes (ORCA / Psi4 / others)

vibe-qc plays well with other quantum-chemistry codes via ASE's
calculator interface. The :mod:`vibeqc.benchmark` cross-validation
framework treats ORCA, NWChem, Gaussian, Q-Chem, GAMESS-US, and
Turbomole as peers: same Atoms goes through every available code,
results land in a shared comparison table, you assert agreement
(or investigate the disagreement).

PySCF, Psi4, and CRYSTAL are handled differently. vibe-qc never
imports them in-process (CLAUDE.md § 10); they are driven as
external programs via subprocess runners under
[`examples/regression/core/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/core/) and
their output is parsed back. See § "PySCF" and § "Psi4" below for
the runner pattern, and § "CRYSTAL14" further down for the
periodic-reference variant.

This page documents:

1. Which external codes vibe-qc's benchmark framework supports.
2. How to install each (we **don't** redistribute external codes,
   user provisions, we point at the installation).
3. The Python-environment realities (Psi4's conda lock-in,
   ASE-Python-API vs ASE-FileIO modes, etc.).
4. Licensing / redistribution constraints worth knowing about.

```{tip}
The fastest sanity check that the codes are wired up correctly is
``python -c "from vibeqc.benchmark import print_calculator_availability;
print_calculator_availability()"`` — it probes every supported
calculator and reports which is available, with diagnostic notes
on what's missing.
```

## Supported calculators at a glance

| Code | Type | Implementation | License | Status |
|---|---|---|---|---|
| **vibe-qc** | Python-native | Built-in | MPL-2.0 | always available |
| **PySCF** | Subprocess reference | Out-of-process runner (`examples/regression/core/runner_pyscf.py`) | Apache-2.0 | `pip install pyscf` |
| **ORCA** | FileIO subprocess | ASE's `ase.calculators.orca` | Proprietary, **academic-free** | user-installed binary |
| **Psi4** | Subprocess reference | Out-of-process runner (`examples/regression/core/runner_psi4.py`) | LGPL-3.0 | `pip install psi4` or psi4conda via `VIBEQC_PSI4_PYTHON` |
| **NWChem** | FileIO subprocess | ASE's `ase.calculators.nwchem` | ECL-2.0 | user-installed binary |
| **Gaussian** | FileIO subprocess | ASE's `ase.calculators.gaussian` | Commercial | user-installed binary |
| **Q-Chem** | FileIO subprocess | ASE's `ase.calculators.qchem` | Commercial | user-installed binary |
| **GAMESS-US** | FileIO subprocess | ASE's `ase.calculators.gamess_us` | Free for academic use | user-installed binary |
| **Turbomole** | FileIO subprocess | ASE's `ase.calculators.turbomole` | Commercial | user-installed binary |

"FileIO subprocess" means the ASE wrapper writes an input file,
spawns the external binary as a subprocess, and parses the output,
so the binary has to be on `$PATH`. "Python-bindings" means ASE
imports the code's Python module, so it has to be importable in
the same interpreter as vibe-qc.

## PySCF (subprocess reference)

[PySCF](https://pyscf.org/) is the Python-native QC stack we use
for cross-validation. It's Apache-2.0, on PyPI, and installs into
the same venv as vibe-qc:

```sh
.venv/bin/pip install pyscf
```

```{important}
Although PySCF is importable in the same interpreter, vibe-qc
does **not** import it in-process. Per CLAUDE.md § 10, other QC
programs are treated as external references and driven
out-of-process. The previous in-process `PySCFCalculator` shim
in `vibeqc.benchmark` was retired in v0.8.x.
```

The canonical pattern is the subprocess runner
[`examples/regression/core/runner_pyscf.py`](../../examples/regression/core/runner_pyscf.py),
which spawns PySCF in a fresh Python process, runs the requested
calculation, and parses the output back. Driver code uses it via
the regression-suite case interface (`compare.py`, `run_suite.py`)
rather than calling `pyscf` directly.

For ad-hoc verification scripts that need PySCF in-process, keep
them under `examples/regression/scripts/` or `tests/` (never inside
`python/vibeqc/`). The worked example is the importorskip-guarded
PySCF cross-check in
[`tests/test_ccsd.py`](../../tests/test_ccsd.py): it runs wherever a
developer has PySCF installed and skips everywhere else.

## ORCA

[ORCA](https://www.faccts.de/orca/) is a feature-rich QC package
from the Max-Planck-Institut für Kohlenforschung + FACCTS GmbH.
**Free for academic use** with registration at
<https://orcaforum.kofo.mpg.de/>; commercial licensing via FACCTS.

```{warning}
**vibe-qc does not redistribute ORCA.** You'll need to download
and install it yourself from the official site. If you're at a
university or non-profit research institution, the academic
license is free; commercial users need a FACCTS license. The
binary install for ORCA 6.x is ~3 GB and includes its own MPI
+ supporting libraries.
```

### macOS install

The macOS binary ships unsigned outside the Mac App Store, so
Gatekeeper will refuse to load the dylibs unless you remove the
quarantine attribute:

```sh
# After unzipping the official tarball to e.g. ~/orca_6_1_1_macosx_arm64_openmpi411
cd ~/orca_6_1_1_macosx_arm64_openmpi411
sudo xattr -r -d com.apple.quarantine .
./orca --version       # should print the ORCA banner
```

### Linux install

Linux binaries usually run as-is once unzipped. Verify with:

```sh
cd ~/orca_6_1_1_linux_x86_64_openmpi411
./orca --version
```

### Make ORCA discoverable

The vibe-qc framework looks for the `orca` binary via, in order:

1. Environment variable `ASE_ORCA_COMMAND` (full command template,
   e.g. `"orca PREFIX.inp > PREFIX.out"`)
2. Environment variable `ORCA_COMMAND` (just the binary path)
3. Environment variable `ORCA_PATH` (install directory; we look
   for `$ORCA_PATH/orca`)
4. `which orca`, `$PATH` lookup

Pick whichever fits your shell / shell-init story. Common pattern:

```sh
# In .bashrc / .zshrc:
export ORCA_PATH="$HOME/orca_6_1_1_macosx_arm64_openmpi411"
export PATH="$ORCA_PATH:$PATH"
```

Then `make_orca_calculator(...)` returns a ready-to-use ASE
calculator:

```python
from vibeqc.benchmark import make_orca_calculator

orca = make_orca_calculator(orcasimpleinput="HF 6-31G* EnGrad")
if orca is None:
    print("ORCA not found — set ORCA_PATH or add orca to PATH")
```

`orcasimpleinput=` is the ORCA "simple input" line, the same
syntax you'd put on the `!` line in an ORCA input file. Always end
with `EnGrad` if you want forces (otherwise ASE falls back to
finite-difference on energies, which is much slower).

### License + redistribution

* ORCA's license **explicitly forbids redistribution** of the
  binaries. Don't bundle ORCA into a Docker image, conda
  package, or shared-server install that other users access.
  Each user provisions their own copy under their own license.
* The ORCA forum at <https://orcaforum.kofo.mpg.de/> requires
  registration but is free for academics; commercial users go
  through <https://www.faccts.de/>.
* Output files (`.out`, `.gbw`, `.hess`) are yours to share /
  publish freely, the license restricts the binary, not the
  results.

## Psi4 (subprocess reference)

[Psi4](https://psicode.org/) is a permissive-licensed (LGPL-3.0)
QC package with strong emphasis on coupled-cluster methods.

```{important}
Like PySCF, vibe-qc treats Psi4 as an external reference and never
imports it in-process (CLAUDE.md § 10). The previous in-process
`make_psi4_calculator` helper in `vibeqc.benchmark` was retired in
v0.8.x; the regression-suite runner now spawns Psi4 in a separate
Python interpreter.
```

### Installing Psi4

Two paths, depending on what Psi4 distribution you can get:

1. **Pip-install psi4 into the vibe-qc venv** (when available):
   ```sh
   pip install psi4
   ```
   Psi4 isn't always on PyPI; track
   <https://psicode.org/psi4manual/master/build_planning.html>.
2. **Use psi4conda as a separate env**. The official installer
   ships a self-contained conda env pinning its own Python
   version. Because the subprocess runner spawns a separate
   interpreter, you do **not** need to install Psi4 into the
   vibe-qc venv: point `VIBEQC_PSI4_PYTHON` at the psi4conda
   env's `python`:
   ```sh
   export VIBEQC_PSI4_PYTHON="$HOME/psi4conda/bin/python"
   ```
   This sidesteps the conda Python-version trap that used to
   plague the in-process wrapper.

### Detection

`detect_calculators()` reports Psi4 as an external-reference-only
entry; it does not attempt `import psi4` in vibe-qc's interpreter:

```text
psi4    ✗ missing  (External reference only; vibe-qc does not
                    import Psi4. Use a subprocess runner/script
                    and parse its output.)
```

The `✓` marker only appears when a `psi4` *binary* is on `$PATH`;
the relevant test for whether the runner will succeed is whether
`$VIBEQC_PSI4_PYTHON -c "import psi4"` works.

### Driving Psi4 from the regression suite

The canonical pattern is the subprocess runner
[`examples/regression/core/runner_psi4.py`](../../examples/regression/core/runner_psi4.py),
which serializes the requested calculation to JSON, launches a
separate Python (`VIBEQC_PSI4_PYTHON`, defaults to `sys.executable`),
runs `psi4.energy(method)`, and parses one result line back.
Driver code uses it via the regression-suite case interface
(`compare.py`, `run_suite.py`) rather than calling `psi4` directly.

For ad-hoc verification scripts that must drive Psi4 in-process,
keep them under `examples/regression/scripts/` or `tests/`, never
under `python/vibeqc/`.

### License + redistribution

* Psi4 is LGPL-3.0; vibe-qc users can install it freely. We
  don't redistribute it because the conda-only distribution +
  Python-version coupling makes redistribution awkward, not
  because licensing forbids it.
* Output files: yours to use freely (LGPL doesn't restrict
  outputs).

## NWChem / Gaussian / Q-Chem / GAMESS-US / Turbomole

These are FileIO subprocess calculators in the same family as
ORCA, install the binary, put it on `$PATH`, the ASE wrapper
spawns it. `detect_calculators()` finds them via `shutil.which()`
on their conventional binary name:

| Code | Binary | License | Notes |
|---|---|---|---|
| NWChem | `nwchem` | ECL-2.0 (academic-friendly) | <https://nwchemgit.github.io/> |
| Gaussian 16 | `g16` | Commercial only | <https://gaussian.com/> |
| Q-Chem | `qchem` | Commercial | <https://www.q-chem.com/> |
| GAMESS-US | `rungms` | Free for academic with registration | <https://www.msg.chem.iastate.edu/gamess/> |
| Turbomole | `dscf` | Commercial | <https://www.turbomole.org/> |

vibe-qc's framework doesn't ship a `make_<code>_calculator()`
helper for each of these, use ASE's wrapper directly:

```python
from ase.calculators.nwchem import NWChem
from vibeqc.benchmark import compare_calculators
from vibeqc.ase import VibeQC

calc_nwchem = NWChem(label="nwchem-h2o",
                     basis="6-31g*", task="energy")

results = compare_calculators(
    atoms,
    [
        ("vibe-qc/RHF/6-31G*", VibeQC(basis="6-31g*")),
        ("NWChem/RHF/6-31G*",  calc_nwchem),
    ],
    properties=("energy", "forces"),
)
```

If you regularly cross-check against one of these, an issue
requesting a `make_<code>_calculator()` helper alongside ORCA / Psi4
is a welcome contribution, same shape as `make_orca_calculator`
in :mod:`vibeqc.benchmark`.

## Sanity-check the setup

`scripts/check_external_codes.sh` (see Phase F roadmap) is the
one-stop diagnostic, it runs `print_calculator_availability()`
and prints the result. Until it ships, the same one-liner works:

```sh
.venv/bin/python -c "
from vibeqc.benchmark import print_calculator_availability
print_calculator_availability()
"
```

Expected output on a system with vibe-qc + ORCA installed but no
other external codes (PySCF is intentionally not in-process here;
use the subprocess runner for parity data):

```text
gamess_us  ✗ missing    (Python wrapper present, but 'rungms' not on $PATH …)
gaussian   ✗ missing    (Python wrapper present, but 'g16' not on $PATH …)
nwchem     ✗ missing    (Python wrapper present, but 'nwchem' not on $PATH …)
orca       ✓ available  (/Users/.../orca_6_1_1_macosx_arm64_openmpi411/orca)
psi4       ✗ missing    (External reference only; vibe-qc does not import Psi4. Use a subprocess runner/script and parse its output.)
pyscf      ✗ missing    (External reference only; vibe-qc does not import PySCF. Use a subprocess runner/script and parse its output.)
qchem      ✗ missing    (Python wrapper present, but 'qchem' not on $PATH …)
turbomole  ✗ missing    (Python wrapper present, but 'dscf' not on $PATH …)
vibeqc     ✓ available  (Python-native)
```

## Per-calculator timeouts

`compare_calculators(..., timeout_s=None)` runs every calculator
**in-process and serially**, that is the historical and default
behaviour. A calculator that hangs (a stuck SCF, a wedged subprocess,
a slow ORCA pseudopotential read on a flaky NFS mount) will hang the
whole benchmark until you Ctrl-C the script. Python threads can't be
cancelled from the outside and a worker blocked in C or Fortran won't
respond to signals from the same process, so there is no honest way to
enforce a wall-time cap on the in-process path.

When you need a hard deadline, for example a CI cross-validation
sweep that must never wedge the runner, pass `timeout_s`:

```python
from vibeqc.benchmark import compare_calculators
from vibeqc.ase import VibeQC

def make_vibeqc_rhf():
    return VibeQC(basis="6-31g*")  # constructed in the child

results = compare_calculators(
    atoms,
    [("vibe-qc/RHF/6-31G*", make_vibeqc_rhf)],
    properties=("energy", "forces"),
    timeout_s=300.0,  # 5 minutes per calculator, hard cap
)
```

With `timeout_s` set, each calculator runs in its own
`multiprocessing.spawn` child process. Past the deadline the child is
SIGTERM'd (then SIGKILL'd if it ignores the signal) and the row is
recorded with `status='timeout'`. A child that dies without returning
a result, a segfault, an unhandled C++ exception, an OOM kill,
records `status='crashed'` with the exit code in the error message.

### Process isolation requires picklability

Both the calculator (or its factory) and the `Atoms` object must be
picklable so the parent can hand them to the child. The framework
checks this **before** spawning and raises `ValueError` with a clear
hint if it can't.

The safest, most ergonomic pattern is a **module-level zero-arg
factory**, a plain `def` in your script (or in a helper module) that
returns a configured calculator. Factories sidestep two common
pitfalls:

1. **Unpicklable bindings.** Configured Psi4 / ORCA / PySCF wrapper
   instances often hold C-extension handles or open file descriptors
   that pickle can't serialise. Construct them inside the child
   process via a factory and the issue disappears.
2. **Inadvertent state sharing.** A factory guarantees each child
   gets a freshly-initialised calculator with no carry-over results
   cache from the parent.

Lambdas and locally-nested functions are **not** picklable by
`spawn`, define factories at module scope.

### When to use which

* **`timeout_s=None`** (default), interactive scripts, notebooks,
  single-calculator runs, anything where you trust every calculator
  in the panel and want zero extra overhead.
* **`timeout_s=<seconds>`**, CI cross-validation, overnight
  benchmark sweeps, panels that mix vibe-qc against external codes
  whose stability you don't control.

The subprocess path adds a per-calculator overhead of roughly one
Python interpreter cold start (≲ 1 s on modern hardware, plus any
`import vibeqc` time inside the child). For a benchmark of dozens of
short calculators that overhead matters; for production-sized SCF
runs it is negligible.

## CRYSTAL14, periodic regression reference (out-of-process)

CRYSTAL14 isn't an ASE-driven `vibeqc.benchmark` calculator: it's a
**regression-suite-only** periodic reference dispatched via vibe-queue
(`vq submit`) and parsed back from its `.out` file.

```{important}
The runner lives at
`examples/regression/core/runner_crystal.py`. It exists to give the
parity suite a third witness for periodic energies alongside vibe-qc
and PySCF.pbc — the asbestos-polymorphs paper consumer needs CRYSTAL
as the published reference column for its POB-TZVP energies. It is
not part of the `vibeqc.benchmark` calculator surface and is not
imported by `python/vibeqc/`.
```

### Why out-of-process

Per [`CLAUDE.md`](../CLAUDE.md) § 10, vibe-qc never imports another
QC program at runtime. CRYSTAL14 is a *program*, not a library, so
the runner spawns it as a subprocess via the
[`vibe-queue`](https://github.com/vibe-qc/vibe-queue) dispatcher: input
deck staged on the local disk, `vq submit` ships it to the configured
host (selected through your queue configuration; any host with `crystal` / `Pcrystal` on
PATH and the `run-crystal.sh` wrapper installed works), `vq status`
polls until terminal, `vq fetch` brings the workspace back, the
parser scrapes the `.out`.

### Opting in

```sh
# Default regression run — two-way vibe-qc vs PySCF.pbc periodic.
.venv/bin/python -m examples.regression.run_suite

# Three-way with CRYSTAL14 as the third column. Requires `vq` on
# PATH and a configured default_host. Periodic cases whose method or
# basis is outside the v1 whitelist quietly land as
# status='unavailable' — the suite still runs end-to-end.
.venv/bin/python -m examples.regression.run_suite --include-crystal
```

### Method + basis whitelist (v1)

| vibe-qc method | CRYSTAL keyword | Notes |
|---|---|---|
| `rks-lda` | `SVWN` | Matches PySCF `slater,vwn5` + ORCA `LDA` |
| `rks-pbe` | `PBE` | |
| `rks-blyp` | `BLYP` | |
| `rks-b3lyp` (also `rks-b3lyp5`) | `B3LYP` | Flavor-matched: CRYSTAL's B3LYP keyword is the **VWN5** variant (verified to 1e-9 Ha on H₂/STO-3G), exactly vibe-qc's bare `b3lyp` (ORCA definition). The Gaussian-flavor spellings (`b3lypg` / `b3lyp/g`) are deliberately unsupported here: CRYSTAL14 cannot express VWN-RPA, and silently mapping them would bias every parity row by the ~6.8 mHa/cell flavor gap. PySCF references are the mirror case, its bare `b3lyp` is the Gaussian flavor, so PySCF-side runs must spell `b3lyp5`. |
| `rhf` | `HF` | No DFT block emitted |
| `uhf` / `uks-*` | - | Out of scope for v1 (CRYSTAL needs an explicit SPINLOCK block) |
| `mp2` / `mp2-df` | - | LCMP2 needs a distinct input topology |

| basis name | CRYSTAL library | Notes |
|---|---|---|
| `sto-3g` | `STO-3G` | |
| `pob-dzvp-rev2` | `POB-DZVP-REV2` | |
| `pob-tzvp-rev2` | `POB-TZVP-REV2` | Asbestos-paper canonical reference |

Anything outside these tables emits `status='unavailable'` with the
specific row showing which axis blocked it. The whitelist grows when
the regression suite needs a new combination, start at
`_CRYSTAL_FUNC_MAP` / `_CRYSTAL_BASIS_LIBRARY` in `runner_crystal.py`.

### Env-var knobs

The runner reads these at every invocation. The wrapper must be configured;
the host defaults to the target selected in your private queue configuration.

| Var | Default | Purpose |
|---|---|---|
| `VIBEQC_CRYSTAL_VQ_HOST` | vq's `default_host` | Override the target host |
| `VIBEQC_CRYSTAL_CPUS` | `14` | MPI ranks passed to `run-crystal.sh` |
| `VIBEQC_CRYSTAL_WALL_TIME_SECONDS` | `7200` | vq watchdog cap (2 h) |
| `VIBEQC_CRYSTAL_POLL_INTERVAL_S` | `30` | `vq status` poll cadence |
| `VIBEQC_CRYSTAL_WRAPPER` | *(required)* | Absolute path to the wrapper *on the remote host*, e.g. `~/gitlab/vibe-queue/contrib/run-crystal.sh`. Unset → status='unavailable' |

### Symmetry handling

The runner emits P1 (space group 1) decks with the full atom list
from the SPEC. CRYSTAL handles them correctly, just without
symmetry-driven SCF acceleration, the cross-code Δ stays meaningful;
the wallclock is higher than a hand-tuned asymmetric-unit deck would
deliver. Symmetry-exploiting deck generation is a future enhancement
for large-cell paper work (per-family asymmetric-unit reduction);
not blocking the regression-suite use case.

### Licensing

CRYSTAL14 is **proprietary, academic-licensed**. vibe-qc does not
redistribute it, ship its inputs, or assume its presence, the
runner self-gates on missing binary / missing vq, and the regression
suite stays runnable without it. Cite CRYSTAL per the per-version
references in <https://www.crystal.unito.it/cite-crystal.php> when
your published work consumes the runner's output.

## See also

* [ASE integration user guide](ase_integration.md), how the
  `vibeqc.ase.VibeQC` calculator works, and how it slots into
  the comparison framework.
* [cross-validation tutorial](../tutorial/cross_validation.md)
  - end-to-end walkthrough using ORCA + Psi4 + PySCF.
* `examples/ase_compare/`, runnable scripts:
  H₂O HF / DFT cross-validation, water dimer dispersion,
  basis-set convergence sweep, NH₃ NEB barrier.
* `examples/regression/core/runner_crystal.py`, source of the
  CRYSTAL14-via-vq runner described above.
