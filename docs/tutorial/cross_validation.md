# Cross-validating against ORCA, Psi4, and PySCF

**You'll learn:** how to run the same calculation through vibe-qc
and one or more external QC codes, build a comparison table,
and assert numerical agreement to a target tolerance.

**Why:** every QC code has bugs; cross-checking against
independent implementations is the gold-standard way to know your
number is right. Cross-validation is also how new methods get
*defensible*, agreement with ORCA-CCSD(T) or PySCF-MP2 is what
turns "vibe-qc says X" into "vibe-qc + ORCA + PySCF agree on X."

**Prerequisites:**
* vibe-qc installed (Phase A v0.5+ for the dipole / Hessian /
  polarizability cross-checks; energy + forces work in any v0.4+).
* PySCF (`pip install pyscf` into the vibe-qc venv, Apache-2.0,
  cheap, recommended).
* Optionally ORCA, Psi4, NWChem, see the
  [external codes user guide](../user_guide/external_codes.md)
  for install paths.

This tutorial works at three levels of degradation:

1. **vibe-qc only**, baseline; ships with the base install.
2. **+ ORCA**, adds dispersion (D3-BJ), grid-density-independent
   DFT cross-checks, and a different integral library so a
   numerical bug in our integrals shows up as a real disagreement.
3. **+ Psi4**, adds a third independent implementation with
   different conventions; useful when you need to pinpoint *where*
   a disagreement comes from.

> **Note on PySCF.** The in-process ``PySCFCalculator`` shim was
> retired in v0.8.x (per the repo standing rule against importing
> external QC programs). Out-of-process parity is done via the
> runner scripts under `[`examples/regression/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/)`.

## Setup: detect what's available

Before building any comparison table, ask the benchmark module which
calculators it can reach. `print_calculator_availability` probes for
vibe-qc plus each supported external code and reports what is wired up:

```python
from vibeqc.benchmark import print_calculator_availability

print_calculator_availability()
# vibeqc     ✓ available  (Python-native)
# orca       ✓ available  (/opt/orca_6_1_1/orca)
# psi4       ✗ missing    (External reference only; use subprocess runner...)
# pyscf      ✗ missing    (External reference only; use subprocess runner...)
# ...
```

The framework auto-detects ORCA via `$ASE_ORCA_COMMAND` /
`$ORCA_COMMAND` / `$ORCA_PATH` / `which orca`. PySCF and Psi4 are
driven out-of-process via the subprocess runners under
[`examples/regression/core/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/core/) and
deliberately show as `missing` in this in-process detector; point
`VIBEQC_PYSCF_PYTHON` / `VIBEQC_PSI4_PYTHON` at an interpreter
that can `import pyscf` / `import psi4`. NWChem / Gaussian / Q-Chem
are auto-detected if their binaries are on `$PATH`. See the
[external codes guide](../user_guide/external_codes.md) for the
full setup matrix.

## Level 1: H₂O / RHF agreement

The simplest validation: same molecule, same basis, three codes,
identical SCF skeleton. Energies and forces should agree to ~µHa.

```python
from ase.build import molecule
from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    compare_calculators, make_orca_calculator,
)

atoms = molecule("H2O")          # ASE's stock H2O geometry

# PySCF is retired as an in-process shim; see tutorial note above.
calculators = [
    ("vibe-qc/RHF/6-31G*", VibeQC(basis="6-31g*")),
]
orca = make_orca_calculator(orcasimpleinput="HF 6-31G* EnGrad")
if orca is not None:
    calculators.append(("ORCA/RHF/6-31G*", orca))

results = compare_calculators(
    atoms,
    calculators,
    properties=("energy", "forces"),
)
results.print_table()
results.assert_agreement(reference="vibe-qc/RHF/6-31G*",
                         tol={"energy": 1e-5, "forces": 1e-4})
```

**Expected output** (verified end-to-end on the v0.5 development
machine):

```text
label              │ status │ wall_time_s │ energy          │ |F|max
────────────────────────────────────────────────────────────────────────
vibe-qc/RHF/6-31G* │ ok     │  0.04s      │ -2068.294643 eV │ 1.4934e+00
ORCA/RHF/6-31G*    │ ok     │  0.72s      │ -2068.294643 eV │ 1.4934e+00

✓ all calculators agree to within 1e-5 eV / 1e-4 eV·Å⁻¹
```

For a PySCF column in this table, use the out-of-process subprocess
runner at [`examples/regression/core/runner_pyscf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/core/runner_pyscf.py)
(the previous in-process `PySCFCalculator` ASE shim was retired, 
CLAUDE.md § 10 forbids in-process imports of external QC programs
under `python/vibeqc/`).

**Six-decimal agreement on energy AND |F|max.** That's the contract
for "different codes, same RHF/basis combination": SCF identifies
the ground state to whatever convergence threshold the code is
set to (typically 1e-7 Hartree), and the energy + forces are then
deterministic.

## Level 2: H₂O / DFT, different grids, different residuals

DFT introduces a **grid-discretization error** that's
calculator-specific. Each code chooses its own DFT integration
grid by default; energies typically agree to a few mHa, not 1 µHa
like HF.

```python
from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    compare_calculators, make_orca_calculator,
)
from ase.build import molecule

atoms = molecule("H2O")

# PySCF was previously a third row via the in-process
# PySCFCalculator ASE shim; retired (CLAUDE.md § 10). For a
# PySCF cross-check, use the out-of-process runner at
# examples/regression/core/runner_pyscf.py.
calculators = [
    ("vibe-qc/PBE/6-31G*", VibeQC(basis="6-31g*",
                                  functional="PBE")),
]
orca = make_orca_calculator(
    orcasimpleinput="PBE 6-31G* EnGrad",
)
if orca is not None:
    calculators.append(("ORCA/PBE/6-31G*", orca))

results = compare_calculators(
    atoms, calculators,
    properties=("energy", "forces", "dipole"),
)
results.print_table()

# Looser tolerance for DFT — different grid choices contribute
# O(0.1-1 mHa) to the energy. Tighter agreement requires matching
# grid sizes across codes (out of scope here).
results.assert_agreement(
    reference="vibe-qc/PBE/6-31G*",
    tol={"energy": 5e-3, "forces": 1e-2, "dipole": 1e-2},
)
```

**Expected output:**

```text
label              │ status  │ wall_time │ energy          │ |F|max     │ dipole
────────────────────────────────────────────────────────────────────────────────
vibe-qc/PBE/6-31G* │ ok      │ 1.47s     │ -2076.781785 eV │ 5.39e-01   │ 0.4256 eÅ
ORCA/PBE/6-31G*    │ ok      │ 1.19s     │ -2076.784666 eV │ 5.36e-01   │ 0.4256 eÅ
```

vibe-qc and ORCA agree on dipole to **4 decimals** (0.4256 eÅ).

The 3 mHa energy gap between vibe-qc and ORCA is the **per-code
DFT-grid difference**. Visible, but well within "chemical accuracy"
(40 mHa = 1 kcal/mol). Tightening this requires aligning the grid
between codes, which is method-and-grid-spec-dependent and beyond
the scope of a smoke comparison.

## Level 3: dispersion (D3-BJ) on the water dimer

Adds Grimme's D3 with Becke-Johnson damping on top of HF/6-31G\*.
Vibe-qc and ORCA both implement D3-BJ from the same Grimme
parameter set, so the **dispersion contribution to the
interaction energy** should agree to ~µHa.

```python
from vibeqc.benchmark import (
    compare_calculators, make_orca_calculator,
)
from vibeqc.ase import VibeQC
from ase import Atoms

monomer = Atoms(symbols=["O", "H", "H"], positions=[
    (0.000, 0.000,  0.000), (0.762, 0.000, -0.566),
    (-0.762, 0.000, -0.566)])
dimer = Atoms(symbols=["O", "H", "H", "O", "H", "H"], positions=[
    (0.000, 0.000,  0.000), (0.762, 0.000, -0.566),
    (-0.762, 0.000, -0.566),
    (0.000, 0.000,  2.920), (-0.000, 0.787, 3.469),
    (-0.000, -0.787, 3.469)])

# With D3-BJ
panel_d3 = [
    ("vibe-qc", VibeQC(basis="6-31g*", dispersion="hf")),
]
orca = make_orca_calculator(
    orcasimpleinput="HF 6-31G* D3BJ EnGrad",
)
if orca is not None:
    panel_d3.append(("ORCA", orca))

# Compare interaction energies E(dimer) - 2·E(monomer)
e_mon = compare_calculators(monomer, panel_d3,
                            properties=("energy",))
e_dim = compare_calculators(dimer, panel_d3,
                            properties=("energy",))

for label in (l for l, _ in panel_d3):
    e_int = (e_dim.rows[0].properties["energy"]
             - 2.0 * e_mon.rows[0].properties["energy"])
    print(f"{label}:  E_int = {e_int*1000:+.2f} meV")
```

The dispersion-only contribution
(E_int with D3 − E_int without D3) should agree across vibe-qc
and ORCA to <0.01 kcal/mol, see
[`examples/ase_compare/compare-water-dimer-dispersion.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_compare/compare-water-dimer-dispersion.py)
for the full script with monomer-and-dimer panels and an explicit
assertion.

## Level 4: basis-convergence sweep across two codes

Runs the same molecule through five basis sets, STO-3G → 6-31G\*
→ cc-pVDZ → cc-pVTZ → def2-TZVP, through every available
calculator. **Every code at every basis should produce the same
RHF energy.**

```python
from ase.build import molecule
from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    compare_calculators, make_orca_calculator,
)

BASES = [
    ("sto-3g",     "STO-3G"),
    ("6-31g*",     "6-31G*"),
    ("cc-pvdz",    "cc-pVDZ"),
    ("cc-pvtz",    "cc-pVTZ"),
    ("def2-tzvp",  "def2-TZVP"),
]

atoms = molecule("H2O")
table = {}
for vqname, orcaname in BASES:
    # PySCF parity at each basis goes through the subprocess runner
    # at examples/regression/core/runner_pyscf.py (CLAUDE.md § 10).
    calcs = [
        ("vibe-qc", VibeQC(basis=vqname)),
    ]
    orca = make_orca_calculator(orcasimpleinput=f"HF {orcaname}")
    if orca:
        calcs.append(("ORCA", orca))

    results = compare_calculators(atoms, calcs,
                                  properties=("energy",))
    table[vqname] = {row.label: row.properties.get("energy")
                     for row in results if row.status == "ok"}

# Cross-code consistency check
for vqname, _ in BASES:
    ref = table[vqname]["vibe-qc"]
    for code, val in table[vqname].items():
        if code == "vibe-qc":
            continue
        gap_meV = (val - ref) * 1000
        assert abs(gap_meV) < 1.0, (
            f"{vqname}/{code} disagrees by {gap_meV} meV"
        )
print("✓ vibe-qc × ORCA at 5 bases agree to <1 meV — "
      "integrals + SCF + basis parsing consistent across the stack.")
```

**Verified result on the v0.5 development machine** (full output
under
[`examples/ase_compare/compare-basis-convergence.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_compare/compare-basis-convergence.py)):

```text
basis        ORCA          vibe-qc
sto-3g       -2039.885358  -2039.885358
6-31g*       -2068.294643  -2068.294643
cc-pvdz      -2068.773588  -2068.773588
cc-pvtz      -2069.592889  -2069.592889
def2-tzvp    -2069.645665  -2069.645665

per-basis Δ from vibe-qc:  0.0001 meV (worst case)
```

**Five basis sets × two codes, all agree to ~0.04 µHa.** That's
the chemistry-grade convergence verification the comparison
framework was built for. (For a three-code triangulation including
PySCF, add a third column populated by the out-of-process runner at
[`examples/regression/core/runner_pyscf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/core/runner_pyscf.py); the rule against
in-process PySCF imports, CLAUDE.md § 10, applies to library code
under `python/vibeqc/`, not to example scripts that drive a separate
PySCF interpreter as a subprocess.)

## Level 5: NEB transition-state cross-check

Both vibe-qc and ORCA can drive ASE's climbing-image NEB. The
script
[`examples/ase_compare/compare-nh3-neb-barrier.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_compare/compare-nh3-neb-barrier.py)
runs the full NH₃ umbrella inversion through both, asserts
agreement on the barrier height to 1 kcal/mol, and plots both
MEPs side-by-side.

```python
from vibeqc.ase import VibeQC
from vibeqc.benchmark import make_orca_calculator
from ase import Atoms
from ase.mep import NEB
from ase.optimize import BFGSLineSearch, FIRE

# (Endpoint relaxation + NEB-CI setup — see the script for the
# full ~80 lines.)

barrier_vqc = run_neb(VibeQC(basis="sto-3g"))
barrier_orca = run_neb(lambda: make_orca_calculator(
    orcasimpleinput="HF STO-3G EnGrad"))

assert abs(barrier_vqc - barrier_orca) < 1.0  # kcal/mol
```

The NEB protocol is the same in both runs (5 intermediates, FIRE,
fmax = 0.10 eV/Å), so the only source of difference is the
underlying SCF, which we've already cross-validated to µHa
above. Barrier agreement is therefore a *system-integration*
check rather than a methods check.

## Tolerances by use case

A reference for how tight to set `tol={...}` based on what you're
validating:

| Use case | Tolerance | Rationale |
|---|---|---|
| Same RHF/UHF, same basis, different codes | `1e-5 eV / 1e-4 eV·Å⁻¹` | Identical SCF skeleton; SCF threshold dominates. |
| Same DFT functional, same basis, different codes | `5e-3 eV / 1e-2 eV·Å⁻¹` | Per-code DFT-grid differences (~mHa). |
| MP2 / CCSD same basis | `1e-3 eV` | Iteration thresholds + integral cutoffs. |
| Different basis → "same chemistry" | (no tolerance, depends on the system) | Use as a basis-convergence check, not agreement. |
| D3-BJ contribution alone | `1e-5 eV (~µHa)` | Same Grimme parameters, deterministic sum. |
| NEB barriers | `1 kcal/mol` | NEB-protocol drift dominates over per-code SCF. |

## Next

After running the five comparison scripts in
[`examples/ase_compare/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/ase_compare/) against vibe-qc + PySCF (and ORCA where
available), you have:

* Energy agreement to µHa across HF in 5 basis sets
* DFT agreement to mHa (grid-difference floor)
* Force agreement to 1e-4 eV/Å on every checked system
* D3-BJ agreement to <0.01 kcal/mol
* NEB barrier agreement to <1 kcal/mol

That's **chemistry-grade cross-validation**, the kind of
comparison table you'd put in a Methods section to defend a
vibe-qc result. For a forwarding strategy (when vibe-qc gains
new methods, what should the cross-check look like?), see the
[ASE-tutorial-parity matrix in the roadmap](../roadmap.md).

## See also

* [external codes user guide](../user_guide/external_codes.md)
  - install + licensing for ORCA / Psi4 / others
* [ASE integration user guide](../user_guide/ase_integration.md)
  - how the `vibeqc.ase.VibeQC` calculator works internally
* [`examples/ase_compare/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/ase_compare/), runnable scripts that this tutorial
  walks through
* [example scripts and generated-output guide](../example_outputs.md),
  the runnable inputs and the local outputs they produce
