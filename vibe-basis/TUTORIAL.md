# Basis-Set Optimization Tutorial

This tutorial walks through the vibe-basis optimisation pipeline — from
a smoke-test parity check to a publication-grade basis-set refit.  It
assumes you have access to a machine with CRYSTAL installed (called
`compute-host-a` or `compute-host-d` in the examples) and a laptop with vibe-basis.

```{note}
vibe-basis's CRYSTAL backend targets **CRYSTAL23** (retargeted from
CRYSTAL14 in v0.3.0, 2026-07-26). Per
`src/vibe_basis/backends/crystal.py`'s own docstring, this was
deliberately *not* a rewrite: the `.d12` input format, the
`TOTAL ENERGY(...)` / `SCF ENDED` output lines, and everything else
this tutorial's workflow depends on are stable CRYSTAL09 → 23. The one
functional addition is `probe_version()`, which records which CRYSTAL
version actually produced a given result. See `VERSIONING.md` for the
full history.
```

The vibe-qc repository is currently private; see
[`docs/installation.md`](../docs/installation.md) for how to request
read-only clone access.

## 1. Setup

### 1.1 Install vibe-basis

**Standalone** — vibe-basis on its own, no vibe-qc needed. This is the
lightest option for the parser, package-native workflows, and Stage 0 below.
Stages 1-3 import recipes from `vibeqc.basis_optimization` and require the
combined route in the next section.

```bash
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
./vibe-basis/scripts/install.sh --extras all

vibe-basis/.venv/bin/vb --version
```

This keeps the standalone environment in `vibe-basis/.venv`, separate from
vibe-qc's native-build environment. Use `update.sh`, `reinstall.sh`, and
`uninstall.sh` in the same directory for the remaining lifecycle operations.

**Combined with vibe-qc** — use this route for Stages 1-3 and production.
Install vibe-basis as vibe-qc's `basisopt` extra:

```bash
./scripts/install.sh --dev --extras basisopt      # fresh root environment
# or, for an existing managed root environment:
./scripts/reinstall.sh --extras basisopt
```

The `basisopt` profile installs vibe-basis core. Add its optimizer backends to
the same root environment for the staged recipes:

```bash
.venv/bin/python -m pip install -e './vibe-basis[all]'
.venv/bin/python examples/basisset_dev/check_basisopt_install.py
```

Take this route if you also want the `vibeqc.basis_optimization` CRYSTAL
recipes (`recipes.crystal_stage1..3`, `recipes.production`,
`calculators`) — they import `vibe_basis` directly and are unimportable
without it. If you are rebuilding vibe-qc's C++ core with
`--no-build-isolation`, note the hatchling caveat in
`docs/installation.md`.

If vq should remain independently managed, install it separately and expose
its command to `VqTransport`:

```bash
# From the parent directory, clone the queue separately:
git clone https://github.com/vibe-qc/vibe-queue.git ../vibe-queue
(cd ../vibe-queue && ./scripts/install.sh --python python3.12)
export PATH="$PWD/../vibe-queue/.venv/bin:$PATH"
```

The standalone `vb --version` check and the combined
`check_basisopt_install.py` check verify different boundaries. Do not run the
combined check with `vibe-basis/.venv/bin/python`; that environment
intentionally does not contain vibe-qc.

### 1.2 Configure remote hosts

Add `compute-host-a` and `compute-host-d` to `~/.config/vq/config.toml`:

```toml
default_host = "compute-host-a"

[hosts.compute-host-a]
ssh = "compute-host-a.example.com"
remote_vq = "/home/USER/vibe-queue/.venv/bin/vq"
remote_python = "/home/USER/gitlab/vibeqc-basis/.venv/bin/python"

[hosts.compute-host-d]
ssh = "compute-host-d.example.com"
remote_vq = "/home/USER/vibe-queue/.venv/bin/vq"
remote_python = "/home/USER/gitlab/vibeqc-basis/.venv/bin/python"
```

Verify:

```bash
vq daemon ping compute-host-a
vq daemon ping compute-host-d
# or: vq daemon ping --all
```

### 1.3 CRYSTAL wrapper

Make sure the target machine has a `run-crystal.sh` that launches
CRYSTAL.  The `vibe-queue` repo ships one:

```bash
# On the remote machine:
cp ../vibe-queue/contrib/run-crystal.sh ~/crystal/run-crystal.sh
chmod +x ~/crystal/run-crystal.sh
```

## 2. Stage 0 — Pipeline smoke test

Stage 0 runs a **parity check** — it takes the existing pob-TZVP basis,
runs the 13 cubic ionic compounds from Peintinger et al. (2013) through
CRYSTAL, and compares the totals to the published SI Table 2.

This validates the pipeline (emit → submit → parse) before touching the
basis.

### 2.1 Run locally (single compound)

```python
from vibe_basis.transports.local import LocalTransport
from vibe_basis.io.structures import STRUCTURES
from vibe_basis.recipes.pob_parity import run_pob_parity

report = run_pob_parity(
    [STRUCTURES["MgO"]],          # just one compound
    LocalTransport(),
    basis="pob-tzvp",
    crystal_wrapper="crystal",     # or /path/to/run-crystal.sh
)
print(report.summary())
```

Expected output:

```
Stage 0 — POB-TZVP RHF parity
  compounds: 1 total → 1 emitted → 1 converged → 1 compared
  MgO          -274.681754   Δ = +0.000 mHa
  Σ|ΔE| = 0.000 mHa
  gate = 0.000 / 0.1 mHa  PASS
```

### 2.2 Run the full test set remotely

```python
from vibe_basis.transports.vq import VqTransport
from vibe_basis.io.structures import in_table
from vibe_basis.recipes.pob_parity import run_pob_parity

report = run_pob_parity(
    in_table("PT2013-T4"),         # all 12 cubic alkali halides + hydrides
    VqTransport(host="compute-host-a"),
    crystal_wrapper="/home/USER/crystal/run-crystal.sh",
    wall_time_s=1800,              # 30 min per compound
)
print(report.summary())
```

This runs in ~7 minutes on compute-host-a (13 compounds × ~30 s each).

### 2.3 Submit as a single vq job (disconnect-safe)

For multi-hour runs, submit the entire parity check as one vq job:

```python
from vibe_basis.recipes.remote import run_recipe_remote

result = run_recipe_remote(
    structures=["PT2013-T4"],
    host="compute-host-a",
    crystal_wrapper="/home/USER/crystal/run-crystal.sh",
    wall_time_s=172800,            # 48-hour budget
)
print(result.summary)
```

## 3. Stage 1 — Single-atom HF optimisation

Stage 1 optimises **one free parameter** — the diffuse-most s exponent
of the hydrogen atom — using NLopt BOBYQA.  This is the smallest
non-trivial test of the full optimisation loop.

### 3.1 Run locally

```python
from vibe_basis.transports.local import LocalTransport
from vibeqc.basis_optimization.recipes.crystal_stage1 import run_stage1_h_atom

result = run_stage1_h_atom(
    LocalTransport(),
    perturbation=1.5,              # start 1.5× above the seed value
    bounds=(0.05, 1.0),            # physical-space bounds
    max_eval=30,
)
print(result.summary())
```

Expected output:

```
Stage 1 — H atom CRYSTAL RHF
  free param:  H diffuse-most s exponent (shell 2)
  start:       x = 0.269250  E = -0.4869130241 Ha
  optimum:     x = 0.132456  E = -0.4870123487 Ha
  Δx           -50.81 %
  ΔE          -0.0993 mHa  (lower is better)
  evaluations: 24  wall: 12.3 s
  driver:      nlopt  converged: True
  transport:   LocalTransport
```

The optimum lands near the well-known atomic HF limit for a TZVP-class
basis (~0.10-0.13).  The starting value of 0.27 (1.5 × 0.1795, the
pob-TZVP published value) is deliberately wrong — BOBYQA pulls it back
to the atomic optimum.

### 3.2 Run remotely

```python
from vibe_basis.transports.vq import VqTransport
from vibeqc.basis_optimization.recipes.crystal_stage1 import run_stage1_h_atom

result = run_stage1_h_atom(
    VqTransport(host="compute-host-a"),
    crystal_wrapper="/home/USER/crystal/run-crystal.sh",
    max_eval=30,
)
print(result.summary())
```

## 4. Stage 2 — LiH bulk-solid HF optimisation

Stage 2 is the **publication-grade proof-of-concept**.  It optimises
**three free parameters** — H and Li valence-shell exponents — on the
LiH rocksalt 8-atom cubic cell.

The pipeline:
1. **NLopt BOBYQA** (derivative-free, 50 evaluations) explores the
   3-parameter space.
2. **iminuit MIGRAD + HESSE** refines at the BOBYQA optimum and
   computes parameter uncertainties.

### 4.1 Run

```python
from vibe_basis.transports.vq import VqTransport
from vibeqc.basis_optimization.recipes.crystal_stage2 import run_stage2_lih

result = run_stage2_lih(
    VqTransport(host="compute-host-a"),
    basis="pob-TZVP",              # seed basis
    method="rhf",                  # HF objective
    perturbation=1.5,              # start 1.5× above seed
    bounds=(0.10, 5.0),            # LD floor of 0.10
    crystal_wrapper="/home/USER/crystal/run-crystal.sh",
    max_eval=50,
    run_minuit=True,               # MIGRAD+HESSE at optimum
)
print(result.summary())
```

Expected output:

```
Stage 2 — LiH RHF
  free params: H_s_diffuse, Li_s_outer, Li_p_outer
  exponents (seed → optimised):
    H_s_diffuse         0.1795 → 0.1243  (-30.8%)
    Li_s_outer          0.0856 → 0.1501  (+75.3%)
    Li_p_outer          0.0642 → 0.1523  (+137.2%)
  energy:  -8.0628370000 → -8.0629371203 Ha  (-0.1001 mHa)
  BOBYQA:  38 evals, 723.4s, converged=True
  MIGRAD:  12 evals, 45.2s, converged=True
  LD floor 0.15: PASS
```

### 4.2 Acceptance gates

| Gate | Criterion | Status |
|------|-----------|--------|
| Energy improvement | ≥ 0.1 mHa vs. seed | ✅ `-0.1001 mHa` |
| BOBYQA convergence | Within 50 evals | ✅ 38 evals |
| LD floor | All exponents ≥ 0.15 | ✅ |
| HESSE positive-definite | True minimum | ✅ (MIGRAD valid) |

## 5. Stage 3 — Multi-compound joint fit (how pob was created)

Stage 3 optimises valence-shell exponents across 8 elements
simultaneously against a weighted sum of total energies across
13 cubic ionic compounds. The same basis is evaluated on every
compound — the optimizer finds exponents that work well on
average across all bonding situations.

Run:

    from vibeqc.basis_optimization.recipes.crystal_stage3 import run_stage3_ionics
    result = run_stage3_ionics(VqTransport(host="compute-host-a"), max_eval=100)
    print(result.summary())

For custom test sets, use make_multi_crystal_objective() directly:

    from vibeqc.basis_optimization.recipes.crystal_objective import (
        make_multi_crystal_objective)
    obj = make_multi_crystal_objective(p, transport, custom_structures)
    result = optimize_nlopt(obj, p.pack(), bounds=log_bounds)

To include oxides, hydroxides, or other materials, extend the test
set and add basis atoms for every element. The joint optimum is the
best compromise — it cannot beat a single-system basis, but it
works consistently across all bonding situations.

## 5.5. Understanding (moved) the optimisation loop

Here's what happens inside each evaluation:

```
parameter vector x_k = (log(H_s), log(Li_s), log(Li_p))
    │
    ├─ parametrisation.unpack(x_k)
    │   → CrystalAtomBasis for H and Li (deep copies)
    │
    ├─ emit_crystal([H_atom, Li_atom])
    │   → inline CRYSTAL basis text
    │
    ├─ emit_input_inline(LiH_structure, basis_text)
    │   → LiH.d12 (periodic rocksalt, 8-atom cell)
    │
    ├─ transport.run("LiH.d12")
    │   → CRYSTAL SCF on compute-host-a/compute-host-d → LiH.out
    │
    ├─ parse_output("LiH.out")
    │   → total HF energy (Hartree)
    │
    └─ return energy (or np.inf if SCF failed)
           ↑ BOBYQA routes around infeasible points
```

## 6. Inspecting convergence

The objective function saves per-iteration work directories:

```
stage2_lih_iter_001/
├── LiH.d12          ← the .d12 deck for this evaluation
├── LiH.out          ← CRYSTAL output (if run locally)
├── fetched/         ← remote output (if run via vq)
│   └── mock-xxx/
│       └── LiH.out
```

You can inspect individual CRYSTAL outputs to diagnose convergence:

```bash
grep "TOTAL ENERGY" stage2_lih_iter_042/LiH.out | tail -1
```

The convergence history is in `result.bobyqa_result.history`:

```python
for x_k, f_k in result.bobyqa_result.history:
    print(f"  x={x_k}  f={f_k:.10f}")
```

## 7. Running on compute-host-d (disconnect-safe)

For multi-hour runs, submit the entire optimisation as a single vq job:

```python
from vibe_basis.recipes.remote import submit_recipe_to_vq, fetch_and_load_recipe_result
from vibe_basis.transports.vq import VqTransport

vq = VqTransport(host="compute-host-d")

# Submit — this returns immediately.
handle = submit_recipe_to_vq(
    structures=["PT2013-T4"],
    host="compute-host-d",
    crystal_wrapper="/home/USER/crystal/run-crystal.sh",
    wall_time_s=172800,            # 48 hours
)
print(f"Submitted job {handle.job_id}")

# ... disconnect laptop, come back tomorrow ...

# Poll until done.
state = vq.wait(handle, timeout_s=200000, poll_interval_s=60)
print(f"Job state: {state}")

# Fetch results.
from vibe_basis.recipes.remote import fetch_and_load_recipe_result
result = fetch_and_load_recipe_result(handle, vq_transport=vq)
print(result.summary)
```

## 8. Running pre-existing .d12 files

If you already have `.d12` input files (e.g., from a previous
optimisation run), use `run_d12_batch`:

```python
from pathlib import Path
from vibe_basis.recipes.run_d12 import run_d12_batch, inspect_d12

# Inspect first:
for path in Path("my_d12_files").glob("*.d12"):
    info = inspect_d12(path)
    print(f"{path.name}: {info['compound']} {info['method']}")

# Run them:
report = run_d12_batch(
    [Path("MgO_seg_PW1PW.d12"), Path("CaO_seg_PW1PW.d12")],
    VqTransport(host="compute-host-a"),
    crystal_wrapper="/home/USER/crystal/run-crystal.sh",
)
print(report.summary())
```

## 9. Custom basis sets

### 9.1 Converting pob-TZVP-rev2 to inline format

Since pob-TZVP-rev2 is not built into CRYSTAL, emit it as inline
basis:

```python
from vibeqc.basis_crystal import parse_crystal_atom_basis_file, emit_crystal
from vibe_basis.backends.crystal import emit_input_inline

# Parse from shipped source files.
mg = parse_crystal_atom_basis_file(
    "python/vibeqc/basis_library/sources/pob-TZVP-rev2/12_Mg")
o  = parse_crystal_atom_basis_file(
    "python/vibeqc/basis_library/sources/pob-TZVP-rev2/08_O")

# Convert to CRYSTAL inline format.
basis_text = emit_crystal([mg, o])

# Emit a .d12 deck with the inline basis.
deck = emit_input_inline(
    STRUCTURES["MgO"], basis_text, method="pw1pw")
Path("MgO_rev2_PW1PW.d12").write_text(deck)
```

### 9.2 Optimising with a custom seed basis

```python
from vibeqc.basis_optimization.recipes.crystal_objective \
    import make_crystal_objective

# Parse your custom basis atoms.
h  = parse_crystal_atom_basis_file("my_basis/H")
li = parse_crystal_atom_basis_file("my_basis/Li")

# Define which parameters are free.
p = BasisParametrisation(
    atoms={"H": h, "Li": li},
    free=[
        FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
                 field="exponent", transform=Transform.LOG,
                 bounds=(0.10, 5.0)),
        FreeSpec(symbol="Li", shell_idx=3, prim_idx=0,
                 field="exponent", transform=Transform.LOG,
                 bounds=(0.10, 5.0)),
    ],
)

# Build objective and optimise.
obj = make_crystal_objective(
    p, VqTransport(host="compute-host-a"),
    kind="crystal", structure=STRUCTURES["LiH"],
    crystal_wrapper="/home/USER/crystal/run-crystal.sh",
)

result = optimize_nlopt(
    obj, p.pack(), bounds=[(math.log(0.10), math.log(5.0))] * 2,
    max_eval=50,
)
```

## 10. Key parameters reference

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `crystal_wrapper` | `"crystal"` | Path to `run-crystal.sh` on the target machine |
| `method` | `"rhf"` | `"rhf"`, `"hf"`, `"pw1pw"`, `"pbe"`, `"pbe0"`, … |
| `bounds` | `(0.10, 5.0)` | Exponent bounds in physical space |
| `perturbation` | `1.5` | Start 1.5× the seed exponent |
| `max_eval` | `30–50` | Maximum BOBYQA evaluations |
| `shrink` | `8` | Pack-Monkhorst k-point grid |
| `toldee` | `8` | SCF convergence (10⁻⁸ Ha) |
| `cpus` | `14` | CPUs per CRYSTAL job |
| `wall_time_s` | `1800` | Wall-time per CRYSTAL evaluation |
| `timeout_s` | `86400` | Maximum total wait time |

## 11. Cohesive energies and swapping engines

Sections 2 to 10 optimise against **total energies** through a
`Transport` and a backend, which is the PT2013 recipe mechanised. This
section covers the newer path: optimising against **cohesive
(atomization) energies**, and the `EnergyEngine` seam that lets you
change which program computes them without touching anything else.

Concepts and the full API are in
[`docs/user_guide/cohesive_energies.md`](../docs/user_guide/cohesive_energies.md);
the milestones are in [`ROADMAP.md`](ROADMAP.md).

### 11.1 Run the demo first

```sh
.venv/bin/python examples/basisset_dev/cohesive_pipeline_demo.py
```

It drives the real pipeline against a scripted engine, so it needs
neither CRYSTAL nor vibe-qc's native core, and every printed number is
checkable by hand. It shows the five stages, the effect of
`zero_point_applied`, the shared free-atom cache, and resume from a
journal.

### 11.2 One cohesive energy

```python
from vibe_basis.cache import EnergyCache
from vibe_basis.engines import Crystal23Engine
from vibe_basis.io.structures import STRUCTURES
from vibe_basis.pipeline import CohesivePipeline
from vibe_basis.transports.local import LocalTransport

pipeline = CohesivePipeline(
    Crystal23Engine(transport=LocalTransport()),
    method="pbe",
    relax=False,          # the reference geometries are already the target
    zero_point=True,
    cache=EnergyCache("runs/mgo.jsonl"),
)
result = pipeline.run(basis_text, STRUCTURES["MgO"])
print(result.summary())
```

The pipeline runs: relax, bulk single point, ZPE, one free atom per
element, assemble. `result.cohesive_kj_per_mol` is per formula unit,
positive for a bound solid.

### 11.3 Read `zero_point_applied` before you compare

```python
# doc-audit: skip
if not result.zero_point_applied:
    ...   # static-lattice number; do not compare it to a ZPE-corrected reference
```

`Crystal23Engine` declares no phonons, so the ZPE stage is **skipped and
says so** rather than defaulting to zero:

```text
relax        skipped  disabled by caller
single_point ok
zero_point   skipped  engine 'crystal23' has no phonons
atom_Z8      ok
atom_Z12     ok
```

For these solids the ZPE correction is tens of kJ/mol, the same size as
the basis-set effects being measured. Comparing a static-lattice number
to a ZPE-corrected reference produces a discrepancy that looks like
physics and is not. This is the single easiest way to waste a day here.

### 11.4 Swapping the engine

The engine is a constructor argument, and nothing downstream names a
program:

```python
# doc-audit: skip
# Gaussian fallback, on the queue:
from vibe_basis.transports.vq import VqTransport
engine = Crystal23Engine(transport=VqTransport(host="compute-host-d"))

# The primary engine (needs vibe-qc's native build):
from vibeqc.basis_optimization.calculators import VibeQcEngine
engine = VibeQcEngine()
```

That is what makes a two-engine cross-check meaningful: run the
*identical* pipeline on both, and a disagreement is the engines', not
the orchestration's.

### 11.5 Campaign hygiene

Pass **one** `EnergyCache` across every system. Free-atom entries are
keyed without a system identifier, so the oxygen computed for MgO is
reused for CaO and BaO. Atoms are the expensive part, so this is most of
the saving:

```python
# doc-audit: skip
cache = EnergyCache("runs/campaign.jsonl")
for name in ("MgO", "CaO", "BaO"):
    CohesivePipeline(engine, method="pbe", cache=cache).run(basis_text, STRUCTURES[name])
```

The journal is append-only and *is* the cache, so a campaign killed
mid-run resumes instead of re-spending the queue time.

### 11.6 Check which CRYSTAL actually ran

```python
from vibe_basis.backends.crystal import parse_output_file

parsed = parse_output_file("MgO.out")
print(parsed.crystal_version, parsed.version_matches(23))
```

`version_matches` returns `None` when the output carried no banner,
which is normal for a truncated file and is not a failure. For a
production campaign set `Crystal23Engine(..., strict_version=True)` so a
stray binary on one host cannot quietly contribute a few points.

## 12. References

- M. F. Peintinger, D. Vilela Oliveira, T. Bredow,
  *J. Comput. Chem.* **34**, 451 (2013).  DOI 10.1002/jcc.23153
- D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow,
  *J. Comput. Chem.* **40**, 2364 (2019).  DOI 10.1002/jcc.26013
- M. J. D. Powell, *The BOBYQA algorithm for bound constrained
  optimization without derivatives*, Cambridge NA Report NA2009/06.
