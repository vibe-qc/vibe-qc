# MP2 method benchmark: vibe-qc vs ORCA on S22 + open-shell

Cross-code, cross-variant validation suite for vibe-qc's MP2 surface.
Every MP2 variant that vibe-qc exposes is exercised on a standard
benchmark set (Jurečka 2006's S22, plus a small open-shell complement)
and compared against ORCA 6.1.1 at *the same converger settings* so
the resulting energy differences reflect the implementations rather
than convergence-control choices.

## Separate queue installation

Live ORCA runs require the separately installed [vibe-queue repository](https://github.com/vibe-qc/vibe-queue)
and its `vq` command; follow [queue setup](../../../docs/user_guide/queue.md).
The S22 driver finds `contrib/run-orca.sh` in `~/gitlab/vibe-queue` on the
**execution host**. For another location, pass `VIBEQC_QUEUE_CHECKOUT` as an
absolute path in the job environment. The queue is not part of this checkout.

## What's compared

| vibe-qc API                            | ORCA keyword                | Notes                                                |
|----------------------------------------|-----------------------------|------------------------------------------------------|
| `run_mp2` (closed shell, canonical)    | `! MP2`                     | 4-index, O(N⁵); skipped above 14 atoms (see below)   |
| `run_mp2(density_fit=True, ...)`       | `! RI-MP2 cc-pVTZ/C`        | Resolution-of-identity MP2 (Weigend RIfit aux)       |
| `run_scs_mp2`                          | `! SCS-MP2 cc-pVTZ/C`       | Grimme 2003: c_os=6/5, c_ss=1/3                      |
| `run_sos_mp2`                          | `! SOS-MP2 cc-pVTZ/C`       | Jung 2004: c_os=1.3, c_ss=0                          |
| `run_b2plyp`                           | `! B2PLYP RIJK def2/JK cc-pVTZ/C` | Grimme 2006 double hybrid                      |
| `run_ump2` (open shell)                | `! MP2`  (UHF-driven)       | Spin-unrestricted analogue                           |
| `run_ump2(density_fit=True, ...)`      | `! RI-MP2 cc-pVTZ/C`        | RI-UMP2                                              |
| `run_scs_ump2`                         | `! SCS-MP2 cc-pVTZ/C`       | SCS coefficients on αβ + αα/ββ Grimme channels       |
| `run_sos_ump2`                         | `! SOS-MP2 cc-pVTZ/C`       | SOS                                                  |

## Same converger guarantee

Both codes are forced to plain Pulay DIIS with matched tolerances —
no second-order step, no trust-region augmented Hessian, no EDIIS
bootstrap. See [`s22/converger.py`](s22/converger.py) for the single
source of truth. Knobs:

* Accelerator: plain DIIS (vibe-qc `SCFAccelerator::DIIS`,
  ORCA `NoSOSCF NoTRAH`).
* Energy tol: 1e-8 Ha (`conv_tol_energy` / `SCFCONV8`).
* Grad tol: 1e-6 (`conv_tol_grad` / TolMaxP 1e-7).
* Max iter: 200.
* DIIS subspace: 8 (matched).
* Linear-dependence threshold: 1e-7 (both code defaults).
* Initial guess: vibe-qc SAD / ORCA PModel — both atom-density-based
  and behave similarly.

The two codes will *never* produce a bit-identical SCF trajectory
(different integral precision, different fine-detail step ordering),
but the converger *family* and *thresholds* match. The interesting
quantity is the converged-energy difference (target: <1e-6 Eh for
RI-MP2 on closed-shell systems).

## Layout

```
mp2_benchmarks/
├── README.md                    — this file
├── compare.py                   — comparator (local outputs → local REPORT.md)
├── parse_orca_mp2.py            — ORCA MP2 / B2PLYP output parser
├── s22/                         — Jurečka 2006 S22 dimers (closed shell)
│   ├── README.md
│   ├── converger.py             — SAME-CONVERGER profile (source of truth)
│   ├── extract_geometries.py    — pulls S22 geometries from Psi4 mirror
│   ├── generate_inputs.py       — emits vq + ORCA inputs for the matrix
│   ├── reference_energies.py    — S220 / S22A / S22B reference E_int
│   ├── geometries/              — extracted xyz files (dimer + monoA + monoB)
│   ├── inputs/vibeqc/<sys>/     — per-system vibe-qc input scripts
│   ├── inputs/orca/<sys>/       — per-system ORCA .inp files
│   ├── run_vibeqc_system.sh     — local driver
│   └── run_orca_system.sh       - vq-submitted driver (compute-reference)
└── open_shell/                  — UMP2 complement (S22 is all closed-shell)
    ├── README.md
    ├── geometries/              — OH, O2, CH3, OH-H2O xyz
    ├── generate_inputs.py       — same shape as S22's
    ├── inputs/vibeqc/<sys>/
    └── inputs/orca/<sys>/
```

The repository also retains 123 historical vibe-qc `.result` files as the
numeric record of the original all-electron wave.  They are intentionally not
regenerated or rewritten by the #140 convention change and therefore predate
the `n_frozen_core` provenance field.  Their paired checked-in input deck is
the protocol record and now pins the old all-electron choice explicitly.  A
new execution of the generated deck writes `n_frozen_core: 0` into its result
payload.

## Cost notes

* **Canonical MP2 (`! MP2`) is skipped above 14 atoms** — O(N⁵)
  on cc-pVTZ across the bigger S22 dimers (uracil dimer, adenine-
  thymine, benzene dimers) is hours per single-point. RI-MP2 +
  SCS-RI-MP2 + SOS-RI-MP2 + B2PLYP still run on every system.
* On a laptop-class machine the full vibe-qc matrix runs in 1–2 h
  wall (most cost is the big DD systems). The ORCA side runs on
  compute-reference via vq; each system's 15 ORCA calls are bundled into one
  vq job (see `s22/run_orca_system.sh`).
* If you need to slim the matrix further, edit
  `s22/generate_inputs.py::CANONICAL_MP2_MAX_ATOMS` (lower it) or
  the `VARIANTS` tuple.

## Reference values

* **Geometries**: Jurečka, Šponer, Černý, Hobza, *PCCP* **8**, 1985
  (2006). Pulled at build time from the Psi4 mirror (LGPL v3); we
  re-emit them as plain xyz files in this repo.
* **S22 interaction energies (CCSD(T)/CBS)**:
  * S220 — Jurečka 2006 (original).
  * S22A — Takatani, Hohenstein, Malagoli, Marshall, Sherrill,
    *JCP* **132**, 144104 (2010). First revision (frozen-core).
  * S22B — Marshall, Burns, Sherrill, *JCP* **135**, 194102 (2011).
    Current standard reference; comparator reports against this.
* **MP2 / cc-pVnZ basis-set limit reference**: Helgaker, Klopper,
  Koch, Olsen, *JCP* **106**, 9639 (1997) — H₂O canonical MP2 at
  cc-pV{D,T,Q,5}Z and the CBS extrapolation. Our cc-pVTZ MP2 on
  the S22 water dimer monomer should land within a few µHa of the
  ref-table value for that basis.
* **SCS-MP2 reference**: Grimme *JCP* **118**, 9095 (2003) for the
  scaling-coefficient origin + small-molecule reference.
* **B2PLYP**: Grimme *JCP* **124**, 034108 (2006).

## How to run

1. **Generate the input matrix** (one-shot, output deterministic):

   ```sh
   .venv/bin/python s22/extract_geometries.py    # pulls geometries
   .venv/bin/python s22/generate_inputs.py       # emits inputs/vibeqc/ + inputs/orca/
   .venv/bin/python open_shell/generate_inputs.py
   ```

2. **Run vibe-qc locally** (per system or all):

   ```sh
   bash s22/run_vibeqc_system.sh s22/inputs/vibeqc/s22-02-water-dimer
   # or loop over all systems
   for d in s22/inputs/vibeqc/*/; do
     bash s22/run_vibeqc_system.sh "$d"
   done
   ```

3. **Submit ORCA to compute-reference via vq** (one vq job per system,
   sequential ORCA calls inside the job):

   ```sh
   cp s22/run_orca_system.sh s22/inputs/orca/<system>/
   vq submit compute-reference -d s22/inputs/orca/<system> \
       --cpus 4 --wall-time-seconds 1800 \
       --name "mp2bench-<system>" \
       -- bash run_orca_system.sh
   ```

4. **Render reports**:

   ```sh
   .venv/bin/python compare.py s22
   .venv/bin/python compare.py open_shell
   ```

   This writes local `s22/REPORT.md` + `open_shell/REPORT.md` files
   with the per-system comparison tables, interaction energies (S22
   only), and aggregate ΔE_corr statistics per variant. Treat those
   reports like the `.log` outputs: keep them with the run record, not
   in the source commit.

## Acceptance bar

For a clean run on a freshly converged SCF on both sides, we expect
ΔE_corr in:

| Variant    | Expected |ΔE_corr| vibe-qc vs ORCA |
|------------|----------------------------------|
| MP2        | < 1e-7 Eh (4-index, same aux)    |
| RI-MP2     | < 1e-7 Eh (same Gaussians)       |
| SCS-MP2    | < 1e-7 Eh (same coefficients)    |
| SOS-MP2    | < 1e-7 Eh                        |
| B2PLYP     | < 1e-5 Eh (different XC grids)   |
| UMP2 family| < 1e-6 Eh (UHF rotation noise)   |

Larger ΔE flags a real implementation difference — file a regression.
