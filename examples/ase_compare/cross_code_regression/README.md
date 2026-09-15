# Cross-code comparison suite — vibe-qc bug-fixing handover

Pinned reproducers for the dev chat to drive bug-hunting against
**PySCF** (always available — also a hard runtime dependency) and
**ORCA** (optional, for an independent third reference). All
scripts are self-contained, bring their own systems, write a CSV
summary, and (where applicable) leave per-config `.dump` / `.perf`
artefacts on failure for post-mortem inspection.

## Suite

### Molecular (works in both v0.6.x and v0.7+)

| Script | What it tests |
|---|---|
| [`compare_h2o_hf_basis_scan.py`](compare_h2o_hf_basis_scan.py) | H₂O, RHF, 5 bases (sto-3g → def2-TZVP). All three codes should agree on `E` to ~µHa per row. |
| [`compare_h2o_dft_basis_scan.py`](compare_h2o_dft_basis_scan.py) | H₂O, RKS-PBE, 5 bases. Expect ~1 mHa agreement at small basis, ~5 mHa at TZ (XC-grid noise). |
| [`compare_h2o_mp2_basis_scan.py`](compare_h2o_mp2_basis_scan.py) | H₂O, MP2, 4 bases. Reports SCF + correlation + total separately so the bug surface is partitioned. Bypasses ASE — calls `vq.run_mp2`, `pyscf.mp.MP2`, and ORCA's `! MP2` directly, with an explicit all-electron recipe (`n_frozen_core=0`, `frozen=0`, `NoFrozenCore`) on all three backends. |

### Periodic (the active bug surface)

| Script | What it tests |
|---|---|
| [`compare_pbc_easy_systems.py`](compare_pbc_easy_systems.py) | Solid Ne, solid Ar, diamond C — closed-shell, wide-gap, well-localised. RKS-LDA / sto-3g / Γ-only on both vibe-qc and `pyscf.pbc.dft.RKS`. If any row disagrees by more than a few mHa **after correcting for the Madelung shift** (or before, if the Madelung-leak fix has already landed), it's a system-specific issue. |
| [`compare_pbc_molecular_limit.py`](compare_pbc_molecular_limit.py) | **The Madelung-leak reproducer.** H₂ in 30/50/100 bohr cubic vacuum boxes; runs molecular RHF (vibe-qc) + periodic RHF (vibe-qc) + periodic RHF (PySCF.pbc); reports the differences against the predicted `α_M·(Q_n²+Q_e²)/(2L)` leak. If `vq pbc - vq mol` matches the prediction but `pyscf pbc - vq mol` doesn't, the bug is squarely in vibe-qc's periodic Coulomb pipeline — exactly what we expect. |

## How to run

From the repo root, with vibe-qc + PySCF + (optionally) ORCA installed in your venv:

```sh
# Molecular suite (~5-10 min total)
.venv/bin/python examples/ase_compare/cross_code_regression/compare_h2o_hf_basis_scan.py
.venv/bin/python examples/ase_compare/cross_code_regression/compare_h2o_dft_basis_scan.py
.venv/bin/python examples/ase_compare/cross_code_regression/compare_h2o_mp2_basis_scan.py

# Periodic suite (~5-15 min total; depends on Schwarz screening + machine)
.venv/bin/python examples/ase_compare/cross_code_regression/compare_pbc_easy_systems.py
.venv/bin/python examples/ase_compare/cross_code_regression/compare_pbc_molecular_limit.py
```

Each script writes a CSV next to itself + (for the periodic ones)
per-config `.dump` / `.perf` / `.system` files under
`output/<scriptname>/`.

ORCA is auto-detected via `$ORCA_COMMAND`, `$ASE_ORCA_COMMAND`, or
`shutil.which("orca")` (in that order). If absent, the ORCA column
shows `—` and the comparison continues with vibe-qc + PySCF.

## What "agreement" means in each setting

| Method | Expected agreement |
|---|---|
| HF / sto-3g, 6-31G* | ~µHa across all three codes — same SCF, same integrals |
| HF / cc-pVTZ, def2-TZVP | ~µHa — basis is identical, integrals are libint-grade everywhere |
| RKS-PBE / sto-3g, 6-31G* | ~1 mHa — XC-grid resolution starts to matter |
| RKS-PBE / cc-pVTZ | ~5 mHa — ditto, larger basis pulls more grid points into focus |
| MP2 / any basis | ~µHa correlation-energy match; identical SCF + identical bases ⇒ identical AO→MO transform ⇒ identical amplitude solve |
| PBC RKS-LDA / sto-3g / Γ | ✗ disagreement of order 0.1-1 Ha PER CELL on small cells (this is the Madelung leak — see molecular-limit script for the magnitude predictor) |

Anything outside these tolerances is a candidate bug. The CSV
columns are stable so dashboards / regression-test fixtures can
ingest them without screen-scraping.

## Fingerprinting / regression tests

The CSVs each carry enough info that any of the rows can be
turned into a regression test fixture:

```python
import csv, math
with open("output-h2o-hf-basis-scan.csv") as fh:
    for row in csv.DictReader(fh):
        if row["basis"] == "cc-pvdz" and row["label"].startswith("vibe-qc"):
            assert math.isclose(float(row["energy_eV"]), -2068.7735, abs_tol=1e-3)
```

When the dev-side fix lands and the periodic numbers come into
agreement, the same CSV row becomes the regression fixture.

## Paired tools elsewhere in the repo

  - [`examples/debug/`](../../debug/) — 9-script diagnostic kit
    for SCF debugging (convergence-aid sweep, trajectory recorder,
    SAD-vs-HCORE A/B, molecular-limit check, vs-PySCF head-to-head
    on Ne/Ar/diamond, perf-knob scaling).
  - [`examples/debug/scf_iteration_recorder.py`](../../debug/scf_iteration_recorder.py)
    — point at any system to capture + plot the SCF trajectory.
    Useful when the comparison shows a convergence (rather than
    energy) divergence between vibe-qc and PySCF.
