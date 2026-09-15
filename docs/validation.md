# Validation and cross-code parity

vibe-qc implements every method it ships (CLAUDE.md sec 10); established codes
are used only as **out-of-process** reference oracles (parsed via the
`examples/regression/runner_<program>.py` subprocess pattern, never imported
into the runtime). This page consolidates the validation surface that the
release paper draws on. Numbers are produced by the in-repo regression suite;
per-system deltas are re-confirmed against the tagged release build.

## Methodology

* **Same-family oracles.** Each route is compared against the established code in
  its own method family, so a discrepancy isolates a real difference rather than
  a cross-family convention mismatch: periodic Gaussian-AO HF/DFT vs **CRYSTAL**
  (BIPOLE) and **PySCF** (GDF); plane-wave-grid DFT vs **CP2K / GPAW** (GPW/GAPW);
  molecular HF/DFT/MP2/CC vs **PySCF**; correlated/CC accuracy cross-checked vs
  **ORCA**; multireference vs **OpenMolcas**.
* **Matched settings.** Cross-code comparisons fix the convergence-relevant
  knobs (k-mesh / SHRINK, integral tolerances, smearing, mixing) so the residual
  reflects the method, not the protocol.

## Molecular methods (vs PySCF / ORCA)

* **HF / DFT / MP2 / gradients / Hessian** -- machine precision vs PySCF
  (see `docs/features.md`).
* **Canonical CCSD(T)** (RHF / UHF / ROHF references) -- validated to uHa vs
  PySCF `cc.CCSD/CCSD(T)` and `cc.UCCSD/UCCSD(T)`, with ORCA 6.1 cross-checks.
* **DLPNO-CCSD(T)** -- the full-domain limit reduces exactly to canonical
  CCSD(T). Retained ORCA/canonical accuracy benchmarks use explicit
  pre-#140/#448 all-electron thresholds and do not validate the current
  published-core/NormalPNO default. Always report the frozen-core convention
  and every DLPNO threshold with an accuracy number.
* **DLPNO-MP2** (closed- and open-shell UMP2) -- the explicit all-electron,
  zero-threshold limit reproduces canonical RI-MP2 to <= 1 uHa. Archived
  recovery percentages use pinned pre-#140/#448 cutoffs and do not validate
  the current named default.
* **Multireference** (CASCI / CASSCF / NEVPT2 / IC-CASPT2) -- validated vs PySCF
  (`mcscf`, `mrpt`) and OpenMolcas where PySCF has no reference.

## Periodic HF / DFT (vs CRYSTAL / PySCF / CP2K)

Periodic Hartree-Fock and Kohn-Sham DFT are vibe-qc's distinctive contribution:
three independent in-house routes with route-specific validation. Periodic
post-HF correlation is out of scope here and has a separate study.

* **BIPOLE.** Exact erfc-screened real-space J/K plus reciprocal AO-pair-FT
  long-range J in a shared Ewald gauge. CRYSTAL runs provide out-of-process
  component and convergence references through
  `examples/regression/crystal_parity/`, but the supported vibe-qc route does
  not execute CRYSTAL's bipolar quartet replacement. Matched k meshes and
  input tolerances alone are therefore not a same-algorithm parity proof.
* **GDF (PySCF-family).** Gaussian density fitting, the default `run_periodic_job`
  backend; uHa parity vs PySCF `KRHF.density_fit` / `KRKS.density_fit` at matched
  k-mesh.
* **GPW / GAPW (CP2K / GPAW-family).** Gaussian-and-plane-waves; validated against
  CP2K / GPAW reference runs.

### CRYSTAL14 reference targets (STO-3G, SHRINK 8 8, TOLDEE 8; per formula unit)

These are the sealed CRYSTAL14 references the BIPOLE/GDF periodic-HF parity tests
match (`examples/regression/crystal_parity/baseline_sto3g/PARITY_TABLE.md`):

| System     | E_total/FU (Ha) |
|------------|-----------------|
| LiH        |    -7.93816810  |
| MgO        |  -271.21814708  |
| NaCl       |  -614.65306571  |
| LiF        |  -105.63898757  |
| C-diamond  |   -74.87695367  |
| Si-diamond |  -571.31994169  |

## QVF visualization fixtures

QVF writer and viewer compatibility is checked with the same static artifacts
used by the documentation examples. The curated
[`chi-ccm-b-qvf`](example_outputs.md#static-reference-bundles) bundle carries
two periodic chi-CCM-B archives: a 3D vacuum-padded H-chain finite BvK torus and a 3D H2-pair
torus. Each bundle entry includes the full input, verbose `.out` log,
sanitized `.system` manifest, validated `.qvf` archive, and headless vibe-view
captures. These fixtures exercise periodic cell scale, density/orbital grid
tiling, wrap-to-cell-centre display, and the `x_ccm.wannier_centers` overlay.

## Where the numbers live

The validation is executed, not asserted by hand: the molecular cross-checks are
pinned in the `tests/` suite (PySCF/ORCA references with inline provenance), and
the periodic parity is driven by `examples/regression/crystal_parity/` and
`examples/regression/bipole_parity/`. The full-suite gate
(`scripts/test_gate/`) keeps these green on every release.
