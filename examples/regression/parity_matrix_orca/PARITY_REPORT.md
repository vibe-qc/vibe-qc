# ORCA parity matrix — vibe-qc vs ORCA 6.1.1

_Generated 2026-05-20T20:42:23+00:00 by `examples/regression/parity_matrix_orca/run_parity.py`._

Per-intermediate energy decomposition, vibe-qc vs ORCA. ORCA's standard output does not split the Coulomb and exchange energies, so the Fock-build bucket is the combined `e_coulomb_plus_exchange` (= ORCA `Two Electron Energy - E(XC)`; vibe-qc `e_coulomb + e_exchange`).

**6 pass, 0 fail, 0 not evaluated** of 6 cells.

## H2O__def2-svp__RMP2 — pass

ORCA `! MP2 def2-SVP NoFrozenCore VeryTightSCF Bohrs` (v6.1.1, jobid `ee090c6747bd`)

| piece | vibe-qc | ORCA | |Δ| | tol | ok |
|---|---:|---:|---:|---:|:--:|
| e_hf | -75.95475970 | -75.95475970 | 9.66e-11 | 1e-09 | ✓ |
| e_corr | -0.20662958 | -0.20662953 | 4.43e-08 | 2e-07 | ✓ |
| e_total | -76.16138928 | -76.16138924 | 4.42e-08 | 2e-07 | ✓ |
| mo (max, n=0) | | | 0.00e+00 | 0e+00 | ✓ |

## H2O__cc-pvdz__RMP2 — pass

ORCA `! MP2 cc-pVDZ NoFrozenCore VeryTightSCF Bohrs` (v6.1.1, jobid `c69e3824819a`)

| piece | vibe-qc | ORCA | |Δ| | tol | ok |
|---|---:|---:|---:|---:|:--:|
| e_hf | -76.02082648 | -76.02082648 | 1.07e-10 | 1e-09 | ✓ |
| e_corr | -0.20714671 | -0.20714666 | 4.49e-08 | 2e-07 | ✓ |
| e_total | -76.22797319 | -76.22797315 | 4.48e-08 | 2e-07 | ✓ |
| mo (max, n=0) | | | 0.00e+00 | 0e+00 | ✓ |

## H2O__cc-pvtz__RMP2__RI — pass

ORCA `! RI-MP2 cc-pVTZ/C cc-pVTZ NoFrozenCore VeryTightSCF Bohrs` (v6.1.1, jobid `c6374a08d6fd`)

| piece | vibe-qc | ORCA | |Δ| | tol | ok |
|---|---:|---:|---:|---:|:--:|
| e_hf | -76.05035584 | -76.05035584 | 1.04e-10 | 1e-09 | ✓ |
| e_corr | -0.27814660 | -0.27814659 | 1.31e-08 | 2e-07 | ✓ |
| e_total | -76.32850244 | -76.32850242 | 1.38e-08 | 2e-07 | ✓ |
| mo (max, n=0) | | | 0.00e+00 | 0e+00 | ✓ |

## H2CO__def2-svp__RMP2 — pass

ORCA `! MP2 def2-SVP NoFrozenCore VeryTightSCF Bohrs` (v6.1.1, jobid `2acbfe8927f3`)

| piece | vibe-qc | ORCA | |Δ| | tol | ok |
|---|---:|---:|---:|---:|:--:|
| e_hf | -113.77831142 | -113.77831142 | 1.26e-10 | 1e-09 | ✓ |
| e_corr | -0.32165082 | -0.32165077 | 5.10e-08 | 2e-07 | ✓ |
| e_total | -114.09996224 | -114.09996218 | 5.11e-08 | 2e-07 | ✓ |
| mo (max, n=0) | | | 0.00e+00 | 0e+00 | ✓ |

## OH__def2-svp__UMP2 — pass

ORCA `! MP2 def2-SVP NoFrozenCore VeryTightSCF Bohrs` (v6.1.1, jobid `d42503c81c3b`)

| piece | vibe-qc | ORCA | |Δ| | tol | ok |
|---|---:|---:|---:|---:|:--:|
| e_hf | -75.32510009 | -75.32510009 | 9.94e-11 | 1e-09 | ✓ |
| e_corr | -0.15061463 | -0.15061465 | 1.35e-08 | 2e-07 | ✓ |
| e_total | -75.47571472 | -75.47571474 | 1.35e-08 | 2e-07 | ✓ |
| mo (max, n=0) | | | 0.00e+00 | 0e+00 | ✓ |

## OH__cc-pvtz__UMP2__RI — pass

ORCA `! RI-MP2 cc-pVTZ/C cc-pVTZ NoFrozenCore VeryTightSCF Bohrs` (v6.1.1, jobid `9a6bac493bb0`)

| piece | vibe-qc | ORCA | |Δ| | tol | ok |
|---|---:|---:|---:|---:|:--:|
| e_hf | -75.41925054 | -75.41925054 | 1.08e-10 | 1e-09 | ✓ |
| e_corr | -0.21198835 | -0.21198839 | 4.38e-08 | 2e-07 | ✓ |
| e_total | -75.63123889 | -75.63123893 | 4.35e-08 | 2e-07 | ✓ |
| mo (max, n=0) | | | 0.00e+00 | 0e+00 | ✓ |

