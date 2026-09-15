# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| cao_rocksalt | 1089.2 | 1096.4 | -7.2 |
| caf2_fluorite | 1589.6 | 1597.8 | -8.2 |
| kcl_rocksalt | 637.9 | 640.0 | -2.1 |

**vs VASP@900eV (n=3):** MD -5.8, MAD 5.8 kJ/mol.

## Cutoff convergence (per system)

### cao_rocksalt — r2scan, a=4.796 Å
_single-cutoff (no extrapolation); atom valence Ca:10, O:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.715471 | -0.300598 | 1089.2 | ✓ |
| **PW limit** | | | **1089.2** | |

### caf2_fluorite — r2scan, a=5.49 Å
_single-cutoff (no extrapolation); atom valence Ca:10, F:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.941682 | -0.336226 | 1589.6 | ✓ |
| **PW limit** | | | **1589.6** | |

### kcl_rocksalt — r2scan, a=6.3257 Å
_single-cutoff (no extrapolation); atom valence K:9, Cl:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.598672 | -0.355701 | 637.9 | ✓ |
| **PW limit** | | | **637.9** | |
