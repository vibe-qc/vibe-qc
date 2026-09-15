# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| li2o_antifluorite | 1136.4 | 1149.7 | -13.3 |
| cdse_zincblende | 495.5 | 489.8 | +5.7 |
| inas_zincblende | 587.0 | 579.9 | +7.1 |
| cscl_cscl | 620.7 | 625.7 | -5.0 |

**vs VASP@900eV (n=4):** MD -1.4, MAD 7.8 kJ/mol.

## Cutoff convergence (per system)

### li2o_antifluorite — r2scan, a=4.621 Å
_single-cutoff (no extrapolation); atom valence Li:1, O:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.613404 | -0.180580 | 1136.4 | ✓ |
| **PW limit** | | | **1136.4** | |

### cdse_zincblende — r2scan, a=6.042 Å
_single-cutoff (no extrapolation); atom valence Cd:12, Se:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -1.065566 | -0.876856 | 495.5 | ✓ |
| **PW limit** | | | **495.5** | |

### inas_zincblende — r2scan, a=6.047 Å
_single-cutoff (no extrapolation); atom valence In:13, As:5_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -1.137655 | -0.914093 | 587.0 | ✓ |
| **PW limit** | | | **587.0** | |

### cscl_cscl — r2scan, a=4.1024 Å
_single-cutoff (no extrapolation); atom valence Cs:9, Cl:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.978878 | -0.742482 | 620.7 | ✓ |
| **PW limit** | | | **620.7** | |
