# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| alas_zincblende | 744.9 | 742.6 | +2.3 |
| aln_zincblende | 1114.7 | 1119.1 | -4.4 |
| gaas_zincblende | 640.3 | 650.0 | -9.7 |
| gan_beta_zincblende | 854.7 | 873.0 | -18.3 |

**vs VASP@900eV (n=4):** MD -7.5, MAD 8.7 kJ/mol.

## Cutoff convergence (per system)

### alas_zincblende — r2scan, a=5.649 Å
_single-cutoff (no extrapolation); atom valence Al:3, As:5_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.774013 | -0.490310 | 744.9 | ✓ |
| **PW limit** | | | **744.9** | |

### aln_zincblende — r2scan, a=4.371 Å
_single-cutoff (no extrapolation); atom valence Al:3, N:5_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.699733 | -0.275170 | 1114.7 | ✓ |
| **PW limit** | | | **1114.7** | |

### gaas_zincblende — r2scan, a=5.64 Å
_single-cutoff (no extrapolation); atom valence Ga:3, As:5_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.891566 | -0.647696 | 640.3 | ✓ |
| **PW limit** | | | **640.3** | |

### gan_beta_zincblende — r2scan, a=4.52 Å
_single-cutoff (no extrapolation); atom valence Ga:3, N:5_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.758085 | -0.432556 | 854.7 | ✓ |
| **PW limit** | | | **854.7** | |
