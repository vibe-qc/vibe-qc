# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| sic_zincblende | 1234.6 | 1242.5 | -7.9 |
| alp_zincblende | 815.8 | 820.2 | -4.4 |
| gap_zincblende | 698.1 | 717.9 | -19.8 |

**vs VASP@900eV (n=3):** MD -10.7, MAD 10.7 kJ/mol.

## Cutoff convergence (per system)

### sic_zincblende — r2scan, a=4.35 Å
_single-cutoff (no extrapolation); atom valence Si:4, C:4_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.703207 | -0.232973 | 1234.6 | ✓ |
| **PW limit** | | | **1234.6** | |

### alp_zincblende — r2scan, a=5.45 Å
_single-cutoff (no extrapolation); atom valence Al:3, P:5_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.623523 | -0.312802 | 815.8 | ✓ |
| **PW limit** | | | **815.8** | |

### gap_zincblende — r2scan, a=5.46 Å
_single-cutoff (no extrapolation); atom valence Ga:3, P:5_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.736087 | -0.470188 | 698.1 | ✓ |
| **PW limit** | | | **698.1** | |
