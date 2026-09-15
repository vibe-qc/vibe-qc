# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| mgo_rocksalt | 1009.5 | 1012.0 | -2.5 |
| mgs_rocksalt | 772.1 | 775.0 | -2.9 |

**vs VASP@900eV (n=2):** MD -2.7, MAD 2.7 kJ/mol.

## Cutoff convergence (per system)

### mgo_rocksalt — r2scan, a=4.224 Å
_single-cutoff (no extrapolation); atom valence Mg:10, O:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.600224 | -0.215711 | 1009.5 | ✓ |
| **PW limit** | | | **1009.5** | |

### mgs_rocksalt — r2scan, a=5.26 Å
_single-cutoff (no extrapolation); atom valence Mg:10, S:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.566022 | -0.271938 | 772.1 | ✓ |
| **PW limit** | | | **772.1** | |
