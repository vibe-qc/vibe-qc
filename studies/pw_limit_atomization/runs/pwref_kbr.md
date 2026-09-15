# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| kbr_rocksalt | 593.6 | 620.8 | -27.2 |

**vs VASP@900eV (n=1):** MD -27.2, MAD 27.2 kJ/mol.

## Cutoff convergence (per system)

### kbr_rocksalt — r2scan, a=6.5888 Å
_single-cutoff (no extrapolation); atom valence K:9, Br:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.783868 | -0.557771 | 593.6 | ✓ |
| **PW limit** | | | **593.6** | |
