# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| lih_rocksalt | 464.3 | 468.3 | -4.0 |

**vs VASP@900eV (n=1):** MD -4.0, MAD 4.0 kJ/mol.

## Cutoff convergence (per system)

### lih_rocksalt — r2scan, a=3.979 Å
_single-cutoff (no extrapolation); atom valence Li:1, H:1_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.246259 | -0.069399 | 464.3 | ✓ |
| **PW limit** | | | **464.3** | |
