# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| beo_wurtzite | 1200.6 | 1210.2 | -9.6 |
| zno_wurtzite | 730.0 | 728.2 | +1.8 |

**vs VASP@900eV (n=2):** MD -3.9, MAD 5.7 kJ/mol.

## Cutoff convergence (per system)

### beo_wurtzite — r2scan, a=2.714 Å
_single-cutoff (no extrapolation); atom valence Be:2, O:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -1.200053 | -0.142750 | 1200.6 | ✓ |
| **PW limit** | | | **1200.6** | |

### zno_wurtzite — r2scan, a=3.25 Å
_single-cutoff (no extrapolation); atom valence Zn:12, O:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -1.274435 | -0.359181 | 730.0 | ✓ |
| **PW limit** | | | **730.0** | |
