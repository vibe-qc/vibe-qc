# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800,1000 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| li2o_antifluorite | 1012.9 | 1149.7 | -136.8 |

**vs VASP@900eV (n=1):** MD -136.8, MAD 136.8 kJ/mol.

## Cutoff convergence (per system)

### li2o_antifluorite — r2scan, a=4.621 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Li:1, O:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.566263 | -0.180580 | 1012.6 | ✓ |
| 1000 | -0.566321 | -0.180605 | 1012.7 | ✓ |
| **PW limit** | | | **1012.9** | |
