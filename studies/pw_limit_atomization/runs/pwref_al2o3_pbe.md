# GPAW plane-wave-limit atomization energies — pbe

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| al2o3_corundum | 2978.3 | 3108.9 | -130.6 |

**vs VASP@900eV (n=1):** MD -130.6, MAD 130.6 kJ/mol.

## Cutoff convergence (per system)

### al2o3_corundum — pbe, a=4.7718 Å
_single-cutoff (no extrapolation); atom valence Al:3, O:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -8.326276 | -0.253334 | 2978.3 | ✓ |
| **PW limit** | | | **2978.3** | |
