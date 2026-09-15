# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800,1000 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| agcl_rocksalt | 523.4 | 526.0 | -2.6 |
| bao_rocksalt | 989.3 | 1001.5 | -12.2 |
| bas_rocksalt | 941.1 | 950.4 | -9.3 |
| zns_zincblende | 615.1 | 615.1 | +0.0 |

**vs VASP@900eV (n=4):** MD -6.0, MAD 6.0 kJ/mol.

## Cutoff convergence (per system)

### agcl_rocksalt — r2scan, a=5.528 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Ag:17, Cl:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.864977 | -0.665957 | 522.5 | ✓ |
| 1000 | -0.865095 | -0.665977 | 522.8 | ✓ |
| **PW limit** | | | **523.4** | |

### bao_rocksalt — r2scan, a=5.515 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Ba:10, O:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -1.056547 | -0.679692 | 989.4 | ✓ |
| 1000 | -1.056616 | -0.679781 | 989.4 | ✓ |
| **PW limit** | | | **989.3** | |

### bas_rocksalt — r2scan, a=6.364 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Ba:10, S:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -1.094468 | -0.735920 | 941.4 | ✓ |
| 1000 | -1.094538 | -0.736022 | 941.3 | ✓ |
| **PW limit** | | | **941.1** | |

### zns_zincblende — r2scan, a=5.45 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Zn:12, S:6_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.649721 | -0.415409 | 615.2 | ✓ |
| 1000 | -0.649784 | -0.415478 | 615.2 | ✓ |
| **PW limit** | | | **615.1** | |
