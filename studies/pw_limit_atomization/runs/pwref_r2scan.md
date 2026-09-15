# GPAW plane-wave-limit atomization energies — r2scan

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800,1000 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV

| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |
|--------|--------------:|---------:|--------------:|
| lif_rocksalt | 840.2 | 866.6 | -26.4 |
| nacl_rocksalt | 633.0 | 627.3 | +5.7 |
| naf_rocksalt | 754.0 | 749.6 | +4.4 |
| licl_rocksalt | 675.4 | 696.7 | -21.3 |
| c_diamond | 723.8 | 726.2 | -2.4 |
| bn_zincblende | 1314.1 | 1311.4 | +2.7 |

**vs VASP@900eV (n=6):** MD -6.2, MAD 10.5 kJ/mol.

## Cutoff convergence (per system)

### lif_rocksalt — r2scan, a=4.045 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Li:1, F:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.427902 | -0.108104 | 839.6 | ✓ |
| 1000 | -0.430574 | -0.110719 | 839.8 | ✓ |
| **PW limit** | | | **840.2** | |

### nacl_rocksalt — r2scan, a=5.569 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Na:7, Cl:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.497826 | -0.257022 | 632.2 | ✓ |
| 1000 | -0.497921 | -0.257029 | 632.5 | ✓ |
| **PW limit** | | | **633.0** | |

### naf_rocksalt — r2scan, a=4.633 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Na:7, F:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.456785 | -0.169732 | 753.7 | ✓ |
| 1000 | -0.459454 | -0.172359 | 753.8 | ✓ |
| **PW limit** | | | **754.0** | |

### licl_rocksalt — r2scan, a=5.15 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence Li:1, Cl:7_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.452305 | -0.195394 | 674.5 | ✓ |
| 1000 | -0.452399 | -0.195388 | 674.8 | ✓ |
| **PW limit** | | | **675.4** | |

### c_diamond — r2scan, a=3.553 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence C:4_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.723438 | -0.085901 | 724.2 | ✓ |
| 1000 | -0.724155 | -0.086296 | 724.1 | ✓ |
| **PW limit** | | | **723.8** | |

### bn_zincblende — r2scan, a=3.592 Å
_E_cut^(-3/2) extrapolation from 800/1000 eV; atom valence B:3, N:5_

| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |
|------------:|-------------:|-------------:|--------------:|:----:|
| 800 | -0.705852 | -0.204850 | 1315.4 | ✓ |
| 1000 | -0.706338 | -0.205480 | 1315.0 | ✓ |
| **PW limit** | | | **1314.1** | |
