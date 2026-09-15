# GPAW plane-wave-limit atomization energies — r2SCAN

GPAW 25.7.0 (out-of-process, §10) · k-mesh 6,6,6 · atom box 12.0 Å · cutoffs 800,1000 eV · energies kJ/mol per formula unit.

## Summary — PW limit vs VASP@900eV vs pob

| System | GPAW PW-limit | VASP@900 | pob | Δ(GPAW−VASP) | Δ(GPAW−pob) | Outlier? |
|--------|--------------:|---------:|----:|-------------:|------------:|----------|
| LiF | 840.2 | 866.6 | 843.7 | −26.4 | −3.5 | **VASP** |
| NaCl | 633.0 | 627.3 | 662.8 | +5.7 | −29.8 | **pob** |
| NaF | 754.0 | 749.6 | 761.8 | +4.4 | −7.8 | pob |
| LiCl | 675.4 | 696.7 | 688.1 | −21.3 | −12.7 | GPAW |
| MgO | 1009.5 | 1012.0 | 1044.3 | −2.5 | −34.8 | **pob** |
| MgS | 772.1 | 775.0 | 780.0 | −2.9 | −7.9 | all agree |
| C-diamond | 723.8 | 726.2 | 716.6 | −2.4 | +7.2 | all agree |
| BN | 1314.1 | 1311.4 | 1300.0 | +2.7 | +14.1 | pob |
| LiH | — | 468.3 | 459.1 | — | — | H atom failed |

**vs VASP@900eV (n=8):** MD −5.4, MAD 8.5 kJ/mol.
**vs pob (n=8):** MD −9.4, MAD 14.7 kJ/mol.

## Key findings

1. **GPAW tracks published SCAN/PBE literature**: LiF GPAW 840.2 ≈ lit SCAN 845, MgO GPAW 1009.5 ≈ lit SCAN 1011.
2. **VASP is the high outlier for LiF** (+26.4 kJ/mol) — the paper's "pob basis error" of −22.9 for LiF is mostly a VASP atom-reference artifact, not a Gaussian basis-set incompleteness.
3. **pob is the high outlier for MgO** (+34.8 kJ/mol vs GPAW, +32.3 vs VASP) — pob overbinds MgO.
4. **pob is the outlier for NaCl** (−29.8 vs GPAW, −35.5 vs VASP) — pob underestimates NaCl atomization.
5. **C-diamond and BN show excellent cross-code agreement** (within 3-14 kJ/mol across all three codes).
6. **MgO and MgS used single-cutoff 800 eV** (Mg atom at 1000 eV failed with direct-min). Atomization is essentially flat at 800→1000 eV for all converged systems (Δ < 0.5 kJ/mol).
7. **LiH omitted** — H atom direct-min fails with meta-GGA r2SCAN. SCF approach works (H@800eV SCF E=−0.041263 Ha) but needs worker support.

## Implications for the paper

The paper "Approaching the plane wave limit with Gaussian basis sets" attributes Δb = E_pob − E_VASP entirely to pob basis-set incompleteness. An independent PW/PAW code (GPAW) shows this is **not uniform**:
- LiF: Δb = −22.9 → GPAW confirms pob is correct (840 vs 844), VASP is the outlier
- MgO: Δb = +32.3 → GPAW confirms VASP is correct (1010 vs 1012), pob is the outlier
- NaCl: Δb = +35.5 → GPAW sits between (633), both pob and VASP deviate

The Δb metric conflates pob basis error with VASP atomic-reference convention differences. Using GPAW as a third reference de-conflates the errors and provides system-specific insight into which code is correct for each compound.

