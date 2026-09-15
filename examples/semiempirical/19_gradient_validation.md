# Semiempirical gradient FD validation

**Date:** 2026-06-06 01:59 UTC
**FD step:** h = 0.001 bohr

## Results

| Molecule | Method | RMSD (Ha/bohr) | Max |g| (Ha/bohr) | Max err | Time |
|---|---:|---:|---:|---:|
| CH4      | DFTB0      | 2.12e-08 | 0.0126 | 2.38e-08 | 0.00s |
| CH4      | GFN2-xTB   | 0.00e+00 | 1.7359 | 0.00e+00 | 0.12s |
| CH4      | PM6        | 0.00e+00 | 2.5382 | 0.00e+00 | 0.04s |
| CH4      | SCC-DFTB   | 1.82e-05 | 0.0281 | 2.03e-05 | 0.01s |
| CO2      | DFTB0      | 3.59e-08 | 0.0103 | 7.62e-08 | 0.00s |
| CO2      | GFN2-xTB   | 0.00e+00 | 9.2163 | 0.00e+00 | 0.06s |
| CO2      | PM6        | 0.00e+00 | 5.6298 | 0.00e+00 | 0.01s |
| CO2      | SCC-DFTB   | 3.70e-04 | 0.0099 | 7.85e-04 | 0.01s |
| H2       | DFTB0      | 6.71e-06 | 0.6697 | 1.16e-05 | 0.00s |
| H2       | GFN2-xTB   | 0.00e+00 | 0.0169 | 0.00e+00 | 0.02s |
| H2       | PM6        | 0.00e+00 | 0.0736 | 0.00e+00 | 0.00s |
| H2       | SCC-DFTB   | 6.71e-06 | 0.6697 | 1.16e-05 | 0.00s |
| H2O      | DFTB0      | 1.64e-07 | 0.0323 | 2.93e-07 | 0.00s |
| H2O      | GFN2-xTB   | 0.00e+00 | 2.0520 | 0.00e+00 | 0.05s |
| H2O      | PM6        | 0.00e+00 | 3.9136 | 0.00e+00 | 0.01s |
| H2O      | SCC-DFTB   | 2.93e-01 | 0.0765 | 5.39e-01 | 0.01s |

## Summary
- **DFTB0**: 4 molecules, avg RMSD = 1.73e-06, avg max err = 3.00e-06 Ha/bohr
- **GFN2-xTB**: 4 molecules, avg RMSD = 0.00e+00, avg max err = 0.00e+00 Ha/bohr
- **PM6**: 4 molecules, avg RMSD = 0.00e+00, avg max err = 0.00e+00 Ha/bohr
- **SCC-DFTB**: 4 molecules, avg RMSD = 7.35e-02, avg max err = 1.35e-01 Ha/bohr
