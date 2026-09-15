# Semiempirical benchmark suite

**Date:** 2026-06-06 06:23 UTC
**Molecules:** H2, H2O, CH4, NH3, CO2, C2H4
**Methods:** DFTB0, SCC-DFTB, GFN2-xTB, PM6

## Results

| Molecule | Method | Conv | Iters | Energy (Ha) | Wall time |
|---|---:|---:|---:|---:|
| C2H4     | DFTB0      | ✅    |   1 |    -4.545175 | 0.0001s |
| C2H4     | GFN2-xTB   | ✅    |  12 |   -24.943408 | 0.0024s |
| C2H4     | PM6        | ✅    |  35 |    13.208770 | 0.0007s |
| C2H4     | SCC-DFTB   | ✅    |   1 |   -11.815258 | 0.0007s |
| CH4      | DFTB0      | ✅    |   1 |    -2.730849 | 0.0001s |
| CH4      | GFN2-xTB   | ✅    |  15 |   -22.466237 | 0.0019s |
| CH4      | PM6        | ✅    |  63 |     1.898650 | 0.0008s |
| CH4      | SCC-DFTB   | ✅    |   1 |    -2.212569 | 0.0003s |
| CO2      | DFTB0      | ✅    |   1 |    -9.331728 | 0.0002s |
| CO2      | GFN2-xTB   | ✅    |  11 |   -65.977742 | 0.0015s |
| CO2      | PM6        | ✅    |  24 |    40.133688 | 0.0002s |
| CO2      | SCC-DFTB   | ✅    |   1 |    -9.331881 | 0.0004s |
| H2       | DFTB0      | ✅    |   1 |    -0.376699 | 0.0012s |
| H2       | GFN2-xTB   | ✅    |   1 |    -1.034041 | 0.0021s |
| H2       | PM6        | ✅    |  14 |    -0.629177 | 0.0006s |
| H2       | SCC-DFTB   | ✅    |   1 |    -0.376699 | 0.0001s |
| H2O      | DFTB0      | ✅    |   1 |    -4.203900 | 0.0002s |
| H2O      | GFN2-xTB   | ✅    |  17 |   -20.138729 | 0.0014s |
| H2O      | PM6        | ✅    |  48 |    -4.497818 | 0.0003s |
| H2O      | SCC-DFTB   | ✅    |   1 |    -5.896637 | 0.0003s |
| NH3      | DFTB0      | ✅    |   1 |    -3.523123 | 0.0001s |
| NH3      | GFN2-xTB   | ✅    |  80 |   -57.554255 | 0.0087s |
| NH3      | PM6        | ✅    |  48 |     0.323353 | 0.0005s |
| NH3      | SCC-DFTB   | ✅    |   1 |    -6.102214 | 0.0003s |

## Summary
- **24/24** calculations converged successfully
- Total wall time: 0.03s

### Per-method convergence
| Method | Converged | Avg iters | Avg time |
|---|---:|---:|---:|
| DFTB0      | 6/6 | 1.0 | 0.000s |
| SCC-DFTB   | 6/6 | 1.0 | 0.000s |
| GFN2-xTB   | 6/6 | 22.7 | 0.003s |
| PM6        | 6/6 | 38.7 | 0.001s |
