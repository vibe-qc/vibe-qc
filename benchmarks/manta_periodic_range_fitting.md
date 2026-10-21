# Physical periodic range fitting

This finite-source acceptance separates the physical `erf`/`erfc` interaction
from the numerical Coulomb split. It covers complete dense blocks, immutable
shell tiles and fixed-q RI consumers. It does not qualify a production bounded
range-separated KS method or public range-fitting job selection.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_range_fitting.py
```

Four isolated workers measure five complete dense builds and five complete
tile constructions/traversals each. Each uses the small skew-cell Gaussian
fixture in `tests/test_manta_periodic_gdf.py`, two ket momenta, a 27-vector
reciprocal support, numerical omega 0.65, physical omega 0.43 and 5 bohr AO-pair
and auxiliary supports. General and reciprocal-Gamma transfers are separate
sources. Libint raw erfc and independent Gaussian-product Fourier integration
provide the reference; dense/tiled matrices agree bitwise.

| Kernel | Transfer | Dense median (ms) | Tiled median (ms) | Executed shell calls | Real-space shell calls |
| --- | --- | ---: | ---: | ---: | ---: |
| erfc | general | 0.707 | 0.798 | 916 | 848 |
| erfc | Gamma | 0.695 | 0.786 | 950 | 848 |
| erf | general | 0.067 | 0.148 | 68 | 0 |
| erf | Gamma | 0.066 | 0.149 | 68 | 0 |

These are different interactions and amounts of work, not a speedup claim.
Maximum independent errors are 1.146e-13 in the metric and 8.529e-14 in the
coupling tensor. The conservative dense storage bound is 25,553,480 bytes;
the tiled bound is 17,035,126 bytes. Both retain the prior Coulomb admission
model, including unused SR reservations for Fourier-only erf. These bounds
describe modeled owners, not process RSS.

Four complete Gamma SCF diagnostics compose native fixed Coulomb-J, physical
range-K and semilocal XC outside the production bounded solver. They use Be
RKS or Li UKS in the skew cell from the fitted KS benchmark, 6 hartree
reciprocal cutoff, 3 bohr pair support, 8 bohr auxiliary support, and the
1,320-point coarse grid with 11 bohr XC images. The explicitly specified
functional has 0.75 PBE exchange plus PBE correlation and 0.25 fitted exchange
with the stated physical kernel. No Madelung correction is applied: erf uses
the strict primed convention and erfc includes its finite zero. These are
finite diagnostic functionals, not an HSE or LC functional implementation.

| Kernel | Spin | Total energy (hartree/cell) | Independent energy error (hartree) |
| --- | --- | ---: | ---: |
| erfc | RKS | -11.541110936464628 | 1.78e-15 |
| erfc | UKS | -6.268232664178729 | 3.55e-15 |
| erf | RKS | -11.223476283217497 | 8.88e-15 |
| erf | UKS | -6.029697525060655 | 3.55e-15 |

The independent response uses libint raw fitting, analytic Gaussian Fourier
and Bloch values/gradients, libxc, NumPy factors/contractions and a separate
SciPy fixed point. Maximum density error is 2.40e-14, same-density Fock error
3.56e-15 and energy-component error 5.33e-15. No matched ORCA/CRYSTAL reference
exists for these custom finite supports/functionals; external parity remains
untested. Coarse-grid agreement does not establish quadrature convergence.

The focused tests additionally check signed angular contractions, complementary
kernels and finite-zero restoration, equal physical/numerical omega, unused
erf split invariance, image gauge, transfer conjugacy, source/frame/density
separation, immutable source controls and every resource refusal. All 126
saved original Coulomb matrix, identity and plan fields remain unchanged.

Native core SHA-256:
`8df5b27028c6e434cd24dc845e7715893503aa74fc9adb3d3d1334322ed9690d`.
