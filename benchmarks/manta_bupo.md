# MANTA molecular BUPO construction benchmark

This diagnostic measures molecular Coulomb construction, independently of
SCF, exchange and symmetry. Run the rebuilt editable checkout with:

```sh
.venv/bin/python benchmarks/manta_bupo.py > /tmp/manta-bupo.json
```

Use `--case water-dimer` for one fixture, `--repeats 5` for more samples, or
`--timeout 600` to increase the per-variant subprocess limit. Each variant
runs in a fresh process with one BLAS/OpenMP thread, one warm-up and at least
three measured repetitions. The JSON records platform and software versions
and the native binary hash, without machine names or checkout paths.

Fixtures are water/STO-3G (7 AOs), a distorted water dimer/STO-3G (14 AOs)
with a 6 bohr displacement, the same pair with a 60 bohr displacement, and
the compact dimer augmented with one normalized diffuse s primitive of
exponent 0.025 on each oxygen (16 AOs). The long-distance fixture exercises
far fields and is not a typical bound molecular dimer. Every timed build
constructs J for two densities: a seeded positive semidefinite trial density
normalized to the electron count in S, and an independent signed response
density. These are supplied densities, not converged SCF solutions.

The six variants separate different contributions:

- Existing zero-threshold optimized direct libint is the practical baseline.
- Ordered libint and ordered MANTA use the same J-only quartet schedule.
- Exact near-only MANTA BUPO uses canonical unordered pair quartets.
- MANTA BUPO adds Gaussian extent admission and far multipoles.
- BUPO with libint near tiles keeps the same partition and MANTA moments.

BUPO uses complete Cartesian order 12, opening ratio 0.8, leaf capacity 4,
and per-integral approximation tolerance `1e-11`. Exact near-only sets
opening zero. No shell screening or incremental-density state is enabled.
Every variant must agree with the existing direct libint J to `1e-8` in
both maximum matrix error and Coulomb energy error for each supplied
density. The script also checks that switching BUPO's near provider leaves
the partition unchanged. The analytic J approximation ledger is reported
separately from observed error and floating-point effects.

Wall and CPU measurements separate constructor setup from repeated builds.
The reported cold cost is setup plus the median two-density build, rather
than a full application runtime. Basis preparation, imports and reference
validation are excluded from that time. BUPO's opt-in timers separate
extent/tree/error-model/partition setup and moment/translation/far/near build
work. Component regions are disjoint within each invocation, but separately
computed medians need not sum to the median total. Traversal and control
overhead appears in the total without a component label. Near integral time
includes the provider's own recurrence, contraction and pure transforms;
near digestion measures the subsequent density contraction.

Peak RSS is the worker process high-water mark, reported both before and
after independent reference validation. It includes the loaded runtime and
retained matrices. It is distinct from the producer's modeled workspace cap.
The optimized libint builder does not expose actual work counters, so those
fields are unavailable; libint primitive evaluations are not inferred from
requested counts. MANTA counters report executed work for the density pair.

Small fixtures establish correctness and identify overhead. They do not
establish linear scaling, a large-system crossover or an end-to-end SCF
speedup. An ordered-versus-BUPO ratio also includes the change in quartet
schedule; the near-only control is necessary before attributing a gain to
the multipole approximation.

## Measured small-system result

The October 2026 run used ARM64 macOS, Python 3.14.8 and NumPy 2.5.3,
with native optimization enabled and binding LTO disabled. All 24 variants
used native binary SHA-256
`9e73654bcec526b242676e027ad67d73612348c362c8b666e9c8314f98b76184`.
The maximum observed J error was `8.55e-14` and maximum Coulomb-energy error
was `1.31e-13`, rounding upward. The run used three measured repetitions
per variant and did not enforce machine-wide exclusivity. Ranges expose
timing variation; these results are diagnostics, not a throughput guarantee.

All build times and work counts below are for the density pair. RSS is the
full process peak including reference validation.

| Case | Variant | Setup ms | Build median ms | Build range ms | Peak MiB | Near or ordered calls | Far blocks |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Water | libint-direct | 0.400 | 0.544 | 0.543-0.549 | 146.27 | unavailable | n/a |
| Water | libint-ordered | 0.032 | 3.216 | 3.202-3.350 | 146.23 | 1250 | n/a |
| Water | manta-ordered | 0.023 | 74.096 | 64.543-75.771 | 147.39 | 1250 | n/a |
| Water | manta-near | 27.726 | 14.678 | 14.601-14.743 | 147.20 | 240 | 0 |
| Water | manta-bupo | 26.726 | 14.687 | 14.625-15.045 | 147.88 | 240 | 0 |
| Water | libint-near-bupo | 28.043 | 0.840 | 0.809-0.905 | 147.72 | 240 | 0 |
| Compact dimer | libint-direct | 0.593 | 6.265 | 5.955-6.320 | 148.11 | unavailable | n/a |
| Compact dimer | libint-ordered | 0.032 | 63.480 | 54.668-66.162 | 149.28 | 20000 | n/a |
| Compact dimer | manta-ordered | 0.033 | 942.515 | 934.549-972.386 | 147.78 | 20000 | n/a |
| Compact dimer | manta-near | 99.034 | 148.858 | 144.266-153.400 | 148.52 | 3080 | 0 |
| Compact dimer | manta-bupo | 104.943 | 189.186 | 179.714-194.383 | 148.12 | 3080 | 0 |
| Compact dimer | libint-near-bupo | 109.687 | 7.980 | 7.436-8.560 | 147.34 | 3080 | 0 |
| Separated dimer | libint-direct | 0.469 | 2.241 | 2.200-2.618 | 147.66 | unavailable | n/a |
| Separated dimer | libint-ordered | 0.032 | 43.392 | 43.329-44.335 | 147.20 | 20000 | n/a |
| Separated dimer | manta-ordered | 0.029 | 982.752 | 936.987-1019.808 | 147.31 | 20000 | n/a |
| Separated dimer | manta-near | 49.282 | 149.319 | 147.765-150.674 | 147.44 | 3080 | 0 |
| Separated dimer | manta-bupo | 196.558 | 685.415 | 663.738-708.311 | 152.98 | 1688 | 48 |
| Separated dimer | libint-near-bupo | 205.140 | 645.187 | 617.403-655.285 | 152.34 | 1688 | 48 |
| Diffuse dimer | libint-direct | 0.450 | 7.851 | 6.992-7.865 | 147.12 | unavailable | n/a |
| Diffuse dimer | libint-ordered | 0.026 | 56.312 | 55.829-58.084 | 147.53 | 41472 | n/a |
| Diffuse dimer | manta-ordered | 0.030 | 1318.110 | 1256.473-1331.404 | 146.94 | 41472 | n/a |
| Diffuse dimer | manta-near | 111.980 | 222.159 | 213.391-222.430 | 147.02 | 6162 | 0 |
| Diffuse dimer | manta-bupo | 121.173 | 218.251 | 215.974-220.226 | 147.45 | 6162 | 0 |
| Diffuse dimer | libint-near-bupo | 105.178 | 8.632 | 8.629-8.637 | 147.05 | 6162 | 0 |

No far block passed the strict error admission in the water, compact-dimer
or diffuse-dimer cases. Their MANTA improvement over ordered evaluation
comes from the canonical quartet schedule. Optimized libint remains faster
than every MANTA path, and is also faster than BUPO with libint near tiles.

For the separated dimer, far approximation reduced near shell calls from
3080 to 1688, but increased MANTA build time from 149 ms to 685 ms. The
48 accepted far blocks required 220 moment panels, 1980 shared primitive
moment expansions and 662 traversal visits across the density pair.
Component medians for that MANTA BUPO run were:

| Component | Milliseconds |
| --- | ---: |
| Leaf moments | 82.343 |
| Upward translations | 85.820 |
| Far error admission and matrix formation | 263.175 |
| Far matrix applications | 0.974 |
| Near integral evaluation | 78.395 |
| Near density digestion | 0.179 |
| Downward translations | 84.727 |
| Reverse moment panels | 82.123 |

Far matrix application itself is inexpensive here. Rebuilding its matrices
and error bounds, generating the forward/reverse moment panels, and
forming translations dominate. These measurements motivate optimization
of bounded geometry work before any speed claim for the multipole path.
The current implementation is a validated construction baseline; this
benchmark does not establish a crossover against optimized libint.
