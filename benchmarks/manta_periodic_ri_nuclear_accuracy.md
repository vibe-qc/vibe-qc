# Complete periodic RI nuclear accuracy admission

`manta_periodic_ri_nuclear_accuracy.py` compares complete RI calculations with
the optional nuclear certificate enabled and disabled. Both configurations
use identical physical sources, finite fitting/projector, SCF settings and
other accuracy controls. Three profiles cover ordinary RI RHF, global PBE0
and compact RHF with an 18-bohr nuclear sphere. Each process runs one cold
pair and three repeat pairs with alternating order. It checks unchanged
energies and densities and records CPU/wall cost, native memory reservations,
peak process RSS and actual source/native identity. There is no speedup claim.

The component targets cover the physical nuclear matrix, attraction energy
over the admitted canonical-density family and nuclear repulsion. They do
not certify whole-method automatic accuracy. Ordinary diffuse-source targets
are deliberately loose to exercise plumbing; the compact profile requires
all three nuclear targets at 1e-8 Eh. All physical supports remain fixed in SCF.

Run from an existing matching environment with one numerical thread:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python benchmarks/manta_periodic_ri_nuclear_accuracy.py --repeats 3
```

Scientific acceptance runs complete calculations through the local queue,
reuses saved accepted physical states and independent nuclear references,
and checks the new density-family contractions with high-precision positive
sums. No matched ORCA/CRYSTAL reference is available for these custom finite
domains. The existing Gaussian/Boys/Fourier/libint and SciPy references
remain the justified comparison. Arithmetic, finite fitting/metric error,
rank stability and relaxation are excluded. See the
[standalone producer benchmark](manta_periodic_nuclear_bounds.md).


## Recorded local acceptance

The 2026-10-19 local queue calculation pins source
`cd5ef6575731b488cf5d4ec74d595b9db7a50e0a` and native SHA256
`ccd1ac6bdf440d7c784978cae93e5b56e2d43242f907e5a00a244786eaea5347`. Five complete
RI calculations exercise ordinary RHF, deliberately capped unequal-spin
UHF, restricted PBE0, and compact RHF with short and tight nuclear spheres.
All five states, energies, Fock, overlap and core matrices agree bitwise with
saved independently accepted calculations. Nuclear matrix bounds and source
identities also agree exactly with the standalone producer's saved evidence.
The capped UHF state is a valid evaluated-state check, not a convergence claim.

Independent 80-digit Decimal sums verify both density-family contractions
and the outward nuclear total bound. Sixteen independently generated complex
occupied spaces per calculation, plus the saved SCF density, satisfy the
entrywise envelope and energy bound. Saved enlarged-support nuclear references
check attraction errors at those sampled densities. These samples supplement
the analytic enclosure; sampling alone is not its proof. Prior independent
Gaussian/Boys/Fourier/libint and SciPy evidence remains the physical reference,
including the compact Fock error below 4.06e-13 Eh and energy error below
4.49e-13 Eh. No matched ORCA/CRYSTAL reference exists for these finite domains.

| Case | Potential matrix bound / Eh | Density-family attraction bound / Eh | Nuclear repulsion bound / Eh |
| --- | ---: | ---: | ---: |
| ordinary-rhf | 3.288080e-01 | 2.213346e+01 | 1.156819e-03 |
| open-uhf | 2.466060e-01 | 1.660010e+01 | 6.507108e-04 |
| global-pbe0 | 3.288080e-01 | 2.213346e+01 | 1.156819e-03 |
| compact-short | 6.348811e-06 | 7.011620e-04 | 7.373897e-07 |
| compact-tight | 2.549864e-12 | 8.598423e-11 | 4.507796e-41 |

The compact tight case passes all three 1e-8 Eh targets. Its density-family
attraction bound is 8.60e-11 Eh; the largest observed attraction change among
its sampled spaces is 3.30e-14 Eh. Ordinary diffuse-source bounds remain
loose and require useful automatic physical support selection. The tiny
nuclear-repulsion tail excludes arithmetic and reuses the standalone
high-precision omitted-shell evidence; full FP64 energy subtraction cannot
resolve that tail independently.

The final implementation passes 76 nuclear checks and 1,108 affected shared
numerical checks, plus 1,474 document/citation checks, syntax, text, privacy
and punctuation checks. The initial test run found two fixture mistakes:
the QVF spin-density tuple required array conversion, and the direct RHF
admission test needed an explicitly restricted fixture. Both were corrected
without changing production equations. Pure/global/range KS and exact-J/RI-K
are covered by the complete-method checks. All five resource caps are tested
before reading poisoned payloads. Failed fixture output is retained privately.

### Measured complete-job cost

Three fresh processes each run a cold pair and three alternating-order
repeat pairs per profile. The table reports medians of nine repeat samples
with one numerical thread and a 4-GiB queue reservation. Enabled and disabled
jobs keep identical physical sources and other accuracy controls.

| Profile | Disabled wall / s | Enabled wall / s | Disabled CPU / s | Enabled CPU / s | Enabled native plan / MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary-rhf | 0.1589 | 0.1591 | 0.1518 | 0.1521 | 876.937 |
| global-pbe0 | 0.1485 | 0.1506 | 0.1414 | 0.1427 | 865.472 |
| compact-tight | 2.5844 | 2.5393 | 2.5661 | 2.5306 | 309.906 |

Maximum process RSS is 291.86 MiB, including Python,
native libraries and retained allocator pages. The timing differences include
run-to-run noise; the lower compact enabled time does not establish a speedup
or negative certification cost. This benchmark measures the complete method,
including shared preparation and output, rather than isolating the enclosure.
