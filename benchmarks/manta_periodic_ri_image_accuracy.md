# Periodic RI overlap and kinetic image admission

Run `python benchmarks/manta_periodic_ri_image_accuracy.py --repeats 3` in
the intended native environment. The script runs complete shifted-mesh Be
RHF jobs at physical pair cutoffs 3 and 10 bohr, with automatic RI-J coupling
selection and independent exact K. Each support has a control without image
certification and completion cubes of radius 2 and 3. Auxiliary support is
12 bohr for the 3-bohr pair source and 12.1 bohr for the 10-bohr pair source.
The exact 12-bohr auxiliary boundary at larger pair support fails the existing
Gamma conjugacy guard; the guard remains enforced. It records source/native
identity, cold/repeated whole-job times, modeled memory, RSS and both separate
certificates. Same-support D/F/S/H and energies must agree bit for bit.

The completion cube is only an enclosure control. Physical pair support
changes integrals. Neither control establishes nuclear, two-electron,
metric/solve, arithmetic or relaxed-state accuracy. This tiny fixture measures
certificate overhead and bound behavior; it is not a production speedup test.


## Qualification

The implementing chat ran twelve complete public HF/KS calculations through
a local vq queue, with same-source controls, capped states, empty beta density,
automatic coupling selection, global hybrids, physical range composition and
exact J with RI K. Independent separable Gauss-Hermite S/T sums at image cubes
3 and 4 and NumPy/Decimal density contractions check the image enclosures.
The first ten cases are reused from an earlier local job that then refused
the exact 12-bohr auxiliary boundary. The final job also confirms that the
same conjugacy refusal occurs without image certification, then uses the
explicitly qualified 12.1-bohr auxiliary source for its last two cases.
The expanded-support comparison required extending the old origin-only HF
reference to actual Bloch S/T, finite nuclear images and the matching
overlap-dependent exchange correction; comparison tolerances are unchanged.
The tighter-support RHF state also agrees with independent finite Gaussian,
Boys/Fourier/libint matrix construction and SciPy SCF. Existing matched physical
exchange evidence is reused; no ORCA/CRYSTAL reference matches these custom
finite domains. Analytic image envelopes are not independently reimplemented.

Scientific source: `b33fa2d7be9c40621964c6f17c3a1326232836b4`. Native SHA256: `2d0a09b32569c008b3c30ffff8b4541eb9d0a485e560325c1a0241a457dc7ccc`.
The packet pins 1887 source/test files, 216 output files and 12 numerical archives.
All 1536 independently generated occupied subspaces satisfy the density/kinetic enclosures.
The focused image suite has 46 passing checks; 1,904 affected regressions
pass. Documentation/citation checks pass after correcting fragment formatting.
All enabled/disabled same-support S/H/D/F arrays, energies and physical source
identities agree bit for bit. Resource limits fail before poisoned numerical
payloads, and insufficient physical support refuses tight targets.

## Measured bounds and cost

Three sequential processes each run one cold and three repeated complete jobs,
with one numerical thread. Times below are medians; memory is a conservative
modeled maximum. RSS is measured separately and includes the interpreter.

| Pair cutoff (bohr) | Completion radius | S bound | T bound (Eh) | Kinetic-energy bound (Eh) | Repeated job (ms) | Modeled GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | disabled | - | - | - | 156.173 | 0.793 |
| 3 | 2 | 1.139711e-04 | 8.385522e-05 | 5.687404e-03 | 157.063 | 0.809 |
| 3 | 3 | 1.139711e-04 | 8.385522e-05 | 5.687404e-03 | 156.472 | 0.809 |
| 10 | disabled | - | - | - | 837.684 | 0.970 |
| 10 | 2 | 5.415334e-12 | 3.984380e-12 | 2.639369e-10 | 837.714 | 0.986 |
| 10 | 3 | 7.555267e-13 | 5.558855e-13 | 3.682346e-11 | 838.471 | 0.986 |

Peak measured RSS across workers: 238.0 MB.
The 3-bohr source uses 0.01 image targets; it cannot meet 1e-6 merely by
increasing the completion cube. The 10-bohr source meets all three 1e-6
targets. Its radius-1 completion cube still refuses because its exterior
bound overcounts retained images. Radius 2 or 3 succeeds. These controls
separate physical missing mass from conservatism in evaluating the enclosure.
No production acceleration or whole-method accuracy claim follows.
