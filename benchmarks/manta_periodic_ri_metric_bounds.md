# Periodic RI nested metric-factor bounds

Run `benchmarks/manta_periodic_ri_metric_bounds.py --repeats 2` in the
checkout environment. It compares the optional combined certificate with
coupling-only admission at the same selected metric and finite source.
Cold and alternating warm pairs must have bitwise equal S/H/F, density and
energy. It tests an unchanged full positive metric and a deliberately
truncated metric, with the finer reference held at 1e-12.

The truncated fixture uses loose admission targets to expose a nonzero
metric contribution. It is not a recommended accuracy preset. The combined
bound encloses error relative to the same finite source with additional
positive metric modes. It excludes finite fitting-basis/domain error,
eigensolver error, arithmetic, rank stability and relaxed-state response.
No matching ORCA/CRYSTAL custom-domain calculation is available; acceptance
uses independent Gaussian/libint/Fourier integrals and NumPy/SciPy factors
and SCF, plus Decimal positive-contraction checks.


## Accepted finite-source calculations

Local `vq` acceptance used source `522b81f9651db8e382d328b313d92ecbc4e28e8b` and an editable
0.17.23 runtime, native SHA256
`bbe2ca0083a48c599515af14f1016c20aa36f9e3cb62e79d39de051a2b10e07c`. All 1,897 source/reference
pins and the native hash were unchanged through the job. One CPU and
4096 MiB were reserved; measured largest-process RSS was
290.98 MiB. Complete source inputs are in
`metric_job` in `tests/test_manta_periodic_ri_metric_bounds.py`.

Three complete calculations converge. RHF uses Be and UHF uses Li;
selected/reference auxiliary ranks are 3/3 or 2/3. The deliberately coarse
selected cutoff is 0.013208957446495381 versus the 1e-12 reference. Acceptance
starts with coupling tolerance 1e-12; the full-rank selector refines it once.
The coarse cases request loose 1e8 J/energy admission targets solely to
measure a nonzero removed-mode contribution.

| Profile | Iterations | Energy (Eh) | Metric delta-J | Metric delta-E (Eh) | Metric J bound | Metric E bound (Eh) |
|---|---:|---:|---:|---:|---:|---:|
| full-rhf | 9 | -8.74524508028746 | 0 | 0 | 0 | 0 |
| truncated-rhf | 9 | -8.93479637392651 | 0.029210355 | 0.19656752 | 2953.949 | 707830.93 |
| truncated-uhf | 13 | -3.71842605525508 | 0.013680906 | 0.043118914 | 2953.949 | 707830.93 |

Metric differences compare the selected and finer finite RI operators at
the same evaluated density. They are not relaxed-state differences. Maximum
coupling-reference Fock/energy differences are 5.0e-15 and 1.12e-13 Eh;
independently converged densities agree within 4.81e-10. QVF spin densities
and shared Fock energy agree. Disabling the certificate at its selected
physical source preserves S/H/F, density and energy bitwise.

The nonzero metric bounds are very conservative: these fixtures establish
enclosure and refusal behavior, not useful metric truncation at a practical
accuracy target. Finite fitting-basis and domain errors remain separate.
22 new checks cover nested/empty/zero-rank limits, five resource caps,
Decimal positive contractions and complete HF/KS compatibility; 265 shared
regressions and 468 changelog/citation checks pass.

## Certificate overhead

Two fresh processes each run one cold pair and two warm pairs per profile;
warm ordering alternates. Times include complete public job execution, but
exclude the fixture's preliminary metric construction used to choose its
cutoff. The full-rank benchmark starts with the fixture coupling tolerance
0.1 and selects an exact coupling partition. The truncated benchmark keeps
0.1 under its loose target, so its energy differs slightly from the tighter
acceptance calculation. Each pair has the same numerical source and passes
bitwise state equality.

| Profile | Warm pairs | Median enabled wall (s) | Coupling-only wall (s) | Median paired wall delta (s) | Enabled/coupling-only CPU (s) |
|---|---:|---:|---:|---:|---:|
| Full | 4 | 0.152526 | 0.152645 | 0.000213 | 0.146613 / 0.146728 |
| Truncated | 4 | 0.116065 | 0.116214 | -0.000444 | 0.110156 / 0.110002 |

The first full-rank cold pair takes 0.2053-0.2077 s
with the certificate and 0.1534-0.1545 s
without it. Benchmark peak RSS is 259.73 MiB.
Warm differences are small relative to timing noise; no speedup or general
production crossover is claimed. Optional factor storage and bound work
are admitted before allocation, without an extra coupling traversal.
