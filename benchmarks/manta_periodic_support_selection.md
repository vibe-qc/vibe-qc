# Automatic periodic RI physical supports

The selector chooses physical AO-pair, nuclear and reciprocal supports before
SCF using shared S/T, nuclear and optional full-Coulomb exchange producers.
Run `benchmarks/manta_periodic_support_selection.py --profile compact` in the
checkout environment for cold and alternating automatic/explicit measurements.
The explicit control uses exactly the selected physical supports and proof
cubes. Energies, S/H/F and densities must agree bitwise. Optional ordinary,
open and global profiles exercise enlarged diffuse sources; the open profile
is deliberately iteration capped.

Every failed trial is charged. The selected trial is recomputed by the final
method to verify source and certificate identities; that preparation cost is
included. Timing compares automatic search with an already-known explicit
source and is not a production speedup claim. Ordinary diffuse sources use
an explicit auxiliary radius of 12.1 bohr to avoid a 12-bohr capsule boundary
tie. The existing refusal for non-conjugate finite support remains tested.
The selector does not change auxiliary radii or conceal that failure.

Requested bounds are 1e-8 per enabled component, not a total relaxed-energy
target. Finite fitting/metric, arithmetic, overlap-rank stability, relaxation,
basis/k convergence and XC errors are excluded. The separate RI coupling
selector retains its own targets. No matching ORCA/CRYSTAL reference is
available for these custom finite domains; acceptance uses independent
Gaussian/Boys/Fourier and libint sources with NumPy/SciPy contractions.

Source `d70e86fa5c690c40364248d0c15ef8b6778a1a59` passed local queue acceptance through `vq`,
one CPU and a 4096 MiB reservation. Native SHA256:
`ee470f4b50d90baf1345de95526aef1ab18edfd8249c01025e5f908bd24f7075`.
All 1,895 source/test/benchmark pins were unchanged during acceptance.

| Case | Trials | Pair radius / bohr | Nuclear radius / bohr | Reciprocal cutoff / Eh | Calculation wall / s | Independent Fock difference | Energy difference / Eh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact | 4 | 1 | 6 | 364.5 | 1.8001 | 2.232e-13 | 2.180e-13 |
| ordinary | 7 | 15.1875 | 9 | 778.4780273 | 254.0892 | 6.883e-15 | 1.386e-13 |
| open | 7 | 15.1875 | 9 | 778.4780273 | 266.2405 | 6.252e-15 | 1.059e-13 |

The compact case converged in 11 iterations; its independently
iterated SciPy density differs by 3.166e-10.
The two diffuse cases each completed one evaluated iteration. Their saved
densities match independent initial occupations, and their Focks and energies
match the independent finite source within the separate RI coupling bound.
They are not converged-state accuracy claims. Every output QVF validates and
the evaluated states cross the shared Fock API without enabling symmetry.
The compact automatic and explicit selected-support states agree bitwise.

The compact final nuclear matrix/attraction/repulsion bounds are
4.60979e-11,
4.18221e-09 and
1.73476e-12 Eh. Its exchange matrix/energy
bounds are 5.86632e-11 and
3.65015e-09 Eh. All eight requested
component targets pass in each case. Overlap bounds are dimensionless.
Diffuse selection retains 67 AO translations,
246,976 Gamma vectors and
987,800 exact-exchange vectors. This is expensive and
does not establish useful production performance or minimal domains.

Two fresh benchmark processes each executed a cold pair followed by two
alternating automatic/explicit pairs. Median warm complete-job wall times
were 1.8562 s automatic and 1.7840 s explicit;
median paired overhead was 0.1358 s. Cold medians were
1.9326 / 1.8318 s, respectively.
Maximum benchmark process RSS was 295.34 MiB. These small-process
measurements include managed output and repeat final certificate preparation;
they are observations, not a portable cost guarantee or speedup claim.

Twenty-one selector checks and 420 shared numerical regressions pass.
The combined selector/changelog/citation run has 1,910 passes and one
pre-existing failure: the main BIPOLE baseline fragment lacks the changelog
test's required opening bullet. The new fragment, staged privacy, syntax and
text checks pass. No unrelated numerical guard or tolerance was changed.


After incoming BIPOLE lattice-key batching and weighted Fourier Hermite
reuse were merged, a refreshed native runtime was checked through `vq` at
source `4aa1cd4ee147e3f3b334831d1dad4a05044786cc`, native SHA256
`a88f742c3914b97a46f3554b22c49cee26913dbec78e2d85b375085df447e125`.
The converged compact S/H/F, density and energy remain bitwise equal to the
accepted state above, with identical source and selection certificates.
The newly required shared native export, QVF and shared Fock checks pass.
Three original pinned files changed in those incoming commits; no MANTA
source changed. All 1,895 pins stayed fixed during the integration check.
The original independent comparisons and diffuse calculations are reused;
the timing measurements above refer to the original accepted runtime.
