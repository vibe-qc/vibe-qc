# MANTA periodic RI UHF cutoff sources

This experiment qualifies the finite cutoff factory and its composition with
fixed-projector Coulomb RI UHF. It does not qualify cutoff convergence, basis
completeness, physical range-separated fitting, BIPOLE/CRYSTAL parity, symmetry
reduction or a large-system speedup.

Run from a fresh own-checkout native build:

```sh
.venv/bin/python benchmarks/manta_periodic_ri_domains.py --repeats 5
```

The driver starts six isolated single-thread workers and verifies that each
loads a native extension newer than the checkout's C++ sources. The measured
native SHA256 is
`cb7dc9f20743d509082e24efafc7c23e0c533ebc2b7d296cfc25ddc41de9b037`.
Each worker reports count-only planning, domain generation (including its
plan), one cold complete run and five complete repeated runs. Every SCF run
reconstructs its sources and projectors. Dense reference setup and iteration
are timed separately; no common cached-factor speedup ratio is claimed.

## Sources and independent witnesses

The cases have a skew cell with lattice rows `(6, .2, .1)`, `(0, 6.2, .15)`,
`(0, 0, 6.4)` in bohr with one H or Li atom, multiplicity two,
two/four orbital/auxiliary s functions for H and three/three for Li. Gamma and
half-shifted 2x1x1 full meshes use both strict-zero-mode and BvK-ewald exchange.
The physical AO pair radius is 3 bohr, auxiliary radius 8 bohr, numerical
fitting omega 0.65, nuclear alpha 0.6 and nuclear radius 4 bohr. The reciprocal
kinetic cutoff is 6 hartree. These controls define the experiment, not an
accuracy recommendation. The source has one AO translation and 168 primed
Gamma vectors. The two-point mesh additionally has 170 vectors for each of
its two off-diagonal pair segments, with empty diagonals. All metric ranks
are full in these cases.

Independent libint GDF raw metric/coupling sources, NumPy positive factors and
contractions, and SciPy generalized-eigenproblem SCF produce the reference.
Independent one-electron and nuclear-energy witnesses check S, H and Ewald
components. Generated and equivalent explicit-source native runs preserve
bitwise D/F and identical scientific source identities.

| Maximum error across six cases | Value |
| --- | ---: |
| Overlap | 2.22e-16 |
| Core Hamiltonian | 2.66e-15 |
| Converged spin density | 2.84e-10 |
| Same-density physical Fock | 1.28e-13 |
| Same-density energy components | 7.63e-14 hartree |
| Independently converged total energy | 7.29e-14 hartree |

The density comparison includes two independently terminated SCF iterations;
the smaller same-density Fock/energy errors isolate the operator evaluation.
The native terminal Fock confirmation passed in every case.

## Timings

| Case | Plan median, microseconds | Generate median, microseconds | Cold SCF, ms | Repeated SCF median [min, max], ms | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| H Gamma strict | 2.79 | 16.83 | 2.820 | 2.690 [2.655, 2.825] | 2 |
| H shifted strict | 3.62 | 42.42 | 31.434 | 31.693 [31.380, 32.003] | 12 |
| Li shifted strict | 4.13 | 43.08 | 64.223 | 62.471 [61.442, 63.022] | 14 |
| H Gamma BvK | 2.96 | 17.29 | 3.608 | 2.685 [2.675, 2.840] | 2 |
| H shifted BvK | 3.67 | 42.75 | 44.560 | 45.269 [44.844, 57.446] | 17 |
| Li shifted BvK | 3.71 | 42.29 | 69.520 | 69.189 [69.076, 69.967] | 16 |

The cached dense reference iteration took 0.23 to 1.57 ms after 5.55 to
23.44 ms of source setup. Streaming refills coupling tiles and deliberately
retains only fixed metrics/projectors and bounded mode blocks. It is slower
than cached dense iteration here. The factory itself is a small part of these
small calculations; this does not predict costs on larger meshes.

## Resource bounds and compatibility

| Case | Candidate plan, MiB | Executed-size plan, MiB | Actual shell calls | Candidate shell bound |
| --- | ---: | ---: | ---: | ---: |
| H Gamma strict | 73.046 | 72.916 | 3,196 | 833,738,052 |
| H shifted strict | 73.501 | 73.288 | 37,556 | 2,494,537,668 |
| Li shifted strict | 73.552 | 73.315 | 72,516 | 4,209,494,292 |
| H Gamma BvK | 73.242 | 73.112 | 3,196 | 833,738,052 |
| H shifted BvK | 73.709 | 73.496 | 51,796 | 2,494,537,668 |
| Li shifted BvK | 73.760 | 73.523 | 82,128 | 4,209,494,292 |

These are modeled simultaneous-owner bounds, not RSS. Domain storage is
0.071 MiB for Gamma and 0.127 MiB for two points. The executed-size plan uses
accepted reciprocal/image counts while retaining the domain-generation
reservation. Work bounds use conservative image boxes, every retained source,
100 maximum SCF builds plus terminal confirmation and no favorable screening.
They substantially overestimate actual calls here. The script reports the
executed-size work bounds separately as well; they remain conservative.
Tighter admission and large-mesh performance remain later work.

`tests/test_manta_periodic_ri_domains.py` independently enumerates skew-cell
integer spheres, odd/even and shifted meshes, reverse-transfer closure and
exact boundaries. It verifies nonempty off-diagonal admission, empty
diagonals, rejection of invalid cutoffs, owned exports, separate fitted versus
direct transfer conventions, and AO-pair membership after independent AO and
auxiliary image shifts. All five caps pass at the bound and refuse one below,
including every auxiliary/projector owner before source generation.

An offline probe admitted 24 generated RI UHF states (72 spin/k channels)
into the current unified FockBuild draft. Evaluated density, physical action,
S/X frame, method energy, fixed source identity, metric populations, immutable
storage and solver-parent identities passed for iteration caps 1, 3 and 100.
The draft is inspected without copying it into the product. Existing shared
symmetry transport checks pass. These checks preserve integration boundaries;
they do not authorize spatial/time-reversal reduction or substitute for
BIPOLE's source and CRYSTAL parity qualification.
