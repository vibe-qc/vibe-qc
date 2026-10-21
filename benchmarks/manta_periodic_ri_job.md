# Public MANTA periodic RI UHF jobs

This benchmark runs the public `run_periodic_job` front door with explicit
RI J and RI K, a resolved auxiliary basis, physical cutoff controls and
`PeriodicFittingOptions`. It checks independently computed SCF results and
the evaluated spin density saved in QVF. It qualifies finite-source public
integration; fitted RHF/KS, mixed exact/RI J/K, cutoff accuracy profiles,
BIPOLE/CRYSTAL parity and symmetry reduction remain separate work.

```sh
.venv/bin/python benchmarks/manta_periodic_ri_job.py --repeats 5
```

Six isolated workers use one numerical thread each. Each verifies a fresh
own-checkout native build. The measured native SHA256 is
`43c4c924ed9343f6d8bfbdcc98b4deac5c994d0188c7b910ce30a6927dd220cf`;
the public adapter SHA256 is
`3eb3432caf9da5383ec606154941ef552b43c4b65a318048b000f02566902691`.
Each worker measures one cold public job and five complete repeated jobs with
separate temporary output stems. Generated files are removed afterward. No
operational paths or raw logs are stored in the repository.

The H/Li, Gamma/half-shifted 2x1x1 cases and their finite sources match
`manta_periodic_ri_domains.md`: a skew cell about 6 bohr on each edge, multiplicity two, H 2/4 and
Li 3/3 orbital/auxiliary s functions, reciprocal kinetic cutoff 6 hartree,
AO pair radius 3 bohr, auxiliary radius 8 bohr, numerical fitting omega 0.65,
nuclear alpha 0.6 and nuclear radius 4 bohr. Both strict and BvK exchange are
explicitly selected. These are benchmark controls, not recommended accuracy
settings. The public job permits 100 HCORE iterations and confirms the final
Fock at the evaluated density.

## Independent results and output

Libint raw GDF metrics/couplings, NumPy positive factors and contractions,
SciPy generalized-eigenproblem SCF, and separate one-electron/nuclear energy
witnesses form the independent reference. Across the six cases:

| Quantity | Maximum absolute difference |
| --- | ---: |
| Independently converged total energy | 7.29e-14 hartree |
| Same-density Fock | 1.28e-13 |
| Converged spin density | 2.84e-10 |
| QVF-restored evaluated spin density | 0, bitwise equality |

Every QVF passes validation. Each job emits nine standard artifacts: `.out`,
`.system`, geometry in XYZ/CIF/XSF, references, BibTeX, structured SCF log and
QVF. Provenance records explicit algorithms, physical/fitting controls,
auxiliary basis, ranks/projectors, method energies and separate source/frame/
evaluated-density identities. The benchmark includes normal output writing;
independent reference construction and the additional QVF readback occur
outside the measured job time.

## Measured complete-job costs

| Case | Cold job, ms | Repeated median [min, max], ms | Iterations | Output bytes |
| --- | ---: | ---: | ---: | ---: |
| H Gamma strict | 340.465 | 146.497 [143.113, 151.892] | 2 | 56,191 |
| H shifted strict | 380.960 | 201.139 [187.297, 230.199] | 12 | 68,772 |
| Li shifted strict | 420.679 | 246.144 [240.593, 256.607] | 14 | 71,442 |
| H Gamma BvK | 325.867 | 146.798 [128.212, 155.375] | 2 | 61,112 |
| H shifted BvK | 446.500 | 223.518 [211.757, 259.771] | 17 | 77,712 |
| Li shifted BvK | 468.812 | 265.010 [243.159, 280.771] | 16 | 78,167 |

Cold calls include lazy Python initialization, citation loading and output
setup. Repeated calls still reconstruct native sources/projectors, solve SCF
and write every artifact. Output size and timing can vary with runtime
metadata and filesystem conditions. These small jobs do not establish a
speedup over cached dense fitting or predict large-mesh scaling.

The native plan, including caller/output reservations, spans 75.585 to
76.284 MiB. The public job separately identifies 2.626 to 2.695 MiB of output
reservation plus 44 KiB of scientific Python provenance at Gamma or 68 KiB
at two points. These are conservative modeled simultaneous-owner bounds,
not measured RSS. Source work bounds retain the conservative image boxes and
maximum-iteration costs documented in the cutoff benchmark.

## Additional acceptance

`tests/test_manta_periodic_ri_job.py` checks full output/provenance equality,
standard citations, all five exact/one-short source-plus-output caps, dry runs,
immutable configuration, invalid/unqualified selection, output opt-outs and
delivery failure status. Separate tests compare rank-one and zero-rank fits
to independent contractions and SCF. Iteration-capped jobs preserve the
evaluated D rather than reconstructing the next orbital projector. Independent
AO/auxiliary image shifts preserve energy, and QVF transforms both spin
densities to the nuclear-center basis without changing the original receipt.

Repository compatibility tests check 24 public states and 72 spin/k channels
against the landed unified FockBuild API, including capped and converged
states. They check physical action, S/X frame, energy, source stability, metric
populations, immutable density and solver-parent roles. Dense exports remain
outside the native bounded solver; no duplicate operator API is introduced.
These integration checks do not issue source-specific symmetry certificates or replace BIPOLE/CRYSTAL parity gates.
