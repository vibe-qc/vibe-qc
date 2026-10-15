# MANTA restricted periodic RI HF

This increment supplies an actual bounded RHF numerical leaf over the shared
fixed-projector periodic RI source, through explicit arrays, generated cutoff
domains and public jobs. It qualifies neutral all-electron 3D closed shells
on full regular meshes. Mixed exact/RI, fitted KS, physical range-separated
fitting, accuracy profiles and source-specific symmetry remain separate work.

```sh
.venv/bin/python benchmarks/manta_periodic_ri_rhf.py --repeats 5
```

Six isolated workers use one numerical thread each and verify a fresh native
build from their own checkout. Native SHA256:
`9aae2d770ab933cb240c4b7b0fa60680dbff1c6ff1f3e628877bdf9caeba261c`.
Public adapter SHA256:
`46f77c2a10234ef9196bc730067fc28e4cb45f2301481bc3251b35c811cff3a0`.
Each measures one cold public job, five complete repeated jobs and five native
cutoff/SCF runs. Every run reconstructs its fixed fitting source. Temporary
outputs are removed; no operational paths or raw logs are stored here.

## Finite source and independent checks

The cell has lattice rows `(6, .2, .1)`, `(0, 6.2, .15)`, `(0, 0, 6.4)` in
bohr. One He or Be nucleus at `(.1, -.15, .2)` has two or four electrons.
He uses two normalized s orbital functions and four s auxiliaries; Be uses
three of each. The orbital/auxiliary exponents are `(0.8, 0.25)` and
`(0.3, 0.5, 1.05, 1.6)` for He, `(2.4, 0.7, 0.2)` and `(0.4, 1.4, 4.8)` for
Be. AO pair radius is 3 bohr, auxiliary radius 8 bohr, numerical fitting omega
0.65, nuclear alpha 0.6, nuclear radius 4 bohr and reciprocal kinetic cutoff
6 hartree. These define the experiment, not recommended accuracy settings.

Gamma and half-shifted 2x1x1 meshes each use strict-zero-mode or BvK-ewald
exchange. The source has 168 primed Gamma vectors; each off-diagonal segment
at two k points has 170 vectors. Diagonals are empty and use the Gamma source.
All auxiliary metrics have full rank in the timed cases.

Independent libint raw metric/three-center arrays, NumPy positive fitting
factors and contractions, SciPy restricted generalized-eigenproblem SCF and
separate one-electron/nuclear-energy witnesses supply the reference. The
reference imposes equal spin halves before constructing the physical Fock,
then solves for the spin-summed restricted density. Across the six cases:

| Quantity | Maximum absolute difference |
| --- | ---: |
| Converged total energy | 1.319e-13 hartree |
| Same-density Fock | 1.267e-13 |
| Converged spin-summed density | 3.313e-10 |
| Native versus public evaluated density | 0, bitwise |
| QVF equal-spin halves versus evaluated density halves | 0, bitwise |

Every QVF validates. All nine standard artifacts are emitted, including
structured history, citations, geometry, manifest and output. Independent
reference work and the additional QVF validation/readback are outside the
measured public job. The energy ledger preserves nuclear, one-electron, J,
uncorrected K and separately selected exchange-zero-mode terms.

## Complete native and public costs

| Case | Native median [min, max], ms | Cold public, ms | Repeated public median [min, max], ms | Iterations | Output bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| He-gamma-strict | 10.237 [10.176, 10.856] | 286.990 | 106.073 [99.853, 113.841] | 10 | 58,991 |
| He-shifted-strict | 22.949 [22.717, 25.490] | 227.312 | 121.036 [114.487, 130.061] | 7 | 62,052 |
| Be-shifted-strict | 46.968 [45.369, 50.289] | 273.841 | 135.996 [129.878, 140.158] | 8 | 62,368 |
| He-gamma-BvK | 16.047 [15.631, 18.142] | 225.083 | 111.403 [105.813, 120.057] | 16 | 66,375 |
| He-shifted-BvK | 50.451 [45.776, 53.526] | 268.870 | 137.483 [131.954, 145.526] | 15 | 70,203 |
| Be-shifted-BvK | 80.098 [72.176, 83.885] | 281.002 | 162.867 [157.916, 179.062] | 14 | 69,903 |

Native modeled aggregate peak ranges from
72.902 to
73.509 MiB; public aggregate
admission including output/evidence ranges from
75.563 to
76.235 MiB. The plan counts
all immutable projectors, streamed spin Fock workspace, scalar history,
restricted solver storage and terminal confirmation. Output and per-transfer
Python evidence are separate reservations. These are conservative modeled
bounds, not measured RSS. Timings depend on runtime and system load; these
small fixtures establish no large-mesh speedup or accuracy-profile claim.

## Additional acceptance and integration

`tests/test_manta_periodic_ri_rhf.py` checks independent stopped recurrences,
restricted metric idempotency and electron counts, BvK factors without an
extra source weight, truncated/zero-rank auxiliary spaces, negative metric
refusal, rank-reduced overlap frames, explicit/generated source equivalence
and all five exact/one-short native/domain/output caps. Public stopped states
save the evaluated density even when the next orbital projector differs.
Independent AO/auxiliary image shifts in a multicenter cell preserve energy
and transport D/F/S/H; QVF recovers the nuclear-center density.

The shared FockBuild API accepts restricted total-density channels and full
or rectangular S/X frames. Method energy, physical action and density epochs
are checked without adding a dense snapshot owner inside the bounded native
solver. Equal-spin UHF and RHF share the fitted source and geometric receipt
but retain distinct method identities, numerical storage and stopping rules.
A before/after comparison of 16 existing UHF capped/converged states preserved
all arrays, spectra, energies, source/projector identities and resource counts
bitwise while their physical source composition was shared with RHF.

Symmetry reduction still requires the selected finite source's own admission.
This fitting source does not substitute BIPOLE AC/BD supports or replace its
CRYSTAL parity gates. Remaining MANTA capabilities stay open.
