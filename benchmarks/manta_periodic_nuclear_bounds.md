# Nuclear accuracy on physical periodic RI supports

Run `python benchmarks/manta_periodic_nuclear_bounds.py --repeats 3`
with this checkout's native runtime and one numerical thread. The benchmark
times complete RI RHF/global-hybrid jobs and their standalone nuclear bound
separately. Repeated states and energies must agree bitwise. The certificate
must identify exactly the physical nuclear source used by the completed job.
It reserves the method plan as other live memory beside the bound producer.

The ordinary sources expose their actual, potentially loose bounds. The
compact helium source uses exponents 4/6, a 6-bohr cell, 1-bohr pair cutoff,
512-Eh reciprocal cutoff and 18-bohr nuclear sphere. Its AO/real/reciprocal
completion radii are 2/5/10. Completion radii control proof work, not the
physical support. A companion shorter-sphere case in acceptance isolates
the physical nuclear-domain improvement.

The independent validation uses ordered Gaussian polynomial erfc integrals,
analytic Fourier moments, separately enumerated nuclear point sums, and
larger domains or a changed Ewald split. Earlier accepted RI Fock states
are reused for compatibility comparisons. There is no matched ORCA/CRYSTAL
reference for these custom finite domains; the remaining external-reference
limitation is explicit. Native envelope producers retain their prior
independent validation.

This recorded increment supplies the standalone nuclear matrix and repulsion
bounds. The subsequent [complete RI adapter](manta_periodic_ri_nuclear_accuracy.md)
adds pre-SCF target admission and canonical-density energy contraction.
Automatic physical-source selection remains pending. It excludes floating integral
arithmetic, finite fitting/metric error, k/basis convergence, changed overlap
spaces and density relaxation. No performance gain or whole-method accuracy
is inferred from these measurements.


## Recorded local acceptance

The 2026-10-18 local queue run pins implementation source
`2f20b2528e33bf38d04cc19b0d9fa2ee7a5335a3` and native SHA256
`3454d33df2f511e091acc07aada21f7c99fe6d576dfea55f74724a8b933aff16`. It exercises five
complete RI states: RHF, deliberately capped unequal-spin UHF, restricted
PBE0, and compact RHF with 4- and 18-bohr nuclear spheres. The first four
states, energies and source identities agree bitwise with saved accepted
evidence. All five certificates identify the physical nuclear operator
actually used by their completed method.

Independent finite nuclear references agree within 7.11e-15 Eh. The new
18-bohr compact calculation agrees with an independent complete Fock within
4.06e-13 Eh, energy within 4.49e-13 Eh, and SciPy SCF density within 3.16e-10.
Independent Ewald split changes and larger domains check the nuclear
limits; 70-digit Decimal contractions check the evaluated-density attraction
bound. There is no matched ORCA/CRYSTAL reference for these finite domains.
The source passes 684 affected numerical checks, plus citation, document,
fragment, syntax and privacy checks. The first test attempt caught a
noncubic-domain adapter refusal and an independent Fourier-sign mismatch;
both were corrected before acceptance. Failed outputs are retained with
the private calculation evidence.

| Case | Potential matrix bound / Eh | Observed nuclear matrix change / Eh | Nuclear repulsion bound / Eh |
| --- | ---: | ---: | ---: |
| ordinary-rhf | 3.288080e-01 | 8.693098e-06 | 1.156819e-03 |
| open-uhf | 2.466060e-01 | 6.519824e-06 | 6.507108e-04 |
| global-pbe0 | 3.288080e-01 | 8.693098e-06 | 1.156819e-03 |
| compact-short | 6.348811e-06 | 1.269431e-06 | 7.373897e-07 |
| compact-tight | 2.549864e-12 | 6.217249e-15 | 4.507796e-41 |

The compact source's potential bound drops from 6.35e-6 to 2.55e-12 Eh
when the physical nuclear sphere grows from 4 to 18 bohr. Its evaluated
attraction bound is 1.37e-11 Eh. Ordinary diffuse-source bounds are still
loose; they are not evidence of useful automatic whole-method accuracy.

The tiny nuclear repulsion limits exclude arithmetic. Differences between
complete FP64 energies resolve only about 1e-16 Eh here. Independent
110-digit positive omitted-shell sums provide an additional check:
the compact real witness is 6.48618e-58 Eh below its 6.51102e-58 bound;
the reciprocal witness is 5.65025e-311 Eh below its 4.50780e-41 bound.
These finite positive witnesses are not a separate proof of the infinite
exterior; the shared analytic envelope supplies that proof.

### Measured cost

Three fresh processes each run one cold and three repeat calculations per
profile. Values below are medians of nine repeated samples, with one CPU
and a 4-GiB queue reservation. Native memory includes a conservative
reservation for the completed method beside the standalone bound producer.

| Profile | Complete calculation / s | Nuclear bound / s | Bound CPU / s | Joint native plan / MiB |
| --- | ---: | ---: | ---: | ---: |
| compact-tight | 2.3546 | 0.014214 | 0.014198 | 309.781 |
| global-pbe0 | 0.1330 | 0.000960 | 0.000951 | 865.346 |
| ordinary-rhf | 0.1452 | 0.000940 | 0.000940 | 876.811 |

Maximum process RSS was 261.67 MiB. It includes Python,
native libraries, all profiles and allocator retention, unlike the
count-only native plan. These measurements describe added certification
cost and do not establish a speedup.
