# Paired periodic RI Coulomb contraction

Measured on 2026-10-14 with one requested OpenMP/BLAS thread. Reproduce with:

```sh
.venv/bin/python benchmarks/manta_periodic_ri_paired.py --workers 3 --repeats 3
```

Three sequential isolated workers each construct a source, measure a cold
J build and three warm builds. Construction and Python startup are outside
the warm timers. The script records source/factor identities, actual work,
native hash, admitted peak and process RSS. This is a small finite-source
contraction benchmark; elapsed times do not establish asymptotic scaling or
a whole-method speedup. Separate sessions may use the same machine.

The fixture has two unnormalized s orbital Gaussians with exponents 0.7 and
1.1, and six auxiliary s Gaussians with exponents 0.5, 1.2 and 3.0 at each
center. Centers are (0.2,0.3,0.4) and (1.5,0.7,0.8) bohr. Lattice rows are
(3.8,0.4,0.1), (0,4.1,0.3), (0,0,4.3) bohr, with lattice vectors in columns.
The Cartesian trial momenta are (0.23,0.09,-0.13) and (-0.15,0.20,0.31),
with weights (0.27,0.73). The deterministic complex Hermitian spin densities
use seed 754 and have no assumed spatial or time-reversal symmetry.
Omega is 0.65, AO-pair cutoff 5 bohr, auxiliary cutoff 8 bohr, and the
reciprocal domain is the 125 labels in [-2,2] cubed. The wider auxiliary and
reciprocal support makes this six-mode finite metric positive without
changing negative-mode refusal or rank thresholds. Mode block is one.

| Contraction | Coupling passes | Coupling shell calls | Warm median ms | Warm min-max ms |
| --- | ---: | ---: | ---: | --- |
| Retained-mode baseline | 6 | 58,500 | 43.356 | 42.501-46.808 |
| Paired raw | 2 | 19,500 | 27.052 | 25.712-28.438 |

Primitive-product counts equal shell-call counts for these single-primitive
s shells. The modeled joint peak is 25,478,844 bytes for both schedules;
it conservatively reserves the original maximum-rank retained-mode work.
Observed process RSS ranges are startup/library dependent and are included
in the worker JSON, separately from the admission model. The measured median
contraction ratio is 1.60 on this fixture. The exact threefold reduction in
coupling evaluations does not promise the same elapsed-time ratio.

Baseline source: `1de4ed35b1110824033bc8b3f3138cc8b0ad76d4`; native SHA-256
`3c88c7b5baade58d3a6299c3a4d98343065a2924ce63deecb2d4981275ffa07b`.
Paired implementation measured source: `0cdabd1a4ee807dbd3a0aa064a4a1a32c10cb167`;
native SHA-256 `576772c24e7e08c82dfb663246daffe41f0c76d1d12a8bd540427b29de6836c6`.
Implementation after rebasing onto incoming documentation and molecular
symmetry integration: `beeb82a3626103859b97c5c33abf9493ef34210b`. Native and MANTA periodic
sources are unchanged. Five public HF/KS integration checks pass, and the
refreshed native binary matches the accepted runtime byte for byte.
The baseline and paired runs have identical source, factor and projector
identities. Maximum J difference is 6.598e-16 and energy difference is
5.552e-17 hartree. The native factor is retained as produced, including its
finite Gram matrix; no whitening or rank repair was applied.

## Complete-method acceptance

One actual local `vq` run executed seven public jobs and seven matched
retained-mode controls. All use the same shifted (2,1,1) mesh and explicit
finite source policies in `tests/test_manta_periodic_ri_paired.py`. HF uses
RI J and exact reciprocal K with the fixed BvK Ewald correction; pure KS
uses PBE and its admitted quadrature. Empty-beta H UHF is included.
The three-iteration Li cases deliberately remain nonconverged.

| Case | Iterations | Status | Total energy, hartree |
| --- | ---: | --- | ---: |
| rhf-Be | 14 | converged | -11.447576224298 |
| uhf-H | 17 | converged | -0.472159226834 |
| uhf-Li | 15 | converged | -6.263650977111 |
| uhf-Li-capped | 3 | iteration limit | -6.263615839417 |
| rks-Be | 9 | converged | -11.695694654271 |
| uks-Li | 13 | converged | -6.397168375806 |
| uks-Li-capped | 3 | iteration limit | -6.397110231130 |

Independent libint raw metric/coupling integrals, Gaussian Fourier
quadrature, NumPy/SciPy factorization and independent SCF supply references.
KS grid contractions are independent while sharing the same libxc
functional. A separate NumPy-only assessment of fetched arrays checks all
results and hashes 126 managed output files. Maximum same-density Fock,
energy and relaxed-density differences are 5.151e-14,
9.415e-14 hartree and 2.791e-10, respectively.
Maximum relaxed-energy difference is 1.030e-13 hartree.
The unchanged gates are 8e-11 for Fock/energy and 5e-9 for density.
Paired versus retained-mode complete-state Fock differences stay below
2.220e-15, and energies below 2.665e-15 hartree.
The Coulomb energy/trace discrepancy is at most 8.882e-16 hartree;
evaluated-density QVF round trips are exact. Shared Fock channels retain
the evaluated state and method-owned energy ledger. Capped states remain
unconfirmed, and no geometry-only symmetry authority is supplied.

The whole effective command took 5.648 seconds wall, 5.473 seconds CPU and
237.891 MiB peak RSS. The first harness attempt stopped on an identity
assertion because independently formed momenta differed from the native
frame by 1.735e-18. The corrected diagnostic uses native frame data and
keeps the strict source identity and every numerical tolerance unchanged.

Twenty-five new focused checks and 146 existing q-channel/target/Fock
checks pass. They cover ranks zero through six, compensated paired
contractions, weighted reciprocity and energy derivatives, nonuniform/zero
weights, independent atom-image transport and exact/one-short joint caps.
The retained-mode joint J/K and K-only routes remain covered.

No suitable ORCA/CRYSTAL reference matches these custom finite GDF and
reciprocal supports. The independent checks qualify the selected finite
operator; they do not establish basis, mesh, source-cutoff or relaxed-state
accuracy. The paired exact baseline implements the adjoint contract of
Neese et al., DOI 10.1021/acs.jpca.4c07415, Eqs. 29-36, on the existing
Sun et al. periodic GDF source, DOI 10.1063/1.4998644. It neither introduces
an approximate periodic RI-BUPO hierarchy nor changes BIPOLE/CRYSTAL
parity domains. Approximate q-aware coupling, producer-qualified symmetry,
automatic whole-method profiles and the remaining MANTA matrix stay open.
