# Whole-mesh MANTA RI Fock and explicit-source UHF

Finite-source acceptance measured with fresh independent single-thread workers,
five repeats per profile. Fixture definitions and independent references live in
`tests/test_manta_periodic_ri_fock.py` and `tests/test_manta_periodic_ri_uhf.py`.
This is experimental scientific API coverage; public fitted job selection,
automatic domain factories and general accuracy profiles remain open.

```sh
.venv/bin/python benchmarks/manta_periodic_ri_uhf.py --repeats 5
```

The parent fixes OpenMP, OpenBLAS and vecLib thread counts to one. Each worker
requires its own checkout's native library to be newer than all C++ sources.
These measurements used native SHA256
`56c0f3cc78d743851d06afb4f801e9067def1f00a926e2c1d62dacc48337c131`.
Timings include binding input copies and the stated work; measured on an ARM64
development runtime. They are observations, not a speedup claim.

## Complete explicit-mesh Fock

Two raw s orbital and two auxiliary functions in a skew cell, general complex
indefinite spin densities, normalized nonuniform weights and 27-vector finite
Gamma/transfer domains. Even the one-point Fock profile is an explicit nonzero
Bloch point; the physical UHF Gamma profile below is a separate fixture.
Every complete Fock sums all ordered source/target pairs. Fixed projectors
are built once; shell/mode coupling is recomputed for each build.

| k points / sources | Setup ms | First build ms | Repeated build median [min,max] ms | Modeled peak MiB | Retained sources MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 | 0.109 | 1.491 | 1.449 [1.448, 1.454] | 24.410 | 0.084 |
| 2 / 3 | 0.211 | 4.245 | 4.220 [4.212, 4.260] | 24.605 | 0.252 |
| 4 / 13 | 0.713 | 18.306 | 18.394 [18.248, 18.617] | 25.562 | 1.090 |

| k points | Metric shell calls once | Shell calls per build | Coupling passes per build |
| --- | ---: | ---: | ---: |
| 1 | 38 | 2064 | 2 |
| 2 | 110 | 5904 | 6 |
| 4 | 470 | 25104 | 26 |

Independent analytic Gaussian/Boys/Fourier integrals plus NumPy projection
and contraction check every J, K, F and electronic energy. Across these probes,
maximum absolute matrix error is 3.730e-14 a.u. and energy error
is 2.132e-14 hartree. The numerical tests also use libint raw sources,
weighted energy derivatives, reciprocity, zero weights, reordered meshes and
full/truncated-rank image transport. The shared symmetry density-push and
operator-pull interface reproduces the independent phase formula.

## Physical fitted UHF

H uses two normalized s AOs, four auxiliary functions and spin counts (1,0);
Li uses three normalized s AOs, three auxiliary functions and (2,1). The cell
is skew, about 6 bohr on each edge. AO pair support is 3 bohr, auxiliary support
8 bohr, numerical fitting omega 0.65, nuclear alpha 0.6 and nuclear support
4 bohr. The reciprocal label cube spans -2 through 2; Gamma excludes zero.
Shifted profiles use a full 2x1x1 mesh with shift (1,0,0). These finite domains
are deliberately specified, not certified converged to the infinite source.

| Profile | First SCF ms | Repeated SCF median [min,max] ms | Iterations / builds | Energy hartree/cell | Modeled peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| H-gamma-strict | 2.284 | 2.154 [2.128, 2.278] | 2 / 3 | -0.253779853399 | 72.818 |
| H-shifted-strict | 25.180 | 24.979 [24.892, 25.144] | 12 / 13 | -0.321069971404 | 73.105 |
| Li-shifted-strict | 49.254 | 48.895 [48.104, 49.122] | 14 / 15 | -5.910156486359 | 73.132 |
| H-gamma-BvK | 3.210 | 2.168 [2.146, 2.264] | 2 / 3 | -0.482451845761 | 73.014 |
| H-shifted-BvK | 35.506 | 34.634 [34.389, 34.931] | 17 / 18 | -0.472361732028 | 73.312 |
| Li-shifted-BvK | 56.166 | 55.097 [54.967, 55.195] | 16 / 17 | -6.364031768230 | 73.340 |

All native states passed their density-closure, population, gap, energy and
fresh same-density Fock-confirmation gates. The extra build is confirmation,
not another reported SCF iteration. Independent libint raw fitting sources,
NumPy projection/contractions, SciPy SCF, Gaussian one-electron quadrature and
a separate Ewald sum give the following maximum absolute differences:

| Quantity | Maximum error |
| --- | ---: |
| Overlap | 2.220e-16 |
| HCORE | 2.665e-15 |
| Same-density physical Fock | 1.274e-13 |
| Independent SCF density | 2.836e-10 |
| Same-density energy component | 7.627e-14 |
| Independent SCF total energy | 7.283e-14 |

| Profile | Metric shell calls once per SCF | Total shell calls | Reference setup ms | Cached-reference SCF ms |
| --- | ---: | ---: | ---: | ---: |
| H-gamma-strict | 148 | 2596 | 5.649 | 0.190 |
| H-shifted-strict | 436 | 30020 | 10.500 | 1.048 |
| Li-shifted-strict | 264 | 57882 | 20.708 | 1.353 |
| H-gamma-BvK | 148 | 2596 | 4.623 | 0.191 |
| H-shifted-BvK | 436 | 41380 | 9.730 | 1.483 |
| Li-shifted-BvK | 264 | 65550 | 19.966 | 1.574 |

The cached dense reference is faster on these small fixtures. Its SCF stopping
rule is an independent absolute density test; the native solver preserves its
existing method gates. These are different execution/storage strategies and
no performance ratio or general acceleration conclusion is assigned. Native
work caps reserve the requested 100 iterations plus confirmation; actual work
stops at the reported state. The JSON output also reports primitive, scalar
and kernel bounds. All byte values are conservative modeled inventories, not
measured RSS. Complete T/B tensors are absent from the production consumer,
but per-transfer source/projector owners still scale quadratically with k count.

## Integration and remaining gates

An offline adapter probe loaded the current FockBuild and OperatorFrame draft
without copying it into MANTA or changing another checkout. It admitted 24
physical RI UHF states, 72 spin/k channels: both species, Gamma/shifted meshes,
both singular conventions and iteration limits 1, 3 and 100. It preserved
evaluated D/F, method energies, fixed source across epochs, metric electron
counts, S/X frames, actions, immutable ownership and solver-parent identities.
Capped evaluated densities differed from the next orbital projectors as
expected. The draft file SHA256 values were:

- Fock state: `d2f9476e8ab36e69413e80f014280ee0a3e75f1bec44f853cd650d81d0650210`.
- Operator/frame: `04469508826b8ed8130c08d89a84f268efc53ac62482e504603bf3a799a89980`.

This records compatibility with those inspected drafts, not their integration
or a replacement of method convergence by a generic operator residual.
BIPOLE retains its own AC/BD support, target-minus-source vector convention,
cache/traversal and CRYSTAL parity gates. GDF uses ket-minus-bra transfers.
No canonical-q sharing or symmetry reduction is inferred: each explicit
finite source and retained projector must qualify before that optimization.
Physical range-separated fitting, fitted KS, cutoff factories, public jobs,
derivatives and the other approved capability rows remain required.
