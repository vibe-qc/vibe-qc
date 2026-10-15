# Independent periodic RI J and K targets

Measured on 2026-10-10 with the matching native extension and one requested
OpenMP/BLAS thread. Reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ri_targets.py --repeats 5
```

The script starts a fresh process for each of nine target/mesh combinations.
It measures one constructor, one cold build and five subsequent builds;
both independent references run outside those timers. Native SHA-256:
`cc13a9e042721f1d0c134d7cc381fa663d7148d501f2b1327bc7620051b018ae`.

The fixture has two unnormalized s orbital Gaussians with exponents 0.7 and
1.1, and two auxiliary s Gaussians with exponents 0.6 and 0.9. Centers are
(0.2,0.3,0.4) and (1.5,0.7,0.8) bohr. Lattice rows are
(3.8,0.4,0.1), (0,4.1,0.3), (0,0,4.3) bohr; columns are lattice vectors.
Numerical split omega is 0.65, and AO-pair/auxiliary cutoffs are 5 bohr.
Gamma uses the 27 reciprocal labels in [-1,1] cubed. Each selected
off-diagonal K transfer uses those labels plus its ket-minus-bra momentum.
The 1/2/4-point meshes are prefixes of the explicit Cartesian momenta
(0.23,0.09,-0.13), (-0.15,0.20,0.31), (0.03,-0.23,0.16),
(0.32,-0.11,0.27), with normalized weights proportional to 1,2,3,4.
These are arbitrary trial meshes, not physical SCF or accuracy profiles.
All retained metric ranks are two, with mode block one and absolute
cutoff 1e-12. The seeded complex Hermitian spin densities are independent
trial matrices, not assumed to have state symmetry or integer occupations.

| nk | Target | Metrics | Metric shell calls | Build shell calls | Coupling passes | Admitted peak bytes | Retained source bytes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | J | 1 | 38 | 2,064 | 2 | 25,595,470 | 87,912 |
| 1 | K | 1 | 38 | 2,064 | 2 | 25,595,470 | 87,912 |
| 1 | J+K | 1 | 38 | 2,064 | 2 | 25,595,470 | 87,912 |
| 2 | J | 1 | 38 | 2,064 | 2 | 25,602,110 | 88,008 |
| 2 | K | 3 | 110 | 5,904 | 6 | 25,799,950 | 263,832 |
| 2 | J+K | 3 | 110 | 5,904 | 6 | 25,799,950 | 263,832 |
| 4 | J | 1 | 38 | 2,064 | 2 | 25,616,926 | 88,200 |
| 4 | K | 13 | 470 | 25,104 | 26 | 26,803,966 | 1,143,144 |
| 4 | J+K | 13 | 470 | 25,104 | 26 | 26,803,966 | 1,143,144 |

The peak is the complete conservative admission model, not measured RSS.
It includes fixed backend margins, all retained metrics/projectors and
source snapshots, shell/mode workspaces, density/response owners and
exports. Each q leaf still reserves its joint J/K workspace. Thus K-only
has the same conservative bound as J+K. J-only removes every off-diagonal
source and requires an empty pair domain. It still validates the full mesh
and admits the offset metadata; this is not a linear-scaling mesh claim.

| nk | Target | Constructor ms | Cold build ms | Warm median ms | Warm min-max ms |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | J | 0.177 | 2.448 | 2.650 | 2.411-2.747 |
| 1 | K | 0.197 | 2.188 | 2.211 | 2.133-2.401 |
| 1 | J+K | 0.159 | 2.621 | 2.414 | 2.376-2.607 |
| 2 | J | 0.182 | 2.760 | 2.554 | 2.402-2.910 |
| 2 | K | 0.372 | 7.864 | 7.664 | 7.626-7.955 |
| 2 | J+K | 0.315 | 7.528 | 7.043 | 6.929-7.236 |
| 4 | J | 0.158 | 2.135 | 1.875 | 1.766-2.118 |
| 4 | K | 0.937 | 24.944 | 23.810 | 23.516-24.268 |
| 4 | J+K | 0.970 | 23.872 | 24.649 | 22.271-24.871 |

These short timings are sensitive to host load. J, K and J+K perform
different requested work; their ratio is not a same-method speedup.
No comparison to another backend, large-crystal scaling, SCF acceleration
or physical source convergence is claimed.

Independent references assemble raw fitting blocks from analytic Gaussian
Fourier/Boys sums and from libint, then use NumPy positive-space factors
and contractions. Across all nine workers and both references, maximum
Fock entry error is 3.731e-14, component-energy error 2.310e-14 hartree and
electronic-energy error 2.221e-14 hartree. Both references use the same
finite images, reciprocal supports and numerical split as the requested
source; agreement is not a complete-basis or infinite-source statement.

`tests/test_manta_periodic_ri_targets.py` also covers truncated/empty
projectors, zero source weights, each target's weighted energy derivative
and reciprocity, exact decomposition of joint J/K, all five simultaneous
resource bounds and one-short refusal before numerical input copies.
Image gauges use the shared density-push/operator-pull API; owned complex
exports also cross the landed `FockBuild` interface. Source flags stay
separate from density and frame identities even when responses are zero.
Eighteen saved prechange joint J/K states retain their arrays, scalar
receipts, fixed-projector identities and admission counts bitwise.

Two additional trial-density diagnostics combine RI-J with exact reciprocal
K, and exact reciprocal J with RI-K. Independent Gaussian sums check the
completed matrices/energies and their weighted energy derivatives, retaining
the opposite transfer conventions and each component's diagonal support.
This diagnostic composition has no production Python aggregate-memory or
SCF admission claim.

This increment provides independently selectable fitted components.
Mixed exact/RI HF or KS routing, physical range-separated fitting,
method-owned exchange corrections, BIPOLE/CRYSTAL source qualification and
symmetry reduction each retain their separate gates.
