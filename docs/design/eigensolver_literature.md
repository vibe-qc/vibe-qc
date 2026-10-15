# Eigensolver literature assessment

Scope: extremal and interior Hermitian problems relevant to AO SCF, k-point
SCF and Hermitian response/CI. Primary papers and authoritative implementation
documentation were checked through 2026-10-05. This is a scoped assessment,
not an exhaustive bibliography or a performance benchmark of external codes.
Publication DOIs below were matched to publisher records. arXiv identifiers
are explicitly preprints unless a journal publication is listed.

## Foundational methods and workload implications

Let n be dimension, k requested roots, m retained search dimension and b block
width. Memory estimates below exclude storage of the input operator. The
cost comparisons are algorithmic assessments; actual crossovers must be
measured with vibe-qc operators and occupations.

| Method and primary source | Iteration, storage and requirements | Assessment for vibe-qc |
|---|---|---|
| Davidson, 1975, [10.1016/0021-9991(75)90065-0](https://www.sciencedirect.com/science/article/abs/pii/0021999175900650) | Expand with approximately preconditioned residuals; O(n m) vectors, projected dense solves; diagonal preconditioning exploits near-diagonal structure. | Retain for a small number of extremal CI/TDA roots and recycled SCF subspaces. AO Fock need not be diagonally dominant after orthogonalization. A poor diagonal can stagnate. |
| Sleijpen and Van der Vorst, 1996, [10.1137/S0895479894270427](https://epubs.siam.org/doi/10.1137/S0895479894270427) | Projected correction equation removes the current eigendirection. Inexact inner solves add operator applications but can handle a better local model; O(n m) outer storage. | A candidate for difficult interior roots when shifted preconditioners exist. The local JD implementation must be validated against the full recurrence before promotion. |
| Knyazev, 2001, [10.1137/S1064827500366124](https://epubs.siam.org/doi/10.1137/S1064827500366124) | Local Rayleigh-Ritz over current, preconditioned residual and previous blocks; O(n b) storage with small problems of order 3b. SPD preconditioning underpins the usual convergence theory. | Useful for extremal Hermitian definite problems and clustered roots with block/locking safeguards. An indefinite inverse around an interior shift is not the same preconditioner contract. |
| Stewart, 2002, [10.1137/S0895479800371529](https://epubs.siam.org/doi/10.1137/S0895479800371529) | Krylov-Schur restarts retain selected Schur information; O(n m) storage and operator applications per expansion. | A better established restart architecture than the current unrestarted Python Lanczos. Interior solves still need harmonic extraction, a filter or shifted linear solves. |
| Zuev et al., 2015, [10.1002/jcc.23800](https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.23800) | Interior Davidson and GPLHR methods demonstrated on EOM-CC ionization/excitation workloads; targeted transitions also considered. | Evidence concerns specific operators and preconditioners, including non-Hermitian EOM-CC. It does not establish universal SCF performance or justify calling every response operator Fock. |
| Vecharynski, Yang and Xue, 2017, [10.1137/15M1027413](https://epubs.siam.org/doi/10.1137/15M1027413), [full preprint](https://arxiv.org/html/1506.06829) | GPLHR uses block locally harmonic Schur iteration with a preconditioner for the shifted pencil and bounded residual/direction blocks; no exact shift inverse required. | Full generalized non-Hermitian algorithm is materially different from the present real harmonic-Davidson variant. Completion would require new generalized Schur, projection and restart evidence, not a small patch. |
| Polizzi, 2009, [10.1103/PhysRevB.79.115112](https://doi.org/10.1103/PhysRevB.79.115112); Tang and Polizzi, 2014, [10.1137/13090866X](https://epubs.siam.org/doi/10.1137/13090866X) | FEAST approximates a spectral projector with contour quadrature and repeated shifted linear solves, then Rayleigh-Ritz. Memory includes factorizations or inner solves and interval subspaces. Quadrature nodes/intervals admit parallel execution. | Appropriate when many roots in an interval justify solve infrastructure. A matvec alone is insufficient for efficient application of the rational filter. Interval capacity, quadrature accuracy and completeness checks matter. |
| Winkelmann, Springer and Di Napoli, 2019, [10.1145/3313828](https://doi.org/10.1145/3313828) | ChASE uses Chebyshev filtered subspace iteration, spectral estimates and optimized polynomial degrees. Roughly degree times b operator-vector applications per filter sweep; O(n b) vectors; block products suit GPUs. | Attractive for repeated dense Hermitian SCF problems with a small spectral fraction and reusable vectors. When most orbitals are needed, dense diagonalization may win. Generalized AO reduction and overlap cost remain. |

Block methods should request guard vectors when the root boundary intersects a
cluster. Locking should be based on residuals and orthogonality, with attention
to drift after restart. Eigenvalues within a cluster do not determine a unique
set of orbital columns. Warm starts should recycle a subspace in the current
metric frame; basis/geometry changes require an explicit transport map. These
are established numerical practices, not new algorithms proposed here.

## Authoritative implementation evidence

[SLEPc Krylov-Schur documentation](https://slepc.upv.es/release/manualpages/EPS/EPSKRYLOVSCHUR.html)
describes thick-restart Lanczos behavior for Hermitian problems, harmonic
extraction, filtering and inertia-based spectrum slicing. Inertia provides
root-count information that residuals alone do not. Its distributed-memory
design is relevant to a future parallel operator interface; it is not an
existing vibe-qc dependency.

[PRIMME documentation](https://www.cs.wm.edu/~axstat/software/doc/intro.html)
describes generalized Hermitian problems, block correction equations,
initial subspaces and GPU execution. It is a useful contract/reference target
for a later optional backend evaluation. No dependency, binding or redistribution
is introduced in this increment.

[ChASE's maintained project description](https://www.fz-juelich.de/en/jsc/about-us/structure/simulation-and-data-labs/sdl-quantum-materials/software-development/chase)
describes real/complex support, sequence recycling, generalized-problem
reduction and distributed GPU implementations. Those capabilities motivate
block application and explicit frame metadata. Reported hardware speedups
are not transferable to this codebase without end-to-end measurements.

## Developments checked in 2025 and 2026

- Zhou, 2025, [arXiv:2509.00495](https://arxiv.org/abs/2509.00495), is an MSc
  thesis on mixed-precision Jacobi diagonalization. Its low-precision starting
  decomposition and higher-precision correction target full dense spectra.
  It is not evidence for matrix-free interior SCF roots. No implementation
  is proposed from this source.
- Di Napoli, Richefort and Wu, April 2026 revision,
  [arXiv:2601.10557v2](https://arxiv.org/abs/2601.10557v2), extends ChASE to
  structured pseudo-Hermitian Hamiltonians using symmetry and an oblique
  Rayleigh-Ritz construction. The preprint record says intended submission to
  SISC. The associated May 2026 [software release](https://zenodo.org/records/20038107)
  has DOI `10.5281/zenodo.20038107`. This does not make arbitrary CC Jacobians
  Hermitian; a future response interface must retain the relevant structure.
- Jia and Liu, May 2026, [arXiv:2605.12846](https://arxiv.org/abs/2605.12846),
  investigate refined projection and removal of spurious Ritz values in a
  filtered Hermitian interval solver. This preprint strengthens the case for
  explicit root-selection diagnostics; it is not a verified replacement for
  the present harmonic-Davidson implementation.
- Gubler et al., August 2026,
  [arXiv:2608.24529](https://arxiv.org/abs/2608.24529), study randomized block
  Davidson for plane-wave DFT. Sketching reduces orthogonalization work but
  produces a generalized projected problem. Their reported SCF advantage from
  sketching is limited when wanted-state count grows. Treat this as a preprint
  and a future benchmark candidate, not a reason to add randomized arithmetic
  to small dense AO solves.

## Proposals selected for this repository

The selected change is architectural and diagnostic: enforce real/complex,
metric, representation and option contracts before adding algorithms. Retain
dense as the complete-spectrum and interior reference; retain Davidson for
current validated partial-spectrum workloads. Keep harmonic-Davidson accurately
named and experimental. Recompute residuals independently and never treat them
as proof that no closer root exists.

The next implementation increment supplies true complex matrix-free Davidson
and real/complex block callback dispatch using the existing recurrence, with
related-operator recycling tests. Deferred experiments, in order: custom
projected preconditioners; then Krylov-Schur or an optional established eigensolver backend for difficult
interior problems. Filtered subspace iteration becomes attractive only after
measuring large repeated matrices and the required orbital fraction. Contour
methods require a shifted-solve contract and memory accounting first. GPU/MPI
support needs device/distribution ownership, synchronization and reduction
contracts; accepting an array-like object is not accelerator support.
