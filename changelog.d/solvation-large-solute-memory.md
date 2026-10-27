### Fixed: solvated SCF and gradient memory no longer scales with the basis squared times the segments

A solvated SCF kept a stack of per-segment Coulomb operators
(`8 * n_segments * n_basis^2` bytes: 3 GB for a 30-atom molecule in def2-SVP,
16 GB for 50 atoms), a second copy of the cavity matrix for its LU
factorization, and the gradient built `(n, n, 3)` separation tensors on top. A
74-atom alkane (15 500 segments, STO-3G) peaked at 9.9 GB in the SCF; it now
peaks at 2.2 GB, and the gradient at 2.7 GB.

The factorization is done in place, the gradient sums over a block of segments at
a time (bit-identical to the unblocked sums), and the operator stack is kept only
up to the new `SolventModel(operator_cache_mb=1024)`. Above it the SCF streams the
potential and the Fock operator one segment at a time, which agrees with the
cached run to 2e-13 Ha (energy) and 6e-14 Ha/bohr (gradient) and costs one extra
integral pass per macro-iteration. Nothing changes below the budget except the
in-place factorization.
