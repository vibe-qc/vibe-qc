# Eigensolver & SCF Solver Framework

vibe-qc ships a unified eigensolver framework for all diagonalization
and SCF convergence needs across molecular, periodic, semiempirical,
and post-HF methods.

## Quick Start with SCF

```python
from vibeqc import run_job, Molecule, Atom

mol = Molecule([Atom(8, [0, 0, 0]),
                Atom(1, [0, 1.43, -0.98]),
                Atom(1, [0, -1.43, -0.98])])  # coordinates in bohr

# Use LOBPCG (fastest iterative solver for large basis sets)
result = run_job(mol, basis="def2-svp", method="rhf", solver="lobpcg")

# Use Davidson (robust default for diagonally dominant matrices)
result = run_job(mol, basis="def2-svp", method="rhf", solver="davidson")

# Use dense exact diagonalization (default, fastest for small matrices)
result = run_job(mol, basis="def2-svp", method="rhf", solver="dense")
```

## Quick start ;  keyword selection

```python
from vibeqc.solvers.eigensolver import (
    EigenProblem, SolverOptions, solve_eigenproblem,
    SCFStep, solve_scf_step,
)

# Standard Hermitian eigenproblem
problem = EigenProblem(matrix=Fock_matrix, n=n_basis)
options = SolverOptions(n_roots=5, tol=1e-10)
result = solve_eigenproblem(problem, options, solver="lobpcg")

# SCF Fock diagonalization with orthogonalizer
step = SCFStep(fock=F, X=orthogonalizer, n_occ=n_occ)
result = solve_scf_step(step, solver="davidson")
```

## Partial-spectrum Davidson controls

`SolverOptions.n_roots` requests how many eigenpairs to return. `n_guess`
controls the starting subspace for both `davidson` and `hermitian_davidson`
when explicit `guess_vectors` are absent:

| `n_guess` | Meaning |
|---|---|
| `0` (default) | Native automatic size: `max(n_roots + 5, 2*n_roots)`, capped at the matrix dimension |
| Positive integer | Explicit initial size, at least `n_roots` |
| Negative or positive but smaller than `n_roots` | Rejected before solving |

For example, after constructing `problem` with at least four rows:

```python
options = SolverOptions(n_roots=4, n_guess=8, tol=1e-8)
result = solve_eigenproblem(problem, options, solver="hermitian_davidson")
if not result.converged:
    raise RuntimeError("Requested eigenpairs did not converge")
print(result.eigenvalues, result.residuals)
```

`max_subspace` controls later expansion/collapse and is separate from the
starting size. `guess_vectors` supplies a warm-start subspace instead of the
automatic diagonal guess. Partial-spectrum convergence concerns the requested
roots, not the entire spectrum. When diagnosing a solve, also check
`A @ X - X * eigenvalues` and `X.conj().T @ X` against the requested tolerance
and identity. The complex-Hermitian path uses conjugate inner products and
reorthogonalizes new corrections against both retained and same-block vectors.
These controls do not guarantee convergence for every matrix or establish
which SCF electronic state is physically appropriate.

### Molecular RHF HCORE seed completion

The native molecular RHF driver's `RHFOptions.davidson.n_eig` is a separate
control from the generic `SolverOptions.n_roots` interface above. When that
driver uses Davidson, `n_eig=0` requests the full retained space; a positive
request smaller than the occupied space is rejected.

For an HCORE starting density, a partial spectrum can end inside a degenerate
frontier. If missing frontier partners cannot be excluded, the driver
completes the seed diagonalization and retains that complete orbital frame
for the first Fock solve. The subsequent iterative Davidson settings stay
unchanged. Equal fractional seed occupations cover only a frontier block
with total width at most `1e-8` hartree and preserve the electron count;
gapped seeds keep integer occupations. This does not change final SCF
occupation rules or establish a global ground state. See the
[initial-guess reference](initial_guess.md#hcore-core-hamiltonian-guess).

## Available solver keywords

| Keyword | Algorithm | Backend | Best for |
|---|---|---|---|
| `davidson` | Block Davidson (Liu 1978) | C++ | Extremal eigenvalues of diagonally-dominant matrices; SCF default |
| `block_davidson` | Alias for `davidson` | C++ | Same |
| `lobpcg` | LOBPCG (Knyazev 2001) | C++ | Faster convergence, lower memory; ill-conditioned matrices |
| `dense` | `np.linalg.eigh` / `scipy.linalg.eigh` | NumPy/SciPy | Small matrices (< 200); reference implementation |
| `eigh` | Alias for `dense` | NumPy | Same |
| `lanczos` | Lanczos with full reorthogonalization | Python | Matrix-free, moderate accuracy; exploratory |
| `jd` | Jacobi-Davidson (Sleijpen 1996) | C++ | Robust for interior eigenvalues; MINRES correction; experimental |
| `jacobi_davidson` | Alias for `jd` | C++ | Same |
| `gplhr` | GPLHR-derived block harmonic iteration (Zuev 2015) | C++ | Interior roots of real symmetric matrices; experimental |
| `hermitian_davidson` | Complex-Hermitian block Davidson | C++ | k ≠ Γ periodic Fock matrices |

### Experimental interior solvers

Pass an `InteriorEigenProblem` with the target shift to `jd` or `gplhr`:

```python
from vibeqc.solvers.eigensolver import InteriorEigenProblem

interior = InteriorEigenProblem(problem=problem, sigma=0.4248)
result = solve_eigenproblem(
    interior, SolverOptions(n_roots=1, tol=1e-8, max_iter=500), solver="gplhr"
)
```

GPLHR remains experimental. Its implementation uses harmonic extraction and
signed shifted-diagonal preconditioning inspired by Zuev et al. (2015), but
retains a growing/restarted harmonic-Davidson search space. It does not
implement the paper's full locally harmonic recurrence with residual-like
blocks and a previous-direction block. Candidates and retained vectors are
selected by distance of the harmonic values to the shift; residual size
does not change that ordering. Returned eigenvalues are Rayleigh quotients
of normalized vectors.
The paper discusses approximating `(A - sigma I)^(-1)` and uses per-root
Davidson denominators `(D - rho_i I)^(-1)` in its implementation. This
variant uses the fixed target-shift approximation `(D - sigma I)^(-1)`.

The default working block includes guard roots, `max(2*n_roots, 4)`, capped
at the matrix dimension. `block_size` controls that block (at least
`n_roots`); `max_subspace` controls the retained search-space budget and
defaults to `min(6*block_size + 10, n)` using the effective block size.
`preshift` bounds the denominator magnitude while preserving its sign.

Earlier versions could discard the target vectors at restart and fail on
weakly coupled near-diagonal matrices even after 500 iterations. The signed
preconditioner, target-preserving restarts, and guard-root expansion repair
that behaviour for regression cases, but do not guarantee convergence or
selection of the nearest root on every matrix. On iteration exhaustion or
stagnation, the Python GPLHR interface raises `RuntimeError` with the shift,
iteration count, residual and a suggestion to try `solver="jd"` or increase
the search-space/iteration budget. Successful results include residual norms.
JD continues to report convergence through `result.converged`.
GPLHR also rejects a singular harmonic pencil (for example, a diagonal
matrix with an eigenvalue exactly at the shift) with a diagnostic error;
it must not drop that root and return a neighbouring eigenpair.

Run `examples/experimental/interior_eigensolvers.py` for a deterministic
comparison against dense reference roots on six matrices of sizes 60, 120
and 300 with off-diagonal coupling scales 0.01 and 0.05. Both solvers remain
experimental; this comparison is a diagnostic, not a production guarantee.

## Solver capability matrix

| Solver | Extremal | Interior | Generalized | Block | Matrix-free | SCF step |
|---|---|---|---|---|---|---|
| `davidson` | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| `lobpcg` | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `dense` | ✅ | ✅¹ | ❌ | N/A | ❌ | ✅ |
| `lanczos` | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `jd` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `gplhr` | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `hermitian_davidson` | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |

¹ Via scipy `subset_by_index`. Generalized and interior problems auto-reduce
via canonical orthogonalization / spectral transformation.

## Architecture

```
SolverStrategy (ABC)
├── DenseSolver           → np.linalg.eigh / scipy.linalg.eigh
├── DavidsonSolver        → _vibeqc_core.davidson_solve (C++)
├── HermitianDavidsonSolver → _vibeqc_core.davidson_solve_hermitian (C++)
├── LOBPCGSolver          → _vibeqc_core.lobpcg_solve (C++)
├── JDSolver              → _vibeqc_core.jd_solve (C++, experimental)
├── GPLHRSolver           → _vibeqc_core.gplhr_solve (C++, experimental)
└── LanczosSolver         → pure Python, tridiagonal reduction
```

Registration is automatic: any `SolverStrategy` subclass is added to the
registry at class-definition time via `__init_subclass__`. Aliases are
supported via the `_aliases` class attribute.

## Problem descriptors

- **`EigenProblem`** ;  Standard Hermitian: H x = λ x. Accepts explicit
  matrix OR matrix-free `matvec` callable.
- **`GeneralizedEigenProblem`** ;  Generalized: H x = λ S x. Auto-reduces
  via canonical orthogonalization when the solver doesn't support it.
- **`InteriorEigenProblem`** ;  Interior eigenvalues near shift σ.
  Auto-reduces via shift-and-invert spectral transformation.
- **`SCFStep`** ;  Fock-matrix diagonalization step in an orthogonalized
  basis. Carries the orthogonalizer X, occupation count, and optional
  previous MO coefficients for warm-start/subspace recycling.

## Extending ;  adding a new solver

```python
class MySolver(SolverStrategy):
    _aliases = ["mysolver", "my"]
    supports_extremal = True
    supports_matrix_free = True

    def solve(self, problem, options):
        # ... your algorithm ...
        return SolverResult(eigenvalues=values, eigenvectors=vectors,
                            n_iter=iterations, converged=converged)
```

After class definition, it's immediately available via
`solve_eigenproblem(..., solver="mysolver")`.

## SCF accelerators (separate subsystem)

SCF convergence acceleration (DIIS, EDIIS, KDIIS, ADIIS, level shift,
dynamic damping, Newton/SOSCF/TRAH) is a **separate** subsystem from the
diagonalization framework. Accelerators operate on the Fock/density
matrices *between* diagonalizations. They are selected via
`opts.scf_accelerator` and the second-order threshold fields. See
`docs/user_guide/scf_convergence.md` for the full accelerator guide.

## References

- Davidson, E. R. *J. Comput. Phys.* 17, 87-94 (1975).
  DOI: 10.1016/0021-9991(75)90065-0
- Liu, B. *NRCC Workshop on Numerical Algorithms in Chemistry*,
  LBL-8158 (1978). Block generalization of Davidson.
- Knyazev, A. V. *SIAM J. Sci. Comput.* 23, 517-541 (2001).
  DOI: 10.1137/S1064827500366124 ;  LOBPCG.
- Kresse, G. & Furthmüller, J. *Phys. Rev. B* 54, 11169 (1996).
  VASP-style blocked Davidson for Kohn-Sham.
- Saunders, V. R. & Hillier, I. H. *Int. J. Quantum Chem.* 7,
  699-705 (1973). DOI: 10.1002/qua.560070505 ;  Level shifting.
- Pulay, P. *Chem. Phys. Lett.* 73, 393-398 (1980). DIIS.
- Kudin, K. N., Scuseria, G. E. & Cancès, E. *J. Chem. Phys.* 116,
  8255 (2002). EDIIS.
- Hu, X. & Yang, W. *J. Chem. Phys.* 132, 054109 (2010). ADIIS.
- Kollmar, C. *Int. J. Quantum Chem.* 62, 617-637 (1997).
  DOI: `10.1002/(SICI)1097-461X(1997)62:6<617::AID-QUA5>3.0.CO;2-Z` ;  KDIIS.
- Neese, F. *Chem. Phys. Lett.* 325, 93-98 (2000). SOSCF.
- Helmich-Paris, B. *J. Chem. Phys.* 156, 054104 (2022). TRAH.
- Sleijpen, G. L. G. & Van der Vorst, H. A. *SIAM J. Matrix Anal.*
  Appl. 17, 401-425 (1996). DOI: 10.1137/S0895479894270427 ;  Jacobi-Davidson.
- Zuev, D., Vecharynski, E., Yang, C., Orms, N. & Krylov, A. I.
  *J. Comput. Chem.* 36, 273-284 (2015).
  DOI: [10.1002/jcc.23800](https://doi.org/10.1002/jcc.23800) ; GPLHR.
  Output citation key: `zuev_gplhr_2015`.
