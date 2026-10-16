---
myst:
  html_meta:
    "description": "Tutorial: the vibe-qc shared Hermitian operator contract and its eigensolvers. Cold and warm Davidson solves, complex callbacks, a factorised AO operator with an overlap metric, diagnostics, nearest-root selection, the interior solvers and the requests the framework refuses, built on examples/tutorials/unified_operators.py."
    "og:title": "vibe-qc tutorial: eigensolvers and the operator contract"
---

# Eigensolvers and the operator contract

Every SCF iteration in vibe-qc ends with a generalized Hermitian eigenvalue
problem, $\mathbf F\mathbf C = \mathbf S\mathbf C\boldsymbol\varepsilon$, and
the response, CI and band-structure codes solve their own Hermitian
problems. Since v0.17.16 these problems go through one boundary: an
operator object that says what it is (a Fock matrix, a Kohn-Sham matrix, a
response or CI Hamiltonian), how to apply it (an explicit matrix or a
callback), in which metric, and at which k point and spin. This tutorial
runs the operator examples that ship with vibe-qc, explains each printed
line, and shows the requests the framework refuses and why.

```{admonition} Status
:class: note

The operator contract and the `dense`, `davidson` and `hermitian_davidson`
solvers are production: every migrated SCF driver uses them. `lobpcg` is
production for explicit real problems outside `run_job`. Jacobi-Davidson
(`jd`), the harmonic-Davidson variant (`harmonic_davidson`, also reachable
under its historical name `gplhr`) and `lanczos` are experimental. The
label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

**Prerequisites.** A working vibe-qc build and basic linear algebra. The
[solver guide](../user_guide/solver_framework.md) is the reference for every
class and option used here. The numbers below were measured with the vibe-qc
0.17.17 development build (main, 2026-10-14); the timings are microseconds
on a laptop and only show the ratio between cold and warm solves.

## Run the example

The script needs no input and writes no files:

```sh
python examples/tutorials/unified_operators.py --size 80
```

```text
cold: seconds=0.000511, iterations=6, vector_applications=27, eigenvalue_error=5.107e-15
warm: seconds=0.000116, iterations=1, vector_applications=9, eigenvalue_error=5.107e-15
Residual convergence is not a completeness certificate: False
complex callback: iterations=6, block_calls=8, vector_applications=29, eigenvalue_error=3.553e-15
factorized AO callback: vector_applications=23, eigenvalue_error=3.553e-15, AO_residual=7.131e-12, projected_residual=6.927e-12
Complex AO diagnostics: {'converged': True, 'eigenvalues': [-0.7217287688002687, -0.06960895668316142], 'residuals': [1.97e-16, 2.63e-16], 'projected_residuals': [2.29e-16, 2.60e-16], 'residual_frame': 'original', 'orthogonality_error': 8.35e-16, 'retained_rank': 3, 'selection_verified': True, ...}
Same-subspace principal cosines: [1. 1.]
Verified nearest roots: [0.40667953 0.46573986]
CI roots: [-1.00317978 -0.49682022]
```

The diagnostics line is abbreviated here. The sections below take the
output one block at a time.

## A matrix-free Davidson solve, cold and warm

The first block builds an 80 by 80 test matrix, a diagonal from -2 to 3 with
a small symmetric perturbation, and hands the solver only its action:

```python
from vibeqc.solvers.eigensolver import SolverOptions
from vibeqc.solvers.operators import HermitianOperator, OperatorMetadata, solve_operator

op = HermitianOperator(
    matvec=lambda v: matrix @ v,
    matmat=lambda v: matrix @ v,
    n=n, diagonal=matrix.diagonal(),
    metadata=OperatorMetadata(kind="response", basis_id="example-response"),
)
opts = SolverOptions(n_roots=3, tol=1e-9, n_guess=8)
cold = solve_operator(op, opts, solver="davidson")
warm = solve_operator(op, opts, solver="davidson", previous=cold)
```

The cold solve takes six iterations and 27 operator applications for the
three lowest roots; the warm solve, started from the previous eigenvectors,
needs one iteration and nine applications. Both agree with the dense
eigenvalues to 5e-15. Three details matter:

- **The diagonal is required.** Davidson preconditions its correction
  vectors with the diagonal, and a callback cannot reveal its diagonal
  cheaply, so the operator must supply it. Without it the solve is refused
  rather than probed.
- **Warm starts must match the frame.** `previous=` is accepted only if the
  earlier result belongs to the same operator kind, basis, spin, k point
  and block. Across SCF iterations the Fock matrix changes but the frame
  stays, which is exactly the case warm starts serve.
- **Converged is not the same as complete.** The third line prints
  `selection_verified`, which is `False`: small residuals prove that the
  returned vectors are eigenvectors, not that they are the three lowest
  ones. An iterative solver can converge onto an eigenvector while missing
  a lower one that its search space never touched. Only a dense solve,
  which sees the whole spectrum, sets the flag. Pass
  `require_selection=True` when a missing root would be a wrong answer;
  an iterative backend then refuses instead of guessing.

## Complex operators and block callbacks

Bloch Hamiltonians at a general k point are complex Hermitian. The second
block adds an antisymmetric imaginary part and supplies only a block action,
`matmat`, which the complex solver calls with several vectors at once:

```text
complex callback: iterations=6, block_calls=8, vector_applications=29, eigenvalue_error=3.553e-15
```

The callback was entered 8 times and applied to 29 vectors in total. Each
Python crossing is expensive compared with a matrix-vector product in native
code, so block actions keep the number of crossings small. The real
`davidson` backend refuses complex input; `hermitian_davidson` handles both.

## An AO operator with an overlap metric

In a Gaussian basis the Fock matrix lives in the non-orthogonal AO basis
with overlap $\mathbf S$. The third block builds a factorised operator
$\mathbf A = \operatorname{diag}(\mathbf h) + \mathbf B\mathbf B^\dagger$
whose action never forms $\mathbf A$, together with an orthogonaliser
$\mathbf X$ ($\mathbf X^\dagger\mathbf S\mathbf X = \mathbf 1$) and the diagonal
of $\mathbf X^\dagger\mathbf A\mathbf X$, which the provider computes from its
own factors:

```text
factorized AO callback: vector_applications=23, eigenvalue_error=3.553e-15, AO_residual=7.131e-12, projected_residual=6.927e-12
```

Two residuals are reported because there are two frames. The projected
residual is measured in the orthogonal frame where the solver iterates; the
AO residual $\lVert \mathbf A\mathbf c - \varepsilon\mathbf S\mathbf c\rVert$
is measured in the original frame. They agree here because $\mathbf X$
keeps the full rank of $\mathbf S$. When canonical orthogonalisation drops
near-linearly-dependent directions they can differ, and the AO residual is
the one that tells you whether the original problem was solved.

## Diagnostics for a complex Kohn-Sham matrix

The fourth block solves a three-function complex Kohn-Sham matrix with an
explicit overlap and prints the portable diagnostic record of
`vibeqc.output.eigensolver.eigensolver_diagnostics`:

| Field | Value | Meaning |
|---|---|---|
| `residuals` | about 2e-16 | AO-frame residual norms per root |
| `projected_residuals` | about 2e-16 | orthogonal-frame residual norms |
| `residual_frame` | `original` | which frame `residuals` refers to |
| `orthogonality_error` | 8e-16 | Frobenius error of $\mathbf C^\dagger\mathbf S\mathbf C - \mathbf 1$ |
| `retained_rank` | 3 | directions kept by the orthogonaliser |
| `selection_verified` | `True` | a dense solve enumerated the spectrum |
| `operator_metadata` | kind, spin, k point, basis id | what was solved, for the log |

`subspace_overlap` then compares two solutions as subspaces, through the
principal cosines in the $\mathbf S$ metric. Comparing a subspace with itself
gives cosines of one. This is the right test for degenerate orbitals, whose
individual vectors have no fixed orientation.

## Roots nearest a shift

The fifth block asks for the two roots closest to $\sigma = 0.4248$ with
`require_selection=True`. The dense solver enumerates the full spectrum
and selects the two nearest values, 0.4067 and 0.4657, so the selection is
certified. No shift-and-invert step is involved, so a root exactly at
$\sigma$ is as well defined as any other.

For large matrices, where a dense solve is too expensive, the experimental
interior solvers aim at the same roots iteratively. On test matrices of
increasing size and coupling the solver writer measured:

| Size | Coupling | Jacobi-Davidson iterations | Harmonic-Davidson iterations |
|---:|---:|---:|---:|
| 60 | 0.01 | 87 | 7 |
| 60 | 0.05 | 75 | 35 |
| 120 | 0.01 | 76 | 17 |
| 120 | 0.05 | 119 | 483 |
| 300 | 0.01 | 88 | 82 |
| 300 | 0.05 | 306 | not converged in 500 |

Both converged roots matched the nearest dense eigenvalue to 1e-15, but
neither can certify its selection, and the harmonic variant stalled on the
largest, most strongly coupled case. It raises instead of returning an
unconverged value:

```text
RuntimeError: Experimental GPLHR did not converge 1 root(s) near sigma=0.4248 after
500 iterations (maximum residual=5.066e-02, tol=1.000e-08). Try solver='jd' or
increase max_subspace/max_iter.
```

## The kind keeps the mathematics honest

The last block solves a two-determinant CI matrix as a `HermitianOperator`
with `kind="ci"`. Trying to wrap the same matrix as a `FockOperator` is
refused:

```text
ValueError: A CI or response operator is not a Fock operator
```

The kind decides which checks apply. A Fock or Kohn-Sham operator may carry
an overlap, spin and k point and is solved for orbitals; a CI or response
operator is not, and a non-Hermitian operator such as a Casida matrix or a
coupled-cluster Jacobian is refused outright and needs its own solver.

## What else is refused

The framework refuses an unsupported request instead of silently doing
something else. A selection of the messages, measured on this build:

| Request | Message |
|---|---|
| `davidson` with a complex matrix | `davidson does not support complex operators` |
| `davidson` with `which="SM"` | `davidson does not support selector SM` |
| `davidson` with an interior shift | `davidson does not support an interior shift` |
| `require_selection=True` on `davidson` | `Requested-root selection is not certified by this iterative backend` |
| `lobpcg` with a callback | `lobpcg does not support matrix-free/sparse storage without materialization` |
| a callback Davidson below its restart floor | `Davidson below its restart floor would materialize the callback operator` |
| `previous=` from another frame | `Previous subspace belongs to a different operator frame` |
| `run_job(solver="lobpcg")` | `run_job does not implement LOBPCG SCF diagonalization; use solver='dense' or solver='davidson'` |
| `run_job(solver="jd")` | `run_job: solver='jd' is not recognised. Supported: 'dense', 'davidson'.` |

`run_job(solver="lobpcg")` deserves a note: older versions accepted it and
silently ran Davidson. It now raises, because a result labelled LOBPCG must
have been computed by LOBPCG.

## Theory

**Davidson iteration.** For the lowest roots of a large Hermitian matrix
$\mathbf A$, Davidson's method grows a search space $\mathbf V$, solves the
small projected problem $\mathbf V^\dagger\mathbf A\mathbf V\,\mathbf y =
\theta\mathbf y$, forms the residual $\mathbf r = \mathbf A\mathbf u -
\theta\mathbf u$ of each Ritz vector $\mathbf u = \mathbf V\mathbf y$ and
adds the preconditioned correction

$$
\mathbf t = (\theta\mathbf 1 - \mathbf D)^{-1}\mathbf r ,
\qquad \mathbf D = \operatorname{diag}(\mathbf A),
$$

orthogonalised against $\mathbf V$ ({vibeqc-cite}`davidson_iterative_1975`).
The method converges fast when $\mathbf A$ is diagonally dominant, which
Fock matrices in a good orbital basis are, and restarts from the current
Ritz vectors when the space reaches its size limit. LOBPCG
({vibeqc-cite}`knyazev_lobpcg_2001`) keeps a fixed three-block space of the
current vectors, their preconditioned residuals and the previous step.

**Generalized problems.** $\mathbf F\mathbf C = \mathbf S\mathbf C
\boldsymbol\varepsilon$ is reduced to standard form with an orthogonaliser,
$\mathbf F' = \mathbf X^\dagger\mathbf F\mathbf X$, $\mathbf C = \mathbf X
\mathbf C'$. Canonical orthogonalisation takes $\mathbf X = \mathbf U
\mathbf s^{-1/2}$ from the eigenpairs of $\mathbf S$ and drops eigenvalues
below a threshold, which removes near-linear dependencies
({vibeqc-cite}`lowdin_ortho_1970`); the dropped count is the difference
between the basis size and `retained_rank`.

**Interior eigenvalues.** Rayleigh-Ritz converges to the ends of the
spectrum. Jacobi-Davidson ({vibeqc-cite}`sleijpen_jd_1996`) targets an
interior value by solving the projected correction equation
$(\mathbf 1 - \mathbf u\mathbf u^\dagger)(\mathbf A - \theta\mathbf 1)
(\mathbf 1 - \mathbf u\mathbf u^\dagger)\mathbf t = -\mathbf r$ approximately.
Harmonic Rayleigh-Ritz extracts approximations from the shifted problem
$(\mathbf A - \sigma\mathbf 1)^{-1}$ without inverting it, which favours the
eigenvalues nearest $\sigma$; vibe-qc's `harmonic_davidson` is a variant
in the family of GPLHR ({vibeqc-cite}`zuev_gplhr_2015`), not a full
implementation of the published algorithm.

## Variations

- Run `--size 120` for a larger matrix: the cold solve needs 29
  applications and the warm one still 9.
- Replace `solver="davidson"` with `solver="dense"` and compare
  `selection_verified`.
- Set `opts = SolverOptions(n_roots=3, tol=1e-9, n_guess=1)` and read the
  refusal: the guess space must hold at least as many vectors as roots.

## Pitfalls

- **Do not read convergence as completeness.** Use a dense reference or
  `require_selection=True` when a missed root matters.
- **Supply the diagonal of the frame you solve in.** For an AO callback
  with an orthogonaliser that is the diagonal of
  $\mathbf X^\dagger\mathbf A\mathbf X$, computed by the provider, not the AO
  diagonal.
- **Callbacks must not mutate their input.** The framework copies their
  output, but the state they capture (densities, coefficients) must stay
  fixed during a solve.

## See also

- [Eigensolvers and the operator contract](../user_guide/solver_framework.md),
  the reference for every class, option and driver.
- [SCF convergence](../user_guide/scf_convergence.md) for what happens
  around the eigensolver in each iteration.
- [Design: the operator contract](../design/operator_contract.md) and the
  [consumer inventory](../design/operator_consumers.md).
