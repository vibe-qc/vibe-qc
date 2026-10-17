---
myst:
  html_meta:
    "description": "The vibe-qc eigensolver framework: dense, Davidson, complex Hermitian Davidson, LOBPCG, Jacobi-Davidson and harmonic-Davidson solvers, the shared Hermitian operator contract (FockOperator, OperatorMetadata, solve_operator, diagnostics, residuals, warm starts), which SCF drivers use it, and the theory behind each solver."
    "og:title": "vibe-qc - eigensolvers and the operator contract"
---

# Eigensolvers and the operator contract

Every self-consistent-field iteration in vibe-qc ends in the same step: a
Hermitian matrix, usually a Fock or Kohn-Sham matrix in a nonorthogonal AO
basis, is diagonalised, and the lowest eigenvectors become the occupied
orbitals. This page documents the solvers that perform that step, the
Hermitian operator contract through which the SCF drivers, the response
code and the post-SCF semicanonicalisation reach them, and the theory of the
algorithms. SCF acceleration (DIIS and its relatives, damping, level shifts,
the second-order finalizers) is a separate subsystem that acts on the Fock
matrix *between* diagonalisations; see
[SCF convergence](scf_convergence.md). Occupation assignment is likewise not
part of eigensolution: a solver returns eigenpairs, and the method decides
which of them to fill.

```{admonition} Status: production (dense, Davidson, LOBPCG); experimental (Jacobi-Davidson, harmonic-Davidson, Lanczos)
:class: tip

The dense solver and the real and complex Davidson solvers are the
production diagonalisers of every molecular and periodic SCF route, covered
by the T1 release gate through the SCF tests they serve. LOBPCG is a
production eigensolver for explicit real symmetric problems through the
library API; no SCF driver routes through it, and `run_job(solver="lobpcg")`
raises. Jacobi-Davidson (`jd`), the GPLHR-derived harmonic-Davidson variant
(`harmonic_davidson`, alias `gplhr`) and the Python Lanczos reference are
**experimental**: `solver_capabilities(name).experimental` is true, no SCF
driver uses them, and their numbers are research output. The label
vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

## Choosing the solver in a job

For the molecular SCF drivers the choice is the `solver` keyword of
`run_job`, or equivalently the `use_davidson` switch on the options object:

```python
from vibeqc import run_job, Molecule, Atom, RHFOptions

mol = Molecule([Atom(8, [0, 0, 0]),
                Atom(1, [0, 1.43, -0.98]),
                Atom(1, [0, -1.43, -0.98])])          # bohr

result = run_job(mol, basis="def2-svp", method="rhf", solver="dense")     # default
result = run_job(mol, basis="def2-svp", method="rhf", solver="davidson")

opts = RHFOptions()
opts.use_davidson = True          # the same request on the options object
opts.davidson_min_dim = 100       # dense below this many AO functions
opts.davidson.conv_tol = 1e-7     # DavidsonOptions: residual threshold per root
opts.davidson.max_iter = 200
```

`run_job` accepts exactly `"dense"` and `"davidson"`. `solver="lobpcg"`
raises `NotImplementedError` ("run_job does not implement LOBPCG SCF
diagonalization; use solver='dense' or solver='davidson'"): earlier versions
silently ran Davidson for that request, and the C++ molecular SCF kernels
expose only the `use_davidson` switch. Any other spelling, including `"jd"`,
raises `ValueError`. The method-specific size threshold still applies: an
SCF that asks for Davidson on a basis smaller than `davidson_min_dim` (100
AO functions by default) solves densely. Periodic drivers carry the same
switch on `PeriodicRHFOptions.use_davidson` with `davidson_min_dim` and the
`davidson` controls; complex k-point Fock matrices go to the complex
Hermitian Davidson automatically. Open-shell Roothaan loops (ROHF/ROKS) and
the spin-locked or quadratic updates of Gamma UHF need the complete retained
spectrum for their occupation rules, so a Davidson request there with too
few roots is refused before any Fock build; see [ROHF and ROKS](rohf.md).

Davidson pays off when only the occupied space is wanted out of a large
basis and the Fock matrix changes little between iterations, because the
previous orbitals warm-start the next solve (one to three outer iterations
per SCF cycle is typical). Below a few hundred AO functions the dense solve
is faster and exact, and it is the only solver whose root selection is
certified (see [Residuals, selection and warm starts](#residuals-selection-and-warm-starts)).

## The solvers

All solvers are reachable by name through the library API. The table lists
the executable capabilities returned by
`vibeqc.solvers.operators.solver_capabilities(name)` on vibe-qc 0.17.15;
these describe what the strict entry point `solve_operator` actually
accepts, not aspirational flags.

| Name (aliases) | Algorithm, backend | Complex | Matrix-free | Selectors | Warm start | Interior shift | Status |
|---|---|---|---|---|---|---|---|
| `dense` (`eigh`) | full diagonalisation, `numpy.linalg.eigh` | yes | no | SA, LA, SM, LM, BE, nearest to `sigma` | no | yes, by enumeration | production |
| `davidson` (`block_davidson`) | blocked Davidson, C++ | no | yes (supplied diagonal) | SA, LA | yes | no | production |
| `hermitian_davidson` | blocked Davidson, complex Hermitian, C++ | yes | yes (supplied diagonal) | SA, LA | yes | no | production |
| `lobpcg` | LOBPCG, C++ | no | no | SA, LA | no | no | production (explicit problems; no SCF route) |
| `jd` (`jacobi_davidson`) | Jacobi-Davidson with MINRES correction equation, C++ | no | no | SA, LA | no | yes | experimental |
| `harmonic_davidson` (`gplhr`) | restarted harmonic-Davidson for roots near a shift, C++ | no | no | SA | no | yes, required | experimental |
| `lanczos` | Lanczos with full reorthogonalisation, Python | yes | yes | SA, LA | no | no | experimental |

`harmonic_davidson` and `gplhr` are two names for one solver class
(`get_solver("harmonic_davidson").name()` returns `"gplhr"`). The
implementation in `cpp/src/gplhr.cpp` uses the harmonic extraction and the
signed shifted-diagonal preconditioner of Zuev and co-workers
({vibeqc-cite}`zuev_gplhr_2015`) inside a growing, restarted Davidson
search space; it does not implement the paper's locally harmonic recurrence
with residual-like blocks and a previous-direction block. `harmonic_davidson`
is therefore the accurate name, `gplhr` is retained for compatibility, and
the citation records provenance rather than equivalence; the decision is
recorded in the [design document](../design/operator_contract.md#gplhr-decision).

Each solver accepts a subset of the `SolverOptions` fields beyond
`n_roots`, `which`, `sigma` and `tol`; a nondefault value of any other field
raises `NotImplementedError` at the strict entry point, so an option is
never silently ignored:

| Solver | Accepted options |
|---|---|
| `dense` | none further (`n_roots=0` means every retained root) |
| `davidson` | `max_iter`, `max_subspace`, `preshift`, `verbosity`, `n_guess`, `guess_vectors` |
| `hermitian_davidson` | `max_iter`, `max_subspace`, `preshift`, `n_guess`, `guess_vectors` |
| `lobpcg` | `max_iter`, `block_size`, `preshift`, `verbosity` |
| `jd` | `max_iter`, `verbosity` |
| `harmonic_davidson` | `max_iter`, `max_subspace`, `block_size`, `preshift`, `verbosity` |
| `lanczos` | `ncv` (at least `n_roots`) |

### Partial-spectrum Davidson controls

`n_roots` is the number of eigenpairs returned (`0` requests all retained
roots). `n_guess` sets the starting subspace of `davidson` and
`hermitian_davidson` when no `guess_vectors` are supplied:

| `n_guess` | Meaning |
|---|---|
| `0` (default) | automatic size `max(n_roots + 5, 2 n_roots)`, capped at the dimension |
| positive, at least `n_roots` | explicit initial size |
| negative, or positive but smaller than `n_roots` | rejected (`ValueError`) before solving |

`max_subspace` is the dimension at which the search space collapses to the
current Ritz vectors; it is separate from the starting size, and a value
below the native restart floor `min(n, 2 n_roots + 6)` is rejected.
`preshift` bounds the magnitude of the preconditioner denominator
$\theta - D_{ii}$ while keeping its sign. Partial-spectrum convergence is a
statement about the requested roots, not about the whole spectrum, and
these controls do not decide which SCF state is physically appropriate.

The native molecular RHF driver's `RHFOptions.davidson.n_eig` is a separate
control: `0` requests the full retained space and a positive request smaller
than the occupied space is rejected. For an HCORE start whose partial
spectrum could end inside a degenerate frontier, that driver completes the
seed diagonalisation and keeps the complete orbital frame for the first
Fock solve; the
[initial-guess reference](initial_guess.md#hcore-core-hamiltonian-guess)
describes the equal-fraction seed occupations that follow.

## The operator contract

The module `vibeqc.solvers.operators` is the boundary between a method that
*builds* an operator and a solver that *diagonalises* it. The builder owns
integrals, densities, image sums and XC approximations; the operator owns one
fixed, immutable representation; the solver owns its iteration space; the
occupation layer owns electron counts and smearing. None substitutes for
another. The full design is in
[the operator contract](../design/operator_contract.md); this section is
the user's view.

### Constructing an operator

```python
import numpy as np
from vibeqc.solvers.operators import FockOperator, HermitianOperator, OperatorMetadata

# A dense AO Fock matrix with its overlap: C = X Y with X^H S X = I is built
# internally by canonical orthogonalisation (overlap cutoff 1e-12).
F = np.array([[1.0, 0.2j], [-0.2j, 2.0]])
S = np.array([[1.0, 0.1], [0.1, 1.0]])
ks = FockOperator(matrix=F, overlap=S,
                  metadata=OperatorMetadata(kind="ks", spin="restricted",
                                            kpoint=(0.25, 0.0, 0.0)))

# A matrix-free operator: supply the action and its diagonal.
n, A = 80, np.diag(np.linspace(-2.0, 3.0, 80))
op = HermitianOperator(matvec=lambda v: A @ v, matmat=lambda V: A @ V,
                       n=n, diagonal=A.diagonal(),
                       metadata=OperatorMetadata(kind="response", basis_id="tda"))
```

`HermitianOperator` takes **one** representation: a dense array, a SciPy
sparse matrix, or callbacks (`matvec` for one vector, `matmat` for a block)
with an explicit dimension and dtype; a matrix together with a callback is
rejected. Dense and sparse input are copied into immutable storage and
checked for finiteness and Hermiticity; callbacks promise a fixed linear
Hermitian action for the whole solve and are never probed with unit vectors.
`FockOperator` is the one-particle specialisation: its `kind` must be
`"fock"`, `"ks"` or `"one_electron"`, so a CI Hamiltonian or a response
matrix (`kind="ci"`, `"response"`) cannot be passed off as a Fock matrix,
while `kind="casida"` or any non-Hermitian label is refused outright.

Metric and frame. Supply `overlap` alone for canonical rank reduction with
`overlap_cutoff` (default `1e-12`, relative to the largest overlap
eigenvalue; an indefinite overlap is an error, not a silent truncation), or
`overlap` together with a rectangular `X` that already satisfies
$X^\dagger S X = I$ (checked to `1e-8`), or `X` alone for an operator the
caller has already orthogonalised. An `OperatorFrame` bundles a validated
`S`/`X` pair with a basis identity, AO ordering and shell-image gauge so
that several solves (both spins, every SCF iteration) share one checked
frame; its identity digest covers the actual `S`/`X` bytes, which is what
prevents a changed retained space from reusing an old subspace under the
same basis label.

Matrix-free Davidson needs the diagonal of the operator *in the frame it
iterates in*. For an orthonormal-frame callback supply `diagonal`. For an AO
callback or sparse matrix with explicit `X`, supply `reduced_diagonal`, the
real diagonal of $X^\dagger A X$ in that exact retained frame, computed by
the provider from its own representation; it is not the AO diagonal and
cannot be estimated by transforming `diag(A)`. Overlap-only rank reduction
is not admitted for callback Davidson, because the provider cannot know the
column gauge of an `X` the solver computes. Dense input computes its reduced
diagonal itself and refuses the extra field. Callbacks may return reusable
scratch buffers: `apply` copies every result, including each column of a
vector-only block application, so a producer that reuses workspace is safe
as long as it does not mutate the input and freezes the density,
coefficient and plan data its action captures.

`OperatorMetadata` annotates a solve; it never instructs the solver to fill
or rotate anything. `kind`, `basis_id`, `generation`, `spin`, `kpoint`,
`symmetry_block` and optional `occupations` identify the problem;
`frame_id`, `build_id` and `density_id` tie the snapshot to an evaluated
Fock build; `role` is `"physical"`, `"effective"` (a Roothaan ROHF operator)
or `"solver"`, the last requiring a `modification` string such as the DIIS
extrapolation or level shift that produced it. A new Fock build is a new
snapshot with a new generation.

### Solving and reading the diagnostics

```python
from vibeqc.solvers.eigensolver import SolverOptions
from vibeqc.solvers.operators import solve_operator
from vibeqc.output.eigensolver import eigensolver_diagnostics

result = solve_operator(ks, SolverOptions(n_roots=1, tol=1e-10))      # dense by default
print(eigensolver_diagnostics(result))
```

`solve_operator(operator, options, solver="dense", previous=None,
require_selection=False)` validates the request against the capability table
*before* applying the operator, reduces an AO problem to the orthonormal
frame, dispatches to the named backend and then **recomputes the residuals
independently** of whatever the backend reported. A solve returns only when
every requested projected residual is at most `tol`, the backend reported
success, the returned eigenvectors are metric-orthonormal to a Frobenius
error of `1e-7` and the requested number of roots came back; otherwise it
raises `RuntimeError`. Returned eigenvalues are ascending, eigenvectors are
in the original (AO) frame, and the `SolverResult` carries:

| Field | Meaning |
|---|---|
| `eigenvalues`, `eigenvectors` | ascending roots and AO-frame columns $C = XY$ |
| `residuals` | $\lVert A C_i - \varepsilon_i S C_i\rVert_2$ when `S` is known (`residual_frame="original"`); with `X` alone the AO metric is unknown and this is the orthogonal-frame residual (`residual_frame="orthogonal"`) |
| `projected_residuals` | $\lVert X^\dagger A X\,Y_i - \varepsilon_i Y_i\rVert_2$, the quantity the convergence test uses |
| `orthogonality_error` | Frobenius norm of $C^\dagger S C - I$ |
| `retained_rank` | dimension of the retained frame after rank reduction |
| `selection_verified` | `True` only for the dense backend |
| `n_iter`, `subspace_dim`, `n_converged` | iteration count and, for Davidson, the final subspace size and the number of roots under `tol` |
| `n_matvec` | observed callback vector applications including the diagnostic ones; `None` for dense input, never estimated |
| `operator_metadata` | the annotations the operator carried |

`eigensolver_diagnostics(result)` turns this into a JSON-compatible record
with the nested `operator_metadata` (kind, basis identifier, generation,
spin, k-point, symmetry block, occupations, and the frame/build/role fields
when they are set) beside the flat `operator_kind`. It is the portable
formatter for solver output; the historical SCF logs and QVF schemas have
not been extended with it. Solver modules write no user files.

### Residuals, selection and warm starts

**Residuals are not a selection certificate.** A small residual proves that
the returned pair is an accurate eigenpair; it does not prove that no other
eigenvalue lies closer to the target, or lower, than the ones returned. Only
the dense backend enumerates the complete finite spectrum, so
`selection_verified` is `True` only there; `require_selection=True` makes
`solve_operator` refuse every iterative backend with `NotImplementedError`.
Dense selectors `SA`, `LA`, `SM`, `LM` and `BE` (both ends, the odd root
going to the upper end) choose the roots and the result is then presented
ascending; a `sigma` selects the roots nearest the shift, including a root
exactly at `sigma`, without any inverse. Iterative solvers accept `SA` and
`LA` only (`harmonic_davidson` only `SA`), and an `LA` request is solved on
$-A$ and flipped back.

**Projected residuals versus AO residuals.** With a truncated metric the
two can differ: a solution that is exact in the retained frame can leave a
nonzero residual along the discarded overlap directions. The result exposes
both and the convergence test uses the projected one; it never presents
projected convergence as exactness in directions the frame dropped.

**Warm starts.** Pass `previous=old_result` to `davidson` or
`hermitian_davidson`. The previous eigenvectors are AO-frame columns, so they
must come from a solve with the same `kind`, `basis_id`, `spin`, `kpoint`,
`symmetry_block` and `frame_id`; the generation may differ, which is what an
SCF iteration does. They are mapped to reduced coordinates by least squares
($XY = C_{\text{prev}}$), which also handles rectangular `X`. A guess is a
subspace seed, never a root identity: `subspace_overlap(previous, current,
overlap=S)` returns the overlap amplitudes and the principal cosines of two
metric-orthonormal subspaces so a caller can track a *cluster* of orbitals
without imposing an orientation on a degenerate pair, and it never reorders
occupations. `SCFStep.prev_C` with `solver="dense"` is refused rather than
silently ignored.

### What is refused

Every row below was exercised on vibe-qc 0.17.15 development build (main,
2026-10-11):

| Request | Outcome |
|---|---|
| `require_selection=True` with `davidson` | `NotImplementedError`: requested-root selection is not certified by this iterative backend |
| `which="SM"` with `davidson` | `NotImplementedError`: selector not supported |
| `preconditioner="none"` with `davidson` | `NotImplementedError`: option not implemented by this backend |
| complex matrix with `davidson`, `lobpcg`, `jd` or `harmonic_davidson` | `NotImplementedError`: use `hermitian_davidson` or `dense` |
| callback operator with `lobpcg`, `jd` or `harmonic_davidson` | `NotImplementedError`: these backends would materialise the matrix |
| `guess_vectors` or `previous` with `lobpcg`, `jd`, `dense`, `lanczos` | `NotImplementedError`: no warm start |
| `sigma` with `davidson` | `NotImplementedError`: no interior shift |
| `harmonic_davidson` without `sigma` | `ValueError`: GPLHR requires sigma (target shift) |
| callback Davidson without `diagonal` | `ValueError`: implicit probing is disabled |
| callback Davidson with `n_roots < n < 2 n_roots + 6` | `NotImplementedError`: the native small-problem fallback would materialise the operator |
| `max_subspace` below the restart floor; `n_guess` negative or below `n_roots` | `ValueError` |
| `n_roots` above the retained rank; `tol` or `max_iter` not positive | `ValueError` |
| `previous` from a different kind/basis/spin/k-point/frame | `ValueError`: previous subspace belongs to a different operator frame |
| `FockOperator` with `kind="ci"`; `OperatorMetadata(kind="casida")` | `ValueError` |
| matrix plus callback; `reduced_diagonal` on dense input | `ValueError` |

The historical descriptor API (`EigenProblem`, `GeneralizedEigenProblem`,
`InteriorEigenProblem`, `SCFStep`, `solve_eigenproblem`, `solve_scf_step`)
remains available for compatibility and routes an operator argument to
`solve_operator`. Its legacy strategies still probe unit vectors for a
missing diagonal and may materialise a callback; new integrations use the
strict interface.

## Which drivers use it

The contract is additive and the migration is incremental; the
[consumer inventory](../design/operator_consumers.md) records every audited
call site and the reason for each remaining bypass. In the current tree:

- **Molecular RHF, UHF, RKS and UKS** diagonalise through the native
  `solve_fock_matrix` boundary, including the opt-in MANTA molecular entry
  points, which select the integral source but not a separate eigensolver.
- **Periodic k-point SCF** (native Bloch and periodic SCF kernels, and the
  shared Python multi-k helper used by the Ewald and GDF multi-k RHF/RKS/UHF/UKS
  drivers) solves complex Hermitian per-k problems with adjoint transforms;
  k-point DFTB0 and SCC-DFTB reach the same helper. **DFTB0/UDFTB0** use the
  native generalized one-electron solve with a strict Cholesky metric.
- **Gamma Ewald and Gamma/PBC GDF HF and KS**, **Gamma GPW/GAPW** including
  range-separated GAPW, **multi-k GPW RKS/UKS/ROKS and GAPW RKS**, and the
  fixed-density band paths use the internal `FockDiagonalizer` or the
  stateless `solve_dense_fock` adapter, with method-owned `X`/`S`, rank
  policies, occupations and Davidson thresholds, separate alpha/beta warm
  subspaces, and projected DIIS matrices kept in their orthogonal frame.
- **Molecular and Gamma GPW ROHF/ROKS and Gamma Ewald ROHF** solve the
  Roothaan effective Fock through the same adapter and canonicalise with
  separate dense solves per closed/open/virtual block
  ([ROHF and ROKS](rohf.md)); the dense HCORE seed, level shifts and the
  alpha-energy occupation rule stay with the method.
- **Iterative TDA** passes its sigma action as a
  `HermitianOperator(kind="response")` with the analytic diagonal.
- **Post-SCF semicanonicalisation** (Brueckner CCD, FNO-CCSD, reference
  UMP2/ROHF-MP2) and the **DLPNO-MP2/UMP2 pair-PNO Fock rotations** use
  dense solves over named MO partitions; the shared unrestricted PNO helper
  also serves the experimental neutral CCM UMP2/UCCSD pair build without
  changing that line's status.

Beyond eigensolution, several Gamma producers now return an **evaluated Fock
state**: Gamma Ewald RHF, RKS, UHF and UKS (3D cells and the 2D slab
routes) and the bare native Gamma GDF RHF/RKS expose `result.fock_build`,
an immutable record of the physical Fock, the density it was built from,
the energy components and the source identity. `next_density` (RHF/RKS) or
`next_density_alpha`/`next_density_beta` (UHF/UKS) and
`density_fixed_point_residual` describe the orbital fixed-point map
separately, so that at an iteration cap the next density is not mistaken
for the evaluated one when energies are recomputed or a restart is written.
Final orbital energies on those routes are physical Rayleigh quotients, also
on capped exits. DIIS-extrapolated or level-shifted operators carry the
same build identity with `role="solver"`; the physical operator is never
replaced.

Gamma Ewald ROHF returns the same kind of record: its `fock_build` holds both
physical spin Focks, their evaluated densities, the energy components and the
source receipt. `effective_fock_operator` is the Roothaan effective Fock with
the same parent build and density identities and `role="effective"`; it is not
an extra energy channel. `effective_fock_residual` reports the projected ROHF
commutator that the constrained convergence test uses, while
`fock_build.commutator_norm` measures the unconstrained spin commutators,
which need not vanish at ROHF convergence. QVF provenance records both
identities, including on capped and zero-iteration runs.

Two policies are visible at this boundary. Real-only consumers (Gamma Ewald
RHF/RKS/UHF/ROHF and the Gamma GDF HF/KS drivers, which build densities
with a transpose) reject active complex Davidson guesses and complex frames
before any J/K build (#494); complex-capable consumers use adjoints. And a
driver whose occupation rule ranks *all* candidates (ROHF, the spin-locked
`PATTERN_HOLD` and quadratic updates of Gamma UHF) refuses a partial Davidson
request before building a Fock. Deliberately outside the contract are the
memory-bounded periodic MANTA RHF kernel with its own Jacobi solver, the
CI and selected-CI sigma engines, the stability-analysis Davidson, metric
and density factorisations, Hessians and non-Hermitian CC Jacobians.

## Theory

Throughout, $A$ is an $n \times n$ Hermitian matrix, $k$ the number of
wanted roots, and the AO problem is $F C = S C \varepsilon$.

**Generalized problems and canonical orthogonalisation.** An orthonormal
frame is any $X$ with $X^\dagger S X = I$; then $C = X Y$ and
$X^\dagger F X\, Y = Y \varepsilon$ is a standard Hermitian problem.
vibe-qc's rank-reducing choice is Löwdin's canonical orthogonalisation
({vibeqc-cite}`lowdin_ortho_1970`): with $S = U s U^\dagger$,
$X = U_{\text{keep}}\, s_{\text{keep}}^{-1/2}$ keeps the overlap
eigenvalues above $\max(c,\, c\, s_{\max})$ with cutoff $c$ (`overlap_cutoff`,
`1e-12` at the operator boundary; the SCF drivers pass their own
`linear_dep_threshold`). Discarded directions are near-linear combinations
of basis functions that the basis cannot resolve; the retained rank is
reported, a negative overlap eigenvalue beyond the cutoff is an error, and
the root count must fit the retained rank. Pivoted Cholesky decomposition
of $S$ ({vibeqc-cite}`lehtola_pivoted_cholesky_2019`) is the alternative
that removes *functions* rather than *directions*; DFTB0 uses a strict SPD
Cholesky metric with no rank deletion. Every complex transformation uses the
adjoint, which is what makes k-point Fock matrices work in the same code
path.

**Davidson subspace iteration** ({vibeqc-cite}`davidson_iterative_1975`; the
block form is Liu's). Given an orthonormal search space $V$, form the
projected matrix $H = V^\dagger A V$, diagonalise it to obtain Ritz pairs
$(\theta_i, u_i = V y_i)$, and compute the residuals
$r_i = A u_i - \theta_i u_i$. Each unconverged root contributes a
correction vector from the diagonal preconditioner,

$$
t_i = (\theta_i\, I - D)^{-1} r_i ,
\qquad D = \operatorname{diag}(A),
$$

with the denominator magnitude floored at `preshift`; the corrections are
orthogonalised against $V$ (and each other, with conjugate inner products
in the complex kernel) and appended. When $V$ reaches `max_subspace` it is
collapsed to the current Ritz vectors. Convergence is declared when all
$\lVert r_i \rVert \le$ `tol`. The method converges fast when $A$ is
diagonally dominant in the frame it is given, which an orthonormalised Fock
matrix of a reasonable guess is, and it needs only $O(nm)$ storage for a
subspace of size $m$. A poor diagonal (a Fock matrix in a frame far from its
eigenbasis) can stall it, and the orthogonal-frame diagonal must be supplied
for a matrix-free operator because probing it costs $n$ applications.

**LOBPCG** ({vibeqc-cite}`knyazev_lobpcg_2001`). The locally optimal block
preconditioned conjugate-gradient method keeps a block of $b$ iterates
$X$, their preconditioned residuals $W = T(AX - X\Theta)$ and the previous
search directions $P$, and performs Rayleigh-Ritz on the $3b$-dimensional
space $[X, W, P]$ at every step; the new $X$ are the lowest Ritz vectors and
$P$ the part of the update orthogonal to the old $X$. With a symmetric
positive-definite preconditioner $T$ this is the locally optimal
three-term recurrence of conjugate gradients applied to the Rayleigh
quotient, and storage stays at $O(nb)$ with a small projected problem of
order $3b$. vibe-qc's kernel uses the diagonal preconditioner; it accepts
explicit real symmetric matrices only.

**Jacobi-Davidson** ({vibeqc-cite}`sleijpen_jd_1996`). The Davidson
correction $t_i$ above is a crude approximation to the Newton step for the
eigenvector. Jacobi-Davidson solves instead the projected correction
equation for $t \perp u$,

$$
(I - u u^\dagger)\,(A - \theta I)\,(I - u u^\dagger)\, t = -r ,
$$

approximately with an inner Krylov solver (MINRES in vibe-qc, capped by
`max_inner`), and expands the search space with $t$. The projection removes
the current eigendirection, so the equation is well posed near convergence,
and no factorisation of $A - \theta I$ is needed; with a target shift
$\sigma$, the Ritz values nearest $\sigma$ are selected at every step. The
outer iteration converges quadratically once the inner solve is accurate,
which is why JD is a candidate for interior roots, at the price of several
operator applications per outer step.

**Harmonic Rayleigh-Ritz for interior eigenvalues.** Ordinary Ritz values
of a growing subspace approach the extremal eigenvalues from inside the
spectrum, so an interior eigenvalue near $\sigma$ is approximated poorly
and spurious Ritz values appear between clustered roots. The harmonic
extraction (used by the GPLHR family, {vibeqc-cite}`zuev_gplhr_2015`)
replaces the Galerkin condition by a Petrov-Galerkin one on the shifted
operator: with $W = (A - \sigma I) V$, find $(\theta, y)$ such that

$$
(A - \sigma I)\, V y - (\theta - \sigma)\, V y \;\perp\; W ,
\qquad\text{i.e.}\qquad
W^\dagger W\, y = (\theta - \sigma)\, W^\dagger V\, y .
$$

The harmonic Ritz values $\theta$ are reciprocals of ordinary Ritz values
of $(A - \sigma I)^{-1}$ over $W$, obtained without applying any inverse,
and the ones nearest $\sigma$ are the reliable interior approximations.
vibe-qc's harmonic-Davidson variant expands the space with preconditioned
residuals of the shifted problem, using the signed denominator
$(D - \sigma I)^{-1}$ with the `preshift` floor, selects candidates and
retained vectors by the distance of their harmonic values to $\sigma$
(residual size does not change that ordering), restarts while preserving
the target vectors, and reports Rayleigh quotients of the normalised
vectors as eigenvalues. A singular harmonic pencil ($W^\dagger V$ singular,
for example a diagonal matrix with an eigenvalue exactly at $\sigma$) is
rejected rather than dropping that root. Because a zero residual does not
certify the absence of a closer omitted root, `selection_verified` is
`False` for every interior iterative solve; the dense nearest-shift solve is
the supported reference for small problems.

**Lanczos.** The Python reference builds the tridiagonal projection of
$A$ with full two-pass reorthogonalisation, diagonalises it and returns
Ritz pairs; it has no restart, block or warm-start machinery, and its
`converged` flag requires the measured residual and the requested root
count. It exists as an exploratory matrix-free reference, not as a
production path.

## Interior eigenvalues: Jacobi-Davidson and harmonic-Davidson

```{admonition} Status: experimental
:class: warning

`solver="jd"` and `solver="harmonic_davidson"` (`"gplhr"`) are experimental
routes reachable only through the library API. No SCF driver uses them,
their numbers are research output, and they are not part of the release
gate. Check every returned pair against $A x - \lambda x$ before use, and
remember that a converged residual does not prove that the nearest root was
found.
```

Pass an `InteriorEigenProblem` with the target shift, or set
`SolverOptions.sigma` on an operator solve:

```python
import numpy as np
from vibeqc.solvers.eigensolver import (
    EigenProblem, InteriorEigenProblem, SolverOptions, solve_eigenproblem,
)

rng = np.random.default_rng(3)
n = 60
A = np.diag(np.linspace(-5.0, 5.0, n)) + 0.01 * rng.standard_normal((n, n))
A = 0.5 * (A + A.T)
sigma = 0.37                     # target: the eigenvalue nearest 0.37
problem = InteriorEigenProblem(problem=EigenProblem(matrix=A, n=n), sigma=sigma)
options = SolverOptions(n_roots=1, tol=1e-8, max_iter=500)

jd = solve_eigenproblem(problem, options, solver="jd")
exact = np.linalg.eigvalsh(A)
print(jd.eigenvalues, jd.converged, jd.n_iter)
print(exact[np.argmin(np.abs(exact - sigma))])

try:
    hd = solve_eigenproblem(problem, options, solver="harmonic_davidson")
    print(hd.eigenvalues, hd.converged, hd.n_iter, hd.residuals)
except RuntimeError as exc:          # stagnation or iteration exhaustion
    print("harmonic-Davidson did not converge:", exc)
```

On this 60 by 60 near-diagonal matrix (vibe-qc 0.17.15 development build,
main, 2026-10-11, Apple M-series laptop) Jacobi-Davidson returns
0.42479002 against the dense value 0.424790021 with `converged=True` after
60 outer iterations and a recomputed residual of 6.6e-9; harmonic-Davidson
returns the same root after 6 iterations with residual 2.1e-9. The two
solvers report failure differently: JD returns `converged=False` with its
last Ritz value, while harmonic-Davidson raises `RuntimeError` naming the
shift, the iteration count and the largest residual and suggesting
`solver="jd"` or a larger `max_subspace`/`max_iter`, which is why the
snippet wraps it. Its working block defaults to `max(2 n_roots, 4)` guard
roots, capped at the dimension (`block_size`, at least `n_roots`), and its
search space to `min(6 \cdot \text{block} + 10, n)` (`max_subspace`).

The seeded comparison `examples/experimental/interior_eigensolvers.py`
(six matrices of size 60, 120 and 300 with off-diagonal coupling 0.01 and
0.05, shift 0.4248) gives, on the same build, converged JD roots on all six
cases in 75 to 306 iterations and converged harmonic-Davidson roots on five
of them in 7 to 483 iterations; on the hardest case (n = 300, coupling
0.05) harmonic-Davidson still stops after 500 iterations with a residual of
5e-2 and raises. That failure is retained deliberately: loosening the
tolerance or accepting a nearby converged pair would not be a fix. Without
a shift, Jacobi-Davidson also reaches the lowest eigenvalues, but
`davidson` and `lobpcg` are the extremal solvers to use.

## Citations in job output

Solver citations are routed through `vibeqc.output.citations`:
`assemble(numerics=["eigensolver_davidson"])` and the keys
`eigensolver_hermitian_davidson`, `eigensolver_lobpcg`, `eigensolver_jd`,
`eigensolver_harmonic_davidson` and `eigensolver_gplhr` select the entries
quoted above. `eigensolver_harmonic_davidson` cites the source of the
variant, not a claim of implementing the full published algorithm.

## Verification

The contract is covered by `tests/test_operator_contract.py` (complex
generalized roots against SciPy's independent generalized driver, metric
normalisation and rank truncation, an exact unwanted invariant subspace
showing why a zero residual is not a selection certificate, callback
scratch reuse, and the refusals listed above), `tests/test_solver_framework.py`
and `tests/test_davidson.py` (dense against iterative SCF energies and the
partial-spectrum controls), `tests/test_tddft_davidson.py` (the response
consumer), `tests/test_semiempirical_dftb0.py` and
`tests/test_periodic_bloch_bridge.py` (generalized and complex k-point
consumers), and the SCF-route tests that compare each migrated driver's
Fock spectrum with generalized LAPACK and its converged energies, densities
and spin populations with the previous dense solve:
`tests/test_periodic_k_gdf.py`, `tests/test_periodic_gapw_open_shell.py`,
`tests/test_periodic_gpw_uks_multik.py`, `tests/test_rohf.py`,
`tests/test_rohf_convergence_controls.py`, `tests/test_periodic_rohf_ewald.py`,
`tests/test_periodic_uhf_ewald.py`, `tests/test_rohf_mp2.py` and
`tests/test_dlpno_mp2.py`. These are eigensolver-migration checks; they do
not certify grid or lattice convergence of the experimental periodic
models. The runnable tutorial is
[`examples/tutorials/unified_operators.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/tutorials/unified_operators.py), built
on `examples/tutorials/unified_operators.py`.

## Extending the framework

```
SolverStrategy (ABC)
├── DenseSolver               -> numpy.linalg.eigh
├── DavidsonSolver            -> _vibeqc_core.davidson_solve (C++)
├── HermitianDavidsonSolver   -> _vibeqc_core.davidson_solve_hermitian (C++)
├── LOBPCGSolver              -> _vibeqc_core.lobpcg_solve (C++)
├── JDSolver                  -> _vibeqc_core.jd_solve (C++, experimental)
├── GPLHRSolver               -> _vibeqc_core.gplhr_solve (C++, experimental)
└── LanczosSolver             -> pure Python
```

A `SolverStrategy` subclass registers itself at class-definition time under
its lower-cased class name and the entries of its `_aliases` list, and is
then available to `solve_eigenproblem(..., solver="mysolver")`. For the
strict entry point it also needs a row in `solver_capabilities` and in the
accepted-option table of `solve_operator`, which is how an unsupported
option becomes a refusal instead of a silent default:

```python
from vibeqc.solvers.eigensolver import SolverStrategy, SolverResult

class MySolver(SolverStrategy):
    _aliases = ["mysolver", "my"]
    supports_extremal = True
    supports_matrix_free = True

    def solve(self, problem, options):
        ...                                   # your algorithm
        return SolverResult(eigenvalues=values, eigenvectors=vectors,
                            n_iter=iterations, converged=converged)
```

## References

The database entries cited above are {vibeqc-cite}`davidson_iterative_1975`,
{vibeqc-cite}`knyazev_lobpcg_2001`, {vibeqc-cite}`sleijpen_jd_1996`,
{vibeqc-cite}`zuev_gplhr_2015`, {vibeqc-cite}`lowdin_ortho_1970` and
{vibeqc-cite}`lehtola_pivoted_cholesky_2019`. Not in the citation database:

- Liu, B. *The simultaneous expansion method for the iterative solution of
  several of the lowest eigenvalues and corresponding eigenvectors of large
  real-symmetric matrices*, in Numerical Algorithms in Chemistry: Algebraic
  Methods, LBL-8158, Lawrence Berkeley Laboratory, 49 (1978). The block
  Davidson method.
- Kresse, G. and Furthmüller, J. *Phys. Rev. B* 54, 11169 (1996).
  DOI: 10.1103/PhysRevB.54.11169. Blocked Davidson for Kohn-Sham orbitals.
- Vecharynski, E., Yang, C. and Xue, F. *SIAM J. Sci. Comput.* 38, A500
  (2016). DOI: 10.1137/15M1027413. The full generalized preconditioned
  locally harmonic residual (GPLHR) algorithm that the local variant does
  not implement.
- Lanczos, C. *J. Res. Nat. Bur. Stand.* 45, 255 (1950). DOI:
  10.6028/jres.045.026. Paige, C. C. *J. Inst. Math. Appl.* 10, 373 (1972).
  DOI: 10.1093/imamat/10.3.373. The Lanczos recurrence and its
  reorthogonalisation.

## See also

- [SCF convergence](scf_convergence.md): accelerators, level shifts and the
  second-order finalizers that act between diagonalisations.
- [Initial guesses](initial_guess.md): the seed the first solve starts from.
- [ROHF and ROKS](rohf.md): the Roothaan loop's complete-spectrum rule.
- [Operator contract](../design/operator_contract.md),
  [consumer inventory](../design/operator_consumers.md) and the
  [eigensolver literature assessment](../design/eigensolver_literature.md).
- [Eigensolvers and operators tutorial](../tutorial/eigensolver_operators.md),
  which walks through the runnable script
  [`examples/tutorials/unified_operators.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/tutorials/unified_operators.py).
