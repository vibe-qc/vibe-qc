# Eigensolver & SCF Solver Framework

vibe-qc provides a shared Hermitian operator contract and dense/iterative
backends. Molecular SCF, the shared multi-k periodic helper, DFTB0 and
iterative TDA now use this boundary. Gamma Ewald and GDF HF/KS also share
a metric-aware adapter with separate spin recycling and unchanged occupations.
Gamma GPW/GAPW RHF/RKS/UHF/UKS and range-separated GAPW use the same adapter
for their SCF Fock solves. Closed-shell GPW retains its Davidson size threshold;
the other grid routes retain dense solves. Their physical support gates are
unchanged. Multi-k GPW RKS/UKS/ROKS, GAPW RKS and the fixed-density band
helper also use shared dense solves, with separate AO/projected coordinates
and per-spin/per-k metadata. Molecular and Gamma GPW ROHF/ROKS use the same
SCF adapter, with separate dense Guest-Saunders block solves and unchanged
fractional-shell policy. Gamma Ewald ROHF effective-Fock solves now use it too.
They honor `PeriodicRHFOptions.use_davidson`, `davidson_min_dim` and the
Davidson controls. Open-shell selection requires the complete retained
spectrum, so an active Davidson request with too few or too many roots is
refused before building J/K. The dense HCORE seed, level-shift policy and
alpha-energy occupation rule remain method-owned. Gamma Ewald ROHF preserves
the final density and canonicalizes within equal-occupation blocks of the
unshifted physical effective Fock, retaining the finalization repair for #487.
Brueckner CCD, FNO-CCSD and reference UMP2/ROHF-MP2
also share dense Fock solves within their semicanonical orbital partitions.
These require complete block spectra and retain their method's frozen-core,
occupation and truncation policies. Other consumers are being migrated;
see the [design and migration status](../design/operator_contract.md).
SCF acceleration and occupation assignment remain separate from eigensolution.

Gamma Ewald UHF also uses independent alpha/beta solver histories and leaves
the supplied Davidson options and guesses unchanged. Its PATOM in-field and
terminal consistency Focks use the same boundary. Ordinary SCF permits partial
spectra covering both occupied spin spaces; PATTERN_HOLD or quadratic updates
require the full retained spectrum and reject smaller requests before J/K
construction. The HCORE seed uses the configured dense/Davidson choice in a
separate one-electron solve, without seeding either spin history. Final orbital energies
are physical Rayleigh quotients in the returned orbital frame, including
unconverged iteration-cap exits.

Gamma Ewald 3D RHF and UHF expose `result.fock_build`, an immutable evaluated
physical state containing the Focks, their evaluated densities, energy
components and source provenance. RHF carries one total density with
occupations zero or two; UHF carries separate spin densities. UHF Focks share
one validated overlap and retained orbital frame. DIIS/shifted Focks carry the same build identity and
an explicit solver role, while the physical operators remain unchanged.
RHF's `next_density`, UHF's `next_density_alpha` and `next_density_beta`, and
`density_fixed_point_residual` describe the separate orbital fixed-point map.
At an iteration cap, the next density must not replace the evaluated density
when recomputing energies or writing a restart. Legacy result arrays remain
writable copies, so modifying them does not change the immutable build.
QVF provenance records the evaluated build/density/source identities and
the orbital-state relation through the existing output route. These records
do not certify finite-support convergence, symmetry reduction or CRYSTAL parity.
For both methods, the returned orbitals and occupations define the evaluated
density. Orbital energies are physical Rayleigh quotients at an iteration cap;
these orbitals need not diagonalize the rebuilt physical Fock before convergence.
The final SCF history entry describes that same evaluated state. A provisional
convergence decision is confirmed against its energy change, commutator and
density fixed-point residual. RHF quadratic updates require a complete retained
orbital spectrum. The slab RHF producer is not included in this migration.
The other method families still need construction/state migration; shared
eigensolution alone does not provide this result contract.

Dense DLPNO-MP2/UMP2 pair-PNO Fock rotations use the same contract, with
separate pair/spin labels. Density truncation, PAO construction and pair
amplitude coupling keep their method-specific rules. The shared unrestricted
PNO helper also serves the experimental neutral CCM UMP2/UCCSD pair build;
it does not change the experimental status of those methods.

## Quick Start with SCF

```python
from vibeqc import run_job, Molecule, Atom

mol = Molecule([Atom(8, [0, 0, 0]),
                Atom(1, [0, 1.43, -0.98]),
                Atom(1, [0, -1.43, -0.98])])  # coordinates in bohr

# Request Davidson (the method-specific dimension threshold still applies)
result = run_job(mol, basis="def2-svp", method="rhf", solver="davidson")

# Use dense diagonalization (default)
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
| `davidson` | Block Davidson (Liu 1978) | C++ | Extremal eigenvalues, especially when a useful diagonal preconditioner is available |
| `block_davidson` | Alias for `davidson` | C++ | Same |
| `lobpcg` | LOBPCG (Knyazev 2001) | C++ | Explicit real extremal problems; no LOBPCG SCF route in run_job |
| `dense` | `np.linalg.eigh` | NumPy | Full spectra and verified dense root selection |
| `eigh` | Alias for `dense` | NumPy | Same |
| `lanczos` | Lanczos with full reorthogonalization | Python | Matrix-free, moderate accuracy; exploratory |
| `jd` | Jacobi-Davidson (Sleijpen 1996) | C++ | Interior eigenvalues; experimental correction-equation implementation |
| `jacobi_davidson` | Alias for `jd` | C++ | Same |
| `gplhr`, `harmonic_davidson` | Experimental harmonic-Davidson variant (inspired by Zuev 2015) | C++ | Interior roots of real symmetric matrices; experimental |
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

## Strict operator capabilities

Use `solver_capabilities(name)` with `solve_operator`. Historical strategy
flags describe legacy adapters and do not reliably describe materialization.

| Solver | Complex | Sparse/callback without dense construction | Interior | Warm start |
|---|---|---|---|---|
| `dense` | Yes | No | Yes, full spectrum then selection | No |
| `davidson` | No | Yes, except below the native restart floor | No | Yes |
| `hermitian_davidson` | Yes | Yes, except below the native restart floor | No | Yes |
| `lobpcg` | No | No | No | No |
| `jd` | No | No | Experimental | No |
| `harmonic_davidson` | No | No | Experimental | No |
| `lanczos` | Yes | Yes | No | No |

Nondefault unsupported options raise. `run_job(solver="lobpcg")` raises too:
older versions silently executed Davidson for this request. Existing
method-specific SCF options and size thresholds remain in force.

## Operator quick start and diagnostics

```python
import numpy as np
from vibeqc.solvers.operators import FockOperator, OperatorMetadata, solve_operator
from vibeqc.solvers.eigensolver import SolverOptions
from vibeqc.output.eigensolver import eigensolver_diagnostics

F = np.array([[1., .2j], [-.2j, 2.]])
S = np.array([[1., .1], [.1, 1.]])
operator = FockOperator(matrix=F, overlap=S,
    metadata=OperatorMetadata(kind="ks", spin="restricted", kpoint=(.25, 0., 0.)))
result = solve_operator(operator, SolverOptions(n_roots=1, tol=1e-10))
print(eigensolver_diagnostics(result))
```

The portable diagnostic record includes `operator_metadata`: kind, basis/frame
identifier, generation, spin, k-point, symmetry block and supplied occupations.
These annotations identify the solve; they do not assign occupations or imply
that a projected residual certifies the full AO problem. Missing metadata is
`None`, and the existing flat `operator_kind` field remains available. The
formatter returns independent JSON-compatible lists for coordinate and
occupation data. This record is available through `vibeqc.output.eigensolver`;
historical SCF logs and QVF schemas have not yet been extended to include it.

`residuals` reports the original AO residual when S is supplied;
`projected_residuals` reports the orthogonal-frame residual used for convergence.
These can differ when overlap directions are discarded. `orthogonality_error`
is the Frobenius error in `C.conj().T @ S @ C`. `retained_rank` makes the
truncation visible. X without S reports residuals in the orthogonal frame.
Real and complex Davidson dispatch a supplied `matmat` as block applications;
a `matvec` alone is applied column by column. Supply `diagonal` for an
orthogonal-frame callback. For an AO callback or sparse matrix, supply explicit
`X` and `reduced_diagonal`, the real diagonal of `X.conj().T @ A @ X` in that
exact retained frame. It is not the AO diagonal and must not be estimated by
transforming only `diag(A)`. The provider must compute it from its own
representation. The operator copies it and checks its length and finiteness.
Overlap-only canonical reduction is not admitted for callback Davidson because
a supplied reduced diagonal does not identify the computed X's column gauge.
Dense input computes its reduced diagonal and rejects the extra field.
S and X are still explicit arrays, even when A is represented by a callback.

Callbacks may return reusable scratch buffers: `apply` copies their results,
including each column of a vector-only block action. They must not mutate the
input. This protects action results from subsequent scratch reuse; the callback
owner must still freeze all density, coefficient and plan data captured by
the action. Build receipts remain method-owned and do not enforce immutability.
The strict solver raises on nonconvergence. It never guesses a missing
matrix-free diagonal or silently converts sparse/callback input to dense.

`selection_verified` is true only for dense spectrum enumeration. Small
residuals on iterative roots do not prove that they are the requested nearest
roots. `require_selection=True` refuses an uncertified iterative solve. Dense
selectors SA, LA, SM, LM and BE select roots separately from ascending output
ordering. A shift selects the roots nearest sigma, including a root exactly
at sigma. Iterative extremal selectors are restricted to SA/LA.

Pass `previous=old_result` to a warm-start-capable solver. The previous vectors
are in the original AO frame and must share kind/basis/spin/k-point/block metadata;
new generations may change the Fock. `subspace_overlap` compares whole clusters
without imposing an arbitrary orientation on degenerate orbitals. Neither
warm starts nor tracking assign occupations. `SCFStep.prev_C` now rejects a
dense solve rather than silently ignoring the guess.

Run the complete numerical tutorial and bounded benchmark:

```sh
python examples/tutorials/unified_operators.py
python examples/tutorials/unified_operators.py --size 120
```

Focused verification, after rebuilding the native extension:

```sh
python -m pytest tests/test_operator_contract.py tests/test_solver_framework.py tests/test_davidson.py tests/test_tddft_davidson.py -q
python -m pytest tests/test_semiempirical_dftb0.py tests/test_periodic_bloch_bridge.py -q
python -m pytest tests/test_periodic_k_gdf.py::test_krhf_gdf_multik_h2_cubic_converges tests/test_periodic_k_gdf.py::test_krhf_gdf_multik_h2_density_matrices_are_hermitian tests/test_periodic_k_gdf.py::test_krhf_gdf_multik_h2_per_k_mo_shapes -q
python -m pytest tests/test_periodic_gapw_open_shell.py::test_gamma_grid_shared_fock_solve_matches_legacy tests/test_periodic_gapw_range_sep.py -q
python -m pytest tests/test_periodic_gpw_uks_multik.py::test_multik_grid_shared_solve_preserves_frames_and_occupations tests/test_periodic_gapw_postscf.py::test_band_path_uses_complex_metric_solve -q
python -m pytest tests/test_rohf.py::test_guest_saunders_rectangular_frame_and_block_order tests/test_rohf.py::test_roothaan_overlap_reduction_keeps_complete_retained_space tests/test_rohf_convergence_controls.py::test_roothaan_shared_solve_matches_legacy tests/test_rohf.py::test_roks_shared_solve_preserves_each_guess_basin -q
python -m pytest tests/test_periodic_rohf_ewald.py -q
python -m pytest tests/test_periodic_uhf_ewald.py -q
python -m pytest tests/test_rohf_mp2.py tests/test_rohf_ccsd_anchor.py tests/test_cc_variants.py::TestBruecknerCCD tests/test_ccsd_fno.py -q
python -m pytest tests/test_dlpno_mp2.py tests/test_dlpno_ump2.py -q
python -m pytest tests/test_ccm_dlpno_ump2.py tests/test_ccm_dlpno_uccsd.py -q
```

The contract tests compare complex generalized roots with SciPy's independent
generalized driver, check metric normalization and rank truncation, and include
an exact unwanted invariant subspace to demonstrate why a zero residual is not
a root-selection certificate. Existing molecular Davidson tests compare dense
and iterative SCF energies. The spin checks
`test_uhf_on_closed_shell_matches_rhf`, `test_uhf_s2_ideal_value_for_doublet`,
`test_uks_on_closed_shell_matches_rks` and
`test_uks_s_squared_for_doublet_is_near_075` cover restricted/unrestricted
consistency. `test_rks_result_energy_components_sum_to_total` checks KS energy
bookkeeping. The TDA tests exercise sigma action, excitations and transition
properties; the Bloch and GDF checks exercise complex physical k-point solves.
The Gamma grid checks compare each Fock spectrum with direct generalized
LAPACK, and compare converged H2/H3 energies, densities and spin populations
with the previous dense diagonalization. These are eigensolver migration
checks, not certification of the experimental grid models or of SCF
convergence for other systems.
Multi-k checks cover compact and vacuum regimes, complex metric spectra,
weighted electron counts, spin/k-point metadata and the compact GAPW fold
refusal. Projected DIIS results report orthogonal-frame residuals and return
reduced coefficients; their driver performs the AO lift and global filling.
Roothaan tests compare the effective Fock with generalized LAPACK, check
dense/Davidson option ownership and physical parity, and preserve each
degenerate-frontier guess basin against its previous dense solve. Separate
canonicalization tests check rectangular coefficients, spin densities and
ordering inside closed/open/virtual blocks without mixing those blocks.
Gamma Ewald ROHF checks also preserve shifted spectra and alpha-energy
occupation ordering, including rectangular metric frames and original versus
projected residuals. Closed- and open-shell finite-grid comparisons use the
same numerical source in both implementations; they do not establish grid or
lattice convergence.

Gamma Ewald UHF checks cover separate spin recycling, caller-owned options,
PATOM seeds, spinlock release, level-shift schedules, quadratic updates and
physical orbital-energy reporting on converged and capped exits. MANTA
crossing tests use the experimental direct batch and streamed RI spin-pair
builders to form fixed trial Focks. Independent integral contractions and
generalized LAPACK check their spectra, and later builds must not change
earlier Fock snapshots. These tests do not select MANTA by default or certify
RI fitting quality.

The opt-in molecular MANTA RHF/UHF entry points also reuse the native shared
Fock solver. Their exact/RI and dense/Davidson checks compare the returned
physical Fock spectra with generalized LAPACK, check AO residuals and metric
normalization, and verify that invalid Davidson controls are not ignored.
These small full-spectrum fixtures do not measure iterative scaling.

Periodic MANTA spin-Fock checks exercise complex spin/k blocks, subsequent
density builds and shell-image gauge changes for Coulomb, erf and erfc exchange.
The bounded periodic RHF driver keeps its specialized solver and memory plan;
export checks compare its spectra with generalized LAPACK and the common
operator API, including reduced overlap rank and both supported exchange
zero-mode policies. Reusing an orbital subspace across an AO image change
requires an explicit coefficient transformation and a new basis identifier.
On an unconverged exit, retain the evaluated density separately from the density
constructed from the returned orbitals.
The selected periodic job's scalar history and QVF density preserve these
evaluated states. Cross-boundary checks follow several capped iterations,
recompute their energies and metric commutators, and verify QVF transport from
independently moved AO centers back to the nuclear-center basis. These checks
do not extend the bounded driver's admitted solver or method capabilities.

Gamma Ewald UHF is a real-only consumer and refuses active complex Davidson
guesses before J/K construction. Its adapter also rejects complex frames and
Fock matrices. Complex-capable consumers must use conjugate transposes in
their density and contraction formulas; the lower operator API retains that
support. Audit #494 tracks this admission policy in the remaining real-only
SCF consumers.

Citation routing is available through
`assemble(numerics=["eigensolver_davidson"])` and equivalent backend keys in
`vibeqc.output.citations`. `eigensolver_harmonic_davidson` cites the source of
the variant, not a claim of implementing the full published GPLHR algorithm.

## Architecture

```
SolverStrategy (ABC)
├── DenseSolver           → np.linalg.eigh
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
  Passes the target shift to a capable backend; it does not create a shift-invert solver.
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
