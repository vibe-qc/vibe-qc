# OpenTrustRegion orbital optimization

OpenTrustRegion is an optional molecular SCF orbital optimizer. vibe-qc
computes every energy, density, orbital derivative and orbital update; the
pinned upstream numerical library owns the trust-region subproblem, acceptance
and optional line search. Nuclear geometry optimizers are unchanged.

Start with the worked tutorials for
[native/optimizer comparisons](../tutorial/opentrustregion.md) and
[READ restarts and orbital stability](../tutorial/opentrustregion_stability.md).
Their downloadable input scripts cover all four SCF methods, a UKS restart
and spin breaking in stretched H2.

The [theory chapter](opentrustregion_theory.md) derives the orbital
parameterization, physical gradient/Hessian action, constrained quadratic
model and actual/predicted energy ratio. It explains the three subsolvers,
their radius conventions, internal stability, and automatic citation output.

## Install and select

After the normal native dependency setup, build the optional component with a
Fortran compiler, Git and LP64 BLAS/LAPACK available:

```sh
CMAKE_ARGS="-DVIBEQC_ENABLE_OPENTRUSTREGION=ON" \
  .venv/bin/pip install -e . --no-build-isolation
.venv/bin/python -c "import vibeqc; print(vibeqc.has_opentrustregion())"
```

CMake fetches upstream revision
`8fa7769ae66233a566868a6bf03cdcbdb1ee69d0` (project version 2.0.0), builds
only its static PIC numerical library, and links
`OpenTrustRegion::opentrustregion`. It uses `INTEGER_SIZE=4`, the matching
`c_int` header convention, and LP64 BLAS/LAPACK. `USE_ILP64` is unsupported.
CMake carries the Fortran runtime dependencies through the link. Host-specific
instruction tuning and upstream tests/Python bindings are disabled in this
build. Default builds leave the option off and require no Fortran compiler.
A requested unavailable backend raises a capability error.

For offline builds, set `FETCHCONTENT_SOURCE_DIR_OPENTRUSTREGION` to a clean
checkout of that exact revision. Do not point it at a moving branch. Binary
packagers must include the MPL notice/source offer and the applicable Fortran
and BLAS runtime notices; see [licensing](../license.md).

Normal Python input files and `run_job` accept:

```python
from vibeqc import Atom, Molecule, OpenTrustRegionOptions, run_job

mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
otr = OpenTrustRegionOptions()
otr.subsystem_solver = "davidson"
otr.stability = "check"
result = run_job(
    mol, basis="sto-3g", method="rhf", output="h2-otr",
    orbital_optimizer="opentrustregion", opentrustregion_options=otr,
)
```

The lower-level Python API uses the same options:

```python
from vibeqc import RHFOptions, BasisSet, run_rhf
opts = RHFOptions()
opts.orbital_optimizer = "opentrustregion"
opts.opentrustregion = otr
result = run_rhf(mol, BasisSet(mol, "sto-3g"), opts)
print(result.opentrustregion.termination)
```

`UHFOptions`, `RKSOptions` and `UKSOptions` have the same fields. For KS,
select the functional and grid normally. Existing `native` defaults and
DIIS/Newton/SOSCF/TRAH behavior are unchanged. An explicit OpenTrustRegion
choice suppresses automatic native-converger retries. Explicit nonzero native
second-order thresholds or a quadratic fallback conflict and are rejected.
DIIS, damping and level shifts are not phases of the OpenTrustRegion solve.
Native restart schedules are also bypassed; `multi_guess_seeds` is rejected.
For multiple starts, run separately chosen initial densities explicitly.

## Supported manifold and options

The backend supports real molecular RHF/UHF and RKS/UKS with integer occupied
subspaces, exact or density-fitted JK, native LDA/GGA and global hybrids.
Range-separated hybrids, meta-GGA, nonlocal or external XC, COSX, DFT+U,
spinlock/state targeting, fractional occupations, solvent-response workflows,
ROHF/ROKS, complex/spinor and periodic optimization are unsupported. Explicit
requests raise diagnostics; periodic Gamma points are also outside this
adapter. The driver may project a non-idempotent atomic *guess* to integer
occupied subspaces, as ordinary molecular SCF does. This is not finite-
temperature or fractional-occupation optimization.

| Option | Default | Meaning |
|---|---:|---|
| `subsystem_solver` | `"davidson"` | `davidson`, `jacobi-davidson`, or `tcg` |
| `gradient_rms_tolerance` | `1e-8` | RMS of physical orbital-energy derivatives |
| `initial_trust_radius` | `0.4` | Radius in the selected subsolver's metric: Euclidean for Davidson, preconditioner-weighted for TCG; see [theory](opentrustregion_theory.md#the-quadratic-model-and-trust-radius) |
| `max_micro_iterations` | `50` | Upstream subproblem iteration cap |
| `line_search` | `False` | Upstream bracketing search, often costly for SCF |
| `seed` | `42` | Upstream reproducible random trial vectors |
| `stability` | `"check"` | `none`, `check`, or `follow` |
| `stability_tolerance` | `1e-4` | Final internal-stability negative-curvature threshold |
| `stability_residual_tolerance` | `1e-8` | Upstream stability eigensolver RMS residual tolerance |
| `stability_max_iterations` | `100` | Final/embedded stability iteration cap |

The outer SCF `max_iter` controls upstream macroiteration attempts. `max_iter=0`
is rejected for this backend. Tolerances/radius must be positive and finite,
iteration caps positive, and the seed nonnegative. Energy change and the
ordinary AO commutator residual must also satisfy `conv_tol_energy` and
`conv_tol_grad`. Upstream machine-precision termination alone is insufficient.
The final physical energy and residual are recomputed before success is
reported. Within occupied and virtual blocks the returned orbitals are
semicanonical; no final diagonalization exchanges an accepted occupied orbital
with a virtual orbital.

## Stability and diagnostics

A stationary point is not necessarily a minimum. `check` runs a final check
without escaping a newly detected final negative mode. `follow` additionally
asks upstream to escape saddles during its solve. At this pinned revision,
upstream **always checks and follows an initial stationary saddle**, including
with `none` or `check`. These options cannot be used to preserve an excited SCF
state. The spin sector, electron counts and occupation ranks remain fixed.

Upstream follows modes below its fixed `-0.01` threshold. vibe-qc performs a
separate final check at `-stability_tolerance` by shifting the check operator by
`stability_tolerance - 0.01`. That shift never changes the Hessian used for
optimization. Thus a small negative mode can be reported after `follow`; no
additional host acceptance or escape loop is hidden around the solver.

The verdict covers real occupied-virtual rotations in the selected manifold.
Restricted internal stability does not certify unrestricted or complex
stability. Failed eigensolves are inconclusive. The upstream C interface
returns no eigenvalue, so none is fabricated in the legacy stability fields.
Use `result.opentrustregion.stability_checked`, `stability_converged` and
`stable` for this backend.

`result.opentrustregion` and standard `.out` output report backend/revision,
termination, upstream error code, callback errors, convergence residuals,
accepted/trial/Fock/response evaluation counts, and the final stability verdict.
Upstream does not expose total macro/micro counts in C: these are `-1`
(unavailable), not numbers inferred from log text. `n_iter` counts accepted
model callbacks, including the initial zero rotation. The logger is captured
and emitted by `vibeqc.output` at debug level. Exceptions never unwind through
Fortran; failed callbacks leave the last accepted state available in the
returned unconverged SCF result. Standard `run_job` failure handling still
applies. Calls are serialized process-wide, including stability calls; nested
entry fails deterministically.

## Troubleshooting

| Observation | Meaning and next step |
|---|---|
| `has_opentrustregion()` is false or selection reports an unavailable backend | Rebuild with the CMake option enabled, then check the capability with the same Python used for the job. An ordinary build does not include the solver. |
| A native-converger conflict is rejected | Remove explicit Newton/SOSCF/TRAH thresholds or quadratic fallback settings from this calculation. Select one orbital optimizer. |
| Unsupported functional, state or workflow | Use a supported response model or explicitly choose a native route that supports the requested physics. Changing optimizer must not silently change the physical model. |
| `iteration_limit`, upstream code `102` | The solve exhausted its macroiteration budget. Inspect residuals, trial/response counts and the initial density before increasing the SCF `max_iter`. |
| `callback_failure` | A vibe-qc energy, update or response callback failed. Read `callback_error`; the returned state is the last accepted state and is unconverged. Resolve the underlying error before retrying. |
| `stability_inconclusive`, upstream code `202` | The stability eigensolve did not converge. Do not interpret `stable=False` alone as a detected negative mode. Inspect `stability_converged` and consider the stability iteration cap. |
| Successful upstream termination but `result.converged=False` | The recomputed physical energy, AO residual or orbital gradient did not meet the requested criteria. Read all convergence diagnostics; upstream success alone is insufficient. |
| Converged but final stability reports negative curvature after `follow` | Final checking can detect a mode below the requested threshold that the pinned upstream escape threshold does not follow. See the stability policy above and compare independently chosen starting densities. |

Normal `run_job` failure handling still applies; the lower-level SCF API
exposes the returned result for diagnostic workflows. The backend does not
automatically retry with native DIIS or TRAH after an explicit selection.

## Derivative and state contract

Coordinates are column-major occupied-virtual blocks, `a + nvir*i`, alpha
followed by beta for unrestricted states. For `C_new = C exp(K)`, `K_ai=k_ai`
and `K_ia=-k_ai`. The restricted gradient is `4 F_ai`; each unrestricted spin
gradient is `2 F_ai`. The exact matrix-free Hessian uses the full occupied and
virtual Fock blocks plus density response, including alpha/beta Coulomb and
XC coupling. Fock-gap diagonals are scaled preconditioner approximations;
they do not floor negative curvature in the physical Hessian.

Accepted updates commit only after energy, gradient and response construction
succeed. Trial energies use a separate state relative to the frozen accepted
reference. Both use stateless JK entry points, avoiding incremental-cache
contamination. Orthogonality is preserved by the matrix exponential in the
retained AO space. Tests compare energy/gradient finite differences,
noncanonical Hessian symmetry, spin coupling, density and orthogonality.

## Follow-on adapters

The chemistry-independent C++ `OrbitalObjective` adapter is reusable:

1. **Localization:** define each objective and its minimization sign (for
   example, negate a maximized localization measure), choose occupied and
   virtual subspaces independently, remove redundant rotations, and validate
   analytic gradients/Hessian actions off stationarity. Begin with a small
   Boys or Pipek-Mezey molecular test and preserve subspace projectors.
2. **CASSCF:** enumerate inactive/active/virtual nonredundant rotations, include
   orbital/CI relaxation in the Hessian action, specify state/root tracking
   and state-averaged weights, and validate against the existing Newton-CG
   machinery. An orbital-only fixed-CI Hessian is not a complete CASSCF adapter.
3. **Periodic:** first define complex anti-Hermitian coordinates, conjugation,
   k weights, Bloch gauge/sewing and occupation conventions. Validate response
   across k points before enabling any periodic capability.

The defining reference is generated from the runtime citation database:

```{vibeqc-cite-entry} greiner_opentrustregion_2026
```

It is cited automatically only when OpenTrustRegion actually runs. Localization
and multiconfigurational follow-ons should also cite
[Høyvik et al. (2012)](https://doi.org/10.1021/ct300473g) and
[Helmich-Paris (2022)](https://doi.org/10.1063/5.0090447), respectively.
