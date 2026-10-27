# MSINDO-SECCM electronic stability

The semiempirical cyclic cluster model (SECCM) is **experimental**. Its
post-SCF stability audit tests the electronic state of the specified finite
cyclic cluster at fixed geometry and embedding. SCF convergence, a passed
local stability test, and a global electronic ground state are different
claims. See [MSINDO](msindo.md) for the method and
[semiempirical methods](semiempirical.md) for supported routes.

## What the audit covers

For a converged closed-shell state with occupied and virtual orbitals, the
audit searches for a negative eigenvalue of the real restricted orbital
Hessian. It rotates occupied and virtual orbitals together for both spins,
preserving the electron count and integer occupations in the orthonormal
INDO basis. The geometry, cyclic translations, Hamiltonian and embedding
remain fixed. The local stability criterion follows Section 10 of
{vibeqc-cite}`lehtola_scf_overview_2020`.

A negative curvature establishes an electronic saddle in this restricted
space. A positive lowest eigenvalue, when established with a converged
eigensolver at a stationary state, supports a local minimum only in that
space. The audit does not test unrestricted spin breaking, complex orbital
rotations, another cluster or periodicity, nuclear displacements or phonons.
It does not compare all distinct SCF solutions or prove a global minimum.

## Read the whole result

`run_seccm` exposes the following result attributes. Its `run_fields` record
uses the corresponding output names for [output integration](output_files.md).

| Result attribute | Output field | Interpretation |
|---|---|---|
| `converged` | General SCF convergence record | Electronic iterations converged; this alone says nothing about stability. |
| `stability_checked` | `seccm_stability_checked` | An orbital stability analysis was attempted. |
| `stability_analysis_converged` | `seccm_stability_analysis_converged` | The stability eigensolver converged. Read this before interpreting its eigenvalue. |
| `stability_eigenvalue` | `seccm_stability_eigenvalue` | Lowest computed full orbital-Hessian curvature in Hartree for the implemented rotation normalization. |
| `internal_instability` | `seccm_internal_instability` | A negative mode passed the finite-descent test described below and remains in the returned state. |
| `n_stability_restarts` | `seccm_stability_restarts` | Number of stability restarts; this is not a count of all possible states. |

The implementation solves a quarter-Hessian problem and reports four times
that eigenvalue. A quarter-Hessian eigenvalue below `-1e-6` Hartree triggers
a scan in both signs of its normalized mode, at angles `0.1`, `0.25`, `0.5`,
`1.0` and `2.0` radians. A scanned energy reduction of at least `1e-6` Hartree
is needed to flag the state or attempt a restart. The negative eigenvalue
is retained even if the scan does not meet that energy threshold. An absent
flag therefore does not by itself prove a nonnegative Hessian.

At a resolved initial frontier gap, the reference policy can return a
converged SCF state even when the stability eigensolver did not converge.
Such a result has `stability_analysis_converged=False`; its eigenvalue
placeholder and absent flag must not be interpreted as a passed audit.

## State-selection policies

Set `CCMOptions.stability` explicitly when comparing states or geometries:

- `"reference"` keeps the Hcore/DIIS trajectory when its initial frontier
  gap is resolved. A detected saddle can be returned with its instability
  flag. At an unresolved frontier gap, the solver attempts state selection
  and stability descent within the iteration budget.
- `"lowest"` attempts descent from detected saddles, including at resolved
  frontier gaps. The name does not promise the lowest of all electronic
  states. Descent can fail or exhaust the budget, and independent calculations
  at neighboring geometries can select different local solutions.

All restarts share the cumulative `max_iter` budget. The native route raises
on SCF nonconvergence; the Python reference route can return an unconverged
result. Preserve refusals and convergence diagnostics when auditing states.
Do not convert a failed alternative into a claim that the reference state
is stable.

## Curves, derivatives and equilibrium claims

Retain each converged alternative separately with its density, occupations,
energy, convergence and stability diagnostics, actual controls and runtime
identity. Compare charge distributions and occupied subspaces, as well as
energies, to detect branch changes. A rotated trial density that lowers the
energy is evidence of descent, not a converged alternative SCF solution.

An electronic finite-difference check rotates the retained occupied space
at fixed geometry and compares the energy curvature with the orbital
Hessian. It neither computes nuclear forces nor establishes a structural
minimum. For a nuclear finite-difference check, verify that both displaced
calculations remain on the intended electronic branch before comparing
their energies with an analytic gradient. Using the same policy string
at both displacements is insufficient evidence of branch continuity.

A fitted volume minimum is a property of the sampled electronic branch.
Before interpreting it as a stable equilibrium, establish electronic
stability and branch continuity around that minimum, together with the
structural stability appropriate to the claim. A stability result at one
representative volume cannot qualify an entire fitted curve. Keep flagged
points and fitting exclusions explicit; do not smooth away state changes
or replace one branch's points with a lower alternative without a separate,
documented analysis.
