# Experimental chi fitted Coulomb support

This opt-in contract builds finite classic or SR fitted operators for fresh,
all-electron, three-dimensional, unsmeared RI RHF/UHF single points. Chi remains
experimental; D93 v1 fitted and post-HF results remain not-qualified. The
contract does not qualify HF, DFT, hybrids, forces, or correlation by inheritance.
The chi construction remains separate from Gamma-CCM.

Import `ChiCoulombControls` and `build_chi_fitted_operator` from
`vibeqc.periodic.chi.fitted`. The low-level builder accepts a system, orbital
basis, auxiliary basis, Cartesian character points, normalized character
weights, an explicit `source='classic'` or `'sr'`, and `controls`. The chi RHF/UHF
library entries accepts `coulomb_controls`; `run_periodic_job` accepts
`chi_coulomb_controls` together with explicit `fit_source`, chi variant and RI
backend. Controls require explicit one-electron and AO-pair radii, reciprocal
kinetic-energy cutoff, and nuclear Ewald real radius, reciprocal radius, and
alpha. There are no cell-independent physical cutoff defaults.

The existing calls without this object preserve their previous behavior.
`support_policy='historical_index_ball'` explicitly selects the classic
origin-centered image ball for reproduction. SR supports only
`physical_pairs`. Historical reproduction is not translation-covariant
physical acceptance. Conflicting legacy cutoff arguments are rejected.

## Finite support and conventions

For classic `physical_pairs`, each AO shell pair includes every lattice image
whose two centers are within `pair_cutoff_bohr`. The shared physical pair
selector supplies a complete candidate enclosure; the native Fourier kernel
applies the actual pair-distance predicate. A common origin shift or changing
an atom's lattice representative therefore changes phases and indices, not
physical membership. Primitive Gaussian Fourier transforms form the reciprocal
three-center source and Coulomb metric. The base reciprocal sphere has radius
`sqrt(2*reciprocal_cutoff_ha)`. An optional tail extends that sphere while fixing
the one-electron support. Base and tail are disjoint, including Gamma boundary
points. Physical fitting screening is currently required to be zero: the
historical screening estimate is not established for the shifted domain.

SR uses the existing native Coulomb-minus-erf real-space integrals with
explicit `omega_bohr_inv`. Its metric uses pair-centered auxiliary spheres;
three-center integrals use complete AO-pair spheres and auxiliary capsules
around those pairs. `auxiliary_cutoff_bohr` controls that distinct real-space
extent. Its long-range reciprocal source uses the explicit reciprocal sphere,
with the existing finite SR zero-mode subtraction. Classic and SR are distinct
finite approximations; matching their requested radii does not equate them.

The Coulomb Fourier kernel is `4*pi/(V*|G+q|^2)`, with its singular zero mode
omitted. Source records specify each engine's zero-mode threshold and SR
correction. Character weights sum to one and energies are per primitive cell.
Raw operator `coulomb(D)` and `exchange(D)` accept Hermitian AO densities;
they exclude the exchange Madelung correction, which SCF adds separately with
its executed overlap and BvK convention.

One-electron AO support is independent of the fitted pair radius. Electron-
nuclear and nuclear-nuclear terms use the same explicit Ewald alpha, real
radius and reciprocal radius. Physical electron-nuclear real-space sources
use the shared AO-midpoint selector. This convention still requires independent
component and support-convergence acceptance; agreement of overlap or shared
builders alone cannot establish physical correctness.

## Identity, recording and resource admission

Returned `ChiFittedOperator` factors are immutable. Its descriptor binds the
source, geometry, orbital and auxiliary bases, q groups, character mesh and
weights, executed domains, metric spectra and retained ranks, kernel and
normalization, and factor digest. `require_identity` rejects reuse under a
different identity. SCF returns `result.fitted_operator` and an executed source
record that also binds the one-electron and nuclear construction. Legacy
`retain_fit_cache` export is refused for this contract. Existing descriptors
without the new domain contract retain their established identities.

Records distinguish exact reciprocal vectors and actual nuclear traversals
from candidate enclosures and lossless selector-replay specifications. A
selector record defines support before integral screening; it is not a claim
that every candidate caused an integral evaluation. Requested radii alone are
not an execution receipt. JSON and numerical allocation budgets, candidate
limits and native workspace limits trigger refusal, never support truncation.
The classic builder requires the native pair kernel; no Python substitution
occurs in this source contract.

## Refinement and diagnostics

`controls.ladder(field, values)` changes one numerical axis while preserving
the source and all other controls. Refine one-electron, pair, nuclear real,
nuclear reciprocal, fitted reciprocal, auxiliary support, metric rank and
screening independently. Classic tail refinement holds the one-electron
support fixed. Auxiliary basis changes are a separate fitting axis. Omega is
a fixed source convention, not a ladder axis. Successive-refinement differences
are measured diagnostics, not certified truncation-error bounds.

`vibeqc.periodic.chi.operator_diagnostics` checks Hermiticity, weighted electron
count and energy decomposition, fixed-density linearity and reciprocity,
directional energy derivatives, metric conditioning/rank, and representation
covariance. An independent energy callback is required for a discriminating
energy derivative check. Separately converged or analytic references must
match the kernel, zero mode, density convention, support and normalization.
SCF convergence cannot replace a fixed-density operator comparison.

Non-diagonal integer SCF extensions, ECP, RIJCOSX, DFT/hybrids, gradients,
optimization, restarts, smearing, +U and post-HF are outside this contract and
are refused at their applicable entry points. Explicit fitted-support
references are rejected even by automatic post-HF source selection. Existing
historical energy pins remain historical evidence; they are not retuned to
certify the new finite domain.


## HF consumers and accepted state

RHF and UHF use the same explicitly selected source. UHF forms
`J[D_alpha + D_beta]` and a separate `K[D_sigma]` for each spin; the corrected
exchange uses the same BvK gauge as RHF. Closed-shell collapse is a property
of identical densities, not a promise that two independent SCF runs find the
same root. The admitted envelope is neutral all-electron 3D, diagonal cyclic
extensions, including skew cells and complex non-Gamma characters, with
integer fixed populations at every character. Fractional or metallic final
occupations and spinlock/MOM under explicit support are refused.

Previous-Fock mixing and density-projector level shifts operate on trial
Focks. They do not rebuild or select another fitted source. Direct RHF/UHF
consume the same shift schedules; explicit fitted UHF uses the spin projector
with occupation one. Initial guesses remain subject to each shared driver's
capability checks. Unsupported combinations raise rather than switching the
Hamiltonian. The final acceptance record (`chi_scf_acceptance`) measures a
fresh unshifted, unmixed Fock, energy, density/orbital residual and electron
counts; iteration limits and terminal failures cannot return a stationary
claim. `stability="not-assessed"` is separate from stationary convergence.

`prepare_chi_hf_reference` in `vibeqc.periodic.chi.reference` builds a sealed,
process-local RHF or UHF state. Supply the same explicit options and source
controls with `hf_reference` to the corresponding library HF entry (or
`chi_hf_reference` to the public runner). Geometry, orbital/auxiliary selection,
spin, occupations, kernel, support, factor and returned-state changes refuse
reuse. Reuse returns the accepted state without another SCF or source build;
it does not convert RHF to UHF or certify a correlation reference. The existing
restricted post-HF admission remains separate and unchanged.

Public results and manifests expose requested controls, the executed support
or factor identity, final-state acceptance and the unresolved stability status.
These diagnostics and references remain experimental and not-qualified;
source tests do not replace independent native component and support-ladder
acceptance. ROHF, metals, ECP, non-diagonal extensions, DFT, hybrids and post-HF
are not admitted by these RHF/UHF checks.
