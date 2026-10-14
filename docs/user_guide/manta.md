---
orphan: true
---

# MANTA integral providers

MANTA is an optional integral and Coulomb construction workstream. Its
experimental interfaces supply a libint reference provider and an independent
McMurchie-Davidson provider for molecular one-electron and Coulomb integrals. Existing calculation defaults are
unchanged. Molecular BUPO-J and RI-BUPO-J are available as low-level consumers and in
the selected RHF/UHF/RKS/UKS adapter with exact terminal confirmation
against their respective direct and RI Coulomb sources;
finite Bloch S/T, Fourier, three-dimensional Ewald nuclear, reciprocal
Hartree and explicit-transfer reciprocal exchange producers are available
for Gamma and multi-k calculations. Bounded all-electron RHF/UHF and pure
LDA/GGA RKS/UKS drivers compose these finite three-dimensional sources;
global and explicit-source range-separated LDA/GGA hybrids are also available. Selected HF and
pure/global/range-separated KS jobs have separate admission and output contracts below. Production
accuracy profiles and CCM MANTA routes remain unqualified. HF and global
hybrid drivers offer strict exchange zero-mode omission and BvK Ewald
correction. Range-separated KS applies that choice only to its resolved
long-range Coulomb coefficient and always retains finite screened zero modes.

Static molecular Mulliken and Lowdin charges, Mayer and Wiberg bond orders,
AO delocalization indices, natural
orbitals, dipoles and the standard population summary follow the one-electron
provider recorded by the actual SCF result. MANTA results use MD overlap and
Cartesian moment integrals. Explicit libint and legacy results retain their
libint route. Selected results require a content-matching basis and, when
supplied, molecule; mismatches fail before integral evaluation. Properties can
inspect an unconverged density without declaring it converged.

Each MANTA property integral build emits a separate
`molecular_property_integrals` output event with its operator/source identity,
measured work and modeled memory. These calls do not modify the SCF receipt.
They use the one-electron adapter's default resource caps; retained earlier
dipole-axis matrices are charged to subsequent axis builds. The budget covers
the integral build, not the retained SCF result or subsequent Python property
algebra. Hirshfeld promolecules require separate qualification: a direct
request refuses and population summaries label them unavailable without
running a legacy atomic SCF. IAO and closed-shell localization use the selected
provider as described below; response properties remain a separate stage.

```python
import numpy as np
from vibeqc import Atom, BasisSet, Molecule, manta

mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])], 0, 1)
basis = BasisSet(mol, "sto-3g")
budget = manta.ResourceBudget()
budget.memory_bytes = 256 * 1024**2
builder = manta.ReferenceJKBuilder(basis, budget=budget)
density = np.eye(basis.nbasis, dtype=np.float64)
result = builder.build(density)
J, K = result.J, result.K
assert result.receipt.provider == "libint"

# Evaluate the same numerical source with MANTA MD primitive kernels.
md_builder = manta.DirectJKBuilder(basis, budget=budget)
md_result = md_builder.build(density)
assert md_result.receipt.provider == "manta"
```

This low-level interface constructs J and K for the supplied density. It does
not perform SCF or assign occupations. `build_g_rhf(D, alpha_hf)` returns
`J(D) - alpha_hf*K(D)/2`. The builder implements the existing native `JKBuilder`
boundary. Separate calls and spin slots do not share incremental density state.

`builder.build_batch([D0, D1, ...], make_j=True, make_k=True)` digests one
ordered quartet traversal into multiple density-specific targets. The result
contains lists `J` and `K` (empty for an unselected target), individual
`density_identities` and a shared receipt. It accepts the same real symmetric
densities as a single build, including indefinite response densities and
density differences. `build_K_pair(D_alpha, D_beta)` uses this fused traversal
for the native spin-pair seam. No density or integral tensor is retained
between calls.

`builder.plan_batch(n_densities, make_j=True, make_k=True)` checks the retained
construction budget before density copies or output allocation. Memory and
scalar contraction work grow with the number of right sides; integral
workspace, shell calls and requested primitive quartets do not. A batch that
exceeds the budget fails explicitly, even when separate single builds fit.
The receipt counts each consumed integral once and every density-target
contraction separately. Its combined density identity records input order;
the single-density identity and provider source identity remain compatible
with `build`. Exported results survive subsequent builds and builder deletion.

The reference provider admits real, full-Coulomb, zeroth-derivative four-center
integrals for single-contraction pure or Cartesian shells through the smaller
of l=6 and the generated libint ERI limit. `capability()` reports that limit.
General contractions must already be represented as separate shell channels;
an unsupported native multi-contraction shell is refused without alteration.
Density arrays must be finite, symmetric, C- or F-contiguous float64 arrays.
Indefinite densities and density differences are allowed.

`DirectJKBuilder` admits real full-Coulomb, erf and erfc orbital kernels through l=6,
independently of the generated libint angular limit. `md_capability()` and
`md_plan()` describe this provider. It uses FP64 Boys functions, Hermite
recurrences, primitive quartet matrix contractions, compensated contraction
sums and exact polynomial pure-shell transforms. General contractions remain
lossless separate shell channels at the basis boundary; an optional bounded
schedule shares their primitive work. Complete force and periodic adapters
require their own subsequent admission.

### Shared contraction channels

`DirectJKBuilder(basis, channel_batch=16)` groups up to 16 consecutive ordered
quartets by their four angular momenta and centers. Channels with exactly
matching primitive exponents share the corresponding Cartesian primitive
integral before contraction. Exponent sets may overlap only partially; no
padding with invented primitives or merging of distinct centers occurs. Pure
and Cartesian channels can share primitive work and apply their own final
transforms. Repeated exponents within a channel use compensated coefficient
aggregation, including signed cancellation.

`channel_batch` is an integer from 1 through 64. The default 1 preserves the
segmented traversal. Sharing composes with `build_batch`, spin-pair exchange
and range kernels. `md_plan(..., channel_batch=...)` reserves the batched
Cartesian sums, representation transforms, tile workspace and coefficient
maps. Work admission includes the worst-case membership scans without
assuming that sharing succeeds. The implementation retains one primitive
quartet workspace and bounded contracted tiles, with no global primitive
quartet cache or tensor over the union of primitive sets.

Receipts retain `primitive_quartets_requested` for comparison with the
unshared shell schedule. For MANTA builds, `primitive_evaluations_measured`
is true and `primitive_quartets_evaluated` reports actual MD primitive
evaluations. The libint adapter leaves that flag false because libint's
internal primitive work is opaque. `contraction_strategy` distinguishes
`segmented` from `shared-channels`. Source identity remains the same because
the physical integral source and screening policy are unchanged. No shell
symmetry reduction or Hermitian averaging is introduced. Savings depend on
channel overlap and ordering; disjoint contractions can incur overhead.

## Range separation

For `DirectJKBuilder`, set `KernelRequest.kernel` to `"coulomb"` for `1/r`,
`"erf"` for `erf(omega*r)/r`, or `"erfc"` for `erfc(omega*r)/r`. `omega` is
finite and nonnegative, in inverse bohr. Full Coulomb requires `omega=0`.
At zero range, erf gives zero and erfc gives full Coulomb. Requests are copied
after admission; later mutation of the request does not change the provider.

```python
request = manta.KernelRequest()
request.kernel = "erfc"
request.omega = 0.3
short_range = manta.DirectJKBuilder(basis, request, budget)
short_range_JK = short_range.build(density)

# The same request structure selects a lower-center fitting kernel.
request.centers = 3
short_range_fit = manta.FittingTiles([basis, basis, basis], budget, request)
```

Fitting requests must set `centers` to the number of supplied basis axes.
Omitting the request preserves full Coulomb. Tile and J/K receipts identify
the kernel and range; source identities include the exact binary64 range
parameter. Full-Coulomb identities remain unchanged. The reference provider
continues to admit full Coulomb only, without silently replacing a range
request. `DirectJKBuilder.build_K_erf(D, omega)` implements the existing native
long-range exchange seam and applies the same strict density admission.

The erf kernel uses scaled Boys values. The erfc kernel integrates the positive
radial interval directly, avoiding cancellation from subtracting nearly equal
full and long-range values. Its adaptive 16/32-point Gauss rules have a relative
target of `2e-14`, at most 256 visited panels and a depth limit of 20. The
exponentially small integration tail is trimmed at `4*n+100` in the shifted
exponent, where n is the maximum radial order. The target and tail control are
numerical evaluation settings, not a rigorous bound on every contracted
integral. Failure to converge within the work cap raises an error. The plan
includes the bounded radial work and recursion workspace; receipts still count
requested primitive products rather than internal quadrature operations.

Raw integral identities do not establish an RI decomposition identity when
separate kernels use different fitting metrics or retained ranks. A fitting
consumer must specify those choices. Range selection here also does not select
a molecular SCF route or a periodic long-range/background convention.

## One-electron matrices

`compute_one_electron(bra, ket, request, budget)` evaluates molecular overlap,
kinetic energy, point-charge potential, or polynomial moments. Both basis sets
may differ in size, geometry, contraction and pure/Cartesian representation;
each admits orbital l through 6. Results contain an owned `values` matrix and
a `receipt` with provider, source identity and actual shell/primitive-pair
counts. No Hermitian averaging is applied to the result.

```python
request = manta.OneElectronRequest()  # overlap
S = manta.compute_one_electron(basis, basis, request, budget).values
request.operation = "kinetic"
T = manta.compute_one_electron(basis, basis, request, budget).values
request.operation = "potential"
request.charges = [(1., [0., 0., 0.]), (1., [0., 0., 1.4])]
V = manta.compute_one_electron(basis, basis, request, budget).values
```

`potential` means the electron potential `-sum(q_A / |r-R_A|)`, with charge
positions in bohr. Use effective charges when that is the intended operator;
this interface does not infer ECP core charges. An empty charge list gives
zero. `moment` evaluates `(x-Ox)^nx (y-Oy)^ny (z-Oz)^nz`, with integer
`powers=(nx, ny, nz)` and `origin=(Ox, Oy, Oz)`. Total polynomial order is at
most 6. Moments have no electron-charge sign. Charges are accepted only for
`potential`; nonzero powers/origin only for `moment`. All numerical parameters
must be finite.

Potential requests also admit `kernel="erf"` or `"erfc"` and a nonnegative
`omega` in inverse bohr:

```python
request.kernel = "erfc"
request.omega = 0.7
V_short = manta.compute_one_electron(basis, basis, request, budget).values
request.kernel = "erf"
V_long = manta.compute_one_electron(basis, basis, request, budget).values
```

These evaluate `-sum(q_A*erfc(omega*r_A)/r_A)` and
`-sum(q_A*erf(omega*r_A)/r_A)`, respectively. Their sum is the full point-charge
potential. At zero range the erf contribution is zero and erfc is the full
potential. The default `kernel="coulomb"` requires `omega=0` and retains the
existing full-range source identity. Other one-electron operations accept
only that default. Receipts record the executed kernel and range.

The nuclear evaluator uses the point-charge limit of the Hermite Coulomb
recurrence, with product exponent `p` and `theta=omega/sqrt(p+omega^2)`.
Short-range seeds integrate the positive interval `[theta,1]`, following
the range-separated radial formulation in SHARK Eq. 54, rather than
subtracting nearly equal full and long-range integrals. This retains tiny
erfc tails at large finite ranges. The existing `manta_md` citation route
covers this primitive. It supplies the screened kernel required by Ewald
nuclear operators; lattice images and reciprocal/background terms must still
be supplied by a separately qualified periodic composition.

`plan_one_electron` admits this rectangular two-index computation independently
of four-center J/K work. `shell_calls` caps shell pairs, the shared budget's
`primitive_quartets` field caps primitive pairs for this adapter, `scalar_terms`
caps matrix elements, and `kernel_terms` includes charge and moment work.
The memory model accounts for input bases/charges, output and binding storage,
one shell-pair workspace and backend headroom. Admission precedes output
allocation and numerical payload scans. Python calls retain the GIL; native
callers must keep the request and bases fixed for the duration of the call.
The source hash includes both basis payloads and all operator parameters.
For erfc requests, the work and storage estimates additionally reserve the
bounded positive-interval radial quadrature, including derivative requests.

## Coulomb metric and three-center tiles

`FittingTiles([aux_left, aux_right])` supplies `(P|Q)` metric tiles.
`FittingTiles([aux, orbital_bra, orbital_ket])` supplies `(P|mu nu)` tiles;
orbital bases may differ. Each auxiliary axis admits l through 7 and each
orbital axis l through 6, in pure or Cartesian representation. The source is
real molecular full, erf or erfc Coulomb with derivative order zero. This provider computes
integrals only; it does not select a fitting rank or solve a metric.

```python
metric = manta.FittingTiles([basis, basis], budget=budget)
metric_tile = metric.evaluate((0, 0))["values"]
three_center = manta.FittingTiles([basis, basis, basis], budget=budget)
three_center_tile = three_center.evaluate((0, 0, 0))["values"]
```

These examples use the orbital basis as an illustrative auxiliary basis; an
actual RI method must choose an appropriate fitting basis. `evaluate` accepts
one shell index per axis and returns an owned array with those shell extents,
last index contiguous, plus source identity, normalization and actual call and
primitive-product counts. The normalization is the same shell-wide axial
convention as the four-center provider. Internal unit factors in the Coulomb
recurrence are not zero-exponent auxiliary functions supplied by the caller.

`plan_fitting(basis_axes, budget)` reserves immutable copies of all axis bases
and one evaluation/export workspace. It does not allocate a dense metric or
three-index tensor. The plan's counts describe one complete ordered pass;
`shell_calls`, `primitive_quartets`, `scalar_terms` and `kernel_terms` bound
shell calls, primitive products, integral elements and modeled kernel work for
that pass. Repeated evaluations are separate calls. Retained exported tiles,
concurrent evaluations and other caller storage belong in `other_live_bytes`.
The provider copies bases only after admission and has no mutable integral
cache. Every ordered tile is available, with no permutation reduction or
screening threshold. No silent backend fallback occurs.

## Streamed molecular RI-J/K

`RIJKBuilder(orbital, auxiliary, options=RIOptions(), request=KernelRequest(),
budget=ResourceBudget())` constructs a separate RI source. It uses MANTA
metric and three-center tiles with the selected full, erf or erfc kernel.
The request retains `centers=4` at this J/K boundary; the builder derives its
two- and three-center requests. Selecting RI changes the approximation from
the exact four-center builder. It does not silently select an auxiliary basis
or replace exact exchange with fitted exchange in another builder.

```python
auxiliary = BasisSet(mol, "def2-svp-jk")
options = manta.RIOptions()
options.mode_block = 8
ri = manta.RIJKBuilder(basis, auxiliary, options, budget=budget)
fitted = ri.build(density)
print(ri.metric_info.rank, fitted.receipt.coupling_passes)
```

Let `T[P,mu,nu] = (P|mu nu)` and `V[P,Q] = (P|Q)`. The builder assembles the
dense auxiliary metric, checks its symmetry, and diagonalizes its lower
triangle without averaging or diagonal shifts. It retains eigenvalues strictly
greater than `max(absolute_cutoff, relative_cutoff * largest_eigenvalue)`.
Defaults are `1e-12` and `1e-10`, respectively. A negative eigenvalue below
`-max(absolute_cutoff, negative_relative_tolerance * largest_eigenvalue)` fails
admission; the default negative relative tolerance is `1e-10`. Tiny discarded
negative values are never inverted. The retained factor is
`W = U_retained / sqrt(eigenvalues_retained)` and defines `V+ = W W.T`.
Metric rank, cutoff, extremal eigenvalues and factor identity are inspectable
through `metric_info`. Diagnostic `metric` and `factor` arrays are copies.

For J, two identical ordered passes compute `g = T D` and
`J = T.T (V+ g)`. The receipt's Coulomb energy is `g.T V+ g / 2`.
Its metric residual is `norm(V V+ g - g)`, including discarded-space content;
it is not solely a retained-space solver error. `metric_residual_evaluated`
distinguishes an evaluated diagnostic from an exchange-only or zero-rank
build. A zero retained rank produces explicit zero J/K with no coupling pass.

For K, a bounded block of retained modes forms
`B[r,mu,nu] = sum_P W[P,r] T[P,mu,nu]`, then accumulates `B[r] D B[r].T`.
The full three-index tensor is not retained. A full J/K build requires
`2 + ceil(rank / mode_block)` three-center passes when rank is nonzero;
`mode_block` is an integer from 1 through 64. Larger blocks trade memory for
fewer integral passes. This reference algorithm can be slower than an
existing in-core RI build. No linear-scaling or speed advantage is claimed.

`plan_ri` reserves the dense metric, eigensolver/factor, fitted blocks and live
targets before copying bases or generating integrals. Its work caps include
metric setup plus a full J/K or paired-spin K build at the worst-case auxiliary rank. Dense
metric storage remains quadratic in auxiliary size and must fit the budget.
`metric_info` reports actual setup calls; each build receipt separately reports
actual coupling passes, calls, primitive products and integral elements.
Source identity includes the raw integral sources, rank policy and computed
retained factor. Other live results and diagnostic copies belong in
`other_live_bytes`.

Native `build_K_pair` shares each fitted block between both spin densities;
it validates both densities before generating coupling tiles.
`build_exchange_pair` exposes the same operation with named `K_alpha` and
`K_beta` matrices, individual density identities and a shared work receipt.
The builder implements the native real-density `JKBuilder` interface and has
focused RHF/UHF/RKS/UKS host checks. Density and ownership rules match the direct
builder. It has no incremental state. An erf or erfc builder uses that same
kernel in both its metric and coupling; separate fitted kernels do not imply
`RI(full) = RI(erf) + RI(erfc)`. The native `build_K_erf` hybrid seam is not
enabled for this RI builder. Molecular job selection uses the separate
selected-engine adapter described below. Periodic fits and nuclear derivatives
require their own adapters.

## RI metric inverse and projector response

`RIMetricResponse(metric, options=RIOptions(), budget=ResourceBudget())`
is a dense molecular derivative primitive. It shares the exact retained
factor and lower-triangle convention of `RIJKBuilder`, including the strict
positive rank cut and refusal of significant negative eigenvalues. It owns
its eigensystem; `inverse`, `projector` and `factor` return independent arrays.
The input and directions/seeds must be nonempty square, finite, real float64
NumPy arrays in C or Fortran contiguous layout. Symmetry is checked with the
RI tolerance `1e-12 * max(1, max(abs(matrix)))`; the lower triangle defines the
source, without averaging the two triangles.

```python
response = manta.RIMetricResponse(ri.metric, options, budget)
derivative = response.apply(metric_direction)
d_inverse, d_projector = derivative.inverse, derivative.projector
metric_adjoint = response.adjoint(inverse_seed, projector_seed)
print(response.info.rank, response.info.rank_stability_radius)
```

The response includes motion between retained and discarded metric modes,
even when discarded eigenvalues are nonzero. An ordinary inverse derivative
`-inverse @ metric_direction @ inverse` misses those terms. Degenerate
retained modes use the analytic divided-difference limit, so arbitrary
eigenvector gauges do not enter. The adjoint uses the full real symmetric
Frobenius pairing: off-diagonal terms occur twice. Seeds describe symmetric
matrices, not packed independent lower-triangle coordinates. The method
citation route is `manta_ri_metric_response` (Higham, 2008, Chapter 3).

`info` reports the cutoff, nearest eigenvalue distance, retained/discarded
gap and a numerical guard `64 * eps * dimension * max(max(abs(eigenvalues)),
cutoff)`. A cutoff distance at or below that guard is refused. For a stable
partition, `rank_stability_radius = (cutoff_distance - guard) /
(1 + relative_cutoff)` describes a conservative local spectral-norm radius
using the computed spectrum and the cutoff's Lipschitz constant. It is a
rank-partition diagnostic, not a rigorous eigensolver error bound or a step
size guarantee for nuclear displacements. The separate negative-eigenvalue
admission still applies at each new geometry. `has_split` distinguishes a
real retained/discarded gap from its zero placeholder when one space is empty.
A resolved zero-rank metric has zero inverse and projector responses.

`plan_ri_metric_response(dimension, options, budget)` admits dense storage and
modeled cubic work before copying matrix payload or solving the eigenproblem.
Its budget covers setup plus one response or adjoint, including binding copies,
live inputs, outputs, eigensolver and multiplication temporaries. Repeated
calls each have that scope; caller-retained exports or earlier results belong
in `other_live_bytes`. This is a workspace model, not a measured RSS bound.
The response reports its metric source and direction identities and modeled
peak. This primitive alone does not enable RI nuclear gradients: those also
need three-center derivatives, auxiliary motion, Pulay and method terms.

## Fixed-density RI Coulomb gradient

`compute_ri_coulomb_gradient(molecule, orbital, auxiliary, density,
options=RIOptions(), request=KernelRequest(), budget=ResourceBudget())`
returns the nuclear derivative of the fitted Coulomb energy
$E_J=\tfrac12(B:D)^T V^+(B:D)$ at fixed AO density. The density is the total
real symmetric density; no spin factor is added by this component. The
request has `centers=4` and derivative order zero, with full, erf or erfc
Coulomb; the consumer derives its two- and three-center derivative requests.
Orbital l through 6 and auxiliary l through 7 retain the underlying tile
capabilities, subject to the aggregate memory/work caps. Both bases must have
valid atom ownership and centers exactly matching the supplied molecule.

```python
gradient = manta.compute_ri_coulomb_gradient(
    mol, basis, auxiliary, density, options, budget=budget
)
print(gradient.values, gradient.receipt.coulomb_energy)
print(gradient.receipt.ri_source_identity)
```

`values`, `coupling` and `metric` are `(n_atoms, 3)` energy derivatives in
atomic units, with `values = coupling + metric`. The coupling term streams
one raw three-center pass for the fitting right-hand side, followed by
analytic derivative tiles for every orbital and auxiliary center. The metric
term contracts two-center derivatives with the retained-inverse adjoint,
including the moving projector. It differentiates the same lower-triangle
metric used for the RI energy. No complete three-index tensor is allocated.

The receipt's `ri_source_identity` and `metric_response.factor_identity`
match an `RIJKBuilder` with the same bases, kernel and options. The gradient
source additionally binds atom ownership and the output atom count. Actual
metric/coupling and derivative shell calls, evaluated primitive integrals,
consumed elements, energy and discarded-space metric residual are reported.
A resolved zero-rank fit returns zero components after metric setup, with no
coupling or derivative passes and `metric_residual_evaluated=False`.
Cutoff crossings use the metric-response refusal described above.

`plan_ri_coulomb_gradient` accepts the same arguments except the density and
admits the combined native work before copying that density. Its conservative
peak sums the component workspaces and margins, plus resident matrices,
provider snapshots and caller-declared `other_live_bytes`. Only dense
auxiliary metric/adjoint matrices and AO density storage are retained between
tiles. `manta_ri_coulomb_gradient` is the citation route.

This is a Coulomb energy derivative component. Exchange, one-electron,
nuclear, Pulay and XC terms are not part of its result, and it is not a full
SCF force or an admission for geometry optimization. Complete HF compositions
are provided by the interface below and the selected RHF/UHF gradient adapters.

## Fixed-density RI exchange gradient

`compute_ri_exchange_gradient(molecule, orbital, auxiliary, density_alpha,
density_beta, options=RIOptions(), request=KernelRequest(),
budget=ResourceBudget())` differentiates
$E_K=-\tfrac12\sum_\sigma\operatorname{Tr}(D_\sigma K_{RI}[D_\sigma])$
at fixed per-spin real symmetric densities. For a restricted total density
use `density_alpha = density_beta = density / 2`. The returned derivative
already includes the negative exchange-energy sign and the one-half factor.
Apply a hybrid exchange fraction separately when composing a method.

`RIKGradientResult` contains `values`, `coupling`, `metric`,
`metric_response` and a receipt with `exchange_energy`, both density
identities and their combined identity. The RI source, factor and paired-spin
density identities match `RIJKBuilder.build_exchange_pair`. The two derivative
components obey the same atom-map, units, full/erf/erfc kernel and rank-cutoff
contracts as the Coulomb component. Opposite spins contribute additively;
there is no cross-spin exchange term. The citation route is
`manta_ri_exchange_gradient`.

The metric adjoint includes all raw auxiliary modes, including discarded
modes needed to differentiate the retained projector. Raw three-center
matrices are processed in auxiliary blocks bounded by `RIOptions.mode_block`;
this is the same block-size option used for retained-mode RI J/K builds, but
here it bounds raw auxiliary blocks. Only bounded blocks and dense auxiliary
metric/adjoint matrices persist between tile evaluations. With
`b = ceil(n_auxiliary / min(mode_block, n_auxiliary))`, the reference schedule
uses `b*(b+1)/2` raw coupling passes and `b` derivative coupling passes. Each
pass visits every ordered shell triple, and the receipt reports those actual
passes, shell calls, evaluated primitive integrals and consumed elements.

`plan_ri_exchange_gradient` covers both density copies, live blocks, shared
metric response and aggregate work. It inherits the Coulomb plan's
conservative workspace allowance and adds exchange block work/storage.
Changing block size changes work and memory, while preserving the numerical
source. This dense-metric reference can require many integral passes; no
speed advantage is claimed. A resolved zero rank returns zero energy and
derivatives after metric setup. This remains an exchange energy component;
complete selected RI SCF forces and optimization require the full method
composition and stationarity checks.

## Complete HF Lagrangian with RI components

`compute_hf_ri_lagrangian_gradient(molecule, basis, auxiliary, density_alpha,
density_beta, energy_weighted_density, options=RIOptions(), fit_j=True,
fit_k=True, budget=ResourceBudget())` combines the nuclear, core, overlap
Pulay, Coulomb and exchange derivatives for a real all-electron molecular
HF Lagrangian. The two densities are per-spin; the energy-weighted density
is their total. Each `fit_j`/`fit_k` flag independently selects exact RI or
exact four-center integrals for that component. At least one flag must be
true; use `compute_hf_lagrangian_gradient` for the all-exact composition.
Both RI components use the explicitly supplied auxiliary basis, rank policy
and full-Coulomb kernel. This HF composition has no XC or range-separated
exchange term.

`RIHFGradientResult` has the existing `values`, `nuclear`, `one_electron`,
`two_electron`, `pulay` and `receipt` fields plus separate `coulomb` and
`exchange` arrays. `coulomb_ri` and `exchange_ri` carry the component receipts,
and `coulomb_metric` and `exchange_metric` carry their rank-response evidence.
An exact component has `None` in the corresponding RI fields. The combined
receipt records the component-specific source identity, the identities of
both densities and the energy-weighted density, and summed executed work.

`plan_hf_ri_lagrangian_gradient` omits the three density arguments. It admits
the complete composition before input copies or generated integrals, sums
component workspaces conservatively, and counts `other_live_bytes` once.
When both components use RI, no four-center derivative tiles are planned or
generated. With one exact component, a single ordered four-center derivative
pass contracts only that component. The citation route is
`manta_ri_hf_gradient`.

The low-level caller owns stationarity and the energy-weighted density.
Independent fixed-density Lagrangian differences qualify each component;
relaxed RHF/UHF energies qualify all three RI/exact combinations, and an
independent density-fitting implementation checks full RI forces including
auxiliary response. The selected RHF/UHF gradient and native geometry adapters
below retain and check the original SCF fitting configuration.

## One-electron derivatives

`compute_one_electron_derivative(bra, ket, request, derivative, budget)` returns
one analytic Cartesian derivative matrix, with the same shape and conventions
as `compute_one_electron`. Both bases retain the orbital limit `l <= 6`.
`OneElectronDerivative.target` selects `"bra"`, `"ket"`, `"charge"` or `"origin"`;
`axis` is 0, 1 or 2. Its defaults select the bra x component.

```python
request = manta.OneElectronRequest()
request.operation = "potential"
request.charges = [(1.0, [0.0, 0.0, 0.0])]
derivative = manta.OneElectronDerivative()
derivative.target = "charge"
derivative.charge_index = 0
derivative.axis = 2
dv_dz = manta.compute_one_electron_derivative(
    basis, basis, request, derivative, budget
)
```

The bra or ket target differentiates the physical center of the shell on that
matrix axis at fixed exponents and normalized coefficients. For a complete
matrix this is also the derivative under a common translation of all centers
on the selected axis. An atomic-gradient consumer must select the rows or
columns belonging to each moving atom and add both roles. The charge target
moves only the indexed point charge in a potential request; its magnitude
stays fixed. The origin target moves the common origin of a polynomial moment.
An absent power along that axis gives an exact zero origin derivative. Other
selector/operator combinations and nonzero charge indices on other targets
are refused.

Basis derivatives use the analytic raised/lowered Cartesian recurrence with
the original shell normalization and pure transformation. Charge derivatives
use the next Hermite Coulomb derivative directly. Moment-origin derivatives
lower the selected polynomial power. These paths do not use finite differences.

Bra, ket and charge derivatives also admit erf/erfc potential requests at
fixed `omega`. The center derivatives and all charge-position derivatives
sum to zero under a common translation. A derivative of `omega` itself, or a
stress term caused by changing an Ewald splitting parameter, is not supplied
by these fixed-range matrices.

`plan_one_electron_derivative` accepts the same arguments and admits one
component over the complete ordered shell-pair traversal. It includes raised
angular work, a bounded component workspace and the returned matrix.
`primitive_evaluations_upper_bound` caps primitive operator evaluations through
`budget.primitive_quartets`; `primitive_pairs` counts unshifted primitive pairs.
The receipt records the actual primitive evaluations, selector and source
identity. An operator evaluation includes the complete explicit charge list
for a basis derivative. Each returned matrix owns its storage.

These matrices do not yet assemble forces, nuclear repulsion, ECP derivatives,
Pulay terms, XC response, RI metric/rank derivatives or cell stress.

## Coulomb shell-center derivatives

`DerivativeTiles(basis_axes)` supplies analytic first derivatives of molecular
two-, three- and four-center Coulomb tiles. Axis roles match `FittingTiles`
for two and three centers; all four axes are orbital bases for four centers.
Orbital angular momenta through 6 and auxiliary angular momenta through 7 are
admitted in pure or Cartesian representation. The derivative recurrence uses
internal raised Cartesian orders while retaining the original shell
normalization and pure transforms.

```python
derivatives = manta.DerivativeTiles([basis, basis, basis, basis], budget)
# Move the center of the first shell factor along Cartesian x.
dx = derivatives.evaluate((0, 0, 0, 0), center=0, axis=0)["values"]
```

`center` selects a supplied basis axis and `axis` is 0, 1 or 2 for x, y or z.
Each evaluation returns one owned component with the ordinary shell extents
and last index contiguous. The derivative is with respect to that shell
center's position in bohr, with coefficients, exponents and range parameter
fixed. Moving a nucleus requires summing contributions from every shell factor
on that nucleus, including repeated factors in one integral. These tiles do
not themselves assemble a molecular force, density response, Pulay term, RI
metric/projector derivative, periodic image derivative or stress.

Omitting the request selects full Coulomb and first derivative. An explicit
`KernelRequest` must set `derivative=1` and `centers` equal to the supplied
axis count. Full, erf and erfc kernels are available. Complex scalars, image
translations, higher derivatives and other kernels are refused. Selecting
first derivative here does not enable derivative requests on `DirectJKBuilder`.

`plan_derivatives(basis_axes, budget, request)` admits one complete ordered
pass over all shell-index combinations and all `3 * centers` derivative
components. `scalar_terms` caps returned component elements;
`primitive_quartets` caps the upper bound on raised/lowered primitive integral
evaluations, rather than just original primitive products. Workspace and
exports hold one component at a time. Tile receipts distinguish
`primitive_products_requested` and actual `primitive_integrals_evaluated`;
s shells need only the raised term. Source identities include the derivative
order, basis roles, kernel and range. The component indices accompany each
tile. There is no finite-difference approximation in production evaluation.

## Molecular BUPO-J

`BupoJBuilder(basis, options=BupoOptions(), budget=budget)` implements a
molecular full-Coulomb J map for a supplied real symmetric density. Pure and
Cartesian orbital shells through l=6 are admitted. `options.near_backend`
selects `manta` (default) or `libint` for exact near-field tiles; libint retains
its generated angular limit. Far moments always use MANTA MD and receipts
report that provider separately. This is a J-only native `JKBuilder`:
exchange requests fail, and a method must compose its own exact or RI K.

```python
options = manta.BupoOptions()
options.order = 12
options.integral_tolerance = 1e-11
options.bubbles.leaf_capacity = 32
bupo = manta.BupoJBuilder(basis, options, budget)
approximate = bupo.build(density)
print(approximate.receipt.near_shell_calls, approximate.receipt.far_blocks)
print(approximate.receipt.approximation_j_bound)
```

The defaults are order 6, opening ratio 0.5 and absolute per-integral
approximation tolerance `1e-10`. Orders 0 through 12 are supported. Extent
and hierarchy controls are nested in `options.extents` and `options.bubbles`.
The builder requests full and tail moments at its own expansion order,
independently of the standalone extent option's export settings. Opening zero
forces an exact near-only partition; tolerance zero admits only a zero-bound
far approximation. The finite absolute density norm scales the reported J
bound, so the per-integral tolerance is not an SCF energy or J tolerance.

Construction fixes the geometry, envelopes and accepted partition. Every
far candidate must satisfy all three Gaussian error terms below. Rejected
candidates descend until their interactions can be evaluated directly.
The partition is independent of density. Its completed hash is checked on
every build, preserving a fixed linear, reciprocal approximate Coulomb map.

Execution contracts streamed shell-pair moment panels into leaf moments,
translates them upward, applies each far block to both participating nodes
using the matrix and its transpose, then translates fields downward with the
upward translation's transpose. A second panel pass forms AO J blocks.
Near interactions use one exact tile per unordered shell-pair pair and digest
both directions together. Off-diagonal shell-pair density weights include
both AO orientations. Canonical pair reconstruction uses transposes without
post-build averaging. Signed, indefinite and delta densities are supported;
no incremental history is retained.

The builder stores linear tree metadata and node moment/field vectors, a
quadratic AO result, and bounded integral/moment/multipole workspaces. It does
not retain a global four-index tensor, global AO-pair moment tensor, or
quadratic list of far blocks. A deterministic traversal is replayed instead.
`builder.plan` reports aggregate setup plus one-build work admission, maximum
accepted approximation bounds, and conservatively modeled memory including
live setup objects, density copies, targets, node vectors and exports.
`shell_calls` also charges extent inventory and setup/execution traversal
visits. `scalar_terms` includes setup envelopes and contraction work.
The opaque libint recurrence work is controlled by shell/primitive caps;
its internal primitive evaluations are not claimed as measured.

`build` returns an owned J and receipt. `coulomb_energy` is `sum(D*J)/2`.
`approximation_j_bound` is the maximum accepted integral approximation bound
times `sum(abs(D))`; component bounds are reported separately. The total and
components each round upward, so their displayed sum need not agree in the
last bits. Floating-point kernel and contraction errors are separate from
this analytic approximation ledger. Actual near calls, primitive requests,
moment panels, primitive moment pairs, far blocks and native build wall time
are recorded. The source and density identities distinguish the immutable
operator from its current input. Retained results and concurrent work belong
in the caller's `other_live_bytes` budget.

Set `options.collect_timings=True` for optional component measurements.
The plan reports provider, extent, tree, error-model and partition-planning
setup times, alongside total native setup time. Build receipts separate
input validation, leaf moments, upward translations, far error admission,
far matrix applications, near integrals, near digestion, downward translations,
reverse moment panels and final reconstruction. These regions are disjoint;
their sum excludes traversal/control overhead and need not equal total time.
Profiling is off by default, leaving component fields zero and avoiding clock
reads per tile or node. It does not change source identity or numerical output.
The molecular diagnostic in `benchmarks/manta_bupo.py` measures setup, repeated
builds, process peak RSS, work counts and accuracy against optimized libint.
Its comparisons include exact near-only BUPO to isolate the canonical
quartet schedule from far-field approximation. Process RSS includes runtime
and validation storage and must not be interpreted as kernel workspace.

The molecular SCF adapter below admits `bupo` with an independent exact
terminal rebuild before claiming convergence. BUPO gradients/stress, incremental builds,
periodic/CCM sources and shared symmetry reduction remain separate gates.
No end-to-end speed advantage is claimed. The citation route is `manta_bupo`.

## Paired BUPO three-center coupling

`BupoFittingCoupling(orbital, auxiliary, options, budget)` supplies the fixed
molecular full-Coulomb approximate coupling B and its adjoint. It is a
low-level fitting primitive; it does not solve the metric or select SCF.
Orbital shells admit l <= 6 and auxiliary shells l <= 7, pure or Cartesian.
Exact near tiles and all moments use MANTA MD. The complete three-index
coupling tensor is never retained.

```python
options = manta.BupoFittingOptions()
options.order = 12
options.coupling_tolerance = 1e-10
coupling = manta.BupoFittingCoupling(orbital_basis, auxiliary_basis, options)
rhs = coupling.forward(density).values
# Solve the independently defined exact metric for coefficients first.
j_matrix = coupling.adjoint(coefficients).values
```

`forward` accepts any finite real AO matrix and returns an auxiliary vector;
`adjoint` accepts an auxiliary vector and returns an AO matrix. Inputs must be
contiguous float64 arrays. The pairing is the full AO sum, including both
orientations of off-diagonal shell pairs: `c @ forward(D) == sum(D * adjoint(c))`
up to floating-point summation. No triangular packing factor is implicit in
user input. Outputs own their storage.

Auxiliary shells are atomic leaves. For each one, a depth-bounded traversal
of the pair tree chooses exact near tiles or a Gaussian-ledger-certified
far block. Forward uses leaf moments, upward W translations and transposed
T interactions. Adjoint uses the same partition and the transposed operations,
then reverse leaf panels. Setup fixes and hashes that partition; both passes
replay it and verify actual work counts. There is no stored rectangular
interaction list. Extent controls, tree controls, order (0 through 12), opening
(default 0.5) and coupling tolerance (default 1e-10) are independent of input
values. Opening zero forces the exact near limit.

The receipt's `approximation_bound` bounds the largest output-component
coupling error by the maximum admitted element error times the input L1 norm.
Compact truncation, penetration and full-moment-tail terms are also reported.
The plan also supplies a conservative uniform `maximum_exact_coupling`
envelope from L1 and potential bounds, without scanning every three-center
combination. These bounds exclude floating-point error, metric conditioning
and RI basis error. Independent libint/libcint arrays check each pass, and adjoint,
linearity, positive-metric energy and directional-derivative tests check their
composition. The metric remains an independently supplied exact positive
retained subspace; no bound relative to exact four-center energy is implied.

The plan conservatively admits setup plus one forward and one adjoint pass,
including both replay traversals, moment panels, node vectors, one exact tile,
query/translation matrices, input/output copies and caller `other_live_bytes`.
Near-provider work admission retains the complete ordered 3c upper bound even
when few near tiles execute. Plan per-pass counts and receipts distinguish
actual near calls, primitive products, far blocks and moment work. No speedup
or large-system scaling is claimed. The citation routes are `manta_bupo` and
`manta_md`.

## Molecular RI-BUPO-J with an exact metric

`BupoRIJBuilder` combines `BupoFittingCoupling` with the existing dense exact
MANTA Coulomb metric and its fixed positive eigenprojector. This is a bounded
small-system molecular J consumer. It does not supply exchange, derivatives,
range-separated fitting or a streamed large-metric solver.

```python
builder = manta.BupoRIJBuilder(
    orbital_basis, auxiliary_basis,
    coupling_options=options,  # BupoFittingOptions
    metric_options=manta.RIOptions(),
)
result = builder.build(density)
j_matrix = result.J
energy = result.receipt.coulomb_energy
exact_ri = builder.build_exact(density)
```

For the same approximate coupling in both passes, it evaluates
`rhs = B_tilde @ D`, `coefficients = V_plus @ rhs`,
`J = B_tilde.T @ coefficients` and `EJ = 0.5 * rhs @ coefficients`.
The actual implementation applies the retained factor F with `V_plus = F @ F.T`
and evaluates energy as `0.5 * norm(F.T @ rhs)**2`. Inputs are contiguous
float64 symmetric real densities; J, rhs, coefficients, metric and factor
exports are owned copies. A zero retained rank gives zero J and energy while
still reporting the discarded rhs norm in the metric residual. No metric
shift, averaging or approximate-metric replacement is introduced.

`build_exact` streams the independent exact MANTA three-center source with
that identical metric/factor and returns the exact RI-J reference. Its source
identity is separate from the approximate source. It does not compare with
exact four-center J, whose difference includes RI basis error. The constructor
and plan admit the retained reference producer and its confirmation work;
confirmation is an explicit call at this low-level boundary.

Each build reports both coupling-pass receipts, positive retained rank,
energy, rhs norm and the full residual norm `norm(V @ coefficients - rhs)`.
That residual includes discarded-space components and is not automatically
an iterative-solver failure. The coupling-error ledger is propagated through
the actual fixed metric factor. With per-entry coupling error eps, auxiliary
size m, exact coupling envelope Bmax, delta = sqrt(m) eps norm(D, 1), and
M >= norm(V_plus, 2), the reported bounds are:

- Energy: `norm(coefficients, 2) * delta + 0.5 * M * delta**2`.
- J, maximum element: `eps * norm(coefficients, 1) + sqrt(m) * Bmax * M * delta`.

The first J term is the adjoint error at fixed coefficients; the second
propagates the forward error through the metric. `metric_inverse_norm_bound`
uses the conservative `norm(F, 'fro')**2` bound. These can be loose for an
ill-conditioned fit. They cover coupling approximation relative to exact RI
with the same factor, excluding floating-point/eigensolve error and RI basis
error. They do not establish a variational upper bound relative to exact
four-center energy. Independent dense metric/coupling references, separate
pass-defect checks, rank-zero/duplicate/nearly-dependent auxiliaries, linearity,
reciprocity and energy finite differences qualify this implementation.

The combined plan includes coupling setup and both passes, exact metric and
reference storage/work, simultaneous result and solve buffers, and caller
live bytes once. Reusing the existing RI reference conservatively retains its
RI-K planning allowance even though RI-BUPO executes only J. No performance or
linear-scaling claim follows from this dense-metric reference. The molecular SCF adapter below applies an independent exact-RI terminal
confirmation with the same metric policy.

## Conservative shell-pair extents

`compute_pair_extents(basis, options, budget)` supplies molecular geometry
metadata for the BUPO consumer. The extent API itself does not build or approximate
J/K. `ExtentOptions` defaults to pointwise tolerance `1e-14`, absolute tail-mass
tolerance `1e-12`, minimum radius 2 bohr, maximum radius 10000 bohr, and tail
moment order 6 (admitted range 0 through 12). Orbital shells admit l through 6.

The owned result has one `PairExtent` for each unordered shell pair `(a,b)`
with `b <= a`, in stable shell order. Each pair records a center, radius,
`pointwise_bound`, `tail_mass_bound`, `tail_moment_bounds`, full-space
`l1_norm_bound`, `linfinity_bound` and `potential_bound`. These common bounds
apply to **each AO product component** in that shell pair; they are not summed
density bounds. A consumer must include absolute density contraction weights.
A radius or search cap leaves `always_near=True` and an explicit reason. Such
pairs must retain direct evaluation, regardless of their reported radius.

For a primitive product with exponent p and Gaussian product center P,
absolute contraction weights and pure-transform row norms multiply
`(r + |P-A|)^la * (r + |P-B|)^lb * exp(-p*r*r)`, where r is distance from P.
Each nonnegative polynomial term is bounded beyond its radial maximum.
Integrated tail bounds use positive upper-incomplete-gamma recurrences;
`tail_moment_bounds[k]` bounds the integral of
`|r-C|^k * |AO_mu(r)*AO_nu(r)|` outside the pair sphere centered at C.
Setting `retain_full_moments=True` also exports `full_moment_bounds[k]`
for the same absolute radial integral over all space. The default is false,
leaving that vector empty. Retained full moments contribute to output and
memory admission; a bound outside finite binary64 fails explicitly. These
full-space bounds are needed when a consumer uses full Gaussian multipoles
but bounds its compact-support remainder separately.
The pointwise criterion has density units and the tail mass has overlap units.
They are different controls. The potential supremum follows by splitting the
Coulomb integral at a radius and optimizing the inside/outside bounds:
`3*(pi/2)^(1/3) * linfinity_bound^(1/3) * l1_norm_bound^(2/3)`.

The deterministic center is the geometric midpoint of the two shell centers.
This conservative three-dimensional envelope is a MANTA extension to
[Neese et al.'s BUPO construction](https://doi.org/10.1021/acs.jpca.4c07415),
which uses a coefficient-weighted center and axial walkers. Absolute primitive
weights protect signed contractions and nodal/off-axis distributions from
cancellation in the extent estimate. Midpoint enclosure can be looser than the
published scan. No published BUPO speedup or J-error certificate is claimed.

Bounds are evaluated in log space using the platform's `long double` type
(which may have binary64 precision), with outward binary64 headroom. Positive
underflows round upward instead of becoming zero; overflowing intermediate
geometry or exponent arithmetic fails explicitly. This is not
certified interval arithmetic. A future Coulomb error ledger must separately
account for tail penetration, finite multipole order and floating-point error.
`plan_pair_extents` preflights output, primitive workspace and bounded search
work. The budget's `shell_calls` and `primitive_quartets` fields cap unordered
shell pairs and primitive-pair inventory here; no integrals are evaluated.
Receipts record actual envelope probes, failed extents, source identity and
modeled memory. The citation route is `manta_bupo_extents`.

## Selecting a molecular SCF engine

The ordinary `vibeqc.run_rhf`, `run_uhf`, `run_rks` and `run_uks` entry points
accept keyword-only engine, Coulomb and exchange options:

```python
import vibeqc as vq

engine = vq.IntegralEngineOptions(backend="manta", channel_batch=8)
result = vq.run_rhf(molecule, basis, integral_engine=engine)
assert result.engine_provenance.full_build_confirmed
assert result.engine_provenance.stationarity_confirmed
forces = -vq.compute_gradient(molecule, basis, result)
```

Omitting all three keywords preserves the existing method route and leaves
`result.engine_provenance` as `None`. Supplying an engine opts into the bounded
experimental molecular adapter. Its default guess, when no SCF options object
is passed, is HCORE. A supplied options object must explicitly select HCORE or
READ with a density already expressed in the same basis. AUTO, other guesses,
restart-file projection and fragment guesses are not yet admitted. Options are
copied before basis-sidecar processing; the caller's object is preserved.

`IntegralEngineOptions.backend` is `"manta"` or `"libint"`. The latter selects
the bounded unscreened libint reference, currently exact J/K only. The default
of the options class is libint. `fallback="error"` is the only currently
qualified policy on these method adapters; unsupported inputs fail before SCF.
Selecting libint explicitly never executes MANTA primitive kernels. Shared
libint basis storage and normalization remain in both routes.

J and K are selected independently, with no inferred change of exchange:

```python
coulomb = vq.CoulombBuildOptions(
    algorithm="ri", auxiliary_basis="def2-svp-jk"
)
exchange = vq.ExchangeBuildOptions(algorithm="exact")
result = vq.run_rhf(
    molecule, basis, integral_engine=engine, coulomb=coulomb, exchange=exchange
)
```

All eight MANTA combinations of exact/RI/BUPO/RI-BUPO J and exact/RI K are available.
`CoulombBuildOptions.auxiliary_basis` supplies the common auxiliary basis when
either selected algorithm uses RI, including exact J with RI K. It may be a
basis name or `BasisSet`. The `metric` field is an `RIOptions` object specifying
the shared rank policy and mode block. Coverage is checked against the orbital
basis before fitting. An auxiliary basis without an RI target is refused.
The two algorithms and their distinct source identities appear in provenance.

For BUPO, pass `CoulombBuildOptions(algorithm="bupo", bupo=options)` with
the `BupoOptions` described above and `IntegralEngineOptions(backend="manta")`.
`options.near_backend="libint"` explicitly selects libint near-field tiles
while retaining MANTA moments and exact MANTA terminal confirmation. Small
systems can remain entirely near-field under the default leaf capacity.
With RI exchange, also supply the common auxiliary basis. BUPO supplies J
only, including for hybrids and range-separated KS functionals.

For RI-BUPO, use `CoulombBuildOptions(algorithm="ri_bupo",
auxiliary_basis=auxiliary, ri_bupo=options)` with `BupoFittingOptions` and
`IntegralEngineOptions(backend="manta")`. The metric field uses the same
`RIOptions` rank policy as exact RI. Both coupling passes use MANTA MD near
tiles and the same approximate far map. Exact or RI exchange is still chosen
independently, including the separate range-exchange metric for hybrids.

This adapter currently admits all-electron molecular RHF/UHF/RKS/UKS. ECP, COSX,
DFT+U, spin-lock/atomic-spin/multi-guess schedules and conventional AO4 storage
are refused. Legacy `density_fit`/`aux_basis` fields conflict with explicit
selection; use the new algorithm options. `scf_mode=AUTO` or `DIRECT` uses the
selected bounded producer, with no hidden AO4 construction. Builds are full,
unscreened and stateless; the old incremental and Schwarz tuning fields do not
enable those unimplemented optimizations. The native SCF iteration and solver
contracts remain shared with the ordinary drivers.

At the returned density the adapter rebuilds J/K and independently recomputes
energy, physical Fock matrices and commutator norms. `full_build_confirmed`
requires energy and maximum Fock differences no greater than `1e-9`;
`stationarity_confirmed` also requires the requested commutator tolerance.
For BUPO, this check replaces J with an independently built exact MANTA J
at the same density, retaining the selected exact/RI exchange and XC source.
For RI-BUPO, the check instead rebuilds exact RI-J with the same auxiliary
basis, retained positive metric and cutoff, keeping exchange and XC unchanged.
The returned approximate energy and Fock are preserved. A failed check
does not trigger hidden polishing or an algorithm change; tighten BUPO
controls and rerun. `confirmation_coulomb_algorithm` and its source identity
identify this separate check. RI J retains its selected RI confirmation.
RI-BUPO confirmation does not impose agreement with four-center J, since
that would include the separate fitting-basis error.
A failed terminal stationarity check clears `result.converged`; a zero-iteration
run never becomes converged through confirmation. The immutable provenance
also records one-electron and J/K sources, per-spin density identities, actual
work counts, component timings and modeled integral memory. J/K shell calls
include RI metric setup and BUPO moment panels; J/K build calls count density contractions.
BUPO provenance additionally reports partition identity, near provider, order,
setup time, accumulated near/far/moment work and the last analytic J bound.
Schema version 4 adds RI-BUPO forward/adjoint source identities, exact metric
factor/rank, last pass bounds, energy bound, residual and inverse-norm bound.
For RI-BUPO, the integral tolerance and maximum integral error fields refer
to three-center coupling entries, and near/far/moment counters include both
passes. `bupo_moment_primitive_products` covers pair and auxiliary expansions;
the older `bupo_moment_primitive_pairs` remains specific to four-center BUPO.
Exact RI metric setup and terminal confirmation are included in actual counts.
Libint primitive
counts are explicitly marked unmeasured. No solver speedup is inferred.

`IntegralEngineOptions.budget` is a `ResourceBudget` applied to each integral
build. Combined admission includes simultaneously retained direct/RI/BUPO/RI-BUPO providers
and Hamiltonian/confirmation buffers. Solver, DIIS, XC quadrature and XC
workspace allocations retain the existing SCF resource policy; this is not a whole-process memory guarantee.
Charge other retained caller data through `other_live_bytes`.
`manta.molecular_engine_provenance(result)` returns a JSON-compatible copy,
including citation routes. The same executed receipt is emitted through
`vibeqc.output` when an output channel or structured log is active.

QVF file and in-memory exports automatically preserve that executed receipt
when the supplied result has `engine_provenance`. The noncritical vendor
section `x_vibeqc.molecular_integral_engine` stores the complete JSON receipt
at `provenance/molecular_integral_engine.json`, with the normal member hash.
Failed confirmation stays failed in the exported record. Requested options
alone never create a provider claim, and legacy results add no engine section.
When an executed receipt is present, its section identity and member path are
reserved against caller-supplied vendor overrides. Other extension declarations
and vendor sections are preserved. This changes only vibe-qc's QVF producer;
it does not require changes to the QVF specification or viewer.

For a confirmed stationary MANTA RHF/UHF result, ordinary `compute_gradient`
and `compute_gradient_uhf` use the MANTA HF derivative consumers with the same
independently selected exact/RI J and K. The native result owns an immutable
copy of the auxiliary basis, RI options and original resource budget. A
custom auxiliary basis is preserved by content, including when its display
name matches a library basis. Changing caller options after SCF cannot alter
the derivative source. Both Python and native entry points verify the molecule,
orbital basis, terminal spin densities and reconstructed RI source identities;
they reject conflicting derivative options and unresolved rank-cutoff crossings.

The gradient uses the original `IntegralEngineOptions.budget` for its aggregate
derivative admission. An SCF-admissible budget may be too small for derivatives;
select a sufficient budget before SCF, or use the component interface with an
explicit budget. The SCF admission reserves storage for the retained context.
Successful Python calls emit `molecular_hf_gradient` through `vibeqc.output`,
including the confirmed source identities and derivative citation route. The
serialized SCF receipt is evidence; it does not serialize this native fitting
context or independently authorize reconstruction of an arbitrary source.

BUPO and RI-BUPO gradients remain gated until their approximate energies have
qualified derivatives, even when exact terminal confirmation passes. Selected
KS gradients remain a separate consumer qualification.

### Kohn-Sham selection and range exchange

For RKS/UKS, pass the same engine/J/K keywords with an `RKSOptions` or
`UKSOptions` specifying the functional. LDA, GGA, meta-GGA, global hybrids,
single-range Coulomb/erf hybrids and their VV10 nonlocal terms use the existing
native XC evaluation. XC is provided by libxc and vibe-qc's grid integration;
MANTA supplies the selected one-electron and J/K integrals. The selected route
refuses full-grid external providers, double-hybrid post-SCF corrections,
Yukawa kernels and mixed exchange ranges that the native CAM assembly cannot
represent. These refusals happen before SCF; no functional is replaced.

```python
opts = vq.RKSOptions()
opts.initial_guess = vq.InitialGuess.HCORE
opts.functional = "hse06"
ks = vq.run_rks(molecule, basis, opts, integral_engine=engine,
                coulomb=coulomb, exchange=exchange)
```

With no options object, the selected route applies `grid_level` just like the
ordinary wrapper, while using HCORE. A supplied options object keeps its grid.
VV10 uses the same full or separately coarsened grid as the ordinary driver.
The final-density evaluation recomputes XC as well as J/K on those exact grids
and compares the individual Coulomb, exchange and XC energies in addition to
the total and Fock matrices. The result retains the native SCF state; this
check does not repair a mismatching state.

Exchange is assembled as `a*K_coulomb + b*K_erf(omega)`, with the resolved
functional coefficients. Both signs of `b` are admitted, including screened
hybrids such as HSE06. When exchange uses RI, the erf source owns its own
kernel-specific metric, retained rank and factor. It never reuses the
full-Coulomb fitting factor or assumes an RI full/erf/erfc decomposition
identity. Admission includes both retained providers. The bounded libint
reference currently refuses range-separated exchange.

Provenance schema 2 adds the method, provider version, channel batch,
functional, XC source/grid identity (including VV10 parameters), exchange
coefficients and range,
range-source identity, and actual range build/shell counts. Range counts are
included in total J/K counts. `exchange_active=False` identifies a pure
functional; configured exchange metadata does not imply that K was evaluated.
Quadrature identity hashes actual points and weights along with the resolved
functional components, spin and exact-exchange coefficients. The XC provider
is `libxc+vibeqc-vv10` when vibe-qc also evaluates nonlocal VV10 correlation.

Selected KS gradients currently refuse explicitly until the matching integral,
XC and moving-grid derivative composition is qualified. Converting an RKS
result to the native RHF result carrier preserves this provenance and cannot
make it an admissible HF gradient source.

## Complete molecular jobs

`run_job` accepts the same `integral_engine`, `coulomb` and `exchange` keywords
for gas-phase RHF/UHF/RKS/UKS single points:

```python
result = vq.run_job(
    molecule, method="rhf", basis="def2-svp", output="manta-rhf",
    integral_engine=vq.IntegralEngineOptions(backend="manta"),
    coulomb=vq.CoulombBuildOptions("ri_bupo", "def2-svp-jk"),
    exchange=vq.ExchangeBuildOptions("exact"),
)
```

The job preserves the selected source through SCF, independent terminal
confirmation, static properties, standard output and QVF export. Structured
logs retain the separate SCF and property receipts. Citations follow executed
providers, including an explicitly selected libint BUPO near field; unscreened
selected builds do not claim the legacy Schwarz-screening route. A qualified
RKS convergence retry keeps the same J/K and range/XC configuration and uses
the returned same-basis density as READ input. Automatic RHF SAD retries are
not enabled until the matching atomic source is qualified. Failed convergence
retains the ordinary job runner's error contract.

Omitted SCF options use HCORE, while supplied HCORE/READ options are privately
copied. `run_job(initial_guess="hcore")` can explicitly select that guess.
The existing `method="auto"` policy still chooses the physical method; if it
chooses a correlated method, the selected engine route refuses instead of
changing that choice to HF. Set the intended mean-field method explicitly.
The ordinary output-discovery dry run performs no selected integral builds;
native engine admission occurs on execution. The engine's integral-build
budget and the runner's general memory estimate have distinct scopes.

Selected jobs retain the ordinary default closed-shell IBO, Boys and
Pipek-Mezey localized wavefunctions. `iao_analysis=True` also supports
restricted and unrestricted determinant charges, spin populations and
IAO-Wiberg bond indices. Both paths use the executed provider for the main
overlap, minimal-reference overlap, cross-basis overlap and Cartesian dipoles.
They retain the existing element, ECP, metric-rank and determinant admission
rules. Explicit `localize=False` still suppresses localized output.

For standalone use, `iao.iao_reference(..., source_result=result)` and
`iao.analyse_localization(..., source_result=result)` check the molecular source
and follow its provider when computing new integrals. Omission retains libint.
Explicit precomputed dipoles, reference panels and IAOs remain caller-provided
numerical inputs. `iao_population.analyse_iao` uses its result's provider
automatically. Separate property receipts identify both actual integral bases;
the SCF evidence is unchanged. The retained minimal overlap is charged while
its cross-basis panel is built; Python localization algebra has its own lifetime
outside the one-electron integral-build budget.

Hirshfeld atomic promolecules, QTAIM, response/TDDFT, Hessians,
atomization, dispersion, solvation, DFT+U, finite-temperature
smearing, READ projection and FRAGMO remain separate qualification stages.
Use the selected J/K options rather than legacy `density_fit`, `aux_basis`
or COSX flags. Unsupported explicit compositions refuse; default population
reports label the unqualified Hirshfeld section unavailable. Periodic/CCM
MANTA method adapters remain separate integration stages.

### Selected HF geometry optimization

`run_job` supports the native `geom_opt` route for selected all-electron RHF/UHF
with independently chosen exact/RI J and K. Setting `geom_opt` activates
optimization for these selected jobs. The provider snapshots orbital and
auxiliary shell content, atom ownership, SCF options, fitting policy and
resource budget. Every geometry rebuilds the shell centers from their owning
atoms; the final single point uses the same auxiliary content at the optimized
coordinates. Display names never replace caller-supplied auxiliary data.

```python
result = vq.run_job(
    molecule, basis="sto-3g", method="rhf", geom_opt="bfgs",
    integral_engine=vq.IntegralEngineOptions(backend="manta"),
    coulomb=vq.CoulombBuildOptions("ri", "sto-3g"),
    exchange=vq.ExchangeBuildOptions("ri"),
    output="selected_hf_geometry",
)
```

For a direct optimizer call, construct
`manta.MolecularHFGeometrySource(molecule, basis, method="rhf", options=None,
integral_engine=None, coulomb=None, exchange=None)` and pass the callable to
`vibeqc.geomopt.run_geomopt`. Its default provider is MANTA; an explicit libint
reference supports exact J/K. `basis_at(molecule)` and `scf_keywords(molecule)`
return owned inputs for a final same-source calculation. `last_result` and
`last_basis` describe the last successful energy/gradient evaluation, which may
be a line-search trial rather than the final optimizer geometry.

Each geometry currently starts from HCORE. Supplied options must select HCORE;
projected READ restarts remain unqualified. Atom order, elements, charge and
multiplicity stay fixed. SCF nonconvergence, incompatible ECP/derivative
sources, insufficient derivative budgets and unresolved metric rank-cutoff
boundaries refuse the evaluation. The SCF and gradient receipts are emitted at
each successful evaluation; complete jobs also cite the executed derivative
route. Python source snapshots and optimizer storage are outside the native
integral-build budget. KS/BUPO gradients and the legacy `optimize=True` route
without `geom_opt` remain gated for selected engines.

## Molecular HF Lagrangian gradient

`compute_hf_lagrangian_gradient(molecule, basis, density_alpha, density_beta,
energy_weighted_density, budget)` assembles all terms of the all-electron,
full-Coulomb molecular HF Lagrangian derivative. It returns owned `(n_atoms, 3)`
matrices `values`, `nuclear`, `one_electron`, `two_electron` and `pulay`, in
Hartree/bohr. `values` is their sum; physical forces have the opposite sign.
The orbital angular limit is `l <= 6`, pure or Cartesian. Ghost centers are
allowed and their moving basis functions contribute even though their nuclear
charge is zero. Shell positions and atom ownership must match the supplied
molecule exactly; coincident charged nuclei are refused.

All three density arguments must be finite real symmetric float64 ndarrays,
C- or F-contiguous, with the basis dimensions. `density_alpha` and
`density_beta` are per-spin densities. The total energy-weighted density is
`W = sum_spin C_occ diag(epsilon_occ) C_occ.T`, with occupied orbitals and
energies from the same physical HF Fock matrices. For RHF use `D/2` for both
spin densities and `W = 2 C_occ diag(epsilon_occ) C_occ.T`.

The returned expression is

```text
D = D_alpha + D_beta
Gamma[ij,kl] = (D[ij]*D[kl] - D_alpha[ik]*D_alpha[jl]
                            - D_beta[ik]*D_beta[jl]) / 2
dL = dE_nuc + D:d(T+V) + Gamma:dERI - W:dS
```

It follows the HF derivative formulation of
[Pople, Krishnan, Schlegel and Binkley (1979)](https://doi.org/10.1002/qua.560160825).
The caller owns stationarity: arbitrary symmetric matrices are valid inputs
for a fixed-density Lagrangian derivative, but are not evidence of a converged
HF force. A stationary RHF/UHF state of the same all-electron exact Hamiltonian
supplies an energy gradient, including basis-motion/Pulay terms. This interface
does not accept or infer ECP, RI, XC, finite-temperature, constrained-state,
range-separated, BUPO, periodic or CCM contributions. Passing densities from
such a Hamiltonian would differentiate only the explicitly defined expression.

`plan_hf_lagrangian_gradient(molecule, basis, budget)` admits the aggregate
one-electron, nuclear and all twelve quartet-center components before matrix
payload copies or numerical contraction. It sums shell, primitive and work
costs, and includes live inputs, component accumulators, exports and the
largest provider workspace. Two-electron derivatives stream one shell
component at a time; no four-index AO tensor or atom-by-AO-matrix derivative
tensor is retained. The initial schedule is unreduced and unscreened.
Receipts carry the actual shell and primitive evaluation counts, source and
ordered `(D_alpha,D_beta,W)` identities, and modeled memory. The
`manta_hf_gradient` citation route is available to output consumers.

## Streamed shell-pair moments

`compute_pair_moments(basis, shells, origin, order, budget)` returns one
shell-pair panel through total degree 12, for pure or Cartesian orbital
shells with l through 6. Each row identifies one AO pair, with the second
shell's AO index contiguous. Columns follow `indices`, the raw Cartesian
moment powers used by the multipole algebra. `ao_shape` records the two AO
dimensions. There are no density weights or off-diagonal pair multiplicities
in the panel.

The primitive Hermite expansion is formed once per primitive pair and reused
for every moment column. Signed contractions use compensated accumulation,
then the existing pure/Cartesian transforms. Receipts distinguish shared
`primitive_expansions` from `primitive_moment_products`. `plan_pair_moments`
admits one panel, transformation and recurrence workspace, and its work before
numerical evaluation. `shell_calls` needs one call for a panel; admission does
not require a whole-basis quadratic or quartic schedule. The output array owns
its storage. Other retained panels belong in `other_live_bytes`.

A consumer can contract each panel into a leaf moment vector and regenerate
the panel during the reverse pass; it need not retain an AO-pair-by-moment
tensor for the whole basis. Translation from origin A to B uses
`panel_at_A @ multipole_translation(A-B, order).values.T`. The source identity
includes the basis, ordered shell pair, origin and order. The citation route
is `manta_md`.

## Auxiliary Gaussian moments and extents

`compute_auxiliary_moments(basis, shell, origin, order, budget)` returns
the raw moments `integral (r-origin)^n chi_P(r) dr` for every function in
one auxiliary shell. It admits pure or Cartesian l through 7 and total
moment degree through 12. The owned matrix has auxiliary-function rows and
the same `indices` columns as pair moments. It integrates a single function;
an orbital self product has different exponents, coefficients and moments.
One primitive expansion is shared across all moment columns. The receipt
reports `primitives`, `primitive_expansions` and `primitive_moment_products`.
`plan_auxiliary_moments` admits a single panel and its work independently
of any fitting or whole-basis Coulomb schedule. Its citation route is `manta_md`.

`compute_auxiliary_extents(basis, options, budget)` produces one
`AuxiliaryExtent` per shell in `result.shells`. Centers are the physical
shell origins, and each record identifies its `shell` index. The same
`ExtentOptions` select pointwise/tail-mass tolerances, radius limits, moment
order and optional full radial bounds. The absolute envelope is the sum of
`abs(c_p) * r**l * exp(-alpha_p*r*r)`, multiplied by the largest absolute
row sum of the pure transform when applicable. Thus signed contractions
and all components of a shell are covered. Full and tail radial moments,
L1/L-infinity norms and potential bounds use the same positive, logarithmic
radial evaluation as shell-pair extents. A radius/search cap labels the
shell `always_near`; it never silently discards the function.

`plan_auxiliary_extents` admits linear shell storage and primitive work.
In both auxiliary APIs, `shell_calls` caps shell evaluations and the shared
budget field `primitive_quartets` caps the single primitive count. Returned
arrays and extent inventories are owned copies, with source identities
covering basis content and numerical controls. Other retained data belongs
in `other_live_bytes`. The envelope citation route is `manta_bupo_extents`.
These are RI-BUPO building blocks used by `BupoFittingCoupling`; selecting a
fitting solve or molecular SCF algorithm requires its own consumer.

## Bubble hierarchy and streamed partition

`BubbleTree(extents, options, budget)` copies a complete canonical shell-pair
extent inventory into an immutable hierarchy. `BubbleOptions` controls
`leaf_capacity` (default 150), `branching` (2 through 8, default 3), and
`maximum_depth` (0 through 64, default 32). Each node splits the longest
coordinate span into balanced groups, with stable pair IDs breaking ties.
Centers are bounding-box midpoints. Parent spheres enclose child spheres;
reaching the depth cap creates a larger leaf without dropping any pair.
This deterministic spatial partition is a MANTA implementation choice.

`tree.traversal(opening=0.5, budget=budget)` streams an unordered partition
of all shell-pair interactions. Distinct node pairs are far candidates when
their summed radii divided by center separation are below `opening`, which
must lie in `[0,1)`. Zero forces an entirely near partition. An always-near
extent prevents its containing node from becoming a far candidate; descent
can still isolate eligible descendants. Near blocks contain terminal leaves.
Each interaction occurs once, including self interactions, so a consumer can
apply paired forward and transpose contributions without double counting.

Iteration accepts these geometry-only candidates. A numerical consumer must
instead use `peek()`, then `accept()` after its own source-specific error
admission, or `refine()` to descend a rejected candidate. Rejecting a terminal
far candidate converts it to an exact near block. Geometry alone does not
bound Gaussian penetration or the use of full Gaussian moments. The BUPO
consumer adds the numerical error admission before using this traversal.

`plan_bubbles` admits linear tree storage and bounded-depth sorting work.
Traversal uses a depth-bounded stack, with no retained quadratic interaction
list. Its budget caps actual node visits (`shell_calls`), modeled visit work
(`kernel_terms`), and represented shell-pair interactions (`scalar_terms`).
A budget failure invalidates the traversal; it cannot resume or report a
complete result. The receipt distinguishes accepted near/far blocks,
refinements, actual visits, and peak stack size. `source_identity` identifies
the tree and opening policy; `partition_identity` is populated only at
completion and also hashes the accepted blocks and their near/far choices.
Exported nodes, pairs, order and receipts are owned snapshots. Charge retained
diagnostic copies through `other_live_bytes` when starting concurrent work.
The low-level citation route is `manta_bubbles`.

## Gaussian far-field error ledger

`BubbleErrorModel(tree, order, budget)` requires retained full and tail
absolute moments through `order`. It translates those radial envelopes to
node centers with the positive binomial theorem, then takes maxima over
descendants. Each node envelope bounds every individual AO product in that
node, rather than a sum over its pairs. Parent-sphere enclosure ensures that
the translated child-tail envelope also bounds the parent tail.

`model.interaction(target, source)` returns the matching Cartesian Coulomb
matrix and its approximation bound; `model.bound` returns just the bound.
Always-near nodes, overlapping pair inventories and overlapping support
spheres are refused. An extreme separation that makes the highest inverse
radial power unrepresentable is also refused for direct evaluation or further
refinement. The source identity records the tree, order and error policy.

The uniform bound on each AO four-center integral has three parts:

- `compact_remainder`: the finite-support multipole remainder using the two
  node radii and full absolute charge norms.
- `tail_penetration`: `tail_mass_A * potential_B + tail_mass_B * potential_A`,
  bounding the exact Coulomb contribution involving the excluded tails.
- `full_moment_tail`: the sum over matrix entries of
  `abs(T[a,b]) * (tail_A[degree(a)] * full_B[degree(b)] + full_A[degree(a)] * tail_B[degree(b)])`.
  This bounds replacing compact-support moments with full Gaussian moments.

`total` includes all three terms. This is a MANTA extension of the BUPO
near/far separation, with an explicit Gaussian approximation ledger.
Log-space positive accumulation and outward binary64 headroom are used;
floating-point kernel and contraction errors remain separate. It is not an
interval-certified bound on a computed J matrix. A consumer must include
density weights and numerical accumulation error, and refine a candidate
whose bound exceeds its allocated tolerance. A fixed uniform per-integral
approximation bound `epsilon` implies the analytic bound
`max(abs(delta_J)) <= epsilon * sum(abs(D))` for a complete AO density matrix.

`plan_bubble_errors` admits retained envelopes, setup work and one interaction
query, including the multipole workspace. Repeated queries are separate
operations; a J consumer must admit aggregate traversal work. Diagnostic
envelope exports are copies. The citation route is `manta_bupo_errors`.

`model.auxiliary_interaction(target_node, auxiliary_extent)` applies the
same three-term ledger to one orbital-pair bubble and a single auxiliary
shell from `compute_auxiliary_extents`. The auxiliary record must retain
full and tail moments through the model's order. Both supports must be
disjoint and neither may be always-near. For pair moments M at the target
node and auxiliary moments A at the shell center, the approximate tile in
pair-by-auxiliary layout is `M @ interaction.matrix.values @ A.T`.
The bound covers every component of every descendant pair against every
function of that auxiliary shell. The transposed tile defines the opposite
coupling pass; density weights and fitting coefficients belong to its consumer.

Auxiliary error identities include the shell label, geometry and actual
envelope values. The model's plan covers its tree envelopes plus one query,
including the bounded auxiliary envelope copy. The caller accounts for
retained auxiliary inventories and repeated queries. This per-integral
ledger does not by itself bound metric conditioning, RI basis error or the
complete fitted J; those remain separate consumer checks.

## Cartesian multipole algebra

`multipole_translation(source_minus_target, order, budget)` and
`multipole_coulomb(target_minus_source, order, budget)` provide bounded
Cartesian matrices through total degree 12. Their `indices` list defines the
three nonnegative Cartesian powers for each row and column. Moments are raw
integrals `M_n(C) = integral (r-C)^n rho(r) dr`, without factorial scaling.
Translation obeys `M(target) = W M(source)`; a downward field uses `W.T`.
The Coulomb energy is `M(target).T @ T @ M(source)`, retaining terms whose
combined target and source degree is at most `order`. This complete-degree
convention is rotationally invariant in exact arithmetic, and
`T(-R) = T(R).T`. It is an explicit Cartesian alternative to the solid
harmonics in [BUPO](https://doi.org/10.1021/acs.jpca.4c07415).

`compact_multipole_remainder(distance, target_radius, source_radius,
target_norm, source_norm, order)` bounds the omitted Coulomb series for two
disjoint, compact supports using their absolute charge norms. With
`theta = (target_radius + source_radius) / distance`, the bound is
`target_norm * source_norm * theta**(order+1) / (distance - target_radius - source_radius)`.
Gaussian tail penetration and replacing truncated moments with full Gaussian
moments require separate bounds. This routine does not certify a Gaussian
pair interaction by itself. Floating-point headroom and upward rounding are
included; the implementation is not interval arithmetic.

`plan_multipoles(order, budget)` admits dense matrix storage and polynomial
work before allocation. Exported matrices own their storage. Source identities
include the operation, displacement, order and normalization. These building
blocks do not by themselves select a Coulomb algorithm. The low-level citation
route is `manta_multipoles`.

## Finite periodic lattice multipoles

`multipole_lattice_tail(cell, target_minus_source, eta, real_indices,
reciprocal_indices, near_indices, order, budget=ResourceBudget())` constructs
raw Cartesian interaction coefficients through total degree 12 for the
3D zero-average conducting Ewald Green function minus exactly the listed
near-cell bare Coulomb interactions. Cell vectors are **columns**, in bohr.
The displacement is target center minus source center. Each integer real
index `n` represents `cell @ n`; a reciprocal index `g` represents
`2*pi*inverse(cell).T @ g`.

The cell and displacement are contiguous float64 arrays of shapes `(3,3)`
and `(3,)`; domains are contiguous int64 arrays of shape `(count,3)`.
C and Fortran storage are accepted. Each domain must be unique and inversion
closed. Real and near domains include zero, near is a subset of real, and
the reciprocal domain excludes zero. Indices are limited to the symmetric
signed 31-bit range. An empty reciprocal domain is an explicitly incomplete
finite source. The routine accepts a finite positive Ewald parameter `eta`
and a finite nonsingular cell. It refuses a supplied non-near point singularity
at the expansion center or nonfinite intermediates/results.

The returned `LatticeTailMatrix` has the same raw moments, factorials and
total-degree contraction as `multipole_coulomb`. Near terms use
`-erf(eta*r)/r`, including its removable limit at zero. Other real terms use
erfc radial derivatives, evaluated through the positive upper incomplete
gamma recurrence. Reciprocal cosine derivatives and the constant
`-pi/(eta**2*volume)` are included once. Consequently the converged local
tail has Laplacian `4*pi/volume`. Its quadratic part needs the trace second
moments as well as dipoles and monopoles. Neither traceless harmonic moments
alone nor a second background correction gives the declared source.

`plan_lattice_tail(order, real_cells, reciprocal_vectors, near_cells, budget)`
admits input snapshots, canonical domain sorting, polynomial work, recurrence
workspace, result storage and one exported copy before any array payload is
read. It includes the backend margin and caller's `other_live_bytes`. The
memory/work counts are conservative models, not measured RSS or FLOPs.
Earlier results and additional exports retained by the caller belong in
`other_live_bytes`. Actual evaluated radial, reciprocal and matrix coefficient
counts are separate fields. Results are immutable; array exports own copies.

`kernel_identity` binds the cell, eta, sorted domains, boundary, algorithm
and normalization; `source_identity` adds expansion displacement and order.
Each domain also has its own identity. Reordering a domain does not alter
its source. The qualification is explicitly `finite-domain-coefficients`:
real/reciprocal convergence, Gaussian penetration and
pair-image truncation remain separate controls. The molecular compact
Coulomb remainder is not a bound for this periodic operator. Point-nuclear
self removal and the complete charge/background energy ledger belong to
the future consumer; the lattice operator also supports separately charged
components in the declared gauge.

This building block does not select periodic BUPO-J/RI-BUPO-J, bypass source
or symmetry admission, or change FockBuild and BIPOLE dispatch. It is the
bounded prerequisite for their future disjoint near/tail traversal. The
citation route is `manta_lattice_tail`; the derivation and boundary convention
are in [the scientific contract](../design/manta_scientific_contract.md).
Independent coefficient, Gaussian energy and resource measurements are in
the repository's `benchmarks/manta_lattice_tail.py` harness and accompanying
`benchmarks/manta_lattice_tail.md` report.

`lattice_tail_remainder` takes the same cell, displacement, eta, domains and
order, followed by `target_radius`, `source_radius`, `target_norm` and
`source_norm`. Radii must enclose the compact supports about the expansion
centers; norms must bound the integrals of **absolute** charge. Signed or
neutral distributions still use these nonnegative norms. The caller is
responsible for their validity. Values must be finite and nonnegative.

The immutable `LatticeTailRemainder` reports `near_remainder`,
`real_remainder`, `reciprocal_remainder` and their conservative `total` in
pair-energy units. Its `expansion_identity` must match the coefficient
matrix's `source_identity`. Changing radii or norms changes the remainder's
own `source_identity`, while preserving that expansion binding. Canonical
domain ordering and geometric conventions are shared with the coefficients.
Every non-near supplied image must be farther from the expansion center than
the sum of the radii, with a numerical separation guard.
`minimum_non_near_distance` reports this remaining distance to a supplied
non-near singularity after subtracting both radii and the guard. All-near finite real
sets have `minimum_non_near_distance=inf`; this says nothing about omitted
images. Zero norms or zero combined radius yield zero remainder after domain
and support validation. Positive tiny contributions never silently become zero.

`plan_lattice_tail_remainder(order, real_cells, reciprocal_vectors,
near_cells, budget)` reserves count-only sorting, snapshots, bounded positive
series work and result exports before payload reads. It does not allocate a
multipole matrix. Its peak includes caller live bytes and the backend margin;
retained coefficient matrices belong in `other_live_bytes`. Actual radial
integral and reciprocal-term counts are reported separately.

The qualification is `finite-domain-compact-taylor`, with
`roundoff_included=False`. The derivation uses directional Gaussian derivatives
and positive integral envelopes, including a conservative floating-point
evaluation guard. It is not an interval certificate, a Gaussian-tail estimate
or an infinite-lattice error bound. Order selection based on `total` controls
only this compact finite-source Taylor contribution. Complete approximation
control still needs the other error terms. See the derivation in the
[scientific contract](../design/manta_scientific_contract.md) and the measured
`benchmarks/manta_lattice_remainder.py` and `.md` files.

For full Gaussian moments, use the joint finite-source ledger:

```python
options = manta.ExtentOptions()
options.moment_order = 8
options.retain_full_moments = True
pair = manta.compute_pair_extents(orbital_basis, options).pairs[0]
aux = manta.compute_auxiliary_extents(auxiliary_basis, options).shells[0]
interaction = manta.lattice_gaussian_interaction(
    cell, eta, real_indices, reciprocal_indices, near_indices,
    target=pair, source=aux, order=8, budget=budget,
)
error = interaction.error
print(error.compact_remainder, error.tail_penetration,
      error.full_moment_tail, error.total)
```

Either source can be a native orbital-pair or auxiliary envelope. The native
producer must retain full moments through the requested order, and its
`always_near` flag must be false. The expansion centers come from these
envelopes. Every supplied non-near image must remain outside their combined
radii. An inadmissible support needs direct evaluation or a different near
partition. A broad envelope can increase the bound at higher order; inspect
the actual total rather than assuming monotonic improvement.

`LatticeGaussianInteraction` owns the matrix, compact remainder, joint plan
and three-term error ledger. Contract moments computed about each envelope's
own center with `interaction.matrix.values`. `error.total` bounds each
AO-component integral of this **finite** Ewald-minus-near source, before
density or fitting contractions. Its penetration term uses global Coulomb
potential envelopes for non-near erfc images and retains bounded near,
reciprocal and constant-gauge terms. Full-moment substitution uses every
Cartesian coefficient, including trace entries. A neutral signed component
still has nonzero absolute envelopes.

`plan_lattice_gaussian_interaction(order, real_cells, reciprocal_vectors,
near_cells, budget)` admits all joint memory and work before array reads or
copies. It covers coefficient and remainder work, reciprocal sorting, the
two envelopes and result/export storage. Other retained basis, moment or
Fock storage belongs in `budget.other_live_bytes`. Cell and integer domains
have the same strict float64/int64 C/F-contiguous contract as the coefficient
API. Outputs are immutable, and exported matrix arrays are owned copies.

`error.source_identity` binds both envelopes and the matching coefficient and
remainder identities. Envelope identities describe consumed numerical bounds,
roles, shell indices and centers; they do not certify a basis or Fock source.
The qualification is `finite-domain-gaussian-approximation`, with
`roundoff_included=False`. Omitted lattice images, pair-image truncation,
numerical roundoff, density aggregation and point-nuclear self corrections
remain separate. This primitive does not yet select a production periodic
BUPO-J or RI-BUPO-J method. The [scientific contract](../design/manta_scientific_contract.md)
derives the three terms; `benchmarks/manta_lattice_gaussian.py` measures their
joint construction and independent Gaussian energy comparisons.
The [measured benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_lattice_gaussian.md)
also records complete local calculation evidence and its qualification limits.
The citation database exposes the `manta_lattice_gaussian` route for reports
using this diagnostic primitive.

## Infinite Ewald domain bounds

`lattice_domain_remainder` bounds the omitted images of a finite Ewald
source for two compact distributions. Supply the same cell, displacement,
eta and real/reciprocal/near index arrays as `multipole_lattice_tail`, plus
enclosing radii and integrals of absolute charge:

```python
error = manta.lattice_domain_remainder(
    cell, displacement, eta, real_indices, reciprocal_indices, near_indices,
    target_radius, source_radius, target_norm, source_norm, budget,
)
domain_error = error.total
```

The real domain must be a complete centered integer cube; the reciprocal
domain must be another such cube with zero excluded. Their radii can differ.
The near set is any valid inversion-closed, origin-containing subset of the
real domain. A shifted or incomplete cube is refused. An empty reciprocal
domain represents radius zero and bounds every nonzero reciprocal vector.
The omitted real images must be separated from the entire compact support;
otherwise enlarge the real cube. Strongly skewed cells can require large
cubes because the metric bound is conservative.

`error.real_remainder` and `error.reciprocal_remainder` remain separate.
`error.kernel_identity` matches the finite lattice-tail kernel, independently
of Taylor order. The near subtraction and gauge cancel from this domain
error. A zero net charge is not a zero absolute norm. The source identity
also binds displacement, radii, norms and the bound algorithm. Metadata are
immutable; the constructor performs no enumeration of the infinite domain.
`plan_lattice_domain_remainder` accepts real, reciprocal and near counts and
admits memory, scalar and kernel work before array payload reads or copies.

The domain bound can supplement a compact Taylor bound. It also controls
the defect in finite real-space periodic covariance; it does not declare
a truncated Ewald sum exactly periodic. Its conditional compact-source
qualification excludes Gaussian tails, omitted AO pair images, roundoff and
complete SCF accuracy. A finite pair-image source does not become a compact
distribution just because it owns Gaussian extent radii. See the
[scientific derivation](../design/manta_scientific_contract.md) and
[measured evidence](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_lattice_domain.md).
The citation route is `manta_lattice_domains`.

## Periodic pair-density moments

`PeriodicPairSource` supplies the native pair-image source for q=0 Coulomb
aggregation. It accepts complex Hermitian densities on a full momentum list,
including non-TRIM points and unequal or zero weights. No relation between
the electronic states at k and -k is required.

```python
options = manta.PeriodicPairOptions()
options.order = 8
options.pair_cutoff_bohr = 8.0  # or None for every listed image
source = manta.PeriodicPairSource(
    basis, cell, image_indices, momenta, weights, options, budget,
)
forward = source.contract(densities)  # complex128 (k, AO, AO)
leaf_moments = forward.moments        # float64 (pair, Cartesian moment)
response = source.adjoint(local_potentials)  # same real shape as leaf_moments
```

The cell has lattice vectors as columns. `image_indices` is int64 `(N,3)`,
unique, inversion closed and containing the origin. Cell, momenta and weights
are float64 arrays; inputs must be C- or F-contiguous. Weights are
nonnegative and sum to one. The constructor owns its numerical snapshots.
`options.extents` selects the native envelope tolerances and radii; this
source always retains full and tail moments through `options.order`.

`source.entries` records each canonical `(bra shell, ket shell, image)`
representative, AO offsets/sizes, actual translation, paired status and native
extent. `source.moment_panel(pair)` produces the corresponding raw Cartesian
panel on demand. `source.extents` retains the complete periodic-source
identity. It is not a molecular triangular inventory and cannot be passed to
the existing molecular bubble-tree constructor. A periodic consumer must keep
the entry/image mapping rather than inventing molecular shell indices.

The forward pass combines image-conjugate density contributions using their
real coefficient sum. The adjoint restores both AO/image orientations with
the original phases. It obeys the weighted trace identity documented in the
[scientific contract](../design/manta_scientific_contract.md). `response.values`
has complex shape `(k,AO,AO)` and is checked for Hermiticity. It is an AO
response to the supplied local potential, not a complete Fock matrix.
All result arrays export owned copies; source/result metadata are read-only.

The forward receipt reports the signed integrated density as `charge`,
separate coefficient L1 norms and an absolute-charge bound. It does not impose
the complete electron/nuclear charge policy. Source, density and potential
identities are distinct. The paired identity applies to periodic potentials;
an arbitrary finite nonperiodic kernel must prove its own covariance.

`plan_periodic_pairs(basis, cells, momenta, options, budget)` admits setup
and one forward/adjoint cycle before numerical array reads or copies. It
reserves retained basis/extent metadata, streamed panels, the two passes and
all result copies. Repeated SCF passes and other live owners need aggregate
admission by the consuming method. The qualification is
`finite-pair-image-moments-and-adjoint`; no Coulomb kernel, exchange source,
production periodic BUPO or SCF accuracy profile is selected automatically.
The citation route is `manta_periodic_pairs`; the
[benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_pairs.md)
records setup/pass cost and independent finite Coulomb comparisons.

## Fourier panels

`compute_fourier_panel(bra, ket, wave_vectors, budget)` returns the integrals
of AO products against `exp(-i*q.r)`. `compute_auxiliary_fourier_panel(auxiliary,
wave_vectors, budget)` transforms individual auxiliary functions. Wave vectors
are Cartesian, in inverse bohr, supplied as a finite C- or F-contiguous
float64 array with shape `(n_vectors, 3)`. There is no cell-volume factor,
Bloch weight, reciprocal-space kernel or periodic background in these panels.

```python
q = np.array([[0., 0., 0.], [0.1, -0.2, 0.3]], dtype=np.float64)
panel = manta.compute_fourier_panel(basis, basis, q, budget)
assert panel.values.shape == (2, basis.nbasis, basis.nbasis)
aux_panel = manta.compute_auxiliary_fourier_panel(basis, q, budget)
assert aux_panel.values.shape == (2, basis.nbasis)
```

Orbital pairs admit pure or Cartesian l through 6; single auxiliary functions
admit l through 7. Each primitive pair's Hermite expansion is formed once for
the entire vector panel. Receipts distinguish `primitive_pairs` from
`vector_primitive_products`; batching shares the expansion, while every vector
still requires its Fourier kernel. Contractions use compensated accumulation.
At q=0 the pair panel equals overlap. Opposite vectors give complex conjugates;
exchanging the basis axes transposes the panel without conjugation at fixed q.

`plan_fourier(bra, ket, n_vectors, budget)` and
`plan_auxiliary_fourier(auxiliary, n_vectors, budget)` admit output, vector
copies, primitive workspace and transform work before scanning or copying
vector values. The usual resource caps apply: `scalar_terms` bounds output
elements and `primitive_quartets` bounds primitive pairs. The caller chooses
a bounded vector batch; no whole reciprocal mesh is constructed implicitly.
An empty `(0, 3)` panel has zero evaluation work. Each exported complex128
array owns its storage. Source identities include both basis contents, the
Fourier convention and the ordered vector values. Other retained panels and
concurrent work must be charged through `other_live_bytes`.

### Finite Bloch Fourier sums

`compute_bloch_fourier_panel(basis, wave_vectors, translations, momenta,
pair_cutoff_bohr=None, budget=ResourceBudget())` evaluates

```text
F[k,mu,nu,q] = sum_R exp(+i*k.R)
              * integral exp(-i*q.r) chi_mu(r) chi_nu(r-R) dr
```

This is the finite translation form of Sun et al. (2017), Eq. 16,
DOI `10.1063/1.4998644`. All three input arrays are finite float64, C- or
F-contiguous `(count,3)` arrays. Translations are Cartesian bohr; wave vectors
and momenta are Cartesian inverse bohr. Supply the complete Fourier argument
`q = G + k_ket - k_bra` and the ket momentum `k = k_ket`. No folding or extra
momentum-transfer shift is applied. Output is complex128 `(n_k,n_AO,n_AO,n_q)`.

The explicit ordered translations are the complete finite source. They are
neither extended, deduplicated nor symmetrized. `pair_cutoff_bohr=None` keeps
all shell pairs; a nonnegative value keeps exactly the pairs satisfying
`norm(origin_bra-origin_ket-R) <= cutoff`. In particular, zero retains only
coincident centers. This is a caller-selected physical support, not amplitude
screening. Empty axes give the corresponding zero sum without integral work.

The native loop streams one translated shell-pair Fourier panel and folds it
into all momenta with compensated accumulation. It does not retain an image
axis of AO tensors. `plan_bloch_fourier(basis, n_q, n_cells, n_k,
pair_cutoff_bohr=None, budget=...)` admits aggregate work, output and
compensation, vector copies, phase storage and conservative MD workspace.
The caller chooses bounded reciprocal and momentum batches. Work estimates
cover every candidate shell pair; receipts distinguish actual calls, physical
support omissions, primitive pairs, vector products and phase accumulation.
The source identity binds basis content, convention, cutoff policy and every
ordered vector, translation and momentum. No performance advantage is claimed.

The existing periodic helpers
`vibeqc._aopair_ft.ao_pair_fourier_transform_bloch` and
`ao_pair_fourier_transform_bloch_multi` accept an explicit
`integral_engine=IntegralEngineOptions(backend="manta")` for these panels and
emit executed receipts through `vibeqc.output`. This selection requires
`fallback="error"`, `channel_batch=1`, zero amplitude screening and no supplied
pair weights or precomputed cell tensors. Omission preserves the existing
reference route. The citation route is `manta_bloch_fourier`.

MANTA retains the libint axial-shell normalization: zero translation and zero
Fourier argument reproduce the physical overlap matrix for pure and Cartesian
shells. The older raw Fourier table uses `r^l Y_lm` above s; comparing its pure
AO axes requires the angular conversion `sqrt(4*pi/(2*l+1))` on each axis for
`l>0`, as performed by the existing RSGDF AO scaling helper. Raw numerical
equality between those two normalization conventions is
not an acceptance criterion. Independent quadrature and overlap comparisons
fix the MANTA convention without relying on that older table.

This primitive does not choose cell volumes, k weights, lattice/reciprocal
cutoffs, the Coulomb kernel, q=0 backgrounds or charge compensation. It does
not establish infinite-lattice convergence or enable a periodic/CCM SCF method
by itself. Those consumers must retain their own source contracts and qualify
consistent AO normalization throughout their composed operators.

### Finite overlap and kinetic Bloch matrices

`compute_bloch_one_electron(basis, operation, translations, momenta,
pair_cutoff_bohr=None, budget=ResourceBudget())` evaluates `operation="overlap"`
or `"kinetic"` on the supplied finite translations:

$$
A_{\mu\nu}(R)=\langle\chi_{\mu,0}|\hat A|\chi_{\nu,R}\rangle,
\qquad A_{\mu\nu}(k)=\sum_R e^{+i k\cdot R}A_{\mu\nu}(R).
$$

`translations` and `momenta` are Cartesian float64 arrays of shape `(n,3)`
in C or Fortran contiguous layout, in bohr and inverse bohr, respectively.
The result exports independent real `cell_values` of shape `(cells,AO,AO)`
and complex `values` of shape `(momenta,AO,AO)`, with read-only `plan` and
`receipt` metadata. An empty momentum array retains the cell blocks; an
empty translation list returns zero Bloch sums. The source identity includes
the basis payload, operation, ordered translations, momenta and physical cutoff.

One translated shell tile supplies both cell storage and every k point.
There is no all-image AO basis or four-index tensor. Folding uses compensated
complex summation. The existing molecular MD overlap/kinetic kernels retain
orbital l through 6, pure/Cartesian conventions and physical libint axial-shell
normalization. The `manta_bloch_one_electron` citation route names MD, SHARK and
the Bloch Gaussian convention of Sun et al. (2017).

The caller supplies the entire finite domain. Translation order and repeated
images are preserved, with no implicit amplitude screening or symmetry
reconstruction. `None` disables the physical pair filter; a finite nonnegative
cutoff keeps exactly `|O_mu-O_nu-R| <= cutoff`, including coincident centers at
zero. A raw maximum Hermiticity residual is reported. Asymmetric finite
supports are returned as supplied, without adding images or repairing matrices.
Consumers requiring a Hermitian source must validate their support and residual.

`plan_bloch_one_electron(basis, operation, cells, momenta, pair_cutoff_bohr,
budget)` bounds all shell pairs, primitive pairs, phase/folding work, real and
complex output owners, input copies, shell workspace and backend margin.
The cutoff can reduce actual receipt counts; admission conservatively includes
all candidate pairs. Caps apply before copying vector payloads. Reported memory
is a workspace model, not measured process RSS. Caller-retained earlier exports
belong in `other_live_bytes`.

For existing cell-resolved consumers:

```python
A_lattice = manta.compute_one_electron_lattice(
    basis, cells, operation="kinetic", pair_cutoff_bohr=pair_cutoff,
    integral_engine=IntegralEngineOptions(backend="manta"),
)
```

This adapter preserves the caller's ordered `LatticeCell` objects and returns
a standard `LatticeMatrixSet`. It admits Python conversion storage before
copying cells and records executed source/work provenance through `vibeqc.output`.
It requires `fallback="error"` and `channel_batch=1`. Existing shared cell
enumerators can provide the physical pair-complete domain for one, two or three
periodic dimensions; this adapter does not generate a replacement domain.
The finite three-dimensional RHF composition is described below. CCM, symmetry
reduction and periodic derivatives require their separate qualification.

### Finite three-dimensional Ewald nuclear attraction

`compute_periodic_nuclear(basis, system, wave_vectors, translations, momenta,
alpha, nuclear_cutoff_bohr, reciprocal_batch=64, pair_cutoff_bohr=None,
budget=ResourceBudget())` builds the all-electron point-nucleus operator

$$
V_R=V_R^{\rm erfc}
-\frac1\Omega\sum_{G\ne0}\frac{4\pi}{G^2}e^{-G^2/(4\alpha^2)}
\left(\sum_A Z_A e^{-iG\cdot R_A}\right)F(G;R)^*
+\frac{\pi\sum_A Z_A}{\alpha^2\Omega}S_R,
\qquad V(k)=\sum_R e^{+ik\cdot R}V_R.
$$

The real-space attraction uses `erfc(alpha*r)/r`. Its nuclear images are
selected by the existing `PairNuclearImageSelector`: a physical sphere of
radius `nuclear_cutoff_bohr` about each AO-pair midpoint, with that selector's
integer-bound and boundary tolerances. This is the nuclear domain used by
pair-complete lattice integrals. An origin-centered image ball is a different
finite source and is not substituted. `alpha` must be finite and positive;
the nuclear radius must be finite and nonnegative. The final overlap term
removes the finite short-range G=0 contribution, producing the zero-average
three-dimensional periodic convention. Other periodic dimensions, ECPs,
exchange singularity corrections and electron-nucleus energy contraction are
not selected by this point-nucleus producer.

The reciprocal and AO translation supports are explicit float64 `(count,3)`
arrays in C or Fortran contiguous layout. Both must be unique and inversion
closed. Vectors must map to reciprocal/direct integer labels within `1e-10`
and labels must be at most `2^40` in magnitude; reciprocal G=0 is refused.
Validation never repairs, wraps, deduplicates or completes a support.
`system.lattice` contains three-dimensional lattice vectors as columns in
bohr. Momenta are Cartesian inverse-bohr values. The nuclear-image selector
also requires its own image labels to fit signed integers.

The physical AO-pair cutoff applies identically to all three terms. `None`
keeps every supplied shell/cell pair and zero retains coincident centers.
The producer applies no overlap-amplitude or reciprocal-damping screen.
The caller must supply the same retained G and cell sets when comparing a
reference that applies such screens. Finite real/reciprocal truncation error
is separate from kernel error; independence from `alpha` requires converged
supports. A converged analytic Gaussian plane-wave series checks this limit.

The result exposes read-only `plan` and `receipt`, with independent arrays:

- `short_range_cells` and `constant_cells`: real `(cells,AO,AO)` blocks.
- `reciprocal_cells` and `cell_values`: complex reciprocal and total blocks.
- `values`: complex `(momenta,AO,AO)` Bloch matrices.

An empty momentum list still builds cell blocks. An empty reciprocal support
retains the screened and constant terms. Raw imaginary cell residuals and
Bloch Hermiticity residuals must be at most `1e-10 * max(1,max(abs(matrix)))`.
They are checked before returning, without real projection or Hermitization.
The source identity records basis, nuclei, lattice, ordered supports, split,
physical cutoffs and convention; computational batch size and budgets do not
change it. The receipt reports actual nuclear images, primitive-charge
products, shell calls and reciprocal work for the executed build.

`plan_periodic_nuclear` takes the same parameters with vector/cell/momentum
counts. It bounds the shared nuclear selector's enclosing index boxes before
allocating charge lists, cumulative radial and reciprocal work, input snapshots,
all exported components, compensated accumulators and one streamed Fourier
panel. A conservative image-box bound can refuse a request whose retained
spherical support would be smaller. The model limits each nuclear fractional
reach to `1e9` and checks overflow; budgets remain practical limits well below
that bound. Memory estimates include backend headroom and are not measured RSS.

`manta.compute_nuclear_lattice(basis, system, cells, wave_vectors, alpha=...,
nuclear_cutoff_bohr=..., integral_engine=IntegralEngineOptions(backend="manta"))`
returns a standard real `LatticeMatrixSet` on the supplied `LatticeCell` order.
Optional `reciprocal_batch` and `pair_cutoff_bohr` retain the native meanings.
The adapter reserves Python conversion storage before copying cell payloads,
checks the native raw residual before converting to real blocks, and emits
`manta_periodic_nuclear` provenance through `vibeqc.output`. Its citation route
includes MD, SHARK, Sun et al. (2017) and the molecular electronic-structure
reference for Coulomb/Ewald conventions. It requires `fallback="error"` and
`channel_batch=1`. This operator does not by itself enable periodic SCF,
CCM, periodic BUPO, symmetry reduction, nuclear gradients or stress.

### Finite periodic Hartree from the whole k mesh

`compute_periodic_hartree(basis, lattice, wave_vectors, translations, momenta,
weights, densities, reciprocal_batch=64, pair_cutoff_bohr=None, budget=...)`
composes the Fourier provider with the three-dimensional zero-average
Coulomb Green function. Lattice vectors are columns in bohr. All vector
arrays use the Bloch Fourier units and strict float64 layout above; the
lattice is `(3,3)`, weights are `(n_k,)`, and densities are contiguous
complex128 `(n_k,n_AO,n_AO)`. The finite source is

```text
rho(G) = sum_k w[k] sum_mu,nu D[k,mu,nu] F[-k,mu,nu,G]
J[R,mu,nu] = (1/V) sum_G (4*pi/G^2) rho(G) conj(F[R,mu,nu,G])
J[k,mu,nu] = sum_R exp(+i*k.R) J[R,mu,nu]
E_J = (1/2) sum_k w[k] Tr(D[k] J[k])
```

This is an implementation of the finite reciprocal Hartree composition in
Sun et al. (2017), Eqs. 16-17, not a periodic extension supplied by the
molecular BUPO paper. `wave_vectors` must contain nonzero reciprocal lattice
vectors. Both this list and `translations` must be unique and inversion
closed; neither is completed or reordered during evaluation. Integer labels
are checked within `1e-10`, with absolute labels at most `2^40`; actual
Cartesian inputs are retained. Empty reciprocal or image supports produce
zero J. `pair_cutoff_bohr` applies the same physical shell-pair support to
density and output Fourier factors.

Weights are nonnegative and sum to one within `1e-12`. Densities may be
indefinite, complex and unrelated by time reversal; their raw Hermiticity
residual must be at most `1e-12*max(1,max_abs_D)`. Every build snapshots and
consumes all k-point densities. This includes the influence of one k point
on the response at other k points. There is no incremental density cache.

The result exposes owned complex128 `cell_values` and `values` arrays and
an immutable receipt with whole-source and whole-density identities,
Hartree energy, actual generation/contraction counts and raw residuals.
Neither array is Hermitized or projected to real values. A real-space
imaginary residual or k-space Hermiticity residual above
`1e-10*max(1,max_abs_response)` causes failure. No executed receipt is emitted
by the Python method adapters on failure.

`plan_periodic_hartree(basis, n_G, n_cells, n_k, reciprocal_batch=64,
pair_cutoff_bohr=None, budget=...)` admits cumulative work across the entire
build before copying inputs. Execution retains one reciprocal batch across
the k mesh, then regenerates one image panel at a time for digestion. Storage
includes dense J and compensation at each output cell and k point, input
snapshots, support validation and conservative leaf workspace. No all-image
AO Fourier tensor or four-index ERI tensor is formed. The modeled peak is
a conservative resource inventory, not a measured process RSS.

The existing `ewald_composed.compute_j_ewald_3d_ft_k_density_to_cells` and
`periodic_fock_multi_k.build_periodic_j_ewald3d_k_from_k_density` accept
`integral_engine=IntegralEngineOptions(backend="manta")`. They preserve the
existing spherical reciprocal cutoff and physical image support. The
cell-block adapter explicitly converts to real storage only after the native
imaginary-residual check. The k-space adapter returns the native complex
result. MANTA factors already have physical AO normalization; the reference
route's angular scaling is not applied again.

These adapters require `fallback="error"`, `channel_batch=1`, and
`cell_chunk_size=None` or `1`. Reciprocal batches must be positive integers.
They admit a conservative enclosing reciprocal index box before generating
the spherical grid, and charge Python grid/input snapshots alongside native
storage. A large enclosing box can therefore be refused even when the
contained sphere would fit. Executed events include the grid cutoff, actual
G count, support policy and citation route `manta_periodic_hartree`.

The Coulomb zero mode is omitted, corresponding to a uniform compensating
background for net charge. No nuclear, exchange, finite-surface or Madelung
correction is added. The accepted `omega` in the existing interfaces has no
effect on this unsplit source. Finite image and reciprocal cutoffs still need
convergence checks. This qualifies the Hartree producer for Gamma and
multi-k densities; production periodic HF/DFT, CCM, periodic BUPO, gradients
and stress remain separately gated.

The reproducible [finite Hartree microbenchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_hartree.md)
reports build/CPU timing, process peak RSS, modeled storage, actual work and
same-source errors for Gamma and multi-k densities. Its small-panel results
do not establish a complete periodic SCF speedup.

### Finite reciprocal exchange across density channels

`compute_periodic_exchange(basis, lattice, wave_vectors, transfer_offsets,
translations, momenta, weights, densities, reciprocal_batch=64,
pair_cutoff_bohr=None, budget=ResourceBudget(), kernel="coulomb", omega=0.)`
builds Coulomb/erf/erfc exchange on explicit momentum-transfer domains. The
default full-Coulomb kernel uses the native convention

$$
Q=G+k_t-k_s,\qquad B(Q;k_t)=F(Q;k_t),\qquad
K_d(k_t)=\sum_s w_s\sum_{Q\in\mathcal Q_{ts}}
\frac{4\pi}{\Omega |Q|^2} B(Q;k_t)^\dagger D_d(k_s) B(Q;k_t).
$$

The Fourier matrix has source-momentum bra rows and target-momentum ket
columns. For the same inversion-closed image support, it is the transpose of
the existing periodic exchange cache's negative-source-phase matrix. Tests
compare both conventions after physical AO normalization, as well as an
independent Gaussian-quadrature construction.

`densities` is a complex128 array of shape `(density,k,AO,AO)`. The leading
axis may hold alpha/beta spin densities or independent trial states. Every
channel has its own K response; each generated Fourier panel is reused across
all channels. All densities are snapshotted before any integral work. They
may be indefinite and need not obey time reversal, but each k block must be
finite and Hermitian within `1e-12 * max(1,max(abs(block)))`. A large value in
another block cannot relax this test. Real-valued densities must be explicitly
promoted to complex128. All inputs must be C or Fortran contiguous arrays.

The remaining input contract is explicit:

- `lattice`: float64 `(3,3)`, finite nonsingular three-dimensional lattice
  vectors as columns in bohr.
- `translations`: float64 `(cells,3)`, ordered unique inversion-closed direct
  lattice vectors. The optional physical AO-pair cutoff has the same inclusive
  meaning as the Bloch Fourier provider; `None` disables it.
- `momenta`: float64 `(k,3)` in inverse bohr. `weights`: float64 `(k,)`,
  nonnegative and summing to one within `1e-12`. Both k and density counts
  must be positive.
- `wave_vectors`: float64 `(total_Q,3)`, containing the concatenated ordered
  Q lists for target-major, source-minor momentum pairs.
- `transfer_offsets`: int64 `(k*k+1,)`, starting at zero, monotone, and ending
  at `total_Q`. Pair `(t,s)` uses the half-open slice
  `transfer_offsets[t*k+s]:transfer_offsets[t*k+s+1]`.

Within each pair, `Q-k_t+k_s` must have unique reciprocal integer labels.
The reverse pair must have the opposite label set; diagonal pair supports
are inversion closed. This makes the weighted density-response pairing
self-adjoint and validates the energy derivative. Lattice-label tolerance is
`1e-10` with magnitude at most `2^40`, matching the other finite periodic
providers. Actual Cartesian values and ordering remain unchanged. No k point,
transfer or support is wrapped, rounded, completed or deduplicated. Different
pairs can have different finite domains, provided reversal holds. A caller
that relabels k by a reciprocal vector can retain the same physical Q arrays.

**Kernel and zero-mode policy:** `kernel` is `coulomb`, `erf` or `erfc`.
Coulomb requires `omega=0`; the screened kernels require finite `omega>=0`
in inverse bohr. Replace the full kernel coefficient in the equation by

$$
v_{\mathrm{erf}}(Q)=\frac{4\pi}{\Omega Q^2}
 e^{-Q^2/(4\omega^2)},\qquad
v_{\mathrm{erfc}}(Q)=\frac{4\pi}{\Omega Q^2}
 \left[1-e^{-Q^2/(4\omega^2)}\right].
$$

Coulomb and erf omit Q=0 and add no Madelung correction. For positive omega,
erfc admits exact Q=0 with the finite coefficient `pi/(omega**2*volume)`.
It is included only when present in the caller's explicit support; the builder
does not insert it. `zero_vectors` reports how many such terms were consumed.
Nonzero vectors whose squared norm underflows to zero are rejected. At omega=0,
erf is identically zero and erfc equals Coulomb on the same nonzero support;
Q=0 remains excluded for both zero-range requests.

On an identical nonzero Q support, erf plus erfc equals Coulomb. Including the
finite erfc zero mode adds `w_s*pi/(omega**2*volume)*B(0)^dagger*D_s*B(0)`;
it must not be silently canceled or combined with a singularity correction.
The erfc evaluator uses `expm1` and a scaled small-argument limit, retaining
positive tails that subtraction from full Coulomb would lose. Tests compare
these tails to high-precision scalar formulas and a converged reciprocal sum
to an independent real-space screened Gaussian sum.

These explicit finite reciprocal sources add no Madelung or Ewald gauge
correction. The execution receipt records `kernel`, `omega` and `boundary`:
`full-coulomb-q0-omitted-no-madelung`, `erf-q0-omitted-no-madelung`,
`erfc-finite-q0-on-explicit-support-no-madelung`, or
`erfc-zero-range-q0-omitted-no-madelung`. Source identity binds the kernel/range;
the established default Coulomb identity bytes are preserved.

The result has read-only `plan` and `receipt`; `values` exports an independent
complex `(density,k,AO,AO)` array. Receipt `energies[d]` is
`-0.5 * sum_k w_k Tr(D_d(k) K_d(k))`, the one-spin exchange functional.
Its density derivative is `-w_k K_d(k)`. Restricted total-density conventions
require their own spin factors; the builder inserts no occupancy multiplier.
Raw response Hermiticity and energy imaginary residuals are checked at relative
scale `1e-10` without repairing the matrices. Source identity is independent
of density count and computational batch size. Per-density identities and a
complete epoch identity record the actual input snapshot.

`plan_periodic_exchange(basis, vectors_per_pair, cells, momenta, densities,
reciprocal_batch, pair_cutoff_bohr, budget, kernel="coulomb", omega=0.)`
validates the kernel before allocation and uses the maximum vector count of
any transfer pair, conservatively applying it to every pair. Compute admission
first bounds the offset metadata scan, reads only the offset counts, and then
admits the complete request before copying numerical arrays. The plan covers
all input/label owners, responses, compensated sums, one Fourier batch and
dense two-index contraction scratch. It never materializes a four-index AO
tensor or all-image AO basis. Empty transfer supports return zero exchange
while still validating the density epoch. Actual shell, primitive, vector,
phase and matrix-product counts are reported separately from upper bounds.

`manta.build_periodic_exchange` accepts the same numerical arrays and optional
batch/cutoff/kernel/range, with `integral_engine=IntegralEngineOptions(backend="manta")`
instead of a raw budget. It returns the native result and records execution
provenance through `vibeqc.output`, including the explicit zero-mode policy and
`manta_periodic_exchange` citations. The adapter requires `fallback="error"`
and `channel_batch=1` and reserves its provenance storage before the build.
This provider alone does not enable complete periodic SCF, CCM, symmetry
reduction, force or stress routes.

### Fixed whole-mesh spin Fock snapshots

`compute_periodic_fock(basis, lattice, hartree_vectors, exchange_vectors,
transfer_offsets, translations, momenta, weights, hcore, spin_densities,
reciprocal_batch=64, pair_cutoff_bohr=None, budget=ResourceBudget(),
kernel="coulomb", omega=0., exchange_scale=1.)` composes the finite J/K
providers into an owned, fixed matrix snapshot:

$$
D=D_\alpha+D_\beta,\qquad
F_\sigma(k)=H(k)+J[D](k)-aK[D_\sigma](k),
$$

$$
E_{\rm el}=\sum_k w_k\operatorname{Tr}[D(k)H(k)]
+\tfrac12\sum_k w_k\operatorname{Tr}[D(k)J(k)]
-\tfrac a2\sum_{k\sigma}w_k\operatorname{Tr}[D_\sigma(k)K_\sigma(k)].
$$

Here `a=exchange_scale` is finite; `kernel` and `omega` select the exchange
kernel under the preceding zero-mode contract. Hartree remains full Coulomb
with G=0 omitted. Hartree G and exchange Q arrays are separate explicit
supports. They need not have the same size. There is no inferred exchange
singularity correction, XC functional, nuclear energy or support convergence.
A caller composing a physical method must supply and qualify those pieces.

`hcore` is contiguous complex128 `(k,AO,AO)` and `spin_densities` is contiguous
complex128 `(2,k,AO,AO)` in alpha/beta order. Their values are snapshotted;
H and each density block must be finite and Hermitian to relative tolerance
`1e-12`. Restricted callers pass each spin as `D_total/2`, giving
`F=H+J[D_total]-a*K[D_total]/2` with the corresponding quarter factor in the
exchange energy. No occupancy factor is inserted implicitly. Indefinite spin
densities are valid response inputs. A large block cannot relax another
block's Hermiticity test. Raw Fock Hermiticity is checked at `1e-10`, without
repairing any matrix.

The native result retains read-only plans/receipts. `values` exports owned
`(2,k,AO,AO)` Fock matrices; `hartree_values` exports `(k,AO,AO)` J;
`exchange_values` exports the **unscaled** `(2,k,AO,AO)` K response.
Each export is an independent copy. The component exchange receipt also
reports unscaled per-spin energies; the enclosing Fock receipt reports the
scaled `exchange_energy` and complete `electronic_energy`. Input changes,
later builds and mutations of exported arrays cannot change the native
snapshot. This is a matrix snapshot, not a new solver/operator interface.

`one_electron_identity` authenticates the supplied H matrix; it does not claim
that MANTA generated it from nuclei. `source_identity` includes that identity,
the exact J/K source identities and exchange scale. `density_identity` binds
both spin densities and their total-density Hartree source.
`snapshot_identity` additionally hashes the actual Fock matrices and energy
components, so it records evaluated bytes as well as the numerical source.
Batch size does not change source or density identity, though roundoff can
change evaluated bytes. Tests check the weighted electronic-energy derivative
against each spin/k Fock block, independent quadrature, spin interchange,
restricted factors, time reversal and atom-image gauge covariance.

`plan_periodic_fock(basis, hartree_vectors, exchange_vectors_per_pair, cells,
momenta, reciprocal_batch, pair_cutoff_bohr, budget, kernel, omega,
exchange_scale)` reserves both component peaks plus the enclosing H, F,
input, sum-density, export and receipt owners. This is conservative: the two
backend workspaces are reserved together even though their integral execution
is sequential. Work budgets bound the sum of J, K and composition work.
Admission occurs before numerical snapshots; a component being affordable
alone does not admit the combined request. No four-index or all-supercell AO
matrix is formed. The exchange response is currently evaluated even when
its coefficient is zero; its work is not hidden in the receipt.

`manta.build_periodic_fock` supplies the same interface with
`integral_engine=IntegralEngineOptions(backend="manta")` in place of a raw
budget. It emits `manta_periodic_fock` through `vibeqc.output`, including both
executed component identities, zero-mode conventions, spin labels, supplied-H
provenance, energies and actual work. It requires `fallback="error"` and
`channel_batch=1`, and reserves Python provenance storage before construction.
The finite RHF composition below supplies a complete iteration on these
sources. Singularity-corrected Hamiltonians, CCM, symmetry and derivative
consumers require their own subsequent source qualification.

### Periodic UHF implementation boundary

The internal bounded UHF numerical solver supports fixed integer alpha and
beta populations on a uniform full k mesh, with at least one virtual band
in each spin and one electron in total. An empty spin is supported. Both
spin densities enter one callback evaluation; spin labels never replace
physical k-point labels. It retains the last evaluated density and Fock
pair on convergence and on capped exits. Occupations, metric idempotency,
commutators and density closure are checked for each spin. The commutator
RMS sums squared residuals over both spins and divides by the number of
physical k points before taking the square root. The insulating
gate uses the smaller within-spin gap at the prescribed spin populations;
it does not assert stability against changing the total magnetization.

The explicit-support physical driver is now available as
`manta.run_periodic_uhf`, with `manta.PeriodicUHFOptions` and the same
`basis`, `system`, full `RegularKMesh`, reciprocal vectors, per-transfer
exchange vectors/offsets and AO translations used by the RHF interface below.
Pass `integral_engine=IntegralEngineOptions(backend="manta", budget=budget)`.
`plan_periodic_uhf` performs count-only admission before vector copies.

The system must be neutral, all-electron and three-dimensional. The specified
multiplicity M fixes `N_alpha=(N+M-1)/2` and `N_beta=(N-M+1)/2`; inconsistent
populations refuse. The driver generates physical S/T/V and nuclear Ewald
energy, then evaluates `F_spin=H+J[D_alpha+D_beta]-K[D_spin]` across the entire
mesh. It supports explicit `strict-zero-mode` omission and `BvK-ewald` exchange.
The latter adds `-xi*S*D_spin*S` separately to each spin Fock and its matching
half-trace energy. At an idempotent fixed point its energy contribution is
`-xi*N/2` per cell, with no extra mesh-size factor.

The result owns evaluated density and Fock arrays of shape `(2,nk,nbf,nbf)`,
coefficients `(2,nk,nbf,rank)`, orbital energies `(2,nk,rank)`, and shared S/H
arrays `(nk,nbf,nbf)`. The two spin slots are alpha then beta. Receipts retain
both populations, physical source identities, energy components, correction
identity and fresh terminal confirmation errors. One scalar history row per
evaluated iteration is reserved before allocation; confirmation is a separate
Fock call. Convergence requires the numerical verdict and a fresh matching
physical Fock/energy build. A capped exit returns the evaluated density, which
may differ from the density reconstructed from terminal coefficients.

Provider selection and executed provenance go through `vibeqc.output`, with
UHF and exchange-correction citations. The selected `run_periodic_job` route described below also supports UHF
and lossless spin-density QVF delivery. Finite-support
SCF convergence is not a claim of cutoff convergence. Periodic RI/BUPO,
reduced-dimensional boundaries, CCM, symmetry reduction and periodic
derivatives remain separate implementation stages.

The [bounded UHF numerical benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_uhf_solver.md)
records the solver's algebraic validation. The
[physical UHF benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_uhf.md)
adds independently integrated finite-source SCF, component energies, executed
work and complete native driver costs.

### Joint functional energy and potential numerical contract

The C++ `bounded_periodic_functional_solver.hpp` interface supplies separate
`solve_bounded_periodic_rks` and `solve_bounded_periodic_uks` numerical entry
points. They retain the bounded HF solvers' fixed occupations, overlap-rank
policy, whole-mesh snapshots and stopping diagnostics. Existing HF entry
points and their half-trace energy convention are unchanged.

A joint native callback must write both `G[D]=F[D]-H` and the complete
interaction energy per primitive cell at the same evaluated density. The
latter excludes one-electron and nuclear energy. The solver evaluates
`E=E_nuclear+sum_spin,k Tr(D H)/K+E_interaction[D]`; it does not infer a
nonquadratic functional's energy from its potential. Every callback resets
its energy destination to a nonfinite sentinel. Missing energy or potential
lanes refuse before publishing a snapshot. Potential Hermitian repair is
reported separately and does not change the provider's functional energy.

The planner admits the joint provider's retained storage, peak workspace and
work, plus its energy scalar and bookkeeping. Restricted density is spin
summed; unrestricted density has separate alpha/beta slots, with uniform
`1/K` weights and no extra spin average. Callers must independently establish
that their potential is the derivative of the supplied energy.

These are numerical contracts, qualified with nonlinear algebraic models;
the [functional solver benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_functional_solver.md)
records independent errors, work and numerical storage.
The explicit XC sources and finite physical KS driver described later add
separate support, energy, variational and resource qualification. Selected
pure-KS jobs are described below, followed by global-hybrid sources and
their public job contracts. KS derivatives retain a separate gate.

### UHF from physical cutoffs

`manta.run_periodic_uhf_from_cutoffs(basis, system, mesh,
reciprocal_energy_cutoff, options=options, integral_engine=engine)` generates
pair-complete AO translations and certified spheres for G and every exchange
transfer. The pair radius is the explicit `options.pair_cutoff_bohr` in bohr;
the reciprocal kinetic cutoff is in hartree. Nuclear and BvK cutoffs retain
their independent controls. Fixed spin populations and source conventions
are the same as the explicit UHF driver above.

`plan_periodic_uhf_from_cutoffs` admits conservative source bounds and the
complete UHF solver before enumeration. Its `uhf` field holds the spin-aware
physical plan. `build_periodic_uhf_domains` returns owned arrays for inspection
and explicit-source reproduction, with the same geometric receipt and identity
schema used by RHF. Domain identity describes the support; the UHF physical
source identity also includes the prescribed spin populations.

### Complete RHF on explicit finite periodic supports

`manta.run_periodic_rhf(basis, system, mesh, reciprocal_vectors,
exchange_vectors, transfer_offsets, translations, options=None,
integral_engine=IntegralEngineOptions(backend="manta"))` constructs S, T,
Ewald nuclear attraction, nuclear energy, Hartree and full exchange, then
solves RHF using the existing bounded periodic numerical solver. It accepts
neutral all-electron three-dimensional singlets with an even electron count
and at least one virtual orbital. Basis centers must match their assigned
nuclei modulo lattice translations. The initial guess is HCORE.

`mesh=manta.RegularKMesh((nx, ny, nz), (sx, sy, sz))` reuses the canonical
exact-address regular mesh, where each shift flag is 0 or 1. Every mesh point
participates with uniform weight; shifted meshes are supported. The driver
does not reduce the mesh by symmetry. Reciprocal G vectors, exchange Q
vectors and transfer offsets obey the preceding component contracts.
Translations specify the finite AO image source. G is shared by Hartree,
nuclear attraction and nuclear energy; Q remains a separate explicit support
for each ordered target/source transfer. Arrays are contiguous float64
`(count,3)` vectors and int64 `(nk*nk+1)` offsets.

`PeriodicRHFOptions` supplies `alpha`, `nuclear_cutoff_bohr`,
`pair_cutoff_bohr`, `reciprocal_batch`, terminal confirmation tolerances and
the existing solver controls through `.scf`. Nuclear real-space energy uses
the physical atom-pair sphere at `nuclear_cutoff_bohr`; its reciprocal, self
and uniform-background terms use the same alpha and G source as attraction.
The electrostatic convention is three-dimensional tin-foil, zero average.
Full exchange omits Q=0 by default (`exchange_q0="strict-zero-mode"`).
`exchange_q0="BvK-ewald"` explicitly adds the probe-charge Ewald correction
described below. No supplied vector support is completed or enlarged. These finite supports define
the evaluated Hamiltonian and must be converged separately for a physical
prediction.

Each Fock call consumes the entire current density with
`D_alpha=D_beta=D_total/2`. The driver recomputes the Fock matrix and physical
energy on the final density before reporting `converged=True`. Both the
solver's convergence checks and this terminal confirmation must pass.
Iteration limits and insufficient band gaps return an unconverged result
with the last evaluated density/Fock state. Invalid inputs, an indefinite
overlap, insufficient retained rank, resource exhaustion and unsupported
solver policies raise errors. DIIS is currently unsupported; the default
`scf.maximum_diis_history=0` must be retained. Redundant positive-semidefinite
overlap directions can be truncated under the configured rank thresholds.

Results export independent owned `overlap`, `hcore`, `density`, `fock`,
`coefficients` and `orbital_energies` arrays. Read-only `solver_snapshot`,
`solver_diagnostics` and `solver_memory` expose residuals, band gap, rank and
resource evidence. The numerical solver permits only bounded, explicitly
diagnosed Hermitian projections; raw component checks still apply. The
receipt's `final_fock_identity` authenticates the raw MANTA producer snapshot,
while exported `fock` is the solver matrix and may include that bounded
projection. Terminal confirmation records their maximum element difference
and the physical-versus-solver energy difference.

The receipt separates all four nuclear contributions, one-electron, Coulomb
and exchange energies. `uncorrected_exchange_energy` and
`exchange_correction_energy` sum to `exchange_energy`. `source_identity` binds the actual S/T/V, nuclear
energy, Fock source and exact mesh; `final_density_identity` binds the final
evaluated density. Changing batch size does not change the numerical source.
The selected adapter requires `fallback="error"` and `channel_batch=1`,
reserves its Python receipt storage and emits `manta_periodic_rhf` through
`vibeqc.output` with convergence status and executed citations.

`plan_periodic_rhf(basis, system, mesh, reciprocal_vectors,
exchange_vectors_per_pair, cells, options, budget)` accepts counts and bounds
the complete run before numerical array copies. It reserves component
workspaces, full-rank solver storage, driver buffers and exports, and bounds
Fock work by `maximum_iterations+1` including terminal confirmation. Its
simultaneous reservations are deliberately conservative; modeled memory is
not measured process RSS. Charge unrelated live storage through
`budget.other_live_bytes`.

The low-level APIs also supply the selected `run_periodic_job` route below.
Periodic KS, RI/BUPO, one- and two-dimensional electrostatics,
CCM, symmetry reduction, forces and stress require
further implementation. The [complete finite RHF benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_rhf.md)
reports independent SCF agreement, actual whole-run work, wall/CPU times and
process RSS; it makes no production speedup or support-convergence claim.

#### Generate the finite RHF source from cutoffs

`manta.run_periodic_rhf_from_cutoffs(basis, system, mesh,
reciprocal_energy_cutoff, options=options, integral_engine=engine)` generates
the finite domains and executes the same complete native RHF calculation.
Set an explicit positive `options.pair_cutoff_bohr`. The reciprocal kinetic
energy cutoff is in hartree. For example:

```python
options = manta.PeriodicRHFOptions()
options.pair_cutoff_bohr = 9.0
options.nuclear_cutoff_bohr = 12.0
options.exchange_q0 = "BvK-ewald"
result = manta.run_periodic_rhf_from_cutoffs(
    basis, system, manta.RegularKMesh((2, 1, 1), (1, 0, 0)),
    reciprocal_energy_cutoff=60.0, options=options,
    integral_engine=manta.IntegralEngineOptions(backend="manta"),
)
```

These cutoffs illustrate the interface, not a universal accuracy prescription.
All existing RHF capability restrictions and resource limits still apply.
Increasing a cutoff can require increasing the explicit resource budget.
The AO translation list comes from the shared pair-complete lattice source:
its enclosing radius is the physical pair cutoff plus the largest shell
center separation. S, T, attraction and J/K retain the same physical
`|O_mu-O_nu-R|` pair filter. An atom represented outside the home cell remains
covered, with its Bloch gauge preserved. An origin-only AO image list is
allowed when the explicit pair cutoff excludes every nonzero image; the
reciprocal periodic electrostatics still use the full lattice.

The shared certified reciprocal enumerator selects `|G|^2/2 <= cutoff` for
Hartree and nuclear terms, and `|G+k_target-k_source|^2/2 <= cutoff` for each
exchange transfer. Exact integer mesh addresses determine centered transfer
representatives. Gamma excludes the exact zero label. Non-Gamma spheres use
the shared enumerator's outward radial tolerance of
`128*eps*max(sqrt(2*cutoff)+|q_centered|, 1)` in inverse bohr. Opposite
domains must pass its exact conjugacy audit, including opposite momentum
bits and equal norms and Coulomb weights. An asymmetric boundary result
raises an error. The factory never repairs it by silently changing a domain.
Nuclear real-space and optional BvK probe cutoffs remain separate controls.

`plan_periodic_rhf_from_cutoffs(basis, system, mesh, cutoff, options, budget)`
bounds source construction and the complete SCF before domain allocation.
It uses certified reciprocal boxes and the shared real-space producer's
enclosing box, which can overestimate the accepted spheres substantially.
It includes conjugacy scans, sorting, copies, exports, native source owners
and all permitted Fock calls. The returned `rhf` subplan contains those
worst-case box counts. No integral evaluations occur during planning.

`build_periodic_rhf_domains` accepts the same arguments and returns read-only
plan/receipt objects with independent owned `translations`,
`reciprocal_vectors`, `exchange_vectors` and int64 `transfer_offsets`
exports. Replaying them through `run_periodic_rhf` evaluates the same source.
The executed RHF receipt records actual vector counts, the cutoff, domain
selection and `domain_identity`. This identity binds the lattice, mesh,
shell centers, cutoffs and generated vectors; the ordinary `source_identity`
continues to bind the actual numerical Hamiltonian. Domain construction
storage and work appear separately in the final plan and selected output.
Cutoff convergence must be checked independently of SCF convergence. The
[cutoff-domain benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_domains.md) reports
complete-run timing, domain construction, actual work, memory and independent
reference checks.

#### Selected periodic jobs

`run_periodic_job` accepts the same `IntegralEngineOptions`,
`CoulombBuildOptions` and `ExchangeBuildOptions` selectors as molecular jobs.
An explicit MANTA selection uses the cutoff factory and complete native RHF or UHF
driver described above. Pure RKS/UKS jobs use the joint grid/domain source
[described below](#selected-periodic-pure-ks-jobs). The ordinary route is unchanged when these selectors
are omitted. The selected exact periodic HF profile requires `method="RHF"` or `method="UHF"`,
`jk_method="auto"`, exact J and exact K, and `use_diis=False`.
The [fitted UHF job profile](#selected-periodic-ri-uhf-jobs) has separate explicit
J/K and fitting selections. For exact HF:

```python
result = vq.run_periodic_job(
    system, basis, method="RHF", kpoints=(2, 1, 1),
    integral_engine=manta.IntegralEngineOptions(backend="manta"),
    coulomb=manta.CoulombBuildOptions(algorithm="exact"),
    exchange=manta.ExchangeBuildOptions(algorithm="exact"),
    periodic_integrals=manta.PeriodicIntegralOptions(
        reciprocal_energy_cutoff=2.0,
        pair_cutoff_bohr=3.0,
        nuclear_cutoff_bohr=4.0,
        alpha=0.6,
    ),
    use_diis=False, initial_guess="HCORE", max_iter=80,
    conv_tol_energy=1e-10, conv_tol_grad=1e-9,
    exchange_exxdiv="ewald", output="periodic_manta",
)
```

These coarse demonstration cutoffs are not a physical accuracy recommendation.
Converge the domains and supply an adequate explicit engine budget for the
system being studied. `PeriodicIntegralOptions` also exposes
`reciprocal_batch`, `madelung_alpha` and `madelung_tolerance`.
`exchange_exxdiv="ewald"` selects the BvK correction; `"none"` selects strict
zero-mode omission. The Ewald split and BIPOLE/GDF-specific controls of other
routes cannot be substituted for these source options. All-electron basis
admission uses the shared periodic ECP resolver.

`kpoints=None` or `1` means Gamma. Integer and three-division tuple/list
inputs are Gamma-centered; `RegularKMesh`, full regular `KPoints` and
`BlochKMesh` preserve their explicit shift. Payloads must match the cell,
canonical address ordering and uniform full-mesh weights. Irreducible meshes,
smearing recommendations and inconsistent mesh metadata are refused.
`initial_guess="AUTO"` resolves to HCORE. `max_iter` and `conv_tol_energy`
set the bounded solver's iteration and energy-change controls;
`conv_tol_grad` sets its commutator Frobenius RMS tolerance. Native density
closure, rank, eigenproblem and terminal confirmation checks still apply.
Unconverged runs retain the last evaluated density/Fock state and report
`result.converged=False`.

The job writes its log and manifest through `vibeqc.output`, with optional
geometry, citation, structured-log and QVF artifacts. QVF carries the
executed integral/domain provenance in `x_vibeqc.periodic_integral_engine`
and the evaluated density in the existing Bloch spin-density format. RHF
uses equal alpha/beta channels; UHF retains its two distinct channels. This preserves the physical state even when
terminal Fock orbitals do not reproduce an unconverged density. The archive
uses nuclear-centered basis functions. If source shells occupy other lattice
images, the adapter transports D into that basis gauge and records the exact
per-AO integer image shifts; the native result keeps the original source
gauge. The extra primitive metadata and serialization owners are included in
the job's memory reservation. The archive includes each evaluated SCF
iteration's energy and, from the second iteration, its energy change.
The log and structured log additionally retain commutator RMS, density
closure and gap diagnostics. Records are delivered after the solve;
live progress callbacks and checkpoints are not yet exposed. QVF marks an
unconverged calculation `failed` while the manifest separately records
artifact completion and `not_converged` calculation status.

`PeriodicMANTAResult` exposes `energy`, `converged`, `iterations`,
`density_k`, `fock_k`, `overlap_k`, `hcore_k`, `mo_coeffs_k`, `mo_energies_k`,
`occupations_k`, `scf_history`, mesh metadata and copied `integral_provenance`. Matrix
properties return owned exports; `.native` retains the complete native
receipt and diagnostics. Canonical orbitals describe the terminal Fock.
The QVF density does not substitute their reconstructed density.

UHF returns `PeriodicMANTAUHFResult`. Its `density_k` is the spin sum;
`density_alpha_k` and `density_beta_k` preserve the evaluated spin states.
Use `fock_alpha_k`/`fock_beta_k`, `mo_coeffs_alpha_k`/`mo_coeffs_beta_k`,
`mo_energies_alpha_k`/`mo_energies_beta_k` and
`occupations_alpha_k`/`occupations_beta_k` for separate spin exports. There
is no common UHF `fock_k`, `mo_coeffs_k`, `mo_energies_k` or `occupations_k`.
Each spin occupation is zero or one; an empty beta channel remains explicit.
System multiplicity fixes the populations at every k point, and convergence
does not establish stability against changing magnetization.

UHF provenance and structured iteration records retain prescribed spin
populations and separate HOMO, LUMO and gap diagnostics. An empty spin's HOMO
and gap serialize as null. The QVF vendor provenance includes this complete
`spin_scf_history`, alongside the standard energy history. Both spin densities
are transported to the nuclear-center gauge for QVF; native exports preserve
the original gauge. The job reserves both sets of matrix exports and scalar
history before native admission.

`scf_history` returns owned dictionaries using the QVF `iter`, `energy_eh`
and optional `delta_e` fields. `.native.scf_history` returns copied,
read-only scalar solver snapshots, including residuals and algebra checks.
Only the final snapshot's status describes termination; earlier snapshots
report the diagnostics of intermediate evaluations.
The native plan reports `history_capacity` and `history_live_bytes` for the
maximum requested iteration count. The job separately reserves Python
record and serialization storage against `max_iter` before native admission.
Each evaluated iteration contributes one row, including capped exits;
the independent final Fock confirmation contributes no extra iteration.
The last solver snapshot's convergence flag precedes that confirmation,
so `result.converged` remains the complete calculation verdict.

Dry runs perform resource admission and write the standard plan without SCF.
Explicit requests for unsupported algorithms, dimensionalities, ECPs, READ,
DIIS, mixing, symmetry reduction, optimization, response, band-property
postprocessing, density grids, Molden, population files or checkpoints fail
before SCF or computed-result artifacts. The standard job lifecycle retains
a minimal manifest for capability refusals. Capability-aware `None` sidecar
defaults emit only supported artifacts. No implicit provider or algorithm
fallback occurs. The [UHF job benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_uhf_job.md)
records independent spin-state and output evidence. The [complete job benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_job.md)
includes standard artifact delivery, numerical reference checks, work counts
and measured process memory.

#### BvK probe-charge Ewald exchange correction

Set `options.exchange_q0="BvK-ewald"` to add the full-Coulomb reciprocal
zero-mode correction. The probe lattice has columns `mesh[d]*lattice[:,d]`.
Its scalar `xi_M=-2*E_Ewald(unit charge + background)` is computed once using
the shared native point-charge Ewald source. It depends on the full regular
mesh and lattice, and is independent of nuclear positions and mesh shifts.
For the total restricted density, the additional terms are

$$
\Delta F(k)=-\tfrac12\xi_M S(k)D(k)S(k),\qquad
\Delta E=-\tfrac14\xi_M\sum_k w_k
\operatorname{Tr}[D(k)S(k)D(k)S(k)].
$$

The actual MANTA overlap, with the same AO image support and pair cutoff,
enters both terms. There is no additional `1/nk` factor in the Fock term:
the supercell scalar already includes that normalization. There is also no
`pi/(alpha**2*V*nk)` subtraction: that subtraction belongs to a real-space
erfc plus reciprocal erf exchange split, while this route evaluates full
reciprocal Coulomb. For an idempotent restricted solution the energy changes
by `-Ne*xi_M/2`, occupied orbital energies change by `-xi_M`, and the
stationary density is unchanged.

`madelung_alpha=None` chooses `sqrt(pi)/V_supercell**(1/3)`; an explicit
positive value changes only the probe's Ewald split. `madelung_tolerance`
defaults to `1e-12` and sets real/reciprocal radii from Gaussian decay. It is
a cutoff heuristic, not a certified total-error bound. Nuclear attraction's
`alpha` and explicit G source remain separate controls. `plan.madelung`
reports the resolved probe alpha, radii, enclosing candidate counts, work
and storage. Those counts and the shared producer's support limit are
admitted before array snapshots or integral work. The complete-run plan
also charges every `S D S` contraction and its scratch storage.

The receipt retains actual probe-vector counts, `madelung`,
`exchange_correction_identity`, both exchange-energy contributions and the
executed `exchange_q0`. `result.exchange_q0` exposes the same convention to
the shared comparison/manifest gate. Final Fock and density identities bind
the corrected source, and terminal confirmation rebuilds the corrected
matrix and energy. The output adds `manta_periodic_madelung` citations only
when the correction executes. The two conventions remain distinct finite
Hamiltonians; neither implies complete image, reciprocal or k-mesh convergence.

Tests compare Gamma, full two-point and shifted-mesh SCF to independent
periodic AFT Hartree-Fock at matching conventions, and check reciprocal and
nuclear-split refinement, mesh scaling, rank/resource policies and independent
finite-source SCF. See the [corrected RHF evidence](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_madelung.md)
for work and timing measurements. The correction follows
[Sundararaman and Arias (2013)](https://doi.org/10.1103/PhysRevB.87.165122),
using the existing Ewald producer rather than a separate lattice-sum algorithm.

## Four-center tiles and J/K

`ShellTileSession(basis, backend="manta", request=KernelRequest(), budget=budget)`
admits a provider and workspace for one shell-quartet evaluation at a time.
The alternative backend is `libint`, with its existing full-Coulomb capability.
`plan_shell_tiles` returns a `BuildPlan` with `schedule="shell-tiles"`, one
worst-case shell call, bounded primitive/tile work, and zero AO target storage.
It scans and owns the full basis inventory without admitting or executing a
whole-basis quartic traversal. This lets an independent near-field scheduler
own its execution plan.

`session.evaluate((a,b,c,d))` returns an owned four-dimensional tile and the
actual provider/source metadata. The workspace is reused; earlier exported
tiles remain valid. Python evaluations retain the GIL and execute serially.
`session.calls` counts executed evaluations. Budgets at this boundary apply
per call, not cumulatively to an arbitrary loop: a consuming scheduler must
admit its aggregate calls/work, density and output buffers, and other live
storage. MANTA also reports measured primitive evaluations; libint reports
requested primitive products and marks its internal evaluation count as
unmeasured. Source identities are identical to the same integral provider in
an ordered J/K build, because changing the schedule does not change a tile.

`builder.evaluate((a, b, c, d))` returns an owned four-dimensional `values`
array for one ordered shell quartet, its source identity and evaluated/null
call counters. The last index is contiguous. No multiplicity or density/spin
factor is embedded in the tile. This diagnostic method creates a workspace
for each call. For a schedule of many quartets, `session = builder.session()`
and `session.evaluate((a, b, c, d))` reuse a single workspace. The session
retains its provider and each exported tile still owns its storage. Python
evaluations retain the GIL and execute serially. Native consumers likewise
reuse one workspace throughout their build.
Cartesian components use libint's shell-wide axial normalization. For example,
Cartesian d-xy has one third the overlap norm of d-xx, not an independently
renormalized unit overlap.

The reference traversal evaluates every ordered shell quartet once with zero
primitive-screening precision. A null libint buffer is recorded and treated
as zero; it is not certified as an exact analytic zero. No final matrix
symmetrization conceals a contraction discrepancy. Receipts report actual
shell calls, requested primitive quartets, consumed elements and accumulated
density terms. Requested primitive quartets are not a count of the backend's
undocumented internal primitive operations.

MANTA MD evaluates each admitted primitive quartet without a screening
threshold. Its `null_tiles` count is zero; arithmetic underflow can still
produce zero values. A nonfinite recurrence or contracted result fails the
build. Neither provider silently falls back to the other.

`plan(basis, budget=budget)` performs count-based admission before basis copies,
engine construction and matrix allocation. Caps cover shell calls, primitive
quartets, scalar contractions and modeled peak memory. The memory model includes
borrowed/copy input, owned basis data, targets, known libint arrays, tile export
and explicit backend headroom. The backend allowance is at least 8 MiB;
allocator capacity and shared-table behavior prevent treating this as an RSS
guarantee. Charge other live results, concurrent sessions and caller storage
through `other_live_bytes`. Exported shell inventories are separate caller-owned
diagnostic copies. No global four-index tensor is built by this consumer.
These J/K plans retain `schedule="ordered-jk"` and their full traversal caps.

The MD plan additionally caps `kernel_terms`, a conservative arithmetic-work
model for recurrences, matrix products and representation transforms. It
reserves one primitive quartet workspace, rather than storage proportional to
the product of all primitive counts. The modeled backend allowance is at least
64 KiB. The current unreduced schedule is a correctness baseline; no speed
advantage over optimized direct libint is claimed.

Provider basis data are copied after admission. J/K results and exported tiles
own their storage and survive later builds or producer destruction. Capturing
a fixed Fock action remains the method's responsibility; generation labels or
source hashes do not freeze mutable closures.

The [architecture](../design/manta_engine.md),
[scientific contracts](../design/manta_scientific_contract.md), and
[staged validation plan](../design/manta_validation_plan.md) describe subsequent
large fitting solves, periodic/CCM sources, symmetry and derivatives. Those planned
capabilities are not enabled by selecting the MD four-center provider.

The implementation follows [McMurchie and Davidson (1978)](https://doi.org/10.1016/0021-9991(78)90092-X)
and the matrix formulation of [Neese (2023)](https://doi.org/10.1002/jcc.26942).
For low-level integrations, the citation assembler accepts
`extra_methods=["manta_md"]`. This selects both algorithm references without
claiming all SHARK features or a different physical Hamiltonian.


### Explicit periodic local exchange

`plan_periodic_local_exchange` and `compute_periodic_local_exchange` add an
experimental finite-source Dirac local-exchange (`lda_x`) leaf. They evaluate
both the exchange energy and its density derivative. This is a physical AO
quadrature component, not a selected RKS/UKS job or a complete XC functional.
The following LDA/GGA and physical KS increments add correlation, GGA and
same-density SCF confirmation. Meta-GGA, hybrids, grid/domain factories and
selected KS jobs retain their separate gates.

Inputs are a basis (Cartesian or pure shells through l=4), a nonsingular 3D
lattice with vectors in columns, explicit Cartesian lattice-image rows,
Cartesian momenta and positive normalized k weights, and explicit per-cell
grid points in bohr with nonnegative quadrature weights in bohr cubed.
`densities` has complex128 shape `(spin, k, AO, AO)`. One spin means the
restricted total density; two means separate alpha/beta densities. Empty
spin channels are accepted. No time-reversal folding or real-density
projection is used. The momenta need not occur in plus/minus pairs.

The Bloch AO is `B_mu(k,r) = sum_T exp(i k.T) chi_mu(r-T)`. The source uses
`rho_s(r) = sum_k w_k B(k,r) D_s(k) B(k,r)^dagger` and returns
`V_s(k) = integral B(k,r)^dagger v_s(r) B(k,r) dr`. In particular, the
returned potential has no extra k weight; the derivative is
`dE = sum_s,k w_k Re Tr(dD_s(k) V_s(k))`. The energy is the Dirac functional,
not half the density-potential trace. Restricted and spin-polarized exchange
obey exact spin scaling. The reference is Dirac (1930),
[doi:10.1017/S0305004100016108](https://doi.org/10.1017/S0305004100016108),
with citation route `numerics.manta_periodic_local_exchange`.

Each shell center is reduced to fractional `[0,1)` before applying the common
image list, then its Bloch phase is transported back to the supplied shell
gauge. Fractional coordinates within `1e-10` of an integer are snapped to that
integer. Independently moving a shell by lattice vector L therefore requires
`D'_ij(k) = exp(i k.(L_i-L_j)) D_ij(k)` and produces the same transformation
of V with invariant energy. This defines the finite support under image-gauge
changes; a finite image sum is not an infinite-lattice tail certificate.
Both density and potential use every supplied image on both AO indices.

Callers must supply and refine an actual per-cell quadrature and adequate
image reach. Arbitrary input weights do not certify a periodic integral.
The receipt reports integrated spin populations, energy, minimum raw grid
density, source/density/snapshot hashes and executed AO work. Hermitian
input is required; materially negative grid spin density is refused. Small
negative roundoff is counted and uses the continuously differentiable
extension `rho -> max(rho,0)` of exchange. This grid check does not certify
that an arbitrary input matrix is positive semidefinite.

The count-only plan admits source arrays and binding copies, the serial AO
workspace, a `grid_batch * k * AO` complex panel, output/compensation/export
matrices, basis metadata and backend allowance before numerical input copies.
No full `grid * k * AO` owner is retained. The existing budget field
`primitive_quartets` bounds `primitive_points` for this leaf; the receipt
reports the correctly named executed count. The memory model is not a process
RSS bound. Inputs must not be concurrently mutated; results own their values.
See `benchmarks/manta_periodic_local_exchange.md` for measured validation.


### Explicit periodic LDA/GGA exchange-correlation

`plan_periodic_semilocal_xc(basis, functional, images, momenta, grid_points,
grid_batch=64, budget=...)` and `compute_periodic_semilocal_xc` extend the
finite complex Bloch component to native libxc LDA/GGA. The compute arguments
match `compute_periodic_local_exchange`, with a `vibeqc.Functional` after
`basis`. Its `spin` must match the density's leading dimension. Results have
`PeriodicSemilocalXCPlan` and `PeriodicSemilocalXCReceipt`, which inherit the
local-exchange geometry/resource fields and add functional component IDs,
weights, backend version, executed functional batches and workspace allowance.

The same explicit lattice, image, quadrature, normalization and shell-gauge
contract applies. GGA evaluates AO values and all three spatial gradients in
bounded panels. Both density gradients and V use those same images. With
`g_s = grad(rho_s)`, the polarized potential flow vectors are
`q_a = 2 v_sigma_aa g_a + v_sigma_ab g_b` and
`q_b = 2 v_sigma_bb g_b + v_sigma_ab g_a`. The potential adds
`sum_c q_s,c ((partial_c B)^dagger B + B^dagger partial_c B)` to its local
density term. This includes the mixed-spin invariant exactly once in each
flow vector. The returned joint energy is the libxc integrand quadrature;
it must not be replaced by an HF half trace.

Complete native pure LDA/GGA components with 3D energy and first-derivative
capabilities are admitted. Hybrid, range-separated, double-hybrid, meta-GGA,
VV10 and external full-grid functionals are refused before source copies.
Their missing components are not silently discarded. Standard qualified
fixtures cover Slater exchange, LDA/VWN, PBE, BLYP, PBEsol and BPW91.
Correlation papers follow the functional's usual citation route; the common
quadrature route is `numerics.manta_periodic_semilocal_xc`.

No full-grid AO or gradient owner is retained. Planning includes the value
and gradient panels, density/invariant/potential arrays, libxc spin buffers
and a conservative per-component backend allowance. The model is not a
process RSS bound. Raw grid spin densities must be nonnegative up to reported
roundoff. A negative roundoff value is clamped to zero together with its
spatial gradient; the returned potential includes this clamp's chain rule.
Exactly empty spins remain valid and may have a nonzero minority-spin
correlation potential. libxc's own low-density regularization is retained.

This component does not by itself enable complete public MANTA RKS/UKS jobs.
The explicit physical driver below supplies Hartree composition and matched
SCF confirmation, with cutoff-domain generation in the following section.
Quadrature generation, calibrated accuracy profiles, general functionals and
job outputs remain separate qualification steps. See `benchmarks/manta_periodic_semilocal_xc.md` for the
independent gradient and variational checks and measured resource evidence.


### Explicit finite-support periodic RKS and UKS

`plan_periodic_ks(basis, system, mesh, functional, reciprocal_vectors, cells,
xc_images, grid_points, options=PeriodicKSOptions(), budget=...)` admits the
complete finite physical source by counts. `run_periodic_ks` takes
`basis, system, mesh, functional, reciprocal_vectors, translations, xc_images,
points, quadrature_weights`, followed by the same options and budget.
Vector arguments are contiguous float64 `(count,3)` arrays, and quadrature
weights are contiguous float64 `(grid_points,)`. The lattice uses columns.

The functional's spin count selects singlet RKS (`spin=1`) or collinear UKS
(`spin=2`). The neutral, all-electron 3D system supplies the electron count
and fixed multiplicity. A uniform full `RegularKMesh` supplies k points and
weights `1/K`. Every spin must have a virtual orbital. Complete pure native
LDA/GGA is admitted; hybrid, range-separated, double-hybrid, meta-GGA, VV10,
external functionals, charged cells, lower dimensions and incompatible spin
populations are refused. The explicit method citation route is
`manta_periodic_ks`, together with the selected functional's citation route.

The driver computes S, T, nuclear attraction and nuclear Ewald energy from
physical inputs. Each evaluated density jointly produces
`F_s = H + J[D_total] + V_XC,s[D]` and
`E = E_nuclear + sum_s,k Tr(D_s,k H_k)/K + E_J + E_XC`.
RKS uses the total density; UKS sums both spin densities for Hartree.
The numerical solver receives the joint energy and potential. XC energy is
the actual integrand quadrature, not an HF half trace. Converged numerical
results receive an additional complete physical build at the returned density;
`converged` also requires energy and Fock confirmation to pass. Capped exits
retain the last evaluated density, Fock, energy and scalar iteration history.

`PeriodicKSOptions` supplies the existing bounded SCF controls, nuclear
`alpha` and `nuclear_cutoff_bohr`, optional `pair_cutoff_bohr`, reciprocal and
grid batch sizes, and positive energy/Fock confirmation tolerances. Hartree
and nuclear G vectors omit G=0, with neutral tin-foil zero-average electrostatics.
Pure semilocal KS has no exact-exchange or exchange Madelung correction.

The physical pair translations and XC canonical shell images are distinct
finite supports. Supply unique inversion-closed physical pair translations
and reciprocal vectors; XC uses the explicit canonical images and per-cell
quadrature of the preceding section. Refine physical pair support, reciprocal
support, nuclear support, XC image support and quadrature separately.
Solver confirmation certifies only this finite composite Hamiltonian.

`PeriodicKSResult` owns S/H `[K,n,n]`, density/Fock `[spin,K,n,n]`, coefficients
`[spin,K,n,rank]` and orbital energies `[spin,K,rank]`; even RKS retains a
leading spin dimension of one. Exports are independent copies. Solver
snapshots, diagnostics, memory and `scf_history` retain the selected spin
contract. The receipt includes separate nuclear, one-electron, Hartree and
XC energies, child source receipts, identity hashes and executed work counts.
The XC receipt reports per-spin quadrature populations;
`maximum_grid_population_error` compares them with fixed metric populations.
Finite pair cutoffs can make these counts differ even on a refined grid.
Charge agreement alone does not certify GGA energy or potential convergence.

Aggregate admission includes all child source plans, simultaneous J/XC
results, copied and summed spin densities, the numerical solver, physical
S/H, binding/export copies, the joint functional scalar, iteration history
and terminal confirmation work. The legacy `primitive_quartets` cap bounds
`primitive_work`, the sum of primitive-pair and primitive/grid-point work;
both components are also reported separately. This is a modeled resource
bound, not a process RSS guarantee. Inputs must remain immutable during a call.

This explicit-support driver has no automatic production accuracy profile.
The joint grid/domain producer and selected public jobs below add their own
admission and output contracts. The following global-hybrid API has its own
exchange contract. Other nonlocal and kinetic-density sources and remaining
MANTA capabilities retain separate qualification gates.

### Explicit finite-support periodic global hybrids

`plan_periodic_global_hybrid_ks` and `run_periodic_global_hybrid_ks` compose a
complete native global LDA/GGA hybrid on the same neutral, all-electron 3D
full-mesh physical source. The plan takes `basis, system, mesh, functional,
reciprocal_vectors, exchange_vectors_per_pair, cells, xc_images, grid_points`,
followed by `options=PeriodicGlobalHybridKSOptions()` and `budget`.
The run takes `basis, system, mesh, functional, reciprocal_vectors,
exchange_vectors, transfer_offsets, translations, xc_images, points,
quadrature_weights`, followed by those options and budget. `transfer_offsets`
is contiguous int64 `[K*K+1]`, starts at zero and ends at the Q-vector count.
For each target/source pair it selects the corresponding Q slice. Reverse
supports must obey the exchange leaf's conjugacy contract. Vector arrays and
grid weights have the float64 layouts used by pure KS.

```python
from vibeqc import manta

options = manta.PeriodicGlobalHybridKSOptions()
options.ks.alpha = 0.7
options.ks.nuclear_cutoff_bohr = 8.0
options.ks.pair_cutoff_bohr = 8.0
options.ks.scf.maximum_iterations = 100
options.exchange_q0 = "BvK-ewald"
```

`options.ks` contains the shared KS and solver controls. Hybrid options are
a separate type so the pure API cannot silently ignore exchange settings.
The functional resolves its own nonzero full-Coulomb exchange coefficient
`a`; built-in PBE0 and B3LYP and custom global LDA/GGA mixtures are supported.
The native semilocal component is evaluated by libxc, and the existing finite
exchange source supplies the omitted exact exchange. Range-separated,
screened, meta-GGA, double-hybrid, VV10 and external functionals are refused
before numerical array copies. Pure functionals use the pure-KS API.

For total-density RKS the Fock contribution is `-a K[D]/2`, with exchange
energy `-a sum_k Tr(D_k K[D]_k)/(4K)`. UKS uses `-a K[D_s]` for each spin and
`-a sum_s,k Tr(D_s,k K[D_s]_k)/(2K)`. The total energy adds nuclear,
one-electron, Hartree, semilocal XC and exact exchange energies. It remains
the method's joint energy functional, not an HF half-trace approximation.

`exchange_q0="strict-zero-mode"` explicitly omits Q=0. `"BvK-ewald"` uses the
shared full-supercell probe-charge Ewald source with optional positive
`madelung_alpha` and `madelung_tolerance` in `(0,1)`. The correction is
`-a xi S D S/2` in RKS and `-a xi S D_s S` in UKS, with its energy obtained
from half the density trace. For an idempotent state the energy correction is
`-a N xi/2`. There is no additional k-weight factor inside this Fock correction.
The exchange source, its zero-mode convention, physical overlap and method
fraction all contribute to complete source identity.

The returned `PeriodicKSResult` has the same owned spin/k arrays, evaluated
scalar history, terminal physical confirmation and separate next orbital
projector as pure KS. The plan adds `exchange`, `madelung`,
`exact_exchange_fraction` and correction/provider work counts. The receipt
adds the unscaled exchange child receipt, `exchange_q0`, Madelung source
identity and vector counts, and `exact_exchange_fraction`.
`xc_energy` is the semilocal component; `uncorrected_exchange_energy` and
`exchange_correction_energy` are already scaled by `a`, and their sum is
`exchange_energy`. The child XC receipt explicitly records
`semilocal_component_only=True` and `required_exact_exchange_fraction`.
It must not be interpreted as a complete hybrid energy by itself.

All five resource caps jointly include Q/offset copies, simultaneous J/K/XC
results, Madelung setup and spin-matrix scratch, the solver and a terminal
rebuild before any numerical payload is copied. This is modeled admission,
not a process RSS cap. Physical pair translations, exchange Q supports and
canonical XC images remain distinct finite sources. Source refinement and
XC energy/potential accuracy profiles remain required.

Independent finite-source tests cover both spin conventions, unequal and
empty spins, both zero-mode policies, complex energy derivatives, source AO
image gauges, rank truncation, capped states and exact resource limits.
At an exactly empty spin, PBE0 and B3LYP inherit the libxc endpoint sensitivity
seen in pure KS: one-epsilon relative changes of the occupied density can
move the minority potential. Separate sensitivity witnesses retain the
actual zero spin and backend regularization. They do not establish a uniform
minority-potential accuracy bound. Strict full-matrix empty-spin tests use
an independently defined B88/LYP global hybrid.

Cutoff-generated sources and joint atomic quadrature are described below.
Public hybrid jobs and QVF output have the separate admission contract below.
BIPOLE dispatch, symmetry reduction, IBZ meshes and derivatives still require
their own scientific qualification. The
exported evaluated D/F and method energy components are compatible with the
shared Fock architecture's snapshot contract; no second generic Fock object
or Python adapter is inserted into the bounded native loop. BIPOLE CRYSTAL
parity and symmetry qualification remain independent of these finite-source
tests. Cite `manta_periodic_global_hybrid_ks` and the selected functional route.

### Explicit finite-support periodic range-separated KS

`plan_periodic_range_separated_ks` and `run_periodic_range_separated_ks`
compose native LDA/GGA semilocal components with every resolved exact-exchange
kernel. They use the same leading arguments, array layouts and count-only
planning interface as the explicit global-hybrid API above, with
`PeriodicRangeSeparatedKSOptions`. Its `ks` member is `PeriodicKSOptions`;
`exchange_q0` defaults to `"strict-zero-mode"`, with `"BvK-ewald"` available.
`madelung_alpha` and `madelung_tolerance` have the global-hybrid meanings.
The neutral, all-electron, fixed-occupation, full uniform 3D mesh and virtual
space requirements remain in force. Both RKS and UKS use the existing bounded
native functional solver and return the same owned `PeriodicKSResult`.

The method reads `Functional.exchange_components`, including distinct ranges
in composite functionals. It canonicalizes their real-space exchange kernel to

$$v_x(r)=A/r+\sum_j B_j\,\operatorname{erfc}(\omega_j r)/r.$$

Here `omega` is in inverse bohr and coefficients include their signs.
An erf term becomes Coulomb minus erfc. Equal ranges are combined;
zero-range erfc becomes Coulomb and zero-range erf vanishes. Exactly cancelled
components are omitted. Aggregate CAM metadata cannot describe multiple
ranges and is not used as a substitute for this list. The short-range fraction
is $A+\sum_j B_j$ and the long-range fraction is $A$.

| Functional | A | Screened components (B, omega) |
| --- | ---: | --- |
| HSE06 | 0 | (0.25, 0.11) |
| CAM-B3LYP | 0.65 | (-0.46, 0.33) |
| LC-omegaPBE (`lc-wpbe`) | 1 | (-1, 0.4) |
| omegaB97X (`wb97x`) | 1 | (-0.842294, 0.3) |

Supply base exchange Q vectors with Q=0 omitted, as for global hybrids.
The method appends one exact zero to each diagonal target/source k domain for
every positive-range erfc kernel. Its finite Fourier value is
$\pi/\omega^2$, with the ordinary source-k weight $1/N_k$ and cell-volume
normalization. Off-diagonal domains keep the supplied support. The Coulomb
term keeps the base nonzero Q domains. For BvK Ewald it also adds

$$\Delta F_{\sigma k}=-\frac{A\xi}{f}\,S_kD_{\sigma k}S_k,
\qquad \Delta E_x=\frac12\sum_{\sigma k}w_k
  \operatorname{Re}\operatorname{Tr}(D_{\sigma k}\Delta F_{\sigma k}),$$

where $f=2$ for restricted total density and $f=1$ for each UKS spin.
There is no extra k weight in this potential correction. Screened kernels
receive no Madelung correction. If $A=0$, the actual `exchange_q0` is
`"not-applicable"`, and inactive Madelung controls do not alter source identity.
The requested choice remains in `exchange_q0_requested`.

This explicitly defines a finite-mesh Coulomb-plus-erfc convention. In an
erf decomposition, the subtracted finite erfc zero remains present. Applying
independent zero-mode omissions to separate erf and erfc leaves, or a
range-dependent probe-charge correction, defines a different finite source.
The admitted calculations do not establish equivalence to those conventions
or convergence in the mesh, Q support, pair support or XC quadrature.

Plan and receipt set `range_separated=True`. Their `exchange_components`
contain `kernel`, `coefficient`, `omega` and the corresponding unscaled
exchange plan or receipt. Each result component also contains its signed,
method-scaled `energy`. `long_range_exchange_fraction` and
`short_range_exchange_fraction` describe the complete list. The old single
`exchange` child is absent and `exact_exchange_fraction` remains zero;
neither field describes a range-separated operator. The XC child records
`semilocal_component_only=True` and the complete
`required_exchange_components`, rather than a single required fraction.

`uncorrected_exchange_energy` sums the component energies, including their
finite erfc zeros. `exchange_correction_energy` is solely the long-range
Madelung term; their sum is `exchange_energy`. The returned density, Fock,
component energies and history describe the same evaluated epoch, including
an unconverged capped exit. Terminal orbitals can describe the next projector.
Each export and component receipt owns its contents.

All five aggregate caps include every resolved exchange leaf, the retained
zero-expanded Q source, plans and receipts, spin scratch, solver, history and
terminal confirmation. Count-only admission precedes payload copies. This is
a conservative numerical-memory model, not a process RSS bound. Pure and
global-hybrid entry points continue to refuse range separation. Meta-GGA,
double hybrids, VV10, Yukawa and external components remain refused here.

Independent finite-source contractions and SCF tests cover HSE06, CAM-B3LYP,
LC-omegaPBE, omegaB97X, mixed ranges and empty-spin LC-BLYP. Range-separated
cutoff/grid factories and public job routing have separate contracts below.
Production accuracy profiles remain required. Exported physical snapshots were checked with
the shared FockBuild draft; no new generic operator or Python adapter enters
the bounded loop. The full mesh, source identities and component list remain
available for later method-owned symmetry admission. BIPOLE AC/BD source
matching and CRYSTAL parity remain independent requirements. Cite
`manta_periodic_range_separated_ks` and the selected functional route.

### Range-separated cutoff domains and atomic quadrature

`plan_periodic_range_separated_ks_from_cutoffs` and
`build_periodic_range_separated_ks_domains` take `basis, system, mesh,
functional, reciprocal_energy_cutoff, xc_image_cutoff_bohr, grid_points,
options, budget`. Supply `PeriodicRangeSeparatedKSOptions` with a positive
finite `options.ks.pair_cutoff_bohr`. The run API
`run_periodic_range_separated_ks_from_cutoffs` replaces `grid_points` with
contiguous float64 `points[count,3]` and `quadrature_weights[count]`.

The domain source uses the shared certified reciprocal kinetic-energy spheres
and physical pair-complete translation sphere, plus a distinct canonical XC
image sphere. Base Q vectors omit zero and are identical to HF/global-hybrid
domains for identical geometric controls. The range-separated method adds
finite erfc zeros and resolves every exchange component after the complete
request has been admitted. All ranges share the same supplied finite Q cutoff;
no component inherits another component's range parameter.

`PeriodicRangeSeparatedKSDomainPlan`, `PeriodicRangeSeparatedKSDomainReceipt`
and `PeriodicRangeSeparatedKSDomains` have the global-hybrid domain fields.
The plan contains the complete range-separated `ks` plan. The receipt exposes
`pair_identity` for the shared G/Q/translation geometry and `xc_image_identity`
for the canonical image source. `exchange_vectors` and the recorded
`domain_exchange_vectors` count the nonzero base Q support; each erfc child
receipt separately reports its added zeros. Changing a functional kernel
changes the complete KS source identity, not the unchanged geometry identity.
Exports own their arrays and stay valid independently of inputs and other exports.

`plan_periodic_range_separated_ks_with_quadrature` and
`run_periodic_range_separated_ks_with_quadrature` take the same leading
arguments, followed by `grid_options, options, budget`, with
`PeriodicQuadratureOptions` replacing the explicit grid. The
`PeriodicRangeSeparatedKSGridPlan` contains `quadrature`, `domains`,
`grid_live_bytes` and aggregate caps. It uses the same 3D radial/Lebedev and
periodic Becke/Stratmann source as the pure and global-hybrid factories.
Grid owners and copied payloads remain reserved through domain generation,
all SCF builds and final confirmation. Grid work is subtracted before
downstream admission. The returned `PeriodicKSResult` records the grid source,
controls, domain counts and all exchange-component receipts.

All five whole-request caps are checked before generating source vectors or
grid points, and explicit array copying follows admission. Independent tests
cover Gamma and shifted skew-cell membership, exact cutoff boundaries, mixed
ranges, restricted and unequal/empty spins, both singularity choices, both
partitions, evaluated capped states and AO image gauges. Generated and
explicit sources agree bitwise in the physical D/F/energy state; separate
Gaussian/libxc contractions and SciPy SCF validate those results.
This qualifies finite source generation and composition, not source convergence,
an automatic grid profile, BIPOLE/CRYSTAL parity
or symmetry reduction.

### Global-hybrid cutoff domains and atomic quadrature

`plan_periodic_global_hybrid_ks_from_cutoffs` and
`build_periodic_global_hybrid_ks_domains` take `basis, system, mesh, functional,
reciprocal_energy_cutoff, xc_image_cutoff_bohr, grid_points, options, budget`.
The run API `run_periodic_global_hybrid_ks_from_cutoffs` replaces `grid_points`
with contiguous float64 `points[count,3]` and `quadrature_weights[count]`.
Supply `PeriodicGlobalHybridKSOptions` with a positive finite
`options.ks.pair_cutoff_bohr`.

The shared HF source generator selects physical pair translations, nonzero
`|G|^2/2 <= reciprocal_energy_cutoff` and all target/source
`|G+k_target-k_source|^2/2 <= reciprocal_energy_cutoff` exchange supports.
It audits opposite-transfer conjugacy before filling the Q arrays.
`xc_image_cutoff_bohr` independently selects the canonical AO-image sphere;
zero explicitly selects the origin only. Pair translations can grow when an
AO center moves by a lattice vector while canonical XC images stay unchanged.
None of these supports substitutes for BIPOLE's separately defined AC/BD
exchange domains.

`PeriodicGlobalHybridKSDomainPlan` contains the complete hybrid `ks` plan,
geometric candidate bounds, modeled domain storage/work allowances and
whole-request caps. The shared generator calls this hybrid planner directly;
it does not substitute restricted HF electron counts or an HF resource model.
`PeriodicGlobalHybridKSDomains` owns independent exported arrays for
translations, G, Q, transfer offsets and XC images. Its receipt's
`pair_identity` equals the HF domain identity when that geometry is identical.
`xc_image_identity` describes the separate image sphere, and `identity` binds
both. The physical result retains `domain_identity`, cutoff values and
actual translation/G/Q/XC/conjugacy counts, including `domain_exchange_vectors`.

For generated atomic quadrature, use
`plan_periodic_global_hybrid_ks_with_quadrature` and
`run_periodic_global_hybrid_ks_with_quadrature`. They take the same leading
arguments through `xc_image_cutoff_bohr`, then `grid_options`, `options` and
`budget`. `grid_options` is the existing `PeriodicQuadratureOptions` described
below. `PeriodicGlobalHybridKSGridPlan` contains the quadrature plan, hybrid
domain plan, retained grid bytes and aggregate caps. A single preflight admits
grid generation, grid ownership/copies, all physical supports, the complete
hybrid solver and final confirmation. The returned `PeriodicKSResult` records
the quadrature identity and options as well as hybrid source/energy receipts.

All five exact/one-short caps are tested before numerical enumeration and
payload copies. Independent integer enumeration checks skew-cell G/Q and
direct-image membership. Whole SCF and individual energy terms agree with
independent Gaussian/libxc contractions and SciPy fixed points for RKS, UKS,
unequal and empty spins under both zero-mode policies. Joint-grid outputs are
bit-for-bit identical to composing the exported finite sources explicitly;
capped exits retain evaluated D/F/E and their grid identity. The same source
AO frame and separate next-projector contract apply to unified Fock consumers.

These factories choose finite supports; they do not certify continuum or
quadrature accuracy. Explicit pair, reciprocal, nuclear and XC-image support
and grid energy/potential refinement are still required. The public
`run_periodic_job` hybrid selector uses this joint source with the output,
provenance and QVF contract below. No symmetry or BIPOLE producer is admitted.


### Pure-KS physical cutoff domains

`plan_periodic_ks_from_cutoffs` and `build_periodic_ks_domains` take
`basis, system, mesh, functional, reciprocal_energy_cutoff,
xc_image_cutoff_bohr, grid_points, options, budget`.
`run_periodic_ks_from_cutoffs` replaces the `grid_points` count with
`points, quadrature_weights`. The quadrature is still supplied explicitly;
these functions generate physical source domains, not quadrature points or
an automatic accuracy profile.

`options.pair_cutoff_bohr` must be finite and positive. The shared
pair-complete source encloses translations through that cutoff plus the
maximum shell-origin separation. Each physical kernel retains its own exact
pair-distance predicate, including unwrapped shell image labels. Hartree G
vectors use the shared certified `|G|^2/2 <= reciprocal_energy_cutoff` source,
with G=0 omitted and an explicit conjugacy audit. Canonical XC images use the
shared Cartesian lattice sphere `|R| <= xc_image_cutoff_bohr`, which may be
zero to request the origin image explicitly. All three domains are independent
of k-mesh size; pure KS does not construct exchange-transfer tables.

`PeriodicKSDomainPlan` bounds the two direct-space boxes and reciprocal box,
source traversal/sorting/owned exports, retained domains and the complete
worst-case KS calculation for the declared grid count. Planning does not
allocate source vectors. Both build and run refuse inadequate aggregate
budgets before source enumeration; run also admits before copying quadrature
payloads. Actual accepted support can be smaller than the planned boxes.
The domain result owns translations, reciprocal vectors and canonical XC
images. Its separate identity binds lattice, shell centers and cutoffs,
while the physical KS identity also binds the selected mesh, functional and
quadrature. Changing a batch size leaves the numerical source unchanged.

The KS receipt reports generated-domain identity, selected cutoffs, actual
support sizes and conjugacy count. Physical and quadrature convergence remain
separate requirements. See `benchmarks/manta_periodic_ks_domains.md` for
membership, independent SCF, gauge and bounded-resource evidence.

At a fully polarized PBE endpoint, the libxc 7.0.0 minority-spin potential
can be sensitive to fp64 input rounding. A tested empty-beta fixture retained
energy/occupied-spin agreement while a one-epsilon alpha-density perturbation
moved the minority potential by several microhartree. The source retains
libxc's regularization. Empty-spin minority potentials and spectra therefore
need their own conditioning checks; energy or occupied-density agreement does
not imply uniform absolute accuracy for that endpoint. This limitation is
independent of cutoff generation and is recorded in its benchmark report.

### Bounded periodic atomic quadrature

`manta.plan_periodic_quadrature(system, options, budget)` and
`manta.build_periodic_quadrature(system, options, budget)` provide an explicit
3D grid source. `PeriodicQuadratureOptions` selects `radial_points` (default
50), `lebedev_order` (29), `partition_radius_bohr` (8) and `stratmann` (false).
The adapter reuses the shared generic Treutler-Ahlrichs radial rule, an
unpruned positive-weight bundled Lebedev rule and the shared point-centered
periodic Becke partition. `stratmann=True` selects the existing piecewise
polynomial partition of Stratmann, Scuseria and Frisch
([1996](https://doi.org/10.1016/0009-2614(96)00600-8)). It does not select a
PySCF grid profile or change the shared producer.

The source convention is explicit: canonicalize every nucleus into fractional
`[0,1)`, construct its raw atom-major periodic quadrature, then fold every
point into that same cell while retaining its weight. The finite Bloch AO
image sum is evaluated at those folded points. Its periodized finite source
can have a boundary discontinuity; refine XC image support independently.
Zero weights remain in the returned arrays. `points` and `weights` are owned
float64 copies suitable for the existing explicit KS and XC interfaces.
The result receipt binds the protocol, canonical geometry and actual arrays
with a source digest, and reports nonzero weights, their sum and cell volume.
Neither volume nor charge agreement certifies XC energy or potential accuracy.

This adapter admits 3D nonsingular cells with lattice singular values between
`1e-4` and `1e4` bohr and condition number at most `1e4`, atomic numbers 1 to
118, 1 to 10000 radial points, and a partition radius in `(0,1e4]` bohr.
Fractional coordinates beyond magnitude `1e6`, unsupported Lebedev orders,
negative-weight rules, coincident periodic nuclei and unrepresentable image
extents are refused. Charge, multiplicity and attached symmetry are not part
of this geometry-only source. Pruning, product angular rules, external grid
profiles and 1D/2D cells require separate admission work.

The count-only plan includes the full shared grid, one simultaneous owned
export of each public array, radial/angular scratch, atom metadata, image
atlas and concurrent per-thread neighbor/partition lists. It bounds both
nearest-image search and partition neighborhoods using the sum of lattice
vector lengths, including conservative floating-point and integer-box slack.
It reserves the current OpenMP maximum thread count without changing global
thread settings. Keep inputs and thread configuration immutable during a
call. Additional retained copies must be included in `other_live_bytes`.
Memory, scalar and kernel caps are checked before grid generation; no shell
or primitive-integral work is charged. These are conservative modeled bounds,
not process RSS guarantees. Work bounds can exceed the default resource caps
even for small grids; explicitly size the budget from the plan and intended
calculation rather than bypassing admission.

When calling grid generation and KS separately, count the retained grid in
the later operation's external live bytes. The joint interface below makes
that reservation automatically; selected pure-KS jobs use that joint plan.
The default grid parameters are construction settings, not an automatic
accuracy promise. Refine radial/angular resolution, partition radius, XC
image support and physical Hamiltonian support separately, checking XC energy
and every spin/k potential as well as populations. The benchmark records a
fixture where a 40-radial/order-59 grid still has potential error above
`1e-5` hartree, and larger radial/angular grids alone are not monotonically
more accurate. A larger partition radius also matters. See
`benchmarks/manta_periodic_quadrature.md` for measured limits.

For citation output select `numerics=["manta_periodic_quadrature"]`, adding
`"stratmann_grid"` when that partition is used. Automatic accuracy profiles remain required MANTA work.


### Joint quadrature, domains and physical KS

`plan_periodic_ks_with_quadrature` and `run_periodic_ks_with_quadrature` take
`basis, system, mesh, functional, reciprocal_energy_cutoff,
xc_image_cutoff_bohr, grid_options, options, budget`. Here `grid_options` is
an explicit `PeriodicQuadratureOptions` and `options` is the existing
`PeriodicKSOptions`. Positive physical pair support is still required.
These calls compose the preceding finite sources without changing their
Hamiltonian, grid convention or fixed occupations.

The `PeriodicKSGridPlan` exposes both child plans, retained `grid_live_bytes`,
aggregate work counts and the maximum modeled peak across generation and
SCF. It admits the complete request before numerical grid generation. Grid
scalar/kernel work is subtracted before domain/SCF admission, and grid owners
plus explicit KS source copies remain charged throughout SCF and terminal
confirmation. Integral shell and primitive budgets apply to the physical
calculation; grid work uses scalar/kernel caps. Exact-cap and one-short
checks cover all five aggregate limits.

Run returns the existing `PeriodicKSResult`. `plan.quadrature` and
`plan.quadrature_live_bytes` describe the generated source and its live
reservation. `receipt.quadrature` preserves its source digest and diagnostics;
`grid_radial_points`, `grid_lebedev_order`, `grid_partition_radius_bohr` and
`grid_stratmann` record the selected controls. The usual domain identity,
physical source identity, same-density energy decomposition, capped-epoch
history and terminal confirmation remain available. Explicit-grid KS calls
leave the optional quadrature plan/receipt unset.

This joint admission removes manual lifetime accounting between these three
steps. It does not infer quadrature accuracy from SCF convergence. Defaults
are still explicit finite construction choices; general accuracy profiles
remain unqualified. See the independent
finite-source SCF benchmark in `benchmarks/manta_periodic_ks_grid.md`.


(selected-periodic-pure-ks-jobs)=
### Selected periodic KS jobs

`run_periodic_job(method="RKS" or "UKS", functional="pbe", ...)` admits
complete native pure, global-hybrid and range-separated LDA/GGA functionals,
such as `"pbe"`, `"pbe0"`, `"hse06"` or `"cam-b3lyp"`, with explicit
`PeriodicXCOptions`.
It requires the same neutral all-electron 3D cell, full regular mesh, fixed
occupations and virtual-space admission as the native KS driver. Select
`integral_engine=IntegralEngineOptions(backend="manta", budget=budget)` and
supply `periodic_integrals=PeriodicIntegralOptions(...)` as for HF, plus:

```python
periodic_xc = manta.PeriodicXCOptions(
    xc_image_cutoff_bohr=11.0,
    radial_points=12,
    lebedev_order=17,
    partition_radius_bohr=8.0,
    stratmann=False,
    grid_batch=64,
)
# Supply periodic_xc=periodic_xc to run_periodic_job, with use_diis=False.
```

These small values reproduce a finite test fixture, not a recommended
accuracy profile. Radii are in bohr; zero XC-image radius selects the home
image only. `radial_points` must be 1 through 10000; the angular order must
be a supported positive-weight bundled Lebedev rule. The default 50 radial
points, order 29 and 8-bohr partition radius are also construction settings,
not convergence certificates. `grid_batch` changes bounded execution only.
The complete grid, domain, SCF, trace and output memory/work request is
admitted before numerical point generation. Dry runs perform this admission
without building a grid or solving SCF.

J uses `CoulombBuildOptions(algorithm="exact")`. Pure KS evaluates no exact
exchange; omitted `exchange` selects `ExchangeBuildOptions(algorithm="none")`,
and explicit exact/RI exchange requests refuse. `exchange_exxdiv` has no
active pure-KS correction: both accepted legacy values (`"ewald"`, `"none"`)
produce `exchange_q0="not-applicable"` and
`exchange_q0_applicability="inactive"`. Nondefault Madelung controls refuse.

Hybrids instead select `ExchangeBuildOptions(algorithm="exact")`
when `exchange` is omitted; explicitly requesting `"none"` or `"ri"` refuses.
The native method resolves all exchange coefficients and ranges from the
functional's components, including registered custom functionals. The public selector does
not infer coefficients from functional names. For a nonzero long-range
Coulomb coefficient, `exchange_exxdiv="ewald"` (the job default) selects
`exchange_q0="BvK-ewald"`; `"none"` selects `"strict-zero-mode"`. Both have
`exchange_q0_applicability="active"` in that case.
`PeriodicIntegralOptions.madelung_alpha` and `madelung_tolerance` control
the BvK probe-charge sum independently of nuclear Ewald and the XC grid.
The underlying low-level hybrid API defaults to strict omission, so set the
same convention explicitly when comparing it with public jobs.

Range-separated jobs use the canonical Coulomb-plus-erfc finite source
described above. The full kernel descriptor chooses the admission route,
including composites with cancelled aggregate CAM metadata. Finite erfc
zero modes always retain their source-k weights. Only the resolved long-range
Coulomb coefficient receives the requested singular correction. HSE06 and
other screened-only sources therefore report `exchange_q0="not-applicable"`
and `exchange_q0_applicability="inactive"`. Their `exchange_q0_requested`
preserves the accepted request; changing valid inactive Madelung controls
does not change their numerical source or result. All components share the
explicit Q cutoff; no range-specific source refinement is inferred.

For example, use the same explicit `periodic_integrals` and `periodic_xc`
objects with `method="UKS", functional="pbe0", exchange_exxdiv="ewald"`.
The cell multiplicity fixes the spin populations. Double-hybrid, meta-GGA,
Yukawa, nonlocal additions and external functionals refuse
before numerical generation. Pure and hybrid paths both retain the required
full-mesh and virtual-space checks.

RKS returns `PeriodicMANTARKSResult` with the restricted k-indexed properties
listed above. Its native result retains the singleton spin axis. UKS returns
`PeriodicMANTAUKSResult` with the same separate spin properties as UHF.
Both preserve the evaluated density, its physical Fock, and the method's
separate one-electron, Hartree, semilocal XC and nuclear energies. Hybrids also
retain `uncorrected_exchange_energy`, `exchange_correction_energy` and their
sum `exchange_energy`. These three energies already contain the functional's
exact-exchange coefficient and the restricted spin factor. Total energy is
nuclear + one-electron + Hartree + semilocal XC + exchange; the correction
must not be added a second time. For global hybrids the child `exchange.energies` are unscaled
`-Tr(D K[D])/2` channel energies; `exchange_energy_scale` records their factor
(`a/2` for total-density RKS, `a` for UKS). The XC child marks a hybrid's
partial component with `semilocal_component_only=True` and its required
exchange fraction.

Range-separated provenance instead contains `exchange_components`, with
each signed `coefficient`, `omega`, `kernel`, method-scaled `energy` and
`energy_scale`, plus its unscaled child `exchange` receipt. Those receipts
retain density identities, finite zero counts and all kernel/source evidence.
The XC child records its complete `required_exchange_components` list.
`long_range_exchange_fraction` and `short_range_exchange_fraction` describe
the full method; the legacy `exact_exchange_fraction` stays zero and the
single `exchange` child is absent. The method's uncorrected exchange sum
already includes finite screened zero terms. The manifest/run fields record
`exchange_source_identities`; full component records remain in structured
output and QVF vendor provenance. Public provenance reserves 49152 base
bytes for hybrids, and range jobs additionally reserve 32768 bytes per
possible canonical component, bounded by twice the input descriptor count,
before any source generation. This includes copied and serialized owners;
it is a modeled allowance, not a process RSS certificate.

Terminal orbital
projectors can describe a later density at an iteration cap. QVF writes the
evaluated spin density, transported to nuclear centers when necessary; it
never substitutes that later projector. Grid populations and their discrepancy
from metric populations remain diagnostics, separate from SCF convergence.

The standard manifest, structured log and QVF vendor evidence retain distinct
Hartree, XC, quadrature, domain, density and Fock source identities, plus the
hybrid exchange and correction identities. Functional
components, libxc version, grid controls, uniform mesh, source AO-image gauge,
spin histories and terminal confirmation remain inspectable. The output text
and `scf_energy_components` log event print evaluated component energies;
the manifest and QVF run fields retain them as machine-readable values.
Citations follow
the executed functional, MANTA source and Becke/Stratmann grid algorithms.

This boundary is compatible with fixed Fock/KS operator snapshots: each
spin/k matrix has its own immutable numerical values, `operator_generation`
identifies the evaluated iteration, and `basis_frame_identity` identifies the
overlap source and coordinate frame. Source receipts and energies stay with
the method, outside solver metadata. A diagnostic consumer can copy a
snapshot into the unified operator object; the native bounded solver keeps
its existing memory admission and does not create Python operator copies
inside its iteration loop. K weights enter densities and energies, never
orbital S normalization. Changing a shell image changes the frame identity.

The route reuses the shared point-centered periodic grid and physical
midpoint nuclear kernels used by BIPOLE. This does not establish matched
BIPOLE or CRYSTAL Hamiltonians or production crystal parity. The Hartree pair
support and XC-image support remain distinct. Hybrid reciprocal Q supports
cannot substitute for BIPOLE's independent AC/BD supports. BIPOLE-specific controls
refuse on this selected route. Symmetry reduction, IBZ reconstruction and
geometry-driven source reduction remain gated until these finite sources,
electronic states and retained spaces are independently qualified. The
RHF-only periodic mean-field state bridge is not used for KS.

The [public KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ks_job.md)
checks independent finite-source SCF, same-density component energies and
lossless QVF output while timing complete artifact delivery. General accuracy
profiles, periodic RI/BUPO, low-dimensional methods, CCM/chi, derivatives,
stress and acceleration remain required capabilities.

### Frozen-density periodic XC grid assessment

`manta.assess_periodic_xc_refinement` compares 2 to 32 explicit atomic grids
at one fixed density and finite Bloch source. It accepts the same pure native
LDA/GGA components and AO envelope as `compute_periodic_semilocal_xc`.
Select the reference index explicitly; the reference is evaluated first,
retained, and compared with each other grid. No SCF iteration is performed.

For a native global or range-separated LDA/GGA hybrid, use
`plan_periodic_hybrid_xc_refinement` and
`assess_periodic_hybrid_xc_refinement` with the same arguments below. This
explicit diagnostic evaluates only the semilocal component. The complete
functional descriptor is admitted, including signed or cancelled range
components; meta-GGA, double-hybrid, VV10, external and Yukawa sources refuse.
The ordinary pure-XC assessor and standalone semilocal leaf continue to
refuse hybrids.

```python
grids = []
for radial, angular, radius in ((40, 59, 8.), (80, 95, 16.)):
    grid = manta.PeriodicQuadratureOptions()
    grid.radial_points = radial
    grid.lebedev_order = angular
    grid.partition_radius_bohr = radius
    grids.append(grid)

# Existing fixed-state arrays use the same source AO frame. The grids below
# are illustrative controls, not a qualified accuracy profile.
plan = manta.plan_periodic_xc_refinement(
    basis, system, functional, len(images), len(momenta),
    grids, reference_index=1, grid_batch=64, budget=budget,
)
assessment = manta.assess_periodic_xc_refinement(
    basis, system, functional, images, momenta, weights, densities,
    grids, reference_index=1, grid_batch=64, budget=budget,
)
```

`images` and `momenta` are float64 `(count,3)` arrays, `weights` is float64
`(nk,)` with positive normalized weights, and `densities` is complex128
`(spin,nk,nao,nao)`. Restricted densities retain their singleton spin axis.
Pass a list or tuple of `PeriodicQuadratureOptions`. The source arrays must
remain unchanged during this synchronous call. Nonuniform weights use the
explicit source; they do not authorize IBZ reconstruction.

Each ordered `assessment.steps` entry owns the chosen options, quadrature
receipt and XC receipt. `energy_difference` is candidate minus reference;
`population_differences` has one entry per spin. `potential_max_abs` and
`potential_frobenius` are flattened in `(spin,k)` order and contain norms of
the complex potential difference in the unchanged source AO frame. K weights
are absent from these matrix norms. `reference_potential` returns an owned
complex `(spin,nk,nao,nao)` array containing the XC component, not the full
Fock matrix. Repeated exports are independent copies.

Hybrid plans and XC receipts set `semilocal_component_only=True`. Global
receipts retain `required_exact_exchange_fraction`; range-separated receipts
retain the canonical `required_exchange_components` list. Exchange and its
zero-mode correction are not evaluated by this diagnostic. At the identical
fixed density and unchanged grid-independent exchange source, a change only
in semilocal grid has the same energy and potential difference as a full
physical hybrid build. This identity does not apply if a grid-dependent K
source, other Hamiltonian term or the density also changes. Tests exercise
this identity using complete global/range-separated RKS/UKS builds.

The plan includes all requested grid/XC work and the simultaneous retained
reference, candidate, point/weight copies, metadata and one potential export.
It checks memory, shell calls, primitive/grid products, scalar work and kernel
work before numerical copies or grid generation. Peak memory is the largest
simultaneous stage, while work is summed across all grids. Modeled bytes are
not process RSS; charge other caller-retained arrays or exports through
`budget.other_live_bytes`. The result reports actual shell calls and
primitive/grid products as well as the aggregate plan.

`density_identity` binds the fixed density and source frame independently of
the grid choices. `source_identity` binds the ordered grids, functional source
and selected reference. Child receipts retain their full grid-dependent
identities. AO-image relabeling changes the frame identity while consistently
transported matrix differences retain their norms. This is evidence for the
method-owned Fock build; it does not introduce an operator object or symmetry
reduction inside the native calculation.

`comparison` is `frozen-density;explicit-reference-grid;not-an-error-bound`.
The hybrid diagnostic inserts `semilocal-component-only` after
`frozen-density` in that string.
The reference row is zero by construction, even for an inaccurate grid. These
AO norms are not basis-independent continuum error bounds. Independently
refine the reference in energy and every spin/k potential; charge agreement
and SCF convergence cannot substitute for that check. Finite-image errors,
density response, full-Hamiltonian matching and BIPOLE/CRYSTAL parity require
separate checks. The
[refinement benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_xc_refinement.md)
records both same-grid contraction errors and remaining reference errors.

### Periodic short-range fitting sources

`plan_periodic_gdf_short_range` and `compute_periodic_gdf_short_range` provide
jointly admitted raw erfc auxiliary metric and three-center blocks. They reuse
the existing GDF sphere/capsule image selector and replace its molecular
integral evaluator with MANTA. This supplies a finite short-range component
for later method assembly. It performs no metric solve, long-range addition,
finite-zero subtraction or exchange singular correction.

```python
options = manta.PeriodicGDFShortRangeOptions()
options.omega = 0.65                         # inverse bohr
options.pair_cutoff_bohr = 5.0
options.auxiliary_cutoff_bohr = 5.0
options.integral_screen_error = 0.0         # no integral-envelope screening
options.image_candidate_cap = 1000000
options.compute_metric = True

plan = manta.plan_periodic_gdf_short_range(
    orbital_basis, auxiliary_basis, system, len(ket_kpoints), options, budget,
)
source = manta.compute_periodic_gdf_short_range(
    orbital_basis, auxiliary_basis, system, q, ket_kpoints, options, budget,
)
# source.metric: complex128 (naux,naux)
# source.three_center: complex128 (nk,naux,nao,nao)
```

These example radii are consistency-fixture settings, not production accuracy
profiles. The source requires a nonsingular 3D cell, nonempty bases, positive
finite omega and radii, and finite Cartesian momenta. `q` is float64 `(3,)`;
`ket_kpoints` is float64 `(nk,3)`, in inverse bohr. Both arrays must be C- or
F-contiguous and remain unchanged during the synchronous call. The orbital
shell limit is l=6 and the auxiliary limit l=7, with the molecular fitting
provider's pure/Cartesian and split contraction-channel conventions.

For each auxiliary shell pair, the metric sums translated centers within the
physical auxiliary cutoff, with phase `exp(+i q.R)`. The three-center source
is `T(k-q,k) = sum_R,T exp(+i k.R-i q.T) (P_T|erfc|mu_0 nu_R)`.
The AO-center separation must lie within the pair cutoff. The translated
auxiliary center must lie within the auxiliary radius of the segment joining
the two AO centers, which contains every primitive product center. Enumeration
is centered on the actual shell offsets and remains valid on skew cells and
under atom-image relabeling. No k weights enter these integral blocks.

The raw finite erfc zero mode is included. In a complete jellium RSGDF
assembly the matching finite zero term must be removed exactly once, as in
[Ye and Berkelbach, JCP 154, 131104 (2021)](https://doi.org/10.1063/5.0046617).
Exchange Madelung or screened-zero restoration belongs to the selected
method. These blocks alone do not specify that method or its fitted projector.
The citation route is `numerics=["manta_periodic_gdf_short_range"]`.

Nonzero `integral_screen_error` uses the same density-independent Gaussian
shell-envelope omission rule as the existing GDF source. It controls the
absolute omission allowance of each raw integral entry, separate from
floating-point arithmetic, finite-image truncation and fitting errors. It is
part of the numerical source identity. No primitive precision cutoff or
unannounced fallback is introduced by MANTA.

The count-only plan bounds every image candidate, primitive product and
phase contraction before copying momenta or evaluating integrals. All five
resource caps apply jointly to both outputs. Memory includes output owners,
compensation, binding copies, one exported copy, MD workspace and metadata;
caller-retained objects use `budget.other_live_bytes`. Modeled bytes are not
process RSS. Evaluation uses one native thread, streams shell/image tiles,
reuses each integral across the ket momenta, and returns owned array copies.
`compute_metric=False` returns a `(0,0)` metric and preserves the identical
three-center source while reducing work and storage.

Receipts keep separate orbital/auxiliary basis identities, metric and
three-center source identities, the joint identity, conventions and actual
shell/primitive/phase counts. Metric identity is independent of orbital
basis, pair radius and ket batch. Resource caps and requested image-enumeration
limits do not change admitted source identity. The method that consumes these
raw components still owns its metric projection, density/Fock epochs and
energy ledger. No complete Fock object or symmetry certificate is inferred.
BIPOLE AC/BD supports, CRYSTAL parity, full MANTA RSGDF/RI J/K selection and
derivatives remain separately qualified stages.

The [short-range fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_fitting.md)
records independent reference errors, resource bounds, actual work and the
current performance relative to the existing primitive evaluator.

### Complete periodic Coulomb fitting sources

`plan_periodic_gdf` and `compute_periodic_gdf` assemble the finite Coulomb
auxiliary metric and shared-q coupling with MANTA SR and Fourier kernels.
They use `PeriodicGDFShortRangeOptions` for omega, the two physical image
radii, optional SR screening and metric selection. The numerical omega splits
the Coulomb interaction; it is not a physical range-separated functional.

```python
plan = manta.plan_periodic_gdf(
    orbital_basis, auxiliary_basis, system,
    len(reciprocal_vectors), len(ket_kpoints), options, budget,
)
source = manta.compute_periodic_gdf(
    orbital_basis, auxiliary_basis, system, q, ket_kpoints,
    reciprocal_vectors, options, budget,
)
# Same owned metric and three_center shapes as the raw SR source.
```

`reciprocal_vectors` is an explicit, nonempty float64 `(ng,3)` array of
Cartesian `G+q` vectors in inverse bohr, C- or F-contiguous. The provider
checks shifted reciprocal-lattice membership after resource admission. It
allows 5e-13 in fractional coordinates for the existing GDF domain generator's
12-decimal transfer labels, plus scaled floating-point roundoff. An
unshifted zero is refused for a resolvable small nonzero q. The
caller selects and converges this domain along with both real-space radii.
Ordering is retained in the source identity. There is no automatic accuracy
profile or reciprocal-domain generation in this entry point.

The auxiliary Fourier transform is `A_P(p)=integral chi_P(r) exp(-i p.r) dr`.
The pair transform sums `exp(+i k.R)` times the Fourier integral of
`chi_mu(r) chi_nu(r-R)` over the same physical pair-radius domain as SR.
For each nonzero p, the LR weight is
`4*pi/volume * exp(-p^2/(4*omega^2))/p^2`. The metric adds
`conj(A_P)*weight*A_Q`; the coupling adds `conj(A_P)*weight*pair(k)`.
The existing GDF jellium kernel determines the zero convention. At Gamma,
including reciprocal Gamma representatives, the source removes the finite
SR zero `pi/(volume*omega^2)` times the matching charge/overlap outer product
exactly once. It does so even when zero is omitted from the supplied list.
A zero vector in that list contributes zero to the LR sum. Exchange Madelung
corrections remain with the consuming method.

The plan jointly bounds SR, reciprocal panels, phases, contractions, inputs,
compensation and exports before copying numerical arrays. It conservatively
reserves a zero panel even for a nonzero transfer. Evaluation uses one native
thread and at most 16 reciprocal vectors per panel. One auxiliary panel and
one shell-pair panel folded into all ket momenta are retained; no full
AO-square by G transient is allocated. Each molecular FT tile is reused
across the k batch. All five resource caps and `other_live_bytes` apply.
Modeled peak bytes remain a conservative inventory, not process RSS.

The receipt embeds the raw `short_range` receipt and adds complete metric,
coupling and joint identities, `finite_zero_removed`, nonzero reciprocal
vector count, actual FT shell/primitive work, phase work and contraction
terms. Metric reuse is independent of the orbital basis, ket batch and pair
cutoff. Changing the reciprocal domain or numerical SR source changes its
identity. `compute_metric=False` preserves the coupling identity and returns
an empty metric.

These are complete finite Coulomb integral blocks, still unfactored. Finite
cutoffs can leave a metric indefinite; the caller must qualify the source,
retained subspace and fitting threshold before forming a fitted operator.
The method owns its density/Fock epochs, energy ledger and exchange seams.
Physical erf/erfc fitted kernels, BIPOLE AC/BD
parity and symmetry reduction are not admitted by this source API. The
citation route is `numerics=["manta_periodic_gdf"]`. The
[complete-source benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_gdf.md)
records the numerical and resource evidence.

### Streamed periodic fitting and transfer components

`PeriodicGDFTiles` snapshots the orbital and auxiliary bases, lattice, transfer,
ket momenta, reciprocal domain and options of the complete Coulomb source.
Its metric and coupling identities match `compute_periodic_gdf` on the same
inputs. `metric()` returns one owned `(naux,naux)` matrix; `evaluate(P,mu,nu)`
returns one owned `(nk,nP,nmu,nnu)` shell tile with AO offsets. It never stores
the complete auxiliary by AO-pair coupling tensor. `plan_periodic_gdf_tiles`
admits construction, one metric and one complete coupling traversal. Repeated
consumers must admit their own total work. At most 16 reciprocal vectors are
live per panel; shell tiling explicitly charges repeated Fourier evaluations.

`PeriodicRIQBuilder` constructs one fixed complex positive metric projector
and streams these tiles into small retained-mode blocks. It supplies a
Coulomb response at reciprocal Gamma and weighted exchange contributions for
one transfer q:

```python
tiles = manta.PeriodicGDFTiles(
    orbital_basis, auxiliary_basis, system, q, ket_kpoints,
    reciprocal_vectors, options, budget,
)
ri_options = manta.RIOptions()
ri_options.mode_block = 8
fit = manta.PeriodicRIQBuilder(tiles, spins=2, options=ri_options, budget=budget)
result = fit.build(spin_densities, source_weights, make_j=is_gamma, make_k=True)
```

The densities are finite Hermitian complex128 arrays of shape
`(spins,nk,nao,nao)`, with one or two channels. Positive semidefiniteness is
not required. Weights are nonnegative float64 `(nk,)` arrays. Both accept C
or F layout. J requires weights summing to one over the supplied mesh; a
partial K channel accepts a sum at most one. A one-channel density has exactly
the normalization supplied by the caller. Restricted spin factors, hybrid
coefficients, the complete exchange sum and singular corrections belong to
the consuming method.

The raw metric must be Hermitian to relative tolerance 1e-12. The lower
triangle, with real diagonal, defines the canonical Hermitian matrix M. For
its eigendecomposition, retain only eigenvalues strictly greater than
`max(absolute_cutoff, relative_cutoff*max(0,lambda_max))`. A negative eigenvalue
below `-max(absolute_cutoff, negative_relative_tolerance*max(0,lambda_max))`
refuses the finite source. There is no shift, absolute-eigenvalue repair or
density-dependent refit. The defaults are 1e-12, 1e-10 and 1e-10, respectively.
The owned `metric`, `factor` and `projector` exports give M,
`F=U_keep/sqrt(lambda_keep)` and `P=U_keep U_keep^H`. The factor is fixed for
the lifetime of the builder, including rank-zero fits.

With `B(k)=F^H T(k-q,k)`, the component equations are

```text
rho_L = sum_k,m,n w_k conj(B_L(k)[m,n]) sum_spin D_spin(k)[m,n]
J(k) = sum_L B_L(k) rho_L                      (q = reciprocal Gamma)
E_J = 0.5 sum_L |rho_L|^2
K_spin(k-q; k) = w_k sum_L B_L(k) D_spin(k) B_L(k)^H
```

J has shape `(nk,nao,nao)` and contains no target weight. K has shape
`(spins,nk,nao,nao)`, indexed by the source ket batch but expressed in each
**bra frame k-q**. It is one exchange contribution, so the caller must place
and sum it at the corresponding target before forming a Fock or exchange
energy. Unrequested J/K exports have a zero leading dimension. A
non-Hermitian assembled response is refused, never averaged into one.

Construction admits the dense metric, eigensolver, retained factor/projector
and every possible coupling pass before metric evaluation. A build keeps at
most `min(mode_block,rank)` fitted mode matrices, input/output spin matrices,
compensation and one raw shell tile. It never materializes full T or full B.
All five caps include construction and one build, using the maximum rank to
admit repeated passes. Work caps are not lifetime quotas. The caller's
`other_live_bytes` is charged once in the joint consumer. Rank-zero J still
visits raw tiles once to report the discarded-space residual; rank-zero
K-only executes no coupling pass.

Receipts separate fitted source/projector, evaluated density, weights,
snapshot and source/target frame identities. Frame identities include the
basis, lattice and actual momenta. Changing density does not change the
projector; changing only the mode-block size does not change the scientific
source. Constructor metric work and per-build coupling work are reported
separately. For J, `metric_residual_norm` is the norm of
`conj(M) conj(F) rho - b`, where
`b_P=sum_k,m,n w_k conj(T_P(k)[m,n]) D_total(k)[m,n]`. It diagnoses the chosen
finite metric equation, not RI basis error, cutoff convergence or SCF
stationarity. K-only marks this diagnostic unevaluated.

This component boundary preserves the shared Fock object's method-owned
energy ledger and evaluated-density epoch. It supplies no alternate solver
or convergence criterion. The complete Fock and explicit-source fitted UHF
consumers below supply the whole mesh exchange sum and the method gates.
The GDF sphere/capsule source
does not replace BIPOLE AC/BD supports or establish CRYSTAL parity. Symmetry
reduction additionally needs source, density and retained-projector transport
proofs; these components retain the explicit supplied mesh. The physical
kernel is Coulomb; numerical omega does not select physical erf/erfc fitting.

Use `numerics=["manta_periodic_gdf_tiles"]` or
`numerics=["manta_periodic_ri_q"]` for citations. The
[streamed fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_q.md)
records independent numerical checks, work bounds and timing limitations.

### Whole-mesh periodic RI Fock

`PeriodicRIFockBuilder` owns fixed projectors for a complete supplied mesh.
It constructs J from both spin densities and sums every source-k exchange
contribution at its target k:

```python
fit = manta.PeriodicRIFockBuilder(
    orbital_basis, auxiliary_basis, system, momenta, weights,
    gamma_vectors, pair_vectors, transfer_offsets,
    options=fitting_options, ri_options=ri_options, budget=budget,
)
evaluated = fit.build(hcore, spin_densities, exchange_scale=1.0)
```

`momenta` is float64 `(nk,3)` with distinct Bloch points modulo the reciprocal
lattice; nonnegative float64 `(nk,)` weights sum to one. This layer allows
nonuniform weights and general complex indefinite trial densities. HCORE is
Hermitian complex128 `(nk,nao,nao)` and D is `(2,nk,nao,nao)`. Arrays accept
C or F layout. The returned `J`, `K` and `values` are owned copies; `values`
contains `H + J - exchange_scale*K` in each spin. The scalar receipt contains
weighted one-electron, Coulomb, exchange and total electronic energy.
No nuclear energy, XC or singular correction is inferred from supplied H.

The constructor and planner also accept `make_j` and `make_k`, both defaulting
to `True`. At least one must be enabled. Selection is fixed at construction:

```python
only_j = manta.PeriodicRIFockBuilder(
    orbital_basis, auxiliary_basis, system, momenta, weights,
    gamma_vectors, np.empty((0, 3), dtype=np.float64),
    np.zeros(len(momenta)**2 + 1, dtype=np.int64),
    options=fitting_options, ri_options=ri_options, budget=budget,
    make_j=True, make_k=False,
)
```

J-only requires an empty pair-vector array and all-zero offsets; its planner
takes zero for `maximum_pair_vectors`. It retains one Gamma source/projector
and performs no exchange contractions or off-diagonal metric construction.
K-only retains Gamma for diagonal exchange and every off-diagonal transfer,
but performs no Coulomb contraction or metric-residual diagnostic. These
selectors do not change the k mesh, weights, density or AO frame.

`plan`, `info` and the build receipt expose `has_j` and `has_k`. Absent J is
an empty `(0,nao,nao)` array; absent K is `(0,nk,nao,nao)`. Its energy is zero.
The returned Fock contains H and the selected contributions. Without K,
`exchange_scale` is recorded as zero; all supplied scales must still be finite.
Conversely, zero scale on a builder that selects K still evaluates and exports
K. Target selection enters source and snapshot identities independently of
density and frame identities, including when a zero-rank fit returns zero.
These are component interfaces. The mixed exact/RI HF methods below provide
joint SCF admission. Fitted KS scientific and public job routes are described
below.

`transfer_offsets` is int64 `(nk*nk+1,)`, in row-major `(target,source)` order.
Diagonal segments are empty because the Gamma source supplies diagonal K.
When K is selected, every off-diagonal segment is nonempty and contains
**G+k_source-k_target**.
This is opposite the direct reciprocal exchange interface: conversion requires
negating each complete vector and removing the diagonal segments. The Gamma
list is inversion complete. Opposite transfer segments have opposite exact
reciprocal labels. Duplicate vectors, duplicate Bloch points and inconsistent
domains are refused before fitting.

Every raw metric retains its fixed positive projector under the preceding
RI rules. Reverse transfers must have equal ranks and conjugate metric,
projector and inverse, to relative maximum-entry tolerance 1e-10, using
`max(1,max_abs(M_q),max_abs(M_-q))` or the corresponding P/inverse scales.
Ill-conditioned inverses can fail even when raw metrics agree; no averaging,
refactor-on-density-change or negative-mode repair is used. Inspect source
cutoffs, conditioning and the explicitly selected rank before proceeding.
General q wrapping or symmetry reduction is not inferred from this check.

`plan_periodic_ri_fock` takes orbital/auxiliary bases, system, the Gamma vector
count, maximum pair-vector count, momentum count, fitting/RI options and budget.
It admits all retained sources, fixed metrics/projectors, one active shell/mode
workspace, spin accumulators and exports together. `info.metrics` lists Gamma
first, then ordered off-diagonal pairs. Constructor metric work and per-build
coupling work have separate counts. A build streams retained-mode blocks and
never stores complete T or B. With K it retains O(nk squared) source owners and
repeats the coupling traversal per mode block; this is bounded execution, not
a demonstrated large-mesh speedup or a solution to another backend's cache.
Each retained q leaf conservatively admits its joint J/K workspace even for
a single selected target. J-only removes all off-diagonal leaves from the
joint admission. The [target-selection benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_targets.md)
reports independent response/energy checks and measured work separately from
the modeled memory bound.

Source/frame/weight identities, every projector, evaluated density and the
complete Fock snapshot remain separately identified. Changing H or exchange
scale changes the operator source without relabeling the same density.
Use `numerics=["manta_periodic_ri_fock"]` for its citations.

### Physical finite-source fitted UHF

The explicit scientific entry point composes this fitted source with MANTA's
physical overlap, kinetic, nuclear attraction and nuclear Ewald producers:

```python
options = manta.PeriodicRIUHFOptions()
options.pair_cutoff_bohr = 6.0
options.fitting.pair_cutoff_bohr = 6.0
options.fitting.auxiliary_cutoff_bohr = 10.0
options.fitting.omega = 0.6
options.exchange_q0 = "BvK-ewald"  # or "strict-zero-mode"

result = manta.run_periodic_ri_uhf(
    orbital_basis, auxiliary_basis, system, mesh,
    reciprocal_vectors, pair_vectors, transfer_offsets, translations,
    options=options, integral_engine=engine,
)
```

Here `engine` is an explicitly selected `IntegralEngineOptions(backend="manta",
budget=budget)` with the usual error fallback and single channel batch.
The example cutoffs illustrate separate controls; they are not an accuracy
profile. `plan_periodic_ri_uhf` takes the two bases, system, regular mesh,
Gamma-vector count, maximum off-diagonal vector count, translation count,
options and budget. The native entry with the same positional numerical
arguments is available in `_vibeqc_core._manta` for consumers that own their
output/provenance admission.

The system must be neutral, all-electron and 3D. Multiplicity fixes the two
integer spin populations. The solver uses a uniform full `RegularKMesh`, with
at least one virtual in each spin; empty beta is allowed. Equal populations
recover the restricted limit. It uses the established bounded UHF HCORE
iteration, overlap admission, gap/population/density-closure/energy checks and
a fresh same-density terminal Fock confirmation. The evaluated D/F arrays and
their method energies remain separate from the returned canonical orbitals
and next projector, especially at an iteration cap.
`result.momenta` exports the actual generated Cartesian coordinates as an
owned array. Reuse them for bitwise frame identities; a separately computed
reciprocal inverse can differ in the last bits despite numerical equivalence.

The Gamma reciprocal list is primed (G=0 omitted), shared with the nuclear
source. The GDF finite SR-zero subtraction remains active once at Gamma,
independently of list membership. Numerical `fitting.omega` does not select
physical erf/erfc fitting. Nuclear `alpha` and its physical image cutoff are
separate. The explicit HF and fitting AO pair cutoffs must agree. Fitting
auxiliary images use the shared sphere/capsule selectors.
`translations` explicitly bounds the one-electron Bloch sums; GDF generates
its AO pair images within the fitting cutoff. Equal cutoff numbers alone do
not establish completeness of the supplied translations. Converge these
supports independently when assessing physical accuracy.

`strict-zero-mode` leaves singular exchange uncorrected. `BvK-ewald` applies
`-xi*S*D_spin*S` to each spin/k Fock with its separately admitted probe-charge
Ewald source. Its energy uses one target weight and the usual half trace,
with no extra source weight. It remains a method correction, separate from
the RI J/K source and uncorrected exchange energy.

The joint plan counts metric construction once and admits the maximum number
of solver builds plus confirmation, every retained projector, history and
simultaneous numerical owner. The inherited `primitive_pairs` field aggregates
one-electron primitive pairs and fitting primitive products; it is a work
count, not solely a two-primitive count. Selected Python output additionally
reserves scalar per-transfer provenance and does not mutate the caller budget.
These are conservative modeled resource bounds, not RSS measurements.

The standard output record includes fit controls and ranks, raw/canonical
metric and fixed-projector identities, spin populations, energy components,
evaluated density/Fock identities and convergence diagnostics. The citation
route is `method="manta_periodic_ri_uhf"`, plus `manta_periodic_madelung` when
that correction is selected. The [fitted Fock/UHF benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_uhf.md)
checks independent finite-source SCF and reports setup, repeated coupling work
and limitations.

These full-mesh snapshots preserve the unified Fock object's frame, evaluated
density, physical operator and method-energy contract. Shared symmetry
transport tests pass for AO/auxiliary image gauges and retained projectors;
they authorize no reduction. The GDF source preserves its own finite domains
and does not replace BIPOLE's exchange supports, cache or CRYSTAL parity gates.
Restricted fitted HF and mixed exact/RI HF are described below. Physical
range-separated fitting, accuracy profiles, symmetry reduction and
the remaining MANTA capability matrix still require their own increments.

### Fitted UHF from physical cutoffs

The native cutoff factory generates complete AO-pair image spheres and
certified reciprocal kinetic-energy spheres before invoking the same fitted
UHF method:

```python
plan = manta.plan_periodic_ri_uhf_from_cutoffs(
    orbital_basis, auxiliary_basis, system, mesh,
    reciprocal_energy_cutoff, options, budget,
)
domains = manta.build_periodic_ri_uhf_domains(
    orbital_basis, auxiliary_basis, system, mesh,
    reciprocal_energy_cutoff, options, budget,
)
result = manta.run_periodic_ri_uhf_from_cutoffs(
    orbital_basis, auxiliary_basis, system, mesh,
    reciprocal_energy_cutoff, options=options, integral_engine=engine,
)
```

The reciprocal cutoff is in hartree and selects `|G|^2/2` and
`|G+k_source-k_target|^2/2` spheres. The Gamma list omits zero. Off-diagonal
`domains.pair_vectors` segments use ket-minus-bra transfers; all diagonal
segments are empty. A nonempty sphere is required for every off-diagonal
pair. `transfer_offsets` is an owned int64 array of length `nk*nk+1` in
target/source order. The factory audits reciprocal conjugacy using integer
labels, including wrapped transfers on shifted meshes. It preserves the full
mesh and does not deduplicate metric sources or enable symmetry reduction.

`domains.translations` is the pair-complete AO image union for the explicit
`options.pair_cutoff_bohr`, which must equal `options.fitting.pair_cutoff_bohr`.
Independent lattice-image shifts of AO centers remain covered. GDF auxiliary
spheres/capsules, nuclear images, numerical fitting split and metric thresholds
retain their own options. The exported arrays are owned copies. The geometric
domain identity covers the lattice, mesh, shifts, cutoffs, AO centers and
generated supports; the method's fitted identity separately covers the
auxiliary basis, fit controls and projectors.

Planning admits candidate boxes, native/exported domain arrays, the actual
RI UHF leaf with all auxiliary/projector owners, repeated builds and terminal
confirmation. It does not use the direct UHF resource plan as an RI bound.
The `plan.uhf` field is a `PeriodicRIUHFPlan`; the outer plan additionally
accounts for domain generation. `exchange_candidates` and
`maximum_exchange_candidates` bound the off-diagonal fitting segments.
`receipt.conjugacy_vectors` counts the unique-transfer audit, including Gamma,
before replication across ordered pairs. Boxes can overestimate both memory
and work; fitting basis size and projector storage remain visible.

The selected run reserves its standard Python provenance in addition to the
native budget without mutating the caller's budget. Output records actual
domain counts, selection, cutoff and identity alongside the fixed fit,
evaluated density, physical Fock and method energy. Generated and equivalent
explicit-source runs preserve those scientific identities. This factory
guarantees finite-support membership and resource admission, not an accuracy
profile or cutoff convergence. See the
[cutoff RI UHF benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_domains.md)
for independent numerical comparisons and the conservative resource bounds.

### Selected periodic RI UHF jobs

`run_periodic_job` supports the fitted UHF cutoff source through explicit
RI choices for both J and K. Supply a resolved auxiliary `BasisSet`, the
usual physical integral cutoffs, and `PeriodicFittingOptions`:

```python
result = vq.run_periodic_job(
    system, orbital_basis, method="UHF", kpoints=mesh,
    integral_engine=manta.IntegralEngineOptions(backend="manta", budget=budget),
    coulomb=manta.CoulombBuildOptions(
        algorithm="ri", auxiliary_basis=auxiliary_basis, metric=manta.RIOptions(),
    ),
    exchange=manta.ExchangeBuildOptions(algorithm="ri"),
    periodic_integrals=manta.PeriodicIntegralOptions(
        reciprocal_energy_cutoff=6.0, pair_cutoff_bohr=3.0,
        nuclear_cutoff_bohr=4.0, alpha=0.6,
    ),
    periodic_fitting=manta.PeriodicFittingOptions(
        omega=0.65, auxiliary_cutoff_bohr=8.0,
    ),
    exchange_exxdiv="ewald", use_diis=False, initial_guess="HCORE",
    max_iter=100, conv_tol_energy=1e-12, output="periodic_ri_uhf",
)
```

The example defines a finite source, not an accuracy profile. `omega` is a
numerical inverse-bohr GDF splitting parameter, independent of nuclear
`alpha`; it does not request physical range-separated exchange.
`auxiliary_cutoff_bohr` sets the fitting image radius.
`integral_screen_error=0.0` and `image_candidate_cap=1000000` are additional
explicit fitting controls. The immutable `PeriodicFittingOptions` object is
exported from both `vibeqc` and `vibeqc.manta`.
The AO pair radius comes from `PeriodicIntegralOptions` and is used for both
the one-electron and fitting sources. The `coulomb.metric` RIOptions controls
absolute/relative projection thresholds, negative-eigenvalue tolerance and
streamed mode-block size. Significant negative metrics are refused; no
automatic shift or auxiliary-basis substitution is applied.

Both bases must be resolved all-electron `BasisSet` objects. Auxiliary names
are not resolved inside this bounded job adapter. Omitting `exchange` retains
its exact default and selects the mixed RI-J/exact-K profile described below;
RI J never silently selects RI K. Restricted fitted HF is also admitted as
described below, followed by the fitted KS job profile. The source is
neutral and 3D, with fixed integer
spin populations from multiplicity, virtual space in each spin, HCORE/AUTO,
uniform full regular meshes and no DIIS or smearing. Supported regular mesh
input forms and output options follow the exact UHF job contract.

Admission includes every generated domain, retained auxiliary/metric/projector
owner, repeated coupling build, scalar history, terminal confirmation and
Python/output serialization. The job additionally reserves auxiliary metadata
and per-transfer evidence without modifying the caller's budget. A dry run
performs count-only admission and writes the usual manifest without solving.
These conservative bounds are not measured RSS.

The result is a `PeriodicMANTAUHFResult`: use its separate alpha/beta density,
Fock, orbital and occupation properties. Its provenance retains the selected
algorithms, auxiliary basis, fitting controls/ranks/projectors, source/frame/
density identities, method energy components and spin history. Standard
output, citations, structured logs and QVF use `vibeqc.output`. QVF preserves
the evaluated spin densities even when the SCF hits its iteration cap; the
returned canonical orbitals may describe a different next projector. AO
lattice-image phases are transported to QVF's nuclear-center basis and the
original source-frame shifts are recorded.

Full-mesh Fock and method-energy ownership remain compatible with the shared
operator boundary. Neither the job selection nor a converged SCF is evidence
of BIPOLE/CRYSTAL parity or authorization for symmetry reduction. See the
[public RI UHF benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_job.md)
for independent SCF and saved-density checks with complete output timings.


### Selected periodic RI RHF

The same finite fitted source now supports closed-shell singlets through a
restricted numerical leaf. Set `method="RHF"` in the public RI job example
above, keeping explicit RI selections for both J and K and the resolved
auxiliary basis. The result is a `PeriodicMANTAResult` with spin-summed
`density_k`, restricted `fock_k`, `mo_coeffs_k`, `mo_energies_k` and occupations
two/zero. Neutral all-electron 3D cells, a uniform full mesh, HCORE/AUTO,
integer occupations, retained virtual space and the existing no-DIIS/no-smearing
limits apply. RHF requires multiplicity one and an even electron count.

The scientific entry points are:

```python
options = manta.PeriodicRIRHFOptions()
options.pair_cutoff_bohr = options.fitting.pair_cutoff_bohr = 3.0
options.fitting.auxiliary_cutoff_bohr = 8.0
options.fitting.omega = 0.65
options.exchange_q0 = "BvK-ewald"

plan = manta.plan_periodic_ri_rhf_from_cutoffs(
    orbital_basis, auxiliary_basis, system, mesh, 6.0, options, budget,
)
result = manta.run_periodic_ri_rhf_from_cutoffs(
    orbital_basis, auxiliary_basis, system, mesh, 6.0,
    options=options, integral_engine=engine,
)
```

These values define an example finite source, not recommended accuracy
settings. `plan.rhf` is a `PeriodicRIRHFPlan` with the actual restricted solver
reservation. `build_periodic_ri_rhf_domains` returns the same geometric
schema and ket-minus-bra convention as the fitted UHF factory. Explicit
arrays are accepted by `run_periodic_ri_rhf` with the same argument names as
`run_periodic_ri_uhf`; `plan_periodic_ri_rhf` provides count-only admission.
All auxiliary/projector owners and the temporary equal-spin embedding remain
in the aggregate resource plan even though the solver stores only one D/F.

The restricted callback supplies `D_alpha=D_beta=D/2` to the shared immutable
fitted Fock producer. It checks equal spin responses before exporting the
restricted Fock. RHF keeps its own spin-summed commutator, density closure,
metric idempotency, electron-count, band-gap and terminal-confirmation gates.
It does not select an unrestricted stopping rule. The method's BvK correction
is `-xi*S*D*S/2`, and its energy is the weighted half trace with the same D.
Metric projectors remain fixed across all iterations and confirmation.

Source/frame identities remain separate from the evaluated density and
physical Fock snapshot. The record names `operator_layout="k,ao,ao"` and
`spin_labels=["restricted-total"]`; fitting provenance records the equal-spin
embedding explicitly. The citation route is `manta_periodic_ri_rhf`. QVF
preserves equal halves of the evaluated restricted density, including
iteration-capped results and the nuclear-center image-gauge conversion.
It does not replace that density with the next canonical orbital projector.

Independent full-rank and truncated-overlap tests also cross the landed
FockBuild API using restricted channels and rectangular S/X frames. Native
bounded storage and method-energy ownership remain intact. These tests do
not authorize a source-specific symmetry reduction or replace BIPOLE's
CRYSTAL parity qualification. The
[restricted RI HF benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_rhf.md)
reports finite-source accuracy and complete native/public costs.

### Mixed exact/RI periodic RHF and UHF

The same restricted and unrestricted method leaves accept independently
selected J and K sources. In the public example above, choose either
`coulomb.algorithm="ri", exchange.algorithm="exact"` or
`coulomb.algorithm="exact", exchange.algorithm="ri"`. Both combinations
require the resolved auxiliary basis and `RIOptions` in `CoulombBuildOptions`,
including when only K is fitted, together with `PeriodicFittingOptions`.
Omitting `exchange` preserves its exact default. Setting both algorithms to
`"ri"` retains the complete fitted profile; setting both to `"exact"` uses
the existing direct profile without fitting controls.

The scientific and native RI-containing HF interfaces select the same
combinations through two additional option fields:

```python
options = manta.PeriodicRIRHFOptions()  # PeriodicRIUHFOptions for UHF
options.coulomb_algorithm = "ri"
options.exchange_algorithm = "exact"
options.pair_cutoff_bohr = options.fitting.pair_cutoff_bohr = 3.0
options.fitting.auxiliary_cutoff_bohr = 8.0
options.fitting.omega = 0.65
options.exchange_q0 = "BvK-ewald"

result = manta.run_periodic_ri_rhf_from_cutoffs(
    orbital_basis, auxiliary_basis, system, mesh, 6.0,
    options=options, integral_engine=engine,
)
```

Both native selectors default to `"ri"`; at least one RI component is
required by these RI-containing interfaces. The example is a finite source,
not an accuracy profile. Neutral all-electron 3D cells, uniform full regular
meshes, integer occupations, retained virtual space, HCORE/AUTO and the
existing no-DIIS/no-smearing method restrictions still apply.

The selected **exchange** source determines the pair-vector convention:

| Selected K | Explicit pair vectors | Diagonal segments | Retained fitting sources |
| --- | --- | --- | --- |
| RI | `G+k_source-k_target` | Empty; Gamma owns diagonal fitting | Gamma and every ordered off-diagonal transfer |
| Exact | `G+k_target-k_source` | Nonempty primed G lists | Gamma only, for RI J |

The cutoff factories generate the requested convention and record it in
`domain_selection`. Exact off-diagonal segments may be empty; every fitted
off-diagonal segment must be nonempty. Exact K omits Q=0. Explicit-source
callers must provide the complete matching support and offsets; changing a
selector does not convert an already supplied domain. The direct component
uses the stated finite reciprocal and AO image sums. The fitted component
retains its own SR/LR source, auxiliary images and fixed positive projectors.
Equal cutoff values do not establish equality of these two finite sources.

Each callback evaluates both components at the same spin density, includes
HCORE once, then applies the separately selected BvK correction once. RHF
retains its equal-spin embedding and restricted convergence gates. UHF
retains separate alpha/beta occupations and gates. The method energy and
fresh terminal confirmation use that same composed Fock and density.

The joint plan admits both sources before generation or numerical copies.
`plan.fock` describes only the selected RI component. `exact_hartree` or
`exact_exchange` describes the complementary direct producer, while
`two_electron_peak_bytes` includes fitting, direct and composition workspace.
Aggregate memory and work also include every repeated SCF build, terminal
confirmation, domain generation and selected output owners. These are
conservative modeled bounds, not measured RSS.

Output records `finite-3d-mixed-ri-rhf` or `finite-3d-mixed-ri-uhf`, separate
J/K algorithms and source identities, fixed fitting evidence and the direct
component receipt. Source, AO frame and evaluated density identities remain
distinct. The existing RI HF citation route additionally cites the direct
Hartree or exchange producer. QVF preserves the evaluated density, including
capped results and nuclear-center image transport.

Owned export tests cross the shared FockBuild API with the complete mixed
method energy and unchanged physical action. They add no dense snapshot
copies inside the bounded solver. Each component retains its source contract;
this composition neither substitutes BIPOLE AC/BD domains nor establishes
CRYSTAL parity or a symmetry-reduction certificate. The
[mixed HF benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_mixed_hf.md)
records independent references and complete native/public costs. Explicit
fitted KS is described next; physical range fitting and the remaining
capability matrix stay open.

### Fitted periodic RKS and UKS

`PeriodicRIKSOptions` and `PeriodicRIGlobalHybridKSOptions` now qualify explicit
finite sources for pure LDA/GGA and full-Coulomb global hybrids. The functional
spin selects the actual bounded RKS or UKS leaf. The neutral all-electron 3D,
full regular mesh, integer occupation, virtual-space and no-DIIS restrictions
of direct KS still apply. Physical range-separated fitting has its separate
explicit interface below. Meta-GGA, VV10 and external functionals remain
required increments. Cutoff-domain, joint
quadrature and public `run_periodic_job` routes are available for the qualified
pure and global fitted methods.
The fitted option types cannot be passed to direct KS entry points, which
would otherwise discard the requested fitting source.

Both option types contain `ks=PeriodicKSOptions()`,
`fitting=PeriodicGDFShortRangeOptions()` and `ri=RIOptions()`. Constructors
start with matching `ks.pair_cutoff_bohr` and `fitting.pair_cutoff_bohr`; callers
must preserve their equality and specify a finite pair radius. Numerical
`fitting.omega` defines the existing Coulomb SR/LR fit, not a functional's
physical exchange range. Auxiliary support, screening and positive-projector
thresholds are part of that source. XC images and quadrature remain distinct
explicit inputs; matching pair radii do not make direct and fitted finite
Hamiltonians identical.

Pure KS uses RI J and no exact exchange. For example, with explicitly prepared
finite sources:

```python
options = manta.PeriodicRIKSOptions()
options.ks.pair_cutoff_bohr = options.fitting.pair_cutoff_bohr = 3.0
options.fitting.auxiliary_cutoff_bohr = 8.0
options.fitting.omega = 0.65
engine = manta.IntegralEngineOptions(backend="manta", budget=budget)

plan = manta.plan_periodic_ri_ks(
    orbital_basis, auxiliary_basis, system, mesh, functional,
    len(reciprocal_vectors), len(translations), len(xc_images), len(points),
    options, budget,
)
result = manta.run_periodic_ri_ks(
    orbital_basis, auxiliary_basis, system, mesh, functional,
    reciprocal_vectors, translations, xc_images, points, quadrature_weights,
    options=options, integral_engine=engine,
)
```

These settings illustrate a finite source, not a recommended accuracy profile.
The selected wrapper includes Python provenance admission and records through
`vibeqc.output`. The count-only native plan excludes that additional wrapper
reservation. Arrays follow the direct KS shape and dtype rules.

Global options also expose independent `coulomb_algorithm` and
`exchange_algorithm` fields, each `"exact"` or `"ri"`, defaulting to both RI.
At least one must be RI. `plan_periodic_ri_global_hybrid_ks` inserts the maximum
exchange-vector count per pair after the reciprocal-vector count.
`run_periodic_ri_global_hybrid_ks` inserts `exchange_vectors, transfer_offsets`
after `reciprocal_vectors`. Both otherwise use the pure fitted signatures.
The selected K determines the same opposite transfer orientations and diagonal
supports as the mixed HF table above. Supplied domains are never converted
when changing a selector. The method retains `exchange_q0`, `madelung_alpha`
and `madelung_tolerance`; its resolved global fraction scales K and its
separate correction exactly once.

Every callback evaluates the complete functional on the actual density:
`E = E_nuclear + E_one_electron + E_J + E_XC + E_K + E_correction`.
RKS supplies equal halves of its total density to the shared fitted producer
but supplies the total density to spin-unpolarized XC. UKS supplies actual
alpha/beta densities to both. The functional solver receives this joint
energy, not an HF half trace. Its physical Fock contains HCORE once. Retained
metric projectors stay fixed through all iterations and terminal confirmation.

`PeriodicKSPlan.fitting` is the selected fitting subplan; it is absent for
direct KS. The full method plan includes retained projectors, streamed fitting
workspace, direct components, XC panels, spin embedding, history and every
allowed build. All five resource caps are admitted before numerical copies.
`hartree` is active only with exact J; `exchange` only with direct global K.
Receipts expose J/K algorithms and component identities, `fitting_info` and
`fitted_fock`. The latter is explicitly a component receipt with embedded
spin densities; its electronic energy omits XC and any complementary direct
term. Only the method receipt is the complete KS energy.

Evaluated-density identity version 2 now binds the actual D, spin layout,
k weights and overlap frame, independently of J/K or grid selection. The
paired Fock snapshot also binds its interaction, one-electron and nuclear
sources. This applies to existing direct KS too: opaque density/snapshot IDs
change, while saved direct numerical arrays, energies, source identities and
original resource counts remain bitwise unchanged.

Independent Gaussian/libint fitting, quadrature and SciPy SCF references
qualify pure LDA/PBE/BLYP and global PBE0/B3LYP, including capped epochs,
truncated/empty projectors and rectangular overlap spaces. Exactly empty PBE,
PBE0 and B3LYP minority potentials retain the documented libxc endpoint
conditioning limitation; occupied-spin matrices and energies are checked
strictly, and one-epsilon input sensitivity is measured separately. The
empty-spin B88/LYP tests retain the full matrix gate. No XC regularization or
uniform endpoint accuracy claim is added.

Owned exports cross the existing `FockBuild` and `OperatorFrame` interfaces
with `kind="ks"`, the method energy, evaluated density and explicit S/X frame.
Independently shifted AO and auxiliary images also pass the shared
`BlockSpaceAction` density-push/operator-pull contract. These checks add no
generic dense snapshots inside the bounded solver. They preserve BIPOLE's
distinct AC/BD source and CRYSTAL parity gates; no spatial or time-reversal
reduction is inferred. The
[fitted KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_ks.md)
reports independent errors and native/selected scientific costs. Citation
routes are `manta_periodic_ri_ks` and `manta_periodic_ri_global_hybrid_ks`,
together with the functional and fitting numerics routes.


#### Fitted KS cutoff and quadrature factories

`plan_periodic_ri_ks_from_cutoffs` and `build_periodic_ri_ks_domains` accept
`basis, auxiliary, system, mesh, functional, reciprocal_energy_cutoff,
xc_image_cutoff_bohr, grid_points, options, budget`. Their hybrid counterparts
replace `ri_ks` with `ri_global_hybrid_ks` and take the matching hybrid options.
The count-only plan admits every source traversal and the complete functional
SCF before numerical enumeration. These calls return owned source arrays.

The selected `run_periodic_ri_ks_from_cutoffs` replaces `grid_points` with
`points, quadrature_weights` and takes keyword-only `options` and
`integral_engine`. Its global hybrid counterpart has the same signature.
All source factories retain pair-complete orbital translations, certified
reciprocal kinetic-energy spheres and an independently bounded canonical XC
image sphere. RI J alone produces no K-transfer table. Exact K uses
`G+k_target-k_source` and primed diagonal spheres; RI K uses
`G+k_source-k_target` and empty pair diagonals supplied by the Gamma fitting
owner. Changing K selection regenerates the appropriate domain.

For an atom-centred quadrature and a single joint resource admission:

```python
grid = manta.PeriodicQuadratureOptions()
grid.radial_points = 12
grid.lebedev_order = 17
grid.partition_radius_bohr = 8.0
plan = manta.plan_periodic_ri_ks_with_quadrature(
    orbital_basis, auxiliary_basis, system, mesh, functional,
    6.0, 11.0, grid, options, budget,
)
result = manta.run_periodic_ri_ks_with_quadrature(
    orbital_basis, auxiliary_basis, system, mesh, functional,
    6.0, 11.0, grid, options=options, integral_engine=engine,
)
```

The numbers are finite-source examples, not an accuracy recommendation.
Global hybrids use `plan_periodic_ri_global_hybrid_ks_with_quadrature` and
`run_periodic_ri_global_hybrid_ks_with_quadrature` with their matching options.
The joint plan adds retained quadrature and domain owners to the actual
fitted KS plan, including projectors, direct components, XC, all iterations
and confirmation. A native plan excludes the selected wrapper's additional
Python output reservation. All five limits cover the complete admitted work.

Pure fitted factories use the general `PeriodicKSDomainPlan`,
`PeriodicKSDomains` and `PeriodicKSGridPlan` schemas with an active nested
`ks.fitting` plan. Globals use `PeriodicRIGlobalHybridKSDomainPlan`,
`PeriodicRIGlobalHybridKSDomains` and `PeriodicRIGlobalHybridKSGridPlan`.
Geometry identity records the enumerated supports independently of auxiliary
projectors and the complete physical source. The hybrid `pair_identity`
shares the corresponding HF geometry contract. These identifiers do not
certify BIPOLE parity, symmetry of the grid or permission to reduce the mesh.

Generated and explicit sources agree bitwise for the same finite supports.
Independent integer membership, raw fitting/functional SCF, empty-spin,
rank, image-gauge, bounded-admission and shared Fock checks are documented in
the [generated fitted KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_ks_domains.md).
These scientific wrappers record through `vibeqc.output` using the existing
fitted KS and quadrature citation routes. The public job profile below adds
standard artifacts and evaluated-density QVF output.


### Selected periodic fitted KS jobs

`run_periodic_job` supports the same qualified fitted pure/global RKS and UKS
sources. Select RI J or RI K explicitly, supply a resolved all-electron
auxiliary `BasisSet` and `RIOptions` through `coulomb`, and provide
`PeriodicFittingOptions`. Pure KS requires RI J and no K. Globals accept
RI-J/exact-K, exact-J/RI-K or both fitted. Omitting `exchange` selects no K
for pure KS and exact K for globals; RI J never implies RI K.

```python
result = vq.run_periodic_job(
    system, orbital_basis, method="UKS", functional="pbe0", kpoints=(2, 1, 1),
    integral_engine=manta.IntegralEngineOptions(backend="manta", budget=budget),
    coulomb=manta.CoulombBuildOptions(
        algorithm="ri", auxiliary_basis=auxiliary_basis, metric=manta.RIOptions(),
    ),
    exchange=manta.ExchangeBuildOptions(algorithm="ri"),
    periodic_integrals=manta.PeriodicIntegralOptions(
        reciprocal_energy_cutoff=6.0, pair_cutoff_bohr=3.0,
        nuclear_cutoff_bohr=4.0, alpha=0.6,
    ),
    periodic_fitting=manta.PeriodicFittingOptions(
        omega=0.65, auxiliary_cutoff_bohr=8.0,
    ),
    periodic_xc=manta.PeriodicXCOptions(
        xc_image_cutoff_bohr=11.0, radial_points=12, lebedev_order=17,
        partition_radius_bohr=8.0,
    ),
    use_diis=False, output="fitted-ks", record_hostname=False,
)
```

These controls illustrate a finite source. Refine fitting, physical domains
and quadrature independently; SCF convergence does not establish their
accuracy. Neutral all-electron 3D cells, full uniform regular meshes, integer
spin occupations, a virtual space per spin and HCORE/AUTO initialization remain
required. Range-separated fitting, meta-GGA, nonlocal XC, unsupported symmetry
requests and BIPOLE-specific options are refused before source generation.
The numerical fitting omega does not select a physical exchange kernel.

Joint admission covers the native grid/domain/functional solver, retained
fitting sources, Python evidence, histories, basis metadata, QVF conversion
and output owners. A dry run admits that same work and emits only its plan
manifest. Insufficient caps refuse before numerical generation or computed
artifacts. The caller's budget is unchanged.

The result is the existing `PeriodicMANTARKSResult` or
`PeriodicMANTAUKSResult`, with owned method state and complete functional
energy. Output records selected J/K algorithms, fitting/projector, grid,
geometry and physical-source identities separately from the evaluated density
and AO frame. Only the complete method energy includes XC and complementary
direct components. Capped states retain their actual evaluated density;
QVF stores RKS equal spin halves or UKS alpha/beta densities in the nuclear
AO frame, including independent orbital-image shifts. Output delivery failure
preserves the computed method status while reporting the artifact failure.

The [public fitted KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_ks_job.md)
compares complete public jobs against native generated sources and independent
functional SCF, including lossless evaluated-density QVF. Shared Fock and
symmetry transport checks preserve method energy and S/X frames. They do not
qualify a reduced mesh or BIPOLE/CRYSTAL source parity.

### Physical range kernels for finite fitting sources

`compute_periodic_gdf`, `PeriodicGDFTiles` and their count-only planners accept
an optional `physical_kernel=PeriodicGDFKernelOptions()` argument. Its default
is Coulomb with zero physical omega, preserving the existing source and plans.
For a screened source, select the interaction explicitly:

```python
physical = manta.PeriodicGDFKernelOptions()
physical.kernel = "erfc"  # also "erf"
physical.omega = 0.43
source = manta.PeriodicGDFTiles(
    orbital, auxiliary, system, q, ket_kpoints, reciprocal_vectors,
    options=fitting_options, budget=budget, physical_kernel=physical,
)
```

`physical.omega` is the positive inverse-length parameter in the physical
interaction. `fitting_options.omega` remains a numerical split. For `erfc`,
the source combines real-space `erfc(eta*r)/r` with the reciprocal difference
of `erf(eta*r)/r` and `erf(omega*r)/r`. At reciprocal Gamma, it removes the
finite numerical zero and restores the physical coefficient
`pi/(volume*omega**2)` exactly once, whether the supplied vector list contains
zero or not. For `erf`, only the primed reciprocal sum is evaluated; no
real-space short-range work or finite-zero subtraction is performed.
The Coulomb and `erf` poles remain omitted. No exchange Madelung correction
belongs to these component sources.

The source receipt distinguishes `zero_transfer`, `finite_zero_removed`,
`physical_zero_included`, `real_short_range` and `physical_omega`. A
`PeriodicRIQBuilder` retains the actual physical metric and fixed projector;
its receipt records the physical kernel and range. Gamma admission is based
on the transfer, independently of finite-zero subtraction. Interaction source
identities bind the physical range while AO frames and evaluated-density
identities retain their existing meaning. Changing unused numerical split or
auxiliary-image controls does not relabel an `erf` source.

The existing conservative plans reserve the short-range workspace and work
bounds even for Fourier-only `erf`. Reported executed short-range calls are
zero in that case. This extension does not claim tighter admission or an
automatic accuracy target. Refine reciprocal, AO-pair and auxiliary supports
independently.

These are finite fitting sources and fixed-q components. Complete bounded
explicit, generated and public range KS are described below. Derivative and
symmetry qualification remain required. The [range-fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_range_fitting.md)
includes independent integral checks and a complete finite Gamma SCF diagnostic
composed outside the production bounded driver. It does not enable those
method routes or establish BIPOLE/CRYSTAL parity.

The whole-mesh `PeriodicRIFockBuilder` and `plan_periodic_ri_fock` also accept
`exchange_kernel=physical`, using the same `PeriodicGDFKernelOptions`. A
non-Coulomb kernel requires explicit `make_j=False, make_k=True`; its fitted
metric cannot also supply the ordinary bare-Coulomb Hartree contribution.
Construct that J source separately. The default Coulomb interface is unchanged.

The Gamma source owns all diagonal K terms, including the finite erfc zero.
Off-diagonal segments retain `G+k_source-k_target`, empty diagonal segments,
matched reverse supports and the existing metric/projector reversal audits.
The selected kernel and physical omega are copied into the immutable source
and exported in `info` and each result receipt. Density/frame identities stay
independent of that interaction selection. No Madelung term is added.

The [whole-mesh range-fitting benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_range_fock.md)
checks nonuniform and zero weights, fixed and truncated projectors, variational
energies, shared image transport and complete shifted two-k SCF diagnostics.
The diagnostic composition is outside the bounded method driver; the separate
explicit method below qualifies joint admission and complete method assembly.

### Bounded explicit fitted range-separated KS

`PeriodicRIRangeSeparatedKSOptions` and `plan_periodic_ri_range_separated_ks`
admit complete finite RKS/UKS with physical range exchange. Options contain
`ks`, `fitting`, `ri`, `coulomb_algorithm`, `exchange_algorithm`, `exchange_q0`
and the existing Madelung controls. Choose `"exact"` or `"ri"` independently
for J and K, with at least one RI target. The physical AO-pair and fitting pair
cutoffs must agree. Neutral all-electron 3D, full regular mesh, fixed spin
occupations, at least one virtual orbital and the existing bounded KS solver
restrictions apply.

The planner accepts `basis, auxiliary, system, mesh, functional`, counts for
`reciprocal_vectors, exchange_vectors_per_pair, cells, xc_images, grid_points`,
then `options, budget`. The selected `run_periodic_ri_range_separated_ks` takes
the same first five objects followed by arrays `reciprocal_vectors,
exchange_vectors, transfer_offsets, translations, xc_images, points,
quadrature_weights`, with keyword `options` and
`integral_engine=IntegralEngineOptions(backend="manta", budget=budget)`.
Generated cutoff and quadrature factories for this route are described below.

RI K uses `G+k_source-k_target` and empty diagonal segments; its Gamma source
handles every diagonal term. Exact K uses `G+k_target-k_source` with primed
diagonal supports; the method adds finite erfc zeros there. Ordinary J always
retains a bare-Coulomb source. A screened exchange metric never replaces it.
The functional resolves to canonical Coulomb and distinct erfc components,
as in direct range KS. Every fitted kernel keeps a separate fixed projector.
Signed coefficients multiply these responses after fitting. Fitting erf
directly is not presumed equal to subtracting two separately fitted kernels.
The physical range comes from the functional; `fitting.omega` remains the
numerical source split.

The complete energy is nuclear + one electron + J + semilocal XC + exchange.
Each exchange coefficient and any long-range Madelung correction applies once.
Screened-only HSE06 has no singular correction; its finite erfc zero remains
part of the source. Every retained metric/projector, setup, workspace, solver
call and final confirmation participates in aggregate admission. Python
provenance is reserved before copying input buffers.

`plan.exchange_components` exposes optional `fitting` plans. Each evaluated
component receipt exposes optional `fitting_info` and `fitted_fock` for RI K,
or the existing `exchange` evidence for exact K. The selected output records
the actual source, physical range, projector identities, density embedding,
method-scaled energies and source/frame/evaluated-density identities. Fitted
receipts are components, not another total functional energy. The citation
route is `manta_periodic_ri_range_separated_ks` plus the selected functional.

The [bounded range KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_range_ks.md)
compares HSE06, CAM-B3LYP and mixed-range RKS/UKS with independent complete
finite-source SCF references. It also verifies shared Fock state and symmetry
transport without enabling reduction. Domain/grid convergence, external
ORCA/CRYSTAL parity and the remaining capability
matrix are still open.

### Generated fitted range-separated KS sources

The same method is available with certified reciprocal spheres, physical pair
images and canonical XC images through
`plan_periodic_ri_range_separated_ks_from_cutoffs`,
`build_periodic_ri_range_separated_ks_domains` and
`run_periodic_ri_range_separated_ks_from_cutoffs`. The first two accept
`basis, auxiliary, system, mesh, functional, reciprocal_energy_cutoff,
xc_image_cutoff_bohr, grid_points, options, budget`. The native runner replaces
`grid_points` with caller-supplied `points, quadrature_weights`. The selected
runner replaces `budget` with keyword `integral_engine` and records scientific
provenance through the standard output layer.

`plan_periodic_ri_range_separated_ks_with_quadrature` and the corresponding
`run_` function replace the explicit grid with `PeriodicQuadratureOptions`.
For example, after constructing the orbital and auxiliary bases, cell,
regular mesh and functional:

```python
options = manta.PeriodicRIRangeSeparatedKSOptions()
options.coulomb_algorithm = "ri"
options.exchange_algorithm = "ri"
options.ks.pair_cutoff_bohr = pair_radius
options.fitting.pair_cutoff_bohr = pair_radius
options.fitting.omega = numerical_split
options.fitting.auxiliary_cutoff_bohr = auxiliary_radius

grid = manta.PeriodicQuadratureOptions()
grid.radial_points = radial_points
grid.lebedev_order = lebedev_order
grid.partition_radius_bohr = partition_radius

result = manta.run_periodic_ri_range_separated_ks_with_quadrature(
    basis, auxiliary, system, mesh, functional, reciprocal_cutoff,
    xc_image_radius, grid, options=options,
    integral_engine=manta.IntegralEngineOptions(backend="manta", budget=budget),
)
```

Controls remain explicit finite-source choices. The domain receipt identifies
geometry independently of physical exchange kernels and fitting projectors.
RI K has empty diagonal pair segments and ket-minus-bra off-diagonal vectors;
exact K has primed diagonal segments and target-minus-source vectors. The
geometry never inserts a physical erfc zero. Its physical source and method
retain the once-only zero and singular-correction ownership described above.

The distinct `PeriodicRIRangeSeparatedKSDomainPlan`, `DomainReceipt`, `Domains`
and `GridPlan` types expose generated supports and the complete nested method
plan. Admission includes the retained quadrature and domain payloads, every
kernel/projector, source traversal, SCF iteration, terminal confirmation and
selected Python output. A rejected budget precedes numerical generation or
explicit grid copies. Exported arrays own their storage.

The [generated range KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_range_ks_domains.md)
compares complete generated, explicit and selected states with independent
finite-source SCF references. Shared Fock states and complex AO/auxiliary image
transport preserve the existing contracts. This does not establish an automatic
accuracy profile, ORCA/CRYSTAL or BIPOLE parity, or symmetry reduction.
The public job profile below adds evaluated-state output and job admission.

### Selected periodic fitted range-KS jobs

`run_periodic_job` supports the same finite fitted range-separated RKS/UKS
methods using explicit `coulomb`, `exchange`, `periodic_fitting`,
`periodic_integrals` and `periodic_xc` controls. Use the fitted KS job example
above with a supported range-separated functional such as `"hse06"` or
`"cam-b3lyp"`. Select `algorithm="ri"` independently for J and K, or combine
one fitted component with `algorithm="exact"` for the other. The resolved
auxiliary basis and `RIOptions` are supplied through `CoulombBuildOptions`,
including K-only fitting. An omitted `exchange` still selects exact K.

The functional determines physical exchange kernels and ranges. Numerical
`PeriodicFittingOptions.omega` does not select or override them. Each physical
K kernel has a separate fixed metric/projector; J retains bare Coulomb.
`exchange_exxdiv="ewald"` applies the method-owned BvK correction only to
nonzero long-range exchange. HSE06 reports that correction as inactive while
retaining its finite screened zero. The complete functional and physical
component checks run before any calculated output. Meta-GGA, incomplete
nonlocal functionals, RI-BUPO, BIPOLE controls and symmetry reduction remain
outside this finite profile.

The returned `PeriodicMANTARKSResult` or `PeriodicMANTAUKSResult`, structured
log and QVF retain the evaluated physical density, complete energy, source
frame and SCF epoch. This applies to iteration-limit exits as well as
converged states. RKS QVF stores equal alpha/beta halves; UKS retains both
evaluated spin densities, including empty beta. AO images are transported
to QVF's nuclear-centre frame without changing the physical source.

`run_fields.fitting_source_identities` and `fitting_frame_identities` list the
J fit, if present, followed by the resolved physical K fits. The ordered
`exchange_source_identities` list contains the actual exact or fitted physical
K sources. Detailed `integral_provenance` retains each kernel, coefficient,
physical range, projector and evaluated-density receipt. There is no synthetic
single fitting identity for a method with several independently fitted kernels.

Job admission reserves basis and numerical exports, all component evidence,
history, QVF conversion and serialized output in addition to native grid,
domain, source and solver allocations. A one-short budget refuses before
numerical generation and calculated artifacts; dry runs write only the
admitted manifest. Output delivery follows the normal writer lifecycle.

The [public fitted range-KS benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_ri_range_ks_job.md)
compares full public jobs with native states and independent finite-source SCF
references. It validates QVF schema and evaluated-density round trips.
Finite-source correctness does not qualify grid/cutoff convergence, external
ORCA/CRYSTAL or BIPOLE parity, source-specific symmetry skipping, gradients
or automatic accuracy settings.

### Infinite Ewald domains for full Gaussian sources

The compact-support API above cannot use a Gaussian extent radius as a support
certificate. Use native full Gaussian majorants for an orbital pair or an
auxiliary shell instead:

```python
target = manta.compute_pair_gaussian_majorant(basis, [0, 1],
                                              ket_translation=[0., 0., 0.])
source = manta.compute_auxiliary_gaussian_majorant(auxiliary_basis, 0)
error = manta.lattice_gaussian_domain_remainder(
    cell, eta, real_indices, reciprocal_indices, near_indices, target, source
)
print(error.real_remainder, error.reciprocal_remainder, error.total)
```

The bound applies uniformly to every AO component of the selected interaction,
including signed contractions and Cartesian/pure shells. Orbital angular
momentum through 6 and auxiliary angular momentum through 7 are supported.
Each immutable producer exports normalized positive spherical Gaussians via
`terms`, a total `charge_bound`, a resource `plan`, and basis/source identities.
There is no user-defined numerical constructor. Pair translations share the
represented ket-center convention of `PeriodicPairSource`.

Both Ewald domains must be complete centered integer cubes; reciprocal zero
is excluded. Near indices retain the finite kernel's existing convention.
The real error uses analytically convolved Gaussian decay, reported separately
as `minimum_effective_eta`. `physical_eta` and `kernel_identity` retain the
actual Ewald split. Increase the real cube if its omitted images cannot be
separated from the majorant centers in the conservative lattice metric.
An admitted cube may still have a loose bound; enlarge it until the requested
error is met. No omitted image is enumerated by the native bound routine.

`plan_pair_gaussian_majorant`, `plan_auxiliary_gaussian_majorant` and
`plan_lattice_gaussian_domain_remainder` report admitted storage and work.
The last plan accepts the two majorant term counts and three domain counts.
All joint array storage and work caps are checked before reading array payloads.
Modeled storage includes copies and exports; it is not process peak RSS.

Qualification `native-gaussian-infinite-ewald-domain-remainder` is a primitive
error estimate, excluding floating-point roundoff, omitted AO pair images,
density aggregation and complete SCF accuracy. Combining it with the finite
Gaussian ledger requires matching native sources and kernel identities.
The derivation is in the scientific contract; measurements and independent
complete smooth-charge energies are in the
[Gaussian domain benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_gaussian_domain.md).
The citation route is `manta_gaussian_domains`.

### Periodic bubble hierarchy

`PeriodicBubbleTree` groups the entries of a native `PeriodicPairSource`
without losing shell-image identities. It supplies moment aggregation,
density-dependent radial bounds, the paired reverse pass and a disjoint
geometry traversal:

```python
bubbles = manta.BubbleOptions()
bubbles.leaf_capacity = 16
hierarchy = manta.PeriodicBubbleTree(pair_source, bubbles)
forward = hierarchy.contract(densities)  # complex128 (k, AO, AO)
node_moments = forward.moments
reverse = hierarchy.adjoint(node_potentials)  # float64 (nodes, moments)
j_by_k = reverse.values
```

The native producer is retained by the hierarchy. `pair_order` maps node
ranges directly to the original `pair_source.entries`; it is not a molecular
triangular shell-pair list. Nodes use the existing enclosing-sphere algorithm.
`hierarchy.traversal(opening=0.5)` exposes `peek`, `accept`, `refine` and a
completion receipt. A far candidate is geometric only: a periodic Coulomb
consumer must separately validate all relevant kernel domains and errors.
Always-near leaves remain ineligible for far admission.

Forward contraction accepts arbitrary Hermitian densities at each momentum,
with the original nonuniform weights and phases. It assumes no time-reversal
relation. `moments` contains every node's full raw Cartesian moments;
`full_moment_bounds`, `tail_moment_bounds` and `coefficient_l1` retain positive
density-dependent bounds. Bounds exclude floating-point evaluation error.
`adjoint` reverses all translations, then the weighted pair-density map.
Potentials at every supplied node contribute; physical interaction assembly
must avoid counting overlapping node interactions twice.

`plan_periodic_bubbles(pair_source, bubbles, budget)` performs joint count-only
admission before allocating hierarchy or driver arrays. The plan includes
retained source storage, one forward/reverse cycle, one live translation
matrix and owned exports. No matrix is cached per tree edge and no global
node-to-AO map is formed. The plan is a conservative storage/work reservation,
not process RSS. Actual translation and pair-shell counts accompany results.
Array dtype, shape and contiguity must match the documented layouts.

The source and density identities distinguish changes in basis, image/cutoff,
lattice, k points, weights, extents, partition and density. The qualification
`periodic-pair-image-hierarchy-and-adjoint` is a finite-source building block;
it does not select periodic BUPO SCF or provide exchange. Kernel far admission,
omitted pair-image errors and aggregate accuracy profiles remain required.
See the [hierarchy benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bubbles.md)
for independent complete Coulomb calculations and measured cost. The citation
route is `manta_periodic_bubbles`.


### Direct Ewald J over the periodic pair source

`PeriodicEwaldSource(pair_source, eta, real_indices, reciprocal_indices,
reciprocal_batch=64, budget=ResourceBudget())` retains a native
`PeriodicPairSource` and builds its complete finite Ewald Coulomb response.
Use `.build(densities)` with contiguous complex128 `(momenta, AO, AO)`
Hermitian densities. The original source supplies its basis, cell, canonical
images, momenta and weights. The returned `.values` is complex J on that
same mesh, and `.receipt` reports real, reciprocal, gauge and total energies,
charge, weighted trace energy, Hermiticity residuals, source/kernel/density
identities and actual execution counts. Each exported array owns its data.

Real and reciprocal index arrays are contiguous int64 `(count,3)`. C and
Fortran layouts and unaligned buffers are accepted. Domains must be unique
and inversion closed, with zero present in the real domain and absent from
the reciprocal domain. Eta is finite and positive. An empty reciprocal
array denotes an intentionally incomplete source. The cell comes from the
pair source; its vectors are columns in bohr. Real vectors are `cell @ n`
and reciprocal vectors are `2*pi*inverse(cell).T @ g`.

The builder streams native erfc Gaussian shell quartets plus damped reciprocal
panels and the constant `-pi/(eta**2*volume)`. It includes finite Gaussian self
integrals and does not apply nuclear point-self subtraction. The gauge term
occurs once in the electronic component. A full-cell charge/background policy
still belongs to a complete electronic+nuclear calculation. For an unrestricted
calculation J uses total density; this source supplies no exchange matrix.

The fixed finite quadratic energy and weighted response satisfy
`E = 0.5*sum_k weights[k]*real(trace(D[k]@J[k]))`. General complex densities
need no time-reversal relation. Pair potentials scatter with the original
Bloch phases. Exact Gaussian integrals are independent of moment order and
extent radius; reciprocal batching changes only the streaming schedule.
There is no global four-index tensor or retained all-pair Fourier panel.

`plan_periodic_ewald(pair_source, real_cells, reciprocal_vectors,
reciprocal_batch=64, budget=ResourceBudget())` admits all five resource caps
before domain payloads are read. The model includes the existing source
reservation, local panels, shifted shell snapshots, density/response storage
and owned exports. It conservatively uses the full-basis Fourier plan as an
upper bound for each shell panel. Reported work/storage are models, not
measured FLOPs or RSS. Retained additional results belong to the caller's
memory budget.

Qualification is `finite-canonical-pair-ewald-j`. Canonical image folding
assumes periodic potentials; a finite real Ewald domain has a covariance
error that must converge or be bounded separately. Omitted pair images,
aggregate Gaussian/domain accuracy, nuclear terms, exchange, production
periodic BUPO and automatic SCF selection remain unqualified. This direct
source prepares the exact finite-domain fallback for future accepted/refined
bubble blocks. It does not modify shared FockBuild/UHF, physical BIPOLE
image domains or qualified symmetry dispatch. See the
[scientific contract](../design/manta_scientific_contract.md) and the repository's
`benchmarks/manta_periodic_ewald.py` and `benchmarks/manta_periodic_ewald.md`.
The citation route is `manta_periodic_ewald`.

### Finite periodic BUPO J with direct fallback

`PeriodicBupoSource(tree, direct, near_indices, options=PeriodicBupoOptions(),
budget=ResourceBudget())` combines the periodic hierarchy and direct Ewald
source above. Both must retain exactly the same native pair-source identity.
`near_indices` is contiguous int64 `(count,3)`, unique, inversion closed,
contains zero and is contained in the direct real domain. The options are
`opening` (default 0.5, between zero inclusive and one exclusive) and
`integral_tolerance` (default 1e-10, nonnegative).

```python
from vibeqc import manta
import numpy as np

# tree and direct retain the same previously constructed pair_source.
options = manta.PeriodicBupoOptions()
options.integral_tolerance = 1e-8
source = manta.PeriodicBupoSource(
    tree, direct, np.zeros((1, 3), dtype=np.int64), options
)
result = source.build(densities)  # complex128 (momenta, AO, AO)
j = result.values
energy = result.receipt.energy
j_bound = result.j_error_bound   # (AO, AO), uniform over the original k mesh
energy_bound = result.receipt.energy_error_bound
```

Geometry alone cannot admit a periodic far block. The source checks all
real images and sums native Gaussian bounds for bare near images plus the
complementary finite Ewald tail. A failing block refines or uses exact finite
Ewald integrals. Zero tolerance selects the complete direct response.
Full Cartesian moments retain trace contributions and the gauge occurs once.
The fixed partition is independent of density; J is linear and is the weighted
derivative of the reported quadratic energy. UHF supplies total alpha plus
beta density for J.

`.partition` reports accepted/direct blocks and pair interactions, refinement
counts, the largest accepted component-integral bound and an identity.
`.receipt` includes that partition, source/tree/kernel/density identities,
direct and far energies, independently evaluated weighted trace, charge and
execution counts. In a mixed partition, `.receipt.direct` describes only
the direct subset, has its own subset identity, and reports that subset's
real/reciprocal/gauge energies and trace. Its charge describes the full source.
Each exported array owns its storage.

`plan_periodic_bupo(tree, direct, near_cells, options, budget)` admits all
five resource caps before reading near indices or copying densities. It
reserves the retained sources, Gaussian envelopes, driver arrays and worst
case direct work. It streams M2L matrices and shell/Fourier panels without a
global four-index tensor or per-edge cache. These conservative models are
distinct from actual counts and process RSS. Small or diffuse systems may
use few far blocks, or be slower than the direct source.

Qualification `finite-periodic-bupo-j` covers the supplied finite canonical
source. Bounds include its Gaussian/Taylor approximation and both canonical
scatter arms, but exclude roundoff, omitted AO pair images, infinite-domain
error and nuclear terms. The integral threshold is not a total-energy or
SCF accuracy target. Production periodic BUPO/RI-BUPO selection, full charge
policy and geometry derivatives remain open. Shared FockBuild/UHF, physical
BIPOLE domains and qualified symmetry dispatch are preserved. See the
[finite BUPO benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bupo.md)
for timings and actual calculation evidence. The citation route is
`manta_periodic_bupo`.

### Adding an infinite-Ewald domain bound

Set `options.bound_infinite_domain = True` when constructing a
`PeriodicBupoSource` to bound the difference from the infinite Ewald kernel
for its retained Gaussian pair source. The option defaults to false.
The calculated finite J, energy and fixed near/far partition are unchanged.
Real and reciprocal domains must be complete centered integer cubes, with
the usual real origin and reciprocal zero exclusion. Every omitted real
image must be separated from the native majorant centers. An inadequate
domain refuses at construction; enlarge it and construct a new source.

With the option enabled, `result.receipt.domain` reports separate
`real_energy_error_bound` and `reciprocal_energy_error_bound`, their sum,
source/domain/envelope/density identities and execution counts.
`result.domain_j_error_bound` is an owned `(AO, AO)` array uniform over k.
Both properties are `None` in finite-only mode. Near-image partition changes
do not change these domain bounds.

The existing `result.receipt.energy_error_bound` and
`result.j_error_bound` include finite BUPO plus domain error when enabled.
`finite_energy_error_bound` and `finite_maximum_j_error_bound` in the main
receipt preserve the finite approximation contribution. Compare aggregate
bounds with the intended energy and J targets; the component-integral
threshold alone is not an accuracy guarantee. Small domains can give very
loose bounds, even when the measured finite error is small.

Native full Gaussian envelopes, original weighted density norms, diagonal
half-energy factors and both canonical scatter arms determine the bounds.
No truncated extent sphere is treated as full Gaussian support. Positive
logarithmic arithmetic keeps nonzero underflowed bounds positive, but
roundoff is not certified. Domain geometry is prepared once and pair bounds
stream without a quadratic cache.

Qualification `periodic-bupo-j-with-retained-source-ewald-bound` covers
the infinite kernel for the same retained pair-image inventory. It includes
the finite kernel's covariance error relative to that periodic limit;
it excludes omitted AO pair images, full electronic+nuclear charge policy,
exchange and geometry derivatives. It does not select production SCF or
RI-BUPO. See the
[domain-bound benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bupo_domain.md)
for domain enlargement, complete reference calculations and cost.

### Adding the omitted AO pair images

Set both `options.bound_infinite_domain = True` and
`options.bound_pair_images = True` to include error from every excluded
AO pair translation. This includes holes in the supplied image set and pairs
excluded by the physical pair cutoff. No new J or energy terms are calculated;
the option bounds their missing contribution to the full electronic Coulomb
response under the infinite zero-average conducting kernel.

`result.receipt.pair_images` records native source/kernel/density identities,
the completion cube, execution counts, conservative lattice metrics, absolute
kernel bounds and retained/missing density masses. Its `energy_error_bound`
and `maximum_j_error_bound` describe pair-image omission alone.
`result.pair_image_j_error_bound` is an owned `(AO, AO)` array uniform over k.
`retained_pair_mass_bound` and `omitted_pair_mass_bound` expose the native
ordered AO component envelopes. All four properties are `None` when disabled.

The overall `energy_error_bound` and `j_error_bound` now add finite BUPO,
retained-source infinite-Ewald and pair-image errors. The J estimate includes
both missing target pairs and missing density. Signed contractions, original
mesh weights, conjugate multiplicities and nonzero diagonal images are
included without assuming a relation between states at k and minus k.
The finite values and fixed partition remain unchanged.

Native primitive Gaussian envelopes dominate all space and sum the infinite
omitted image complement. No extent sphere is treated as compact Gaussian
support. Plans admit the finite completion-cube work and owned arrays before
payload reads; oversized sparse domains can exceed these resource caps.
Positive logarithmic arithmetic preserves small nonzero bounds. Roundoff is
excluded, and small domains or diffuse functions can give very loose bounds.

Qualification `periodic-bupo-j-with-full-pair-image-ewald-bound` covers the
declared full electronic Coulomb source. Compare both total bounds with the
intended accuracy target. Enlarge source and Ewald domains or decline the
target when it is not met. The matched electronic+nuclear charge policy is
available through the source below. Exchange, RI-BUPO, automatic production
SCF selection and derivatives remain separate requirements. See the
[pair-image benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_image_bounds.md)
for measured errors, convergence and overhead.


## Matched electronic and nuclear electrostatics

`PeriodicElectrostaticSource(electronic, positions, charges, options, budget)`
adds point nuclei to a native `PeriodicBupoSource` whose infinite-domain and
pair-image bounds are both enabled. Positions are contiguous float64
`(points, 3)` in bohr and charges are float64 `(points,)`. Positive atomic
numbers give physical nuclei; signed fractional charges are also accepted.
The constructor computes nuclear attraction and point Ewald energy using
the electronic source's exact finite domains and gauge. It cannot reuse a
nuclear matrix generated with a different cutoff convention.

```python
policy = manta.PeriodicElectrostaticOptions()  # neutral, tolerance 1e-10
source = manta.PeriodicElectrostaticSource(electronic, positions, charges, policy)
result = source.build(densities)  # complex128 (original k points, AO, AO)
electrostatic_energy = result.receipt.energy
response = result.values  # weighted derivative J + V_nuc
```

The default `neutral` policy requires the absolute retained net charge plus
the bound on omitted electron-image charge to be within `charge_tolerance`.
A density normalized against a small retained overlap may therefore refuse.
Enlarge the pair source, or explicitly choose
`policy.charge_policy = "uniform-background"` for a charged periodic source.
The latter records compensating retained charge and its uncertainty. Its
background contribution is already present in the zero-average Ewald gauge.
The point-nuclear self term is subtracted once; electronic Gaussian self
interaction is retained.

The receipt resolves electronic energy; attraction real, reciprocal and
gauge energies; nuclear real, reciprocal, point-self and gauge energies;
net charge, uncertainty and background; and the full error ledger.
`result.electronic` retains the underlying BUPO result and fixed partition.
`source.overlap` and `source.nuclear_values` export owned arrays. So do
`result.values`, `j_error_bound`, and `nuclear_j_error_bound`. Matrix bounds
are `(AO, AO)`, uniform over the original k mesh. `values` is J+V, not a
complete HF/KS Fock matrix; kinetic, exchange and XC must be composed by the
method driver. Its energy is not one half of Tr(D*(J+V)).

`plan_periodic_electrostatic(electronic, points, options, budget)` checks
combined memory and all four work limits before scanning point payloads.
Duplicate nonzero periodic points and an insufficient real-domain separation
refuse. Zero-charge points have no interaction. Coordinates are not wrapped
automatically. No fitting or production SCF source is selected by this API.

Qualification `full-pair-image-electronic-point-electrostatics` includes
retained Ewald-domain and omitted electronic-image errors for all three
charge components. Compare both total matrix and energy bounds with the
intended targets. Roundoff is excluded; diffuse/sparse cases can have loose
bounds. See the [electrostatic benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_electrostatic.md)
and scientific contract Section 5.10. The citation route is
`manta_periodic_electrostatic`.


## Periodic RHF/UHF with matched BUPO electrostatics

`run_periodic_bupo_hf` runs bounded native RHF or UHF using an existing
`PeriodicBupoSource`, its matched nuclear electrostatics and explicit finite
reciprocal exchange. It requires a neutral all-electron 3D physical system
and the source's original uniform `RegularKMesh`. Enable both electronic
error options before constructing the source. Its original ordered image
inventory and physical pair cutoff also supply overlap, kinetic and exchange.

```python
options = manta.PeriodicBupoHFOptions()
options.spin_mode = "unrestricted"  # alternatively "restricted"
options.exchange_q0 = "BvK-ewald"   # default: "strict-zero-mode"
options.maximum_electrostatic_energy_error = 1e-8
options.maximum_electrostatic_matrix_error = 1e-8
options.maximum_overlap_image_error = 1e-10
options.maximum_kinetic_image_error = 1e-8
options.maximum_kinetic_energy_error = 1e-8
result = manta.run_periodic_bupo_hf(
    electronic, system, mesh, exchange_vectors, transfer_offsets,
    options=options, integral_engine=engine,
)
energy_per_cell = result.receipt.energy
fock = result.fock
```

`engine` is an `IntegralEngineOptions(backend="manta", budget=budget)`.
Exchange vectors are contiguous float64 `(count, 3)` in inverse bohr;
int64 offsets `(nk*nk+1,)` delimit target-major/source-minor transfer
channels. Supply the same full reciprocal exchange convention as the native
periodic exchange API. `plan_periodic_bupo_hf(electronic, system, mesh,
maximum_vectors_per_transfer, options, budget)` checks memory and all four
work caps. The selected Python wrapper reserves additional output storage.

Multiplicity sets the UHF populations. RHF requires multiplicity one;
each spin needs at least one virtual after overlap rank reduction. Empty
beta density is supported. The solver uses fixed integer occupations,
retains the original mesh order and performs a same-density terminal Fock
and energy confirmation before reporting `converged=True`.

Owned density and Fock arrays always have shape `(spin, k, AO, AO)`, with
one restricted-total channel or alpha/beta channels. Coefficients have
shape `(spin, k, AO, retained_rank)` and orbital energies
`(spin, k, retained_rank)`; overlap and hcore are `(k, AO, AO)`. The evaluated
density is authoritative, including a run stopped at its iteration cap.
Coefficients can describe the next fixed-point trial and must not replace
that density when constructing a shared `FockBuild` snapshot.

Every Fock call must pass neutral charge admission, both declared
electrostatic targets and the density-weighted kinetic image target.
Overlap and kinetic matrix image targets are checked before SCF. A refusal requires changing the source/domain or an
explicit target choice. The driver never silently changes the partition.
The receipt reports complete HF energy components, the original BUPO
partition, exchange convention, nested electrostatic ledger and maximum
observed error bounds. Output uses `vibeqc.output` and citation route
`manta_periodic_bupo_hf`; BvK correction adds `manta_periodic_madelung`.

The default targets bound electrostatics and omitted overlap/kinetic images
at the evaluated density. Optional [exchange certification](#full-coulomb-exchange-error-bounds)
also bounds exchange and its overlap-dependent BvK correction at fixed mesh
and Madelung coefficient. Neither covers basis incompleteness, SCF-state,
overlap-rank stability or roundoff errors. No automatic whole-HF profile,
public job routing, q-aware RI-BUPO, symmetry reduction or external SCF
parity is implied.
For cutoff sources, enlarging the original image cube can tighten the
missing-image enclosure while leaving all retained integrals unchanged.
See scientific contract Section 5.11 and the
[BUPO HF benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_bupo_hf.md).


## Full overlap and kinetic image bounds

`bound_periodic_one_electron_images(source, budget)` takes an immutable
`PeriodicPairSource` and returns nonnegative `(AO, AO)` matrices
`overlap_error_bound` and `kinetic_error_bound`. They enclose every omitted
AO-image contribution, uniformly over k, including physical-cutoff
exclusions and sparse interior holes. They do not change retained S or T.
The bound uses the original image cube before the physical cutoff.

```python
plan = manta.plan_periodic_one_electron_images(pair_source, budget)
bounds = manta.bound_periodic_one_electron_images(pair_source, budget)
max_overlap_error = bounds.receipt.maximum_overlap_error
max_kinetic_error = bounds.receipt.maximum_kinetic_error
```

The native producer uses signed-contraction absolute Gaussian majorants and
kinetic gradient-product majorants, with a checked infinite lattice sum.
No user numerical envelope, Coulomb approximation or k-point cancellation
is accepted. Source/geometry identities, actual counts and all five resource
limits accompany the owned arrays. Sparse or diffuse inventories can give
very loose bounds and should fail tight target admission.

Periodic BUPO HF includes this result as `result.one_electron_images`.
Its receipt records the three target values, `one_electron_accuracy_passed`,
`kinetic_energy_error_bound` for the last evaluated total density and
`maximum_observed_kinetic_energy_error`. Positive finite targets default to
1e-10 for overlap, 1e-8 for the kinetic matrix and 1e-8 hartree for kinetic
energy. Output carries this ledger and citation route
`manta_periodic_one_electron_images`.

The S/T bounds exclude roundoff, basis error and changes to the converged
state or overlap rank. The [separate exchange certificate](#full-coulomb-exchange-error-bounds)
uses the overlap bound for its BvK correction. These bounds do not authorize
a whole-HF accuracy profile or symmetry reduction. See scientific contract
Section 5.12 and the
[one-electron image benchmark](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_periodic_one_electron_images.md).


## Full-Coulomb exchange error bounds

`PeriodicExchangeBoundSource` bounds the error from both omitted AO images
and omitted reciprocal vectors at the original fixed k mesh and density.
It uses the same full-Coulomb, omitted-zero-mode convention as
`compute_periodic_exchange`. The explicit reciprocal completion cube must
contain every supplied transfer-lattice label; it may contain holes, which
are included in the error. It does not change the retained calculation.

```python
certificate = manta.PeriodicExchangeBoundSource(
    pair_source, exchange_vectors, transfer_offsets,
    reciprocal_cube_radius=reciprocal_radius, budget=budget,
)
errors = certificate.build(spin_densities, budget)
maximum_k_error = errors.receipt.maximum_error
spin_energy_errors = errors.receipt.energy_error_bounds
```

`spin_densities` has shape `(spin, k, AO, AO)`. The owned arrays
`pair_image_error_bound`, `reciprocal_error_bound` and `error_bound` have
that same shape and contain nonnegative entrywise K bounds. Energy bounds
use the unscaled spin-channel exchange functional `-0.5 Tr(D K)` and the
source's original weights. Signed Hermitian trial densities are supported.
No input numerical envelope, inferred symmetry or refitted density is used.
`plan_source` and `plan_build` admit all five resource limits before payload
copies. Source/geometry/transfer and density identities accompany the bounds.

Periodic BUPO RHF/UHF can require the certificate in every Fock evaluation:

```python
options.bound_exchange = True
options.exchange_bound_cube_radius = reciprocal_radius
options.maximum_exchange_matrix_error = 1e-8
options.maximum_exchange_energy_error = 1e-8
```

The targets apply after the method's restricted/unrestricted exchange scale
and include the overlap error in the BvK correction when selected. The
receipt keeps the last and maximum observed errors, the separate correction
errors, `exchange_accuracy_passed`, and the nested native `exchange_bounds`
receipt. A failed target refuses before exchange evaluation; enlarge the
actual AO/reciprocal support and retry at the same target when appropriate.
Enlarging only the completion cube may tighten the enclosure, but does not
remove the error of missing physical images or reciprocal vectors.

Exchange certification defaults to false, retaining the explicitly finite
exchange route. Enabled certification covers the integral errors at evaluated
density and fixed mesh/Madelung coefficient. It excludes k-mesh convergence,
Madelung evaluation error, basis completeness, relaxed-state/overlap-rank
stability and roundoff. Range-separated exchange and automatic whole-method
profiles need separate qualification. See scientific contract Section 5.13.
