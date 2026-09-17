# Theory of OpenTrustRegion molecular SCF

OpenTrustRegion minimizes the electronic energy by rotating occupied and
virtual molecular orbitals. The molecular geometry, basis, charge, spin
electron counts and energy functional stay fixed. This chapter explains the
coordinates and derivatives supplied by vibe-qc and how the pinned optimizer
uses them. See the [reference](opentrustregion.md) for supported methods and
options, or the [worked comparison](../tutorial/opentrustregion.md) to run it.

## What is being optimized?

Let $C$ contain the molecular-orbital coefficients in an AO basis with overlap
matrix $S$. The retained orbitals satisfy $C^TSC=I$. For a restricted
integer-occupation state, the spin-summed density is

$$
D=2C_oC_o^T,
$$

where $C_o$ contains occupied columns. With $h$ the one-electron Hamiltonian,
$J[D]$ and $K[D]$ the Coulomb and exchange matrices, and $a_x$ the fraction of
exact exchange, the supported restricted energy has the form

$$
E[D]=E_{\mathrm{nuc}}+\operatorname{Tr}(Dh)
+\frac12\operatorname{Tr}(DJ[D])
-\frac{a_x}{4}\operatorname{Tr}(DK[D])+E_{\mathrm{xc}}[D].
$$

RHF has $a_x=1$ and no semilocal $E_{\mathrm{xc}}$. RKS uses the selected
supported functional, including its semilocal mixing weights. Unrestricted
states use separate $D_\sigma=C_{o\sigma}C_{o\sigma}^T$, with Coulomb built
from $D_\alpha+D_\beta$, same-spin exact exchange, and spin-dependent XC.
Every term is evaluated by vibe-qc. The optimizer receives an energy and its
derivatives, without implementing a separate electronic-structure model.

An occupied-occupied rotation leaves the density unchanged, as does a
virtual-virtual rotation. Those redundant coordinates are omitted. One
restricted calculation therefore has $n_o n_v$ real parameters; an
unrestricted calculation has $n_{o\alpha}n_{v\alpha}
+n_{o\beta}n_{v\beta}$.

## Orbital rotations preserve the constraints

Arrange the nonredundant parameters in a virtual-by-occupied matrix $k$:

$$
K(k)=\begin{pmatrix}0&-k^T\\k&0\end{pmatrix},\qquad
C(k)=C\exp[K(k)].
$$

Because $K^T=-K$, its exponential is orthogonal. Consequently,

$$
C(k)^TSC(k)=\exp(-K)\exp(K)=I.
$$

The density remains integer-occupied and its electron count is preserved.
This also explains why the trial point is a rotation of the current accepted
orbitals, rather than an addition to the AO density. For unrestricted states
each spin has its own rotation, with both spin blocks optimized together.
vibe-qc flattens each $k$ column by column, with alpha preceding beta.

After accepting a step, the reference orbitals move to that point and a new
local model is built around zero rotation. Rejected trial energies leave the
accepted orbitals, density, Fock matrix and response model untouched. This
is essential: mixing a trial density into an incremental Fock cache would
change the function represented by the next model.

## Physical gradient and Hessian-vector products

Write $F_{ai}$ for a virtual-occupied element of the Fock matrix in the
current MO basis. In the convention above, the energy gradients are

$$
g_{ai}=4F_{ai}\quad\text{(restricted)},\qquad
g^\sigma_{ai}=2F^\sigma_{ai}\quad\text{(unrestricted)}.
$$

The factors differ because a restricted spatial orbital carries two
electrons. A stationary SCF determinant has a zero occupied-virtual Fock
block. Its occupied and virtual blocks need not already be diagonal.

For a trial direction $v$, reshaped into $k$, the density response is

$$
\delta D=s\left(C_vkC_o^T+C_ok^TC_v^T\right),
\qquad s=2\ \text{(restricted)},\quad s=1\ \text{(each spin)}.
$$

Applying the physical Fock response to this density change gives

$$
(Hv)_{vo}=f\left(F_{vv}k-kF_{oo}+C_v^T\delta F C_o\right),
\qquad f=4\ \text{(restricted)},\quad f=2\ \text{(each spin)}.
$$

The first two terms account for orbital rotation of the existing Fock matrix;
the last accounts for the change in the electronic potential. The full
$F_{oo}$ and $F_{vv}$ blocks are retained, including away from stationarity.
For HF, $\delta F$ contains Coulomb and exchange response. For supported KS
functionals it also contains the XC kernel, including density-gradient
response for GGA. In unrestricted calculations both spin directions enter the
Coulomb response and the coupled spin-XC kernel.

The adapter supplies this action without constructing a dense matrix with
$n_{\mathrm{rot}}^2$ entries. Its approximate diagonal
$d_{ai}=f(F_{aa}-F_{ii})$ is used for preconditioning: it helps the iterative
solver choose useful directions. It does not replace $Hv$, and negative
physical curvature is not removed by flooring the preconditioner. DFT AO
tables, matrix exponentials and JK builds still have their own memory and
runtime costs; a matrix-free orbital Hessian does not imply a linear-scaling
electronic-structure calculation.

## The quadratic model and trust radius

At an accepted state with energy $E_0$, approximate the local energy by

$$
m(p)=E_0+g^Tp+\frac12p^THp.
$$

An unrestricted Newton step solves $Hp=-g$. A very small curvature can make
that step too large for the local approximation; negative curvature can make
the quadratic model unbounded below. A finite trust region instead asks for

$$
\min_p m(p)\quad\text{subject to}\quad\|p\|_2\leq\Delta.
$$

For this Euclidean problem, a minimizing step satisfies

$$
(H+\lambda I)p=-g,\qquad H+\lambda I\succeq0,\qquad
\lambda\geq0,\qquad \lambda(\|p\|_2-\Delta)=0.
$$

When a positive-definite Newton model gives a step inside the radius,
$\lambda=0$ is possible. Otherwise a shift restricts the step. This shift
belongs to the subproblem; it does not modify the physical Fock matrix or the
energy being minimized. The paper uses the equivalent sign convention
$\mu=-\lambda$ for its level shift.

The published OpenTrustRegion method develops a scaled augmented-Hessian
formulation and iterative reduced-space solution, allowing the host to supply
only objective and derivative callbacks. A Davidson space carries trial
vectors and their Hessian images; the reduced problem determines a step and
shift without diagonalizing the full orbital Hessian. The scaling/shift is
adjusted to meet the radius. See sections 2 and 3 of
[Greiner et al.](https://arxiv.org/html/2509.13931v2#S2) for the derivation and
algorithmic details. The following implementation distinctions refer to the
[pinned upstream source](https://github.com/eriksen-lab/opentrustregion/blob/8fa7769ae66233a566868a6bf03cdcbdb1ee69d0/src/opentrustregion.f90).

| Subsolver | How it builds the step | Radius convention in the pinned version |
|---|---|---|
| `davidson` | Expand a preconditioned reduced space and solve the shifted subproblem there | Euclidean norm of the independent rotation vector |
| `jacobi-davidson` | Begin with Davidson and use an iterative correction equation when its switching criterion is met | The same Euclidean convention |
| `tcg` | Truncated preconditioned conjugate gradients, stopping or moving to the boundary on negative curvature or a radius crossing | Preconditioner-weighted norm |

Specifically, with this adapter's supplied diagonal and upstream's default
preconditioner, TCG measures $\sqrt{p^TPp}$, where
$P_{jj}=1/\max(|d_j|,10^{-10})$. Its boundary is not the Euclidean sphere
written above. A numerical `initial_trust_radius` therefore need not imply
the same rotation size across subsolvers. The finite-radius formulation is
the common idea; the metric and numerical solver are implementation choices.

## Accepting a step using an actual energy

The model predicts an energy decrease, but the actual energy must be checked.
For a step with a positive predicted reduction, define

$$
\rho=\frac{E_0-E(p)}{m(0)-m(p)}.
$$

A value near one means the quadratic model described that trial well. A
negative value means the actual energy rose despite a predicted decrease.
With the default `line_search=False`, the pinned library applies this policy:

| Outcome | Step | Next radius |
|---|---|---|
| Subproblem not converged, $\rho<0$, or an individual rotation exceeds $\pi/4$ in magnitude | Reject | $0.7\Delta$ |
| Otherwise, $0\leq\rho<0.25$ | Accept | $0.7\Delta$ |
| Otherwise, $0.25\leq\rho<0.75$ | Accept | Unchanged |
| Otherwise, $\rho\geq0.75$ | Accept | $1.2\Delta$ |

For example, a predicted reduction of `0.010` Ha and an actual reduction of
`0.008` Ha give $\rho=0.8$. Subject to the other checks, the step is accepted
and the radius expands. If the actual energy instead rises by `0.001` Ha,
$\rho=-0.1$: the trial is rejected and the radius contracts. These are
illustrative model values, not molecular benchmark energies.

An optional line search subsequently searches along the proposed direction
using additional trial energies. It can change the step length and add
substantial work; the radius alone then does not describe the final rotation.
The library owns these decisions. vibe-qc commits an accepted update only
after its new energy, gradient and response state have been constructed
successfully. An outer accepted update can require many trial energies and
Hessian actions, which is why iteration counts alone are a poor cost measure.

## Convergence, curvature and the final orbitals

vibe-qc requires a small accepted energy change, a small AO commutator
$FDS-SDF$ and a small orbital-gradient RMS. The RMS is
$\|g\|_2/\sqrt{n_{\mathrm{rot}}}$, with both spin blocks included for an
unrestricted state. The AO residual uses the Frobenius norm, or the maximum
of the two spin norms. The final physical Fock matrix and energy are rebuilt
before reporting success. An upstream small-step or machine-precision exit
does not by itself prove these conditions.

A small gradient establishes stationarity, while negative Hessian curvature
identifies a downhill direction. For a stationary point and direction $u$,
$E(tu)-E_0=\tfrac12t^2u^THu+O(t^3)$: if $u^THu<0$, arbitrarily small steps
can lower the energy. Conversely, passing a finite-tolerance internal check
only certifies the tested real orbital manifold to that numerical threshold.
It does not prove a global minimum or stability against additional spin or
complex rotations.

The [stability reference](opentrustregion.md#stability-and-diagnostics)
explains the pinned library's initial-saddle behavior and the different
following and final-check thresholds. The
[stretched-H2 tutorial](../tutorial/opentrustregion_stability.md) demonstrates
why a stationary restricted state can lower its energy in an unrestricted
space while keeping the electron counts fixed.

Finally, vibe-qc diagonalizes the occupied and virtual Fock blocks separately.
This produces semicanonical orbital energies while preserving the accepted
occupied projector. Diagonalizing and refilling the entire Fock matrix could
replace that state with a different determinant.

## Method reference and automatic attribution

The defining library paper is:

```{vibeqc-cite-entry} greiner_opentrustregion_2026
```

The [author preprint](https://arxiv.org/abs/2509.13931) provides an accessible
derivation. The journal article is the citation used in generated
bibliographies. Its metadata comes from the same database that produces the
job output: four authors, year, volume, issue, pages and DOI. The prose style
shortens the author list with "et al."; BibTeX retains all four names.

For a successful public `run_job` calculation, the executed backend report
selects this paper in the `.out` References block, `.references` and
`.bibtex`. An ordinary native SCF does not add it. An OpenTrustRegion SCF
does not receive default DIIS credit merely because it is an SCF calculation.
`vibeqc-cite` preserves the recorded paper selection when reprinting or
regenerating references; see [automatic citations](citations.md).

The lower-level `run_rhf`/`run_uhf`/`run_rks`/`run_uks` calls return numerical
results without creating a job bibliography. Use `run_job` for automatic
user-facing output, or cite the paper explicitly in a custom driver. The
adapter's derivative/state contract is documented in the
[reference](opentrustregion.md#derivative-and-state-contract). Localization
and CASSCF remain future adapters; their background references do not imply
those methods ran in an OpenTrustRegion molecular SCF calculation.
