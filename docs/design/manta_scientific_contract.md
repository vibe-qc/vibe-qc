---
orphan: true
---

# MANTA scientific contracts and literature mapping

This is the scientific part of the [MANTA proposal](manta_engine.md).
`Published` below identifies an algorithm described in a source; `existing`
identifies inspected vibe-qc behavior; `proposed` identifies a MANTA choice or
extension requiring the [acceptance gates](manta_validation_plan.md).
Equations below use explicit MANTA conventions and are not transcriptions of
every source's notation or normalization.

## 1. What was read and what it establishes

The local reading guide supplied discovery and version information. The
papers themselves, including their theory, implementation and numerical
sections, supplied the algorithmic evidence. Critical extent and lattice-sum
equations were also checked in rendered PDF pages. SI was inspected separately.

| Source and stored version | Sections/equations used | Evidence and limit |
| --- | --- | --- |
| Neese, *The SHARK integral generation and digestion system*, online 2022, print 2023, [10.1002/jcc.26942](https://doi.org/10.1002/jcc.26942), journal PDF | Eqs. 11-26; Sections 3.1-3.10; Eqs. 38-50, 54 | Matrix MD, folded pure transforms, segmented/general/partial-general contractions, multi-target digestion, Split-J/RI-J, radial replacement and derivatives. Its libint hybrid is evidence that one kernel family is not fastest everywhere. Its large matrices are not permission for unbounded MANTA storage. |
| Neese et al., *The Bubblepole (BUPO) Method for Linear-Scaling Coulomb Matrix Construction with or without Density Fitting*, 2025, [10.1021/acs.jpca.4c07415](https://doi.org/10.1021/acs.jpca.4c07415), journal PDF and SI ZIP | Eqs. 2-24, 29-36; Sections 2.4-2.7, Schemes 1-3, Sections 4.1-4.3, Appendix Eqs. 37-49 | Molecular direct BUPO-J is outlined; the documented numerical evaluation concentrates on RI-BUPO-J. SI contains 25 XYZ geometries, not a periodic derivation or hidden exchange implementation. No periodic, exact-exchange, force or stress capability follows from this paper. |
| Colinet, Neese and Helmich-Paris, *Improving the Efficiency of Electrostatic Embedding Using the Fast Multipole Method*, 2025, [10.1002/jcc.27532](https://doi.org/10.1002/jcc.27532), journal PDF and 18-page SI | Main Eqs. 1, 4-12, 17-21; Sections 3.2-3.7; SI Sections 1-3, 6, 8, 13 | Point-charge embedding, octree M2M/M2L/L2L, shell extents, adaptive expansion order and roundoff/timing accounting. A crystal embedded in finite point charges is not a periodic Bloch J/K solver. SI and main-text adaptive-order inequality wording must be reconciled against the actual error inequality, not copied as a loop condition. |
| Reine, Helgaker and Lindh, *Multi-electron integrals*, 2012, [10.1002/wcms.78](https://doi.org/10.1002/wcms.78), published PDF | Gaussian products/Boys functions; Eqs. 12-21; Cartesian, Hermite, solid-harmonic and derivative sections | Reference for integral families and conventions. Its Hermite normalization is not assumed identical to SHARK or the convention below. |
| Asadchev and Valeev, high-l 4c integrals, [10.1021/acs.jpca.3c04574](https://doi.org/10.1021/acs.jpca.3c04574), arXiv:2307.03452v3 | Eqs. 3-15 and implementation discussion | Matrix MD on accelerators and memory-aware contraction ordering. The stored text is an author manuscript. GPU versus single-CPU speedups are not MANTA expectations. |
| Asadchev and Valeev, memory-efficient 3c integrals, [10.1021/acs.jctc.2c00995](https://doi.org/10.1021/acs.jctc.2c00995), arXiv:2210.03192v2 | Sections 2.3-2.4, multiquantal VRR/HRR and in-place storage | This is primarily an OS/HGP memory-footprint design, not the same MD-GEMM algorithm as the 4c paper. Borrow scheduling lessons; do not misattribute its recurrences. |
| Palethorpe and Barca, high-l GPU J engine, [10.1021/acs.jctc.5c00775](https://doi.org/10.1021/acs.jctc.5c00775), ChemRxiv v2 plus journal SI | Sections 3.2-6; Algorithms 1-4; SI kernel and J timings | MD search-tree heuristic, pruning and directional sub-batches balance registers against recomputation. This is a J engine, not evidence of exact K or periodic boundary implementation merely because a benchmark has a crystal-shaped structure. Stored main text is a preprint. |
| Kudin and Scuseria, periodic Gaussian DFT, 2000, [10.1103/PhysRevB.61.16440](https://doi.org/10.1103/PhysRevB.61.16440), journal PDF | Eqs. 2.2-2.10, 2.13-2.20; Section III A/C/G | Periodic FMM, density/Bloch transforms and separate atomic/cell derivative terms. Neutral electrostatics and accurate lattice sums matter even with diffuse bases. This supplies periodic precedents, not a BUPO implementation. |
| Kudin and Scuseria, *Revisiting infinite lattice sums with the periodic fast multipole method*, 2004, [10.1063/1.1771634](https://doi.org/10.1063/1.1771634), journal PDF | Eqs. 9-24 and 25-31; Tables I-II; Section V | Reusable lattice translation sums, renormalization recurrence, and additional stress sums. Convergence of coefficients and low-order charge/dipole treatment need explicit handling. |

Additional periodic context was checked against the primary publication pages
for [Kudin and Scuseria 1998, periodic Gaussian FMM](https://doi.org/10.1016/S0009-2614(98)00468-0)
and [arbitrary cell geometries](https://doi.org/10.1016/S0009-2614(97)01329-8).
Kudin's [2000 dissertation](https://scuseria.rice.edu/preprints/kudin-dissertation.pdf),
Chapter 3, was read for its explicit discussion of neutrality, surface dipoles
and the potential's spheropole offset. It explains why matching lattice
energies does not alone establish a matching potential zero. The 2004 paper
also discusses the Ewald-style lattice-sum lineage, including
[Challacombe, White and Head-Gordon 1997](https://doi.org/10.1063/1.474150);
that paper was not separately available as full text in this audit and is not
used here as an equation-level implementation specification.

## 2. Basis, integrals and density conventions

Use atomic units. Primitive Cartesian functions are unnormalized monomials

$$
g_{\mathbf a}(\alpha,\mathbf A;\mathbf r)
=\prod_{i=x,y,z}(r_i-A_i)^{a_i}e^{-\alpha|\mathbf r-\mathbf A|^2}.
$$

Coefficients and component factors supplied by the basis adapter carry all
primitive/contracted normalization. `ShellInfo` coefficients must not be
normalized a second time. Publish a normalization version and a component
scale map; verify AO values, overlaps and pure transforms independently.
General contraction is a primitive-by-channel matrix over a shared exponent
set. Group equal exponent sets exactly, retaining a reversible AO index map;
never merge nearly equal exponents or modify a contraction to improve batching.
Mixed pure/Cartesian shells, ghosts, reordered radial channels and cross-basis
overlap are part of the interface.

Pure real components are `m=-l,...,+l`, with the existing real-harmonic phases.
Cartesian order is descending x then descending y. A pure transform acts on
libint's shell-wide axially normalized Cartesian components, which are not
individually unit-normalized for l greater than one. The existing transform
includes the required component factors. Multiplying them into the shell
coefficients again is a wrong-answer bug. Basis functions are
initially real Gaussians; complex Bloch coefficients are supported by complex
contractions. GIAO, spinor and magnetic integrals need additional capabilities.

For general complex functions define

$$
(\mu\nu|\lambda\sigma)=\iint
 \phi_\mu^*(1)\phi_\nu(1)v(1,2)
 \phi_\lambda^*(2)\phi_\sigma(2)\,d1\,d2,
\qquad D=CfC^\dagger.
$$

Then $\rho=\sum_{\lambda\sigma}D_{\lambda\sigma}
\phi_\lambda\phi_\sigma^*$,
$J_{\mu\nu}=\sum_{\lambda\sigma}D_{\lambda\sigma}
(\mu\nu|\sigma\lambda)$ and
$K_{\mu\nu}=\sum_{\lambda\sigma}D_{\lambda\sigma}
(\mu\lambda|\sigma\nu)$. These reduce to the existing real-AO formulas.
Real primitive integral permutation symmetry does not license eightfold
equality of complex Bloch tensors: retain pair exchange and conjugation
identities with their momentum labels.

For RHF, $D=2C_oC_o^\dagger$, $F=h+J[D]-K[D]/2$ and
$E_2=\tfrac12\operatorname{Tr}(DJ)-\tfrac14\operatorname{Tr}(DK)$.
For UHF use $J[D_\alpha+D_\beta]-K[D_s]$ in spin $s$ and
$E_2=\tfrac12\operatorname{Tr}(D_{tot}J)-
\tfrac12\sum_s\operatorname{Tr}(D_sK_s)$. Real parts remove roundoff only
after checking the imaginary residual. KS exchange fractions/ranges, XC and
ROHF effective-operator policy remain method-owned. Kernels never infer spin
factors from the number of density panels.

## 3. Matrix MD and direct digestion

### 3.1 Equation-to-kernel mapping

For primitive pairs $ab$ and $cd$, set
$p=\alpha+\beta$, $q=\gamma+\delta$,
$\mathbf P=(\alpha\mathbf A+\beta\mathbf B)/p$,
$\mathbf Q=(\gamma\mathbf C+\delta\mathbf D)/q$,
$\zeta=pq/(p+q)$ and $T=\zeta|\mathbf P-\mathbf Q|^2$.
Define derivative Hermites $\Lambda_{\mathbf t}=\partial_{\mathbf P}^{\mathbf t}
e^{-p|\mathbf r-\mathbf P|^2}$, with the Gaussian-product prefactor included
once in E. The one-dimensional recurrence is

$$
E_t^{a+1,b}=\frac{E_{t-1}^{ab}}{2p}+(P-A)E_t^{ab}
 +(t+1)E_{t+1}^{ab},\qquad
E_0^{00}=e^{-\alpha\beta(A-B)^2/p}.
$$

Multiplying the x/y/z E factors includes the full Gaussian prefactor. Zero
out-of-range E entries are exact zeros. Coulomb auxiliary integrals use

$$
F_n(T)=\int_0^1u^{2n}e^{-Tu^2}du,\quad
R_{000}^{n}=(-2\zeta)^nF_n(T),\quad
R_{\mathbf t+\mathbf e_i}^{n}
=t_iR_{\mathbf t-\mathbf e_i}^{n+1}+(P_i-Q_i)R_{\mathbf t}^{n+1}.
$$

Thus, with $c=2\pi^{5/2}/(pq\sqrt{p+q})$,

$$
I_{ab,cd}=c\sum_{\mathbf t\mathbf u}E_{ab,\mathbf t}
 (-1)^{|\mathbf u|}R_{\mathbf t+\mathbf u}^{0}E_{cd,\mathbf u}
 =[E_{ab}\mathcal R_{pq}E_{cd}^{T}]_{ab,cd}.
$$

This is the SHARK Eq. 21 structure and the explicit MD4 formulation of
Asadchev/Valeev Eqs. 6-15, under the normalization just defined. Validate the
ssss prefactor analytically before higher angular momenta. Do not copy a
prefactor or a power of p from extracted PDF text without reconciling the
Hermite convention. $n_H(L)=(L+1)(L+2)(L+3)/6$ controls the Hermite panel
size; compact R recurrence storage can be expanded into GEMM tiles on demand.

| Published mapping | Proposed kernel/consumer |
| --- | --- |
| SHARK Eqs. 11-24 | `pair_hermite_coefficients`, `coulomb_auxiliary`, `hermite_to_pair`; optionally fold pure transforms into E |
| SHARK Eqs. 25-28 | Primitive-tiled early contraction and grouped channel GEMM; never allocate the full published giant R matrix |
| SHARK Section 3.4 | Exact intermediate-basis transformation for partial-general contractions |
| SHARK Eqs. 29-37 | Multi-target J/K digestion with explicit permutations, diagonal coincidences and scale factors |
| SHARK Eqs. 38-41 | Split-J: $d_H=E^Td$, $v_H=\mathcal R d_H$, $j=Ev_H$ in local pair panels |
| SHARK Eqs. 42-50; BUPO Eqs. 12-17 | Two streamed RI passes and a metric solve, with the same pair/auxiliary transforms |
| SHARK Eq. 54 | Range-separated Boys seed below |
| BUPO Appendix Eqs. 37-49 | Gaussian Cartesian moments then existing-convention real solid-harmonic transform |
| Colinet Eqs. 17-21 | Upward multipoles and adjoint downward potential propagation for an optional tree consumer |
| GPU MD4 and GPU J papers | Later tile/recurrence-DAG schedules, measured against identical FP64 work |

2c Coulomb $(P|Q)$ and 3c $(P|\mu\nu)$ use one-function Hermite
expansions on the relevant sides, with dedicated arity/prefactor tests. They
are not obtained by inserting a normalized zero-exponent orbital into the
user basis. S, cross-basis S, kinetic, nuclear/point-charge potential, dipole
and higher multipole moments are required 1e kernels. Nuclear charge sign is
applied explicitly. AO-pair and auxiliary Fourier transforms are required
for periodic reciprocal partners. ECP continues through libecpint; XC grid
and AO-value kernels retain their own providers. Momentum, magnetic response,
relativistic corrections and Hessians are separate later capabilities.

The one-electron conventions can be checked without a four-center engine.
For a primitive pair with product coefficients E,
$S_{ab}=(\pi/p)^{3/2}E_{ab,000}$. For a nucleus of charge Z at C,
$V_{ab}=-Z(2\pi/p)\sum_{\mathbf t}E_{ab,\mathbf t}
R_{\mathbf t}^{0}(p,\mathbf P-\mathbf C)$, where the recurrence now uses
$\zeta=p$. Kinetic energy follows from
$\partial_x^2g_b=b_x(b_x-1)g_{b-2e_x}
-2\beta(2b_x+1)g_b+4\beta^2g_{b+2e_x}$ and its y/z counterparts,
with the overall $-1/2$. Gaussian moments use polynomial insertion into
overlap, and center derivatives use
$\partial_{A_x}g_a=2\alpha g_{a+e_x}-a_xg_{a-e_x}$.

Define Fourier transforms by $\widehat f(Q)=\int e^{-iQ\cdot r}f(r)dr$.
Then

$$\widehat\Lambda_{\mathbf t}(Q)=(-iQ)^{\mathbf t}
(\pi/p)^{3/2}e^{-Q^2/(4p)}e^{-iQ\cdot P}.$$

The opposite FT convention requires conjugating every relevant phase, not
just the final result. Test FT Hermite signs independently by differentiating
the analytic s Gaussian transform. The shared periodic adapters must state
which convention their raw and whitened tensor legs use.

### 3.2 Contractions and screening

Segmented batches group angular class, representation, kernel, derivative and
similar primitive counts. Small s/p cases may use specialized loops; GEMM
overhead is measured before choosing it. General contraction shares primitive
work across all matching radial channels, but tiles channels/primitives to a
budget. Partial-general contraction uses an intermediate basis $\chi$ with
$\phi=\chi C$: $D_\chi=CD_\phi C^\dagger$ and
$F_\phi=C^\dagger F_\chi C$. Transform density/target tiles locally where
possible; a huge fully decontracted global density is not required. This is
an exact reorganization, not an accuracy approximation.

Digestion keeps multiple right sides contiguous and fuses J and K when the
same integral tile is needed. A J-only task contracts density before the
second AO expansion; K's crossed indices generally require another schedule.
Permutation factors apply exactly once. Test all shell equalities separately;
a generic orbit multiplier is insufficient. Non-Hermitian response targets
must not be symmetrized or treated as SCF densities.

Use $|(ab|cd)|\le Q_{ab}Q_{cd}$ with shell bounds covering every admitted
component. Density-dependent screening bounds the *sum* of omitted target
contributions using absolute density envelopes, including all requested
right sides and exchange permutations. A maximum-element threshold per
quartet is not a global Fock-error bound. Separate primitive prefactor
screening, shell screening and final density screening in error receipts.
The zero-screening path remains available. General contraction and intermediate
densities need their actual transformed envelopes. Pointwise bounds on an
attenuated potential alone are not a proof for signed AO products; use
positive-kernel Coulomb-metric Schwarz bounds or separately derived bounds.

For $v(r)=f/r+g\,\operatorname{erf}(\omega r)/r$ set
$\kappa=\omega^2/(\omega^2+\zeta)$ and replace

$$F_n(T)\longmapsto fF_n(T)+g\kappa^{n+1/2}F_n(\kappa T).$$

erfc corresponds algebraically to $(f,g)=(1,-1)$, with a stable numerical
branch. Test zero/infinite range limits and full = erf + erfc. Physical
exchange range $\omega$ and the Ewald split parameter $\eta$ have distinct
identities. Unscreened solid-harmonic $1/r$ translation formulas cannot be
reused for erfc or erf merely by changing a label.

## 4. Molecular BUPO-J and RI-BUPO-J

### 4.1 Extents, moments and hierarchy

Published BUPO retains a coefficient-weighted contracted-pair center
(Eqs. 29-30), estimates the enclosing radial extent using two inward walkers
on the intercenter axis (0.01 bohr steps), sets a 2-bohr minimum, and accepts
far interactions when $R-e_A-e_B>R_{allow}$, using 1 bohr in the paper.
Its K-means bubbles enclose child centers **and extents**, with roughly 150
shell pairs per leaf and about three children per parent in the reported
settings. Diffuse objects can remain always near. These are published
implementation choices, not universal accuracy certificates.

Proposed MANTA starts with that molecular algorithm as a comparison mode,
then qualifies a conservative envelope for production. Signed contraction
coefficients can cancel in the published center denominator. Use a documented
nonnegative envelope-weighted center or a primitive-center representation when
the center is ill-conditioned; this is an explicit MANTA deviation. Bounds
sum absolute primitive coefficients and include angular polynomial/component
factors. Locate outer envelope crossings with brackets beyond all radial
maxima; an axial sample alone is not accepted as a proof of a 3D integral-tail
bound. Conservative primitive-sphere enclosure is the initial fallback.
Validate off-axis, nodal, signed, general and diffuse contractions.

Gaussian tails are never compactly supported. Record both a pointwise extent
criterion and an integrated Coulomb-tail/penetration estimate; they have
different units and meanings. A pair that cannot meet the tail budget stays
near, even if its plotted density is tiny. The 2-bohr minimum is a comparison
parameter, not a substitute for stable arithmetic at tight exponents.

For a real solid-harmonic basis of regular functions $R_{lm}$ define

$$M_{lm}^{a}(C)=\int R_{lm}(r-C)\rho_a(r)dr,\quad
 (a|b)_{FF}\simeq M_a^T T(C_a-C_b)M_b,\quad
 M(C_{parent})=W(C_{parent}-C_{child})M(C_{child}).$$

Use one tested normalization for R, irregular harmonics, T and W, matching
BUPO Eqs. 18-24 after conversion. Do not mix the factorial conventions in
Kudin 2004 with BUPO's scaled harmonics. M2M, M2L and L2L are distinct actions;
the downward action is the adjoint of the corresponding upward moment map.
Primitive pair harmonic moments at the product center terminate at
$l_a+l_b$; an atom-centered auxiliary shell terminates at its l. Contracted
two-center pairs and translated distributions generally do not terminate.
Finite moment termination does not eliminate Gaussian charge penetration.

Choose deterministic bubble clustering with stable IDs and recorded tie
handling. Store one object exactly once at the leaf level. Parent radius is
$\max_c(|C_c-C_p|+e_c)$. Source moments accumulate contracted density blocks;
retain complex moments if the supplied density/transfer requires them even
though angular transformations are real. Tree depth, order caps and inability
to meet a budget trigger descent/direct evaluation, never silent clipping.

For each target, an accepted parent replaces all its descendants. Otherwise
descend; near leaves enter the direct MD/libint consumer. Near and far lists
form a disjoint exhaustive partition of the same retained interactions.
Unordered pair traversal with reciprocal accumulation, or an explicitly
paired forward/adjoint schedule, is required for a self-adjoint J map.

For compact interior supports, a useful conservative kernel remainder is

$$|\delta v_L|\le\frac{\theta^{L+1}}{R-a-b},\qquad
\theta=(a+b)/R<1.$$

Multiply by appropriate absolute source/target charge norms, add separately
bounded Gaussian tails, and sum over accepted interactions to budget matrix
or energy error. This derives from the Legendre-series remainder also used
by Colinet Eq. 12; applying it to integrated signed Gaussian distributions
and a hierarchy is a proposed extension, not that paper's ready-made bound.
Successive-order differences are useful estimators but not rigorous bounds.
Up-translation must retain enough source moments for the requested parent
order; increasing an array length cannot restore discarded child moments.

### 4.2 Fitting and an energy-consistent two-pass contract

Flatten real AO pairs with explicit full-pair or packed-pair weights. Let
$B_{Pa}=(P|a)$ and $V_{PQ}=(P|Q)$ in the Coulomb metric. Exact RI-J is

$$b=Bd,\qquad Vc=b,\qquad j=B^Tc,\qquad
 E_J^{RI}=\tfrac12 b^TV^+b.$$

The pseudoinverse means an explicitly retained positive subspace, not
unconditional inversion. The initial small-system reference uses a pivoted
Cholesky or eigenprojector with documented absolute/relative cutoff. Preserve
an existing source's cutoff convention when adapting it. For large systems
offer streamed V actions and preconditioned solves under a residual bound;
retaining a dense $N_{aux}^2$ metric is an admitted small-system option, not
linear-scaling proof. Metric solve/conditioning error is separate from RI
basis error. Do not make the metric itself BUPO-approximate in the first RI
increment. A future approximate metric must remain positive on the admitted
space and be used consistently in both energy and derivative.

Published RI-BUPO builds b from pair bubbles, solves V, then builds J from
auxiliary bubbles. Atomic auxiliary leaves make one side's multipoles short
and terminating. The two passes have different measured sensitivities; the
paper varies their orders independently. MANTA proposes a stricter paired
contract: define a fixed approximate coupling $\widetilde B$ and use

$$\widetilde b=\widetilde B d,\quad
 \widetilde j=\widetilde B^T V^+\widetilde b,\quad
 \widetilde E_J=\tfrac12\widetilde b^TV^+\widetilde b.$$

The reverse pass is the adjoint of the same approximate coupling, even if
implemented using a different traversal. Different forward and reverse
couplings do not generally give the derivative of this energy. Jointly
choose orders/partitions or tighten a common coupling. For complex spaces
use the declared adjoint and pairing, not a blind transpose. Test
$\langle x,\widetilde B y\rangle=
\langle\widetilde B^\dagger x,y\rangle$ before SCF.

At fixed rank, a forward error $\delta b$ obeys the useful estimate
$|\delta E|\le\|V^+b\|\|\delta b\|+
\tfrac12\|V^+\|\|\delta b\|^2$. An ill-conditioned fit can amplify
small coupling errors. Measure pass-one and pass-two defects independently
against exact B, then the coupled result; do not assume the near additivity
observed in one published benchmark is universal. For exact Coulomb metric
and a fixed positive retained space, RI's variational residual interpretation
applies. Arbitrary approximate B need not preserve a bound relative to exact
four-center Coulomb energy, despite yielding a positive quadratic form.

A robust-fit energy $E=(\rho|\widetilde\rho)-
\tfrac12(\widetilde\rho|\widetilde\rho)$ is a possible alternative, with
its own derivative and residual terms. It is not silently substituted for
the stated paired-coupling functional. Charge-constrained/compensated
periodic fits likewise require their own declared equations and references.

### 4.3 Incremental builds and error ledger

For a fixed linear J map,
$J[D_n]=J[D_0]+\sum_{i=1}^nJ[D_i-D_{i-1}]$.
BUPO Eqs. 35-36 use this identity, with periodic full rebuilds (ten steps in
the reported tests) and tighter restart settings. Density screening, changing
orders and approximate solves introduce drift. MANTA tracks a cumulative
error estimate and chain identity per J/K spin slot and whole-mesh epoch.

Rebuild on interval/budget exhaustion, changed source/tree/rank/range/symmetry,
threshold tightening, restart, a large density step or drift/stagnation.
Before declaring SCF convergence, perform a nonincremental build at the
accepted density and final accuracy, recompute energies and commutator, and
reseed both density and J/K. If that check fails, continue SCF. Full here means
nonincremental at the selected approximation; a separate tight analytic/RI
comparison measures BUPO error. Build-plan density adaptation is frozen during
each operation and its nonlinearity is bounded/tested; it is not an exact
linear-response operator unless a fixed plan is used. Energy directional-
derivative and response tests use that fixed plan (or zero screening); the
separate adaptive-plan test measures the error introduced by changing it.

Report individual contributions from integral roundoff, primitive/shell and
density screening, Gaussian tails, multipole truncation/translation, auxiliary
basis, metric rank and solve, incremental drift, periodic image/G truncation,
symmetry reconstruction and SCF residual. Only proven compatible bounds may
be summed as a guaranteed budget. Empirical estimators are labeled as such.

## 5. Proposed periodic extension

Nothing in the molecular BUPO paper supplies the following implementation.
First validate MANTA kernels inside the existing same-source erfc/reciprocal
and fitted builders. Then introduce the periodic bubble construction below.

### 5.1 Bloch and whole-mesh contract

Choose translated real AOs $\chi_{\mu R}(r)=\chi_\mu(r-R)$ and Bloch sums
with $e^{+ik\cdot R}$. With the corresponding per-cell normalization,

$$A(k)=\sum_R e^{+ik\cdot R}A_{0R},\qquad
 D_{0R}=\sum_k w_k e^{-ik\cdot R}D(k),\qquad
 \sum_k w_k=1.$$

These signs match `_bloch_sum_blocks`; adapters to historical transposed pair
layouts must be explicit. $D_{0,-R}=D_{0R}^\dagger$. Do not drop imaginary
parts before checking this relation. A complete uniform finite character
mesh has an exact inverse transform; general quadrature weights or an IBZ
list do not automatically have one. chi's `_transform_inputs` already makes
this distinction. Electrons satisfy
$N_e=\sum_kw_k\operatorname{Tr}[D(k)S(k)]$; weights never enter orbital
normalization.

In complex chemists' notation,

$$J_{\mu\nu}(k)=\sum_{k'\lambda\sigma}w_{k'}D_{\lambda\sigma}(k')
(\mu k\,\nu k|\sigma k'\,\lambda k'),$$
$$K_{\mu\nu}(k)=\sum_{k'\lambda\sigma}w_{k'}D_{\lambda\sigma}(k')
(\mu k\,\lambda k'|\sigma k'\,\nu k).$$

J uses zero momentum transfer but depends on every density block. K samples
$q=k'-k$ modulo a reciprocal vector. Store both canonical q and its integer
reciprocal wrap, bra/ket k indices and exact atom-image shifts. Two q-equivalent
pairs can share a metric without sharing their AO-pair phases. The existing
native B convention is $q=k_{ket}-k_{bra}$,
$K(k_{ket})=\sum_{k_{bra}P}w_{bra}B_P^\dagger D(k_{bra})B_P$;
document the axis/conjugation conversion to the equations above.

Atom relabeling by an integer cell vector changes Bloch basis phases. If
$\phi'=\phi U(k)$, require $A'=U^\dagger AU$ and
$D'=U^{-1}D U^{-\dagger}$. Nonorthogonal Cartesian symmetry requires this
distinction between operator pullback and density pushforward. Preserve
fractional translations through exact atom images; do not add a second
global phase already present in that transport. Time reversal is an admitted
antiunitary action, not an assumed property of arbitrary trial densities.

One build consumes one immutable all-k/all-spin epoch. Stream density-image
and q batches, but never combine iterations from different k points. IBZ
execution requires separately admitted source and state symmetry and correct
star weights. Changing just one k density must affect the appropriate J/K
blocks elsewhere in the mesh.

The explicit whole-mesh RI consumer stores one fixed Gamma source and one
source for each ordered off-diagonal target/source pair. For target i and
source j it uses q=k_j-k_i and T(k_i,k_j); its exchange contribution is
`w_j sum_L B_L D_spin(k_j) B_L^H`. The Gamma source supplies J and diagonal
exchange. The supplied pair segments have empty diagonals. This sign is
opposite the direct reciprocal exchange vector interface's target-minus-source
convention. Converting that source requires negating the complete vectors,
then omitting its diagonal segments, not changing q alone.

Admission checks exact reciprocal labels, inversion/reverse-transfer closure,
matching retained ranks and numerical conjugacy of M, P and F F^H. It does not
average matrices or silently wrap q to share a finite domain. Fixed explicit
source, frame, weights and evaluated density have separate identities. H and
an optional exchange scale belong to the complete Fock receipt. A method-owned
singular exchange correction changes the operator source and snapshot, never
the identity of an unchanged evaluated density in its frame.

The neutral all-electron RI UHF method reuses the physical one-electron and
nuclear Ewald sources. Its numerical fitting split is independent of the
nuclear Ewald split. The AO pair cutoffs must coincide; fitting auxiliary-image
support remains separate. The existing fixed-spin solver still requires
uniform full regular meshes, integer populations and virtual space in both
spins. Its density closure, energy, population, gap and independent terminal
Fock confirmation remain method gates. A generic Fock object's commutator
diagnostic cannot replace them. This GDF source neither changes BIPOLE's AC/BD
supports nor qualifies reduction of a fitted source from geometry alone.

The RI UHF cutoff factory generates the complete union of AO-pair image
spheres, including independent unwrapped AO centers. It reuses the certified
reciprocal energy-sphere enumerator and its integer-label conjugacy audit.
For each off-diagonal target/source pair it negates the whole direct-exchange
physical vector to obtain the fitted ket-minus-bra convention. Diagonal pair
segments stay empty; Gamma owns their metric/coupling source. This conversion
does not identify or share distinct finite q sources. The geometric domain
identity is separate from auxiliary/fitting/projector and evaluated-density
identities. Its joint plan admits generated/exported supports, every retained
fit owner and the complete bounded SCF before enumeration. Auxiliary and
nuclear image radii remain separate; generated spheres certify membership,
not convergence with respect to their cutoffs.

Public RI-containing HF selection accepts independent exact/RI J and K choices.
An auxiliary basis or an RI-J request does not implicitly replace exact K.
Periodic fitting controls supply the numerical split and auxiliary image
domain; the physical AO pair radius is shared with the one-electron source,
and the Coulomb selector owns the metric projection controls. The job records
the complete fixed fit separately from the method's singular correction,
evaluated density and canonical next orbital projector. QVF exports that
evaluated spin density in the nuclear-center frame, including capped states.
Public admission and serialization do not change the method's convergence
tests, infer a spatial subgroup or reinterpret BIPOLE's finite source.

Restricted RI HF embeds the spin-summed density as two equal halves in the
same immutable fitted source and solves the restricted generalized problem.
Its physical Fock is `H + J[D] - K[D]/2`; the separately selected BvK term is
`-xi*S*D*S/2`. Each two-electron energy, including that correction, is the
weighted half trace with the evaluated total D. Restricted idempotency is
`D*S*D = 2*D`, with the RHF leaf's own retained-space residual and closure
criteria. One restricted density/Fock is stored by the solver; admission
still counts the two-spin callback and all fixed auxiliary/projector owners.
Equal response channels are checked, never averaged. Restricted and singlet
unrestricted methods can share a fitted source but keep distinct method
identities and stopping rules. QVF stores equal spin halves of the evaluated
RHF density, preserving capped states and explicit source-image transport.

Mixed exact/RI HF composes one fixed fitting target with the complementary
direct reciprocal response inside the same admitted native method callback.
Both see the same evaluated spin density; HCORE and the method-owned BvK
correction enter once. Each component contributes its own weighted energy
and retains its actual finite support. RI K uses ket-minus-bra transfers and
empty diagonal pair segments owned by Gamma. Exact K uses
`Q=G+k_target-k_source`, explicit nonempty primed diagonal segments and no
implicit zero-vector insertion. Cutoff factories generate the selected
convention; explicit callers must supply it. RI-J/exact-K retains only the
Gamma fitting projector, whereas exact-J/RI-K retains every required transfer
projector. Direct and fitted sources need not agree at finite cutoffs.

The aggregate method plan admits fitting, direct and composition owners,
including repeated builds and terminal confirmation. Its fitting subplan is
not the whole mixed-method memory bound. Receipts retain independent J/K
source identities and optional direct-component evidence; the composed source
identity does not relabel the evaluated density or AO frame. Method energy
and owned Fock exports preserve this distinction. Composition grants no
BIPOLE/CRYSTAL parity or symmetry admission beyond the component contracts.

Explicit fitted pure and global LDA/GGA KS now composes the same selected
fixed-projector sources inside actual RKS/UKS functional callbacks. Pure KS
uses only fitted J; globals independently select exact/RI J/K with at least
one fit. RKS embeds total D as equal spin halves only for fitting and retains
total D for unpolarized XC. UKS keeps its actual spin populations. The
functional energy is nuclear plus one-electron, J, XC, method-scaled K and
its separately recorded singular correction. An HF half trace of the final
KS matrix is not this energy. Projectors stay fixed through confirmation;
numerical fitting omega cannot stand in for a physical range kernel.

Joint admission includes retained fitting, complementary direct responses,
XC panels, embedding and confirmation, without replacing the method's
solver leaf or convergence norm. The fitted child receipt is marked as a
component: its embedded density and partial energy cannot be mislabeled as
the complete KS state. Method density identity version 2 binds actual D,
spin layout, weights and overlap frame independently of interaction source.
The associated Fock snapshot additionally binds interaction, one-electron
and nuclear sources. Direct KS adopts the same identity semantics without
changing its numerical states or original resource counts.

Shared FockBuild exports own the evaluated D/F with a KS kind, method energy
and explicit, possibly rectangular S/X frame. Shared symmetry transport
checks AO-image covariance; they do not prove electronic-state, grid, fitted
projector or source symmetry. Explicit fitted sources retain the full mesh
and the independent BIPOLE/CRYSTAL parity boundary. Generated cutoff and joint
quadrature sources now share that contract. Their geometry plan calls the
actual fitted functional planner, never an HF proxy, before allocating sources
or quadrature. The complete admission retains domain/grid payloads, fixed
projectors, XC, complementary direct components and confirmation. Pure fitting
has no K table; the selected global K controls opposite transfer orientations
and Gamma versus pair-diagonal ownership. Geometry, physical source and
evaluated state remain distinct. Public fitted KS jobs use these admitted
native sources with resolved auxiliary/metric/fitting controls. Pure RI J
retains no K; globals require explicit RI K selection. Output admission adds
all Python, history, basis and serialization owners before generation. Method
result types and complete energies remain KS; the partial fitted child cannot
replace them. QVF records actual evaluated spin densities in the nuclear AO
frame, including capped iterations and orbital-image conversion. Physical
range fitting follows the same job contract with ordered identities for each
resolved physical fitted/exchange source. No source reduction is implied.

### 5.2 Periodic Green function and disjoint partitions

For 3D cell volume $\Omega$, choose the zero-average conducting-boundary
periodic Coulomb Green function. Away from its singularities it can be written

$$G_\eta(r)=\sum_R\frac{\operatorname{erfc}(\eta|r+R|)}{|r+R|}
+\frac{4\pi}{\Omega}\sum_{G\ne0}
 \frac{e^{-G^2/(4\eta^2)}}{G^2}e^{iG\cdot r}
-\frac{\pi}{\eta^2\Omega}.$$

Here $G$ in a sum is a reciprocal vector; $G_\eta$ is the Green function.
The constant cancels the cell-average real-space term. Its Poisson equation is
$-\nabla^2G_\eta=4\pi(\delta_{per}-1/\Omega)$.
Neutrality is the default for the complete electron+nuclear system. A charged
cell requires explicit uniform-background policy; otherwise refuse. Individual
electron/electron and electron/nuclear terms use the same Green convention,
even though each source is charged. The raw Ewald total-charge correction
$-\pi Q^2/(2\eta^2\Omega)$ is already represented by the constant above;
do not add it twice. Remove nuclear point self terms consistently
($-\eta\sum_A Z_A^2/\sqrt\pi$ in the usual split energy). Gaussian
electron self interactions are not nuclear point self terms.

There are two exclusive implementations of this same limiting Green function:

1. **Initial kernel adapter:** direct erfc real-space shell tiles plus the
   matching reciprocal partner and gauge/self terms. Finite supports and
   cutoffs are the existing source's. A full-Coulomb bubble far field cannot
   be added on top of this complete sum. Screened multipole kernels, if later
   used here, must be derived from screened radial derivatives.
2. **Proposed periodic BUPO:** select a symmetric finite near-cell set
   $\mathcal N$, large enough to enclose interacting Gaussian extents, and use
   the exact identity

   $$G_\eta(r)=\sum_{R\in\mathcal N}|r+R|^{-1}+G_{far,\mathcal N}(r),$$
   $$G_{far,\mathcal N}=G_\eta-\sum_{R\in\mathcal N}|r+R|^{-1}.$$

   Treat interactions in $\mathcal N$ with a molecular-style direct/bubble
   partition. Evaluate a lattice M2L expansion for the complementary field,
   obtained by differentiating the Ewald expression and subtracting those
   **same** near-cell bare interactions. Reciprocal terms thus occur once,
   in the tail construction. Every accepted local bubble and tail interaction
   has a unique source/target/image identity. Near cells and the lattice tail
   never overlap. Varying $\mathcal N$ transfers terms between the two parts;
   it must not change the converged result.

There is an additional constraint often missed by a harmonic-only design.
In a singularity-free near region $\nabla^2G_{far,\mathcal N}=4\pi/\Omega$,
so this field is not purely harmonic. Write locally

$$G_{far,\mathcal N}(r)=H_{\mathcal N}(r)+\frac{2\pi}{3\Omega}|r|^2,
\qquad \nabla^2H_{\mathcal N}=0.$$

The quadratic term requires monopole, dipole and **trace second moment**, in
addition to traceless solid harmonics. For distributions with moments
$Q_a=\int\rho_a$, $m_a=\int r\rho_a$ and $s_a=\int r^2\rho_a$ in a
common origin, its pair contribution is

$$\frac{2\pi}{3\Omega}(s_aQ_b+Q_as_b-2m_a\cdot m_b).$$

Retain the constant potential gauge as well. Translation tests must include
these polynomial terms; finite total-charge cancellation cannot justify
dropping them from separately reported J and nuclear matrices. Combining
electron and nuclear moments can improve conditioning, but preserve the
component energy ledger. This polynomial-plus-harmonic decomposition is a
proposed derivation from the declared Green function, not a BUPO paper formula.

For an alternative direct-lattice implementation, Kudin/Scuseria 2004
Eqs. 23-24 give $S_0=L_*$,
$S_{n+1}=\mathcal U_L(S_n)\otimes M_*+L_*$, with
$\mathcal U_L(L_{lm})=3^{-(l+1)}L_{lm}$ and M2L contraction denoted by
$\otimes$. Eqs. 30-31 add cell-index-weighted stress sums. Reuse that
recurrence only after converting moment conventions, excluding the identical
near-cell set, and resolving low-order conditional sums/surface/gauge terms.
Convergence of high-l lattice coefficients does not resolve a charged monopole
divergence or a surface dipole by itself. Ewald-derived coefficients are the
initial reference; direct renormalization is a later independent schedule.

In both cases, Gaussian penetration/tails, finite pair-image supports, lattice
coefficient convergence, real/G Ewald truncation and multipole truncation have
separate controls. Derivatives of the full Green function should be obtained
before numerical truncation choices are frozen for a force comparison.
Strongly skewed cells and diffuse bases can require larger near sets; runtime
is not universally linear. Infinite tail coefficients may be precomputed per
cell/kernel/boundary, not per density. Their workspace/order is budgeted.

The first bounded prerequisite is now `multipole_lattice_tail`: finite
explicit Ewald domains and a contained near set, raw Cartesian M2L through
total degree twelve, with the constant gauge and trace terms retained.
Near images use the analytic `-erf/r` complement even at the origin; other
images use upper incomplete gamma radial derivatives. Domain canonicalization,
recurrences and result exports have count-only admission. Kernel and expansion
identities are distinct. This qualifies finite coefficients, not a production
periodic BUPO traversal, infinite-sum error bound, Gaussian support
certificate, symmetry shortcut, complete charge ledger or nuclear self policy.

#### Finite-source compact Taylor remainder

`lattice_tail_remainder` now bounds the order-$p$ Taylor remainder of this
same **finite** kernel, conditional on enclosing radii $a,b$ and absolute
charge integrals $Q_a,Q_b$ supplied by the caller. These can describe signed
compact densities or finite measures. Set $n=p+1$, $\rho=a+b$, and, for each
supplied real image, $D=|r+R|$, $d=\max(0,D-\rho)$, $M=D+\rho$.
Non-near images require $d>0$. Near images have no remaining singularity.
The real-image contribution is bounded by

$$B_R=\frac{2\rho^n}{\sqrt\pi}
\sum_{k=0}^{\lfloor n/2\rfloor}
\frac{2^{n-2k}M^{n-2k}}{k!(n-2k)!}
\int_I s^{2(n-k)}e^{-d^2s^2}\,ds,$$

where $I=[0,\eta]$ for the near $-\operatorname{erf}/r$ complement and
$I=[\eta,\infty)$ for the other real images. This follows from the
one-dimensional Taylor theorem along the actual displacement $h$, with
$|h|\le\rho$. Differentiating $\exp(-s^2|x+t u|^2)$, $|u|=1$, gives
terms of magnitude at most
$n!e^{-s^2d^2}(2s^2M)^{n-2k}s^{2k}/[k!(n-2k)!]$.
The Taylor $1/n!$ cancels this prefactor. Integration and the triangle
inequality give the displayed bound; no harmonic-kernel assumption enters.
Each supplied reciprocal term contributes at most

$$B_G=\frac{4\pi}{\Omega G^2}e^{-G^2/(4\eta^2)}
\frac{(\rho|G|)^n}{n!}.$$

The constant gauge is reproduced exactly by the order-zero coefficient and
has zero Taylor remainder. The pair-energy bound is
$Q_aQ_b(\sum_R B_R+\sum_G B_G)$, with separate near, non-near real and
reciprocal fields. Absolute charge norms, never signed net charges, multiply
the kernel bound. In particular a neutral component still has a nonzero bound.

Positive incomplete-gamma envelopes avoid subtraction and premature underflow.
The lower integral uses a positive series with a geometric upper bound on its
uncomputed tail, or the complete gamma as an upper bound. The upper integral
starts from $\Gamma(1/2,x)\le\min(\sqrt\pi,e^{-x}/\sqrt{x})$ and recurs
upward with positive terms. A decay argument above $10^6$ is conservatively
replaced by $10^6$. Calculations remain logarithmic through charge scaling;
positive subnormal results round upward instead of becoming a false zero.
Geometric and final relative guards cover routine evaluation noise, but these
are not an interval arithmetic proof for the platform's transcendental functions.

The `finite-domain-compact-taylor` qualification is an analytical approximation
bound, separate from coefficient/moment/contraction roundoff. It does not
verify the caller's support/norm claims, bound omitted Ewald images or Gaussian
tails, or certify a full Gaussian-moment substitution. Its `expansion_identity`
matches the exact `LatticeTailMatrix.source_identity`; its own identity also
binds both radii, both norms and the bound algorithm. No symmetry skipping,
Fock producer or production periodic BUPO route is admitted by this primitive.

#### Finite-source Gaussian approximation ledger

`lattice_gaussian_interaction` pairs the identical finite coefficient matrix
and compact bound with existing native `PairExtent` or `AuxiliaryExtent`
envelopes. Both producers must retain full and tail absolute radial moments
through the requested order. Their centers define the expansion displacement;
an `always_near` envelope or a non-near singularity inside the combined
support requires direct evaluation. The compact bound uses the full absolute
masses as upper bounds on the interior masses.

Write $Q_A,Q_B$ for full absolute masses, $t_A,t_B$ for exterior absolute
masses and $V_A,V_B$ for global absolute Coulomb potential bounds. With
$N_{\rm out}=|\mathcal R\setminus\mathcal N|$, define

$$
B=|\mathcal N|\frac{2\eta}{\sqrt\pi}
 +\sum_{G\in\mathcal G}\frac{4\pi}{VG^2}e^{-G^2/(4\eta^2)}
 +\frac{\pi}{\eta^2V}.
$$

The difference between the full and interior Gaussian interactions is bounded
by

$$
E_{\rm penetration}\le B(t_AQ_B+Q_At_B)
 +N_{\rm out}(t_AV_B+t_BV_A).
$$

This follows by writing the density-product difference as tail times full
plus interior times tail. Each near $-\operatorname{erf}(\eta r)/r$ term is
globally bounded by $2\eta/\sqrt\pi$; reciprocal cosines and the constant
gauge use their absolute amplitudes. Each non-near erfc term is nonnegative
and no larger than bare Coulomb, so a translated tail is bounded by its mass
times the other source's global potential bound. This derivation uses the
finite complementary partition in section 5.2, including the gauge once.

If $f_A[j],t_A[j]$ bound full and exterior radial absolute moments, replacing
interior moments by full moments contributes

$$
E_{\rm moments}\le\sum_{ij}|T_{ij}|
 \bigl(t_A[|i|]f_B[|j|]+f_A[|i|]t_B[|j|]\bigr).
$$

All Cartesian matrix entries, including traces and the constant, are used.
The three fields `compact_remainder`, `tail_penetration` and
`full_moment_tail` remain distinct and sum upward to `total`. A vanishing
signed monopole does not replace an absolute norm by zero. Larger orders can
increase a conservative bound for broad supports; a consumer must check the
actual result and refine or evaluate directly when its target is not met.

Joint count-only admission precedes input reads/copies and covers both native
components, reciprocal sorting, retained envelopes, the matrix and export
peak. Results own their coefficient/remainder pairing. Envelope identities
fingerprint the consumed numerical bounds, centers, roles and shell indices;
they are not basis identities or authorization for a Fock or symmetry source.
`finite-domain-gaussian-approximation` excludes omitted lattice/pair-image
domains, floating-point coefficient/moment/contraction errors and point-nuclear
self policies. Positive log-domain accumulation has conservative finite guards
but is not an interval-libm certificate. Production periodic BUPO/RI-BUPO
must still aggregate these bounds and bind every source before using them.

#### Infinite Ewald domain remainder for compact sources

`lattice_domain_remainder` now bounds the infinite Ewald Green function
minus its explicit finite real/reciprocal domains. Both domains must be
complete centered integer cubes, of radii $N_R,N_G$ in the maximum norm;
the reciprocal cube excludes zero. Canonicalization proves uniqueness,
origin convention and inversion closure before maximum coordinate and count
prove cube completeness. An arbitrary inversion-closed subset is refused.
The identical near-cell subtraction cancels from the infinite-minus-finite
difference. The result binds the existing finite kernel identity, including
its near set, without changing its gauge or point-self policy.

Let $s_R>0$ satisfy $|An|\ge s_R|n|$, and let
$s_G=2\pi/\|A\|_F$, a lower bound for the reciprocal lattice metric.
For a computed inverse $B$, the real bound uses

$$s_R\le\frac{1-\|I-BA\|_F}{\|B\|_F},$$

and refuses a nonpositive residual bound. This follows by applying
$I=BA+(I-BA)$ to any vector; it does not assume that a computed inverse is
exact. Guarded norms and outward distance rounding protect the analytical
inequalities from routine evaluation noise, but are not interval arithmetic.
For compact supports of radii $a,b$, set
$B_r=|r|+a+b$, $m=N_R+1$, and $d=s_Rm-B_r$. Require $d>0$ so every
omitted real image is separated from the entire difference support.

The shell $|n|_\infty=j$ contains exactly $24j^2+2$ points. Using
$\operatorname{erfc}(x)\le e^{-x^2}/(\sqrt\pi x)$ and the decreasing
shell prefactor, the omitted real kernel is bounded by

$$U_R=\frac{24m^2+2}{\sqrt\pi\eta d^2}e^{-\eta^2d^2}
\left(1+\frac{1}{2\eta^2s_Rd}\right).$$

The parenthesis bounds a positive decreasing Gaussian series by its first
term plus its integral, applying the same erfc inequality to that integral.
For $n=N_G+1$ and $c=s_G/(2\eta)$, reciprocal phases have unit modulus,
giving the independent bound

$$U_G=\frac{4\pi}{\Omega s_G^2}\left(24+\frac{2}{n^2}\right)
e^{-c^2n^2}\left(1+\frac{1}{2c^2n}\right).$$

The pair-energy bounds are $Q_aQ_bU_R$ and $Q_aQ_bU_G$, using absolute
charge norms even for neutral signed distributions. The constant gauge is
identical in both kernels. The shell-envelope derivation is a MANTA bound
on Ewald's split, not a formula attributed to the original paper. Ewald
(1921), DOI 10.1002/andp.19213690304, section IV.1, Eqs. 37-40 supplies
the neutral split and regular point-self reference used by the energy tests.
The separately declared zero-average component convention remains in force.

Logarithmic accumulation and conservative decay caps keep positive subnormal
bounds positive; unrepresentable bounds are refused. Zero absolute charge
norms give zero only after all domain and support checks. Count-only plans
admit sorting, cube validation, metric work, copies and exports before input
payload reads. No omitted cells are enumerated at runtime.

For translated compact sources, two corresponding domain bounds limit the
finite-kernel periodicity defect by the triangle inequality through the same
infinite Green function. Reciprocal terms are already periodic, so only the
real bounds are needed for a pure lattice translation. Finite real cubes
are not declared exactly covariant. These bounds can be added to the compact
Taylor remainder, with both errors and source identities kept separate.
The qualification `compact-infinite-ewald-domain-remainder` does not cover
unbounded Gaussian tails, finite AO pair-image omission, moment/roundoff
errors, automatic source support certification or a complete SCF profile.
Those remain necessary before production periodic BUPO admission.

#### Full Gaussian infinite-domain bounds

`compute_pair_gaussian_majorant` and `compute_auxiliary_gaussian_majorant`
produce immutable positive spherical Gaussian sums that dominate the absolute
value of every AO component of one selected source over all space. Orbital
shells support angular momentum through 6 and auxiliary shells through 7,
including Cartesian and pure representations and signed contractions. Pair
translation uses the same represented ket center as `PeriodicPairSource`.
These are full-source envelopes; an extent radius is not used as compact
support. Basis, selected shells, translation and actual envelope terms bind
an immutable native source identity. There is no numerical envelope constructor.

For each primitive product the Gaussian product theorem and the maximum
row sum of the Cartesian-to-pure transformation give

$$|p_i(x)|\le e^w\sum_n c_n r_P^n e^{-p r_P^2},\qquad c_n\ge0.$$

Choose $\gamma=9p/16$ and $\alpha$ just below $p/2$. With
$M_0=1$ and $M_n=[n/(2e(p-\gamma))]^{n/2}$, each polynomial term is
bounded by $M_n e^{-\gamma r_P^2}$. For a stored center $C$ with guarded
coordinate error $\delta\ge|P-C|$, completing the square gives

$$e^{-\gamma|x-P|^2}\le
 e^{\gamma\alpha\delta^2/(\gamma-\alpha)}e^{-\alpha|x-C|^2}.$$

Thus the normalized majorant $Q(\alpha/\pi)^{3/2}e^{-\alpha|x-C|^2}$
uses $Q\ge e^w\sum_n c_nM_n\,e^{\gamma\alpha\delta^2/(\gamma-\alpha)}
(\pi/\alpha)^{3/2}$. Auxiliary primitives need no product-center shift.
Absolute primitive coefficients prevent cancellation from understating the
bound. Logarithmic sums, outward charge conversion and positive underflow
handling are retained. Unrepresentable decay or charge is refused. The
coordinate guards are conservative floating-point headroom, not a claim of
interval certification.

For two normalized majorants define
$\beta^{-1}=\alpha_a^{-1}+\alpha_b^{-1}$ and
$\zeta^{-2}=\eta^{-2}+\alpha_a^{-1}+\alpha_b^{-1}$. Their full-space
convolution with the Ewald real kernel is

$$Q_aQ_b\frac{\operatorname{erf}(\sqrt\beta d)-\operatorname{erf}(\zeta d)}{d}
\le Q_aQ_b\frac{\operatorname{erfc}(\zeta d)}{d}.$$

`lattice_gaussian_domain_remainder` applies the preceding cube-shell real
bound with this effective decay and each pair of majorant centers, then adds
all positive terms. Omitted images must be separated from these centers in
the conservative lattice metric. This condition concerns centers, not bounded
Gaussian support. Reciprocal terms use the physical $\eta$ and the product
of total majorant charges, since a Fourier transform is bounded by its
absolute charge norm. The effective real decay does not change the physical
kernel, its gauge, near subtraction, source identity or self convention.
Canonical domain validation and sorting run once, not once per primitive pair.

This convolution and envelope derivation extends the Ewald split; it is not
attributed to Ewald's original point-charge formula. The existing Gaussian
product and Ewald references apply (McMurchie-Davidson 1978; Ewald 1921,
section IV.1, Eqs. 37-40). Complete smooth-charge energies include finite
Gaussian self integrals, without a point-charge self subtraction.

Qualification `native-gaussian-infinite-ewald-domain-remainder` covers one
component interaction of the full native pair/auxiliary source. It can be
added to the finite Gaussian approximation ledger for matching sources and
identical kernel domains. A consumer must still bind the moment source to
these same native producers and account for density contraction multiplicity.
It does not bound omitted AO pair images, roundoff, a whole Fock build or a
selected SCF accuracy profile. Production periodic BUPO/RI-BUPO traversal,
charge policy, symmetry, lower-dimensional Green functions and derivatives
remain distinct gates. The shared FockBuild/JKBuilder UHF interface and
BIPOLE physical AC/BD/exchange domain ownership are unchanged.

#### Periodic pair-density source and its adjoint

`PeriodicPairSource` now owns the finite orbital pair-image inventory needed
by a q=0 periodic Coulomb consumer. It retains the native basis, cell, integer
image set, momenta, weights, physical pair cutoff and Gaussian extent policy.
The image set is unique, inversion closed and contains the origin. A physical
pair cutoff is applied once to each conjugate orbit, retaining or omitting
both directions together. This is independent of BIPOLE AC/BD/exchange
domains and grants no alternative interpretation of those domains.

For real orbital functions, let
$p_{abR}(r)=\phi_a(r)\phi_b(r-R)$. Its partner obeys
$p_{ba,-R}(r)=p_{abR}(r+R)$. Their integrals against a periodic potential
are equal. In the q=0 specialization of Sun et al. (2017), Eq. 16,
DOI 10.1063/1.4998644, the corresponding density coefficients are

$$c_{abR}=\sum_k w_kD_{ab}(k)e^{-ik\cdot R},\qquad
c_{ba,-R}=c_{abR}^*.$$

The second equality uses Hermiticity at each k, with no relation between
different k points. Each two-member orbit therefore contributes
$2\operatorname{Re}c_{abR}$ times one representative product. A diagonal
shell at the origin keeps its full AO block with real density coefficients;
the imaginary antisymmetric part cancels against symmetric product integrals.
No spatial subgroup or time-reversal reduction is admitted by this identity.
Arbitrary nonreciprocal Fourier arguments and finite nonperiodic Green
functions do not inherit the translation identity. A consuming approximation
must separately establish its source, covariance and omitted-domain errors.

The source generates native shifted-shell extents without copying an image
supercell basis. It streams one full Cartesian moment panel at a time about
the representative pair midpoint. Full and tail absolute radial bounds are
retained through the requested order, including trace moments. The forward
map contracts the paired density coefficients into real leaf moments.
Its charge is the signed integrated AO density of this finite pair source;
it neither imposes neutrality nor adds nuclei or a background correction.
Per-pair coefficient L1 norms and the resulting absolute-charge envelope
remain distinct from signed monopoles.

For real local moment potentials $L_p$, the adjoint generates the AO
response $V(k)$ with the original Bloch phases and satisfies

$$\sum_p M_p[D]^T L_p
=\sum_k w_k\operatorname{Re}\operatorname{tr}[D(k)V(k)].$$

Weights enter the forward contraction, not the returned AO response. Both
members of each orbit are restored in the adjoint, including their distinct
AO index order. Native Hermiticity is checked without averaging matrices.
Zero-weight k points retain response values. The source, evaluated density
and local-potential identities are separate; an adjoint response alone is
not a complete FockBuild or a method energy.

Count-only admission covers the retained source and one forward/adjoint
cycle, including full moment panels, density/response copies and exports.
It is not a lifetime call quota; a method must budget repeated passes and
other retained objects. The existing molecular bubble tree still requires
its complete triangular shell inventory and refuses these periodic images.
The typed `PeriodicBubbleTree` now preserves the explicit entry/image mapping.
The qualification `finite-pair-image-moments-and-adjoint` does not select
periodic BUPO, exchange, fitting, SCF, nuclear self terms or an accuracy
profile. Those consumers must bind this source and their own kernel ledgers.

### 5.3 Fitting, exchange and dimensionality

The implemented raw short-range GDF substitution retains the existing
physical auxiliary sphere and AO-pair/auxiliary-segment capsule domains.
MANTA replaces only contracted erfc integral evaluation. The metric phase is
`exp(+i q.R)` and the coupling phase is `exp(+i k.R-i q.T)` for
`T(k-q,k)`. Shared-q batches reuse molecular tiles; source-k weights enter
later method contractions, never these raw blocks. Included finite erfc zero
modes are removed once by a matching jellium assembler, separately from
exchange singular corrections. Optional Gaussian-envelope screening uses the
existing density-independent rule, recorded in source identity; it is not a
symmetry qualification. The method retains ownership of the fitted projector,
charge ledger, full Fock/energy build and derivative source. A raw finite
metric is not automatically a positive retained-space fit or a BIPOLE source.

The complete finite Coulomb source adds streamed auxiliary and AO-pair Fourier
panels on an explicit `G+q` domain. It uses the same physical AO-pair cutoff as
its short-range component, the existing GDF jellium kernel and no k weights.
The finite erfc zero is removed exactly once for reciprocal Gamma transfers,
even when the supplied primed reciprocal list omits zero. Zero in that list
adds no long-range contribution. Independent bare-Coulomb reciprocal sums
check converged numerical splitting. Metric and coupling identities retain
both the actual SR source and the ordered reciprocal domain. These complete
integral components remain unfactored; no retained subspace, response operator,
exchange correction or symmetry certificate is selected by the provider.

The immutable shell-tile form is the identical finite source, with the same
component identities. A separate q-channel RI consumer fixes the lower-triangle
Hermitian positive-eigenvalue projector at construction, refuses significant
negative eigenvalues and never refits by density. It streams `B=F^H T` in
bounded retained-mode blocks. Gamma J uses weighted total spin density and
returns an unweighted target response with energy `0.5*|rho|^2`. Each K output
is `w_k B(k) D_spin(k) B(k)^H` in the bra frame k-q, indexed by its source ket
batch. It is not a complete exchange sum. Rank truncation, the exact factor
and projector, raw source, weights and evaluated density have separate
identities. The frame also includes the lattice and actual momenta. Metric
residuals diagnose the chosen finite projected equation, not approximation
accuracy. The consumer admits all repeated shell traversals jointly and
retains neither complete T nor complete B.

The exact J-only baseline can contract the same raw tiles in a paired
forward/adjoint schedule: `b=sum_kij w_k conj(T_kPij) D_total,kij`,
`rho=F^T b`, `c=conj(F) rho`, and `J_kij=sum_P T_kPij c_P`, with
`E_J=0.5*|rho|^2`. The two sweeps share the immutable coupling and fixed
native factor, including its actual finite `F^H M F`; no second whitening or
rank change is allowed. This factor convention requires the displayed
conjugations. It implements the paired contract of Section 4 on the existing
Sun periodic source, without an approximate bubble hierarchy.

Select it only for J without K when the retained-mode schedule would exceed
two passes. Fixed rank/block metadata supplies this decision; density, timing
profiles and geometric symmetry do not. Joint J/K and K-only keep their
existing traversal. Actual coupling receipts record `paired-raw`,
`retained-mode` or `none`; source identities remain independent of execution
schedule. The original maximum-rank joint admission remains conservative.
Weighted reciprocity, energy derivatives, finite-rank image transport and
complete evaluated method states qualify this reordering, rather than a
claim of bitwise equality or periodic RI-BUPO completion.

`PeriodicGDFTiles.forward(X)` and `.adjoint(c)` now expose that raw paired
coupling as producer-owned actions. They compute `sum_kij conj(T_kPij) X_kij`
and `sum_P T_kPij c_P`, respectively, under the ordinary complex Frobenius
pairing. Arbitrary complex transition inputs are valid at any admitted q;
the raw producer does not require Hermiticity or insert k weights, spin sums,
metric solves, canonical image folding or a symmetry operation.

Each action streams the existing ordered tiles and admits one complete pass,
including input/export copies and compensated output. A method must jointly
admit repeated passes and its other live state. Paired RI-J supplies the
weighted total spin density to the forward action and applies its original
finite metric factor between actions. Source/factor/projector identities and
the conservative joint RI plan retain their meanings. The new coupling
receipt identifies the raw source and arbitrary input, not an SCF density.

The optional `coupling_screen_error=epsilon` introduces a first fixed
approximate coupling using the existing `ShellEnvelope`, `GaussianSRBound`
and `ThreeCenterImageDomain`. It is mutually exclusive with common metric/
coupling screening. Only the three-center selector receives epsilon; metric
evaluation and its source identity remain unchanged. For each ordered shell
triple, divide epsilon by the conservative pair/auxiliary image candidate
counts. The shared selector bounds each omitted component with a positive
Gaussian erfc majorant. Unit-modulus Bloch phases and the triangle inequality
then give $|\widetilde T_{kPij}-T_{kPij}|\leq\epsilon$. Tightened enumeration
uses that same envelope. The finite LR sum and zero term are identical.
This reuses the existing general angular/contraction bound, not an assumed
Gaussian decay of a Schwarz factor. Epsilon zero retains previous identities.

With $X_{kij}=w_k\sum_sD_{skij}$, the paired producer gives
$\|\delta b\|_2\leq e=\sqrt{m}\epsilon\|X\|_1$. Keep the exact native F,
$W=\overline F F^T$, and $\widetilde c=W\widetilde b$.
The same algebra as Section 4 applies for complex Bloch arrays:

$$
|\widetilde E-E|\leq\|\widetilde c\|_2 e+
\tfrac12\|W\|_2 e^2,
$$

$$
\|\widetilde J-J\|_{\max}\leq
\epsilon\|\widetilde c\|_1+
\sqrt m\,(\|\widetilde T\|_{\max}+\epsilon)\|W\|_2 e.
$$

Use $\|W\|_2\leq\|F\|_F^2$, without replacing the finite Gram matrix
by the identity or re-whitening. The forward pass measures the maximum
coupling magnitude while streaming existing tiles. No dense tensor or extra
integral sweep is needed. Scalar bound arithmetic is outward guarded;
integral, factor, phase and contraction roundoff remain separate. Zero
density and zero retained rank give zero J/energy bounds. The screen is fixed
before SCF, so derivative/reciprocity tests apply to the selected quadratic
form. Its receipt records the unchanged finite reference coupling identity.

This first consumer admits Coulomb J only. A positive coupling-only screen
refuses fitted K, including K requested with zero energy coefficient. Mixed
HF composes the selected J with its existing exact exchange and correction;
pure KS retains its XC path. Method receipts preserve the evaluated-density
bounds without claiming density-relaxation or whole-method accuracy. The
shared Fock state and output retain their original density and frame contracts.
Canonical pair symmetry and operation-qualified RI block reduction remain
separate. Screening supplies no new symmetry certificate.

This boundary is deliberately distinct from the existing canonical periodic
BUPO tree. That tree's finite Ewald image support is not the GDF producer's
ordered sphere/capsule support. A future approximation must qualify its
partition and paired error propagation against the selected coupling before
method selection; sharing a lattice or a metric is insufficient.

The optional `coupling_multipole_error=epsilon` now supplies an experimental
ordered-image multipole approximation through that same producer. It is
mutually exclusive with both screening controls. The metric, reciprocal
panels and finite-zero subtraction remain exactly the selected finite source.
Zero epsilon preserves earlier source identities. Positive epsilon adds the
order and effective native extent controls to the coupling identity, without
changing the metric identity.

For each ordered auxiliary/AO shell triple, first count its actual selected
image pairs using `ThreeCenterImageDomain`, without storing the interaction
list. Divide epsilon by this count. The second traversal uses the same
sphere/capsule support. Native shifted-shell extents, absolute moments and
potential bounds qualify each separated auxiliary-image/AO-pair interaction.
Use the total-degree Cartesian Taylor matrix of the isolated
$\operatorname{erfc}(\eta r)/r$ kernel, with target-minus-source displacement.
The shared erfc radial recurrences retain trace terms; this is not a harmonic
Coulomb expansion. There is no lattice gauge or near-field subtraction in
this individual image kernel.

The shared Gaussian ledger bounds the compact Taylor remainder, penetrating
Gaussian tails and substitution of full moments. Accept a far leaf only when
their sum fits its image allowance. An overlap, always-near extent, overflow
of the bound or excessive error selects the existing direct MD integral.
Both forward and adjoint reuse this density-independent decision. Auxiliary
moments and shifted pair moments come from the existing native producers;
no replicated basis, dense global coupling tensor or new metric is introduced.

The executed receipt sums the accepted image bounds within a shell tile and
takes the maximum across tiles. Unit-modulus phases preserve this component
bound. Paired RI propagates this evaluated epsilon using the equations above,
the original finite factor, and the actual density. The source plan still
records the requested upper limit for admission and algorithm selection.
Empty far partitions therefore have zero evaluated approximation error while
remaining explicitly selected multipole runs. Integral/contraction roundoff,
finite domains, fitting-basis error and density relaxation remain separate.

This shell-leaf implementation is an integration baseline for hierarchical
periodic RI-BUPO. It does not yet group translated products into bubbles or
perform upward/downward translations. Conservative Gaussian derivative bounds
can make higher orders less admissible; an order increase is not a monotonic
accuracy or performance guarantee. The fixed source is compatible with shared
Fock composition and full-orbit validation, but supplies no symmetry reduction
authority or BIPOLE/CRYSTAL parity certificate.

These component records can feed the shared method-owned Fock build. They
do not supply an exchange energy, SCF state, terminal-density criterion,
BIPOLE parity claim or symmetry certificate. Whole-mesh fitted method
selection, physical range-separated fits and projector symmetry/response
qualification remain separate acceptance steps.

Whole-mesh RI target selection is fixed before metric construction. J-only
retains Gamma and no off-diagonal transfer owners; K-only retains Gamma
diagonal exchange and all selected source-to-target transfers. Absent
components have no output storage or energy. The finite-source identity
records J/K selection, while the evaluated spin-density and AO/Bloch frame
identities remain independent. A zero coefficient is not a target selector.
Each q leaf retains its conservative joint workspace admission. These
components may be composed only with separately admitted sources, preserving
their reciprocal orientation, finite supports and method-owned corrections.
No BIPOLE support, source equality or symmetry reduction follows from sharing
the completed Fock consumer.

Periodic RI uses $V_{PQ}(q)$ and three-center tensors with the actual Bloch
characters and compensated auxiliary charges of the selected producer.
At $q=0$, the chosen Green metric is semidefinite on charge modes; a fit must
state how it represents/removes that mode and preserves the electron count.
Use the source's compensation or an explicit constrained solve, for example
a KKT system enforcing $Q^Tc=N_e$, only when that is the specified fit.
Do not insert the constraint into an existing fit silently. RI-BUPO first
accelerates q=0 J's coupling using a paired approximation. q-dependent
exchange factors remain separate. Removing an auxiliary mode by threshold
can change symmetry, charge and numerical source identity.

Hartree zero average and exchange `exxdiv` are different operations. Full
Coulomb K has an integrable $q+G=0$ singularity whose finite-mesh treatment
(such as the method's Ewald/Madelung correction) is explicit and validated.
Finite-BvK corrections can couple through overlap/density, not merely add a
constant eigenvalue shift. Retain existing seam coefficients and spin factors.
For erfc exchange the Fourier kernel
$4\pi[1-e^{-Q^2/(4\omega^2)}]/Q^2$ has finite limit
$\pi/\omega^2$ at $Q=0$; it must not inherit the full-Coulomb singular
correction. erf retains a singular long-range part. BUPO-J makes no exchange
approximation decision and cannot make a hybrid calculation K-free.

The admitted explicit-source range-separated KS method resolves the complete
functional list to $A/r+\sum_j B_j\operatorname{erfc}(\omega_jr)/r$.
For each nonzero range, the supplied nonzero Q support is extended by the
finite screened zero at each diagonal k transfer. This term retains its
ordinary source-k weight and has no singular correction. Only the total
Coulomb coefficient $A$ multiplies the chosen BvK probe-charge correction,
$\Delta F_{\sigma k}=-A\xi S_kD_{\sigma k}S_k/f$, where $f=2$ for restricted
total density and $f=1$ for unrestricted spin density. Its energy is
$\tfrac12\sum_{\sigma k}w_k\operatorname{Re}\operatorname{Tr}
(D_{\sigma k}\Delta F_{\sigma k})$. Strict-zero-mode omits that correction;
screened-only exchange has no such choice in its actual source.

The distinction between singular Coulomb and nonsingular screened quadrature
follows Sundararaman and Arias, *Physical Review B* **87**, 165122 (2013),
Section II.A. The canonical decomposition defines this particular finite
source: rewriting erf as full Coulomb minus erfc retains the subtracted finite
screened zero. Separately omitting the zero of both leaves, or choosing a
range-dependent probe-charge prescription, would change the finite-mesh
Hamiltonian. Independent contraction/SCF acceptance establishes this stated
convention; it does not establish continuum accuracy or parity with a
different prescription. Component receipts bind every coefficient, range,
support and spin convention and keep raw K distinct from method-scaled energy.

The cutoff factory reuses the same certified nonzero G/Q spheres as HF and
global-hybrid KS and records that geometric child identity. Its common Q
cutoff applies to every resolved exchange range; only the method adds finite
screened zeros. Canonical XC images remain separate from the physical pair
translation source. Joint atomic-grid generation preserves the same evaluated
Hamiltonian as explicit composition, while reserving all source owners and
generation work through confirmation. Neither unchanged geometry identity
nor agreement on one finite grid establishes source convergence or a symmetry
certificate.

1D and 2D sources require their own Green functions, long-wavelength terms,
open-direction boundary conditions and dimensional stress definitions. The
3D formula above with a long vacuum vector is a convergence experiment, not
a definition of those methods. Reuse separately validated slab/wire source
implementations where available; otherwise refuse until derived. There is no
automatic lifting of current low-dimensional chi refusals.

## 6. CCM and chi adapters preserve operator identity

Store `source_kind`, domain selector and actual supports/weights, BvK extension,
character normalization, charge ledger, fit projector, nuclear boundary and
exchange seam in every build receipt. Existing routing distinguishes:

* An isolated finite cluster has free-space kernels and finite electron/nuclear
  sources. It may directly use molecular BUPO; no periodic completion is added.
* `four-center` CCM is the union-and-weight/WSSC lineage. `union12`, alternative
  weights and padded domains must remain separately named. Its known
  nonvariational source cannot be turned into a variational one by calling its
  matrix Hermitian. Preserve J/K and the historical energy expression; refuse
  a variational derivative claim where the source lacks that property.
* Neutral `neutral-bloch` and `real-gamma` are two representations of the
  **same specified neutral fitted torus** under their common conventions.
  A Fourier representation test is valid here. It does not identify that
  Hamiltonian with the union-weight source, chi, or infinite periodic HF.
* chi's finite-character Hamiltonian has Wigner-Seitz tie weights forming a
  partition of unity over translation classes. Those are not arbitrary
  center-dependent multipliers on free-space ERIs. `four_center`, `ri` and
  `rijcosx` preserve their specified finite source plus their declared fit/K
  approximation; they are not permission to use a generic periodic surrogate.
* SECCM variants retain their parameterized two-center/self/multipole models.
  MANTA can only accelerate an explicitly equivalent electrostatic term. It
  does not replace their Hamiltonians with Gaussian ab initio integrals.

A weighted quartet functional can use BUPO only if its weights admit the
required source-target factorization (or a bounded sum of such channels).
Derive that factorization, including permutation/adjoint relations, or keep
that part direct. A weight depending on all four centers cannot be pulled
inside one density multipole by assumption. RI substitutions also need a
defined same-weight metric and three-center construction; free-space fitting
is not automatically a fit of the weighted Hamiltonian.

Block-circulant torus actions stream cell-pair blocks and finite characters.
Small explicit supercell matrices are capped test oracles only. Source
restrictions, post-HF occupied/virtual spaces and experimental labels survive
provider changes. Neither an eigensolver migration nor MANTA parity closes a
pre-existing physical-support gate.

## 7. Symmetry in generation and digestion

Use existing `FiniteGroup`, `SpaceIdentity`, `BlockSpaceAction`, `Orbit`,
`OperatorContract`, `QualificationEvidence`, scatter/gather-adjoint and
retained-projector audits. A MANTA plan carries actual shell-pair and
quartet orbits with tensor actions on every leg. Integral permutation symmetry,
spatial symmetry and time reversal are separate operations with stabilizers.
Cartesian AO rotations are generally not Euclidean-unitary; use the metric
contract. For localized CCM/correlation spaces allow dense mixing, not only
pair permutations.

Generation computes representatives only after finite-source admission.
Digestion either reconstructs bounded tiles before the unreduced consumer or
gathers density panels with the adjoint action, contracts representatives and
scatters target blocks. Prove the gather/scatter pairing and count stabilizers
once, including coincident shells and even-mesh fixed images. A scalar orbit
size cannot replace tensor rotations or antiunitary conjugation.

The method qualifies $F[g.D]=g.F[D]$ for arbitrary trial densities with
covariant/contravariant transport. Invariance of a converged Fock requires
state invariance too. A density-dependent screen must be orbit-consistent;
take the maximum bound over the orbit or execute it unreduced. Do not average
the density or Fock to manufacture a passing residual.

Specific approximation gates are mandatory:

* Screening and finite image/G supports retain complete admitted orbits or
  use the subgroup preserving the actual support. Orbit-completing a cutoff
  changes the finite source and needs separate physical acceptance.
* Generic K-means, seeds and tie-breaking do not guarantee symmetric bubbles.
  Build mapped bubble families or use unreduced BUPO; retain multipole blocks
  with all m at each l and consistent orders over mapped nodes. Test raw
  covariance of the approximate source, not only the exact integral provider.
* Auxiliary rank cuts must transport the retained projector. Degenerate
  eigenspaces allow dense gauge rotations; splitting a degenerate multiplet
  by a threshold can fail admission. Do not average metric eigenvalues to
  repair it. Existing SR/LR GDF audits are reusable diagnostics, not production
  authorizations.
* Periodic truncations, atom images, q wraps and finite-character weights
  must satisfy exact group/cocycle laws. Subspace/operator leakage evidence
  remains attached when further selections are made.

Retain an identity-only/unreduced path for **the same** basis, finite source,
fit rank, bubble approximation and numerical settings. Reduced-versus-full
error is measured separately from BUPO-versus-analytic error. A qualified
symmetry plan may still be slower; benchmark before selecting it automatically.

## 8. Derivative obligations

At a stationary variational SCF state, differentiate the selected approximate
energy, including overlap Pulay terms $-\operatorname{Tr}(W\,dS)$, core,
J/K, nuclear and XC contributions. Write the locally fixed-rank RI inverse as
$G=U\operatorname{diag}(g_i)U^T$, with $g_i=1/\lambda_i$ for retained positive
metric modes and zero for discarded modes. Then

$$dE_J=c^T db+\tfrac12 b^T(dG)b,\qquad c=Gb,\quad b=\widetilde B d,$$

$$dG=U[L\circ(U^T(dV)U)]U^T.$$

Here $L_{ij}=-1/(\lambda_i\lambda_j)$ for two retained modes, zero for two
discarded modes, and $(g_i-g_j)/(\lambda_i-\lambda_j)$ between the two spaces.
The retained-projector response uses the same divided difference with the
retention indicator in place of $g_i$. This is the spectral Frechet derivative
(Higham, *Functions of Matrices*, 2008, Chapter 3). Degenerate eigenvalues
within either space require no eigenvector gauge derivative. The ordinary
inverse formula $dG=-G(dV)G$, and hence
$dE_J=c^T db-\tfrac12c^T(dV)c$, holds for a full-rank inverse. For a truncated
inverse it omits cross-space response unless the contraction makes those
terms vanish, for example when $b$ lies entirely in the retained space.

Use adjoints/real parts in complex form. The explicit integral/geometry
derivative of b is distinguished from density response. Include orbital and
auxiliary center derivatives, contraction/normalization derivatives when
parameters vary, retained-projector motion and any charge
constraint/compensation response. A rank crossing is a nonsmooth event
requiring documented handling, not omission of $dV$. The dense molecular
`RIMetricResponse` primitive implements the real symmetric response and its
Frobenius adjoint, with the same factorization as `RIJKBuilder`. It refuses
cutoff distances at or below its documented floating-point spectral guard;
this is a numerical admission test, not an interval error certificate.

BUPO derivatives include pair/bubble centers, multipole integrals,
translations and interaction tensors. Hard screening, near/far transitions,
clustering changes and order changes can introduce discontinuities. Either
differentiate an explicitly smooth partition (including its weights) or
qualify frozen-topology derivatives with crossing detection and convergence
of the discontinuity. Freeze neither geometry-dependent centers nor basis
derivatives merely because the traversal topology is frozen. Incremental
state is reset for displaced geometries.

Stress additionally differentiates image vectors, reciprocal vectors/volume,
Green/lattice coefficients, the polynomial/background and nuclear self terms,
exchange seam, and basis/Pulay terms. State whether k points are fixed in
fractional or Cartesian coordinates: under fixed fractional k, phase changes
must include the compensating reciprocal transformation. Metallic smearing
requires free-energy/occupation terms; q and quadrature identities must remain
consistent. Low-dimensional stress cannot use an arbitrary vacuum volume as
a physical normalization.

CCM derivatives also track domain weights, tied Wigner-Seitz representatives
and finite-source boundaries; each source's variational status governs what
can be claimed. Post-HF forces need their own response/Lagrangian terms.
Validate distorted geometries, cell strains and boundary crossings separately
from energy parity. Zero forces at a symmetric equilibrium are not sufficient.

#### Explicit periodic atomic quadrature source

The bounded 3D atomic-grid adapter canonicalizes nuclei, reuses the shared
point-centered periodic partition, and folds raw points into the canonical
cell without changing weights. The resulting periodization of a finite XC
image source is part of its definition, not an assertion of infinite-image
accuracy. The source digest binds actual coordinates and weights along with
its radial/angular/partition choices. Grid population or volume agreement
cannot replace separate XC energy and spin/k potential refinement. The
adapter's modeled resource bounds include current OpenMP worker scratch and
one owned export of each array. Separate KS calls must count the retained grid as external live memory.
The joint grid/domain/KS plan now reserves that storage and source copies,
subtracts grid work from downstream budgets, and admits all stages before
point generation. It preserves the same finite Hamiltonian and does not
turn SCF confirmation into quadrature accuracy qualification.

A frozen-density refinement assessment now admits all requested grid
generations and XC contractions together. Basis, canonical image domain,
functional, spin densities, k points, weights and AO frame stay fixed. The
reference grid is an explicit input; it is evaluated first and its potential
is retained while other grids are streamed. Each candidate reports signed
energy and spin-population differences, and maximum-entry and Frobenius norms
of every spin/k potential difference. These are source-AO matrix norms with no
k weights inside them, not overlap-orthogonalized or basis-independent bounds.
Only XC changes in this assessment, not the complete Fock or SCF density.

An explicitly named hybrid assessment shares this comparator while evaluating
only the semilocal component. It admits the complete native LDA/GGA descriptor
and records the required global exchange fraction or signed range-kernel list.
At fixed density and unchanged grid-independent exchange, its difference is
also the grid-induced difference of the complete physical Fock and energy.
The exported reference potential itself remains partial. Exchange, singular
corrections and other terms belong to the method that assembles the full build.

The fixed-state density identity excludes quadrature selection. The ordered
grid/source identities and reference index identify the comparison protocol.
Neither zero reference-row differences nor identical neighboring grids prove
accuracy. A separately refined integration witness must qualify the chosen
reference in energy and every spin/k potential. Charge agreement alone is
insufficient. Finite-image and Hamiltonian errors are outside this diagnostic.
It produces component evidence for a method-owned Fock build, not a second
operator or solver contract. The shared grid and explicit full source are
preserved; this adds neither BIPOLE/CRYSTAL parity nor symmetry admission.


#### Public KS snapshot and neighboring workstreams

Selected public pure, global-hybrid and range-separated KS jobs preserve the same finite joint source and
method-supplied component energies. The physical Fock belongs to its evaluated
density epoch; the orbital projector produced by diagonalizing that Fock is a
separate state at a capped exit. Hybrid exchange energies already contain the
method coefficient and restricted spin factor; the child K receipt remains
unscaled. Semilocal XC and exact exchange are separate contributions, and the
exchange total includes its explicitly recorded zero-mode correction.
Range-separated jobs retain each signed kernel coefficient, range and raw
child receipt, with separately scaled component energy and actual/requested
singular conventions. A screened-only source has no active Coulomb correction.
The generic Fock object's commutator diagnostic does not replace the native
method's convergence norm, threshold or exact-confirmation gate.
Source, density, Fock, quadrature and basis
frame identities remain separate. A generic Hermitian KS operator may own a
copy of one spin/k snapshot for diagnostics, without moving source ownership
or adding copies inside the bounded native solver. A symmetry label cannot
qualify the grid, image domain, electronic state or retained projector.

The shared periodic grid and midpoint nuclear kernels retain compatibility
with BIPOLE's physical definitions, but source matching and the independent
CRYSTAL parity gate remain prerequisites for comparison. Selected hybrid KS
uses explicit reciprocal Q spheres over the full mesh; these cannot qualify
or replace BIPOLE's independent AC/BD exchange supports. No RHF-only state
contract is relabeled as KS to obtain symmetry admission. These public jobs
retain the full mesh until source-specific symmetry proofs are supplied.

### Physical range fitting component boundary

The dense and streamed periodic GDF interfaces accept a distinct
`PeriodicGDFKernelOptions` physical interaction. Coulomb defaults retain their
existing identities and counts. Physical `erfc` combines numerical erfc SR
with the difference of numerical and physical erf Fourier kernels, using a
stable exponential difference. Its finite Gamma coefficient is retained once.
Physical `erf` is reciprocal-only and omits the singular zero. Receipt fields
separate Gamma transfer, removed numerical zero, included physical zero and
executed real-space source. RI-q admission uses Gamma transfer itself and
exports the actual kernel/range; it never equates Gamma with SR subtraction.

The physical range participates in source identity. Neither interaction choice
nor unused erf split controls relabel the AO frame or evaluated density.
Metric projectors remain fixed and negative-eigenvalue guards are unchanged.
Native ownership is still one metric or one coupling tile; conservative
Coulomb admission also bounds these source variants. No complete dense generic
Fock snapshot is introduced inside a bounded solve. Method-level range KS,
Madelung ownership and joint solver admission are qualified separately below.
Source support does not grant BIPOLE parity, response
support or a symmetry certificate.

Whole-mesh physical range K now composes these fixed-q sources in
`PeriodicRIFockBuilder`. Non-Coulomb source selection refuses a simultaneous J
request: the caller must retain a separate bare-Coulomb Hartree source. Every
q metric uses the same physical kernel/range, with unchanged reverse-support,
metric, projector and inverse audits. The Gamma owner handles diagonal K and
its zero convention; off-diagonal labels remain ket minus bra. Scalar weighted
energies, evaluated-density/frame identity and owned snapshots keep the existing
contract. This component supplies neither a method singular correction nor
joint admission with complementary J/XC sources.

The explicit fitted range KS method now supplies that complete composition.
Its functional resolves to the same canonical Coulomb-plus-erfc terms as
direct range KS before fitting. Each kernel/range has its own positive-metric
projector; signed coefficients scale the resulting variational responses.
This defines the finite fitted functional. Fitting erf directly is not assumed
equivalent to subtracting separately fitted Coulomb and erfc responses.
Bare-Coulomb J remains separate, with exact/RI choices independent of K.
The erfc metric source owns its finite Gamma zero. Only the resolved long-range
Coulomb coefficient scales the selected Madelung energy and potential.

All retained kernel sources, metric setup, component receipts, workspaces,
solver iterations and terminal confirmation are charged together. The complete
energy includes H once and the actual semilocal functional energy. Partial RI
Fock receipts remain labelled components. RKS embeds half its total density
in each fitting spin; UKS uses the actual evaluated spin blocks. Source,
AO frame and evaluated-density identities remain distinct. Exported shared
`FockBuild`/`EnergyComponents`/`OperatorFrame` states and `BlockSpaceAction`
transport retain the existing contracts without changing the bounded solver
storage. This does not grant BIPOLE AC/BD parity or production symmetry reduction.

Generated fitted range KS preserves the same physical method. Count-only
cutoff boxes and atomic quadrature plans jointly admit geometry traversal,
retained grid/domain payloads, every kernel/projector and the complete solver.
Geometry identity remains distinct from physical kernel and projector identity;
changing functional range or metric cutoff need not change the generated
support. RI K geometry has empty diagonal segments and ket-minus-bra transfers,
while exact K keeps primed target-minus-source supports. Physical finite-zero
and method singular-correction ownership is unchanged. Selected provenance
reserves and records all actual kernel sources, grid and domain receipts.
Finite-source agreement does not qualify cutoff convergence. Public fitted
range-KS jobs add separate job/output admission and evaluated-state delivery.
J and physical K fitting identities remain an ordered collection, including
the absent J-fit case. Component evidence preserves actual physical ranges
and projectors; no aggregate fit is invented. QVF retains the evaluated
spin densities in the nuclear-centre gauge, including capped solves. Public
selection does not expand source-specific symmetry or external parity claims.

### Typed periodic bubble hierarchy and weighted adjoint

`PeriodicBubbleTree` accepts only a native `PeriodicPairSource`. Its private
geometry adapter preserves the complete retained pair-image index map,
including diagonal images and conjugate-orbit multiplicities. It does not
relax the public molecular `BubbleTree` triangular-inventory requirement.
Tree identity includes the pair producer's basis, image/cutoff, lattice,
momentum/weight and extent identities, plus the deterministic partition.
Stable median partitioning, child-sphere enclosure, always-near propagation
and disjoint unordered traversal reuse the qualified geometry algorithm.
A geometric `far_candidate` still requires a periodic kernel error proof;
separation of reference-cell centers alone is insufficient.

Let $W_{ab}=W(C_a-C_b)$ translate full raw Cartesian moments from center
$a$ to center $b$. First contract the entire complex Hermitian density mesh
with the source's original weights and phases. No relation between $D(k)$
and $D(-k)$ is imposed. Leaf and parent moments then satisfy

$$Q_n=\sum_{p\in n}W_{pn}Q_p,\qquad
Q_n=\sum_{c\in\operatorname{children}(n)}W_{cn}Q_c.$$

The implementation streams one translation matrix at a time, with no global
node-to-AO map or retained matrix per tree edge. For arbitrary node potentials
$V_n$, the reverse pass adds $W_{cn}^{T}V_n$ to each child, then
$W_{pn}^{T}V_n$ to each pair, and invokes the original pair-source adjoint.
It is the derivative of $\sum_n Q_n^T V_n$ with respect to the weighted mesh
inner product. Potentials supplied at multiple ancestors contribute additively;
a physical interaction consumer must choose a disjoint interaction partition.
Raw Cartesian trace moments remain present in both passes. These are the
ordinary moment translations used by BUPO (Neese et al., 2025,
DOI 10.1021/acs.jpca.4c07415), composed with the q=0 Bloch source, not a new
symmetry or time-reversal reduction.

Density-dependent absolute full and tail radial moments are propagated with
positive binomial bounds. For child-to-parent separation $d$,

$$B_{n,k}\le\sum_c\sum_{j=0}^k {k\choose j}d^{k-j}B_{c,j}.$$

Here the right side defines the reported bound; at a leaf each pair bound
is multiplied by the absolute component-coefficient sum from the native
weighted density contraction. The same expression applies to tails because
the parent ball encloses every child ball: outside the parent, each child's
tail bound applies. Absolute coefficient sums propagate without cancellation.
Logarithmic positive accumulation retains subnormal bounds and refuses
unrepresentable results. Floating-point headroom is not interval arithmetic.

Joint count-only admission includes the existing source reservation, tree
construction/copies, one forward/reverse cycle, all node result/input and
correction arrays, radial bounds, owned exports and one translation matrix.
The plan's conservative work bounds include the existing source plan; actual
shell and translation counts are reported separately. No input array is read
or copied until its shape matches the pre-admitted source/node dimensions.
Retaining arbitrarily many independent results remains the caller's budget.

Qualification `periodic-pair-image-hierarchy-and-adjoint` covers the retained
finite source and its geometry/moment operations. Complete finite reciprocal
Coulomb calculations validate the composition against independent ordered
Gaussian Bloch sums and weighted energy derivatives. Their Fourier domain,
zero mode and Taylor approximation are declared explicitly. These calculations
do not qualify omitted pair images, an infinite Ewald accuracy profile,
Gaussian-kernel far admission, charge/background/self selection, exchange,
derivatives of moving geometry, or a production periodic SCF route. The
shared FockBuild/JKBuilder and BIPOLE domain contracts are unchanged.

### 5.6 Direct finite Ewald response over canonical pair images

`PeriodicEwaldSource` retains the same immutable native `PeriodicPairSource`
as the periodic hierarchy. It supplies the complete direct finite-domain
quadratic form needed for a later kernel-admitted fallback. Its real and
reciprocal integer domains are unique and inversion closed; the real domain
contains zero and the reciprocal domain excludes it. An empty reciprocal
domain is allowed as an explicitly incomplete finite source. Cell vectors,
eta and the gauge follow Section 5.2. Shared domain validation and hashes
are identical to the lattice-tail implementation. The full-kernel identity
excludes a near partition, since this direct source contains none.

For retained pair-image component $u$, let $f_u(r)$ be its real Gaussian
product and $c_u$ its native whole-mesh coefficient,

$$c_{p,ij}=m_p\operatorname{Re}\sum_k w_k D_{ij}(k)e^{-ik\cdot R_p},
\qquad m_p=\begin{cases}2&\text{paired image orbit},\\1&\text{otherwise}.\end{cases}$$

The kernel matrix is the full Gaussian integral of the finite Green function,

$$I_{uv}=\sum_{R\in\mathcal R}(f_u|\operatorname{erfc}(\eta|r-r'+R|)/|r-r'+R||f_v)
+\frac{4\pi}{\Omega}\sum_{G\in\mathcal G}
  \frac{e^{-G^2/(4\eta^2)}}{G^2}\operatorname{Re}[F_u(G)^*F_v(G)]
-\frac{\pi}{\eta^2\Omega}q_uq_v,$$

where $F_u(G)=\int f_u(r)e^{-iG\cdot r}\,dr$ and $q_u=F_u(0)$.
The energy is $E_J=\tfrac12 c^TIc$. Each canonical pair block is visited
once. An off-diagonal block contributes its action and transpose to the two
pair potentials. For a diagonal block, inversion closure makes its complete
real-domain sum symmetric; applying $(I+I^T)/2$ while digesting the images
implements that identity with a fixed numerical Hessian. It does not repair
or discard a computed non-Hermitian Fock matrix. All off-diagonal AO
components in an origin same-shell pair remain present.

Real-space terms use native contracted erfc shell quartets directly, including
overlapping and identical Gaussian sources. Both shells of the source pair
are translated by the same `-cell @ index`. No bare-minus-erf subtraction,
point-charge singularity or Gaussian electronic self removal is introduced.
One quartet is live at a time. Each reciprocal tile is streamed twice, first
to contract $\rho(G)=\sum_u c_u F_u(G)$, then to apply its adjoint to all
pair components. No all-pair Fourier panel or global four-index array is
retained. Overlap panels provide the charge and the gauge response. Real,
reciprocal and gauge energies are reported separately; the constant occurs
exactly once. The algorithm uses the Ewald split (Ewald 1921, IV.1,
Eqs. 37-40), Gaussian MD integrals and the q=0 Bloch contraction of
Sun et al. (2017), Eqs. 16-17, with the declared component convention.

The original phase map scatters pair potentials to complex $J(k)$. Its
weighted derivative is

$$dE_J=\sum_k w_k\operatorname{Re}\operatorname{Tr}[dD(k)J(k)],\qquad
E_J=\tfrac12\sum_k w_k\operatorname{Re}\operatorname{Tr}[D(k)J(k)].$$

Neither positivity nor a relation between $D(k)$ and $D(-k)$ is assumed.
An unrestricted consumer supplies total density $D_\alpha+D_\beta$ for J;
exchange remains separate. Source and density identities match the original
pair map; the new source also binds the finite kernel. Moment order, extent
radii or reciprocal batching do not change these exact Gaussian integrals.
They are not multipole approximations.

Qualification `finite-canonical-pair-ewald-j` is deliberately finite.
Canonical folding is exact against a periodic potential, while a finite real
Ewald sum has a covariance defect. This response is the exact derivative of
the declared folded finite quadratic form, not proof that an arbitrary short
real domain equals an ordered infinite Bloch calculation. Domain enlargement
and separate native Gaussian bounds remain necessary. Omitted AO pair images,
full-cell charge/background selection, whole-method accuracy, nuclear energy,
exchange and production SCF selection are not certified here. No interval
roundoff bound is claimed. Kernel-admitted bubble traversal and aggregation
of these error sources remain required before production periodic BUPO.

Count-only admission precedes domain copies and all density payload reads.
It reserves the retained pair source, input/output/correction arrays, domain
sorting, shell snapshots, two local coefficient/potential panels, one quartet
or Fourier tile and independent exports. The existing full-basis Fourier
plan conservatively bounds each streamed shell panel; its work reservation
can exceed executed work. It creates no corresponding global panel cache.
All five resource caps are enforced and actual shell/primitive calls are
reported separately. Shared FockBuild/UHF, physical BIPOLE AC/BD and exchange
domains, and source-qualified symmetry dispatch remain unchanged.

### 5.7 Bound-admitted finite periodic BUPO response

`PeriodicBupoSource` retains a native `PeriodicBubbleTree` and
`PeriodicEwaldSource` with exactly matching pair-source identities. The
partition is fixed at construction by geometry, the finite kernel, moment
order and an absolute integral tolerance. It never depends on the current
density. This composes the BUPO separation of Neese et al. (2025,
DOI 10.1021/acs.jpca.4c07415) with the Ewald convention of Section 5.6
and the q=0 Bloch coefficients of Sun et al. (2017), Eqs. 16-17.

A geometric far candidate is tested against every retained real image.
The support spheres must be separated at all singular images, with a
roundoff guard on translated-center separation. A near set contains the
origin, is unique and inversion closed, and is a subset of the direct real
domain. For candidate nodes A and B, the approximate matrix is

$$T_{AB}=\sum_{R\in\mathcal N}T^{1/r}_{AB}(R)
       +T^{G_{\mathcal R,\mathcal G}-\sum_{R\in\mathcal N}1/r}_{AB}.$$

The native Gaussian interaction ledger supplies compact Taylor, penetration
and full-moment tail bounds for each term. Their positive sum must be no
larger than `integral_tolerance`; otherwise the traversal refines, eventually
falling back to exact finite Gaussian Ewald integrals. A nonrepresentable
far field also refines. Every unordered canonical pair interaction occurs
once in either an admitted far block or a direct block.

Node error envelopes are density-independent maxima over descendant AO
components, translated with the binomial theorem. They differ from the
density-weighted bounds returned by `tree.contract`. If their interaction
bound is $\delta_{AB}$, it bounds the error of each descendant component
integral uniformly. With $L_A=\sum_{u\in A}|c_u|$ and the same definition
for B, disjoint off-diagonal far blocks give

$$|\Delta E_J|\le\sum_{(A,B)\in\mathcal F}\delta_{AB}L_A L_B,\qquad
|\Delta v_u|\le\sum_{(A,B)\in\mathcal F,\ u\in A}\delta_{AB}L_B
             +\sum_{(A,B)\in\mathcal F,\ u\in B}\delta_{AB}L_A.$$

Ancestor errors add down to the leaves. Absolute Bloch phases are one, so
scattering these component errors supplies a uniform-over-k AO matrix
`j_error_bound`. Both canonical conjugate arms contribute, including the
two arms of a nonzero-image diagonal AO component. Positive logarithmic
accumulation keeps underflowed nonzero bounds positive. The bounds exclude
floating-point roundoff and do not constitute interval arithmetic.

The forward pass aggregates full raw Cartesian Gaussian moments. For each
far block, the local potentials receive $T_{AB}Q_B$ and $T_{AB}^TQ_A$ and
the energy receives $Q_A^TT_{AB}Q_B$. The transposed hierarchy and original
pair adjoint recover J. No trace projection is valid for the nonharmonic
complementary Ewald kernel. Direct blocks reuse the same erfc, reciprocal
and gauge primitives as Section 5.6. Near plus tail contains each reciprocal
and gauge contribution exactly once. With no accepted far blocks, the
source calls the existing complete direct builder.

The density-independent partition therefore defines one fixed symmetric
finite quadratic form. Its J is linear in D and satisfies the weighted
first derivative and half-trace identities of Section 5.6. Recomputed
partition identities must match construction. The receipt separates direct
and far energies, reports independently evaluated trace energy, and binds
the pair source, tree, finite kernel, partition, tolerance and density.
For a mixed partition, the nested direct receipt has a distinct subset
source identity and qualification `finite-ewald-direct-partition-subset`.
Its real/reciprocal/gauge energies and trace concern only direct blocks;
its charge is that of the complete retained source.

Count-only joint admission includes retained producers, Gaussian envelopes,
worst-case direct blocks, two traversal/admission passes, forward and reverse
moments, corrections and owned exports. One M2L matrix and local shell or
Fourier panels are live; there is no per-edge matrix cache or global
four-index tensor. Conservative work counts are reservations, not FLOPs or
measured RSS. Many retained result copies remain the caller's responsibility.

Qualification `finite-periodic-bupo-j` compares the approximation to exactly
the finite canonical kernel supplied by the matching direct source.
`integral_tolerance` is a component-integral threshold, not an SCF or
total-energy accuracy profile. The receipt's density-dependent aggregate
bounds must be assessed separately. Infinite Ewald/domain and omitted
pair-image errors, canonical finite-domain covariance, full-cell charge and
nuclear policy, RI fitting and production selection remain separate work.
Total alpha plus beta density supplies UHF J; exchange, shared FockBuild,
physical BIPOLE domains and qualified symmetry dispatch are unchanged.

### 5.8 Retained-source infinite-Ewald error aggregation

`PeriodicBupoOptions.bound_infinite_domain=True` adds the full Gaussian
domain bounds of Section 5.2 to the same finite response from Section 5.7.
It does not change the partition, moments, J or energy. The default remains
the finite-source mode, which also admits non-cubic finite domains.

The native consumer constructs a full Gaussian majorant from each retained
pair's original immutable basis, selected shells and represented translation.
The caller cannot inject a numerical envelope. Canonical domain validation,
complete-cube checks and conservative lattice metrics are prepared once.
Every pair of majorant centers must be separated from the omitted real
images, even if the current density is zero. Construction refuses a domain
that cannot establish this condition. Majorants require linear storage in
the retained primitive-pair inventory; component interaction bounds stream
without a quadratic cache.

Let $\epsilon^R_{pq}$ and $\epsilon^G_{pq}$ be the uniform full-Gaussian
real and reciprocal domain bounds for every AO component of retained pair
entries p and q. They refer to the physical Ewald eta and the exact same
finite real/reciprocal domains as the direct source. Near subtraction and
the constant gauge cancel in the infinite-minus-finite difference. With
$L_p=\sum_{u\in p}|c_u|$, the consumer reports

$$B_E^X=\tfrac12\sum_p \epsilon^X_{pp}L_p^2
       +\sum_{p<q}\epsilon^X_{pq}L_pL_q,\qquad X\in\{R,G\},$$

$$B_{v,p}=\sum_q(\epsilon^R_{pq}+\epsilon^G_{pq})L_q.$$

These bounds use full signed component coefficient norms, including original
mesh weights and canonical multiplicities. The diagonal energy factor is
one half; the diagonal potential factor is one. Each component of pair p
receives $B_{v,p}$, then the absolute original Bloch scatter adds both
conjugate arms to a uniform-over-k AO matrix. A nonzero-image diagonal AO
component contributes twice. Positive logarithmic sums/products preserve
nonzero underflowed bounds. Separate normalized coefficient scales avoid
premature underflow in the density norm.

The result's `domain_j_error_bound` and nested `receipt.domain` are present
only when the option is enabled. The nested receipt reports real/reciprocal
energy bounds, their sum, actual pair/majorant counts, minimum effective
decay and omitted-center distance, and source/majorant/domain/kernel/density
identities. Its source identity excludes the unrelated near partition.
`finite_energy_error_bound` and `finite_maximum_j_error_bound` retain the
Section 5.7 approximation bounds. Overall `energy_error_bound` and
`j_error_bound` add finite BUPO and infinite-domain errors outward.
Density changes scale these bounds without changing the computed Hessian.

Joint count-only admission adds majorant production, construction and build
domain passes, coefficient norms, retained envelopes and owned AO bound
arrays. It occurs before reading near payloads or copying density arrays.
Domain geometry is not rebuilt for every primitive or pair interaction.
The conservative work model includes worst-case component evaluations;
it is not measured work or a production scaling claim.

Qualification `periodic-bupo-j-with-retained-source-ewald-bound` compares
the finite approximation with the infinite zero-average conducting Coulomb
kernel for the same retained canonical Gaussian source. Canonical folding
is exact in that periodic limit, so the domain bound also covers its finite
kernel covariance defect. This does not establish that the finite pair-image
inventory equals an untruncated AO density. Pair-image omission, roundoff,
full electronic+nuclear charge/background/self policy, RI fitting, automatic
accuracy-profile selection and geometry derivatives remain separate work.
The derivation reuses the full Gaussian convolution bound and Ewald1921
IV.1 Eqs.37-40, not a compact-support assumption or point-charge self term.
Shared FockBuild/UHF, physical BIPOLE domains and qualified symmetry are
unchanged.

### 5.9 Full AO pair-image omission bounds

`PeriodicBupoOptions.bound_pair_images=True` additionally bounds all omitted
AO pair translations for the infinite zero-average conducting Coulomb kernel.
It requires the retained-source infinite-domain bound of Section 5.8.
The finite calculated values and density-independent partition are unchanged.
This is a native source-bound envelope, not a user-supplied error estimate.

For one primitive of angular degree l and exponent a, absorb its Cartesian
polynomial into half its Gaussian using

$$r^l e^{-a r^2}\le H_l(a)e^{-a r^2/2},\qquad
H_0(a)=1,\quad H_l(a)=\left(\frac{l}{ae}\right)^{l/2}\ (l>0).$$

The maximum absolute row sum of the Cartesian-to-shell transformation bounds
every pure or Cartesian component. For primitive exponents a,b, absolute
contraction coefficients and these row norms give a positive amplitude M.
Their product is bounded by a spherical Gaussian of exponent
$\alpha=(a+b)/2$, product center P and absolute integral

$$q(n)=M H_{l_a}(a)H_{l_b}(b)
       (\pi/\alpha)^{3/2}e^{-\mu|A-B-Ln|^2},\qquad
\mu=\frac{ab}{2(a+b)}.$$

All primitive contributions add positively. No signed cancellation, extent
radius or asymptotic compact-support assumption enters this envelope. The
weaker separation decay is intentional: it removes image-dependent angular
polynomials while still giving a convergent all-image sum.

Let N be the largest absolute integer coordinate in the original admitted
image inventory, before applying the physical pair cutoff. Complete the
integer cube of radius N, recognize retained
entries through the original pair involution and sum every missing entry
inside it explicitly. This handles sparse input images, interior holes and
cutoff exclusions. Every image outside the cube is missing. Retaining the
original image inventory lets a larger declared cube tighten the omitted
tail even when its extra pairs fail the physical cutoff. This changes the
error enclosure and admitted work without changing the finite Hamiltonian.
Never subtract one unrelated upper bound
from another to estimate the missing source.

For the admitted lower real metric s, put $a_0=\mu s^2$,
$\tau=|A-B|/s$ and $u=N+1-\tau$. Each shifted one-dimensional Gaussian
sum is bounded by $\Theta=2+\sqrt{\pi/a_0}$, from the two nearest samples
and the integral of the remaining monotone tails. Thus the omitted 3D sum
is at most $\Theta^3$. If $u>0$, the union of three omitted coordinate
slabs gives the additional bound

$$6\Theta^2 e^{-a_0u^2}\left(1+\frac{1}{2a_0u}\right).$$

Use the smaller of these bounds. The second follows by bounding each
half-line by its first term plus its integral and applying the erfc
inequality from Section 5.2. It tends to zero with N. Guarded lower distances,
positive logarithmic sums and conservative decay caps preserve nonzero
underflowed bounds; they do not provide interval roundoff certification.

The result is an ordered AO component mass bound $T^r_{ab}$ for the retained
inventory and $T^m_{ab}$ for its entire missing complement. For off-diagonal
shells the reverse image inventory gives the transposed matrix. For a
same-shell nonzero image both images enter, including twice on a diagonal
AO component. The native matrices are independent of the density and k.

To propagate these masses to the infinite Coulomb action, let
$\alpha_{\max}$ bound all envelope exponents. Convolution of two normalized
Gaussians has exponent $\beta\le\alpha_{\max}/2$. Its periodization is
uniformly at most $(2\sqrt{\beta/\pi}+1/s)^3$, using the shifted Gaussian
sum above. Since $\int_{\mathbb R^3}\operatorname{erfc}(\eta r)/r\,dr
=\pi/\eta^2$, a uniform bound on the real Ewald interaction is

$$C_R=\frac{\pi}{\eta^2}
       \left(2\sqrt{\frac{\alpha_{\max}}{2\pi}}+\frac1s\right)^3.$$

Use $V_0\le s^3\le\Omega$ as a conservative volume lower bound and
the admitted reciprocal metric g. With $t=2\sqrt\pi\eta/g$, positive
reciprocal coefficients and unit phase modulus give

$$C_G=\frac{4\pi}{V_0g^2}(3t+3t^2+t^3),\qquad
C_0=\frac{\pi}{\eta^2 V_0},\qquad C=C_R+C_G+C_0.$$

Here $3t+3t^2+t^3$ bounds the nonzero integer Gaussian sum: each centered
one-dimensional sum is at most $1+t$, and $|n|^2\ge1$ bounds the reciprocal
denominator. $C_0$ bounds the constant zero-average gauge. This proves
$|B(f,h)|\le C Q_fQ_h$ for any signed functions dominated by these Gaussian
sums, independently of their centers. This derivation extends the Ewald1921
IV.1 Eqs.37-40 split; the bound is not attributed to the point-charge paper.

Define $W_{ab}=\sum_k w_k\max(|D_{ab}(k)|,|D_{ba}(k)|)$. The maximum
equals either magnitude for Hermitian D and also covers the small accepted
Hermiticity residual. Set $Q_r=\sum_{ab}W_{ab}T^r_{ab}$ and
$Q_m=\sum_{ab}W_{ab}T^m_{ab}$. Then

$$B_E^{\rm image}=C(Q_rQ_m+\tfrac12 Q_m^2),$$

$$B_{J,ab}^{\rm image}
  =C\left[Q_mT^r_{ab}+(Q_r+Q_m)T^m_{ab}\right].$$

The second expression includes missing target pairs as well as missing
density. Bounding only the retained target response would be insufficient.
Absolute Bloch phases are one, so the J bound is uniform over the original
mesh, including zero-weight points. These expressions have the expected
quadratic energy and linear J scaling without changing the fixed Hessian.

`receipt.pair_images` binds the native pair/kernel/density identities and
reports the completion cube, actual counts, lattice metrics, three kernel
bounds, density mass bounds and resulting errors. Owned result arrays expose
`retained_pair_mass_bound`, `omitted_pair_mass_bound` and
`pair_image_j_error_bound`. They are `None` when disabled. Overall bounds add
finite BUPO, retained-source Ewald-domain and pair-image errors outward;
the nested domain receipt continues to describe only its own contribution.
Joint admission includes completion enumeration, membership lookup, primitive
envelopes, contraction and owned arrays before near payloads or density copies.
No dense supercell tensor or infinite image enumeration is allocated.

Qualification `periodic-bupo-j-with-full-pair-image-ewald-bound` compares
with the full AO pair-image electronic Coulomb source under the declared
zero-average kernel. Small-domain bounds can be very loose. Compare both
total J and energy bounds with the desired target and enlarge domains or
refuse that target; this option does not choose a production accuracy profile.
This electronic-only receipt excludes roundoff, full electronic+nuclear
charge/background/self policy, exchange, RI fitting, automatic SCF selection
and geometry derivatives. Section 5.10 supplies the matched charge composition.
The shared Fock/UHF, BIPOLE physical domains and symmetry interfaces are
unchanged.


### 5.10 Matched electronic and point-nuclear electrostatics

`PeriodicElectrostaticSource` composes a native `PeriodicBupoSource` with
finite point positions and signed charges. Both electronic options
`bound_infinite_domain` and `bound_pair_images` are required. The constructor
uses that source's basis, canonical AO images, original k mesh and weights,
physical eta, real and reciprocal cubes, and zero-average conducting kernel.
The existing pair-midpoint nuclear cutoff driver has a different finite
support and cannot supply these components. Native McMurchie-Davidson
potential and Fourier kernels are reused without changing either interface.

Let the electronic density be positive electron number density, $Q_e$ its
retained charge, $Q_n=\sum_A Z_A$, $Q=Q_n-Q_e$, and
$c=-\pi/(\eta^2\Omega)$. The energy and its weighted density derivative are

$$E=\tfrac12 B(\rho_e,\rho_e)-B(\rho_e,\rho_n)+E_{nn},
\qquad \delta E=\sum_k w_k\operatorname{Tr}[\delta D(k)(J(k)+V_n(k))].$$

$E_{nn}$ includes the ordered real point sum with $(A,A,0)$ omitted,
$\frac12\sum_{G\ne0}\lambda_G|\rho_n(G)|^2$, point self removal
$-\eta\sum_A Z_A^2/\sqrt\pi$, and $cQ_n^2/2$. Gaussian electronic self
interaction is finite and receives no point-self subtraction. The three
gauge terms are $cQ_e^2/2$, $-cQ_eQ_n$, and $cQ_n^2/2$: their sum is
$cQ^2/2$. A separate background energy must not be added again.
This is the conducting three-dimensional Ewald split in Ewald (1921),
IV.1 Eqs. 37-40, [DOI 10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304).
There is no surface dipole term or lower-dimensional boundary claim.

The default `neutral` policy admits a density only when
$|Q|+Q_m\le\epsilon_Q$, where $Q_m$ is the full omitted-image charge bound
from Section 5.9. Retained $\operatorname{Tr}(DS)=Q_n$ alone is insufficient.
The alternative `uniform-background` policy explicitly records compensating
retained charge $-Q$ and the same uncertainty. Both policies use the same
zero-average kernel; the choice changes admission and identity. The overlap
and point potential are immutable owned exports. Charge admission precedes
the electronic build. Charge tolerance is positive and finite. It is a
physical admission tolerance, not a floating-point interval guarantee.

For point pairs use the domain proof in Section 5.8 with effective real
exponent eta; for a Gaussian majorant of exponent alpha against a point use
$\eta_{\rm eff}=(\eta^{-2}+\alpha^{-1})^{-1/2}$. The Gaussian/Gaussian
convolution's second variance is absent. The omitted real domain must be
separated from each retained pair of centers; unsupported domains refuse
independently of the density. Reciprocal bounds use absolute point charge
and the same Gaussian lattice shell bound. Gauge and point-self terms cancel
between finite and infinite domains. Thus

$$B_{nn}=\tfrac12\sum_{AB}|Z_A Z_B|\,\delta_{AB},\quad
\delta_p=\sum_A|Z_A|\,\delta_{pA},\quad
B_{en}^{\rm domain}=\sum_p L_p\delta_p.$$

$L_p$ is the absolute canonical density coefficient sum. Scattering
$\delta_p$ through both canonical arms gives the uniform AO potential bound.
For missing electronic images use the Section 5.9 mass envelopes and replace
its Gaussian/Gaussian real kernel constant with

$$C_{R,\rm point}=\frac{\pi}{\eta^2}
\left(2\sqrt{\alpha_{\max}/\pi}+\frac1s\right)^3,
\qquad C_{\rm point}=C_{R,\rm point}+C_G+C_0.$$

This is the periodized normalized Gaussian supremum times the integral of
erfc(eta*r)/r; a point contributes no convolution variance. With
$Z_{\rm abs}=\sum_A|Z_A|$, the missing-source energy and missing-target
potential errors are

$$B_{en}^{\rm image}=C_{\rm point}Z_{\rm abs}Q_m,\qquad
B_{V,ab}^{\rm image}=C_{\rm point}Z_{\rm abs}T^m_{ab}.$$

These bounds are derived here; they are not attributed to the point-charge
paper. Positive logarithmic sums preserve nonzero underflow. The final
energy bound adds the electronic, point-domain and both attraction errors;
the final matrix bound adds electronic and nuclear potential errors. No
image cancellation, k/-k relation or neutrality is used to shrink an error.

`plan_periodic_electrostatic` admits combined source storage, shell calls,
primitive products, scalar and kernel work before point payload copies.
Count-based limits include exported arrays, density copies, integral/Fourier
scratch, point-pair sums and envelope work. Identical or unresolved periodic
nonzero point charges refuse. Zero charges are ignored in interactions.
Representatives are not silently wrapped; enlarge the finite real cube when
center separation prevents its error proof. Signed or fractional charges
are allowed; positive atomic numbers represent physical nuclei.

Qualification `full-pair-image-electronic-point-electrostatics` refers to
this total electrostatic component. `values` is J+V, excluding kinetic,
exchange and XC terms, and is not a complete HF/KS Fock matrix. Roundoff,
production SCF accuracy-profile selection, RI-BUPO, gradients and stress
remain outside this increment. The shared Fock/UHF, BIPOLE physical-domain
and qualified symmetry contracts stay unchanged; exporting this component
does not establish a new symmetry certificate.


### 5.11 Bounded periodic HF with matched BUPO electrostatics

`run_periodic_bupo_hf` composes the immutable source of Sections 5.7-5.10
with native Bloch kinetic integrals and exact finite reciprocal exchange.
Its scope is neutral all-electron 3D RHF or collinear UHF, with integer spin
occupations on the original uniform full mesh. Both spins use the same frozen
source, physical image inventory, pair cutoff, Ewald domains and partition.
No density-dependent repartition or symmetry reduction occurs during SCF.
The physical system must match the source lattice and shell centers modulo
that lattice. The driver creates point nuclei from its atomic numbers and
uses the source's matched attraction and nuclear energy, including self and
gauge terms. It does not import the older midpoint-sphere nuclear support.

Put $H=T+V_n$ and $D_t=D_\alpha+D_\beta$. The spin Fock and total energy are

$$F_\sigma=H+J[D_t]-K[D_\sigma]+\Delta F_\sigma,$$

$$E=E_{nn}+\sum_k w_k\operatorname{Tr}(D_tH)
  +E_J[D_t]-\tfrac12\sum_{\sigma k}w_k
       \operatorname{Tr}(D_\sigma K[D_\sigma])+E_{\rm corr}.$$

RHF uses one total density, a factor one half in front of its exchange
matrix, and exchange energy $-\frac14\sum_k w_k\operatorname{Tr}(DK[D])$.
These are the existing HF spin conventions, with the fixed periodic BUPO
quadratic functional replacing only J. The Gaussian Bloch exchange panels
follow Sun et al. (2017), Sec. II, DOI
[10.1063/1.4998644](https://doi.org/10.1063/1.4998644), with an explicitly
supplied finite reciprocal domain. The periodic BUPO functional is the
project derivation above; Neese et al. (2025), Eqs. 29-36, do not supply
periodic HF, exchange or an SCF convergence proof.

For `BvK-ewald`, the original full mesh defines the Born-von Karman cell.
The existing full-Coulomb correction is
$\Delta F_\sigma=-\xi S D_\sigma S$, with the RHF factor one half, and
$E_{\rm corr}=\frac12\sum_{\sigma k}w_k
\operatorname{Tr}(D_\sigma\Delta F_\sigma)$.
See Sundararaman and Arias (2013), Sec. II.A,
[10.1103/PhysRevB.87.165122](https://doi.org/10.1103/PhysRevB.87.165122).
There is no extra short-range erfc zero-mode subtraction in this full
reciprocal exchange path. `strict-zero-mode` applies no such correction.
Neither choice certifies convergence of the finite exchange domain.

Every evaluated density must satisfy neutral charge admission and positive
finite electrostatic energy and maximum matrix error targets, both default
$10^{-8}$. The driver fails immediately when a target is exceeded. It
records the last and maximum observed bounds, so a final acceptable density
cannot conceal an earlier failed target. The source is never silently
expanded, repartitioned or assigned a looser target. Qualification
`electrostatic-and-one-electron-images-at-evaluated-density;finite-exchange`
also includes the S/T image bounds and admission in Section 5.12. Exchange,
basis, state-relaxation and roundoff errors remain excluded. This is
an explicit bounded SCF consumer, not an automatic whole-HF accuracy profile.

The existing bounded solvers retain the common overlap projector, fixed
occupations, coupled spin epoch and last evaluated D/F/E state. Capped runs
export that evaluated density, not the next density reconstructed from the
returned eigenvectors. Numerical convergence additionally requires a full
native Fock and energy rebuild at the same final density. Confirmation is
not a second SCF iteration. Scalar history excludes matrix snapshots.

`plan_periodic_bupo_hf` combines source, kinetic, exchange, Madelung, solver,
scalar history and owned-export storage and work under all five resource
caps before exchange payload copies. Integral counters report actual native
integral calls; the larger planned limits also cover envelope and solver
work. Source and density identities, complete energy components, nested
charge/error and partition receipts accompany the full Fock export.
Spin-major original-mesh output maps into the shared `FockBuild` contract.
Its unreduced frame carries no symmetry certificate and changes no BIPOLE
physical domain. Automatic public job/profile selection, q-aware RI-BUPO,
external SCF parity, gradients and stress remain separate requirements.


### 5.12 Full AO-image overlap and kinetic bounds

`bound_periodic_one_electron_images` consumes the immutable original
`PeriodicPairSource`. It bounds every omitted translation in overlap and
kinetic matrices, including sparse interior holes and physical-cutoff
exclusions. It uses no Coulomb kernel, eta, auxiliary fit, density or Bloch
phase cancellation. The complete integer cube and exterior Gaussian sum
are the native construction in Section 5.9, with the same checked lower
lattice metric. This is a new bound derived here from the Gaussian and
kinetic integral conventions, not an Ewald or BUPO formula.

For a Cartesian monomial of total degree l, Section 5.9 supplies
$|p_l(r)|e^{-a r^2}\le H_l(a)e^{-a r^2/2}$. Its gradient obeys

$$|\nabla[p_l(r)e^{-a r^2}]|
 \le [l r^{l-1}+2a r^{l+1}]e^{-a r^2}
 \le G_l(a)e^{-a r^2/2},$$

$$G_l(a)=lH_{l-1}(a)+2aH_{l+1}(a),$$

with the first term omitted for l=0. The Cartesian polynomial gradient
norm is at most $l r^{l-1}$ because the sum of squared component powers is
at most $l^2$. Absolute contraction coefficients and maximum representation
row sums then bound each pure or Cartesian component and its gradient.

For one primitive pair, let C include these coefficient/representation
factors, $p=(a+b)/2$, $\mu=ab/[2(a+b)]$ and $d_n=|A-B-Ln|$. Bounds on the
overlap and kinetic contribution of translation n are respectively

$$q_S(n)=C H_l(a)H_m(b)(\pi/p)^{3/2}e^{-\mu d_n^2},$$

$$q_T(n)=\tfrac12 C G_l(a)G_m(b)(\pi/p)^{3/2}e^{-\mu d_n^2}.$$

The second uses the exact localized-Gaussian identity
$T_{ab}=\frac12\int\nabla\phi_a\cdot\nabla\phi_b$, obtained by integration
by parts with vanishing Gaussian surface terms. Both use the same decay
and exterior lattice sum. This is an integral-error enclosure, not a new
nuclear-gradient or stress implementation.

Summing only the disjoint missing images gives nonnegative AO matrices
$B^S,B^T$ with $|\Delta S_{ij}(k)|\le B^S_{ij}$ and
$|\Delta T_{ij}(k)|\le B^T_{ij}$ uniformly over all k. The original pair
involution restores both ordered arms, including nonzero diagonal images.
No relation between states at k and minus k is assumed. Logarithmic positive
sums retain a positive subnormal enclosure instead of rounding a nonzero
bound to zero. Roundoff is still excluded from the scientific bound.

For the total evaluated spin density and original nonnegative weights,
put $W_{ij}=\sum_k w_k\max(|D_{ij}(k)|,|D_{ji}(k)|)$. Then

$$|\Delta E_T[D]|\le\sum_{ij}W_{ij}B^T_{ij}.$$

Periodic BUPO HF computes these immutable envelopes once after joint
admission. It checks maximum overlap and kinetic matrix errors before SCF
(default targets 1e-10 and 1e-8), and the positive density-weighted kinetic
energy error at every evaluated state (default 1e-8 hartree). Targets must
be positive and finite. The last and maximum observed kinetic energy
bounds accompany the existing electrostatic ledger and confirmation.
A target failure refuses; it never changes the fixed source or projector.

`plan_periodic_one_electron_images` admits the retained pair source,
primitive gradient envelopes, both error matrices and owned exports under
all five caps. Counts include finite completion enumeration and omitted
primitive evaluation. The enclosing HF plan additionally budgets this setup
and its retained matrices. Receipts bind the pair source, cube and metric,
report maxima and counts, and declare `roundoff_included=False`.

These are bounds at fixed evaluated D, not a guarantee about the relaxed
SCF state, overlap-rank stability or basis completeness. Exchange, including
its overlap-dependent BvK correction, remains a separately finite source
unless the explicit certification in Section 5.13 is requested.
No whole-HF accuracy profile, public source selection or symmetry certificate
follows from the new ledger. Existing Fock, BIPOLE and symmetry contracts
are unchanged; q-aware RI-BUPO and the rest of the capability matrix remain.

### 5.13 Full reciprocal and AO-image exchange bounds

`PeriodicExchangeBoundSource` certifies the full-Coulomb exchange convention
of Section 5.3 at fixed original k points, weights and evaluated spin density.
It consumes the immutable pair source, the actual ordered Q vectors and
offsets, and an explicit reciprocal completion-cube radius. It does not
apply the q=0 pair involution to an exchange panel. The involution is used
only to recover the absolute retained and omitted ordered-image amplitudes.
The Gaussian Fourier and Bloch conventions are those of Sun et al. (2017),
Section II.A, Eqs. (3), (10) and (16), DOI 10.1063/1.4998644. The error
majorants below are a project derivation, not a published MDF approximation.

For a primitive pair of degrees l,m, write P=a+b. Shifting the entire
Gaussian Fourier contour by $-iQ/(2P)$ gives the exact Gaussian factor
$\exp[-Q^2/(4P)]$ and polynomial coordinates shifted by the same amount.
The absolute Cartesian polynomial is bounded by
$(|r-A|+|Q|/(2P))^l(|r-B|+|Q|/(2P))^m$. Expand the binomials, absorb
spatial powers in half Gaussians with Section 5.9's H functions, and use

$$\sup_{q\ge0}q^d e^{-q^2/(8P)}=(4Pd/e)^{d/2},$$

where the d=0 factor is one. Each primitive image then obeys

$$|F_{ab,n}(Q)|\le A_{ab,n}\exp[-Q^2/(8P)],$$

$$A_{ab,n}=C(2\pi/P)^{3/2}e^{-ab|A-B-Ln|^2/(2P)}
 \sum_{u=0}^l\sum_{v=0}^m {l\choose u}{m\choose v}
 H_{l-u}(a)H_{m-v}(b)
 \frac{(4P(u+v)/e)^{(u+v)/2}}{(2P)^{u+v}}.$$

C includes absolute normalized contraction coefficients and representation
row sums. Summing the disjoint retained and missing images with Section
5.9's checked metric gives AO matrices R and M. A common positive
$\alpha\le1/(8P)$ for every primitive yields
$|B_{ret}(Q,k)|\le R e^{-\alpha Q^2}$ and
$|B_\infty-B_{ret}|\le M e^{-\alpha Q^2}$, entrywise and uniformly in k.
For an entirely s-type source, no Fourier polynomial is absorbed and the
stronger common $\alpha\le1/(4P)$ applies.

For each target/source transfer $\delta=k_t-k_s$, let

$$c_{ts}=\sum_{Q\in G_{ret}+\delta}\frac{4\pi}{\Omega Q^2}
 e^{-2\alpha Q^2},\qquad
 h_{ts}=\sum_{Q\notin G_{ret}+\delta}\frac{4\pi}{\Omega Q^2}
 e^{-2\alpha Q^2}.$$

Both exclude Q=0 exactly, matching full Coulomb exchange. Native upper
bounds enumerate the declared completion cube, including sparse holes.
The exterior uses the shifted Gaussian slab bound and
$|Q|\ge\sigma_G(R_G+1)-|\delta|>0$. An insufficient separation refuses.
The reciprocal metric takes the stronger of the checked inverse-residual
bound and a guarded Gershgorin lower bound on the reciprocal Gram matrix;
the latter needs no approximate eigenvector or lattice-axis assumption.
The checked real metric cubed bounds volume from below. Supplied vectors
must lie on their transfer lattice, inside the declared cube, be unique,
nonzero and closed under transfer reversal. No support is silently added.

With $A=R+M$ and nonnegative source weights, the two disjoint K errors are

$$B^{image}_t=\sum_s w_s c_{ts}
 [R^T|D_s|M+M^T|D_s|R+M^T|D_s|M],$$

$$B^{reciprocal}_t=\sum_s w_s h_{ts} A^T|D_s|A.$$

Their sum bounds the full-image, full-reciprocal K error for arbitrary
Hermitian D, including signed densities. The exchange energy error in each
spin channel is bounded by half the weighted absolute density contraction
with this matrix. Positive logarithmic matrix products retain nonzero
subnormal bounds; zero density produces exactly zero. No four-index tensor
or all-q AO panel is retained. Joint plans admit constructor image/cube work,
source storage, density copies, positive cubic contractions and owned exports
under all five resource caps before numerical payload reads.

Explicit periodic BUPO HF certification uses `bound_exchange=True` and
`exchange_bound_cube_radius`. It scales K bounds by one half for restricted
total density and by one for each unrestricted spin. For the declared fixed
Madelung coefficient, the additional overlap-dependent correction is bounded
entrywise by

$$s|\xi|[|S||D|B^S+B^S|D||S|+B^S|D|B^S].$$

This follows by expanding $(S+\Delta S)D(S+\Delta S)-SDS$. The coefficient
and original BvK cell remain method-owned as in Section 5.11. The sum of
uncorrected and correction errors must pass the positive finite matrix and
energy targets at every evaluated density, default 1e-8 in atomic units.
The last and maximum observed bounds and native source/density identities
are recorded. Failure refuses before exchange evaluation and does not enlarge
the physical domains, change the Hamiltonian, or alter the SCF state.

`bound_exchange=False` preserves the explicitly finite exchange route and
its prior qualification. With certification enabled the qualification is
`electrostatic-one-electron-and-exchange-at-evaluated-density;fixed-mesh-and-madelung`.
This still excludes Brillouin-zone quadrature error, Madelung evaluation error,
basis incompleteness, relaxed-state and overlap-rank stability, and roundoff.
It is not an automatic whole-method accuracy profile or a symmetry certificate.
Range-separated exchange, q-aware RI-BUPO, public job selection and the
remaining capability matrix require their own method qualification.

## 5.14 BUPO HF reciprocal cutoff selection

The exchange-domain factory consumes the immutable electronic BUPO source
from Sections 5.7-5.11 and the full exchange certificate from Section 5.13.
It selects physical vectors by the shared certified reciprocal-energy
predicate, for every centered target-minus-source mesh fraction. Gamma excludes
the integer zero vector. Non-Gamma channels retain the existing predicate's
small numerical boundary padding; a new approximate norm test is not used.
This is the same Gaussian Bloch exchange formulation of Sun et al. (2017),
Section II, DOI 10.1063/1.4998644, with project-owned finite support and error
accounting. It is not a new published periodic BUPO approximation.

For each transfer, certified integer box edges bound all candidate labels.
Exact integer wraps map these edges into the original raw-transfer label
frame: `q_raw = q_centered + wrap`, `n_raw = n_centered - wrap`.
The mapped edges' maximum absolute component defines the exchange
certificate's completion cube. Counting accepted vectors never shrinks it.
The certificate therefore accounts for both unselected vectors inside the
cube and the exterior tail. Centered opposite fractions can differ from
exact negatives at a half-grid boundary; their integer partner map includes
the corresponding reciprocal shift. In the raw frame it is `n -> -n`.
Equal accepted counts and the shared opposite-source predicate audit
establish closure before any exchange
calculation. Empty spheres can be represented, but their missing-exchange
bound must still pass the unchanged HF targets before evaluation.

Count-only planning reserves all transfer plans, counting, conjugacy audits,
replay, domain hashing, native and owned export storage, and complete HF
work. All five caps precede vector enumeration. Source-specific electronic,
AO-image, tree, near/far and nuclear domains remain unchanged. The factory
requires exchange certification and derives only the completion radius;
it does not alter density-dependent tolerances or adapt the Hamiltonian.
Result identities bind the original source, lattice, mesh, cutoff, radius,
offsets and executed vectors. Numerical HF state remains the same as an
explicit-vector run over those arrays, including nonconverged states and
fixed-Madelung energy corrections.

The complete RHF/UHF calculation still reports evaluated-density integral
bounds rather than basis, mesh, relaxed-state or roundoff accuracy. This
factory is a prerequisite for public BUPO source selection; it does not
complete automatic AO/Ewald selection, whole-method accuracy profiles,
q-aware RI-BUPO or operation-qualified symmetry admission.

## 5.15 Complete native BUPO HF source preparation

`prepare_periodic_bupo_hf` constructs the pair-image source, bubble tree,
Ewald source, fixed near/far partition and certified exchange spheres from
the physical basis, neutral all-electron 3D system and full regular mesh.
The source policy explicitly supplies integer cube radii, pair cutoff,
Gaussian extents, moment order, hierarchy controls, Ewald eta and partition
targets. It composes the existing producers in Sections 5.7-5.14 without
changing their algorithms, domains or error definitions. Each nuclear
center must still match its original basis center modulo a lattice image.

Preparation executes scientific work. A later stage's plan depends on
earlier immutable source contents, so this API is not a count-only dry run.
It first admits generated labels, momenta, physical-input copies and
metadata. It then plans and admits each source stage before constructing it.
All five cumulative limits retain the earlier work charges. The live memory
reservation sums the existing stage peak models, including their shared
parent and workspace reservations. This deliberately overcounts storage;
it is a conservative ownership model, not measured allocation or RSS.
The final exchange-domain plan includes generation and complete bounded HF
under the remaining work caps and the accumulated live reservation.
Failure at any stage releases the partially constructed source.

The prepared handle owns physical-input and policy copies, source objects
and exchange support. It contains no density, Fock matrix, orbitals or SCF
history. Every run starts a new electronic state over those fixed inputs.
One active run lease per handle prevents simultaneous workspace ownership
from exceeding its admission; separate handles need separate budgets.
Each result reports both source preparation and run costs, conservatively
charging preparation even when an existing handle is run again.

The receipt identifies the original pair, tree, Ewald and BUPO sources and
the copied source policy. Its physical identity also binds the nuclear
input, exchange support, spin convention and zero-mode correction. Solver
iterations and accuracy targets do not become physical-source identity.
The returned evaluated density, Fock and energy retain the existing shared
Fock consumer contract, including capped states. The prepared source grants
no operation, support-equivariance or roundoff attestation for symmetry.
It does not alter BIPOLE domains or establish CRYSTAL parity.

Both source bounds and full exchange bounds are mandatory. Every evaluated
density must pass the existing electrostatic, S/T and exchange targets.
This establishes explicit source construction and a selected HF API;
ordinary job dispatch is described in Section 5.16. Automatic whole-method
profiles, q-aware RI-BUPO, mesh and relaxed-state accuracy remain separate capabilities.

## 5.16 Public BUPO HF job dispatch

The public job consumer uses the prepared source of Section 5.15 without
changing any numerical kernel, integral target, fixed domain or evaluated
state. Typed `PeriodicBupoIntegralOptions` declares matched integer-cube
Ewald domains separately from radial integral options. The existing
`CoulombBuildOptions.bupo` supplies moment, extent and tree controls. The
route admits only neutral all-electron 3D RHF/UHF, exact reciprocal K and
full regular meshes. Both source certificates and exchange bounds remain
mandatory. Geometric symmetry and producer qualification remain distinct.

Before source construction, the runner reserves output/history storage,
policy/provenance copies, primitive shell serialization and their modeled
work. A count-only pair plan supplies the primitive census. Complete source
preparation then runs once under the remaining budget and keeps its live
source reservation through SCF and output. The provenance distinguishes the
conservative preparation admission from actual-support result modeling;
complete-job scalar/kernel totals include the output work reservation.
The native memory model already includes caller/output live reservations.

Output discovery does not execute scientific preparation. Its estimate is
the declared complete-job memory cap, explicitly an upper reservation for
admitted execution, not a source-content prediction or feasibility verdict.
All scientific stage checks still run during execution. This preserves
cheap dry-run artifact discovery without silently doing integral work.

Restricted results remove the native singleton spin axis at the job export
boundary. Spin-resolved UHF, empty beta, nonconverged density/Fock/energy and
QVF AO-gauge conversion retain the existing physical state contract. The
job does not reconstruct the evaluated density from terminal orbitals.
No post-HF, gradient, symmetry, RI-BUPO or whole-method accuracy capability
is granted by this dispatch increment.
