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

For a plain s-s pair the polynomial is constant, so no absorption is
necessary. Let A be the largest primitive exponent with nonzero coefficient
in the basis. The original half-Gaussian model already has maximum exponent
A, from that primitive paired with itself. Choose
$t=\min(1,A/(a+b))$, which lies in $[1/2,1]$. For a constant polynomial,
reducing both primitive exponents by t gives a pointwise majorant with no
extra prefactor. Its product mass has $\alpha=t(a+b)$,
$\mu=tab/(a+b)$ and amplitude $M(\pi/\alpha)^{3/2}$. This uses the exact
product exponents when $a+b\le A$ and otherwise the tightest common scaling
that stays within the original kernel ceiling. This keeps the common kernel bound unchanged while
reducing or preserving every positive retained and omitted mass, so image
J/energy enclosures cannot worsen, including mixed angular bases. The
maximum-exponent scan fits within the existing primitive-work reservation.
Contraction coefficients still enter through absolute primitive products, including signed
contractions. Angular pairs keep the half-Gaussian envelope above.
Gradient and Fourier-contour producers retain their separate half-Gaussian
formulas; the derivative of an s Gaussian has a nonconstant polynomial.

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

### Complete RI methods: canonical-density image admission

Periodic RI HF and KS can request three positive image targets in
`PeriodicOneElectronAccuracyOptions`: maximum overlap entry, maximum kinetic
entry and kinetic energy in hartree. All zero disables this optional ledger.
The same shared image producer constructs B^S and B^T from the actual finite
Bloch translations and physical shell-pair cutoff. Integer lattice labels,
Cartesian reconstruction and both ordered cutoff arms are checked. A retained
block absent from the physical Bloch source is a refusal, never a hidden
change of domain. Integral/phase roundoff is excluded.

An explicit `completion_cube_radius` chooses the enclosure evaluation cube.
It need not contain every retained image: the exterior positive Gaussian sum
then includes some retained terms, which overbounds the missing sum. Increasing
this radius does not change S, T, the fitting domain or the Hamiltonian. The
legacy source-only overload still uses its original inventory radius.

The bounded solver's shared canonical-overlap helper supplies the uniform
entry envelope H[k,i,j] and total L described in the automatic RI density
sections below. Both apply to every audited occupied subspace, including
restricted-total density and an empty spin channel. Thus, before SCF,

```text
max(B^S) <= overlap_error
max(B^T) <= kinetic_error
min(sum_kij H[k,i,j] B^T[i,j], L max(B^T)) <= kinetic_energy_error
```

The last inequality holds for the same finite canonical-overlap density
family, with fixed spin occupations and original uniform full-mesh weights.
It does not compare densities or overlap ranks obtained from different
physical domains. The existing log-positive trace producer prevents a
nonzero subnormal enclosure from becoming zero. Automatic RI coupling
selection and image admission reuse one canonical density analysis; an
explicit RI density override keeps its own provenance unchanged.

Count-only plans admit pair-source construction, Gaussian bounds, the
canonical analysis, both error matrices, owned exports and output records
before copying inputs. Source, envelope, targets and result have separate
identities. Complete HF, pure KS, global hybrid and physical range compositions
retain their Fock source identities and energies when this ledger is enabled;
exact J with RI K is included. Non-RI public jobs refuse this option. Nuclear,
two-electron, metric/solve, arithmetic and relaxed-state errors remain outside
this certificate. No symmetry reduction or whole-method accuracy claim follows.

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

#### Canonical occupied-space exchange enclosure

`bound_occupied_subspaces(S, n_alpha, n_beta, scf)` reuses the bounded
solver's canonical overlap and occupied-vector Gram audits. It requires the
source's original weights to equal $1/N_k$. Its returned density envelope
has the stronger spin-separated property

$$H_{kij}\ge \frac{1}{N_k}\sum_\sigma |D_{\sigma,kij}|,
\qquad L_k\ge\sum_{\sigma,ij}|D_{\sigma,kij}|.$$

Indeed each spin contributes its own Gram bound times the product of the
two canonical row norms. Summing those positive bounds before applying the
triangle inequality encloses the sum of absolute spin entries, including an
empty spin channel. It is not merely a bound on a possibly cancelling spin
sum. The retained canonical space, occupations and algebra audits stay fixed.

Let $P_k=N_k H_k$, rounded outward, with each transpose pair replaced by its
positive maximum. The existing exchange enclosure is a positive linear map
$\mathcal B$. Therefore $\mathcal B(P)$ bounds the omitted-image plus omitted
reciprocal error in every spin K throughout this density family. It also
bounds the error for the total density. With $B_k=\mathcal B(P)_k$,

$$|\Delta E_x|\le \frac12\sum_{kij}H_{kji}B_{kij},\qquad
|\Delta E_x|\le\frac{1}{2N_k}\sum_k L_k\max_{ij}B_{kij}.$$

The implementation intersects the outward positive contractions. Its entry
contraction uses the slightly larger symmetrized P, preserving the inequality.
The energy is the sum of unscaled spin exchange energies. Restricted total
D uses one half of the corresponding total-density K response; the returned
per-spin enclosure still bounds that response. A hybrid coefficient requires
its absolute method-owned scale. These are fixed-density-family component
errors, not changes between relaxed states or different retained spaces.

`bound_density_envelope` accepts an existing native canonical envelope to
avoid a second overlap analysis in future method consumers. Its qualification
is conditional on that declared envelope; it does not prove that the supplied
overlap equals the physical pair source's overlap. It checks dimensions,
rank, occupations and positive bounds, and accepts no user array in place of
the typed envelope. Source and envelope identities are distinct. A certificate
identity binds both and the resulting matrix/energy limits. No evaluated SCF
density or Fock snapshot is manufactured from the positive envelope.

Count-only plans cover analysis, envelope copies, logarithmic contractions,
source storage and owned exports before overlap values are copied. The
already-built source has no additional shell or primitive integral calls.
The two energy limits and the image/reciprocal matrices remain inspectable.
The existing evaluated-density path retains its numerical results and identity.
The standalone API does not select physical supports, impose complete RI
method accuracy, or include the BvK correction or Madelung numerical error,
or certify erfc exchange, fitting/metric error, roundoff, state relaxation
or symmetry reduction. Those contributions remain separate required work.

#### Complete RI exchange accuracy admission

RI HF and RI full-Coulomb global hybrids can now impose separate positive
matrix and energy targets through `PeriodicExchangeAccuracyOptions`. The
selected K must be the independent exact reciprocal producer. Fitted K,
pure KS and physical range kernels explicitly refuse this option; their
missing error terms cannot be certified by a Coulomb component alone.

The method reuses its single canonical density analysis for automatic RI
coupling, optional S/T image accuracy and exchange accuracy. It reconstructs
the exact finite Cartesian AO-image source with the same shared integer-label,
roundtrip and both-arm cutoff audit used by S/T admission. Reconstruction
must not retain a block absent from the actual Bloch producer. The original
ordered exchange Q vectors, transfer offsets and uniform full mesh enter the
shared exchange-bound source unchanged. An explicit AO completion cube must
contain every retained pair image; omitted inventory labels may lie outside.
This keeps the retained Fourier amplitude enclosure valid. The standalone
producer's omitted optional radius retains its previous source and values.

Write $B^K$ for the uniform positive exchange bound and
$P=N_kH$ for the outward density envelope. For fixed finite overlap S,
image bound $B^S$, and the method's fixed BvK coefficient $\xi$, define

$$B^C=|\xi|\,[|S|P B^S+B^S P|S|+B^S P B^S].$$

The admitted method matrix bound is $B=s(B^K+B^C)$, where
$s=|a_x|/2$ for restricted total density and $s=|a_x|$ for separate spin
densities; HF uses $a_x=1$. Omitted-zero-mode calculations have $B^C=0$.
The energy bound intersects the positive entry contraction
$\frac12\sum_{kij}H_{kji}B_{kij}$ with
$\sum_k L_k\max_{ij}B_{kij}/(2N_k)$. Scaling and positive products are
performed in logarithmic arithmetic so a nonzero subnormal coefficient is
not silently dropped. This compares the same finite-canonical-space density
family under complete AO/reciprocal K and, when selected, complete overlap
in the fixed BvK correction. It does not change the Madelung convention.

Both targets must pass before the RI builder and SCF iteration are created.
Failure refuses; it neither enlarges an in-flight source nor changes the
Fock/energy equations. The receipt keeps method scale, fixed coefficient,
uncorrected exchange and overlap components, combined limits and distinct
source/envelope/overlap/certificate identities. Automatic RI density
provenance remains separate from an explicitly supplied RI density limit.
Count-only plans cover reconstruction, source construction, envelope reuse,
positive correction work, retained results and Python output before input
copies. Child budgets reserve the already admitted outer live state once.

Madelung numerical error, finite fitting/metric error, other Hamiltonian
components, arithmetic, mesh convergence, overlap-space changes and state
relaxation remain outside this ledger. Physical source selection and
whole-method automatic RI-BUPO accuracy are still incomplete. Unified Fock
states, BIPOLE/CRYSTAL conventions and source-qualified symmetry permissions
are unchanged by enabling this optional certificate.

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


### Ordered GDF auxiliary-image hierarchy

The optional `coupling_multipole_hierarchy` refines the ordered finite GDF
coupling approximation. For each ordered pair image, the existing capsule
selector supplies the complete auxiliary image inventory. The native bubble
producer groups that inventory using its deterministic median tree, with
leaf capacity one, branching three and maximum depth 32. No canonical
periodic AO-pair source replaces this support, and no second selector is used.

For auxiliary shell component P, node v carries the complex raw Cartesian
moments `M_v(P) = sum_T exp(-i q.T) W(c_T-c_v) M_P`. The shared translation
producer evaluates W; child moments translate upward and are reused at their
parent. The accepted contribution is `M_v T_erfc(c_v-c_pair) M_pair^T`,
followed by the unchanged `exp(i k.R)` pair phase. Both paired actions consume
these same density-independent tiles. The finite metric, retained factor,
reciprocal panels and zero correction are unchanged.

The shared bubble envelope bounds each descendant distribution separately.
Its translated full and tail moments are maxima, never phase-cancelled sums.
An accepted N-image node therefore contributes at most N times the shared
erfc Gaussian bound. The allowance remains the requested shell-tile epsilon
divided by the total selected image count. Overlap, always-near flags,
unrepresentable interaction bounds and excessive error force descent; terminal
refusals use direct MD. The enclosing sphere and the translated tail envelopes
are essential: applying an isolated-shell bound at a node center is invalid.

This uses the upward translation and auxiliary hierarchy of Neese et al.,
DOI `10.1021/acs.jpca.4c07415`, section 2.3 equations 23-24 and section 3.2,
with raw Cartesian total-degree erfc expansions instead of harmonic Coulomb
moments. Order stays fixed during translation; the explicit node remainder
controls the resulting error. The paper's automatic order increments and
density-compressed two-tree implementation are not yet implemented here.

The uncached route retains one pair image's selected tree, radial envelopes
and complex moment panels at a time. Admission includes tree construction, at most
2N-1 error queries, upward translations, moment contractions and simultaneous
workspace. There is no full auxiliary-by-pair tensor or rectangular image
interaction list. Bounded partition reuse and compressed hierarchical actions
are described below; the streaming route remains the numerical reference.
Counters distinguish represented images, accepted blocks, nodes and
translations. This source does not certify symmetry reduction, BIPOLE or
CRYSTAL parity. Existing approximation exclusions, including RI exchange and
neutral correlated-fit exports, remain in force.


### Producer-owned periodic RI Gaussian preparation

`PeriodicGDFTiles` owns an optional bounded preparation cache. Auxiliary
entries are keyed by source shell; pair entries are keyed by the ordered bra
and ket shells and the pair-image ordinal from the shared unscreened GDF
selector. A cached pair verifies its shifted ket center on reuse. The
coupling-only multipole mode forbids integral screening, so every auxiliary
shell observes the same ordered geometric pair inventory. The source owns
copied bases, cell and controls for its entire lifetime; no global basis or
image registry participates.

The same native extent and moment functions produce each entry. A mutex
serializes extent/support-count insertion and an entry-local `call_once`
publishes each moment panel. Published numerical arrays are immutable.
Moments remain lazy, preserving always-near/direct behavior and avoiding
unneeded moment evaluations. A failed initialization is not published.
Separate producer instances have independent caches; access order, density
and caller mutations do not alter their numerical source or partition.

One integer per shell triple retains the exact selected-image count used in
the existing epsilon allocation. It stores no image interaction decisions.
The actual capsule selector still visits every integral/image needed by a
tile. There is no full T/B, replicated basis, or auxiliary-by-pair image
interaction list. Node charges and fields remain call-local. A separately
bounded cache can retain trees and certified decisions as described below.

Let I bound pair-image candidates, S be the orbital shell count, A the
auxiliary shell count, n and m the AO/auxiliary dimensions and H the number
of Cartesian moments. Admission reserves I*S*S pair slots, A auxiliary
slots, A*S*S support-count scalars and H*(I*n*n+m) real moment elements,
including map/entry ownership and extent vectors. This capacity is live in
every parent RI/Fock plan even before materialization. Cold numerical work
remains reserved on all calls, together with bounded lookup/publication work;
a warm cache never permits bypassing admission before input inspection.

`coupling_multipole_reuse` changes execution/storage policy only, so it does
not change the coupling source identity. Cold, warm, streaming and dense
paths must agree with the same numerical partition. The finite metric and
paired contraction remain unchanged. Method receipts distinguish newly built
moments/envelopes and reused data. These counters do not turn approximation
bounds into arithmetic-error or symmetry certificates. The published
algorithms and their limits remain those of the shared extent, raw moment,
BUPO translation and erfc producers cited above.


### Compressed periodic RI far actions

For one accepted auxiliary node and ordered pair image R, write its coupling
as `exp(i k.R) A I P^T`, where P is the real native AO-pair moment panel,
I is the same real erfc translation matrix and A is the complex auxiliary
moment panel produced by phase-weighted upward translations. The shared
finite selector and error queries choose the same density-independent
antichain as the materialized tile route.

The raw forward action forms `z = P^T sum_k exp(-i k.R) X_k`. Each accepted
node receives field `I z`. Transposed native translations propagate this
field downward, and each auxiliary image contributes `exp(+i q.T) M_aux f`.
The raw adjoint starts from `M_aux^T c`, translates these charges upward with
`exp(-i q.T)`, accumulates `I^T z_node` in pair moment space, then projects
once with P and distributes `exp(+i k.R)` over the output mesh. Ordinary
transpose and complex conjugation have distinct roles; the two maps remain
adjoints under the complex Frobenius product for arbitrary complex inputs.

This is the input-moment/output-field organization of Neese et al.,
DOI `10.1021/acs.jpca.4c07415`, equations 29-36, with MANTA's already qualified
ordered periodic support, nonharmonic erfc kernel and fixed shared partition.
The existing molecular method's canonical pair weights do not transfer to
this ordered Bloch source. The finite metric and paired RI energy derivative
are unchanged; no approximate metric, weight folding or new Fock owner appears.

The far map is not reconstructed for its magnitude diagnostic. At each
auxiliary leaf let b be the componentwise maximum absolute row of its moment
panel. Upward `sum |W| b_child` bounds every row of A without phase-cancellation
credit. Likewise p bounds each row of P. Then `b^T |I| p` bounds every entry
of that far block. Sum these positive bounds over accepted blocks/images and
add the measured maximum of the near/reciprocal/zero residual tile. Guarded
positive accumulation supplies the existing `maximum_coupling` upper bound
used in fixed-density RI J error propagation. The approximation ledger is
unchanged; arithmetic and phase roundoff remain explicitly excluded.

Per-action charges, downward fields, compensation and magnitude envelopes
have admitted storage and work. Structural activity follows accepted nodes,
including for zero density; it does not change the partition. Immutable source
preparation can be shared across concurrent calls, but compressed inputs and
fields are call-local. No source retains a density or metric solution.
Dense tile exports continue through the full materialized producer path.

This compresses the actions of the selected auxiliary hierarchy, but retains
one tree per shell triple and pair image. Global AO-pair bubble aggregation,
tighter automatic accuracy, source-qualified symmetry and the rest of P3b
remain unfinished. Bounded cross-pass partition reuse is described below. Existing J-only exchange,
correlated-fit export and physical-kernel restrictions stay in force.


### Source-owned bounded periodic RI partitions

Partition reuse is an execution policy over the same qualified numerical
source. A cache key comprises auxiliary, ordered bra and ket shell indices,
and the pair-image ordinal supplied by the exact unscreened GDF selector.
The source owns copied geometry and immutable approximation settings. Reuse
verifies the selected-image count and shifted ket center; no canonical Ewald
inventory or symmetry label substitutes for the ordered support.

Each immutable entry contains the shared `AuxiliaryImageBubbles` tree and an
ordered antichain of certified far-node matrices/bounds or direct leaves.
Construction uses the unchanged `BubbleErrorModel`, strict separation test,
native erfc interaction and per-selected-image tolerance. Replay preserves
traversal and compensated contraction order, represented-image factors and
the evaluated error sum. Gaussian moments remain independently prepared;
input charges, fields and magnitude envelopes remain private to each action.
No density, metric solution, full T/B or rectangular image tensor is cached.

A source-local mutex serializes lookup, bounded FIFO eviction and complete
entry publication. Failure never publishes a partial entry. Oversized entries
stream without retention. Before construction, a conservative bound covers
the shared tree and at most 2N-1 decision records with H-by-H Cartesian
interaction matrices. Actual retained accounting includes tree storage,
decision-vector capacity, matrix elements and per-entry/map bookkeeping.
All arithmetic for storage and work admission uses checked counts.

The source plan reserves the configured capacity plus 8 KiB bookkeeping,
and a separate full-capacity workspace for an in-flight or evicted borrowed
entry. Existing producer workspace, cold certification and translation costs
remain admitted, with additional lookup/copy work. Parent RI/Fock plans compose
these bounds before numerical input inspection. Cache history never relaxes
admission, changes the mathematical source identity or authorizes symmetry.

This is bounded reuse of the published BUPO translation/certification path
already mapped above, not a new periodic approximation or a claim that the
molecular paper specifies a FIFO cache. Global AO-pair aggregation, automatic
accuracy and the remaining capability matrix still require implementation.


### Call-wide periodic RI charges and fields

The optional global-action policy factors repeated moment contractions out
of the per-pair auxiliary-tree traversal. For ordered pair image r, form
`z_r = P_r^T sum_k exp(-i k.R_r) X_k` once. If `f_a,r,T` is the auxiliary
leaf field obtained from the unchanged accepted interactions and downward
translations, accumulate `h_a = sum_r,T exp(+i q.T) f_a,r,T` with compensation.
The forward far contribution is `y_a = M_a h_a`, projecting each auxiliary
shell only once after all pair images. Near, reciprocal and finite-zero terms
remain in their original residual tile and are added separately.

The adjoint starts from one `M_a^T c_a` per auxiliary shell. The existing
phase-weighted upward translations and real erfc interactions contribute
fields to each ordered pair r. Sum those fields across every auxiliary shell
before projecting with `P_r` and distributing `exp(+i k.R_r)`. These operations
are the conjugate transpose of the forward map. Canonical molecular pair
weights, mesh weights and metric solves do not enter this raw coupling stage.

This uses the same input-moment/output-field equations of Neese et al.,
DOI `10.1021/acs.jpca.4c07415`, equations 29-36, with the qualified periodic
phases and finite masks described above. The pair key is the ordered bra/ket
shell and exact producer image ordinal; the auxiliary key is its AO offset.
Entries borrow immutable source-owned moments and verify their owners. All
charges, fields, compensation and phases are action-local; no density or
metric solution becomes part of source preparation or partition retention.

With I candidate pair images, S orbital shells, A auxiliary shells, H moments
and K mesh points, additional admitted storage is
`8192 + I*S*S*(2048 + 96*H + 64*K) + A*(1024 + 64*H)` bytes.
Checked counts also reserve map work for `A*I*S*S + A` visits. Existing cold
producer, compressed-action and partition-cache allowances remain in force;
saved contractions do not discount admission. This is an upper bound, not
measured residency, and is composed into parent RI/Fock plans before data
inspection. Exact, leaf, unprepared, materialized and reciprocal-only routes
do not activate it.

The execution policy changes snapshot identity and summation order, not the
mathematical coupling identity, Gaussian error ledger or fixed metric. It
provides no roundoff, symmetry or external-program parity certificate. A true
global pair-bubble traversal remains required: pair-dependent auxiliary
capsule masks cannot be replaced by the canonical Ewald pair inventory.


### Producer-qualified global pair geometry

The shared GDF pair tree preserves the original ordered sphere selector and
the exact pair-dependent auxiliary capsules. For bra shell s, choose an integer
frame f_s from its fractional position (roundoff-close integers are stabilized
before taking the floor). Translate both endpoints and the Gaussian extent by
`-L f_s`. This changes their representation only. For auxiliary shell a with
frame f_a and canonical image t, its original image for that pair is
`t_original = t + f_s - f_a`.

The raw coupling phase factors exactly as
`exp(i k.R - i q.L f_s) * exp(-i q.L(t-f_a))`.
Their product is the original `exp(i k.R - i q.L t_original)`. The first factor
belongs to the pair input/output stage and the second to the auxiliary stage;
the nonzero-q bra-frame term must not be dropped. This source mapping permits
global spatial aggregation without substituting a different pair inventory.
Neither a reduction certificate nor any new Fock normalization follows.

For a node, enclose all first endpoints around A and all corresponding second
endpoints around B with a common displacement bound delta. Every descendant
segment is within Hausdorff distance delta of segment AB. Therefore the distance
d from an auxiliary point to AB gives uniform capsule bounds d-delta and
d+delta. A node is wholly inside if the upper bound is strictly below the
cutoff, and wholly outside if the lower bound is strictly above it. Roundoff
guards turn uncertain cases into refinement. Single-pair leaves evaluate the
original shared double-precision predicate in the original source frame.
These masks are exact discrete support decisions, separate from Gaussian
extent radii and the erfc approximation ledger.

Every selected auxiliary image lies within (pair cutoff + auxiliary cutoff)
of at least one canonical bra center. The existing conservative integer-box
enumerator supplies a superset; integer labels are deduplicated and exact masks
remove empty images. This preserves skew-cell and image-frame support without
retaining a rectangular membership tensor. The union capacity is the orbital
shell count times the expanded image-box bound. Admission includes its ordered
set, result/export copies, per-query traversal and the source-owned preparation.
Tree/endpoint construction is bounded by the shared depth and candidate pair
inventory. All resource caps apply before preparation is materialized.

The shared native bubble builder, source-owned Gaussian extents and published
BUPO translations remain those mapped above. This increment supplies qualified
global pair geometry and masks. The consumer below retains its source
identity and separately certifies interactions over wholly included nodes;
geometry does not supply an approximation or accuracy certificate.


### Global ordered pair-tree RI-BUPO coupling

`coupling_multipole_pair_tree` defines a new fixed approximate short-range
coupling. It reuses the above source frames/masks, native bubble error model,
Gaussian panels and Neese equations 29-36. Each complete auxiliary-image
union is traversed against the global ordered pair tree. An accepted block
must be wholly inside its finite capsules and separately pass the native
erfc Gaussian interaction bound. Otherwise it refines; an unaccepted leaf
uses its original source-frame integral.

For an accepted node b, forward charges are the sum of translated pair
moments times the pair-phase-folded input. The interaction maps that charge
to the auxiliary field with the conjugate auxiliary phase. The adjoint uses
the same fixed interaction matrix, unconjugated auxiliary phase, transposed
translations and unconjugated pair phases. Both share the same antichain;
all input, charges and fields are call-private. Dense exports expand the same
interactions. No density-dependent accept/reject decision enters this map.

Let N be the largest actual count of admitted (R,T) interactions over source
shell triples. A block is admitted only if its uniform per-descendant error
is at most epsilon/N, with a conservative roundoff guard on that allocation.
The evaluated coupling bound sums accepted errors down the shared pair tree,
then sums original R images only within each orbital shell pair. If E[a,b]
is that positive sum for auxiliary shell a and orbital shell pair b, the
uniform bound is max(E). The earlier N times largest interaction bound is
retained as an independent ceiling. Refining masks ensures every original
interaction appears once. Exact near, reciprocal and finite-zero terms have
no Gaussian approximation charge. Their arithmetic error remains separate.

For a forward input X, let x[b] be its L1 norm over the momenta and AO
components of shell pair b. Each output in auxiliary shell a has error at
most e[a] = sum_b E[a,b] x[b]. The maximum of e bounds the action component
error; sqrt(sum_a size(a) e[a]^2) bounds its Euclidean error. The adjoint
instead accumulates e[b] = sum_a E[a,b] ||c[a]||_1, with Euclidean bound
sqrt(sum_b nk size(b) e[b]^2). No cancellation or density-dependent partition
selection is used. Both action bounds are also capped by their independently
valid uniform counterparts. The RI energy/J ledger uses the forward Euclidean
bound with the same fixed retained metric. At Gamma, average the two raw-pass
bounds by the triangle inequality, including the transposed input/output.

The coupling magnitude envelope follows the same block reduction: positive
absolute moment/interaction envelopes propagate to leaves, and only images
of the same shell pair are summed. Exact near magnitudes join that block;
unrelated auxiliary outputs or shell pairs contribute through a maximum.
The reciprocal/zero residual retains its separately evaluated maximum. The
new ledgers change neither coupling values, partitions nor source identities.
One auxiliary shell is processed at a time using admitted per-node and
per-shell-pair scratch; no auxiliary-by-pair ledger is materialized. Positive
additions round outwards; the scalar guards are not an integral-roundoff proof.

At reciprocal Gamma, qualify the literal finite support under
`(mu,nu,R,T) -> (nu,mu,-R,T-R)` before completing its conjugate orbit.
For short-range coupling S, define
`S_H[k,P,i,j] = (S_raw[k,P,i,j] + conj(S_raw[k,P,j,i]))/2`.
Its forward action is half the raw action on X plus the conjugate raw action
on `conj(transpose(X))`; its adjoint is half the raw action on c plus the
conjugate transpose of the raw action on `conj(c)`. Thus both maps represent
the same linear coupling for arbitrary complex inputs. Since the exact
finite short-range source has this symmetry, convex averaging preserves
the same uniform approximation bound. The unchanged reciprocal/zero terms
are added afterwards and retain their own domain requirements. This does
not repair an unqualified exact source or grant a reduction certificate.

Each ordered pair must have its reverse integer image, and every included
auxiliary image must map to an included counterpart. Floating-point cutoff
ties that fail this literal test are refused. A transfer merely within the
older Gamma classification tolerance must also be a reciprocal representative
within roundoff. The coupling identity records this completion rule; the
exact reference coupling, metric, retained projector and J-only qualification
retain their existing owners. Fixed-rank variational checks and final Fock
Hermiticity gates remain in force.

Admission includes global geometry, the error model, local moment state,
Gamma orbit lookup and two sequential raw passes, a bounded shared partition
cache and a complete uncached antichain workspace. Entries borrow the common
tree. Source, dense export, action and RI parent limits apply before copies
and preparation, without discounting work for expected cache hits. No full
T/B or rectangular pair-image membership array is retained. Uniform envelopes
and positive magnitude sums remain deliberately conservative; useful-accuracy
profiles and broader production crossover are still open.



### Isolated erfc derivative and interior-moment bounds

The shared `erfc_gaussian_interaction` now tightens two independent steps in
the isolated erfc Taylor ledger. The finite Ewald-complement bound retains
its existing formula. Both reuse the native Taylor matrix, Gaussian extents
and positive upper-incomplete-gamma recurrence. This is a MANTA derivation for
the nonharmonic erfc kernel; the molecular harmonic remainder is not substituted.

For a directional derivative of order $n=p+1$, let $r=n-2k$, $m=n-k$,
$d=D-a-b>0$ and $s\ge\omega$. Instead of bounding the polynomial by
$(D+a+b)^r$ while bounding the Gaussian at $d$, keep their common radial
argument $z$:

$$
\sup_{z\ge d} z^r e^{-s^2z^2}
=\begin{cases}
 (r/(2e))^{r/2}s^{-r},&s<\sqrt{r/2}/d,\\
 d^r e^{-s^2d^2},&s\ge\sqrt{r/2}/d.
\end{cases}
$$

For $r=0$ only the second branch applies. Multiply by $s^{2m}$ and integrate
over $[\omega,\infty)$. The first branch is a positive integral of $s^n$
on a finite interval; the second is
$\Gamma(m+1/2,d^2\max(\omega,\sqrt{r/2}/d)^2)/(2d^{n+1})$.
The native upper-gamma recurrence bounds that tail. Taking the smaller of
this envelope and the earlier separate polynomial/decay envelope is valid
term by term. The compact remainder retains the factors
$2^{r+1}(a+b)^nQ_AQ_B/(\sqrt\pi k!r!)$ from the directional Taylor theorem.
Logarithms and `expm1` avoid premature overflow and cancellation at the branch
boundary. Existing geometric guards and positive-subnormal handling apply.

The Gaussian ledger can additionally replace the worst-case radius/mass
factor using existing full absolute radial moments through order $p$.
For either source, define

$$
U_A(j)=\min_{0\le k\le\min(j,p)} a^{j-k}f_A[k],\qquad
\mu_n=\sum_{j=0}^{n}\binom nj U_A(j)U_B(n-j).
$$

On the interior sphere, $|x|^j\le a^{j-k}|x|^k$, so $U_A(j)$ bounds its
absolute radial moment. The triangle and binomial inequalities then bound
the integrated $|x-y|^n$ by $\mu_n$. Thus the Gaussian compact remainder is
the conditional radius/mass remainder times
$\min(1,\mu_n/((a+b)^nQ_AQ_B))$, rounded upwards. No order-$p+1$ native
moment is required, even at $p=12$. Signed contractions use absolute producer
envelopes, and the full Gaussian tail-penetration and full-moment-substitution
terms remain unchanged. Gaussian tails never become compact support.

`LatticeGaussianError.compact_moment_scale` exposes this factor (one for the
unchanged lattice-complement ledger). Its `compact_remainder` is the tightened
Gaussian compact contribution; `LatticeGaussianInteraction.compact.total`
retains the separate conditional radius/mass bound. The reported total sums
the tightened compact term and both unchanged Gaussian-tail terms. These are
approximation bounds, not interval certificates for arithmetic roundoff.

Both additional radial evaluation and moment-cap work are admitted before
payload evaluation. The isolated compact and Gaussian error identities use
version two. All periodic GDF multipole consumers share this producer and
bind `ordered-erfc-multipoles.v2;joint-radial-and-interior-moment-bounds` in
their approximate coupling identity, because tighter acceptance can change
the fixed partition. Exact metric/reference identities, finite image support,
Bloch conventions and Gamma completion remain unchanged. Dense exports,
forward and adjoint actions still use the same source-qualified partition.
No density-dependent selection or new Fock/symmetry permission is introduced.


### Lossless global partition storage

The global pair-tree cache keeps the exact existing Taylor interaction, error
and node decision. With total order p and Hermite row degree d, only
(p-d+1)(p-d+2)(p-d+3)/6 columns can be nonzero. Packing retains every allowed
binary64 coefficient verbatim and checks that omitted positions are zero.
There are binomial(p+6,6) stored values rather than binomial(p+3,3)^2; at
order six this is 924 rather than 7056. No radial coefficient is recomputed
and no sparsity threshold is used. A single zero-filled dense buffer restores
the original matrix before the same forward, adjoint, magnitude-bound and
export contractions. Allowed signed zeros are retained.

The shared source owns the fixed tree and bounded partition cache. Entries
retain only their accepted antichain, not a full-tree-sized vector reservation.
The global action's existing complete-entry workspace permits constructing an
entry before cache eviction. Actual allocated coefficient/vector capacities
determine the retained charge. An entry above the cache cap is returned to the
current action without publication or duplicate construction; entries that fit
evict only enough FIFO storage for their actual charge. Immutable borrowed
entries and concurrent calls retain the existing separate workspace contract.
The legacy auxiliary-image route keeps its prior reserve-before-build policy.

Packing, expansion and validation work is explicitly admitted. The conservative
dense-antichain memory bound remains, covering compact construction and the
temporary metadata move; the existing single-interaction workspace covers
expansion. No expected cache hit reduces admission. Cache history/capacity
does not enter numerical source identity. This storage change grants no new
method, approximation, BIPOLE/CRYSTAL parity or symmetry-reduction permission.


### Componentwise fixed-metric periodic RI error transport

The shared `PeriodicGDFTiles.forward` and `adjoint` results retain
`approximation_bounds` in the same shape/order as their numerical output.
Each entry bounds that output's omission relative to the producer's fixed
finite reference. Exact sources return zero entries. Approximate sources also
return `reference_magnitude_bounds[P]`, an upper bound on the largest absolute
exact-reference coupling for auxiliary component P over all momenta and AO
pairs. Exact sources leave this second diagnostic empty. Both Python exports
are owned read-only copies; they neither change source identity nor retain an
auxiliary-by-pair tensor.

For the global pair tree, auxiliary-shell a and orbital-shell-pair b already
have omission E[a,b] and approximate-magnitude M[a,b] ledgers. Forward entries
retain sum_b E[a,b] ||X[b]||_1; adjoint entries retain
sum_a E[a,b] ||c[a]||_1. The short-range exact-reference column envelope is
max_b (M[a,b]+E[a,b]), shared by components of auxiliary shell a. At Gamma,
average the component envelopes along the same two raw passes and output
transpose as the numerical action. Average the auxiliary column maxima as
well; transposing the AO pairs preserves that maximum. Exact reciprocal/zero
tiles add their absolute component magnitudes to the short-range envelope.
Other routes use their existing uniform omission and direct/factored far
magnitude bounds. Independently valid scalar ceilings still cap the entries.

Let b~ be the approximate forward result, F the actual retained metric factor,
rho~=F^T b~, c~=conj(F) rho~, and e the componentwise forward omission vector.
Positive triangle propagation gives

```text
u_L = sum_P |F_PL| e_P
g_P = sum_L |F_PL| u_L
|delta rho_L| <= u_L,   |delta c_P| <= g_P
|delta E| <= min(sum_P |c~_P| e_P, sum_L |rho~_L| u_L)
             + 0.5 sum_L u_L^2
|delta J|_max <= adjoint_error(c~) + sum_P C_P g_P
```

Here C is the producer's exact-reference column envelope. These inequalities
follow directly from the paired quadratic RI expression described above and
the triangle inequality. They do not assume exact eigenvector orthogonality,
replace F with an ideal inverse, change rank/projector, refit a metric or use
cancellation. The previous scalar Euclidean/Frobenius bounds remain separate
valid bounds; the reported energy/J bounds take the smaller result. Positive
long-double accumulation and outward binary64 guards protect these envelopes,
not integral arithmetic. Rank zero and zero input retain exact zero bounds.

The action plan adds conservative storage for component/column vectors, Gamma
mates, residual workspace and binding exports. The RI parent jointly admits
the two retained action results and factor-propagation vector. Scalar/kernel
plans include positive propagation work; cold and warm admission is identical.
No extra source sweep, dense coupling, density-selected partition or Fock path
is introduced. The finite source, paired maps, shared Fock, exact K/XC and
source-qualified symmetry remain unchanged. These bounds concern fixed-density
coupling error only: basis/domain, arithmetic, metric solve and SCF relaxation
errors remain separate. Automatic whole-method accuracy is still unfinished.


### Fixed-envelope automatic periodic RI coupling accuracy

`PeriodicRIAccuracyOptions` selects a fixed Gamma Coulomb pair-tree source
before SCF. Both maximum-entry J and Coulomb-energy targets are positive,
with an explicit density envelope L. The envelope is
`sum_k w_k sum_ij |sum_spin D_spin,k,ij| <= L`, in the admitted AO basis;
it is not the electron count in a nonorthogonal basis. Each evaluated density,
including the initial and final confirmation densities, must pass this check.
No partition or tolerance changes during SCF.

For producer-owned exact-reference auxiliary column magnitudes C, evaluated
uniform coupling omission epsilon and the actual retained factor F, define

```text
r_Lmode = sum_P |F_P,Lmode| (C_P + epsilon) L
u_Lmode = sum_P |F_P,Lmode| epsilon L
c_P = sum_Lmode |F_P,Lmode| r_Lmode
g_P = sum_Lmode |F_P,Lmode| u_Lmode
J_bound = epsilon sum_P c_P + sum_P C_P g_P
energy_bound = 0.5 L J_bound
```

The first term bounds the approximate-adjoint omission at the approximate
coefficient; the second bounds exact-adjoint transport of the coefficient
error. The paired Hermitian Gamma quadratic expression gives the energy
inequality. These triangle inequalities extend the paired RI contractions
of Neese et al., DOI 10.1021/acs.jpca.4c07415, equations 29-36; the envelope
selection policy is an implementation choice. No ideal inverse, cancellation
or additional orthogonality assumption is used. Positive accumulation and
outward guards protect the envelope, not the underlying integral arithmetic.

A zero-density action obtains source-owned magnitudes and the evaluated
coupling omission without constructing a dense coupling tensor. If either
target fails, the shared producer creates a tighter immutable source from
the same basis, cell, domains and physical kernel. Metric/reference identities
must match, and the original metric factor remains unchanged. With
`r = min(J_target/J_bound, E_target/E_bound)`, a failed trial proposes
`epsilon_next = epsilon_requested * min(0.5, 2*r)`. This implementation
heuristic probes above the linear target estimate because the discrete
partition can have an evaluated omission far below its requested tolerance.
It reduces the request by at least half, but assumes neither linear scaling
nor monotone certificates. Only the evaluated bounds accept a trial.

Underflow clamps the proposal to the smallest positive double. If no smaller
positive request exists, selection reports unmet targets instead of passing
a zero tolerance to the shared producer. Zero finite-coupling omission does
not certify integral arithmetic. One through 32 attempts are admitted in
advance, including worst-case work and simultaneous old/new source storage.
Exhaustion returns an error. The heuristic need not find the loosest valid
partition or the fastest implementation.

The certificate records targets, density envelope, selected tolerance, both
bounds, attempts, setup work, source/reference and certificate identities.
Per-density component and scalar bounds remain available when tighter.
Automatic accuracy admits J-only fitting with independent K/XC. It gives no
new BIPOLE boundary convention, CRYSTAL parity or symmetry permission.
Only coupling error relative to the same finite metric/source is covered.
Basis/domain error, metric eigensolver error, arithmetic and density relaxation
remain separate; automatic whole-method accuracy is still unfinished.


### Canonical-overlap density envelope for automatic RI selection

Complete bounded HF/KS methods may request automatic coupling selection with
zero `density_l1_limit`. Raw RI builders still require a positive value.
The method derives L from the same immutable overlap, numerical projection,
rank cutoff and eigensolver as its bounded SCF leaf. The shared numerical
helper is independent of MANTA physical sources and adds no symmetry claim.

Let X be the actual retained canonical overlap transform at one k point.
For a spin channel with m occupied columns, the solver constructs
`D = X U_occ U_occ^H X^H`. Its reduced eigenvector audit tests every entry
of `U^H U` against identity with absolute tolerance a and relative tolerance r.
For r below one, Gershgorin and the diagonal sum give

```text
g_m = (1 + m a)/(1-r),  d = (1+a)/(1-r)
||U_occ||_2^2 <= g_m,   ||U_occ||_F^2 <= m d
A = (sum_i ||X_i,:||_2)^2
Q = ||X||_F^2,   H = min(Q, ||X||_1 ||X||_infinity)
||D||_entrywise-1 <= min(g_m A, n min(g_m Q, m d H))
```

The first branch uses row Cauchy; the second uses positivity,
`||D||_entrywise-1 <= n Tr(D)`, and the Frobenius/spectral product bounds.
A zero occupied channel contributes zero. Sum alpha and beta bounds, then
average over the uniform full k mesh. Restricted density uses the same two
equal occupied channels. These inequalities are an implementation derivation,
not a new fitted Hamiltonian. Positive accumulation/outward conversion protects
the envelope; the integral and solver arithmetic retain their own scope.

The helper audits overlap projection, negative eigenvalues, a common retained
rank and occupied-plus-virtual capacity using the existing solver routines.
Its count-only plan admits overlap traversal, one eigenanalysis per k point,
norm accumulation and owned per-k exports before reading values. Parent HF/KS
plans include the added setup work and storage. The returned bound is a fixed
source-selection input, independent of HCORE, later Fock matrices and density
history. Per-density enforcement remains mandatory, so an exceeded bound is
a failure, never a request to change partitions mid-SCF. Explicit positive
limits and all raw-builder, exchange, BIPOLE and symmetry guards remain intact.



### Componentwise density envelopes for automatic RI coupling selection

The shared canonical-overlap helper also returns a positive entry envelope
`H[k,i,j] = (sum_spin g_m) ||X[k,i,:]||_2 ||X[k,j,:]||_2 / nk`, with zero
contribution from an empty spin channel. Cauchy and the same audited occupied
Gram bound imply `|w[k] sum_spin D[k,i,j]| <= H[k,i,j]` for the uniform full
mesh. Keep the independently derived total L1 bound L as an additional
constraint. Neither bound selects a density, Hamiltonian or symmetry action.

An explicit positive L may also carry `density_entry_limits`, a finite
nonnegative row-major vector over the entire weighted spin-total AO-pair
input. The raw builder owns a snapshot and checks both constraints at every
call. A complete method with L=0 derives both bounds; an explicit entry
override with L=0 is rejected. The count-only method plan copies no entry
vector and admits its eventual ownership and up to two source probes per
selection attempt before physical setup. Native receipts and managed output
identify the additional constraint by a digest; the numerical helper exports
an owned read-only `(nk,n,n)` array for reproduction and explicit controls.

The existing coupling producer now returns `reference_action_bounds` for
approximate sources. In forward order it bounds `|T_exact|^T |input|`; in
adjoint order it bounds `|T_exact| |input|`. These are positive triangle
actions, valid for every complex input dominated entrywise by the supplied
magnitudes. Shared pair-tree shell-block ledgers include near magnitudes,
far moment envelopes and omission bounds. Gamma conjugate-orbit averaging
transports these envelopes with the same input/output permutations as the
paired map. Exact reciprocal/zero residual magnitudes are added separately.
Other coupling routes retain their own uniform omission/far envelopes.
Only one auxiliary shell's block ledger is live; no auxiliary-by-AO-pair
tensor is retained. Storage, scalar work and exported arrays are admitted.

For the fixed H, let the shared forward probe provide per-auxiliary error
e, exact-reference absolute action B, and exact column maxima C. Intersect
the bounds with `e <= epsilon L` and `B <= C L`. For the actual retained
metric factor F, define positive vectors

```text
R = |F|^T (B+e), U = |F|^T e
c = |F| R,       g = |F| U
J_bound = min(epsilon sum(c), adjoint_error(c)) + C.g
E_bound = min(L J_bound/2, R.U + U.U/2)
```

The additional adjoint probe supplies the first term through the same
partition and producer error ledger. It uses the fixed coefficient envelope
c, never a current SCF coefficient. The proof is the paired-map identity
`delta J = delta T c_tilde + T_exact delta c` and the quadratic RI energy
identity. Every trial is accepted only by evaluated bounds. Retained source,
metric, factor and reference identities still obey the original selector
contract. Explicit L without entry limits keeps the earlier scalar selector.

This refines finite-source coupling allocation only. Overlap truncation,
basis/domain, metric solve, arithmetic and density relaxation remain separate.
Independent exchange/XC, unified Fock, BIPOLE/CRYSTAL conventions and qualified
symmetry retain their existing owners and admission rules. Useful whole-method
accuracy and production performance still require their full acceptance.


### 5.17 Nuclear bounds for the physical midpoint-sphere source

The existing RI nuclear producer uses a sphere about each ordered AO-pair
midpoint, not the common real cube of Section 5.10. The new standalone
`bound_periodic_nuclear` certifies this exact finite convention against the
infinite 3D zero-average Ewald potential. It consumes the original Cartesian
AO and G supports, k order, nuclear positions, alpha and pair/nuclear cutoffs.
Its physical-source identity is generated by the existing nuclear producer's
shared identity routine. No integral, Fock state or symmetry permission changes.

The kernel and phase conventions follow Ewald (1921), IV.1, Eqs.37-40,
DOI `10.1002/andp.19213690304`, and Sun et al. (2017), Sec.II.A, Eq.16,
DOI `10.1063/1.4998644`. The finite-support enclosure below is a derived
extension; neither paper claims this particular bound. The shared native
Gaussian majorants absorb contraction signs and Cartesian/spherical
components before the nuclear estimate.

For a normalized envelope Gaussian of exponent a and charge Q, set
$\eta_*=(\alpha^{-2}+a^{-1})^{-1/2}$. Its real interaction with a unit
point at distance d is bounded by

$$Q\min\left[2\sqrt{a/\pi},
\frac{\exp(-\eta_*^2d^2)}{\sqrt\pi\,\eta_*d^2}\right].$$

The first expression applies at d=0. The second follows from the exact
Gaussian erfc convolution
$[\operatorname{erf}(\sqrt a d)-\operatorname{erf}(\eta_*d)]/d$
and the Gaussian erfc tail inequality. For two points use the second
expression with $\eta_*=\alpha$; the same-point self term is excluded,
and an unresolved distinct point collision refuses qualification.

For each ordered AO-pair arm and nucleus, recenter an integer completion
cube about the physical midpoint. Obtain retained image labels from the
shared `PairNuclearImageSelector`, including its boundary tolerance. Sum
the above bound only over cube holes. Add the shared logarithmic
Gaussian-point exterior-shell bound outside the cube. Rounded shifted
centers carry an outward positional uncertainty. The exterior may include
retained physical images; this positive overcount is safe. An exterior
that cannot be separated from the majorant centers refuses. Completing
the opposite arm independently avoids assuming that a rounded physical
sphere authorizes a new symmetry relation.

The reciprocal omission is bounded by

$$C_G=\sum_{G\notin\mathcal G,\,G\ne0}
\frac{4\pi}{\Omega|G|^2}\exp[-|G|^2/(4\alpha^2)].$$

The shared reciprocal Gaussian producer evaluates missing labels inside
an independent completion cube and bounds its exterior. Physical supports
may be noncubic, have interior holes, or extend beyond that cube. Their
labels must be unique, inversion closed and agree with the Cartesian
vectors to roundoff. A retained pair's reciprocal bound is
$Q\sum_A|Z_A|C_G$. Summed ordered shell maxima give real and reciprocal
AO matrices uniform in k.

The omitted-AO-image contribution is
$B^{\rm image}_{ij}=C_{\rm point}\sum_A|Z_A|M_{ij}$, reusing the existing
Gaussian image mass M and the periodized Gaussian-point kernel bound from
Section 5.10. That shared kernel includes real, reciprocal and gauge terms.
The retained source's constant $+\pi\sum_AZ_A S/(\alpha^2\Omega)$ has no
additional nuclear-domain error; its missing AO images are already in M.
The total matrix bound is the positive sum of all three components.

For nuclear repulsion, the same selector, recentered holes and point tail
apply to each ordered nuclear pair with weight $|Z_A Z_B|/2$. The
reciprocal error is bounded by $(\sum_A|Z_A|)^2C_G/2$. Self removal and
the uniform-background constant remain exact shared terms, with no second
gauge contribution. All positive sums and products use outward logarithmic
arithmetic and retain a positive binary64 subnormal floor.

Count-only admission covers pair-source reconstruction, Gaussian majorants,
selector labels, image and reciprocal completions, both ordered arms,
nuclear pairs and owned matrix exports under all five resource caps.
Immutable receipts distinguish physical source, pair source, image envelope
and certificate identities. Options are returned by value; arrays are owned
and read-only on export.

Complete RI RHF/UHF and pure/global/range-separated KS can impose separate
positive matrix, attraction-energy and nuclear-repulsion targets before SCF.
`PeriodicNuclearAccuracyOptions` inherits the proof cubes and leaves physical
supports fixed. The method reuses its single native canonical overlap-space
analysis for all enabled RI, S/T, exchange and nuclear certificates. Let
$H_{k,ij}$ bound the weighted spin-summed density entries in that space,
let $L_1$ be its weighted entrywise norm bound, and let $B_{ij}$ be the
nuclear matrix enclosure. The one-body contraction has no half factor:

$$
|\Delta E_{\rm en}| \le \min\left(
  \sum_{kij} H_{k,ji} B_{ij},\ L_1\max_{ij}B_{ij}\right).
$$

The shared positive trace producer evaluates both bounds; their minimum
and the outward sum with the nuclear-repulsion bound are recorded separately.
The envelope is not an evaluated SCF density. All three targets must pass
before the Fock provider starts. Native count-only admission includes the
envelope, contraction scratch, nested producer, retained receipts and owned
exports under all five resource caps before array copies. The certificate's
physical nuclear identity must equal the method's nuclear receipt. This
adapter also supports exact-J/RI-K because nuclear bounds do not depend on
the exchange kernel. Its scalar provenance retains targets and exclusions.

The optional selector below can enlarge physical supports. Whole-method
error remains open. Floating
integral arithmetic, basis and k convergence, finite fitting/metric errors,
overlap-rank stability and state relaxation remain excluded. Unified Fock,
BIPOLE/CRYSTAL conventions and source-qualified symmetry continue to use
their existing physical producers.


### 5.18 Automatic physical support selection before RI SCF

`select_periodic_supports` is a bounded source selector over the existing
physical AO-pair sphere, nuclear midpoint sphere and reciprocal kinetic
cutoff. `PeriodicSupportRequest` carries S/T and nuclear component targets,
with optional full-Coulomb exchange targets and the method's exchange scale.
It is independent of an HF or KS solver. The public RI job adapter supplies
the correct restricted/unrestricted and hybrid coefficient. Range-separated
and fitted exchange retain their existing exchange-certificate refusals.

Every trial uses the shared cutoff-domain generator, one canonical overlap
density envelope and the shared S/T, nuclear and optional exchange bound
producers. A trial may return unqualified component values; ordinary complete
method admission still throws on a failed target. Failed S/T grows the pair
radius. The largest normalized nuclear matrix/energy component selects pair,
real or reciprocal growth, while nuclear repulsion compares its real and
reciprocal parts. Exchange compares its image/correction and reciprocal
components. All selected radii grow by `growth_factor`; reciprocal kinetic
cutoff grows by its square. This deterministic policy does not promise a
minimal or optimal support. Attempt, physical support, completion-cube and
resource limits are explicit; exhaustion cannot start SCF.

Proof cubes are enlarged to enclose conservative physical support boxes and
never shrink the requested proof radii. Physical cutoffs change the operator;
proof cubes only change its enclosure. The selected fitting pair cutoff
follows the physical AO-pair cutoff. Auxiliary cutoff, fitting omega, metric
projector, coupling-accuracy policy, Ewald alpha, BvK convention and XC grid
remain their independent controls. Existing finite-source conjugacy and
metric failures propagate; the selector never projects away such a failure
or silently adjusts the auxiliary source.

Count-only admission precedes domain arrays. The five joint resource caps
include every discarded trial, the selected trial and retained scalar
history. Initial count-only preparation is conservatively charged by its
whole scalar/kernel trial plan. Trial counters are admitted upper bounds,
not measured execution counts. The public job deducts these charges before
admitting the final method, and keeps the history live beside its output.
Dry runs reserve the declared complete-job memory cap and defer numerical
selection. No domain or operator changes during SCF.

The final complete method recomputes its normal component certificates;
overlap and nuclear source identities and S/T, nuclear and exchange
certificate identities must equal the selected trial. This guarded adapter
currently repeats final source preparation. It does not yet transfer a
prepared-source cache from the selector. Scalar history, component values,
growth decisions, targets, limits and identities are exported in provenance.
Unified Fock consumes the same final source, and no new symmetry permission
or BIPOLE/CRYSTAL convention follows from successful selection.

Qualification covers only requested component targets in the final finite
canonical density family. It excludes finite RI fitting/metric error,
integral arithmetic, canonical-rank stability, state relaxation, basis/k
convergence and XC quadrature. Existing automatic RI coupling selection is
separate and still executes at the final physical source. A complete
automatic RI-BUPO total-error profile and the remaining capability matrix
remain unfinished.


### 5.19 Nested periodic RI metric-factor error

`PeriodicRIAccuracyOptions.metric_reference_cutoff` optionally extends the
Gamma J-only pair-tree certificate to a finer absolute positive-mode cutoff.
The reference cutoff must be finite, nonnegative and no greater than the
resolved selected cutoff. `None` preserves the existing coupling-only
contract, numerical operator and certificate. Both cutoffs act on the same
lower-triangle canonical metric and the same stored eigensystem. Negative
and zero modes remain excluded; this never replaces a failed metric audit.

Sun et al., DOI [10.1063/1.4998644](https://doi.org/10.1063/1.4998644),
Section II.B, Eqs. 20-23 motivates the positive-mode fitting space. The paired
RI-J map follows Neese, DOI
[10.1021/acs.jpca.4c07415](https://doi.org/10.1021/acs.jpca.4c07415),
Section 2.1, Eqs. 9-11. We reuse the existing finite GDF source and its metric;
we do not introduce Sun's mixed-basis transformation or Neese's molecular
boundary condition. The following inequalities are derived here for the
stored factors and the already qualified periodic coupling envelopes.

Let F contain the selected columns and H contain the additional positive
columns admitted by the reference cutoff. The reference factor is their
concatenation, so the paired inverse action differs by `conj(H) H^T` exactly
as an algebraic factor identity. It does not assume numerically exact
orthogonality or `F^H M F = I`. The existing shared probe supplies C, an
exact-reference coupling column maximum, and R, a bound on the forward
action for every admitted weighted spin-total density X. Use `R=C L` or
the tighter componentwise intersection with the positive source action.

```text
z = |H|^T R
metric_J_bound = C^T |H| z
metric_E_bound = min(z^T z / 2, L metric_J_bound / 2)
combined_J_bound = coupling_J_bound + metric_J_bound
combined_E_bound = coupling_E_bound + metric_E_bound
```

The nested exact-reference fitted energy increases by
`||H^T T_exact^H X||^2/2`, regardless of density definiteness. The selected
approximate coupling additionally contributes the existing two-sided
coupling error. Triangle addition yields the combined bounds; the same
fixed density family is enforced at every build. Positive accumulation and
outward guards enclose the scalar bound arithmetic, including subnormals.
An empty removed space or zero-entry density family has exactly zero metric
contribution. No evaluated density is used to select or change a metric.

Target admission now uses the combined bounds when the reference is set.
The coupling selector may refine its existing multipole partition within
the admitted attempt limit, but the selected metric rank remains explicit.
It refuses unmet targets before SCF. The receipt keeps both components,
selected/reference ranks, absolute reference cutoff, discarded-factor and
reference identities. The raw builder exports an owned discarded-factor
copy for independent inspection. The optional storage and scalar work are
charged before allocation; no extra coupling traversal or full T/B tensor
is introduced. Parent RI HF/KS plans inherit these reservations.

`approximation_j_bound` and `approximation_energy_bound` on an evaluated
Fock receipt remain coupling-only. Combined family bounds are the accuracy
receipt's `j_bound` and `energy_bound`; output explicitly identifies the
additional metric-truncation component and its reference. Physical sources,
unified Fock ownership, BIPOLE/CRYSTAL conventions and symmetry permissions
are unchanged. This is not the error against an unfitted Coulomb operator.
Finite fitting-basis and image/reciprocal-domain errors, metric eigensolver
error, floating integral/contraction arithmetic, rank stability and state
relaxation remain excluded. The full automatic RI-BUPO goal remains open.


### 5.20 Full-Coulomb fitting enclosure from shared response maps

`PeriodicRIFittingBoundSource` compares a fixed `PeriodicRIQBuilder` with
an existing `PeriodicBupoSource` that supplies both infinite-Ewald-domain
and omitted-AO-image bounds. Both must have identical orbital basis, cell
and literal momentum frame; RI must be literal Gamma Coulomb. The reference
owns immutable weights. This compares with the full zero-average Coulomb J
on that orbital basis and mesh. It does not alter either physical source,
metric factor, Fock owner, BIPOLE/CRYSTAL convention or symmetry permission.

Sun et al., [10.1063/1.4998644](https://doi.org/10.1063/1.4998644), II.B,
Eqs. 20-23 and Neese, [10.1021/acs.jpca.4c07415](https://doi.org/10.1021/acs.jpca.4c07415),
Section 2.1, Eqs. 9-11 define the fitted source interpretation used above.
The new comparison below is a derived linear-map enclosure. In particular,
independently truncated periodic metric and coupling domains need not form
an exact Coulomb Hilbert projection. No variational residual, positive
Schur complement or monotonic fitting-energy assumption is used.

For weighted spin-total `X(k)=w(k) sum_spin D(k)`, probe a real Hermitian
basis `Eii`, `(Eij+Eji)/2`, and `i(Eij-Eji)/2` at every nonzero-weight k.
Call the existing RI and BUPO builders on scaled probes and divide out the
scale. For each output entry let a and b bound the magnitude of the finite
RI/BUPO difference and the BUPO remainder respectively. Diagonal inputs use
these directly; off-diagonal inputs use `hypot(a_real,a_imag)` and the
corresponding `hypot(b_real,b_imag)`. For `Xij=u+iv`, Cauchy bounds each
paired contribution by `2 |Xij| hypot(a_real,a_imag)`, and similarly for b.
Thus each row is bounded by its maximum coefficient times the fixed L1
limit, or by its positive contraction with the optional fixed entry limits.
Keep the tighter bound separately for the two terms, then add them.

This spans arbitrary complex Hermitian spin-total densities, including
indefinite inputs and broken time reversal. Zero-weight input rows and
zero-entry subspaces are skipped; all output momenta remain included.
The energy bound is the smaller of half the L1 limit times the maximum
J bound and half the fixed entry-envelope contraction with the per-entry
J bounds. This bounds the Coulomb energy at the same density, not the
energy difference between separately converged states.

The certificate records the selected RI, full reference, pair source and
family identities, finite discrepancy, reference remainder and combined
bounds. Its `build` validates fixed weights and density limits and delegates
the unchanged J build to the retained RI owner. Optional J/energy targets
refuse the certificate before it can be consumed. Probe scales also respect
any existing RI accuracy family; unrepresentable scales refuse explicitly.

Count-only admission covers at most `n_k n_AO^2` complete probe pairs,
both retained producers, scratch, owned exports and one consumer build.
The explicit probe cap and all five resource caps apply before allocation.
There is no stored AO-four-index response, but the work is expensive and
is not a linear-scaling fitting algorithm. This is a bounded qualification
path; automatic auxiliary/domain selection and complete-job attachment
remain to be implemented. It encloses the chosen stored RI operator's
fitting, rank, coupling and finite-domain discrepancy together. Do not add
those component bounds again. Floating integral/contraction arithmetic,
rank stability and density relaxation are excluded. No total SCF energy
accuracy or completion of the full MANTA matrix follows.


### 5.21 Shared full-Coulomb source preparation

`PreparedPeriodicBupoSource` separates electronic reference construction
from the existing `PreparedPeriodicBupoHF` method consumer. Both use one
sequence of the original pair, bubble tree, Ewald and BUPO producers.
No integral formula, extent or domain estimate, multipole partition or
Coulomb convention changes. The source factory owns no nuclear, exchange,
occupation or SCF policy. Charge/spin restrictions remain method-owned.

The generator and each producer are admitted cumulatively before that
stage allocates. The source plan preserves the existing conservative sum
of stage peaks. Its remaining-budget copy includes all retained source
reservations and subtracts preparation work under all five resource caps.
RI certification separately admits its probe and consumer cost; passing
that remaining budget deliberately overcounts some shared parent peaks
rather than omitting them. The generated uniform weights and literal
native Cartesian mesh preserve the source identity used by complete jobs.

The returned immutable source retains its original producers, including
both full Ewald-domain and omitted AO-image bounds. Consumers retain the
same source directly, without a second reference implementation. Existing
HF preparation and final numerical/source identities are unchanged; only
the generic preparation wrapper has its own identity. This is staged
construction, not a numerical-free dry run or a new accuracy proof.
Automatic complete RI admission, useful source selection and the full
remaining method/error matrix stay required.


### 5.22 Complete RI full-Coulomb fitting admission

Optional `PeriodicRIFittingAccuracyOptions` supplies one immutable prepared
BUPO reference and fixed J/energy targets. The RHF/UHF and pure/global/range
KS planners jointly reserve reference construction, count-only maximum
Hermitian probes, canonical density analysis and every later family check.
The certificate count planner depends only on admitted RI/reference plans;
it does not run probes or assert target success during dry-run planning.

After overlap construction, the method derives the same canonical fixed
occupation envelope already used by physical component accuracy. Its
weighted entry bounds and L1 limit are passed to the Section 5.20 proof.
The existing `PeriodicRIFockBuilder` shares its actual Gamma owner with
that certificate; no nominally equal metric, coupling or momentum frame
is reconstructed. Positive finite targets must pass before the solver
is entered. Every later build validates the fixed family, then uses the
unchanged numerical RI J/K contraction. K, XC, nuclear and symmetry
ownership and BIPOLE/CRYSTAL conventions are unaffected.

The reference preparation and probe plan are separate admitted-work
records; evaluated physical Fock counters keep their original scope.
Parent reservations include the full certificate/reference storage even
when the solver does not own those objects. Cumulative preparation work
and all five caps are enforced before certificate construction. Existing
coupling/metric certificates retain their own family provenance; the new
full-Coulomb proof is not added to them as another independent error term.

This reuses the Sections 5.20-5.21 producers and proof without extending
their arithmetic, rank-stability or relaxation scope. It is automatic
pre-SCF admission for explicit reference domains, not useful automatic
auxiliary/reference selection or a total SCF-energy accuracy guarantee.


### 5.23 Bounded full-Coulomb metric selection

An optional metric search retains the Section 5.22 canonical density family,
auxiliary basis, fitting domains, coupling producer and prepared BUPO
reference. Start with the requested RI absolute/relative cutoff. When the
combined full-Coulomb J or same-density energy target fails, reduce the
resolved cutoff by a specified factor in (0,1), clamped at a nonnegative
absolute floor. Subsequent candidates use this absolute cutoff and zero
relative cutoff. At most 31 refinements follow the initial candidate.
The floor and attempt count can refuse a request; neither constitutes an
accuracy certificate. Negative-mode validation is unchanged.

Each candidate constructs a positive metric factor through the original
RI producer, sharing the same coupling source. It is compared against the
same full-Coulomb reference using all Hermitian probes. The first candidate
meeting both targets is retained directly as the actual Gamma Fock owner;
its certificate and source identities describe that owner. Rejected
certificates cannot build a density. No SCF iteration or relaxed state is
used to choose the metric. Source identities are refreshed after a factor
change, and no new symmetry or Fock composition authority is introduced.

This uses the existing Sun2017 Section II.B fitting representation and
Section 5.20 triangle enclosure. It does not assume that error decreases
monotonically as rank increases, or that independently truncated fitting
and reference sources form an exact variational projection. Every trial
must pass the complete combined bound. All allowed probes, factorization
work and rejected trials are admitted in advance under the five resource
caps. Replacement permits two concurrent RI owners; trial receipts and
serialized copies have explicit reservations. Plans remain conservative
when the search stops early. The fixed-cutoff default is unchanged.

The initial search supports fitted Coulomb with independent exchange and
without a separate coupling/metric component-accuracy policy. Those
combinations refuse during count-only planning rather than silently
changing a second contract. The combined certificate already includes
coupling error. The existing fixed-cutoff policies remain available.
Auxiliary basis and domain selection, automatic reference preparation,
arithmetic and rank stability, relaxation and the full method matrix remain
required. This selection does not imply total SCF-energy accuracy.


### 5.24 Automatic shared BUPO reference selection

An optional `PeriodicRIFittingReferenceOptions` replaces the explicit prepared
reference in Section 5.22. It constructs the original Section 5.21 electronic
source inside the actual Gamma RI owner, using the method's orbital basis,
cell, native full mesh and canonical density family. An unset initial pair
cutoff inherits the selected coupling source cutoff. No integral formula,
Fock composition, BIPOLE/CRYSTAL convention or symmetry authority changes.

Every candidate is probed against the same selected RI map using Section
5.20. A complete J/energy pass accepts immediately. Otherwise, reference
selection continues until its J remainder is at most `reference_fraction`
of the J target and its Section 5.28 entrywise energy enclosure is at
most the same fraction of the energy target. This conservative trace
bound only guides selection; it is not another additive certificate.
Optional Section 5.23 metric selection then uses that immutable reference.
The final combined J and same-density energy bounds must both pass before
SCF. No monotonic improvement or variational fitting identity is assumed.

After a rejected reference, pair, real and reciprocal cube radii increase
by one up to `maximum_cube_radius`; the pair cutoff increases by
`growth_factor` (with a next-cutoff minimum of one bohr), capped at
`maximum_pair_cutoff_bohr`. The multipole tolerance decreases by ten.
The policy permits at most 31 refinements. Failure at a resource, attempt
or domain limit refuses the calculation; a limit does not assert accuracy.

The eventual tree depends on numerically admitted pair-source values.
The count-only complete plan therefore reserves a declared search budget,
not an asserted prediction of the eventual reference cost. Its memory and
four work caps are included in the parent plan. Each shared preparation
stage, probe and metric replacement is subsequently admitted against the
remaining reservation. Rejected work is never refunded. Rejected numerical
sources are released before preparing the next reference; receipts remain.
The reservation includes retained RI/reference owners, bounded receipt and
family storage, and both RI owners during refactorization. Output reports
declared caps and cumulative admitted costs, rather than presenting the
caps as measured work. Dry runs construct no reference and claim no target
success. The physical Fock counters retain their original scope.

The Sun2017 Section II.B representation, Neese2025 Section 2.1 motivation
and original periodic Ewald/image proofs remain those of Sections 5.20-5.23.
Automatic reference selection also supports fixed-cutoff fitted K and the
separate component certificate; metric refinement keeps its existing
J-only restrictions. Auxiliary and physical fitting-domain selection,
arithmetic/eigensolver/rank and relaxation ledgers, and all other unqualified
MANTA capabilities remain required. This is a fixed-density Coulomb bound.


### 5.25 Bounded physical fitting-domain selection

`PeriodicRIFittingDomainOptions` adds a bounded search over the actual Gamma
GDF coupling and metric source. The orbital and auxiliary bases, cell, AO
frame, mesh, weights, physical Coulomb kernel, split parameter and coupling
approximation controls stay fixed. Complete RHF/UHF and pure, global-hybrid
and range-separated KS admit this policy for fitted J with independent
exchange. Fitted K and a separate component accuracy policy refuse domain
selection, because replacing their shared producer would change another
contract. The existing input pair-cutoff agreement is preserved; only the
certified J owner receives the selected supports. Nuclear, S/T, exchange,
XC, unified Fock composition and symmetry authority do not change.

The shared `PeriodicGDFTiles.with_domains` factory preserves every original
numerical reciprocal vector in its original order and appends missing
nonzero vectors from a complete integer cube in lexicographic order. It
requires a reversal-closed Gamma source and nested pair/auxiliary image
radii. Its count plan admits the union upper bound, label/vector generation,
producer snapshots and work before allocating the enlarged vectors. The
original GDF metric and coupling implementations evaluate every candidate;
there is no second integral formula or imported metric. Basis-space fitting
error remains part of the existing Section 5.20 enclosure.

The default policy permits three refinements, multiplying the pair and
auxiliary image radii by 1.5 with a next-radius minimum of one bohr and caps
of 32 bohr. The reciprocal cube increases by one, capped at eight. Optional
initial radii must contain the original source; the initial cube defaults
to zero. These are bounded search heuristics, not a published error
estimator or a monotonic-convergence assumption. The Sun2017 Section II.B
representation and Neese2025 Section 2.1 factorization motivation remain
those cited in Sections 5.20-5.24. A valid initial finite metric is required;
negative-mode refusal is never replaced by damping or a repaired metric.

The reference is prepared/selected once by Section 5.24 or supplied
explicitly. Every physical domain begins with the originally requested
metric controls and may run Section 5.23 metric refinement. Both combined
full-Coulomb bounds must pass on the same canonical density family. The
first passing numerical owner and certificate are retained directly for
SCF. All failed trials remain in provenance. An attempt, physical-domain,
metric or resource limit refuses before SCF. Section 5.27 adds optional
selection among auxiliary basis candidates.

Complete count-only plans reserve the largest permitted fitting source for
all solver/Fock calls, before values or density analysis. Each actual
candidate is checked against that capacity. The search separately reserves
a declared memory/four-work cap. With automatic reference selection its
effective caps are the componentwise minimum of the two requested caps,
with the larger backend margin. Preparation, generation, metric factors,
probes, rejected work, bounded history and simultaneous original/current/
replacement owners are charged cumulatively. No work is refunded. The
explicit reference's preparation is charged once. Search caps are upper
reservations, not measured operation counts or timing predictions.

A zero-refinement policy with the selected radii/cube and metric reproduces
the accepted source without changing complementary physical inputs. Output
separates requested supports, selected supports, maximum SCF capacity,
actual selected-source plan and all trial source/certificate identities.
The certificate still excludes contraction arithmetic, eigensolver/rank
stability and density relaxation. It is not a whole-method energy bound.


### 5.26 Shared primitive-product auxiliary candidates

`vibeqc.auxiliary_generation` provides a shared, bounded basis metadata
producer. `plan_product_auxiliary_basis` counts resources and
`prepare_product_auxiliary_basis` returns an owned basis, plan and receipt.
The construction follows the same-center primitive-product exponent sums
and even-tempered motivation in Stoychev, Auer and Neese (2017),
DOI [10.1021/acs.jctc.6b01041](https://doi.org/10.1021/acs.jctc.6b01041),
Section 2, Equations 5-7. The implementation is independently written.
It does not implement the paper's effective-exponent, valence/ECP or
angular-pruning heuristics and must not be identified as ORCA AutoAux.

For each atom and product angular degree L = l1 + l2, all same-atom orbital
shell pairs contribute the minimum and maximum primitive exponent sums.
Coefficients do not prune this envelope. The geometric grid starts at the
minimum and extends through the first beta power covering the maximum;
rounding at the endpoint may require one further interval. Beta defaults
to two. Level zero keeps the coarse endpoints; levels one through eight
successively insert dyadic midpoints in log-exponent space. Every coarser
exponent remains bitwise present. Generated shells are Cartesian: their
homogeneous polynomial space contains products of either pure or Cartesian
orbital angular functions. This angular statement does not assert exact
radial fitting of every off-grid exponent or cross-center product.

An optional seed retains every shell, coefficient and order as an exact
prefix. Only exact same-center, same-degree, single-primitive seed matches
with nonzero coefficient suppress generated duplicates. Pure s/p shells
span the corresponding Cartesian space; pure d and higher shells do not
replace Cartesian trace components. Generated primitives use the existing
libint normalization convention. Unrepresentable normalization, nonfinite
ranges, indistinguishable exponents, mismatched centers or an angular cap
that would omit a product channel refuse. No metric repair is introduced.

Planning admits shell/function counts, modeled peak bytes and work before
candidate construction. Defaults are 4096 shells, 32768 functions, 256 MiB
and one billion modeled work units. Counts conservatively include possible
seed duplicates, input snapshots, output constructors, normalization,
identity/duplicate scans and receipt exports. The orbital angular limit
is 16 and generated limit is configurable up to 32, default eight. These
are metadata capabilities; each integral consumer retains its own limits.
The receipt records input/output content identities, exponent spans,
controls, counts and an explicit candidate-only qualification. Source
matching ignores display names and retains the numerical basis frame.

Selected periodic RI jobs accept the prepared object directly as
`CoulombBuildOptions.auxiliary_basis`. They verify its orbital identity,
reserve its complete peak/work in addition to existing job costs, pass
its owned `BasisSet` to the existing GDF producer, and route generation
provenance/citations through standard output and QVF. The plain `.basis`
remains available to other basis consumers without this job-level receipt.
No integral formula, metric authority, unified Fock composition,
BIPOLE/CRYSTAL convention or symmetry permission changes.

Generation alone has no fitting or energy accuracy certificate. Optional
Sections 5.20-5.25 certification still evaluates the actual numerical source
on its fixed density family, including finite metric and fitting effects.
A larger auxiliary basis need not improve that certificate monotonically.
Section 5.27 adds bounded selection among these basis candidates.
Arithmetic, eigensolver/rank, relaxation and whole-method ledgers remain unfinished.


### 5.27 Automatic auxiliary selection under the fitting certificate

`PeriodicRIFittingAccuracyOptions.auxiliary_selection` enables a bounded
search over the original seed, then the shared Section 5.26 candidates in
increasing refinement order. `PeriodicRIFittingAuxiliaryOptions` captures
the initial generator controls, generation limits, maximum refinements and
search budget. The default is three generated levels starting at zero;
the largest requested level must not exceed eight. This order is a search
heuristic, not a monotonic error or least-cost assertion.

The shared `ProductAuxiliarySequence` plans all candidates before building
any basis metadata and owns the resulting immutable candidates. Its plan
reserves all retained candidates, construction scratch and temporary atom
metadata together, with cumulative generation work. Complete periodic jobs
prepare it automatically. Direct native complete-method callers first use
`prepare_periodic_ri_auxiliary_selection`; count plans require a captured
sequence matching the numerical orbital and seed identities. Preparation
cannot certify an integral error or guarantee that later source admission
will pass. Metadata construction still requires the generator's matching
atom-centered input frame; this does not broaden AO gauge support.

The shared GDF `plan_auxiliary_domains` and `with_auxiliary_domains` methods
retain the original reciprocal vector order and union convention while
changing the admitted auxiliary basis. Existing integral formulas and
finite metric positivity/rank authority are reused. A separate replacement
guard checks the generated candidate and unchanged orbital identity; the
existing physical-domain replacement guard still forbids an auxiliary
identity change. No metric repair or alternate Fock composition is added.

One supplied or automatically selected full-Coulomb reference and one
canonical density family apply to every basis trial. Each basis starts at
the original requested physical fitting domains and metric controls.
Optional domain expansion and metric refinement run inside that basis
trial. Without domain selection, fitting supports remain fixed. The first
actual source passing both J and same-density Coulomb energy targets is
retained unchanged for SCF with its certificate. Exhausted candidates,
metric/domain limits or resource caps refuse before SCF.

Planning takes componentwise maximum source costs over every candidate and
the allowed domains before integral values or density analysis. Shared
reference, domain and auxiliary searches use the componentwise minimum of
their requested resource caps and the largest backend margin. Generation,
source construction, factors, probes, rejected work and bounded history
remain charged. Concurrent original, replacement and reference owners are
reserved; no rejected work is refunded. A dry run is resource admission,
not evidence that an accuracy target passes.

Provenance separates requested seed and selected auxiliary counts, maximum
SCF capacity, selected-source plan, candidate/generation/source identities,
all domain/metric trials, and requested/effective/charged search budgets.
The selected generator receipt and citation use standard output and QVF.
A fixed selected basis, reference, domains and metric reproduces the
accepted owner. The policy requires fitted J, independent exchange and no
separate component accuracy policy. Unified Fock, BIPOLE/CRYSTAL source
conventions and existing symmetry permissions retain their authority.

This extends the finite fitting/metric certificate, not its mathematical
scope. Arithmetic, eigensolver and rank stability, density relaxation,
other energy components and the remaining capability matrix are still
outside this acceptance. Conservative fixture targets are not recommended
production tolerances or a whole-method energy guarantee.


### 5.28 Separate fitting and reference energy enclosures

The fixed-family certificate also reports `finite_difference_energy_bound`
and `reference_remainder_energy_bound`. Let a_x and b_x be the existing
entrywise bounds for the finite RI-minus-BUPO map and the BUPO-to-full-Coulomb
remainder, respectively. They already include the same Hermitian probes,
weighted density entry caps and L1 cap used for the combined certificate.
For either vector v, the same-density trace inequality gives

$$
 B_E(v)=\min\left(\tfrac12 L_1\max_x v_x,
                  \tfrac12\sum_x\min(H_x,H_{x^T})v_x\right).
$$

Only positive-weight k entries enter the sum. Without entry caps, use the
first expression alone. The shared outward positive arithmetic preserves
zero and guards compound sums. The two new receipts report B_E(a) and
B_E(b). The combined energy certificate still uses B_E(a+b). Taking a minimum
between two upper estimates does not make this formula linear or imply
an ordering relative to B_E(a)+B_E(b). The component bounds are separate
enclosures, not an additive measured-error decomposition. Integral values, density families and final target
checks remain unchanged. Count admission includes the extra contraction
work; no response tensor or additional source owner is retained.

Automatic reference readiness now compares B_E(b) with the requested
reference fraction of the energy target, alongside the existing reference
J criterion. This replaces the former looser scalar expression L1*max(b)/2 used for
Section 5.24 readiness. A reference can therefore be retained when its energy envelope
is adequate even if that scalar estimate is too large. Passing this stage
only permits the fitting/metric/domain search to continue. The actual
selected source must still pass both combined full-Coulomb targets before
SCF; reference fractions do not weaken or replace those targets.

The finite/reference energy bounds appear with the final certificate and
trial provenance. The underlying full-Coulomb fitting proof, shared producer
identities, unified Fock, BIPOLE/CRYSTAL convention and symmetry permissions
are unchanged. These bounds still exclude arithmetic, rank stability and
density relaxation, and do not certify the full SCF energy.


### 5.29 Bounded shared reference actions

`PeriodicEwaldSource` and `PeriodicBupoSource` expose `plan_batch(count,
 budget)` and `build_batch(densities, budget)` for one to 32 independent
complex Hermitian densities with shape `(batch, momenta, AO, AO)`. Admission
precedes input copies and numerical work. Memory covers simultaneously live
inputs, responses and exports, coefficient panels, compensated sums,
reciprocal panels, bounds and receipts. Work retains conservative cold scalar
reservations plus batch setup; rejected search work is never refunded.

The Ewald producer streams each real tile, Fourier panel and overlap panel
once across a batch. Each density retains the scalar reduction order,
charged electronic gauge and source identity. Direct BUPO partitions also
share each Gaussian majorant-pair evaluation, then apply the same positive
bound arithmetic and pair-image envelopes independently. No four-index,
all-probe response tensor or quadratic pair-bound cache is retained.
Approximate BUPO partitions explicitly use the scalar fallback for each
density; this increment does not accelerate their multipole actions.

Batch receipts count actual source passes, kernel evaluations and domain
majorant evaluations. Per-density receipts retain logical scalar counts
for comparison; summing those counts would double-count shared evaluation
work. Physical values, energies, density identities and bound components
have the same meaning as scalar results. Returned arrays own their copies.

Fitting options use `probe_batch_size=8`, admitted between one and 32.
One selects the unchanged scalar reference path. The fitting consumer holds
only a bounded batch and at most one pending real-probe envelope when a
batch boundary splits a real/imaginary pair. The original `hypot`, positive
sum and certificate identity order are preserved, including partial batches,
zero entry caps and zero k weights. Final and trial provenance reports
`reference_batches`, `reference_source_passes`, `reference_shell_calls` and
`reference_majorant_pairs`.

This reorganizes the existing Ewald1921 and Sun2017 producer evaluations;
it changes no integral formula, fitting proof, error target, source family,
unified Fock composition, BIPOLE/CRYSTAL convention or symmetry permission.
The stored-integral and fixed-density scope of Sections 5.20 through 5.28
remains in force, including the arithmetic and relaxation exclusions.


### 5.30 Reference contributions and directed refinement

The existing shared BUPO Gaussian majorant producer already separates real
and reciprocal lattice remainders. Its scalar and batch consumers now retain
separate positive AO response envelopes, alongside the finite multipole
envelope before adding domain and pair-image omissions. The original total
response reduction remains unchanged. Extra live arrays, exported copies,
positive contractions and receipt history are included in count admission.

The fitting consumer applies the same normalized Hermitian probes, complex
hypotenuse bound, weighted entry caps and L1 cap to each reference component.
It reports maximum J and Section 5.28 trace energy enclosures for finite
approximation, retained-pair real-domain omission, retained-pair reciprocal
omission and omitted pair images. No ordering between the sum of separately
minimized bounds and the combined bound is assumed. Components are distinct
from the finite RI-minus-reference discrepancy. For a fixed source, the
original combined certificate, digest and final target checks are unchanged.

`PeriodicRIFittingReferenceOptions.selective_refinement` defaults to true.
After a failed reference-readiness check, consider only components with a
control that can still advance. Select the largest of each component's J
or energy bound divided by its target and the reference fraction. Ties use
approximation, real domain, reciprocal domain, then pair images. Advance
only that control: reduce tolerance by ten, grow the corresponding lattice
cube by one, or grow pair cube and cutoff within their caps. Record the
action on the next trial. The default allows twelve component steps; the
explicit refinement cap remains authoritative. False preserves the former
joint schedule.

This is a deterministic bounded greedy policy, not an optimality or
convergence theorem. Saturation or exhausted trials still refuse before SCF.
The existing combined reference readiness and full fitting target tests
remain authoritative; a component estimate alone never accepts a source.
Rejected work is charged and the actual first passing RI owner is retained.

The derivation reuses the Ewald1921 and Sun2017 producers and Section 5.20
fixed-family proof; no new integral formula or variational fit assumption
is introduced. Unified Fock, BIPOLE/CRYSTAL conventions and symmetry
permissions are unchanged. Stored-integral, fixed-density scope and the
arithmetic, rank and relaxation exclusions remain in force.


### 5.31 Stored metric spectrum and retained rank

Each periodic RI owner audits its existing Hermitian eigensystem before
constructing the same retained factor. The shared `certify_hermitian_spectrum`
leaf accepts a finite exactly Hermitian complex128 matrix A, ascending real
values L and a square approximate eigenvector matrix Q. Count-only admission
precedes array copies and numerical inspection. Cubic interval work and
linear norm scratch are included in the periodic source plan; no additional
quadratic retained object or coupling tensor is introduced.

Outward binary64 operations enclose R=AQ-QL and G=Q*Q-I. Entry magnitudes use
|real|+|imag|; the maximum row/column sum bounds each spectral norm. Write
r >= ||R||2 and e >= ||G||2. When e<1, sigma_min(Q) >= sqrt(1-e) >= 1-e.
The all-column theorem of Rump and Lange2023, Sections 3 and 4, Eq. (3.2),
Eq. (4.2) and Theorem 4.1, therefore pairs every exact eigenvalue with an
approximation within delta=r/(1-e). Every bound operation rounds outward.
This conservative implementation does not use the paper's cluster-specific
algorithms or assume that a union of individual residual intervals covers
the whole spectrum. See DOI
[10.1016/j.cam.2023.115332](https://doi.org/10.1016/j.cam.2023.115332).

The exact adaptive cutoff max(absolute, relative*max(0,lambda_max)) has its
own interval. Every retained eigenvalue interval must lie strictly above
that interval, and every discarded interval strictly below it. If these
separations have positive minimum m, Weyl perturbation theory gives the
strict radius m/(1+relative): every additional Hermitian matrix perturbation
with spectral norm smaller than this radius preserves the selected rank,
including recomputation of the relative cutoff. Empty retained/discarded
spaces and repeated eigenvalues within a space are allowed. A cutoff crossing
or unproved full rank of Q reports an unresolved certificate, never a repair.

The API returns `spectrum_certified`, `rank_certified`, rank, residual and
orthogonality bounds, eigenvalue error, cutoff interval and rank radius.
When a qualification flag is false its downstream bounds are unavailable;
zero is a sentinel, not a zero-error claim. Unsupported arithmetic modes or
unrepresentable interval operations refuse. The implementation requires
round-to-nearest IEEE binary64 with gradual underflow and forbids fast math.
It neither changes global rounding mode nor relies on BLAS for the proof.

The scope is the exact stored canonical Hermitian metric. It does not bound
errors in integral construction, retained eigenvectors/factors, inverse or
Fock arithmetic, overlap rank, density relaxation or total SCF energy. The
existing full-Coulomb fixed-density bounds retain their arithmetic exclusions.
The radius is diagnostic in this increment, not a new automatic rejection
policy. Source/factor identities, unified Fock composition, BIPOLE/CRYSTAL
conventions and symmetry authority are unchanged. The real metric-response
heuristic guard remains separate and is not promoted to a verified bound.


### 5.32 Rank-aware fitting admission

`PeriodicRIFittingAccuracyOptions.metric_rank_perturbation` optionally requests
a finite nonnegative spectral-norm perturbation margin for the actual selected
Gamma Coulomb metric. `PeriodicRIFittingBoundOptions` exposes the same contract
for raw fixed-family consumers. None retains the earlier acceptance policy;
zero requires a resolved rank even though no positive margin is requested.
The shared Section 5.31 certificate must report `rank_certified` and its strict
`rank_stability_radius` must exceed the requested margin. Equality refuses.
A perturbation whose norm is at most the requested margin then preserves the
selected rank. The value is a caller-declared margin, not an estimate of the
unknown error in Gaussian integrals or the metric itself.

Fixed sources refuse before SCF when the condition fails. Bounded metric,
physical-domain and auxiliary searches use it together with the unchanged
full-Coulomb J/energy targets, retaining the first actual owner that passes
all conditions. The existing refinement order, cutoff floor, domain/basis
caps and cumulative budgets remain authoritative. Neither rank nor radius
is assumed monotone. Exhausted searches refuse; no eigenvalue shift, absolute
value repair, alternate factor or post-SCF retry is introduced.

Reference readiness remains a J/energy reference-quality decision. An
unqualified provisional rank cannot force the reference producer to refine
its lattice domains. Final admission still checks the selected Gamma owner
through the rank condition. Each reference, metric, domain and auxiliary
trial, and the final fitting receipt, exports the requested margin, actual
spectrum and qualification. The full-Coulomb certificate digest continues
to identify its original map/family/targets; the rank contract is separately
reported against the same `ri_source_identity`. An unenforced rejected raw
trial cannot be consumed later. Plans include scalar checks, receipt copies
and all existing reserved failed trials before numerical work.

This reuses the Rump/Lange2023 and Weyl proof in Section 5.31 and the shared
Sun2017/Neese2025 fitting/reference producers. It grants no additional bound
on eigenvectors, factor/inverse or Fock arithmetic, no integral-error estimate,
no overlap-rank guarantee and no relaxed total-energy certificate. It applies
to Gamma fitted J, not every exchange-transfer metric. Unified Fock,
BIPOLE/CRYSTAL conventions, physical source identity and symmetry permissions
are unchanged; `roundoff_included` in the J/energy certificate remains false.


### 5.33 Stored factor inverse error

`certify_hermitian_inverse(A, values, vectors, factor, options, budget)` bounds
`||F F* - C||2`, where F is the actual stored factor and C is the positive
retained inverse of the exact stored canonical Hermitian metric A. It reuses
Section 5.31 once, requires certified adaptive-cutoff rank, and does not
compute another eigensystem. Complete periodic RI owners export the result
as `metric_info.inverse_certificate`; final method provenance carries it
alongside the actual source and factor identities for each metric owner.

Write the sorted approximate eigenvalues as mu, Q for the approximate
vectors, D for the diagonal retained inverse of mu, R = A Q - Q diag(mu),
and e for the verified orthogonality error. Let alpha be the lower enclosure
of the smallest retained exact eigenvalue, beta the smallest retained mu,
dA the upper enclosure of the largest discarded exact eigenvalue, and dB
the largest discarded mu. For a nonempty retained space, the certificate
requires alpha > 0 and, when a discarded space exists, alpha-dB > 0 and
beta-dA > 0. All arithmetic in these bounds expands outwards.

In an exact unitary eigenbasis V of A, put W = V* Q. Entrywise,
`[V*(C Q-Q D)]ij = (f(lambda_i)-f(mu_j)) Wij` and
`[V*R]ij = (lambda_i-mu_j) Wij`, where f is the retained inverse on each
certified spectrum. The finite divided difference is bounded by
`L = max(1/(alpha beta), 1/(alpha (alpha-dB)), 1/(beta (beta-dA)))`.
The mixed terms are omitted for a full retained space; a repeated eigenvalue
inside a space uses the same inverse derivative or zero. Thus
`||C Q-Q D||2 <= L ||R||F`, without dropping higher-order terms. This is a
finite intertwining identity, not a first-order Frechet error estimate.

Using `C-Q D Q* = C(I-Q Q*) + (C Q-Q D)Q*`, square Q gives
`||I-Q Q*||2 = ||I-Q*Q||2 <= e`, `||Q||2 <= 1+e`, and
`||C||2 <= 1/alpha`. The reported eigensystem contribution is therefore
`e/alpha + (1+e) L ceil(sqrt(n)) residual_bound`. The integer square-root
ceiling conservatively converts the spectral residual bound to Frobenius
norm without relying on a rounded square root. No eigenvector orientation
or within-space eigenvalue separation is assumed.

An independent interval contraction encloses every entry of
`F F* - Q D Q*` using the actual stored F, Q and mu. The maximum absolute
row/column sum bounds its spectral norm, including factor normalization
rounding. Adding this `factor_product_bound` to the residual and
orthogonality terms gives `inverse_error_bound`. A certified empty retained
space has an empty factor and exactly zero inverse/error. An unresolved
rank gives `qualified=false`; its downstream zero fields are unavailable,
not zero-error claims. Unrepresentable interval arithmetic refuses.

The shared plan reserves public copies before payload access and adds a
second bounded cubic contraction with linear norm scratch. Periodic owners
reuse their existing eigensystem and factor storage; they do not retain a
new dense inverse. No physical factor, projector, Fock source, symmetry
permission or BIPOLE/CRYSTAL convention changes. This certificate is a
numerical diagnostic, not a new selection target.

The construction combines the residual enclosures of Rump/Lange2023,
Sections 3 and 4, with spectral divided differences described by Higham2008,
Chapter 3 and the author's Frechet-derivative exposition. The finite
identity and conservative norm conversion above are the implementation's
explicit extension; no first-order formula is used as a finite bound.
The new citation feature is `manta_hermitian_inverse`.

This measures an absolute inverse-operator error, which can be conservative
for nearly dependent auxiliary functions. It does not bound errors in the
stored integral metric, coupling integrals, Fock contractions, overlap rank,
relaxed energy, XC, basis or k sampling. The existing full-Coulomb fixed-density
certificate still reports `roundoff_included=false`. A small inverse bound
is not by itself a certified SCF energy tolerance.


### 5.34 Coupling-projected metric action bounds

The optional `metric_action_j_error` and `metric_action_energy_error` controls
in `PeriodicRIFittingAccuracyOptions` and `PeriodicRIFittingBoundOptions`
require a fixed-family numerical metric contribution below both targets.
Both zero disables the new policy; otherwise both must be positive and
finite. The bound compares the exact product of the actual stored RI factor
with the exact retained inverse of the stored metric, applied through the
same stored explicit coupling tiles. It is separate from the full-Coulomb
fitting difference, which already compares the actual factor's map to its
reference. These bounds are not added to that difference or counted twice.

For a coupling column t, define b = F* t, x = F b, e = t-A x and
G = F* A F-I. Let C be the exact retained inverse, P its exact retained
projector, and Pd = I-P. The finite identity for two columns p and q is

`tp* (F F*-C) tq = bp* G bq + xp* Pd tq + tp* Pd xq - xp* A Pd xq - ep* C eq`.

It follows by substituting e and using C A = A C = P. No first-order
remainder is omitted. For full rank the three discarded-space terms vanish.
The action audit reuses the Section 5.33 certificate and its lower retained
eigenvalue alpha. It encloses A F and every entry of G using the shared
binary64 interval arithmetic. If gj bounds the absolute row sum of G, then
`diag(g) +/- G` is positive semidefinite by diagonal dominance. Consequently
`|bp* G bq| <= sp sq`, where `sp^2 = sum_j gj |bjp|^2`.

The exact b and e are enclosed per coupling column. Complex componentwise
absolute sums bound Euclidean norms; outward sums and products propagate
them. Square-root bounds use exact powers of two from the binary exponent,
not an assumed libm square-root error. For partial rank, Section 5.33 gives
`||Pd F|| <= sqrt(inverse_error_bound)`, since `Pd C Pd = 0`. Thus an upper
bound lp on `||Pd xp||` is obtained from this factor bound and the projected
column norm. A stored-metric row norm bounds `||A||`. The scalar pair bound is

`sp sq + ||ep|| ||eq||/alpha + lp ||tq|| + ||tp|| lq + ||A|| lp lq`.

An independent fallback is `inverse_error_bound ||tp|| ||tq||`. The discarded
space estimate can be conservative; the implementation never assumes a
rounded projector is exact. A certified empty retained space has exactly
zero metric action. An unresolved spectrum/rank remains unqualified.

Each positive outer-product term is contracted against the fixed weighted
spin-total density family. For any scalar column envelope v, both
`L max(v)` and `sum(hq vq)` bound its contraction when L and component bounds
h are available. The implementation also uses both constraints jointly.
For every nonnegative breakpoint lambda, the elementary dual inequality

`sum(vq dq) <= lambda L + sum(hq max(vq-lambda,0))`

holds whenever `0 <= dq <= hq` and `sum(dq) <= L`. Sorting v in descending
order and accumulating capacities selects a breakpoint near the primal
allocation optimum. That selection need not be exact: the inequality holds
for every selected lambda. The dual sum, positive differences and products
are enclosed outward with the same binary64 arithmetic as the metric audit.
The result is the minimum of this enclosure and the two preceding valid
bounds. No rounded unused capacity is treated as an exact remainder. Zero
component limits, ties, nonbinding total limits and near-equal capacities
remain valid. Heap sorting and linear scratch are included in public and
aggregate admission plans; sorting work scales as `N log(N)`.

Each output keeps the smaller of the projected and fallback bounds. The
energy bound applies the same joint support function to the resulting J
entry envelopes and then multiplies by one half. This is a half-trace
inequality, not the exact maximum of the quadratic energy error. This remains valid for complex indefinite Hermitian densities,
zero component limits and zero-weight k points. There is no positivity or
variational-fit assumption on independently truncated Coulomb integrals.

`HermitianInverseAction` provides reusable admitted column/family APIs.
Columns are tied to their immutable owner; mixing owners refuses. Public
array copies and all cubic setup, quadratic column work and family reductions
are admitted before payload access. Periodic fitting reads one AO shell-pair
panel across auxiliary shells via the existing `PeriodicGDFTiles.evaluate`
API. It retains only scalar column envelopes and final entry bounds, never a
complete auxiliary/AO-pair tensor. The actual factor and eigensystem remain
owned by the existing RI producer; no alternate Fock factor is substituted.

The projected policy supports explicit tile actions, including coupling
screening and ordered multipole tiles. Section 5.35 extends metric admission
to ordered compressed/global moment representations using norm bounds.
Pair-tree actions retain the guard; their existing approximation controls
remain available. Actual contraction arithmetic remains excluded in all cases.

Fixed sources reject unmet targets before SCF. Existing bounded metric,
domain and auxiliary searches require these targets along with the original
full-Coulomb and optional rank targets, and retain the actual first passing
owner. Reference readiness remains a separate full-Coulomb reference-quality
decision. No monotonicity of the metric action error is assumed. Exhaustion
refuses, and a rejected raw trial cannot later be consumed. Final and trial
`metric_action` provenance reports targets, qualification, bounds, column and
producer work counts beside the same physical RI source identity.

Rump2010 Section 10.8, Eq. (10.52) supplies the inverse-norm/residual principle;
Rump/Lange2023 supplies the verified retained spectrum. The quadratic and
discarded-projector identities above are this implementation's explicit
finite extension, not a copied Cholesky verification algorithm. The original
Sun2017/Neese2025 coupling and periodic conventions are unchanged. Citations
are emitted under `manta_hermitian_inverse_action` when this policy is used.

The bounds cover metric eigensolver and factor-construction errors relative
to exact stored metric/coupling entries. They do not include integral
construction, actual coupling/Fock contraction roundoff, overlap rank,
relaxed SCF energy, XC, basis or k sampling. The separate full-Coulomb
certificate still reports `roundoff_included=false`. Unified Fock,
BIPOLE/CRYSTAL partitions and symmetry authority are preserved.


### 5.35 Compressed coupling norms for metric action

For ordered compressed auxiliary hierarchies, define T as the exact real
arithmetic graph of the actual stored residual tile entries, auxiliary and
pair moment panels, translation matrices, interaction matrices and complex
phases. This is a precise comparison object; a separately materialized,
rounded dense tensor need not equal T entrywise. Global-panel reuse evaluates
the same graph algebraically and shares this bound. The global pair-tree
extension below covers its distinct gathering and Gamma orbit averaging.

The producer propagates positive component envelopes on its existing upward
auxiliary traversal. Leaf panels use their maximum absolute entry per moment
component and the componentwise absolute sum of each actual stored phase.
Each parent applies `u_parent += abs(W) u_child`. For each selected far block,
the positive contraction `sum_ij u_i abs(I_ij) p_j`, multiplied by the maximum
stored pair-phase component sum, bounds every auxiliary-row/AO-pair entry.
Here p is the maximum absolute pair-panel entry per component. Adding all
accepted far blocks and the absolute real/imaginary components of each stored
residual tile gives an entry bound. Summing these over auxiliary rows gives
`c_q >= ||T_q||_2` via the L1 norm. All envelope sums and products use the
shared outward binary64 primitives and their arithmetic guard. No computed
phase is assumed to have exact unit modulus; no cancellation is credited.

Let eta be the verified Section 5.33 bound on `||F F*-C||_2`, where C is the
exact retained inverse of the stored metric. Then

`|T_p* (F F*-C) T_q| <= eta c_p c_q`.

Using the joint density support H from Section 5.34 gives
`J_p <= eta c_p H(c)` and `E <= H(J)/2`, with outward arithmetic throughout.
An unresolved inverse certificate remains unqualified. A certified empty
retained inverse and factor give exact zero. Supplied norm envelopes must be
finite and nonnegative. These statements require no density positivity or
variational-fit assumption.

`PeriodicGDFTiles.coupling_norm_bounds` owns the traversal and fixed source
partition. `HermitianInverseAction.norm_family` owns inverse-error propagation
and the shared density support; periodic fitting only composes these APIs.
The audit runs a zero forward action with additional scalar envelopes and
reuses magnitude panels. It leaves normal forward/adjoint actions, source
identity, actual RI factors, unified Fock and symmetry authority unchanged.
Count plans reserve the extra traversal, exported column norms and outward
work. Aggregate fitting admission conservatively retains explicit-panel
reservations as well. No dense coupling tensor is materialized by the audit.

Neese2025 Eqs. (29)-(36) supplies the compressed gather/translate/scatter
structure; Sun2017 Eqs. (16)-(17) supplies the periodic convention. The positive
DAG enclosure and norm-only metric propagation above are this implementation's
explicit extension, using the verified inverse result rather than asserting
a new integral algorithm. Integral construction, compressed/Fock contraction
roundoff and relaxed SCF energy are excluded. This is not a complete arithmetic
ledger or a claim of useful accuracy for arbitrary production systems.

#### Global pair-tree stored graph

For each leaf, the maximum absolute pair-panel coefficient per moment is
multiplied by the maximum real/imaginary component sum of its stored Bloch
phases. At a parent, the componentwise maximum of `abs(W) u_child` bounds
any one descendant column. This is a maximum across different leaves, not
their charge sum. Each accepted auxiliary interaction contributes
`abs(phase_aux) sum_ij a_i abs(I_ij) u_j` uniformly to its represented leaves,
where a is the componentwise maximum of the stored auxiliary panel.

These positive interaction envelopes propagate down the existing tree and
are summed over original images of each AO shell pair. Exact near blocks
contribute their maximum stored absolute entry times both stored phase
bounds. Multiplication by the auxiliary shell size and summation over shells
then bound each column's auxiliary L1 norm. All positive sums and products
use the same outward binary64 primitives as the ordered audit. The source
partition, packed total-degree interactions and selected finite masks stay
unchanged; the audit reuses the existing magnitude workspace.

At Gamma, the producer's conjugate-completed coupling is
`T(k,mu,nu) = (T0(k,mu,nu) + conj(T0(k,nu,mu)))/2`.
Apply that same convex average to the two column bounds with the AO indices
transposed on the mate. Add the norm of the stored reciprocal/zero residual
after this averaging. The extra column and mate buffers fit the existing
norm-audit reservation; the count plan charges outward traversal work before
execution. No auxiliary-by-AO-pair tensor is allocated in production.

The resulting column norms enter the unchanged inverse-error and joint
density-family support formulas above. This admits pair-tree metric-action
targets without certifying errors in integral construction, actual gathering,
translation, scattering or Fock arithmetic. Source symmetry admission and
evaluated Fock/solver policy remain separate authorities.

### 5.36 Executed paired metric arithmetic

The shared `certify_paired_metric_roundoff` leaf accepts the actual stored
factor F, forward-action output x, computed projection r, reconstruction c
and Coulomb energy e. Its reference is exact arithmetic on these stored F
and x: `r0 = transpose(F) x`, `c0 = conjugate(F) r0`,
`e0 = sum_j (Re(r0_j)^2 + Im(r0_j)^2)/2`. This is the paired RI convention
of Section 5.34, not an adjoint substituted for the transpose.

Outward complex interval products and sums enclose r0. Those intervals then
propagate through the stored conjugated factor to c0 and through the squared
norm to e0. Subtracting the actual returned r, c and e gives per-component
and maximum absolute error bounds. Complex errors use the sum of absolute
real/imaginary component errors. The replay does not assume a BLAS reduction
order, ignore cancellation or assume every product is normal. It uses the
existing round-to-nearest/gradual-underflow guard and fails closed on
unrepresentable enclosures. An empty factor and exact zero outputs have
exact zero arithmetic error.

`plan_paired_metric_roundoff` admits work and storage before copying Python
payloads. Native callers borrow F; the periodic q/Fock plans conservatively
reserve the maximum-rank audit alongside their existing workspaces. The
audit does not recompute or substitute a numerical factor, change a density
or alter physical source identities. Complete paired RI builds requesting
metric-action certification report `metric_roundoff_evaluated` and separate
`metric_projection_roundoff_bound`, `metric_coefficient_roundoff_bound` and
`metric_energy_roundoff_bound` values. Raw q builds may request the same audit
with `audit_metric_arithmetic=True`. Retained-mode and fused J/K paths report
unevaluated; the request does not force a different contraction route.

These are actual-input, post-build arithmetic bounds. They are not the
pre-SCF fixed-family inverse error and are not folded into its target or the
full-Coulomb discrepancy twice. The coefficient error alone is not a J error. The propagation below applies
when an admitted fitting owner retains the required source norm. Forward/adjoint
coupling arithmetic, spin/k-weighted density construction, final J/K/Fock
assembly, integral construction and relaxed SCF energy remain excluded.
Consequently `roundoff_included=false` remains correct for the full-Coulomb
receipt, and the wider executed-arithmetic work package remains open.


#### Metric arithmetic contribution through the stored adjoint

The fitting owner retains `metric_action_adjoint_norm_bound`, the maximum
of the outward auxiliary L1 column norms already computed during metric-action
admission. Explicit sources reuse their streamed column certificates; ordered,
global-panel and pair-tree sources reuse the producer's stored-graph norm
audit of Section 5.35. No new integral sweep, coupling tensor or numerical
factor is needed. The plan charges the scalar scan and outward propagation.

Let epsilon_c bound the maximum component of the executed coefficient error,
and let N bound every stored coupling column's auxiliary L1 norm. The exact
stored adjoint obeys `max_p |sum_a T_ap delta_c_a| <= N epsilon_c`.
An outward multiplication reports `metric_j_roundoff_bound` and sets
`metric_j_roundoff_evaluated` only for an audited paired metric result belonging
to this same admitted source. Complex component L1 bounds dominate the complex
modulus and remain valid under cancellation. Zero coefficient error yields
exact zero contribution. The norm is immutable across builds and physical
source/snapshot identities remain unchanged.

This is the metric substep's contribution through the exact adjoint, not a
bound on rounding in the executed coupling adjoint. Raw q builds without a
fitting certificate, retained-mode and fused J/K paths leave propagation
unevaluated. It remains a separate post-build diagnostic; inverse-action
pre-SCF targets, full-Coulomb `roundoff_included=false`, forward-input errors,
weighted density and final Fock arithmetic are unchanged. This advances #624
without closing its remaining executed-coupling and assembly scope.


### 5.37 Executed periodic coupling arithmetic

The producer's optional `audit_roundoff=True` action replays outward complex
intervals alongside the existing forward or adjoint schedule. Its reference
is exact arithmetic on the declared stored representation of Section 5.35:
explicit residual tiles, stored moment panels, translations and interactions,
near blocks and actual stored phases. Kernel, moment, phase and residual-tile
construction errors remain separate. This is not exact arithmetic on ideal
integrals or on a rounded exported compressed tensor.

For ordered/global panels, the replay folds phases and input into pair charges,
propagates auxiliary hierarchy charges or fields, then performs the existing
panel projections. Global actions retain interval charges/fields beside their
source-owned panel references. Pair-tree actions replay upward and downward
translations on the same nodes and accepted antichain, including near blocks.
Gamma completion applies the same conjugation, AO transpose and convex average
to the interval outputs. The residual tile contraction is then included in the
same output interval. No new partition, density-dependent selection or dense
auxiliary-by-AO-pair tensor is introduced.

All products and sums use the existing binary64 outward interval primitives.
The final interval is subtracted from the actual returned numerical output,
so the bound includes its compensation/reduction order without assuming a
normal-only relative-error model or assigning an exact unit modulus to a
stored phase. Per-output `roundoff_bounds` and maximum `roundoff_bound` use
complex component L1 error; `roundoff_evaluated` distinguishes evaluated zero
from an absent audit. Arithmetic mode guards, nonfinite checks and enclosure
overflow fail closed. Exact zero input has zero output error.

`plan_coupling_roundoff` preflights all five caps before input copies. It
reserves interval vectors and temporary products beside existing action/tree
workspaces, Gamma mates and binding exports; its work allowance covers the
outward replay. Numerical values, source/snapshot identity and integral passes
remain unchanged. Complete paired RI builds requesting metric arithmetic
also audit their actual forward and adjoint inputs, reporting
`coupling_forward_roundoff_bound`, `coupling_adjoint_roundoff_bound` and
`coupling_roundoff_evaluated` through the shared q/Fock receipts and provenance.
Retained-mode and fused J/K routes do not claim this paired audit.

The two action bounds are separate actual-input contributions. Section 5.38 composes forward error through the fixed metric and adjoint;
the adjoint bound alone is not their sum. Spin/k-weighted input construction, integral/source construction, final
J/K/Fock assembly and relaxed SCF energy remain outside this ledger. Pre-SCF
inverse targets and full-Coulomb `roundoff_included=false` remain unchanged.
This advances #624 without closing its composition and assembly requirements.


### 5.38 Paired periodic J and energy arithmetic composition

For the actual stored spin-summed, k-weighted input d, let the exact stored
coupling graph define x0 = conjugate(T) d. The paired reference is
rho0 = transpose(F) x0, c0 = conjugate(F) rho0, J0 = transpose(T) c0,
and E0 = norm(rho0)^2 / 2. F is the existing stored retained factor.
This extends the finite paired convention of Sun2017 Eqs. 16-17 and
Neese2025 Eqs. 29-36 with the outward arithmetic of Sections 5.36-5.37;
it does not replace the inverse-approximation reference of Section 5.34.

The executed forward action supplies componentwise complex L1 bounds eta
on x minus x0. `certify_paired_metric_action` accepts these nonnegative finite
bounds alongside the actual x, projected vector, coefficient vector and energy.
It encloses each exact input in a complex rectangle of radius eta about the
stored x, then replays both fixed-factor products and the squared norm with
outward intervals. Subtracting the actual numerical witnesses yields combined
forward-plus-metric projection, coefficient and energy bounds. It includes
all propagated input and metric arithmetic, including higher-order terms.
It does not add the metric-only diagnostic a second time. Zero input radii
reuse the original point replay; zero rank consumes no uncertain input and
has exactly zero error for zero numerical outputs. Shape, finite, arithmetic
mode, resource and overflow guards fail closed.

Let epsilon_c be the maximum combined coefficient bound, N the same-source
auxiliary L1 column norm admitted in Section 5.36, and epsilon_adj the executed
adjoint error at the actual computed coefficients from Section 5.37. Then
max_p abs(J_p - J0_p) <= N epsilon_c + epsilon_adj. Both final operations
round outwards. The energy bound is the combined metric energy bound because
the numerical energy is computed before the adjoint. The same argument uses
complex component L1 norms, which dominate complex modulus.

Q receipts report `forward_metric_roundoff_evaluated`,
`forward_metric_coefficient_roundoff_bound` and
`forward_metric_energy_roundoff_bound`. The fitting owner combines these with
its existing source-qualified norm and actual adjoint bound, setting
`paired_roundoff_evaluated`, `paired_j_roundoff_bound` and
`paired_energy_roundoff_bound`. Shared Fock receipts and standard provenance
carry all fields. Raw q calls without a fitting norm expose only the combined
metric diagnostic. Retained-mode and fused J/K paths remain unevaluated.

`plan_paired_metric_action` charges interval input storage and widened products
in addition to metric replay; the q plan includes both old and combined audits
and their live outputs. No new source sweep or dense coupling tensor is needed.
The existing numerical schedule, source, metric rank, physical evaluated state,
Fock/eigensolver and symmetry admission stay unchanged.

The comparison begins at the actual stored weighted input and stored graph.
Spin summation and k weighting, integral/phase/moment/source construction,
final J/K/Fock assembly and relaxed-SCF error remain separate obligations under
#623-#627. Existing pre-SCF inverse-action targets still apply to their original
contribution. Full-Coulomb `roundoff_included=false` remains mandatory. These
post-build paired bounds do not by themselves establish automatic whole-method
accuracy or close #624.


### 5.39 Stored-density spin and k weighting

The paired arithmetic comparison can now begin at the supplied stored spin
densities D_s and k weights w, before spin summation and multiplication.
For each AO-pair entry p at k, the reference is
u0_p = w_k sum_s D_s,p, evaluated in exact arithmetic on those stored operands.
An outward complex interval sums the original spin entries and multiplies by
the stored weight. Subtracting the actual numerical weighted input u gives a
per-entry complex L1 error eta. The numerical summation and multiplication
order are unchanged. `weighted_input_roundoff_evaluated` and
`weighted_input_roundoff_bound` report this input step and its maximum error.

The shared producer forward/adjoint APIs accept optional `input_errors`, a
contiguous float64 vector of nonnegative finite complex L1 radii. Supplying it
requires `audit_roundoff=True` and matching flattened input length. The
existing interval graph starts from rectangles about the actual input, with
radius eta in each real/imaginary component. Zero radii preserve the point
replay. The Gamma mate receives the conjugate/permuted input intervals,
including the AO transpose for a forward density, before its normal action.
All explicit residual, ordered/global-panel, pair-tree near and hierarchical
paths use these same intervals. Numerical values and source/input/snapshot
identities stay unchanged; `input_error_included` identifies the extended
comparison scope on the action receipt. The caller supplies the input-error
hypothesis; the producer verifies shape, arithmetic mode, finite values,
resources and enclosure overflow, not an external error model's derivation.

The returned forward enclosure covers both input uncertainty and executed
coupling arithmetic. Its rectangle contains the original stored input too,
so it remains a valid, possibly wider, actual-input rounding bound. The
paired metric audit of Section 5.38 consumes these per-RHS enclosures once.
The fitting owner then combines its coefficient enclosure with the existing
same-source norm and actual-input adjoint error. The reported paired J and
energy bounds therefore enclose the exact stored paired graph at u0, without
adding the input contribution or metric arithmetic twice.
`coupling_forward_input_error_included` and `paired_input_roundoff_included`
mark the extended scope on q/Fock receipts and standard provenance. They are
false on unevaluated routes. Raw q calls without the fitting norm still cannot
claim the composed J bound, though their forward/metric audit includes input
weighting. Retained-mode and fused J/K execution are not replaced to obtain
an audit.

Resource plans charge input rectangles, Gamma copies, binding error arrays
and the spin/weight interval work. No new source sweep, partition or dense
coupling tensor is added. This uses the paired convention and outward
arithmetic references in Sections 5.36-5.38. Source/kernel/support, retained
factor, Fock/eigensolver and symmetry authority remain unchanged.

These bounds begin at stored D_s and w; they do not certify the orbitals or
operations that produced D_s, or uncertainty in the physical inputs.
Integral/source construction, final J/K/Fock accumulation, other energy terms
and relaxed-SCF error remain excluded. Full-Coulomb `roundoff_included=false`
and existing pre-SCF inverse-action targets remain unchanged. #624 now
requires final assembly and its remaining non-paired/K routes; #623-#627 and
the full capability matrix retain their other obligations.


### 5.40 Evaluated physical Fock and energy assembly

For an admitted paired arithmetic path, the native HF/KS method adapter now
retains a shared interval audit of its stored physical components. With stored
method density D_s, k weights w, one-electron H, Coulomb J, semilocal Vxc,
exchange components K_t with signed method coefficients a_t, and actual stored
zero-mode correction C, its exact reference is
F_s = H + J + Vxc_s - sum_t a_t K_ts + C_s.
The energy reference is Tr_w(D H) + EJ + EXC + Enuc
- (1/2) sum_t a_t Tr_w(D K_t) + (1/2) Tr_w(D C).
Here EJ, EXC and Enuc are the method's actual stored component energies, and
Tr_w is the real spin-summed, k-weighted matrix trace. RKS direct exchange uses
a_t equal to half the physical exchange fraction; fitted K built from equal
spin halves uses the full fraction. Range components retain their signed
coefficients. HF uses the corresponding stored K and unit coefficient.
The audit consumes the method's actual component arrays, with no new source
calculation, partition, eigensolve or numerical contraction in the physical path.

The existing outward binary64 interval operations enclose these expressions
at exact stored operands, including density weighting and trace arithmetic.
This reuses the verified-computation basis in Sections 5.36-5.39 and Rump
(2010), DOI 10.1017/S096249291000005X. The audit then compares that enclosure
with the actual public evaluated Fock and method energy. Thus the reported
complex entrywise L1 maximum covers both method assembly and the numerical
solver's callback reconstruction and Hermitian projection. Capped states use
the last evaluated state; converged states retain the existing confirmation.
No additional confirmation call is introduced. The shared Fock/eigensolver
policy, physical/source identities and source-specific symmetry admission
remain unchanged.

`assembly_roundoff` is a method receipt, separate from the partial RI
`fitted_fock`. Its `evaluated`, `fock_bound` and `energy_bound` describe only
this stored-component comparison. `paired_coulomb_included` marks composition
with the accepted paired J/energy bounds of Section 5.39:
combined_fock_bound = outward(fock_bound + paired_j_roundoff_bound),
combined_energy_bound = outward(energy_bound + paired_energy_roundoff_bound).
No forward, metric, weighting or adjoint term is added a second time.
The paired graph begins at the stored source spin-density embedding, whereas
the assembly traces use the public method density. Errors in generating those
densities, including restricted-to-source embedding, remain outside the paired
source certificate. The combined comparison holds the stored H, K, Vxc,
EXC, Enuc and correction values fixed; it replaces only the stored Coulomb
response/energy by their exact stored paired graph values.

Persistent interval F and weighted-density arrays and all audit work are
reserved in complete HF/KS driver plans before SCF. The raw diagnostic
`plan_periodic_assembly_roundoff(n_basis, momenta, spins, budget)` admits the
single-exchange comparison. `certify_periodic_assembly_roundoff` accepts
complex128 H/J[k,n,n], D/K/Vxc/C/F[spin,k,n,n], float64 weights[k], a signed
exchange scale, stored EJ/EXC/Enuc and the actual total energy. It admits counts
before copying arrays or inspecting payload values, requires finite operands,
and refuses arithmetic overflow or unsupported rounding modes. This diagnostic
has no independently supplied paired-source certificate and therefore returns
assembly-only bounds. The actual method may add multiple signed exchange
components through the same helper.

The audit is active only when the paired Coulomb arithmetic is qualified;
other routes explicitly leave `evaluated=false`. Integral/phase/moment/source
construction, the actual K/XC/correction construction, density generation,
rank/relaxation errors and non-paired execution remain separate obligations.
Pre-SCF inverse-action targets and full-Coulomb `roundoff_included=false` are
unchanged. #623-#627 and the full capability matrix remain unfinished.


### 5.41 Executed direct periodic exchange contraction

The existing finite reciprocal exchange producer optionally audits its actual
stored Bloch Fourier panels B, input density channels D and real reciprocal
factors f. For each state s and target k its exact stored-input reference is
K0_s,k = sum_(source,Q) f_(source,Q) B_Q^dagger D_s,source B_Q.
The factor is the actual stored coefficient used by the numerical producer,
including its source k weight, Fourier kernel and cell-volume division. Its
construction is held fixed. The interval audit encloses D B and then B^dagger
(D B), scales by that stored factor and accumulates over the existing support.
It compares the resulting intervals with the actual compensated output K.
The numerical Eigen products, panel batching, compensation and source schedule
are unchanged. This uses the same Sun2017 Fourier convention and shared
Rump2010 outward arithmetic as the earlier paired and assembly audits.

`audit_roundoff=True` on `plan_periodic_exchange` or
`compute_periodic_exchange` admits the interval response and scratch storage,
per-entry error output and cubic audit work before payload copies or values.
The plan marks `roundoff_requested`. The result's `roundoff_bounds` array has
shape [density,k,AO,AO] and encloses each entry's complex L1 difference from
K0. The receipt reports `roundoff_evaluated`, its maximum `roundoff_bound`, and
`roundoff_scope=stored-bloch-panels-and-reciprocal-factors`. Unaudited calls
return no entry bounds and keep the scope flag false. There is no additional
Fourier evaluation or four-index AO tensor. Empty or zero-factor support and
zero densities retain exact zero bounds. Bare Coulomb, erf and erfc use their
original reciprocal and zero-mode conventions.

The producer also encloses each reported channel energy against
E0_s = -(1/2) sum_k w_k Re Tr(D_s,k K0_s,k), using exact stored target weights
and densities. `energy_roundoff_bounds` includes both K-contraction and final
weighted energy-trace arithmetic. It is a complete producer energy diagnostic,
not an extra term to add to the method assembly bound of Section 5.40.

Complete RI HF/KS methods requesting the existing metric-action accuracy
policy admit this audit for their complementary direct exchange components.
When the paired Coulomb path qualifies, the existing assembly helper consumes
those K-entry bounds and the actual signed exchange coefficients a_t. It
reports `direct_exchange_roundoff_included`,
`direct_exchange_fock_bound` and `direct_exchange_energy_bound`. For each Fock
entry the added radius is sum_t |a_t| eta_t,entry. Its energy contribution is
(1/2) sum_(t,s,k,i,j) |a_t| |w_k D_s,k,j,i|_1 eta_t,s,k,i,j,
with outward bounds on each term. The maximum entry radius is added once to
the combined Fock bound, and the energy contribution once to the combined
energy bound, after the accepted paired Coulomb and assembly terms. The
producer energy-trace bound is not added again because the assembly audit
already compares the reported method energy with exact traces at stored K.
Restricted/global/range scaling stays method-owned. Pure KS adds no direct-K
term, and unaudited or fitted K is not silently relabelled as direct exchange.

This extends the combined comparison to exact direct-K contraction at stored
B and f. Fourier integrals, Bloch phase and reciprocal-factor construction,
density generation and restricted source embedding, correction and XC source
construction, fitted/retained/fused K arithmetic, rank and relaxed-SCF errors
remain separate obligations. Full-Coulomb `roundoff_included=false` and
pre-SCF inverse-action targets are unchanged. No Fock object, solver policy,
BIPOLE/CRYSTAL source convention or symmetry permission changes. #624 and
full MANTA #617 remain open.

The raw `certify_periodic_assembly_roundoff` diagnostic also accepts optional
`exchange_errors`, a finite nonnegative float64 [spin,k,n,n] array of complex
L1 radii about its stored K. The caller owns the validity of that hypothesis.
It uses the same propagation and records the direct-exchange scope flag; zero
radii recover the assembly-only numerical bounds. Admission precedes payload
inspection. Exact rational perturbed-K references qualify signed coefficients,
spin channels and the weighted trace composition independently.


### 5.42 Stored-input exchange zero-mode correction arithmetic

The existing BvK exchange correction is C0 = -a m S D S, with exact
interpretations of the stored overlap S, source spin density D, Madelung
coefficient m and method exchange scale a. The shared outward arithmetic
from Section 5.40 encloses both matrix products and the product of the two
stored scalar coefficients, then compares each entry with the actual
compensated correction. It does not replace S with its adjoint or change the
physical correction order. Restricted HF audits its actual half-density
source channel and propagates the radius using the public total density;
KS retains its method-owned restricted or independent-spin exchange scale.

The method receipt and standard provenance expose
`exchange_correction_roundoff_included`, `exchange_correction_fock_bound`
and `exchange_correction_energy_bound`. Given per-entry complex L1 radii
eta, the Fock contribution is max eta, and the energy contribution is the
outward sum (1/2) sum_(s,k,i,j) |w_k D_s,k,j,i|_1 eta_s,k,i,j.
These contributions are added once to the combined assembly, paired Coulomb
and direct-K bounds. The assembly-only reference still holds its stored C
fixed and already includes the correction trace arithmetic; the construction
radius must not be added twice. Strict-zero-mode and pure KS have no
correction contribution. Converged and capped evaluated states follow the
same existing state and terminal-confirmation contracts.

The raw `plan_periodic_correction_roundoff` and
`certify_periodic_correction_roundoff` APIs expose this same comparison for
S[k,n,n], D[spin,k,n,n] and actual C[spin,k,n,n], all complex128. The diagnostic
returns float64 radii of D's shape and accepts one or two spin channels.
Memory and cubic arithmetic caps are checked before copies or payload
inspection; parent HF/KS plans charge audit scratch and every callback's
work. The optional `correction_errors` argument of
`certify_periodic_assembly_roundoff` consumes finite nonnegative radii with
that shape; the caller owns their validity. Zero radii preserve the previous
assembly-only numerical bounds. No additional physical-source sweep or
four-index tensor is introduced.

This extends the reference only to exact arithmetic on stored S, D, m and a.
Overlap-integral and Madelung-source construction, restricted density
embedding, fitted/retained/fused RI arithmetic, XC, rank and relaxed-SCF errors
remain outside it. Full-Coulomb `roundoff_included=false`, source identities,
pre-SCF inverse targets, shared Fock/eigensolver, BIPOLE/CRYSTAL and symmetry
policies are unchanged. Exact rational independent four-index expansions and
complete-method controls qualify the new contribution. The Rump2010 reference
and shared arithmetic assumptions from Section 5.40 apply. #624 and #617 remain
open.


### 5.43 Retained-mode periodic RI arithmetic

The optional `audit_metric_arithmetic` build request also audits the existing
retained-mode route. At each streamed mode block, the exact stored-input
reference is B = F^dagger T, with the actual retained factor F and the exact
values of the tiles T returned in that sweep. Shared outward binary64
arithmetic encloses the factor contraction without another source sweep or a
full auxiliary-by-AO-pair tensor. The physical compensation, block size,
route selection and source identities are unchanged.

For J, rho_L = sum_(s,k,i,j) w_k conj(B_L,k,i,j) D_s,k,i,j,
J_k,i,j = sum_L B_L,k,i,j rho_L, and E_J = (1/2) sum_L |rho_L|^2.
The interval comparison includes spin addition, stored k weights and all
retained-mode products/reductions, including energy accumulation. For K,
K_s,k = w_k sum_L B_L,k D_s,k B_L,k^dagger, with its existing target/ket
q-channel convention. The audit encloses both matrix products, the weight
and the reduction across mode blocks. The Rump2010 arithmetic assumptions of
Section 5.40 and the Sun2017 source conventions of the producer apply.

`PeriodicRIQResult.J_roundoff_bounds` and `K_roundoff_bounds` contain complex
L1 entry radii with the respective J/K shapes when audited; unaudited arrays
have empty leading dimensions. The receipt separately reports
`retained_j_roundoff_evaluated`, `retained_k_roundoff_evaluated`, their maximum
`retained_j_roundoff_bound` and `retained_k_roundoff_bound`, and the absolute
`retained_energy_roundoff_bound` for E_J. Zero-rank or zero-density results
retain exact zero bounds. The existing count-only plan reserves interval mode
blocks, response arrays, spin/weight inputs, scratch and cubic contraction
work at maximum rank before construction. The audit allocates no unbounded
coupling owner and executes no additional integral evaluation.

Complete RI HF/KS with the existing metric-action request propagate the
retained J and E_J radii through the shared evaluated assembly ledger.
`retained_coulomb_included` distinguishes this contribution from
`paired_coulomb_included`; the two routes cannot be counted simultaneously.
The retained contribution is included once in combined Fock and energy bounds,
followed by any admitted direct-K and BvK-correction arithmetic. The physical
J/K build, native eigensolver and terminal/capped state contracts are preserved.

Raw fused K radii refer to one q channel. Section 5.44 composes these
radii through the complete fitted-K cross-q accumulation and evaluated method
assembly; the present section defines only the individual channel reference. Tile/integral construction, factor/eigensystem and rank
perturbations, restricted density embedding, XC and relaxed-SCF errors remain
excluded. No metric-only or paired bound is added again to the retained
reference. Full-Coulomb `roundoff_included=false`, #624 and the full #617 matrix
remain open, with source construction and whole-method accuracy separate.


### 5.44 Whole-mesh fitted exchange arithmetic

The existing ordered-q RI Fock builder optionally encloses accumulation of
its stored Gamma and transfer-channel K values. For each target spin, k point
and AO entry, let c_q be the computed channel value and e_q its retained
complex L1 radius from Section 5.43. Outward interval addition encloses
sum_q c_q. Comparing the actual compensated sum with that enclosure gives
an accumulation radius e_sum. The resulting whole-mesh radius is the upward
sum e_sum + sum_q e_q. Each channel error enters once; no auxiliary tensor,
extra integral sweep or replacement physical contraction is introduced.
The Rump2010 arithmetic assumptions of Section 5.40 apply.

`PeriodicRIFockBuilder.build(audit_arithmetic=True)` returns these radii in
`K_roundoff_bounds[spin,k,i,j]`. The receipt records
`fitted_exchange_roundoff_evaluated` and the maximum entry radius
`fitted_exchange_roundoff_bound`. An unaudited result has an empty leading
dimension. The K radius does not depend on the requested exchange scale.
Zero rank or zero density retains zero error. The existing count-only Fock
plan reserves the interval accumulator, channel and result radii and all
ordered-q summation work before constructing sources. Physical source,
projector, frame and density identities are unchanged.

Complete RI HF and global/range KS reuse the existing metric-action request
and evaluated assembly audit. A fitted exchange component with signed
coefficient a contributes |a| e_K to the Fock radius and
(1/2) sum_(s,k,i,j) |w_k D_s,k,j,i| |a| e_K,s,k,i,j to the energy radius.
Outward arithmetic also encloses coefficient multiplication, density pairing
and final assembly. Range components retain their separate physical kernels,
projectors and coefficients, including negative coefficients. The method
owns the BvK correction and its separate Section 5.42 contribution.

`assembly_roundoff.fitted_exchange_roundoff_included`,
`fitted_exchange_fock_bound` and `fitted_exchange_energy_bound` distinguish
this contribution from direct-exchange errors. Combined bounds include the
selected Coulomb route, direct and fitted exchange contributions, and the
correction once each. The diagnostic raw assembly function uses
`fitted_exchange=True` to classify supplied `exchange_errors` accordingly.
The raw Fock-builder K certificate alone does not certify its energy or Fock
assembly; that further claim requires the method assembly receipt.

The reference holds the input density, weights, coupling tiles and retained
factors fixed. Integral/source construction, eigensystem and rank changes,
restricted density embedding, XC construction and relaxed-SCF errors remain
excluded. Full-Coulomb `roundoff_included` remains false. This does not close
#624, #626 or full #617, change shared Fock/eigensolver or source-qualified
symmetry policy, or establish BIPOLE/CRYSTAL source equivalence. The known
reference-budget refusals in #643 remain separate.


### 5.45 Full-Coulomb fitted exchange on a fixed density family

`certify_periodic_ri_exchange_fitting` consumes an existing ordered-q RI
Fock owner and the existing periodic pair source, finite reciprocal exchange
producer and full-image/full-reciprocal exchange enclosure. It compares their
linear K actions; it does not assume a positive or variational fit residual
between independently truncated metric and coupling integrals. Sun2017,
Section II.B, defines the periodic fitting convention. The existing exchange
source and tail contracts retain their physical meaning and zero-mode policy.
Rump2010 outward arithmetic encloses probe subtraction and propagation.

The family is explicitly unweighted: sum_(s,k,i,j) |D_s,k,i,j| <= L, with
optional symmetric nonnegative B_k,i,j bounding
|D_alpha,k,i,j| + |D_beta,k,i,j|. D must be exactly Hermitian. This is not a
bound on the potentially cancelling signed spin total. Canonical solver
bounds may be converted from their weighted convention with outward products;
taking the maximum of two transposed entry bounds supplies a symmetric upper
envelope without projecting a physical density or granting symmetry reduction.
The source basis, literal cell, k frame and weights must agree exactly.

At each source k, the probe basis is E_ii, P_R = E_ij + E_ji and
P_I = i(E_ij - E_ji), for i < j. There are n_k n_AO^2 probes, independent of
spin count. The existing `build_exchange` component action reuses the actual
Gamma and transfer owners and their retained factors. It omits J and its
full-Coulomb fitting-family check; existing provider-specific admission still
applies. Its source identity distinguishes the component action from a full
Fock build. No second metric/source owner or Coulomb algorithm is introduced.

For each output entry, the finite difference coefficient includes the
outward difference of the computed RI and finite-reference probe responses
plus both accepted contraction-arithmetic radii. The reference coefficient
is supplied separately by the existing full-image/full-reciprocal bound on
that same probe. For either coefficient family, an off-diagonal pair with
real/imaginary bounds a_R and a_I has coefficient
c = sqrt(a_R^2 + a_I^2), computed outward. Its component-family contribution
is B_ij c; its coefficient per unit matrix L1 norm is c/2. Diagonal probes
contribute B_ii a and coefficient a. The minimum of the component sum and
L times the largest normalized coefficient bounds each spin response. The
finite and reference bounds are added once. No AO^4 response tensor is stored.

The unscaled spin exchange-energy error is bounded by
(1/2) sum_(k,i,j) w_k B_k,i,j e_k,i,j, or by
(1/2) L max_(k,i,j) w_k e_k,i,j when smaller. Finite-map discrepancy and
reference remainder retain separate matrix and energy contributions. The
result records the RI, pair, finite reference and tail-reference identities,
probe counts, declared family and certificate identity. Strict paired matrix
and energy targets refuse before returning an unqualified result.

Count-only preflight reserves both producers, a previously attached J
certificate's retained memory, all probe evaluations, tail preparation,
interval propagation and bounded response scratch before reading vector
payloads. It conservatively uses the total reference-vector count as a
per-channel ceiling. Source tables and work are not hidden behind a dense
fallback. The result owns its bound arrays; `validate_density` checks the
family before a consumer uses the certificate.

This component bounds the exact stored-input RI map against the full-Coulomb
exchange reference at fixed density and mesh, with q=0 omitted. Probe
arithmetic is included; integral/Fourier/source construction, factor and
rank perturbations, restricted density embedding and SCF relaxation are not.
An executed K response adds its own Section 5.44 arithmetic radius. Method
coefficients, BvK corrections and energy/Fock assembly retain their separate
owners. Physical range kernels and XC remain unsupported by this certificate.
Automatic complete-method attachment and useful selection remain #626/#627;
this does not close #623-#627 or the full #617 matrix, change source-qualified
symmetry, or establish BIPOLE/CRYSTAL parity. Existing #643 failures remain.


### 5.46 Preconstruction exchange-certificate admission

`plan_periodic_ri_exchange_fitting_capacity` consumes the shared RI capacity
plan, basis and reference image/vector counts before either reference-pair
construction or numerical vector payload access. The original pair-source
planner supplies maximum entries and retained bytes; the shared exchange-tail
planner uses those capacities. The certificate aggregation is the same code
used by the realized-owner planner. Capacity admission additionally includes
reference-pair preparation work and peak storage, and explicitly reserves any
retained J-fitting certificate. The enclosing method still charges RI setup.
All five resource caps and checked integer products apply before construction.

A capacity plan is not a scientific source identity or execution permission.
The actual certificate still verifies orbital basis, cell, momenta, weights,
full-Coulomb K ownership, reference completion and the Hermitian density family.
A capacity can exceed the realized plan because screening removes pair entries;
no capacity receipt certifies a physical matrix or relaxed energy. This seam is
required for automatic complete-method admission; its existence alone does not
attach the Section 5.45 certificate to SCF or supply useful accuracy targets.

### 5.47 Pre-SCF fitted exchange attachment

Periodic RI RHF/UHF accept optional `PeriodicRIExchangeAccuracyOptions` through
`exchange_fitting_accuracy`, or public
`PeriodicFittingOptions.full_coulomb_exchange_accuracy`. Both positive targets
refer to the raw full-Coulomb K matrix and its spin-summed exchange energy,
with the reciprocal zero mode omitted. They do not include BvK/source
corrections, other Fock components or nonlinear SCF relaxation.

The complete method admits the Section 5.46 capacity, including the fixed
reference image and reciprocal cubes, before allocating sources. Shared cube
enumeration retains the existing BUPO ordering. The reference uses the original
pair, finite reciprocal K and full-image/full-domain tail producers; it is not
a second Fock implementation. Reference support remains fixed during SCF.

The existing canonical overlap preparation supplies weighted entry bounds on
`(|D_alpha| + |D_beta|) / nk`. Multiplying these and the weighted L1 bound by
`nk` with outward rounding gives the component's unweighted family. Taking the
maximum of transposed entry limits preserves exact Hermitian symmetry without
narrowing the family. RHF embeds each spin as D/2 before family validation;
UHF retains independent spins. Every evaluated and terminal-confirmation density
must remain within the admitted family. No observed SCF density chooses the
certificate envelope or fitting projector.

The certificate is built against the actual method-owned RI source after any
admitted J fitting setup and before the first SCF callback. Unmet targets refuse
before a result is produced. The same certificate and source identity are
retained in output. Dry-run admits counts without claiming numerical acceptance.
Every executed K response requests its separate arithmetic radius. That radius
and evaluated assembly arithmetic are reported separately from the raw component
targets; they must be composed before claiming a computed Fock-error tolerance.
Input/factor construction, restricted embedding, rank perturbation and relaxation
retain their previous exclusions. Source preparation counters conservatively
charge the admitted certificate work, including reference preparation.

Joint J/K policies retain separate projectors, density conventions, certificates
and reservations. The enclosing method admits simultaneous lifetime costs once;
child source plans remain local so certificate composition does not count the
parent's entire allocation as another child owner. Assembly and BvK arithmetic
workspace/work are charged when K accuracy activates their audits. There is no
change to the shared orbital eigensolver, Fock state, symmetry permission or
physical exchange convention. Section 5.48 extends this attachment to global
hybrids; physical range and XC accuracy remain separate work.

### 5.48 Global-hybrid fitted exchange attachment

Periodic RI global-hybrid RKS/UKS consume the same Section 5.47 helper and actual
method-owned `PeriodicRIFockBuilder`. The existing canonical overlap family is
converted by the shared helper; restricted KS embeds the total density as two
D/2 spin blocks and unrestricted KS preserves both spin blocks. Every evaluated
and terminal-confirmation density is validated against that fixed family before
requesting the existing fitted K response and its separate arithmetic audit.

Targets and qualification retain the raw full-Coulomb K convention, with q=0
omitted. For physical exchange coefficient a, output additionally records outward
bounds |a| times the raw matrix and spin-summed exchange-energy bounds. This
scaling applies to signed coefficients admitted by the existing functional API.
It does not include coefficient multiplication or assembly arithmetic, BvK/source
corrections, source/factor construction, restricted embedding, rank perturbation,
XC, other Fock components or SCF relaxation. Those retain their separate ledgers
and exclusions. A global-hybrid component bound is not a total Fock certificate.

The enclosing KS plan reserves certificate construction, simultaneous J evidence,
canonical family, response/assembly/correction audits and repeated density checks
before constructing owners. Child plans consume local reservations after global
admission; they do not recursively add the entire enclosing method allocation.
Exact-J/fitted-K and fitted-J/fitted-K use existing source admission. Pure KS,
exact K and physical range-separated KS refuse this optional certificate. A
count-only dry run still supplies no numerical acceptance.

No new Fock, eigensolver, XC, fitting or symmetry implementation is introduced.
Source identities, retained projectors, physical coefficients, source corrections
and terminal evaluated states stay with their shared owners. Useful automatic
bounds, physical range and XC remain requirements under #626/#627; this optional
attachment does not enlarge the qualified 0.18 controls.


### 5.49 Physical range exchange component enclosures

The shared `PeriodicExchangeBoundSource` accepts `kernel="erf"` or `"erfc"`
and a finite nonnegative `omega`, in addition to its unchanged Coulomb default.
This extends the Section 5.13 absolute image/reciprocal enclosure, not the
physical method or fitting projector. For nonzero Q,

$$0\le v_{\mathrm{erf},\omega}(Q),v_{\mathrm{erfc},\omega}(Q)
\le 4\pi/Q^2.$$

The existing Coulomb Gaussian majorant therefore bounds every retained
nonzero-Q image error and every omitted nonzero-Q contribution for either
physical range kernel. It deliberately gives up their extra range decay.
At positive range, erfc has the finite zero coefficient $\pi/\omega^2$;
include it in the retained coefficient sum when that Q is supplied and in
the omitted coefficient sum otherwise. The same conservative cell-volume
bound applies. This makes a missing screened zero explicit in the error
ledger instead of silently identifying it with the singular Coulomb zero.
Coulomb, erf and zero-range erfc still refuse a supplied Q=0. At zero range,
erfc uses the Coulomb nonzero enclosure, while erf vanishes identically.

The Fourier convention is Sun et al. (2017), Section II.A; singular versus
screened zero-mode treatment follows Sundararaman and Arias (2013), Section
II.A and the physical range boundary above. The new inequality and placement
of the finite zero are enclosure constructions, not claims from those papers
that an arbitrary truncated RI residual is positive. No range-dependent
Madelung prescription is introduced. Every certificate identifies its actual
physical kernel, range, supports, source and density convention. Default
Coulomb source/certificate identities and arithmetic remain unchanged.

`certify_periodic_ri_exchange_fitting` now consumes a same-frame physical
Coulomb, erf or erfc K owner and invokes that owner's existing exchange action
and the matching shared direct reciprocal response for each Hermitian probe.
Both responses retain their separate probe arithmetic enclosures. The shared
range tail enclosure supplies the remainder to the full physical interaction
on the fixed mesh. The fixed Hermitian spin-family aggregation is unchanged;
no Coulomb projector or variational residual is substituted for a range fit.
The existing fitted-source guard still requires strictly positive physical range;
zero-range limits above apply to the tail source, not to a newly admitted RI fit.

Count-only admission remains a conservative common capacity, including all
explicit reference vectors and finite zeros, retained J evidence where present,
probe work and output storage. Actual construction still validates kernel,
range, frame, reversal and support. This raw component API does not attach
range bounds to a complete KS method or combine signed functional terms.
Automatic method attachment and useful bounds remain required under #626/#627.
Source construction, factor/rank perturbation, XC, method correction/assembly
and relaxed-SCF errors retain their exclusions. Shared Fock/eigensolver,
PBC/BIPOLE/CRYSTAL and symmetry contracts remain authoritative.


### 5.50 Pre-SCF signed range exchange admission

Periodic fitted range-separated KS accepts the explicit optional
`range_exchange_accuracy` policy. Its `PeriodicRIExchangeAccuracyOptions`
controls reference support and two positive targets on the sum of physical
exchange components. The existing `full_coulomb_exchange_accuracy` public
control retains its raw Coulomb meaning; the policies are mutually exclusive.
Pure/global KS, HF and exact range K refuse the new range policy.

The method resolves its original signed Coulomb-plus-erfc components before
admission. Every component keeps its actual RI owner, physical range, projector,
source identity and coefficient. A single shared canonical overlap preparation
supplies the spin-density family before the first SCF call. Each component uses
the same Section 5.47 family conversion and Section 5.49 physical certificate.
Restricted KS embeds total D as two D/2 blocks; UKS retains independent spins.
Every evaluated and terminal density must remain in every admitted family.

For component coefficient $a_c$, entry bound $B_{c,ij}$ and spin-summed
exchange-energy bound $E_c$, compose with outward arithmetic:

$$B_{ij}=\sum_c |a_c|B_{c,ij},\qquad E=\sum_c |a_c|E_c.$$

Admit only when the maximum entry of B and E meet the requested targets. Raw
component certificates are built without individual target enforcement; the
combined targets are enforced before SCF. This avoids arbitrarily dividing a
method budget among physical components. Signed coefficients remain unchanged
in the computed functional. Cancellation does not reduce the certified bound.
The reference includes the finite screened zero for erfc and omits the singular
Coulomb zero. The count-only helper takes an explicit physical kernel/range so
its generated-reference capacity includes the appropriate number of vectors.

The enclosing method reserves all physical sources, component certificates,
reference preparation, retained J evidence, canonical family, composition and
serialized exports, per-density checks and applicable response/correction/
assembly audit work. After that admission, child construction uses local caps;
it does not recursively count the whole parent allocation. Exact-J/fitted-K,
joint J/K policies and multiple distinct ranges retain their existing sources.
An impossible target refuses before an output state; dry-run remains count-only.

The aggregate receipt binds ordered component certificate identities and signed
coefficients, exports their separate raw evidence and reports the outward sum.
Each actual fitted response requests its separate arithmetic enclosure. Target
bounds exclude response/assembly arithmetic, fixed-Madelung/source corrections,
source/factor construction, restricted embedding, rank perturbation, XC and
other Fock components, and nonlinear SCF relaxation. The resolved long-range
coefficient alone continues to scale the original singular correction.

This is automatic admission of the physical exchange-component sum, not useful
whole-method accuracy or a new Fock/eigensolver. The existing method, PBC/BIPOLE/
CRYSTAL and source-qualified symmetry owners remain authoritative. Useful bounds,
XC and the remaining error ledgers/matrix requirements stay open under #626/#627
and their existing dependencies. The bounded #557 release controls do not change.


### 5.51 Positive Fourier-decay groups for exchange references

The optional `fourier_decay_bins` control partitions the existing positive
primitive Fourier envelopes of Section 5.13. It does not partition or replace
the physical integral, fitted metric, Bloch source or evaluated Fock. One bin
is the original algorithm and preserves its values and identities. The allowed
range is one through eight; unsupported counts refuse before reference payloads.
Both raw exchange-bound sources and RI exchange accuracy policies reuse this
native producer, including signed range-separated method admission.

Let M be the largest orbital primitive exponent, including any zero-coefficient
primitive, and let P=a+b for a primitive pair. Assign the term to the largest
j below the requested bin count for which P <= 2M/2^j. The first bin includes
all pairs that cannot move to a later bin. Empty bins contribute zero. The
original absolute coefficients, angular polynomial envelope, normalization,
image membership and image-tail proof remain unchanged. For bin j choose a
guarded lower decay alpha_j <= 2^j/(16M), strengthened to 2^j/(8M) when the
entire source is s-type. These are the same contour-shift inequalities from
Sun2017 II.A and Section 5.13, with a tighter maximum P in each positive sum.
This grouping and its cross-term bound are a project derivation.

Let R_j and M_j bound retained and omitted image amplitudes in bin j, and
A_j=R_j+M_j. The retained and omitted reciprocal sums c_jl and h_jl use decay
alpha_j+alpha_l, with the same physical kernel, transfer, weights and supports
as before. The image error is bounded by the sum over j,l of

$$c_{jl}[R_j^T|D|M_l+M_j^T|D|R_l+M_j^T|D|M_l],$$

and the reciprocal error by the sum of

$$h_{jl}A_j^T|D|A_l.$$

Each term is nonnegative; signed primitive contractions are already enclosed
by their absolute coefficients. No cancellation or positivity of a truncated
RI residual is assumed. Finite erfc zero is assigned by the original support;
its value is independent of the decay. The native implementation retains the
original bound and takes the smaller valid bound separately for image and
reciprocal errors. Thus optional grouping cannot enlarge reported bounds.
The legacy diagnostic amplitude/reciprocal arrays continue to describe the
valid original global envelope. The plan records the bin count; the bound
source identity additionally binds every group decay, amplitude and reciprocal
coefficient. Physical pair, RI and finite-reference identities are unchanged.

Count-only admission charges original plus cross-bin work, with at most
1+B^2 work units per original envelope unit. It reserves additional group
matrices, reciprocal tables, log-domain intermediates and result copies.
Existing owners and backend margin are retained once, not multiplied by the
bin count. Actual budget limits remain authoritative. Every RI certificate
continues to enclose the same canonical Hermitian family and report finite-fit
and reciprocal/image remainder contributions separately. Method targets,
response arithmetic, correction, source/rank/XC and relaxation exclusions keep
their existing meaning. This sharper reference bound does not establish useful
automatic whole-method accuracy or complete the remaining capability matrix.

### 5.52 Fitted-exchange contribution to the fixed-density HF ledger

The optional RHF/UHF component ledger consumes the existing Section 5.47
fitted-exchange energy enclosure at unit HF exchange coefficient. For the
same canonical density family, kinetic, nuclear, Coulomb and exchange
energy discrepancies are disjoint terms; the triangle inequality permits
an outward sum of their stored nonnegative bounds. The original spin-summed
exchange bound already includes its one-half trace convention. No extra
restricted factor or reweighting is introduced. Simultaneous direct and
fitted exchange certificates claim the same component and refuse composition.

Automatic attachment exports `canonical_density_envelope_identity` for the
original weighted entry/L1 envelope, using the existing kinetic/nuclear/J
byte convention before the outward conversion to unweighted spin limits.
The raw probe API leaves this identity empty. The ledger requires the
actual whole-mesh RI source, the canonical-family identity shared by other
requested canonical components, and the literal zero-mode-omitted Coulomb
qualification. Missing or incompatible identities, invalid bounds,
unqualified receipts and unsupported method scaling refuse the combined
ledger. The existing numerical certificate digest is unchanged. Identity
construction is charged before execution: 512 additional modeled bytes,
4096*(nk*nAO*nAO+8) scalar terms and eight times that kernel allowance.

This composes the previously accepted Sun2017/Neese2025 response-map and
spin-family contracts of Sections 5.45-5.47; it does not extend their proof.
The reference is full Coulomb K with its reciprocal zero mode omitted.
Source/BvK corrections, executed response/assembly arithmetic, restricted
embedding arithmetic, source construction, factor/rank and overlap
perturbations, basis/BZ errors, XC and nonlinear relaxation remain excluded.
The closed-shell relaxation delimiter remains an uncertified scale estimate.
Section 5.53 extends aggregation to KS; useful automatic whole-method accuracy
remains subsequent work; no new Fock, eigensolver, physical source or symmetry owner
is created.


### 5.53 KS fixed-density exchange composition

The existing Sections 5.48 and 5.50 certificates may enter the same fixed-density
component ledger as Section 5.52. For global-hybrid coefficient a and raw
spin-summed exchange energy enclosure b, |a|b encloses the method component
error. Binary64 multiplication is rounded outward, with exact zero preserved;
an unrepresentable finite result refuses qualification. The raw certificate
already contains its trace/spin convention, so no further restricted factor
is applied. Its signed coefficient must match the actual method receipt and
its original canonical family must match the other canonical components.

For physical range components, Section 5.50 already encloses the signed sum
by summing |a_i|b_i outward. Composition validates the ordered complete physical
kernel/range/coefficient list against the functional requirements and each
actual RI owner. Every child must identify the original canonical family,
be qualified for the stated physical reference and have no standalone target
claim. The native aggregate must enclose the outward energy sum; composition
then adds that aggregate once, never its children. A competing direct/global
exchange claim, a missing physical component or incompatible source/family
refuses the ledger. An erf/erfc child must retain a strictly positive physical
range; the screened zero convention stays with its existing producer.

This is triangle-inequality composition of the previously accepted
Sun2017/Neese2025 and Sundararaman/Arias2013 physical-response contracts, not
an extension of their component proofs. Scaling and scalar summation enclose
stored bound arithmetic only. Executed coefficient multiplication, response,
assembly and restricted embedding arithmetic, source construction, BvK/source
corrections, rank/overlap perturbations, XC/grid, basis/BZ and relaxation retain
their exclusions. KS receives no relaxation delimiter estimate. The existing
shared Fock, eigensolver, source and symmetry APIs remain authoritative; no
new provider or quadrature is created.

### 5.54 Conditional overlap rank under omitted images

`PeriodicOneElectronAccuracyOptions.overlap_image_rank` is an optional
pre-SCF guard for periodic RI HF and KS. It consumes the existing full AO
image-overlap entry bounds, not a user-supplied perturbation estimate. The
fixed-source method refuses an unresolved guard before SCF. Physical-support
selection marks that trial unqualified and grows the AO pair support within
its existing limits. Defaults, physical kernels, fitting owners, symmetry
permissions and the shared Fock/eigensolver conventions are unchanged.

Let S be the stored finite overlap, C its actual shared canonical Hermitian
projection, and B the producer's nonnegative omitted-image entry bounds.
Outward interval subtraction encloses each entry of C-S, including imaginary
diagonals. The maximum row or column sum of B+|C-S| gives t >= ||Delta S||2
for the declared image perturbation relative to C. The implementation reuses
the bounded solver's exact projection, Jacobi, negativity and residual checks.
Section 5.31 then certifies its complete eigensystem and the strict radius r,
including movement of an adaptive relative cutoff. Every k point must have
t<r and the same retained rank as the already prepared canonical density
family. Equality, an unresolved spectrum or a rank mismatch refuses. This is
the existing Rump/Lange2023 and Weyl argument; it introduces no alternative
orbital solver or tolerance repair.

The raw `bound_periodic_overlap_image_rank` producer accepts stored overlaps,
entry bounds, the expected rank and explicit shared SCF controls. Its
`plan_periodic_overlap_image_rank` admits copies, Jacobi and interval work,
per-k certificates and output storage before inspecting payloads. Complete
methods reserve that work in their one-electron plans; Python separately
reserves per-k dictionaries and serialization. Rejected support trials retain
their charges. Typed receipts record per-k spectra, perturbation and
projection bounds, the minimum radius and an identity covering the stored
inputs and rank policy. The combined image certificate includes that identity;
selected and final sources must still match.

Qualification is conditional on the image-bound model. Gaussian integral,
phase and source-assembly arithmetic remain excluded. This does not prove
stability against all overlap errors, preserve occupied subspaces or certify
the density family after changing the overlap. It does not bound overlap
contributions to an energy, orbital response, density relaxation or total
relaxed energy. Global ledger exclusions therefore remain in place. Source
perturbations of the auxiliary metric and rigorous construction arithmetic
remain under #623/#625. MANTA remains experimental.


### 5.55 Conditional fitting metric rank under assembly roundoff

`PeriodicFittingOptions.metric_assembly_rank=True` enables an optional
pre-SCF guard in the existing periodic RI transfer owners. The GDF tile
producer retains the same short-range integrals, phases, Fourier panels,
weights, compensated accumulation and physical source identity. Alongside
that route, shared outward complex intervals enclose the exact sum of stored
short-range phase/integral products and stored weighted Fourier products,
including finite-zero subtraction. The interval sum spans the short-range
to long-range compensation reset. Complex L1 distances from the final stored
entries give nonnegative per-entry error radii. This uses the Rump2010
verified-arithmetic principles already used by the shared spectral producers;
it does not treat evaluated transcendental or Gaussian operands as exact
physical integrals.

Both matrices in the perturbation argument use the established RI lower
triangle and real diagonal. The upper triangle is its conjugate, rather than
an average of independently assembled entries. Symmetrizing the lower-triangle
radii and summing outward gives t >= ||Delta M||2. The existing spectral
certificate from Sections 5.31-5.32 must resolve the same retained rank as the
actual factor, with t strictly below its rank-stability radius. This includes
movement of the adaptive relative cutoff under the existing Rump/Lange2023
and Weyl argument. Equality or an unresolved certificate refuses before SCF;
no cutoff adjustment, alternate factor or fallback repairs the request.

The tile receipt exports entry radii, their maximum and an identity covering
the physical source, stored matrix and radii. Every RI metric receipt exports
the requested/qualified flags, perturbation bound and certificate identity.
Source and factor identities remain unchanged. Optional source planning
charges interval arrays, output copies, lower-triangle norm workspace and
conservative audit work; Python reserves additional per-channel provenance
storage. Dense raw GDF APIs refuse this option because they do not implement
its audit. The tile producer alone reports assembly radii; rank qualification
belongs to its RI consumer. Default behavior remains unchanged.

The proof is conditional on stored primitive integrals, Fourier transforms,
phases and kernel weights. Their construction, contractions and pure
transforms, phase/volume/weight evaluation, source truncation and screening
remain excluded. Rank stability does not certify retained-space rotation,
inverse response, Fock/energy assembly, density relaxation or total relaxed
energy. This increment adds no general bulk CRYSTAL parity, speedup or
production-scale qualification. Global ledger exclusions remain; #623/#625
and the broader #617 programme stay open. MANTA remains experimental.


### 5.56 Finite inverse response to conditional metric assembly error

The opaque shared `HermitianInverseCertificate` now provides
`bound_perturbation(t, budget)`. Given its actual stored Hermitian matrix A,
it bounds the difference between the exact retained spectral functions of
A and B=A+E for any Hermitian E with ||E||2 <= t. Both spectra use the same
absolute/relative cutoff policy. The existing spectrum proof must establish
t strictly below its rank-stability radius; an unresolved or crossed cutoff
returns an unqualified result. Nonfinite/negative inputs and unrepresentable
arithmetic refuse. No eigensolve, factor change or dense inverse is added.

Let alpha be the verified lower retained eigenvalue of A, g its verified
retained/discarded gap, and beta=alpha-t. Weyl gives a lower retained value
beta for B and a mixed-space gap at least h=g-t. All bounds expand outward.
For nonempty rank require beta>0; partial rank additionally requires h>0.
The inverse divided difference is bounded by
`L=max(1/(alpha beta),1/(beta h))`, omitting the second term at full rank.
A power-of-two ceiling for sqrt(n) gives T >= ||E||F from t. The finite
intertwining identity in exact unitary eigenbases U of B and V of A is
`[U*(f(B)-f(A))V]ij = f[b_i,a_j] [U*E V]ij`.
Consequently `||f(B)-f(A)||2 <= L T`. The inverse function is 1/x on each
retained spectrum and zero on its discarded spectrum. Repeated eigenvalues
within either space do not require an eigenvector orientation or internal gap.
This is the same Higham2008 Chapter 3 divided-difference argument as Section
5.33, extended to two exact spectra; no first-order remainder is discarded.

For the exact retained projectors, the mixed divided difference is bounded
by 1/h, giving `||P(B)-P(A)||2 <= min(1,T/h)`. Empty and full retained spaces
have zero projector response. Empty rank also has zero inverse response.
This projector bound concerns exact spectral projectors, not the rounding
of the exported numerical projector. Adding the existing Section 5.33
stored-factor inverse error outward gives a bound on `||F F*-f(B)||2` for
the actual stored F. It is a bound on its exact matrix product, not an
executed Fock, J, K or energy contraction.

The scalar plan reserves 4096 bytes and conservative 512 scalar/4096 kernel
operations before inspecting t; periodic RI plans include that workspace
and work. Python separately reserves the per-channel result dictionaries
and serialized copies. There is no dimension-sized new allocation or second
spectrum audit. Public certificates remain immutable and carry the producer's
dimension, rank, retained minimum and gap metadata.

With `metric_assembly_rank=True`, each actual RI transfer owner feeds its
Section 5.55 assembly norm bound into this shared operation. The resulting
`assembly_response` receipt records qualification, t, T, beta, h, L and
projector, inverse and stored-factor-plus-source bounds. Its audit identity
is included in the assembly receipt; physical source, factor and evaluated
state identities remain unchanged. Rank admission keeps its original rule.
A response can remain unqualified even when rank is resolved, and its zero
fields then mean unavailable rather than zero error. The response is a
conditional diagnostic, not a new automatic acceptance target.

The input t still excludes primitive integral, contraction/pure-transform,
Fourier, phase and kernel-weight evaluation, source domains and screening.
The new inverse response does not cover coupling perturbations, propagation
to J/K/Fock/energy, overlap or occupied-space changes, density relaxation or
total relaxed energy. Global ledger exclusions stay in place. No general
parity or speedup follows. MANTA remains experimental and #623/#625/#617
retain their remaining obligations.


### 5.57 Conditional metric-source contribution to fixed-density J and energy

`PeriodicRIFittingAccuracyOptions.metric_action_include_assembly` is an
explicit opt-in for the existing metric-action J/energy targets. It requires
positive metric-action targets and `PeriodicFittingOptions.metric_assembly_rank`.
The default remains false. The raw fitting-bound options expose the same
choice. A missing source proof or an unqualified finite inverse response
cannot satisfy the action target. Existing selected-source and search owners
retain their physical sources, rank rules, density families and budget limits.
An unresolved assembly rank still refuses at its existing source guard; this
option does not add a recovery path across a rank boundary.

Let C(A) and C(B) be the exact truncated inverses of the stored canonical
metric and its conditional exact stored-operand assembly reference. Section
5.56 gives eta >= ||C(B)-C(A)||2. For the declared stored coupling columns t_p,
let n_p >= ||t_p||2. Then the fixed coupling response obeys
`|Delta K_pq| <= eta n_p n_q`. For the existing weighted spin-total density
family `|d_p|<=h_p`, `sum |d_p|<=L`, use the shared positive support function
`s(v)=sup sum v_p |d_p|`. The source contribution is bounded entrywise by
`J_p <= eta n_p s(n)` and in energy by `E <= 0.5 s(J)`.
All products, sums and support evaluations round outward. This reuses the
Section 5.34 norm-family producer, including its bounded dual support solve;
it introduces no independent density weighting or Coulomb convention.

`HermitianInversePerturbationCertificate.norm_family` exposes this source-only
contribution. It uses `inverse_error_bound`, not the already combined
`factor_inverse_error_bound`. The periodic consumer adds its entries and
energy outward to the existing stored-factor action certificate, which may
use a sharper residual proof. Thus the stored-factor error is counted once.
The combined action target compares the actual stored-factor map with the
conditional assembled-metric inverse, at fixed coupling and density.

Explicit sources reuse the admitted column audit. Supported compressed and
pair-tree sources reuse their existing metric-column norm producer, without
materializing a full auxiliary/AO-pair tensor or changing the graph.
Planning adds one norm-family contraction, source/combined entry arrays,
identity work and output storage. Python includes the added trial metadata
in its export reservations. No additional primitive source traversal is
needed solely to propagate the metric perturbation.

Receipts distinguish `stored_j_bound`, `stored_energy_bound`,
`assembly_j_bound`, `assembly_energy_bound`, requested/qualified flags and
an assembly-action identity. That identity binds the actual RI source,
assembly response, coupling norms and fixed density limits. Existing combined
`j_bound`/`energy_bound` drive admission in every existing fitting selection
loop. The raw API exports both combined and source-only per-entry bounds.
The numerical J, Fock, factor and physical source remain unchanged.

This is an additional metric-action target, not a replacement for or a sum
with the separate full-Coulomb fitting certificate. The global component
ledger is unchanged because those reference objects need separate accounting.
The source contribution excludes primitive/Fourier/phase/weight construction,
metric-domain and screening errors, coupling construction/perturbations,
executed J/Fock/energy arithmetic, K response, overlap/occupied-space changes
and nonlinear relaxation. No total relaxed-energy, bulk parity or speedup
claim follows. MANTA remains experimental; #623/#625/#627/#617 stay open.


### 5.58 Conditional metric-source response of ordered exchange channels

The raw `PeriodicRIExchangeFittingOptions` and complete-method
`PeriodicRIExchangeAccuracyOptions` expose paired optional
`metric_assembly_matrix_error` / `metric_assembly_energy_error` targets.
Zero disables this separate component; positive targets require the existing
`metric_assembly_rank` proof on every actual K channel. Invalid, missing or
unqualified proofs refuse. These are raw physical-component targets, including
when used by a signed range method: each child must satisfy them separately.
They are not targets on the signed aggregate or on relaxed energy.

For target k and source l, retain the actual ordered q owner and its exact
truncated inverse response eta_kl from Section 5.56. Gamma supplies diagonal
channels, with the corresponding k-dependent coupling columns; off-diagonal
channels retain their individual source, metric and range kernel. The
existing coupling norm producer supplies n_kl(mu,a) >= ||T_kl(:,mu,a)||2.
It may use a conservative complex/auxiliary L1 envelope. For each spin,
the fixed-source map is the existing `B D B*` contraction (Sun2017, q-dependent
fitting; Section 5.44). The operator-norm inequality gives

```
c_kl(mu,nu,a,b) = w_l eta_kl n_kl(mu,a) n_kl(nu,b)
|Delta K_sigma(k,mu,nu)| <= sum_l,a,b c_kl(mu,nu,a,b) |D_sigma(l,a,b)|.
```

With unweighted spin-absolute density caps h and total L, each output bound
is the minimum of `L max_l,a,b c` and `sum_l,a,b h(l,a,b) c` when entry caps
are present; otherwise use the first expression. The spin-summed energy
bound is the minimum of `0.5 L max_k,mu,nu(w_k bound_k,mu,nu)` and
`0.5 sum_k,mu,nu h(k,mu,nu) w_k bound_k,mu,nu`. Symmetric entry caps justify
the trace transpose. All elementary products and sums round outward; zero
source weights contribute zero. There is no extra restricted-spin factor.
This encloses all Hermitian densities in the declared family, including
indefinite ones, with positive, negative or zero perturbation responses.

The producer streams one norm pass per actual metric owner. Gamma norms
are retained once; pair norms and coefficient loops are streamed without
an auxiliary-by-pair or four-index coefficient tensor. Plans charge all
additional source passes, norms, contraction/hash work and receipt/export
storage before reading payloads. The separate identity binds actual RI and
assembly identities, every norm, weights, family limits and returned bounds.

Receipts export `metric_assembly_requested`, `metric_assembly_qualified`,
`metric_assembly_matrix_bound`, `metric_assembly_energy_bound`, per-entry
bounds, channel count and identity. Complete-method provenance records them
under the raw exchange component's `metric_assembly` object. Existing
fitting certificates, signed aggregate, physical factors, source/state
identities and global ledger remain unchanged. Do not add this response to
those unlike reference objects without a separate composition proof.

The response compares exact C(A+E) and C(A) at fixed stored T, so stored-factor
error is explicitly excluded rather than counted again. Primitive/Boys,
contraction/pure transforms, Fourier/phase/weight evaluation, coupling-source
construction, metric screening/domain perturbations, executed K/Fock/energy
arithmetic, BvK correction, XC/grid, overlap/occupied-space response, basis/BZ
and relaxed density remain outside this component. Physical erf/erfc owners
keep their existing finite/singular zero conventions. This grants no new
symmetry reduction, total-energy accuracy, bulk parity or speedup. MANTA
remains experimental and the broader #617 scope stays open.


### 5.59 Exact stored-vector Bloch phase in metric construction

The existing `metric_assembly_rank` audit now includes evaluation of the
short-range metric Bloch phase `exp(i q.R)`, including the dot product, for
exact stored binary64 vectors q and R. This replaces the stored-phase
premise of Section 5.55 for that term only. It preserves the actual numerical
phase, stored metric, factor, physical support and evaluated SCF state.
Fourier-panel phases, coupling phases, integral evaluation and construction
of q/R remain outside this increment.

The shared `certify_bloch_phase(momentum, translation, evaluated_phase,
budget)` returns an immutable rectangular enclosure of the exact phase
and an absolute complex-L1 error bound for any finite supplied witness.
It makes no correctness assumption about the witness's math library.
First enclose all three products and their sum using the existing outward
binary64 interval operations. Reject nonfinite inputs or unrepresentable
intermediate endpoints. Scale this dot interval by an exact power of two,
x = angle / 2^s, chosen to put its magnitude at about 1/16 or below.
Zero is kept exact. There are at most 1028 squarings for a finite dot interval.

Compute degree-24 cosine and degree-25 sine Taylor polynomials with interval
recurrences and outward division by exact positive integer denominators.
Taylor's theorem applied at degrees 25/26 bounds the omitted terms by
`|x|^26/26!` and `|x|^27/27!` respectively, because every real sine/cosine
derivative has magnitude at most one. Bound those remainders from the
next term's interval and enlarge both components. This is a finite
Taylor/Lagrange proof, not an empirical error estimate or an assumed libm
ULP bound. No pi reduction or transcendental function is used by the proof.
Square the complex enclosure s times, intersecting each result with
`[-1,1] + i[-1,1]`, which contains every exact unit-modulus phase. For very
large or uncertain angles this can become a broad enclosure; it never
silently becomes a precision claim. Compare the actual stored phase with
that rectangle to bound its error. Exact point/witness equality has zero
error, including Gamma.

This reuses the verified arithmetic conventions of Rump2010 and the Bloch
phase convention of Sun2017 Eq.16; the Taylor remainder derivation above
states the additional assumptions explicitly. The shared diagnostic is
count-admitted with 4096 bytes plus external/backend storage, 262144 scalar
and 2097152 kernel operations. The periodic metric planner charges that
worst-case work for every possible short-range shell-image phase and adds
workspace before evaluating any payload. The proof uses round-to-nearest
binary64 and gradual underflow, checked by the existing arithmetic guard.

The actual short-range metric assembly multiplies its stored integral by
the exact-phase enclosure. It then follows the unchanged interval sum and
long-range/finite-zero boundary. The tile receipt records
`metric_phase_included`, `maximum_metric_phase_error`, and a versioned
assembly certificate identity. The RI metric receipt carries
`assembly_phase_included`, exported as `assembly_rank.phase_included`.
The existing canonical lower-triangle perturbation norm feeds strict rank
admission and finite inverse/projector response; the optional J and raw K
component targets automatically consume this expanded proof. No new
reference object, eigensolver, rank repair or signed/global ledger sum is
introduced. The overall source-construction flag remains false because
other construction terms are still missing.

Independent validation uses 800-digit AGM pi with argument reduction and
scalar sine/cosine series, rather than binary phase squaring, and reconstructs
signed contracted short-range metrics from stored integral tiles with those
phases. This proves the new phase contribution, not the integrals themselves.
Primitive/Boys/radial/contraction/pure-transform evaluation, Fourier and
coupling phases, kernel weights, q/R generation, screening and omitted domains,
XC/grid, overlap/occupied-space/density response and total relaxed energy
remain separate. MANTA stays experimental; #623/#625/#617 remain open.


### 5.60 Reciprocal metric weight construction from stored prefactor

The optional `metric_assembly_rank` proof includes reciprocal metric weight
construction, in addition to Section 5.59's short-range phase. Its reference
uses exact stored momentum components, numerical range eta, physical range
omega and reciprocal prefactor C. C is the actual stored `4*pi/volume`
value; evaluating pi, the cell determinant and C remains excluded. Stored
auxiliary Fourier panels remain conditional operands. No numerical weight,
physical support, source identity, factor or SCF operation is replaced.

For squared momentum t and the existing nonsingular branch, the reference
weights are `C/t * exp(-t/(4*eta^2))` for Coulomb,
`C/t * exp(-t/(4*omega^2))` for erf, and their difference for erfc.
The finite-zero correction is `-C/(4*eta^2)` for Coulomb and
`C/4 * (1/omega^2 - 1/eta^2)` for erfc. Erf has no finite-zero correction.
The physical erfc finite zero mode and numerical SR subtraction are thus
still included once. Equal stored ranges give an exact zero erfc reference,
even when the evaluated finite-zero expression has a rounding residual.
These are the existing Sun2017 range decomposition and zero conventions.

The shared `certify_periodic_gdf_weight` diagnostic returns an immutable
interval and a bound for any finite supplied evaluated weight. It encloses
the sum of three stored-vector squares, range squares, divisions and final
products/subtractions using the existing outward binary64 arithmetic.
The existing singular threshold is the stored binary64 value 1e-24. An
interval straddling it refuses; a wholly singular interval has zero weight.
Selection of a transfer's Gamma/finite-zero branch remains with the source.
The finite-zero diagnostic requires a zero vector. Nonpositive denominator
intervals, nonfinite inputs and unrepresentable intermediate endpoints refuse.

For `exp(-x)`, x is a nonnegative interval. Scale by a power of two to at
most about 1/16, evaluate a degree-24 Taylor polynomial with outward integer
division, and enlarge by the order-25 Lagrange remainder. Every derivative
of exp(-x) on the nonnegative axis has magnitude at most one, so the next
term's absolute interval bounds that remainder. Square at most 1028 times,
intersecting with [0,1] after every step. There is no correctly rounded
exp/expm1 premise and no transcendental call in the proof. Exact zero
arguments remain exact. Underflow is enclosed with gradual-underflow
arithmetic; a conservative enclosure may cause a downstream target refusal.
This follows the shared Rump2010 verified-arithmetic convention, not an
empirical ULP allowance. Subtracting the two decay intervals covers either
sign of the erfc difference, including nearly equal ranges.

The weight plan reserves 4096 bytes plus external/backend storage and
262144 scalar / 2097152 kernel operations for both possible exponentials.
The metric planner charges each reciprocal vector and the possible zero
term, plus interval panel storage, before payload evaluation. The unchanged
numerical weights multiply the unchanged Fourier panels; separately, the
ideal assembly uses the enclosed weights. Version-3 assembly identities,
`metric_weight_included`, `maximum_metric_weight_error` and
`assembly_rank.weight_included` identify this extension. The canonical
lower-triangle rank/inverse producers and optional J/raw-K component targets
consume it without adding a new eigensolver, Fock path or global ledger term.

Independent Decimal exponentials and exact stored-scalar rational arithmetic
check all physical kernels, both zero conventions, equal/near-equal ranges,
underflow, large momenta and arbitrary returned witnesses. Stored-panel
Fourier Gram reconstructions and finite-zero contraction checks exercise the
actual metric producer. These certify the stated weight contribution, not
Fourier amplitudes or the primitive integral reference. Prefactor and vector
generation, primitive/Boys/radial/contraction/pure transforms, Fourier and
coupling construction, domains/screening, BvK corrections, XC/grid,
overlap/occupied-space/density relaxation and total relaxed-energy accuracy
remain separate. MANTA stays experimental and #623/#625/#617 remain open.


### 5.61 Reciprocal prefactor from the stored lattice

The optional metric assembly proof now removes Section 5.60's stored
reciprocal-prefactor premise. Its reference is `4*pi/abs(det A)` for exact
stored binary64 lattice entries A and mathematical pi. The numerical lattice
determinant, pi value, weights, source, factor and SCF state remain unchanged.
Construction of the lattice entries and reciprocal momentum vectors remains
outside the reference. This grants no new symmetry or coordinate equivalence.

The shared interval implementation encloses the three cyclic cofactor terms
of the determinant. A strictly negative determinant interval is reflected;
a strictly positive one is retained. An interval containing zero refuses,
including when a floating determinant happens to be nonzero. Overflow in
intermediate products also refuses. The proof does not repair the cell or
substitute a determinant algorithm into the numerical path.

Pi is enclosed using the identity `pi = 16*atan(1/5) - 4*atan(1/239)`.
For each exact rational argument x, enclose its first 40 alternating odd
powers `sum((-1)^j*x^(2j+1)/(2j+1), j=0..39)`. The alternating-series
remainder has magnitude at most `x^81/81`; enlarge the interval by that
quantity. The rational input divisions, powers, sums and final linear
combination all use shared outward arithmetic. The tangent addition identity
and the first-quadrant value establish the Machin identity. No libm pi,
arctangent or correctly rounded constant is assumed. This follows the
Rump2010 verified-arithmetic conventions already used by the weight proof.
Outward division of four times this pi interval by the positive volume
interval encloses the reciprocal prefactor.

`certify_periodic_gdf_lattice_weight(lattice, momentum, omega,
physical_kernel, zero, evaluated_weight, budget)` uses this enclosure in the
existing weight producer. The earlier `certify_periodic_gdf_weight` remains
a diagnostic explicitly conditional on its supplied stored prefactor.
The actual metric path constructs its prefactor enclosure once and shares
it across all reciprocal and finite-zero terms. The common weight plan
reserves 8192 bytes plus external/backend storage, 278528 scalar and 2228224
kernel operations, including both possible exponentials and prefactor proof.
The metric planner conservatively charges that work per possible weight,
plus the enlarged workspace, before payload evaluation.

Version-4 assembly identities, `metric_prefactor_included` and exported
`assembly_rank.prefactor_included` distinguish the expanded reference.
The same lower-triangle rank, finite inverse/projector and optional J/raw-K
bounds consume it. Original physical source and fitting/signed/global
reference ledgers remain unchanged; the complete-source flag remains false.
The weight proof still uses the owner's actual Gamma/finite-zero convention
and strictly refuses unresolved momentum pole classification.

Independent AGM pi and Decimal determinants validate all three physical
kernels, finite-zero terms, skew cells, orientation reversal, cancellation
and lattice scales 1e-100 through 1e100. Contracted stored-Fourier Gram and
finite-zero assembly witnesses now use the lattice-derived prefactor.
They do not establish primitive or Fourier accuracy. Lattice/vector
construction, primitive/Boys/radial/contraction/pure transforms, Fourier and
coupling construction, source domains/screening, BvK corrections, XC/grid,
basis/BZ, overlap/occupied-space/density relaxation and total relaxed-energy
accuracy remain separate. MANTA and the unfinished #617 programme remain
experimental; shared Fock/PBC/BIPOLE/symmetry ownership is unchanged.


### 5.62 Auxiliary Fourier construction enclosure

The optional metric audit now encloses the full auxiliary Fourier evaluation
for exact stored Gaussian exponents, centers, normalized contraction
coefficients and Cartesian-to-pure representation coefficients. This replaces
the stored auxiliary-panel premise in Sections 5.55-5.61. It does not cover
generation of those basis/representation data, AO-pair Fourier construction,
short-range Coulomb integrals or reciprocal-vector generation. Supported
angular order and normalization are unchanged: auxiliary Cartesian or pure
shells through l=7. No new physical provider or symmetry permission follows.

The shared `certify_auxiliary_fourier_panel(auxiliary, wave_vectors, budget)`
returns the unchanged evaluated Fourier result, a rectangular enclosure and
an absolute complex-L1 error bound for each element. Its plan admits basis,
vector, shell, primitive, scalar, kernel and storage costs; the binding admits
counts before copying vector payload. The same internal shell enclosure is
consumed by the actual metric producer, with no second numerical integral
route or replacement of the evaluated panels.

For a primitive of exponent a at center A, use the exact Fourier seed
`(pi/a)^(3/2) * exp(-q.q/(4a)) * exp(-i q.A)`. The dimensionless Cartesian
moments obey `h_0=1`, `h_1=-i q/(2a)` and
`h_n=(-i q/(2a))*h_(n-1)+(n-1)/(2a)*h_(n-2)` independently on each axis.
Integration by parts against the decaying Gaussian proves this recurrence;
its surface term vanishes. The product of three moments multiplies the
seed and stored contraction coefficient. This is the same exact Gaussian
polynomial integral as the existing MD/Hermite Fourier formulation
(Neese2023 and Sun2017), not an assumption about the rounded MD E matrix.

Shared interval routines enclose the squared norm, divisions, pi and exp.
The existing phase proof uses exact stored centers, thus also covering the
numerical producer's rounded product-center/phase arithmetic in this
single-center case. Square-root endpoints are bounded by two finite binary
searches on a power-of-two-scaled mantissa: one certifies an upper squared
endpoint below the argument, the other a lower squared endpoint above it.
Each search takes at most 64 steps, preserves its initial bracket and
returns a proven endpoint. It assumes no libm square-root accuracy.

Outward complex products and sums enclose the primitive moment recurrence,
signed contraction and application of the shared stored representation
matrix. The Cartesian identity transform does not change the reference.
Unrepresentable intermediate enclosures refuse; there is no clamping of
physical integrals or weakening of a target. The reusable pi/exp/division
proofs moved into a shared internal helper without changing their arithmetic.
Only the certificate computes these enclosures; evaluated numerical values
continue through the existing Fourier producer.

The construction plan conservatively charges per primitive/vector moment,
transcendental and contraction work, per shell/vector phase and transform
work, the existing numerical producer, exported enclosures and workspace.
The metric planner includes every possible reciprocal/finite-zero panel,
full auxiliary interval storage and the proof workspace. Version-5 metric
assembly identity, `metric_fourier_included` and
`assembly_rank.fourier_included` identify the expanded reference. The Gram
assembly uses these enclosures on both auxiliary factors with its existing
weight enclosure. Shared rank/inverse/J/raw-K consumers propagate the result;
physical source/factors/states and fitting/signed/global ledgers stay intact.

Independent 250-digit finite Gaussian-moment sums use factorial coefficients
rather than the native recurrence, AGM pi, Decimal sqrt/exp and the earlier
independent phase oracle. They cover l=0..7, Cartesian/pure shells, signed
coefficient scales, diffuse/tight primitives and deep cancellation. The
representation data are deliberately shared conditional operands; this does
not validate their construction. Existing quadrature and independent native
Fourier references remain separate checks of the physical convention.
Basis/representation and lattice/vector data generation, short-range
integral and coupling construction, screening/domains, stored-factor/executed
Fock/energy arithmetic, BvK, XC/grid, basis/BZ and overlap/occupied/density
relaxation remain outside a total relaxed-energy claim. MANTA remains
experimental; #623/#625/#617 and the broader capability matrix stay open.


### 5.63 Two-center metric integral construction enclosure

The optional periodic metric audit now encloses short-range two-center
Coulomb integral construction, in addition to Section 5.62 auxiliary Fourier
construction and the earlier phases and weights. Exact stored exponents,
normalized contraction coefficients, centers, representation coefficients,
image translations and reciprocal vectors define the reference. The audit
includes rounding of center-plus-image addition. Generation of those input
data, three-center coupling and AO-pair Fourier construction, omitted source
domains/screening and relaxation remain separate.

`plan_metric_construction` and `certify_metric_construction` expose the same
proof for full, erf and erfc two-center tiles through auxiliary l=7. The
certificate contains the existing `FittingTiles` numerical result, lower and
upper element enclosures and absolute error bounds. The numerical MD producer
is unchanged. The internal shell enclosure also feeds the actual periodic
GDF metric producer; it is not an alternative numerical integral route.

For stored primitive exponents p and q, outward arithmetic encloses
rho=p*q/(p+q), center displacement, T=rho*|A-B|^2 and the prefactor
2*pi^(5/2)/(p*q*sqrt(p+q)). Pi and square root reuse Section 5.62 proofs.
The Boys integral F_n(T)=integral_0^1 u^(2n)*exp(-T*u^2) du uses the positive
series exp(-T)/(2n+1) times the sum over k of
(2T)^k/product_(j=1..k)(2n+2j+1). The first omitted term divided by one minus
the next ratio bounds the positive tail once that ratio is below one; all
subsequent ratios decrease. At most 2048 terms are admitted for T upper
endpoint at most 512. A relative-tail comparison only chooses the stopping
index: the explicit outward remainder, not that comparison, certifies error.
Positive downward recurrence supplies lower orders.

For larger arguments the integral over [0,infinity) bounds F_n from above
by Gamma(n+1/2)/(2*T^(n+1/2)), also capped by 1/(2n+1); the lower endpoint
is zero. If the T interval includes zero, the latter elementary upper bound
alone applies. This branch is deliberately conservative and may prevent a
requested rank/action target from qualifying. It is not an asymptotic equality
or a silent accuracy fallback. No arbitrary large-T series truncation is used.

With kappa=omega^2/(omega^2+rho), erf seeds are
kappa^(n+1/2)*F_n(kappa*T). Erfc seeds are the difference from F_n(T),
intersected with nonnegative values. This follows the same physical integral
as Neese2023 Eq. (54); the evaluated producer's positive-interval quadrature
is retained. The proof can widen in deep cancellation, which remains visible
to downstream admission. Zero and infinite range limits are explicit.

Single-center Cartesian Hermite coefficients have the finite exact form
E_(l,t)=l!/[((l-t)/2)!*t!*2^l*a^((l+t)/2)] for matching parity and t<=l,
and are zero otherwise. Products over axes provide the sparse expansion.
The usual McMurchie-Davidson Coulomb derivative recurrence consumes interval
seeds (-2*rho)^n F_n(T), with the ket derivative parity sign. Outward sums
and products enclose primitive contraction, signed stored coefficients and
application of the shared stored Cartesian-to-pure matrices. This follows
McMurchieDavidson1978 and Neese2023, using Rump2010 verified arithmetic.
The proof does not certify generation of basis or pure-matrix coefficients.

Plans account for finite series, derivative cache, sparse Hermite products,
primitive/shell work, transformation and diagnostic output storage before
evaluation. Periodic planning charges every candidate metric image. Invalid
or unrepresentable intermediates and budget exhaustion refuse; no numerical
integral, cell or density is repaired. The complete-source flag stays false.

Version-6 metric assembly identity and `assembly_rank.integrals_included`
identify the enlarged construction reference. Existing shared rank/inverse
and optional J/raw-K consumers propagate the perturbation without changing
physical source/factor/state identities or original fitting/signed/global
ledgers. Independent high-precision alternating radial integrals and finite
closed Cartesian derivative sums test the proof against a different scalar
and derivative evaluation, with stored representation data conditional.
Existing libint comparisons remain independent physical-integral checks.
No total relaxed-energy bound, automatic whole-method accuracy, bulk CRYSTAL
parity, speedup, new symmetry permission or production-ready claim follows.


### 5.64 Positive short-range radial enclosure

The Section 5.63 short-range audit now encloses the positive integral
integral_theta^1 u^(2n)*exp(-T*u^2) du directly. It no longer subtracts two
Boys enclosures for finite positive erfc range. The evaluated MD quadrature
is unchanged, and the ordinary Boys enclosure remains unchanged for full/
long-range integrals and other producer certificates. Coverage, exact stored
operand assumptions and the remaining exclusions of Section 5.63 stay intact.

Let kappa=omega^2/(omega^2+rho), theta=sqrt(kappa) and
c=rho/(omega^2+rho)=1-kappa. Compute c directly rather than subtracting
rounded kappa from one. Choose a stored fraction f in (0,1] of this squared
interval and partition it into at most nine panels with squared width c*f/N.
A scalar choice approximately caps the total exponential gain at 64. Its
rounded accuracy is not an assumption: explicit outward checks admit each
panel gain at most eight, otherwise evaluation refuses. Small outward guards
on the count avoid unstable decisions at exact panel boundaries.

For a panel [a,b], its width h is evaluated as (b^2-a^2)/(a+b); this retains
very narrow intervals even when a and b round alike. The last endpoint of a
complete interval is exactly one. With u=a+h*y and y in [0,1], the integral is
h*exp(-T*b^2) times integral_0^1 (a+h*y)^(2n)*exp(Q(y)) dy, where
Q(y)=T*h*(1-y)*(a+b+h*y) is nonnegative and at most the admitted panel gain.
Expand exp(Q) in positive powers. Products of the linear factors use
nonnegative Bernstein coefficients: multiplying a degree-d polynomial by a
linear factor with endpoints L0,L1 gives coefficient i equal to
((d+1-i)/(d+1))*L0*c_i + (i/(d+1))*L1*c_(i-1).
The integral of a Bernstein polynomial is the mean of its coefficients.
Every operation and coefficient is enclosed outward.

After term k-1, the omitted positive tail is bounded by the integral of term
k divided by 1-g/(k+1), when g/(k+1)<1, where g bounds Q. All subsequent term
ratios are smaller. At most 64 terms per order and panel are admitted. The
relative-tail check only chooses a stopping index; the explicit geometric
remainder always supplies the certificate, including at the iteration cap.
There is no assumed quadrature or libm accuracy and no cancellation between
full and long-range Coulomb terms.

If f<1, the remaining squared interval is c*(1-f), and the exact remaining
width is c*(1-f)/(1+b). Its integral is bounded by that width times
exp(-T*b^2), using u^(2n)<=1. When T and b are positive, an independent upper
bound exp(-T*b^2)/(2*T*b) follows by extending the Gaussian integral to infinity
and using u/b>=1. The smaller outward upper bound is added as a nonnegative
tail interval for every order. Truncation never silently discards the tail.
Thus large T and narrow erfc intervals need neither the earlier full-Boys
subtraction nor an unqualified asymptotic equality.

The proof is a verified evaluation of Neese2023 Eq. (54), with Rump2010
outward arithmetic; the polynomial identities above give its derivation.
It adds no new physical integral source. The construction plan charges nine
panels, 64 terms per order, Bernstein products and integral/remainder work
before evaluation. Overflow, unresolved panel bounds and resource limits
refuse. Version-7 metric assembly identity identifies the sharper algorithm;
rank/inverse/J/raw-K still consume the same construction reference and bounds.
Neither the original fitting ledger nor physical source/factor/state identity
changes. Independent high-precision alternating radial integrals, half-line
Gaussian limits and finite derivatives test the proof, including severe
cancellation, large T, sub-ulp widths and panel boundaries. Coupling and
basis/vector-data construction, domains/screening and relaxed response remain
outside a total-energy or automatic whole-method accuracy claim.


### 5.65 Cartesian-to-pure coefficient construction

The shared auxiliary Fourier and two-center metric construction proofs now
include generation of the Cartesian-to-pure representation coefficients.
They no longer treat the rounded representation matrix as exact. Exact stored
normalized basis coefficients, exponents, centers and vector data remain the
reference inputs; their generation and all coupling/domain/relaxation
exclusions remain separate. The numerical MD transforms are unchanged.

`plan_representation(l, pure, budget)` admits fixed storage and work before
`certify_representation` returns the existing evaluated matrix, lower/upper
coefficient enclosures and absolute errors. Both construction proofs consume
the same internal enclosure. Supported l=0..7, Cartesian order, real row order
m=-l..l and common axial Gaussian normalization are unchanged. Cartesian
identity coefficients are exact. This is coefficient construction evidence,
not authority to omit finite operator audits or assume angular equivariance.

The proof uses the same mathematical solid harmonics as the existing
producer: [DLMF 14.30.1](https://dlmf.nist.gov/14.30.E1) and
[14.30.2](https://dlmf.nist.gov/14.30.E2), expressed through derivatives of
the Legendre polynomial. The complex phase convention is converted to real
sine/cosine components, with the Condon-Shortley sign removed and Racah rather
than unit-sphere normalization. The coefficient scale is
sqrt((2 if m>0 else 1)*(l-m)!/(l+m)!). The finite expansion of
(x+i*y)^m times the differentiated polynomial, homogenized by powers of
x^2+y^2+z^2, supplies the remaining rational coefficients. This matches the
existing convention; it does not redefine the basis.

All factorials through 14! are exactly representable integers below 2^53.
Outward multiplication, division and signed summation enclose the finite
polynomial coefficients; the verified square-root routine bounds the scale.
No libm square-root accuracy assumption enters. Products with these intervals
now replace products with point-valued stored transform entries in both the
auxiliary Fourier and two-center metric proofs. Count-only plans charge the
coefficient proof per shell/panel or metric shell pair and include storage
for the coefficient intervals. Existing work/storage limits refuse before
evaluation. No numerical source or requested accuracy target is modified.

The metric version-8 assembly reference and
`assembly_rank.representation_included` identify the expanded reference.
Existing rank/inverse/J/raw-K consumers propagate it through the shared metric
bound; original fitting/signed/global ledgers and physical state identities
remain intact. The complete-source flag stays false.

Independent witnesses generate Legendre coefficients by an exact rational
three-term recurrence, differentiate and homogenize them, then use high-precision
square roots. They avoid the producer's factorial Legendre coefficient formula.
The integral and Fourier witnesses now use these mathematical coefficients,
not the rounded native matrix. Existing SciPy spherical-harmonic comparisons
check real convention and normalization independently. None of these checks
certifies numerical angular transport, omitted operator members or geometry
mismatch in the separate symmetry machinery. Basis/vector data generation,
AO-pair Fourier/three-center coupling construction, domains/screening and
relaxed response remain outside automatic total-energy accuracy claims.


### 5.66 Orbital-pair Fourier construction component

`plan_pair_fourier_construction` and `certify_pair_fourier_panel` extend the
shared Fourier proof to products of two stored orbital Gaussians, each with
l=0..6, in pure or Cartesian form. The numerical `compute_fourier_panel` is
unchanged. The certificate returns that panel and entrywise real/imaginary
lower/upper bounds and a conservative absolute error, in `(q,bra,ket)` order.
The internal shell producer returns the same enclosures for future coupling
consumers. This is a separately admitted component diagnostic, not a complete
three-center construction certificate or a new SCF accuracy admission.

For positive exponents a,b and exact stored centers A,B, define p=a+b,
P=A+(b/p)(B-A), and rho=a*b/p. The Gaussian product theorem gives a common
seed `(pi/p)^(3/2) exp(-rho*|A-B|^2-q.q/(4*p)) exp(-i*q.P)`.
Product-center, squared-distance, exponent and prefactor arithmetic are all
enclosed. The existing phase proof now accepts an interval angle internally;
no rounded product center is treated as exact. The public point-vector phase
contract and its arithmetic are unchanged. Pi, exponential and square-root
proofs reuse Sections 5.59-5.62 without a libm accuracy assumption.

Expand each Cartesian pair about P using finite binomial products of P-A
and P-B. For each axis, normalized Fourier moments obey h0=1,
h1=-i*q/(2*p) and hn=(-i*q/(2*p))*h(n-1)+(n-1)/(2*p)*h(n-2).
Outward complex arithmetic encloses all moments through order 12, polynomial
products, signed primitive contractions and both exact mathematical pure
transforms from Section 5.65. Binomial coefficients through order six are
small exactly representable integers. Intersections with known positivity
and unit-circle bounds retain the exact reference. Overflow or unresolved
positive denominators refuse; underflow is retained by outward endpoints.

This is the Gaussian-product and Fourier convention of Neese2023 and
Sun2017 (DOI 10.1002/jcc.26942 and 10.1063/1.4998644), with Rump2010 verified
arithmetic. The explicit derivation above bounds the same finite Gaussian
integral rather than introducing a different physical source. Independent
witnesses shift a Gaussian contour and sum its finite even central moments
in high precision; they do not use the native h recurrence. Exact rational
Legendre transforms, separate high-precision phase/elementary evaluation and
existing real-space Gauss-Hermite integration check convention and values.

Count-only planning precedes vector copying and evaluation. It includes the
existing numerical panel, all certificate arrays, shell metadata, contraction
and transform workspace, at most 13 moments per axis, at most 49 binomial
products per Cartesian axis pair, and shared phase/elementary/representation
proof work. Storage, scalar, kernel, shell and primitive caps remain strict;
conservative reservations are not measured allocation or speedup claims.
Empty vector panels retain their declared shape without primitive evaluation.

The exact stored normalized basis data and wave vectors define this component
reference. Their generation, lattice/image translation, periodic Bloch sums,
three-center short-range integrals, reciprocal coupling products and sums,
zero-mode terms, screening/domain truncation and coupling action propagation
remain separate. No metric, RI-BUPO, Fock, symmetry or automatic-selection
certificate silently acquires this coverage. Existing component budgets and
complete-source flags are unchanged. This diagnostic establishes neither
total relaxed-energy accuracy, bulk CRYSTAL parity, numerical equivariance
nor production readiness for the full MANTA programme.


### 5.67 Streamed three-center integral construction

`ThreeCenterConstructionTiles(auxiliary, bra, ket, request, budget)` returns
separately owned construction certificates for individual ordered `(P,mu,nu)`
shell tiles. `plan_three_center_construction` counts the complete requested
shell sweep and its maximum live tile storage before copying basis data.
Auxiliary l=0..7 and each orbital l=0..6, mixed pure/Cartesian shells and
Coulomb, erf and erfc kernels retain the existing `FittingTiles` conventions.
Range parameters must be finite and nonnegative; full Coulomb requires zero.
No global three-index tensor is allocated. Numerical tiles and their physical
source identity are unchanged; lower/upper/error arrays have the same shape.

The reference is the exact integral of the stored normalized Gaussians.
For orbital exponents b,c at B,C, p=b+c, P=B+(c/p)(C-B), and the product decay
is exp(-b*c/p*|B-C|^2). All product-center and decay arithmetic is enclosed.
For auxiliary exponent a, rho=a*p/(a+p), displacement D=A-P and T=rho*D.D.
The primitive prefactor is `2*pi^(5/2)/(a*p*sqrt(a+p))` times the product decay
and stored signed contraction coefficients. Shared verified elementary and
representation producers cover pi, exponentials, roots and pure transforms.

The one-dimensional Hermite raising identity is
`E(i+1,j,t)=E(i,j,t-1)/(2*p)+(P-B)*E(i,j,t)+(t+1)*E(i,j,t+1)`;
raising the other orbital uses P-C. The seed E(0,0,0)=1 leaves the common decay
in the prefactor. Auxiliary coefficients use the same identity with zero
center shift and a unit partner. Outward arithmetic encloses every coefficient;
zero terms are omitted only when both interval endpoints are exactly zero.

The existing metric radial producer supplies Boys, erf and positive erfc
seed enclosures through order 19 via a thin shared kernel wrapper. Its
implementation and ordinary Boys interface are unchanged. Seeds multiplied
by `(-2*rho)^n` enter the standard Cartesian Coulomb derivative recurrence;
ket derivatives carry `(-1)^(tx+ty+tz)`. Successive interval contractions of
the Hermite kernel with orbital and auxiliary coefficients avoid a four-axis
Cartesian/Hermite loop. Three successive mathematical representation
transforms and signed primitive sums produce the final tile enclosure.
The error compares the actual numerical tile to that enclosure.

This is the McMurchieDavidson1978 / Neese2023 Gaussian-product formulation
(DOI 10.1002/jcc.26942), with the radial conventions of Sections 5.63-5.64
and Rump2010 outward verification. Independent witnesses expand the Cartesian
product polynomial about P, use closed factorial Hermite coefficients and
finite Cartesian derivative sums with high-precision alternating radial
integrals. They avoid both native raising recurrences. Existing libint
three-center comparisons check the numerical source separately. Selected
maximum-order Cartesian witnesses supplement all-angular-axis, mixed,
contracted, signed, underflow and range-limit cases.

Admission charges basis copies, numerical and proof tiles, three transform
stages, sparse Hermite tables, derivative/kernel scratch, all primitive
products, radial work and output copies. Shell, primitive, scalar, kernel
and memory caps refuse before evaluating a tile; invalid arity, indices,
angular roles, derivatives or nonfinite input refuse. Numerical and proof
scratch scale with a shell tile, not the number of global tensor entries.
Conservative reservations are not measured allocation or performance claims.

This streamed component supplies a prerequisite for complete periodic
coupling construction. It does not yet compose with pair/auxiliary Fourier,
reciprocal weights, Bloch/image sums or zero-mode terms into a coupling action
bound. Exact stored centers do not include errors in constructing translated
image centers. Normalized basis/vector generation, periodic domains/screening,
BvK, XC/grid, basis/BZ, overlap/occupied/density relaxation and total relaxed
energy remain separate. Existing metric/Fock/SCF budgets and complete-source
flags are unchanged. No symmetry reduction, bulk CRYSTAL parity, general
speedup or automatic whole-method accuracy permission follows.


The shared scalar fitting planner now explicitly rejects derivative requests.
Its capability query describes several producers and previously admitted a
first-derivative request that the scalar tile evaluator did not execute.
`DerivativeTiles` remains the supported first-derivative producer. Metric and
three-center construction diagnostics inherit this corrected refusal; ordinary
integrals and their scientific results are unchanged.

### 5.68 Finite periodic coupling construction and paired actions

`PeriodicGDFTiles.plan_coupling_construction(budget)` admits the complete
finite construction proof and one unchanged numerical coupling action.
`certify_coupling_tile(p, mu, nu, budget)` returns one owned numerical tile
and flattened real/imaginary lower/upper/error arrays in its native order.
`certify_forward(input, budget)` and `certify_adjoint(input, budget)` return
the unchanged numerical action plus the corresponding output enclosures.
Inputs are exact stored complex128 values; C/F contiguous layouts agree.
All five work/storage caps are checked before binding input copies.

The reference is the unscreened literal finite integer image set selected by
the shared sphere/capsule producer for the supplied cutoffs. Selection uses
its existing rounded geometry and finite-zero convention. For each selected
integer triple n, the proof encloses the exact product L*n and the addition
of each stored center. It then encloses the Gaussian integral on those exact
centers. This distinguishes an exact finite-set integral from an assertion
that the selected set is a mathematically complete geometric domain.

For a tile, the reference is the sum of the real-space erfc term
`sum_R exp(i*k.R) sum_T exp(-i*q.T) (P_T|mu nu_R)_erfc`, the reciprocal term
`sum_G w(G+q) conj(auxFT(P,G+q)) sum_R exp(i*k.R) pairFT(mu,nu_R,G+q)`,
and the existing finite-zero correction when applicable. Shared three-center,
orbital-pair/auxiliary Fourier, Bloch-phase, radial, representation and lattice
weight producers enclose all terms. Coulomb, physical erf and physical erfc
keep their existing decompositions and zero-mode conventions. A reciprocal
pole classification whose interval crosses the threshold refuses. No new
radial algorithm, image selector or symmetry reduction is introduced.

Exact finite-source forward and adjoint contractions are respectively
`conj(T):X` and `T*c`, including arbitrary complex, non-Hermitian inputs.
Per-output error is the outward complex L1 distance from the actual numerical
value to its reference rectangle. Therefore this single certificate includes
construction, omitted/screened/multipole terms and executed contraction error
for the admitted numerical route. It must not be added to overlapping old
approximation or roundoff budgets. Screened, explicit, compressed, global and
pair-tree actions are compared directly with the same reference, rather than
assuming their real-arithmetic maps or Gamma averaging are identical.

`source_identity` identifies the actual numerical coupling; `reference_identity`
identifies the physical finite reference and selected integer supports;
`certificate_identity` additionally binds actual values, input snapshot and
bounds. `reference_terms` counts selected SR auxiliary images and LR pair
images per reciprocal panel, not scalar operations or unique images. The
numerical receipt retains its original plan and counters; the separate
construction plan reserves the additional proof work. Only shell tiles,
Fourier panels and action vectors are retained, never the global coupling
tensor. Conservative component-plan multiplicities and explicit folding/
geometry/contraction charges bound work and live scratch. Those reservations
are not measured allocations, costs or speedup evidence.

The formulation reuses Sun et al. 2017 (DOI 10.1063/1.4998644), Ye and
Berkelbach 2021 range-separated Gaussian fitting, McMurchieDavidson1978 /
Neese2023 Gaussian products and Rump2010 verification from preceding sections.
Independent witnesses use a brute-force finite image census, exact Decimal
lattice products, closed polynomial Gaussian moments/derivatives and
high-precision scalar phase/weight evaluation. They do not consume the native
interval endpoints. Separate numerical-source and paired-action regressions
preserve the shared Fock/PBC/symmetry contracts.

This remains an experimental finite-source diagnostic. Existing SCF/Fock
admissions and metric budgets do not automatically consume it. Stored
normalized basis, lattice and reciprocal-vector generation, selection/domain
completeness and infinite tails, fitting-basis incompleteness, BvK, XC/grid,
basis/BZ, overlap/occupied/density relaxation and total relaxed energy remain
separate. No bulk CRYSTAL parity, general speedup or useful automatic
whole-method accuracy follows. Subsequent composition must account for
coupling and metric response without counting the same error twice.

### 5.69 Shared screened primitive source

MANTA's dense and streamed real-space three-center paths now use the same
`PrimitiveTripleScreen` as libint values and gradients. Previously they
shared shell/image decisions but retained different primitive sets within
contracted shells. This caused the retained-source comparison in #687 to
fail at 5.65e-11 for a requested omission budget of 0.01; it was not a new
coupling-certificate error or an unbudgeted primitive-epsilon screen.

The existing Gaussian/angular envelope of Ye and Berkelbach 2021 SI S4
assigns each image term a fraction of the shell budget, then divides that
fraction by its primitive-triple count. Each omitted primitive obeys the
same absolute envelope; all survivors retain their original signed normalized
coefficients. No fragment is renormalized. If every primitive survives, the
original contracted evaluation is retained; if none survives, no numerical
integral call is made. The selector, bounds and traversal are shared without
changing the libint implementation. Zero screening and multipole routes keep
their previous numerical paths.

Screened source identities now include `shared-budgeted-primitive-triples.v1`;
unscreened source identities and the exact finite reference from Section 5.68
are unchanged. Common and coupling-only screening both use the shared
primitive selector; the latter leaves the metric unchanged. Split survivors
can increase numerical shell invocations while reducing evaluated primitive
products, so neither fewer shell calls nor general speedup is promised.

Count admission reserves one call/transform per possible primitive triple,
selector envelopes/survivors and copied primitive-shell storage, plus phase
accumulation work. This applies before dense or streamed numerical allocation.
Actual counters report executed numerical calls and primitive products.
Existing five resource caps, source ownership, Fock/PBC/symmetry interfaces
and omission targets remain intact. Independent libint comparisons retain
their original tolerance; complete calculations exercise contracted screened
sources. These are finite-source checks, not total relaxed-energy or bulk
CRYSTAL parity claims. MANTA and the broader accuracy programme remain
experimental and unfinished.
