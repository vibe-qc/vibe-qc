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
periodic BUPO traversal, infinite-sum/Taylor error bound, Gaussian support
certificate, symmetry shortcut, complete charge ledger or nuclear self policy.

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
