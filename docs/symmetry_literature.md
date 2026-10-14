# Literature-to-code assessment, October 2026

The baseline assessment below preceded the complex Abelian increment. The
[architecture](design_shared_symmetry.md) and [status](symmetry_status.md)
remain the authority for production scope. No external QC implementation or
library data is bundled. The operator-contract and operator-consumer designs are now integrated.
Their original workstream versions and the MANTA engine/scientific designs
were initially inspected read-only; the interface mapping in the shared
symmetry design follows those contracts without duplicating implementations. The coordinated MANTA P0 libint
reference provider is now integrated for the subsequent quartet consumer.

| Literature inspected | Mathematical addition | Existing foundation / immediate consumer | Missing admission and evidence |
| --- | --- | --- | --- |
| Melega et al., [2026](https://doi.org/10.1021/acs.jpca.5c08579), published article, Sections 2.1, 2.3-2.6, 3 and 4.3; geometry supplement | CD/RI derivative reformulation, Lambda and orbital Z-vector response, symmetry-blocked density intermediates | Shared retained-auxiliary audits and correlation tensors can supply representations | No shared implementation of the relaxed CD-CCSD gradient; differentiated metric, fitting and support terms and displaced-geometry validation remain method-owned |
| Hellmann and Gauss, [2026](https://doi.org/10.1021/acs.jpca.6c00311), published article, Sections 2.3-2.7 and 3 | Resolve `E x E = A1 + A2 + E`; convert reduced/full four-index tensors for CC contractions, permutations and spin adaptation | Compact noncommuting actions and retained-space group audits; future CC residual consumer | Demonstrated closed-shell C3v CCSD pilot, not a general non-Abelian implementation; partner gauges, multiplicity spaces and recoupling are absent |
| Kitsaras and Stopkowicz, [journal DOI](https://doi.org/10.1021/acs.jpca.6c00998), inspected **[arXiv:2602.12457v1 preprint](https://arxiv.org/abs/2602.12457v1)**, Sections II A-C and III | Complex-conjugate irreps in double-coset integrals, creation/annihilation selection rules and block contractions | Complex AO/Bloch actions exist; molecular finite-ERI J/K is an immediate execution consumer | Explicit conjugate/product labels, tensor variance and source-qualified sparse execution are missing; antiunitary operations need separate admission |
| Goodlett et al., [MolSym 2024](https://doi.org/10.1063/5.0216738), accepted manuscript; representation and RHF supplements | Complete matrix projectors and consistently oriented partners, including repeated irreps; nonredundant RHF blocks | Existing character tables, molecular rotations and metric/subspace audits | Character projection alone does not resolve partners. RHF degeneracy factors assume an invariant state; neither geometry repair nor a character table qualifies arbitrary-density Fock production |
| Johansson and Veryazov, [libmsym 2017](https://doi.org/10.1186/s13321-017-0193-3), pp. 5-7 | Trace projection plus subgroup subduction and partner alignment for real spherical harmonics | Geometry and pure-shell representation construction | This is a representation reference, not a certificate of finite-source symmetry or an integral/contraction engine; its realification is insufficient for complex Abelian tensor labels |

The selected increment is now implemented as `symmetry_abelian.py`: an opt-in,
bounded complex-Abelian ERI block consumer for J/K. It inspects a complete
supplied finite tensor before omitting forbidden blocks, retains its numerical
identity and omitted norm, and accepts arbitrary complex trial densities
without state projection. Actual counts and measurements appear in the
[status](symmetry_status.md). Integral evaluation is unchanged, and no
production driver is enabled.

The next gap was integral work: the block consumer computes all ERIs before
reducing contractions. `symmetry_quartets.py` now adds real molecular shell
representatives and full angular stabilizer checks over the MANTA P0 tile
provider. It audits the complete finite source once, then requests only
representatives on repeated arbitrary-density J/K builds. This implements
the petite-list execution lesson of Dupuis and King without assuming a
symmetric state. Full C3v AO block actions are supported; irreducible-product
resolution, CC recoupling, complex integral sources and producer-side
screening proofs remain missing. Measurements separate integral calls,
unchanged contraction work, transport, setup and memory.

The subsequent opt-in native executor compiles that four-leg transport and
J/K digestion without changing its mathematical or source-admission scope.
Dupuis and King's nonunitary AO ordering is retained explicitly; real-ERI
permutations act on indices, while complex trial densities keep their
conjugate/index-variance convention. It is an execution improvement, not
new admission of complex integral kernels, antiunitary symmetry or CC
reduced products. Native/Python comparisons and the retained audit cost
are recorded in the status page.

The density-batch increment reuses each admitted petite-list tile and angular
rotation across independent spin or response densities. This follows the
linearity of the existing contraction, not a new state selection rule.
Arbitrary transition densities retain their conjugation/index variance;
source work is shared while contractions and output storage still grow
with the number of channels. The immediate consumer boundary is spin J/K
and density-response application. Producer-qualified screening, reduced CC
products and method-specific derivative response remain missing.

Selective batch targets now omit algebraically unneeded J or K contractions
and their output panels. This applies the existing density/index convention
without introducing an irrep selection rule: an unrestricted caller can
request one total-density J and two spin-resolved K targets. Source admission,
non-Abelian full-AO transport and complex trial densities are unchanged.
Producer screening, complex integral kernels and reduced CC tensor products
still require the separate qualifications described below.

The next execution increment specializes selected native contractions by
resolving member permutations and matrix addresses before scalar work. No
new mathematical symmetry is claimed. Dupuis-King AO transport, the complete
P0 source audit and arbitrary complex density conventions are retained. The
current consumer is the same opt-in quartet batch API; the method-owned
evaluated Fock-state adapter remains a separate draft. Same-source timing,
counts and memory are measured independently of the unchanged admission
cost. Periodic finite domains, source-qualified screening, complex integral
kernels, partner/multiplicity resolution and derivative acceptance remain
missing production capabilities.

## Source qualification and screening increment

| Mathematical capability | Already implemented | Immediate consumer | Still missing |
| --- | --- | --- | --- |
| Fixed omission of complete shell orbits, with an aggregate finite-tensor norm budget | Full ordered P0 source audit, inverse AO transport, representative/stabilizer schedules and arbitrary-density J/K | Stored-plan `jk_batch`, including selected total-J/spin-K channels | Scalable producer bounds, streaming certificate storage, method-owned dispatch and acceptance on other sources |
| Distinguish actual mask identity from screening-policy identity and density-specific error bounds | Immutable native producer/session and complete finite-source audit | MANTA receipts and the separate Fock evaluated-state source record | Agreed general producer schema for kernels, finite supports, auxiliary cuts and BUPO trees |

[Maurer et al. (2012)](https://doi.org/10.1063/1.3693908), Section II,
Eq. (1), gives a far-field multipole bound with extent and separation
conditions. Its QQR Eq. (2) substitutes Schwarz factors and is explicitly
an estimate, not a rigorous bound. Section II also explains why a small
individual-integral cutoff alone is not a useful total-energy error bound.
[Thompson and Ochsenfeld (2017)](https://doi.org/10.1063/1.4994190),
Eqs. (6)-(8), combines the ordinary Schwarz product and two distance-sensitive
products using their minimum. The required positive kernel/inner-product
conditions matter. Eqs. (9)-(12) instead define tighter nonrigorous CSA
estimates. Eqs. (18)-(19) multiply integral estimates by density entries and
use separate J/K thresholds; an arbitrary-density equivariance certificate
cannot silently replace that rule by a fixed-state bound.

This increment implements none of those kernel estimates. MANTA owns them.
It uses measured full-orbit norms from the already mandatory finite-source
audit and Cauchy-Schwarz contraction bounds. A fixed whole-orbit mask is
compatible with Dupuis-King transport and Davidson/Pitzer orbit reduction
without requiring an invariant electronic state. All actual members enter
the norm: normalized Cartesian actions are metric-unitary, so multiplying
one representative's Euclidean norm by the square root of its multiplicity
is not generally valid. Numerical norm and contraction roundoff remain
outside the reported exact-arithmetic bound. The measured results and
explicit unsupported domains are recorded in the [status](symmetry_status.md).

Neither this bound nor a tighter integral estimate supplies the Melega
response/gradient terms, Hellmann-Gauss reduced CC products, complex-kernel
conjugation rules, periodic finite-domain proof or BUPO tree covariance.
Those earlier representation and source qualifications remain unchanged.

## Producer-envelope increment

The next coherent increment consumes MANTA's existing full-space product
mass/potential envelopes. The new mathematics at the symmetry boundary is
`|(ab|cd)| <= min(Q_AB V_CD, Q_CD V_AB)` followed by a Frobenius bound over
all AO entries and every ordered member of an orbit. This is a direct
positive-Coulomb inequality, not the distance-dependent QQR/CSA estimates
above. Symmetry implements no new envelope kernel or BUPO hierarchy.

The existing native envelopes, immutable molecular source, Davidson-style
pair-stabilizer schedule and Dupuis-King AO transport supply the foundations.
The current consumer is compact screened molecular J/K and its unchanged
native RHF/UHF `FockBuild` bridge. Whole-orbit omission now precedes integral
reads; retained members and stabilizers still require the numerical audit.
Bounds do not certify equivariance, and this partial plan cannot reconstruct
an unmasked source. [Measured acceptance](symmetry_status.md) demonstrates
actual avoided calls on separated fragments and the lack of pruning on NH3.

Missing production capabilities remain tighter/hierarchical producer bounds,
producer-issued equivariance and finite-support qualification, a cost-aware
public dispatch policy, complex integral-source admission and the non-Abelian
partner/product/multiplicity and derivative machinery identified above.
The new molecular Q/V receipt does not qualify a periodic source, auxiliary
rank cut, local projector, gradient or stress. Roundoff remains separate from
the analytic integral bound.

## Hierarchical bound refinement

The next consumer reuses MANTA's existing native tree and error model.
Neese et al., [BUPO 2025](https://doi.org/10.1021/acs.jpca.4c07415),
Sections 2.5-2.7 and Eqs. (29)-(34), defines contracted centers, extents and
near/far separation before recursive bubbles. The paper's axial inward
walkers and pointwise threshold are not an integrated Gaussian-tail bound.
MANTA instead supplies its documented absolute polynomial envelopes and
separate compact, penetration and full-moment-tail errors; symmetry consumes
that implementation without reproducing it.

The new capability is an aggregate ordered-tensor bound for a native bubble
block, using all AO dimensions and pair/exchange multiplicities, followed by
intersection with complete Davidson/Dupuis-King quartet orbits. A generic
hierarchy need not map to itself under the group. A candidate block therefore
cannot authorize an uncovered orbit member. The immediate consumer remains
fixed-source molecular HF; geometric candidates never certify operator or
state symmetry. Full node magnitude includes the bounded monopole term and
all native errors, with Q/V fallback. This is different from using a compact
far-field remainder alone or the nonrigorous QQR/CSA estimates discussed above.

Numerical acceptance must check independent native integrals, omitted tensor
norms, arbitrary-density covariance, actual node/read/contraction counts and
complete SCFs. Hierarchical equivariance attestations, efficient dispatch,
periodic finite domains, auxiliary rank/projector transport and method-specific
derivatives remain missing production capabilities.

Orbit-local coverage reuse adds no mathematical screening capability.
It avoids repeated lookups of identical real-ERI unordered shell-pair labels
inside the existing Davidson/Dupuis-King orbit intersection. It does not
equate spatially related native bounds or require an equivariant BUPO tree.
Molecular HF remains the current consumer. Exact agreement with uncached
mask/source qualification, measured avoided queries and bounded storage are
required; periodic/complex-source and derivative admissions remain separate.

## Equations and qualifications that affect the implementation

Kitsaras and Stopkowicz's preprint Eq. (2) conjugates the character in the
projector; Eqs. (8)-(10) retain small angular rotation matrices inside the
double-coset sums. Replacing those matrices by signs is restricted to the
real Abelian case discussed there. Eqs. (16)-(21) distinguish creation from
annihilation characters, reference-state from excitation symmetry, and
transition from expectation densities. Eq. (25)'s roughly quadratic
contraction saving assumes two symmetry-selected tensors with balanced irrep
populations. Arbitrary-density J/K has only one selected operand, so that
estimate does not apply. Small blocks can be slower, as their timing discussion
and our measurements both show. Their implementation treats finite-field
complex Abelian groups of order at most eight; importing the mathematical
selection rule does not implement London orbitals or magnetic groups here.

Hellmann and Gauss's Eq. (4) resolves the C3v `E x E` product. Eqs. (5)-(8)
give normalized partner combinations, including the antisymmetric A2 and
two E partners. Eqs. (9)-(12) show why reduction followed by naive index
permutation is wrong: a resort can require omitted partner elements.
Sections 2.6-2.7 also require full/reduced conversion for spin adaptation and
different degeneracy factors in lower-order contractions. Their closed-shell
CCSD pilot uses Cs integrals, aligns E partner phases and partly hard-codes
C3v. The NH3/PH3 operation-count tables are not general wall-time guarantees.
The paper explicitly excludes interpreting realified complex C3 irreps as
this same construction. Generalization requires the representation and
recoupling contract in the [design](design_shared_symmetry.md).

Melega et al.'s Eq. (6) differentiates the RI-like representation using two
three-center derivatives and a negative metric-derivative term. Eqs. (9)-(14)
introduce the stationary CC Lagrangian and Lambda equations; Eq. (29) supplies
orbital Z-vector response. Eqs. (40)-(43) include relaxed density, overlap
(Pulay) and fitted two-electron terms. Eqs. (44)-(49) contract lower-rank
intermediates with derivative integrals, avoiding a stored full AO two-body
density. Their two-step CD chooses pivots before constructing full vectors;
this is not the eigenthreshold-based SR/LR GDF producer. Symmetry-adapted
displacements and integral derivatives are additional representations.
The method is closed-shell CD-CCSD, not an open-shell, triples or stress
implementation. A source-aware gradient must retain rank/pivot policy and
all response terms even when the energy contracts correctly.

The [Melega geometry supplement](https://zenodo.org/records/18550263) was
inspected as data: initial/final coronene
ZMATs (C24H12), initial/final hexabenzocoronene ZMATs (C42H18), and the
24-atom planar azobenzene XYZ. The ZMAT dummy center, variable references
and starred optimization variables matter. For example, coronene's R1
changes from 1.4 to approximately 1.41310473 in the supplied files. These are
optimization inputs/results, not a displacement-resolved analytic-gradient
oracle. Future reproduction must preserve declared units, basis, frozen-core
and CD thresholds from the paper; it must compare displaced energies and
gradients, rather than just compare the final coordinates. No supplement
geometry or third-party program is bundled or executed by this increment.

MolSym's main Eq. (3) uses matrix elements of each irrep to produce similarly
oriented partners across repeated copies. The representation supplement
lists complex C3 rows separately, which provides a useful independent check
of conjugate labels. The RHF supplement Eqs. (3)-(4) weight nonredundant
one-electron blocks by irrep dimension and use a per-spin density convention
(`F = H + 2J - K`). Its Table S1 measures Fock builds averaged over 50
iterations, not complete job time. Its smallest NH3 example already shows
that full symmetry need not beat C1. Our J/K adapter leaves density
normalization to the caller and never uses degeneracy weights to reconstruct
a broken-symmetry density. libmsym's pp. 5-6 explain why a trace projector
loses partner information and use subgroup subduction to recover it. Both
sources motivate explicit partner gauges, rather than more character tables.

## Connections to the established literature

* [Davidson 1975](https://doi.org/10.1063/1.430484), Eqs. (11)-(14): the
  double-coset degeneracy is `|H intersect g K g^-1|`, generally dependent
  on the representative. The Abelian simplification `|H intersect K|`
  must not migrate into a non-Abelian planner. Existing `Orbit` stabilizers
  are reused by the new capped full shell-orbit scheduler. A scalable
  pair-stabilizer iterator now avoids retaining that full ordered inventory.
  It enumerates `H_a` orbits in `G/H_b`, resolves exchange using inverse
  transporters as required after Eq. (17), and keeps all angular matrices.
  This index-space implementation does not implement the paper's complete
  symmetry-orbital integral formulas. Complete qualification visits every ordered quartet. The new Q/V route
  avoids reads only for bounded omitted orbits; hierarchical producer
  admission remains absent.
* [Pitzer 1973](https://doi.org/10.1063/1.1680474), symmetry-adapted charge
  distributions and coupling-coefficient orthogonality: resolving pair
  products is different from assigning one label to a non-Abelian pair.
  The paper explicitly notes the need for extra notation in nonsimply
  reducible groups. Modern multiplicity labels make that requirement concrete.
* [Dupuis and King 1977](https://doi.org/10.1002/qua.560110408), Eqs. (8)
  and (11): density and operator transformations have opposite ordering in
  a nonunitary AO representation. Petite-list shell orbits can reduce
  integral work without first constructing SALCs.
  [Paper II, 1978](https://doi.org/10.1063/1.436313), around Eqs. (33)-(34),
  states the symmetric-density assumption, separately for both UHF spins.
  Derivative-center orbits are not scalar energy multiplicities.
* [Stanton et al. 1991](https://doi.org/10.1063/1.460620), Eqs. (14)-(15):
  direct-product pair dimensions determine actual contraction costs. The
  real D2h multiplication simplification is replaced by explicit dual
  characters here. Future CC residual integration must retain its different
  pairings, spin adaptation and occupied/virtual masks.
* [Dovesi 1986](https://doi.org/10.1002/qua.560290608), Eqs. (30)-(36):
  orbital rotations act together with distinct integer shifts of each
  center and of the finite summation set. Finite Coulomb/exchange truncation
  is part of the operator, not an implementation detail to average away.
* Zicovich-Wilson and Dovesi, [1998 I](https://doi.org/10.1002/(SICI)1097-461X(1998)67:5%3C299::AID-QUA3%3E3.0.CO;2-Q),
  Eqs. (31)-(33) and concluding discussion, requires aligned partners,
  k-dependent actions and projective little-group representations at some
  nonsymmorphic boundary points. [1998 II](https://doi.org/10.1002/(SICI)1097-461X(1998)67:5%3C311::AID-QUA4%3E3.0.CO;2-Y)
  applies SACO blocking to the generalized SCF eigenproblem. An ordinary
  molecular Abelian character table is insufficient at those k points.
* [Casassa et al. 2006](https://doi.org/10.1007/s00214-006-0119-z), Eq. (3):
  localized Wannier subsets mix through matrices as well as translations.
  PAO/PNO/TNO projectors must transform with those matrices; a pair
  permutation is only a special case.
* [Rusakov et al. 2013](https://doi.org/10.1063/1.4821352), Eqs. (5)-(7)
  and Algorithm 1: reanchor every transformed pair using the first center's
  full image shift. Translational density invariance is an explicit premise;
  a universal reduced list of right-hand pairs is generally unavailable.
* [Orlando et al. 2014](https://doi.org/10.1063/1.4895113), Eqs. (1)-(5):
  combine sparse direct-space reconstruction, Fourier transformation and
  SACO transformation instead of materializing intermediate full matrices.
  The useful lesson is the execution schedule and screening interaction,
  not simply the theoretical number of symmetry orbits.

## Validation boundary

The new tests evaluate native full molecular ERIs, J/K and trial-density RHF
energy functionals, native converged RHF Fock comparisons and density-energy
directional finite differences, including complex nonsymmetric/non-Hermitian densities,
nonorthogonal Cartesian AOs, dense gauges within repeated irreps, separate
scalar conjugation and displaced-source rejection. Their complex C3 action
is tested on C3v ammonia, without claiming full non-Abelian reduction.
Existing shared/GDF/periodic tests supply independent coverage of retained
auxiliary factors, fractional translations, Gamma and multi-k transport,
odd/even stabilizers and incompatible finite domains. These existing tests
do not turn the new molecular block consumer into a periodic tensor engine.

No CC residual, analytic gradient or stress code changes in this increment.
Consequently no numerical acceptance for those quantities is claimed.
Full reduced-versus-unreduced CC residuals, relaxed energies and displaced
finite-difference derivative tests remain required before enabling those
consumers. No tolerance or legacy fallback has been changed.

## Cost-routing increment

MolSym's RHF supplement Table S1 compares measured Fock-build times and
already cautions against inferring speed from group order. The new explicit
molecular router measures paired ordered and Davidson/Dupuis-King
representative execution for exactly the same retained source. This is a
cost policy over existing mathematics, not new irreducible-product reduction.
The current consumer is the real RHF/UHF bridge and its shared FockBuild.
Native producer construction, tile metadata and complete/retained numerical
audits qualify the source; a broader producer-issued equivariance/support
attestation is still missing. Pitzer/Goodlett partner representations,
Hellmann-Gauss multiplicity resolution, Kitsaras-Stopkowicz complex dual
indices and Melega gradient response retain their separate acceptance gates.
The periodic literature's finite-domain and phase requirements are unchanged;
a cheaper molecular execution does not qualify a periodic Hamiltonian.

The prepared-source increment amortizes the already measured finite-source
audit and cost profiles across independent HF states. It adds no mathematical
equivariance assumption: exact nuclei/basis matching preserves the original
source, while charge, spin and retained-frame decisions remain method-owned.
It removes repeated work rather than inferring a saving from group order.
Audits of fresh producers still require the existing numerical evidence;
an algebraic producer attestation remains a distinct missing capability.

The public molecular HF entry now consumes that same source and cost policy.
Its new capability is method-owned evaluated-state/output consistency, not
new group theory: overlap and Hcore must satisfy the covariant AO action,
while density-defining natural orbitals preserve broken or stopped states.
Complete HF calculations, including displaced geometry and Cartesian shells,
test the integration. They do not supply the response, fitting or support
derivatives in Melega's Eqs. (6)-(8), or extend the demonstrated C3v reduced
CC scope of Hellmann and Gauss. Producer-issued operation/support evidence
and representation-specific downstream consumers remain missing.
