# χ-CCM / aiccm2026dev-b: an independent finite-torus derivation

## Status and scope

The D128 occupied-action diagnostic now contracts the D127 torus AO action
with actual localized coefficients and overlap, validates metric/occupied
covariance and distinguishes phase-permutations from general orbital mixing.
It is a prerequisite for physical symmetry reduction, not its execution.
The D129 companion now checks a closed supplied spatial quotient, its integer
lattice cocycle and the occupied translation/group laws. Crystal-group
completeness, antiunitary and physical Fock/frozen/virtual/domain/amplitude
certification remain open. See the
[occupied-space contract](experimental/aiccm2026dev_b_symmetry.md#occupied-space-action).

D130 connects actual full-mesh chi RHF output to the shared native numerical
mean-field and masked orbital-sewing checks. It preserves the chi convention
and exact caller-specified frozen bands, rather than importing the periodic
team's finite-Gaussian HF source as if it were chi. This diagnostic bridge
does not authenticate the integral source, its density response, or a reduced
correlation executor; those remain separate integration gates.

D131 isolates and repairs the dominant #704 defect in the shared short-range
nuclear-attraction support, including its analytic derivative partner. An
alpha-only point-charge radius misses displaced Gaussian-product tails.
The three-character He2 sewing control now passes; the even-mesh control
retains a finite-support residual and remains refused. These are numerical
source-qualification steps, not representative-only correlation execution.

χ-CCM[^xccm-convention] is the prose name for the `aiccm2026dev-b`
development line: the finite-translation-group character approach to the
variational finite-BvK-torus CCM. It is an
experimental, independently derived restricted and unrestricted Hartree--Fock
and Kohn--Sham method with finite-torus MP2 and coupled-cluster extensions.
Its sibling is Γ-CCM, selected as `aiccm2026dev-a`, which uses the
union-and-weight/Wigner--Seitz integral-weighting construction. χ-CCM coexists
with the earlier CCM
implementation and is selected with

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention, currently
    `exchange_q0="bvk-ewald"` for the production χ-CCM path. Their distinction
    is not a choice of Coulomb kernels, and equality for a specified
    operator/route is evidence to establish, not a naming premise.

D89 makes the B identity explicit in every finite-torus convention record:
`ccm_approach="chi-ccm"`,
`ccm_construction="finite-translation-group-character"`, and
`evaluation_representation="gamma-centred-character-mesh"`. The first two
identify the approach and mathematical construction; the third identifies how
that χ-defined Hamiltonian is evaluated. Its real-Gamma form is an exact
finite-Fourier representation control but is not assigned a Γ-CCM identity.
The fields are immutable and are carried through B diagnostics/results, fleet
`finite_torus_convention`, QVF vendor metadata, and `.system` run metadata.
QVF wrapper schema version 2 retains the legacy version 1 `representation`
string for compatibility; `evaluation_representation` is normative. Pre-D89
records are not numerically retracted solely because these fields are absent,
but they cannot support a Γ/χ construction-comparison claim.

```python
result = vibeqc.run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="aiccm2026dev-b",
    aiccm_lattice_extension=(4, 4, 4),
    aiccm_backend="four_center",  # or "ri" / "rijcosx"
)
```

The implementation does **not** use `vibeqc.periodic.ccm` and does not copy
the historical program. The earlier implementation, the historical program,
and published CCM formulae are comparison targets. The present definition is
the finite Born--von Karman (BvK) torus Hamiltonian restricted to
translation-invariant determinants. It is evaluated in the
finite translation group's reciprocal irreducible representations. This is
an exact change of basis, not an appeal to reciprocal-space convergence.

The implemented phase is deliberately narrow:

- RHF, RKS, UHF, and UKS (including hybrid functionals), integer occupations,
  neutral primitive cell;
- finite-group and Wigner--Seitz constructions in 1D, 2D, and 3D, while all
  production SCF backends currently fail closed below 3D until the shared
  wire/slab Coulomb convention described below is derived;
- direct corrected-gauge four-centre, pair-resolved RI, and RIJCOSX
  electron-repulsion backends;
- the existing periodic integral, GDF, Ewald, DIIS, basis, and lattice
  infrastructure;
- 3D canonical RHF RI-MP2, unrestricted RI-MP2, full-domain UCCSD(T), and
  restricted/unrestricted local-PNO routes on the exact real torus;
- gauge-invariant SCF one-particle properties; no charged-cell background,
  total analytic gradients, Berry-phase polarization, or relaxed correlated
  density matrices yet. Independently checkable 3D Ewald nuclear-repulsion,
  fixed-density Ewald electron-nuclear, fixed-density kinetic, fixed
  energy-weighted overlap, and restricted/unrestricted active BvK seam
  derivative components are exposed only as derivation aids.

The principal source studied was Peintinger and Bredow, *J. Comput. Chem.*
**35**, 839 (2014), DOI `10.1002/jcc.23550`, together with Chapter 8 of
Peintinger's thesis and the earlier CCM papers cited there. The KS-ADFT CCM
of Janetzko, Köster, and Salahub, *J. Chem. Phys.* **128**, 024102 (2008),
DOI `10.1063/1.2817582`, was audited separately because it introduces
two- and three-center multiplicative weights in deMon2k. Those sources
motivate the questions; they do not define the implementation below.

## 1. Finite cyclic translation group

Let the primitive direct lattice be

\[
 A=(\mathbf a_1,\mathbf a_2,\mathbf a_3), \qquad
 \mathbf R_{\mathbf n}=A\mathbf n,
\]

with only the first \(d\) directions active. The mathematical matrix \(A\)
has lattice vectors as columns, exactly as `PeriodicSystem.lattice` does. Thus
Cartesian translations are \(\mathbf R_{\mathbf n}=A\mathbf n\), and the BvK
supercell matrix is

\[
 A_N=A\operatorname{diag}(N_1,N_2,N_3).
\]

The finite-torus descriptor records this executable contract as
`lattice_vector_convention="columns"`, and successful fleet payloads record
`primitive_lattice_bohr`. The B fleet audit binds the exact
\(A\operatorname{diag}(N)\) matrix to those fields; the Γ/χ comparator defines
no comparison when the binding is absent or inconsistent.

D88 supersedes the false row-lattice premises in D17 and D81. The article
theory already uses the column convention correctly; the repair aligns the
code and its fingerprints with that derivation and changes neither the
Coulomb kernel nor the signed-Madelung convention. On the skew audit

\[
 A=\begin{pmatrix}
 7&0.4&0.2\\
 0.3&8&0.5\\
 0.1&0.6&9
 \end{pmatrix}, \qquad N=(2,3,1),
\]

the corrected positive code convention is
\(\xi_M=0.138352993811598\). The pre-D88 fitted helper returned
\(0.143621616230995\), and the pre-D88 four-center probe returned
\(0.138913224426180\). The fitted error corresponds to 5.268622 mHa/cell of
overbinding for a two-electron RHF seam. Because the Madelung, probe-charge,
and fleet helpers are shared, all affected pre-D88 Γ-CCM and χ-CCM records
require rerun under the fingerprint rule; no missing convention field is
inferred. A symmetric lattice that commutes with `diag(mesh)` is numerically
outside this defect class, but its old record is not newly attested. The shared
builders for `graphene`, `mgo-slab`, `ice-ih`, `co2-dryice`, and
`sio2-quartz` require fresh Γ and χ fingerprints. Ordinary pre-D88 periodic
GDF exact-exchange and BIPOLE J/K records for affected lattice/mesh
combinations also require audit and rerun because they traverse the shared
helpers. A concrete ordinary-GDF control confirms the impact: the
c-diamond/sto-3g `(2,1,1)` fixed point moves from the pre-D88 post-D78 pin
`-74.6405167828 Ha` to `-74.4119904741 Ha`. The old and corrected positive
Madelung constants are 0.5078067935392725 and 0.46971907524744966; multiplying
their difference by six occupied closed-shell bands predicts the measured
`+0.228526309751 Ha` shift. D89 records that Γ-CCM and χ-CCM are distinct
approaches compared at a declared common exchange-q=0 convention; their names
do not select different Coulomb kernels.

For positive integers
\(N_i\), with \(N_i=1\) on inactive directions, define

\[
 \mathcal T_N = \mathbb Z_{N_1}\times\mathbb Z_{N_2}
                 \times\mathbb Z_{N_3}, \qquad
 N_c=N_1N_2N_3.
\]

The BvK identifications are

\[
 \mathbf R_{\mathbf n+N_i\mathbf e_i}\equiv\mathbf R_{\mathbf n}.
\]

This quotient is the cyclic cluster. It is not a finite open cluster and it
has no surface. Its superlattice vectors are \(N_i\mathbf a_i\). AO labels
are \((\mu,\mathbf n)\), with \(\mu\) in the primitive cell and
\(\mathbf n\in\mathcal T_N\).

The primary user parameter is the real-space tuple \((N_1,N_2,N_3)\), called
the lattice extension. Alternatively, a shell radius \(s_i\) requests the odd
extension

\[
 N_i=2s_i+1,
\]

whose representatives run from \(-s_i\) through \(+s_i\). The Wigner--Seitz
cell of the superlattice has a half-extent \(N_i/2\) in primitive lattice
coordinates. This is analogous to k sampling because the reciprocal net below
is its exact character group. It is not an additional real-space truncation:
the selected Coulomb kernel remains BvK-periodic, and its Ewald or fitted
long-range terms are not discarded outside a per-atom sphere.

The characters of this finite Abelian group are

\[
 \chi_{\mathbf m}(\mathbf n)
 =\exp\left(2\pi i\sum_i \frac{m_i n_i}{N_i}\right),
 \qquad m_i=0,\ldots,N_i-1,
\]

corresponding to the **Γ-centred** mesh

\[
 \mathbf k_{\mathbf m}=\sum_i \frac{m_i}{N_i}\mathbf b_i
 \pmod{\mathcal L^*}.
\]

An even-mesh half shift is a different boundary condition and is therefore
not permitted. Symmetry reduction is also not used: all \(N_c\) characters
belong to the exact transform.

For any block-circulant AO matrix,

\[
 X_{\mu\mathbf0,\nu\mathbf R}=X_{\mu\nu}(\mathbf R),
\]

the finite Fourier pair is

\[
 X_{\mu\nu}(\mathbf k)
  =\sum_{\mathbf R\in\mathcal T_N}
   e^{+i\mathbf k\cdot\mathbf R}X_{\mu\nu}(\mathbf R),
\]

\[
 X_{\mu\nu}(\mathbf R)
  =\frac1{N_c}\sum_{\mathbf k}
   e^{-i\mathbf k\cdot\mathbf R}X_{\mu\nu}(\mathbf k).
\]

Consequently, a Gamma-point calculation on the \(N_c\)-cell BvK
supercell, constrained to commute with primitive translations, and a
primitive-cell calculation on the above full mesh are unitarily identical.
This equality holds for every finite \(N\); it is not merely a large-cluster
limit.

## 2. Wigner--Seitz representatives and boundary weights

For centres \(A\) and \(B\) with intra-cell displacement
\(\boldsymbol\delta_{AB}=\boldsymbol\tau_B-\boldsymbol\tau_A\), an element
\([\mathbf n]\in\mathcal T_N\) has infinitely many representatives

\[
 \boldsymbol\delta_{AB}+A\left(\mathbf n+
 \operatorname{diag}(N_1,N_2,N_3)\mathbf z\right),
 \qquad \mathbf z\in\mathbb Z^d.
\]

Let \(M_{AB}([\mathbf n])\) be the set that minimises this Cartesian
distance from centre \(A\). If
\(m_{AB,[\mathbf n]}=|M_{AB}([\mathbf n])|\), the weight of each tied
representative is

\[
 w_{AB}(\mathbf r\mid[\mathbf n])=\frac1{m_{AB,[\mathbf n]}},\qquad
 \sum_{\mathbf r\in M_{AB}([\mathbf n])}
 w_{AB}(\mathbf r\mid[\mathbf n])=1.
\]

For coincident basis offsets in a one-dimensional four-cell cluster, residue 2 has
representatives \(+2\) and \(-2\), each with weight \(1/2\). In skew cells
there may be higher-multiplicity edge or vertex ties. The code solves the
closest-vector problem and stops only when a smallest-singular-value bound
proves that no unexamined image can tie or improve the minimum.
The implementation accepts the explicit centre-pair offset, so a
non-Bravais basis is not reduced to the coincident-centre special case.

### Where the weights enter

The weights select a representative of a **translation equivalence class**.
They do not multiply an AO merely because its atom lies on a geometric
boundary. For a periodised one-electron kernel,

\[
 h^N_{\mu\nu}([\mathbf R])
 =\sum_{\mathbf r\in M([\mathbf R])}
   w(\mathbf r\mid[\mathbf R])h^N_{\mu\nu}(\mathbf r).
\]

All terms on the right are identical under a superlattice translation, so
the weights form a partition of unity. Similarly, after fixing the first AO
at the origin, a periodised four-centre integral is

\[
 (\mu\mathbf0\,\nu[\mathbf R]\mid
  \lambda[\mathbf S],\sigma[\mathbf T])_N
 =\sum_{\mathbf r,\mathbf s,\mathbf t}
   w_{\mathbf R}(\mathbf r)w_{\mathbf S}(\mathbf s)w_{\mathbf T}(\mathbf t)
 (\mu\mathbf0\,\nu\mathbf r\mid
  \lambda\mathbf s\,\sigma\mathbf t)_N.
\]

Again, this is an average over equivalent representatives of the *same
periodised integral*. It therefore leaves the value unchanged. Applying the
same class rule to every symmetry-related term preserves

\[
 (12\mid34)=(21\mid34)=(12\mid43)=(34\mid12)^*.
\]

These identities are necessary for the Coulomb and exchange matrices to be
functional derivatives of one scalar RHF energy.

This differs fundamentally from multiplying an ordinary free-space ERI by
products of pair-dependent boundary factors. Pair-product factors generally
change under \((12\mid34)\leftrightarrow(34\mid12)\) and therefore do not,
without an additional derivation or symmetrisation, define the usual RHF
energy functional.

## 3. Finite-torus Coulomb Hamiltonian

The Coulomb kernel is periodised on the BvK supercell. In three dimensions it
is the zero-average solution

\[
 -\nabla^2G_N(\mathbf r)
 =4\pi\left(\sum_{\mathbf L_N}\delta(\mathbf r-\mathbf L_N)
             -\frac1{\Omega_N}\right),
\]

or, equivalently,

\[
 G_N(\mathbf r)=\frac{4\pi}{\Omega_N}
 \sum_{\mathbf G_N\ne\mathbf0}
 \frac{e^{i\mathbf G_N\cdot\mathbf r}}{|\mathbf G_N|^2}.
\]

The \(\mathbf G=0\) coefficient is fixed to zero. The constant background
in the Poisson equation makes each separated charge component finite; for a
neutral electron-plus-nucleus cell the arbitrary constant cancels in the
total energy. The current implementation rejects charged primitive cells rather than silently
choosing a jellium convention.

The electron--nucleus, nucleus--nucleus, and Hartree terms must use this same
gauge. The implementation obtains all three from vibe-qc's common periodic
Ewald/GDF path. Exchange uses the same periodised interaction. Its finite
mesh \(q=0\) singular channel is treated with the BvK-supercell Madelung
(`exxdiv="ewald"`) correction already implemented in the periodic GDF
backend. Using the primitive-cell Madelung constant here would define a
different finite Hamiltonian and is incorrect.

Every executable χ-CCM result now records this finite-N convention explicitly:
`coulomb_kernel="3d-periodic-g0"`, `exchange_q0="bvk-ewald"`,
`exchange_q0_applicability`, the boundary model, the full character mesh, and
the BvK supercell used for the Madelung constant. Applicability is `active`
only when the actual operator has a nonzero full-range exact-exchange arm. It
is therefore `inactive` for pure functionals and for HSE06, whose exact
exchange is the finite short-range `erfc` kernel and has no singular q=0 seam.
A fleet record is reportable only when its live applicability matches the
independent route contract: RHF and post-HF routes and PBE0 are `active`, while
PBE is `inactive`. Both `audit_b.py` and `compare_b.py` enforce that match;
serializing the field without validating it is not sufficient evidence.
The separate B-owned `AICCM2026DevBExactExchangeAssembly` records the
route-resolved `c_full`, `c_sr`, and physical `omega_screen_bohr_inv` obtained
from the shared live periodic exchange resolver, together with the fixed
v1 schema, resolver, screened applicability, and screened assembly
provenance. It belongs to route-resolved operator provenance rather than to
the finite-torus family convention:
`exchange_q0_applicability` is `active` exactly when `c_full` is nonzero,
while an HSE-type screened arm has `c_full=0`, `c_sr>0`, positive
`omega_screen_bohr_inv`, screened applicability `active`, and the implemented
`short-range-direct` assembly. The active screened label is admitted only
after matching branch-level BIPOLE evidence; it is not an independent
numerical-support attestation or matrix oracle.
A matched descriptor is necessary for any future Γ-CCM/χ-CCM comparison, but
it is not sufficient to establish equality of the two constructions. The
current reportable approach map is empty.
A strict-zero-mode full-range exchange reference is a different finite-N
Hamiltonian with the same thermodynamic target, not a hidden gauge choice.
The familiar rank-1 Madelung expression is only the leading molecular-limit
piece of the exchange seam. At finite solid-like cells the remaining
non-rank-1 contribution is ordinary finite-size physics of the same
\(v_E\) kernel; it is not an RI error and not a gauge freedom that can be
discarded independently.

The 3D direct four-center realization also requires a coherent real-space cell
set for neutral charge. When the Ewald J split is active, the nuclear real-space
cutoff may not extend beyond the electronic J/K cutoff: otherwise nuclear
point-charge images are included without their compensating electronic charge.
The current default therefore resolves from 15 bohr electronic and 25 bohr
nuclear to 15/15 bohr. Fleet records serialize these post-driver values as
`direct_lattice_cutoffs`; the field is `null` for RI and RIJCOSX because those
numbers are not their two-electron direct-lattice cutoffs. Four-center records
made before the shared clamp in `6fed8620` are revision-bound and require a
rerun before quantitative use.

The finite-group construction itself applies unchanged in one, two, or three
periodic dimensions. In 3D the four-center route uses the neutral Ewald J split
and the Ewald exact-exchange correction above. In 1D and 2D the available direct
BIPOLE fallback is a direct-truncated active-lattice kernel, not the neutral
wire/slab finite-torus Green function. `aiccm2026dev-b` therefore blocks the
lower-dimensional four-center route before SCF. The shared lower-dimensional
neutral-RI/GDF mesh is also not a Coulomb kernel: pinning every transverse
reciprocal component to zero removes transverse Coulomb structure and gives a
sheet-like term with vacuum-padding artifacts. `aiccm2026dev-b` therefore
blocks every lower-dimensional SCF backend before SCF. Archived 1D/2D RI and
RIJCOSX numbers are failure evidence, not 3D-periodic-in-vacuum model results
that may be reported as absolute energies. A common 1D wire or 2D slab Green
function must be established before any backend can be re-enabled and backend
differences can be called fitting errors.

### Low-dimensional shared-kernel stub

The fail-closed production path is intentional. The next lower-dimensional
kernel module should be shared by four-center, RI, RIJCOSX, HF, KS, and
post-HF routes, and should provide:

- a 2D slab Green function with isolated transverse boundary;
- a 1D wire Green function with isolated transverse boundary;
- the matching self-potential and exchange q=0 seam derived from the same
  kernel;
- tests proving that this mixed-boundary kernel, not the 3D
  `exchange_q0="bvk-ewald"` convention, is the finite-N Hamiltonian being
  correlated.

Until those items exist, isolated wire/slab absolute-energy claims remain
out of scope.

## 4. RHF energy, Fock matrix, and double counting

Use a spin-summed density \(D(\mathbf k)\). With uniform weight
\(w_{\mathbf k}=1/N_c\), the electron count per primitive cell is

\[
 N_e=\sum_{\mathbf k}w_{\mathbf k}
       \operatorname{Tr}[D(\mathbf k)S(\mathbf k)].
\]

The closed-shell projector condition is

\[
 D(\mathbf k)S(\mathbf k)D(\mathbf k)=2D(\mathbf k).
\]

Define the periodised Coulomb and exchange contractions in the conventional
way,

\[
 J_{12}[D]=\sum_{34}D_{34}(12\mid34)_N,
 \qquad
 K_{12}[D]=\sum_{34}D_{34}(13\mid24)_N.
\]

Translation conservation makes these block diagonal in \(\mathbf k\), with
the usual momentum-transfer sums implicit in the GDF contraction. The energy
per primitive cell is

\[
 \mathcal E_N[D]=E_{NN}^N
 +\sum_{\mathbf k}w_{\mathbf k}
   \operatorname{Tr}[D(\mathbf k)h(\mathbf k)]
 +\frac12\sum_{\mathbf k}w_{\mathbf k}
   \operatorname{Tr}\left[D(\mathbf k)
   \left(J(\mathbf k)-\frac12K(\mathbf k)\right)\right].
\]

Its derivative is

\[
 F(\mathbf k)=h(\mathbf k)+J(\mathbf k)-\frac12K(\mathbf k).
\]

The factors \(1/2\) and \(1/2\) are respectively the pair-counting factor
and closed-shell exchange factor. No separate “central cell” energy is added,
and no Wigner--Seitz factor is applied a second time. Each finite-group
translation class occurs exactly once; boundary representative weights sum
to one. This is the double-counting argument.

For RKS, let \(E_{\mathrm{xc}}[\rho_D]\) be a supported XC functional and
\(\alpha\) its declared exact-exchange fraction. The finite-torus energy is

\[
 \mathcal E_N^{\mathrm{RKS}}[D]=E_{NN}^N
 +\operatorname{Tr}_w[Dh]
 +\frac12\operatorname{Tr}_w[DJ]
 -\frac{\alpha}{4}\operatorname{Tr}_w[DK]
 +E_{\mathrm{xc}}[\rho_D],
\]

where \(\operatorname{Tr}_w\) includes the uniform finite-character weights.
Its derivative is

\[
 F^{\mathrm{RKS}}=h+J-\frac{\alpha}{2}K+V_{\mathrm{xc}}.
\]

Thus pure semilocal DFT has \(\alpha=0\), whereas hybrids use the same
periodised exchange tensor and finite-size convention as RHF.

The implementation also admits a deliberately narrow nonlocal full-grid
boundary. A capability-version-1 external provider is variationally evaluated
on one complete atom-major periodic grid with the explicit periodic-lattice
density domain. Current χ-CCM support requires 3D RKS/UKS, `four_center`, and
\(\alpha=0\). RI, RIJCOSX, lower-dimensional external XC, and
external-provider hybrids fail closed; the general hybrid statement above
continues to describe the ordinary built-in functional path.

## 5. Variational statement

For a fixed real-space extension, basis, periodised Coulomb gauge, and nuclear
geometry, `aiccm2026dev-b` minimises the appropriate SCF functional over

\[
 \mathcal V_N=\{D:\ D^\dagger=D,\ DSD=2D,\
 \operatorname{Tr}(DS)=N_cN_e,\ [D,T_{\mathbf R}]=0\}.
\]

Equivalently, it minimises independently represented occupied subspaces at
all finite-group \(\mathbf k\), subject to the common electron count. The
generalised Roothaan equations are

\[
 F(\mathbf k)C(\mathbf k)
 =S(\mathbf k)C(\mathbf k)\varepsilon(\mathbf k).
\]

DIIS and damping may alter the path to a stationary point but not the
functional. A converged solution is stationary in \(\mathcal V_N\); as with
ordinary RHF or approximate KS-DFT, SCF alone does not prove it is the global
minimum. A fully
unconstrained Gamma-supercell calculation has a larger determinant space and
may break primitive translation symmetry. Such a broken-symmetry result is
not required to equal the primitive-cell mesh result.

SCF path controls are numerical provenance rather than terms in the finite
Hamiltonian. The B fleet therefore keeps the requested Fock-mixing fraction in
`scf_options.fock_mixing`, while `AICCM2026DevBDiagnostics.fock_mixing` and
`convergence_diagnostics.fock_mixing` record the executed effective weight of
the previous Fock matrix after backend defaults are resolved. In the
four-center KS backend, a request of `0.0` can execute as `0.30` when DIIS is
disabled. D86 changes only the accuracy of that provenance record; it does not
change the SCF algorithm, energy, or finite Hamiltonian. Pre-D86 DIIS-off
four-center KS records are not convergence-fingerprint-complete.

D87 resolves the direct-call input before backend dispatch. A non-`None`
`fock_mixing=` keyword overrides `options.fock_mixing`; otherwise the options
field supplies the request. The resolver validates `[0, 1)` and does not
rewrite the caller's mixing field. Every `four_center` route and multi-cell
fitted RHF/RKS can execute nonzero mixing. Three-dimensional Gamma-only RI RHF
accepts resolved zero but fails closed on nonzero rather than switching to the
legacy cutoff-selected Gamma GDF fallback; the other one-cell fitted
restrictions remain unchanged. UHF and UKS can execute nonzero mixing on
`four_center`, including both phases of a spin schedule, but fitted RI/RIJCOSX
UHF/UKS fail closed because their multi-k open-shell GDF loop has no
previous-Fock update.
An explicit keyword zero therefore overrides a nonzero options value; on a
DIIS-off four-center KS route, that requested zero remains subject to the
separate automatic `0.30` execution rule. On supported execution routes this
can change the SCF trajectory or stationary basin, but not the finite
Hamiltonian or backend mixing formula. The Gamma explicit-zero correction
deliberately restores the declared operator instead of preserving the old
route-selection bug. Successful pre-D87 direct calls with differing
keyword/options values, plus fitted RI/RIJCOSX UHF/UKS calls with a nonzero
options-only request, have incomplete request fingerprints.
Pre-D87 Gamma-only RI RHF calls where either input source was nonzero,
including an explicit-zero keyword over nonzero options, followed the legacy
operator and are not χ-CCM-B results. The closed-shell fleet is options-only
and its records are unchanged. D72 and the analytic total-gradient fail-close
remain in force. The shared D72 dimension guard precedes every backend-specific
Gamma and mixing guard; this closes the former 1D one-cell RI-RHF escape, whose
absolute-energy values are invalid and unreportable.

D97 applies the same execution discipline to unrestricted level shifting.
For a restricted density \(D=2P\), the virtual-space shift has the form
\(F_b=F+bS-(b/2)SDS\). For a unit-occupation unrestricted spin projector, the
corresponding form is
\(F^\sigma_b=F^\sigma+bS-bSP^\sigma S\). The shared four-center UHF/UKS loop
currently applies the restricted `/2` coefficient to each spin density, while
the fitted multi-k UHF/UKS loop does not execute the request. The B entry
points therefore reject a nonzero `options.level_shift` or any nonzero
`options.level_shift_schedule` entry for every unrestricted backend before
dispatch. `run_periodic_job(..., level_shift=...)` propagates its resolved
value to the B options object before dispatch, rather than dropping a public
request. Empty and all-zero schedules remain valid. The common D72 guard still
runs first, so this convergence-control gate cannot expose a lower-dimensional
energy. Restricted RHF/RKS driver semantics were otherwise unchanged by D97,
and the forwarding repair makes their supported public requests reach the
selector. D107 closes one fitted restricted exception. At a `(1,1,1)`
character mesh, the shared RHF dispatcher admits its compensated PBC-GDF route
only when the static level shift is zero. A nonzero static shift falls through
to the legacy Gamma driver, which receives neither the requested RSGDF/MDF
method nor its cutoff/tail and can select a different cutoff-dependent Gamma
operator. A nonzero schedule with a zero static value reaches PBC-GDF but is
not executed. The χ RHF/RI boundary therefore rejects either nonzero source
before backend SCF. Empty and all-zero schedules remain valid. True
multi-character fitted RHF retains static and scheduled shifting without
changing route class. Four-center RHF/RKS retains static shifting with an empty
schedule, but D108 rejects every nonempty options-level schedule: the BIPOLE
drivers consume a separate `LevelShiftSchedule` argument that the χ wrapper
does not transport. The public schedule contract supersedes a simultaneous
static shift, so even an all-zero schedule cannot be treated as inert when the
static value is nonzero. This turns a silent mismatched request into a
pre-dispatch error without widening the implementation to shared driver work.
The previous-Fock-mixing guard retains precedence, D72 remains the first
lower-dimensional failure, and a post-HF call inherits the Gamma-RI protection
through its χ RHF reference.

D109 applies the same executed-control rule to the C1c quadratic-SCF fallback.
The periodic options objects expose an activation iteration, denominator
shift, and trust-region cap, and the B campaign producer records all three.
The current χ four-center and fitted RHF/RKS/UHF/UKS engines execute none of
them: only separate periodic Ewald drivers own the quadratic orbital-rotation
update. Every B entry point therefore validates
`quadratic_fallback_iter` as a non-negative integer and rejects a positive
value before backend SCF. Zero is the supported inactive state; the associated
shift and maximum-step values are dormant when activation is zero. The guard
runs after existing selector checks, so D72, the D107 Gamma-RI mixing and
shift errors, and the D108 restricted four-center schedule error keep their
established precedence. Closed-shell post-HF routes inherit the same stop from
their B RHF reference. A pre-D109 3D `ok`/`not_converged` SCF or derived
post-HF record that accepted a positive activation is conservatively
request-fingerprint-invalid and revision-bound because its selected backend
neither supported nor attested that conditional request. This quarantine does
not show that the trigger was reached or that the energy changed; only
`iterations > quadratic_fallback_iter` establishes that the intended trigger
was crossed. Unsupported/error records remain valid failure evidence, and
nondefault shift/step values remain valid and dormant when activation is zero.
D109 does not reclassify the default SCF operator or finite Hamiltonian, adds
no gradient term, and changes no construction, Coulomb convention, or
numerical-support qualification.

D110 closes the remaining restricted four-center static-shift mismatch. With
an empty explicit `level_shift_schedule`, BIPOLE consumes
`options.level_shift` as a persistent value unless its separate
`LevelShiftSchedule` argument is supplied. The χ wrapper formerly copied
`options.level_shift_warmup_cycles` into diagnostics without lowering it into
that argument. A nonzero static shift with the default `-1` auto request or a
positive warm-up length therefore remained active for the full SCF even
though the record described a finite startup interval.

For restricted four-center RHF and RKS, D110 resolves the warm-up before
dispatch. The static shift must be finite and non-negative. When it is
nonzero, the warm-up must be an integer greater than or equal to zero or `-1`
for auto. Auto means up to five shifted cycles. A positive request is capped
at `max_iter - 1`, so at least one scheduled tail cycle is unshifted. An
active auto or positive warm-up requires `max_iter >= 2`; a smaller budget
fails before SCF because it cannot contain both a shifted startup and an
unshifted tail. For a resolved length greater than zero, the wrapper supplies

\[
 (\underbrace{b,\ldots,b}_{n_w\ \mathrm{cycles}},0)
\]

to BIPOLE; the terminal zero is reused for every later iteration. An explicit
warm-up of zero deliberately keeps the historic persistent static-shift mode,
including a one-cycle budget. Any warm-up modifier is dormant and
uninterpreted when the static shift is zero. The caller's options object is
not rewritten. Its `level_shift_warmup_cycles` remains the request, whereas
`AICCM2026DevBDiagnostics.level_shift_warmup_cycles` records the executed
effective length. With an empty explicit schedule, diagnostic zero means
persistent when the static shift is nonzero and inactive when it is zero. On
fitted routes, the same diagnostic uses backend-resolved telemetry; a
nonempty explicit schedule supersedes the warm-up modifier and records that
modifier as zero, so the request schedule must also be inspected.

D108 remains an independent boundary: every nonempty options-level schedule
still fails closed before D110, including an all-zero schedule beside a
nonzero static shift. D72 remains the first lower-dimensional error, and D109
continues to run after the level-shift checks. D97 unrestricted and D107
Gamma-only RI policies are unchanged. Pre-D110 declared-3D restricted
four-center RHF/RKS `ok` or `not_converged` records with a nonzero static
shift, an empty explicit schedule, and auto or positive warm-up are
request-fingerprint-invalid and revision-bound because the old route executed
a persistent shift while recording a release request. Only a trajectory that
extends beyond the resolved warm-up proves that the intended release point
was crossed; the quarantine does not by itself prove an energy change.
Error/unsupported evidence, zero-shift dormant modifiers, and explicit
persistent warm-up zero remain valid. This repairs convergence-control
execution and provenance only. It does not change the finite Hamiltonian,
the Coulomb convention, the distinct Γ-CCM and χ-CCM constructions, or the
analytic-gradient boundary.

### Unrestricted variational space

Let \(P^\alpha\) and \(P^\beta\) be spin densities with unit occupation and
let \(P=P^\alpha+P^\beta\). The finite-torus UHF functional per primitive
cell is

\[
 \mathcal E_N^{\mathrm{UHF}}[P^\alpha,P^\beta]
 =E_{NN}^N+\sum_{\sigma\in\{\alpha,\beta\}}
 \operatorname{Tr}_w[P^\sigma h]
 +\frac12\operatorname{Tr}_w[PJ[P]]
 -\frac12\sum_\sigma\operatorname{Tr}_w[P^\sigma K[P^\sigma]].
\]

Its independent functional derivatives are

\[
 F^\sigma=h+J[P]-K[P^\sigma].
\]

The minimized set is

\[
 \mathcal V_N^{\mathrm U}=\{(P^\alpha,P^\beta):
 (P^\sigma)^\dagger=P^\sigma,\quad
 P^\sigma S P^\sigma=P^\sigma,\quad
 [P^\sigma,T_{\mathbf R}]=0,\quad
 \operatorname{Tr}(P^\sigma S)=N_c n_\sigma\}.
\]

Here \(n_\alpha=(N_e+2S)/2\) and \(n_\beta=(N_e-2S)/2\) are integers fixed
by the primitive-cell multiplicity. This is a genuine unrestricted
variation: alpha and beta orbitals are not constrained to share a spatial
subspace. UHF stationarity does not prove spin purity. The implementation
therefore reports both \(\langle S^2\rangle\) and the ideal \(S(S+1)\).

For UKS with a hybrid fraction \(a_x\), replace the UHF exchange energy by

\[
 -\frac{a_x}{2}\sum_\sigma
 \operatorname{Tr}_w[P^\sigma K[P^\sigma]]
 +E_{\mathrm{xc}}[\rho_\alpha,\rho_\beta],
\]

which gives

\[
 F^\sigma_{\mathrm{UKS}}=h+J[P]-a_xK[P^\sigma]
 +V_{\mathrm{xc}}^\sigma.
\]

No new Wigner--Seitz weight appears in these equations. Both spin projectors
use the same declared finite Hamiltonian, including
`exchange_q0="bvk-ewald"` when exact exchange is present, and the same
translation-class partition.

## 6. Algorithm

1. Validate neutral, integer-occupation, zero-temperature RHF/RKS/UHF/UKS input
   in three periodic dimensions. For 1D/2D inputs, stop before SCF until one
   shared neutral wire/slab Coulomb kernel exists for every backend.
2. Form \(\mathcal T_N\), its exact Wigner--Seitz representative partition,
   and the unreduced Γ-centred character mesh.
3. Build lattice-summed overlap and one-electron matrices with the common
   periodic gauge.
4. Build the periodised J and K matrices with one of the three backends
   defined below. Apply the BvK-supercell exchange-divergence correction only
   in the 3D Ewald gauge.
5. Solve the per-k generalised eigenproblems using the existing canonical
   linear-dependence handling and SCF accelerator.
6. Return energy per primitive cell and measure
   \(\max_k\lVert D_kS_kD_k-2D_k\rVert_F\), electron-count error,
   Wigner--Seitz partition error, and the imaginary residue of the inverse
   Bloch density.

The inverse Bloch transform stored in the implementation is also the bridge
for future explicitly real-space post-HF work.

### Direct-ERI output-cell parallelism

D114 adds the first χ-owned MPI execution layer to restricted 3D
`four_center` RHF, and D116 extends the same complete-output contract to
non-screened RKS. It farms the outer output-block list of the direct
short-range traversal. For an output cell (g), one rank evaluates a complete
J block and, when exact exchange is present, the corresponding complete K
block:

\[
 J_{\mu\nu}(g),\quad K_{\mu\nu}(g)
 \quad\text{with the full declared }(c_\lambda,c_\sigma)\text{ sum}.
\]

Only the outer output index is partitioned. The padded M5 internal cell list,
density support, shell masks, erfc operator, reciprocal long-range Hartree
term, long-range exchange channels, and BvK q=0 correction are unchanged.
Complete blocks are allgathered in canonical output order before the
incremental-Fock accumulator sees them. This avoids a floating reduction and
leaves serial arithmetic bit-identical. An oversubscribed rank with no output
task does not call the native domains builder because that older interface
uses an empty output list to mean all cells.

`LatticeOutputPartition` verifies exact ownership and coverage using the task
count, cyclic or block strategy, source rank, and a SHA-256 fingerprint of the
ordered cell keys. The χ selector currently chooses cyclic assignment, which
spreads a radial cost ordering more evenly. The same helper is used before
pair-resolved or symmetry-representative reconstruction, with every output
shell mask kept beside its owned index. The result records the executed world
size, per-rank task counts, task count, strategy, cell-order fingerprint,
complete-internal-sum contract, and complete-block allgather mode.

The execution task is named `chi-direct-output-cell` deliberately. A radial
direct-ERI output task is not a Γ-CCM union-and-weight WSC and is not, without
an additional group-level mapping, one χ translation residue. D114 performs
no representative/residue partition and assigns no physical Wigner weight.
Any future residue-level scheduler must aggregate tied Wigner--Seitz
representatives before interpreting one residue contribution. D114 therefore
establishes an exact SCF scheduling seam but does not yet implement
character/residue farming or translation-representative local correlation.
It also does not make the Coulomb interaction vanish beyond a distance. A
future DLPNO reduction must exploit correlation locality and domain
compressibility while retaining the declared global periodic Coulomb
operator.

The fleet validation for this seam is deliberately separate from the ordinary
campaign producer, which is serial and owns one result file. The
`run.sh --d114-mpi` path performs one source/core preflight before launch,
requires exactly two active MPI ranks, evaluates both the replicated unfarmed
BIPOLE reference and the farmed χ selector on every rank, and requires exact
energy, Fock, and density parity. Per-rank scheduling and finalized process
identity are gathered before rank zero atomically writes the sole artifact.
That artifact is implementation-parity evidence only; it is not a D93/D103/
D104 qualification or an independent absolute-energy benchmark.
The wrapper also owns the hybrid resource contract: it creates both MPI ranks,
divides the total PBS `VQ_CPUS` allocation evenly between them, and rejects a
separate scheduler-task declaration because Torque does not expand the node
allocation from that field. It selects an explicit disjoint OpenMP core domain
for recognized Intel MPI or Open MPI launchers and fails closed for an unknown
launcher. Both ranks verify that the native OpenMP maximum agrees with the
requested threads per rank, and the
artifact binds those counts, the binding policy, thread-library caps, and the
fixed SCF and Ewald seam inputs.

D115 separates installed MPI capability from executed MPI state. A serial
preflight or ordinary `import vibeqc` does not load `mpi4py.MPI`, does not call
`MPI_Init`, and retains the inactive identity world. Validated launcher-rank
evidence or an explicit required-size contract admits communicator binding;
the D114 wrapper declares that MPI is required and that the expected world has
exactly two ranks. Missing MPI,
malformed or contradictory rank evidence, inadequate thread support, and a
communicator that disagrees with the launcher fail closed. This changes only
process discovery. It does not alter D114's direct-output task map, the
complete internal translation sum, or any scientific operator.

D116 uses this seam for pure semilocal and global-hybrid RKS. Without
symmetry-representative reduction, the pure route farms the padded short-range
J traversal, while the global hybrid farms the fused padded J/K traversal.
When representative masks are active, the shared fused domains builder may
compute K beside J for a pure functional and RKS discards that K. XC grid
construction and quadrature, reciprocal
Hartree, long-range exchange, the BvK q=0 seam, character transforms, and
diagonalization remain replicated. The executed scheduling record is carried
onto both the BIPOLE RKS result and χ diagnostics. HSE-type screened hybrids
stay serial at this layer because they contain a base J traversal and a
separate screened-K traversal; the v1 execution record cannot attest both
phases without ambiguity. The generic RKS driver rejects an explicit farming
request for such a functional. This is a provenance boundary, not a claim
that screened exchange cannot be parallelized.

The rank-safe `run.sh --d116-rks-mpi` producer supplies the fleet acceptance
contract for this extension. It fixes c-diamond/STO-3G, a `(2,2,2)` character
mesh, 15-bohr cutoffs, one PBE cycle, zero Fock mixing, and symmetry off. Two
ranks compare the farmed result with the unfarmed reference under
`d116-rks-mpi-validation/v2`. Absolute total, electronic, nuclear, and XC
energy differences must not exceed `1e-10` Ha; maximum elementwise differences
in every Fock and density block must not exceed `1e-12`. The same bounds apply
when the recorded energy observables are compared across ranks. Array block
shapes, convergence and iteration state, executed zero mixing, D83
exchange-q=0 applicability `inactive`, rank and task census, canonical task
order and fingerprint, launcher controls, and source/core process identity
remain exact. The artifact records both observational `exact_parity` and the
normative v2 `contract_parity` verdict plus its complete tolerance contract.
The parallel-execution record remains v1 because its scheduling and binding
semantics did not change. Existing D116 v1 artifacts keep their exact-contract
meaning and are not upgraded by reinterpretation.
This is an implementation-parity receipt, not absolute-energy validation or a
claim that replicated XC and reciprocal work scales.

The D114/D116 work unit remains a radial direct-ERI AO output block. It is
neither a Γ-CCM union-and-weight WSC nor a χ finite-group residue.

### Finite-group residue diagnostic parallelism

D117 adds the first scheduler whose task is genuinely one χ finite-translation
group residue. It applies only to the inverse finite-character transform used
by the attached SCF density diagnostics. The ordered task keys are the unique
residues of the declared BvK group. Before ownership is assigned, every tied
minimum-image representative in the zero-offset cell set used by this density
diagnostic is grouped with its residue. Alias weights must be equal and sum to
one. The weights certify that representative partition; they do not multiply
the residue block again. This record does not attest the offset-dependent
representative sets needed by AO pairs, quartets, or local-correlation
domains.

For an owned residue (g), one rank evaluates the complete inner character
sum

\[
 D(g)=\sum_q w_q\exp(-2\pi i q\mathbin{\cdot}g)D(q).
\]

Thus MPI partitions residues, not characters, and does not turn the exact
finite Fourier sum into a partial reduction. The existing native transform is
called only for the rank's nonempty residue subset. Empty ranks contribute an
empty parcel because the native empty-subset convention means all residues.
The scheduler first validates that the declared mesh, complete uniform dual
character net, normalized weights, residue coverage, and translation
congruences define the same finite group. Active ranks allgather a
construction fingerprint over that evidence and the zero-offset
representatives. `LatticeOutputPartition` independently verifies the
ordered-residue fingerprint, strategy, source ownership, and exact-once
census before allgathering the complete blocks in canonical residue order.
Restricted density uses one such schedule. Unrestricted density transforms
alpha and beta separately and requires identical execution records before
forming the diagnostic maximum.

The result exposes schema
`vibeqc.aiccm2026dev-b.residue-inverse-transform-execution/v1` through
`result.residue_inverse_transform_execution` and
`result.aiccm2026dev_b.residue_inverse_transform`. The record includes the
character mesh and labels, unique residue keys, the representative scope,
multiplicities and unit weight sums, local and per-rank ownership, separate
ordered-key and construction fingerprints, complete-character-sum truth, and
an explicit false value for extra Wigner-weight application. Odd, even, skew,
two-rank, three-rank, cross-rank-drift, unrestricted-spin, and
oversubscribed-empty-rank controls pin the grouping and exact serial result.

D117 is a real χ construction task map, unlike the construction-neutral
D114/D116 radial output map. It is nevertheless diagnostics parallelism, not
SCF scaling: J/K construction, XC, diagonalization, localization, post-HF,
QVF/property transforms, and gradients are unchanged. It is not Γ-CCM WSC
farming, changes no Coulomb or exchange-q=0 convention, introduces no
distance cutoff, and makes no DLPNO scaling claim.

### Integral backends

All backends act on the same finite-character density at the declared
`coulomb_kernel="3d-periodic-g0"` and `exchange_q0="bvk-ewald"` convention.
Backend differences are then representations of the electron-repulsion tensor,
not silent Hamiltonian changes.

1. `four_center` uses the periodic BIPOLE engine in 3D: direct four-centre
   short-range J/K plus the reciprocal long-range Hartree term and BvK
   exchange correction. The 1D/2D four-centre route fails closed because the
   current direct-truncated fallback is not the neutral wire/slab finite-torus
   Green function and over-binds chain benchmarks by a Madelung-scale shift.
   D105 explicitly disables BIPOLE's quartet far-field prototype on every χ
   dispatch. The shared drivers now fail closed on an explicit request because
   the prototype does not preserve the exact three-translation Fock domain.
   D93 attests the direct M5 image domain, not far-field activation, order,
   penetration dispatch, or approximation error. Because this complete
   operator contains no density fit, D122 rejects any explicit `aux_basis`
   before BIPOLE dispatch instead of silently attaching an unexecuted fitted
   control to a four-center record.
2. `ri` uses pair-resolved RSGDF or MDF. With auxiliary functions \(P,Q\)
   and Coulomb metric \(V_{PQ}=(P|Q)\),

   \[
    (\mu\nu|\lambda\sigma)_{\mathrm{RI}}
    =\sum_{PQ}(\mu\nu|P)(V^{-1})_{PQ}(Q|\lambda\sigma).
   \]

   The fit is applied consistently to J and K.
3. `rijcosx` uses the same pair-resolved RI J and evaluates exact exchange
   by periodic chain-of-spheres quadrature. It is therefore a controlled
   three-centre/seminumerical approximation, not a different cyclic
   Hamiltonian.

The `aux_basis` keyword belongs only to `ri` and `rijcosx`; both fitted
selectors forward its exact value to their GDF driver. Passing any non-`None`
value with `four_center` is a contradictory request and raises before SCF.

#### Fitted nuclear assembly and inherited support guards

D99 pins the shared fitted-route behavior at the χ-CCM-B selector. Since
`ddbe859d1`, every supported 3D fitted Gamma-only path and every multi-k
restricted or open-shell route evaluates nucleus--nucleus repulsion with
`ewald_nuclear_repulsion` rather than `nuclear_repulsion_per_cell`. This pins
routing only. D102 records the historical defect: the shared helper first
preselected translations by an unshifted norm and could omit a shifted pair
inside the requested cutoff. The `four_center` route used the same affected
helper. Same-helper equality therefore proved assembly, not convergence, and
every successful or non-converged χ-B absolute-energy row from that interval
was quarantined. Here Gamma-only identifies a one-character evaluation path
inside χ-CCM; it does not assign Γ-CCM construction identity.

That historical quarantine is permanently bound to exact
`vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v1` evidence. Version 1
requires the affected implementation label, `qualification="not-qualified"`,
and `repair_commit=null`; it has no qualified state and is never upgraded.

D104 binds the repaired shared helper independently. Commit
`e578b86c00268a172b651cae29c7845836e31e17` enumerates pairs from the centered
source-observer displacement with exact interplanar bounds. Before entering
SCF, the producer verifies repair ancestry and runs an 18-bohr MgO
point-charge canary that shifts the oxygen basis charge by `37 a1`. The stale
core differs by about 15.08 Ha; the repaired core gives zero difference
against a `1e-10` Ha tolerance. Failure stops before any absolute energy is
assembled. The cached canary is then bound after calculation to finalized D77
identity in exact
`vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v2` evidence: implementation
label, repair SHA, full producer commit, core build ID, probe-attestation ID,
verified ancestry, and the exact canary payload. The audit and comparator
cross-bind those identities to provenance and the nested attestation. A fresh
v2 row may clear this shared-Ewald component only; D93, D98, and D103 remain
independent.

Since `6ce142339`, the shared GDF layer rejects compact dense-core MDF systems
in the MgO/STO-3G class, and that failure propagates through the B selector.
The separately validated vacuum-padded MDF envelope is unchanged, but D93
continues to mark every fitted B row `not-qualified`. Commit `ad96a2f63` also
gives the fitted B routes a bounded shared-q auxiliary Fourier-transform cache.
That inheritance is bit-identical and changes memory/performance only. Every
fitted B RHF/RKS/UHF/UKS dispatch additionally pins `ibz_native=False` and
`compute_gradient=False`. Future generic IBZ or GDF-gradient defaults therefore
cannot replace the complete unreduced character mesh or open a total-gradient
route. None of these shared changes widens B: 1D/2D SCF and all analytic
total-gradient routes still fail closed, and D83, D88, and D98 are unchanged.
Under D106 the same four fitted dispatches may carry an explicit
`rsgdf_tail_ke_cutoff`, but only with RI or RIJCOSX and
`gdf_method="rsgdf"`. The base `rsgdf_ke_cutoff` must be positive and finite,
and the tail must be finite and strictly larger; its exact value is forwarded
to the shared driver and recorded as
`aiccm_resolved_rsgdf_tail_ke_cutoff`. Four-center and MDF
combinations fail before SCF. The generic Gamma RHF driver can auto-size a
dense-core tail when the input is null, but χ does not inherit that implicit
choice: Gamma-only RHF/RI stops before SCF whenever the shared resolver would
extend the cutoff strictly above the base. An automatic value equal to an
already sufficient base builds no complementary shell and is normalized as
inactive. The caller must provide an extended value explicitly or use a
nontrivial character mesh. Thus every successful null B result means no
complementary tail executed. This proves transport, not convergence or D93
support qualification.
The `run_ccm_rhf_gdf` campaign producer is a neutral fitted-torus Bloch/GDF
representation control, not Γ-CCM production, and separately recomputes its
executed nuclear scalar through the same shared Ewald helper. It emits D104 v2
for that helper but always remains quantitatively `not-qualified` because it
is a Bloch/GDF representation control with unqualified fitted support. It is
not the mapped real-Gamma control. The mapped A-owned `aiccm-hf-direct`
producer currently emits no D104 v2 evidence, so the optional control remains
unavailable.
The dedicated D101 `run_case_cmp_b.py` producer is instead the high-level
χ-CCM RI route-control diagnostic assigned to this line. Neither producer
realizes the union-and-weight/Wigner--Seitz Γ-CCM construction, so their
labels and common lower-level ancestry cannot establish a Γ-CCM/χ-CCM
approach result.

For global hybrids such as PBE0, all three backends use the same full-range
exchange convention described above. The 3D four-center RKS/UKS route also
supports HSE06 as \(0.25K_{\operatorname{erfc}}(\omega=0.11)\), with no
full-range exchange seam. RI and RIJCOSX currently fail closed for HSE06 and
all other range-separated functionals because the χ fitted-backend contract
has no B-owned screened-COSX validation or provenance. The shared generic COSX
backend can build HSE exchange, but this does not widen the χ construction by
inheritance. This backend split prevents the former silent HSE06 to PBE0
substitution; it is not a Γ-CCM/χ-CCM construction or approach distinction. The
four-center HSE algebra now uses the shared M5 padded screened-exchange
traversal. D98 records this as `short-range-direct` only when the K-erfc
branch emits matching execution evidence; it never infers the label from the
functional name. HSE remains outside the B fleet and still needs
route-specific quantitative validation before its absolute energies are
reportable.

Full-grid external providers do not inherit that hybrid/backend matrix. Their
current accepted shape is pure capability-version-1 XC on the 3D
`four_center` RKS/UKS route, with the complete Gamma-centred character mesh and
an explicit periodic-lattice density domain. Fitted backends and nonzero
external exact-exchange fractions are rejected before the backend build.

All three backends are currently 3D-only. Lower-dimensional inputs fail before
the backend build; RI and RIJCOSX do not retain a 3D-periodic-in-vacuum
exception.

The q-only `compcell` fit is rejected: on tight cells it does not represent
one consistent four-centre finite-torus tensor and is known to admit
non-physical converged fixed points. The present native bridges also reject
one-cell RI/RIJCOSX RKS and one-cell RIJCOSX RHF rather than silently
substituting a different Gamma-only algorithm. In 3D, one-cell RI RHF is
supported only at resolved `fock_mixing=0`: a nonzero request would make the
shared dispatcher substitute the legacy molecular-limit GDF exchange operator for
the declared pair-resolved `exxdiv="ewald"` route, so χ-CCM-B fails closed.

## 7. Canonical finite-torus RI-MP2

MP2 is built on the converged χ-CCM RI-RHF determinant, not on the legacy
Gamma-supercell GDF driver. That distinction is numerical as well as formal:
on the two-cell 8 x 12 x 12 bohr H2 control, the legacy Gamma-supercell HF
reference differs from χ-CCM RI-HF by 0.0046408868 Ha per cell. Reusing it would
change the zeroth-order Hamiltonian. The difference is not a gauge-invariant
post-HF offset. The exchange q=0 seam shifts the HF orbital energies, so the
MP2 denominator
\[
 \Delta_{ij}^{ab}=\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b
\]
inherits the declared convention. External KMP2 or CCSD(T) parity is meaningful
only when the reference and the correlated calculation both use
`exchange_q0="bvk-ewald"` or both use some explicitly labelled alternative.

Let (i,j) denote occupied bands, (a,b) virtual bands, and let
(N_k=N_c). The pair-resolved canonical auxiliary factor is

\[
 \widetilde L^{\mathbf k_i\mathbf k_a}_{P\mu\nu}
 =\left[U_{\mathbf q}\lambda_{\mathbf q,+}^{-1/2}
 U_{\mathbf q}^{\dagger}T_{\mathbf q}\right]_{P\mu\nu},
 \qquad \mathbf q=\mathbf k_a-\mathbf k_i.
\]

Here only metric eigenvalues above the declared threshold enter. Keeping the
factor in the original auxiliary-AO basis is important: the compact factor
(\lambda^{-1/2}U^{\dagger}T) is invariant in an SCF contraction with its own
adjoint, but independently diagonalised (\mathbf q) and (-\mathbf q)
blocks have arbitrary eigenvector phases. Those phases need not cancel in an
MP2 contraction between two blocks. The canonical matrix function above is
unique on the retained subspace and obeys the required conjugation relation.

Transform one occupied and one virtual index,

\[
 L^{\mathbf k_i\mathbf k_a}_{Pia}
 =\sum_{\mu\nu}C_{\mu i}^{*}(\mathbf k_i)
 \widetilde L^{\mathbf k_i\mathbf k_a}_{P\mu\nu}
 C_{\nu a}(\mathbf k_a).
\]

For every triple ((\mathbf k_i,\mathbf k_a,\mathbf k_j)), the fourth
character is fixed exactly by

\[
 \mathbf k_b=\mathbf k_i-\mathbf k_a+\mathbf k_j+\mathbf G,
\]

where (mathbf G) returns the point to the finite reciprocal group. The
RI two-electron block in the normalization used here is

\[
 V_{ij}^{ab}(\mathbf k_i,\mathbf k_j,\mathbf k_a)
 =\frac1{N_k}\sum_P
 L^{\mathbf k_i\mathbf k_a}_{Pia}
 L^{\mathbf k_j\mathbf k_b}_{Pjb}.
\]

With

\[
 \Delta_{ij}^{ab}=\varepsilon_{i\mathbf k_i}
 +\varepsilon_{j\mathbf k_j}-\varepsilon_{a\mathbf k_a}
 -\varepsilon_{b\mathbf k_b},
\]

the closed-shell spin components per primitive cell are

\[
 E_{\mathrm{OS}}^{(2)}=\frac1{N_k}
 \sum_{\mathbf k_i\mathbf k_j\mathbf k_a}\sum_{ijab}
 \frac{|V_{ij}^{ab}(\mathbf k_a)|^2}{\Delta_{ij}^{ab}},
\]

\[
 E_{\mathrm{SS}}^{(2)}=\frac1{N_k}
 \sum_{\mathbf k_i\mathbf k_j\mathbf k_a}\sum_{ijab}
 \frac{|V_{ij}^{ab}(\mathbf k_a)|^2
 -V_{ij}^{ab}(\mathbf k_a)^{*}V_{ij}^{ba}(\mathbf k_b)}
 {\Delta_{ij}^{ab}},
\]

and (E_{\mathrm{MP2}}^{(2)}=E_{\mathrm{OS}}^{(2)}+
E_{\mathrm{SS}}^{(2)}). MP2 is not variational: it is the second-order
Rayleigh--Schrodinger correction for the finite RI Hamiltonian around the
stationary RHF determinant. The finite-group Fourier transform nevertheless
makes it exactly equivalent to canonical RI-MP2 in the corresponding BvK
supercell when the same Coulomb kernel, exchange q=0 convention, and fit are
used.

The implemented 3D H2/STO-3G two-point mesh agrees with an out-of-process
PySCF KRHF/KMP2 calculation to (5.9\times10^{-10}) Ha in HF and
(2.4\times10^{-10}) Ha in correlation energy per cell. The machine-readable
record is `docs/manuscripts/aiccm_comparison/data/h2_mp2_2026-06-21.json`.
MP2 currently fails closed in 1D and 2D until O1 supplies a common
lower-dimensional Coulomb gauge.

### Unrestricted MP2

For an unrestricted reference, occupied and virtual labels carry an explicit
spin. With antisymmetrized same-spin integrals and ordinary opposite-spin
integrals, the second-order correction is

\[
 E_{\mathrm{UMP2}}^{(2)}=E_{\alpha\alpha}^{(2)}
 +E_{\beta\beta}^{(2)}+E_{\alpha\beta}^{(2)},
\]

\[
 E_{\sigma\sigma}^{(2)}=
 \frac14\sum_{ij\in\sigma}\sum_{ab\in\sigma}
 \frac{|\langle i_\sigma j_\sigma\Vert
 a_\sigma b_\sigma\rangle|^2}
 {\varepsilon_i^\sigma+\varepsilon_j^\sigma
 -\varepsilon_a^\sigma-\varepsilon_b^\sigma},
\]

\[
 E_{\alpha\beta}^{(2)}=
 \sum_{i a\in\alpha}\sum_{j b\in\beta}
 \frac{|(i_\alpha a_\alpha|j_\beta b_\beta)|^2}
 {\varepsilon_i^\alpha+\varepsilon_j^\beta
 -\varepsilon_a^\alpha-\varepsilon_b^\beta}.
\]

Every sum also contains the finite characters, with the same momentum
conservation and one final division by \(N_c\) as the restricted expression.
The implementation evaluates this equation after the exact inverse transform
of the χ-CCM RI Hamiltonian. Zero PNO threshold and canonical occupieds give the
full UMP2 result. Independent unitary localization of the alpha and beta
occupied projectors leaves both spin densities invariant; the coupled local
MP2 equations are then required because each localized occupied Fock block is
not diagonal.

## 8. Real-torus local correlation and coupled cluster

### Exact inverse transform of the fitted Hamiltonian

Let the normalized finite-group Bloch AO be

\[
 |\mu\mathbf{k}\rangle
 =N_c^{-1/2}\sum_{\mathbf R\in\mathcal T_N}
 e^{+i\mathbf k\cdot\mathbf R}|\mu\mathbf R\rangle .
\]

For any translation-invariant one-particle matrix, the full finite-torus AO
matrix is therefore

\[
 X_{\mu\mathbf R,\nu\mathbf S}
 =\frac{1}{N_c}\sum_{\mathbf k}
 e^{+i\mathbf k\cdot\mathbf R}X_{\mu\nu}(\mathbf k)
 e^{-i\mathbf k\cdot\mathbf S}.
\]

This formula is used for both overlap and Fock. It is a unitary change of
representation, not a new Gamma-point SCF.

The RI factors need one more character. Let
\(\widetilde L^{\mathbf k_i\mathbf k_a}_{P,\mu\nu}\) be the canonical
auxiliary-AO factor of Section 7 and
\(\mathbf q=\mathbf k_a-\mathbf k_i\). Define

\[
B_{P\mathbf T,\mu\mathbf R,\nu\mathbf S}
=\frac{1}{N_c^2}\sum_{\mathbf k_i\mathbf k_a}
 e^{+i\mathbf k_i\cdot\mathbf R}
 e^{-i\mathbf k_a\cdot\mathbf S}
 e^{+i\mathbf q\cdot\mathbf T}
 \widetilde L^{\mathbf k_i\mathbf k_a}_{P,\mu\nu}.
\]

Translation covariance makes the auxiliary-cell index exactly redundant at
the AO-factor storage level. Defining the home-auxiliary representative

\[
 H_{P,\mu\mathbf R,\nu\mathbf S}
 =B_{P\mathbf 0,\mu\mathbf R,\nu\mathbf S},
\]

the preceding equation gives

\[
 B_{P\mathbf T,\mu\mathbf R,\nu\mathbf S}
 =H_{P,\mu(\mathbf R-\mathbf T),\nu(\mathbf S-\mathbf T)}.
\]

D111 therefore retains only \(H\). If \(A\) is the primitive auxiliary rank
and \(M\) the primitive AO rank, its shape is
\((A,N_cM,N_cM)\), rather than the logical dense shape
\((N_cA,N_cM,N_cM)\). Retained real AO-factor storage falls exactly from
\(8AN_c^3M^2\) to \(8AN_c^2M^2\) bytes. The identity requires a complete
cyclic translation group and its dual character mesh, plus one common full
primitive-auxiliary \(P\) frame for every momentum pair. The production
builder enforces the latter with `canonical_auxiliary_basis=True`; a reduced,
q-dependent auxiliary-eigenvector frame cannot use this representation.

The logical auxiliary orbit is reconstructed only during AO-to-MO
transformation. Changing variables in the contraction shows that auxiliary
translation \(\mathbf T\) uses coefficient rows
\(C_{\mu\mathbf R,p}\mapsto C_{\mu(\mathbf R+\mathbf T),p}\). The native
circulant kernel applies that shift and returns the unchanged logical output
shape \((N_cA,n_{\rm left},n_{\rm right})\). It supports the ordinary real
transpose and the complex conjugate-left transform independently, so an odd
mesh pins both the translation sign and complex convention. The production
boundary rejects incomplete translation lists, nondual or duplicate
characters, nonfinite or out-of-range character coordinates, nonpositive
meshes, and overflowing mesh products before discarding any translated row.
The old dense inverse transform remains only as a small regression oracle.
No home tensor is exposed under `.B` or `.three_center`, whose former shape
meant the full logical auxiliary family, and no unused identity metric is
allocated. The complete-space MP2 correction also consumes the exact
circulant transform rather than reconstructing a dense AO factor.

The corresponding real-torus MO coefficient is

\[
 C_{\mu\mathbf R,p\mathbf k}
 =N_c^{-1/2}e^{+i\mathbf k\cdot\mathbf R}C_{\mu p}(\mathbf k).
\]

Contracting the previous two equations gives

\[
 B_{P\mathbf T,i\mathbf k_i,a\mathbf k_a}
 =N_c^{-1}e^{+i\mathbf q\cdot\mathbf T}
 L^{\mathbf k_i\mathbf k_a}_{Pia}.
\]

Thus a product of factors with transfers \(\mathbf q\) and \(-\mathbf q\)
satisfies

\[
 \sum_{P\mathbf T}B_{P\mathbf T,ia}B_{P\mathbf T,jb}
 =\frac{1}{N_c}\sum_P L_{Pia}L_{Pjb},
\]

while any nonconserving transfer vanishes by character orthogonality. This is
exactly the normalization in the momentum-space MP2 integral. The real and
character representations therefore describe the same χ-defined fitted finite
Hamiltonian at the recorded Coulomb and exchange q=0 convention. This internal
Fourier equivalence does not assert equality with the union-and-weight Γ-CCM
approach.

The D111 storage reduction does not remove the \(N_c^2\) momentum-pair cache
or the full logical MO factors, and the correlated solvers still evaluate all
pairs and ordered triples. It is an exact setup-memory and inverse-transform
change, not a local-domain, screening, representative-amplitude, or
reduced-scaling correlation approximation.

### Production target: MgO on the 8 x 8 x 8 finite translation group

The long-range scaling target is a primitive-cell MgO calculation on the
finite translation group

\[
 \mathcal T_8 = \mathbb Z_8\times\mathbb Z_8\times\mathbb Z_8,
 \qquad N_c=512.
\]

The χ calculation uses the complete dual character mesh
\(\widehat{\mathcal T}_8\).

Calling this "8 x 8 x 8 equivalent" requires more than matching the number
of cells. The primitive lattice and basis, complete character set and uniform
weights, charge and spin, frozen-core selection, auxiliary basis and metric,
Coulomb convention, exchange q=0 convention, and triples definition must all
be identical to the finite-k reference. Character and real-torus forms are
unitarily related representations of this one χ Hamiltonian. A truncated
DLPNO calculation is an accuracy-controlled approximation to it, not an exact
canonical CCSD(T) calculation merely because the mesh matches.

For the working frozen-core count of four active occupied bands per primitive
cell, the full torus contains 2,048 active occupied orbitals and 2,098,176
unordered placed pairs. Store the representative pair

\[
 p_{ab\mathbf L}=\{(a,\mathbf 0),(b,\mathbf L)\}
\]

and its orbit

\[
 \mathcal O_{ab\mathbf L}
 =\left\{\{(a,\mathbf T),(b,\mathbf T+\mathbf L)\}:\mathbf T\in\mathcal T_8\right\}
\]

for each orbit under simultaneous translation. Unordered-pair exchange gives

\[
 (a,b,\mathbf L)\sim(b,a,-\mathbf L),
 \qquad
 w_{\mathcal O}=\frac{|\mathcal O|}{N_c}
 =\frac{1}{|\operatorname{Stab}(p_{ab\mathbf L})|}.
\]

The even mesh makes the stabilizer explicit. There are seven nonzero
translations satisfying \(2\mathbf L=0\). For each of the four same-band
families, its seven antipodal representatives have multiplicity 256 and
per-cell weight one half. Home diagonals have multiplicity 512 and weight one.
All other representatives have multiplicity 512 and weight one. D123
therefore stores 4,112 compact representatives, including 28 half-weight
antipodal rows, with a weighted per-cell census of 4,098. Expanding the plan
reproduces the existing full pair-orbit closure on small odd, even, and
anisotropic meshes. Production must consume these representatives directly;
reconstructing their members would recreate the quadratic allocation the plan
removes.

Translation is the mandatory first quotient, not the endpoint. Rocksalt MgO's
cubic point group is the largest exact scaling lever after it: a generic
translation-pair orbit can in principle inherit a large point-symmetry
multiplicity before any numerical screening is applied. The realised factor
cannot be taken as the point-group order, however. The localized occupied
gauge, atomic and auxiliary domains, PNO/TNO spaces, and amplitudes transform
with orbital representation matrices and may have nontrivial little groups.
The executable contract must therefore derive those covariant maps, compute
each combined stabilizer, and validate expansion against the unreduced
translation plan. Reducing displacement vectors geometrically without the
orbital and domain maps is forbidden. Only after this exact symmetry quotient
do distance, pair-energy, and occupation thresholds remove numerical work.

D126 implements the compact support-orbit algebra for a caller-asserted
monomial support skeleton while keeping that physical boundary explicit. Let
`pi_g` be a caller-supplied source-to-target permutation of the complete
occupied labels. Before it may act on a D123 row, the planner proves that
every `pi_g` is a permutation and normalizes the finite pure-translation
subgroup. It can then induce an exact permutation of the D123 row set,

\[
  [p_{ab\mathbf L}] \longmapsto [\pi_g p_{ab\mathbf L}],
\]

where brackets denote the already-formed simultaneous-translation orbit. For
a point orbit `P` of D123 rows, the combined placed-pair multiplicity and
per-cell weight are

\[
  M_{\mathcal P}=\sum_{r\in\mathcal P}m_r,
  \qquad w_{\mathcal P}=M_{\mathcal P}/N_c.
\]

The factory-only result stores only D123 row indices, offsets, and combined
multiplicities. It never reconstructs the placed-pair space. On the idealized
symmetry-adapted MgO `A1g + T1u` support basis (`s, p_x, p_y, p_z`), the
support permutations induced by the 48 signed-axis operations of `O_h` reduce
4,112 D123 rows to 260 work rows while keeping the exact 2,098,176 placed-pair
and 4,098 per-cell censuses. This number is an algebraic support-orbit ceiling,
not a measured speedup or an attestation of a physical monomial gauge.

The qualification matters particularly for MgO. Casassa, Zicovich-Wilson,
and Pisani, *Theor. Chem. Acc.* **116**, 726 (2006),
doi:10.1007/s00214-006-0119-z, explain that Boys-localized cubic-anion LWFs
may be arbitrary `sp3` combinations. Their procedure subsequently transforms
such LWFs into symmetry-adapted localized Wannier functions organized as
petals, flowers, and bunches. The representation \(W^R\) within a flower is
generally matrix-valued; their favourable single-petal case \(W=\pm1\) is a
special limit, so a flower is not itself synonymous with a dense action.
D126 therefore accepts only a caller-asserted scalar support permutation and
supplies no physical attestation itself. Its support-generator fingerprint
carries neither orbital signs nor phases and does not prove group
multiplication or localized-orbital covariance. A subsequent χ-owned
covariance contract must either construct and attest a monomial gauge or
carry the general matrix-valued petal/flower representation, cell-shift
cocycle, phases, and transformations of PAO/auxiliary/PNO/TNO domains and
amplitudes before representative execution is admissible.

D127 now supplies the AO-level spatial action prerequisite from which that
physical covariance must be derived. The symmetry plan stores wrapped
fractional atom coordinates \(\mathbf f_a\) and shifts

\[
 W_g\mathbf f_a+\mathbf w_g
 =\mathbf f_{\pi_g(a)}+\mathbf q_{g,a}.
\]

The supplied `PeriodicSystem` may instead use an equivalent representative
\(\widetilde{\mathbf f}_a=\mathbf f_a+\mathbf n_a\), with
\(\mathbf n_a\in\mathbb Z^3\). The factory recovers and fingerprints those
reference-cell offsets and acts with the corrected shift

\[
 \widetilde{\mathbf q}_{g,a}
 =\mathbf q_{g,a}+W_g\mathbf n_a-\mathbf n_{\pi_g(a)}.
\]

For a primitive AO \(\mu\) on atom \(a(\mu)\), source cell \(\mathbf R\),
and destination AO \(\nu\), define

\[
 [\mathcal U_g]_{(\mathbf S,\nu),(\mathbf R,\mu)}
 =P^g_{\nu\mu}\,
 \delta_{\mathbf S,\,
 W_g\mathbf R+\widetilde{\mathbf q}_{g,a(\mu)}
 \pmod{\mathbf N}} .
\]

The primitive matrix convention is `P[destination, source]`; consequently
the real-space operation is an active `+q` scatter. This is distinct
from helper APIs that record the opposite bring-back-to-cell shift; D127 uses
those helpers only for the primitive AO rotation and owns the χ cell routing.

The action factory also imposes an independent
\(10^{-5}\)-bohr maximum on the recomputed atom-mapping residual. The current
symmetry-plan payload records the observed residual but not the `symprec`
that admitted it. Consequently a diagnostic plan built with a looser
`symprec` can remain valid as a plan while its operation is rejected by the
D127 action factory. Finite-mesh compatibility alone is not an
action-admission certificate.

With cell blocks in C order, the compact action retains one primitive
\(n_{\rm AO}\) squared matrix and one \(N_c\) by \(n_{\rm atom}\) cell-image
table. It therefore avoids the squared \(N_c n_{\rm AO}\) torus operator.
The action and its convention-critical inputs, including the atom reference
offsets and corrected shifts, are copied to immutable bytes-backed arrays and
fingerprinted. That digest identifies the compact
action payload only. It does not bind the system lattice, atomic positions or
species as independent inputs; radial exponents or contraction coefficients;
localized coefficients or overlap; a source/run/build identity; or a stable
spglib operation identifier. In particular, the stored operation indices may
change if an equivalent operation list is enumerated in a different order.

The factory verifies basis-shell origins against their system atoms and exact
ordered radial-shell equality across every mapped atom. Those radial values
are deliberately absent from the action payload and its fingerprint. The
supported angular envelope is scalar orbital AOs with spherical
\(l\geq1\) shells; Cartesian \(p\)-and-higher shells fail closed, while the
Cartesian/spherical distinction is immaterial for \(s\). No action is defined
here for spinors, auxiliary functions, PAOs, PNOs, or TNOs. Applying
\(\mathcal U_g\) acts on AO rows only. It does not decide how localized-
orbital columns, their own reference offsets, centres, phases, or domains
transform. Application retains the result plus atom-block-sized temporaries
while still avoiding the dense torus operator.

For two affine representatives, with \(h\) applied first,

\[
 W_{gh}=W_gW_h,\qquad
 \pi_{gh}=\pi_g\!\circ\!\pi_h,\qquad
 \widetilde{\mathbf q}_{gh,a}
 =W_g\widetilde{\mathbf q}_{h,a}
 +\widetilde{\mathbf q}_{g,\pi_h(a)},\qquad
 P_{gh}=P_gP_h.
\]

Space-group representatives returned modulo a primitive translation need not
close without that translation. Let \(k\) denote the stored representative
with \(W_k=W_gW_h\) and fractional translation \(\mathbf w_k\). Then

\[
 \boldsymbol\ell_{gh}=W_g\mathbf w_h+\mathbf w_g-\mathbf w_k
 \in\mathbb Z^3,
 \qquad
 \mathcal U_g\mathcal U_h
 =T_{\boldsymbol\ell_{gh}}\mathcal U_k.
\]

Thus \(\boldsymbol\ell_{gh}\) is the integer translation factor, or cocycle,
of the chosen coset representatives; it is not an extra Coulomb or phase
convention. D127 retains the atom shifts needed to derive it but does not yet
publish this group table or representation seal. A diamond fixture pins one
nonzero instance of the displayed law, not the complete group table.
Likewise, under the localization convention
\(v_{\mathbf R\mu}=e^{+2\pi i\mathbf k\cdot\mathbf R}c_\mu(\mathbf k)\),
the action sends \(\mathbf k\) to \(\mathbf k'=W_g^{-T}\mathbf k\) with
primitive sewing phase

\[
 B_g(\mathbf k)_{\nu\mu}
 =P^g_{\nu\mu}
 e^{-2\pi i\mathbf k'\cdot
 \widetilde{\mathbf q}_{g,a(\mu)}}.
\]

This real-space definition implies that sign and phase formula for the next
χ-owned milestone; D127 neither emits nor tests a Bloch sewing matrix. That
milestone must form the occupied sewing from the actual localized
coefficients and torus overlap, including the column/reference-offset map,
prove the full affine group/cocycle law, and classify the result as monomial
or general matrix-valued. Only a certified signed/phased monomial action may
supply D126's scalar support permutation. A general petal/flower action
instead requires matrix-valued domain and amplitude transport. D127 does not
perform Casassa's a posteriori SALWF construction or classification, is not
connected to SCF or correlation execution, and changes no energy or workload.

The dense-small tests materialize the torus action from the same trusted
primitive AO rotation, so they validate D127's cell scatter and compact
factorization rather than independently rederive the real-spherical rotation.
The skew-cell covariance fixture uses a deliberately symmetrized synthetic
metric, not the χ torus overlap; contraction with the physical overlap is the
next gate described above.

The present code is still far outside that production envelope. For the
current planning ranks of 37 AOs and 189 provisional auxiliary functions per
primitive cell, the D111 home factor alone is about 0.49 TiB at 8 cubed, and
the preceding momentum-pair cache brings setup to roughly 1.48 TiB. The
auxiliary choice is a sizing proxy, not yet a validated MgO correlation-fit
basis. Passing the logical factor to the molecular solver would then request
about 19.9 TiB for occupied-virtual, 3.0 TiB for occupied-occupied, and
134.6 TiB for virtual-virtual factors. No increase in a single-node memory
request makes that algorithm suitable. A production path must make all of
those global shapes impossible by construction.

### Direct reference and screened local-correlation architecture

"Direct" describes storage and execution: do not retain a global four-center
ERI or three-center MO tensor. It is separate from the operator-realisation
axis `four_center`, RI-JK, or RIJCOSX. Thus a streamed four-center route and a
streamed RI-JK route can both be direct, but one cannot silently satisfy a
request for the other's operator. The intended production reference is
streamed RI-JK: generate screened AO and auxiliary shell batches, apply the
declared periodic metric, contract them into J and K, and release each batch.
The high-G wrong-answer and memory failures tracked by issues #121 and #142
must be closed before that fitted route is called production-qualified. The
four-center BIPOLE route remains the small-mesh operator oracle. RIJCOSX may
be a useful faster reference, but its numerical quadrature is an additional
approximation and must carry its own convergence ladder against RI-JK. It
cannot silently satisfy an RI-JK equivalence claim.

The required locality hierarchy is:

1. Apply shell-pair Schwarz and density bounds before integral generation.
2. Build a k-local covariant occupied gauge and store only home-cell Wannier
   or IAO data plus finite translations.
3. Select PAO domains with a periodic differential-overlap-integral (DOI) or
   an independently bounded population criterion, using exact minimum-image
   geometry.
4. Form pair-local auxiliary domains and stream only the requested
   three-center blocks. Occupied-virtual transition densities are chargeless,
   so each local auxiliary domain must project out its charged component, or
   equivalently constrain the fitted transition density to zero total charge.
   The surface/dipole term must be derived for the declared χ finite-character
   Coulomb operator; the infinite-lattice neutral-GDF correction is not
   imported by analogy.
5. Use the PAO to OSV to PNO cascade. The pair estimate selects strong, weak,
   and distant work; the PNO occupation threshold controls the retained pair
   virtual rank, and the discarded-space MP2 correction remains explicit.
6. Build a sparse occupied-coupling graph for CCSD from minimum-image distance,
   DOI penetration, and pair-energy evidence. Distance alone is not an error
   bound. The long-range correlation tail decays algebraically and must be
   bounded or evaluated by an explicit asymptotic correction.
7. Generate triple representatives from the strong-pair graph, then build
   triple-specific natural orbitals. Local `(T0)` is an oracle and screening
   rung; the production target is the iterative local `(T1)` definition or a
   separately proved equivalent that retains off-diagonal occupied-Fock
   effects.

Every approximation has an independent error coordinate: SCF integral
support, RI metric, local auxiliary domain, pair omission, PAO/OSV/PNO
truncation, CC residual, TNO/triples truncation, and finite mesh. A named
threshold bundle may set those coordinates, but it must not replace their
individual values in provenance. The simultaneous zero-threshold, full-domain,
all-representative limit must reproduce canonical χ MP2, CCSD, and triples on
the small meshes where those oracles fit.

MPI ownership is by compact pair or triple representative, costed by domain
and PNO/TNO ranks. Integral q/k/AO tiles are streamed to those owners; factors,
amplitudes, and DIIS histories are not globally all-gathered. Checkpoints are
pair/triple keyed so a wall-time continuation does not replay completed work.
The common periodic-local resource planner may admit a run only after both its
static dimensions and measured domain census fit RAM and scratch budgets.

GPU acceleration is viable only after this streaming boundary exists. The
useful kernels are batched AO-to-local three-index transforms, pair-density
formation, PNO/TNO diagonalization, pair-local CCSD GEMMs, and triples
contractions. Use one MPI rank per GPU, bucket similarly sized domains, retain
one pair or triple through several contractions, and overlap a CPU Libint
producer with GPU consumers. Sending individual tiny pair operations across
PCIe would lose the advantage. Published GPU results establish useful
prefactors for sparse periodic RI and molecular RI-CCSD(T), not an end-to-end
periodic DLPNO-CCSD(T) timing; the 8 cubed resource estimate must come from the
measured 3 cubed through 6 cubed χ ladder.

ORCA's SHARK engine is useful architectural literature for screened shell
batches, integral digestion, and producer-consumer scheduling, but it is not a
theoretical dependency. vibe-qc will implement the required machinery on its
own Libint-based C++ core. ORCA, VASP, Turbomole, and other QC programs remain
out-of-process validation references only.

The implementation and validation order is fixed as follows:

1. compact pair and triple group plans, including even-mesh stabilizers;
2. covariant k-local occupied gauge and its exact little-group action;
3. combined translation/point-symmetry representatives plus minimum-image
   domain providers;
4. streamed charge-constrained local RI and representative DLPNO-MP2;
5. MgO 3 cubed and 5 cubed periodic DLPNO-MP2 reproduction;
6. sparse representative CCSD with 1 cubed and 2 cubed canonical parity;
7. graph-generated TNO triples, first `(T0)` then iterative `(T1)`;
8. distributed checkpoints and CPU scaling;
9. batched optional GPU kernels; and
10. MgO 3 cubed, 4 cubed, 5 cubed, 6 cubed, then 8 cubed production admission.

The closest published anchors are periodic BvK-DLPNO-MP2 for MgO
([Nejad et al., 2025](https://doi.org/10.1063/5.0290816) and
[Zhu et al., 2025](https://doi.org/10.1063/5.0290819)), periodic
LNO-CCSD(T) with k-point symmetry
([Ye and Berkelbach, 2024](https://doi.org/10.1021/acs.jctc.4c00936)), and
embedded periodic CIM-DLPNO-CCSD(T)
([Wang et al., 2022](https://doi.org/10.1021/acs.jctc.2c00412)). The last is
not exact finite-8-cubed parity. The published MgO DLPNO calculation reaches
7 cubed and 12,691 AOs at MP2, not CCSD(T); it is a validation target rather
than a defensible CCSD(T) runtime extrapolation. Sparse periodic RI GPU work
([Bussy et al., 2023](https://doi.org/10.1063/5.0144493)) and molecular
GPU RI-CCSD(T) work
([Datta and Gordon, 2023](https://doi.org/10.1021/acs.jctc.3c00876)) motivate
the accelerator boundary but do not change that distinction. SHARK itself is
described by [Neese, 2022](https://doi.org/10.1002/jcc.26942).

### Local MP2 and CCSD(T)

The present local implementation deliberately builds the whole finite torus.
It includes every occupied pair and every ordered occupied triple required by
the spin-integrated equations, obtains a total correlation energy
\(E_{\mathrm{corr},N}\), and reports

\[
 E_{\mathrm{corr}}^{\mathrm{cell}}
 =\frac{E_{\mathrm{corr},N}}{N_c}.
\]

There is no additional boundary or pair multiplicity. A future translation-
reduced implementation may retain one orbit under simultaneous translation
of all occupied indices, but then its multiplicity must follow the
orbit-stabilizer theorem. Selecting visually unique pairs and multiplying by
an occurrence factor would recreate the historical ambiguity.

Occupied orbitals are localized with the Pipek--Mezey objective built from
Mulliken populations. This is invariant to the choice of position origin and
does not use the ill-defined ordinary position operator on a torus. Molecular
Foster--Boys localization is rejected. The occupied rotation is unitary, so
with complete pair domains, no PNO truncation, no pair screening, and complete
occupied coupling, local MP2 and local CCSD reproduce their canonical
finite-torus limits. With complete triple virtual spaces the same statement
holds for the perturbative triples correction.

The current defaults allow PNO truncation but keep all occupied pairs and all
occupied couplings. Distance-based pair screening and local auxiliary fitting
are disabled because their existing molecular implementations use Euclidean
distances across the cell boundary. They require minimum-image Wannier and
auxiliary domains before they can be enabled without breaking translation
symmetry.

For an explicit local-(T0) triples calculation with positive `tcut_tno`, the
native TNO occupation-density builder and the native amplitude projector must
consume the same complete ordered occupied-pair inventory. The inventory is
constructed outside the per-triple domain builder so neither native path can
depend on branch-local initialization. Native density formation is pinned to
the NumPy equation by energy and retained-domain parity. This issue-298
contract changes neither the TNO occupation threshold nor the default `(T1)`
triples route.

The local-correlation result records the same finite-torus convention as its
HF reference. This is required because the semicanonical occupied and virtual
energy blocks used in MP2, CCSD, triples, and future DLPNO denominators inherit
the exchange q=0 seam from that reference.

The gauge-invariant SCF property object, the finite-torus band-structure
wrapper, and the finite-torus Mayer bond-order analysis also record that
convention. These analyses are derived from the same one-particle Fock,
density, and overlap matrices; the descriptor is provenance for the finite
Hamiltonian, not an additional factor in the population formula or band
interpolation.

Neither MP2 nor CCSD(T) is variational. CCSD is stationary only through its
left-right coupled-cluster Lagrangian, and the perturbative triples energy has
no upper-bound property. PNO/domain truncations add controlled numerical
approximations, not a new Hamiltonian or a variational subspace theorem.

On the H2/STO-3G two-cell control, no-truncation real-torus local MP2 equals
the character-space result within \(7\times10^{-18}\) hartree per cell. On the
LiH/STO-3G two-cell control, no-truncation local and canonical finite-torus
DF-CCSD correlation energies differ by \(6.9\times10^{-12}\) hartree total;
their nonzero triples corrections differ by \(1.4\times10^{-12}\) hartree.
These are finite-Hamiltonian exact-limit tests, not external infinite-crystal
CCSD(T) benchmarks.

For UCCSD, use spin orbitals and

\[
 |\Psi_{\mathrm{CCSD}}\rangle=e^{T_1+T_2}|\Phi_0\rangle,
\qquad
 \langle\Phi_\mu|e^{-T}He^T|\Phi_0\rangle=0
\]

for every single and double excitation \(\mu\). The energy is

\[
 E_{\mathrm{CCSD}}=\langle\Phi_0|e^{-T}He^T|\Phi_0\rangle.
\]

The unrestricted PNO pilot expands each pair amplitude into the full
finite-torus spin-orbital virtual space, evaluates the complete projected
residual, and projects back. With zero PNO threshold this projection is a
unitary coordinate change and recovers full-domain UCCSD. Perturbative triples
are evaluated from the converged amplitudes. This establishes the equations
and exact PNO limit, but the current implementation remains an explicitly
cost-capped O(N^6) oracle: it does not yet skip translation-equivalent pairs
or triples and therefore does not yet establish reduced scaling.

### Post-SCF occupied localization

Let the complete finite character group contain \(N_c\) points. For each
isolated occupied band, the canonical back transform is

\[
 |w_{n\mathbf R}^{0}\rangle
 =\frac{1}{\sqrt{N_c}}\sum_{\mathbf k}
 e^{-i\mathbf k\cdot\mathbf R}|\psi_{n\mathbf k}\rangle .
\]

All \(N_c n_{\mathrm{occ}}\) functions are rotated together,
\(C^{\mathrm{loc}}=C^{0}U\), with \(U^\dagger U=I\). Consequently

\[
 C^{\mathrm{loc}}C^{\mathrm{loc}\dagger}
 =C^0UU^\dagger C^{0\dagger}=C^0C^{0\dagger}.
\]

This proves invariance of the finite-torus density and every RHF or RKS
energy term. Localizing independent k blocks would not produce the required
real-space family.

D100 evaluates the corresponding numerical audits without forming either
full AO projector. For \(P_i=C_iC_i^\dagger\),

\[
 \|P_1-P_0\|_F^2
 =\|C_1^\dagger C_1\|_F^2+\|C_0^\dagger C_0\|_F^2
 -2\|C_1^\dagger C_0\|_F^2,
\]

and the one-particle check uses

\[
 \operatorname{Tr}[(P_1-P_0)F]
 =\operatorname{Tr}(C_1^\dagger F C_1)
 -\operatorname{Tr}(C_0^\dagger F C_0).
\]

For a cyclic AO-row permutation \(T\), translation covariance is checked from

\[
 \|TP_1T^\dagger-P_1\|_F^2
 =2\|C_1^\dagger C_1\|_F^2
 -2\|(TC_1)^\dagger C_1\|_F^2.
\]

The native kernel keeps only coefficient and occupied-space work arrays,
reducing diagnostic scratch from \(O(n_{\mathrm{AO}}^2)\) to
\(O(n_{\mathrm{AO}}n_{\mathrm{occ}}+n_{\mathrm{occ}}^2)\). The earlier dense
projector equations remain a private parity oracle. When the Gram-norm
subtraction is close enough to exact cancellation to lose relative precision,
the native path streams the AO-pair projector difference and accumulates its
norm without storing the matrix. This changes only how the three scalar
diagnostics are evaluated; the localization rotation and every downstream PNO
or correlated quantity are unchanged.

Exact Marzari--Vanderbilt localization requires cross-k matrix elements of
the exponential position operator. Those integrals are not exposed by the
current native API. The implemented Wannier route jointly diagonalizes
Lowdin-projected AO-centre operators \(\cos(2\pi r_\alpha/L_\alpha)\) and
\(\sin(2\pi r_\alpha/L_\alpha)\). If

\[
 z_{n\alpha}=\langle w_n|e^{2\pi i r_\alpha/L_\alpha}|w_n\rangle,
\qquad
 \Omega_{n\alpha}
 =-\left(\frac{L_\alpha}{2\pi}\right)^2\log|z_{n\alpha}|^2,
\]

the latter is the reported circular spread. It has the correct torus
topology but is labelled a projected circular AO-centre approximation.
Exact continuum spreads and Wannier90 parity remain open.

The IAO alternative constructs polarized intrinsic atomic orbitals from a
minimal reference and maximizes squared atomic populations. Its present
cross overlap is an explicit-supercell integral rather than a periodized
torus integral. A four-cell H2 run gives translation covariance residuals of
\(1.17\times10^{-13}\) for the Wannier path and \(7.08\times10^{-6}\) for
IAO. Density invariance remains at machine precision. IAO translation maps
are therefore diagnostic and are not used to skip correlated pairs.

Wraparound is flagged when the antipodal-shell probability exceeds 0.05 or
when \(\sqrt{\Omega_n}\) exceeds one quarter of the shortest cyclic-cluster
vector. A nonpositive indirect gap fails closed. Metallic-band
disentanglement is not implemented.

### PAOs, PNOs, and pair orbits

For an \(S\)-orthonormal occupied coefficient matrix \(C_o\), the coefficient
projector out of the occupied space is

\[
 Q=I-C_oC_o^\dagger S,\qquad C_o^\dagger S Q=0.
\]

If \(E_D\) selects AOs in a pair domain, let
\(\widetilde V_D=QE_D\) and diagonalize its Gram matrix:

\[
 G_D=\widetilde V_D^\dagger S\widetilde V_D=XgX^\dagger,
\qquad
 V_D=\widetilde V_DX_+g_+^{-1/2}.
\]

Then \(V_D^\dagger S V_D=I\) and \(C_o^\dagger S V_D=0\). This removes both
occupied leakage and redundant PAOs.

For pair amplitude matrix \(T_{ij}\), the model pair density is

\[
 D_{ij}=\frac{T_{ij}T_{ij}^\dagger+T_{ij}^\dagger T_{ij}}
 {1+\delta_{ij}}.
\]

It is Hermitian positive semidefinite because
\(x^\dagger D_{ij}x=(\|T_{ij}^\dagger x\|^2+
\|T_{ij}x\|^2)/(1+\delta_{ij})\ge0\). Its eigenvectors are PNOs. A zero
threshold retains the complete PAO span, and positive thresholds give nested
ranks. This proves rank nesting, not monotonic MP2 energy error: MP2 is
nonvariational and convergence of the energy must be measured.

Translations act simultaneously on both indices of an unordered occupied
pair. For measured localization permutations \(p_g\),

\[
 \mathcal O(i,j)=\{\operatorname{sort}(p_g(i),p_g(j)):g\in G_N\}.
\]

The per-cell coefficient is \(|\mathcal O|/N_c\), including smaller orbits
with nontrivial stabilizers such as half-cell pairs on an even ring. The code
verifies a disjoint partition of all unordered pairs, but it still evaluates
all pairs. The orbit reduction factor is not a timing claim.

At complete PAO domains and zero PNO thresholds, the B MP2 wrapper also
evaluates a rotation-invariant full-space DF-MP2 contraction. The raw
iterative local-pair energy and the correction are both reported. Before the
real local solver, a full-rank real time-reversal gauge is constructed and
its occupied projector is checked against the complex localized projector.
This avoids silently discarding imaginary coefficients. No correction is
applied to truncated calculations.

### Finite-cluster space-group symmetry

Write primitive lattice vectors as columns of \(A\), and let
\(N=\operatorname{diag}(N_1,N_2,N_3)\). A crystallographic operation
\(g=\{W|\mathbf w\}\) preserves the finite torus exactly when

\[
 N^{-1}WN\in\mathbb Z^{3\times3}.
\]

For primitive atom \(a\), retain both the atom permutation and integer image
shift,

\[
 W\mathbf f_a+\mathbf w=\mathbf f_{p_g(a)}+\mathbf q_{g,a},
\]

so \((a,\mathbf r)\) maps to

\[
 \left(p_g(a),W\mathbf r+\mathbf q_{g,a}\pmod{\mathbf N}\right).
\]

The image shift is essential for screw and glide operations. Reciprocal
characters transform as \(\mathbf k'=W^{-T}\mathbf k\), and a k-orbit has
weight \(|\mathcal O_k|/N_c\). spglib supplies the crystallographic group;
the B module builds its exact cluster-compatible subgroup.

Symmetry is diagnostic only. At Gamma, AO matrices can be checked with the
Reynolds projector

\[
 \mathcal P_G[M]=\frac{1}{|G_N|}\sum_g U_g M U_g^T,
\]

whose idempotency follows from group closure. At general k, nonsymmorphic
sewing phases depend on each backend's Bloch convention. Therefore
`symmetry_mode="integrals"` fails closed until representative shell-pair and
quartet scattering passes full-build Fock and energy parity. Primitive shell
pairs and intrinsically eightfold-unique shell quartets are already
partitioned into exact point-group orbits and reported as reduction counts;
no libint call is skipped from those counts yet.

## 9. Limits and convergence

### Exact finite-size identity

For the same declared finite-torus Hamiltonian, in particular the same
`exchange_q0="bvk-ewald"` finite-size convention, and the same
translation-invariant variational space,

\[
 E_N^{\Gamma\text{-supercell}}/N_c
 =E_N^{\Gamma\text{-centred mesh}}
\]

up to numerical error. This is the strongest internal reference and should
hold at every mesh size.

### Infinite-crystal limit

As all active \(N_i\to\infty\), the character sum approaches the Brillouin
zone integral and the BvK Green's function approaches the chosen infinite
periodic Coulomb convention. Insulators with localised density matrices are
expected to converge rapidly in short-range pieces; exact exchange and the
Coulomb finite-size terms can converge algebraically. The sequence need not
be monotone because each \(N\) defines a different finite Hamiltonian.

### Molecular limit

For a one-cell cluster with increasing vacuum, image interactions and the
BvK Madelung correction vanish, and the result approaches molecular RHF or
RKS in the same basis. Basis linear dependence is handled per k by canonical
orthogonalisation; dropping different ranks at different k is diagnosed and
can spoil smooth convergence.

BSSE is not repaired by the cyclic construction. Counterpoise or basis
extrapolation is a separate modelling choice.

## 10. Assumptions and explicit divergences from historical CCM

### Audit of the 2008 deMon2k KS-ADFT CCM

Janetzko, Köster, and Salahub define the two-center occurrence weight

\[
 \omega_{MN}=\frac{1}{n_{MN}},\qquad
 T^{\mathrm{CCM}}=\sum_{\mu\nu}P^{\mathrm{CCM}}_{\mu\nu}
 \omega_{MN}\sum_{\nu'}T_{\mu\nu'}.
\]

For a three-center quantity they do not merely multiply one selected integral.
Their Eq. (20) averages both nested enumerations (M'\to N') and
(N'\to M'), divided by

\[
 n_{MNC}=n_{MN}(n_{MC}+n_{NC}),\qquad \omega_{MNC}=n_{MNC}^{-1}.
\]

That explicit two-order average matters. It means the simple
(M\leftrightarrow N) objection to the 2014 four-center factor cannot be
transferred unchanged to the deMon2k three-center tensor. If both enumerations
are complete, Eq. (20) is symmetric in the two AO centers by construction.

The two-center formula still requires the stronger Hermiticity identity

\[
 \omega_{MN}\sum_{\nu'}I_{\mu\nu'}
 =\omega_{NM}\sum_{\mu'}I_{\nu\mu'}.
\]

The free-space identity (I_{\mu\nu}=I_{\nu\mu}), cited in the paper, does
not by itself prove this equality for two different center-specific Wigner--
Seitz enumerations. Likewise, variational auxiliary fitting requires the
weighted three-center map and weighted auxiliary Coulomb metric to be adjoints
under one finite inner product. The paper supplies a fitted-energy expression
and its derivative, so the intended scalar functional is clearer than in the
2014 four-center derivation, but it does not prove this quotient-space adjoint
condition for arbitrary non-Bravais or skew clusters.

There is also a separate finite-size issue. The paper explicitly states that
its CCM and the corresponding periodic calculation are not fully equivalent
because interactions outside the Wigner--Seitz regions are omitted. Its
auxiliary-density lattice sum is truncated after one neighboring shell, and
the XC quadrature is evaluated on a reference-cluster grid with translated
auxiliary functions. Those can converge with cluster size, but they are not an
exact finite-(N) BvK identity. Thus the deMon2k multiplicative weights remain
a symmetry and variational audit target; the present derivation does not label
them broken where Eq. (20) already enforces the relevant AO-pair symmetry.

1. **Finite torus, not a weighted open cluster.** The method is defined by a
   BvK Hamiltonian first. Wigner--Seitz geometry is a representation of its
   translation classes.
2. **Boundary weights are a partition of unity.** They average only tied
   representatives of the same periodised object. They are not attached to
   atoms, shells, or arbitrary AO pairs.
3. **No historical product-weighted ERI.** The published pair-product
   four-centre factors are not invariant under all ERI permutations in their
   displayed form. Without a separate scalar functional derivation, using
   them directly would make the variational and double-counting status
   unclear. `aiccm2026dev-b` therefore does not use them.
4. **No Fock symmetrisation as a repair.** Hermiticity follows from the
   Hamiltonian. A non-Hermitian raw Fock is treated as a bug.
5. **Long range is part of the Hamiltonian.** It is not an after-the-fact
   Madelung energy added to a short-range SCF.
6. **Charged cells fail closed.** No implicit uniform-background energy is
   reported.
7. **Exchange is not independently truncated.** It uses the finite-torus
   periodised kernel and BvK-size divergence correction.
8. **The reciprocal implementation is exact, not a reference shortcut.** It
   is the diagonal representation of the finite cyclic translation group.
9. **Source agreement is secondary.** Historical numerical agreement does
   not override idempotency, permutation symmetry, variational consistency,
   gauge consistency, or finite-supercell/mesh equivalence.

## 11. Validation ladder

### Fixed-density kinetic derivative component

For a caller-supplied real-torus density whose cell list is fixed by declared
`lattice_options`, the kinetic contribution is

\[
 E_T[D;R] = \sum_g \operatorname{Tr}[D(g) T(g;R)].
\]

`compute_aiccm2026dev_b_fixed_density_kinetic_energy(...)` evaluates this
scalar, while
`compute_aiccm2026dev_b_fixed_density_kinetic_gradient(...)` evaluates
\(\partial E_T/\partial R_A\) from analytic AO-centre derivatives at fixed
\(D(g)\). The derivative is independently checked against central differences
of the matching scalar after rebuilding the displaced basis. Both paths
require exactly the same lattice-cell indices and Cartesian translations,
AO dimension, and block shapes.

There is no SCF convenience wrapper for this component. The current SCF result
does not attest the resolved one-electron lattice cutoff, so reconstructing an
operator from a guessed or default cutoff could silently compare a different
cell support. The stationary energy-weighted density construction and
Pulay/adjoint assembly, BvK exchange-q=0 seam or screened-exchange derivative,
RI/RIJCOSX three-centre and metric response, DFT exchange-correlation
quadrature/grid derivatives where applicable, and post-HF response terms
remain open. The public total-gradient route therefore still fails closed.
Γ-CCM and χ-CCM are distinct approaches compared at a declared common
exchange-q=0 convention; their names do not select different Coulomb kernels.

### Fixed energy-weighted overlap derivative component

For a caller-supplied real-torus energy-weighted density \(W(g)\), define the
fixed overlap-constraint scalar

\[
 \mathcal L_S[W;R]
 =-\sum_{g\mu\nu}W_{\mu\nu}(g)S_{\mu\nu}(g;R)
 =-\sum_g\operatorname{Tr}[W(g)^{\mathsf T}S(g;R)].
\]

The transpose in the final expression is important: individual real-space
residue blocks need not be symmetric even when the complete finite-character
object obeys Hermiticity and time reversal. At fixed numerical \(W(g)\) and
fixed lattice,

\[
 \left.\frac{\partial\mathcal L_S}{\partial R_{A\alpha}}\right|_{W,\mathrm{lattice}}
 =-\sum_{g\mu\nu}W_{\mu\nu}(g)
   \frac{\partial S_{\mu\nu}(g;R)}{\partial R_{A\alpha}}.
\]

`compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian(...)`
evaluates the scalar, and
`compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient(...)`
evaluates the analytic AO-centre derivative. A 3D odd-character conjugate
pair produces real but individually nonsymmetric \(W(g)\) blocks, pinning the
Frobenius orientation and sign against central differences of the public
scalar. Both helpers require identical ordered cell indices, Cartesian
translations, AO dimension, and block shapes.

This scalar is a Lagrangian constraint companion, not a separately additive
SCF energy. The helpers do not construct the stationary χ-CCM energy-weighted
density from orbitals, occupations, spin channels, backend, or exchange
convention. They also do not include the separate explicit derivative of the
BvK exchange seam or screened-exchange kernel. Consequently, they prove only
the overlap skeleton; B-owned adjoint construction and variational assembly
remain open, and the total gradient still fails closed. Γ-CCM and χ-CCM are
distinct approaches compared at a declared common exchange-q=0 convention;
their names do not select different Coulomb kernels.

### Fixed-input restricted energy-weighted-density audit

D112 adds the algebraic bridge into the D80 fixed-\(W\) skeleton without
claiming the stationary D85 binding. For explicit restricted occupied
coefficients and an explicit candidate variational Fock at character \(q\),
define

\[
 \Lambda(q)=C_o(q)^\dagger F(q)C_o(q),
 \qquad
 W(q)=2C_o(q)\Lambda(q)C_o(q)^\dagger.
\]

The occupied coefficients must obey
\(C_o(q)^\dagger S(q)C_o(q)=I\). If
\(D(q)=2C_o(q)C_o(q)^\dagger\), the same object satisfies the independent
algebraic identity

\[
 W(q)=\frac{1}{2}D(q)F(q)D(q).
\]

For an arbitrary occupied unitary \(U(q)\), replacing
\(C_o(q)\) by \(C_o(q)U(q)\) sends
\(\Lambda(q)\) to \(U(q)^\dagger\Lambda(q)U(q)\) and leaves \(W(q)\)
unchanged. The public audit helper verifies this object before applying

\[
 W(g)=\sum_q w_q e^{-2\pi i q\cdot g}W(q)
\]

onto the exact overlap cell list generated from the caller's
`lattice_options`. The explicit \(S(q)\) input must equal the forward Bloch
sum of that list. The boundary also requires finite Hermitian \(S(q)\) and
\(F(q)\), a common positive occupied rank, occupied overlap orthonormality,
the complete unreduced uniform dual character net, and time-reversal
consistency of \(S\), \(F\), the occupied projector, and \(W\). It rejects a
character coordinate outside the bounded exact-integer label range, any
nonfinite intermediate produced by the algebra, and a non-real
inverse-transform residue instead of silently taking its real part.

`compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(...)`
returns a `LatticeMatrixSet` that can be passed directly to the D80 overlap
Lagrangian scalar and AO-centre derivative. This composition is checked by a
central difference with the returned numerical \(W(g)\) held fixed. Even,
odd, and multi-axis character meshes pin the transform orientation, and a
complex occupied rotation pins covariance.

The word `fixed-input` is load-bearing. The helper neither reads stored
orbital energies nor infers occupations. It does not rebuild the final
physical unmixed and unshifted Fock, attest retained projectors or numerical
support, establish backend variational closure, or bind the object to the
executed SCF state. It is therefore not a stationary energy-weighted density,
a Pulay assembly, or a total force. Those requirements remain in D85 below.

### Fixed-input unrestricted energy-weighted-density audit

D113 extends the same explicit-input algebra to the supported collinear
unrestricted routes. For \(s\in\{\alpha,\beta\}\), let the occupied
coefficients and candidate variational Fock at character \(q\) be
\(C_o^s(q)\) and \(F^s(q)\). Define

\[
 \Lambda^s(q)=C_o^s(q)^\dagger F^s(q)C_o^s(q),
 \qquad
 W^s(q)=C_o^s(q)\Lambda^s(q)C_o^s(q)^\dagger.
\]

With the occupation-one spin density
\(D^s(q)=C_o^s(q)C_o^s(q)^\dagger\), the independent identity is

\[
 W^s(q)=D^s(q)F^s(q)D^s(q),
 \qquad
 W(q)=W^\alpha(q)+W^\beta(q).
\]

There is no restricted factor of two within either spin block, and the
spin-independent overlap constraint contains no alpha-beta cross term. An
independent occupied rotation \(C_o^s\mapsto C_o^sU^s\) sends
\(\Lambda^s\mapsto U^{s\dagger}\Lambda^sU^s\) and leaves \(W^s\)
unchanged.

The occupied ranks may differ between alpha and beta, but each spin rank must
be constant over the complete character net. Either rank may be zero through
an explicit coefficient stack of shape
`(n_character, n_ao, 0)`; both ranks zero are rejected as a vacuous audit.
For a zero-rank channel, its Gram matrix and \(\Lambda^s\) are empty and its
projector and \(W^s\) are exactly zero. Its explicit \(F^s(q)\) remains a
finite square AO input and must still be Hermitian and obey the supported
per-spin sewing relation.

The boundary checks \(C_o^{s\dagger}S C_o^s=I\), candidate Focks, occupied
projectors \(C_o^sC_o^{s\dagger}\), and \(W^s\) separately in each spin.
The coefficient matrices themselves are not sewn because their occupied
gauges are arbitrary. In the supported collinear current-free domain, the
matrix condition is the per-spin complex-conjugation sewing
\(M^s(-q)=M^s(q)^*\). This code-level spatial condition does not assert that
a spin-polarized determinant is invariant under the full spin time-reversal
operator. Each spin's inverse transform must also satisfy the declared
imaginary-residue tolerance before the spin sum is formed. Per-spin validation
prevents opposite alpha and beta defects from cancelling in the sum.

The total character block is inverse-transformed with the same convention as
D112,

\[
 W(g)=\sum_q w_q e^{-2\pi i q\cdot g}
      \left[W^\alpha(q)+W^\beta(q)\right],
\]

onto the exact overlap cell list selected by `lattice_options`. The complete
bounded dual character net, exact forward overlap-support match, finite
derived algebra, and real inverse-transform residue remain fail-closed
requirements. The returned
`compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(...)`
object can be passed directly to the D80 overlap scalar and AO-centre
derivative with its numerical blocks fixed.

The closed-shell reduction pins all spin factors. If
\(C_o^\alpha=C_o^\beta=C_o\) and \(F^\alpha=F^\beta=F\), then

\[
 W_{\mathrm U}=2C_o\Lambda C_o^\dagger=W_{\mathrm R}.
\]

Equivalently, for the spin-summed restricted density
\(D=2C_oC_o^\dagger\), \(D^\alpha=D^\beta=D/2\) gives
\(\sum_sD^sFD^s=DFD/2\), exactly the D112 identity.

This helper remains fixed-input. It does not infer integer occupations from a
result, read stored eigenvalues or densities, rebuild either final unmixed and
unshifted physical Fock, test a Brillouin or SCF stationarity residual, attest
backend variational closure, or bind the inputs to the executed state and its
support digests. A seam or screened-exchange contribution enters \(W^s\) only
if it is already present in the caller's candidate \(F^s\); its explicit
nuclear derivative remains a separate term. D85 stationary binding,
RI/RIJCOSX and XC derivatives, response, post-HF terms, and the total force
remain open.

### Stationary energy-weighted-density binding contract

D85 specifies the object that must eventually connect the fixed-`W` D80
skeleton to a stationary SCF Lagrangian. For character \(q\) and spin
\(\sigma\), rebuild the final physical variational Fock
\(F^\sigma_{\mathrm{var}}(q)=\partial E/\partial D^\sigma(q)\) without DIIS
mixing, damping, level shift, clipping, or a stale pre-final operator. With an
orthonormal occupied block \(C_o^{\sigma\dagger}S C_o^\sigma=I\), define

\[
 \Lambda^\sigma=C_o^{\sigma\dagger}F^\sigma_{\mathrm{var}}C_o^\sigma,
 \qquad
 W(q)=\begin{cases}
 2C_o\Lambda C_o^\dagger,&\text{RHF/RKS},\\
 \sum_\sigma C_o^\sigma\Lambda^\sigma C_o^{\sigma\dagger},
 &\text{UHF/UKS}.
 \end{cases}
\]

This definition is invariant under occupied rotations and phases. The first
contract version accepts only the existing zero-temperature integer
occupations. Unrestricted spin blocks remain separately attested; only their
sum enters the spin-independent overlap term. The contract inverse-transforms
the verified time-reversal-consistent character blocks as
\(W(g)=\sum_q w_q e^{-2\pi i q\cdot g}W(q)\); it must not repair a failed
sewing relation by taking an unexplained real part.

The immutable binding includes the atom order and effective charges/ECPs,
basis shell and AO order, finite-torus descriptor, retained AO projectors at
every character, canonical digests of \(S,F_{\mathrm{var}},D,C_o\), occupations
and \(W\), the resolved one-electron cell support, and the route-specific
two-electron and XC-grid support. A rebuilt-Fock stationarity and directional
variational-closure check is required for each backend. RIJCOSX fails closed
unless its approximate exchange energy and Fock satisfy that closure or a
derived adjoint supplies the missing response.

Under D95, the result-owned exact-exchange assembly records its full-range and
screened coefficients, physical screening parameter, and consistency with
q=0 applicability. Every active BvK seam and screened-exchange term enters the
stationary \(F_{\mathrm{var}}\), hence \(W\). Their explicit fixed-density
derivatives remain separate: the BvK seam derivative is present only when its
applicability is `active`, while the finite HSE `erfc` exchange derivative uses
the same screening parameter and numerical domain as the SCF. Neither is
hidden in or counted twice with the overlap derivative. This first contract
covers atomic displacements at fixed primitive lattice; stress remains open.

The seam record stores an effective operator coefficient \(\eta\) rather than
relying on a signed-Madelung naming convention. The raw full-range exchange
operator and its restricted variational-Fock and energy contributions are

\[
 \Delta K_{\mathrm{raw}}^R=\xi_{\mathrm{BvK}}SDS,
 \qquad \Delta F_{\mathrm{seam}}^R=-\frac{\eta}{2}SDS,
 \qquad E_{\mathrm{seam}}^R=-\frac{\eta}{4}
 \operatorname{Tr}[(DS)^2],
\]

and for unrestricted spin densities,

\[
 \Delta K_{\sigma,\mathrm{raw}}=\xi_{\mathrm{BvK}}SD_\sigma S,
 \qquad \Delta F_{\sigma,\mathrm{seam}}=-\eta SD_\sigma S,
 \qquad E_{\mathrm{seam}}^U=-\frac{\eta}{2}\sum_\sigma
 \operatorname{Tr}[(D_\sigma S)^2].
\]

Here \(\eta=\xi_{\mathrm{BvK}}\) for RHF/UHF,
\(\eta=c_{\mathrm{full}}\xi_{\mathrm{BvK}}\) for a global hybrid, and
\(\eta=0\) when applicability is `inactive`. HSE06 instead records `c_full=0`,
`c_sr=0.25`, and `omega_screen_bohr_inv=0.11`. Its fixed-density exchange
functional is
\(E_{x,\mathrm{sr}}^R=-c_{\mathrm{sr}}\operatorname{Tr}[D
K_{\mathrm{erfc}}[D]]/4\), or the corresponding unrestricted spin sum with
coefficient \(-c_{\mathrm{sr}}/2\). No q=0 seam is attached to that finite
screened kernel. Force derivation must differentiate these exact recorded
functionals using the same numerical support as the SCF.

In these equations \(\xi_{\mathrm{BvK}}>0\) is the code's probe-charge
Madelung coefficient. A theory notation that instead calls the signed
self-potential \(\xi_N<0\) maps to this operator coefficient as
\(\eta=-\xi_N\) for RHF/UHF and
\(\eta=-c_{\mathrm{full}}\xi_N\) for a global hybrid. Mixing those sign
conventions without this map reverses both the seam scalar and derivative.

Each approach must map its own stationary state into the binding contract and
derive the fixed-density seam or screened terms against that contract. Within
χ-CCM, the character and real-Gamma forms meet by the exact finite Fourier
transform. Cross-approach equality is a separate result to establish with
operator-specific and finite-difference gates under the common exchange-q=0
convention.

### Route-resolved exact-exchange assembly provenance

D95 introduces the immutable B-owned
`AICCM2026DevBExactExchangeAssembly` with fields `c_full`, `c_sr`,
`omega_screen_bohr_inv`, and fixed resolver provenance
`resolver="vibeqc.periodic_screened_exchange.resolve_periodic_exchange"`.
D98 extends that historical unversioned shape to the first formal nested
schema,
`schema="vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"`, with
`screened_exchange_applicability` and `screened_exchange_assembly`.
The object is attached as
`result.aiccm2026dev_b.exact_exchange_assembly`; the result also exposes
`result.exact_exchange_assembly` as the same object. Fleet records serialize
the exact seven-field v1 object under top-level `exact_exchange_assembly`,
alongside rather than inside `finite_torus_convention`. The separation is
normative: the latter declares the exchange-q=0 family, whereas the former
records which full-range and finite screened exact-exchange arms the selected
route resolved and how an active screened arm was assembled.

The values come from that shared live resolver, not from a second B
functional-name table. The supported contracts are:

| Route | `c_full` | `c_sr` | `omega_screen_bohr_inv` | screened applicability | screened assembly |
|---|---:|---:|---:|---|---|
| RHF/UHF | 1 | 0 | 0 | inactive | not-applicable |
| pure DFT | 0 | 0 | 0 | inactive | not-applicable |
| global hybrid | nonzero resolved fraction | 0 | 0 | inactive | not-applicable |
| HSE type | 0 | positive resolved fraction | positive physical value | active | short-range-direct |

`omega_screen_bohr_inv` is in inverse bohr and is zero when the screened arm
is absent. Finite values, fixed resolver provenance, and the route-specific
zero/nonzero pattern are part of the descriptor contract. D83 remains the
independent convention-use flag and must agree exactly:
`exchange_q0_applicability="active"` if and only if `c_full` is nonzero. A
screened-only HSE-type record is therefore inactive at the full-range seam
even though `c_sr`, its separate screened applicability, and its physical
omega are active. The schema vocabulary distinguishes `short-range-direct`
from `full-range-minus-long-range`, which are different finite-N operators
with the same thermodynamic target. Current χ-CCM-B implements only
`short-range-direct`.

For an active screened arm, the four-center RKS/UKS Fock branch emits
`vibeqc.pbc-bipole.screened-exchange-execution/v1` only after the direct
K-erfc matrix has been folded, symmetrized where needed, and energy
contracted. The B wrapper requires the emitted coefficient and omega to equal
the live resolver. The shared corrected-gauge route may still record
`exchange_ewald_split=True` and `exchange_exxdiv="ewald"` when `c_full=0`;
those selector labels do not mean a full-range K correction or seam
contributed. D83 therefore remains coefficient-derived rather than inferred
from raw route flags. Missing, stray, or contradictory evidence fails before
diagnostics attach. This is branch-level action evidence, not a matrix digest,
independent numerical oracle, or domain qualification. D93 support
qualification remains separate. D92 and D94 continue to require an explicit
positive eta because their caller-supplied densities, BvK self-potential,
basis, and overlap support are not SCF-bound. D98 changes no Hamiltonian and
adds no screened-exchange derivative or total-gradient route.
It also proves no cross-approach identity: Γ-CCM remains the
union-and-weight/Wigner--Seitz integral-weighting construction, χ-CCM remains
the finite-translation-group character construction, and a common declared
exchange-q=0 convention does not identify them.

### Fleet acceptance of exact-exchange assembly provenance

D96 made the D95 object normative for stored quantitative-status fleet
records. D98 supersedes its payload shape with exact v1 keys. The producer
still obtains coefficients from the live periodic exchange resolver.
Independently, `b_routes.py` declares the exact expected
`(c_full,c_sr,omega_screen_bohr_inv)` tuple and screened pair for every
registered route and requires the fixed schema and resolver identifiers. The
current route oracle is

| Fleet route family | `c_full` | `c_sr` | `omega_screen_bohr_inv` | screened pair |
|---|---:|---:|---:|---|
| RHF and post-HF on the RHF reference | 1 | 0 | 0 | inactive / not-applicable |
| RKS/PBE | 0 | 0 | 0 | inactive / not-applicable |
| RKS/PBE0 | 0.25 | 0 | 0 | inactive / not-applicable |

The stored object must have exactly the seven D98 v1 keys. Coefficients must
be finite JSON numbers rather than booleans. The route declaration must agree
with D83 for `c_full`, while screened applicability must be active exactly
when `c_sr` is nonzero and the assembly label must match the independent route
oracle. All current fleet routes are screened-inactive; HSE remains outside
the fleet. Legacy unversioned four-field records, extra fields, an active
screened label on a current route, and
`full-range-minus-long-range` on a current route fail closed rather than being
backfilled. The audit exposes a specific failure reason; the comparator
refuses the row before emitting an energy or representation-control table.
This requirement applies to `ok` and `not_converged` records. `unsupported`
and `error` records remain failure evidence and do not need to claim a
successful assembly.

The fleet declaration is an independent metadata oracle. For current
four-center HSE results outside that fleet, the B wrapper additionally consumes
the branch-emitted execution value described above. Neither layer proves the
numerical matrix independently. D93 numerical-support qualification, D85
state binding, screened-exchange derivatives, HSE quantitative validation,
and the total-gradient fail-close remain separate.

### Fixed-density restricted active BvK seam derivative component

D92 implements the restricted full-range seam part of the preceding contract
as a fixed-input component audit. On the complete character net, with the
spin-summed restricted density, primitive lattice, and positive finite operator
coefficient \(\eta\) held fixed,

\[
 E_{\mathrm{seam}}^R[D;R]
 =-\frac{\eta}{4}\sum_q w_q
 \operatorname{Tr}[D(q)S(q;R)D(q)S(q;R)],
\]

and therefore

\[
 \left.\frac{\partial E_{\mathrm{seam}}^R}{\partial R_{A\alpha}}
 \right|_{D,\eta,\mathrm{lattice}}
 =-\frac{\eta}{2}\sum_q w_q
 \operatorname{Tr}\left[D(q)S(q)D(q)
 \frac{\partial S(q)}{\partial R_{A\alpha}}\right].
\]

The implementation forms
\(M(q)=\eta D(q)S(q)D(q)/2\), inverse-transforms it with the exact
finite-character convention, rejects a non-real residue, and evaluates
\(-\sum_g M_{\mu\nu}(g)\,\partial S_{\mu\nu}(g)\) with the native overlap
AO-centre derivative. The factor \(1/2\) in \(M\), rather than the energy's
\(1/4\), is required because the scalar contains two overlap factors. An odd
three-character, skew-cell restricted projector makes the real-space blocks
individually nonsymmetric and pins the transform orientation, sign, factor of
two, linearity in \(\eta\), and translational sum. Central differences of the
public scalar agree with the analytic component below \(10^{-7}\) Ha/bohr.

Both public helpers require a 3D `bvk-ewald` descriptor whose
`exchange_q0_applicability` is `active`, a restricted RHF/RKS record, the
complete unreduced uniform unshifted character net, and finite Hermitian,
time-reversal-consistent density blocks. `operator_coefficient_eta` is a
required positive input. D95 records the route-resolved full-range fraction,
but does not bind the caller's density, the BvK self-potential, basis, or
overlap support into the numeric \(\eta\). The helper therefore does not infer
eta from the descriptor or a method label. Inactive pure and HSE records
refuse the seam component rather than returning a zero that could be mistaken
for the screened-exchange derivative. Negative full-range fractions are
outside this first validated component contract.

The result supplies only the declared χ convention and executed character
net. It does not attest that the caller's density, \(\eta\), basis, or
`lattice_options` reproduce the final SCF realization. There is consequently
no SCF convenience wrapper and no stationary-force claim. The separate
unrestricted fixed-density seam pair is described by D94 below. HSE
screened-exchange derivatives, D85 state and support binding, RI/RIJCOSX
response, and the total χ gradient remain fail-closed.

### Fixed-density unrestricted active BvK seam derivative component

D94 extends the same fixed-input audit to the active unrestricted full-range
seam without forming a spin-summed exchange density. For independently
supplied alpha and beta character densities,

\[
 E_{\mathrm{seam}}^U[D_\alpha,D_\beta;R]
 =-\frac{\eta}{2}\sum_q w_q\sum_{s\in\{\alpha,\beta\}}
 \operatorname{Tr}[D_s(q)S(q;R)D_s(q)S(q;R)],
\]

so at fixed spin densities, eta, and primitive lattice,

\[
 \left.\frac{\partial E_{\mathrm{seam}}^U}{\partial R_{A\alpha}}
 \right|_{D_\alpha,D_\beta,\eta,\mathrm{lattice}}
 =-\eta\sum_q w_q\sum_{s\in\{\alpha,\beta\}}
 \operatorname{Tr}\left[D_s(q)S(q)D_s(q)
 \frac{\partial S(q)}{\partial R_{A\alpha}}\right].
\]

The corresponding per-spin Fock shift and overlap-response weight are

\[
 \Delta F_s(q)=-\eta S(q)D_s(q)S(q),\qquad
 M(q)=\eta\left[D_\alpha(q)S(q)D_\alpha(q)
 +D_\beta(q)S(q)D_\beta(q)\right].
\]

The implementation inverse-transforms \(M(q)\) onto the exact overlap support
and uses the native overlap AO-centre derivative. Alpha and beta are contracted
separately: replacing them by their sum would introduce unphysical cross-spin
exchange. The closed-shell substitution
\(D_\alpha=D_\beta=D/2\) gives
\(M=\eta DSD/2\) and reduces both the scalar and derivative exactly to D92.
An odd three-character skew-cell broken-spin fixture uses distinct alpha and
beta projectors to compare against a separate-spin oracle and to prove that a
total-density contraction gives a different, incorrect answer. The same gate
checks linearity in eta, the translational sum, and the analytic derivative
against central differences below \(10^{-7}\) Ha/bohr; an independent
closed-shell fixture pins the exact D92 reduction.

Both unrestricted helpers require a 3D `bvk-ewald` descriptor with active
`exchange_q0_applicability`, `method="UHF"` or a nonempty
`method="UKS/<functional>"`, and the complete unreduced uniform unshifted
character net. Alpha and beta blocks are validated independently for count,
shape, finiteness, Hermiticity, and time reversal; a zero beta density is a
valid saturated-spin input. `operator_coefficient_eta` remains a required
positive finite input. Active applicability declares that the route resolves
a nonzero full-range exact-exchange arm, and D95 records that arm's fraction,
but neither binds the caller's numeric eta or its remaining state and support
inputs. Restricted records fail the method guard.
Production pure-functional and HSE records carry inactive
applicability and fail the applicability guard; the helper does not recompute
applicability from a functional label.

As in D92, this is a fixed-input component audit. The result does not bind the
supplied spin densities, eta, basis identity, or resolved overlap support to
the SCF state. There is no SCF wrapper, D85 stationary-state binding,
screened-exchange derivative, density or orbital response, RI/RIJCOSX
three-center or metric response, or total-gradient claim.

### Gradient-audit finite-torus binding

Every component helper now checks that its result descriptor and active
system identify the same finite torus. For the present direct-product,
Γ-centred construction,

\[
 N_{\mathrm{result}}=N_{\mathrm{character}}
 =N_{\mathrm{BvK}}, \qquad
 A_{\mathrm{BvK}}=A_{\mathrm{system}}\operatorname{diag}(N),
\]

where `PeriodicSystem.lattice` stores primitive vectors as columns. The
recorded BvK lattice must match within \(10^{-12}\) bohr. A skew,
unequal-mesh test distinguishes this column scaling from the incorrect
row-scaled expression \(\operatorname{diag}(N)A_{\mathrm{system}}\).
Diagnostics and top-level
convention descriptors must agree, and the boundary model must be explicitly
`3d-periodic`.

SCF-density folding additionally validates the complete unreduced,
unshifted Γ-centred character set. Fractional labels are reduced modulo the
integer mesh, so equivalent representatives such as \(-1/3\) and \(2/3\)
are accepted, but duplicate, incomplete, shifted, or nonuniformly weighted
nets fail before the inverse transform.

This is a provenance guard, not another force term. It binds the torus size
and lattice but does not attest the atoms, basis identity, resolved
one-electron cutoff, or the source of caller-supplied \(D(g)\) or \(W(g)\).
Atomic displacements at fixed lattice remain valid for finite-difference
component checks; stresses and cell derivatives are outside this contract.
The equality above would also need revision for a future twisted boundary or
non-diagonal supercell construction.

### Two-electron numerical-support qualification

The declared 3D Hamiltonian does not by itself prove that a finite numerical
realization has converged all integration domains. The combined MgO audits
identified two independent support problems. Direct BIPOLE J/K needs an
internal ket-image traversal padded by the smeared AO-pair kernel range even
when the physical density and nuclear cutoffs are coherent. Fitted RI and
RIJCOSX avoid that traversal, but default RSGDF KE=200 has a distinct
dense-core high-G limitation. The split audit at `51e3b250` establishes the
direct issue; the dense-tail audit at `2355f52e`, pinned by
`tests/test_rsgdf_dense_mesh_tail.py`, establishes the fitted issue.

The shared BIPOLE workstream owns the direct fix. M4a defined an explicit
absolute-radius oracle. M4b supplied the separation-aware QQR bound and
pair-resolved compositions, and M5 made `sr_image_precision=1e-6` the default
for every erfc short-range build. The driver now resolves an absolute radius
larger than the physical electronic cutoff and stores it as
`sr_image_extent_bohr`. Pure restricted semilocal RKS has used the same padded
SR+LR default since the 2026-07-18 exact-FT retirement; the finite-KE
analytic-FT route remains only an explicit oracle. The shared HSE traversal
and unrestricted spin recursion are padded as well. HSE is not a fleet route
and still requires its own quantitative validation. Commit `2fd23eff` further
keeps the physical Bloch density unmasked when pair-resolved SYM3b is active,
applying the pair mask only inside the private direct-SR tensor build.
Symmetry-enabled four-center records made before that repair are
revision-bound; default symmetry-off radial runs are outside that defect.
The quartet far-field prototype is separately outside this contract. D105 pins
`use_multipole_far_field=False` at all four χ wrappers, and the shared drivers
reject explicit `True` before setup. Records made while χ inherited the
unrecorded shared default are revision-bound. Any future route needs a
three-translation domain identity plus executed-path telemetry and
matched-support quantitative validation before it can extend D93.

D93 implements the reportability transport as
`vibeqc.aiccm2026dev-b.two-electron-support/v1`. A current fleet
four-center RHF, PBE, or PBE0 row is qualified only when its record contains
the preserved low-level `pbc-bipole` backend, exact executed M5
`sr_image_precision=1e-6` policy, electronic and nuclear cutoffs, and a finite
resolved absolute radius strictly larger than the electronic cutoff. The
nested cutoffs must match the independent `direct_lattice_cutoffs`
serialization. This consumes the shared
BIPOLE result and does not copy its kernel into the χ line. RI, RIJCOSX, MDF,
and post-HF rows serialize their active base/runtime settings but remain
`not-qualified`: no accepted high-G RSGDF tail, COSX exchange-domain, MDF, or
correlation-factor support contract exists. Missing, malformed, internally
contradictory, route-inconsistent, or unqualified quantitative rows fail
independently in `audit_b.py` and
`compare_b.py`; unsupported and error rows remain visible failure evidence.
Existing fitted SCF convergence and route-plumbing records remain useful but
are not external absolute-energy validation. These are numerical-support
conditions inside the declared convention, not new gauges, Coulomb kernels,
or Γ/χ construction physics.

D106 exposes a diagnostic API path to the shared high-G RSGDF machinery after
`017c10bf4` made multi-k pure KS consume the requested tail. D93 v1 campaign
producers deliberately continue to require `rsgdf_tail_ke_cutoff=null`; an
interactive tailed result is therefore outside the accepted fleet schema and
remains quantitatively unqualified. The pre-D106 Gamma RHF/RI path could
silently replace that null input with the shared dense-core automatic tail;
such fitted and derived post-HF v1 records are revision-bound, and the selector
now stops that path before SCF. A future qualification must version the B-owned
support contract and bind the executed tail, route, base cutoff, auxiliary
basis, and numerical convergence evidence. API availability alone
cannot supply that evidence.

D103 adds an independent direct-fold contract,
`vibeqc.aiccm2026dev-b.overlap-fold-support/v1`. Every corrected-gauge
BIPOLE result preserves the executed maximum overlap-fold drift over the
character mesh. A drift above `1e-2` fails before the first Fock build;
`1e-4 < drift <= 1e-2` retains the shared note-only execution path but is
`not-qualified`; and `drift <= 1e-4` meets the quantitative fold target. A
direct fleet payload fixes
`diagnostic="max-abs-element-overlap-bloch-fold-drift/v1"` and
`reference_cutoff_factor=1.5`, then binds the drift to the executed electronic
cutoff, character mesh, two fixed thresholds, and its `four_center` route. Its
cutoff must equal the independent `direct_lattice_cutoffs` value and its mesh
must equal the row mesh. RI, RIJCOSX, MDF, and post-HF payloads use the same
exact schema as `not-applicable`, with null drift, cutoff, reference factor,
and mesh. Missing, malformed, nonfinite, negative, cutoff-inconsistent,
mesh-inconsistent, or moderate direct evidence fails in both `audit_b.py` and
`compare_b.py`.

D93 and D103 are independent necessary support gates. Neither establishes the
other. D104 permits a fresh exact-v2 record to clear only the old shared-Ewald
hold; every v1 record remains permanently quarantined. These are
numerical-support conditions within the χ-CCM construction at the declared
exchange-q=0 convention. They neither identify Γ-CCM with χ-CCM nor imply
different Coulomb kernels.

### Pinned nonquantitative χ route-control producer

D101 implements the χ-owned route in
`studies/aiccm-2026/COMPARISON_1D2D3D_2026-07-15.md` through the public high-level
entry point. `run_case_cmp_b.py` calls only
`run_periodic_job(..., method="RHF", functional=None,
jk_method="aiccm2026dev-b", aiccm_backend="ri")`. The chain and monolayer
labels are embedded objects in vacuum, but their `PeriodicSystem` instances
remain deliberately declared `dim=3`, like the crystal. The producer binds
the exact campaign lattices, atoms, meshes, STO-3G basis, resolved auxiliary
basis, RSGDF KE cutoff, HCORE guess, zero smearing, zero static and dynamic
damping, zero level shift, DIIS settings, convergence tolerances, and requested
and executed zero Fock mixing. Explicit inputs, source-resolved defaults, and
executed evidence are serialized separately so a default cannot masquerade as
an executed pin. The result's GDF method and cutoff fields are explicitly
labelled as B-selector-resolved settings forwarded to the fitted driver, not
backend-owned execution telemetry or D93 qualification. Likewise,
`declared_model="3d-periodic-in-vacuum"` describes a vacuum-padded geometry,
while `boundary_model="3d-periodic"` names the fully 3D periodic Green
function. The pair is intentional and does not select an isolated wire/slab
kernel or bypass D72/O1.

Acceptance requires contract checks of the complete character residue
product, reciprocal Cartesian consistency, the D83 exchange-q=0 applicability,
the D88 column-vector BvK lattice and positive code Madelung coefficient, the
route-resolved exact-exchange assembly, the shared Ewald nuclear scalar, and
the total/electronic/nuclear energy decomposition. That last energy check is
boolean assembly evidence only. Because D93 leaves RI support
`not-qualified`, neither total nor per-atom energy is serialized in the D101
campaign JSON; ordinary runner and SCF-log energy output remains a
nonreportable diagnostic artifact. The Madelung and nuclear values are
separately recomputed through the same shared helpers used by SCF. They provide
dispatch and assembly evidence, not an independent kernel or Ewald
implementation.

The nuclear comparison is intentionally same-helper only. D102 records the
historical shifted-pair omission, and D104 records its pair-complete repair and
same-process canary. The D101 equality still establishes route and assembly
consistency rather than an independent Ewald implementation. D101 remains
usable as a nonquantitative diagnostic because it emits no energy and D93
still leaves the fitted route unqualified.

The 3D crystal must complete first. Each embedded-object run consumes the
exact contract-valid nonquantitative 3D JSON by file SHA256 and records its vq
job, Slurm job, and
producer-attestation identifiers. All three runs must share the stable D77
source/version/core/attestor/library/payload tuple; the Slurm node hostname and
full per-job attestation digest may differ. The producer rejects bundle provenance, checks exact
current `origin/main` plus the required repair ancestors, and compares the
tracked producer, probe, test-set, route-registry, and launcher bytes before
and after SCF. The normative D91 bundle contract exists, but its
transport/producer/auditor implementation remains prospective. The compute-managed
managed runtime audited on 2026-08-11 was bundle-shaped, but its
version-and-short-SHA artifact path and sibling checksum were replaceable and
its launcher did not bind a literal digest. It was not an accepted immutable
D91 artifact. A routine bundle refresh cannot satisfy D101 even at the right
SHA. An accepted run needs either
a separate D77-clean Git checkout or a future implemented D91 contract with
no-clobber publication, graph-proven ancestry, and digest-bound launch.
Successful JSON always combines
`status="ok"`, `evidence_role="route-plumbing-diagnostic"`,
`quantitative_status="not-qualified"`, and
`comparison_status="no-gamma-chi-construction-comparison-defined"`.
`audit_cmp_b.py` revalidates the exact schema without printing an energy. A
`1d` or `2d` audit requires the exact fetched 3D JSON so its file SHA and parent
identifiers can be recomputed rather than trusted from the child record.
Here `status="ok"` means only that execution and the internal contract audit
succeeded. The asserted `vq_target_host="compute-managed"` and numeric Slurm job id
record operator intent and scheduler context; vq currently exports no field
that cryptographically proves the target alias.

D101 deliberately emits neither the historical v2 `comparison_input` nor a
`comparison_contract`. It defines no paired representation result or Γ/χ
approach pairing and does not
assert identical discretized Hamiltonians merely because convention labels
match. It changes no construction, Coulomb convention, backend support, or
lower-dimensional and total-gradient fail-close.

The committed tests cover the selector, exact mesh, Wigner--Seitz ties,
partition unity, inverse Fourier transform, metric idempotency, fail-closed
domain, RHF/RKS runner dispatch, native 3D SCF runs through all three integral
backends, explicit 1D/2D rejection across every backend, and χ-CCM native
post-HF setup kernels. The native
OpenMP kernels are parity-tested against the earlier NumPy equations for the
one-body inverse transform, pair-resolved RI-factor inverse transform, and
three-index AO-to-MO contractions. D111 additionally uses explicit dense
test-only reconstruction on even, odd, and multi-axis cyclic groups to pin
home-auxiliary recovery, the \(\mathbf R-\mathbf T\) factor orientation, the
\(\mathbf R+\mathbf T\) coefficient shift, complex left conjugation, and the
exact one-cell-orbit storage ratio. Multi-cell restricted and one-cell
unrestricted exact-limit correlation sentinels exercise the same production
adapter; the translated transform itself is spin-independent and is pinned on
every synthetic mesh. The benchmark harness adds:

1. the archived pre-guard 1D alternating hydrogen-chain ladder as failure and
   historical comparison evidence, not as reportable absolute energies;
2. direct comparison with both the historical `union12` and separately
   developed symmetric `aiccm2026dev-a` weights in the in-repo CCM on identical
   geometry;
3. exact comparison with the native Γ-centred k-mesh representation;
4. energy, convergence, idempotency, electron count, wall time, and
   implementation-to-implementation differences;
5. later 3D LiH, MgO, NaCl, Al2O3, and diamond ladders only after each
   basis, gauge, internal traversal, and reciprocal tail is pinned to an
   out-of-process reference input.

The reportable Γ/χ approach map is empty. The A RI and RIJCOSX harness routes
call the same multi-k GDF SCF drivers as B, so agreement between those routes
tests their shared implementation and cannot isolate an approach change. B
`rhf-ri` versus A `aiccm-hf-direct` is a neutral-torus representation control,
not evidence for the union-and-weight Γ-CCM approach. The direct solver
assembles and minimizes a real Γ-supercell problem, but its folded cderi and
one-electron matrices intentionally reuse the common per-q RSGDF fits and
lattice-sum primitives. The control therefore checks Γ-supercell versus
character/Bloch assembly, not an independent RI implementation or a Γ-CCM
versus χ-CCM approach delta. It is still not reportable
because comparator contract `aiccm2026-gamma-chi/v2` is only an incomplete
validation scaffold and must not be emitted unchanged. A superseding contract
needs the operator and retained-space evidence below.

Contract v2 requires the canonical `comparison_input` to be embedded beside
its exact `comparison_contract.input_sha256` hash, validates its required
physical/numerical fields, and reconciles them with the result and contract. It
distinguishes the AO
linear-dependence threshold, currently `1e-7`, from the auxiliary-metric
linear-dependence threshold, currently `1e-9`. It fingerprints and reconciles
requested and reported SCF smearing and accepts only the zero-temperature
setting (`smearing_temperature=0.0` Ha). Contract v2 does not bind Fock
mixing, and D86 does not change or upgrade that incomplete comparator
contract. A superseding contract must bind requested and executed accelerator
controls before using them as comparison evidence. Contract v2 also requires matched clean source,
native-core, host, package-version, and successful composite producer-process
attestation. The cached `aiccm-host-probe/v2` preflight remains the trusted
dispatch gate, but it is not accepted as result evidence by itself. Each
producer binds that preflight to its own current identity and a canonical
manifest of its copied producer, probe, test-set, launcher, and stream-specific
route files. In
the producer process it records linked native-library versions and always runs
a cheap shifted-mesh native-versus-Python AO-pair Fourier-transform canary. If
the native-core path SHA256 changed after preflight, it reruns the full v2 host
checks in process, including API shape, reciprocal cutoff, and LiH
direct-versus-GDF checks; the cheap canary alone is not sufficient for a changed
core.

Immediately before result serialization the producer re-reads its clean source,
host, package, native-core path SHA256, native-library versions, and payload and
requires them to be unchanged. The SHA256 describes bytes currently at the
imported module's filesystem path, not bytes already mapped into the process.
The same-process numerical checks therefore attest loaded behavior, but do not
claim a cryptographic identity for the mapped image. The comparator independently
validates the composite and nested canonical digests, exact schemas, check
versions, and tolerances. It compares A and B using their current producer
identities; different valid preflight histories and the intentionally different
stream payload manifests need not match. A sidecar added during later curation
cannot establish any of these facts retrospectively, and old rows are not
upgraded by the new schema.

D91 defines the requirements for one possible future alternative on an
immutable-bundle host where a clean Git checkout is structurally unavailable.
A bundle-attested producer would have to bind the checksum-verified artifact to
a full source SHA proven resolvable on `origin/main`, record package,
native-core, and native-library identities, run the full versioned numerical
checks and loaded-core canary in process, bind its copied producer payload, and
recheck every identity layer before serialization. The row would record
`attestation="bundle"`; a source reconstructed as a clean checkout records
`attestation="git-checkout"`. The current B producer, curator, and comparator
accept only the D77 clean-checkout schema, so bundle rows remain fail-closed
until an authoritative immutable-artifact manifest is created, its full source
SHA is proven resolvable on `origin/main`, and the second schema and its tests
land. D95 exchange-assembly provenance supplies none of those identity proofs.
Bundle attestation cannot repair old
rows, replace the numerical-input or two-electron-support contracts, or turn a
finite-torus route control into a Γ-CCM/χ-CCM approach comparison.

The factor audit also exposed a finite-cutoff reciprocal-support seam.
An ordered character pair can report a raw transfer `q + G` while the
neutral-torus real-Gamma fold uses the equivalent finite-group transfer `q`.
The ket Bloch cell
phase is unchanged, but shifting one fixed finite `|G|` ball by those two raw
labels selected different boundary crescents. On the compact H2 `(3,1,1)`
control, the relative fitted-Gram residual was `9.73e-8` at 40 Ha and
`2.46e-11` at 200 Ha. A half-open centered representative fixes that odd-mesh
case but breaks time reversal at an even-mesh Nyquist transfer; keeping both
Nyquist signs restores time reversal but not reciprocal-label invariance. The
shared Bloch RSGDF builder now enumerates the physical shifted sphere
`0 < |G+q| <= sqrt(2 E_cut)` directly. That support is both modulo-G invariant
and inversion covariant. On skew H2 `(2,1,1)` and `(3,1,1)` controls, the
canonical χ-CCM factor inverse transform now matches the neutral-torus folded
fitted-Gram control at the `1e-14` numerical floor. Separate gates pin reciprocal
relabelling of the compact fitted Gram and canonical factor, plus
canonical-frame Nyquist time reversal. This is a numerical representation fix
inside the same finite-torus Hamiltonian and Coulomb convention.
The q=0 support and accumulation order are byte-preserved. Finite nonzero-q
RSGDF support changed, so pre-D78 exact-GDF-K RHF/UHF and hybrid RI,
neutral-torus fold, and RI post-HF/full-pair records are revision-bound until
rerun.
Semilocal RI and RIJCOSX SCF build only q=0 RSGDF J factors and are numerically
unaffected by this support change.

Numerical-input emission remains disabled pending the remaining contract
corrections. The legacy real-Gamma control builder blocks use the canonical
auxiliary-AO embedding, but the final control cderi is an unfolded-k Fourier
fold followed by a real
`q/-q` stack and a numerical-null-row prune. B SCF instead self-contracts one
compact metric-eigenmode factor per ordered character pair. Raw factors need
not match. Acceptance must name the common Bloch pair-density phase, record
the route-specific factor pipelines, and establish q-resolved agreement of the
retained auxiliary projectors together with the gauge-invariant fitted Gram
operators. Matching metric ranks, thresholds, or projectors alone is
insufficient.

A real-Γ supercell producer screens one absolute supercell overlap, whereas
χ-CCM performs normalized canonical orthogonalization independently at each
character. This retained-space distinction is an evaluation-control fact; it
does not bind the real-Γ producer to the union-and-weight Γ-CCM construction.
Equal threshold numbers do not prove equal retained spaces. For the current
control route, acceptance requires full/no-drop at every B character together
with the finite-Fourier overlap residual, so every character retains all
primitive-cell AO directions. If a future comparison permits projected
spaces, it must compare Fourier-related retained-space projectors rather than
rank counts alone. No Γ/χ approach delta can be emitted until these conditions
are specified and tested and a route actually realizing the union-and-weight
Γ-CCM construction is paired with χ-CCM. Equality between the distinct
Γ-CCM and χ-CCM approaches for a
specified operator and route remains evidence to establish under the declared
common exchange-q=0 convention.

External CRYSTAL or PySCF comparisons remain out of process and are weak
signals unless their cell, Coulomb, exchange-divergence, basis, and k-mesh
conventions are matched explicitly.
