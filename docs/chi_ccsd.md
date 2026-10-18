# Experimental chi same-source RHF/UHF CCSD and triples

The chi CC route consumes the sealed stationary classic RI RHF or UHF
state and its retained canonical factors. The public runner selects it with
`method="aiccm", variant="chi", correlation="ccsd", scf_reference="rhf"`
or `scf_reference="uhf"`, `aiccm_backend="ri"`, `fit_source="classic"`,
and the same explicit classic physical-pair controls required by
[chi MP2](chi_fitted_support.md). It computes CCSD plus the precisely defined
canonical spin-orbital triples correction below by default.
Pass `chi_cc_options=ChiCCOptions(compute_triples=False)` for CCSD alone;
`ChiCCOptions` lives in `vibeqc.periodic.chi.canonical_cc`. The library entry
point is `vibeqc.periodic.chi.correlation.run_chi_ccsd(reference, ...)`.

The source, support, kernel, metric rank and HF orbitals are never rebuilt or
substituted. Integer gapped explicit-electron occupations, complete unshifted
characters and the existing tight stationary-HF acceptance are required.
`chi_frozen_core` counts lowest occupied orbitals per character in each spin.
An integer applies to both spins; UHF also accepts `(ncore_alpha, ncore_beta)`.
The remaining alpha/beta occupied and virtual dimensions may differ.
All-electron and [classic scalar ECP references](chi_ecp.md) are admitted.
Frozen spaces contain only explicit occupied orbitals, after ECP core removal.
SR CC, ROHF-CC, KS/hybrid, spin-orbit ECPs and fractional occupations are
unsupported in this increment. The accepted SR RHF MP2 route is unchanged;
original SR10 negative metric modes and SR32 complex UHF operator failures
remain gates. Chi is experimental; D93 fitted/post-HF is not-qualified.
R5.3 is an unrestricted source implementation. Restricted R5.2 scientific
acceptance remains separate, and neither it nor accepted scoped UMP2 results
provide CC acceptance by inheritance.

## Equations and conventions

The implementation uses the complex spin-orbital form of
[Stanton et al. (1991)](https://doi.org/10.1063/1.460620), section II,
equations 1-13, with `t1[a,i]`, `t2[a,b,i,j]` and antisymmetrized
`g[p,q,r,s] = <pq||rs>`. Amplitudes multiply excitation operators and are
not conjugated in the CCSD energy or residual equations. Bra orbitals in
integrals are conjugated. The factor product is `(pq|rs)=L[pq] L[sr]^H`,
using one shared transfer metric, including at complex characters.

The occupied and virtual lists carry `(character, canonical MO, spin)` labels.
Each UHF spin channel uses its own sealed coefficients, orbital energies,
density, occupations and Fock matrix. The consumer checks orbital
orthonormality, `F_sigma C_sigma = S C_sigma epsilon_sigma`, and the occupied
projector against that spin's density. It makes no semicanonical rotation and
does not identify alpha and beta spatial orbitals. Freezing occurs after the
accepted canonical frame is established, without further core/valence mixing.
Source seals bind both spin channels and the original source, kernel, q=0
convention, support, metric rank and factors.

`T1 = sum_ai t1[a,i] a^dagger_a a_i` and
`T2 = (1/4) sum_abij t2[a,b,i,j] a^dagger_a a^dagger_b a_j a_i`.
The full antisymmetric spin-orbital amplitudes contain independent alpha/beta
singles and aa/ab/bb doubles. The energy is
`sum_ia f[i,a] t1[a,i] + (1/4) sum_ijab g[i,j,a,b] t2[a,b,i,j]`
`+ (1/2) sum_ijab g[i,j,a,b] t1[a,i] t1[b,j]`.
For spin-integrated doubles, the same-spin contributions therefore have
weight `1/4` and the mixed-spin contribution has weight one. No extra
closed-shell factor of two is applied. The residuals are projections of
`exp(-T) H exp(T)` and retain all spin sectors allowed by the determinant.

Each excitation conserves crystal momentum modulo a reciprocal vector.
As in [McClain et al. (2017)](https://doi.org/10.1021/acs.jctc.7b00049),
section 3, equations 26-28, the reference is a finite complete-character
Hamiltonian. Primitive-cell factors give supercell integrals after division
by `Nk`; the resulting supercell correlation energy is divided by `Nk`.
Dense spin amplitudes currently retain zero forbidden sectors. This costs
more memory than fully momentum-compressed amplitudes without changing the
mathematics. There is no CCM construction or molecular-source substitution.

The CCSD contractions use NumPy's greedy contraction planner, which factors
multi-operand sums into matrix products and dispatches eligible products to
BLAS. Intermediate arrays stay within the planner's largest input/output
array limit and the admitted workspace reservation. This reduces the cost of
larger character groups while preserving the complex Stanton equations and
the retained physical source. Completed checkpoints retain their existing
convergence revalidation.

The retained factors omit zero momentum. HF already includes the BvK
Madelung exchange term `-xi S D_sigma S`. `ChiCCOptions.coulomb_zero_mode`
explicitly selects the finite-mesh CC convention:

- `"hf-energy-only"` preserves the historical default: corrected HF energies
  and bare retained ERIs. This is an energy-only finite-size approximation;
  it is not a constant-kernel many-electron Hamiltonian.
- `"matched-eri"` also adds `Nk*xi` to a primitive-cell Coulomb ERI when
  both orbital pairs are diagonal in character and canonical MO. After
  division by `Nk`, its spin-orbital correction is
  `xi*(delta_pr delta_qs - delta_ps delta_qr)`.

Both use the actual physical HF Focks and unchanged fitted factors. The
executed scalar is recovered from both spin Focks and checked against every
matrix element without rebuilding native integrals. Output identifies the
convention and `madelung_hartree`. Library calls and the public runner use
the same `ChiCCOptions`; pass `chi_cc_options` to `run_periodic_job`.

The matched treatment follows
[Xing and Lin (2024)](https://doi.org/10.1103/PhysRevX.14.011059), section II B,
equations 7-11; their signed constant is `-xi` here. A constant interaction
contributes only a constant on a fixed-electron Hilbert space after the
matching uniform Hartree/background convention. Its occupied-energy and
general-ERI contributions therefore cancel in the converged CCSD residual.
Independent fixed-electron determinant tests check this cancellation, also
for complex amplitudes. The energy-only approximation does not obey that
invariance; the two conventions must not be compared as the same finite
Hamiltonian. Their common thermodynamic limit does not establish accuracy
at a tested finite mesh. Changing the historical default remains a separate
maintainer decision, consistent with the
[periodic source design record](design_dlpno_correlation_source.md).

Frozen spaces retain the same scalar; ECP-removed electrons never enter the
active labels. Occupied-virtual MP2 numerators have no diagonal zero-mode
contribution, so MP2 is unchanged. The defined perturbative triples formula
retains its HF denominators in both CC modes; constant-kernel invariance of
converged CCSD is not a claim of invariance of finite-order triples.

The missing explicit CC convention, unavailable matched-ERI treatment and
absence of a convention-specific restart boundary are repaired here. HF
archives remain reusable, and old CC v1 archives remain readable only as
energy-only calculations. Changing modes requires fresh correlation from
the retained HF, as documented in [restart](chi_restart.md).
The [finite-size diagnostic](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/chi_finite_size.md) separates this
choice from the remaining HF cell/character and fitted-cutoff errors.

The triples definition is
`canonical-spin-orbital-hermitian-(T)/v1`, based on
[Raghavachari et al. (1989)](https://doi.org/10.1016/S0009-2614(89)87395-6),
section 5, equation 14. With the normal-ordered fluctuation operator `V`,
define `C=<triple|(V T2)_connected|reference>` and
`D=<triple|V T1|reference>`. For `i<j<k`, `a<b<c`, the correction is
`sum C.conj()*(C+D)/Delta`, with
`Delta = epsilon_i+epsilon_j+epsilon_k-epsilon_a-epsilon_b-epsilon_c`.
More explicitly, with `A_ijk A_abc` the signed sums over all 36 permutations,

```text
C_ijk^abc = (1/4) A_ijk A_abc [sum_e t_ij^ae g_bc,ek
                              - sum_m t_im^ab g_mc,jk]
D_ijk^abc = (1/4) A_ijk A_abc [t_i^a g_bc,jk]
```

The factor `1/4` accounts for each seed's already antisymmetric pairs;
unique triples replace the fully ordered energy sum's `1/36`. All allowed
aaa, aab, abb and bbb spin sectors are included, with spin and momentum
conserved between holes and particles. The singles coefficient is one,
as in equation 14; this is neither doubles-only CCSD+T(CCSD) nor QCISD(T).

For a real canonical UHF determinant this is the conventional spin-orbital
CCSD(T) expression. For complex orbitals the stated conjugation is an
explicit Hermitian extension; the real-orbital paper alone does not establish
its accuracy for every complex Hamiltonian. The implementation refuses an
imaginary correction exceeding `1e-9` Ha in the supercell sum instead of
silently taking its real part. Independent orbital-phase and determinant
checks establish the stated finite-Hamiltonian contraction, not a general
open-shell triples qualification. Result aliases are `UCCSD` / `UCCSD(T)`
with `reference_method="UHF"`; `triples_variant` records the definition and
citation. RHF aliases remain `CCSD` / `CCSD(T)`.

The formula requires canonical HF with zero occupied-virtual Fock elements.
It does not implement additional general-reference Fock terms, ROHF
semicanonical choices, spin-adapted open-shell CC, Lambda-CCSD(T), or CCSD-T.
Those names must not be used interchangeably with this UHF route. Two
independent HF calculations are not a same-root closed-shell reduction test;
that test requires identical alpha/beta orbitals, energies and occupied spaces.

## Resource and convergence contract

The virtual ladder streams one external virtual index. Non-virtual integral
blocks are built from cached MO factor panels; a full four-index spin integral tensor
and a six-index triples tensor are not required. The conservative complex
array workspace estimate includes cached blocks, intermediates, amplitudes
and eight DIIS vectors/errors, plus the MO pair factor cache and transform
workspace. It excludes the borrowed HF factors and
external BLAS/interpreter overhead. `memory_byte_cap` refuses an excessive
estimate before MO blocks or amplitudes are allocated. No contributions are
dropped to fit the budget. Triples still have canonical seventh-order cost.

Both the maximum projected residual and energy change must converge.
`max_iter`, `energy_tolerance`, `residual_tolerance` and
`denominator_tolerance` are explicit `ChiCCOptions` controls. Capped or
nonfinite iterations raise an error. Single, double and triple denominators
use the summed occupied minus virtual canonical energies. Nonnegative or
small denominators in allowed sectors are refused, never clamped. Forbidden
spin/momentum sectors and repeated antisymmetric indices do not require
denominators. DIIS affects iteration only, not the target equations.

Output separates HF energy, CCSD correlation, triples, total correlation and
correlated total, all per primitive cell. The nuclear contribution and
correlated electronic energy (total minus nuclear repulsion) are separate
manifest fields, with electronic energy also printed in `.out`. The `.out` and `.system` records
include source/reference seals, active counts, retained ranks, convergence,
amplitude maxima, residuals, workspace estimate and actual citation route.
Active counts and canonical residual diagnostics are resolved by spin.
The wrapper's `energy` remains the HF energy, matching existing periodic
correlated runners; use `e_total_correlated` for the correlated total.

Independent tests construct a determinant Hamiltonian and evaluate
`exp(-T) H exp(T)` at arbitrary complex amplitudes without calling the
production residual builder. A separate determinant contraction checks a
nonzero triples correction. These algebra checks do not establish physical
basis, auxiliary-space or thermodynamic convergence or production qualification.

## Persistent continuation

`run_chi_ccsd` and `evaluate_same_source_cc` accept `checkpoint_path`,
`resume_from` and `checkpoint_interval`. The classic source, canonical/frozen
spaces and equation controls must match. Increasing `ChiCCOptions.max_iter`
continues an iteration-limited solve with its amplitudes, cumulative trace and
compatible DIIS history. A triples checkpoint retains the converged CC state
and verifies complete-prefix coverage before continuing its partial sum.
Neither an iteration-limited CC solve nor incomplete triples is a completed
result. See [checkpoint and restart](chi_restart.md) for the `.vqr` format,
public runner options, atomic publication and exact compatibility policy.

## Validator cases and comparison boundary

Submit fresh R5.3 calculations through `vq` using the delivered runtime that
contains this source. Preserve existing restricted jobs and their runtime.
Record the source and imported Python/native runtime identities, complete
input, basis and auxiliary basis, geometry/cell, spin root, source descriptor,
frozen orbitals, SCF acceptance, CC thresholds and separated energies once on
the tracking issue. These are proposed cases, not measured acceptance results.

A reproducible small unrestricted starting case is neutral LiH2, multiplicity
two, with Cartesian positions in bohr `Li=(0,0,0)`, `H=(0,0,3)`,
`H=(2.5,0,-1)`, in an orthorhombic `24 x 26 x 28` bohr three-dimensional
cell. Use STO-3G, the explicit `def2-SVP-JKFIT` auxiliary basis, classic
physical-pair support, no frozen core, and full unshifted character meshes
`(1,1,1)` then `(3,1,1)`. This gives three alpha and two beta electrons
per character with nonempty virtual spaces and admits nonzero triples;
measure the correction rather than inferring it from electron count.
Fix initial support at 12 bohr for one-electron and pair radii,
20 Ha reciprocal cutoff, and nuclear Ewald `alpha=0.4` inverse bohr,
12 bohr real radius and 8 inverse bohr reciprocal radius. Use zero fitting
screening and record the retained metric ranks. These values define a
finite test input, not converged physical cutoffs. Refine each support axis
independently under the [fitted-support contract](chi_fitted_support.md).
Use CC energy tolerance `1e-10` Ha, residual tolerance `1e-8`, denominator
tolerance `1e-12` Ha, and an explicit adequate memory cap.

For this same finite Hamiltonian, independently form the complete spin-orbital
integrals from the retained alpha/beta factors and compare the CCSD residuals
and converged energy with a determinant-space calculation where feasible.
Evaluate the stated triples correction by independent determinant contractions,
including a nonzero aab or abb sector. This compares exactly the executed
Hamiltonian but cannot verify the underlying fitted operator's physical
accuracy. Additional cases should cover a same-root closed-shell reference,
a two-electron reference (CCSD equals finite-space FCI and triples vanish),
unequal frozen counts `(1,0)`, and allowed orbital phases/permutations.

For a molecular-limit comparison, use the same LiH2 geometry and STO-3G
orbital basis in canonical ORCA UHF-CCSD(T), all electrons correlated:

```text
! UHF CCSD(T) STO-3G VeryTightSCF NoFrozenCore Bohrs
%mdci
  Include_4thOrder_DT_in_Triples false
end
* xyz 0 2
Li  0.0  0.0  0.0
H   0.0  0.0  3.0
H   2.5  0.0 -1.0
*
```

The ORCA 6.1 manual, section 3.10.2, documents canonical RHF/UHF MDCI and
the additional open-shell fourth-order doubles-triples switch. Record this
switch and verify the matched canonical UHF root; QRO, ROHF, DLPNO and KS
orbitals are different references. First compare CCSD and triples separately.
The ORCA molecular four-center Hamiltonian differs from a finite periodic
fitted chi Hamiltonian, so direct equality is not an acceptance criterion.
Increase the cell and physical supports, refine auxiliary fitting, and quantify
the remaining image, fitting and q=0 differences before using ORCA as a
molecular-limit comparison. CRYSTAL HF agreement can test matched periodic
reference conventions; it does not supply unrestricted CCSD or this triples
correction by inference. If no external program reproduces the finite chi
Hamiltonian, report the determinant comparison and its physical limitation
explicitly, leaving native scientific acceptance pending.

[Classic ROHF and Knowles RMP2](chi_rohf_mp2.md) use a separate shared-orbital
reference and perturbation consumer. They do not admit ROHF-CCSD or triples
through the unrestricted CC solver.
