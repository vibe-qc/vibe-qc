# Experimental chi ROHF and Knowles RMP2

The classic physical-pair chi route provides a stationary restricted open-shell
Hartree-Fock reference and the named `ROHF-RMP2` method. This is the RMP2
partition of [Knowles et al. (1991)](https://doi.org/10.1016/S0009-2614(91)85118-G),
section 2, equations 6-16. It includes singles and doubles and is distinct
from the alternative ROMP partition discussed in that paper. Chi remains
experimental; D93 fitted/post-HF remains not-qualified. Source and algebra
checks are separate from pending native scientific acceptance. Neither R5 nor
ROHF coupled cluster is completed by this increment.

Use `method="aiccm", variant="chi", scf_reference="rohf"`,
`aiccm_backend="ri"`, `fit_source="classic"`, and explicit
`chi_coulomb_controls=ChiCoulombControls(...)`. Add `correlation="mp2"`,
`conv_tol_energy=1e-11` and `conv_tol_grad=1e-9` for RMP2. The library uses
`prepare_chi_hf_reference(..., method="ROHF", backend="ri",
fit_source="classic", coulomb_controls=...)` followed by
`vibeqc.periodic.chi.correlation.run_chi_mp2(reference, ...)`.
The accepted source must be all-electron or use the [scalar ECP contract](chi_ecp.md),
neutral, three dimensional, full
Coulomb, with complete unshifted uniformly weighted characters and fixed
integer occupations at every character. Gamma and complex non-Gamma frames
use the same equations. Historical defaults remain unchanged.

SR ROHF, signed/indefinite retained metrics, spin-orbit ECPs, KS/hybrid correlation,
fractional occupations, lower dimensions, constrained occupations, ROHF-CCSD
and ROHF triples remain unsupported. Original SR metric/operator failures
remain guarded. The implementation imports no CCM construction and never
substitutes a UHF SCF solution.

## Shared spatial reference

For `N` explicit electrons (after ECP core removal) and multiplicity `M`, `n_alpha=(N+M-1)/2` and
`n_beta=(N-M+1)/2` must be integers with `0 <= n_beta <= n_alpha <= rank(S)`.
In one S-orthonormal spatial frame C, the closed, open and virtual spaces are
`c=[0,n_beta)`, `o=[n_beta,n_alpha)` and `v=[n_alpha,n_mo)`. Alpha occupies
c+o and beta occupies c. Thus `D_alpha=C_(c+o) C_(c+o)^H` and
`D_beta=C_c C_c^H`, with total occupations 2/1/0. Each character has the same
populations. The returned `s_squared` and `s_squared_ideal` preserve the
existing periodic diagnostic convention: `S_cell=(n_alpha-n_beta)/2` and
`S_cell*(S_cell+1)`, using the primitive-cell populations. These fields are
not the total spin squared of the finite torus or that total divided by its
cell count. The spin-pure high-spin determinant on `N_k` complete characters
has `S_torus=N_k*S_cell` and total spin squared
`S_torus*(S_torus+1)`.

Both spin Focks use the retained operator:
`F_sigma=h+J[D_alpha+D_beta]-K[D_sigma]`, including the recorded BvK exchange
zero-mode correction. Energy is the character-weighted sum of
`Tr(h D_total) + Tr(J D_total)/2 - sum_sigma Tr(K_sigma D_sigma)/2`, plus the
matched nuclear term. Physical Focks, densities and energy are reconstructed
at the final orbitals; trial optimization operators do not define the returned
Hamiltonian.

With `Fc=(Fa+Fb)/2`, the Hermitian effective Fock in c/o/v order is

```text
        c       o       v
c       Fc      Fb      Fc
o       Fb      Fc      Fa
v       Fc      Fa      Fc
```

This explicitly chooses Fc in all diagonal blocks. It follows the general
coupling freedom in [Guest and Saunders (1974)](https://doi.org/10.1080/00268977400102171),
section 2, equations 12-13, and section 3, equation 15. The arbitrary diagonal
blocks do not change the variational conditions. The ROHF stationarity
conditions are `Fb_co=0`, `(Fa+Fb)_cv=0` and `Fa_ov=0`, as in Knowles equations
6-8. For real rotations the energy derivatives are respectively twice these
blocks; complex rotations also require their imaginary components. A UHF
commutator or a small energy change alone is insufficient.

Optimization uses complex exponential rotations between different occupation
spaces and L-BFGS, followed near stationarity by an analytic gradient-response
Newton correction when energy line searches reach floating-point resolution.
A correction must reduce the physical ROHF residual and preserve energy
consistency. The response differentiates the same retained J/K, including
the exchange zero-mode term; it does not rebuild native integrals. Generic
DIIS and density damping are not used. Only AUTO/HCORE initial orbitals and
zero Fock mixing and level shifts are supported in this increment.
After convergence, display canonicalization diagonalizes Fc independently
inside c, o and v, recorded as `closed-open-virtual-average-fock/v1`.
It preserves both physical spin densities. These display energies are not
RMP2 denominators. The choice differs from the molecular helper's convention
that uses Fa in its open block. No equivalence of displayed orbital energies
with another program is assumed, and stationarity is not a stability proof.

## Perturbation partition and complete expression

Starting from the accepted ROHF state, independently diagonalize Fa in its
occupied (c+o) and virtual (v) spaces, and Fb in its occupied (c) and virtual
(o+v) spaces. This is Knowles section 2, steps 1-5, and equations 10-13.
The resulting spin frames may differ; they are rotations of the same
accepted occupied projectors and leave the determinant unchanged. This is
post-SCF semicanonicalization, not a second SCF optimization.

The result records `knowles-1991-spin-semicanonical/v1` and binds the transformed
frames, energies and active spaces to the accepted source. With spin-orbital
labels and a normal-ordered partition, choose
`H0=E_ROHF+sum_p epsilon_p {a_p^dagger a_p}` and `V=H-H0`.
The spin occupied-virtual physical Fock blocks remain in V. ROHF stationarity
does not force them separately to vanish.

Knowles equations 15-16 give

```text
E2_singles = sum_(i,a) |f_ai|^2 / (epsilon_i-epsilon_a)
E2_doubles = (1/4) sum_(i,j,a,b) |<ab||ij>|^2
             / (epsilon_i+epsilon_j-epsilon_a-epsilon_b)
```

The singles sum is within each spin at the same character. Ordered same-spin
doubles have factor 1/4; opposite-spin doubles have factor one. No extra
closed-shell factor is used. Results separate alpha/beta singles and aa/bb/ab
doubles. In the identical-root closed-shell limit the singles vanish and the
energy reduces to ordinary RHF MP2.

Factors are transported without refitting, whitening, changing retained rank,
kernel, q=0 convention, support or normalization. Complex bra coefficients
are conjugated. `(ia|jb)=L(ki,ka)[i,a] L(kb,kj)[b,j]^H` uses a common transfer
metric, with `ki+kj=ka+kb` modulo reciprocal vectors. Primitive-cell factors
imply doubles normalization `Nk^-3`; singles use `Nk^-1`. The orthogonality
of occupied and virtual frames makes the overlap-only exchange correction
vanish in transition integrals; its contribution to Fock energies remains.
Every allowed excitation must have a finite denominator strictly below
`-denominator_tolerance`; nonnegative and near-zero denominators refuse
without clamping. Momentum-forbidden sectors never contribute.

The all-electron result is invariant to unitary rotations within the original
c/o/v spaces, and to allowed phases, because the spin blocks are rediagonalized.
The reference seal still forbids mutating an accepted object: equivalent
frames must be independently validated and sealed, not edited in place.

## Frozen closed core

`chi_frozen_core=n` (library `frozen_core=n`) selects n lowest eigenvectors
of Fc in the closed space at every character. Equal spin counts are required;
open-shell orbitals cannot be frozen. A degenerate core/active boundary
refuses. The common frozen projector is removed from both spin occupied
spaces before active occupied and virtual spin semicanonicalization.
Its contribution stays in the physical Focks and HF energy, while excitations
from it are excluded. Full retained virtual spaces are used.

This is an explicit projected closed-core extension of Knowles RMP2, not
independent truncation of each spin's lowest canonical orbitals. Frozen-core
invariance is with respect to the fixed selected projector and allowed
rotations; mixing a frozen subspace with active orbitals changes the
approximation. Compare with another implementation only after matching the
frozen projector, active Fock diagonalization and singles convention.

## Persistent reference and RMP2 continuation

The [checkpoint and restart contract](chi_restart.md) persists the shared
ROHF frame, physical alpha/beta Focks and closed/open/virtual identities.
Loading verifies the retained-source Focks, energy and generalized stationarity
before resealing; it never substitutes UHF orbitals. `run_chi_mp2` accepts
`checkpoint_path`, `resume_from` and `checkpoint_interval` for Knowles RMP2.
The saved prefix keeps alpha/beta singles separate from aa/bb/ab doubles and
binds the selected frozen projector and spin-semicanonical convention.
A saved reference is distinct from an explicit convergence-required guess;
the current ROHF driver still refuses initial-density guesses. ROHF-CC and
native scientific acceptance remain separate.

## Validator cases and comparison boundary

The runnable [LiH2 input](../examples/periodic/chi_rohf_rmp2.py) takes
`--mesh 1` or `--mesh 3`, `--frozen-core 0` or `--frozen-core 1`, and a required
`--output` directory outside the source checkout.

The following are reproducible proposed cases, not measured acceptance.
Use the delivered runtime containing this source and submit via `vq`.
Preserve existing #518/#521 CC jobs. Record actual source/Python/native
identities, complete input, geometry, charge/spin, basis/auxiliary basis,
source and reference seals, metric ranks, achieved stationarity, canonicalization,
frozen projector, separated energies and tolerances on the tracking issue.

A neutral LiH2 doublet uses positions in bohr `Li=(0,0,0)`, `H=(0,0,3)`,
`H=(2.5,0,-1)` in an orthorhombic `24 x 26 x 28` bohr cell, STO-3G orbitals,
explicit `def2-SVP-JKFIT` auxiliary basis, and meshes `(1,1,1)` then `(3,1,1)`.
Use one-electron and pair radii 12 bohr, reciprocal cutoff 20 Ha, and nuclear
Ewald `alpha=0.4` inverse bohr, real radius 12 bohr and reciprocal radius
8 inverse bohr. Use zero screening, no reciprocal tail and first no frozen
core, then one shared closed orbital. These define finite cases; refine
supports and auxiliary basis separately before claiming physical accuracy.
Measure singles explicitly and retain a case with a nonzero singles term.

Add H2 singlet in the same cell, positions `(0,0,0)` and `(0,0,1.4)` bohr,
for the identical-root ROHF/RHF and RMP2/MP2 limit. A neutral oxygen atom,
triplet, STO-3G, tests multiple open orbitals. A degenerate requested frozen
boundary or nonnegative excitation gap must be reported as a refusal, not
regularized into a result. Complex meshes test momentum conservation and
phases independently of Gamma success.

For an exact finite-source comparison, construct a small spin-orbital
determinant Hamiltonian independently from the retained factors and physical
one-electron operator. Form H0 using the stated spin semicanonical energies
and compare second-order determinant sums, separately for singles and doubles.
Do not call the production residual or energy contraction helpers. This
isolates algebra and source transport; it does not verify physical fitting
or convergence to the infinite solid.

For a molecular-limit check, converge the image/support/auxiliary limits and
compare the matched ROHF root against ORCA or CRYSTAL where supported.
ORCA 6.1 manual section 3.1.1 describes several ROHF effective-Fock choices;
record the actual choice and compare density and total HF energy before
orbital energies. Default ORCA ROHF canonicalization is not the convention
chosen here. Do not use an unspecified ORCA MP2/QRO or UHF calculation as an
RMP2 reference. A correlation comparison requires the Knowles spin-block H0,
retained singles and matching frozen projector; otherwise state the partition
difference and use the independent determinant construction. Periodic chi
and molecular four-center Hamiltonians also differ in image interactions,
finite fitting and q=0 treatment, so unmatched total energies are not an
acceptance criterion.

For scalar ECP references, freezing applies only to explicit occupied orbitals.
ECP-removed cores are recorded separately and never subtracted a second time;
see the [ECP active-space and persistence conventions](chi_ecp.md).
