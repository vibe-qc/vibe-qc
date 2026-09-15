# BIPOLE Implementation Status

Last updated: 2026-09-09

## Summary

BIPOLE is vibe-qc's own Ewald-J-split periodic SCF path.
All four methods support Gamma and multi-k SCF and the exact finite-difference
atomic-force path. Fixed-cell atomic relaxation is available through
`run_periodic_job(optimize=True)`. Variable-cell and coupled atom/cell
optimization fail closed pending a certified strain convention and terminal
same-geometry force/stress convergence check.

The exact Ewald-J-split path is the production route. Both multipole
far-field implementations are unavailable from the SCF drivers. Passing
`use_multipole_far_field=True` now fails before setup because the dormant
quartet implementation does not preserve the exact contraction's periodic
translation domain. Independent measurement corroborates the retirement:
with the quartet path exercised against exact ERIs, LiH/STO-3G
shows 6.9e-2 Ha (14 bohr), 6.3e-2 Ha (18 bohr), and 5.9e-2 Ha (24 bohr)
error that barely decays with separation, consistent with a domain
mismatch rather than a truncation artefact.

The opt-in `LatticeSumOptions.pair_complete_1e=True` contract uses physical
AO-product separations for the one- and two-electron operators. Its
`cutoff_bohr` controls pair separation; `eri_interaction_cutoff_bohr` controls
the distance between product midpoints (zero uses `cutoff_bohr`). An explicit
or automatically resolved BIPOLE SR extent supplies the latter distance.
Exchange may therefore need output cells beyond the one-electron support.
Physical mode uses the full direct Fock build: the legacy symmetry projector
does not cover that wider exchange support. Explicit projection requests,
periodic COSX and the diagnostic grid Hartree backend refuse this combination.
The global default remains off pending cutoff convergence and performance review.
Finite-domain agreement alone does not certify infinite-lattice convergence;
pair support, interaction reach and reciprocal resolution must all converge.
The [standalone continuation record](bipole_erfc_resume.md) distinguishes
completed local checks from remaining acceptance work.

LiH and Si PBE cold starts at physical 20-bohr support agree with reference-density warm starts within 1e-8 Ha;
20 fixed-density grid cases and the default-screening comparisons are
recorded there. This does not certify the global default or a native
fine-grid SCF limit.

> **⚠ Absolute energies on ionic crystals, FIXED at Γ (all four
> drivers); multi-k corrected gauge available opt-in (2026-06-11).**
> The root-caused defects (spheropole gauge double-count; Γ-locality
> projection dropping real cross-cell 1e/J terms; a formally
> divergent full-Coulomb direct-K series with the Bloch density) are
> fixed by the **Ewald exchange split** (option (b) redesign):
> `K = K_SR(erfc ω) + K_LR(reciprocal, K≠0) + (ξ_M − π/(Vω²))·S·D·S`,
> full-Bloch density, no spheropole term. Validated out-of-process vs
> PySCF GDF: exchange element-wise to 8e-4 / E_K to 0.3 mHa at the
> converged MgO density (ω-invariant); H₂ 12-bohr box converges to
> PySCF to ~1 µHa (RHF), ~0.05 mHa (RKS/SVWN + PBE, the projection
> removal also fixes the XC grid density), ~0.08 mHa (PBE0, the
> α_HF-scaled hybrid split path), +0.07 mHa (UHF triplet) and
> ~1 mHa (UKS).
> **All four drivers default to the corrected gauge at BOTH Γ and
> multi-k (Phase-5 flip, 2026-06-13).** The multi-k q = k−k′ ≠ 0
> LR-exchange channels (option (b) Phase 3, 2026-06-11/12), 
> per-(k,k′) momentum-transfer channels on q-shifted reciprocal
> meshes + the BvK-supercell Madelung correction (α_HF-scaled for
> hybrids, per spin for UHF/UKS), are validated on the H₂ box
> [2,1,1] vs out-of-process PySCF (RHF +0.12 mHa at cutoff 7,
> −0.002 at 12; RKS SVWN/PBE0 +0.04/+0.06; UHF +0.03; UKS k-shift
> 0.013) and to +0.0001 mHa/cell against the explicitly-doubled
> supercell at Γ (the supercell-unfolding identity, the rigorous
> internal proof). The multi-k corrected gauge needs a Monkhorst-Pack
> mesh; an ad-hoc k-list (band path / explicit list) at multi-k falls
> back to the legacy gauge under the auto default. Pass
> `use_exchange_ewald_split=False` for the legacy gauge.
> **Outstanding heavy confirmation:** the MgO [2,2,2] c12 RHF+RKS
> parity run (`examples/regression/bipole_parity/mgo_multik_parity.py`)
> is staged for the queue, belt-and-suspenders on top of the
> supercell identity, fleet-blocked at the 2026-06-13 flip.
> **Two practical caveats for ionic Γ work:** (1) the corrected gauge
> contracts full Bloch folds, diffuse AO tails (e.g. STO-3G Mg 3sp)
> need fold-converged cutoffs (MgO/STO-3G: ≥12-16 bohr; the driver
> fails before SCF above ``1e-2`` S(Γ)-fold drift and reports intermediate
> truncation until the quantitative ``1e-4`` target is met); (2)
> minimal-basis ionic Γ
> SCF has multiple genuine solutions (PySCF itself lands on different
> totals from different starts on MgO/STO-3G Γ), check the basin
> against a reference density for absolute claims. Forensics + tables:
> `handovers/HANDOVER_BIPOLE_PRODUCTION.md` §0a;
> `examples/regression/bipole_parity/mgo_exchange_split_probe.py`.

> **Production forces use finite differences of a converged driver energy.**
> `compute_bipole_gradient_fd` differentiates the actual driver energy,
> so it follows whichever exchange gauge the SCF used (corrected or
> legacy); `bipole_optimize`,
> `run_periodic_job(optimize=True)`, and periodic NEB default to it.
> **Asymmetric legacy-gauge HF is not a supported converged-SCF route.**
> The BeH2 and BeH legacy-gauge regression rows failed the exact terminal
> stationarity check even after their overlap folds were converged; those
> capability claims are retired (#66). Increasing the cutoff does not make
> a nonstationary density valid for a finite-difference gradient.
> **The analytic BIPOLE gradient began with the LEGACY Gamma-local gauge.**
> Its maintained scope is narrower than the corrected-gauge paths below.
> **Corrected-gauge analytic-gradient prototypes landed 2026-06-17 for all
> four methods (RHF/UHF/RKS/UKS) at both Γ and multi-k** (G3 row below).
> They remain a gated preview; the legacy-gauge analytic preview remains
> available and covers:
> * **RHF/UHF Gamma and multi-k**, maintained symmetric controls; the
>   asymmetric legacy-gauge HF rows no longer certify this preview.
> * **RKS/UKS Gamma-local**, maintained LDA asymmetric cells with
>   XC Pulay, moving-grid correction, and KS Bloch-CPHF.
> * **RKS/UKS multi-k**, diagonal-Z + corrected-W (the v_bg/spheropole
>   W convention) + J^LR + XC Pulay; warns. KS CPHF gated to Gamma.
> * Finite-temperature/fractional-occupation analytic gradients are now
>   **landed in the corrected gauge** (Mermin free-energy form; FD-validated
>   RKS Gamma ~2e-8 / multi-k ~4e-9). The legacy gauge above still rejects
>   them. The padded M5 Gamma composition also fails closed for fractional
>   occupations until its finite-domain J/K map has a variational
>   density-transpose completion.
> * Meta-GGA tau-Pulay is **landed** (un-gated 2026-06-20; validated to
>   1e-9 against an independent reimplementation; see G3).
> * The production M5 padded erfc domain is now differentiated for
>   **corrected-gauge RHF, UHF, and integer-occupation RKS/UKS Gamma** with
>   radial or pair-resolved SYM3b output. Pure-RKS radial parity on the post-exact-FT
>   production SR+LR surface is 1.14e-7 Ha/bohr; pair-resolved UHF is
>   4.1e-6 Ha/bohr and pair-resolved LDA/PBE0 RKS pass below 1e-5 Ha/bohr.
>   The radial finite-domain density adjoint closes the fully occupied
>   triplet-H2 UHF discriminator from 2.906e-3 to below 1e-6 Ha/bohr and brings
>   padded LDA/PBE0 UKS below 1e-5 Ha/bohr. The pair-mask transpose is also
>   complete for UHF: its off-boundary saturated-spin discriminator improves
>   from 2.31e-3 to 9.85e-8 Ha/bohr. Pair-resolved LDA/PBE0 UKS pass at
>   1.11e-7 Ha/bohr. Corrected-gauge integer-occupation RHF also supports the
>   padded radial domain at multi-k: the finite-domain density adjoint and full
>   real-linear k-coupled response pass the asymmetric BeH2 [2,1,1]
>   production-energy finite difference at the fold-reliable 13-bohr cutoff.
>   The RHF response now solves the weighted transpose equation with consistent
>   complex rotations; the former negative-curvature expected failure is an
>   ordinary passing regression at its unchanged 5e-5-Ha/bohr tolerance.
>   See [the continuation evidence](bipole_erfc_resume.md) for the operator
>   diagnostic and validation scope. Corrected-gauge
>   integer-occupation UHF also supports padded radial multi-k: preserving the
>   full BvK support of `D_alpha + D_beta` plus the spin-coupled response
>   closes asymmetric BeH [2,1,1] parity to 6.83e-8 Ha/bohr. Padded multi-k
>   KS, pair-resolved, and fractional routes remain fail-closed.
>   Fixed-density J/K and method-level full-SCF finite differences pin the
>   supported slices; FD remains the production force path.
> * The **legacy multi-k KS-CPHF** remains a
>   ratified deferral (2026-06-20), legacy-gauge-only, since the
>   corrected-gauge multi-k KS path is variational and needs no CPHF; see
>   `handovers/HANDOVER_BIPOLE_GRADIENT.md` Case 3.
> * Space-group symmetry: attached symmetry auto-enables both the
>   bit-exact SYM2c one-electron S/T reduction and, on the corrected
>   Ewald exchange split, the SYM3b pair-resolved Fock reduction. The
>   M5 Fock default is safe because every erfc SR build now uses a
>   precision-derived padded ket-image ball with charge-pair Schwarz
>   screening that includes angular and contraction tails. The full pair-resolved output, density, and internal
>   domains are group-invariant: reduce equals enforce end to end, and
>   J-block orbit asymmetry is 6.3e-15 on MgO, 5.4e-15 on diamond, and
>   4.4e-15 on Si at cutoff 6. Pass
>   `use_fock_symmetry_reduce=False` to disable the automatic reduction;
>   explicit enforcement-only diagnostics use
>   `use_fock_symmetry=True, use_fock_symmetry_reduce=False`.
> * The M5 erfc SR default is `sr_image_precision=1e-6`. It derives an
>   absolute radius as `cutoff_bohr + bipole_sr_image_extent(...)` and
>   enables `LatticeSumOptions.sr_range_screening`. Pass
>   `sr_image_precision=None` only to reproduce the historical truncated
>   domain; on an active erfc SR route this also suppresses automatic SYM3b
>   reduction unless it is explicitly requested. An explicit
>   `sr_image_extent_bohr` remains the absolute-radius oracle and takes
>   precedence. HSE screened exchange uses the same padded internal domain;
>   the longer-ranged of its erfc K kernel and the Ewald erfc J kernel sets the
>   precision-derived radius. Periodic `run_neb` and `relax_atoms` expose and
>   forward the same `sr_image_precision=1e-6` production default. Their
>   `None` setting is an explicit historical-domain diagnostic, not a hidden
>   cap on the production image domain.
> All production force paths default to the exact FD gradient
> `compute_bipole_gradient_fd`.

## Completed

| Milestone | Description |
|---|---|
| M1-M12 | Foundation, spheropole diagnosis, multipole infrastructure |
| M13 | Multi-k via `kpoints` kwarg |
| M14 | RKS/UKS/UHF promoted to user API |
| M15 | Multipole near/far split fix |
| G2 | C++ spheropole kernel (all cases through d-d) |
| G3 | FD gradient (all 4 methods, exact) + analytic research preview |
| M16 | 2D/1D gradient support documented |
| M17 | `optimize=True` in `run_periodic_job` |
| M18 | Force-virial diagnostic; production cell optimization fails closed |

## User API

```python
from vibeqc.periodic_runner import run_periodic_job

# SCF + atomic relaxation
result = run_periodic_job(system, basis, method="RHF", jk_method="bipole",
                          optimize=True, optimize_max_iter=30)

# Standalone force virial diagnostic (not the periodic stress)
from vibeqc.bipole_gradient import compute_stress_tensor
virial = compute_stress_tensor(system, gradient)
```

## Remaining Gaps

| # | Task | Priority | Status |
|---|---|---|---|
| G1 | Cell-level multipole far-pair | HIGH | **RETIRED / DRIVER-UNREACHABLE.** The G1 builder is retained only as a research artifact. No direct or public driver can activate it; explicit multipole requests fail before SCF. |
| G2 | Spheropole higher-l terms | HIGH | **LANDED**, +4.112 Ha vs CRYSTAL +4.119 Ha |
| G3 | Production force/stress coverage | MEDIUM | Finite-difference atomic forces differentiate the supported exact driver energy. Variable-cell optimization and the standalone far-field gradient prototypes are not certified and fail closed. |
| G4 | Space-group symmetry for one- and two-electron integrals | MEDIUM | **LANDED**, bit-exact SYM2c S/T and pair-resolved SYM3b Fock reduction auto-enabled for attached-symmetry corrected-split crystals; reduce equals enforce to machine precision with the M5 padded SR domain |
| G5 | Exact-zone / fold-range decoupling (BIPOLE-EXACT-ZONE-UNBOUNDED) | HIGH | **Increment 1 LANDED 2026-08-06** (`exact_zone_bohr`, RHF + RKS, opt-in): the exact erfc output zone is bounded independently of the operator/fold cutoff; far operator cells carry `J_LR` + background (the Ewald far-field model -- the Dovesi 1983 Eq. (24a) bielectronic/monoelectronic partition in the split gauge; NOT the retired G1 cell-level branch, which stays retired). Measured fixed-density tails: MgO/STO-3G 2.7e-3 Ha @ zone 10, 3.2e-4 @ 12; LiH-class diffuse bases keep mHa tails to ~14 bohr -- per-case ladder knob. Auto SYM3b reduction yields to a zone request; explicit reduction and analytic gradients fail closed. Next: UHF/UKS + auto-derived fold cutoff wired with the zone, then the quartet-level *screened* bipolar far field (pair-resolved, S92 Eq. (92) Boys-function radial ladder) as increment 2 -- the scaling fix proper. Any future quartet far-field increment must first preserve the common translation sum in Pisani-Dovesi-Roetti 1988 Eq. II.4.10; this exact-zone result is not evidence for that route. |
| G6 | Quartet-level bipolar far-field | **P0** | **FAIL-CLOSED 2026-08-13.** The driver gate was dormant, and enabling it directly would be unsafe: the exact Fock contraction spans three lattice translations and contracts density at their relative translation, while the dispatch, skip mask, and add-back span only two. Additional blockers include the penetration rule, screened kernel, total-order ceiling, Coulomb/exchange composition, symmetry reconstruction, and gradients. All four drivers default to `False`; explicit `True` raises before SCF. Independent exact-vs-far-field measurement corroborates the retirement: LiH/STO-3G shows 6.9e-2 Ha (14 bohr), 6.3e-2 Ha (18 bohr), 5.9e-2 Ha (24 bohr), barely decaying with separation -- a domain mismatch, not truncation. The restricted Fock builder's dormant G6 branch was also made internally consistent (skip-masked near-K for both the Fock and the exchange energy) so any future re-wiring starts from a coherent reference. |
| G7 | Penetration dispatch and contractor prototypes | HIGH | **NOT ROUTE-CERTIFIED.** Isolated Python/C++ parity tests cover only internal representations. They do not establish a literature basis for the implemented classifier, do not validate the published penetration criterion or the exact/add-back periodic domain. Order zero is currently overloaded as an exact sentinel, the classifier omits the second pair width, and screened caches and symmetry reconstruction need independent equation-level oracles before reuse. |
