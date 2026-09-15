# Periodic (k-point) analytic basis-parameter gradient: design / scoping

This is the periodic counterpart of
[`ENERGY_GRADIENT_DESIGN.md`](ENERGY_GRADIENT_DESIGN.md) (the molecular
analytic basis-optimisation gradient, now complete: RHF/UHF/RKS/UKS ×
exponents + coefficients, all wired into `optimize_bdiis`). Periodic basis
optimisation at the DFT level is the actual CRYSTAL **OPTBASIS** target, and
the production multi-crystal recipe (`recipes/production.py`) currently
optimises derivative-free (NLopt BOBYQA). An analytic gradient unlocks the
robust BDIIS loop for crystals.

This document scopes that work. **Implemented and validated: Phase P0 (assembly),
Phase P1 (one-electron overlap + kinetic exponent derivatives), the Phase P2
assembly skeleton (behind a provider seam), and Phase P3 (RKS + UKS LDA/GGA
explicit-XC term). The one piece still missing for an end-to-end gradient is the
P2 derivative kernel itself: the Ewald-gauge dV(k)/dG(k), filed as R13.** It captures the
approach, the reusable pieces, and the hard parts so a focused effort can resume
with a plan. Per-phase status is marked in "Hard parts" below.

## The assembly (Phase P0: DONE, build-independent)

For a periodic SCF with normalised k-weights `w_k` (Σ_k w_k = 1) and complex
Hermitian per-k density `P(k)`, the frozen-density Pulay energy gradient is the
BZ sum of the molecular expression:

    dE/dη = Σ_k w_k · Re[ tr(P(k)·∂Hcore(k)/∂η) + ½ tr(P(k)·∂G(k)/∂η)
                          − tr(W(k)·∂S(k)/∂η) ]

with `W(k) = ½·P(k)·F(k)·P(k)`, `F(k) = Hcore(k) + G(k)`. The single-k Γ-only
real limit is exactly the molecular `energy_gradient_fd`.

Implemented and validated build-free against a multi-k mock (per-k Hermitian
generalized eigenproblem) in
[`periodic_energy_gradient.py`](../../python/vibeqc/basis_optimization/periodic_energy_gradient.py)
/ `test_periodic_energy_gradient_assembly.py`. The integral source is an
injected `PeriodicIntegralProvider`; the hard pieces below drop in behind it,
exactly as the molecular Phase-1 integral derivatives dropped in behind
`IntegralProvider`.

## Reusable pieces

* **Molecular integral exponent/coeff derivatives** (`cpp/src/basis_param_gradient.cpp`
  `*_exponent_derivative`; the augmented-basis coefficient + grid-XC `∂φ/∂η`
  machinery in `energy_gradient.py`). The real-space periodic derivatives
  `∂S(R)/∂η` between a cell-0 shell and a cell-R image are the *same* molecular
  integral derivatives evaluated against a shifted shell, the Bloch sum then
  layers on top.
* **The existing periodic nuclear gradient** (`compute_gradient_periodic_rhf_gamma`
  / `_multi_k`, `compute_gradient_periodic_rks_*`, and the FD reference
  `compute_gradient_periodic_rhf_fd`). This already carries the lattice-sum
  Pulay structure, the GDF/Ewald Coulomb-derivative machinery, and the
  periodic-grid XC derivative for ∂/∂R, the periodic-basis gradient replaces
  ∂/∂R with ∂/∂η at the same contraction sites.
* **The periodic XC builders** (`build_xc_periodic`, `build_xc_periodic_uks`,
  `build_periodic_becke_grid`) for the explicit XC term.

## Hard parts (phased)

* **P1: Bloch-summed one-electron derivatives. [DONE for overlap + kinetic.]**
  `∂S(k)/∂η = Σ_R e^{ik·R} ∂S(R)/∂η` and the kinetic analogue ∂T(k)/∂η are
  implemented in
  [`periodic_energy_gradient.py`](../../python/vibeqc/basis_optimization/periodic_energy_gradient.py)
  (`bloch_summed_one_electron_exponent_derivatives`). Each real-space block
  `∂S(R)/∂η` reuses the molecular `overlap_/kinetic_exponent_derivative` bindings
  between a home-cell shell and its image at lattice vector R (a doubled-cell
  molecule; sum the home and image copies of the shared exponent, read the
  home x image cross-block), Bloch-summed over the SCF's own overlap-lattice
  cells. Validated to ~1e-10 vs a central FD of the Bloch-summed lattice
  integral at fixed k, off the reference exponent, in
  `tests/basisset_dev/test_periodic_one_electron_exponent_deriv.py`. The
  nuclear-attraction `∂(nuclear)/∂η` is NOT clean two-centre: the GDF multi-k
  path builds V via an Ewald-3D-gauge nuclear lattice
  (`compute_nuclear_lattice_dispatch` at `_gauge_lat_opts_for_v_ne_and_e_nuc`),
  so it moves to P2 with the rest of the Ewald gauge.
* **P2: Coulomb derivative ∂G(k)/∂η + nuclear ∂V(k)/∂η (the hardest).** Periodic
  J/K is GDF (3-index `L_pq(k)` density-fit tensors) or real-space Ewald. The
  basis-param derivative needs ∂L/∂η (3-index integral derivatives w.r.t.
  exponents/coeffs, *including the auxiliary-basis response if the aux is
  basis-dependent*) plus the long-range Ewald derivative. The one-electron
  nuclear term ∂V(k)/∂η rides here too: V is built with the Ewald-3D gauge
  (`compute_nuclear_lattice_dispatch`), the same gauge as J, so it must be
  differentiated consistently (cf. the d/dR analogue in
  [`periodic_gdf_gradient.py`](../../python/vibeqc/periodic_gdf_gradient.py),
  which does V_ne via a gauge-consistent integral FD,
  `_v_ne_gradient_numerical`). Open question: whether to differentiate the GDF
  fit + Ewald V analytically or fall back to an integral-FD `∂G(k)` / `∂V(k)`
  behind the provider for a first working version. A fully analytic route would
  want a lattice-summed *exponent*-derivative 3c/2c kernel, the ∂/∂η sibling of
  the existing `compute_3c_eri_lattice_gradient_weighted` (∂/∂R); that is the
  one piece that would touch periodic-SCF C++ (filed as R13 in
  [`REQUIREMENTS-PERIODIC.md`](REQUIREMENTS-PERIODIC.md); coordinate per CLAUDE.md
  s11). **The assembly skeleton that consumes it is already in place:**
  `periodic_energy_gradient_analytic` pulls per-k P(k)/F(k)/k-weights from the
  GDF result, takes dS(k)/dT(k) from P1, and injects dV(k)/dG(k) through a
  `PeriodicCoulombNuclearDerivativeProvider` seam (a mock provider validates the
  wiring + the `_periodic_pulay_contract` math today); the day R13 lands, the
  full gradient is a drop-in. Scope: closed-shell RHF (+ hybrid exact exchange in
  dG); exponent params only. Pure-DFT XC is P3 (RKS + UKS done).
* **P3: periodic explicit XC term. [DONE for RKS + UKS, LDA/GGA.]**
  `∂E_xc/∂η = Σ_g w_g [v_ρ ∂ρ/∂η + 2 v_σ ∇ρ·∂∇ρ/∂η]` is implemented in
  `periodic_xc_param_gradient_term` (closed-shell) and
  `periodic_xc_param_gradient_term_uks` (open-shell, the polarised
  v_ρα/v_ρβ/v_σαα/v_σαβ/v_σββ contraction), with the
  `build_periodic_xc_gradient_grid`/`_uks` frozen-grid builders. The
  density is the full cross-cell periodic lattice form `ρ(r) = Σ_a Σ_s Σ_μν
  χ_μ(r−R_a) P_μν(s−a) χ_ν(r−R_s)` on the periodic Becke grid
  (`build_periodic_becke_grid`): BOTH AOs sum over lattice cells (bra cell a, ket
  cell s, block at their difference g = s−a), matching the `build_xc_periodic`
  energy since the 295f7ada dense-crystal fix. The bra/ket cell sets mirror the
  C++ `active_density_cells` split (ket screened by `cutoff_bohr`, bra-image by
  `min(becke_image_radius_bohr, cutoff_bohr)`), so pass the SCF's `lattice_opts`
  to the builder. The molecular `_param_dphi_on_grid` `∂φ_μ/∂η` carries over
  verbatim, evaluated at the shifted points r−R_c of whichever cell the AO sits
  in; η drives the target shell in BOTH the bra-image χ_μ(r−R_a) and the ket
  χ_ν(r−R_s), so `∂ρ/∂η` carries a derivative term for each (the Phase-P1 "home +
  image copies of the shared parameter" structure, extended to the bra-image
  cells). The Python density assembly reproduces `build_xc_periodic`(`_uks`)'s
  E_xc to ~1e-11 (pinned convention test, the C++ side independently checked
  against a NumPy oracle in `tests/test_periodic_xc_cross_cell.py`), so
  `∂E_xc/∂η` is validated against a frozen-density E_xc FD (RKS + UKS, LDA + GGA)
  in `tests/basisset_dev/test_periodic_xc_param_gradient.py`. Meta-GGA (tau term)
  is later. Exponent params validate in isolation; a coefficient derivative needs
  the full energy FD (its normalisation response only cancels there) so it
  validates with R13.
* **P4: expose per-k P(k)/F(k). [Already satisfied for the GDF multi-k path.]**
  Re-audited 2026-06-21: `PeriodicKRHFGDFResult` / `PeriodicKRKSGDFResult`
  (`periodic_k_gdf.py`) already expose, as length-nkpts Python lists, the
  converged per-k `density`, `fock`, `overlap`, `hcore`, `mo_coeffs`,
  `mo_energies`, and `occupations`, plus `kpoints_cart` / `kpoint_weights`. That
  is exactly the per-k P(k)/F(k)/S(k)/Hcore(k) the assembly consumes, so the GDF
  multi-k SCF can feed it with no new C++ binding. The Ewald multi-k result
  (`PeriodicRHFMultiKEwaldResult`) exposes per-k `fock`/`overlap`/`hcore`/
  `mo_coeffs`/`mo_energies` but only a real-space Bloch-folded `density`
  (LatticeMatrixSet); its per-k P(k) is reconstructable from `mo_coeffs` +
  `occupations`. Target the GDF path first; no P4 binding work is needed there.

## Validation strategy (mirrors molecular)

1. Per-k assembly vs mock total-energy FD: **done** (P0).
2. Each Bloch-summed integral derivative vs FD of the Bloch integral at fixed k:
   **done for overlap + kinetic** (P1,
   `test_periodic_one_electron_exponent_deriv.py`); nuclear + two-electron with
   P2.
3. Full analytic gradient vs a periodic full-energy FD (a basis-param analogue
   of `compute_gradient_periodic_rhf_fd`), on a small cell at a coarse k-mesh,
   validated **off the reference basis** and against PySCF.pbc out-of-process
   (per CLAUDE.md § 10) where feasible. Heed the molecular lessons: validate
   off x0 (a reference/current slip is invisible at x0), and use a precise
   (ShellInfo, not `.g94`) FD for tiny exponent steps.

## Escalation

P4 (per-k matrix exposure) turned out to need no C++: the GDF multi-k result
already exposes everything the assembly consumes (see P4 above). P2 (GDF/Ewald
Coulomb + nuclear derivative) remains the large piece. Its first **working** (if
not fully analytic) crystal-BDIIS path can use an integral-FD `∂G(k)` / `∂V(k)`
behind the provider, fed by a real GDF multi-k SCF, with no periodic-SCF C++
change. Only the *fully analytic* GDF/Ewald route needs a new lattice-summed
exponent-derivative 3c/2c kernel in C++ (the ∂/∂η sibling of
`compute_3c_eri_lattice_gradient_weighted`); surface that to the periodic-SCF
chat before implementing (CLAUDE.md s11), since it lives in their integral
layer.
