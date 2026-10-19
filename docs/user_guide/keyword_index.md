---
myst:
  html_meta:
    "description": "Flat index of commonly used SCF and job option keywords in vibe-qc with their default values: convergence tolerances, SCF acceleration and damping, second-order methods, initial guess, Fock-build strategy, basis and ECP, DFT grid, open-shell and periodic options, and the option enumerations."
    "og:title": "vibe-qc keyword index (common options and defaults)"
---

# Keyword index

Common SCF and job option keywords in vibe-qc, with their default values. These
options live on the per-method option objects (`RHFOptions`, `RKSOptions`,
`UHFOptions`, `UKSOptions`, and their periodic counterparts
`PeriodicRHFOptions` and `PeriodicKSOptions`), which you pass to a run via
`rhf_options=`, `rks_options=`, `ks_options=`, and so on. The defaults below
are the values a freshly constructed options object carries; this page is
the flat lookup, and each section links the topic page with the physics.

## Convergence and tolerances

| Keyword | Default | Meaning |
|---|---|---|
| `max_iter` | `100` | Maximum SCF iterations. |
| `conv_tol_energy` | `1e-8` | SCF energy-change convergence threshold (Hartree). |
| `conv_tol_grad` | `1e-6` | Orbital-gradient (DIIS-error) convergence threshold. |

See [scf_convergence](scf_convergence.md).

## SCF acceleration and damping

| Keyword | Default | Meaning |
|---|---|---|
| `scf_accelerator` | `EDIIS_DIIS` (molecular + periodic) | Extrapolation scheme; one of `DIIS`, `EDIIS`, `EDIIS_DIIS`, `ADIIS`, `ADIIS_DIIS`, `KDIIS`, `R_CDIIS`, `AD_CDIIS`. |
| `use_diis` | `True` | Enable DIIS extrapolation. |
| `diis_start_iter` | `2` | Iteration at which DIIS begins. |
| `diis_subspace_size` | `8` | Number of stored vectors. EDIIS, ADIIS, and their hybrids accept 2-12; plain DIIS-family accelerators may use deeper histories. |
| `ediis_diis_switch_threshold` | `0.1` | Intensive (RMS matrix-element) DIIS error at which EDIIS hands over to plain DIIS. |
| `ediis_diis_switch_threshold_open_shell` | `1e-3` (UHF only) | Same, used instead when n_α ≠ n_β (#377). |
| `diis_restart_tau` | `1e-4` | R_CDIIS restart parameter τ ∈ (0,1) (Chupin et al. 2021); smaller ⇒ deeper history. |
| `diis_adaptive_delta` | `1e-4` | AD_CDIIS depth parameter δ > 0 (Chupin et al. 2021); smaller ⇒ deeper history. |
| `damping` | `0.5` | Finite static density/Fock damping fraction in `[0, 1)`. |
| `dynamic_damping` | `True` (molecular + periodic) | Adjust the damping factor automatically (ORCA/CRYSTAL-style; default v0.15.x). |
| `dynamic_damping_min` / `_max` | `0.0` / `0.95` | Bounds for dynamic damping. |
| `fock_mixing` | `0.0` | Finite CRYSTAL-style static Fock-mixing fraction in `[0, 1)` (FMIXING). |
| `level_shift` | `0.0` | Saunders-Hillier virtual-orbital level shift (Hartree). Molecular + periodic. |
| `level_shift_warmup_cycles` | `-1` | Auto-reducing shift warm-up: `-1` auto (shift a few startup cycles then release), `0` persistent, `N` explicit length. Molecular + periodic. |
| `level_shift_schedule` | `[]` | Explicit per-iteration shift curve (CRYSTAL `LEVSHIFT B IRESET`); supersedes `level_shift` + warm-up. Lower a `LevelShiftSchedule` via `.apply_to(opts)`. Molecular + periodic. |

See [scf_convergence](scf_convergence.md).

## Second-order and Newton methods

Each is off (`0.0`) by default and turns on when the DIIS error drops below
the given threshold.

| Keyword | Default | Meaning |
|---|---|---|
| `newton_threshold` | `0.0` (off) | Switch to full-Hessian Newton below this DIIS error. |
| `soscf_threshold` | `0.0` (off) | Opt-in: switch to the second-order SOSCF (L-BFGS quasi-Newton) finalizer below this error norm; `0.0` disables. |
| `trah_threshold` | `0.0` (off) | Switch to trust-region augmented-Hessian (TRAH). |
| `quadratic_fallback_iter` | `0` (off) | Iteration at which to trigger the quadratic fallback. |

## Iterative diagonalization (Davidson)

| Keyword | Default | Meaning |
|---|---|---|
| `use_davidson` | `False` | Replace full diagonalization with blocked Davidson. |
| `davidson_min_dim` | `100` | Minimum AO basis size to activate Davidson. |
| `davidson` | `DavidsonOptions()` | Per-knob Davidson control. |
| `solver` | `"dense"` | Diagonalization keyword: `"dense"`, `"davidson"`, or `"lobpcg"`. |

## Initial guess

| Keyword | Default | Meaning |
|---|---|---|
| `initial_guess` | `AUTO` (molecular and periodic) | Starting density; one of `AUTO`, `HCORE`, `SAD`, `SAP`, `PATOM`, `HUECKEL` (`HUCKEL` alias), `MINAO`, `READ`, or `FRAGMO` on supported molecular and periodic routes. |
| `read_path` | `''` | File (`.qvf` / `.molden`) to read orbitals from, for `initial_guess=READ`. |

See [initial_guess](initial_guess.md).

## Fock-build strategy

| Keyword | Default | Meaning |
|---|---|---|
| `scf_mode` | `AUTO` | `AUTO`, `CONVENTIONAL` (in-core), or `DIRECT` (on-the-fly, screened). |
| `scf_mode_auto_threshold` | `140` | Basis size above which `AUTO` selects `DIRECT`. |
| `density_fit` | `False` | Use RI / density fitting for the Coulomb (J) build. |
| `cosx` | `False` | Use COSX-K inside the density-fitting route. `run_job(cosx=True)` enables DF automatically; low-level option objects must also set `density_fit=True`. |
| `schwarz_threshold` | `1e-10` | Schwarz integral-screening threshold. |
| `schwarz_threshold_loose` | `1e-7` | Early-iteration Schwarz threshold; set no larger than `schwarz_threshold` to disable loose screening. |
| `schwarz_threshold_tighten_at` | `1e-3` | SCF gradient norm below which direct SCF switches to `schwarz_threshold`. |
| `incremental_fock` | `True` | Build the Fock matrix incrementally between iterations on the direct path. |
| `incremental_fock_reset_freq` | `8` | Full direct-Fock rebuild interval used to bound incremental drift. |

See [scf_modes](scf_modes.md) and [density_fitting](density_fitting.md).

## Basis, auxiliary, and ECP

| Keyword | Default | Meaning |
|---|---|---|
| `aux_basis` | `''` | Auxiliary fitting-basis label. `run_job` auto-selects one for DF/COSX when possible; low-level `run_rhf` / `run_rks` / `run_uhf` / `run_uks` require an explicit value. |
| `ecp_library` | `''` | Effective-core-potential library name. |
| `ecp_centers` | `[]` | Per-center ECP assignments. |
| `linear_dep_threshold` | `1e-7` | Overlap-eigenvalue cutoff for linear-dependence filtering. |

See [basis_sets](basis_sets.md), [ecp](ecp.md), [linear_dependence](linear_dependence.md).

## DFT (RKS / UKS)

| Keyword | Default | Meaning |
|---|---|---|
| `functional` | `'LDA'` | XC functional: a libxc name, weighted-sum specification, or registered external full-grid provider such as `skala-1.1`. |
| `grid` | *(medium grid)* | XC quadrature grid; a `GridOptions` object (see below). |
| `grid_level` | `"orca-defgrid3"` *(argument of `run_job`)* | High-level grid preset, applied whenever the KS options' grid is untouched (no options object, or one whose `grid` was never customised, #663); a customised grid wins. Exact SKALA aliases automatically select `"skala"` on an untouched grid. |

See [functionals](functionals.md) and [Microsoft SKALA-1.1](skala.md).

### XC grid (`GridOptions`)

| Keyword | Default | Meaning |
|---|---|---|
| `partition` | `Becke` | Atomic partitioning scheme. |
| `angular` | `ProductGaussLegendre` | Angular quadrature; `ProductGaussLegendre` or `Lebedev`. |
| `n_radial` | `75` | Radial quadrature points. |
| `n_theta` / `n_phi` | `17` / `36` | Product-grid angular points. |
| `lebedev_order` | `29` | Lebedev order (used when `angular=Lebedev`): 5, 7, 11, 15, 17, 23, 27, 29, 31, 35, 41, 47, 53, and for convergence studies 59, 65, 77, 95 (1202 to 3074 points). See the [grid tutorial](../tutorial/dft_grid_convergence.md). |
| `becke_k` | `3` | Becke partition smoothing order. |
| `vv10_grid_factor` | `3.0` | VV10-paired functionals (`vv10`, `wb97x-v`, `wb97m-v`) evaluate the nonlocal double integral on a separate grid with the radial and angular resolution divided by this factor (Vydrov and Van Voorhis 2010 practice). `1.0` (or `0`) evaluates VV10 on the full XC grid, the bitwise pre-v0.16 reference. |
| `atomic_grid_profile` | `Generic` | Coherent atom-specific grid contract. `"pyscf-level3"` or `"skala"` selects vibe-qc's pinned SKALA parity profile and overrides the individual radial, angular, pruning, and partition controls. |

## Open-shell (UHF / UKS, and periodic)

| Keyword | Default | Meaning |
|---|---|---|
| `atomic_spins` | `[]` | Per-atom broken-symmetry spin seed (the ATOMSPIN analogue). |
| `spinlock_mode` | `OFF` | Broken-symmetry hold mode for the spin pattern. |
| `spinlock_iterations` | `0` | Iterations to hold the spin pattern. |
| `spinlock_value` | `0` | The locked spin value. |
| `stability_check` | `True` | Molecular UHF/UKS: post-convergence internal stability analysis + corrective restart; implicit UKS checks skip when the complete response is unavailable (meta-GGA, range-separated, VV10, DFT+U, active hybrid RIJCOSX, or coupled solvent) ([details](scf_convergence.md#internal-stability-analysis-molecular-uhfuks-default-on)). |
| `stability_tol` | `1e-4` | Instability threshold on the lowest orbital-rotation Hessian eigenvalue. |
| `stability_max_retries` | `3` | Rotate-and-reconverge escapes before reporting the instability unresolved. |
| `stability_davidson_max_iter` | `80` (UHF), `60` (UKS) | Matvec cap per stability eigensolve (one J + two K builds each). |

## Periodic (`PeriodicRHFOptions` / `PeriodicKSOptions`)

The periodic option objects share the convergence, acceleration, damping,
level-shift, initial-guess, and iterative-diagonalization controls above. They
do not carry the molecular `scf_mode`, density-fitting, COSX, or direct-SCF
screening fields; periodic J/K algorithms are selected by route instead. The
periodic `scf_accelerator` default is `EDIIS_DIIS`, the same as molecular.
Periodic jobs additionally expose:

| Keyword | Default | Meaning |
|---|---|---|
| `jk_method` | `AUTO` *(argument of `run_periodic_job`)* | Coulomb route: `GDF`, `BIPOLE`, `GPW`, `GAPW`, `RSGDF`, and others. See [choosing a periodic method](../tutorial/periodic_methods_compared.md). The AICCM formulations are not `jk_method` values: select them with `method="aiccm"` and `variant=` (see [AICCM](aiccm.md)); with `method="aiccm"`, `jk_method` stays `AUTO` or must equal the variant's own route. |
| `variant` | *(none; mandatory with `method="aiccm"`)* | AICCM formulation: `"real-gamma"`, `"neutral-bloch"`, `"chi"`, and `"four-center"`. All four runner arms are wired and experimental; each keeps its documented, fail-closed capability envelope. |
| `scf_reference` | `None` *(argument of `run_periodic_job`, `method="aiccm"` only)* | `None` infers the SCF reference (`functional` decides HF versus KS; the multiplicity and electron parity decide restricted versus unrestricted); `"rhf"` / `"uhf"` request a Hartree-Fock reference explicitly on every variant (no `functional`; `"uhf"` mainly for a closed-shell singlet, `"rhf"` refused on an odd electron count); `"rohf"` / `"roks"` request the restricted open-shell references explicitly (no variant implements them yet; fails closed). |
| `aiccm_lattice_extension` | `None` *(argument of `run_periodic_job`)* | Born-von Karman torus `(N1, N2, N3)` for every AICCM variant; `kpoints` is the legacy Γ-centred alias for the torus mesh, never both. `aiccm_wigner_seitz_shells=s` is the odd-extension shorthand for `2s+1`. |
| `aiccm_cderi_storage` | `None` *(argument of `run_periodic_job`, `variant="real-gamma"` only)* | How the real-Gamma producer holds its fitted integrals: `"dense"` (what `None` runs; memory grows as the cube of the torus size), `"block"` (one block per transfer of the block-circulant torus) or `"block-star"` (only the fits of the k-pair star plan). One operator in every storage; recorded as `cderi_storage_executed`. Not combinable with `correlation=`. See [AICCM](aiccm.md#storage-of-the-fitted-integrals). |
| `smearing_temperature` | `0.0` | Fermi-Dirac smearing temperature (Hartree); `0.0` is no smearing. |
| `use_periodic_becke` | `True` (KS) | Use the periodic Becke partition for the XC grid. |
| `becke_image_radius_bohr` | `10.0` (KS) | Minimum point-centered image radius for the periodic Becke partition; converge with the grid resolution. |
| `lattice_opts` | *(`LatticeSumOptions`)* | Real-space cutoffs for the lattice sums (`cutoff_bohr`, `nuclear_cutoff_bohr`, ...). |
| `bipole_cutoff_bohr` | `15.0` *(argument of `run_periodic_job`, BIPOLE route)* | Direct-lattice cutoff of the electronic BIPOLE Fock sums; raise until the reported Bloch-overlap fold drift is < 1e-4 (a drift ≥ 1e-2 refuses SCF). |
| `use_exchange_ewald_split` | `None` *(auto)* | Corrected Ewald exchange split (`K = K_SR + K_LR + G=0` term). Auto-ON for 3D BIPOLE runs at Γ and Monkhorst-Pack multi-k; `False` restores the legacy Γ-locality gauge. |
| `exchange_exxdiv` | `"ewald"` | Exact-exchange G=0 convention: `"ewald"` (probe-charge Madelung, PySCF-equivalent, already near the k-converged limit) or `"none"`. The two differ by exactly `½·N_e·ξ_M(BvK supercell)`; see [from_crystal](from_crystal.md) § Comparing total energies. |
| `ewald_precision` | `1e-8` | Truncation target of the real/reciprocal Ewald lattice sums (BIPOLE route). |
| `rcut_strategy` | `"pyscf_auto"` *(argument of the GDF drivers)* | Sizes the one-electron domain (pair complete, measured to 1e-16 of the largest overlap) and the KS exchange-correlation image domain (`2 r_ao + span`) from the basis; `"flat"` keeps `cutoff_bohr` everywhere. The fitted J/K sums use `cutoff_bohr` either way. See the [lattice-cutoff tutorial](../tutorial/periodic_lattice_cutoff_convergence.md). |
| `fit_source` | `"auto"` *(argument of `run_periodic_job` and the GDF drivers)* | GDF fit: the default range-separated source, or `"classic"`. |
| `classic_support_policy` | `None` (physical pairs) *(GDF drivers)* | AO-pair support of the classic fit; `"historical_index_ball"` replays the pre-#466 operator. A positive `fit_screen_threshold` needs the historical ball. |
| `check_energy_sanity` | `True` *(GDF drivers)* | Raise `RuntimeError` on a converged total energy outside the physical range; `False` returns the raw value for diagnostics. |
| `symmetry_reduce_k` | `False` *(argument of `run_periodic_job`)* | Build exact exchange on the irreducible wedge and transport it over each star (preview; needs `symmetry="attach"`, a multi-k mesh and exact exchange). See the [tutorial](../tutorial/symmetry_reduced_k_mesh.md). |
| `conv_tol_grad` | `None` *(argument of `run_periodic_job`)* | Explicit SCF gradient tolerance, also on the BIPOLE routes; the executed value is printed in the `.out`. |

See [periodic_methods](periodic_methods.md), [smearing](smearing.md),
[multi_k_scf](multi_k_scf.md).

## Correlation and integral-engine selection

| Keyword | Default | Meaning |
|---|---|---|
| `dlpno_thresholds` | `None` (NormalPNO) *(argument of `run_job` and `run_periodic_job`)* | `"loose"`, `"normal"`, `"tight"` or a `DLPNOThresholds` record. See [DLPNO](dlpno_mp2.md). |
| `dlpno_ccsd_options` | `None` | `LocalCCSDOptions` (RHF) or `LocalUCCSDOptions` (UHF); on crystals a `LocalUCCSDOptions` also selects a UHF reference. |
| `frozen_core` | `None` (published count) | ORCA 6.1 count-only convention; `False` correlates all electrons; an integer is an explicit count (per cell on crystals). |
| `integral_engine` | `None` *(argument of `run_job` and `run_periodic_job`)* | `IntegralEngineOptions(backend="libint" \| "manta", budget=...)`; experimental. See [integral engines](integral_engines.md). |
| `coulomb` | `None` | `CoulombBuildOptions(algorithm="exact" \| "ri" \| "bupo" \| "ri_bupo", auxiliary_basis=...)`. |
| `exchange` | `None` | `ExchangeBuildOptions(algorithm="exact" \| "ri")`. |
| `periodic_integrals` | `None` *(argument of `run_periodic_job`)* | `PeriodicIntegralOptions(reciprocal_energy_cutoff, pair_cutoff_bohr, nuclear_cutoff_bohr, alpha=0.7)`; required with a periodic MANTA job. |
| `periodic_xc` | `None` | `PeriodicXCOptions(xc_image_cutoff_bohr, radial_points=50, lebedev_order=29, ...)`; required for periodic MANTA RKS/UKS. |
| `periodic_fitting` | `None` | `PeriodicFittingOptions(omega, auxiliary_cutoff_bohr, ...)`; required when a periodic MANTA J or K is RI. |
| `symmetry_reduction` | `None` *(argument of `run_job`, `run_rhf`, `run_uhf`)* | Experimental molecular point-group reduction of the RHF/UHF J/K build: `MolecularSymmetryOptions`, `MolecularRISymmetryOptions` or a process-local `PreparedMolecularSymmetry`. Refuses geometry derivatives, response, correlation, solvation, dispersion and other engines. See the [developer reference](../developer_shared_symmetry.md) and [symmetry status](../symmetry_status.md). |

## Enumerations

The values each option enum accepts (the default is marked):

- **`InitialGuess`**: `AUTO` *(molecular and periodic default)*, `HCORE`, `SAD`,
  `SAP`, `PATOM`, `HUECKEL` (`HUCKEL` alias), `MINAO`, `READ`, `FRAGMO`.
  Availability depends on the [execution route](initial_guess.md#route-coverage).
- **`SCFAccelerator`**: `DIIS`, `EDIIS`, `EDIIS_DIIS` *(molecular and periodic default)*,
  `ADIIS`, `ADIIS_DIIS`, `KDIIS`, `R_CDIIS`, `AD_CDIIS` *(adaptive-depth
  CDIIS, Chupin et al. 2021)*.
- **`SCFMode`**: `AUTO` *(default)*, `CONVENTIONAL`, `DIRECT`.
- **`PeriodicJKMethod`**: `AUTO` *(default)*, `GDF`, `BIPOLE`, `GPW`, `GAPW`,
  `SLAB_EWALD_2D`, `RIJCOSX`, `DIRECT`, `FFT_POISSON` *(retired)*, `RSGDF` and
  `CFMM` *(reserved)*, and the four AICCM members `AICCM2026DEV_A`
  (`"aiccm2026dev-a"`), `AICCM2026DEV_A_REAL_GAMMA` (value `"real-gamma"`),
  `NEUTRAL_BLOCH` (`"neutral-bloch"`) and `AICCM2026DEV_B`
  (`"aiccm2026dev-b"`), which are reached through
  `run_periodic_job(method="aiccm", variant=...)`, not through `jk_method`.
- **`CoulombMethod`**: `EWALD_3D`, `DIRECT_TRUNCATED`, `NEUTRALIZED_1D`,
  `SLAB_EWALD_2D`.
- **`SpinlockMode`**: `OFF` *(default)*, and the broken-symmetry hold modes.

## See also

- [scf_convergence](scf_convergence.md), [scf_modes](scf_modes.md),
  [initial_guess](initial_guess.md), [smearing](smearing.md): the physics
  behind the convergence and Fock-build keywords.
- [output_files](output_files.md): the `run_job` / `run_periodic_job` entry
  points that consume these options.
