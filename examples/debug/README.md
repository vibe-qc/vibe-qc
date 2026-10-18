# `examples/debug/`: diagnostic scripts for vibe-qc

Pinned reproducers and one-question probes written while the periodic
SCF, the native Gaussian density fitting (GDF) driver and the BIPOLE
route were being built. They are kept because they document how a bug
was localised and give the next debugging session a starting point.
They are not part of the test suite, and most of them predate the
current high-level APIs (`run_periodic_job`, `jk_method="gdf"`), so
expect old driver names and pre-fix numbers in their docstrings.

Many scripts compare vibe-qc against PySCF. PySCF is an optional
external reference here: it is imported only inside `examples/`, never
from `python/vibeqc/`, and each script that needs it says so in its
first lines. Without PySCF those scripts either skip the reference
column or stop at the import.

Run a script with the virtual environment's Python from the repository
root, for example `.venv/bin/python examples/debug/scf_easy_periodic.py`.
Where a docstring shows `VIBEQC_AOPAIR_FT_BACKEND=python`, set that
variable as shown. The companion note
[`PERIODIC_SCF_BUG_ANALYSIS.md`](PERIODIC_SCF_BUG_ANALYSIS.md) is the
root-cause write-up of the v0.6.x translation-invariance bug that the
first group of scripts was written to chase.

## Periodic SCF health checks

The original decision-tree set. These wrap the SCF in
`vibeqc.crash_dump_context` + `vibeqc.perf_log`, so a failure leaves a
`.dump` and a `.perf` file, and the sweeps write a `summary.csv`.
Output lands under `examples/debug/output/<scriptname>/`.

| Script | Question it answers | Wall |
|---|---|---|
| [`scf_easy_periodic.py`](scf_easy_periodic.py) | Is the periodic SCF stack itself healthy? Three closed-shell wide-gap insulators (Ne, Ar, diamond C) that must converge in under 10 iterations from an Hcore guess. | ~30 s |
| [`scf_minimal_reproducer.py`](scf_minimal_reproducer.py) | Smallest possible 3-D periodic RKS: one He atom, STO-3G (one basis function), 3 A cubic cell, Gamma-only RKS-LDA. | ~30 s |
| [`scf_size_bisection.py`](scf_size_bisection.py) | At what system size does a bug start? He, Mg, NaCl primitive, NaCl conventional, MgO conventional. `VIBEQC_FAST_DEBUG=1` coarsens every knob. | 5 to 15 min (1 to 3 min fast) |
| [`scf_convergence_sweep.py`](scf_convergence_sweep.py) | Which convergence aid fixes a divergence? Initial guess x damping x level shift x DIIS x smearing on NaCl/STO-3G/RKS-LDA, 16 configurations. | 2 to 10 min |
| [`scf_sad_vs_hcore.py`](scf_sad_vs_hcore.py) | Does the SAD guess rescue cases HCORE cannot? A/B on LiH, NaCl and MgO. | 5 to 15 min |
| [`scf_iteration_recorder.py`](scf_iteration_recorder.py) | What does a failed SCF trajectory look like? Records (E, dE, the commutator norm, DIIS dimension) per iteration and plots a four-panel figure (matplotlib). | depends on the system you point it at |
| [`scf_molecular_limit_check.py`](scf_molecular_limit_check.py) | Does a vacuum-padded periodic H2 reproduce the molecular RHF energy? Scans L = 30, 50, 100 bohr and checks the 1/L scaling of any residual. | ~1 min |
| [`scf_vs_pyscf.py`](scf_vs_pyscf.py) | Is a divergence a vibe-qc bug or a hard system? Head to head against `pyscf.pbc.dft.RKS` on Ne, Ar, diamond C (PySCF column skipped when absent). | 1 to 2 min |
| [`scf_perf_scaling.py`](scf_perf_scaling.py) | Wall time against precision for each EWALD_3D knob (`spacing_bohr`, `cutoff_bohr`, `omega`) on NaCl/STO-3G/RKS-LDA. | 3 to 10 min |

Decision tree: start with `scf_easy_periodic.py` (if any of the three
fails, the framework has regressed and nothing else matters yet); then
`scf_iteration_recorder.py` on your own system to classify the failure
(oscillating energy points at DIIS, a monotone climb at the initial
guess, a flat commutator norm at a stuck saddle); then
`scf_convergence_sweep.py` or `scf_sad_vs_hcore.py` for a fix,
`scf_size_bisection.py` for the smallest reproducing case and
`scf_vs_pyscf.py` to decide whether the system is hard for every code.
The convergence keywords themselves are documented in
[`docs/tutorial/periodic_scf_convergence.md`](../../docs/tutorial/periodic_scf_convergence.md).

## Gauge and translation-invariance probes (Ewald-3D era)

Written for the v0.6/v0.7 bug in
[`PERIODIC_SCF_BUG_ANALYSIS.md`](PERIODIC_SCF_BUG_ANALYSIS.md). They
call the Ewald-3D drivers directly.

| Script | One line |
|---|---|
| [`scf_translation_invariance_check.py`](scf_translation_invariance_check.py) | Shift the molecule inside the cell by a constant vector; the periodic SCF energy must not move. The cleanest test of the J / V_ne gauge pairing. |
| [`gradient_decomposition_check.py`](gradient_decomposition_check.py) | Gradient-side companion: which piece of the analytic Gamma-only RHF gradient disagrees with finite differences. |
| [`check_j_translation.py`](check_j_translation.py) | J and K translation invariance for the same Hcore-derived density. |
| [`check_v_ne_translation.py`](check_v_ne_translation.py) | Compares every one-electron matrix (T, V_ne, J_LR, J_SR, K) between H2 at the origin and H2 at L/2 to find which one breaks invariance. |
| [`probe_v_lr.py`](probe_v_lr.py) | Samples the long-range potential V_LR(r) directly for the origin and centred H2. |
| [`prototype_periodic_ao.py`](prototype_periodic_ao.py) | Does summing AO images over a 3x3x3 cube cure the grid integral of the density? |
| [`probe_spheropole.py`](probe_spheropole.py) | Spheropole (second-moment) contributions by shell type. |
| [`probe_spheropole_v2.py`](probe_spheropole_v2.py) | The same per primitive pair, with the weights. |
| [`lih_iter_trace.py`](lih_iter_trace.py) | Full per-iteration trace of the v0.7.0 LiH RKS-LDA SCF to see where it converges to the wrong stationary point. |
| [`fock_triangulator.py`](fock_triangulator.py) | Which Fock component (V_ne, J, V_xc) drives the LiH and NaCl disagreement with PySCF on the iteration-1 SAD density (needs PySCF). |
| [`mgo_al2o3_parity.py`](mgo_al2o3_parity.py) | The v0.7 correctness bar: MgO and cubic Al2O3, RHF and RKS-PBE, against PySCF.pbc. Slow on the RKS arms (needs PySCF). |
| [`scratch_parity_rhf_only.py`](scratch_parity_rhf_only.py) | The RHF arm of the parity test on the GDF driver, as a fast pass/fail (needs PySCF). |

## Native GDF driver development

| Script | One line |
|---|---|
| [`scratch_test_gdf_driver.py`](scratch_test_gdf_driver.py) | First test of the periodic GDF RHF driver on MgO: converge in at most 60 iterations and land within 1 mHa of the PySCF reference. |
| [`scratch_test_al2o3_gdf.py`](scratch_test_al2o3_gdf.py) | The same standalone parity check on the cubic Al2O3 model. |
| [`scratch_plain_roothaan.py`](scratch_plain_roothaan.py) | Plain Roothaan iterations (no DIIS, no damping) on MgO and Al2O3, printing every step. |
| [`scratch_alpha_al2o3.py`](scratch_alpha_al2o3.py) | Plain Roothaan on real hexagonal corundum (30 atoms) through the GDF driver, which handles non-orthorhombic cells. |
| [`scratch_alpha_al2o3_diis.py`](scratch_alpha_al2o3_diis.py) | The corundum run with DIIS, to compare iteration counts. |
| [`scratch_aux_eri_lattice_smoke.py`](scratch_aux_eri_lattice_smoke.py) | Unit smoke for the periodic two- and three-centre ERI kernels without PySCF: the zero-cell limit must reproduce the molecular integrals. |
| [`scratch_lpq_native_smoke.py`](scratch_lpq_native_smoke.py) | Does the bare image-summed two-centre metric converge fast enough for sub-microhartree parity without a compensating charge? |
| [`scratch_per_shell_rcut.py`](scratch_per_shell_rcut.py) | Tests the hypothesis that per-shell-pair cutoff screening keeps the bare two-centre metric finite and positive definite on diffuse auxiliary sets. |
| [`scratch_2c_pyscf_vs_screened.py`](scratch_2c_pyscf_vs_screened.py) | Zeroes shell-pair blocks of a large-cutoff two-centre metric to see whether per-shell screening reproduces PySCF's matrix (needs PySCF). |
| [`gdf_bare_aux_divergence.py`](gdf_bare_aux_divergence.py) | Demonstrates that the direct image sum of the two-centre metric diverges on a diffuse JKfit auxiliary basis. |
| [`gdf_mgo_conv_pyscf_parity.py`](gdf_mgo_conv_pyscf_parity.py) | The MgO eight-atom conventional-cell GDF energy against PySCF (needs PySCF). |
| [`gdf_multik_pyscf_parity.py`](gdf_multik_pyscf_parity.py) | Multi-k compensated-cell GDF SCF on H2 at several k-meshes against `KRHF.density_fit()` (needs PySCF). |
| [`gdf_lih_primitive_multik_parity.py`](gdf_lih_primitive_multik_parity.py) | The same multi-k parity on a real ionic crystal with band dispersion, the LiH primitive cell (needs PySCF). |
| [`gdf_profile_hot_path.py`](gdf_profile_hot_path.py) | Per-stage wall-time profile of the compensated-cell GDF SCF (Lpq build, per-iteration J/K). |
| [`gdf_multik_profile_hot_path.py`](gdf_multik_profile_hot_path.py) | The production multi-k source and SCF timed across thread counts on a fully allocated node; meant to run through `vq`. |

## AO ordering and normalisation conventions (libint against libcint)

vibe-qc integrates with libint, PySCF with libcint. Before any matrix
can be compared element by element the two AO orderings and the
spherical-harmonic normalisations have to be reconciled; these scripts
established the permutation and the per-AO factors. All of them need
PySCF.

| Script | One line |
|---|---|
| [`scratch_basis_map.py`](scratch_basis_map.py) | Establishes the AO-ordering permutation between vibe-qc and PySCF on MgO/STO-3G at Gamma. |
| [`scratch_permutation.py`](scratch_permutation.py) | Builds that permutation matrix from the per-atom shell orderings. |
| [`scratch_periodic_perm.py`](scratch_periodic_perm.py) | Confirms the permutation on periodic MgO S, T and V_ne with a sufficient lattice cutoff. |
| [`scratch_basis_exponents.py`](scratch_basis_exponents.py) | Prints the STO-3G exponents and contractions of Mg and O from both codes and re-checks the overlap eigenvalues at larger cutoffs. |
| [`scratch_mgo_molecule.py`](scratch_mgo_molecule.py) | Compares the overlap matrix of an isolated Mg + O pair between the two integral libraries, without any lattice. |
| [`scratch_pyscf_gdf_inspect.py`](scratch_pyscf_gdf_inspect.py) | Probes how PySCF builds its GDF for MgO/STO-3G: auxiliary choice, compensated shells, and the shapes it serialises. |
| [`gdf_aux_convention_libint_vs_libcint.py`](gdf_aux_convention_libint_vs_libcint.py) | Quantifies the auxiliary-basis convention mismatch on H2/def2-SVP-JK in the molecular limit. |
| [`gdf_libint_contraction_norm_probe.py`](gdf_libint_contraction_norm_probe.py) | Characterises libint's per-shell contraction normalisation empirically from single-shell self-overlaps. |
| [`gdf_per_ao_norm_calibration.py`](gdf_per_ao_norm_calibration.py) | Calibrates the per-AO libint-to-libcint normalisation factors. |
| [`gdf_pair_ft_pyscf_diff.py`](gdf_pair_ft_pyscf_diff.py) | Element-wise comparison of the AO-pair Fourier transform, the last convention question after single-AO transforms matched. |

## AFT, compensated cell and lattice three-centre sums

| Script | One line |
|---|---|
| [`gdf_aft_calibration.py`](gdf_aft_calibration.py) | Per-L calibration of the analytic Fourier-transform convention against libint's two-centre integrals. |
| [`gdf_aft_pyscf_diff.py`](gdf_aft_pyscf_diff.py) | Compares every intermediate of the compensated-cell AFT pipeline with PySCF on H2 (needs PySCF). |
| [`gdf_aft_pyscf_diff_lih.py`](gdf_aft_pyscf_diff_lih.py) | The LiH variant, which brings an L = 1 orbital shell into the comparison (needs PySCF). |
| [`gdf_j3c_p_same_mesh_lih.py`](gdf_j3c_p_same_mesh_lih.py) | Element-wise three-centre comparison on the same G-mesh, after per-block norms were found to hide permutation errors (needs PySCF). |
| [`gdf_m1_step1_tlat_pyscf.py`](gdf_m1_step1_tlat_pyscf.py) | Verifies the lattice-summed three-centre ERI tensor against PySCF's `pbc_intor('int3c2e')` on a fused orbital + auxiliary cell (needs PySCF). |
| [`gdf_m1_step2_j3c_bloch_fix.py`](gdf_m1_step2_j3c_bloch_fix.py) | Lattice three-centre = molecular three-centre + images, and Bloch pair-FT parity of the three-centre tensor (needs PySCF). |

## Mixed density fitting (MDF)

| Script | One line |
|---|---|
| [`mdf_g0_multipole_probe.py`](mdf_g0_multipole_probe.py) | Measures the exact per-auxiliary G = 0 self-term gap to show no higher multipoles are missing. |
| [`mdf_ne_convergence.py`](mdf_ne_convergence.py) | Converges both vibe-qc MDF and PySCF MDF on a Ne box in their mesh sizes to settle a 0.1 mHa residual (needs PySCF). |
| [`mdf_ne_eta_probe.py`](mdf_ne_eta_probe.py) | Characterises the same residual as a function of the compensating-charge exponent (needs PySCF). |

## Molecular density fitting and gradients

| Script | One line |
|---|---|
| [`df_smoke.py`](df_smoke.py) | Reproducer for the v0.7.3 density-fitted SCF segmentation fault (H2O / def2-SVP / def2-SVP-JK). |
| [`gradient_component_diagnostic.py`](gradient_component_diagnostic.py) | Isolates analytic gradient contributions atom by atom, component by component and shell by shell for the direct and the DF paths. |

## BIPOLE route (`bipole_debug/`)

| Script | One line |
|---|---|
| [`bipole_debug/mgo_fast_diag.py`](bipole_debug/mgo_fast_diag.py) | MgO through BIPOLE: RHF first (no DFT grid), then the RKS trajectory. |
| [`bipole_debug/mgo_rks_convergence.py`](bipole_debug/mgo_rks_convergence.py) | Reproduces the MgO RKS BIPOLE oscillation and separates a gauge bug from a missing convergence aid. |
| [`bipole_debug/multik_decompose.py`](bipole_debug/multik_decompose.py) | Decomposes the multi-k RHF gradient error by energy term (analytic against FD per component). |
| [`bipole_debug/multik_gradient_test.py`](bipole_debug/multik_gradient_test.py) | Fast multi-k RHF gradient test on H2 in a box with a 2x1x1 mesh. |
| [`bipole_debug/rks_asymmetric_gradient.py`](bipole_debug/rks_asymmetric_gradient.py) | Quantifies the RKS Gamma gradient error on an asymmetric cell, where orbital relaxation matters. |
| [`bipole_debug/rks_gamma_gradient_test.py`](bipole_debug/rks_gamma_gradient_test.py) | RKS Gamma analytic gradient against FD on a symmetric H2 box, LDA first, then PBE. |
| [`bipole_debug/uks_gamma_gradient_test.py`](bipole_debug/uks_gamma_gradient_test.py) | The UKS counterpart on a triplet H2 box. |

## Historical, no longer runnable

These six import helpers that were removed together with the
PySCF-backed GDF spike they probed. Each carries a comment at the top
saying which name is gone. They are kept as a record of the diagnostic
and are not expected to run.

| Script | Missing helper | What it probed |
|---|---|---|
| [`scratch_direct_vs_pyscf.py`](scratch_direct_vs_pyscf.py) | `periodic_rhf_gdf._pyscf_cell_from_vibeqc`, `_build_permutation_matrix` | The real-space DIRECT two-electron Fock build against PySCF on MgO RHF. |
| [`scratch_lpq_native_energy.py`](scratch_lpq_native_energy.py) | the same two plus `_lpq_tensor_from_pyscf` | End-to-end energy with the native Lpq tensor swapped into the old driver. |
| [`scratch_modrho_smoke.py`](scratch_modrho_smoke.py) | `periodic_rhf_gdf._pyscf_cell_from_vibeqc`, `_build_permutation_matrix` | Compensating-charge rescaling of the periodic two-centre metric and Lpq against PySCF. |
| [`scratch_2c_vs_pyscf.py`](scratch_2c_vs_pyscf.py) | `aux_basis.modrho_renormalise` | Element-wise lattice two-centre ERIs against PySCF's `int2c2e`. |
| [`gdf_m1_lattice_3c_diagnostic.py`](gdf_m1_lattice_3c_diagnostic.py) | `_aopair_ft.per_ao_libint_norm_factor` | Lattice three-centre tensor diagnostic and an MgO remeasurement. |
| [`lih_energy_decomposition.py`](lih_energy_decomposition.py) | the module `vibeqc.xc_lattice` | Iteration-1 energy decomposition of the LiH rocksalt SCF. |

## Adding a debug script

Recommendations, not rules: one question per script, a docstring that
states the expected answer, output under `output/<scriptname>/` or next
to the script, and a wrapper in `crash_dump_context` + `perf_log` when
the SCF might fail (the first group shows the pattern). Shorter is
better; a few hundred lines is a good ceiling for a script someone else
has to read during an incident. Add a row here.
