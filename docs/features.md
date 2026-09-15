# Feature matrix

The capability matrix for the **v0.15.x** release line (codename *Neese's Cheetah*).
For when each feature shipped, see
[`CHANGELOG.md`](https://github.com/vibe-qc/vibe-qc/blob/main/CHANGELOG.md).
For what's coming next, see [`roadmap.md`](roadmap.md).

## Molecular SCF

| Method | Driver | Open-shell | Gradient | Hessian (FD) | Hessian (analytic) | MP2 | Validated vs PySCF |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| RHF | `run_rhf` | - | ✓ | ✓ (`compute_hessian_fd`) | ✓ (`compute_hessian_rhf_analytic`) | ✓ (`run_mp2`) | ✓ machine precision |
| UHF | `run_uhf` | ✓ | ✓ | ✓ | (roadmap 17c) | ✓ (`run_ump2`) | ✓ machine precision |
| RKS | `run_rks` | - | ✓ | ✓ | (roadmap 17e) | - | ✓ grid-accuracy |
| UKS | `run_uks` | ✓ | ✓ | ✓ | (roadmap 17f) | - | ✓ grid-accuracy |
| ROHF | `run_rohf` | ✓ (spin-pure) | ✓ analytic | ✓ (freqs) | (roadmap) | ✓ (`run_rohf_mp2`, semicanonical) | PySCF-parity tests¹ |
| ROKS | `run_roks` | ✓ (spin-pure) | ✓ (FD, opt) | (roadmap M5b) | (roadmap M5b) | - | PySCF-parity tests¹ |

¹ ROHF/ROKS: the Roothaan-coupling + energy math is verified directly
(closed-shell→RHF/RKS reduction, SCF stationarity, and ROKS fractional
degenerate frontier shells such as OH(2Pi)); the end-to-end PySCF-parity
and ROHF-on-singlet==RHF tests are written and run in the full dev-box
suite (`tests/test_rohf.py`, `tests/test_roks.py`).

Mean-field functionals libxc supports work through RKS / UKS / ROKS (500+
functionals including LDA, GGAs, hybrid GGAs, the τ-dependent
meta-GGA family (TPSS, TPSSh, M06-L, M06-2X, SCAN, r²SCAN, r²SCAN01),
and the range-separated hybrids ωB97X, ωB97X-D, and the VV10-paired
ωB97X-V / ωB97M-V; direct double-hybrid ROKS calls still route through
the double-hybrid dispatcher). HF and DFT
analytic gradients are available for LDA, GGA, hybrid-GGA, and
the τ-dependent meta-GGA family (modulo the
[f-shell gradient bug](#known-issues) on open-shell UHF / UKS).
Meta-GGA analytic *Hessians* and range-separated-hybrid analytic
*gradients* are still queued. Range-separated hybrids run via
direct SCF (not yet density-fitting).

**Functionals without a complete analytic gradient (GitLab #571).**
The molecular RKS / UKS gradient kernel differentiates exact exchange
through the global fraction only; it omits the long-range
erf-attenuated exchange term of every range-separated hybrid (HSE06,
ωB97X, ωB97X-D, CAM-B3LYP, ωB97X-V, ωB97M-V, ...) and the VV10 nonlocal
correlation gradient (VV10, ωB97X-V, ωB97M-V). Measured on a bent O/H/H
molecule in STO-3G the partial analytic gradient sits 5.6e-3 (HSE06),
7.0e-4 (VV10) and 5.6e-2 (ωB97X-V) Ha/bohr off the full-energy
finite-difference gradient, against a 2.1e-6 noise floor for PBE. Since
this fix `vibeqc.compute_gradient_rks` / `compute_gradient_uks` refuse
such a functional with a `NotImplementedError` naming the missing terms
(pass `allow_incomplete=True` to receive the partial gradient knowingly),
`vibeqc.functional_gradient_terms_missing(name)` reports them, and every
molecular geometry optimizer (`run_job(optimize=True)` on the ASE and
native backends, `optimize_molecule`, `optimize_molecule_brent`, the
`geomopt` provider, the `VibeQC` ASE calculator) falls back to full-energy
central finite differences (step 0.005 bohr) and says so in the `.out`.
Routes that finite-difference the *analytic* gradient (the FD Hessian /
frequencies, the analytic-Hessian skeletons, CPCM gradients) and NEB fail
closed with the same message. The analytic terms themselves are a
separate feature.

## Wavefunction methods (v0.9.0-v0.12.0)

Non-mean-field electronic-structure methods that do not
conceptually rely on a Hartree-Fock reference. All reachable
through `run_job(method=…)` and as a standalone
`vibeqc.solvers` import. See
[`user_guide/non_hf_solvers.md`](user_guide/non_hf_solvers.md)
and the [`non_hf_solvers` tutorial](tutorial/non_hf_solvers.md).

## Post-HF correlation methods

Correlated wavefunction methods that build on a Hartree-Fock
reference. All reachable through `run_job(method=…)`.

| Method | `run_job(method=…)` | Reference | Notes |
|--------|---------------------|-----------|-------|
| CISD | `"cisd"` or `method="ci", citype="cisd"` | HF orbitals | Fixed-space variational singles+doubles CI via `vibeqc.cisd`; `CISDOptions` controls roots and determinant-space guards. |
| CC2 | `"cc2"` or `method="ci", citype="cc2"` | RHF | Ground-state second-order approximate coupled cluster. The singles projection is complete and the doubles equation retains the T1-transformed Hamiltonian plus `[F,T2]`; DF and exact-integral routes are supported. Closed-shell energies only; no perturbative triples. Validated against a spin-orbital oracle and ORCA AUTOCI-CC2. |
| CCSD | `"ccsd"` | RHF / UHF / ROHF | Canonical dense CCSD, density-fitted by default or on exact four-index integrals (`CCSDOptions(density_fit=False)`, conventional cross-code parity route); closed-shell and open-shell (UCCSD on a UHF reference, ROHF-reference via the spin-orbital kernel), auto-dispatched from the SCF reference. Also `run_ccsd` with `CCSDOptions`. |
| CC3 | `"cc3"` or `method="ci", citype="cc3"` | RHF | Ground-state iterative CC3 with converged singles/doubles and approximate connected triples regenerated at every iteration. Exact MO integrals and frozen cores are supported. Energy-only and benchmark-scale because the dense determinant representation grows combinatorially. Validated against out-of-process ORCA 6.1 UHF AUTOCI-CC3 on closed-shell references. |
| CCSDT | `"ccsdt"` or `method="ci", citype="ccsdt"` | RHF / ROHF | Full iterative singles, doubles, and triples from the exact determinant-space similarity projection. Exact MO integrals, frozen cores, and common-orbital open shells are supported. Energy-only and benchmark-scale because the dense determinant representation grows combinatorially. |
| CCSD(T) | `"ccsd(t)"` | RHF / UHF / ROHF | Dense small-molecule accuracy reference. Closed-shell DF triples support requested-memory fast/blocked/direct/disk execution; open-shell triples support fast/direct but retain the dense spin-orbital integral floor. The dense CC iteration and the FNO setup remain production-size memory gates. Validated vs PySCF (uHa; conventional route to a few nHa) with ORCA 6.1 cross-checks. |
| A-CCSD(T) | `method="ccsd", triples="A-CCSD(T)"` | RHF | Closed-shell asymmetric/Lambda triples correction on CCSD amplitudes. This separate implementation remains dense and rejects bounded triples modes. The effective method and citation route are `a-ccsd(t)`, with Kucharski-Bartlett 1998 and Crawford-Stanton 1998 references surfaced in output. |
| QCISD / QCISD(T) | `"qcisd"`, `"qcisd(t)"` (also via `citype=`) | RHF | Closed-shell quadratic CI singles+doubles via exact monomial selection from the canonical CCSD residual. QCISD(T) uses the original Pople-Head-Gordon-Raghavachari `E[T] + 2*E_ST` triples convention on QCISD amplitudes. Pinned vs the in-repo spin-orbital oracle and out-of-process ORCA 6.1 RI-QCISD(T). |
| CCD / LCCD / LCCSD / CEPA(0..3) | `"ccd"`, `"lccd"`, `"lccsd"`, `"cepa(0)"`..`"cepa(3)"` (also via `citype=`) | RHF | Coupled-pair variants of the closed-shell DF-CCSD kernel (`CCSDOptions` field `cc_variant`): CCD freezes singles; LCCD/LCCSD linearize the amplitude equations (exact polynomial-stencil linearization of the canonical residual); CEPA(1..3) add Meyer's pair-specific EPV shifts. CEPA validated to sub-uHa vs out-of-process ORCA 6.1 RI-CEPA/n (canonical orbitals). No (T); closed-shell only. |
| DLPNO-CCSD | `"dlpno-ccsd"` | RHF / UHF | Domain-based local pair natural orbital CCSD. Closed-shell: reduced-scaling local solver (pair classification, PAO domains, per-pair PNO solver). Open-shell (multiplicity > 1) auto-routes to the local spin-orbital DLPNO-UCCSD solver; it applies the shared Liakos threshold triple but does not claim Saitow's single-spatial-orbital SOMO-pair scaling. The O(N⁶), `max_nbf`-capped pilot remains opt-in. |
| DLPNO-CCSD(T) | `"dlpno-ccsd(t)"` | RHF / UHF | Closed-shell: local solver plus local DLPNO-(T) (per-triple TNO domains), full-domain == canonical CCSD(T). Retained accuracy benchmarks use explicitly pinned pre-#140/#448 all-electron recipes; they are not evidence for the current published-core/NormalPNO default. Open-shell auto-routes to the reduced-scaling UHF-reference spin-orbital DLPNO-UCCSD(T) solver; the dense O(N⁶), `max_nbf`-capped pilot is selected only by explicitly passing `DLPNOUCCSDPilotOptions`. |
| DLPNO-MP2 | `"dlpno-mp2"` | RHF / UHF | Domain-based local pair natural orbital MP2 (Pinski 2015); the open-shell DLPNO-UMP2 extension retains independent UHF alpha/beta orbital and PNO spaces. Boys-localised occupieds, per-pair PNO virtual space; the zero-threshold all-electron limit reproduces canonical RI-MP2 to <= 1 uHa. Historical recovery data use explicitly pinned pre-#140/#448 cutoffs and are not claims about the current named default. |
| IC-CASPT2 | `"caspt2"` | CASSCF/CASCI | Internally-contracted CASPT2, un-gated at v0.12.0. Large-basis direct active-CI path. |
| CASCI + CASSCF | `"casci"` / `"casscf"` | - | Multi-root CASCI (`casci_options=CASCIOptions(nroots=N)`) with state-averaged CASSCF (`casscf_options=CASSCFOptions(nroots=N)`); orbital-optimised. The CASSCF **analytic nuclear gradient** is FD-tight (~1e-7, default `compute_wz=False` path), validated in v0.15.0. State-averaged + open-shell CASSCF gradients fall back to FD. CASPT2/NEVPT2 gradients experimental (full-energy FD). |
| TDDFT (Casida + TDA) | `run_tddft_tda` / `run_tddft_casida` / `run_job(tddft=True)` | RHF/RKS/UHF/UKS | Vertical excitation energies, oscillator strengths, transition dipoles, dominant orbital transitions. Runner integration: `run_job(tddft=True, nto=True, tddft_type="tda"|"casida")` computes excitation energies (→ `.out` table), Natural Transition Orbitals (→ QVF + molden), FD excited-state gradients (`tddft_gradient=True`; the closed-shell RHF + CIS (TDA) surface only, labelled as such in the `.out`; RKS/TDDFT, Casida and open-shell runs are refused before the calculation, #570), and UV/Vis spectrum (`tddft_spectrum=True`). UHF NTOs via `compute_nto_uhf()`. C++ `eri_ao_to_mo_blocks` kernel replaces Python einsum. `vibeqc tddft` and `vibeqc tddft-opt` CLI. Periodic `run_periodic_job(tddft=True)` at Γ-point. Adiabatic LDA/GGA XC kernel; global hybrids carry their exact-exchange admixture. Range-separated hybrids refused. |
| CIS / TDA | `run_tddft_tda` (TDA limit) | HF / MSINDO | Configuration-interaction singles; Tamm-Dancoff approximation. Finite-difference excited-state gradients (state-tracked). |
| Green's function (GF2 / OVGF) | `"ovgf"` | HF / MSINDO | Quasiparticle IPs and EAs (result `.ovgf`); renormalised GF2 with full third-order self-energy + geometric screening auto-printed for small systems. Open-shell (UHF) propagator supported. MSINDO reference: `vibeqc.semiempirical.methods.msindo.msindo_ovgf` (`renormalize=True` for renormalised GF2). |

| Method | `run_job(method=…)` | Solver entry point | Notes |
|--------|---------------------|--------------------|-------|
| Full CI | `"fci"` | (dense `eigh` on the full determinant Hamiltonian) | Exact within the (active) orbital space; reference for the approximate solvers. Small active spaces only. |
| Selected CI (CIPSI) | `"selected_ci"` | `solve_selected_ci` | Iterative perturbative selection; `target_size`, `pt2_threshold`, optional PT2 correction. Spin-unrestricted determinant basis by default (#107, 2026-09-03); `spin_restricted=True` selects the closed-shell seniority-zero (DOCI) basis, coupled through the pair-move integral ⟨ii\|aa⟩ (#639), which cannot represent open-shell configurations. |
| DMRG | `"dmrg"` | `solve_dmrg` | Two-site sweep, MPS wavefunction, bond-dimension schedule. Minimal Python implementation, ≤12 orbitals. |
| Variational 2-RDM | `"v2rdm"` | `solve_v2rdm` | Exact small-active-space N-representable 1/2-RDM from the in-tree determinant Hamiltonian; Q/G SDP feedback remains unimplemented. |
| Transcorrelated CI | `"transcorrelated_ci"` | `build_transcorrelated_hamiltonian` + `solve_selected_ci` | Similarity-transformed Hamiltonian (Gaussian / Jastrow correlator) fed to selected CI; same determinant-basis default as selected CI (#107). |

* **Active-space truncation**: pass `active_space=(n_orbitals,
  n_electrons)` to `run_job` to restrict any solver to a
  frozen-core / CAS-style window. The runner projects the full
  MO-basis `Hamiltonian` onto the active block via
  `Hamiltonian.active_space`, which applies the standard CAS
  partition (lowest `(nelec − n_elec)/2` orbitals as a
  doubly-occupied inactive core) with the **proper frozen-core
  dressing**, the effective one-electron term
  `h̃_pq = h_pq + Σ_c (2⟨pc|qc⟩ − ⟨pc|cq⟩)` and the constant
  `E_core` offset folded into the nuclear-repulsion term, so the
  reported energy is the full CAS total energy. The dressing is
  shared with the `casci` path, so `fci`/`selected_ci`/`dmrg`/`v2rdm`
  on a given active space agree with `casci` on the same window to
  numerical precision (`E_FCI ≤ E_CAS ≤ E_HF`). See
  `python/vibeqc/solvers/ACTIVE_SPACE.md` for the contract and
  `tests/test_solvers_active_space_api.py` for the
  variational-sandwich + cross-method coverage.
* **Orbital source is decoupled**, the solvers operate on an
  abstract `Hamiltonian` in an orthonormal MO basis;
  `get_hf_orbital_provider` is a convenience, not a requirement.
* **Geometry optimisation**, wavefunction methods optimise via
  numerical gradients (`run_job(method="fci", optimize=True)`);
  analytic gradients are roadmap.
* **Parity**, He / OH FCI parity tests and DMRG-vs-FCI
  cross-checks ship in `tests/test_solvers_*`.

## Density fitting + RIJCOSX (v0.8.0)

`JKBuilder` polymorphic Fock build via three concrete kernels;
see [`user_guide/density_fitting.md`](user_guide/density_fitting.md).

| Kernel | Driver flags | When |
|--------|--------------|------|
| `FourIndexJKBuilder` | (default) | small (≤250 BFs) |
| `DFJKBuilder` (RIJK) | `density_fit=True, aux_basis="..."` | medium (~250-1000 BFs), hybrid DFT |
| `COSXJKBuilder` (RIJCOSX) | `density_fit=True, cosx=True, aux_basis="..."` | large (>1000 BFs), hybrid DFT |

* RIJK gradient: production-ready. The historical ~115 mHa
  glycine bug closed in v0.8.0; see
  [`user_guide/density_fitting.md`](user_guide/density_fitting.md).
* RIJCOSX SCF **energy**: validated to **0.26 mHa vs ORCA 6.1.1**
  on glycine / def2-TZVP. (An earlier revision of this bullet
  claimed the analytic gradient was "validated" on the strength of
  that energy comparison; the gradient claim was unproven and,
  pre-2026-07-29, wrong.)
* RIJCOSX analytic gradient: matches finite differences of the
  RIJCOSX energy surface to the neglected grid-motion band
  (~2e-4 Ha/bohr on H2O/def2-tzvp, ~3e-3 worst case OH/def2-svp;
  gate: `tests/test_rijcosx_gradient_fd.py`). Against ORCA 6.1.1
  EnGrad on glycine/def2-TZVP: |grad| 0.08696 vs 0.08703 Ha/bohr,
  componentwise RMS 3.3e-3, max 1.1e-2; the residual is the
  documented fitted-Fock response term plus neglected
  grid-motion/weight derivatives (see
  `compute_cosx_k_gradient_contribution` notes in
  `cpp/include/vibeqc/cosx.hpp`).
* `default_aux_basis_for(orbital_basis_name, kind="jk"/"ri")`
  helper picks aux basis automatically.

## Periodic SCF (native GDF / GPW / BIPOLE / Γ-CCM / χ-CCM)

Γ-CCM and χ-CCM[^xccm-convention] are distinct union-and-weight and
finite-translation-group character approaches. The code selectors remain
`aiccm2026dev-a` and `aiccm2026dev-b`.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    union-and-weight/Wigner-Seitz integral weighting and finite-translation-group
    characters, respectively. A declared common exchange-q=0 convention is a
    comparison constraint, not an identification of the constructions. Their
    distinction is not a choice of Coulomb kernels, and equality for a specified
    operator and route must be established rather than inferred from the labels.

| Method | Driver | k-point support | Notes |
|--------|--------|-----------------|-------|
| RHF, Γ-only GDF | `run_rhf_periodic_gamma_gdf` | Γ, dim=1/3 (slab raises) | native vibe-qc J/K via Gaussian density fitting; no in-process PySCF |
| RKS / hybrids, Γ-only GDF | `run_rhf_periodic_gamma_gdf(functional=…)` | Γ, dim=1/3 (slab raises) | native libxc `V_xc`; hybrid exact exchange uses the functional's HF fraction |
| KRHF multi-k GDF | `run_krhf_periodic_gdf` | Γ + multi-k, dim=1/2/3 | native k-dependent GDF. On `dim=2`, an explicit full Gamma-centered `(n1,n2,1)` mesh selects the signed slab-truncated metric and rigorous 2D one-electron gauge; custom/IBZ meshes, smearing, restarts, and open shell stay closed. Closed-shell slab analytic gradients ship. Also reachable via `run_periodic_job(jk_method="gdf", kpoints=...)` |
| KRKS multi-k GDF | `run_krks_periodic_gdf` | Γ + multi-k, dim=1/2/3 | native multi-k RKS / full-range hybrid path. The bounded `dim=2` route has the same restrictions as KRHF and rejects range-separated functionals; closed-shell slab analytic gradients ship; PySCF / CRYSTAL are out-of-process references only |
| RIJCOSX periodic SCF | `run_periodic_job(jk_method="rijcosx")` | Gamma RHF; vacuum-padded Gamma RKS/UHF/UKS; true multi-k RHF/RKS/UHF/UKS | RI-J from the native GDF loop plus periodic COSX exchange (`k_exchange="cosx"`) on true multi-k meshes (weighted full-BZ quadratures supported); one-cell tight-cell RKS/UHF/UKS fail closed rather than relabeling a GDF-only route |
| χ-CCM SCF / post-HF | `run_aiccm2026dev_b_rhf/rks/uhf/uks`, restricted and unrestricted MP2/CCSD(T)/PNO APIs; `run_periodic_job(jk_method="aiccm2026dev-b")` | real-space cyclic lattice extension, 3D absolute energies only | finite-character (Γ-centred character-mesh) CCM with four-centre, RI, and RIJCOSX restricted/unrestricted SCF plus 3D finite-torus correlation; every 1D/2D B backend fails closed pending a shared wire/slab Coulomb convention; local CC remains an O(N^6) correctness pilot, not production reduced scaling |
| χ-CCM RI-MP2 | `run_aiccm2026dev_b_mp2` | finite cyclic mesh, 3D | momentum-conserving canonical RI-MP2 on the χ-CCM RI-RHF reference; external KMP2 parity; experimental |
| Γ-CCM / aiccm2026dev-a | `vibeqc.periodic.ccm`, `method="aiccm2026dev-a"`; `run_ccm_rhf`, `run_ccm_rks`, `run_ccm_mp2`, `run_ccm_ump2`, `run_ccm_ccsd` | real-space cyclic cluster extension (nrep) | union-and-weight/Wigner-Seitz integral-weighting CCM: HF/KS with the four-center construction, construction-specific RIJCOSX research, MP2/UMP2, CCSD(T), analytic gradients, properties, localization, and symmetry; experimental (every SCF driver emits `AICCM2026DevAExperimentalWarning`) |
| A-namespace neutral fitted-torus controls | `run_ccm_rhf_gdf`, `run_ccm_rhf_direct`, `run_ccm_ri_mp2`, `run_ccm_ri_ccsd`, `run_ccm_uccsd`, `ccm_dlpno_*` | separately declared finite-torus control operator | character/Bloch and real-Gamma evaluations plus restricted/unrestricted canonical and DLPNO correlation on the matching neutral control. These APIs are in `vibeqc.periodic.ccm` for historical reasons; they are not Γ-CCM or χ-CCM construction results; experimental (emit `AICCM2026DevAExperimentalWarning`) |
| Cell-resolved DF integral blocks | `compute_2c_eri_lattice_blocks`, `compute_3c_eri_lattice_blocks`, `bloch_sum_*_eri_blocks`, `build_lpq_bloch_native` | dim=1/2/3 | native translation-resolved 2c/3c storage and Bloch/Lpq assembly for k-dependent GDF; Γ tensors are recovered by summing blocks |
| 2D slab SCF (SLAB_EWALD_2D) | `run_periodic_job(jk_method="auto")` on a `dim=2` system; `vibeqc.slab_2d(a1, a2, atoms)` to build one | Γ + multi-k, dim=2 only | Rigorous vacuum-free 2D Ewald (Parry 1975; de Leeuw and Perram 1979) for RHF / RKS / UKS. No vacuum gap: the third lattice column is synthesized bookkeeping and the total energy is invariant to it; `E_nuclear` is k-mesh independent. AUTO selects it for every `dim=2` system. Explicit `jk_method="gdf"` is a bounded closed-shell RHF/RKS alternative with a signed slab-truncated fit and analytic gradients; BIPOLE / GPW / GAPW / RIJCOSX remain closed. Fermi-Dirac smearing and slab-GDF open shell are not shipped yet. |
| BIPOLE RHF / UHF / RKS / UKS | `run_pbc_bipole_rhf`, `..._uhf`, `..._rks`, `..._uks` | Γ + multi-k, 3D through `run_periodic_job` | Production exact Ewald-J energy and finite-difference atomic-force route for closed- and open-shell periodic SCF. Fixed-cell atomic relaxation ships; variable-cell work fails closed. Analytic gradients remain a gated preview. The quartet multipole far-field prototypes are driver-unreachable: all four drivers default to `False`, and explicit `True` raises before setup |
| GPW RHF / ROHF / ROKS / UHF / RKS / UKS (v0.12.0) | `run_periodic_job(jk_method="gpw")` | Gamma supports all six methods; multi-k pure-DFT RKS/ROKS/UKS supports LDA/GGA/meta-GGA in molecular-limit and compact Bloch regimes through the public runner or the matching `run_periodic_*_gpw_multi_k` API; multi-k UHF/ROHF and hybrids remain gated | Production Gaussian Plane Waves route (serial), unified `run_periodic_job` user surface; ROHF and ROKS are maintained preview with one restricted orbital set and integer 2/1/0 occupations; GAPW all-electron augmentation; DFT+U for Gamma RHF/RKS/UHF/UKS plus multi-k RKS; D3-BJ dispersion; Gamma-point analytic gradient; FD Hessian; smearing. The MPI z-slab grid overlay (`GpwJBuilder(mpi_aware=True)`) is experimental: its multi-rank Hartree-J build was broken through v0.12.0, fixed post-release, pending validation on a real multi-rank run |
| RHF / RKS / UHF / UKS Ewald-3D (legacy) | `run_*_periodic_*_ewald3d` | Γ + multi-k | native debug path; not the CRYSTAL-parity target |
| Multi-k UHF / UKS via GDF | `run_kuhf_periodic_gdf`, `run_kuks_periodic_gdf`; public runner | Γ + multi-k, dim=1/3 | Public GDF open-shell route. Slab open shell remains fail-closed. |
| ROHF periodic, Ewald-3D | `run_rohf_periodic_gamma_ewald3d`, `run_rohf_periodic_multi_k_ewald3d` | Γ + multi-k, dim=3 | Standalone HF drivers with the shared Roothaan coupling and integer 2/1/0 occupations; not wired through `run_periodic_job` |
| ROHF periodic, native GDF | `run_periodic_job(method="ROHF", jk_method="gdf")`, `run_krohf_periodic_gdf` | Γ (a `(1,1,1)` mesh) + full multi-k, dim=3 | Public and standalone density-fitted ROHF on the per-(k_i,k_j) cderi cache: same J / per-spin K / `exxdiv='ewald'` machinery as the PySCF-pinned KRHF/KUHF drivers, with the Roothaan coupling as the only new step. Exchange contracts against the occupied MO blocks. Multiplicity 1 reproduces `run_krhf_periodic_gdf` exactly; Li/STO-3G Γ is +2.86e-5 Ha from PySCF KROHF/GDF, collapsing to +2.2e-7 Ha with the rsgdf mesh. ROKS, smearing, 2D slabs, gradients, optimization, restarts, COSX and `ibz_native` remain fail closed |
| ROHF / ROKS periodic, GPW | `run_periodic_job(method="ROHF", jk_method="gpw")`, `run_periodic_job(method="ROKS", functional=..., jk_method="gpw")`, `run_periodic_rohf_gpw`, `run_periodic_roks_gpw`, `run_periodic_roks_gpw_multi_k` | ROHF: Gamma only; ROKS: Gamma plus full multi-k pure-DFT meshes; dim=3 | Maintained-preview restricted-open-shell route: GPW Hartree J, per-spin exact K where required, spin-polarised XC for ROKS, the Ewald one-electron/nuclear gauge, and complex-safe Roothaan coupling. Unsupported dimensions, gradients, GAPW, multi-k hybrids, range-separated ROKS, and electronic smearing fail closed; `smearing_alpha` is nuclear smoothing |
| Atomic gradients (periodic) | route-specific FD forces; `compute_gradient_periodic_*` previews | Γ + multi-k by route | BIPOLE production forces use finite differences of the executed energy; analytic coverage is route- and envelope-specific and remains preview where documented |
| Periodic force-virial diagnostic | `compute_stress_tensor` | Γ + multi-k | Diagnostic only; not the periodic stress. BIPOLE/GDF variable-cell optimization fails closed |

Reference benchmark: **ΔE = -6.8e-12 Ha** for MgO/sto-3g/Γ
RHF vs out-of-process PySCF.pbc.GDF. Multi-k KRHF / KRKS GDF is now
native, the multi-k Ewald-gauge fix brings a multi-k mesh to within
**0.000 mHa of the Γ-point energy** in the molecular limit; there is
no in-process PySCF backend.

See [`user_guide/multi_k_scf.md`](user_guide/multi_k_scf.md)
for the full multi-k story including the dense-Lpq RAM
ceiling caveat.

## Semiempirical methods + MLIP (v0.11.0-v0.12.0)

Status vocabulary: **production** = validated within the stated scope;
**experimental** = runs, but the route is not validated for published
numbers; **gated** = fails closed with a scientific error message;
**retirement target** = still executable, but the maintainer's target
matrix removes it from the portfolio. "FD" is finite difference. No
semiempirical or MLIP method has a nuclear Hessian: `run_job(hessian=True)`
is basis-free and skips vibrational analysis with a note in the `.out`.

| Method | `run_job(method=…)` | Elements | Molecular: energy / gradient / Hessian | Periodic routes (dimension, k-mesh) | Open-shell | Status and gate |
|--------|---------------------|----------|----------------------------------------|--------------------------------------|------------|-----------------|
| MSINDO | `"msindo"`; SECCM via `"seccm"` / `"ccm"` + `ccm_options` | INDO H-Xe (Z 1-54); NDDO H, Li-F, Na-Cl closed shell | INDO energy RHF/UHF and closed-shell analytic gradient. NDDO closed-shell energy and analytic gradient for H, Li-F, Na-Cl, including source-parity SPDD terms for Al through Cl. No Hessian | No Bloch Gamma and no full-k route (both fail closed). SECCM cyclic cluster 1-D/2-D/3-D, closed shell; native C++ for H-Kr, deterministic Python reference for Rb-Xe | INDO molecular UHF within validated fixtures; NDDO molecular and SECCM closed-shell only | **Production within scope.** Independent MSINDO INDO/NDDO implementation. Native C++ production routes cover supported molecular energies and INDO closed-shell analytic gradients; closed-shell NDDO gradients use a mixed native-energy/Python-analytic route. CCM energy/gradients use C++ for H-Kr and the deterministic Python reference for Rb-Xe; a selected native calculation does not fall back after failure. Python also remains for reference/orchestration paths including excited-state/root-tracking workflows |
| DFTB0 / UDFTB0 | `"dftb0"` | 91 in-house parameter records | Energy and analytic gradient, both native; no Hessian | Bloch Gamma energy + analytic gradient (native, 15-bohr image domain). Full-k Bloch energy/bands experimental; gradient and stress by one native batched FD pass; Monkhorst-Pack mesh of three positive integers, default 1x1x1. DFTB0-SECCM 1-D/2-D/3-D with analytic fixed-topology gradients | UDFTB0 molecular and Gamma; full-k and SECCM closed-shell only | **Screening / preoptimization; not production-validated.** In-house parameters, no DFT-fitted repulsives, no DFTB+ (mio-1-1) parity. Full-k derivatives gated pending analytic derivatives and DFTB+ parity. SECCM is gated experimental and scoped to the explicit repulsive-pair table (H, C, N, O, F, P, S, Cl). Absolute energies are not claimable while the repulsive is an A/R¹² placeholder (issue #306) |
| SCC-DFTB / USCC-DFTB | `"scc_dftb"` | 91 in-house parameter records | Energy and analytic gradient, both native; no Hessian | Bloch Gamma energy native, gradient by central differences of the converged energy. Full-k Bloch SCC energy/bands experimental with optional Fermi occupations; gradient and stress by one native batched FD pass; Monkhorst-Pack mesh of three positive integers, default 1x1x1. SCC-DFTB-SECCM 1-D/2-D/3-D with analytic fixed-topology gradients; charged cells only through the opt-in Madelung/Ewald embedding | USCC molecular and Gamma; full-k and SECCM closed-shell only | **Screening / preoptimization; not production-validated.** In-house parameters, no DFTB+ parity. Analytic SCC derivatives reject nonconverged results. Full-k derivatives and SECCM remain gated experimental |
| GFN2-xTB | `"gfn2_xtb"` | H-Rn (Z 1-86), fetched on demand | Energy, analytic gradient, scoped post-SCF native D4; no Hessian | Bloch Gamma energy native; the native gradient and all-nine stress match gapped finite differences. Full-k route fails closed. GFN2-SECCM 1-D/2-D/3-D, energy only, with optional Madelung embedding; Ewald-gamma supports 1-D/2-D and rejects non-molecular 3-D (#444) | None: the route is closed-shell only at every boundary | **Gated experimental.** Molecular H0, shell SCC, third-order, AES, analytic gradients, scoped post-SCF native D4, and FD-validated Gamma-periodic derivatives are implemented. Faithful periodic AES electrostatics and external `xtb` parity remain production gates. Native D4 covers H, He, B, C, N, O, F, Ne and returns zero with `GFN2D4UnsupportedWarning` elsewhere |
| PM6 / UPM6 | `"pm6"` | 75 chemical-element records from the bundled MOPAC cache | Energy native NDDO; gradient by native batched FD (no analytic gradient); no Hessian | Bloch Gamma energy native, gradient and stress by one native batched FD pass (mixed-native, experimental). Full-k route fails closed. PM6-SECCM 1-D/2-D/3-D energy with FD fixed-topology gradients; s/p AO channels only, so elements whose PM6 record needs d orbitals fail closed | UPM6 molecular only; periodic Gamma and SECCM closed-shell only | **Experimental: molecular development route.** Exact MOPAC heat-of-formation convention parity is open, and spherical Klopman-Ohno gamma/core-core fallbacks apply where the bundled MOPAC diatomic data is incomplete. Periodic PM6 is experimental |
| PM6-SECCM CCM Madelung embedding | opt-in `madelung=True` on `vibeqc.semiempirical.seccm.run_pm6_seccm` | as PM6 | not applicable (SECCM boundary only) | 1-D and 2-D only; three cyclic dimensions fail closed because the WS-folded remainder has no thermodynamic limit. Without the embedding a nontrivial cyclic topology needs an explicit `allow_truncated_electrostatics=True` acknowledgement | closed shell only | **Experimental, opt-in, acknowledgement-gated (issue #217).** The MSINDO CCM operator: the classical point-charge field enters the Fock diagonal and the potential is the full Ewald lattice sum minus the WS-internal bare 1/r, WS-weighted with the self image excluded. Unembedded cyclic runs are an algorithm-mechanics probe, not a solid-state prediction: the WS-truncated monopole sum reproduces the exact rocksalt Madelung constant only to -43%..+38% |
| OM2 / OM3 | `"om2"`, `"om3"` | 5 published (H, C, N, O, F) | Energy native NDDO; gradient by native batched FD; no Hessian | Bloch Gamma and full-k routes both fail closed: the retired prototype was not the published ORT/ECP/penetration Hamiltonian. OM2/OM3-SECCM 1-D/2-D/3-D energy only, no gradient; cyclic-image execution requires an explicit truncated-electrostatics acknowledgement; published eq-13 three-center image weighting is the default, eq-10 available for comparison | Molecular only; periodic and SECCM closed-shell or gated | **Retirement target** (maintainer target matrix, 2026-09-02: no molecular, Bloch or CCM target; fail closed, deprecate, remove from the public portfolio). Still executable today as a validated molecular development/prescreening surface; bond minima remain about 0.1-0.3 Å from published references and external implementation parity is open. Nothing has been removed yet, so no release has dropped it |
| OM1 | `"om1"` | 5 published (H, C, N, O, F) | Energy native NDDO with an experimental warning; gradient by native batched FD; no Hessian | Bloch Gamma, full-k, and OM1-SECCM all fail closed | Molecular only | **Retirement target** (same maintainer decision as OM2/OM3). Experimental: the defining analytic core-valence ECP (Kolb-Thiel 1993) is not implemented, which also closes OM1-SECCM. Nothing has been removed yet, so no release has dropped it |
| PM7 / UPM7 | `"pm7"`, `"upm7"`: recognized spellings that fail closed | 75 MOPAC parameter records retained, queryable only | Gated: no energy, no gradient, no Hessian | Gated at every boundary: Bloch Gamma, full-k, and SECCM all fail closed | Gated | **Retired as an implementation target** (maintainer target matrix, 2026-09-02). Tombstone only: every request raises "PM7 is gated until the published feathered electrostatics and all coupled NDDO terms are implemented and validated." Stewart's smooth exact-electrostatic/point-charge feathering is not implemented. PM7 has never been executable in any release, so there is no last version that carried it |
| MACE MLIP (v0.11.x) | `"mace"` for molecules; direct periodic drivers | Registered MACE-MP/MPA-0 materials models and MACE-OFF23 organic models | Energy, forces, optimization, and NEB; no Hessian | Direct periodic energy/forces support 1D, true 2D slabs, and 3D; `vibeqc.mlip.mace.optimize_periodic_mace_positions` provides fixed-cell relaxation with frozen atoms. Stress and `vibeqc.mlip.mace.optimize_periodic_mace_cell` are 3D-only. `run_periodic_job(method="mace")` is not supported | model-dependent | Optional `mace-torch` engine on Python <= 3.13. Model-specific energy scales are not electronic totals |

Sources for the statuses above include the `ACCEPTANCE`-managed families in
[`semiempirical_acceptance_matrix.py`](semiempirical_acceptance_matrix.py), the queryable
`vibeqc.semiempirical.semiempirical_route_status(route)` registry, the
boundary gates in `vibeqc.semiempirical.routes.SemiempiricalRoutePlan`,
and `handovers/HANDOVER_SEMIEMPIRICAL_ROADMAP.md` (§ "2026-09-02
maintainer-approved target matrix" and § "Paper-safe status"). The
per-route user-guide table is at
[`user_guide/semiempirical.md`](user_guide/semiempirical.md) § Method
status.

## Effective core potentials

libecpint 1.0.7 vendored. Heavy-element chemistry (d-block,
lanthanides, actinides) reachable without all-electron basis.

| ECP library | Family | Coverage |
|-------------|--------|----------|
| `ecp10mdf` | Stuttgart-Köln MDF, 10-electron core | K-Kr |
| `ecp28mdf` | Stuttgart-Köln MDF, 28-electron core | Rb-Xe |
| `ecp46mdf` | Stuttgart-Köln MDF, 46-electron core | Cs, Ba |
| `ecp60mdf` | Stuttgart-Köln MDF, 60-electron core | Hf-Rn, Ac-U |
| `ecp78mdf` | Stuttgart-Köln MDF, 78-electron core | Fr, Ra |
| `lanl2dz` | Hay-Wadt LANL | Na-La, Hf-Bi, U-Pu |

These are the elements actually present in each bundled XML, read from the
files themselves; they are narrower than the family names suggest. In
particular **no bundled XML library covers Ce-Lu**: the lanthanides are
reached through a basis' own `.ecp` sidecar (pob-TZVP-rev2 carries La-Lu),
not through `ecp_library`.

A bundled basis' `.ecp` sidecar is attached inline and per element by
the molecular SCF wrappers; the XML libraries above serve explicit
`ecp_centers` + `ecp_library` requests. Whether an atom needs an ECP in
a basis is answered by `vibeqc.basis_registry.ecp_requirements`, never by
the basis name. MP2, CC, DLPNO, the double hybrids, ROHF/ROKS and the
determinant solvers consume an ECP reference through its valence count
and an ECP-aware frozen core; CASCI/CASSCF/MRCI/NEVPT2/CASPT2 and OVGF
still refuse one. See [`user_guide/ecp.md`](user_guide/ecp.md).

Analytic nuclear gradients are ECP-aware on all four molecular
drivers when the exact XML-library or inline-primitive ECP provenance is
repeated on `GradientOptions`. The ASE force path copies those fields
automatically. The gradient applies Z_eff nuclear pieces plus dV_ECP/dR via
libecpint first derivatives, with analytic-vs-FD agreement pinned to
<= 1e-5 Ha/bohr. Gas-phase molecular ECP geometry optimization and
finite-difference Hessians preserve the same operator while centers follow
their atoms. Analytic ECP Hessians, CPCM forces and Hessians, and top-level
ECP NEB currently fail closed rather than differentiating the wrong energy.
See
[`user_guide/ecp.md`](user_guide/ecp.md) § Gradients.

Reference: Pt UHF/LANL2DZ = **−118.227 Ha** (22 BFs, 125 SCF
iters).

## SCF convergence aids

| Aid | API | Coverage | Notes |
|-----|-----|----------|-------|
| DIIS extrapolation | `*_options.use_diis` | molecular + periodic Γ + multi-k | default on, ~10× iter speedup |
| **EDIIS + EDIIS+DIIS hybrid** (v0.8.0) | `*_options.scf_accelerator=SCFAccelerator.EDIIS_DIIS` | molecular + C++ direct-truncated periodic + Python Γ-Ewald (RHF/RKS/UHF/UKS) | flagship for stiff convergence; rollout to Python multi-k Ewald / BIPOLE / GDF in progress |
| **KDIIS** (Kollmar) | `*_options.scf_accelerator=SCFAccelerator.KDIIS` | molecular + C++ Γ-only periodic + Python Γ-Ewald (RHF/RKS/UHF/UKS) | orbital-rotation gradient error; ORCA's opt-in `!KDIIS`; multi-k path needs per-k MO-basis design |
| **ADIIS** (Augmented Roothaan-Hall) | `*_options.scf_accelerator=SCFAccelerator.ADIIS` | molecular + C++ direct-truncated periodic + Python Γ-Ewald | EDIIS sibling, no per-iter energy needed |
| **Dynamic damping** (Zerner-Hehenberger 1979) | `*_options.dynamic_damping`, `dynamic_damping_min/_max` | molecular + C++ direct-truncated periodic + Python Γ-Ewald (RHF/RKS/UHF/UKS) | adaptive density mixing α; composes freely with the accelerator family |
| Damping (static) | `*_options.damping` | all | density mixing |
| CRYSTAL-style FMIXING | `*_options.fock_mixing`; `run_periodic_job(fmixing_percent=30)` | molecular + native periodic direct + supported GDF routes | Fock/KS matrix mixing; `FMIXING 30` = 0.30, separate from density damping; GDF exceptions are fail-closed or AUTO-filtered as documented in [`scf_convergence.md`](user_guide/scf_convergence.md) |
| Saunders-Hillier level shift | `*_options.level_shift` | molecular + periodic Γ + multi-k, with route-dependent GDF coverage | Phase C1a; GDF exceptions are fail-closed or AUTO-filtered as documented in [`scf_convergence.md`](user_guide/scf_convergence.md) |
| Level-shift warm-up/restart | `*_options.level_shift_warmup_cycles`; `run_*_periodic_gdf(..., level_shift_warmup_cycles=...)` | native Γ-GDF + Γ KRHF/KRKS bridge | `-1` auto, `0` persistent shift, positive values restart unshifted after that many cycles |
| Quadratic ("Newton") SCF | `*_options.quadratic_fallback_iter` | molecular + periodic Ewald Γ/multi-k | Phase C1c, fallback for small-gap systems; experimental `aiccm2026dev-b` routes require 0 and fail closed on activation |
| Fermi-Dirac smearing | `*_options.smearing_temperature`; `run_periodic_job(smearing_temperature="auto"|"metal"|...)`; `vq.SmearingOptions`, `vq.apply_smearing()`, `vq.resolve_smearing_temperature()` | periodic multi-k Ewald RHF/RKS + multi-k GDF KRHF (Γ-only Ewald RHF/RKS inherits via dispatch) + BIPOLE RKS/UHF/UKS (Γ and multi-k); BIPOLE RHF intentionally rejects smearing, while remaining non-BIPOLE open-shell runner routes, Γ-only GDF smearing and C++-direct Ewald Γ remain queued (Methfessel-Paxton and Marzari-Vanderbilt smearing ship at v0.14.0), see [`docs/user_guide/smearing.md`](user_guide/smearing.md) and [`docs/design_smearing.md`](design_smearing.md) | Phase C1b; accepts Hartree/K/eV/Ry, named presets, conservative auto guess; result reports occupations, entropy, free energy, and Fermi level. Shared `vibeqc.smearing` package consolidates the utility (v0.10.x M0/M1 refactor) |
| **Periodic density mixers** (Anderson / Broyden / Kerker, v0.14) | `run_periodic_job(..., density_mixer=...)`; `run_rks_periodic_scf(...)` / `run_rks_periodic_multi_k_ewald3d(...)` | closed-shell explicit-kpoint GDF dispatch + lower-level multi-k RKS Ewald | Kerker preconditions the density residual without moving the SCF fixed point; unsupported BIPOLE, GPW/GAPW, open-shell, and implicit-Gamma routes fail closed |
| `auto_optimize_truncation` | `PeriodicRHFOptions(auto_optimize_truncation=True)` | periodic | default on; jointly optimises lattice cutoffs + Schwarz screening |

## XC functionals (v0.8.0 expanded)

Resolved through libxc 7.0.0; see
[`user_guide/functionals.md`](user_guide/functionals.md). Short
aliases the resolver knows:

| Alias | Type | HF-exchange | libxc id |
|-------|------|-------------|----------|
| `lda` / `svwn` | LDA | 0 % | 1 + 7 |
| `pbe` | GGA | 0 % | 101 + 130 |
| `blyp` | GGA | 0 % | 106 + 131 |
| `b3lyp` (= `b3lyp5`) | hybrid | 20 % | 475 (VWN5, the ORCA / TURBOMOLE / ADF / CRYSTAL convention) |
| `b3lyp/g` (= `b3lypg`) | hybrid | 20 % | 402 (Gaussian/VWN-RPA variant, what Gaussian / PySCF / Psi4 mean by the bare name) |
| `pbe0` / `pbeh` | hybrid | 25 % | 406 |
| `pw1pw` | hybrid | 20 % | weighted-sum |
| `hse06` | range-sep hybrid | 25 % SR | 425 |
| `vv10` (v0.14) | GGA + nonlocal | 0 % | 255 |
| `wb97x-v` (v0.14) | range-sep + VV10 | 100 % LR | 466 |
| `wb97m-v` (v0.14) | range-sep meta-GGA + VV10 | 100 % LR | 531 |
| `r2scan01` (v0.14) | meta-GGA | 0 % | 642 + 645 |
| `revdsd-pbep86` (v0.14) | double hybrid (D4) | 69 % | 101 + 132 |
| `skala-1.1` (introduced v0.15.158; periodic expansion v0.15.160; **experimental**) | neural XC (Microsoft SKALA-1.1, MIT-licensed checkpoint fetched on first use) | 0 % | external provider, not libxc; needs the `[skala]` extra (PyTorch); energies only, fixed geometry; molecular RKS / UKS / ROKS plus selected periodic AO-grid routes (GDF, multi-k RIJCOSX, 3D BIPOLE, GPW/GAPW, and AICCM real-Gamma, literal-WSSC, and four-centre chi routes); forces, Hessians, TDDFT and response paths fail closed; see [`user_guide/skala.md`](user_guide/skala.md) |

Plus everything libxc accepts directly + custom weighted-sum
strings (PW1PW pattern: `"0.2*HF + 0.8*GGA_X_PW91, GGA_C_PW91"`).

**B3LYP convention**: vibe-qc ships the ORCA definition, bare
`b3lyp` is libxc id 475 (VWN5), the same flavor ORCA, TURBOMOLE,
ADF, CRYSTAL, and CP2K mean by the name; Gaussian, libxc, PySCF,
Psi4, NWChem, and Q-Chem mean the VWN-RPA variant (id 402), which
vibe-qc spells `b3lyp/g` / `b3lypg`. The flavor gap is ~6.8 mHa on
H₂/STO-3G, ~10-15 mHa per heavy atom. v0.12.0 added the explicit
`b3lyp5` / `b3lypg` spellings and a measured cross-code audit. See
[`user_guide/functionals.md`](user_guide/functionals.md) for the
camp table, the measured values, and why there is no `b3lyp3`
alias.

## Ewald summation (Phase 12e, mostly shipped)

See [`user_guide/ewald.md`](user_guide/ewald.md) for the math
+ API.

| Sub-phase | Scope | Status |
|-----------|-------|:-:|
| 12e-a | Classical Ewald nuclear lattice sum (`CoulombMethod.EWALD_3D`) | ✅ |
| 12e-b | erfc-screened nuclear attraction `compute_nuclear_erfc_lattice` | ✅ |
| 12e-c-1 | Gaussian-charge Ewald *V(g)* via grid integration | ✅ |
| 12e-c-2 | erfc-screened ERIs for Ewald short-range J/K | ✅ |
| 12e-c-3a | FFTW3 build dep + `solve_poisson_*` | ✅ |
| 12e-c-3b | `build_j_long_range` + `auto_grid` | ✅ |
| 12e-c-3c | Quartet multipole far-pair replacement | Fail-closed research prototype |
| 12e-c-4 | End-to-end EWALD_3D SCF dispatch (RHF/RKS/UHF/UKS, Γ + multi-k) | ✅ |

Madelung-constant validation:

| Crystal | M reproduced | Reference |
|---------|--------------|-----------|
| NaCl rocksalt | 1.7475645946 to 1e-8 | 1.7475645946... |
| CsCl | 1.762674773 to 1e-8 | 1.762674773... |
| ZnS zincblende | 1.6380550533 to 1e-9 | 1.6380550533... |
| Simple-cubic jellium | 1.4186487 to 1e-6 | 1.4186487 (Nijboer-De Wette 1957) |

## External data fetcher (`vqfetch`, v0.8.0)

Pulls structures + reference data from public databases on
demand. Per-record provenance + license preserved.

| Subcommand | Source | Default license |
|------------|--------|-----------------|
| `vqfetch optimade --formula MgO` | OPTIMADE federation | per-provider |
| `vqfetch mp --id mp-1265` | Materials Project | CC-BY 4.0 |
| `vqfetch cod --id 1011027` | COD | CC0 / public domain |
| `vqfetch canonical <slug>` | (5 round-trip-verified slugs) | (per primary) |
| `vqfetch reference --cas 7732-18-5` | NIST CCCBDB | US Govt public domain |

See [`user_guide/external_structures.md`](user_guide/external_structures.md)
+ [`user_guide/reference_data.md`](user_guide/reference_data.md).
End-to-end compute-reference round-trip: MgO sto-3g RKS-LDA E =
**−950.4204308512 Ha** (13 SCF iters).

## Job queue (`vq` v0.5.6, v0.8.0)

SSH-backed remote job submission, installed separately from
[the independent vibe-queue project](https://github.com/vibe-qc/vibe-queue). Cgroup-v2 enforcement, web dashboard,
pause/resume, multi-venv `--branch` routing. See
[`user_guide/queue.md`](user_guide/queue.md).

| Capability | Since |
|-----------|-------|
| Local queue + CLI + daemon + JobSpec v1 + systemd unit | v0.1.0 |
| Cross-machine SSH submit + `~/.config/vq/config.toml` + `vq fetch` | v0.2.0 |
| Resource watchdog (`--mem-mb`, `--wall-time-seconds`, OOM_KILLED / STARVED / TIME_EXCEEDED) | v0.3.0 |
| Cgroup-v2 enforcement (systemd-run scope), pgid-based daemon recovery | v0.4.0 |
| `ABORTED_BY_QUEUE` terminal state | v0.4.1 |
| Read-only web dashboard (FastAPI + htmx) | v0.5.0 |
| `vq pause` / `vq resume` + bearer-token auth + write API | v0.5.1 |
| `--all` flag on pause/resume + queue-wide `POST /api/v1/queue/*` | v0.5.2 |
| CRYSTAL14 + PROPERTIES14 reachable through dispatched jobs | v0.5.3 |
| Parallel CRYSTAL14 (Pcrystal/Pproperties default `--np 14`) + clean systemd PATH drop-in | v0.5.4-5 |
| Multi-venv routing via `--branch` for vibe-qc dev/release dispatch | v0.5.6 |

## Properties

| Property | API | Methods | Notes |
|----------|-----|---------|-------|
| Mulliken charges | `mulliken_charges` | RHF / UHF / RKS / UKS | atomic populations |
| Löwdin charges | `loewdin_charges` | all | symmetric-orth populations |
| Mayer bond orders | `mayer_bond_orders` | all | per-pair indices |
| Dipole moment | `dipole_moment` | all | with optional origin |
| Natural orbitals | `natural_orbitals` | all | for post-SCF analysis |
| Static polarizability α | `dipole_polarizability_rhf` | RHF (closed-shell) | via CPHF, Phase 17b-1; UHF/KS roadmap |
| Vibrational frequencies (FD) | `compute_hessian_fd` + `HessianResult.frequencies_cm1` | all | FD on analytic gradient; trans/rot projection |
| Vibrational frequencies (analytic) | `compute_hessian_rhf_analytic` | RHF | CPHF + libint deriv_order=2; Phase 17b-3 |
| IR intensities | `ir_intensities` | all | dipole derivatives along normal modes |
| Thermochemistry (ZPE / U / H / S / G / Cv) | `compute_thermochemistry` | all | rigid-rotor + harmonic-oscillator + ideal-gas |
| **D3(BJ) dispersion** | `compute_d3bj` | all | Grimme D3 with Becke-Johnson damping |
| **D4 dispersion** (v0.8.0) | `compute_d4` | all | Caldeweyher-Bannwarth-Grimme 2019 |
| Cube file output | `write_cube` | all | density / orbital / spin density |
| Molden file export | `write_molden` | RHF / UHF / RKS / UKS | verified by PySCF round-trip |
| Periodic band structure | `band_structure` | periodic SCF | k-path sampling |
| Periodic density of states | `density_of_states` | periodic SCF | Gaussian broadening on a sampled mesh |
| Atomisation energy | `run_job(..., atomization=True)` (`vibeqc.atomization.atomization_energy`) | HF / DFT | per-element free-atom references at the same level, cached; main-group H-Kr. MSINDO reports its binding energy by default. |
| Green's function IP / EA | `run_job(method="ovgf")` → `result.ovgf`; MSINDO: `msindo_ovgf` | HF / MSINDO | quasiparticle ionisation potentials and electron affinities (`.ip_homo_ev` / `.ea_lumo_ev` on the MSINDO result) |
| MPI substrate (v0.12.0) | `mpi4py` optional extra; `vibeqc.mpi` helpers | experimental GPW z-slab overlay plus low-level collectives | install with `[mpi]` extra. Production distributed HF / DFT / MP2 drivers are not wired yet; see [MPI Parallelization](design_mpi_parallelization.md). The GPW z-slab grid overlay is experimental (multi-rank Hartree-J build broken through v0.12.0, fixed post-release; pending real multi-rank validation) |
| **COOP/COHP** (v0.15.0) | `compute_coop_cohp` | periodic SCF | energy-resolved Crystal Orbital Overlap/Hamilton Population; C++ kernels, QVF `dos.coop`/`dos.cohp`, plotters, `vibeqc coop` CLI |
| **Periodic Mayer bond orders** (v0.15.0) | `periodic_mayer_bond_orders` | periodic SCF | k-space generalisation; QVF `bond_orders` |
| **QTAIM** (v0.15.0) | `vibeqc.qtaim.qtaim_analysis` | all | critical-point search + bond-path tracing; analytic C++ Hessian; QVF `topology.qtaim` |
| **Fat bands** (v0.15.0) | `band_structure_projected` | periodic SCF | Mulliken-projected band weights; QVF `bands` projections |
| **Wiberg bond indices + delocalization index** | `vibeqc.bond_analysis.wiberg_bond_orders` / `.delocalization_index` / `.bond_order_summary` | all | Löwdin-basis Wiberg 1968 index; AO-approximated DI; auto-emitted in `.population.{txt,json}` |
| **NPA charges + NBO search** | `vibeqc.nbo.npa_charges` / `.nbo_search` / `.donor_acceptor_analysis` | all | Weinhold NPA (auto-emitted in `.population.*`); BD/LP/CR/BD*/RY* classification; E(2) donor-acceptor |
| **Energy decomposition analysis** | `vibeqc.eda.eda_lmo` / `.eda_morokuma` | HF / DFT | LMO-EDA (Su-Li 2009) + Morokuma 1971 two-fragment decomposition |
| **Orbital entanglement** | `vibeqc.entanglement.entanglement_from_density` / `.single_orbital_entropy` / `.mutual_information` | all | single-orbital entropy, mutual information, entanglement bond orders, correlation clusters |

## Solvation (CPCM / COSMO v0.9.0, COSMO-RS v0.15.158)

See [`user_guide/solvation.md`](user_guide/solvation.md).

| Capability | API | Methods | Status | Notes |
|------------|-----|---------|--------|-------|
| CPCM / COSMO implicit solvation | `run_job(..., solvent=...)` via `vibeqc.solvation.run_cpcm_scf` | RHF / UHF / RKS / UKS; MSINDO through its own COSMO driver | production | macro-iteration on a union-of-spheres cavity; presets in `vibeqc.solvation.presets` (`SOLVENT_PRESETS`) or a custom `SolventModel`; `epsilon=inf` is the exact conductor limit (#548) |
| Solvent analytic gradient / geometry optimisation in solvent | `vibeqc.solvation.cpcm_gradient` (finite-difference check: `vibeqc.solvation.cpcm_gradient_fd`) | RHF / UHF / RKS / UKS | production | the screening factor is carried from the run's own variant (#546, #548, #549) |
| COSMO-RS / COSMOSPACE thermodynamics | `vibeqc.solvation.cosmors` on a `vibeqc.solvation.surface.ConductorSurface` | any method that can produce a conductor surface | **experimental** (v0.15.158) | sigma profiles and potentials, chemical potentials, activity and partition coefficients, solvation free energies (Klamt 1995; Klamt, Jonas, Bürger, Lohrenz 1998; `klamt1998` set); the shipped cavity differs from the one the 1998 constants were fitted on, FreeSolv validation pending (#558) |
| Solvent with correlated, CAS, semiempirical (except MSINDO) or MLIP methods | `solvent=` | mp2 / cc / dlpno / cas / gfn2_xtb / pm6 / mace, … | **gated** (refuses with `ValueError`) | no validated post-SCF-in-solvent composition yet; ROHF / ROKS raise `NotImplementedError` |

## Input / output

| Capability | API | Notes |
|------------|-----|-------|
| High-level "run-a-job" driver | `run_job` / `run_periodic_job` | dispatches to right SCF driver, writes .out + .molden, runs BFGS if asked |
| MPI substrate (v0.12.0) | `pip install -e '.[mpi]'` | optional `mpi4py` extra; low-level collectives plus experimental GPW grid overlay. Production MPI strategy is tracked in [MPI Parallelization](design_mpi_parallelization.md) |
| Formatted SCF log | `format_scf_trace`, `log_scf_trace` | banner, iteration trace, energy components, orbital table, HOMO-LUMO gap |
| Molden export | `write_molden` | verified by PySCF round-trip |
| TREXIO export / import (v0.15.x, #573) | `run_job(trexio=True)`, `write_trexio` / `read_trexio` | optional `[trexio]` extra (BSD-3-Clause, never bundled); HDF5 and text back ends. Written groups: `metadata`, `nucleus`, `electron`, `basis` (Gaussian, spherical), `ao`, `mo`, `ao_1e_int` (overlap, kinetic, potential_n_e, core_hamiltonian), `state`; RHF / UHF / RKS / UKS and restricted open-shell molecular results. Not written: `ecp` (ECP runs are refused), `rdm`, `ao_2e_int`, `mo_1e_int` / `mo_2e_int`, `cell` / `pbc`, determinants. Conventions pinned by an exact in-process round trip and an out-of-process PySCF energy rebuild from the stored MOs; see [TREXIO](user_guide/trexio.md) |
| Geometry trajectory | `run_job(..., optimize=True)` | ASE .traj file |
| XYZ / Extended-XYZ load | `Molecule.from_xyz` / `from_xyz` | molecular XYZ by default; `periodic=True` reads Extended-XYZ `Lattice` / `pbc` into `PeriodicSystem`; `symmetrise=True` returns `SymmetriseResult` |
| POSCAR load / save | `read_poscar`, `write_poscar` | VASP 5 format |
| CIF load / save | `read_cif`, `Crystal.from_cif`, `write_cif` | cell parameters + atom-site loops; symmetry operations expand asymmetric-unit sites |
| Cube load / save | `read_cube`, `write_cube` | volumetric data |
| BXSF (band structure) | `write_bxsf` | XCrySDen 3D Fermi-surface format |

## Basis sets

* **255 bundled Gaussian `.g94` basis files** in the runtime
  `basis/` library: 90 libint-inherited standard files plus 165
  vibe-qc custom / BSE-derived overlay files, with matching
  QVF-Basis sidecars for every set.
* **Custom and BSE-derived sets** in
  `python/vibeqc/basis_library/custom/`: pob-TZVP,
  pob-DZVP-rev2, pob-TZVP-rev2 (Bredow-group periodic-tuned),
  plus the BSE-sourced and release-paper reproduction bases.
* **CRYSTAL-format parser**, imports arbitrary CRYSTAL
  per-element basis files; converts to libint-compatible
  `.g94`. CRYSTAL `INPUT` ECP-block parsing is available for
  basis/ECP migration work.
* **ECP basis sets**, fully wired through libecpint (see
  ECP section above).
* **Basis-set optimisation** (v0.14), re-optimise a basis set's
  exponents and coefficients against a molecular energy via the
  analytic-gradient BDIIS recipe (`optimize_molecular_basis`); see
  [`user_guide/basis_optimization.md`](user_guide/basis_optimization.md).
* **Basis inventory and attribution** documented in
  [`docs/license.md`](license.md#3a-bundled-basis-sets), including
  the libint-inherited files, the Bredow-group pob-\* bases, and
  the BSE-derived custom overlay.

## Crystal / lattice infrastructure

* **spglib** integration: `Crystal`, `analyze_symmetry`,
  `detect_spacegroup`, `symmetrise`, `to_primitive`,
  `irreducible_kpoints`.
* CIF, POSCAR, and periodic Extended-XYZ input, with opt-in
  `symmetrise=True` standardisation into a calculation-ready
  `PeriodicSystem`.
* Monkhorst-Pack k-mesh generation with IBZ reduction, density-based
  auto-mesh helpers, and on-the-fly generalized regular grids via
  `KPoints.optimal`.
* Standardise-then-compress helpers for one-electron lattice
  integrals: `compute_overlap_lattice_symmetrised_with_orbits` plus
  kinetic / nuclear siblings rebuild the basis on the cleaned cell and
  return the SYM3a orbit-compressed storage view.
* 1D / 2D / 3D `PeriodicSystem` geometry.
* Hexagonal and other skew-cell geometry support; the native
  EWALD_3D Hartree path uses the analytical AO-pair FT on the full
  reciprocal metric.

## Validation

| Test class | Count | Level |
|------------|------:|-------|
| Molecular 1e integrals vs PySCF | 24 | machine precision |
| RHF / UHF / RKS / UKS energies vs PySCF | 59 | machine precision (HF), grid-accuracy (DFT) |
| MP2 / UMP2 / RI-MP2 / SCS-MP2 / SOS-MP2 / B2PLYP vs PySCF | 55 | 1e-9 Ha (MP2 family) / 1e-7 Ha (B2PLYP) |
| Gradients (HF, DFT) vs FD / PySCF | 17 | 1e-6 Ha/bohr |
| FD Hessian skeleton vs FD-on-gradient | 8 | 1e-4 to 1e-3 |
| Analytic RHF Hessian vs PySCF analytic | 8 | 2.5e-9 Ha/bohr², freqs <0.01 cm⁻¹ |
| FD Hessian + IR + thermochemistry vs PySCF | 51 | 1e-7 Ha |
| CPHF + polarizability vs PySCF FD | 11 | 1e-5 a.u. |
| Periodic machinery (invariants, molecular limit, Bloch folding) | 76 | machine precision |
| Periodic SCF convergence aids | 27 | per-test scoped |
| **Periodic native GDF/FFTDF parity** | planned | compare parsed out-of-process CRYSTAL and PySCF outputs |
| **Multi-k KRHF/KRKS native parity** | planned | no in-process PySCF backend |
| **JKBuilder + RIJCOSX vs ORCA 6.1.1** (v0.8.0) | curated | RIJCOSX SCF energy max \|Δ\| = 0.26 mHa on glycine def2-TZVP (energy only; gradient parity tracked separately, see § Density fitting + RIJCOSX) |
| **vqfetch acceptance harness** (v0.8.0) | 5 structures + 8 molecules | round-trip end-to-end |
| Crystal / POSCAR / basis parsing | 18 | - |
| **Bundled basis-library integrity** | 255 `.g94` files + 44 ECP sidecars | generated QVF sidecars, provenance / citation coverage, ECP split, privacy checks |

Plus the **full regression suite** at
[`examples/regression/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression)
(since v0.7.2 *Boys' Crucible*) covers the cross-code parity
matrix (vibe-qc vs PySCF vs ORCA) on a curated S22 / X23 /
pob-TZVP test set.

Full pytest suite: ~1800 test functions across ~190 files
covering the molecular, periodic, wavefunction-solver, and
output stacks; runs in ~3-4 minutes on an M-class MacBook
(faster with `pytest -n auto`).

## Known issues

See the warning admonition on [`docs/index.md`](index.md) for
the live open-bugs list. Highlights:

* DFTB0 and SECCM-DFTB0 absolute energies are not chemistry while the
  repulsive term is an A/R¹² placeholder (issue #306, open); the
  semiempirical matrix above therefore labels DFTB0 and SCC-DFTB
  screening/preoptimization rather than production. Fixed-geometry
  differences stay valid, absolute energies and EOS fits do not.
  Periodic GFN2-xTB analytic gradient and stress are not derivatives of
  the periodic energy (issue #338, open), so the periodic analytic
  stress is gated off.
* Periodic GDF parity, Γ-only and multi-k µHa parity vs PySCF on LiH
  FCC shipped at v0.11.0 (*Sun's Stingray*). Mixed Density Fitting (MDF)
  shipped at v0.13.0, closing the all-electron GDF accuracy floor
  (Sun-Berkelbach 2017) for MgO. Dense-crystal periodic-DFT XC error
  fixed in v0.15.0 (cross-cell density + SCF convergence).
* Multi-k dense Lpq RAM ceiling (multi-TB on production
  systems). Streaming-Lpq tracks with the periodic GDF work.
* Si-diamond / C-diamond RKS/PBE oscillation on RI path -- SCF-settings
  investigation ongoing (damping/DIIS interplay, not algorithmic).
  Density mixers (Anderson/Broyden) shipped in v0.14.0 as an alternative
  convergence path.
* BIPOLE gradients: production forces finite-difference the supported exact
  SCF energy. Corrected- and legacy-gauge analytic implementations remain
  gated previews with a narrower maintained regression envelope; they are not
  evidence that the fail-closed quartet multipole gradient is integrated.
* Open-shell UHF / UKS analytic gradient on f-shells with two
  or more different second-row elements still disagrees with
  ORCA. Closed-shell direct gradients now auto-route through
  the DF gradient path; the open-shell auto-route is queued.
* AUTO initial guess (SAP on closed-shell light-atom systems)
  oscillates on long n-alkanes and on H2CO + PBE. Pin
  `InitialGuess.SAD` to work around.
* `pob-TZVP` / `pob-TZVP-rev2` missing Ne entry.
* Heavy-atom basis-load test OOMs a 16 GB laptop in batch mode;
  route via `vq submit` to a larger box.
* ωB97X / ωB97X-D UKS on orbital-near-degenerate radicals needs
  a level shift to converge.
* 3c composite methods are calibrated on molecular systems; the
  periodic flavours need recalibration before being trusted.

The B3LYP local-correlation convention (bare `b3lyp` = VWN5 /
ORCA definition since v0.8.0; explicit `b3lyp5` / `b3lypg`
spellings since v0.12.0) is documented in
[`user_guide/functionals.md`](user_guide/functionals.md) as
**informational, not a bug**, both flavors are correct B3LYPs;
codes differ in which one the bare name means.
