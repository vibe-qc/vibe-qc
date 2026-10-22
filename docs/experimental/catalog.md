# Experimental features

vibe-qc ships a number of features behind an **experimental gate**. They
are reachable, you can turn one on and get a result, but they are not yet
production-certified: a feature may be incomplete, may change or break
between releases without the usual deprecation cycle, or is known to be
inaccurate in some regime. Most of them emit a feature-specific warning
class every time you use them, so an experimental result never slips into a
script unnoticed. Where a route has no warning class, the label lives
somewhere else that the table names: the `.out` route line, the `.system`
manifest's `method_status`, the semiempirical status registry, a source
docstring, or a fail-closed opt-in.

This page catalogs every experimental feature the code declares, whether or
not it warns. Methods labeled for screening or validated prescreening on
their own status pages are not thereby external-parity production methods
merely because they do not emit a warning. The four labels this page uses
(production, preview, experimental, gated) are defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

## What "experimental" means here

There are two reasons a feature carries the gate, and the difference matters
for whether you can trust its numbers:

- **Production-quality in its supported regime, labelled because the broader
  feature is still being built.** GPW is the example: the shipped envelope
  (Gamma RHF/UHF/RKS/UKS, multi-k pure-DFT RKS/ROKS/UKS, Gamma analytic
  gradients, DFT+U, D3-BJ) reproduces its references and is production within
  that envelope, but the runner prints an experimental tag in the `.out` route
  description and the grid infrastructure warns once per build while the
  GPW/GAPW family is completed. Safe to use within the
  documented regime. GAPW, the all-electron augmentation on top of it, is
  experimental in its own right.
- **Not yet accurate enough for published numbers.** GFN2-xTB, PM6, the OMx
  family, SKALA, COSMO-RS and every AICCM variant are in this group: they
  run, but a known defect, a missing term or missing external parity makes
  them development and qualitative only. The native D4 backend is narrower:
  it is quantitative for its documented H, He, B, C, N, O, F, Ne scope and
  fails closed outside it. The "Quantitative" column below flags these.

Roadmap features are different: those raise `NotImplementedError` and are
not reachable at all (see [the roadmap](../roadmap.md)). Experimental features
*are* reachable; they just carry the label.

The BIPOLE quartet multipole far-field belongs with unavailable features, not
with the reachable experimental catalog. It is unavailable from every SCF
driver. The supported exact Ewald-J route defaults to
`use_multipole_far_field=False`, and explicit `True` raises before setup.

## Catalog

In this table, Γ-CCM and χ-CCM[^xccm-convention] are distinct constructions.
Γ-CCM is the neutral finite-torus construction (variants `real-gamma` and
`neutral-bloch`); the `four-center` variant is the 2014 union-and-weight
lineage kept in the same development line; χ-CCM is the
finite-translation-group character construction.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    Γ-CCM is the neutral finite-torus construction with two producers,
    `variant="real-gamma"` and `variant="neutral-bloch"` (ruling R1, 2026-08-21),
    while the `four-center` variant is the 2014 union-and-weight/Wigner-Seitz
    lineage kept in the same `aiccm2026dev-a` development line; χ-CCM is the
    finite-translation-group character construction. A declared common
    exchange-q=0 convention is a comparison constraint, not an identification
    of the constructions. Their distinction is not a choice of Coulomb kernels,
    and equality for a specified operator and route must be established rather
    than inferred from the labels.

### Periodic routes

| Feature | Area | Opt in | Status and key caveat | Quantitative |
|---|---|---|---|---|
| **GPW** (Gaussian plane waves) | periodic | `run_periodic_job(jk_method="gpw")` | **production within its shipped envelope**: Gamma RHF/UHF/RKS/UKS, multi-k pure-DFT RKS/ROKS/UKS, Gamma analytic gradients, DFT+U, D3-BJ. ROHF/ROKS are a maintained preview with one integer-occupation restricted orbital set and reject gradients and smearing; multi-k ROKS hybrids remain gated. The `.out` route line carries an experimental tag while the GPW/GAPW family is completed; the runner silences the driver-level `GAPWExperimentalWarning` with `quiet=True`, but the plane-wave grid and J-builder infrastructure still emit it once per build, so a GPW run does warn | Yes, in the supported regime |
| **GAPW** (all-electron augmentation) | periodic | `run_periodic_job(jk_method="gapw")` (`gapw_molecular_limit=True` required for RHF/UHF) | **experimental**; the run-level warning was retired but the route stays experimental, and the library augmentation, range-separation and OT drivers still emit `GAPWExperimentalWarning` when called directly. Not the JK AUTO default; declared molecular-limit Gamma RHF/UHF uses fit-free analytic one-centre augmentation, Gamma RKS/UKS and multi-k pure-DFT RKS use the block construction; multi-k HF/hybrids and compact-crystal augmentation validation remain gated; converge atom and plane-wave grids for absolute totals | Partial |
| **Periodic RIJCOSX** | periodic | `run_periodic_job(jk_method="rijcosx")` | **experimental**: RI-J from the native GDF loop plus periodic COSX exchange (`k_exchange="cosx"`); Gamma RHF plus true multi-k RHF/RKS/UHF/UKS. No warning class; the `.out` route description string says "experimental". One-cell tight-cell RKS/UHF/UKS fail closed rather than relabeling a GDF-only route. See [periodic methods](../user_guide/periodic_methods.md) | Partial |
| **BIPOLE multi-k KS analytic gradient** | periodic | `relax_atoms(..., force_mode="analytic")` or direct `compute_bipole_gradient_*` calls | maintained preview (plain `UserWarning`); the corrected gauge is finite-difference-validated, broader KS certification is pending. High-level optimization defaults to the exact finite-difference force path | Partial |
| **MPI GPW grid overlay** | periodic | `GpwJBuilder(mpi_aware=True)` (not via `run_periodic_job`) | distributed-memory Hartree-J; the multi-rank build was fixed after v0.12 and is pending validation on a real multi-rank run | No |
| **Surface embedding** (Green's function) | periodic | the embedded-surface calculator, see [surface embedding](../user_guide/surface_embedding.md) | emits `EmbeddedSurfaceExperimentalWarning`; the 3D Gaussian foundation is validated, thick-slab GDF parity is pending, so absolute energies and densities are qualitative | No |
| **AICCM, Γ-CCM line / aiccm2026dev-a** | periodic | `run_periodic_job(method="aiccm", variant="real-gamma" \| "neutral-bloch" \| "four-center")`; library `vibeqc.periodic.ccm` (`method="aiccm2026dev-a"` selects the four-centre lineage, `run_ccm_scf(route=...)` selects any of the three) | Γ-CCM is the neutral finite-torus construction with two producers, `real-gamma` and `neutral-bloch` (ruling R1); the `four-center` variant is the 2014 union-and-weight/Wigner-Seitz lineage kept in the same development line, with HF, KS, MP2 and CCSD(T). Every SCF driver emits `AICCM2026DevAExperimentalWarning` and the `.system` manifest records `method_status = "experimental"`; the runner arms support 3-D RHF/RKS/UHF/UKS. See the [line's page](../aiccm2026dev_a.md) and the [AICCM page](../user_guide/aiccm.md) | Partial |
| **AICCM, χ-CCM / aiccm2026dev-b** | periodic | `run_periodic_job(method="aiccm", variant="chi")` (legacy `jk_method="aiccm2026dev-b"` warns); restricted/unrestricted χ-CCM APIs | finite-character (Γ-centred character-mesh) CCM with independent finite-torus RHF/RKS/UHF/UKS in 3D via four-center, RI, and RIJCOSX; every SCF driver emits `AICCM2026DevBExperimentalWarning` and the `.system` manifest records `method_status = "experimental"`; all 1D/2D absolute-energy routes fail closed pending a shared wire/slab Coulomb convention; restricted and unrestricted real-torus correlation in 3D; occupied [localization](../tutorial/aiccm2026dev_b_localization.md) (`AICCM2026DevBLocalizationWarning`) and [space-group diagnostics](aiccm2026dev_b_symmetry.md) are χ-CCM-only; symmetry integral reduction, mixed-boundary Green functions, minimum-image local screens, and production reduced-scaling CC remain open; see its [separate page](../user_guide/aiccm2026dev_b.md) | Partial |
| **Real-Γ atomic relaxation** | periodic | `run_periodic_job(method="aiccm", variant="real-gamma", optimize=True)` (fixed cell) | the runner's refusal text for every other route labels the real-Γ direct-torus optimisation "EXPERIMENTAL, parity-verified per run"; BIPOLE and the captured analytic-gradient GDF route are the only other Hamiltonians `optimize=True` is wired to | Partial |
| **MANTA periodic HF and KS** | periodic | `run_periodic_job(integral_engine=IntegralEngineOptions(backend="manta"), coulomb=..., exchange=..., periodic_integrals=PeriodicIntegralOptions(...))` | **experimental**: neutral all-electron 3-D RHF/UHF/RKS/UKS on explicit finite reciprocal, pair and nuclear domains (plus `periodic_xc` for KS and `periodic_fitting` for RI), full regular meshes, BvK Ewald exchange correction; requires `use_diis=False` and an HCORE guess. No automatic accuracy profile: on H2/STO-3G in a 6 bohr cube the energy reaches the GDF value only as all domains grow (0.15 mHa at the largest tested on (2, 2, 2)). See [integral engines](../user_guide/integral_engines.md#periodic-jobs) | No |
| **Periodic SKALA-1.1** | periodic DFT | `run_periodic_job(..., functional="skala-1.1")` | **experimental**, energies only; no warning class, the periodic dispersion refusal text calls it "bare experimental periodic SKALA" (dispersion with periodic SKALA raises). See [periodic SKALA](../user_guide/skala.md#periodic-skala-experimental) | No |

### Molecular routes

| Feature | Area | Opt in | Status and key caveat | Quantitative |
|---|---|---|---|---|
| **MANTA integral engine and J/K selectors** | integrals, SCF | `run_job(integral_engine=IntegralEngineOptions(backend="manta"), coulomb=CoulombBuildOptions(...), exchange=ExchangeBuildOptions(...))` | **experimental**: RHF/UHF/RKS/UKS gas-phase single points (and RHF/UHF `geom_opt`) with exact, RI, BUPO or RI-BUPO J and exact or RI K; the exact build reproduces libint to 1e-10 Ha, but the route refuses Hessians, response, solvation, dispersion and correlation, and is slower than libint. No warning class; the `.out` and QVF record the executed provider. See [integral engines](../user_guide/integral_engines.md) | Yes for exact J/K; RI and BUPO carry their fitting and multipole errors |
| **SKALA-1.1** neural XC | DFT | `run_job(method="rks", functional="skala-1.1")` with the `[skala]` extra | the whole route is **experimental**: energies only at a fixed geometry; forces, Hessians, TDDFT and response fail closed. The molecular envelope is documented in [SKALA](../user_guide/skala.md); no warning class | No |
| **COSMO-RS / COSMOSPACE thermodynamics** | solvation | `run_job(..., cosmors=CosmoRSOptions(solvent="water"))` (solvents by name from `vibeqc.solvation.cosmors.available_solvents()`, with a per-user surface cache) or the library `vibeqc.solvation.cosmors` on a `ConductorSurface`; `CosmoRSOptions(density="direct")` for the Direct COSMO-RS density and `activity_coefficient=True` for ln gamma at infinite dilution; Direct COSMO-RS through `SolventModel(variant="dcosmo-rs", ...)`, with an analytic nuclear gradient | **experimental**: the `klamt1998` constants were fitted on DMol/BPW91/DNP and are transferred, not refitted, to vibe-qc's cavity and methods (#558). On the eight pure-liquid vapour pressures of Klamt 1995 Table 2 at the recommended BPW91/def2-TZVP protocol: rms 1.76 kcal/mol with a mean of -1.43 on the conductor density, and rms 2.93 on the Direct COSMO-RS density (`density="direct"`), which overpolarizes water, methanol and ethanol by 1.5 to 3.2 kcal/mol. `run_job(cosmors=...)` and every Direct COSMO-RS SCF emit `CosmoRSExperimentalWarning`, and the `.system` manifest records `method_status = "experimental"` with `experimental_component`. CPCM/COSMO energies and gradients themselves are production. See [COSMO-RS solvation free energies](../user_guide/solvation.md#cosmo-rs-solvation-free-energies) | No |
| **Jacobi-Davidson, harmonic-Davidson (GPLHR) and Lanczos eigensolvers** | SCF diagonalisation | `vibeqc.solvers.operators.solve_operator(..., solver="jd" \| "harmonic_davidson" \| "lanczos")` (`"gplhr"` is the historical name of `"harmonic_davidson"`); `run_job(solver=)` accepts only `"dense"` and `"davidson"` and refuses `"lobpcg"` | **experimental** (`solver_capabilities(name).experimental` is `True`): JD and harmonic-Davidson target interior roots near a shift but cannot certify the selection, and harmonic-Davidson raises on stagnation. No warning class. Dense, Davidson, Hermitian Davidson and LOBPCG (explicit real problems) are production. See the [eigensolver tutorial](../tutorial/eigensolver_operators.md) | No |
| **DLPNO local density fitting** | correlation | `DLPNOMP2Options(local_df=True, fit_buffer=...)` on `run_job(method="dlpno-mp2")` | **experimental** reduced-scaling increment: each pair is refit in the auxiliary functions near its domain instead of the global RI metric; a full fit domain reproduces the global RI exactly; default `local_df=False`; no warning class. See [DLPNO-MP2](../user_guide/dlpno_mp2.md#local-density-fitting-experimental-reduced-scaling) | Partial |
| **DLPNO-UCCSD O(N^6) pilot** | correlation | pass `DLPNOUCCSDPilotOptions` as the DLPNO options of an open-shell `run_job(method="dlpno-ccsd" \| "dlpno-ccsd(t)")`, or call `vibeqc.dlpno.run_dlpno_uccsd_pilot` | dense full-domain correctness pilot, opt-in only; the production open-shell path is the reduced-scaling local solver; no warning class | Yes, as a small-system oracle |
| **SC-CASPT2** | multireference | `CASPT2Options(variant="sc")` with `VIBEQC_EXPERIMENTAL_MULTIREF=1` in the environment | **gated** behind the environment variable; without it the route raises `RuntimeError`. The strongly-contracted implementation is superseded by IC-CASPT2 and not separately validated; IC-CASPT2 and NEVPT2 energies are production | No |
| **CASSCF `compute_wz=True`** | multireference | `CASSCFOptions(compute_wz=True)` | a retired experimental spelling, not an experimental route: the former W^z correction was removed for producing spurious components (#516); the flag is now a no-op alias of the analytic gradient and emits a `FutureWarning`; `compute_wz="numerical"` gives a finite-difference cross-check | Yes (the analytic gradient) |
| **DMRG** | exact solvers | `run_job(method="dmrg", active_space=...)` or `vibeqc.solvers` | "EXPERIMENTAL / TOY" in the source: dense full-Fock-space two-site DMRG, at most 6 spatial orbitals (12 spin orbitals), raises beyond; cross-checked against FCI; no warning class. See [non-HF solvers](../user_guide/non_hf_solvers.md) | No |
| **Embedded-cluster CAS driver** | multireference, periodic | `from vibeqc import embed` (deliberately not re-exported at the `vibeqc` top level) | docstring "EXPERIMENTAL / RESEARCH PROVENANCE. NOT VALIDATED FOR PRODUCTION": landed to preserve work when its branch was retired; audited defects (formal charges instead of the periodic density, mis-enumerated translated cells, missing AIMP projector terms, no NEVPT2) are unfixed | No |
| **D4 dispersion, native backend** | dispersion | `compute_d4(..., backend="native")` | parity-validated for H, He, B, C, N, O, F, Ne and raises outside that catalogue. `D4NativeExperimentalWarning` is defined but the backend never emits it today. The default `backend="dftd4"` (optional package) remains the full-periodic-table route | Yes, in the supported regime |
| **aiccm-wannier relocalisation** | analysis | the `vibe-qc-relocalize` worker with `method="aiccm-wannier"` and `allow_experimental: true` | the request fails closed without the opt-in and every result carries `experimental: true`; full unshifted cyclic mesh, restricted singlet, isolated occupied manifold only. See the [relocalization worker](../relocalization_worker.md) | No |

### Semiempirical routes

| Feature | Area | Opt in | Status and key caveat | Quantitative |
|---|---|---|---|---|
| **GFN2-xTB** | semiempirical | `run_job(method="gfn2_xtb")` | emits `GFN2ExperimentalWarning`; not quantitative; external xtb parity and periodic AES/molecular-limit terms are production gates. Post-SCF native D4 is validated only for H, He, B, C, N, O, F, Ne; outside that set GFN2 returns zero D4 with `GFN2D4UnsupportedWarning` | No |
| **PM6 and periodic PM6** | semiempirical | `run_job(method="pm6")`; `run_periodic_job(method="pm6")` (Gamma) | **experimental**, no warning class; the status registry (`vibeqc.semiempirical.status`) records that H-only and H-heavy s/p interactions are source-correct while full heavy-heavy two-centre tensor parity remains open, gradients are native finite differences, and periodic PM6 Gamma energy with finite-difference gradient and stress "remains experimental pending broader periodic validation". See [PM6 periodic](../user_guide/semiempirical.md#pm6-periodic-and-the-pm7omx-gates) | No |
| **OM1, OM2, OM3** | semiempirical | `run_job(method="om1" \| "om2" \| "om3")` | OM1 emits `NDDOExperimentalWarning`: the analytic core-valence ECP is not implemented, so bonds to heavy atoms come out roughly 0.3 angstrom short. OM2/OM3 do not warn, run through native NDDO kernels with published relative-energy fixtures, and are a validated development/prescreening surface with open external parity. All three are **experimental and a maintainer retirement target (2026-09-02)**, still executable; Bloch-periodic OMx is gated | No |
| **Full-k DFTB0 / SCC-DFTB** | semiempirical | `run_periodic_job(method="dftb0" \| "scc_dftb", kpoints=...)` | **gated experimental** in `vibeqc.semiempirical.status`: single-point energies and bands on the native complex Bloch prototype with weighted occupations; gradients and stress by one native batched finite-difference pass; public analytic derivatives and external DFTB+ parity remain gated. Gamma DFTB0 / SCC-DFTB stays preview. See [DFTB periodic](../user_guide/semiempirical.md#dftb-periodic) | No |
| **SECCM adapters: DFTB0, SCC-DFTB, PM6, OM2/OM3, GFN2** | semiempirical | library only: `run_dftb0_seccm`, `run_scc_dftb_seccm`, `run_pm6_seccm`, `run_omx_seccm`, `run_gfn2_seccm` in `vibeqc.semiempirical`; the runner's `method="seccm"` is the MSINDO route | **gated experimental** in `vibeqc.semiempirical.status`: neutral closed-shell insulating cyclic clusters over frozen Wigner-Seitz topologies; DFTB0-SECCM is scoped to an explicit repulsive-pair table (H, C, N, O, F, P, S, Cl); PM6-SECCM is energy only with finite-difference gradients; OM2/OM3-SECCM cyclic images need an explicit truncated-electrostatics acknowledgement and OM1-SECCM fails closed. MSINDO-SECCM is the mature route with validated closed-shell energies within scope. See [SECCM](../user_guide/semiempirical.md#dftb0-seccm-experimental) | No |

The periodic GPW/GAPW family is documented in full in [GPW and GAPW](../user_guide/gapw.md)
and the [GAPW design note](../design_periodic_gapw.md). The native-D4-backend
status, the BIPOLE gradient preview, and the GFN2-xTB / OM1 limits are in
[troubleshooting](../troubleshooting.md); the semiempirical models also have a
[user-guide page](../user_guide/semiempirical.md).

The χ-CCM research interfaces have separate pages for
[occupied localization](../tutorial/aiccm2026dev_b_localization.md),
[PAO/PNO local correlation](aiccm2026dev_b_pno.md), and
[space-group diagnostics](aiccm2026dev_b_symmetry.md).

The full AICCM documentation suite covers both lines:
[quickstart](../tutorial/aiccm_quickstart.md),
[troubleshooting](aiccm2026dev_troubleshooting.md),
[basis sets](aiccm2026dev_basis.md),
[cross-stream comparison](aiccm2026dev_compare.md),
[open-shell](aiccm2026dev_open_shell.md),
[visualization](aiccm2026dev_viz.md), and the
[documentation map](index.md#aiccm-documentation-map).

## Keeping competing versions on purpose

Some experimental features exist in more than one version on purpose, because
the right approach is itself an open research question and we want to measure
which one wins before committing to it. These are not duplicates to be merged
away; all are maintained until the comparison is settled.

The active example is the **ab-initio CCM (AICCM)**. Four selectable
formulations are kept visible while two development streams remain
independent:

- **`method="union12"`** (the library default of `run_ccm_rhf`), the
  historical product weight from Peintinger and Bredow 2014 (eq. 18) on the
  four-centre route. Validated in 1-D **in a minimal basis only**, but it
  breaks the 8-fold permutational symmetry of the electron-repulsion
  integrals in two and three dimensions.
- **`method="aiccm2026dev-a"`** on the library, **`variant="four-center"`**
  on the runner: the symmetrised union-and-weight/Wigner-Seitz four-centre
  weighting, the 2014 lineage kept in the Γ-CCM development line inside
  `vibeqc.periodic.ccm`.
- **Γ-CCM proper**, **`variant="real-gamma"`** and
  **`variant="neutral-bloch"`**: the neutral finite-torus construction in its
  real-Γ supercell and Bloch representations (`vibeqc.periodic.ccm.direct` and
  `vibeqc.periodic.ccm.ri`). They are one Hamiltonian in two representations
  and agree to about 1e-13 Ha per cell on H2/STO-3G through the front door;
  the bare spellings `gamma`, `gamma-ccm` and `gamma_ccm` fail closed because
  one word cannot pick a producer (ruling R1, D124).
- **`variant="chi"`** (legacy `jk_method="aiccm2026dev-b"`), χ-CCM, the
  finite-character (Γ-centred character-mesh) CCM evaluation in
  `vibeqc.periodic.chi`. It periodizes the declared finite-torus Hamiltonian
  first, records `coulomb_kernel="3d-periodic-g0"` and
  `exchange_q0="bvk-ewald"`, and uses Wigner-Seitz ties only to choose
  equivalent representatives; it does not import the Γ-CCM core.

Archived pre-guard 1D B tests are defect evidence only; current χ-CCM-B
absolute energies fail closed in every 1D/2D backend. The Γ-CCM line and
χ-CCM are developed independently on purpose: if they converge to the same
answer under a matched Coulomb gauge, that is evidence for the common limit;
if they diverge, the tensor symmetries and scalar energy derivatives decide
which construction is admissible. The duplication is the experiment, not an
oversight.

The fleet study uses one shared geometry registry but separate executable
drivers. Its current cross-approach status is `not-defined`. See the
[χ-CCM fleet inputs](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README_B.md) for
the complete RHF/RKS backend matrix and the 3D post-HF inputs.

## Internal and research-only paths (not for users)

A few experimental code paths are research scaffolding rather than
user-facing features. They have no `run_job` / `run_periodic_job` entry, or
are superseded, so they live in the roadmap rather than here: the GAPW
orbital-transformation (OT) direct-minimisation solver, the BIPOLE
legacy-gauge diagonal-Z gradient preview, and omega-B97M(2) (the functional
exists in libxc but has no user alias yet). Density mixers no longer belong in
this list: `run_periodic_job(density_mixer="anderson" | "broyden")`, with
optional Kerker preconditioning, ships on the closed-shell multi-k GDF route
and on the multi-k RIJCOSX route; every other route fails closed instead of
silently changing its SCF update scheme.

## Silencing an experimental warning

The warnings exist so an experimental path never runs unnoticed. Once you
have read the caveat and accept it, silence it the standard Python way by
filtering on the feature's warning class (for example `GAPWExperimentalWarning`,
importable from the module that raises it):

```python
import warnings
warnings.filterwarnings("ignore", category=GAPWExperimentalWarning)
```

Some constructors also take a `warn=False` argument, for example
`GFN2Model(mol, warn=False)`. Routes without a warning class have nothing to
silence; their label stays in the `.out` route line or the `.system`
manifest.

## Adding to this page

This is the designated home for experimental-feature status. When you ship a
feature behind an `ExperimentalWarning` (or a subclass), or label a route
experimental through the `.out` route line, the `.system` manifest, the
semiempirical status registry or a source docstring, add a row to the
catalog with its opt-in and its one key caveat, and link to its detailed
page. A feature that is labelled experimental at runtime but is invisible
here is exactly the gap this page exists to close.
