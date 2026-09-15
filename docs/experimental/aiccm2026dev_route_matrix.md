---
myst:
  html_meta:
    "description": "Which methods each AICCM route supports: HF, KS, double hybrids, canonical post-HF, DLPNO local correlation, and analytic gradients, across the Gamma-CCM routes and the chi-CCM backends."
    "og:title": "vibe-qc - AICCM route support matrix"
---

# AICCM route support matrix

One page answering "can this route run that method?" across both AICCM
lines. It is a capability index, not a tutorial: each cell names the entry
point and links to the reference page that explains it.

The information here is otherwise spread across four module docstrings and
the `run_periodic_job` guard block, so treat this page as a map and the
linked references as normative if the two ever disagree.

```{important}
Everything on this page is experimental. Every AICCM SCF driver emits
`AICCM2026DevAExperimentalWarning` or `AICCM2026DevBExperimentalWarning`,
and correlation, gradient, and property drivers inherit the warning through
the SCF they run. Route coverage is not a quantitative-status claim: see the
qualification warnings in the
[chi-CCM user guide](../user_guide/aiccm2026dev_b.md) and the neutral-control
caveats in the [Gamma-CCM reference](../aiccm2026dev_a.md).
```

## The two lines, and what "route" means in each

The lines are independent implementations of the variational finite-BvK-torus
CCM family, and they use the word differently.

| line | selector | a "route" is | where it lives |
|---|---|---|---|
| Gamma-CCM, `aiccm2026dev-a` | library `run_ccm_scf(ccm, route=...)`; runner `run_periodic_job(method="aiccm", variant=...)` for `"real-gamma"`, `"neutral-bloch"`, or `"four-center"` | a **construction plus representation**: which finite Hamiltonian is built, and in which basis it is diagonalized | `vibeqc.periodic.ccm` |
| chi-CCM, `aiccm2026dev-b` | `run_periodic_job(method="aiccm", variant="chi", ...)`, `aiccm_backend=...` (legacy `jk_method="aiccm2026dev-b"` warns) | an **electron-repulsion backend** for one fixed finite-character Hamiltonian | `vibeqc.periodic.chi` |

So the three Gamma-CCM routes can differ in their answers by construction,
while the three chi-CCM backends are three ways to evaluate the same
functional. Do not read the two route columns as parallel.

### Gamma-CCM route identity

`CCM_ROUTES = ("four-center", "neutral-bloch", "real-gamma")`.

| route | what it builds | aliases |
|---|---|---|
| `four-center` | the 2014 union-and-weight / Wigner-Seitz lineage, diagnosed non-variational in Paper 1 | `4c`, `aiccm-hf`, `aiccm2026dev-a`, `bare` |
| `neutral-bloch` | the neutral fitted-torus construction, multi-k GDF representation | `gdf`, `gdf-control`, `bloch-control`, `aiccm-ri` |
| `real-gamma` | the same neutral construction, k-free real-Gamma supercell representation | `direct`, `direct-torus`, `non-k`, `aiccm-hf-direct` |

The paper-facing spellings `gamma`, `gamma-ccm`, `gamma_ccm` fail closed
(ruling R1): Paper-1 Gamma-CCM is the neutral construction, which has two
admissible producers, so a single keyword cannot choose between them. Full
discussion in [Gamma-CCM reference](../aiccm2026dev_a.md).

## Energy method coverage

Rows are routes, columns are method families. Entries name the driver.

| | `four-center` | `neutral-bloch` | `real-gamma` | chi-CCM (all three backends) |
|---|---|---|---|---|
| **RHF / UHF** | `run_ccm_rhf`, `run_ccm_rhf_scalable`, `run_ccm_uhf` | `run_ccm_rhf_gdf`, `run_ccm_uhf_gdf`, `run_ccm_rhf_ri_neutral` | `run_ccm_rhf_direct`, `run_ccm_uhf_direct` | `run_aiccm2026dev_b_rhf`, `run_aiccm2026dev_b_uhf` |
| **RKS / UKS**, any libxc functional | `run_ccm_rks`, `run_ccm_uks` | `run_ccm_rks_gdf`, `run_ccm_uks_gdf` | `run_ccm_rks_direct`, `run_ccm_uks_direct` | `run_aiccm2026dev_b_rks`, `run_aiccm2026dev_b_uks` |
| **External full-grid XC**, including SKALA | RKS/UKS through `variant="four-center"` or the low-level four-centre WSSC drivers; 3-D, all-electron, zero temperature | **no**; fails closed pending proof that the atom-block periodic partition is representation invariant | RKS/UKS; 3-D, all-electron, zero temperature | RKS/UKS only with `backend="four_center"`; 3-D, all-electron, zero temperature |
| **Screened (HSE-class) hybrids** | no dedicated path, see below | yes, but only with `k_exchange="cosx"` | yes, experimental, on the direct KS drivers | yes, via the shared periodic exchange resolver |
| **Double hybrids** | no | no | `run_ccm_double_hybrid_direct`, closed shell only | no |
| **RIJCOSX variant** | `run_ccm_rhf_rijcosx`, bare research operator | no | `run_ccm_rhf_direct_rijcosx`, dim 3 | `backend="rijcosx"` |
| **Canonical MP2 / UMP2** | `run_ccm_mp2`, `run_ccm_ump2` with `method=` | `run_ccm_ri_mp2(reference="neutral")` | `run_ccm_ri_mp2(reference="direct")` | `run_aiccm2026dev_b_mp2`, `run_aiccm2026dev_b_ump2` |
| **Canonical CCSD(T)** | `run_ccm_ccsd` with `method=` | `run_ccm_ri_ccsd(reference="neutral")`, `run_ccm_uccsd` | `run_ccm_ri_ccsd(reference="direct")`, `run_ccm_uccsd` | `run_aiccm2026dev_b_ccsd`, `_ccsd_t`, `_uccsd`, `_uccsd_t` |
| **DLPNO local correlation** | **no** | yes | yes, with one-call wrappers | yes |
| **Analytic nuclear gradients** | yes, the complete set | see below | yes, composed | **no**, fails closed |

Three entries need reading carefully.

`run_ccm_mp2`, `run_ccm_ump2`, and `run_ccm_ccsd` are dual-mode: passing
`method=` runs them on the four-center weighted ERIs, passing `cderi=L` runs
them on the neutral fit. The route is set by which argument you pass, not by
the function name. `run_ccm_uccsd` is neutral-only and has no four-center mode.

Screened hybrids on the fitted routes need the chain-of-spheres exchange
backend. The Lpq-contracted GDF K is full-range only and fails closed on any
range-separated functional (`reject_unscreened_range_separated`), so a
screened hybrid must not silently run as its full-range twin. Pass
`k_exchange="cosx", use_compcell=True` through to the multi-k driver. Fully
range-separated functionals with `c_full > 0` (wb97x, cam-b3lyp, and
similar) fail closed on every backend.

The four-center KS driver has no dedicated screened-exchange path and no
range-separated guard of its own, so do not assume an HSE-class functional is
handled correctly there. Use a fitted route, or verify against a reference
before trusting the number.

### Cluster-size regime

Route coverage and reachable system size are different questions.

| path | regime |
|---|---|
| four-center SCF via `run_ccm_rhf_scalable` | reaches genuine 3-D |
| four-center dense ERI and all four-center post-HF and gradients | dense `n_pad_ao**4`, so small, 1-D, and feasibly 2-D clusters only |
| neutral RI post-HF (`run_ccm_ri_mp2`, `run_ccm_ri_ccsd`, the DLPNO stack) | density-fit throughout, no dense AO four-center, reaches moderate 3-D |
| chi-CCM | 3-D only; every 1-D and 2-D absolute-energy backend fails closed |

### The `union12` default

`run_ccm_rhf` and its siblings still default to `method="union12"`, the
historical eq-18 product weight. That weighting carries a negative subspace on
any basis with more than one function per centre, so its energies are
unbounded rather than merely inaccurate (issue #242, and the
`vibeqc.periodic.ccm` module docstring for the measured numbers). Prefer
`method="aiccm2026dev-a"`, which is what `run_ccm_scf(route="four-center")`
selects for you when `method` is left unset.

## Properties and analysis

Each line carries its own property surface, and the names do not overlap. The
Gamma-CCM functions take `(scf_result, ccm)` and are route-agnostic: they work
on a converged SCF from any of the three routes. The chi-CCM functions consume
a chi result and its finite-character convention.

| capability | Gamma-CCM (any route) | chi-CCM |
|---|---|---|
| HOMO-LUMO gap | `ccm_homo_lumo_gap` | via `derive_aiccm2026dev_b_scf_properties` |
| Mulliken / Loewdin charges | `ccm_mulliken_charges`, `ccm_lowdin_charges` | via `derive_aiccm2026dev_b_scf_properties` |
| Mayer bond orders | `ccm_mayer_bond_orders` | `aiccm2026dev_b_mayer_bond_orders` |
| band structure (folded torus spectrum) | `ccm_band_structure` | `aiccm2026dev_b_band_structure` |
| dipole | `ccm_dipole` | `aiccm2026dev_b_one_electron_expectation` |
| occupied localization | `localise_ccm` (delegates to `vibeqc.periodic_localise`) | `localize_aiccm2026dev_b_occupied`, `_blocks`, `_unrestricted_occupied` |
| symmetry analysis | `analyze_ccm_symmetry`, `ccm_symmetry_invariance_residuals` | `build_aiccm2026dev_b_symmetry_plan`, `shell_pair_orbits`, `shell_quartet_orbits` |
| numerical (finite-difference) gradient | `ccm_numerical_gradient` | not exposed |
| QVF archive | `write_ccm_periodic_qvf` | via the runner's QVF vendor section |
| COOP / COHP | through the generic periodic path | **refused**: the generic path rebuilds a fixed-cutoff Ewald surrogate rather than the converged finite-character Hamiltonian |

`ccm_numerical_gradient` is the finite-difference gate the analytic Gamma-CCM
gradients are validated against. It displaces each unit-cell atom and rebuilds
the whole `CCMSystem`, so every periodic image moves together, at a cost of
`6 * n_basis_atoms` SCF evaluations. It is a validation tool, not a production
force.

## DLPNO: why `four-center` is excluded

DLPNO on this code base is not a truncation switch you can point at any
reference. The pipeline density-fits its `(ia|jb)` integrals from the
**neutral** cderi, because the ionic Madelung background shifts the
occupied-virtual denominators, so the correlation reference has to be the
neutral fitted torus rather than the bare 1/r four-center.

| route | DLPNO entry points |
|---|---|
| `four-center` | none |
| `neutral-bloch` | `ccm_dlpno_mp2`, `ccm_dlpno_ump2`, `ccm_dlpno_ccsd`, `ccm_dlpno_ccsd_coupled`, `ccm_dlpno_uccsd` |
| `real-gamma` | the same five, plus one-call `run_ccm_dlpno_mp2_direct`, `run_ccm_dlpno_ump2_direct`, `run_ccm_dlpno_ccsd_direct` |
| chi-CCM | `run_aiccm2026dev_b_dlpno_mp2`, `_dlpno_ump2`, `_dlpno_ccsd`, `_dlpno_ccsd_t`, `_dlpno_uccsd`, `_dlpno_uccsd_t` |

```{warning}
`ccm_dlpno_mp2` and its siblings only **document** the neutral-reference
requirement; there is no runtime guard. Handing one a four-center
`scf_result` produces a number rather than an error, and that number is
silently wrong. The `*_direct` one-call wrappers exist precisely to remove
this footgun: they guarantee the SCF reference and the cderi are the same
kernel. Prefer them.
```

At the default zero truncations (`tcut_pno = tcut_mkn = 0`) every DLPNO
driver reproduces the canonical correlation on the same reference, which is
the correctness gate rather than a production setting.

## Analytic gradients

Three genuinely different situations, which the single word "supported" would
hide.

| route | status | entry points | scope |
|---|---|---|---|
| `four-center` | **derived**: the CCM gradient is differentiated term by term | `run_ccm_rhf_gradient`, `run_ccm_uhf_gradient`, `run_ccm_rks_gradient`, `run_ccm_uks_gradient`, `run_ccm_mp2_gradient`, `run_ccm_ump2_gradient`, `run_ccm_ccsd_gradient`, `run_ccm_uccsd_gradient` | requires `method="aiccm2026dev-a"` (`union12` raises); dense `n_pad_ao**4`, so small, 1-D, 2-D only |
| `real-gamma` | **composed**: rides the production multi-k GDF analytic gradient through the representation identity, with a per-run numerical parity gate | `run_ccm_direct_gradient`, `run_ccm_direct_optimize` | dim 3, `exxdiv="ewald"`, pure and global hybrids, closed and open shell; screened hybrids and double hybrids fail closed |
| `neutral-bloch` | no gradient entry in the CCM namespace | the underlying `run_k{r,u}h{f,s}_periodic_gdf(compute_gradient=True)` has one | as for the multi-k GDF driver |
| chi-CCM | **fails closed by design** | `compute_aiccm2026dev_b_gradient` raises; `aiccm2026dev_b_gradient_status(result)` reports | see below |

The four-center set is the only place where an AICCM gradient is derived from
the CCM energy expression itself. Derivation and validation are in
[the analytic gradient note](../aiccm2026dev_a_analytic_gradient.md).

`run_ccm_direct_gradient` is worth understanding before you trust a force from
it. It converges the direct-torus SCF, runs the multi-k GDF control on the same
cell and mesh, and **verifies the premise numerically**: the two energies must
agree to `parity_tol` per cell, otherwise it fails closed rather than returning
forces belonging to a slightly different Hamiltonian. `run_ccm_direct_optimize`
relaxes unit-cell positions on that surface at fixed lattice, checking parity at
the endpoints by default.

chi-CCM returns no analytic force at all. It is not an oversight:
`aiccm2026dev_b_gradient_status(result)` records the finite-torus convention
alongside an explicit list of implemented component derivatives (the 3-D Ewald
nuclear term, fixed-density kinetic, energy-weighted overlap Pulay, and the
BvK exchange-q=0 seam) and the terms that remain blocked. Neither the
union-and-weight WSSC gradient nor the neutral-control gradient is substituted,
because each differentiates a different finite Hamiltonian.

## What computes through `run_periodic_job`

Route availability in the library and in the unified runner are not the same
thing.

| selector | resolves to | methods | notable refusals |
|---|---|---|---|
| `method="aiccm", variant="chi"` (legacy `jk_method="aiccm2026dev-b"`, `"chi"`, `"chi-ccm"`: `DeprecationWarning`) | chi-CCM | RHF, RKS, UHF, UKS | `exchange_q0` pinned to `bvk-ewald`; no DFT+U, COOP/COHP, `optimize`, or `hessian`; external full-grid XC is 3-D, all-electron, zero-temperature and `aiccm_backend="four_center"` only |
| `method="aiccm", variant="real-gamma"` (legacy `jk_method="real-gamma"`, `"real_gamma"`: `DeprecationWarning`) | the neutral real-Gamma producer | RHF, RKS, UHF, UKS | dim 3 only; no DFT+U, smearing, `hessian`, or cell relaxation; explicit `damping` / `fock_mixing` / `fmixing_percent` / `density_mixer` / `level_shift` fail closed (the loop would drop them). External full-grid XC additionally requires an all-electron basis. `optimize=True` with `optimize_cell=False` works for ordinary functionals; external-XC gradients remain gated |
| `method="aiccm", variant="neutral-bloch"` | the neutral Bloch producer (`periodic/ccm/ri.py` through `periodic/ccm/neutral_bloch_runner.py`) | RHF, RKS, UHF, UKS | dim 3 only; the Gamma-centred mesh is the torus; no DFT+U, `optimize`, `hessian`, restart, `symmetry_reduce_k` or alternative BZ integration; external full-grid XC fails closed pending representation-invariance evidence; `exchange_exxdiv` fixed to ewald; explicit `damping` / `fock_mixing` / `fmixing_percent` / `density_mixer` / `level_shift` fail closed; the producer additionally refuses a vacuum-padded cluster and a positive converged neutral energy (IID 291). The former `jk_method` aliases `"neutral-bloch"`, `"aiccm-ri"`, `"gdf-control"`, `"bloch-control"` (plain unit-cell GDF) stay retired (D-2b) |
| `method="aiccm", variant="four-center"` (legacy `jk_method="aiccm2026dev-a"`: `DeprecationWarning`) | the union-and-weight / Wigner-Seitz four-centre lineage through `periodic/ccm/four_center_runner.py` | RHF, RKS, UHF, UKS | dim 3 only; the Gamma-centred mesh is the torus; no DFT+U, smearing, gradients, optimization, Hessians, or explicit mixing controls. External full-grid XC is all-electron, zero-temperature, and pure-functional only |
| `jk_method="gamma"`, `"gamma-ccm"`, `"gamma_ccm"` | nothing | **fails closed** | ruling R1 (D124): one word cannot pick a neutral producer and must never select the four-centre lineage; name the variant |

Post-HF is library-only on every line. With `method="aiccm"` the SCF
reference is inferred (`functional` decides HF versus KS, the multiplicity
and electron parity decide restricted versus unrestricted;
`scf_reference="rohf" | "roks"` is explicit and refused by every variant
today), so MP2, CCSD(T), and DLPNO are reached through the drivers tabulated
above, not through `run_periodic_job`.

The A-prefixed spellings for the real-Gamma control (`aiccm2026dev-a-real-gamma`,
`aiccm2026dev-a-direct`) are rejected deliberately: the `aiccm2026dev-a` prefix
denotes the union-and-weight construction, while real-Gamma is a producer of the
neutral construction. The bare `gamma` spellings and the four GDF aliases are
retired for the complementary reason: a word must not mean two Hamiltonians.

## See also

- [Gamma-CCM reference](../aiccm2026dev_a.md), the per-driver method stack and
  the route-keyword semantics.
- [chi-CCM user guide](../user_guide/aiccm2026dev_b.md), backends, qualification
  status, and the worked examples.
- [Open-shell AICCM](aiccm2026dev_open_shell.md), the unrestricted APIs across
  both lines.
- [Comparing Gamma-CCM and chi-CCM](aiccm2026dev_compare.md), why the two lines
  stay separate and what a cross-approach comparison would require.
- [Analytic gradient note](../aiccm2026dev_a_analytic_gradient.md), the
  four-center force derivation.
