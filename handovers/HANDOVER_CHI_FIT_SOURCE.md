# Prompt F: explicit chi fit source

The explicit classic profile meets the measured SCF energy/Fock and MP2
representation checks on the requested classic cells, using the tighter LiH
222 follow-up where needed. The retained SR H2 check passes; LiH SR remains
not measured after its deadline, so the complete SR acceptance matrix is open.

Work on current main `06173cc4e78698090674fad83cea9df457903eaf`, on the local
`codex/chi-fit-source` branch, in an independent clone with its own editable
install and rebuilt native core. No push. The reference campaign and paper
repositories were read-only. D56/D74 firewall remains intact. No energy
comparison between CCM constructions was calculated or used.

## Result and API

The front door accepts `run_periodic_job(method="aiccm", variant="chi",
aiccm_backend="ri", fit_source="classic", ...)`. The shared RSGDF selector
accepts exactly `auto`, `classic`, and `sr`; explicit choices require 3D RSGDF.
`None`, invalid values, incompatible backends, private fit adapters, and
unsupported explicit-classic gradients fail closed. Omitting the selector
preserves existing defaults. All chi RHF/RKS/UHF/UKS gateways transport it.

Explicit classic SCF retains canonical modrho auxiliary factors and post-HF
borrows the same identified source, validating support, mesh, engine, frame
and factor hash. The independent classic pair builders remain unchanged as
oracles. Historical auto post-HF still builds its independent classic
correlation source; that executed source is now distinguished from its SR
reference and from a streamed occupied-virtual projection. Explicit SR
post-HF is refused until a canonical SR frame adapter is implemented.

The frozen executed descriptor identifies the engine/version, orbital and
auxiliary fingerprints, actual numerical support, metric threshold, auxiliary
frame, canonical q groups, base reciprocal cutoff and actual factors. An
independent frozen request retains requested source/base/tail. Results and
`.system` distinguish requested and executed tails; mesh arrays remain arrays.
The loaded-process triple is present on chi results and runtime `.system`
records. Admission consults the selected engine, so an inactive classic
Gamma auto-tail cannot reject SR execution.

D93 is permanent in v1: every fitted and post-HF chi result remains
**not-qualified**. Four-center eligibility is not qualification. This change
does not validate SR theory, change fit screening or the one-electron resolver.
Exchange q=0/Madelung sign, hybrid coefficients and primitive-cell factors are
preserved. Classic and SR have separate baselines; conversion between them is
not claimed to be an identity.

## Measurement identities

Every numerical/timing row below is associated with a process cohort. The
cohort denotes the complete imported release/source/native triple, not a
release pin. `common.provenance()` supplied each raw process block.

| Cohort | Imported release | Source revision | Loaded native SHA-256 |
|---|---|---|---|
| A | 0.17.7.dev0 | e6c3cb2c6104864d6ca5209b6635c757dfd873c3 | 65a66af187f67dd6889ca9ff728663c910ab551a12b914f118c9d8135d832b5b |
| B | 0.17.7.dev0 | b09d6f64f211d47c9e79084ff89726190554f39a | 65a66af187f67dd6889ca9ff728663c910ab551a12b914f118c9d8135d832b5b |

The unchanged baseline engine is the saved main file at
`06173cc4e78698090674fad83cea9df457903eaf`. Classic baseline payloads replace
only the in-process `bulk_sr` selector with false. They retain the current
one-electron implementation. Its exact original and executed payload digests
are in each baseline record. The process source above remains the source of
the imported package. The cold baseline also retains the independent classic
post-HF builder. No old one-electron cutoff control was run in this campaign.

Geometries and inputs are in each exact payload: H2/cube6 has a 6-bohr cubic
cell and H coordinates (3,3,2.3)/(3,3,3.7); LiH and MgO are primitive rocksalt
cells at conventional lattice constants 4.0840 and 4.2239 angstrom. All use
STO-3G, the selected default auxiliary basis, base reciprocal cutoff 200 Ha,
fit lattice cutoff 15 bohr, metric threshold 1e-9 and unchanged resolved
one-electron support. Reference SCF energy tolerance is 1e-11 and gradient tolerance
1e-9. After the first LiH 222 rebuilt-Fock residual missed its criterion,
the LiH follow-up and pending MgO 222 runs used gradient tolerance 1e-10.

The following are same-source energy differences, after minus unchanged
baseline, in Ha per primitive cell. They are representation/implementation
checks, not absolute-energy or fitted-source qualification.

| Pair record prefix | Cohorts before/after | Before E | After E | After - before |
|---|---|---|---|---|
| h2_cube6-211-classic | A/A | -1.194530912196504 | -1.194530912196504 | 4.441e-16 |
| h2_cube6-211-sr | A/A | -1.194530912626963 | -1.194530912626963 | 0.000e+00 |
| lih-211-classic | A/A | -8.166356470664418 | -8.166356470664418 | 0.000e+00 |
| lih-211-sr | A/A | not measured | not measured | not measured |
| lih-221-classic | B/B | -8.005647832657075 | -8.005647832657083 | -7.105e-15 |
| lih-222-classic | B/B | -7.921949452290561 | -7.921949452290574 | -1.243e-14 |
| mgo-221-classic | B/B | -271.672068370509805 | -271.672068370511568 | -1.762e-12 |
| mgo-222-classic | B/B | -271.702626226188386 | -271.702626226171674 | 1.671e-11 |


Operator/orbital checks at the returned SCF orbital density use the selected fit and q=0 seam. Max norms are maximum absolute elements.

| After record | Cohort | Rebuilt F - stored F | Rebuilt commutator | FC - SCε | Real SCF - character E | Oracle random-D J-K/2 | Real - character MP2 |
|---|---|---|---|---|---|---|---|
| h2_cube6-211-classic | A | 2.776e-16 | not measured | not measured | -2.220e-16 | 4.990e-14 | -5.204e-18 |
| h2_cube6-211-sr | A | 2.220e-16 | 2.220e-16 | 2.776e-16 | -6.661e-16 | not measured | not measured |
| lih-211-classic | A | 9.042e-12 | not measured | not measured | 1.776e-15 | 5.718e-15 | -8.327e-17 |
| lih-221-classic | B | 8.203e-12 | 1.448e-11 | 2.540e-11 | 5.329e-15 | 9.326e-15 | -6.939e-17 |
| lih-222-classic | B | 1.343e-10 | 1.266e-10 | 1.363e-10 | 1.776e-15 | 1.172e-13 | -4.163e-17 |
| lih-222-classic-tight | B | 1.293e-12 | 2.542e-12 | 5.926e-12 | 0.000e+00 | 1.688e-14 | 3.123e-17 |
| mgo-221-classic | B | 6.598e-11 | 1.264e-10 | 7.089e-11 | 5.684e-14 | not measured | not measured |
| mgo-222-classic | B | 2.243e-12 | 4.374e-12 | 3.959e-12 | -5.684e-14 | not measured | -2.914e-16 |


The independently converged real-torus SCF begins from a recorded perturbed
density and uses the same selected fit with a full real eigensolve and
translation-projected density, with energy tolerance 1e-12 and maximum
commutator tolerance 1e-10. The correlation probe compares canonical
character, semicanonical, real-torus and degenerate-gauge MP2. Orbital-gauge
freedom is not tested by comparing individual coefficients. Full pair-builder
factor entries can differ more than the fitted operator; the random-density
operator check is the relevant canonical-frame conversion check.

The first LiH 222 pair meets the campaign SCF stopping criteria but its
1.34e-10 rebuilt-Fock residual (cohort B) fails the stricter operator criterion. It is
retained as such. The tighter pair is separate evidence. Its supplemental
oracle uses the saved unchanged reduced-frame shared-q builder and checks
same-q fitted Gram matrices; it does not relabel a missing independent
pair-builder measurement.


## Local and independent CC controls

Raw local and corrected returned MP2 are both recorded (Ha/cell). The zero-cut canonical correction is not evidence that the raw algorithm is identical.

| Record | Cohort | Consumer | Energy observations |
|---|---|---|---|
| h2_cube6-211-classic-after | A | canonical_character_mp2 | e_corr_per_cell=-0.009708718695624109 |
| h2_cube6-211-classic-after | A | dlpno_zero_none | e_corr_per_cell=-0.009708718695624114; raw_local_e_corr_per_cell=-0.009708718695624119; complete_space_correction_per_cell=5.204170427930421e-18 |
| h2_cube6-211-classic-after | A | dlpno_zero_wannier | e_corr_per_cell=-0.009708718695624114; raw_local_e_corr_per_cell=-0.00970871869562223; complete_space_correction_per_cell=-1.8839096949108125e-15 |
| lih-211-classic-after | A | canonical_character_mp2 | e_corr_per_cell=-0.01220609451695943 |
| lih-211-classic-after | A | dlpno_zero_none | e_corr_per_cell=-0.012206094516959514; raw_local_e_corr_per_cell=-0.01220609451695951; complete_space_correction_per_cell=-3.469446951953614e-18 |
| lih-211-classic-after | A | dlpno_zero_wannier | e_corr_per_cell=-0.012206094516959514; raw_local_e_corr_per_cell=-0.012206094516949196; complete_space_correction_per_cell=-1.0318135235110049e-14 |
| lih-221-classic-after | B | canonical_character_mp2 | e_corr_per_cell=-0.015338231024038161 |
| lih-221-classic-after | B | dlpno_zero_none | e_corr_per_cell=-0.015338231024038234; raw_local_e_corr_per_cell=-0.015338231024038224; complete_space_correction_per_cell=-1.0408340855860843e-17 |
| lih-221-classic-after | B | dlpno_zero_wannier | e_corr_per_cell=-0.01533823102403823; raw_local_e_corr_per_cell=-0.015338231024022746; complete_space_correction_per_cell=-1.548414174656898e-14 |
| lih-222-classic-after | B | canonical_character_mp2 | e_corr_per_cell=-0.01691106715711655 |
| lih-222-classic-after | B | dlpno_zero_none | e_corr_per_cell=-0.01691106715711659; raw_local_e_corr_per_cell=-0.01691106715711656; complete_space_correction_per_cell=-2.7755575615628914e-17 |
| lih-222-classic-after | B | dlpno_zero_wannier | e_corr_per_cell=-0.01691106715711659; raw_local_e_corr_per_cell=-0.01691106715714056; complete_space_correction_per_cell=2.397040899104752e-14 |


Canonical and local CC share a solver; the last row per cell is the independent full-space tensor control on the same chi factors.

| Record | Cohort | Solver path | CC correlation Ha/cell | Triples Ha/cell |
|---|---|---|---|---|
| h2_cube6-211-classic-after | A | public_canonical_ccsd_t | -0.011378869627449 | -0.000000000000000 |
| h2_cube6-211-classic-after | A | dlpno_zero_ccsd_t | -0.011378869627449 | -0.000000000000000 |
| h2_cube6-211-classic-after | A | full_space_tensor_ccsd_t | -0.011378869627446 | -0.000000000000000 |
| lih-211-classic-after | A | public_canonical_ccsd_t | -0.011577714006635 | -0.000041266190605 |
| lih-211-classic-after | A | dlpno_zero_ccsd_t | -0.011577714006635 | -0.000041266190605 |
| lih-211-classic-after | A | full_space_tensor_ccsd_t | -0.011577714006661 | -0.000041266190604 |


## Performance observations

Cold public classic MP2 also preserves its unchanged-path total energy
(Ha per primitive cell), including the current one-electron reference:

| Pair record prefix | Cohorts before/after | Before total E | After total E | After - before |
|---|---|---|---|---|
| h2_cube6-211-classic-cold-mp2 | A/A | -1.204239630892133 | -1.204239630892128 | 5.329e-15 |
| lih-211-classic-cold-mp2 | A/A | -8.178562565181377 | -8.178562565181377 | 0.000e+00 |

All walls are upper-bound observations on a loaded shared host. There is no
quiet-host speedup claim, and no timing comparison of classic versus SR is
used as an identity. Each row includes the one-minute load at the exact wall
boundaries; all three load averages are in the raw records. Every process
requested OpenMP 8 and one BLAS thread (including threadpoolctl inside outer
pools). Loaded runtime reports OpenMP 8 and native Accelerate BLAS. Its actual
BLAS worker count is **not measured**. At most two heavy jobs ran concurrently.
Explicit stage clocks are used; cProfile allocations are not timing oracles.

Named-cell total SCF and cold public classic MP2 walls:

| Record | Cohort | Wall seconds | Load start -> end |
|---|---|---|---|
| h2_cube6-211-classic-before | A | 4.639178 | 65.14 -> 65.05 |
| h2_cube6-211-classic-after | A | 6.478789 | 69.54 -> 69.74 |
| lih-211-classic-before | A | 122.400672 | 68.06 -> 79.43 |
| lih-211-classic-after | A | 122.239230 | 68.06 -> 80.59 |
| h2_cube6-211-sr-before | A | 112.020703 | 73.71 -> 81.44 |
| h2_cube6-211-sr-after | A | 111.169040 | 73.48 -> 84.38 |
| lih-211-sr-before | A | not measured | not measured |
| lih-211-sr-after | A | not measured | not measured |
| h2_cube6-211-classic-cold-mp2-before | A | 15.242351 | 79.94 -> 75.26 |
| h2_cube6-211-classic-cold-mp2-after | A | 5.447863 | 77.55 -> 76.94 |
| lih-211-classic-cold-mp2-before | A | 142.319755 | 74.85 -> 73.72 |
| lih-211-classic-cold-mp2-after | A | 113.892687 | 74.92 -> 75.76 |


Explicit named-cell SCF stages (seconds; load beside every wall):

| Record | Cohort | Stage | Wall seconds | Load start -> end |
|---|---|---|---|---|
| h2_cube6-211-classic-before | A | integrals_lattice | 1.730315 | 65.14 -> 65.14 |
| h2_cube6-211-classic-before | A | aux_basis | 0.013704 | 65.14 -> 65.14 |
| h2_cube6-211-classic-before | A | gdf_cderi | 2.630559 | 65.14 -> 65.05 |
| h2_cube6-211-classic-after | A | integrals_lattice | 1.791989 | 69.54 -> 69.54 |
| h2_cube6-211-classic-after | A | aux_basis | 0.015464 | 69.54 -> 69.54 |
| h2_cube6-211-classic-after | A | gdf_cderi | 3.981886 | 69.54 -> 69.74 |
| lih-211-classic-before | A | integrals_lattice | 103.158869 | 68.06 -> 77.30 |
| lih-211-classic-before | A | aux_basis | 0.138526 | 77.30 -> 77.30 |
| lih-211-classic-before | A | gdf_cderi | 18.640175 | 77.30 -> 79.43 |
| lih-211-classic-after | A | integrals_lattice | 103.340426 | 68.06 -> 76.79 |
| lih-211-classic-after | A | aux_basis | 0.026454 | 76.79 -> 76.79 |
| lih-211-classic-after | A | gdf_cderi | 18.223960 | 76.79 -> 80.59 |
| h2_cube6-211-sr-before | A | integrals_lattice | 1.902657 | 73.71 -> 74.21 |
| h2_cube6-211-sr-before | A | aux_basis | 0.126248 | 74.21 -> 74.21 |
| h2_cube6-211-sr-before | A | gdf_cderi | 109.645080 | 74.21 -> 81.44 |
| h2_cube6-211-sr-after | A | integrals_lattice | 2.379827 | 73.48 -> 71.60 |
| h2_cube6-211-sr-after | A | aux_basis | 0.009027 | 71.60 -> 71.60 |
| h2_cube6-211-sr-after | A | gdf_cderi | 107.996411 | 71.60 -> 84.76 |
| lih-211-sr-before | A | integrals_lattice | 97.158489 | 79.01 -> 78.10 |
| lih-211-sr-before | A | aux_basis | 0.017963 | 78.10 -> 78.10 |
| lih-211-sr-after | A | integrals_lattice | 93.706163 | 83.30 -> 77.09 |
| lih-211-sr-after | A | aux_basis | 0.122757 | 77.09 -> 77.09 |


Classic correlation factor reuse removes the repeated independent pair fit
from the explicit profile. These cold-public measurements support that narrow
performance claim. The named-cell SCF walls do not establish a general SCF
speedup; current one-electron work still dominates LiH. Native external stack
samples are retained for H2/LiH SR and MgO classic construction. No in-process
sampling thread was used.

## Coverage, files and continuation

All raw records are under `$VIBE_CHI_FIT_RECORDS/<record-name>/`. Each closed
record contains `results.json`, exact payloads and input specification,
`command.json`, stdout/stderr, all emitted `.system` files, `process.json` and
`files.sha256`. The controller owns each child PID, writes incremental stage
records and stops only those PIDs on its explicit deadline. Source/native
paths and private operational details remain outside the product repository.
`acceptance-analysis.json` and this handover contain interpretation separately
from raw observations. `baseline_gdf.py` and its digest make cold classic and
default SR paths separately reproducible on the current one-electron path.

### Tighter LiH 222 follow-up

| Cohorts before/after | Before E Ha/cell | After E Ha/cell | After - before Ha/cell | Same-q fitted Gram max |
|---|---|---|---|---|
| B/B | -7.921949452290562 | -7.921949452290573 | -1.066e-14 | 5.951e-14 |

### Supplemental MgO 221 representation

This separate process uses the selected canonical source and the saved unchanged reduced-frame shared-q oracle. The primary independent pair-builder attempts remain not measured.

| Cohort | SCF E Ha/cell | Rebuilt F max | Rebuilt commutator | FC - SCε | Same-q Gram max | Random-D operator max | Real - character MP2 Ha/cell |
|---|---|---|---|---|---|---|---|
| B | -271.672068370511568 | 6.598e-11 | 1.264e-10 | 7.089e-11 | 7.079e-13 | 3.725e-13 | -4.996e-16 |

### Front door and emitted runtime metadata

| Record | Cohort | Requested source | Executed source | Mesh is array | Mesh matches | Process triple matches | D93 label |
|---|---|---|---|---|---|---|---|
| h2_cube6-211-classic-frontdoor | B | classic | classic | True | True | True | not-qualified |
| lih-211-classic-frontdoor | B | classic | classic | True | True | True | not-qualified |
| h2_cube6-211-default-sr-frontdoor | B | auto | sr | True | True | True | not-qualified |
| h2_cube6-111-sr-tail-300.0 | B | sr | sr | True | True | True | not-qualified |

| Record | Cohort | Rebuilt F max | Rebuilt commutator | FC - SCε |
|---|---|---|---|---|
| h2_cube6-211-classic-frontdoor | B | 2.776e-16 | 2.220e-16 | 2.220e-16 |
| lih-211-classic-frontdoor | B | 9.042e-12 | 1.083e-11 | 2.342e-11 |

### Gamma admission

The no-tail and explicit-tail classic calls are distinct fits; no energy identity between them is claimed. SR preserves the requested classic-tail setting while executing no classic tail. Missing descriptors, invalid selectors, and incompatible backend/source pairs are refused by the T3 contracts.

| Record | Cohort | Requested tail Ha | Executed tail Ha | Executed source | Converged |
|---|---|---|---|---|---|
| h2_cube6-111-classic-tail-None | B | None | None | classic | True |
| h2_cube6-111-classic-tail-300.0 | B | 300.0 | 300.0 | classic | True |
| h2_cube6-111-sr-tail-None | B | None | None | sr | True |
| h2_cube6-111-sr-tail-300.0 | B | 300.0 | None | sr | True |

### Spin and hybrid transport

Closed-shell UHF/UKS collapse and explicit exchange assembly are checked on the classic H2 profile. Full per-spin orbital residuals and source descriptors remain in the raw record.

| Record | Cohort | U - R energy Ha/cell | Alpha F - R F | Beta F - R F | Alpha commutator | Beta commutator | Exchange energy residual Ha/cell | Hybrid coefficient |
|---|---|---|---|---|---|---|---|---|
| h2_cube6-211-classic-spin-None | B | 0.000e+00 | 1.110e-16 | 1.110e-16 | 1.110e-16 | 1.110e-16 | 0.000e+00 | 1.0 |
| h2_cube6-211-classic-spin-pbe0 | B | -2.220e-16 | 5.551e-17 | 5.551e-17 | 1.388e-15 | 1.388e-15 | 0.000e+00 | 0.25 |

### Historical auto post-HF source

This is source telemetry for the historical route, not an identity between different fits.

| Record | Cohort | Reference engine | Correlation engine | Correlation factor space | D93 label |
|---|---|---|---|---|---|
| h2_cube6-211-auto-posthf-source | B | native-sr-lr-shared-q | native-classic-fft-pair | occupied-virtual | not-qualified |

### Deadline observations and limitations

The elapsed process walls below are lower bounds on time to complete the unfinished work, not converged SCF timings. Their prior completed stages remain measured.

| Record | Cohort | Unfinished stage | Elapsed process seconds | Load start -> end |
|---|---|---|---|---|
| lih-211-sr-after | A | scf | 910.1252592910314 | 84.38 -> 83.22 |
| lih-211-sr-before | A | scf | 910.0737440410303 | 81.44 -> 81.80 |
| mgo-221-classic-after | B | posthf_classic_fit | 910.1228296250338 | 98.33 -> 84.00 |
| mgo-221-classic-before | B | posthf_classic_fit | 910.1331907080021 | 98.33 -> 84.00 |
| mgo-222-classic-before | B | posthf_classic_fit | 910.0894282499794 | 84.00 -> 91.77 |

LiH SR 211 final energy and fixed-density Fock are not measured; both processes
hit the recorded 900-second deadline. Thus the named-cell SR acceptance matrix
is incomplete. H2 SR 211 independently matches its unchanged SR path. SR SCF
on LiH/MgO 221 and 222 is not measured. Explicit SR post-HF is refused on the
listed cells/meshes before SCF; correlation energies for that profile are not
measured. Historical auto post-HF telemetry is not an SR representation test.

Classic MgO independent pair-builder correlation baselines are not measured.
The supplementary MgO 221 shared-q Gram/operator probe is separate evidence;
MgO 222 has SCF baseline/Fock and MP2 representation checks but no completed
independent pair-builder or shared-q Gram probe. Local/CC checks on the MgO
cells and independent full-space CC on the larger LiH meshes are not measured.
Open-shell correlation identities and full SR theory validation are outside
these numerical results. Canonical storage can use more memory than reduced
metric eigenmodes; the full AO-pair cache remains owned by the explicit classic
result through correlation and for its lifetime. No screening threshold or one-electron cutoff was relaxed.

The first LiH 222 residual miss, timed-out stages and all source refusals remain
visible in the records. These results do not establish numerical qualification
or full completion of every requested SR measurement. R12 submission reruns
remain separate from this current-main development campaign.


## Validation

`contracts-initial` (cohort A): 180 tests passed. `contracts-source-provenance`
(cohort B): 84 tests passed, with overlap; do not add these as unique tests.
Exact pytest target lists, stdout/stderr and process triples are in those
records. Coverage includes source validation/transport, the live canonical
operator/MP2 check, chi/CCM import firewall, generated suite manifest and
experimental marker consistency, output serialization/core hash, shared-q
canonical/tail/SR adapter tests and existing post-HF defaults/streaming.
All new tests have module-level `pytestmark = pytest.mark.experimental` and a
T3 manifest row generated through `scripts/test_gate/update_suite_manifest.py`.
Sphinx HTML documentation built successfully. No native source was modified
after the independent install and no loaded core was overwritten. Final source
changes after cohort B are documentation only, including the shared UHF
docstring. All 36 closed records and 213 top-level runtime manifests passed
checksum/triple verification; nested pytest fixtures remain hashed as fixtures.
Replay payloads into fresh record directories, preserving the original records.
