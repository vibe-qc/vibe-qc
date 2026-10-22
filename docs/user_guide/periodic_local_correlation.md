---
myst:
  html_meta:
    "description": "DLPNO-CCSD and DLPNO-CCSD(T) on 3-D crystals in vibe-qc: the run_periodic_job front door at the Gamma point and on Gamma-centred k meshes folded to the Born-von Karman supercell, per-cell energies, the reference the route accepts and the requests it refuses, the output, manifest and QVF fields, the PySCF KRCCSD(T) parity evidence, and the finite-size and folding theory behind the numbers."
    "og:title": "vibe-qc user guide: local correlation on crystals (periodic DLPNO-CCSD(T))"
---

# Local correlation on crystals: periodic DLPNO-CCSD(T)

`run_periodic_job(system, basis, method="dlpno-ccsd(t)")` correlates a
periodic Hartree-Fock reference with the molecular DLPNO-CCSD(T) engine:
RHF for a singlet cell, UHF for an open-shell one. The route runs on a 3-D cell at the Gamma point or on
a Gamma-centred `kpoints=` mesh, folds the mesh to the real Born-von Karman
(BvK) supercell, localises the occupied orbitals with Pipek-Mezey, and
reports every energy in total and per unit cell. Nothing periodic gets its
own coupled-cluster loop: the solver, the triples, the thresholds and the
option objects are the ones the [DLPNO guide](dlpno_mp2.md) describes for
molecules, read through the correlation-source contract of
[`vibeqc.dlpno.source`](../design_dlpno_correlation_source.md).

```{admonition} Status: preview
:class: warning

The periodic DLPNO route is reachable, maintained and tested, but its
validation envelope is narrower than the molecular route's. It runs for
Hartree-Fock references on the Gaussian density-fitting route (RHF for
singlet cells, UHF for open-shell cells), one dimensionality (`dim=3`) and
integer occupations; everything else fails
closed before the SCF with a `ValueError` or `NotImplementedError` that
names the reason. Its full-domain limit equals canonical k-point CCSD(T)
(PySCF KRCCSD(T)) to the fit tolerance, but its accuracy at the shipped
thresholds on real crystals and its cost against cell size have not been
measured, so the `.out` block says "this route is not production-gated yet"
and the [design page](../design_dlpno_correlation_source.md) lists the open
gates. The route emits no `ExperimentalWarning`. The label vocabulary is
defined in [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

## What runs

| | |
|---|---|
| Entry point | `run_periodic_job(system, basis, method="dlpno-ccsd(t)")` or `method="dlpno-ccsd"` |
| Library route | `vibeqc.periodic_dlpno.run_periodic_dlpno_ccsd(system, "basis-name", kmesh=(2, 2, 2))` |
| Reference | RHF (`run_krhf_periodic_gdf`) for a singlet cell, UHF (`run_kuhf_periodic_gdf`) for `multiplicity > 1` or when `LocalUCCSDOptions` are passed; multi-k GDF engine with `fit_source="classic"` and the fit retained |
| Sampling | Gamma (`kpoints=None`) or a Gamma-centred mesh `kpoints=(n1, n2, n3)`, folded to the real BvK supercell of `n1 n2 n3` cells |
| Correlation factors | the SCF's own retained RSGDF fit in the canonical auxiliary frame (`same_source=True`) |
| Occupied orbitals | Pipek-Mezey localised active block; the frozen core stays canonical |
| Pair distances | minimum image on the BvK supercell (for `coupling_radius` and the local (T0) triple keys) |
| Energies | total for the supercell and per unit cell; `result.energy_total` is the correlated total per cell |
| Controls | `dlpno_ccsd_options=LocalCCSDOptions(...)` (RHF) or `LocalUCCSDOptions(...)` (UHF), `dlpno_thresholds="loose"|"normal"|"tight"`, `frozen_core=`, `aux_basis=` |
| Artifacts | `.out` block, `.system` manifest (`dlpno_*`, `correlation_route`), structured log event `dlpno_ccsd_done`, citations, QVF section `x_vibeqc.dlpno_correlation` |

The engine behind both methods is the compiled PNO-space solver
(`residual_domain="pno"`) of the molecular route, labelled
`local PNO-space (M3c)` in every artifact, followed for `"dlpno-ccsd(t)"` by
the closed-shell triples in the mode `LocalCCSDOptions.triples_mode`
selects: the default `"t1"` (canonical (T) on the rotated PNO amplitudes),
`"local"` (the compiled DLPNO-(T0) kernel) or `"exact"`. The
[DLPNO guide](dlpno_mp2.md#dlpno-ccsd-and-dlpno-ccsdt) explains the
three; on a crystal they mean the same thing, evaluated on the supercell
orbitals.

An open-shell cell runs the spin-orbital DLPNO-UCCSD solver of the molecular
route on the KUHF reference instead, with spin-pure PNOs per spin block and
its default semicanonical local (T0) (`LocalUCCSDOptions.triples_mode="t0"`;
`"t1"` and `"t1-iterative"` are the alternatives). The citation route and
the engine name in every artifact become `periodic-gdf-dlpno-uccsd(t)`, and
the manifest adds `dlpno_reference_kind`, `dlpno_n_occ_alpha` and
`dlpno_n_vir_alpha`.

## When to use it

Use this route when you want a coupled-cluster correlation energy per cell
for a gapped 3-D crystal on a Gaussian basis, closed or open shell, and you can
afford a reference on the full Gamma-centred mesh. It is the right tool for
checking a periodic MP2 or DFT number against CCSD(T) on a small cell, for
converging a correlation energy per cell against the mesh at fixed
thresholds, and for comparing crystals that share a basis and a mesh. It is
not yet the tool for a published absolute energy: the shipped NormalPNO
thresholds have been measured against full domains on the H2 molecular
crystal only (23 µHa and 4 µHa per cell on two meshes, see
[Evidence](#evidence)), and no crystal with a real band structure has been
measured at the shipped thresholds.

Two other periodic local-correlation lines exist and stay experimental:
the AICCM Gamma-CCM and chi-CCM DLPNO drivers
([AICCM](aiccm.md), [chi-CCM](aiccm2026dev_b.md)), which warn at runtime,
and the megacell route (`examples/periodic/periodic-megacell-local-correlation.py`),
which correlates an open finite supercell with the molecular drivers. The
front door on this page is the one that reads the periodic SCF's own fit
and reports the source identity.

## Quick start

```python
import numpy as np
import vibeqc as vq

a, d = 6.0, 1.4   # bohr: a dilute simple-cubic H2 molecular crystal
system = vq.PeriodicSystem(3, np.diag([a, a, a]), [
    vq.Atom(1, [a / 2, a / 2, a / 2 - d / 2]),
    vq.Atom(1, [a / 2, a / 2, a / 2 + d / 2]),
])
basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")

result = vq.run_periodic_job(system, basis, method="dlpno-ccsd(t)",
                             kpoints=(1, 1, 2), output="h2_112")
c = result.correlation
print(c.n_cells, c.e_hf_per_cell, c.e_corr_per_cell, c.e_t_per_cell)
print(result.energy_total)          # correlated total per cell, Ha
print(c.source_kind, c.same_source) # 'bvk-supercell', True
```

Measured with the vibe-qc 0.17.15 development build (main, 2026-10-11),
def2-SVP with the driver's def2-SVP-JK auxiliary, all electrons correlated:
`E_HF = -1.1889076672`, `E_corr(CCSD) = -0.0262675726`,
`E(T) = -6.162e-05` Ha per cell, total `-1.2152368592` Ha per cell, in
10 s of wall time on a loaded laptop. The
[tutorial](../tutorial/periodic_dlpno_ccsd_t.md) runs the mesh series and
the refusals.

## The API

### `run_periodic_job`

The front door accepts the usual periodic arguments plus the three DLPNO
controls of `run_job`. Everything it cannot run is refused before any SCF
work, with the reason in the exception.

| Argument | Default | Meaning on the DLPNO front door |
|---|---|---|
| `method` | | `"dlpno-ccsd"` or `"dlpno-ccsd(t)"` (case-insensitive). Any other method refuses the three DLPNO controls below. |
| `kpoints` | `None` | `None` is the Gamma point. An integer `n` means `(n, n, n)`; a 2-tuple `(n1, n2)` means `(n1, n2, 1)`; a 3-tuple is the mesh. The route builds the unreduced Gamma-centred mesh in cyclic order itself (`vibeqc.periodic.chi.scf.cyclic_gamma_mesh`); a `KPoints` object is refused because it may carry a half shift or a symmetry reduction. |
| `dlpno_ccsd_options` | `None` | A `LocalCCSDOptions` (from `vibeqc.dlpno.ccsd_local_solver`) for an RHF reference, or a `LocalUCCSDOptions` (from `vibeqc.dlpno.uccsd_local_solver`), which also selects UHF for a singlet cell. For `LocalCCSDOptions`, `localise` must stay at its default `"boys"` (replaced silently by `"none"`, the adapter localises) or be `"none"`; any other value is refused. `compute_triples` follows the method. `n_frozen` is overridden by `frozen_core=`. Pilot option types are refused. |
| `dlpno_thresholds` | `None` (NormalPNO) | `"loose"`, `"normal"`, `"tight"` or a `DLPNOThresholds` record, applied to the supported subset of the options exactly as on `run_job`. A typo is caught before the reference runs. |
| `frozen_core` | `None` (published) | ORCA 6.1 Table 2.69 count-only convention per unit cell; `False` correlates all electrons; an integer is an explicit count per cell. The engine freezes that count times the number of cells. |
| `aux_basis` | `None` | The JK auxiliary basis of the reference, which the fit retains and the correlation reads; `None` resolves `default_aux_for(basis)` (def2-SVP-JK for STO-3G and def2-SVP). |
| `jk_method` | `"auto"` | `"auto"` selects GDF; `"gdf"` is accepted; every other route (`"bipole"`, `"gpw"`, `"gapw"`, the AICCM spellings) is refused. |
| `fit_source` | `"auto"` | `"auto"` or `"classic"`; the reference always runs the classic per-pair RSGDF fit, the one the driver can retain. |
| `rhf_options` / `uhf_options`, `gdf_method`, `rsgdf_ke_cutoff`, ... | as for RHF / UHF | Forwarded to the GDF reference unchanged. A UHF reference needs `uhf_options`, not `rhf_options`. |
| `structured_log` | `False` | `True` also writes the `dlpno_ccsd_done` event to `{output}.scf.jsonl`. |

The returned object forwards the SCF result (so `result.energy` is still
the SCF energy per cell and `result.converged` the SCF flag) and adds:

| Attribute | Type | Content |
|---|---|---|
| `result.correlation` | stamp | `method`, `backend` (the citation route key), `kmesh`, `n_cells`, `source_kind`, `source_identity`, `same_source`, `exchange_q0`, `localisation`, `pno_norm`, `e_hf`, `e_corr`, `e_t`, `e_total` (supercell totals), `e_hf_per_cell`, `e_corr_per_cell`, `e_t_per_cell`, `e_total_per_cell`, and `e_correlation` (CCSD plus (T) per cell) |
| `result.dlpno_ccsd` | `LocalCCSDResult` | the engine result on the supercell: `n_pairs`, `n_screened`, `pno_per_pair`, `n_iter`, `converged`, `e_pno_correction`, `avg_tno`, `n_triple_keys`, `source_kind`, `source_identity`, ... |
| `result.periodic_dlpno` | `PeriodicDLPNOCCSDResult` | the library result below, including `hf_result` and the `CorrelationSource` |
| `result.energy_total` | float | SCF plus CCSD plus (T) per cell, Hartree |
| `result.run_fields` | dict | the `dlpno_*` manifest fields |

### `run_periodic_dlpno_ccsd`

The library route runs the same engine without the runner's artifacts:

```python
from vibeqc.periodic_dlpno import run_periodic_dlpno_ccsd

res = run_periodic_dlpno_ccsd(system, "def2-svp", kmesh=(2, 2, 2),
                              with_triples=True, aux_basis=None,
                              cc_options=None, localise="pipek-mezey",
                              progress=False)
```

`cc_options` is a `LocalCCSDOptions` (shipped defaults when `None`;
`localise` must be `"none"` or left at the default); `localise` here is the
adapter's localiser, `"pipek-mezey"` or `"none"` (canonical supercell
orbitals, the exactness setting); `with_triples=False` is `"dlpno-ccsd"`;
`**gdf_kwargs` go to `run_krhf_periodic_gdf`. The frozen core is
`cc_options.n_frozen` per cell. The result is a `PeriodicDLPNOCCSDResult`
with `kmesh`, `n_cells`, `n_bf`, `n_occ`, `n_vir`, `n_aux`, `e_hf`,
`e_corr`, `e_t`, `e_total`, their `_per_cell` versions, `converged`,
`exchange_q0`, `source_kind`, `source_identity`, `same_source`,
`localisation`, `cc_result`, `hf_result` and `source`. Two lower-level
pieces are public as well: `periodic_dlpno_scf_mesh(system, kmesh)` returns
the k-point set the reference must run on, and
`correlate_periodic_reference(system, basis, hf, kmesh=..., options=...)`
correlates a reference you converged yourself.

### The adapters

`vibeqc.dlpno.source.gamma_gdf_source(system, basis, hf, n_frozen=0,
localise="pipek-mezey")` wraps a single-k Gamma result, and
`bvk_supercell_source(system, basis, hf, mesh=(2, 2, 2),
n_frozen_per_cell=0)` folds a full-mesh result. Both return a
`CorrelationSource` whose `kind` is `"gamma-gdf"` or `"bvk-supercell"`,
whose `identity` is the SHA-256 of a provenance record (lattice, atoms,
charge, bases and their sizes, the driver's executed fit record with the
SHA-256 of the factors it contracted, the frozen-core count, the
localisation, the reference energy and the conventions), and whose
`conventions` carry `exchange_q0`, `kmesh`, `bloch_normalisation` and
`same_source`. The engine records `source_kind` and `source_identity` on
its result, and every artifact repeats them, so a correlation energy names
the Hamiltonian it was computed on.

## What the reference must be

The route builds the reference itself, so these conditions hold
automatically through `run_periodic_job`; they matter when you converge
the SCF yourself and hand it to `correlate_periodic_reference` or an
adapter.

* A `run_krhf_periodic_gdf` result on `periodic_dlpno_scf_mesh(system, kmesh)`:
  the Gamma point for `(1, 1, 1)`, otherwise the unreduced Gamma-centred
  mesh in cyclic order. A Monkhorst-Pack mesh with a half shift on an even
  axis is refused by the fold with the message "the reference mesh is not
  the Gamma-centred unreduced mesh in cyclic order".
* Run with `fit_source="classic"` and `retain_fit_cache=True`
  (`vibeqc.periodic_dlpno.PERIODIC_DLPNO_SCF_KWARGS`). The default fit
  source takes a Gamma fast path that retains no fit and is refused with
  that reason; on a mesh the retained fit must hold every `(k_i, k_j)` pair.
  The two fit sources agree on the SCF energy to the fit tolerance (H2
  cube, STO-3G: 6e-10 Ha).
* Converged, with an integer aufbau occupation of every spin channel at
  every k-point; smeared occupations are refused. On a mesh the folded spectrum
  must have a gap at the supercell Fermi level, otherwise the mesh's filling
  is not a global aufbau and the fold refuses.
* On the GDF route. BIPOLE references have no certified same-source
  correlation factors, and symmetry-reduced references keep their orbitals
  on the irreducible wedge only; both are refused (design decision 2).
* `dim=3`. A chain or slab is run in a vacuum box, as a 3-D cell.

## Refusals

Every refusal fires before the reference SCF, so a request the route
cannot honour costs nothing. The messages below are the measured text of
the 0.17.15 development build, abbreviated to their reason.

| Request | Exception | Reason given |
|---|---|---|
| `functional="pbe"` | `ValueError` | DLPNO-CCSD(T) correlates a Hartree-Fock reference; drop `functional=`. |
| `jk_method="bipole"`, `"gpw"`, `"gapw"` or any non-GDF route | `NotImplementedError` | The DLPNO front door runs on the GDF route only: a BIPOLE reference has no certified same-source correlation factors. Leave `jk_method="auto"` or pass `"gdf"`. |
| `symmetry_reduce_k=True` | `NotImplementedError` | A symmetry-reduced reference keeps its orbitals on the irreducible wedge only. Run the full mesh. |
| `kpoints=KPoints(...)` | `ValueError` | Pass `kpoints=` as a mesh tuple; a `KPoints` object may carry a half shift or a symmetry reduction. |
| `fit_source="rsgdf"` or any value other than `"auto"`, `"classic"` | `ValueError` | The reference runs with `fit_source="classic"`, the per-pair RSGDF fit the driver can retain as same-source factors. |
| `smearing=` or `smearing_temperature > 0` | `NotImplementedError` | The engine needs an integer-aufbau determinant. |
| `bz_integration=` other than the mesh sum | `NotImplementedError` | Not wired for the DLPNO front door. |
| `optimize=True` or `hessian=True` | `NotImplementedError` | DLPNO-CCSD(T) gradients do not exist; run the energy only. |
| `dft_plus_u=` | `NotImplementedError` | Applies to Kohn-Sham references only. |
| `dim < 3` | `NotImplementedError` | The GDF route runs on 3-D cells; embed a chain or slab in a vacuum box. |
| `dlpno_ccsd_options=` of neither local option class | `TypeError` | The dense correctness pilot is molecular only; a UHF reference needs `LocalUCCSDOptions`. |
| `LocalCCSDOptions(localise="pipek-mezey")` or any value but `"boys"`/`"none"` | `ValueError` | The adapter localises the occupied orbitals; the engine's localiser is not periodic-safe. |
| `dlpno_ccsd_options=`, `dlpno_thresholds=` or `frozen_core=` with another `method` | `ValueError` | These belong to the DLPNO front door. |

The adapters raise `vibeqc.dlpno.source.CorrelationSourceError` (a
`ValueError`) for the reference conditions of the previous section: a
BIPOLE or symmetry-reduced reference, a missing retained fit, a multi-k
result handed to the Gamma adapter, a half-shifted or incomplete mesh, a
non-Hermitian fold (residual above `residual_tol=1e-7`), a folded spectrum
that does not reproduce the per-k orbital energies to 1e-6 Ha, and
fractional occupations. A reference that did not converge raises
`RuntimeError` in the library route; through `run_periodic_job` the
correlation step is skipped with a line in the `.out` and no energy.

## Output, manifest and QVF

The `.out` carries one block after the SCF. This is the measured block of
the H2 crystal in STO-3G on the (2, 2, 2) mesh (0.17.15 development build):

```text
  DLPNO-CCSD(T) local PNO-space (M3c) on the periodic GDF reference (RI: def2-svp-jk)
  -----------------------------------------------------------------------------------
  reference                      = KRHF/GDF, fit_source=classic; the retained fit is the correlation factors (same source)
  k-mesh / cells                 = (2, 2, 2) / 8 (Gamma-centred, folded to the real BvK supercell)
  correlation source             = bvk-supercell, identity f3d779756aaecd36... (full in .system)
  exchange q=0                   = BvK-ewald
  occupied localisation          = pipek-mezey (active block, by the adapter)
  E(RHF reference) / cell        =    -1.1203818266 Ha
  Threshold convention           = NormalPNO
  Truncation thresholds          = tcut_pairs 0.0001  tcut_pno 3.33e-07  tcut_mkn 0.001  tcut_tno 0.0
  Residual domain                = pno
  PNO density norm               = mp2
  PNO MP2 correction             = on
  frozen core / cell             = 0 orbitals (ORCA 6.1 published count-only default)
  Triples algorithm              = rotated-occupied (T1) compatibility correction (triples_mode='t1')
  pairs / avg PNOs               = 12 / 1.3
  E(PNO truncation corr)         =     0.0000000000 Ha
  E(CCSD correlation) / cell     =    -0.0158587108 Ha
  triples / avg TNOs             = 92 / 8.0   (smallest domain: 8)
  E((T) correction) / cell       =    -0.0000000000 Ha
  E(DLPNO-CCSD(T) corr) / cell   =    -0.0158587108 Ha
  E(DLPNO-CCSD(T) total) / cell  =    -1.1362405374 Ha
  supercell E(corr) / E(total)   =    -0.1268696862 Ha /    -9.0899242989 Ha   (8 cells)
  note: PNO-space local solver on the periodic adapter; rotated-occupied (T1) compatibility correction on the converged amplitudes. Finite local domains remain approximate; this route is not production-gated yet (docs/design_dlpno_correlation_source.md).
```

At the Gamma point the `k-mesh / cells` line reads `(1, 1, 1) / 1 (Gamma
point)`, the source is `gamma-gdf`, and the supercell line is omitted. The
pair count, the PNO averages and the triples count describe the supercell
(here 8 occupied orbitals, 12 iterated pairs out of 36, 24 screened by
`tcut_pairs`); the energies are per cell except on the supercell line. The
(T) of this minimal-basis model is zero to the printed precision: each H2
contributes one virtual orbital, so the triples have no space to act in
(the def2-SVP numbers in the [tutorial](../tutorial/periodic_dlpno_ccsd_t.md)
are the ones to look at for a (T)).

The `.system` manifest `[run]` table carries the flat threshold provenance
of the molecular route (`dlpno_threshold_preset`, `dlpno_tcut_*`,
`dlpno_residual_domain`, `dlpno_pno_norm`, `dlpno_domain_criterion`,
`dlpno_tcut_do`, `dlpno_tcut_pno_diag_scale`, `dlpno_pno_correction`,
`dlpno_triples_mode`, `dlpno_triples_algorithm`) and the periodic fields:

```toml
correlation_method = "dlpno-ccsd(t)"
correlation_route = "periodic-gdf-dlpno-ccsd(t)"
frozen_core_orbitals = 0            # per cell
frozen_core_orbitals_total = 0      # on the supercell
frozen_core_convention = "orca-6.1-table-2.69-count-only-default"
dlpno_engine   = "local PNO-space (M3c)"
dlpno_source_kind = "bvk-supercell"
dlpno_source_identity = "f3d779756aaecd36..."   # 64 hex digits
dlpno_same_source = true
dlpno_exchange_q0 = "BvK-ewald"
dlpno_localisation = "pipek-mezey"
dlpno_kmesh    = [2, 2, 2]
dlpno_n_cells  = 8
dlpno_converged = true
dlpno_n_pairs  = 12
dlpno_e_hf_per_cell = -1.1203818266
dlpno_e_corr_per_cell = -0.0158587108
dlpno_e_t_per_cell = -2.76e-33
dlpno_e_total_per_cell = -1.1362405374
```

With `structured_log=True` the `dlpno_ccsd_done` event carries the same
facts with `periodic = true`, `kmesh`, `n_cells`, `aux_basis`, the
threshold request, applied and settings records, `n_pairs`,
`avg_pno_per_pair`, `n_tno_triples`, `n_triple_keys_screened`, the supercell
totals and the per-cell energies.

The QVF archive gets a first-party vendor section, `id`
`x_vibeqc_dlpno_correlation`, `kind` `x_vibeqc.dlpno_correlation`, stored
as `vendor/x_vibeqc_dlpno_correlation/dlpno_correlation.json`. Its payload
(`schema_version` 1) holds `method`, `citation_route`, `engine`, `kmesh`,
`n_cells`, `source_kind`, `source_identity`, `same_source`, `exchange_q0`,
`localisation`, `converged`, the four per-cell energies, the four supercell
totals and `source_provenance`, the full record the identity hashes. The
SCF's Bloch wavefunction and density go into the archive as for any periodic
job; the correlated orbitals are not exported.

The citation sidecars list the route `periodic-gdf-dlpno-ccsd(t)` (or
`periodic-gdf-dlpno-ccsd`): {vibeqc-cite}`purvis_bartlett_ccsd_1982`,
{vibeqc-cite}`stanton_gauss_watts_bartlett_ccsd_1991` and
{vibeqc-cite}`deprince_sherrill_df_ccsd_2013` for the density-fitted CCSD
equations the compiled kernels transcribe, {vibeqc-cite}`riplinger_dlpno_2013`,
{vibeqc-cite}`riplinger_sparsemaps_2016` and
{vibeqc-cite}`pinski_dlpno_mp2_2015` for the PNO machinery,
{vibeqc-cite}`riplinger_triples_2013` and
{vibeqc-cite}`raghavachari_ccsdt_1989` for the triples,
{vibeqc-cite}`feyereisen_rimp2_1993` for the fit,
{vibeqc-cite}`pipek_mezey_1989` for the localisation and
{vibeqc-cite}`liakos_dlpno_thresholds_2015` for the threshold names. A run
whose closed-shell local (T0) kernel executed (`triples_mode="local"`) adds
{vibeqc-cite}`rendell_lee_komornicki_triples_1991` at runtime.

## Evidence

The numbers below are the measured state of the route. None of them is a
claim about crystals with a real band structure at the shipped thresholds.

**Exactness.** At full domains (`tcut_pno=0`, `tcut_mkn=0`, `tcut_pairs=0`,
`coupling_radius=0`, canonical supercell orbitals) the route reproduces
canonical CCSD on the same source to the engine's exactness ratchet, the
Coulomb energy rebuilt from its factors equals the SCF's, and the folded
(1, 1, 2) and (1, 1, 3) meshes reproduce the explicit Gamma supercell to
2e-13 Ha in the reference energy and 1e-15 Ha in the correlation energy,
triples included (`tests/test_periodic_dlpno_gamma.py`).

**PySCF KRCCSD(T) parity**, recorded on main on 2026-10-10 with
[`examples/regression/parity_periodic_dlpno_vs_pyscf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/parity_periodic_dlpno_vs_pyscf.py)
on the H2 molecular crystal (a = 6 bohr, d = 1.4 bohr) with the same
def2-SVP-JK auxiliary on both sides and all electrons correlated. PySCF
runs KRHF/GDF with `exxdiv="ewald"` and KRCCSD(T) in a child interpreter;
vibe-qc never imports it. In STO-3G on the (1, 1, 1), (1, 1, 2) and
(2, 2, 2) meshes and in def2-SVP on (1, 1, 1) and (1, 1, 2), the reference
energy per cell agrees to 0.04 µHa, the full-domain CCSD correlation energy
per cell to 0.01 µHa, and the def2-SVP (T) correction per cell on (1, 1, 2)
to 0.001 µHa, all with PySCF's `KRCCSD(mf).keep_exxdiv = True`; the GDF
integrals and orbital energies agree to eight digits. PySCF's default
(`keep_exxdiv=False`) rebuilds the coupled-cluster Fock operator without the
Ewald exchange zero-mode term, which raises every occupied orbital energy
by the Madelung constant inside the amplitude equations, and sits 4.6 to
8.8 mHa per cell lower on these cells. Both conventions share the
thermodynamic limit; which one the production route should carry is a
pending maintainer decision on the design page, whose recommendation is the
one the route implements (one Hamiltonian for reference and correlation,
convention named on every artifact). The shipped NormalPNO thresholds with
Pipek-Mezey orbitals sat 23 µHa (1, 1, 2) and 4 µHa (2, 2, 2) per cell
above full domains in STO-3G.

**Measured on this build** (vibe-qc 0.17.15 development build, main,
2026-10-11; wall times from a loaded laptop, eighteen threads, shared with
other jobs; the mesh is `kpoints=`, every run at the shipped defaults):

| System, basis | Mesh | Cells | E_HF per cell, Ha | E_corr(CCSD) per cell, Ha | E(T) per cell, Ha | Pairs (screened) | Wall, s |
|---|---|---:|---:|---:|---:|---:|---:|
| H2 crystal, STO-3G | (1, 1, 1) | 1 | -1.2385562896 | -0.0094929996 | 0 | 1 (0) | 1.6 |
| | (1, 1, 2) | 2 | -1.1725050397 | -0.0156121011 | 0 | 3 (0) | 1.7 |
| | (2, 2, 2) | 8 | -1.1203818266 | -0.0158587108 | 0 | 12 (24) | 3.7 |
| | (3, 3, 3) | 27 | -1.1177121108 | -0.0172763687 | -7.1e-07 | 54 (324) | 46 |
| H2 crystal, def2-SVP | (1, 1, 1) | 1 | -1.2561489570 | -0.0191910611 | 0 | 1 (0) | 7.5 |
| | (1, 1, 2) | 2 | -1.1889076672 | -0.0262675726 | -6.16e-05 | 3 (0) | 10 |
| | (2, 2, 2) | 8 | -1.1321596850 | -0.0285352330 | -3.36e-05 | 20 (16) | 40 |
| LiH rocksalt (primitive, a = 4.084 angstrom), STO-3G | (1, 1, 1) | 1 | -8.3351540806 | -0.0068114824 | -2.50e-06 | 3 (0) | 41 |
| | (2, 2, 2) | 8 | -7.9222475699 | -0.0162333820 | -2.81e-04 | 92 (44) | 77 |

The H2 single-cell (T) is identically zero (two electrons); the STO-3G
(1, 1, 2) and (2, 2, 2) values are below 1e-30 Ha, and the LiH table
freezes no core because the ORCA 6.1 table freezes nothing on Li or H.
These are model systems in minimal and double-zeta bases on meshes that
are not converged; the table shows the route running and the sizes of the
finite-size effects, not converged energies.

**Still open** (the production gates on the design page): the exactness
ratchet against the canonical supercell CCSD(T) of the chi and CCM lines,
a matched accuracy table at the shipped thresholds on real crystals, the
n-alkane chain as the 1-D polymer limit of the molecular table, and the
cost against cell size.

## Theory

### One engine, two adapters

The molecular DLPNO-CCSD(T) engine needs a reference (real MO coefficients,
AO Fock matrix, AO overlap, energy, electron count, frozen-core count),
three-centre factors $B^P_{pq}$ with $(pq|rs) = \sum_P B^P_{pq} B^P_{rs}$
over a real orthonormal auxiliary frame, and, on a crystal, the distance
function its locality screens must use. The periodic adapters produce
exactly that from a periodic GDF result and pass it through
`CorrelationSource`; the solver, the PNO construction, the pair screens and
the triples of the [DLPNO guide](dlpno_mp2.md#theory) then run unchanged.
Two rules make the result meaningful: the factors are the SCF's own
retained fit (one Hamiltonian for reference and correlation, `same_source`),
and the Fock matrix the amplitude equations see is the SCF's, exchange
zero-mode term included.

### Finite-size behaviour of the energies per cell

A Gamma-centred mesh of $N_k = N_1 N_2 N_3$ points samples the Brillouin
zone of the unit cell, or equivalently describes the BvK supercell of
$N_k$ cells with periodic boundary conditions; both views are the same
finite translation group. Each energy per cell therefore carries a
finite-size error that vanishes as $N_k \to \infty$, and the two parts of
the total converge at different rates.

The Hartree-Fock exchange energy contains the $G = 0$ singularity of the
Coulomb kernel. The route handles it with the Ewald probe-charge
convention of Gygi and Baldereschi ({vibeqc-cite}`gygi_baldereschi_exx_1986`),
reported as `exchange_q0 = "BvK-ewald"`: the divergent term is replaced
by the Madelung energy of a probe charge in the BvK supercell. The
leading remaining error of $E_{\rm HF}$ per cell is then proportional to
that Madelung constant, which scales as $L^{-1} \propto N_k^{-1/3}$ with
the supercell length $L$ ({vibeqc-cite}`makov_payne_1995`). The table
above shows it: the H2 crystal's HF energy per cell moves by 66 mHa from
(1, 1, 1) to (1, 1, 2) and by 2.7 mHa from (2, 2, 2) to (3, 3, 3).

The correlation energy per cell of a gapped system has no such
singularity; it is a sum over pairs whose contribution decays as
$R^{-6}$ with the pair distance, and its finite-size error behaves as
$N_k^{-1}$, one power of the supercell volume (Gruber, Liao, Tsatsoulis,
Hummel and Grüneis, *Phys. Rev. X* **8**, 021043 (2018),
[doi:10.1103/PhysRevX.8.021043](https://doi.org/10.1103/PhysRevX.8.021043);
for periodic MP2 and CCSD on Gaussian bases see
{vibeqc-cite}`marsman_periodic_mp2_2009` and
{vibeqc-cite}`mcclain_berkelbach_periodic_cc_2017`). Converge the two
errors separately: a mesh converged for the correlation energy per cell
may leave a Hartree-Fock error of millihartrees, and the correlated total
per cell inherits the larger of the two. At the Gamma point of a small
cell the correlation energy is also wrong for a second reason: with one
k-point every occupied orbital is a unit-cell function and the pairs
across the cell boundary are described through their periodic images
only, so the per-cell correlation energy of the H2 crystal grows from
9.5 to 15.6 mHa (STO-3G) when the second cell appears.

### Folding Bloch orbitals to the real BvK supercell

Let the unit cell carry $n$ basis functions and the mesh $N_c = N_k$
cells at lattice vectors $\mathbf R_c$. Every block-circulant operator
$M$ of the supercell (overlap, core Hamiltonian, Fock, density) is
diagonalised by the finite Fourier transform of the torus,

$$
U_{\mathbf k \mathbf R_c} = N_c^{-1/2}\, e^{i \mathbf k \cdot \mathbf R_c},
\qquad
M^{\rm sc}_{(c\mu),(c'\nu)} = \frac{1}{N_c} \sum_{\mathbf k}
  e^{i \mathbf k \cdot (\mathbf R_c - \mathbf R_{c'})}\, M_{\mathbf k}[\mu\nu],
$$

and the per-k orbitals $\psi_{p\mathbf k}$ with coefficients
$N_c^{-1/2} e^{i\mathbf k\cdot\mathbf R_c} C_{\mathbf k}[\mu p]$ are
eigenfunctions of the folded Fock operator with the per-k eigenvalues
(Theorem 1 of `periodic/ccm/bloch_reference.py`). The real supercell
Gamma problem of the folded Hamiltonian is therefore the full-mesh Bloch
problem, and this is what `bvk_supercell_source` builds: it
inverse-transforms the per-k overlap, the per-k Fock and every retained
$(\mathbf k_i, \mathbf k_j)$ block of the fit to the real supercell with
the finite-character kernels of the chi line, checks that the folded
matrices are Hermitian to `residual_tol=1e-7`, diagonalises the real
supercell Fock in the canonical orthogonal frame
($X = S^{-1/2}$ on the kept eigenvectors of $S^{\rm sc}$), and checks
the eigenvalues against the sorted union of the per-k ones to 1e-6 Ha.
The engine then sees one real problem of $N_c n$ functions with
$N_c n_{\rm occ}$ occupied orbitals, and the energies it returns are
divided by $N_c$ for the per-cell numbers. The fold needs the complete
unreduced Gamma-centred mesh in cyclic order: a half-shifted
Monkhorst-Pack mesh ({vibeqc-cite}`monkhorst_pack_1976`) is not a
representation of the cyclic translation group, and a symmetry-reduced
mesh does not carry every $\mathbf k$, so both are refused. At the Gamma
point ($N_c = 1$) the same construction reduces to realifying the
eigenvectors, which the driver returns complex with arbitrary phases:
`gamma_gdf_source` realifies them block by block and re-diagonalises the
real Fock inside each degenerate block.

### Minimum-image pair distances

A localised orbital on the torus has images in every cell. The Euclidean
distance between two orbital centroids in the home cell overestimates
the separation of a pair that is adjacent across the boundary (by up to
half a supercell period), so the open-cluster distance would silently
screen or under-couple pairs that are neighbours on the torus. The
adapters therefore hand the engine the exact minimum-image distance

$$
d_{ij} = \min_{\mathbf m \in \mathbb Z^3}
  \bigl\lVert (\mathbf r_j - \mathbf r_i) - A_{\rm sc}\,\mathbf m \bigr\rVert ,
\qquad
A_{\rm sc} = A\,\mathrm{diag}(N_1, N_2, N_3),
$$

evaluated by lattice reduction and sphere decoding
(`vibeqc.periodic_toroidal_mp2.toroidal_pair_distance_fn`), with
$\mathbf r_i$ the dipole centroid of localised orbital $i$. The distance
enters the occupied coupling sets of `coupling_radius` (12 bohr by
default) and the triple-key screen of the local (T0). The `tcut_pairs`
screen is an energy criterion, the semicanonical MP2 pair energy in the
full virtual space, and uses no distance; on the (2, 2, 2) H2 crystal it
moves 24 of the 36 supercell pairs to the MP2 level.

### Pipek-Mezey localisation on a torus

Foster-Boys localisation maximises the spread of orbital centroids and
needs the position operator, which is not periodic. Pipek-Mezey
({vibeqc-cite}`pipek_mezey_1989`) maximises the sum of squared Mulliken
atomic populations,

$$
\mathcal P = \sum_i \sum_A \bigl(Q^A_i\bigr)^2,
\qquad
Q^A_i = \sum_{\mu \in A} \sum_{\nu} C_{\mu i}\, S_{\mu\nu}\, C_{\nu i},
$$

and needs only the overlap and the atom-to-AO map, so it is well defined
for orbitals that wrap the cell. The adapters localise the active
occupied block of the real supercell this way
(`vibeqc.periodic_localise.localise_periodic_gamma`); the frozen core
stays canonical, so `n_frozen` keeps its meaning. The pair screens and
PNO spaces are built on the localised orbitals, as on a molecule. The
symmetry-adapted Wannier functions of {vibeqc-cite}`casassa_symmetry_wannier_2006`
and the Bloch intrinsic atomic orbitals of
{vibeqc-cite}`zhu_tew_bloch_iao_wannier_2024` are the localisers the
periodic local-correlation literature uses on k-point references; the
route on this page works on the real supercell and needs neither.

### Where this sits in the literature

Periodic local correlation on Gaussian bases goes back to the CRYSCOR
local MP2 of {vibeqc-cite}`pisani_cryscor_2012`; k-point coupled cluster
with Gaussian density fitting is {vibeqc-cite}`mcclain_berkelbach_periodic_cc_2017`,
the external reference of the parity script. The periodic DLPNO-MP2 of
{vibeqc-cite}`nejad_periodic_dlpno_pbc_2025` and its megacell embedding
{vibeqc-cite}`zhu_periodic_dlpno_megacell_2025` work with translational
symmetry inside the correlation step (unique home-cell pairs); that is
increment D of the design page. The route here instead expands the mesh
to its real supercell and runs the molecular engine on it, which is exact
by Theorem 1 and costs a supercell problem; it is the reference every
later k-aware increment is checked against.

## Limitations

* **Not production.** See the status admonition. The shipped thresholds
  are unmeasured on crystals with a real band structure; cost against
  cell size is unmeasured.
* **Supercell cost.** The engine solves a problem of $N_k n$ basis
  functions; the static factors of the PNO-space solver grow with the
  number of kept pairs and the auxiliary dimension of the supercell. The
  `max_nbf=1500` guard of `LocalCCSDOptions` applies to the supercell.
* **Hartree-Fock, 3-D, GDF, integer occupations.** Kohn-Sham and ROHF
  references, chains and slabs as `dim < 3`, BIPOLE and symmetry-reduced
  references, and metals with smearing are refused. Open-shell cells run on
  a UHF reference; their spin contamination is that of the UHF determinant.
* **Energies only.** No gradients, no stress, no `optimize`, no `hessian`.
* **Default triples cost.** `triples_mode="t1"` evaluates the canonical
  (T) on the supercell's rotated occupied orbitals, so its cost is that of
  a canonical (T) on the supercell; `triples_mode="local"` is the mode
  that scales.
* **The `(T)` of a minimal basis** can vanish or be meaningless; use at
  least a double-zeta basis before reading a triples correction.
* **Fock convention.** The amplitude equations use the SCF Fock with the
  Ewald exchange term; a code that strips it (PySCF's default) reports a
  lower finite-mesh correlation energy with the same thermodynamic limit.

## See also

* [Tutorial: DLPNO-CCSD(T) on a crystal](../tutorial/periodic_dlpno_ccsd_t.md),
  the H2 crystal mesh series, LiH and the refusals, measured.
* [DLPNO methods](dlpno_mp2.md): thresholds, triples modes, residual
  domains, the molecular accuracy table and the crystals section.
* [Design: one DLPNO engine for molecules and crystals](../design_dlpno_correlation_source.md),
  the correlation-source contract, the increments and the production gates.
* [Periodic methods](periodic_methods.md#3-gdf-native-gaussian-density-fitting),
  the GDF reference and its fit sources.
* [Bloch functions and k-points](../tutorial/kpoints_brillouin_bloch.md),
  the sampling the fold consumes.
* [`examples/periodic/input-lih-dlpno-ccsd-t.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-lih-dlpno-ccsd-t.py)
  and
  [`examples/regression/parity_periodic_dlpno_vs_pyscf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/parity_periodic_dlpno_vs_pyscf.py).
