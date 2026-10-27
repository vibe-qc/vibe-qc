---
myst:
  html_meta:
    "description": "Selecting the ab initio cyclic cluster model in vibe-qc: run_periodic_job(method='aiccm', variant=...), the four variants, the inferred SCF reference, the BvK torus keyword, convergence keywords, and the legacy selector spellings. Experimental."
    "og:title": "vibe-qc - AICCM (experimental)"
---

# AICCM: the ab initio cyclic cluster model (experimental)

```{warning}
**Experimental.** Every AICCM variant is a research method: the drivers emit
`vibeqc.AICCM2026DevAExperimentalWarning` (Gamma-CCM producers) or
`vibeqc.AICCM2026DevBExperimentalWarning` (chi-CCM), the `.system` manifest
records `[run].method_status = "experimental"` for every AICCM job, the test
files stay in the manual research lanes, and the experimental catalog rows
stay. "Standard" means the method is reached, configured, recorded and cited
like every other vibe-qc method, not that it stops being experimental.
```

## Selecting it

One front door selects every formulation; `variant` names the formulation
and is mandatory:

```python
import vibeqc as vq

res = vq.run_periodic_job(
    system, basis,
    method="aiccm",
    variant="real-gamma",                # real-gamma | chi | neutral-bloch | four-center
    functional=None,                     # None -> Hartree-Fock, a functional name -> Kohn-Sham
    scf_reference=None,                  # None -> inferred; "rhf" | "uhf" | "rohf" | "roks" explicit
    aiccm_lattice_extension=(2, 2, 2),   # the Born-von Karman torus, every variant
    correlation=None,                    # None -> SCF only; mp2 | ccsd | dlpno-*
    output="lih-aiccm",
)
```

Omitting `variant` with `method="aiccm"` fails closed listing the four
values. `jk_method` stays `"auto"` (the default) or equals the variant's own
underlying route; anything else raises `ValueError` naming both. Passing
`variant=`, `scf_reference=` or `correlation=` with any other `method` is
also refused rather than ignored.

## The variants

| `variant=` | what it runs | Hamiltonian | code path | status today |
|---|---|---|---|---|
| `"real-gamma"` | the neutral Gamma-CCM construction in its k-free real-Gamma supercell representation | neutral finite torus | `vibeqc.periodic.ccm.direct` through the per-unit-cell adapter `periodic/ccm/real_gamma_runner.py` | RHF, RKS, UHF, UKS; dim 3 only; `optimize=True` with a fixed cell works |
| `"chi"` | chi-CCM, the finite-translation-group character construction on a Gamma-centred character mesh | neutral, chi-defined | `vibeqc.periodic.chi` | RHF, RKS, UHF, UKS; classic physical-pair RI ROHF; dim 3 only; `aiccm_backend="four_center" \| "ri" \| "rijcosx"` plus the `aiccm_symmetry*` and `aiccm_wigner_seitz_shells` keywords |
| `"neutral-bloch"` | the same neutral construction as `real-gamma`, in the Bloch representation: the producer `run_ccm_*_gdf` on the torus mesh, through the adapter `periodic/ccm/neutral_bloch_runner.py` | neutral finite torus | `vibeqc.periodic.ccm.ri` | RHF, RKS, UHF, UKS; dim 3 only; the Gamma-centred mesh IS the torus; no gradients, no DFT+U, no restart |
| `"four-center"` | the union-and-weight / Wigner-Seitz four-centre lineage (2014 AICCM, symmetrised) | bare 1/r on the torus | `vibeqc.periodic.ccm.scf`, `dft`, `uhf` through the adapter `periodic/ccm/four_center_runner.py` | RHF, RKS, UHF, UKS; dim 3 only; HF on the scalable builder; no gradients, DFT+U or mixing controls |

`real-gamma` and `neutral-bloch` are one Hamiltonian in two representations
(Theorem 1 of the Gamma-CCM paper). Through the front door the two agree to
about 1e-13 Ha per cell on H2/STO-3G at torus sizes (1,1,1), (2,1,1) and
(2,2,1); the regression pin sits at 1e-9, and the library pair has its own
1e-8 pin. `four-center` and `chi` are distinct constructions from
it and from each other: a difference between variants is a construction
difference, never a convergence or representation artefact (D74, D89, D90
in the [decisions log](../aiccm2026dev_b_decisions.md)). There is no
`variant="gamma"`: the bare spellings `gamma`, `gamma-ccm`, `gamma_ccm` fail
closed on the runner exactly as on the library (ruling R1, 2026-08-21,
D124), because one word cannot choose between the two neutral producers and
must never silently select the four-centre lineage.

### External full-grid XC providers

The experimental SKALA full-grid provider is available for 3-D, all-electron,
zero-temperature RKS and UKS with `variant="real-gamma"` or
`variant="four-center"`. Pure full-grid providers are also available with
`variant="chi"` when `aiccm_backend="four_center"`; the chi RI and RIJCOSX
backends remain gated.
`variant="neutral-bloch"` deliberately rejects external providers: the
provider's atom-block periodic partition has not yet been proved invariant
under the real-Gamma/Bloch representation transform. This refusal is enforced
both by the front door and by the low-level neutral Bloch RKS/UKS producers.
See [Microsoft SKALA-1.1 neural XC](skala.md) for the complete provider
contract and remaining limitations.

## Correlation

`correlation=` adds a post-HF treatment on top of the variant's SCF. It is
optional -- `None`, the default, means an SCF-only run, which is the
ordinary complete request -- and it is orthogonal to `variant`: the
reference stays whatever the variant and the inference below chose.

```python
res = vq.run_periodic_job(
    system, basis,
    method="aiccm", variant="four-center", correlation="mp2",
    aiccm_lattice_extension=(2, 1, 1),
    output="h2-aiccm-mp2",
)
```

Today the vocabulary is `"mp2"`, `"ccsd"`, `"dlpno-mp2"` and `"dlpno-ccsd"`,
wired on `variant="four-center"` (the first two), `variant="real-gamma"` and
`variant="neutral-bloch"`. The driver is chosen to
**match the construction the SCF ran**, never to substitute another one, so
each arm cites its own lineage:

| `variant=` | `correlation=` | construction | citation route |
|---|---|---|---|
| `"four-center"` | `"mp2"` | union-and-weight | `aiccm2026dev-a-mp2` |
| `"four-center"` | `"ccsd"` | union-and-weight | `aiccm2026dev-a-ccsd(t)` |
| `"real-gamma"` | `"mp2"` | neutral fitted torus | `aiccm2026dev-a-ri-mp2` |
| `"real-gamma"` | `"ccsd"` | neutral fitted torus | `aiccm2026dev-a-ri-ccsd(t)` |
| `"real-gamma"` | `"dlpno-mp2"` | neutral fitted torus | `aiccm2026dev-a-dlpno-mp2` |
| `"real-gamma"` | `"dlpno-ccsd"` | neutral fitted torus | `aiccm2026dev-a-dlpno-ccsd(t)` |
| `"neutral-bloch"` | `"mp2"` | neutral fitted torus | `aiccm2026dev-a-ri-mp2` |
| `"neutral-bloch"` | `"ccsd"` | neutral fitted torus | `aiccm2026dev-a-ri-ccsd(t)` |
| `"neutral-bloch"` | `"dlpno-mp2"` | neutral fitted torus | `aiccm2026dev-a-dlpno-mp2` |
| `"neutral-bloch"` | `"dlpno-ccsd"` | neutral fitted torus | `aiccm2026dev-a-dlpno-ccsd(t)` |

Open-shell clusters take the `u` siblings of those rows automatically
(`aiccm2026dev-a-ri-ump2`, `-ri-uccsd(t)`, `-dlpno-ump2`, `-dlpno-uccsd(t)`).

`"ccsd"` is CCSD(T): the perturbative triples run by default, and the
citation key records **what the call actually computed** -- with triples off
at the library level the key loses its `(t)` and the reference list loses
Raghavachari, because citing a triples paper for a run without triples is the
misattribution per-call stamping exists to prevent. `"ccsd"` is closed-shell
only on both arms and refuses an open-shell reference: `run_ccm_uccsd` takes
no `method=`, so it cannot build the union-and-weight reference, and nothing
stamps a bare four-centre open-shell CCSD route. On `real-gamma` open-shell
CCSD **does** run: `run_ccm_uccsd` being neutral-only is precisely what makes
it right on that reference. `"mp2"` has an open-shell driver on both arms.

The two `dlpno-*` treatments are **fitted-reference only** (`real-gamma`
and `neutral-bloch`), and the reason is
structural rather than a gap: DLPNO screens pair energies on a fitted
reference and every driver takes the neutral `cderi`, while the bare
four-centre operator has no RI decomposition -- which is exactly why that arm
is dense and small-cluster only. At the default zero truncations DLPNO *is*
the canonical RI correlation on the same reference and the same `L`
(measured: `dlpno-mp2` matches `mp2` to 1.7e-18 Ha and `dlpno-ccsd` matches
`ccsd` exactly on H2/STO-3G), so the truncation thresholds are what buy the
scaling. The citation routes still differ, because a DLPNO run owes the
local-correlation papers a canonical one does not. Both DLPNO treatments work open-shell.

The two never share a citation row: handing a union-and-weight SCF to a
neutral-RI driver would return a different construction's number (ruling
R1). Both arms consume the SCF the front door already converged rather than
building a second one -- on `real-gamma` the adapter keeps the neutral
`cderi` the reference rode and hands the correlation that same array, so the
number reported is the correlation *of the energy printed beside it*, by
identity rather than by two builds agreeing.

`variant="neutral-bloch"` correlates through Theorem 1 (since 2026-09-25;
it used to refuse). Its reference is the per-k Bloch representation while
the correlation drivers work in the real-Gamma supercell space, and ruling
R1 makes the two the **same neutral Hamiltonian**: the arm runs the multi-k
SCF with its classic per-pair fit retained, folds overlap, core
Hamiltonian, Fock and density onto the supercell (the inverse finite
Fourier transform of the torus), builds real S-orthonormal canonical
supercell orbitals from every time-reversal pair or self-paired mesh
point, folds the driver's own fit to the supercell `cderi` (a rotation,
not a refit) and hands that reference to the same neutral-RI drivers the
`real-gamma` arm uses. The transform checks itself (fold realness,
orthonormality, Fock diagonality, eigenvalues, density, electron count,
integer occupations) and refuses a reference that fails; the `.out`
names the representation and the manifest records the residuals
(`aiccm_correlation_reference`, `aiccm_bloch_fold_*`, see
[output files](output_files.md)). One SCF then serves both the Bloch-side
artefacts (DOS, bands) and the correlation, and the eigenproblem is `N_c`
matrices of size `n` instead of one of size `N_c·n`. On the small tori
measured, LiH and MgO at (2,2,2), the two SCFs cost the same, because both
producers now build the same shared-q fit. The identity of the
correlation energies from the two producers is pinned at 1e-9 Ha/cell in
`tests/test_ccm_bloch_reference.py`; `real-gamma` stays the independent
validation producer. The same transform serves the library:
`vibeqc.periodic.ccm.ccm_reference_from_bloch` turns a
`run_ccm_{rhf,rks,uhf,uks}_gdf` result into the object
`run_ccm_rhf_direct` returns, every correlated entry point accepts a
Bloch result directly, and the one-call wrappers `run_ccm_ri_mp2`,
`run_ccm_ri_ccsd` and `run_ccm_dlpno_mp2_direct` default to
`reference="bloch"` (closed-shell unit cells; `"direct"` and `"neutral"`
stay selectable). The transform checks that the occupied Bloch orbitals
reproduce the Bloch density to 1e-8, which a Bloch SCF stopped at the
multi-k default `conv_tol_grad` of 1e-6 does not: LiH and MgO
pob-DZVP-rev2 at `(1,1,1)` were refused at 4e-7 and 2e-7. The one-call
wrappers and the runner's correlated `variant="neutral-bloch"` arm therefore
converge the Bloch SCF to `conv_tol_grad` 1e-8
(`BLOCH_REFERENCE_SCF_CONV_TOL_GRAD`) unless you set it; converge a Bloch
result you fold yourself the same way. `variant="chi"` has separate admitted explicit-source
[MP2/UMP2](aiccm2026dev_b.md#r51-same-source-canonical-mp2-and-ump2),
[ROHF RMP2](../chi_rohf_mp2.md) and [RHF/UHF CCSD(T)](../chi_ccsd.md)
routes. Their classic sources support [persistent restart](../chi_restart.md);
legacy `vibeqc.periodic.chi.posthf` entries retain their separate scope.

`correlation="mp2"` needs a Hartree-Fock reference; with a `functional=`
set, the inferred reference is RKS or UKS and the request fails closed
naming that. The reported `E(corr) / cell` and `E(SCF+corr) / cell` are
per unit cell, and the run cites the bare-lineage route
`aiccm2026dev-a-mp2` rather than the SCF's own row.

The cost is the constraint. The bare drivers form the dense `n_ref_ao**4`
AO tensor, so this is a small-cluster tool: there is no dimensionality
guard, but the tensor grows as the fourth power of the cluster basis size.
Size a torus for it deliberately.

## The SCF reference is inferred

`method="aiccm"` takes the place of the SCF string, so the reference is
derived the way the library's `run_ccm_scf` already derives it:

* `functional=None` runs Hartree-Fock; a functional name runs Kohn-Sham.
* The system decides restricted versus unrestricted: a multiplicity of 1
  with an even electron count is restricted (RHF or RKS); any other
  multiplicity, or an odd electron count, is unrestricted (UHF or UKS).
* `scf_reference="rhf"` or `"uhf"` asks for a Hartree-Fock reference
  explicitly, on every variant, so both reject a `functional`. `"uhf"` is
  mainly useful on a closed-shell singlet, which the rule above would run as
  RHF. `"rhf"` on an odd electron count is refused by the closed-shell check
  rather than changed to UHF. Kohn-Sham references have no explicit
  spelling: `functional=` selects them and the system decides restricted
  versus unrestricted.
* `scf_reference="rohf"` or `"roks"` asks for the restricted open-shell
  references explicitly (`"rohf"` rejects a `functional`, `"roks"` requires
  one). Classic physical-pair RI chi supports ROHF under its
  [shared-orbital contract](../chi_rohf_mp2.md). Other ROHF/ROKS combinations
  retain their capability refusals.

The keyword is not called `reference`: on the library correlation drivers
that name already means the correlation reference (`"neutral"` versus
`"direct"`). A UHF reference on a closed-shell singlet is requested with
`scf_reference="uhf"`. That alone does not break the spin symmetry: from a
spin-symmetric initial guess the UHF iterations can stay on the restricted
solution.

## The torus

Every variant reads `aiccm_lattice_extension=(N1, N2, N3)` (or an integer
for a cubic extension) as the Born-von Karman torus, and
`aiccm_wigner_seitz_shells=s` as the odd-extension shorthand `2s+1`. The
k-mesh argument `kpoints=` stays an accepted alias for the torus mesh
(inputs written before the front door keep running), for every variant the
rule is "the real-space control or the legacy k-mesh alias, not both", and a
shifted mesh is rejected because the mesh defines the torus, not a Bloch
sampling.

## The transverse vacuum of a padded cell

The two neutral producers refuse a cell declared 3-D whose atom positions
leave an empty slab above 25 bohr (and above four times the shortest lattice
vector) along a lattice direction, because the 3-D neutral construction then
corrects for periodic images that do not exist and the energy becomes a
padding artefact (the measured-dimensionality guard, IID 291). Since ruling
R6 (2026-09-10) the transverse vacuum of a chain is a *parameter whose
convergence is tested*, so the guard has an explicit opt-in:

```python
run_periodic_job(system, basis, method="aiccm", variant="neutral-bloch",
                 aiccm_lattice_extension=(4, 1, 1),
                 aiccm_allow_vacuum_padding=True, output="chain_L40")
```

With the opt-in the run proceeds, emits an `AICCMVacuumPaddingWarning`
naming the padded lattice directions, prints `allow_vacuum_padding = True`
in the `.out` header, and records `aiccm_allow_vacuum_padding = true` and
`aiccm_vacuum_padded_axes = "[2, 3]"` (the 1-based lattice directions that
measured as vacuum) in the `.system` manifest. The library drivers take the
same switch as `allow_vacuum_padding=True` and stamp the result with
`vacuum_padding_allowed` and `vacuum_padded_axes`. Three things do not
change: the default is off, so an unmarked padded run is still refused; the
keyword is refused on `four-center` and `chi`, which carry no such guard, so
it can never be silently ignored; and the second arm of IID 291, the refusal
of a converged *positive* energy on a neutral cell, stays in force. A
padded-regime energy is reportable only together with the padding it was
computed at, which is why the record carries it.

## Convergence keywords

Both lines gate the SCF on an energy change and on a gradient norm (the
DIIS commutator residual). Energy-converged is not density-converged, so a
tight `conv_tol_energy` alone does not give a converged density. The
`run_periodic_job` keywords map onto the drivers as follows:

| `run_periodic_job` keyword | `chi` (`PeriodicRHFOptions` / `PeriodicKSOptions`) | `real-gamma` (`run_ccm_*_direct` keywords) |
|---|---|---|
| `max_iter` (default 80) | `options.max_iter` | `max_iter` |
| `conv_tol_energy` (default 1e-7) | `options.conv_tol_energy` | `conv_tol` |
| gradient criterion | `options.conv_tol_grad`, fixed at 1e-6 | `conv_tol_grad`, fixed at 1e-6 |

Direct Python CCM SCF entry points require a finite, strictly positive
`conv_tol_grad`. They reject an invalid value before selecting an initial
guess or accessing the system; the error names the requested route.

A convergence control that is set explicitly but that the selected variant
cannot execute fails closed before SCF instead of being dropped. The
`real-gamma` loop implements no damping, Fock mixing, density mixing or
level shift (its executed values are structural zeros, recorded per D86),
so an explicit `damping=`, `fock_mixing=`, `fmixing_percent=`,
`density_mixer=` or `level_shift=` on that variant raises;
`dynamic_damping=` is chi-only, as before. The real-gamma adapter forwards
neither the DIIS controls (`use_diis`, `diis_start_iter`,
`diis_subspace_size`) nor `solver`; those keep their defaults on that
variant. It forwards `initial_guess`: HCORE (the AUTO default), a prepared
READ density, and an explicit SAD built on the supercell (see
[initial guesses](initial_guess.md#real-gamma-ccm-prepared-density)).
Both neutral variants take `fit_source="sr"`, the multi-k driver's default
SR/LR fit, which needs no RSGDF tail on core-heavy cells; the two variants
then record one neutral-source identity. The classic fit remains their
default (#519 RG-12). `convergence="auto"` is wired for the GDF and
BIPOLE routes only and is reported as "plain defaults" on every AICCM
variant.

## The real-Gamma producer in Python

The front door runs the real-Gamma producer with its defaults. The library
drivers `run_ccm_rhf_direct`, `run_ccm_rks_direct`, `run_ccm_uhf_direct` and
`run_ccm_uks_direct` in `vibeqc.periodic.ccm.direct` take the controls
below (#519). Only the fit, one-electron and XC sources change the
Hamiltonian, and each of them enters the neutral-source identity, so a
comparison across two sources is refused by identity instead of averaged.

### Storage of the fitted integrals

| `cderi_storage=` | what is kept | use |
|---|---|---|
| `"dense"` (default) | the supercell tensor `L[P, mu, nu]` | the reference; memory grows as `N_c^3` |
| `"block"` | one block per transfer `q` of the block-circulant torus | `N_c > 1`; J and K are built per channel |
| `"block-star"` | only the fits the k-pair star plan builds, every other member as a recipe on its representative; the `q = 0` channel whole | symmetric cells; implies `cderi_symmetry=True` |

The three storages are one operator. J and K agree to rounding on any
density, including one without the crystal symmetry, because the auxiliary
map of a star member is unitary and cancels in the exchange sum; the storage
is therefore not part of the source identity. `cderi_info["star_storage"]`
records the stored and total fit counts and the build's recipe check. The
front door selects the storage with
`run_periodic_job(..., method="aiccm", variant="real-gamma",
aiccm_cderi_storage="block")` and records it in the `.system` manifest as
`cderi_storage_executed`; dense stays the default, and a job with
`correlation=` keeps the dense array its treatment reads.

`exchange_wedge=True` (block or star storage built with symmetry) builds the
exchange at one bra per k-star and transports it to the others,
`K(Rk) = Om K(k) Om^H`. That identity needs a density with the star plan's
symmetry, so every call first measures the density's distance from it and
builds the full exchange when the distance is above `1e-10` of the density's
scale. `cderi_info["exchange_wedge"]` counts the calls, the wedge builds and
the fallbacks.

### Fit, one-electron and XC sources

`fit_source=` takes a fold source object:

* `ClassicRSGDFFitSource` (the default), the classic all-reciprocal fit of
  the multi-k GDF driver with its auto-sized high-|G| tail;
* `RangeSeparatedFitSource()`, the SR/LR range-separated fit of Ye and
  Berkelbach (J. Chem. Phys. 154, 131104 (2021)), planned from a raw
  integral error and needing no tail; the front door's `fit_source="sr"`;
* `MANTAFitSource(options, reciprocal_energy_cutoff)`, MANTA's periodic GDF
  tiles. Its Bloch control is MANTA's own whole-mesh RI method.

The classic fit takes its AO pairs from a ball of lattice cells of radius
`lat_opts.cutoff_bohr` (15 bohr by default) around the origin (#466). On cells
with diffuse shells that truncates pair densities once the torus resolves
them: LiH/STO-3G at `(3,3,3)` moves by 161 uHa/cell between a 15 and a
25 bohr list and is converged at 25 bohr. Give both producers the larger
`lat_opts.cutoff_bohr`, or use the SR fit, which has no such ball; its build
is still slow at `N_c > 1` (#543).

With the MANTA fit, `oneelectron=MANTAOneElectronSource(...)` takes the
overlap, core Hamiltonian and nuclear repulsion from MANTA too, and the KS
drivers take `xc_source=MANTAXCSource(...)` for MANTA's semilocal XC on its
own image domain. Together they reproduce MANTA's fitted HF and KS energies
per cell. The classes live in `vibeqc.periodic.ccm.fit_source`,
`vibeqc.periodic.ccm.one_electron_source` and
`vibeqc.periodic.ccm.xc_source`.

`run_ccm_rks_direct(xc_torus_complete=True)` (and the UKS sibling) sums the
XC density over every pair of images that reaches a grid batch, through the
torus residue of the pair, instead of the pair graph truncated at the XC
image domain. That is the domain-converged XC term, and much cheaper on cells
with diffuse shells: LiH/STO-3G PBE at `(1,1,1)` converges in 40 s instead of
444 s, 2.9e-11 Ha/cell away. It is opt-in because the Bloch producer keeps the
truncated pair graph, so the two differ by the domain's truncation (up to
about 1e-10 Ha/cell) and record different neutral-source identities.

### Initial guess and state selection

The default guess is HCORE. `initial_guess="sad"` builds the superposition
of atomic densities on the supercell, and `initial_density=` takes a
prepared READ density, for example the Bloch reference folded by
`ccm_reference_from_bloch` from a Bloch SCF converged to `conv_tol_grad`
1e-8. An ionic cell started from HCORE can converge to
a saddle point: MgO/pob-DZVP-rev2 at `(1,1,1)` lands about 0.26 Ha/cell
above the ground state, with a negative singlet stability eigenvalue.

`run_ccm_rhf_direct(stability_check=True)` evaluates the lowest RHF
orbital-Hessian eigenvalues in the singlet and triplet sectors after
convergence and stamps the molecular drivers' verdict fields
(`stability_checked`, `stability_eigenvalue`, `internal_instability`).
`follow_instability=True` rotates the occupied orbitals along a
translation-invariant singlet mode below `-stability_tol`, keeps the lowest
energy of an angle scan and converges again, at most
`max_instability_follow` times (default 3); `instability_follow` records
the steps. The runner arm `run_real_gamma_scf` takes the same two switches,
and the `.system` manifest records the frontier gap as `frontier_gap_ha`.

### The spin sector of the torus

Replicating a unit cell over the torus fixes its electron count,
`N_c N_e`, but not its spin sector. Before #519 RG-14 each neutral producer
had its own convention: the real-Gamma supercell drivers read the supercell
multiplicity that `CCMSystem` derives (an explicit cell multiplicity of 3 or
more replicates ferromagnetically, every other cell is paired by parity, so
a doublet cell on an even torus runs as a supercell singlet), while the
Bloch producer hands the unit cell to the multi-k drivers, which occupy the
cell's `(n_alpha, n_beta)` at every k point. On the Li doublet cube
(STO-3G, 7 bohr) at `(2,1,1)` the real-Gamma supercell singlet (3 alpha,
3 beta) and the Bloch per-cell doublet (4 alpha, 2 beta) are 4.8 mHa/cell
apart with one operator identity: two sectors of one Hamiltonian, not a
representation error.

`CCMSystem(..., spin_state=)`, `run_real_gamma_scf(spin_state=)`,
`run_neutral_bloch_scf(spin_state=)` and
`run_periodic_job(aiccm_spin_state=)` name the sector; it is resolved on
the cluster before the shell-aware dispatch, and an impossible request
(parity, count) fails at construction on every mesh.

| `spin_state` | real-Gamma executes | neutral-Bloch executes |
|---|---|---|
| `"legacy"` (default) | the historical supercell rule above | the cell's counts at every k point, as always |
| `"per-cell"` | the cell's `(n_alpha, n_beta)` replicated with spins aligned: `N_c (multiplicity - 1)` unpaired electrons | the same sector |
| an integer `M` | `M` as the total torus multiplicity, `n_alpha - n_beta = M - 1` | the same sector when `M - 1` is a multiple of `N_c`; otherwise refused by name |

The default changes nothing: both producers run what they ran, and the
executed sector is now recorded on every result (`result.spin_state`, a
`CCMSpinState` with the per-cell and torus counts, the arrangement and
whether it is uniform per cell), in the `.system` manifest
(`aiccm_spin_policy`, `aiccm_spin_state`) and in the QVF convention record.
A total multiplicity fixes `n_alpha - n_beta` and nothing else: it does not
select an antiferromagnetic or any other broken-symmetry arrangement of
local moments, which the producers take no per-cell seed for. The Bloch
producer represents only sectors that are `N_c` copies of one integer pair
(Theorem 2 of the theory notes maps block-circulant densities with
`Tr[D_sigma S] = N_c n_sigma` onto the Bloch state space); the supercell
producer also runs the non-uniform ones, alone.

A comparison between the producers needs both halves:
`assert_matched_neutral_source` (one operator) and
`assert_compatible_spin_state` (one executed sector: cells, charge,
electrons, alpha and beta counts), or `assert_comparable_neutral_results`
for both. A legacy pair at `N_c > 1` is refused by sector; a `"per-cell"`
pair is one state and one energy. A READ density carries its sector too:
the paired `(3, 3)` densities cannot start a `(4, 2)` run.

Integer occupations are a further condition, and every result records it
(`result.occupation`, an `OccupationRecord`: integer or ensemble, the
largest departure from an integer, the per-k counts of the Bloch producer,
the frontier gap and whether it is degenerate; in the manifest as
`aiccm_occupation`). The real-Gamma loops occupy an aufbau determinant on
the torus; the multi-k driver's zero-temperature fill gives every member of
a frontier-degenerate group the same fraction, an ensemble with no
determinant on the torus. On the Li doublet at `(2,2,1)` the per-cell
quintet has such a frontier at the zone-boundary points (occupations 0.5),
the counts match, the energies differ by 11 mHa/cell, and
`assert_comparable_neutral_results` refuses the pair as an ensemble before
any fold (`ccm_reference_from_bloch` refuses it too). A fixed per-k integer
fill is not offered by the shared multi-k driver; such a torus is run on the
real-Gamma producer alone, with its recorded frontier. The H cube doublet
on `(2,2,1)` and `(2,2,2)` has no such degeneracy and the two producers
agree to 1e-13 Ha/cell (`tests/test_ccm_spin_state.py`).

### Cross-code evidence

`studies/aiccm-2026/ladder_pyscf.py` compares the real-Gamma energy per cell
with PySCF's KRHF on the same Gamma-centred mesh, with vibe-qc's AO and
auxiliary shells transported to PySCF and `exxdiv='ewald'` on both sides. On
diamond, LiH, MgO and fcc Be (STO-3G) from `(1,1,1)` to `(2,2,2)` and MgO
pob-DZVP-rev2 at `(1,1,1)`, the SR fit agrees with PySCF's range-separated GDF
to 18 uHa/cell or better, the size of PySCF's own spread between its two fits.
The classic fit sits 32 to 215 uHa/cell above PySCF's GDF, its documented
dense-core floor. The measured tables are in the study's README.

## Legacy selector spellings

| spelling | before the front door | since the front door |
|---|---|---|
| `jk_method="real-gamma"`, `"real_gamma"` | the real-Gamma control | resolves to `variant="real-gamma"` with a `DeprecationWarning` naming the `method="aiccm", variant=...` form |
| `jk_method="aiccm2026dev-b"`, `"chi"`, `"chi-ccm"` | chi-CCM | resolves to `variant="chi"` with the same warning |
| `jk_method="aiccm2026dev-a"` | the four-centre lineage | resolves to `variant="four-center"` with the warning and dispatches the four-centre runner arm |
| `jk_method="gamma"`, `"gamma-ccm"`, `"gamma_ccm"` | the four-centre selector | **retired**: fails closed with the ruling-R1 message |
| `jk_method="neutral-bloch"`, `"bloch-control"`, `"gdf-control"`, `"aiccm-ri"` | plain unit-cell GDF | **retired**: fails closed with a pointer at `variant="neutral-bloch"` and the library entry (the same words name the neutral producer there; a word must not mean two Hamiltonians) |
| `jk_method="aiccm2026dev-a-real-gamma"`, `"aiccm2026dev-a-direct"` | rejected | rejected, unchanged (the A prefix names the four-centre construction) |
| `jk_method=PeriodicJKMethod.AICCM2026DEV_B` (or the other two legacy members) | dispatched silently | dispatched with the same `DeprecationWarning`; `PeriodicJKMethod.NEUTRAL_BLOCH` as a `jk_method` is refused |

The warning is attributed to your own call site, not to a vibe-qc frame,
which is what makes it visible: Python's default filters show a
`DeprecationWarning` only when it is blamed on `__main__`. The `.out` carries
the same pointer for logs and for callers that filter warnings away: the line
under "J/K method" reads `(user-requested: 'real-gamma'; deprecated spelling
of method='aiccm', variant='real-gamma')`.

## What the outputs record

The `.out` "J/K method" line reads `aiccm (<variant>) ...; experimental`,
followed by `(selected by method='aiccm', variant='...'; SCF reference RHF
inferred)`. The `.system` manifest carries `[run].method_status =
"experimental"`, `aiccm_variant`, `aiccm_selector` (`"front-door"` or
`"legacy-jk_method"`), the variant's route under `jk_method_requested` /
`_resolved` / `_executed` (a front-door job records the route value in all
three), the inferred SCF reference under `[plan].method`, and the variant's
convention record (`exchange_q0`, torus size, representation; see
[output files](output_files.md)). `neutral-bloch` adds the fit accounting
that makes the two producers auditable against each other:
`lpq_pair_symmetry`, `gdf_pair_builds`, `gdf_pair_total`,
`gdf_pair_reduction_factor`, `rsgdf_tail_ke_cutoff_executed` and
`rsgdf_fit_source_executed`; `real-gamma` records the last two as well.
Both record `neutral_source_identity`, one digest of the whole operator
(#519): a Theorem-1 pair must agree on it before its energies are compared.
Without the star reduction both pair counts are the number of `(k_i, k_j)`
fit tensors the multi-k driver built, read from its result
(`gdf_cderi_pairs`), not re-derived from its routing. The
two producers must agree on the tail and on the fit source, or they are not
one operator: since v0.17.1 the multi-k driver's default source for 3-D cells
is its range-separated fit (`sr`), while the real-Γ fold builds the classic
all-reciprocal fit, so the `neutral-bloch` producer pins `classic` and reads
the executed source back from the driver (vibe-qc #378). Pass
`fit_source="sr"` to the front door to run the driver's own default
deliberately; it is a different fitted operator.
Both neutral
variants record the vacuum opt-in (`aiccm_allow_vacuum_padding`,
`aiccm_vacuum_padded_axes`). Every periodic run whose producer keeps a
per-cycle trace also records the convergence it actually reached, beside
the thresholds it was compared against: `scf_cycles_recorded`,
`scf_final_delta_e_ha`, `scf_final_residual` and
`scf_final_residual_metric` (`max_abs_orthonormalised_commutator` for the
real-Gamma and lean neutral loops, `gradient_norm` for the multi-k driver
behind `neutral-bloch`; `not-recorded` where the loop keeps no trace). The
`[vibeqc]` section carries `native_core_sha256` and `native_core_bytes`,
the checksum of the compiled core the process actually loaded, so the
three-layer build identity (release, source revision, core checksum) is read
from the run without an out-of-tree wrapper. The
references block cites the variant's own route (`real-gamma`,
`neutral-bloch`, `aiccm2026dev-a`, or `aiccm2026dev-b`) rather than a plain
periodic SCF.

## Limitations of this milestone

* The front door always executes the symmetric Born-von Karman-torus
  four-centre weighting, `aiccm2026dev-a`. The library drivers still default
  to the historical `union12` weight, whose Coulomb supermatrix carries a
  negative subspace on any basis with more than one function per centre, and
  there is no runner keyword through which you could ask for it.
* `four-center` runs Hartree-Fock on the scalable lattice-sum builder, which
  applies the weights inside the shell-quartet loop. The dense builder that
  materialises the full effective integral tensor stays reachable as the
  library entry `run_ccm_rhf`, and remains the validation reference the two
  agree against; it runs out of memory on the first real three-dimensional
  cell. Open-shell HF has no scalable entry yet, so `UHF` uses the dense
  route and inherits its cluster-size ceiling. The `.system` records which
  builder ran.
* Both neutral producers refuse a few inputs plain GDF accepts, because they
  carry the neutral-torus guards: a cell declared 3-D whose transverse
  directions are measured as pure vacuum (a gap above 25 bohr and above the
  ratio to the shortest lattice vector) and a converged POSITIVE energy on a
  neutral cell are refused rather than reported. The first refusal has the
  explicit, warned, recorded opt-in `aiccm_allow_vacuum_padding=True`
  described above; the second has none.
* None of `real-gamma`, `neutral-bloch` and `four-center` forwards a mixing
  control, so an explicit `damping`, `fock_mixing`, `fmixing_percent`,
  `density_mixer` or `level_shift` fails closed on all three rather than
  being silently dropped.
* Open-shell spin bookkeeping differs between the two neutral producers
  under the default `aiccm_spin_state=None` (`"legacy"`): `neutral-bloch`
  uses the multi-k convention (the UNIT cell's multiplicity, replicated
  over the mesh), while `real-gamma` derives the spin from the SUPERCELL.
  They ask the same question only at a torus of `(1,1,1)`.
  `aiccm_spin_state="per-cell"` (or a total multiplicity) makes both run one
  sector; see [The spin sector of the torus](#the-spin-sector-of-the-torus).
* Post-HF from the runner is `correlation="mp2"` or `"ccsd"` on
  `variant="four-center"`, `variant="real-gamma"` and
  `variant="neutral-bloch"` (see [Correlation](#correlation)), plus
  `"dlpno-mp2"` / `"dlpno-ccsd"` on the two fitted producers. On
  `neutral-bloch` the reference is the Bloch SCF folded to real-Gamma
  orbitals (Theorem 1). Chi has separate admitted same-source canonical
  correlation routes described above.
* External full-grid XC is available on the `real-gamma`, `four-center`, and
  four-center-backed `chi` KS routes described above; `neutral-bloch` fails
  closed pending representation-invariance evidence.
* chi-CCM has no analytic gradient by design; `real-gamma` relaxes atoms on
  a fixed cell only; the four-centre gradients are dense small-cluster only
  and are not wired to the runner, so `optimize` fails closed there.
* `four-center` writes no population sidecar yet: its population is a crystal
  population over folded blocks and needs a registration in the shared output
  planner, which is requested and pending. An explicit
  `write_population_file=True` is refused with that reason rather than
  producing a molecular proxy.
* Every variant is 3-D only; double hybrids are reachable on no variant
  from the runner.

## See also

* [chi-CCM user guide](aiccm2026dev_b.md) and the
  [Gamma-CCM reference](../aiccm2026dev_a.md): backends, qualification,
  route keywords.
* [AICCM route support matrix](../experimental/aiccm2026dev_route_matrix.md):
  which route supports which method.
* [Periodic methods](periodic_methods.md) and
  [periodic JK routes](../periodic_jk_routes.md): the other Coulomb routes
  and the parity policy.
* [chi-CCM decisions log](../aiccm2026dev_b_decisions.md): D56, D74, D89,
  D90, D124.

The explicit classic chi route also provides a shared-orbital ROHF reference
and Knowles RMP2 including singles. Select `scf_reference="rohf"` and
`correlation="mp2"` with the [ROHF/RMP2 controls](../chi_rohf_mp2.md).
Scientific acceptance is pending and ROHF-CC remains unsupported.

Classic chi HF and admitted canonical correlation also provide
[source-preserving `.vqr` restart](../chi_restart.md). The four explicit paths
are `chi_reference_checkpoint`, `chi_reference_from`,
`chi_correlation_checkpoint` and `chi_correlation_resume`;
`chi_checkpoint_interval` controls complete iteration/work-unit publication.
Accepted-reference reconstruction is distinct from a convergence-required
guess, and neither storage integrity nor successful continuation establishes
scientific acceptance.
