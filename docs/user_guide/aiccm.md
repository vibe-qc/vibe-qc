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
    scf_reference=None,                  # None -> inferred; "rohf" | "roks" explicit
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
| `"chi"` | chi-CCM, the finite-translation-group character construction on a Gamma-centred character mesh | neutral, chi-defined | `vibeqc.periodic.chi` | RHF, RKS, UHF, UKS; dim 3 only; `aiccm_backend="four_center" \| "ri" \| "rijcosx"` plus the `aiccm_symmetry*` and `aiccm_wigner_seitz_shells` keywords |
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

Today the vocabulary is `"mp2"` and `"ccsd"`, wired on
`variant="four-center"` and `variant="real-gamma"`. The driver is chosen to
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

The two `dlpno-*` treatments are **`real-gamma` only**, and the reason is
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

`variant="neutral-bloch"` refuses, and for a reason worth knowing: its
reference is the per-k Bloch representation while the correlation drivers
work in the real-Gamma supercell space. Ruling R1 makes `real-gamma` the
**same neutral Hamiltonian** in the representation those drivers speak, so
the refusal names it as an exact substitute. `variant="chi"` refuses as
simply not yet wired; its drivers stay reachable as
`vibeqc.periodic.chi.posthf`.

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
* `scf_reference="rohf"` or `"roks"` asks for the restricted open-shell
  references explicitly (`"rohf"` rejects a `functional`, `"roks"` requires
  one). No variant implements them today, so the request fails closed with
  the runner's ROHF/ROKS refusal rather than being downgraded.

The keyword is not called `reference`: on the library correlation drivers
that name already means the correlation reference (`"neutral"` versus
`"direct"`). A broken-symmetry UHF on a closed-shell singlet cannot be
expressed through the front door; the deprecated `jk_method` spellings with
an explicit `method="UHF"` still reach it.

## The torus

Every variant reads `aiccm_lattice_extension=(N1, N2, N3)` (or an integer
for a cubic extension) as the Born-von Karman torus, and
`aiccm_wigner_seitz_shells=s` as the odd-extension shorthand `2s+1`. The
k-mesh argument `kpoints=` stays an accepted alias for the torus mesh
(inputs written before the front door keep running), for every variant the
rule is "the real-space control or the legacy k-mesh alias, not both", and a
shifted mesh is rejected because the mesh defines the torus, not a Bloch
sampling.

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

A convergence control that is set explicitly but that the selected variant
cannot execute fails closed before SCF instead of being dropped. The
`real-gamma` loop implements no damping, Fock mixing, density mixing or
level shift (its executed values are structural zeros, recorded per D86),
so an explicit `damping=`, `fock_mixing=`, `fmixing_percent=`,
`density_mixer=` or `level_shift=` on that variant raises;
`dynamic_damping=` is chi-only, as before. The real-gamma adapter forwards
neither the DIIS controls (`use_diis`, `diis_start_iter`,
`diis_subspace_size`) nor `initial_guess` / `solver`; those keep their
defaults on that variant. `convergence="auto"` is wired for the GDF and
BIPOLE routes only and is reported as "plain defaults" on every AICCM
variant.

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
`gdf_pair_reduction_factor` and `rsgdf_tail_ke_cutoff_executed`. The
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
  neutral cell are refused rather than reported.
* None of `real-gamma`, `neutral-bloch` and `four-center` forwards a mixing
  control, so an explicit `damping`, `fock_mixing`, `fmixing_percent`,
  `density_mixer` or `level_shift` fails closed on all three rather than
  being silently dropped.
* Open-shell spin bookkeeping differs between the two neutral producers:
  `neutral-bloch` uses the multi-k convention (the UNIT cell's multiplicity,
  replicated over the mesh), while `real-gamma` derives the spin from the
  SUPERCELL. They ask the same question only at a torus of `(1,1,1)`, so an
  open-shell comparison between the variants at a larger torus needs an
  explicit spin-state decision first.
* Post-HF from the runner is `correlation="mp2"` or `"ccsd"` on
  `variant="four-center"` and `variant="real-gamma"` (see
  [Correlation](#correlation)), plus `"dlpno-mp2"` / `"dlpno-ccsd"` on
  `real-gamma`. Correlation on `chi` stays library-only; `neutral-bloch` has
  no runner arm by representation rather than by schedule. Those are later
  milestones.
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
