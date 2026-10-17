---
myst:
  html_meta:
    "description": "Choosing the integral engine and the Coulomb and exchange algorithms in vibe-qc: libint (the default) and the experimental MANTA engine, exact, RI, BUPO and RI-BUPO Coulomb, exact and RI exchange, the IntegralEngineOptions, CoulombBuildOptions and ExchangeBuildOptions selectors on run_job and run_periodic_job, provenance, refusals, measured comparisons and the integral theory behind them."
    "og:title": "vibe-qc user guide: integral engines (libint, MANTA, RI, BUPO)"
---

# Integral engines: libint, MANTA, RI and BUPO

Every Hartree-Fock and Kohn-Sham calculation spends most of its time on
electron-repulsion integrals and on contracting them with the density into
the Coulomb matrix $\mathbf J$ and the exchange matrix $\mathbf K$. vibe-qc
computes them with libint by default. Since v0.17.16 a second engine,
MANTA, can be selected explicitly, together with an independent choice of
how $\mathbf J$ and $\mathbf K$ are built: exactly, by density fitting
(RI), or, for $\mathbf J$, by the bubble-pole multipole method (BUPO). This
page is the user guide to those selectors. The
[MANTA developer reference](manta.md) documents every primitive the engine
exposes.

```{admonition} Status: experimental
:class: warning

Selecting `integral_engine`, `coulomb` or `exchange` is experimental. With
none of the three keywords, every job runs exactly as before on libint and
its usual J/K routes. The selected routes reproduce the libint energies
where they should (the exact MANTA build matches libint to 1e-10 Ha in the
measurements below) and confirm their final density with a full build, but
they are bounded development routes: they refuse many workflows, they are
slower than libint today, and their accuracy profiles are not qualified for
production. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

## When to use which

| You want | Use |
|---|---|
| a production calculation | nothing: the default libint route |
| the same answer from an independent integral code | `integral_engine=IntegralEngineOptions(backend="manta")` with exact J and K |
| a density-fitted Coulomb build on a molecule | `coulomb=CoulombBuildOptions("ri", "cc-pvdz-jkfit")` |
| fitted exchange as well | add `exchange=ExchangeBuildOptions("ri")` |
| a multipole-accelerated Coulomb build | `coulomb=CoulombBuildOptions("bupo")` (MANTA only) |
| fitted Coulomb with multipole far field | `coulomb=CoulombBuildOptions("ri_bupo", aux)` (MANTA only) |
| a periodic MANTA job | the same three selectors on `run_periodic_job`, plus explicit `periodic_integrals=` domains |
| everyday RI-J or RIJCOSX | `density_fit=True` / `cosx=True` on the default route ([density fitting](density_fitting.md)) |

The last row matters: the long-standing `density_fit`, `aux_basis` and
`cosx` keywords of `run_job` are a separate, production route. They cannot
be combined with the selectors on this page, which refuse them with the
reason.

## The three selectors

All three are frozen dataclasses importable from `vibeqc.manta` (they live
in `vibeqc.manta_scf`).

**`IntegralEngineOptions`** picks the provider and its resource limits.

| Field | Default | Meaning |
|---|---|---|
| `backend` | `"libint"` | `"libint"` or `"manta"`. Selecting `"libint"` explicitly routes the job through the selected-engine path with libint integrals, which records the same provenance as MANTA. |
| `fallback` | `"error"` | Only `"error"` is admitted: a request the engine cannot honour raises; there is no silent fallback to another provider. |
| `channel_batch` | `1` | Number of contraction channels batched per integral traversal. |
| `budget` | `ResourceBudget()` | Explicit memory and work limits (`memory_bytes`, `shell_calls`, `primitive_quartets`, `scalar_terms`, `kernel_terms`); a job that would exceed one refuses before it starts. |

**`CoulombBuildOptions`** picks the $\mathbf J$ algorithm and, when anything
is fitted, the auxiliary basis.

| Field | Default | Meaning |
|---|---|---|
| `algorithm` | `"exact"` | `"exact"`, `"ri"`, `"bupo"` (MANTA only) or `"ri_bupo"` (MANTA only) |
| `auxiliary_basis` | `None` | Required when $\mathbf J$ or $\mathbf K$ is fitted, refused otherwise; a basis name or a `BasisSet` |
| `metric` | `RIOptions()` | Coulomb-metric rank policy of the fit |
| `bupo` | `BupoOptions()` | multipole order, near-field provider and error tolerance of BUPO |
| `ri_bupo` | `BupoFittingOptions()` | the same for the fitted three-centre coupling of RI-BUPO |

**`ExchangeBuildOptions`** picks the $\mathbf K$ algorithm, independently of
$\mathbf J$: `algorithm="exact"` (default) or `"ri"`. For a pure functional
exchange is inactive and the record says so.

## Molecular jobs

```python
import vibeqc as vq
from vibeqc import manta

water = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0]),
                     vq.Atom(1, [0.0, 1.4304, -1.1060]),
                     vq.Atom(1, [0.0, -1.4304, -1.1060])])

def budget():
    # Explicit limits; a job that would exceed one refuses before it starts.
    b = manta.ResourceBudget()
    b.memory_bytes = 2**30
    b.shell_calls = 10**9
    b.primitive_quartets = 10**13
    b.scalar_terms = 10**17
    b.kernel_terms = 10**18
    return b

res = vq.run_job(
    water, basis="cc-pvdz", method="rhf", initial_guess="hcore",
    integral_engine=manta.IntegralEngineOptions(backend="manta", budget=budget()),
    coulomb=manta.CoulombBuildOptions("ri", "cc-pvdz-jkfit"),
    exchange=manta.ExchangeBuildOptions("exact"),
    output="h2o-manta-rij",
)
print(res.energy, manta.molecular_engine_provenance(res)["coulomb_algorithm"])
```

What the molecular route admits:

- **Methods.** RHF, UHF, RKS and UKS, gas phase. Kohn-Sham runs keep the
  native libxc and VV10 composition and range-separated exchange with the
  kernel's own erf/erfc split.
- **Starting guess.** HCORE (the default when no options are passed) or a
  same-basis READ density. Atomic guesses are not admitted yet.
- **What follows the executed provider.** Population analysis, dipoles,
  natural orbitals, IAO populations and closed-shell IBO, Boys and
  Pipek-Mezey localisation are computed with the same one-electron provider
  that ran the SCF, and the QVF archive carries the localised orbitals.
- **Geometry optimisation.** RHF and UHF with exact or RI $\mathbf J$ and
  $\mathbf K$ through `geom_opt=` (the native optimiser), restarting each
  geometry from HCORE and keeping the same fitting source, recentred, for
  the final single point. Gradients and citations follow the executed
  method.

What it refuses, before any integral is computed, with
`NotImplementedError: Selected integral engines in run_job do not yet
qualify: ...` naming the item: geometry optimisation without `geom_opt`,
Hessians, TDDFT and response, atomisation energies, correlated method
selectors, QTAIM, solvation, dispersion, DFT+U, READ projection and
FRAGMO, the legacy `density_fit`/`aux_basis`/`cosx` keywords, and
finite-temperature smearing. A method other than RHF, UHF, RKS or UKS is
refused, as is `fallback` other than `"error"`, BUPO or RI-BUPO with
`backend="libint"`, an RI algorithm without `auxiliary_basis`, and an
auxiliary basis without an RI algorithm.

### Measured: water / cc-pVDZ

The vibe-qc 0.17.17 development build (main, 2026-10-14) on a shared laptop
(load about 50), HCORE start, auxiliary basis cc-pVDZ-JKFIT:

| Engine | J | K | Energy, Ha | Difference to exact, µHa | Wall, s |
|---|---|---|---:|---:|---:|
| libint (default route) | exact | exact | -76.0268153023 | reference | 0.4 |
| libint (selected) | exact | exact | -76.0268153023 | 0.0 | 2.7 |
| MANTA | exact | exact | -76.0268153023 | 0.0 | 44.6 |
| MANTA | RI | exact | -76.0268445288 | -29.2 | 41.0 |
| MANTA | RI | RI | -76.0267944326 | +20.9 | 37.6 |
| MANTA | BUPO | exact | -76.0268153023 | 0.0 | 45.1 |
| MANTA | RI-BUPO | exact | -76.0268445288 | -29.2 | 36.5 |
| libint (default route), PBE | exact | | -76.3333672257 | reference | 0.7 |
| MANTA, PBE | exact | | -76.3333672257 | 0.0 | 25.2 |
| MANTA, PBE | RI | | -76.3333947179 | -27.5 | 4.0 |

Read the table as a consistency check, not a benchmark:

- **Exact is exact.** MANTA's exact build reproduces libint to the printed
  ten digits for HF and PBE, so the engines agree on the integrals.
- **Fitting error.** RI-J moves the energy by about -29 µHa, the usual size
  of a JKFIT Coulomb fitting error on a small molecule; fitting exchange as
  well brings it to +21 µHa. Both are far below chemical accuracy and of the
  size the default route's `density_fit=True` gives.
- **BUPO.** On a molecule this small every shell pair is a near-field pair
  (`bupo_far_blocks = 0` in the record), so BUPO reduces to the exact build
  and RI-BUPO to RI-J. The multipole far field only pays off on larger
  systems.
- **Speed.** MANTA is a development engine and is tens of times slower than
  libint on this system. Its value today is the independent implementation,
  the explicit error ledgers and the provenance, not the wall time.

## Periodic jobs

`run_periodic_job` accepts the same three selectors for neutral,
all-electron 3-D cells, with explicit finite domains instead of the GDF
route's automatic cutoffs:

```python
import numpy as np
import vibeqc as vq
from vibeqc import manta

# budget() as in the molecular example above
cell = vq.PeriodicSystem(3, 6.0 * np.eye(3),
                         [vq.Atom(1, [-0.7, 0.0, 0.0]), vq.Atom(1, [0.7, 0.0, 0.0])])
basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")

res = vq.run_periodic_job(
    cell, basis, method="RHF", kpoints=(2, 2, 2),
    use_diis=False, conv_tol_energy=1e-10, exchange_exxdiv="ewald",
    integral_engine=manta.IntegralEngineOptions(backend="manta", budget=budget()),
    coulomb=manta.CoulombBuildOptions("exact"),
    exchange=manta.ExchangeBuildOptions("exact"),
    periodic_integrals=manta.PeriodicIntegralOptions(
        reciprocal_energy_cutoff=12.0, pair_cutoff_bohr=9.0,
        nuclear_cutoff_bohr=12.0, alpha=0.6),
    output="h2-manta-k222",
)
```

| Option class | Fields | Role |
|---|---|---|
| `PeriodicIntegralOptions` | `reciprocal_energy_cutoff` (Ha), `pair_cutoff_bohr`, `nuclear_cutoff_bohr`, `alpha` (Ewald split, default 0.7), `reciprocal_batch`, `madelung_alpha`, `madelung_tolerance` | the finite reciprocal sphere and real-space pair and nuclear domains |
| `PeriodicXCOptions` | `xc_image_cutoff_bohr`, `radial_points` (50), `lebedev_order` (29), `partition_radius_bohr` (8), `stratmann`, `grid_batch` | the exchange-correlation images and the periodic atomic quadrature, required for RKS/UKS |
| `PeriodicFittingOptions` | `omega`, `auxiliary_cutoff_bohr`, `integral_screen_error`, `image_candidate_cap` | the fitted Coulomb source, required when J or K is RI |

The periodic route runs RHF, UHF, RKS and UKS with pure, global-hybrid and
range-separated LDA/GGA functionals, exact or RI $\mathbf J$ and $\mathbf K$
(pure KS: RI-J and no exchange), the Born-von Karman Ewald exchange
correction (`exchange_exxdiv="ewald"` or `"none"`), periodic BUPO for RHF
and UHF with exact reciprocal exchange (`PeriodicBupoIntegralOptions`), and
full regular k meshes. It requires `jk_method="auto"`, `use_diis=False`, an
HCORE or AUTO guess and an all-electron basis, and refuses everything else
by name.

### Measured: H2 in a 6 bohr cube

The same build, STO-3G, the default GDF route against MANTA's exact
reciprocal RHF at three domain sizes (reciprocal cutoff in hartree, pair
cutoff in bohr, nuclear cutoff in bohr):

| Mesh | Route | Domains | Energy per cell, Ha | Difference to GDF, mHa |
|---|---|---|---:|---:|
| Gamma | GDF (default) | automatic | -1.2385562902 | reference |
| Gamma | MANTA exact | 2 / 3 / 4 | -1.1925039776 | +46.1 |
| Gamma | MANTA exact | 6 / 6 / 8 | -1.2129340733 | +25.6 |
| Gamma | MANTA exact | 12 / 9 / 12 | -1.2378207327 | +0.74 |
| (2, 2, 2) | GDF (default) | automatic | -1.1203818271 | reference |
| (2, 2, 2) | MANTA exact | 2 / 3 / 4 | -1.1548284131 | -34.4 |
| (2, 2, 2) | MANTA exact | 6 / 6 / 8 | -1.1229616309 | -2.6 |
| (2, 2, 2) | MANTA exact | 12 / 9 / 12 | -1.1205312789 | -0.15 |

The finite-domain energy approaches the GDF value only as all three domains
grow, and not monotonically. This is what "explicit domains" means: the
route computes exactly what you ask for and nothing more, so a MANTA
periodic number is only as converged as the domain series behind it. Run
such a series for every new cell.

## Provenance

A selected job records what actually ran. The `.out` prints, for example,

```text
  Integral engine: manta; J=bupo (manta); K=exact (manta).
  Full-density confirmation: True; stationarity: True; J/K shell calls: 1019388.
  BUPO: order 6; near provider manta; far blocks 0; moment panels 0.
```

`vibeqc.manta.molecular_engine_provenance(result)` returns the same record
as a dictionary: providers and algorithms for the one-electron terms,
$\mathbf J$ and $\mathbf K$, source identities, shell calls and measured
primitive work, wall times, whether a full build confirmed the final
density (`full_build_confirmed`) and whether the converged state is
stationary on that build (`stationarity_confirmed`), the BUPO order, far
blocks and analytic $\mathbf J$ bound, the RI-BUPO metric rank and bounds,
and the exchange fractions of a hybrid. QVF archives keep it in a
non-critical vendor section; a requested option alone creates no claim, only
executed work does. Periodic results (`PeriodicMANTAResult`) carry the
executed route in `run_fields` (`integral_backend_executed`,
`jk_method_executed`, `coulomb_algorithm`, `exchange_algorithm`, the domain
cutoffs and the Ewald $\alpha$) and the exchange zero-mode convention in
`exchange_q0`.

## Theory

**Gaussian integrals.** The product of two Gaussians is a Gaussian on a
point between their centres, so every two-electron integral over contracted
Gaussians reduces to integrals over Gaussian charge distributions
$\Omega_{ab}(\mathbf r) = \chi_a(\mathbf r)\chi_b(\mathbf r)$,

$$
(ab|cd) = \iint \frac{\Omega_{ab}(\mathbf r_1)\,\Omega_{cd}(\mathbf r_2)}{|\mathbf r_1 - \mathbf r_2|}\,\mathrm d\mathbf r_1\,\mathrm d\mathbf r_2 .
$$

MANTA expands each distribution in Hermite Gaussians and evaluates the
Coulomb interaction of Hermite functions through the Boys function, the
McMurchie-Davidson scheme ({vibeqc-cite}`mcmurchie_davidson_1978`); libint
uses Obara-Saika and Head-Gordon-Pople recursions
({vibeqc-cite}`obara_saika_integrals_1986`, {vibeqc-cite}`libint_valeev`).
The textbook treatment of both is {vibeqc-cite}`helgaker_jorgensen_olsen_2000`.
Integrals that are negligible are skipped with the Schwarz bound
$|(ab|cd)| \le (ab|ab)^{1/2}(cd|cd)^{1/2}$
({vibeqc-cite}`haser_ahlrichs_schwarz_1989`).

**Density fitting.** RI approximates each orbital product by an expansion in
an auxiliary basis $\{P\}$ whose coefficients minimise the Coulomb
self-interaction of the fitting error ({vibeqc-cite}`whitten_1973`,
{vibeqc-cite}`dunlap_1979`),

$$
(ab|cd) \approx \sum_{PQ} (ab|P)\,[\mathbf V^{-1}]_{PQ}\,(Q|cd),
\qquad V_{PQ} = (P|Q),
$$

which makes the error in $\mathbf J$ quadratic in the fitting error and the
energy error typically tens of microhartree with a JKFIT set
({vibeqc-cite}`eichkorn_rij_1995`, {vibeqc-cite}`weigend_jfit_2006`,
{vibeqc-cite}`weigend_jk_aux_2008`). `metric` controls how rank-deficient
parts of $\mathbf V$ are handled.

**Multipole far field.** Two well-separated charge distributions interact
through their multipole moments. BUPO groups shell pairs into bubbles,
computes their Cartesian multipoles up to a fixed order, and evaluates the
far-field Coulomb interaction from those moments while near pairs are
integrated exactly, with an explicit error ledger that bounds the far-field
and Gaussian-tail errors ({vibeqc-cite}`neese_bubblepole_2025`; the
fast-multipole background is {vibeqc-cite}`greengard_rokhlin_1987` and
{vibeqc-cite}`white_cfmm_1994`). RI-BUPO applies the same split to the
fitted three-centre coupling.

**Range separation.** Range-separated functionals split the Coulomb kernel,
$1/r = \operatorname{erf}(\omega r)/r + \operatorname{erfc}(\omega r)/r$,
and MANTA provides two-, three- and four-centre integrals for both parts so
that each exchange component uses its own kernel.

**Periodic Coulomb sums.** In a crystal the Hartree and nuclear terms are
conditionally convergent lattice sums. The Ewald split evaluates the short
range in real space and the long range in reciprocal space
({vibeqc-cite}`ewald_lattice_sum_1921`), which is what
`reciprocal_energy_cutoff`, `pair_cutoff_bohr`, `nuclear_cutoff_bohr` and
`alpha` control, and the integrable singularity of periodic exact exchange
at $\mathbf q = 0$ is treated with the probe-charge Ewald correction of the
BvK supercell ({vibeqc-cite}`gygi_baldereschi_exx_1986`), selected by
`exchange_exxdiv`.

## Limitations

- **Experimental, slower than libint, and not production-qualified.** Use
  the default route for production numbers.
- **Explicit domains on crystals.** Periodic MANTA has no automatic accuracy
  profile; converge the domains yourself.
- **Narrow workflow coverage.** Properties beyond those listed, response,
  correlation, solvation, dispersion and Hessians are refused on the
  selected route.

## See also

- [MANTA developer reference](manta.md), every primitive the engine exposes.
- [Density fitting](density_fitting.md), the production RI-J, RIJK and
  RIJCOSX routes.
- [Design: MANTA engine](../design/manta_engine.md),
  [scientific contract](../design/manta_scientific_contract.md) and
  [validation plan](../design/manta_validation_plan.md).
- [Periodic methods](periodic_methods.md), the default periodic J/K routes.
