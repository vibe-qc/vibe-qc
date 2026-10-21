# DLPNO methods (MP2, CCSD, CCSD(T))

DLPNO-MP2 is local second-order Møller-Plesset theory in the
domain-based pair-natural-orbital framework of Pinski, Riplinger,
Valeev and Neese (*J. Chem. Phys.* **143**, 034108 (2015)). Occupied
orbitals are Foster-Boys localised, every occupied pair gets a compact
virtual space of pair natural orbitals built inside its own projected
atomic-orbital domain, and the MP2 residual equations are solved with
the localised-orbital Fock coupling. The result tracks canonical
RI-MP2 at a fraction of the asymptotic cost, with controllable
truncation thresholds.

The molecular MP2 drivers diagonalize retained PNO Fock blocks through the
shared dense operator contract, checking residuals and orthogonality for
each pair and spin. The pair-density diagonalization still determines the
retained subspace using the selected threshold, with at least one PNO per
nonempty pair space. This preserves the semicanonical denominators,
truncation correction and coupled-pair equations. These dense pair solves
do not introduce a memory-budget guarantee.

The project-wide default freezes the chemical core and applies the supported
subset of Liakos *NormalPNO* on every restricted/unrestricted MP2 and CCSD
DLPNO solver. The chemical-core count follows ORCA 6.1 Table 2.69: ORCA's
frozen-electron count is converted to closed-shell spatial orbitals and summed
over the atoms. Use `frozen_core=False` on `run_job` for the uniform
all-electron escape corresponding to ORCA `NoFrozenCore`.

The implementation is count-only. It freezes that many lowest-energy
occupied orbitals, but does not reproduce ORCA's additional
atomic-orbital-character reorder for unusual canonical orderings. A
molecular ECP reference is accepted since 2026-09: the occupied space is
partitioned from the valence electron count the SCF result carries, and
the frozen-core table subtracts the ECP core per atom (see
[Effective core potentials](ecp.md#correlated-open-shell-and-determinant-methods-on-an-ecp-reference)).
`frozen_core=False` controls correlation-space freezing only; it does not
restore ECP-removed electrons.

DLPNO-MP2, DLPNO-CCSD and DLPNO-CCSD(T) through `run_job` are production
routes at molecular scale. Two things on this page are not: the
domain-restricted local density fitting is an experimental
reduced-scaling increment, and the dense O(N^6) "pilot" solvers are
correctness pilots rather than production methods; both are labelled
where they appear. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

## Theory

**Why local correlation.** Canonical MP2 correlates every occupied pair
$ij$ through every virtual pair $ab$, so the amplitude tensor
$t^{ij}_{ab}$ has $O(o^2 v^2)$ elements and the cost grows as $O(N^5)$
for MP2, $O(N^6)$ for CCSD and $O(N^7)$ for (T). In a gapped system the
correlation energy of a pair decays as $R_{ij}^{-6}$ with the distance
between the two occupied orbitals, and each pair needs only a small
virtual space once that space is adapted to the pair. The
domain-based local pair-natural-orbital method
({vibeqc-cite}`riplinger_dlpno_2013`; {vibeqc-cite}`riplinger_sparsemaps_2016`;
DLPNO-MP2: {vibeqc-cite}`pinski_dlpno_mp2_2015`) exploits both facts
and reduces the asymptotic cost to near-linear while keeping the
truncation under numerical control.

**Localised occupied orbitals and PAO domains.** The occupied orbitals
are Foster-Boys localised (`localise="boys"`). The virtual space is
spanned by projected atomic orbitals,
$\tilde\chi_\mu = (1 - \sum_i |i\rangle\langle i|)\,\chi_\mu$, which are
atom-centred, so a *domain* can be assigned to each localised orbital:
the atoms that have a projected atomic orbital whose differential overlap
integral with $i$ exceeds `tcut_do` (ORCA's `TCutDO`, the default
`domain_criterion="doi"`; the older Mulliken-population criterion at
`tcut_mkn` remains available, and `tcut_mkn=0` gives full domains under
either). A pair domain is the union of the two orbital domains, and every
pair quantity below lives in that PAO space; the
[PAO domains](#pao-domains) section below gives the thresholds and an
example where the two criteria differ.

**Pair prescreening.** Before any integrals are built, every pair is
classified from a multipole estimate of its correlation energy: the
dipole approximation of the pair energy decays as $R_{ij}^{-6}$ and is
evaluated from the LMO-PAO dipole integrals. A pair whose estimate is
below `tcut_pairs` is *distant*: its estimate is kept as `e_distant`
and it never enters the iterations. The remaining pairs are *strong*
(coupled equations in their PNO spaces); a custom
`tcut_pairs_weak > tcut_pairs` interval defines *weak* pairs that are
treated at the semicanonical MP2 level with the looser
`tcut_pno_weak`. This is the step that makes the pair list, and with it
the cost, grow linearly with the molecule.

**Pair natural orbitals.** For each surviving pair the semicanonical
MP2 amplitudes in the pair PAO domain define a pair density,

$$
D^{ij} = \tilde T^{ij\dagger}\, T^{ij} + \tilde T^{ij}\, T^{ij\dagger},
\qquad
\tilde T^{ij} = \frac{4\,T^{ij} - 2\,T^{ij\,\mathsf T}}{1 + \delta_{ij}},
$$

(Riplinger and Neese 2013, Eq. 23; the `pno_norm="mp2"` convention).
Diagonalising $D^{ij}$ gives the pair natural orbitals and their
occupation numbers; the PNOs with $n^{ij}_a \ge$ `tcut_pno` are kept,
typically ten to twenty per pair against hundreds of virtuals. The
energy the truncation discards is estimated at the semicanonical level
and added back as `e_pno_correction`,
$\Delta E^{\rm PNO}_{ij} = E^{\rm SC}_{ij}[\text{full domain}] - E^{\rm SC}_{ij}[\text{PNO}]$.
The strong-pair amplitudes are then iterated to convergence in the PNO
bases. Because the localised orbitals are not canonical, the occupied
Fock matrix couples different pairs, and the residual contains terms of
the form $\sum_k F_{ik}\, S^{ij,kj}\, T^{kj}\, S^{kj,ij}$ with overlap
matrices $S^{ij,kj}$ between the PNO spaces of different pairs; that
coupling is what `coupling_radius` truncates on the CCSD route. The
closed-shell energy is

$$
E_{\rm corr} = \sum_{ij} \sum_{ab \in [ij]} \left(2\,t^{ij}_{ab} - t^{ij}_{ba}\right) K^{ij}_{ab}
  + \sum_{ij} \Delta E^{\rm PNO}_{ij} + E_{\rm distant},
\qquad K^{ij}_{ab} = (ia|jb),
$$

with the exchange integrals density-fitted in the pair's PNO basis.

**Thresholds and the canonical limit.** The three thresholds
`tcut_pairs`, `tcut_pno` and `tcut_mkn` are the whole approximation.
The published LoosePNO / NormalPNO / TightPNO sets
({vibeqc-cite}`liakos_dlpno_thresholds_2015`; for the MP2-specific
family, {vibeqc-cite}`pinski_dlpno_mp2_thresholds_2019`) trade about
99.8 to 99.95 percent recovery of the canonical correlation energy
against cost. With every threshold at zero the PNOs span the full pair
virtual space, no pair is screened, and the equations are the canonical
ones written in a rotated basis, so the energy reproduces canonical
RI-MP2 (or CCSD, or CCSD(T)) with the same fitting basis to the
microhartree; that exactness limit is asserted in the test suite and is
the anchor every threshold is measured against. The truncation error is
smooth and systematic, not variational: a tighter threshold lowers the
energy toward the canonical value but a DLPNO energy is not an upper
bound on anything.

**Pair natural orbitals from semicanonical amplitudes.** The pair
density above is built from the semicanonical MP2 amplitudes of the
pair in its PAO domain, where the PAOs have been orthogonalised and
canonicalised so that the virtual Fock block is diagonal with energies
$\epsilon_{\tilde a}$:

$$
T^{ij}_{\tilde a \tilde b} = -\frac{K^{ij}_{\tilde a \tilde b}}
  {\epsilon_{\tilde a} + \epsilon_{\tilde b} - F_{ii} - F_{jj}},
\qquad
K^{ij}_{\tilde a \tilde b} = (i\tilde a|j\tilde b) = \sum_P B^P_{i\tilde a} B^P_{j\tilde b},
$$

with the localised-orbital diagonal Fock elements $F_{ii}$ in the
denominator. The PNOs are the eigenvectors of $D^{ij}$ and a PNO is
kept when its occupation number satisfies $n^{ij}_a \ge T_{\rm CutPNO}$
(`tcut_pno`); the diagonal pairs, whose space also carries the singles,
are cut at `tcut_pno_diag_scale * tcut_pno`. On the DLPNO-MP2 route the
prescreening estimate of a pair is the dipole-dipole approximation to the
same expression, which decays as $R_{ij}^{-6}$; on the local CCSD route
the criterion is the semicanonical MP2 pair energy in the full canonical
virtual space, $|\epsilon^{\rm MP2}_{ij}| < T_{\rm CutPairs}$
(`tcut_pairs`), and a pair below it keeps that MP2 energy instead of
being iterated.

**Coupled cluster.** DLPNO-CCSD keeps the amplitudes in the pair PNO
bases and solves the density-fitted closed-shell CCSD equations of
{vibeqc-cite}`stanton_gauss_watts_bartlett_ccsd_1991` in the
factorised form of {vibeqc-cite}`deprince_sherrill_df_ccsd_2013`. The
default `residual_domain="pno"` contracts every term of the residual in
the PNO space of the amplitude that carries the contracted index and
projects the free indices into the target pair through the PNO overlaps
$S^{ij,kl} = U_{ij}^{\mathsf T} S\, U_{kl}$ (Riplinger and Neese 2013,
Sec. II C, Eqs. 26 to 30); the atom-based extended PAO domain of the
same paper (`residual_domain="extended"`) is kept as the exactness
oracle, and `"full"` contracts in the whole virtual space. Pairs below
`tcut_pairs` stay at the MP2 level.

**The triples, (T0) and (T1).** The perturbative triples
({vibeqc-cite}`riplinger_triples_2013`; canonical form
{vibeqc-cite}`raghavachari_ccsdt_1989`) use triples natural orbitals,
the eigenvectors of the averaged density
$D^{ijk} = \tfrac13\,(D^{ij} + D^{ik} + D^{jk})$ truncated by
`tcut_tno`, one domain per occupied triple. In spin orbitals the
correction is

$$
E_{(T)} = \frac{1}{36} \sum_{ijk}\sum_{abc}
  t^{abc}_{ijk}(c)\, D^{abc}_{ijk}\,
  \bigl[t^{abc}_{ijk}(c) + t^{abc}_{ijk}(d)\bigr],
\qquad
D^{abc}_{ijk} = \epsilon_i + \epsilon_j + \epsilon_k - \epsilon_a - \epsilon_b - \epsilon_c ,
$$

with the connected triples $D\,t(c) = P(i/jk)P(a/bc)\bigl[\sum_d
\langle bc\|dk\rangle t^{ad}_{ij} - \sum_l \langle lc\|jk\rangle
t^{ab}_{il}\bigr]$ and the disconnected $D\,t(d) = P(i/jk)P(a/bc)\,
t^a_i \langle jk\|bc\rangle$. The denominator assumes a diagonal Fock
matrix. Localised occupied orbitals are not canonical, so the original
**(T0)** replaces $\epsilon_i$ by the diagonal elements $F_{ii}$ of the
localised occupied Fock matrix and neglects its off-diagonal elements:
a semicanonical approximation that costs 0.07 kcal/mol on water and
about 1 kcal/mol on C10H22 in the molecular table below. vibe-qc's
closed-shell default `triples_mode="t1"`
removes it by rotating the occupied space to canonical orbitals and
evaluating the full canonical (T) on the rotated PNO amplitudes, which
is exact in the occupied coupling but costs a canonical (T); the
compiled `triples_mode="local"` evaluates the (T0) with the spatial
W/V expression of {vibeqc-cite}`rendell_lee_komornicki_triples_1991`
in per-triple TNO domains and is the mode that scales. The iterative
local-basis (T1) of Guo and coworkers ({vibeqc-cite}`guo_dlpno_t1_2018`),
which keeps the triples in the localised basis and iterates the
off-diagonal coupling, is implemented only as the open-shell
`triples_mode="t1-iterative"`. For an open-shell UHF reference the
solver keeps independent alpha and beta localised orbitals and builds
spin-orbital pair PNO spaces ({vibeqc-cite}`saitow_openshell_dlpno_2017`
is the method-family source); its local (T0) follows
{vibeqc-cite}`guo_openshell_dlpno_triples_2020` and evaluates the
open-shell (T) expression of
{vibeqc-cite}`watts_gauss_bartlett_openshell_triples_1993`. The earlier
LPNO scheme without domains is {vibeqc-cite}`neese_lpno_2009`.

**Where it helps and where it fails.** DLPNO is the right tool for
large gapped molecules where canonical CCSD(T) is out of reach and a
few tenths of a kcal/mol of controlled truncation error is acceptable;
it is the wrong tool for delocalised systems (conjugated chains, small
clusters of metals), where pair correlation is genuinely long-ranged and
`coupling_radius` or `tcut_pairs` drop real physics (the H10 chain
example further down loses 90 percent of the triples energy to a
5 bohr radius). Interaction energies of weakly bound complexes are the
most threshold-sensitive quantity and want TightPNO. Strongly
multireference systems are outside single-reference coupled cluster
regardless of the local approximation.

## At a glance

| | |
|---|---|
| Entry point | `run_job(method="dlpno-mp2")` |
| Reference | closed-shell RHF, or UHF for open shells (auto-routed to DLPNO-UMP2) |
| Auxiliary basis | auto-resolved correlation ("ri") fit of the orbital basis |
| Public controls | `frozen_core=...`, `dlpno_thresholds="normal"|"tight"|"loose"` |
| Route-specific options | `dlpno_options=DLPNOMP2Options(...)` |
| Threshold default | Liakos NormalPNO, on every threshold the selected solver implements |
| Exactness limit | thresholds → 0 reproduces canonical RI-MP2 to ≤ 1 µHa |

## Quick start

```python
from vibeqc import run_job
from vibeqc._vibeqc_core import Atom, Molecule

mol = Molecule(
    [
        Atom(8, [0.000, 0.000, 0.000]),
        Atom(1, [0.000, 1.499, -1.160]),
        Atom(1, [0.000, -1.499, -1.160]),
    ],  # bohr
    charge=0,
    multiplicity=1,
)

result = run_job(mol, basis="def2-svp", method="dlpno-mp2", output="h2o")
print(result.dlpno_mp2.e_corr)   # DLPNO-MP2 correlation energy (Ha)
print(result.energy_total)       # RHF + correlation
```

This call uses the chemical-core/NormalPNO project default. The exact energy
therefore differs from output captured under the former all-electron,
MP2-specific defaults.

The `.out` file reports the energy decomposition, iterated pair
energies, the semicanonical PNO-truncation correction, and the
distant-pair dipole estimate:

```{note}
The block below is retained historical output from the pre-#140/#448
all-electron recipe (`tcut_pairs=1e-6`, `tcut_pno=1e-8`,
`tcut_mkn=1e-3`). It is not the output of the current quick-start defaults.
Its numbers remain here so archived calculations stay interpretable.
```

```
  DLPNO-MP2 (Pinski 2015; RI: def2-svp-rifit)
  ------------------------------------------------------------------------------
  E(RHF reference)       =   -75.9547597194 Ha
  pairs kept / screened  = 15 / 0   (frozen core: 0)
  avg PNOs per pair      =           14.9
  E(iterated pairs)      =    -0.2062668853 Ha
  E(PNO truncation corr) =    -0.0000032714 Ha
  E(distant-pair est.)   =     0.0000000000 Ha
  E(DLPNO-MP2 corr)      =    -0.2062701567 Ha
  E(DLPNO-MP2 total)     =   -76.1610298761 Ha
```

## Thresholds and options

```python
result = run_job(
    mol,
    basis="def2-svp",
    method="dlpno-mp2",
    frozen_core=False,        # ORCA NoFrozenCore / explicit all-electron
    dlpno_thresholds="tight", # published Liakos TightPNO
    output="h2o_tight",
)
```

For a low-level or route-specific control, build the option type from the same
named preset and then change only the extra knobs that route owns:

```python
from vibeqc.dlpno import options_from_dlpno_thresholds
from vibeqc.dlpno.mp2 import DLPNOMP2Options

opts = options_from_dlpno_thresholds(DLPNOMP2Options, "tight")
opts.local_df = True
result = run_job(mol, basis="def2-svp", method="dlpno-mp2",
                 frozen_core=False, dlpno_options=opts)
```

| Option | Default | Meaning |
|---|---|---|
| `tcut_pno` | 3.33e-7 | NormalPNO occupation cutoff (strong pairs). Smaller means more PNOs and a result closer to canonical. |
| `tcut_pno_weak` | 3.33e-6 | Stored project weak-pair companion. It is inactive for a named preset because that preset's `tcut_pairs_weak` equals the earlier screening boundary; a custom `tcut_pairs_weak > tcut_pairs` interval activates it. |
| `tcut_mkn` | 1e-3 | Mulliken threshold for domain atoms (0 = full domains). |
| `tcut_pairs` | 1e-4 | NormalPNO distant-pair screening threshold. Screened pairs contribute through `e_distant`; set `0` for the exactness configuration. |
| `frozen_core` (`run_job`) | `None` (`"published"`) | ORCA 6.1 Table 2.69 count-only convention. Set `False` for all-electron. Low-level option classes expose the resolved orbital count as `n_frozen`. |
| `localise` | `"boys"` | `"none"` keeps canonical occupieds (diagnostic). |
| `local_df` | `False` | Domain-restricted (local) density fitting, see below. |
| `fit_buffer` | 4.0 | Fit-domain extension (bohr) when `local_df=True`. |

Setting every threshold to zero is a supported validation mode: the
energy then reproduces canonical RI-MP2 with the same fitting basis to
≤ 1 µHa (this is asserted in `tests/test_dlpno_mp2.py`).

### The published threshold sets

The DLPNO literature names three threshold sets, tabulated in Liakos,
Sparta, Kesharwani, Martin & Neese, *J. Chem. Theory Comput.* **11**, 1525
(2015), [doi:10.1021/ct501129s](https://doi.org/10.1021/ct501129s), Table 1.
`vibeqc.dlpno` exports them as `DLPNOThresholds` records:

| Constant | Published name | `tcut_pairs` | `tcut_pno` | `tcut_mkn` | Recommended for |
|---|---|---|---|---|---|
| `DLPNO_LOOSE` | LoosePNO | 1e-3 | 1e-6 | 1e-3 | rapid estimates |
| `DLPNO_DEFAULTS` | NormalPNO | 1e-4 | 3.33e-7 | 1e-3 | general thermochemistry and kinetics |
| `DLPNO_TIGHT` | TightPNO | 1e-5 | 1e-7 | 1e-4 | noncovalent interactions, conformational equilibria |

NormalPNO is also the original DLPNO-CCSD default set of Riplinger &
Neese, *J. Chem. Phys.* **138**, 034106 (2013),
[doi:10.1063/1.4773581](https://doi.org/10.1063/1.4773581).

Which set to pick is a physics decision. Against CCSD(T₀)/cc-pVTZ on the
S66 noncovalent benchmark, Liakos 2015 § 4.3.2 measures MAD = 0.31 / 0.26 /
0.10 kcal/mol for Loose / Normal / Tight, and gives the reason: the
DLPNO-CCSD part of an interaction energy is *heavily* dependent on
`TCutPairs`, so the paper recommends TightPNO whenever weak interactions
are the target.

The project default is `"normal"`. Select a named set directly with
`run_job(..., dlpno_thresholds="loose"|"normal"|"tight")`. If an option
object is needed, use
`options_from_dlpno_thresholds(OptionType, "tight")`; do not copy the table
by hand.

Each of the six solver classes consumes only the thresholds its algorithm
implements. A named preset applies that supported subset and never attaches
decorative, unused metadata:

| Solver options | Applied preset fields |
|---|---|
| `DLPNOMP2Options` | `tcut_pairs`, `tcut_pno`, `tcut_mkn`; derived `tcut_pairs_weak` and `tcut_pno_weak` are stored but inactive for named presets; `tcut_do` under DOI domains |
| `DLPNOUMP2Options` | `tcut_pairs`, `tcut_pno` |
| `DLPNOCCSDPilotOptions` | `tcut_pno`, `tcut_mkn` |
| `DLPNOUCCSDPilotOptions` | `tcut_pno` |
| `LocalCCSDOptions` | `tcut_pairs`, `tcut_pno`, `tcut_mkn`; `tcut_do` under DOI domains |
| `LocalUCCSDOptions` | `tcut_pairs`, `tcut_pno`, `tcut_mkn` |

The `.out` method block reports the supported thresholds actually consumed.
An omitted field is unsupported by that solver, not silently set to zero. A
supported field that is inactive in the selected mode is reported as inactive
rather than applied; for example, UMP2 pair screening is inactive with
`localise="none"`. Closed-shell MP2's named weak-pair companions are likewise
inactive: pairs below the equal boundary have already been screened. They
participate only when a custom recipe sets `tcut_pairs_weak > tcut_pairs`.

The machine-readable artifacts carry the same distinction. The structured
completion event records the complete three-coordinate request in
`threshold_requested`, the published/derived subset in `threshold_applied`,
and every configured solver cutoff that actually participates in the route in
`threshold_settings`. The `.system` `[run]` table exposes the corresponding
flat `dlpno_tcut_*` fields, including active custom MP2 weak-pair cutoffs and
local-CC singles/TNO cutoffs. Unsupported or inactive values are empty rather than
invented as zero; `tcut_tno` is omitted for plain CCSD and exact-triples modes,
which do not consume a TNO threshold.

```{important}
The Liakos names and three-knob table were published for the DLPNO-CCSD
family. The name collision is real: Pinski & Neese 2019 Table I separately
defines DLPNO-MP2 LoosePNO / NormalPNO / TightPNO as `TCutPNO` = 1e-7 / 1e-8
/ 1e-9 and `TCutDO` = 2e-2 / 1e-2 / 5e-3
([doi:10.1063/1.5086544](https://doi.org/10.1063/1.5086544)). Pinski 2015's
DLPNO-MP2 validation likewise used MP2-specific settings. vibe-qc deliberately
uses the Liakos CC-family names as one cross-route provenance policy; this is
not a claim that its DLPNO-MP2 numerical presets reproduce the same-named
Pinski sets. Reproduce Pinski-era or pre-#140/#448 MP2 numbers with explicit
options.
```

Every DLPNO run records the supported thresholds it actually resolved in its own
`.out`, in a `truncation thresholds` row inside the method block, so the
truncation level behind a number is checkable from the artifact rather than
only from the script that produced it (issue #417):

```
  DLPNO-MP2 (Pinski 2015; RI: cc-pvdz-ri)
  ---------------------------------------
  E(RHF reference)       =   -76.0268440874 Ha
  Truncation thresholds  = tcut_pairs 1e-06  tcut_pno 1e-08  tcut_mkn 0.001  ...
  pairs kept / screened  = 15 / 0   (frozen core: 0)
```

This is the same pre-#140/#448 all-electron/old-MP2-threshold artifact shown
above; it is retained as compatibility evidence, not as current-default
output.

### PAO domains

Each localised occupied orbital gets an atom domain, and a pair's projected
atomic orbital (PAO) domain is the union of its two orbitals' domains. The
closed-shell DLPNO-MP2 and local DLPNO-CCSD routes and the open-shell local
DLPNO-UCCSD route choose those domains by
differential overlap (`domain_criterion="doi"`, the default): an atom joins an
orbital's domain when one of its PAOs has a differential overlap integral
with the orbital above `tcut_do` (Pinski, Riplinger, Valeev and Neese 2015,
Eq. 4). This is the criterion ORCA uses, and the presets carry ORCA 6.1.1's
`TCutDO` of 2e-2 / 1e-2 / 5e-3 for Loose / Normal / Tight, the same values
Pinski & Neese 2019 tabulate for DLPNO-MP2. `domain_criterion="mulliken"`
selects the older Mulliken-population domains at `tcut_mkn`, which every
result recorded before the DOI default used. `tcut_mkn=0` keeps every atom
under either criterion; it remains the full-domain exactness setting.

The two criteria differ where a Mulliken population is tiny next to real
density. NH3's Foster-Boys lone pair carries 1.7e-4 Mulliken population on
each hydrogen, so at `tcut_mkn=1e-3` its domain is the nitrogen alone. At
NormalPNO/cc-pVDZ that costs 0.34 kcal/mol in the DLPNO-CCSD correlation
energy against canonical DF-CCSD; the DOI domain keeps the hydrogens and the
error is 0.022 kcal/mol, as for ORCA 6.1.1 (0.021). The open-shell local
CCSD route judges each spin orbital against its own spin's PAOs; DLPNO-UMP2
builds no PAO domains.

Before v0.15.x these three constants carried in-house values that matched
no publication: `DLPNO_TIGHT` was `tcut_pairs` 5e-5 / `tcut_pno` 1e-8 /
`tcut_mkn` 1e-3, agreeing with published TightPNO on none of the three
knobs and screening pairs five times more loosely. They now carry the
published numbers (issue #416).

### Local density fitting (experimental, reduced-scaling)

```{admonition} Status: experimental
:class: warning

Domain-restricted local density fitting (`DLPNOMP2Options(local_df=True)`) is an
experimental route. It is off by default, it complements rather than replaces the
global RI path, and its per-pair fit dimension is reported in
`result.dlpno_mp2.fit_dim_per_pair` so the truncation is visible. Its numbers are
research output, not validated results, and it is not part of the release gate.
The label vocabulary is defined in [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

With `local_df=True`, each pair's exchange integrals are refit using
only the auxiliary functions within `fit_buffer` bohr of the pair's
domain, instead of the global RI metric, the per-pair fit cost then
stays bounded by the local neighbourhood as the molecule grows (the
foundation of linear scaling; `result.dlpno_mp2.fit_dim_per_pair`
records the per-pair fit dimension). A full fit domain reproduces the
global RI exactly. This is the first reduced-scaling increment; it
currently complements rather than replaces the global path (default
`local_df=False`), and the end-to-end speedup arrives with the
integral-direct local build. Remaining work is tracked in the canonical
[GitLab issue list](https://github.com/vibe-qc/vibe-qc/issues).

## Result object

`result.dlpno_mp2` carries the full decomposition:
`e_corr = e_corr_iterated + e_pno_correction + e_distant`, per-pair
energies (`pair_energies`, absolute occupied indices), PNO counts per
pair (`pno_per_pair`), per-pair fit dimensions (`fit_dim_per_pair`),
screening statistics, and the iteration trace.

`result.dlpno_ccsd` additionally names the correlation source the engine
read: `source_kind` (the adapter, `"molecular-df"` for a molecule,
`"gamma-gdf"` or `"bvk-supercell"` on a crystal) and `source_identity`,
the SHA-256 of the provenance record that defines the reference and its
three-centre factors (geometry, bases, reference kind and energy,
conventions). `run_job` writes both to the `.system` manifest as
`dlpno_source_kind` and `dlpno_source_identity` and to the structured
log, so a correlation energy names the Hamiltonian it was computed on.
The contract itself is `vibeqc.dlpno.source.CorrelationSource`; see the
[design page](../design_dlpno_correlation_source.md). With
`LocalCCSDOptions(report_extended_domain=True)` the result also carries
`avg_extended_domain`, described in the next section.

## DLPNO-CCSD and DLPNO-CCSD(T)

**`method="dlpno-ccsd"`** runs the **local solver**
(`dlpno.ccsd_local_solver`): amplitudes remain in pair PNO bases, and the
default `residual_domain="pno"` solves the amplitude equations in the
compiled `dlpno_ccsd_pno_solve` kernel, which contracts every term of the
closed-shell residual in the PNO space of the amplitude that carries the
contracted index (Riplinger and Neese 2013, Sec. II C) and projects the free
indices into the target pair through the PNO overlaps. Singles products are
folded into the density-fitting factors over the full virtual space, so
nothing is truncated before it is contracted; the cost per iteration is
O(n_pairs nL² n_aux n_pno²) and does not grow with the extended-domain
dimension. It is FCI-anchored: in the full-domain limit it reproduces
canonical closed-shell CCSD (≤ 1 µHa; on H₂ that equals FCI), and at any
truncation it reproduces the extended-domain oracle
(`residual_domain="extended"`, Riplinger and Neese's atom-based extended PAO
domain contracted once per group and projected back) to the convergence
tolerance. The project default applies the supported subset of NormalPNO.
The singles of each occupied orbital live in its diagonal pair's PNO space,
and those diagonal pairs are cut at `tcut_pno_diag_scale * tcut_pno`
(default 0.03, ORCA's `TCutPNOSingles` / `TCutPNO` ratio): the semicanonical
PNO correction cannot recover singles truncation, and at scale 1.0
n-butane/cc-pVDZ NormalPNO CCSD sits 1.62 kcal/mol above canonical against
1.08 at 0.03 (ORCA 6.1.1: 1.16).
Historical recovery percentages later on this page are labelled with the
explicit all-electron recipes that produced them.

```python
from vibeqc import run_job
result = run_job(mol, basis="def2-svp", method="dlpno-ccsd",
                 dlpno_thresholds="tight",
                 output="h2o_cc")
print(result.dlpno_ccsd.e_corr, result.energy_total)
```

Each pair couples only to occupied orbitals within `coupling_radius` (bohr)
of either pair index, the lever that takes the occupied coupling sums from
O(N⁴) to O(N²). The **12-bohr default is a bit-identical no-op on any
molecule under ~12-bohr extent** (the common case); on larger systems it
keeps the dropped long-range coupling well below the PNO truncation error
(~7 µHa on a 25-bohr H₂ chain vs the ~0.1% PNO error, the same
controlled-locality bargain as the default sparse pair list). Set
`coupling_radius=0` for full coupling, the exact reference the full-domain
ratchet pins. `result.dlpno_ccsd` reports `avg_coupled_occ`, the mean
local-set size, which saturates as the system grows at fixed radius
(`tests/test_dlpno_ccsd_solver.py::TestSparseCoupling`). The virtual-side
counterpart is `avg_extended_domain`: the mean dimension of the atom-based
extended domain, the PAOs on every atom in the domain of any orbital in a
pair's coupling set. Its ratio to the number of virtuals falls below 1 only
once the molecule is wider than a few such domains. The default
`residual_domain="pno"` route does not build that domain, so it reports
the dimension only with `LocalCCSDOptions(report_extended_domain=True)`,
a diagnostic that changes no energy.

**`method="dlpno-ccsd(t)"`** runs the same local solver plus the closed-shell
**(T1)** triples (`dlpno.triples_local`, `triples_mode="t1"`) on the converged
amplitudes. The occupied space is rotated to canonical orbitals and, at the
default `tcut_tno=0`, the full canonical (T) is evaluated on the rotated PNO
amplitudes in the compiled kernel. That removes both the (T0) semicanonical
approximation and any TNO truncation, so the PNO truncation of the amplitudes
is the only approximation left in the triples. The price is canonical (T)
cost: O(o^3 v^4) work on o^2 v^2 amplitudes, so (T1) is not a reduced-scaling
triples phase. `coupling_radius` does not screen these triples, because the
rotated orbitals are delocalised and a distance between localised orbitals
means nothing for them. Before f6cc3977 it did screen them, which lost 18% of
the C6H14/cc-pVDZ (T) (#506). The open-shell solver separately
exposes ``triples_mode="t1-iterative"``, a spin-orbital generalization of Guo
et al.'s 2018 closed-shell Eq. (2); their 2020 paper is the dedicated
open-shell method-family source. This opt-in correction currently builds
dense spin-orbital source tensors under the solver's ``max_nbf`` cap; it is
an experimental correctness/generalization route, not a reduced-scaling
triples phase. Set ``triples_mode="local"``
for the older DLPNO-(T0) (diagonal localised Fock, ~0.1 kcal/mol looser).
It runs in a compiled kernel that projects, for each occupied key, only the
pairs containing one of its occupieds and forms only that key's integral
blocks, so it is the closed-shell triples mode that scales to large
molecules; the default ``"t1"`` rotates to canonical occupieds and holds
o^2 v^2 amplitudes.
At full domains with canonical occupieds it reproduces canonical CCSD(T)
**exactly** (<= 1 nHa, the (T) parity ratchet).  The (T) vanishes
identically for two-electron systems.

The engine is labelled `local PNO-space (M3c)` in the `.out` block header,
in the `.system` manifest (`dlpno_engine`) and in the structured log, and
the block's note says "PNO-space local solver". The label names what the
solver does, contract every residual term in per-pair PNO spaces; it is
not a scaling claim, because how the cost grows with molecule size is
still being measured (the profiles are in [Accuracy](#accuracy)). The
closed-shell `triples_mode` values and what the artifacts call them:

| `triples_mode` | Artifact name | What runs | Scales? | Runtime citation |
|---|---|---|---|---|
| `"t1"` (default) | `rotated-occupied (T1) compatibility correction` | canonical (T) on the rotated PNO amplitudes; every canonical triple at `tcut_tno=0` | no, canonical (T) cost | none beyond the route |
| `"local"` | `DLPNO-(T0), diagonal local occupied-Fock` | compiled per-key TNO-domain (T0); `coupling_radius` screens the triple keys | yes | {vibeqc-cite}`rendell_lee_komornicki_triples_1991` |
| `"exact"` | `canonical (T) contraction on local amplitudes` | canonical (T) in the full virtual space after diagonalising the active occupied and virtual Fock blocks; the oracle | no, O(N^7) | none beyond the route |

The `residual_domain` values are `"pno"` (default, compiled, exact against
the extended oracle at the same coupling set), `"extended"` (the atom-based
extended PAO domain, the oracle), `"full"` (the whole virtual space, the
stationarity condition of the subspace ansatz) and `"pair"` (the historical
target-pair-first projection, which is wrong under truncation and kept only
to reproduce old numbers). The open-shell solver's values are `"pno"` and
`"union"`.

### Larger systems: the SCF reference and memory

DLPNO keeps its amplitudes in per-pair PNO spaces, but it runs on top of
a mean-field SCF reference, and that reference is what sets the memory
footprint on larger systems:

* **The DLPNO step holds no dense ``n_occ^2 n_virt^2`` or ``n_mo^4``
  tensor** (per-pair PNO domains). n-octane / cc-pVTZ (492 basis
  functions) DLPNO-MP2 peaks at a few GB, not hundreds. CCSD profiles through
  C40 and paired `coupling_radius=0` controls at C20 and C40 are measured
  below. The C60 controls remain open on #190; these results do not yet
  establish a size-scaling exponent.

* **Use a density-fitted SCF reference for larger systems.** Set
  ``density_fit=True`` (with an ``aux_basis``) on the reference options. A
  conventional in-core 4-index SCF would materialise the ``n_basis^4`` ERI
  tensor (~436 GB at 492 functions); the integral-direct SCF (the default
  above 200 functions) avoids it, and density fitting is faster still:

  ```python
  from vibeqc import run_job
  run_job(mol, basis="cc-pvtz", method="dlpno-mp2",
          density_fit=True, aux_basis="cc-pvtz-jkfit")  # ~13 GB
  ```

* **The compiled PNO-space CCSD route holds static factors per kept pair.**
  Each strong pair keeps its ``(ab|P)`` factor in its PNO basis and one
  ``(ma|P)`` factor row per coupled occupied, so the resident size is about
  ``n_pairs * n_aux * (n_pno^2 + n_coupled * n_pno) * 8`` bytes, plus the
  global ``(vw|P)`` factor (``n_aux * n_vir^2 * 8``) and the density-fitting
  object's own AO tensors. Measured on all-trans alkanes/cc-pVDZ at the
  default thresholds: 1.2 GB at C8H18 (202 functions), 3.3 GB at C12H26
  (298) and 6.8 GB at C16H34 (394). ``LocalCCSDResult.static_factor_bytes``
  reports the actual number.

The **O(N^6) pilots** (``DLPNOCCSDPilotOptions`` and
``DLPNOUCCSDPilotOptions``, opt-in) are a different animal. In the words
of their own module docstring they are *correctness pilots, not
production methods*: the small-system oracles the local solvers are
validated against. They form dense full-virtual-space integrals (memory
~ ``n_mo^4``, set by the basis size and independent of ``tcut_pno``) and
are hard-capped at ``max_nbf=64`` for that reason; raising the cap is a
deliberate override, not a scaling claim. Do not use them for production
runs; the default ``run_job(method="dlpno-ccsd")`` already uses the
local solver, whose ``max_nbf`` guard is 1500.

### Accuracy

Measured at the shipped defaults (chemical frozen core, NormalPNO,
`pno_norm="mp2"`, `residual_domain="pno"`, closed-shell `(T1)`) against
canonical DF-CCSD(T) in the same cc-pVDZ basis with the same frozen core, on
17 closed-shell molecules. ORCA 6.1.1 ran the same geometries, basis and
frozen core with its `NormalPNO` and `TightPNO` keywords; its default
triples are the semicanonical (T0). The last column is vibe-qc's own (T0),
`triples_mode="local"`, on the NormalPNO amplitudes. Errors in kcal/mol,
CCSD / CCSD(T); the alkanes are all-trans chains, the dimers are S22
geometries, and the inputs are recorded on #190 and #506.

| | vibe-qc NormalPNO | ORCA NormalPNO | vibe-qc TightPNO | ORCA TightPNO | vibe-qc NormalPNO, `triples_mode="local"` |
|---|---:|---:|---:|---:|---:|
| H₂O | +0.016 / +0.019 | +0.022 / +0.093 | +0.002 / +0.002 | +0.001 / +0.070 | +0.016 / +0.090 |
| NH₃ | +0.019 / +0.023 | +0.021 / +0.091 | +0.005 / +0.006 | +0.004 / +0.070 | +0.019 / +0.092 |
| CH₄ | +0.018 / +0.021 | +0.022 / +0.070 | +0.002 / +0.003 | +0.002 / +0.049 | +0.018 / +0.068 |
| HF | +0.008 / +0.009 | +0.012 / +0.065 | +0.000 / +0.000 | 0.000 / +0.052 | +0.008 / +0.063 |
| CO | +0.026 / +0.028 | +0.040 / +0.588 | +0.000 / +0.000 | +0.001 / +0.549 | +0.026 / +0.579 |
| N₂ | +0.016 / +0.017 | +0.050 / +0.581 | +0.000 / +0.000 | +0.003 / +0.533 | +0.016 / +0.552 |
| H₂CO | +0.063 / +0.079 | +0.053 / +0.499 | +0.023 / +0.029 | +0.018 / +0.453 | +0.063 / +0.507 |
| n-butane | +1.084 / +1.594 | +1.161 / +2.816 | +0.630 / +0.918 | +0.317 / +1.608 | +1.084 / +2.658 |
| NH₃, S22-01 monomer A | +0.019 / +0.023 | +0.021 / +0.090 | +0.005 / +0.006 | +0.003 / +0.069 | +0.019 / +0.091 |
| NH₃, S22-01 monomer B | +0.019 / +0.023 | +0.021 / +0.090 | +0.005 / +0.006 | +0.003 / +0.069 | +0.019 / +0.091 |
| (NH₃)₂, S22-01 | +0.064 / +0.123 | +0.039 / +0.255 | +0.029 / +0.038 | +0.015 / +0.163 | +0.064 / +0.270 |
| benzene | +0.443 / +0.595 | +0.583 / +2.242 | +0.263 / +0.339 | +0.196 / +1.699 | +0.443 / +1.995 |
| (H₂O)₂, S22-02 | +0.070 / +0.098 | +0.084 / +0.270 | +0.031 / +0.039 | +0.034 / +0.191 | +0.070 / +0.252 |
| (CH₄)₂, S22-08 | +0.017 / +0.060 | +0.024 / +0.177 | +0.015 / +0.020 | +0.019 / +0.118 | +0.017 / +0.157 |
| C₆H₁₄ | +0.575 / +0.827 | +0.611 / +1.609 | +0.319 / +0.434 | +0.185 / +0.936 | +0.575 / +1.471 |
| C₈H₁₈ | +0.834 / +1.214 | +0.868 / +2.260 | +0.455 / +0.629 | +0.252 / +1.274 | +0.834 / +2.065 |
| C₁₀H₂₂ | +1.121 / +1.632 | +1.161 / +2.951 | +0.595 / +0.831 | +0.335 / +1.631 | +1.121 / +2.694 |
| MAE | 0.259 / 0.376 | 0.282 / 0.868 | 0.140 / 0.194 | 0.082 / 0.561 | 0.259 / 0.806 |

What the numbers say:

* **CCSD matches ORCA at NormalPNO** (MAE 0.26 against 0.28 kcal/mol). At
  TightPNO the small molecules match too, but on n-butane and the alkanes
  the vibe-qc CCSD error is about twice ORCA's. The n-butane probes below
  identify the PNO cutoff as the largest measured sensitivity; confirmation
  on longer chains remains open (#190).
* **CCSD(T) is closer to canonical than ORCA's default** because the
  closed-shell `(T1)` evaluates the canonical (T) on the PNO amplitudes
  while ORCA's (T0) keeps the semicanonical approximation: on C₁₀H₂₂ at
  NormalPNO it recovers 98.4 % of the canonical (T) against 94.3 %. At
  TightPNO the corresponding recoveries are 99.2 % and 95.9 %. With
  `triples_mode="local"` vibe-qc's own (T0) lands at ORCA's level
  (MAE 0.81 against 0.87).
* **The `(T1)` costs a canonical (T).** C₁₀H₂₂/cc-pVDZ took 16 hours on 16
  cores, against 11 minutes for the whole run with `triples_mode="local"`.

The C₁₀H₂₂ TightPNO result and the following controls were collected on
2026-10-10. NormalPNO n-butane and benzene reproduce all four recorded
energies (HF, CCSD correlation, triples and total) exactly between the
pre-fix `3d9cf71d` and post-fix `1a80e229` builds, using eight threads in
both runs. These are implementation-workstream measurements; the
independent verification of the canonical-triple screening fix remains
open on #506.

One-coordinate TightPNO probes on the same n-butane geometry, basis and
frozen core, all on `1a80e229` with four threads:

| Change from TightPNO | CCSD error, kcal/mol | CCSD(T) error, kcal/mol |
|---|---:|---:|
| None, same-build baseline | +0.630 | +0.918 |
| `tcut_pno=1e-8` | +0.140 | +0.212 |
| `tcut_do=0` | +0.630 | +0.918 |
| `tcut_pairs=0` | +0.630 | +0.918 |
| `residual_domain="extended"` | +0.630 | +0.918 |
| `tcut_pno_diag_scale=0.003` | +0.546 | +0.812 |

The domain, pair-screening and residual-route probes leave the n-butane energy
unchanged to the displayed precision. Tightening the singles-space cutoff
helps less than tightening the pair PNO cutoff.

The corresponding C6H14 probes, collected on 2026-10-11, use the same
geometry, basis and frozen core as the accuracy table, build `ed9bfb42`,
16 threads and the canonical `(T1)` correction:

| Change from TightPNO | CCSD error, kcal/mol | CCSD(T) error, kcal/mol | Whole-job wall, s |
|---|---:|---:|---:|
| None, same-build baseline | +0.319 | +0.434 | 1326 |
| `tcut_pno=1e-8` | +0.102 | +0.147 | 1372 |
| `tcut_do=0` | +0.311 | +0.425 | 1328 |
| `tcut_pairs=0` | +0.317 | +0.420 | 1337 |
| `residual_domain="extended"` | +0.319 | +0.434 | 4003 |
| `tcut_pno_diag_scale=0.003` | +0.293 | +0.405 | 1325 |

The C6 baseline reproduces the earlier `1a80e229` correlation and triples
energies within 1e-15 Ha. As on n-butane, the pair PNO cutoff is the
largest measured source of truncation error among these controls. Full
domains and retaining all pairs help slightly; the extended residual
reproduces the baseline energy but takes about three times as long for the
whole job. These two molecules do not establish a new preset, and the
shipped thresholds are unchanged.

The NormalPNO triples timing controls use the same `1a80e229` build,
machine and 16 threads for both modes. Whole-job wall times, rounded to
seconds:

| Molecule | Local (T0), s | Canonical (T1), s | (T0) CCSD(T) error, kcal/mol | (T1) CCSD(T) error, kcal/mol |
|---|---:|---:|---:|---:|
| C₆H₁₄ | 158 | 1309 | +1.471 | +0.827 |
| C₈H₁₈ | 361 | 8627 | +2.065 | +1.214 |
| C₁₀H₂₂ | 658 | 58679 | +2.694 | +1.632 |

Their CCSD correlation energies agree within 3.2e-14 Ha. These timings
measure the two triples choices on these jobs, not the size scaling of
the CCSD kernel. That requires the C20/C40/C60 ladder and its
`coupling_radius=0` controls on #190.

The first radius control is complete for C20H42/cc-pVDZ, NormalPNO, chemical
frozen core, without triples (`ac46948b`, 48 threads, 17 iterations):

| `coupling_radius`, bohr | Mean extended domain / virtuals | Mean coupled occupied / active | Setup, s | Iteration total, s | Peak RSS, GB |
|---|---:|---:|---:|---:|---:|
| 12 | 288.3 / 409 | 29.2 / 61 | 22.17 | 225.33 | 30.7 |
| 0, unrestricted | 409 / 409 | 61 / 61 | 25.67 | 544.99 | 33.0 |

The CCSD correlation energies are -3.160389344536 and -3.160375865934 Ha,
respectively. The C40H82 control uses the same build, basis and thresholds,
96 threads and **only three CCSD iterations** in each run:

| `coupling_radius`, bohr | Mean extended domain / virtuals | Mean coupled occupied / active | Setup, s | Seconds / iteration | Static factors, GB | Peak RSS, GB |
|---|---:|---:|---:|---:|---:|---:|
| 12 | 326.6 / 809 | 31.2 / 121 | 219.28 | 63.21 | 61.1 | 194.4 |
| 0, unrestricted | 809 / 809 | 121 / 121 | 261.16 | 305.52 | 100.0 | 228.0 |

These bounded probes are not converged energy comparisons. The unrestricted
control takes 4.83 times as long per iteration on these allocations. Both
retain 1054 pairs, screen 6327 pairs and average 23.6 PNOs per retained pair.

Separate profiles with the 12-bohr radius converge on the same build with
96 threads:

| Molecule | Iterations | Mean extended domain / virtuals | Mean coupled occupied / active | Setup, s | Seconds / iteration | Static factors, GB | Peak RSS, GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| C24H50 | 17 | 301.3 / 489 | 29.9 / 73 | 29.99 | 16.14 | 20.1 | 56.6 |
| C32H66 | 17 | 317.2 / 649 | 30.7 / 97 | 92.37 | 35.07 | 37.9 | 115.6 |
| C40H82 | 19 | 326.6 / 809 | 31.2 / 121 | 227.62 | 66.39 | 61.1 | 193.1 |

Local domain and coupling counts grow slowly across these profiles, while
resident factors, peak memory and iteration cost continue to grow. GB here
means 10^9 bytes. These are separate scheduler allocations; C20 also used
fewer threads and a different iteration count from the bounded C40 controls.
The C60 controls and a matched timing protocol remain necessary before
claiming a scaling exponent.

The historical def2-SVP, all-electron, `tcut_mkn=0`,
`residual_domain="pair"` recipe (MAE 0.37 kcal/mol on seven small
molecules; 0.17 at `tcut_pno=1e-8`) is still reproducible with
`examples/molecular/benchmark-dlpno-ccsd-t.py`, but it measures that
recipe, not the defaults above.

**Pair screening** (``tcut_pairs=1e-4``, the published NormalPNO
``TCutPairs`` coordinate, default-on)
treats a pair whose full-virtual MP2 energy is below the threshold at MP2
level instead of CCSD: accuracy-neutral on compact molecules (a no-op on
5-occupied cases, the weak ~10-14 % tail on larger ones) and a ~2x
speedup on extended systems; set ``tcut_pairs=0`` to disable.

### Closed-shell `(T1)`: choosing `tcut_tno`

`LocalCCSDOptions.tcut_tno` defaults to `0.0`, the full TNO union span, so a
default `dlpno-ccsd(t)` job performs no triples-domain truncation. If you opt
into a positive value, these are the measured trade-offs on H2O/def2-SVP
(2026-07-29; full domain `e_t = -0.003009059087680666` Ha at 19.0 TNOs per
triple, which is the entire virtual space):

| `tcut_tno` | error vs full domain | mean TNOs | smallest domain | collapsed triples |
|---|---|---|---|---|
| `1e-9` | `-0.000` µHa | 18.52 | 7 | 0 |
| `1e-7` | `+0.018` µHa | 18.20 | 4 | 0 |
| `1e-6` | `+0.086` µHa | 17.80 | 4 | 0 |
| `1e-5` | `+33.39` µHa | 15.68 | 1 | 1 |
| `1e-4` | `+582.45` µHa | 10.68 | 1 | 1 |

The closed-shell route tolerates much coarser thresholds than the open-shell
engine: `1e-6` costs under `0.1` µHa here, where the open-shell `(T1)` costs
30 to 64 µHa at the same setting.

The error curve breaks exactly where the first triple collapses. A spatial
triple excitation promotes three electrons into three *distinct* virtuals, so
a domain retaining fewer than three cannot host one and contributes exactly
zero rather than approximately. Between `1e-6` and `1e-5` one triple falls to
a single virtual and the error jumps 400-fold. `LocalCCSDResult` reports this
directly:

```python
result = run_local_dlpno_ccsd(mol, basis, rhf, df, options)
result.avg_tno                    # mean retained domain per distinct triple
result.n_degenerate_tno_triples   # triples truncated below three virtuals
result.tno_per_triple             # per-triple domain sizes
```

Treat a nonzero `n_degenerate_tno_triples` as a signal that `tcut_tno` has
truncated past the point where part of the correction exists, rather than as
a merely coarse setting. `triples_mode="exact"` expands to the full virtual
space and builds no TNO domain, so it reports an empty `tno_per_triple`.
It diagonalises the Fock within the active occupied and virtual blocks
before evaluating the canonical correction, including when a source adapter
has already localised the orbitals. Frozen orbitals stay outside that rotation.

A `run_job(method="dlpno-ccsd(t)")` run reports the same thing without any
Python bookkeeping. The `.out` block carries a domain line beside the pair
line, and a collapsed domain raises a warning:

The numerical block below is retained pre-#140/#448 all-electron evidence.
Its pair and virtual counts are not current-default counts.

```text
  pairs / avg PNOs         = 15 / 19.0   (frozen core: 0)
  E(CCSD correlation)      =    -0.2130338714 Ha
  triples / avg TNOs       = 25 / 15.7   (smallest domain: 1)
  E((T) correction)        =    -0.0029756723 Ha
  ...
  WARNING: tcut_tno truncated 1 of 25 triples below three virtuals; those
  contribute exactly zero to (T), so part of the correction is missing rather
  than approximated. Lower tcut_tno (0 keeps the full domain).
```

For local `(T0)`, `coupling_radius` is the occupied-side counterpart: it
drops whole localised occupied triple keys rather than shrinking each
domain. The counts expose this approximation:

```text
  triples / avg TNOs       = 9 / 5.0   (smallest domain: 5)   (16 screened by coupling_radius)
```

`LocalCCSDResult.n_triple_keys` and `.n_triple_keys_screened` carry the same
numbers. Closed-shell `(T1)` rotates to canonical occupied orbitals and
evaluates every canonical triple key; localised-orbital distances do not
screen those keys. Earlier measurements of distance-screened `(T1)` used
the incorrect pre-#506 behaviour and do not describe the current route.
The radius still controls the local CCSD coupling sets that produce the
amplitudes used by either triples mode.

The `dlpno_ccsd_done` structured-log event carries `n_tno_triples`,
`avg_tno_per_triple`, `min_tno_per_triple`, `n_degenerate_tno_triples` and
`n_triple_keys_screened` alongside the existing pair fields. The open-shell
local route reports its corresponding pair/PNO and, when requested,
triple/TNO diagnostics. The opt-in dense UCCSD(T) pilot builds no TNO domain,
so that pilot omits the domain line rather than printing zeros.

## Crystals: the GDF route at Gamma and on a mesh

The dedicated guide, [Local correlation on crystals](periodic_local_correlation.md),
carries the option tables, every refusal with its message, the output and
QVF fields, the measured mesh series and the finite-size and folding
theory; this section is the summary.

The front door is `run_periodic_job(system, basis, method="dlpno-ccsd(t)")`
(or `"dlpno-ccsd"`), with `kpoints=(2, 2, 2)` for a mesh and
`dlpno_ccsd_options=`, `dlpno_thresholds=` and `frozen_core=` meaning what
they mean on `run_job`. The job runs an RHF or UHF GDF reference on the
k-mesh engine with its classic fit retained, correlates it through the
adapters described below, and writes the result everywhere the SCF goes:
the `.out` gets a `DLPNO-CCSD(T) ... on the periodic GDF reference` block
with the source identity and the energies per cell, the `.system` manifest
gets the `dlpno_*` fields and `correlation_route`, the structured log a
`dlpno_ccsd_done` event, the citations the DLPNO sources plus the triples
kernel that ran, and the QVF an `x_vibeqc.dlpno_correlation` section. The
returned result forwards the SCF result and carries `dlpno_ccsd` (the
engine result), `periodic_dlpno` (totals and per-cell energies) and
`correlation`; `energy_total` is the correlated total per cell. The front
door refuses before any SCF work, with the reason: a functional,
`dim < 3`, a BIPOLE or other non-GDF `jk_method`,
`symmetry_reduce_k`, smearing, `optimize`, `hessian`, `dft_plus_u`, a
`KPoints` object in place of the mesh tuple, and a `fit_source` other than
`auto` or `classic`.

Non-singlet cells select KUHF and the molecular open-shell engine,
`run_local_dlpno_uccsd`. Use `LocalUCCSDOptions` for
`dlpno_ccsd_options`; explicitly passing that type also selects UHF for a
singlet. The library route accepts `uhf_options` for the reference SCF.
The two spin Fock matrices keep their SCF exchange zero-mode terms and
share the retained spin-independent fit. Each spin's active occupied block
is localised separately, preserving the frozen and virtual blocks. On an
N-cell BvK mesh, charge, electron counts and the frozen-core count scale by
N; multiplicity becomes `1 + N * (multiplicity_cell - 1)`.
The artifacts record `dlpno_reference_kind` and total per-spin occupied
and virtual counts, with citation routes `periodic-gdf-dlpno-uccsd[(t)]`.
`periodic_dlpno.n_occ` and `n_vir` count spatial orbitals on RHF and spin
orbitals on UHF; `n_occ_alpha`, `n_occ_beta`, `n_vir_alpha` and
`n_vir_beta` are explicit on both routes.

This UHF implementation retains independent alpha/beta spaces. It does not
implement the shared ROHF/QRO orbitals and NEV-PNO construction of
[Saitow et al. (2017)](https://doi.org/10.1063/1.4981521).
The finite-mesh to Gamma-supercell equivalence is also used in
[Ye and Berkelbach (2024)](https://doi.org/10.1021/acs.jctc.4c00936), whose
LNO fragment approximation assumes a spin-restricted reference. Here the
existing pair-specific DLPNO engine solves the whole folded supercell.
The translational home-cell pair reduction of
[Nejad et al. (2025)](https://doi.org/10.1063/5.0290816) remains a later
increment, separate from this spin extension.

The library route,
`vibeqc.periodic_dlpno.run_periodic_dlpno_ccsd(system, "cc-pvdz", with_triples=True)`,
runs the same engine on a 3-D crystal at the Gamma point without the
runner's artifacts. The reference is
the multi-k GDF engine at one k-point (`run_krhf_periodic_gdf` or
`run_kuhf_periodic_gdf` with
`fit_source="classic"`, the per-pair RSGDF fit the driver can retain, and
the Ewald exchange zero mode), and the correlation factors are that same
fit in the canonical auxiliary frame, so reference and factors are one
Hamiltonian: the result records `source_kind="gamma-gdf"`, the source
identity (which includes the driver's executed-source record and the
SHA-256 of the factors it contracted) and `same_source=True`. The default
fit source takes a Gamma fast path that retains no fit and is refused with
that reason; the two sources agree on the SCF energy to the fit tolerance. The active occupied orbitals are
Pipek-Mezey localised (safe on a cell whose orbitals wrap the boundary), and
the pair screens and `coupling_radius` use the minimum-image distance of the
cell. The Fock matrix keeps the Ewald exchange zero-mode term the SCF
applied; the factors are the bare fitted operator with G=0 omitted, as the
SCF contracted them.

A k-mesh, `kmesh=(2, 2, 2)` for example, runs the SCF on the unreduced
Gamma-centred mesh in cyclic order and folds it to the real BvK supercell
(`vibeqc.dlpno.source.bvk_supercell_source`): the per-k overlap, Fock and
retained fit are inverse-transformed with the finite-character kernels of
the chi line, the real supercell Fock is diagonalised in the canonical
orthogonal frame and its eigenvalues are checked against the union of the
per-k ones, so the engine sees one real supercell problem whose answer is
the full-mesh Bloch problem. Energies are reported in total and per cell;
the frozen-core count is per cell. The fold equals the explicit Gamma
supercell run to the fit tolerance (`tests/test_periodic_dlpno_gamma.py`).

The route refuses, with the reason in the error: BIPOLE references (no
certified same-source factors), symmetry-reduced references whose orbitals
live on the irreducible wedge only, smeared occupations, a reference whose
mesh is not the Gamma-centred unreduced one (a Monkhorst-Pack mesh carries
a half shift on even axes), and cells with `dim < 3`. Each spin must have
an integer occupied band count at every k-point and a global gap between
occupied and virtual bands. Fractional fillings, band overlap and reduced
orbital rank on UHF are refused instead of changing the determinant.
When freezing core orbitals, the frozen and active occupied bands must
also have a global gap, so folding or realifying a degenerate block cannot
change which orbitals are frozen.

Closed-shell evidence so far: at full domains the route reproduces the canonical CCSD on
the same source to the engine's exactness ratchet
(`tests/test_periodic_dlpno_gamma.py`), the Coulomb energy rebuilt from
its factors equals the SCF's, and the folded mesh equals the explicit Gamma
supercell to 1e-13 Ha. Against PySCF
(`examples/regression/parity_periodic_dlpno_vs_pyscf.py`, the H2 molecular
crystal in STO-3G on the (1,1,1), (1,1,2) and (2,2,2) meshes, the same
def2-SVP-JK auxiliary), the GDF reference energy and the full-domain
CCSD(T) correlation energy per cell agree with KRHF/GDF and KRCCSD(T) to
0.00 µHa when PySCF keeps the Ewald exchange term in the coupled-cluster
Fock operator (`KRCCSD(mf).keep_exxdiv = True`), which is this route's
convention; in def2-SVP the (T) correction per cell on the (1,1,2) mesh
agrees to 0.001 µHa as well. PySCF's default rebuilds that Fock without the
term and sits 4.6 to 8.8 mHa per cell lower on these cells, a finite-mesh
convention with the same thermodynamic limit. Its accuracy at the shipped thresholds on
real crystals and its cost against cell size have not been measured, so it
is not a production route yet; the gates are on the
[design page](../design_dlpno_correlation_source.md).

The historical toroidal DLPNO-MP2 and CCSD(T) wrappers and the chi
DLPNO-MP2, UMP2/DLPNO-UMP2 and CC wrappers also use the shared source
contract, with `source_kind="toroidal-compcell"` and `"chi-real-torus"`,
respectively. They pass their cyclic AO overlap directly to the engine,
so interleaved periodic and molecular calculations retain their own metrics.
The MP2, UMP2 and unrestricted CC pilot engines accept the optional
`source` keyword while keeping existing molecular positional calls.
The wrapper result exposes `source_kind`, `source_identity`, `same_source`
and `exchange_q0`; the underlying engine result carries the source identity.
Chi keeps its compact home-auxiliary factors. Its validated classic fit reuse
reports `same_source=True`, while its separate correlation refit and the
toroidal compensated-cell fit report false. These experimental comparisons
retain their historical thresholds and residual conventions. The UHF
source keeps alpha and beta orbitals, Fock matrices, energies and electron
counts independent, sharing only the AO metric and spin-independent factors.
Frozen-core spaces, finite-torus normalisation, exchange zero-mode
conventions and admission guards are unchanged. Caller options and
references are preserved on successful and failed solves, including the
toroidal CCSD/CCSD(T) triples selection. This migration removes global
overlap replacement; it does not qualify the periodic routes for production.
The toroidal wrappers run their supercell HF on the GDF Hamiltonian at Gamma
with `exxdiv='ewald'` exchange (`run_krhf_periodic_gdf` at kmesh (1, 1, 1)),
so the supercell HF equals the unit-cell KRHF on the N-point mesh, and the
wrappers report `exchange_q0="BvK-ewald"`. Until #566 they used Ewald J with
the molecular-limit exchange, which keeps every density element in the home
image: on the 6-bohr H2 chain a translation of the torus by one cell changed
that K by 3.2e-3 (three cells) and 4.4e-3 (four cells). In a cell padded with
vacuum the BvK HF energy converges only as 1/N along the replicated axis, to a
limit that depends on the transverse vacuum (-1.115959 Ha per cell at 20 bohr
and -1.116296 at 40 bohr on that chain). `toroidal_mp2_gamma_pairs` splits
the MP2 energy over Wannier pairs, because the canonical orbitals of a
translation-symmetric torus are Bloch states with no cell of their own.
The separate experimental API below connects automatic translated
orbital/domain selection to representative MP2 iteration. Those wrappers
continue to use their established solvers.

For bounded experiments, `vibeqc.dlpno.translation.TranslationPairSpaces`
describes real restricted pair spaces on a finite torus, and
`vibeqc.dlpno.translation_mp2` provides `translation_mp2_residual` and
`translation_pair_energy_per_cell`. Callers supply the occupied coefficients
in the declared cell-major translation gauge, the AO overlap and Fock,
one virtual basis per representative pair, and the pair tensors. The
contract checks actual orbital translation, overlap/Fock covariance and
pair-space orthonormality and virtuality. Optional `frozen_occupied`
coefficients supply complete frozen bands in the same gauge; pair bases
exclude both active and frozen occupied spaces. The API retains the correct
pair weights on even meshes, including same-band antipodal pairs, and
transports cross-pair overlaps and Fock couplings with modular cell
arithmetic. This is an experimental algebra API: it has no SCF or PNO
builder, iterative solver, screening or discarded-space correction. It
does not accept arbitrary localized gauges or provide an unrestricted
approximation, and tensor consistency alone does not certify the supplied
Coulomb factors or establish `same_source`. The current job runners continue
to use their established solvers. The API does not establish periodic
production accuracy or measured scaling. See the
[design](../design_dlpno_correlation_source.md#increments) for its
finite-torus normalization and literature comparison.

`vibeqc.dlpno.translation_builder.build_translation_pair_pnos` prepares
those pair spaces from a real restricted `CorrelationSource`. It requires
an explicit cell-major active occupied gauge and, when freezing core,
complete `frozen_occupied` bands whose projector separately matches the
source's core space. `pair_ao_domains` accepts an AO-index set per
representative; omitting it uses full AO domains. The builder retains PAO
projection tails and uses the source's overlap, Fock and factors. Generic
providers undergo an audit of every translated origin's untruncated PAO
exchange block before reduction to representative PNO spaces. The compact
factor path described below establishes covariance by construction instead.
Both preserve the source's existing `same_source` value; neither certifies
that a separate correlation refit equals the SCF fit.

The defaults `tcut_pno=0` and `pno_norm="mp2"` retain the admitted PAO space
and explicitly select the shared Riplinger density. Positive cutoffs retain
the existing strict occupation comparison and largest-PNO fallback;
incompatible self-inverse pair spaces fail instead of being silently
enlarged. The returned `TranslationPNOBuild` contains `spaces`, `integrals`,
`amplitudes`, ranks, occupations and source/build identities.
Its `pno_correction_per_cell` estimates the semicanonical energy omitted by
PNO truncation within the chosen PAO domains. The amplitudes are initial
semicanonical estimates, not a converged coupled MP2 solution. This bounded
experimental builder checks translation covariance and makes no
cost-scaling claim. The following driver adds automatic gauge/domain
selection and iteration; the builder itself continues to return prepared
inputs only.

`vibeqc.dlpno.translation_driver.run_translation_dlpno_mp2(source, mesh, ...)`
is the explicit experimental driver for a real restricted finite-torus
`CorrelationSource`. It builds separate active and frozen occupied frames
from projected AO seed families and symmetric Löwdin orthogonalization,
selects domains, constructs D2's representative pair spaces, and invokes
the compiled Jacobi MP2 solver. The selector preserves each source
projector and validates the actual translation gauge. It is a bounded
deterministic frame construction, not an optimized Wannier localizer; a
projector without an admitted common AO-seed frame is refused even when
a more general frame might exist. `n_frozen` counts frozen spatial
orbitals over the entire finite torus and must contain complete bands.
It selects the first occupied source columns, not a chemical-core policy.

Home atom AO groups normally come from the source's repeated cell-major
shell map. Model sources without shells can provide `home_atom_aos`, a
partition of every AO index in one primitive cell. For positive
`tcut_mkn`, atoms pass the strict absolute gross Mulliken population test
on home orbitals. The selected masks are translated exactly and united
for each pair; an empty orbital domain is an error. This is not DOI or
Nejad's OSV domain selection. Defaults `tcut_mkn=0`, `tcut_pno=0` retain
full AO domains and the admitted virtual space, with `pno_norm="mp2"`.
Nonzero cutoffs introduce independent domain and PNO approximations;
the PNO correction does not recover omitted AO domains.

Iteration defaults are `max_iter=100`, `conv_tol_energy=1e-9`,
`conv_tol_residual=1e-7` and `damping=1.0`. The solver stores one amplitude
per representative pair, evaluates the full raw residual with all placed
occupied couplings, and tests energy and residual on the same amplitude
state. It adds no Fock cutoff, denominator floor or DIIS. The return value
exposes `occupied_selection`, `domain_selection`, `pair_build` and
`solution`, with source and consumed-input identities. Energies and
convergence status live on `solution`: `e_pair_per_cell` is the iterated
energy, `e_pno_correction_per_cell` is the fixed semicanonical correction,
and `e_corr_per_cell` is their sum. Neither the correction nor reference
energy enters the residual. `n_iter` counts evaluated states, and
`max_iter=1` cannot establish convergence. A returned unconverged state
is not an accepted correlation result. Factor-covariance checks preserve
the source's `same_source` value and do not certify a separate correlation
refit as the SCF fit.

This API does not change `run_job` or periodic-wrapper defaults, does not
accept unrestricted or complex references and makes no production or
scaling claim. The compact preparation below reduces specific storage and
repeated work, while dense source data and unscreened pair couplings still
limit the route to bounded experiments. The
[design comparison](../design_dlpno_correlation_source.md#increments)
distinguishes this gauge and population-domain policy from published
optimized IBO, fragment LNO and ROHF/QRO methods.

`vibeqc.dlpno.translation_factors.TranslationHomeFactors` stores an
immutable snapshot of the real home-auxiliary factors. All other auxiliary
cells are defined by simultaneous cyclic translation of both AO indices.
The pair-exchange contraction streams one auxiliary cell at a time,
avoiding translated AO-factor copies and the full logical auxiliary MO
transform. Existing consumers can still call `mo_transform` for the full
transform. This representation preserves the supplied finite Coulomb
operator; it is not a local density-fitting approximation.

The builder and experimental driver accept `factor_audit="auto"`, which
uses constructive covariance only for that exact provider on the matching
mesh. Other providers keep the every-origin numerical audit.
`factor_audit="all-origins"` forces the reference audit for either provider.
The build records its factor identity and audit mode;
`max_translation_error=None` denotes the constructive path, not a measured
zero error. The source's `same_source` and exchange-zero-mode conventions
retain their meaning.

Projected-PAO preparation computes selected-domain Gram and Fock matrices
using the complete active and frozen occupied union, avoiding a full
virtual projector. A retained-space Cholesky refinement uses selected full
AO overlap columns and the shared overlap-times-occupied product to
control cancellation for nearly dependent PAOs. It introduces no second
rank cutoff or threshold change and avoids a full overlap-times-pair-basis
product. Preparation retains local coefficients and occupied projection
tails in `projected_pair_bases` as well as the compatibility AO coefficient
arrays. The solver prepares fixed cross-pair metric links before iteration;
Jacobi updates then use pair-space links without AO translation or metric
contraction in each iteration. The finite residual, energy weights,
thresholds and convergence criteria are unchanged.

If conservative roundoff and occupied-identity defect estimates indicate
cancellation in a target's compact links, preparation uses selected full
AO overlap columns and contracts translated source coefficient cell blocks.
It avoids a full dense overlap-times-pair-basis product and a translated
coefficient copy, but performs AO-row contractions during preparation.
Spaces and admission tolerances stay unchanged. `transport.stable_links`
counts these evaluations; transport and solver provenance record the count
as `stable_selected_column_links`. No such AO work enters the iteration.

The driver and solver default to `transport_backend="prepared"` with
`max_transport_bytes=256*1024**2` (256 MiB). The bound covers retained
native transport storage and conservative preparation workspace; it
excludes existing caller arrays, binding copies, allocator overhead and
the subsequent solver's amplitude/residual workspace. It is not a total
process-memory limit. Exceeding it refuses preparation without silently
switching backends. `transport_backend="ao"` selects the independent AO
reference implementation. The standalone
`vibeqc.dlpno.translation_transport.prepare_translation_transport(build,
max_bytes=...)` returns reusable transport for
`solve_translation_mp2(build, transport=...)`; reuse checks its input
identity against the current spaces. Transport provenance records
link/coupling counts and retained, workspace and required bytes. Custom
builds without projected-basis metadata prepare links by contracting AO
pair coefficients directly; their iterations still use pair-space links.

The explicit block-source API below removes the dense AO overlap/Fock
dependency from preparation. Retained AO pair coefficients and home
factors remain dense; the latter still occupy the home auxiliary rank
times the square of the total AO count. Their pair contraction sums every
auxiliary cell. Prepared links require storage proportional to the sum
of products of connected pair ranks.

`vibeqc.dlpno.translation_operators.CyclicAOOperator(home_row_blocks, mesh)`
stores all home-row displacement blocks with shape `(N, b, b)`, where N
is the number of cells and b is the AO count per cell. The matrix element
between cells R and T uses block `T-R`, modulo the mesh. Supply matching
operators as both `source.overlap` and `reference.fock` to the experimental
translation driver or builder. The representation is immutable and uses
block actions and selected columns without expanding a full AO matrix.
Symmetry and reciprocal-space metric positivity are checked; no metric
eigenvalue clipping, threshold change or distance cutoff is introduced.

This is an explicit API. Existing dense sources and periodic adapters
keep their behavior; adapters are not converted automatically. The legacy
`source.overlap_matrix()` and `transport_backend="ao"` refuse block
sources. Use the prepared backend. The source's finite Hamiltonian,
normalization, frozen/active partition, `same_source` and zero-mode
conventions remain the caller's declared source data; a cyclic operator
alone does not certify physical provenance.

The block path covers admission, automatic occupied selection, population
domains, PAO construction, pair-space checks and native transport
preparation. Projector comparisons and hashes stream AO rows with the
same maximum-entry tolerance, and occupied selection computes only the
candidate seed Gram matrices it needs. D4's retained-space refinement and
stable transport still use selected full AO columns when necessary.
The iterative equations, ranks, cutoffs and convergence criteria are
unchanged.

Overlap/Fock storage falls from `N*N*b*b` to `N*b*b` elements per operator.
Occupied/orbital and pair coefficient arrays, occupied Fock, pair Gram
matrices and home-auxiliary factors remain dense. Full domains may still
require projected matrices of full AO dimension, filled by rows without
a second source AO matrix, and no block displacement is screened. This
bounded API makes no production, timing or scaling claim.

`vibeqc.periodic.chi.translation.prepare_chi_translation_source` supplies
the block operators directly from a fresh chi RI-RHF calculation by
default. It returns a real restricted `CorrelationSource` with kind
`"chi-translation-blocks"`, ready for the explicit translation driver.
The function takes `system`, `basis`, `mesh=(1, 1, 1)` and `options=None`,
plus the same auxiliary and fit controls as `prepare_chi_reference`, and
`n_frozen=0`. It copies caller options before invoking SCF. Defaults use
strict accepted-reference SCF controls; explicit options must request
`conv_tol_energy <= 1e-11` and `conv_tol_grad <= 1e-9`.

For an existing three-dimensional closed-shell `system` and its `basis`:

```python
from vibeqc.periodic.chi.translation import prepare_chi_translation_source
from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2

mesh = (2, 1, 1)
n_frozen = 0
source = prepare_chi_translation_source(
    system, basis, mesh, fit_source="classic", n_frozen=n_frozen,
)
result = run_translation_dlpno_mp2(
    source, mesh, n_frozen=n_frozen, tcut_mkn=0.0, tcut_pno=0.0,
)
if not result.solution.converged:
    raise RuntimeError("Experimental translation MP2 did not converge")
energy_per_cell = result.solution.e_corr_per_cell
```

`n_frozen` counts spatial orbitals over the entire torus, must contain
complete translated bands, and must leave active occupied orbitals. Pass
the same count to both calls. Admission requires a converged, unsmeared
full-rank reference on the complete Gamma-centred cyclic mesh with uniform
weights, a global occupied/virtual gap and a frozen/active gap when that
split is present. The adapter checks the per-k generalized Fock
eigenproblem and realifies complete eigenspaces without assuming fixed
orbital phases or a particular degenerate rotation.
Each complete degenerate space is folded directly from compact Bloch
coefficients. Preparation avoids the full complex finite-torus orbital
array and its sorted copy, and checks the final real orbitals in column
panels. Degenerate spaces are never split to fit a memory target.
`max_folded_orbital_columns` and `orbital_validation_columns` in the
Bloch-transform provenance describe the actual panel widths. These are
array-size diagnostics, not a total memory budget. Generic GDF adaptation
and dense-to-block accepted-reference migration use this same preparation;
an already prepared block handle continues to reuse its sealed transforms.
It forms home-row S/F blocks by the inverse Bloch transform with one
factor of `1/N`, without allocating full finite-torus S/F matrices. The
SCF energy remains extensive on the returned reference and the MP2 result
is per primitive cell. The existing exchange-zero-mode term stays in F.

`fit_source="classic"` reuses validated retained canonical SCF factors
and records `same_source=True`. The default `fit_source="auto"` keeps the
selected SCF and separate classic correlation refit distinct and records
`same_source=False`. Both retain compact home-auxiliary factors. Source
provenance includes the SCF and correlation-fit identities, consumed Bloch
data and transform residuals, frozen count and factor identity; this is
not a numerical qualification of either fit.

This entry point also accepts the reusable handle described below and
keeps `run_job` and existing wrapper defaults. It retains the refusals
for unrestricted references, ECPs, explicit fitted support, explicit SR
post-HF and corrected RIJCOSX. The full canonical coefficient array and
downstream occupied/pair coefficients, projected domain matrices and home
factors remain dense by default. The optional occupied projectors below
reduce retained canonical storage. Smaller orbital/factor payloads and qualified
localization and screening remain implementation gaps. This finite-source
representation change does not establish periodic production accuracy or
cost scaling.
The [design comparison](../design_dlpno_correlation_source.md#increments)
explains its relation to the finite Fourier transforms in Nejad et al.
and Ye and Berkelbach, without equating this route to their localization,
local fitting or fragment approximations.

### Experimental translation sources and reference reuse

`vibeqc.dlpno.translation_gdf.gdf_translation_source` converts an existing
direct multi-k GDF RHF result into a block source without rerunning SCF or
refitting. It accepts `system`, `basis`, `result` and keyword-only `mesh`,
`aux_basis_name=None`, `n_frozen=0`, `compact_occupied=False`. For an already converged `hf_result`
on the full Gamma-centred cyclic `mesh`:

```python
from vibeqc.dlpno.translation_gdf import gdf_translation_source
from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2

source = gdf_translation_source(
    system, basis, hf_result, mesh=mesh, n_frozen=n_frozen,
)
result = run_translation_dlpno_mp2(source, mesh, n_frozen=n_frozen)
```

The SCF must have used `fit_source="classic", k_exchange="gdf"` and
retained its attested canonical AO-pair factors. `retain_fit_cache=True`
is optional; a legacy fit cache alone is insufficient.
Admission checks the factor digest, source frame, basis, mesh and finite
support before recording `same_source=True`. The executed
`physical_pairs` or `historical_index_ball` policy is preserved.
Initially only `BvK-ewald` exchange zero mode is admitted; UHF, ECP,
COSX, eigenmode factors, SR, explicit fitted support, explicit reciprocal
tails and symmetry-unfolded references are refused. Cartesian-only k
points are converted to a detached fractional-coordinate view and
validated against the same complete mesh. Existing fractional coordinates
must also match. The
adapter preserves the original result and applies the D6 orbital, rank,
gap and normalization guards above. `n_frozen` again counts total-torus
spatial orbitals, rather than the per-cell count of the historical
`bvk_supercell_source` API.

For repeated chi consumers, prepare one accepted block reference:

```python
from vibeqc.periodic.chi.reference import prepare_chi_reference
from vibeqc.periodic.chi.translation import prepare_chi_translation_source
from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2

controls = dict(fit_source="classic")
accepted = prepare_chi_reference(
    system, basis, mesh, matrix_representation="cyclic-blocks",
    n_frozen=n_frozen, **controls,
)
source = prepare_chi_translation_source(
    system, basis, mesh, reference=accepted, n_frozen=n_frozen, **controls,
)
result = run_translation_dlpno_mp2(source, mesh, n_frozen=n_frozen)
```

Check `result.solution.converged` before accepting either calculation's
energy, as in the cold-source example above.

Block handles bind the exact total frozen count and reuse S/F, orbital
and compact-factor transforms. Match the original source, auxiliary,
cutoff and fit controls; explicit SCF options must also match, while
omitting options accepts the handle's recorded SCF controls. Validation
checks geometry/bases, process/runtime identity, fractional and Cartesian
k points, state, factors and metadata before reuse. Returned source
orbital/energy arrays and provenance are detached snapshots; immutable
block operators and compact factors can be shared across consumers.

To retain compact occupied projectors, add `compact_occupied=True` to
`prepare_chi_translation_source` or `gdf_translation_source`. For accepted
chi references, include it in `controls` in the example above and pass it
to both preparation and reuse; handle preparation also requires
`matrix_representation="cyclic-blocks"`. A block handle refuses a different
occupied representation or frozen count. The default remains `False`.

The compact reference has `occupied_projectors` with separate immutable
active/frozen home-row blocks and no `mo_coeffs` array. Selection uses the
same normalized AO seed families and checks each selected partition against
its source projector. Source admission and seals bind both projectors, their
original audit residuals, counts and overlap/Fock identities without
expanding a square AO matrix. The returned mutable reference and provenance
remain detached from the handle; immutable projectors can be shared.

This option never assembles the full real canonical array, including
during admission. The complete-spectrum audit replays real eigenspaces
to check every cross-eigenspace overlap and AO stationarity residual.
Separate frozen and active projector comparisons visit every AO row in
tiles. The existing residual definitions and tolerances remain unchanged.
`orbital_audit="replayed-complete-eigenspaces"` and
`orbital_audit_block_evaluations` in the Bloch-transform provenance record
every block reconstruction. Small eigenspace transforms are cached to
avoid repeated Gram and projected-Fock diagonalizations, while AO columns
are reconstructed on every visit. The cache retains no AO coefficients
and lasts only for that preparation call. Its numeric payload is bounded
by one primitive complex Bloch coefficient tensor, or `16*N*b*b` bytes
for `N` cells and `b` primitive AOs. Complete recipes that do not fit are
recomputed. `orbital_audit_recipe_cache` reports the policy, budget,
stored bytes, entries, hits and fresh realifications. This budget excludes
Python metadata and transient workspace. Replay still performs all global
checks and does not promise faster preparation. A large complete degeneracy
still requires large workspace and is never split for a memory target.
It changes no domain/PNO
threshold, pair metric, residual, factor fit, energy normalization or
`same_source` claim. It adds no unrestricted or QRO conversion.

`prepare_chi_reference` still defaults to `matrix_representation="dense"`
and `n_frozen=0`. Its existing MP2/CC consumers require dense handles.
The translation source adapter can also consume a dense handle: it folds
the stored Bloch state directly to blocks, audits the requested frozen
partition and reuses the compact home factors. It does not remove the
dense arrays already held by that handle. Nonzero `n_frozen` at handle
preparation is supported only with `matrix_representation="cyclic-blocks"`.

With an accepted handle, `correlation_fit_status="reused"` means that no
new correlation fit was built. A reused `auto` handle still has
`same_source=False`, because its correlation fit was separate from SCF.
`source.provenance["adapter"]["accepted_reference"]` records the handle
seal and state identities and marks the Bloch transform as `"reused"` or
`"built-from-sealed-bloch"`. This remains process-local reuse, not disk
caching or a qualification of the finite Hamiltonian. Eigenspace audit
workspace, selected occupied and pair AO coefficients, compact home factors
and unscreened links remain dense payloads. Reducing them and extending attested source
coverage remain concrete development gaps.

For an already admitted classic physical-pair RI `ChiHFReference`, use
[`run_chi_local_mp2`](../chi_local_mp2.md). This sealed-reference route keeps
the actual SCF Hamiltonian and active/frozen projectors, verifies real
operator and factor transport, and reuses the shared translation engine.
It exposes independent domain/PNO and iteration controls and records raw
energy separately from the discarded-PNO estimate through `vibeqc.output`.
Its scalar-ECP metadata support does not claim #530 acceptance. Both routes
remain experimental; D93 fitted/post-HF remains not-qualified.

## Current limitations

* **DLPNO-MP2** runs on closed-shell RHF *and* open-shell UHF references:
  `run_job(method="dlpno-mp2")` auto-routes a multiplicity > 1 system to
  UHF + DLPNO-UMP2 (spin-channel resolved αα/ββ/αβ; `res.dlpno_ump2`). Pass a
  `DLPNOUMP2Options` as `dlpno_options` to control its thresholds
  (`localise`, `tcut_pno`, `tcut_pairs`, `n_frozen`). **DLPNO-CCSD/(T)**
  also run on open-shell UHF references. `run_job(method="dlpno-ccsd"|
  "dlpno-ccsd(t)")` auto-routes a multiplicity > 1 system to UHF plus the
  local `run_local_dlpno_uccsd` engine (`res.dlpno_ccsd`; use
  `LocalUCCSDOptions` for route-specific controls). Its amplitude equations
  are solved in a compiled spin-orbital PNO-space kernel
  (`residual_domain="pno"`, the default): every residual term is contracted
  in the PNO space of the amplitude that carries the contracted virtual
  (Riplinger and Neese 2013, Sec. II C), the singles stay in the full virtual
  space of their spin (or their singles-PNO space), and the pair spaces are
  built per spin block, so every PNO is spin-pure. It is exact against
  canonical UCCSD at full domains and accepts up to 1500 basis functions.
  `residual_domain="union"` keeps the earlier Python route, which contracts
  each residual in the union of its coupling set's pair spaces with a dense
  spin-orbital kernel and is capped at 160 basis functions. Passing
  `DLPNOUCCSDPilotOptions` explicitly selects the O(N⁶), `max_nbf`-capped
  correctness pilot.

  Each spin-orbital pair's PNOs come from a two-sided MP2 pair density
  (`pno_density="two-sided"`, the default): an opposite-spin pair adds each
  spin side's density mapped into the other spin's virtual space through the
  alpha-beta virtual overlap, and a same-spin pair adds the two-sided density
  of its direct amplitude, because the antisymmetrised MP2 amplitude vanishes
  wherever it is locally symmetric while the CCSD amplitude does not. The
  older one-sided density (`pno_density="one-sided"`) truncated far harder
  than the closed-shell route at the same `tcut_pno`: singlet H2O/cc-pVDZ
  through this solver sat 0.274 kcal/mol above canonical at NormalPNO against
  0.016 for the closed-shell solver; the two-sided density gives 0.022. At
  NormalPNO/cc-pVDZ against canonical DF-UCCSD the CCSD correlation energy
  is 0.0025 (OH), 0.0125 (NH2), 0.029 (CH3), 0.0099 (triplet CH2) and 0.071
  (HCO) kcal/mol too high. The route also adds the semicanonical MP2 PNO
  correction (`pno_correction`) and chooses DOI domains per spin orbital,
  as the closed-shell routes do, and it contracts each residual once per
  group of targets that share a coupling set and domain
  (`n_residual_groups`): at the default full coupling that is two
  contractions per iteration.

  The local engine's
  ``tcut_pairs`` option screens weak
  spin-orbital pairs using their full-virtual MP2 energies and retains those
  MP2 contributions. The same `tcut_pairs` applies to every spin-orbital
  pair. Saitow 2017 section III B's additional `TScaleTCutPairs=0.33` is not
  claimed here: it assumes the paper's single ROHF/QRO-like spatial-orbital
  set, while this solver retains independent UHF alpha and beta spaces.
  Compact OH and a separated H + H2 doublet pin the
  explicit zero-threshold exactness setting and the lossless distant-pair
  limit. The ``tcut_mkn`` option selects each pair's atoms
  from the two occupied-orbital Mulliken populations and builds separate
  alpha/beta semicanonical PAOs on their union. A separated OH + H2 doublet
  pins the reduced pair-virtual span and exact fragment limit. The default-off
  ``tcut_pno_singles`` option sums each occupied orbital's full-virtual MP2
  pair-density contributions and retains same-spin singles-specific natural
  orbitals above the threshold, optionally inside that occupied orbital's PAO
  atom domain. The zero default keeps the full spin-virtual singles space. A
  separated OH + H2 doublet pins both the domain reduction and its exact
  fragment limit. The same direct solver has an experimental, default-off
  open-shell DLPNO-(T0) path: set ``compute_triples=True`` and control the
  triple-natural-orbital occupation cutoff with ``tcut_tno``. It streams each
  distinct occupied triple through the native spin-orbital kernel, reports
  ``e_t`` plus triple/TNO diagnostics, and reproduces the dense UCCSD(T) pilot
  when every threshold is zero and ``localise="none"``. Set
  ``triples_mode="t1"`` to opt into
  spin-block semicanonical occupied coupling: it removes the `(T0)` localized
  occupied-Fock error while retaining per-triple TNO streaming. That rotation
  takes the occupied indices out of the localised basis, where a weak-pair
  label or an occupied distance has no meaning, so `(T1)` cannot carry a
  screened triple list. It is decided on the *realised* screened set rather
  than on the raw thresholds: ``tcut_pairs`` or ``coupling_radius`` values that
  screen no triple leave `(T1)` reachable and exact on the full triple list,
  while any setting that actually removes one fails closed with the offending
  count. Use ``triples_mode="t0"`` for a genuinely screened open-shell triples
  run. Broader radical benchmarks remain gated. In the 2026-10-10
  cc-pVDZ, chemical-frozen-core campaign, HCO converged CCSD at both
  NormalPNO and TightPNO but its compiled local (T0) returned non-finite
  triples and total energies (source `ac46948b`); the NormalPNO result
  also reproduced with one thread. Eigen's divide-and-conquer SVD returned
  invalid left singular vectors for rank-deficient triple unions, including
  finite vectors that did not span the input. The native kernel now checks
  finiteness, orthonormality and the retained range, and retries a failed
  decomposition with Jacobi SVD at the same singular-value cutoff. The solver
  also rejects any remaining non-finite triples contribution or total
  correction with an error, identifying the first affected occupied triple
  when available. A finite CCSD result alone does not qualify CCSD(T);
  the broader open-shell accuracy gate remains open.

  The 2026-10-11 rebuilt-core check converged all nine radicals at both
  NormalPNO and TightPNO with finite energies. HCO's `(T0)` corrections are
  -0.009578978838032 and -0.009595191615705 Ha, respectively; the NormalPNO
  result agrees with the Python union control within 4e-16 Ha. Canonical
  references for the C6 and C8 radicals are still missing, so these 18
  finite runs do not complete the nine-system accuracy gate.

  The same rank-deficient pair spaces also reproduced the failure in the
  closed-shell local `(T0)` kernel using synthetic factors and amplitudes.
  Both native kernels share the checked union builder and reject non-finite
  corrections. This algebraic control is not a closed-shell calculation on
  HCO; it checks that an equivalent pair basis gives the same energy.

  Open-shell ORCA comparisons must use total energies: even with `UHF`
  input, the measured ORCA 6.1.1 DLPNO runs construct quasi-restricted
  orbitals and report their reference energy as `E(0)`. Subtracting that
  correlation energy directly from the canonical UHF correlation energy
  mixes references. The separate UHF SCF energies and final total
  energies are the quantities to compare.

  CCSD-only cost profiles for primary alkyl radicals use cc-pVDZ,
  chemical frozen core, NormalPNO and the open-shell default
  `coupling_radius=0` (build `ac46948b`). All four converge in 17
  iterations. Occupied counts below are active **spin orbitals**:

  | Radical | Threads | Coupled occupied | Mean PNOs | Setup, s | Seconds / iteration | Static factors, GB | Peak RSS, GB |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | C8H17 | 48 | 49 | 36.0 | 1.68 | 3.08 | 4.6 | 8.1 |
  | C12H25 | 48 | 73 | 35.8 | 7.03 | 13.97 | 12.9 | 20.7 |
  | C16H33 | 96 | 97 | 35.8 | 17.52 | 28.25 | 27.5 | 45.7 |
  | C20H41 | 96 | 121 | 35.7 | 55.66 | 70.02 | 50.2 | 80.0 |

  A C12H25 control at `coupling_radius=12` also converges in 17
  iterations with 48 threads. The mean coupled occupied count falls from
  73 to 50.6, setup takes 6.32 s and each iteration averages 7.90 s, with
  11.7 GB of static factors and 18.9 GB peak RSS. The correlation energies
  are -1.872632654463 Ha at radius 0 and -1.872636384960 Ha at radius 12.
  The C16 radius comparison remains outstanding; these
  separate allocations and differing thread counts do not establish a
  scaling exponent or justify a new open-shell radius default. GB means
  10^9 bytes.

  Positive ``tcut_tno`` under `(T1)` is calibrated on OH/def2-SVP
  (2026-07-29; full domain `e_t = -1.778474224055274e-03` Ha at 29.0 spin
  virtuals per triple). Every sampled truncation in this calibration loses
  correlation, so every table entry is a positive error. That observed sign
  is not a variational guarantee for other molecules or domains:

  | ``tcut_tno`` | error vs full domain | mean TNOs/triple |
  |---|---|---|
  | `1e-12` | `+11.27` µHa | 26.07 |
  | `1e-9`  | `+12.39` µHa | 24.73 |
  | `1e-8`  | `+20.05` µHa | 23.49 |
  | `1e-7`  | `+34.39` µHa | 21.85 |
  | `1e-6`  | `+63.37` µHa | 19.92 |
  | `1e-5`  | `+156.16` µHa | 16.77 |
  | `1e-4`  | `+759.06` µHa | 10.55 (collapsing; see below) |

  `1e-9`, the `T_CutTNO` default of Guo *et al.* (2018), costs `12 µHa`
  (0.008 kcal/mol) for a 15% smaller triples domain and is the recommended
  starting point. There is no free threshold: the smallest value tested
  already costs `11 µHa`, so ``tcut_tno=0`` remains the only exact setting.

  The same sweep on two more radicals, including a high-spin case, gives the
  same picture (error vs full domain; `d` = triples that collapsed below three
  spin virtuals):

  | system | multiplicity | `1e-9` | `1e-6` | `1e-4` |
  |---|---|---|---|---|
  | OH  | 2 | `+12.39` µHa | `+63.37` µHa | `+759.06` µHa, `d=13` |
  | NH2 | 2 | `+7.62` µHa | `+37.02` µHa | `+1177.91` µHa, `d=9` |
  | CH2 | 3 | `+5.73` µHa | `+30.25` µHa | `+706.18` µHa, `d=10` |

  Across all three, `1e-9` stays within `12` µHa and `1e-6` within `64` µHa,
  and the collapse sets in between `1e-6` and `1e-4` in every case. The triplet
  is not an outlier, so the recommendation does not depend on spin
  multiplicity over this (small) sample.

  **`1e-6` is the coarsest calibrated value on every system tested.** Through `1e-6`
  every triple keeps at least three spin virtuals (smallest domain 8), so the
  error is pure truncation. By `1e-4` the smallest domain is a *single*
  virtual and 13 of the 84 triples have collapsed to identically zero, so that
  row's `+759` µHa mixes truncation with collapse and should not be read as a
  truncation error.

  ```{warning}
  Do not calibrate this threshold on a minimal basis. OH/STO-3G retains only
  3.0 TNOs per triple at full domain, and a triple excitation needs three
  *distinct* spin virtuals, so any positive ``tcut_tno`` drops 75 of its 84
  triples below three and `e_t` collapses from `-2.29e-07` Ha to order
  `1e-46`: the correction vanishes identically instead of degrading. The new
  ``LocalUCCSDResult.n_degenerate_tno_triples`` counts triples in that state;
  a nonzero value means the threshold has truncated past the point where the
  correction exists, not that it is merely small.
  ```
* Energies only, no analytic gradients.
* `dlpno-ccsd(t)` runs the local solver + the closed-shell (T1), the
  canonical (T) on the rotated converged amplitudes (exact == canonical
  CCSD(T) at full domains); the
  (T) uses a spatial closed-shell kernel (no spin-orbital redundancy),
  validated to machine precision against the spin-orbital reference. The
  default `residual_domain="pno"` contracts the residual in PNO spaces in the
  compiled `dlpno_ccsd_pno_solve` kernel, with the target pairs distributed
  over OpenMP threads; `LocalCCSDResult` reports its factor bookkeeping as
  `n_offlist_rows` (factor rows a pair stores for occupieds outside its own
  coupling set because another target needs them), `static_factor_bytes`,
  `setup_seconds` and `iteration_seconds`. The oracle
  `residual_domain="extended"` contracts once per distinct (occupied coupling
  set, extended atom domain) group in the compiled
  `vibeqc::dlpno_pair_residual` kernel, and every pair in the group reads
  its own block; `n_residual_domains` reports its group count. The legacy
  `residual_domain="pair"` mode uses the same kernel in its pair-PNO basis;
  `use_cpp_kernel=False` selects the NumPy transcription for those two modes
  (the PNO-space route is compiled only).
* `tcut_pno` is a cut on **pair-density occupation numbers**, so it only means
  what a preset intends under the density it was calibrated against. That
  density is Riplinger and Neese's Eq. 23, `D = Tt T^T + Tt^T T` with
  `Tt = (4 T - 2 T^T) / (1 + delta_ij)`, selectable as `pno_norm="mp2"`
  (the pre-2013 LPNO variant is `pno_norm="iepa"`). Molecular options default
  to `pno_norm="mp2"`. The explicit `pno_norm="legacy"` convention uses
  vibe-qc's historical density of the bare amplitudes,
  which retains 5 % to 19 % fewer PNOs at the same nominal threshold and is
  therefore effectively looser than the preset name implies. For low-level
  API compatibility, `resolve_pno_norm(None)` and `pair_density` with an
  omitted or `None` norm still select the pre-#65 `legacy` convention. These
  helpers do not select the molecular options default; pass `"mp2"` explicitly
  when using them to construct that recipe. Retained periodic reference paths
  request `"legacy"` explicitly. The convention in force is printed in the `.out`
  block and recorded as `dlpno_pno_norm` in the `.system` manifest.
  The (T) is NumPy (already BLAS-bound). The local solver
  uses the 12-bohr `coupling_radius` default described above, so occupied
  coupling remains local on extended systems while it is a bit-identical
  no-op on smaller molecules. Set `coupling_radius=0` to couple every pair for
  the full-domain reference. Remaining production follow-ups are tracked in
  the canonical
  [GitLab issue list](https://github.com/vibe-qc/vibe-qc/issues).

## Citations

Jobs running `dlpno-mp2` emit the method papers into the
`.references` / `.bibtex` outputs automatically: Møller-Plesset 1934,
Feyereisen 1993 (RI), Pinski 2015 (DLPNO-MP2), Foster-Boys 1960
(localisation), Liakos et al. 2015 for vibe-qc's cross-route
LoosePNO/NormalPNO/TightPNO policy, and Pinski and Neese 2019 to disclose
that the same three names denote a different MP2-specific threshold family.
The coupled-cluster routes (`dlpno-ccsd`, `dlpno-ccsd(t)` and their
open-shell siblings) add the CCSD working equations and the DLPNO papers
named in the [Theory](#theory) section: {vibeqc-cite}`purvis_bartlett_ccsd_1982`,
{vibeqc-cite}`stanton_gauss_watts_bartlett_ccsd_1991` and
{vibeqc-cite}`deprince_sherrill_df_ccsd_2013` for the spin-orbital CCSD
equations and the density-fitted factorisation the compiled PNO-space
kernels transcribe, {vibeqc-cite}`riplinger_dlpno_2013`,
{vibeqc-cite}`riplinger_sparsemaps_2016` and
{vibeqc-cite}`pinski_dlpno_mp2_2015` for the local method, and for the
triples {vibeqc-cite}`riplinger_triples_2013` with
{vibeqc-cite}`raghavachari_ccsdt_1989`; the open-shell routes cite
{vibeqc-cite}`saitow_openshell_dlpno_2017` as the method-family source.
Four sources are added at runtime only after the algorithm they describe
executed, so a static route never over-claims: a closed-shell run with
`triples_mode="local"` adds {vibeqc-cite}`rendell_lee_komornicki_triples_1991`,
whose spatial (T) expression the compiled local (T0) kernel evaluates; an
open-shell local (T0) adds {vibeqc-cite}`guo_openshell_dlpno_triples_2020`
and {vibeqc-cite}`watts_gauss_bartlett_openshell_triples_1993`; the
rotated open-shell `(T1)` adds the latter; and the open-shell
`triples_mode="t1-iterative"` adds {vibeqc-cite}`guo_dlpno_t1_2018` and
{vibeqc-cite}`guo_openshell_dlpno_triples_2020`. The default closed-shell
`(T1)` and the `"exact"` oracle use neither Guo paper and do not cite them.
The periodic routes cite {vibeqc-cite}`pipek_mezey_1989` in place of
Foster-Boys and, on a mesh, the finite BvK equivalence of
{vibeqc-cite}`ye_berkelbach_periodic_lno_2024`; see the
[crystals guide](periodic_local_correlation.md#output-manifest-and-qvf).

## See also

* [Local correlation with DLPNO](../tutorial/dlpno_local_correlation.md)
  - worked walkthrough with the TCutPNO convergence experiment.
* [Local correlation on crystals](periodic_local_correlation.md) and the
  [periodic DLPNO-CCSD(T) tutorial](../tutorial/periodic_dlpno_ccsd_t.md),
  the `run_periodic_job` front door, its refusals, output fields and the
  finite-size theory.
* [`examples/molecular/input-h2o-dlpno-mp2.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dlpno-mp2.py)
  and [`…-dlpno-ccsd-t.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dlpno-ccsd-t.py)
  - ready-to-run scripts.
* [MP2 and double hybrids](mp2_and_double_hybrids.md), canonical MP2,
  SCS/SOS variants, double-hybrid functionals.
* [Density fitting](density_fitting.md), auxiliary-basis families and
  auto-resolution.
* [AICCM A-namespace correlation APIs](../aiccm2026dev_a.md):
  `run_ccm_mp2` and `run_ccm_ccsd` with `method="aiccm2026dev-a"` are
  union-and-weight Γ-CCM methods. The `ccm_dlpno_mp2` and
  `ccm_dlpno_ccsd` APIs instead use a neutral fitted-torus control; their
  A-namespace location does not make them Γ-CCM or χ-CCM results.
* [χ-CCM / aiccm2026dev-b](aiccm2026dev_b.md): periodic DLPNO-MP2 and
  DLPNO-CCSD(T) on the finite torus (χ-CCM, 3-D).
