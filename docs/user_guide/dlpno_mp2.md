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

The project-wide default freezes the chemical core and applies the supported
subset of Liakos *NormalPNO* on every restricted/unrestricted MP2 and CCSD
DLPNO solver. The chemical-core count follows ORCA 6.1 Table 2.69: ORCA's
frozen-electron count is converted to closed-shell spatial orbitals and summed
over the atoms. Use `frozen_core=False` on `run_job` for the uniform
all-electron escape corresponding to ORCA `NoFrozenCore`.

The implementation is count-only. It freezes that many lowest-energy
occupied orbitals, but does not reproduce ORCA's additional
atomic-orbital-character reorder for unusual canonical orderings. Molecular
ECP references are currently rejected because their variational electron
count differs from the molecule's physical count. `frozen_core=False` does
not restore ECP-removed electrons and therefore cannot bypass that guard.

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
| `DLPNOMP2Options` | `tcut_pairs`, `tcut_pno`, `tcut_mkn`; derived `tcut_pairs_weak` and `tcut_pno_weak` are stored but inactive for named presets |
| `DLPNOUMP2Options` | `tcut_pairs`, `tcut_pno` |
| `DLPNOCCSDPilotOptions` | `tcut_pno`, `tcut_mkn` |
| `DLPNOUCCSDPilotOptions` | `tcut_pno` |
| `LocalCCSDOptions` | `tcut_pairs`, `tcut_pno`, `tcut_mkn` |
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

Before v0.15.x these three constants carried in-house values that matched
no publication: `DLPNO_TIGHT` was `tcut_pairs` 5e-5 / `tcut_pno` 1e-8 /
`tcut_mkn` 1e-3, agreeing with published TightPNO on none of the three
knobs and screening pairs five times more loosely. They now carry the
published numbers (issue #416).

### Local density fitting (experimental, reduced-scaling)

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

## DLPNO-CCSD and DLPNO-CCSD(T)

**`method="dlpno-ccsd"`** runs the **reduced-scaling local solver**
(`dlpno.ccsd_local_solver`): amplitudes remain in pair PNO bases, while the
default residual is contracted in Riplinger and Neese's atom-based extended
PAO domain and projected back into the target pair. It is FCI-anchored; in the
full-domain limit it reproduces canonical closed-shell CCSD bit-for-bit
(≤ 1 µHa; on H₂ that equals FCI). The project default applies the supported
subset of NormalPNO with `residual_domain="extended"`. Historical recovery
percentages later on this page are labelled with the explicit all-electron
recipes that produced them.

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
(`tests/test_dlpno_ccsd_solver.py::TestSparseCoupling`).

**`method="dlpno-ccsd(t)"`** runs the same local solver plus the **DLPNO-(T1)**
(`dlpno.triples_local`) on the converged amplitudes: the (T) correction is
evaluated per occupied triple in a TNO domain after diagonalising and rotating
the occupied space. It removes the (T0) semicanonical approximation while
retaining finite PNO/TNO-domain truncation. The open-shell solver separately
exposes ``triples_mode="t1-iterative"``, a spin-orbital generalization of Guo
et al.'s 2018 closed-shell Eq. (2); their 2020 paper is the dedicated
open-shell method-family source. This opt-in correction currently builds
dense spin-orbital source tensors under the solver's ``max_nbf`` cap; it is
an experimental correctness/generalization route, not a reduced-scaling
triples phase. Set ``triples_mode="local"``
for the older DLPNO-(T0) (diagonal localised Fock, ~0.1 kcal/mol looser).
At full domains with canonical occupieds it reproduces canonical CCSD(T)
**exactly** (<= 1 nHa, the (T) parity ratchet).  The (T) vanishes
identically for two-electron systems.

### Larger systems: the SCF reference and memory

DLPNO is a reduced-scaling correlation method, but it runs on top of a
mean-field SCF reference, and that reference is what sets the memory
footprint on larger systems:

* **The DLPNO step itself is reduced-scaling** (per-pair PNO domains, no
  dense ``n_occ^2 n_virt^2`` or ``n_mo^4`` tensor). n-octane / cc-pVTZ
  (492 basis functions) DLPNO-MP2 peaks at a few GB, not hundreds.

* **Use a density-fitted SCF reference for larger systems.** Set
  ``density_fit=True`` (with an ``aux_basis``) on the reference options. A
  conventional in-core 4-index SCF would materialise the ``n_basis^4`` ERI
  tensor (~436 GB at 492 functions); the integral-direct SCF (the default
  above 200 functions) avoids it, and density fitting is faster still:

  ```python
  from vibeqc import RHFOptions, run_job
  o = RHFOptions(density_fit=True, aux_basis="cc-pvtz-jkfit")
  run_job(mol, basis="cc-pvtz", method="dlpno-mp2", rhf_options=o)  # ~13 GB
  ```

The **O(N^6) pilot** (``DLPNOCCSDPilotOptions``, opt-in) is a different
animal: it is the small-system *correctness oracle* the local solver is
validated against, and it forms dense full-virtual-space integrals (memory
~ ``n_mo^4``, set by the basis size and independent of ``tcut_pno``). It is
hard-capped at ``max_nbf=64`` for that reason. Do not use it for production
runs; the default ``run_job(method="dlpno-ccsd")`` already uses the
reduced-scaling local solver, which has no such cap.

### Accuracy

The following retained benchmark predates #140/#448. Against canonical
CCSD(T) on a 7-molecule set it used def2-SVP, **explicit all-electron**
correlation, `tcut_pno=1e-7`, `tcut_pairs=1e-4`, `tcut_mkn=0`, and
DLPNO-(T1), giving a mean absolute error of **0.37 kcal/mol**. It is valuable
recipe-specific evidence, but it is not a measurement of the current
chemical-core/NormalPNO default:

```python
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd
legacy = LocalCCSDOptions(
    n_frozen=0,
    tcut_pno=1e-7,
    tcut_pairs=1e-4,
    tcut_mkn=0.0,
    residual_domain="pair",
    compute_triples=True,
    triples_mode="t1",
)
r = run_local_dlpno_ccsd(mol, basis, hf, df, legacy)
# MAE 0.37 kcal/mol vs canonical CCSD(T); tcut_pno=1e-8 → 0.17 (ORCA: 0.16)
```

Tightening the same explicit legacy pair-domain recipe to ``tcut_pno=1e-8``
(where the compiled C++ residual is available) brings the MAE to
**0.17 kcal/mol**, matching ORCA 6.1's 0.16.
An earlier ``tcut_pno=1e-7`` + DLPNO-(T0) pairing gave MAE 0.30 only via an
accidental cancellation (CCSD over-recovery against the (T0) error); (T1)
removes the (T0) error honestly and 1e-8 removes the over-recovery, so they
target *different* errors and combine to the cancellation-free 0.17.
``triples_mode="local"`` selects DLPNO-(T0) for fast scans and
``triples_mode="exact"`` the O(N⁷) (T) oracle.

**Pair screening** (``tcut_pairs=1e-4``, the published NormalPNO
``TCutPairs`` coordinate, default-on)
treats a pair whose full-virtual MP2 energy is below the threshold at MP2
level instead of CCSD: accuracy-neutral on compact molecules (a no-op on
5-occupied cases, the weak ~10-14 % tail on larger ones) and a ~2x
linear-scaling speedup on extended systems; set ``tcut_pairs=0`` to
disable. The reproducible benchmark and ORCA comparison are
`examples/molecular/benchmark-dlpno-ccsd-t.py`.

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

`coupling_radius` is the occupied-side counterpart: it drops whole occupied
triple keys rather than shrinking each domain, and it can be far more
aggressive than it looks. On an H10 chain at 1.8 bohr spacing (16.2 bohr long,
strongly coupled occupieds) a `coupling_radius` of 5 bohr drops 16 of the 25
distinct triple keys and takes **90% of the triples energy** with them. That
is a property of the system rather than a defect: correlation in a delocalized
chain is genuinely long-ranged, so a locality cutoff is the wrong
approximation there. The count is now reported so the choice is visible:

```text
  triples / avg TNOs       = 9 / 5.0   (smallest domain: 5)   (16 screened by coupling_radius)
```

`LocalCCSDResult.n_triple_keys` and `.n_triple_keys_screened` carry the same
numbers. Screening behaves comparably in both triples modes: measured against
its own unscreened reference it costs 0.074 µHa under `(T1)` and 0.071 µHa
under `(T0)` on H2O + H2 at 8 Å, and 1459 µHa against 1142 µHa on the H10
chain.

The `dlpno_ccsd_done` structured-log event carries `n_tno_triples`,
`avg_tno_per_triple`, `min_tno_per_triple`, `n_degenerate_tno_triples` and
`n_triple_keys_screened` alongside the existing pair fields. The open-shell
local route reports its corresponding pair/PNO and, when requested,
triple/TNO diagnostics. The opt-in dense UCCSD(T) pilot builds no TNO domain,
so that pilot omits the domain line rather than printing zeros.

## Current limitations

* **DLPNO-MP2** runs on closed-shell RHF *and* open-shell UHF references:
  `run_job(method="dlpno-mp2")` auto-routes a multiplicity > 1 system to
  UHF + DLPNO-UMP2 (spin-channel resolved αα/ββ/αβ; `res.dlpno_ump2`). Pass a
  `DLPNOUMP2Options` as `dlpno_options` to control its thresholds
  (`localise`, `tcut_pno`, `tcut_pairs`, `n_frozen`). **DLPNO-CCSD/(T)**
  also run on open-shell UHF references. `run_job(method="dlpno-ccsd"|
  "dlpno-ccsd(t)")` auto-routes a multiplicity > 1 system to UHF plus the
  reduced-scaling `run_local_dlpno_uccsd` engine (`res.dlpno_ccsd`; use
  `LocalUCCSDOptions` for route-specific controls). It evaluates native UCCSD
  residuals in per-pair PNO/local-occupied domains and is exact against the
  dense pilot at full domains. Passing `DLPNOUCCSDPilotOptions` explicitly
  selects the O(N⁶), `max_nbf`-capped correctness pilot. The local engine's
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
  run. Broader radical benchmarks remain gated.

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
* `dlpno-ccsd(t)` runs the local solver + local DLPNO-(T1) on the
  converged amplitudes (exact == canonical CCSD(T) at full domains); the
  (T) uses a spatial closed-shell kernel (no spin-orbital redundancy),
  validated to machine precision against the spin-orbital reference. The
  default atom-based extended residual domain is contracted once per
  distinct (occupied coupling set, extended atom domain) group in the
  compiled `vibeqc::dlpno_pair_residual` kernel, and every pair in the group
  reads its own block; on a compact molecule that is one canonical-shaped
  residual per iteration rather than one per pair (S22-01 ammonia dimer at
  NormalPNO: 406 s to 5.5 s, energy unchanged). `LocalCCSDResult`
  reports the group count as `n_residual_domains`. The legacy
  `residual_domain="pair"` mode uses the same kernel in its pair-PNO basis;
  `use_cpp_kernel=False` selects the NumPy transcription for either mode.
* `tcut_pno` is a cut on **pair-density occupation numbers**, so it only means
  what a preset intends under the density it was calibrated against. That
  density is Riplinger and Neese's Eq. 23, `D = Tt T^T + Tt^T T` with
  `Tt = (4 T - 2 T^T) / (1 + delta_ij)`, selectable as `pno_norm="mp2"`
  (the pre-2013 LPNO variant is `pno_norm="iepa"`). The default is
  `pno_norm="legacy"`, vibe-qc's historical density of the bare amplitudes,
  which retains 5 % to 19 % fewer PNOs at the same nominal threshold and is
  therefore effectively looser than the preset name implies. The default is
  unchanged pending a decision on which convention ships, because moving it
  moves every DLPNO energy. The convention in force is printed in the `.out`
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

## See also

* [Local correlation with DLPNO](../tutorial/dlpno_local_correlation.md)
  - worked walkthrough with the TCutPNO convergence experiment.
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
